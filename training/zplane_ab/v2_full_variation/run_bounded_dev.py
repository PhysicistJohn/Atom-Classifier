"""Selection-only harness for the next bounded classifier/U-Net experiments.

This script deliberately does *not* produce a release score.  The corrected ground-state
test rows have already been reported once, so they are removed from the data object before
model construction and are never passed to training or evaluation code here.

The three questions this harness answers on one corrected preprocessing/data ground are:

1. What does a freshly initialized incumbent ``model.Embedding`` CNN achieve?
2. What is the corrected U-Net's classification ceiling with ``w_rec=0``?
3. Does ``U-Net denoised I/Q -> incumbent CNN`` help or hurt the incumbent?

Every available path gets the same development diagnostics:

* closed-set accuracy, balanced accuracy, clean/impaired accuracy, and confusion matrix;
* cross-fitted open-set threshold diagnostics on the selection population; and
* the canonical capture-length sweep.

Novelty and canonical probes are development fixtures, not a fresh release holdout.  Model
selection and every metric in this file use only train, enrollment, and selection rows.

Examples
--------
Fast end-to-end wiring check (writes only development artifacts)::

    .venv-training/bin/python -u \
      training/zplane_ab/v2_full_variation/run_bounded_dev.py \
      --smoke --device mps --tag bounded_dev_smoke

Full bounded comparison::

    .venv-training/bin/python -u \
      training/zplane_ab/v2_full_variation/run_bounded_dev.py \
      --device mps --tag bounded_dev_seed20260728 \
      --stages cnn,ceiling \
      --cnn-episodes 8000 --ceiling-episodes 4000 --eval-every 500
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn


HERE = Path(__file__).resolve().parent
ZPAB = HERE.parent
TRAINING = ZPAB.parent
REPO = TRAINING.parent
for path in (str(TRAINING), str(ZPAB), str(HERE)):
    if path not in os.sys.path:
        os.sys.path.insert(0, path)

import canonical_probe_net as canonical  # noqa: E402
import openset_eval  # noqa: E402
import pool_cache  # noqa: E402
import preprocess as pp  # noqa: E402
from ground_state_data import prepare_ground_state, selection_data  # noqa: E402
from length_aug import CropConfig  # noqa: E402
from model import EMBED_DIM, N_FEATURES, Embedding  # noqa: E402
from run_corrected_unet import ARCHITECTURE  # noqa: E402
from train import auroc, embed_all, nearest, prototypes_from  # noqa: E402
from train_transfer import train_transfer  # noqa: E402
from unet_transfer import TransferUNet  # noqa: E402


GROUND_SPLIT_SEED = 20260727
DEFAULT_SEED = 20260727
RAW_CAPTURE_LENGTH = 16384
STAGES = ("cnn", "ceiling", "joint", "cascade")


def seed_everything(seed: int) -> None:
    """Seed model initialization and every episode sampler before each independent arm."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(name)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("w") as f:
        json.dump(_jsonable(payload), f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


def selection_only_view(ground: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Strip a prepared ground-state object to its original selection population.

    The split seed is part of the ground-state definition. Changing it would silently move
    rows from the already-consumed test half into development, so callers never get a split
    seed option. The returned audit contains counts and an overlap assertion, but never
    returns or serializes consumed-test corpus identifiers.
    """
    selection_ids = np.asarray(ground["va_idx"])[np.asarray(ground["selection_rows"])]
    consumed_test_ids = np.asarray(ground["va_idx"])[np.asarray(ground["test_rows"])]
    overlap = np.intersect1d(selection_ids, consumed_test_ids)
    if len(overlap):
        raise AssertionError("selection rows overlap the consumed test population")

    dev = selection_data(ground)
    dev.pop("selection_rows", None)
    dev.pop("test_rows", None)
    dev["rng"] = np.random.default_rng(GROUND_SPLIT_SEED)
    dev["development_only"] = True
    dev["ground_state"] = {
        **dict(ground["ground_state"]),
        "selection_only": True,
        "validation_rows_exposed": int(len(dev["yva"])),
        "consumed_test_rows_exposed": 0,
    }
    audit = {
        "selection_rows": int(len(selection_ids)),
        "consumed_test_rows_exposed": 0,
        "selection_test_overlap": 0,
        "ground_state": dict(dev["ground_state"]),
        "cache_contract": copy.deepcopy(ground.get("cache_contract")),
        "preprocessing": pp.preprocess_metadata(),
        "selection_class_counts": {
            str(dev["classes"][c]): int(np.sum(np.asarray(dev["yva"]) == c))
            for c in range(dev["n_classes"])
        },
    }
    # Do not retain the full ground object after the selection-only view exists.
    del ground, consumed_test_ids
    return dev, audit


def development_data() -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the corrected ground and expose only its immutable development half."""
    ground = prepare_ground_state(
        pool_cache.load("resampled", seed=GROUND_SPLIT_SEED),
        seed=GROUND_SPLIT_SEED,
    )
    return selection_only_view(ground)


def training_view(data: dict[str, Any], seed: int) -> dict[str, Any]:
    """Give each arm the same independent episode stream without copying large arrays."""
    out = copy.copy(data)
    out["rng"] = np.random.default_rng(seed)
    return out


def stratified_folds(y: np.ndarray, impaired: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    """Deterministic class/impairment-stratified folds over the development population."""
    y = np.asarray(y, dtype=np.int64)
    impaired = np.asarray(impaired, dtype=bool)
    if n_folds < 2:
        raise ValueError("n_folds must be at least two")
    rng = np.random.default_rng(seed)
    buckets: list[list[int]] = [[] for _ in range(n_folds)]
    for cls in np.unique(y):
        for imp in (False, True):
            rows = rng.permutation(np.where((y == cls) & (impaired == imp))[0])
            for offset, row in enumerate(rows):
                buckets[offset % n_folds].append(int(row))
    folds = [np.asarray(sorted(rows), dtype=np.int64) for rows in buckets]
    joined = np.concatenate(folds)
    if len(joined) != len(y) or len(np.unique(joined)) != len(y):
        raise AssertionError("development folds must be disjoint and exhaustive")
    if not np.array_equal(np.sort(joined), np.arange(len(y))):
        raise AssertionError("development folds do not cover every selection row")
    return folds


class EmbeddingView(nn.Module):
    """Use an encoder-only fast path when a model exposes one."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        fast = getattr(self.model, "forward_embedding", None)
        return fast(x, feat) if callable(fast) else self.model(x, feat)


class DenoiseThenEmbed(nn.Module):
    """Evaluation-only ``TransferUNet reconstruction -> incumbent CNN`` cascade.

    The U-Net target is a unit-RMS canonical waveform, so reconstruction is normalized once
    before entering the CNN.  The 12 scalar features are recomputed from the reconstructed
    waveform and transformed with the same train-only statistics as every other path.
    """

    def __init__(
        self,
        denoiser: TransferUNet,
        incumbent: Embedding,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.incumbent = incumbent
        self._feature_mean = np.asarray(feature_mean, dtype=np.float32).copy()
        self._feature_std = np.asarray(feature_std, dtype=np.float32).copy()
        if (
            self._feature_mean.shape != (N_FEATURES,)
            or self._feature_std.shape != (N_FEATURES,)
            or np.any(self._feature_std <= 0)
        ):
            raise ValueError("cascade feature statistics are malformed")

    @torch.no_grad()
    def forward(self, x: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        self.denoiser.eval()
        self.incumbent.eval()
        self.denoiser(x, feat)
        recon = self.denoiser.last_recon
        if recon is None:
            raise RuntimeError("denoiser did not expose last_recon")
        rms = torch.sqrt((recon.real.square() + recon.imag.square()).mean(-1, keepdim=True) + 1e-12)
        recon = recon / rms
        channels = torch.stack([recon.real, recon.imag], dim=1).to(dtype=torch.float32)

        recon_np = recon.detach().cpu().numpy()
        raw_features = np.stack([pp.iq_features(row) for row in recon_np]).astype(np.float32)
        z_features = (raw_features - self._feature_mean) / self._feature_std
        feat_out = torch.from_numpy(z_features.astype(np.float32)).to(channels.device)
        return self.incumbent(channels, feat_out)


def _load_state(path: str | None) -> dict[str, torch.Tensor] | None:
    if path is None:
        return None
    payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"{path} does not contain a model state_dict")
    return payload


def _nonfinite_paths(value: Any, prefix: str = "") -> list[str]:
    """Return paths to non-finite numeric values in a nested training record."""
    bad: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            bad.extend(_nonfinite_paths(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            bad.extend(_nonfinite_paths(item, f"{prefix}[{index}]"))
    elif isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
        bad.append(prefix)
    return bad


def _assert_finite_model(net: nn.Module, name: str) -> None:
    bad = [
        key
        for key, value in net.state_dict().items()
        if torch.is_floating_point(value) and not torch.isfinite(value).all()
    ]
    if bad:
        raise FloatingPointError(f"{name} contains non-finite tensors: {bad[:8]}")


def _train_or_load(
    name: str,
    net: nn.Module,
    checkpoint: str | None,
    data: dict[str, Any],
    dev: torch.device,
    seed: int,
    episodes: int,
    eval_every: int,
    w_rec: float,
    lr: float,
    weight_decay: float,
    warmup_frac: float,
    label_smoothing: float,
) -> tuple[nn.Module, dict[str, Any]]:
    state = _load_state(checkpoint)
    if state is not None:
        net.load_state_dict(state)
        _assert_finite_model(net, name)
        return net.eval(), {
            "source": "checkpoint",
            "checkpoint": str(Path(checkpoint).expanduser().resolve()),
            "episodes": 0,
        }
    if episodes <= 0:
        raise ValueError(f"{name}: episodes must be positive when no checkpoint is supplied")

    net, history, wall = train_transfer(
        net,
        training_view(data, seed),
        dev,
        episodes=episodes,
        eval_every=max(1, min(eval_every, episodes)),
        log_tag=name,
        lr=lr,
        weight_decay=weight_decay,
        warmup_frac=warmup_frac,
        label_smoothing=label_smoothing,
        w_rec=w_rec,
        crop_cfg=CropConfig(mode="off", guard_frac=0.35),
        align_pair=True,
        mask_clean_recon=True,
        excess_loss=False,
        log_every=max(1, min(50, episodes)),
        seed=seed,
        condition="resampled",
    )
    _assert_finite_model(net, name)
    bad_history = _nonfinite_paths(history)
    if bad_history:
        raise FloatingPointError(
            f"{name} produced non-finite training history values: {bad_history[:8]}"
        )
    return net.eval(), {
        "source": "fresh_training",
        "episodes": int(episodes),
        "wall_clock_s": float(wall),
        "history": history,
    }


def _closed_report(
    emb: np.ndarray,
    labels: np.ndarray,
    impaired: np.ndarray,
    protos: np.ndarray,
    classes: list[str],
    folds: Iterable[np.ndarray],
) -> dict[str, Any]:
    pred, distances = nearest(emb, protos)
    labels = np.asarray(labels, dtype=np.int64)
    impaired = np.asarray(impaired, dtype=bool)
    n_classes = len(classes)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for truth, guess in zip(labels, pred):
        confusion[int(truth), int(guess)] += 1
    recalls = np.diag(confusion) / np.maximum(confusion.sum(1), 1)
    fold_rows = []
    for i, rows in enumerate(folds):
        fold_rows.append(
            {
                "fold": i,
                "n": int(len(rows)),
                "accuracy": float(np.mean(pred[rows] == labels[rows])),
            }
        )
    return {
        "n_selection": int(len(labels)),
        "accuracy": float(np.mean(pred == labels)),
        "balanced_accuracy": float(np.mean(recalls)),
        "clean_accuracy": float(np.mean(pred[~impaired] == labels[~impaired])),
        "impaired_accuracy": float(np.mean(pred[impaired] == labels[impaired])),
        "per_class": {classes[c]: float(recalls[c]) for c in range(n_classes)},
        "confusion": confusion.tolist(),
        "cv_folds": fold_rows,
        "nearest_distance_median": float(np.median(distances.min(1))),
    }


def _open_set_cv_report(
    net: nn.Module,
    dev: torch.device,
    data: dict[str, Any],
    protos: np.ndarray,
    known_emb: np.ndarray,
    folds: list[np.ndarray],
    novelty: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Open-set AUROC plus cross-fitted 95th-percentile operating thresholds."""
    _, known_dist = nearest(known_emb, protos)
    known_score = known_dist.min(1)

    novel_scores: dict[str, np.ndarray] = {}
    for name, (xnov, fnov_raw) in novelty.items():
        fnov = (
            (np.asarray(fnov_raw, dtype=np.float32) - np.asarray(data["fmean"]))
            / np.asarray(data["fstd"])
        ).astype(np.float32)
        emb = embed_all(net, xnov, fnov, dev)
        _, dist = nearest(emb, protos)
        novel_scores[name] = dist.min(1)

    all_novel = np.concatenate(list(novel_scores.values()))
    fold_rows = []
    crossfit_known_unknown = np.zeros(len(known_score), dtype=bool)
    for i, eval_rows in enumerate(folds):
        calibration_rows = np.setdiff1d(np.arange(len(known_score)), eval_rows, assume_unique=True)
        threshold = float(np.percentile(known_score[calibration_rows], 95))
        crossfit_known_unknown[eval_rows] = known_score[eval_rows] > threshold
        row = {
            "fold": i,
            "n_known_eval": int(len(eval_rows)),
            "n_known_calibration": int(len(calibration_rows)),
            "threshold_p95_calibration": threshold,
            "known_false_unknown_rate": float(
                np.mean(known_score[eval_rows] > threshold)
            ),
            "auroc_overall": float(auroc(all_novel, known_score[eval_rows])),
        }
        for name, score in novel_scores.items():
            row[f"auroc_{name}"] = float(auroc(score, known_score[eval_rows]))
            row[f"flagged_unknown_{name}"] = float(np.mean(score > threshold))
        fold_rows.append(row)

    return {
        "development_fixture": True,
        "novelty_types": list(novel_scores),
        "n_known": int(len(known_score)),
        "n_novel": int(len(all_novel)),
        "auroc_overall": float(auroc(all_novel, known_score)),
        **{
            f"auroc_{name}": float(auroc(score, known_score))
            for name, score in novel_scores.items()
        },
        "known_false_unknown_rate_crossfit": float(np.mean(crossfit_known_unknown)),
        **{
            f"flagged_unknown_{name}_mean": float(
                np.mean([row[f"flagged_unknown_{name}"] for row in fold_rows])
            )
            for name in novel_scores
        },
        "cv_folds": fold_rows,
    }


def evaluate_development_path(
    name: str,
    net: nn.Module,
    data: dict[str, Any],
    dev: torch.device,
    folds: list[np.ndarray],
    novelty: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, Any], np.ndarray]:
    """Evaluate one embedding surface without touching a consumed test row."""
    view = EmbeddingView(net) if not isinstance(net, DenoiseThenEmbed) else net
    view = view.to(dev).eval()
    enroll_emb = embed_all(view, data["xen"], data["fen"], dev)
    protos = prototypes_from(enroll_emb, data["yen"], data["n_classes"])
    selection_emb = embed_all(view, data["xva"], data["fva"], dev)
    closed = _closed_report(
        selection_emb,
        data["yva"],
        data["imp_va"],
        protos,
        list(data["classes"]),
        folds,
    )
    open_set = _open_set_cv_report(
        view, dev, data, protos, selection_emb, folds, novelty
    )
    canonical_report = canonical.sweep(
        view,
        protos,
        list(data["classes"]),
        dev,
        fmean=data["fmean"],
        fstd=data["fstd"],
        condition="resampled",
        in_len=int(data["input_length"]),
        matched_length=RAW_CAPTURE_LENGTH,
        verbose=True,
    )
    report = {
        "name": name,
        "development_only": True,
        "consumed_test_rows_used": 0,
        "closed_selection": closed,
        "open_set_selection_cv": open_set,
        "canonical_length_development": canonical_report,
    }
    return report, protos


def _parse_stages(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(values).difference(STAGES))
    if unknown:
        raise ValueError(f"unknown stages {unknown}; choices are {STAGES}")
    if not values:
        raise ValueError("at least one stage is required")
    return values


def _architecture_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Materialize every U-Net architecture choice into the run record."""
    architecture = {
        **ARCHITECTURE,
        "base": int(args.unet_base),
        "depth": int(args.unet_depth),
        "taps": int(args.unet_taps),
        "hidden": int(args.unet_hidden),
        "magnorm": bool(args.unet_magnorm),
        "modrelu_init": float(args.unet_modrelu_init),
        "skip_dropout": float(args.unet_skip_dropout),
        "feat_dropout": float(args.unet_feat_dropout),
        "antialias": bool(args.unet_antialias),
        "head_pool": str(args.unet_head_pool),
    }
    if architecture["base"] <= 0 or architecture["depth"] <= 0:
        raise ValueError("U-Net base and depth must be positive")
    if architecture["taps"] <= 0 or architecture["taps"] % 2 == 0:
        raise ValueError("U-Net taps must be a positive odd integer")
    if architecture["hidden"] <= 0:
        raise ValueError("U-Net hidden width must be positive")
    for key in ("skip_dropout", "feat_dropout"):
        if not 0.0 <= architecture[key] < 1.0:
            raise ValueError(f"U-Net {key} must lie in [0, 1)")
    if not np.isfinite(architecture["modrelu_init"]):
        raise ValueError("U-Net modrelu_init must be finite")
    return architecture


def run(args: argparse.Namespace) -> dict[str, Any]:
    dev = resolve_device(args.device)
    stages = _parse_stages(args.stages)
    architecture = _architecture_from_args(args)
    episodes = {
        "cnn": 2 if args.smoke else args.cnn_episodes,
        "ceiling": 2 if args.smoke else args.ceiling_episodes,
        "joint": 2 if args.smoke else args.joint_episodes,
    }
    novelty_n = min(args.novelty_n, 8) if args.smoke else args.novelty_n

    print("[bounded-dev] loading selection-only corrected ground", flush=True)
    data, data_audit = development_data()
    folds = stratified_folds(
        data["yva"], data["imp_va"], n_folds=args.cv_folds, seed=args.seed + 101
    )
    novelty = openset_eval.build_novelty(
        "resampled",
        n_each=novelty_n,
        seed=args.seed + 211,
        in_len=int(data["input_length"]),
        raw_len=RAW_CAPTURE_LENGTH,
    )

    tag = args.tag or f"bounded_dev_seed{args.seed}"
    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else HERE / "artifacts" / "bounded_dev" / tag
    )
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"{out_dir} is not empty; use a new --tag/--output-dir to preserve prior runs"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        out_dir / "DEVELOPMENT_ONLY.json",
        {
            "development_only": True,
            "release_evidence": False,
            "consumed_test_rows_used": 0,
            "reason": "the prior corrected-run test population has already been consumed",
            "data_audit": data_audit,
        },
    )

    models: dict[str, nn.Module] = {}
    training: dict[str, Any] = {}
    reports: dict[str, Any] = {}
    prototypes: dict[str, np.ndarray] = {}

    if "cnn" in stages or "cascade" in stages or args.cnn_checkpoint:
        seed_everything(args.seed)
        cnn = Embedding(EMBED_DIM, N_FEATURES)
        cnn, training["cnn"] = _train_or_load(
            "cnn",
            cnn,
            args.cnn_checkpoint,
            data,
            dev,
            args.seed,
            episodes["cnn"],
            args.eval_every,
            0.0,
            args.cnn_lr,
            args.cnn_weight_decay,
            args.cnn_warmup_frac,
            args.cnn_label_smoothing,
        )
        models["cnn"] = cnn
        torch.save(cnn.state_dict(), out_dir / "cnn_state_dict.pt")
        _write_json(
            out_dir / "cnn_training.json",
            {
                "model": "incumbent_cnn_fresh",
                "architecture": {
                    "embed_dim": EMBED_DIM,
                    "n_features": N_FEATURES,
                    "parameter_count": int(sum(p.numel() for p in cnn.parameters())),
                },
                "optimizer": {
                    "lr": args.cnn_lr,
                    "weight_decay": args.cnn_weight_decay,
                    "warmup_frac": args.cnn_warmup_frac,
                    "label_smoothing": args.cnn_label_smoothing,
                },
                **training["cnn"],
            },
        )

    if "ceiling" in stages or args.ceiling_checkpoint:
        seed_everything(args.seed)
        ceiling = TransferUNet(**architecture)
        ceiling, training["ceiling"] = _train_or_load(
            "unet_ceiling",
            ceiling,
            args.ceiling_checkpoint,
            data,
            dev,
            args.seed,
            episodes["ceiling"],
            args.eval_every,
            0.0,
            args.unet_lr,
            args.unet_weight_decay,
            args.unet_warmup_frac,
            args.unet_label_smoothing,
        )
        models["ceiling"] = ceiling
        torch.save(ceiling.state_dict(), out_dir / "unet_ceiling_state_dict.pt")
        _write_json(
            out_dir / "unet_ceiling_training.json",
            {
                "model": "corrected_unet_classification_ceiling",
                "architecture": architecture,
                "parameter_count": int(sum(p.numel() for p in ceiling.parameters())),
                "w_rec": 0.0,
                "optimizer": {
                    "lr": args.unet_lr,
                    "weight_decay": args.unet_weight_decay,
                    "warmup_frac": args.unet_warmup_frac,
                    "label_smoothing": args.unet_label_smoothing,
                },
                **training["ceiling"],
            },
        )

    if "joint" in stages or "cascade" in stages or args.joint_checkpoint:
        seed_everything(args.seed)
        joint = TransferUNet(**architecture)
        joint, training["joint"] = _train_or_load(
            "unet_joint",
            joint,
            args.joint_checkpoint,
            data,
            dev,
            args.seed,
            episodes["joint"],
            args.eval_every,
            args.joint_w_rec,
            args.unet_lr,
            args.unet_weight_decay,
            args.unet_warmup_frac,
            args.unet_label_smoothing,
        )
        models["joint"] = joint
        torch.save(joint.state_dict(), out_dir / "unet_joint_state_dict.pt")
        _write_json(
            out_dir / "unet_joint_training.json",
            {
                "model": "corrected_unet_joint",
                "architecture": architecture,
                "parameter_count": int(sum(p.numel() for p in joint.parameters())),
                "w_rec": args.joint_w_rec,
                "optimizer": {
                    "lr": args.unet_lr,
                    "weight_decay": args.unet_weight_decay,
                    "warmup_frac": args.unet_warmup_frac,
                    "label_smoothing": args.unet_label_smoothing,
                },
                **training["joint"],
            },
        )

    for name in ("cnn", "ceiling", "joint"):
        if name not in models or name not in stages:
            continue
        print(f"[bounded-dev] evaluating {name} on selection/CV only", flush=True)
        reports[name], prototypes[name] = evaluate_development_path(
            name, models[name], data, dev, folds, novelty
        )
        np.save(out_dir / f"{name}_prototypes.npy", prototypes[name])
        _write_json(out_dir / f"{name}_dev_metrics.json", reports[name])

    if "cascade" in stages:
        if "cnn" not in models or "joint" not in models:
            raise RuntimeError("cascade requires a CNN and joint U-Net checkpoint or training arm")
        cascade = DenoiseThenEmbed(
            models["joint"], models["cnn"], data["fmean"], data["fstd"]
        )
        print("[bounded-dev] evaluating denoised-IQ -> incumbent cascade", flush=True)
        reports["cascade"], prototypes["cascade"] = evaluate_development_path(
            "cascade", cascade, data, dev, folds, novelty
        )
        np.save(out_dir / "cascade_prototypes.npy", prototypes["cascade"])
        _write_json(out_dir / "cascade_dev_metrics.json", reports["cascade"])

    result = {
        "status": "complete",
        "development_only": True,
        "release_evidence": False,
        "consumed_test_rows_used": 0,
        "tag": tag,
        "seed": args.seed,
        "device": str(dev),
        "stages": stages,
        "data_audit": data_audit,
        "training": training,
        "metrics": reports,
        "configuration": {
            "architecture": architecture,
            "episodes": episodes,
            "joint_w_rec": args.joint_w_rec,
            "cv_folds": args.cv_folds,
            "novelty_n_each": novelty_n,
            "raw_capture_length": RAW_CAPTURE_LENGTH,
            "ground_split_seed": GROUND_SPLIT_SEED,
            "crop_mode": "off",
        },
    }
    _write_json(out_dir / "dev_metrics.json", result)
    print(f"[bounded-dev] complete -> {out_dir}", flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--tag")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--stages",
        default="cnn,ceiling,joint,cascade",
        help="comma-separated subset of cnn,ceiling,joint,cascade",
    )
    parser.add_argument("--smoke", action="store_true", help="two episodes per trained arm")
    parser.add_argument(
        "--cnn-episodes",
        type=int,
        default=8000,
        help="incumbent budget; use 4000 for the bounded screen or 8000 for the full control",
    )
    parser.add_argument(
        "--ceiling-episodes",
        type=int,
        default=4000,
        help="decoder-free corrected U-Net classification-ceiling budget",
    )
    parser.add_argument("--joint-episodes", type=int, default=4000)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=500,
        help="500 is the bounded default; 1000 is suitable for an 8000-episode CNN run",
    )
    parser.add_argument("--joint-w-rec", type=float, default=1.0)
    parser.add_argument("--cv-folds", type=int, default=2)
    parser.add_argument("--novelty-n", type=int, default=300)
    parser.add_argument("--cnn-checkpoint")
    parser.add_argument("--ceiling-checkpoint")
    parser.add_argument("--joint-checkpoint")

    # Architecture screens remain development-only, but each switch is explicit and
    # serialized so a winning checkpoint can be reconstructed exactly.
    parser.add_argument("--unet-base", type=int, default=ARCHITECTURE["base"])
    parser.add_argument("--unet-depth", type=int, default=ARCHITECTURE["depth"])
    parser.add_argument("--unet-taps", type=int, default=ARCHITECTURE["taps"])
    parser.add_argument("--unet-hidden", type=int, default=ARCHITECTURE["hidden"])
    parser.add_argument(
        "--unet-head-pool",
        choices=("meanstd", "acf", "acf_complex", "chunk4"),
        default=ARCHITECTURE["head_pool"],
    )
    parser.add_argument(
        "--unet-feat-dropout",
        type=float,
        default=ARCHITECTURE["feat_dropout"],
    )
    parser.add_argument(
        "--unet-skip-dropout",
        type=float,
        default=ARCHITECTURE["skip_dropout"],
    )
    parser.add_argument(
        "--unet-modrelu-init",
        type=float,
        default=ARCHITECTURE["modrelu_init"],
    )
    parser.add_argument(
        "--unet-magnorm",
        action=argparse.BooleanOptionalAction,
        default=ARCHITECTURE["magnorm"],
    )
    parser.add_argument(
        "--unet-antialias",
        action=argparse.BooleanOptionalAction,
        default=ARCHITECTURE["antialias"],
    )

    # The CNN uses its incumbent optimizer defaults; both U-Net arms share the corrected
    # runner's settings.  Every value is explicit in dev_metrics.json.
    parser.add_argument("--cnn-lr", type=float, default=1e-3)
    parser.add_argument("--cnn-weight-decay", type=float, default=2e-4)
    parser.add_argument("--cnn-warmup-frac", type=float, default=0.0)
    parser.add_argument("--cnn-label-smoothing", type=float, default=0.0)
    parser.add_argument("--unet-lr", type=float, default=7e-4)
    parser.add_argument("--unet-weight-decay", type=float, default=5e-4)
    parser.add_argument("--unet-warmup-frac", type=float, default=0.05)
    parser.add_argument("--unet-label-smoothing", type=float, default=0.05)
    return parser


if __name__ == "__main__":
    started = time.perf_counter()
    output = run(build_parser().parse_args())
    print(
        json.dumps(
            {
                "status": output["status"],
                "development_only": True,
                "consumed_test_rows_used": 0,
                "elapsed_s": time.perf_counter() - started,
            },
            indent=2,
        )
    )
