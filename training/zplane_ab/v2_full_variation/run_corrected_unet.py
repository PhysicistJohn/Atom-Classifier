"""One auditable ground-state run of the corrected transfer U-Net.

This is intentionally a focused runner rather than another campaign grid.  It fixes the
experimental coordinate system before training and gives the old validation pool two
different jobs:

* ``selection_rows`` alone choose the checkpoint during training;
* ``test_rows`` remain untouched until the final report.

The run is the edge-safe, resampled (1024-sample) condition.  Its reconstruction target is
the clean pair expressed in the input front-end's carrier frame (``derotate_onto`` after
paired preprocessing).  That is a tractable denoising target; it is not described as a
learned global-CFO equalizer.

Artifacts are sufficient for inference and a from-scratch reproduction:

* trained and random-initialization state dicts;
* prototypes, corrected feature statistics, and open-set threshold;
* the exact corpus indices in every split;
* architecture, optimizer/training arguments, RNG seed, environment, source hashes, corpus
  manifest hash, metrics, scalar-transfer head, and training history.

The current ``train_transfer`` API restores the best model but does not expose optimizer
state.  Consequently this runner guarantees a reproducible complete run, not mid-episode
resume.

Examples
--------
Validate all invariants without training or writing artifacts::

    .venv-training/bin/python -u \
      training/zplane_ab/v2_full_variation/run_corrected_unet.py --dry-run

Run the corrected ground state (MPS when available)::

    .venv-training/bin/python -u \
      training/zplane_ab/v2_full_variation/run_corrected_unet.py --episodes 700
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ZPAB = HERE.parent
TRAINING = ZPAB.parent
REPO = TRAINING.parent
for path in (str(TRAINING), str(ZPAB), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import canonical_probe_net as cpn  # noqa: E402
import denoise_eval as de  # noqa: E402
import pool_cache  # noqa: E402
import preprocess as pp  # noqa: E402
from evaluate_ab_v2 import evaluate_v2  # noqa: E402
from ground_state_data import (  # noqa: E402
    prepare_ground_state,
    selection_data,
    test_data,
)
from length_aug import CropConfig  # noqa: E402
from openset_eval import evaluate_openset  # noqa: E402
from scalar_transfer import scalar_transfer_report  # noqa: E402
from train import embed_all, nearest  # noqa: E402
from train_transfer import train_transfer  # noqa: E402
from unet_transfer import TransferUNet, encoder_health  # noqa: E402


DEFAULT_SEED = 20260727
DEFAULT_EPISODES = 700
RAW_CAPTURE_LENGTH = 16384
TARGET_POLICY = (
    "input-carrier-aligned denoising: the paired clean waveform is preprocessed with the "
    "impaired input's center, bandwidth, and resample fraction, then fine-derotated into "
    "the input carrier frame; reconstruction loss is masked to impaired rows"
)

# Every constructor argument is explicit in the inference bundle.  TransferUNet.config()
# intentionally records only historical switches, which is not enough to reconstruct a
# non-default model safely.
ARCHITECTURE = {
    "base": 8,
    "depth": 4,
    "taps": 7,
    "embed_dim": 32,
    "n_features": 12,
    "n_profiles": 37,
    "hidden": 128,
    "magnorm": True,
    "modrelu_init": -2.0,
    "skip_dropout": 0.5,
    "feat_dropout": 0.5,
    "antialias": False,
    "head_pool": "meanstd",
    "residual_recon": True,
}

SOURCE_FILES = (
    Path(__file__),
    HERE / "ground_state_data.py",
    HERE / "pool_cache.py",
    HERE / "length_aug.py",
    HERE / "train_transfer.py",
    HERE / "unet_transfer.py",
    HERE / "evaluate_ab_v2.py",
    HERE / "openset_eval.py",
    HERE / "canonical_probe_net.py",
    HERE / "denoise_eval.py",
    HERE / "scalar_transfer.py",
    TRAINING / "preprocess.py",
    TRAINING / "train.py",
)


def seed_everything(seed: int) -> None:
    """Seed every RNG used by this process before pool or model construction."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)
    # Warn rather than abort when MPS lacks a deterministic implementation.  The source
    # manifest records this policy; PyTorch cannot promise bit identity across releases.
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested, but torch.backends.mps is unavailable")
    return torch.device(requested)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git(args: list[str]) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        return proc.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _jsonable(value: Any) -> Any:
    """Strict-JSON conversion: NaN/Inf become null rather than non-standard literals."""
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
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    """Atomically publish JSON so an interrupted evaluation cannot leave a partial file."""
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("w") as f:
        json.dump(_jsonable(payload), f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


def _cpu_state_dict(net: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def _validate_ground_state(data: dict[str, Any]) -> dict[str, Any]:
    """Fail before training if the four experimental populations overlap or are incomplete."""
    selection = selection_data(data)
    test = test_data(data)
    split_ids = {
        "train": np.asarray(data["tr_idx"], dtype=np.int64),
        "enroll": np.asarray(data["en_idx"], dtype=np.int64),
        "selection": np.asarray(selection["va_idx"], dtype=np.int64),
        "test": np.asarray(test["va_idx"], dtype=np.int64),
    }
    for left, left_ids in split_ids.items():
        if len(np.unique(left_ids)) != len(left_ids):
            raise AssertionError(f"{left} contains duplicate corpus rows")
        for right, right_ids in split_ids.items():
            if left >= right:
                continue
            overlap = np.intersect1d(left_ids, right_ids)
            if len(overlap):
                raise AssertionError(
                    f"{left}/{right} overlap at {len(overlap)} corpus rows; "
                    f"first={int(overlap[0])}"
                )

    support = {}
    for name, part, suffix in (
        ("train", data, "tr"),
        ("enroll", data, "en"),
        ("selection", selection, "va"),
        ("test", test, "va"),
    ):
        y = np.asarray(part[f"y{suffix}"])
        impaired = np.asarray(part[f"imp_{suffix}"], dtype=bool)
        support[name] = {
            str(data["classes"][c]): {
                "clean": int(np.sum((y == c) & ~impaired)),
                "impaired": int(np.sum((y == c) & impaired)),
            }
            for c in range(data["n_classes"])
        }
        missing = [
            (cls, state)
            for cls, counts in support[name].items()
            for state, n in counts.items()
            if n == 0
        ]
        if missing:
            raise AssertionError(f"{name} lacks class/impairment support: {missing}")

    if not np.all(np.isfinite(data["fmean"])) or not np.all(np.isfinite(data["fstd"])):
        raise AssertionError("corrected feature statistics contain non-finite values")
    if np.any(np.asarray(data["fstd"]) <= 0):
        raise AssertionError("corrected feature standard deviations must be positive")
    return {
        "split_counts": {name: int(len(ids)) for name, ids in split_ids.items()},
        "class_impairment_support": support,
    }


def _residual_identity_check(net: TransferUNet, data: dict[str, Any]) -> float:
    """The corrected reconstruction head must start exactly as passthrough."""
    net.eval()
    xb = torch.from_numpy(np.asarray(data["xtr"][:3], dtype=np.float32))
    fb = torch.from_numpy(np.asarray(data["ftr"][:3], dtype=np.float32))
    with torch.no_grad():
        net(xb, fb)
        expected = torch.complex(xb[:, 0], xb[:, 1])
        error = float((net.last_recon - expected).abs().max())
    if error != 0.0:
        raise AssertionError(f"zero-initialized residual head is not identity (max error {error})")
    return error


def _confusion(net, dev, data, protos) -> tuple[list[list[int]], float]:
    emb = embed_all(net, data["xva"], data["fva"], dev)
    pred, _ = nearest(emb, protos)
    cm = np.zeros((data["n_classes"], data["n_classes"]), dtype=np.int64)
    for truth, guess in zip(data["yva"], pred):
        cm[int(truth), int(guess)] += 1
    recalls = np.diag(cm) / np.maximum(cm.sum(1), 1)
    return cm.tolist(), float(recalls.mean())


def _corrected_denoise_features(
    impaired: np.ndarray,
    corpus_idx: np.ndarray,
    data: dict[str, Any],
    test: dict[str, Any],
) -> tuple[np.ndarray, float]:
    """Recompute features in corrected coordinates and prove test-pool parity.

    ``load_eval_rows`` reads feature arrays built before the edge-safe population was
    selected.  Those cached z-scores are therefore wrong for this run even though the
    waveform is right.  Recomputing raw features from the exact waveform and applying the
    corrected statistics prevents the campaign-4 train/eval mismatch from recurring here.
    """
    raw = np.stack([pp.iq_features(row) for row in impaired]).astype(np.float32)
    corrected = (
        (raw - np.asarray(data["fmean"], dtype=np.float32))
        / np.asarray(data["fstd"], dtype=np.float32)
    ).astype(np.float32)
    lookup = {
        int(idx): np.asarray(feat, dtype=np.float32)
        for idx, feat in zip(test["va_idx"], test["fva"])
    }
    expected = np.stack([lookup[int(idx)] for idx in corpus_idx])
    max_diff = float(np.max(np.abs(corrected - expected)))
    if max_diff > 2e-5:
        raise AssertionError(
            "denoise feature coordinates do not match the corrected held-out pool "
            f"(max |difference|={max_diff:.3e})"
        )
    return corrected, max_diff


def _compact_denoise(res: dict) -> dict:
    """Keep auditable aggregates and verdicts without embedding large per-row arrays."""
    impaired = np.asarray(res["impaired_mask"], dtype=bool)
    out = {
        "n": int(res["n"]),
        "n_impaired": int(res["n_impaired"]),
        "aggregates": res["agg"],
        "by_snr": res.get("by_snr"),
        "low_snr": res.get("low_snr"),
        "high_snr_passthrough": res.get("high_snr_passthrough"),
        "verdict": de.verdict(res),
    }
    if "coh_recon_vs_input" in res:
        out["coh_recon_vs_input_impaired_mean"] = float(
            np.asarray(res["coh_recon_vs_input"])[impaired].mean()
        )
    if "relative_change" in res:
        out["relative_change_impaired_mean"] = float(
            np.asarray(res["relative_change"])[impaired].mean()
        )
    return out


def _denoise_report(
    state_dict: dict[str, torch.Tensor],
    data: dict[str, Any],
    test: dict[str, Any],
    n_rows: int,
    seed: int,
    with_oracle_fir: bool,
) -> dict:
    cached_impaired, clean_unaligned, meta = de.load_eval_rows(
        "val",
        n=n_rows,
        condition="resampled",
        seed=seed,
        align_target=False,
        allowed_corpus_idx=test["va_idx"],
    )
    # Ground-state evaluation uses the exact rebuilt test waveform.  The historical
    # *validation* cache was already built without scale jitter and should match it exactly
    # (only the historical training cache had jitter), so this is also a guard that the
    # clean-target builder and held-out input still share preprocessing geometry.
    waveform_lookup = {
        int(idx): np.asarray(ch[0], dtype=np.float64) + 1j * np.asarray(ch[1], dtype=np.float64)
        for idx, ch in zip(test["va_idx"], test["xva"])
    }
    impaired = np.stack([waveform_lookup[int(idx)] for idx in meta["corpus_idx"]])
    loader_ground_waveform_max_abs = float(np.max(np.abs(impaired - cached_impaired)))
    if loader_ground_waveform_max_abs > 2e-6:
        raise AssertionError(
            "denoise loader input differs from the rebuilt held-out waveform "
            f"(max |difference|={loader_ground_waveform_max_abs:.3e}); the paired clean "
            "target can no longer be assumed to share the test input's geometry"
        )
    features, parity_error = _corrected_denoise_features(
        impaired, meta["corpus_idx"], data, test
    )
    eval_net = TransferUNet(**ARCHITECTURE)
    eval_net.load_state_dict(state_dict)
    recon = de.run_model(
        None,
        impaired,
        features,
        net=eval_net,
        batch=32,
    )
    clean_aligned = np.stack(
        [
            de.derotate_onto(clean_unaligned[i], impaired[i]).astype(np.complex128)
            for i in range(len(impaired))
        ]
    )

    aligned = de.score_denoising(
        clean_aligned,
        impaired,
        recon,
        meta["snr_db"],
        meta["frac_bw"],
        meta["impaired"],
        with_oracle_fir=with_oracle_fir,
    )
    # The unaligned score keeps the target narrowing visible.  The expensive oracle FIR is
    # only needed on the training-aligned target.
    unaligned = de.score_denoising(
        clean_unaligned,
        impaired,
        recon,
        meta["snr_db"],
        meta["frac_bw"],
        meta["impaired"],
        with_oracle_fir=False,
    )
    return {
        "target_policy": TARGET_POLICY,
        "selected_corpus_idx": meta["corpus_idx"].tolist(),
        "corrected_feature_parity_max_abs": parity_error,
        "loader_vs_ground_waveform_max_abs": loader_ground_waveform_max_abs,
        "input_waveform_source": (
            "exact rebuilt test_data xva; load_eval_rows input is parity-checked"
        ),
        "aligned_training_target": _compact_denoise(aligned),
        "unaligned_audit_target": _compact_denoise(unaligned),
        "oracle_fir_scored_on_aligned_target": bool(with_oracle_fir),
    }


def _environment_manifest() -> dict:
    source_hashes = {
        str(path.relative_to(REPO)): _sha256(path)
        for path in SOURCE_FILES
        if path.exists()
    }
    return {
        "created_unix_s": time.time(),
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "argv": sys.argv,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_policy": "enabled with warn_only=True",
        "git_head": _git(["rev-parse", "HEAD"]),
        "git_status_porcelain": _git(["status", "--porcelain"]),
        "source_sha256": source_hashes,
    }


def _artifact_inventory(out_dir: Path) -> dict[str, dict[str, Any]]:
    inventory = {}
    for path in sorted(out_dir.iterdir()):
        if path.is_file() and path.name not in {"COMPLETE.json", "run_manifest.json"}:
            inventory[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    return inventory


def _prepare_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    existing = list(path.iterdir())
    if existing:
        names = ", ".join(sorted(p.name for p in existing[:5]))
        raise FileExistsError(
            f"refusing to overwrite non-empty artifact directory {path} "
            f"(contains {names})"
        )


def run(args: argparse.Namespace) -> dict:
    seed_everything(args.seed)
    dev = resolve_device(args.device)
    print(f"[ground] loading corrected resampled pool (seed={args.seed})", flush=True)
    data = prepare_ground_state(
        pool_cache.load("resampled", seed=args.seed),
        seed=args.seed,
    )
    data["condition"] = "resampled"
    validation = _validate_ground_state(data)
    with (TRAINING / "artifacts" / "signallab-corpus" / "corpus.json").open() as f:
        raw_capture_length = int(json.load(f)["sampleCount"])
    if raw_capture_length != RAW_CAPTURE_LENGTH:
        raise AssertionError(
            f"expected raw corpus capture length {RAW_CAPTURE_LENGTH}, "
            f"manifest says {raw_capture_length}"
        )

    net = TransferUNet(**ARCHITECTURE)
    identity_error = _residual_identity_check(net, data)
    initial_state = _cpu_state_dict(net)
    health_rows = np.linspace(0, len(data["xtr"]) - 1, 12).astype(int)
    health_x = torch.from_numpy(np.asarray(data["xtr"][health_rows], dtype=np.float32))
    health_f = torch.from_numpy(np.asarray(data["ftr"][health_rows], dtype=np.float32))
    health_initial = encoder_health(net, health_x, health_f)

    plan = {
        "device": str(dev),
        "seed": args.seed,
        "episodes": args.episodes,
        "architecture": ARCHITECTURE,
        "crop": {"mode": "off", "guard_frac": 0.35},
        "condition": "resampled",
        "input_length": int(data["input_length"]),
        "raw_capture_length": raw_capture_length,
        "target_policy": TARGET_POLICY,
        "ground_state": data["ground_state"],
        "validation": validation,
        "initial_identity_max_abs_error": identity_error,
        "encoder_health_initial": health_initial,
    }
    print(json.dumps(_jsonable(plan), indent=2), flush=True)
    if args.dry_run:
        print("[ground] dry-run complete; no training or artifact writes", flush=True)
        return {"dry_run": True, **plan}

    tag = args.tag or f"corrected_resampled_cropoff_seed{args.seed}"
    if args.smoke:
        tag = f"{tag}_smoke"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", tag):
        raise ValueError("--tag may contain only letters, digits, '.', '_' and '-'")
    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else HERE / "artifacts" / "ground_state" / tag
    )
    _prepare_output_dir(out_dir)

    environment = _environment_manifest()
    _write_json(
        out_dir / "run_started.json",
        {
            "status": "training",
            "plan": plan,
            "environment": environment,
            "note": "COMPLETE.json is written last; its absence means the run is incomplete",
        },
    )
    torch.save(initial_state, out_dir / "random_init_state_dict.pt")

    episodes = 2 if args.smoke else args.episodes
    eval_every = 1 if args.smoke else min(args.eval_every, episodes)
    log_every = 1 if args.smoke else min(args.log_every, episodes)
    t0 = time.perf_counter()
    print(
        f"[ground] training {episodes} episodes on {dev}; checkpoint selection uses "
        f"{len(data['selection_rows'])} selection rows only",
        flush=True,
    )
    net, history, train_wall = train_transfer(
        net,
        data,
        dev,
        episodes=episodes,
        eval_every=eval_every,
        log_tag=tag,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_frac=args.warmup_frac,
        label_smoothing=args.label_smoothing,
        w_rec=args.w_rec,
        crop_cfg=CropConfig(mode="off", guard_frac=0.35),
        align_pair=True,
        mask_clean_recon=True,
        excess_loss=False,
        log_every=log_every,
        seed=args.seed,
        condition="resampled",
    )
    trained_state = _cpu_state_dict(net)
    torch.save(trained_state, out_dir / "state_dict.pt")
    _write_json(out_dir / "training_history.json", history)
    print(
        f"[ground] best-selection checkpoint saved after {train_wall:.1f}s; "
        "starting untouched-test report",
        flush=True,
    )

    selection = selection_data(data)
    heldout = test_data(data)
    openset_n = min(args.openset_n, 16) if args.smoke else args.openset_n
    denoise_n = min(args.denoise_n, 16) if args.smoke else args.denoise_n

    net = net.to(dev).eval()
    closed, prototypes = evaluate_v2(
        net,
        dev,
        heldout,
        include_fewshot=False,
    )
    confusion, balanced = _confusion(net, dev, heldout, prototypes)
    open_set = evaluate_openset(
        net,
        dev,
        heldout,
        prototypes,
        fmean=data["fmean"],
        fstd=data["fstd"],
        n_each=openset_n,
        calibration_data=selection,
        novelty_raw_len=raw_capture_length,
    )
    print(
        f"[ground] test closed={closed['closed_overall']:.4f} "
        f"balanced={balanced:.4f} open-AUROC={open_set['auroc_overall']:.4f}",
        flush=True,
    )

    random_net = TransferUNet(**ARCHITECTURE)
    random_net.load_state_dict(initial_state)
    scalar = scalar_transfer_report(
        net,
        random_net,
        data,
        selection,
        heldout,
        dev,
        target_key="snrDb",
    )
    net = net.to("cpu").eval()
    random_net = random_net.to("cpu")
    health_final = encoder_health(net, health_x, health_f)

    probe = cpn.sweep(
        net,
        prototypes,
        list(data["classes"]),
        torch.device("cpu"),
        fmean=data["fmean"],
        fstd=data["fstd"],
        condition="resampled",
        in_len=int(data["input_length"]),
        verbose=True,
    )
    denoise = _denoise_report(
        trained_state,
        data,
        heldout,
        n_rows=denoise_n,
        seed=args.seed + 29,
        with_oracle_fir=not args.skip_oracle_fir and not args.smoke,
    )

    metrics = {
        "status": "complete",
        "closed_set_test": {
            **closed,
            "balanced_accuracy": balanced,
            "confusion": confusion,
            "classes": list(data["classes"]),
            "n_test": int(len(heldout["yva"])),
        },
        "open_set_test": open_set,
        "canonical_probe": probe,
        "denoising_test": denoise,
        "scalar_transfer_test": scalar,
        "encoder_health": {
            "initial": health_initial,
            "final": health_final,
        },
        "selection": {
            "n": int(len(selection["yva"])),
            "best_closed_accuracy": history["best_val_acc"],
            "trace": history["val_acc"],
        },
        "timing": {
            "training_s": train_wall,
            "total_s": time.perf_counter() - t0,
        },
    }
    _write_json(out_dir / "metrics.json", metrics)
    _write_json(out_dir / "scalar_transfer.json", scalar)
    np.save(out_dir / "prototypes.npy", np.asarray(prototypes, dtype=np.float32))
    np.savez(
        out_dir / "feature_stats.npz",
        mean=np.asarray(data["fmean"], dtype=np.float32),
        std=np.asarray(data["fstd"], dtype=np.float32),
    )
    np.savez(
        out_dir / "split_corpus_indices.npz",
        train=np.asarray(data["tr_idx"], dtype=np.int64),
        enroll=np.asarray(data["en_idx"], dtype=np.int64),
        selection=np.asarray(selection["va_idx"], dtype=np.int64),
        test=np.asarray(heldout["va_idx"], dtype=np.int64),
    )

    training_config = {
        "episodes": episodes,
        "eval_every": eval_every,
        "log_every": log_every,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_frac": args.warmup_frac,
        "label_smoothing": args.label_smoothing,
        "w_rec": args.w_rec,
        "crop": {"mode": "off", "guard_frac": 0.35},
        "align_pair": True,
        "mask_clean_recon": True,
        "excess_loss": False,
        "condition": "resampled",
        "selection_rule": "highest nearest-prototype closed accuracy on selection_rows",
    }
    inference_bundle = {
        "schema_version": 1,
        "model": "TransferUNet",
        "architecture": ARCHITECTURE,
        "state_dict": trained_state,
        "classes": list(data["classes"]),
        "prototypes": torch.from_numpy(np.asarray(prototypes, dtype=np.float32)),
        "feature_mean": torch.from_numpy(np.asarray(data["fmean"], dtype=np.float32)),
        "feature_std": torch.from_numpy(np.asarray(data["fstd"], dtype=np.float32)),
        "open_set_threshold": float(open_set["threshold_p95_known"]),
        "preprocessing": {
            "condition": "resampled",
            "input_length": int(data["input_length"]),
            "module": "training/preprocess.py",
        },
        "target_policy": TARGET_POLICY,
        "scalar_transfer_head": scalar["head"],
    }
    torch.save(inference_bundle, out_dir / "inference_bundle.pt")

    manifest = {
        "schema_version": 1,
        "tag": tag,
        "seed": args.seed,
        "condition": "resampled",
        "architecture": ARCHITECTURE,
        "architecture_effective": net.config(),
        "param_count": int(sum(p.numel() for p in net.parameters())),
        "training": training_config,
        "target_policy": TARGET_POLICY,
        "ground_state": data["ground_state"],
        "split_validation": validation,
        "environment": environment,
        "artifacts": _artifact_inventory(out_dir),
        "reproduction_command": [
            sys.executable,
            "-u",
            str(Path(__file__).relative_to(REPO)),
            "--episodes",
            str(episodes),
            "--seed",
            str(args.seed),
            "--device",
            str(dev),
            "--tag",
            f"{tag}_reproduction",
            "--eval-every",
            str(eval_every),
            "--log-every",
            str(log_every),
            "--openset-n",
            str(openset_n),
            "--denoise-n",
            str(denoise_n),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--warmup-frac",
            str(args.warmup_frac),
            "--label-smoothing",
            str(args.label_smoothing),
            "--w-rec",
            str(args.w_rec),
            *(["--skip-oracle-fir"] if args.skip_oracle_fir or args.smoke else []),
        ],
        "resume_note": (
            "train_transfer restores the best model but does not expose optimizer/scheduler "
            "state; rerun from the recorded seed for reproduction rather than mid-run resume"
        ),
    }
    _write_json(out_dir / "run_manifest.json", manifest)
    complete = {
        "status": "complete",
        "tag": tag,
        "artifact_dir": str(out_dir),
        "total_s": metrics["timing"]["total_s"],
        "headline": {
            "closed_overall": closed["closed_overall"],
            "closed_balanced": balanced,
            "closed_clean": closed["closed_clean"],
            "closed_impaired": closed["closed_impaired"],
            "open_auroc": open_set["auroc_overall"],
            "canonical_worst_balanced": probe["worst_bal"],
            "canonical_valid": probe["valid"],
            "scalar_transfers": scalar["transfers"],
            "denoise_delta_vs_passthrough": denoise["aligned_training_target"][
                "aggregates"
            ]["model"]["delta_vs_passthrough"],
        },
        "manifest_sha256": _sha256(out_dir / "run_manifest.json"),
    }
    _write_json(out_dir / "COMPLETE.json", complete)
    print(json.dumps(_jsonable(complete), indent=2), flush=True)
    return complete


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--openset-n", type=int, default=300)
    parser.add_argument("--denoise-n", type=int, default=256)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-frac", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--w-rec", type=float, default=1.0)
    parser.add_argument(
        "--skip-oracle-fir",
        action="store_true",
        help="omit the expensive aligned-target oracle FIR comparator",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="two full-architecture episodes and reduced evaluation sizes",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate data/splits/architecture without training or writing artifacts",
    )
    args = parser.parse_args(argv)
    positive_ints = {
        "--episodes": args.episodes,
        "--eval-every": args.eval_every,
        "--log-every": args.log_every,
        "--openset-n": args.openset_n,
        "--denoise-n": args.denoise_n,
    }
    bad = [name for name, value in positive_ints.items() if value <= 0]
    if bad:
        parser.error(f"{', '.join(bad)} must be positive")
    if args.lr <= 0 or args.weight_decay < 0 or not 0 <= args.warmup_frac <= 1:
        parser.error("require lr > 0, weight_decay >= 0, and 0 <= warmup_frac <= 1")
    if not 0 <= args.label_smoothing < 1 or args.w_rec < 0:
        parser.error("require 0 <= label_smoothing < 1 and w_rec >= 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
