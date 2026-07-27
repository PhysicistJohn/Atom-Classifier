"""Selection-only development harness for the invariant patch-set classifier.

This runner never exposes or scores the already-consumed test half.  It builds a
source-hashed pool from train, enrollment, and the immutable selection half,
trains one or both small patch encoders, then reports:

* closed-set accuracy on the selection population;
* known-only calibrated open-set diagnostics on development novelty fixtures;
* the existing canonical capture-length gate through the invariant front end;
* a physical scale sweep where sample count changes inversely with normalized
  bandwidth, preserving observation duration.

All reported novelty and canonical measurements are development evidence.
Release evidence requires a newly sealed synthetic population and real SDR data.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
ZPAB = HERE.parent
TRAINING = ZPAB.parent
REPO = TRAINING.parent
for path in (str(TRAINING), str(ZPAB), str(HERE)):
    if path not in os.sys.path:
        os.sys.path.insert(0, path)

import canonical_probe_net as canonical_net  # noqa: E402
import invariant_patch_data as invariant_data  # noqa: E402
import invariant_patch_preprocess as invariant_pp  # noqa: E402
import openset_eval  # noqa: E402
import preprocess as pp  # noqa: E402
from canonical_probe import GATE, LENGTHS, build_probes  # noqa: E402
from invariant_patch_cnn import (  # noqa: E402
    InvariantPatchCNN,
    InvariantPatchConfig,
)
from run_bounded_dev import (  # noqa: E402
    _closed_report,
    _open_set_cv_report,
    stratified_folds,
)
from train import (  # noqa: E402
    embed_all,
    nearest,
    prototypes_from,
    sq_dist,
    supcon_loss,
)


DEFAULT_SEED = 20260727
RAW_CAPTURE_LENGTH = 16384
K_SHOT = 5
Q_QUERY = 5
SCALE_FACTORS = (0.5, 0.75, 1.0, 1.5, 2.0)


def seed_everything(seed: int) -> None:
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
        raise RuntimeError("MPS was requested but is unavailable")
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
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(
            _jsonable(payload),
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    os.replace(temporary, path)


def _balanced_accuracy(pred: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    return float(
        np.mean(
            [
                np.mean(pred[labels == class_index] == class_index)
                for class_index in range(n_classes)
            ]
        )
    )


def _phase_augment(x: torch.Tensor) -> torch.Tensor:
    phase = 2.0 * math.pi * torch.rand(
        x.shape[0], 1, 1, dtype=x.dtype, device=x.device
    )
    cosine, sine = torch.cos(phase), torch.sin(phase)
    real = x[:, 0:1] * cosine - x[:, 1:2] * sine
    imag = x[:, 0:1] * sine + x[:, 1:2] * cosine
    return torch.cat((real, imag), dim=1)


def _prototype_orthogonality_loss(prototypes: torch.Tensor) -> torch.Tensor:
    """Keep known class directions separated on the unit embedding sphere."""
    unit = F.normalize(prototypes, dim=-1)
    gram = unit @ unit.t()
    mask = ~torch.eye(
        len(prototypes), dtype=torch.bool, device=prototypes.device
    )
    return gram[mask].square().mean()


def _sample_episode_positions(
    idx_by_class: list[np.ndarray],
    rng: np.random.Generator,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = np.concatenate(
        [
            rng.choice(
                idx_by_class[class_index],
                K_SHOT + Q_QUERY,
                replace=False,
            )
            for class_index in range(n_classes)
        ]
    )
    step = K_SHOT + Q_QUERY
    support = np.concatenate(
        [
            selected[c * step : c * step + K_SHOT]
            for c in range(n_classes)
        ]
    )
    query = np.concatenate(
        [
            selected[c * step + K_SHOT : (c + 1) * step]
            for c in range(n_classes)
        ]
    )
    support_labels = np.repeat(np.arange(n_classes), K_SHOT)
    query_labels = np.repeat(np.arange(n_classes), Q_QUERY)
    return support, query, support_labels, query_labels


def train_model(
    net: InvariantPatchCNN,
    data: dict[str, Any],
    dev: torch.device,
    *,
    episodes: int,
    eval_every: int,
    seed: int,
    lr: float,
    weight_decay: float,
    warmup_frac: float,
    label_smoothing: float,
    phase_augmentation: bool,
    separation_weight: float = 0.0,
    supcon_weight: float = 0.0,
) -> tuple[InvariantPatchCNN, dict[str, Any]]:
    """Prototypical training with a positive, restored logit scale."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if separation_weight < 0.0 or supcon_weight < 0.0:
        raise ValueError("auxiliary loss weights must be non-negative")
    rng = np.random.default_rng(seed)
    net = net.to(dev)
    log_scale = torch.nn.Parameter(
        torch.tensor(math.log(10.0), dtype=torch.float32, device=dev)
    )
    optimizer = torch.optim.Adam(
        [
            {"params": list(net.parameters()), "weight_decay": weight_decay},
            {"params": [log_scale], "weight_decay": 0.0},
        ],
        lr=lr,
    )
    warmup = max(1, int(episodes * warmup_frac))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(episodes - warmup, 1)
    )

    def evaluate_selection() -> tuple[float, float]:
        enrollment = embed_all(net, data["xen"], data["fen"], dev)
        prototypes = prototypes_from(
            enrollment, data["yen"], data["n_classes"]
        )
        selection = embed_all(net, data["xva"], data["fva"], dev)
        prediction, _ = nearest(selection, prototypes)
        return (
            float(np.mean(prediction == data["yva"])),
            _balanced_accuracy(prediction, data["yva"], data["n_classes"]),
        )

    history: dict[str, Any] = {
        "loss": [],
        "loss_ce": [],
        "loss_separation": [],
        "loss_supcon": [],
        "selection": [],
        "logit_scale": [],
    }
    best_score = (-1.0, -1.0)
    best_state: tuple[dict[str, torch.Tensor], float] | None = None
    running = {"total": 0.0, "ce": 0.0, "separation": 0.0, "supcon": 0.0}
    started = time.perf_counter()
    idx_by_class = [
        np.asarray(rows, dtype=np.int64) for rows in data["idx_by_class"]
    ]

    for episode in range(episodes):
        net.train()
        if episode < warmup:
            for group in optimizer.param_groups:
                group["lr"] = lr * (episode + 1) / warmup
        support, query, support_labels, query_labels = _sample_episode_positions(
            idx_by_class, rng, data["n_classes"]
        )
        positions = np.concatenate((support, query))
        x = torch.from_numpy(np.asarray(data["xtr"])[positions]).to(dev)
        features = torch.from_numpy(np.asarray(data["ftr"])[positions]).to(dev)
        if phase_augmentation:
            x = _phase_augment(x)
        embeddings = net(x, features)
        support_embeddings = embeddings[: len(support)]
        query_embeddings = embeddings[len(support) :]
        support_labels_t = torch.from_numpy(support_labels).to(dev)
        prototypes = torch.stack(
            [
                support_embeddings[support_labels_t == class_index].mean(0)
                for class_index in range(data["n_classes"])
            ]
        )
        scale = log_scale.exp().clamp(1e-3, 100.0)
        logits = -sq_dist(query_embeddings, prototypes) * scale
        loss_ce = F.cross_entropy(
            logits,
            torch.from_numpy(query_labels).to(dev),
            label_smoothing=label_smoothing,
        )
        separation = _prototype_orthogonality_loss(prototypes)
        all_labels = torch.from_numpy(
            np.concatenate((support_labels, query_labels))
        ).to(dev)
        contrastive = (
            supcon_loss(embeddings, all_labels)
            if supcon_weight > 0.0
            else embeddings.new_zeros(())
        )
        loss = (
            loss_ce
            + separation_weight * separation
            + supcon_weight * contrastive
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if episode >= warmup:
            scheduler.step()
        running["total"] += float(loss.detach().cpu())
        running["ce"] += float(loss_ce.detach().cpu())
        running["separation"] += float(separation.detach().cpu())
        running["supcon"] += float(contrastive.detach().cpu())

        log_every = max(1, min(250, episodes))
        if (episode + 1) % log_every == 0:
            mean_loss = {
                name: value / log_every for name, value in running.items()
            }
            running = {
                "total": 0.0,
                "ce": 0.0,
                "separation": 0.0,
                "supcon": 0.0,
            }
            current_scale = float(log_scale.detach().exp().cpu())
            history["loss"].append(
                {"episode": episode + 1, "value": mean_loss["total"]}
            )
            history["loss_ce"].append(
                {"episode": episode + 1, "value": mean_loss["ce"]}
            )
            history["loss_separation"].append(
                {"episode": episode + 1, "value": mean_loss["separation"]}
            )
            history["loss_supcon"].append(
                {"episode": episode + 1, "value": mean_loss["supcon"]}
            )
            history["logit_scale"].append(
                {"episode": episode + 1, "value": current_scale}
            )
            elapsed = time.perf_counter() - started
            eta_minutes = (
                elapsed / (episode + 1) * (episodes - episode - 1) / 60.0
            )
            print(
                f"  [{net.cfg.encoder}] ep {episode + 1:5d}/{episodes} "
                f"loss {mean_loss['total']:.4f} ce {mean_loss['ce']:.4f} "
                f"sep {mean_loss['separation']:.3f} "
                f"sup {mean_loss['supcon']:.3f} scale {current_scale:.2f} "
                f"eta {eta_minutes:.1f}m",
                flush=True,
            )

        if (episode + 1) % eval_every == 0 or episode + 1 == episodes:
            accuracy, balanced = evaluate_selection()
            row = {
                "episode": episode + 1,
                "accuracy": accuracy,
                "balanced_accuracy": balanced,
            }
            history["selection"].append(row)
            score = (balanced, accuracy)
            marker = ""
            if score > best_score:
                best_score = score
                best_state = (
                    copy.deepcopy(net.state_dict()),
                    float(log_scale.detach().cpu()),
                )
                marker = " <- best"
            print(
                f"    [{net.cfg.encoder}] selection acc {accuracy:.4f} "
                f"bal {balanced:.4f}{marker}",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("training completed without a selection checkpoint")
    net.load_state_dict(best_state[0], strict=True)
    with torch.no_grad():
        log_scale.fill_(best_state[1])
    history.update(
        {
            "episodes": episodes,
            "best_selection_balanced_accuracy": best_score[0],
            "best_selection_accuracy": best_score[1],
            "final_logit_scale": float(log_scale.detach().exp().cpu()),
            "wall_clock_s": time.perf_counter() - started,
            "separation_weight": float(separation_weight),
            "supcon_weight": float(supcon_weight),
        }
    )
    return net.eval(), history


def build_development_novelty(
    *,
    n_each: int,
    seed: int,
    patch_length: int,
    patch_count: int,
    target_frac: float,
    raw_length: int = RAW_CAPTURE_LENGTH,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build novelty through the identical invariant front end."""
    if n_each <= 0 or raw_length <= 0:
        raise ValueError("novelty count and raw length must be positive")
    rng = np.random.default_rng(seed)
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, generator in openset_eval.NOVELTY.items():
        waveforms, features = [], []
        for _ in range(n_each):
            raw = generator(raw_length, rng)
            packed, raw_features, _ = invariant_pp.preprocess(
                raw,
                patch_length=patch_length,
                patch_count=patch_count,
                target_frac=target_frac,
            )
            waveforms.append(packed)
            features.append(raw_features)
        result[name] = (
            np.stack(waveforms).astype(np.float32),
            np.stack(features).astype(np.float32),
        )
    return result


def _front_config(net: torch.nn.Module) -> InvariantPatchConfig:
    """Return the invariant-front geometry for a branch or compatible fusion."""
    config = getattr(net, "cfg", None)
    if isinstance(config, InvariantPatchConfig):
        return config
    real_branch = getattr(net, "real_branch", None)
    config = getattr(real_branch, "cfg", None)
    if isinstance(config, InvariantPatchConfig):
        return config
    raise TypeError(
        "net must expose InvariantPatchConfig as cfg or real_branch.cfg"
    )


def _probe_embeddings(
    net: torch.nn.Module,
    probes: list[tuple[str, str, np.ndarray]],
    classes: list[str],
    data: dict[str, Any],
    dev: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    config = _front_config(net)
    waveforms, features, wanted, contexts = [], [], [], []
    for _label, class_name, raw in probes:
        packed, raw_features, context = invariant_pp.preprocess(
            np.asarray(raw, dtype=np.complex128),
            patch_length=config.patch_length,
            patch_count=config.patch_count,
            target_frac=float(data["cache_contract"]["config"]["target_frac"]),
        )
        waveforms.append(packed)
        features.append(raw_features)
        wanted.append(classes.index(class_name))
        contexts.append(context)
    feature_array = (
        np.stack(features).astype(np.float32) - np.asarray(data["fmean"])
    ) / np.asarray(data["fstd"])
    embeddings = embed_all(
        net,
        np.stack(waveforms).astype(np.float32),
        feature_array.astype(np.float32),
        dev,
    )
    return embeddings, np.asarray(wanted, dtype=np.int64), contexts


def _probe_row(
    embeddings: np.ndarray,
    wanted: np.ndarray,
    contexts: list[dict[str, Any]],
    prototypes: np.ndarray,
    classes: list[str],
) -> dict[str, Any]:
    prediction, _ = nearest(embeddings, prototypes)
    recalls = [
        float(np.mean(prediction[wanted == class_index] == class_index))
        for class_index in np.unique(wanted)
    ]
    unit = embeddings / (
        np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    )
    upper = np.triu_indices(len(unit), 1)
    return {
        "correct": float(np.mean(prediction == wanted)),
        "bal": float(np.mean(recalls)),
        "cos": float((unit @ unit.T)[upper].mean()),
        "n_predicted_classes": int(np.unique(prediction).size),
        "active_length_median": float(
            np.median([context["active_length"] for context in contexts])
        ),
        "active_length_min": int(
            min(context["active_length"] for context in contexts)
        ),
        "active_length_max": int(
            max(context["active_length"] for context in contexts)
        ),
        "per_class": {
            classes[class_index]: float(
                np.mean(
                    prediction[wanted == class_index]
                    == class_index
                )
            )
            for class_index in np.unique(wanted)
        },
    }


def canonical_length_sweep(
    net: InvariantPatchCNN,
    prototypes: np.ndarray,
    classes: list[str],
    data: dict[str, Any],
    dev: torch.device,
    *,
    lengths: tuple[int, ...] = LENGTHS,
    matched_length: int = RAW_CAPTURE_LENGTH,
    seed: int = 7,
) -> dict[str, Any]:
    rows = []
    reference: np.ndarray | None = None
    for length in lengths:
        probes = build_probes(np.random.default_rng(seed), int(length))
        embeddings, wanted, contexts = _probe_embeddings(
            net, probes, classes, data, dev
        )
        row = _probe_row(
            embeddings, wanted, contexts, prototypes, classes
        )
        row["n"] = int(length)
        row["input_packed_length"] = net.packed_length
        rows.append(row)
        if int(length) == int(matched_length):
            reference = embeddings
    summary = canonical_net._summarize_rows(
        rows, matched_length=matched_length, gate=GATE
    )
    summary.update(
        {
            "rows": rows,
            "lengths": [int(value) for value in lengths],
            "matched_length": int(matched_length),
            "gate": dict(GATE),
            "valid_reference_present": reference is not None,
            "representation": "fixed-u patches with symmetric set pooling",
            "input_geometry_changes_with_length": False,
        }
    )
    return summary


def physical_scale_sweep(
    net: InvariantPatchCNN,
    prototypes: np.ndarray,
    classes: list[str],
    data: dict[str, Any],
    dev: torch.device,
    *,
    factors: tuple[float, ...] = SCALE_FACTORS,
    base_length: int = RAW_CAPTURE_LENGTH,
    seed: int = 7,
) -> dict[str, Any]:
    """Scale frequencies by ``s`` and samples by ``1/s`` at fixed duration."""
    base_probes = build_probes(np.random.default_rng(seed), base_length)
    rows = []
    embedding_by_factor: dict[float, np.ndarray] = {}
    for factor in factors:
        if not np.isfinite(factor) or factor <= 0.0:
            raise ValueError("scale factors must be finite and positive")
        scaled_length = max(2, int(round(base_length / factor)))
        scaled_probes = [
            (
                label,
                class_name,
                pp.lin_resample(
                    np.asarray(raw, dtype=np.complex128), scaled_length
                ),
            )
            for label, class_name, raw in base_probes
        ]
        embeddings, wanted, contexts = _probe_embeddings(
            net, scaled_probes, classes, data, dev
        )
        embedding_by_factor[float(factor)] = embeddings
        row = _probe_row(
            embeddings, wanted, contexts, prototypes, classes
        )
        row.update(
            {
                "scale_factor": float(factor),
                "raw_length": scaled_length,
                "input_packed_length": net.packed_length,
            }
        )
        rows.append(row)

    reference = embedding_by_factor.get(1.0)
    if reference is None:
        raise ValueError("physical scale sweep must include factor 1.0")
    reference = reference / (
        np.linalg.norm(reference, axis=1, keepdims=True) + 1e-12
    )
    for row in rows:
        current = embedding_by_factor[row["scale_factor"]]
        current = current / (
            np.linalg.norm(current, axis=1, keepdims=True) + 1e-12
        )
        agreement = np.sum(reference * current, axis=1)
        row["embedding_cosine_to_scale1_mean"] = float(agreement.mean())
        row["embedding_cosine_to_scale1_min"] = float(agreement.min())

    matched = next(row for row in rows if row["scale_factor"] == 1.0)
    worst_bal = min(float(row["bal"]) for row in rows)
    worst_cos = max(float(row["cos"]) for row in rows)
    valid = (
        float(matched["bal"]) >= float(GATE["min_valid_bal_acc"])
        and int(matched["n_predicted_classes"]) >= 2
    )
    return {
        "rows": rows,
        "factors": [float(value) for value in factors],
        "base_length": int(base_length),
        "matched_bal": float(matched["bal"]),
        "worst_bal": worst_bal,
        "best_bal": max(float(row["bal"]) for row in rows),
        "worst_cos": worst_cos,
        "valid": bool(valid),
        "passes_gate": bool(
            valid
            and worst_bal >= float(GATE["min_bal_acc"])
            and worst_cos <= float(GATE["max_mean_pairwise_cos"])
        ),
        "gate": dict(GATE),
        "definition": (
            "frequency/sample-rate scale factor s with raw sample count N/s, "
            "so physical observation duration is held fixed"
        ),
        "input_geometry_changes_with_scale": False,
    }


def evaluate_model(
    net: InvariantPatchCNN,
    data: dict[str, Any],
    dev: torch.device,
    *,
    novelty_n: int,
    seed: int,
    cv_folds: int,
) -> tuple[dict[str, Any], np.ndarray]:
    folds = stratified_folds(
        data["yva"],
        data["imp_va"],
        n_folds=cv_folds,
        seed=seed + 101,
    )
    enrollment = embed_all(net, data["xen"], data["fen"], dev)
    prototypes = prototypes_from(
        enrollment, data["yen"], data["n_classes"]
    )
    selection = embed_all(net, data["xva"], data["fva"], dev)
    closed = _closed_report(
        selection,
        data["yva"],
        data["imp_va"],
        prototypes,
        list(data["classes"]),
        folds,
    )
    novelty = build_development_novelty(
        n_each=novelty_n,
        seed=seed + 211,
        patch_length=net.cfg.patch_length,
        patch_count=net.cfg.patch_count,
        target_frac=float(data["cache_contract"]["config"]["target_frac"]),
    )
    open_set = _open_set_cv_report(
        net,
        dev,
        data,
        prototypes,
        selection,
        folds,
        novelty,
    )
    length = canonical_length_sweep(
        net, prototypes, list(data["classes"]), data, dev
    )
    scale = physical_scale_sweep(
        net, prototypes, list(data["classes"]), data, dev
    )
    return (
        {
            "development_only": True,
            "release_evidence": False,
            "consumed_test_rows_used": 0,
            "closed_selection": closed,
            "open_set_selection_cv": open_set,
            "canonical_length_development": length,
            "canonical_scale_development": scale,
        },
        prototypes,
    )


def _parse_encoders(raw: str) -> list[str]:
    encoders = [value.strip() for value in raw.split(",") if value.strip()]
    if not encoders or any(value not in {"real", "complex"} for value in encoders):
        raise ValueError("encoders must be a comma-separated subset of real,complex")
    if len(set(encoders)) != len(encoders):
        raise ValueError("encoders may not contain duplicates")
    return encoders


def run(args: argparse.Namespace) -> dict[str, Any]:
    dev = resolve_device(args.device)
    encoders = _parse_encoders(args.encoders)
    episodes = 2 if args.smoke else args.episodes
    novelty_n = min(args.novelty_n, 4) if args.smoke else args.novelty_n
    eval_every = min(args.eval_every, episodes)
    tag = args.tag or f"invariant_patch_seed{args.seed}"
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else HERE / "artifacts" / "invariant_patch" / tag
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"{output_dir} is not empty; use a fresh tag/output directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[invariant] loading train/enroll/selection-only pool", flush=True)
    data = invariant_data.load(
        patch_length=args.patch_length,
        patch_count=args.patch_count,
        target_frac=args.target_frac,
        model_seed=args.seed,
        build_if_missing=True,
    )
    if data["data_audit"].get("consumed_test_rows_exposed") != 0:
        raise AssertionError("invariant data loader exposed consumed test rows")
    _write_json(
        output_dir / "DEVELOPMENT_ONLY.json",
        {
            "development_only": True,
            "release_evidence": False,
            "consumed_test_rows_used": 0,
            "reason": (
                "the corrected ground-state test half was already consumed; "
                "this run uses only train, enrollment, and selection"
            ),
            "data_audit": data["data_audit"],
            "cache_contract": data["cache_contract"],
        },
    )

    model_records: dict[str, Any] = {}
    for encoder in encoders:
        # Every architecture receives the identical episode sequence and the same
        # development novelty population. Architecture construction consumes a
        # different number of Torch draws, but must not change data sampling.
        model_seed = args.seed
        seed_everything(model_seed)
        config = InvariantPatchConfig(
            patch_length=args.patch_length,
            patch_count=args.patch_count,
            encoder=encoder,
            patch_dim=args.patch_dim,
            hidden=args.hidden,
            set_pool=args.set_pool,
            dropout=args.dropout,
        )
        net = InvariantPatchCNN(config)
        parameter_count = int(sum(p.numel() for p in net.parameters()))
        print(
            f"[invariant] training {encoder}, params={parameter_count:,}",
            flush=True,
        )
        net, history = train_model(
            net,
            data,
            dev,
            episodes=episodes,
            eval_every=eval_every,
            seed=model_seed,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_frac=args.warmup_frac,
            label_smoothing=args.label_smoothing,
            phase_augmentation=args.phase_augmentation,
            separation_weight=args.separation_weight,
            supcon_weight=args.supcon_weight,
        )
        print(f"[invariant] evaluating {encoder}", flush=True)
        metrics, prototypes = evaluate_model(
            net,
            data,
            dev,
            novelty_n=novelty_n,
            seed=args.seed,
            cv_folds=args.cv_folds,
        )
        torch.save(net.state_dict(), output_dir / f"{encoder}_state_dict.pt")
        np.save(output_dir / f"{encoder}_prototypes.npy", prototypes)
        model_record = {
            "encoder": encoder,
            "seed": model_seed,
            "architecture": net.config(),
            "parameter_count": parameter_count,
            "training": history,
            "metrics": metrics,
        }
        model_records[encoder] = model_record
        _write_json(output_dir / f"{encoder}_dev_metrics.json", model_record)

    winner = max(
        model_records,
        key=lambda name: (
            model_records[name]["metrics"]["closed_selection"][
                "balanced_accuracy"
            ],
            model_records[name]["metrics"]["closed_selection"]["accuracy"],
        ),
    )
    result = {
        "status": "complete",
        "development_only": True,
        "release_evidence": False,
        "consumed_test_rows_used": 0,
        "tag": tag,
        "device": str(dev),
        "seed": args.seed,
        "models": model_records,
        "selection_winner": winner,
        "configuration": {
            "encoders": encoders,
            "episodes": episodes,
            "eval_every": eval_every,
            "novelty_n_each": novelty_n,
            "cv_folds": args.cv_folds,
            "phase_augmentation": args.phase_augmentation,
            "separation_weight": args.separation_weight,
            "supcon_weight": args.supcon_weight,
            "optimizer": {
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "warmup_frac": args.warmup_frac,
                "label_smoothing": args.label_smoothing,
            },
        },
        "data_audit": data["data_audit"],
        "cache_contract": data["cache_contract"],
    }
    _write_json(output_dir / "dev_metrics.json", result)
    print(f"[invariant] complete -> {output_dir}", flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--tag")
    parser.add_argument("--output-dir")
    parser.add_argument("--encoders", default="real,complex")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--novelty-n", type=int, default=300)
    parser.add_argument("--cv-folds", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--patch-length", type=int, default=64)
    parser.add_argument("--patch-count", type=int, default=16)
    parser.add_argument("--target-frac", type=float, default=pp.TARGET_FRAC)
    parser.add_argument("--patch-dim", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument(
        "--set-pool", choices=("mean", "mean_std"), default="mean_std"
    )
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--warmup-frac", type=float, default=0.03)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--separation-weight", type=float, default=0.0)
    parser.add_argument("--supcon-weight", type=float, default=0.0)
    parser.add_argument(
        "--phase-augmentation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


if __name__ == "__main__":
    started = time.perf_counter()
    output = run(build_parser().parse_args())
    print(
        json.dumps(
            {
                "status": output["status"],
                "selection_winner": output["selection_winner"],
                "elapsed_s": time.perf_counter() - started,
            },
            indent=2,
        )
    )
