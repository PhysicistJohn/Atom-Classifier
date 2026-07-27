"""Leak-safe development evaluation of the strict time-domain patch geometry.

This script uses only the exposed train, enrollment, and immutable selection
indices from ``invariant_patch_data``.  It never reads, derives, or serializes
the consumed held-out half.  The production hybrid detector is evaluated only
as a reference; every ``time-correlation-v1`` model input is produced through
the forced, transform-free geometry path.

The frozen 4,000-episode checkpoints were trained with hybrid-v3 geometry.
Their scores here are therefore compatibility diagnostics, not a fair estimate
of the ceiling after retraining.  A two-episode real-CNN smoke proves that the
new preprocessing tensors can traverse the complete training loop.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
TRAINING = HERE.parent.parent
for path in (TRAINING, HERE.parent, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import invariant_patch_data as invariant_data  # noqa: E402
import preprocess as hybrid_preprocess  # noqa: E402
import run_invariant_cnn_dev as runner  # noqa: E402
import time_domain_invariant_patch_preprocess as td_preprocess  # noqa: E402
from invariant_patch_cnn import InvariantPatchCNN, InvariantPatchConfig  # noqa: E402
from train import embed_all, nearest, prototypes_from  # noqa: E402


CORPUS = TRAINING / "artifacts" / "signallab-corpus"
DEFAULT_CHECKPOINT_DIR = (
    HERE
    / "artifacts"
    / "invariant_patch"
    / "invariant_patch_screen4000_seed20260727"
)
DEFAULT_OUTPUT = (
    HERE
    / "artifacts"
    / "invariant_patch"
    / "time_domain_geometry_development.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _wrap(value: float) -> float:
    return float((value + 0.5) % 1.0 - 0.5)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
    }


def _geometry_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "time_vs_manifest_center_circular_abs": _summary(
            np.asarray([row["td_manifest_center_error"] for row in rows])
        ),
        "time_vs_manifest_bandwidth_abs": _summary(
            np.asarray([row["td_manifest_bandwidth_error"] for row in rows])
        ),
        "hybrid_vs_manifest_center_circular_abs": _summary(
            np.asarray([row["hybrid_manifest_center_error"] for row in rows])
        ),
        "hybrid_vs_manifest_bandwidth_abs": _summary(
            np.asarray([row["hybrid_manifest_bandwidth_error"] for row in rows])
        ),
        "time_vs_hybrid_center_circular_abs": _summary(
            np.asarray([row["td_hybrid_center_error"] for row in rows])
        ),
        "time_vs_hybrid_bandwidth_abs": _summary(
            np.asarray([row["td_hybrid_bandwidth_error"] for row in rows])
        ),
    }


def _preprocess_split(
    raw: np.memmap,
    indices: np.ndarray,
    items: list[dict[str, Any]],
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    channels = np.empty((len(indices), 2, 1024), dtype=np.float32)
    features = np.empty((len(indices), 12), dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for output_row, corpus_row in enumerate(np.asarray(indices, dtype=np.int64)):
        pair = np.asarray(raw[int(corpus_row)], dtype=np.float64)
        iq = pair[:, 0] + 1j * pair[:, 1]
        packed, raw_features, context = td_preprocess.preprocess(iq)
        hybrid_center, hybrid_bandwidth = hybrid_preprocess.estimate_band(iq)
        item = items[int(corpus_row)]
        manifest_center = float(item["centreOffsetFrac"])
        manifest_bandwidth = float(
            item["bandwidthHz"] / item["sampleRateHz"]
        )
        td_center = float(context["center"])
        td_bandwidth = float(context["bw"])
        channels[output_row] = packed
        features[output_row] = raw_features
        rows.append(
            {
                "class": str(item["cls"]),
                "impaired": bool(item["impaired"]),
                "snr_db": float(item["snrDb"]),
                "td_manifest_center_error": abs(
                    _wrap(td_center - manifest_center)
                ),
                "td_manifest_bandwidth_error": abs(
                    td_bandwidth - manifest_bandwidth
                ),
                "hybrid_manifest_center_error": abs(
                    _wrap(float(hybrid_center) - manifest_center)
                ),
                "hybrid_manifest_bandwidth_error": abs(
                    float(hybrid_bandwidth) - manifest_bandwidth
                ),
                "td_hybrid_center_error": abs(
                    _wrap(td_center - float(hybrid_center))
                ),
                "td_hybrid_bandwidth_error": abs(
                    td_bandwidth - float(hybrid_bandwidth)
                ),
            }
        )
    print(f"[time geometry] preprocessed {name}: {len(indices)}", flush=True)
    return channels, features, rows


def _closed(
    net: InvariantPatchCNN,
    data: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray]:
    enrollment = embed_all(net, data["xen"], data["fen"], device)
    prototypes = prototypes_from(
        enrollment, data["yen"], data["n_classes"]
    )
    selection = embed_all(net, data["xva"], data["fva"], device)
    prediction, _ = nearest(selection, prototypes)
    labels = np.asarray(data["yva"])
    per_class = {
        name: float(
            np.mean(prediction[labels == index] == index)
        )
        for index, name in enumerate(data["classes"])
    }
    return (
        {
            "accuracy": float(np.mean(prediction == labels)),
            "balanced_accuracy": float(np.mean(list(per_class.values()))),
            "per_class_recall": per_class,
        },
        prototypes,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    with (CORPUS / "corpus.json").open() as handle:
        manifest = json.load(handle)

    source = invariant_data.load(build_if_missing=False)
    if source["data_audit"].get("consumed_test_rows_exposed") != 0:
        raise AssertionError("consumed test rows were exposed")
    raw = np.memmap(
        CORPUS / "corpus.f32",
        dtype="<f4",
        mode="r",
        shape=(
            int(manifest["count"]),
            int(manifest["sampleCount"]),
            2,
        ),
    )
    split_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    geometry_rows: dict[str, list[dict[str, Any]]] = {}
    for name, key in (
        ("train", "tr_idx"),
        ("enroll", "en_idx"),
        ("selection", "va_idx"),
    ):
        channels, features, rows = _preprocess_split(
            raw,
            np.asarray(source[key]),
            manifest["items"],
            name=name,
        )
        split_arrays[name] = channels, features
        geometry_rows[name] = rows
    del raw

    train_features = split_arrays["train"][1]
    feature_mean = train_features.astype(np.float64).mean(axis=0).astype(np.float32)
    feature_std = (
        train_features.astype(np.float64).std(axis=0) + 1e-6
    ).astype(np.float32)

    def standardized(name: str) -> np.ndarray:
        return (
            (split_arrays[name][1] - feature_mean) / feature_std
        ).astype(np.float32)

    data = {
        **source,
        "xtr": split_arrays["train"][0],
        "ftr": standardized("train"),
        "xen": split_arrays["enroll"][0],
        "fen": standardized("enroll"),
        "xva": split_arrays["selection"][0],
        "fva": standardized("selection"),
        "fmean": feature_mean,
        "fstd": feature_std,
        "condition": td_preprocess.PREPROCESS_VERSION,
    }
    device = runner.resolve_device(args.device)
    runner.invariant_pp = td_preprocess
    models: dict[str, Any] = {}
    for encoder in ("real", "complex"):
        checkpoint = checkpoint_dir / f"{encoder}_state_dict.pt"
        net = InvariantPatchCNN(
            InvariantPatchConfig(encoder=encoder)
        )
        net.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True),
            strict=True,
        )
        net = net.to(device).eval()
        closed, prototypes = _closed(net, data, device)
        models[encoder] = {
            "checkpoint_sha256": _sha256(checkpoint),
            "frozen_hybrid_trained_checkpoint": True,
            "closed_selection": closed,
            "canonical_length": runner.canonical_length_sweep(
                net,
                prototypes,
                list(data["classes"]),
                data,
                device,
            ),
            "physical_scale": runner.physical_scale_sweep(
                net,
                prototypes,
                list(data["classes"]),
                data,
                device,
            ),
        }

    runner.seed_everything(args.seed)
    smoke_net, smoke_history = runner.train_model(
        InvariantPatchCNN(InvariantPatchConfig(encoder="real")),
        data,
        device,
        episodes=2,
        eval_every=2,
        seed=args.seed,
        lr=1e-3,
        weight_decay=2e-4,
        warmup_frac=0.03,
        label_smoothing=0.0,
        phase_augmentation=True,
    )
    smoke_closed, _ = _closed(smoke_net, data, device)

    selection_rows = geometry_rows["selection"]
    geometry = {
        name: {
            "all": _geometry_report(rows),
            "clean": _geometry_report(
                [row for row in rows if not row["impaired"]]
            ),
            "impaired": _geometry_report(
                [row for row in rows if row["impaired"]]
            ),
            "snr_ge_10_db": _geometry_report(
                [row for row in rows if row["snr_db"] >= 10.0]
            ),
        }
        for name, rows in geometry_rows.items()
    }
    geometry["selection"]["per_class"] = {
        class_name: _geometry_report(
            [row for row in selection_rows if row["class"] == class_name]
        )
        for class_name in source["classes"]
    }
    result = {
        "status": "complete",
        "development_only": True,
        "release_evidence": False,
        "consumed_test_rows_used": 0,
        "frontend": td_preprocess.preprocess_metadata(),
        "source_cache_contract": source["cache_contract"],
        "geometry": geometry,
        "frozen_checkpoint_compatibility": models,
        "two_episode_training_smoke": {
            "completed": True,
            "training": smoke_history,
            "closed_selection": smoke_closed,
        },
        "device": str(device),
        "wall_clock_s": time.perf_counter() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w") as handle:
        json.dump(
            _jsonable(result),
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    temporary.replace(output)
    print(f"[time geometry] wrote {output}", flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
