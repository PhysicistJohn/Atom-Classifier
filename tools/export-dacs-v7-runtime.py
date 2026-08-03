#!/usr/bin/env python3
"""Export and validate the DACS v7 encoder as a hash-bound browser package.

The training checkpoint is episodic: it contains an encoder but no frozen
class prototypes.  This exporter makes the missing production decision state
explicit by freezing one 64-row prototype mean per class and dwell, then
measuring those *fixed* prototypes against five independent query-offset
plans.  It exports only the encoder and confidence head; the training-only
decoder and optimizer state are excluded.

The package is deliberately closed-set.  Atomizer must retain its released
open-set gate ahead of DACS and may invoke DACS only for captures at the exact
20 Msps training rate.  The confidence head is included for telemetry but is
not calibrated and has no release decision authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO = Path(__file__).resolve().parents[1]
V7_DIR = REPO / "training/zplane_ab/v2_full_variation/v7_probe"
DEFAULT_CHECKPOINT = (
    REPO
    / "training/artifacts/sota-v2-2014019-12e7eef-20260801"
    / "content_v2_bf16.safetensors"
)
DEFAULT_CORPUS = REPO / "training/artifacts/longdwell-production-corpus-v2"

PACKAGE_SCHEMA = "atomos.dacs-v7.runtime-package"
PROTOTYPE_SCHEMA = "atomos.dacs-v7.fixed-prototypes"
VALIDATION_SCHEMA = "atomos.dacs-v7.fixed-prototype-validation"
MODEL_FILENAME = "dacs-v7-encoder.onnx"
PROTOTYPE_FILENAME = "dacs-v7-prototypes.json"
VALIDATION_FILENAME = "dacs-v7-validation.json"
MANIFEST_FILENAME = "runtime-package-manifest.json"
CLASSES = ("am", "bluetooth", "cw", "dsss", "fm", "gsm", "ofdm")
DWELLS = {"1ms": 20_000, "2.5ms": 50_000, "10ms": 200_000}
EVAL_SEEDS = (20260740, 20260741, 20260742, 20260743, 20260744)
TARGET_SAMPLE_RATE_HZ = 20_000_000
PROTOTYPE_ROWS = 64


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value))


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO, text=True, stderr=subprocess.STDOUT
    ).strip()


def require_clean_source() -> tuple[str, str]:
    status = git_output("status", "--porcelain", "--untracked-files=all")
    if status:
        raise SystemExit(
            "release export requires a clean Atom-Classifier worktree:\n"
            + status
        )
    return git_output("rev-parse", "HEAD"), git_output("branch", "--show-current")


def descriptor(path: Path, schema: str, role: str) -> dict[str, Any]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "schema": schema,
        "runtime_role": role,
    }


def load_training_modules():
    sys.path.insert(0, str(V7_DIR))
    import mlx.core as mx  # type: ignore
    import torch  # type: ignore
    import v7_trainer as torch_trainer  # type: ignore
    import v7_trainer_mlx as mlx_trainer  # type: ignore

    return mx, torch, torch_trainer, mlx_trainer


def verify_checkpoint(checkpoint: Path, mlx_trainer: Any) -> dict[str, Any]:
    sidecar_path = Path(str(checkpoint) + ".json")
    if not sidecar_path.is_file():
        raise SystemExit(f"checkpoint sidecar is missing: {sidecar_path}")
    sidecar = json.loads(sidecar_path.read_text())
    model = mlx_trainer.TrainState(
        width=int(sidecar["config"]["width"]),
        embed_dim=int(sidecar["config"]["embed_dim"]),
    )
    kind, _optimizer, loaded_sidecar = mlx_trainer.load_init_checkpoint(
        model, str(checkpoint)
    )
    if kind != "native" or loaded_sidecar is None:
        raise SystemExit("DACS release export requires a native MLX checkpoint")
    measured = mlx_trainer.params_sha256(model)
    expected = loaded_sidecar.get("params_sha256")
    if measured != expected:
        raise SystemExit(
            f"checkpoint parameter hash mismatch: {measured} != {expected}"
        )
    return {
        "sidecar": sidecar,
        "params_sha256": measured,
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_sidecar_sha256": sha256_file(sidecar_path),
    }


def torch_encoder_from_mlx(
    checkpoint: Path,
    mx: Any,
    torch: Any,
    trainer: Any,
    *,
    width: int,
    embed_dim: int,
):
    arrays, _metadata = mx.load(str(checkpoint), return_metadata=True)
    state: dict[str, Any] = {}
    for key, value in arrays.items():
        if not key.startswith("params.net."):
            continue
        name = key[len("params.net.") :].replace("conf.layers.", "conf.")
        array = np.asarray(value.astype(mx.float32))
        if name.endswith("conv.weight") or name == "out.weight":
            array = np.ascontiguousarray(array.transpose(0, 3, 1, 2))
        state[name] = torch.from_numpy(array)

    net = trainer.V7Net(width=width, embed_dim=embed_dim)
    net.load_state_dict(state, strict=True)
    net.eval()

    class Encoder(torch.nn.Module):
        def __init__(self, model: Any):
            super().__init__()
            self.model = model

        def forward(self, spectrogram: Any):
            embedding, confidence_logit, _bottleneck = self.model.encode(
                spectrogram
            )
            return embedding, confidence_logit

    return net, Encoder(net).eval()


def export_onnx(encoder: Any, torch: Any, path: Path) -> dict[str, Any]:
    dummy = torch.zeros((1, 3, 624, 33), dtype=torch.float32)
    torch.onnx.export(
        encoder,
        (dummy,),
        str(path),
        input_names=["spectrogram"],
        output_names=["embedding", "confidence_logit"],
        opset_version=18,
        dynamo=False,
        dynamic_axes={
            "spectrogram": {0: "batch", 2: "frames"},
            "embedding": {0: "batch"},
            "confidence_logit": {0: "batch"},
        },
        do_constant_folding=True,
    )
    import onnx  # type: ignore

    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    allowed = {
        "Add",
        "Clip",
        "Concat",
        "Constant",
        "Conv",
        "Div",
        "Expand",
        "Gemm",
        "InstanceNormalization",
        "Mul",
        "ReduceL2",
        "ReduceMax",
        "ReduceMean",
        "Reshape",
        "Shape",
        "Sigmoid",
        "Squeeze",
    }
    operators = {node.op_type for node in model.graph.node}
    if not operators <= allowed:
        raise SystemExit(f"unexpected ONNX operators: {sorted(operators - allowed)}")
    return {
        "nodes": len(model.graph.node),
        "operators": sorted(operators),
        "initializer_values": int(
            sum(
                np.prod(initializer.dims, dtype=np.int64)
                for initializer in model.graph.initializer
            )
        ),
    }


def chunks(values: list[int], size: int) -> Iterable[list[int]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def embed_rows(
    net: Any,
    trainer: Any,
    torch: Any,
    noisy: Any,
    row_ids: list[int],
    offsets: dict[int, int],
    length: int,
    device: Any,
) -> tuple[np.ndarray, np.ndarray]:
    batch_size = 16 if length <= 20_000 else 8 if length <= 50_000 else 2
    embeddings: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    with torch.no_grad():
        for batch_ids in chunks(row_ids, batch_size):
            iq = np.stack(
                [
                    np.asarray(
                        noisy[row, offsets[int(row)] : offsets[int(row)] + length],
                        dtype=np.complex64,
                    )
                    for row in batch_ids
                ]
            )
            tensor = torch.from_numpy(iq).to(device)
            spec = trainer.spectrogram(trainer.rms_normalize(tensor))
            embedding, confidence, _bottleneck = net.encode(spec)
            embeddings.append(embedding.detach().cpu().numpy())
            confidences.append(confidence.detach().cpu().numpy())
    return np.concatenate(embeddings), np.concatenate(confidences)


def split_manifest(manifest: dict[str, Any]):
    train_by_class = {label: [] for label in CLASSES}
    eval_rows: list[dict[str, Any]] = []
    for row in manifest["rows"]:
        if row["role"] == "train":
            train_by_class[row["cls"]].append(int(row["row"]))
        else:
            eval_rows.append(row)
    all_train = [row for label in CLASSES for row in train_by_class[label]]
    return train_by_class, eval_rows, all_train


def confusion_and_metrics(
    eval_rows: list[dict[str, Any]], predictions: np.ndarray, confidence: np.ndarray
) -> dict[str, Any]:
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    profiles: dict[str, list[int]] = {}
    for row, predicted, conf in zip(eval_rows, predictions, confidence):
        truth = CLASSES.index(row["cls"])
        confusion[truth, int(predicted)] += 1
        profiles.setdefault(row["profile"], []).append(int(truth == predicted))
        if not np.isfinite(conf):
            raise SystemExit("non-finite confidence logit in release evaluation")
    totals = confusion.sum(axis=1)
    recalls = np.diag(confusion) / np.maximum(totals, 1)
    profile_recalls = {key: sum(value) / len(value) for key, value in profiles.items()}
    answered = confidence > 0
    correct = np.fromiter(
        (CLASSES[int(pred)] == row["cls"] for row, pred in zip(eval_rows, predictions)),
        dtype=np.bool_,
    )
    return {
        "balanced_accuracy": float(recalls.mean()),
        "accuracy": float(np.diag(confusion).sum() / confusion.sum()),
        "min_profile_recall": float(min(profile_recalls.values())),
        "errors_total": int(confusion.sum() - np.diag(confusion).sum()),
        "per_class_recall": {
            label: float(recalls[index]) for index, label in enumerate(CLASSES)
        },
        "worst_profiles": sorted(profile_recalls.items(), key=lambda item: item[1])[:5],
        "confidence_head_diagnostic_only": {
            "answered_fraction_at_zero_logit": float(answered.mean()),
            "answered_accuracy_at_zero_logit": (
                float(correct[answered].mean()) if answered.any() else 0.0
            ),
        },
        "confusion": confusion.tolist(),
    }


def spread(results: dict[str, dict[str, dict[str, Any]]], key: str) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for dwell in DWELLS:
        values = np.asarray(
            [results[str(seed)][dwell][key] for seed in EVAL_SEEDS],
            dtype=np.float64,
        )
        summary[dwell] = {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
            "std": float(values.std(ddof=1)),
            "values": [float(value) for value in values],
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prototype-seed", type=int, default=EVAL_SEEDS[0])
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    args = parser.parse_args()

    revision, branch = require_clean_source()
    checkpoint = args.checkpoint.resolve()
    corpus = args.corpus.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise SystemExit(f"release output directory must be empty: {output}")

    mx, torch, torch_trainer, mlx_trainer = load_training_modules()
    checkpoint_evidence = verify_checkpoint(checkpoint, mlx_trainer)
    sidecar = checkpoint_evidence.pop("sidecar")
    manifest = json.loads((corpus / "manifest.json").read_text())
    if manifest.get("targetSampleRateHz") != TARGET_SAMPLE_RATE_HZ:
        raise SystemExit("DACS release corpus is not at the exact 20 Msps runtime rate")
    provenance = manifest.get("sourceProvenance", {})
    if not provenance or any(
        source.get("workingTreeClean") is not True for source in provenance.values()
    ):
        raise SystemExit("DACS release corpus does not have clean source provenance")

    train_by_class, eval_rows, all_train = split_manifest(manifest)
    # Qualified inventories: v2/v3 (34 profiles) and v4 (36 profiles, adds
    # fm-broadcast-mpx and am-voice). 96 eval rows per profile either way.
    if len(eval_rows) not in (3264, 3456) or any(
        len(train_by_class[label]) < PROTOTYPE_ROWS for label in CLASSES
    ):
        raise SystemExit("DACS release corpus split does not match the qualified inventory")

    if args.device == "mps" or (
        args.device == "auto" and torch.backends.mps.is_available()
    ):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    width = int(sidecar["config"]["width"])
    embed_dim = int(sidecar["config"]["embed_dim"])
    net, encoder = torch_encoder_from_mlx(
        checkpoint,
        mx,
        torch,
        torch_trainer,
        width=width,
        embed_dim=embed_dim,
    )
    model_path = output / MODEL_FILENAME
    onnx_inventory = export_onnx(encoder, torch, model_path)
    net = net.to(device).eval()

    _mid, _query_offsets, prototype_offsets, prototype_plan = (
        torch_trainer.build_eval_plan(
            args.prototype_seed,
            eval_rows,
            all_train,
            int(manifest["rowSamples"]),
            max(DWELLS.values()),
            offset_mode="random",
            subsample_mode="random",
        )
    )
    noisy = np.load(corpus / "noisy.npy", mmap_mode="r")
    prototypes: dict[str, list[list[float]]] = {}
    t0 = time.perf_counter()
    for dwell, length in DWELLS.items():
        per_class: list[list[float]] = []
        for label in CLASSES:
            row_ids = train_by_class[label][:PROTOTYPE_ROWS]
            embedding, _confidence = embed_rows(
                net,
                torch_trainer,
                torch,
                noisy,
                row_ids,
                prototype_offsets,
                length,
                device,
            )
            per_class.append(
                embedding.mean(axis=0, dtype=np.float64).astype(np.float32).tolist()
            )
        prototypes[dwell] = per_class
        print(f"fixed prototypes {dwell} complete", flush=True)

    prototype_payload = {
        "schema": PROTOTYPE_SCHEMA,
        "schema_version": 1,
        "classes": list(CLASSES),
        "sample_rate_hz": TARGET_SAMPLE_RATE_HZ,
        "dwell_samples": DWELLS,
        "embedding_dimension": embed_dim,
        "embedding_l2_norm": 4.0,
        "distance": "squared_euclidean",
        "logit_scale": float(mx.load(str(checkpoint))["params.log_scale"].item()),
        "prototype_rows_per_class": PROTOTYPE_ROWS,
        "prototype_seed": args.prototype_seed,
        "prototype_plan_sha256": prototype_plan["plan_sha256"],
        "prototypes_by_dwell": prototypes,
    }
    prototype_path = output / PROTOTYPE_FILENAME
    write_json(prototype_path, prototype_payload)

    eval_results: dict[str, dict[str, dict[str, Any]]] = {}
    eval_plans: dict[str, dict[str, Any]] = {}
    eval_row_ids = [int(row["row"]) for row in eval_rows]
    prototype_arrays = {
        dwell: np.asarray(value, dtype=np.float32) for dwell, value in prototypes.items()
    }
    for seed in EVAL_SEEDS:
        _mid, query_offsets, _unused_prototype_offsets, plan = (
            torch_trainer.build_eval_plan(
                seed,
                eval_rows,
                all_train,
                int(manifest["rowSamples"]),
                max(DWELLS.values()),
                offset_mode="random",
                subsample_mode="random",
            )
        )
        eval_plans[str(seed)] = plan
        eval_results[str(seed)] = {}
        for dwell, length in DWELLS.items():
            embedding, confidence = embed_rows(
                net,
                torch_trainer,
                torch,
                noisy,
                eval_row_ids,
                query_offsets,
                length,
                device,
            )
            distances = (
                (embedding[:, None, :] - prototype_arrays[dwell][None, :, :]) ** 2
            ).sum(axis=2)
            predictions = distances.argmin(axis=1)
            eval_results[str(seed)][dwell] = confusion_and_metrics(
                eval_rows, predictions, confidence
            )
        print(f"fixed-prototype validation seed {seed} complete", flush=True)

    summary = {
        "balanced_accuracy": spread(eval_results, "balanced_accuracy"),
        "min_profile_recall": spread(eval_results, "min_profile_recall"),
        "errors_total": spread(eval_results, "errors_total"),
    }
    gates = {
        "1ms": {"balanced_accuracy": 0.89, "min_profile_recall": 0.15},
        "2.5ms": {"balanced_accuracy": 0.94, "min_profile_recall": 0.50},
        "10ms": {"balanced_accuracy": 0.99, "min_profile_recall": 0.94},
    }
    failures = []
    for dwell, gate in gates.items():
        for metric, floor in gate.items():
            measured = summary[metric][dwell]["mean"]
            if measured < floor:
                failures.append(
                    f"{dwell} {metric} {measured:.6f} is below {floor:.6f}"
                )
    if failures:
        raise SystemExit("DACS release validation failed:\n- " + "\n- ".join(failures))

    elapsed_seconds = time.perf_counter() - t0
    validation_payload = {
        "schema": VALIDATION_SCHEMA,
        "schema_version": 1,
        "runtime_numeric_path": "torch-fp32-reference-for-onnx-fp32-export",
        "device": str(device),
        "fixed_prototype_seed": args.prototype_seed,
        "query_offset_seeds": list(EVAL_SEEDS),
        "query_rows_per_seed": len(eval_rows),
        "eval_plans": eval_plans,
        "gates": gates,
        "summary_over_query_seeds": summary,
        "results": eval_results,
    }
    validation_path = output / VALIDATION_FILENAME
    write_json(validation_path, validation_payload)

    runtime_constraints = {
        "input": "contiguous complex baseband I/Q",
        "sample_rate_hz": TARGET_SAMPLE_RATE_HZ,
        "resampling_in_runtime": False,
        "dwell_selection": "largest_supported_prefix_not_exceeding_capture_length",
        "supported_dwell_samples": list(DWELLS.values()),
        "closed_set_only": True,
        "released_open_set_gate_required_ahead_of_dacs": True,
        "confidence_head_decision_authority": False,
        "training_decoder_exported": False,
    }
    assets = {
        "encoder": descriptor(model_path, "onnx.ModelProto.opset18", "dacs_encoder"),
        "prototypes": descriptor(
            prototype_path, PROTOTYPE_SCHEMA, "fixed_class_prototypes"
        ),
        "validation": descriptor(
            validation_path, VALIDATION_SCHEMA, "release_evidence_not_runtime"
        ),
    }
    package_manifest = {
        "schema": PACKAGE_SCHEMA,
        "schema_version": 1,
        "status": "release",
        "development_only": False,
        "classes": list(CLASSES),
        "architecture": {
            "name": "DACS",
            "model_line": "v7",
            "width": width,
            "embedding_dimension": embed_dim,
            "exported_initializer_values": onnx_inventory["initializer_values"],
            "onnx_inventory": onnx_inventory,
            "preprocessing": {
                "rms_normalization_floor": 1e-9,
                "nfft": 64,
                "hop": 32,
                "periodic_hann": True,
                "bins": 33,
                "channels": ["real", "imaginary", "log1p_magnitude"],
            },
        },
        "runtime_constraints": runtime_constraints,
        "training_provenance": {
            "corpus_source_provenance": provenance,
            "corpus_manifest_sha256": sha256_file(corpus / "manifest.json"),
            "checkpoint_episode": int(sidecar["episode"]),
            "checkpoint_config": {
                key: sidecar["config"][key]
                for key in (
                    "seed",
                    "dtype",
                    "episodes",
                    "width",
                    "embed_dim",
                    "k_shot",
                    "q_query",
                )
            },
            **checkpoint_evidence,
        },
        "export_provenance": {
            "atom_classifier_revision": revision,
            "atom_classifier_branch": branch,
            "working_tree_clean": True,
            "torch_version": torch.__version__,
            "mlx_version": mx.__version__,
            "onnx_opset": 18,
        },
        "validation_summary": summary,
        "assets": assets,
    }
    manifest_path = output / MANIFEST_FILENAME
    write_json(manifest_path, package_manifest)
    print(
        f"release package {output} manifest_sha256={sha256_file(manifest_path)} "
        f"validation_seconds={elapsed_seconds:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
