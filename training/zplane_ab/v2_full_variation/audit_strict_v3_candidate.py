"""Assemble and audit the strict time-domain v3 closed/invariance candidate.

The script makes no open-set claim and cannot write below a release/sealed
directory.  It rebuilds only exposed train/enrollment/selection tensors, fits
the predeclared centered-fusion centers on training rows, fits prototypes on
enrollment rows, and evaluates:

* full immutable development-selection closed/clean/high-SNR performance;
* exact-prefix length behavior on the release-compatible N=4096 occupied
  population; and
* the metadata-safe physical sample-scale population.

The resulting assets are development evidence and inputs to the later open-set
and runtime assembler.  They are not release evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
TRAINING = HERE.parent.parent
REPO = TRAINING.parent
for path in (TRAINING, HERE.parent, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import preprocess as production_preprocess  # noqa: E402
from assemble_invariant_candidate import (  # noqa: E402
    FEWSHOT_K,
    FEWSHOT_SEED,
    FEWSHOT_TRIALS,
    _calibrate_temperature,
    _closed_report,
    _fewshot_report,
    _git_head,
    _tensor_state,
)
from evaluate_invariant_release_suite import GATE_FLOORS  # noqa: E402
from strict_v3_candidate_common import (  # noqa: E402
    ALPHA_COMPLEX,
    ALPHA_REAL,
    WEIGHT_REAL,
    fit_fusion_context,
    load_branch_artifact,
    prepare_strict_data,
    release_compatible_occupancy,
    sha256,
)
from train import nearest  # noqa: E402
from v3_scale import run_time_domain_dev as time_domain_dev  # noqa: E402


DEFAULT_ROOT = HERE / "artifacts" / "invariant_patch" / "v3_scale"
DEFAULT_REAL = DEFAULT_ROOT / "timecorr_real_4000_seed20260730_v3a"
DEFAULT_COMPLEX = DEFAULT_ROOT / "timecorr_complex_4000_seed20260730_run1"
DEFAULT_OUTPUT = DEFAULT_ROOT / "strict_v3_fusion_seed20260730_closed_audit"
LENGTHS = (4096, 8192, 16384)
SCALE_FACTORS = (0.5, 0.75, 1.0, 1.5, 2.0)
INCUMBENT = (
    HERE
    / "artifacts"
    / "bounded_dev"
    / "circularv2_baselines_seed20260728"
    / "dev_metrics.json"
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
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


def _write_json_exclusive(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(
                _jsonable(payload),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _development_output(path: Path) -> Path:
    output = Path(path).expanduser().resolve()
    if "releases" in output.parts or any(
        "sealed" in part.lower() for part in output.parts
    ):
        raise ValueError("closed audit refuses release/sealed output paths")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    return output


def _snr(manifest: Mapping[str, Any], indices: np.ndarray) -> np.ndarray:
    value = np.asarray(
        [
            float(manifest["items"][int(index)]["snrDb"])
            for index in np.asarray(indices, dtype=np.int64)
        ],
        dtype=np.float64,
    )
    if not np.isfinite(value).all():
        raise ValueError("manifest contains non-finite snrDb")
    return value


def _embed_captures(
    fusion: torch.nn.Module,
    captures: Sequence[np.ndarray],
    data: Mapping[str, Any],
    device: torch.device,
) -> np.ndarray:
    packed, raw_features, _contexts = time_domain_dev._preprocess_rows(
        captures,
        patch_length=fusion.real_branch.cfg.patch_length,
        patch_count=fusion.real_branch.cfg.patch_count,
        target_frac=0.5,
    )
    features = (
        (raw_features - np.asarray(data["fmean"]))
        / np.asarray(data["fstd"])
    ).astype(np.float32)
    from assemble_invariant_candidate import _embed_all

    return _embed_all(fusion, packed, features, device)


def _length_audit(
    fusion: torch.nn.Module,
    prototypes: np.ndarray,
    data: Mapping[str, Any],
    manifest: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    all_indices = np.asarray(data["va_idx"], dtype=np.int64)
    eligible, eligibility = release_compatible_occupancy(
        manifest,
        all_indices,
        list(data["classes"]),
    )
    positions = np.flatnonzero(np.isin(all_indices, eligible))
    if not np.array_equal(all_indices[positions], eligible):
        raise AssertionError("occupancy subset no longer preserves selection order")
    raw = time_domain_dev._raw_memmap(dict(manifest))
    base = [
        time_domain_dev._complex_row(raw, int(index))
        for index in eligible
    ]
    del raw
    labels, impaired, snr_db = time_domain_dev._metadata_arrays(
        dict(manifest),
        eligible,
        data["classes"],
    )
    rows = []
    embeddings: dict[int, np.ndarray] = {}
    predictions: dict[int, np.ndarray] = {}
    for length in LENGTHS:
        value = (
            np.asarray(data["xva"])[positions]
            if length == max(LENGTHS)
            else None
        )
        if value is None:
            current = _embed_captures(
                fusion,
                [capture[:length] for capture in base],
                data,
                device,
            )
        else:
            from assemble_invariant_candidate import _embed_all

            current = _embed_all(
                fusion,
                value,
                np.asarray(data["fva"])[positions],
                device,
            )
        report, prediction = time_domain_dev._classification_report(
            current,
            labels,
            prototypes,
            data["classes"],
            impaired=impaired,
            snr_db=snr_db,
        )
        embeddings[length] = current
        predictions[length] = prediction
        rows.append({"length": length, **report})
    reference_length = max(LENGTHS)
    for row in rows:
        length = int(row["length"])
        row["paired_to_matched"] = time_domain_dev._paired_report(
            embeddings[length],
            embeddings[reference_length],
            predictions[length],
            predictions[reference_length],
            labels,
            data["classes"],
        )
    return {
        "definition": (
            "exact observed prefixes on the historical development rows that "
            "satisfy the predeclared release N=4096 clean-occupancy rule"
        ),
        "development_only": True,
        "eligibility": eligibility,
        "rows": rows,
        "worst_accuracy": min(float(row["accuracy"]) for row in rows),
        "worst_balanced_accuracy": min(
            float(row["balanced_accuracy"]) for row in rows
        ),
        "worst_clean_accuracy": min(
            float(row["clean_accuracy"]) for row in rows
        ),
        "worst_high_snr_accuracy": min(
            float(row["high_snr_accuracy"]) for row in rows
        ),
        "maximum_mean_pairwise_cosine": max(
            float(row["mean_pairwise_cosine"]) for row in rows
        ),
        "worst_prediction_agreement_to_matched": min(
            float(row["paired_to_matched"]["prediction_agreement"])
            for row in rows
        ),
        "worst_embedding_cosine_to_matched": min(
            float(row["paired_to_matched"]["embedding_cosine_mean"])
            for row in rows
        ),
        "n32768_status": (
            "not synthesized from the historical N=16384 development corpus; "
            "the sealed protocol must generate N=32768 once"
        ),
    }


def _scale_audit(
    fusion: torch.nn.Module,
    prototypes: np.ndarray,
    data: Mapping[str, Any],
    manifest: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    indices, eligibility = time_domain_dev._scale_eligible(
        dict(manifest),
        np.asarray(data["va_idx"], dtype=np.int64),
        data["classes"],
    )
    labels, impaired, snr_db = time_domain_dev._metadata_arrays(
        dict(manifest),
        indices,
        data["classes"],
    )
    raw = time_domain_dev._raw_memmap(dict(manifest))
    base = [
        time_domain_dev._complex_row(raw, int(index))
        for index in indices
    ]
    del raw
    rows = []
    embeddings: dict[float, np.ndarray] = {}
    predictions: dict[float, np.ndarray] = {}
    base_length = int(manifest["sampleCount"])
    for factor in SCALE_FACTORS:
        new_length = max(64, int(round(base_length / factor)))
        captures = (
            base
            if factor == 1.0
            else [
                production_preprocess.lin_resample(capture, new_length)
                for capture in base
            ]
        )
        current = _embed_captures(fusion, captures, data, device)
        report, prediction = time_domain_dev._classification_report(
            current,
            labels,
            prototypes,
            data["classes"],
            impaired=impaired,
            snr_db=snr_db,
        )
        embeddings[factor] = current
        predictions[factor] = prediction
        rows.append(
            {
                "factor": factor,
                "effective_factor": (base_length - 1) / (new_length - 1),
                "raw_length": new_length,
                **report,
            }
        )
    for row in rows:
        factor = float(row["factor"])
        row["paired_to_factor1"] = time_domain_dev._paired_report(
            embeddings[factor],
            embeddings[1.0],
            predictions[factor],
            predictions[1.0],
            labels,
            data["classes"],
        )
    return {
        "definition": (
            "normalized frequencies multiplied by s through deterministic "
            "linear resampling to N/s; common metadata-safe no-alias subset"
        ),
        "development_only": True,
        "eligible_rows": int(len(indices)),
        **eligibility,
        "rows": rows,
        "worst_balanced_accuracy": min(
            float(row["balanced_accuracy"]) for row in rows
        ),
        "maximum_mean_pairwise_cosine": max(
            float(row["mean_pairwise_cosine"]) for row in rows
        ),
        "worst_prediction_agreement_to_factor1": min(
            float(row["paired_to_factor1"]["prediction_agreement"])
            for row in rows
        ),
        "worst_embedding_cosine_to_factor1": min(
            float(row["paired_to_factor1"]["embedding_cosine_mean"])
            for row in rows
        ),
    }


def _gate(value: float, threshold: float, comparison: str = "min") -> dict[str, Any]:
    if comparison not in {"min", "max"}:
        raise ValueError("comparison must be min or max")
    passes = value >= threshold if comparison == "min" else value <= threshold
    return {
        "value": float(value),
        "threshold": float(threshold),
        "comparison": comparison,
        "passes": bool(passes),
    }


def _gates(
    length: Mapping[str, Any],
    scale: Mapping[str, Any],
    closed: Mapping[str, Any],
    fewshot: Mapping[str, Any],
) -> dict[str, Any]:
    values = {
        "closed_full_balanced": _gate(
            float(closed["balanced_accuracy"]),
            GATE_FLOORS["closed_fine"],
        ),
        "closed_fine_worst_length": _gate(
            float(length["worst_accuracy"]),
            GATE_FLOORS["closed_fine"],
        ),
        "closed_family_worst_length": _gate(
            float(length["worst_accuracy"]),
            GATE_FLOORS["closed_family"],
        ),
        "closed_clean_worst_length": _gate(
            float(length["worst_clean_accuracy"]),
            GATE_FLOORS["closed_clean"],
        ),
        "closed_high_snr_worst_length": _gate(
            float(length["worst_high_snr_accuracy"]),
            GATE_FLOORS["closed_high_snr"],
        ),
        "fewshot_simultaneous_mean": _gate(
            float(
                fewshot["simultaneous_7way"]["balanced_accuracy_mean"]
            ),
            GATE_FLOORS["five_shot"],
        ),
        "length_worst_balanced_accuracy": _gate(
            float(length["worst_balanced_accuracy"]),
            GATE_FLOORS["invariant_worst_balanced_accuracy"],
        ),
        "length_max_mean_pairwise_cosine": _gate(
            float(length["maximum_mean_pairwise_cosine"]),
            GATE_FLOORS["invariant_max_mean_pairwise_cosine"],
            "max",
        ),
        "length_worst_prediction_agreement": _gate(
            float(length["worst_prediction_agreement_to_matched"]),
            GATE_FLOORS["length_pair_prediction_agreement"],
        ),
        "length_worst_embedding_cosine": _gate(
            float(length["worst_embedding_cosine_to_matched"]),
            GATE_FLOORS["length_pair_embedding_cosine"],
        ),
        "scale_worst_balanced_accuracy": _gate(
            float(scale["worst_balanced_accuracy"]),
            GATE_FLOORS["invariant_worst_balanced_accuracy"],
        ),
        "scale_max_mean_pairwise_cosine": _gate(
            float(scale["maximum_mean_pairwise_cosine"]),
            GATE_FLOORS["invariant_max_mean_pairwise_cosine"],
            "max",
        ),
        "scale_worst_prediction_agreement": _gate(
            float(scale["worst_prediction_agreement_to_factor1"]),
            GATE_FLOORS["scale_pair_prediction_agreement"],
        ),
        "scale_worst_embedding_cosine": _gate(
            float(scale["worst_embedding_cosine_to_factor1"]),
            GATE_FLOORS["scale_pair_embedding_cosine"],
        ),
    }
    return {
        "values": values,
        "all_closed_invariance_gates_pass": all(
            item["passes"] for item in values.values()
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = _development_output(args.output_dir)
    real = load_branch_artifact(Path(args.real_dir), "real")
    complex_artifact = load_branch_artifact(
        Path(args.complex_dir),
        "complex",
    )
    model_seed = int(real.report["data_audit"]["model_seed"])
    if int(complex_artifact.report["data_audit"]["model_seed"]) != model_seed:
        raise ValueError("real and complex model seeds differ")
    data, manifest = prepare_strict_data(
        model_seed=model_seed,
        patch_length=real.config.patch_length,
        patch_count=real.config.patch_count,
        target_frac=0.5,
    )
    device = time_domain_dev.runner.resolve_device(args.device)
    context = fit_fusion_context(real, complex_artifact, data, device)
    snr_db = _snr(manifest, np.asarray(data["va_idx"]))
    closed, prediction = _closed_report(
        context.embeddings["fusion"]["va"],
        context.prototypes,
        data,
        snr_db,
    )
    temperature = _calibrate_temperature(
        context.embeddings["fusion"]["en"],
        np.asarray(data["yen"]),
        int(data["n_classes"]),
    )
    fewshot = _fewshot_report(
        context.embeddings["fusion"]["en"],
        context.embeddings["fusion"]["va"],
        data,
        context.prototypes,
        closed["per_class"],
        seed=FEWSHOT_SEED,
        trials=FEWSHOT_TRIALS,
        k=FEWSHOT_K,
    )
    length = _length_audit(
        context.fusion,
        context.prototypes,
        data,
        manifest,
        device,
    )
    scale = _scale_audit(
        context.fusion,
        context.prototypes,
        data,
        manifest,
        device,
    )
    gates = _gates(length, scale, closed, fewshot)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "fusion_state_dict.pt"
    prototype_path = output / "fused_prototypes.npy"
    feature_path = output / "feature_standardization.npz"
    assets_path = output / "closed_assets.pt"
    torch.save(_tensor_state(context.fusion), state_path)
    np.save(prototype_path, context.prototypes)
    np.savez_compressed(
        feature_path,
        mean=np.asarray(data["fmean"], dtype=np.float32),
        std=np.asarray(data["fstd"], dtype=np.float32),
    )
    torch.save(
        {
            "schema": 1,
            "kind": "strict_time_domain_v3_closed_assets",
            "frontend": {
                "version": "invariant-patch-time-domain-v1",
                "uses_frequency_transform": False,
            },
            "real_config": real.config.__dict__,
            "complex_config": complex_artifact.config.__dict__,
            "real_state_dict": _tensor_state(context.real),
            "complex_state_dict": _tensor_state(context.complex),
            "real_center": torch.from_numpy(context.centers["real"]),
            "complex_center": torch.from_numpy(context.centers["complex"]),
            "alpha_real": ALPHA_REAL,
            "alpha_complex": ALPHA_COMPLEX,
            "weight_real": WEIGHT_REAL,
            "feature_mean": torch.from_numpy(
                np.asarray(data["fmean"], dtype=np.float32)
            ),
            "feature_std": torch.from_numpy(
                np.asarray(data["fstd"], dtype=np.float32)
            ),
            "classes": list(data["classes"]),
            "prototypes": torch.from_numpy(context.prototypes),
            "temperature": float(temperature["temperature"]),
            "development_only": True,
            "consumed_test_rows_used": 0,
        },
        assets_path,
    )
    artifact_hashes = {
        path.name: sha256(path)
        for path in (state_path, prototype_path, feature_path, assets_path)
    }
    with INCUMBENT.open(encoding="utf-8") as handle:
        incumbent = json.load(handle)
    incumbent_closed = incumbent["metrics"]["cnn"]["closed_selection"]
    report = {
        "status": (
            "closed_invariance_development_pass"
            if gates["all_closed_invariance_gates_pass"]
            else "closed_invariance_development_fail"
        ),
        "development_only": True,
        "release_evidence": False,
        "consumed_test_rows_used": 0,
        "sealed_release_data_used": 0,
        "open_set_status": "not_evaluated_by_this_closed-only_audit",
        "closed_selection": closed,
        "fewshot_selection": fewshot,
        "release_compatible_length": length,
        "physical_scale": scale,
        "gates": gates,
        "incumbent_comparison": {
            "path": INCUMBENT,
            "sha256": sha256(INCUMBENT),
            "incumbent_accuracy": incumbent_closed["accuracy"],
            "candidate_accuracy": closed["accuracy"],
            "absolute_accuracy_gain": (
                float(closed["accuracy"])
                - float(incumbent_closed["accuracy"])
            ),
        },
        "fusion": {
            "alpha_real": ALPHA_REAL,
            "alpha_complex": ALPHA_COMPLEX,
            "weight_real": WEIGHT_REAL,
            "center_fit_population": "exposed training rows only",
            "prototype_fit_population": "disjoint enrollment rows only",
            "temperature_fit_population": "disjoint enrollment rows only",
        },
        "artifacts": artifact_hashes,
        "provenance": {
            "real_source": {
                "directory": real.directory,
                "report_sha256": real.report_sha256,
                "state_sha256": real.state_sha256,
            },
            "complex_source": {
                "directory": complex_artifact.directory,
                "report_sha256": complex_artifact.report_sha256,
                "state_sha256": complex_artifact.state_sha256,
            },
            "source_sha256": {
                "audit_strict_v3_candidate.py": sha256(
                    Path(__file__).resolve()
                ),
                "strict_v3_candidate_common.py": sha256(
                    HERE / "strict_v3_candidate_common.py"
                ),
                "time_domain_geometry.py": sha256(
                    TRAINING / "time_domain_geometry.py"
                ),
                "time_domain_invariant_patch_preprocess.py": sha256(
                    TRAINING / "time_domain_invariant_patch_preprocess.py"
                ),
                "invariant_patch_preprocess.py": sha256(
                    TRAINING / "invariant_patch_preprocess.py"
                ),
            },
            "data_audit": data["data_audit"],
            "git_head": _git_head(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": device,
            "model_seed": model_seed,
            "prediction_sha256": hashlib.sha256(
                np.asarray(prediction, dtype="<i8").tobytes()
            ).hexdigest(),
        },
    }
    _write_json_exclusive(output / "closed_candidate_report.json", report)
    _write_json_exclusive(
        output / "DEVELOPMENT_ONLY.json",
        {
            "development_only": True,
            "release_evidence": False,
            "consumed_test_rows_used": 0,
            "sealed_release_data_used": 0,
            "remaining": [
                "strict open-set policy replication",
                "complete Python/TypeScript/browser model parity",
                "one untouched sealed release evaluation",
                "real SDR field validation",
            ],
        },
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--real-dir", default=str(DEFAULT_REAL))
    parser.add_argument("--complex-dir", default=str(DEFAULT_COMPLEX))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="cpu")
    return parser


if __name__ == "__main__":
    result = run(build_parser().parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "closed_accuracy": result["closed_selection"]["accuracy"],
                "closed_invariance_gates_pass": result["gates"][
                    "all_closed_invariance_gates_pass"
                ],
            },
            indent=2,
        )
    )
