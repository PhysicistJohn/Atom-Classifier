"""Export a corrected TransferUNet to a versioned, paired-real staging artifact.

This exporter never writes ``src/embedding/assets``.  Its output is deliberately
marked staging-only until model-quality, calibration, preprocessing-parity, and
product-latency gates pass.

Example
-------
    .venv-training/bin/python \
      training/zplane_ab/v2_full_variation/export_transfer_unet.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
TRAINING = HERE.parents[1]
REPO = TRAINING.parent
for candidate in (TRAINING, HERE):
    value = str(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)

from paired_real_inference import make_paired_real_model  # noqa: E402
import preprocess as pp  # noqa: E402
from unet_transfer import TransferUNet  # noqa: E402


SCHEMA_ID = "atomos.transfer-unet.paired-real"
SCHEMA_VERSION = 1
ONNX_OPSET = 17
DEFAULT_CHECKPOINT = (
    HERE
    / "artifacts"
    / "ground_state"
    / "corrected_resampled_cropoff_seed20260727"
    / "inference_bundle.pt"
)
DEFAULT_OUTPUT = (
    HERE
    / "artifacts"
    / "staging"
    / "transfer_unet_paired_real_v1"
)
ONNX_NAME = "transfer-unet-paired-real-v1.onnx"
FIXTURE_NAME = "paired-real-parity-v1.json"
PREPROCESS_SOURCE = TRAINING / "preprocess.py"
ORT_PARITY_LIMIT = 2e-5
ORT_WARMUP_RUNS = 5
ORT_TIMED_RUNS = 20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _require_empty_output(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    contents = list(path.iterdir())
    if contents:
        names = ", ".join(sorted(item.name for item in contents[:5]))
        raise FileExistsError(
            f"refusing to overwrite non-empty staging directory {path} "
            f"(contains {names})"
        )


def _load_bundle(path: Path) -> dict[str, Any]:
    # The input is a trusted local artifact emitted by run_corrected_unet.py.
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "schema_version",
        "model",
        "architecture",
        "state_dict",
        "classes",
        "prototypes",
        "feature_mean",
        "feature_std",
        "open_set_threshold",
        "preprocessing",
        "target_policy",
    }
    missing = sorted(required.difference(bundle))
    if missing:
        raise ValueError(f"inference bundle is missing keys: {missing}")
    if bundle["model"] != "TransferUNet":
        raise ValueError(f"expected TransferUNet bundle, got {bundle['model']!r}")
    return bundle


def _fixture_input(
    input_length: int,
    n_features: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic normalized I/Q and standardized feature probe."""
    sample = torch.arange(input_length, dtype=torch.float32)
    phase = (
        2.0
        * torch.pi
        * (0.071 * sample + 0.000021 * sample.square())
    )
    envelope = 0.72 + 0.21 * torch.cos(2.0 * torch.pi * 0.009 * sample)
    real = envelope * torch.cos(phase)
    imaginary = envelope * torch.sin(phase)
    rms = torch.sqrt((real.square() + imaginary.square()).mean())
    iq = torch.stack([real / rms, imaginary / rms], dim=0).unsqueeze(0)
    features = torch.linspace(-0.75, 0.75, n_features).unsqueeze(0)
    return iq, features


@torch.no_grad()
def _reference_outputs(
    bundle: dict[str, Any],
    iq: torch.Tensor,
    features: torch.Tensor,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, float],
    torch.nn.Module,
]:
    architecture = dict(bundle["architecture"])
    reference = TransferUNet(**architecture)
    reference.load_state_dict(bundle["state_dict"], strict=True)
    reference.eval()
    embedding = reference(iq, features)
    expected = {
        "embedding": embedding.detach().cpu(),
        "bottleneck": reference.last_bottleneck_pool.detach().cpu(),
        "denoised_real": reference.last_recon.real.detach().cpu(),
        "denoised_imaginary": reference.last_recon.imag.detach().cpu(),
    }

    paired = make_paired_real_model(architecture, bundle["state_dict"])
    actual_tuple = paired(iq, features)
    names = (
        "embedding",
        "bottleneck",
        "denoised_real",
        "denoised_imaginary",
    )
    errors = {
        name: float((expected[name] - actual).abs().max())
        for name, actual in zip(names, actual_tuple)
    }
    if max(errors.values()) > 2e-6:
        raise AssertionError(f"paired-real reference parity failed: {errors}")
    return expected, errors, paired


def _assert_real_trace(
    model: torch.nn.Module,
    iq: torch.Tensor,
    features: torch.Tensor,
) -> None:
    traced = torch.jit.trace(model, (iq, features), strict=True)
    graph = str(traced.inlined_graph)
    forbidden = ("aten::complex", "aten::real", "aten::imag", "ComplexFloat")
    present = [token for token in forbidden if token in graph]
    if present:
        raise AssertionError(f"paired-real graph contains complex operators: {present}")


def _export_onnx(
    model: torch.nn.Module,
    iq: torch.Tensor,
    features: torch.Tensor,
    expected: dict[str, torch.Tensor],
    destination: Path,
) -> dict[str, Any]:
    try:
        import onnx
    except ImportError as failure:
        raise RuntimeError(
            "the staging ONNX export requires the training environment's onnx package"
        ) from failure

    # This artifact deliberately uses the mature TorchScript exporter.  The
    # newer dynamo exporter has a different dependency and operator surface;
    # changing exporters is a release-contract change, not a warning-driven
    # refactor.  Keep warnings-as-errors useful by suppressing only PyTorch's
    # legacy-export deprecations at this explicit compatibility boundary.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        torch.onnx.export(
            model,
            (iq, features),
            destination,
            input_names=["iq", "features"],
            output_names=[
                "embedding",
                "bottleneck",
                "denoised_real",
                "denoised_imaginary",
            ],
            dynamic_axes={
                "iq": {0: "batch"},
                "features": {0: "batch"},
                "embedding": {0: "batch"},
                "bottleneck": {0: "batch"},
                "denoised_real": {0: "batch"},
                "denoised_imaginary": {0: "batch"},
            },
            opset_version=ONNX_OPSET,
            do_constant_folding=True,
        )
    graph = onnx.load(str(destination))
    onnx.checker.check_model(graph)
    value_infos = [*graph.graph.input, *graph.graph.output]
    non_float = [
        value.name
        for value in value_infos
        if value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT
    ]
    if non_float:
        raise AssertionError(f"ONNX public tensors are not all float32: {non_float}")
    return {
        "checker": "passed",
        "onnx_version": onnx.__version__,
        "opset": ONNX_OPSET,
        "node_count": len(graph.graph.node),
        "initializer_count": len(graph.graph.initializer),
        "public_tensor_dtype": "float32",
        "execution": _execute_onnx_runtime(
            destination,
            iq,
            features,
            expected,
        ),
    }


def _execute_onnx_runtime(
    model_path: Path,
    iq: torch.Tensor,
    features: torch.Tensor,
    expected: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Optionally execute the checked graph with ONNX Runtime CPU.

    ONNX structural validation is mandatory and occurs before this function.
    Runtime execution is optional only because the lightweight training
    environment may omit ``onnxruntime``.  Absence is reported as ``not_run``;
    it is never represented as a parity pass.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return {
            "status": "not_run",
            "reason": (
                "onnxruntime is not installed; ONNX checker passed, but "
                "execution parity and latency were not measured"
            ),
            "onnxruntime_version": None,
            "execution_provider": None,
            "max_abs_error": None,
            "latency_ms": None,
        }

    provider = "CPUExecutionProvider"
    available = list(ort.get_available_providers())
    if provider not in available:
        raise RuntimeError(
            f"onnxruntime {ort.__version__} lacks required {provider}; "
            f"available={available}"
        )
    session = ort.InferenceSession(str(model_path), providers=[provider])
    active = list(session.get_providers())
    if not active or active[0] != provider:
        raise AssertionError(
            f"requested {provider}, but ONNX Runtime activated {active}"
        )

    output_names = [
        "embedding",
        "bottleneck",
        "denoised_real",
        "denoised_imaginary",
    ]
    feeds = {
        "iq": iq.detach().cpu().numpy(),
        "features": features.detach().cpu().numpy(),
    }
    for _ in range(ORT_WARMUP_RUNS):
        session.run(output_names, feeds)

    durations_ms = []
    actual = None
    for _ in range(ORT_TIMED_RUNS):
        started = time.perf_counter_ns()
        actual = session.run(output_names, feeds)
        durations_ms.append((time.perf_counter_ns() - started) / 1e6)
    assert actual is not None

    errors = {}
    for name, value in zip(output_names, actual):
        actual_tensor = torch.from_numpy(value).to(expected[name].dtype)
        if actual_tensor.shape != expected[name].shape:
            raise AssertionError(
                f"ONNX Runtime output {name!r} shape {actual_tensor.shape} "
                f"does not match reference {expected[name].shape}"
            )
        if not bool(torch.isfinite(actual_tensor).all()):
            raise AssertionError(f"ONNX Runtime output {name!r} is non-finite")
        errors[name] = float((expected[name] - actual_tensor).abs().max())
    worst = max(errors.values())
    if worst > ORT_PARITY_LIMIT:
        raise AssertionError(
            f"ONNX Runtime parity failed (limit={ORT_PARITY_LIMIT}): {errors}"
        )

    ordered = sorted(durations_ms)
    p95_index = min(
        len(ordered) - 1,
        max(0, int(0.95 * len(ordered) + 0.999999) - 1),
    )
    return {
        "status": "passed",
        "onnxruntime_version": ort.__version__,
        "execution_provider": provider,
        "active_providers": active,
        "available_providers": available,
        "max_abs_error": errors,
        "parity_limit": ORT_PARITY_LIMIT,
        "latency_ms": {
            "warmup_runs": ORT_WARMUP_RUNS,
            "timed_runs": ORT_TIMED_RUNS,
            "min": min(durations_ms),
            "mean": sum(durations_ms) / len(durations_ms),
            "median": 0.5
            * (
                ordered[(len(ordered) - 1) // 2]
                + ordered[len(ordered) // 2]
            ),
            "p95": ordered[p95_index],
            "scope": "batch=1, all four graph outputs, local CPU wall time",
        },
    }


def _source_run_metadata(checkpoint: Path) -> dict[str, Any]:
    source_dir = checkpoint.parent
    output: dict[str, Any] = {}
    for name in ("run_manifest.json", "run_started.json", "COMPLETE.json"):
        path = source_dir / name
        if not path.exists():
            continue
        output[name] = {
            "sha256": _sha256(path),
            "payload": json.loads(path.read_text()),
        }
    return output


def _preprocess_provenance(
    bundle: dict[str, Any],
    source_runs: dict[str, Any],
) -> dict[str, Any]:
    """Separate the checkpoint's authoring source from the export environment.

    The legacy bundle names ``training/preprocess.py`` but does not embed either
    its source digest or its numeric/algorithmic contract.  Source-run manifests
    do carry the authoring digest.  Current constants may only be promoted into
    the checkpoint contract when that digest exactly matches the file the
    exporter imported.
    """
    checkpoint_sha = None
    run_manifest = source_runs.get("run_manifest.json", {}).get("payload", {})
    source_hashes = run_manifest.get("environment", {}).get("source_sha256", {})
    candidate = source_hashes.get("training/preprocess.py")
    if isinstance(candidate, str) and candidate:
        checkpoint_sha = candidate

    current_sha = _sha256(PREPROCESS_SOURCE)
    current_parameters = {
        "estimator_version": pp.PREPROCESS_VERSION,
        "l_out": int(pp.L_OUT),
        "target_frac": float(pp.TARGET_FRAC),
        "nfft": int(pp.NFFT),
        "energy_edge": float(pp.ENERGY_EDGE),
        "noise_floor_scale": float(pp.NOISE_FLOOR_SCALE),
        "smooth": int(pp.SMOOTH),
        "full_band_bw": float(pp.FULL_BAND_BW),
        "white_flatness_deficit": float(pp.WHITE_FLATNESS_DEFICIT),
        "guard_floor_scale": float(pp.GUARD_FLOOR_SCALE),
        "min_excess_fraction": float(pp.MIN_EXCESS_FRACTION),
        "hybrid_legacy_wide_bw": float(pp.HYBRID_LEGACY_WIDE_BW),
        "hybrid_circular_broad_bw": float(pp.HYBRID_CIRCULAR_BROAD_BW),
        "hybrid_seam_tolerance_bins": float(pp.HYBRID_SEAM_TOLERANCE_BINS),
    }
    compatible = checkpoint_sha is not None and checkpoint_sha == current_sha
    if checkpoint_sha is None:
        reason = (
            "checkpoint authoring preprocess SHA is absent; current exporter "
            "parameters cannot be attributed to the checkpoint"
        )
    elif not compatible:
        reason = (
            "checkpoint authoring preprocess SHA differs from the current "
            "exporter preprocess SHA; current parameters are diagnostic only"
        )
    else:
        reason = "checkpoint and exporter preprocessing sources match exactly"

    return {
        "bundle_declaration": dict(bundle["preprocessing"]),
        "checkpoint_authoring": {
            "module": "training/preprocess.py",
            "source_sha256": checkpoint_sha,
            # These constants are known to describe the checkpoint only when
            # the complete source file matches, because algorithm semantics are
            # not captured by six numeric values.
            "parameters": current_parameters if compatible else None,
        },
        "exporter_environment": {
            "module": str(PREPROCESS_SOURCE.relative_to(REPO)),
            "source_sha256": current_sha,
            "parameters": current_parameters,
        },
        "compatible_with_checkpoint": compatible,
        "compatibility_reason": reason,
    }


def export_staging(
    checkpoint: Path,
    output_dir: Path,
    *,
    raw_capture_length: int = 16384,
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    output_dir = output_dir.resolve()
    if raw_capture_length <= 0:
        raise ValueError("raw_capture_length must be positive")
    _require_empty_output(output_dir)
    bundle = _load_bundle(checkpoint)

    architecture = dict(bundle["architecture"])
    input_length = int(bundle["preprocessing"]["input_length"])
    n_features = int(architecture["n_features"])
    iq, features = _fixture_input(input_length, n_features)
    expected, parity_errors, paired = _reference_outputs(bundle, iq, features)
    _assert_real_trace(paired, iq, features)

    onnx_path = output_dir / ONNX_NAME
    onnx_report = _export_onnx(paired, iq, features, expected, onnx_path)

    fixture_path = output_dir / FIXTURE_NAME
    _write_json(
        fixture_path,
        {
            "schema": f"{SCHEMA_ID}.parity",
            "schema_version": SCHEMA_VERSION,
            "input": {
                "iq_shape": list(iq.shape),
                "iq": iq.flatten().tolist(),
                "features_shape": list(features.shape),
                "features": features.flatten().tolist(),
            },
            "expected": {
                name: {
                    "shape": list(tensor.shape),
                    "values": tensor.flatten().tolist(),
                }
                for name, tensor in expected.items()
            },
            "paired_real_max_abs_error_vs_transfer_unet": parity_errors,
        },
    )

    source_runs = _source_run_metadata(checkpoint)
    preprocess_provenance = _preprocess_provenance(bundle, source_runs)

    # A posterior temperature has not yet been fitted for this checkpoint. Keep
    # that absence explicit so a staging file cannot be mistaken for a releasable
    # PrototypeSet.
    release_blockers = [
        "posterior temperature is not calibrated/exported",
        "open-set AUROC and novelty flag rate do not meet product gates",
        "4096-sample production capture geometry does not meet the length gate",
        "browser-worker latency and memory are not yet measured",
    ]
    if not preprocess_provenance["compatible_with_checkpoint"]:
        release_blockers.append(
            "checkpoint preprocessing source is incompatible with or unverifiable "
            "against the exporter environment; matched retraining/parity is required"
        )
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "schema": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "status": "staging_not_release",
        "created_unix_s": time.time(),
        "source": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "runs": source_runs,
            "paired_real_source": str(
                (HERE / "paired_real_inference.py").relative_to(REPO)
            ),
            "paired_real_source_sha256": _sha256(HERE / "paired_real_inference.py"),
            "exporter_source": str(Path(__file__).resolve().relative_to(REPO)),
            "exporter_source_sha256": _sha256(Path(__file__).resolve()),
        },
        "model": {
            "name": "TransferUNet",
            "architecture": architecture,
            "parameter_count": sum(
                int(parameter.numel()) for parameter in paired.parameters()
            ),
            "complex_representation": "paired real float32 tensors",
        },
        "inputs": {
            "iq": ["batch", 2, input_length],
            "features": ["batch", n_features],
            "raw_capture_length_required": int(raw_capture_length),
        },
        "outputs": {
            "embedding": ["batch", int(architecture["embed_dim"])],
            "bottleneck": ["batch", int(paired.pool_dim)],
            "denoised_real": ["batch", input_length],
            "denoised_imaginary": ["batch", input_length],
            "denoised_coordinate_frame": bundle["target_policy"],
        },
        "preprocessing": preprocess_provenance,
        "classification": {
            "classes": bundle["classes"],
            "prototypes": bundle["prototypes"],
            "unknown_threshold": float(bundle["open_set_threshold"]),
            "temperature": None,
            "calibration_ready": False,
        },
        "feature_standardization": {
            "mean": bundle["feature_mean"],
            "std": bundle["feature_std"],
        },
        "scalar_transfer_head": bundle.get("scalar_transfer_head"),
        "parity": {
            "reference": "original complex TransferUNet eval-mode forward",
            "max_abs_error": parity_errors,
            "limit": 2e-6,
            "real_trace_has_complex_operators": False,
        },
        "onnx": {
            **onnx_report,
            "file": ONNX_NAME,
            "bytes": onnx_path.stat().st_size,
            "sha256": _sha256(onnx_path),
        },
        "parity_fixture": {
            "file": FIXTURE_NAME,
            "bytes": fixture_path.stat().st_size,
            "sha256": _sha256(fixture_path),
        },
        "release_blockers": release_blockers,
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-capture-length", type=int, default=16384)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = export_staging(
        args.checkpoint,
        args.output_dir,
        raw_capture_length=args.raw_capture_length,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output_dir": str(args.output_dir.resolve()),
                "onnx": manifest["onnx"],
                "parity": manifest["parity"],
                "release_blockers": manifest["release_blockers"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
