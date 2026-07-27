"""One-shot fresh-release evaluation for the frozen optional U-Net denoiser.

This evaluator does not route the classifier through the U-Net and does not
modify a release suite.  It binds the frozen neural graph, checkpoint metadata,
authoring-time preprocessing source, and target-alignment implementation by
SHA-256 before reading any release rows.  Results are written exclusively to a
new path outside the sealed release directory.

The exported paired-real staging directory is *not* a release-ready pipeline:
its manifest correctly says that the exporter-time preprocessing source did not
match the checkpoint.  This evaluator uses the ONNX file only as a frozen neural
graph and supplies a separately recovered, byte-exact copy of the checkpoint's
authoring preprocessing source.  Any digest or contract mismatch is fatal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for candidate in (HERE, REPO):
    value = str(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)

import denoise_eval as de  # noqa: E402
import frozen_preprocess_linear_v1 as frozen_pp  # noqa: E402


SCHEMA = "atomos.unet-denoiser.fresh-release-evaluation"
SCHEMA_VERSION = 1
PROTOCOL_VERSION = "unet-denoiser-release-v1"
RELEASE_PROTOCOL = "signallab-matched-length-release-v1"
MATCHED_CAPTURE_LENGTH = 16384
INPUT_LENGTH = 1024
LOW_SNR_DB = 6.0
HIGH_SNR_DB = 18.0
MINIMUM_SLICE_ROWS = 20
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260729
BOOTSTRAP_CONFIDENCE = 0.95
TARGET_POLICY = (
    "input-carrier-aligned denoising: the paired clean waveform is "
    "preprocessed with the impaired input's center, bandwidth, and resample "
    "fraction, then fine-derotated into the input carrier frame; "
    "reconstruction loss is masked to impaired rows"
)
SUPPORT_RULE = (
    "abs(centreOffsetFrac) + bandwidthHz/(2*sampleRateHz) < 0.5"
)

# Frozen before the fresh release suite is generated.
GATES = {
    "impaired_delta_mean_min": 0.02,
    "impaired_delta_ci_lower_strict_min": 0.0,
    "low_snr_delta_mean_min": 0.05,
    "low_snr_delta_ci_lower_strict_min": 0.0,
    "operational_impaired_delta_ci_lower_strict_min": 0.0,
    "operational_low_snr_delta_ci_lower_strict_min": 0.0,
    "clean_delta_ci_lower_min": -0.05,
    "high_snr_delta_ci_lower_min": -0.02,
    "impaired_recon_input_coherence_max": 0.99,
}

GROUND = (
    HERE
    / "artifacts"
    / "ground_state"
    / "corrected_resampled_cropoff_seed20260727"
)
STAGING = HERE / "artifacts" / "staging" / "transfer_unet_paired_real_v1"
DEFAULT_BUNDLE = GROUND / "inference_bundle.pt"
DEFAULT_ONNX = STAGING / "transfer-unet-paired-real-v1.onnx"
DEFAULT_STAGING_MANIFEST = STAGING / "manifest.json"
DEFAULT_PARITY_FIXTURE = STAGING / "paired-real-parity-v1.json"
DEFAULT_PREPROCESS_SOURCE = HERE / "frozen_preprocess_linear_v1.py"
DEFAULT_TARGET_SOURCE = HERE / "denoise_eval.py"

EXPECTED_SHA256 = {
    "bundle": "ebaf3be8272ab7be2de94668078523321d6f062f806c6324f0be0b5062bb0f9c",
    "onnx": "b0c67fa57dfb315a34d2e06ee4ec35239de36c737d279b05aba8fd90c2e5fba8",
    "staging_manifest": "d7b2012adb68a97e7a15d582aaf7042f95d5e1e94a1f306ee0fdd15c661e9aaa",
    "parity_fixture": "e690fe3f7c65cfed44c5d6d4fd5feacfb6bb92c64a2ccbeaed062a32763779bc",
    "preprocess_source": "5bc1678e8a889d1237d97c00a9528e9604e61806122f8205eb63cdbe8ac4c10f",
    "target_source": "262c4d8e5f35296466f4c1dd7cfbbdd139ec1e6457158f0c7a1eff7d2c818ccb",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_regular_file(path: Path, label: str) -> Path:
    lexical = path.expanduser().absolute()
    if lexical.is_symlink():
        raise ValueError(f"{label} may not be a symlink: {lexical}")
    if not lexical.is_file():
        raise FileNotFoundError(f"{label} is not a regular file: {lexical}")
    resolved = lexical.resolve()
    if resolved != lexical:
        raise ValueError(f"{label} may not use a symlink alias: {lexical}")
    return resolved


def _assert_real_directory(path: Path, label: str) -> Path:
    lexical = path.expanduser().absolute()
    if lexical.is_symlink():
        raise ValueError(f"{label} may not be a symlink: {lexical}")
    if not lexical.is_dir():
        raise FileNotFoundError(f"{label} is not a directory: {lexical}")
    resolved = lexical.resolve()
    if resolved != lexical:
        raise ValueError(f"{label} may not use a symlink alias: {lexical}")
    return resolved


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def load_json_strict(path: Path) -> Any:
    raw = path.read_bytes()

    def reject_constant(value: str) -> None:
        raise ValueError(f"JSON contains non-finite constant {value}")

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as failure:
        raise ValueError(f"{path} is not strict UTF-8 JSON: {failure}") from failure


def _require_sha(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(_assert_regular_file(path, label))
    if actual != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def _tensor_numpy(value: Any, label: str) -> np.ndarray:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    return value.detach().cpu().numpy()


def _fixture_array(payload: dict[str, Any], label: str) -> np.ndarray:
    shape = payload.get("shape")
    values = payload.get("values")
    if not isinstance(shape, list) or not all(
        isinstance(value, int) and value > 0 for value in shape
    ):
        raise ValueError(f"{label}.shape is invalid")
    array = np.asarray(values, dtype=np.float32)
    if array.size != math.prod(shape):
        raise ValueError(
            f"{label} has {array.size} values for declared shape {shape}"
        )
    array = array.reshape(shape)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains non-finite values")
    return array


def verify_frozen_artifacts(
    *,
    bundle_path: Path = DEFAULT_BUNDLE,
    onnx_path: Path = DEFAULT_ONNX,
    staging_manifest_path: Path = DEFAULT_STAGING_MANIFEST,
    parity_fixture_path: Path = DEFAULT_PARITY_FIXTURE,
    preprocess_source_path: Path = DEFAULT_PREPROCESS_SOURCE,
    target_source_path: Path = DEFAULT_TARGET_SOURCE,
) -> tuple[dict[str, Any], Any, np.ndarray, np.ndarray]:
    """Bind every executable contract and execute the frozen parity fixture."""
    paths = {
        "bundle": _assert_regular_file(bundle_path, "frozen inference bundle"),
        "onnx": _assert_regular_file(onnx_path, "frozen ONNX graph"),
        "staging_manifest": _assert_regular_file(
            staging_manifest_path, "staging manifest"
        ),
        "parity_fixture": _assert_regular_file(
            parity_fixture_path, "parity fixture"
        ),
        "preprocess_source": _assert_regular_file(
            preprocess_source_path, "authoring preprocessing source"
        ),
        "target_source": _assert_regular_file(
            target_source_path, "target-alignment source"
        ),
    }
    observed_hashes = {
        name: _require_sha(path, EXPECTED_SHA256[name], name)
        for name, path in paths.items()
    }

    # The imported modules must be the exact files just bound above.
    if Path(frozen_pp.__file__).resolve() != paths["preprocess_source"]:
        raise ValueError("imported preprocessing module is not the bound source file")
    if Path(de.__file__).resolve() != paths["target_source"]:
        raise ValueError("imported target-alignment module is not the bound source file")

    manifest = load_json_strict(paths["staging_manifest"])
    if manifest.get("schema") != "atomos.transfer-unet.paired-real":
        raise ValueError("unexpected staging manifest schema")
    if manifest.get("schema_version") != 1:
        raise ValueError("unexpected staging manifest schema version")
    if manifest.get("status") != "staging_not_release":
        raise ValueError("staging manifest must remain marked staging_not_release")
    if manifest.get("onnx", {}).get("sha256") != EXPECTED_SHA256["onnx"]:
        raise ValueError("staging manifest does not bind the frozen ONNX graph")
    if (
        manifest.get("source", {}).get("checkpoint_sha256")
        != EXPECTED_SHA256["bundle"]
    ):
        raise ValueError("staging manifest does not bind the frozen bundle")
    preprocessing = manifest.get("preprocessing", {})
    authoring_sha = (
        preprocessing.get("checkpoint_authoring", {}).get("source_sha256")
    )
    if authoring_sha != EXPECTED_SHA256["preprocess_source"]:
        raise ValueError("manifest authoring preprocessing SHA does not match")
    if preprocessing.get("compatible_with_checkpoint") is not False:
        raise ValueError(
            "this protocol expects the staging integration to remain explicitly "
            "incompatible, not silently relabelled"
        )
    exporter_sha = (
        preprocessing.get("exporter_environment", {}).get("source_sha256")
    )
    if exporter_sha == authoring_sha:
        raise ValueError("manifest no longer records the audited preprocessing mismatch")
    execution = manifest.get("onnx", {}).get("execution", {})
    if execution.get("status") != "passed":
        raise ValueError("frozen ONNX execution parity was not recorded as passed")
    recorded_errors = execution.get("max_abs_error", {})
    parity_limit = float(execution.get("parity_limit", 0.0))
    if (
        not recorded_errors
        or not np.all(np.isfinite(list(recorded_errors.values())))
        or max(float(value) for value in recorded_errors.values()) > parity_limit
    ):
        raise ValueError("recorded ONNX parity errors exceed their limit")

    # Trusted local artifact: its digest is checked before pickle deserialization.
    bundle = torch.load(paths["bundle"], map_location="cpu", weights_only=False)
    required_bundle = {
        "schema_version",
        "model",
        "architecture",
        "state_dict",
        "classes",
        "feature_mean",
        "feature_std",
        "preprocessing",
        "target_policy",
    }
    missing = sorted(required_bundle.difference(bundle))
    if missing:
        raise ValueError(f"frozen bundle is missing keys: {missing}")
    if bundle["schema_version"] != 1 or bundle["model"] != "TransferUNet":
        raise ValueError("unexpected frozen bundle model contract")
    classes = bundle["classes"]
    if (
        not isinstance(classes, list)
        or not classes
        or len(set(classes)) != len(classes)
        or not all(isinstance(name, str) and name for name in classes)
    ):
        raise ValueError("frozen bundle classes are invalid")
    if bundle["target_policy"] != TARGET_POLICY:
        raise ValueError("frozen bundle target policy does not match this evaluator")
    if manifest.get("outputs", {}).get("denoised_coordinate_frame") != TARGET_POLICY:
        raise ValueError("ONNX manifest target coordinate frame does not match")
    if manifest.get("inputs", {}).get("iq") != ["batch", 2, INPUT_LENGTH]:
        raise ValueError("ONNX manifest I/Q input shape changed")
    if (
        manifest.get("inputs", {}).get("raw_capture_length_required")
        != MATCHED_CAPTURE_LENGTH
    ):
        raise ValueError("ONNX manifest raw capture length changed")
    if bundle["preprocessing"] != {
        "condition": "resampled",
        "input_length": INPUT_LENGTH,
        "module": "training/preprocess.py",
    }:
        raise ValueError("frozen bundle preprocessing declaration changed")
    architecture = bundle["architecture"]
    if architecture != manifest.get("model", {}).get("architecture"):
        raise ValueError("bundle and ONNX architecture declarations differ")
    if int(architecture.get("n_features", -1)) != frozen_pp.N_FEATURES:
        raise ValueError("preprocessing/model feature counts differ")
    feature_mean = _tensor_numpy(bundle["feature_mean"], "feature_mean").astype(
        np.float32, copy=False
    )
    feature_std = _tensor_numpy(bundle["feature_std"], "feature_std").astype(
        np.float32, copy=False
    )
    if feature_mean.shape != (frozen_pp.N_FEATURES,):
        raise ValueError("feature_mean has the wrong shape")
    if (
        feature_std.shape != feature_mean.shape
        or not np.all(np.isfinite(feature_mean))
        or not np.all(np.isfinite(feature_std))
        or np.any(feature_std <= 0.0)
    ):
        raise ValueError("feature standardization is invalid")

    try:
        import onnxruntime as ort
    except ImportError as failure:
        raise RuntimeError(
            "onnxruntime is required for the frozen denoiser evaluation"
        ) from failure
    if "CPUExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("onnxruntime lacks CPUExecutionProvider")
    session = ort.InferenceSession(
        str(paths["onnx"]), providers=["CPUExecutionProvider"]
    )
    if session.get_providers()[0] != "CPUExecutionProvider":
        raise RuntimeError(
            f"onnxruntime activated unexpected providers {session.get_providers()}"
        )

    fixture = load_json_strict(paths["parity_fixture"])
    fixture_iq = np.asarray(
        fixture.get("input", {}).get("iq"), dtype=np.float32
    ).reshape(fixture.get("input", {}).get("iq_shape", []))
    fixture_features = np.asarray(
        fixture.get("input", {}).get("features"), dtype=np.float32
    ).reshape(fixture.get("input", {}).get("features_shape", []))
    if fixture_iq.shape != (1, 2, INPUT_LENGTH):
        raise ValueError("parity fixture I/Q shape changed")
    if fixture_features.shape != (1, frozen_pp.N_FEATURES):
        raise ValueError("parity fixture feature shape changed")
    if not np.all(np.isfinite(fixture_iq)) or not np.all(
        np.isfinite(fixture_features)
    ):
        raise ValueError("parity fixture input is non-finite")
    output_names = (
        "embedding",
        "bottleneck",
        "denoised_real",
        "denoised_imaginary",
    )
    actual = session.run(
        list(output_names),
        {"iq": fixture_iq, "features": fixture_features},
    )
    fixture_errors: dict[str, float] = {}
    for name, value in zip(output_names, actual):
        expected = _fixture_array(fixture["expected"][name], f"expected.{name}")
        if value.shape != expected.shape or not np.all(np.isfinite(value)):
            raise ValueError(f"ONNX parity output {name} is invalid")
        fixture_errors[name] = float(np.max(np.abs(value - expected)))
    if max(fixture_errors.values()) > parity_limit:
        raise ValueError(
            f"live ONNX parity failed: {fixture_errors}, limit={parity_limit}"
        )

    public = {
        "artifact_sha256": observed_hashes,
        "bundle_path": str(paths["bundle"]),
        "onnx_path": str(paths["onnx"]),
        "preprocess_source_path": str(paths["preprocess_source"]),
        "target_source_path": str(paths["target_source"]),
        "staging_integration": {
            "status": "blocked_not_release_ready",
            "compatible_with_checkpoint": False,
            "checkpoint_authoring_preprocess_sha256": authoring_sha,
            "exporter_environment_preprocess_sha256": exporter_sha,
            "compatibility_reason": preprocessing.get("compatibility_reason"),
            "release_blockers": manifest.get("release_blockers"),
        },
        "evaluation_composite": {
            "status": "ready",
            "scope": "denoising evaluation only; never classifier routing",
            "reason": (
                "the neural graph is frozen by ONNX SHA and the separately "
                "bound preprocessing file exactly matches the checkpoint's "
                "authoring SHA"
            ),
        },
        "live_onnx_parity_max_abs_error": fixture_errors,
        "onnx_parity_limit": parity_limit,
        "onnxruntime_version": ort.__version__,
        "execution_provider": "CPUExecutionProvider",
        "target_policy": TARGET_POLICY,
    }
    return public, session, feature_mean, feature_std


def _record_for(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _verify_declared_record(
    path: Path, declared: Any, label: str
) -> dict[str, Any]:
    if not isinstance(declared, dict):
        raise ValueError(f"{label} has no declared file record")
    actual = _record_for(_assert_regular_file(path, label))
    if declared != actual:
        raise ValueError(
            f"{label} does not match release manifest: "
            f"declared={declared}, actual={actual}"
        )
    return actual


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def validate_corpus_manifest(
    corpus: Any,
    *,
    target_per_class: int,
    expected_classes: tuple[str, ...],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    if not isinstance(corpus, dict):
        raise ValueError("corpus.json must contain an object")
    if corpus.get("sampleCount") != MATCHED_CAPTURE_LENGTH:
        raise ValueError("denoiser release corpus has the wrong sampleCount")
    if corpus.get("format") != "cf32le-interleaved":
        raise ValueError("denoiser release corpus has the wrong format")
    if corpus.get("hasCleanPairs") is not True:
        raise ValueError("denoiser evaluation requires paired clean I/Q")
    if corpus.get("exactTargetPerClass") is not True:
        raise ValueError("release corpus must use exactTargetPerClass")
    if corpus.get("targetPerClass") != target_per_class:
        raise ValueError("corpus and release targetPerClass differ")
    classes = corpus.get("classes")
    if classes != list(expected_classes):
        raise ValueError(
            f"release classes {classes!r} do not match bundle classes "
            f"{list(expected_classes)!r}"
        )
    count = corpus.get("count")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count != target_per_class * len(expected_classes)
    ):
        raise ValueError("release corpus count is inconsistent with exact class target")
    items = corpus.get("items")
    if not isinstance(items, list) or len(items) != count:
        raise ValueError("release corpus items length differs from count")

    counts = {name: 0 for name in expected_classes}
    supported = np.zeros(count, dtype=bool)
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise TypeError(f"items[{index}] must be an object")
        cls = item.get("cls")
        if cls not in counts:
            raise ValueError(f"items[{index}].cls is not in bundle classes")
        counts[cls] += 1
        if not isinstance(item.get("impaired"), bool):
            raise TypeError(f"items[{index}].impaired must be boolean")
        sample_rate = _finite_number(
            item.get("sampleRateHz"), f"items[{index}].sampleRateHz"
        )
        bandwidth = _finite_number(
            item.get("bandwidthHz"), f"items[{index}].bandwidthHz"
        )
        center = _finite_number(
            item.get("centreOffsetFrac"), f"items[{index}].centreOffsetFrac"
        )
        _finite_number(item.get("snrDb"), f"items[{index}].snrDb")
        if sample_rate <= 0.0 or bandwidth <= 0.0 or bandwidth > sample_rate:
            raise ValueError(f"items[{index}] has invalid RF geometry")
        supported[index] = (
            abs(center) + bandwidth / (2.0 * sample_rate) < 0.5
        )
    if any(value != target_per_class for value in counts.values()):
        raise ValueError(f"per-class counts are not exact: {counts}")
    return items, supported


def bind_release_corpus(
    release_root: Path,
    *,
    expected_classes: tuple[str, ...],
) -> tuple[
    dict[str, Any],
    Path,
    Path,
    list[dict[str, Any]],
    np.ndarray,
    dict[str, Any],
]:
    root = _assert_real_directory(release_root, "release root")
    release_manifest_path = _assert_regular_file(
        root / "RELEASE_MANIFEST.json", "release manifest"
    )
    release_manifest = load_json_strict(release_manifest_path)
    if release_manifest.get("protocol") != RELEASE_PROTOCOL:
        raise ValueError("unexpected release generation protocol")
    if release_manifest.get("status") != "complete":
        raise ValueError("release generation is not complete")
    if release_manifest.get("development_data_used") is not False:
        raise ValueError("release manifest does not declare development_data_used=false")
    lengths = release_manifest.get("capture_lengths")
    if not isinstance(lengths, list) or MATCHED_CAPTURE_LENGTH not in lengths:
        raise ValueError("release suite lacks the denoiser's matched capture length")
    evaluation_protocol = release_manifest.get("evaluation_protocol", {})
    if not isinstance(evaluation_protocol, dict):
        raise ValueError("release evaluation_protocol must be an object")
    if evaluation_protocol.get("matched_capture_length") != MATCHED_CAPTURE_LENGTH:
        raise ValueError("release evaluation protocol changed matched_capture_length")
    if evaluation_protocol.get("high_snr_db") != HIGH_SNR_DB:
        raise ValueError("release high-SNR threshold differs from denoiser protocol")
    target_per_class = release_manifest.get("target_per_class")
    if (
        isinstance(target_per_class, bool)
        or not isinstance(target_per_class, int)
        or target_per_class < 80
    ):
        raise ValueError("release target_per_class must be an integer >= 80")

    corpus_records = release_manifest.get("corpora")
    if not isinstance(corpus_records, list):
        raise ValueError("release manifest corpora must be an array")
    matching = [
        record
        for record in corpus_records
        if isinstance(record, dict)
        if record.get("capture_length") == MATCHED_CAPTURE_LENGTH
    ]
    if len(matching) != 1:
        raise ValueError("release manifest must bind exactly one matched corpus")
    record = matching[0]
    corpus_dir = _assert_real_directory(
        root / f"n{MATCHED_CAPTURE_LENGTH}", "matched corpus directory"
    )
    declared_dir = Path(record.get("directory", "")).expanduser().resolve()
    if declared_dir != corpus_dir:
        raise ValueError("matched corpus directory differs from release manifest")
    corpus_manifest_path = corpus_dir / "corpus.json"
    raw_path = corpus_dir / "corpus.f32"
    clean_path = corpus_dir / "corpus_clean.f32"
    file_records = {
        name: _verify_declared_record(
            corpus_dir / name, record.get("files", {}).get(name), name
        )
        for name in ("corpus.json", "corpus.f32", "corpus_clean.f32")
    }
    corpus = load_json_strict(corpus_manifest_path)
    items, supported = validate_corpus_manifest(
        corpus,
        target_per_class=target_per_class,
        expected_classes=expected_classes,
    )
    expected_binary_bytes = (
        len(items) * MATCHED_CAPTURE_LENGTH * 2 * np.dtype("<f4").itemsize
    )
    for name in ("corpus.f32", "corpus_clean.f32"):
        if file_records[name]["bytes"] != expected_binary_bytes:
            raise ValueError(
                f"{name} size is incompatible with release corpus geometry"
            )

    release_binding = {
        "release_root": str(root),
        "release_seed": release_manifest.get("release_seed"),
        "release_manifest_sha256": sha256_file(release_manifest_path),
        "release_protocol": release_manifest.get("protocol"),
        "capture_length": MATCHED_CAPTURE_LENGTH,
        "target_per_class": target_per_class,
        "corpus_directory": str(corpus_dir),
        "corpus_files": file_records,
    }
    return (
        release_binding,
        raw_path,
        clean_path,
        items,
        supported,
        release_manifest,
    )


def _bootstrap_seed(label: str) -> int:
    payload = f"{BOOTSTRAP_SEED}\0{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def paired_bootstrap_ci(
    delta: np.ndarray,
    *,
    label: str,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[float, float]:
    values = np.asarray(delta, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("paired bootstrap requires a finite non-empty vector")
    if not isinstance(replicates, int) or replicates < 100:
        raise ValueError("bootstrap replicates must be an integer >= 100")
    rng = np.random.default_rng(_bootstrap_seed(label))
    boot = np.empty(replicates, dtype=np.float64)
    block = max(1, min(512, replicates))
    for start in range(0, replicates, block):
        stop = min(start + block, replicates)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        boot[start:stop] = values[indices].mean(axis=1)
    tail = (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0
    low, high = np.quantile(boot, [tail, 1.0 - tail])
    return float(low), float(high)


def metric_summary(
    passthrough: np.ndarray,
    model: np.ndarray,
    mask: np.ndarray,
    *,
    label: str,
) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool)
    p = np.asarray(passthrough, dtype=np.float64)[selected]
    m = np.asarray(model, dtype=np.float64)[selected]
    if len(p) < MINIMUM_SLICE_ROWS:
        raise ValueError(
            f"slice {label!r} has {len(p)} rows; need at least "
            f"{MINIMUM_SLICE_ROWS}"
        )
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(m)):
        raise ValueError(f"slice {label!r} contains non-finite coherence")
    delta = m - p
    ci = paired_bootstrap_ci(delta, label=label)
    return {
        "n": int(len(delta)),
        "passthrough_mean": float(p.mean()),
        "denoiser_mean": float(m.mean()),
        "paired_delta_mean": float(delta.mean()),
        "paired_delta_median": float(np.median(delta)),
        "paired_delta_win_rate": float(np.mean(delta > 0.0)),
        "paired_delta_ci95": [ci[0], ci[1]],
        "bootstrap": {
            "method": "paired iid percentile bootstrap of mean row delta",
            "confidence": BOOTSTRAP_CONFIDENCE,
            "replicates": BOOTSTRAP_REPLICATES,
            "base_seed": BOOTSTRAP_SEED,
            "slice_seed_derivation": (
                "uint64 little-endian first 8 bytes of "
                "SHA-256(utf8(base_seed_decimal + NUL + slice_label))"
            ),
        },
    }


def _slice_masks(
    items: list[dict[str, Any]], supported: np.ndarray
) -> dict[str, np.ndarray]:
    impaired = np.asarray([item["impaired"] for item in items], dtype=bool)
    snr = np.asarray([item["snrDb"] for item in items], dtype=np.float64)
    return {
        "contract_all": supported.copy(),
        "clean": supported & ~impaired,
        "impaired": supported & impaired,
        "impaired_low_snr": supported & impaired & (snr < LOW_SNR_DB),
        "impaired_high_snr": supported & impaired & (snr >= HIGH_SNR_DB),
    }


def _run_onnx(
    session: Any,
    iq: np.ndarray,
    features: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    outputs = []
    for start in range(0, len(iq), batch_size):
        stop = min(start + batch_size, len(iq))
        actual = session.run(
            ["denoised_real", "denoised_imaginary"],
            {
                "iq": np.asarray(iq[start:stop], dtype=np.float32),
                "features": np.asarray(features[start:stop], dtype=np.float32),
            },
        )
        real, imaginary = actual
        if real.shape != (stop - start, INPUT_LENGTH) or imaginary.shape != real.shape:
            raise ValueError("ONNX denoised output has the wrong shape")
        if not np.all(np.isfinite(real)) or not np.all(np.isfinite(imaginary)):
            raise ValueError("ONNX denoised output contains NaN or infinity")
        outputs.append(
            real.astype(np.float64) + 1j * imaginary.astype(np.float64)
        )
    return np.concatenate(outputs, axis=0)


def evaluate_release(
    release_root: Path,
    *,
    output_path: Path,
    batch_size: int = 32,
    bundle_path: Path = DEFAULT_BUNDLE,
    onnx_path: Path = DEFAULT_ONNX,
    staging_manifest_path: Path = DEFAULT_STAGING_MANIFEST,
    parity_fixture_path: Path = DEFAULT_PARITY_FIXTURE,
    preprocess_source_path: Path = DEFAULT_PREPROCESS_SOURCE,
    target_source_path: Path = DEFAULT_TARGET_SOURCE,
) -> dict[str, Any]:
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    root = _assert_real_directory(release_root, "release root")
    output = output_path.expanduser().absolute()
    resolved_output = output.resolve(strict=False)
    if resolved_output == root or root in resolved_output.parents:
        raise ValueError("evaluation output must be outside the sealed release root")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output {output}")

    artifacts, session, feature_mean, feature_std = verify_frozen_artifacts(
        bundle_path=bundle_path,
        onnx_path=onnx_path,
        staging_manifest_path=staging_manifest_path,
        parity_fixture_path=parity_fixture_path,
        preprocess_source_path=preprocess_source_path,
        target_source_path=target_source_path,
    )
    bundle = torch.load(
        _assert_regular_file(bundle_path, "frozen inference bundle"),
        map_location="cpu",
        weights_only=False,
    )
    expected_classes = tuple(bundle["classes"])
    (
        release_binding,
        raw_path,
        clean_path,
        items,
        supported,
        _release_manifest,
    ) = bind_release_corpus(root, expected_classes=expected_classes)

    count = len(items)
    raw = np.memmap(
        raw_path,
        dtype="<f4",
        mode="r",
        shape=(count, MATCHED_CAPTURE_LENGTH, 2),
    )
    clean = np.memmap(
        clean_path,
        dtype="<f4",
        mode="r",
        shape=(count, MATCHED_CAPTURE_LENGTH, 2),
    )
    processed = np.empty((count, 2, INPUT_LENGTH), dtype=np.float32)
    targets = np.empty((count, INPUT_LENGTH), dtype=np.complex64)
    features = np.empty((count, frozen_pp.N_FEATURES), dtype=np.float32)
    centers = np.empty(count, dtype=np.float64)
    bandwidths = np.empty(count, dtype=np.float64)

    for index in range(count):
        observed_raw = (
            np.asarray(raw[index, :, 0], dtype=np.float32)
            + 1j * np.asarray(raw[index, :, 1], dtype=np.float32)
        )
        clean_raw = (
            np.asarray(clean[index, :, 0], dtype=np.float32)
            + 1j * np.asarray(clean[index, :, 1], dtype=np.float32)
        )
        if not np.all(np.isfinite(observed_raw)) or not np.all(
            np.isfinite(clean_raw)
        ):
            raise ValueError(f"release pair {index} contains NaN or infinity")
        observed_norm, context = frozen_pp.preprocess(
            observed_raw,
            l_out=INPUT_LENGTH,
            scale_jitter=0.0,
        )
        clean_norm, _ = frozen_pp.preprocess(
            clean_raw,
            l_out=INPUT_LENGTH,
            scale_jitter=0.0,
            force_center=context["center"],
            force_bw=context["bw"],
            force_resample_frac=context["resample_frac"],
        )
        aligned = de.derotate_onto(clean_norm, observed_norm)
        processed[index] = frozen_pp.to_channels(observed_norm)
        targets[index] = aligned
        raw_features = frozen_pp.iq_features(observed_norm)
        features[index] = (raw_features - feature_mean) / feature_std
        centers[index] = context["center"]
        bandwidths[index] = context["bw"]
    del raw, clean
    if not np.all(np.isfinite(processed)) or not np.all(np.isfinite(features)):
        raise ValueError("preprocessing produced NaN or infinity")

    recon = _run_onnx(
        session, processed, features, batch_size=batch_size
    )
    observed_complex = (
        processed[:, 0].astype(np.float64)
        + 1j * processed[:, 1].astype(np.float64)
    )
    target_complex = np.asarray(targets, dtype=np.complex128)
    passthrough = de.coherence(observed_complex, target_complex)
    model = de.coherence(recon, target_complex)
    recon_input = de.coherence(recon, observed_complex)
    for label, value in (
        ("passthrough", passthrough),
        ("model", model),
        ("recon_input", recon_input),
    ):
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{label} coherence contains NaN or infinity")
        if np.any(value < -1e-9) or np.any(value > 1.0 + 1e-6):
            raise ValueError(f"{label} coherence is outside [0, 1]")

    supported_masks = _slice_masks(items, supported)
    supported_slices = {
        name: metric_summary(
            passthrough, model, mask, label=f"supported:{name}"
        )
        for name, mask in supported_masks.items()
    }
    # The operational sidecar abstains outside its authoring support and returns
    # the original I/Q. Report the full-release behavior as well as the supported
    # subset so exclusion cannot inflate the shipping claim.
    operational_model = np.where(supported, model, passthrough)
    operational_masks = _slice_masks(
        items, np.ones(count, dtype=bool)
    )
    operational_slices = {
        name: metric_summary(
            passthrough,
            operational_model,
            mask,
            label=f"operational:{name}",
        )
        for name, mask in operational_masks.items()
    }
    impaired_recon_input = float(
        recon_input[supported_masks["impaired"]].mean()
    )
    gate_results = {
        "impaired_delta_mean": {
            "passes": (
                supported_slices["impaired"]["paired_delta_mean"]
                >= GATES["impaired_delta_mean_min"]
            ),
            "observed": supported_slices["impaired"]["paired_delta_mean"],
            "minimum": GATES["impaired_delta_mean_min"],
        },
        "impaired_delta_ci_lower": {
            "passes": (
                supported_slices["impaired"]["paired_delta_ci95"][0]
                > GATES["impaired_delta_ci_lower_strict_min"]
            ),
            "observed": supported_slices["impaired"][
                "paired_delta_ci95"
            ][0],
            "strict_minimum": GATES["impaired_delta_ci_lower_strict_min"],
        },
        "low_snr_delta_mean": {
            "passes": (
                supported_slices["impaired_low_snr"]["paired_delta_mean"]
                >= GATES["low_snr_delta_mean_min"]
            ),
            "observed": supported_slices["impaired_low_snr"][
                "paired_delta_mean"
            ],
            "minimum": GATES["low_snr_delta_mean_min"],
        },
        "low_snr_delta_ci_lower": {
            "passes": (
                supported_slices["impaired_low_snr"][
                    "paired_delta_ci95"
                ][0]
                > GATES["low_snr_delta_ci_lower_strict_min"]
            ),
            "observed": supported_slices["impaired_low_snr"][
                "paired_delta_ci95"
            ][0],
            "strict_minimum": GATES["low_snr_delta_ci_lower_strict_min"],
        },
        "operational_impaired_delta_ci_lower": {
            "passes": (
                operational_slices["impaired"]["paired_delta_ci95"][0]
                > GATES[
                    "operational_impaired_delta_ci_lower_strict_min"
                ]
            ),
            "observed": operational_slices["impaired"][
                "paired_delta_ci95"
            ][0],
            "strict_minimum": GATES[
                "operational_impaired_delta_ci_lower_strict_min"
            ],
        },
        "operational_low_snr_delta_ci_lower": {
            "passes": (
                operational_slices["impaired_low_snr"][
                    "paired_delta_ci95"
                ][0]
                > GATES[
                    "operational_low_snr_delta_ci_lower_strict_min"
                ]
            ),
            "observed": operational_slices["impaired_low_snr"][
                "paired_delta_ci95"
            ][0],
            "strict_minimum": GATES[
                "operational_low_snr_delta_ci_lower_strict_min"
            ],
        },
        "clean_preservation": {
            "passes": (
                operational_slices["clean"]["paired_delta_ci95"][0]
                >= GATES["clean_delta_ci_lower_min"]
            ),
            "observed_ci_lower": operational_slices["clean"][
                "paired_delta_ci95"
            ][0],
            "minimum": GATES["clean_delta_ci_lower_min"],
        },
        "high_snr_preservation": {
            "passes": (
                operational_slices["impaired_high_snr"][
                    "paired_delta_ci95"
                ][0]
                >= GATES["high_snr_delta_ci_lower_min"]
            ),
            "observed_ci_lower": operational_slices["impaired_high_snr"][
                "paired_delta_ci95"
            ][0],
            "minimum": GATES["high_snr_delta_ci_lower_min"],
        },
        "not_identity_passthrough": {
            "passes": (
                impaired_recon_input
                < GATES["impaired_recon_input_coherence_max"]
            ),
            "observed": impaired_recon_input,
            "strict_maximum": GATES[
                "impaired_recon_input_coherence_max"
            ],
        },
    }
    all_gates_pass = all(
        bool(result["passes"]) for result in gate_results.values()
    )

    class_support = {
        name: {
            "supported": int(
                sum(
                    bool(supported[index]) and item["cls"] == name
                    for index, item in enumerate(items)
                )
            ),
            "unsupported": int(
                sum(
                    not bool(supported[index]) and item["cls"] == name
                    for index, item in enumerate(items)
                )
            ),
        }
        for name in expected_classes
    }
    if any(value["supported"] == 0 for value in class_support.values()):
        raise ValueError("authoring preprocessing support has an empty class")

    # Re-hash after inference. The report is invalid if either input changed
    # between the preflight binding and scoring.
    post_records = {
        "corpus.f32": _record_for(raw_path),
        "corpus_clean.f32": _record_for(clean_path),
    }
    for name, actual in post_records.items():
        if actual != release_binding["corpus_files"][name]:
            raise ValueError(f"{name} changed during denoiser evaluation")

    report = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if all_gates_pass else "failed",
        "role": (
            "optional denoising-only sidecar; classifier inputs, embeddings, "
            "labels, rejection scores, and abstention policy are unchanged"
        ),
        "protocol": {
            "version": PROTOCOL_VERSION,
            "capture_length": MATCHED_CAPTURE_LENGTH,
            "input_length": INPUT_LENGTH,
            "target_policy": TARGET_POLICY,
            "support_rule": SUPPORT_RULE,
            "unsupported_action": "abstain from U-Net; preserve original I/Q",
            "low_snr_db": LOW_SNR_DB,
            "high_snr_db": HIGH_SNR_DB,
            "minimum_slice_rows": MINIMUM_SLICE_ROWS,
            "gates": GATES,
        },
        "artifact_binding": artifacts,
        "release_binding": release_binding,
        "support": {
            "supported_rows": int(supported.sum()),
            "unsupported_rows": int((~supported).sum()),
            "per_class": class_support,
        },
        "metrics": {
            "supported_contract_slices": supported_slices,
            "operational_pass_through_slices": operational_slices,
            "impaired_recon_input_coherence_mean": impaired_recon_input,
            "preprocess_context": {
                "center_min": float(centers.min()),
                "center_max": float(centers.max()),
                "bandwidth_min": float(bandwidths.min()),
                "bandwidth_max": float(bandwidths.max()),
            },
        },
        "gate_results": gate_results,
        "all_gates_pass": all_gates_pass,
        "environment": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "evaluator_sha256": sha256_file(Path(__file__).resolve()),
            "warnings_policy": os.environ.get("PYTHONWARNINGS"),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="verify frozen artifacts and live ONNX parity without reading release data",
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument(
        "--staging-manifest", type=Path, default=DEFAULT_STAGING_MANIFEST
    )
    parser.add_argument(
        "--parity-fixture", type=Path, default=DEFAULT_PARITY_FIXTURE
    )
    parser.add_argument(
        "--preprocess-source", type=Path, default=DEFAULT_PREPROCESS_SOURCE
    )
    parser.add_argument(
        "--target-source", type=Path, default=DEFAULT_TARGET_SOURCE
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    artifact_kwargs = {
        "bundle_path": args.bundle,
        "onnx_path": args.onnx,
        "staging_manifest_path": args.staging_manifest,
        "parity_fixture_path": args.parity_fixture,
        "preprocess_source_path": args.preprocess_source,
        "target_source_path": args.target_source,
    }
    if args.preflight_only:
        if args.release_root is not None or args.output is not None:
            raise ValueError(
                "--preflight-only may not receive --release-root or --output"
            )
        report, _session, _mean, _std = verify_frozen_artifacts(
            **artifact_kwargs
        )
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.release_root is None or args.output is None:
        raise ValueError("--release-root and --output are required")
    report = evaluate_release(
        args.release_root,
        output_path=args.output,
        batch_size=args.batch_size,
        **artifact_kwargs,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(args.output.absolute()),
                "all_gates_pass": report["all_gates_pass"],
                "support": report["support"],
                "gate_results": report["gate_results"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0 if report["all_gates_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
