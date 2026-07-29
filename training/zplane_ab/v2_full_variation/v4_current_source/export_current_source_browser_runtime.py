"""Export a v4 current-source fusion as deterministic browser JSON.

The input is the development fusion directory emitted by
``assemble_current_source_fusion.py``.  The exporter verifies every recorded
artifact digest, validates the seven-class source/profile prototype-bank
mapping, and emits:

* ``time-domain-profile-bank-v4.json`` -- both encoder branches, centered
  fusion, feature moments, and the mapped source/profile prototype bank;
* ``time-domain-profile-bank-probe-v4.json`` -- Python reference forwards for
  the 4,096/8,192/16,384 raw prefix buckets;
* ``export-manifest.json`` -- deterministic SHA-256 records for both files.

This path is deliberately isolated from the live browser assets.  It refuses
release/sealed/consumed-test paths, refuses the live ``src/embedding/assets``
directory, and writes only to a fresh empty output directory.  It does not
promote or deploy anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
V2 = HERE.parent
ZPLANE = V2.parent
TRAINING = ZPLANE.parent
REPO = TRAINING.parent
V3_SCALE = V2 / "v3_scale"
for path in (TRAINING, ZPLANE, V2, V3_SCALE, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import export_v3_browser_weights as encoder_export  # noqa: E402
import current_source_data as corpus_data  # noqa: E402
import time_domain_invariant_patch_preprocess as td_preprocess  # noqa: E402
from invariant_fusion import CenteredInvariantFusion  # noqa: E402
from invariant_patch_cnn import (  # noqa: E402
    InvariantPatchCNN,
    InvariantPatchConfig,
)


BROWSER_SCHEMA = (
    "atomos.v4.time-domain-current-source-profile-bank.browser-weights"
)
BROWSER_SCHEMA_VERSION = 2
PROBE_SCHEMA = (
    "atomos.v4.time-domain-current-source-profile-bank.probe-fixture"
)
PROBE_SCHEMA_VERSION = 2
EXPORT_MANIFEST_SCHEMA = (
    "atomos.v4.time-domain-current-source-profile-bank.export-manifest"
)
EXPORT_MANIFEST_SCHEMA_VERSION = 2

WEIGHTS_NAME = "time-domain-profile-bank-v4.json"
PROBE_NAME = "time-domain-profile-bank-probe-v4.json"
EXPORT_MANIFEST_NAME = "export-manifest.json"

PUBLIC_CLASSES = ("am", "bluetooth", "cw", "dsss", "fm", "gsm", "ofdm")
RUNTIME_INPUT_LENGTHS = (4_096, 8_192, 16_384)
RUNTIME_BUCKET_RULE = (
    "largest_supported_prefix_not_exceeding_valid_sample_count"
)
PROFILE_BANK_DECISION_RULE = "minimum_squared_distance_per_public_class"
TRUSTED_SOURCE_ROUTING_KIND = (
    "trusted_acquisition_source_to_prototype_source_v1"
)
SIGNAL_LAB_PROFILE_ROUTING_RULE = (
    "current_only_for_exact_current_profile_inventory_else_historical"
)
TRUSTED_SOURCE_ROUTING = {
    "kind": TRUSTED_SOURCE_ROUTING_KIND,
    "default_untagged": "historical",
    "acquisition_source_map": {
        "signal-lab": "selected-profile-conditioned",
        "physical-sdr": "historical",
        "untagged": "historical",
    },
    "signal_lab_selected_profile_rule": SIGNAL_LAB_PROFILE_ROUTING_RULE,
    "current_profile_inventory": sorted(
        corpus_data.CURRENT_PROFILE_PUBLIC_CLASS_MAP
    ),
}
ASSEMBLY_DECISION_RULE = (
    "minimum squared distance across source/profile centroids "
    "mapped to each public class"
)
ASSEMBLY_ENROLLMENT_RULE = (
    "each source/profile centroid pools every eligible "
    "4,096/8,192/16,384 prefix view"
)
PROBE_TOLERANCE = 3e-6

REQUIRED_ARTIFACTS = (
    "state_dict",
    "prototype_bank",
    "prototype_labels",
    "prototype_metadata",
    "real_center",
    "complex_center",
    "feature_mean",
    "feature_std",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _export_source_hashes() -> dict[str, str]:
    paths = {
        "v4/export_current_source_browser_runtime.py":
            Path(__file__).resolve(),
        "v4/current_source_data.py":
            Path(corpus_data.__file__).resolve(),
        "v3/export_v3_browser_weights.py":
            Path(encoder_export.__file__).resolve(),
        "training/time_domain_invariant_patch_preprocess.py":
            Path(td_preprocess.__file__).resolve(),
        "v2/invariant_fusion.py":
            V2 / "invariant_fusion.py",
        "v2/invariant_patch_cnn.py":
            V2 / "invariant_patch_cnn.py",
    }
    return {
        name: _sha256(path)
        for name, path in sorted(paths.items())
    }


def _assert_source_snapshot_unchanged(
    expected: Mapping[str, str],
    observed: Mapping[str, str],
) -> None:
    if dict(expected) == dict(observed):
        return
    changed = sorted(
        key
        for key in set(expected) | set(observed)
        if expected.get(key) != observed.get(key)
    )
    raise RuntimeError(
        "executed source changed during v4 browser export: "
        + ", ".join(changed)
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"refusing to serialize {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            _jsonable(payload),
            handle,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        handle.write("\n")
    os.replace(temporary, path)


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON object {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def reject_sensitive_path(path: str | Path, role: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    parts = [part.lower().replace("_", "-") for part in resolved.parts]
    if (
        any("release" in part for part in parts)
        or any("sealed" in part for part in parts)
        or any(
            "consumed-test" in part or "consumedtest" in part
            for part in parts
        )
    ):
        raise ValueError(
            "v4 browser exporter refuses release, sealed, or consumed-test "
            f"{role} paths: {resolved}"
        )
    return resolved


def validate_empty_output(path: str | Path) -> Path:
    output = reject_sensitive_path(path, "output")
    live_assets = (REPO / "src" / "embedding" / "assets").resolve()
    if output == live_assets or live_assets in output.parents:
        raise ValueError(
            f"refusing to write into the live model asset directory: {output}"
        )
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f"output is not a directory: {output}")
        contents = sorted(item.name for item in output.iterdir())
        if contents:
            raise FileExistsError(
                f"refusing to overwrite non-empty output {output}: "
                f"{contents[:5]}"
            )
    return output


def _plain_artifact_path(
    directory: Path,
    artifacts: Mapping[str, Any],
    key: str,
) -> tuple[Path, str]:
    filename = artifacts.get(key)
    recorded = artifacts.get(f"{key}_sha256")
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
    ):
        raise ValueError(f"fusion artifact {key} must be a plain filename")
    if (
        not isinstance(recorded, str)
        or len(recorded) != 64
        or any(character not in "0123456789abcdef" for character in recorded)
    ):
        raise ValueError(f"fusion artifact {key} has no lowercase SHA-256")
    path = (directory / filename).resolve()
    if path.parent != directory or not path.is_file():
        raise ValueError(f"fusion artifact {key} is missing: {path}")
    actual = _sha256(path)
    if actual != recorded:
        raise ValueError(
            f"fusion artifact {key} SHA mismatch: {actual} != {recorded}"
        )
    return path, actual


def _load_float32_array(
    path: Path,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load {name}") from exc
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be an NPY array")
    if value.dtype != np.float32:
        raise ValueError(f"{name} must be float32, got {value.dtype}")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


def _load_state(path: Path) -> Mapping[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older torch compatibility
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, Mapping):
        raise ValueError("fusion_state_dict.pt must contain a mapping")
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError("fusion state dict keys/tensors are invalid")
        if (
            (value.is_floating_point() or value.is_complex())
            and not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"fusion state tensor {name!r} is non-finite")
    return state


def _validate_development_metrics(metrics: Mapping[str, Any]) -> None:
    required = {
        "status": "complete",
        "development_only": True,
        "release_evidence": False,
        "sealed_release_data_used": 0,
        "consumed_historical_test_rows_used": 0,
        "consumed_test_rows_exposed": 0,
        "encoder": "fusion",
    }
    for key, expected in required.items():
        if metrics.get(key) != expected:
            raise ValueError(
                f"fusion dev_metrics {key} must be {expected!r}, "
                f"got {metrics.get(key)!r}"
            )
    run = metrics.get("run_configuration")
    if (
        not isinstance(run, Mapping)
        or run.get("schema") != "v4-current-source-development-fusion-v1"
    ):
        raise ValueError("input is not a v4 current-source fusion assembly")
    frontend = metrics.get("frontend")
    if (
        not isinstance(frontend, Mapping)
        or frontend.get("version") != "invariant-patch-time-domain-v1"
        or frontend.get("uses_frequency_transform") is not False
    ):
        raise ValueError("fusion frontend is not the FFT-free v4 frontend")
    audit = metrics.get("data_audit")
    policy = (
        audit.get("preprocessing", {}).get("runtime_bucket_policy", {})
        if isinstance(audit, Mapping)
        else {}
    )
    if (
        not isinstance(policy, Mapping)
        or policy.get("training_prefixes") != list(RUNTIME_INPUT_LENGTHS)
        or policy.get("views_never_exceed_valid_sample_count") is not True
    ):
        raise ValueError("fusion runtime bucket policy is missing or changed")
    fitting = metrics.get("fitting_contract")
    if (
        not isinstance(fitting, Mapping)
        or fitting.get("prototype_grouping") != "(source, profile)"
        or fitting.get("consumed_historical_test_rows_loaded") != 0
        or fitting.get("sealed_release_rows_loaded") != 0
    ):
        raise ValueError("fusion fitting/prototype contract is invalid")


def _validate_architecture(
    metrics: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], int, int, int]:
    architecture = metrics.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError("fusion architecture must be an object")
    real = architecture.get("real")
    complex_config = architecture.get("complex")
    if not isinstance(real, Mapping) or not isinstance(complex_config, Mapping):
        raise ValueError("fusion architecture needs real and complex branches")
    if real.get("encoder") != "real" or complex_config.get("encoder") != "complex":
        raise ValueError("fusion branch encoder identities are invalid")
    for field in (
        "patch_length",
        "patch_count",
        "embed_dim",
        "n_features",
        "patch_dim",
        "hidden",
        "set_pool",
    ):
        if real.get(field) != complex_config.get(field):
            raise ValueError(f"fusion branch architecture mismatch for {field}")
    patch_length = int(real["patch_length"])
    patch_count = int(real["patch_count"])
    embed_dim = int(real["embed_dim"])
    n_features = int(real["n_features"])
    if min(patch_length, patch_count, embed_dim, n_features) <= 0:
        raise ValueError("fusion architecture dimensions must be positive")
    frontend = metrics["frontend"]
    if (
        frontend.get("patch_length") != patch_length
        or frontend.get("patch_count") != patch_count
        or not (0.0 < float(frontend.get("target_frac", 0.0)) <= 1.0)
    ):
        raise ValueError("fusion frontend and architecture disagree")
    return real, complex_config, embed_dim, n_features, patch_length * patch_count


def _validate_profile_bank_metadata(
    value: Mapping[str, Any],
    labels: np.ndarray,
) -> list[dict[str, Any]]:
    if tuple(value.get("classes", ())) != PUBLIC_CLASSES:
        raise ValueError(
            "prototype metadata classes must be the canonical seven classes"
        )
    if value.get("decision_rule") != ASSEMBLY_DECISION_RULE:
        raise ValueError("prototype metadata decision rule changed")
    if value.get("enrollment_view_rule") != ASSEMBLY_ENROLLMENT_RULE:
        raise ValueError("prototype metadata enrollment rule changed")
    recorded_labels = value.get("prototype_public_class_indices")
    if recorded_labels != labels.tolist():
        raise ValueError("prototype metadata labels disagree with NPY labels")
    groups = value.get("groups")
    if not isinstance(groups, list) or len(groups) != len(labels):
        raise ValueError("prototype metadata groups must align with bank rows")
    seen: set[tuple[str, str]] = set()
    seen_current_profiles: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, item in enumerate(groups):
        if not isinstance(item, Mapping):
            raise ValueError(f"prototype group {index} must be an object")
        source = item.get("source")
        profile = item.get("profile")
        public_class = item.get("public_class")
        enrollment_views = item.get("enrollment_views")
        eligible_lengths = item.get("eligible_lengths")
        if (
            source not in {"current", "historical"}
            or not isinstance(profile, str)
            or not profile
        ):
            raise ValueError(
                f"prototype group {index} source must be current or "
                "historical and profile must be nonempty"
            )
        pair = (source, profile)
        if pair in seen:
            raise ValueError(f"duplicate prototype group {source}:{profile}")
        seen.add(pair)
        label = int(labels[index])
        if public_class != PUBLIC_CLASSES[label]:
            raise ValueError(
                f"prototype group {index} public class disagrees with its label"
            )
        if source == "current":
            expected_class = (
                corpus_data.CURRENT_PROFILE_PUBLIC_CLASS_MAP.get(profile)
            )
            if expected_class is None:
                raise ValueError(
                    f"prototype group {index} is not in the canonical current "
                    "profile inventory"
                )
            if public_class != expected_class:
                raise ValueError(
                    f"prototype group {index} current profile {profile} "
                    f"must map to {expected_class}"
                )
            seen_current_profiles.add(profile)
        if (
            not isinstance(enrollment_views, int)
            or isinstance(enrollment_views, bool)
            or enrollment_views <= 0
        ):
            raise ValueError(
                f"prototype group {index} enrollment_views is invalid"
            )
        if (
            not isinstance(eligible_lengths, list)
            or not eligible_lengths
            or eligible_lengths != sorted(set(eligible_lengths))
            or any(length not in RUNTIME_INPUT_LENGTHS for length in eligible_lengths)
        ):
            raise ValueError(
                f"prototype group {index} eligible lengths are invalid"
            )
        validated.append(
            {
                "source": source,
                "profile": profile,
                "public_class": public_class,
                "enrollment_views": enrollment_views,
                "eligible_lengths": eligible_lengths,
            }
        )
    expected_current_profiles = set(
        corpus_data.CURRENT_PROFILE_PUBLIC_CLASS_MAP
    )
    if seen_current_profiles != expected_current_profiles:
        missing = sorted(expected_current_profiles - seen_current_profiles)
        extra = sorted(seen_current_profiles - expected_current_profiles)
        raise ValueError(
            "current prototype groups must contain the exact 31-profile "
            f"inventory; missing={missing}, extra={extra}"
        )
    return validated


def load_verified_fusion(directory: str | Path) -> dict[str, Any]:
    source = reject_sensitive_path(directory, "input")
    if not source.is_dir():
        raise FileNotFoundError(source)
    metrics_path = source / "dev_metrics.json"
    metrics = _load_json(metrics_path)
    _validate_development_metrics(metrics)
    artifacts = metrics.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("fusion metrics artifacts must be an object")
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for key in REQUIRED_ARTIFACTS:
        path, digest = _plain_artifact_path(source, artifacts, key)
        paths[key] = path
        hashes[path.name] = digest

    real_config, complex_config, embed_dim, n_features, packed_length = (
        _validate_architecture(metrics)
    )
    state = _load_state(paths["state_dict"])
    prototype_bank = _load_float32_array(
        paths["prototype_bank"],
        "fusion prototype bank",
    )
    if (
        prototype_bank.ndim != 2
        or prototype_bank.shape[0] < len(PUBLIC_CLASSES)
        or prototype_bank.shape[1] != 2 * embed_dim
    ):
        raise ValueError(
            "fusion prototype bank must be [P, 2*embed_dim] with P >= 7"
        )
    try:
        labels = np.load(paths["prototype_labels"], allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError("cannot load prototype labels") from exc
    if (
        not isinstance(labels, np.ndarray)
        or labels.dtype != np.int64
        or labels.shape != (len(prototype_bank),)
        or np.any(labels < 0)
        or np.any(labels >= len(PUBLIC_CLASSES))
    ):
        raise ValueError("prototype labels must be int64 [P] in [0, 7)")
    missing = [
        PUBLIC_CLASSES[index]
        for index in range(len(PUBLIC_CLASSES))
        if not np.any(labels == index)
    ]
    if missing:
        raise ValueError(f"prototype bank is missing public classes {missing}")
    groups = _validate_profile_bank_metadata(
        _load_json(paths["prototype_metadata"]),
        labels,
    )

    centers = {
        "real": _load_float32_array(
            paths["real_center"], "real center", (embed_dim,)
        ),
        "complex": _load_float32_array(
            paths["complex_center"], "complex center", (embed_dim,)
        ),
    }
    feature_mean = _load_float32_array(
        paths["feature_mean"], "feature mean", (n_features,)
    )
    feature_std = _load_float32_array(
        paths["feature_std"], "feature std", (n_features,)
    )
    if np.any(feature_std <= 0.0):
        raise ValueError("feature std must be strictly positive")
    for name, state_name in (
        ("real", "real_center"),
        ("complex", "complex_center"),
    ):
        state_value = state.get(state_name)
        if (
            not isinstance(state_value, torch.Tensor)
            or state_value.dtype != torch.float32
            or not np.array_equal(state_value.numpy(), centers[name])
        ):
            raise ValueError(f"state dict {state_name} disagrees with NPY center")

    return {
        "directory": source,
        "metrics_path": metrics_path,
        "metrics": metrics,
        "paths": paths,
        "hashes": hashes,
        "state": state,
        "real_config": real_config,
        "complex_config": complex_config,
        "embed_dim": embed_dim,
        "n_features": n_features,
        "packed_length": packed_length,
        "prototype_bank": prototype_bank,
        "prototype_labels": labels,
        "prototype_groups": groups,
        "centers": centers,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
    }


def build_browser_payload(fusion: Mapping[str, Any]) -> dict[str, Any]:
    metrics = fusion["metrics"]
    state = fusion["state"]
    embed_dim = int(fusion["embed_dim"])
    frontend = metrics["frontend"]
    payload = {
        "schema": BROWSER_SCHEMA,
        "schema_version": BROWSER_SCHEMA_VERSION,
        "status": "staging_not_release",
        "development_only": True,
        "runtime_role": "accepted_known_classifier",
        "packed_length": int(fusion["packed_length"]),
        "parameter_count": int(metrics["parameter_count"]),
        "frontend": {
            "version": "invariant-patch-time-domain-v1",
            "patch_length": int(frontend["patch_length"]),
            "patch_count": int(frontend["patch_count"]),
            "target_frac": float(frontend["target_frac"]),
            "feature_count": int(fusion["n_features"]),
            "uses_frequency_transform": False,
            "runtime_input_lengths": list(RUNTIME_INPUT_LENGTHS),
            "runtime_bucket_rule": RUNTIME_BUCKET_RULE,
            "geometry": frontend.get("geometry"),
        },
        "real": encoder_export._real_branch(
            state, fusion["real_config"]
        ),
        "complex": encoder_export._complex_branch(
            state, fusion["complex_config"]
        ),
        "fusion": {
            "real_center": encoder_export._f32_vector(
                state["real_center"], "real_center", (embed_dim,)
            ),
            "complex_center": encoder_export._f32_vector(
                state["complex_center"], "complex_center", (embed_dim,)
            ),
            "alpha_real": encoder_export._scalar(state, "alpha_real"),
            "alpha_complex": encoder_export._scalar(state, "alpha_complex"),
            "weight_real": encoder_export._scalar(state, "weight_real"),
            "eps": encoder_export._scalar(state, "eps"),
        },
        "embedding_normalize_eps": encoder_export.EMBEDDING_NORMALIZE_EPS,
        "feature_standardization": {
            "mean": fusion["feature_mean"].tolist(),
            "std": fusion["feature_std"].tolist(),
            "rule": "float32((raw_feature - training_mean) / training_std)",
        },
        "classification": {
            "classes": list(PUBLIC_CLASSES),
            "prototype_bank": fusion["prototype_bank"].tolist(),
            "prototype_public_class_indices":
                fusion["prototype_labels"].tolist(),
            "prototype_groups": fusion["prototype_groups"],
            "distance": "squared_euclidean",
            "decision_rule": PROFILE_BANK_DECISION_RULE,
            "routing": TRUSTED_SOURCE_ROUTING,
        },
        "rejection": {
            "state": "unset",
            "runtime_behaviour": "closed_set_only_no_abstention",
            "external_policy_required_for_abstention": True,
        },
        "provenance": {
            "source_fusion_schema":
                metrics["run_configuration"]["schema"],
            "source_fusion_dev_metrics_sha256":
                _sha256(fusion["metrics_path"]),
            "source_fusion_artifacts_sha256": dict(
                sorted(fusion["hashes"].items())
            ),
            "source_sha256": metrics.get("source_sha256"),
            "exporter": Path(__file__).name,
            "exporter_sha256": _sha256(Path(__file__).resolve()),
            "encoder_serializer": Path(encoder_export.__file__).name,
            "encoder_serializer_sha256":
                _sha256(Path(encoder_export.__file__).resolve()),
            "profile_inventory_source":
                Path(corpus_data.__file__).name,
            "profile_inventory_source_sha256":
                _sha256(Path(corpus_data.__file__).resolve()),
            "batch_norm_folded": False,
        },
        "release_evidence": False,
    }
    return payload


def minimum_public_class_distances(
    embeddings: np.ndarray,
    prototype_bank: np.ndarray,
    prototype_labels: np.ndarray,
    n_classes: int = len(PUBLIC_CLASSES),
) -> tuple[np.ndarray, np.ndarray]:
    """NumPy reference for the mapped profile-bank browser decision."""
    value = np.asarray(embeddings, dtype=np.float32)
    bank = np.asarray(prototype_bank, dtype=np.float32)
    labels = np.asarray(prototype_labels, dtype=np.int64)
    if (
        value.ndim != 2
        or bank.ndim != 2
        or value.shape[1] != bank.shape[1]
        or labels.shape != (len(bank),)
        or n_classes <= 0
    ):
        raise ValueError("profile-bank distance inputs have invalid shapes")
    prototype_distances = (
        (value[:, None, :] - bank[None, :, :]) ** 2
    ).sum(axis=-1)
    class_distances = np.empty((len(value), n_classes), dtype=np.float32)
    closest = np.empty((len(value), n_classes), dtype=np.int64)
    for class_index in range(n_classes):
        positions = np.flatnonzero(labels == class_index)
        if len(positions) == 0:
            raise ValueError(f"profile bank has no row for class {class_index}")
        within = prototype_distances[:, positions]
        local = within.argmin(axis=1)
        class_distances[:, class_index] = within[
            np.arange(len(value)), local
        ]
        closest[:, class_index] = positions[local]
    return class_distances, closest


def minimum_source_public_class_distances(
    embeddings: np.ndarray,
    prototype_bank: np.ndarray,
    prototype_labels: np.ndarray,
    prototype_sources: Sequence[str],
    prototype_source: str,
    n_classes: int = len(PUBLIC_CLASSES),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mapped distances from exactly one trusted source; never union fallback."""
    if prototype_source not in {"current", "historical"}:
        raise ValueError("prototype_source must be current or historical")
    sources = np.asarray(prototype_sources, dtype=object)
    bank = np.asarray(prototype_bank, dtype=np.float32)
    labels = np.asarray(prototype_labels, dtype=np.int64)
    if sources.shape != (len(bank),) or labels.shape != (len(bank),):
        raise ValueError("prototype source rows must align with bank and labels")
    admitted = sources == prototype_source
    if not np.any(admitted):
        raise ValueError(
            f"prototype bank has no rows for source {prototype_source}"
        )
    value = np.asarray(embeddings, dtype=np.float32)
    if value.ndim != 2 or bank.ndim != 2 or value.shape[1] != bank.shape[1]:
        raise ValueError("profile-bank distance inputs have invalid shapes")
    prototype_distances = (
        (value[:, None, :] - bank[None, :, :]) ** 2
    ).sum(axis=-1)
    class_distances = np.full(
        (len(value), n_classes), np.inf, dtype=np.float32
    )
    closest = np.full((len(value), n_classes), -1, dtype=np.int64)
    supported = np.zeros(n_classes, dtype=np.bool_)
    for class_index in range(n_classes):
        positions = np.flatnonzero(admitted & (labels == class_index))
        if len(positions) == 0:
            continue
        supported[class_index] = True
        within = prototype_distances[:, positions]
        local = within.argmin(axis=1)
        class_distances[:, class_index] = within[
            np.arange(len(value)), local
        ]
        closest[:, class_index] = positions[local]
    if not np.any(supported):
        raise ValueError(
            f"prototype bank source {prototype_source} supports no class"
        )
    return class_distances, closest, supported


def _probe_iq(length: int, variant: int) -> np.ndarray:
    sample = np.arange(length, dtype=np.float64)
    base = (0.031 + 0.013 * variant) * sample
    chirp = (0.7e-6 + 0.2e-6 * variant) * sample * sample
    wobble = (0.12 + 0.03 * variant) * np.sin(
        2.0 * np.pi * sample / (97.0 + 17.0 * variant)
    )
    envelope = (
        0.68
        + 0.19 * np.sin(2.0 * np.pi * sample / (257.0 + 23.0 * variant))
    )
    phase = 2.0 * np.pi * (base + chirp) + wobble
    iq = envelope * np.exp(1j * phase)
    return np.asarray(iq, dtype=np.complex64)


def _load_fusion_module(fusion: Mapping[str, Any]) -> CenteredInvariantFusion:
    real = InvariantPatchCNN(
        InvariantPatchConfig(**dict(fusion["real_config"])).validate()
    )
    complex_branch = InvariantPatchCNN(
        InvariantPatchConfig(**dict(fusion["complex_config"])).validate()
    )
    module = CenteredInvariantFusion(
        real,
        complex_branch,
        fusion["centers"]["real"],
        fusion["centers"]["complex"],
    )
    module.load_state_dict(fusion["state"], strict=True)
    return module.eval()


def build_probe_fixture(fusion: Mapping[str, Any]) -> dict[str, Any]:
    module = _load_fusion_module(fusion)
    prototype_bank = fusion["prototype_bank"]
    prototype_labels = fusion["prototype_labels"]
    prototype_sources = [
        group["source"] for group in fusion["prototype_groups"]
    ]
    cases: list[dict[str, Any]] = []
    for variant, length in enumerate(RUNTIME_INPUT_LENGTHS):
        iq = _probe_iq(length, variant)
        packed, raw_features, context = td_preprocess.preprocess(
            iq,
            patch_length=int(fusion["real_config"]["patch_length"]),
            patch_count=int(fusion["real_config"]["patch_count"]),
            target_frac=float(fusion["metrics"]["frontend"]["target_frac"]),
        )
        standardized = np.asarray(
            (
                np.asarray(raw_features, dtype=np.float32)
                - fusion["feature_mean"]
            )
            / fusion["feature_std"],
            dtype=np.float32,
        )
        x = torch.from_numpy(
            np.asarray(packed, dtype=np.float32)[None, ...]
        )
        feature_tensor = torch.from_numpy(standardized[None, ...])
        with torch.no_grad():
            real_embedding = module.real_branch(x, feature_tensor)
            complex_embedding = module.complex_branch(x, feature_tensor)
            fused_embedding = module(x, feature_tensor)
        fused_numpy = fused_embedding.numpy().astype(np.float32, copy=False)
        for prototype_source in ("current", "historical"):
            class_distances, closest, supported = (
                minimum_source_public_class_distances(
                    fused_numpy,
                    prototype_bank,
                    prototype_labels,
                    prototype_sources,
                    prototype_source,
                )
            )
            winner = int(class_distances[0].argmin())
            cases.append(
                {
                    "name": (
                        f"deterministic-runtime-prefix-{length}-"
                        f"{prototype_source}"
                    ),
                    "valid_sample_count": length,
                    "runtime_input_length": length,
                    "prototype_source": prototype_source,
                    "raw": {
                        "in_phase": iq.real.tolist(),
                        "quadrature": iq.imag.tolist(),
                    },
                    "expected": {
                        "context": {
                            "version": context["version"],
                            "patch_length": context["patch_length"],
                            "patch_count": context["patch_count"],
                            "target_frac": context["target_frac"],
                            "packed_length": context["packed_length"],
                            "uses_frequency_transform": False,
                        },
                        "packed_in_phase": packed[0].tolist(),
                        "packed_quadrature": packed[1].tolist(),
                        "raw_features":
                            np.asarray(
                                raw_features, dtype=np.float32
                            ).tolist(),
                        "standardized_features": standardized.tolist(),
                        "real_embedding": real_embedding.numpy()[0].tolist(),
                        "complex_embedding":
                            complex_embedding.numpy()[0].tolist(),
                        "fused_embedding": fused_numpy[0].tolist(),
                        "squared_public_class_distances":
                            [
                                (
                                    float(distance)
                                    if np.isfinite(distance)
                                    else None
                                )
                                for distance in class_distances[0]
                            ],
                        "closest_prototype_indices": closest[0].tolist(),
                        "supported_public_class_mask": supported.tolist(),
                        "prototype_source": prototype_source,
                        "closed_winner_index": winner,
                        "closed_label": PUBLIC_CLASSES[winner],
                    },
                }
            )
    return {
        "schema": PROBE_SCHEMA,
        "schema_version": PROBE_SCHEMA_VERSION,
        "tolerance": PROBE_TOLERANCE,
        "uses_frequency_transform": False,
        "runtime_input_lengths": list(RUNTIME_INPUT_LENGTHS),
        "classes": list(PUBLIC_CLASSES),
        "cases": cases,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_hashes_at_start = _export_source_hashes()
    fusion = load_verified_fusion(args.fusion)
    output = validate_empty_output(args.output)
    payload = build_browser_payload(fusion)
    payload["provenance"]["export_source_sha256"] = source_hashes_at_start
    fixture = build_probe_fixture(fusion)
    _assert_source_snapshot_unchanged(
        source_hashes_at_start, _export_source_hashes()
    )
    output.mkdir(parents=True, exist_ok=True)
    weights_path = output / WEIGHTS_NAME
    probe_path = output / PROBE_NAME
    _write_json(weights_path, payload)
    _write_json(probe_path, fixture)
    _assert_source_snapshot_unchanged(
        source_hashes_at_start, _export_source_hashes()
    )
    emitted = {
        path.name: {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in (weights_path, probe_path)
    }
    manifest = {
        "schema": EXPORT_MANIFEST_SCHEMA,
        "schema_version": EXPORT_MANIFEST_SCHEMA_VERSION,
        "status": "staging_not_release",
        "development_only": True,
        "source_fusion_dev_metrics_sha256":
            _sha256(fusion["metrics_path"]),
        "export_source_sha256": source_hashes_at_start,
        "emitted": emitted,
        "deterministic_json": True,
        "release_evidence": False,
    }
    _write_json(output / EXPORT_MANIFEST_NAME, manifest)
    print(f"[v4 browser export] wrote {output}", flush=True)
    return {
        "output": output,
        "payload": payload,
        "fixture": fixture,
        "manifest": manifest,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--fusion",
        required=True,
        help="v4 assemble_current_source_fusion.py output directory",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="fresh development browser-output directory",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
