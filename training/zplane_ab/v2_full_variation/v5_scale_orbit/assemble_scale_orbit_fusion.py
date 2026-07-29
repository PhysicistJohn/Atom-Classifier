"""Assemble a leak-safe v5 scale-orbit real/complex fusion candidate.

The two input directories must be independent branches emitted by
``run_scale_orbit_dev.py`` under one identical data, frontend, training, and
software contract.  This assembler reloads only the historical loader's
exposed train/enrollment/development-selection rows, seed 20264101's current
training/enrollment rows, and seed 20262904's adaptive-selection-only rows.

Fitting is deliberately narrow:

* feature moments are rebuilt from merged training views and must reproduce
  both branches;
* real and complex embedding centers are fitted from merged training views;
* fused prototypes are fitted from combined enrollment views, with one
  centroid per ``(source, profile)`` and every eligible runtime length pooled;
* selection rows are scored only.  They never fit a center, feature moment, or
  prototype.

This creates development evidence, not release evidence.  Release, sealed, and
consumed-test paths or provenance are refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
V2 = HERE.parent
ZPLANE = V2.parent
TRAINING = ZPLANE.parent
V4 = V2 / "v4_current_source"
REPO = HERE.parents[3]
for path in (TRAINING, ZPLANE, V2, V4, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import current_source_data as corpus_data  # noqa: E402
import run_scale_orbit_dev as branch_runner  # noqa: E402
import scale_orbit_data  # noqa: E402
import run_invariant_cnn_dev as device_runner  # noqa: E402
import time_domain_invariant_patch_preprocess as td_preprocess  # noqa: E402
import validate_identity_firewall_amendment as firewall_validator  # noqa: E402
import validate_pretraining_amendment as amendment_validator  # noqa: E402
from invariant_fusion import CenteredInvariantFusion  # noqa: E402
from invariant_patch_cnn import (  # noqa: E402
    InvariantPatchCNN,
    InvariantPatchConfig,
)
from train import embed_all  # noqa: E402


ALPHA_REAL = 0.2
ALPHA_COMPLEX = 0.2
DEFAULT_WEIGHT_REAL = 0.5
FUSION_EPS = 1e-12
FUSION_ASSEMBLY_TOLERANCE = 2e-6
DEFAULT_ARTIFACT_REPRODUCTION_TOLERANCE = 1e-4
DEFAULT_MOMENT_TOLERANCE = 1e-7
HISTORICAL_CORPUS = TRAINING / "artifacts" / "signallab-corpus"
AMENDMENT_SHA256 = (
    "3e4098d0475477bf1ba26e79992ca17848c16dd77deb44b7562c4452544003fd"
)
BRANCH_SCHEMA = "v5-scale-orbit-development-training-v1"
FUSION_SCHEMA = "v5-scale-orbit-development-fusion-v1"
COMPOSITE_AUDIT_SCHEMA = (
    "v5-current-service-scale-orbit-composite-audit-v1"
)

REQUIRED_BRANCH_KEYS = frozenset(
    {
        "status",
        "development_only",
        "release_evidence",
        "consumed_historical_test_rows_used",
        "encoder",
        "architecture",
        "parameter_count",
        "run_configuration",
        "training",
        "final_evaluation",
        "data_audit",
        "artifacts",
        "source_sha256",
        "device",
    }
)
REQUIRED_ARTIFACT_FIELDS = {
    "state_dict": "state_dict_sha256",
    "combined_prototypes": "combined_prototypes_sha256",
    "combined_prototype_bank": "combined_prototype_bank_sha256",
    "feature_moments": "feature_moments_sha256",
}
EXPECTED_COMPOSITE_AUDIT_KEYS = frozenset(
    {
        "schema",
        "lineage",
        "amendment_sha256",
        "parent_protocol_sha256",
        "seed_registry_sha256",
        "corpora",
        "roles",
        "rows_by_role",
        "pair_identities_by_role",
        "profile_role_counts",
        "profiles_present",
        "profile_public_class_map",
        "classes_present",
        "public_class_order",
        "scale_factors",
        "runtime_input_lengths",
        "identity_separation",
        "fitting_firewall",
        "development_only",
        "release_evidence",
        "consumed_test_rows_exposed",
    }
)
EXPECTED_CORPUS_BINDING_KEYS = frozenset(
    {
        "eval_seed",
        "manifest_relative_path",
        "manifest_sha256",
        "raw_relative_path",
        "raw_sha256",
        "registry_allocation",
        "statistical_role",
        "allowed_roles",
    }
)
EXPECTED_IDENTITY_AUDIT_KEYS = frozenset(
    {
        "applicable_collision_counts",
        "inapplicable_keys_by_replay",
        "prefix_lengths_hashed",
        "paired_views_deduplicated",
        "all_applicable_collision_counts_zero",
    }
)
EXPECTED_CORPUS_ROLES = {
    "20264101": {
        "eval_seed": 20264101,
        "registry_allocation": "training_scale",
        "statistical_role": "training_only",
        "allowed_roles": ["train", "enrollment"],
    },
    "20262904": {
        "eval_seed": 20262904,
        "registry_allocation": "adaptive_scale_development",
        "statistical_role": "adaptive_development_only",
        "allowed_roles": ["selection"],
    },
}
EXPECTED_ROWS_BY_ROLE = {
    "train": 31 * 40 * 4,
    "enrollment": 31 * 24 * 4,
    "selection": 31 * 64 * 4,
}
EXPECTED_PAIR_IDENTITIES_BY_ROLE = {
    "train": 31 * 40,
    "enrollment": 31 * 24,
    "selection": 31 * 64,
}
EXPECTED_ROLES = {
    "20264101": ["train", "enrollment"],
    "20262904": ["selection"],
}
EXPECTED_PROFILE_ROLE_COUNTS = {
    role: {
        profile: count
        for profile in corpus_data.CURRENT_PROFILES
    }
    for role, count in {
        "train": 40 * 4,
        "enrollment": 24 * 4,
        "selection": 64 * 4,
    }.items()
}
EXPECTED_APPLICABLE_COLLISION_COUNTS = {
    "pair_id": 0,
    "content_sha256": 0,
    "runtime_prefix_sha256": 0,
    "receiver_realization_channel_seed": 0,
    "receiver_realization_seed_when_non_null": 0,
    "cyclic_base_transmitter_identity": 0,
    "cyclic_profile_and_phase_native_sample": 0,
}
EXPECTED_INAPPLICABLE_KEYS_BY_REPLAY = {
    "cyclic": ["profile_and_payload_seed"],
    "one-shot": list(
        amendment_validator.EXPECTED_INAPPLICABLE_ONE_SHOT_KEYS
    ),
}


def _validated_source_share_contract(
    value: Any,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an exact rational object")
    if set(value) != {"numerator", "denominator", "value"}:
        raise ValueError(
            f"{label} must contain numerator, denominator, and value"
        )
    numerator = value["numerator"]
    denominator = value["denominator"]
    decimal = value["value"]
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator <= 0
        or numerator <= 0
        or numerator >= denominator
        or isinstance(decimal, bool)
        or not isinstance(decimal, (int, float))
        or not np.isfinite(float(decimal))
        or float(decimal) != numerator / denominator
    ):
        raise ValueError(f"{label} is not a canonical share in (0, 1)")
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": float(decimal),
    }


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
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            _jsonable(payload),
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    os.replace(temporary, path)


def reject_sensitive_path(path: str | Path, role: str) -> Path:
    """Resolve a path and reject release, sealed, or consumed-test locations."""
    resolved = Path(path).expanduser().resolve()
    components = [part.lower().replace("_", "-") for part in resolved.parts]
    if (
        any("release" in part for part in components)
        or any("sealed" in part for part in components)
        or any(
            "consumed-test" in part or "consumedtest" in part
            for part in components
        )
    ):
        raise ValueError(
            "v5 development fusion refuses release, sealed, or consumed-test "
            f"{role} paths: {resolved}"
        )
    return resolved


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _bound_contract_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"composite audit {field} must be a relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"composite audit {field} escapes the repository")
    resolved = (REPO / relative).resolve()
    try:
        resolved.relative_to(REPO)
    except ValueError as exc:
        raise ValueError(
            f"composite audit {field} escapes the repository"
        ) from exc
    return resolved


def validate_composite_current_audit(
    value: Any,
    *,
    current_directory: Path | None = None,
    selection_directory: Path | None = None,
) -> Mapping[str, Any]:
    """Enforce the frozen two-corpus amendment audit before fusion work."""
    if not isinstance(value, Mapping):
        raise ValueError("branch current audit must be an object")
    if set(value) != EXPECTED_COMPOSITE_AUDIT_KEYS:
        raise ValueError("branch current audit keys differ from the amendment")
    if (
        value.get("schema") != COMPOSITE_AUDIT_SCHEMA
        or value.get("lineage") != "v5-scale-orbit"
        or value.get("amendment_sha256") != AMENDMENT_SHA256
        or value.get("parent_protocol_sha256")
        != amendment_validator.EXPECTED_PROTOCOL_SHA256
        or value.get("seed_registry_sha256")
        != amendment_validator.EXPECTED_REGISTRY_SHA256
        or value.get("development_only") is not True
        or value.get("release_evidence") is not False
        or value.get("consumed_test_rows_exposed") != 0
    ):
        raise ValueError("branch current audit lineage/firewall binding changed")

    corpora = value.get("corpora")
    if not isinstance(corpora, Mapping) or set(corpora) != set(
        EXPECTED_CORPUS_ROLES
    ):
        raise ValueError("branch current audit corpus seeds changed")
    expected_directories = {
        "20264101": current_directory,
        "20262904": selection_directory,
    }
    for seed_text, expected in EXPECTED_CORPUS_ROLES.items():
        binding = corpora.get(seed_text)
        if not isinstance(binding, Mapping) or set(binding) != (
            EXPECTED_CORPUS_BINDING_KEYS
        ):
            raise ValueError(
                f"branch current audit corpus {seed_text} binding changed"
            )
        for field, expected_value in expected.items():
            if binding.get(field) != expected_value:
                raise ValueError(
                    f"branch current audit corpus {seed_text} {field} changed"
                )
        for field in ("manifest_sha256", "raw_sha256"):
            if not _is_sha256(binding.get(field)):
                raise ValueError(
                    f"branch current audit corpus {seed_text} {field} is invalid"
                )
        manifest = _bound_contract_path(
            binding.get("manifest_relative_path"),
            f"corpora.{seed_text}.manifest_relative_path",
        )
        raw = _bound_contract_path(
            binding.get("raw_relative_path"),
            f"corpora.{seed_text}.raw_relative_path",
        )
        if manifest.parent != raw.parent:
            raise ValueError(
                f"branch current audit corpus {seed_text} paths disagree"
            )
        expected_directory = expected_directories[seed_text]
        if (
            expected_directory is not None
            and manifest.parent != expected_directory.resolve()
        ):
            raise ValueError(
                f"branch current audit corpus {seed_text} path does not match "
                "the supplied branch argument"
            )

    if value.get("roles") != EXPECTED_ROLES:
        raise ValueError("branch current audit seed/role map changed")
    if value.get("rows_by_role") != EXPECTED_ROWS_BY_ROLE:
        raise ValueError("branch current audit stored-row counts changed")
    if (
        value.get("pair_identities_by_role")
        != EXPECTED_PAIR_IDENTITIES_BY_ROLE
    ):
        raise ValueError("branch current audit pair-identity counts changed")
    if value.get("profile_role_counts") != EXPECTED_PROFILE_ROLE_COUNTS:
        raise ValueError("branch current audit profile/role counts changed")
    if value.get("profiles_present") != list(corpus_data.CURRENT_PROFILES):
        raise ValueError("branch current audit profile inventory changed")
    if (
        value.get("profile_public_class_map")
        != dict(corpus_data.CURRENT_PROFILE_PUBLIC_CLASS_MAP)
    ):
        raise ValueError("branch current audit profile/class mapping changed")
    if value.get("classes_present") != list(corpus_data.CURRENT_CLASSES):
        raise ValueError("branch current audit current classes changed")
    if value.get("public_class_order") != list(
        scale_orbit_data.scale_data.PUBLIC_CLASSES
    ):
        raise ValueError("branch current audit public class order changed")
    if value.get("scale_factors") != list(
        scale_orbit_data.SCALE_ORBIT_EXPECTED_FACTORS
    ):
        raise ValueError("branch current audit scale factors changed")
    if value.get("runtime_input_lengths") != list(
        corpus_data.RUNTIME_INPUT_LENGTHS
    ):
        raise ValueError("branch current audit runtime input lengths changed")

    identity = value.get("identity_separation")
    if not isinstance(identity, Mapping) or set(identity) != (
        EXPECTED_IDENTITY_AUDIT_KEYS
    ):
        raise ValueError("branch current audit identity schema changed")
    if (
        identity.get("applicable_collision_counts")
        != EXPECTED_APPLICABLE_COLLISION_COUNTS
        or identity.get("prefix_lengths_hashed")
        != list(corpus_data.RUNTIME_INPUT_LENGTHS)
        or identity.get("paired_views_deduplicated") is not True
        or identity.get("all_applicable_collision_counts_zero") is not True
    ):
        raise ValueError("branch current audit identity separation failed")
    inapplicable = identity.get("inapplicable_keys_by_replay")
    if inapplicable != EXPECTED_INAPPLICABLE_KEYS_BY_REPLAY:
        raise ValueError("branch current audit one-shot disclosure changed")
    if value.get("fitting_firewall") != amendment_validator.EXPECTED_FIREWALL:
        raise ValueError("branch current audit fitting firewall changed")
    return value


def assert_branch_corpus_arguments(
    arguments: Mapping[str, Any],
    *,
    current_directory: Path,
    selection_directory: Path,
    historical_directory: Path,
) -> None:
    """Require branch-recorded inputs to equal all supplied assembler inputs."""
    expected = {
        "current_corpus": current_directory.resolve(),
        "current_selection_corpus": selection_directory.resolve(),
        "historical_corpus": historical_directory.resolve(),
    }
    for field, expected_path in expected.items():
        if field not in arguments:
            raise ValueError(f"branch arguments are missing {field}")
        recorded = reject_sensitive_path(
            arguments[field], f"recorded branch {field}"
        )
        if recorded != expected_path:
            raise ValueError(
                f"branch argument {field} does not match the supplied "
                "assembler corpus"
            )


def validate_branch_weight(value: Any) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("branch weight must be a finite scalar") from exc
    if not np.isfinite(weight):
        raise ValueError("branch weight must be a finite scalar")
    if not 0.0 <= weight <= 1.0:
        raise ValueError("branch weight must lie in [0, 1]")
    return weight


def _positive_tolerance(value: Any, name: str) -> float:
    try:
        tolerance = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be positive and finite") from exc
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return tolerance


def _safe_unit(value: np.ndarray, eps: float = FUSION_EPS) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("fusion embeddings must be a finite [N, D] array")
    norm = np.sqrt(np.sum(array * array, axis=1, keepdims=True))
    output = array / np.maximum(norm, np.float32(eps))
    cancelled = norm[:, 0] <= eps
    if cancelled.any():
        output[cancelled] = 0.0
        output[cancelled, 0] = 1.0
    return output.astype(np.float32, copy=False)


def fuse_numpy(
    real: np.ndarray,
    complex_embedding: np.ndarray,
    real_center: np.ndarray,
    complex_center: np.ndarray,
    *,
    weight_real: float,
) -> np.ndarray:
    """Mirror ``CenteredInvariantFusion`` exactly enough for assembly checks."""
    real_value = np.asarray(real, dtype=np.float32)
    complex_value = np.asarray(complex_embedding, dtype=np.float32)
    if real_value.shape != complex_value.shape or real_value.ndim != 2:
        raise ValueError("real and complex embeddings must have equal [N, D] shapes")
    if not np.isfinite(real_value).all() or not np.isfinite(complex_value).all():
        raise ValueError("real and complex embeddings must be finite")
    real_center_value = np.asarray(real_center, dtype=np.float32)
    complex_center_value = np.asarray(complex_center, dtype=np.float32)
    expected_center_shape = (real_value.shape[1],)
    if (
        real_center_value.shape != expected_center_shape
        or complex_center_value.shape != expected_center_shape
        or not np.isfinite(real_center_value).all()
        or not np.isfinite(complex_center_value).all()
    ):
        raise ValueError(
            f"fusion centers must be finite vectors {expected_center_shape}"
        )
    weight = validate_branch_weight(weight_real)
    centered_real = _safe_unit(
        real_value - np.float32(ALPHA_REAL) * real_center_value
    )
    centered_complex = _safe_unit(
        complex_value - np.float32(ALPHA_COMPLEX) * complex_center_value
    )
    return _safe_unit(
        np.concatenate(
            (
                np.sqrt(np.float32(weight)) * centered_real,
                np.sqrt(np.float32(1.0 - weight)) * centered_complex,
            ),
            axis=1,
        )
    )


def fit_center(branch_embeddings: Mapping[str, np.ndarray]) -> np.ndarray:
    """Fit a branch center from the ``training`` population and nothing else."""
    if "training" not in branch_embeddings:
        raise KeyError("center fitting requires the 'training' population")
    training = np.asarray(branch_embeddings["training"], dtype=np.float64)
    if (
        training.ndim != 2
        or training.shape[0] == 0
        or not np.isfinite(training).all()
    ):
        raise ValueError("training embeddings must be a nonempty finite [N, D] array")
    return training.mean(axis=0).astype(np.float32)


class FusionEmbedder(torch.nn.Module):
    """Expose a centered fusion through ``embed_all``'s branch interface."""

    def __init__(self, fusion: CenteredInvariantFusion):
        super().__init__()
        if not isinstance(fusion, CenteredInvariantFusion):
            raise TypeError("fusion must be a CenteredInvariantFusion")
        self.fusion = fusion

    @property
    def cfg(self) -> InvariantPatchConfig:
        return self.fusion.real_branch.cfg

    def forward(self, x: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        return self.fusion(x, feat)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_path(directory: Path, filename: Any, field: str) -> Path:
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
    ):
        raise ValueError(f"branch artifact {field} must be a plain filename")
    path = (directory / filename).resolve()
    if path.parent != directory or not path.is_file():
        raise ValueError(f"branch artifact {field} is missing: {path}")
    return path


def _validate_no_consumed_provenance(value: Any, *, path: str = "metrics") -> None:
    """Recursively fail on a nonzero consumed/sealed/test exposure counter."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower().replace("-", "_")
            item_path = f"{path}.{key}"
            # A positive count of rows deliberately kept behind the consumed
            # historical-test boundary proves non-exposure; it is not itself
            # an exposure counter.  Keep this exception exact so a similarly
            # named "used"/"loaded"/"exposed" field still fails closed.
            safe_withheld_count = key_text == "consumed_test_held_out"
            consumed_test_exposure = (
                "consumed" in key_text
                and (
                    "test" in key_text
                    or "validation" in key_text
                    or key_text.endswith((
                        "_rows_used",
                        "_rows_loaded",
                        "_rows_exposed",
                    ))
                )
            )
            sensitive = (
                not safe_withheld_count
                and (
                    consumed_test_exposure
                    or "sealed" in key_text
                    or key_text in {
                        "test_rows_used",
                        "test_rows_loaded",
                        "test_rows_exposed",
                    }
                )
            )
            if sensitive and (
                item is True
                or (
                    isinstance(item, (int, float, np.integer, np.floating))
                    and not isinstance(item, (bool, np.bool_))
                    and float(item) != 0.0
                )
            ):
                raise RuntimeError(
                    f"branch provenance reports forbidden exposure at {item_path}"
                )
            _validate_no_consumed_provenance(item, path=item_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_no_consumed_provenance(item, path=f"{path}[{index}]")


def validate_branch_metrics(
    payload: Mapping[str, Any],
    *,
    encoder: str,
) -> Mapping[str, Any]:
    """Validate the public branch schema and its development-only provenance."""
    if not isinstance(payload, Mapping):
        raise ValueError("branch dev_metrics.json must be an object")
    missing = sorted(REQUIRED_BRANCH_KEYS - set(payload))
    if missing:
        raise ValueError(f"branch dev_metrics.json is missing keys {missing}")
    expected = {
        "status": "complete",
        "development_only": True,
        "release_evidence": False,
        "consumed_historical_test_rows_used": 0,
        "encoder": encoder,
    }
    for key, required in expected.items():
        if payload.get(key) != required:
            raise ValueError(
                f"branch dev_metrics.json {key} must be {required!r}, "
                f"got {payload.get(key)!r}"
            )
    if payload.get("sealed_release_data_used", 0) != 0:
        raise ValueError("branch reports sealed release data use")
    data_audit = payload.get("data_audit")
    if not isinstance(data_audit, Mapping):
        raise ValueError("branch data_audit must be an object")
    if data_audit.get("consumed_test_rows_exposed", 0) != 0:
        raise ValueError("branch data audit exposed consumed historical test rows")
    historical = data_audit.get("historical")
    if not isinstance(historical, Mapping):
        raise ValueError("branch data audit lacks historical provenance")
    if historical.get("consumed_test_rows_exposed") != 0:
        raise ValueError("branch historical loader exposed consumed test rows")
    validate_composite_current_audit(data_audit.get("current"))
    run_configuration = payload.get("run_configuration")
    if not isinstance(run_configuration, Mapping):
        raise ValueError("branch run_configuration must be an object")
    if run_configuration.get("schema") != BRANCH_SCHEMA:
        raise ValueError("branch run_configuration is not the exact v5 run")
    checkpoint = run_configuration.get("checkpoint_selection")
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("contract") != branch_runner.CHECKPOINT_SCORE_CONTRACT
        or checkpoint.get("predeclared") is not True
        or checkpoint.get("tunable") is not False
    ):
        raise ValueError("branch checkpoint-selection contract changed")
    arguments = run_configuration.get("arguments")
    randomness = run_configuration.get("randomness")
    training = payload.get("training")
    if (
        not isinstance(arguments, Mapping)
        or arguments.get("encoder") != encoder
        or not isinstance(randomness, Mapping)
        or randomness.get("seed") != arguments.get("seed")
        or not isinstance(training, Mapping)
        or training.get("episodes") != arguments.get("episodes")
        or training.get("sampler_contract") != branch_runner.SAMPLER_CONTRACT
    ):
        raise ValueError("branch arguments/randomness/sampler contract is inconsistent")
    source_share = _validated_source_share_contract(
        arguments.get("current_source_share"),
        "branch arguments current_source_share",
    )
    if training.get("current_source_share") != source_share:
        raise ValueError(
            "branch training current-source share disagrees with arguments"
        )
    preprocessing = data_audit.get("preprocessing")
    if not isinstance(preprocessing, Mapping):
        raise ValueError("branch lacks preprocessing audit")
    frontend = preprocessing.get("frontend")
    if frontend != td_preprocess.preprocess_metadata():
        raise ValueError("branch frontend does not match the current FFT-free frontend")
    if frontend.get("uses_frequency_transform") is not False:
        raise ValueError("branch frontend is not FFT-free")
    preprocessing_training = preprocessing.get("training")
    runtime_policy = preprocessing.get("runtime_bucket_policy")
    if (
        not isinstance(preprocessing_training, Mapping)
        or preprocessing_training.get("hierarchical_sampler_contract")
        != branch_runner.SAMPLER_CONTRACT
        or not isinstance(runtime_policy, Mapping)
        or runtime_policy.get("episode_rule") != branch_runner.SAMPLER_CONTRACT
    ):
        raise ValueError("branch preprocessing sampler contract changed")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("branch artifacts must be an object")
    for filename_field, hash_field in REQUIRED_ARTIFACT_FIELDS.items():
        if filename_field not in artifacts or hash_field not in artifacts:
            raise ValueError(
                f"branch artifacts must declare {filename_field} and {hash_field}"
            )
    _validate_no_consumed_provenance(payload)
    return payload


def _load_moments(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        archive = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read feature moments {path}") from exc
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError("branch feature_moments must be an NPZ with mean and std")
    try:
        if set(archive.files) != {"mean", "std"}:
            raise ValueError(
                "branch feature_moments must contain exactly mean and std"
            )
        mean = np.asarray(archive["mean"], dtype=np.float32).copy()
        std = np.asarray(archive["std"], dtype=np.float32).copy()
    finally:
        archive.close()
    if (
        mean.ndim != 1
        or std.shape != mean.shape
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or np.any(std <= 0.0)
    ):
        raise ValueError("branch feature moments must be finite vectors with std > 0")
    return mean, std


def _load_state_dict(path: Path) -> Mapping[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - old PyTorch compatibility
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, Mapping):
        raise ValueError("branch state_dict artifact is not a mapping")
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"branch state_dict value {name!r} is not a tensor")
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            raise ValueError(f"branch state_dict value {name!r} is non-finite")
    return state


def load_branch(directory: str | Path, encoder: str) -> dict[str, Any]:
    """Load one v5 branch, checking every recorded artifact digest."""
    if encoder not in {"real", "complex"}:
        raise ValueError("encoder must be real or complex")
    source = reject_sensitive_path(directory, "branch input")
    if not source.is_dir():
        raise FileNotFoundError(source)
    metrics_path = source / "dev_metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    with metrics_path.open(encoding="utf-8") as handle:
        metrics = validate_branch_metrics(json.load(handle), encoder=encoder)
    artifacts = metrics["artifacts"]
    artifact_paths: dict[str, Path] = {}
    artifact_hashes: dict[str, str] = {}
    for filename_field, hash_field in REQUIRED_ARTIFACT_FIELDS.items():
        artifact_path = _artifact_path(
            source, artifacts[filename_field], filename_field
        )
        actual = _sha256(artifact_path)
        expected = artifacts[hash_field]
        if not isinstance(expected, str) or actual != expected:
            raise RuntimeError(
                f"{encoder} artifact {artifact_path.name} does not match its "
                "recorded SHA-256"
            )
        artifact_paths[filename_field] = artifact_path
        artifact_hashes[artifact_path.name] = actual

    try:
        config = InvariantPatchConfig(**metrics["architecture"]).validate()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{encoder} branch architecture is invalid") from exc
    if config.encoder != encoder:
        raise ValueError(f"{encoder} branch architecture encoder disagrees")
    net = InvariantPatchCNN(config)
    net.load_state_dict(
        _load_state_dict(artifact_paths["state_dict"]),
        strict=True,
    )
    prototypes = np.load(
        artifact_paths["combined_prototypes"], allow_pickle=False
    )
    prototypes = np.asarray(prototypes, dtype=np.float32)
    if prototypes.ndim != 2 or not np.isfinite(prototypes).all():
        raise ValueError(f"{encoder} stored prototype bank is invalid")
    with artifact_paths["combined_prototype_bank"].open(
        encoding="utf-8"
    ) as handle:
        prototype_contract = json.load(handle)
    if not isinstance(prototype_contract, dict):
        raise ValueError(f"{encoder} prototype contract must be an object")
    moments = _load_moments(artifact_paths["feature_moments"])
    return {
        "directory": source,
        "metrics": metrics,
        "encoder": encoder,
        "net": net.eval(),
        "stored_prototypes": prototypes,
        "stored_prototype_contract": prototype_contract,
        "moments": moments,
        "hashes": {
            "dev_metrics.json": _sha256(metrics_path),
            **artifact_hashes,
        },
    }


def _without_encoder(arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in arguments.items()
        if str(key) != "encoder"
    }


def _training_contract(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Extract immutable sampler/checkpoint terms, not learned outcomes."""
    training = metrics["training"]
    if not isinstance(training, Mapping):
        raise ValueError("branch training report must be an object")
    contract_keys = sorted(
        key
        for key in training
        if (
            key == "episodes"
            or "contract" in key
            or key in {"k_shot", "q_query", "hierarchy"}
        )
    )
    contract = {key: training[key] for key in contract_keys}
    if "episodes" not in contract or not any(
        "sampler" in key for key in contract
    ):
        raise ValueError("branch training report lacks episode/sampler contract")
    return contract


def assert_branches_match(
    real: Mapping[str, Any],
    complex_branch: Mapping[str, Any],
) -> dict[str, Any]:
    """Require one identical source/corpus/frontend/config/training contract."""
    real_metrics = real["metrics"]
    complex_metrics = complex_branch["metrics"]
    real_architecture = dict(real_metrics["architecture"])
    complex_architecture = dict(complex_metrics["architecture"])
    if real_architecture.pop("encoder", None) != "real":
        raise ValueError("real branch architecture encoder changed")
    if complex_architecture.pop("encoder", None) != "complex":
        raise ValueError("complex branch architecture encoder changed")
    if real_architecture != complex_architecture:
        raise ValueError("branch architecture/config contracts disagree")

    real_run = real_metrics["run_configuration"]
    complex_run = complex_metrics["run_configuration"]
    for field in (
        "schema",
        "checkpoint_selection",
        "optimizer",
        "scheduler",
        "randomness",
        "software",
    ):
        if real_run.get(field) != complex_run.get(field):
            raise ValueError(f"branch run-configuration {field} disagrees")
    real_arguments = real_run.get("arguments")
    complex_arguments = complex_run.get("arguments")
    if not isinstance(real_arguments, Mapping) or not isinstance(
        complex_arguments, Mapping
    ):
        raise ValueError("branch run-configuration arguments must be objects")
    if _without_encoder(real_arguments) != _without_encoder(complex_arguments):
        raise ValueError("branch training argument contracts disagree")

    for field in ("source_sha256", "data_audit"):
        if real_metrics.get(field) != complex_metrics.get(field):
            raise ValueError(f"branch {field} contracts disagree")
    real_training_contract = _training_contract(real_metrics)
    complex_training_contract = _training_contract(complex_metrics)
    if real_training_contract != complex_training_contract:
        raise ValueError("branch sampler/training contracts disagree")
    return {
        "architecture_without_encoder": real_architecture,
        "arguments_without_encoder": _without_encoder(real_arguments),
        "training_contract": real_training_contract,
        "source_sha256": real_metrics["source_sha256"],
        "data_audit_sha256": corpus_data.sha256_json(real_metrics["data_audit"]),
    }


def _max_abs_error(
    left: np.ndarray,
    right: np.ndarray,
    *,
    name: str,
) -> float:
    left_value = np.asarray(left)
    right_value = np.asarray(right)
    if left_value.shape != right_value.shape:
        raise RuntimeError(
            f"{name} shape mismatch: {left_value.shape} != {right_value.shape}"
        )
    if not np.isfinite(left_value).all() or not np.isfinite(right_value).all():
        raise RuntimeError(f"{name} contains non-finite values")
    return float(
        np.max(np.abs(left_value.astype(np.float64) - right_value.astype(np.float64)))
    )


def _source_hashes() -> dict[str, str]:
    paths = {
        **branch_runner._executed_source_paths(),
        "v5/assemble_scale_orbit_fusion.py": Path(__file__).resolve(),
        "v5/pretraining_amendment.json":
            amendment_validator.AMENDMENT_PATH.resolve(),
        "v5/recovery_protocol.json":
            amendment_validator.PROTOCOL_PATH.resolve(),
        "v5/seed_registry.json": amendment_validator.REGISTRY_PATH.resolve(),
        "v5/validate_pretraining_amendment.py":
            Path(amendment_validator.__file__).resolve(),
        "v5/identity_firewall_amendment.json":
            firewall_validator.AMENDMENT_PATH.resolve(),
        "v5/seed20262904_identity_rejection.json":
            firewall_validator.REJECTION_PATH.resolve(),
        "v5/seed20262904_identity_firewall_acceptance.json":
            scale_orbit_data.SCALE_ORBIT_IDENTITY_ACCEPTANCE_PATH.resolve(),
        "v5/validate_identity_firewall_amendment.py":
            Path(firewall_validator.__file__).resolve(),
        "v2/invariant_fusion.py": V2 / "invariant_fusion.py",
    }
    return {
        name: _sha256(path)
        for name, path in sorted(paths.items())
    }


def _assert_source_snapshot_unchanged(
    expected: Mapping[str, str],
    observed: Mapping[str, str],
    *,
    operation: str,
) -> None:
    if dict(expected) == dict(observed):
        return
    changed = sorted(
        key
        for key in set(expected) | set(observed)
        if expected.get(key) != observed.get(key)
    )
    raise RuntimeError(
        f"executed source changed during {operation}: {', '.join(changed)}"
    )


def _expected_branch_source_hashes() -> dict[str, str]:
    """Rebuild the exact source-hash map emitted by the v5 branch runner."""
    return branch_runner._source_hashes()


def _reproduce_branch_prototypes(
    branch: Mapping[str, Any],
    data: Mapping[str, Any],
    device: torch.device,
    *,
    tolerance: float,
) -> dict[str, Any]:
    net = branch["net"].to(device).eval()
    enrollment = embed_all(net, data["xen"], data["fen"], device)
    prototypes, labels, metadata = branch_runner._profile_prototype_bank(
        enrollment,
        np.asarray(data["yen"], dtype=np.int64),
        data["enrollment_sources"],
        data["enrollment_profiles"],
        data["enrollment_lengths"],
        data["classes"],
    )
    error = _max_abs_error(
        prototypes,
        branch["stored_prototypes"],
        name=f"{branch['encoder']} branch prototype reproduction",
    )
    if error > tolerance:
        raise RuntimeError(
            f"{branch['encoder']} branch prototypes did not reproduce: "
            f"{error:g} > {tolerance:g}"
        )
    expected_contract = {
        "decision_rule": (
            "minimum squared distance across source/profile centroids "
            "mapped to each public class"
        ),
        "enrollment_view_rule": (
            "each source/profile centroid pools every eligible route "
            "view; historical rows retain eligible prefix views and "
            "current rows contribute canonicalized physical-scale views"
        ),
        "groups": metadata,
        "prototype_public_class_indices": labels.tolist(),
    }
    if branch["stored_prototype_contract"] != expected_contract:
        raise RuntimeError(
            f"{branch['encoder']} branch prototype metadata did not reproduce"
        )
    return {
        "enrollment_embeddings": enrollment.astype(np.float32, copy=False),
        "prototype_reproduction_max_abs_error": error,
        "prototype_labels": labels,
        "prototype_metadata": metadata,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    if _sha256(amendment_validator.AMENDMENT_PATH) != AMENDMENT_SHA256:
        raise RuntimeError("frozen pre-training amendment bytes changed")
    amendment_validation = (
        amendment_validator.validate_repository_amendment()
    )
    identity_firewall_validation = firewall_validator.validate_repository()
    source_hashes_at_start = _source_hashes()
    expected_branch_sources_at_start = _expected_branch_source_hashes()
    output = reject_sensitive_path(args.output_dir, "output")
    real_dir = reject_sensitive_path(args.real_dir, "branch input")
    complex_dir = reject_sensitive_path(args.complex_dir, "branch input")
    current_dir = reject_sensitive_path(args.current_corpus, "current corpus")
    selection_dir = reject_sensitive_path(
        args.current_selection_corpus,
        "current adaptive-selection corpus",
    )
    historical_dir = reject_sensitive_path(
        args.historical_corpus, "historical corpus"
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    if real_dir == complex_dir:
        raise ValueError("real and complex branch directories must differ")
    if len({current_dir, selection_dir, historical_dir}) != 3:
        raise ValueError(
            "training/enrollment, adaptive-selection, and historical corpus "
            "directories must all differ"
        )
    weight_real = validate_branch_weight(args.branch_weight)
    artifact_tolerance = _positive_tolerance(
        args.artifact_tolerance, "artifact tolerance"
    )
    moment_tolerance = _positive_tolerance(
        args.moment_tolerance, "moment tolerance"
    )

    branches = {
        "real": load_branch(real_dir, "real"),
        "complex": load_branch(complex_dir, "complex"),
    }
    matched_contract = assert_branches_match(
        branches["real"], branches["complex"]
    )
    if (
        branches["real"]["metrics"]["source_sha256"]
        != expected_branch_sources_at_start
    ):
        raise RuntimeError(
            "branch source hashes do not reproduce from the current v5 source "
            "tree"
        )
    branch_arguments = branches["real"]["metrics"]["run_configuration"][
        "arguments"
    ]
    assert_branch_corpus_arguments(
        branch_arguments,
        current_directory=current_dir,
        selection_directory=selection_dir,
        historical_directory=historical_dir,
    )
    seed = int(branch_arguments["seed"])
    configuration = branches["real"]["net"].cfg
    target_frac = float(branch_arguments["target_frac"])
    if (
        int(branch_arguments["patch_length"]) != configuration.patch_length
        or int(branch_arguments["patch_count"]) != configuration.patch_count
    ):
        raise ValueError("branch architecture and preprocessing arguments disagree")

    historical, historical_contract = corpus_data.load_historical_exposed(
        historical_dir,
        patch_length=configuration.patch_length,
        patch_count=configuration.patch_count,
        target_frac=target_frac,
        seed=seed,
    )
    classes = list(historical_contract["classes"])
    current = scale_orbit_data.load_scale_orbit_training_corpus(
        current_dir,
        selection_directory=selection_dir,
        class_index={name: index for index, name in enumerate(classes)},
    )
    validate_composite_current_audit(
        current.audit,
        current_directory=current_dir,
        selection_directory=selection_dir,
    )
    data, preprocessing_audit = branch_runner._prepare_data(
        historical,
        current,
        classes=classes,
        patch_length=configuration.patch_length,
        patch_count=configuration.patch_count,
        target_frac=target_frac,
    )
    if historical.audit.get("consumed_test_rows_exposed") != 0:
        raise AssertionError("historical loader exposed consumed test rows")
    rebuilt_audit = {
        "historical": historical.audit,
        "current": current.audit,
        "preprocessing": preprocessing_audit,
        "historical_contract": historical_contract,
        "consumed_test_rows_exposed": 0,
    }
    for name, branch in branches.items():
        if branch["metrics"]["data_audit"] != rebuilt_audit:
            raise RuntimeError(
                f"{name} branch source/corpus/preprocessing audit does not "
                "reproduce from the supplied corpora"
            )

    rebuilt_mean = np.asarray(data["fmean"], dtype=np.float32)
    rebuilt_std = np.asarray(data["fstd"], dtype=np.float32)
    moment_errors: dict[str, dict[str, float]] = {}
    for name, branch in branches.items():
        mean, std = branch["moments"]
        errors = {
            "mean_max_abs_error": _max_abs_error(
                mean, rebuilt_mean, name=f"{name} feature mean"
            ),
            "std_max_abs_error": _max_abs_error(
                std, rebuilt_std, name=f"{name} feature std"
            ),
        }
        if max(errors.values()) > moment_tolerance:
            raise RuntimeError(
                f"{name} feature moments did not reproduce within "
                f"{moment_tolerance:g}: {errors}"
            )
        moment_errors[name] = errors
    cross_mean_error = _max_abs_error(
        branches["real"]["moments"][0],
        branches["complex"]["moments"][0],
        name="cross-branch feature mean",
    )
    cross_std_error = _max_abs_error(
        branches["real"]["moments"][1],
        branches["complex"]["moments"][1],
        name="cross-branch feature std",
    )
    if max(cross_mean_error, cross_std_error) > moment_tolerance:
        raise RuntimeError("real and complex branch feature moments disagree")

    device = device_runner.resolve_device(args.device)
    device_runner.seed_everything(seed)
    for branch in branches.values():
        branch["net"] = branch["net"].to(device).eval()
    branch_reproduction = {
        name: _reproduce_branch_prototypes(
            branch,
            data,
            device,
            tolerance=artifact_tolerance,
        )
        for name, branch in branches.items()
    }

    branch_training_embeddings = {
        name: embed_all(
            branch["net"], data["xtr"], data["ftr"], device
        ).astype(np.float32, copy=False)
        for name, branch in branches.items()
    }
    centers = {
        name: fit_center({"training": embeddings})
        for name, embeddings in branch_training_embeddings.items()
    }
    fusion = (
        CenteredInvariantFusion(
            branches["real"]["net"],
            branches["complex"]["net"],
            centers["real"],
            centers["complex"],
            alpha_real=ALPHA_REAL,
            alpha_complex=ALPHA_COMPLEX,
            weight_real=weight_real,
            eps=FUSION_EPS,
        )
        .to(device)
        .eval()
    )
    embedder = FusionEmbedder(fusion).to(device).eval()

    fused_enrollment = fuse_numpy(
        branch_reproduction["real"]["enrollment_embeddings"],
        branch_reproduction["complex"]["enrollment_embeddings"],
        centers["real"],
        centers["complex"],
        weight_real=weight_real,
    )
    prototype_bank, prototype_labels, prototype_metadata = (
        branch_runner._profile_prototype_bank(
            fused_enrollment,
            np.asarray(data["yen"], dtype=np.int64),
            data["enrollment_sources"],
            data["enrollment_profiles"],
            data["enrollment_lengths"],
            data["classes"],
        )
    )
    final_evaluation, evaluated_prototype_bank = branch_runner._evaluate(
        embedder, data, device
    )
    prototype_evaluation_error = _max_abs_error(
        prototype_bank,
        evaluated_prototype_bank,
        name="fused prototype evaluation reproduction",
    )
    if prototype_evaluation_error > artifact_tolerance:
        raise RuntimeError(
            "fused enrollment prototype paths disagree: "
            f"{prototype_evaluation_error:g} > {artifact_tolerance:g}"
        )
    expected_fused_contract = {
        "decision_rule": (
            "minimum squared distance across source/profile centroids "
            "mapped to each public class"
        ),
        "enrollment_view_rule": (
            "each source/profile centroid pools every eligible route "
            "view; historical rows retain eligible prefix views and "
            "current rows contribute canonicalized physical-scale views"
        ),
        "groups": prototype_metadata,
        "prototype_public_class_indices": prototype_labels.tolist(),
    }
    if final_evaluation["prototype_bank"] != expected_fused_contract:
        raise RuntimeError("fused prototype metadata paths disagree")

    probe_count = min(64, len(data["historical_selection"]["pooled"]["y"]))
    probe_bucket = data["historical_selection"]["pooled"]
    real_probe = embed_all(
        branches["real"]["net"],
        probe_bucket["x"][:probe_count],
        probe_bucket["f"][:probe_count],
        device,
    )
    complex_probe = embed_all(
        branches["complex"]["net"],
        probe_bucket["x"][:probe_count],
        probe_bucket["f"][:probe_count],
        device,
    )
    numpy_probe = fuse_numpy(
        real_probe,
        complex_probe,
        centers["real"],
        centers["complex"],
        weight_real=weight_real,
    )
    module_probe = embed_all(
        embedder,
        probe_bucket["x"][:probe_count],
        probe_bucket["f"][:probe_count],
        device,
    )
    fusion_assembly_error = _max_abs_error(
        numpy_probe, module_probe, name="NumPy/Torch fusion assembly"
    )
    if fusion_assembly_error > FUSION_ASSEMBLY_TOLERANCE:
        raise RuntimeError(
            f"NumPy/Torch fusion mismatch: {fusion_assembly_error:g} > "
            f"{FUSION_ASSEMBLY_TOLERANCE:g}"
        )

    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "state_dict": output / "fusion_state_dict.pt",
        "prototype_bank": output / "fusion_prototype_bank.npy",
        "prototype_labels": output / "fusion_prototype_labels.npy",
        "prototype_metadata": output / "fusion_prototype_bank.json",
        "real_center": output / "real_center.npy",
        "complex_center": output / "complex_center.npy",
        "feature_mean": output / "feature_mean.npy",
        "feature_std": output / "feature_std.npy",
    }
    torch.save(
        {
            key: value.detach().cpu()
            for key, value in fusion.state_dict().items()
        },
        paths["state_dict"],
    )
    np.save(paths["prototype_bank"], prototype_bank)
    np.save(paths["prototype_labels"], prototype_labels.astype(np.int64))
    _write_json(
        paths["prototype_metadata"],
        {
            **expected_fused_contract,
            "classes": list(data["classes"]),
        },
    )
    # Plain .npy files have deterministic bytes; NPZ embeds zip metadata.
    np.save(paths["real_center"], centers["real"])
    np.save(paths["complex_center"], centers["complex"])
    np.save(paths["feature_mean"], rebuilt_mean)
    np.save(paths["feature_std"], rebuilt_std)
    artifact_hashes = {
        key: _sha256(path)
        for key, path in paths.items()
    }
    artifact_contract: dict[str, str] = {}
    for key, path in paths.items():
        artifact_contract[key] = path.name
        artifact_contract[f"{key}_sha256"] = artifact_hashes[key]

    _assert_source_snapshot_unchanged(
        source_hashes_at_start,
        _source_hashes(),
        operation="v5 scale-orbit fusion assembly",
    )
    _assert_source_snapshot_unchanged(
        expected_branch_sources_at_start,
        _expected_branch_source_hashes(),
        operation="v5 scale-orbit fusion branch-source verification",
    )
    result = {
        "status": "complete",
        "development_only": True,
        "release_evidence": False,
        "sealed_release_data_used": 0,
        "consumed_historical_test_rows_used": 0,
        "consumed_test_rows_exposed": 0,
        "encoder": "fusion",
        "architecture": fusion.config(),
        "parameter_count": int(fusion.parameter_counts()["total"]),
        "frontend": td_preprocess.preprocess_metadata(),
        "run_configuration": {
            "schema": FUSION_SCHEMA,
            "branch_weight_real": weight_real,
            "branch_weight_complex": float(1.0 - weight_real),
            "artifact_reproduction_tolerance": artifact_tolerance,
            "moment_tolerance": moment_tolerance,
            "device_requested": args.device,
            "device_resolved": str(device),
            "seed_inherited_from_matched_branches": seed,
            "pretraining_amendment_validation": amendment_validation,
            "identity_firewall_amendment_validation":
                identity_firewall_validation,
            "training_enrollment_corpus": str(current_dir),
            "adaptive_selection_corpus": str(selection_dir),
            "current_source_share_inherited_from_matched_branches":
                _validated_source_share_contract(
                    branch_arguments["current_source_share"],
                    "matched branch current_source_share",
                ),
        },
        "matched_branch_contract": matched_contract,
        "source_branches": {
            name: {
                "directory": str(branch["directory"]),
                "encoder": name,
                "architecture": branch["metrics"]["architecture"],
                "parameter_count": branch["metrics"]["parameter_count"],
                "hashes": branch["hashes"],
                "best_episode": branch["metrics"]["training"].get(
                    "best_episode"
                ),
                "best_checkpoint_score": branch["metrics"]["training"].get(
                    "best_checkpoint_score"
                ),
                **{
                    key: value
                    for key, value in branch_reproduction[name].items()
                    if key == "prototype_reproduction_max_abs_error"
                },
            }
            for name, branch in branches.items()
        },
        "fusion": {
            "kind": "centered_invariant_fusion",
            "rule": (
                "unit(concat(sqrt(w) * unit(real - 0.2 * real_center), "
                "sqrt(1-w) * unit(complex - 0.2 * complex_center)))"
            ),
            "weight_real": weight_real,
            "weight_complex": float(1.0 - weight_real),
            "alpha_real": ALPHA_REAL,
            "alpha_complex": ALPHA_COMPLEX,
            "eps": FUSION_EPS,
            "numpy_vs_module_max_abs_error": fusion_assembly_error,
            "numpy_vs_module_tolerance": FUSION_ASSEMBLY_TOLERANCE,
        },
        "fitting_contract": {
            "center_fit_population": (
                "merged historical plus seed-20264101 current training views only"
            ),
            "center_view_count": int(len(data["xtr"])),
            "feature_moment_fit_population": (
                "merged historical plus seed-20264101 current training views only"
            ),
            "feature_moment_reproduction": {
                **moment_errors,
                "cross_branch_mean_max_abs_error": cross_mean_error,
                "cross_branch_std_max_abs_error": cross_std_error,
            },
            "prototype_fit_population": (
                "combined historical plus seed-20264101 current enrollment "
                "views only"
            ),
            "prototype_grouping": "(source, profile)",
            "prototype_view_count": int(len(data["yen"])),
            "prototype_group_count": int(len(prototype_bank)),
            "prototype_evaluation_max_abs_error":
                prototype_evaluation_error,
            "selection_population_role": (
                "seed-20262904 adaptive selection metrics only; never weight, "
                "center, feature-moment, prototype, threshold, or rank fit"
            ),
            "consumed_historical_test_rows_loaded": 0,
            "sealed_release_rows_loaded": 0,
        },
        "final_evaluation": final_evaluation,
        "data_audit": rebuilt_audit,
        "artifacts": artifact_contract,
        "source_sha256": source_hashes_at_start,
        "device": str(device),
        "wall_clock_s": time.perf_counter() - started,
    }
    _validate_no_consumed_provenance(result)
    metrics_path = output / "dev_metrics.json"
    _write_json(metrics_path, result)
    historical_worst = final_evaluation["historical"][
        "worst_length_balanced_accuracy"
    ]
    current_scale_worst = final_evaluation["current"][
        "worst_scale_present_class_balanced_accuracy"
    ]
    current_profile_scale_worst = final_evaluation["current"][
        "worst_profile_scale_recall"
    ]
    combined_pooled = final_evaluation["combined"]["pooled"][
        "balanced_accuracy"
    ]
    print(
        "[v5 scale-orbit fusion] "
        f"hist-worst={historical_worst:.4f} "
        f"current-scale-worst={current_scale_worst:.4f} "
        f"current-profile-scale-worst={current_profile_scale_worst:.4f} "
        f"combined-pooled={combined_pooled:.4f}",
        flush=True,
    )
    print(f"[v5 scale-orbit fusion] wrote {output}", flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--complex-dir", required=True)
    parser.add_argument("--current-corpus", required=True)
    parser.add_argument("--current-selection-corpus", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--historical-corpus",
        default=str(HISTORICAL_CORPUS),
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps"), default="auto"
    )
    parser.add_argument(
        "--branch-weight",
        type=float,
        default=DEFAULT_WEIGHT_REAL,
    )
    parser.add_argument(
        "--artifact-tolerance",
        type=float,
        default=DEFAULT_ARTIFACT_REPRODUCTION_TOLERANCE,
    )
    parser.add_argument(
        "--moment-tolerance",
        type=float,
        default=DEFAULT_MOMENT_TOLERANCE,
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
