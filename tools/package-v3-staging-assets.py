#!/usr/bin/env python3
"""Build the fail-closed, deployable v3 dual-fusion staging package.

Only four runtime-required JSON assets are copied: the role-bound rejector and
classifier weights, the staged open-set policy, and their dual binding. Probe
and parity fixtures remain training evidence and are hash-bound externally.
Every deployable file must be strictly smaller than 25 MiB.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
import hashlib
import json
import math
import os
import re
import shutil
import struct
import tempfile
from pathlib import Path
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
ARTIFACT_STAGING = (
    REPO / "training/zplane_ab/v2_full_variation/artifacts/staging"
)
DEFAULT_REJECTOR_EXPORT = (
    ARTIFACT_STAGING / "time_domain_v3_rejector_fusion_dual"
)
DEFAULT_CLASSIFIER_EXPORT = (
    ARTIFACT_STAGING / "time_domain_v3_classifier_fusion_dual"
)
DEFAULT_OPENSET_EXPORT = ARTIFACT_STAGING / "time_domain_v3_openset_dual"
DEFAULT_DESTINATION = REPO / "src/embedding/assets-v3-dual-staging"
LIVE_V2_ASSETS = (REPO / "src/embedding/assets").resolve()
LEGACY_V3_STAGING = (REPO / "src/embedding/assets-v3-staging").resolve()

PACKAGE_SCHEMA = "atomos.v3.time-domain-classifier.dual-runtime-package"
PACKAGE_SCHEMA_VERSION = 1
STAGING_STATUS = "staging_not_release"
CANDIDATE_ID = "v3.4-q97-decoupled-8k-classifier-4k-rejector"
DESIGN_NOVELTY_SEED = 20260955
VALIDATION_NOVELTY_SEEDS = [20260953, 20260954]
RELEASE_SEED_NOT_SPENT = 20260736
PARITY_NOVELTY_SEED = 20269101
REQUIRED_PARITY_LENGTHS = [4096, 8192, 16384, 32768]
MAX_DEPLOYABLE_BYTES = 25 * 1024 * 1024

FUSION_EXPORT_SCHEMA = (
    "atomos.v3.time-domain-invariant-fusion.browser-weights.export-manifest"
)
FUSION_EXPORT_SCHEMA_VERSION = 2
FUSION_WEIGHTS_SCHEMA = (
    "atomos.v3.time-domain-invariant-fusion.browser-weights"
)
OPENSET_EXPORT_SCHEMA = "time-domain-v3-dual-openset-staging-manifest-v1"
OPENSET_SCHEMA = "atomos.v3.time-domain-openset.staged"
OPENSET_SCHEMA_VERSION = 4
DUAL_BINDING_SCHEMA = "atomos.v3.time-domain-dual-fusion.binding"
PARITY_SCHEMA_VERSION = 4
STAGE_TWO_POLICY_KIND = "v3_known_only_lof_frequency_dispersion_rank_blend"
FRONTEND_VERSION = "invariant-patch-time-domain-v1"
TIME_DOMAIN_FEATURE_COUNT = 12
NOISE_PREFILTER_VERSION = "noise-prefilter-v1"
PREFILTER_FEATURE_NAMES = [
    "log10_relative_lag_magnitude_mean",
    "relative_lag_magnitude_decay_slope",
    "bandwidth_log_dispersion",
    "carrier_phase_coherence",
    "high_lag_qualified_fraction",
    "prefix_bandwidth_log_dispersion",
    "peak_normalized_mean_power",
]
REQUIRED_STAGE_ONE_LENGTHS = [4096, 8192, 16384]
FROZEN_GEOMETRY_FEATURE = "across_patch_frequency_dispersion"
EXPECTED_LOF_COMPONENTS = {
    "real": {"weight": 0.4, "neighbors": 2},
    "complex": {"weight": 0.6, "neighbors": 64},
}
STAGED_POLICY_SCHEMA = 4
STAGED_POLICY_VERSION = "v3-staged-openset-policy-v4-composite-survivor-q97"
STAGED_POLICY_KIND = (
    "v3_staged_noise_prefilter_then_q97_composite_survivor_lof_geometry"
)
COMPOSITE_SURVIVOR_SCORE = (
    "max(stage2_enrollment_rank, stage1_score_enrollment_rank)"
)
STAGE_TWO_THRESHOLD_QUANTILE = 0.95
COMPOSITE_THRESHOLD_QUANTILE = 0.97
STAGE_ONE_KNOWN_FALSE_POSITIVE_BUDGET = 0.01
SURVIVOR_KNOWN_FALSE_POSITIVE_BUDGET = 1.0 - COMPOSITE_THRESHOLD_QUANTILE
NOMINAL_ENROLLMENT_FALSE_UNKNOWN_BUDGET = (
    STAGE_ONE_KNOWN_FALSE_POSITIVE_BUDGET
    + (1.0 - STAGE_ONE_KNOWN_FALSE_POSITIVE_BUDGET)
    * SURVIVOR_KNOWN_FALSE_POSITIVE_BUDGET
)
COMPOSITE_ONLY_POLICY_CHANGE = (
    "composite survivor enrollment threshold quantile q99 -> q97"
)

REJECTOR_ROLE = "known_unknown_rejector"
CLASSIFIER_ROLE = "accepted_known_classifier"
REJECTOR_RESPONSIBILITY = "known_unknown_only"
CLASSIFIER_RESPONSIBILITY = "accepted_known_label_only"

FUSION_MANIFEST = "export-manifest.json"
OPENSET_MANIFEST = "manifest.json"
REJECTOR_WEIGHTS = "time-domain-v3-rejector-weights.json"
CLASSIFIER_WEIGHTS = "time-domain-v3-classifier-weights.json"
REJECTOR_PROBE = "time-domain-v3-rejector-probe.json"
CLASSIFIER_PROBE = "time-domain-v3-classifier-probe.json"
OPENSET_POLICY = "time-domain-v3-openset-policy.json"
DUAL_BINDING = "time-domain-v3-dual-binding.json"
OPENSET_PARITY = "time-domain-openset-parity-v1.json"
PACKAGE_MANIFEST = "runtime-package-manifest.json"
DEPLOYABLE_ASSETS = (
    REJECTOR_WEIGHTS,
    CLASSIFIER_WEIGHTS,
    OPENSET_POLICY,
    DUAL_BINDING,
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class PackageError(RuntimeError):
    """An input cannot safely become a dual-fusion runtime package."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise PackageError(f"{label} must be a lowercase SHA-256")
    return value


def _require_exact_scalar(value: Any, expected: Any, label: str) -> None:
    """Reject Python/JSON equality aliases at the package trust boundary."""
    if type(value) is not type(expected) or value != expected:
        raise PackageError(
            f"{label}={value!r} ({type(value).__name__}), expected exact "
            f"{expected!r} ({type(expected).__name__})"
        )


def _require_exact_int_list(
    value: Any, expected: list[int], label: str
) -> None:
    if (
        type(value) is not list
        or len(value) != len(expected)
        or any(type(item) is not int for item in value)
        or value != expected
    ):
        raise PackageError(f"{label} must be the exact integer list {expected}")


def _verify_evidence_provenance(
    provenance: Any, label: str
) -> Mapping[str, Any]:
    if not isinstance(provenance, Mapping):
        raise PackageError(f"{label} must be an evidence object")
    expected = {
        "candidate_id": CANDIDATE_ID,
        "staged_validation_status": "development_openset_pass",
        "staged_validation_role": "validate",
        "staged_validation_all_pass": True,
        "development_only": True,
        "release_evidence": False,
        "sealed_release_data_used": 0,
        "consumed_test_rows_used": 0,
        "release_seed_not_spent": RELEASE_SEED_NOT_SPENT,
    }
    for name, wanted in expected.items():
        _require_exact_scalar(
            provenance.get(name), wanted, f"{label}.{name}"
        )
    return provenance


def _finite_float(value: Any, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise PackageError(f"{label} must be an exact finite JSON float")
    return value


def _float_vector(
    value: Any,
    label: str,
    *,
    nonempty: bool = True,
    sorted_values: bool = False,
    positive: bool = False,
    nonnegative: bool = False,
) -> list[float]:
    if type(value) is not list or (nonempty and not value):
        raise PackageError(f"{label} must be a JSON float vector")
    output: list[float] = []
    for index, scalar in enumerate(value):
        number = _finite_float(scalar, f"{label}[{index}]")
        if positive and number <= 0.0:
            raise PackageError(f"{label}[{index}] must be positive")
        if nonnegative and number < 0.0:
            raise PackageError(f"{label}[{index}] must be nonnegative")
        if sorted_values and output and number < output[-1]:
            raise PackageError(f"{label} must be sorted")
        output.append(number)
    return output


def _float_matrix(
    value: Any, label: str, *, columns: int, nonempty: bool = True
) -> list[list[float]]:
    if type(value) is not list or (nonempty and not value):
        raise PackageError(f"{label} must be a JSON float matrix")
    output: list[list[float]] = []
    for index, row in enumerate(value):
        parsed = _float_vector(row, f"{label}[{index}]")
        if len(parsed) != columns:
            raise PackageError(f"{label}[{index}] has the wrong width")
        output.append(parsed)
    return output


def _verify_frontend(
    value: Any,
    label: str,
    *,
    require_packed_length: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PackageError(f"{label} must be an object")
    _require_exact_scalar(
        value.get("version"), FRONTEND_VERSION, f"{label}.version"
    )
    _require_exact_scalar(
        value.get("uses_frequency_transform"),
        False,
        f"{label}.uses_frequency_transform",
    )
    patch_length = value.get("patch_length")
    patch_count = value.get("patch_count")
    target_frac = value.get("target_frac")
    if (
        type(patch_length) is not int
        or patch_length <= 0
        or type(patch_count) is not int
        or patch_count <= 0
        or type(target_frac) is not float
        or not math.isfinite(target_frac)
        or not 0.0 < target_frac <= 1.0
    ):
        raise PackageError(f"{label} geometry/types are invalid")
    packed_length = patch_length * patch_count
    if require_packed_length:
        _require_exact_scalar(
            value.get("packed_length"),
            packed_length,
            f"{label}.packed_length",
        )
    else:
        feature_count = value.get("feature_count")
        _require_exact_scalar(
            feature_count,
            TIME_DOMAIN_FEATURE_COUNT,
            f"{label}.feature_count",
        )
    return {
        "version": FRONTEND_VERSION,
        "patch_length": patch_length,
        "patch_count": patch_count,
        "target_frac": target_frac,
        "packed_length": packed_length,
        "uses_frequency_transform": False,
    }


def _positive_int(value: Any, label: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise PackageError(f"{label} must be an exact integer >= {minimum}")
    return value


def _verify_linear(
    value: Any,
    label: str,
    *,
    expected_input: int,
    expected_output: int,
) -> None:
    if not isinstance(value, Mapping):
        raise PackageError(f"{label} must be an object")
    _require_exact_scalar(value.get("in"), expected_input, f"{label}.in")
    _require_exact_scalar(value.get("out"), expected_output, f"{label}.out")
    if len(
        _float_vector(value.get("weight"), f"{label}.weight")
    ) != expected_input * expected_output:
        raise PackageError(f"{label}.weight has the wrong shape")
    if len(
        _float_vector(value.get("bias"), f"{label}.bias")
    ) != expected_output:
        raise PackageError(f"{label}.bias has the wrong shape")


def _verify_branch_config(
    value: Any, label: str, encoder: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PackageError(f"{label} must be an object")
    _require_exact_scalar(value.get("encoder"), encoder, f"{label}.encoder")
    set_pool = value.get("set_pool")
    if set_pool not in {"mean", "mean_std"}:
        raise PackageError(f"{label}.set_pool is invalid")
    dropout = _finite_float(value.get("dropout"), f"{label}.dropout")
    if not 0.0 <= dropout < 1.0:
        raise PackageError(f"{label}.dropout must lie in [0, 1)")
    return {
        "patch_length": _positive_int(
            value.get("patch_length"), f"{label}.patch_length", 16
        ),
        "patch_count": _positive_int(
            value.get("patch_count"), f"{label}.patch_count"
        ),
        "patch_dim": _positive_int(
            value.get("patch_dim"), f"{label}.patch_dim"
        ),
        "hidden": _positive_int(value.get("hidden"), f"{label}.hidden"),
        "embed_dim": _positive_int(
            value.get("embed_dim"), f"{label}.embed_dim"
        ),
        "n_features": _positive_int(
            value.get("n_features"), f"{label}.n_features", 0
        ),
        "set_pool": set_pool,
    }


def _verify_real_branch(value: Any) -> dict[str, Any]:
    label = "fusion weights real"
    if not isinstance(value, Mapping):
        raise PackageError(f"{label} must be an object")
    config = _verify_branch_config(value.get("config"), f"{label}.config", "real")
    blocks = value.get("blocks")
    if type(blocks) is not list or not blocks:
        raise PackageError(f"{label}.blocks must be non-empty")
    channels = 2
    for index, block in enumerate(blocks):
        block_label = f"{label}.blocks[{index}]"
        if not isinstance(block, Mapping):
            raise PackageError(f"{block_label} must be an object")
        input_channels = _positive_int(
            block.get("in_channels"), f"{block_label}.in_channels"
        )
        output_channels = _positive_int(
            block.get("out_channels"), f"{block_label}.out_channels"
        )
        kernel = _positive_int(block.get("kernel"), f"{block_label}.kernel")
        if input_channels != channels:
            raise PackageError(f"{block_label} breaks the channel chain")
        _positive_int(block.get("stride"), f"{block_label}.stride")
        _positive_int(block.get("padding"), f"{block_label}.padding", 0)
        if len(
            _float_vector(
                block.get("conv_weight"), f"{block_label}.conv_weight"
            )
        ) != output_channels * input_channels * kernel:
            raise PackageError(f"{block_label}.conv_weight has the wrong shape")
        batch_norm = block.get("batch_norm")
        if not isinstance(batch_norm, Mapping):
            raise PackageError(f"{block_label}.batch_norm must be an object")
        eps = _finite_float(
            batch_norm.get("eps"), f"{block_label}.batch_norm.eps"
        )
        if eps <= 0.0:
            raise PackageError(f"{block_label}.batch_norm.eps must be positive")
        for name in ("weight", "bias", "running_mean", "running_var"):
            vector = _float_vector(
                batch_norm.get(name), f"{block_label}.batch_norm.{name}"
            )
            if len(vector) != output_channels:
                raise PackageError(
                    f"{block_label}.batch_norm.{name} has the wrong shape"
                )
            if name == "running_var" and any(item < 0.0 for item in vector):
                raise PackageError(
                    f"{block_label}.batch_norm.running_var is negative"
                )
        channels = output_channels
    _verify_linear(
        value.get("patch_projection"),
        f"{label}.patch_projection",
        expected_input=2 * channels,
        expected_output=config["patch_dim"],
    )
    set_width = config["patch_dim"] * (
        2 if config["set_pool"] == "mean_std" else 1
    )
    _verify_linear(
        value.get("fc1"),
        f"{label}.fc1",
        expected_input=set_width + config["n_features"],
        expected_output=config["hidden"],
    )
    _verify_linear(
        value.get("fc2"),
        f"{label}.fc2",
        expected_input=config["hidden"],
        expected_output=config["embed_dim"],
    )
    return config


def _verify_complex_branch(value: Any) -> dict[str, Any]:
    label = "fusion weights complex"
    if not isinstance(value, Mapping):
        raise PackageError(f"{label} must be an object")
    config = _verify_branch_config(
        value.get("config"), f"{label}.config", "complex"
    )
    stages = value.get("stages")
    if type(stages) is not list or not stages:
        raise PackageError(f"{label}.stages must be non-empty")
    channels = 1
    for index, stage in enumerate(stages):
        stage_label = f"{label}.stages[{index}]"
        if not isinstance(stage, Mapping):
            raise PackageError(f"{stage_label} must be an object")
        input_channels = _positive_int(
            stage.get("in_channels"), f"{stage_label}.in_channels"
        )
        output_channels = _positive_int(
            stage.get("out_channels"), f"{stage_label}.out_channels"
        )
        kernel = _positive_int(stage.get("kernel"), f"{stage_label}.kernel")
        if input_channels != channels:
            raise PackageError(f"{stage_label} breaks the channel chain")
        _positive_int(stage.get("padding"), f"{stage_label}.padding", 0)
        for name in ("weight_real", "weight_imag"):
            if len(
                _float_vector(stage.get(name), f"{stage_label}.{name}")
            ) != output_channels * input_channels * kernel:
                raise PackageError(f"{stage_label}.{name} has the wrong shape")
        if len(
            _float_vector(
                stage.get("threshold_raw"), f"{stage_label}.threshold_raw"
            )
        ) != output_channels:
            raise PackageError(f"{stage_label}.threshold_raw has the wrong shape")
        channels = output_channels
    for name in ("mag_norm_eps", "modrelu_magnitude_eps"):
        if _finite_float(value.get(name), f"{label}.{name}") <= 0.0:
            raise PackageError(f"{label}.{name} must be positive")
    lowpass = _float_vector(value.get("lowpass"), f"{label}.lowpass")
    if lowpass != [0.25, 0.5, 0.25]:
        raise PackageError(f"{label}.lowpass differs")
    lags_value = value.get("lags")
    if (
        type(lags_value) is not list
        or not lags_value
        or any(type(lag) is not int or lag <= 0 for lag in lags_value)
        or any(
            right <= left
            for left, right in zip(lags_value, lags_value[1:])
        )
    ):
        raise PackageError(f"{label}.lags must be strictly increasing integers")
    statistic_width = channels * (2 + 2 * len(lags_value))
    _verify_linear(
        value.get("patch_projection"),
        f"{label}.patch_projection",
        expected_input=statistic_width,
        expected_output=config["patch_dim"],
    )
    set_width = config["patch_dim"] * (
        2 if config["set_pool"] == "mean_std" else 1
    )
    _verify_linear(
        value.get("fc1"),
        f"{label}.fc1",
        expected_input=set_width + config["n_features"],
        expected_output=config["hidden"],
    )
    _verify_linear(
        value.get("fc2"),
        f"{label}.fc2",
        expected_input=config["hidden"],
        expected_output=config["embed_dim"],
    )
    return config


def _verify_fusion_runtime_payload(
    weights: Mapping[str, Any],
    frontend: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    real = _verify_real_branch(weights.get("real"))
    complex_branch = _verify_complex_branch(weights.get("complex"))
    for name in ("patch_length", "patch_count", "embed_dim", "n_features"):
        if real[name] != complex_branch[name]:
            raise PackageError(f"{label} branch {name} values differ")
    if (
        real["patch_length"] != frontend["patch_length"]
        or real["patch_count"] != frontend["patch_count"]
        or real["n_features"] != TIME_DOMAIN_FEATURE_COUNT
    ):
        raise PackageError(f"{label} branch geometry differs from frontend")
    packed_length = weights.get("packed_length")
    _require_exact_scalar(
        packed_length, frontend["packed_length"], f"{label}.packed_length"
    )
    embed_dim = real["embed_dim"]
    fusion = weights.get("fusion")
    if not isinstance(fusion, Mapping):
        raise PackageError(f"{label}.fusion must be an object")
    centers: dict[str, list[float]] = {}
    for name in ("real_center", "complex_center"):
        centers[name] = _float_vector(
            fusion.get(name), f"{label}.fusion.{name}"
        )
        if len(centers[name]) != embed_dim:
            raise PackageError(f"{label}.fusion.{name} has the wrong width")
    alpha_real = _finite_float(
        fusion.get("alpha_real"), f"{label}.fusion.alpha_real"
    )
    alpha_complex = _finite_float(
        fusion.get("alpha_complex"), f"{label}.fusion.alpha_complex"
    )
    weight_real = _finite_float(
        fusion.get("weight_real"), f"{label}.fusion.weight_real"
    )
    if not 0.0 <= weight_real <= 1.0:
        raise PackageError(f"{label}.fusion.weight_real must lie in [0, 1]")
    eps = _finite_float(fusion.get("eps"), f"{label}.fusion.eps")
    if eps <= 0.0:
        raise PackageError(f"{label}.fusion.eps must be positive")
    standardization = weights.get("feature_standardization")
    if not isinstance(standardization, Mapping):
        raise PackageError(f"{label}.feature_standardization must be an object")
    mean = _float_vector(
        standardization.get("mean"),
        f"{label}.feature_standardization.mean",
    )
    std = _float_vector(
        standardization.get("std"),
        f"{label}.feature_standardization.std",
        positive=True,
    )
    if len(mean) != real["n_features"] or len(std) != real["n_features"]:
        raise PackageError(f"{label}.feature_standardization width differs")
    classification = weights.get("classification")
    if not isinstance(classification, Mapping):
        raise PackageError(f"{label}.classification must be an object")
    classes = classification.get("classes")
    if (
        type(classes) is not list
        or not classes
        or any(type(name) is not str or not name for name in classes)
        or len(set(classes)) != len(classes)
    ):
        raise PackageError(f"{label}.classification.classes is invalid")
    prototypes = classification.get("prototypes")
    if type(prototypes) is not list or len(prototypes) != len(classes):
        raise PackageError(f"{label}.classification.prototypes row count differs")
    parsed_prototypes = _float_matrix(
        prototypes,
        f"{label}.classification.prototypes",
        columns=2 * embed_dim,
    )
    return {
        "classes": classes,
        "embed_dimensions": {
            "real": real["embed_dim"],
            "complex": complex_branch["embed_dim"],
        },
        "fusion": {
            "real_center": centers["real_center"],
            "complex_center": centers["complex_center"],
            "alpha_real": alpha_real,
            "alpha_complex": alpha_complex,
            "weight_real": weight_real,
            "eps": eps,
        },
        "feature_standardization": {"mean": mean, "std": std},
        "prototypes": parsed_prototypes,
    }


def _verify_stage_one(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise PackageError("open-set stage_one must be an object")
    _require_exact_scalar(
        value.get("kind"),
        NOISE_PREFILTER_VERSION,
        "open-set stage_one.kind",
    )
    _require_exact_scalar(
        value.get("feature_names"),
        PREFILTER_FEATURE_NAMES,
        "open-set stage_one.feature_names",
    )
    _require_exact_scalar(
        value.get("match_frontend_min_bandwidth"),
        True,
        "open-set stage_one.match_frontend_min_bandwidth",
    )
    _sha(value.get("set_sha256"), "open-set stage_one.set_sha256")
    models = value.get("models")
    expected_keys = {str(length) for length in REQUIRED_STAGE_ONE_LENGTHS}
    if not isinstance(models, Mapping) or set(models) != expected_keys:
        raise PackageError("open-set stage_one model lengths differ")
    expected_model_keys = {
        "capture_length",
        "feature_names",
        "mean",
        "scale",
        "coefficients",
        "intercept",
        "threshold_score",
    }
    for length in REQUIRED_STAGE_ONE_LENGTHS:
        label = f"open-set stage_one.models.{length}"
        model = models[str(length)]
        if not isinstance(model, Mapping) or set(model) != expected_model_keys:
            raise PackageError(f"{label} field set differs")
        _require_exact_scalar(
            model.get("capture_length"), length, f"{label}.capture_length"
        )
        _require_exact_scalar(
            model.get("feature_names"),
            PREFILTER_FEATURE_NAMES,
            f"{label}.feature_names",
        )
        mean = _float_vector(model.get("mean"), f"{label}.mean")
        scale = _float_vector(
            model.get("scale"), f"{label}.scale", positive=True
        )
        coefficients = _float_vector(
            model.get("coefficients"), f"{label}.coefficients"
        )
        if not (
            len(mean)
            == len(scale)
            == len(coefficients)
            == len(PREFILTER_FEATURE_NAMES)
        ):
            raise PackageError(f"{label} arrays have the wrong width")
        _finite_float(model.get("intercept"), f"{label}.intercept")
        _finite_float(
            model.get("threshold_score"), f"{label}.threshold_score"
        )


def _verify_lof_components(
    value: Any, expected_embed_dimensions: Mapping[str, int]
) -> None:
    if type(value) is not list or len(value) != len(EXPECTED_LOF_COMPONENTS):
        raise PackageError("open-set stage_two.lof_components must be exact")
    expected_keys = {
        "branch",
        "weight",
        "neighbors",
        "mean",
        "scale",
        "reference",
        "reference_k_distance",
        "reference_local_density",
        "calibration",
    }
    seen: set[str] = set()
    for index, component in enumerate(value):
        label = f"open-set stage_two.lof_components[{index}]"
        if not isinstance(component, Mapping) or set(component) != expected_keys:
            raise PackageError(f"{label} field set differs")
        branch = component.get("branch")
        if branch not in EXPECTED_LOF_COMPONENTS or branch in seen:
            raise PackageError(f"{label}.branch differs or is duplicated")
        seen.add(branch)
        expected = EXPECTED_LOF_COMPONENTS[branch]
        _require_exact_scalar(
            component.get("weight"), expected["weight"], f"{label}.weight"
        )
        _require_exact_scalar(
            component.get("neighbors"),
            expected["neighbors"],
            f"{label}.neighbors",
        )
        mean = _float_vector(component.get("mean"), f"{label}.mean")
        scale = _float_vector(
            component.get("scale"), f"{label}.scale", positive=True
        )
        if len(mean) != len(scale):
            raise PackageError(f"{label} mean/scale widths differ")
        if len(mean) != expected_embed_dimensions[branch]:
            raise PackageError(
                f"{label} width differs from rejector {branch} embedding"
            )
        reference = _float_matrix(
            component.get("reference"),
            f"{label}.reference",
            columns=len(mean),
        )
        if len(reference) <= expected["neighbors"]:
            raise PackageError(f"{label}.reference has too few rows")
        k_distance = _float_vector(
            component.get("reference_k_distance"),
            f"{label}.reference_k_distance",
            nonnegative=True,
        )
        local_density = _float_vector(
            component.get("reference_local_density"),
            f"{label}.reference_local_density",
            positive=True,
        )
        if len(k_distance) != len(reference) or len(local_density) != len(
            reference
        ):
            raise PackageError(f"{label} reference metadata lengths differ")
        _float_vector(
            component.get("calibration"),
            f"{label}.calibration",
            sorted_values=True,
        )
    if seen != set(EXPECTED_LOF_COMPONENTS):
        raise PackageError("open-set stage_two LOF branch set differs")


def read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PackageError(f"{path} must be a regular non-symlink file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackageError(f"{path} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise PackageError(f"{path} must contain a JSON object")
    return value


def json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _record(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def _verify_record(
    path: Path,
    value: Any,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"bytes", "sha256"}:
        raise PackageError(f"{label} must contain exactly bytes and sha256")
    size = value.get("bytes")
    if type(size) is not int or size <= 0:
        raise PackageError(f"{label}.bytes must be a positive integer")
    digest = _sha(value.get("sha256"), f"{label}.sha256")
    if path.is_symlink() or not path.is_file():
        raise PackageError(f"{path} must be a regular non-symlink file")
    if path.stat().st_size != size or sha256(path) != digest:
        raise PackageError(f"{label} differs from its recorded bytes")
    return {"bytes": size, "sha256": digest}


def _verify_linked_record(
    value: Any,
    *,
    name: str,
    expected: Mapping[str, Any],
    label: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "name",
        "bytes",
        "sha256",
    }:
        raise PackageError(f"{label} linked record field set differs")
    _require_exact_scalar(value.get("name"), name, f"{label}.name")
    _require_exact_scalar(
        value.get("bytes"), expected["bytes"], f"{label}.bytes"
    )
    _require_exact_scalar(
        value.get("sha256"), expected["sha256"], f"{label}.sha256"
    )


def _source_path(path: Path, destination: Path) -> str:
    """A package-parent-relative evidence locator; never serialize absolutes."""
    return Path(
        os.path.relpath(Path(path).resolve(), Path(destination).resolve())
    ).as_posix()


def _asset_record(
    path: Path,
    payload: Mapping[str, Any],
    *,
    runtime_role: str | None = None,
) -> dict[str, Any]:
    record = {
        "path": path.name,
        **_record(path),
        "schema": payload.get("schema"),
        "schema_version": payload.get("schema_version"),
        "status": payload.get("status"),
    }
    if runtime_role is not None:
        record["runtime_role"] = runtime_role
    return record


def _load_role_export(
    directory: Path,
    *,
    role: str,
    weights_name: str,
    probe_name: str,
) -> dict[str, Any]:
    if directory.is_symlink() or not directory.is_dir():
        raise PackageError(f"{role} export must be a non-symlink directory")
    manifest_path = directory / FUSION_MANIFEST
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema") != FUSION_EXPORT_SCHEMA
        or manifest.get("runtime_role") != role
    ):
        raise PackageError(f"{role} export manifest has the wrong role/schema")
    _require_exact_scalar(
        manifest.get("schema_version"),
        FUSION_EXPORT_SCHEMA_VERSION,
        f"{role} export schema_version",
    )
    bundle_sha = _sha(
        manifest.get("source_bundle_manifest_sha256"),
        f"{role} source bundle manifest",
    )
    emitted = manifest.get("emitted")
    if not isinstance(emitted, Mapping) or set(emitted) != {
        weights_name,
        probe_name,
    }:
        raise PackageError(f"{role} export has missing or aliased role files")
    weights_path = directory / weights_name
    probe_path = directory / probe_name
    weights_record = _verify_record(
        weights_path, emitted[weights_name], f"{role} weights"
    )
    probe_record = _verify_record(
        probe_path, emitted[probe_name], f"{role} probe"
    )
    weights = read_json(weights_path)
    if (
        weights.get("schema") != FUSION_WEIGHTS_SCHEMA
        or weights.get("status") != STAGING_STATUS
        or weights.get("runtime_role") != role
    ):
        raise PackageError(f"{role} weights do not carry their exact role")
    _require_exact_scalar(
        weights.get("schema_version"), 1, f"{role} weights schema_version"
    )
    frontend = _verify_frontend(
        weights.get("frontend"),
        f"{role} weights frontend",
        require_packed_length=False,
    )
    runtime = _verify_fusion_runtime_payload(
        weights, frontend, f"{role} weights"
    )
    provenance = weights.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("source_bundle_manifest_sha256") != bundle_sha
    ):
        raise PackageError(f"{role} weights lose their source-bundle binding")
    return {
        "directory": directory,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "manifest_record": _record(manifest_path),
        "bundle_sha256": bundle_sha,
        "weights_path": weights_path,
        "weights": weights,
        "frontend": frontend,
        "classes": runtime["classes"],
        "embed_dimensions": runtime["embed_dimensions"],
        "fusion": runtime["fusion"],
        "feature_standardization": runtime["feature_standardization"],
        "prototypes": runtime["prototypes"],
        "weights_record": weights_record,
        "probe_path": probe_path,
        "probe_record": probe_record,
    }


def _verify_binding(
    binding: Mapping[str, Any],
    rejector: Mapping[str, Any],
    classifier: Mapping[str, Any],
    policy_record: Mapping[str, Any],
    expected_frontend: Mapping[str, Any],
) -> None:
    if (
        binding.get("schema") != DUAL_BINDING_SCHEMA
        or binding.get("status") != STAGING_STATUS
        or binding.get("candidate_id") != CANDIDATE_ID
    ):
        raise PackageError("dual binding has the wrong schema/status/candidate")
    _require_exact_scalar(
        binding.get("schema_version"), 1, "dual binding schema_version"
    )
    binding_frontend_value = binding.get("frontend")
    if (
        not isinstance(binding_frontend_value, Mapping)
        or set(binding_frontend_value)
        != {
            "version",
            "patch_length",
            "patch_count",
            "target_frac",
            "packed_length",
            "uses_frequency_transform",
        }
    ):
        raise PackageError("dual binding frontend field set differs")
    binding_frontend = _verify_frontend(
        binding_frontend_value,
        "dual binding frontend",
        require_packed_length=True,
    )
    if binding_frontend != expected_frontend:
        raise PackageError("dual binding frontend differs from open-set policy")
    if binding.get("execution_order") != [
        "stage_one_noise_gate",
        "rejector_known_unknown",
        "classifier_known_label",
    ]:
        raise PackageError("dual binding execution order changed")
    roles = binding.get("roles")
    if not isinstance(roles, Mapping) or set(roles) != {
        "rejector",
        "classifier",
    }:
        raise PackageError("dual binding must contain exactly two named roles")
    expected = {
        "rejector": (
            roles["rejector"],
            REJECTOR_WEIGHTS,
            REJECTOR_ROLE,
            REJECTOR_RESPONSIBILITY,
            rejector,
        ),
        "classifier": (
            roles["classifier"],
            CLASSIFIER_WEIGHTS,
            CLASSIFIER_ROLE,
            CLASSIFIER_RESPONSIBILITY,
            classifier,
        ),
    }
    for label, (
        value,
        filename,
        role,
        responsibility,
        source,
    ) in expected.items():
        if not isinstance(value, Mapping):
            raise PackageError(f"binding role {label} is not an object")
        if (
            value.get("asset") != filename
            or value.get("runtime_role") != role
            or value.get("responsibility") != responsibility
            or value.get("asset_sha256")
            != source["weights_record"]["sha256"]
            or value.get("runtime_bundle_manifest_sha256")
            != source["bundle_sha256"]
        ):
            raise PackageError(f"binding role {label} is not byte/role bound")
        _sha(
            value.get("fusion_directory_sha256"),
            f"binding {label} fusion directory",
        )
    rejector_role = roles["rejector"]
    classifier_role = roles["classifier"]
    for field in (
        "asset_sha256",
        "runtime_bundle_manifest_sha256",
        "fusion_directory_sha256",
    ):
        if rejector_role[field] == classifier_role[field]:
            raise PackageError(f"dual binding aliases role {field}")

    openset = binding.get("openset_policy")
    if not isinstance(openset, Mapping):
        raise PackageError("binding has no open-set policy")
    if (
        openset.get("asset") != OPENSET_POLICY
        or openset.get("asset_sha256") != policy_record["sha256"]
        or openset.get("rejector_asset_sha256")
        != rejector["weights_record"]["sha256"]
        or openset.get("fitted_rejector_runtime_bundle_manifest_sha256")
        != rejector["bundle_sha256"]
    ):
        raise PackageError("open-set policy is not bound to the rejector role")
    report_sha = _sha(
        openset.get("staged_validation_report_sha256"),
        "binding staged validation report",
    )
    staged_hashes = openset.get("staged_artifacts_sha256")
    expected_staged = {
        "v3_branch_lof_components.npz",
        "v3_open_policy_stage_two.npz",
        "v3_staged_composite_policy.npz",
    }
    if not isinstance(staged_hashes, Mapping) or set(staged_hashes) != expected_staged:
        raise PackageError("binding staged policy asset set is incomplete")
    for name, digest in staged_hashes.items():
        _sha(digest, f"binding staged asset {name}")
    validation = binding.get("validation")
    if not isinstance(validation, Mapping):
        raise PackageError("binding validation evidence is not the frozen pair")
    if (
        validation.get("report_sha256") != report_sha
        or validation.get("role") != "validate"
        or validation.get("status") != "development_openset_pass"
    ):
        raise PackageError("binding validation evidence is not the frozen pair")
    _require_exact_scalar(
        validation.get("design_novelty_seed"),
        DESIGN_NOVELTY_SEED,
        "binding validation design_novelty_seed",
    )
    _require_exact_int_list(
        validation.get("novelty_seeds"),
        VALIDATION_NOVELTY_SEEDS,
        "binding validation novelty_seeds",
    )
    _require_exact_scalar(
        validation.get("release_seed_not_spent"),
        RELEASE_SEED_NOT_SPENT,
        "binding validation release_seed_not_spent",
    )
    fail_closed = binding.get("fail_closed")
    required_flags = {
        "role_assets_bound_by_sha256",
        "distinct_role_assets",
        "role_asset_sha256_must_differ",
        "classifier_runs_only_after_rejector_acceptance",
        "public_known_label_from_classifier_only",
    }
    if (
        not isinstance(fail_closed, Mapping)
        or set(fail_closed) != required_flags
        or any(fail_closed[name] is not True for name in required_flags)
    ):
        raise PackageError("dual binding fail-closed contract is incomplete")


def _numpy_linear_quantile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        raise PackageError("composite calibration cannot be empty")
    virtual = quantile * (len(sorted_values) - 1)
    index = math.floor(virtual)
    fraction = virtual - index
    if index + 1 >= len(sorted_values):
        return sorted_values[-1]
    low = sorted_values[index]
    high = sorted_values[index + 1]
    if fraction < 0.5:
        return low + fraction * (high - low)
    return high - (high - low) * (1.0 - fraction)


def _finite_sorted_vector(value: Any, label: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise PackageError(f"{label} must be a non-empty vector")
    output: list[float] = []
    for index, scalar in enumerate(value):
        if (
            type(scalar) is not float
            or not math.isfinite(scalar)
        ):
            raise PackageError(
                f"{label}[{index}] must be an exact finite JSON float"
            )
        number = scalar
        if output and number < output[-1]:
            raise PackageError(f"{label} must be sorted")
        output.append(number)
    return output


def _verify_policy_schema4(
    policy: Mapping[str, Any],
    expected_embed_dimensions: Mapping[str, int],
) -> dict[str, Any]:
    """Require q97 outer semantics while preserving the inner q95 policy."""
    contract = policy.get("contract")
    expected_contract_keys = {
        "additive_only",
        "changes_closed_label",
        "gates_before_classification",
        "architecture_contract_change",
    }
    if (
        not isinstance(contract, Mapping)
        or set(contract) != expected_contract_keys
        or contract.get("additive_only") is not False
        or contract.get("changes_closed_label") is not True
        or contract.get("gates_before_classification") is not True
        or not isinstance(contract.get("architecture_contract_change"), str)
        or not contract["architecture_contract_change"]
    ):
        raise PackageError("open-set architecture contract is incomplete")
    frontend_value = policy.get("frontend")
    expected_frontend_keys = {
        "version",
        "patch_length",
        "patch_count",
        "target_frac",
        "packed_length",
        "uses_frequency_transform",
    }
    if (
        not isinstance(frontend_value, Mapping)
        or set(frontend_value) != expected_frontend_keys
    ):
        raise PackageError("open-set frontend field set differs")
    frontend = _verify_frontend(
        frontend_value,
        "open-set frontend",
        require_packed_length=True,
    )
    _verify_stage_one(policy.get("stage_one"))

    stage_two = policy.get("stage_two")
    stage_two_policy = (
        stage_two.get("policy") if isinstance(stage_two, Mapping) else None
    )
    if (
        not isinstance(stage_two, Mapping)
        or set(stage_two) != {"kind", "lof_components", "policy"}
        or stage_two.get("kind") != STAGE_TWO_POLICY_KIND
        or not isinstance(stage_two_policy, Mapping)
    ):
        raise PackageError("open-set stage two is not the frozen q95 policy")
    _verify_lof_components(
        stage_two.get("lof_components"), expected_embed_dimensions
    )
    expected_stage_two_policy_keys = {
        "branch_lof_rank_weight",
        "geometry_weight",
        "threshold_quantile",
        "threshold",
        "geometry_feature",
        "class_geometry_mean",
        "class_geometry_scale",
        "geometry_calibration",
        "combined_calibration",
    }
    if set(stage_two_policy) != expected_stage_two_policy_keys:
        raise PackageError("open-set stage two policy field set differs")
    for name, wanted in (
        ("branch_lof_rank_weight", 0.8),
        ("geometry_weight", 0.2),
        ("threshold_quantile", STAGE_TWO_THRESHOLD_QUANTILE),
        ("geometry_feature", FROZEN_GEOMETRY_FEATURE),
    ):
        _require_exact_scalar(
            stage_two_policy.get(name), wanted, f"open-set stage two {name}"
        )
    stage_two_threshold = stage_two_policy.get("threshold")
    if type(stage_two_threshold) is not float or not math.isfinite(
        stage_two_threshold
    ):
        raise PackageError(
            "open-set stage-two threshold must be an exact finite JSON float"
        )
    if not 0.0 <= stage_two_threshold < 1.0:
        raise PackageError("open-set stage-two threshold must lie in [0, 1)")
    class_mean = _float_vector(
        stage_two_policy.get("class_geometry_mean"),
        "open-set stage two class_geometry_mean",
    )
    class_scale = _float_vector(
        stage_two_policy.get("class_geometry_scale"),
        "open-set stage two class_geometry_scale",
        positive=True,
    )
    if len(class_mean) != len(class_scale):
        raise PackageError("open-set stage-two class geometry lengths differ")
    geometry_calibration = _float_vector(
        stage_two_policy.get("geometry_calibration"),
        "open-set stage two geometry_calibration",
        sorted_values=True,
    )
    combined_calibration = _float_vector(
        stage_two_policy.get("combined_calibration"),
        "open-set stage two combined_calibration",
        sorted_values=True,
    )
    if (
        len(geometry_calibration) != len(combined_calibration)
        or any(value < 0.0 or value >= 1.0 for value in combined_calibration)
    ):
        raise PackageError("open-set stage-two calibrations are inconsistent")
    calibration_scores = [
        bisect_left(combined_calibration, value)
        / (len(combined_calibration) + 1)
        for value in combined_calibration
    ]
    expected_stage_two_threshold = _numpy_linear_quantile(
        calibration_scores, STAGE_TWO_THRESHOLD_QUANTILE
    )
    if stage_two_threshold != expected_stage_two_threshold:
        raise PackageError(
            "open-set stage-two threshold is not q95 of enrollment ranks"
        )

    composite = policy.get("composite")
    if not isinstance(composite, Mapping):
        raise PackageError("open-set policy has no composite q97 policy")
    expected = {
        "schema": STAGED_POLICY_SCHEMA,
        "kind": STAGED_POLICY_KIND,
        "policy_version": STAGED_POLICY_VERSION,
        "survivor_score": COMPOSITE_SURVIVOR_SCORE,
        "threshold_quantile": COMPOSITE_THRESHOLD_QUANTILE,
        "stage_two_threshold": stage_two_threshold,
        "threshold_population": "enrollment_stage_one_survivors_only",
        "training_rows_used_for_threshold": 0,
        "selection_rows_used_for_threshold": 0,
        "novelty_rows_used_for_threshold": 0,
        "release_rows_used_for_threshold": 0,
        "stage_one_known_false_positive_budget": (
            STAGE_ONE_KNOWN_FALSE_POSITIVE_BUDGET
        ),
        "survivor_known_false_positive_budget": (
            SURVIVOR_KNOWN_FALSE_POSITIVE_BUDGET
        ),
        "nominal_enrollment_false_unknown_budget": (
            NOMINAL_ENROLLMENT_FALSE_UNKNOWN_BUDGET
        ),
        "only_policy_change": COMPOSITE_ONLY_POLICY_CHANGE,
        "stage_one_changed": False,
        "rejector_cnn_fusion_changed": False,
        "classifier_cnn_fusion_changed": False,
        "gate_contract_changed": False,
    }
    for name, wanted in expected.items():
        _require_exact_scalar(
            composite.get(name), wanted, f"open-set composite {name}"
        )
    stage_one_calibration = _finite_sorted_vector(
        composite.get("stage_one_calibration_raw"),
        "composite stage-one calibration",
    )
    composite_calibration = _finite_sorted_vector(
        composite.get("composite_calibration_raw"),
        "composite calibration",
    )
    if (
        len(stage_one_calibration) != len(composite_calibration)
        or any(value < 0.0 or value >= 1.0 for value in composite_calibration)
    ):
        raise PackageError("composite survivor calibrations are inconsistent")
    enrollment_rows = composite.get("enrollment_rows")
    enrollment_gated = composite.get("enrollment_gated_rows")
    if (
        type(enrollment_rows) is not int
        or type(enrollment_gated) is not int
        or enrollment_rows
        != enrollment_gated + len(stage_one_calibration)
        or type(composite.get("enrollment_capture_length")) is not int
        or composite.get("enrollment_capture_length") != 16384
    ):
        raise PackageError("composite enrollment row accounting differs")
    expected_threshold = _numpy_linear_quantile(
        composite_calibration,
        COMPOSITE_THRESHOLD_QUANTILE,
    )
    if (
        type(composite.get("threshold")) is not float
        or composite.get("threshold") != expected_threshold
    ):
        raise PackageError(
            "composite threshold is not q97 of its stored calibration"
        )
    return frontend


PARITY_ROW_BASE_KEYS = {
    "capture_length",
    "classifier",
    "classifier_executed",
    "classifier_standardized_features",
    "composite_score",
    "decision_label",
    "family",
    "iq",
    "name",
    "packed_iq",
    "public_result",
    "raw_features",
    "rejected_stage",
    "rejector_closed_winner_index",
    "rejector_closed_winner_label",
    "rejector_standardized_features",
    "stage_one",
    "stage_one_survivor_rank",
    "stage_two",
    "staged_score",
    "staged_threshold",
    "standardized_features",
}
PARITY_KNOWN_ROW_KEYS = {"class_index", "class_label", "corpus_index"}
PARITY_CAUSAL_PREFIX_ROW_KEYS = {"parity_source_name"}
PARITY_STAGE_ONE_KEYS = {
    "capture_length",
    "evaluated_length",
    "causal_prefix_applied",
    "features",
    "score",
    "rank",
    "gated",
    "threshold_score",
}
PARITY_STAGE_TWO_KEYS = {
    "runtime_role",
    "real_embedding",
    "complex_embedding",
    "fused_embedding",
    "predicted_class_index",
    "predicted_class_label",
    "squared_prototype_distances",
    "lof",
    "lof_rank",
    "geometry_statistic",
    "geometry_deviation",
    "geometry_rank",
    "combined_raw",
    "score",
    "threshold",
}
PARITY_CLASSIFIER_KEYS = {
    "runtime_role",
    "real_embedding",
    "complex_embedding",
    "fused_embedding",
    "predicted_class_index",
    "predicted_class_label",
    "squared_prototype_distances",
}
PARITY_LOF_KEYS = {"branch", "weight", "neighbors", "raw", "rank"}


def _float32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", value))[0]


def _numpy_sum_float32(values: list[float]) -> float:
    """Mirror NumPy's float32 fast-axis pairwise sum for short vectors."""
    numbers = [_float32(value) for value in values]
    if len(numbers) < 8:
        total = 0.0
        for value in numbers:
            total = _float32(total + value)
        return total
    # NumPy's pairwise block uses eight accumulators for vectors up to 128
    # values. Runtime embeddings are at most 64 values here.
    accumulators = numbers[:8]
    consumed = 8
    while consumed + 8 <= len(numbers):
        for offset in range(8):
            accumulators[offset] = _float32(
                accumulators[offset] + numbers[consumed + offset]
            )
        consumed += 8
    left = _float32(
        _float32(accumulators[0] + accumulators[1])
        + _float32(accumulators[2] + accumulators[3])
    )
    right = _float32(
        _float32(accumulators[4] + accumulators[5])
        + _float32(accumulators[6] + accumulators[7])
    )
    total = _float32(left + right)
    for value in numbers[consumed:]:
        total = _float32(total + value)
    return total


def _safe_unit_float32(values: list[float], eps: float) -> list[float]:
    canonical = [_float32(value) for value in values]
    squared = [_float32(value * value) for value in canonical]
    norm = _float32(math.sqrt(_numpy_sum_float32(squared)))
    epsilon = _float32(eps)
    if norm <= epsilon:
        return [1.0, *([0.0] * (len(canonical) - 1))]
    return [_float32(value / norm) for value in canonical]


def _fuse_float32(
    real: list[float],
    complex_embedding: list[float],
    fusion: Mapping[str, Any],
) -> list[float]:
    alpha_real = _float32(fusion["alpha_real"])
    alpha_complex = _float32(fusion["alpha_complex"])
    centered_real = [
        _float32(
            _float32(value)
            - _float32(alpha_real * _float32(center))
        )
        for value, center in zip(real, fusion["real_center"])
    ]
    centered_complex = [
        _float32(
            _float32(value)
            - _float32(alpha_complex * _float32(center))
        )
        for value, center in zip(
            complex_embedding, fusion["complex_center"]
        )
    ]
    unit_real = _safe_unit_float32(centered_real, fusion["eps"])
    unit_complex = _safe_unit_float32(centered_complex, fusion["eps"])
    weight_real = _float32(fusion["weight_real"])
    real_scale = _float32(math.sqrt(weight_real))
    complex_scale = _float32(
        math.sqrt(_float32(1.0 - fusion["weight_real"]))
    )
    concatenated = [
        *[_float32(real_scale * value) for value in unit_real],
        *[_float32(complex_scale * value) for value in unit_complex],
    ]
    return _safe_unit_float32(concatenated, fusion["eps"])


def _standardize_float32(
    raw: list[float], standardization: Mapping[str, list[float]]
) -> list[float]:
    return [
        _float32(
            _float32(_float32(value) - _float32(mean))
            / _float32(std)
        )
        for value, mean, std in zip(
            raw, standardization["mean"], standardization["std"]
        )
    ]


def _squared_distances_float32(
    fused: list[float], prototypes: list[list[float]]
) -> list[float]:
    """Mirror NumPy float32 subtraction, square and row-wise sum."""
    output: list[float] = []
    for prototype in prototypes:
        squared_values: list[float] = []
        for value, center in zip(fused, prototype):
            delta = _float32(_float32(value) - _float32(center))
            squared_values.append(_float32(delta * delta))
        output.append(_numpy_sum_float32(squared_values))
    return output


def _verify_parity_prediction(
    value: Any,
    *,
    label: str,
    expected_keys: set[str],
    expected_role: str,
    embed_dimensions: Mapping[str, int],
    classes: list[str],
    prototypes: list[list[float]],
    fusion: Mapping[str, Any],
) -> tuple[int, str, list[float]]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise PackageError(f"{label} field set differs")
    _require_exact_scalar(
        value.get("runtime_role"), expected_role, f"{label}.runtime_role"
    )
    real = _float_vector(
        value.get("real_embedding"), f"{label}.real_embedding"
    )
    complex_embedding = _float_vector(
        value.get("complex_embedding"), f"{label}.complex_embedding"
    )
    fused = _float_vector(
        value.get("fused_embedding"), f"{label}.fused_embedding"
    )
    if (
        len(real) != embed_dimensions["real"]
        or len(complex_embedding) != embed_dimensions["complex"]
        or len(fused)
        != embed_dimensions["real"] + embed_dimensions["complex"]
    ):
        raise PackageError(f"{label} embedding widths differ from its role")
    expected_fused = _fuse_float32(real, complex_embedding, fusion)
    _require_exact_scalar(
        fused,
        expected_fused,
        f"{label}.fused_embedding from branch embeddings",
    )
    predicted = value.get("predicted_class_index")
    if type(predicted) is not int or not 0 <= predicted < len(classes):
        raise PackageError(f"{label}.predicted_class_index is invalid")
    predicted_label = value.get("predicted_class_label")
    _require_exact_scalar(
        predicted_label,
        classes[predicted],
        f"{label}.predicted_class_label",
    )
    distances = _float_vector(
        value.get("squared_prototype_distances"),
        f"{label}.squared_prototype_distances",
        nonnegative=True,
    )
    if len(distances) != len(classes):
        raise PackageError(f"{label} prototype distance width differs")
    expected_distances = _squared_distances_float32(fused, prototypes)
    _require_exact_scalar(
        distances,
        expected_distances,
        f"{label}.squared_prototype_distances from fused embedding",
    )
    winner = min(range(len(distances)), key=distances.__getitem__)
    _require_exact_scalar(
        predicted, winner, f"{label}.predicted_class_index from distances"
    )
    return predicted, predicted_label, distances


def _verify_parity_stage_two(
    value: Any,
    *,
    label: str,
    policy: Mapping[str, Any],
    embed_dimensions: Mapping[str, int],
    classes: list[str],
    prototypes: list[list[float]],
    fusion: Mapping[str, Any],
) -> tuple[float, int, str]:
    predicted, predicted_label, _distances = _verify_parity_prediction(
        value,
        label=label,
        expected_keys=PARITY_STAGE_TWO_KEYS,
        expected_role=REJECTOR_ROLE,
        embed_dimensions=embed_dimensions,
        classes=classes,
        prototypes=prototypes,
        fusion=fusion,
    )
    components = policy["stage_two"]["lof_components"]
    component_by_branch = {
        component["branch"]: component for component in components
    }
    lof = value.get("lof")
    if type(lof) is not list or len(lof) != len(EXPECTED_LOF_COMPONENTS):
        raise PackageError(f"{label}.lof must contain both frozen branches")
    seen: set[str] = set()
    weighted_rank = 0.0
    for index, score in enumerate(lof):
        score_label = f"{label}.lof[{index}]"
        if not isinstance(score, Mapping) or set(score) != PARITY_LOF_KEYS:
            raise PackageError(f"{score_label} field set differs")
        branch = score.get("branch")
        if branch not in EXPECTED_LOF_COMPONENTS or branch in seen:
            raise PackageError(f"{score_label}.branch differs or is duplicated")
        seen.add(branch)
        expected = EXPECTED_LOF_COMPONENTS[branch]
        _require_exact_scalar(
            score.get("weight"), expected["weight"], f"{score_label}.weight"
        )
        _require_exact_scalar(
            score.get("neighbors"),
            expected["neighbors"],
            f"{score_label}.neighbors",
        )
        raw = _finite_float(score.get("raw"), f"{score_label}.raw")
        calibration = component_by_branch[branch]["calibration"]
        expected_rank = bisect_left(calibration, raw) / (
            len(calibration) + 1
        )
        _require_exact_scalar(
            score.get("rank"), expected_rank, f"{score_label}.rank"
        )
        weighted_rank += expected["weight"] * expected_rank
    if seen != set(EXPECTED_LOF_COMPONENTS):
        raise PackageError(f"{label}.lof branch set differs")
    _require_exact_scalar(
        value.get("lof_rank"), weighted_rank, f"{label}.lof_rank"
    )
    geometry_statistic = _finite_float(
        value.get("geometry_statistic"), f"{label}.geometry_statistic"
    )
    deviation = _finite_float(
        value.get("geometry_deviation"), f"{label}.geometry_deviation"
    )
    if deviation < 0.0:
        raise PackageError(f"{label}.geometry_deviation must be nonnegative")
    stage_two_policy = policy["stage_two"]["policy"]
    expected_deviation = abs(
        (
            geometry_statistic
            - stage_two_policy["class_geometry_mean"][predicted]
        )
        / stage_two_policy["class_geometry_scale"][predicted]
    )
    _require_exact_scalar(
        deviation,
        expected_deviation,
        f"{label}.geometry_deviation from predicted-class geometry",
    )
    geometry_calibration = stage_two_policy["geometry_calibration"]
    geometry_rank = bisect_left(geometry_calibration, deviation) / (
        len(geometry_calibration) + 1
    )
    _require_exact_scalar(
        value.get("geometry_rank"),
        geometry_rank,
        f"{label}.geometry_rank",
    )
    combined = (
        stage_two_policy["branch_lof_rank_weight"] * weighted_rank
        + stage_two_policy["geometry_weight"] * geometry_rank
    )
    _require_exact_scalar(
        value.get("combined_raw"), combined, f"{label}.combined_raw"
    )
    combined_calibration = stage_two_policy["combined_calibration"]
    stage_two_score = bisect_left(combined_calibration, combined) / (
        len(combined_calibration) + 1
    )
    _require_exact_scalar(
        value.get("score"), stage_two_score, f"{label}.score"
    )
    _require_exact_scalar(
        value.get("threshold"),
        stage_two_policy["threshold"],
        f"{label}.threshold",
    )
    return stage_two_score, predicted, predicted_label


def _load_openset_export(
    directory: Path,
    rejector: Mapping[str, Any],
    classifier: Mapping[str, Any],
) -> dict[str, Any]:
    if directory.is_symlink() or not directory.is_dir():
        raise PackageError("open-set export must be a non-symlink directory")
    manifest_path = directory / OPENSET_MANIFEST
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema") != OPENSET_EXPORT_SCHEMA
        or manifest.get("status") != STAGING_STATUS
        or manifest.get("candidate_id") != CANDIDATE_ID
    ):
        raise PackageError("open-set export is not the dual staging schema")
    _require_exact_scalar(
        manifest.get("schema_version"), 1, "open-set export schema_version"
    )
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != {
        OPENSET_POLICY,
        DUAL_BINDING,
        OPENSET_PARITY,
    }:
        raise PackageError("open-set export output set differs from dual contract")
    paths = {name: directory / name for name in outputs}
    records = {
        name: _verify_record(paths[name], outputs[name], f"open-set {name}")
        for name in outputs
    }

    role_exports = manifest.get("role_exports")
    if not isinstance(role_exports, Mapping) or set(role_exports) != {
        "rejector",
        "classifier",
    }:
        raise PackageError("open-set manifest lacks exact role exports")
    for label, role, source, weights_name, probe_name in (
        ("rejector", REJECTOR_ROLE, rejector, REJECTOR_WEIGHTS, REJECTOR_PROBE),
        (
            "classifier",
            CLASSIFIER_ROLE,
            classifier,
            CLASSIFIER_WEIGHTS,
            CLASSIFIER_PROBE,
        ),
    ):
        value = role_exports[label]
        if (
            not isinstance(value, Mapping)
            or value.get("runtime_role") != role
            or value.get("manifest_sha256")
            != source["manifest_record"]["sha256"]
        ):
            raise PackageError(f"open-set manifest {label} role link differs")
        _verify_linked_record(
            value.get("weights"),
            name=weights_name,
            expected=source["weights_record"],
            label=f"open-set manifest {label} weights",
        )
        _verify_linked_record(
            value.get("probe"),
            name=probe_name,
            expected=source["probe_record"],
            label=f"open-set manifest {label} probe",
        )

    policy = read_json(paths[OPENSET_POLICY])
    if (
        policy.get("schema") != OPENSET_SCHEMA
        or policy.get("status") != STAGING_STATUS
    ):
        raise PackageError("open-set policy schema/status differs")
    _require_exact_scalar(
        policy.get("schema_version"),
        OPENSET_SCHEMA_VERSION,
        "open-set policy schema_version",
    )
    policy_frontend = _verify_policy_schema4(
        policy, rejector["embed_dimensions"]
    )
    if (
        rejector["frontend"] != classifier["frontend"]
        or policy_frontend != rejector["frontend"]
    ):
        raise PackageError("runtime-role/open-set frontends differ")
    if rejector["classes"] != classifier["classes"]:
        raise PackageError("runtime-role class orders differ")
    if (
        len(policy["stage_two"]["policy"]["class_geometry_mean"])
        != len(rejector["classes"])
    ):
        raise PackageError("open-set class geometry differs from class order")
    if paths[OPENSET_POLICY].stat().st_size >= MAX_DEPLOYABLE_BYTES:
        raise PackageError("open-set policy is not below the 25 MiB limit")
    provenance = policy.get("provenance")
    provenance = _verify_evidence_provenance(
        provenance, "open-set policy provenance"
    )
    runtime_roles = provenance.get("runtime_roles")
    if not isinstance(runtime_roles, Mapping):
        raise PackageError("open-set policy has no role provenance")
    for label, role, source, filename in (
        ("rejector", REJECTOR_ROLE, rejector, REJECTOR_WEIGHTS),
        ("classifier", CLASSIFIER_ROLE, classifier, CLASSIFIER_WEIGHTS),
    ):
        value = runtime_roles.get(label)
        if (
            not isinstance(value, Mapping)
            or value.get("runtime_role") != role
            or value.get("browser_asset") != filename
            or value.get("browser_asset_sha256")
            != source["weights_record"]["sha256"]
            or value.get("browser_export_manifest_sha256")
            != source["manifest_record"]["sha256"]
            or value.get("runtime_bundle_manifest_sha256")
            != source["bundle_sha256"]
        ):
            raise PackageError(f"open-set policy {label} provenance differs")
        _sha(
            value.get("fusion_directory_sha256"),
            f"open-set policy {label} fusion directory",
        )

    binding = read_json(paths[DUAL_BINDING])
    _verify_binding(
        binding,
        rejector,
        classifier,
        records[OPENSET_POLICY],
        policy_frontend,
    )
    openset_binding = binding["openset_policy"]
    if (
        provenance.get("staged_validation_report_sha256")
        != openset_binding["staged_validation_report_sha256"]
        or provenance.get("staged_artifact_sha256")
        != openset_binding["staged_artifacts_sha256"]
    ):
        raise PackageError("policy provenance differs from binding evidence pins")
    parity = read_json(paths[OPENSET_PARITY])
    if (
        parity.get("schema") != "time-domain-openset-parity-v1"
        or parity.get("status") != STAGING_STATUS
        or parity.get("candidate_id") != CANDIDATE_ID
        or parity.get("policy_version") != STAGED_POLICY_VERSION
        or parity.get("policy_kind") != STAGED_POLICY_KIND
    ):
        raise PackageError("parity evidence is not the dual-role fixture")
    _require_exact_scalar(
        parity.get("contract"), policy["contract"], "parity contract"
    )
    _require_exact_scalar(
        parity.get("classes"), rejector["classes"], "parity classes"
    )
    parity_frontend = _verify_frontend(
        parity.get("frontend"),
        "parity frontend",
        require_packed_length=True,
    )
    if parity_frontend != policy_frontend:
        raise PackageError("parity frontend differs from runtime frontend")
    for name, wanted in (
        ("schema_version", PARITY_SCHEMA_VERSION),
        ("policy_schema", STAGED_POLICY_SCHEMA),
        ("design_novelty_seed", DESIGN_NOVELTY_SEED),
        ("release_seed_not_spent", RELEASE_SEED_NOT_SPENT),
    ):
        _require_exact_scalar(
            parity.get(name), wanted, f"parity evidence {name}"
        )
    _require_exact_int_list(
        parity.get("validation_novelty_seeds"),
        VALIDATION_NOVELTY_SEEDS,
        "parity validation_novelty_seeds",
    )
    _require_exact_scalar(
        parity.get("parity_novelty_seed"),
        PARITY_NOVELTY_SEED,
        "parity parity_novelty_seed",
    )
    _require_exact_scalar(
        parity.get("parity_seed_is_evaluation_evidence"),
        False,
        "parity parity_seed_is_evaluation_evidence",
    )
    _require_exact_int_list(
        parity.get("fixture_lengths"),
        REQUIRED_PARITY_LENGTHS,
        "parity fixture_lengths",
    )
    parity_provenance = _verify_evidence_provenance(
        parity.get("provenance"), "parity provenance"
    )
    if parity_provenance != provenance:
        raise PackageError("parity and policy provenance differ")
    _require_exact_scalar(
        parity.get("staged_threshold"),
        policy["composite"]["threshold"],
        "parity staged_threshold",
    )
    _require_exact_scalar(
        parity.get("stage_two_threshold"),
        policy["stage_two"]["policy"]["threshold"],
        "parity stage_two_threshold",
    )
    causal_prefix = parity.get("causal_prefix_parity")
    if (
        not isinstance(causal_prefix, Mapping)
        or set(causal_prefix)
        != {
            "capture_length",
            "evaluated_length",
            "row_names",
            "covers_gated_and_survivor_paths",
        }
    ):
        raise PackageError("parity causal-prefix evidence is missing")
    for name, wanted in (
        ("capture_length", 32768),
        ("evaluated_length", 16384),
        ("covers_gated_and_survivor_paths", True),
    ):
        _require_exact_scalar(
            causal_prefix.get(name), wanted, f"parity causal_prefix {name}"
        )
    role_contract = parity.get("role_contract")
    if not isinstance(role_contract, Mapping):
        raise PackageError("parity role contract is missing")
    for name in (
        "classifier_runs_only_after_rejector_acceptance",
        "public_known_label_from_classifier_only",
    ):
        _require_exact_scalar(
            role_contract.get(name), True, f"parity role_contract {name}"
        )
    rows = parity.get("rows")
    if type(rows) is not list or not rows:
        raise PackageError("parity rows must be a non-empty evidence list")
    derived = {
        "rows": len(rows),
        "known": 0,
        "noise": 0,
        "chirp": 0,
        "probe": 0,
        "gated_stage_one": 0,
        "rejected_stage_two": 0,
        "accepted": 0,
        "classifier_executed": 0,
        "accepted_role_winner_disagreements": 0,
    }
    disagreement_rows: list[str] = []
    names: set[str] = set()
    observed_lengths: set[int] = set()
    causal_prefix_row_names: list[str] = []
    causal_prefix_gated: set[bool] = set()
    classes = rejector["classes"]
    composite_policy = policy["composite"]
    stage_one_calibration = composite_policy["stage_one_calibration_raw"]
    for row_index, row in enumerate(rows):
        row_label = f"parity rows[{row_index}]"
        if not isinstance(row, Mapping):
            raise PackageError("parity row is not an evidence object")
        family = row.get("family")
        capture_length = row.get("capture_length")
        expected_keys = set(PARITY_ROW_BASE_KEYS)
        if family == "known":
            expected_keys.update(PARITY_KNOWN_ROW_KEYS)
        if capture_length == 32768:
            expected_keys.update(PARITY_CAUSAL_PREFIX_ROW_KEYS)
        if set(row) != expected_keys:
            raise PackageError(f"{row_label} full field set differs")
        name = row.get("name")
        if type(name) is not str or not name or name in names:
            raise PackageError(f"{row_label}.name is invalid or duplicated")
        names.add(name)
        family = row.get("family")
        if (
            type(family) is not str
            or family not in {"known", "noise", "chirp", "probe"}
        ):
            raise PackageError("parity row family is invalid")
        derived[family] += 1
        if (
            type(capture_length) is not int
            or capture_length not in REQUIRED_PARITY_LENGTHS
        ):
            raise PackageError(f"{row_label}.capture_length is invalid")
        observed_lengths.add(capture_length)

        if family == "known":
            class_index = row.get("class_index")
            if (
                type(class_index) is not int
                or not 0 <= class_index < len(classes)
            ):
                raise PackageError(f"{row_label}.class_index is invalid")
            _require_exact_scalar(
                row.get("class_label"),
                classes[class_index],
                f"{row_label}.class_label",
            )
            if type(row.get("corpus_index")) is not int or row["corpus_index"] < 0:
                raise PackageError(f"{row_label}.corpus_index is invalid")
        if capture_length == 32768:
            parity_source_name = row.get("parity_source_name")
            if type(parity_source_name) is not str or not parity_source_name:
                raise PackageError(
                    f"{row_label}.parity_source_name is invalid"
                )
            causal_prefix_row_names.append(name)

        iq = row.get("iq")
        if not isinstance(iq, Mapping) or set(iq) != {
            "in_phase",
            "quadrature",
        }:
            raise PackageError(f"{row_label}.iq field set differs")
        for channel in ("in_phase", "quadrature"):
            values = _float_vector(iq.get(channel), f"{row_label}.iq.{channel}")
            if len(values) != capture_length:
                raise PackageError(
                    f"{row_label}.iq.{channel} length differs from capture"
                )
        packed_iq = row.get("packed_iq")
        if not isinstance(packed_iq, Mapping) or set(packed_iq) != {
            "in_phase",
            "quadrature",
        }:
            raise PackageError(f"{row_label}.packed_iq field set differs")
        for channel in ("in_phase", "quadrature"):
            values = _float_vector(
                packed_iq.get(channel),
                f"{row_label}.packed_iq.{channel}",
            )
            if len(values) != policy_frontend["packed_length"]:
                raise PackageError(
                    f"{row_label}.packed_iq.{channel} width differs"
                )

        raw_features = _float_vector(
            row.get("raw_features"), f"{row_label}.raw_features"
        )
        standardized = _float_vector(
            row.get("standardized_features"),
            f"{row_label}.standardized_features",
        )
        rejector_standardized = _float_vector(
            row.get("rejector_standardized_features"),
            f"{row_label}.rejector_standardized_features",
        )
        if not (
            len(raw_features)
            == len(standardized)
            == len(rejector_standardized)
            == TIME_DOMAIN_FEATURE_COUNT
        ):
            raise PackageError(f"{row_label} frontend feature widths differ")
        expected_rejector_standardized = _standardize_float32(
            raw_features, rejector["feature_standardization"]
        )
        _require_exact_scalar(
            rejector_standardized,
            expected_rejector_standardized,
            f"{row_label}.rejector_standardized_features from raw features",
        )
        _require_exact_scalar(
            standardized,
            rejector_standardized,
            f"{row_label}.standardized_features rejector alias",
        )

        stage_one = row.get("stage_one")
        if (
            not isinstance(stage_one, Mapping)
            or set(stage_one) != PARITY_STAGE_ONE_KEYS
        ):
            raise PackageError(f"{row_label}.stage_one field set differs")
        _require_exact_scalar(
            stage_one.get("capture_length"),
            capture_length,
            f"{row_label}.stage_one.capture_length",
        )
        causal = capture_length == 32768
        evaluated_length = 16384 if causal else capture_length
        _require_exact_scalar(
            stage_one.get("evaluated_length"),
            evaluated_length,
            f"{row_label}.stage_one.evaluated_length",
        )
        _require_exact_scalar(
            stage_one.get("causal_prefix_applied"),
            causal,
            f"{row_label}.stage_one.causal_prefix_applied",
        )
        stage_one_features = _float_vector(
            stage_one.get("features"), f"{row_label}.stage_one.features"
        )
        if len(stage_one_features) != len(PREFILTER_FEATURE_NAMES):
            raise PackageError(f"{row_label}.stage_one feature width differs")
        stage_one_score = _finite_float(
            stage_one.get("score"), f"{row_label}.stage_one.score"
        )
        stage_one_threshold = _finite_float(
            stage_one.get("threshold_score"),
            f"{row_label}.stage_one.threshold_score",
        )
        _require_exact_scalar(
            stage_one_threshold,
            policy["stage_one"]["models"][str(evaluated_length)][
                "threshold_score"
            ],
            f"{row_label}.stage_one.threshold_score policy binding",
        )
        stage_one_model = policy["stage_one"]["models"][str(evaluated_length)]
        expected_stage_one_score = stage_one_model["intercept"]
        for feature, mean, scale, coefficient in zip(
            stage_one_features,
            stage_one_model["mean"],
            stage_one_model["scale"],
            stage_one_model["coefficients"],
        ):
            expected_stage_one_score += coefficient * (
                (feature - mean) / scale
            )
        _require_exact_scalar(
            stage_one_score,
            expected_stage_one_score,
            f"{row_label}.stage_one.score from fitted model",
        )
        stage_one_rank = 0.5 * (
            stage_one_score / (1.0 + abs(stage_one_score))
        ) + 0.5
        _require_exact_scalar(
            stage_one.get("rank"),
            stage_one_rank,
            f"{row_label}.stage_one.rank",
        )
        gated = stage_one_score >= stage_one_threshold
        _require_exact_scalar(
            stage_one.get("gated"), gated, f"{row_label}.stage_one.gated"
        )
        if causal:
            causal_prefix_gated.add(gated)

        _require_exact_scalar(
            row.get("staged_threshold"),
            composite_policy["threshold"],
            f"{row_label}.staged_threshold",
        )
        if gated:
            for field in (
                "stage_two",
                "stage_one_survivor_rank",
                "composite_score",
                "rejector_closed_winner_index",
                "rejector_closed_winner_label",
            ):
                _require_exact_scalar(
                    row.get(field), None, f"{row_label}.{field}"
                )
            expected_rejected: int | None = 1
            expected_staged_score = 1.0 + stage_one_rank
            expected_decision = "noise"
            expected_public = {
                "outcome": "noise",
                "label": "noise",
                "knownLabel": None,
                "knownWinnerIndex": None,
                "squaredPrototypeDistances": None,
            }
        else:
            stage_two_score, rejector_index, rejector_label = (
                _verify_parity_stage_two(
                    row.get("stage_two"),
                    label=f"{row_label}.stage_two",
                    policy=policy,
                    embed_dimensions=rejector["embed_dimensions"],
                    classes=classes,
                    prototypes=rejector["prototypes"],
                    fusion=rejector["fusion"],
                )
            )
            _require_exact_scalar(
                row.get("rejector_closed_winner_index"),
                rejector_index,
                f"{row_label}.rejector_closed_winner_index",
            )
            _require_exact_scalar(
                row.get("rejector_closed_winner_label"),
                rejector_label,
                f"{row_label}.rejector_closed_winner_label",
            )
            survivor_rank = bisect_left(
                stage_one_calibration, stage_one_score
            ) / (len(stage_one_calibration) + 1)
            _require_exact_scalar(
                row.get("stage_one_survivor_rank"),
                survivor_rank,
                f"{row_label}.stage_one_survivor_rank",
            )
            composite_score = max(stage_two_score, survivor_rank)
            _require_exact_scalar(
                row.get("composite_score"),
                composite_score,
                f"{row_label}.composite_score",
            )
            expected_rejected = (
                2
                if composite_score > composite_policy["threshold"]
                else None
            )
            expected_staged_score = composite_score
            expected_decision = (
                "unknown" if expected_rejected == 2 else None
            )
            expected_public = (
                {
                    "outcome": "unknown",
                    "label": "unknown",
                    "knownLabel": None,
                    "knownWinnerIndex": None,
                    "squaredPrototypeDistances": None,
                }
                if expected_rejected == 2
                else None
            )

        _require_exact_scalar(
            row.get("rejected_stage"),
            expected_rejected,
            f"{row_label}.rejected_stage",
        )
        _require_exact_scalar(
            row.get("staged_score"),
            expected_staged_score,
            f"{row_label}.staged_score",
        )
        classifier_result = row.get("classifier")
        classifier_executed = row.get("classifier_executed")
        if expected_rejected is None:
            _require_exact_scalar(
                classifier_executed,
                True,
                f"{row_label}.classifier_executed",
            )
            classifier_standardized = _float_vector(
                row.get("classifier_standardized_features"),
                f"{row_label}.classifier_standardized_features",
            )
            if len(classifier_standardized) != TIME_DOMAIN_FEATURE_COUNT:
                raise PackageError(
                    f"{row_label} classifier feature width differs"
                )
            expected_classifier_standardized = _standardize_float32(
                raw_features, classifier["feature_standardization"]
            )
            _require_exact_scalar(
                classifier_standardized,
                expected_classifier_standardized,
                f"{row_label}.classifier_standardized_features from raw "
                "features",
            )
            classifier_index, classifier_label, classifier_distances = (
                _verify_parity_prediction(
                    classifier_result,
                    label=f"{row_label}.classifier",
                    expected_keys=PARITY_CLASSIFIER_KEYS,
                    expected_role=CLASSIFIER_ROLE,
                    embed_dimensions=classifier["embed_dimensions"],
                    classes=classes,
                    prototypes=classifier["prototypes"],
                    fusion=classifier["fusion"],
                )
            )
            expected_decision = classifier_label
            expected_public = {
                "outcome": "known",
                "label": classifier_label,
                "knownLabel": classifier_label,
                "knownWinnerIndex": classifier_index,
                "squaredPrototypeDistances": classifier_distances,
            }
            derived["accepted"] += 1
            derived["classifier_executed"] += 1
            if rejector_label != classifier_label:
                derived["accepted_role_winner_disagreements"] += 1
                disagreement_rows.append(name)
        else:
            if classifier_executed is not False or classifier_result is not None:
                raise PackageError(
                    f"{row_label}: rejected parity row serialized "
                    "classifier output"
                )
            _require_exact_scalar(
                row.get("classifier_standardized_features"),
                None,
                f"{row_label}.classifier_standardized_features",
            )
            derived[
                "gated_stage_one"
                if expected_rejected == 1
                else "rejected_stage_two"
            ] += 1
        _require_exact_scalar(
            row.get("decision_label"),
            expected_decision,
            f"{row_label}.decision_label",
        )
        _require_exact_scalar(
            row.get("public_result"),
            expected_public,
            f"{row_label}.public_result",
        )
    if observed_lengths != set(REQUIRED_PARITY_LENGTHS):
        raise PackageError(
            "parity row capture-length coverage must be exactly "
            f"{REQUIRED_PARITY_LENGTHS}"
        )
    _require_exact_scalar(
        causal_prefix.get("row_names"),
        causal_prefix_row_names,
        "parity causal_prefix row_names",
    )
    if causal_prefix_gated != {False, True}:
        raise PackageError(
            "parity causal-prefix rows do not cover gated and survivor paths"
        )
    counts = parity.get("counts")
    if not isinstance(counts, Mapping) or set(counts) != set(derived):
        raise PackageError("parity count key set differs")
    for name, expected_count in derived.items():
        _require_exact_scalar(
            counts.get(name), expected_count, f"parity counts.{name}"
        )
    _require_exact_scalar(
        parity.get("accepted_role_winner_disagreement_rows"),
        disagreement_rows,
        "parity accepted_role_winner_disagreement_rows",
    )
    return {
        "directory": directory,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "manifest_record": _record(manifest_path),
        "paths": paths,
        "records": records,
        "policy": policy,
        "binding": binding,
        "parity": parity,
    }


def _materialize(destination: Path, files: Mapping[str, bytes]) -> None:
    if destination.exists():
        if destination.is_symlink() or not destination.is_dir():
            raise PackageError("destination must be absent or an empty directory")
        if next(destination.iterdir(), None) is not None:
            raise PackageError("destination must be empty")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
        )
    )
    moved = False
    removed_empty = False
    try:
        for name in DEPLOYABLE_ASSETS:
            (temporary / name).write_bytes(files[name])
        (temporary / PACKAGE_MANIFEST).write_bytes(files[PACKAGE_MANIFEST])
        if destination.exists():
            destination.rmdir()
            removed_empty = True
        temporary.rename(destination)
        moved = True
    except OSError as exc:
        raise PackageError("cannot atomically install staging package") from exc
    finally:
        if not moved:
            shutil.rmtree(temporary, ignore_errors=True)
            if removed_empty and not destination.exists():
                destination.mkdir()


def package(
    rejector_export: Path,
    classifier_export: Path,
    openset_export: Path,
    destination: Path,
) -> dict[str, Any]:
    destination = Path(destination).resolve()
    if destination in {LIVE_V2_ASSETS, LEGACY_V3_STAGING}:
        raise PackageError("refusing live v2 or legacy single-fusion staging")
    if (
        LIVE_V2_ASSETS in destination.parents
        or LEGACY_V3_STAGING in destination.parents
    ):
        raise PackageError("refusing a descendant of a legacy asset directory")

    rejector = _load_role_export(
        Path(rejector_export).resolve(),
        role=REJECTOR_ROLE,
        weights_name=REJECTOR_WEIGHTS,
        probe_name=REJECTOR_PROBE,
    )
    classifier = _load_role_export(
        Path(classifier_export).resolve(),
        role=CLASSIFIER_ROLE,
        weights_name=CLASSIFIER_WEIGHTS,
        probe_name=CLASSIFIER_PROBE,
    )
    if (
        rejector["weights_record"]["sha256"]
        == classifier["weights_record"]["sha256"]
        or rejector["bundle_sha256"] == classifier["bundle_sha256"]
    ):
        raise PackageError("rejector and classifier role assets are aliased")
    openset = _load_openset_export(
        Path(openset_export).resolve(),
        rejector,
        classifier,
    )

    source_paths = {
        REJECTOR_WEIGHTS: rejector["weights_path"],
        CLASSIFIER_WEIGHTS: classifier["weights_path"],
        OPENSET_POLICY: openset["paths"][OPENSET_POLICY],
        DUAL_BINDING: openset["paths"][DUAL_BINDING],
    }
    for name, path in source_paths.items():
        if path.stat().st_size >= MAX_DEPLOYABLE_BYTES:
            raise PackageError(
                f"deployable asset {name} is not below 25 MiB"
            )

    payloads = {
        REJECTOR_WEIGHTS: rejector["weights"],
        CLASSIFIER_WEIGHTS: classifier["weights"],
        OPENSET_POLICY: openset["policy"],
        DUAL_BINDING: openset["binding"],
    }
    asset_records = {
        REJECTOR_WEIGHTS: _asset_record(
            source_paths[REJECTOR_WEIGHTS],
            payloads[REJECTOR_WEIGHTS],
            runtime_role=REJECTOR_ROLE,
        ),
        CLASSIFIER_WEIGHTS: _asset_record(
            source_paths[CLASSIFIER_WEIGHTS],
            payloads[CLASSIFIER_WEIGHTS],
            runtime_role=CLASSIFIER_ROLE,
        ),
        OPENSET_POLICY: _asset_record(
            source_paths[OPENSET_POLICY], payloads[OPENSET_POLICY]
        ),
        DUAL_BINDING: _asset_record(
            source_paths[DUAL_BINDING], payloads[DUAL_BINDING]
        ),
    }
    binding = openset["binding"]
    manifest = {
        "schema": PACKAGE_SCHEMA,
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "status": STAGING_STATUS,
        "candidate_id": CANDIDATE_ID,
        "architecture": {
            "execution_order": [
                "stage_one_noise_gate",
                "rejector_known_unknown",
                "classifier_known_label",
            ],
            "classifier_runs_only_after_rejector_acceptance": True,
            "public_known_label_from_classifier_only": True,
        },
        "assets": asset_records,
        "roles": {
            "rejector": {
                "runtime_role": REJECTOR_ROLE,
                "responsibility": REJECTOR_RESPONSIBILITY,
                "asset": asset_records[REJECTOR_WEIGHTS],
                "source_bundle_manifest_sha256": rejector["bundle_sha256"],
                "fusion_directory_sha256": binding["roles"]["rejector"][
                    "fusion_directory_sha256"
                ],
            },
            "classifier": {
                "runtime_role": CLASSIFIER_ROLE,
                "responsibility": CLASSIFIER_RESPONSIBILITY,
                "asset": asset_records[CLASSIFIER_WEIGHTS],
                "source_bundle_manifest_sha256": classifier["bundle_sha256"],
                "fusion_directory_sha256": binding["roles"]["classifier"][
                    "fusion_directory_sha256"
                ],
            },
        },
        "openset_policy": {
            **asset_records[OPENSET_POLICY],
            "fitted_rejector_runtime_bundle_manifest_sha256": (
                binding["openset_policy"][
                    "fitted_rejector_runtime_bundle_manifest_sha256"
                ]
            ),
            "staged_validation_report_sha256": binding["openset_policy"][
                "staged_validation_report_sha256"
            ],
            "staged_artifacts_sha256": binding["openset_policy"][
                "staged_artifacts_sha256"
            ],
        },
        "dual_binding": asset_records[DUAL_BINDING],
        "external_evidence": {
            "rejector_export_manifest": {
                "path": _source_path(rejector["manifest_path"], destination),
                **rejector["manifest_record"],
                "schema": FUSION_EXPORT_SCHEMA,
                "schema_version": FUSION_EXPORT_SCHEMA_VERSION,
                "runtime_role": REJECTOR_ROLE,
            },
            "classifier_export_manifest": {
                "path": _source_path(classifier["manifest_path"], destination),
                **classifier["manifest_record"],
                "schema": FUSION_EXPORT_SCHEMA,
                "schema_version": FUSION_EXPORT_SCHEMA_VERSION,
                "runtime_role": CLASSIFIER_ROLE,
            },
            "openset_export_manifest": {
                "path": _source_path(openset["manifest_path"], destination),
                **openset["manifest_record"],
                "schema": OPENSET_EXPORT_SCHEMA,
                "schema_version": 1,
                "status": STAGING_STATUS,
            },
            "rejector_probe": {
                "path": _source_path(rejector["probe_path"], destination),
                **rejector["probe_record"],
            },
            "classifier_probe": {
                "path": _source_path(classifier["probe_path"], destination),
                **classifier["probe_record"],
            },
            "parity": {
                "path": _source_path(
                    openset["paths"][OPENSET_PARITY], destination
                ),
                **openset["records"][OPENSET_PARITY],
                "schema": openset["parity"]["schema"],
                "schema_version": openset["parity"]["schema_version"],
                "status": openset["parity"]["status"],
                "packaged": False,
            },
        },
        "size_contract": {
            "maximum_file_bytes_exclusive": MAX_DEPLOYABLE_BYTES,
            "all_deployable_files_below_limit": True,
        },
    }
    manifest_raw = json_bytes(manifest)
    if len(manifest_raw) >= MAX_DEPLOYABLE_BYTES:
        raise PackageError("runtime package manifest is not below 25 MiB")
    files = {
        name: source_paths[name].read_bytes() for name in DEPLOYABLE_ASSETS
    }
    files[PACKAGE_MANIFEST] = manifest_raw
    _materialize(destination, files)
    expected_names = {*DEPLOYABLE_ASSETS, PACKAGE_MANIFEST}
    if {path.name for path in destination.iterdir()} != expected_names:
        raise PackageError("materialized package has unexpected files")
    for name, raw in files.items():
        path = destination / name
        if path.is_symlink() or path.read_bytes() != raw:
            raise PackageError(f"materialized package changed {name}")
        if path.stat().st_size >= MAX_DEPLOYABLE_BYTES:
            raise PackageError(f"materialized package exceeds size limit: {name}")
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--rejector-export", type=Path, default=DEFAULT_REJECTOR_EXPORT
    )
    result.add_argument(
        "--classifier-export", type=Path, default=DEFAULT_CLASSIFIER_EXPORT
    )
    result.add_argument(
        "--openset-export", type=Path, default=DEFAULT_OPENSET_EXPORT
    )
    result.add_argument(
        "--destination", type=Path, default=DEFAULT_DESTINATION
    )
    return result


def main() -> None:
    arguments = parser().parse_args()
    manifest = package(
        arguments.rejector_export,
        arguments.classifier_export,
        arguments.openset_export,
        arguments.destination,
    )
    print(
        f"packaged {len(manifest['assets'])} verified dual v3 runtime assets "
        f"into {arguments.destination}"
    )


if __name__ == "__main__":
    main()
