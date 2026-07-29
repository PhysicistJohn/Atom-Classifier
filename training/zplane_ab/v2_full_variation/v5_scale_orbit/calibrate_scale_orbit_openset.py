"""Leak-safe open-set calibration and evaluation for the v5 scale-orbit route.

This module is intentionally a small adaptation layer over the frozen v4
open-set mathematics.  It reuses only route-independent operations: empirical
ranks, grouped thresholds, instantaneous-phase linearity, public-class
distance composition, and report summaries.  It does *not* reuse the v4
context loader or either v4 raw-current scoring path.

Every trusted-current observation is represented without a selected profile
or selected class.  Its measured ``sample_rate_hz`` and
``native_sample_rate_hz`` are mandatory and the observation is transformed by
``trusted_geometry_canonicalizer`` before pose/noise features,
instantaneous-phase features, or the FFT-free classifier frontend run.
Historical observations explicitly carry ``None`` for both rates and retain
the v4 causal-prefix behaviour unchanged.

The fitting boundary is structural:

* the fusion is loaded from the v5 two-corpus assembly;
* a fixed stage-one score model is supplied before calibration;
* only v5 enrollment observations reach :func:`fit_enrollment_policy`;
* seed-20262904 adaptive-selection observations are accepted only by scoring
  and gate-report functions.

This is development machinery, not release evidence.  It has no sealed-seed
generation entry point and refuses release, sealed, and consumed-test paths.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
V2 = HERE.parent
ZPLANE = V2.parent
TRAINING = ZPLANE.parent
REPO = HERE.parents[3]
V4 = V2 / "v4_current_source"
for path in (TRAINING, ZPLANE, V2, V4, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import assemble_scale_orbit_fusion as fusion_assembler  # noqa: E402
import calibrate_current_source_openset as v4_openset  # noqa: E402
import current_source_data as corpus_data  # noqa: E402
import export_current_source_browser_runtime as browser_export  # noqa: E402
import noise_prefilter  # noqa: E402
import run_scale_orbit_dev as branch_runner  # noqa: E402
import scale_orbit_data  # noqa: E402
import time_domain_invariant_patch_preprocess as td_preprocess  # noqa: E402
import trusted_geometry_canonicalizer as geometry_canonicalizer  # noqa: E402
import validate_identity_firewall_amendment as firewall_validator  # noqa: E402
import validate_pretraining_amendment as amendment_validator  # noqa: E402
import validate_recovery_protocol as recovery_validator  # noqa: E402


POLICY_SCHEMA = "atomos.v5.scale-orbit.external-openset-policy"
POLICY_SCHEMA_VERSION = 1
REPORT_SCHEMA = "atomos.v5.scale-orbit.openset-evaluation"
REPORT_SCHEMA_VERSION = 1
FUSION_SCHEMA = fusion_assembler.FUSION_SCHEMA
COMPOSITE_AUDIT_SCHEMA = fusion_assembler.COMPOSITE_AUDIT_SCHEMA
PUBLIC_CLASSES = tuple(browser_export.PUBLIC_CLASSES)
RUNTIME_INPUT_LENGTHS = tuple(browser_export.RUNTIME_INPUT_LENGTHS)
PROTOTYPE_SOURCE_ROUTES = ("historical", "current")
CURRENT_EFFECTIVE_INPUT_LENGTH = (
    geometry_canonicalizer.OUTPUT_SAMPLES
)
RUNTIME_LENGTHS_BY_PROTOTYPE_SOURCE = {
    "historical": RUNTIME_INPUT_LENGTHS,
    "current": (CURRENT_EFFECTIVE_INPUT_LENGTH,),
}
EXPECTED_PUBLIC_CLASS_SUPPORT_BY_PROTOTYPE_SOURCE = {
    key: tuple(value)
    for key, value in
    v4_openset.EXPECTED_PUBLIC_CLASS_SUPPORT_BY_PROTOTYPE_SOURCE.items()
}
WIFI_HR_DSSS_SOURCE_PROFILE = "current:wifi-hr-dsss-11m"
CONDITIONAL_CELL_MINIMUM_CLOSED_CORRECT_ROWS = 20
REQUIRED_CURRENT_EVALUATION_LENGTHS = (4_096, 8_192, 16_384)
REQUIRED_CURRENT_EVALUATION_SCALES = (1.0, 1.25, 1.5, 2.0)
REQUIRED_CURRENT_EVALUATION_MINIMUM_PAIRS_PER_PROFILE = (
    CONDITIONAL_CELL_MINIMUM_CLOSED_CORRECT_ROWS
)
REQUIRED_CURRENT_PROFILE_PUBLIC_CLASS_MAP = {
    "bluetooth-classic-connected": "bluetooth",
    "bluetooth-le-advertising": "bluetooth",
    "gsm-16qam-higher-symbol-rate-burst": "gsm",
    "gsm-32qam-higher-symbol-rate-burst": "gsm",
    "gsm-8psk-normal-burst": "gsm",
    "gsm-900-loaded-bcch": "gsm",
    "gsm-aqpsk-normal-burst": "gsm",
    "gsm-normal-burst": "gsm",
    "gsm-qpsk-higher-symbol-rate-burst": "gsm",
    "lte-band3-fdd-20m": "ofdm",
    "lte-band38-tdd-10m": "ofdm",
    "lte-etm1.1": "ofdm",
    "lte-etm3.1": "ofdm",
    "lte-etm3.1a": "ofdm",
    "lte-etm3.1b": "ofdm",
    "lte-nbiot-guard-isolated-component": "ofdm",
    "lte-nbiot-inband-isolated-component": "ofdm",
    "lte-ntm": "ofdm",
    "nr-fr1-tm1.1": "ofdm",
    "nr-fr1-tm3.1": "ofdm",
    "nr-fr1-tm3.1a": "ofdm",
    "nr-fr1-tm3.1b": "ofdm",
    "nr-n3-fdd-20m": "ofdm",
    "nr-n78-tdd-100m": "ofdm",
    "nr-nbiot-inband-isolated-component": "ofdm",
    "wifi-hr-dsss-11m": "dsss",
    "wifi-ofdm-20m": "ofdm",
    "wifi6-he-er-su": "ofdm",
    "wifi6-he-mu": "ofdm",
    "wifi6-he-su": "ofdm",
    "wifi6-he-tb": "ofdm",
}
if (
    tuple(RUNTIME_INPUT_LENGTHS) != REQUIRED_CURRENT_EVALUATION_LENGTHS
    or tuple(scale_orbit_data.SCALE_ORBIT_EXPECTED_FACTORS)
        != REQUIRED_CURRENT_EVALUATION_SCALES
    or dict(corpus_data.CURRENT_PROFILE_PUBLIC_CLASS_MAP)
        != REQUIRED_CURRENT_PROFILE_PUBLIC_CLASS_MAP
):
    raise RuntimeError("frozen v5 current evaluation inventory changed")
AMENDMENT_SHA256 = fusion_assembler.AMENDMENT_SHA256
PROTOCOL_SHA256 = scale_orbit_data.SCALE_ORBIT_PROTOCOL_SHA256
SEED_REGISTRY_SHA256 = scale_orbit_data.SCALE_ORBIT_SEED_REGISTRY_SHA256
IDENTITY_FIREWALL_AMENDMENT_SHA256 = (
    scale_orbit_data.SCALE_ORBIT_IDENTITY_FIREWALL_AMENDMENT_SHA256
)
IDENTITY_REJECTION_SHA256 = (
    scale_orbit_data.SCALE_ORBIT_IDENTITY_REJECTION_SHA256
)
IDENTITY_ACCEPTANCE_SHA256 = (
    scale_orbit_data.SCALE_ORBIT_IDENTITY_ACCEPTANCE_SHA256
)
CURRENT_NOVELTY_GEOMETRY_SCHEMA = (
    "atomos.v5.scale-orbit.current-novelty-geometry"
)

V5_ASSEMBLY_DECISION_RULE = (
    "minimum squared distance across source/profile centroids "
    "mapped to each public class"
)
V5_ASSEMBLY_ENROLLMENT_RULE = (
    "each source/profile centroid pools every eligible route view; "
    "historical rows retain eligible prefix views and current rows "
    "contribute canonicalized physical-scale views"
)

FITTING_FIREWALL = {
    "adaptive_selection_rows_used_for_stage_one_fit": 0,
    "adaptive_selection_rows_used_for_prototype_or_enrollment_fit": 0,
    "adaptive_selection_rows_used_for_threshold_or_rank_fit": 0,
    "sealed_validation_rows_used_for_any_fit_or_selection": 0,
    "consumed_historical_test_rows_used_for_any_fit_or_selection": 0,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _executed_source_paths() -> dict[str, Path]:
    """Return every local source imported by the v5 adaptation boundary."""
    sources = dict(v4_openset._executed_source_paths())
    sources.update({
        "v5/calibrate_scale_orbit_openset.py": Path(__file__).resolve(),
        "v5/assemble_scale_orbit_fusion.py":
            Path(fusion_assembler.__file__).resolve(),
        "v5/run_scale_orbit_dev.py": Path(branch_runner.__file__).resolve(),
        "v5/scale_orbit_data.py": Path(scale_orbit_data.__file__).resolve(),
        "v5/trusted_geometry_canonicalizer.py":
            Path(geometry_canonicalizer.__file__).resolve(),
        "v5/validate_pretraining_amendment.py":
            Path(amendment_validator.__file__).resolve(),
        "v5/validate_recovery_protocol.py":
            Path(recovery_validator.__file__).resolve(),
        "v5/validate_identity_firewall_amendment.py":
            Path(firewall_validator.__file__).resolve(),
        "v5/pretraining_amendment.json":
            amendment_validator.AMENDMENT_PATH.resolve(),
        "v5/recovery_protocol.json":
            recovery_validator.PROTOCOL_PATH.resolve(),
        "v5/seed_registry.json": recovery_validator.REGISTRY_PATH.resolve(),
        "v5/identity_firewall_amendment.json":
            firewall_validator.AMENDMENT_PATH.resolve(),
        "v5/seed20262904_identity_rejection.json":
            firewall_validator.REJECTION_PATH.resolve(),
        "v5/seed20262904_identity_firewall_acceptance.json":
            scale_orbit_data.SCALE_ORBIT_IDENTITY_ACCEPTANCE_PATH.resolve(),
        "v4/calibrate_current_source_openset.py":
            Path(v4_openset.__file__).resolve(),
        "v4/export_current_source_browser_runtime.py":
            Path(browser_export.__file__).resolve(),
        "v4/current_source_data.py": Path(corpus_data.__file__).resolve(),
        "training/time_domain_invariant_patch_preprocess.py":
            Path(td_preprocess.__file__).resolve(),
        "training/noise_prefilter.py":
            Path(noise_prefilter.__file__).resolve(),
    })
    return sources


def _source_hashes() -> dict[str, str]:
    return {
        name: _sha256(path)
        for name, path in sorted(_executed_source_paths().items())
    }


def _assert_source_snapshot_unchanged(
    expected: Mapping[str, str],
    *,
    operation: str,
) -> None:
    observed = _source_hashes()
    if dict(expected) == observed:
        return
    changed = sorted(
        key
        for key in set(expected) | set(observed)
        if expected.get(key) != observed.get(key)
    )
    raise RuntimeError(
        f"executed source changed during {operation}: {', '.join(changed)}"
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _validate_contract_files() -> dict[str, Any]:
    if _sha256(amendment_validator.AMENDMENT_PATH) != AMENDMENT_SHA256:
        raise RuntimeError("frozen v5 pre-training amendment bytes changed")
    if _sha256(recovery_validator.PROTOCOL_PATH) != PROTOCOL_SHA256:
        raise RuntimeError("frozen v5 recovery protocol bytes changed")
    if _sha256(recovery_validator.REGISTRY_PATH) != SEED_REGISTRY_SHA256:
        raise RuntimeError("frozen v5 seed registry bytes changed")
    if (
        _sha256(firewall_validator.AMENDMENT_PATH)
        != IDENTITY_FIREWALL_AMENDMENT_SHA256
    ):
        raise RuntimeError("frozen v5 identity-firewall amendment changed")
    if (
        _sha256(firewall_validator.REJECTION_PATH)
        != IDENTITY_REJECTION_SHA256
    ):
        raise RuntimeError("frozen v5 identity-rejection declaration changed")
    if (
        _sha256(scale_orbit_data.SCALE_ORBIT_IDENTITY_ACCEPTANCE_PATH)
        != IDENTITY_ACCEPTANCE_SHA256
    ):
        raise RuntimeError("frozen v5 identity-acceptance declaration changed")
    amendment = amendment_validator.validate_repository_amendment()
    recovery = recovery_validator.validate_repository_contract()
    firewall = firewall_validator.validate_repository()
    return {
        "amendment": amendment,
        "recovery": recovery,
        "identity_firewall": firewall,
        "amendment_sha256": AMENDMENT_SHA256,
        "recovery_protocol_sha256": PROTOCOL_SHA256,
        "seed_registry_sha256": SEED_REGISTRY_SHA256,
        "identity_firewall_amendment_sha256":
            IDENTITY_FIREWALL_AMENDMENT_SHA256,
        "identity_rejection_sha256": IDENTITY_REJECTION_SHA256,
        "identity_acceptance_sha256": IDENTITY_ACCEPTANCE_SHA256,
    }


def validate_prototype_source(value: Any) -> str:
    source = str(value)
    if source not in PROTOTYPE_SOURCE_ROUTES:
        raise ValueError(
            "prototype_source must be exactly 'historical' or 'current'"
        )
    return source


def validate_current_novelty_geometry(
    value: Any,
) -> dict[str, Any]:
    """Validate a caller-frozen, class-independent canonical-domain geometry."""
    if not isinstance(value, Mapping):
        raise ValueError("current novelty geometry must be an object")
    required_keys = {
        "schema",
        "sample_rate_hz",
        "native_sample_rate_hz",
        "sample_rate_ratio",
        "canonical_domain_identity_geometry",
        "selected_profile_or_class_used",
        "frozen_before_adaptive_novelty_scoring",
    }
    if set(value) != required_keys:
        raise ValueError("current novelty geometry schema changed")
    try:
        sample_rate = float(value["sample_rate_hz"])
        native_rate = float(value["native_sample_rate_hz"])
        recorded_ratio = float(value["sample_rate_ratio"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("current novelty geometry rates are invalid") from exc
    actual_ratio = sample_rate / native_rate if native_rate > 0.0 else np.nan
    if (
        value["schema"] != CURRENT_NOVELTY_GEOMETRY_SCHEMA
        or not np.isfinite(sample_rate)
        or not np.isfinite(native_rate)
        or sample_rate <= 0.0
        or native_rate <= 0.0
        or actual_ratio != geometry_canonicalizer.MAX_SAMPLE_RATE_RATIO
        or recorded_ratio != actual_ratio
        or value["canonical_domain_identity_geometry"] is not True
        or value["selected_profile_or_class_used"] is not False
        or value["frozen_before_adaptive_novelty_scoring"] is not True
    ):
        raise ValueError(
            "current novelty geometry must be a frozen, class-independent "
            "2x/native canonical-domain identity geometry"
        )
    return {
        "schema": CURRENT_NOVELTY_GEOMETRY_SCHEMA,
        "sample_rate_hz": sample_rate,
        "native_sample_rate_hz": native_rate,
        "sample_rate_ratio": actual_ratio,
        "canonical_domain_identity_geometry": True,
        "selected_profile_or_class_used": False,
        "frozen_before_adaptive_novelty_scoring": True,
    }


def reject_sensitive_path(path: str | Path, role: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    components = [
        component.lower().replace("_", "-")
        for component in resolved.parts
    ]
    if (
        any("release" in component for component in components)
        or any("sealed" in component for component in components)
        or any(
            "consumed-test" in component or "consumedtest" in component
            for component in components
        )
    ):
        raise ValueError(
            "v5 open-set development refuses release, sealed, or "
            f"consumed-test {role} paths: {resolved}"
        )
    return resolved


@dataclass(frozen=True)
class InferenceObservation:
    """Inference-only observation; deliberately has no profile/class field."""

    iq: np.ndarray
    prototype_source: str
    sample_rate_hz: float | int | None
    native_sample_rate_hz: float | int | None


@dataclass(frozen=True)
class EvaluationObservation:
    """Truth metadata kept outside the inference-only observation."""

    inference: InferenceObservation
    truth_index: int
    source_profile: str
    base_identity: str
    pair_id: str
    observation_length: int
    physical_scale_factor: float


@dataclass(frozen=True)
class PreparedObservation:
    effective_iq: np.ndarray
    prototype_source: str
    observation_length: int
    effective_input_length: int
    canonicalization: Mapping[str, Any] | None
    is_exact_no_signal: bool
    pose_features: np.ndarray | None
    phase_linearity_score: float | None
    packed: np.ndarray | None
    raw_frontend_features: np.ndarray | None
    frontend_context: Mapping[str, Any] | None


@dataclass
class DevelopmentContext:
    fusion: dict[str, Any]
    historical: corpus_data.RawCorpus
    current: scale_orbit_data.ScaleOrbitCorpus
    data: dict[str, Any]
    preprocessing_audit: dict[str, Any]
    module: torch.nn.Module
    device: torch.device
    patch_length: int
    patch_count: int
    target_frac: float


@dataclass(frozen=True)
class _EnrollmentCalibration:
    """Private, internally derived enrollment arrays for policy fitting."""

    stage_one_score_by_length: Mapping[int, np.ndarray]
    phase_linearity_score_by_length: Mapping[int, np.ndarray]
    class_distance_by_length: Mapping[int, np.ndarray]
    calibration_class_distance_by_length: Mapping[int, np.ndarray]
    enrollment_group_by_length: Mapping[int, Sequence[str]]
    enrollment_source_by_length: Mapping[int, Sequence[str]]
    enrollment_identity_by_length: Mapping[int, Sequence[str]]
    audit: Mapping[str, Any]


@dataclass(frozen=True)
class ScoredGateCell:
    source_profile: str
    truth: str
    predicted: str | None
    rejected: bool
    pair_id: str
    observation_length: int
    physical_scale_factor: float

    @property
    def outcome(self) -> str:
        return "unknown" if self.rejected else str(self.predicted)

    @property
    def closed_correct(self) -> bool:
        return self.predicted == self.truth


def _complex_observation(value: Any) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or len(raw) == 0 or not np.iscomplexobj(raw):
        raise ValueError("I/Q observation must be a nonempty complex vector")
    if (
        not np.isfinite(raw.real).all()
        or not np.isfinite(raw.imag).all()
    ):
        raise ValueError("I/Q observation contains NaN or infinity")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.ascontiguousarray(raw, dtype=np.complex64)
    if (
        not np.isfinite(result.real).all()
        or not np.isfinite(result.imag).all()
    ):
        raise ValueError("I/Q observation exceeds float32 range")
    return result


def _historical_effective_length(observation_length: int) -> int:
    legal = [
        length
        for length in RUNTIME_INPUT_LENGTHS
        if length <= int(observation_length)
    ]
    if not legal:
        raise ValueError(
            "historical observation is shorter than the minimum runtime input"
        )
    return max(legal)


def canonicalize_inference_observation(
    iq: Any,
    *,
    prototype_source: Any,
    sample_rate_hz: Any,
    native_sample_rate_hz: Any,
) -> tuple[np.ndarray, int, Mapping[str, Any] | None]:
    """Return the route-effective I/Q before any learned/analytic features.

    Both rate arguments are intentionally required keyword arguments.  They
    must be present for trusted current and must both be ``None`` for
    historical input.  There is no selected-profile or selected-class input.
    """
    source = validate_prototype_source(prototype_source)
    raw = _complex_observation(iq)
    if source == "current":
        if sample_rate_hz is None or native_sample_rate_hz is None:
            raise ValueError(
                "trusted-current scoring requires explicit sample_rate_hz "
                "and native_sample_rate_hz"
            )
        canonical, audit = geometry_canonicalizer.canonicalize(
            raw,
            sample_rate_hz=sample_rate_hz,
            native_sample_rate_hz=native_sample_rate_hz,
        )
        if (
            audit.get("uses_frequency_transform") is not False
            or audit.get("uses_selected_profile_or_class") is not False
            or audit.get("trust_scope") != "trusted-current"
        ):
            raise AssertionError(
                "trusted-current canonicalizer contract changed"
            )
        return (
            np.ascontiguousarray(canonical, dtype=np.complex64),
            CURRENT_EFFECTIVE_INPUT_LENGTH,
            dict(audit),
        )
    if sample_rate_hz is not None or native_sample_rate_hz is not None:
        raise ValueError(
            "historical scoring is unchanged and forbids current-rate metadata"
        )
    effective = _historical_effective_length(len(raw))
    return (
        np.ascontiguousarray(raw[:effective], dtype=np.complex64),
        effective,
        None,
    )


def prepare_inference_observation(
    observation: InferenceObservation,
    *,
    patch_length: int,
    patch_count: int,
    target_frac: float,
) -> PreparedObservation:
    """Canonicalize first, then derive every open-set/classifier feature."""
    effective_iq, effective_length, canonicalization = (
        canonicalize_inference_observation(
            observation.iq,
            prototype_source=observation.prototype_source,
            sample_rate_hz=observation.sample_rate_hz,
            native_sample_rate_hz=observation.native_sample_rate_hz,
        )
    )
    is_zero = v4_openset._is_exact_no_signal(effective_iq)
    if is_zero:
        return PreparedObservation(
            effective_iq=effective_iq,
            prototype_source=validate_prototype_source(
                observation.prototype_source
            ),
            observation_length=len(np.asarray(observation.iq)),
            effective_input_length=effective_length,
            canonicalization=canonicalization,
            is_exact_no_signal=True,
            pose_features=None,
            phase_linearity_score=None,
            packed=None,
            raw_frontend_features=None,
            frontend_context=None,
        )

    # Ordering below is part of the contract: ``effective_iq`` is already the
    # canonicalized trusted-current signal (or unchanged historical prefix).
    pose = v4_openset._pose_feature_row(
        effective_iq,
        patch_length=int(patch_length),
        target_frac=float(target_frac),
    )
    phase = v4_openset.instantaneous_phase_linearity_chirp_score(
        effective_iq
    )
    packed, raw_features, frontend_context = td_preprocess.preprocess(
        effective_iq,
        patch_length=int(patch_length),
        patch_count=int(patch_count),
        target_frac=float(target_frac),
    )
    if frontend_context.get("uses_frequency_transform") is not False:
        raise AssertionError("v5 open-set path admitted an FFT frontend")
    return PreparedObservation(
        effective_iq=effective_iq,
        prototype_source=validate_prototype_source(
            observation.prototype_source
        ),
        observation_length=len(np.asarray(observation.iq)),
        effective_input_length=effective_length,
        canonicalization=canonicalization,
        is_exact_no_signal=False,
        pose_features=np.asarray(pose, dtype=np.float64),
        phase_linearity_score=float(phase),
        packed=np.asarray(packed, dtype=np.float32),
        raw_frontend_features=np.asarray(raw_features, dtype=np.float32),
        frontend_context=dict(frontend_context),
    )


def _validate_profile_bank_metadata(
    value: Mapping[str, Any],
    labels: np.ndarray,
) -> list[dict[str, Any]]:
    if tuple(value.get("classes", ())) != PUBLIC_CLASSES:
        raise ValueError("v5 prototype metadata class order changed")
    if value.get("decision_rule") != V5_ASSEMBLY_DECISION_RULE:
        raise ValueError("v5 prototype decision rule changed")
    if value.get("enrollment_view_rule") != V5_ASSEMBLY_ENROLLMENT_RULE:
        raise ValueError("v5 prototype enrollment-view rule changed")
    if value.get("prototype_public_class_indices") != labels.tolist():
        raise ValueError("v5 prototype labels disagree with metadata")
    groups = value.get("groups")
    if not isinstance(groups, list) or len(groups) != len(labels):
        raise ValueError("v5 prototype groups do not align with labels")
    seen: set[tuple[str, str]] = set()
    seen_current: set[str] = set()
    output: list[dict[str, Any]] = []
    for index, item in enumerate(groups):
        if not isinstance(item, Mapping):
            raise ValueError(f"prototype group {index} is not an object")
        source = item.get("source")
        profile = item.get("profile")
        if (
            source not in PROTOTYPE_SOURCE_ROUTES
            or not isinstance(profile, str)
            or not profile
        ):
            raise ValueError(f"prototype group {index} route/profile invalid")
        key = (str(source), profile)
        if key in seen:
            raise ValueError(f"prototype group repeats {source}:{profile}")
        seen.add(key)
        label = int(labels[index])
        public_class = item.get("public_class")
        if public_class != PUBLIC_CLASSES[label]:
            raise ValueError(
                f"prototype group {index} public class disagrees with label"
            )
        lengths = item.get("eligible_lengths")
        expected_lengths = (
            [CURRENT_EFFECTIVE_INPUT_LENGTH]
            if source == "current"
            else sorted(
                set(
                    int(length)
                    for length in lengths
                )
            )
            if isinstance(lengths, list)
            else None
        )
        if (
            not isinstance(lengths, list)
            or not lengths
            or lengths != sorted(set(lengths))
            or any(length not in RUNTIME_INPUT_LENGTHS for length in lengths)
            or lengths != expected_lengths
        ):
            raise ValueError(
                f"prototype group {index} eligible lengths are invalid"
            )
        if source == "current":
            expected_class = (
                corpus_data.CURRENT_PROFILE_PUBLIC_CLASS_MAP.get(profile)
            )
            if expected_class != public_class:
                raise ValueError(
                    f"current prototype {profile} must map to {expected_class}"
                )
            seen_current.add(profile)
        enrollment_views = item.get("enrollment_views")
        if (
            isinstance(enrollment_views, bool)
            or not isinstance(enrollment_views, int)
            or enrollment_views <= 0
        ):
            raise ValueError(
                f"prototype group {index} enrollment view count is invalid"
            )
        output.append(
            {
                "source": str(source),
                "profile": profile,
                "public_class": str(public_class),
                "enrollment_views": int(enrollment_views),
                "eligible_lengths": list(lengths),
            }
        )
    expected_current = set(corpus_data.CURRENT_PROFILE_PUBLIC_CLASS_MAP)
    if seen_current != expected_current:
        raise ValueError(
            "v5 current prototype bank lacks the exact 31-profile inventory"
        )
    return output


def load_verified_v5_fusion(directory: str | Path) -> dict[str, Any]:
    """Load the v5 fusion artifacts without accepting the v4 schema."""
    source = reject_sensitive_path(directory, "fusion input")
    if not source.is_dir():
        raise FileNotFoundError(source)
    metrics_path = source / "dev_metrics.json"
    metrics = browser_export._load_json(metrics_path)
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
                f"v5 fusion metric {key} must be {expected!r}"
            )
    run = metrics.get("run_configuration")
    if not isinstance(run, Mapping) or run.get("schema") != FUSION_SCHEMA:
        raise ValueError("input is not a v5 scale-orbit fusion")
    audit = metrics.get("data_audit")
    if not isinstance(audit, Mapping):
        raise ValueError("v5 fusion has no data audit")
    current_audit = audit.get("current")
    if not isinstance(current_audit, Mapping):
        raise ValueError("v5 fusion has no composite-current audit")
    frontend = metrics.get("frontend")
    if (
        not isinstance(frontend, Mapping)
        or frontend.get("uses_frequency_transform") is not False
    ):
        raise ValueError("v5 fusion frontend is not FFT-free")
    fitting = metrics.get("fitting_contract")
    if (
        not isinstance(fitting, Mapping)
        or fitting.get("prototype_grouping") != "(source, profile)"
        or fitting.get("consumed_historical_test_rows_loaded") != 0
        or fitting.get("sealed_release_rows_loaded") != 0
        or "seed-20262904 adaptive selection metrics only"
        not in str(fitting.get("selection_population_role"))
    ):
        raise ValueError("v5 fusion fitting firewall changed")

    artifacts = metrics.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("v5 fusion artifact contract is missing")
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for key in browser_export.REQUIRED_ARTIFACTS:
        path, digest = browser_export._plain_artifact_path(
            source, artifacts, key
        )
        paths[key] = path
        hashes[path.name] = digest
    real_config, complex_config, embed_dim, n_features, packed_length = (
        browser_export._validate_architecture(metrics)
    )
    state = browser_export._load_state(paths["state_dict"])
    prototype_bank = browser_export._load_float32_array(
        paths["prototype_bank"],
        "v5 fusion prototype bank",
    )
    if (
        prototype_bank.ndim != 2
        or prototype_bank.shape[0] < len(PUBLIC_CLASSES)
        or prototype_bank.shape[1] != 2 * embed_dim
    ):
        raise ValueError("v5 fusion prototype bank shape is invalid")
    try:
        labels = np.load(paths["prototype_labels"], allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError("cannot load v5 prototype labels") from exc
    if (
        not isinstance(labels, np.ndarray)
        or labels.dtype != np.int64
        or labels.shape != (len(prototype_bank),)
        or np.any(labels < 0)
        or np.any(labels >= len(PUBLIC_CLASSES))
        or set(labels.tolist()) != set(range(len(PUBLIC_CLASSES)))
    ):
        raise ValueError("v5 fusion prototype labels are invalid")
    groups = _validate_profile_bank_metadata(
        browser_export._load_json(paths["prototype_metadata"]),
        labels,
    )
    centers = {
        "real": browser_export._load_float32_array(
            paths["real_center"], "v5 real center", (embed_dim,)
        ),
        "complex": browser_export._load_float32_array(
            paths["complex_center"], "v5 complex center", (embed_dim,)
        ),
    }
    feature_mean = browser_export._load_float32_array(
        paths["feature_mean"], "v5 feature mean", (n_features,)
    )
    feature_std = browser_export._load_float32_array(
        paths["feature_std"], "v5 feature std", (n_features,)
    )
    if np.any(feature_std <= 0):
        raise ValueError("v5 feature standard deviations must be positive")
    for route, state_name in (
        ("real", "real_center"),
        ("complex", "complex_center"),
    ):
        state_value = state.get(state_name)
        if (
            not isinstance(state_value, torch.Tensor)
            or state_value.dtype != torch.float32
            or not np.array_equal(state_value.numpy(), centers[route])
        ):
            raise ValueError(
                f"v5 fusion state {state_name} disagrees with NPY"
            )
    recorded_sources = metrics.get("source_sha256")
    expected_sources = fusion_assembler._source_hashes()
    if recorded_sources != expected_sources:
        raise RuntimeError(
            "v5 fusion source hashes do not reproduce from this source tree"
        )
    return {
        "directory": source,
        "metrics_path": metrics_path,
        "metrics": dict(metrics),
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


def load_development_context(
    fusion_dir: str | Path,
    historical_dir: str | Path,
    current_dir: str | Path,
    current_selection_dir: str | Path,
    *,
    device_name: str,
) -> DevelopmentContext:
    """Rebuild the exact two-corpus v5 context and verified fusion."""
    _validate_contract_files()
    fusion_path = reject_sensitive_path(fusion_dir, "fusion input")
    historical_path = reject_sensitive_path(
        historical_dir, "historical corpus input"
    )
    current_path = reject_sensitive_path(
        current_dir, "training/enrollment corpus input"
    )
    selection_path = reject_sensitive_path(
        current_selection_dir, "adaptive-selection corpus input"
    )
    if len(
        {fusion_path, historical_path, current_path, selection_path}
    ) != 4:
        raise ValueError(
            "fusion, historical, training/enrollment, and adaptive-selection "
            "directories must all differ"
        )
    fusion = load_verified_v5_fusion(fusion_path)
    metrics = fusion["metrics"]
    run = metrics["run_configuration"]
    if (
        reject_sensitive_path(
            run.get("training_enrollment_corpus", ""),
            "recorded training/enrollment corpus",
        )
        != current_path
        or reject_sensitive_path(
            run.get("adaptive_selection_corpus", ""),
            "recorded adaptive-selection corpus",
        )
        != selection_path
    ):
        raise ValueError("v5 fusion is bound to different current corpora")
    seed = int(run["seed_inherited_from_matched_branches"])
    target_frac = float(metrics["frontend"]["target_frac"])
    patch_length = int(fusion["real_config"]["patch_length"])
    patch_count = int(fusion["real_config"]["patch_count"])
    historical, historical_contract = corpus_data.load_historical_exposed(
        historical_path,
        patch_length=patch_length,
        patch_count=patch_count,
        target_frac=target_frac,
        seed=seed,
    )
    classes = list(historical_contract["classes"])
    if tuple(classes) != PUBLIC_CLASSES:
        raise ValueError("historical public-class order changed")
    current = scale_orbit_data.load_scale_orbit_training_corpus(
        current_path,
        selection_directory=selection_path,
        class_index={
            name: index for index, name in enumerate(PUBLIC_CLASSES)
        },
    )
    fusion_assembler.validate_composite_current_audit(
        current.audit,
        current_directory=current_path,
        selection_directory=selection_path,
    )
    data, preprocessing_audit = branch_runner._prepare_data(
        historical,
        current,
        classes=classes,
        patch_length=patch_length,
        patch_count=patch_count,
        target_frac=target_frac,
    )
    rebuilt_audit = {
        "historical": historical.audit,
        "current": current.audit,
        "preprocessing": preprocessing_audit,
        "historical_contract": historical_contract,
        "consumed_test_rows_exposed": 0,
    }
    if rebuilt_audit != metrics.get("data_audit"):
        raise RuntimeError(
            "v5 fusion audit does not reproduce from the supplied corpora"
        )
    if historical.audit.get("consumed_test_rows_exposed") != 0:
        raise AssertionError("historical loader exposed consumed test rows")
    device = branch_runner.v3_runner.resolve_device(str(device_name))
    branch_runner.v3_runner.seed_everything(seed)
    module = browser_export._load_fusion_module(fusion).to(device).eval()
    return DevelopmentContext(
        fusion=fusion,
        historical=historical,
        current=current,
        data=data,
        preprocessing_audit=preprocessing_audit,
        module=module,
        device=device,
        patch_length=patch_length,
        patch_count=patch_count,
        target_frac=target_frac,
    )


def _embed(
    context: DevelopmentContext,
    packed: np.ndarray,
    features: np.ndarray,
) -> np.ndarray:
    return branch_runner.embed_all(
        context.module,
        packed,
        features,
        context.device,
    ).astype(np.float32, copy=False)


def _public_distances(
    context: DevelopmentContext,
    embeddings: np.ndarray,
    *,
    prototype_source: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # This v4 helper is route/payload agnostic and consumes only the verified
    # fusion bank plus embeddings produced after the v5 frontend.
    return v4_openset._public_distances(
        context,
        embeddings,
        prototype_source=prototype_source,
    )


def _validate_stage_one_template(
    template: v4_openset.StageOneTemplate,
) -> None:
    if tuple(sorted(template.models)) != RUNTIME_INPUT_LENGTHS:
        raise ValueError("fixed stage-one template lacks a runtime length")
    if not _is_sha256(template.set_sha256):
        raise ValueError("fixed stage-one template set digest is invalid")
    for length in RUNTIME_INPUT_LENGTHS:
        model = template.models[length]
        metadata = template.metadata.get(length)
        if (
            model.capture_length != length
            or tuple(model.feature_names)
            != tuple(noise_prefilter.PREFILTER_FEATURES)
            or template.bundle_sha256.get(length) is None
            or not _is_sha256(template.bundle_sha256[length])
            or not isinstance(metadata, Mapping)
            or not _is_sha256(metadata.get("source_policy_sha256"))
            or metadata.get("fixed_before_v5_adaptive_selection") is not True
        ):
            raise ValueError(f"fixed stage-one N{length} contract changed")
        provenance = model.fit_provenance
        for key in (
            "selection_rows_used_in_fit",
            "novelty_design_or_validation_rows_used_in_fit",
            "consumed_test_rows_used_in_fit",
        ):
            if provenance.get(key) != 0:
                raise ValueError(
                    f"fixed stage-one N{length} provenance {key} is not zero"
                )
    if len(
        {
            template.metadata[length]["source_policy_sha256"]
            for length in RUNTIME_INPUT_LENGTHS
        }
    ) != 1:
        raise ValueError("fixed stage-one source policy changed by length")


def load_fixed_v4_stage_one_template(
    policy_path: str | Path,
) -> v4_openset.StageOneTemplate:
    """Load only the fixed stage-one coefficients from a verified v4 policy."""
    source = reject_sensitive_path(policy_path, "fixed v4 policy input")
    policy = browser_export._load_json(source)
    v4_openset.validate_policy(policy)
    fitting = policy.get("fitting_contract")
    required_zeros = {
        "selection_rows_used": 0,
        "current_selection_rows_used": 0,
        "novelty_rows_used": 0,
        "sealed_release_rows_used": 0,
        "consumed_historical_test_rows_used": 0,
    }
    if not isinstance(fitting, Mapping) or any(
        fitting.get(key) != expected
        for key, expected in required_zeros.items()
    ):
        raise ValueError("fixed v4 stage-one policy fitting firewall changed")
    source_digest = _sha256(source)
    models: dict[int, noise_prefilter.NoisePrefilter] = {}
    bundle_hashes: dict[int, str] = {}
    metadata: dict[int, Mapping[str, Any]] = {}
    recorded = policy["stage_one"][
        "coefficient_artifact_by_length_sha256"
    ]
    for length in RUNTIME_INPUT_LENGTHS:
        loaded = v4_openset._policy_stage_one_model(policy, length)
        model = noise_prefilter.NoisePrefilter(
            feature_names=tuple(loaded.feature_names),
            capture_length=loaded.capture_length,
            mean=np.asarray(loaded.mean, dtype=np.float64),
            scale=np.asarray(loaded.scale, dtype=np.float64),
            coefficients=np.asarray(
                loaded.coefficients, dtype=np.float64
            ),
            intercept=float(loaded.intercept),
            threshold_score=0.0,
            fit_provenance={
                "source": "fixed verified v4 external policy",
                "source_policy_sha256": source_digest,
                "selection_rows_used_in_fit": 0,
                "novelty_design_or_validation_rows_used_in_fit": 0,
                "consumed_test_rows_used_in_fit": 0,
            },
            threshold_provenance={
                "chosen": False,
                "v4_threshold_or_rank_reused": False,
            },
        )
        digest = v4_openset._stage_one_parameter_sha256(
            mean=model.mean,
            scale=model.scale,
            coefficients=model.coefficients,
            intercept=model.intercept,
        )
        if digest != recorded.get(str(length)):
            raise ValueError(
                f"fixed v4 stage-one N{length} parameter digest changed"
            )
        models[length] = model
        bundle_hashes[length] = digest
        metadata[length] = {
            "source_policy_sha256": source_digest,
            "fixed_before_v5_adaptive_selection": True,
        }
    set_digest = v4_openset._stage_one_parameter_set_sha256(
        bundle_hashes
    )
    if (
        set_digest
        != policy["stage_one"]["coefficient_artifact_set_sha256"]
    ):
        raise ValueError("fixed v4 stage-one parameter-set digest changed")
    template = v4_openset.StageOneTemplate(
        directory=source.parent,
        models=models,
        set_sha256=set_digest,
        bundle_sha256=bundle_hashes,
        metadata=metadata,
    )
    _validate_stage_one_template(template)
    return template


def _role_evaluation_observations(
    corpus: corpus_data.RawCorpus,
    role: str,
    *,
    expand_current_observation_lengths: bool,
) -> list[EvaluationObservation]:
    if role not in corpus_data.ROLES:
        raise ValueError(f"invalid corpus role {role!r}")
    output: list[EvaluationObservation] = []
    for row in corpus.rows_by_role[role]:
        if not isinstance(row, scale_orbit_data.ScaleOrbitRowRef):
            minimum = corpus.prefix(
                row, corpus_data.MINIMUM_INPUT_LENGTH
            )
            if float(np.max(np.abs(minimum))) == 0.0:
                # Preserve the frozen historical admission rule exactly.
                # Trusted-current exact-zero handling occurs only after its
                # mandatory physical-time canonicalization.
                continue
        if isinstance(row, scale_orbit_data.ScaleOrbitRowRef):
            if row.source != "current":
                raise ValueError("scale-orbit row lost current source identity")
        elif row.source == "current":
            # Refuse an untyped current row because it would lack trusted
            # sample-rate geometry and could bypass canonicalization.
            raise TypeError(
                "current open-set rows must be ScaleOrbitRowRef instances"
            )
        if isinstance(row, scale_orbit_data.ScaleOrbitRowRef):
            lengths = (
                tuple(
                    length
                    for length in RUNTIME_INPUT_LENGTHS
                    if length <= row.valid_sample_count
                )
                if expand_current_observation_lengths
                else (CURRENT_EFFECTIVE_INPUT_LENGTH,)
            )
            if CURRENT_EFFECTIVE_INPUT_LENGTH not in lengths:
                raise ValueError(
                    f"current {role} row {row.identity} lacks N4096"
                )
            rates = (row.sample_rate_hz, row.native_sample_rate_hz)
            pair_id = row.pair_id
            scale = row.physical_scale_factor
        else:
            lengths = corpus_data.training_view_lengths(
                row.valid_sample_count
            )
            rates = (None, None)
            pair_id = row.identity
            scale = 1.0
        for length in lengths:
            output.append(
                EvaluationObservation(
                    inference=InferenceObservation(
                        iq=np.ascontiguousarray(
                            corpus.prefix(row, int(length)),
                            dtype=np.complex64,
                        ),
                        prototype_source=row.source,
                        sample_rate_hz=rates[0],
                        native_sample_rate_hz=rates[1],
                    ),
                    truth_index=int(row.label),
                    source_profile=f"{row.source}:{row.profile_id}",
                    base_identity=str(row.identity),
                    pair_id=str(pair_id),
                    observation_length=int(length),
                    physical_scale_factor=float(scale),
                )
            )
    if not output:
        raise ValueError(f"{corpus.source} {role} has no admitted observations")
    return output


def _prepared_components(
    context: DevelopmentContext,
    template: v4_openset.StageOneTemplate,
    observations: Sequence[EvaluationObservation],
) -> dict[str, np.ndarray]:
    values = list(observations)
    if not values:
        raise ValueError("component extraction needs enrollment observations")
    prepared = [
        prepare_inference_observation(
            row.inference,
            patch_length=context.patch_length,
            patch_count=context.patch_count,
            target_frac=context.target_frac,
        )
        for row in values
    ]
    if any(row.is_exact_no_signal for row in prepared):
        raise ValueError("enrollment contains an exact no-signal observation")
    sources = np.asarray(
        [row.prototype_source for row in prepared], dtype=object
    )
    lengths = np.asarray(
        [row.effective_input_length for row in prepared], dtype=np.int64
    )
    pose = np.stack(
        [np.asarray(row.pose_features, dtype=np.float64) for row in prepared]
    )
    phase = np.asarray(
        [float(row.phase_linearity_score) for row in prepared],
        dtype=np.float64,
    )
    packed = np.stack(
        [np.asarray(row.packed, dtype=np.float32) for row in prepared]
    )
    raw_features = np.stack(
        [
            np.asarray(row.raw_frontend_features, dtype=np.float32)
            for row in prepared
        ]
    )
    standardized = (
        (
            raw_features
            - np.asarray(context.fusion["feature_mean"], dtype=np.float32)
        )
        / np.asarray(context.fusion["feature_std"], dtype=np.float32)
    ).astype(np.float32)
    embeddings = _embed(context, packed, standardized)
    stage_one = np.empty(len(values), dtype=np.float64)
    distances = np.full(
        (len(values), len(PUBLIC_CLASSES)),
        np.inf,
        dtype=np.float64,
    )
    closest = np.full_like(distances, -1, dtype=np.int64)
    for length in RUNTIME_INPUT_LENGTHS:
        length_mask = lengths == length
        if not np.any(length_mask):
            continue
        model = template.models[length]
        stage_one[length_mask] = np.asarray(
            model.score(pose[length_mask], capture_length=length),
            dtype=np.float64,
        )
        for source in PROTOTYPE_SOURCE_ROUTES:
            mask = length_mask & (sources == source)
            if not np.any(mask):
                continue
            current, current_closest, _support = _public_distances(
                context,
                embeddings[mask],
                prototype_source=source,
            )
            distances[mask] = current
            closest[mask] = current_closest
    if (
        not np.isfinite(stage_one).all()
        or not np.isfinite(phase).all()
    ):
        raise RuntimeError("enrollment analytic features are non-finite")
    return {
        "stage_one": stage_one,
        "phase_linearity": phase,
        "embeddings": embeddings,
        "class_distances": distances,
        "closest_prototype_indices": closest,
        "sources": sources,
        "lengths": lengths,
    }


def _build_enrollment_calibration(
    context: DevelopmentContext,
    stage_one_template: v4_openset.StageOneTemplate,
) -> _EnrollmentCalibration:
    """Build calibration arrays from enrollment and no other corpus role."""
    _validate_stage_one_template(stage_one_template)
    observations = (
        _role_evaluation_observations(
            context.historical,
            "enrollment",
            expand_current_observation_lengths=False,
        )
        + _role_evaluation_observations(
            context.current,
            "enrollment",
            expand_current_observation_lengths=False,
        )
    )
    components = _prepared_components(
        context, stage_one_template, observations
    )
    embeddings = components["embeddings"]
    sources = components["sources"]
    lengths = components["lengths"]
    groups = np.asarray(
        [row.source_profile for row in observations], dtype=object
    )
    identities = np.asarray(
        [row.base_identity for row in observations], dtype=object
    )
    loo = np.full_like(components["class_distances"], np.inf)
    for source in PROTOTYPE_SOURCE_ROUTES:
        route_mask = sources == source
        route_positions = np.flatnonzero(route_mask)
        if not len(route_positions):
            raise ValueError(f"enrollment lacks route {source}")
        bank, labels, metadata, _positions, support = (
            v4_openset._prototype_route_bank(context.fusion, source)
        )
        loo[route_mask] = (
            v4_openset.leave_one_base_identity_out_class_distances(
                embeddings[route_mask],
                [
                    tuple(str(groups[int(index)]).split(":", 1))
                    for index in route_positions
                ],
                [
                    str(identities[int(index)])
                    for index in route_positions
                ],
                bank,
                labels,
                [
                    (str(row["source"]), str(row["profile"]))
                    for row in metadata
                ],
                supported_public_class_mask=support,
            )
        )
    stage_one_by_length: dict[int, np.ndarray] = {}
    phase_by_length: dict[int, np.ndarray] = {}
    distance_by_length: dict[int, np.ndarray] = {}
    loo_by_length: dict[int, np.ndarray] = {}
    group_by_length: dict[int, Sequence[str]] = {}
    source_by_length: dict[int, Sequence[str]] = {}
    identity_by_length: dict[int, Sequence[str]] = {}
    for length in RUNTIME_INPUT_LENGTHS:
        mask = lengths == length
        if not np.any(mask):
            raise ValueError(f"enrollment lacks N{length}")
        observed_sources = set(str(value) for value in sources[mask])
        required_sources = {
            source
            for source, legal in
            RUNTIME_LENGTHS_BY_PROTOTYPE_SOURCE.items()
            if length in legal
        }
        if observed_sources != required_sources:
            raise ValueError(
                f"N{length} enrollment route inventory changed: "
                f"{sorted(observed_sources)} != {sorted(required_sources)}"
            )
        stage_one_by_length[length] = components["stage_one"][mask]
        phase_by_length[length] = components["phase_linearity"][mask]
        distance_by_length[length] = components["class_distances"][mask]
        loo_by_length[length] = loo[mask]
        group_by_length[length] = groups[mask].astype(str).tolist()
        source_by_length[length] = sources[mask].astype(str).tolist()
        identity_by_length[length] = identities[mask].astype(str).tolist()
    audit = {
        "schema": "v5-scale-orbit-enrollment-calibration-input-v1",
        "role_read": "enrollment",
        "roles_not_read": ["train", "selection"],
        "rows_by_effective_length": {
            str(length): int(len(stage_one_by_length[length]))
            for length in RUNTIME_INPUT_LENGTHS
        },
        "routes_by_effective_length": {
            str(length): sorted(set(source_by_length[length]))
            for length in RUNTIME_INPUT_LENGTHS
        },
        "trusted_current_canonicalized_before_all_features": True,
        "uses_frequency_transform": False,
        **FITTING_FIREWALL,
    }
    return _EnrollmentCalibration(
        stage_one_score_by_length=stage_one_by_length,
        phase_linearity_score_by_length=phase_by_length,
        class_distance_by_length=distance_by_length,
        calibration_class_distance_by_length=loo_by_length,
        enrollment_group_by_length=group_by_length,
        enrollment_source_by_length=source_by_length,
        enrollment_identity_by_length=identity_by_length,
        audit=audit,
    )


def _gate_contract() -> dict[str, Any]:
    inherited = [
        {
            "name": name,
            "comparison": comparison,
            "threshold": threshold,
        }
        for name, (comparison, threshold)
        in recovery_validator.EXPECTED_INHERITED_GATES.items()
    ]
    directional = [
        {"name": name, **dict(row)}
        for name, row in recovery_validator.EXPECTED_DIRECTIONAL_GATES.items()
    ]
    return {
        "inherited_gates": inherited,
        "added_directional_confusion_gates": directional,
        "inherited_gate_count": len(inherited),
        "conditional_cell_minimum_closed_head_correct_rows":
            CONDITIONAL_CELL_MINIMUM_CLOSED_CORRECT_ROWS,
        "required_current_source_profile_count":
            len(REQUIRED_CURRENT_PROFILE_PUBLIC_CLASS_MAP),
        "required_current_observation_lengths":
            list(REQUIRED_CURRENT_EVALUATION_LENGTHS),
        "required_current_physical_scales":
            list(REQUIRED_CURRENT_EVALUATION_SCALES),
        "required_views_per_current_pair": (
            len(REQUIRED_CURRENT_EVALUATION_LENGTHS)
            * len(REQUIRED_CURRENT_EVALUATION_SCALES)
        ),
        "minimum_independent_pairs_per_current_source_profile":
            REQUIRED_CURRENT_EVALUATION_MINIMUM_PAIRS_PER_PROFILE,
        "weakening_inherited_gates_forbidden": True,
        "all_inherited_and_added_gates_required": True,
    }


def _fit_enrollment_policy_from_verified_enrollment(
    calibration: _EnrollmentCalibration,
    *,
    stage_one_template: v4_openset.StageOneTemplate,
    fusion_binding: Mapping[str, Any],
    patch_length: int,
    patch_count: int,
    target_frac: float,
    current_novelty_geometry: Mapping[str, Any],
    known_false_unknown_budget: float =
        v4_openset.DEFAULT_COMPOSITE_KNOWN_BUDGET,
) -> dict[str, Any]:
    """Fit v5 ranks/thresholds from internally derived enrollment arrays.

    This is deliberately private.  The public fitting entry point accepts a
    verified development context and derives this object itself, so a caller
    cannot label arbitrary selection or held arrays as enrollment.
    """
    _validate_contract_files()
    _validate_stage_one_template(stage_one_template)
    _validate_fusion_binding(fusion_binding)
    novelty_geometry = validate_current_novelty_geometry(
        current_novelty_geometry
    )
    if calibration.audit.get("role_read") != "enrollment":
        raise ValueError("policy fit input is not enrollment-only")
    for key, expected in FITTING_FIREWALL.items():
        if calibration.audit.get(key) != expected:
            raise ValueError(f"enrollment calibration firewall {key} changed")
    per_source: dict[str, dict[str, Any]] = {
        source: {
            "supported_public_class_mask": None,
            "per_length": {},
        }
        for source in PROTOTYPE_SOURCE_ROUTES
    }
    total_rows = 0
    for length in RUNTIME_INPUT_LENGTHS:
        stage_one = np.asarray(
            calibration.stage_one_score_by_length[length],
            dtype=np.float64,
        ).reshape(-1)
        phase = np.asarray(
            calibration.phase_linearity_score_by_length[length],
            dtype=np.float64,
        ).reshape(-1)
        distance = np.asarray(
            calibration.class_distance_by_length[length],
            dtype=np.float64,
        )
        loo = np.asarray(
            calibration.calibration_class_distance_by_length[length],
            dtype=np.float64,
        )
        groups = np.asarray(
            calibration.enrollment_group_by_length[length],
            dtype=object,
        )
        sources = np.asarray(
            [
                validate_prototype_source(value)
                for value in
                calibration.enrollment_source_by_length[length]
            ],
            dtype=object,
        )
        if (
            distance.shape != (len(stage_one), len(PUBLIC_CLASSES))
            or loo.shape != distance.shape
            or phase.shape != stage_one.shape
            or groups.shape != stage_one.shape
            or sources.shape != stage_one.shape
            or not len(stage_one)
        ):
            raise ValueError(f"N{length} enrollment arrays lost alignment")
        required_sources = {
            source
            for source, lengths in
            RUNTIME_LENGTHS_BY_PROTOTYPE_SOURCE.items()
            if length in lengths
        }
        if set(sources.tolist()) != required_sources:
            raise ValueError(f"N{length} enrollment routes changed")
        for source in sorted(required_sources):
            route_mask = sources == source
            row = v4_openset._fit_route_length_policy(
                stage_one=stage_one[route_mask],
                phase_linearity=phase[route_mask],
                distances=distance[route_mask],
                calibration_distances=loo[route_mask],
                enrollment_groups=groups[route_mask],
                prototype_source=source,
                composite_budget=float(known_false_unknown_budget),
            )
            support = row.pop("supported_public_class_mask")
            source_policy = per_source[source]
            if source_policy["supported_public_class_mask"] is None:
                source_policy["supported_public_class_mask"] = support
            elif source_policy["supported_public_class_mask"] != support:
                raise RuntimeError(
                    f"{source} support changed across runtime lengths"
                )
            source_policy["per_length"][str(length)] = row
            total_rows += int(row["enrollment_rows"])
    for source, lengths in RUNTIME_LENGTHS_BY_PROTOTYPE_SOURCE.items():
        row = per_source[source]
        if (
            row["supported_public_class_mask"]
            != list(
                EXPECTED_PUBLIC_CLASS_SUPPORT_BY_PROTOTYPE_SOURCE[source]
            )
            or set(row["per_length"])
            != {str(length) for length in lengths}
        ):
            raise RuntimeError(f"{source} enrollment policy is incomplete")

    parameter_hashes = {
        str(length): stage_one_template.bundle_sha256[length]
        for length in RUNTIME_INPUT_LENGTHS
    }
    source_hashes = _source_hashes()
    policy = {
        "schema": POLICY_SCHEMA,
        "schema_version": POLICY_SCHEMA_VERSION,
        "status": "development_external_policy",
        "development_only": True,
        "release_evidence": False,
        "public_classes": list(PUBLIC_CLASSES),
        "prototype_source_routes": list(PROTOTYPE_SOURCE_ROUTES),
        "runtime_input_lengths": list(RUNTIME_INPUT_LENGTHS),
        "runtime_lengths_by_prototype_source": {
            source: list(lengths)
            for source, lengths
            in RUNTIME_LENGTHS_BY_PROTOTYPE_SOURCE.items()
        },
        "frontend": {
            "version": td_preprocess.PREPROCESS_VERSION,
            "patch_length": int(patch_length),
            "patch_count": int(patch_count),
            "target_frac": float(target_frac),
            "uses_frequency_transform": False,
        },
        "trusted_current_geometry_canonicalizer":
            geometry_canonicalizer.canonicalizer_metadata(),
        "inference_contract": {
            "current_rates_required": [
                "sample_rate_hz",
                "native_sample_rate_hz",
            ],
            "canonicalization_precedes_pose_noise_phase_and_frontend": True,
            "selected_profile_is_input": False,
            "selected_class_is_input": False,
            "historical_geometry_unchanged": True,
            "current_novelty_geometry": novelty_geometry,
        },
        "stage_zero": {
            "kind": v4_openset.STAGE_ZERO_KIND,
            "decision": "exact zero after route-effective prefixing",
            "runs_before_pose_phase_and_frontend_features": True,
        },
        "stage_one": {
            "kind": v4_openset.STAGE_ONE_KIND,
            "fixed_before_v5_enrollment_calibration": True,
            "source_v4_policy_sha256": (
                stage_one_template.metadata[
                    RUNTIME_INPUT_LENGTHS[0]
                ]["source_policy_sha256"]
            ),
            "coefficient_artifact_set_sha256":
                stage_one_template.set_sha256,
            "coefficient_artifact_by_length_sha256": parameter_hashes,
            "feature_names": list(noise_prefilter.PREFILTER_FEATURES),
            "uses_frequency_transform": False,
            "gates_before_classification": False,
            "parameters_by_length": {
                str(length): {
                    "mean":
                        stage_one_template.models[length].mean.tolist(),
                    "scale":
                        stage_one_template.models[length].scale.tolist(),
                    "coefficients":
                        stage_one_template.models[length].coefficients.tolist(),
                    "intercept":
                        float(stage_one_template.models[length].intercept),
                }
                for length in RUNTIME_INPUT_LENGTHS
            },
        },
        "stage_two": {
            "kind": v4_openset.STAGE_TWO_KIND,
            "rank_fit_population": "matching-route v5 enrollment only",
            "calibration_centroid_rule":
                "leave one complete base identity out",
            "uses_frequency_transform": False,
        },
        "stage_chirp": {
            "kind": v4_openset.STAGE_CHIRP_KIND,
            "rank_fit_population": "matching-route v5 enrollment only",
            "uses_frequency_transform": False,
        },
        "composition": {
            "kind": v4_openset.COMPOSITION_KIND,
            "score": (
                "max(effective_stage_one_rank, stage_two_rank, "
                "phase_linearity_rank)"
            ),
            "threshold_fit_population":
                "matching-route v5 enrollment only",
        },
        "per_prototype_source": per_source,
        "classifier_binding": dict(fusion_binding),
        "fitting_contract": {
            "policy_fit_function_reads": ["enrollment"],
            "policy_fit_function_cannot_address": [
                "train",
                "adaptive_selection",
                "novelty",
                "sealed_validation",
                "consumed_historical_test",
            ],
            "bound_classifier_checkpoint_selection_population": (
                "seed-20262904 adaptive development; classifier checkpoint "
                "selection occurred upstream and is not threshold/rank fit"
            ),
            "enrollment_rows": total_rows,
            **FITTING_FIREWALL,
        },
        "gate_contract": _gate_contract(),
        "provenance": {
            "source_sha256": source_hashes,
            "amendment_sha256": AMENDMENT_SHA256,
            "recovery_protocol_sha256": PROTOCOL_SHA256,
            "seed_registry_sha256": SEED_REGISTRY_SHA256,
            "identity_firewall_amendment_sha256":
                IDENTITY_FIREWALL_AMENDMENT_SHA256,
            "identity_rejection_sha256": IDENTITY_REJECTION_SHA256,
            "identity_acceptance_sha256": IDENTITY_ACCEPTANCE_SHA256,
            "enrollment_calibration_audit":
                dict(calibration.audit),
        },
    }
    validate_policy(policy)
    _assert_source_snapshot_unchanged(
        source_hashes,
        operation="v5 enrollment-only open-set calibration",
    )
    return policy


def fit_enrollment_policy(
    context: DevelopmentContext,
    *,
    fixed_stage_one_policy_path: str | Path,
    current_novelty_geometry: Mapping[str, Any],
    known_false_unknown_budget: float =
        v4_openset.DEFAULT_COMPOSITE_KNOWN_BUDGET,
) -> dict[str, Any]:
    """Derive and fit every open-set rank/threshold from verified enrollment.

    There is intentionally no public stage-one object, calibration-array,
    fusion-binding, frontend-geometry, selection, novelty, or held-population
    argument.  The fixed stage-one parameters are loaded from a verified v4
    policy path.  Enrollment observations and the classifier binding are
    derived from the same verified :class:`DevelopmentContext` in this call.
    """
    _validate_contract_files()
    source_hashes_at_start = _source_hashes()
    stage_one_template = load_fixed_v4_stage_one_template(
        fixed_stage_one_policy_path
    )
    _validate_stage_one_template(stage_one_template)
    frontend = _frontend_contract_from_context(context)
    calibration = _build_enrollment_calibration(
        context,
        stage_one_template,
    )
    fusion_binding = build_fusion_binding(context.fusion)
    policy = _fit_enrollment_policy_from_verified_enrollment(
        calibration,
        stage_one_template=stage_one_template,
        fusion_binding=fusion_binding,
        patch_length=int(frontend["patch_length"]),
        patch_count=int(frontend["patch_count"]),
        target_frac=float(frontend["target_frac"]),
        current_novelty_geometry=current_novelty_geometry,
        known_false_unknown_budget=known_false_unknown_budget,
    )
    if policy["provenance"]["source_sha256"] != source_hashes_at_start:
        raise RuntimeError(
            "executed source changed before enrollment calibration completed"
        )
    _assert_source_snapshot_unchanged(
        source_hashes_at_start,
        operation="v5 enrollment derivation and open-set calibration",
    )
    return policy


def build_fusion_binding(fusion: Mapping[str, Any]) -> dict[str, Any]:
    metrics = fusion.get("metrics")
    if (
        not isinstance(metrics, Mapping)
        or metrics.get("run_configuration", {}).get("schema")
        != FUSION_SCHEMA
    ):
        raise ValueError("classifier binding requires a verified v5 fusion")
    metrics_path = Path(fusion["metrics_path"])
    hashes = fusion.get("hashes")
    if not isinstance(hashes, Mapping) or not all(
        _is_sha256(value) for value in hashes.values()
    ):
        raise ValueError("v5 fusion artifact hashes are invalid")
    binding = {
        "kind": "v5-scale-orbit-development-fusion",
        "source_fusion_schema": FUSION_SCHEMA,
        "source_fusion_dev_metrics_sha256": _sha256(metrics_path),
        "source_fusion_artifacts_sha256": dict(sorted(hashes.items())),
        "current_composite_audit_schema": COMPOSITE_AUDIT_SCHEMA,
        "trusted_current_geometry_canonicalizer":
            geometry_canonicalizer.canonicalizer_metadata(),
        "classifier_runtime_role_required": "accepted_known_classifier",
        "selected_profile_is_classifier_input": False,
        "selected_class_is_classifier_input": False,
    }
    _validate_fusion_binding(binding)
    return binding


def _validate_fusion_binding(binding: Mapping[str, Any]) -> None:
    required = {
        "kind": "v5-scale-orbit-development-fusion",
        "source_fusion_schema": FUSION_SCHEMA,
        "current_composite_audit_schema": COMPOSITE_AUDIT_SCHEMA,
        "trusted_current_geometry_canonicalizer":
            geometry_canonicalizer.canonicalizer_metadata(),
        "classifier_runtime_role_required": "accepted_known_classifier",
        "selected_profile_is_classifier_input": False,
        "selected_class_is_classifier_input": False,
    }
    for key, expected in required.items():
        if binding.get(key) != expected:
            raise ValueError(f"v5 classifier binding {key} changed")
    if not _is_sha256(
        binding.get("source_fusion_dev_metrics_sha256")
    ):
        raise ValueError("v5 classifier binding metrics digest is invalid")
    artifacts = binding.get("source_fusion_artifacts_sha256")
    if (
        not isinstance(artifacts, Mapping)
        or not artifacts
        or any(
            not isinstance(name, str)
            or Path(name).name != name
            or not _is_sha256(digest)
            for name, digest in artifacts.items()
        )
    ):
        raise ValueError("v5 classifier binding artifact hashes are invalid")


def _frontend_contract_from_context(
    context: DevelopmentContext,
) -> dict[str, Any]:
    if not isinstance(context, DevelopmentContext):
        raise TypeError(
            "v5 open-set fitting/scoring requires a DevelopmentContext "
            "returned by load_development_context"
        )
    metrics = context.fusion.get("metrics")
    recorded = (
        metrics.get("frontend")
        if isinstance(metrics, Mapping)
        else None
    )
    frozen = td_preprocess.preprocess_metadata()
    expected = {
        "version": td_preprocess.PREPROCESS_VERSION,
        "patch_length": int(context.patch_length),
        "patch_count": int(context.patch_count),
        "target_frac": float(context.target_frac),
        "uses_frequency_transform": False,
    }
    if (
        not isinstance(recorded, Mapping)
        or dict(recorded) != frozen
        or expected
        != {
            "version": frozen.get("version"),
            "patch_length": frozen.get("patch_length"),
            "patch_count": frozen.get("patch_count"),
            "target_frac": frozen.get("target_frac"),
            "uses_frequency_transform":
                frozen.get("uses_frequency_transform"),
        }
    ):
        raise ValueError(
            "v5 context frontend does not equal the frozen fusion frontend"
        )
    return expected


def _validate_policy_context_binding(
    context: DevelopmentContext,
    policy: Mapping[str, Any],
) -> None:
    """Require the scoring context to be the exact policy-bound classifier."""
    expected_frontend = _frontend_contract_from_context(context)
    expected_binding = build_fusion_binding(context.fusion)
    if policy.get("frontend") != expected_frontend:
        raise ValueError(
            "v5 open-set policy frontend does not match scoring context"
        )
    if (
        policy.get("trusted_current_geometry_canonicalizer")
        != geometry_canonicalizer.canonicalizer_metadata()
        or policy.get("classifier_binding") != expected_binding
    ):
        raise ValueError(
            "v5 open-set policy classifier/canonicalizer binding does not "
            "match scoring context"
        )


def _finite_sorted_vector(
    value: Any,
    *,
    name: str,
    allow_empty: bool,
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if (
        result.ndim != 1
        or (not allow_empty and len(result) == 0)
        or not np.isfinite(result).all()
        or (
            len(result) > 1
            and np.any(result[1:] < result[:-1])
        )
    ):
        raise ValueError(f"{name} must be a finite sorted vector")
    return result


def _validate_class_calibration_vectors(
    value: Any,
    *,
    name: str,
) -> list[np.ndarray]:
    if not isinstance(value, list) or len(value) != len(PUBLIC_CLASSES):
        raise ValueError(f"{name} must have one vector per public class")
    return [
        _finite_sorted_vector(
            row,
            name=f"{name}[{index}]",
            allow_empty=True,
        )
        for index, row in enumerate(value)
    ]


def _validate_route_length_policy(
    value: Any,
    *,
    source: str,
    enrollment_rows: int,
) -> None:
    if (
        not isinstance(value, Mapping)
        or value.get("prototype_source") != source
        or value.get("enrollment_rows") != enrollment_rows
        or isinstance(enrollment_rows, bool)
        or not isinstance(enrollment_rows, int)
        or enrollment_rows <= 0
    ):
        raise ValueError(f"{source} route-length policy identity changed")
    stage_one = value.get("stage_one")
    stage_two = value.get("stage_two")
    chirp = value.get("stage_chirp")
    composition = value.get("composition")
    attribution = value.get("enrollment_attribution")
    if not all(
        isinstance(row, Mapping)
        for row in (
            stage_one,
            stage_two,
            chirp,
            composition,
            attribution,
        )
    ):
        raise ValueError(f"{source} route-length policy sections are missing")
    if (
        stage_one.get("hard_gate_enabled") is not False
        or stage_one.get("hard_threshold_fitted") is not False
        or stage_one.get("pooled_rank_always_evaluated") is not True
        or stage_one.get("empirical_rank_rule")
        != v4_openset.EMPIRICAL_RANK_RULE
    ):
        raise ValueError(f"{source} stage-one rank contract changed")
    one_by_class = _validate_class_calibration_vectors(
        stage_one.get(
            "score_rank_calibration_by_predicted_class_sorted"
        ),
        name=f"{source} stage-one calibration",
    )
    one_pooled = _finite_sorted_vector(
        stage_one.get("pooled_score_rank_calibration_sorted"),
        name=f"{source} pooled stage-one calibration",
        allow_empty=False,
    )
    if (
        len(one_pooled) != enrollment_rows
        or sum(len(row) for row in one_by_class) != enrollment_rows
    ):
        raise ValueError(f"{source} stage-one calibration counts changed")
    if (
        stage_two.get("union_head_used") is not False
        or stage_two.get("empirical_rank_rule")
        != v4_openset.EMPIRICAL_RANK_RULE
    ):
        raise ValueError(f"{source} stage-two rank contract changed")
    two_by_class = _validate_class_calibration_vectors(
        stage_two.get(
            "predicted_class_distance_calibration_sorted"
        ),
        name=f"{source} stage-two calibration",
    )
    two_pooled = _finite_sorted_vector(
        stage_two.get("pooled_distance_calibration_sorted"),
        name=f"{source} pooled stage-two calibration",
        allow_empty=False,
    )
    counts = stage_two.get("predicted_class_calibration_counts")
    if (
        not isinstance(counts, list)
        or len(counts) != len(PUBLIC_CLASSES)
        or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or item < 0
            for item in counts
        )
        or
        counts != [len(row) for row in two_by_class]
        or len(two_pooled) != enrollment_rows
        or sum(counts) != enrollment_rows
    ):
        raise ValueError(f"{source} stage-two calibration counts changed")
    chirp_by_class = _validate_class_calibration_vectors(
        chirp.get("score_rank_calibration_by_predicted_class_sorted"),
        name=f"{source} phase calibration",
    )
    chirp_pooled = _finite_sorted_vector(
        chirp.get("pooled_score_rank_calibration_sorted"),
        name=f"{source} pooled phase calibration",
        allow_empty=False,
    )
    if (
        chirp.get("empirical_rank_rule")
        != v4_openset.EMPIRICAL_RANK_RULE
        or len(chirp_pooled) != enrollment_rows
        or sum(len(row) for row in chirp_by_class) != enrollment_rows
    ):
        raise ValueError(f"{source} phase calibration counts changed")
    try:
        thresholds = np.asarray(
            composition.get("threshold_by_predicted_public_class"),
            dtype=np.float64,
        )
        pooled_threshold = float(
            composition.get("pooled_fallback_threshold", np.nan)
        )
        budget = float(
            composition.get("known_false_positive_budget", np.nan)
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{source} composite threshold contract changed"
        ) from exc
    calibration_rows = composition.get("threshold_calibration_rows")
    if (
        thresholds.shape != (len(PUBLIC_CLASSES),)
        or not np.isfinite(thresholds).all()
        or np.any(thresholds < 0.0)
        or np.any(
            thresholds
            > 1.0 + v4_openset.RUNTIME_COMPARISON_MARGIN
        )
        or not np.isfinite(pooled_threshold)
        or pooled_threshold < 0.0
        or pooled_threshold
            > 1.0 + v4_openset.RUNTIME_COMPARISON_MARGIN
        or not np.isfinite(budget)
        or budget <= 0.0
        or budget >= 1.0
        or not isinstance(calibration_rows, Mapping)
    ):
        raise ValueError(f"{source} composite threshold contract changed")
    for key in (
        "stage_one_score",
        "instantaneous_phase_linearity_score",
        "leave_one_base_identity_out_winning_distance",
        "predicted_public_class_index",
        "source_profile_group",
    ):
        if (
            not isinstance(calibration_rows.get(key), list)
            or len(calibration_rows[key]) != enrollment_rows
        ):
            raise ValueError(
                f"{source} threshold calibration rows {key} changed"
            )
    stage_one_gated = attribution.get("stage_one_gated")
    stage_two_gated = attribution.get("stage_two_gated")
    accepted = attribution.get("accepted")
    if (
        stage_one_gated != 0
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in (stage_two_gated, accepted)
        )
        or stage_two_gated + accepted != enrollment_rows
    ):
        raise ValueError(f"{source} enrollment attribution changed")


def validate_policy(policy: Mapping[str, Any]) -> None:
    """Fail closed on v5 schema, frontend, fitting, gates, and provenance."""
    required = {
        "schema": POLICY_SCHEMA,
        "schema_version": POLICY_SCHEMA_VERSION,
        "status": "development_external_policy",
        "development_only": True,
        "release_evidence": False,
        "public_classes": list(PUBLIC_CLASSES),
        "prototype_source_routes": list(PROTOTYPE_SOURCE_ROUTES),
        "runtime_input_lengths": list(RUNTIME_INPUT_LENGTHS),
    }
    for key, expected in required.items():
        if policy.get(key) != expected:
            raise ValueError(f"v5 open-set policy {key} changed")
    if policy.get("runtime_lengths_by_prototype_source") != {
        source: list(lengths)
        for source, lengths in RUNTIME_LENGTHS_BY_PROTOTYPE_SOURCE.items()
    }:
        raise ValueError("v5 route-specific runtime lengths changed")
    frontend = policy.get("frontend")
    if (
        not isinstance(frontend, Mapping)
        or frontend.get("uses_frequency_transform") is not False
        or int(frontend.get("patch_length", 0)) <= 0
        or int(frontend.get("patch_count", 0)) <= 0
        or not (0.0 < float(frontend.get("target_frac", 0.0)) <= 1.0)
    ):
        raise ValueError("v5 policy frontend contract changed")
    if (
        policy.get("trusted_current_geometry_canonicalizer")
        != geometry_canonicalizer.canonicalizer_metadata()
    ):
        raise ValueError("v5 policy canonicalizer metadata changed")
    inference = policy.get("inference_contract")
    if (
        not isinstance(inference, Mapping)
        or inference.get(
            "canonicalization_precedes_pose_noise_phase_and_frontend"
        )
        is not True
        or inference.get("selected_profile_is_input") is not False
        or inference.get("selected_class_is_input") is not False
        or inference.get("historical_geometry_unchanged") is not True
        or validate_current_novelty_geometry(
            inference.get("current_novelty_geometry")
        )
        != inference.get("current_novelty_geometry")
    ):
        raise ValueError("v5 inference/canonicalization contract changed")
    fitting = policy.get("fitting_contract")
    if (
        not isinstance(fitting, Mapping)
        or fitting.get("policy_fit_function_reads") != ["enrollment"]
        or fitting.get(
            "bound_classifier_checkpoint_selection_population"
        )
        != (
            "seed-20262904 adaptive development; classifier checkpoint "
            "selection occurred upstream and is not threshold/rank fit"
        )
    ):
        raise ValueError("v5 enrollment-only fitting signature changed")
    for key, expected in FITTING_FIREWALL.items():
        if fitting.get(key) != expected:
            raise ValueError(f"v5 fitting firewall {key} changed")
    if policy.get("gate_contract") != _gate_contract():
        raise ValueError("v5 directional/scale gate contract changed")
    stage_one = policy.get("stage_one")
    if (
        not isinstance(stage_one, Mapping)
        or stage_one.get("uses_frequency_transform") is not False
        or stage_one.get("gates_before_classification") is not False
        or not _is_sha256(stage_one.get("source_v4_policy_sha256"))
        or set(stage_one.get("parameters_by_length", {}))
        != {str(length) for length in RUNTIME_INPUT_LENGTHS}
    ):
        raise ValueError("v5 stage-one policy changed")
    recorded_by_length = stage_one.get(
        "coefficient_artifact_by_length_sha256"
    )
    parameters_by_length = stage_one["parameters_by_length"]
    if (
        not isinstance(recorded_by_length, Mapping)
        or set(recorded_by_length)
        != {str(length) for length in RUNTIME_INPUT_LENGTHS}
    ):
        raise ValueError("v5 stage-one parameter digests changed")
    reproduced_hashes: dict[int, str] = {}
    feature_count = len(noise_prefilter.PREFILTER_FEATURES)
    for length in RUNTIME_INPUT_LENGTHS:
        row = parameters_by_length[str(length)]
        if not isinstance(row, Mapping):
            raise ValueError(f"v5 stage-one N{length} parameters are invalid")
        mean = np.asarray(row.get("mean"), dtype=np.float64)
        scale = np.asarray(row.get("scale"), dtype=np.float64)
        coefficients = np.asarray(
            row.get("coefficients"), dtype=np.float64
        )
        intercept = float(row.get("intercept", np.nan))
        if (
            mean.shape != (feature_count,)
            or scale.shape != mean.shape
            or coefficients.shape != mean.shape
            or not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or not np.all(scale > 0.0)
            or not np.isfinite(coefficients).all()
            or not np.isfinite(intercept)
        ):
            raise ValueError(
                f"v5 stage-one N{length} parameter values are invalid"
            )
        digest = v4_openset._stage_one_parameter_sha256(
            mean=mean,
            scale=scale,
            coefficients=coefficients,
            intercept=intercept,
        )
        if digest != recorded_by_length[str(length)]:
            raise ValueError(
                f"v5 stage-one N{length} parameter digest changed"
            )
        reproduced_hashes[length] = digest
    if (
        v4_openset._stage_one_parameter_set_sha256(reproduced_hashes)
        != stage_one.get("coefficient_artifact_set_sha256")
    ):
        raise ValueError("v5 stage-one parameter-set digest changed")
    per_source = policy.get("per_prototype_source")
    if not isinstance(per_source, Mapping):
        raise ValueError("v5 route policies are missing")
    for source, lengths in RUNTIME_LENGTHS_BY_PROTOTYPE_SOURCE.items():
        row = per_source.get(source)
        if (
            not isinstance(row, Mapping)
            or row.get("supported_public_class_mask")
            != list(
                EXPECTED_PUBLIC_CLASS_SUPPORT_BY_PROTOTYPE_SOURCE[source]
            )
            or set(row.get("per_length", {}))
            != {str(length) for length in lengths}
        ):
            raise ValueError(f"v5 {source} route policy changed")
        for length in lengths:
            route_policy = row["per_length"][str(length)]
            enrollment_rows = route_policy.get("enrollment_rows")
            _validate_route_length_policy(
                route_policy,
                source=source,
                enrollment_rows=enrollment_rows,
            )
    binding = policy.get("classifier_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("v5 classifier binding is missing")
    _validate_fusion_binding(binding)
    provenance = policy.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("amendment_sha256") != AMENDMENT_SHA256
        or provenance.get("recovery_protocol_sha256") != PROTOCOL_SHA256
        or provenance.get("seed_registry_sha256") != SEED_REGISTRY_SHA256
        or provenance.get("identity_firewall_amendment_sha256")
        != IDENTITY_FIREWALL_AMENDMENT_SHA256
        or provenance.get("identity_rejection_sha256")
        != IDENTITY_REJECTION_SHA256
        or provenance.get("identity_acceptance_sha256")
        != IDENTITY_ACCEPTANCE_SHA256
        or provenance.get("source_sha256") != _source_hashes()
    ):
        raise ValueError("v5 open-set provenance/source hashes changed")


def _copy_scores(
    destination: v4_openset.PopulationScores,
    positions: np.ndarray,
    source: v4_openset.PopulationScores,
) -> None:
    v4_openset._copy_scores_into(destination, positions, source)


def score_observations(
    context: DevelopmentContext,
    policy: Mapping[str, Any],
    observations: Sequence[InferenceObservation],
) -> v4_openset.PopulationScores:
    """Score mixed route observations while preserving original row order."""
    validate_policy(policy)
    _validate_policy_context_binding(context, policy)
    source_hashes_at_start = dict(policy["provenance"]["source_sha256"])
    values = list(observations)
    if not values:
        raise ValueError("v5 open-set scoring needs observations")
    prepared = [
        prepare_inference_observation(
            row,
            patch_length=context.patch_length,
            patch_count=context.patch_count,
            target_frac=context.target_frac,
        )
        for row in values
    ]
    output = v4_openset._empty_population_scores(len(values))
    stage_zero = np.asarray(
        [row.is_exact_no_signal for row in prepared], dtype=bool
    )
    output.stage_zero[:] = stage_zero
    output.rejected[stage_zero] = True
    output.final_score[stage_zero] = 4.0
    groups = sorted(
        {
            (row.prototype_source, row.effective_input_length)
            for row in prepared
            if not row.is_exact_no_signal
        }
    )
    for source, length in groups:
        positions = np.asarray(
            [
                index
                for index, row in enumerate(prepared)
                if (
                    not row.is_exact_no_signal
                    and row.prototype_source == source
                    and row.effective_input_length == length
                )
            ],
            dtype=np.int64,
        )
        rows = [prepared[int(index)] for index in positions]
        pose = np.stack(
            [np.asarray(row.pose_features, dtype=np.float64) for row in rows]
        )
        phase = np.asarray(
            [float(row.phase_linearity_score) for row in rows],
            dtype=np.float64,
        )
        stage_one_model = v4_openset._policy_stage_one_model(
            policy, int(length)
        )
        stage_one = np.asarray(
            stage_one_model.score(pose, capture_length=int(length)),
            dtype=np.float64,
        )
        packed = np.stack(
            [np.asarray(row.packed, dtype=np.float32) for row in rows]
        )
        raw_features = np.stack(
            [
                np.asarray(row.raw_frontend_features, dtype=np.float32)
                for row in rows
            ]
        )
        standardized = (
            (
                raw_features
                - np.asarray(
                    context.fusion["feature_mean"], dtype=np.float32
                )
            )
            / np.asarray(
                context.fusion["feature_std"], dtype=np.float32
            )
        ).astype(np.float32)
        embeddings = _embed(context, packed, standardized)
        distances, _closest, _support = _public_distances(
            context,
            embeddings,
            prototype_source=source,
        )
        scored = v4_openset._score_from_distances(
            policy,
            length=int(length),
            prototype_source=source,
            stage_one_score=stage_one,
            phase_linearity_score=phase,
            class_distances=distances,
        )
        _copy_scores(output, positions, scored)
    output.stage_zero[:] = stage_zero
    output.rejected[stage_zero] = True
    output.final_score[stage_zero] = 4.0
    _assert_source_snapshot_unchanged(
        source_hashes_at_start,
        operation="v5 open-set scoring",
    )
    return output


def score_evaluation_observations(
    context: DevelopmentContext,
    policy: Mapping[str, Any],
    observations: Sequence[EvaluationObservation],
) -> tuple[v4_openset.PopulationScores, list[ScoredGateCell]]:
    """Score inference payloads first, then join truth/profile metadata."""
    rows = list(observations)
    scores = score_observations(
        context,
        policy,
        [row.inference for row in rows],
    )
    cells: list[ScoredGateCell] = []
    for row, rejected, prediction in zip(
        rows,
        scores.rejected,
        scores.prediction,
    ):
        truth_index = int(row.truth_index)
        if truth_index < 0 or truth_index >= len(PUBLIC_CLASSES):
            raise ValueError("evaluation truth index is outside public classes")
        predicted = (
            None
            if int(prediction) < 0
            else PUBLIC_CLASSES[int(prediction)]
        )
        cells.append(
            ScoredGateCell(
                source_profile=str(row.source_profile),
                truth=PUBLIC_CLASSES[truth_index],
                predicted=predicted,
                rejected=bool(rejected),
                pair_id=str(row.pair_id),
                observation_length=int(row.observation_length),
                physical_scale_factor=float(row.physical_scale_factor),
            )
        )
    return scores, cells


def selection_evaluation_observations(
    context: DevelopmentContext,
) -> list[EvaluationObservation]:
    """Expose score-only selection observations; never a fitting object."""
    return (
        _role_evaluation_observations(
            context.historical,
            "selection",
            expand_current_observation_lengths=False,
        )
        + _role_evaluation_observations(
            context.current,
            "selection",
            expand_current_observation_lengths=True,
        )
    )


def held_scale_evaluation_observations(
    corpus: Any,
) -> list[EvaluationObservation]:
    """Build explicit-rate inference payloads from an already loaded corpus.

    Loading and seed claiming remain outside this adaptive-development module.
    This function therefore cannot open or generate a sealed corpus.
    """
    output: list[EvaluationObservation] = []
    class_index = {
        name: index for index, name in enumerate(PUBLIC_CLASSES)
    }
    for row in corpus.rows:
        for length in row.available_lengths:
            if int(length) not in RUNTIME_INPUT_LENGTHS:
                continue
            output.append(
                EvaluationObservation(
                    inference=InferenceObservation(
                        iq=np.ascontiguousarray(
                            corpus.prefix(row, int(length)),
                            dtype=np.complex64,
                        ),
                        prototype_source="current",
                        sample_rate_hz=row.sample_rate_hz,
                        native_sample_rate_hz=row.native_sample_rate_hz,
                    ),
                    truth_index=class_index[row.class_name],
                    source_profile=f"current:{row.profile}",
                    base_identity=str(row.pair_id),
                    pair_id=str(row.pair_id),
                    observation_length=int(length),
                    physical_scale_factor=float(row.scale_factor),
                )
            )
    if not output:
        raise ValueError("held-scale corpus has no legal runtime observations")
    return output


def novelty_inference_observations(
    rows: Sequence[np.ndarray],
    *,
    prototype_source: str,
    sample_rate_hz: Any,
    native_sample_rate_hz: Any,
    current_novelty_geometry: Mapping[str, Any] | None,
) -> list[InferenceObservation]:
    """Bind explicit trusted-current rates before any novelty scoring."""
    source = validate_prototype_source(prototype_source)
    if source == "historical":
        if (
            sample_rate_hz is not None
            or native_sample_rate_hz is not None
            or current_novelty_geometry is not None
        ):
            raise ValueError("historical novelty route forbids rate metadata")
    elif sample_rate_hz is None or native_sample_rate_hz is None:
        raise ValueError("current novelty route requires both sample rates")
    else:
        frozen = validate_current_novelty_geometry(
            current_novelty_geometry
        )
        if (
            float(sample_rate_hz) != frozen["sample_rate_hz"]
            or float(native_sample_rate_hz)
            != frozen["native_sample_rate_hz"]
        ):
            raise ValueError(
                "current novelty rates must equal the caller-frozen "
                "class-independent geometry"
            )
    result = [
        InferenceObservation(
            iq=np.asarray(row),
            prototype_source=source,
            sample_rate_hz=sample_rate_hz,
            native_sample_rate_hz=native_sample_rate_hz,
        )
        for row in rows
    ]
    if not result:
        raise ValueError("novelty scoring needs at least one row")
    return result


def evaluate_known_selection(
    context: DevelopmentContext,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Score adaptive selection without returning any fitting input."""
    observations = selection_evaluation_observations(context)
    scores, cells = score_evaluation_observations(
        context, policy, observations
    )
    truth = np.asarray(
        [row.truth_index for row in observations], dtype=np.int64
    )
    groups = [row.source_profile for row in observations]
    current_cells = [
        cell
        for row, cell in zip(observations, cells)
        if row.inference.prototype_source == "current"
    ]
    return {
        "schema": REPORT_SCHEMA,
        "schema_version": REPORT_SCHEMA_VERSION,
        "population_role":
            "historical selection plus seed-20262904 adaptive selection; "
            "score only",
        "summary": v4_openset._population_summary(
            scores,
            truth=truth,
            known_groups=groups,
        ),
        "current_gate_summary": evaluate_current_gate_contract(
            current_cells,
        ),
        "fitting_firewall": dict(FITTING_FIREWALL),
        "selection_rows_returned_to_fitting": 0,
        "development_only": True,
        "release_evidence": False,
        "source_sha256": _source_hashes(),
    }


def evaluate_loaded_held_scale(
    context: DevelopmentContext,
    policy: Mapping[str, Any],
    corpus: Any,
) -> dict[str, Any]:
    """Score a caller-verified held corpus; this function never loads one."""
    observations = held_scale_evaluation_observations(corpus)
    scores, cells = score_evaluation_observations(
        context, policy, observations
    )
    truth = np.asarray(
        [row.truth_index for row in observations], dtype=np.int64
    )
    return {
        "schema": REPORT_SCHEMA,
        "schema_version": REPORT_SCHEMA_VERSION,
        "population_role":
            "caller-verified held trusted-current scale corpus; score only",
        "summary": v4_openset._population_summary(
            scores,
            truth=truth,
            known_groups=[
                row.source_profile for row in observations
            ],
        ),
        "current_gate_summary": evaluate_current_gate_contract(
            cells,
        ),
        "loader_or_seed_claim_performed_here": False,
        "fitting_firewall": dict(FITTING_FIREWALL),
        "development_only": True,
        "release_evidence": False,
        "source_sha256": _source_hashes(),
    }


def evaluate_novelty_populations(
    context: DevelopmentContext,
    policy: Mapping[str, Any],
    novelty_by_family_and_length: Mapping[
        str, Mapping[int, Sequence[np.ndarray]]
    ],
    known_score_by_source_and_effective_length: Mapping[
        str, Mapping[int, np.ndarray]
    ],
    *,
    current_novelty_geometry: Mapping[str, Any],
) -> dict[str, Any]:
    """Score caller-generated adaptive novelty; never fit or generate rows."""
    validate_policy(policy)
    frozen_geometry = validate_current_novelty_geometry(
        current_novelty_geometry
    )
    policy_geometry = policy["inference_contract"][
        "current_novelty_geometry"
    ]
    if frozen_geometry != policy_geometry:
        raise ValueError(
            "novelty geometry differs from the frozen policy binding"
        )
    if not novelty_by_family_and_length:
        raise ValueError("novelty evaluation requires at least one family")
    report: dict[str, Any] = {}
    for source in PROTOTYPE_SOURCE_ROUTES:
        source_known = known_score_by_source_and_effective_length.get(
            source
        )
        if not isinstance(source_known, Mapping):
            raise ValueError(f"novelty known-score route {source} is missing")
        source_report: dict[str, Any] = {}
        for family, by_length in sorted(
            novelty_by_family_and_length.items()
        ):
            if not isinstance(family, str) or not family:
                raise ValueError("novelty family name is invalid")
            family_report: dict[str, Any] = {}
            for observation_length, raw_rows in sorted(by_length.items()):
                observed = int(observation_length)
                if observed not in RUNTIME_INPUT_LENGTHS:
                    raise ValueError(
                        "novelty observation length is not a runtime bucket"
                    )
                if source == "current":
                    sample_rate = frozen_geometry["sample_rate_hz"]
                    native_rate = frozen_geometry[
                        "native_sample_rate_hz"
                    ]
                    geometry = frozen_geometry
                    effective = CURRENT_EFFECTIVE_INPUT_LENGTH
                else:
                    sample_rate = None
                    native_rate = None
                    geometry = None
                    effective = observed
                observations = novelty_inference_observations(
                    raw_rows,
                    prototype_source=source,
                    sample_rate_hz=sample_rate,
                    native_sample_rate_hz=native_rate,
                    current_novelty_geometry=geometry,
                )
                scores = score_observations(context, policy, observations)
                known = np.asarray(
                    source_known.get(effective), dtype=np.float64
                )
                if (
                    known.ndim != 1
                    or len(known) == 0
                    or not np.isfinite(known).all()
                ):
                    raise ValueError(
                        f"{source} known scores lack effective N{effective}"
                    )
                family_report[str(observed)] = {
                    **v4_openset._population_summary(scores),
                    "observation_runtime_input_length": observed,
                    "effective_runtime_input_length": effective,
                    "auroc_vs_route_known": float(
                        v4_openset.auroc(scores.final_score, known)
                    ),
                    "unknown_recall": float(np.mean(scores.rejected)),
                    "trusted_current_geometry": (
                        dict(frozen_geometry)
                        if source == "current"
                        else None
                    ),
                }
            source_report[family] = family_report
        report[source] = source_report
    return {
        "schema": REPORT_SCHEMA,
        "schema_version": REPORT_SCHEMA_VERSION,
        "population_role": "caller-generated adaptive novelty; score only",
        "by_prototype_source": report,
        "current_novelty_geometry": frozen_geometry,
        "rows_used_for_fit_or_selection": 0,
        "fitting_firewall": dict(FITTING_FIREWALL),
        "development_only": True,
        "release_evidence": False,
        "source_sha256": _source_hashes(),
    }


def _mean(values: Sequence[bool | float], *, empty: float = 0.0) -> float:
    return (
        float(np.mean(np.asarray(values, dtype=np.float64)))
        if values
        else float(empty)
    )


def _group_cells(
    cells: Sequence[ScoredGateCell],
    key,
) -> dict[Any, list[ScoredGateCell]]:
    output: dict[Any, list[ScoredGateCell]] = {}
    for row in cells:
        output.setdefault(key(row), []).append(row)
    return output


def _closed_accuracy(rows: Sequence[ScoredGateCell]) -> float:
    return _mean([row.closed_correct for row in rows])


def _abstention_damage(rows: Sequence[ScoredGateCell]) -> float:
    correct = [row for row in rows if row.closed_correct]
    return _mean([row.rejected for row in correct], empty=1.0)


def _present_class_balanced_accuracy(
    rows: Sequence[ScoredGateCell],
) -> float:
    by_truth = _group_cells(rows, lambda row: row.truth)
    return _mean(
        [_closed_accuracy(group) for group in by_truth.values()]
    )


def _agreement(rows: Sequence[ScoredGateCell]) -> bool:
    return len({row.outcome for row in rows}) == 1


def _gate_result(
    value: bool | float,
    comparison: str,
    threshold: bool | float,
) -> bool:
    if comparison == "equals":
        return value is threshold
    numeric = float(value)
    target = float(threshold)
    if not np.isfinite(numeric):
        return False
    if comparison == "greater_than_or_equal":
        return numeric >= target
    if comparison == "less_than_or_equal":
        return numeric <= target
    raise ValueError(f"unknown gate comparison {comparison!r}")


def _required_current_evaluation_cell_keys(
) -> set[tuple[str, int, float]]:
    return {
        (f"current:{profile}", int(length), float(scale))
        for profile in REQUIRED_CURRENT_PROFILE_PUBLIC_CLASS_MAP
        for length in REQUIRED_CURRENT_EVALUATION_LENGTHS
        for scale in REQUIRED_CURRENT_EVALUATION_SCALES
    }


def evaluate_current_gate_contract(
    cells: Sequence[ScoredGateCell],
) -> dict[str, Any]:
    """Evaluate the exact inherited 19 gates plus two directional gates."""
    rows = list(cells)
    if not rows:
        raise ValueError("current gate evaluation needs scored observations")
    canonical_profile_truth = {
        f"current:{profile}": public_class
        for profile, public_class
        in REQUIRED_CURRENT_PROFILE_PUBLIC_CLASS_MAP.items()
    }
    for row in rows:
        if (
            row.source_profile not in canonical_profile_truth
            or row.truth != canonical_profile_truth[row.source_profile]
            or (
                row.predicted is not None
                and row.predicted not in PUBLIC_CLASSES
            )
            or row.observation_length
                not in REQUIRED_CURRENT_EVALUATION_LENGTHS
            or row.physical_scale_factor
                not in REQUIRED_CURRENT_EVALUATION_SCALES
            or not row.pair_id
        ):
            raise ValueError("current gate cell metadata is invalid")

    required_cell_keys = _required_current_evaluation_cell_keys()
    required_pair_views = {
        (int(length), float(scale))
        for length in REQUIRED_CURRENT_EVALUATION_LENGTHS
        for scale in REQUIRED_CURRENT_EVALUATION_SCALES
    }
    observed_cell_counts: dict[tuple[str, int, float], int] = {}
    pair_views: dict[str, set[tuple[int, float]]] = {}
    pair_row_counts: dict[str, int] = {}
    pair_profiles: dict[str, str] = {}
    duplicate_pair_view = False
    inconsistent_pair_profile = False
    pairs_by_profile: dict[str, set[str]] = {
        profile: set() for profile in canonical_profile_truth
    }
    for row in rows:
        cell_key = (
            row.source_profile,
            int(row.observation_length),
            float(row.physical_scale_factor),
        )
        observed_cell_counts[cell_key] = (
            observed_cell_counts.get(cell_key, 0) + 1
        )
        pair_id = str(row.pair_id)
        pair_view = (
            int(row.observation_length),
            float(row.physical_scale_factor),
        )
        views = pair_views.setdefault(pair_id, set())
        if pair_view in views:
            duplicate_pair_view = True
        views.add(pair_view)
        pair_row_counts[pair_id] = pair_row_counts.get(pair_id, 0) + 1
        prior_profile = pair_profiles.setdefault(
            pair_id, row.source_profile
        )
        if prior_profile != row.source_profile:
            inconsistent_pair_profile = True
        pairs_by_profile[row.source_profile].add(pair_id)

    observed_cell_keys = set(observed_cell_counts)
    cell_counts_meet_floor = (
        observed_cell_keys == required_cell_keys
        and all(
            observed_cell_counts[key]
            >= REQUIRED_CURRENT_EVALUATION_MINIMUM_PAIRS_PER_PROFILE
            for key in required_cell_keys
        )
    )
    pair_inventory_complete = (
        not duplicate_pair_view
        and not inconsistent_pair_profile
        and all(
            views == required_pair_views
            and pair_row_counts[pair_id] == len(required_pair_views)
            for pair_id, views in pair_views.items()
        )
    )
    profile_pair_floor_complete = all(
        len(pair_ids)
        >= REQUIRED_CURRENT_EVALUATION_MINIMUM_PAIRS_PER_PROFILE
        for pair_ids in pairs_by_profile.values()
    )
    cell_counts_equal_profile_pair_counts = all(
        observed_cell_counts.get(
            (profile, int(length), float(scale)),
            -1,
        )
        == len(pairs_by_profile[profile])
        for profile in canonical_profile_truth
        for length in REQUIRED_CURRENT_EVALUATION_LENGTHS
        for scale in REQUIRED_CURRENT_EVALUATION_SCALES
    )
    exact_profile_coverage = (
        cell_counts_meet_floor
        and profile_pair_floor_complete
        and cell_counts_equal_profile_pair_counts
    )
    exact_observation_coverage = (
        exact_profile_coverage
        and pair_inventory_complete
        and len(rows)
            == len(pair_views) * len(required_pair_views)
    )

    observed_profile_counts: dict[str, int] = {}
    for (profile, length, scale), count in sorted(
        observed_cell_counts.items()
    ):
        key = (
            f"{profile}|N={length}"
            f"|scale={scale:g}"
        )
        observed_profile_counts[key] = int(count)
    expected_profile_counts = {
        (
            f"{profile}|N={length}|scale={scale:g}"
        ): (
            f">={REQUIRED_CURRENT_EVALUATION_MINIMUM_PAIRS_PER_PROFILE}"
        )
        for profile, length, scale in sorted(required_cell_keys)
    }
    canonical_profiles = set(canonical_profile_truth)
    observed_profiles = {row.source_profile for row in rows}

    legal_cells = _group_cells(
        rows,
        lambda row: (
            row.source_profile,
            row.observation_length,
            row.physical_scale_factor,
        ),
    )
    length_scale_cells = _group_cells(
        rows,
        lambda row: (
            row.observation_length,
            row.physical_scale_factor,
        ),
    )
    profile_pooled = _group_cells(rows, lambda row: row.source_profile)
    per_length = _group_cells(rows, lambda row: row.observation_length)
    per_truth = _group_cells(rows, lambda row: row.truth)

    physical_groups = _group_cells(
        rows, lambda row: (row.pair_id, row.observation_length)
    )
    physical_agreement = {
        key: _agreement(group) for key, group in physical_groups.items()
    }
    physical_by_length = {
        length: _mean(
            [
                value
                for (pair_id, current_length), value
                in physical_agreement.items()
                if current_length == length
            ]
        )
        for length in sorted(per_length)
    }
    causal_groups = _group_cells(
        rows,
        lambda row: (row.pair_id, row.physical_scale_factor),
    )
    causal_agreement = {
        key: _agreement(group) for key, group in causal_groups.items()
    }
    causal_by_profile: dict[str, list[bool]] = {}
    profile_by_pair = {
        row.pair_id: row.source_profile for row in rows
    }
    for (pair_id, _scale), value in causal_agreement.items():
        causal_by_profile.setdefault(
            profile_by_pair[pair_id], []
        ).append(value)
    pair_groups = _group_cells(rows, lambda row: row.pair_id)
    pair_agreement = {
        pair_id: _agreement(group)
        for pair_id, group in pair_groups.items()
    }
    pair_by_profile: dict[str, list[bool]] = {}
    for pair_id, value in pair_agreement.items():
        pair_by_profile.setdefault(
            profile_by_pair[pair_id], []
        ).append(value)

    supported_profile_damage = [
        _abstention_damage(group)
        for group in profile_pooled.values()
        if sum(row.closed_correct for row in group)
        >= CONDITIONAL_CELL_MINIMUM_CLOSED_CORRECT_ROWS
    ]
    supported_truth_damage = [
        _abstention_damage(group)
        for group in per_truth.values()
        if sum(row.closed_correct for row in group)
        >= CONDITIONAL_CELL_MINIMUM_CLOSED_CORRECT_ROWS
    ]
    wifi_cells = [
        group
        for (profile, _length, _scale), group in legal_cells.items()
        if profile == WIFI_HR_DSSS_SOURCE_PROFILE
    ]
    wifi_accuracy = min(
        (_closed_accuracy(group) for group in wifi_cells),
        default=0.0,
    )
    wifi_directional = max(
        (
            _mean(
                [
                    row.predicted == "bluetooth"
                    for row in group
                    if row.truth == "dsss"
                ],
                empty=1.0,
            )
            for group in wifi_cells
        ),
        default=1.0,
    )
    bluetooth_cells = [
        group
        for (profile, _length, _scale), group in legal_cells.items()
        if any(row.truth == "bluetooth" for row in group)
        and profile.startswith("current:bluetooth-")
    ]
    expected_bluetooth_profiles = {
        f"current:{profile}"
        for profile, public_class
        in REQUIRED_CURRENT_PROFILE_PUBLIC_CLASS_MAP.items()
        if public_class == "bluetooth"
    }
    observed_bluetooth_profiles = {
        row.source_profile for row in rows if row.truth == "bluetooth"
    }
    bluetooth_directional = (
        max(
            (
                _mean(
                    [
                        row.predicted == "dsss"
                        for row in group
                        if row.truth == "bluetooth"
                    ],
                    empty=1.0,
                )
                for group in bluetooth_cells
            ),
            default=1.0,
        )
        if observed_bluetooth_profiles == expected_bluetooth_profiles
        else 1.0
    )

    values: dict[str, bool | float] = {
        "causal_prefix_outcome_agreement_across_observation_lengths":
            _mean(list(causal_agreement.values())),
        "causal_prefix_outcome_agreement_every_source_profile":
            min(
                (
                    _mean(value)
                    for value in causal_by_profile.values()
                ),
                default=0.0,
            ),
        "classifier_development_current_service_every_legal_cell_closed_accuracy":
            min(
                (_closed_accuracy(group) for group in legal_cells.values()),
                default=0.0,
            ),
        "classifier_development_current_service_pooled_closed_accuracy":
            _closed_accuracy(rows),
        "classifier_development_current_service_pooled_profile_balanced_accuracy":
            _mean(
                [
                    _closed_accuracy(group)
                    for group in profile_pooled.values()
                ]
            ),
        "classifier_development_current_service_wifi_hr_dsss_worst_legal_cell_accuracy":
            wifi_accuracy,
        "classifier_development_current_service_worst_cell_present_class_balanced_accuracy":
            min(
                (
                    _present_class_balanced_accuracy(group)
                    for group in length_scale_cells.values()
                ),
                default=0.0,
            ),
        "classifier_development_current_service_worst_profile_legal_cell_closed_accuracy":
            min(
                (_closed_accuracy(group) for group in legal_cells.values()),
                default=0.0,
            ),
        "classifier_development_current_service_worst_profile_pooled_closed_accuracy":
            min(
                (
                    _closed_accuracy(group)
                    for group in profile_pooled.values()
                ),
                default=0.0,
            ),
        "exact_eligible_pair_and_observation_coverage":
            exact_observation_coverage,
        "exact_profile_length_scale_key_and_count_coverage":
            exact_profile_coverage,
        "known_abstention_damage_given_closed_head_correct_pooled":
            _abstention_damage(rows),
        "known_abstention_damage_given_closed_head_correct_worst_observation_length":
            max(
                (
                    _abstention_damage(group)
                    for group in per_length.values()
                ),
                default=1.0,
            ),
        "known_abstention_damage_given_closed_head_correct_worst_supported_source_profile_cell":
            max(supported_profile_damage, default=1.0),
        "known_abstention_damage_given_closed_head_correct_worst_supported_true_class_cell":
            max(supported_truth_damage, default=1.0),
        "length_scale_outcome_agreement_every_source_profile":
            min(
                (
                    _mean(value)
                    for value in pair_by_profile.values()
                ),
                default=0.0,
            ),
        "length_scale_outcome_agreement_pooled":
            _mean(list(pair_agreement.values())),
        "physical_scale_outcome_agreement_every_observation_length":
            min(physical_by_length.values(), default=0.0),
        "physical_scale_outcome_agreement_pooled":
            _mean(list(physical_agreement.values())),
        "wifi_hr_dsss_to_bluetooth_worst_legal_cell_confusion_rate":
            wifi_directional,
        "bluetooth_to_dsss_worst_legal_cell_confusion_rate":
            bluetooth_directional,
    }
    expected_names = (
        set(recovery_validator.EXPECTED_INHERITED_GATES)
        | set(recovery_validator.EXPECTED_DIRECTIONAL_GATES)
    )
    if set(values) != expected_names:
        raise AssertionError("v5 gate value inventory changed")
    gates: dict[str, dict[str, Any]] = {}
    for name, (comparison, threshold) in (
        recovery_validator.EXPECTED_INHERITED_GATES.items()
    ):
        value = values[name]
        gates[name] = {
            "value": value,
            "comparison": comparison,
            "threshold": threshold,
            "passes": _gate_result(value, comparison, threshold),
        }
    for name, contract in (
        recovery_validator.EXPECTED_DIRECTIONAL_GATES.items()
    ):
        value = values[name]
        comparison = str(contract["comparison"])
        threshold = float(contract["threshold"])
        gates[name] = {
            "value": value,
            **dict(contract),
            "passes": _gate_result(value, comparison, threshold),
        }
    return {
        "schema": REPORT_SCHEMA,
        "schema_version": REPORT_SCHEMA_VERSION,
        "development_only": True,
        "release_evidence": False,
        "gate_contract": _gate_contract(),
        "gates": gates,
        "all_pass": all(row["passes"] is True for row in gates.values()),
        "coverage": {
            "required_source_profile_length_scale_cells":
                len(required_cell_keys),
            "observed_source_profile_length_scale_cells":
                len(observed_cell_keys),
            "required_views_per_pair": len(required_pair_views),
            "observed_pairs": len(pair_views),
            "minimum_pairs_per_source_profile":
                REQUIRED_CURRENT_EVALUATION_MINIMUM_PAIRS_PER_PROFILE,
            "minimum_required_observations": (
                len(canonical_profiles)
                * REQUIRED_CURRENT_EVALUATION_MINIMUM_PAIRS_PER_PROFILE
                * len(required_pair_views)
            ),
            "observed_observations": len(rows),
            "pair_inventory_complete": pair_inventory_complete,
            "profile_pair_floor_complete": profile_pair_floor_complete,
            "cell_counts_equal_profile_pair_counts":
                cell_counts_equal_profile_pair_counts,
            "exact_observation_coverage": exact_observation_coverage,
            "required_profile_length_scale_counts":
                expected_profile_counts,
            "observed_profile_length_scale_counts":
                observed_profile_counts,
            "exact_profile_length_scale_coverage":
                exact_profile_coverage,
            "expected_source_profiles": sorted(canonical_profiles),
            "observed_source_profiles": sorted(observed_profiles),
        },
        "fitting_firewall": dict(FITTING_FIREWALL),
        "source_sha256": _source_hashes(),
    }


def assert_no_selected_profile_or_class_inference_input() -> None:
    """Static runtime guard for the two public inference entry points."""
    forbidden = {
        "selected_profile",
        "selected_class",
        "profile_id",
        "class_name",
    }
    for function in (
        canonicalize_inference_observation,
        prepare_inference_observation,
        score_observations,
    ):
        overlap = forbidden & set(inspect.signature(function).parameters)
        if overlap:
            raise AssertionError(
                f"{function.__name__} exposes forbidden inference inputs "
                f"{sorted(overlap)}"
            )


assert_no_selected_profile_or_class_inference_input()
