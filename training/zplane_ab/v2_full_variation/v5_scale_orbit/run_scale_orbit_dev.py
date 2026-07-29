"""Train a v5 scale-orbit branch on historical plus paired-rate current I/Q.

This new-lineage runner reuses the frozen FFT-free patch CNN while adding the
trusted-current physical-time canonicalizer preregistered for v5. Historical
row membership comes only from
``invariant_patch_data.load``; the already-consumed historical test half is
neither loaded nor addressable.  A caller-supplied current corpus contributes
independent train, enrollment, and selection rows.

Checkpoint selection is predeclared and lexicographic.  Its leading term is
the minimum of the historical floor and every scale-sensitive current-route
bottleneck (worst scale, worst profile/scale, WiFi HR-DSSS, paired-scale
agreement, and the directional DSSS/Bluetooth margin).  Remaining terms break
ties without allowing pooled accuracy to hide a failed scale/profile cell.

Do not change or tune that order after observing a run.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
V2 = HERE.parent
ZPLANE = V2.parent
TRAINING = ZPLANE.parent
V4 = V2 / "v4_current_source"
for path in (TRAINING, ZPLANE, V2, V4, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import current_source_data as corpus_data  # noqa: E402
import run_invariant_cnn_dev as v3_runner  # noqa: E402
import time_domain_invariant_patch_preprocess as td_preprocess  # noqa: E402
import scale_orbit_data  # noqa: E402
import trusted_geometry_canonicalizer as geometry_canonicalizer  # noqa: E402
from invariant_patch_cnn import InvariantPatchCNN, InvariantPatchConfig  # noqa: E402
from train import embed_all, sq_dist  # noqa: E402


HISTORICAL_CORPUS = TRAINING / "artifacts" / "signallab-corpus"
CHECKPOINT_SCORE_CONTRACT = (
    "lexicographic(min(historical_worst_length_balanced_accuracy,"
    "current_worst_scale_present_class_balanced_accuracy,"
    "current_worst_profile_scale_recall,"
    "current_physical_scale_prediction_agreement_rate,"
    "current_wifi_hr_dsss_worst_scale_accuracy,"
    "one_minus_current_worst_directional_dsss_bluetooth_confusion_rate),"
    "current_worst_scale_present_class_balanced_accuracy,"
    "current_wifi_hr_dsss_worst_scale_accuracy,"
    "current_worst_profile_scale_recall,"
    "current_physical_scale_prediction_agreement_rate,"
    "one_minus_current_worst_directional_dsss_bluetooth_confusion_rate,"
    "current_pooled_profile_balanced_accuracy,"
    "current_pooled_accuracy,"
    "combined_pooled_balanced_accuracy,"
    "historical_worst_length_balanced_accuracy)"
)
CHECKPOINT_SCORE_TERMS = 10
SAMPLER_CONTRACT = (
    "numpy.default_rng(seed); for each public class allocate k_shot+q_query "
    "distinct base identities; current_source_share=1/2 preserves randomized "
    "balanced round-robin over nonempty sources; any other exact rational "
    "share uses deterministic cumulative apportionment across eligible "
    "(episode,class) units and randomized slot order, spilling only when a "
    "source exhausts; within each chosen source allocate by randomized "
    "balanced round-robin over nonempty profiles; then select one uniform "
    "eligible runtime-prefix view from each selected base identity"
)
ATTEMPT_1_ADAPTATION_PATH = (
    HERE / "scale_consistency_adaptation_attempt_1.json"
)
ATTEMPT_1_ADAPTATION_SHA256 = (
    "e3166235ed53835465b3bd7afa6ad03982e31016b4d20c6875879288b3304a90"
)
ATTEMPT_1_MISS_REPORT_PATH = (
    HERE / "scale_consistency_adaptation_attempt_1_miss_report.json"
)
ATTEMPT_1_MISS_REPORT_SHA256 = (
    "1c0e6d94d418e3954a54aa44396ef5cc9d8603eb2895dd69c0a0c458c254bc9d"
)
SUPERVISED_PAIR_ADAPTATION_PATH = (
    HERE / "scale_consistency_adaptation_attempt_2.json"
)
SUPERVISED_PAIR_ADAPTATION_SHA256 = (
    "5e0cc7bbd58ebe050bf983f854e154c43595dc804aedee3c1c3849cf5691b0a9"
)
SUPERVISED_PAIR_PREREGISTRATION_COMMIT = (
    "53ae9eb2060a0f65a2ffcf607d98866bb695e65c"
)
SUPERVISED_PAIR_WEIGHT = 0.2
SUPERVISED_PAIR_RNG_XOR = 0x5CA1E
SUPERVISED_PAIR_EXPECTED_PROFILE_COUNT = 31
SUPERVISED_PAIR_PUBLIC_CLASS_ORDER = (
    "am",
    "bluetooth",
    "cw",
    "dsss",
    "fm",
    "gsm",
    "ofdm",
)
SUPERVISED_PAIR_PROFILES = (
    "bluetooth-classic-connected",
    "bluetooth-le-advertising",
    "gsm-16qam-higher-symbol-rate-burst",
    "gsm-32qam-higher-symbol-rate-burst",
    "gsm-8psk-normal-burst",
    "gsm-900-loaded-bcch",
    "gsm-aqpsk-normal-burst",
    "gsm-normal-burst",
    "gsm-qpsk-higher-symbol-rate-burst",
    "lte-band3-fdd-20m",
    "lte-band38-tdd-10m",
    "lte-etm1.1",
    "lte-etm3.1",
    "lte-etm3.1a",
    "lte-etm3.1b",
    "lte-nbiot-guard-isolated-component",
    "lte-nbiot-inband-isolated-component",
    "lte-ntm",
    "nr-fr1-tm1.1",
    "nr-fr1-tm3.1",
    "nr-fr1-tm3.1a",
    "nr-fr1-tm3.1b",
    "nr-n3-fdd-20m",
    "nr-n78-tdd-100m",
    "nr-nbiot-inband-isolated-component",
    "wifi-hr-dsss-11m",
    "wifi-ofdm-20m",
    "wifi6-he-er-su",
    "wifi6-he-mu",
    "wifi6-he-su",
    "wifi6-he-tb",
)
SUPERVISED_PAIR_PROFILE_PUBLIC_CLASS_MAP = {
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
SUPERVISED_PAIR_PROFILE_CLASS_INDICES = (
    1, 1,
    5, 5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    3,
    6, 6, 6, 6, 6,
)
SUPERVISED_PAIR_TARGET_CONSTRUCTION = (
    "repeat(profile_public_class_indices, 2)"
)
SUPERVISED_PAIR_TARGET_SHA256 = (
    "838de46075dac50b6c58dee9c29fd4149711cb74bcda692b8f9edd8913b25927"
)
SUPERVISED_PAIR_PAIR_CONTRACT = (
    "each episode independently selects one seed20264101 current train "
    "identity uniformly within each of the literal 31 current profiles, then "
    "selects two distinct physical-scale views uniformly without replacement"
)
SUPERVISED_PAIR_LOSS_CONTRACT = (
    "episodic_cross_entropy_plus_0.2_times_mean_supervised_public_class_cross_"
    "entropy_over_62_profile_contiguous_scale_pair_views_against_detached_"
    "current_episode_source_mixed_prototypes_and_detached_logit_scale"
)


@dataclass(frozen=True)
class SupervisedPairIdentity:
    """One train-only physical identity and its exact scale-view positions."""

    identity: str
    source: str
    role: str
    population_seed: int
    profile_id: str
    public_class_name: str
    public_class_index: int
    scale_factors: tuple[float, ...]
    view_positions: tuple[int, ...]


def _source_share_fraction(value: Any) -> Fraction:
    if isinstance(value, bool):
        raise ValueError("current_source_share must be an exact fraction")
    try:
        share = value if isinstance(value, Fraction) else Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(
            "current_source_share must be a decimal or fraction"
        ) from exc
    if share <= 0 or share >= 1:
        raise ValueError("current_source_share must lie strictly between 0 and 1")
    return share


def _parse_source_share(value: str) -> Fraction:
    return _source_share_fraction(value)


def _supervised_pair_profile_class_indices(
    classes: Sequence[str],
) -> tuple[int, ...]:
    """Return the frozen class-index vector in literal profile order."""
    if tuple(classes) != SUPERVISED_PAIR_PUBLIC_CLASS_ORDER:
        raise ValueError(
            "supervised-pair public class order must remain "
            f"{list(SUPERVISED_PAIR_PUBLIC_CLASS_ORDER)}"
        )
    if (
        tuple(SUPERVISED_PAIR_PROFILE_PUBLIC_CLASS_MAP)
        != SUPERVISED_PAIR_PROFILES
        or tuple(corpus_data.CURRENT_PROFILES) != SUPERVISED_PAIR_PROFILES
        or dict(corpus_data.CURRENT_PROFILE_PUBLIC_CLASS_MAP)
        != SUPERVISED_PAIR_PROFILE_PUBLIC_CLASS_MAP
    ):
        raise RuntimeError(
            "the exact supervised-pair profile/public-class map changed"
        )
    class_index = {
        class_name: index for index, class_name in enumerate(classes)
    }
    result = tuple(
        int(class_index[SUPERVISED_PAIR_PROFILE_PUBLIC_CLASS_MAP[profile]])
        for profile in SUPERVISED_PAIR_PROFILES
    )
    if result != SUPERVISED_PAIR_PROFILE_CLASS_INDICES:
        raise RuntimeError(
            "the frozen supervised-pair profile class-index vector changed"
        )
    return result


def _supervised_pair_target_sha256(
    profile_class_indices: Sequence[int],
) -> str:
    """Hash the profile-contiguous `repeat(labels, 2)` target vector."""
    indices = np.asarray(profile_class_indices)
    if (
        indices.shape != (SUPERVISED_PAIR_EXPECTED_PROFILE_COUNT,)
        or indices.dtype.kind not in {"i", "u"}
        or np.any(indices < 0)
    ):
        raise ValueError(
            "supervised-pair profile class indices must be 31 non-negative "
            "integers"
        )
    targets = np.repeat(indices.astype(np.int64, copy=False), 2)
    observed = corpus_data.sha256_json(targets.tolist())
    if (
        tuple(int(value) for value in indices)
        == SUPERVISED_PAIR_PROFILE_CLASS_INDICES
        and observed != SUPERVISED_PAIR_TARGET_SHA256
    ):
        raise RuntimeError("frozen supervised-pair target SHA-256 changed")
    return observed


def _read_bound_adaptation_document(
    path: Path,
    expected_sha256: str,
    *,
    name: str,
) -> tuple[dict[str, Any], str]:
    observed = corpus_data.sha256_file(path)
    if observed != expected_sha256:
        raise RuntimeError(
            f"{name} SHA-256 changed: {observed}"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{name} is unreadable") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must contain one JSON object")
    return value, observed


def _validate_supervised_pair_adaptation_source() -> dict[str, Any]:
    """Bind attempt 1, its miss, and the exact pre-source attempt-2 intent."""
    attempt_1, attempt_1_sha256 = _read_bound_adaptation_document(
        ATTEMPT_1_ADAPTATION_PATH,
        ATTEMPT_1_ADAPTATION_SHA256,
        name="attempt-1 adaptation intent",
    )
    miss_report, miss_report_sha256 = _read_bound_adaptation_document(
        ATTEMPT_1_MISS_REPORT_PATH,
        ATTEMPT_1_MISS_REPORT_SHA256,
        name="attempt-1 miss report",
    )
    attempt_2, attempt_2_sha256 = _read_bound_adaptation_document(
        SUPERVISED_PAIR_ADAPTATION_PATH,
        SUPERVISED_PAIR_ADAPTATION_SHA256,
        name="attempt-2 supervised-pair intent",
    )
    if (
        attempt_1.get("adaptation_attempt") != 1
        or attempt_1.get("status")
        != "frozen_before_scale_consistency_source_change_or_training"
        or attempt_1.get("development_only") is not True
        or attempt_1.get("release_evidence") is not False
    ):
        raise RuntimeError("attempt-1 adaptation evidence changed")
    if (
        miss_report.get("schema")
        != "atomos.v5.scale-orbit.adaptation-attempt-miss-report"
        or miss_report.get("adaptation_attempt") != 1
        or miss_report.get("full_gate_assessment", {}).get(
            "full_21_gate_status"
        )
        != "not_evaluated"
        or miss_report.get("development_only") is not True
        or miss_report.get("release_evidence") is not False
    ):
        raise RuntimeError("attempt-1 miss evidence changed")
    parent_bindings = attempt_2.get("parent_bindings")
    change = attempt_2.get("predeclared_change")
    sampling = (
        change.get("pair_sampling")
        if isinstance(change, Mapping)
        else None
    )
    auxiliary = (
        change.get("supervised_pair_auxiliary")
        if isinstance(change, Mapping)
        else None
    )
    fitting = attempt_2.get("fitting_firewall")
    if (
        attempt_2.get("schema")
        != (
            "atomos.v5.scale-orbit.post-score-supervised-pair-"
            "adaptation-intent"
        )
        or attempt_2.get("status")
        != (
            "frozen_before_attempt_2_trainer_or_assembler_change_training_"
            "or_inference"
        )
        or attempt_2.get("adaptation_attempt") != 2
        or attempt_2.get("development_only") is not True
        or attempt_2.get("release_evidence") is not False
        or not isinstance(parent_bindings, Mapping)
        or parent_bindings.get("attempt_1_intent_sha256")
        != attempt_1_sha256
        or parent_bindings.get("attempt_1_miss_report_sha256")
        != miss_report_sha256
        or not isinstance(sampling, Mapping)
        or sampling.get("literal_profiles")
        != list(SUPERVISED_PAIR_PROFILES)
        or sampling.get("literal_profile_public_class_map")
        != SUPERVISED_PAIR_PROFILE_PUBLIC_CLASS_MAP
        or sampling.get("views_per_episode") != 62
        or not isinstance(auxiliary, Mapping)
        or auxiliary.get("auxiliary_coefficient")
        != SUPERVISED_PAIR_WEIGHT
        or auxiliary.get("cross_entropy", {}).get("mean_denominator") != 62
        or auxiliary.get("public_class_prototypes", {}).get(
            "detached_for_auxiliary"
        )
        is not True
        or auxiliary.get("logits", {}).get(
            "logit_scale_detached_for_auxiliary"
        )
        is not True
        or not isinstance(fitting, Mapping)
        or fitting.get(
            "supervised_pairs_from_seed20264101_current_train_role_only"
        )
        is not True
        or fitting.get(
            "seed20264101_enrollment_rows_used_for_supervised_pair_auxiliary"
        )
        != 0
        or fitting.get(
            "seed20262904_selection_rows_used_for_any_gradient_or_optimizer_step"
        )
        != 0
        or fitting.get(
            "seed20262904_selection_rows_used_for_weight_center_or_feature_moment_fit"
        )
        != 0
        or fitting.get(
            "seed20262904_selection_rows_used_for_persistent_prototype_fit"
        )
        != 0
        or fitting.get(
            "seed20262904_selection_rows_used_for_open_set_threshold_or_rank_fit"
        )
        != 0
        or fitting.get(
            "seed20262904_selection_metrics_may_retain_the_existing_development_"
            "checkpoint_and_candidate_selection_role"
        )
        is not True
        or fitting.get(
            "sealed_rows_used_for_any_fit_checkpoint_or_candidate_selection"
        )
        != 0
    ):
        raise RuntimeError(
            "attempt-2 supervised-pair intent no longer matches the "
            "implemented loss or fitting firewall"
        )
    return {
        "path": str(SUPERVISED_PAIR_ADAPTATION_PATH),
        "sha256": attempt_2_sha256,
        "status": attempt_2["status"],
        "adaptation_attempt": 2,
        "preregistration_commit":
            SUPERVISED_PAIR_PREREGISTRATION_COMMIT,
        "attempt_1_intent_path": str(ATTEMPT_1_ADAPTATION_PATH),
        "attempt_1_intent_sha256": attempt_1_sha256,
        "attempt_1_miss_report_path": str(ATTEMPT_1_MISS_REPORT_PATH),
        "attempt_1_miss_report_sha256": miss_report_sha256,
    }


def _validate_supervised_pair_weight(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("supervised_pair_weight must be the frozen 0.2")
    try:
        weight = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "supervised_pair_weight must be the frozen 0.2"
        ) from exc
    if not math.isfinite(weight) or weight != SUPERVISED_PAIR_WEIGHT:
        raise ValueError(
            "supervised_pair_weight must equal the preregistered 0.2"
        )
    return weight


def _validate_adaptation_run_configuration(args: argparse.Namespace) -> None:
    """Reject any run that differs from frozen supervised-pair attempt 2."""
    _validate_supervised_pair_weight(args.supervised_pair_weight)
    expected = {
        "seed": 20_260_740,
        "episodes": 8_000,
        "eval_every": 500,
        "k_shot": 5,
        "q_query": 5,
        "patch_length": 64,
        "patch_count": 16,
        "target_frac": 0.5,
        "patch_dim": 64,
        "hidden": 96,
        "set_pool": "mean_std",
        "dropout": 0.35,
        "lr": 1e-3,
        "weight_decay": 5e-4,
        "warmup_frac": 0.03,
        "label_smoothing": 0.0,
        "phase_augmentation": True,
    }
    changed = [
        name
        for name, value in expected.items()
        if getattr(args, name, None) != value
    ]
    if changed:
        raise ValueError(
            "supervised-pair attempt 2 configuration differs from the frozen "
            f"baseline for: {', '.join(changed)}"
        )
    if _source_share_fraction(args.current_source_share) != Fraction(1, 3):
        raise ValueError(
            "supervised-pair attempt 2 requires current_source_share=1/3"
        )
    if getattr(args, "encoder", None) not in {"real", "complex"}:
        raise ValueError(
            "supervised-pair attempt 2 encoder must be real or complex"
        )


def _executed_source_paths() -> dict[str, Path]:
    """Return every local Python source executed by the v5 branch runner."""
    return {
        "v5/run_scale_orbit_dev.py": Path(__file__).resolve(),
        "v5/scale_orbit_data.py": Path(scale_orbit_data.__file__).resolve(),
        "v5/trusted_geometry_canonicalizer.py":
            Path(geometry_canonicalizer.__file__).resolve(),
        "v5/identity_firewall_amendment.json":
            scale_orbit_data.SCALE_ORBIT_IDENTITY_FIREWALL_AMENDMENT_PATH.resolve(),
        "v5/seed20262904_identity_rejection.json":
            scale_orbit_data.SCALE_ORBIT_IDENTITY_REJECTION_PATH.resolve(),
        "v5/seed20262904_identity_firewall_acceptance.json":
            scale_orbit_data.SCALE_ORBIT_IDENTITY_ACCEPTANCE_PATH.resolve(),
        "v5/scale_consistency_adaptation_attempt_1.json":
            ATTEMPT_1_ADAPTATION_PATH.resolve(),
        "v5/scale_consistency_adaptation_attempt_1_miss_report.json":
            ATTEMPT_1_MISS_REPORT_PATH.resolve(),
        "v5/scale_consistency_adaptation_attempt_2.json":
            SUPERVISED_PAIR_ADAPTATION_PATH.resolve(),
        "v4/current_source_data.py": Path(corpus_data.__file__).resolve(),
        "v4/evaluate_current_scale.py":
            Path(scale_orbit_data.scale_data.__file__).resolve(),
        "v2/run_invariant_cnn_dev.py": Path(v3_runner.__file__).resolve(),
        "v3/time_domain_invariant_patch_preprocess.py":
            TRAINING / "time_domain_invariant_patch_preprocess.py",
        "v3/time_domain_geometry.py": TRAINING / "time_domain_geometry.py",
        "v3/invariant_patch_preprocess.py":
            TRAINING / "invariant_patch_preprocess.py",
        "v3/invariant_patch_cnn.py": V2 / "invariant_patch_cnn.py",
        "v3/invariant_patch_data.py": V2 / "invariant_patch_data.py",
        "v2/complex_multiscale_backbone.py":
            V2 / "complex_multiscale_backbone.py",
        "v2/unet_transfer.py": V2 / "unet_transfer.py",
        "v2/vit_backbone.py": V2 / "vit_backbone.py",
        "v2/multitask_autoencoder.py": V2 / "multitask_autoencoder.py",
        "training/model.py": TRAINING / "model.py",
        "training/preprocess.py": TRAINING / "preprocess.py",
        "training/train.py": TRAINING / "train.py",
    }


def _source_hashes() -> dict[str, str]:
    return {
        name: corpus_data.sha256_file(path)
        for name, path in sorted(_executed_source_paths().items())
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
    if isinstance(value, Fraction):
        return {
            "numerator": int(value.numerator),
            "denominator": int(value.denominator),
            "value": float(value),
        }
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


def checkpoint_score(
    *,
    historical_balanced: float,
    current_worst_scale_present_class_balanced: float,
    current_worst_profile_scale_recall: float,
    current_physical_scale_prediction_agreement: float,
    current_wifi_hr_dsss_worst_scale_accuracy: float,
    current_worst_directional_confusion: float,
    current_pooled_profile_balanced: float,
    current_pooled_accuracy: float,
    combined_balanced: float,
) -> tuple[float, ...]:
    """Return the immutable v5 scale-sensitive checkpoint tuple."""
    values = (
        historical_balanced,
        current_worst_scale_present_class_balanced,
        current_worst_profile_scale_recall,
        current_physical_scale_prediction_agreement,
        current_wifi_hr_dsss_worst_scale_accuracy,
        current_worst_directional_confusion,
        current_pooled_profile_balanced,
        current_pooled_accuracy,
        combined_balanced,
    )
    if any(not np.isfinite(float(value)) for value in values):
        raise ValueError("checkpoint metrics must be finite")
    if any(float(value) < 0.0 or float(value) > 1.0 for value in values):
        raise ValueError("checkpoint metrics must lie in [0, 1]")
    directional_margin = 1.0 - float(current_worst_directional_confusion)
    bottleneck = min(
        float(historical_balanced),
        float(current_worst_scale_present_class_balanced),
        float(current_worst_profile_scale_recall),
        float(current_physical_scale_prediction_agreement),
        float(current_wifi_hr_dsss_worst_scale_accuracy),
        directional_margin,
    )
    return (
        bottleneck,
        float(current_worst_scale_present_class_balanced),
        float(current_wifi_hr_dsss_worst_scale_accuracy),
        float(current_worst_profile_scale_recall),
        float(current_physical_scale_prediction_agreement),
        directional_margin,
        float(current_pooled_profile_balanced),
        float(current_pooled_accuracy),
        float(combined_balanced),
        float(historical_balanced),
    )


def _preprocess(
    corpus: corpus_data.RawCorpus,
    row: corpus_data.RowRef,
    length: int,
    *,
    patch_length: int,
    patch_count: int,
    target_frac: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    try:
        raw = corpus.prefix(row, length)
        canonicalization: dict[str, Any] | None = None
        if isinstance(row, scale_orbit_data.ScaleOrbitRowRef):
            raw, canonicalization = geometry_canonicalizer.canonicalize(
                raw,
                sample_rate_hz=row.sample_rate_hz,
                native_sample_rate_hz=row.native_sample_rate_hz,
            )
        packed, features, context = td_preprocess.preprocess(
            raw,
            patch_length=patch_length,
            patch_count=patch_count,
            target_frac=target_frac,
        )
    except Exception as exc:
        raise ValueError(
            f"FFT-free preprocessing failed for {row.identity} "
            f"at prefix {length}"
        ) from exc
    if context.get("uses_frequency_transform") is not False:
        raise AssertionError("v5 admitted a frequency-transform frontend")
    if canonicalization is not None:
        context = {
            **context,
            "trusted_geometry_canonicalization": canonicalization,
        }
    return (
        np.asarray(packed, dtype=np.float32),
        np.asarray(features, dtype=np.float32),
        context,
    )


def _view_lengths(row: corpus_data.RowRef) -> tuple[int, ...]:
    """Return non-duplicated model views for native or scale-orbit rows."""
    if isinstance(row, scale_orbit_data.ScaleOrbitRowRef):
        if row.valid_sample_count < geometry_canonicalizer.CONSUMED_INPUT_SAMPLES:
            raise ValueError(
                f"scale-orbit row {row.identity} is shorter than the "
                "canonicalizer input"
            )
        # The v5 current route always consumes the causal first 4K before
        # physical-time canonicalization. Observation-length invariance is
        # verified separately on 4K/8K/16K prefixes.
        return (geometry_canonicalizer.CONSUMED_INPUT_SAMPLES,)
    return corpus_data.training_view_lengths(row.valid_sample_count)


def _geometry_summary(contexts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not contexts:
        return {"count": 0}
    bandwidth = np.asarray(
        [float(context["bw"]) for context in contexts],
        dtype=np.float64,
    )
    center = np.asarray(
        [float(context["center"]) for context in contexts],
        dtype=np.float64,
    )
    return {
        "count": len(contexts),
        "bandwidth_median": float(np.median(bandwidth)),
        "bandwidth_min": float(np.min(bandwidth)),
        "bandwidth_max": float(np.max(bandwidth)),
        "center_median": float(np.median(center)),
        "all_fft_free": all(
            context.get("uses_frequency_transform") is False
            for context in contexts
        ),
    }


def _stack(values: list[np.ndarray], name: str) -> np.ndarray:
    if not values:
        raise ValueError(f"cannot construct empty {name}")
    result = np.stack(values)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains a non-finite value")
    return result


def _validate_supervised_pair_pool(
    pool: Mapping[str, Sequence[SupervisedPairIdentity]],
    *,
    training_view_count: int,
    classes: Sequence[str],
) -> dict[str, Any]:
    """Validate the exact train-only population addressable by attempt 2."""
    profile_class_indices = _supervised_pair_profile_class_indices(classes)
    if (
        len(SUPERVISED_PAIR_PROFILES)
        != SUPERVISED_PAIR_EXPECTED_PROFILE_COUNT
        or len(set(SUPERVISED_PAIR_PROFILES))
        != SUPERVISED_PAIR_EXPECTED_PROFILE_COUNT
    ):
        raise RuntimeError(
            "the literal supervised-pair profile contract changed"
        )
    if (
        isinstance(training_view_count, bool)
        or not isinstance(training_view_count, (int, np.integer))
        or int(training_view_count) <= 0
    ):
        raise ValueError("training_view_count must be a positive integer")
    if set(pool) != set(SUPERVISED_PAIR_PROFILES):
        missing = sorted(set(SUPERVISED_PAIR_PROFILES) - set(pool))
        extra = sorted(set(pool) - set(SUPERVISED_PAIR_PROFILES))
        raise ValueError(
            "supervised-pair pool must contain exactly the literal 31 "
            f"profiles; missing={missing}, extra={extra}"
        )

    expected_scales = tuple(
        float(value) for value in scale_orbit_data.SCALE_ORBIT_EXPECTED_FACTORS
    )
    expected_per_profile = int(
        scale_orbit_data.SCALE_ORBIT_TRAINING_SPLIT_COUNTS["train"]
    )
    identities: list[str] = []
    positions: list[int] = []
    per_profile: dict[str, int] = {}
    class_name_by_profile: dict[str, str] = {}
    class_index_by_profile: dict[str, int] = {}
    for profile_position, profile in enumerate(SUPERVISED_PAIR_PROFILES):
        entries = tuple(pool[profile])
        if len(entries) != expected_per_profile:
            raise ValueError(
                f"supervised-pair profile {profile!r} must expose exactly "
                f"{expected_per_profile} seed20264101 train identities"
            )
        per_profile[profile] = len(entries)
        expected_class_name = SUPERVISED_PAIR_PROFILE_PUBLIC_CLASS_MAP[
            profile
        ]
        expected_class_index = profile_class_indices[profile_position]
        class_name_by_profile[profile] = expected_class_name
        class_index_by_profile[profile] = expected_class_index
        for entry in entries:
            if not isinstance(entry, SupervisedPairIdentity):
                raise TypeError(
                    "supervised-pair pool entries must be "
                    "SupervisedPairIdentity values"
                )
            if (
                entry.source != "current"
                or entry.role != "train"
                or entry.population_seed
                != scale_orbit_data.SCALE_ORBIT_TRAINING_SEED
                or entry.profile_id != profile
                or entry.public_class_name != expected_class_name
                or entry.public_class_index != expected_class_index
            ):
                raise ValueError(
                    f"supervised-pair identity {entry.identity!r} is not an "
                    "exactly mapped seed20264101 current train identity"
                )
            if (
                tuple(float(value) for value in entry.scale_factors)
                != expected_scales
                or len(entry.view_positions) != len(expected_scales)
                or len(set(entry.view_positions)) != len(expected_scales)
            ):
                raise ValueError(
                    f"supervised-pair identity {entry.identity!r} lacks "
                    "the exact four distinct physical-scale views"
                )
            if any(
                isinstance(position, bool)
                or not isinstance(position, (int, np.integer))
                or int(position) < 0
                or int(position) >= int(training_view_count)
                for position in entry.view_positions
            ):
                raise ValueError(
                    f"supervised-pair identity {entry.identity!r} has an "
                    "out-of-range training view"
                )
            identities.append(entry.identity)
            positions.extend(int(value) for value in entry.view_positions)
    if len(set(identities)) != len(identities):
        raise ValueError(
            "a supervised-pair training identity occurs in multiple profiles"
        )
    if len(set(positions)) != len(positions):
        raise ValueError(
            "a supervised-pair training view occurs in multiple identities"
        )
    expected_identity_count = (
        SUPERVISED_PAIR_EXPECTED_PROFILE_COUNT * expected_per_profile
    )
    if len(identities) != expected_identity_count:
        raise AssertionError("supervised-pair identity count changed")
    paired_target_sha256 = _supervised_pair_target_sha256(
        profile_class_indices
    )
    if paired_target_sha256 != SUPERVISED_PAIR_TARGET_SHA256:
        raise RuntimeError("supervised-pair target audit changed")
    return {
        "contract": SUPERVISED_PAIR_PAIR_CONTRACT,
        "loss_contract": SUPERVISED_PAIR_LOSS_CONTRACT,
        "source": "current",
        "population_seed": scale_orbit_data.SCALE_ORBIT_TRAINING_SEED,
        "role": "train",
        "literal_profile_count": SUPERVISED_PAIR_EXPECTED_PROFILE_COUNT,
        "literal_profiles": list(SUPERVISED_PAIR_PROFILES),
        "literal_profile_public_class_map": class_name_by_profile,
        "literal_profile_public_class_index_map": class_index_by_profile,
        "public_class_indices_in_literal_profile_order":
            list(profile_class_indices),
        "target_construction": SUPERVISED_PAIR_TARGET_CONSTRUCTION,
        "paired_targets_sha256": paired_target_sha256,
        "identities_per_profile": per_profile,
        "identity_count": len(identities),
        "scale_views_per_identity": len(expected_scales),
        "physical_scale_factors": list(expected_scales),
        "addressable_training_scale_views": len(positions),
        "merged_training_view_count": int(training_view_count),
        "identity_sha256": corpus_data.sha256_json(sorted(identities)),
        "pair_rng": (
            "separate numpy.default_rng(seed xor 0x5CA1E); episodic RNG "
            "state is never passed to supervised-pair sampling"
        ),
        "pairs_per_episode": SUPERVISED_PAIR_EXPECTED_PROFILE_COUNT,
        "views_per_episode": SUPERVISED_PAIR_EXPECTED_PROFILE_COUNT * 2,
        "fitting_firewall": {
            "seed20264101_train_identities_addressable": len(identities),
            "seed20264101_enrollment_rows_addressable": 0,
            "seed20262904_selection_rows_addressable": 0,
            "sealed_rows_addressable": 0,
        },
}


def _sample_supervised_pairs(
    pool: Mapping[str, Sequence[SupervisedPairIdentity]],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Sample one identity and two distinct scales for every frozen profile."""
    if not isinstance(rng, np.random.Generator):
        raise TypeError("supervised-pair RNG must be numpy.Generator")
    pair_positions: list[tuple[int, int]] = []
    identities: list[str] = []
    selected_scales: list[tuple[float, float]] = []
    profile_class_indices: list[int] = []
    for profile_position, profile in enumerate(SUPERVISED_PAIR_PROFILES):
        entries = tuple(pool.get(profile, ()))
        if not entries:
            raise ValueError(
                f"supervised-pair profile {profile!r} has no identities"
            )
        entry = entries[int(rng.integers(0, len(entries)))]
        expected_class_index = SUPERVISED_PAIR_PROFILE_CLASS_INDICES[
            profile_position
        ]
        if (
            entry.source != "current"
            or entry.role != "train"
            or entry.population_seed
            != scale_orbit_data.SCALE_ORBIT_TRAINING_SEED
            or entry.profile_id != profile
            or entry.public_class_name
            != SUPERVISED_PAIR_PROFILE_PUBLIC_CLASS_MAP[profile]
            or entry.public_class_index != expected_class_index
        ):
            raise ValueError(
                "supervised-pair sampler encountered an invalid mapped entry"
            )
        scale_indices = np.asarray(
            rng.choice(len(entry.scale_factors), size=2, replace=False),
            dtype=np.int64,
        )
        if (
            scale_indices.shape != (2,)
            or int(scale_indices[0]) == int(scale_indices[1])
        ):
            raise AssertionError(
                "supervised-pair sampler reused one physical scale"
            )
        pair_positions.append(
            (
                int(entry.view_positions[int(scale_indices[0])]),
                int(entry.view_positions[int(scale_indices[1])]),
            )
        )
        selected_scales.append(
            (
                float(entry.scale_factors[int(scale_indices[0])]),
                float(entry.scale_factors[int(scale_indices[1])]),
            )
        )
        identities.append(entry.identity)
        profile_class_indices.append(entry.public_class_index)
    positions_array = np.asarray(pair_positions, dtype=np.int64)
    scales_array = np.asarray(selected_scales, dtype=np.float64)
    labels_array = np.asarray(profile_class_indices, dtype=np.int64)
    if (
        positions_array.shape
        != (SUPERVISED_PAIR_EXPECTED_PROFILE_COUNT, 2)
        or scales_array.shape
        != (SUPERVISED_PAIR_EXPECTED_PROFILE_COUNT, 2)
        or labels_array.shape
        != (SUPERVISED_PAIR_EXPECTED_PROFILE_COUNT,)
        or tuple(labels_array.tolist())
        != SUPERVISED_PAIR_PROFILE_CLASS_INDICES
        or np.any(positions_array[:, 0] == positions_array[:, 1])
        or np.any(scales_array[:, 0] == scales_array[:, 1])
    ):
        raise AssertionError("supervised-pair batch changed shape or labels")
    return {
        "positions": positions_array,
        "profiles": SUPERVISED_PAIR_PROFILES,
        "identities": tuple(identities),
        "scale_factors": scales_array,
        "profile_class_indices": labels_array,
        "paired_targets_sha256":
            _supervised_pair_target_sha256(labels_array),
    }


def _combine_adaptation_losses(
    cross_entropy: torch.Tensor,
    supervised_pair_cross_entropy: torch.Tensor,
    *,
    weight: Any,
) -> torch.Tensor:
    frozen_weight = _validate_supervised_pair_weight(weight)
    if (
        cross_entropy.ndim != 0
        or supervised_pair_cross_entropy.ndim != 0
        or not torch.isfinite(cross_entropy)
        or not torch.isfinite(supervised_pair_cross_entropy)
    ):
        raise ValueError("adaptation loss components must be finite scalars")
    return (
        cross_entropy
        + frozen_weight * supervised_pair_cross_entropy
    )


@contextmanager
def _auxiliary_batch_norm_eval(net: torch.nn.Module):
    """Prevent the auxiliary forward from mutating BatchNorm running state."""
    batch_norm_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
    )
    batch_norm_modules = [
        module
        for module in net.modules()
        if isinstance(module, batch_norm_types)
    ]
    original_modes = [
        bool(module.training) for module in batch_norm_modules
    ]
    try:
        for module in batch_norm_modules:
            module.train(False)
        yield
    finally:
        for module, training in zip(batch_norm_modules, original_modes):
            module.train(training)


def _supervised_pair_cross_entropy_for_pairs(
    net: InvariantPatchCNN,
    data: Mapping[str, Any],
    pair_positions: np.ndarray,
    profile_class_indices: np.ndarray,
    prototypes: torch.Tensor,
    log_scale: torch.Tensor,
    device: torch.device,
    *,
    phase_augmentation: bool,
) -> torch.Tensor:
    """Return mean CE for 62 views against detached episode prototypes."""
    positions = np.asarray(pair_positions, dtype=np.int64)
    labels = np.asarray(profile_class_indices, dtype=np.int64)
    if positions.shape != (SUPERVISED_PAIR_EXPECTED_PROFILE_COUNT, 2):
        raise ValueError("supervised-pair positions must have shape [31,2]")
    if (
        labels.shape != (SUPERVISED_PAIR_EXPECTED_PROFILE_COUNT,)
        or tuple(labels.tolist()) != SUPERVISED_PAIR_PROFILE_CLASS_INDICES
        or _supervised_pair_target_sha256(labels)
        != SUPERVISED_PAIR_TARGET_SHA256
    ):
        raise ValueError(
            "supervised-pair labels must equal the frozen profile class map"
        )
    if np.any(positions[:, 0] == positions[:, 1]):
        raise ValueError("supervised pairs must use distinct views")
    if (
        np.any(positions < 0)
        or np.any(positions >= len(data["xtr"]))
        or len(data["xtr"]) != len(data["ftr"])
    ):
        raise ValueError(
            "supervised-pair position is outside aligned training views"
        )
    if phase_augmentation is not True:
        raise ValueError(
            "phase augmentation must remain enabled for both paired views"
        )
    if (
        prototypes.ndim != 2
        or prototypes.shape[0] != int(data["n_classes"])
        or prototypes.shape[1] <= 0
        or not torch.isfinite(prototypes).all()
    ):
        raise ValueError(
            "episode prototypes must be finite [public_class, embedding]"
        )
    if (
        log_scale.ndim != 0
        or not torch.isfinite(log_scale)
    ):
        raise ValueError("episode logit scale must be a finite scalar")
    flattened = positions.reshape(-1)
    targets = np.repeat(labels, 2)
    if (
        flattened.shape != (SUPERVISED_PAIR_EXPECTED_PROFILE_COUNT * 2,)
        or targets.shape != (SUPERVISED_PAIR_EXPECTED_PROFILE_COUNT * 2,)
        or corpus_data.sha256_json(targets.tolist())
        != SUPERVISED_PAIR_TARGET_SHA256
    ):
        raise AssertionError(
            "supervised-pair flattening or repeated targets changed"
        )
    pair_x = torch.from_numpy(
        np.asarray(data["xtr"])[flattened]
    ).to(device)
    pair_features = torch.from_numpy(
        np.asarray(data["ftr"])[flattened]
    ).to(device)
    pair_x = v3_runner._phase_augment(pair_x)
    with _auxiliary_batch_norm_eval(net):
        paired_embeddings = net(pair_x, pair_features)
    if (
        paired_embeddings.ndim != 2
        or paired_embeddings.shape
        != (SUPERVISED_PAIR_EXPECTED_PROFILE_COUNT * 2, prototypes.shape[1])
        or not torch.isfinite(paired_embeddings).all()
    ):
        raise ValueError(
            "paired embeddings must be finite [62, embedding_dim]"
        )
    # Detachment is deliberately inside this helper. The auxiliary may update
    # the encoder through paired_embeddings, but cannot update episodic support
    # embeddings/prototype construction or the shared logit-scale parameter.
    logits = (
        -sq_dist(paired_embeddings, prototypes.detach())
        * log_scale.detach().exp().clamp(1e-3, 100.0)
    )
    loss = F.cross_entropy(
        logits,
        torch.from_numpy(targets).to(device),
        label_smoothing=0.0,
        reduction="mean",
    )
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise ValueError(
            "supervised-pair cross entropy must be a finite scalar"
        )
    return loss


def _build_sampling_hierarchy(
    labels: np.ndarray,
    sources: Sequence[str],
    profiles: Sequence[str],
    n_classes: int,
) -> tuple[dict[str, dict[str, np.ndarray]], ...]:
    """Partition each base identity by public class, source, then profile."""
    truth = np.asarray(labels, dtype=np.int64)
    source_array = np.asarray(sources, dtype=object)
    profile_array = np.asarray(profiles, dtype=object)
    if (
        truth.ndim != 1
        or source_array.shape != truth.shape
        or profile_array.shape != truth.shape
    ):
        raise ValueError("sampling hierarchy inputs have inconsistent shapes")
    if n_classes <= 1 or np.any(truth < 0) or np.any(truth >= n_classes):
        raise ValueError("sampling hierarchy has invalid public-class labels")
    hierarchy: list[dict[str, dict[str, np.ndarray]]] = []
    for class_index in range(n_classes):
        class_groups: dict[str, dict[str, np.ndarray]] = {}
        class_mask = truth == class_index
        for source in sorted(str(value) for value in np.unique(
            source_array[class_mask]
        )):
            source_mask = class_mask & (source_array == source)
            profile_groups: dict[str, np.ndarray] = {}
            for profile in sorted(str(value) for value in np.unique(
                profile_array[source_mask]
            )):
                positions = np.where(
                    source_mask & (profile_array == profile)
                )[0].astype(np.int64, copy=False)
                if len(positions) == 0:
                    raise AssertionError("empty source/profile sampling group")
                profile_groups[profile] = positions
            class_groups[source] = profile_groups
        if not class_groups:
            raise ValueError(
                f"sampling hierarchy has no bases for class {class_index}"
            )
        hierarchy.append(class_groups)
    flattened = [
        int(base)
        for class_groups in hierarchy
        for profile_groups in class_groups.values()
        for positions in profile_groups.values()
        for base in positions
    ]
    if sorted(flattened) != list(range(len(truth))):
        raise AssertionError(
            "class/source/profile hierarchy does not partition base identities"
        )
    return tuple(hierarchy)


def _prepare_data(
    historical: corpus_data.RawCorpus,
    current: corpus_data.RawCorpus,
    *,
    classes: Sequence[str],
    patch_length: int,
    patch_count: int,
    target_frac: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Preprocess merged training and all eligible evaluation populations."""
    corpora = {"historical": historical, "current": current}
    n_classes = len(classes)
    if n_classes <= 1:
        raise ValueError("at least two public classes are required")
    supervised_pair_profile_class_indices = (
        _supervised_pair_profile_class_indices(classes)
    )
    for source_name, corpus in corpora.items():
        enrolled_profiles = {
            row.profile_id for row in corpus.rows_by_role["enrollment"]
        }
        required_profiles = {
            row.profile_id
            for role in corpus_data.ROLES
            for row in corpus.rows_by_role[role]
        }
        missing_profiles = sorted(required_profiles - enrolled_profiles)
        if missing_profiles:
            raise ValueError(
                f"{source_name} enrollment has no row for profiles "
                f"{missing_profiles}; every source/profile mode needs its own "
                "prototype centroid"
            )

    train_rows = (
        list(historical.rows_by_role["train"])
        + list(current.rows_by_role["train"])
    )
    train_x: list[np.ndarray] = []
    train_f_raw: list[np.ndarray] = []
    train_contexts: list[dict[str, Any]] = []
    base_view_positions: list[np.ndarray] = []
    base_labels: list[int] = []
    base_sources: list[str] = []
    base_profiles: list[str] = []
    excluded_zero_prefix: dict[str, dict[str, int]] = {
        source: {class_name: 0 for class_name in classes}
        for source in corpora
    }
    excluded_evaluation_zero_prefix = {
        source: {
            role: {class_name: 0 for class_name in classes}
            for role in ("enrollment", "selection")
        }
        for source in corpora
    }
    training_view_counts = {
        source: {str(length): 0 for length in corpus_data.RUNTIME_INPUT_LENGTHS}
        for source in corpora
    }
    admitted_base_identities: list[str] = []
    supervised_pair_pool_mutable: dict[
        str, list[SupervisedPairIdentity]
    ] = {
        profile: [] for profile in SUPERVISED_PAIR_PROFILES
    }
    # One manifest can carry several stored variants of the same physical base
    # acquisition (for example phase/length materializations sharing a
    # baseRowId).  They are views of one episode-sampling unit, not independent
    # support/query evidence.  Group them before constructing class pools.
    training_groups: dict[str, list[corpus_data.RowRef]] = {}
    for row in train_rows:
        training_groups.setdefault(row.identity, []).append(row)
    repeated_identity_groups = {
        identity: len(rows)
        for identity, rows in training_groups.items()
        if len(rows) > 1
    }
    for identity, group_rows in sorted(training_groups.items()):
        first = group_rows[0]
        if any(
            row.label != first.label
            or row.class_name != first.class_name
            or row.source != first.source
            or row.profile_id != first.profile_id
            for row in group_rows
        ):
            raise ValueError(
                f"training identity {identity!r} crosses class, source, or profile"
            )
        positions: list[int] = []
        current_scale_views: list[
            tuple[scale_orbit_data.ScaleOrbitRowRef, int]
        ] = []
        for row in group_rows:
            corpus = corpora[row.source]
            minimum = corpus.prefix(
                row, corpus_data.MINIMUM_INPUT_LENGTH
            )
            if float(np.max(np.abs(minimum))) == 0.0:
                excluded_zero_prefix[row.source][row.class_name] += 1
                continue
            for length in _view_lengths(row):
                packed, features, context = _preprocess(
                    corpus,
                    row,
                    length,
                    patch_length=patch_length,
                    patch_count=patch_count,
                    target_frac=target_frac,
                )
                view_position = len(train_x)
                positions.append(view_position)
                train_x.append(packed)
                train_f_raw.append(features)
                train_contexts.append(context)
                training_view_counts[row.source][str(length)] += 1
                if isinstance(row, scale_orbit_data.ScaleOrbitRowRef):
                    current_scale_views.append((row, view_position))
        if not positions:
            continue
        if first.source == "current":
            if not all(
                isinstance(row, scale_orbit_data.ScaleOrbitRowRef)
                for row in group_rows
            ):
                raise ValueError(
                    f"current identity {identity!r} mixes row implementations"
                )
            scale_rows = [
                row for row in group_rows
                if isinstance(row, scale_orbit_data.ScaleOrbitRowRef)
            ]
            if (
                len(scale_rows)
                != len(scale_orbit_data.SCALE_ORBIT_EXPECTED_FACTORS)
                or len(current_scale_views) != len(scale_rows)
            ):
                raise ValueError(
                    f"current train identity {identity!r} lacks one admitted "
                    "view at every physical scale"
                )
            profiles = {row.profile_id for row in scale_rows}
            pair_ids = {row.pair_id for row in scale_rows}
            realizations = {row.realization_index for row in scale_rows}
            if (
                len(profiles) != 1
                or len(pair_ids) != 1
                or len(realizations) != 1
                or any(
                    row.source != "current"
                    or row.role != "train"
                    or row.population_seed
                    != scale_orbit_data.SCALE_ORBIT_TRAINING_SEED
                    or row.identity != identity
                    for row in scale_rows
                )
            ):
                raise ValueError(
                    f"current supervised-pair identity {identity!r} crosses "
                    "source, role, seed, profile, pair, or realization"
                )
            profile = next(iter(profiles))
            if profile not in supervised_pair_pool_mutable:
                raise ValueError(
                    f"current train identity {identity!r} has unregistered "
                    f"profile {profile!r}"
                )
            expected_class_name = (
                SUPERVISED_PAIR_PROFILE_PUBLIC_CLASS_MAP[profile]
            )
            profile_position = SUPERVISED_PAIR_PROFILES.index(profile)
            expected_class_index = (
                supervised_pair_profile_class_indices[profile_position]
            )
            if (
                first.class_name != expected_class_name
                or first.label != expected_class_index
                or classes[expected_class_index] != expected_class_name
            ):
                raise ValueError(
                    f"current train profile {profile!r} has class/index "
                    f"{first.class_name!r}/{first.label}, expected "
                    f"{expected_class_name!r}/{expected_class_index}"
                )
            position_by_scale: dict[float, int] = {}
            for row, view_position in current_scale_views:
                scale = float(row.physical_scale_factor)
                if scale in position_by_scale:
                    raise ValueError(
                        f"current train identity {identity!r} repeats scale "
                        f"{scale:g}"
                    )
                position_by_scale[scale] = int(view_position)
            expected_scales = tuple(
                float(value)
                for value in scale_orbit_data.SCALE_ORBIT_EXPECTED_FACTORS
            )
            if tuple(sorted(position_by_scale)) != tuple(
                sorted(expected_scales)
            ):
                raise ValueError(
                    f"current train identity {identity!r} lacks the exact "
                    "physical-scale orbit"
                )
            supervised_pair_pool_mutable[profile].append(
                SupervisedPairIdentity(
                    identity=identity,
                    source="current",
                    role="train",
                    population_seed=
                        scale_orbit_data.SCALE_ORBIT_TRAINING_SEED,
                    profile_id=profile,
                    public_class_name=first.class_name,
                    public_class_index=int(first.label),
                    scale_factors=expected_scales,
                    view_positions=tuple(
                        position_by_scale[scale] for scale in expected_scales
                    ),
                )
            )
        base_view_positions.append(np.asarray(positions, dtype=np.int64))
        base_labels.append(first.label)
        base_sources.append(first.source)
        base_profiles.append(first.profile_id)
        admitted_base_identities.append(identity)

    xtr = _stack(train_x, "merged training patches")
    raw_ftr = _stack(train_f_raw, "merged training features")
    supervised_pair_pool = {
        profile: tuple(
            sorted(
                entries,
                key=lambda entry: entry.identity,
            )
        )
        for profile, entries in supervised_pair_pool_mutable.items()
    }
    supervised_pair_audit = _validate_supervised_pair_pool(
        supervised_pair_pool,
        training_view_count=len(xtr),
        classes=classes,
    )
    base_labels_array = np.asarray(base_labels, dtype=np.int64)
    base_by_class = [
        np.where(base_labels_array == class_index)[0].astype(
            np.int64, copy=False
        )
        for class_index in range(n_classes)
    ]
    if any(len(rows) == 0 for rows in base_by_class):
        missing = [
            classes[index]
            for index, rows in enumerate(base_by_class)
            if len(rows) == 0
        ]
        raise ValueError(f"merged training is missing classes {missing}")
    if len(base_sources) != len(base_profiles):
        raise AssertionError("base source/profile vectors lost alignment")
    sampling_hierarchy = _build_sampling_hierarchy(
        base_labels_array,
        base_sources,
        base_profiles,
        n_classes,
    )
    source_profile_pairs = list(zip(base_sources, base_profiles))

    feature_mean = (
        raw_ftr.astype(np.float64).mean(axis=0).astype(np.float32)
    )
    feature_std = (
        raw_ftr.astype(np.float64).std(axis=0) + 1e-6
    ).astype(np.float32)

    def prepare_role_views(
        source_name: str,
        role: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        corpus = corpora[source_name]
        rows = corpus.rows_by_role[role]
        buckets: dict[int, dict[str, list[Any]]] = {}
        all_contexts: list[dict[str, Any]] = []
        for row in rows:
            minimum = corpus.prefix(
                row, corpus_data.MINIMUM_INPUT_LENGTH
            )
            if float(np.max(np.abs(minimum))) == 0.0:
                excluded_evaluation_zero_prefix[source_name][role][
                    row.class_name
                ] += 1
                if source_name == "current":
                    raise ValueError(
                        f"{source_name} {role} row {row.identity} has an "
                        f"all-zero eligible prefix of "
                        f"{corpus_data.MINIMUM_INPUT_LENGTH} samples; the "
                        "current corpus generator must guarantee occupied "
                        "evaluation starts"
                    )
                # The legacy historical corpus legitimately contains idle TDD
                # and pre-burst starts.  An exact-zero minimum prefix is
                # no-signal/open-set evidence, not a modulation-class example.
                # Match the established v3 common-minimum-prefix eligibility
                # rule by omitting the complete row from every paired length.
                continue
            for length in _view_lengths(row):
                packed, features, context = _preprocess(
                    corpus,
                    row,
                    length,
                    patch_length=patch_length,
                    patch_count=patch_count,
                    target_frac=target_frac,
                )
                bucket = buckets.setdefault(
                    length,
                    {"x": [], "f_raw": [], "y": [], "rows": []},
                )
                bucket["x"].append(packed)
                bucket["f_raw"].append(features)
                bucket["y"].append(row.label)
                bucket["rows"].append(row)
                all_contexts.append(context)
        prepared_by_length: dict[int, dict[str, Any]] = {}
        for length, bucket in sorted(buckets.items()):
            packed_array = _stack(
                bucket["x"],
                f"{source_name} {role} {length} patches",
            )
            raw_feature_array = _stack(
                bucket["f_raw"],
                f"{source_name} {role} {length} features",
            )
            prepared_by_length[length] = {
                "x": packed_array,
                "f": (
                    (raw_feature_array - feature_mean) / feature_std
                ).astype(np.float32),
                "y": np.asarray(bucket["y"], dtype=np.int64),
                "rows": list(bucket["rows"]),
            }
        if not prepared_by_length:
            raise ValueError(f"{source_name} {role} has no eligible views")
        pooled = {
            key: np.concatenate(
                [prepared_by_length[length][key]
                 for length in sorted(prepared_by_length)],
                axis=0,
            )
            for key in ("x", "f", "y")
        }
        pooled["rows"] = [
            row
            for length in sorted(prepared_by_length)
            for row in prepared_by_length[length]["rows"]
        ]
        return {
            "by_length": prepared_by_length,
            "pooled": pooled,
        }, all_contexts

    h_en, h_en_contexts = prepare_role_views(
        "historical", "enrollment"
    )
    c_en, c_en_contexts = prepare_role_views(
        "current", "enrollment"
    )
    h_selection, h_sel_contexts = prepare_role_views(
        "historical", "selection"
    )
    c_selection, c_sel_contexts = prepare_role_views(
        "current", "selection"
    )
    xen = np.concatenate(
        (h_en["pooled"]["x"], c_en["pooled"]["x"]), axis=0
    )
    fen = np.concatenate(
        (h_en["pooled"]["f"], c_en["pooled"]["f"]), axis=0
    )
    yen = np.concatenate(
        (h_en["pooled"]["y"], c_en["pooled"]["y"]), axis=0
    )
    enrollment_rows = (
        h_en["pooled"]["rows"] + c_en["pooled"]["rows"]
    )
    if any(np.sum(yen == class_index) == 0 for class_index in range(n_classes)):
        raise ValueError("combined enrollment does not cover every public class")

    ftr = ((raw_ftr - feature_mean) / feature_std).astype(np.float32)
    data = {
        "classes": list(classes),
        "n_classes": n_classes,
        "xtr": xtr,
        "ftr": ftr,
        "base_view_positions": tuple(base_view_positions),
        "base_labels": base_labels_array,
        "base_by_class": base_by_class,
        "base_sampling_hierarchy": sampling_hierarchy,
        "supervised_pair_pool": supervised_pair_pool,
        "xen": xen,
        "fen": fen,
        "yen": yen,
        "enrollment_sources": np.asarray(
            [row.source for row in enrollment_rows],
            dtype=object,
        ),
        "enrollment_profiles": np.asarray(
            [row.profile_id for row in enrollment_rows],
            dtype=object,
        ),
        "enrollment_lengths": np.asarray(
            [
                length
                for population in (h_en, c_en)
                for length in sorted(population["by_length"])
                for _row in population["by_length"][length]["rows"]
            ],
            dtype=np.int64,
        ),
        "historical_selection": h_selection,
        "current_selection": c_selection,
        "fmean": feature_mean,
        "fstd": feature_std,
    }
    audit = {
        "frontend": td_preprocess.preprocess_metadata(),
        "trusted_current_geometry_canonicalizer":
            geometry_canonicalizer.canonicalizer_metadata(),
        "runtime_bucket_policy": {
            "training_prefixes": list(corpus_data.RUNTIME_INPUT_LENGTHS),
            "episode_rule": (
                SAMPLER_CONTRACT
            ),
            "evaluation_rule": (
                "historical enrollment and selection expand every eligible "
                "live-runtime bucket; trusted-current rows consume only the "
                "causal first 4,096 samples and report exact physical-scale "
                "cells after trusted-geometry canonicalization"
            ),
            "checkpoint_length_rule": (
                "historical checkpoint terms use worst per-length accuracy; "
                "current checkpoint terms use worst physical-scale and "
                "profile/scale cells plus paired-scale outcome agreement"
            ),
            "views_never_exceed_valid_sample_count": True,
        },
        "training": {
            "stored_rows_before_zero_filter": len(train_rows),
            "independent_base_identity_groups_before_zero_filter":
                len(training_groups),
            "independent_base_identity_groups_admitted":
                len(base_view_positions),
            "views": len(xtr),
            "base_rows_by_class": {
                class_name: int(len(base_by_class[index]))
                for index, class_name in enumerate(classes)
            },
            "base_rows_by_source": {
                source: int(sum(value == source for value in base_sources))
                for source in corpora
            },
            "base_rows_by_source_profile": {
                f"{source}:{profile}": int(
                    sum(
                        row_source == source and row_profile == profile
                        for row_source, row_profile in source_profile_pairs
                    )
                )
                for source, profile in sorted(
                    set(source_profile_pairs)
                )
            },
            "hierarchical_sampler_groups": {
                classes[class_index]: {
                    source: {
                        profile: int(len(positions))
                        for profile, positions in sorted(
                            profile_groups.items()
                        )
                    }
                    for source, profile_groups in sorted(
                        sampling_hierarchy[class_index].items()
                    )
                }
                for class_index in range(n_classes)
            },
            "hierarchical_sampler_contract": SAMPLER_CONTRACT,
            "supervised_pair_pool": supervised_pair_audit,
            "views_by_source_and_length": training_view_counts,
            "excluded_all_zero_4096_prefix_by_source_and_class":
                excluded_zero_prefix,
            "evaluation_all_zero_4096_prefix_policy": {
                "rule": (
                    "historical enrollment/selection rows with an exact-zero "
                    "4096-sample prefix are omitted from every paired runtime "
                    "length; current rows fail closed"
                ),
                "excluded_by_source_role_class":
                    excluded_evaluation_zero_prefix,
            },
            "admitted_base_identity_sha256": corpus_data.sha256_json(
                sorted(admitted_base_identities)
            ),
            "repeated_identity_group_audit": {
                "group_count": len(repeated_identity_groups),
                "stored_rows_in_groups": int(
                    sum(repeated_identity_groups.values())
                ),
                "maximum_stored_rows_per_identity": int(
                    max(repeated_identity_groups.values(), default=1)
                ),
                "group_sizes_sha256": corpus_data.sha256_json(
                    repeated_identity_groups
                ),
                "episode_independence_rule": (
                    "all stored rows sharing one base identity contribute "
                    "candidate views to exactly one sampling unit; support and "
                    "query sample distinct identity-group indices"
                ),
            },
        },
        "evaluation": {
            "combined_enrollment": len(yen),
            "historical_enrollment_views":
                len(h_en["pooled"]["y"]),
            "current_enrollment_views": len(c_en["pooled"]["y"]),
            "historical_enrollment_base_rows":
                len(historical.rows_by_role["enrollment"]),
            "current_enrollment_base_rows":
                len(current.rows_by_role["enrollment"]),
            "historical_selection_views":
                len(h_selection["pooled"]["y"]),
            "current_selection_views":
                len(c_selection["pooled"]["y"]),
            "historical_selection_base_rows":
                len(historical.rows_by_role["selection"]),
            "current_selection_base_rows":
                len(current.rows_by_role["selection"]),
            "enrollment_views_by_source_length": {
                source: {
                    str(length): len(population["by_length"][length]["y"])
                    for length in sorted(population["by_length"])
                }
                for source, population in (
                    ("historical", h_en),
                    ("current", c_en),
                )
            },
            "selection_views_by_source_length": {
                source: {
                    str(length): len(population["by_length"][length]["y"])
                    for length in sorted(population["by_length"])
                }
                for source, population in (
                    ("historical", h_selection),
                    ("current", c_selection),
                )
            },
            "current_selection_present_classes": [
                classes[int(index)]
                for index in np.unique(c_selection["pooled"]["y"])
            ],
            "combined_enrollment_profile_groups": sorted(
                {
                    f"{row.source}:{row.profile_id}"
                    for row in enrollment_rows
                }
            ),
            "historical_selection_profile_groups": sorted(
                {
                    f"{row.source}:{row.profile_id}"
                    for row in h_selection["pooled"]["rows"]
                }
            ),
            "current_selection_profile_groups": sorted(
                {
                    f"{row.source}:{row.profile_id}"
                    for row in c_selection["pooled"]["rows"]
                }
            ),
        },
        "geometry": {
            "training_views": _geometry_summary(train_contexts),
            "historical_enrollment": _geometry_summary(h_en_contexts),
            "current_enrollment": _geometry_summary(c_en_contexts),
            "historical_selection": _geometry_summary(h_sel_contexts),
            "current_selection": _geometry_summary(c_sel_contexts),
        },
        "feature_moments_fit": (
            "all admitted historical+current training views only"
        ),
    }
    return data, audit


def _classification_report(
    prediction: np.ndarray,
    labels: np.ndarray,
    classes: Sequence[str],
    rows: Sequence[corpus_data.RowRef] | None = None,
) -> dict[str, Any]:
    predicted = np.asarray(prediction, dtype=np.int64)
    truth = np.asarray(labels, dtype=np.int64)
    if predicted.shape != truth.shape or truth.ndim != 1 or len(truth) == 0:
        raise ValueError("classification report requires nonempty paired vectors")
    recalls: list[float | None] = []
    per_class: dict[str, dict[str, Any]] = {}
    for class_index, class_name in enumerate(classes):
        mask = truth == class_index
        count = int(np.sum(mask))
        recall = (
            float(np.mean(predicted[mask] == class_index))
            if count > 0
            else None
        )
        recalls.append(recall)
        per_class[class_name] = {"count": count, "recall": recall}
    present = [value for value in recalls if value is not None]
    if not present:
        raise AssertionError("nonempty labels produced no present class")
    report = {
        "count": len(truth),
        "accuracy": float(np.mean(predicted == truth)),
        # The all-public-class score assigns zero recall to a class absent from
        # this particular population.  The present-class score is the one used
        # in the worst-domain checkpoint floor for a subset current corpus.
        "balanced_accuracy": float(
            np.mean([0.0 if value is None else value for value in recalls])
        ),
        "present_class_balanced_accuracy": float(np.mean(present)),
        "present_class_count": len(present),
        "per_class": per_class,
    }
    if rows is None:
        return report
    if len(rows) != len(truth):
        raise ValueError("profile report rows do not align with labels")
    profile_groups: dict[str, list[int]] = {}
    profile_metadata: dict[str, tuple[str, str, int]] = {}
    for index, row in enumerate(rows):
        if int(truth[index]) != row.label:
            raise ValueError("profile report row label disagrees with truth")
        key = f"{row.source}:{row.profile_id}"
        expected = (row.source, row.class_name, row.label)
        previous = profile_metadata.get(key)
        if previous is not None and previous != expected:
            raise ValueError(f"profile report group {key!r} crosses classes")
        profile_metadata[key] = expected
        profile_groups.setdefault(key, []).append(index)
    per_profile: dict[str, dict[str, Any]] = {}
    profile_recalls: list[float] = []
    for key, positions_list in sorted(profile_groups.items()):
        positions = np.asarray(positions_list, dtype=np.int64)
        source, class_name, label = profile_metadata[key]
        recall = float(np.mean(predicted[positions] == label))
        profile_recalls.append(recall)
        per_profile[key] = {
            "source": source,
            "profile": key.split(":", 1)[1],
            "public_class": class_name,
            "count": len(positions),
            "recall": recall,
        }
    worst = min(profile_recalls)
    report.update(
        {
            "profile_count": len(per_profile),
            "profile_balanced_accuracy": float(np.mean(profile_recalls)),
            "worst_profile_recall": worst,
            "worst_profiles": [
                key
                for key, value in per_profile.items()
                if float(value["recall"]) == worst
            ],
            "per_profile": per_profile,
        }
    )
    if all(
        isinstance(row, scale_orbit_data.ScaleOrbitRowRef)
        for row in rows
    ):
        scale_groups: dict[float, list[int]] = {}
        profile_scale_groups: dict[tuple[str, float], list[int]] = {}
        pair_groups: dict[str, list[int]] = {}
        for index, row in enumerate(rows):
            assert isinstance(row, scale_orbit_data.ScaleOrbitRowRef)
            scale = float(row.physical_scale_factor)
            profile = f"{row.source}:{row.profile_id}"
            scale_groups.setdefault(scale, []).append(index)
            profile_scale_groups.setdefault((profile, scale), []).append(
                index
            )
            pair_groups.setdefault(str(row.pair_id), []).append(index)
        expected_scales = tuple(
            float(value)
            for value in scale_orbit_data.SCALE_ORBIT_EXPECTED_FACTORS
        )
        if tuple(sorted(scale_groups)) != expected_scales:
            raise ValueError(
                "scale-orbit classification lacks the exact preregistered "
                "physical-scale factors"
            )
        per_scale: dict[str, Any] = {}
        for scale, positions_list in sorted(scale_groups.items()):
            positions = np.asarray(positions_list, dtype=np.int64)
            per_scale[f"{scale:g}"] = _classification_report(
                predicted[positions],
                truth[positions],
                classes,
            )
        per_profile_scale: dict[str, Any] = {}
        for (profile, scale), positions_list in sorted(
            profile_scale_groups.items()
        ):
            positions = np.asarray(positions_list, dtype=np.int64)
            label_values = np.unique(truth[positions])
            if len(label_values) != 1:
                raise ValueError(
                    f"profile/scale cell {profile} scale {scale:g} "
                    "crosses public classes"
                )
            label = int(label_values[0])
            per_profile_scale[f"{profile}|scale={scale:g}"] = {
                "rows": len(positions),
                "public_class": classes[label],
                "recall": float(np.mean(predicted[positions] == label)),
            }

        pair_agreement: dict[str, bool] = {}
        pair_profile: dict[str, str] = {}
        for pair_id, positions_list in sorted(pair_groups.items()):
            positions = np.asarray(positions_list, dtype=np.int64)
            pair_rows = [rows[int(index)] for index in positions]
            if not all(
                isinstance(row, scale_orbit_data.ScaleOrbitRowRef)
                for row in pair_rows
            ):
                raise AssertionError("scale pair group contains a native row")
            profiles = {
                f"{row.source}:{row.profile_id}"
                for row in pair_rows
            }
            labels = {int(row.label) for row in pair_rows}
            observed_scales = tuple(
                sorted(float(row.physical_scale_factor) for row in pair_rows)
            )
            if (
                len(profiles) != 1
                or len(labels) != 1
                or observed_scales != expected_scales
            ):
                raise ValueError(
                    f"scale pair {pair_id!r} lacks exact profile/scale coverage"
                )
            pair_profile[pair_id] = next(iter(profiles))
            pair_agreement[pair_id] = bool(
                np.all(predicted[positions] == predicted[positions[0]])
            )
        per_profile_pair_agreement: dict[str, float] = {}
        for profile in sorted(set(pair_profile.values())):
            values = [
                pair_agreement[pair_id]
                for pair_id in sorted(pair_agreement)
                if pair_profile[pair_id] == profile
            ]
            if not values:
                raise AssertionError("empty scale-pair profile group")
            per_profile_pair_agreement[profile] = float(np.mean(values))

        dsss = classes.index("dsss")
        bluetooth = classes.index("bluetooth")
        dsss_mask = truth == dsss
        bluetooth_mask = truth == bluetooth
        if not np.any(dsss_mask) or not np.any(bluetooth_mask):
            raise ValueError(
                "scale-orbit selection must include DSSS and Bluetooth"
            )
        wifi_profile = "current:wifi-hr-dsss-11m"
        wifi_scale_cells = {
            key: value
            for key, value in per_profile_scale.items()
            if key.startswith(f"{wifi_profile}|scale=")
        }
        if set(
            float(key.rsplit("=", 1)[1]) for key in wifi_scale_cells
        ) != set(expected_scales):
            raise ValueError(
                "scale-orbit selection lacks exact WiFi HR-DSSS scale cells"
            )

        directional_cells: dict[str, dict[str, Any]] = {}
        for (profile, scale), positions_list in sorted(
            profile_scale_groups.items()
        ):
            positions = np.asarray(positions_list, dtype=np.int64)
            label_values = np.unique(truth[positions])
            if len(label_values) != 1:
                raise AssertionError("profile/scale directional cell crosses classes")
            label = int(label_values[0])
            if label == dsss and profile == wifi_profile:
                directional_cells[f"{profile}|scale={scale:g}"] = {
                    "truth": "dsss",
                    "predicted_class": "bluetooth",
                    "rows": len(positions),
                    "confusion_rate": float(
                        np.mean(predicted[positions] == bluetooth)
                    ),
                }
            elif label == bluetooth:
                directional_cells[f"{profile}|scale={scale:g}"] = {
                    "truth": "bluetooth",
                    "predicted_class": "dsss",
                    "rows": len(positions),
                    "confusion_rate": float(
                        np.mean(predicted[positions] == dsss)
                    ),
                }
        dsss_directional = [
            float(value["confusion_rate"])
            for value in directional_cells.values()
            if value["truth"] == "dsss"
        ]
        bluetooth_directional = [
            float(value["confusion_rate"])
            for value in directional_cells.values()
            if value["truth"] == "bluetooth"
        ]
        if (
            len(dsss_directional) != len(expected_scales)
            or not bluetooth_directional
        ):
            raise ValueError(
                "scale-orbit directional confusion cells are incomplete"
            )
        worst_dsss_directional = max(dsss_directional)
        worst_bluetooth_directional = max(bluetooth_directional)
        pair_agreement_rate = float(
            np.mean(list(pair_agreement.values()))
        )
        report.update(
            {
                "per_scale": per_scale,
                "expected_physical_scale_factors": list(expected_scales),
                "exact_scale_pair_coverage": True,
                "scale_pair_count": len(pair_groups),
                "worst_scale_present_class_balanced_accuracy": min(
                    float(value["present_class_balanced_accuracy"])
                    for value in per_scale.values()
                ),
                "worst_scale_accuracy": min(
                    float(value["accuracy"])
                    for value in per_scale.values()
                ),
                "per_profile_scale": per_profile_scale,
                "worst_profile_scale_recall": min(
                    float(value["recall"])
                    for value in per_profile_scale.values()
                ),
                "wifi_hr_dsss_worst_scale_accuracy": min(
                    float(value["recall"])
                    for value in wifi_scale_cells.values()
                ),
                "physical_scale_prediction_agreement_rate":
                    pair_agreement_rate,
                "per_profile_physical_scale_prediction_agreement_rate":
                    per_profile_pair_agreement,
                "worst_profile_physical_scale_prediction_agreement_rate": min(
                    per_profile_pair_agreement.values()
                ),
                "directional_dsss_bluetooth_cells": directional_cells,
                "dsss_to_bluetooth_confusion_rate": float(
                    np.mean(predicted[dsss_mask] == bluetooth)
                ),
                "bluetooth_to_dsss_confusion_rate": float(
                    np.mean(predicted[bluetooth_mask] == dsss)
                ),
                "wifi_hr_dsss_to_bluetooth_worst_scale_confusion_rate":
                    worst_dsss_directional,
                "bluetooth_to_dsss_worst_profile_scale_confusion_rate":
                    worst_bluetooth_directional,
                "worst_directional_dsss_bluetooth_confusion_rate": max(
                    worst_dsss_directional,
                    worst_bluetooth_directional,
                ),
            }
        )
    return report


def _profile_prototype_bank(
    embeddings: np.ndarray,
    labels: np.ndarray,
    sources: Sequence[str],
    profiles: Sequence[str],
    lengths: Sequence[int],
    classes: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Build one centroid per (source, profile), mapped to a public class."""
    value = np.asarray(embeddings, dtype=np.float32)
    truth = np.asarray(labels, dtype=np.int64)
    source_array = np.asarray(sources, dtype=object)
    profile_array = np.asarray(profiles, dtype=object)
    length_array = np.asarray(lengths, dtype=np.int64)
    if (
        value.ndim != 2
        or truth.shape != (len(value),)
        or source_array.shape != truth.shape
        or profile_array.shape != truth.shape
        or length_array.shape != truth.shape
    ):
        raise ValueError("profile prototype inputs have inconsistent shapes")
    keys = sorted(
        {
            (str(source_array[index]), str(profile_array[index]))
            for index in range(len(value))
        }
    )
    prototypes: list[np.ndarray] = []
    prototype_labels: list[int] = []
    metadata: list[dict[str, Any]] = []
    for source, profile in keys:
        mask = (source_array == source) & (profile_array == profile)
        group_labels = np.unique(truth[mask])
        if len(group_labels) != 1:
            raise ValueError(
                f"enrollment profile group {source}:{profile} crosses "
                "public classes"
            )
        label = int(group_labels[0])
        if label < 0 or label >= len(classes):
            raise ValueError("enrollment profile group has an invalid class")
        prototypes.append(value[mask].mean(axis=0))
        prototype_labels.append(label)
        metadata.append(
            {
                "source": source,
                "profile": profile,
                "public_class": classes[label],
                "enrollment_views": int(np.sum(mask)),
                "eligible_lengths": sorted(
                    int(length) for length in np.unique(length_array[mask])
                ),
            }
        )
    bank = np.stack(prototypes).astype(np.float32)
    bank_labels = np.asarray(prototype_labels, dtype=np.int64)
    missing = [
        classes[class_index]
        for class_index in range(len(classes))
        if not np.any(bank_labels == class_index)
    ]
    if missing:
        raise ValueError(
            f"profile prototype bank is missing public classes {missing}"
        )
    return bank, bank_labels, metadata


def _predict_from_profile_bank(
    embeddings: np.ndarray,
    prototypes: np.ndarray,
    prototype_labels: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict a public class by its nearest source/profile centroid."""
    value = np.asarray(embeddings, dtype=np.float32)
    bank = np.asarray(prototypes, dtype=np.float32)
    labels = np.asarray(prototype_labels, dtype=np.int64)
    if value.ndim != 2 or bank.ndim != 2 or value.shape[1] != bank.shape[1]:
        raise ValueError("embedding/prototype dimensions disagree")
    if labels.shape != (len(bank),):
        raise ValueError("prototype labels disagree with prototype rows")
    prototype_distance = (
        (value[:, None, :] - bank[None, :, :]) ** 2
    ).sum(axis=-1)
    class_distance = np.empty((len(value), n_classes), dtype=np.float32)
    for class_index in range(n_classes):
        mask = labels == class_index
        if not np.any(mask):
            raise ValueError(
                f"profile bank has no prototype for class {class_index}"
            )
        class_distance[:, class_index] = prototype_distance[:, mask].min(
            axis=1
        )
    return class_distance.argmin(axis=1), class_distance


def _evaluate(
    net: InvariantPatchCNN,
    data: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray]:
    enrollment = embed_all(
        net, data["xen"], data["fen"], device
    )
    prototypes, prototype_labels, prototype_metadata = _profile_prototype_bank(
        enrollment,
        np.asarray(data["yen"], dtype=np.int64),
        data["enrollment_sources"],
        data["enrollment_profiles"],
        data["enrollment_lengths"],
        data["classes"],
    )

    def evaluate_source(
        population: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[int, np.ndarray], dict[int, np.ndarray]]:
        reports: dict[str, Any] = {}
        predictions: dict[int, np.ndarray] = {}
        labels_by_length: dict[int, np.ndarray] = {}
        for length, bucket in sorted(population["by_length"].items()):
            embeddings = embed_all(
                net, bucket["x"], bucket["f"], device
            )
            prediction, _ = _predict_from_profile_bank(
                embeddings,
                prototypes,
                prototype_labels,
                int(data["n_classes"]),
            )
            labels = np.asarray(bucket["y"], dtype=np.int64)
            predictions[int(length)] = prediction
            labels_by_length[int(length)] = labels
            reports[str(length)] = _classification_report(
                prediction,
                labels,
                data["classes"],
                bucket["rows"],
            )
        pooled_prediction = np.concatenate(
            [predictions[length] for length in sorted(predictions)]
        )
        pooled_labels = np.concatenate(
            [labels_by_length[length] for length in sorted(labels_by_length)]
        )
        balanced = [
            float(report["balanced_accuracy"])
            for report in reports.values()
        ]
        present_balanced = [
            float(report["present_class_balanced_accuracy"])
            for report in reports.values()
        ]
        return {
            "pooled": _classification_report(
                pooled_prediction,
                pooled_labels,
                data["classes"],
                population["pooled"]["rows"],
            ),
            "per_length": reports,
            "worst_length_balanced_accuracy": min(balanced),
            "worst_length_present_class_balanced_accuracy":
                min(present_balanced),
            "worst_balanced_lengths": [
                int(length)
                for length, report in reports.items()
                if float(report["balanced_accuracy"]) == min(balanced)
            ],
            "worst_present_class_balanced_lengths": [
                int(length)
                for length, report in reports.items()
                if float(report["present_class_balanced_accuracy"])
                == min(present_balanced)
            ],
            "worst_profile_recall_across_lengths": min(
                float(report["worst_profile_recall"])
                for report in reports.values()
            ),
            "worst_profile_length_cells": [
                {
                    "length": int(length),
                    "profile": profile,
                    "recall": float(report["worst_profile_recall"]),
                }
                for length, report in reports.items()
                for profile in report["worst_profiles"]
                if float(report["worst_profile_recall"])
                == min(
                    float(candidate["worst_profile_recall"])
                    for candidate in reports.values()
                )
            ],
        }, predictions, labels_by_length

    historical, historical_prediction, historical_labels = evaluate_source(
        data["historical_selection"]
    )
    current, current_prediction, current_labels = evaluate_source(
        data["current_selection"]
    )
    required_current_scale_metrics = (
        "worst_scale_present_class_balanced_accuracy",
        "worst_profile_scale_recall",
        "wifi_hr_dsss_worst_scale_accuracy",
        "physical_scale_prediction_agreement_rate",
        "worst_directional_dsss_bluetooth_confusion_rate",
    )
    missing_current_scale_metrics = [
        key for key in required_current_scale_metrics
        if key not in current["pooled"]
    ]
    if missing_current_scale_metrics:
        raise ValueError(
            "current adaptive selection is not an exact scale-orbit "
            f"population: {missing_current_scale_metrics}"
        )
    current.update(
        {
            key: current["pooled"][key]
            for key in required_current_scale_metrics
        }
    )
    combined_prediction_by_length: dict[int, np.ndarray] = {}
    combined_labels_by_length: dict[int, np.ndarray] = {}
    combined_per_length: dict[str, Any] = {}
    combined_rows_by_length: dict[int, list[corpus_data.RowRef]] = {}
    all_lengths = sorted(
        set(historical_prediction) | set(current_prediction)
    )
    for length in all_lengths:
        predictions = [
            population[length]
            for population in (
                historical_prediction,
                current_prediction,
            )
            if length in population
        ]
        labels = [
            population[length]
            for population in (historical_labels, current_labels)
            if length in population
        ]
        combined_prediction_by_length[length] = np.concatenate(predictions)
        combined_labels_by_length[length] = np.concatenate(labels)
        combined_rows_by_length[length] = [
            row
            for population in (
                data["historical_selection"],
                data["current_selection"],
            )
            if length in population["by_length"]
            for row in population["by_length"][length]["rows"]
        ]
        combined_per_length[str(length)] = _classification_report(
            combined_prediction_by_length[length],
            combined_labels_by_length[length],
            data["classes"],
            combined_rows_by_length[length],
        )
    combined_pooled_prediction = np.concatenate(
        [
            combined_prediction_by_length[length]
            for length in all_lengths
        ]
    )
    combined_pooled_labels = np.concatenate(
        [combined_labels_by_length[length] for length in all_lengths]
    )
    combined_pooled_rows = [
        row
        for length in all_lengths
        for row in combined_rows_by_length[length]
    ]
    combined_balanced = [
        float(report["balanced_accuracy"])
        for report in combined_per_length.values()
    ]
    combined = {
        "pooled": _classification_report(
            combined_pooled_prediction,
            combined_pooled_labels,
            data["classes"],
            combined_pooled_rows,
        ),
        "per_length": combined_per_length,
        "worst_length_balanced_accuracy": min(combined_balanced),
        "worst_balanced_lengths": [
            int(length)
            for length, report in combined_per_length.items()
            if float(report["balanced_accuracy"]) == min(combined_balanced)
        ],
        "worst_profile_recall_across_lengths": min(
            float(report["worst_profile_recall"])
            for report in combined_per_length.values()
        ),
        "worst_profile_length_cells": [
            {
                "length": int(length),
                "profile": profile,
                "recall": float(report["worst_profile_recall"]),
            }
            for length, report in combined_per_length.items()
            for profile in report["worst_profiles"]
            if float(report["worst_profile_recall"])
            == min(
                float(candidate["worst_profile_recall"])
                for candidate in combined_per_length.values()
            )
        ],
    }
    score = checkpoint_score(
        historical_balanced=historical[
            "worst_length_balanced_accuracy"
        ],
        current_worst_scale_present_class_balanced=current[
            "worst_scale_present_class_balanced_accuracy"
        ],
        current_worst_profile_scale_recall=current[
            "worst_profile_scale_recall"
        ],
        current_physical_scale_prediction_agreement=current[
            "physical_scale_prediction_agreement_rate"
        ],
        current_wifi_hr_dsss_worst_scale_accuracy=current[
            "wifi_hr_dsss_worst_scale_accuracy"
        ],
        current_worst_directional_confusion=current[
            "worst_directional_dsss_bluetooth_confusion_rate"
        ],
        current_pooled_profile_balanced=current["pooled"][
            "profile_balanced_accuracy"
        ],
        current_pooled_accuracy=current["pooled"]["accuracy"],
        combined_balanced=combined["pooled"]["balanced_accuracy"],
    )
    return {
        "historical": historical,
        "current": current,
        "combined": combined,
        "prototype_bank": {
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
        },
        "checkpoint_score": list(score),
        "checkpoint_score_contract": CHECKPOINT_SCORE_CONTRACT,
    }, prototypes


def _select_hierarchical_bases(
    hierarchy: Sequence[Mapping[str, Mapping[str, np.ndarray]]],
    rng: np.random.Generator,
    *,
    per_class: int,
    current_source_share: Any = Fraction(1, 2),
    allocation_round: int = 0,
) -> list[np.ndarray]:
    """Select balanced, distinct base identities for every public class."""
    if per_class <= 0:
        raise ValueError("per_class must be positive")
    share = _source_share_fraction(current_source_share)
    if (
        isinstance(allocation_round, bool)
        or not isinstance(allocation_round, (int, np.integer))
        or int(allocation_round) < 0
    ):
        raise ValueError("allocation_round must be a non-negative integer")
    allocation_round = int(allocation_round)
    all_declared: list[int] = []
    for class_groups in hierarchy:
        for profile_groups in class_groups.values():
            for positions in profile_groups.values():
                values = np.asarray(positions, dtype=np.int64)
                if values.ndim != 1:
                    raise ValueError(
                        "sampling hierarchy base groups must be vectors"
                    )
                all_declared.extend(int(value) for value in values)
    if len(set(all_declared)) != len(all_declared):
        raise ValueError(
            "a base identity occurs in more than one sampling hierarchy group"
        )
    invalid_sources = sorted(
        {
            source
            for class_groups in hierarchy
            for source in class_groups
            if source not in {"historical", "current"}
        }
    )
    if invalid_sources:
        raise ValueError(
            f"sampling hierarchy has unsupported sources {invalid_sources}"
        )
    share_eligible_classes = [
        class_index
        for class_index, class_groups in enumerate(hierarchy)
        if {"historical", "current"}.issubset(class_groups)
    ]
    eligible_rank = {
        class_index: rank
        for rank, class_index in enumerate(share_eligible_classes)
    }

    selected_by_class: list[np.ndarray] = []
    for class_index, class_groups in enumerate(hierarchy):
        remaining: dict[tuple[str, str], list[int]] = {}
        for source, profile_groups in sorted(class_groups.items()):
            for profile, positions in sorted(profile_groups.items()):
                candidates = np.asarray(positions, dtype=np.int64)
                remaining[(source, profile)] = [
                    int(value) for value in rng.permutation(candidates)
                ]
        available_count = sum(len(values) for values in remaining.values())
        if available_count < per_class:
            raise ValueError(
                f"class {class_index} needs {per_class} distinct base "
                f"identities but only {available_count} are available"
            )

        source_usage = {source: 0 for source in class_groups}
        profile_usage = {
            (source, profile): 0
            for source, profile_groups in class_groups.items()
            for profile in profile_groups
        }
        selected: list[int] = []
        source_plan: list[str] | None = None
        if (
            share != Fraction(1, 2)
            and class_index in eligible_rank
        ):
            unit_index = (
                allocation_round * len(share_eligible_classes)
                + eligible_rank[class_index]
            )
            numerator_per_unit = per_class * share.numerator
            current_slots = (
                ((unit_index + 1) * numerator_per_unit)
                // share.denominator
                - (unit_index * numerator_per_unit)
                // share.denominator
            )
            if current_slots < 0 or current_slots > per_class:
                raise AssertionError("source-share apportionment is invalid")
            source_plan = [
                "current"
            ] * current_slots + [
                "historical"
            ] * (per_class - current_slots)
            source_plan = [
                str(value)
                for value in rng.permutation(np.asarray(source_plan, dtype=object))
            ]

        for slot in range(per_class):
            available_sources = sorted(
                source
                for source, profile_groups in class_groups.items()
                if any(
                    remaining[(source, profile)]
                    for profile in profile_groups
                )
            )
            if not available_sources:
                raise AssertionError("source allocator exhausted early")
            if source_plan is not None and source_plan[slot] in available_sources:
                source = source_plan[slot]
            else:
                # This is the legacy 1/2 rule and the deterministic spill rule
                # when a requested source lacks enough distinct identities.
                minimum_source_usage = min(
                    source_usage[source] for source in available_sources
                )
                tied_sources = [
                    source
                    for source in available_sources
                    if source_usage[source] == minimum_source_usage
                ]
                source = tied_sources[
                    int(rng.integers(0, len(tied_sources)))
                ]

            available_profiles = sorted(
                profile
                for profile in class_groups[source]
                if remaining[(source, profile)]
            )
            minimum_profile_usage = min(
                profile_usage[(source, profile)]
                for profile in available_profiles
            )
            tied_profiles = [
                profile
                for profile in available_profiles
                if profile_usage[(source, profile)]
                == minimum_profile_usage
            ]
            profile = tied_profiles[
                int(rng.integers(0, len(tied_profiles)))
            ]
            selected.append(remaining[(source, profile)].pop())
            source_usage[source] += 1
            profile_usage[(source, profile)] += 1

        if len(set(selected)) != per_class:
            raise AssertionError(
                "hierarchical sampler reused a base identity"
            )
        selected_by_class.append(np.asarray(selected, dtype=np.int64))
    return selected_by_class


def _sample_episode(
    base_sampling_hierarchy: Sequence[
        Mapping[str, Mapping[str, np.ndarray]]
    ],
    base_view_positions: Sequence[np.ndarray],
    rng: np.random.Generator,
    *,
    k_shot: int,
    q_query: int,
    current_source_share: Any = Fraction(1, 2),
    allocation_round: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    per_class = k_shot + q_query
    if k_shot <= 0 or q_query <= 0:
        raise ValueError("k_shot and q_query must be positive")
    selected_bases = _select_hierarchical_bases(
        base_sampling_hierarchy,
        rng,
        per_class=per_class,
        current_source_share=current_source_share,
        allocation_round=allocation_round,
    )
    selected_views = [
        np.asarray(
            [
                rng.choice(base_view_positions[int(base)])
                for base in class_bases
            ],
            dtype=np.int64,
        )
        for class_bases in selected_bases
    ]
    support = np.concatenate(
        [views[:k_shot] for views in selected_views]
    )
    query = np.concatenate(
        [views[k_shot:] for views in selected_views]
    )
    support_labels = np.repeat(
        np.arange(len(base_sampling_hierarchy), dtype=np.int64), k_shot
    )
    query_labels = np.repeat(
        np.arange(len(base_sampling_hierarchy), dtype=np.int64), q_query
    )
    return support, query, support_labels, query_labels


def _train(
    net: InvariantPatchCNN,
    data: dict[str, Any],
    device: torch.device,
    *,
    episodes: int,
    eval_every: int,
    seed: int,
    k_shot: int,
    q_query: int,
    lr: float,
    weight_decay: float,
    warmup_frac: float,
    label_smoothing: float,
    phase_augmentation: bool,
    current_source_share: Any,
    supervised_pair_weight: Any,
) -> tuple[InvariantPatchCNN, dict[str, Any]]:
    if episodes <= 0 or eval_every <= 0:
        raise ValueError("episodes and eval_every must be positive")
    if not 0.0 <= warmup_frac < 1.0:
        raise ValueError("warmup_frac must lie in [0, 1)")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, (int, np.integer))
        or int(seed) < 0
    ):
        raise ValueError("seed must be a non-negative integer")
    if phase_augmentation is not True:
        raise ValueError(
            "supervised-pair adaptation requires the existing phase "
            "augmentation for episodic and paired views"
        )
    supervised_pair_weight_value = _validate_supervised_pair_weight(
        supervised_pair_weight
    )
    supervised_pair_pool_audit = _validate_supervised_pair_pool(
        data["supervised_pair_pool"],
        training_view_count=len(data["xtr"]),
        classes=data["classes"],
    )
    source_share = _source_share_fraction(current_source_share)
    rng = np.random.default_rng(seed)
    pair_rng_seed = int(seed) ^ SUPERVISED_PAIR_RNG_XOR
    pair_rng = np.random.default_rng(pair_rng_seed)
    net = net.to(device)
    log_scale = torch.nn.Parameter(
        torch.tensor(math.log(10.0), dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
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
    best_score = tuple(-1.0 for _ in range(CHECKPOINT_SCORE_TERMS))
    best_state: tuple[dict[str, torch.Tensor], float, int] | None = None
    history: dict[str, Any] = {
        "loss": [],
        "evaluation": [],
        "checkpoint_score_contract": CHECKPOINT_SCORE_CONTRACT,
        "supervised_pair": {
            "loss_contract": SUPERVISED_PAIR_LOSS_CONTRACT,
            "pair_sampling_contract": SUPERVISED_PAIR_PAIR_CONTRACT,
            "weight": supervised_pair_weight_value,
            "auxiliary": "public_class_cross_entropy",
            "reduction": "mean_over_62_profile_contiguous_views",
            "targets": SUPERVISED_PAIR_TARGET_CONSTRUCTION,
            "paired_targets_sha256": SUPERVISED_PAIR_TARGET_SHA256,
            "prototypes": (
                "detached_current_episode_source_mixed_public_class_"
                "prototypes"
            ),
            "logit_scale":
                "detached_numeric_current_episode_logit_scale",
            "pair_rng_seed": pair_rng_seed,
            "pair_rng_seed_rule": "seed xor 0x5CA1E",
            "pair_rng_separate_from_episodic_rng": True,
            "phase_augmentation":
                "one_independent_draw_per_paired_view",
            "batch_norm_stat_firewall": True,
            "pool": supervised_pair_pool_audit,
        },
    }
    running_loss = 0.0
    running_cross_entropy = 0.0
    running_supervised_pair_cross_entropy = 0.0
    started = time.perf_counter()
    for episode in range(episodes):
        net.train()
        if episode < warmup:
            for group in optimizer.param_groups:
                group["lr"] = lr * (episode + 1) / warmup
        support, query, support_labels, query_labels = _sample_episode(
            data["base_sampling_hierarchy"],
            data["base_view_positions"],
            rng,
            k_shot=k_shot,
            q_query=q_query,
            current_source_share=source_share,
            allocation_round=episode,
        )
        positions = np.concatenate((support, query))
        x = torch.from_numpy(np.asarray(data["xtr"])[positions]).to(device)
        features = torch.from_numpy(
            np.asarray(data["ftr"])[positions]
        ).to(device)
        if phase_augmentation:
            x = v3_runner._phase_augment(x)
        embeddings = net(x, features)
        support_embeddings = embeddings[: len(support)]
        query_embeddings = embeddings[len(support) :]
        support_labels_t = torch.from_numpy(support_labels).to(device)
        prototypes = torch.stack(
            [
                support_embeddings[support_labels_t == class_index].mean(0)
                for class_index in range(data["n_classes"])
            ]
        )
        logits = (
            -sq_dist(query_embeddings, prototypes)
            * log_scale.exp().clamp(1e-3, 100.0)
        )
        cross_entropy = F.cross_entropy(
            logits,
            torch.from_numpy(query_labels).to(device),
            label_smoothing=label_smoothing,
        )
        paired = _sample_supervised_pairs(
            data["supervised_pair_pool"],
            pair_rng,
        )
        supervised_pair_cross_entropy = (
            _supervised_pair_cross_entropy_for_pairs(
                net,
                data,
                np.asarray(paired["positions"], dtype=np.int64),
                np.asarray(
                    paired["profile_class_indices"],
                    dtype=np.int64,
                ),
                prototypes,
                log_scale,
                device,
                phase_augmentation=phase_augmentation,
            )
        )
        loss = _combine_adaptation_losses(
            cross_entropy,
            supervised_pair_cross_entropy,
            weight=supervised_pair_weight_value,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if episode >= warmup:
            scheduler.step()
        running_loss += float(loss.detach().cpu())
        running_cross_entropy += float(cross_entropy.detach().cpu())
        running_supervised_pair_cross_entropy += float(
            supervised_pair_cross_entropy.detach().cpu()
        )

        log_every = max(1, min(100, episodes))
        if (episode + 1) % log_every == 0:
            elapsed = time.perf_counter() - started
            eta_minutes = (
                elapsed / (episode + 1) * (episodes - episode - 1) / 60.0
            )
            mean_loss = running_loss / log_every
            mean_cross_entropy = running_cross_entropy / log_every
            mean_supervised_pair_cross_entropy = (
                running_supervised_pair_cross_entropy / log_every
            )
            running_loss = 0.0
            running_cross_entropy = 0.0
            running_supervised_pair_cross_entropy = 0.0
            history["loss"].append(
                {
                    "episode": episode + 1,
                    "value": mean_loss,
                    "cross_entropy": mean_cross_entropy,
                    "supervised_pair_cross_entropy":
                        mean_supervised_pair_cross_entropy,
                    "weighted_supervised_pair_cross_entropy": (
                        supervised_pair_weight_value
                        * mean_supervised_pair_cross_entropy
                    ),
                }
            )
            print(
                f"[v5/{net.cfg.encoder}] ep {episode + 1}/{episodes} "
                f"loss={mean_loss:.4f} ce={mean_cross_entropy:.4f} "
                f"supervised-pair-ce="
                f"{mean_supervised_pair_cross_entropy:.4f} "
                f"eta={eta_minutes:.1f}m",
                flush=True,
            )

        if (episode + 1) % eval_every == 0 or episode + 1 == episodes:
            reports, _ = _evaluate(net, data, device)
            score = tuple(float(value) for value in reports["checkpoint_score"])
            if len(score) != CHECKPOINT_SCORE_TERMS:
                raise AssertionError("v5 checkpoint score changed dimensionality")
            improved = score > best_score
            if improved:
                best_score = score
                best_state = (
                    copy.deepcopy(net.state_dict()),
                    float(log_scale.detach().cpu()),
                    episode + 1,
                )
            history["evaluation"].append(
                {
                    "episode": episode + 1,
                    **reports,
                    "selected_as_best": improved,
                }
            )
            print(
                f"[v5/{net.cfg.encoder}] eval ep={episode + 1} "
                f"hist-worst="
                f"{reports['historical']['worst_length_balanced_accuracy']:.4f} "
                f"current-scale-worst="
                f"{reports['current']['worst_scale_present_class_balanced_accuracy']:.4f} "
                f"current-profile-scale-worst="
                f"{reports['current']['worst_profile_scale_recall']:.4f} "
                f"current-pooled="
                f"{reports['current']['pooled']['accuracy']:.4f} "
                f"score={score}{' <- best' if improved else ''}",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("training completed without an evaluated checkpoint")
    net.load_state_dict(best_state[0], strict=True)
    history.update(
        {
            "episodes": episodes,
            "best_episode": best_state[2],
            "best_checkpoint_score": list(best_score),
            "final_logit_scale": float(math.exp(best_state[1])),
            "wall_clock_s": time.perf_counter() - started,
            "sampler_contract": SAMPLER_CONTRACT,
            "current_source_share": {
                "numerator": source_share.numerator,
                "denominator": source_share.denominator,
                "value": float(source_share),
            },
        }
    )
    return net.eval(), history


def _run_configuration(
    args: argparse.Namespace,
    *,
    device: torch.device,
    adaptation_binding: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "v5-scale-orbit-development-training-v2",
        "arguments": {
            key: _jsonable(value)
            for key, value in sorted(vars(args).items())
            if key != "output_dir"
        },
        "checkpoint_selection": {
            "contract": CHECKPOINT_SCORE_CONTRACT,
            "comparison": "strict Python tuple lexicographic greater-than",
            "predeclared": True,
            "tunable": False,
        },
        "adaptation": {
            **dict(adaptation_binding),
            "loss_contract": SUPERVISED_PAIR_LOSS_CONTRACT,
            "pair_sampling_contract": SUPERVISED_PAIR_PAIR_CONTRACT,
            "supervised_pair_weight": float(
                args.supervised_pair_weight
            ),
            "auxiliary": "public_class_cross_entropy",
            "pairs_per_episode":
                SUPERVISED_PAIR_EXPECTED_PROFILE_COUNT,
            "views_per_episode":
                SUPERVISED_PAIR_EXPECTED_PROFILE_COUNT * 2,
            "target_construction":
                SUPERVISED_PAIR_TARGET_CONSTRUCTION,
            "prototype_gradient": "detached_for_auxiliary",
            "logit_scale_gradient": "detached_for_auxiliary",
            "phase_augmentation":
                "one_independent_draw_per_paired_view",
            "batch_norm_stat_firewall": True,
            "fitting_firewall": {
                "pair_source": "seed20264101 current train role only",
                (
                    "supervised_pairs_from_seed20264101_current_train_"
                    "role_only"
                ): True,
                (
                    "seed20264101_enrollment_rows_used_for_"
                    "supervised_pair_auxiliary"
                ): 0,
                (
                    "seed20262904_selection_rows_used_for_any_gradient_or_"
                    "optimizer_step"
                ): 0,
                (
                    "seed20262904_selection_rows_used_for_weight_center_or_"
                    "feature_moment_fit"
                ): 0,
                (
                    "seed20262904_selection_rows_used_for_persistent_"
                    "prototype_fit"
                ): 0,
                (
                    "seed20262904_selection_rows_used_for_open_set_"
                    "threshold_or_rank_fit"
                ): 0,
                (
                    "seed20262904_selection_metrics_may_retain_the_existing_"
                    "development_checkpoint_and_candidate_selection_role"
                ): True,
                (
                    "sealed_rows_used_for_any_fit_checkpoint_or_candidate_"
                    "selection"
                ): 0,
            },
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": float(args.lr),
            "network_weight_decay": float(args.weight_decay),
            "logit_scale_weight_decay": 0.0,
        },
        "scheduler": {
            "name": "linear warmup then cosine annealing",
            "warmup_fraction": float(args.warmup_frac),
        },
        "randomness": {
            "seed": int(args.seed),
            "phase_augmentation": bool(args.phase_augmentation),
            "episodic_rng": "numpy.default_rng(seed)",
            "supervised_pair_rng": {
                "kind": "numpy.default_rng",
                "seed": int(args.seed) ^ SUPERVISED_PAIR_RNG_XOR,
                "seed_rule": "seed xor 0x5CA1E",
                "separate_from_episodic_rng": True,
            },
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "requested_device": args.device,
            "resolved_device": str(device),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    adaptation_binding = _validate_supervised_pair_adaptation_source()
    _validate_adaptation_run_configuration(args)
    source_hashes_at_start = _source_hashes()
    required_adaptation_sources = {
        "v5/scale_consistency_adaptation_attempt_1.json":
            ATTEMPT_1_ADAPTATION_SHA256,
        "v5/scale_consistency_adaptation_attempt_1_miss_report.json":
            ATTEMPT_1_MISS_REPORT_SHA256,
        "v5/scale_consistency_adaptation_attempt_2.json":
            SUPERVISED_PAIR_ADAPTATION_SHA256,
    }
    if any(
        source_hashes_at_start.get(key) != expected
        for key, expected in required_adaptation_sources.items()
    ):
        raise RuntimeError(
            "executed source snapshot omitted or changed the exact "
            "attempt-1 intent, miss report, or attempt-2 intent"
        )
    output = Path(args.output_dir).expanduser().resolve()
    lowered = {part.lower() for part in output.parts}
    if "releases" in lowered or any("sealed" in part for part in lowered):
        raise ValueError("v5 development trainer refuses release/sealed outputs")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    historical_directory = Path(args.historical_corpus).expanduser().resolve()
    current_directory = Path(args.current_corpus).expanduser().resolve()
    selection_directory = Path(
        args.current_selection_corpus
    ).expanduser().resolve()
    if historical_directory == current_directory:
        raise ValueError("current corpus must be separate from historical corpus")
    if (
        selection_directory == current_directory
        or selection_directory == historical_directory
    ):
        raise ValueError(
            "adaptive selection corpus must be separate from training and "
            "historical corpora"
        )
    output.mkdir(parents=True, exist_ok=True)

    historical, historical_contract = corpus_data.load_historical_exposed(
        historical_directory,
        patch_length=args.patch_length,
        patch_count=args.patch_count,
        target_frac=args.target_frac,
        seed=args.seed,
    )
    classes = list(historical_contract["classes"])
    current = scale_orbit_data.load_scale_orbit_training_corpus(
        current_directory,
        selection_directory=selection_directory,
        class_index={name: index for index, name in enumerate(classes)},
    )
    data, preprocessing_audit = _prepare_data(
        historical,
        current,
        classes=classes,
        patch_length=args.patch_length,
        patch_count=args.patch_count,
        target_frac=args.target_frac,
    )
    if historical.audit.get("consumed_test_rows_exposed") != 0:
        raise AssertionError("consumed historical test rows reached v5")

    device = v3_runner.resolve_device(args.device)
    v3_runner.seed_everything(args.seed)
    configuration = InvariantPatchConfig(
        patch_length=args.patch_length,
        patch_count=args.patch_count,
        encoder=args.encoder,
        patch_dim=args.patch_dim,
        hidden=args.hidden,
        set_pool=args.set_pool,
        dropout=args.dropout,
    )
    net = InvariantPatchCNN(configuration)
    net, training = _train(
        net,
        data,
        device,
        episodes=args.episodes,
        eval_every=args.eval_every,
        seed=args.seed,
        k_shot=args.k_shot,
        q_query=args.q_query,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_frac=args.warmup_frac,
        label_smoothing=args.label_smoothing,
        phase_augmentation=args.phase_augmentation,
        current_source_share=args.current_source_share,
        supervised_pair_weight=args.supervised_pair_weight,
    )
    final_evaluation, combined_prototypes = _evaluate(net, data, device)

    state_path = output / f"{args.encoder}_state_dict.pt"
    prototypes_path = output / "combined_prototypes.npy"
    prototype_contract_path = output / "combined_prototype_bank.json"
    moments_path = output / "feature_moments.npz"
    torch.save(net.state_dict(), state_path)
    np.save(prototypes_path, combined_prototypes)
    _write_json(
        prototype_contract_path,
        final_evaluation["prototype_bank"],
    )
    np.savez(
        moments_path,
        mean=np.asarray(data["fmean"], dtype=np.float32),
        std=np.asarray(data["fstd"], dtype=np.float32),
    )

    _assert_source_snapshot_unchanged(
        source_hashes_at_start,
        _source_hashes(),
        operation="v5 scale-orbit branch training",
    )
    result = {
        "status": "complete",
        "development_only": True,
        "release_evidence": False,
        "consumed_historical_test_rows_used": 0,
        "encoder": args.encoder,
        "architecture": net.config(),
        "parameter_count": int(sum(parameter.numel() for parameter in net.parameters())),
        "run_configuration": _run_configuration(
            args,
            device=device,
            adaptation_binding=adaptation_binding,
        ),
        "training": training,
        "final_evaluation": final_evaluation,
        "data_audit": {
            "historical": historical.audit,
            "current": current.audit,
            "preprocessing": preprocessing_audit,
            "historical_contract": historical_contract,
            "consumed_test_rows_exposed": 0,
        },
        "artifacts": {
            "state_dict": state_path.name,
            "state_dict_sha256": corpus_data.sha256_file(state_path),
            "combined_prototypes": prototypes_path.name,
            "combined_prototypes_sha256":
                corpus_data.sha256_file(prototypes_path),
            "combined_prototype_bank": prototype_contract_path.name,
            "combined_prototype_bank_sha256":
                corpus_data.sha256_file(prototype_contract_path),
            "feature_moments": moments_path.name,
            "feature_moments_sha256": corpus_data.sha256_file(moments_path),
        },
        "source_sha256": source_hashes_at_start,
        "device": str(device),
        "wall_clock_s": time.perf_counter() - started,
    }
    metrics_path = output / "dev_metrics.json"
    _write_json(metrics_path, result)
    print(f"[v5 scale-orbit] wrote {output}", flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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
    parser.add_argument("--seed", type=int, default=20260740)
    parser.add_argument(
        "--encoder", choices=("real", "complex"), default="real"
    )
    parser.add_argument("--episodes", type=int, default=8_000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--q-query", type=int, default=5)
    parser.add_argument("--patch-length", type=int, default=64)
    parser.add_argument("--patch-count", type=int, default=16)
    parser.add_argument("--target-frac", type=float, default=0.5)
    parser.add_argument("--patch-dim", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument(
        "--set-pool", choices=("mean", "mean_std"), default="mean_std"
    )
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-frac", type=float, default=0.03)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--current-source-share",
        type=_parse_source_share,
        default=Fraction(1, 3),
        help=(
            "Exact current-source share for classes present in both sources "
            "(supervised-pair attempt 2 is frozen at 1/3)"
        ),
    )
    parser.add_argument(
        "--supervised-pair-weight",
        type=float,
        required=True,
        help=(
            "Required explicit supervised-pair loss weight; attempt 2 is "
            "frozen at exactly 0.2"
        ),
    )
    parser.add_argument(
        "--phase-augmentation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
