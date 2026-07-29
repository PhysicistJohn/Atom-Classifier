"""Validate attempt-1 miss evidence and the frozen attempt-2 intent.

The validator reads only the four append-only contract documents and the six
explicitly bound development ``dev_metrics.json`` files.  It never opens a
corpus, checkpoint, prototype bank, or sealed-validation artifact.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PROTOCOL_PATH = HERE / "recovery_protocol.json"
ATTEMPT_1_PATH = HERE / "scale_consistency_adaptation_attempt_1.json"
MISS_REPORT_PATH = (
    HERE / "scale_consistency_adaptation_attempt_1_miss_report.json"
)
ATTEMPT_2_PATH = HERE / "scale_consistency_adaptation_attempt_2.json"

EXPECTED_PROTOCOL_SHA256 = (
    "cc79496ba398e4ce2fcb6875d8b810bcbc1ae7f89cb75271fdf2c65e6ad2afff"
)
EXPECTED_ATTEMPT_1_SHA256 = (
    "e3166235ed53835465b3bd7afa6ad03982e31016b4d20c6875879288b3304a90"
)
EXPECTED_MISS_REPORT_SHA256 = (
    "1c0e6d94d418e3954a54aa44396ef5cc9d8603eb2895dd69c0a0c458c254bc9d"
)
EXPECTED_ATTEMPT_2_SHA256 = (
    "5e0cc7bbd58ebe050bf983f854e154c43595dc804aedee3c1c3849cf5691b0a9"
)
EXPECTED_PARENT_COMMIT = "2d6dbd2ba71286233dee660605e095122eb7c26a"
EXPECTED_PARENT_SUBJECT = "Train v5 with frozen physical-scale consistency"

METRIC_BINDINGS = {
    "baseline_real": (
        ".artifacts/v5-scale-orbit-real-seed20260740-e8000-reg/"
        "dev_metrics.json",
        "80c653923dfabdf4d516c499562969a9c402f939e819e9b893106053161acf31",
    ),
    "baseline_complex": (
        ".artifacts/v5-scale-orbit-complex-seed20260740-e8000-reg/"
        "dev_metrics.json",
        "a3b15398fa5fe145996eeb4f273e0d566decddb55bc79b622ae3c019f2c23043",
    ),
    "baseline_fusion": (
        ".artifacts/v5-scale-orbit-fusion-w0.5/dev_metrics.json",
        "1c02705081cb4f973e1ea092a0b995501ca9c84e609f09d7931f2337462b28b5",
    ),
    "attempt_1_real": (
        ".artifacts/v5-scale-orbit-real-seed20260740-e8000-reg-sc02/"
        "dev_metrics.json",
        "16909f12bd6ad193718938f8cbdfefa046bb8b8e8420a4a82b8ae2bf0dc52580",
    ),
    "attempt_1_complex": (
        ".artifacts/v5-scale-orbit-complex-seed20260740-e8000-reg-sc02/"
        "dev_metrics.json",
        "3f9d56fbda775bcabdb1ae11f38231d94352ec69ed7e0a7b965fa3e23aecda5e",
    ),
    "attempt_1_fusion": (
        ".artifacts/v5-scale-orbit-fusion-w0.5-sc02/dev_metrics.json",
        "d23c8717261961dec99a07cc45f0476cdc784fdacaf827742ee1677275186e7b",
    ),
}
METRIC_PATHS = {
    name: REPO / relative
    for name, (relative, _sha256_value) in METRIC_BINDINGS.items()
}
CONTRACT_PATHS = {
    PROTOCOL_PATH,
    ATTEMPT_1_PATH,
    MISS_REPORT_PATH,
    ATTEMPT_2_PATH,
}
ALLOWED_PATHS = CONTRACT_PATHS | set(METRIC_PATHS.values())

LITERAL_PROFILES = [
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
]
PROFILE_PUBLIC_CLASS_MAP = {
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if path not in {candidate.resolve() for candidate in ALLOWED_PATHS}:
        raise ValueError(f"refusing undeclared evidence path: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def _metric_value(document: Mapping[str, Any], dotted_path: str) -> Any:
    value: Any = document
    for component in dotted_path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise ValueError(f"missing metric path {dotted_path!r}")
        value = value[component]
    return value


def _expected_parent_bindings(*, include_miss: bool) -> dict[str, Any]:
    bindings: dict[str, Any] = {
        "recovery_protocol_relative_path": str(
            PROTOCOL_PATH.relative_to(REPO)
        ),
        "recovery_protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "attempt_1_intent_relative_path": str(
            ATTEMPT_1_PATH.relative_to(REPO)
        ),
        "attempt_1_intent_sha256": EXPECTED_ATTEMPT_1_SHA256,
    }
    if include_miss:
        bindings.update(
            {
                "attempt_1_miss_report_relative_path": str(
                    MISS_REPORT_PATH.relative_to(REPO)
                ),
                "attempt_1_miss_report_sha256":
                    EXPECTED_MISS_REPORT_SHA256,
            }
        )
    return bindings


def _expected_candidates() -> dict[str, Any]:
    def bound(name: str) -> dict[str, str]:
        relative, digest = METRIC_BINDINGS[name]
        return {
            "dev_metrics_relative_path": relative,
            "dev_metrics_sha256": digest,
        }

    return {
        "baseline": {
            "description":
                "frozen pre-adaptation v5 scale-orbit development candidate",
            "adaptation": "none",
            "real_branch": bound("baseline_real"),
            "complex_branch": bound("baseline_complex"),
            "fusion": bound("baseline_fusion"),
        },
        "attempt_1_sc02": {
            "description":
                "attempt-1 cosine-distance auxiliary with frozen coefficient 0.2",
            "adaptation": "one_minus_cosine_similarity",
            "attempt_1_intent_sha256": EXPECTED_ATTEMPT_1_SHA256,
            "real_branch": bound("attempt_1_real"),
            "complex_branch": bound("attempt_1_complex"),
            "fusion": bound("attempt_1_fusion"),
        },
    }


def _validate_metric_bindings(
    metrics: Mapping[str, Mapping[str, Any]],
    observed_hashes: Mapping[str, str],
) -> None:
    if set(metrics) != set(METRIC_BINDINGS):
        raise ValueError("development metric evidence set changed")
    if set(observed_hashes) != set(METRIC_BINDINGS):
        raise ValueError("development metric hash evidence set changed")
    for name, (_relative, expected_sha256) in METRIC_BINDINGS.items():
        if observed_hashes.get(name) != expected_sha256:
            raise ValueError(f"{name} raw dev_metrics SHA-256 changed")
        metric = metrics[name]
        if (
            metric.get("status") != "complete"
            or metric.get("development_only") is not True
            or metric.get("release_evidence") is not False
        ):
            raise ValueError(f"{name} is not completed development evidence")

    for prefix in ("baseline", "attempt_1"):
        fusion = metrics[f"{prefix}_fusion"]
        source_branches = fusion.get("source_branches")
        if not isinstance(source_branches, Mapping):
            raise ValueError(f"{prefix} fusion lacks branch bindings")
        for encoder in ("real", "complex"):
            branch = source_branches.get(encoder)
            hashes = (
                branch.get("hashes")
                if isinstance(branch, Mapping)
                else None
            )
            expected = METRIC_BINDINGS[f"{prefix}_{encoder}"][1]
            if (
                not isinstance(hashes, Mapping)
                or hashes.get("dev_metrics.json") != expected
            ):
                raise ValueError(
                    f"{prefix} fusion does not bind its {encoder} branch"
                )

    for encoder in ("real", "complex"):
        baseline = metrics[f"baseline_{encoder}"]
        attempt_1 = metrics[f"attempt_1_{encoder}"]
        if "adaptation" in baseline.get("run_configuration", {}):
            raise ValueError("baseline branch unexpectedly has adaptation")
        adaptation = attempt_1.get("run_configuration", {}).get(
            "adaptation"
        )
        if (
            not isinstance(adaptation, Mapping)
            or adaptation.get("adaptation_attempt") != 1
            or adaptation.get("distance") != "one_minus_cosine_similarity"
            or adaptation.get("scale_consistency_weight") != 0.2
            or adaptation.get("sha256") != EXPECTED_ATTEMPT_1_SHA256
        ):
            raise ValueError("attempt-1 branch adaptation binding changed")


def _frozen_gate(
    protocol: Mapping[str, Any],
    name: str,
) -> Mapping[str, Any]:
    gates = protocol.get("gate_contract", {}).get("inherited_gates")
    if not isinstance(gates, list):
        raise ValueError("recovery protocol inherited gates are missing")
    matches = [
        item for item in gates
        if isinstance(item, Mapping) and item.get("name") == name
    ]
    if len(matches) != 1:
        raise ValueError(f"frozen gate {name!r} is not unique")
    return matches[0]


def _validate_miss_report(
    report: Mapping[str, Any],
    attempt_1: Mapping[str, Any],
    protocol: Mapping[str, Any],
    metrics: Mapping[str, Mapping[str, Any]],
) -> None:
    expected_header = {
        "schema": "atomos.v5.scale-orbit.adaptation-attempt-miss-report",
        "schema_version": 1,
        "status": (
            "adaptation_attempt_1_did_not_recover_two_predeclared_"
            "recovery_goals"
        ),
        "authored_at": "2026-07-29T06:37:21-07:00",
        "lineage": "v5-scale-orbit",
        "development_only": True,
        "release_evidence": False,
        "append_only": True,
        "adaptation_attempt": 1,
        "parent_source_commit": EXPECTED_PARENT_COMMIT,
        "parent_source_commit_subject": EXPECTED_PARENT_SUBJECT,
    }
    for key, value in expected_header.items():
        if report.get(key) != value:
            raise ValueError(f"attempt-1 miss report changed {key}")
    if report.get("parent_bindings") != _expected_parent_bindings(
        include_miss=False
    ):
        raise ValueError("attempt-1 miss report parent bindings changed")
    if report.get("evaluated_candidates") != _expected_candidates():
        raise ValueError("attempt-1 evaluated candidate bindings changed")
    if (
        attempt_1.get("adaptation_attempt") != 1
        or attempt_1.get("failed_frozen_goals") is None
        or protocol.get("lineage", {}).get("id") != "v5-scale-orbit"
    ):
        raise ValueError("attempt-1 or protocol lineage changed")

    expected_goals = [
        (
            "classifier_development_current_service_pooled_closed_accuracy",
            "final_evaluation.current.pooled.accuracy",
            0.940398185483871,
            0.9343497983870968,
        ),
        (
            "length_scale_outcome_agreement_every_source_profile",
            (
                "final_evaluation.current.pooled."
                "worst_profile_physical_scale_prediction_agreement_rate"
            ),
            0.875,
            0.875,
        ),
    ]
    reported = report.get(
        "reassessed_attempt_1_predeclared_failed_goals"
    )
    if not isinstance(reported, list) or len(reported) != len(expected_goals):
        raise ValueError(
            "attempt-1 reassessed predeclared goal set changed"
        )
    attempt_1_goals = {
        item.get("name"): item
        for item in attempt_1.get("failed_frozen_goals", [])
        if isinstance(item, Mapping)
    }
    for row, (name, path, baseline_value, attempt_value) in zip(
        reported, expected_goals
    ):
        gate = _frozen_gate(protocol, name)
        intent_gate = attempt_1_goals.get(name)
        expected = {
            "name": name,
            "comparison": "greater_than_or_equal",
            "threshold": 0.95,
            "fusion_metric_path": path,
            "baseline_observed": baseline_value,
            "baseline_pass": False,
            "attempt_1_sc02_observed": attempt_value,
            "attempt_1_sc02_pass": False,
        }
        if row != expected:
            raise ValueError(
                f"miss report reassessed predeclared goal {name!r} changed"
            )
        if (
            gate.get("comparison") != "greater_than_or_equal"
            or gate.get("threshold") != 0.95
            or not isinstance(intent_gate, Mapping)
            or intent_gate.get("comparison") != gate.get("comparison")
            or intent_gate.get("threshold") != gate.get("threshold")
        ):
            raise ValueError(f"goal {name!r} no longer matches its parents")
        actual_baseline = _metric_value(metrics["baseline_fusion"], path)
        actual_attempt = _metric_value(metrics["attempt_1_fusion"], path)
        if (
            actual_baseline != baseline_value
            or actual_attempt != attempt_value
            or actual_baseline >= 0.95
            or actual_attempt >= 0.95
        ):
            raise ValueError(f"goal {name!r} miss is not reproduced")

    expected_diagnostics = [
        {
            "name": "current_worst_profile_scale_recall",
            "fusion_metric_path": (
                "final_evaluation.current.pooled."
                "worst_profile_scale_recall"
            ),
            "baseline_observed": 0.796875,
            "attempt_1_sc02_observed": 0.78125,
            "direction": "decreased",
            "related_but_not_equivalent_gate_name": (
                "classifier_development_current_service_every_legal_cell_"
                "closed_accuracy"
            ),
            "related_gate_threshold": 0.9,
            "formal_full_gate_result": False,
            "reason": (
                "This pooled worst-profile-scale diagnostic is not the exact "
                "legal observation inventory required by the related gate."
            ),
        }
    ]
    if report.get("diagnostic_disclosures") != expected_diagnostics:
        raise ValueError("attempt-1 diagnostic disclosure changed")
    diagnostic = expected_diagnostics[0]
    path = str(diagnostic["fusion_metric_path"])
    if (
        _metric_value(metrics["baseline_fusion"], path) != 0.796875
        or _metric_value(metrics["attempt_1_fusion"], path) != 0.78125
    ):
        raise ValueError("attempt-1 diagnostic is not reproduced")
    related_gate = _frozen_gate(
        protocol,
        str(diagnostic["related_but_not_equivalent_gate_name"]),
    )
    if (
        related_gate.get("comparison") != "greater_than_or_equal"
        or related_gate.get("threshold") != 0.9
    ):
        raise ValueError("related diagnostic gate binding changed")

    expected_full_gate_assessment = {
        "full_21_gate_status": "not_evaluated",
        "reason": (
            "No open-set policy/abstention/exact legal inventory runner was "
            "executed for these development artifacts."
        ),
        "inherited_gate_count": 19,
        "added_directional_confusion_gate_count": 2,
        "other_gate_results_assessed": False,
        "other_gates_claimed_passed": False,
    }
    if report.get("full_gate_assessment") != expected_full_gate_assessment:
        raise ValueError("full 21-gate non-evaluation boundary changed")

    expected_outcome = {
        "reassessed_predeclared_recovery_goal_count": 2,
        (
            "attempt_1_recovered_reassessed_predeclared_recovery_goal_"
            "count"
        ): 0,
        (
            "neither_of_two_reassessed_predeclared_recovery_goals_"
            "recovered_by_attempt_1"
        ): True,
        "attempt_1_selected_for_continuation": False,
        "attempt_1_is_independent_validation_evidence": False,
        "attempt_1_is_release_evidence": False,
        "scientific_conclusion": (
            "Neither of the two preregistered recovery goals was recovered "
            "by the frozen cosine-distance auxiliary; the result is retained "
            "only as append-only adaptive-development evidence and is not a "
            "full 21-gate evaluation."
        ),
    }
    if report.get("outcome") != expected_outcome:
        raise ValueError("attempt-1 miss outcome changed")
    expected_boundary = {
        "seed20262904_role": "adaptive_development_only",
        "seed20262904_metrics_informed_this_miss_report": True,
        (
            "seed20262904_rows_used_for_weight_center_feature_moment_"
            "prototype_threshold_or_rank_fit"
        ): 0,
        "sealed_validation_seed_generated_or_read": False,
        "sealed_rows_used_for_any_fit_or_selection": 0,
        "candidate_bytes_frozen_for_independent_validation": False,
    }
    if report.get("fitting_and_validation_boundary") != expected_boundary:
        raise ValueError("attempt-1 miss fitting boundary changed")


def _expected_sampling() -> dict[str, Any]:
    return {
        "population_seed": 20264101,
        "population_source": "current",
        "population_role": "train",
        "literal_profile_count": 31,
        "literal_profiles": LITERAL_PROFILES,
        "identities_per_profile_addressable": 40,
        "identity_selection_per_profile_per_episode": 1,
        "identity_selection":
            "uniform independently within each literal profile",
        "physical_scale_factors": [1.0, 1.25, 1.5, 2.0],
        "scale_views_per_selected_identity": 2,
        "scale_selection":
            "two distinct physical-scale views uniformly without replacement",
        "pairs_per_episode": 31,
        "views_per_episode": 62,
        "same_identity_within_each_pair": True,
        "same_profile_and_public_class_within_each_pair": True,
        "literal_profile_public_class_map": PROFILE_PUBLIC_CLASS_MAP,
        "profile_class_map_validation": {
            "map_keys_must_equal_literal_profiles_in_exact_order": True,
            (
                "map_values_must_equal_training_corpus_public_class_for_"
                "every_selected_view"
            ): True,
            "selected_pair_views_must_match_selected_profile": True,
            "mismatch_fails_before_auxiliary_forward": True,
        },
    }


def _validate_attempt_2(document: Mapping[str, Any]) -> None:
    expected_header = {
        "schema":
            "atomos.v5.scale-orbit.post-score-supervised-pair-adaptation-intent",
        "schema_version": 1,
        "status": (
            "frozen_before_attempt_2_trainer_or_assembler_change_training_or_"
            "inference"
        ),
        "authored_at": "2026-07-29T06:39:00-07:00",
        "lineage": "v5-scale-orbit",
        "development_only": True,
        "release_evidence": False,
        "append_only": True,
        "adaptation_attempt": 2,
        "parent_source_commit": EXPECTED_PARENT_COMMIT,
        "parent_source_commit_subject": EXPECTED_PARENT_SUBJECT,
    }
    for key, value in expected_header.items():
        if document.get(key) != value:
            raise ValueError(f"attempt-2 intent changed {key}")
    if document.get("parent_bindings") != _expected_parent_bindings(
        include_miss=True
    ):
        raise ValueError("attempt-2 parent bindings changed")
    expected_observed = {
        "source": "attempt_1_miss_report_only",
        (
            "neither_of_two_reassessed_predeclared_recovery_goals_"
            "recovered_by_attempt_1"
        ): True,
        "attempt_1_full_21_gate_status": "not_evaluated",
        "attempt_1_selected_for_continuation": False,
        "attempt_1_cosine_auxiliary_reused": False,
        "attempt_1_sc02_fusion_dev_metrics_sha256":
            METRIC_BINDINGS["attempt_1_fusion"][1],
    }
    if document.get("observed_adaptive_evidence") != expected_observed:
        raise ValueError("attempt-2 observed evidence changed")

    change = document.get("predeclared_change")
    if not isinstance(change, Mapping):
        raise ValueError("attempt-2 predeclared change is missing")
    if (
        change.get("name")
        != "profile_balanced_supervised_scale_pair_public_class_cross_entropy"
        or change.get("replaces_attempt_1_auxiliary")
        != {
            "removed_distance": "one_minus_cosine_similarity",
            "removed_reduction": "mean_over_one_pair_per_literal_profile",
            "cosine_distance_term_in_attempt_2": False,
            "attempt_1_auxiliary_coexists_with_attempt_2": False,
        }
        or change.get("pair_sampling") != _expected_sampling()
    ):
        raise ValueError("attempt-2 replacement or pair sampling changed")
    expected_randomness = {
        "episodic_rng": "numpy.default_rng(seed)",
        "pair_rng": "numpy.default_rng(seed xor 0x5CA1E)",
        "pair_rng_seed_rule": "seed xor 0x5CA1E",
        "pair_rng_seed_for_seed20260740": 19983770,
        "pair_rng_separate_from_episodic_rng": True,
        "pair_sampling_does_not_advance_episodic_rng": True,
    }
    if change.get("randomness") != expected_randomness:
        raise ValueError("attempt-2 RNG separation changed")
    expected_phase = {
        "enabled": True,
        "applied_after_flattening_all_62_paired_views": True,
        "draw": "one independent uniform phase in [0, 2*pi) per paired view",
        "two_views_in_a_pair_share_phase_draw": False,
        "existing_episodic_phase_augmentation_changed": False,
    }
    if change.get("phase_augmentation") != expected_phase:
        raise ValueError("attempt-2 independent phase augmentation changed")
    expected_bn = {
        "applies_to": "auxiliary_forward_only",
        "batch_norm_module_types": [
            "BatchNorm1d",
            "BatchNorm2d",
            "BatchNorm3d",
            "SyncBatchNorm",
        ],
        "auxiliary_forward_uses_batch_norm_eval_mode": True,
        "running_mean_variance_or_num_batches_tracked_mutated": False,
        "original_batch_norm_training_modes_restored_after_forward": True,
        "dropout_or_non_batch_norm_training_mode_changed": False,
        "gradient_tracking_disabled_for_paired_embeddings": False,
    }
    if change.get("batch_norm_stat_firewall") != expected_bn:
        raise ValueError("attempt-2 BatchNorm-stat firewall changed")

    expected_auxiliary = {
        "paired_embedding_count": 62,
        "target_for_each_view":
            "public class of its selected literal current profile",
        "target_tensor_construction": (
            "repeat(profile_public_class_indices, 2) in exact literal "
            "profile order after flattening each profile's two views "
            "contiguously"
        ),
        "public_class_prototypes": {
            "source": (
                "same current-episode source-mixed episodic support embeddings "
                "used by the main episodic cross-entropy"
            ),
            "source_mixing": (
                "inherits the unchanged episodic sampler and exact "
                "current_source_share 1/3; prototypes are not "
                "current-source-only"
            ),
            "class_order_and_count": (
                "same seven public-class prototypes built for the current "
                "episode main episodic cross-entropy"
            ),
            "detached_for_auxiliary": True,
            "auxiliary_gradient_into_support_embeddings_or_prototype_construction":
                False,
            "main_episodic_cross_entropy_prototype_gradient_changed": False,
            "persistent_enrollment_prototype_bank_used": False,
            "prototype_fit_population_changed": False,
        },
        "logits": {
            "formula": (
                "-squared_euclidean_distance(paired_embedding, "
                "detached_current_episode_public_class_prototype) * "
                "exp(detach(logit_scale)).clamp(1e-3, 100.0)"
            ),
            "distance":
                "same squared Euclidean distance as current episodic cross-entropy",
            "logit_scale_parameter": (
                "detached numeric value of the same learnable current-episode "
                "logit_scale used by episodic cross-entropy"
            ),
            "logit_scale_detached_for_auxiliary": True,
            "auxiliary_gradient_into_logit_scale": False,
            "main_episodic_cross_entropy_is_only_logit_scale_optimizer_signal":
                True,
            "new_temperature_or_margin": False,
        },
        "cross_entropy": {
            "label_smoothing": 0.0,
            "reduction": "mean",
            "mean_denominator": 62,
        },
        "auxiliary_coefficient": 0.2,
        "total_loss": (
            "episodic_cross_entropy + 0.2 * "
            "mean_supervised_pair_cross_entropy_over_62_views"
        ),
    }
    if change.get("supervised_pair_auxiliary") != expected_auxiliary:
        raise ValueError("attempt-2 supervised-pair CE contract changed")

    expected_run = {
        "seed": 20260740,
        "episodes": 8000,
        "evaluation_interval": 500,
        "dropout": 0.35,
        "weight_decay": 0.0005,
        "current_source_share": {
            "numerator": 1,
            "denominator": 3,
            "value": 1 / 3,
        },
        "phase_augmentation": True,
        "encoders": ["real", "complex"],
        "branch_architecture": {
            "hidden": 96,
            "embed_dim": 32,
            "n_features": 12,
            "patch_length": 64,
            "patch_count": 16,
            "patch_dim": 64,
            "set_pool": "mean_std",
        },
        "branch_fusion": {
            "kind": "centered_invariant_fusion",
            "weight_real": 0.5,
            "weight_complex": 0.5,
            "alpha_real": 0.2,
            "alpha_complex": 0.2,
            "eps": 1e-12,
            "rule": (
                "unit(concat(sqrt(w) * unit(real - 0.2 * real_center), "
                "sqrt(1-w) * unit(complex - 0.2 * complex_center)))"
            ),
        },
    }
    if document.get("inherited_run_contract") != expected_run:
        raise ValueError("attempt-2 inherited run or fusion contract changed")
    expected_no_change = {
        "model_architecture_changed": False,
        "frontend_or_trusted_geometry_canonicalizer_changed": False,
        "training_enrollment_or_adaptive_selection_data_changed": False,
        "seed_role_allocation_changed": False,
        "episodic_sampler_or_main_cross_entropy_changed": False,
        "feature_moment_or_center_fit_changed": False,
        "persistent_prototype_fit_population_grouping_or_procedure_changed":
            False,
        "checkpoint_score_order_threshold_or_comparison_changed": False,
        "development_or_release_gate_name_threshold_or_comparison_changed":
            False,
        "branch_fusion_changed": False,
        "class_or_profile_added_removed_or_remapped": False,
        "open_set_threshold_or_rank_fit_changed": False,
    }
    if document.get("no_other_change_contract") != expected_no_change:
        raise ValueError("attempt-2 no-other-change firewall changed")
    expected_fitting = {
        "supervised_pairs_from_seed20264101_current_train_role_only": True,
        "seed20264101_enrollment_rows_used_for_supervised_pair_auxiliary": 0,
        "seed20262904_selection_rows_used_for_any_gradient_or_optimizer_step": 0,
        "seed20262904_selection_rows_used_for_weight_center_or_feature_moment_fit":
            0,
        "seed20262904_selection_rows_used_for_persistent_prototype_fit": 0,
        "seed20262904_selection_rows_used_for_open_set_threshold_or_rank_fit":
            0,
        (
            "seed20262904_selection_metrics_may_retain_the_existing_"
            "development_checkpoint_and_candidate_selection_role"
        ): True,
        "sealed_rows_used_for_any_fit_checkpoint_or_candidate_selection": 0,
    }
    if document.get("fitting_firewall") != expected_fitting:
        raise ValueError("attempt-2 selection/sealed fitting firewall changed")
    expected_timing = {
        "attempt_1_adaptive_selection_metrics_observed_before_attempt_2": True,
        "attempt_2_trainer_source_changed_before_this_intent": False,
        "attempt_2_assembler_source_changed_before_this_intent": False,
        "attempt_2_training_started_before_this_intent": False,
        "attempt_2_inference_started_before_this_intent": False,
        "attempt_2_may_be_selected_on_seed20262904": True,
        "attempt_2_is_independent_validation_evidence": False,
        "candidate_must_be_frozen_before_any_sealed_generation": True,
        "sealed_seed_generated_or_read": False,
    }
    if document.get("timing_and_validation_boundary") != expected_timing:
        raise ValueError("attempt-2 timing/validation boundary changed")


def validate_documents(
    attempt_2: Mapping[str, Any],
    miss_report: Mapping[str, Any],
    attempt_1: Mapping[str, Any],
    protocol: Mapping[str, Any],
    metrics: Mapping[str, Mapping[str, Any]],
    observed_metric_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Validate already-loaded documents, for focused mutation tests."""
    _validate_metric_bindings(metrics, observed_metric_hashes)
    _validate_miss_report(miss_report, attempt_1, protocol, metrics)
    _validate_attempt_2(attempt_2)
    return {
        "schema":
            "atomos.v5.scale-orbit.supervised-pair-attempt-2-validation",
        "schema_version": 1,
        "valid": True,
        "reassessed_predeclared_failed_goal_count": 2,
        "literal_profile_count": 31,
        "paired_views_per_episode": 62,
        "supervised_pair_coefficient": 0.2,
        "development_metric_files_read": 6,
        "corpus_files_read": 0,
        "checkpoint_files_read": 0,
        "sealed_files_read": 0,
    }


def validate_repository_attempt_2() -> dict[str, Any]:
    """Validate exact repository bytes and their declared evidence."""
    expected_contract_hashes = {
        PROTOCOL_PATH: EXPECTED_PROTOCOL_SHA256,
        ATTEMPT_1_PATH: EXPECTED_ATTEMPT_1_SHA256,
        MISS_REPORT_PATH: EXPECTED_MISS_REPORT_SHA256,
        ATTEMPT_2_PATH: EXPECTED_ATTEMPT_2_SHA256,
    }
    for path, expected in expected_contract_hashes.items():
        if _sha256(path) != expected:
            raise ValueError(f"{path.name} raw SHA-256 changed")
    metrics = {
        name: _read_json(path)
        for name, path in METRIC_PATHS.items()
    }
    metric_hashes = {
        name: _sha256(path)
        for name, path in METRIC_PATHS.items()
    }
    summary = validate_documents(
        _read_json(ATTEMPT_2_PATH),
        _read_json(MISS_REPORT_PATH),
        _read_json(ATTEMPT_1_PATH),
        _read_json(PROTOCOL_PATH),
        metrics,
        metric_hashes,
    )
    return {
        **summary,
        "attempt_1_miss_report_sha256": EXPECTED_MISS_REPORT_SHA256,
        "attempt_2_intent_sha256": EXPECTED_ATTEMPT_2_SHA256,
    }


def main() -> None:
    print(
        json.dumps(
            validate_repository_attempt_2(),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
