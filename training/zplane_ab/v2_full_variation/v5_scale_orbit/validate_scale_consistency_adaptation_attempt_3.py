"""Validate the attempt-2 miss evidence and outcome-blind attempt-3 intent.

Only exact append-only JSON contracts, five explicitly bound development
``dev_metrics.json`` files, and two explicitly bound committed Python source
blobs are read.  After the intent is committed, its explicitly bound committed
JSON blob is read once more to prove commit ancestry.  No corpus, checkpoint,
prototype, calibration, novelty row, or sealed-validation artifact is read.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PROTOCOL_PATH = HERE / "recovery_protocol.json"
SEED_REGISTRY_PATH = HERE / "seed_registry.json"
ATTEMPT_1_PATH = HERE / "scale_consistency_adaptation_attempt_1.json"
ATTEMPT_1_MISS_PATH = (
    HERE / "scale_consistency_adaptation_attempt_1_miss_report.json"
)
ATTEMPT_2_PATH = HERE / "scale_consistency_adaptation_attempt_2.json"
ATTEMPT_2_MISS_PATH = (
    HERE / "scale_consistency_adaptation_attempt_2_miss_report.json"
)
ATTEMPT_3_PATH = HERE / "scale_consistency_adaptation_attempt_3.json"
NOVELTY_PATH = HERE / "openset_novelty_adaptive_amendment.json"

EXPECTED_CONTRACT_HASHES = {
    PROTOCOL_PATH:
        "cc79496ba398e4ce2fcb6875d8b810bcbc1ae7f89cb75271fdf2c65e6ad2afff",
    SEED_REGISTRY_PATH:
        "b832e09e8eba264e21d7845f72c0d52f671aa1f9187d55ddce94fb833347e0cd",
    ATTEMPT_1_PATH:
        "e3166235ed53835465b3bd7afa6ad03982e31016b4d20c6875879288b3304a90",
    ATTEMPT_1_MISS_PATH:
        "1c0e6d94d418e3954a54aa44396ef5cc9d8603eb2895dd69c0a0c458c254bc9d",
    ATTEMPT_2_PATH:
        "5e0cc7bbd58ebe050bf983f854e154c43595dc804aedee3c1c3849cf5691b0a9",
    ATTEMPT_2_MISS_PATH:
        "ccdcbbac1e77ff9886b32a7982c64ffcf19819156ac2c08eaae3a54c3a8fdbee",
    ATTEMPT_3_PATH:
        "8cf6b53875be644a56b970012d332bff6e8a4d0408de83d466deedb74b5d721e",
    NOVELTY_PATH:
        "5622458262d51a255fa70f00f1a8baeb150500d32bb9af65db44c641cf4e1154",
}

METRIC_BINDINGS = {
    "baseline_fusion": (
        ".artifacts/v5-scale-orbit-fusion-w0.5/dev_metrics.json",
        "1c02705081cb4f973e1ea092a0b995501ca9c84e609f09d7931f2337462b28b5",
    ),
    "attempt_1_fusion": (
        ".artifacts/v5-scale-orbit-fusion-w0.5-sc02/dev_metrics.json",
        "d23c8717261961dec99a07cc45f0476cdc784fdacaf827742ee1677275186e7b",
    ),
    "attempt_2_real": (
        ".artifacts/v5-scale-orbit-real-seed20260740-e8000-reg-spair02/"
        "dev_metrics.json",
        "8128780c4b81f44934d29ca77ad80409289d10d1c488e024a7fdfdf3cbfce34e",
    ),
    "attempt_2_complex": (
        ".artifacts/v5-scale-orbit-complex-seed20260740-e8000-reg-spair02/"
        "dev_metrics.json",
        "e834e323a6da40c678a826d6376733ea4585e8da9cdb9ab51b16af7e1fd74071",
    ),
    "attempt_2_fusion": (
        ".artifacts/v5-scale-orbit-fusion-w0.5-spair02/dev_metrics.json",
        "13d25fdfd81916a49ba2e48bca49992bbd4524e3b91191299ca6fb43ebbebd87",
    ),
}
METRIC_PATHS = {
    name: REPO / relative
    for name, (relative, _digest) in METRIC_BINDINGS.items()
}
ALLOWED_PATHS = set(EXPECTED_CONTRACT_HASHES) | set(METRIC_PATHS.values())

PREREG_COMMIT = "53ae9eb2060a0f65a2ffcf607d98866bb695e65c"
PREREG_PARENT = "2d6dbd2ba71286233dee660605e095122eb7c26a"
PREREG_SUBJECT = "Freeze supervised scale-pair adaptation"
SOURCE_COMMIT = "e14a7f2160e6792a07a77619be7281c55c8c0704"
SOURCE_SUBJECT = "Train v5 with supervised scale pairs"
TRAINER_RELATIVE = (
    "training/zplane_ab/v2_full_variation/v5_scale_orbit/"
    "run_scale_orbit_dev.py"
)
ASSEMBLER_RELATIVE = (
    "training/zplane_ab/v2_full_variation/v5_scale_orbit/"
    "assemble_scale_orbit_fusion.py"
)
TRAINER_SHA256 = (
    "017077137b88fdce5c5af9c3691a9303aceee08521c607ab28ab3994a45152aa"
)
ASSEMBLER_SHA256 = (
    "4f4443ff9dfc938a9e1e925edcfb3de7653582ed3b6b5feced7886d7842ca224"
)
COMMITTED_SOURCE_BLOB_BINDINGS = {
    TRAINER_RELATIVE: TRAINER_SHA256,
    ASSEMBLER_RELATIVE: ASSEMBLER_SHA256,
}
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
    "current_pooled_profile_balanced_accuracy,current_pooled_accuracy,"
    "combined_pooled_balanced_accuracy,"
    "historical_worst_length_balanced_accuracy)"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved not in {item.resolve() for item in ALLOWED_PATHS}:
        raise ValueError(f"refusing undeclared evidence path: {resolved}")
    with resolved.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def _metric(document: Mapping[str, Any], dotted_path: str) -> Any:
    value: Any = document
    for component in dotted_path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise ValueError(f"missing metric path {dotted_path!r}")
        value = value[component]
    return value


def _gate(protocol: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    rows = protocol.get("gate_contract", {}).get("inherited_gates", [])
    matches = [
        row for row in rows
        if isinstance(row, Mapping) and row.get("name") == name
    ]
    if len(matches) != 1:
        raise ValueError(f"frozen gate {name!r} is not unique")
    return matches[0]


def _validate_metrics(
    metrics: Mapping[str, Mapping[str, Any]],
    observed_hashes: Mapping[str, str],
) -> None:
    if set(metrics) != set(METRIC_BINDINGS):
        raise ValueError("development metric evidence set changed")
    if set(observed_hashes) != set(METRIC_BINDINGS):
        raise ValueError("development metric hash evidence set changed")
    for name, (_relative, expected) in METRIC_BINDINGS.items():
        if observed_hashes.get(name) != expected:
            raise ValueError(f"{name} raw dev_metrics SHA-256 changed")
        document = metrics[name]
        if (
            document.get("status") != "complete"
            or document.get("development_only") is not True
            or document.get("release_evidence") is not False
        ):
            raise ValueError(f"{name} is not completed development evidence")

    for name in ("attempt_2_real", "attempt_2_complex"):
        branch = metrics[name]
        adaptation = branch.get("run_configuration", {}).get("adaptation")
        arguments = branch.get("run_configuration", {}).get("arguments")
        selection = branch.get("run_configuration", {}).get(
            "checkpoint_selection"
        )
        if (
            not isinstance(adaptation, Mapping)
            or adaptation.get("adaptation_attempt") != 2
            or adaptation.get("sha256")
            != EXPECTED_CONTRACT_HASHES[ATTEMPT_2_PATH]
            or adaptation.get("preregistration_commit") != PREREG_COMMIT
            or adaptation.get("supervised_pair_weight") != 0.2
            or adaptation.get("auxiliary") != "public_class_cross_entropy"
            or adaptation.get("pairs_per_episode") != 31
            or adaptation.get("views_per_episode") != 62
        ):
            raise ValueError(f"{name} attempt-2 adaptation binding changed")
        if (
            not isinstance(arguments, Mapping)
            or arguments.get("seed") != 20260740
            or arguments.get("episodes") != 8000
            or arguments.get("eval_every") != 500
            or arguments.get("dropout") != 0.35
            or arguments.get("weight_decay") != 0.0005
            or arguments.get("supervised_pair_weight") != 0.2
            or arguments.get("current_source_share", {}).get("numerator") != 1
            or arguments.get("current_source_share", {}).get("denominator")
            != 3
            or arguments.get("phase_augmentation") is not True
        ):
            raise ValueError(f"{name} frozen run tuple changed")
        if (
            not isinstance(selection, Mapping)
            or selection.get("contract") != CHECKPOINT_SCORE_CONTRACT
            or selection.get("predeclared") is not True
            or selection.get("tunable") is not False
        ):
            raise ValueError(f"{name} checkpoint tuple changed")
        source = branch.get("source_sha256", {})
        if (
            source.get("v5/run_scale_orbit_dev.py") != TRAINER_SHA256
            or source.get("v5/scale_consistency_adaptation_attempt_2.json")
            != EXPECTED_CONTRACT_HASHES[ATTEMPT_2_PATH]
        ):
            raise ValueError(f"{name} source snapshot changed")

    fusion = metrics["attempt_2_fusion"]
    branches = fusion.get("source_branches", {})
    for encoder in ("real", "complex"):
        bound = branches.get(encoder, {}).get("hashes", {}).get(
            "dev_metrics.json"
        )
        if bound != METRIC_BINDINGS[f"attempt_2_{encoder}"][1]:
            raise ValueError(
                f"attempt-2 fusion does not bind its {encoder} branch"
            )
    if (
        fusion.get("source_sha256", {}).get(
            "v5/assemble_scale_orbit_fusion.py"
        )
        != ASSEMBLER_SHA256
    ):
        raise ValueError("attempt-2 fusion assembler snapshot changed")


def _validate_report(
    report: Mapping[str, Any],
    protocol: Mapping[str, Any],
    novelty: Mapping[str, Any],
    metrics: Mapping[str, Mapping[str, Any]],
) -> None:
    expected_header = {
        "schema": "atomos.v5.scale-orbit.adaptation-attempt-miss-report",
        "schema_version": 2,
        "status": (
            "adaptation_attempt_2_recovered_pooled_accuracy_but_not_"
            "every_profile_scale_agreement"
        ),
        "authored_at": "2026-07-29T07:41:20-07:00",
        "lineage": "v5-scale-orbit",
        "development_only": True,
        "release_evidence": False,
        "append_only": True,
        "adaptation_attempt": 2,
        "parent_source_commit": SOURCE_COMMIT,
        "parent_source_commit_subject": SOURCE_SUBJECT,
        "parent_source_commit_parent": PREREG_COMMIT,
    }
    for key, expected in expected_header.items():
        if report.get(key) != expected:
            raise ValueError(f"attempt-2 miss report changed {key}")

    rows = report.get("reassessed_predeclared_recovery_goals")
    expected_rows = [
        (
            "classifier_development_current_service_pooled_closed_accuracy",
            "final_evaluation.current.pooled.accuracy",
            0.940398185483871,
            0.9343497983870968,
            0.96875,
            True,
        ),
        (
            "length_scale_outcome_agreement_every_source_profile",
            (
                "final_evaluation.current.pooled."
                "worst_profile_physical_scale_prediction_agreement_rate"
            ),
            0.875,
            0.875,
            0.90625,
            False,
        ),
    ]
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("attempt-2 reassessed recovery goal set changed")
    for row, (name, path, baseline, attempt_1, attempt_2, passed) in zip(
        rows, expected_rows
    ):
        frozen = _gate(protocol, name)
        if row != {
            "name": name,
            "comparison": "greater_than_or_equal",
            "threshold": 0.95,
            "fusion_metric_path": path,
            "baseline_observed": baseline,
            "attempt_1_sc02_observed": attempt_1,
            "attempt_2_spair02_observed": attempt_2,
            "attempt_2_spair02_pass": passed,
            "recovered_by_attempt_2": passed,
        }:
            raise ValueError(f"attempt-2 recovery row {name!r} changed")
        if (
            frozen.get("comparison") != "greater_than_or_equal"
            or frozen.get("threshold") != 0.95
            or _metric(metrics["baseline_fusion"], path) != baseline
            or _metric(metrics["attempt_1_fusion"], path) != attempt_1
            or _metric(metrics["attempt_2_fusion"], path) != attempt_2
            or (attempt_2 >= 0.95) is not passed
        ):
            raise ValueError(f"attempt-2 recovery row {name!r} not reproduced")

    pooled = metrics["attempt_2_fusion"]["final_evaluation"]["current"][
        "pooled"
    ]
    below_cells = [
        {
            "profile_scale": key,
            "public_class": value["public_class"],
            "rows": value["rows"],
            "recall": value["recall"],
        }
        for key, value in pooled["per_profile_scale"].items()
        if value["recall"] < 0.9
    ]
    diagnostics = report.get("exact_subthreshold_classifier_diagnostics", {})
    if (
        diagnostics.get("legal_cell_accuracy_reference_threshold") != 0.9
        or diagnostics.get("formal_21_gate_result") is not False
        or diagnostics.get("profile_scale_cells_below_reference_threshold")
        != below_cells
    ):
        raise ValueError("attempt-2 subthreshold profile/scale cells changed")
    below_agreement = {
        key: value
        for key, value in
        pooled["per_profile_physical_scale_prediction_agreement_rate"].items()
        if value < 0.95
    }
    reported_agreement = {
        row["profile"]: row["agreement"]
        for row in diagnostics.get("profiles_below_agreement_threshold", [])
    }
    reported_needed = {
        row["profile"]: row[
            "additional_consistent_pairs_required_to_reach_next_64_row_"
            "fraction_at_or_above_threshold"
        ]
        for row in diagnostics.get("profiles_below_agreement_threshold", [])
    }
    expected_needed = {
        "current:lte-band3-fdd-20m": 3,
        "current:wifi6-he-mu": 1,
        "current:wifi6-he-tb": 2,
    }
    if (
        diagnostics.get("every_profile_agreement_threshold") != 0.95
        or reported_agreement != below_agreement
        or reported_needed != expected_needed
        or any(
            row.get("paired_identity_count") != 64
            for row in diagnostics.get(
                "profiles_below_agreement_threshold", []
            )
        )
    ):
        raise ValueError("attempt-2 subthreshold profile agreement changed")

    gate_boundary = report.get("full_gate_assessment", {})
    full = novelty.get("full_gate_contract", {})
    if (
        gate_boundary.get("known_21_gate_status") != "not_evaluated"
        or gate_boundary.get("known_plus_novelty_27_gate_status")
        != "not_evaluated"
        or gate_boundary.get("known_gate_count") != 21
        or gate_boundary.get("novelty_gate_count") != 6
        or gate_boundary.get("total_gate_count") != 27
        or gate_boundary.get("classifier_diagnostics_are_not_formal_gate_rows")
        is not True
        or gate_boundary.get("other_gates_claimed_passed") is not False
        or gate_boundary.get("release_claim_permitted") is not False
        or full.get("exact_known_gate_count") != 21
        or full.get("exact_novelty_gate_count") != 6
        or full.get("exact_total_gate_count") != 27
    ):
        raise ValueError("attempt-2 21/27-gate non-evaluation changed")

    wifi_bt = report.get("wifi_bluetooth_preservation_diagnostics", {})
    if (
        wifi_bt.get("current_wifi_hr_dsss_profile_recall")
        != pooled["per_profile"]["current:wifi-hr-dsss-11m"]["recall"]
        or wifi_bt.get("current_wifi_hr_dsss_profile_scale_agreement")
        != pooled["per_profile_physical_scale_prediction_agreement_rate"][
            "current:wifi-hr-dsss-11m"
        ]
        or wifi_bt.get("current_bluetooth_classic_profile_recall")
        != pooled["per_profile"]["current:bluetooth-classic-connected"][
            "recall"
        ]
        or wifi_bt.get("current_bluetooth_le_profile_recall")
        != pooled["per_profile"]["current:bluetooth-le-advertising"][
            "recall"
        ]
        or wifi_bt.get(
            "directional_wifi_hr_dsss_to_bluetooth_confusion_rate"
        ) != 0.0
        or wifi_bt.get("directional_bluetooth_to_dsss_confusion_rate") != 0.0
    ):
        raise ValueError("attempt-2 WiFi/Bluetooth preservation changed")


def _validate_intent(
    intent: Mapping[str, Any],
    report: Mapping[str, Any],
    attempt_2: Mapping[str, Any],
) -> None:
    expected_header = {
        "schema": (
            "atomos.v5.scale-orbit.post-score-hard-view-supervised-pair-"
            "adaptation-intent"
        ),
        "schema_version": 1,
        "status": (
            "frozen_before_attempt_3_trainer_or_assembler_change_training_or_"
            "inference"
        ),
        "authored_at": "2026-07-29T07:41:20-07:00",
        "lineage": "v5-scale-orbit",
        "development_only": True,
        "release_evidence": False,
        "append_only": True,
        "adaptation_attempt": 3,
        "parent_source_commit": SOURCE_COMMIT,
        "parent_source_commit_subject": SOURCE_SUBJECT,
        "parent_source_commit_parent": PREREG_COMMIT,
    }
    for key, expected in expected_header.items():
        if intent.get(key) != expected:
            raise ValueError(f"attempt-3 intent changed {key}")
    bindings = intent.get("parent_bindings", {})
    if (
        bindings.get("attempt_2_miss_report_sha256")
        != EXPECTED_CONTRACT_HASHES[ATTEMPT_2_MISS_PATH]
        or bindings.get("attempt_2_intent_sha256")
        != EXPECTED_CONTRACT_HASHES[ATTEMPT_2_PATH]
        or bindings.get("recovery_protocol_sha256")
        != EXPECTED_CONTRACT_HASHES[PROTOCOL_PATH]
        or bindings.get("seed_registry_sha256")
        != EXPECTED_CONTRACT_HASHES[SEED_REGISTRY_PATH]
        or bindings.get("openset_novelty_adaptive_amendment_sha256")
        != EXPECTED_CONTRACT_HASHES[NOVELTY_PATH]
    ):
        raise ValueError("attempt-3 parent contract bindings changed")
    observed = intent.get("observed_adaptive_evidence", {})
    outcome = report.get("outcome", {})
    if (
        observed.get("source") != "attempt_2_miss_report_only"
        or observed.get("attempt_2_full_21_gate_status") != "not_evaluated"
        or observed.get("attempt_2_full_27_gate_status") != "not_evaluated"
        or observed.get("attempt_2_recovered_pooled_accuracy_goal")
        is not outcome.get("attempt_2_recovered_pooled_accuracy_goal")
        or observed.get("attempt_2_recovered_every_profile_agreement_goal")
        is not outcome.get(
            "attempt_2_recovered_every_profile_agreement_goal"
        )
    ):
        raise ValueError("attempt-3 observed adaptive evidence changed")

    change = intent.get("predeclared_change", {})
    auxiliary = change.get("supervised_pair_auxiliary", {})
    reduction = auxiliary.get("cross_entropy", {})
    if (
        change.get("name")
        != (
            "per_profile_max_over_two_scale_views_supervised_public_class_"
            "cross_entropy"
        )
        or change.get("only_changed_axis")
        != "supervised_pair_cross_entropy_reduction"
        or change.get(
            "attempt_2_pair_population_sampling_order_targets_logits_and_"
            "coefficient_changed"
        ) is not False
        or reduction.get("per_view_reduction") != "none"
        or reduction.get("per_view_loss_shape_before_profile_reduction")
        != [31, 2]
        or reduction.get("within_profile_reduction")
        != "maximum_over_exactly_two_contiguous_scale_view_losses"
        or reduction.get("across_profile_reduction")
        != "arithmetic_mean_over_31_literal_profile_hard_losses"
        or reduction.get("across_profile_mean_denominator") != 31
        or reduction.get(
            "maximum_is_not_taken_over_profiles_classes_or_batch_episodes"
        ) is not True
        or auxiliary.get("auxiliary_coefficient") != 0.2
        or auxiliary.get("cosine_distance_term") is not False
    ):
        raise ValueError("attempt-3 hard-view reduction changed")

    parent_change = attempt_2.get("predeclared_change", {})
    if (
        change.get("randomness_inherited_exactly_from_attempt_2")
        != parent_change.get("randomness")
        or change.get("phase_augmentation_inherited_exactly_from_attempt_2")
        != parent_change.get("phase_augmentation")
        or change.get(
            "batch_norm_stat_firewall_inherited_exactly_from_attempt_2"
        ) != parent_change.get("batch_norm_stat_firewall")
        or auxiliary.get("public_class_prototypes")
        != parent_change.get("supervised_pair_auxiliary", {}).get(
            "public_class_prototypes"
        )
        or auxiliary.get("logits")
        != parent_change.get("supervised_pair_auxiliary", {}).get("logits")
    ):
        raise ValueError("attempt-3 inherited pair machinery changed")

    weighting = auxiliary.get("profile_and_class_weighting", {})
    if (
        weighting.get("each_literal_profile_weight_before_auxiliary_coefficient")
        != 1 / 31
        or weighting.get("class_balancing_added") is not False
        or weighting.get("hard_profile_mining_across_profiles_added") is not False
        or weighting.get("induced_profile_count_weights")
        != {
            "bluetooth": {"profile_count": 2, "total_weight": 2 / 31},
            "dsss": {"profile_count": 1, "total_weight": 1 / 31},
            "gsm": {"profile_count": 7, "total_weight": 7 / 31},
            "ofdm": {"profile_count": 21, "total_weight": 21 / 31},
        }
    ):
        raise ValueError("attempt-3 equal-profile weighting changed")

    inherited = dict(intent.get("inherited_run_contract", {}))
    checkpoint = inherited.pop("checkpoint_score_contract", None)
    if (
        inherited != attempt_2.get("inherited_run_contract")
        or checkpoint != CHECKPOINT_SCORE_CONTRACT
    ):
        raise ValueError("attempt-3 frozen run/checkpoint tuple changed")
    if intent.get("fitting_firewall") != attempt_2.get("fitting_firewall"):
        raise ValueError("attempt-3 fitting firewall changed")
    no_change = intent.get("no_other_change_contract", {})
    if not no_change or any(value is not False for value in no_change.values()):
        raise ValueError("attempt-3 no-other-change firewall changed")
    full = intent.get("full_gate_boundary", {})
    if (
        full.get("attempt_2_known_21_gate_status") != "not_evaluated"
        or full.get("attempt_2_known_plus_novelty_27_gate_status")
        != "not_evaluated"
        or full.get(
            "attempt_3_may_not_inherit_or_claim_any_formal_gate_pass_from_"
            "classifier_diagnostics"
        ) is not True
        or full.get("attempt_3_requires_a_new_exact_21_known_gate_run")
        is not True
        or full.get(
            "attempt_3_requires_a_new_exact_27_known_plus_novelty_gate_run"
        ) is not True
    ):
        raise ValueError("attempt-3 21/27-gate boundary changed")
    timing = intent.get("timing_and_validation_boundary", {})
    forbidden_true = [
        "attempt_3_trainer_source_changed_before_this_intent",
        "attempt_3_assembler_source_changed_before_this_intent",
        "attempt_3_training_started_before_this_intent",
        "attempt_3_inference_started_before_this_intent",
        "attempt_3_is_independent_validation_evidence",
        "sealed_seed_generated_or_read",
    ]
    if any(timing.get(key) is not False for key in forbidden_true):
        raise ValueError("attempt-3 timing/validation boundary changed")


def validate_documents(
    intent: Mapping[str, Any],
    report: Mapping[str, Any],
    attempt_2: Mapping[str, Any],
    protocol: Mapping[str, Any],
    novelty: Mapping[str, Any],
    metrics: Mapping[str, Mapping[str, Any]],
    observed_metric_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Validate already-loaded values for focused mutation tests."""
    _validate_metrics(metrics, observed_metric_hashes)
    _validate_report(report, protocol, novelty, metrics)
    _validate_intent(intent, report, attempt_2)
    return {
        "schema":
            "atomos.v5.scale-orbit.hard-view-attempt-3-validation",
        "schema_version": 1,
        "valid": True,
        "attempt_2_recovered_goal_count": 1,
        "attempt_2_remaining_predeclared_goal_miss_count": 1,
        "attempt_2_subthreshold_profile_scale_cell_count": 4,
        "attempt_2_subthreshold_profile_agreement_count": 3,
        "literal_profile_count": 31,
        "paired_views_per_episode": 62,
        "hard_views_per_profile": 2,
        "supervised_pair_coefficient": 0.2,
        "known_gate_count": 21,
        "full_gate_count": 27,
        "development_metric_files_read": 5,
        "committed_source_blobs_read":
            len(COMMITTED_SOURCE_BLOB_BINDINGS),
        "corpus_files_read": 0,
        "checkpoint_files_read": 0,
        "prototype_files_read": 0,
        "novelty_row_files_read": 0,
        "sealed_files_read": 0,
        "model_inference_runs": 0,
    }


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=REPO,
        check=check,
        capture_output=True,
        text=True,
    )


def _validate_source_ancestry() -> dict[str, bool]:
    prereg = _git(
        "show", "-s", "--format=%H%n%P%n%s", PREREG_COMMIT
    ).stdout.splitlines()
    source = _git(
        "show", "-s", "--format=%H%n%P%n%s", SOURCE_COMMIT
    ).stdout.splitlines()
    if prereg != [PREREG_COMMIT, PREREG_PARENT, PREREG_SUBJECT]:
        raise ValueError("attempt-2 preregistration commit object changed")
    if source != [SOURCE_COMMIT, PREREG_COMMIT, SOURCE_SUBJECT]:
        raise ValueError("attempt-2 source commit object or direct parent changed")
    if _git(
        "merge-base", "--is-ancestor", PREREG_COMMIT, SOURCE_COMMIT,
        check=False,
    ).returncode != 0:
        raise ValueError("attempt-2 preregistration is not source ancestor")
    if _git(
        "merge-base", "--is-ancestor", SOURCE_COMMIT, "HEAD",
        check=False,
    ).returncode != 0:
        raise ValueError("attempt-2 source is not repository HEAD ancestor")
    for relative, expected in COMMITTED_SOURCE_BLOB_BINDINGS.items():
        blob = subprocess.run(
            ["git", "show", f"{SOURCE_COMMIT}:{relative}"],
            cwd=REPO,
            check=True,
            capture_output=True,
        ).stdout
        if hashlib.sha256(blob).hexdigest() != expected:
            raise ValueError(f"attempt-2 source blob changed: {relative}")

    relative = str(ATTEMPT_3_PATH.relative_to(REPO))
    intent_commit = _git(
        "log", "-1", "--format=%H", "--", relative
    ).stdout.strip()
    if not intent_commit:
        return {
            "attempt_2_preregistration_to_source_ancestry_verified": True,
            "attempt_2_source_to_repository_head_ancestry_verified": True,
            "attempt_2_source_to_attempt_3_intent_commit_ancestry_verified":
                False,
            "attempt_3_intent_commit_ancestry_deferred_until_commit": True,
            "committed_attempt_3_intent_blobs_read": 0,
        }
    committed = subprocess.run(
        ["git", "show", f"{intent_commit}:{relative}"],
        cwd=REPO,
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(committed).hexdigest() != EXPECTED_CONTRACT_HASHES[
        ATTEMPT_3_PATH
    ]:
        raise ValueError("committed attempt-3 intent blob changed")
    if _git(
        "merge-base", "--is-ancestor", SOURCE_COMMIT, intent_commit,
        check=False,
    ).returncode != 0:
        raise ValueError(
            "attempt-2 source is not attempt-3 intent commit ancestor"
        )
    return {
        "attempt_2_preregistration_to_source_ancestry_verified": True,
        "attempt_2_source_to_repository_head_ancestry_verified": True,
        "attempt_2_source_to_attempt_3_intent_commit_ancestry_verified": True,
        "attempt_3_intent_commit_ancestry_deferred_until_commit": False,
        "committed_attempt_3_intent_blobs_read": 1,
    }


def validate_repository_attempt_3() -> dict[str, Any]:
    """Validate exact repository bytes, evidence, and source ancestry."""
    for path, expected in EXPECTED_CONTRACT_HASHES.items():
        if _sha256(path) != expected:
            raise ValueError(f"{path.name} raw SHA-256 changed")
    metrics = {
        name: _read_json(path) for name, path in METRIC_PATHS.items()
    }
    observed_hashes = {
        name: _sha256(path) for name, path in METRIC_PATHS.items()
    }
    summary = validate_documents(
        _read_json(ATTEMPT_3_PATH),
        _read_json(ATTEMPT_2_MISS_PATH),
        _read_json(ATTEMPT_2_PATH),
        _read_json(PROTOCOL_PATH),
        _read_json(NOVELTY_PATH),
        metrics,
        observed_hashes,
    )
    # Hash-only parent bindings are also opened above and here; no unbound
    # semantic claim is inferred from them.
    for path in (
        SEED_REGISTRY_PATH,
        ATTEMPT_1_PATH,
        ATTEMPT_1_MISS_PATH,
    ):
        _read_json(path)
    return {
        **summary,
        **_validate_source_ancestry(),
        "attempt_2_miss_report_sha256":
            EXPECTED_CONTRACT_HASHES[ATTEMPT_2_MISS_PATH],
        "attempt_3_intent_sha256":
            EXPECTED_CONTRACT_HASHES[ATTEMPT_3_PATH],
    }


def main() -> None:
    print(
        json.dumps(
            validate_repository_attempt_3(),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
