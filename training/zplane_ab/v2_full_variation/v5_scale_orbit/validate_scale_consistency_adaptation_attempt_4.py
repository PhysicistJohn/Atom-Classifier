"""Validate attempt-3 miss evidence and the outcome-blind attempt-4 intent.

The validator reads only exact append-only JSON contracts, the explicitly
bound development metrics, and the two committed attempt-3 source blobs.  It
does not read a corpus, checkpoint, prototype, novelty row, or sealed artifact,
and it does not run model inference.
"""
from __future__ import annotations

import hashlib
import json
import math
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
ATTEMPT_3_MISS_PATH = (
    HERE / "scale_consistency_adaptation_attempt_3_miss_report.json"
)
ATTEMPT_4_PATH = HERE / "scale_consistency_adaptation_attempt_4.json"
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
    ATTEMPT_3_MISS_PATH:
        "1401442dfe97f0fbe0967fe6ea926876345304d7f6dcaaf0004a623f2e4c58e0",
    ATTEMPT_4_PATH:
        "9b4ad11257a5e1ab6b1bfb4a8c6c459ef7822ca47165a1cde5243d127a4ea74e",
    NOVELTY_PATH:
        "5622458262d51a255fa70f00f1a8baeb150500d32bb9af65db44c641cf4e1154",
}

METRIC_BINDINGS = {
    "attempt_2_fusion": (
        ".artifacts/v5-scale-orbit-fusion-w0.5-spair02/dev_metrics.json",
        "13d25fdfd81916a49ba2e48bca49992bbd4524e3b91191299ca6fb43ebbebd87",
    ),
    "attempt_3_real": (
        ".artifacts/v5-scale-orbit-real-seed20260740-e8000-reg-hview02/"
        "dev_metrics.json",
        "d40d5d4857d382427effb6ff2219e8171797f5f035edc5b4c2a886e3003e98b2",
    ),
    "attempt_3_complex": (
        ".artifacts/v5-scale-orbit-complex-seed20260740-e8000-reg-hview02/"
        "dev_metrics.json",
        "283cb1eede9d59b20c0cdccdfc6f2c49f89d9ae6a0f014b3d34e95cd792fe7bb",
    ),
    "attempt_3_fusion": (
        ".artifacts/v5-scale-orbit-fusion-w0.5-hview02/dev_metrics.json",
        "015728bfbce3b71818979adf9587057e464e045bcf6ac04ce1d7e09564415d4c",
    ),
}
METRIC_PATHS = {
    name: REPO / relative
    for name, (relative, _digest) in METRIC_BINDINGS.items()
}
ALLOWED_PATHS = set(EXPECTED_CONTRACT_HASHES) | set(METRIC_PATHS.values())

PREREG_COMMIT = "64e6799a8bf3d38bbbed62d2ae5006c60f79c936"
SOURCE_COMMIT = "20a26849a1f16f6ecbd9a169f2d74289e2dd39e0"
SOURCE_PARENT = PREREG_COMMIT
SOURCE_SUBJECT = "Train v5 with hard-view scale pairs"
TRAINER_RELATIVE = (
    "training/zplane_ab/v2_full_variation/v5_scale_orbit/"
    "run_scale_orbit_dev.py"
)
ASSEMBLER_RELATIVE = (
    "training/zplane_ab/v2_full_variation/v5_scale_orbit/"
    "assemble_scale_orbit_fusion.py"
)
TRAINER_SHA256 = (
    "3f40a186d1175017653773db350b40cf726bba8358209cbfc949efacfda61101"
)
ASSEMBLER_SHA256 = (
    "f61de52cc38bc97612edac950a08d3a50bd988a0192df5a16adbec1a41d476c9"
)
COMMITTED_SOURCE_BLOB_BINDINGS = {
    TRAINER_RELATIVE: TRAINER_SHA256,
    ASSEMBLER_RELATIVE: ASSEMBLER_SHA256,
}
ATTEMPT_3_TARGET_SHA256 = (
    "838de46075dac50b6c58dee9c29fd4149711cb74bcda692b8f9edd8913b25927"
)
ATTEMPT_4_TARGET_SHA256 = (
    "f5e20b198bae039295f76093c16c9b375089f656ddb3dbfafcf2a2981428458f"
)
SCALE_FACTORS = [1.0, 1.25, 1.5, 2.0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if path.resolve() not in {item.resolve() for item in ALLOWED_PATHS}:
        raise ValueError(f"refusing undeclared evidence path: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def _metric(value: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for component in dotted_path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise ValueError(f"missing metric path {dotted_path!r}")
        current = current[component]
    return current


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )


def _assert_close(observed: Any, expected: float, name: str) -> None:
    if (
        isinstance(observed, bool)
        or not isinstance(observed, (int, float))
        or not math.isclose(
            float(observed), expected, rel_tol=0.0, abs_tol=1e-15
        )
    ):
        raise ValueError(f"{name} changed: {observed!r}")


def _validate_metrics(
    metrics: Mapping[str, Mapping[str, Any]],
    metric_hashes: Mapping[str, str],
) -> None:
    if set(metrics) != set(METRIC_BINDINGS):
        raise ValueError("development metric evidence set changed")
    if dict(metric_hashes) != {
        name: digest for name, (_path, digest) in METRIC_BINDINGS.items()
    }:
        raise ValueError("raw dev_metrics SHA-256 evidence changed")
    for name, document in metrics.items():
        if (
            document.get("status") != "complete"
            or document.get("development_only") is not True
            or document.get("release_evidence") is not False
        ):
            raise ValueError(f"{name} is not completed development evidence")

    for name, encoder, best_episode in (
        ("attempt_3_real", "real", 7500),
        ("attempt_3_complex", "complex", 8000),
    ):
        document = metrics[name]
        run = document.get("run_configuration")
        adaptation = run.get("adaptation") if isinstance(run, Mapping) else None
        arguments = run.get("arguments") if isinstance(run, Mapping) else None
        training_pair = document.get("training", {}).get("supervised_pair")
        source = document.get("source_sha256")
        if (
            document.get("encoder") != encoder
            or document.get("training", {}).get("best_episode")
            != best_episode
            or not isinstance(run, Mapping)
            or run.get("schema") != "v5-scale-orbit-development-training-v3"
            or not isinstance(adaptation, Mapping)
            or adaptation.get("adaptation_attempt") != 3
            or adaptation.get("sha256")
            != EXPECTED_CONTRACT_HASHES[ATTEMPT_3_PATH]
            or adaptation.get("preregistration_commit") != PREREG_COMMIT
            or adaptation.get("supervised_pair_weight") != 0.2
            or adaptation.get("per_view_reduction") != "none"
            or adaptation.get(
                "per_view_loss_shape_before_profile_reduction"
            )
            != [31, 2]
            or adaptation.get("within_profile_reduction")
            != "maximum_over_exactly_two_contiguous_scale_view_losses"
            or adaptation.get("across_profile_mean_denominator") != 31
            or not isinstance(training_pair, Mapping)
            or training_pair.get("paired_targets_sha256")
            != ATTEMPT_3_TARGET_SHA256
            or training_pair.get("weight") != 0.2
            or not isinstance(arguments, Mapping)
            or arguments.get("seed") != 20260740
            or arguments.get("episodes") != 8000
            or arguments.get("eval_every") != 500
            or arguments.get("supervised_pair_weight") != 0.2
            or arguments.get("phase_augmentation") is not True
            or arguments.get("current_source_share", {}).get("numerator")
            != 1
            or arguments.get("current_source_share", {}).get("denominator")
            != 3
            or not isinstance(source, Mapping)
            or source.get("v5/run_scale_orbit_dev.py") != TRAINER_SHA256
            or source.get("v5/scale_consistency_adaptation_attempt_3.json")
            != EXPECTED_CONTRACT_HASHES[ATTEMPT_3_PATH]
        ):
            raise ValueError(f"{name} exact attempt-3 branch contract changed")

    fusion = metrics["attempt_3_fusion"]
    branches = fusion.get("source_branches")
    source = fusion.get("source_sha256")
    if (
        fusion.get("encoder") != "fusion"
        or fusion.get("run_configuration", {}).get("schema")
        != "v5-scale-orbit-development-fusion-v3"
        or not isinstance(branches, Mapping)
        or any(
            branches.get(encoder, {}).get("hashes", {}).get(
                "dev_metrics.json"
            )
            != METRIC_BINDINGS[f"attempt_3_{encoder}"][1]
            for encoder in ("real", "complex")
        )
        or not isinstance(source, Mapping)
        or source.get("v5/assemble_scale_orbit_fusion.py")
        != ASSEMBLER_SHA256
        or source.get("v5/run_scale_orbit_dev.py") != TRAINER_SHA256
        or source.get("v5/scale_consistency_adaptation_attempt_3.json")
        != EXPECTED_CONTRACT_HASHES[ATTEMPT_3_PATH]
    ):
        raise ValueError("attempt-3 fusion or branch binding changed")


def _subthreshold_cells(
    fusion: Mapping[str, Any],
) -> list[dict[str, Any]]:
    cells = _metric(
        fusion, "final_evaluation.current.pooled.per_profile_scale"
    )
    result = []
    for key in sorted(cells):
        row = cells[key]
        recall = row["recall"]
        if recall < 0.9:
            rows = 64
            correct = int(round(recall * rows))
            result.append(
                {
                    "profile_scale": key,
                    "public_class": row["public_class"],
                    "rows": rows,
                    "correct_rows": correct,
                    "recall": recall,
                    (
                        "additional_correct_rows_required_to_reach_next_64_"
                        "row_fraction_at_or_above_threshold"
                    ): max(0, math.ceil(0.9 * rows) - correct),
                }
            )
    return result


def _subthreshold_agreements(
    fusion: Mapping[str, Any],
) -> list[dict[str, Any]]:
    values = _metric(
        fusion,
        "final_evaluation.current.pooled."
        "per_profile_physical_scale_prediction_agreement_rate",
    )
    result = []
    for key in sorted(values):
        agreement = values[key]
        if agreement < 0.95:
            pairs = 64
            consistent = int(round(agreement * pairs))
            result.append(
                {
                    "profile": key,
                    "paired_identity_count": pairs,
                    "consistent_pairs": consistent,
                    "agreement": agreement,
                    (
                        "additional_consistent_pairs_required_to_reach_next_"
                        "64_row_fraction_at_or_above_threshold"
                    ): max(0, math.ceil(0.95 * pairs) - consistent),
                }
            )
    return result


def _validate_report(
    report: Mapping[str, Any],
    metrics: Mapping[str, Mapping[str, Any]],
) -> None:
    expected_header = {
        "schema": "atomos.v5.scale-orbit.adaptation-attempt-miss-report",
        "schema_version": 3,
        "status": (
            "adaptation_attempt_3_improved_hard_scale_and_agreement_but_did_"
            "not_clear_every_frozen_classifier_diagnostic"
        ),
        "development_only": True,
        "release_evidence": False,
        "append_only": True,
        "adaptation_attempt": 3,
        "parent_source_commit": SOURCE_COMMIT,
        "parent_source_commit_subject": SOURCE_SUBJECT,
        "parent_source_commit_parent": SOURCE_PARENT,
    }
    if any(report.get(key) != value for key, value in expected_header.items()):
        raise ValueError("attempt-3 miss report header changed")
    bindings = report.get("parent_bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("attempt-3 report parent bindings are missing")
    for path, digest in EXPECTED_CONTRACT_HASHES.items():
        if path in {ATTEMPT_3_MISS_PATH, ATTEMPT_4_PATH}:
            continue
        stem = {
            PROTOCOL_PATH: "recovery_protocol",
            SEED_REGISTRY_PATH: "seed_registry",
            ATTEMPT_1_PATH: "attempt_1_intent",
            ATTEMPT_1_MISS_PATH: "attempt_1_miss_report",
            ATTEMPT_2_PATH: "attempt_2_intent",
            ATTEMPT_2_MISS_PATH: "attempt_2_miss_report",
            ATTEMPT_3_PATH: "attempt_3_intent",
            NOVELTY_PATH: "openset_novelty_adaptive_amendment",
        }.get(path)
        if stem and bindings.get(f"{stem}_sha256") != digest:
            raise ValueError(f"attempt-3 report parent binding {stem} changed")

    evaluated = report.get("evaluated_candidates", {})
    attempt_3 = evaluated.get("attempt_3_hview02", {})
    if (
        evaluated.get("attempt_2_spair02_fusion", {}).get(
            "dev_metrics_sha256"
        )
        != METRIC_BINDINGS["attempt_2_fusion"][1]
        or attempt_3.get("source_commit") != SOURCE_COMMIT
        or attempt_3.get("real_branch", {}).get("dev_metrics_sha256")
        != METRIC_BINDINGS["attempt_3_real"][1]
        or attempt_3.get("complex_branch", {}).get("dev_metrics_sha256")
        != METRIC_BINDINGS["attempt_3_complex"][1]
        or attempt_3.get("fusion", {}).get("dev_metrics_sha256")
        != METRIC_BINDINGS["attempt_3_fusion"][1]
    ):
        raise ValueError("attempt-3 report metric bindings changed")

    prior = metrics["attempt_2_fusion"]
    fusion = metrics["attempt_3_fusion"]
    expected_branches = {}
    for encoder in ("real", "complex"):
        branch = metrics[f"attempt_3_{encoder}"]
        branch_current = _metric(branch, "final_evaluation.current")
        branch_pooled = branch_current["pooled"]
        expected_branches[encoder] = {
            "best_episode": _metric(branch, "training.best_episode"),
            "current_pooled_closed_accuracy": branch_pooled["accuracy"],
            "current_worst_scale_present_class_balanced_accuracy":
                branch_current[
                    "worst_scale_present_class_balanced_accuracy"
                ],
            "current_worst_profile_scale_recall":
                branch_current["worst_profile_scale_recall"],
            "wifi_hr_dsss_worst_scale_accuracy":
                branch_current["wifi_hr_dsss_worst_scale_accuracy"],
            "current_physical_scale_prediction_agreement_rate":
                branch_current[
                    "physical_scale_prediction_agreement_rate"
                ],
            "current_worst_profile_physical_scale_prediction_agreement_rate":
                branch_pooled[
                    "worst_profile_physical_scale_prediction_agreement_rate"
                ],
            "historical_worst_length_balanced_accuracy": _metric(
                branch,
                "final_evaluation.historical."
                "worst_length_balanced_accuracy",
            ),
        }
    if report.get("attempt_3_branch_diagnostics") != expected_branches:
        raise ValueError("attempt-3 branch diagnostics do not reproduce")

    current = _metric(fusion, "final_evaluation.current")
    pooled = current["pooled"]
    diagnostics = report.get("attempt_3_fusion_diagnostics", {})
    expected_diagnostics = {
        "current_pooled_closed_accuracy": pooled["accuracy"],
        "current_pooled_profile_balanced_accuracy":
            pooled["profile_balanced_accuracy"],
        "current_worst_scale_present_class_balanced_accuracy":
            current["worst_scale_present_class_balanced_accuracy"],
        "current_physical_scale_prediction_agreement_rate":
            current["physical_scale_prediction_agreement_rate"],
        "current_worst_profile_physical_scale_prediction_agreement_rate":
            pooled[
                "worst_profile_physical_scale_prediction_agreement_rate"
            ],
        "current_worst_profile_scale_recall":
            current["worst_profile_scale_recall"],
        "wifi_hr_dsss_worst_scale_accuracy":
            current["wifi_hr_dsss_worst_scale_accuracy"],
        "wifi_hr_dsss_to_bluetooth_worst_scale_confusion_rate":
            pooled[
                "wifi_hr_dsss_to_bluetooth_worst_scale_confusion_rate"
            ],
        "bluetooth_to_dsss_worst_profile_scale_confusion_rate":
            pooled[
                "bluetooth_to_dsss_worst_profile_scale_confusion_rate"
            ],
        "historical_worst_length_balanced_accuracy": _metric(
            fusion,
            "final_evaluation.historical.worst_length_balanced_accuracy",
        ),
        "combined_pooled_balanced_accuracy": _metric(
            fusion, "final_evaluation.combined.pooled.balanced_accuracy"
        ),
    }
    if any(
        diagnostics.get(key) != value
        for key, value in expected_diagnostics.items()
    ):
        raise ValueError("attempt-3 fusion diagnostics do not reproduce")
    prior_current = _metric(prior, "final_evaluation.current")
    prior_pooled = prior_current["pooled"]
    expected_deltas = {
        "current_pooled_closed_accuracy":
            pooled["accuracy"] - prior_pooled["accuracy"],
        "current_net_correct_rows_out_of_7936": int(
            round(
                (pooled["accuracy"] - prior_pooled["accuracy"])
                * pooled["count"]
            )
        ),
        "current_worst_scale_present_class_balanced_accuracy":
            current["worst_scale_present_class_balanced_accuracy"]
            - prior_current[
                "worst_scale_present_class_balanced_accuracy"
            ],
        "current_worst_profile_scale_recall":
            current["worst_profile_scale_recall"]
            - prior_current["worst_profile_scale_recall"],
        "wifi_hr_dsss_worst_scale_accuracy":
            current["wifi_hr_dsss_worst_scale_accuracy"]
            - prior_current["wifi_hr_dsss_worst_scale_accuracy"],
        "current_physical_scale_prediction_agreement_rate":
            current["physical_scale_prediction_agreement_rate"]
            - prior_current["physical_scale_prediction_agreement_rate"],
        "current_worst_profile_physical_scale_prediction_agreement_rate":
            pooled["worst_profile_physical_scale_prediction_agreement_rate"]
            - prior_pooled[
                "worst_profile_physical_scale_prediction_agreement_rate"
            ],
        "historical_worst_length_balanced_accuracy": _metric(
            fusion,
            "final_evaluation.historical.worst_length_balanced_accuracy",
        )
        - _metric(
            prior,
            "final_evaluation.historical.worst_length_balanced_accuracy",
        ),
        "combined_pooled_balanced_accuracy": _metric(
            fusion, "final_evaluation.combined.pooled.balanced_accuracy"
        )
        - _metric(
            prior, "final_evaluation.combined.pooled.balanced_accuracy"
        ),
        "directional_wifi_bluetooth_confusion_rate":
            current["worst_directional_dsss_bluetooth_confusion_rate"]
            - prior_current[
                "worst_directional_dsss_bluetooth_confusion_rate"
            ],
    }
    if report.get("attempt_3_minus_attempt_2_fusion_deltas") != (
        expected_deltas
    ):
        raise ValueError("attempt-3 fusion deltas do not reproduce")

    exact = report.get("exact_subthreshold_classifier_diagnostics", {})
    if (
        exact.get("legal_cell_accuracy_reference_threshold") != 0.9
        or exact.get("profile_scale_cells_below_reference_threshold")
        != _subthreshold_cells(fusion)
        or exact.get("every_profile_agreement_threshold") != 0.95
        or exact.get("profiles_below_agreement_threshold")
        != _subthreshold_agreements(fusion)
    ):
        raise ValueError("exact attempt-3 failure inventory changed")

    prior_cells = _metric(
        prior, "final_evaluation.current.pooled.per_profile_scale"
    )
    cells = _metric(
        fusion, "final_evaluation.current.pooled.per_profile_scale"
    )
    prior_agreements = _metric(
        prior,
        "final_evaluation.current.pooled."
        "per_profile_physical_scale_prediction_agreement_rate",
    )
    agreements = _metric(
        fusion,
        "final_evaluation.current.pooled."
        "per_profile_physical_scale_prediction_agreement_rate",
    )
    prior_bad_cell_keys = [
        key for key in sorted(prior_cells)
        if prior_cells[key]["recall"] < 0.9
    ]
    expected_cell_transitions = [
        {
            "profile_scale": key,
            "rows": 64,
            "attempt_2_recall": prior_cells[key]["recall"],
            "attempt_3_recall": cells[key]["recall"],
            "delta":
                cells[key]["recall"] - prior_cells[key]["recall"],
            "attempt_3_at_or_above_0_9": cells[key]["recall"] >= 0.9,
        }
        for key in prior_bad_cell_keys
    ]
    prior_bad_agreement_keys = [
        key for key in sorted(prior_agreements)
        if prior_agreements[key] < 0.95
    ]
    expected_agreement_transitions = [
        {
            "profile": key,
            "paired_identity_count": 64,
            "attempt_2_agreement": prior_agreements[key],
            "attempt_3_agreement": agreements[key],
            "delta": agreements[key] - prior_agreements[key],
            "attempt_3_at_or_above_0_95": agreements[key] >= 0.95,
        }
        for key in prior_bad_agreement_keys
    ]
    transitions = report.get("attempt_2_failure_transition_inventory", {})
    if (
        transitions.get("profile_scale_cells")
        != expected_cell_transitions
        or transitions.get("profile_agreements")
        != expected_agreement_transitions
    ):
        raise ValueError("attempt-2 failure transition inventory changed")
    cell_deltas = [
        cells[key]["recall"] - prior_cells[key]["recall"]
        for key in sorted(cells)
    ]
    agreement_deltas = [
        agreements[key] - prior_agreements[key]
        for key in sorted(agreements)
    ]
    summary = report.get("complete_cell_and_agreement_delta_summary", {})
    expected_summary = {
        "profile_scale_cell_count": 124,
        "profile_scale_cells_improved": sum(v > 0 for v in cell_deltas),
        "profile_scale_cells_regressed": sum(v < 0 for v in cell_deltas),
        "profile_scale_cells_unchanged": sum(v == 0 for v in cell_deltas),
        "profile_scale_recall_delta_sum": sum(cell_deltas),
        "net_correct_rows": int(round(sum(cell_deltas) * 64)),
        "profile_agreement_count": 31,
        "profile_agreements_improved": sum(
            v > 0 for v in agreement_deltas
        ),
        "profile_agreements_regressed": sum(
            v < 0 for v in agreement_deltas
        ),
        "profile_agreements_unchanged": sum(
            v == 0 for v in agreement_deltas
        ),
        "profile_agreement_delta_sum": sum(agreement_deltas),
    }
    if summary != expected_summary:
        raise ValueError("complete cell/agreement delta summary changed")

    wifi = report.get("wifi_bluetooth_preservation_diagnostics", {})
    expected_wifi = {
        "current_wifi_hr_dsss_profile_recall":
            pooled["per_profile"]["current:wifi-hr-dsss-11m"]["recall"],
        "current_wifi_hr_dsss_profile_scale_agreement":
            agreements["current:wifi-hr-dsss-11m"],
        "current_bluetooth_classic_profile_recall":
            pooled["per_profile"][
                "current:bluetooth-classic-connected"
            ]["recall"],
        "current_bluetooth_classic_profile_scale_agreement":
            agreements["current:bluetooth-classic-connected"],
        "current_bluetooth_le_profile_recall":
            pooled["per_profile"][
                "current:bluetooth-le-advertising"
            ]["recall"],
        "current_bluetooth_le_profile_scale_agreement":
            agreements["current:bluetooth-le-advertising"],
        "directional_wifi_hr_dsss_to_bluetooth_confusion_rate":
            pooled[
                "wifi_hr_dsss_to_bluetooth_worst_scale_confusion_rate"
            ],
        "directional_bluetooth_to_dsss_confusion_rate":
            pooled[
                "bluetooth_to_dsss_worst_profile_scale_confusion_rate"
            ],
    }
    if wifi != expected_wifi:
        raise ValueError("WiFi/Bluetooth preservation diagnostics changed")

    gate = report.get("full_gate_assessment", {})
    if (
        gate.get("known_21_gate_status") != "not_evaluated"
        or gate.get("known_gate_count") != 21
        or gate.get("known_plus_novelty_27_gate_status") != "not_evaluated"
        or gate.get("total_gate_count") != 27
        or gate.get("other_gates_claimed_passed") is not False
        or gate.get("release_claim_permitted") is not False
    ):
        raise ValueError("attempt-3 21/27-gate boundary changed")
    if (
        report.get("outcome", {}).get(
            "attempt_3_retained_as_parent_for_attempt_4"
        )
        is not True
        or report.get("outcome", {}).get(
            "attempt_3_selected_as_final_candidate"
        )
        is not False
        or report.get("fitting_and_validation_boundary", {}).get(
            "sealed_validation_seed_generated_or_read"
        )
        is not False
    ):
        raise ValueError("attempt-3 outcome or fitting boundary changed")


def _validate_intent(
    intent: Mapping[str, Any],
    report: Mapping[str, Any],
) -> None:
    expected_header = {
        "schema": (
            "atomos.v5.scale-orbit.post-score-all-four-hard-view-supervised-"
            "pair-adaptation-intent"
        ),
        "schema_version": 1,
        "status": (
            "frozen_before_attempt_4_trainer_or_assembler_change_training_or_"
            "inference"
        ),
        "development_only": True,
        "release_evidence": False,
        "append_only": True,
        "adaptation_attempt": 4,
        "parent_source_commit": SOURCE_COMMIT,
        "parent_source_commit_subject": SOURCE_SUBJECT,
        "parent_source_commit_parent": SOURCE_PARENT,
    }
    if any(intent.get(key) != value for key, value in expected_header.items()):
        raise ValueError("attempt-4 intent header changed")
    bindings = intent.get("parent_bindings", {})
    if (
        bindings.get("attempt_3_intent_sha256")
        != EXPECTED_CONTRACT_HASHES[ATTEMPT_3_PATH]
        or bindings.get("attempt_3_miss_report_sha256")
        != EXPECTED_CONTRACT_HASHES[ATTEMPT_3_MISS_PATH]
        or bindings.get("recovery_protocol_sha256")
        != EXPECTED_CONTRACT_HASHES[PROTOCOL_PATH]
        or bindings.get("seed_registry_sha256")
        != EXPECTED_CONTRACT_HASHES[SEED_REGISTRY_PATH]
        or bindings.get("openset_novelty_adaptive_amendment_sha256")
        != EXPECTED_CONTRACT_HASHES[NOVELTY_PATH]
    ):
        raise ValueError("attempt-4 parent bindings changed")
    ancestry = intent.get("source_ancestry_contract", {})
    if (
        ancestry.get("attempt_3_preregistration_commit") != PREREG_COMMIT
        or ancestry.get("attempt_3_source_commit") != SOURCE_COMMIT
        or ancestry.get("attempt_3_source_commit_direct_parent")
        != SOURCE_PARENT
        or ancestry.get("attempt_3_source_trainer_blob_sha256")
        != TRAINER_SHA256
        or ancestry.get("attempt_3_source_assembler_blob_sha256")
        != ASSEMBLER_SHA256
    ):
        raise ValueError("attempt-4 source ancestry binding changed")

    change = intent.get("predeclared_change", {})
    identity = change.get("identity_sampling_inherited_exactly_from_attempt_3")
    rng = change.get("legacy_pair_rng_trajectory_preservation")
    emission = change.get("all_four_view_emission")
    phase = change.get("phase_augmentation")
    bn = change.get(
        "batch_norm_stat_firewall_inherited_exactly_from_attempt_3"
    )
    auxiliary = change.get("supervised_pair_auxiliary")
    cross_entropy = (
        auxiliary.get("cross_entropy")
        if isinstance(auxiliary, Mapping)
        else None
    )
    prototypes = (
        auxiliary.get("public_class_prototypes")
        if isinstance(auxiliary, Mapping)
        else None
    )
    logits = (
        auxiliary.get("logits")
        if isinstance(auxiliary, Mapping)
        else None
    )
    if (
        change.get("only_changed_axis")
        != "supervised_pair_views_per_selected_profile_identity"
        or change.get("decision_consensus")
        != (
            "unanimous_after_independent_attempt_3_metric_and_contract_"
            "review"
        )
        or change.get(
            "attempt_3_two_sampled_views_replaced_by_all_four_ordered_views"
        )
        is not True
        or change.get(
            "attempt_3_identity_population_selection_order_targets_logits_"
            "coefficient_and_reduction_family_changed"
        )
        is not False
        or not isinstance(identity, Mapping)
        or identity.get("population_seed") != 20264101
        or identity.get("literal_profile_count") != 31
        or identity.get("identities_per_profile_addressable") != 40
        or identity.get("selected_identity_count_per_episode") != 31
        or not isinstance(rng, Mapping)
        or rng.get("pair_rng_seed_for_seed20260740") != 19983770
        or rng.get("per_profile_operation_order")
        != [
            "draw one identity index uniformly using rng.integers(0, len(entries))",
            "call rng.choice(4, size=2, replace=false) exactly as attempt 3",
            "validate that the compatibility draw has shape [2] and distinct indices",
            "discard both compatibility-draw values without using them to choose emitted views",
            "emit all four selected-identity view positions in frozen scale-factor order",
        ]
        or rng.get(
            "compatibility_draw_preserves_attempt_3_pair_rng_call_count_order_"
            "and_identity_trajectory"
        )
        is not True
        or not isinstance(emission, Mapping)
        or emission.get("physical_scale_factors_in_exact_emission_order")
        != SCALE_FACTORS
        or emission.get("positions_shape") != [31, 4]
        or emission.get("views_per_episode") != 124
        or emission.get("no_scale_view_is_sampled_dropped_repeated_or_reordered")
        is not True
        or not isinstance(phase, Mapping)
        or phase.get("independent_phase_draw_count_per_episode") != 124
        or phase.get("views_within_one_identity_share_phase_draw") is not False
        or not isinstance(bn, Mapping)
        or bn.get("auxiliary_forward_uses_batch_norm_eval_mode") is not True
        or bn.get("dropout_remains_live_during_auxiliary_forward") is not True
        or not isinstance(auxiliary, Mapping)
        or auxiliary.get("paired_embedding_count") != 124
        or auxiliary.get("paired_logits_shape") != [31, 4, 7]
        or auxiliary.get("paired_targets_shape") != [31, 4]
        or auxiliary.get("target_construction_symbolic")
        != "repeat(profile_public_class_indices, 4)"
        or auxiliary.get("paired_targets_sha256") != ATTEMPT_4_TARGET_SHA256
        or auxiliary.get("auxiliary_coefficient") != 0.2
        or auxiliary.get("cosine_distance_term") is not False
        or not isinstance(cross_entropy, Mapping)
        or cross_entropy.get("label_smoothing") != 0.0
        or cross_entropy.get("per_view_reduction") != "none"
        or cross_entropy.get(
            "per_view_loss_shape_before_profile_reduction"
        )
        != [31, 4]
        or cross_entropy.get("within_profile_reduction")
        != "maximum_over_exactly_four_contiguous_scale_view_losses"
        or cross_entropy.get("across_profile_mean_denominator") != 31
        or cross_entropy.get(
            "maximum_is_not_taken_over_profiles_classes_or_batch_episodes"
        )
        is not True
        or cross_entropy.get(
            "all_four_view_losses_participate_in_forward_before_maximum"
        )
        is not True
        or not isinstance(prototypes, Mapping)
        or prototypes.get("detached_for_auxiliary") is not True
        or not isinstance(logits, Mapping)
        or logits.get("logit_scale_detached_for_auxiliary") is not True
    ):
        raise ValueError("attempt-4 all-four hard-view objective changed")

    run = intent.get("inherited_run_contract", {})
    if (
        run.get("seed") != 20260740
        or run.get("episodes") != 8000
        or run.get("evaluation_interval") != 500
        or run.get("current_source_share", {}).get("numerator") != 1
        or run.get("current_source_share", {}).get("denominator") != 3
        or run.get("branch_fusion", {}).get("weight_real") != 0.5
        or run.get("branch_fusion", {}).get("weight_complex") != 0.5
        or not isinstance(run.get("checkpoint_score_contract"), str)
    ):
        raise ValueError("attempt-4 inherited run/checkpoint/fusion tuple changed")
    metadata = intent.get("future_implementation_metadata_contract", {})
    if metadata != {
        "branch_run_configuration_schema":
            "v5-scale-orbit-development-training-v4",
        "fusion_run_configuration_schema":
            "v5-scale-orbit-development-fusion-v4",
        "attempt_3_training_v3_branch_metadata_must_be_rejected": True,
        "attempt_3_fusion_v3_metadata_must_be_rejected": True,
        "source_snapshots_must_bind_attempt_3_miss_report_and_attempt_4_intent":
            True,
    }:
        raise ValueError("attempt-4 future metadata schema contract changed")
    no_other = intent.get("no_other_change_contract")
    if (
        not isinstance(no_other, Mapping)
        or not no_other
        or any(value is not False for value in no_other.values())
    ):
        raise ValueError("attempt-4 no-other-change contract changed")
    firewall = intent.get("fitting_firewall", {})
    if (
        firewall.get(
            "supervised_views_from_seed20264101_current_train_role_only"
        )
        is not True
        or firewall.get(
            "seed20262904_selection_metrics_may_retain_the_existing_"
            "development_checkpoint_and_candidate_selection_role"
        )
        is not True
        or any(
            firewall.get(key) != 0
            for key in (
                "seed20264101_enrollment_rows_used_for_supervised_auxiliary",
                "seed20262904_selection_rows_used_for_any_gradient_or_optimizer_step",
                "seed20262904_selection_rows_used_for_weight_center_or_feature_moment_fit",
                "seed20262904_selection_rows_used_for_persistent_prototype_fit",
                "seed20262904_selection_rows_used_for_open_set_threshold_or_rank_fit",
                "sealed_rows_used_for_any_fit_checkpoint_or_candidate_selection",
            )
        )
    ):
        raise ValueError("attempt-4 fitting firewall changed")
    gate = intent.get("full_gate_boundary", {})
    timing = intent.get("timing_and_validation_boundary", {})
    if (
        gate.get("attempt_3_known_21_gate_status") != "not_evaluated"
        or gate.get("attempt_3_known_plus_novelty_27_gate_status")
        != "not_evaluated"
        or gate.get("attempt_4_requires_a_new_exact_21_known_gate_run")
        is not True
        or gate.get(
            "attempt_4_requires_a_new_exact_27_known_plus_novelty_gate_run"
        )
        is not True
        or timing.get("attempt_4_trainer_source_changed_before_this_intent")
        is not False
        or timing.get("attempt_4_assembler_source_changed_before_this_intent")
        is not False
        or timing.get("attempt_4_training_started_before_this_intent")
        is not False
        or timing.get("attempt_4_inference_started_before_this_intent")
        is not False
        or timing.get("sealed_seed_generated_or_read") is not False
    ):
        raise ValueError("attempt-4 timing or 21/27-gate boundary changed")
    if report.get("outcome", {}).get(
        "attempt_3_retained_as_parent_for_attempt_4"
    ) is not True:
        raise ValueError("attempt-4 does not bind its declared miss outcome")


def validate_documents(
    intent: Mapping[str, Any],
    report: Mapping[str, Any],
    attempt_3: Mapping[str, Any],
    protocol: Mapping[str, Any],
    novelty: Mapping[str, Any],
    metrics: Mapping[str, Mapping[str, Any]],
    metric_hashes: Mapping[str, str],
) -> dict[str, Any]:
    _validate_metrics(metrics, metric_hashes)
    _validate_report(report, metrics)
    _validate_intent(intent, report)
    if (
        attempt_3.get("adaptation_attempt") != 3
        or attempt_3.get("status")
        != (
            "frozen_before_attempt_3_trainer_or_assembler_change_training_or_"
            "inference"
        )
        or novelty.get("development_only") is not True
        or novelty.get("release_evidence") is not False
    ):
        raise ValueError("prior intent or novelty contract changed")
    inherited_gates = protocol.get("gate_contract", {}).get(
        "inherited_gates"
    )
    if not isinstance(inherited_gates, list) or len(inherited_gates) != 19:
        raise ValueError("inherited 19-gate protocol changed")
    return {
        "valid": True,
        "attempt_3_subthreshold_profile_scale_cell_count": 1,
        "attempt_3_subthreshold_profile_agreement_count": 2,
        "known_gate_count": 21,
        "full_gate_count": 27,
    }


def validate_repository_attempt_4() -> dict[str, Any]:
    for path, expected in EXPECTED_CONTRACT_HASHES.items():
        observed = _sha256(path)
        if observed != expected:
            raise ValueError(
                f"{path.name} SHA-256 changed: {observed}"
            )
    metric_hashes = {}
    for name, path in METRIC_PATHS.items():
        observed = _sha256(path)
        expected = METRIC_BINDINGS[name][1]
        if observed != expected:
            raise ValueError(f"{name} raw dev_metrics SHA-256 changed")
        metric_hashes[name] = observed

    documents = {
        path: _read_json(path) for path in EXPECTED_CONTRACT_HASHES
    }
    metrics = {
        name: _read_json(path) for name, path in METRIC_PATHS.items()
    }
    summary = validate_documents(
        documents[ATTEMPT_4_PATH],
        documents[ATTEMPT_3_MISS_PATH],
        documents[ATTEMPT_3_PATH],
        documents[PROTOCOL_PATH],
        documents[NOVELTY_PATH],
        metrics,
        metric_hashes,
    )

    source_parent = _git(
        "show", "-s", "--format=%P", SOURCE_COMMIT
    ).stdout.strip()
    if source_parent != SOURCE_PARENT:
        raise ValueError("attempt-3 source direct parent changed")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", PREREG_COMMIT, SOURCE_COMMIT],
        cwd=REPO,
    ).returncode != 0:
        raise ValueError("attempt-3 preregistration ancestry failed")
    for relative, expected in COMMITTED_SOURCE_BLOB_BINDINGS.items():
        blob = subprocess.run(
            ["git", "show", f"{SOURCE_COMMIT}:{relative}"],
            cwd=REPO,
            check=True,
            capture_output=True,
        ).stdout
        if _sha256_bytes(blob) != expected:
            raise ValueError(f"committed source blob changed: {relative}")

    head = _git("rev-parse", "HEAD").stdout.strip()
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, head],
        cwd=REPO,
    ).returncode != 0:
        raise ValueError("attempt-3 source is not an ancestor of HEAD")
    relative_intent = str(ATTEMPT_4_PATH.relative_to(REPO))
    intent_commit = _git(
        "log", "-1", "--format=%H", "--", relative_intent
    ).stdout.strip()
    deferred = not intent_commit
    committed_intent_reads = 0
    if not deferred:
        if subprocess.run(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                SOURCE_COMMIT,
                intent_commit,
            ],
            cwd=REPO,
        ).returncode != 0:
            raise ValueError(
                "attempt-3 source is not an ancestor of attempt-4 intent"
            )
        blob = subprocess.run(
            ["git", "show", f"{intent_commit}:{relative_intent}"],
            cwd=REPO,
            check=True,
            capture_output=True,
        ).stdout
        committed_intent_reads = 1
        if _sha256_bytes(blob) != EXPECTED_CONTRACT_HASHES[ATTEMPT_4_PATH]:
            raise ValueError("committed attempt-4 intent blob changed")

    summary.update(
        {
            "attempt_3_preregistration_to_source_ancestry_verified": True,
            "attempt_3_source_to_repository_head_ancestry_verified": True,
            "attempt_4_intent_commit_ancestry_deferred_until_commit":
                deferred,
            "development_metric_files_read": len(METRIC_PATHS),
            "committed_source_blobs_read":
                len(COMMITTED_SOURCE_BLOB_BINDINGS),
            "committed_attempt_4_intent_blobs_read":
                committed_intent_reads,
            "corpus_files_read": 0,
            "checkpoint_files_read": 0,
            "model_inference_runs": 0,
        }
    )
    return summary


if __name__ == "__main__":
    print(
        json.dumps(
            validate_repository_attempt_4(),
            indent=2,
            sort_keys=True,
        )
    )
