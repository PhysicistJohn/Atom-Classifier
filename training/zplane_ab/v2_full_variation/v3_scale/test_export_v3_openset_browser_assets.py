"""Unit tests for the v3 staged open-set browser exporter.

These cover the deterministic, cheap pieces: seed hygiene, payload contracts,
prefilter conversion, and the raw-LOF decomposition agreeing with the shipped
scorer.  The heavy end-to-end export runs as a script and self-checks every
row against ``fit_v3_openset.score_rows`` before writing anything.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

import export_v3_openset_browser_assets as exporter
import fit_v3_openset_staged as staged
import noise_prefilter
from known_only_patch_openset import KnownOnlyLOFOpenSet


def synthetic_composite(
    stage_two_threshold: float = 0.9,
    survivors: int = 24,
    gated: int = 3,
) -> staged.CompositeSurvivorPolicy:
    rng = np.random.default_rng(23)
    composite_calibration = np.sort(rng.uniform(0.0, 0.999, size=survivors))
    return staged.CompositeSurvivorPolicy(
        stage_one_calibration_raw=np.sort(rng.normal(size=survivors)),
        composite_calibration_raw=composite_calibration,
        threshold=float(
            np.quantile(
                composite_calibration, staged.COMPOSITE_THRESHOLD_QUANTILE
            )
        ),
        stage_two_threshold=float(stage_two_threshold),
        enrollment_rows=survivors + gated,
        enrollment_gated_rows=gated,
        enrollment_capture_length=16384,
    )


def synthetic_prefilter(
    threshold: float = 1.0,
    capture_length: int = 4096,
) -> noise_prefilter.NoisePrefilter:
    width = len(noise_prefilter.PREFILTER_FEATURES)
    return noise_prefilter.NoisePrefilter(
        feature_names=noise_prefilter.PREFILTER_FEATURES,
        capture_length=capture_length,
        mean=np.linspace(-1.0, 1.0, width),
        scale=np.linspace(0.5, 2.0, width),
        coefficients=np.linspace(-0.3, 0.4, width),
        intercept=0.25,
        threshold_score=threshold,
        fit_provenance={"synthetic": True},
        threshold_provenance={"synthetic": True},
    )


def valid_staged_report() -> dict:
    stage_two_threshold = 0.812345
    gates = {
        name: {
            "floor": float(floor),
            "worst": float(floor) + 0.01,
            "passes": True,
        }
        for name, floor in staged.GATE_FLOORS.items()
    }
    gates["known_false_unknown_rate"] = {
        "ceiling": float(staged.KNOWN_FUR_CEILING),
        "worst": 80 / 1908,
        "passes": True,
    }
    attribution = {
        "rows": 1908,
        "staged_threshold": float(
            staged.FROZEN_POLICY_V4_Q97_ENROLLMENT_THRESHOLD
        ),
        "unstaged_threshold": stage_two_threshold,
        "staged_false_unknown_rate": 80 / 1908,
        "stage_one_gate_rate": 21 / 1908,
        "stage_two_false_unknown_rate_marginal": 59 / 1908,
        "stage_two_false_unknown_rate_among_survivors": 59 / 1887,
        "unstaged_false_unknown_rate": 89 / 1908,
        "staged_rejected_total": 80,
        "rejected_by_stage_one_total": 21,
        "stage_one_also_rejected_by_unstaged_control": 2,
        "rejected_by_stage_two_only": 59,
        "attribution_partition_total": 80,
        "attribution_partition_exact": True,
        "attribution": exporter.KNOWN_ATTRIBUTION_TEXT,
    }
    novelty = {
        str(seed): {
            str(length): {
                "overall": {
                    "auroc": gates["overall_auroc"]["worst"],
                },
                "noise": {
                    "auroc": gates["noise_auroc"]["worst"],
                    "threshold_recall": gates[
                        "noise_threshold_recall"
                    ]["worst"],
                    "gated_at_stage_one_fraction": 0.05,
                    "rejected_at_stage_two_fraction": (
                        gates["noise_threshold_recall"]["worst"] - 0.05
                    ),
                    "compute": {
                        "rows": 300,
                        "short_circuited": 15,
                        "short_circuited_fraction": 0.05,
                        "stage_two_rows_scored_staged": 285,
                        "stage_two_rows_scored_unstaged": 300,
                        "prefilter_module_compute_saving": {
                            "rows": 300,
                            "gated_rows": 15,
                            "downstream_rows_evaluated": 285,
                            "gated_fraction": 0.05,
                            "measured": True,
                        },
                    },
                },
                "chirp": {
                    "auroc": gates["chirp_auroc"]["worst"],
                    "threshold_recall": gates[
                        "chirp_threshold_recall"
                    ]["worst"],
                    "gated_at_stage_one_fraction": 0.05,
                    "rejected_at_stage_two_fraction": (
                        gates["chirp_threshold_recall"]["worst"] - 0.05
                    ),
                    "compute": {
                        "rows": 300,
                        "short_circuited": 15,
                        "short_circuited_fraction": 0.05,
                        "stage_two_rows_scored_staged": 285,
                        "stage_two_rows_scored_unstaged": 300,
                        "prefilter_module_compute_saving": {
                            "rows": 300,
                            "gated_rows": 15,
                            "downstream_rows_evaluated": 285,
                            "gated_fraction": 0.05,
                            "measured": True,
                        },
                    },
                },
                "known_false_unknown_rate": gates[
                    "known_false_unknown_rate"
                ]["worst"],
                "known_false_unknown_by_stage": dict(attribution),
            }
            for length in exporter.REQUIRED_VALIDATION_PREFIX_LENGTHS
        }
        for seed in exporter.VALIDATION_NOVELTY_SEEDS
    }
    return {
        "status": "development_openset_pass",
        "role": "validate",
        "gates_are_evidence": True,
        "development_only": True,
        "release_evidence": False,
        "sealed_release_data_used": 0,
        "consumed_test_rows_used": 0,
        "sealed_release_paths_read": 0,
        "release_seed_20260736_used": False,
        "all_pass": True,
        "closed_label_can_be_gated": True,
        "additive_only": False,
        "changes_closed_label": True,
        "gates_before_classification": True,
        "architecture": {
            "kind": staged.STAGED_POLICY_KIND,
            "schema": staged.STAGED_POLICY_SCHEMA,
            "staged_policy_version": staged.STAGED_POLICY_VERSION,
            "stage_two_policy_kind": exporter.FROZEN_POLICY_KIND,
            "stage_one_fitted_here": False,
            "stage_two_refit_on_survivors": False,
        },
        "gates": gates,
        "stage_one": {
            "capture_lengths": list(exporter.FIXTURE_LENGTHS),
        },
        "protocol": {
            "prefix_lengths": list(
                exporter.REQUIRED_VALIDATION_PREFIX_LENGTHS
            ),
            "stage_one_feature_length_by_prefix_length": {
                "4096": 4096,
                "8192": 8192,
                "16384": 16384,
                "32768": 16384,
            },
            "stage_one_causal_prefix_rule_lengths": [32768],
            "known_classes": 7,
            "known_rows": 1908,
            "novelty_n_each": 300,
            "novelty_families": ["noise", "chirp"],
            "stage_one_features_verified_against_stage_two_captures": True,
        },
        "seeds": {
            "design_novelty_seed": exporter.DESIGN_NOVELTY_SEED,
            "default_validation_novelty_seeds": list(
                exporter.VALIDATION_NOVELTY_SEEDS
            ),
            "novelty_seeds": list(exporter.VALIDATION_NOVELTY_SEEDS),
            "release_seed_not_spent": exporter.RELEASE_SEED_NOT_SPENT,
        },
        "stage_two": {
            "kind": exporter.FROZEN_POLICY_KIND,
            "threshold_quantile": float(exporter.FROZEN_THRESHOLD_QUANTILE),
            "threshold": stage_two_threshold,
        },
        "composite": {
            "schema": staged.STAGED_POLICY_SCHEMA,
            "kind": staged.STAGED_POLICY_KIND,
            "policy_version": staged.STAGED_POLICY_VERSION,
            "survivor_score": staged.COMPOSITE_SURVIVOR_SCORE,
            "threshold_quantile": staged.COMPOSITE_THRESHOLD_QUANTILE,
            "threshold": staged.FROZEN_POLICY_V4_Q97_ENROLLMENT_THRESHOLD,
            "stage_two_threshold_unchanged": stage_two_threshold,
            "threshold_population": (
                "enrollment stage-1 survivors only (enrollment only, as every "
                "rank and threshold in this chain)"
            ),
            "training_rows_used_for_threshold": 0,
            "selection_rows_used_for_threshold": 0,
            "novelty_rows_used_for_threshold": 0,
            "release_rows_used_for_threshold": 0,
            "stage_one_known_false_positive_budget": (
                staged.STAGE_ONE_KNOWN_FALSE_POSITIVE_BUDGET
            ),
            "survivor_known_false_positive_budget": (
                staged.SURVIVOR_KNOWN_FALSE_POSITIVE_BUDGET
            ),
            "nominal_enrollment_false_unknown_budget": (
                staged.NOMINAL_ENROLLMENT_FALSE_UNKNOWN_BUDGET
            ),
            "only_policy_change": staged.STAGED_POLICY_SPEC.only_policy_change,
            "stage_one_changed": False,
            "rejector_cnn_fusion_changed": False,
            "classifier_cnn_fusion_changed": False,
            "gate_floors_and_known_fur_ceiling_unchanged": True,
            "enrollment_rows": 100,
            "enrollment_gated_rows": 2,
            "enrollment_survivor_rows": 98,
            "enrollment_capture_length": 16384,
            "enrollment_stage_one_gate_rate": 0.02,
        },
        "known": {
            "false_unknown_rate": gates["known_false_unknown_rate"]["worst"],
            "by_stage": attribution,
        },
        "novelty": novelty,
    }


def evaluated_known_rows(
    *,
    stage_one_total: int = 21,
    stage_two_total: int = 59,
) -> tuple[staged.StagedOutcome, np.ndarray]:
    rows = 1908
    gated = np.zeros(rows, dtype=bool)
    gated[:stage_one_total] = True
    survivors = ~gated
    composite_score = np.zeros(rows, dtype=np.float64)
    composite_score[gated] = np.nan
    stage_two_indices = np.arange(
        stage_one_total,
        stage_one_total + stage_two_total,
    )
    composite_score[stage_two_indices] = 0.99
    stage_two_score = np.where(gated, np.nan, composite_score)
    survivor_rank = np.where(gated, np.nan, 0.0)
    stage_one_rank = np.full(rows, 0.5, dtype=np.float64)
    staged_score = np.where(gated, 1.5, composite_score)
    outcome = staged.StagedOutcome(
        gated=gated,
        stage_one_raw=np.zeros(rows, dtype=np.float64),
        stage_one_rank=stage_one_rank,
        stage_two_score=stage_two_score,
        stage_one_survivor_rank=survivor_rank,
        composite_score=composite_score,
        staged_score=staged_score,
        survivor_index=np.flatnonzero(survivors),
        feature_seconds=0.0,
        stage_one_seconds=0.0,
        stage_two_seconds=0.0,
    )
    unstaged = np.zeros(rows, dtype=np.float64)
    unstaged[:2] = 0.9
    unstaged[stage_one_total : stage_one_total + 87] = 0.9
    return outcome, unstaged


class ParitySeedHygiene(unittest.TestCase):
    def test_default_seed_is_accepted(self) -> None:
        self.assertEqual(
            exporter.validate_parity_seed(exporter.PARITY_NOVELTY_SEED),
            exporter.PARITY_NOVELTY_SEED,
        )

    def test_novelty_namespace_is_refused(self) -> None:
        for seed in (20260942, 20260999, 20260900):
            with self.assertRaises(ValueError):
                exporter.validate_parity_seed(seed)

    def test_fitting_band_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            exporter.validate_parity_seed(20261001)

    def test_release_and_sealed_seeds_are_refused(self) -> None:
        for seed in (20260731, 20260729, 20260938):
            with self.assertRaises(ValueError):
                exporter.validate_parity_seed(seed)

    def test_seed_equality_aliases_are_refused(self) -> None:
        for seed in (
            float(exporter.PARITY_NOVELTY_SEED),
            str(exporter.PARITY_NOVELTY_SEED),
            True,
        ):
            with self.subTest(seed=seed):
                with self.assertRaisesRegex(ValueError, "exact integer"):
                    exporter.validate_parity_seed(seed)


class StagedEvidenceAdmission(unittest.TestCase):
    def test_passing_validate_role_strict_evidence_is_accepted(self) -> None:
        exporter.validate_staged_evidence_report(valid_staged_report())

    def test_design_artifact_is_refused_even_if_its_metrics_pass(self) -> None:
        report = valid_staged_report()
        report["role"] = "design"
        report["status"] = "design_selection_pass"
        report["gates_are_evidence"] = False
        with self.assertRaisesRegex(ValueError, "validate-role evidence"):
            exporter.validate_staged_evidence_report(report)

    def test_any_failed_gate_is_refused(self) -> None:
        report = valid_staged_report()
        report["gates"]["noise_auroc"]["passes"] = False
        with self.assertRaisesRegex(ValueError, "noise_auroc"):
            exporter.validate_staged_evidence_report(report)

    def test_gate_claiming_pass_below_its_floor_is_refused(self) -> None:
        for name, floor in staged.GATE_FLOORS.items():
            with self.subTest(gate=name):
                report = valid_staged_report()
                report["gates"][name]["worst"] = float(floor) - 0.01
                report["gates"][name]["passes"] = True
                with self.assertRaisesRegex(ValueError, name):
                    exporter.validate_staged_evidence_report(report)

    def test_gate_worst_must_be_derived_from_novelty_rows(self) -> None:
        report = valid_staged_report()
        report["novelty"]["20260953"]["4096"]["noise"]["auroc"] = 0.0
        with self.assertRaisesRegex(ValueError, "noise_auroc.*not derived"):
            exporter.validate_staged_evidence_report(report)

    def test_known_fur_must_be_derived_from_evidence_rows(self) -> None:
        report = valid_staged_report()
        report["novelty"]["20260953"]["4096"][
            "known_false_unknown_rate"
        ] = 1.0
        with self.assertRaisesRegex(ValueError, "not derived"):
            exporter.validate_staged_evidence_report(report)

    def test_nonfinite_gate_worst_is_refused(self) -> None:
        for name, value in (
            ("noise_auroc", float("nan")),
            ("chirp_threshold_recall", float("inf")),
            ("known_false_unknown_rate", float("nan")),
        ):
            with self.subTest(gate=name):
                report = valid_staged_report()
                report["gates"][name]["worst"] = value
                with self.assertRaises(ValueError):
                    exporter.validate_staged_evidence_report(report)

    def test_string_floor_or_ceiling_is_refused(self) -> None:
        for name, bound in (
            ("noise_auroc", "floor"),
            ("known_false_unknown_rate", "ceiling"),
        ):
            with self.subTest(gate=name):
                report = valid_staged_report()
                report["gates"][name][bound] = str(
                    report["gates"][name][bound]
                )
                with self.assertRaises(ValueError):
                    exporter.validate_staged_evidence_report(report)

    def test_relaxed_known_fur_ceiling_is_refused(self) -> None:
        report = valid_staged_report()
        report["gates"]["known_false_unknown_rate"] = {
            "ceiling": 0.12,
            "worst": 0.11,
            "passes": True,
        }
        with self.assertRaisesRegex(ValueError, "strict development ceiling"):
            exporter.validate_staged_evidence_report(report)

    def test_missing_n32768_causal_prefix_validation_is_refused(self) -> None:
        report = valid_staged_report()
        report["protocol"]["prefix_lengths"] = [4096, 8192, 16384]
        with self.assertRaisesRegex(ValueError, "validation prefixes"):
            exporter.validate_staged_evidence_report(report)

    def test_wrong_n32768_feature_length_is_refused(self) -> None:
        report = valid_staged_report()
        report["protocol"]["stage_one_feature_length_by_prefix_length"][
            "32768"
        ] = 32768
        with self.assertRaisesRegex(ValueError, "N32768 -> N16384"):
            exporter.validate_staged_evidence_report(report)

    def test_validation_seed_tuple_and_order_are_frozen(self) -> None:
        for field in ("novelty_seeds", "default_validation_novelty_seeds"):
            with self.subTest(field=field):
                report = valid_staged_report()
                report["seeds"][field] = list(
                    reversed(exporter.VALIDATION_NOVELTY_SEEDS)
                )
                with self.assertRaisesRegex(ValueError, "seed|pair"):
                    exporter.validate_staged_evidence_report(report)

    def test_design_and_release_seeds_are_frozen(self) -> None:
        for field, value in (
            ("design_novelty_seed", exporter.DESIGN_NOVELTY_SEED + 1),
            ("release_seed_not_spent", exporter.RELEASE_SEED_NOT_SPENT + 1),
        ):
            with self.subTest(field=field):
                report = valid_staged_report()
                report["seeds"][field] = value
                with self.assertRaisesRegex(ValueError, "seed"):
                    exporter.validate_staged_evidence_report(report)

    def test_composite_hygiene_cannot_be_weakened(self) -> None:
        mutations = {
            "threshold_population": "selection",
            "training_rows_used_for_threshold": 1,
            "selection_rows_used_for_threshold": 1,
            "novelty_rows_used_for_threshold": 1,
            "release_rows_used_for_threshold": 1,
            "stage_one_known_false_positive_budget": 0.02,
            "survivor_known_false_positive_budget": 0.04,
            "nominal_enrollment_false_unknown_budget": 0.05,
            "only_policy_change": "anything else",
            "stage_one_changed": True,
            "rejector_cnn_fusion_changed": True,
            "classifier_cnn_fusion_changed": True,
            "gate_floors_and_known_fur_ceiling_unchanged": False,
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                report = valid_staged_report()
                report["composite"][field] = value
                with self.assertRaisesRegex(ValueError, field):
                    exporter.validate_staged_evidence_report(report)

    def test_corrected_attribution_partition_is_required_everywhere(self) -> None:
        report = valid_staged_report()
        report["known"]["by_stage"]["rejected_by_stage_one_total"] = 1
        with self.assertRaisesRegex(ValueError, "partition"):
            exporter.validate_staged_evidence_report(report)

        report = valid_staged_report()
        report["novelty"]["20260953"]["4096"][
            "known_false_unknown_by_stage"
        ] = dict(report["known"]["by_stage"])
        report["novelty"]["20260953"]["4096"][
            "known_false_unknown_by_stage"
        ]["stage_one_also_rejected_by_unstaged_control"] = 0
        with self.assertRaisesRegex(ValueError, "attribution differs"):
            exporter.validate_staged_evidence_report(report)

    def test_novelty_stage_fractions_derive_threshold_recall(self) -> None:
        report = valid_staged_report()
        report["novelty"]["20260953"]["4096"]["noise"][
            "rejected_at_stage_two_fraction"
        ] = 0.0
        with self.assertRaisesRegex(ValueError, "derive threshold_recall"):
            exporter.validate_staged_evidence_report(report)

    def test_attribution_rows_are_bound_to_protocol_known_rows(self) -> None:
        report = valid_staged_report()
        bodies = [report["known"]["by_stage"]]
        bodies.extend(
            row["known_false_unknown_by_stage"]
            for by_length in report["novelty"].values()
            for row in by_length.values()
        )
        for body in bodies:
            body["rows"] = 1909
            body["staged_false_unknown_rate"] = 80 / 1909
            body["stage_one_gate_rate"] = 21 / 1909
            body["stage_two_false_unknown_rate_marginal"] = 59 / 1909
            body["stage_two_false_unknown_rate_among_survivors"] = 59 / 1888
            body["unstaged_false_unknown_rate"] = 89 / 1909
        report["known"]["false_unknown_rate"] = 80 / 1909
        report["gates"]["known_false_unknown_rate"]["worst"] = 80 / 1909
        for by_length in report["novelty"].values():
            for row in by_length.values():
                row["known_false_unknown_rate"] = 80 / 1909
        with self.assertRaisesRegex(
            ValueError, "known.by_stage.rows differs"
        ):
            exporter.validate_staged_evidence_report(report)

    def test_attribution_partition_is_bound_to_evaluated_known_rows(
        self,
    ) -> None:
        report = valid_staged_report()
        outcome, unstaged_score = evaluated_known_rows()
        exporter.bind_known_attribution_to_row_evaluation(
            report,
            outcome,
            unstaged_score,
            staged_threshold=float(
                staged.FROZEN_POLICY_V4_Q97_ENROLLMENT_THRESHOLD
            ),
            unstaged_threshold=0.812345,
        )

        # Keep the staged rejection total at 80 while moving one actual row
        # from stage two to stage one. JSON-only conservation still passes,
        # but the independently evaluated partition must not.
        forged_outcome, forged_unstaged = evaluated_known_rows(
            stage_one_total=22,
            stage_two_total=58,
        )
        with self.assertRaisesRegex(
            ValueError, "actual known-row evaluation"
        ):
            exporter.bind_known_attribution_to_row_evaluation(
                report,
                forged_outcome,
                forged_unstaged,
                staged_threshold=float(
                    staged.FROZEN_POLICY_V4_Q97_ENROLLMENT_THRESHOLD
                ),
                unstaged_threshold=0.812345,
            )

    def test_evaluated_known_row_vectors_are_strictly_aligned(self) -> None:
        report = valid_staged_report()
        base_outcome, unstaged_score = evaluated_known_rows()
        nan_raw = base_outcome.stage_one_raw.copy()
        nan_raw[0] = np.nan
        nan_composite = base_outcome.composite_score.copy()
        nan_composite[base_outcome.survivor_index[0]] = np.nan
        mutations = (
            (
                "reversed-survivor-index",
                replace(
                    base_outcome,
                    survivor_index=base_outcome.survivor_index[::-1],
                ),
                "survivor_index",
            ),
            (
                "float-survivor-index",
                replace(
                    base_outcome,
                    survivor_index=base_outcome.survivor_index.astype(
                        np.float64
                    ),
                ),
                "survivor_index",
            ),
            (
                "matrix-survivor-index",
                replace(
                    base_outcome,
                    survivor_index=base_outcome.survivor_index[:, None],
                ),
                "survivor_index",
            ),
            (
                "matrix-staged-score",
                replace(
                    base_outcome,
                    staged_score=base_outcome.staged_score[:, None],
                ),
                "staged_score",
            ),
            (
                "nan-stage-one",
                replace(base_outcome, stage_one_raw=nan_raw),
                "stage_one_raw",
            ),
            (
                "nan-survivor-composite",
                replace(base_outcome, composite_score=nan_composite),
                "composite_score",
            ),
        )
        for label, outcome, message in mutations:
            with self.subTest(label=label), self.assertRaisesRegex(
                ValueError, message
            ):
                exporter.bind_known_attribution_to_row_evaluation(
                    report,
                    outcome,
                    unstaged_score,
                    staged_threshold=float(
                        staged.FROZEN_POLICY_V4_Q97_ENROLLMENT_THRESHOLD
                    ),
                    unstaged_threshold=0.812345,
                )

    def test_novelty_fractions_are_exact_counts_over_n(self) -> None:
        report = valid_staged_report()
        report["novelty"]["20260953"]["4096"]["noise"][
            "gated_at_stage_one_fraction"
        ] = 0.0501
        with self.assertRaisesRegex(ValueError, "exact count/300"):
            exporter.validate_staged_evidence_report(report)

    def test_novelty_compute_counts_are_bound_to_fractions_and_n(self) -> None:
        report = valid_staged_report()
        report["novelty"]["20260953"]["4096"]["noise"]["compute"][
            "short_circuited"
        ] = 14
        with self.assertRaisesRegex(ValueError, "short_circuited"):
            exporter.validate_staged_evidence_report(report)

        report = valid_staged_report()
        saving = report["novelty"]["20260953"]["4096"]["chirp"]["compute"][
            "prefilter_module_compute_saving"
        ]
        saving["gated_rows"] = 14
        with self.assertRaisesRegex(ValueError, "gated_rows"):
            exporter.validate_staged_evidence_report(report)

    def test_attribution_thresholds_are_bound_to_their_distinct_policies(
        self,
    ) -> None:
        report = valid_staged_report()
        report["known"]["by_stage"]["staged_threshold"] = report["stage_two"][
            "threshold"
        ]
        with self.assertRaisesRegex(ValueError, "composite q97 threshold"):
            exporter.validate_staged_evidence_report(report)

        report = valid_staged_report()
        report["known"]["by_stage"]["unstaged_threshold"] = report[
            "composite"
        ]["threshold"]
        with self.assertRaisesRegex(ValueError, "stage-two q95 threshold"):
            exporter.validate_staged_evidence_report(report)

        report = valid_staged_report()
        report["composite"]["stage_two_threshold_unchanged"] = report[
            "composite"
        ]["threshold"]
        with self.assertRaisesRegex(ValueError, "stage-two q95 threshold"):
            exporter.validate_staged_evidence_report(report)

    def test_unstaged_rate_and_overlap_are_exactly_count_derived(self) -> None:
        report = valid_staged_report()
        report["known"]["by_stage"]["unstaged_false_unknown_rate"] = 0.081
        with self.assertRaisesRegex(ValueError, "exact row-count rate"):
            exporter.validate_staged_evidence_report(report)

        report = valid_staged_report()
        report["known"]["by_stage"][
            "unstaged_false_unknown_rate"
        ] = 1 / 1908
        report["known"]["by_stage"][
            "stage_one_also_rejected_by_unstaged_control"
        ] = 2
        with self.assertRaisesRegex(
            ValueError, "exceeds the derived unstaged rejection count"
        ):
            exporter.validate_staged_evidence_report(report)

    def test_evidence_scalar_equality_aliases_fail_closed(self) -> None:
        mutations = (
            (
                "boolean-as-int",
                lambda report: report.__setitem__("gates_are_evidence", 1),
            ),
            (
                "zero-count-as-bool",
                lambda report: report.__setitem__(
                    "sealed_release_data_used", False
                ),
            ),
            (
                "zero-count-as-float",
                lambda report: report.__setitem__(
                    "consumed_test_rows_used", 0.0
                ),
            ),
            (
                "architecture-schema-as-float",
                lambda report: report["architecture"].__setitem__(
                    "schema", float(staged.STAGED_POLICY_SCHEMA)
                ),
            ),
            (
                "gate-flag-as-int",
                lambda report: report["gates"]["noise_auroc"].__setitem__(
                    "passes", 1
                ),
            ),
            (
                "protocol-count-as-float",
                lambda report: report["protocol"].__setitem__(
                    "known_rows", 1908.0
                ),
            ),
            (
                "protocol-flag-as-int",
                lambda report: report["protocol"].__setitem__(
                    "stage_one_features_verified_against_stage_two_captures",
                    1,
                ),
            ),
            (
                "capture-length-as-float",
                lambda report: report["stage_one"].__setitem__(
                    "capture_lengths", [4096.0, 8192, 16384]
                ),
            ),
            (
                "feature-length-as-float",
                lambda report: report["protocol"][
                    "stage_one_feature_length_by_prefix_length"
                ].__setitem__("4096", 4096.0),
            ),
            (
                "design-seed-as-float",
                lambda report: report["seeds"].__setitem__(
                    "design_novelty_seed",
                    float(exporter.DESIGN_NOVELTY_SEED),
                ),
            ),
            (
                "validation-seed-as-float",
                lambda report: report["seeds"].__setitem__(
                    "novelty_seeds",
                    [
                        float(exporter.VALIDATION_NOVELTY_SEEDS[0]),
                        exporter.VALIDATION_NOVELTY_SEEDS[1],
                    ],
                ),
            ),
            (
                "release-seed-as-float",
                lambda report: report["seeds"].__setitem__(
                    "release_seed_not_spent",
                    float(exporter.RELEASE_SEED_NOT_SPENT),
                ),
            ),
            (
                "hygiene-zero-as-float",
                lambda report: report["composite"].__setitem__(
                    "training_rows_used_for_threshold", 0.0
                ),
            ),
            (
                "hygiene-zero-as-bool",
                lambda report: report["composite"].__setitem__(
                    "release_rows_used_for_threshold", False
                ),
            ),
            (
                "hygiene-flag-as-int",
                lambda report: report["composite"].__setitem__(
                    "stage_one_changed", 0
                ),
            ),
            (
                "enrollment-count-as-float",
                lambda report: report["composite"].__setitem__(
                    "enrollment_rows", 100.0
                ),
            ),
            (
                "attribution-count-as-float",
                lambda report: report["known"]["by_stage"].__setitem__(
                    "staged_rejected_total", 9.0
                ),
            ),
            (
                "attribution-flag-as-int",
                lambda report: report["known"]["by_stage"].__setitem__(
                    "attribution_partition_exact", 1
                ),
            ),
            (
                "nested-attribution-count-as-float",
                lambda report: report["novelty"]["20260953"]["4096"][
                    "known_false_unknown_by_stage"
                ].__setitem__("rows", 100.0),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                report = valid_staged_report()
                mutate(report)
                with self.assertRaises(ValueError):
                    exporter.validate_staged_evidence_report(report)


class BrowserFusionExportAdmission(unittest.TestCase):
    def write_export(
        self,
        directory: Path,
        *,
        manifest_schema_version: object = 2,
        weights_schema_version: object = 1,
    ) -> dict:
        role = exporter.REJECTOR_RUNTIME_ROLE
        weights_name = exporter.REJECTOR_WEIGHTS_NAME
        probe_name = exporter.REJECTOR_PROBE_NAME
        bundle_sha = "a" * 64
        weights = {
            "schema": (
                "atomos.v3.time-domain-invariant-fusion.browser-weights"
            ),
            "schema_version": weights_schema_version,
            "status": "staging_not_release",
            "runtime_role": role,
            "provenance": {
                "source_bundle_manifest_sha256": bundle_sha,
            },
        }
        weights_path = directory / weights_name
        weights_path.write_text(
            json.dumps(weights, allow_nan=False), encoding="utf-8"
        )
        probe_path = directory / probe_name
        probe_path.write_text("{}", encoding="utf-8")
        manifest = {
            "schema": exporter.FUSION_EXPORT_SCHEMA,
            "schema_version": manifest_schema_version,
            "runtime_role": role,
            "source_bundle_manifest_sha256": bundle_sha,
            "emitted": {
                weights_name: {
                    "bytes": weights_path.stat().st_size,
                    "sha256": exporter._sha256(weights_path),
                },
                probe_name: {
                    "bytes": probe_path.stat().st_size,
                    "sha256": exporter._sha256(probe_path),
                },
            },
        }
        (directory / exporter.FUSION_EXPORT_MANIFEST_NAME).write_text(
            json.dumps(manifest, allow_nan=False), encoding="utf-8"
        )
        return {
            "directory": directory,
            "expected_role": role,
            "expected_weights_name": weights_name,
            "expected_probe_name": probe_name,
            "expected_bundle_manifest_sha256": bundle_sha,
        }

    def test_exact_integer_schema_versions_are_admitted(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            kwargs = self.write_export(Path(temp))
            loaded = exporter.load_browser_fusion_export(**kwargs)
            self.assertEqual(
                loaded["weights_name"], exporter.REJECTOR_WEIGHTS_NAME
            )

    def test_schema_versions_are_exact_json_integers(self) -> None:
        mutations = (
            ("manifest-float", 2.0, 1),
            ("weights-bool", 2, True),
            ("weights-float", 2, 1.0),
        )
        for label, manifest_version, weights_version in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp:
                kwargs = self.write_export(
                    Path(temp),
                    manifest_schema_version=manifest_version,
                    weights_schema_version=weights_version,
                )
                with self.assertRaisesRegex(
                    ValueError, "schema|runtime role"
                ):
                    exporter.load_browser_fusion_export(**kwargs)


class LoadedThresholdBinding(unittest.TestCase):
    def test_report_thresholds_must_match_loaded_npz_state(self) -> None:
        report = valid_staged_report()
        policy = SimpleNamespace(threshold=report["stage_two"]["threshold"])
        composite = SimpleNamespace(
            threshold=report["composite"]["threshold"],
            stage_two_threshold=report["stage_two"]["threshold"],
        )
        exporter.bind_report_thresholds_to_loaded_state(
            report, policy, composite
        )

        report["stage_two"]["threshold"] = 0.7
        report["composite"]["stage_two_threshold_unchanged"] = 0.7
        for body in [report["known"]["by_stage"], *[
            row["known_false_unknown_by_stage"]
            for by_length in report["novelty"].values()
            for row in by_length.values()
        ]]:
            body["unstaged_threshold"] = 0.7
        with self.assertRaisesRegex(RuntimeError, "q95 threshold differs"):
            exporter.bind_report_thresholds_to_loaded_state(
                report, policy, composite
            )

    def test_report_composite_threshold_must_match_loaded_npz_state(self) -> None:
        report = valid_staged_report()
        policy = SimpleNamespace(threshold=report["stage_two"]["threshold"])
        composite = SimpleNamespace(
            threshold=report["composite"]["threshold"],
            stage_two_threshold=report["stage_two"]["threshold"],
        )
        report["composite"]["threshold"] = 0.7
        for body in [report["known"]["by_stage"], *[
            row["known_false_unknown_by_stage"]
            for by_length in report["novelty"].values()
            for row in by_length.values()
        ]]:
            body["staged_threshold"] = 0.7
        with self.assertRaisesRegex(RuntimeError, "q97 threshold differs"):
            exporter.bind_report_thresholds_to_loaded_state(
                report, policy, composite
            )


class StagedArtifactHashAdmission(unittest.TestCase):
    def test_exact_report_hashes_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            names = (
                "v3_branch_lof_components.npz",
                "v3_open_policy_stage_two.npz",
                staged.COMPOSITE_POLICY_FILENAME,
            )
            for index, name in enumerate(names):
                (directory / name).write_bytes(f"asset-{index}".encode())
            report = {
                "artifacts": {
                    name: exporter._sha256(directory / name)
                    for name in names
                }
            }
            self.assertEqual(
                exporter.verify_staged_artifact_hashes(report, directory),
                report["artifacts"],
            )
            (directory / names[0]).write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "SHA"):
                exporter.verify_staged_artifact_hashes(report, directory)


class RuntimeBundleAdmission(unittest.TestCase):
    def test_legacy_not_refit_bundle_is_refused_before_assets_are_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            exporter.write_json(
                bundle / "bundle_manifest.json",
                {
                    "kind": "v3-time-domain-centered-invariant-fusion",
                    "runtime_role": exporter.REJECTOR_RUNTIME_ROLE,
                    "development_only": True,
                    "rejection": {
                        "state": "unset",
                        "external_staged_policy_required_for_abstention": False,
                    },
                    "release_blockers": [
                        "open-set rejector is not refit against this fusion"
                    ],
                    "assets": {},
                },
            )
            with self.assertRaisesRegex(RuntimeError, "staged rejection"):
                exporter.load_bundle(
                    bundle,
                    expected_role=exporter.REJECTOR_RUNTIME_ROLE,
                )

    def test_anonymous_or_swapped_role_is_refused_first(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            exporter.write_json(
                bundle / "bundle_manifest.json",
                {
                    "kind": "v3-time-domain-centered-invariant-fusion",
                    "runtime_role": exporter.CLASSIFIER_RUNTIME_ROLE,
                    "development_only": True,
                    "assets": {},
                },
            )
            with self.assertRaisesRegex(ValueError, "runtime_role"):
                exporter.load_bundle(
                    bundle,
                    expected_role=exporter.REJECTOR_RUNTIME_ROLE,
                )


class PrefilterConversion(unittest.TestCase):
    def test_payload_carries_every_parameter(self) -> None:
        model = synthetic_prefilter()
        payload = exporter.prefilter_model_payload(model)
        self.assertEqual(payload["capture_length"], 4096)
        self.assertEqual(
            payload["feature_names"], list(noise_prefilter.PREFILTER_FEATURES)
        )
        np.testing.assert_array_equal(payload["mean"], model.mean)
        np.testing.assert_array_equal(payload["scale"], model.scale)
        np.testing.assert_array_equal(payload["coefficients"], model.coefficients)
        self.assertEqual(payload["intercept"], model.intercept)
        self.assertEqual(payload["threshold_score"], model.threshold_score)

    def test_unthresholded_bundle_is_refused(self) -> None:
        model = synthetic_prefilter(threshold=float("nan"))
        with self.assertRaises(ValueError):
            exporter.prefilter_model_payload(model)


class CausalPrefixParity(unittest.TestCase):
    @staticmethod
    def capture(length: int) -> np.ndarray:
        sample = np.arange(length, dtype=np.float64)
        envelope = 0.8 + 0.2 * np.cos(2.0 * np.pi * sample / 613.0)
        return envelope * np.exp(1j * 2.0 * np.pi * 0.017 * sample)

    def test_long_row_is_bit_identical_to_its_fitted_prefix(self) -> None:
        prefilters = {4096: synthetic_prefilter(capture_length=4096)}
        capture = self.capture(8192)
        long = exporter.stage_one_row(
            prefilters,
            capture,
            patch_length=64,
            target_frac=0.5,
        )
        prefix = exporter.stage_one_row(
            prefilters,
            capture[:4096],
            patch_length=64,
            target_frac=0.5,
        )
        self.assertEqual(long["capture_length"], 8192)
        self.assertEqual(long["evaluated_length"], 4096)
        self.assertIs(long["causal_prefix_applied"], True)
        self.assertIs(prefix["causal_prefix_applied"], False)
        np.testing.assert_array_equal(long["features"], prefix["features"])
        self.assertEqual(long["score"], prefix["score"])
        self.assertEqual(long["rank"], prefix["rank"])
        self.assertEqual(long["gated"], prefix["gated"])

    def test_uncovered_shorter_length_is_refused(self) -> None:
        prefilters = {4096: synthetic_prefilter(capture_length=4096)}
        with self.assertRaisesRegex(KeyError, "applies only above"):
            exporter.stage_one_row(
                prefilters,
                self.capture(2048),
                patch_length=64,
                target_frac=0.5,
            )

    def test_release_long_row_names_are_stable(self) -> None:
        self.assertEqual(
            exporter.CAUSAL_PREFIX_GATED_ROW_NAME,
            "causal-prefix-gated-N32768",
        )
        self.assertEqual(
            exporter.CAUSAL_PREFIX_SURVIVOR_ROW_NAME,
            "causal-prefix-survivor-N32768",
        )


class JsonDeterminism(unittest.TestCase):
    def test_paths_are_refused_in_payloads(self) -> None:
        from pathlib import Path

        with self.assertRaises(TypeError):
            exporter._jsonable({"path": Path("/tmp/x")})

    def test_numpy_values_become_plain_json(self) -> None:
        value = exporter._jsonable(
            {
                "array": np.asarray([1.5, 2.5], dtype=np.float32),
                "integer": np.int64(3),
                "flag": np.bool_(True),
            }
        )
        self.assertEqual(value["array"], [1.5, 2.5])
        self.assertEqual(value["integer"], 3)
        self.assertIs(value["flag"], True)

    def test_compact_encoding_has_no_indentation_and_is_deterministic(self) -> None:
        payload = {"z": [1, 2], "a": {"x": True}}
        raw = exporter.json_bytes(payload, compact=True)
        self.assertEqual(raw, b'{"a":{"x":true},"z":[1,2]}\n')


class RawLofDecomposition(unittest.TestCase):
    def test_raw_matches_scorer_rank_ordering(self) -> None:
        rng = np.random.default_rng(7)
        training = rng.normal(size=(64, 4))
        enrollment = rng.normal(size=(32, 4))
        scorer = KnownOnlyLOFOpenSet.fit(training, enrollment, neighbors=3)
        queries = rng.normal(size=(8, 4))
        raw = exporter.compute_branch_lof_raw(scorer, queries)
        rank = scorer.score(queries)
        # The rank is the empirical rank of exactly this raw value, so ranking
        # raw with the scorer's calibration must reproduce the scorer output.
        expected = np.searchsorted(scorer.calibration, raw, side="left") / (
            len(scorer.calibration) + 1.0
        )
        np.testing.assert_array_equal(rank, expected)


class ContractKeys(unittest.TestCase):
    def test_weights_payload_carries_the_architecture_contract(self) -> None:
        prefilters = {4096: synthetic_prefilter()}
        rng = np.random.default_rng(11)
        training = rng.normal(size=(40, 4))
        enrollment = rng.normal(size=(24, 4))
        components = [
            ("real", 0.4, KnownOnlyLOFOpenSet.fit(training, enrollment, neighbors=2)),
            (
                "complex",
                0.6,
                KnownOnlyLOFOpenSet.fit(
                    training, enrollment, neighbors=3
                ),
            ),
        ]

        class PolicyStub:
            geometry_feature_index = 0
            threshold = 0.9

            class geometry:  # noqa: N801 - mirrors the dataclass attribute
                class_mean = np.zeros((7, 1))
                class_scale = np.ones((7, 1))

            geometry_calibration_raw = np.sort(rng.normal(size=(24,)))
            combined_calibration_raw = np.sort(rng.normal(size=(24,)))

        composite = synthetic_composite(stage_two_threshold=0.9)
        payload = exporter.openset_weights_payload(
            prefilters,
            "0" * 64,
            components,
            PolicyStub(),
            composite,
            {"patch_length": 64, "patch_count": 16, "target_frac": 0.5,
             "packed_length": 1024},
            {"development_only": True},
        )
        contract = payload["contract"]
        for key in noise_prefilter.CONTRACT_KEYS_FOR_RELEASE_CLAIM:
            self.assertIn(key, contract)
        self.assertIs(contract["additive_only"], False)
        self.assertIs(contract["changes_closed_label"], True)
        self.assertIs(contract["gates_before_classification"], True)
        self.assertEqual(
            payload["stage_two"]["policy"]["branch_lof_rank_weight"], 0.8
        )
        self.assertEqual(payload["stage_two"]["policy"]["geometry_weight"], 0.2)
        self.assertEqual(
            payload["stage_two"]["policy"]["threshold_quantile"], 0.95
        )
        self.assertEqual(payload["schema_version"], 4)
        self.assertEqual(
            payload["stage_two"]["kind"], exporter.FROZEN_POLICY_KIND
        )


class CompositePayload(unittest.TestCase):
    def test_payload_mirrors_the_npz_spelling_and_constants(self) -> None:
        composite = synthetic_composite()
        payload = exporter.composite_policy_payload(composite)
        self.assertEqual(payload["schema"], staged.STAGED_POLICY_SCHEMA)
        self.assertEqual(payload["kind"], staged.STAGED_POLICY_KIND)
        self.assertEqual(
            payload["policy_version"], staged.STAGED_POLICY_VERSION
        )
        self.assertEqual(
            payload["survivor_score"], staged.COMPOSITE_SURVIVOR_SCORE
        )
        self.assertEqual(
            payload["threshold_quantile"],
            float(staged.COMPOSITE_THRESHOLD_QUANTILE),
        )
        self.assertEqual(payload["threshold"], composite.threshold)
        self.assertEqual(
            payload["stage_two_threshold"], composite.stage_two_threshold
        )
        np.testing.assert_array_equal(
            payload["stage_one_calibration_raw"],
            composite.stage_one_calibration_raw,
        )
        np.testing.assert_array_equal(
            payload["composite_calibration_raw"],
            composite.composite_calibration_raw,
        )
        self.assertEqual(
            payload["enrollment_rows"],
            payload["enrollment_gated_rows"]
            + len(payload["stage_one_calibration_raw"]),
        )
        # The stored threshold is exactly the frozen quantile of the stored
        # calibration, which is what both loaders (Python and TypeScript)
        # recompute and verify.
        self.assertEqual(
            payload["threshold"],
            float(
                np.quantile(
                    np.asarray(payload["composite_calibration_raw"]),
                    staged.COMPOSITE_THRESHOLD_QUANTILE,
                )
            ),
        )
        expected_hygiene = staged._policy_hygiene_expected(
            staged.STAGED_POLICY_SPEC
        )
        assert expected_hygiene is not None
        for name, value in expected_hygiene.items():
            self.assertEqual(payload[name], value)


class DualBindingContract(unittest.TestCase):
    def kwargs(self) -> dict:
        return {
            "frontend": {
                "version": "invariant-patch-time-domain-v1",
                "patch_length": 64,
                "patch_count": 16,
                "target_frac": 0.5,
                "packed_length": 1024,
                "uses_frequency_transform": False,
            },
            "rejector_asset_sha256": "1" * 64,
            "classifier_asset_sha256": "2" * 64,
            "openset_asset_sha256": "3" * 64,
            "rejector_bundle_manifest_sha256": "4" * 64,
            "classifier_bundle_manifest_sha256": "5" * 64,
            "rejector_fusion_directory_sha256": "6" * 64,
            "classifier_fusion_directory_sha256": "7" * 64,
            "staged_validation_report_sha256": "8" * 64,
            "staged_artifacts_sha256": {
                "v3_branch_lof_components.npz": "9" * 64,
                "v3_open_policy_stage_two.npz": "a" * 64,
                staged.COMPOSITE_POLICY_FILENAME: "b" * 64,
            },
            "novelty_seeds": list(exporter.VALIDATION_NOVELTY_SEEDS),
        }

    def payload(self) -> dict:
        return exporter.dual_binding_payload(**self.kwargs())

    def test_roles_and_execution_are_explicit(self) -> None:
        payload = self.payload()
        self.assertEqual(payload["schema"], exporter.DUAL_BINDING_SCHEMA)
        self.assertEqual(
            payload["execution_order"],
            [
                "stage_one_noise_gate",
                "rejector_known_unknown",
                "classifier_known_label",
            ],
        )
        self.assertEqual(
            payload["roles"]["rejector"]["runtime_role"],
            exporter.REJECTOR_RUNTIME_ROLE,
        )
        self.assertEqual(
            payload["roles"]["classifier"]["runtime_role"],
            exporter.CLASSIFIER_RUNTIME_ROLE,
        )
        self.assertTrue(
            payload["fail_closed"][
                "classifier_runs_only_after_rejector_acceptance"
            ]
        )
        self.assertEqual(
            payload["openset_policy"][
                "fitted_rejector_runtime_bundle_manifest_sha256"
            ],
            "4" * 64,
        )
        self.assertEqual(
            payload["validation"]["design_novelty_seed"],
            exporter.DESIGN_NOVELTY_SEED,
        )
        self.assertEqual(
            payload["validation"]["novelty_seeds"],
            list(exporter.VALIDATION_NOVELTY_SEEDS),
        )
        self.assertEqual(
            payload["validation"]["release_seed_not_spent"],
            exporter.RELEASE_SEED_NOT_SPENT,
        )

    def test_same_role_asset_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "assets must differ"):
            exporter.dual_binding_payload(
                frontend={
                    "version": "invariant-patch-time-domain-v1",
                    "patch_length": 64,
                    "patch_count": 16,
                    "target_frac": 0.5,
                    "packed_length": 1024,
                    "uses_frequency_transform": False,
                },
                rejector_asset_sha256="1" * 64,
                classifier_asset_sha256="1" * 64,
                openset_asset_sha256="3" * 64,
                rejector_bundle_manifest_sha256="4" * 64,
                classifier_bundle_manifest_sha256="5" * 64,
                rejector_fusion_directory_sha256="6" * 64,
                classifier_fusion_directory_sha256="7" * 64,
                staged_validation_report_sha256="8" * 64,
                staged_artifacts_sha256={
                    "v3_branch_lof_components.npz": "9" * 64,
                    "v3_open_policy_stage_two.npz": "a" * 64,
                    staged.COMPOSITE_POLICY_FILENAME: "b" * 64,
                },
                novelty_seeds=list(exporter.VALIDATION_NOVELTY_SEEDS),
            )

    def test_validation_seed_order_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "validation seeds"):
            exporter.dual_binding_payload(
                frontend={
                    "version": "invariant-patch-time-domain-v1",
                    "patch_length": 64,
                    "patch_count": 16,
                    "target_frac": 0.5,
                    "packed_length": 1024,
                    "uses_frequency_transform": False,
                },
                rejector_asset_sha256="1" * 64,
                classifier_asset_sha256="2" * 64,
                openset_asset_sha256="3" * 64,
                rejector_bundle_manifest_sha256="4" * 64,
                classifier_bundle_manifest_sha256="5" * 64,
                rejector_fusion_directory_sha256="6" * 64,
                classifier_fusion_directory_sha256="7" * 64,
                staged_validation_report_sha256="8" * 64,
                staged_artifacts_sha256={
                    "v3_branch_lof_components.npz": "9" * 64,
                    "v3_open_policy_stage_two.npz": "a" * 64,
                    staged.COMPOSITE_POLICY_FILENAME: "b" * 64,
                },
                novelty_seeds=list(
                    reversed(exporter.VALIDATION_NOVELTY_SEEDS)
                ),
            )

    def test_frontend_integer_equality_alias_is_refused(self) -> None:
        kwargs = self.kwargs()
        kwargs["frontend"]["patch_length"] = True
        with self.assertRaisesRegex(ValueError, "frontend geometry/types"):
            exporter.dual_binding_payload(**kwargs)

    def test_validation_seed_float_alias_is_refused(self) -> None:
        kwargs = self.kwargs()
        kwargs["novelty_seeds"][0] = float(
            exporter.VALIDATION_NOVELTY_SEEDS[0]
        )
        with self.assertRaisesRegex(ValueError, "exact integers"):
            exporter.dual_binding_payload(**kwargs)


if __name__ == "__main__":
    unittest.main()
