from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest
from unittest import mock

import validate_scale_consistency_adaptation_attempt_2 as validator


class ScaleConsistencyAdaptationAttempt2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.attempt_2 = json.loads(
            validator.ATTEMPT_2_PATH.read_text(encoding="utf-8")
        )
        cls.miss_report = json.loads(
            validator.MISS_REPORT_PATH.read_text(encoding="utf-8")
        )
        cls.attempt_1 = json.loads(
            validator.ATTEMPT_1_PATH.read_text(encoding="utf-8")
        )
        cls.protocol = json.loads(
            validator.PROTOCOL_PATH.read_text(encoding="utf-8")
        )
        cls.metrics = {
            name: json.loads(path.read_text(encoding="utf-8"))
            for name, path in validator.METRIC_PATHS.items()
        }
        cls.metric_hashes = {
            name: digest
            for name, (_relative, digest)
            in validator.METRIC_BINDINGS.items()
        }

    def validate(
        self,
        *,
        attempt_2: dict | None = None,
        miss_report: dict | None = None,
        metrics: dict | None = None,
        metric_hashes: dict | None = None,
    ) -> dict:
        return validator.validate_documents(
            self.attempt_2 if attempt_2 is None else attempt_2,
            self.miss_report if miss_report is None else miss_report,
            self.attempt_1,
            self.protocol,
            self.metrics if metrics is None else metrics,
            self.metric_hashes if metric_hashes is None else metric_hashes,
        )

    def test_repository_evidence_and_intent_are_exact(self) -> None:
        summary = validator.validate_repository_attempt_2()
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["development_metric_files_read"], 6)
        self.assertEqual(summary["corpus_files_read"], 0)
        self.assertEqual(summary["checkpoint_files_read"], 0)
        self.assertEqual(summary["sealed_files_read"], 0)
        self.assertEqual(summary["literal_profile_count"], 31)
        self.assertEqual(summary["paired_views_per_episode"], 62)
        self.assertEqual(
            summary["reassessed_predeclared_failed_goal_count"],
            2,
        )
        self.assertEqual(
            summary["attempt_1_miss_report_sha256"],
            validator.EXPECTED_MISS_REPORT_SHA256,
        )
        self.assertEqual(
            summary["attempt_2_intent_sha256"],
            validator.EXPECTED_ATTEMPT_2_SHA256,
        )

    def test_validator_opens_only_declared_contract_and_metric_json(self) -> None:
        opened: list[Path] = []
        original = Path.open

        def recording_open(path: Path, *args, **kwargs):
            opened.append(path.resolve())
            return original(path, *args, **kwargs)

        with mock.patch.object(Path, "open", recording_open):
            validator.validate_repository_attempt_2()
        self.assertEqual(
            set(opened),
            {path.resolve() for path in validator.ALLOWED_PATHS},
        )
        self.assertTrue(all(path.suffix == ".json" for path in opened))
        self.assertTrue(
            all(
                path.name
                not in {
                    "corpus.json",
                    "scale_eval.json",
                    "combined_prototype_bank.json",
                }
                for path in opened
            )
        )

    def test_all_six_raw_metric_hashes_are_mandatory(self) -> None:
        changed = dict(self.metric_hashes)
        changed["attempt_1_real"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "raw dev_metrics SHA-256"):
            self.validate(metric_hashes=changed)

    def test_fusion_must_mutually_bind_both_branch_metrics(self) -> None:
        changed = copy.deepcopy(self.metrics)
        changed["attempt_1_fusion"]["source_branches"]["real"]["hashes"][
            "dev_metrics.json"
        ] = "0" * 64
        with self.assertRaisesRegex(ValueError, "bind.*real branch"):
            self.validate(metrics=changed)

    def test_miss_report_reproduces_two_predeclared_goal_misses(self) -> None:
        changed = copy.deepcopy(self.metrics)
        changed["attempt_1_fusion"]["final_evaluation"]["current"]["pooled"][
            "accuracy"
        ] = 0.95
        with self.assertRaisesRegex(ValueError, "miss is not reproduced"):
            self.validate(metrics=changed)

        changed_report = copy.deepcopy(self.miss_report)
        changed_report[
            "reassessed_attempt_1_predeclared_failed_goals"
        ][1]["threshold"] = 0.94
        with self.assertRaisesRegex(ValueError, "reassessed predeclared goal"):
            self.validate(miss_report=changed_report)

    def test_full_21_gate_status_is_explicitly_not_evaluated(self) -> None:
        assessment = self.miss_report["full_gate_assessment"]
        self.assertEqual(
            assessment["full_21_gate_status"],
            "not_evaluated",
        )
        self.assertFalse(assessment["other_gate_results_assessed"])
        self.assertFalse(assessment["other_gates_claimed_passed"])

        changed = copy.deepcopy(self.miss_report)
        changed["full_gate_assessment"]["full_21_gate_status"] = "passed"
        with self.assertRaisesRegex(ValueError, "21-gate non-evaluation"):
            self.validate(miss_report=changed)

        changed = copy.deepcopy(self.miss_report)
        changed["full_gate_assessment"][
            "other_gates_claimed_passed"
        ] = True
        with self.assertRaisesRegex(ValueError, "21-gate non-evaluation"):
            self.validate(miss_report=changed)

    def test_worst_profile_scale_recall_is_diagnostic_not_a_gate_result(
        self,
    ) -> None:
        disclosure = self.miss_report["diagnostic_disclosures"][0]
        self.assertEqual(disclosure["baseline_observed"], 0.796875)
        self.assertEqual(disclosure["attempt_1_sc02_observed"], 0.78125)
        self.assertEqual(disclosure["related_gate_threshold"], 0.9)
        self.assertFalse(disclosure["formal_full_gate_result"])

        changed = copy.deepcopy(self.miss_report)
        changed["diagnostic_disclosures"][0][
            "formal_full_gate_result"
        ] = True
        with self.assertRaisesRegex(ValueError, "diagnostic disclosure"):
            self.validate(miss_report=changed)

        changed_metrics = copy.deepcopy(self.metrics)
        changed_metrics["attempt_1_fusion"]["final_evaluation"]["current"][
            "pooled"
        ]["worst_profile_scale_recall"] = 0.9
        with self.assertRaisesRegex(ValueError, "diagnostic is not reproduced"):
            self.validate(metrics=changed_metrics)

    def test_attempt_1_cosine_auxiliary_is_replaced_not_combined(self) -> None:
        changed = copy.deepcopy(self.attempt_2)
        replacement = changed["predeclared_change"][
            "replaces_attempt_1_auxiliary"
        ]
        replacement["cosine_distance_term_in_attempt_2"] = True
        with self.assertRaisesRegex(ValueError, "replacement"):
            self.validate(attempt_2=changed)

    def test_sampler_is_one_train_identity_and_two_distinct_scales_per_profile(
        self,
    ) -> None:
        sampling = self.attempt_2["predeclared_change"]["pair_sampling"]
        self.assertEqual(
            list(sampling["literal_profile_public_class_map"]),
            sampling["literal_profiles"],
        )
        self.assertEqual(sampling["identity_selection_per_profile_per_episode"], 1)
        self.assertEqual(sampling["scale_views_per_selected_identity"], 2)
        self.assertEqual(sampling["pairs_per_episode"], 31)
        self.assertEqual(sampling["views_per_episode"], 62)

        changed = copy.deepcopy(self.attempt_2)
        changed["predeclared_change"]["pair_sampling"][
            "scale_selection"
        ] = "two physical-scale views with replacement"
        with self.assertRaisesRegex(ValueError, "pair sampling"):
            self.validate(attempt_2=changed)

    def test_exact_profile_class_map_and_repeat_targets_are_frozen(self) -> None:
        changed = copy.deepcopy(self.attempt_2)
        changed["predeclared_change"]["pair_sampling"][
            "literal_profile_public_class_map"
        ]["wifi-hr-dsss-11m"] = "bluetooth"
        with self.assertRaisesRegex(ValueError, "pair sampling"):
            self.validate(attempt_2=changed)

        changed = copy.deepcopy(self.attempt_2)
        changed["predeclared_change"]["supervised_pair_auxiliary"][
            "target_tensor_construction"
        ] = "tile(profile_public_class_indices, 2)"
        with self.assertRaisesRegex(ValueError, "supervised-pair CE"):
            self.validate(attempt_2=changed)

    def test_pair_rng_and_independent_phase_draws_are_frozen(self) -> None:
        changed = copy.deepcopy(self.attempt_2)
        changed["predeclared_change"]["randomness"][
            "pair_rng_separate_from_episodic_rng"
        ] = False
        with self.assertRaisesRegex(ValueError, "RNG separation"):
            self.validate(attempt_2=changed)

        changed = copy.deepcopy(self.attempt_2)
        changed["predeclared_change"]["phase_augmentation"][
            "two_views_in_a_pair_share_phase_draw"
        ] = True
        with self.assertRaisesRegex(ValueError, "independent phase"):
            self.validate(attempt_2=changed)

    def test_auxiliary_forward_cannot_mutate_batch_norm_statistics(self) -> None:
        changed = copy.deepcopy(self.attempt_2)
        changed["predeclared_change"]["batch_norm_stat_firewall"][
            "running_mean_variance_or_num_batches_tracked_mutated"
        ] = True
        with self.assertRaisesRegex(ValueError, "BatchNorm-stat firewall"):
            self.validate(attempt_2=changed)

    def test_source_mixed_episode_prototypes_are_detached_for_auxiliary(
        self,
    ) -> None:
        changed = copy.deepcopy(self.attempt_2)
        prototypes = changed["predeclared_change"][
            "supervised_pair_auxiliary"
        ]["public_class_prototypes"]
        prototypes["source"] = "current-source-only support embeddings"
        with self.assertRaisesRegex(ValueError, "supervised-pair CE"):
            self.validate(attempt_2=changed)

        changed = copy.deepcopy(self.attempt_2)
        prototypes = changed["predeclared_change"][
            "supervised_pair_auxiliary"
        ]["public_class_prototypes"]
        prototypes["detached_for_auxiliary"] = False
        with self.assertRaisesRegex(ValueError, "supervised-pair CE"):
            self.validate(attempt_2=changed)

    def test_auxiliary_uses_detached_numeric_episodic_logit_scale(self) -> None:
        logits = self.attempt_2["predeclared_change"][
            "supervised_pair_auxiliary"
        ]["logits"]
        self.assertTrue(logits["logit_scale_detached_for_auxiliary"])
        self.assertFalse(logits["auxiliary_gradient_into_logit_scale"])
        self.assertTrue(
            logits[
                "main_episodic_cross_entropy_is_only_logit_scale_optimizer_signal"
            ]
        )

        changed = copy.deepcopy(self.attempt_2)
        changed_logits = changed["predeclared_change"][
            "supervised_pair_auxiliary"
        ]["logits"]
        changed_logits["auxiliary_gradient_into_logit_scale"] = True
        with self.assertRaisesRegex(ValueError, "supervised-pair CE"):
            self.validate(attempt_2=changed)

    def test_mean_over_62_views_and_coefficient_point_two_are_frozen(
        self,
    ) -> None:
        changed = copy.deepcopy(self.attempt_2)
        auxiliary = changed["predeclared_change"][
            "supervised_pair_auxiliary"
        ]
        auxiliary["cross_entropy"]["mean_denominator"] = 31
        with self.assertRaisesRegex(ValueError, "supervised-pair CE"):
            self.validate(attempt_2=changed)

        changed = copy.deepcopy(self.attempt_2)
        changed["predeclared_change"]["supervised_pair_auxiliary"][
            "auxiliary_coefficient"
        ] = 0.1
        with self.assertRaisesRegex(ValueError, "supervised-pair CE"):
            self.validate(attempt_2=changed)

    def test_run_fusion_and_no_other_change_contract_are_exact(self) -> None:
        changed = copy.deepcopy(self.attempt_2)
        changed["inherited_run_contract"]["episodes"] = 8001
        with self.assertRaisesRegex(ValueError, "inherited run"):
            self.validate(attempt_2=changed)

        changed = copy.deepcopy(self.attempt_2)
        changed["inherited_run_contract"]["branch_fusion"][
            "weight_real"
        ] = 0.6
        with self.assertRaisesRegex(ValueError, "inherited run"):
            self.validate(attempt_2=changed)

        changed = copy.deepcopy(self.attempt_2)
        changed["no_other_change_contract"][
            "checkpoint_score_order_threshold_or_comparison_changed"
        ] = True
        with self.assertRaisesRegex(ValueError, "no-other-change"):
            self.validate(attempt_2=changed)

    def test_selection_and_sealed_rows_cannot_reach_fit(self) -> None:
        changed = copy.deepcopy(self.attempt_2)
        changed["fitting_firewall"][
            "seed20262904_selection_rows_used_for_any_gradient_or_optimizer_step"
        ] = 1
        with self.assertRaisesRegex(ValueError, "fitting firewall"):
            self.validate(attempt_2=changed)

        changed = copy.deepcopy(self.attempt_2)
        changed["fitting_firewall"][
            "sealed_rows_used_for_any_fit_checkpoint_or_candidate_selection"
        ] = 1
        with self.assertRaisesRegex(ValueError, "fitting firewall"):
            self.validate(attempt_2=changed)

    def test_intent_precedes_attempt_2_source_training_and_inference(self) -> None:
        changed = copy.deepcopy(self.attempt_2)
        changed["timing_and_validation_boundary"][
            "attempt_2_training_started_before_this_intent"
        ] = True
        with self.assertRaisesRegex(ValueError, "timing/validation"):
            self.validate(attempt_2=changed)


if __name__ == "__main__":
    unittest.main()
