from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest
from unittest import mock

import validate_scale_consistency_adaptation_attempt_3 as validator


class ScaleConsistencyAdaptationAttempt3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.intent = json.loads(
            validator.ATTEMPT_3_PATH.read_text(encoding="utf-8")
        )
        cls.report = json.loads(
            validator.ATTEMPT_2_MISS_PATH.read_text(encoding="utf-8")
        )
        cls.attempt_2 = json.loads(
            validator.ATTEMPT_2_PATH.read_text(encoding="utf-8")
        )
        cls.protocol = json.loads(
            validator.PROTOCOL_PATH.read_text(encoding="utf-8")
        )
        cls.novelty = json.loads(
            validator.NOVELTY_PATH.read_text(encoding="utf-8")
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
        intent: dict | None = None,
        report: dict | None = None,
        metrics: dict | None = None,
        metric_hashes: dict | None = None,
    ) -> dict:
        return validator.validate_documents(
            self.intent if intent is None else intent,
            self.report if report is None else report,
            self.attempt_2,
            self.protocol,
            self.novelty,
            self.metrics if metrics is None else metrics,
            self.metric_hashes if metric_hashes is None else metric_hashes,
        )

    def test_repository_contracts_metrics_and_ancestry_are_exact(self) -> None:
        summary = validator.validate_repository_attempt_3()
        self.assertTrue(summary["valid"])
        self.assertTrue(
            summary[
                "attempt_2_preregistration_to_source_ancestry_verified"
            ]
        )
        self.assertTrue(
            summary[
                "attempt_2_source_to_repository_head_ancestry_verified"
            ]
        )
        self.assertEqual(summary["development_metric_files_read"], 5)
        self.assertEqual(summary["committed_source_blobs_read"], 2)
        expected_committed_intent_reads = int(
            not summary[
                "attempt_3_intent_commit_ancestry_deferred_until_commit"
            ]
        )
        self.assertEqual(
            summary["committed_attempt_3_intent_blobs_read"],
            expected_committed_intent_reads,
        )
        self.assertEqual(summary["corpus_files_read"], 0)
        self.assertEqual(summary["checkpoint_files_read"], 0)
        self.assertEqual(summary["model_inference_runs"], 0)
        self.assertEqual(summary["attempt_2_subthreshold_profile_scale_cell_count"], 4)
        self.assertEqual(summary["attempt_2_subthreshold_profile_agreement_count"], 3)
        self.assertEqual(summary["known_gate_count"], 21)
        self.assertEqual(summary["full_gate_count"], 27)

    def test_validator_opens_only_declared_json_evidence(self) -> None:
        opened: list[Path] = []
        original = Path.open

        def recording_open(path: Path, *args, **kwargs):
            opened.append(path.resolve())
            return original(path, *args, **kwargs)

        with mock.patch.object(Path, "open", recording_open):
            validator.validate_repository_attempt_3()
        self.assertEqual(
            set(opened),
            {path.resolve() for path in validator.ALLOWED_PATHS},
        )
        self.assertTrue(all(path.suffix == ".json" for path in opened))

    def test_validator_reads_only_declared_committed_blobs(self) -> None:
        calls: list[tuple[str, ...]] = []
        original = validator.subprocess.run

        def recording_run(args, *run_args, **run_kwargs):
            command = tuple(str(value) for value in args)
            if (
                len(command) == 3
                and command[:2] == ("git", "show")
                and ":" in command[2]
            ):
                calls.append(command)
            return original(args, *run_args, **run_kwargs)

        with mock.patch.object(
            validator.subprocess,
            "run",
            side_effect=recording_run,
        ):
            summary = validator.validate_repository_attempt_3()
        expected_source = {
            (
                "git",
                "show",
                f"{validator.SOURCE_COMMIT}:{relative}",
            )
            for relative in validator.COMMITTED_SOURCE_BLOB_BINDINGS
        }
        expected = set(expected_source)
        if not summary[
            "attempt_3_intent_commit_ancestry_deferred_until_commit"
        ]:
            relative = str(
                validator.ATTEMPT_3_PATH.relative_to(validator.REPO)
            )
            intent_commit = validator._git(
                "log", "-1", "--format=%H", "--", relative
            ).stdout.strip()
            expected.add(("git", "show", f"{intent_commit}:{relative}"))
        self.assertEqual(set(calls), expected)
        self.assertEqual(len(calls), len(expected))
        self.assertEqual(
            summary["committed_source_blobs_read"],
            len(expected_source),
        )
        self.assertEqual(
            summary["committed_attempt_3_intent_blobs_read"],
            len(expected - expected_source),
        )

    def test_all_attempt_2_metric_hashes_and_branch_bindings_are_mandatory(
        self,
    ) -> None:
        changed_hashes = dict(self.metric_hashes)
        changed_hashes["attempt_2_real"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "raw dev_metrics SHA-256"):
            self.validate(metric_hashes=changed_hashes)

        changed_metrics = copy.deepcopy(self.metrics)
        changed_metrics["attempt_2_fusion"]["source_branches"]["complex"][
            "hashes"
        ]["dev_metrics.json"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "bind.*complex branch"):
            self.validate(metrics=changed_metrics)

    def test_attempt_2_one_recovered_and_one_remaining_goal_are_reproduced(
        self,
    ) -> None:
        changed = copy.deepcopy(self.metrics)
        changed["attempt_2_fusion"]["final_evaluation"]["current"]["pooled"][
            "accuracy"
        ] = 0.94
        with self.assertRaisesRegex(ValueError, "not reproduced"):
            self.validate(metrics=changed)

        changed_report = copy.deepcopy(self.report)
        changed_report["reassessed_predeclared_recovery_goals"][1][
            "attempt_2_spair02_pass"
        ] = True
        with self.assertRaisesRegex(ValueError, "recovery row"):
            self.validate(report=changed_report)

    def test_exact_four_cells_and_three_profile_agreements_are_bound(
        self,
    ) -> None:
        diagnostics = self.report[
            "exact_subthreshold_classifier_diagnostics"
        ]
        self.assertEqual(
            len(diagnostics["profile_scale_cells_below_reference_threshold"]),
            4,
        )
        self.assertEqual(
            len(diagnostics["profiles_below_agreement_threshold"]),
            3,
        )

        changed = copy.deepcopy(self.report)
        changed["exact_subthreshold_classifier_diagnostics"][
            "profile_scale_cells_below_reference_threshold"
        ][0]["recall"] = 0.9
        with self.assertRaisesRegex(ValueError, "profile/scale cells"):
            self.validate(report=changed)

        changed = copy.deepcopy(self.report)
        changed["exact_subthreshold_classifier_diagnostics"][
            "profiles_below_agreement_threshold"
        ][0][
            "additional_consistent_pairs_required_to_reach_next_64_row_"
            "fraction_at_or_above_threshold"
        ] = 2
        with self.assertRaisesRegex(ValueError, "profile agreement"):
            self.validate(report=changed)

    def test_full_21_and_27_gate_statuses_are_explicitly_not_evaluated(
        self,
    ) -> None:
        gate = self.report["full_gate_assessment"]
        self.assertEqual(gate["known_21_gate_status"], "not_evaluated")
        self.assertEqual(
            gate["known_plus_novelty_27_gate_status"], "not_evaluated"
        )
        self.assertFalse(gate["other_gates_claimed_passed"])
        self.assertFalse(gate["release_claim_permitted"])

        changed = copy.deepcopy(self.report)
        changed["full_gate_assessment"][
            "known_plus_novelty_27_gate_status"
        ] = "passed"
        with self.assertRaisesRegex(ValueError, "21/27-gate"):
            self.validate(report=changed)

    def test_attempt_3_is_exactly_max_two_views_then_mean_31_at_weight_point2(
        self,
    ) -> None:
        auxiliary = self.intent["predeclared_change"][
            "supervised_pair_auxiliary"
        ]
        reduction = auxiliary["cross_entropy"]
        self.assertEqual(reduction["per_view_loss_shape_before_profile_reduction"], [31, 2])
        self.assertEqual(reduction["across_profile_mean_denominator"], 31)
        self.assertEqual(auxiliary["auxiliary_coefficient"], 0.2)
        self.assertFalse(auxiliary["cosine_distance_term"])
        self.assertFalse(
            auxiliary["profile_and_class_weighting"]["class_balancing_added"]
        )

        changed = copy.deepcopy(self.intent)
        changed["predeclared_change"]["supervised_pair_auxiliary"][
            "cross_entropy"
        ]["within_profile_reduction"] = "mean"
        with self.assertRaisesRegex(ValueError, "hard-view reduction"):
            self.validate(intent=changed)

        changed = copy.deepcopy(self.intent)
        changed["predeclared_change"]["supervised_pair_auxiliary"][
            "profile_and_class_weighting"
        ]["class_balancing_added"] = True
        with self.assertRaisesRegex(ValueError, "equal-profile weighting"):
            self.validate(intent=changed)

    def test_frozen_tuple_firewall_and_timing_cannot_drift(self) -> None:
        changed = copy.deepcopy(self.intent)
        changed["inherited_run_contract"]["episodes"] = 8001
        with self.assertRaisesRegex(ValueError, "run/checkpoint tuple"):
            self.validate(intent=changed)

        changed = copy.deepcopy(self.intent)
        changed["no_other_change_contract"]["branch_fusion_changed"] = True
        with self.assertRaisesRegex(ValueError, "no-other-change"):
            self.validate(intent=changed)

        changed = copy.deepcopy(self.intent)
        changed["fitting_firewall"][
            "seed20262904_selection_rows_used_for_any_gradient_or_optimizer_step"
        ] = 1
        with self.assertRaisesRegex(ValueError, "fitting firewall"):
            self.validate(intent=changed)

        changed = copy.deepcopy(self.intent)
        changed["timing_and_validation_boundary"][
            "attempt_3_inference_started_before_this_intent"
        ] = True
        with self.assertRaisesRegex(ValueError, "timing/validation"):
            self.validate(intent=changed)


if __name__ == "__main__":
    unittest.main()
