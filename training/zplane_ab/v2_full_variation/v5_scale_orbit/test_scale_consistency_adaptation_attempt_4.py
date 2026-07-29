from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import validate_scale_consistency_adaptation_attempt_4 as validator


class ScaleConsistencyAdaptationAttempt4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.intent = json.loads(
            validator.ATTEMPT_4_PATH.read_text(encoding="utf-8")
        )
        cls.report = json.loads(
            validator.ATTEMPT_3_MISS_PATH.read_text(encoding="utf-8")
        )
        cls.attempt_3 = json.loads(
            validator.ATTEMPT_3_PATH.read_text(encoding="utf-8")
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
            self.attempt_3,
            self.protocol,
            self.novelty,
            self.metrics if metrics is None else metrics,
            self.metric_hashes if metric_hashes is None else metric_hashes,
        )

    def test_repository_contracts_metrics_and_ancestry_are_exact(self) -> None:
        summary = validator.validate_repository_attempt_4()
        self.assertTrue(summary["valid"])
        self.assertTrue(
            summary[
                "attempt_3_preregistration_to_source_ancestry_verified"
            ]
        )
        self.assertTrue(
            summary[
                "attempt_3_source_to_repository_head_ancestry_verified"
            ]
        )
        self.assertEqual(summary["development_metric_files_read"], 4)
        self.assertEqual(summary["committed_source_blobs_read"], 2)
        self.assertEqual(summary["corpus_files_read"], 0)
        self.assertEqual(summary["checkpoint_files_read"], 0)
        self.assertEqual(summary["model_inference_runs"], 0)
        self.assertEqual(
            summary["attempt_3_subthreshold_profile_scale_cell_count"],
            1,
        )
        self.assertEqual(
            summary["attempt_3_subthreshold_profile_agreement_count"],
            2,
        )
        self.assertEqual(summary["known_gate_count"], 21)
        self.assertEqual(summary["full_gate_count"], 27)

    def test_validator_opens_only_declared_json_evidence(self) -> None:
        opened: list[Path] = []
        original = Path.open

        def recording_open(path: Path, *args, **kwargs):
            opened.append(path.resolve())
            return original(path, *args, **kwargs)

        with mock.patch.object(Path, "open", recording_open):
            validator.validate_repository_attempt_4()
        self.assertEqual(
            set(opened),
            {path.resolve() for path in validator.ALLOWED_PATHS},
        )
        self.assertTrue(all(path.suffix == ".json" for path in opened))

    def test_all_branch_hashes_and_fusion_bindings_are_mandatory(self) -> None:
        changed_hashes = dict(self.metric_hashes)
        changed_hashes["attempt_3_real"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.validate(metric_hashes=changed_hashes)

        changed = copy.deepcopy(self.metrics)
        changed["attempt_3_fusion"]["source_branches"]["complex"]["hashes"][
            "dev_metrics.json"
        ] = "0" * 64
        with self.assertRaisesRegex(ValueError, "fusion or branch"):
            self.validate(metrics=changed)

        changed = copy.deepcopy(self.metrics)
        changed["attempt_3_real"]["source_sha256"][
            "v5/run_scale_orbit_dev.py"
        ] = "0" * 64
        with self.assertRaisesRegex(ValueError, "branch contract"):
            self.validate(metrics=changed)

    def test_exact_failure_inventories_and_deltas_are_reproduced(self) -> None:
        exact = self.report["exact_subthreshold_classifier_diagnostics"]
        self.assertEqual(
            exact["profile_scale_cells_below_reference_threshold"],
            [
                {
                    "profile_scale": "current:wifi6-he-mu|scale=1",
                    "public_class": "ofdm",
                    "rows": 64,
                    "correct_rows": 56,
                    "recall": 0.875,
                    (
                        "additional_correct_rows_required_to_reach_next_64_"
                        "row_fraction_at_or_above_threshold"
                    ): 2,
                }
            ],
        )
        self.assertEqual(
            [row["profile"] for row in exact[
                "profiles_below_agreement_threshold"
            ]],
            [
                "current:lte-band3-fdd-20m",
                "current:wifi6-he-tb",
            ],
        )
        summary = self.report["complete_cell_and_agreement_delta_summary"]
        self.assertEqual(
            (summary["profile_scale_cells_improved"],
             summary["profile_scale_cells_regressed"],
             summary["profile_scale_cells_unchanged"],
             summary["net_correct_rows"]),
            (43, 6, 75, 52),
        )
        self.assertEqual(
            (summary["profile_agreements_improved"],
             summary["profile_agreements_regressed"],
             summary["profile_agreements_unchanged"]),
            (15, 2, 14),
        )

        changed = copy.deepcopy(self.report)
        changed["exact_subthreshold_classifier_diagnostics"][
            "profile_scale_cells_below_reference_threshold"
        ][0]["correct_rows"] = 57
        with self.assertRaisesRegex(ValueError, "failure inventory"):
            self.validate(report=changed)

        changed = copy.deepcopy(self.report)
        changed["attempt_2_failure_transition_inventory"][
            "profile_agreements"
        ][0]["delta"] = 0.0
        with self.assertRaisesRegex(ValueError, "transition inventory"):
            self.validate(report=changed)

        changed = copy.deepcopy(self.report)
        changed["attempt_3_minus_attempt_2_fusion_deltas"][
            "current_net_correct_rows_out_of_7936"
        ] = 51
        with self.assertRaisesRegex(ValueError, "deltas"):
            self.validate(report=changed)

    def test_all_four_target_hash_is_independently_reproduced(self) -> None:
        profile_class_indices = (
            [1, 1]
            + [5] * 7
            + [6] * 16
            + [3]
            + [6] * 5
        )
        self.assertEqual(len(profile_class_indices), 31)
        targets = [
            label
            for label in profile_class_indices
            for _ in range(4)
        ]
        encoded = json.dumps(
            targets,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.assertEqual(
            hashlib.sha256(encoded).hexdigest(),
            validator.ATTEMPT_4_TARGET_SHA256,
        )
        auxiliary = self.intent["predeclared_change"][
            "supervised_pair_auxiliary"
        ]
        self.assertEqual(
            auxiliary["paired_targets_sha256"],
            validator.ATTEMPT_4_TARGET_SHA256,
        )
        self.assertEqual(auxiliary["paired_logits_shape"], [31, 4, 7])
        self.assertEqual(auxiliary["paired_targets_shape"], [31, 4])

    def test_max_four_reduction_and_detachment_are_frozen(self) -> None:
        change = self.intent["predeclared_change"]
        auxiliary = change["supervised_pair_auxiliary"]
        reduction = auxiliary["cross_entropy"]
        self.assertEqual(
            reduction["per_view_loss_shape_before_profile_reduction"],
            [31, 4],
        )
        self.assertEqual(
            reduction["within_profile_reduction"],
            "maximum_over_exactly_four_contiguous_scale_view_losses",
        )
        self.assertEqual(reduction["across_profile_mean_denominator"], 31)
        self.assertEqual(auxiliary["auxiliary_coefficient"], 0.2)
        self.assertTrue(
            auxiliary["public_class_prototypes"][
                "detached_for_auxiliary"
            ]
        )
        self.assertTrue(
            auxiliary["logits"]["logit_scale_detached_for_auxiliary"]
        )
        bn = change[
            "batch_norm_stat_firewall_inherited_exactly_from_attempt_3"
        ]
        self.assertTrue(bn["auxiliary_forward_uses_batch_norm_eval_mode"])
        self.assertTrue(bn["dropout_remains_live_during_auxiliary_forward"])

        changed = copy.deepcopy(self.intent)
        changed["predeclared_change"]["supervised_pair_auxiliary"][
            "cross_entropy"
        ]["within_profile_reduction"] = "mean"
        with self.assertRaisesRegex(ValueError, "all-four hard-view"):
            self.validate(intent=changed)

        changed = copy.deepcopy(self.intent)
        changed["predeclared_change"]["supervised_pair_auxiliary"][
            "auxiliary_coefficient"
        ] = 0.3
        with self.assertRaisesRegex(ValueError, "all-four hard-view"):
            self.validate(intent=changed)

    def test_legacy_choice_is_consumed_and_discarded_before_all_four_emit(
        self,
    ) -> None:
        rng = self.intent["predeclared_change"][
            "legacy_pair_rng_trajectory_preservation"
        ]
        self.assertEqual(
            rng["per_profile_operation_order"],
            [
                (
                    "draw one identity index uniformly using "
                    "rng.integers(0, len(entries))"
                ),
                (
                    "call rng.choice(4, size=2, replace=false) exactly as "
                    "attempt 3"
                ),
                (
                    "validate that the compatibility draw has shape [2] and "
                    "distinct indices"
                ),
                (
                    "discard both compatibility-draw values without using "
                    "them to choose emitted views"
                ),
                (
                    "emit all four selected-identity view positions in frozen "
                    "scale-factor order"
                ),
            ],
        )
        self.assertTrue(
            rng[
                "compatibility_draw_preserves_attempt_3_pair_rng_call_count_"
                "order_and_identity_trajectory"
            ]
        )
        emission = self.intent["predeclared_change"][
            "all_four_view_emission"
        ]
        self.assertEqual(
            emission["physical_scale_factors_in_exact_emission_order"],
            [1.0, 1.25, 1.5, 2.0],
        )
        self.assertEqual(emission["views_per_episode"], 124)

        changed = copy.deepcopy(self.intent)
        changed["predeclared_change"][
            "legacy_pair_rng_trajectory_preservation"
        ]["per_profile_operation_order"].pop(1)
        with self.assertRaisesRegex(ValueError, "all-four hard-view"):
            self.validate(intent=changed)

    def test_full_gate_timing_firewall_and_future_schemas_cannot_drift(
        self,
    ) -> None:
        gate = self.report["full_gate_assessment"]
        self.assertEqual(gate["known_21_gate_status"], "not_evaluated")
        self.assertEqual(
            gate["known_plus_novelty_27_gate_status"],
            "not_evaluated",
        )
        metadata = self.intent["future_implementation_metadata_contract"]
        self.assertEqual(
            metadata["branch_run_configuration_schema"],
            "v5-scale-orbit-development-training-v4",
        )
        self.assertEqual(
            metadata["fusion_run_configuration_schema"],
            "v5-scale-orbit-development-fusion-v4",
        )
        self.assertTrue(
            metadata[
                "attempt_3_training_v3_branch_metadata_must_be_rejected"
            ]
        )
        self.assertTrue(
            metadata["attempt_3_fusion_v3_metadata_must_be_rejected"]
        )

        changed = copy.deepcopy(self.report)
        changed["full_gate_assessment"]["known_21_gate_status"] = "passed"
        with self.assertRaisesRegex(ValueError, "21/27-gate"):
            self.validate(report=changed)

        changed = copy.deepcopy(self.intent)
        changed["timing_and_validation_boundary"][
            "attempt_4_training_started_before_this_intent"
        ] = True
        with self.assertRaisesRegex(ValueError, "timing"):
            self.validate(intent=changed)

        changed = copy.deepcopy(self.intent)
        changed["fitting_firewall"][
            "seed20262904_selection_rows_used_for_any_gradient_or_optimizer_step"
        ] = 1
        with self.assertRaisesRegex(ValueError, "fitting firewall"):
            self.validate(intent=changed)

        changed = copy.deepcopy(self.intent)
        changed["no_other_change_contract"]["branch_fusion_changed"] = True
        with self.assertRaisesRegex(ValueError, "no-other-change"):
            self.validate(intent=changed)


if __name__ == "__main__":
    unittest.main()
