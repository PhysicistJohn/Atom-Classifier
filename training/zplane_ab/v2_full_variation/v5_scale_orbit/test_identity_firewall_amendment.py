from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import validate_identity_firewall_amendment as validator  # noqa: E402


class IdentityFirewallAmendmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with validator.AMENDMENT_PATH.open("r", encoding="utf-8") as handle:
            cls.document = json.load(handle)

    def test_repository_bindings_and_outcome_boundary_are_exact(self) -> None:
        result = validator.validate_repository()
        self.assertTrue(result["valid"])
        self.assertEqual(result["corpus_files_read"], 0)
        self.assertEqual(result["adaptive_seed"], 20262904)

    def test_replacement_generation_cannot_overwrite_rejected_bytes(self) -> None:
        changed = copy.deepcopy(self.document)
        changed["replacement_adaptive_corpus"][
            "must_not_overwrite_rejected_directory"
        ] = False
        with self.assertRaises(ValueError):
            validator.validate_document(changed)

    def test_identity_collision_ceiling_cannot_be_weakened(self) -> None:
        for field in validator.ZERO_COLLISION_FIELDS:
            with self.subTest(field=field):
                changed = copy.deepcopy(self.document)
                changed["identity_firewall"][field] = 1
                with self.assertRaises(ValueError):
                    validator.validate_document(changed)

    def test_training_or_inference_before_amendment_is_rejected(self) -> None:
        for field in (
            "first_seed20262904_model_training_started",
            "first_seed20262904_model_inference_started",
            "first_seed20262904_candidate_metrics_observed",
            "replacement_generation_started_before_this_amendment",
        ):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.document)
                changed["timing_and_observation_boundary"][field] = True
                with self.assertRaises(ValueError):
                    validator.validate_document(changed)

    def test_generator_exclusion_binding_cannot_be_removed(self) -> None:
        changed = copy.deepcopy(self.document)
        changed["generator_binding"][
            "required_identity_exclusion_environment"
        ] = {}
        with self.assertRaises(ValueError):
            validator.validate_document(changed)

    def test_gates_counts_and_fitting_firewall_remain_unchanged(self) -> None:
        mutations = {
            "gate_threshold_or_comparison_changed": True,
            "class_or_profile_removed": True,
            "realization_or_scale_count_reduced": True,
            "selection_rows_used_for_any_fit": 1,
            "sealed_seed_generated_or_read": True,
            "stop_on_identity_collision": False,
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                changed = copy.deepcopy(self.document)
                changed["unchanged_contracts"][field] = value
                with self.assertRaises(ValueError):
                    validator.validate_document(changed)


if __name__ == "__main__":
    unittest.main()
