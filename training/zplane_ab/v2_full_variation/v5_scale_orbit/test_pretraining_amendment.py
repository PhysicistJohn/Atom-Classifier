from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest
from unittest import mock

import validate_pretraining_amendment as validator


class PretrainingAmendmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.amendment = json.loads(
            validator.AMENDMENT_PATH.read_text(encoding="utf-8")
        )
        cls.protocol = json.loads(
            validator.PROTOCOL_PATH.read_text(encoding="utf-8")
        )
        cls.registry = json.loads(
            validator.REGISTRY_PATH.read_text(encoding="utf-8")
        )

    def validate(self, amendment: dict) -> dict:
        return validator.validate_amendment(
            amendment,
            self.protocol,
            self.registry,
            verify_git_commit=False,
        )

    def test_repository_amendment_and_parent_commit_are_exact(self) -> None:
        summary = validator.validate_repository_amendment()
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["corpus_files_read"], 0)
        self.assertEqual(
            summary["protocol_sha256"],
            validator.EXPECTED_PROTOCOL_SHA256,
        )
        self.assertEqual(
            summary["seed_registry_sha256"],
            validator.EXPECTED_REGISTRY_SHA256,
        )

    def test_validator_only_opens_three_contract_json_files(self) -> None:
        opened: list[Path] = []
        original = Path.open

        def recording_open(path: Path, *args, **kwargs):
            opened.append(path.resolve())
            return original(path, *args, **kwargs)

        with mock.patch.object(Path, "open", recording_open):
            validator.validate_repository_amendment(
                verify_git_commit=False
            )
        self.assertEqual(
            set(opened),
            {
                validator.AMENDMENT_PATH,
                validator.PROTOCOL_PATH,
                validator.REGISTRY_PATH,
            },
        )
        self.assertTrue(
            all(path.suffix == ".json" for path in opened)
        )
        self.assertTrue(
            all(
                path.name
                not in {"corpus.json", "scale_eval.json"}
                for path in opened
            )
        )

    def test_outcome_blind_timing_cannot_be_rewritten(self) -> None:
        changed = copy.deepcopy(self.amendment)
        changed["timing_and_outcome_blindness"][
            "seed20264101_manifest_read_before_amendment"
        ] = True
        with self.assertRaisesRegex(ValueError, "outcome-blind"):
            self.validate(changed)

    def test_seed20264101_cannot_supply_selection(self) -> None:
        changed = copy.deepcopy(self.amendment)
        allocation = changed["role_allocation"]["training_and_enrollment"]
        allocation["allowed_roles"].append("selection")
        with self.assertRaisesRegex(ValueError, "seed/role"):
            self.validate(changed)

    def test_seed20262904_cannot_reach_fit_population(self) -> None:
        changed = copy.deepcopy(self.amendment)
        changed["role_allocation"]["adaptive_selection"][
            "rows_used_for_weight_fit"
        ] = 1
        with self.assertRaisesRegex(ValueError, "seed/role"):
            self.validate(changed)

    def test_role_boundaries_are_exact(self) -> None:
        changed = copy.deepcopy(self.amendment)
        changed["role_allocation"]["training_and_enrollment"][
            "realization_rules"
        ][0]["end_exclusive"] = 41
        with self.assertRaisesRegex(ValueError, "seed/role"):
            self.validate(changed)

    def test_composite_prefix_dispatch_is_mandatory(self) -> None:
        changed = copy.deepcopy(self.amendment)
        changed["composite_loader_contract"][
            "prefix_dispatch_uses_row_source_seed"
        ] = False
        with self.assertRaisesRegex(ValueError, "composite loader"):
            self.validate(changed)

    def test_exact_realization_index_set_is_mandatory(self) -> None:
        changed = copy.deepcopy(self.amendment)
        changed["composite_loader_contract"][
            "exact_realization_index_set_required_per_profile_per_seed"
        ].pop()
        with self.assertRaisesRegex(ValueError, "composite loader"):
            self.validate(changed)

    def test_cyclic_phase_separation_cannot_be_weakened(self) -> None:
        changed = copy.deepcopy(self.amendment)
        changed["identity_applicability_and_separation"]["cyclic_replay"][
            "phase_identity_separation_weakened"
        ] = True
        with self.assertRaisesRegex(ValueError, "cyclic"):
            self.validate(changed)

    def test_payload_separation_remains_required_when_present(self) -> None:
        changed = copy.deepcopy(self.amendment)
        changed["identity_applicability_and_separation"]["payload_seed"][
            "required_when_emitted_by_generator"
        ] = False
        with self.assertRaisesRegex(ValueError, "payload"):
            self.validate(changed)

    def test_one_shot_transmitter_independence_cannot_be_claimed(self) -> None:
        changed = copy.deepcopy(self.amendment)
        changed["identity_applicability_and_separation"][
            "one_shot_replay"
        ]["current_generator_emits_independent_transmitter_waveforms_per_realization"] = True
        with self.assertRaisesRegex(ValueError, "one-shot"):
            self.validate(changed)

    def test_content_receiver_and_prefix_collision_ceilings_are_zero(self) -> None:
        changed = copy.deepcopy(self.amendment)
        changed["identity_applicability_and_separation"][
            "required_cross_partition_collision_counts"
        ]["runtime_prefix_sha256"] = 1
        with self.assertRaisesRegex(ValueError, "collision"):
            self.validate(changed)

    def test_required_audit_uses_native_compatibility_keys(self) -> None:
        required = self.amendment["required_composite_audit"][
            "exact_top_level_keys"
        ]
        for key in (
            "profiles_present",
            "profile_role_counts",
            "runtime_input_lengths",
            "identity_separation",
            "fitting_firewall",
        ):
            self.assertIn(key, required)
        changed = copy.deepcopy(self.amendment)
        changed["required_composite_audit"]["exact_top_level_keys"].remove(
            "profiles_present"
        )
        with self.assertRaisesRegex(ValueError, "audit"):
            self.validate(changed)

    def test_stop_before_training_cannot_be_disabled(self) -> None:
        changed = copy.deepcopy(self.amendment)
        changed["stop_conditions"][
            "stop_before_training_on_first_failure"
        ] = False
        with self.assertRaisesRegex(ValueError, "stop"):
            self.validate(changed)


if __name__ == "__main__":
    unittest.main()
