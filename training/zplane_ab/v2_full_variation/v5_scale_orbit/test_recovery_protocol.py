from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest
from unittest import mock

import validate_recovery_protocol as recovery


class RecoveryProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = recovery._load_json(
            recovery.PROTOCOL_PATH, "test recovery protocol"
        )
        cls.registry = recovery._load_json(
            recovery.REGISTRY_PATH, "test seed registry"
        )

    def test_repository_contract_and_bound_evidence_are_exact(self) -> None:
        result = recovery.validate_repository_contract()
        self.assertTrue(result["valid"])
        self.assertFalse(result["independence_from_seed20262903"])
        self.assertEqual(result["inherited_gate_count"], 19)
        self.assertEqual(result["added_directional_gate_count"], 2)
        self.assertEqual(result["corpus_files_read"], 0)
        self.assertEqual(
            recovery._sha256(recovery.REGISTRY_PATH),
            recovery.EXPECTED_REGISTRY_SHA256,
        )

    def test_validator_only_opens_non_corpus_metadata_evidence(self) -> None:
        checked: list[str] = []
        original = recovery._metadata_evidence_path

        def record(relative_value, role):
            checked.append(str(relative_value))
            return original(relative_value, role)

        with mock.patch.object(
            recovery,
            "_metadata_evidence_path",
            side_effect=record,
        ):
            result = recovery.validate_repository_contract()
        self.assertTrue(checked)
        self.assertEqual(result["corpus_files_read"], 0)
        for relative in checked:
            path = Path(relative)
            self.assertEqual(path.suffix, ".json")
            self.assertNotIn(path.name, {"corpus.json", "scale_eval.json"})
            self.assertFalse(
                any("signallab" in part.lower() for part in path.parts)
            )

    def test_quarantined_corpus_paths_are_never_valid_evidence_paths(
        self,
    ) -> None:
        for row in self.registry["spent_or_quarantined"]:
            quarantine = row.get("quarantined_corpus_relative_path")
            if quarantine is None:
                continue
            with self.subTest(seed=row["seed"]):
                with self.assertRaisesRegex(
                    ValueError, "non-corpus JSON evidence"
                ):
                    recovery._metadata_evidence_path(
                        quarantine, "quarantined corpus"
                    )

    def test_ancestor_failure_and_adaptation_boundary_are_immutable(
        self,
    ) -> None:
        ancestor = self.protocol["lineage"]["ancestor_failed_evidence"]
        self.assertEqual(ancestor, recovery.EXPECTED_ANCESTOR)
        self.assertFalse(
            self.protocol["lineage"][
                "independence_from_seed20262903"
            ]
        )
        self.assertFalse(ancestor["all_pass"])
        self.assertFalse(ancestor["release_evidence"])
        self.assertTrue(ancestor["development_only"])

        mutated = deepcopy(self.protocol)
        mutated["lineage"]["independence_from_seed20262903"] = True
        with self.assertRaisesRegex(ValueError, "lineage/adaptation"):
            recovery.validate_contract(
                mutated,
                self.registry,
                verify_evidence=False,
            )

    def test_inherited_gate_weakening_is_rejected(self) -> None:
        mutated = deepcopy(self.protocol)
        for row in mutated["gate_contract"]["inherited_gates"]:
            if row["name"].endswith("pooled_closed_accuracy"):
                row["threshold"] = 0.94
                break
        with self.assertRaisesRegex(ValueError, "19-gate"):
            recovery.validate_contract(
                mutated,
                self.registry,
                verify_evidence=False,
            )

    def test_directional_dsss_bluetooth_gate_change_is_rejected(self) -> None:
        mutated = deepcopy(self.protocol)
        mutated["gate_contract"]["added_directional_confusion_gates"][0][
            "threshold"
        ] = 0.03
        with self.assertRaisesRegex(ValueError, "directional gates"):
            recovery.validate_contract(
                mutated,
                self.registry,
                verify_evidence=False,
            )

    def test_train_dev_validation_identity_firewall_is_required(self) -> None:
        mutated = deepcopy(self.protocol)
        mutated["data_contract"]["identity_separation"][
            "paired_views_may_not_cross_partitions"
        ] = False
        with self.assertRaisesRegex(ValueError, "identity firewall"):
            recovery.validate_contract(
                mutated,
                self.registry,
                verify_evidence=False,
            )

    def test_fresh_seed_overlap_with_spent_inventory_is_rejected(self) -> None:
        mutated = deepcopy(self.registry)
        mutated["fresh_allocations"]["training_scale"]["seeds"][0] = (
            20_262_903
        )
        with self.assertRaisesRegex(ValueError, "training_scale changed"):
            recovery.validate_contract(
                self.protocol,
                mutated,
                verify_evidence=False,
            )

    def test_sealed_seed_role_change_is_rejected(self) -> None:
        mutated = deepcopy(self.registry)
        mutated["fresh_allocations"]["sealed_known_scale_validation"][
            "generation_allowed_before_candidate_freeze"
        ] = True
        with self.assertRaisesRegex(
            ValueError, "sealed_known_scale_validation changed"
        ):
            recovery.validate_contract(
                self.protocol,
                mutated,
                verify_evidence=False,
            )

    def test_stop_on_first_failure_cannot_be_disabled(self) -> None:
        mutated = deepcopy(self.protocol)
        mutated["validation_sequence"][2]["stop_on_failure"] = False
        with self.assertRaisesRegex(ValueError, "one-shot stage"):
            recovery.validate_contract(
                mutated,
                self.registry,
                verify_evidence=False,
            )

    def test_validation_seed_sequence_matches_registry(self) -> None:
        stages = {
            row["name"]: row
            for row in self.protocol["validation_sequence"]
        }
        self.assertEqual(
            stages["known_scale_validation_primary"]["seeds"],
            [20_262_920],
        )
        self.assertEqual(
            stages["known_scale_validation_confirmation"]["seeds"],
            [20_262_921],
        )
        self.assertEqual(
            stages["novelty_validation_joint"]["seeds"],
            [20_263_002, 20_263_003],
        )
        self.assertTrue(
            stages["known_scale_validation_primary"][
                "generation_claim_ledger_before_corpus_generation"
            ]
        )
        self.assertTrue(
            stages["known_scale_validation_primary"][
                "score_claim_ledger_before_first_model_inference"
            ]
        )


if __name__ == "__main__":
    unittest.main()
