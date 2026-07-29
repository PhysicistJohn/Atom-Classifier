from __future__ import annotations

import copy
import io
from pathlib import Path
import unittest
from contextlib import redirect_stderr

import assemble_scale_orbit_fusion as assembler


def _binding(
    seed: int,
    directory: str,
    *,
    allocation: str,
    statistical_role: str,
    allowed_roles: list[str],
) -> dict:
    return {
        "eval_seed": seed,
        "manifest_relative_path": f"{directory}/scale_eval.json",
        "manifest_sha256": f"{seed:064x}",
        "raw_relative_path": f"{directory}/scale_eval.f32",
        "raw_sha256": f"{seed + 1:064x}",
        "registry_allocation": allocation,
        "statistical_role": statistical_role,
        "allowed_roles": allowed_roles,
    }


def _audit() -> dict:
    current = assembler.corpus_data
    identity = assembler.amendment_validator
    return {
        "schema": assembler.COMPOSITE_AUDIT_SCHEMA,
        "lineage": "v5-scale-orbit",
        "amendment_sha256": assembler.AMENDMENT_SHA256,
        "parent_protocol_sha256": identity.EXPECTED_PROTOCOL_SHA256,
        "seed_registry_sha256": identity.EXPECTED_REGISTRY_SHA256,
        "corpora": {
            "20264101": _binding(
                20264101,
                ".artifacts/synthetic-v5-training",
                allocation="training_scale",
                statistical_role="training_only",
                allowed_roles=["train", "enrollment"],
            ),
            "20262904": _binding(
                20262904,
                ".artifacts/synthetic-v5-selection",
                allocation="adaptive_scale_development",
                statistical_role="adaptive_development_only",
                allowed_roles=["selection"],
            ),
        },
        "roles": copy.deepcopy(assembler.EXPECTED_ROLES),
        "rows_by_role": dict(assembler.EXPECTED_ROWS_BY_ROLE),
        "pair_identities_by_role":
            dict(assembler.EXPECTED_PAIR_IDENTITIES_BY_ROLE),
        "profile_role_counts":
            copy.deepcopy(assembler.EXPECTED_PROFILE_ROLE_COUNTS),
        "profiles_present": list(current.CURRENT_PROFILES),
        "profile_public_class_map":
            dict(current.CURRENT_PROFILE_PUBLIC_CLASS_MAP),
        "classes_present": list(current.CURRENT_CLASSES),
        "public_class_order":
            list(assembler.scale_orbit_data.scale_data.PUBLIC_CLASSES),
        "scale_factors":
            list(assembler.scale_orbit_data.SCALE_ORBIT_EXPECTED_FACTORS),
        "runtime_input_lengths": list(current.RUNTIME_INPUT_LENGTHS),
        "identity_separation": {
            "applicable_collision_counts":
                dict(assembler.EXPECTED_APPLICABLE_COLLISION_COUNTS),
            "inapplicable_keys_by_replay":
                copy.deepcopy(
                    assembler.EXPECTED_INAPPLICABLE_KEYS_BY_REPLAY
                ),
            "prefix_lengths_hashed": list(current.RUNTIME_INPUT_LENGTHS),
            "paired_views_deduplicated": True,
            "all_applicable_collision_counts_zero": True,
        },
        "fitting_firewall": dict(identity.EXPECTED_FIREWALL),
        "development_only": True,
        "release_evidence": False,
        "consumed_test_rows_exposed": 0,
    }


def _branch_metrics(encoder: str = "real") -> dict:
    share = {"numerator": 1, "denominator": 2, "value": 0.5}
    return {
        "status": "complete",
        "development_only": True,
        "release_evidence": False,
        "consumed_historical_test_rows_used": 0,
        "encoder": encoder,
        "architecture": {"encoder": encoder},
        "parameter_count": 1,
        "run_configuration": {
            "schema": assembler.BRANCH_SCHEMA,
            "arguments": {
                "encoder": encoder,
                "seed": 7,
                "episodes": 1,
                "current_source_share": share,
            },
            "randomness": {"seed": 7},
            "checkpoint_selection": {
                "contract": assembler.branch_runner.CHECKPOINT_SCORE_CONTRACT,
                "predeclared": True,
                "tunable": False,
            },
        },
        "training": {
            "episodes": 1,
            "sampler_contract": assembler.branch_runner.SAMPLER_CONTRACT,
            "current_source_share": share,
        },
        "final_evaluation": {},
        "data_audit": {
            "historical": {"consumed_test_rows_exposed": 0},
            "current": _audit(),
            "preprocessing": {
                "frontend": assembler.td_preprocess.preprocess_metadata(),
                "training": {
                    "hierarchical_sampler_contract":
                        assembler.branch_runner.SAMPLER_CONTRACT,
                },
                "runtime_bucket_policy": {
                    "episode_rule": assembler.branch_runner.SAMPLER_CONTRACT,
                },
            },
            "consumed_test_rows_exposed": 0,
        },
        "artifacts": {
            "state_dict": "state.pt",
            "state_dict_sha256": "1" * 64,
            "combined_prototypes": "prototypes.npy",
            "combined_prototypes_sha256": "2" * 64,
            "combined_prototype_bank": "prototypes.json",
            "combined_prototype_bank_sha256": "3" * 64,
            "feature_moments": "moments.npz",
            "feature_moments_sha256": "4" * 64,
        },
        "source_sha256": {},
        "device": "cpu",
    }


class ScaleOrbitFusionAssemblerTests(unittest.TestCase):
    def test_consumed_model_input_is_not_consumed_test_exposure(self) -> None:
        assembler._validate_no_consumed_provenance({
            "trusted_current_geometry_canonicalizer": {
                "consumed_input_samples": 4_096,
            },
            "consumed_test_held_out": 1_000,
            "consumed_test_rows_exposed": 0,
        })

    def test_nonzero_consumed_test_exposure_still_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "forbidden exposure"):
            assembler._validate_no_consumed_provenance({
                "trusted_current_geometry_canonicalizer": {
                    "consumed_input_samples": 4_096,
                },
                "consumed_test_rows_exposed": 1,
            })

    def test_imports_exact_v5_runner(self) -> None:
        self.assertEqual(
            Path(assembler.branch_runner.__file__).resolve(),
            (assembler.HERE / "run_scale_orbit_dev.py").resolve(),
        )
        self.assertEqual(
            assembler.BRANCH_SCHEMA,
            "v5-scale-orbit-development-training-v1",
        )

    def test_parser_requires_both_current_corpora(self) -> None:
        parser = assembler.build_parser()
        arguments = [
            "--real-dir",
            "real",
            "--complex-dir",
            "complex",
            "--current-corpus",
            "training",
            "--output-dir",
            "output",
        ]
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(arguments)
        parsed = parser.parse_args(
            [
                *arguments,
                "--current-selection-corpus",
                "selection",
            ]
        )
        self.assertEqual(parsed.current_corpus, "training")
        self.assertEqual(parsed.current_selection_corpus, "selection")

    def test_branch_metrics_require_exact_v5_schema(self) -> None:
        value = _branch_metrics()
        self.assertIs(
            assembler.validate_branch_metrics(value, encoder="real"),
            value,
        )
        changed = copy.deepcopy(value)
        changed["run_configuration"]["schema"] = (
            "v4-current-source-development-training-v1"
        )
        with self.assertRaisesRegex(ValueError, "exact v5"):
            assembler.validate_branch_metrics(changed, encoder="real")

    def test_composite_audit_accepts_only_amended_seed_roles(self) -> None:
        value = _audit()
        validated = assembler.validate_composite_current_audit(value)
        self.assertIs(validated, value)

        changed = copy.deepcopy(value)
        changed["corpora"]["20264101"]["allowed_roles"].append("selection")
        with self.assertRaisesRegex(ValueError, "allowed_roles"):
            assembler.validate_composite_current_audit(changed)

    def test_composite_audit_rejects_nonzero_identity_collision(self) -> None:
        changed = _audit()
        changed["identity_separation"]["applicable_collision_counts"][
            "runtime_prefix_sha256"
        ] = 1
        with self.assertRaisesRegex(ValueError, "identity separation"):
            assembler.validate_composite_current_audit(changed)

    def test_composite_schema_roles_and_profile_counts_are_exact(self) -> None:
        for field, replacement in (
            ("schema", "v4-current-source-development-audit-v1"),
            (
                "roles",
                {
                    "20264101": ["train", "enrollment", "selection"],
                    "20262904": [],
                },
            ),
            ("profile_role_counts", {}),
        ):
            with self.subTest(field=field):
                changed = _audit()
                changed[field] = replacement
                with self.assertRaises(ValueError):
                    assembler.validate_composite_current_audit(changed)

    def test_composite_audit_paths_must_match_supplied_arguments(self) -> None:
        value = _audit()
        training = (
            assembler.REPO / ".artifacts" / "synthetic-v5-training"
        ).resolve()
        selection = (
            assembler.REPO / ".artifacts" / "synthetic-v5-selection"
        ).resolve()
        assembler.validate_composite_current_audit(
            value,
            current_directory=training,
            selection_directory=selection,
        )
        with self.assertRaisesRegex(ValueError, "supplied branch argument"):
            assembler.validate_composite_current_audit(
                value,
                current_directory=selection,
                selection_directory=training,
            )

    def test_branch_arguments_must_match_both_supplied_corpora(self) -> None:
        training = (assembler.REPO / ".artifacts" / "training").resolve()
        selection = (assembler.REPO / ".artifacts" / "selection").resolve()
        historical = (assembler.REPO / "training" / "artifacts" / "h").resolve()
        arguments = {
            "current_corpus": str(training),
            "current_selection_corpus": str(selection),
            "historical_corpus": str(historical),
        }
        assembler.assert_branch_corpus_arguments(
            arguments,
            current_directory=training,
            selection_directory=selection,
            historical_directory=historical,
        )
        changed = dict(arguments)
        changed["current_selection_corpus"] = str(training)
        with self.assertRaisesRegex(
            ValueError,
            "current_selection_corpus does not match",
        ):
            assembler.assert_branch_corpus_arguments(
                changed,
                current_directory=training,
                selection_directory=selection,
                historical_directory=historical,
            )

    def test_source_contract_pins_amendment_and_v5_assembler(self) -> None:
        hashes = assembler._source_hashes()
        self.assertEqual(
            hashes["v5/pretraining_amendment.json"],
            assembler.AMENDMENT_SHA256,
        )
        self.assertIn("v5/assemble_scale_orbit_fusion.py", hashes)
        self.assertIn("v5/run_scale_orbit_dev.py", hashes)
        self.assertEqual(
            hashes[
                "v5/seed20262904_identity_firewall_acceptance.json"
            ],
            assembler.scale_orbit_data.SCALE_ORBIT_IDENTITY_ACCEPTANCE_SHA256,
        )


if __name__ == "__main__":
    unittest.main()
