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


def _supervised_pair_pool_audit() -> dict:
    runner = assembler.branch_runner
    identities_per_profile = (
        assembler.scale_orbit_data.SCALE_ORBIT_TRAINING_SPLIT_COUNTS["train"]
    )
    identity_count = (
        len(runner.SUPERVISED_PAIR_PROFILES)
        * identities_per_profile
    )
    scale_count = len(
        assembler.scale_orbit_data.SCALE_ORBIT_EXPECTED_FACTORS
    )
    return {
        "contract": runner.SUPERVISED_PAIR_PAIR_CONTRACT,
        "loss_contract": runner.SUPERVISED_PAIR_LOSS_CONTRACT,
        "source": "current",
        "population_seed":
            assembler.scale_orbit_data.SCALE_ORBIT_TRAINING_SEED,
        "role": "train",
        "literal_profile_count": len(runner.SUPERVISED_PAIR_PROFILES),
        "literal_profiles": list(runner.SUPERVISED_PAIR_PROFILES),
        "literal_profile_public_class_map":
            dict(runner.SUPERVISED_PAIR_PROFILE_PUBLIC_CLASS_MAP),
        "literal_profile_public_class_index_map": {
            profile: runner.SUPERVISED_PAIR_PROFILE_CLASS_INDICES[index]
            for index, profile in enumerate(
                runner.SUPERVISED_PAIR_PROFILES
            )
        },
        "public_class_indices_in_literal_profile_order":
            list(runner.SUPERVISED_PAIR_PROFILE_CLASS_INDICES),
        "paired_targets_sha256": runner.SUPERVISED_PAIR_TARGET_SHA256,
        "target_construction": runner.SUPERVISED_PAIR_TARGET_CONSTRUCTION,
        "identities_per_profile": {
            profile: identities_per_profile
            for profile in runner.SUPERVISED_PAIR_PROFILES
        },
        "identity_count": identity_count,
        "scale_views_per_identity": scale_count,
        "physical_scale_factors": list(
            assembler.scale_orbit_data.SCALE_ORBIT_EXPECTED_FACTORS
        ),
        "addressable_training_scale_views": identity_count * scale_count,
        "merged_training_view_count": identity_count * scale_count + 128,
        "identity_sha256": "a" * 64,
        "pair_rng": (
            "separate numpy.default_rng(seed xor 0x5CA1E); episodic RNG "
            "state is never passed to supervised-pair sampling"
        ),
        "pairs_per_episode": len(runner.SUPERVISED_PAIR_PROFILES),
        "views_per_episode": 2 * len(runner.SUPERVISED_PAIR_PROFILES),
        "fitting_firewall": {
            "seed20264101_train_identities_addressable": identity_count,
            "seed20264101_enrollment_rows_addressable": 0,
            "seed20262904_selection_rows_addressable": 0,
            "sealed_rows_addressable": 0,
        },
    }


def _branch_metrics(encoder: str = "real") -> dict:
    runner = assembler.branch_runner
    arguments = runner.build_parser().parse_args(
        [
            "--current-corpus",
            "training",
            "--current-selection-corpus",
            "selection",
            "--output-dir",
            "output",
            "--encoder",
            encoder,
            "--supervised-pair-weight",
            "0.2",
        ]
    )
    adaptation_binding = (
        runner._validate_supervised_pair_adaptation_source()
    )
    run_configuration = runner._run_configuration(
        arguments,
        device=assembler.torch.device("cpu"),
        adaptation_binding=adaptation_binding,
    )
    share = {
        "numerator": 1,
        "denominator": 3,
        "value": 1.0 / 3.0,
    }
    pool = _supervised_pair_pool_audit()
    pair_rng_seed = (
        arguments.seed ^ runner.SUPERVISED_PAIR_RNG_XOR
    )
    return {
        "status": "complete",
        "development_only": True,
        "release_evidence": False,
        "consumed_historical_test_rows_used": 0,
        "encoder": encoder,
        "architecture": {"encoder": encoder},
        "parameter_count": 1,
        "run_configuration": run_configuration,
        "training": {
            "episodes": arguments.episodes,
            "sampler_contract": runner.SAMPLER_CONTRACT,
            "current_source_share": share,
            "supervised_pair": {
                "loss_contract": runner.SUPERVISED_PAIR_LOSS_CONTRACT,
                "pair_sampling_contract":
                    runner.SUPERVISED_PAIR_PAIR_CONTRACT,
                "weight": runner.SUPERVISED_PAIR_WEIGHT,
                "auxiliary": "public_class_cross_entropy",
                "reduction": "mean_over_62_profile_contiguous_views",
                "targets": runner.SUPERVISED_PAIR_TARGET_CONSTRUCTION,
                "paired_targets_sha256":
                    runner.SUPERVISED_PAIR_TARGET_SHA256,
                "prototypes": (
                    "detached_current_episode_source_mixed_public_class_"
                    "prototypes"
                ),
                "logit_scale":
                    "detached_numeric_current_episode_logit_scale",
                "pair_rng_seed": pair_rng_seed,
                "pair_rng_seed_rule": "seed xor 0x5CA1E",
                "pair_rng_separate_from_episodic_rng": True,
                "phase_augmentation":
                    "one_independent_draw_per_paired_view",
                "batch_norm_stat_firewall": True,
                "pool": pool,
            },
        },
        "final_evaluation": {},
        "data_audit": {
            "historical": {"consumed_test_rows_exposed": 0},
            "current": _audit(),
            "preprocessing": {
                "frontend": assembler.td_preprocess.preprocess_metadata(),
                "training": {
                    "hierarchical_sampler_contract":
                        runner.SAMPLER_CONTRACT,
                    "supervised_pair_pool": pool,
                },
                "runtime_bucket_policy": {
                    "episode_rule": runner.SAMPLER_CONTRACT,
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
        "source_sha256": assembler._expected_branch_source_hashes(),
        "device": "cpu",
    }


def _runner_supervised_pair_pool_audit() -> dict:
    """Exercise the runner's concrete pool auditor without loading a corpus."""
    runner = assembler.branch_runner
    scales = tuple(
        float(value)
        for value in assembler.scale_orbit_data.SCALE_ORBIT_EXPECTED_FACTORS
    )
    identities_per_profile = int(
        assembler.scale_orbit_data.SCALE_ORBIT_TRAINING_SPLIT_COUNTS["train"]
    )
    position = 0
    pool = {}
    for profile_position, profile in enumerate(
        runner.SUPERVISED_PAIR_PROFILES
    ):
        entries = []
        for realization in range(identities_per_profile):
            view_positions = tuple(
                range(position, position + len(scales))
            )
            position += len(scales)
            entries.append(
                runner.SupervisedPairIdentity(
                    identity=f"integration:{profile}:{realization}",
                    source="current",
                    role="train",
                    population_seed=(
                        assembler.scale_orbit_data.SCALE_ORBIT_TRAINING_SEED
                    ),
                    profile_id=profile,
                    public_class_name=(
                        runner.SUPERVISED_PAIR_PROFILE_PUBLIC_CLASS_MAP[
                            profile
                        ]
                    ),
                    public_class_index=(
                        runner.SUPERVISED_PAIR_PROFILE_CLASS_INDICES[
                            profile_position
                        ]
                    ),
                    scale_factors=scales,
                    view_positions=view_positions,
                )
            )
        pool[profile] = tuple(entries)
    return runner._validate_supervised_pair_pool(
        pool,
        training_view_count=position + 128,
        classes=runner.SUPERVISED_PAIR_PUBLIC_CLASS_ORDER,
    )


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
            "v5-scale-orbit-development-training-v2",
        )
        self.assertEqual(
            assembler.FUSION_SCHEMA,
            "v5-scale-orbit-development-fusion-v2",
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
            "v5-scale-orbit-development-training-v1"
        )
        with self.assertRaisesRegex(ValueError, "exact v5"):
            assembler.validate_branch_metrics(changed, encoder="real")

    def test_branch_accepts_concrete_runner_supervised_pair_pool_audit(
        self,
    ) -> None:
        value = _branch_metrics()
        pool = _runner_supervised_pair_pool_audit()
        self.assertEqual(
            pool["pair_rng"],
            "separate numpy.default_rng(seed xor 0x5CA1E); episodic RNG "
            "state is never passed to supervised-pair sampling",
        )
        value["training"]["supervised_pair"]["pool"] = pool
        value["data_audit"]["preprocessing"]["training"][
            "supervised_pair_pool"
        ] = pool
        self.assertIs(
            assembler.validate_branch_metrics(value, encoder="real"),
            value,
        )

    def test_branch_run_metadata_schema_is_exact(self) -> None:
        mutations = (
            (
                ("run_configuration", "checkpoint_selection", "comparison"),
                "greater-than-or-equal",
            ),
            (
                ("run_configuration", "optimizer", "learning_rate"),
                0.5,
            ),
            (
                ("run_configuration", "scheduler", "warmup_fraction"),
                0.5,
            ),
            (
                ("run_configuration", "randomness", "episodic_rng"),
                "shared RNG",
            ),
            (
                ("run_configuration", "software", "resolved_device"),
                "cuda",
            ),
        )
        for path, replacement in mutations:
            with self.subTest(path=path):
                changed = copy.deepcopy(_branch_metrics())
                target = changed
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = replacement
                with self.assertRaises(ValueError):
                    assembler.validate_branch_metrics(
                        changed,
                        encoder="real",
                    )

        changed = copy.deepcopy(_branch_metrics())
        changed["run_configuration"]["arguments"][
            "undeclared_argument"
        ] = True
        with self.assertRaisesRegex(ValueError, "arguments/randomness"):
            assembler.validate_branch_metrics(changed, encoder="real")

    def test_branch_metrics_reject_adaptation_contract_mutations(self) -> None:
        mutations = (
            (
                ("run_configuration", "adaptation", "supervised_pair_weight"),
                0.1,
            ),
            (
                ("run_configuration", "adaptation", "sha256"),
                "0" * 64,
            ),
            (
                (
                    "run_configuration",
                    "adaptation",
                    "attempt_1_miss_report_sha256",
                ),
                "0" * 64,
            ),
            (
                (
                    "run_configuration",
                    "adaptation",
                    "preregistration_commit",
                ),
                "0" * 40,
            ),
            (
                ("run_configuration", "adaptation", "auxiliary"),
                "one_minus_cosine_similarity",
            ),
            (
                ("run_configuration", "adaptation", "target_construction"),
                "tile(profile_public_class_indices, 2)",
            ),
            (
                ("run_configuration", "adaptation", "prototype_gradient"),
                "not_detached",
            ),
            (
                ("run_configuration", "adaptation", "logit_scale_gradient"),
                "not_detached",
            ),
            (
                (
                    "run_configuration",
                    "adaptation",
                    "batch_norm_stat_firewall",
                ),
                False,
            ),
            (
                (
                    "run_configuration",
                    "adaptation",
                    "fitting_firewall",
                    "seed20262904_selection_rows_used_for_any_gradient_or_optimizer_step",
                ),
                1,
            ),
            (
                ("training", "supervised_pair", "pair_rng_seed"),
                1,
            ),
            (
                ("training", "supervised_pair", "targets"),
                "tile(profile_public_class_indices, 2)",
            ),
            (
                ("training", "supervised_pair", "prototypes"),
                "attached_current_episode_prototypes",
            ),
            (
                ("training", "supervised_pair", "logit_scale"),
                "attached_current_episode_logit_scale",
            ),
            (
                ("training", "supervised_pair", "batch_norm_stat_firewall"),
                False,
            ),
            (
                (
                    "training",
                    "supervised_pair",
                    "pool",
                    "literal_profile_public_class_map",
                    "wifi-hr-dsss-11m",
                ),
                "bluetooth",
            ),
            (
                (
                    "training",
                    "supervised_pair",
                    "pool",
                    "paired_targets_sha256",
                ),
                "0" * 64,
            ),
            (
                (
                    "training",
                    "supervised_pair",
                    "pool",
                    "fitting_firewall",
                    "seed20262904_selection_rows_addressable",
                ),
                1,
            ),
        )
        for path, replacement in mutations:
            with self.subTest(path=path):
                changed = copy.deepcopy(_branch_metrics())
                target = changed
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = replacement
                if path[:3] == (
                    "training",
                    "supervised_pair",
                    "pool",
                ):
                    changed["data_audit"]["preprocessing"]["training"][
                        "supervised_pair_pool"
                    ] = copy.deepcopy(
                        changed["training"]["supervised_pair"]["pool"]
                    )
                with self.assertRaisesRegex(
                    ValueError,
                    "adaptation|supervised-pair|source hashes",
                ):
                    assembler.validate_branch_metrics(
                        changed,
                        encoder="real",
                    )

    def test_attempt_1_metadata_cannot_coexist_with_attempt_2(self) -> None:
        changed = copy.deepcopy(_branch_metrics())
        changed["training"]["scale_consistency"] = {}
        with self.assertRaisesRegex(ValueError, "forbidden attempt-1"):
            assembler.validate_branch_metrics(changed, encoder="real")

        changed = copy.deepcopy(_branch_metrics())
        changed["run_configuration"]["randomness"][
            "scale_consistency_pair_rng"
        ] = copy.deepcopy(
            changed["run_configuration"]["randomness"][
                "supervised_pair_rng"
            ]
        )
        with self.assertRaisesRegex(ValueError, "supervised-pair RNG"):
            assembler.validate_branch_metrics(changed, encoder="real")

    def test_every_selection_and_sealed_fitting_firewall_is_zero(self) -> None:
        keys = (
            (
                "seed20264101_enrollment_rows_used_for_supervised_pair_"
                "auxiliary"
            ),
            (
                "seed20262904_selection_rows_used_for_any_gradient_or_"
                "optimizer_step"
            ),
            (
                "seed20262904_selection_rows_used_for_weight_center_or_"
                "feature_moment_fit"
            ),
            (
                "seed20262904_selection_rows_used_for_persistent_prototype_"
                "fit"
            ),
            (
                "seed20262904_selection_rows_used_for_open_set_threshold_or_"
                "rank_fit"
            ),
            (
                "sealed_rows_used_for_any_fit_checkpoint_or_candidate_"
                "selection"
            ),
        )
        for key in keys:
            with self.subTest(key=key):
                changed = copy.deepcopy(_branch_metrics())
                changed["run_configuration"]["adaptation"][
                    "fitting_firewall"
                ][key] = 1
                with self.assertRaisesRegex(ValueError, "attempt 2"):
                    assembler.validate_branch_metrics(
                        changed,
                        encoder="real",
                    )

    def test_profile_contiguous_repeat_targets_are_hash_bound(self) -> None:
        contract = assembler._expected_supervised_pair_targets()
        profiles = assembler.branch_runner.SUPERVISED_PAIR_PROFILES
        self.assertEqual(len(contract["profile_class_indices"]), 31)
        self.assertEqual(len(contract["paired_targets"]), 62)
        for index in range(len(profiles)):
            self.assertEqual(
                contract["paired_targets"][2 * index:2 * index + 2],
                [contract["profile_class_indices"][index]] * 2,
            )
        self.assertEqual(
            contract["paired_targets_sha256"],
            assembler.branch_runner.SUPERVISED_PAIR_TARGET_SHA256,
        )

    def test_cross_branch_adaptation_swap_is_rejected(self) -> None:
        real = {"metrics": _branch_metrics("real")}
        complex_branch = {"metrics": _branch_metrics("complex")}
        assembler.assert_branches_match(real, complex_branch)

        changed = copy.deepcopy(complex_branch)
        changed["metrics"]["run_configuration"]["adaptation"][
            "supervised_pair_weight"
        ] = 0.1
        with self.assertRaisesRegex(
            ValueError,
            "run-configuration adaptation",
        ):
            assembler.assert_branches_match(real, changed)

        changed = copy.deepcopy(complex_branch)
        changed["metrics"]["training"]["supervised_pair"][
            "pair_rng_seed"
        ] += 1
        with self.assertRaisesRegex(
            ValueError,
            "data_audit|sampler/training contracts",
        ):
            assembler.assert_branches_match(real, changed)

        changed = copy.deepcopy(complex_branch)
        changed_pool = changed["metrics"]["training"]["supervised_pair"][
            "pool"
        ]
        changed_pool["paired_targets_sha256"] = "0" * 64
        changed["metrics"]["data_audit"]["preprocessing"]["training"][
            "supervised_pair_pool"
        ] = copy.deepcopy(changed_pool)
        with self.assertRaisesRegex(
            ValueError,
            "data_audit|sampler/training contracts",
        ):
            assembler.assert_branches_match(real, changed)

        changed = copy.deepcopy(complex_branch)
        changed["metrics"]["source_sha256"][
            "v5/scale_consistency_adaptation_attempt_2.json"
        ] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source_sha256"):
            assembler.assert_branches_match(real, changed)

    def test_branch_source_snapshot_exactly_binds_attempt_2_lineage(self) -> None:
        value = _branch_metrics()
        expected = assembler._expected_branch_source_hashes()
        self.assertEqual(value["source_sha256"], expected)
        self.assertEqual(
            expected[
                "v5/scale_consistency_adaptation_attempt_1_miss_report.json"
            ],
            assembler.ATTEMPT_1_MISS_REPORT_SHA256,
        )
        self.assertEqual(
            expected["v5/scale_consistency_adaptation_attempt_2.json"],
            assembler.ATTEMPT_2_INTENT_SHA256,
        )

        changed = copy.deepcopy(value)
        changed["source_sha256"][
            "v5/scale_consistency_adaptation_attempt_2.json"
        ] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source hashes"):
            assembler.validate_branch_metrics(changed, encoder="real")

    def test_branch_arguments_are_frozen_and_cross_branch_equal(self) -> None:
        value = _branch_metrics()
        changed = copy.deepcopy(value)
        changed["run_configuration"]["arguments"]["seed"] += 1
        changed["run_configuration"]["randomness"]["seed"] += 1
        changed["run_configuration"]["randomness"]["supervised_pair_rng"][
            "seed"
        ] += 1
        changed["training"]["supervised_pair"]["pair_rng_seed"] += 1
        with self.assertRaisesRegex(ValueError, "attempt 2"):
            assembler.validate_branch_metrics(changed, encoder="real")

        real = {"metrics": _branch_metrics("real")}
        complex_branch = {"metrics": _branch_metrics("complex")}
        complex_branch["metrics"]["run_configuration"]["arguments"][
            "current_corpus"
        ] = "different-training"
        with self.assertRaisesRegex(ValueError, "argument contracts"):
            assembler.assert_branches_match(real, complex_branch)

    def test_fusion_weight_is_exactly_preregistered(self) -> None:
        self.assertEqual(
            assembler.validate_branch_weight(0.5),
            assembler.DEFAULT_WEIGHT_REAL,
        )
        for changed in (0.0, 0.49, 0.5000001, 1.0, True, float("nan")):
            with self.subTest(changed=changed):
                with self.assertRaisesRegex(ValueError, "frozen 0.5"):
                    assembler.validate_branch_weight(changed)

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
                "v5/scale_consistency_adaptation_attempt_1_miss_report.json"
            ],
            assembler.ATTEMPT_1_MISS_REPORT_SHA256,
        )
        self.assertEqual(
            hashes["v5/scale_consistency_adaptation_attempt_2.json"],
            assembler.ATTEMPT_2_INTENT_SHA256,
        )
        self.assertEqual(
            hashes[
                "v5/seed20262904_identity_firewall_acceptance.json"
            ],
            assembler.scale_orbit_data.SCALE_ORBIT_IDENTITY_ACCEPTANCE_SHA256,
        )


if __name__ == "__main__":
    unittest.main()
