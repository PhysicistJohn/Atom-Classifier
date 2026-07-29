from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import validate_openset_novelty_adaptive_amendment as validator  # noqa: E402


class OpensetNoveltyAdaptiveAmendmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.amendment = json.loads(
            validator.AMENDMENT_PATH.read_text(encoding="utf-8")
        )
        cls.protocol = json.loads(
            validator.RECOVERY_PROTOCOL_PATH.read_text(encoding="utf-8")
        )
        cls.registry = json.loads(
            validator.SEED_REGISTRY_PATH.read_text(encoding="utf-8")
        )
        cls.attempt_2 = json.loads(
            validator.ATTEMPT_2_INTENT_PATH.read_text(encoding="utf-8")
        )
        cls.acceptance = json.loads(
            validator.IDENTITY_ACCEPTANCE_PATH.read_text(encoding="utf-8")
        )
        cls.stage_one_policy = json.loads(
            validator.V4_STAGE_ONE_POLICY_PATH.read_text(encoding="utf-8")
        )

    def validate(
        self,
        *,
        amendment: dict | None = None,
        protocol: dict | None = None,
        registry: dict | None = None,
        attempt_2: dict | None = None,
        acceptance: dict | None = None,
        stage_one_policy: dict | None = None,
    ) -> dict:
        return validator.validate_documents(
            self.amendment if amendment is None else amendment,
            self.protocol if protocol is None else protocol,
            self.registry if registry is None else registry,
            self.attempt_2 if attempt_2 is None else attempt_2,
            self.acceptance if acceptance is None else acceptance,
            (
                self.stage_one_policy
                if stage_one_policy is None
                else stage_one_policy
            ),
        )

    def test_repository_bindings_and_exact_counts_are_valid(self) -> None:
        result = validator.validate_repository()
        self.assertTrue(result["valid"])
        self.assertTrue(result["development_only"])
        self.assertFalse(result["release_evidence"])
        self.assertEqual(
            result["adaptive_novelty_seeds"],
            [20263005, 20263006, 20263007, 20263008],
        )
        self.assertEqual(result["adaptive_novelty_exact_cells"], 72)
        self.assertEqual(result["adaptive_novelty_exact_route_scores"], 21600)
        self.assertEqual(result["known_exact_scored_observations"], 23808)
        self.assertEqual(
            result["known_exact_profile_length_scale_cells"],
            372,
        )
        self.assertEqual(result["known_gate_count"], 21)
        self.assertEqual(result["novelty_gate_count"], 6)
        self.assertEqual(result["full_gate_count"], 27)
        self.assertEqual(result["corpus_files_read"], 0)
        self.assertEqual(result["checkpoint_files_read"], 0)
        self.assertEqual(result["stage_one_policy_files_read"], 1)
        self.assertEqual(
            result["prototype_or_calibration_rows_used_as_evidence"],
            0,
        )
        self.assertEqual(result["model_inference_runs"], 0)
        self.assertEqual(result["sealed_files_read"], 0)
        self.assertEqual(
            result[
                "attempt_2_commit_ancestry_deferred_until_amendment_commit"
            ],
            not result["attempt_2_commit_ancestry_verified"],
        )

    def test_validator_opens_only_declared_static_bindings(self) -> None:
        opened: list[Path] = []
        original = Path.open

        def recording_open(path: Path, *args, **kwargs):
            opened.append(path.resolve())
            return original(path, *args, **kwargs)

        with mock.patch.object(Path, "open", recording_open):
            validator.validate_repository()
        self.assertEqual(
            set(opened),
            {path.resolve() for path in validator.ALLOWED_PATHS},
        )
        self.assertTrue(
            all(
                path.suffix in {".json", ".py", ".ts"}
                for path in opened
            )
        )
        self.assertFalse(
            any(
                "training/artifacts" in path.as_posix()
                or path.suffix in {".f32", ".npy", ".npz", ".pt", ".pth"}
                for path in opened
            )
        )

    def test_delegated_scope_does_not_authorize_generation_or_inference(
        self,
    ) -> None:
        authorization = self.amendment["delegated_authorization"]
        self.assertIn(
            "open or generate an adaptive or sealed novelty corpus",
            authorization["not_authorized_by_this_amendment"],
        )
        self.assertIn(
            "run model inference",
            authorization["not_authorized_by_this_amendment"],
        )
        changed = copy.deepcopy(self.amendment)
        changed["delegated_authorization"][
            "not_authorized_by_this_amendment"
        ].remove("run model inference")
        with self.assertRaisesRegex(ValueError, "delegated_authorization"):
            self.validate(amendment=changed)

    def test_all_four_adaptive_seeds_are_joint_and_not_cherry_picked(
        self,
    ) -> None:
        inventory = self.amendment["adaptive_novelty_inventory"]
        self.assertEqual(inventory["seeds"], validator.ADAPTIVE_NOVELTY_SEEDS)
        self.assertTrue(
            inventory["all_four_seeds_consumed_and_reported_jointly"]
        )
        self.assertFalse(inventory["seed_cherry_pick_or_omission_allowed"])

        changed = copy.deepcopy(self.amendment)
        changed["adaptive_novelty_inventory"]["seeds"] = [20263005]
        with self.assertRaisesRegex(ValueError, "adaptive_novelty_inventory"):
            self.validate(amendment=changed)

        changed = copy.deepcopy(self.amendment)
        changed["adaptive_novelty_inventory"][
            "seed_cherry_pick_or_omission_allowed"
        ] = True
        with self.assertRaisesRegex(ValueError, "adaptive_novelty_inventory"):
            self.validate(amendment=changed)

    def test_exact_novelty_cell_and_route_score_inventory_cannot_shrink(
        self,
    ) -> None:
        inventory = self.amendment["adaptive_novelty_inventory"]
        self.assertEqual(inventory["families"], ["no_signal", "noise", "chirp"])
        self.assertEqual(inventory["observation_lengths"], [4096, 8192, 16384])
        self.assertEqual(inventory["routes"], ["historical", "current"])
        self.assertEqual(inventory["rows_per_cell"], 300)
        self.assertEqual(inventory["exact_cell_count"], 72)
        self.assertEqual(inventory["exact_route_score_count"], 21600)

        for field, changed_value in (
            ("rows_per_cell", 299),
            ("exact_cell_count", 71),
            ("exact_route_score_count", 21300),
        ):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.amendment)
                changed["adaptive_novelty_inventory"][field] = changed_value
                with self.assertRaisesRegex(
                    ValueError,
                    "adaptive_novelty_inventory",
                ):
                    self.validate(amendment=changed)

    def test_max_length_generation_prefix_rule_and_digest_manifest_are_frozen(
        self,
    ) -> None:
        inventory = self.amendment["adaptive_novelty_inventory"]
        self.assertEqual(inventory["generated_once_at_maximum_length"], 16384)
        self.assertEqual(inventory["dtype"], "complex64")
        self.assertTrue(
            inventory["first_generation_digest_manifest"][
                "immutable_after_first_generation"
            ]
        )
        self.assertFalse(
            inventory["no_signal_disclosure"]["seed_independent_draws_claimed"]
        )

        changed = copy.deepcopy(self.amendment)
        changed["adaptive_novelty_inventory"][
            "shorter_observation_rule"
        ] = "independently regenerated at each length"
        with self.assertRaisesRegex(ValueError, "adaptive_novelty_inventory"):
            self.validate(amendment=changed)

    def test_generator_is_time_domain_and_v4_source_is_rationale_only(
        self,
    ) -> None:
        mechanics = self.amendment["normative_generator_mechanics"]
        rationale = self.amendment["parent_development_rationale_only"]
        self.assertFalse(mechanics["uses_frequency_transform"])
        self.assertEqual(
            mechanics["family_seed_derivation"]["message_utf8"],
            "atomos-v4-openset:{base_seed}:{family}",
        )
        self.assertTrue(rationale["not_v5_evidence"])
        self.assertTrue(rationale["may_not_supply_v5_gate_values"])

        changed = copy.deepcopy(self.amendment)
        changed["normative_generator_mechanics"][
            "uses_frequency_transform"
        ] = True
        with self.assertRaisesRegex(ValueError, "normative_generator_mechanics"):
            self.validate(amendment=changed)

        changed = copy.deepcopy(self.amendment)
        changed["parent_development_rationale_only"][
            "may_not_supply_v5_gate_values"
        ] = False
        with self.assertRaisesRegex(
            ValueError,
            "parent_development_rationale_only",
        ):
            self.validate(amendment=changed)

    def test_current_geometry_is_fixed_ratio_two_and_class_independent(
        self,
    ) -> None:
        current = self.amendment["route_geometry"]["current"]
        historical = self.amendment["route_geometry"]["historical"]
        self.assertEqual(current["sample_rate_hz"], 16_000_000)
        self.assertEqual(current["native_sample_rate_hz"], 8_000_000)
        self.assertEqual(current["sample_rate_ratio"], 2.0)
        self.assertTrue(current["class_independent"])
        self.assertTrue(current["profile_independent"])
        self.assertEqual(
            current["effective_runtime_input_length_for_every_observation_length"],
            4096,
        )
        self.assertIsNone(historical["sample_rate_hz"])
        self.assertIsNone(historical["native_sample_rate_hz"])

        changed = copy.deepcopy(self.amendment)
        changed["route_geometry"]["current"]["sample_rate_ratio"] = 1.5
        with self.assertRaisesRegex(ValueError, "route_geometry"):
            self.validate(amendment=changed)

        changed = copy.deepcopy(self.amendment)
        changed["route_geometry"]["current"]["class_independent"] = False
        with self.assertRaisesRegex(ValueError, "route_geometry"):
            self.validate(amendment=changed)

    def test_known_inventory_hashes_counts_and_score_only_role_are_exact(
        self,
    ) -> None:
        known = self.amendment["known_inventory_and_score_references"]
        corpus = known["adaptive_known_corpus"]
        firewall = known["fitting_firewall"]
        self.assertEqual(corpus["eval_seed"], 20262904)
        self.assertEqual(
            corpus["manifest_sha256"],
            "7a345bc109e46661427c2f81b46125fb48a00048e9b2b7da77505dc49135d998",
        )
        self.assertEqual(
            corpus["raw_sha256"],
            "53d4ab20faf031ed2aacd3017d55ada1622d7c51f85b200aa490f3d3854dc928",
        )
        self.assertEqual(corpus["profile_count"], 31)
        self.assertEqual(corpus["identities_per_profile"], 64)
        self.assertEqual(corpus["views_per_identity"], 12)
        self.assertEqual(corpus["exact_scored_observation_count"], 23808)
        self.assertEqual(corpus["exact_profile_length_scale_cell_count"], 372)
        zero_use_fields = [
            "seed20262904_rows_used_for_gradient_or_optimizer_step",
            "seed20262904_rows_used_for_weight_center_or_feature_moment_fit",
            "seed20262904_rows_used_for_persistent_prototype_fit",
            "seed20262904_rows_used_for_open_set_threshold_or_rank_fit",
            "adaptive_novelty_rows_used_for_any_fit_threshold_or_rank",
            "sealed_rows_used_for_any_fit_or_adaptive_selection",
        ]
        self.assertTrue(all(firewall[field] == 0 for field in zero_use_fields))
        self.assertTrue(
            firewall[
                "adaptive_novelty_outcomes_may_be_used_for_adaptive_"
                "development_candidate_selection"
            ]
        )
        self.assertTrue(
            firewall[
                "adaptive_candidate_rescoring_must_reuse_the_same_immutable_"
                "generated_bytes"
            ]
        )

        changed = copy.deepcopy(self.amendment)
        changed["known_inventory_and_score_references"][
            "adaptive_known_corpus"
        ]["exact_scored_observation_count"] = 7936
        with self.assertRaisesRegex(
            ValueError,
            "known_inventory_and_score_references",
        ):
            self.validate(amendment=changed)

    def test_known_corpus_values_must_match_identity_acceptance(self) -> None:
        changed = copy.deepcopy(self.acceptance)
        changed["accepted_replacement_corpus"]["raw_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "known corpus raw_sha256"):
            self.validate(acceptance=changed)

    def test_stage_one_is_exact_coefficient_only_inheritance(self) -> None:
        stage_one = self.amendment["fixed_stage_one_inheritance"]
        self.assertEqual(
            stage_one["coefficient_artifact_set_sha256"],
            "de2909bad49a4af49433f09616fa74e05eb7fa9fe4cd7a3aabc01139f78af74d",
        )
        self.assertEqual(stage_one["inherited_component"], "stage-one coefficients only")
        self.assertFalse(stage_one["v4_classifier_reused"])
        self.assertFalse(stage_one["v4_prototype_bank_reused"])
        self.assertFalse(stage_one["v4_thresholds_or_ranks_reused"])
        self.assertFalse(stage_one["v4_gate_results_or_validation_claims_reused"])

        changed = copy.deepcopy(self.amendment)
        changed["fixed_stage_one_inheritance"]["v4_thresholds_or_ranks_reused"] = True
        with self.assertRaisesRegex(ValueError, "fixed_stage_one_inheritance"):
            self.validate(amendment=changed)

        changed = copy.deepcopy(self.amendment)
        changed["fixed_stage_one_inheritance"]["source_v4_policy_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "fixed_stage_one_inheritance"):
            self.validate(amendment=changed)

        changed_policy = copy.deepcopy(self.stage_one_policy)
        changed_policy["stage_one"]["parameters_by_length"]["4096"][
            "coefficients"
        ][0] += 1e-6
        with self.assertRaisesRegex(ValueError, "do not reproduce"):
            self.validate(stage_one_policy=changed_policy)

    def test_device_batch_and_known_only_budget_are_frozen(self) -> None:
        execution = self.amendment["execution_contract"]
        fit = self.amendment["known_only_open_set_fit_contract"]
        self.assertEqual(execution["device"], "cpu")
        self.assertEqual(execution["embedding_batch_size"], 256)
        self.assertFalse(execution["automatic_device_selection_allowed"])
        self.assertEqual(fit["known_false_unknown_composite_budget"], 0.03)
        self.assertEqual(
            fit["adaptive_novelty_rows_used_for_threshold_or_rank_fit"],
            0,
        )

        changed = copy.deepcopy(self.amendment)
        changed["execution_contract"]["device"] = "auto"
        with self.assertRaisesRegex(ValueError, "execution_contract"):
            self.validate(amendment=changed)

        changed = copy.deepcopy(self.amendment)
        changed["known_only_open_set_fit_contract"][
            "known_false_unknown_composite_budget"
        ] = 0.05
        with self.assertRaisesRegex(
            ValueError,
            "known_only_open_set_fit_contract",
        ):
            self.validate(amendment=changed)

    def test_v5_source_hashes_are_deferred_not_fabricated(self) -> None:
        timing = self.amendment["timing_and_source_boundary"]
        self.assertFalse(timing["v5_novelty_generator_source_sha256_frozen_here"])
        self.assertFalse(timing["v5_novelty_scorer_source_sha256_frozen_here"])
        self.assertFalse(timing["v5_full_gate_source_sha256_frozen_here"])
        self.assertTrue(
            timing["generation_before_pre_generation_source_freeze_forbidden"]
        )
        self.assertTrue(
            timing["scoring_before_candidate_execution_manifest_forbidden"]
        )
        pre_generation = timing[
            "later_immutable_pre_generation_source_freeze_must_bind_exact_"
            "sha256_for"
        ]
        candidate = timing[
            "later_candidate_execution_manifest_before_each_adaptive_score_"
            "must_bind_exact_sha256_for"
        ]
        self.assertIn("final v5 novelty generator source", pre_generation)
        self.assertNotIn(
            "generated adaptive novelty provenance and immutable digest manifest",
            pre_generation,
        )
        self.assertIn("final v5 novelty generator source", candidate)
        self.assertIn("final v5 novelty scorer source", candidate)
        self.assertIn("final v5 full-gate evaluator source", candidate)
        self.assertIn(
            "final open-set calibrator and threshold-rank fitter source",
            candidate,
        )
        self.assertIn(
            "generated adaptive novelty provenance and immutable digest manifest",
            candidate,
        )
        result_manifest = timing["later_result_manifest_contract"]
        self.assertTrue(
            result_manifest[
                "binds_exact_21600_per_row_final_score_and_reject_outputs_or_"
                "canonical_content_hashes"
            ]
        )
        self.assertTrue(
            result_manifest[
                "binds_all_21_known_gate_rows_and_all_6_aggregated_novelty_"
                "gate_rows"
            ]
        )
        self.assertTrue(
            result_manifest["binds_exact_27_gate_name_inventory"]
        )
        self.assertTrue(result_manifest["binds_overall_all_pass"])

        changed = copy.deepcopy(self.amendment)
        changed["timing_and_source_boundary"][
            "v5_full_gate_source_sha256_frozen_here"
        ] = True
        with self.assertRaisesRegex(ValueError, "timing_and_source_boundary"):
            self.validate(amendment=changed)

        changed = copy.deepcopy(self.amendment)
        changed["timing_and_source_boundary"]["later_result_manifest_contract"][
            "binds_overall_all_pass"
        ] = False
        with self.assertRaisesRegex(ValueError, "timing_and_source_boundary"):
            self.validate(amendment=changed)

        changed = copy.deepcopy(self.amendment)
        changed["timing_and_source_boundary"][
            "later_candidate_execution_manifest_before_each_adaptive_score_"
            "must_bind_exact_sha256_for"
        ].remove("final open-set calibrator and threshold-rank fitter source")
        with self.assertRaisesRegex(ValueError, "timing_and_source_boundary"):
            self.validate(amendment=changed)

    def test_novelty_thresholds_are_per_family_worst_cell(self) -> None:
        novelty = self.amendment["novelty_gate_contract"]
        self.assertEqual(novelty["exact_gate_count"], 6)
        self.assertEqual(
            [row["name"] for row in novelty["gates"]],
            validator.NOVELTY_GATE_NAMES,
        )
        self.assertEqual(
            novelty["aggregation"]["method"],
            "minimum_worst_cell",
        )
        self.assertEqual(novelty["aggregation"]["cell_count_per_family"], 24)

        changed = copy.deepcopy(self.amendment)
        changed["novelty_gate_contract"]["gates"][2]["threshold"] = 0.79
        with self.assertRaisesRegex(ValueError, "novelty_gate_contract"):
            self.validate(amendment=changed)

    def test_averaging_cannot_replace_worst_cell_aggregation(self) -> None:
        changed = copy.deepcopy(self.amendment)
        changed["novelty_gate_contract"]["aggregation"]["method"] = "mean"
        with self.assertRaisesRegex(ValueError, "novelty_gate_contract"):
            self.validate(amendment=changed)

        changed = copy.deepcopy(self.amendment)
        changed["novelty_gate_contract"]["aggregation"][
            "averaging_across_seeds_routes_lengths_or_families_allowed"
        ] = True
        with self.assertRaisesRegex(ValueError, "novelty_gate_contract"):
            self.validate(amendment=changed)

        changed = copy.deepcopy(self.amendment)
        changed["novelty_gate_contract"]["aggregation"][
            "pooled_overall_auroc_may_substitute_for_family_gate"
        ] = True
        with self.assertRaisesRegex(ValueError, "novelty_gate_contract"):
            self.validate(amendment=changed)

    def test_known_21_only_can_never_silently_report_full_pass(self) -> None:
        full = self.amendment["full_gate_contract"]
        self.assertEqual(full["exact_known_gate_count"], 21)
        self.assertEqual(full["exact_novelty_gate_count"], 6)
        self.assertEqual(full["exact_total_gate_count"], 27)
        self.assertFalse(full["known_21_only_may_report_full_pass"])

        changed = copy.deepcopy(self.amendment)
        changed["full_gate_contract"]["exact_total_gate_count"] = 21
        with self.assertRaisesRegex(ValueError, "full_gate_contract"):
            self.validate(amendment=changed)

        changed = copy.deepcopy(self.amendment)
        changed["full_gate_contract"][
            "known_21_only_may_report_full_pass"
        ] = True
        with self.assertRaisesRegex(ValueError, "full_gate_contract"):
            self.validate(amendment=changed)

        changed = copy.deepcopy(self.amendment)
        changed["full_gate_contract"][
            "all_pass_rule"
        ] = "true iff all 21 known gates pass"
        with self.assertRaisesRegex(ValueError, "full_gate_contract"):
            self.validate(amendment=changed)

    def test_exact_known_and_novelty_coverage_cannot_be_relaxed(self) -> None:
        for field in (
            "exact_known_inventory_coverage_required",
            "exact_novelty_inventory_coverage_required",
            "unexpected_duplicate_or_missing_gate_name_fails",
        ):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.amendment)
                changed["full_gate_contract"][field] = False
                with self.assertRaisesRegex(ValueError, "full_gate_contract"):
                    self.validate(amendment=changed)

    def test_recovery_protocol_gate_inventory_is_cross_checked(self) -> None:
        changed = copy.deepcopy(self.protocol)
        changed["gate_contract"]["inherited_gates"] = changed[
            "gate_contract"
        ]["inherited_gates"][:-1]
        with self.assertRaisesRegex(ValueError, "known-gate contract"):
            self.validate(protocol=changed)

    def test_registry_allocations_are_cross_checked(self) -> None:
        changed = copy.deepcopy(self.registry)
        changed["fresh_allocations"]["adaptive_novelty_development"][
            "seeds"
        ] = [20263005, 20263006, 20263007]
        with self.assertRaisesRegex(ValueError, "adaptive novelty registry"):
            self.validate(registry=changed)

        changed = copy.deepcopy(self.registry)
        changed["fresh_allocations"]["sealed_novelty_validation"][
            "generation_allowed_before_candidate_freeze"
        ] = True
        with self.assertRaisesRegex(ValueError, "sealed novelty registry"):
            self.validate(registry=changed)

    def test_unused_scale_reserve_remains_untouched(self) -> None:
        reserve = self.amendment["unused_adaptive_scale_reserve"]
        self.assertEqual(reserve["seeds"], validator.UNUSED_ADAPTIVE_SCALE_SEEDS)
        self.assertEqual(
            reserve["state_under_this_amendment"],
            "allocated_unspent_and_untouched",
        )

        changed = copy.deepcopy(self.amendment)
        changed["unused_adaptive_scale_reserve"]["seeds"] = [20262905]
        with self.assertRaisesRegex(
            ValueError,
            "unused_adaptive_scale_reserve",
        ):
            self.validate(amendment=changed)

    def test_sealed_seeds_remain_unread_and_separately_frozen_later(
        self,
    ) -> None:
        sealed = self.amendment["sealed_novelty_boundary"]
        self.assertEqual(sealed["seeds"], [20263002, 20263003])
        self.assertFalse(
            sealed["this_amendment_authorizes_generation_read_or_scoring"]
        )
        self.assertTrue(sealed["candidate_freeze_required_before_generation"])
        self.assertTrue(
            sealed[
                "joint_claim_ledger_required_before_either_seed_generation_or_read"
            ]
        )
        self.assertTrue(
            sealed[
                "known_reference_population_for_sealed_stage_requires_"
                "separate_owner_authorized_freeze"
            ]
        )
        copied = sealed[
            "later_sealed_contract_is_separate_and_must_copy_unchanged"
        ]
        self.assertIn(
            (
                "route-specific metric and routing mechanics, explicitly "
                "excluding known-reference populations"
            ),
            copied,
        )
        self.assertNotIn("route-specific known-reference rules", copied)

        changed = copy.deepcopy(self.amendment)
        changed["sealed_novelty_boundary"][
            "this_amendment_authorizes_generation_read_or_scoring"
        ] = True
        with self.assertRaisesRegex(ValueError, "sealed_novelty_boundary"):
            self.validate(amendment=changed)


if __name__ == "__main__":
    unittest.main()
