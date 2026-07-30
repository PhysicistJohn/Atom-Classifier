from __future__ import annotations

from copy import deepcopy
import inspect
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import calibrate_current_source_openset as openset
import noise_prefilter


def _template() -> openset.StageOneTemplate:
    width = len(noise_prefilter.PREFILTER_FEATURES)
    models = {
        length: noise_prefilter.NoisePrefilter(
            feature_names=tuple(noise_prefilter.PREFILTER_FEATURES),
            capture_length=length,
            mean=np.linspace(-0.2, 0.2, width),
            scale=np.linspace(0.8, 1.2, width),
            coefficients=np.linspace(-0.5, 0.5, width),
            intercept=-0.1,
            threshold_score=0.0,
            fit_provenance={
                "schema": "v4-current-source-stage-one-fit-v1",
                "known_rows": 280,
                "known_public_classes": list(openset.PUBLIC_CLASSES),
                "known_profile_weighting": (
                    "0.5 total mass; equal public-class mass; equal "
                    "source/profile mass within class; equal base-identity "
                    "then view mass within source/profile"
                ),
                "fitting_noise_rows": 60,
                "fitting_seed": openset.DEFAULT_STAGE_ONE_FIT_SEED,
                "fitting_noise_uses_frequency_transform": False,
                "enrollment_rows_used_in_fit": 0,
                "selection_rows_used_in_fit": 0,
                "novelty_design_or_validation_rows_used_in_fit": 0,
                "consumed_test_rows_used_in_fit": 0,
                "v3_frozen_hyperplane_reused": False,
            },
            threshold_provenance={"population": "enroll"},
        )
        for length in openset.RUNTIME_INPUT_LENGTHS
    }
    bundle_hashes = {
        length: openset._stage_one_parameter_sha256(
            mean=model.mean,
            scale=model.scale,
            coefficients=model.coefficients,
            intercept=model.intercept,
        )
        for length, model in models.items()
    }
    return openset.StageOneTemplate(
        directory=Path("/development/prefilter"),
        models=models,
        set_sha256=openset._stage_one_parameter_set_sha256(bundle_hashes),
        bundle_sha256=bundle_hashes,
        metadata={length: {} for length in openset.RUNTIME_INPUT_LENGTHS},
    )


def _binding() -> dict:
    return {
        "kind": "v4-development-fusion",
        "source_fusion_schema": "v4-current-source-development-fusion-v1",
        "source_fusion_dev_metrics_sha256": "b" * 64,
        "source_fusion_artifacts_sha256": {"fusion_state_dict.pt": "c" * 64},
        "classifier_schema_required": openset.browser_export.BROWSER_SCHEMA,
        "classifier_schema_version_required":
            openset.browser_export.BROWSER_SCHEMA_VERSION,
        "classifier_runtime_role_required": "accepted_known_classifier",
        "browser_frontend": {
            "version": openset.td_preprocess.PREPROCESS_VERSION,
            "patch_length": 64,
            "patch_count": 16,
            "target_frac": 0.5,
            "uses_frequency_transform": False,
            "runtime_input_lengths": list(openset.RUNTIME_INPUT_LENGTHS),
            "runtime_bucket_rule": openset.browser_export.RUNTIME_BUCKET_RULE,
        },
        "browser_classifier_asset_sha256": "d" * 64,
        "browser_classifier_asset_bound": True,
        "classifier_routing": {
            "kind": "unit-test-trusted-profile-routing-v1",
            "prototype_source_option_required": True,
        },
        "supported_public_class_mask_by_prototype_source": {
            source: list(
                openset.EXPECTED_PUBLIC_CLASS_SUPPORT_BY_PROTOTYPE_SOURCE[
                    source
                ]
            )
            for source in openset.PROTOTYPE_SOURCE_ROUTES
        },
    }


def _fit_policy(rows: int = 140) -> dict:
    stage_one: dict[int, np.ndarray] = {}
    phase_linearity: dict[int, np.ndarray] = {}
    distance: dict[int, np.ndarray] = {}
    calibration_distance: dict[int, np.ndarray] = {}
    groups: dict[int, list[str]] = {}
    sources: dict[int, list[str]] = {}
    for offset, length in enumerate(openset.RUNTIME_INPUT_LENGTHS):
        stage_one[length] = np.linspace(-3.0, 2.0, rows) + offset * 0.01
        phase_linearity[length] = np.linspace(0.0, 0.8, rows)
        values = np.full((rows, len(openset.PUBLIC_CLASSES)), 1.5)
        for index in range(rows):
            source = (
                "historical" if index < rows // 2 else "current"
            )
            supported = np.flatnonzero(
                openset.EXPECTED_PUBLIC_CLASS_SUPPORT_BY_PROTOTYPE_SOURCE[
                    source
                ]
            )
            cls = int(supported[index % len(supported)])
            values[index, cls] = 0.02 + 0.5 * index / rows
            if source == "current":
                values[
                    index,
                    ~np.asarray(
                        openset
                        .EXPECTED_PUBLIC_CLASS_SUPPORT_BY_PROTOTYPE_SOURCE[
                            source
                        ],
                        dtype=bool,
                    ),
                ] = np.inf
        distance[length] = values
        calibration_distance[length] = values + 0.01
        groups[length] = [
            (
                f"{'historical' if index < rows // 2 else 'current'}:"
                f"profile-{index % 14}"
            )
            for index in range(rows)
        ]
        sources[length] = [
            "historical" if index < rows // 2 else "current"
            for index in range(rows)
        ]
    policy = openset.fit_enrollment_policy(
        stage_one_score_by_length=stage_one,
        phase_linearity_score_by_length=phase_linearity,
        class_distance_by_length=distance,
        calibration_class_distance_by_length=calibration_distance,
        enrollment_group_by_length=groups,
        enrollment_source_by_length=sources,
        stage_one_template=_template(),
        fusion_binding=_binding(),
        patch_length=64,
        patch_count=16,
        target_frac=0.5,
        design_seed=openset.DEFAULT_DESIGN_SEED,
        validation_protocol=openset.build_validation_protocol(
            classifier_asset_sha256="d" * 64,
            held_scale_manifest_sha256="1" * 64,
            held_scale_raw_sha256="2" * 64,
        ),
    )
    policy["provenance"] = {
        "source_sha256": openset._source_hashes(),
        "stage_one_v4_fitted_artifact_set_sha256":
            policy["stage_one"]["coefficient_artifact_set_sha256"],
        "stage_one_fitting_seed": openset.DEFAULT_STAGE_ONE_FIT_SEED,
        "fusion_dev_metrics_sha256":
            policy["classifier_binding"][
                "source_fusion_dev_metrics_sha256"
            ],
        "historical_data_audit_sha256": "e" * 64,
        "current_data_audit_sha256": "f" * 64,
        "consumed_test_rows_exposed": 0,
    }
    return policy


class RankAndThresholdTests(unittest.TestCase):
    def test_empirical_rank_is_strict_and_browser_reproducible(self) -> None:
        calibration = np.asarray([1.0, 2.0, 2.0, 4.0])
        query = np.asarray([0.0, 1.0, 2.0, 3.0, 5.0])
        np.testing.assert_array_equal(
            openset.empirical_rank(query, calibration),
            np.asarray([0.0, 0.0, 0.2, 0.6, 0.8]),
        )
        with self.assertRaisesRegex(ValueError, "sorted"):
            openset.empirical_rank(query, calibration[::-1])

    def test_budget_threshold_honours_budget_with_ties(self) -> None:
        scores = np.asarray([0.0, 1.0, 1.0, 1.0, 2.0])
        threshold, detail = openset.budget_threshold(scores, 0.2)
        self.assertGreater(threshold, 1.0)
        self.assertLess(threshold, 2.0)
        self.assertEqual(detail["allowance"], 1)
        self.assertEqual(detail["realised_known_false_positive_rate"], 0.2)

    def test_grouped_threshold_satisfies_every_profile_budget(self) -> None:
        scores = np.asarray([0.0, 1.0, 9.0, 10.0, 0.0, 0.1, 0.2, 0.3])
        groups = ["tail"] * 4 + ["quiet"] * 4
        threshold, detail = openset.grouped_budget_threshold(
            scores,
            groups,
            pooled_budget=0.25,
            group_budget=0.25,
        )
        self.assertGreater(threshold, 9.0)
        self.assertLessEqual(
            detail["worst_group_realised_false_positive_rate"], 0.25
        )


class SeedAndNoveltyTests(unittest.TestCase):
    def test_seed_roles_are_disjoint_and_old_namespaces_refused(self) -> None:
        self.assertEqual(
            openset.validate_seed_plan(
                "design",
                [openset.DEFAULT_DESIGN_SEED],
                design_seed=openset.DEFAULT_DESIGN_SEED,
            ),
            (openset.DEFAULT_DESIGN_SEED,),
        )
        self.assertEqual(
            openset.validate_seed_plan(
                "validate",
                openset.DEFAULT_VALIDATION_SEEDS,
                design_seed=openset.DEFAULT_DESIGN_SEED,
            ),
            openset.DEFAULT_VALIDATION_SEEDS,
        )
        with self.assertRaisesRegex(ValueError, "redraw"):
            openset.validate_seed_plan(
                "validate",
                [openset.DEFAULT_DESIGN_SEED],
                design_seed=openset.DEFAULT_DESIGN_SEED,
            )
        with self.assertRaisesRegex(ValueError, "unavailable"):
            openset.validate_seed(20_260_955)

    def test_novelty_is_deterministic_prefix_matched_and_transform_free(
        self,
    ) -> None:
        first, provenance = openset.generate_novelty_raw(
            20_265_099, n_each=3, _unit_test_only=True
        )
        second, _ = openset.generate_novelty_raw(
            20_265_099, n_each=3, _unit_test_only=True
        )
        for family in openset.NOVELTY_FAMILIES:
            for row_index in range(3):
                longest = first[family][16_384][row_index]
                np.testing.assert_array_equal(
                    first[family][4_096][row_index], longest[:4_096]
                )
                np.testing.assert_array_equal(
                    first[family][8_192][row_index], longest[:8_192]
                )
                np.testing.assert_array_equal(
                    first[family][16_384][row_index],
                    second[family][16_384][row_index],
                )
            self.assertFalse(
                provenance[family]["uses_frequency_transform"]
            )
            self.assertTrue(provenance[family]["unit_test_only"])
        generator_source = (
            inspect.getsource(openset._noise_row)
            + inspect.getsource(openset._chirp_row)
            + inspect.getsource(openset.generate_novelty_raw)
        )
        self.assertNotIn("np.fft", generator_source)
        self.assertNotIn("scipy.fft", generator_source)


class StageOneContractTests(unittest.TestCase):
    def test_checked_in_prefilter_contract_is_audit_compatible_only(self) -> None:
        template = openset.load_frozen_v3_stage_one_for_audit(
            openset.FROZEN_V3_PREFILTER_AUDIT_DIR
        )
        self.assertEqual(
            tuple(sorted(template.models)),
            openset.RUNTIME_INPUT_LENGTHS,
        )
        self.assertEqual(len(template.set_sha256), 64)

    def test_stage_one_score_is_scale_and_phase_invariant(self) -> None:
        template = openset.load_frozen_v3_stage_one_for_audit(
            openset.FROZEN_V3_PREFILTER_AUDIT_DIR
        )
        rng = np.random.default_rng(774)
        raw = (
            rng.standard_normal(4_096)
            + 1j * rng.standard_normal(4_096)
        ).astype(np.complex64)
        transformed = raw * np.complex64(7.0 * np.exp(1j * 0.93))
        score = openset.stage_one_scores(
            template,
            [raw, transformed],
            length=4_096,
            patch_length=64,
            target_frac=0.5,
        )
        self.assertAlmostEqual(float(score[0]), float(score[1]), places=5)

    def test_v4_refit_is_class_profile_base_balanced_and_not_v3(self) -> None:
        rng = np.random.default_rng(919)
        width = openset.pose_degeneracy.FEATURE_COUNT
        known = rng.normal(0.0, 1.0, size=(60, width))
        noise = rng.normal(1.0, 1.0, size=(40, width))
        groups = ["current:rare"] * 10 + ["historical:common"] * 50
        classes = ["dsss"] * 10 + ["ofdm"] * 50
        identities = [f"base-{index}" for index in range(60)]
        weights = openset._balanced_known_weights(
            groups, classes, identities
        )
        self.assertAlmostEqual(float(np.sum(weights[:10])), 0.25)
        self.assertAlmostEqual(float(np.sum(weights[10:])), 0.25)
        model = openset._fit_v4_stage_one_model(
            known,
            groups,
            classes,
            identities,
            noise,
            length=4_096,
            fitting_seed=openset.DEFAULT_STAGE_ONE_FIT_SEED,
        )
        self.assertFalse(model.fit_provenance["v3_frozen_hyperplane_reused"])
        self.assertEqual(
            model.fit_provenance["known_profile_weighting"],
            "0.5 total mass; equal public-class mass; equal source/profile "
            "mass within class; equal base-identity then view mass within "
            "source/profile",
        )
        self.assertEqual(model.capture_length, 4_096)


class InstantaneousPhaseLinearityTests(unittest.TestCase):
    @staticmethod
    def _linear_fm(length: int = 16_384) -> np.ndarray:
        sample = np.arange(length, dtype=np.float64)
        normalized = sample / (length - 1)
        envelope = 0.55 + 0.35 * np.cos(
            2.0 * np.pi * 3.0 * normalized + 0.4
        )
        phase = 2.0 * np.pi * (
            0.17 * sample + 0.31 * sample * normalized
        )
        return np.asarray(envelope * np.exp(1j * phase), dtype=np.complex64)

    def test_score_is_prefix_scale_phase_and_carrier_invariant(self) -> None:
        raw = self._linear_fm()
        scores = []
        for length in openset.RUNTIME_INPUT_LENGTHS:
            prefix = raw[:length]
            sample = np.arange(length, dtype=np.float64)
            transformed = (
                prefix.astype(np.complex128)
                * 1e4
                * np.exp(1j * (0.71 + 0.013 * sample))
            )
            first = openset.instantaneous_phase_linearity_chirp_score(
                prefix
            )
            second = openset.instantaneous_phase_linearity_chirp_score(
                transformed
            )
            self.assertAlmostEqual(first, second, places=11)
            self.assertGreater(first, 0.999999999)
            scores.append(first)
        self.assertLess(max(scores) - min(scores), 1e-10)

    def test_cw_variance_floor_and_time_scaling_contract(self) -> None:
        sample = np.arange(4_096, dtype=np.float64)
        cw = np.exp(1j * (0.2 * sample + 0.3))
        self.assertEqual(
            openset.instantaneous_phase_linearity_chirp_score(cw),
            0.0,
        )
        scores = []
        for length in openset.RUNTIME_INPUT_LENGTHS:
            coordinate = np.arange(length, dtype=np.float64)
            normalized = coordinate / (length - 1)
            # Changing N linearly rescales time and chirp slope, while R²
            # remains a length-free linearity statistic.
            phase = 2.0 * np.pi * (
                -0.11 * coordinate
                + 0.23 * coordinate * normalized
            )
            scores.append(
                openset.instantaneous_phase_linearity_chirp_score(
                    np.exp(1j * phase)
                )
            )
        self.assertTrue(all(value > 0.999999999 for value in scores))

    def test_envelope_noise_and_phase_jitter_robustness_is_non_gating(
        self,
    ) -> None:
        raw = self._linear_fm().astype(np.complex128)
        sample = np.arange(len(raw), dtype=np.float64)
        rng = np.random.default_rng(20_265_711)
        additive = (
            rng.standard_normal(len(raw))
            + 1j * rng.standard_normal(len(raw))
        ) * 0.01
        jitter = 0.015 * np.sin(2.0 * np.pi * sample / 701.0)
        perturbed = raw * np.exp(1j * jitter) + additive
        score = openset.instantaneous_phase_linearity_chirp_score(
            perturbed
        )
        self.assertGreater(score, 0.95)
        source = inspect.getsource(
            openset.instantaneous_phase_linearity_chirp_score
        )
        self.assertNotIn("fft", source.lower())


class LeaveOneIdentityOutTests(unittest.TestCase):
    def test_runtime_public_distances_support_partial_current_route(self) -> None:
        labels = np.asarray(
            [0, 1, 2, 3, 4, 5, 6, 1, 3, 5, 6], dtype=np.int64
        )
        bank = np.stack(
            [
                np.asarray([float(index), float(index) + 0.25])
                for index in range(len(labels))
            ]
        ).astype(np.float32)
        groups = [
            {
                "source": "historical" if index < 7 else "current",
                "profile": f"profile-{index}",
                "public_class": openset.PUBLIC_CLASSES[int(labels[index])],
            }
            for index in range(len(labels))
        ]
        context = SimpleNamespace(
            fusion={
                "prototype_bank": bank,
                "prototype_labels": labels,
                "prototype_groups": groups,
            }
        )
        distances, closest, support = openset._public_distances(
            context,
            bank[[8]],
            prototype_source="current",
        )
        self.assertEqual(
            support.tolist(),
            [False, True, False, True, False, True, True],
        )
        self.assertTrue(np.isinf(distances[0, [0, 2, 4]]).all())
        self.assertEqual(closest[0, [0, 2, 4]].tolist(), [-1, -1, -1])
        self.assertEqual(float(distances[0, 3]), 0.0)
        self.assertEqual(int(closest[0, 3]), 8)

    def test_all_views_of_same_base_identity_are_excluded(self) -> None:
        embeddings: list[list[float]] = []
        groups: list[tuple[str, str]] = []
        identities: list[str] = []
        labels: list[int] = []
        metadata: list[tuple[str, str]] = []
        bank: list[np.ndarray] = []
        for class_index in range(len(openset.PUBLIC_CLASSES)):
            group = ("source", f"profile-{class_index}")
            metadata.append(group)
            if class_index == 0:
                rows = np.asarray(
                    [[0.0, 0.0], [0.2, 0.0], [2.0, 0.0], [2.2, 0.0]]
                )
                row_identities = ["base-a", "base-a", "base-b", "base-b"]
            else:
                rows = np.asarray(
                    [
                        [10.0 * class_index, 0.0],
                        [10.0 * class_index + 0.2, 0.0],
                    ]
                )
                row_identities = [
                    f"base-{class_index}-a",
                    f"base-{class_index}-b",
                ]
            embeddings.extend(rows.tolist())
            groups.extend([group] * len(rows))
            identities.extend(row_identities)
            bank.append(rows.mean(axis=0))
            labels.append(class_index)
        output = openset.leave_one_base_identity_out_class_distances(
            np.asarray(embeddings, dtype=np.float32),
            groups,
            identities,
            np.asarray(bank, dtype=np.float32),
            np.asarray(labels, dtype=np.int64),
            metadata,
        )
        # Row zero excludes both base-a views, so its own profile centroid is
        # mean([2.0, 2.2]) = 2.1 and squared distance is 4.41.
        self.assertAlmostEqual(float(output[0, 0]), 4.41, places=5)
        full_centroid_distance = (0.0 - 1.1) ** 2
        self.assertGreater(float(output[0, 0]), full_centroid_distance)


class PolicyTests(unittest.TestCase):
    def test_live_assets_are_refused_for_input_and_output(self) -> None:
        live = openset.REPO / "src" / "embedding" / "assets" / "v4.json"
        for role in ("browser classifier input", "output"):
            with self.assertRaisesRegex(ValueError, "live classifier"):
                openset.reject_sensitive_path(live, role)

    def test_fit_signature_excludes_selection_and_novelty(self) -> None:
        parameters = inspect.signature(
            openset.fit_enrollment_policy
        ).parameters
        self.assertNotIn("selection", parameters)
        self.assertNotIn("current_selection", parameters)
        self.assertNotIn("novelty", parameters)
        self.assertIn("stage_one_score_by_length", parameters)
        self.assertIn("phase_linearity_score_by_length", parameters)
        self.assertIn("class_distance_by_length", parameters)

    def test_calibration_prediction_is_leave_one_identity_out(self) -> None:
        rows = 140
        stage_one: dict[int, np.ndarray] = {}
        phase_linearity: dict[int, np.ndarray] = {}
        production: dict[int, np.ndarray] = {}
        calibration: dict[int, np.ndarray] = {}
        groups: dict[int, list[str]] = {}
        sources: dict[int, list[str]] = {}
        for length in openset.RUNTIME_INPUT_LENGTHS:
            stage_one[length] = np.linspace(-2.0, 2.0, rows)
            phase_linearity[length] = np.linspace(0.0, 0.8, rows)
            distances = np.full(
                (rows, len(openset.PUBLIC_CLASSES)), 5.0
            )
            for index in range(rows):
                source = (
                    "historical" if index < rows // 2 else "current"
                )
                support = np.asarray(
                    openset
                    .EXPECTED_PUBLIC_CLASS_SUPPORT_BY_PROTOTYPE_SOURCE[
                        source
                    ],
                    dtype=bool,
                )
                supported = np.flatnonzero(support)
                distances[index, ~support] = np.inf
                distances[index, supported[index % len(supported)]] = 0.1
            production[length] = distances
            calibration[length] = distances.copy()
            groups[length] = [
                (
                    f"{'historical' if index < rows // 2 else 'current'}:"
                    f"profile-{index % 14}"
                )
                for index in range(rows)
            ]
            sources[length] = [
                "historical" if index < rows // 2 else "current"
                for index in range(rows)
            ]
        # Self-included bank predicts class zero, while the cross-fit bank
        # predicts class one for this identity.
        calibration[4_096][0, 0] = 9.0
        calibration[4_096][0, 1] = 0.05
        policy = openset.fit_enrollment_policy(
            stage_one_score_by_length=stage_one,
            phase_linearity_score_by_length=phase_linearity,
            class_distance_by_length=production,
            calibration_class_distance_by_length=calibration,
            enrollment_group_by_length=groups,
            enrollment_source_by_length=sources,
            stage_one_template=_template(),
            fusion_binding=_binding(),
            patch_length=64,
            patch_count=16,
            target_frac=0.5,
            design_seed=openset.DEFAULT_DESIGN_SEED,
            validation_protocol=openset.build_validation_protocol(
                classifier_asset_sha256="d" * 64,
                held_scale_manifest_sha256="1" * 64,
                held_scale_raw_sha256="2" * 64,
            ),
        )
        stage_two = policy["per_prototype_source"]["historical"][
            "per_length"
        ]["4096"]["stage_two"]
        self.assertEqual(
            stage_two["predicted_class_calibration_counts"][0], 9
        )
        self.assertEqual(
            stage_two["predicted_class_calibration_counts"][1], 11
        )
        self.assertAlmostEqual(
            stage_two["self_included_prediction_change_rate"],
            1.0 / (rows // 2),
        )

    def test_policy_scores_exact_zero_then_composite_without_hard_gate(
        self,
    ) -> None:
        policy = openset.validate_policy(
            _fit_policy(), expected_fusion_binding=_binding()
        )
        length = 4_096
        stage_one = np.asarray([-3.0, 8.0, -2.0])
        distances = np.full((3, len(openset.PUBLIC_CLASSES)), 2.0)
        distances[:, 3] = np.asarray([0.2, 0.2, 10.0])
        distances[2, 5] = 0.1
        scored = openset._score_from_distances(
            policy,
            length=length,
            prototype_source="historical",
            stage_one_score=stage_one,
            phase_linearity_score=np.asarray([0.1, 1.0, 0.2]),
            class_distances=distances,
        )
        self.assertFalse(np.any(scored.stage_one))
        self.assertEqual(scored.prediction[1], 3)
        self.assertEqual(scored.prediction[0], 3)
        self.assertEqual(scored.prediction[2], 5)
        self.assertTrue(scored.stage_two[1])
        self.assertTrue(scored.rejected[1])
        self.assertLessEqual(float(np.max(scored.final_score)), 1.0)
        self.assertFalse(
            policy["stage_one"]["v3_frozen_hyperplanes_reused"]
        )
        self.assertFalse(
            policy["stage_one"]["gates_before_classification"]
        )
        self.assertEqual(
            policy["browser_contract"][
                "runtime_bucket_examples_by_prototype_source"
            ]["current"]["valid_sample_count_12160"],
            4_096,
        )

        zeros = [np.zeros(length, dtype=np.complex64) for _ in range(2)]
        no_signal = openset.score_raw_population(
            None,
            policy,
            zeros,
            length=length,
            prototype_source="historical",
        )
        self.assertTrue(np.all(no_signal.stage_zero))
        self.assertTrue(np.all(no_signal.rejected))
        np.testing.assert_array_equal(no_signal.final_score, [4.0, 4.0])

    def test_runtime_length_policy_is_route_scoped_and_fail_closed(
        self,
    ) -> None:
        policy = openset.validate_policy(_fit_policy())
        self.assertEqual(
            set(policy["per_prototype_source"]["current"]["per_length"]),
            {"4096"},
        )
        self.assertEqual(
            set(
                policy["per_prototype_source"]["historical"]["per_length"]
            ),
            {"4096", "8192", "16384"},
        )
        for observed in openset.RUNTIME_INPUT_LENGTHS:
            self.assertEqual(
                openset.effective_runtime_input_length(
                    policy,
                    prototype_source="current",
                    observation_length=observed,
                ),
                4_096,
            )
            self.assertEqual(
                openset.effective_runtime_input_length(
                    policy,
                    prototype_source="historical",
                    observation_length=observed,
                ),
                observed,
            )
        tampered = _fit_policy()
        tampered["runtime_length_policy_by_prototype_source"]["current"][
            "effective_runtime_input_length"
        ] = 8_192
        with self.assertRaisesRegex(ValueError, "runtime-length"):
            openset.effective_runtime_input_length(
                tampered,
                prototype_source="current",
                observation_length=16_384,
            )

    def test_current_long_observation_is_rescored_from_raw_first_4096(
        self,
    ) -> None:
        policy = openset.validate_policy(_fit_policy())
        raw = np.arange(16_384, dtype=np.float32).astype(
            np.complex64
        )[None, :]
        sentinel = object()
        with mock.patch.object(
            openset, "score_raw_population", return_value=sentinel
        ) as score:
            observed = openset.score_observed_raw_population(
                None,
                policy,
                list(raw),
                observation_length=16_384,
                prototype_source="current",
            )
        self.assertIs(observed, sentinel)
        effective_rows = score.call_args.args[2]
        self.assertEqual(score.call_args.kwargs["length"], 4_096)
        self.assertEqual(effective_rows[0].shape, (4_096,))
        np.testing.assert_array_equal(effective_rows[0], raw[0, :4_096])

    def test_stage_one_uses_maximum_of_class_and_route_length_pooled_ranks(
        self,
    ) -> None:
        policy = _fit_policy()
        row = policy["per_prototype_source"]["historical"]["per_length"][
            "4096"
        ]
        row["stage_one"][
            "score_rank_calibration_by_predicted_class_sorted"
        ][0] = [100.0]
        row["stage_one"][
            "pooled_score_rank_calibration_sorted"
        ] = [0.0, 1.0]
        row["composition"][
            "threshold_by_predicted_public_class"
        ][0] = 0.5
        distances = np.full((1, len(openset.PUBLIC_CLASSES)), 2.0)
        distances[0, 0] = 0.0
        scored = openset._score_from_distances(
            policy,
            length=4_096,
            prototype_source="historical",
            stage_one_score=np.asarray([2.0]),
            phase_linearity_score=np.asarray([0.0]),
            class_distances=distances,
        )
        self.assertEqual(
            float(scored.stage_one_predicted_class_rank[0]), 0.0
        )
        self.assertAlmostEqual(
            float(scored.stage_one_pooled_rank[0]), 2.0 / 3.0
        )
        self.assertAlmostEqual(
            float(scored.stage_one_rank[0]), 2.0 / 3.0
        )
        self.assertTrue(bool(scored.rejected[0]))

    def test_policy_validation_rejects_split_contamination(self) -> None:
        policy = _fit_policy()
        policy["fitting_contract"]["current_selection_rows_used"] = 1
        with self.assertRaisesRegex(ValueError, "contaminated"):
            openset.validate_policy(policy)

    def test_policy_validation_rejects_tampered_stage_one_parameters(
        self,
    ) -> None:
        policy = _fit_policy()
        policy["stage_one"]["parameters_by_length"]["4096"][
            "intercept"
        ] += 0.25
        with self.assertRaisesRegex(ValueError, "parameter hash"):
            openset.validate_policy(policy)

    def test_policy_validation_rejects_frontend_and_threshold_tampering(
        self,
    ) -> None:
        frontend = _fit_policy()
        frontend["frontend"]["patch_length"] = 32
        with self.assertRaisesRegex(ValueError, "binding"):
            openset.validate_policy(frontend)

        threshold = _fit_policy()
        threshold["per_prototype_source"]["historical"]["per_length"][
            "4096"
        ]["composition"][
            "threshold_by_predicted_public_class"
        ] = [-1.0] * len(openset.PUBLIC_CLASSES)
        with self.assertRaisesRegex(ValueError, "threshold"):
            openset.validate_policy(threshold)

        stage_zero = _fit_policy()
        stage_zero["stage_zero"]["decision"] = (
            "reject as no_signal iff max(abs(iq)) == 0"
        )
        with self.assertRaisesRegex(ValueError, "stage zero"):
            openset.validate_policy(stage_zero)

    def test_policy_validation_rejects_stage_one_fit_contamination(
        self,
    ) -> None:
        policy = _fit_policy()
        policy["stage_one"]["coefficient_fit_provenance_by_length"][
            "8192"
        ]["selection_rows_used_in_fit"] = 1
        with self.assertRaisesRegex(ValueError, "fit provenance"):
            openset.validate_policy(policy)

    def test_policy_serialization_is_deterministic_json(self) -> None:
        policy = openset.validate_policy(_fit_policy())
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first.json"
            second = Path(temporary) / "second.json"
            openset._write_json(first, policy)
            openset._write_json(second, policy)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(openset._load_json(first)["schema"], openset.POLICY_SCHEMA)


class EvaluationGateTests(unittest.TestCase):
    def test_current_causal_prefix_agreement_joins_exact_stored_rows(
        self,
    ) -> None:
        record = SimpleNamespace(
            source_index=17,
            identity="current:base:one",
            profile_id="wifi-ofdm-20m",
            valid_sample_count=8_192,
        )
        records = {4_096: [record], 8_192: [record]}
        first = openset._empty_population_scores(1)
        second = openset._empty_population_scores(1)
        first.prediction[:] = 6
        second.prediction[:] = 3
        result = openset._current_selection_causal_prefix_agreement(
            records, {4_096: first, 8_192: second}
        )
        self.assertTrue(result["exact_expected_observed_cell_coverage"])
        self.assertEqual(result["expected_observation_length_cells"], 2)
        self.assertEqual(
            result["prediction_or_abstention_agreement_rate"], 0.0
        )
        self.assertEqual(result["failing_source_indices"], [17])

    @staticmethod
    def _perfect_held_classifier_report(
        *,
        observed_ble_16k_scales: tuple[float, ...] = (1.5, 2.0),
    ) -> dict:
        row_count = 24
        ble_profile = "current:bluetooth-le-advertising"

        def profile_is_expected(
            profile: str,
            length: int,
            scale: float,
        ) -> bool:
            return not (
                profile == ble_profile
                and length == 16_384
                and scale in (1.0, 1.25)
            )

        def profile_is_observed(
            profile: str,
            length: int,
            scale: float,
        ) -> bool:
            if profile == ble_profile and length == 16_384:
                return scale in observed_ble_16k_scales
            return True

        expected_cell_rows = {}
        observed_cell_rows = {}
        per_length_scale = {}
        combined_rows = 0
        for length in openset.RUNTIME_INPUT_LENGTHS:
            length_rows = {}
            for scale in openset.scale_eval.SCALE_FACTORS:
                profiles = {}
                class_counts = {}
                for raw_profile in openset.corpus_data.CURRENT_PROFILES:
                    profile = f"current:{raw_profile}"
                    expected_key = openset._profile_legal_cell_key(
                        profile, length, scale
                    )
                    if profile_is_expected(profile, length, scale):
                        expected_cell_rows[expected_key] = row_count
                    if not profile_is_observed(profile, length, scale):
                        continue
                    observed_cell_rows[expected_key] = row_count
                    profiles[profile] = {
                        "rows": row_count,
                        "closed_head_correct_rows": row_count,
                        "closed_head_accuracy": 1.0,
                    }
                    public_class = (
                        openset.corpus_data
                        .CURRENT_PROFILE_PUBLIC_CLASS_MAP[raw_profile]
                    )
                    class_counts[public_class] = (
                        class_counts.get(public_class, 0) + row_count
                    )
                rows = sum(
                    profile_row["rows"]
                    for profile_row in profiles.values()
                )
                combined_rows += rows
                length_rows[str(scale)] = {
                    "rows": rows,
                    "closed_head_correct_rows": rows,
                    "closed_head_accuracy": 1.0,
                    "per_true_public_class": {
                        public_class: {
                            "rows": count,
                            "closed_head_correct_rows": count,
                            "closed_head_accuracy": 1.0,
                        }
                        for public_class, count in class_counts.items()
                    },
                    "per_source_profile": profiles,
                }
            per_length_scale[str(length)] = length_rows
        return {
            "per_length_scale": per_length_scale,
            "combined": {
                "rows": combined_rows,
                "pooled_closed_head_correct_rows": combined_rows,
                "pooled_closed_head_accuracy": 1.0,
            },
            "paired_length_scale_invariance": {
                "expected_pair_cells":
                    openset.EXPECTED_HELD_PAIR_CELLS,
                "cells": openset.EXPECTED_HELD_PAIR_CELLS,
                "expected_observation_cells":
                    sum(expected_cell_rows.values()),
                "observed_observation_cells":
                    sum(observed_cell_rows.values()),
                "expected_profile_legal_cell_rows": expected_cell_rows,
                "observed_profile_legal_cell_rows": observed_cell_rows,
                "expected_profile_legal_cells": len(expected_cell_rows),
                "observed_profile_legal_cells": len(observed_cell_rows),
                "expected_profile_legal_observations":
                    sum(expected_cell_rows.values()),
                "observed_profile_legal_observations":
                    sum(observed_cell_rows.values()),
                "complete_pair_coverage_rate": 1.0,
            },
        }

    @staticmethod
    def _valid_held_gate_summary_inputs() -> tuple[dict, dict]:
        profiles = [
            f"current:{profile}"
            for profile in openset.corpus_data.CURRENT_PROFILES
        ]
        physical_expected = {}
        physical_cells = []
        causal_expected = {}
        causal_by_profile = {}
        paired_by_profile = {}
        for profile in profiles:
            pair_ids = [
                f"{profile}:pair-{index}"
                for index in range(
                    openset.DEFAULT_HELD_SCALE_REALIZATIONS_PER_PROFILE
                )
            ]
            for pair_id in pair_ids:
                for length in openset.RUNTIME_INPUT_LENGTHS:
                    legal_scales = (
                        [1.5, 2.0]
                        if (
                            profile
                            == "current:bluetooth-le-advertising"
                            and length == 16_384
                        )
                        else [
                            float(scale)
                            for scale in openset.scale_eval.SCALE_FACTORS
                        ]
                    )
                    cell_key = f"{pair_id}|N={length}"
                    physical_expected[cell_key] = legal_scales
                    physical_cells.append(
                        {
                            "pair_id": pair_id,
                            "source_profile": profile,
                            "true_public_class": "bluetooth",
                            "observation_length": length,
                            "expected_legal_scale_factors":
                                list(legal_scales),
                            "observed_legal_scale_factors":
                                list(legal_scales),
                            "distinct_legal_scale_count":
                                len(legal_scales),
                            "exact_legal_scale_coverage": True,
                            "all_scale_abstention_decisions_agree": True,
                            "all_scale_predictions_or_abstentions_agree":
                                True,
                            "maximum_final_score_spread": 0.0,
                        }
                    )
                for scale in openset.scale_eval.SCALE_FACTORS:
                    legal_lengths = (
                        [4_096, 8_192]
                        if (
                            profile
                            == "current:bluetooth-le-advertising"
                            and float(scale) in (1.0, 1.25)
                        )
                        else list(openset.RUNTIME_INPUT_LENGTHS)
                    )
                    causal_expected[
                        f"{pair_id}|scale={float(scale):g}"
                    ] = legal_lengths
            causal_by_profile[profile] = {
                "expected_pair_scale_cells":
                    len(pair_ids) * len(openset.scale_eval.SCALE_FACTORS),
                "observed_pair_scale_cells":
                    len(pair_ids) * len(openset.scale_eval.SCALE_FACTORS),
                "exact_cell_coverage": True,
                "minimum_distinct_observation_lengths": (
                    2
                    if profile == "current:bluetooth-le-advertising"
                    else 3
                ),
                "prediction_or_abstention_agreement_rate": 1.0,
                "failing_pair_scale_cells": [],
            }
            paired_by_profile[profile] = {
                "expected_pair_cells": len(pair_ids),
                "cells": len(pair_ids),
                "complete_pair_coverage_rate": 1.0,
                "minimum_distinct_runtime_lengths": 2,
                "prediction_or_abstention_agreement_rate": 1.0,
                "failing_pair_ids": [],
            }
        self_contained_cell = {
            "value": 0.0,
            "source": "held",
            "length": 4_096,
            "scale_factor": 1.0,
            "name": "ofdm",
            "rows": 24,
            "closed_head_accuracy": 1.0,
            "closed_head_correct_rows": 24,
            "rejected_closed_head_correct_rows": 0,
            "abstention_damage_given_closed_head_correct": 0.0,
            "correct_and_accepted_rate": 1.0,
        }
        known = {
            "combined": {
                "per_length_false_unknown_rate": {
                    str(length): 0.0
                    for length in openset.RUNTIME_INPUT_LENGTHS
                },
                "per_length_abstention_damage_given_closed_head_correct": {
                    str(length): 0.0
                    for length in openset.RUNTIME_INPUT_LENGTHS
                },
                "per_length_closed_head_correct_rows": {
                    str(length): 1_000
                    for length in openset.RUNTIME_INPUT_LENGTHS
                },
                "pooled_abstention_damage_given_closed_head_correct": 0.0,
                "pooled_closed_head_correct_rows": 3_000,
                "pooled_closed_head_accuracy": 1.0,
                "per_length_closed_head_accuracy": {
                    str(length): 1.0
                    for length in openset.RUNTIME_INPUT_LENGTHS
                },
                "pooled_false_unknown_rate": 0.0,
                "per_length_correct_and_accepted_rate": {
                    str(length): 1.0
                    for length in openset.RUNTIME_INPUT_LENGTHS
                },
                "pooled_correct_and_accepted_rate": 1.0,
            },
            "paired_physical_scale_invariance": {
                "expected_cells":
                    openset.EXPECTED_HELD_PHYSICAL_SCALE_CELLS,
                "cells": openset.EXPECTED_HELD_PHYSICAL_SCALE_CELLS,
                "expected_observation_cells":
                    openset.EXPECTED_HELD_OBSERVATION_CELLS,
                "observed_observation_cells":
                    openset.EXPECTED_HELD_OBSERVATION_CELLS,
                "expected_legal_scales_by_pair_observation_length":
                    physical_expected,
                "observed_legal_scales_by_pair_observation_length":
                    deepcopy(physical_expected),
                "exact_eligible_cell_coverage": True,
                "minimum_distinct_legal_scales": 2,
                "prediction_or_abstention_agreement_rate": 1.0,
                "by_observation_length": {
                    str(length): {
                        "expected_cells":
                            openset
                            .EXPECTED_HELD_PHYSICAL_SCALE_CELLS_PER_LENGTH,
                        "observed_cells":
                            openset
                            .EXPECTED_HELD_PHYSICAL_SCALE_CELLS_PER_LENGTH,
                        "minimum_distinct_legal_scales": (
                            2 if length == 16_384 else 4
                        ),
                        "prediction_or_abstention_agreement_rate": 1.0,
                    }
                    for length in openset.RUNTIME_INPUT_LENGTHS
                },
                "pair_observation_length_cells": physical_cells,
            },
            "causal_prefix_length_invariance": {
                "expected_cells":
                    openset.EXPECTED_HELD_CAUSAL_PREFIX_CELLS,
                "cells": openset.EXPECTED_HELD_CAUSAL_PREFIX_CELLS,
                "expected_observation_cells":
                    openset.EXPECTED_HELD_OBSERVATION_CELLS,
                "observed_observation_cells":
                    openset.EXPECTED_HELD_OBSERVATION_CELLS,
                "expected_observation_lengths_by_pair_scale":
                    causal_expected,
                "observed_observation_lengths_by_pair_scale":
                    deepcopy(causal_expected),
                "exact_eligible_cell_coverage": True,
                "minimum_distinct_observation_lengths": 2,
                "prediction_or_abstention_agreement_rate": 1.0,
                "by_source_profile": causal_by_profile,
            },
            "paired_length_scale_invariance": {
                "expected_pair_cells": openset.EXPECTED_HELD_PAIR_CELLS,
                "cells": openset.EXPECTED_HELD_PAIR_CELLS,
                "expected_observation_cells":
                    openset.EXPECTED_HELD_OBSERVATION_CELLS,
                "observed_observation_cells":
                    openset.EXPECTED_HELD_OBSERVATION_CELLS,
                "complete_pair_coverage_rate": 1.0,
                "prediction_or_abstention_agreement_rate": 1.0,
                "distinct_runtime_lengths_minimum": 2,
                "by_source_profile": paired_by_profile,
            },
            "gate_cells": {
                "true_class": [self_contained_cell],
                "source_profile": [
                    {**self_contained_cell, "name": profiles[0]}
                ],
            },
        }
        length_report = {
            family: {
                "auroc_vs_route_known": 1.0,
                "unknown_recall": 1.0,
            }
            for family in openset.NOVELTY_FAMILIES
        }
        novelty = {
            "20263002": {
                source: {
                    str(length): length_report
                    for length in openset.RUNTIME_INPUT_LENGTHS
                }
                for source in openset.PROTOTYPE_SOURCE_ROUTES
            }
        }
        return known, novelty

    @staticmethod
    def _valid_design_causal_gate_summary_inputs() -> tuple[dict, dict]:
        known, novelty = (
            EvaluationGateTests._valid_held_gate_summary_inputs()
        )
        expected_cells = {
            f"identity-cell-{index}": True
            for index in range(
                openset
                .EXPECTED_DESIGN_CURRENT_SELECTION_OBSERVATION_CELLS
            )
        }
        known["historical"] = {"diagnostic_only": True}
        known["current"] = {
            "causal_prefix_length_agreement": {
                "expected_base_identity_row_length_cells":
                    expected_cells,
                "observed_base_identity_row_length_cells":
                    deepcopy(expected_cells),
                "exact_expected_observed_cell_coverage": True,
                "stored_observations":
                    openset
                    .EXPECTED_DESIGN_CURRENT_SELECTION_STORED_OBSERVATIONS,
                "stored_observations_by_distinct_observation_length_count": {
                    "2":
                        openset
                        .EXPECTED_DESIGN_CURRENT_SELECTION_TWO_LENGTH_OBSERVATIONS,
                    "3":
                        openset
                        .EXPECTED_DESIGN_CURRENT_SELECTION_THREE_LENGTH_OBSERVATIONS,
                },
                "prediction_or_abstention_agreement_rate": 1.0,
                "by_source_profile": {
                    f"current:{profile}": {
                        "expected_observation_length_cells": 1,
                        "observed_observation_length_cells": 1,
                        "exact_length_coverage": True,
                        "stored_observations": 1,
                        "prediction_or_abstention_agreement_rate": 1.0,
                    }
                    for profile in openset.corpus_data.CURRENT_PROFILES
                },
            }
        }
        return known, novelty

    def test_gate_summary_rejects_adversarial_physical_scale_reports(
        self,
    ) -> None:
        known, novelty = self._valid_held_gate_summary_inputs()

        def summarize(candidate: dict) -> dict:
            with mock.patch.object(
                openset,
                "_classifier_gate_summary",
                return_value=(
                    "independent_held_current_service",
                    {
                        "classifier_smoke": {
                            "value": 1.0,
                            "passes": True,
                        }
                    },
                ),
            ):
                return openset._gate_summary(candidate, novelty)

        baseline = summarize(known)
        self.assertTrue(
            baseline["gates"][
                "held_physical_scale_outcome_agreement_pooled"
            ]["passes"]
        )
        self.assertTrue(
            baseline["gates"][
                "held_physical_scale_outcome_agreement_worst_"
                "observation_length"
            ]["passes"]
        )

        pooled_failure = deepcopy(known)
        pooled_failure["paired_physical_scale_invariance"][
            "prediction_or_abstention_agreement_rate"
        ] = 0.949
        pooled_result = summarize(pooled_failure)
        self.assertFalse(
            pooled_result["gates"][
                "held_physical_scale_outcome_agreement_pooled"
            ]["passes"]
        )

        length_failure = deepcopy(known)
        length_failure["paired_physical_scale_invariance"][
            "prediction_or_abstention_agreement_rate"
        ] = 0.96
        length_failure["paired_physical_scale_invariance"][
            "by_observation_length"
        ]["8192"]["prediction_or_abstention_agreement_rate"] = 0.949
        length_result = summarize(length_failure)
        self.assertTrue(
            length_result["gates"][
                "held_physical_scale_outcome_agreement_pooled"
            ]["passes"]
        )
        self.assertFalse(
            length_result["gates"][
                "held_physical_scale_outcome_agreement_worst_"
                "observation_length"
            ]["passes"]
        )

        for label, tampered_scales in (
            ("missing", [2.0]),
            ("illegal", [1.0, 1.5, 2.0]),
        ):
            with self.subTest(label=label):
                tampered = deepcopy(known)
                physical = tampered[
                    "paired_physical_scale_invariance"
                ]
                cell = next(
                    row
                    for row in physical[
                        "pair_observation_length_cells"
                    ]
                    if (
                        row["source_profile"]
                        == "current:bluetooth-le-advertising"
                        and row["observation_length"] == 16_384
                    )
                )
                cell_key = (
                    f"{cell['pair_id']}|N={cell['observation_length']}"
                )
                cell["expected_legal_scale_factors"] = list(
                    tampered_scales
                )
                cell["observed_legal_scale_factors"] = list(
                    tampered_scales
                )
                cell["distinct_legal_scale_count"] = len(
                    tampered_scales
                )
                physical[
                    "expected_legal_scales_by_pair_observation_length"
                ][cell_key] = list(tampered_scales)
                physical[
                    "observed_legal_scales_by_pair_observation_length"
                ][cell_key] = list(tampered_scales)
                result = summarize(tampered)
                gate = result["gates"][
                    "held_physical_scale_outcome_agreement_pooled"
                ]
                self.assertFalse(gate["exact_coverage"])
                self.assertFalse(gate["passes"])

    def test_gate_summary_rejects_adversarial_causal_reports(
        self,
    ) -> None:
        known, novelty = self._valid_held_gate_summary_inputs()

        def summarize(candidate: dict) -> dict:
            with mock.patch.object(
                openset,
                "_classifier_gate_summary",
                return_value=(
                    "independent_held_current_service",
                    {
                        "classifier_smoke": {
                            "value": 1.0,
                            "passes": True,
                        }
                    },
                ),
            ):
                return openset._gate_summary(candidate, novelty)

        pooled_failure = deepcopy(known)
        pooled_failure["causal_prefix_length_invariance"][
            "prediction_or_abstention_agreement_rate"
        ] = 0.999
        self.assertFalse(
            summarize(pooled_failure)["gates"][
                "held_causal_prefix_outcome_agreement_pooled"
            ]["passes"]
        )

        profile_failure = deepcopy(known)
        profile = next(
            iter(
                profile_failure["causal_prefix_length_invariance"][
                    "by_source_profile"
                ]
            )
        )
        profile_failure["causal_prefix_length_invariance"][
            "by_source_profile"
        ][profile]["prediction_or_abstention_agreement_rate"] = 0.999
        profile_result = summarize(profile_failure)
        self.assertTrue(
            profile_result["gates"][
                "held_causal_prefix_outcome_agreement_pooled"
            ]["passes"]
        )
        self.assertFalse(
            profile_result["gates"][
                "held_causal_prefix_outcome_agreement_worst_source_profile"
            ]["passes"]
        )

        for label in ("omission", "substitution", "count"):
            with self.subTest(label=label):
                tampered = deepcopy(known)
                rows = tampered[
                    "causal_prefix_length_invariance"
                ]["by_source_profile"]
                profile, row = next(iter(rows.items()))
                if label == "omission":
                    rows.pop(profile)
                elif label == "substitution":
                    rows["current:not-in-frozen-inventory"] = rows.pop(
                        profile
                    )
                else:
                    row["expected_pair_scale_cells"] = 95
                    row["observed_pair_scale_cells"] = 95
                result = summarize(tampered)
                self.assertFalse(
                    result["gates"][
                        "held_causal_prefix_outcome_agreement_pooled"
                    ]["passes"]
                )
                self.assertFalse(result["all_pass"])

    def test_gate_summary_requires_exact_held_paired_profile_inventory(
        self,
    ) -> None:
        known, novelty = self._valid_held_gate_summary_inputs()

        def summarize(candidate: dict) -> dict:
            with mock.patch.object(
                openset,
                "_classifier_gate_summary",
                return_value=(
                    "independent_held_current_service",
                    {
                        "classifier_smoke": {
                            "value": 1.0,
                            "passes": True,
                        }
                    },
                ),
            ):
                return openset._gate_summary(candidate, novelty)

        for label in ("omission", "substitution", "count"):
            with self.subTest(label=label):
                tampered = deepcopy(known)
                rows = tampered[
                    "paired_length_scale_invariance"
                ]["by_source_profile"]
                profile, row = next(iter(rows.items()))
                if label == "omission":
                    rows.pop(profile)
                elif label == "substitution":
                    rows["current:not-in-frozen-inventory"] = rows.pop(
                        profile
                    )
                else:
                    row["expected_pair_cells"] = 23
                    row["cells"] = 23
                gate = summarize(tampered)["gates"][
                    "known_paired_length_scale_outcome_agreement_"
                    "worst_source_profile"
                ]
                self.assertFalse(gate["passes"])

    def test_gate_summary_requires_exact_design_causal_profile_inventory(
        self,
    ) -> None:
        known, novelty = self._valid_design_causal_gate_summary_inputs()

        def summarize(candidate: dict) -> dict:
            with mock.patch.object(
                openset,
                "_classifier_gate_summary",
                return_value=(
                    "design_current_routed_selection",
                    {
                        "classifier_smoke": {
                            "value": 1.0,
                            "passes": True,
                        }
                    },
                ),
            ):
                return openset._gate_summary(candidate, novelty)

        pooled_failure = deepcopy(known)
        pooled_failure["current"]["causal_prefix_length_agreement"][
            "prediction_or_abstention_agreement_rate"
        ] = 0.999
        self.assertFalse(
            summarize(pooled_failure)["gates"][
                "current_selection_causal_prefix_outcome_agreement_pooled"
            ]["passes"]
        )

        profile_failure = deepcopy(known)
        profile = next(
            iter(
                profile_failure["current"][
                    "causal_prefix_length_agreement"
                ]["by_source_profile"]
            )
        )
        profile_failure["current"]["causal_prefix_length_agreement"][
            "by_source_profile"
        ][profile]["prediction_or_abstention_agreement_rate"] = 0.999
        profile_result = summarize(profile_failure)
        self.assertTrue(
            profile_result["gates"][
                "current_selection_causal_prefix_outcome_agreement_pooled"
            ]["passes"]
        )
        self.assertFalse(
            profile_result["gates"][
                "current_selection_causal_prefix_outcome_agreement_"
                "worst_source_profile"
            ]["passes"]
        )

        for label in ("omission", "substitution"):
            with self.subTest(label=label):
                tampered = deepcopy(known)
                rows = tampered["current"][
                    "causal_prefix_length_agreement"
                ]["by_source_profile"]
                profile, row = next(iter(rows.items()))
                if label == "omission":
                    rows.pop(profile)
                else:
                    rows["current:not-in-frozen-inventory"] = rows.pop(
                        profile
                    )
                with (
                    mock.patch.object(
                        openset,
                        "_classifier_gate_summary",
                        return_value=(
                            "design_current_routed_selection",
                            {
                                "classifier_smoke": {
                                    "value": 1.0,
                                    "passes": True,
                                }
                            },
                        ),
                    ),
                    self.assertRaisesRegex(
                        ValueError,
                        "causal-prefix coverage is incomplete",
                    ),
                ):
                    openset._gate_summary(tampered, novelty)

    def test_abstention_cell_gate_excludes_underpowered_cells(self) -> None:
        base = {
            "source": "historical",
            "length": 4_096,
            "name": "tiny",
            "rows": 5,
            "closed_head_accuracy": 1.0,
            "closed_head_correct_rows": 5,
            "rejected_closed_head_correct_rows": 2,
            "abstention_damage_given_closed_head_correct": 0.4,
        }
        gate = openset._abstention_cell_gate(
            [base],
            label_field="name",
            label_output_field="source_profile",
        )
        self.assertFalse(gate["applicable"])
        self.assertIsNone(gate["passes"])
        supported = {
            **base,
            "name": "supported",
            "rows": 20,
            "closed_head_correct_rows": 20,
            "rejected_closed_head_correct_rows": 3,
            "abstention_damage_given_closed_head_correct": 0.15,
        }
        gate = openset._abstention_cell_gate(
            [base, supported],
            label_field="name",
            label_output_field="source_profile",
        )
        self.assertTrue(gate["applicable"])
        self.assertFalse(gate["passes"])
        self.assertEqual(gate["source_profile"], "supported")

    def test_design_classifier_gates_use_routed_closed_predictions_only(
        self,
    ) -> None:
        per_length = {}
        for length in openset.RUNTIME_INPUT_LENGTHS:
            per_profile = {}
            per_class_counts = {
                name: [0, 0]
                for name in openset.corpus_data.CURRENT_CLASSES
            }
            for profile in openset.corpus_data.CURRENT_PROFILES:
                public_class = (
                    openset.corpus_data.CURRENT_PROFILE_PUBLIC_CLASS_MAP[
                        profile
                    ]
                )
                per_profile[f"current:{profile}"] = {
                    "rows": 1,
                    "closed_head_correct_rows": 1,
                    "closed_head_accuracy": 1.0,
                }
                per_class_counts[public_class][0] += 1
                per_class_counts[public_class][1] += 1
            per_length[str(length)] = {
                "rows": len(per_profile),
                "closed_head_correct_rows": len(per_profile),
                "closed_head_accuracy": 1.0,
                # Deliberately unrelated union-bank diagnostic: gates must
                # use the trusted current-route prediction summary above.
                "union_bank_closed_head_accuracy": 0.1,
                "per_true_public_class": {
                    name: {
                        "rows": counts[0],
                        "closed_head_correct_rows": counts[1],
                        "closed_head_accuracy": 1.0,
                    }
                    for name, counts in per_class_counts.items()
                },
                "per_source_profile": per_profile,
            }
        known = {
            "historical": {"diagnostic_only": True},
            "current": {
                "per_length": per_length,
                "pooled": {
                    "rows": (
                        len(openset.corpus_data.CURRENT_PROFILES)
                        * len(openset.RUNTIME_INPUT_LENGTHS)
                    ),
                    "closed_head_correct_rows": (
                        len(openset.corpus_data.CURRENT_PROFILES)
                        * len(openset.RUNTIME_INPUT_LENGTHS)
                    ),
                    "closed_head_accuracy": 1.0,
                    "false_unknown_rate": 0.75,
                },
            },
        }
        role, gates = openset._classifier_gate_summary(known)
        self.assertEqual(role, "design_current_routed_selection")
        self.assertTrue(all(row["passes"] for row in gates.values()))
        self.assertEqual(
            gates["classifier_design_current_pooled_closed_accuracy"][
                "value"
            ],
            1.0,
        )
        # Rejection rate is intentionally irrelevant to every closed-head
        # classifier gate.
        known["current"]["pooled"]["false_unknown_rate"] = 1.0
        _role, repeated = openset._classifier_gate_summary(known)
        self.assertEqual(gates, repeated)

    def test_held_classifier_gates_use_exact_non_cartesian_legal_map(
        self,
    ) -> None:
        known = self._perfect_held_classifier_report()
        role, gates = openset._classifier_gate_summary(known)
        self.assertEqual(role, "independent_held_current_service")
        self.assertTrue(all(row["passes"] for row in gates.values()))
        coverage = gates["classifier_held_current_service_coverage"]
        self.assertEqual(coverage["expected_profile_legal_cells"], 370)
        self.assertEqual(
            coverage["expected_profile_legal_observations"], 8_880
        )
        self.assertEqual(coverage["expected_wifi_legal_cells"], 12)
        expected = known["paired_length_scale_invariance"][
            "expected_profile_legal_cell_rows"
        ]
        ble_low = openset._profile_legal_cell_key(
            "current:bluetooth-le-advertising", 16_384, 1.0
        )
        ble_high = openset._profile_legal_cell_key(
            "current:bluetooth-le-advertising", 16_384, 1.5
        )
        self.assertNotIn(ble_low, expected)
        self.assertEqual(expected[ble_high], 24)

        missing = self._perfect_held_classifier_report(
            observed_ble_16k_scales=(2.0,)
        )
        _role, missing_gates = openset._classifier_gate_summary(missing)
        self.assertFalse(
            missing_gates[
                "classifier_held_current_service_coverage"
            ]["passes"]
        )
        self.assertFalse(
            missing_gates[
                "classifier_held_current_service_coverage"
            ]["exact_profile_legal_cell_key_and_count_coverage"]
        )
        self.assertFalse(
            all(row["passes"] for row in missing_gates.values())
        )

        illegal = self._perfect_held_classifier_report(
            observed_ble_16k_scales=(1.0, 1.5, 2.0)
        )
        _role, illegal_gates = openset._classifier_gate_summary(illegal)
        self.assertFalse(
            illegal_gates[
                "classifier_held_current_service_coverage"
            ]["passes"]
        )
        self.assertFalse(
            illegal_gates[
                "classifier_held_current_service_coverage"
            ]["exact_profile_legal_cell_key_and_count_coverage"]
        )
        self.assertFalse(
            all(row["passes"] for row in illegal_gates.values())
        )

    def test_held_scale_evaluator_requires_complete_per_profile_pairs(
        self,
    ) -> None:
        manifest_sha = "1" * 64
        raw_sha = "2" * 64
        protocol = openset.build_validation_protocol(
            classifier_asset_sha256="d" * 64,
            held_scale_manifest_sha256=manifest_sha,
            held_scale_raw_sha256=raw_sha,
        )
        rows = [
            SimpleNamespace(
                pair_id="pair-1",
                profile="wifi-ofdm-20m",
                class_name="ofdm",
                scale_factor=float(scale),
                available_lengths=(4_096, 8_192),
            )
            for scale in openset.scale_eval.SCALE_FACTORS
        ]
        corpus = SimpleNamespace(
            rows=rows,
            audit={
                "manifest_sha256": manifest_sha,
                "raw_sha256": raw_sha,
                "realizations_per_profile":
                    openset.DEFAULT_HELD_SCALE_REALIZATIONS_PER_PROFILE,
            },
            manifest={"evalSeed": openset.DEFAULT_HELD_SCALE_EVAL_SEED},
            prefix=lambda _row, length: np.ones(
                length, dtype=np.complex64
            ),
        )
        context = SimpleNamespace(
            fusion={"metrics": {"data_audit": {"current": {}}}}
        )

        def accepted(
            _context,
            _policy,
            raw_rows,
            *,
            length,
            prototype_source,
        ):
            self.assertIn(length, (4_096, 8_192))
            self.assertEqual(prototype_source, "current")
            scores = openset._empty_population_scores(len(raw_rows))
            scores.prediction[:] = 6
            scores.final_score[:] = 0.2
            return scores

        with (
            mock.patch.object(
                openset.scale_eval,
                "load_scale_eval_corpus",
                return_value=corpus,
            ),
            mock.patch.object(
                openset.scale_eval,
                "validate_reference_separation",
                return_value={"separate": True},
            ),
            mock.patch.object(
                openset,
                "score_raw_population",
                side_effect=accepted,
            ),
        ):
            report, scores = openset.evaluate_held_scale_known(
                context,
                {
                    "runtime_length_policy_by_prototype_source":
                        openset.RUNTIME_LENGTH_POLICY_BY_PROTOTYPE_SOURCE
                },
                "held",
                "current",
                validation_protocol=protocol,
            )
        profile = report["paired_length_scale_invariance"][
            "by_source_profile"
        ]["current:wifi-ofdm-20m"]
        self.assertEqual(profile["complete_pair_coverage_rate"], 1.0)
        self.assertEqual(profile["minimum_distinct_runtime_lengths"], 2)
        self.assertEqual(
            profile["prediction_or_abstention_agreement_rate"], 1.0
        )
        physical = report["paired_physical_scale_invariance"]
        self.assertTrue(physical["exact_eligible_cell_coverage"])
        self.assertEqual(physical["expected_cells"], 2)
        self.assertEqual(
            set(physical["by_observation_length"]), {"4096", "8192", "16384"}
        )
        self.assertEqual(
            physical["by_observation_length"]["4096"][
                "prediction_or_abstention_agreement_rate"
            ],
            1.0,
        )
        causal = report["causal_prefix_length_invariance"]
        self.assertTrue(causal["exact_eligible_cell_coverage"])
        self.assertEqual(causal["expected_cells"], 4)
        self.assertEqual(
            causal["prediction_or_abstention_agreement_rate"], 1.0
        )
        self.assertEqual(set(scores), {4_096, 8_192})

    def test_known_selection_evaluation_call_reports_closed_correctness(
        self,
    ) -> None:
        def corpus(source: str) -> SimpleNamespace:
            row = SimpleNamespace(
                source=source,
                profile_id=f"{source}-wifi-ofdm",
                identity=f"{source}-base",
                source_index=0,
                valid_sample_count=16_384,
            )
            return SimpleNamespace(
                source=source,
                rows_by_role={"selection": [row]},
                prefix=lambda _row, length: np.ones(
                    length, dtype=np.complex64
                ),
            )

        prepared = {
            "by_length": {
                length: {
                    "x": np.zeros((1, 1), dtype=np.float32),
                    "f": np.zeros((1, 1), dtype=np.float32),
                    "y": np.asarray([6], dtype=np.int64),
                }
                for length in openset.RUNTIME_INPUT_LENGTHS
            }
        }
        context = SimpleNamespace(
            historical=corpus("historical"),
            current=corpus("current"),
            data={
                "historical_selection": prepared,
                "current_selection": prepared,
            },
        )

        def accepted(*_args, **_kwargs):
            scores = openset._empty_population_scores(1)
            scores.prediction[:] = 6
            scores.final_score[:] = 0.2
            return scores

        with (
            mock.patch.object(
                openset, "score_prepared_population", side_effect=accepted
            ),
            mock.patch.object(
                openset,
                "score_observed_raw_population",
                side_effect=accepted,
            ),
        ):
            report, _scores = openset.evaluate_known_selection(
                context,
                {
                    "runtime_length_policy_by_prototype_source":
                        openset.RUNTIME_LENGTH_POLICY_BY_PROTOTYPE_SOURCE
                },
            )
        self.assertEqual(
            report["combined"]["pooled_correct_and_accepted_rate"], 1.0
        )
        self.assertTrue(
            all(
                row["correct_and_accepted_rate"] == 1.0
                for row in report["gate_cells"]["true_class"]
            )
        )
        self.assertTrue(
            all(
                row["correct_and_accepted_rate"] == 1.0
                for row in report["gate_cells"]["source_profile"]
            )
        )
        causal = report["current"]["causal_prefix_length_agreement"]
        self.assertTrue(
            causal["exact_expected_observed_cell_coverage"]
        )
        self.assertEqual(
            causal["prediction_or_abstention_agreement_rate"], 1.0
        )

    def test_held_length_scale_decision_agreement_is_a_gate(self) -> None:
        known = {
            "combined": {
                "per_length_false_unknown_rate": {
                    str(length): 0.01
                    for length in openset.RUNTIME_INPUT_LENGTHS
                },
                "per_length_abstention_damage_given_closed_head_correct": {
                    str(length): 0.01
                    for length in openset.RUNTIME_INPUT_LENGTHS
                },
                "per_length_closed_head_correct_rows": {
                    str(length): 100
                    for length in openset.RUNTIME_INPUT_LENGTHS
                },
                "pooled_abstention_damage_given_closed_head_correct":
                    0.01,
                "pooled_closed_head_correct_rows": 300,
                "pooled_closed_head_accuracy": 0.99,
                "per_length_closed_head_accuracy": {
                    str(length): 0.99
                    for length in openset.RUNTIME_INPUT_LENGTHS
                },
                "pooled_false_unknown_rate": 0.01,
                "per_length_correct_and_accepted_rate": {
                    str(length): 0.99
                    for length in openset.RUNTIME_INPUT_LENGTHS
                },
                "pooled_correct_and_accepted_rate": 0.99,
            },
            "paired_length_scale_invariance": {
                "cells": 20,
                "expected_pair_cells": 20,
                "complete_pair_coverage_rate": 1.0,
                "prediction_or_abstention_agreement_rate": 0.96,
                "distinct_runtime_lengths_minimum": 2,
                "by_source_profile": {
                    "current:wifi-ofdm-20m": {
                        "cells": 20,
                        "expected_pair_cells": 20,
                        "complete_pair_coverage_rate": 1.0,
                        "minimum_distinct_runtime_lengths": 2,
                        "prediction_or_abstention_agreement_rate": 0.94,
                        "failing_pair_ids": ["pair-1"],
                    }
                },
            },
            "gate_cells": {
                "true_class": [
                    {
                        "value": 0.01,
                        "source": "held",
                        "length": 4_096,
                        "scale_factor": 2.0,
                        "name": "ofdm",
                        "rows": 100,
                        "closed_head_accuracy": 1.0,
                        "closed_head_correct_rows": 100,
                        "rejected_closed_head_correct_rows": 1,
                        "abstention_damage_given_closed_head_correct": 0.01,
                        "correct_and_accepted_rate": 0.99,
                    }
                ],
                "source_profile": [
                    {
                        "value": 0.01,
                        "source": "held",
                        "length": 4_096,
                        "scale_factor": 2.0,
                        "name": "current:wifi-ofdm-20m",
                        "rows": 100,
                        "closed_head_accuracy": 1.0,
                        "closed_head_correct_rows": 100,
                        "rejected_closed_head_correct_rows": 1,
                        "abstention_damage_given_closed_head_correct": 0.01,
                        "correct_and_accepted_rate": 0.99,
                    }
                ],
            },
        }
        length_report = {
            family: {
                "auroc_vs_route_known": 0.90,
                "unknown_recall": 0.50,
            }
            for family in openset.NOVELTY_FAMILIES
        }
        novelty = {
            "20263002": {
                source: {
                    str(length): length_report
                    for length in openset.RUNTIME_INPUT_LENGTHS
                }
                for source in openset.PROTOTYPE_SOURCE_ROUTES
            }
        }
        with mock.patch.object(
            openset,
            "_classifier_gate_summary",
            return_value=(
                "unit_test",
                {"classifier_smoke": {"value": 1.0, "passes": True}},
            ),
        ):
            result = openset._gate_summary(known, novelty)
        self.assertEqual(
            result["closed_head_classifier_summary"]["role"],
            "combined_historical_current_diagnostic",
        )
        self.assertFalse(
            result["closed_head_classifier_summary"][
                "is_classifier_gate_result"
            ]
        )
        agreement = result["gates"][
            "known_paired_length_scale_outcome_agreement"
        ]
        self.assertTrue(agreement["passes"])
        self.assertEqual(agreement["floor"], 0.95)
        self.assertFalse(
            result["gates"][
                "known_paired_length_scale_outcome_agreement_"
                "worst_source_profile"
            ]["passes"]
        )
        self.assertEqual(
            result["gates"][
                "known_abstention_damage_given_closed_head_correct_"
                "worst_true_class_cell"
            ]["physical_scale_factor"],
            2.0,
        )
        known["gate_cells"]["source_profile"][0].update(
            {
                "rejected_closed_head_correct_rows": 20,
                "abstention_damage_given_closed_head_correct": 0.2,
            }
        )
        with mock.patch.object(
            openset,
            "_classifier_gate_summary",
            return_value=(
                "unit_test",
                {"classifier_smoke": {"value": 1.0, "passes": True}},
            ),
        ):
            damaged = openset._gate_summary(known, novelty)
        self.assertFalse(
            damaged["gates"][
                "known_abstention_damage_given_closed_head_correct_"
                "worst_source_profile_cell"
            ]["passes"]
        )
        self.assertFalse(damaged["all_pass"])


class RunContractTests(unittest.TestCase):
    def test_validation_ledger_is_atomic_and_single_use(self) -> None:
        protocol = openset.build_validation_protocol(
            classifier_asset_sha256="d" * 64,
            held_scale_manifest_sha256="1" * 64,
            held_scale_raw_sha256="2" * 64,
        )
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "validation-ledger.json"
            with mock.patch.object(
                openset,
                "_resolve_precommitted_repo_path",
                return_value=ledger,
            ):
                path, digest = openset.claim_validation_ledger(
                    policy_sha256="a" * 64,
                    protocol=protocol,
                )
                self.assertEqual(path, ledger)
                self.assertEqual(openset._sha256(path), digest)
                with self.assertRaisesRegex(
                    RuntimeError, "already been claimed"
                ):
                    openset.claim_validation_ledger(
                        policy_sha256="a" * 64,
                        protocol=protocol,
                    )

    def test_source_snapshot_detects_mid_run_mutation(self) -> None:
        snapshot = openset._source_hashes()
        changed = dict(snapshot)
        first = next(iter(changed))
        changed[first] = "0" * 64
        with mock.patch.object(
            openset, "_source_hashes", return_value=changed
        ):
            with self.assertRaisesRegex(RuntimeError, "changed during"):
                openset._assert_source_snapshot_unchanged(
                    snapshot, operation="unit-test publication"
                )

    def test_run_design_call_contract_smoke(self) -> None:
        rows = 140
        stage_one = {
            length: np.linspace(-3.0, 2.0, rows)
            for length in openset.RUNTIME_INPUT_LENGTHS
        }
        enrollment_distance: dict[int, dict[str, np.ndarray]] = {}
        for length in openset.RUNTIME_INPUT_LENGTHS:
            distances = np.full(
                (rows, len(openset.PUBLIC_CLASSES)), 1.5
            )
            for index in range(rows):
                source = (
                    "historical" if index < rows // 2 else "current"
                )
                support = np.asarray(
                    openset
                    .EXPECTED_PUBLIC_CLASS_SUPPORT_BY_PROTOTYPE_SOURCE[
                        source
                    ],
                    dtype=bool,
                )
                supported = np.flatnonzero(support)
                distances[index, ~support] = np.inf
                distances[index, supported[index % len(supported)]] = (
                    0.02 + 0.5 * index / rows
                )
            enrollment_distance[length] = {
                "class_distances": distances,
                "leave_one_base_identity_out_class_distances":
                    distances + 0.01,
                "groups": np.asarray(
                    [
                        f"source:profile-{index % 14}"
                        for index in range(rows)
                    ],
                    dtype=object,
                ),
                "prototype_sources": np.asarray(
                    [
                        (
                            "historical"
                            if index < rows // 2
                            else "current"
                        )
                        for index in range(rows)
                    ],
                    dtype=object,
                ),
            }
            enrollment_distance[length]["groups"] = np.asarray(
                [
                    (
                        f"{'historical' if index < rows // 2 else 'current'}:"
                        f"profile-{index % 14}"
                    )
                    for index in range(rows)
                ],
                dtype=object,
            )
        context = SimpleNamespace(
            historical=SimpleNamespace(audit={"source": "historical"}),
            current=SimpleNamespace(audit={"source": "current"}),
            fusion={},
            patch_length=64,
            patch_count=16,
            target_frac=0.5,
        )
        known_scores = {
            source: {
                length: np.linspace(0.0, 0.5, 10)
                for length in openset.RUNTIME_INPUT_LENGTHS
            }
            for source in openset.PROTOTYPE_SOURCE_ROUTES
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "design"
            args = SimpleNamespace(
                output_dir=output,
                novelty_seeds=[openset.DEFAULT_DESIGN_SEED],
                design_seed=openset.DEFAULT_DESIGN_SEED,
                fusion="fusion",
                historical_corpus="historical",
                current_corpus="current",
                device="cpu",
                stage_one_fitting_seed=openset.DEFAULT_STAGE_ONE_FIT_SEED,
                stage_one_fitting_noise_n=10,
                classifier_asset="classifier.json",
                composite_budget=openset.DEFAULT_COMPOSITE_KNOWN_BUDGET,
                novelty_n=3,
                held_scale_manifest_sha256="1" * 64,
                held_scale_raw_sha256="2" * 64,
            )
            with (
                mock.patch.object(
                    openset,
                    "load_development_context",
                    return_value=context,
                ),
                mock.patch.object(
                    openset,
                    "fit_v4_stage_one_template",
                    return_value=_template(),
                ),
                mock.patch.object(
                    openset,
                    "_combined_enrollment_raw",
                    return_value={
                        length: [object()] * rows
                        for length in openset.RUNTIME_INPUT_LENGTHS
                    },
                ),
                mock.patch.object(
                    openset,
                    "_enrollment_distance_inputs",
                    return_value=enrollment_distance,
                ),
                mock.patch.object(
                    openset,
                    "stage_one_scores",
                    side_effect=lambda *_args, **kwargs:
                        stage_one[int(kwargs["length"])],
                ),
                mock.patch.object(
                    openset,
                    "instantaneous_phase_linearity_scores",
                    return_value=np.linspace(0.0, 0.8, rows),
                ),
                mock.patch.object(
                    openset,
                    "build_classifier_binding",
                    return_value=_binding(),
                ),
                mock.patch.object(
                    openset,
                    "evaluate_known_selection",
                    return_value=({"smoke": True}, known_scores),
                ),
                mock.patch.object(
                    openset,
                    "evaluate_novelty",
                    return_value=({}, {}),
                ),
                mock.patch.object(
                    openset,
                    "_build_report",
                    return_value={"gate_summary": {"all_pass": True}},
                ),
            ):
                report = openset.run_design(args)
            self.assertTrue(report["gate_summary"]["all_pass"])
            self.assertTrue((output / openset.POLICY_NAME).is_file())
            self.assertTrue((output / openset.MANIFEST_NAME).is_file())

    def test_run_validate_call_contract_smoke(self) -> None:
        policy = _fit_policy()
        protocol = policy["validation_protocol"]
        known_scores = {
            source: {
                length: np.linspace(0.0, 0.5, 10)
                for length in openset.RUNTIME_INPUT_LENGTHS
            }
            for source in openset.PROTOTYPE_SOURCE_ROUTES
        }
        current_scores = known_scores["current"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy_path = root / openset.POLICY_NAME
            openset._write_json(policy_path, policy)
            output = root / "validation"
            ledger = root / "ledger.json"
            ledger.touch()
            args = SimpleNamespace(
                output_dir=output,
                policy=policy_path,
                novelty_seeds=list(openset.DEFAULT_VALIDATION_SEEDS),
                fusion="fusion",
                historical_corpus="historical",
                current_corpus="current",
                held_scale_corpus="held",
                device="cpu",
                classifier_asset="classifier.json",
                novelty_n=openset.DEFAULT_NOVELTY_ROWS,
            )
            context = SimpleNamespace(fusion={})
            with (
                mock.patch.object(
                    openset,
                    "load_development_context",
                    return_value=context,
                ),
                mock.patch.object(
                    openset,
                    "build_classifier_binding",
                    return_value=_binding(),
                ),
                mock.patch.object(
                    openset,
                    "validate_policy",
                    return_value=policy,
                ),
                mock.patch.object(
                    openset,
                    "_validate_policy_corpus_binding",
                ),
                mock.patch.object(
                    openset,
                    "validate_validation_protocol",
                    return_value=protocol,
                ),
                mock.patch.object(
                    openset,
                    "claim_validation_ledger",
                    return_value=(ledger, "9" * 64),
                ),
                mock.patch.object(
                    openset,
                    "evaluate_known_selection",
                    return_value=({"auxiliary": True}, known_scores),
                ),
                mock.patch.object(
                    openset,
                    "evaluate_held_scale_known",
                    return_value=({"held": True}, current_scores),
                ) as held_mock,
                mock.patch.object(
                    openset,
                    "evaluate_novelty",
                    return_value=({}, {}),
                ),
                mock.patch.object(
                    openset,
                    "_build_report",
                    return_value={"gate_summary": {"all_pass": True}},
                ),
            ):
                report = openset.run_validate(args)
            self.assertTrue(report["gate_summary"]["all_pass"])
            self.assertTrue(
                (output / openset.VALIDATION_REPORT_NAME).is_file()
            )
            manifest = openset._load_json(output / openset.MANIFEST_NAME)
            self.assertEqual(
                manifest["validation_ledger_sha256"], "9" * 64
            )
            self.assertEqual(
                held_mock.call_args.kwargs["validation_protocol"],
                protocol,
            )


if __name__ == "__main__":
    unittest.main()
