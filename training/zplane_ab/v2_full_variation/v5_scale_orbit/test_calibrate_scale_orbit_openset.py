from __future__ import annotations

import ast
from copy import deepcopy
import inspect
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import calibrate_scale_orbit_openset as openset  # noqa: E402


def _stage_one_template() -> openset.v4_openset.StageOneTemplate:
    width = len(openset.noise_prefilter.PREFILTER_FEATURES)
    models = {}
    bundle_hashes = {}
    for length in openset.RUNTIME_INPUT_LENGTHS:
        model = openset.noise_prefilter.NoisePrefilter(
            feature_names=tuple(
                openset.noise_prefilter.PREFILTER_FEATURES
            ),
            capture_length=length,
            mean=np.zeros(width, dtype=np.float64),
            scale=np.ones(width, dtype=np.float64),
            coefficients=np.linspace(-0.2, 0.2, width),
            intercept=-0.1,
            threshold_score=0.0,
            fit_provenance={
                "selection_rows_used_in_fit": 0,
                "novelty_design_or_validation_rows_used_in_fit": 0,
                "consumed_test_rows_used_in_fit": 0,
            },
            threshold_provenance={"chosen": False},
        )
        models[length] = model
        bundle_hashes[length] = openset.v4_openset._stage_one_parameter_sha256(
            mean=model.mean,
            scale=model.scale,
            coefficients=model.coefficients,
            intercept=model.intercept,
        )
    return openset.v4_openset.StageOneTemplate(
        directory=Path("<synthetic-fixed-stage-one>"),
        models=models,
        set_sha256=openset.v4_openset._stage_one_parameter_set_sha256(
            bundle_hashes
        ),
        bundle_sha256=bundle_hashes,
        metadata={
            length: {
                "source_policy_sha256": "f" * 64,
                "fixed_before_v5_adaptive_selection": True,
                "synthetic_only": True,
            }
            for length in openset.RUNTIME_INPUT_LENGTHS
        },
    )


def _calibration() -> openset._EnrollmentCalibration:
    width = len(openset.PUBLIC_CLASSES)
    stage_one = {}
    phase = {}
    distance = {}
    loo = {}
    groups = {}
    sources = {}
    identities = {}
    for offset, length in enumerate(openset.RUNTIME_INPUT_LENGTHS):
        route_values = (
            ["historical"] * 14 + ["current"] * 10
            if length == openset.CURRENT_EFFECTIVE_INPUT_LENGTH
            else ["historical"] * 14
        )
        rows = len(route_values)
        values = np.full((rows, width), 1.5, dtype=np.float64)
        for index, source in enumerate(route_values):
            support = np.asarray(
                openset
                .EXPECTED_PUBLIC_CLASS_SUPPORT_BY_PROTOTYPE_SOURCE[source],
                dtype=bool,
            )
            if source == "current":
                values[index, ~support] = np.inf
            supported = np.flatnonzero(support)
            values[index, supported[index % len(supported)]] = (
                0.01 + index / 100
            )
        stage_one[length] = (
            np.linspace(-1.0, 1.0, rows) + 0.01 * offset
        )
        phase[length] = np.linspace(0.0, 0.8, rows)
        distance[length] = values
        loo[length] = values + np.where(np.isfinite(values), 0.01, 0.0)
        groups[length] = [
            f"{source}:profile-{index % 7}"
            for index, source in enumerate(route_values)
        ]
        sources[length] = route_values
        identities[length] = [
            f"{source}:identity-{index}"
            for index, source in enumerate(route_values)
        ]
    audit = {
        "schema": "v5-scale-orbit-enrollment-calibration-input-v1",
        "role_read": "enrollment",
        **openset.FITTING_FIREWALL,
    }
    return openset._EnrollmentCalibration(
        stage_one_score_by_length=stage_one,
        phase_linearity_score_by_length=phase,
        class_distance_by_length=distance,
        calibration_class_distance_by_length=loo,
        enrollment_group_by_length=groups,
        enrollment_source_by_length=sources,
        enrollment_identity_by_length=identities,
        audit=audit,
    )


def _binding() -> dict:
    return {
        "kind": "v5-scale-orbit-development-fusion",
        "source_fusion_schema": openset.FUSION_SCHEMA,
        "source_fusion_dev_metrics_sha256": "a" * 64,
        "source_fusion_artifacts_sha256": {
            "fusion_state_dict.pt": "b" * 64,
        },
        "current_composite_audit_schema":
            openset.COMPOSITE_AUDIT_SCHEMA,
        "trusted_current_geometry_canonicalizer":
            openset.geometry_canonicalizer.canonicalizer_metadata(),
        "classifier_runtime_role_required": "accepted_known_classifier",
        "selected_profile_is_classifier_input": False,
        "selected_class_is_classifier_input": False,
    }


def _novelty_geometry() -> dict:
    return {
        "schema": openset.CURRENT_NOVELTY_GEOMETRY_SCHEMA,
        "sample_rate_hz": 16_000_000,
        "native_sample_rate_hz": 8_000_000,
        "sample_rate_ratio": 2.0,
        "canonical_domain_identity_geometry": True,
        "selected_profile_or_class_used": False,
        "frozen_before_adaptive_novelty_scoring": True,
    }


def _context() -> openset.DevelopmentContext:
    return openset.DevelopmentContext(
        fusion={
            "metrics": {
                "frontend": openset.td_preprocess.preprocess_metadata(),
            },
        },
        historical=None,
        current=None,
        data={},
        preprocessing_audit={},
        module=None,
        device=None,
        patch_length=64,
        patch_count=16,
        target_frac=0.5,
    )


def _fit_policy(
    calibration: openset._EnrollmentCalibration | None = None,
    binding: dict | None = None,
) -> dict:
    with (
        mock.patch.object(
            openset,
            "_validate_contract_files",
            return_value={},
        ),
        mock.patch.object(
            openset,
            "_build_enrollment_calibration",
            return_value=calibration or _calibration(),
        ),
        mock.patch.object(
            openset,
            "build_fusion_binding",
            return_value=binding or _binding(),
        ),
        mock.patch.object(
            openset,
            "load_fixed_v4_stage_one_template",
            return_value=_stage_one_template(),
        ),
    ):
        return openset.fit_enrollment_policy(
            _context(),
            fixed_stage_one_policy_path=
                "/development/fixed-v4-stage-one-policy.json",
            current_novelty_geometry=_novelty_geometry(),
        )


def _perfect_gate_cells(
    pairs_per_profile: int = 20,
) -> list[openset.ScoredGateCell]:
    output: list[openset.ScoredGateCell] = []
    profiles = tuple(
        (f"current:{profile}", public_class)
        for profile, public_class in sorted(
            openset.corpus_data.CURRENT_PROFILE_PUBLIC_CLASS_MAP.items()
        )
    )
    for profile, truth in profiles:
        for pair_index in range(pairs_per_profile):
            pair = f"{profile}:pair-{pair_index}"
            for length in openset.RUNTIME_INPUT_LENGTHS:
                for scale in openset.scale_orbit_data.SCALE_ORBIT_EXPECTED_FACTORS:
                    output.append(
                        openset.ScoredGateCell(
                            source_profile=profile,
                            truth=truth,
                            predicted=truth,
                            rejected=False,
                            pair_id=pair,
                            observation_length=length,
                            physical_scale_factor=scale,
                        )
                    )
    return output


class CanonicalizationBoundaryTests(unittest.TestCase):
    def test_public_inference_signatures_have_no_profile_or_class_input(self):
        openset.assert_no_selected_profile_or_class_inference_input()
        forbidden = {
            "selected_profile",
            "selected_class",
            "profile_id",
            "class_name",
        }
        for function in (
            openset.canonicalize_inference_observation,
            openset.prepare_inference_observation,
            openset.score_observations,
        ):
            self.assertFalse(
                forbidden
                & set(inspect.signature(function).parameters)
            )
        signature = inspect.signature(
            openset.canonicalize_inference_observation
        )
        self.assertIs(
            signature.parameters["sample_rate_hz"].default,
            inspect.Parameter.empty,
        )
        self.assertIs(
            signature.parameters["native_sample_rate_hz"].default,
            inspect.Parameter.empty,
        )

    def test_current_canonicalization_precedes_every_feature_extractor(self):
        events: list[str] = []
        sentinel = np.full(
            openset.CURRENT_EFFECTIVE_INPUT_LENGTH,
            3 + 4j,
            dtype=np.complex64,
        )

        def canonicalize(iq, *, sample_rate_hz, native_sample_rate_hz):
            events.append("canonicalize")
            self.assertEqual(len(iq), 8192)
            self.assertEqual(sample_rate_hz, 10_000_000)
            self.assertEqual(native_sample_rate_hz, 8_000_000)
            return sentinel.copy(), {
                "trust_scope": "trusted-current",
                "uses_frequency_transform": False,
                "uses_selected_profile_or_class": False,
            }

        def pose(value, *, patch_length, target_frac):
            events.append("pose")
            np.testing.assert_array_equal(value, sentinel)
            self.assertEqual(patch_length, 64)
            self.assertEqual(target_frac, 0.5)
            return np.ones(
                openset.v4_openset.pose_degeneracy.FEATURE_COUNT,
                dtype=np.float64,
            )

        def phase(value):
            events.append("phase")
            np.testing.assert_array_equal(value, sentinel)
            return 0.25

        def frontend(value, *, patch_length, patch_count, target_frac):
            events.append("frontend")
            np.testing.assert_array_equal(value, sentinel)
            return (
                np.ones((2, 3), dtype=np.float32),
                np.ones(4, dtype=np.float32),
                {"uses_frequency_transform": False},
            )

        observation = openset.InferenceObservation(
            iq=np.ones(8192, dtype=np.complex64),
            prototype_source="current",
            sample_rate_hz=10_000_000,
            native_sample_rate_hz=8_000_000,
        )
        with (
            mock.patch.object(
                openset.geometry_canonicalizer,
                "canonicalize",
                side_effect=canonicalize,
            ),
            mock.patch.object(
                openset.v4_openset,
                "_pose_feature_row",
                side_effect=pose,
            ),
            mock.patch.object(
                openset.v4_openset,
                "instantaneous_phase_linearity_chirp_score",
                side_effect=phase,
            ),
            mock.patch.object(
                openset.td_preprocess,
                "preprocess",
                side_effect=frontend,
            ),
        ):
            prepared = openset.prepare_inference_observation(
                observation,
                patch_length=64,
                patch_count=16,
                target_frac=0.5,
            )
        self.assertEqual(
            events,
            ["canonicalize", "pose", "phase", "frontend"],
        )
        np.testing.assert_array_equal(prepared.effective_iq, sentinel)
        self.assertEqual(prepared.effective_input_length, 4096)

    def test_historical_path_is_unchanged_and_rejects_rate_metadata(self):
        raw = (
            np.arange(9000, dtype=np.float32)
            + 1j * np.ones(9000, dtype=np.float32)
        ).astype(np.complex64)
        with mock.patch.object(
            openset.geometry_canonicalizer,
            "canonicalize",
            side_effect=AssertionError("historical path called canonicalizer"),
        ):
            effective, length, audit = (
                openset.canonicalize_inference_observation(
                    raw,
                    prototype_source="historical",
                    sample_rate_hz=None,
                    native_sample_rate_hz=None,
                )
            )
        self.assertEqual(length, 8192)
        self.assertIsNone(audit)
        np.testing.assert_array_equal(effective, raw[:8192])
        with self.assertRaisesRegex(ValueError, "forbids"):
            openset.canonicalize_inference_observation(
                raw,
                prototype_source="historical",
                sample_rate_hz=8_000_000,
                native_sample_rate_hz=8_000_000,
            )

    def test_current_path_requires_both_rates(self):
        raw = np.ones(4096, dtype=np.complex64)
        for sample_rate, native_rate in (
            (None, None),
            (8_000_000, None),
            (None, 8_000_000),
        ):
            with self.subTest(
                sample_rate=sample_rate, native_rate=native_rate
            ):
                with self.assertRaisesRegex(ValueError, "requires explicit"):
                    openset.canonicalize_inference_observation(
                        raw,
                        prototype_source="current",
                        sample_rate_hz=sample_rate,
                        native_sample_rate_hz=native_rate,
                    )

    def test_exact_zero_is_canonicalized_before_feature_early_exit(self):
        events: list[str] = []

        def canonicalize(iq, **kwargs):
            del iq, kwargs
            events.append("canonicalize")
            return np.zeros(4096, dtype=np.complex64), {
                "trust_scope": "trusted-current",
                "uses_frequency_transform": False,
                "uses_selected_profile_or_class": False,
            }

        observation = openset.InferenceObservation(
            iq=np.zeros(4096, dtype=np.complex64),
            prototype_source="current",
            sample_rate_hz=16_000_000,
            native_sample_rate_hz=8_000_000,
        )
        with (
            mock.patch.object(
                openset.geometry_canonicalizer,
                "canonicalize",
                side_effect=canonicalize,
            ),
            mock.patch.object(
                openset.v4_openset,
                "_pose_feature_row",
                side_effect=AssertionError("pose called for zero"),
            ),
            mock.patch.object(
                openset.td_preprocess,
                "preprocess",
                side_effect=AssertionError("frontend called for zero"),
            ),
        ):
            prepared = openset.prepare_inference_observation(
                observation,
                patch_length=64,
                patch_count=16,
                target_frac=0.5,
            )
        self.assertEqual(events, ["canonicalize"])
        self.assertTrue(prepared.is_exact_no_signal)

    def test_module_ast_has_no_frequency_transform_calls(self):
        tree = ast.parse(Path(openset.__file__).read_text(encoding="utf-8"))
        calls: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            pieces: list[str] = []
            while isinstance(function, ast.Attribute):
                pieces.append(function.attr)
                function = function.value
            if isinstance(function, ast.Name):
                pieces.append(function.id)
            name = ".".join(reversed(pieces))
            if "fft" in name.lower():
                calls.append(name)
        self.assertEqual(calls, [])


class RouteConstructionTests(unittest.TestCase):
    def test_novelty_current_geometry_is_fixed_and_class_independent(self):
        rows = [np.ones(4096, dtype=np.complex64)]
        geometry = _novelty_geometry()
        current = openset.novelty_inference_observations(
            rows,
            prototype_source="current",
            sample_rate_hz=geometry["sample_rate_hz"],
            native_sample_rate_hz=geometry["native_sample_rate_hz"],
            current_novelty_geometry=geometry,
        )
        self.assertEqual(len(current), 1)
        self.assertFalse(hasattr(current[0], "profile_id"))
        self.assertFalse(hasattr(current[0], "class_name"))
        with self.assertRaisesRegex(
            ValueError, "frozen class-independent"
        ):
            openset.novelty_inference_observations(
                rows,
                prototype_source="current",
                sample_rate_hz=12_000_000,
                native_sample_rate_hz=8_000_000,
                current_novelty_geometry=geometry,
            )
        historical = openset.novelty_inference_observations(
            rows,
            prototype_source="historical",
            sample_rate_hz=None,
            native_sample_rate_hz=None,
            current_novelty_geometry=None,
        )
        self.assertIsNone(historical[0].sample_rate_hz)

    def test_novelty_evaluator_routes_current_only_with_frozen_rates(self):
        geometry = _novelty_geometry()
        policy = {
            "inference_contract": {
                "current_novelty_geometry": geometry,
            }
        }
        seen: list[openset.InferenceObservation] = []

        def score(_context, _policy, observations):
            seen.extend(observations)
            result = openset.v4_openset._empty_population_scores(
                len(observations)
            )
            result.final_score[:] = 1.0
            return result

        novelty = {
            "noise": {
                4096: [np.ones(4096, dtype=np.complex64)],
                8192: [np.ones(8192, dtype=np.complex64)],
            }
        }
        known = {
            "historical": {
                4096: np.asarray([0.0]),
                8192: np.asarray([0.0]),
            },
            "current": {4096: np.asarray([0.0])},
        }
        with (
            mock.patch.object(openset, "validate_policy"),
            mock.patch.object(
                openset, "score_observations", side_effect=score
            ),
        ):
            report = openset.evaluate_novelty_populations(
                object(),
                policy,
                novelty,
                known,
                current_novelty_geometry=geometry,
            )
        current = [
            row for row in seen if row.prototype_source == "current"
        ]
        historical = [
            row for row in seen if row.prototype_source == "historical"
        ]
        self.assertEqual(len(current), 2)
        self.assertEqual(len(historical), 2)
        self.assertTrue(
            all(
                row.sample_rate_hz == geometry["sample_rate_hz"]
                and row.native_sample_rate_hz
                    == geometry["native_sample_rate_hz"]
                for row in current
            )
        )
        self.assertTrue(
            all(
                row.sample_rate_hz is None
                and row.native_sample_rate_hz is None
                for row in historical
            )
        )
        self.assertEqual(report["rows_used_for_fit_or_selection"], 0)

    def test_held_scale_views_carry_each_rows_explicit_rates(self):
        row = SimpleNamespace(
            available_lengths=(4096, 8192),
            sample_rate_hz=10_000_000,
            native_sample_rate_hz=8_000_000,
            class_name="dsss",
            profile="wifi-hr-dsss-11m",
            pair_id="pair-1",
            scale_factor=1.25,
        )

        class Corpus:
            rows = [row]

            @staticmethod
            def prefix(_row, length):
                return np.ones(length, dtype=np.complex64)

        observations = openset.held_scale_evaluation_observations(
            Corpus()
        )
        self.assertEqual(
            [item.observation_length for item in observations],
            [4096, 8192],
        )
        for item in observations:
            self.assertEqual(item.inference.sample_rate_hz, 10_000_000)
            self.assertEqual(
                item.inference.native_sample_rate_hz, 8_000_000
            )
            self.assertFalse(hasattr(item.inference, "source_profile"))

    def test_truth_and_profile_are_joined_only_after_inference_scoring(self):
        observation = openset.EvaluationObservation(
            inference=openset.InferenceObservation(
                iq=np.ones(4096, dtype=np.complex64),
                prototype_source="current",
                sample_rate_hz=16_000_000,
                native_sample_rate_hz=8_000_000,
            ),
            truth_index=openset.PUBLIC_CLASSES.index("dsss"),
            source_profile="current:wifi-hr-dsss-11m",
            base_identity="identity",
            pair_id="pair",
            observation_length=4096,
            physical_scale_factor=2.0,
        )
        fake = openset.v4_openset._empty_population_scores(1)
        fake.prediction[0] = openset.PUBLIC_CLASSES.index("dsss")
        seen = []

        def score(_context, _policy, inference):
            seen.extend(inference)
            return fake

        with mock.patch.object(
            openset, "score_observations", side_effect=score
        ):
            _scores, cells = openset.score_evaluation_observations(
                object(), {}, [observation]
            )
        self.assertEqual(seen, [observation.inference])
        self.assertFalse(hasattr(seen[0], "truth_index"))
        self.assertEqual(cells[0].truth, "dsss")
        self.assertEqual(cells[0].source_profile, observation.source_profile)


class FittingFirewallTests(unittest.TestCase):
    def test_fit_signature_cannot_address_selection_or_novelty(self):
        parameters = set(
            inspect.signature(openset.fit_enrollment_policy).parameters
        )
        forbidden = {
            "train",
            "training",
            "selection",
            "adaptive_selection",
            "novelty",
            "sealed",
            "profile",
            "class_name",
            "calibration",
            "fusion_binding",
            "stage_one_template",
            "patch_length",
            "patch_count",
            "target_frac",
        }
        self.assertFalse(parameters & forbidden)
        self.assertIn("context", parameters)

    def test_enrollment_builder_cannot_request_train_or_selection_role(self):
        calls = []
        historical_row = object()
        current_row = object()

        def role_views(corpus, role, **kwargs):
            calls.append((corpus, role, kwargs))
            return [
                historical_row
                if corpus == "historical"
                else current_row
            ]

        context = SimpleNamespace(
            historical="historical",
            current="current",
        )
        with (
            mock.patch.object(
                openset,
                "_role_evaluation_observations",
                side_effect=role_views,
            ),
            mock.patch.object(
                openset,
                "_prepared_components",
                side_effect=RuntimeError("stop after population routing"),
            ),
            self.assertRaisesRegex(RuntimeError, "stop after"),
        ):
            openset._build_enrollment_calibration(
                context,
                _stage_one_template(),
            )
        self.assertEqual(
            [(role, kwargs) for _corpus, role, kwargs in calls],
            [
                (
                    "enrollment",
                    {"expand_current_observation_lengths": False},
                ),
                (
                    "enrollment",
                    {"expand_current_observation_lengths": False},
                ),
            ],
        )

    def test_selection_observation_builder_is_score_only_role(self):
        calls = []

        def role_views(corpus, role, **kwargs):
            calls.append((corpus, role, kwargs))
            return [corpus]

        context = SimpleNamespace(
            historical="historical",
            current="current",
        )
        with mock.patch.object(
            openset,
            "_role_evaluation_observations",
            side_effect=role_views,
        ):
            result = openset.selection_evaluation_observations(context)
        self.assertEqual(result, ["historical", "current"])
        self.assertEqual(
            calls,
            [
                (
                    "historical",
                    "selection",
                    {"expand_current_observation_lengths": False},
                ),
                (
                    "current",
                    "selection",
                    {"expand_current_observation_lengths": True},
                ),
            ],
        )

    def test_policy_uses_v5_schema_and_current_only_at_effective_n4096(self):
        policy = _fit_policy()
        self.assertEqual(policy["schema"], openset.POLICY_SCHEMA)
        self.assertEqual(
            set(
                policy["per_prototype_source"]["current"]["per_length"]
            ),
            {"4096"},
        )
        self.assertEqual(
            set(
                policy["per_prototype_source"]["historical"]["per_length"]
            ),
            {"4096", "8192", "16384"},
        )
        self.assertEqual(
            policy["inference_contract"]["current_novelty_geometry"],
            _novelty_geometry(),
        )
        for key, expected in openset.FITTING_FIREWALL.items():
            self.assertEqual(policy["fitting_contract"][key], expected)

    def test_policy_fit_rejects_selection_contamination(self):
        calibration = _calibration()
        contaminated_audit = dict(calibration.audit)
        contaminated_audit[
            "adaptive_selection_rows_used_for_threshold_or_rank_fit"
        ] = 1
        contaminated = openset._EnrollmentCalibration(
            stage_one_score_by_length=
                calibration.stage_one_score_by_length,
            phase_linearity_score_by_length=
                calibration.phase_linearity_score_by_length,
            class_distance_by_length=
                calibration.class_distance_by_length,
            calibration_class_distance_by_length=
                calibration.calibration_class_distance_by_length,
            enrollment_group_by_length=
                calibration.enrollment_group_by_length,
            enrollment_source_by_length=
                calibration.enrollment_source_by_length,
            enrollment_identity_by_length=
                calibration.enrollment_identity_by_length,
            audit=contaminated_audit,
        )
        with self.assertRaisesRegex(ValueError, "firewall"):
            _fit_policy(calibration=contaminated)

    def test_policy_validator_rejects_gate_or_source_hash_mutation(self):
        policy = _fit_policy()
        changed = deepcopy(policy)
        changed["gate_contract"]["inherited_gates"][0]["threshold"] = 0.99
        with self.assertRaisesRegex(ValueError, "gate contract"):
            openset.validate_policy(changed)
        changed = deepcopy(policy)
        changed["provenance"]["source_sha256"][
            "v5/calibrate_scale_orbit_openset.py"
        ] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source hashes"):
            openset.validate_policy(changed)
        changed = deepcopy(policy)
        changed["provenance"]["identity_acceptance_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source hashes"):
            openset.validate_policy(changed)
        changed = deepcopy(policy)
        changed["stage_one"]["parameters_by_length"]["4096"][
            "intercept"
        ] += 0.1
        with self.assertRaisesRegex(ValueError, "parameter digest"):
            openset.validate_policy(changed)
        changed = deepcopy(policy)
        changed["classifier_binding"][
            "selected_profile_is_classifier_input"
        ] = True
        with self.assertRaisesRegex(ValueError, "classifier binding"):
            openset.validate_policy(changed)
        changed = deepcopy(policy)
        changed["inference_contract"]["current_novelty_geometry"][
            "sample_rate_ratio"
        ] = 1.5
        with self.assertRaisesRegex(ValueError, "novelty geometry"):
            openset.validate_policy(changed)

    def test_policy_scoring_context_binding_rejects_fusion_cross_swap(self):
        policy = _fit_policy()
        context = _context()
        with mock.patch.object(
            openset,
            "build_fusion_binding",
            return_value=_binding(),
        ):
            openset._validate_policy_context_binding(context, policy)
        swapped = _binding()
        swapped["source_fusion_dev_metrics_sha256"] = "c" * 64
        with (
            mock.patch.object(
                openset,
                "build_fusion_binding",
                return_value=swapped,
            ),
            self.assertRaisesRegex(ValueError, "does not match"),
        ):
            openset._validate_policy_context_binding(context, policy)

    def test_scoring_checks_binding_before_any_feature_extraction(self):
        observation = openset.InferenceObservation(
            iq=np.ones(4096, dtype=np.complex64),
            prototype_source="current",
            sample_rate_hz=16_000_000,
            native_sample_rate_hz=8_000_000,
        )
        with (
            mock.patch.object(openset, "validate_policy"),
            mock.patch.object(
                openset,
                "_validate_policy_context_binding",
                side_effect=RuntimeError("cross-swapped classifier"),
            ),
            mock.patch.object(
                openset,
                "prepare_inference_observation",
                side_effect=AssertionError("feature extraction ran"),
            ),
            self.assertRaisesRegex(RuntimeError, "cross-swapped"),
        ):
            openset.score_observations(
                object(),
                {},
                [observation],
            )


class ExactGateContractTests(unittest.TestCase):
    def test_perfect_inventory_passes_all_exact_21_gates(self):
        cells = _perfect_gate_cells()
        report = openset.evaluate_current_gate_contract(cells)
        expected_names = (
            set(openset.recovery_validator.EXPECTED_INHERITED_GATES)
            | set(openset.recovery_validator.EXPECTED_DIRECTIONAL_GATES)
        )
        self.assertEqual(set(report["gates"]), expected_names)
        self.assertEqual(len(report["gates"]), 21)
        self.assertTrue(report["all_pass"])
        self.assertTrue(
            report["coverage"]["exact_observation_coverage"]
        )

    def test_single_length_single_scale_inventory_cannot_pass(self):
        cells = [
            row
            for row in _perfect_gate_cells()
            if (
                row.observation_length == 4096
                and row.physical_scale_factor == 1.0
            )
        ]
        report = openset.evaluate_current_gate_contract(cells)
        self.assertFalse(report["all_pass"])
        self.assertFalse(
            report["gates"][
                "exact_eligible_pair_and_observation_coverage"
            ]["passes"]
        )
        self.assertFalse(
            report["gates"][
                "exact_profile_length_scale_key_and_count_coverage"
            ]["passes"]
        )
        self.assertEqual(
            report["coverage"][
                "observed_source_profile_length_scale_cells"
            ],
            len(openset.corpus_data.CURRENT_PROFILE_PUBLIC_CLASS_MAP),
        )

    def test_every_required_cell_must_have_preregistered_minimum_pairs(self):
        report = openset.evaluate_current_gate_contract(
            _perfect_gate_cells(
                openset
                .REQUIRED_CURRENT_EVALUATION_MINIMUM_PAIRS_PER_PROFILE
                - 1
            )
        )
        self.assertFalse(
            report["coverage"]["profile_pair_floor_complete"]
        )
        self.assertFalse(report["all_pass"])

    def test_wifi_to_bluetooth_directional_failure_is_not_pooled_away(self):
        cells = _perfect_gate_cells()
        target = next(
            index
            for index, row in enumerate(cells)
            if (
                row.source_profile
                == openset.WIFI_HR_DSSS_SOURCE_PROFILE
                and row.observation_length == 4096
                and row.physical_scale_factor == 1.0
            )
        )
        row = cells[target]
        cells[target] = openset.ScoredGateCell(
            source_profile=row.source_profile,
            truth=row.truth,
            predicted="bluetooth",
            rejected=False,
            pair_id=row.pair_id,
            observation_length=row.observation_length,
            physical_scale_factor=row.physical_scale_factor,
        )
        report = openset.evaluate_current_gate_contract(cells)
        gate = report["gates"][
            "wifi_hr_dsss_to_bluetooth_worst_legal_cell_confusion_rate"
        ]
        self.assertEqual(gate["value"], 0.05)
        self.assertEqual(gate["threshold"], 0.02)
        self.assertFalse(gate["passes"])
        self.assertFalse(report["all_pass"])

    def test_bluetooth_to_dsss_directional_failure_is_exact(self):
        cells = _perfect_gate_cells()
        target = next(
            index
            for index, row in enumerate(cells)
            if (
                row.truth == "bluetooth"
                and row.observation_length == 8192
                and row.physical_scale_factor == 1.25
            )
        )
        row = cells[target]
        cells[target] = openset.ScoredGateCell(
            source_profile=row.source_profile,
            truth=row.truth,
            predicted="dsss",
            rejected=False,
            pair_id=row.pair_id,
            observation_length=row.observation_length,
            physical_scale_factor=row.physical_scale_factor,
        )
        report = openset.evaluate_current_gate_contract(cells)
        gate = report["gates"][
            "bluetooth_to_dsss_worst_legal_cell_confusion_rate"
        ]
        self.assertEqual(gate["value"], 0.05)
        self.assertFalse(gate["passes"])

    def test_missing_cell_fails_both_exact_coverage_gates(self):
        cells = _perfect_gate_cells()
        report = openset.evaluate_current_gate_contract(cells[:-1])
        self.assertFalse(
            report["gates"][
                "exact_eligible_pair_and_observation_coverage"
            ]["passes"]
        )
        self.assertFalse(
            report["gates"][
                "exact_profile_length_scale_key_and_count_coverage"
            ]["passes"]
        )

    def test_duplicate_extra_pair_view_fails_exact_coverage(self):
        cells = _perfect_gate_cells()
        report = openset.evaluate_current_gate_contract(
            cells + [cells[0]]
        )
        self.assertFalse(report["all_pass"])
        self.assertFalse(
            report["coverage"]["pair_inventory_complete"]
        )
        self.assertFalse(
            report["coverage"]["exact_observation_coverage"]
        )
        self.assertFalse(
            report["coverage"]["exact_profile_length_scale_coverage"]
        )

    def test_missing_bluetooth_profile_cannot_hide_directional_damage(self):
        cells = _perfect_gate_cells()
        missing_profile = next(
            f"current:{profile}"
            for profile, public_class
            in openset.corpus_data.CURRENT_PROFILE_PUBLIC_CLASS_MAP.items()
            if public_class == "bluetooth"
        )
        reduced = [
            row for row in cells
            if row.source_profile != missing_profile
        ]
        report = openset.evaluate_current_gate_contract(reduced)
        self.assertFalse(
            report["gates"][
                "exact_profile_length_scale_key_and_count_coverage"
            ]["passes"]
        )
        directional = report["gates"][
            "bluetooth_to_dsss_worst_legal_cell_confusion_rate"
        ]
        self.assertEqual(directional["value"], 1.0)
        self.assertFalse(directional["passes"])

    def test_frozen_gate_contract_is_byte_for_byte_validator_contract(self):
        contract = openset._gate_contract()
        inherited = {
            row["name"]: (row["comparison"], row["threshold"])
            for row in contract["inherited_gates"]
        }
        directional = {
            row["name"]: {
                key: value
                for key, value in row.items()
                if key != "name"
            }
            for row in contract["added_directional_confusion_gates"]
        }
        self.assertEqual(
            inherited,
            openset.recovery_validator.EXPECTED_INHERITED_GATES,
        )
        self.assertEqual(
            directional,
            openset.recovery_validator.EXPECTED_DIRECTIONAL_GATES,
        )


class LoaderAndProvenanceTests(unittest.TestCase):
    def test_context_requires_four_distinct_paths_before_any_corpus_load(self):
        with (
            mock.patch.object(
                openset, "_validate_contract_files", return_value={}
            ),
            mock.patch.object(
                openset,
                "load_verified_v5_fusion",
                side_effect=AssertionError("fusion load should not run"),
            ),
            self.assertRaisesRegex(ValueError, "must all differ"),
        ):
            openset.load_development_context(
                "/development/same",
                "/development/same",
                "/development/current",
                "/development/selection",
                device_name="cpu",
            )

    def test_v4_fusion_schema_is_rejected_before_artifact_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(
                openset.browser_export,
                "_load_json",
                return_value={
                    "status": "complete",
                    "development_only": True,
                    "release_evidence": False,
                    "sealed_release_data_used": 0,
                    "consumed_historical_test_rows_used": 0,
                    "consumed_test_rows_exposed": 0,
                    "encoder": "fusion",
                    "run_configuration": {
                        "schema":
                            "v4-current-source-development-fusion-v1"
                    },
                },
            ):
                with self.assertRaisesRegex(ValueError, "not a v5"):
                    openset.load_verified_v5_fusion(temporary)

    def test_source_snapshot_pins_v5_contracts_and_reused_v4_math(self):
        paths = openset._executed_source_paths()
        for name in (
            "v5/calibrate_scale_orbit_openset.py",
            "v5/assemble_scale_orbit_fusion.py",
            "v5/run_scale_orbit_dev.py",
            "v5/scale_orbit_data.py",
            "v5/trusted_geometry_canonicalizer.py",
            "v5/pretraining_amendment.json",
            "v5/recovery_protocol.json",
            "v5/seed_registry.json",
            "v5/identity_firewall_amendment.json",
            "v5/seed20262904_identity_rejection.json",
            "v5/seed20262904_identity_firewall_acceptance.json",
            "v5/validate_identity_firewall_amendment.py",
            "training/zplane_ab/v2_full_variation/v4_current_source/"
            "calibrate_current_source_openset.py",
        ):
            self.assertIn(name, paths)
            self.assertTrue(paths[name].is_file())
        hashes = openset._source_hashes()
        self.assertEqual(set(hashes), set(paths))
        self.assertTrue(all(openset._is_sha256(value) for value in hashes.values()))


if __name__ == "__main__":
    unittest.main()
