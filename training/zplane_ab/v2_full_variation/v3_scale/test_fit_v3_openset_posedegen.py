"""Unit tests for the pose-degeneracy open-set rejector variant.

These tests never touch the development corpus, the consumed test half, or any
sealed release suite.  Every population is synthetic, small, and constructed in
the test.  No novelty seed used here is a reserved validation seed except where
the test exists to prove that such a seed is refused.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fit_v3_openset as base  # noqa: E402
import fit_v3_openset_posedegen as subject  # noqa: E402
import time_domain_geometry as geometry  # noqa: E402
import v3_time_domain_openset as frozen  # noqa: E402
from test_fit_v3_openset import FusionArtifactValidationTests  # noqa: E402


PATCH_COUNT = frozen.PATCH_COUNT
PATCH_LENGTH = frozen.PATCH_LENGTH
PACKED_WIDTH = PATCH_COUNT * PATCH_LENGTH

# A design seed that is deliberately none of the reserved, spent, sealed or
# release seeds.  Matches the module's proposal.
DESIGN_SEED = subject.PROPOSED_DESIGN_NOVELTY_SEED


def _packed(rows: int, seed: int, *, scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(scale=scale, size=(rows, 2, PACKED_WIDTH)).astype(np.float32)


def _pose(rows: int, features: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(rows, features)).astype(np.float64)


def _fake_posedegen(
    *,
    name: str = "fake_posedegen",
    version: str = "fake-v1",
    feature_names: tuple[str, ...] = ("a", "b", "c"),
    features=None,
) -> types.ModuleType:
    """A minimal module satisfying (or, for negative tests, violating) the contract."""
    module = types.ModuleType(name)
    module.POSE_DEGENERACY_VERSION = version
    module.FEATURE_NAMES = feature_names

    def default_features(iq, *, min_bandwidth=None, **_kwargs):
        capture = np.asarray(iq)
        magnitude = float(np.mean(np.abs(capture)))
        return np.asarray(
            [magnitude, float(len(capture)), float(min_bandwidth or 0.0)][
                : len(feature_names)
            ],
            dtype=np.float64,
        )

    module.pose_degeneracy_features = features or default_features
    return module


def _install(module: types.ModuleType) -> None:
    sys.modules[module.__name__] = module


class SiblingModuleContractTests(unittest.TestCase):
    def tearDown(self) -> None:
        for name in list(sys.modules):
            if name.startswith("fake_posedegen") or name == "broken_posedegen":
                del sys.modules[name]

    def test_the_real_sibling_module_satisfies_the_contract(self) -> None:
        source = subject.load_posedegen_module()
        self.assertEqual(source.module_name, subject.POSEDEGEN_MODULE_DEFAULT)
        self.assertTrue(source.version)
        self.assertEqual(source.feature_count, len(source.feature_names))
        self.assertGreater(source.feature_count, 0)
        self.assertEqual(len(source.source_sha256 or ""), 64)

    def test_an_absent_module_fails_loudly_and_names_the_contract(self) -> None:
        with self.assertRaises(RuntimeError) as caught:
            subject.load_posedegen_module("no_such_pose_degeneracy_module")
        message = str(caught.exception)
        self.assertIn("not importable", message)
        self.assertIn("will not silently degrade", message)
        self.assertIn("POSE_DEGENERACY_VERSION", message)

    def test_the_import_is_lazy(self) -> None:
        """Importing this module must not require the sibling to exist."""
        source = Path(subject.__file__).read_text(encoding="utf-8")
        head = source.split("# ---", 1)[0]
        self.assertNotIn("import pose_degeneracy", head)
        self.assertIn("importlib.import_module", source)

    def test_missing_symbols_are_listed(self) -> None:
        module = _fake_posedegen(name="broken_posedegen")
        del module.FEATURE_NAMES
        _install(module)
        with self.assertRaises(RuntimeError) as caught:
            subject.load_posedegen_module("broken_posedegen")
        self.assertIn("FEATURE_NAMES", str(caught.exception))

    def test_a_non_callable_extractor_is_refused(self) -> None:
        module = _fake_posedegen(name="broken_posedegen")
        module.pose_degeneracy_features = 3
        _install(module)
        with self.assertRaisesRegex(RuntimeError, "non-callable"):
            subject.load_posedegen_module("broken_posedegen")

    def test_a_string_feature_name_list_is_refused(self) -> None:
        module = _fake_posedegen(name="broken_posedegen")
        module.FEATURE_NAMES = "abc"
        _install(module)
        with self.assertRaisesRegex(RuntimeError, "not a sequence"):
            subject.load_posedegen_module("broken_posedegen")

    def test_duplicate_feature_names_are_refused(self) -> None:
        module = _fake_posedegen(name="broken_posedegen", feature_names=("a", "a"))
        _install(module)
        with self.assertRaisesRegex(RuntimeError, "unique feature names"):
            subject.load_posedegen_module("broken_posedegen")

    def test_an_empty_version_is_refused(self) -> None:
        module = _fake_posedegen(name="broken_posedegen", version="")
        _install(module)
        with self.assertRaisesRegex(RuntimeError, "POSE_DEGENERACY_VERSION"):
            subject.load_posedegen_module("broken_posedegen")

    def test_an_empty_module_name_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty module name"):
            subject.load_posedegen_module("")

    def test_a_wrong_width_feature_vector_is_refused(self) -> None:
        module = _fake_posedegen(
            name="broken_posedegen",
            features=lambda iq, **_kwargs: np.zeros(2, dtype=np.float64),
        )
        _install(module)
        source = subject.load_posedegen_module("broken_posedegen")
        with self.assertRaisesRegex(RuntimeError, "expected"):
            subject.pose_matrix(
                source,
                [np.zeros(256, dtype=np.complex128)],
                patch_length=PATCH_LENGTH,
                target_frac=0.5,
                population="probe",
            )

    def test_a_non_finite_feature_vector_is_refused(self) -> None:
        module = _fake_posedegen(
            name="broken_posedegen",
            features=lambda iq, **_kwargs: np.full(3, np.nan, dtype=np.float64),
        )
        _install(module)
        source = subject.load_posedegen_module("broken_posedegen")
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            subject.pose_matrix(
                source,
                [np.zeros(256, dtype=np.complex128)],
                patch_length=PATCH_LENGTH,
                target_frac=0.5,
                population="probe",
            )

    def test_an_empty_population_is_refused(self) -> None:
        source = subject.load_posedegen_module()
        with self.assertRaisesRegex(ValueError, "at least one row"):
            subject.pose_matrix(
                source,
                [],
                patch_length=PATCH_LENGTH,
                target_frac=0.5,
                population="probe",
            )

    def test_features_are_computed_at_the_frontend_resolution_floor(self) -> None:
        """The degeneracy must describe the pose the frontend actually adopted."""
        self.assertTrue(subject.POSEDEGEN_MATCH_FRONTEND_MIN_BANDWIDTH)
        for raw_length in (1024, 4096, 16384):
            self.assertEqual(
                subject.frontend_min_bandwidth(raw_length, PATCH_LENGTH, 0.5),
                geometry.minimum_bandwidth_for_active_span(
                    raw_length, PATCH_LENGTH, 0.5
                ),
            )

    def test_the_floor_is_passed_through_to_the_extractor(self) -> None:
        seen: list[float] = []

        def recording(iq, *, min_bandwidth=None, **_kwargs):
            seen.append(min_bandwidth)
            return np.zeros(3, dtype=np.float64)

        module = _fake_posedegen(name="broken_posedegen", features=recording)
        _install(module)
        source = subject.load_posedegen_module("broken_posedegen")
        subject.pose_row(
            source,
            np.zeros(4096, dtype=np.complex128),
            patch_length=PATCH_LENGTH,
            target_frac=0.5,
        )
        self.assertEqual(
            seen,
            [geometry.minimum_bandwidth_for_active_span(4096, PATCH_LENGTH, 0.5)],
        )


class InheritedContractTests(unittest.TestCase):
    """Everything reused from the base module must be the *same* object."""

    def test_gate_floors_are_the_base_objects(self) -> None:
        self.assertIs(subject.GATE_FLOORS, base.GATE_FLOORS)
        self.assertIs(subject.KNOWN_FUR_CEILING, base.KNOWN_FUR_CEILING)

    def test_gate_floors_are_the_frozen_values(self) -> None:
        self.assertEqual(
            subject.GATE_FLOORS,
            {
                "overall_auroc": 0.72,
                "noise_auroc": 0.80,
                "chirp_auroc": 0.80,
                "noise_threshold_recall": 0.10,
                "chirp_threshold_recall": 0.10,
            },
        )
        self.assertEqual(subject.KNOWN_FUR_CEILING, 0.10)

    def test_rank_weights_and_quantile_are_the_frozen_constants(self) -> None:
        self.assertIs(subject.BRANCH_LOF_RANK_WEIGHT, frozen.FROZEN_V2_WEIGHT)
        self.assertIs(subject.GEOMETRY_RANK_WEIGHT, frozen.FROZEN_GEOMETRY_WEIGHT)
        self.assertIs(
            subject.THRESHOLD_QUANTILE, frozen.FROZEN_THRESHOLD_QUANTILE
        )
        self.assertAlmostEqual(
            subject.BRANCH_LOF_RANK_WEIGHT + subject.GEOMETRY_RANK_WEIGHT, 1.0
        )

    def test_branch_lof_shape_is_inherited_and_unflagged(self) -> None:
        self.assertIs(subject.BRANCH_LOF, base.BRANCH_LOF)
        flags = {
            action.dest for action in subject.build_parser()._actions
        }
        self.assertNotIn("branch_lof", flags)
        self.assertNotIn("posedegen_reducer", flags)

    def test_closed_label_and_path_guards_are_the_base_functions(self) -> None:
        self.assertIs(
            subject.assert_closed_label_unchanged,
            base.assert_closed_label_unchanged,
        )
        self.assertIs(subject.reject_sealed_path, base.reject_sealed_path)
        self.assertIs(subject.gate_summary, base.gate_summary)
        self.assertIs(subject.family_metrics, base.family_metrics)
        self.assertIs(subject.load_fusion_artifact, base.load_fusion_artifact)

    def test_release_and_sealed_paths_are_refused(self) -> None:
        for candidate in (
            "training/artifacts/releases/whatever",
            "artifacts/releases/20260731",
        ):
            with self.assertRaises(Exception):
                subject.reject_sealed_path(Path(candidate), "output")


class WeightHyperparameterTests(unittest.TestCase):
    def test_the_weight_flag_is_required_and_has_no_default(self) -> None:
        actions = {
            action.dest: action for action in subject.build_parser()._actions
        }
        self.assertTrue(actions["posedegen_weight"].required)
        self.assertIsNone(actions["posedegen_weight"].default)
        self.assertTrue(actions["design_novelty_seed"].required)
        self.assertIsNone(actions["design_novelty_seed"].default)
        self.assertTrue(actions["role"].required)

    def test_the_parser_refuses_a_missing_weight(self) -> None:
        parser = subject.build_parser()
        with io.StringIO() as noise, contextlib.redirect_stderr(noise):
            with self.assertRaises(SystemExit):
                parser.parse_args(
                    [
                        "--fusion-dir",
                        "x",
                        "--output-dir",
                        "y",
                        "--role",
                        "validate",
                        "--design-novelty-seed",
                        str(DESIGN_SEED),
                    ]
                )
            self.assertIn("--posedegen-weight", noise.getvalue())

    def test_invalid_weights_are_refused(self) -> None:
        for candidate in (None, np.nan, np.inf, -0.001, 1.0, 1.5, "abc"):
            with self.assertRaises(ValueError):
                subject.validate_posedegen_weight(candidate)

    def test_valid_weights_are_accepted(self) -> None:
        for candidate in (0.0, 0.05, 0.5, 0.999):
            self.assertEqual(
                subject.validate_posedegen_weight(candidate), float(candidate)
            )

    def test_the_missing_weight_message_names_the_fresh_design_seed(self) -> None:
        with self.assertRaises(ValueError) as caught:
            subject.validate_posedegen_weight(None)
        message = str(caught.exception)
        self.assertIn(str(subject.PROPOSED_DESIGN_NOVELTY_SEED), message)
        self.assertIn(str(subject.RESERVED_VALIDATION_SEEDS[0]), message)

    def test_the_selection_protocol_is_documented_on_the_parser(self) -> None:
        epilog = subject.build_parser().epilog or ""
        self.assertIn(str(subject.PROPOSED_DESIGN_NOVELTY_SEED), epilog)
        self.assertIn("20260938", epilog)
        self.assertIn("evidence rule 4", epilog)


class SeedHygieneTests(unittest.TestCase):
    def _plan(self, role, design, seeds):
        return subject.validate_seed_plan(role, design, seeds)

    def test_a_validation_seed_may_not_be_the_design_seed(self) -> None:
        for seed in subject.RESERVED_VALIDATION_SEEDS:
            with self.assertRaises(ValueError) as caught:
                self._plan("design", seed, [seed])
            self.assertIn("validation evidence into training evidence", str(caught.exception))

    def test_the_spent_sealed_and_release_seeds_are_refused_as_design(self) -> None:
        for seed in (
            subject.SPENT_DESIGN_NOVELTY_SEED,
            subject.SEALED_RELEASE_SEED,
            subject.RELEASE_SEED_NEVER_SPENT_HERE,
        ):
            with self.assertRaises(ValueError):
                self._plan("design", seed, [seed])

    def test_validate_role_refuses_design_and_novelty_overlap(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be the seed that validates"):
            self._plan("validate", DESIGN_SEED, [DESIGN_SEED])

    def test_validate_role_accepts_the_reserved_pair(self) -> None:
        role, design, seeds = self._plan(
            "validate", DESIGN_SEED, list(subject.RESERVED_VALIDATION_SEEDS)
        )
        self.assertEqual(role, "validate")
        self.assertEqual(design, DESIGN_SEED)
        self.assertEqual(seeds, subject.RESERVED_VALIDATION_SEEDS)

    def test_design_role_refuses_reserved_validation_seeds(self) -> None:
        with self.assertRaisesRegex(ValueError, "stay untouched for validation"):
            self._plan(
                "design",
                DESIGN_SEED,
                [DESIGN_SEED, subject.RESERVED_VALIDATION_SEEDS[0]],
            )

    def test_design_role_requires_its_own_design_seed(self) -> None:
        with self.assertRaisesRegex(ValueError, "must draw novelty at its declared"):
            self._plan("design", DESIGN_SEED, [20260942])

    def test_design_role_accepts_the_declared_seed(self) -> None:
        role, design, seeds = self._plan("design", DESIGN_SEED, [DESIGN_SEED])
        self.assertEqual((role, design, seeds), ("design", DESIGN_SEED, (DESIGN_SEED,)))

    def test_spent_sealed_and_release_seeds_are_refused_as_novelty(self) -> None:
        for seed in (
            subject.SPENT_DESIGN_NOVELTY_SEED,
            subject.SEALED_RELEASE_SEED,
            subject.RELEASE_SEED_NEVER_SPENT_HERE,
        ):
            with self.assertRaises(ValueError):
                self._plan("design", DESIGN_SEED, [DESIGN_SEED, seed])

    def test_a_non_integer_or_absent_design_seed_is_refused(self) -> None:
        for candidate in (None, 1.5, "20260941", True):
            with self.assertRaises(ValueError):
                self._plan("design", candidate, [DESIGN_SEED])

    def test_an_unknown_role_is_refused(self) -> None:
        for candidate in ("Design", "evidence", "", None):
            with self.assertRaises(ValueError):
                subject.validate_role(candidate)

    def test_novelty_seed_defaults_are_role_dependent(self) -> None:
        validate_args = argparse.Namespace(role="validate", novelty_seeds=None)
        self.assertEqual(
            subject.resolve_novelty_seeds(validate_args),
            list(subject.RESERVED_VALIDATION_SEEDS),
        )
        design_args = argparse.Namespace(role="design", novelty_seeds=None)
        with self.assertRaisesRegex(ValueError, "never a design default"):
            subject.resolve_novelty_seeds(design_args)
        explicit = argparse.Namespace(role="design", novelty_seeds=[DESIGN_SEED])
        self.assertEqual(subject.resolve_novelty_seeds(explicit), [DESIGN_SEED])


FEATURES = ("a", "b", "c")


def _fit(
    *,
    weight: float,
    n_classes: int = 2,
    training_packed: np.ndarray | None = None,
    enrollment_packed: np.ndarray | None = None,
    training_pose: np.ndarray | None = None,
    enrollment_pose: np.ndarray | None = None,
    enrollment_ranks: np.ndarray | None = None,
) -> tuple[subject.PoseDegenOpenSet, dict[str, np.ndarray]]:
    training_packed = _packed(12, 101) if training_packed is None else training_packed
    enrollment_packed = (
        _packed(9, 202) if enrollment_packed is None else enrollment_packed
    )
    training_pose = (
        _pose(len(training_packed), len(FEATURES), 303)
        if training_pose is None
        else training_pose
    )
    enrollment_pose = (
        _pose(len(enrollment_packed), len(FEATURES), 404)
        if enrollment_pose is None
        else enrollment_pose
    )
    training_labels = (np.arange(len(training_packed)) % n_classes).astype(np.int64)
    enrollment_predicted = (
        np.arange(len(enrollment_packed)) % n_classes
    ).astype(np.int64)
    if enrollment_ranks is None:
        enrollment_ranks = np.linspace(
            0.0, 0.9, len(enrollment_packed), dtype=np.float64
        )
    policy = subject.fit_policy(
        training_packed,
        training_labels,
        training_pose,
        enrollment_packed,
        enrollment_predicted,
        enrollment_ranks,
        enrollment_pose,
        n_classes=n_classes,
        posedegen_weight=weight,
        posedegen_feature_names=FEATURES,
        posedegen_version="fake-v1",
    )
    return policy, {
        "training_packed": training_packed,
        "training_labels": training_labels,
        "training_pose": training_pose,
        "enrollment_packed": enrollment_packed,
        "enrollment_predicted": enrollment_predicted,
        "enrollment_pose": enrollment_pose,
        "enrollment_ranks": np.asarray(enrollment_ranks, dtype=np.float64),
    }


class BlendTests(unittest.TestCase):
    def test_weight_zero_is_bit_identical_to_the_frozen_policy(self) -> None:
        policy, parts = _fit(weight=0.0)
        reference = frozen.FrozenV3OpenSet.fit(
            parts["training_packed"],
            parts["training_labels"],
            parts["enrollment_packed"],
            parts["enrollment_predicted"],
            parts["enrollment_ranks"],
            n_classes=2,
        )
        self.assertEqual(policy.threshold, reference.threshold)
        np.testing.assert_array_equal(
            policy.calibration_scores, reference.calibration_scores
        )
        np.testing.assert_array_equal(
            policy.combined_calibration_raw, reference.combined_calibration_raw
        )
        np.testing.assert_array_equal(
            policy.score(
                parts["enrollment_ranks"],
                parts["enrollment_packed"],
                parts["enrollment_pose"],
                parts["enrollment_predicted"],
            ),
            reference.score(
                parts["enrollment_ranks"],
                parts["enrollment_packed"],
                parts["enrollment_predicted"],
            ),
        )

    def test_a_positive_weight_changes_the_score(self) -> None:
        parts = _fit(weight=0.0)[1]
        scores = []
        for weight in (0.0, 0.4):
            policy, _ = _fit(weight=weight, **{
                key: parts[key]
                for key in (
                    "training_packed",
                    "enrollment_packed",
                    "training_pose",
                    "enrollment_pose",
                    "enrollment_ranks",
                )
            })
            scores.append(
                policy.score(
                    parts["enrollment_ranks"],
                    parts["enrollment_packed"],
                    parts["enrollment_pose"],
                    parts["enrollment_predicted"],
                )
            )
        self.assertFalse(np.array_equal(scores[0], scores[1]))

    def test_the_frozen_pair_keeps_its_proportions(self) -> None:
        lof = np.array([0.1, 0.9], dtype=np.float64)
        geom = np.array([0.9, 0.1], dtype=np.float64)
        pose = np.array([0.5, 0.5], dtype=np.float64)
        blended = subject.blend(lof, geom, pose, 0.25)
        expected = 0.75 * (0.80 * lof + 0.20 * geom) + 0.25 * pose
        np.testing.assert_array_equal(blended, expected)

    def test_the_blend_is_nondecreasing_in_the_pose_rank(self) -> None:
        lof = np.full(4, 0.5)
        geom = np.full(4, 0.5)
        low = subject.blend(lof, geom, np.full(4, 0.1), 0.3)
        high = subject.blend(lof, geom, np.full(4, 0.9), 0.3)
        self.assertTrue(np.all(high >= low))
        self.assertTrue(np.any(high > low))

    def test_the_blend_refuses_an_invalid_weight(self) -> None:
        with self.assertRaises(ValueError):
            subject.blend(np.zeros(2), np.zeros(2), np.zeros(2), 1.0)

    def test_score_is_nondecreasing_in_the_pose_nonconformity(self) -> None:
        policy, parts = _fit(weight=0.5)
        near = policy.pose.class_mean[parts["enrollment_predicted"]]
        far = near + 50.0 * policy.pose.class_scale[parts["enrollment_predicted"]]
        low = policy.score(
            parts["enrollment_ranks"],
            parts["enrollment_packed"],
            near,
            parts["enrollment_predicted"],
        )
        high = policy.score(
            parts["enrollment_ranks"],
            parts["enrollment_packed"],
            far,
            parts["enrollment_predicted"],
        )
        self.assertTrue(np.all(high >= low))
        self.assertGreater(float(np.mean(high)), float(np.mean(low)))


class TrainOnlyDensityTests(unittest.TestCase):
    def test_pose_statistics_are_fit_on_training_rows_only(self) -> None:
        training_pose = _pose(12, len(FEATURES), 303)
        labels = (np.arange(12) % 2).astype(np.int64)
        fitted = subject.KnownOnlyPoseDegeneracy.fit(
            training_pose, labels, n_classes=2, feature_names=FEATURES
        )
        for class_index in range(2):
            rows = training_pose[labels == class_index]
            np.testing.assert_allclose(
                fitted.class_mean[class_index], rows.mean(axis=0)
            )
            np.testing.assert_allclose(
                fitted.class_scale[class_index],
                np.maximum(rows.std(axis=0), 1e-5),
            )

    def test_fit_policy_takes_no_selection_or_novelty_argument(self) -> None:
        import inspect

        names = set(inspect.signature(subject.fit_policy).parameters)
        for forbidden in ("selection", "novelty", "release", "test"):
            self.assertFalse(
                any(forbidden in name for name in names),
                f"fit_policy exposes a {forbidden!r} parameter",
            )

    def test_a_class_with_one_row_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two rows"):
            subject.KnownOnlyPoseDegeneracy.fit(
                _pose(2, len(FEATURES), 5),
                np.array([0, 1], dtype=np.int64),
                n_classes=2,
                feature_names=FEATURES,
            )

    def test_a_wrong_width_or_non_finite_matrix_is_refused(self) -> None:
        labels = (np.arange(6) % 2).astype(np.int64)
        with self.assertRaisesRegex(ValueError, "matrix"):
            subject.KnownOnlyPoseDegeneracy.fit(
                _pose(6, len(FEATURES) + 1, 5),
                labels,
                n_classes=2,
                feature_names=FEATURES,
            )
        poisoned = _pose(6, len(FEATURES), 5)
        poisoned[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            subject.KnownOnlyPoseDegeneracy.fit(
                poisoned, labels, n_classes=2, feature_names=FEATURES
            )

    def test_the_scale_floor_prevents_a_zero_divisor(self) -> None:
        constant = np.ones((6, len(FEATURES)), dtype=np.float64)
        labels = (np.arange(6) % 2).astype(np.int64)
        fitted = subject.KnownOnlyPoseDegeneracy.fit(
            constant, labels, n_classes=2, feature_names=FEATURES
        )
        self.assertTrue(np.all(fitted.class_scale >= 1e-5))
        self.assertTrue(
            np.isfinite(fitted.nonconformity(constant, labels)).all()
        )

    def test_a_schema_change_is_refused_at_scoring_time(self) -> None:
        fitted = subject.KnownOnlyPoseDegeneracy.fit(
            _pose(6, len(FEATURES), 5),
            (np.arange(6) % 2).astype(np.int64),
            n_classes=2,
            feature_names=FEATURES,
        )
        with self.assertRaisesRegex(RuntimeError, "schema changed"):
            fitted.deviations(
                _pose(6, len(FEATURES) + 2, 7),
                (np.arange(6) % 2).astype(np.int64),
            )


class EnrollmentOnlyRankTests(unittest.TestCase):
    def test_threshold_is_the_enrollment_q95(self) -> None:
        policy, _ = _fit(weight=0.3)
        self.assertEqual(
            policy.threshold,
            float(
                np.quantile(
                    policy.calibration_scores, frozen.FROZEN_THRESHOLD_QUANTILE
                )
            ),
        )

    def test_calibration_lengths_match_enrollment(self) -> None:
        policy, parts = _fit(weight=0.3)
        rows = len(parts["enrollment_packed"])
        self.assertEqual(len(policy.pose_calibration_raw), rows)
        self.assertEqual(len(policy.geometry_calibration_raw), rows)
        self.assertEqual(len(policy.combined_calibration_raw), rows)
        self.assertEqual(len(policy.calibration_scores), rows)

    def test_a_different_enrollment_pose_moves_the_threshold(self) -> None:
        first, parts = _fit(weight=0.3)
        second, _ = _fit(
            weight=0.3,
            training_packed=parts["training_packed"],
            enrollment_packed=parts["enrollment_packed"],
            training_pose=parts["training_pose"],
            enrollment_pose=parts["enrollment_pose"] * 7.0 + 1.0,
            enrollment_ranks=parts["enrollment_ranks"],
        )
        self.assertNotEqual(
            first.pose_calibration_raw.tolist(),
            second.pose_calibration_raw.tolist(),
        )

    def test_mismatched_row_counts_are_refused(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "aligned with rows|row counts differ"
        ):
            _fit(weight=0.3, enrollment_pose=_pose(3, len(FEATURES), 9))

    def test_mismatched_row_counts_are_refused_at_scoring_time(self) -> None:
        policy, parts = _fit(weight=0.3)
        with self.assertRaisesRegex(
            ValueError, "aligned with rows|row counts differ"
        ):
            policy.score(
                parts["enrollment_ranks"][:3],
                parts["enrollment_packed"],
                parts["enrollment_pose"],
                parts["enrollment_predicted"],
            )

    def test_a_rank_outside_the_unit_interval_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\[0, 1\)"):
            _fit(weight=0.3, enrollment_ranks=np.linspace(0.0, 1.0, 9))


class SerializationTests(unittest.TestCase):
    def test_round_trip_is_exact(self) -> None:
        policy, _ = _fit(weight=0.35)
        restored = subject.PoseDegenOpenSet.from_payload(policy.to_payload())
        self.assertEqual(restored.threshold, policy.threshold)
        self.assertEqual(restored.posedegen_weight, policy.posedegen_weight)
        self.assertEqual(restored.posedegen_version, policy.posedegen_version)
        self.assertEqual(restored.pose.feature_names, policy.pose.feature_names)
        np.testing.assert_array_equal(
            restored.pose.class_mean, policy.pose.class_mean
        )
        np.testing.assert_array_equal(
            restored.pose_calibration_raw, policy.pose_calibration_raw
        )

    def test_the_payload_is_pickle_free(self) -> None:
        policy, _ = _fit(weight=0.35)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "policy.npz"
            np.savez_compressed(path, **policy.to_payload())
            with np.load(path, allow_pickle=False) as loaded:
                restored = subject.PoseDegenOpenSet.from_payload(
                    {key: loaded[key] for key in loaded.files}
                )
        self.assertEqual(restored.threshold, policy.threshold)

    def test_a_missing_or_extra_key_is_refused(self) -> None:
        policy, _ = _fit(weight=0.35)
        payload = policy.to_payload()
        broken = dict(payload)
        broken.pop("pose_calibration_raw")
        with self.assertRaisesRegex(ValueError, "keys differ"):
            subject.PoseDegenOpenSet.from_payload(broken)
        extra = dict(payload)
        extra["surprise"] = np.asarray(1.0)
        with self.assertRaisesRegex(ValueError, "keys differ"):
            subject.PoseDegenOpenSet.from_payload(extra)

    def test_a_tampered_constant_is_refused(self) -> None:
        policy, _ = _fit(weight=0.35)
        for key, value in (
            ("branch_lof_rank_weight", 0.5),
            ("geometry_weight", 0.5),
            ("threshold_quantile", 0.9),
            ("posedegen_reducer", "max"),
            ("kind", "something-else"),
        ):
            payload = policy.to_payload()
            payload[key] = np.asarray(value)
            with self.assertRaisesRegex(ValueError, "does not match policy"):
                subject.PoseDegenOpenSet.from_payload(payload)

    def test_a_tampered_threshold_or_weight_is_refused(self) -> None:
        policy, _ = _fit(weight=0.35)
        payload = policy.to_payload()
        payload["threshold"] = np.asarray(0.123, dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "threshold is invalid"):
            subject.PoseDegenOpenSet.from_payload(payload)
        payload = policy.to_payload()
        payload["posedegen_weight"] = np.asarray(1.4, dtype=np.float64)
        with self.assertRaises(ValueError):
            subject.PoseDegenOpenSet.from_payload(payload)

    def test_an_unsorted_calibration_is_refused(self) -> None:
        policy, _ = _fit(weight=0.35)
        payload = policy.to_payload()
        payload["pose_calibration_raw"] = payload["pose_calibration_raw"][::-1]
        with self.assertRaisesRegex(ValueError, "must be sorted"):
            subject.PoseDegenOpenSet.from_payload(payload)

    def test_the_serialized_kind_records_both_policies(self) -> None:
        policy, _ = _fit(weight=0.35)
        payload = policy.to_payload()
        self.assertEqual(
            str(payload["kind"].item()), subject.POSEDEGEN_POLICY_KIND
        )
        self.assertEqual(
            str(payload["base_policy_kind"].item()), frozen.FROZEN_POLICY_KIND
        )


class PoseRecorderTests(unittest.TestCase):
    def _recorder(self) -> subject._PoseRecorder:
        module = _fake_posedegen(name="fake_posedegen_recorder")
        _install(module)
        source = subject.load_posedegen_module("fake_posedegen_recorder")
        calls: list[int] = []

        def inner(captures, *, patch_length, patch_count, target_frac):
            calls.append(len(captures))
            rows = len(captures)
            return (
                np.zeros((rows, 2, PACKED_WIDTH), dtype=np.float32),
                np.zeros((rows, 12), dtype=np.float32),
                [{} for _ in range(rows)],
            )

        recorder = subject._PoseRecorder(inner, source, target_frac=0.5)
        recorder.calls = calls
        return recorder

    def tearDown(self) -> None:
        sys.modules.pop("fake_posedegen_recorder", None)

    def test_the_wrapped_call_is_delegated_unchanged(self) -> None:
        recorder = self._recorder()
        captures = [np.ones(512, dtype=np.complex128) for _ in range(3)]
        packed, features, contexts = recorder(
            captures, patch_length=PATCH_LENGTH, patch_count=PATCH_COUNT,
            target_frac=0.5,
        )
        self.assertEqual(recorder.calls, [3])
        self.assertEqual(packed.shape, (3, 2, PACKED_WIDTH))
        self.assertEqual(features.shape, (3, 12))
        self.assertEqual(len(contexts), 3)
        self.assertEqual(recorder.matrices[0].shape, (3, 3))

    def test_recorded_matrices_are_keyed_by_split_in_order(self) -> None:
        recorder = self._recorder()
        for rows in (5, 3, 2):
            recorder(
                [np.ones(512, dtype=np.complex128) for _ in range(rows)],
                patch_length=PATCH_LENGTH,
                patch_count=PATCH_COUNT,
                target_frac=0.5,
            )
        data = {
            "xtr": np.zeros(5),
            "xen": np.zeros(3),
            "xva": np.zeros(2),
        }
        keyed = subject._recorded_pose(recorder, data)
        self.assertEqual(
            set(keyed), {subject.CENTER_SPLIT, subject.PROTOTYPE_SPLIT, subject.METRIC_SPLIT}
        )
        self.assertEqual(len(keyed[subject.CENTER_SPLIT]), 5)
        self.assertEqual(len(keyed[subject.METRIC_SPLIT]), 2)

    def test_a_missing_population_is_refused(self) -> None:
        recorder = self._recorder()
        recorder(
            [np.ones(512, dtype=np.complex128)],
            patch_length=PATCH_LENGTH,
            patch_count=PATCH_COUNT,
            target_frac=0.5,
        )
        with self.assertRaisesRegex(AssertionError, "no longer routes"):
            subject._recorded_pose(recorder, {"xtr": np.zeros(1)})

    def test_a_misaligned_population_is_refused(self) -> None:
        recorder = self._recorder()
        for rows in (5, 3, 2):
            recorder(
                [np.ones(512, dtype=np.complex128) for _ in range(rows)],
                patch_length=PATCH_LENGTH,
                patch_count=PATCH_COUNT,
                target_frac=0.5,
            )
        with self.assertRaisesRegex(AssertionError, "misaligned"):
            subject._recorded_pose(
                recorder,
                {"xtr": np.zeros(4), "xen": np.zeros(3), "xva": np.zeros(2)},
            )


class NoveltyGenerationTests(unittest.TestCase):
    def _generate(self, seeds=(DESIGN_SEED,), lengths=(1024, 2048)):
        source = subject.load_posedegen_module()
        return (
            subject.generate_novelty(
                seeds,
                n_each=2,
                lengths=lengths,
                feature_mean=np.zeros(12, np.float32),
                feature_std=np.ones(12, np.float32),
                patch_length=PATCH_LENGTH,
                patch_count=PATCH_COUNT,
                target_frac=0.5,
                posedegen=source,
            ),
            source,
        )

    def test_every_fixture_carries_an_aligned_pose_matrix(self) -> None:
        (fixtures, _provenance), source = self._generate()
        for family in subject.NOVELTY_FAMILIES:
            for length in (1024, 2048):
                fixture = fixtures[DESIGN_SEED][family][length]
                self.assertEqual(fixture.packed.shape, (2, 2, PACKED_WIDTH))
                self.assertEqual(
                    fixture.pose.shape, (2, source.feature_count)
                )
                self.assertTrue(np.isfinite(fixture.pose).all())

    def test_generation_is_deterministic(self) -> None:
        (first, _), _ = self._generate()
        (second, _), _ = self._generate()
        np.testing.assert_array_equal(
            first[DESIGN_SEED]["noise"][1024].pose,
            second[DESIGN_SEED]["noise"][1024].pose,
        )
        np.testing.assert_array_equal(
            first[DESIGN_SEED]["chirp"][2048].packed,
            second[DESIGN_SEED]["chirp"][2048].packed,
        )

    def test_the_prefix_provenance_is_recorded(self) -> None:
        (_, provenance), _ = self._generate()
        entry = provenance[str(DESIGN_SEED)]["noise"]
        self.assertEqual(entry["generated_once_at"], 2048)
        self.assertEqual(set(entry["prefix_sha256"]), {"1024", "2048"})

    def test_noise_and_chirp_pose_features_differ(self) -> None:
        """The features must actually separate the two novelty families."""
        (fixtures, _), _ = self._generate()
        noise = fixtures[DESIGN_SEED]["noise"][2048].pose
        chirp = fixtures[DESIGN_SEED]["chirp"][2048].pose
        self.assertFalse(np.allclose(noise.mean(axis=0), chirp.mean(axis=0)))

    def test_an_empty_or_duplicated_seed_set_is_refused(self) -> None:
        source = subject.load_posedegen_module()
        for seeds in ([], [DESIGN_SEED, DESIGN_SEED]):
            with self.assertRaises(ValueError):
                subject.generate_novelty(
                    seeds,
                    n_each=1,
                    lengths=(1024,),
                    feature_mean=np.zeros(12, np.float32),
                    feature_std=np.ones(12, np.float32),
                    patch_length=PATCH_LENGTH,
                    patch_count=PATCH_COUNT,
                    target_frac=0.5,
                    posedegen=source,
                )


class EndToEndRunTests(unittest.TestCase):
    """Drive ``run`` over a tiny synthetic fusion, without the real corpus.

    The corpus-bound half (``rebuild_populations``) is substituted; everything
    downstream -- density fit, enrollment ranks, threshold, novelty generation
    with real pose-degeneracy features, scoring, gate reporting and the written
    contract -- is the real code path.
    """

    def _populations(self, feature_count: int) -> "subject.Populations":
        import torch
        from invariant_patch_cnn import InvariantPatchCNN, InvariantPatchConfig
        from train import embed_all

        torch.manual_seed(0)
        common = {
            "patch_length": PATCH_LENGTH,
            "patch_count": PATCH_COUNT,
            "patch_dim": 8,
            "hidden": 12,
            "embed_dim": 6,
            "n_features": 12,
            "dropout": 0.0,
        }
        nets = {
            "real": InvariantPatchCNN(
                InvariantPatchConfig(encoder="real", **common)
            ).eval(),
            "complex": InvariantPatchCNN(
                InvariantPatchConfig(encoder="complex", **common)
            ).eval(),
        }
        device = torch.device("cpu")
        n_classes = 3
        rows = {"tr": 90, "en": 45, "va": 24}
        packed = {
            split: _packed(count, 900 + index)
            for index, (split, count) in enumerate(rows.items())
        }
        features = {
            split: np.random.default_rng(31 + index)
            .normal(size=(count, 12))
            .astype(np.float32)
            for index, (split, count) in enumerate(rows.items())
        }
        branch_embeddings = {
            name: {
                split: embed_all(
                    net, packed[split], features[split], device
                ).astype(np.float32)
                for split in ("tr", "en", "va")
            }
            for name, net in nets.items()
        }
        real_center = subject.assemble.fit_center(branch_embeddings["real"])
        complex_center = subject.assemble.fit_center(branch_embeddings["complex"])
        fused = {
            split: subject.assemble.fuse_numpy(
                branch_embeddings["real"][split],
                branch_embeddings["complex"][split],
                real_center,
                complex_center,
                weight_real=0.5,
            )
            for split in ("tr", "en", "va")
        }
        labels = {
            split: (np.arange(count) % n_classes).astype(np.int64)
            for split, count in rows.items()
        }
        prototypes = subject.assemble.fit_prototypes(
            fused, labels["en"], n_classes
        )
        data = {
            "xtr": packed["tr"],
            "ytr": labels["tr"],
            "xen": packed["en"],
            "yen": labels["en"],
            "xva": packed["va"],
            "yva": labels["va"],
            "n_classes": n_classes,
            "fmean": np.zeros(12, np.float32),
            "fstd": np.ones(12, np.float32),
            "data_audit": {"consumed_test_rows_exposed": 0},
            "cache_contract": {
                "config": {
                    "patch_length": PATCH_LENGTH,
                    "patch_count": PATCH_COUNT,
                    "target_frac": 0.5,
                }
            },
        }
        return subject.Populations(
            data=data,
            nets=nets,
            branch_embeddings=branch_embeddings,
            fused=fused,
            real_center=real_center,
            complex_center=complex_center,
            prototypes=prototypes,
            reproduction={
                "tolerance": 1e-4,
                "bit_identical_to_artifact": True,
            },
            branch_hashes={"real": {}, "complex": {}},
            pose={
                "tr": _pose(rows["tr"], feature_count, 11),
                "en": _pose(rows["en"], feature_count, 12),
                "va": _pose(rows["va"], feature_count, 13),
            },
        )

    def _args(self, root: Path, **overrides) -> argparse.Namespace:
        fusion_dir = FusionArtifactValidationTests()._artifact(root / "fusion")
        values = {
            "fusion_dir": str(fusion_dir),
            "output_dir": str(root / "openset"),
            "role": "validate",
            "posedegen_weight": 0.25,
            "design_novelty_seed": DESIGN_SEED,
            "posedegen_module": subject.POSEDEGEN_MODULE_DEFAULT,
            "device": "cpu",
            "novelty_seeds": [subject.RESERVED_VALIDATION_SEEDS[0]],
            "novelty_n": 2,
            "prefix_lengths": [1024, 2048],
            "reproduction_tolerance": 1e-4,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def _run(self, root: Path, **overrides) -> dict:
        args = self._args(root, **overrides)
        source = subject.load_posedegen_module(args.posedegen_module)
        populations = self._populations(source.feature_count)
        with mock.patch.object(
            subject, "rebuild_populations", return_value=populations
        ):
            return subject.run(args)

    def test_run_writes_the_declared_contract(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            report = self._run(Path(root))
            written = json.loads(
                (Path(root) / "openset" / "openset_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual(written["sealed_release_data_used"], 0)
        self.assertEqual(written["consumed_test_rows_used"], 0)
        self.assertFalse(written["release_seed_20260731_used"])
        self.assertFalse(written["closed_label_altered"])
        self.assertEqual(
            written["fitting_contract"]["density_population"], "training only"
        )
        self.assertEqual(
            written["fitting_contract"]["rank_and_threshold_population"],
            "enrollment only",
        )
        self.assertFalse(written["fitting_contract"]["policy_searched_here"])
        self.assertEqual(
            written["policy"]["branch_lof_rank_weight"],
            subject.BRANCH_LOF_RANK_WEIGHT,
        )
        self.assertEqual(written["policy"]["posedegen_rank_weight"], 0.25)
        self.assertTrue(written["policy"]["reduces_to_frozen_policy_at_weight_zero"])
        self.assertEqual(report["status"], written["status"])

    def test_the_report_records_the_design_seed_and_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            report = self._run(Path(root))
        hyper = report["hyperparameter"]
        self.assertEqual(hyper["design_novelty_seed"], DESIGN_SEED)
        self.assertFalse(hyper["posedegen_weight_has_default"])
        self.assertTrue(hyper["design_novelty_seed_is_fresh"])
        self.assertEqual(
            hyper["reserved_validation_seeds"],
            list(subject.RESERVED_VALIDATION_SEEDS),
        )
        self.assertIn("evidence rule 4", hyper["selection_protocol"])
        self.assertEqual(report["seeds"]["design_novelty_seed"], DESIGN_SEED)
        self.assertEqual(
            report["seeds"]["spent_design_novelty_seed_excluded"], 20260938
        )

    def test_the_report_records_the_pose_module_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            report = self._run(Path(root))
        pose = report["posedegen"]
        self.assertEqual(pose["module"], subject.POSEDEGEN_MODULE_DEFAULT)
        self.assertEqual(pose["reducer"], subject.POSEDEGEN_REDUCER)
        self.assertTrue(pose["min_bandwidth_matches_frontend"])
        self.assertFalse(pose["reaches_closed_classifier"])
        self.assertEqual(len(pose["source_sha256"]), 64)
        self.assertEqual(set(pose["population_sha256"]), {
            "training", "enrollment", "selection"
        })
        self.assertIn(
            "fit_v3_openset_posedegen.py", report["source_sha256"]
        )
        self.assertIn("pose_degeneracy.py", report["source_sha256"])

    def test_validate_and_design_roles_label_their_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            validated = self._run(Path(root))
        self.assertEqual(validated["role"], "validate")
        self.assertTrue(validated["gates_are_evidence"])
        self.assertTrue(validated["status"].startswith("development_openset_"))

        with tempfile.TemporaryDirectory() as root:
            designed = self._run(
                Path(root),
                role="design",
                novelty_seeds=[DESIGN_SEED],
            )
        self.assertEqual(designed["role"], "design")
        self.assertFalse(designed["gates_are_evidence"])
        self.assertTrue(designed["status"].startswith("design_selection_"))
        self.assertIn("NOT validation evidence", designed["evidence_role"])

    def test_run_reports_every_family_at_every_prefix_length(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            report = self._run(Path(root))
        rows = report["novelty"][str(subject.RESERVED_VALIDATION_SEEDS[0])]
        self.assertEqual(set(rows), {"1024", "2048"})
        for row in rows.values():
            for family in ("noise", "chirp", "overall"):
                self.assertIn("auroc", row[family])
                self.assertIn("threshold_recall", row[family])
        self.assertIn("known_false_unknown_rate", report["gates"])

    def test_run_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as first_root:
            first = self._run(Path(first_root))
        with tempfile.TemporaryDirectory() as second_root:
            second = self._run(Path(second_root))
        self.assertEqual(first["novelty"], second["novelty"])
        self.assertEqual(first["known"], second["known"])
        self.assertEqual(
            first["policy"]["threshold"], second["policy"]["threshold"]
        )
        self.assertEqual(first["posedegen"], second["posedegen"])

    def test_an_absent_pose_module_stops_the_run_before_any_output(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            args = self._args(
                Path(root), posedegen_module="no_such_pose_degeneracy_module"
            )
            with self.assertRaisesRegex(RuntimeError, "not importable"):
                subject.run(args)
            self.assertFalse((Path(root) / "openset").exists())

    def test_a_missing_weight_stops_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            args = self._args(Path(root), posedegen_weight=None)
            with self.assertRaisesRegex(ValueError, "required and has no default"):
                subject.run(args)
            self.assertFalse((Path(root) / "openset").exists())

    def test_a_validation_seed_as_the_design_seed_stops_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            args = self._args(
                Path(root),
                design_novelty_seed=subject.RESERVED_VALIDATION_SEEDS[0],
            )
            with self.assertRaisesRegex(ValueError, "training evidence"):
                subject.run(args)
            self.assertFalse((Path(root) / "openset").exists())

    def test_run_refuses_a_non_empty_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            existing = Path(root) / "openset"
            existing.mkdir()
            (existing / "stale.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                self._run(Path(root))


if __name__ == "__main__":
    unittest.main()
