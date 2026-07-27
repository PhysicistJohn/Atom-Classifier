"""Unit suite for the pose-degeneracy rejector features.

The properties under test are the ones the module is only useful if it has:

* determinism, byte for byte, on repeated evaluation;
* strictly no frequency transform, checked both by source inspection and by
  executing the extractor with the transform APIs patched to raise;
* a fixed, total, finite schema;
* exact invariance to a global phase rotation and to a global amplitude scale,
  which is what lets an additive rejector consume these features without
  reintroducing the scale dependence the whole v3 front end exists to remove;
* documented behaviour on the degenerate inputs that motivated the module:
  all-zero, DC, a pure tone, and white noise.
"""
from __future__ import annotations

import inspect
import json
import os
import sys
import unittest
from unittest import mock

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
_V2_DIR = os.path.dirname(HERE)
_ZPAB_DIR = os.path.dirname(_V2_DIR)
_TRAINING_DIR = os.path.dirname(_ZPAB_DIR)
for _path in (_TRAINING_DIR, _ZPAB_DIR, _V2_DIR, HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import pose_degeneracy as pdg  # noqa: E402
import time_domain_geometry as td  # noqa: E402


def _multitone(length: int, *, centre: float = 0.11328125) -> np.ndarray:
    """A well-conditioned narrowband emission, as in the geometry suite."""
    n = np.arange(length, dtype=np.float64)
    offsets = np.array((-0.03125, -0.015625, 0.0, 0.015625, 0.03125))
    phases = np.array((0.2, -1.1, 0.7, 2.0, -0.4))
    frequencies = centre + offsets
    return np.sum(
        np.exp(1j * (2.0 * np.pi * frequencies[:, None] * n + phases[:, None])),
        axis=0,
    )


def _white_noise(length: int, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal(length) + 1j * rng.standard_normal(length)
    ) / np.sqrt(2.0)


def _tone(length: int, frequency: float = 0.1) -> np.ndarray:
    return np.exp(2j * np.pi * frequency * np.arange(length, dtype=np.float64))


class SchemaTest(unittest.TestCase):
    def test_feature_schema_is_fixed_named_and_unique(self):
        names = pdg.feature_names()
        self.assertIsInstance(names, tuple)
        self.assertEqual(len(names), pdg.FEATURE_COUNT)
        # Pinned: the vector is a published contract, so a change here must be
        # a deliberate edit to this number, not a silent widening.
        self.assertEqual(pdg.FEATURE_COUNT, 21)
        self.assertEqual(len(set(names)), len(names))
        self.assertEqual(names, pdg.FEATURE_NAMES)
        for name in names:
            self.assertRegex(name, r"^[a-z0-9_]+$")

    def test_metadata_is_explicit_and_serializable(self):
        metadata = pdg.pose_degeneracy_metadata()
        self.assertEqual(metadata["version"], pdg.POSE_DEGENERACY_VERSION)
        self.assertEqual(metadata["estimator_version"], td.ESTIMATOR_VERSION)
        self.assertEqual(metadata["feature_names"], list(pdg.FEATURE_NAMES))
        self.assertEqual(metadata["feature_count"], pdg.FEATURE_COUNT)
        self.assertFalse(metadata["uses_frequency_transform"])
        self.assertFalse(metadata["learned_calibration"])
        self.assertFalse(metadata["uses_absolute_scale"])
        self.assertFalse(metadata["changes_closed_label"])
        self.assertTrue(metadata["invariant_to_global_phase"])
        self.assertTrue(metadata["invariant_to_global_scale"])
        json.dumps(metadata, allow_nan=False)

    def test_report_shape_ordering_and_finiteness(self):
        report = pdg.pose_degeneracy_report(_multitone(4096))
        features = report["features"]
        self.assertEqual(features.dtype, np.float64)
        self.assertEqual(features.shape, (pdg.FEATURE_COUNT,))
        self.assertTrue(np.all(np.isfinite(features)))
        self.assertEqual(tuple(report["values"]), pdg.FEATURE_NAMES)
        np.testing.assert_array_equal(
            features,
            np.asarray(
                [report["values"][name] for name in pdg.FEATURE_NAMES],
                dtype=np.float64,
            ),
        )
        pdg.assert_schema(report["values"])
        with self.assertRaises(ValueError):
            pdg.assert_schema({"not_a_feature": 0.0})

    def test_context_is_serializable_and_carries_no_absolute_scale(self):
        report = pdg.pose_degeneracy_report(_multitone(2048))
        context = report["context"]
        json.dumps(context, allow_nan=False)
        self.assertEqual(context["version"], pdg.POSE_DEGENERACY_VERSION)
        self.assertFalse(context["uses_frequency_transform"])
        self.assertNotIn("raw_peak", context["estimator_context"])
        self.assertNotIn("raw_peak", context)

    def test_matrix_helper_stacks_in_order(self):
        captures = [_multitone(1024), _white_noise(1024), _tone(1024)]
        matrix = pdg.pose_degeneracy_matrix(captures)
        self.assertEqual(matrix.shape, (3, pdg.FEATURE_COUNT))
        for row, capture in enumerate(captures):
            np.testing.assert_array_equal(
                matrix[row], pdg.pose_degeneracy_features(capture)
            )
        self.assertEqual(
            pdg.pose_degeneracy_matrix([]).shape, (0, pdg.FEATURE_COUNT)
        )


class DeterminismTest(unittest.TestCase):
    def test_repeated_evaluation_is_bit_for_bit_identical(self):
        for capture in (
            _multitone(4096),
            _white_noise(4096),
            _tone(4096),
            np.ones(4096, dtype=np.complex128),
        ):
            with self.subTest(capture=len(capture)):
                first = pdg.pose_degeneracy_features(capture)
                second = pdg.pose_degeneracy_features(capture)
                self.assertTrue(np.array_equal(first, second))

    def test_context_is_reproduced_exactly(self):
        capture = _white_noise(2048, seed=5)
        first = pdg.pose_degeneracy_report(capture)["context"]
        second = pdg.pose_degeneracy_report(capture)["context"]
        self.assertEqual(json.dumps(first), json.dumps(second))

    def test_input_capture_is_not_mutated(self):
        capture = _multitone(1024)
        before = capture.copy()
        pdg.pose_degeneracy_features(capture)
        self.assertTrue(np.array_equal(capture, before))


class TransformFreeTest(unittest.TestCase):
    def test_source_contains_no_transform_call(self):
        source = inspect.getsource(pdg)
        for forbidden in ("np.fft.", "numpy.fft.", "welch_psd", "scipy.signal"):
            self.assertNotIn(forbidden, source.replace("``welch_psd``", ""))

    def test_extractor_runs_with_transform_apis_forbidden(self):
        import preprocess  # noqa: F401 -- patched below, never called

        forbidden = AssertionError("frequency transform must not be called")
        with mock.patch("numpy.fft.fft", side_effect=forbidden), mock.patch(
            "numpy.fft.ifft", side_effect=forbidden
        ), mock.patch(
            "numpy.fft.fftshift", side_effect=forbidden
        ), mock.patch(
            "numpy.fft.rfft", side_effect=forbidden
        ), mock.patch(
            "preprocess.welch_psd", side_effect=forbidden
        ), mock.patch(
            "preprocess.estimate_band", side_effect=forbidden
        ):
            for capture in (_multitone(4096), _white_noise(4096)):
                report = pdg.pose_degeneracy_report(capture)
                self.assertTrue(np.all(np.isfinite(report["features"])))
                self.assertFalse(report["context"]["uses_frequency_transform"])

    def test_lag_arithmetic_matches_the_estimator_exactly(self):
        capture = _multitone(777)
        normalized = capture / np.max(np.abs(capture))
        for lag in (1, 2, 4, 8, 16, 32):
            with self.subTest(lag=lag):
                self.assertEqual(
                    pdg._nonzero_lag(normalized, lag),
                    td._nonzero_lag(normalized, lag),
                )


class InvarianceTest(unittest.TestCase):
    def test_global_phase_rotation_leaves_every_feature_unchanged(self):
        for capture in (_multitone(4096), _white_noise(4096), _tone(4096)):
            baseline = pdg.pose_degeneracy_features(capture)
            for phase in (0.731, -2.4, np.pi):
                with self.subTest(phase=phase, length=len(capture)):
                    rotated = pdg.pose_degeneracy_features(
                        np.exp(1j * phase) * capture
                    )
                    np.testing.assert_allclose(
                        rotated, baseline, rtol=0.0, atol=1e-8
                    )

    def test_power_of_two_gain_is_bit_for_bit_invariant(self):
        for capture in (_multitone(4096), _white_noise(4096)):
            baseline = pdg.pose_degeneracy_features(capture)
            for gain in (2.0 ** -40, 2.0 ** -7, 2.0 ** 7, 2.0 ** 40):
                with self.subTest(gain=gain, length=len(capture)):
                    scaled = pdg.pose_degeneracy_features(gain * capture)
                    self.assertTrue(np.array_equal(scaled, baseline))

    def test_arbitrary_gain_is_invariant_to_rounding(self):
        for capture in (_multitone(4096), _white_noise(4096)):
            baseline = pdg.pose_degeneracy_features(capture)
            for gain in (3.7e-5, 0.3, 11.9, 4.2e6):
                with self.subTest(gain=gain, length=len(capture)):
                    scaled = pdg.pose_degeneracy_features(gain * capture)
                    np.testing.assert_allclose(
                        scaled, baseline, rtol=0.0, atol=1e-8
                    )

    def test_gain_and_phase_together_are_invariant(self):
        capture = _multitone(2048)
        baseline = pdg.pose_degeneracy_features(capture)
        moved = pdg.pose_degeneracy_features(
            5.5e3 * np.exp(1j * 0.9) * capture
        )
        np.testing.assert_allclose(moved, baseline, rtol=0.0, atol=1e-8)


class DegenerateInputTest(unittest.TestCase):
    def _value(self, capture, name):
        return pdg.pose_degeneracy_report(capture)["values"][name]

    def test_invalid_captures_fail_closed_like_the_estimator(self):
        invalid = (
            np.array([], dtype=np.complex128),
            np.ones(2, dtype=np.complex128),
            np.ones((2, 3), dtype=np.complex128),
            np.ones(8, dtype=np.float64),
            np.array([1.0 + 0.0j, 2.0 + 0.0j, np.nan + 0.0j]),
            np.array([1.0 + 0.0j, 2.0 + 0.0j, np.inf + 0.0j]),
        )
        for capture in invalid:
            with self.subTest(shape=capture.shape, dtype=str(capture.dtype)):
                with self.assertRaises((TypeError, ValueError)):
                    td.estimate_band(capture)
                with self.assertRaises((TypeError, ValueError)):
                    pdg.pose_degeneracy_features(capture)

    def test_all_zero_capture_is_rejected_not_posed(self):
        zeros = np.zeros(4096, dtype=np.complex128)
        with self.assertRaises(ValueError):
            td.estimate_band(zeros)
        with self.assertRaises(ValueError):
            pdg.pose_degeneracy_features(zeros)

    def test_dc_capture_is_maximally_coherent_and_stable(self):
        report = pdg.pose_degeneracy_report(np.ones(4096, dtype=np.complex128))
        values = report["values"]
        self.assertAlmostEqual(values["log10_relative_lag1_magnitude"], 0.0, places=12)
        self.assertAlmostEqual(values["peak_normalized_mean_power"], 1.0, places=12)
        self.assertAlmostEqual(values["carrier_phase_coherence"], 1.0, places=9)
        self.assertEqual(values["bandwidth_log_dispersion"], 0.0)
        self.assertEqual(values["prefix_bandwidth_log_dispersion"], 0.0)
        self.assertEqual(values["prefix_centre_dispersion"], 0.0)
        self.assertEqual(values["degenerate_full_band_fallback"], 0.0)
        self.assertEqual(values["bandwidth_clipped_to_minimum"], 1.0)

    def test_pure_tone_is_coherent_and_prefix_stable(self):
        values = pdg.pose_degeneracy_report(_tone(4096))["values"]
        self.assertGreater(values["log10_lag1_sampling_margin"], 1.0)
        self.assertAlmostEqual(values["carrier_phase_coherence"], 1.0, places=6)
        self.assertEqual(values["prefix_bandwidth_log_dispersion"], 0.0)
        self.assertLess(values["prefix_relative_lag1_log_dispersion"], 1e-12)
        self.assertEqual(values["high_lag_qualified_fraction"], 1.0)

    def test_white_noise_reads_as_a_degenerate_pose(self):
        """This is the mechanism claim of HANDOFF 17, stated as a test."""
        noise = pdg.pose_degeneracy_report(_white_noise(4096))["values"]
        emission = pdg.pose_degeneracy_report(_multitone(4096))["values"]
        self.assertLess(
            noise["log10_relative_lag1_magnitude"],
            emission["log10_relative_lag1_magnitude"] - 1.0,
        )
        self.assertLess(noise["log10_lag1_sampling_margin"], 0.0)
        self.assertGreater(emission["log10_lag1_sampling_margin"], 0.0)
        self.assertEqual(noise["high_lag_qualified_fraction"], 0.0)
        self.assertGreater(
            noise["prefix_bandwidth_log_dispersion"],
            emission["prefix_bandwidth_log_dispersion"],
        )
        self.assertLess(
            noise["carrier_phase_coherence"],
            emission["carrier_phase_coherence"],
        )
        self.assertLess(noise["peak_normalized_mean_power"], 0.5)

    def test_short_capture_below_the_prefix_floor_still_returns_the_schema(self):
        short = _multitone(40)
        self.assertLess(len(short), pdg.MIN_PREFIX_SAMPLES)
        report = pdg.pose_degeneracy_report(short)
        self.assertEqual(report["features"].shape, (pdg.FEATURE_COUNT,))
        self.assertTrue(np.all(np.isfinite(report["features"])))
        self.assertEqual(report["context"]["prefix_observations"], [])
        self.assertEqual(
            report["values"]["prefix_bandwidth_log_dispersion"], 0.0
        )

    def test_gated_capture_with_an_all_zero_prefix_is_counted_not_crashed(self):
        """The dev corpus contains windows whose emission starts after 4096."""
        capture = np.zeros(4096, dtype=np.complex128)
        capture[2048:] = _multitone(2048)
        report = pdg.pose_degeneracy_report(capture)
        values = report["values"]
        self.assertGreater(values["zero_magnitude_prefix_fraction"], 0.0)
        self.assertTrue(np.all(np.isfinite(report["features"])))
        zero_audit = [
            entry
            for entry in report["context"]["prefix_observations"]
            if entry["zero_magnitude"]
        ]
        self.assertTrue(zero_audit)
        for entry in zero_audit:
            self.assertIsNone(entry["bandwidth"])
        json.dumps(report["context"], allow_nan=False)

    def test_no_zero_prefix_scores_zero_on_that_feature(self):
        for capture in (_multitone(4096), _white_noise(4096), _tone(4096)):
            with self.subTest(length=len(capture)):
                self.assertEqual(
                    self._value(capture, "zero_magnitude_prefix_fraction"), 0.0
                )

    def test_tiny_capture_shorter_than_the_maximum_lag(self):
        report = pdg.pose_degeneracy_report(_multitone(9))
        self.assertTrue(np.all(np.isfinite(report["features"])))
        self.assertLessEqual(
            len(report["context"]["per_lag_bandwidth"]), len(td.BASE_LAGS)
        )

    def test_degenerate_fallback_row_is_flagged_and_total(self):
        """A gated capture has an exactly zero lag one and no carrier at all."""
        capture = np.array([1.0, 0.0] * 2048, dtype=np.complex128)
        _centre, bandwidth, context = td.estimate_band(capture)
        self.assertTrue(context["degenerate_full_band_fallback"])
        self.assertEqual(bandwidth, td.FULL_BAND_BW)
        values = pdg.pose_degeneracy_report(capture)["values"]
        self.assertEqual(values["degenerate_full_band_fallback"], 1.0)
        self.assertEqual(values["high_lag_qualified_fraction"], 0.0)
        self.assertTrue(
            np.all(np.isfinite(pdg.pose_degeneracy_features(capture)))
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
