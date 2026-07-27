"""Focused invariance and invalid-input tests for the no-transform geometry path."""
from __future__ import annotations

import inspect
import json
import os
import sys
import unittest
from unittest import mock

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import time_domain_geometry as td  # noqa: E402
import time_domain_invariant_patch_preprocess as tdp  # noqa: E402


def _multitone(
    length: int,
    *,
    centre: float = 0.11328125,
    scale: float = 1.0,
) -> np.ndarray:
    n = np.arange(length, dtype=np.float64)
    offsets = np.array((-0.03125, -0.015625, 0.0, 0.015625, 0.03125))
    phases = np.array((0.2, -1.1, 0.7, 2.0, -0.4))
    frequencies = scale * (centre + offsets)
    return np.sum(
        np.exp(1j * (2.0 * np.pi * frequencies[:, None] * n + phases[:, None])),
        axis=0,
    )


def _complex_patches(channels: np.ndarray) -> np.ndarray:
    packed = channels[0].astype(np.float64) + 1j * channels[1].astype(np.float64)
    return packed.reshape(16, 64)


class TimeDomainGeometryTest(unittest.TestCase):
    def test_metadata_is_explicit_and_serializable(self):
        metadata = td.estimator_metadata()
        self.assertEqual(metadata["version"], "time-correlation-v1")
        self.assertEqual(metadata["base_lags"], [1, 2, 4, 8, 16])
        self.assertEqual(metadata["maximum_lag"], 32)
        self.assertFalse(metadata["uses_frequency_transform"])
        self.assertFalse(metadata["learned_calibration"])
        json.dumps(metadata, allow_nan=False)

    def test_gain_and_global_phase_invariance_at_extreme_scales(self):
        raw = _multitone(4096)
        expected = td.estimate_band(raw)
        for gain in (1e-200, 1e-100, 1e-12, 1e12, 1e100, 1e200):
            for phase in (0.0, 0.731):
                with self.subTest(gain=gain, phase=phase):
                    centre, bandwidth, context = td.estimate_band(
                        gain * np.exp(1j * phase) * raw
                    )
                    self.assertAlmostEqual(centre, expected[0], places=13)
                    self.assertAlmostEqual(bandwidth, expected[1], places=13)
                    self.assertEqual(
                        context["bandwidth_clipped_to_minimum"],
                        expected[2]["bandwidth_clipped_to_minimum"],
                    )

    def test_carrier_shift_equivariance_and_width_invariance(self):
        raw = _multitone(4096)
        centre, bandwidth, _ = td.estimate_band(raw)
        shift = -0.071
        n = np.arange(len(raw), dtype=np.float64)
        shifted = raw * np.exp(1j * 2.0 * np.pi * shift * n)
        shifted_centre, shifted_bandwidth, _ = td.estimate_band(shifted)
        expected_centre = (centre + shift + 0.5) % 1.0 - 0.5
        self.assertAlmostEqual(shifted_centre, expected_centre, places=13)
        self.assertAlmostEqual(shifted_bandwidth, bandwidth, places=13)

    def test_stationary_geometry_does_not_encode_capture_length(self):
        rows = [td.estimate_band(_multitone(length))[:2] for length in (1024, 2048, 4096)]
        for centre, bandwidth in rows[1:]:
            # Ordinary (non-circular) lag estimates have O(1/N) endpoint terms.
            self.assertLess(abs(centre - rows[0][0]), 3e-5)
            self.assertLess(abs(bandwidth - rows[0][1]), 7e-4)

    def test_unresolved_width_floor_keeps_one_patch_across_lengths(self):
        for length in (512, 1024, 4096, 16384):
            n = np.arange(length, dtype=np.float64)
            raw = np.exp(1j * 2.0 * np.pi * 0.071 * n)
            _channels, _features, context = tdp.preprocess(raw)
            self.assertEqual(context["active_length"], 64)
            self.assertAlmostEqual(
                context["bw"],
                63.0 * 0.5 / (length - 1),
                places=14,
            )

    def test_physical_sample_scale_law_on_analytic_waveform(self):
        # Frequencies scale by s while sample count scales by 1/s, preserving
        # physical observation duration.  All tones remain away from Nyquist.
        estimates = {}
        for scale in (0.5, 1.0, 2.0):
            length = int(round(4096 / scale))
            centre, bandwidth, _ = td.estimate_band(
                _multitone(length, centre=0.078125, scale=scale)
            )
            estimates[scale] = (centre / scale, bandwidth / scale)
        for scaled in estimates.values():
            self.assertLess(abs(scaled[0] - estimates[1.0][0]), 3e-6)
            self.assertLess(abs(scaled[1] - estimates[1.0][1]), 2.5e-3)

    def test_end_to_end_frontend_executes_when_transform_apis_are_forbidden(self):
        raw = _multitone(2048)
        forbidden = AssertionError("frequency transform must not be called")
        with mock.patch("numpy.fft.fft", side_effect=forbidden), mock.patch(
            "numpy.fft.fftshift", side_effect=forbidden
        ), mock.patch(
            "preprocess.welch_psd", side_effect=forbidden
        ), mock.patch(
            "preprocess.estimate_band", side_effect=forbidden
        ):
            channels, features, context = tdp.preprocess(raw)
        self.assertEqual(channels.shape, (2, 1024))
        self.assertEqual(features.shape, (12,))
        self.assertTrue(np.isfinite(channels).all())
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(context["version"], tdp.PREPROCESS_VERSION)
        self.assertEqual(context["estimator_version"], td.ESTIMATOR_VERSION)
        self.assertFalse(context["uses_frequency_transform"])
        self.assertIsNone(context["nfft"])
        json.dumps(context, allow_nan=False)

    def test_frontend_preserves_phase_equivariance(self):
        raw = _multitone(2048)
        rotation = np.exp(1j * 0.481)
        channels_a, features_a, context_a = tdp.preprocess(raw)
        channels_b, features_b, context_b = tdp.preprocess(rotation * raw)
        np.testing.assert_allclose(
            _complex_patches(channels_b),
            rotation * _complex_patches(channels_a),
            rtol=0.0,
            atol=3e-6,
        )
        np.testing.assert_allclose(features_b, features_a, rtol=0.0, atol=3e-5)
        self.assertAlmostEqual(context_a["center"], context_b["center"], places=14)
        self.assertAlmostEqual(context_a["bw"], context_b["bw"], places=14)

    def test_deterministic_byte_for_byte(self):
        raw = _multitone(1733)
        first = tdp.preprocess(raw)
        second = tdp.preprocess(raw)
        self.assertTrue(np.array_equal(first[0], second[0]))
        self.assertTrue(np.array_equal(first[1], second[1]))
        self.assertEqual(first[2], second[2])

    def test_invalid_capture_and_configuration_fail_closed(self):
        invalid = (
            np.array([], dtype=np.complex128),
            np.ones(2, dtype=np.complex128),
            np.ones((2, 3), dtype=np.complex128),
            np.ones(8, dtype=np.float64),
            np.array([1.0 + 0.0j, 2.0 + 0.0j, np.nan + 0.0j]),
            np.array([1.0 + 0.0j, 2.0 + 0.0j, np.inf + 0.0j]),
            np.zeros(8, dtype=np.complex128),
        )
        for capture in invalid:
            with self.subTest(shape=capture.shape, dtype=str(capture.dtype)):
                with self.assertRaises((TypeError, ValueError)):
                    td.estimate_band(capture)
                with self.assertRaises((TypeError, ValueError)):
                    tdp.preprocess(capture)

        raw = _multitone(64)
        for value in (0.0, -0.1, 1.1, np.nan, np.inf, "bad"):
            with self.subTest(min_bandwidth=value):
                with self.assertRaises(ValueError):
                    td.estimate_band(raw, min_bandwidth=value)
        with self.assertRaisesRegex(ValueError, "may not exceed"):
            td.estimate_band(raw, min_bandwidth=0.8, full_bandwidth=0.7)

    def test_estimator_source_has_no_transform_call(self):
        source = inspect.getsource(td)
        # Documentation may discuss the excluded methods; executable member
        # access is what must never appear in the estimator implementation.
        self.assertNotIn("np.fft.", source)
        self.assertNotIn("numpy.fft.", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
