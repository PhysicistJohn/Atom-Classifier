"""Tests for the isolated invariant-patch preprocessing contract."""
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

import invariant_patch_preprocess as ip  # noqa: E402
import preprocess as pp  # noqa: E402


def _signal(length: int = 4096) -> np.ndarray:
    n = np.arange(length, dtype=np.float64)
    envelope = 1.0 + 0.24 * np.cos(2.0 * np.pi * 0.0031 * n)
    phase = 2.0 * np.pi * 0.137 * n + 0.31 * np.sin(2.0 * np.pi * 0.007 * n)
    return (envelope * np.exp(1j * phase)).astype(np.complex128)


def _complex_patches(channels: np.ndarray, patch_count: int, patch_length: int) -> np.ndarray:
    packed = channels[0].astype(np.float64) + 1j * channels[1].astype(np.float64)
    return packed.reshape(patch_count, patch_length)


class InvariantPatchPreprocessTest(unittest.TestCase):
    def test_stable_api_and_default_output_contract(self):
        signature = inspect.signature(ip.preprocess)
        self.assertEqual(signature.parameters["patch_length"].default, 64)
        self.assertEqual(signature.parameters["patch_count"].default, 16)
        self.assertEqual(signature.parameters["target_frac"].default, pp.TARGET_FRAC)
        self.assertEqual(signature.parameters["nfft"].default, pp.NFFT)

        channels, features, context = ip.preprocess(_signal())
        self.assertEqual(channels.shape, (2, 1024))
        self.assertEqual(channels.dtype, np.float32)
        self.assertEqual(features.shape, (pp.N_FEATURES,))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.isfinite(channels).all())
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(context["version"], ip.PREPROCESS_VERSION)
        self.assertEqual(context["estimator_version"], "hybrid-v3")
        self.assertEqual(context["packed_length"], 1024)
        self.assertEqual(context["padding_samples"], 0)
        self.assertEqual(len(context["patch_starts"]), 16)
        self.assertEqual(context["center"], context["detected_center"])
        self.assertEqual(context["bw"], context["detected_bw"])
        self.assertEqual(context["resample_frac"], context["bw"])
        json.dumps(context, allow_nan=False)

    def test_positive_gain_invariance_across_extreme_finite_scales(self):
        base = _signal()
        expected_channels, expected_features, expected_context = ip.preprocess(base)
        for gain in (1e-200, 1e-100, 1e-12, 1e12, 1e100, 1e200):
            with self.subTest(gain=gain):
                channels, features, context = ip.preprocess(base * gain)
                np.testing.assert_allclose(
                    channels, expected_channels, rtol=0.0, atol=3e-6
                )
                np.testing.assert_allclose(
                    features, expected_features, rtol=0.0, atol=3e-5
                )
                self.assertEqual(context["center"], expected_context["center"])
                self.assertEqual(context["bw"], expected_context["bw"])
                self.assertEqual(
                    context["resample_frac"], expected_context["resample_frac"]
                )
                self.assertEqual(
                    context["patch_starts"], expected_context["patch_starts"]
                )

    def test_global_phase_equivariance_and_feature_invariance(self):
        base = _signal()
        rotation = np.exp(1j * 0.731)
        channels_a, features_a, context_a = ip.preprocess(base)
        channels_b, features_b, context_b = ip.preprocess(rotation * base)
        patches_a = _complex_patches(channels_a, 16, 64)
        patches_b = _complex_patches(channels_b, 16, 64)
        np.testing.assert_allclose(
            patches_b, rotation * patches_a, rtol=0.0, atol=3e-6
        )
        np.testing.assert_allclose(features_b, features_a, rtol=0.0, atol=3e-5)
        self.assertEqual(context_b["center"], context_a["center"])
        self.assertEqual(context_b["bw"], context_a["bw"])
        self.assertEqual(context_b["patch_starts"], context_a["patch_starts"])

    def test_forced_geometry_scale_equivalence_on_dimensionless_u(self):
        target = 0.5
        u_end = 128.0

        def waveform(u: np.ndarray) -> np.ndarray:
            return (
                1.0
                + 0.17 * np.exp(1j * 2.0 * np.pi * 0.019 * u)
                + 0.11j * (u / u_end)
            )

        bw_a, bw_b = 0.25, 0.125
        u_a = np.arange(257, dtype=np.float64) * bw_a / target
        u_b = np.arange(513, dtype=np.float64) * bw_b / target
        raw_a, raw_b = waveform(u_a), waveform(u_b)

        channels_a, features_a, context_a = ip.preprocess(
            raw_a,
            force_center=0.0,
            force_bw=bw_a,
            force_resample_frac=bw_a,
        )
        channels_b, features_b, context_b = ip.preprocess(
            raw_b,
            force_center=0.0,
            force_bw=bw_b,
            force_resample_frac=bw_b,
        )
        self.assertEqual(context_a["active_length"], 129)
        self.assertEqual(context_b["active_length"], 129)
        self.assertEqual(context_a["active_u_end"], 128)
        self.assertEqual(context_b["active_u_end"], 128)
        self.assertEqual(context_a["patch_starts"], context_b["patch_starts"])
        np.testing.assert_allclose(channels_a, channels_b, rtol=0.0, atol=2e-6)
        np.testing.assert_allclose(features_a, features_b, rtol=0.0, atol=2e-5)

    def test_even_valid_starts_and_no_inserted_padding(self):
        raw = np.full(129, 1.0 + 0.75j, dtype=np.complex128)
        channels, _features, context = ip.preprocess(
            raw,
            force_center=0.0,
            force_bw=pp.TARGET_FRAC,
            force_resample_frac=pp.TARGET_FRAC,
        )
        maximum = 129 - 64
        expected = (
            2 * np.arange(16, dtype=np.int64) * maximum + 15
        ) // 30
        self.assertEqual(context["active_length"], 129)
        self.assertEqual(context["patch_starts"], expected.tolist())
        self.assertEqual(context["patch_starts"][0], 0)
        self.assertEqual(context["patch_starts"][-1], maximum)
        self.assertFalse(context["patches_repeated_from_short_active"])
        self.assertEqual(context["active_repeat_count"], 1)
        self.assertEqual(context["padding_samples"], 0)
        patches = _complex_patches(channels, 16, 64)
        self.assertTrue(np.all(np.abs(patches) > 0.0))

    def test_short_active_sequence_repeats_samples_instead_of_padding(self):
        n = np.arange(17, dtype=np.float64)
        raw = (1.1 + 0.03 * n) + 1j * (0.7 - 0.01 * n)
        channels, features, context = ip.preprocess(
            raw,
            force_center=0.0,
            force_bw=pp.TARGET_FRAC,
            force_resample_frac=pp.TARGET_FRAC,
        )
        peak_normalized, _ = ip._peak_normalize(raw.astype(np.complex128))
        active, _ = ip._resample_on_u(
            peak_normalized,
            resample_frac=pp.TARGET_FRAC,
            target_frac=pp.TARGET_FRAC,
        )
        active, _ = ip._rms_normalize(active)
        expected_patch = np.tile(active, 4)[:64]
        patches = _complex_patches(channels, 16, 64)

        self.assertEqual(context["active_length"], 17)
        self.assertTrue(context["patches_repeated_from_short_active"])
        self.assertEqual(context["active_repeat_count"], 4)
        self.assertEqual(context["patch_starts"], [0] * 16)
        self.assertEqual(context["padding_samples"], 0)
        for patch in patches:
            np.testing.assert_allclose(patch, expected_patch, rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(
            features, pp.iq_features(active), rtol=0.0, atol=1e-6
        )

    def test_deterministic_byte_for_byte(self):
        kwargs = dict(
            patch_length=37,
            patch_count=9,
            force_center=0.137,
            force_bw=0.23,
            force_resample_frac=0.19,
        )
        first = ip.preprocess(_signal(777), **kwargs)
        second = ip.preprocess(_signal(777), **kwargs)
        self.assertTrue(np.array_equal(first[0], second[0]))
        self.assertTrue(np.array_equal(first[1], second[1]))
        self.assertEqual(first[2], second[2])

    def test_forced_geometry_path_has_no_fft_or_center_fit(self):
        raw = _signal(257)
        with mock.patch.object(
            pp, "estimate_band", side_effect=AssertionError("must not estimate")
        ), mock.patch.object(
            pp, "center_fit", side_effect=AssertionError("must not center-fit")
        ), mock.patch.object(
            np.fft, "fft", side_effect=AssertionError("must not FFT after forcing geometry")
        ):
            channels, features, context = ip.preprocess(
                raw,
                force_center=0.137,
                force_bw=0.25,
                force_resample_frac=0.25,
            )
        self.assertEqual(channels.shape, (2, 1024))
        self.assertEqual(features.shape, (12,))
        self.assertIsNone(context["detected_center"])
        self.assertIsNone(context["detected_bw"])

    def test_partial_geometry_override_is_auditable(self):
        _channels, _features, context = ip.preprocess(
            _signal(),
            force_center=-0.23,
        )
        self.assertIsNotNone(context["detected_center"])
        self.assertIsNotNone(context["detected_bw"])
        self.assertAlmostEqual(context["center"], -0.23, places=15)
        self.assertEqual(context["bw"], context["detected_bw"])

    def test_invalid_capture_inputs_raise(self):
        invalid = (
            np.array([], dtype=np.complex128),
            np.array([1.0 + 1.0j]),
            np.ones((2, 3), dtype=np.complex128),
            np.ones(8, dtype=np.float64),
            np.array([1.0 + 1.0j, np.nan + 0.0j]),
            np.array([1.0 + 1.0j, np.inf + 0.0j]),
            np.zeros(32, dtype=np.complex128),
        )
        for capture in invalid:
            with self.subTest(shape=capture.shape, dtype=str(capture.dtype)):
                with self.assertRaises((TypeError, ValueError)):
                    ip.preprocess(capture)

    def test_invalid_configuration_raises(self):
        raw = _signal(256)
        for name in ("patch_length", "patch_count"):
            for value in (0, -1, 1.5, True, "4"):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        ip.preprocess(raw, **{name: value})
        with self.assertRaisesRegex(ValueError, "exceeds safety limit"):
            ip.preprocess(
                raw,
                patch_length=ip.MAX_PACKED_LENGTH,
                patch_count=2,
            )

        for value in (0.0, -0.1, 1.01, np.nan, np.inf, "bad"):
            with self.subTest(target_frac=value):
                with self.assertRaises(ValueError):
                    ip.preprocess(raw, target_frac=value)

        for value in (0, 8, 18, 512.0, True):
            with self.subTest(nfft=value):
                with self.assertRaises(ValueError):
                    ip.preprocess(raw, nfft=value)

        for value in (np.nan, np.inf, "bad"):
            with self.subTest(force_center=value):
                with self.assertRaises(ValueError):
                    ip.preprocess(
                        raw,
                        force_center=value,
                        force_bw=0.2,
                    )

        for name in ("force_bw", "force_resample_frac"):
            for value in (0.0, -0.1, 1.01, np.nan, np.inf, "bad"):
                with self.subTest(name=name, value=value):
                    kwargs = dict(force_center=0.0, force_bw=0.2)
                    kwargs[name] = value
                    with self.assertRaises(ValueError):
                        ip.preprocess(raw, **kwargs)

        with self.assertRaisesRegex(ValueError, "fewer than two samples"):
            ip.preprocess(
                raw[:4],
                force_center=0.0,
                force_bw=0.2,
                force_resample_frac=1e-6,
            )

    def test_estimator_version_drift_fails_closed(self):
        with mock.patch.object(pp, "PREPROCESS_VERSION", "future-v4"):
            with self.assertRaisesRegex(RuntimeError, "requires hybrid-v3"):
                ip.preprocess(_signal(128))


if __name__ == "__main__":
    unittest.main(verbosity=2)
