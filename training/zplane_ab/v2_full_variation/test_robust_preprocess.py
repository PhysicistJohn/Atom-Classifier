"""Focused synthetic tests for the training-only circular front end."""
from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import robust_preprocess as rp  # noqa: E402


def circular_error(a: float, b: float) -> float:
    return abs((a - b + 0.5) % 1.0 - 0.5)


def bandlimited_noise(
    length: int,
    center: float,
    bw: float,
    seed: int,
    snr_db: float = 20.0,
) -> np.ndarray:
    """Stationary complex noise occupying one known circular FFT interval."""
    rng = np.random.default_rng(seed)
    freq = np.fft.fftfreq(length)
    distance = np.abs((freq - center + 0.5) % 1.0 - 0.5)
    occupied = distance <= bw / 2.0
    spectrum = np.zeros(length, dtype=np.complex128)
    spectrum[occupied] = (
        rng.standard_normal(int(occupied.sum()))
        + 1.0j * rng.standard_normal(int(occupied.sum()))
    )
    signal = np.fft.ifft(spectrum) * np.sqrt(length / int(occupied.sum()))
    noise_scale = np.sqrt(np.mean(np.abs(signal) ** 2) / 2.0) * 10 ** (-snr_db / 20.0)
    noise = noise_scale * (
        rng.standard_normal(length) + 1.0j * rng.standard_normal(length)
    )
    return signal + noise


class CircularBandEstimatorTest(unittest.TestCase):
    def test_wraparound_bands_have_correct_circular_center_and_width(self):
        cases = [
            (0.38, 0.30),
            (-0.40, 0.25),
            (0.45, 0.10),
            (-0.47, 0.08),
            (0.47, 0.70),
        ]
        for index, (center, bw) in enumerate(cases):
            with self.subTest(center=center, bw=bw):
                iq = bandlimited_noise(
                    16384,
                    center,
                    bw,
                    seed=100 + index,
                    snr_db=6.0 if bw > 0.5 else 20.0,
                )
                got_center, got_bw = rp.estimate_band(iq)
                self.assertLessEqual(circular_error(got_center, center), 3.0 / rp.NFFT)
                self.assertLessEqual(abs(got_bw - bw), 8.0 / rp.NFFT)

    def test_non_wrapping_band_remains_accurate(self):
        center, bw = 0.20, 0.30
        got_center, got_bw = rp.estimate_band(
            bandlimited_noise(16384, center, bw, seed=200)
        )
        self.assertLessEqual(circular_error(got_center, center), 3.0 / rp.NFFT)
        self.assertLessEqual(abs(got_bw - bw), 8.0 / rp.NFFT)

    def test_estimator_and_output_are_invariant_to_global_gain(self):
        base = bandlimited_noise(4096, 0.38, 0.22, seed=300)
        reference, reference_ctx = rp.preprocess(base)
        for gain in (1e-200, 1e-12, 1.0, 1e12, 1e200):
            with self.subTest(gain=gain):
                got, ctx = rp.preprocess(base * gain)
                self.assertEqual(ctx["center"], reference_ctx["center"])
                self.assertEqual(ctx["bw"], reference_ctx["bw"])
                self.assertEqual(ctx["resample_frac"], reference_ctx["resample_frac"])
                np.testing.assert_allclose(got, reference, rtol=0.0, atol=3e-6)

    def test_zero_and_scaled_white_noise_use_scale_invariant_fallback(self):
        zero, zero_ctx = rp.preprocess(np.zeros(4096, dtype=np.complex128))
        self.assertTrue(np.array_equal(zero, np.zeros_like(zero)))
        self.assertEqual(zero_ctx["center"], 0.0)
        self.assertEqual(zero_ctx["bw"], rp.FULL_BAND_BW)

        for length in (512, 1024, 4096, 16384):
            rng = np.random.default_rng(400 + length)
            noise = rng.standard_normal(length) + 1.0j * rng.standard_normal(length)
            reference = rp.estimate_band(noise)
            self.assertEqual(reference, (0.0, rp.FULL_BAND_BW))
            for gain in (1e-200, 1.0, 1e200):
                self.assertEqual(rp.estimate_band(noise * gain), reference)

    def test_pair_contract_reuses_center_bandwidth_and_jitter_draw(self):
        rng = np.random.default_rng(500)
        impaired = bandlimited_noise(4096, -0.41, 0.18, seed=501)
        clean = impaired + 0.01 * (
            rng.standard_normal(len(impaired)) + 1.0j * rng.standard_normal(len(impaired))
        )

        _, primary_ctx = rp.preprocess(
            impaired,
            scale_jitter=0.08,
            rng=np.random.default_rng(502),
        )
        with mock.patch.object(
            rp,
            "estimate_band",
            side_effect=AssertionError("forced pair must not redetect its band"),
        ):
            paired, paired_ctx = rp.preprocess(
                clean,
                force_center=primary_ctx["center"],
                force_bw=primary_ctx["bw"],
                force_resample_frac=primary_ctx["resample_frac"],
            )

        self.assertEqual(paired_ctx["center"], primary_ctx["center"])
        self.assertEqual(paired_ctx["bw"], primary_ctx["bw"])
        self.assertEqual(paired_ctx["resample_frac"], primary_ctx["resample_frac"])
        self.assertEqual(paired.shape, (rp.L_OUT,))

    def test_native_mode_skips_resampling_and_remains_gain_invariant(self):
        base = bandlimited_noise(4096, 0.12, 0.15, seed=600)
        reference, ctx = rp.preprocess(base, l_out=len(base), resample=False)
        self.assertFalse(ctx["resampled"])
        self.assertIsNone(ctx["resample_frac"])
        got, got_ctx = rp.preprocess(base * 1e-12, l_out=len(base), resample=False)
        self.assertEqual(got_ctx, ctx)
        np.testing.assert_allclose(got, reference, rtol=0.0, atol=3e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
