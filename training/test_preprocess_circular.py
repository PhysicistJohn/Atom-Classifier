"""Focused production hybrid-v3 preprocessing tests."""
from __future__ import annotations

import json
import os
import unittest
from unittest import mock

import numpy as np

import magnitude as mag
import preprocess as pp


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(
    ROOT,
    "src",
    "embedding",
    "test-fixtures",
    "iq-preprocess-hybrid-v3.json",
)


class HybridProductionPreprocessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(FIXTURE) as f:
            cls.fixture = json.load(f)

    def test_fixture_declares_current_contract(self):
        self.assertEqual(self.fixture["preprocess_version"], pp.PREPROCESS_VERSION)
        self.assertEqual(self.fixture["parameters"], pp.preprocess_metadata())

    def test_fixture_reproduces_every_gain(self):
        for case in self.fixture["cases"]:
            base = np.asarray(case["input_i"]) + 1.0j * np.asarray(case["input_q"])
            expected = (
                np.asarray(case["expected_i"], dtype=np.float32)
                + 1.0j * np.asarray(case["expected_q"], dtype=np.float32)
            )
            for gain in case["gains"]:
                with self.subTest(case=case["name"], gain=gain):
                    actual, context = pp.preprocess(base * gain)
                    self.assertEqual(context["center"], case["center"])
                    self.assertEqual(context["bw"], case["bw"])
                    self.assertEqual(context["resample_frac"], case["resample_frac"])
                    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=3e-6)
                    shape, features = mag.magnitude_from_iq(base * gain)
                    np.testing.assert_allclose(
                        shape,
                        case["magnitude_shape"],
                        rtol=0.0,
                        atol=3e-6,
                    )
                    np.testing.assert_allclose(
                        features,
                        case["magnitude_features"],
                        rtol=0.0,
                        atol=3e-6,
                    )

    def test_candidate_routing_is_explicit_and_reproducible(self):
        for case in self.fixture["cases"]:
            base = np.asarray(case["input_i"]) + 1.0j * np.asarray(case["input_q"])
            scaled, peak = pp._peak_normalized(base)
            if peak == 0.0:
                linear = circular = (0.0, pp.FULL_BAND_BW)
            else:
                linear = pp.estimate_band_linear_v1(scaled)
                circular = pp.estimate_band_circular_v2(scaled)
            if circular == (0.0, pp.FULL_BAND_BW):
                expected_route = "full-band-fallback"
            elif pp._hybrid_uses_circular(linear, circular, pp.NFFT):
                expected_route = "circular-v2"
            else:
                expected_route = "linear-v1"
            with self.subTest(case=case["name"]):
                self.assertEqual(linear, (case["linear_center"], case["linear_bw"]))
                self.assertEqual(
                    circular,
                    (case["circular_center"], case["circular_bw"]),
                )
                self.assertEqual(expected_route, case["route"])

    def test_known_bands_remain_compact(self):
        for case in self.fixture["cases"]:
            if case["true_center"] is None:
                continue
            with self.subTest(case=case["name"]):
                center_error = abs(
                    (case["center"] - case["true_center"] + 0.5) % 1.0 - 0.5
                )
                self.assertLessEqual(center_error, 4.0 / pp.NFFT)
                self.assertLessEqual(
                    abs(case["bw"] - case["true_bw"]),
                    10.0 / pp.NFFT,
                )

    def test_zero_and_white_noise_have_scale_invariant_fallback(self):
        for name in ("white-noise", "exact-zero"):
            case = next(c for c in self.fixture["cases"] if c["name"] == name)
            self.assertEqual(case["center"], 0.0)
            self.assertEqual(case["bw"], pp.FULL_BAND_BW)

    def test_normal_band_preserves_linear_v1_geometry_and_waveform(self):
        case = next(c for c in self.fixture["cases"] if c["name"] == "normal-band")
        base = np.asarray(case["input_i"]) + 1.0j * np.asarray(case["input_q"])
        hybrid, context = pp.preprocess(base)
        center, bw = pp.estimate_band_linear_v1(base)
        self.assertEqual(case["route"], "linear-v1")
        self.assertEqual((context["center"], context["bw"]), (center, bw))

        # Frozen linear-v1 pipeline reference. Hybrid-v3 uses scale-stable
        # normalization, so floating-point output may differ by a few ulps while
        # preserving the exact detector/resampling geometry.
        n = np.arange(len(base))
        legacy = base * np.exp(-1j * 2.0 * np.pi * center * n)
        new_len = max(64, round(len(base) * bw / pp.TARGET_FRAC))
        legacy = pp.center_fit(pp.lin_resample(legacy, new_len), pp.L_OUT)
        legacy = (
            legacy / np.sqrt(np.mean(np.abs(legacy) ** 2) + 1e-12)
        ).astype(np.complex64)
        np.testing.assert_allclose(hybrid, legacy, rtol=0.0, atol=3e-6)

    def test_forced_pair_contract_bypasses_redetection(self):
        case = self.fixture["cases"][0]
        iq = np.asarray(case["input_i"]) + 1.0j * np.asarray(case["input_q"])
        _, context = pp.preprocess(
            iq,
            scale_jitter=0.08,
            rng=np.random.default_rng(44),
        )
        with mock.patch.object(
            pp,
            "estimate_band",
            side_effect=AssertionError("forced pair must not redetect"),
        ):
            paired, paired_context = pp.preprocess(
                iq,
                force_center=context["center"],
                force_bw=context["bw"],
                force_resample_frac=context["resample_frac"],
            )
        self.assertEqual(paired.shape, (pp.L_OUT,))
        self.assertEqual(paired_context, context)


if __name__ == "__main__":
    unittest.main(verbosity=2)
