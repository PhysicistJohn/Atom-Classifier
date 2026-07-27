"""Regression tests for clean/impaired reconstruction-target carrier handling.

Run directly:
    .venv-training/bin/python \
        training/zplane_ab/v2_full_variation/test_pair_target_alignment.py

These tests intentionally use a synthetic pair with known carrier frequencies.
They distinguish three operations that had been conflated in the target-builder
commentary:

* applying one common downconversion to both pair members;
* independently removing each member's known carrier; and
* rotating the clean target onto the impaired member's carrier coordinate.
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import native_preprocess as npp  # noqa: E402
from denoise_eval import best_freq_offset, coherence, derotate_onto  # noqa: E402


class PairTargetAlignmentTest(unittest.TestCase):
    LENGTH = 4096
    FINE_NFFT = 1 << 16
    CLEAN_CENTER = 0.081
    RELATIVE_CFO = 37 / FINE_NFFT

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(20260727)
        n = np.arange(cls.LENGTH)
        symbols = rng.choice(
            np.array([1.0, -1.0, 1.0j, -1.0j], dtype=np.complex128),
            size=(cls.LENGTH + 7) // 8,
        )
        payload = np.repeat(symbols, 8)[: cls.LENGTH]
        payload *= 0.75 + 0.25 * np.cos(2.0 * np.pi * n / 257.0)

        cls.clean = payload * np.exp(2.0j * np.pi * cls.CLEAN_CENTER * n)
        cls.impaired = (
            1.7
            * np.exp(0.43j)
            * cls.clean
            * np.exp(2.0j * np.pi * cls.RELATIVE_CFO * n)
        )

    @staticmethod
    def _coherence(a: np.ndarray, b: np.ndarray) -> float:
        return float(coherence(a[None, :], b[None, :])[0])

    def test_common_center_cannot_change_pair_coherence(self):
        """A shared coarse center is common-mode, not a source of relative CFO."""
        raw = self._coherence(self.clean, self.impaired)
        arbitrary_coarse_center = 0.173
        clean_common, _ = npp.preprocess(
            self.clean,
            l_out=self.LENGTH,
            force_center=arbitrary_coarse_center,
        )
        impaired_common, _ = npp.preprocess(
            self.impaired,
            l_out=self.LENGTH,
            force_center=arbitrary_coarse_center,
        )
        common_centered = self._coherence(clean_common, impaired_common)

        self.assertLess(raw, 0.02)
        self.assertAlmostEqual(common_centered, raw, places=7)

    def test_derotate_onto_tracks_the_known_relative_cfo(self):
        """derotate_onto rotates the clean target to the impaired coordinate."""
        common_center = 0.173
        clean_common, _ = npp.preprocess(
            self.clean, l_out=self.LENGTH, force_center=common_center
        )
        impaired_common, _ = npp.preprocess(
            self.impaired, l_out=self.LENGTH, force_center=common_center
        )

        aligned, estimated_cfo, aligned_coherence = best_freq_offset(
            clean_common,
            impaired_common,
            n_fine=self.FINE_NFFT,
        )
        via_public_helper = derotate_onto(
            clean_common,
            impaired_common,
            n_fine=self.FINE_NFFT,
        )

        self.assertAlmostEqual(estimated_cfo, self.RELATIVE_CFO, places=12)
        self.assertGreater(aligned_coherence, 0.9999)
        np.testing.assert_allclose(via_public_helper, aligned, rtol=0.0, atol=1e-6)

    def test_independent_known_centers_make_a_baseband_clean_target(self):
        """Removing each known carrier recovers the clean baseband coordinate."""
        clean_baseband, _ = npp.preprocess(
            self.clean,
            l_out=self.LENGTH,
            force_center=self.CLEAN_CENTER,
        )
        impaired_baseband, _ = npp.preprocess(
            self.impaired,
            l_out=self.LENGTH,
            force_center=self.CLEAN_CENTER + self.RELATIVE_CFO,
        )

        self.assertGreater(self._coherence(clean_baseband, impaired_baseband), 0.9999)


if __name__ == "__main__":
    unittest.main(verbosity=2)
