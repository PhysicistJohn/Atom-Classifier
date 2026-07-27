from __future__ import annotations

import os
import sys
import unittest

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from known_only_patch_openset import (  # noqa: E402
    KnownOnlyLOFOpenSet,
    KnownOnlyPatchOpenSet,
    patch_statistics,
)


class PatchStatisticsTest(unittest.TestCase):
    def test_phase_and_patch_order_invariance(self):
        rng = np.random.default_rng(4)
        packed = rng.standard_normal((5, 2, 16 * 64)).astype(np.float32)
        base = patch_statistics(packed)

        z = packed[:, 0] + 1j * packed[:, 1]
        z *= np.exp(1j * 1.234)
        rotated = np.stack((z.real, z.imag), axis=1).astype(np.float32)
        np.testing.assert_allclose(
            patch_statistics(rotated), base, atol=2e-6, rtol=2e-6
        )

        patches = packed.reshape(5, 2, 16, 64)
        permuted = patches[:, :, rng.permutation(16)].reshape(packed.shape)
        np.testing.assert_allclose(
            patch_statistics(permuted), base, atol=1e-6, rtol=1e-6
        )

    def test_rejects_nonfinite_and_bad_geometry(self):
        with self.assertRaises(ValueError):
            patch_statistics(np.zeros((2, 2, 100), np.float32))
        value = np.zeros((2, 2, 16 * 64), np.float32)
        value[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            patch_statistics(value)


class KnownOnlyPatchOpenSetTest(unittest.TestCase):
    def _fixture(self):
        rng = np.random.default_rng(7)
        n_train, n_enroll = 18, 12
        train_y = np.repeat(np.arange(3), n_train // 3)
        enroll_y = np.repeat(np.arange(3), n_enroll // 3)
        train_e = rng.standard_normal((n_train, 6))
        enroll_e = rng.standard_normal((n_enroll, 6))
        train_x = rng.standard_normal((n_train, 2, 16 * 64)).astype(np.float32)
        enroll_x = rng.standard_normal((n_enroll, 2, 16 * 64)).astype(np.float32)
        return train_e, train_x, train_y, enroll_e, enroll_x, enroll_y

    def test_fit_and_score_are_finite_and_bounded(self):
        values = self._fixture()
        fitted = KnownOnlyPatchOpenSet.fit(*values)
        embeddings, packed, predicted = values[3], values[4], values[5]
        score = fitted.score(embeddings, packed, predicted)
        self.assertEqual(score.shape, (len(embeddings),))
        self.assertTrue(np.isfinite(score).all())
        self.assertTrue(np.all((score >= 0.0) & (score < 1.0)))
        self.assertEqual(fitted.calibration_scores.shape, score.shape)

    def test_predictions_are_inputs_not_outputs(self):
        values = self._fixture()
        fitted = KnownOnlyPatchOpenSet.fit(*values)
        embeddings, packed, predicted = values[3], values[4], values[5]
        score_a = fitted.score(embeddings, packed, predicted)
        changed = (predicted + 1) % 3
        score_b = fitted.score(embeddings, packed, changed)
        self.assertEqual(score_a.shape, score_b.shape)
        self.assertTrue(np.any(score_a != score_b))

    def test_fit_is_deterministic(self):
        values = self._fixture()
        first = KnownOnlyPatchOpenSet.fit(*values)
        second = KnownOnlyPatchOpenSet.fit(*values)
        score_first = first.score(values[3], values[4], values[5])
        score_second = second.score(values[3], values[4], values[5])
        np.testing.assert_array_equal(score_first, score_second)

    def test_input_validation(self):
        values = self._fixture()
        with self.assertRaises(ValueError):
            KnownOnlyPatchOpenSet.fit(*values, shrinkage=-0.1)
        fitted = KnownOnlyPatchOpenSet.fit(*values)
        with self.assertRaises(ValueError):
            fitted.score(values[3], values[4][:-1], values[5])


class KnownOnlyLOFOpenSetTest(unittest.TestCase):
    def test_score_is_bounded_deterministic_and_separate_from_labels(self):
        rng = np.random.default_rng(13)
        training = rng.standard_normal((30, 5))
        enrollment = rng.standard_normal((16, 5))
        query = rng.standard_normal((8, 5))
        first = KnownOnlyLOFOpenSet.fit(training, enrollment, neighbors=5)
        second = KnownOnlyLOFOpenSet.fit(training, enrollment, neighbors=5)
        score = first.score(query)
        np.testing.assert_array_equal(score, second.score(query))
        self.assertTrue(np.isfinite(score).all())
        self.assertTrue(np.all((score >= 0.0) & (score < 1.0)))
        self.assertFalse(hasattr(first, "predict"))

    def test_validation(self):
        rng = np.random.default_rng(17)
        training = rng.standard_normal((12, 4))
        enrollment = rng.standard_normal((8, 4))
        with self.assertRaises(ValueError):
            KnownOnlyLOFOpenSet.fit(training, enrollment, neighbors=12)
        fitted = KnownOnlyLOFOpenSet.fit(training, enrollment)
        with self.assertRaises(ValueError):
            fitted.score(rng.standard_normal((2, 3)))


if __name__ == "__main__":
    unittest.main()
