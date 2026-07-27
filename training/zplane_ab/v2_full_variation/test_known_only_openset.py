from __future__ import annotations

import os
import sys
import unittest

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from known_only_openset import (  # noqa: E402
    ClassRadiusOpenSet,
    known_quantile_threshold,
)


class ClassRadiusOpenSetTests(unittest.TestCase):
    def setUp(self):
        # Class 0 is deliberately tight and class 1 broad. A global raw radius
        # confuses their natural scales; the conditional score should not.
        self.enrollment = np.array(
            [
                [-1.05, 0.00],
                [-0.95, 0.00],
                [0.50, 0.00],
                [1.50, 0.00],
            ],
            dtype=np.float32,
        )
        self.enrollment_y = np.array([0, 0, 1, 1])
        self.calibration = np.array(
            [
                [-1.04, 0.00],
                [-1.02, 0.01],
                [-0.98, -0.01],
                [-0.96, 0.00],
                [0.40, 0.00],
                [0.70, 0.20],
                [1.30, -0.20],
                [1.60, 0.00],
            ],
            dtype=np.float32,
        )
        self.calibration_y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        self.model = ClassRadiusOpenSet.fit(
            self.enrollment,
            self.enrollment_y,
            self.calibration,
            self.calibration_y,
        )

    def test_fit_uses_ordinary_enrollment_means(self):
        expected = np.stack(
            [
                self.enrollment[self.enrollment_y == class_index].mean(0)
                for class_index in range(2)
            ]
        )
        np.testing.assert_array_equal(self.model.prototypes, expected)

    def test_prediction_is_exact_nearest_prototype(self):
        query = np.array([[-1.00, 0.02], [0.90, 0.00]], dtype=np.float32)
        distance = ((query[:, None] - self.model.prototypes[None]) ** 2).sum(-1)
        np.testing.assert_array_equal(self.model.predict(query), distance.argmin(1))
        predicted, score = self.model.predict_and_score(query)
        np.testing.assert_array_equal(predicted, distance.argmin(1))
        self.assertEqual(score.shape, (2,))

    def test_score_is_monotonic_within_each_class_and_bounded(self):
        query = np.array(
            [
                [-1.00, 0.00],
                [-1.10, 0.00],
                [-1.30, 0.00],
                [1.00, 0.00],
                [1.70, 0.00],
            ],
            dtype=np.float32,
        )
        predicted = np.array([0, 0, 0, 1, 1])
        score = self.model.score(query, predicted)
        self.assertTrue(np.all((score >= 0.0) & (score < 1.0)))
        self.assertLessEqual(score[0], score[1])
        self.assertLessEqual(score[1], score[2])
        self.assertLessEqual(score[3], score[4])

    def test_class_conditioning_corrects_different_natural_radii(self):
        tight_outlier = np.array([[-1.20, 0.00]], dtype=np.float32)
        broad_known = np.array([[1.30, 0.00]], dtype=np.float32)
        raw_tight = float(
            ((tight_outlier[0] - self.model.prototypes[0]) ** 2).sum()
        )
        raw_broad = float(
            ((broad_known[0] - self.model.prototypes[1]) ** 2).sum()
        )
        self.assertLess(raw_tight, raw_broad)
        scores = self.model.score(
            np.concatenate([tight_outlier, broad_known]),
            np.array([0, 1]),
        )
        self.assertGreater(scores[0], scores[1])

    def test_ties_are_conservative(self):
        calibration_row = self.calibration[[0]]
        score = self.model.score(calibration_row, np.array([0]))[0]
        distance = float(
            ((calibration_row[0] - self.model.prototypes[0]) ** 2).sum()
        )
        calibration = self.model.calibration_distances[0]
        expected = np.searchsorted(calibration, distance, side="left") / (
            len(calibration) + 1.0
        )
        self.assertEqual(score, expected)

    def test_known_threshold_and_validation(self):
        self.assertEqual(
            known_quantile_threshold(
                np.array([0.0, 0.1, 0.2, 0.8]),
                false_unknown_rate=0.25,
            ),
            0.8,
        )
        with self.assertRaisesRegex(ValueError, "absent"):
            ClassRadiusOpenSet.fit(
                self.enrollment,
                self.enrollment_y,
                self.calibration[:4],
                self.calibration_y[:4],
            )
        with self.assertRaisesRegex(ValueError, "dimension"):
            self.model.score(np.ones((2, 3), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "false_unknown_rate"):
            known_quantile_threshold([0.1, 0.2], false_unknown_rate=0.0)


if __name__ == "__main__":
    unittest.main()
