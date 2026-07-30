from __future__ import annotations

import math
import unittest

import numpy as np

import fit_current_source_display_calibration as calibration


def _binary_rows() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Two correct-margin rows and one incorrect-margin row have NLL optimum
    # exp(scale)=2, hence scale=log(2).
    distances = np.full((3, 7), np.inf, dtype=np.float64)
    distances[0, :2] = [0.0, 1.0]
    distances[1, :2] = [0.0, 1.0]
    distances[2, :2] = [1.0, 0.0]
    truth = np.zeros(3, dtype=np.int64)
    support = np.asarray(
        [True, True, False, False, False, False, False], dtype=bool
    )
    return distances, truth, support


class DisplayCalibrationTests(unittest.TestCase):
    def test_bisection_reproduces_known_convex_nll_root(self) -> None:
        distances, truth, support = _binary_rows()
        scale, detail = calibration.fit_positive_distance_logit_scale(
            distances, truth, support
        )
        self.assertAlmostEqual(scale, math.log(2.0), places=14)
        self.assertLess(abs(detail["derivative_at_scale"]), 1e-15)
        before = calibration.display_metrics(
            distances, truth, support, 1.0
        )
        after = calibration.display_metrics(
            distances, truth, support, scale
        )
        self.assertLess(after["nll"], before["nll"])
        self.assertEqual(
            before["prediction_sha256"], after["prediction_sha256"]
        )

    def test_profile_balance_is_invariant_to_unequal_profile_duplication(
        self,
    ) -> None:
        _base, _truth, support = _binary_rows()
        distances = np.full((5, 7), np.inf, dtype=np.float64)
        distances[:, :2] = np.asarray(
            [
                [0.0, 1.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        )
        truth = np.zeros(5, dtype=np.int64)
        groups = np.asarray(
            ["profile-a", "profile-a", "profile-a", "profile-b", "profile-b"]
        )
        balanced = calibration.profile_balanced_weights(groups)
        balanced_root, _ = calibration.fit_positive_distance_logit_scale(
            distances,
            truth,
            support,
            sample_weight=balanced,
        )

        # Duplicate only profile A. View weighting changes, while equal-profile
        # weighting assigns that profile the same total mass as before.
        duplicated_distance = np.concatenate(
            (distances[:3], distances[:3], distances[3:]), axis=0
        )
        duplicated_truth = np.zeros(8, dtype=np.int64)
        duplicated_groups = np.asarray(
            [
                "profile-a",
                "profile-a",
                "profile-a",
                "profile-a",
                "profile-a",
                "profile-a",
                "profile-b",
                "profile-b",
            ]
        )
        duplicated_balanced_root, _ = (
            calibration.fit_positive_distance_logit_scale(
                duplicated_distance,
                duplicated_truth,
                support,
                sample_weight=calibration.profile_balanced_weights(
                    duplicated_groups
                ),
            )
        )
        original_view_root, _ = (
            calibration.fit_positive_distance_logit_scale(
                distances, truth, support
            )
        )
        duplicated_view_root, _ = (
            calibration.fit_positive_distance_logit_scale(
                duplicated_distance, duplicated_truth, support
            )
        )
        self.assertAlmostEqual(
            duplicated_balanced_root, balanced_root, places=14
        )
        self.assertGreater(
            abs(duplicated_view_root - original_view_root), 0.1
        )

    def test_unsupported_truth_and_nonpositive_weights_fail_closed(self) -> None:
        distances, truth, support = _binary_rows()
        bad_truth = truth.copy()
        bad_truth[0] = 6
        with self.assertRaisesRegex(ValueError, "invalid"):
            calibration.fit_positive_distance_logit_scale(
                distances, bad_truth, support
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            calibration.fit_positive_distance_logit_scale(
                distances,
                truth,
                support,
                sample_weight=np.asarray([1.0, 0.0, 1.0]),
            )


if __name__ == "__main__":
    unittest.main()
