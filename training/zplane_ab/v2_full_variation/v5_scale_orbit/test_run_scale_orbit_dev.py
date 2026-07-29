from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_scale_orbit_dev as runner  # noqa: E402
import scale_orbit_data  # noqa: E402


CLASSES = ("am", "bluetooth", "cw", "dsss", "fm", "gsm", "ofdm")
SCALES = (1.0, 1.25, 1.5, 2.0)


def _rows() -> tuple[list[scale_orbit_data.ScaleOrbitRowRef], np.ndarray]:
    rows: list[scale_orbit_data.ScaleOrbitRowRef] = []
    labels: list[int] = []
    profiles = (
        ("bluetooth-le-advertising", "bluetooth", 1),
        ("wifi-hr-dsss-11m", "dsss", 3),
    )
    for profile_index, (profile, class_name, label) in enumerate(profiles):
        pair_id = f"pair-{profile}"
        for scale_index, scale in enumerate(SCALES):
            rows.append(
                scale_orbit_data.ScaleOrbitRowRef(
                    source="current",
                    source_index=len(rows),
                    role="selection",
                    class_name=class_name,
                    label=label,
                    profile_id=profile,
                    valid_sample_count=4096,
                    storage_sample_count=16384,
                    identity=f"scale-orbit:20262904:{pair_id}",
                    content_sha256=f"{len(rows):064x}",
                    zero_padded_after_valid=True,
                    pair_id=pair_id,
                    realization_index=profile_index,
                    physical_scale_factor=scale,
                    sample_rate_hz=int(8_000_000 * scale),
                    native_sample_rate_hz=8_000_000,
                    capture_bandwidth_hz=4_000_000,
                    population_seed=20262904,
                    replay="cyclic",
                    phase_native_sample=profile_index,
                    receiver_channel_seed=100 + profile_index,
                    receiver_seed=200 + scale_index,
                )
            )
            labels.append(label)
    return rows, np.asarray(labels, dtype=np.int64)


class ScaleOrbitRunnerTests(unittest.TestCase):
    def test_checkpoint_score_is_scale_sensitive_and_fixed_width(self):
        common = {
            "historical_balanced": 0.98,
            "current_worst_scale_present_class_balanced": 0.96,
            "current_worst_profile_scale_recall": 0.75,
            "current_physical_scale_prediction_agreement": 0.97,
            "current_worst_directional_confusion": 0.01,
            "current_pooled_profile_balanced": 0.96,
            "current_pooled_accuracy": 0.97,
            "combined_balanced": 0.97,
        }
        good = runner.checkpoint_score(
            **common,
            current_wifi_hr_dsss_worst_scale_accuracy=0.98,
        )
        bad = runner.checkpoint_score(
            **common,
            current_wifi_hr_dsss_worst_scale_accuracy=0.25,
        )
        self.assertEqual(len(good), runner.CHECKPOINT_SCORE_TERMS)
        self.assertGreater(good, bad)
        self.assertEqual(good[0], 0.75)
        with self.assertRaises(ValueError):
            runner.checkpoint_score(
                **common,
                current_wifi_hr_dsss_worst_scale_accuracy=float("nan"),
            )

    def test_scale_report_exposes_exact_pair_and_directional_metrics(self):
        rows, labels = _rows()
        perfect = runner._classification_report(
            labels,
            labels,
            CLASSES,
            rows,
        )
        self.assertTrue(perfect["exact_scale_pair_coverage"])
        self.assertEqual(perfect["scale_pair_count"], 2)
        self.assertEqual(
            perfect["expected_physical_scale_factors"],
            list(SCALES),
        )
        self.assertEqual(
            perfect["worst_scale_present_class_balanced_accuracy"],
            1.0,
        )
        self.assertEqual(perfect["worst_profile_scale_recall"], 1.0)
        self.assertEqual(
            perfect["wifi_hr_dsss_worst_scale_accuracy"],
            1.0,
        )
        self.assertEqual(
            perfect["physical_scale_prediction_agreement_rate"],
            1.0,
        )
        self.assertEqual(
            perfect["worst_directional_dsss_bluetooth_confusion_rate"],
            0.0,
        )

        confused = labels.copy()
        wifi_scale_125 = next(
            index
            for index, row in enumerate(rows)
            if (
                row.profile_id == "wifi-hr-dsss-11m"
                and row.physical_scale_factor == 1.25
            )
        )
        confused[wifi_scale_125] = CLASSES.index("bluetooth")
        report = runner._classification_report(
            confused,
            labels,
            CLASSES,
            rows,
        )
        self.assertEqual(report["wifi_hr_dsss_worst_scale_accuracy"], 0.0)
        self.assertEqual(
            report[
                "wifi_hr_dsss_to_bluetooth_worst_scale_confusion_rate"
            ],
            1.0,
        )
        self.assertEqual(
            report["physical_scale_prediction_agreement_rate"],
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
