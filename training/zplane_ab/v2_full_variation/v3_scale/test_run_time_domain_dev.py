from __future__ import annotations

import argparse
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_time_domain_dev as subject


def _item(cls: str, center: float, bandwidth: float) -> dict[str, object]:
    return {
        "cls": cls,
        "centreOffsetFrac": center,
        "bandwidthHz": bandwidth,
        "sampleRateHz": 1.0,
        "impaired": False,
        "snrDb": 20.0,
    }


class TimeDomainDevelopmentAuditTests(unittest.TestCase):
    def test_nonzero_prefix_mask_is_exact_and_fails_short_rows(self) -> None:
        captures = [
            np.zeros(8, dtype=np.complex64),
            np.asarray([0, 0, 0, 1, 0, 0, 0, 0], dtype=np.complex64),
            np.asarray([0, 0, 0, 0, 1, 0, 0, 0], dtype=np.complex64),
        ]
        self.assertEqual(
            subject._nonzero_prefix_mask(captures, 4).tolist(),
            [False, True, False],
        )
        with self.assertRaisesRegex(ValueError, "shorter"):
            subject._nonzero_prefix_mask([np.zeros(3, dtype=np.complex64)], 4)

    def test_multiview_sampler_uses_distinct_bases_and_different_pair(self) -> None:
        views = np.arange(36, dtype=np.int64).reshape(12, 3)
        by_class = (np.arange(0, 6), np.arange(6, 12))
        sample = subject._sample_multiview_episode(
            by_class,
            views,
            np.random.default_rng(123),
            per_class=4,
            include_secondary=True,
        )
        self.assertEqual(len(np.unique(sample["base"][:4])), 4)
        self.assertEqual(len(np.unique(sample["base"][4:])), 4)
        self.assertTrue(np.all(sample["view"] != sample["secondary_view"]))
        self.assertTrue(np.all(sample["primary"] != sample["secondary"]))

    def test_scale_eligibility_is_common_and_metadata_only(self) -> None:
        classes = ("a", "b")
        manifest = {
            "sampleCount": 16384,
            "items": [
                _item("a", 0.01, 0.1),
                _item("a", 0.24, 0.1),
                _item("b", -0.02, 0.08),
            ],
        }
        with mock.patch.object(subject, "SCALE_MIN_PER_CLASS", 1):
            eligible, report = subject._scale_eligible(
                manifest, np.asarray([0, 1, 2]), classes
            )
        self.assertEqual(eligible.tolist(), [0, 2])
        self.assertEqual(report["eligible_by_class"], {"a": 1, "b": 1})
        self.assertEqual(report["excluded_by_class"], {"a": 1, "b": 0})

    def test_paired_report_has_exact_identity_metrics(self) -> None:
        embeddings = np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32
        )
        prediction = np.asarray([0, 1, 0], dtype=np.int64)
        labels = np.asarray([0, 1, 0], dtype=np.int64)
        report = subject._paired_report(
            embeddings,
            embeddings.copy(),
            prediction,
            prediction.copy(),
            labels,
            ("a", "b"),
        )
        self.assertEqual(report["prediction_agreement"], 1.0)
        self.assertEqual(report["embedding_cosine_mean"], 1.0)
        self.assertEqual(report["embedding_cosine_p05"], 1.0)

    def test_run_refuses_release_or_sealed_output_before_loading(self) -> None:
        for path in ("/tmp/releases/v3", "/tmp/Sealed-v3"):
            with self.subTest(path=path), mock.patch.object(
                subject.invariant_data,
                "load",
                side_effect=AssertionError("must reject path before data load"),
            ):
                args = argparse.Namespace(output_dir=path)
                with self.assertRaisesRegex(ValueError, "release or sealed"):
                    subject.run(args)

    def test_high_snr_threshold_matches_frozen_protocol(self) -> None:
        self.assertEqual(subject.HIGH_SNR_DB, 18.0)


if __name__ == "__main__":
    unittest.main()
