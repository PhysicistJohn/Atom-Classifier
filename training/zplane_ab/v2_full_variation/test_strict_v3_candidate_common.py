from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from strict_v3_candidate_common import (
    OCCUPANCY_MIN_POWER,
    prefix_mean_power,
    release_compatible_occupancy,
)


class StrictV3CandidateCommonTest(unittest.TestCase):
    def test_prefix_mean_power_is_exact_and_validates_inputs(self) -> None:
        clean = np.zeros((3, 8, 2), dtype=np.float32)
        clean[1, :4, 0] = 2.0
        clean[2, :4, 1] = np.asarray([1.0, 2.0, 3.0, 4.0])
        value = prefix_mean_power(
            clean,
            np.asarray([2, 0, 1], dtype=np.int64),
            length=4,
            block_rows=1,
        )
        np.testing.assert_array_equal(value, [7.5, 0.0, 4.0])
        with self.assertRaises(ValueError):
            prefix_mean_power(clean, np.asarray([-1]), length=4)
        with self.assertRaises(ValueError):
            prefix_mean_power(clean, np.asarray([0]), length=9)
        with self.assertRaises(ValueError):
            prefix_mean_power(clean[:, :, 0], np.asarray([0]), length=4)

    def test_release_compatible_occupancy_is_label_and_model_blind(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clean = np.memmap(
                root / "corpus_clean.f32",
                dtype="<f4",
                mode="w+",
                shape=(4, 4096, 2),
            )
            clean[:] = 0.0
            clean[1, 0, 0] = np.sqrt(
                4096.0 * OCCUPANCY_MIN_POWER * 2.0
            )
            clean[2, :, 1] = 0.5
            clean.flush()
            del clean
            manifest = {
                "count": 4,
                "sampleCount": 4096,
                "items": [
                    {"cls": "a", "impaired": False},
                    {"cls": "a", "impaired": True},
                    {"cls": "b", "impaired": False},
                    {"cls": "b", "impaired": True},
                ],
            }
            selected, audit = release_compatible_occupancy(
                manifest,
                np.asarray([3, 1, 0, 2], dtype=np.int64),
                ["a", "b"],
                corpus_dir=root,
            )
            np.testing.assert_array_equal(selected, [1, 2])
            self.assertEqual(audit["eligible_by_class"], {"a": 1, "b": 1})
            self.assertEqual(audit["excluded_by_class"], {"a": 1, "b": 1})
            self.assertEqual(
                audit["excluded_by_impairment"],
                {"clean": 1, "impaired": 1},
            )
            self.assertFalse(audit["selection_uses_labels"])
            self.assertFalse(audit["selection_uses_model_outputs"])
            self.assertEqual(
                audit["maximum_excluded_clean_prefix_power"],
                0.0,
            )
            self.assertGreater(
                audit["minimum_included_clean_prefix_power"],
                OCCUPANCY_MIN_POWER,
            )

    def test_clean_corpus_size_must_match_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "corpus_clean.f32").write_bytes(b"\0" * 8)
            manifest = {
                "count": 1,
                "sampleCount": 4096,
                "items": [{"cls": "a", "impaired": False}],
            }
            with self.assertRaisesRegex(ValueError, "size"):
                release_compatible_occupancy(
                    manifest,
                    np.asarray([0], dtype=np.int64),
                    ["a"],
                    corpus_dir=root,
                )


if __name__ == "__main__":
    unittest.main()
