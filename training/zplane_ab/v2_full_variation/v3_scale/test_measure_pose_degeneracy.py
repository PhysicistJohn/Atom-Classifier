"""Unit suite for the pose-degeneracy separation measurement.

These tests cover the parts of the measurement that can be wrong silently: the
direction handling of a single-feature AUROC, the worst-of collapse, the
resolution-floor selection, and the seed guardrails that must be inherited from
``fit_v3_openset`` rather than re-implemented (the spent design seed 20260938 and
the unspent release seed 20260731 must both be refused structurally).

Nothing here touches the corpus, so the suite is fast and hermetic.
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
_V2_DIR = os.path.dirname(HERE)
_ZPAB_DIR = os.path.dirname(_V2_DIR)
_TRAINING_DIR = os.path.dirname(_ZPAB_DIR)
for _path in (_TRAINING_DIR, _ZPAB_DIR, _V2_DIR, HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import measure_pose_degeneracy as measure  # noqa: E402
import pose_degeneracy as pdg  # noqa: E402
import time_domain_geometry as geometry  # noqa: E402


class AurocTest(unittest.TestCase):
    def test_perfect_separation_in_both_directions(self):
        known = np.zeros((8, pdg.FEATURE_COUNT), dtype=np.float64)
        novel = np.ones((8, pdg.FEATURE_COUNT), dtype=np.float64)
        novel[:, 0] = -1.0
        values = measure.feature_auroc(known, novel)
        self.assertEqual(values.shape, (pdg.FEATURE_COUNT,))
        self.assertEqual(values[0], 0.0)
        self.assertEqual(values[1], 1.0)
        self.assertEqual(measure.oriented(values[0]), 1.0)
        self.assertEqual(measure.oriented(values[1]), 1.0)
        self.assertEqual(measure.direction(values[0]), "novelty_low")
        self.assertEqual(measure.direction(values[1]), "novelty_high")

    def test_identical_populations_score_one_half(self):
        rows = np.arange(12, dtype=np.float64).reshape(-1, 1)
        rows = np.repeat(rows, pdg.FEATURE_COUNT, axis=1)
        values = measure.feature_auroc(rows, rows.copy())
        np.testing.assert_allclose(values, 0.5)
        self.assertEqual(measure.direction(0.5), "none")

    def test_mismatched_feature_axes_are_refused(self):
        with self.assertRaises(ValueError):
            measure.feature_auroc(np.zeros((4, 3)), np.zeros((4, 5)))
        with self.assertRaises(ValueError):
            measure.feature_auroc(np.zeros(4), np.zeros((4, 4)))


class SummaryTest(unittest.TestCase):
    def _cells(self, first, second):
        return [
            {"cell": "a", "auroc": np.full(pdg.FEATURE_COUNT, first)},
            {"cell": "b", "auroc": np.full(pdg.FEATURE_COUNT, second)},
        ]

    def test_worst_of_is_the_minimum_oriented_value(self):
        summary = measure.summarize_cells(self._cells(0.9, 0.8))
        entry = summary[pdg.FEATURE_NAMES[0]]
        self.assertAlmostEqual(entry["oriented_worst"], 0.8)
        self.assertAlmostEqual(entry["auroc_min"], 0.8)
        self.assertAlmostEqual(entry["auroc_max"], 0.9)
        self.assertEqual(entry["direction"], "novelty_high")
        self.assertTrue(entry["direction_consistent"])
        self.assertEqual(sorted(entry["cells"]), ["a", "b"])

    def test_a_direction_flip_withholds_the_oriented_worst(self):
        summary = measure.summarize_cells(self._cells(0.9, 0.1))
        entry = summary[pdg.FEATURE_NAMES[0]]
        self.assertFalse(entry["direction_consistent"])
        self.assertIsNone(entry["oriented_worst"])
        self.assertEqual(entry["direction"], "mixed")

    def test_low_direction_is_oriented_not_discarded(self):
        summary = measure.summarize_cells(self._cells(0.05, 0.2))
        entry = summary[pdg.FEATURE_NAMES[0]]
        self.assertTrue(entry["direction_consistent"])
        self.assertAlmostEqual(entry["oriented_worst"], 0.8)
        self.assertEqual(entry["direction"], "novelty_low")

    def test_summary_covers_every_feature_and_ranks_them(self):
        cells = [{"cell": "a", "auroc": np.linspace(0.5, 1.0, pdg.FEATURE_COUNT)}]
        summary = measure.summarize_cells(cells)
        self.assertEqual(tuple(sorted(summary)), tuple(sorted(pdg.FEATURE_NAMES)))
        order = measure.ranked(summary)
        self.assertEqual(sorted(order), sorted(pdg.FEATURE_NAMES))
        self.assertEqual(order[0], pdg.FEATURE_NAMES[-1])

    def test_empty_cells_are_refused(self):
        with self.assertRaises(ValueError):
            measure.summarize_cells([])


class ProtocolTest(unittest.TestCase):
    def test_resolution_floor_modes(self):
        self.assertEqual(
            measure._min_bandwidth(4096, "fixed"), geometry.MIN_BANDWIDTH
        )
        frontend = measure._min_bandwidth(4096, "frontend")
        self.assertAlmostEqual(
            frontend,
            geometry.minimum_bandwidth_for_active_span(
                4096, measure.PATCH_LENGTH, measure.TARGET_FRAC
            ),
        )
        # The frontend floor scales with the observation, the fixed one does not.
        self.assertGreater(frontend, measure._min_bandwidth(16384, "frontend"))
        with self.assertRaises(ValueError):
            measure._min_bandwidth(4096, "welch")

    def test_spent_and_unspent_seeds_are_refused(self):
        parser = measure.build_parser()
        defaults = parser.parse_args([])
        self.assertEqual(defaults.novelty_seeds, [20260939, 20260940])
        self.assertEqual(defaults.min_bandwidth_mode, "frontend")
        with self.assertRaises(ValueError):
            measure.novelty_features(
                [20260938],
                n_each=1,
                lengths=(4096,),
                min_bandwidth_mode="frontend",
            )
        with self.assertRaises(ValueError):
            measure.novelty_features(
                [20260731],
                n_each=1,
                lengths=(4096,),
                min_bandwidth_mode="frontend",
            )

    def test_unknown_known_split_is_refused(self):
        with self.assertRaises(ValueError):
            measure.known_features(
                "selection", lengths=(4096,), min_bandwidth_mode="frontend"
            )
        with self.assertRaises(ValueError):
            measure.known_features(
                "test", lengths=(4096,), min_bandwidth_mode="frontend"
            )

    def test_novelty_features_are_deterministic_per_seed(self):
        first, provenance = measure.novelty_features(
            [20260939],
            n_each=3,
            lengths=(4096,),
            min_bandwidth_mode="frontend",
        )
        second, _ = measure.novelty_features(
            [20260939],
            n_each=3,
            lengths=(4096,),
            min_bandwidth_mode="frontend",
        )
        for family in ("noise", "chirp"):
            self.assertTrue(
                np.array_equal(
                    first[20260939][family][4096],
                    second[20260939][family][4096],
                )
            )
            self.assertEqual(
                first[20260939][family][4096].shape, (3, pdg.FEATURE_COUNT)
            )
        self.assertEqual(provenance["20260939"]["noise"]["n"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
