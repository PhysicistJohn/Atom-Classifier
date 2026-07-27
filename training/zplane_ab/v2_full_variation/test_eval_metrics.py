from __future__ import annotations

import os
import sys
import unittest

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.dirname(os.path.dirname(HERE))
for path in (TRAINING, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

from train import auroc  # noqa: E402


class AurocTest(unittest.TestCase):
    def test_all_ties_are_half_regardless_of_order(self):
        self.assertEqual(auroc(np.ones(7), np.ones(11)), 0.5)
        self.assertEqual(auroc(np.ones(11), np.ones(7)), 0.5)

    def test_tied_example_matches_pairwise_definition(self):
        pos = np.array([0.0, 1.0, 1.0, 3.0])
        neg = np.array([0.0, 1.0, 2.0])
        pairwise = np.mean(
            (pos[:, None] > neg[None, :])
            + 0.5 * (pos[:, None] == neg[None, :])
        )
        self.assertAlmostEqual(auroc(pos, neg), float(pairwise))

    def test_perfect_orderings(self):
        self.assertEqual(auroc([2, 3], [0, 1]), 1.0)
        self.assertEqual(auroc([0, 1], [2, 3]), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
