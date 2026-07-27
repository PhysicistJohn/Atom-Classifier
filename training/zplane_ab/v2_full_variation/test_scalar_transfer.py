from __future__ import annotations

import os
import sys
import unittest

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from scalar_transfer import fit_ridge, metrics, tune_ridge  # noqa: E402


class ScalarTransferMathTest(unittest.TestCase):
    def test_ridge_recovers_held_out_linear_scalar(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(500, 12))
        w = rng.normal(size=12)
        y = np.dot(x, w) + 0.05 * rng.normal(size=500)
        head = tune_ridge(x[:300], y[:300], x[300:400], y[300:400])
        score = metrics(y[400:], head.predict(x[400:]))
        self.assertGreater(score["r2"], 0.99)

    def test_constant_predictor_has_nonpositive_r2_off_fit_distribution(self):
        y = np.arange(10.0)
        score = metrics(y, np.zeros_like(y))
        self.assertLessEqual(score["r2"], 0.0)

    def test_serialized_head_prediction_contract(self):
        x = np.arange(20.0).reshape(10, 2)
        y = 3 * x[:, 0] - x[:, 1]
        head = fit_ridge(x, y, 1e-4)
        payload = head.serializable()
        z = (x - np.asarray(payload["mean"])) / np.asarray(payload["scale"])
        z = np.concatenate([z, np.ones((len(z), 1))], axis=1)
        got = np.dot(z, np.asarray(payload["weights"]))
        np.testing.assert_allclose(got, head.predict(x), atol=1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
