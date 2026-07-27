from __future__ import annotations

import os
import sys
import unittest

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import pool_cache  # noqa: E402
import preprocess as pp  # noqa: E402
from ground_state_data import prepare_ground_state, selection_data, test_data  # noqa: E402
from length_aug import CropConfig, CropStream  # noqa: E402
from train_transfer import standardize_online_features  # noqa: E402


class GroundStateDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cached = pool_cache.load("resampled", build_if_missing=False)
        except pool_cache.CacheUnavailableError as exc:
            raise unittest.SkipTest(str(exc)) from exc
        cls.data = prepare_ground_state(cached, seed=123)

    def test_selection_and_test_are_disjoint_at_corpus_level(self):
        sel = selection_data(self.data)
        tst = test_data(self.data)
        self.assertEqual(len(np.intersect1d(sel["va_idx"], tst["va_idx"])), 0)
        self.assertEqual(len(sel["va_idx"]) + len(tst["va_idx"]), len(self.data["va_idx"]))

    def test_each_half_contains_every_class_and_impairment_state(self):
        for part in (selection_data(self.data), test_data(self.data)):
            for cls in range(self.data["n_classes"]):
                for impaired in (False, True):
                    self.assertGreater(
                        int(np.sum((part["yva"] == cls) & (part["imp_va"] == impaired))),
                        0,
                    )

    def test_training_features_are_refit_to_unit_coordinates(self):
        np.testing.assert_allclose(self.data["ftr"].mean(0), 0.0, atol=2e-5)
        # Float32 reduction over ~5k rows accumulates around 1e-4 on the largest
        # sixth-order cumulants; that is numeric summation error, not a coordinate mismatch.
        np.testing.assert_allclose(self.data["ftr"].std(0), 1.0, atol=2e-4)

    def test_crop_off_is_exactly_the_ground_training_transform(self):
        rows = np.array([0, len(self.data["tr_idx"]) // 2, len(self.data["tr_idx"]) - 1])
        stream = CropStream(
            self.data["tr_idx"], pp, CropConfig(mode="off"), seed=8, want_clean=False
        )
        online_x, raw_f, _ = stream.batch(rows)
        online_f = standardize_online_features(raw_f, self.data)
        np.testing.assert_allclose(online_x, self.data["xtr"][rows], rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(online_f, self.data["ftr"][rows], rtol=0.0, atol=1e-6)

    def test_no_retained_band_nominally_crosses_fft_seam(self):
        import json
        from ground_state_data import MANIFEST

        with open(MANIFEST) as f:
            items = json.load(f)["items"]
        for key in ("tr_idx", "en_idx", "va_idx"):
            for i in self.data[key]:
                item = items[int(i)]
                edge = abs(item["centreOffsetFrac"]) + (
                    item["bandwidthHz"] / item["sampleRateHz"]
                ) / 2
                self.assertLess(edge, 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
