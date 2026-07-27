"""Focused regressions for the live-checkpoint canonical probe."""
from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

import numpy as np
import torch


HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import canonical_probe_net as cpn


class _FeatureRecordingNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features_seen = None

    def forward(self, xb, fb):
        self.features_seen = fb.detach().cpu().numpy().copy()
        return torch.eye(xb.shape[0], 3, dtype=xb.dtype, device=xb.device)


class CanonicalProbeNetTest(unittest.TestCase):
    def test_standardize_features_matches_training_transform(self):
        raw = np.array([[3.0, 10.0], [5.0, 14.0]], dtype=np.float32)
        got = cpn._standardize_features(
            raw,
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([2.0, 4.0], dtype=np.float32),
        )
        np.testing.assert_allclose(got, [[1.0, 2.0], [2.0, 3.0]], rtol=0, atol=0)

    def test_standardize_features_rejects_bad_statistics(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            cpn._standardize_features(
                np.ones((2, 2), dtype=np.float32),
                np.zeros(2, dtype=np.float32),
                np.array([1.0, 0.0], dtype=np.float32),
            )

    def test_gate_uses_balanced_not_raw_accuracy(self):
        # The old raw-accuracy gate would accept this CW-majority result.
        rows = [dict(
            n=16384, correct=0.7895, bal=0.60, cos=0.10, n_predicted_classes=3)]
        got = cpn._summarize_rows(rows, matched_length=16384)
        self.assertTrue(got["valid"])
        self.assertFalse(got["passes_gate"])
        self.assertEqual(got["worst_bal"], 0.60)

    def test_matched_condition_must_be_nondegenerate(self):
        rows = [dict(
            n=16384, correct=0.80, bal=0.80, cos=0.10, n_predicted_classes=1)]
        got = cpn._summarize_rows(rows, matched_length=16384)
        self.assertFalse(got["valid"])
        self.assertFalse(got["passes_gate"])
        self.assertTrue(any("fewer than two" in x for x in got["validity_reasons"]))

    def test_sweep_applies_training_feature_statistics(self):
        classes = ["am", "cw", "fm"]
        raw_features = np.arange(12, dtype=np.float32) + 10.0
        fmean = np.arange(12, dtype=np.float32)
        fstd = np.full(12, 2.0, dtype=np.float32)
        probes = [
            ("am", "am", np.zeros(8, dtype=np.complex64)),
            ("cw", "cw", np.zeros(8, dtype=np.complex64)),
            ("fm", "fm", np.zeros(8, dtype=np.complex64)),
        ]
        net = _FeatureRecordingNet()

        def fake_preprocess(_iq, l_out):
            return (
                np.zeros(l_out, dtype=np.complex64),
                {"center": 0.0, "bw": 0.1, "resample_frac": 0.1},
            )

        with (
            mock.patch.object(cpn, "build_probes", side_effect=lambda _rng, _n: probes),
            mock.patch.object(cpn.npp, "preprocess", side_effect=fake_preprocess),
            mock.patch.object(
                cpn.npp, "to_channels",
                side_effect=lambda z: np.stack([z.real, z.imag]).astype(np.float32)),
            mock.patch.object(cpn.npp, "iq_features", return_value=raw_features),
        ):
            result = cpn.sweep(
                net,
                np.eye(3, dtype=np.float32),
                classes,
                torch.device("cpu"),
                fmean=fmean,
                fstd=fstd,
                condition="native",
                in_len=8,
                lengths=(8,),
                matched_length=8,
            )

        np.testing.assert_allclose(
            net.features_seen,
            np.full((3, 12), 5.0, dtype=np.float32),
            rtol=0,
            atol=0,
        )
        self.assertEqual(result["worst_bal"], 1.0)
        self.assertTrue(result["valid"])
        self.assertTrue(result["passes_gate"])


if __name__ == "__main__":
    unittest.main()
