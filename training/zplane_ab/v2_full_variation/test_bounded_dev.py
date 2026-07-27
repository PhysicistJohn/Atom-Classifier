from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from run_bounded_dev import (  # noqa: E402
    DenoiseThenEmbed,
    EmbeddingView,
    _architecture_from_args,
    _closed_report,
    build_parser,
    selection_only_view,
    stratified_folds,
)


class _TinyEmbedding(nn.Module):
    def forward(self, x, feat):
        h = torch.stack(
            [x[:, 0].mean(-1), x[:, 1].mean(-1), feat[:, 0], feat[:, 1]], dim=-1
        )
        return F.normalize(h, dim=-1)


class _IdentityDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_recon = None

    def forward(self, x, feat):
        del feat
        self.last_recon = torch.complex(x[:, 0], x[:, 1])
        return torch.zeros((len(x), 2), dtype=x.dtype, device=x.device)


class _FastPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.full_calls = 0
        self.fast_calls = 0

    def forward(self, x, feat):
        self.full_calls += 1
        return x[:, :1, 0].repeat(1, 2)

    def forward_embedding(self, x, feat):
        del feat
        self.fast_calls += 1
        return x[:, :1, 0].repeat(1, 2)


class BoundedDevelopmentTests(unittest.TestCase):
    def test_stratified_folds_are_disjoint_exhaustive_and_supported(self):
        y = np.repeat(np.arange(3), 8)
        impaired = np.tile(np.array([False, True]), 12)
        folds = stratified_folds(y, impaired, n_folds=2, seed=11)
        self.assertEqual(len(np.intersect1d(folds[0], folds[1])), 0)
        self.assertTrue(np.array_equal(np.sort(np.concatenate(folds)), np.arange(len(y))))
        for rows in folds:
            self.assertEqual(set(y[rows]), {0, 1, 2})
            self.assertEqual(set(impaired[rows]), {False, True})

    def test_embedding_view_prefers_encoder_only_surface(self):
        model = _FastPath()
        view = EmbeddingView(model)
        x = torch.ones(3, 2, 4)
        out = view(x, torch.ones(3, 2))
        self.assertEqual(tuple(out.shape), (3, 2))
        self.assertEqual(model.fast_calls, 1)
        self.assertEqual(model.full_calls, 0)

    def test_identity_cascade_matches_direct_embedding_after_normalization(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(5, 2, 32)).astype(np.float32)
        rms = np.sqrt(np.mean(x[:, 0] ** 2 + x[:, 1] ** 2, axis=1, keepdims=True))
        x /= rms[:, None, :]
        feat_mean = np.zeros(12, dtype=np.float32)
        feat_std = np.ones(12, dtype=np.float32)
        # The tiny incumbent reads only the first two recomputed features in addition to
        # waveform means, so compare to the same explicitly recomputed direct surface.
        from preprocess import iq_features

        raw_feat = np.stack([iq_features(row[0] + 1j * row[1]) for row in x])
        incumbent = _TinyEmbedding()
        cascade = DenoiseThenEmbed(
            _IdentityDenoiser(), incumbent, feature_mean=feat_mean, feature_std=feat_std
        )
        xt = torch.from_numpy(x)
        got = cascade(xt, torch.from_numpy(raw_feat))
        want = incumbent(xt, torch.from_numpy(raw_feat))
        self.assertTrue(torch.allclose(got, want, atol=2e-5, rtol=2e-5))

    def test_closed_report_uses_balanced_accuracy(self):
        labels = np.array([0, 0, 0, 1])
        impaired = np.array([False, True, False, True])
        # Three class-0 embeddings and one misclassified class-1 embedding.
        emb = np.array([[1, 0], [1, 0], [1, 0], [1, 0]], dtype=np.float32)
        protos = np.array([[1, 0], [0, 1]], dtype=np.float32)
        report = _closed_report(
            emb, labels, impaired, protos, ["a", "b"], [np.array([0, 3]), np.array([1, 2])]
        )
        self.assertEqual(report["accuracy"], 0.75)
        self.assertEqual(report["balanced_accuracy"], 0.5)

    def test_selection_only_view_drops_test_selectors_and_rows(self):
        n = 6
        ground = {
            "classes": ["a", "b"],
            "n_classes": 2,
            "xva": np.arange(n * 2 * 4, dtype=np.float32).reshape(n, 2, 4),
            "fva": np.arange(n * 3, dtype=np.float32).reshape(n, 3),
            "yva": np.array([0, 0, 1, 1, 0, 1]),
            "imp_va": np.array([False, True, False, True, False, True]),
            "va_idx": np.array([100, 101, 102, 103, 104, 105]),
            "selection_rows": np.array([0, 2, 4]),
            "test_rows": np.array([1, 3, 5]),
            "ground_state": {"seed": 20260727},
        }
        dev, audit = selection_only_view(ground)
        self.assertNotIn("selection_rows", dev)
        self.assertNotIn("test_rows", dev)
        self.assertTrue(np.array_equal(dev["va_idx"], np.array([100, 102, 104])))
        self.assertEqual(audit["consumed_test_rows_exposed"], 0)
        self.assertEqual(audit["selection_test_overlap"], 0)
        self.assertEqual(dev["ground_state"]["consumed_test_rows_exposed"], 0)

    def test_architecture_overrides_are_explicit_and_validated(self):
        args = build_parser().parse_args(
            [
                "--unet-head-pool",
                "acf_complex",
                "--unet-feat-dropout",
                "0.1",
                "--unet-skip-dropout",
                "0.25",
                "--no-unet-antialias",
            ]
        )
        architecture = _architecture_from_args(args)
        self.assertEqual(architecture["head_pool"], "acf_complex")
        self.assertEqual(architecture["feat_dropout"], 0.1)
        self.assertEqual(architecture["skip_dropout"], 0.25)
        self.assertFalse(architecture["antialias"])

        args.unet_taps = 4
        with self.assertRaises(ValueError):
            _architecture_from_args(args)


if __name__ == "__main__":
    unittest.main()
