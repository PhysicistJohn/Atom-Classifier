"""Focused regression tests for the transfer U-Net data contract.

Run directly:
    .venv-training/bin/python training/zplane_ab/v2_full_variation/test_transfer_pipeline.py
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import torch


HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import native_preprocess as npp  # noqa: E402
import preprocess as pp  # noqa: E402
import pool_cache  # noqa: E402
from length_aug import CropConfig, CropStream, crop_pair  # noqa: E402
from train_transfer import standardize_online_features  # noqa: E402
from unet_transfer import TransferUNet  # noqa: E402


class ClassificationOnlyForwardTest(unittest.TestCase):
    def test_embedding_only_matches_full_forward_and_skips_reconstruction(self):
        torch.manual_seed(17)
        net = TransferUNet(
            base=2,
            depth=2,
            taps=3,
            hidden=16,
            n_profiles=3,
            magnorm=True,
            residual_recon=True,
        ).eval()
        x = torch.randn(4, 2, 64)
        feat = torch.randn(4, 12)
        with torch.no_grad():
            expected = net(x, feat)
            self.assertIsNotNone(net.last_recon)
            actual = net.forward_embedding(x, feat)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        self.assertIsNone(net.last_recon)


class OnlineFeatureContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.data = pool_cache.load("native", build_if_missing=False)
        except pool_cache.CacheUnavailableError as exc:
            raise unittest.SkipTest(str(exc)) from exc

    def test_off_mode_matches_cached_training_coordinates(self):
        """No-crop online samples must equal their cached train-pool counterparts."""
        rows = np.array([0, len(self.data["tr_idx"]) // 2, len(self.data["tr_idx"]) - 1])
        stream = CropStream(
            self.data["tr_idx"],
            npp,
            CropConfig(mode="off"),
            seed=123,
            want_clean=False,
        )
        online_x, raw_f, _ = stream.batch(rows)
        online_f = standardize_online_features(raw_f, self.data)

        cached_x = np.asarray(self.data["xtr"][rows], dtype=np.float32)
        cached_f = np.asarray(self.data["ftr"][rows], dtype=np.float32)
        np.testing.assert_allclose(online_x, cached_x, rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(online_f, cached_f, rtol=0.0, atol=1e-6)

    def test_raw_features_do_not_accidentally_match_standardized_features(self):
        """Guards against deleting the normalization because a smoke test still runs."""
        stream = CropStream(
            self.data["tr_idx"],
            npp,
            CropConfig(mode="off"),
            seed=321,
            want_clean=False,
        )
        _, raw_f, _ = stream.batch([0])
        cached_f = np.asarray(self.data["ftr"][[0]], dtype=np.float32)
        self.assertGreater(float(np.max(np.abs(raw_f - cached_f))), 1e-2)

    def test_pair_crop_uses_clean_occupancy_and_never_returns_empty_target(self):
        """Receiver noise must not make an idle clean window look occupied."""
        n = 4096
        clean = np.zeros(n, dtype=np.complex64)
        clean[3100:3200] = np.exp(2j * np.pi * 0.07 * np.arange(100)).astype(np.complex64)
        rng_noise = np.random.default_rng(99)
        impaired = clean + (
            rng_noise.standard_normal(n) + 1j * rng_noise.standard_normal(n)
        ).astype(np.complex64)
        cfg = CropConfig(mode="uniform", n_min=256, n_max=256,
                         guard_frac=0.35, guard_tries=1)

        for seed in range(20):
            _, target, meta = crop_pair(
                impaired, clean, np.random.default_rng(seed), cfg
            )
            self.assertGreater(float(np.mean(np.abs(target) ** 2)), 0.0)
            self.assertEqual(meta["guard_source"], "paired")

    def test_resampled_pair_reuses_input_geometry(self):
        """Clean and impaired pair members must use one center and resample grid."""
        rng = np.random.default_rng(7)
        n = np.arange(4096)
        primary = np.exp(2j * np.pi * 0.17 * n)
        paired = primary + 0.01 * (
            rng.standard_normal(len(n)) + 1j * rng.standard_normal(len(n))
        )
        _, ctx = pp.preprocess(
            primary, scale_jitter=0.08, rng=np.random.default_rng(11)
        )
        _, paired_ctx = pp.preprocess(
            paired,
            force_center=ctx["center"],
            force_bw=ctx["bw"],
            force_resample_frac=ctx["resample_frac"],
        )
        self.assertEqual(paired_ctx["center"], ctx["center"])
        self.assertEqual(paired_ctx["bw"], ctx["bw"])
        self.assertEqual(paired_ctx["resample_frac"], ctx["resample_frac"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
