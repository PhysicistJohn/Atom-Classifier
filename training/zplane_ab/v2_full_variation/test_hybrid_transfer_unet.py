"""Focused contract tests for :mod:`hybrid_transfer_unet`.

Run directly:
    .venv-training/bin/python \
      training/zplane_ab/v2_full_variation/test_hybrid_transfer_unet.py
"""

from __future__ import annotations

import os
import sys
import unittest

import torch


HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.dirname(os.path.dirname(HERE))
for path in (TRAINING, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

from hybrid_transfer_unet import HybridTransferUNet  # noqa: E402
from model import Embedding  # noqa: E402


def _tiny_hybrid(source: str = "original") -> HybridTransferUNet:
    return HybridTransferUNet(
        base=2,
        depth=2,
        taps=3,
        n_profiles=3,
        hidden=16,
        magnorm=True,
        skip_dropout=0.0,
        feat_dropout=0.0,
        residual_recon=True,
        classify_source=source,
    )


class HybridTransferUNetTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260727)
        self.x = torch.randn(3, 2, 64)
        self.feat = torch.randn(3, 12)

    def test_original_route_matches_loaded_incumbent_exactly(self):
        incumbent = Embedding().eval()
        hybrid = _tiny_hybrid("original").eval()
        result = hybrid.load_incumbent_state_dict(incumbent.state_dict())
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])

        with torch.no_grad():
            expected = incumbent(self.x, self.feat)
            actual = hybrid(self.x, self.feat)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            actual.norm(dim=-1),
            torch.ones(len(actual)),
            rtol=0.0,
            atol=1e-7,
        )

    def test_identity_reconstruction_route_also_matches_incumbent_exactly(self):
        incumbent = Embedding().eval()
        hybrid = _tiny_hybrid("reconstruction").eval()
        hybrid.load_incumbent_state_dict({"state_dict": incumbent.state_dict()})

        with torch.no_grad():
            expected = incumbent(self.x, self.feat)
            actual = hybrid(self.x, self.feat)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            hybrid.last_recon,
            torch.complex(self.x[:, 0], self.x[:, 1]),
            rtol=0.0,
            atol=0.0,
        )

    def test_component_and_complete_state_loading_are_strict(self):
        source = _tiny_hybrid("original").eval()
        target = _tiny_hybrid("reconstruction").eval()

        # Component helpers accept a complete prefixed hybrid state.
        result_cnn = target.load_incumbent_state_dict(source.state_dict())
        result_unet = target.load_unet_state_dict(source.state_dict())
        self.assertEqual(result_cnn.missing_keys + result_cnn.unexpected_keys, [])
        self.assertEqual(result_unet.missing_keys + result_unet.unexpected_keys, [])
        for key, value in source.state_dict().items():
            torch.testing.assert_close(
                target.state_dict()[key],
                value,
                rtol=0.0,
                atol=0.0,
            )

        # A normal full-checkpoint round trip remains strict as well.
        clone = _tiny_hybrid("original")
        result = clone.load_state_dict(source.state_dict(), strict=True)
        self.assertEqual(result.missing_keys + result.unexpected_keys, [])

    def test_decoder_identity_and_latent_surfaces_are_preserved(self):
        hybrid = _tiny_hybrid("original").eval()
        with torch.no_grad():
            hybrid(self.x, self.feat)
        expected = torch.complex(self.x[:, 0], self.x[:, 1])
        torch.testing.assert_close(
            hybrid.last_recon,
            expected,
            rtol=0.0,
            atol=0.0,
        )
        self.assertIs(hybrid.last_latent, hybrid.last_bottleneck_pool)
        self.assertIs(hybrid.latent, hybrid.last_bottleneck_pool)
        self.assertIsNotNone(hybrid.last_unet_embedding)
        self.assertEqual(tuple(hybrid.last_unet_embedding.shape), (3, 32))
        self.assertIsNotNone(hybrid.last_profile_logits)
        self.assertIsNotNone(hybrid.last_params)

    def test_reconstruction_route_has_finite_classifier_and_decoder_gradients(self):
        hybrid = _tiny_hybrid("reconstruction").train()
        embedding = hybrid(self.x, self.feat)
        target = torch.complex(
            0.9 * self.x[:, 0] + 0.1 * self.x[:, 1],
            0.9 * self.x[:, 1] - 0.1 * self.x[:, 0],
        )
        reconstruction_loss = (
            (hybrid.last_recon.real - target.real).square()
            + (hybrid.last_recon.imag - target.imag).square()
        ).mean()
        class_targets = embedding.new_tensor([-0.25, 0.10, 0.35])
        classification_loss = (
            embedding[:, 0] - class_targets
        ).square().mean()
        loss = classification_loss + reconstruction_loss
        loss.backward()

        gradients = {
            name: parameter.grad
            for name, parameter in hybrid.named_parameters()
            if parameter.grad is not None
        }
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(grad).all() for grad in gradients.values()))
        classifier_grad = sum(
            float(grad.abs().sum())
            for name, grad in gradients.items()
            if name.startswith("classifier.")
        )
        decoder_output_grad = sum(
            float(grad.abs().sum())
            for name, grad in gradients.items()
            if name.startswith("unet.out.")
        )
        self.assertGreater(classifier_grad, 0.0, sorted(gradients))
        self.assertGreater(decoder_output_grad, 0.0, sorted(gradients))

    def test_default_parameter_count_is_explicit_and_bounded(self):
        hybrid = HybridTransferUNet()
        counts = hybrid.parameter_counts()
        config = hybrid.config()
        self.assertEqual(config["unet"]["base"], 8)
        self.assertEqual(config["unet"]["depth"], 4)
        self.assertTrue(config["unet"]["residual_recon"])
        self.assertEqual(config["classifier"]["hidden"], 96)
        self.assertEqual(config["classify_source"], "original")
        self.assertEqual(counts["classifier"], 38_208)
        self.assertEqual(counts["unet"], 807_467)
        self.assertEqual(counts["total"], 845_675)
        self.assertEqual(counts["trainable"], counts["total"])
        self.assertEqual(
            counts["total"],
            sum(parameter.numel() for parameter in hybrid.parameters()),
        )
        self.assertLess(counts["total"], 1_000_000)

    def test_embedding_fast_path_matches_full_original_route(self):
        hybrid = _tiny_hybrid("original").eval()
        with torch.no_grad():
            expected = hybrid(self.x, self.feat)
            self.assertIsNotNone(hybrid.last_recon)
            actual = hybrid.forward_embedding(self.x, self.feat)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        self.assertIsNone(hybrid.last_recon)
        self.assertIsNotNone(hybrid.last_latent)

    def test_invalid_route_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "classify_source"):
            HybridTransferUNet(classify_source="latent")  # type: ignore[arg-type]
        hybrid = _tiny_hybrid()
        with self.assertRaisesRegex(ValueError, "classify_source"):
            hybrid.set_classify_source("latent")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
