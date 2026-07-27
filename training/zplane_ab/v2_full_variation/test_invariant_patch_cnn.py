from __future__ import annotations

import os
import sys
import unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from invariant_patch_cnn import InvariantPatchCNN, InvariantPatchConfig


class InvariantPatchCNNTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)

    def test_shapes_norms_and_finite_gradients(self):
        for encoder in ("real", "complex"):
            net = InvariantPatchCNN(encoder=encoder)
            x = torch.randn(4, 2, net.packed_length)
            feat = torch.randn(4, net.cfg.n_features)
            out = net(x, feat)
            self.assertEqual(tuple(out.shape), (4, net.cfg.embed_dim))
            self.assertTrue(torch.isfinite(out).all())
            self.assertTrue(
                torch.allclose(out.norm(dim=-1), torch.ones(4), atol=1e-6)
            )
            out.square().sum().backward()
            grads = [p.grad for p in net.parameters() if p.requires_grad]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))

    def test_patch_permutation_is_invariant(self):
        for encoder in ("real", "complex"):
            net = InvariantPatchCNN(encoder=encoder).eval()
            x = torch.randn(5, 2, net.packed_length)
            feat = torch.randn(5, net.cfg.n_features)
            perm = torch.randperm(net.cfg.patch_count)
            xp = (
                x.reshape(5, 2, net.cfg.patch_count, net.cfg.patch_length)[:, :, perm]
                .reshape_as(x)
            )
            with torch.no_grad():
                a, b = net(x, feat), net(xp, feat)
            self.assertLess(float((a - b).abs().max()), 3e-6)

    def test_complex_encoder_is_global_phase_invariant(self):
        net = InvariantPatchCNN(encoder="complex").eval()
        x = torch.randn(4, 2, net.packed_length)
        feat = torch.randn(4, net.cfg.n_features)
        z = torch.complex(x[:, 0], x[:, 1])
        phase = torch.tensor(1.713)
        rotated = z * torch.complex(torch.cos(phase), torch.sin(phase))
        xr = torch.stack((rotated.real, rotated.imag), dim=1)
        with torch.no_grad():
            a, b = net(x, feat), net(xr, feat)
        self.assertLess(float((a - b).abs().max()), 4e-5)

    def test_bad_configuration_and_inputs_fail_closed(self):
        for kwargs in (
            {"patch_length": 8},
            {"patch_count": 0},
            {"encoder": "fft"},
            {"set_pool": "ordered"},
            {"dropout": 1.0},
        ):
            with self.assertRaises(ValueError):
                InvariantPatchCNN(**kwargs)
        net = InvariantPatchCNN()
        feat = torch.randn(2, net.cfg.n_features)
        with self.assertRaises(ValueError):
            net(torch.randn(2, 2, net.packed_length - 1), feat)
        with self.assertRaises(ValueError):
            net(torch.randn(2, 2, net.packed_length), feat[:, :-1])

    def test_config_round_trip(self):
        cfg = InvariantPatchConfig(
            patch_length=80,
            patch_count=12,
            encoder="complex",
            set_pool="mean",
        )
        net = InvariantPatchCNN(cfg)
        self.assertEqual(net.config(), cfg.__dict__)
        clone = InvariantPatchCNN(InvariantPatchConfig(**net.config()))
        clone.load_state_dict(net.state_dict(), strict=True)


if __name__ == "__main__":
    unittest.main()
