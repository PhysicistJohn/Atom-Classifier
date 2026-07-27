from __future__ import annotations

import math
import os
import sys
import unittest

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from invariant_fusion import CenteredInvariantFusion
from invariant_patch_cnn import InvariantPatchCNN, InvariantPatchConfig


def _branches(*, embed_dim: int = 8, dtype: torch.dtype = torch.float32):
    common = dict(
        patch_length=16,
        patch_count=2,
        patch_dim=8,
        hidden=12,
        embed_dim=embed_dim,
        n_features=3,
        dropout=0.0,
    )
    real = InvariantPatchCNN(InvariantPatchConfig(encoder="real", **common)).to(dtype)
    complex_branch = InvariantPatchCNN(
        InvariantPatchConfig(encoder="complex", **common)
    ).to(dtype)
    return real, complex_branch


def _inputs(model: CenteredInvariantFusion, batch: int = 3):
    dtype = next(model.parameters()).dtype
    x = torch.randn(batch, 2, model.packed_length, dtype=dtype)
    feat = torch.randn(
        batch, model.real_branch.cfg.n_features, dtype=dtype
    )
    return x, feat


class CenteredInvariantFusionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260728)

    def test_exact_formula_and_unit_norm(self):
        real, complex_branch = _branches(dtype=torch.float64)
        center_real = torch.linspace(-0.2, 0.3, real.cfg.embed_dim)
        center_complex = torch.linspace(0.25, -0.1, real.cfg.embed_dim)
        model = CenteredInvariantFusion(
            real,
            complex_branch,
            center_real,
            center_complex,
            alpha_real=0.35,
            alpha_complex=0.8,
            weight_real=0.3,
        ).eval()
        x, feat = _inputs(model, batch=4)
        with torch.no_grad():
            real_embedding = real(x, feat)
            complex_embedding = complex_branch(x, feat)
            expected_real = F.normalize(
                real_embedding - 0.35 * model.real_center, dim=-1
            )
            expected_complex = F.normalize(
                complex_embedding - 0.8 * model.complex_center, dim=-1
            )
            expected = F.normalize(
                torch.cat(
                    (
                        math.sqrt(0.3) * expected_real,
                        math.sqrt(0.7) * expected_complex,
                    ),
                    dim=-1,
                ),
                dim=-1,
            )
            actual = model(x, feat)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(
            actual.norm(dim=-1),
            torch.ones(actual.shape[0], dtype=actual.dtype),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_weight_endpoints_select_one_branch_without_rescaling(self):
        real, complex_branch = _branches()
        center_real = torch.randn(real.cfg.embed_dim) * 0.1
        center_complex = torch.randn(real.cfg.embed_dim) * 0.1
        zero = CenteredInvariantFusion(
            real,
            complex_branch,
            center_real,
            center_complex,
            alpha_real=0.2,
            alpha_complex=0.4,
            weight_real=0.0,
        ).eval()
        one = CenteredInvariantFusion(
            real,
            complex_branch,
            center_real,
            center_complex,
            alpha_real=0.2,
            alpha_complex=0.4,
            weight_real=1.0,
        ).eval()
        x, feat = _inputs(zero)
        with torch.no_grad():
            real_expected = F.normalize(
                real(x, feat) - 0.2 * center_real, dim=-1
            )
            complex_expected = F.normalize(
                complex_branch(x, feat) - 0.4 * center_complex, dim=-1
            )
            at_zero = zero(x, feat)
            at_one = one(x, feat)
        torch.testing.assert_close(
            at_zero[:, : real.cfg.embed_dim],
            torch.zeros_like(complex_expected),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            at_zero[:, real.cfg.embed_dim :], complex_expected, atol=2e-7, rtol=2e-7
        )
        torch.testing.assert_close(
            at_one[:, : real.cfg.embed_dim], real_expected, atol=2e-7, rtol=2e-7
        )
        torch.testing.assert_close(
            at_one[:, real.cfg.embed_dim :],
            torch.zeros_like(real_expected),
            rtol=0.0,
            atol=0.0,
        )

    def test_exact_cancellation_has_deterministic_unit_fallback(self):
        real, complex_branch = _branches(embed_dim=4)
        x = torch.randn(2, 2, real.packed_length)
        feat = torch.randn(2, real.cfg.n_features)
        real.eval()
        complex_branch.eval()
        with torch.no_grad():
            real_center = real(x[:1], feat[:1]).squeeze(0)
        model = CenteredInvariantFusion(
            real,
            complex_branch,
            real_center,
            torch.zeros(4),
            alpha_real=1.0,
            alpha_complex=0.0,
            weight_real=1.0,
        ).eval()
        with torch.no_grad():
            result = model(x[:1], feat[:1])
        expected = torch.zeros_like(result)
        expected[0, 0] = 1.0
        torch.testing.assert_close(result, expected, rtol=0.0, atol=0.0)

    def test_buffers_config_parameter_counts_and_state_dict_round_trip(self):
        real, complex_branch = _branches()
        model = CenteredInvariantFusion(
            real,
            complex_branch,
            torch.arange(8) / 20,
            -torch.arange(8) / 30,
            alpha_real=0.25,
            alpha_complex=0.75,
            weight_real=0.4,
        ).eval()
        named_buffers = dict(model.named_buffers())
        for name in (
            "real_center",
            "complex_center",
            "alpha_real",
            "alpha_complex",
            "weight_real",
            "eps",
        ):
            self.assertIn(name, named_buffers)
        self.assertFalse(any(name.endswith("center") for name, _ in model.named_parameters()))
        counts = model.parameter_counts()
        self.assertEqual(counts["fusion"], 0)
        self.assertEqual(
            counts["total"], sum(parameter.numel() for parameter in model.parameters())
        )
        self.assertEqual(counts["total"], counts["real"] + counts["complex"])
        self.assertEqual(model.config()["embed_dim"], 16)
        self.assertEqual(model.config()["weight_real"], model.weight_real.item())

        clone_real, clone_complex = _branches()
        clone = CenteredInvariantFusion(
            clone_real, clone_complex, torch.zeros(8), torch.zeros(8)
        ).eval()
        clone.load_state_dict(model.state_dict(), strict=True)
        x, feat = _inputs(model)
        with torch.no_grad():
            torch.testing.assert_close(
                clone(x, feat), model(x, feat), rtol=0.0, atol=0.0
            )
        self.assertEqual(clone.config(), model.config())

    def test_invalid_configuration_centers_and_inputs_fail_closed(self):
        real, complex_branch = _branches()
        center = torch.zeros(real.cfg.embed_dim)
        for kwargs in (
            {"weight_real": -0.01},
            {"weight_real": 1.01},
            {"weight_real": float("nan")},
            {"alpha_real": float("nan")},
            {"alpha_complex": float("inf")},
            {"eps": 0.0},
        ):
            with self.assertRaises(ValueError):
                CenteredInvariantFusion(
                    real, complex_branch, center, center, **kwargs
                )
        with self.assertRaises(ValueError):
            CenteredInvariantFusion(real, complex_branch, center[:-1], center)
        bad_center = center.clone()
        bad_center[2] = float("inf")
        with self.assertRaises(ValueError):
            CenteredInvariantFusion(real, complex_branch, bad_center, center)
        with self.assertRaises(ValueError):
            CenteredInvariantFusion(complex_branch, real, center, center)

        mismatch_real, mismatch_complex = _branches()
        mismatch_complex = InvariantPatchCNN(
            InvariantPatchConfig(
                encoder="complex",
                patch_length=16,
                patch_count=3,
                patch_dim=8,
                hidden=12,
                embed_dim=8,
                n_features=3,
                dropout=0.0,
            )
        )
        with self.assertRaisesRegex(ValueError, "patch_count"):
            CenteredInvariantFusion(
                mismatch_real, mismatch_complex, center, center
            )

        model = CenteredInvariantFusion(real, complex_branch, center, center)
        x, feat = _inputs(model)
        x[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            model(x, feat)

        bad_state = model.state_dict()
        bad_state["weight_real"] = torch.tensor(1.5)
        with self.assertRaisesRegex(ValueError, "weight_real"):
            model.load_state_dict(bad_state)

    def test_gradients_reach_both_branches_and_remain_finite(self):
        real, complex_branch = _branches()
        model = CenteredInvariantFusion(
            real,
            complex_branch,
            torch.randn(real.cfg.embed_dim) * 0.1,
            torch.randn(real.cfg.embed_dim) * 0.1,
            alpha_real=0.4,
            alpha_complex=0.6,
            weight_real=0.45,
        ).train()
        x, feat = _inputs(model, batch=4)
        target = torch.linspace(-0.5, 0.7, model.embed_dim).unsqueeze(0)
        loss = (model(x, feat) * target).sum()
        loss.backward()
        for branch in (model.real_branch, model.complex_branch):
            gradients = [
                parameter.grad
                for parameter in branch.parameters()
                if parameter.requires_grad
            ]
            self.assertTrue(gradients)
            self.assertTrue(
                all(gradient is not None for gradient in gradients)
            )
            self.assertTrue(
                all(torch.isfinite(gradient).all() for gradient in gradients)
            )
            self.assertGreater(
                sum(float(gradient.abs().sum()) for gradient in gradients), 0.0
            )
        self.assertFalse(model.real_center.requires_grad)
        self.assertFalse(model.complex_center.requires_grad)


if __name__ == "__main__":
    unittest.main()
