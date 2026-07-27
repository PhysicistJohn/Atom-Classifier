from __future__ import annotations

import unittest

import torch

from invariant_fusion import CenteredInvariantFusion
from invariant_paired_real_inference import (
    make_paired_real_branch,
    make_paired_real_fusion,
)
from invariant_patch_cnn import InvariantPatchCNN, InvariantPatchConfig


class InvariantPairedRealInferenceTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260727)
        self.real_config = InvariantPatchConfig(encoder="real", dropout=0.0)
        self.complex_config = InvariantPatchConfig(
            encoder="complex", dropout=0.0
        )
        self.iq = torch.randn(3, 2, 1024)
        self.features = torch.randn(3, 12)

    def test_each_branch_matches_original_complex_graph(self) -> None:
        for config in (self.real_config, self.complex_config):
            with self.subTest(encoder=config.encoder):
                reference = InvariantPatchCNN(config).eval()
                paired = make_paired_real_branch(
                    config, reference.state_dict()
                )
                with torch.no_grad():
                    expected = reference(self.iq, self.features)
                    actual = paired(self.iq, self.features)
                self.assertEqual(actual.shape, expected.shape)
                self.assertTrue(bool(torch.isfinite(actual).all()))
                self.assertLessEqual(
                    float((expected - actual).abs().max()),
                    2e-6,
                )

    def test_complete_fusion_state_loads_strictly_and_matches(self) -> None:
        real = InvariantPatchCNN(self.real_config)
        complex_branch = InvariantPatchCNN(self.complex_config)
        real_center = torch.linspace(-0.2, 0.3, 32)
        complex_center = torch.linspace(0.25, -0.1, 32)
        reference = CenteredInvariantFusion(
            real,
            complex_branch,
            real_center,
            complex_center,
            alpha_real=0.35,
            alpha_complex=0.8,
            weight_real=0.3,
            eps=1e-10,
        ).eval()
        paired = make_paired_real_fusion(
            self.real_config,
            self.complex_config,
            reference.state_dict(),
        )
        with torch.no_grad():
            expected = reference(self.iq, self.features)
            actual = paired(self.iq, self.features)
        self.assertEqual(actual.shape, (3, 64))
        self.assertLessEqual(
            float((expected - actual).abs().max()),
            2e-6,
        )
        self.assertTrue(
            torch.allclose(
                actual.norm(dim=-1),
                torch.ones(3),
                atol=2e-6,
            )
        )

    def test_paired_graph_contains_no_complex_operators(self) -> None:
        reference = CenteredInvariantFusion(
            InvariantPatchCNN(self.real_config),
            InvariantPatchCNN(self.complex_config),
            torch.zeros(32),
            torch.zeros(32),
        ).eval()
        paired = make_paired_real_fusion(
            self.real_config,
            self.complex_config,
            reference.state_dict(),
        )
        traced = torch.jit.trace(
            paired,
            (self.iq[:1], self.features[:1]),
            strict=True,
        )
        graph = str(traced.inlined_graph)
        for forbidden in (
            "aten::complex",
            "aten::real",
            "aten::imag",
            "ComplexFloat",
        ):
            self.assertNotIn(forbidden, graph)

    def test_strict_load_rejects_missing_weights(self) -> None:
        reference = InvariantPatchCNN(self.complex_config).eval()
        state = dict(reference.state_dict())
        state.pop("fc2.bias")
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            make_paired_real_branch(self.complex_config, state)


if __name__ == "__main__":
    unittest.main()
