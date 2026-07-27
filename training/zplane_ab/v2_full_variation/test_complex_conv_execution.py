"""Parity and compatibility tests for fused ComplexConv1d execution."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from complex_multiscale_backbone import (  # noqa: E402
    ComplexConv1d,
    set_complex_conv_execution,
)
from unet_transfer import TransferUNet  # noqa: E402


CORRECTED_BUNDLE = (
    HERE
    / "artifacts"
    / "ground_state"
    / "corrected_resampled_cropoff_seed20260727"
    / "inference_bundle.pt"
)


def _module(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    execution: str,
) -> ComplexConv1d:
    return ComplexConv1d(
        in_channels,
        out_channels,
        kernel_size,
        execution=execution,
    )


class ComplexConvExecutionTest(unittest.TestCase):
    def test_forward_and_backward_match_legacy_within_one_e_minus_six(self):
        shapes = (
            (1, 8, 127, 7),
            (8, 16, 65, 7),
            (64, 128, 33, 7),
            (8, 8, 64, 1),
        )
        for index, (cin, cout, length, taps) in enumerate(shapes):
            with self.subTest(shape=(cin, cout, length, taps)):
                torch.manual_seed(100 + index)
                legacy = _module(cin, cout, taps, "legacy")
                fused = _module(cin, cout, taps, "fused_grouped")
                fused.load_state_dict(legacy.state_dict(), strict=True)

                real_a = torch.randn(3, cin, length, requires_grad=True)
                imag_a = torch.randn(3, cin, length, requires_grad=True)
                real_b = real_a.detach().clone().requires_grad_(True)
                imag_b = imag_a.detach().clone().requires_grad_(True)
                z_a = torch.complex(real_a, imag_a)
                z_b = torch.complex(real_b, imag_b)

                expected = legacy(z_a)
                actual = fused(z_b)
                torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-6)

                grad_real = torch.randn_like(expected.real)
                grad_imag = torch.randn_like(expected.imag)
                # Training losses in this codebase are means. An unnormalised sum makes
                # absolute gradient tolerances scale arbitrarily with tensor size.
                loss_a = (expected.real * grad_real + expected.imag * grad_imag).mean()
                loss_b = (actual.real * grad_real + actual.imag * grad_imag).mean()
                loss_a.backward()
                loss_b.backward()

                torch.testing.assert_close(real_b.grad, real_a.grad, rtol=0.0, atol=1e-6)
                torch.testing.assert_close(imag_b.grad, imag_a.grad, rtol=0.0, atol=1e-6)
                torch.testing.assert_close(
                    fused.conv_re.weight.grad,
                    legacy.conv_re.weight.grad,
                    rtol=0.0,
                    atol=1e-6,
                )
                torch.testing.assert_close(
                    fused.conv_im.weight.grad,
                    legacy.conv_im.weight.grad,
                    rtol=0.0,
                    atol=1e-6,
                )

    def test_state_dict_is_identical_and_execution_is_not_persistent(self):
        legacy = _module(4, 7, 5, "legacy")
        fused = _module(4, 7, 5, "fused_grouped")
        self.assertEqual(set(legacy.state_dict()), set(fused.state_dict()))
        self.assertNotIn("execution", fused.state_dict())
        fused.load_state_dict(legacy.state_dict(), strict=True)
        legacy.load_state_dict(fused.state_dict(), strict=True)
        auto = _module(4, 7, 5, "auto")
        z = torch.complex(torch.randn(2, 4, 31), torch.randn(2, 4, 31))
        self.assertEqual(auto._resolved_execution(z), "fused_grouped")
        with torch.no_grad():
            self.assertEqual(auto._resolved_execution(z), "fused_grouped")

    def test_phase_equivariance_and_runtime_fallback(self):
        torch.manual_seed(31)
        layer = _module(5, 9, 7, "fused_grouped").eval()
        z = torch.complex(torch.randn(4, 5, 97), torch.randn(4, 5, 97))
        phase = torch.tensor(0.731)
        rotation = torch.exp(1j * phase)
        with torch.no_grad():
            expected = rotation * layer(z)
            actual = layer(rotation * z)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-6)

        layer.set_execution("legacy")
        self.assertEqual(layer.execution, "legacy")
        with self.assertRaisesRegex(ValueError, "unknown complex-conv execution"):
            layer.set_execution("not-an-execution")

        net = TransferUNet(
            base=2,
            depth=2,
            taps=3,
            hidden=16,
            n_profiles=3,
            magnorm=True,
        ).eval()
        set_complex_conv_execution(net, "fused_grouped")
        iq = torch.randn(3, 2, 128)
        features = torch.randn(3, 12)
        complex_iq = torch.complex(iq[:, 0], iq[:, 1])
        rotated_iq = rotation * complex_iq
        with torch.no_grad():
            embedding_a = net(iq, features)
            reconstruction_a = net.last_recon.clone()
            embedding_b = net(
                torch.stack((rotated_iq.real, rotated_iq.imag), dim=1),
                features,
            )
            reconstruction_b = net.last_recon
        relative_reconstruction_error = float(
            (reconstruction_b - rotation * reconstruction_a).abs().max()
            / reconstruction_a.abs().max().clamp_min(1e-12)
        )
        self.assertLess(relative_reconstruction_error, 1e-4)
        torch.testing.assert_close(
            embedding_b, embedding_a, rtol=0.0, atol=1e-4
        )

    def test_transfer_unet_switches_every_layer_without_changing_embedding(self):
        torch.manual_seed(41)
        net = TransferUNet(
            base=2,
            depth=2,
            taps=3,
            hidden=16,
            n_profiles=3,
            magnorm=True,
            residual_recon=True,
        ).eval()
        iq = torch.randn(4, 2, 128)
        features = torch.randn(4, 12)
        count = set_complex_conv_execution(net, "legacy")
        self.assertGreater(count, 0)
        keys = set(net.state_dict())
        with torch.no_grad():
            expected = net.forward_embedding(iq, features)
        self.assertEqual(set_complex_conv_execution(net, "fused_grouped"), count)
        self.assertEqual(set(net.state_dict()), keys)
        with torch.no_grad():
            actual = net.forward_embedding(iq, features)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-6)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_mps_forward_backward_and_embedding_parity(self):
        device = torch.device("mps")
        torch.manual_seed(53)
        torch.mps.manual_seed(53)
        legacy = _module(64, 128, 7, "legacy").to(device)
        fused = _module(64, 128, 7, "fused_grouped").to(device)
        fused.load_state_dict(legacy.state_dict(), strict=True)

        real_a = torch.randn(3, 64, 64, device=device, requires_grad=True)
        imag_a = torch.randn(3, 64, 64, device=device, requires_grad=True)
        real_b = real_a.detach().clone().requires_grad_(True)
        imag_b = imag_a.detach().clone().requires_grad_(True)
        expected = legacy(torch.complex(real_a, imag_a))
        actual = fused(torch.complex(real_b, imag_b))
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-6)
        grad_real = torch.randn_like(expected.real)
        grad_imag = torch.randn_like(expected.imag)
        (expected.real * grad_real + expected.imag * grad_imag).mean().backward()
        (actual.real * grad_real + actual.imag * grad_imag).mean().backward()
        torch.mps.synchronize()
        for left, right in (
            (real_b.grad, real_a.grad),
            (imag_b.grad, imag_a.grad),
            (fused.conv_re.weight.grad, legacy.conv_re.weight.grad),
            (fused.conv_im.weight.grad, legacy.conv_im.weight.grad),
        ):
            torch.testing.assert_close(left, right, rtol=0.0, atol=1e-6)

        auto = _module(2, 3, 3, "auto").to(device)
        auto_input = torch.complex(
            torch.randn(1, 2, 16, device=device),
            torch.randn(1, 2, 16, device=device),
        )
        self.assertEqual(auto._resolved_execution(auto_input), "fused_grouped")
        with torch.no_grad():
            self.assertEqual(auto._resolved_execution(auto_input), "legacy")

        net = TransferUNet(
            base=2,
            depth=2,
            taps=3,
            hidden=16,
            n_profiles=3,
            magnorm=True,
            residual_recon=True,
        ).to(device).eval()
        iq = torch.randn(4, 2, 128, device=device)
        features = torch.randn(4, 12, device=device)
        set_complex_conv_execution(net, "legacy")
        with torch.no_grad():
            expected_embedding = net.forward_embedding(iq, features)
        set_complex_conv_execution(net, "fused_grouped")
        with torch.no_grad():
            actual_embedding = net.forward_embedding(iq, features)
        torch.testing.assert_close(
            actual_embedding, expected_embedding, rtol=0.0, atol=1e-6
        )

    @unittest.skipUnless(
        CORRECTED_BUNDLE.exists(),
        "local corrected checkpoint artifact is not present",
    )
    def test_trained_checkpoint_loads_strictly_without_execution_state(self):
        bundle = torch.load(CORRECTED_BUNDLE, map_location="cpu", weights_only=False)
        net = TransferUNet(**bundle["architecture"])
        incompatible = net.load_state_dict(bundle["state_dict"], strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        modes = {
            child.execution
            for child in net.modules()
            if isinstance(child, ComplexConv1d)
        }
        self.assertEqual(modes, {"auto"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
