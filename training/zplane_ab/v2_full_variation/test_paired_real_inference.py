"""Parity and export tests for the paired-real TransferUNet deployment seam."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from export_transfer_unet import (  # noqa: E402
    _execute_onnx_runtime,
    export_staging,
)
from paired_real_inference import (  # noqa: E402
    PairedRealTransferUNet,
    make_paired_real_model,
)
from unet_transfer import TransferUNet  # noqa: E402


CORRECTED_CHECKPOINT = (
    HERE
    / "artifacts"
    / "ground_state"
    / "corrected_resampled_cropoff_seed20260727"
    / "inference_bundle.pt"
)

CORRECTED_ARCHITECTURE = {
    "base": 8,
    "depth": 4,
    "taps": 7,
    "embed_dim": 32,
    "n_features": 12,
    "n_profiles": 37,
    "hidden": 128,
    "magnorm": True,
    "modrelu_init": -2.0,
    "skip_dropout": 0.5,
    "feat_dropout": 0.5,
    "antialias": False,
    "head_pool": "meanstd",
    "residual_recon": True,
}


def _run_reference(
    reference: TransferUNet,
    paired: torch.nn.Module,
    iq: torch.Tensor,
    features: torch.Tensor,
) -> dict[str, float]:
    reference.eval()
    paired.eval()
    with torch.no_grad():
        embedding = reference(iq, features)
        expected = (
            embedding,
            reference.last_bottleneck_pool,
            reference.last_recon.real,
            reference.last_recon.imag,
        )
        actual = paired(iq, features)
    return {
        name: float((left - right).abs().max())
        for name, left, right in zip(
            ("embedding", "bottleneck", "denoised_real", "denoised_imaginary"),
            expected,
            actual,
        )
    }


class PairedRealInferenceTest(unittest.TestCase):
    def test_random_corrected_architecture_matches_complex_reference(self):
        torch.manual_seed(20260727)
        reference = TransferUNet(**CORRECTED_ARCHITECTURE).eval()
        paired = make_paired_real_model(
            CORRECTED_ARCHITECTURE,
            reference.state_dict(),
        )
        self.assertEqual(set(reference.state_dict()), set(paired.state_dict()))
        self.assertFalse(
            any(torch.is_complex(parameter) for parameter in paired.parameters())
        )

        generator = torch.Generator().manual_seed(19)
        iq = torch.randn(2, 2, 1024, generator=generator)
        features = torch.randn(2, 12, generator=generator)
        errors = _run_reference(reference, paired, iq, features)
        self.assertLessEqual(max(errors.values()), 2e-6, errors)

    @unittest.skipUnless(
        CORRECTED_CHECKPOINT.exists(),
        "local corrected checkpoint artifact is not present",
    )
    def test_trained_corrected_checkpoint_matches_complex_reference(self):
        bundle = torch.load(
            CORRECTED_CHECKPOINT,
            map_location="cpu",
            weights_only=False,
        )
        reference = TransferUNet(**bundle["architecture"])
        reference.load_state_dict(bundle["state_dict"], strict=True)
        paired = make_paired_real_model(
            bundle["architecture"],
            bundle["state_dict"],
        )

        generator = torch.Generator().manual_seed(23)
        iq = torch.randn(3, 2, 1024, generator=generator)
        features = torch.randn(3, 12, generator=generator)
        errors = _run_reference(reference, paired, iq, features)
        self.assertLessEqual(max(errors.values()), 2e-6, errors)

    def test_traced_inference_graph_contains_no_complex_operators(self):
        architecture = {
            **CORRECTED_ARCHITECTURE,
            "base": 2,
            "depth": 2,
            "taps": 3,
            "embed_dim": 8,
            "n_features": 4,
            "n_profiles": 3,
            "hidden": 12,
        }
        torch.manual_seed(5)
        reference = TransferUNet(**architecture).eval()
        paired = make_paired_real_model(architecture, reference.state_dict())
        iq = torch.zeros(1, 2, 128)
        features = torch.zeros(1, 4)
        traced = torch.jit.trace(paired, (iq, features), strict=True)
        graph = str(traced.inlined_graph)
        for token in ("aten::complex", "aten::real", "aten::imag", "ComplexFloat"):
            self.assertNotIn(token, graph)

    def test_unsupported_experimental_pool_is_rejected_explicitly(self):
        with self.assertRaisesRegex(
            ValueError,
            "unsupported head_pool 'acf_complex'",
        ):
            PairedRealTransferUNet(head_pool="acf_complex")

    def test_missing_onnxruntime_is_reported_as_not_run(self):
        with mock.patch.dict(sys.modules, {"onnxruntime": None}):
            report = _execute_onnx_runtime(
                Path("not-opened-when-runtime-is-absent.onnx"),
                torch.zeros(1, 2, 8),
                torch.zeros(1, 4),
                {},
            )
        self.assertEqual(report["status"], "not_run")
        self.assertIn("not installed", report["reason"])
        self.assertIsNone(report["onnxruntime_version"])
        self.assertIsNone(report["max_abs_error"])
        self.assertIsNone(report["latency_ms"])

    def test_exporter_writes_checked_versioned_staging_artifact(self):
        try:
            import onnx  # noqa: F401
        except ImportError:
            self.skipTest("onnx package is not installed")

        architecture = {
            **CORRECTED_ARCHITECTURE,
            "base": 2,
            "depth": 2,
            "taps": 3,
            "embed_dim": 8,
            "n_features": 4,
            "n_profiles": 3,
            "hidden": 12,
        }
        torch.manual_seed(11)
        reference = TransferUNet(**architecture).eval()
        bundle = {
            "schema_version": 1,
            "model": "TransferUNet",
            "architecture": architecture,
            "state_dict": reference.state_dict(),
            "classes": ["alpha", "beta"],
            "prototypes": torch.randn(2, 8),
            "feature_mean": torch.zeros(4),
            "feature_std": torch.ones(4),
            "open_set_threshold": 0.25,
            "preprocessing": {
                "condition": "resampled",
                "input_length": 128,
                "module": "training/preprocess.py",
            },
            "target_policy": "test normalized carrier frame",
            "scalar_transfer_head": None,
        }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "bundle.pt"
            output = root / "staging-v1"
            torch.save(bundle, checkpoint)
            authored_sha = "0" * 64
            (root / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "environment": {
                            "source_sha256": {
                                "training/preprocess.py": authored_sha,
                            }
                        }
                    }
                )
            )
            manifest = export_staging(
                checkpoint,
                output,
                raw_capture_length=4096,
            )
            persisted = json.loads((output / "manifest.json").read_text())

            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(persisted["schema"], "atomos.transfer-unet.paired-real")
            self.assertEqual(persisted["status"], "staging_not_release")
            self.assertEqual(persisted["inputs"]["raw_capture_length_required"], 4096)
            self.assertIsNone(persisted["classification"]["temperature"])
            self.assertFalse(persisted["classification"]["calibration_ready"])
            self.assertEqual(persisted["onnx"]["checker"], "passed")
            execution = persisted["onnx"]["execution"]
            try:
                import onnxruntime as ort
            except ImportError:
                self.assertEqual(execution["status"], "not_run")
                self.assertIn("not installed", execution["reason"])
            else:
                self.assertEqual(execution["status"], "passed")
                self.assertEqual(
                    execution["onnxruntime_version"],
                    ort.__version__,
                )
                self.assertEqual(
                    execution["execution_provider"],
                    "CPUExecutionProvider",
                )
                self.assertLessEqual(
                    max(execution["max_abs_error"].values()),
                    execution["parity_limit"],
                )
                self.assertGreater(execution["latency_ms"]["timed_runs"], 0)
                self.assertGreater(execution["latency_ms"]["median"], 0)
            provenance = persisted["preprocessing"]
            self.assertFalse(provenance["compatible_with_checkpoint"])
            self.assertEqual(
                provenance["checkpoint_authoring"]["source_sha256"],
                authored_sha,
            )
            self.assertIsNone(
                provenance["checkpoint_authoring"]["parameters"],
            )
            self.assertNotEqual(
                provenance["exporter_environment"]["source_sha256"],
                authored_sha,
            )
            self.assertEqual(
                provenance["exporter_environment"]["parameters"]["l_out"],
                1024,
            )
            self.assertIn(
                "checkpoint preprocessing source is incompatible",
                " ".join(persisted["release_blockers"]),
            )
            self.assertGreater((output / persisted["onnx"]["file"]).stat().st_size, 0)
            self.assertGreater(
                (output / persisted["parity_fixture"]["file"]).stat().st_size,
                0,
            )
            self.assertLessEqual(
                max(persisted["parity"]["max_abs_error"].values()),
                persisted["parity"]["limit"],
            )

            with self.assertRaises(FileExistsError):
                export_staging(
                    checkpoint,
                    output,
                    raw_capture_length=4096,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
