from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from export_invariant_fusion import (
    E2E_FIXTURE_NAME,
    FIXTURE_NAME,
    MANIFEST_NAME,
    WEIGHTS_NAME,
    _validate_bundle,
    _validate_output_path,
    export_staging,
)
from invariant_patch_cnn import InvariantPatchCNN, InvariantPatchConfig
from known_only_patch_openset import KnownOnlyLOFOpenSet


def _component(
    branch: str,
    *,
    embed_dim: int,
    neighbors: int,
    weight: float,
) -> dict:
    rng = np.random.default_rng(10 if branch == "real" else 20)
    rows = 80
    reference = rng.standard_normal((rows, embed_dim))
    return {
        "branch": branch,
        "weight": weight,
        "neighbors": neighbors,
        "embedding_reference": torch.from_numpy(reference),
        "embedding_mean": torch.from_numpy(
            rng.standard_normal(embed_dim) * 0.1
        ),
        "embedding_scale": torch.from_numpy(
            0.8 + rng.random(embed_dim)
        ),
        "reference_k_distance": torch.from_numpy(
            0.05 + rng.random(rows)
        ),
        "reference_local_density": torch.from_numpy(
            0.2 + rng.random(rows)
        ),
        "sorted_calibration_raw": torch.from_numpy(
            np.sort(0.5 + rng.random(50))
        ),
        "selection_or_novelty_used_in_fit": False,
    }


def _bundle() -> dict:
    torch.manual_seed(20260727)
    common = dict(
        patch_length=16,
        patch_count=2,
        patch_dim=8,
        hidden=12,
        embed_dim=32,
        n_features=12,
        dropout=0.0,
    )
    real_config = InvariantPatchConfig(encoder="real", **common)
    complex_config = InvariantPatchConfig(encoder="complex", **common)
    real = InvariantPatchCNN(real_config).eval()
    complex_branch = InvariantPatchCNN(complex_config).eval()
    threshold = 0.90
    training = Path(__file__).resolve().parents[2]
    frontend_path = training / "invariant_patch_preprocess.py"
    preprocess_path = training / "preprocess.py"
    return {
        "schema": 2,
        "kind": "invariant_centered_fusion_classifier",
        "real_config": real.config(),
        "complex_config": complex_branch.config(),
        "real_state_dict": real.state_dict(),
        "complex_state_dict": complex_branch.state_dict(),
        "real_center": torch.linspace(-0.1, 0.2, 32),
        "complex_center": torch.linspace(0.15, -0.2, 32),
        "alpha_real": 0.2,
        "alpha_complex": 0.2,
        "weight_real": 0.6,
        "eps": 1e-12,
        "feature_mean": torch.linspace(-0.3, 0.5, 12),
        "feature_std": torch.linspace(0.4, 1.1, 12),
        "classes": ["alpha", "beta"],
        "prototypes": torch.randn(2, 64),
        # This is the weighted LOF-rank threshold, not a prototype-distance
        # threshold. The exporter deliberately emits null for the latter.
        "unknown_threshold": threshold,
        "temperature": 0.25,
        "open_set": {
            "kind": "known_only_lof_rank_ensemble",
            "components": [
                _component(
                    "real", embed_dim=32, neighbors=2, weight=0.4
                ),
                _component(
                    "complex", embed_dim=32, neighbors=64, weight=0.6
                ),
            ],
            "threshold": threshold,
            "threshold_quantile": 0.95,
            "selection_or_novelty_used_in_density_or_threshold_fit": False,
            "selection_or_novelty_used_for_hyperparameter_choice": True,
            "cannot_change_fused_closed_label": True,
        },
        "development_only": True,
        "provenance": {
            "release_evidence": False,
            "consumed_test_rows_used": 0,
            "center_fit_population": "training only",
            "prototype_population": "enrollment only",
            "unknown_calibration_population": "enrollment only",
            "cache_contract": {
                "schema": 1,
                "frontend": "invariant-patch-v1",
                "id": "synthetic-test-contract",
                "config": {
                    "patch_length": 16,
                    "patch_count": 2,
                    "target_frac": 0.5,
                },
                "source_sha256": {
                    "training/invariant_patch_preprocess.py": (
                        hashlib.sha256(frontend_path.read_bytes()).hexdigest()
                    ),
                    "training/preprocess.py": hashlib.sha256(
                        preprocess_path.read_bytes()
                    ).hexdigest(),
                },
            },
        },
    }


class InvariantFusionExporterTest(unittest.TestCase):
    def test_exports_checked_runtime_weights_and_three_case_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            bundle_path = source / "runtime_bundle.pt"
            output = root / "staging"
            source_bundle = _bundle()
            torch.save(source_bundle, bundle_path)

            manifest = export_staging(bundle_path, output)
            persisted = json.loads((output / MANIFEST_NAME).read_text())
            weights = json.loads((output / WEIGHTS_NAME).read_text())
            fixture = json.loads((output / FIXTURE_NAME).read_text())
            end_to_end = json.loads(
                (output / E2E_FIXTURE_NAME).read_text()
            )

            self.assertEqual(
                weights["schema"], "atomos.invariant-fusion.paired-real"
            )
            self.assertEqual(weights["status"], "staging_not_release")
            self.assertTrue(weights["provisional"])
            self.assertTrue(
                all(
                    block["batch_norm_folded"]
                    for block in weights["real"]["blocks"]
                )
            )
            state = source_bundle["real_state_dict"]
            gamma = state["patch_encoder.blocks.0.bn.weight"]
            variance = state["patch_encoder.blocks.0.bn.running_var"]
            mean = state["patch_encoder.blocks.0.bn.running_mean"]
            beta = state["patch_encoder.blocks.0.bn.bias"]
            scale = gamma / torch.sqrt(variance + 1e-5)
            expected_weight = (
                state["patch_encoder.blocks.0.conv.weight"]
                * scale[:, None, None]
            )
            expected_bias = beta - mean * scale
            np.testing.assert_allclose(
                weights["real"]["blocks"][0]["weight"],
                expected_weight.reshape(-1).numpy(),
                rtol=0.0,
                atol=0.0,
            )
            np.testing.assert_allclose(
                weights["real"]["blocks"][0]["bias"],
                expected_bias.numpy(),
                rtol=0.0,
                atol=0.0,
            )
            self.assertIsNone(
                weights["classification"]["unknown_threshold"]
            )
            self.assertEqual(weights["classification"]["temperature"], 0.25)
            self.assertEqual(
                [
                    (component["branch"], component["weight"], component["neighbors"])
                    for component in weights["rejection"]["components"]
                ],
                [("real", 0.4, 2), ("complex", 0.6, 64)],
            )
            self.assertEqual(len(fixture["case_names"]), 3)
            self.assertEqual(
                fixture["input"]["packed_iq"]["shape"], [3, 2, 32]
            )
            self.assertEqual(
                [case["raw"]["length"] for case in end_to_end["cases"]],
                [32768, 16384, 8192, 4096, 2048, 4096, 51, 777],
            )
            self.assertEqual(
                [case["physical_scale"] for case in end_to_end["cases"]],
                [0.125, 0.25, 0.5, 1.0, 2.0, 1.0, 1.0, 1.0],
            )
            self.assertTrue(
                all(
                    case["expected"]["context"]["packed_length"] == 32
                    for case in end_to_end["cases"]
                )
            )
            edge_cases = {
                case["name"]: case for case in end_to_end["cases"]
            }
            short_context = edge_cases["nextafter-short-repeat"]["expected"][
                "context"
            ]
            self.assertEqual(short_context["active_length"], 30)
            self.assertEqual(
                short_context["patches_repeated_from_short_active"],
                short_context["active_length"]
                < weights["real"]["config"]["patch_length"],
            )
            self.assertEqual(
                edge_cases["non-endpoint-u-grid"]["expected"]["context"][
                    "active_length"
                ],
                295,
            )
            self.assertEqual(weights["preprocess"]["estimator_version"], "hybrid-v3")
            self.assertEqual(weights["preprocess"]["nfft"], 512)
            self.assertEqual(weights["preprocess"]["feature_count"], 12)
            self.assertEqual(len(weights["preprocess"]["feature_names"]), 12)
            self.assertEqual(end_to_end["held_out_rows_loaded"], 0)
            self.assertFalse(end_to_end["data_or_corpus_loaded"])
            self.assertEqual(
                [row["branch"] for row in fixture["expected"]["rejection_components"]],
                ["real", "complex"],
            )
            weighted = np.zeros(3)
            for expected_component, source_component in zip(
                fixture["expected"]["rejection_components"],
                source_bundle["open_set"]["components"],
            ):
                calibration = source_component["sorted_calibration_raw"].numpy()
                scorer = KnownOnlyLOFOpenSet(
                    embedding_reference=source_component[
                        "embedding_reference"
                    ].numpy(),
                    embedding_mean=source_component["embedding_mean"].numpy(),
                    embedding_scale=source_component["embedding_scale"].numpy(),
                    reference_k_distance=source_component[
                        "reference_k_distance"
                    ].numpy(),
                    reference_local_density=source_component[
                        "reference_local_density"
                    ].numpy(),
                    calibration=calibration,
                    calibration_scores=np.zeros_like(calibration),
                    neighbors=source_component["neighbors"],
                )
                branch_embedding = np.asarray(
                    fixture["expected"][
                        f"{expected_component['branch']}_embedding"
                    ]
                )
                np.testing.assert_allclose(
                    expected_component["rank"],
                    scorer.score(branch_embedding),
                    rtol=0.0,
                    atol=0.0,
                )
                weighted += expected_component["weight"] * np.asarray(
                    expected_component["rank"]
                )
            np.testing.assert_allclose(
                fixture["expected"]["weighted_rejection_score"],
                weighted,
                rtol=0.0,
                atol=0.0,
            )
            self.assertLessEqual(
                max(
                    fixture["native_vs_paired_real_max_abs_error"].values()
                ),
                fixture["parity_limit"],
            )
            self.assertEqual(
                persisted["weights"]["sha256"],
                hashlib.sha256((output / WEIGHTS_NAME).read_bytes()).hexdigest(),
            )
            self.assertEqual(
                persisted["end_to_end_parity_fixture"]["sha256"],
                hashlib.sha256(
                    (output / E2E_FIXTURE_NAME).read_bytes()
                ).hexdigest(),
            )
            self.assertEqual(manifest, persisted)
            self.assertFalse(
                persisted["validation"]["data_or_corpus_loaded"]
            )
            self.assertEqual(
                persisted["validation"]["held_out_rows_loaded"], 0
            )

            with self.assertRaises(FileExistsError):
                export_staging(bundle_path, output)

    def test_rejects_wrong_frozen_policy_and_nonfinite_state(self):
        wrong_policy = _bundle()
        wrong_policy["open_set"]["components"][1]["neighbors"] = 5
        with self.assertRaisesRegex(ValueError, "k=64"):
            _validate_bundle(wrong_policy)

        nonfinite = _bundle()
        nonfinite["real_state_dict"] = dict(nonfinite["real_state_dict"])
        nonfinite["real_state_dict"]["fc2.bias"] = torch.full(
            (32,), float("nan")
        )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            _validate_bundle(nonfinite)

    def test_rejects_live_src_and_source_bundle_descendants(self):
        source = Path("/tmp/runtime-source/runtime_bundle.pt").resolve()
        with self.assertRaisesRegex(ValueError, "live src"):
            _validate_output_path(
                source,
                (
                    Path(__file__).resolve().parents[3]
                    / "src"
                    / "embedding"
                    / "assets"
                    / "invariant"
                ).resolve(),
            )
        with self.assertRaisesRegex(ValueError, "source bundle"):
            _validate_output_path(
                source,
                (source.parent / "staging").resolve(),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
