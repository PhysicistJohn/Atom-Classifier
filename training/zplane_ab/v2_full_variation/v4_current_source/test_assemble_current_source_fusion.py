from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_current_source_dev as branch_runner  # noqa: E402
from assemble_current_source_fusion import (  # noqa: E402
    ALPHA_COMPLEX,
    ALPHA_REAL,
    _load_moments,
    _sha256,
    _validate_no_consumed_provenance,
    assert_branches_match,
    build_parser,
    fit_center,
    fuse_numpy,
    load_branch,
    reject_sensitive_path,
    validate_branch_metrics,
    validate_branch_weight,
)
from invariant_patch_cnn import (  # noqa: E402
    InvariantPatchCNN,
    InvariantPatchConfig,
)
import time_domain_invariant_patch_preprocess as td_preprocess  # noqa: E402


def _metrics(encoder: str) -> dict:
    config = InvariantPatchConfig(
        encoder=encoder,
        patch_length=64,
        patch_count=16,
        patch_dim=16,
        hidden=20,
        dropout=0.0,
    )
    return {
        "status": "complete",
        "development_only": True,
        "release_evidence": False,
        "consumed_historical_test_rows_used": 0,
        "encoder": encoder,
        "architecture": config.__dict__,
        "parameter_count": 1,
        "run_configuration": {
            "schema": "v4-current-source-development-training-v1",
            "arguments": {
                "encoder": encoder,
                "seed": 71,
                "episodes": 2,
                "patch_length": 64,
                "patch_count": 16,
                "target_frac": 0.5,
                "current_corpus": "/current",
                "historical_corpus": "/historical",
                "current_source_share": {
                    "numerator": 1,
                    "denominator": 3,
                    "value": 1 / 3,
                },
            },
            "checkpoint_selection": {
                "contract": branch_runner.CHECKPOINT_SCORE_CONTRACT,
                "comparison": "strict Python tuple lexicographic greater-than",
                "predeclared": True,
                "tunable": False,
            },
            "optimizer": {"name": "AdamW"},
            "scheduler": {"name": "test"},
            "randomness": {"seed": 71, "phase_augmentation": True},
            "software": {
                "python": "test",
                "numpy": "test",
                "torch": "test",
                "requested_device": "cpu",
                "resolved_device": "cpu",
            },
        },
        "training": {
            "episodes": 2,
            "sampler_contract": branch_runner.SAMPLER_CONTRACT,
            "current_source_share": {
                "numerator": 1,
                "denominator": 3,
                "value": 1 / 3,
            },
        },
        "final_evaluation": {},
        "data_audit": {
            "historical": {
                "consumed_test_rows_exposed": 0,
                "manifest_sha256": "historical-manifest",
            },
            "current": {"manifest_sha256": "current-manifest"},
            "preprocessing": {
                "frontend": td_preprocess.preprocess_metadata(),
                "training": {
                    "hierarchical_sampler_contract":
                        branch_runner.SAMPLER_CONTRACT,
                },
                "runtime_bucket_policy": {
                    "episode_rule": branch_runner.SAMPLER_CONTRACT,
                },
            },
            "historical_contract": {"classes": ["a", "b"]},
            "consumed_test_rows_exposed": 0,
        },
        "artifacts": {
            "state_dict": f"{encoder}_state_dict.pt",
            "state_dict_sha256": "",
            "combined_prototypes": "combined_prototypes.npy",
            "combined_prototypes_sha256": "",
            "combined_prototype_bank": "combined_prototype_bank.json",
            "combined_prototype_bank_sha256": "",
            "feature_moments": "feature_moments.npz",
            "feature_moments_sha256": "",
        },
        "source_sha256": {"run_current_source_dev.py": "same"},
        "device": "cpu",
    }


def _write_branch(directory: Path, encoder: str) -> dict:
    directory.mkdir()
    metrics = _metrics(encoder)
    config = InvariantPatchConfig(**metrics["architecture"]).validate()
    net = InvariantPatchCNN(config)
    state_path = directory / metrics["artifacts"]["state_dict"]
    torch.save(net.state_dict(), state_path)
    prototype_path = directory / "combined_prototypes.npy"
    np.save(
        prototype_path,
        np.zeros((2, config.embed_dim), dtype=np.float32),
    )
    prototype_contract_path = directory / "combined_prototype_bank.json"
    prototype_contract_path.write_text(
        json.dumps({"groups": []}),
        encoding="utf-8",
    )
    moment_path = directory / "feature_moments.npz"
    np.savez(
        moment_path,
        mean=np.zeros(config.n_features, dtype=np.float32),
        std=np.ones(config.n_features, dtype=np.float32),
    )
    for name, hash_name in (
        ("state_dict", "state_dict_sha256"),
        ("combined_prototypes", "combined_prototypes_sha256"),
        ("combined_prototype_bank", "combined_prototype_bank_sha256"),
        ("feature_moments", "feature_moments_sha256"),
    ):
        metrics["artifacts"][hash_name] = _sha256(
            directory / metrics["artifacts"][name]
        )
    (directory / "dev_metrics.json").write_text(
        json.dumps(metrics),
        encoding="utf-8",
    )
    return metrics


class V4FusionAssemblyTests(unittest.TestCase):
    def test_fusion_constants_and_numpy_geometry(self) -> None:
        self.assertEqual(ALPHA_REAL, 0.2)
        self.assertEqual(ALPHA_COMPLEX, 0.2)
        real = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        complex_embedding = np.asarray(
            [[0.0, 1.0], [1.0, 0.0]], dtype=np.float32
        )
        fused = fuse_numpy(
            real,
            complex_embedding,
            np.asarray([0.2, 0.1], dtype=np.float32),
            np.asarray([0.1, 0.2], dtype=np.float32),
            weight_real=0.5,
        )
        self.assertEqual(fused.shape, (2, 4))
        np.testing.assert_allclose(
            np.linalg.norm(fused, axis=1),
            np.ones(2),
            atol=1e-6,
        )
        parsed = build_parser().parse_args(
            [
                "--real-dir",
                "real",
                "--complex-dir",
                "complex",
                "--current-corpus",
                "current",
                "--output-dir",
                "output",
            ]
        )
        self.assertEqual(parsed.branch_weight, 0.5)

    def test_center_reads_training_only(self) -> None:
        expected = np.asarray([2.0, 3.0], dtype=np.float32)
        actual = fit_center(
            {
                "training": np.asarray(
                    [[1.0, 2.0], [3.0, 4.0]], dtype=np.float32
                ),
                "enrollment": np.asarray([[999.0, 999.0]], dtype=np.float32),
                "selection": np.asarray([[-999.0, -999.0]], dtype=np.float32),
            }
        )
        np.testing.assert_array_equal(actual, expected)

    def test_weight_and_sensitive_path_fail_closed(self) -> None:
        self.assertEqual(validate_branch_weight("0.25"), 0.25)
        for value in (-0.1, 1.1, float("nan"), "bad"):
            with self.assertRaises(ValueError):
                validate_branch_weight(value)
        for path in (
            "/tmp/releases/candidate",
            "/tmp/release/candidate",
            "/tmp/release_candidate/candidate",
            "/tmp/sealed-suite/candidate",
            "/tmp/consumed_test/rows",
        ):
            with self.assertRaisesRegex(ValueError, "refuses"):
                reject_sensitive_path(path, "test")

    def test_branch_contract_requires_exact_source_and_sampler(self) -> None:
        real = {"metrics": _metrics("real")}
        complex_branch = {"metrics": _metrics("complex")}
        matched = assert_branches_match(real, complex_branch)
        self.assertEqual(
            matched["training_contract"]["sampler_contract"],
            branch_runner.SAMPLER_CONTRACT,
        )

        changed_source = copy.deepcopy(complex_branch)
        changed_source["metrics"]["source_sha256"]["run_current_source_dev.py"] = (
            "different"
        )
        with self.assertRaisesRegex(ValueError, "source_sha256"):
            assert_branches_match(real, changed_source)

        changed_sampler = copy.deepcopy(complex_branch)
        changed_sampler["metrics"]["training"]["sampler_contract"] = "changed"
        with self.assertRaisesRegex(ValueError, "sampler"):
            assert_branches_match(real, changed_sampler)

        changed_share = copy.deepcopy(complex_branch)
        changed_share["metrics"]["run_configuration"]["arguments"][
            "current_source_share"
        ] = {"numerator": 1, "denominator": 2, "value": 0.5}
        with self.assertRaisesRegex(ValueError, "argument contracts"):
            assert_branches_match(real, changed_share)

    def test_metrics_reject_consumed_or_non_fft_free_provenance(self) -> None:
        metrics = _metrics("real")
        validate_branch_metrics(metrics, encoder="real")

        inconsistent_share = copy.deepcopy(metrics)
        inconsistent_share["training"]["current_source_share"] = {
            "numerator": 1,
            "denominator": 2,
            "value": 0.5,
        }
        with self.assertRaisesRegex(ValueError, "share disagrees"):
            validate_branch_metrics(inconsistent_share, encoder="real")

        consumed = copy.deepcopy(metrics)
        consumed["data_audit"]["historical"]["consumed_test_rows_exposed"] = 1
        with self.assertRaisesRegex(ValueError, "consumed"):
            validate_branch_metrics(consumed, encoder="real")

        wrong_frontend = copy.deepcopy(metrics)
        wrong_frontend["data_audit"]["preprocessing"]["frontend"][
            "uses_frequency_transform"
        ] = True
        with self.assertRaisesRegex(ValueError, "frontend"):
            validate_branch_metrics(wrong_frontend, encoder="real")

    def test_withheld_consumed_test_count_is_evidence_of_nonexposure(self) -> None:
        _validate_no_consumed_provenance(
            {"counts": {"consumed_test_held_out": 1_915}}
        )
        for key in (
            "consumed_test_rows_used",
            "consumed_test_rows_loaded",
            "consumed_test_rows_exposed",
        ):
            with self.subTest(key=key):
                with self.assertRaisesRegex(RuntimeError, "forbidden exposure"):
                    _validate_no_consumed_provenance({key: 1})

    def test_load_branch_verifies_hashes_and_strict_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            branch_dir = Path(temporary) / "branch"
            _write_branch(branch_dir, "real")
            branch = load_branch(branch_dir, "real")
            self.assertEqual(branch["net"].cfg.encoder, "real")
            np.testing.assert_array_equal(
                branch["moments"][1],
                np.ones(branch["net"].cfg.n_features, dtype=np.float32),
            )

            metrics_path = branch_dir / "dev_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["artifacts"]["state_dict_sha256"] = "0" * 64
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                load_branch(branch_dir, "real")

    def test_moment_archive_requires_exact_mean_std_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            good = root / "good.npz"
            np.savez(
                good,
                mean=np.zeros(3, dtype=np.float32),
                std=np.ones(3, dtype=np.float32),
            )
            mean, std = _load_moments(good)
            self.assertEqual(mean.shape, (3,))
            self.assertEqual(std.shape, (3,))

            extra = root / "extra.npz"
            np.savez(
                extra,
                mean=np.zeros(3, dtype=np.float32),
                std=np.ones(3, dtype=np.float32),
                ignored=np.ones(3, dtype=np.float32),
            )
            with self.assertRaisesRegex(ValueError, "exactly mean and std"):
                _load_moments(extra)


if __name__ == "__main__":
    unittest.main()
