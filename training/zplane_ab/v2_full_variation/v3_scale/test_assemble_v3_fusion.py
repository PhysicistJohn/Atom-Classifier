from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import assemble_v3_fusion as subject  # noqa: E402
from invariant_fusion import CenteredInvariantFusion  # noqa: E402
from invariant_patch_cnn import (  # noqa: E402
    InvariantPatchCNN,
    InvariantPatchConfig,
)


def _models() -> tuple[InvariantPatchCNN, InvariantPatchCNN]:
    common = {
        "patch_length": 16,
        "patch_count": 2,
        "patch_dim": 8,
        "hidden": 12,
        "embed_dim": 4,
        "n_features": 3,
        "dropout": 0.0,
    }
    return (
        InvariantPatchCNN(InvariantPatchConfig(encoder="real", **common)).eval(),
        InvariantPatchCNN(
            InvariantPatchConfig(encoder="complex", **common)
        ).eval(),
    )


class FittingContractTests(unittest.TestCase):
    def test_centers_use_training_rows_only(self) -> None:
        rng = np.random.default_rng(11)
        train = rng.normal(size=(9, 4)).astype(np.float32)
        poisoned = np.full((5, 4), np.nan, dtype=np.float32)
        center = subject.fit_center(
            {"tr": train, "en": poisoned, "va": poisoned}
        )
        self.assertTrue(np.isfinite(center).all())
        np.testing.assert_allclose(
            center,
            train.astype(np.float64).mean(axis=0).astype(np.float32),
            rtol=0.0,
            atol=0.0,
        )

    def test_center_fitting_refuses_a_missing_training_split(self) -> None:
        with self.assertRaisesRegex(KeyError, "training"):
            subject.fit_center({"en": np.zeros((4, 4), dtype=np.float32)})
        with self.assertRaisesRegex(ValueError, "non-empty"):
            subject.fit_center({"tr": np.zeros((0, 4), dtype=np.float32)})

    def test_prototypes_use_enrollment_rows_only(self) -> None:
        rng = np.random.default_rng(12)
        enrollment = rng.normal(size=(8, 4)).astype(np.float32)
        labels = np.asarray([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int64)
        poisoned = np.full((8, 4), np.nan, dtype=np.float32)
        prototypes = subject.fit_prototypes(
            {"tr": poisoned, "en": enrollment, "va": poisoned}, labels, 2
        )
        self.assertTrue(np.isfinite(prototypes).all())
        for class_index in range(2):
            np.testing.assert_allclose(
                prototypes[class_index],
                enrollment[labels == class_index].mean(axis=0),
                rtol=0.0,
                atol=1e-7,
            )

    def test_prototype_fitting_refuses_missing_or_empty_classes(self) -> None:
        enrollment = np.zeros((4, 4), dtype=np.float32)
        with self.assertRaisesRegex(KeyError, "enrollment"):
            subject.fit_prototypes({"tr": enrollment}, np.zeros(4, np.int64), 2)
        with self.assertRaisesRegex(ValueError, "enrollment row"):
            subject.fit_prototypes(
                {"en": enrollment}, np.zeros(4, dtype=np.int64), 2
            )

    def test_metric_split_is_never_a_fitting_population(self) -> None:
        self.assertEqual(subject.CENTER_SPLIT, "tr")
        self.assertEqual(subject.PROTOTYPE_SPLIT, "en")
        self.assertEqual(subject.METRIC_SPLIT, "va")
        self.assertNotIn(
            subject.METRIC_SPLIT,
            {subject.CENTER_SPLIT, subject.PROTOTYPE_SPLIT},
        )


class SealedPathRefusalTests(unittest.TestCase):
    def test_reject_sealed_path_refuses_release_and_sealed_locations(self) -> None:
        for path in (
            "/tmp/releases/v3",
            "/tmp/Sealed-v3",
            "/tmp/a/invariant_fusion_v2_sealed_seed20260729",
            "/tmp/RELEASES/v3",
        ):
            with self.subTest(path=path):
                with self.assertRaisesRegex(ValueError, "release or sealed"):
                    subject.reject_sealed_path(Path(path), "output")

    def test_reject_sealed_path_allows_development_locations(self) -> None:
        resolved = subject.reject_sealed_path(
            Path("/tmp/artifacts/invariant_patch/v3_scale/fusion"), "output"
        )
        self.assertTrue(str(resolved).endswith("v3_scale/fusion"))

    def test_run_refuses_release_or_sealed_paths_before_any_load(self) -> None:
        cases = (
            {
                "output_dir": "/tmp/releases/v3_fusion",
                "real_dir": "/tmp/dev/real",
                "complex_dir": "/tmp/dev/complex",
            },
            {
                "output_dir": "/tmp/dev/out",
                "real_dir": "/tmp/artifacts/releases/real",
                "complex_dir": "/tmp/dev/complex",
            },
            {
                "output_dir": "/tmp/dev/out",
                "real_dir": "/tmp/dev/real",
                "complex_dir": "/tmp/sealed_seed20260729/complex",
            },
        )
        for case in cases:
            with self.subTest(**case), mock.patch.object(
                subject,
                "load_branch",
                side_effect=AssertionError("must reject path before load"),
            ), mock.patch.object(
                subject.invariant_data,
                "load",
                side_effect=AssertionError("must reject path before data load"),
            ):
                args = argparse.Namespace(
                    branch_weight=0.5,
                    prototype_tolerance=1e-4,
                    seed=20260730,
                    device="cpu",
                    **case,
                )
                with self.assertRaisesRegex(ValueError, "release or sealed"):
                    subject.run(args)

    def test_load_branch_refuses_a_sealed_source_directory(self) -> None:
        with self.assertRaisesRegex(ValueError, "release or sealed"):
            subject.load_branch(Path("/tmp/releases/real"), "real")


class BranchWeightTests(unittest.TestCase):
    def test_parser_default_is_half_and_complex_is_the_remainder(self) -> None:
        args = subject.build_parser().parse_args(
            [
                "--real-dir",
                "/tmp/real",
                "--complex-dir",
                "/tmp/complex",
                "--output-dir",
                "/tmp/out",
            ]
        )
        self.assertEqual(args.branch_weight, 0.5)
        self.assertEqual(subject.DEFAULT_WEIGHT_REAL, 0.5)
        self.assertEqual(
            subject.validate_branch_weight(args.branch_weight), 0.5
        )

    def test_branch_weight_validation_fails_closed(self) -> None:
        for bad in (-0.01, 1.01, float("nan"), float("inf"), "not-a-number"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    subject.validate_branch_weight(bad)

    def test_extreme_weights_select_a_single_branch(self) -> None:
        rng = np.random.default_rng(3)
        real = rng.normal(size=(6, 4)).astype(np.float32)
        complex_embedding = rng.normal(size=(6, 4)).astype(np.float32)
        centers = (
            rng.normal(size=4).astype(np.float32),
            rng.normal(size=4).astype(np.float32),
        )
        only_real = subject.fuse_numpy(
            real, complex_embedding, *centers, weight_real=1.0
        )
        only_complex = subject.fuse_numpy(
            real, complex_embedding, *centers, weight_real=0.0
        )
        np.testing.assert_allclose(only_real[:, 4:], 0.0, atol=1e-7)
        np.testing.assert_allclose(
            np.linalg.norm(only_real[:, :4], axis=1), 1.0, atol=1e-6
        )
        np.testing.assert_allclose(only_complex[:, :4], 0.0, atol=1e-7)
        np.testing.assert_allclose(
            np.linalg.norm(only_complex[:, 4:], axis=1), 1.0, atol=1e-6
        )

    def test_weight_sets_the_relative_energy_of_each_half(self) -> None:
        rng = np.random.default_rng(4)
        real = rng.normal(size=(5, 4)).astype(np.float32)
        complex_embedding = rng.normal(size=(5, 4)).astype(np.float32)
        centers = (
            np.zeros(4, dtype=np.float32),
            np.zeros(4, dtype=np.float32),
        )
        for weight in (0.25, 0.5, 0.6, 0.75):
            with self.subTest(weight=weight):
                fused = subject.fuse_numpy(
                    real, complex_embedding, *centers, weight_real=weight
                )
                real_energy = np.sum(fused[:, :4] ** 2, axis=1)
                complex_energy = np.sum(fused[:, 4:] ** 2, axis=1)
                np.testing.assert_allclose(real_energy, weight, atol=1e-6)
                np.testing.assert_allclose(
                    complex_energy, 1.0 - weight, atol=1e-6
                )

    def test_numpy_fusion_matches_the_torch_module_at_each_weight(self) -> None:
        torch.manual_seed(9)
        real_net, complex_net = _models()
        x = torch.randn(7, 2, real_net.packed_length)
        features = torch.randn(7, real_net.cfg.n_features)
        with torch.no_grad():
            real_embedding = real_net(x, features).numpy()
            complex_embedding = complex_net(x, features).numpy()
        real_center = subject.fit_center({"tr": real_embedding[:4]})
        complex_center = subject.fit_center({"tr": complex_embedding[:4]})
        for weight in (0.0, 0.25, 0.5, 0.6, 1.0):
            with self.subTest(weight=weight):
                fusion = CenteredInvariantFusion(
                    real_net,
                    complex_net,
                    real_center,
                    complex_center,
                    alpha_real=subject.ALPHA_REAL,
                    alpha_complex=subject.ALPHA_COMPLEX,
                    weight_real=weight,
                    eps=subject.FUSION_EPS,
                ).eval()
                with torch.no_grad():
                    module_output = fusion(x, features).numpy()
                numpy_output = subject.fuse_numpy(
                    real_embedding,
                    complex_embedding,
                    real_center,
                    complex_center,
                    weight_real=weight,
                )
                error = float(np.max(np.abs(module_output - numpy_output)))
                self.assertLess(error, subject.FUSION_ASSEMBLY_TOLERANCE)

    def test_fusion_embedder_exposes_the_branch_patch_geometry(self) -> None:
        torch.manual_seed(10)
        real_net, complex_net = _models()
        fusion = CenteredInvariantFusion(
            real_net,
            complex_net,
            np.zeros(real_net.cfg.embed_dim, dtype=np.float32),
            np.zeros(real_net.cfg.embed_dim, dtype=np.float32),
            weight_real=0.5,
        ).eval()
        embedder = subject.FusionEmbedder(fusion).eval()
        self.assertEqual(embedder.cfg.patch_length, real_net.cfg.patch_length)
        self.assertEqual(embedder.cfg.patch_count, real_net.cfg.patch_count)
        x = torch.randn(3, 2, real_net.packed_length)
        features = torch.randn(3, real_net.cfg.n_features)
        with torch.no_grad():
            np.testing.assert_array_equal(
                embedder(x, features).numpy(), fusion(x, features).numpy()
            )
        with self.assertRaises(TypeError):
            subject.FusionEmbedder(real_net)


class DeterminismTests(unittest.TestCase):
    @staticmethod
    def _assemble(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        subject.runner.seed_everything(seed)
        real_net, complex_net = _models()
        generator = torch.Generator().manual_seed(seed)
        x = torch.randn(
            12, 2, real_net.packed_length, generator=generator
        )
        features = torch.randn(
            12, real_net.cfg.n_features, generator=generator
        )
        with torch.no_grad():
            real_embedding = real_net(x, features).numpy()
            complex_embedding = complex_net(x, features).numpy()
        splits = {
            "tr": slice(0, 6),
            "en": slice(6, 10),
            "va": slice(10, 12),
        }
        real_center = subject.fit_center(
            {name: real_embedding[value] for name, value in splits.items()}
        )
        complex_center = subject.fit_center(
            {name: complex_embedding[value] for name, value in splits.items()}
        )
        fused = {
            name: subject.fuse_numpy(
                real_embedding[value],
                complex_embedding[value],
                real_center,
                complex_center,
                weight_real=0.5,
            )
            for name, value in splits.items()
        }
        prototypes = subject.fit_prototypes(
            fused, np.asarray([0, 0, 1, 1], dtype=np.int64), 2
        )
        return real_center, complex_center, prototypes

    def test_assembly_is_bit_identical_for_a_fixed_seed(self) -> None:
        first = self._assemble(20260730)
        second = self._assemble(20260730)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)

    def test_a_different_seed_changes_the_assembly(self) -> None:
        first = self._assemble(20260730)
        other = self._assemble(20260731)
        self.assertFalse(np.array_equal(first[0], other[0]))


class SchemaTests(unittest.TestCase):
    @staticmethod
    def _payload(**overrides: object) -> dict[str, object]:
        payload = {key: None for key in subject.REQUIRED_DEV_METRICS_KEYS}
        payload.update(
            {
                "development_only": True,
                "release_evidence": False,
                "sealed_release_data_used": 0,
                "consumed_test_rows_used": 0,
            }
        )
        payload.update(overrides)
        return payload

    def test_required_schema_matches_the_branch_runner(self) -> None:
        with (HERE / "run_time_domain_dev.py").open(encoding="utf-8") as handle:
            self.assertIn("sealed_release_data_used", handle.read())
        for key in (
            "closed_selection",
            "population_causal_length",
            "population_physical_scale",
            "training_views",
            "frontend",
            "data_audit",
            "source_index_contract",
        ):
            self.assertIn(key, subject.REQUIRED_DEV_METRICS_KEYS)

    def test_schema_validation_accepts_a_complete_payload(self) -> None:
        payload = self._payload(source_branches={}, fusion={})
        self.assertIs(subject.validate_result_schema(payload), payload)

    def test_schema_validation_rejects_missing_keys(self) -> None:
        payload = self._payload()
        payload.pop("population_physical_scale")
        with self.assertRaisesRegex(ValueError, "missing required keys"):
            subject.validate_result_schema(payload)

    def test_schema_validation_rejects_release_or_consumed_claims(self) -> None:
        for key, bad in (
            ("development_only", False),
            ("release_evidence", True),
            ("sealed_release_data_used", 1),
            ("consumed_test_rows_used", 3),
        ):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, key):
                    subject.validate_result_schema(self._payload(**{key: bad}))


class BranchLoadingTests(unittest.TestCase):
    def _write_branch(self, root: Path, encoder: str) -> Path:
        directory = root / encoder
        directory.mkdir(parents=True)
        torch.manual_seed(5)
        real_net, complex_net = _models()
        net = real_net if encoder == "real" else complex_net
        state_path = directory / f"{encoder}_state_dict.pt"
        prototypes_path = directory / f"{encoder}_prototypes.npy"
        torch.save(net.state_dict(), state_path)
        np.save(
            prototypes_path,
            np.zeros((2, net.cfg.embed_dim), dtype=np.float32),
        )
        metrics = {
            "encoder": encoder,
            "development_only": True,
            "sealed_release_data_used": 0,
            "consumed_test_rows_used": 0,
            "data_audit": {"consumed_test_rows_exposed": 0},
            "frontend": {
                "version": subject.td_preprocess.PREPROCESS_VERSION,
                "uses_frequency_transform": False,
            },
            "architecture": net.config(),
            "parameter_count": int(
                sum(p.numel() for p in net.parameters())
            ),
            "closed_selection": {"balanced_accuracy": 0.5},
            "training": {"episodes": 4000},
            "training_views": {
                "enabled": True,
                "lengths": [4096, 8192, 16384],
                "eligible_corpus_indices_sha256": "abc",
            },
            "source_index_contract": {"contract": {"id": "test"}},
            "artifacts": {
                "state_dict": state_path.name,
                "state_dict_sha256": subject.dev._sha256(state_path),
                "prototypes": prototypes_path.name,
                "prototypes_sha256": subject.dev._sha256(prototypes_path),
            },
        }
        with (directory / "dev_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f)
        return directory

    def test_load_branch_reads_state_dict_and_prototypes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = self._write_branch(Path(temporary), "real")
            branch = subject.load_branch(directory, "real")
        self.assertEqual(branch["encoder"], "real")
        self.assertEqual(branch["net"].cfg.encoder, "real")
        self.assertEqual(branch["prototypes"].shape[1], 4)
        self.assertIn("dev_metrics.json", branch["hashes"])

    def test_load_branch_refuses_the_wrong_encoder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = self._write_branch(Path(temporary), "real")
            with self.assertRaisesRegex(ValueError, "expected 'complex'"):
                subject.load_branch(directory, "complex")

    def test_load_branch_refuses_a_tampered_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = self._write_branch(Path(temporary), "real")
            metrics_path = directory / "dev_metrics.json"
            with metrics_path.open(encoding="utf-8") as handle:
                metrics = json.load(handle)
            metrics["artifacts"]["state_dict_sha256"] = "0" * 64
            with metrics_path.open("w", encoding="utf-8") as handle:
                json.dump(metrics, handle)
            with self.assertRaisesRegex(RuntimeError, "recorded SHA-256"):
                subject.load_branch(directory, "real")

    def test_load_branch_refuses_a_frequency_transform_frontend(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = self._write_branch(Path(temporary), "real")
            metrics_path = directory / "dev_metrics.json"
            with metrics_path.open(encoding="utf-8") as handle:
                metrics = json.load(handle)
            metrics["frontend"]["version"] = "invariant-patch-v1"
            with metrics_path.open("w", encoding="utf-8") as handle:
                json.dump(metrics, handle)
            with self.assertRaisesRegex(RuntimeError, "frontend"):
                subject.load_branch(directory, "real")

    def test_load_branch_refuses_consumed_or_sealed_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = self._write_branch(Path(temporary), "real")
            metrics_path = directory / "dev_metrics.json"
            with metrics_path.open(encoding="utf-8") as handle:
                metrics = json.load(handle)
            metrics["consumed_test_rows_used"] = 4
            with metrics_path.open("w", encoding="utf-8") as handle:
                json.dump(metrics, handle)
            with self.assertRaisesRegex(RuntimeError, "sealed or consumed"):
                subject.load_branch(directory, "real")

    def test_mismatched_branches_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = subject.load_branch(self._write_branch(root, "real"), "real")
            complex_branch = subject.load_branch(
                self._write_branch(root, "complex"), "complex"
            )
        subject._assert_branches_match(real, complex_branch)
        complex_branch["metrics"]["training"]["episodes"] = 1000
        with self.assertRaisesRegex(ValueError, "episode budgets"):
            subject._assert_branches_match(real, complex_branch)
        complex_branch["metrics"]["training"]["episodes"] = 4000
        complex_branch["metrics"]["training_views"]["lengths"] = [16384]
        with self.assertRaisesRegex(ValueError, "lengths disagree"):
            subject._assert_branches_match(real, complex_branch)


if __name__ == "__main__":
    unittest.main()
