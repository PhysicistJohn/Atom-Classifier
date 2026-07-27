from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from assemble_invariant_candidate import (  # noqa: E402
    ALPHA_COMPLEX,
    ALPHA_REAL,
    UNKNOWN_QUANTILE,
    WEIGHT_REAL,
    _fewshot_report,
    _fuse_numpy,
    _lof_ensemble_tensor_bundle,
    _reload_lof_ensemble,
    _save_lof_ensemble,
    _score_lof_ensemble,
    load_runtime_bundle,
    validate_runtime_bundle,
)
from invariant_fusion import CenteredInvariantFusion  # noqa: E402
from invariant_patch_cnn import (  # noqa: E402
    InvariantPatchCNN,
    InvariantPatchConfig,
)
from known_only_patch_openset import KnownOnlyLOFOpenSet  # noqa: E402
from train import prototypes_from  # noqa: E402


class CandidateAssemblyTest(unittest.TestCase):
    @staticmethod
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
            InvariantPatchCNN(
                InvariantPatchConfig(encoder="real", **common)
            ).eval(),
            InvariantPatchCNN(
                InvariantPatchConfig(encoder="complex", **common)
            ).eval(),
        )

    def test_numpy_fusion_matches_module(self):
        torch.manual_seed(8)
        real, complex_branch = self._models()
        x = torch.randn(7, 2, real.packed_length)
        features = torch.randn(7, real.cfg.n_features)
        with torch.no_grad():
            real_embedding = real(x, features).numpy()
            complex_embedding = complex_branch(x, features).numpy()
        real_center = real_embedding[:4].mean(axis=0)
        complex_center = complex_embedding[:4].mean(axis=0)
        fusion = CenteredInvariantFusion(
            real,
            complex_branch,
            real_center,
            complex_center,
            alpha_real=ALPHA_REAL,
            alpha_complex=ALPHA_COMPLEX,
            weight_real=WEIGHT_REAL,
        ).eval()
        with torch.no_grad():
            expected = fusion(x, features).numpy()
        actual = _fuse_numpy(
            real_embedding,
            complex_embedding,
            real_center,
            complex_center,
        )
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)

    def test_fewshot_report_is_deterministic(self):
        rng = np.random.default_rng(11)
        n_classes, per_class, dimension = 3, 12, 6
        labels = np.repeat(np.arange(n_classes), per_class)
        centers = np.eye(n_classes, dimension, dtype=np.float32)
        enrollment = centers[labels] + 0.02 * rng.standard_normal(
            (len(labels), dimension)
        )
        selection = centers[labels] + 0.03 * rng.standard_normal(
            (len(labels), dimension)
        )
        data = {
            "yen": labels,
            "yva": labels,
            "classes": ["a", "b", "c"],
        }
        prototypes = prototypes_from(enrollment, labels, n_classes)
        args = (
            enrollment,
            selection,
            data,
            prototypes,
            {"a": 1.0, "b": 1.0, "c": 1.0},
        )
        first = _fewshot_report(
            *args, seed=17, trials=20, k=5
        )
        second = _fewshot_report(
            *args, seed=17, trials=20, k=5
        )
        self.assertEqual(first, second)
        self.assertEqual(
            first["leave_one_prototype_out"]["resolvable_mean"], 1.0
        )

    def test_lof_ensemble_serialization_round_trip(self):
        rng = np.random.default_rng(21)
        training = rng.normal(size=(80, 4))
        enrollment = rng.normal(size=(30, 4))
        query_real = rng.normal(size=(17, 4))
        query_complex = rng.normal(size=(17, 4))
        real = KnownOnlyLOFOpenSet.fit(
            training, enrollment, neighbors=2
        )
        complex_scorer = KnownOnlyLOFOpenSet.fit(
            training + 0.1, enrollment - 0.1, neighbors=7
        )
        components = [
            ("real", 0.4, real),
            ("complex", 0.6, complex_scorer),
        ]
        enrollment_score = _score_lof_ensemble(
            components,
            {"real": enrollment, "complex": enrollment - 0.1},
        )
        threshold = float(np.quantile(enrollment_score, UNKNOWN_QUANTILE))
        expected = _score_lof_ensemble(
            components,
            {"real": query_real, "complex": query_complex},
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lof.npz"
            _save_lof_ensemble(path, components, threshold)
            loaded, loaded_threshold = _reload_lof_ensemble(path)
            actual = _score_lof_ensemble(
                loaded,
                {"real": query_real, "complex": query_complex},
            )
        self.assertEqual(loaded_threshold, threshold)
        np.testing.assert_array_equal(actual, expected)

    def _runtime_bundle(self) -> dict:
        torch.manual_seed(13)
        real, complex_branch = self._models()
        rng = np.random.default_rng(13)
        training = rng.normal(size=(80, real.cfg.embed_dim))
        enrollment = rng.normal(size=(30, real.cfg.embed_dim))
        first = KnownOnlyLOFOpenSet.fit(training, enrollment, neighbors=2)
        second = KnownOnlyLOFOpenSet.fit(
            training + 0.2, enrollment - 0.1, neighbors=7
        )
        components = [
            ("real", 0.4, first),
            ("complex", 0.6, second),
        ]
        enrollment_score = _score_lof_ensemble(
            components,
            {"real": enrollment, "complex": enrollment - 0.1},
        )
        threshold = float(np.quantile(enrollment_score, UNKNOWN_QUANTILE))
        return {
            "schema": 2,
            "kind": "invariant_centered_fusion_classifier",
            "real_config": real.config(),
            "complex_config": complex_branch.config(),
            "real_state_dict": real.state_dict(),
            "complex_state_dict": complex_branch.state_dict(),
            "real_center": torch.zeros(real.cfg.embed_dim),
            "complex_center": torch.zeros(complex_branch.cfg.embed_dim),
            "alpha_real": ALPHA_REAL,
            "alpha_complex": ALPHA_COMPLEX,
            "weight_real": WEIGHT_REAL,
            "eps": 1e-12,
            "feature_mean": torch.zeros(real.cfg.n_features),
            "feature_std": torch.ones(real.cfg.n_features),
            "classes": ["a", "b"],
            "prototypes": torch.randn(2, 2 * real.cfg.embed_dim),
            "unknown_threshold": threshold,
            "temperature": 0.1,
            "open_set": _lof_ensemble_tensor_bundle(
                components, threshold
            ),
            "development_only": True,
            "provenance": {
                "release_evidence": False,
                "consumed_test_rows_used": 0,
            },
        }

    def test_runtime_bundle_schema_reload_and_fail_closed(self):
        bundle = self._runtime_bundle()
        self.assertIs(validate_runtime_bundle(bundle), bundle)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bundle.pt"
            torch.save(bundle, path)
            loaded = load_runtime_bundle(path)
        self.assertEqual(loaded["schema"], 2)
        self.assertEqual(loaded["classes"], ["a", "b"])

        missing = copy.deepcopy(bundle)
        del missing["prototypes"]
        with self.assertRaises(ValueError):
            validate_runtime_bundle(missing)

        nonfinite = copy.deepcopy(bundle)
        nonfinite["real_center"][0] = float("nan")
        with self.assertRaises(ValueError):
            validate_runtime_bundle(nonfinite)

        wrong_threshold = copy.deepcopy(bundle)
        wrong_threshold["unknown_threshold"] += 0.01
        with self.assertRaises(ValueError):
            validate_runtime_bundle(wrong_threshold)

        bad_state = copy.deepcopy(bundle)
        first_name = next(iter(bad_state["real_state_dict"]))
        bad_state["real_state_dict"][first_name].reshape(-1)[0] = float("nan")
        with self.assertRaises(ValueError):
            validate_runtime_bundle(bad_state)


if __name__ == "__main__":
    unittest.main()
