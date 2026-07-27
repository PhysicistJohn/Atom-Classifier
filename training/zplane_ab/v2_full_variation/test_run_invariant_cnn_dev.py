from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import torch


HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from invariant_patch_cnn import InvariantPatchCNN  # noqa: E402
from run_invariant_cnn_dev import (  # noqa: E402
    _balanced_accuracy,
    _parse_encoders,
    _phase_augment,
    _prototype_orthogonality_loss,
    _sample_episode_positions,
)


class InvariantRunnerUnitTest(unittest.TestCase):
    def test_encoder_parser_fails_closed(self):
        self.assertEqual(_parse_encoders("real,complex"), ["real", "complex"])
        for value in ("", "fft", "real,real"):
            with self.assertRaises(ValueError):
                _parse_encoders(value)

    def test_episode_positions_are_balanced_and_disjoint(self):
        pools = [np.arange(c * 20, (c + 1) * 20) for c in range(3)]
        support, query, support_labels, query_labels = _sample_episode_positions(
            pools, np.random.default_rng(7), 3
        )
        self.assertEqual(len(support), 15)
        self.assertEqual(len(query), 15)
        self.assertEqual(len(np.intersect1d(support, query)), 0)
        np.testing.assert_array_equal(
            np.bincount(support_labels), np.full(3, 5)
        )
        np.testing.assert_array_equal(
            np.bincount(query_labels), np.full(3, 5)
        )

    def test_phase_augmentation_preserves_magnitude(self):
        x = torch.randn(9, 2, 1024)
        augmented = _phase_augment(x)
        before = x[:, 0].square() + x[:, 1].square()
        after = augmented[:, 0].square() + augmented[:, 1].square()
        self.assertLess(float((before - after).abs().max()), 5e-6)

    def test_balanced_accuracy(self):
        labels = np.array([0, 0, 0, 1])
        prediction = np.array([0, 0, 0, 0])
        self.assertAlmostEqual(
            _balanced_accuracy(prediction, labels, n_classes=2), 0.5
        )

    def test_prototype_separation_is_zero_for_orthogonal_directions(self):
        orthogonal = torch.eye(4)
        collapsed = torch.ones(4, 4)
        self.assertAlmostEqual(
            float(_prototype_orthogonality_loss(orthogonal)), 0.0
        )
        self.assertGreater(
            float(_prototype_orthogonality_loss(collapsed)), 0.9
        )

    def test_two_episode_cpu_training_surface(self):
        # Full cache and corpus isolation are covered in invariant_patch_data tests.
        # This checks only the differentiable episodic surface on tiny fake arrays.
        from run_invariant_cnn_dev import train_model

        rng = np.random.default_rng(11)
        net = InvariantPatchCNN(
            patch_length=16,
            patch_count=2,
            patch_dim=8,
            hidden=12,
            dropout=0.0,
        )
        n_classes, rows_per_class = 2, 12
        labels = np.repeat(np.arange(n_classes), rows_per_class)
        x = rng.standard_normal(
            (len(labels), 2, net.packed_length), dtype=np.float32
        )
        features = rng.standard_normal(
            (len(labels), net.cfg.n_features), dtype=np.float32
        )
        enroll_rows = np.r_[0:5, 12:17]
        selection_rows = np.r_[5:10, 17:22]
        data = {
            "xtr": x,
            "ftr": features,
            "ytr": labels,
            "idx_by_class": [
                np.where(labels == class_index)[0]
                for class_index in range(n_classes)
            ],
            "xen": x[enroll_rows],
            "fen": features[enroll_rows],
            "yen": labels[enroll_rows],
            "xva": x[selection_rows],
            "fva": features[selection_rows],
            "yva": labels[selection_rows],
            "n_classes": n_classes,
        }
        trained, history = train_model(
            net,
            data,
            torch.device("cpu"),
            episodes=2,
            eval_every=1,
            seed=5,
            lr=1e-3,
            weight_decay=0.0,
            warmup_frac=0.0,
            label_smoothing=0.0,
            phase_augmentation=True,
        )
        self.assertIsInstance(trained, InvariantPatchCNN)
        self.assertEqual(history["episodes"], 2)
        self.assertTrue(np.isfinite(history["best_selection_accuracy"]))


if __name__ == "__main__":
    unittest.main()
