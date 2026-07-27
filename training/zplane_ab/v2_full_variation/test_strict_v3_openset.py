from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from design_strict_v3_openset import (  # noqa: E402
    BASE_WEIGHT_GRID,
    CAPTURE_LENGTHS,
    GEOMETRY_SPECS,
    GATES,
    THRESHOLD_QUANTILE_SEQUENCE,
    StrictV3OpenSetPolicy,
    _assert_development_path,
    _choose_candidate,
)
from v3_time_domain_openset import (  # noqa: E402
    KnownOnlyGeometry,
    time_domain_geometry,
)


class StrictGridContractTest(unittest.TestCase):
    def test_grid_is_bounded_positive_and_uses_existing_features(self):
        rng = np.random.default_rng(201)
        packed = rng.standard_normal((2, 2, 16 * 64)).astype(np.float32)
        _, names = time_domain_geometry(packed)
        available = set(names)

        self.assertEqual(len(GEOMETRY_SPECS), 40)
        self.assertEqual(len({spec.name for spec in GEOMETRY_SPECS}), 40)
        self.assertEqual(BASE_WEIGHT_GRID, (0.25, 0.50, 0.75, 0.90))
        self.assertEqual(
            THRESHOLD_QUANTILE_SEQUENCE,
            (0.95, 0.925, 0.90),
        )
        self.assertEqual(CAPTURE_LENGTHS, (4096, 8192, 16384, 32768))
        for weight in BASE_WEIGHT_GRID:
            self.assertGreater(weight, 0.0)
            self.assertLess(weight, 1.0)
        for spec in GEOMETRY_SPECS:
            self.assertTrue(set(spec.feature_names) <= available)
            self.assertIn(spec.reducer, {"max", "rms"})

    def test_refuses_release_and_sealed_paths(self):
        with self.assertRaises(ValueError):
            _assert_development_path(
                Path("/tmp/project/releases/candidate"),
                "test",
            )
        with self.assertRaises(ValueError):
            _assert_development_path(
                Path("/tmp/project/sealed-suite"),
                "test",
            )
        accepted = _assert_development_path(
            Path("/tmp/project/development"),
            "test",
        )
        self.assertEqual(
            accepted,
            Path("/tmp/project/development").resolve(),
        )


class QuantilePreferenceTest(unittest.TestCase):
    @staticmethod
    def _candidate(name: str, *, passed: bool, quality: float):
        values = {
            "overall_auroc": quality,
            "noise_auroc": quality,
            "chirp_auroc": quality,
            "noise_threshold_recall": quality,
            "chirp_threshold_recall": quality,
        }
        return {
            "name": name,
            "summary": {
                "all_pass": passed,
                "gate_count": len(GATES) + int(passed),
                "minimum_normalized_margin": quality,
                "mean_overall_auroc": quality,
                "values": values,
                "gates": {"known_fur": {"value": 0.05}},
            },
        }

    def test_q95_wins_before_better_lower_quantile(self):
        candidates = {
            0.95: [self._candidate("q95", passed=True, quality=0.81)],
            0.925: [self._candidate("q925", passed=True, quality=0.99)],
            0.90: [self._candidate("q90", passed=True, quality=1.0)],
        }
        selected, reason = _choose_candidate(candidates)
        self.assertEqual(selected["name"], "q95")
        self.assertIn("0.95", reason)

    def test_lower_quantile_requires_earlier_failure(self):
        candidates = {
            0.95: [self._candidate("q95", passed=False, quality=0.99)],
            0.925: [self._candidate("q925", passed=True, quality=0.82)],
            0.90: [self._candidate("q90", passed=True, quality=1.0)],
        }
        selected, reason = _choose_candidate(candidates)
        self.assertEqual(selected["name"], "q925")
        self.assertIn("0.925", reason)

    def test_no_pass_fails_closed_to_q95(self):
        candidates = {
            0.95: [
                self._candidate("q95-low", passed=False, quality=0.5),
                self._candidate("q95-best", passed=False, quality=0.7),
            ],
            0.925: [self._candidate("q925", passed=False, quality=0.9)],
            0.90: [self._candidate("q90", passed=False, quality=1.0)],
        }
        selected, reason = _choose_candidate(candidates)
        self.assertEqual(selected["name"], "q95-best")
        self.assertIn("fail closed", reason)


class StrictPolicySerializationTest(unittest.TestCase):
    @staticmethod
    def _fixture():
        rng = np.random.default_rng(203)
        train_y = np.repeat(np.arange(3), 8)
        enroll_y = np.repeat(np.arange(3), 5)
        train = rng.standard_normal(
            (len(train_y), 2, 16 * 64)
        ).astype(np.float32)
        enroll = rng.standard_normal(
            (len(enroll_y), 2, 16 * 64)
        ).astype(np.float32)
        geometry = KnownOnlyGeometry.fit(train, train_y, n_classes=3)
        deviations = geometry.deviations(enroll, enroll_y)
        base = np.linspace(0.01, 0.91, len(enroll_y))
        return geometry, deviations, base

    def test_fit_score_and_allow_pickle_false_roundtrip(self):
        geometry, deviations, base = self._fixture()
        policy = StrictV3OpenSetPolicy.calibrate(
            geometry,
            GEOMETRY_SPECS[0],
            deviations,
            base,
            base_weight=0.75,
            threshold_quantile=0.95,
        )
        score = policy.score_from_deviations(base, deviations)
        self.assertTrue(np.isfinite(score).all())
        self.assertTrue(np.all((score >= 0.0) & (score < 1.0)))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.npz"
            np.savez_compressed(path, **policy.to_payload())
            with np.load(path, allow_pickle=False) as stored:
                restored = StrictV3OpenSetPolicy.from_payload(
                    {name: stored[name] for name in stored.files}
                )
        np.testing.assert_array_equal(
            restored.score_from_deviations(base, deviations),
            score,
        )
        self.assertEqual(restored.threshold, policy.threshold)
        self.assertEqual(restored.spec, policy.spec)

    def test_rejects_policy_values_outside_frozen_grid(self):
        geometry, deviations, base = self._fixture()
        with self.assertRaises(ValueError):
            StrictV3OpenSetPolicy.calibrate(
                geometry,
                GEOMETRY_SPECS[0],
                deviations,
                base,
                base_weight=0.8,
                threshold_quantile=0.95,
            )
        policy = StrictV3OpenSetPolicy.calibrate(
            geometry,
            GEOMETRY_SPECS[0],
            deviations,
            base,
            base_weight=0.75,
            threshold_quantile=0.95,
        )
        payload = policy.to_payload()
        payload["threshold_quantile"] = np.asarray(0.91)
        with self.assertRaises(ValueError):
            StrictV3OpenSetPolicy.from_payload(payload)


if __name__ == "__main__":
    unittest.main()
