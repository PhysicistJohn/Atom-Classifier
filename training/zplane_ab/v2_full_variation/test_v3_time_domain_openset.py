from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from v3_time_domain_openset import (  # noqa: E402
    FROZEN_GEOMETRY_FEATURE,
    FROZEN_GEOMETRY_WEIGHT,
    FROZEN_V2_WEIGHT,
    FrozenV3OpenSet,
    KnownOnlyGeometry,
    aggregate_deviation,
    empirical_rank,
    time_domain_geometry,
)


class TimeDomainGeometryTest(unittest.TestCase):
    def test_schema_is_finite_and_stable(self):
        rng = np.random.default_rng(101)
        packed = rng.standard_normal((4, 2, 16 * 64)).astype(np.float32)
        values, names = time_domain_geometry(packed)
        self.assertEqual(values.shape, (4, 63))
        self.assertEqual(len(names), 63)
        self.assertEqual(len(set(names)), len(names))
        self.assertIn(FROZEN_GEOMETRY_FEATURE, names)
        self.assertTrue(np.isfinite(values).all())

    def test_global_phase_and_patch_order_invariance(self):
        rng = np.random.default_rng(103)
        packed = rng.standard_normal((5, 2, 16 * 64)).astype(np.float32)
        base, names = time_domain_geometry(packed)

        z = packed[:, 0] + 1j * packed[:, 1]
        z *= np.exp(1j * 1.234)
        rotated = np.stack((z.real, z.imag), axis=1).astype(np.float32)
        rotated_value, rotated_names = time_domain_geometry(rotated)
        self.assertEqual(rotated_names, names)
        np.testing.assert_allclose(
            rotated_value,
            base,
            atol=2e-6,
            rtol=2e-6,
        )

        patch = packed.reshape(5, 2, 16, 64)
        permuted = patch[:, :, rng.permutation(16)].reshape(packed.shape)
        permuted_value, permuted_names = time_domain_geometry(permuted)
        self.assertEqual(permuted_names, names)
        np.testing.assert_allclose(
            permuted_value,
            base,
            atol=1e-6,
            rtol=1e-6,
        )

    def test_invalid_geometry_fails_closed(self):
        with self.assertRaises(ValueError):
            time_domain_geometry(np.zeros((2, 2, 100), dtype=np.float32))
        invalid = np.zeros((2, 2, 16 * 64), dtype=np.float32)
        invalid[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            time_domain_geometry(invalid)


class KnownOnlyPolicyTest(unittest.TestCase):
    @staticmethod
    def _fixture():
        rng = np.random.default_rng(107)
        train_per_class = 8
        enroll_per_class = 5
        n_classes = 3
        train_y = np.repeat(np.arange(n_classes), train_per_class)
        enroll_y = np.repeat(np.arange(n_classes), enroll_per_class)
        train = rng.standard_normal(
            (len(train_y), 2, 16 * 64)
        ).astype(np.float32)
        enroll = rng.standard_normal(
            (len(enroll_y), 2, 16 * 64)
        ).astype(np.float32)
        enroll_v2 = np.linspace(0.02, 0.92, len(enroll_y), dtype=np.float64)
        return train, train_y, enroll, enroll_y, enroll_v2

    def test_empirical_rank_is_monotone_and_bounded(self):
        calibration = np.asarray((1.0, 2.0, 3.0, 4.0))
        query = np.asarray((-1.0, 1.0, 1.5, 4.0, 8.0))
        rank = empirical_rank(query, calibration)
        self.assertTrue(np.all(rank[1:] >= rank[:-1]))
        self.assertTrue(np.all(rank >= 0.0))
        self.assertTrue(np.all(rank < 1.0))
        np.testing.assert_array_equal(
            rank,
            np.asarray((0.0, 0.0, 0.2, 0.6, 0.8)),
        )

    def test_known_geometry_is_deterministic_and_prediction_conditioned(self):
        train, train_y, enroll, enroll_y, _ = self._fixture()
        first = KnownOnlyGeometry.fit(train, train_y, n_classes=3)
        second = KnownOnlyGeometry.fit(train, train_y, n_classes=3)
        np.testing.assert_array_equal(first.class_mean, second.class_mean)
        np.testing.assert_array_equal(first.class_scale, second.class_scale)
        score = first.deviations(enroll, enroll_y)
        changed = first.deviations(enroll, (enroll_y + 1) % 3)
        self.assertTrue(np.any(score != changed))

    def test_frozen_contract_fit_score_and_roundtrip(self):
        train, train_y, enroll, enroll_y, enroll_v2 = self._fixture()
        fitted = FrozenV3OpenSet.fit(
            train,
            train_y,
            enroll,
            enroll_y,
            enroll_v2,
            n_classes=3,
        )
        self.assertEqual(FROZEN_V2_WEIGHT, 0.80)
        self.assertEqual(FROZEN_GEOMETRY_WEIGHT, 0.20)
        self.assertEqual(
            fitted.geometry.feature_names[fitted.geometry_feature_index],
            FROZEN_GEOMETRY_FEATURE,
        )
        score = fitted.score(enroll_v2, enroll, enroll_y)
        self.assertTrue(np.isfinite(score).all())
        self.assertTrue(np.all((score >= 0.0) & (score < 1.0)))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rejector.npz"
            np.savez_compressed(path, **fitted.to_payload())
            with np.load(path, allow_pickle=False) as stored:
                restored = FrozenV3OpenSet.from_payload(
                    {name: stored[name] for name in stored.files}
                )
        np.testing.assert_array_equal(
            restored.score(enroll_v2, enroll, enroll_y),
            score,
        )
        self.assertEqual(restored.threshold, fitted.threshold)

    def test_policy_rejects_bad_scores_and_serialized_contract(self):
        train, train_y, enroll, enroll_y, enroll_v2 = self._fixture()
        fitted = FrozenV3OpenSet.fit(
            train,
            train_y,
            enroll,
            enroll_y,
            enroll_v2,
            n_classes=3,
        )
        bad = enroll_v2.copy()
        bad[0] = 1.0
        with self.assertRaises(ValueError):
            fitted.score(bad, enroll, enroll_y)
        payload = fitted.to_payload()
        payload["v2_weight"] = np.asarray(0.75)
        with self.assertRaises(ValueError):
            FrozenV3OpenSet.from_payload(payload)

    def test_aggregate_deviation_reducers(self):
        value = np.asarray(((1.0, 2.0), (3.0, 4.0)))
        index = np.asarray((0, 1))
        np.testing.assert_array_equal(
            aggregate_deviation(value, index, reducer="max"),
            np.asarray((2.0, 4.0)),
        )
        with self.assertRaises(ValueError):
            aggregate_deviation(value, index, reducer="other")


if __name__ == "__main__":
    unittest.main()
