"""Unit tests for the v3 staged open-set browser exporter.

These cover the deterministic, cheap pieces: seed hygiene, payload contracts,
prefilter conversion, and the raw-LOF decomposition agreeing with the shipped
scorer.  The heavy end-to-end export runs as a script and self-checks every
row against ``fit_v3_openset.score_rows`` before writing anything.
"""
from __future__ import annotations

import unittest

import numpy as np

import export_v3_openset_browser_assets as exporter
import fit_v3_openset_staged as staged
import noise_prefilter
from known_only_patch_openset import KnownOnlyLOFOpenSet


def synthetic_composite(
    stage_two_threshold: float = 0.9,
    survivors: int = 24,
    gated: int = 3,
) -> staged.CompositeSurvivorPolicy:
    rng = np.random.default_rng(23)
    composite_calibration = np.sort(rng.uniform(0.0, 0.999, size=survivors))
    return staged.CompositeSurvivorPolicy(
        stage_one_calibration_raw=np.sort(rng.normal(size=survivors)),
        composite_calibration_raw=composite_calibration,
        threshold=float(
            np.quantile(
                composite_calibration, staged.COMPOSITE_THRESHOLD_QUANTILE
            )
        ),
        stage_two_threshold=float(stage_two_threshold),
        enrollment_rows=survivors + gated,
        enrollment_gated_rows=gated,
        enrollment_capture_length=16384,
    )


def synthetic_prefilter(threshold: float = 1.0) -> noise_prefilter.NoisePrefilter:
    width = len(noise_prefilter.PREFILTER_FEATURES)
    return noise_prefilter.NoisePrefilter(
        feature_names=noise_prefilter.PREFILTER_FEATURES,
        capture_length=4096,
        mean=np.linspace(-1.0, 1.0, width),
        scale=np.linspace(0.5, 2.0, width),
        coefficients=np.linspace(-0.3, 0.4, width),
        intercept=0.25,
        threshold_score=threshold,
        fit_provenance={"synthetic": True},
        threshold_provenance={"synthetic": True},
    )


class ParitySeedHygiene(unittest.TestCase):
    def test_default_seed_is_accepted(self) -> None:
        self.assertEqual(
            exporter.validate_parity_seed(exporter.PARITY_NOVELTY_SEED),
            exporter.PARITY_NOVELTY_SEED,
        )

    def test_novelty_namespace_is_refused(self) -> None:
        for seed in (20260942, 20260999, 20260900):
            with self.assertRaises(ValueError):
                exporter.validate_parity_seed(seed)

    def test_fitting_band_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            exporter.validate_parity_seed(20261001)

    def test_release_and_sealed_seeds_are_refused(self) -> None:
        for seed in (20260731, 20260729, 20260938):
            with self.assertRaises(ValueError):
                exporter.validate_parity_seed(seed)


class PrefilterConversion(unittest.TestCase):
    def test_payload_carries_every_parameter(self) -> None:
        model = synthetic_prefilter()
        payload = exporter.prefilter_model_payload(model)
        self.assertEqual(payload["capture_length"], 4096)
        self.assertEqual(
            payload["feature_names"], list(noise_prefilter.PREFILTER_FEATURES)
        )
        np.testing.assert_array_equal(payload["mean"], model.mean)
        np.testing.assert_array_equal(payload["scale"], model.scale)
        np.testing.assert_array_equal(payload["coefficients"], model.coefficients)
        self.assertEqual(payload["intercept"], model.intercept)
        self.assertEqual(payload["threshold_score"], model.threshold_score)

    def test_unthresholded_bundle_is_refused(self) -> None:
        model = synthetic_prefilter(threshold=float("nan"))
        with self.assertRaises(ValueError):
            exporter.prefilter_model_payload(model)


class JsonDeterminism(unittest.TestCase):
    def test_paths_are_refused_in_payloads(self) -> None:
        from pathlib import Path

        with self.assertRaises(TypeError):
            exporter._jsonable({"path": Path("/tmp/x")})

    def test_numpy_values_become_plain_json(self) -> None:
        value = exporter._jsonable(
            {
                "array": np.asarray([1.5, 2.5], dtype=np.float32),
                "integer": np.int64(3),
                "flag": np.bool_(True),
            }
        )
        self.assertEqual(value["array"], [1.5, 2.5])
        self.assertEqual(value["integer"], 3)
        self.assertIs(value["flag"], True)


class RawLofDecomposition(unittest.TestCase):
    def test_raw_matches_scorer_rank_ordering(self) -> None:
        rng = np.random.default_rng(7)
        training = rng.normal(size=(64, 4))
        enrollment = rng.normal(size=(32, 4))
        scorer = KnownOnlyLOFOpenSet.fit(training, enrollment, neighbors=3)
        queries = rng.normal(size=(8, 4))
        raw = exporter.compute_branch_lof_raw(scorer, queries)
        rank = scorer.score(queries)
        # The rank is the empirical rank of exactly this raw value, so ranking
        # raw with the scorer's calibration must reproduce the scorer output.
        expected = np.searchsorted(scorer.calibration, raw, side="left") / (
            len(scorer.calibration) + 1.0
        )
        np.testing.assert_array_equal(rank, expected)


class ContractKeys(unittest.TestCase):
    def test_weights_payload_carries_the_architecture_contract(self) -> None:
        prefilters = {4096: synthetic_prefilter()}
        rng = np.random.default_rng(11)
        training = rng.normal(size=(40, 4))
        enrollment = rng.normal(size=(24, 4))
        components = [
            ("real", 0.4, KnownOnlyLOFOpenSet.fit(training, enrollment, neighbors=2)),
            ("complex", 0.6, KnownOnlyLOFOpenSet.fit(training, enrollment, neighbors=3)),
        ]

        class PolicyStub:
            geometry_feature_index = 0
            threshold = 0.9

            class geometry:  # noqa: N801 - mirrors the dataclass attribute
                class_mean = np.zeros((7, 1))
                class_scale = np.ones((7, 1))

            geometry_calibration_raw = np.sort(rng.normal(size=(24,)))
            combined_calibration_raw = np.sort(rng.normal(size=(24,)))

        composite = synthetic_composite(stage_two_threshold=0.9)
        payload = exporter.openset_weights_payload(
            prefilters,
            "0" * 64,
            components,
            PolicyStub(),
            composite,
            {"patch_length": 64, "patch_count": 16, "target_frac": 0.5,
             "packed_length": 1024},
            {"development_only": True},
        )
        contract = payload["contract"]
        for key in noise_prefilter.CONTRACT_KEYS_FOR_RELEASE_CLAIM:
            self.assertIn(key, contract)
        self.assertIs(contract["additive_only"], False)
        self.assertIs(contract["changes_closed_label"], True)
        self.assertIs(contract["gates_before_classification"], True)
        self.assertEqual(
            payload["stage_two"]["policy"]["branch_lof_rank_weight"], 0.8
        )
        self.assertEqual(payload["stage_two"]["policy"]["geometry_weight"], 0.2)
        self.assertEqual(
            payload["stage_two"]["policy"]["threshold_quantile"], 0.95
        )
        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(
            payload["stage_two"]["kind"], staged.STAGED_POLICY_KIND
        )


class CompositePayload(unittest.TestCase):
    def test_payload_mirrors_the_npz_spelling_and_constants(self) -> None:
        composite = synthetic_composite()
        payload = exporter.composite_policy_payload(composite)
        self.assertEqual(payload["schema"], staged.STAGED_POLICY_SCHEMA)
        self.assertEqual(payload["kind"], staged.STAGED_POLICY_KIND)
        self.assertEqual(
            payload["policy_version"], staged.STAGED_POLICY_VERSION
        )
        self.assertEqual(
            payload["survivor_score"], staged.COMPOSITE_SURVIVOR_SCORE
        )
        self.assertEqual(
            payload["threshold_quantile"],
            float(staged.COMPOSITE_THRESHOLD_QUANTILE),
        )
        self.assertEqual(payload["threshold"], composite.threshold)
        self.assertEqual(
            payload["stage_two_threshold"], composite.stage_two_threshold
        )
        np.testing.assert_array_equal(
            payload["stage_one_calibration_raw"],
            composite.stage_one_calibration_raw,
        )
        np.testing.assert_array_equal(
            payload["composite_calibration_raw"],
            composite.composite_calibration_raw,
        )
        self.assertEqual(
            payload["enrollment_rows"],
            payload["enrollment_gated_rows"]
            + len(payload["stage_one_calibration_raw"]),
        )
        # The stored threshold is exactly the frozen quantile of the stored
        # calibration, which is what both loaders (Python and TypeScript)
        # recompute and verify.
        self.assertEqual(
            payload["threshold"],
            float(
                np.quantile(
                    np.asarray(payload["composite_calibration_raw"]),
                    staged.COMPOSITE_THRESHOLD_QUANTILE,
                )
            ),
        )


if __name__ == "__main__":
    unittest.main()
