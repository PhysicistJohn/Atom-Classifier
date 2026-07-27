"""Unit tests for the v3-fusion open-set fit.

These tests never touch the development corpus, the consumed test half, or any
sealed release suite.  Every population is synthetic, small, and constructed in
the test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fit_v3_openset as subject  # noqa: E402
import v3_time_domain_openset as frozen  # noqa: E402


PATCH_COUNT = frozen.PATCH_COUNT
PATCH_LENGTH = frozen.PATCH_LENGTH
PACKED_WIDTH = PATCH_COUNT * PATCH_LENGTH


def _packed(rows: int, seed: int, *, scale: float = 1.0) -> np.ndarray:
    """Synthetic packed I/Q patches with the shape the frozen policy expects."""
    rng = np.random.default_rng(seed)
    value = rng.normal(scale=scale, size=(rows, 2, PACKED_WIDTH))
    return value.astype(np.float32)


def _poison(like: np.ndarray) -> np.ndarray:
    return np.full(like.shape, np.nan, dtype=like.dtype)


def _embeddings(rows: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(rows, dim)).astype(np.float32)


def _branch_embeddings(
    train_rows: int = 90,
    enroll_rows: int = 40,
    metric_rows: int = 25,
    dim: int = 6,
) -> dict[str, dict[str, np.ndarray]]:
    return {
        "real": {
            "tr": _embeddings(train_rows, dim, 1),
            "en": _embeddings(enroll_rows, dim, 2),
            "va": _embeddings(metric_rows, dim, 3),
        },
        "complex": {
            "tr": _embeddings(train_rows, dim, 4),
            "en": _embeddings(enroll_rows, dim, 5),
            "va": _embeddings(metric_rows, dim, 6),
        },
    }


def _fit_small_policy(
    *,
    training_packed: np.ndarray | None = None,
    enrollment_packed: np.ndarray | None = None,
    n_classes: int = 2,
) -> tuple[frozen.FrozenV3OpenSet, dict[str, np.ndarray]]:
    training_packed = (
        _packed(12, 101) if training_packed is None else training_packed
    )
    enrollment_packed = (
        _packed(9, 202) if enrollment_packed is None else enrollment_packed
    )
    training_labels = np.arange(len(training_packed)) % n_classes
    enrollment_predicted = np.arange(len(enrollment_packed)) % n_classes
    enrollment_ranks = np.linspace(
        0.0, 0.9, len(enrollment_packed), dtype=np.float64
    )
    policy = subject.fit_policy(
        training_packed,
        training_labels.astype(np.int64),
        enrollment_packed,
        enrollment_predicted.astype(np.int64),
        enrollment_ranks,
        n_classes=n_classes,
    )
    return policy, {
        "training_packed": training_packed,
        "training_labels": training_labels.astype(np.int64),
        "enrollment_packed": enrollment_packed,
        "enrollment_predicted": enrollment_predicted.astype(np.int64),
        "enrollment_ranks": enrollment_ranks,
    }


class FrozenPolicyReuseTests(unittest.TestCase):
    def test_policy_constants_are_imported_not_redefined(self) -> None:
        self.assertIs(
            subject.BRANCH_LOF_RANK_WEIGHT, frozen.FROZEN_V2_WEIGHT
        )
        self.assertIs(subject.GEOMETRY_RANK_WEIGHT, frozen.FROZEN_GEOMETRY_WEIGHT)
        self.assertIs(
            subject.THRESHOLD_QUANTILE, frozen.FROZEN_THRESHOLD_QUANTILE
        )
        self.assertAlmostEqual(
            subject.BRANCH_LOF_RANK_WEIGHT + subject.GEOMETRY_RANK_WEIGHT,
            1.0,
        )

    def test_branch_lof_rank_weight_is_not_bound_to_v2_embeddings(self) -> None:
        """The frozen weight blends any caller-supplied rank vector in [0, 1).

        This is the empirical form of the docstring claim: two different rank
        vectors of the same length produce the documented linear blend, so the
        constant carries no v2-specific binding.
        """
        policy, _fitted = _fit_small_policy()
        packed = _packed(6, 303)
        predicted = np.arange(len(packed), dtype=np.int64) % 2
        low = np.full(len(packed), 0.05)
        high = np.full(len(packed), 0.95)
        geometry_rank = frozen.empirical_rank(
            policy.geometry.deviations(packed, predicted)[
                :, policy.geometry_feature_index
            ],
            policy.geometry_calibration_raw,
        )
        for supplied in (low, high):
            expected_raw = (
                subject.BRANCH_LOF_RANK_WEIGHT * supplied
                + subject.GEOMETRY_RANK_WEIGHT * geometry_rank
            )
            np.testing.assert_allclose(
                policy.score(supplied, packed, predicted),
                frozen.empirical_rank(
                    expected_raw, policy.combined_calibration_raw
                ),
            )


class TrainOnlyDensityTests(unittest.TestCase):
    def test_class_geometry_uses_training_rows_only(self) -> None:
        training = _packed(12, 11)
        policy, _fitted = _fit_small_policy(training_packed=training)
        reference = frozen.KnownOnlyGeometry.fit(
            training,
            (np.arange(len(training)) % 2).astype(np.int64),
            n_classes=2,
        )
        np.testing.assert_array_equal(
            policy.geometry.class_mean, reference.class_mean
        )
        np.testing.assert_array_equal(
            policy.geometry.class_scale, reference.class_scale
        )

    def test_density_ignores_selection_and_novelty_rows(self) -> None:
        """A poisoned selection/novelty population cannot reach the density.

        ``fit_policy`` has no parameter through which such rows could arrive, so
        fitting with only train/enrollment reproduces exactly.
        """
        policy_a, fitted = _fit_small_policy()
        policy_b = subject.fit_policy(
            fitted["training_packed"],
            fitted["training_labels"],
            fitted["enrollment_packed"],
            fitted["enrollment_predicted"],
            fitted["enrollment_ranks"],
            n_classes=2,
        )
        np.testing.assert_array_equal(
            policy_a.geometry.class_mean, policy_b.geometry.class_mean
        )
        self.assertEqual(policy_a.threshold, policy_b.threshold)
        self.assertEqual(
            set(subject.fit_policy.__code__.co_varnames[
                : subject.fit_policy.__code__.co_argcount
            ]),
            {
                "training_packed",
                "training_labels",
                "enrollment_packed",
                "enrollment_predicted_classes",
                "enrollment_branch_lof_ranks",
            },
        )

    def test_branch_lof_reads_only_train_and_enrollment_splits(self) -> None:
        embeddings = _branch_embeddings()
        poisoned = {
            branch: {
                "tr": splits["tr"],
                "en": splits["en"],
                "va": _poison(splits["va"]),
            }
            for branch, splits in embeddings.items()
        }
        clean = subject.fit_branch_lof(embeddings)
        poisoned_components = subject.fit_branch_lof(poisoned)
        for (_b, _w, left), (_b2, _w2, right) in zip(
            clean, poisoned_components
        ):
            np.testing.assert_array_equal(left.calibration, right.calibration)
            np.testing.assert_array_equal(
                left.embedding_reference, right.embedding_reference
            )

    def test_branch_lof_refuses_a_missing_training_split(self) -> None:
        embeddings = _branch_embeddings()
        del embeddings["real"]["tr"]
        with self.assertRaisesRegex(KeyError, "tr"):
            subject.fit_branch_lof(embeddings)

    def test_branch_lof_uses_the_inherited_v2_shape_without_flags(self) -> None:
        self.assertEqual(
            subject.BRANCH_LOF,
            (("real", 0.40, 2), ("complex", 0.60, 64)),
        )
        flags = {
            action.dest for action in subject.build_parser()._actions
        }
        self.assertFalse(
            {"real_neighbors", "complex_neighbors", "lof_weight"} & flags
        )


class EnrollmentOnlyRankTests(unittest.TestCase):
    def test_threshold_is_the_enrollment_q95(self) -> None:
        policy, _fitted = _fit_small_policy()
        self.assertAlmostEqual(
            policy.threshold,
            float(
                np.quantile(
                    policy.calibration_scores, subject.THRESHOLD_QUANTILE
                )
            ),
        )

    def test_ranks_are_calibrated_on_enrollment_rows(self) -> None:
        policy, fitted = _fit_small_policy()
        self.assertEqual(
            len(policy.combined_calibration_raw),
            len(fitted["enrollment_packed"]),
        )
        self.assertEqual(
            len(policy.geometry_calibration_raw),
            len(fitted["enrollment_packed"]),
        )
        self.assertTrue(
            np.all(np.diff(policy.combined_calibration_raw) >= 0.0)
        )

    def test_changing_enrollment_changes_the_threshold_only_through_enrollment(
        self,
    ) -> None:
        base, fitted = _fit_small_policy()
        other = subject.fit_policy(
            fitted["training_packed"],
            fitted["training_labels"],
            _packed(9, 999),
            fitted["enrollment_predicted"],
            fitted["enrollment_ranks"],
            n_classes=2,
        )
        np.testing.assert_array_equal(
            base.geometry.class_mean, other.geometry.class_mean
        )
        self.assertFalse(
            np.array_equal(
                base.geometry_calibration_raw, other.geometry_calibration_raw
            )
        )

    def test_lof_calibration_length_matches_enrollment(self) -> None:
        embeddings = _branch_embeddings(enroll_rows=31)
        for _branch, _weight, scorer in subject.fit_branch_lof(embeddings):
            self.assertEqual(len(scorer.calibration), 31)


class RankMonotonicityTests(unittest.TestCase):
    def test_score_is_nondecreasing_in_the_branch_lof_rank(self) -> None:
        policy, _fitted = _fit_small_policy()
        packed = _packed(8, 404)
        predicted = np.zeros(len(packed), dtype=np.int64)
        previous = None
        for level in np.linspace(0.0, 0.95, 12):
            score = policy.score(
                np.full(len(packed), float(level)), packed, predicted
            )
            if previous is not None:
                self.assertTrue(np.all(score >= previous - 1e-12))
            previous = score

    def test_empirical_rank_is_monotone_in_its_query(self) -> None:
        calibration = np.linspace(0.0, 1.0, 41)
        queries = np.linspace(-0.5, 1.5, 23)
        ranks = frozen.empirical_rank(queries, calibration)
        self.assertTrue(np.all(np.diff(ranks) >= 0.0))
        self.assertTrue(np.all((ranks >= 0.0) & (ranks < 1.0)))

    def test_lof_ensemble_is_a_convex_combination_of_component_ranks(
        self,
    ) -> None:
        embeddings = _branch_embeddings()
        components = subject.fit_branch_lof(embeddings)
        query = {
            branch: embeddings[branch]["va"] for branch in ("real", "complex")
        }
        combined = subject.score_branch_lof(components, query)
        expected = sum(
            weight * scorer.score(query[branch].astype(np.float64))
            for branch, weight, scorer in components
        )
        np.testing.assert_allclose(combined, expected)
        self.assertTrue(np.all((combined >= 0.0) & (combined < 1.0)))

    def test_lof_ensemble_refuses_non_convex_weights(self) -> None:
        components = subject.fit_branch_lof(_branch_embeddings())
        broken = [
            (branch, weight * 2.0, scorer)
            for branch, weight, scorer in components
        ]
        with self.assertRaisesRegex(ValueError, "sum to one"):
            subject.score_branch_lof(
                broken, {"real": np.zeros((2, 6)), "complex": np.zeros((2, 6))}
            )


class ClosedLabelInvarianceTests(unittest.TestCase):
    def _fixture(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(77)
        embeddings = rng.normal(size=(17, 5)).astype(np.float32)
        prototypes = rng.normal(size=(3, 5)).astype(np.float32)
        from train import nearest  # local import keeps module import cheap

        prediction, _distance = nearest(embeddings, prototypes)
        return embeddings, prototypes, np.asarray(prediction, dtype=np.int64)

    def test_additive_rejector_passes(self) -> None:
        embeddings, prototypes, prediction = self._fixture()
        scores = np.linspace(0.0, 0.9, len(prediction))
        result = subject.assert_closed_label_unchanged(
            embeddings, prototypes, prediction, scores
        )
        np.testing.assert_array_equal(result, prediction)

    def test_an_altered_label_is_refused(self) -> None:
        embeddings, prototypes, prediction = self._fixture()
        tampered = prediction.copy()
        tampered[0] = (tampered[0] + 1) % len(prototypes)
        with self.assertRaisesRegex(AssertionError, "additive only"):
            subject.assert_closed_label_unchanged(
                embeddings,
                prototypes,
                tampered,
                np.zeros(len(prediction)),
            )

    def test_misaligned_or_non_finite_scores_are_refused(self) -> None:
        embeddings, prototypes, prediction = self._fixture()
        with self.assertRaisesRegex(AssertionError, "aligned"):
            subject.assert_closed_label_unchanged(
                embeddings, prototypes, prediction, np.zeros(3)
            )
        broken = np.zeros(len(prediction))
        broken[2] = np.nan
        with self.assertRaisesRegex(AssertionError, "non-finite"):
            subject.assert_closed_label_unchanged(
                embeddings, prototypes, prediction, broken
            )

    def test_policy_score_does_not_expose_a_label_output(self) -> None:
        policy, _fitted = _fit_small_policy()
        packed = _packed(5, 505)
        predicted = np.zeros(len(packed), dtype=np.int64)
        score = policy.score(np.full(len(packed), 0.4), packed, predicted)
        self.assertEqual(score.shape, (len(packed),))
        self.assertTrue(np.issubdtype(score.dtype, np.floating))


class SealedPathRefusalTests(unittest.TestCase):
    def test_release_and_sealed_paths_are_refused(self) -> None:
        for candidate in (
            "/tmp/atom/training/artifacts/releases/seed20260731",
            "/tmp/atom/sealed_suite_v2",
            "/tmp/atom/RELEASES/x",
            "/tmp/atom/a_sealed_thing/out",
        ):
            with self.assertRaisesRegex(ValueError, "release or sealed"):
                subject.reject_sealed_path(Path(candidate), "output")

    def test_a_development_path_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            resolved = subject.reject_sealed_path(
                Path(directory) / "openset", "output"
            )
            self.assertTrue(str(resolved).endswith("openset"))

    def test_load_fusion_artifact_refuses_a_release_directory(self) -> None:
        with self.assertRaisesRegex(ValueError, "release or sealed"):
            subject.load_fusion_artifact(
                Path("/tmp/atom/artifacts/releases/v3_fusion")
            )

    def test_run_refuses_a_release_output_directory(self) -> None:
        args = argparse.Namespace(
            fusion_dir="/tmp/atom/dev/fusion",
            output_dir="/tmp/atom/artifacts/releases/openset",
            device="cpu",
            novelty_seeds=[20260939],
            novelty_n=4,
            prefix_lengths=[4096],
            reproduction_tolerance=1e-4,
        )
        with self.assertRaisesRegex(ValueError, "release or sealed"):
            subject.run(args)

    def test_release_seed_cannot_be_used_as_a_novelty_seed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unspent release seed"):
            subject.validate_novelty_seeds([20260731])

    def test_the_policy_design_seed_is_refused_as_evidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "designed the frozen policy"):
            subject.validate_novelty_seeds([20260939, 20260938])

    def test_default_novelty_seeds_are_the_untouched_pair(self) -> None:
        self.assertEqual(
            subject.DEFAULT_NOVELTY_SEEDS, (20260939, 20260940)
        )
        self.assertEqual(
            subject.validate_novelty_seeds(subject.DEFAULT_NOVELTY_SEEDS),
            (20260939, 20260940),
        )

    def test_duplicate_and_empty_seed_sets_are_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "distinct"):
            subject.validate_novelty_seeds([20260939, 20260939])
        with self.assertRaisesRegex(ValueError, "distinct"):
            subject.validate_novelty_seeds([])


class FusionArtifactValidationTests(unittest.TestCase):
    def _artifact(self, directory: Path, **overrides: object) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {
            "prototypes": ("fusion_prototypes.npy", np.zeros((3, 4), np.float32)),
            "real_center": ("real_center.npy", np.zeros(4, np.float32)),
            "complex_center": ("complex_center.npy", np.zeros(4, np.float32)),
            "feature_mean": ("feature_mean.npy", np.zeros(12, np.float32)),
            "feature_std": ("feature_std.npy", np.ones(12, np.float32)),
        }
        (directory / "fusion_state_dict.pt").write_bytes(b"not-a-real-checkpoint")
        artifacts: dict[str, object] = {
            "state_dict": "fusion_state_dict.pt",
            "state_dict_sha256": subject._sha256(
                directory / "fusion_state_dict.pt"
            ),
        }
        for key, (name, value) in arrays.items():
            np.save(directory / name, value)
            artifacts[key] = name
            artifacts[f"{key}_sha256"] = subject._sha256(directory / name)
        metrics = {
            "encoder": "fusion",
            "development_only": True,
            "release_evidence": False,
            "sealed_release_data_used": 0,
            "consumed_test_rows_used": 0,
            "data_audit": {"consumed_test_rows_exposed": 0},
            "frontend": {
                "version": "invariant-patch-time-domain-v1",
                "uses_frequency_transform": False,
            },
            "artifacts": artifacts,
            "fusion": {"weight_real": 0.5},
            "seed": 20260730,
            "device": "cpu",
            "closed_selection": {"balanced_accuracy": 0.86},
            "source_branches": {
                "real": {
                    "directory": str(directory / "branch_real"),
                    "sha256": {},
                },
                "complex": {
                    "directory": str(directory / "branch_complex"),
                    "sha256": {},
                },
            },
        }
        metrics.update(overrides)
        (directory / "dev_metrics.json").write_text(
            json.dumps(metrics), encoding="utf-8"
        )
        return directory

    def test_a_valid_artifact_loads_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            directory = self._artifact(Path(root) / "fusion")
            artifact = subject.load_fusion_artifact(directory)
            self.assertEqual(artifact.seed, 20260730)
            self.assertEqual(artifact.weight_real, 0.5)
            self.assertEqual(len(artifact.directory_sha256), 64)
            self.assertIn("dev_metrics.json", artifact.file_sha256)

    def test_a_v2_frontend_artifact_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            directory = self._artifact(
                Path(root) / "fusion",
                frontend={
                    "version": "invariant-patch-v1",
                    "uses_frequency_transform": True,
                },
            )
            with self.assertRaisesRegex(RuntimeError, "frontend"):
                subject.load_fusion_artifact(directory)

    def test_a_non_fusion_artifact_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            directory = self._artifact(Path(root) / "branch", encoder="real")
            with self.assertRaisesRegex(ValueError, "not 'fusion'"):
                subject.load_fusion_artifact(directory)

    def test_sealed_or_consumed_provenance_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            directory = self._artifact(
                Path(root) / "fusion", sealed_release_data_used=1
            )
            with self.assertRaisesRegex(RuntimeError, "sealed_release_data_used"):
                subject.load_fusion_artifact(directory)

    def test_a_tampered_artifact_file_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            directory = self._artifact(Path(root) / "fusion")
            np.save(directory / "real_center.npy", np.ones(4, np.float32))
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                subject.load_fusion_artifact(directory)

    def test_artifact_hash_is_stable_and_content_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            first = subject.load_fusion_artifact(
                self._artifact(Path(root) / "a")
            ).directory_sha256
            again = subject.load_fusion_artifact(
                Path(root) / "a"
            ).directory_sha256
            other = subject.load_fusion_artifact(
                self._artifact(Path(root) / "b", seed=20260732)
            ).directory_sha256
            self.assertEqual(first, again)
            self.assertNotEqual(first, other)


class NoveltyGenerationTests(unittest.TestCase):
    def _generate(self, seed: int = 20260939, n_each: int = 2):
        return subject.generate_novelty(
            [seed],
            n_each=n_each,
            lengths=(1024, 2048),
            feature_mean=np.zeros(12, np.float32),
            feature_std=np.ones(12, np.float32),
            patch_length=PATCH_LENGTH,
            patch_count=PATCH_COUNT,
            target_frac=0.5,
        )

    def test_families_lengths_and_shapes(self) -> None:
        fixtures, provenance = self._generate()
        self.assertEqual(set(fixtures[20260939]), {"noise", "chirp"})
        for family in ("noise", "chirp"):
            for length in (1024, 2048):
                fixture = fixtures[20260939][family][length]
                self.assertEqual(
                    fixture.packed.shape, (2, 2, PACKED_WIDTH)
                )
                self.assertEqual(fixture.features.shape, (2, 12))
                self.assertTrue(np.isfinite(fixture.packed).all())
        self.assertEqual(
            provenance["20260939"]["noise"]["generated_once_at"], 2048
        )

    def test_shorter_lengths_are_exact_prefixes_of_one_draw(self) -> None:
        _fixtures, provenance = self._generate()
        record = provenance["20260939"]["chirp"]
        self.assertEqual(
            record["prefix_sha256"]["2048"], record["longest_capture_sha256"]
        )
        self.assertNotEqual(
            record["prefix_sha256"]["1024"], record["prefix_sha256"]["2048"]
        )

    def test_generation_is_deterministic_for_a_seed(self) -> None:
        first, first_provenance = self._generate()
        second, second_provenance = self._generate()
        self.assertEqual(first_provenance, second_provenance)
        for family in ("noise", "chirp"):
            np.testing.assert_array_equal(
                first[20260939][family][1024].packed,
                second[20260939][family][1024].packed,
            )
            np.testing.assert_array_equal(
                first[20260939][family][1024].features,
                second[20260939][family][1024].features,
            )

    def test_different_seeds_produce_different_captures(self) -> None:
        _a, provenance_a = self._generate(seed=20260939)
        _b, provenance_b = self._generate(seed=20260940)
        self.assertNotEqual(
            provenance_a["20260939"]["noise"]["longest_capture_sha256"],
            provenance_b["20260940"]["noise"]["longest_capture_sha256"],
        )

    def test_invalid_lengths_and_counts_are_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "increasing"):
            subject.validate_prefix_lengths((8192, 4096))
        with self.assertRaisesRegex(ValueError, "increasing"):
            subject.validate_prefix_lengths((4096, 4096))
        with self.assertRaisesRegex(ValueError, "positive"):
            subject.validate_prefix_lengths((0,))

    def test_novelty_uses_the_v3_time_domain_frontend(self) -> None:
        with mock.patch.object(
            subject.td_preprocess,
            "preprocess",
            wraps=subject.td_preprocess.preprocess,
        ) as spy:
            self._generate(n_each=1)
        self.assertEqual(spy.call_count, 2 * 2 * 1)


class DeterminismTests(unittest.TestCase):
    def test_policy_fit_is_deterministic(self) -> None:
        first, fitted = _fit_small_policy()
        second = subject.fit_policy(
            fitted["training_packed"],
            fitted["training_labels"],
            fitted["enrollment_packed"],
            fitted["enrollment_predicted"],
            fitted["enrollment_ranks"],
            n_classes=2,
        )
        np.testing.assert_array_equal(
            first.geometry.class_mean, second.geometry.class_mean
        )
        np.testing.assert_array_equal(
            first.combined_calibration_raw, second.combined_calibration_raw
        )
        self.assertEqual(first.threshold, second.threshold)

    def test_scoring_is_deterministic(self) -> None:
        policy, _fitted = _fit_small_policy()
        packed = _packed(7, 606)
        predicted = np.arange(len(packed), dtype=np.int64) % 2
        ranks = np.linspace(0.1, 0.8, len(packed))
        np.testing.assert_array_equal(
            policy.score(ranks, packed, predicted),
            policy.score(ranks, packed, predicted),
        )

    def test_branch_lof_fit_is_deterministic(self) -> None:
        embeddings = _branch_embeddings()
        first = subject.fit_branch_lof(embeddings)
        second = subject.fit_branch_lof(embeddings)
        query = {
            branch: embeddings[branch]["va"] for branch in ("real", "complex")
        }
        np.testing.assert_array_equal(
            subject.score_branch_lof(first, query),
            subject.score_branch_lof(second, query),
        )

    def test_policy_round_trips_through_its_serialized_payload(self) -> None:
        policy, _fitted = _fit_small_policy()
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "policy.npz"
            np.savez_compressed(path, **policy.to_payload())
            with np.load(path, allow_pickle=False) as handle:
                restored = frozen.FrozenV3OpenSet.from_payload(
                    {name: handle[name] for name in handle.files}
                )
        self.assertEqual(restored.threshold, policy.threshold)
        packed = _packed(4, 707)
        predicted = np.zeros(len(packed), dtype=np.int64)
        ranks = np.full(len(packed), 0.3)
        np.testing.assert_array_equal(
            restored.score(ranks, packed, predicted),
            policy.score(ranks, packed, predicted),
        )

    def test_branch_lof_serialization_is_pickle_free_and_complete(self) -> None:
        components = subject.fit_branch_lof(_branch_embeddings())
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "lof.npz"
            subject.save_branch_lof(path, components)
            with np.load(path, allow_pickle=False) as handle:
                self.assertEqual(int(handle["component_count"]), 2)
                np.testing.assert_array_equal(
                    handle["component_0_calibration"],
                    components[0][2].calibration,
                )
                self.assertEqual(int(handle["component_1_neighbors"]), 64)


class GateReportingTests(unittest.TestCase):
    def _row(self, value: float) -> dict[str, object]:
        family = {
            "auroc": value,
            "threshold_recall": value,
            "score_median": value,
        }
        return {"noise": family, "chirp": dict(family), "overall": dict(family)}

    def test_gate_floors_are_the_frozen_values(self) -> None:
        self.assertEqual(
            subject.GATE_FLOORS,
            {
                "overall_auroc": 0.72,
                "noise_auroc": 0.80,
                "chirp_auroc": 0.80,
                "noise_threshold_recall": 0.10,
                "chirp_threshold_recall": 0.10,
            },
        )
        self.assertEqual(subject.KNOWN_FUR_CEILING, 0.10)

    def test_gates_report_the_worst_row(self) -> None:
        summary = subject.gate_summary(
            [self._row(0.95), self._row(0.81)], known_fur=0.05
        )
        self.assertEqual(summary["gates"]["noise_auroc"]["worst"], 0.81)
        self.assertTrue(summary["all_pass"])

    def test_a_failing_row_fails_the_summary(self) -> None:
        summary = subject.gate_summary(
            [self._row(0.95), self._row(0.05)], known_fur=0.05
        )
        self.assertFalse(summary["all_pass"])
        self.assertFalse(summary["gates"]["chirp_auroc"]["passes"])

    def test_a_high_known_false_unknown_rate_fails(self) -> None:
        summary = subject.gate_summary([self._row(0.95)], known_fur=0.5)
        self.assertFalse(summary["all_pass"])
        self.assertFalse(
            summary["gates"]["known_false_unknown_rate"]["passes"]
        )


class ReportShapeTests(unittest.TestCase):
    def test_json_writer_is_atomic_and_rejects_nan(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "openset_metrics.json"
            subject.write_json(path, {"a": 1, "b": [1.5, True]})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"a": 1, "b": [1.5, True]},
            )
            self.assertEqual(list(Path(root).iterdir()), [path])

    def test_non_finite_floats_are_recorded_as_null_not_nan(self) -> None:
        self.assertIsNone(subject._jsonable(float("nan")))
        self.assertIsNone(subject._jsonable(np.float64("inf")))

    def test_source_hashes_cover_the_fitting_chain(self) -> None:
        hashes = subject._source_hashes()
        for name in (
            "fit_v3_openset.py",
            "assemble_v3_fusion.py",
            "v3_time_domain_openset.py",
            "time_domain_geometry.py",
            "time_domain_invariant_patch_preprocess.py",
        ):
            self.assertIn(name, hashes)
            self.assertEqual(len(hashes[name]), 64)


class EndToEndRunTests(unittest.TestCase):
    """Drive ``run`` over a tiny synthetic fusion, without the real corpus.

    The corpus-bound half (``rebuild_populations``) is substituted; everything
    downstream of it -- density fit, enrollment ranks, threshold, novelty
    generation, scoring, gate reporting and the written contract -- is the real
    code path.
    """

    def _populations(self) -> "subject.Populations":
        import torch
        from invariant_patch_cnn import InvariantPatchCNN, InvariantPatchConfig
        from train import embed_all

        torch.manual_seed(0)
        common = {
            "patch_length": PATCH_LENGTH,
            "patch_count": PATCH_COUNT,
            "patch_dim": 8,
            "hidden": 12,
            "embed_dim": 6,
            "n_features": 12,
            "dropout": 0.0,
        }
        nets = {
            "real": InvariantPatchCNN(
                InvariantPatchConfig(encoder="real", **common)
            ).eval(),
            "complex": InvariantPatchCNN(
                InvariantPatchConfig(encoder="complex", **common)
            ).eval(),
        }
        device = torch.device("cpu")
        n_classes = 3
        rows = {"tr": 90, "en": 45, "va": 24}
        packed = {
            split: _packed(count, 900 + index)
            for index, (split, count) in enumerate(rows.items())
        }
        features = {
            split: np.random.default_rng(31 + index)
            .normal(size=(count, 12))
            .astype(np.float32)
            for index, (split, count) in enumerate(rows.items())
        }
        branch_embeddings = {
            name: {
                split: embed_all(
                    net, packed[split], features[split], device
                ).astype(np.float32)
                for split in ("tr", "en", "va")
            }
            for name, net in nets.items()
        }
        real_center = subject.assemble.fit_center(branch_embeddings["real"])
        complex_center = subject.assemble.fit_center(
            branch_embeddings["complex"]
        )
        fused = {
            split: subject.assemble.fuse_numpy(
                branch_embeddings["real"][split],
                branch_embeddings["complex"][split],
                real_center,
                complex_center,
                weight_real=0.5,
            )
            for split in ("tr", "en", "va")
        }
        labels = {
            split: (np.arange(count) % n_classes).astype(np.int64)
            for split, count in rows.items()
        }
        prototypes = subject.assemble.fit_prototypes(
            fused, labels["en"], n_classes
        )
        data = {
            "xtr": packed["tr"],
            "ytr": labels["tr"],
            "xen": packed["en"],
            "yen": labels["en"],
            "xva": packed["va"],
            "yva": labels["va"],
            "n_classes": n_classes,
            "fmean": np.zeros(12, np.float32),
            "fstd": np.ones(12, np.float32),
            "data_audit": {"consumed_test_rows_exposed": 0},
            "cache_contract": {
                "config": {
                    "patch_length": PATCH_LENGTH,
                    "patch_count": PATCH_COUNT,
                    "target_frac": 0.5,
                }
            },
        }
        return subject.Populations(
            data=data,
            nets=nets,
            branch_embeddings=branch_embeddings,
            fused=fused,
            real_center=real_center,
            complex_center=complex_center,
            prototypes=prototypes,
            reproduction={
                "tolerance": 1e-4,
                "bit_identical_to_artifact": True,
            },
            branch_hashes={"real": {}, "complex": {}},
        )

    def _run(self, root: Path) -> dict:
        fusion_dir = FusionArtifactValidationTests()._artifact(
            root / "fusion"
        )
        populations = self._populations()
        args = argparse.Namespace(
            fusion_dir=str(fusion_dir),
            output_dir=str(root / "openset"),
            device="cpu",
            novelty_seeds=[20260939],
            novelty_n=2,
            prefix_lengths=[1024, 2048],
            reproduction_tolerance=1e-4,
        )
        with mock.patch.object(
            subject, "rebuild_populations", return_value=populations
        ):
            return subject.run(args)

    def test_run_writes_the_declared_contract(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            report = self._run(Path(root))
            written = json.loads(
                (Path(root) / "openset" / "openset_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual(written["sealed_release_data_used"], 0)
        self.assertEqual(written["consumed_test_rows_used"], 0)
        self.assertFalse(written["release_seed_20260731_used"])
        self.assertFalse(written["closed_label_altered"])
        self.assertEqual(
            written["fitting_contract"]["density_population"], "training only"
        )
        self.assertEqual(
            written["fitting_contract"]["rank_and_threshold_population"],
            "enrollment only",
        )
        self.assertFalse(written["fitting_contract"]["policy_searched_here"])
        self.assertEqual(len(written["fusion"]["directory_sha256"]), 64)
        self.assertEqual(
            written["policy"]["branch_lof_rank_weight"],
            subject.BRANCH_LOF_RANK_WEIGHT,
        )
        self.assertEqual(
            written["policy"]["threshold_quantile"], subject.THRESHOLD_QUANTILE
        )
        self.assertEqual(written["seeds"]["novelty_seeds"], [20260939])
        self.assertEqual(written["seeds"]["model_seed"], 20260730)
        self.assertEqual(
            written["seeds"]["release_seed_not_spent"], 20260731
        )
        self.assertEqual(report["status"], written["status"])

    def test_run_reports_every_family_at_every_prefix_length(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            report = self._run(Path(root))
        rows = report["novelty"]["20260939"]
        self.assertEqual(set(rows), {"1024", "2048"})
        for row in rows.values():
            for family in ("noise", "chirp", "overall"):
                self.assertIn("auroc", row[family])
                self.assertIn("threshold_recall", row[family])
            self.assertEqual(
                row["known_false_unknown_rate"],
                report["known"]["false_unknown_rate"],
            )
        self.assertIn("known_false_unknown_rate", report["gates"])

    def test_run_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as first_root:
            first = self._run(Path(first_root))
        with tempfile.TemporaryDirectory() as second_root:
            second = self._run(Path(second_root))
        self.assertEqual(first["novelty"], second["novelty"])
        self.assertEqual(first["known"], second["known"])
        self.assertEqual(first["policy"]["threshold"], second["policy"]["threshold"])
        self.assertEqual(
            first["fixture_provenance"], second["fixture_provenance"]
        )

    def test_run_refuses_a_non_empty_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            existing = Path(root) / "openset"
            existing.mkdir()
            (existing / "stale.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                self._run(Path(root))


if __name__ == "__main__":
    unittest.main()
