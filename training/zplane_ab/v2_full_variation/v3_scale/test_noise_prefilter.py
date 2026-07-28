"""Unit suite for the staged noise/not-noise prefilter.

The properties under test are the ones that make or break the evidence claim,
not the ones that are easy to assert:

* **determinism** -- the same inputs produce bit-identical coefficients, a
  bit-identical threshold and a bit-identical bundle SHA;
* **train-only fitting** -- the coefficient fit refuses any population but
  training, and says so in its provenance;
* **enrollment-only thresholding** -- the operating point refuses any population
  but enrollment, is a stated budget rather than an inherited quantile, and
  leaves the fitted direction untouched;
* **ledger refusal** -- every spent, sealed, release and clean-validation seed is
  refused as a fitting seed, and the fit-only band is refused as validation;
* **serialisation round-trip** -- plain ``.npy`` / ``.json``, no pickle, no
  timestamp, exact arrays back, stable SHA, and a bundle that has been tampered
  with is refused;
* **phase and scale invariance inherited from the features** -- a global
  rotation or gain must not move the probability or the decision;
* **degenerate input** -- zero-variance columns, non-finite values, wrong
  widths, wrong capture lengths and a gate with no operating point all fail
  loudly;
* **the architecture contract change** -- the bundle must say, in machine
  readable form, that this component alters the closed label.

Nothing here reads the corpus, so the suite is fast and hermetic.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
_V2_DIR = os.path.dirname(HERE)
_ZPAB_DIR = os.path.dirname(_V2_DIR)
_TRAINING_DIR = os.path.dirname(_ZPAB_DIR)
for _path in (_TRAINING_DIR, _ZPAB_DIR, _V2_DIR, HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import noise_prefilter as npf  # noqa: E402
import pose_degeneracy as pdg  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _full_matrix(rows: int, offset: float, seed: int) -> np.ndarray:
    """A ``(rows, 21)`` pose-degeneracy-shaped matrix, separable by ``offset``.

    Values are plausible rather than real: the point of these fixtures is to
    exercise the estimator-free parts of the module, and the real feature
    behaviour is covered by ``test_pose_degeneracy`` and by the invariance test
    below, which uses genuine captures.
    """
    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=0.0, scale=1.0, size=(rows, pdg.FEATURE_COUNT))
    columns = npf.feature_columns()
    matrix[:, columns] += offset
    return np.ascontiguousarray(matrix)


def _known(rows: int = 80, seed: int = 11) -> np.ndarray:
    return _full_matrix(rows, 0.0, seed)


def _noise(rows: int = 80, seed: int = 12) -> np.ndarray:
    return _full_matrix(rows, 3.0, seed)


def _fitted(threshold: bool = True, **kwargs):
    model = npf.fit_prefilter(
        _known(),
        _noise(),
        capture_length=4096,
        known_population=npf.TRAIN_SPLIT,
        fitting_seed=npf.PROPOSED_FITTING_SEED,
        **kwargs,
    )
    if threshold:
        model = npf.fit_threshold(
            model,
            _known(rows=100, seed=13),
            population=npf.ENROLLMENT_SPLIT,
            budget=npf.DEFAULT_KNOWN_FALSE_POSITIVE_BUDGET,
        )
    return model


def _multitone(length: int, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = np.arange(length, dtype=np.float64)
    centre = 0.11328125
    offsets = np.array((-0.03125, -0.015625, 0.0, 0.015625, 0.03125))
    phases = rng.uniform(-np.pi, np.pi, size=offsets.shape)
    return np.sum(
        np.exp(
            1j
            * (
                2.0 * np.pi * (centre + offsets)[:, None] * samples
                + phases[:, None]
            )
        ),
        axis=0,
    )


# ---------------------------------------------------------------------------
# feature selection
# ---------------------------------------------------------------------------


class FeatureSelectionTest(unittest.TestCase):
    def test_uses_a_handful_of_features_and_names_them(self):
        self.assertEqual(len(npf.PREFILTER_FEATURES), 7)
        self.assertEqual(
            len(set(npf.PREFILTER_FEATURES)), len(npf.PREFILTER_FEATURES)
        )
        for name in npf.PREFILTER_FEATURES:
            self.assertIn(name, pdg.FEATURE_NAMES)
            self.assertIn(name, npf.FEATURE_RATIONALE)
            self.assertGreater(len(npf.FEATURE_RATIONALE[name]), 40)

    def test_the_four_measured_dead_features_are_dropped(self):
        for name in (
            "degenerate_full_band_fallback",
            "bandwidth_clipped_to_full",
            "zero_magnitude_prefix_fraction",
            "quotient_clip_fraction",
        ):
            with self.subTest(feature=name):
                self.assertNotIn(name, npf.PREFILTER_FEATURES)
                self.assertIn(name, npf.DROPPED_FEATURES)

    def test_every_source_feature_is_either_used_or_explained(self):
        used = set(npf.PREFILTER_FEATURES)
        dropped = set(npf.DROPPED_FEATURES)
        self.assertEqual(used & dropped, set())
        self.assertEqual(used | dropped, set(pdg.FEATURE_NAMES))
        for name, reason in npf.DROPPED_FEATURES.items():
            with self.subTest(feature=name):
                self.assertGreater(len(reason), 30)

    def test_the_four_strong_noise_features_are_all_retained(self):
        # HANDOFF 19.1: these are the measured noise separators.  A prefilter
        # that silently dropped one of them would be a different experiment.
        for name in (
            "carrier_phase_coherence",
            "log10_relative_lag_magnitude_mean",
            "relative_lag_magnitude_decay_slope",
            "high_lag_qualified_fraction",
        ):
            self.assertIn(name, npf.PREFILTER_FEATURES)

    def test_the_chirp_carrier_is_deliberately_left_downstream(self):
        self.assertNotIn("prefix_centre_dispersion", npf.PREFILTER_FEATURES)
        self.assertIn("chirp", npf.DROPPED_FEATURES["prefix_centre_dispersion"].lower())

    def test_rationale_table_is_serialisable_and_complete(self):
        table = npf.selection_rationale()
        payload = json.loads(json.dumps(table))
        self.assertEqual(payload["source_feature_count"], pdg.FEATURE_COUNT)
        self.assertEqual(payload["used_feature_count"], len(npf.PREFILTER_FEATURES))
        self.assertEqual(
            len(payload["used"]) + len(payload["dropped"]), pdg.FEATURE_COUNT
        )

    def test_column_selection_matches_the_frozen_feature_order(self):
        columns = npf.feature_columns()
        for index, name in zip(columns, npf.PREFILTER_FEATURES):
            self.assertEqual(pdg.FEATURE_NAMES[int(index)], name)
        matrix = _known(rows=4)
        selected = npf.select_features(matrix)
        self.assertEqual(selected.shape, (4, len(npf.PREFILTER_FEATURES)))
        np.testing.assert_array_equal(selected, matrix[:, columns])

    def test_unknown_or_duplicated_features_are_refused(self):
        with self.assertRaises(ValueError):
            npf.feature_columns(("not_a_feature",))
        with self.assertRaises(ValueError):
            npf.feature_columns(("carrier_phase_coherence",) * 2)
        with self.assertRaises(ValueError):
            npf.feature_columns(())


# ---------------------------------------------------------------------------
# the seed ledger
# ---------------------------------------------------------------------------


class SeedLedgerTest(unittest.TestCase):
    def test_spent_seeds_are_refused_as_fitting_seeds(self):
        for seed in (20260938, 20260939, 20260940, 20260941):
            with self.subTest(seed=seed):
                with self.assertRaises(ValueError) as caught:
                    npf.validate_fitting_seed(seed)
                self.assertIn("spent", str(caught.exception))

    def test_sealed_and_release_seeds_are_refused_as_fitting_seeds(self):
        with self.assertRaises(ValueError) as sealed:
            npf.validate_fitting_seed(npf.CONSUMED_SEALED_SEED)
        self.assertIn("sealed", str(sealed.exception))
        with self.assertRaises(ValueError) as release:
            npf.validate_fitting_seed(npf.RELEASE_SEED_NEVER_SPENT_HERE)
        self.assertIn("release", str(release.exception))

    def test_clean_validation_seeds_are_refused_as_fitting_seeds(self):
        # 20260942 onward are the validation evidence for this work.  Fitting
        # against one would convert it, which is the exact failure the ledger
        # exists to prevent.
        for seed in (20260942, 20260943, 20260999):
            with self.subTest(seed=seed):
                with self.assertRaises(ValueError) as caught:
                    npf.validate_fitting_seed(seed)
                self.assertIn("validation", str(caught.exception))

    def test_the_whole_novelty_namespace_is_refused_as_a_block(self):
        low, high = npf.NOVELTY_SEED_NAMESPACE
        for seed in (low, (low + high) // 2, high):
            with self.subTest(seed=seed):
                with self.assertRaises(ValueError):
                    npf.validate_fitting_seed(seed)

    def test_a_fitting_seed_must_come_from_the_fit_only_band(self):
        low, high = npf.FITTING_SEED_NAMESPACE
        self.assertEqual(npf.validate_fitting_seed(low), low)
        self.assertEqual(npf.validate_fitting_seed(high), high)
        self.assertEqual(
            npf.validate_fitting_seed(npf.PROPOSED_FITTING_SEED),
            npf.PROPOSED_FITTING_SEED,
        )
        for seed in (low - 1, high + 1, 0, 12345):
            with self.subTest(seed=seed):
                with self.assertRaises(ValueError):
                    npf.validate_fitting_seed(seed)

    def test_the_two_namespaces_cannot_overlap(self):
        novelty = set(range(*npf.NOVELTY_SEED_NAMESPACE))
        fitting = set(range(*npf.FITTING_SEED_NAMESPACE))
        self.assertEqual(novelty & fitting, set())

    def test_non_integer_fitting_seeds_are_refused(self):
        for value in ("20261001", 20261001.5, None, True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    npf.validate_fitting_seed(value)

    def test_validation_seeds_mirror_the_refusals(self):
        self.assertEqual(
            npf.validate_validation_seeds([20260942, 20260943]),
            (20260942, 20260943),
        )
        for seeds in (
            [],
            [20260942, 20260942],
            [20260939],
            [20260941],
            [npf.CONSUMED_SEALED_SEED],
            [npf.RELEASE_SEED_NEVER_SPENT_HERE],
            [npf.PROPOSED_FITTING_SEED],
            [20260900],
        ):
            with self.subTest(seeds=seeds):
                with self.assertRaises(ValueError):
                    npf.validate_validation_seeds(seeds)

    def test_fitting_the_model_revalidates_the_declared_seed(self):
        # The caller has already drawn the noise by this point; the seed is
        # still checked, because the emitted provenance claims it.
        with self.assertRaises(ValueError):
            npf.fit_prefilter(
                _known(),
                _noise(),
                capture_length=4096,
                known_population=npf.TRAIN_SPLIT,
                fitting_seed=20260939,
            )

    def test_noise_synthesis_refuses_a_ledger_seed_before_generating(self):
        with self.assertRaises(ValueError):
            npf.synthesize_fitting_noise(
                fitting_seed=20260940, n_rows=1, lengths=(256,)
            )


class FittingNoiseTest(unittest.TestCase):
    """The fitting noise must be reproducible and must follow its seed.

    Short lengths keep this hermetic and fast; the draw protocol, not the
    capture length, is what is under test.
    """

    def test_the_draw_is_reproducible_from_the_seed_alone(self):
        first, first_provenance = npf.synthesize_fitting_noise(
            fitting_seed=npf.PROPOSED_FITTING_SEED, n_rows=3, lengths=(256, 512)
        )
        second, second_provenance = npf.synthesize_fitting_noise(
            fitting_seed=npf.PROPOSED_FITTING_SEED, n_rows=3, lengths=(256, 512)
        )
        self.assertTrue(np.array_equal(first[256], second[256]))
        self.assertTrue(np.array_equal(first[512], second[512]))
        self.assertEqual(
            first_provenance["longest_capture_sha256"],
            second_provenance["longest_capture_sha256"],
        )

    def test_a_different_fit_seed_draws_different_noise(self):
        _first, first_provenance = npf.synthesize_fitting_noise(
            fitting_seed=20261001, n_rows=3, lengths=(256,)
        )
        _second, second_provenance = npf.synthesize_fitting_noise(
            fitting_seed=20261002, n_rows=3, lengths=(256,)
        )
        self.assertNotEqual(
            first_provenance["longest_capture_sha256"],
            second_provenance["longest_capture_sha256"],
        )

    def test_only_the_noise_family_is_drawn(self):
        _matrices, provenance = npf.synthesize_fitting_noise(
            fitting_seed=npf.PROPOSED_FITTING_SEED, n_rows=2, lengths=(256,)
        )
        self.assertEqual(provenance["family"], "noise")
        self.assertFalse(provenance["chirp_drawn"])
        self.assertEqual(provenance["seed_role"], "fit only")

    def test_bad_lengths_and_row_counts_are_refused(self):
        for kwargs in (
            {"n_rows": 0, "lengths": (256,)},
            {"n_rows": 2, "lengths": ()},
            {"n_rows": 2, "lengths": (0,)},
            {"n_rows": 2, "lengths": (512, 256)},
            {"n_rows": 2, "lengths": (256, 256)},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    npf.synthesize_fitting_noise(
                        fitting_seed=npf.PROPOSED_FITTING_SEED, **kwargs
                    )

    def test_only_train_and_enroll_known_rows_are_reachable(self):
        for split in (npf.SELECTION_SPLIT, "test", "sealed"):
            with self.subTest(split=split):
                with self.assertRaises(ValueError) as caught:
                    npf.load_known_features(split, lengths=(4096,))
                self.assertIn("reachable", str(caught.exception))


# ---------------------------------------------------------------------------
# train-only fitting
# ---------------------------------------------------------------------------


class TrainOnlyFitTest(unittest.TestCase):
    def test_only_the_training_population_may_be_fitted_on(self):
        for population in (npf.ENROLLMENT_SPLIT, npf.SELECTION_SPLIT, "test", ""):
            with self.subTest(population=population):
                with self.assertRaises(ValueError) as caught:
                    npf.fit_prefilter(
                        _known(),
                        _noise(),
                        capture_length=4096,
                        known_population=population,
                        fitting_seed=npf.PROPOSED_FITTING_SEED,
                    )
                self.assertIn("fitted on", str(caught.exception))

    def test_fit_provenance_states_the_populations_it_did_not_touch(self):
        model = _fitted(threshold=False)
        provenance = model.fit_provenance
        self.assertEqual(provenance["known_population"], npf.TRAIN_SPLIT)
        self.assertEqual(provenance["enrollment_rows_used_in_fit"], 0)
        self.assertEqual(provenance["selection_rows_used_in_fit"], 0)
        self.assertEqual(provenance["consumed_test_rows_used_in_fit"], 0)
        self.assertEqual(provenance["fitting_seed"], npf.PROPOSED_FITTING_SEED)
        self.assertTrue(provenance["fitting_seed_is_not_a_validation_seed"])

    def test_the_fitted_direction_does_not_depend_on_the_noise_row_count(self):
        # Class weights and the standardiser are both balanced, so drawing more
        # noise must not silently reweight the fit.
        few = npf.fit_prefilter(
            _known(),
            _noise(rows=40, seed=21),
            capture_length=4096,
            known_population=npf.TRAIN_SPLIT,
            fitting_seed=npf.PROPOSED_FITTING_SEED,
        )
        many = npf.fit_prefilter(
            _known(),
            np.vstack([_noise(rows=40, seed=21)] * 3),
            capture_length=4096,
            known_population=npf.TRAIN_SPLIT,
            fitting_seed=npf.PROPOSED_FITTING_SEED,
        )
        np.testing.assert_allclose(
            few.coefficients, many.coefficients, rtol=1e-9, atol=1e-9
        )
        np.testing.assert_allclose(few.mean, many.mean, rtol=1e-9, atol=1e-9)

    def test_an_unregularised_fit_is_refused(self):
        for penalty in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(l2=penalty):
                with self.assertRaises(ValueError):
                    _fitted(threshold=False, l2=penalty)

    def test_the_fit_converges_and_reports_how(self):
        model = _fitted(threshold=False)
        self.assertTrue(model.fit_provenance["converged"])
        self.assertLessEqual(
            model.fit_provenance["iterations"], npf.MAX_NEWTON_ITERATIONS
        )
        self.assertEqual(model.fit_provenance["l2_penalty"], npf.L2_PENALTY)

    def test_coefficients_point_the_measured_way(self):
        # The fixture makes noise the HIGH side of every selected column, so a
        # correct fit must give every coefficient a positive sign.  This is a
        # sanity check on the solver, not a claim about real data.
        model = _fitted(threshold=False)
        self.assertTrue(np.all(model.coefficients > 0.0))


# ---------------------------------------------------------------------------
# enrollment-only threshold
# ---------------------------------------------------------------------------


class EnrollmentOnlyThresholdTest(unittest.TestCase):
    def test_only_the_enrollment_population_may_set_the_operating_point(self):
        model = _fitted(threshold=False)
        for population in (npf.TRAIN_SPLIT, npf.SELECTION_SPLIT, "test"):
            with self.subTest(population=population):
                with self.assertRaises(ValueError) as caught:
                    npf.fit_threshold(
                        model, _known(), population=population, budget=0.02
                    )
                self.assertIn("chosen on", str(caught.exception))

    def test_thresholding_does_not_move_the_fitted_direction(self):
        before = _fitted(threshold=False)
        after = npf.fit_threshold(
            before,
            _known(rows=100, seed=13),
            population=npf.ENROLLMENT_SPLIT,
            budget=0.05,
        )
        self.assertTrue(np.array_equal(before.coefficients, after.coefficients))
        self.assertTrue(np.array_equal(before.mean, after.mean))
        self.assertTrue(np.array_equal(before.scale, after.scale))
        self.assertEqual(before.intercept, after.intercept)
        self.assertFalse(before.has_threshold)
        self.assertTrue(after.has_threshold)

    def test_the_budget_is_honoured_exactly_on_the_enrollment_rows(self):
        model = _fitted(threshold=False)
        enrollment = _known(rows=200, seed=17)
        for budget in (0.0, 0.01, 0.02, 0.05, 0.1, 0.25):
            with self.subTest(budget=budget):
                gated = npf.fit_threshold(
                    model,
                    enrollment,
                    population=npf.ENROLLMENT_SPLIT,
                    budget=budget,
                )
                realised = float(np.mean(gated.decide(enrollment)))
                self.assertLessEqual(realised, budget + 1e-12)
                self.assertEqual(
                    realised,
                    gated.threshold_provenance[
                        "realised_known_false_positive_rate"
                    ],
                )

    def test_the_budget_threshold_is_the_smallest_one_that_fits(self):
        # A threshold that is conservative by accident throws away recall it was
        # allowed to have, so the chosen point must be tight against the budget.
        scores = np.arange(100, dtype=np.float64)
        threshold, detail = npf.budget_threshold(scores, 0.05)
        self.assertEqual(detail["allowance"], 5)
        self.assertEqual(int(np.count_nonzero(scores >= threshold)), 5)
        lower = float(np.nextafter(threshold, -np.inf))
        self.assertGreater(int(np.count_nonzero(scores >= lower)), 5)

    def test_ties_cannot_overspend_the_budget(self):
        scores = np.zeros(100, dtype=np.float64)
        threshold, _detail = npf.budget_threshold(scores, 0.05)
        self.assertEqual(int(np.count_nonzero(scores >= threshold)), 0)

    def test_a_zero_budget_gates_no_known_row(self):
        scores = np.arange(10, dtype=np.float64)
        threshold, _detail = npf.budget_threshold(scores, 0.0)
        self.assertEqual(int(np.count_nonzero(scores >= threshold)), 0)

    def test_the_threshold_lives_in_score_space_not_probability_space(self):
        # Probabilities saturate to exactly 1.0 in float64; a quantile of a
        # saturated column is not an operating point.
        model = _fitted()
        self.assertEqual(
            model.threshold_provenance["threshold_space"].split()[0], "linear"
        )
        self.assertTrue(np.isfinite(model.threshold_score))
        self.assertTrue(0.0 <= model.threshold_probability <= 1.0)

    def test_the_operating_point_is_not_inherited_from_another_model(self):
        provenance = _fitted().threshold_provenance
        self.assertEqual(provenance["population"], npf.ENROLLMENT_SPLIT)
        self.assertFalse(provenance["inherited_from_another_model"])
        self.assertEqual(provenance["training_rows_used_for_threshold"], 0)
        self.assertEqual(provenance["novelty_rows_used_for_threshold"], 0)
        self.assertEqual(provenance["selection_rows_used_for_threshold"], 0)

    def test_a_fixed_score_operating_point_is_available_and_explicit(self):
        model = _fitted(threshold=False)
        fixed = npf.fit_threshold(
            model,
            _known(rows=50, seed=19),
            population=npf.ENROLLMENT_SPLIT,
            policy=npf.FIXED_SCORE_POLICY,
            fixed_score=0.25,
        )
        self.assertEqual(fixed.threshold_score, 0.25)
        self.assertIsNone(
            fixed.threshold_provenance["known_false_positive_budget"]
        )
        with self.assertRaises(ValueError):
            npf.fit_threshold(
                model,
                _known(rows=50, seed=19),
                population=npf.ENROLLMENT_SPLIT,
                policy=npf.FIXED_SCORE_POLICY,
            )

    def test_unknown_policies_and_impossible_budgets_are_refused(self):
        model = _fitted(threshold=False)
        with self.assertRaises(ValueError):
            npf.fit_threshold(
                model, _known(), population=npf.ENROLLMENT_SPLIT, policy="q95"
            )
        for budget in (-0.01, 1.0, 2.0, float("nan")):
            with self.subTest(budget=budget):
                with self.assertRaises(ValueError):
                    npf.budget_threshold(np.arange(10.0), budget)


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


class DeterminismTest(unittest.TestCase):
    def test_refitting_the_same_inputs_is_bit_identical(self):
        first = _fitted()
        second = _fitted()
        self.assertTrue(np.array_equal(first.mean, second.mean))
        self.assertTrue(np.array_equal(first.scale, second.scale))
        self.assertTrue(np.array_equal(first.coefficients, second.coefficients))
        self.assertEqual(first.intercept, second.intercept)
        self.assertEqual(first.threshold_score, second.threshold_score)

    def test_scoring_is_bit_identical_on_repeated_calls(self):
        model = _fitted()
        rows = _noise(rows=30, seed=23)
        self.assertTrue(np.array_equal(model.score(rows), model.score(rows)))
        self.assertTrue(
            np.array_equal(model.predict_proba(rows), model.predict_proba(rows))
        )
        self.assertTrue(np.array_equal(model.decide(rows), model.decide(rows)))

    def test_the_bundle_sha_is_stable_across_writes(self):
        model = _fitted()
        with tempfile.TemporaryDirectory() as directory:
            first = model.save(Path(directory) / "a")
            second = model.save(Path(directory) / "b")
            self.assertEqual(npf.bundle_sha256(first), npf.bundle_sha256(second))

    def test_the_bundle_carries_no_timestamp(self):
        model = _fitted()
        with tempfile.TemporaryDirectory() as directory:
            path = model.save(Path(directory) / "bundle")
            text = (path / "meta.json").read_text(encoding="utf-8")
        lowered = text.lower()
        for banned in ("created", "timestamp", "elapsed", "_utc", "hostname"):
            with self.subTest(token=banned):
                self.assertNotIn(banned, lowered)


# ---------------------------------------------------------------------------
# serialisation
# ---------------------------------------------------------------------------


class SerialisationTest(unittest.TestCase):
    def test_round_trip_is_exact(self):
        model = _fitted()
        with tempfile.TemporaryDirectory() as directory:
            path = model.save(Path(directory) / "bundle")
            restored = npf.NoisePrefilter.load(path)
        self.assertEqual(restored.feature_names, model.feature_names)
        self.assertEqual(restored.capture_length, model.capture_length)
        self.assertTrue(np.array_equal(restored.mean, model.mean))
        self.assertTrue(np.array_equal(restored.scale, model.scale))
        self.assertTrue(np.array_equal(restored.coefficients, model.coefficients))
        self.assertEqual(restored.intercept, model.intercept)
        self.assertEqual(restored.threshold_score, model.threshold_score)
        rows = _noise(rows=25, seed=27)
        self.assertTrue(np.array_equal(restored.score(rows), model.score(rows)))
        self.assertTrue(np.array_equal(restored.decide(rows), model.decide(rows)))

    def test_the_bundle_is_exactly_the_declared_plain_files(self):
        model = _fitted()
        with tempfile.TemporaryDirectory() as directory:
            path = model.save(Path(directory) / "bundle")
            present = sorted(entry.name for entry in path.iterdir())
            self.assertEqual(tuple(present), npf.BUNDLE_FILES)
            for name in present:
                with self.subTest(name=name):
                    self.assertTrue(name.endswith(".npy") or name == "meta.json")

    def test_no_array_needs_pickle_to_load(self):
        model = _fitted()
        with tempfile.TemporaryDirectory() as directory:
            path = model.save(Path(directory) / "bundle")
            for name in npf.BUNDLE_FILES:
                if not name.endswith(".npy"):
                    continue
                with self.subTest(name=name):
                    values = np.load(path / name, allow_pickle=False)
                    self.assertEqual(values.dtype, np.dtype(np.float64))

    def test_a_model_without_an_operating_point_round_trips_too(self):
        model = _fitted(threshold=False)
        with tempfile.TemporaryDirectory() as directory:
            path = model.save(Path(directory) / "bundle")
            restored = npf.NoisePrefilter.load(path)
        self.assertFalse(restored.has_threshold)
        with self.assertRaises(ValueError):
            restored.decide(_noise(rows=3, seed=29))

    def test_a_tampered_bundle_is_refused(self):
        model = _fitted()
        with tempfile.TemporaryDirectory() as directory:
            path = model.save(Path(directory) / "bundle")
            npf._write_array(
                path / "threshold_score.npy",
                np.asarray([model.threshold_score + 1.0], dtype=np.float64),
            )
            with self.assertRaises(ValueError) as caught:
                npf.NoisePrefilter.load(path)
            self.assertIn("integrity", str(caught.exception))

    def test_an_unexpected_file_invalidates_the_sha(self):
        model = _fitted()
        with tempfile.TemporaryDirectory() as directory:
            path = model.save(Path(directory) / "bundle")
            (path / "stray.txt").write_text("x", encoding="utf-8")
            with self.assertRaises(ValueError):
                npf.bundle_sha256(path)

    def test_a_foreign_bundle_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle"
            path.mkdir()
            (path / "meta.json").write_text('{"schema": "other"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                npf.NoisePrefilter.load(path)

    def test_a_content_change_changes_the_sha(self):
        model = _fitted()
        other = npf.fit_threshold(
            npf.fit_prefilter(
                _known(),
                _noise(),
                capture_length=4096,
                known_population=npf.TRAIN_SPLIT,
                fitting_seed=npf.PROPOSED_FITTING_SEED,
            ),
            _known(rows=100, seed=13),
            population=npf.ENROLLMENT_SPLIT,
            budget=0.10,
        )
        with tempfile.TemporaryDirectory() as directory:
            first = model.save(Path(directory) / "a")
            second = other.save(Path(directory) / "b")
            self.assertNotEqual(npf.bundle_sha256(first), npf.bundle_sha256(second))

    def test_a_length_set_round_trips_and_hashes(self):
        models = {}
        for length in (4096, 8192):
            models[length] = npf.fit_threshold(
                npf.fit_prefilter(
                    _known(),
                    _noise(),
                    capture_length=length,
                    known_population=npf.TRAIN_SPLIT,
                    fitting_seed=npf.PROPOSED_FITTING_SEED,
                ),
                _known(rows=100, seed=13),
                population=npf.ENROLLMENT_SPLIT,
                budget=0.02,
            )
        with tempfile.TemporaryDirectory() as directory:
            path = npf.save_prefilter_set(Path(directory) / "set", models)
            restored = npf.load_prefilter_set(path)
            self.assertEqual(sorted(restored), [4096, 8192])
            self.assertEqual(restored[8192].capture_length, 8192)
            digest = npf.prefilter_set_sha256(path)
            copy = Path(directory) / "copy"
            shutil.copytree(path, copy)
            self.assertEqual(npf.prefilter_set_sha256(copy), digest)

    def test_a_model_filed_under_the_wrong_length_is_refused(self):
        model = _fitted()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                npf.save_prefilter_set(Path(directory) / "set", {8192: model})


# ---------------------------------------------------------------------------
# the architecture contract change
# ---------------------------------------------------------------------------


class ArchitectureContractTest(unittest.TestCase):
    def test_metadata_says_this_component_alters_the_closed_label(self):
        meta = _fitted().metadata()
        self.assertFalse(meta["additive_only"])
        self.assertTrue(meta["changes_closed_label"])
        self.assertTrue(meta["gates_before_classification"])
        self.assertIn("deliberate", meta["architecture_contract_change"])

    def test_the_contract_keys_are_present_in_the_saved_bundle(self):
        model = _fitted()
        with tempfile.TemporaryDirectory() as directory:
            path = model.save(Path(directory) / "bundle")
            payload = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        for key in npf.CONTRACT_KEYS_FOR_RELEASE_CLAIM:
            with self.subTest(key=key):
                self.assertIn(key, payload)

    def test_the_evidence_contract_is_recorded_alongside_it(self):
        meta = _fitted().metadata()
        self.assertEqual(meta["fit_population"], npf.TRAIN_SPLIT)
        self.assertEqual(meta["threshold_population"], npf.ENROLLMENT_SPLIT)
        self.assertEqual(meta["selection_rows_used"], 0)
        self.assertEqual(meta["consumed_test_rows_used"], 0)
        self.assertEqual(meta["sealed_release_paths_read"], 0)
        self.assertEqual(
            meta["release_seed_not_spent"], npf.RELEASE_SEED_NEVER_SPENT_HERE
        )

    def test_the_frequency_transform_claim_is_split_honestly(self):
        meta = _fitted().metadata()
        self.assertFalse(meta["inference_uses_frequency_transform"])
        self.assertTrue(meta["fitting_noise_generator_uses_frequency_transform"])
        self.assertFalse(meta["uses_absolute_scale"])

    def test_it_says_it_only_decides_noise(self):
        self.assertIn("chirp", _fitted().metadata()["decides"])


# ---------------------------------------------------------------------------
# scoring, gating and the measured saving
# ---------------------------------------------------------------------------


class ScoringTest(unittest.TestCase):
    def test_probabilities_are_the_logistic_of_the_score(self):
        model = _fitted()
        rows = np.vstack([_known(rows=20, seed=31), _noise(rows=20, seed=32)])
        scores = model.score(rows)
        probabilities = model.predict_proba(rows)
        self.assertTrue(np.all(probabilities >= 0.0))
        self.assertTrue(np.all(probabilities <= 1.0))
        np.testing.assert_allclose(
            probabilities, 1.0 / (1.0 + np.exp(-scores)), rtol=1e-12, atol=1e-12
        )
        # monotone: the ordering by probability is the ordering by score
        self.assertTrue(
            np.array_equal(np.argsort(scores, kind="stable"),
                           np.argsort(probabilities, kind="stable"))
        )

    def test_the_hard_gate_is_the_score_against_the_threshold(self):
        model = _fitted()
        rows = np.vstack([_known(rows=20, seed=33), _noise(rows=20, seed=34)])
        expected = model.score(rows) >= model.threshold_score
        self.assertTrue(np.array_equal(model.decide(rows), expected))

    def test_the_gate_refuses_to_run_without_an_operating_point(self):
        model = _fitted(threshold=False)
        with self.assertRaises(ValueError) as caught:
            model.decide(_noise(rows=4, seed=35))
        self.assertIn("enrollment", str(caught.exception))

    def test_gate_returns_scores_decisions_and_the_measured_saving(self):
        model = _fitted()
        rows = np.vstack([_known(rows=40, seed=36), _noise(rows=60, seed=37)])
        result = model.gate(rows)
        self.assertEqual(result["score"].shape, (100,))
        self.assertEqual(result["gated_as_noise"].dtype, np.dtype(bool))
        saving = result["compute_saving"]
        self.assertEqual(saving["rows"], 100)
        self.assertEqual(
            saving["gated_rows"] + saving["downstream_rows_evaluated"], 100
        )
        self.assertAlmostEqual(
            saving["gated_fraction"], saving["gated_rows"] / 100.0
        )
        self.assertTrue(saving["measured"])

    def test_compute_saving_is_arithmetic_not_a_claim(self):
        saving = npf.compute_saving([True, False, True, True])
        self.assertEqual(saving["rows"], 4)
        self.assertEqual(saving["gated_rows"], 3)
        self.assertEqual(saving["downstream_rows_evaluated"], 1)
        self.assertEqual(saving["gated_fraction"], 0.75)
        with self.assertRaises(ValueError):
            npf.compute_saving([])

    def test_a_capture_length_mismatch_is_refused(self):
        model = _fitted()
        rows = _noise(rows=5, seed=38)
        model.score(rows, capture_length=4096)
        with self.assertRaises(ValueError) as caught:
            model.score(rows, capture_length=8192)
        self.assertIn("length dependent", str(caught.exception))

    def test_selected_and_full_matrices_score_identically(self):
        model = _fitted()
        rows = _noise(rows=12, seed=39)
        self.assertTrue(
            np.array_equal(model.score(rows), model.score(npf.select_features(rows)))
        )


# ---------------------------------------------------------------------------
# reporting: threshold recall is the point, AUROC is context
# ---------------------------------------------------------------------------


class EvaluationTest(unittest.TestCase):
    def test_both_auroc_and_threshold_recall_are_reported(self):
        model = _fitted()
        report = npf.evaluate(
            model,
            _known(rows=120, seed=41),
            _noise(rows=120, seed=42),
            known_population=npf.ENROLLMENT_SPLIT,
        )
        self.assertIn("noise_auroc", report)
        self.assertIn("noise_threshold_recall", report)
        self.assertIn("known_false_positive_rate", report)
        self.assertGreaterEqual(report["noise_auroc"], 0.9)
        self.assertGreaterEqual(report["noise_threshold_recall"], 0.9)
        self.assertTrue(report["passes_noise_auroc"])
        self.assertTrue(report["passes_noise_threshold_recall"])
        # The budget is a guarantee on the ENROLLMENT rows the threshold was
        # chosen on, not on an unseen draw, so this asserts the flag exists and
        # is honestly computed rather than asserting it is always true.
        self.assertIsInstance(report["within_known_false_positive_budget"], bool)
        self.assertEqual(
            report["within_known_false_positive_budget"],
            report["known_false_positive_rate"]
            <= report["known_false_positive_budget"],
        )
        self.assertLessEqual(
            report["known_false_positive_rate"],
            report["known_false_unknown_ceiling"],
        )
        json.dumps(report)

    def test_recall_is_undefined_without_an_operating_point(self):
        report = npf.evaluate(
            _fitted(threshold=False),
            _known(rows=30, seed=43),
            _noise(rows=30, seed=44),
            known_population=npf.ENROLLMENT_SPLIT,
        )
        self.assertIsNone(report["noise_threshold_recall"])

    def test_the_gates_are_imported_from_fit_v3_openset_not_retyped(self):
        import fit_v3_openset as base

        floors = npf.gate_floors()
        self.assertEqual(floors["noise_auroc"], base.GATE_FLOORS["noise_auroc"])
        self.assertEqual(
            floors["noise_threshold_recall"],
            base.GATE_FLOORS["noise_threshold_recall"],
        )
        self.assertEqual(
            floors["known_false_unknown_ceiling"], base.KNOWN_FUR_CEILING
        )

    def test_the_default_budget_leaves_room_under_the_house_ceiling(self):
        # HANDOFF 17: the existing v3 rejector already spends 0.0466 of the
        # 0.10 known-false-unknown ceiling.  A staged design spends both.
        ceiling = npf.gate_floors()["known_false_unknown_ceiling"]
        self.assertLess(
            npf.DEFAULT_KNOWN_FALSE_POSITIVE_BUDGET + 0.0466, ceiling
        )

    def test_auroc_matches_the_shipped_evaluator(self):
        from train import auroc

        positive = np.asarray([3.0, 2.0, 5.0])
        negative = np.asarray([1.0, 0.0])
        self.assertEqual(
            npf.mann_whitney_auroc(positive, negative), auroc(positive, negative)
        )


# ---------------------------------------------------------------------------
# invariance inherited from the features
# ---------------------------------------------------------------------------


class InheritedInvarianceTest(unittest.TestCase):
    """The detector must not reintroduce the scale dependence v3 removed.

    The features are exactly invariant under a power-of-two gain and invariant
    to rounding otherwise (``test_pose_degeneracy``).  A linear score over them
    inherits both properties, and the gate decision, being a comparison of that
    score, inherits them too.  These use real captures rather than fixtures,
    because the claim is about the feature extractor as much as about the model.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = _fitted()
        cls.capture = _multitone(2048)
        cls.baseline = pdg.pose_degeneracy_features(cls.capture)[None, :]

    def _features(self, capture):
        return pdg.pose_degeneracy_features(capture)[None, :]

    def test_a_power_of_two_gain_leaves_the_probability_bit_identical(self):
        for gain in (2.0 ** -20, 2.0 ** -3, 2.0 ** 3, 2.0 ** 20):
            with self.subTest(gain=gain):
                moved = self._features(gain * self.capture)
                self.assertTrue(
                    np.array_equal(
                        self.model.predict_proba(moved),
                        self.model.predict_proba(self.baseline),
                    )
                )
                self.assertTrue(
                    np.array_equal(
                        self.model.decide(moved), self.model.decide(self.baseline)
                    )
                )

    def test_a_global_phase_rotation_does_not_move_the_score(self):
        for phase in (0.731, -2.4, np.pi):
            with self.subTest(phase=phase):
                moved = self._features(np.exp(1j * phase) * self.capture)
                np.testing.assert_allclose(
                    self.model.score(moved),
                    self.model.score(self.baseline),
                    rtol=0.0,
                    atol=1e-7,
                )
                self.assertTrue(
                    np.array_equal(
                        self.model.decide(moved), self.model.decide(self.baseline)
                    )
                )

    def test_an_arbitrary_gain_and_phase_together_do_not_move_the_score(self):
        moved = self._features(5.5e3 * np.exp(1j * 0.9) * self.capture)
        np.testing.assert_allclose(
            self.model.score(moved),
            self.model.score(self.baseline),
            rtol=0.0,
            atol=1e-7,
        )

    def test_the_extractor_still_declares_the_invariance_this_relies_on(self):
        metadata = pdg.pose_degeneracy_metadata()
        self.assertTrue(metadata["invariant_to_global_phase"])
        self.assertTrue(metadata["invariant_to_global_scale"])
        self.assertFalse(metadata["uses_absolute_scale"])


# ---------------------------------------------------------------------------
# degenerate input
# ---------------------------------------------------------------------------


class DegenerateInputTest(unittest.TestCase):
    def test_a_zero_variance_column_is_refused_rather_than_amplified(self):
        known = _known()
        noise = _noise()
        column = int(npf.feature_columns()[0])
        known[:, column] = 1.25
        noise[:, column] = 1.25
        with self.assertRaises(ValueError) as caught:
            npf.fit_prefilter(
                known,
                noise,
                capture_length=4096,
                known_population=npf.TRAIN_SPLIT,
                fitting_seed=npf.PROPOSED_FITTING_SEED,
            )
        self.assertIn("degenerate", str(caught.exception))
        self.assertIn(npf.PREFILTER_FEATURES[0], str(caught.exception))

    def test_non_finite_features_are_refused_at_fit_and_at_score(self):
        known = _known()
        known[3, int(npf.feature_columns()[2])] = np.nan
        with self.assertRaises(ValueError):
            npf.fit_prefilter(
                known,
                _noise(),
                capture_length=4096,
                known_population=npf.TRAIN_SPLIT,
                fitting_seed=npf.PROPOSED_FITTING_SEED,
            )
        model = _fitted()
        bad = _noise(rows=4, seed=45)
        bad[1, int(npf.feature_columns()[0])] = np.inf
        with self.assertRaises(ValueError):
            model.score(bad)

    def test_empty_and_misshaped_matrices_are_refused(self):
        model = _fitted()
        for bad in (
            np.empty((0, pdg.FEATURE_COUNT)),
            np.zeros(pdg.FEATURE_COUNT),
            np.zeros((3, 5)),
        ):
            with self.subTest(shape=np.asarray(bad).shape):
                with self.assertRaises(ValueError):
                    model.score(bad)

    def test_too_few_rows_to_fit_are_refused(self):
        with self.assertRaises(ValueError):
            npf.fit_prefilter(
                _known(rows=1),
                _noise(),
                capture_length=4096,
                known_population=npf.TRAIN_SPLIT,
                fitting_seed=npf.PROPOSED_FITTING_SEED,
            )
        with self.assertRaises(ValueError):
            npf.fit_prefilter(
                _known(rows=5),
                _noise(),
                capture_length=4096,
                known_population=npf.TRAIN_SPLIT,
                fitting_seed=npf.PROPOSED_FITTING_SEED,
            )

    def test_an_invalid_capture_length_is_refused(self):
        with self.assertRaises(ValueError):
            npf.fit_prefilter(
                _known(),
                _noise(),
                capture_length=0,
                known_population=npf.TRAIN_SPLIT,
                fitting_seed=npf.PROPOSED_FITTING_SEED,
            )

    def test_a_degenerate_capture_fails_in_the_extractor_not_silently_here(self):
        # An all-zero window admits no pose at all.  The prefilter must inherit
        # the estimator's refusal rather than invent a feature vector for it.
        with self.assertRaises(Exception):
            pdg.pose_degeneracy_features(np.zeros(2048, dtype=np.complex128))

    def test_identical_classes_still_fit_and_report_chance(self):
        # Not an error: if the classes are identical the detector should say so
        # through its AUROC, not crash.
        rows = _known(rows=60, seed=47)
        model = npf.fit_prefilter(
            rows,
            rows.copy(),
            capture_length=4096,
            known_population=npf.TRAIN_SPLIT,
            fitting_seed=npf.PROPOSED_FITTING_SEED,
        )
        model = npf.fit_threshold(
            model, rows, population=npf.ENROLLMENT_SPLIT, budget=0.02
        )
        report = npf.evaluate(
            model, rows, rows.copy(), known_population=npf.ENROLLMENT_SPLIT
        )
        self.assertAlmostEqual(report["noise_auroc"], 0.5, places=9)

    def test_a_prefilter_cannot_be_built_with_a_broken_scale(self):
        model = _fitted()
        with self.assertRaises(ValueError):
            npf.NoisePrefilter(
                feature_names=model.feature_names,
                capture_length=4096,
                mean=model.mean,
                scale=np.zeros_like(model.scale),
                coefficients=model.coefficients,
                intercept=0.0,
                threshold_score=0.0,
                fit_provenance={},
                threshold_provenance={},
            )
        with self.assertRaises(ValueError):
            npf.NoisePrefilter(
                feature_names=model.feature_names,
                capture_length=4096,
                mean=model.mean,
                scale=model.scale,
                coefficients=model.coefficients,
                intercept=float("inf"),
                threshold_score=0.0,
                fit_provenance={},
                threshold_provenance={},
            )


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


class CommandLineTest(unittest.TestCase):
    def test_the_fitting_seed_and_budget_have_no_defaults(self):
        parser = npf.build_parser()
        with io.StringIO() as sink, contextlib.redirect_stderr(sink):
            with self.assertRaises(SystemExit):
                parser.parse_args([])
            with self.assertRaises(SystemExit):
                parser.parse_args(["--fitting-seed", "20261001"])
        args = parser.parse_args(
            ["--fitting-seed", "20261001", "--known-false-positive-budget", "0.02"]
        )
        self.assertEqual(args.fitting_seed, 20261001)
        self.assertEqual(args.known_false_positive_budget, 0.02)
        self.assertEqual(tuple(args.features), npf.PREFILTER_FEATURES)


if __name__ == "__main__":
    unittest.main()
