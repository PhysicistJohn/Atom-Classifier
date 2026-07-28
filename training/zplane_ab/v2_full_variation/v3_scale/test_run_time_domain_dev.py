from __future__ import annotations

import argparse
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

import run_time_domain_dev as subject


def _item(cls: str, center: float, bandwidth: float) -> dict[str, object]:
    return {
        "cls": cls,
        "centreOffsetFrac": center,
        "bandwidthHz": bandwidth,
        "sampleRateHz": 1.0,
        "impaired": False,
        "snrDb": 20.0,
    }


class TimeDomainDevelopmentAuditTests(unittest.TestCase):
    def test_run_configuration_records_all_model_affecting_arguments(self) -> None:
        args = subject.build_parser().parse_args(
            [
                "--output-dir",
                "/tmp/ignored-destination",
                "--device",
                "mps",
                "--seed",
                "20260730",
                "--encoder",
                "complex",
                "--episodes",
                "8000",
                "--eval-every",
                "1000",
                "--dropout",
                "0.35",
                "--weight-decay",
                "0.0005",
                "--multilength-train",
            ]
        )
        result = subject._run_configuration(
            args,
            resolved_device=torch.device("mps"),
        )
        self.assertEqual(
            result["schema"],
            "time-domain-v3-training-configuration-v1",
        )
        self.assertNotIn("output_dir", result["arguments"])
        self.assertEqual(result["arguments"]["episodes"], 8000)
        self.assertEqual(result["arguments"]["eval_every"], 1000)
        self.assertEqual(result["arguments"]["dropout"], 0.35)
        self.assertEqual(result["arguments"]["weight_decay"], 0.0005)
        self.assertTrue(result["arguments"]["multilength_train"])
        self.assertEqual(
            result["optimizer"]["network_weight_decay"],
            0.0005,
        )
        self.assertEqual(result["randomness"]["model_seed"], 20260730)
        self.assertEqual(result["software"]["requested_device"], "mps")
        self.assertEqual(result["software"]["resolved_device"], "mps")

    def test_nonzero_prefix_mask_is_exact_and_fails_short_rows(self) -> None:
        captures = [
            np.zeros(8, dtype=np.complex64),
            np.asarray([0, 0, 0, 1, 0, 0, 0, 0], dtype=np.complex64),
            np.asarray([0, 0, 0, 0, 1, 0, 0, 0], dtype=np.complex64),
        ]
        self.assertEqual(
            subject._nonzero_prefix_mask(captures, 4).tolist(),
            [False, True, False],
        )
        with self.assertRaisesRegex(ValueError, "shorter"):
            subject._nonzero_prefix_mask([np.zeros(3, dtype=np.complex64)], 4)

    def test_multiview_sampler_uses_distinct_bases_and_different_pair(self) -> None:
        views = np.arange(36, dtype=np.int64).reshape(12, 3)
        by_class = (np.arange(0, 6), np.arange(6, 12))
        sample = subject._sample_multiview_episode(
            by_class,
            views,
            np.random.default_rng(123),
            per_class=4,
            include_secondary=True,
        )
        self.assertEqual(len(np.unique(sample["base"][:4])), 4)
        self.assertEqual(len(np.unique(sample["base"][4:])), 4)
        self.assertTrue(np.all(sample["view"] != sample["secondary_view"]))
        self.assertTrue(np.all(sample["primary"] != sample["secondary"]))

    def test_scale_eligibility_is_common_and_metadata_only(self) -> None:
        classes = ("a", "b")
        manifest = {
            "sampleCount": 16384,
            "items": [
                _item("a", 0.01, 0.1),
                _item("a", 0.24, 0.1),
                _item("b", -0.02, 0.08),
            ],
        }
        with mock.patch.object(subject, "SCALE_MIN_PER_CLASS", 1):
            eligible, report = subject._scale_eligible(
                manifest, np.asarray([0, 1, 2]), classes
            )
        self.assertEqual(eligible.tolist(), [0, 2])
        self.assertEqual(report["eligible_by_class"], {"a": 1, "b": 1})
        self.assertEqual(report["excluded_by_class"], {"a": 1, "b": 0})

    def test_paired_report_has_exact_identity_metrics(self) -> None:
        embeddings = np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32
        )
        prediction = np.asarray([0, 1, 0], dtype=np.int64)
        labels = np.asarray([0, 1, 0], dtype=np.int64)
        report = subject._paired_report(
            embeddings,
            embeddings.copy(),
            prediction,
            prediction.copy(),
            labels,
            ("a", "b"),
        )
        self.assertEqual(report["prediction_agreement"], 1.0)
        self.assertEqual(report["embedding_cosine_mean"], 1.0)
        self.assertEqual(report["embedding_cosine_p05"], 1.0)

    def test_run_refuses_release_or_sealed_output_before_loading(self) -> None:
        for path in ("/tmp/releases/v3", "/tmp/Sealed-v3"):
            with self.subTest(path=path), mock.patch.object(
                subject.invariant_data,
                "load",
                side_effect=AssertionError("must reject path before data load"),
            ):
                args = argparse.Namespace(output_dir=path)
                with self.assertRaisesRegex(ValueError, "release or sealed"):
                    subject.run(args)

    def test_high_snr_threshold_matches_frozen_protocol(self) -> None:
        self.assertEqual(subject.HIGH_SNR_DB, 18.0)


class NoveltyProxyTests(unittest.TestCase):
    def test_effective_rank_matches_known_spectra(self) -> None:
        two_dimensional = np.asarray(
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
        )
        self.assertAlmostEqual(
            subject._effective_rank(two_dimensional), 2.0, places=12
        )
        rank_one = np.asarray([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        self.assertAlmostEqual(
            subject._effective_rank(rank_one), 1.0, places=12
        )

    def test_effective_rank_of_total_collapse_is_zero(self) -> None:
        identical = np.ones((5, 3), dtype=np.float64)
        self.assertEqual(subject._effective_rank(identical), 0.0)

    def test_effective_rank_rejects_bad_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "rows>=2"):
            subject._effective_rank(np.ones((1, 4)))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            subject._effective_rank(
                np.asarray([[0.0, np.nan], [1.0, 2.0]])
            )

    def test_novelty_proxies_are_seed_free_and_exact_at_prototypes(self) -> None:
        selection = np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float64
        )
        enrollment = np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
            dtype=np.float64,
        )
        prototypes = selection.copy()
        state_before = np.random.get_state()
        _prediction, squared = subject.nearest(selection, prototypes)
        proxies = subject._novelty_proxies(selection, enrollment, squared)
        state_after = np.random.get_state()
        self.assertEqual(str(state_before), str(state_after))
        self.assertEqual(
            set(proxies),
            {
                "selection_embedding_effective_rank",
                "enrollment_embedding_effective_rank",
                "selection_nearest_prototype_distance_mean",
                "selection_nearest_prototype_distance_p05",
            },
        )
        self.assertEqual(
            proxies["selection_nearest_prototype_distance_mean"], 0.0
        )
        self.assertEqual(
            proxies["selection_nearest_prototype_distance_p05"], 0.0
        )
        self.assertAlmostEqual(
            proxies["enrollment_embedding_effective_rank"], 2.0, places=12
        )
        for value in proxies.values():
            self.assertIsInstance(value, float)
            self.assertTrue(np.isfinite(value))

    def test_novelty_proxies_reject_non_finite_distances(self) -> None:
        embeddings = np.eye(3)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            subject._novelty_proxies(
                embeddings, embeddings, np.asarray([[np.inf, 1.0]] * 3)
            )

    def test_declared_constants_are_the_predeclared_ones(self) -> None:
        self.assertEqual(
            subject.SELECTION_POLICIES, ("closed", "closed-with-floor")
        )
        self.assertEqual(subject.NOVELTY_FLOOR_FRACTION, 0.25)
        self.assertEqual(
            subject.NOVELTY_PROXY_KEY, "selection_embedding_effective_rank"
        )


class CheckpointSelectorTests(unittest.TestCase):
    @staticmethod
    def _proxies(value: float) -> dict[str, float]:
        return {subject.NOVELTY_PROXY_KEY: float(value)}

    def test_closed_policy_matches_historical_tuple_tiebreak(self) -> None:
        selector = subject._CheckpointSelector("closed")
        improved, _ = selector.observe(
            250, accuracy=0.80, balanced=0.80, proxies=self._proxies(5.0)
        )
        self.assertTrue(improved)
        improved, _ = selector.observe(
            500, accuracy=0.80, balanced=0.80, proxies=self._proxies(5.0)
        )
        self.assertFalse(improved)  # equal tuple must not replace the best
        improved, _ = selector.observe(
            750, accuracy=0.90, balanced=0.80, proxies=self._proxies(5.0)
        )
        self.assertTrue(improved)  # accuracy breaks the balanced tie
        self.assertEqual(selector.selected_best, (0.80, 0.90))
        self.assertEqual(selector.selected_best_episode, 750)

    def test_floor_and_closed_choose_different_checkpoints(self) -> None:
        # Synthetic collapse: closed-set keeps improving while the novelty
        # proxy collapses at the last checkpoint.  Closed must take the last
        # checkpoint; the floor policy must refuse it and keep episode 500.
        checkpoints = (
            (250, 0.80, 30.0),
            (500, 0.85, 28.0),
            (750, 0.90, 10.0),
        )
        closed = subject._CheckpointSelector("closed", floor_fraction=0.25)
        floored = subject._CheckpointSelector(
            "closed-with-floor", floor_fraction=0.25
        )
        eligibility = []
        for episode, balanced, proxy in checkpoints:
            closed.observe(
                episode,
                accuracy=balanced,
                balanced=balanced,
                proxies=self._proxies(proxy),
            )
            _improved, eligible = floored.observe(
                episode,
                accuracy=balanced,
                balanced=balanced,
                proxies=self._proxies(proxy),
            )
            eligibility.append(eligible)
        self.assertEqual(eligibility, [True, True, False])
        self.assertEqual(closed.selected_best_episode, 750)
        self.assertEqual(floored.selected_best_episode, 500)
        self.assertEqual(closed.selected_best[0], 0.90)
        self.assertEqual(floored.selected_best[0], 0.85)
        # Both trackers run in both selectors, so either artifact shows the
        # divergence machine-readably.
        for selector in (closed, floored):
            summary = selector.summary()
            self.assertEqual(summary["closed_best_episode"], 750)
            self.assertEqual(summary["floor_best_episode"], 500)
            self.assertEqual(summary["novelty_floor_fraction"], 0.25)

    def test_floor_uses_prefix_max_semantics(self) -> None:
        selector = subject._CheckpointSelector(
            "closed-with-floor", floor_fraction=0.25
        )
        results = [
            selector.observe(
                episode,
                accuracy=0.5,
                balanced=0.5,
                proxies=self._proxies(proxy),
            )[1]
            for episode, proxy in ((250, 10.0), (500, 30.0), (750, 21.0))
        ]
        # 10 is eligible against its own running max; 21 < 0.75 * 30 is not.
        self.assertEqual(results, [True, True, False])

    def test_first_checkpoint_is_always_floor_eligible(self) -> None:
        selector = subject._CheckpointSelector(
            "closed-with-floor", floor_fraction=0.25
        )
        improved, eligible = selector.observe(
            250, accuracy=0.1, balanced=0.1, proxies=self._proxies(1e-9)
        )
        self.assertTrue(improved)
        self.assertTrue(eligible)

    def test_selector_validates_policy_fraction_and_proxy(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown selection policy"):
            subject._CheckpointSelector("best-novelty")
        for fraction in (0.0, 1.0, -0.1, float("nan")):
            with self.subTest(fraction=fraction):
                with self.assertRaisesRegex(ValueError, "floor_fraction"):
                    subject._CheckpointSelector(
                        "closed-with-floor", floor_fraction=fraction
                    )
        selector = subject._CheckpointSelector("closed-with-floor")
        with self.assertRaisesRegex(ValueError, "finite"):
            selector.observe(
                250,
                accuracy=0.5,
                balanced=0.5,
                proxies=self._proxies(float("nan")),
            )


class SelectionPolicyRunGateTests(unittest.TestCase):
    @staticmethod
    def _args(**overrides: object) -> argparse.Namespace:
        values: dict[str, object] = {
            "selection_policy": "closed",
            "novelty_floor_fraction": subject.NOVELTY_FLOOR_FRACTION,
            "multilength_train": True,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_parser_defaults_preserve_historical_behaviour(self) -> None:
        args = subject.build_parser().parse_args(["--output-dir", "x"])
        self.assertEqual(args.selection_policy, "closed")
        self.assertEqual(
            args.novelty_floor_fraction, subject.NOVELTY_FLOOR_FRACTION
        )

    def test_run_rejects_floor_policy_without_multilength(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(
                output_dir=str(Path(tmp) / "out"),
                selection_policy="closed-with-floor",
                multilength_train=False,
            )
            with mock.patch.object(
                subject.invariant_data,
                "load",
                side_effect=AssertionError("must reject before data load"),
            ):
                with self.assertRaisesRegex(ValueError, "multilength"):
                    subject.run(args)

    def test_run_rejects_out_of_range_floor_fraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(
                output_dir=str(Path(tmp) / "out"),
                novelty_floor_fraction=1.0,
            )
            with mock.patch.object(
                subject.invariant_data,
                "load",
                side_effect=AssertionError("must reject before data load"),
            ):
                with self.assertRaisesRegex(ValueError, "floor-fraction"):
                    subject.run(args)


class MultiviewTrainerSelectionTests(unittest.TestCase):
    """Tiny CPU training runs proving the loop wiring, not model quality."""

    N_CLASSES = 2
    BASES_PER_CLASS = 12
    VIEWS = 2

    def _data(self, rng: np.random.Generator) -> dict[str, object]:
        bases = self.N_CLASSES * self.BASES_PER_CLASS
        rows = bases * self.VIEWS
        config = subject.InvariantPatchConfig(
            patch_length=16,
            patch_count=2,
            encoder="real",
            patch_dim=8,
            hidden=16,
            set_pool="mean",
            dropout=0.0,
        )
        packed_length = config.patch_length * config.patch_count
        eval_rows = 12
        return {
            "n_classes": self.N_CLASSES,
            "multiview_view_positions": np.arange(
                rows, dtype=np.int64
            ).reshape(bases, self.VIEWS),
            "multiview_base_indices_by_class": [
                np.arange(0, self.BASES_PER_CLASS, dtype=np.int64),
                np.arange(
                    self.BASES_PER_CLASS, bases, dtype=np.int64
                ),
            ],
            "xtr": rng.standard_normal(
                (rows, 2, packed_length)
            ).astype(np.float32),
            "ftr": rng.standard_normal(
                (rows, config.n_features)
            ).astype(np.float32),
            "xen": rng.standard_normal(
                (eval_rows, 2, packed_length)
            ).astype(np.float32),
            "fen": rng.standard_normal(
                (eval_rows, config.n_features)
            ).astype(np.float32),
            "yen": np.repeat(
                np.arange(self.N_CLASSES), eval_rows // self.N_CLASSES
            ),
            "xva": rng.standard_normal(
                (eval_rows, 2, packed_length)
            ).astype(np.float32),
            "fva": rng.standard_normal(
                (eval_rows, config.n_features)
            ).astype(np.float32),
            "yva": np.repeat(
                np.arange(self.N_CLASSES), eval_rows // self.N_CLASSES
            ),
            "_config": config,
        }

    def _train(self, policy: str) -> dict[str, object]:
        data = self._data(np.random.default_rng(7))
        torch.manual_seed(0)
        net = subject.InvariantPatchCNN(data.pop("_config"))
        _net, history = subject._train_model_multiview(
            net,
            data,
            torch.device("cpu"),
            episodes=2,
            eval_every=1,
            seed=11,
            lr=1e-3,
            weight_decay=0.0,
            warmup_frac=0.0,
            label_smoothing=0.0,
            phase_augmentation=False,
            consistency_weight=0.0,
            selection_policy=policy,
        )
        return history

    def test_history_records_proxies_and_policy_additively(self) -> None:
        history = self._train("closed")
        self.assertEqual(history["selection_policy"], "closed")
        self.assertEqual(
            history["novelty_floor_fraction"],
            subject.NOVELTY_FLOOR_FRACTION,
        )
        self.assertEqual(
            history["novelty_proxy_key"], subject.NOVELTY_PROXY_KEY
        )
        self.assertEqual(len(history["selection"]), 2)
        for entry in history["selection"]:
            self.assertIn("novelty_proxies", entry)
            self.assertIsInstance(entry["novelty_floor_eligible"], bool)
            for value in entry["novelty_proxies"].values():
                self.assertTrue(np.isfinite(value))
        # The historical criterion decides the historical keys exactly.
        best = max(
            (entry["balanced_accuracy"], entry["accuracy"])
            for entry in history["selection"]
        )
        self.assertEqual(
            (
                history["best_selection_balanced_accuracy"],
                history["best_selection_accuracy"],
            ),
            best,
        )
        self.assertEqual(
            history["selected_checkpoint_episode"],
            history["closed_best_episode"],
        )

    def test_floor_policy_consumes_the_identical_rng_stream(self) -> None:
        closed = self._train("closed")
        floored = self._train("closed-with-floor")
        self.assertEqual(closed["loss"], floored["loss"])
        self.assertEqual(closed["loss_ce"], floored["loss_ce"])
        self.assertEqual(len(closed["selection"]), len(floored["selection"]))
        for closed_entry, floored_entry in zip(
            closed["selection"], floored["selection"]
        ):
            self.assertEqual(
                closed_entry["balanced_accuracy"],
                floored_entry["balanced_accuracy"],
            )
            self.assertEqual(
                closed_entry["novelty_proxies"],
                floored_entry["novelty_proxies"],
            )
        self.assertEqual(floored["selection_policy"], "closed-with-floor")
        self.assertEqual(
            floored["selected_checkpoint_episode"],
            floored["floor_best_episode"],
        )
        # Both histories expose both trackers, so the trade stays visible in
        # every artifact regardless of the active policy.
        self.assertEqual(
            closed["closed_best_episode"], floored["closed_best_episode"]
        )


if __name__ == "__main__":
    unittest.main()
