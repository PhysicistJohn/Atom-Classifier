from __future__ import annotations

import copy
import hashlib
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

import assemble_v3_fusion as assemble  # noqa: E402
import evaluate_invariant_release_suite as release  # noqa: E402
import measure_v3_remaining_gates as subject  # noqa: E402
from invariant_fusion import CenteredInvariantFusion  # noqa: E402
from invariant_patch_cnn import (  # noqa: E402
    InvariantPatchCNN,
    InvariantPatchConfig,
)


CLASSES = ("am", "bluetooth", "cw")


def _item(index: int, class_name: str, *, impaired: bool) -> dict[str, object]:
    """A manifest row carrying exactly the fields the dev corpus carries."""
    return {
        "adcBits": None,
        "bandwidthHz": 2000 + index,
        "centerHz": 1_500_000_000 + index,
        "centreOffsetFrac": 0.1 + index * 1e-4,
        "cleanPower": 1,
        "clockErrorPpm": -4.7,
        "cls": class_name,
        "impaired": impaired,
        "multipathTaps": 3,
        "noiseFigureDb": 11.6,
        "profile": class_name,
        "rxPowerDbm": -61.4,
        "sampleRateHz": 1_000_000,
        "snrDb": 20 + (index % 7),
        "startSampleIndex": 0,
    }


def _population(per_class: int = 12) -> tuple[list[dict[str, object]], np.ndarray]:
    items: list[dict[str, object]] = []
    labels: list[int] = []
    index = 0
    for class_index, class_name in enumerate(CLASSES):
        for offset in range(per_class):
            items.append(
                _item(index, class_name, impaired=bool(offset % 2))
            )
            labels.append(class_index)
            index += 1
    return items, np.asarray(labels, dtype=np.int64)


class SplitSeedTests(unittest.TestCase):
    def test_unspent_release_seed_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "20260731"):
            subject.validate_split_seed(20260731)

    def test_consumed_sealed_seed_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, str(release.NOVELTY_SEED)):
            subject.validate_split_seed(release.NOVELTY_SEED)

    def test_development_seeds_are_accepted(self) -> None:
        for seed in (20260730, 20260732, 0):
            self.assertEqual(subject.validate_split_seed(seed), seed)

    def test_non_integer_and_negative_seeds_fail_closed(self) -> None:
        for value in ("20260730", 1.5, True, None):
            with self.assertRaises(ValueError):
                subject.validate_split_seed(value)
        with self.assertRaises(ValueError):
            subject.validate_split_seed(-1)

    def test_tolerance_must_be_positive_and_finite(self) -> None:
        self.assertEqual(subject.validate_tolerance(1e-3), 1e-3)
        for value in (0.0, -1.0, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                subject.validate_tolerance(value)


class RowIdentityTests(unittest.TestCase):
    def test_identity_fields_are_the_release_fields_minus_impairment_seed(
        self,
    ) -> None:
        items, _labels = _population()
        fields = subject.dev_row_identity_fields(items)
        self.assertEqual(
            fields,
            tuple(
                name
                for name in release.ROW_IDENTITY_FIELDS
                if name != "impairmentSeed"
            ),
        )
        self.assertNotIn("cleanPower", fields)

    def test_a_further_missing_identity_field_fails_closed(self) -> None:
        items, _labels = _population()
        stripped = [dict(item) for item in items]
        for item in stripped:
            del item["clockErrorPpm"]
        with self.assertRaisesRegex(ValueError, "unexpected release identity"):
            subject.dev_row_identity_fields(stripped)

    def test_rows_disagreeing_on_fields_fail_closed(self) -> None:
        items, _labels = _population()
        mixed = [dict(item) for item in items]
        del mixed[3]["adcBits"]
        with self.assertRaisesRegex(ValueError, "disagree"):
            subject.dev_row_identity_fields(mixed)

    def test_identity_json_is_canonical_and_field_ordered(self) -> None:
        items, _labels = _population()
        fields = subject.dev_row_identity_fields(items)
        encoded = subject.dev_row_identity(items[0], fields)
        self.assertEqual(json.loads(encoded).keys(), {*fields})
        # Canonical form: sorted keys, no whitespace. Same encoder settings the
        # release evaluator uses, so only the field set differs.
        self.assertNotIn(" ", encoded)
        self.assertEqual(
            list(json.loads(encoded)), sorted(fields)
        )


class FiveShotSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.items, self.labels = _population()
        self.fields = subject.dev_row_identity_fields(self.items)

    def _split(self, seed: int = 20260730):
        return subject.dev_five_shot_split(
            self.items,
            self.labels,
            CLASSES,
            seed,
            identity_fields=self.fields,
        )

    def test_exactly_five_support_rows_per_class(self) -> None:
        support, report = self._split()
        self.assertEqual(len(support), release.FIVE_SHOT_K * len(CLASSES))
        for class_name in CLASSES:
            self.assertEqual(
                len(report["support_positions_by_class"][class_name]),
                release.FIVE_SHOT_K,
            )
        counts = np.bincount(self.labels[support], minlength=len(CLASSES))
        np.testing.assert_array_equal(
            counts, np.full(len(CLASSES), release.FIVE_SHOT_K)
        )

    def test_support_is_the_lowest_digests_per_class(self) -> None:
        _support, report = self._split()
        for class_index, class_name in enumerate(CLASSES):
            digests = []
            for position in range(len(self.items)):
                if int(self.labels[position]) != class_index:
                    continue
                payload = (
                    release.SPLIT_SALT
                    + "\0"
                    + "20260730"
                    + "\0"
                    + subject.dev_row_identity(self.items[position], self.fields)
                ).encode("utf-8")
                digests.append(hashlib.sha256(payload).hexdigest())
            expected = sorted(digests)[: release.FIVE_SHOT_K]
            self.assertEqual(report["support_digest_by_class"][class_name], expected)

    def test_the_hash_payload_matches_the_release_layout(self) -> None:
        _support, report = self._split(seed=4242)
        self.assertEqual(report["salt"], release.SPLIT_SALT)
        self.assertEqual(report["version"], release.SPLIT_VERSION)
        self.assertEqual(report["k_per_class"], release.FIVE_SHOT_K)
        self.assertEqual(report["split_seed"], 4242)
        # Same rule, different seed, so the draw must move.
        baseline, _ = self._split()
        moved, _ = self._split(seed=4242)
        self.assertFalse(np.array_equal(baseline, moved))

    def test_the_draw_is_deterministic(self) -> None:
        first, _ = self._split()
        second, _ = self._split()
        np.testing.assert_array_equal(first, second)

    def test_row_order_does_not_change_which_rows_are_chosen(self) -> None:
        support, _report = self._split()
        chosen = {
            subject.dev_row_identity(self.items[int(p)], self.fields)
            for p in support
        }
        order = np.random.default_rng(3).permutation(len(self.items))
        shuffled_items = [self.items[int(i)] for i in order]
        shuffled_labels = self.labels[order]
        reordered, _ = subject.dev_five_shot_split(
            shuffled_items,
            shuffled_labels,
            CLASSES,
            20260730,
            identity_fields=self.fields,
        )
        self.assertEqual(
            chosen,
            {
                subject.dev_row_identity(shuffled_items[int(p)], self.fields)
                for p in reordered
            },
        )

    def test_a_class_without_more_than_k_rows_is_refused(self) -> None:
        items = [_item(i, "am", impaired=False) for i in range(5)]
        labels = np.zeros(5, dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "needs more than"):
            subject.dev_five_shot_split(
                items, labels, ("am",), 1, identity_fields=self.fields
            )

    def test_the_release_seed_cannot_be_bound_into_the_split(self) -> None:
        with self.assertRaises(ValueError):
            self._split(seed=20260731)


class CleanGateTests(unittest.TestCase):
    """The clean gate must be the evaluator's quantity, not a look-alike."""

    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        self.classes = list(CLASSES)
        self.prototypes = np.eye(3, dtype=np.float32)
        self.labels = np.asarray([0, 0, 1, 1, 2, 2, 0, 1, 2], dtype=np.int64)
        self.impaired = np.asarray(
            [False, True, False, True, False, True, True, False, True],
            dtype=np.bool_,
        )
        self.embeddings = (
            np.eye(3, dtype=np.float32)[self.labels]
            + 0.01 * rng.normal(size=(9, 3)).astype(np.float32)
        )

    def test_clean_accuracy_is_plain_accuracy_over_unimpaired_rows(self) -> None:
        report = subject.closed_clean_report(
            self.embeddings,
            self.labels,
            self.impaired,
            self.prototypes,
            self.classes,
        )
        prediction, _ = release._nearest(self.embeddings, self.prototypes)
        expected = float(
            np.mean(
                prediction[~self.impaired] == self.labels[~self.impaired]
            )
        )
        self.assertEqual(report["clean_accuracy"], expected)
        self.assertEqual(report["clean_subset"]["n"], int((~self.impaired).sum()))

    def test_clean_accuracy_matches_the_release_closed_report(self) -> None:
        mine = subject.closed_clean_report(
            self.embeddings,
            self.labels,
            self.impaired,
            self.prototypes,
            self.classes,
        )
        theirs, _prediction = release._closed_report(
            self.embeddings,
            self.labels,
            # Both SNR subsets must be non-empty and carry every class, or the
            # evaluator's own closed report refuses the call.
            np.asarray(
                [20.0, 10.0, 20.0, 10.0, 20.0, 10.0, 20.0, 10.0, 20.0]
            ),
            self.impaired,
            self.prototypes,
            self.classes,
        )
        for key in ("accuracy", "balanced_accuracy", "clean_accuracy"):
            self.assertEqual(mine[key], theirs[key])
        self.assertEqual(mine["clean_subset"], theirs["clean_subset"])

    def test_clean_balanced_accuracy_is_reported_beside_the_gate_value(
        self,
    ) -> None:
        report = subject.closed_clean_report(
            self.embeddings,
            self.labels,
            self.impaired,
            self.prototypes,
            self.classes,
        )
        self.assertIn("clean_balanced_accuracy", report)
        self.assertEqual(
            report["clean_balanced_accuracy"],
            report["clean_subset"]["balanced_accuracy_present_classes"],
        )

    def test_a_clean_subset_missing_a_class_fails_closed(self) -> None:
        impaired = self.impaired.copy()
        impaired[self.labels == 2] = True
        with self.assertRaisesRegex(ValueError, "omits classes"):
            subject.closed_clean_report(
                self.embeddings,
                self.labels,
                impaired,
                self.prototypes,
                self.classes,
            )

    def test_impaired_mask_shape_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "impaired mask"):
            subject.closed_clean_report(
                self.embeddings,
                self.labels,
                self.impaired[:-1],
                self.prototypes,
                self.classes,
            )


class FiveShotVariantTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(21)
        self.classes = list(CLASSES)
        self.support_labels = np.repeat(
            np.arange(3, dtype=np.int64), release.FIVE_SHOT_K
        )
        self.support = (
            np.eye(3, dtype=np.float32)[self.support_labels]
            + 0.02 * rng.normal(size=(15, 3)).astype(np.float32)
        )
        self.query_labels = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
        self.query_by_length = {
            length: (
                np.eye(3, dtype=np.float32)[self.query_labels]
                + 0.05 * rng.normal(size=(6, 3)).astype(np.float32)
            )
            for length in subject.DEV_LENGTHS
        }

    def test_prototypes_are_the_support_mean_at_the_matched_length(self) -> None:
        result = subject.five_shot_variant(
            self.support,
            self.support_labels,
            self.query_by_length,
            self.query_labels,
            self.classes,
        )
        expected_prototypes = np.stack(
            [
                self.support[self.support_labels == index].mean(axis=0)
                for index in range(3)
            ]
        ).astype(np.float32)
        for length, query in self.query_by_length.items():
            prediction, _ = release._nearest(query, expected_prototypes)
            row = result["per_length"][str(length)]
            self.assertEqual(
                row["accuracy"],
                float(np.mean(prediction == self.query_labels)),
            )

    def test_one_prototype_set_is_reused_at_every_length(self) -> None:
        # Feeding the same query embeddings at every length must give the same
        # report, which is only true if the prototypes never move.
        shared = self.query_by_length[min(subject.DEV_LENGTHS)]
        result = subject.five_shot_variant(
            self.support,
            self.support_labels,
            {length: shared for length in subject.DEV_LENGTHS},
            self.query_labels,
            self.classes,
        )
        rows = list(result["per_length"].values())
        for row in rows[1:]:
            self.assertEqual(row["balanced_accuracy"], rows[0]["balanced_accuracy"])

    def test_support_rows_are_never_scored(self) -> None:
        poisoned = {
            length: np.concatenate((value,))
            for length, value in self.query_by_length.items()
        }
        clean = subject.five_shot_variant(
            self.support,
            self.support_labels,
            poisoned,
            self.query_labels,
            self.classes,
        )
        # Corrupting the support embeddings changes the prototypes, but the
        # query count must be unchanged: support rows are not appended to query.
        for length in subject.DEV_LENGTHS:
            self.assertEqual(
                clean["per_length"][str(length)]["query_rows"],
                len(self.query_labels),
            )
            self.assertEqual(
                clean["per_length"][str(length)]["support_rows"],
                len(self.support_labels),
            )

    def test_worst_is_the_minimum_over_lengths(self) -> None:
        result = subject.five_shot_variant(
            self.support,
            self.support_labels,
            self.query_by_length,
            self.query_labels,
            self.classes,
        )
        self.assertEqual(
            result["worst_length_balanced_accuracy"],
            min(
                float(row["balanced_accuracy"])
                for row in result["per_length"].values()
            ),
        )

    def test_k_is_enforced_by_the_release_report(self) -> None:
        with self.assertRaisesRegex(ValueError, "not exactly"):
            subject.five_shot_variant(
                self.support[:-1],
                self.support_labels[:-1],
                self.query_by_length,
                self.query_labels,
                self.classes,
            )

    def test_mismatched_query_length_fails_closed(self) -> None:
        broken = dict(self.query_by_length)
        broken[8192] = broken[8192][:-1]
        with self.assertRaisesRegex(ValueError, "disagree with the label vector"):
            subject.five_shot_variant(
                self.support,
                self.support_labels,
                broken,
                self.query_labels,
                self.classes,
            )


class GateAssemblyTests(unittest.TestCase):
    @staticmethod
    def _closed(values: dict[int, float]) -> dict[str, dict[str, object]]:
        return {
            str(length): {"clean_accuracy": value}
            for length, value in values.items()
        }

    @staticmethod
    def _five(primary: float, mirror: float) -> dict[str, dict[str, object]]:
        return {
            "enrollment_support": {"worst_length_balanced_accuracy": primary},
            "selection_support": {"worst_length_balanced_accuracy": mirror},
        }

    def test_thresholds_come_from_the_release_gate_floors(self) -> None:
        gates = subject.assemble_gates(
            self._closed({4096: 0.9, 8192: 0.9, 16384: 0.9}),
            self._five(0.9, 0.9),
        )
        self.assertEqual(
            gates["closed_clean_n4096"]["threshold"],
            release.GATE_FLOORS["closed_clean"],
        )
        self.assertEqual(
            gates["closed_clean_worst_length"]["threshold"],
            release.GATE_FLOORS["closed_clean"],
        )
        self.assertEqual(
            gates["five_shot_worst_length_balanced_primary"]["threshold"],
            release.GATE_FLOORS["five_shot"],
        )
        self.assertEqual(release.GATE_FLOORS["closed_clean"], 0.85)
        self.assertEqual(release.GATE_FLOORS["five_shot"], 0.85)

    def test_n4096_clean_is_reported_separately_from_the_worst(self) -> None:
        gates = subject.assemble_gates(
            self._closed({4096: 0.8464, 8192: 0.90, 16384: 0.95}),
            self._five(0.9, 0.9),
        )
        self.assertEqual(gates["closed_clean_n4096"]["value"], 0.8464)
        self.assertFalse(gates["closed_clean_n4096"]["passes"])
        self.assertEqual(gates["closed_clean_worst_length"]["value"], 0.8464)

    def test_worst_clean_can_come_from_a_longer_length(self) -> None:
        gates = subject.assemble_gates(
            self._closed({4096: 0.90, 8192: 0.86, 16384: 0.84}),
            self._five(0.9, 0.9),
        )
        self.assertTrue(gates["closed_clean_n4096"]["passes"])
        self.assertEqual(gates["closed_clean_worst_length"]["value"], 0.84)
        self.assertFalse(gates["closed_clean_worst_length"]["passes"])

    def test_the_knife_edge_is_decided_the_way_the_evaluator_decides_it(
        self,
    ) -> None:
        # ``_gate`` uses ``value >= floor``: exactly 0.85 passes.
        gates = subject.assemble_gates(
            self._closed({4096: 0.85, 8192: 0.85, 16384: 0.85}),
            self._five(0.85, 0.85),
        )
        self.assertTrue(gates["closed_clean_n4096"]["passes"])
        self.assertTrue(
            gates["five_shot_worst_length_balanced_conservative"]["passes"]
        )

    def test_the_conservative_five_shot_gate_is_never_weaker(self) -> None:
        gates = subject.assemble_gates(
            self._closed({4096: 0.9, 8192: 0.9, 16384: 0.9}),
            self._five(0.87, 0.83),
        )
        self.assertTrue(
            gates["five_shot_worst_length_balanced_enrollment_support"]["passes"]
        )
        self.assertFalse(
            gates["five_shot_worst_length_balanced_selection_support"]["passes"]
        )
        self.assertEqual(
            gates["five_shot_worst_length_balanced_conservative"]["value"], 0.83
        )
        self.assertFalse(
            gates["five_shot_worst_length_balanced_conservative"]["passes"]
        )

    def test_both_five_shot_variants_are_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "variants are missing"):
            subject.assemble_gates(
                self._closed({4096: 0.9, 8192: 0.9, 16384: 0.9}),
                {"enrollment_support": {"worst_length_balanced_accuracy": 0.9}},
            )

    def test_an_empty_clean_report_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one length"):
            subject.assemble_gates({}, self._five(0.9, 0.9))


class ReproductionCheckTests(unittest.TestCase):
    @staticmethod
    def _metrics(sha: str, rows: list[dict[str, object]]) -> dict[str, object]:
        return {
            "population_causal_length": {
                "eligible_indices_sha256": sha,
                "rows": rows,
            }
        }

    def test_a_different_population_fails_closed(self) -> None:
        metrics = self._metrics(
            "a" * 64, [{"length": 4096, "accuracy": 0.7, "balanced_accuracy": 0.7}]
        )
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            subject.reproduction_check(
                metrics,
                {"4096": {"accuracy": 0.7, "balanced_accuracy": 0.7}},
                "b" * 64,
                tolerance=1e-3,
            )

    def test_drift_above_tolerance_fails_closed(self) -> None:
        metrics = self._metrics(
            "a" * 64,
            [{"length": 4096, "accuracy": 0.7745, "balanced_accuracy": 0.7678}],
        )
        with self.assertRaisesRegex(RuntimeError, "differs by"):
            subject.reproduction_check(
                metrics,
                {"4096": {"accuracy": 0.70, "balanced_accuracy": 0.7678}},
                "a" * 64,
                tolerance=1e-3,
            )

    def test_drift_inside_tolerance_is_recorded(self) -> None:
        metrics = self._metrics(
            "a" * 64,
            [{"length": 4096, "accuracy": 0.7745, "balanced_accuracy": 0.7678}],
        )
        checks = subject.reproduction_check(
            metrics,
            {"4096": {"accuracy": 0.7746, "balanced_accuracy": 0.7678}},
            "a" * 64,
            tolerance=1e-3,
        )
        self.assertTrue(checks["matches_artifact_population"])
        self.assertLess(checks["worst_absolute_delta"], 1e-3)

    def test_no_comparable_rows_fails_closed(self) -> None:
        metrics = self._metrics("a" * 64, [])
        with self.assertRaisesRegex(RuntimeError, "no comparable"):
            subject.reproduction_check(
                metrics, {"4096": {"accuracy": 0.7, "balanced_accuracy": 0.7}},
                "a" * 64,
                tolerance=1e-3,
            )


def _tiny_fusion_artifact(root: Path) -> tuple[Path, dict[str, object]]:
    """Write a minimal but structurally valid v3 fusion directory."""
    common = {
        "patch_length": 16,
        "patch_count": 2,
        "patch_dim": 8,
        "hidden": 12,
        "embed_dim": 4,
        "n_features": 3,
        "dropout": 0.0,
        "set_pool": "mean_std",
    }
    real = InvariantPatchCNN(
        InvariantPatchConfig(encoder="real", **common)
    ).eval()
    complex_branch = InvariantPatchCNN(
        InvariantPatchConfig(encoder="complex", **common)
    ).eval()
    real_center = np.arange(4, dtype=np.float32)
    complex_center = np.arange(4, dtype=np.float32)[::-1].copy()
    fusion = CenteredInvariantFusion(
        real,
        complex_branch,
        real_center,
        complex_center,
        alpha_real=assemble.ALPHA_REAL,
        alpha_complex=assemble.ALPHA_COMPLEX,
        weight_real=0.5,
        eps=assemble.FUSION_EPS,
    ).eval()

    directory = root / "v3_fusion_tiny"
    directory.mkdir(parents=True)
    torch.save(
        {key: value.detach().cpu() for key, value in fusion.state_dict().items()},
        directory / "fusion_state_dict.pt",
    )
    np.save(directory / "fusion_prototypes.npy", np.eye(3, 8, dtype=np.float32))
    np.save(directory / "real_center.npy", real_center)
    np.save(directory / "complex_center.npy", complex_center)
    np.save(directory / "feature_mean.npy", np.zeros(3, dtype=np.float32))
    np.save(directory / "feature_std.npy", np.ones(3, dtype=np.float32))

    def sha(name: str) -> str:
        digest = hashlib.sha256()
        digest.update((directory / name).read_bytes())
        return digest.hexdigest()

    import time_domain_invariant_patch_preprocess as td_preprocess

    metrics = {
        "status": "complete",
        "development_only": True,
        "release_evidence": False,
        "sealed_release_data_used": 0,
        "consumed_test_rows_used": 0,
        "encoder": "fusion",
        "seed": 20260730,
        "parameter_count": 1,
        "frontend": {
            "version": td_preprocess.PREPROCESS_VERSION,
            "uses_frequency_transform": False,
        },
        "architecture": fusion.config(),
        "closed_selection": {
            "balanced_accuracy": 0.8675,
            "clean_accuracy": 0.8921,
        },
        "data_audit": {"consumed_test_rows_exposed": 0},
        "source_index_contract": {"contract": {"config": {"target_frac": 0.5}}},
        "population_causal_length": {
            "eligible_indices_sha256": "c" * 64,
            "rows": [],
        },
        "artifacts": {
            "state_dict": "fusion_state_dict.pt",
            "state_dict_sha256": sha("fusion_state_dict.pt"),
            "prototypes": "fusion_prototypes.npy",
            "prototypes_sha256": sha("fusion_prototypes.npy"),
            "real_center": "real_center.npy",
            "real_center_sha256": sha("real_center.npy"),
            "complex_center": "complex_center.npy",
            "complex_center_sha256": sha("complex_center.npy"),
            "feature_mean": "feature_mean.npy",
            "feature_mean_sha256": sha("feature_mean.npy"),
            "feature_std": "feature_std.npy",
            "feature_std_sha256": sha("feature_std.npy"),
        },
    }
    (directory / "dev_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    return directory, metrics


class ArtifactLoadingTests(unittest.TestCase):
    def test_a_valid_artifact_rebuilds_its_fusion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory, metrics = _tiny_fusion_artifact(Path(tmp))
            artifact = subject.load_fusion_artifact(directory)
            self.assertIsInstance(artifact["fusion"], CenteredInvariantFusion)
            np.testing.assert_array_equal(
                artifact["fusion"].real_center.detach().cpu().numpy(),
                np.load(directory / "real_center.npy"),
            )
            self.assertEqual(
                artifact["fusion"].config()["weight_real"],
                metrics["architecture"]["weight_real"],
            )
            self.assertEqual(artifact["prototypes"].shape, (3, 8))

    def test_frozen_assets_are_loaded_verbatim_and_never_refit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory, _metrics = _tiny_fusion_artifact(Path(tmp))
            artifact = subject.load_fusion_artifact(directory)
            frozen = subject.frozen_inference_assets(artifact)
            np.testing.assert_array_equal(
                frozen["fmean"], np.load(directory / "feature_mean.npy")
            )
            np.testing.assert_array_equal(
                frozen["fstd"], np.load(directory / "feature_std.npy")
            )
            self.assertEqual(
                frozen["cache_contract"]["config"]["target_frac"], 0.5
            )

    def test_a_tampered_asset_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory, _metrics = _tiny_fusion_artifact(Path(tmp))
            np.save(
                directory / "fusion_prototypes.npy",
                np.zeros((3, 8), dtype=np.float32),
            )
            with self.assertRaisesRegex(RuntimeError, "recorded SHA-256"):
                subject.load_fusion_artifact(directory)

    def test_release_or_consumed_provenance_is_refused(self) -> None:
        for key, value in (
            ("release_evidence", True),
            ("sealed_release_data_used", 1),
            ("consumed_test_rows_used", 1),
            ("development_only", False),
        ):
            with tempfile.TemporaryDirectory() as tmp:
                directory, metrics = _tiny_fusion_artifact(Path(tmp))
                broken = copy.deepcopy(metrics)
                broken[key] = value
                (directory / "dev_metrics.json").write_text(
                    json.dumps(broken), encoding="utf-8"
                )
                with self.assertRaises(RuntimeError):
                    subject.load_fusion_artifact(directory)

    def test_a_frequency_transform_frontend_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory, metrics = _tiny_fusion_artifact(Path(tmp))
            broken = copy.deepcopy(metrics)
            broken["frontend"]["uses_frequency_transform"] = True
            (directory / "dev_metrics.json").write_text(
                json.dumps(broken), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "FFT-free"):
                subject.load_fusion_artifact(directory)

    def test_a_single_branch_artifact_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory, metrics = _tiny_fusion_artifact(Path(tmp))
            broken = copy.deepcopy(metrics)
            broken["encoder"] = "real"
            (directory / "dev_metrics.json").write_text(
                json.dumps(broken), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "expected 'fusion'"):
                subject.load_fusion_artifact(directory)

    def test_sealed_and_release_directories_are_refused(self) -> None:
        for candidate in (
            "/tmp/training/artifacts/releases/suite",
            "/tmp/sealed_v2_suite",
        ):
            with self.assertRaisesRegex(ValueError, "refuses release or sealed"):
                subject.load_fusion_artifact(Path(candidate))


class DriverTests(unittest.TestCase):
    """End-to-end report assembly with the corpus and encoder stubbed out.

    The heavy pieces -- corpus IO, preprocessing, inference -- are replaced by
    deterministic stand-ins so the report wiring, gate assembly, provenance
    block, and overwrite refusal are exercised without a real run.
    """

    PER_CLASS = 8

    def _population(self, sha: str) -> dict[str, object]:
        labels = np.repeat(np.arange(3, dtype=np.int64), self.PER_CLASS)
        impaired = np.tile(
            np.asarray([False, True, False, True], dtype=np.bool_),
            len(labels) // 4,
        )
        items = [
            _item(index, CLASSES[int(labels[index])], impaired=bool(impaired[index]))
            for index in range(len(labels))
        ]
        return {
            "indices": np.arange(len(labels), dtype=np.int64),
            # Each stand-in capture carries its own class, so the fake encoder
            # stays correct no matter which rows the digest split picks.
            "captures": [
                np.full(16384, float(label), dtype=np.complex128)
                for label in labels
            ],
            "labels": labels,
            "impaired": impaired,
            "snr_db": np.full(len(labels), 20.0),
            "items": items,
            "included_by_class": {name: self.PER_CLASS for name in CLASSES},
            "excluded_all_zero_n4096_prefix_by_class": {
                name: 0 for name in CLASSES
            },
            "indices_sha256": sha,
        }

    def _run(self, tmp: Path, **overrides: object) -> dict[str, object]:
        from unittest import mock

        directory, metrics = _tiny_fusion_artifact(tmp)
        metrics = copy.deepcopy(metrics)
        metrics["population_causal_length"] = {
            "eligible_indices_sha256": "c" * 64,
            "rows": [
                {"length": length, "accuracy": 1.0, "balanced_accuracy": 1.0}
                for length in subject.DEV_LENGTHS
            ],
        }
        (directory / "dev_metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
        )

        selection = self._population("c" * 64)
        enrollment = self._population("d" * 64)
        populations = iter((selection, enrollment))

        def fake_eligible(manifest, indices, classes, raw):
            return next(populations)

        def fake_embed(embedder, captures, length, frozen, device):
            # Perfectly separable embeddings, so every recorded number is 1.0
            # and the reproduction check has an exact target to hit.
            labels = np.asarray(
                [int(round(float(capture[0].real))) for capture in captures],
                dtype=np.int64,
            )
            return np.eye(3, 8, dtype=np.float32)[labels]

        source = {
            "va_idx": np.arange(len(selection["labels"]), dtype=np.int64),
            "en_idx": np.arange(len(enrollment["labels"]), dtype=np.int64),
            "classes": list(CLASSES),
            "cache_contract": metrics["source_index_contract"]["contract"],
            "data_audit": {"consumed_test_rows_exposed": 0},
        }

        args = subject.build_parser().parse_args(
            ["--fusion-dir", str(directory), "--device", "cpu"]
        )
        for key, value in overrides.items():
            setattr(args, key, value)

        with mock.patch.object(
            subject.invariant_data, "load", return_value=source
        ), mock.patch.object(
            subject.dev, "_load_manifest", return_value={"items": selection["items"]}
        ), mock.patch.object(
            subject.dev, "_raw_memmap", return_value=None
        ), mock.patch.object(
            subject, "eligible_population", side_effect=fake_eligible
        ), mock.patch.object(
            subject, "embed_at_length", side_effect=fake_embed
        ):
            return subject.run(args), directory

    def test_report_is_written_with_the_expected_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, directory = self._run(Path(tmp))
            written = json.loads(
                (directory / subject.REPORT_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(written["status"], "complete")
            self.assertEqual(written["measurement"], subject.MEASUREMENT_VERSION)
            self.assertIs(written["development_only"], True)
            self.assertIs(written["release_evidence"], False)
            self.assertEqual(written["sealed_release_data_used"], 0)
            self.assertEqual(written["consumed_test_rows_used"], 0)
            self.assertIs(written["retraining_performed"], False)
            self.assertIs(written["recalibration_performed"], False)
            self.assertEqual(written["population"]["training"]["rows_scored"], 0)
            self.assertIn("source_sha256", written)
            self.assertIn("data_audit", written)
            self.assertIn("source_index_contract", written)
            self.assertEqual(result["device"], "cpu")

    def test_every_measured_length_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, _directory = self._run(Path(tmp))
            self.assertEqual(
                sorted(int(key) for key in result["closed_clean_per_length"]),
                sorted(subject.DEV_LENGTHS),
            )
            for row in result["closed_clean_per_length"].values():
                self.assertIn("clean_accuracy", row)
                self.assertIn("clean_balanced_accuracy", row)

    def test_both_five_shot_variants_and_all_gates_are_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, _directory = self._run(Path(tmp))
            self.assertEqual(
                set(result["five_shot_per_variant"]), set(subject.FIVE_SHOT_VARIANTS)
            )
            self.assertEqual(
                set(result["gates"]),
                {
                    "closed_clean_n4096",
                    "closed_clean_worst_length",
                    "five_shot_worst_length_balanced_enrollment_support",
                    "five_shot_worst_length_balanced_selection_support",
                    "five_shot_worst_length_balanced_conservative",
                    "five_shot_worst_length_balanced_primary",
                },
            )
            self.assertTrue(result["all_measured_gates_pass"])

    def test_the_deviation_list_ships_inside_the_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, _directory = self._run(Path(tmp))
            deviations = result["protocol"]["deviations_from_sealed_protocol"]
            self.assertTrue(any("32768" in text for text in deviations))
            self.assertTrue(any("all-zero" in text for text in deviations))
            self.assertTrue(any("impairmentSeed" in text for text in deviations))
            self.assertEqual(
                result["protocol"]["unmeasurable_release_capture_lengths"], [32768]
            )

    def test_an_existing_report_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _result, directory = self._run(Path(tmp))
            # The refusal happens before any corpus is loaded, so an unmocked
            # re-run is enough to prove it.
            args = subject.build_parser().parse_args(
                ["--fusion-dir", str(directory), "--device", "cpu"]
            )
            with self.assertRaisesRegex(FileExistsError, subject.REPORT_NAME):
                subject.run(args)

    def test_the_release_seed_is_refused_at_the_command_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "20260731"):
                self._run(Path(tmp), split_seed=20260731)


class DocumentedDeviationTests(unittest.TestCase):
    """The caveats are load-bearing; keep them from silently disappearing."""

    def test_n32768_is_named_as_unmeasurable(self) -> None:
        self.assertEqual(subject.UNMEASURABLE_RELEASE_LENGTHS, (32768,))
        self.assertEqual(
            set(subject.DEV_LENGTHS) | set(subject.UNMEASURABLE_RELEASE_LENGTHS),
            set(release.REQUIRED_CAPTURE_LENGTHS),
        )

    def test_the_matched_length_is_measurable(self) -> None:
        self.assertIn(release.MATCHED_CAPTURE_LENGTH, subject.DEV_LENGTHS)

    def test_the_docstring_states_each_deviation(self) -> None:
        text = subject.__doc__ or ""
        for phrase in (
            "32768",
            "all-zero",
            "class balance",
            "impairmentSeed",
            "scored, never fit",
            "not release evidence",
        ):
            self.assertIn(phrase.lower(), text.lower(), phrase)

    def test_the_primary_variant_does_not_fit_on_selection(self) -> None:
        self.assertEqual(subject.PRIMARY_FIVE_SHOT_VARIANT, "enrollment_support")


if __name__ == "__main__":
    unittest.main()
