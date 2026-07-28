from __future__ import annotations

import copy
import hashlib
import inspect
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

# The subject module inserts the training/zplane_ab/v2_full_variation paths
# into sys.path on import, so it must be imported first.
import measure_v2_bundle_dev_gates as subject  # noqa: E402
import evaluate_invariant_release_suite as release  # noqa: E402
import measure_v3_remaining_gates as gates  # noqa: E402
from assemble_invariant_candidate import (  # noqa: E402
    ALPHA_COMPLEX,
    ALPHA_REAL,
    WEIGHT_REAL,
)
from invariant_patch_cnn import (  # noqa: E402
    InvariantPatchCNN,
    InvariantPatchConfig,
)


CLASSES = ("am", "bluetooth", "cw")

TINY_COMMON = {
    "patch_length": 16,
    "patch_count": 2,
    "patch_dim": 8,
    "hidden": 12,
    "embed_dim": 4,
    "n_features": 3,
    "dropout": 0.0,
    "set_pool": "mean_std",
}
TINY_CONTRACT = {
    "frontend": "invariant-patch-v1",
    "config": {"patch_length": 16, "patch_count": 2, "target_frac": 0.5},
}


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


def _tiny_payload() -> dict[str, object]:
    """A structurally valid v2 runtime-bundle payload with tiny branches."""
    real = InvariantPatchCNN(
        InvariantPatchConfig(encoder="real", **TINY_COMMON)
    ).eval()
    complex_branch = InvariantPatchCNN(
        InvariantPatchConfig(encoder="complex", **TINY_COMMON)
    ).eval()
    return {
        "schema": subject.V2_BUNDLE_SCHEMA,
        "kind": subject.V2_BUNDLE_KIND,
        "development_only": True,
        "real_config": {"encoder": "real", **TINY_COMMON},
        "complex_config": {"encoder": "complex", **TINY_COMMON},
        "real_state_dict": {
            key: value.detach().cpu()
            for key, value in real.state_dict().items()
        },
        "complex_state_dict": {
            key: value.detach().cpu()
            for key, value in complex_branch.state_dict().items()
        },
        "real_center": torch.arange(4, dtype=torch.float32),
        "complex_center": torch.arange(4, dtype=torch.float32).flip(0),
        "alpha_real": ALPHA_REAL,
        "alpha_complex": ALPHA_COMPLEX,
        "weight_real": WEIGHT_REAL,
        "eps": float(np.float32(1e-12)),
        "feature_mean": torch.zeros(3, dtype=torch.float32),
        "feature_std": torch.ones(3, dtype=torch.float32),
        "classes": list(CLASSES),
        "prototypes": torch.eye(3, 8, dtype=torch.float32),
        "provenance": {
            "release_evidence": False,
            "consumed_test_rows_used": 0,
            "center_fit_population": "training only",
            "prototype_population": "enrollment only",
            "cache_contract": copy.deepcopy(TINY_CONTRACT),
        },
    }


def _tiny_bundle_dir(root: Path) -> tuple[Path, str]:
    """Write (or reuse) a tiny bundle; reuse keeps the digest stable when a
    test drives two runs against the same bundle."""
    directory = root / "v2_bundle_tiny"
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / "runtime_bundle.pt").exists():
        torch.save(_tiny_payload(), directory / "runtime_bundle.pt")
    digest = hashlib.sha256(
        (directory / "runtime_bundle.pt").read_bytes()
    ).hexdigest()
    return directory, digest


class ProtocolIdentityTests(unittest.TestCase):
    """Comparability by construction: the protocol pieces ARE the v3 objects."""

    def test_protocol_functions_are_the_v3_objects(self) -> None:
        self.assertIs(subject.eligible_population, gates.eligible_population)
        self.assertIs(subject.dev_five_shot_split, gates.dev_five_shot_split)
        self.assertIs(
            subject.dev_row_identity_fields, gates.dev_row_identity_fields
        )
        self.assertIs(subject.closed_clean_report, gates.closed_clean_report)
        self.assertIs(subject.five_shot_variant, gates.five_shot_variant)
        self.assertIs(subject.assemble_gates, gates.assemble_gates)
        self.assertIs(subject.validate_split_seed, gates.validate_split_seed)

    def test_lengths_report_name_and_primary_variant_match_v3(self) -> None:
        self.assertIs(subject.DEV_LENGTHS, gates.DEV_LENGTHS)
        self.assertEqual(subject.REPORT_NAME, gates.REPORT_NAME)
        self.assertEqual(
            subject.PRIMARY_FIVE_SHOT_VARIANT, gates.PRIMARY_FIVE_SHOT_VARIANT
        )
        self.assertEqual(
            subject.UNMEASURABLE_RELEASE_LENGTHS,
            gates.UNMEASURABLE_RELEASE_LENGTHS,
        )

    def test_the_recorded_bundle_hash_is_the_handoff_one(self) -> None:
        self.assertEqual(
            subject.EXPECTED_BUNDLE_SHA256,
            "b94ac51b15c1813ca0f94bf3cc3b28ee5f903206b2885ec14dbaa8fddca031e3",
        )
        # And it is the load function's compiled-in default, so a caller
        # cannot forget to verify it.
        signature = inspect.signature(subject.load_v2_bundle)
        self.assertEqual(
            signature.parameters["expected_sha256"].default,
            subject.EXPECTED_BUNDLE_SHA256,
        )

    def test_the_docstring_states_the_load_bearing_facts(self) -> None:
        text = subject.__doc__ or ""
        for phrase in (
            "21.3",
            "eligib",
            "SHA-256",
            "development evidence, not release evidence",
            "sealed_release_data_used",
            "consumed_test_rows_used",
        ):
            self.assertIn(phrase.lower(), text.lower(), phrase)


class BundleValidationTests(unittest.TestCase):
    def test_a_valid_payload_returns_the_frozen_assets(self) -> None:
        validated = subject.validate_bundle_payload(_tiny_payload())
        self.assertEqual(validated["embed_dim"], 4)
        self.assertEqual(validated["n_features"], 3)
        self.assertEqual(validated["classes"], list(CLASSES))
        self.assertEqual(validated["prototypes"].shape, (3, 8))
        self.assertEqual(validated["real_center"].shape, (4,))
        self.assertEqual(validated["feature_std"].shape, (3,))
        self.assertEqual(
            validated["cache_contract"]["frontend"], "invariant-patch-v1"
        )

    def test_a_missing_key_fails_closed(self) -> None:
        payload = _tiny_payload()
        del payload["prototypes"]
        with self.assertRaisesRegex(ValueError, "missing keys"):
            subject.validate_bundle_payload(payload)

    def test_wrong_schema_or_kind_fails_closed(self) -> None:
        payload = _tiny_payload()
        payload["schema"] = 3
        with self.assertRaisesRegex(ValueError, "schema"):
            subject.validate_bundle_payload(payload)
        payload = _tiny_payload()
        payload["kind"] = "something_else"
        with self.assertRaisesRegex(ValueError, "kind"):
            subject.validate_bundle_payload(payload)

    def test_release_or_consumed_provenance_fails_closed(self) -> None:
        for key, value in (
            ("release_evidence", True),
            ("consumed_test_rows_used", 1),
            ("center_fit_population", "selection"),
            ("prototype_population", "selection"),
        ):
            payload = _tiny_payload()
            payload["provenance"][key] = value
            with self.assertRaises(RuntimeError):
                subject.validate_bundle_payload(payload)
        payload = _tiny_payload()
        payload["development_only"] = False
        with self.assertRaises(RuntimeError):
            subject.validate_bundle_payload(payload)

    def test_a_non_v2_frontend_contract_fails_closed(self) -> None:
        payload = _tiny_payload()
        payload["provenance"]["cache_contract"]["frontend"] = "time-domain-v3"
        with self.assertRaisesRegex(RuntimeError, "invariant-patch-v1"):
            subject.validate_bundle_payload(payload)

    def test_fusion_constants_must_match_the_assembly_module(self) -> None:
        for name in ("alpha_real", "alpha_complex", "weight_real"):
            payload = _tiny_payload()
            payload[name] = float(payload[name]) + 0.05
            with self.assertRaisesRegex(RuntimeError, "_fuse_numpy"):
                subject.validate_bundle_payload(payload)

    def test_wrong_prototype_shape_fails_closed(self) -> None:
        payload = _tiny_payload()
        payload["prototypes"] = torch.eye(3, 4, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "prototypes"):
            subject.validate_bundle_payload(payload)

    def test_non_positive_feature_std_fails_closed(self) -> None:
        payload = _tiny_payload()
        payload["feature_std"] = torch.tensor([1.0, 0.0, 1.0])
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            subject.validate_bundle_payload(payload)

    def test_mismatched_branch_encoders_fail_closed(self) -> None:
        payload = _tiny_payload()
        payload["complex_config"] = {"encoder": "real", **TINY_COMMON}
        with self.assertRaisesRegex(ValueError, "encoder"):
            subject.validate_bundle_payload(payload)


class BundleLoadingTests(unittest.TestCase):
    def test_a_valid_bundle_loads_read_only_and_rebuilds_eval_models(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory, digest = _tiny_bundle_dir(Path(tmp))
            before = (directory / "runtime_bundle.pt").read_bytes()
            bundle = subject.load_v2_bundle(directory, expected_sha256=digest)
            self.assertEqual(bundle["sha256"]["runtime_bundle.pt"], digest)
            self.assertFalse(bundle["real"].training)
            self.assertFalse(bundle["complex"].training)
            self.assertGreater(bundle["parameter_count"], 0)
            self.assertEqual(bundle["prototypes"].shape, (3, 8))
            # Read-only: the file is untouched.
            self.assertEqual(
                (directory / "runtime_bundle.pt").read_bytes(), before
            )

    def test_a_hash_mismatch_refuses_before_deserializing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory, _digest = _tiny_bundle_dir(Path(tmp))
            with mock.patch.object(
                subject.torch, "load", side_effect=AssertionError("must not load")
            ):
                with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                    subject.load_v2_bundle(
                        directory, expected_sha256="0" * 64
                    )

    def test_sealed_and_release_directories_are_refused(self) -> None:
        for candidate in (
            "/tmp/training/artifacts/releases/suite",
            "/tmp/sealed_v2_suite",
        ):
            with self.assertRaisesRegex(ValueError, "refuses release or sealed"):
                subject.load_v2_bundle(Path(candidate))

    def test_a_missing_bundle_file_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                subject.load_v2_bundle(Path(tmp))


class StandardizeFeaturesTests(unittest.TestCase):
    def test_the_bundle_statistics_are_applied(self) -> None:
        raw = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        mean = np.asarray([1.0, 2.0], dtype=np.float32)
        std = np.asarray([2.0, 4.0], dtype=np.float32)
        result = subject.standardize_features(raw, mean, std)
        np.testing.assert_allclose(
            result, [[0.0, 0.0], [1.0, 0.5]], rtol=0, atol=0
        )
        self.assertEqual(result.dtype, np.float32)

    def test_shape_mismatches_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "raw features"):
            subject.standardize_features(
                np.zeros((2, 3), dtype=np.float32),
                np.zeros(2, dtype=np.float32),
                np.ones(2, dtype=np.float32),
            )
        with self.assertRaisesRegex(ValueError, "disagree"):
            subject.standardize_features(
                np.zeros((2, 2), dtype=np.float32),
                np.zeros(2, dtype=np.float32),
                np.ones(3, dtype=np.float32),
            )

    def test_non_finite_output_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            subject.standardize_features(
                np.asarray([[np.inf, 0.0]], dtype=np.float32),
                np.zeros(2, dtype=np.float32),
                np.ones(2, dtype=np.float32),
            )


class EmbedPrefixTests(unittest.TestCase):
    def _bundle(self) -> dict[str, object]:
        with tempfile.TemporaryDirectory() as tmp:
            directory, digest = _tiny_bundle_dir(Path(tmp))
            return subject.load_v2_bundle(directory, expected_sha256=digest)

    def test_prefixes_are_preprocessed_standardized_and_fused(self) -> None:
        bundle = self._bundle()
        packed_length = 16 * 2
        calls: list[int] = []

        def fake_preprocess(prefix, *, patch_length, patch_count, target_frac):
            self.assertEqual(patch_length, 16)
            self.assertEqual(patch_count, 2)
            self.assertEqual(target_frac, 0.5)
            calls.append(len(prefix))
            rng = np.random.default_rng(len(calls))
            packed = rng.normal(size=(2, packed_length)).astype(np.float32)
            features = rng.normal(size=3).astype(np.float32)
            return packed, features, {}

        captures = [
            np.zeros(256, dtype=np.complex128),
            np.ones(300, dtype=np.complex128),
        ]
        with mock.patch.object(
            subject.v2_preprocess, "preprocess", side_effect=fake_preprocess
        ):
            fused = subject.v2_embed_prefixes(
                bundle, captures, 128, torch.device("cpu")
            )
        self.assertEqual(calls, [128, 128])
        self.assertEqual(fused.shape, (2, 8))
        self.assertEqual(fused.dtype, np.float32)
        # _fuse_numpy output is unit-normalized.
        np.testing.assert_allclose(
            np.linalg.norm(fused, axis=1), np.ones(2), rtol=0, atol=1e-5
        )

    def test_a_short_capture_fails_closed(self) -> None:
        bundle = self._bundle()
        with self.assertRaisesRegex(ValueError, "shorter than"):
            subject.v2_embed_prefixes(
                bundle,
                [np.zeros(64, dtype=np.complex128)],
                128,
                torch.device("cpu"),
            )

    def test_a_non_positive_length_fails_closed(self) -> None:
        bundle = self._bundle()
        with self.assertRaisesRegex(ValueError, "positive"):
            subject.v2_embed_prefixes(
                bundle,
                [np.zeros(64, dtype=np.complex128)],
                0,
                torch.device("cpu"),
            )


class ReplayCompareTests(unittest.TestCase):
    def test_agreement_returns_the_delta(self) -> None:
        recomputed = np.asarray([1.0, 2.0], dtype=np.float64)
        recorded = recomputed + 1e-8
        delta = subject.replay_compare("centers", recomputed, recorded, 2e-6)
        self.assertLess(delta, 2e-6)

    def test_drift_above_tolerance_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "replay mismatch"):
            subject.replay_compare(
                "prototypes",
                np.asarray([1.0]),
                np.asarray([1.1]),
                2e-6,
            )

    def test_non_finite_deviation_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "replay mismatch"):
            subject.replay_compare(
                "centers", np.asarray([np.nan]), np.asarray([1.0]), 2e-6
            )


class FrontendConsistencyTests(unittest.TestCase):
    def test_agreement_is_recorded(self) -> None:
        value = np.random.default_rng(3).normal(size=(5, 8))
        report = subject.frontend_consistency_check(
            value, value + 1e-5, tolerance=2e-3
        )
        self.assertLessEqual(report["max_abs_delta"], 2e-3)
        self.assertEqual(report["rows_compared"], 5)

    def test_drift_fails_closed(self) -> None:
        value = np.zeros((3, 4))
        with self.assertRaisesRegex(RuntimeError, "deviate"):
            subject.frontend_consistency_check(
                value, value + 0.1, tolerance=2e-3
            )

    def test_shape_mismatch_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            subject.frontend_consistency_check(
                np.zeros((3, 4)), np.zeros((4, 4)), tolerance=2e-3
            )


class SelectionPositionsTests(unittest.TestCase):
    def test_positions_are_into_the_cached_array_order(self) -> None:
        corpus = np.asarray([10, 4, 7, 99, 3], dtype=np.int64)
        eligible = np.asarray([4, 99, 10], dtype=np.int64)
        np.testing.assert_array_equal(
            subject.selection_positions(corpus, eligible),
            np.asarray([1, 3, 0], dtype=np.int64),
        )

    def test_an_unknown_corpus_index_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "drifted"):
            subject.selection_positions(
                np.asarray([1, 2], dtype=np.int64),
                np.asarray([3], dtype=np.int64),
            )


class ResolveSplitSeedTests(unittest.TestCase):
    def test_explicit_seed_wins(self) -> None:
        seed, source = subject.resolve_split_seed(
            4242, {"protocol": {"split_seed": 20260730}}
        )
        self.assertEqual(seed, 4242)
        self.assertIn("explicit", source)

    def test_the_v3_report_seed_is_adopted(self) -> None:
        seed, source = subject.resolve_split_seed(
            None, {"protocol": {"split_seed": 20260730}}
        )
        self.assertEqual(seed, 20260730)
        self.assertIn("v3-report", source)

    def test_the_default_is_the_bundle_model_seed(self) -> None:
        seed, source = subject.resolve_split_seed(None, None)
        self.assertEqual(seed, subject.V2_BUNDLE_SEED)
        self.assertIn("bundle", source)

    def test_release_seeds_are_refused_from_every_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "20260731"):
            subject.resolve_split_seed(20260731, None)
        with self.assertRaisesRegex(ValueError, str(release.NOVELTY_SEED)):
            subject.resolve_split_seed(
                None, {"protocol": {"split_seed": release.NOVELTY_SEED}}
            )


def _fake_bundle_block() -> dict[str, object]:
    return {
        "classes": list(CLASSES),
        "embed_dim": 4,
        "sha256": {"runtime_bundle.pt": "f" * 64},
    }


def _fake_variants(sha_a: str, sha_b: str) -> dict[str, dict[str, object]]:
    return {
        "enrollment_support": {"split": {"support_positions_sha256": sha_a}},
        "selection_support": {"split": {"support_positions_sha256": sha_b}},
    }


def _fake_v3_report(
    selection_sha: str,
    enrollment_sha: str,
    *,
    split_seed: int = 20260730,
    support_shas: tuple[str, str] = ("a" * 64, "b" * 64),
) -> dict[str, object]:
    return {
        "_path": "/tmp/dev/v3/remaining_gates.json",
        "seed": 20260730,
        "measurement": gates.MEASUREMENT_VERSION,
        "development_only": True,
        "sealed_release_data_used": 0,
        "consumed_test_rows_used": 0,
        "protocol": {"split_seed": split_seed},
        "population": {
            "selection": {"eligible_indices_sha256": selection_sha},
            "enrollment": {"eligible_indices_sha256": enrollment_sha},
        },
        "five_shot_per_variant": {
            "enrollment_support": {
                "split": {"support_positions_sha256": support_shas[0]}
            },
            "selection_support": {
                "split": {"support_positions_sha256": support_shas[1]}
            },
        },
        "gates": {"five_shot_worst_length_balanced_primary": {"value": 0.7564}},
    }


class ComparabilityBlockTests(unittest.TestCase):
    def test_without_a_v3_report_the_digests_are_still_recorded(self) -> None:
        block = subject.comparability_block(
            bundle=_fake_bundle_block(),
            selection_sha256="c" * 64,
            enrollment_sha256="d" * 64,
            split_seed=20260727,
            split_seed_source="v2 bundle model seed",
            five_shot_by_variant=_fake_variants("a" * 64, "b" * 64),
            v3_report=None,
        )
        self.assertFalse(block["v3_report_cross_check"]["performed"])
        shared = block["shared_with_v3_measurement"]
        self.assertEqual(shared["eligible_selection_indices_sha256"], "c" * 64)
        self.assertEqual(shared["eligible_enrollment_indices_sha256"], "d" * 64)
        self.assertEqual(shared["gate_floors"], dict(release.GATE_FLOORS))
        differs = block["differs_from_v3_measurement"]
        self.assertTrue(
            differs["frontend"]["this_measurement"]["uses_frequency_transform"]
        )
        self.assertFalse(
            differs["frontend"]["v3_measurement"]["uses_frequency_transform"]
        )
        self.assertEqual(differs["fused_embedding_dim"]["this_measurement"], 8)

    def test_a_matching_v3_report_proves_the_shared_population(self) -> None:
        block = subject.comparability_block(
            bundle=_fake_bundle_block(),
            selection_sha256="c" * 64,
            enrollment_sha256="d" * 64,
            split_seed=20260730,
            split_seed_source="adopted from --v3-report",
            five_shot_by_variant=_fake_variants("a" * 64, "b" * 64),
            v3_report=_fake_v3_report("c" * 64, "d" * 64),
        )
        check = block["v3_report_cross_check"]
        self.assertTrue(check["performed"])
        self.assertTrue(check["eligible_index_sets_match"])
        self.assertTrue(check["support_draw"]["support_draws_identical"])
        self.assertIn("five_shot_worst_length_balanced_primary", check["v3_gates_for_reference"])

    def test_a_population_mismatch_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not be comparable"):
            subject.comparability_block(
                bundle=_fake_bundle_block(),
                selection_sha256="c" * 64,
                enrollment_sha256="d" * 64,
                split_seed=20260730,
                split_seed_source="adopted from --v3-report",
                five_shot_by_variant=_fake_variants("a" * 64, "b" * 64),
                v3_report=_fake_v3_report("e" * 64, "d" * 64),
            )

    def test_a_support_draw_mismatch_with_the_same_seed_fails_closed(
        self,
    ) -> None:
        with self.assertRaisesRegex(RuntimeError, "support draw"):
            subject.comparability_block(
                bundle=_fake_bundle_block(),
                selection_sha256="c" * 64,
                enrollment_sha256="d" * 64,
                split_seed=20260730,
                split_seed_source="adopted from --v3-report",
                five_shot_by_variant=_fake_variants("a" * 64, "b" * 64),
                v3_report=_fake_v3_report(
                    "c" * 64, "d" * 64, support_shas=("9" * 64, "b" * 64)
                ),
            )

    def test_different_split_seeds_are_recorded_not_compared(self) -> None:
        block = subject.comparability_block(
            bundle=_fake_bundle_block(),
            selection_sha256="c" * 64,
            enrollment_sha256="d" * 64,
            split_seed=20260727,
            split_seed_source="v2 bundle model seed",
            five_shot_by_variant=_fake_variants("9" * 64, "8" * 64),
            v3_report=_fake_v3_report("c" * 64, "d" * 64, split_seed=20260730),
        )
        support = block["v3_report_cross_check"]["support_draw"]
        self.assertFalse(support["split_seed_matches"])
        self.assertFalse(support["support_draws_identical"])


class V3ReportLoadingTests(unittest.TestCase):
    def _write(self, tmp: Path, report: dict[str, object]) -> Path:
        path = tmp / "remaining_gates.json"
        report = {
            key: value for key, value in report.items() if key != "_path"
        }
        path.write_text(json.dumps(report), encoding="utf-8")
        return path

    def test_a_valid_report_loads_and_records_its_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(Path(tmp), _fake_v3_report("c" * 64, "d" * 64))
            report = subject.load_v3_report(path)
            self.assertEqual(report["_path"], str(path.resolve()))

    def test_a_non_v3_measurement_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            broken = _fake_v3_report("c" * 64, "d" * 64)
            broken["measurement"] = "something-else"
            path = self._write(Path(tmp), broken)
            with self.assertRaisesRegex(ValueError, "not a"):
                subject.load_v3_report(path)

    def test_a_sealed_claiming_report_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            broken = _fake_v3_report("c" * 64, "d" * 64)
            broken["sealed_release_data_used"] = 1
            path = self._write(Path(tmp), broken)
            with self.assertRaisesRegex(RuntimeError, "refusing"):
                subject.load_v3_report(path)

    def test_release_paths_are_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "refuses release or sealed"):
            subject.load_v3_report(
                Path("/tmp/training/artifacts/releases/remaining_gates.json")
            )


class DriverTests(unittest.TestCase):
    """End-to-end report assembly with corpus, preprocessing, and inference
    stubbed out, mirroring the v3 measurement's own driver tests."""

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

    def _run(
        self,
        tmp: Path,
        *,
        output_name: str = "out",
        **overrides: object,
    ) -> tuple[dict[str, object], Path]:
        directory, digest = _tiny_bundle_dir(tmp)
        output_dir = tmp / output_name

        selection = self._population("c" * 64)
        enrollment = self._population("d" * 64)
        populations = iter((selection, enrollment))

        def fake_eligible(manifest, indices, classes, raw):
            return next(populations)

        def fake_embed(bundle, captures, length, device, **_kwargs):
            labels = np.asarray(
                [int(round(float(capture[0].real))) for capture in captures],
                dtype=np.int64,
            )
            return np.eye(3, 8, dtype=np.float32)[labels]

        def fake_embed_cached(bundle, packed, features, device, **_kwargs):
            return np.eye(3, 8, dtype=np.float32)[selection["labels"]]

        n_selection = len(selection["labels"])
        source = {
            "va_idx": np.arange(n_selection, dtype=np.int64),
            "en_idx": np.arange(len(enrollment["labels"]), dtype=np.int64),
            "xva": np.zeros((n_selection, 2), dtype=np.float32),
            "fva": np.zeros((n_selection, 3), dtype=np.float32),
            "fmean": np.zeros(3, dtype=np.float32),
            "fstd": np.ones(3, dtype=np.float32),
            "classes": list(CLASSES),
            "cache_contract": copy.deepcopy(TINY_CONTRACT),
            "data_audit": {"consumed_test_rows_exposed": 0},
        }

        replay_stub = {
            "tolerance": subject.DEFAULT_REPLAY_TOLERANCE,
            "max_abs_deltas": {"prototypes": 0.0},
            "worst_abs_delta": 0.0,
            "fit_performed": False,
        }

        args = subject.build_parser().parse_args(
            [
                "--bundle-dir",
                str(directory),
                "--output-dir",
                str(output_dir),
                "--device",
                "cpu",
            ]
        )
        for key, value in overrides.items():
            setattr(args, key, value)

        real_load = subject.load_v2_bundle
        with mock.patch.object(
            subject,
            "load_v2_bundle",
            side_effect=lambda d: real_load(d, expected_sha256=digest),
        ), mock.patch.object(
            subject.invariant_data, "load", return_value=source
        ), mock.patch.object(
            subject.dev, "_load_manifest", return_value={"items": selection["items"]}
        ), mock.patch.object(
            subject.dev, "_raw_memmap", return_value=None
        ), mock.patch.object(
            subject, "eligible_population", side_effect=fake_eligible
        ), mock.patch.object(
            subject, "v2_embed_prefixes", side_effect=fake_embed
        ), mock.patch.object(
            subject, "v2_embed_cached", side_effect=fake_embed_cached
        ), mock.patch.object(
            subject, "bundle_replay_check", return_value=replay_stub
        ):
            return subject.run(args), output_dir

    def test_report_is_written_with_the_expected_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, output_dir = self._run(Path(tmp))
            written = json.loads(
                (output_dir / subject.REPORT_NAME).read_text(encoding="utf-8")
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
            self.assertEqual(written["seed"], subject.V2_BUNDLE_SEED)
            self.assertIn("source_sha256", written)
            self.assertIn("comparability", written)
            self.assertIn("reproduction_check", written)
            self.assertEqual(result["device"], "cpu")

    def test_every_length_both_variants_and_all_gates_are_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, _output_dir = self._run(Path(tmp))
            self.assertEqual(
                sorted(int(key) for key in result["closed_clean_per_length"]),
                sorted(subject.DEV_LENGTHS),
            )
            self.assertEqual(
                set(result["five_shot_per_variant"]),
                set(subject.FIVE_SHOT_VARIANTS),
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

    def test_the_frontend_block_declares_the_fft_bearing_v2_frontend(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, _output_dir = self._run(Path(tmp))
            frontend = result["frontend"]
            self.assertEqual(frontend["version"], "invariant-patch-v1")
            self.assertEqual(frontend["estimator_version"], "hybrid-v3")
            self.assertIs(frontend["uses_frequency_transform"], True)

    def test_the_comparability_digests_are_the_population_digests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, _output_dir = self._run(Path(tmp))
            shared = result["comparability"]["shared_with_v3_measurement"]
            self.assertEqual(
                shared["eligible_selection_indices_sha256"],
                result["population"]["selection"]["eligible_indices_sha256"],
            )
            self.assertEqual(
                shared["eligible_enrollment_indices_sha256"],
                result["population"]["enrollment"]["eligible_indices_sha256"],
            )

    def test_a_matching_v3_report_is_cross_checked_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first, _output_dir = self._run(
                Path(tmp), output_name="first", split_seed=20260730
            )
            v3_report = _fake_v3_report(
                first["population"]["selection"]["eligible_indices_sha256"],
                first["population"]["enrollment"]["eligible_indices_sha256"],
                split_seed=20260730,
                support_shas=(
                    first["five_shot_per_variant"]["enrollment_support"][
                        "split"
                    ]["support_positions_sha256"],
                    first["five_shot_per_variant"]["selection_support"][
                        "split"
                    ]["support_positions_sha256"],
                ),
            )
            report_path = Path(tmp) / "v3_remaining_gates.json"
            report_path.write_text(
                json.dumps(
                    {k: v for k, v in v3_report.items() if k != "_path"}
                ),
                encoding="utf-8",
            )
            second, _second_dir = self._run(
                Path(tmp), output_name="second", v3_report=str(report_path)
            )
            check = second["comparability"]["v3_report_cross_check"]
            self.assertTrue(check["performed"])
            self.assertTrue(check["support_draw"]["support_draws_identical"])
            # The split seed was adopted from the report.
            self.assertEqual(second["protocol"]["split_seed"], 20260730)

    def test_a_mismatched_v3_population_fails_the_whole_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            v3_report = _fake_v3_report("9" * 64, "8" * 64)
            report_path = Path(tmp) / "v3_remaining_gates.json"
            report_path.write_text(
                json.dumps(
                    {k: v for k, v in v3_report.items() if k != "_path"}
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "not be comparable"):
                self._run(Path(tmp), v3_report=str(report_path))
            # And no report was written.
            self.assertFalse(
                (Path(tmp) / "out" / subject.REPORT_NAME).exists()
            )

    def test_an_existing_report_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _result, output_dir = self._run(Path(tmp))
            args = subject.build_parser().parse_args(
                [
                    "--bundle-dir",
                    str(Path(tmp) / "v2_bundle_tiny"),
                    "--output-dir",
                    str(output_dir),
                    "--device",
                    "cpu",
                ]
            )
            with self.assertRaisesRegex(FileExistsError, subject.REPORT_NAME):
                subject.run(args)

    def test_the_release_seed_is_refused_at_the_command_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "20260731"):
                self._run(Path(tmp), split_seed=20260731)

    def test_release_and_sealed_paths_are_refused_before_any_io(self) -> None:
        for flag, value in (
            ("--bundle-dir", "/tmp/training/artifacts/releases/bundle"),
            ("--output-dir", "/tmp/sealed_suite/out"),
        ):
            args = subject.build_parser().parse_args(
                [flag, value, "--device", "cpu"]
                + (
                    ["--output-dir", "/tmp/dev-out"]
                    if flag == "--bundle-dir"
                    else ["--bundle-dir", "/tmp/dev-bundle"]
                )
            )
            with self.assertRaisesRegex(ValueError, "refuses release or sealed"):
                subject.run(args)


class DefaultPathTests(unittest.TestCase):
    def test_the_default_output_is_not_the_bundle_directory(self) -> None:
        self.assertNotEqual(
            subject.DEFAULT_OUTPUT_DIR.resolve(),
            subject.DEFAULT_BUNDLE_DIR.resolve(),
        )
        self.assertNotIn(
            subject.DEFAULT_BUNDLE_DIR.resolve(),
            subject.DEFAULT_OUTPUT_DIR.resolve().parents,
        )

    def test_neither_default_is_a_release_or_sealed_path(self) -> None:
        for path in (subject.DEFAULT_BUNDLE_DIR, subject.DEFAULT_OUTPUT_DIR):
            lowered = {part.lower() for part in path.resolve().parts}
            self.assertNotIn("releases", lowered)
            self.assertFalse(any("sealed" in part for part in lowered))

    def test_the_default_bundle_is_the_frozen_v2_bundle(self) -> None:
        self.assertEqual(
            subject.DEFAULT_BUNDLE_DIR.name,
            "invariant_fusion_runtime_bundle_v2_seed20260727",
        )


if __name__ == "__main__":
    unittest.main()
