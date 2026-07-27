"""Unit tests for the v3 time-domain fusion runtime bundle exporter.

Every fixture here is synthetic and closed-form.  No corpus, split, sealed
release suite, or consumed test row is opened by this suite.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
V2 = HERE.parent
ZPAB = V2.parent
TRAINING = ZPAB.parent
REPO = TRAINING.parent
for _path in (TRAINING, ZPAB, V2, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import export_v3_fusion_runtime as subject  # noqa: E402
import time_domain_geometry as geometry  # noqa: E402
import time_domain_invariant_patch_preprocess as td_preprocess  # noqa: E402
import v3_time_domain_openset as openset  # noqa: E402
from invariant_fusion import CenteredInvariantFusion  # noqa: E402
from invariant_patch_cnn import (  # noqa: E402
    InvariantPatchCNN,
    InvariantPatchConfig,
)


PATCH_LENGTH = 16
PATCH_COUNT = 2
EMBED_DIM = 4
N_FEATURES = 12
# The real label set.  The exporter cross-checks the derived class order
# against the corpus manifest when that manifest is present, so a synthetic
# fixture must still use the shipping names.
CLASSES = ("am", "bluetooth", "cw", "dsss", "fm", "gsm", "ofdm")
N_CLASSES = len(CLASSES)
TARGET_FRAC = 0.5

# Short probe captures keep the suite fast; the export path is identical.
TEST_SPECIFICATIONS = (
    {
        "name": "two-tone-scale-1x",
        "length": 1024,
        "physical_scale": 1.0,
        "center": 0.13,
        "kind": "two_tone",
    },
    {
        "name": "two-tone-scale-2x",
        "length": 512,
        "physical_scale": 2.0,
        "center": 0.13,
        "kind": "two_tone",
    },
)


def _configs() -> tuple[InvariantPatchConfig, InvariantPatchConfig]:
    common = {
        "patch_length": PATCH_LENGTH,
        "patch_count": PATCH_COUNT,
        "patch_dim": 8,
        "hidden": 12,
        "embed_dim": EMBED_DIM,
        "n_features": N_FEATURES,
        "dropout": 0.0,
    }
    return (
        InvariantPatchConfig(encoder="real", **common).validate(),
        InvariantPatchConfig(encoder="complex", **common).validate(),
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_fusion_artifact(directory: Path, *, seed: int = 3) -> dict:
    """Write a synthetic but structurally faithful assemble_v3_fusion output."""
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    real_config, complex_config = _configs()
    fusion = CenteredInvariantFusion(
        InvariantPatchCNN(real_config),
        InvariantPatchCNN(complex_config),
        rng.normal(size=EMBED_DIM).astype(np.float32),
        rng.normal(size=EMBED_DIM).astype(np.float32),
        alpha_real=0.2,
        alpha_complex=0.2,
        weight_real=0.5,
        eps=1e-12,
    ).eval()

    state_path = directory / "fusion_state_dict.pt"
    torch.save(
        {key: value.detach().cpu() for key, value in fusion.state_dict().items()},
        state_path,
    )
    arrays = {
        "fusion_prototypes.npy": rng.normal(
            size=(N_CLASSES, 2 * EMBED_DIM)
        ).astype(np.float32),
        "real_center.npy": fusion.real_center.detach().cpu().numpy(),
        "complex_center.npy": fusion.complex_center.detach().cpu().numpy(),
        "feature_mean.npy": rng.normal(size=N_FEATURES).astype(np.float32),
        "feature_std.npy": (
            np.abs(rng.normal(size=N_FEATURES)) + 0.5
        ).astype(np.float32),
    }
    for name, value in arrays.items():
        np.save(directory / name, value)

    def digest(name: str) -> str:
        return _sha256_bytes((directory / name).read_bytes())

    frontend = dict(td_preprocess.preprocess_metadata())
    frontend["patch_length"] = PATCH_LENGTH
    frontend["patch_count"] = PATCH_COUNT
    frontend["target_frac"] = TARGET_FRAC

    counts = {name: 10 + index for index, name in enumerate(CLASSES)}
    metrics = {
        "status": "complete",
        "development_only": True,
        "release_evidence": False,
        "sealed_release_data_used": 0,
        "consumed_test_rows_used": 0,
        "encoder": "fusion",
        "seed": 20260730,
        "frontend": frontend,
        "architecture": fusion.config(),
        "parameter_count": 1234,
        "fusion": {
            "kind": "centered_invariant_fusion",
            "rule": "unit(concat(...))",
            "weight_real": 0.5,
            "weight_complex": 0.5,
            "alpha_real": 0.2,
            "alpha_complex": 0.2,
            "eps": 1e-12,
        },
        "fitting_contract": {
            "center_fit_population": "training only",
            "prototype_population": "enrollment only",
            "sealed_release_rows_loaded": 0,
            "consumed_test_rows_loaded": 0,
        },
        "closed_selection": {"balanced_accuracy": 0.8675},
        "data_audit": {
            "consumed_test_rows_exposed": 0,
            "class_counts": {
                "train": dict(counts),
                "enroll": dict(counts),
                "selection": dict(counts),
            },
        },
        "source_index_contract": {
            "contract": {"config": {"target_frac": TARGET_FRAC}},
        },
        "source_sha256": {
            "time_domain_geometry.py": subject.frontend_source_hashes()[
                "training/time_domain_geometry.py"
            ],
            "time_domain_invariant_patch_preprocess.py": (
                subject.frontend_source_hashes()[
                    "training/time_domain_invariant_patch_preprocess.py"
                ]
            ),
        },
        "source_branches": {
            "real": {
                "encoder": "real",
                # assemble_v3_fusion records an absolute path here.
                "directory": str(
                    TRAINING
                    / "zplane_ab"
                    / "v2_full_variation"
                    / "artifacts"
                    / "branch_real"
                ),
                "sha256": {"dev_metrics.json": "0" * 64},
            },
            "complex": {
                "encoder": "complex",
                "directory": str(
                    TRAINING
                    / "zplane_ab"
                    / "v2_full_variation"
                    / "artifacts"
                    / "branch_complex"
                ),
                "sha256": {"dev_metrics.json": "1" * 64},
            },
        },
        "artifacts": {
            "state_dict": "fusion_state_dict.pt",
            "state_dict_sha256": digest("fusion_state_dict.pt"),
            "prototypes": "fusion_prototypes.npy",
            "prototypes_sha256": digest("fusion_prototypes.npy"),
            "real_center": "real_center.npy",
            "real_center_sha256": digest("real_center.npy"),
            "complex_center": "complex_center.npy",
            "complex_center_sha256": digest("complex_center.npy"),
            "feature_mean": "feature_mean.npy",
            "feature_mean_sha256": digest("feature_mean.npy"),
            "feature_std": "feature_std.npy",
            "feature_std_sha256": digest("feature_std.npy"),
        },
    }
    _write(directory / "dev_metrics.json", metrics)
    return metrics


def _write(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _rewrite_metrics(directory: Path, mutate) -> None:
    path = directory / "dev_metrics.json"
    with path.open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    mutate(metrics)
    _write(path, metrics)


def _export(source: Path, output: Path, **kwargs) -> dict:
    return subject.export_bundle(
        source,
        output,
        specifications=TEST_SPECIFICATIONS,
        **kwargs,
    )


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256_bytes(path.read_bytes())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class _Fixture(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        # addCleanup rather than tearDown: a setUp that fails after the
        # directory exists must still remove it, or the suite leaks temp trees
        # and PYTHONWARNINGS=error turns the exit-time ResourceWarning into a
        # second, misleading failure.
        self.addCleanup(self._temporary.cleanup)
        self.root = Path(self._temporary.name)
        self.source = self.root / "assembly"
        write_fusion_artifact(self.source)
        self.output = self.root / "bundle"


class SchemaIdentityTests(_Fixture):
    def test_schema_id_is_not_a_v2_or_hybrid_name(self) -> None:
        self.assertNotIn(subject.SCHEMA_ID, subject.FORBIDDEN_SCHEMA_IDS)
        self.assertNotIn("invariant-patch-v1", subject.SCHEMA_ID)
        self.assertNotIn("hybrid-v3", subject.SCHEMA_ID)
        for forbidden in ("invariant-patch-v1", "hybrid-v3"):
            self.assertIn(forbidden, subject.FORBIDDEN_SCHEMA_IDS)

    def test_validate_schema_identity_rejects_a_forbidden_id(self) -> None:
        original = subject.SCHEMA_ID
        try:
            subject.SCHEMA_ID = "hybrid-v3"
            with self.assertRaises(RuntimeError):
                subject.validate_schema_identity()
        finally:
            subject.SCHEMA_ID = original
        subject.validate_schema_identity()

    def test_manifest_declares_the_v3_schema_and_disclaims_the_others(self) -> None:
        manifest = _export(self.source, self.output)
        self.assertEqual(manifest["schema"], subject.SCHEMA_ID)
        self.assertEqual(manifest["schema_version"], subject.SCHEMA_VERSION)
        self.assertIn("invariant-patch-v1", manifest["not_compatible_with_schema_ids"])
        self.assertIn("hybrid-v3", manifest["not_compatible_with_schema_ids"])
        serialized = json.dumps(manifest)
        self.assertNotIn('"schema": "hybrid-v3"', serialized)


class FrontendBindingTests(_Fixture):
    def test_bound_sources_are_the_two_time_domain_files(self) -> None:
        self.assertEqual(
            sorted(subject.BOUND_FRONTEND_SOURCES),
            [
                "training/time_domain_geometry.py",
                "training/time_domain_invariant_patch_preprocess.py",
            ],
        )

    def test_bound_hashes_match_the_live_files(self) -> None:
        hashes = subject.frontend_source_hashes()
        for name, path in subject.BOUND_FRONTEND_SOURCES.items():
            self.assertEqual(hashes[name], _sha256_bytes(path.read_bytes()))

    def test_manifest_binds_both_frontend_hashes(self) -> None:
        manifest = _export(self.source, self.output)
        self.assertEqual(
            manifest["frontend"]["source_sha256"],
            subject.frontend_source_hashes(),
        )

    def test_export_refuses_a_stale_frontend_binding(self) -> None:
        _rewrite_metrics(
            self.source,
            lambda metrics: metrics["source_sha256"].__setitem__(
                "time_domain_geometry.py", "0" * 64
            ),
        )
        with self.assertRaises(RuntimeError):
            _export(self.source, self.output)

    def test_export_refuses_a_missing_frontend_hash_record(self) -> None:
        _rewrite_metrics(
            self.source,
            lambda metrics: metrics["source_sha256"].pop(
                "time_domain_invariant_patch_preprocess.py"
            ),
        )
        with self.assertRaises(RuntimeError):
            _export(self.source, self.output)

    def test_manifest_embeds_module_metadata_with_no_frequency_transform(self) -> None:
        manifest = _export(self.source, self.output)
        frontend = manifest["frontend"]
        self.assertIs(frontend["uses_frequency_transform"], False)
        self.assertEqual(
            frontend["module_metadata"], td_preprocess.preprocess_metadata()
        )
        self.assertIs(
            frontend["module_metadata"]["uses_frequency_transform"], False
        )
        self.assertIs(
            frontend["module_metadata"]["geometry"]["uses_frequency_transform"],
            False,
        )
        self.assertEqual(frontend["version"], td_preprocess.PREPROCESS_VERSION)
        self.assertEqual(
            frontend["estimator_version"], td_preprocess.ESTIMATOR_VERSION
        )

    def test_export_refuses_an_fft_bearing_assembly(self) -> None:
        def mutate(metrics: dict) -> None:
            metrics["frontend"]["uses_frequency_transform"] = True

        _rewrite_metrics(self.source, mutate)
        with self.assertRaises(RuntimeError):
            _export(self.source, self.output)

    def test_export_refuses_a_changed_geometry_estimator_record(self) -> None:
        def mutate(metrics: dict) -> None:
            metrics["frontend"]["geometry"]["maximum_lag"] = 999

        _rewrite_metrics(self.source, mutate)
        with self.assertRaises(RuntimeError):
            _export(self.source, self.output)
        self.assertIs(
            geometry.estimator_metadata()["uses_frequency_transform"], False
        )


class ClassOrderTests(_Fixture):
    def test_class_order_is_the_sorted_audit_order(self) -> None:
        with (self.source / "dev_metrics.json").open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        self.assertEqual(subject.class_order(metrics), sorted(CLASSES))

    def test_class_order_rejects_disagreeing_populations(self) -> None:
        with (self.source / "dev_metrics.json").open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        metrics["data_audit"]["class_counts"]["enroll"] = {"alpha": 1, "delta": 2}
        with self.assertRaises(ValueError):
            subject.class_order(metrics)

    def test_class_order_rejects_a_missing_audit(self) -> None:
        with self.assertRaises(ValueError):
            subject.class_order({"data_audit": {}})

    def test_cross_check_accepts_an_agreeing_corpus_manifest(self) -> None:
        path = self.root / "corpus.json"
        _write(path, {"classes": list(reversed(CLASSES))})
        subject._cross_check_class_order(sorted(CLASSES), path)

    def test_cross_check_rejects_a_disagreeing_corpus_manifest(self) -> None:
        path = self.root / "corpus.json"
        _write(path, {"classes": ["am", "fm"]})
        with self.assertRaises(RuntimeError):
            subject._cross_check_class_order(sorted(CLASSES), path)

    def test_cross_check_is_skipped_when_the_manifest_is_absent(self) -> None:
        subject._cross_check_class_order(
            sorted(CLASSES), self.root / "does-not-exist.json"
        )

    def test_cross_check_reads_only_the_manifest_json(self) -> None:
        self.assertEqual(subject.CORPUS_MANIFEST.name, "corpus.json")
        self.assertNotIn("corpus.f32", str(subject.CORPUS_MANIFEST))

    def test_export_refuses_an_unexpected_class_set(self) -> None:
        def mutate(metrics: dict) -> None:
            metrics["data_audit"]["class_counts"] = {
                population: {"alpha": 3, "beta": 4}
                for population in ("train", "enroll", "selection")
            }

        _rewrite_metrics(self.source, mutate)
        with self.assertRaises((ValueError, RuntimeError)):
            _export(self.source, self.output)

    def test_export_refuses_a_prototype_class_count_mismatch(self) -> None:
        path = self.source / "fusion_prototypes.npy"
        np.save(path, np.zeros((N_CLASSES - 1, 2 * EMBED_DIM), dtype=np.float32))
        # Re-bless the hash so the shape guard, not the integrity guard, fires.
        _rewrite_metrics(
            self.source,
            lambda metrics: metrics["artifacts"].__setitem__(
                "prototypes_sha256", _sha256_bytes(path.read_bytes())
            ),
        )
        with self.assertRaises(ValueError):
            _export(self.source, self.output)


class BundleContentTests(_Fixture):
    def setUp(self) -> None:
        super().setUp()
        self.manifest = _export(self.source, self.output)

    def test_carries_both_branch_weights(self) -> None:
        fusion = self.manifest["fusion"]
        self.assertEqual(fusion["weight_real"], 0.5)
        self.assertEqual(fusion["weight_complex"], 0.5)
        self.assertAlmostEqual(
            fusion["weight_real"] + fusion["weight_complex"], 1.0, places=12
        )

    def test_carries_train_only_centers(self) -> None:
        centers = self.manifest["centers"]
        self.assertEqual(centers["fit_population"], "training only")
        for key in ("real_asset", "complex_asset"):
            self.assertTrue((self.output / centers[key]).is_file())
        real = np.load(self.output / centers["real_asset"], allow_pickle=False)
        self.assertEqual(real.shape, (EMBED_DIM,))

    def test_carries_feature_standardization(self) -> None:
        standardization = self.manifest["feature_standardization"]
        self.assertEqual(
            standardization["fit_population"], "training raw features only"
        )
        self.assertEqual(standardization["feature_count"], N_FEATURES)
        std = np.load(
            self.output / standardization["std_asset"], allow_pickle=False
        )
        self.assertTrue(np.all(std > 0.0))

    def test_carries_enrollment_prototypes_and_class_order(self) -> None:
        classification = self.manifest["classification"]
        self.assertEqual(classification["classes"], sorted(CLASSES))
        self.assertEqual(classification["prototype_population"], "enrollment only")
        prototypes = np.load(
            self.output / classification["prototype_asset"], allow_pickle=False
        )
        self.assertEqual(
            prototypes.shape, (len(classification["classes"]), 2 * EMBED_DIM)
        )

    def test_every_declared_asset_hash_matches_the_written_file(self) -> None:
        for name, record in self.manifest["assets"].items():
            path = self.output / name
            self.assertTrue(path.is_file(), name)
            self.assertEqual(record["sha256"], _sha256_bytes(path.read_bytes()))
            self.assertEqual(record["bytes"], path.stat().st_size)

    def test_no_npz_container_is_written(self) -> None:
        self.assertEqual(list(self.output.rglob("*.npz")), [])
        self.assertIs(self.manifest["determinism"]["npz_containers_used"], False)

    def test_manifest_records_no_timestamp(self) -> None:
        # The determinism block names the property it disclaims, so exclude it
        # from the token scan rather than weakening the scan.
        without_disclaimer = {
            key: value
            for key, value in self.manifest.items()
            if key != "determinism"
        }
        serialized = json.dumps(without_disclaimer).lower()
        for token in (
            "timestamp",
            "wall_clock",
            "mtime",
            "created_at",
            "generated_at",
            "exported_at",
        ):
            self.assertNotIn(token, serialized)
        self.assertIs(self.manifest["determinism"]["timestamps_recorded"], False)

    def test_manifest_json_is_sorted_and_reparses(self) -> None:
        text = (self.output / subject.MANIFEST_NAME).read_text(encoding="utf-8")
        parsed = json.loads(text)
        self.assertEqual(
            text,
            json.dumps(parsed, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )

    def test_bundle_is_marked_development_only(self) -> None:
        self.assertIs(self.manifest["development_only"], True)
        self.assertIs(self.manifest["release_evidence"], False)
        self.assertEqual(self.manifest["sealed_release_data_used"], 0)
        self.assertEqual(self.manifest["consumed_test_rows_used"], 0)
        self.assertIs(self.manifest["provenance"]["corpus_or_split_loaded"], False)
        self.assertTrue(self.manifest["release_blockers"])

    def test_provenance_path_is_host_independent(self) -> None:
        recorded = self.manifest["provenance"]["source_fusion_artifact"]
        self.assertFalse(Path(recorded).is_absolute())
        self.assertNotIn(str(REPO), json.dumps(self.manifest))

    def test_branch_provenance_paths_are_repo_relative(self) -> None:
        branches = self.manifest["provenance"]["source_branches"]
        self.assertEqual(sorted(branches), ["complex", "real"])
        for record in branches.values():
            self.assertFalse(Path(record["directory"]).is_absolute())
            self.assertTrue(
                record["directory"].startswith("training/zplane_ab/"),
                record["directory"],
            )
            self.assertIn("sha256", record)


class RejectionSlotTests(_Fixture):
    def test_slot_exists_and_is_explicitly_unset(self) -> None:
        manifest = _export(self.source, self.output)
        rejection = manifest["rejection"]
        self.assertIn("rejection", manifest)
        self.assertEqual(rejection["state"], "unset")
        self.assertIs(rejection["fitted"], False)
        self.assertIsNone(rejection["policy"])
        self.assertIsNone(rejection["asset_directory"])
        self.assertIn("HANDOFF 10.5", rejection["reason"])
        self.assertIs(rejection["cannot_change_closed_label"], True)
        contract = rejection["required_contract"]
        self.assertEqual(contract["v2_weight"], openset.FROZEN_V2_WEIGHT)
        self.assertEqual(
            contract["geometry_weight"], openset.FROZEN_GEOMETRY_WEIGHT
        )
        self.assertEqual(
            contract["threshold_quantile"], openset.FROZEN_THRESHOLD_QUANTILE
        )
        self.assertEqual(contract["density_fit_population"], "training only")
        self.assertEqual(
            contract["rank_and_threshold_population"], "enrollment only"
        )

    def _write_policy(self, directory: Path, dev_metrics_sha256: str) -> None:
        rng = np.random.default_rng(19)
        rows = 24
        packed = rng.normal(
            size=(rows, 2, openset.PATCH_COUNT * openset.PATCH_LENGTH)
        ).astype(np.float32)
        labels = np.arange(rows) % 2
        scores = rng.uniform(0.0, 0.999, size=rows)
        policy = openset.FrozenV3OpenSet.fit(
            packed,
            labels,
            packed,
            labels,
            scores,
            n_classes=2,
        )
        directory.mkdir(parents=True, exist_ok=True)
        for key, value in policy.to_payload().items():
            np.save(directory / f"{key}.npy", np.asarray(value))
        _write(
            directory / subject.REJECTOR_PROVENANCE_NAME,
            {
                "fitted_against_fusion_dev_metrics_sha256": dev_metrics_sha256,
                "density_fit_population": "training only",
                "rank_and_threshold_population": "enrollment only",
                "selection_or_novelty_used_in_fit": False,
                "cannot_change_closed_label": True,
                "patch_length": PATCH_LENGTH,
                "patch_count": PATCH_COUNT,
            },
        )

    def _dev_metrics_sha(self) -> str:
        return _sha256_bytes((self.source / "dev_metrics.json").read_bytes())

    def test_attached_policy_populates_the_slot(self) -> None:
        rejector = self.root / "rejector_source"
        self._write_policy(rejector, self._dev_metrics_sha())
        manifest = _export(self.source, self.output, rejector_dir=rejector)
        rejection = manifest["rejection"]
        self.assertEqual(rejection["state"], "fitted")
        self.assertIs(rejection["fitted"], True)
        self.assertEqual(rejection["asset_directory"], subject.REJECTOR_DIRNAME)
        self.assertEqual(
            rejection["policy"]["kind"], openset.FROZEN_POLICY_KIND
        )
        self.assertEqual(
            rejection["policy"]["v2_weight"], openset.FROZEN_V2_WEIGHT
        )
        self.assertTrue(
            (self.output / subject.REJECTOR_DIRNAME / "threshold.npy").is_file()
        )
        self.assertEqual(
            list((self.output / subject.REJECTOR_DIRNAME).glob("*.npz")), []
        )

    def test_attached_policy_must_be_bound_to_this_assembly(self) -> None:
        rejector = self.root / "rejector_source"
        self._write_policy(rejector, "f" * 64)
        with self.assertRaises(RuntimeError):
            _export(self.source, self.output, rejector_dir=rejector)

    def test_attached_policy_must_declare_known_only_populations(self) -> None:
        rejector = self.root / "rejector_source"
        self._write_policy(rejector, self._dev_metrics_sha())
        path = rejector / subject.REJECTOR_PROVENANCE_NAME
        with path.open(encoding="utf-8") as handle:
            provenance = json.load(handle)
        provenance["selection_or_novelty_used_in_fit"] = True
        _write(path, provenance)
        with self.assertRaises(RuntimeError):
            _export(self.source, self.output, rejector_dir=rejector)

    def test_attached_policy_must_match_the_bundle_patch_geometry(self) -> None:
        rejector = self.root / "rejector_source"
        self._write_policy(rejector, self._dev_metrics_sha())
        path = rejector / subject.REJECTOR_PROVENANCE_NAME
        with path.open(encoding="utf-8") as handle:
            provenance = json.load(handle)
        provenance["patch_length"] = PATCH_LENGTH + 1
        _write(path, provenance)
        with self.assertRaises(RuntimeError):
            _export(self.source, self.output, rejector_dir=rejector)


class SelfVerificationTests(_Fixture):
    def test_export_reports_reload_reproduction_within_tolerance(self) -> None:
        manifest = _export(self.source, self.output)
        verification = manifest["self_verification"]
        self.assertLessEqual(
            verification["worst_max_abs_error"], verification["tolerance"]
        )
        self.assertEqual(
            verification["tolerance"], subject.SELF_VERIFICATION_TOLERANCE
        )
        self.assertIs(verification["closed_label_agreement"], True)
        self.assertEqual(
            verification["probe_case_count"], len(TEST_SPECIFICATIONS)
        )
        for key in (
            "packed",
            "raw_features",
            "standardized_features",
            "real_embedding",
            "complex_embedding",
            "fused_embedding",
        ):
            self.assertIn(key, verification["max_abs_error"])

    def test_probe_fixture_records_the_same_verification(self) -> None:
        manifest = _export(self.source, self.output)
        with (self.output / subject.PROBE_NAME).open(encoding="utf-8") as handle:
            fixture = json.load(handle)
        self.assertEqual(fixture["self_verification"], manifest["self_verification"])
        self.assertEqual(len(fixture["cases"]), len(TEST_SPECIFICATIONS))
        case = fixture["cases"][0]
        self.assertEqual(
            len(case["expected"]["packed_iq"]), 2 * PATCH_LENGTH * PATCH_COUNT
        )
        self.assertEqual(
            len(case["expected"]["fused_embedding"]), 2 * EMBED_DIM
        )
        self.assertIs(fixture["data_or_corpus_loaded"], False)

    def test_verification_fails_when_the_bundle_weights_are_tampered(self) -> None:
        _export(self.source, self.output)
        with (self.output / subject.MANIFEST_NAME).open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        state_path = self.output / "fusion_state_dict.pt"
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        state["real_center"] = state["real_center"] + 7.0
        torch.save(state, state_path)
        loaded = subject.load_fusion_artifact(self.source)
        config = loaded["real_config"]
        expected = subject.probe_embeddings(
            loaded["fusion"],
            loaded["feature_mean"],
            loaded["feature_std"],
            patch_length=int(config.patch_length),
            patch_count=int(config.patch_count),
            n_features=int(config.n_features),
            target_frac=loaded["target_frac"],
            specifications=TEST_SPECIFICATIONS,
        )
        with self.assertRaises(AssertionError):
            subject.verify_bundle(
                self.output,
                {
                    **expected,
                    "prototypes": loaded["prototypes"],
                    "classes": loaded["classes"],
                },
                specifications=TEST_SPECIFICATIONS,
            )
        self.assertEqual(manifest["schema"], subject.SCHEMA_ID)

    def test_probe_captures_are_deterministic_and_data_blind(self) -> None:
        first = subject.probe_captures(TEST_SPECIFICATIONS)
        second = subject.probe_captures(TEST_SPECIFICATIONS)
        self.assertEqual(len(first), len(TEST_SPECIFICATIONS))
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
            self.assertTrue(np.isfinite(left).all())


class DeterminismTests(_Fixture):
    def test_two_exports_are_byte_identical(self) -> None:
        second = self.root / "bundle_b"
        first_manifest = _export(self.source, self.output)
        second_manifest = _export(self.source, second)
        self.assertEqual(first_manifest, second_manifest)
        left = _tree_hashes(self.output)
        right = _tree_hashes(second)
        self.assertEqual(sorted(left), sorted(right))
        self.assertEqual(left, right)
        self.assertIn(subject.MANIFEST_NAME, left)
        self.assertIn(subject.PROBE_NAME, left)

    def test_two_exports_with_a_rejector_are_byte_identical(self) -> None:
        rejector = self.root / "rejector_source"
        rng = np.random.default_rng(5)
        rows = 24
        packed = rng.normal(
            size=(rows, 2, openset.PATCH_COUNT * openset.PATCH_LENGTH)
        ).astype(np.float32)
        labels = np.arange(rows) % 2
        policy = openset.FrozenV3OpenSet.fit(
            packed,
            labels,
            packed,
            labels,
            rng.uniform(0.0, 0.999, size=rows),
            n_classes=2,
        )
        rejector.mkdir(parents=True, exist_ok=True)
        for key, value in policy.to_payload().items():
            np.save(rejector / f"{key}.npy", np.asarray(value))
        _write(
            rejector / subject.REJECTOR_PROVENANCE_NAME,
            {
                "fitted_against_fusion_dev_metrics_sha256": _sha256_bytes(
                    (self.source / "dev_metrics.json").read_bytes()
                ),
                "density_fit_population": "training only",
                "rank_and_threshold_population": "enrollment only",
                "selection_or_novelty_used_in_fit": False,
                "cannot_change_closed_label": True,
                "patch_length": PATCH_LENGTH,
                "patch_count": PATCH_COUNT,
            },
        )
        second = self.root / "bundle_b"
        _export(self.source, self.output, rejector_dir=rejector)
        _export(self.source, second, rejector_dir=rejector)
        self.assertEqual(_tree_hashes(self.output), _tree_hashes(second))

    def test_copied_assets_keep_the_assembly_hashes(self) -> None:
        manifest = _export(self.source, self.output)
        with (self.source / "dev_metrics.json").open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        for key, name in subject.COPIED_ASSETS:
            self.assertEqual(
                manifest["assets"][name]["sha256"],
                metrics["artifacts"][f"{key}_sha256"],
            )


class GuardTests(_Fixture):
    def test_refuses_a_release_output_path(self) -> None:
        with self.assertRaises(ValueError):
            _export(self.source, self.root / "artifacts" / "releases" / "x")

    def test_refuses_a_sealed_source_path(self) -> None:
        sealed = self.root / "sealed_suite_v2"
        shutil.copytree(self.source, sealed)
        with self.assertRaises(ValueError):
            _export(sealed, self.output)

    def test_refuses_a_non_empty_output_directory(self) -> None:
        self.output.mkdir(parents=True)
        (self.output / "stale.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            _export(self.source, self.output)

    def test_refuses_to_write_below_live_src(self) -> None:
        with self.assertRaises(ValueError):
            _export(self.source, REPO / "src" / "embedding" / "assets" / "v3")

    def test_refuses_to_write_inside_the_source_assembly(self) -> None:
        with self.assertRaises(ValueError):
            _export(self.source, self.source / "bundle")

    def test_refuses_an_artifact_claiming_release_evidence(self) -> None:
        _rewrite_metrics(
            self.source,
            lambda metrics: metrics.__setitem__("release_evidence", True),
        )
        with self.assertRaises(RuntimeError):
            _export(self.source, self.output)

    def test_refuses_an_artifact_that_touched_sealed_rows(self) -> None:
        _rewrite_metrics(
            self.source,
            lambda metrics: metrics.__setitem__("sealed_release_data_used", 1),
        )
        with self.assertRaises(RuntimeError):
            _export(self.source, self.output)

    def test_refuses_an_artifact_that_exposed_consumed_test_rows(self) -> None:
        _rewrite_metrics(
            self.source,
            lambda metrics: metrics["data_audit"].__setitem__(
                "consumed_test_rows_exposed", 3
            ),
        )
        with self.assertRaises(RuntimeError):
            _export(self.source, self.output)

    def test_refuses_a_relaxed_fitting_contract(self) -> None:
        _rewrite_metrics(
            self.source,
            lambda metrics: metrics["fitting_contract"].__setitem__(
                "center_fit_population", "training and enrollment"
            ),
        )
        with self.assertRaises(RuntimeError):
            _export(self.source, self.output)

    def test_refuses_a_tampered_source_asset(self) -> None:
        path = self.source / "fusion_prototypes.npy"
        value = np.load(path, allow_pickle=False)
        np.save(path, value + np.float32(1.0))
        with self.assertRaises(RuntimeError):
            _export(self.source, self.output)

    def test_refuses_a_non_fusion_artifact(self) -> None:
        _rewrite_metrics(
            self.source,
            lambda metrics: metrics.__setitem__("encoder", "real"),
        )
        with self.assertRaises(RuntimeError):
            _export(self.source, self.output)

    def test_refuses_a_target_frac_disagreement(self) -> None:
        _rewrite_metrics(
            self.source,
            lambda metrics: metrics["source_index_contract"]["contract"][
                "config"
            ].__setitem__("target_frac", 0.25),
        )
        with self.assertRaises(RuntimeError):
            _export(self.source, self.output)

    def test_refuses_a_branch_weight_that_does_not_sum_to_one(self) -> None:
        def mutate(metrics: dict) -> None:
            metrics["fusion"]["weight_complex"] = 0.4

        _rewrite_metrics(self.source, mutate)
        with self.assertRaises(ValueError):
            _export(self.source, self.output)

    def test_jsonable_refuses_an_unserializable_value(self) -> None:
        with self.assertRaises(TypeError):
            subject._jsonable({"bad": object()})

    def test_deepcopy_of_specifications_does_not_change_captures(self) -> None:
        captures = subject.probe_captures(copy.deepcopy(TEST_SPECIFICATIONS))
        reference = subject.probe_captures(TEST_SPECIFICATIONS)
        for left, right in zip(captures, reference):
            np.testing.assert_array_equal(left, right)


if __name__ == "__main__":
    unittest.main()
