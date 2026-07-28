"""Synthetic tests for the fail-closed v3 release promoter.

No repository runtime asset or sealed evaluation artifact is read by this
suite.  Every package, hash, and report is generated inside a temporary
directory.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable


MODULE_PATH = Path(__file__).with_name("promote-v3-release-assets.py")
SPEC = importlib.util.spec_from_file_location(
    "promote_v3_release_assets_under_test", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot load {MODULE_PATH}")
promoter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = promoter
SPEC.loader.exec_module(promoter)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.write_bytes(canonical_bytes(payload))


def file_record(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def historical_metadata() -> dict[str, Any]:
    return {
        "release_seed": promoter.HISTORICAL_RELAXED_SEED,
        "active_for_current_protocol": False,
        "current_protocol_gate_source": "imported_v2_gate_floors_unchanged",
        "gates_redeclared": {
            "five_shot_worst_length_balanced": {
                "v2_level": 0.85,
                "v3_level": 0.84,
                "owner_decision": "synthetic inactive historical record",
            },
            "open_known_false_unknown_worst_length": {
                "v2_level": 0.10,
                "v3_level": 0.12,
                "owner_decision": "synthetic inactive historical record",
            },
        },
    }


def release_gates(candidate_sha256: str) -> dict[str, dict[str, Any]]:
    gates: dict[str, dict[str, Any]] = {
        name: {"value": True, "expected": True, "passes": True}
        for name in promoter.BOOLEAN_GATE_NAMES
    }
    gates[promoter.CANDIDATE_GATE_NAME] = {
        "value": candidate_sha256,
        "expected": candidate_sha256,
        "passes": True,
    }
    for name, (threshold, comparison) in (
        promoter.NUMERIC_GATE_CONTRACT.items()
    ):
        gates[name] = {
            "value": threshold,
            "threshold": threshold,
            "comparison": comparison,
            "passes": True,
        }
    return gates


def candidate_hashes() -> dict[str, Any]:
    bundle_assets = {
        name: {
            "bytes": index + 1,
            "sha256": digest(f"bundle asset {name}"),
        }
        for index, name in enumerate(sorted(promoter.EXPECTED_BUNDLE_ASSETS))
    }
    staged = {
        name: digest(f"staged asset {name}")
        for name in sorted(promoter.EXPECTED_STAGED_ASSETS)
    }
    return {
        "bundle_manifest": digest("runtime bundle manifest"),
        "bundle_assets": bundle_assets,
        "fusion_directory": digest("fusion directory"),
        "fusion_files": {
            "dev_metrics.json": digest("fusion dev metrics"),
            "selected_fusion_state.pt": bundle_assets[
                "fusion_state_dict.pt"
            ]["sha256"],
        },
        "fusion_seed": 20260730,
        "staged": staged,
        "staged_validation": digest("staged validation report"),
        "prefilter": digest("prefilter set"),
        "release_intent": digest("release intent"),
        "release_manifest": digest("release manifest"),
    }


def staging_payloads(hashes: dict[str, Any]) -> dict[str, dict[str, Any]]:
    common_provenance = {
        "runtime_bundle": "synthetic-runtime-bundle",
        "runtime_bundle_manifest_sha256": hashes["bundle_manifest"],
        "fusion_artifact": "synthetic-fusion",
        "fusion_directory_sha256": hashes["fusion_directory"],
        "staged_artifact": "synthetic-staged-validation",
        "staged_artifact_sha256": hashes["staged"],
        "staged_validation_report_sha256": hashes["staged_validation"],
        "staged_validation_status": promoter.STAGED_STATUS,
        "staged_validation_role": "validate",
        "staged_validation_all_pass": True,
        "prefilter_bundles": "synthetic-prefilters/bundles",
        "prefilter_set_sha256": hashes["prefilter"],
        "development_only": True,
        "release_evidence": False,
        "sealed_release_data_used": 0,
        "consumed_test_rows_used": 0,
        "release_seed_not_spent": promoter.RELEASE_SEED,
    }
    fusion = {
        "schema": promoter.ASSET_CONTRACTS[promoter.FUSION_WEIGHTS][0],
        "schema_version": promoter.ASSET_CONTRACTS[promoter.FUSION_WEIGHTS][1],
        "status": promoter.STAGING_STATUS,
        "development_only": True,
        "release_evidence": False,
        "provenance": {
            "source_bundle_schema": promoter.RUNTIME_BUNDLE_SCHEMA,
            "source_bundle_assets_sha256": {
                name: record["sha256"]
                for name, record in hashes["bundle_assets"].items()
            },
            "assembly_seed": hashes["fusion_seed"],
            "exporter": "synthetic",
            "batch_norm_folded": False,
        },
        "weights": [0.25, 0.75],
    }
    classifier = {
        "schema": promoter.ASSET_CONTRACTS[promoter.CLASSIFIER_WEIGHTS][0],
        "schema_version": promoter.ASSET_CONTRACTS[
            promoter.CLASSIFIER_WEIGHTS
        ][1],
        "status": promoter.STAGING_STATUS,
        "provenance": copy.deepcopy(common_provenance),
        "classification": {"classes": ["a", "b"]},
    }
    openset = {
        "schema": promoter.ASSET_CONTRACTS[promoter.OPENSET_WEIGHTS][0],
        "schema_version": promoter.ASSET_CONTRACTS[
            promoter.OPENSET_WEIGHTS
        ][1],
        "status": promoter.STAGING_STATUS,
        "provenance": copy.deepcopy(common_provenance),
        "contract": {
            "additive_only": False,
            "changes_closed_label": True,
            "gates_before_classification": True,
        },
        "composite": {
            "schema": promoter.STAGED_POLICY_SCHEMA,
            "kind": promoter.STAGED_POLICY_KIND,
            "policy_version": promoter.STAGED_POLICY_VERSION,
        },
    }
    return {
        promoter.FUSION_WEIGHTS: fusion,
        promoter.CLASSIFIER_WEIGHTS: classifier,
        promoter.OPENSET_WEIGHTS: openset,
    }


def refresh_staging_manifests(staging: Path) -> None:
    export_manifest = {
        "schema": promoter.FUSION_EXPORT_SCHEMA,
        "schema_version": 1,
        "emitted": {
            name: file_record(staging / name)
            for name in (promoter.FUSION_WEIGHTS, promoter.FUSION_PROBE)
        },
    }
    write_json(staging / promoter.FUSION_EXPORT_MANIFEST, export_manifest)

    manifest_path = staging / promoter.PACKAGE_MANIFEST
    external_sources = {
        "openset-export-manifest.json": digest("open-set export manifest"),
        "time-domain-openset-parity-v1.json": digest("open-set parity"),
    }
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name in external_sources:
            external_sources[name] = old["sources"].get(
                name, external_sources[name]
            )
    manifest = {
        "schema": promoter.PACKAGE_SCHEMA,
        "schema_version": 1,
        "status": promoter.STAGING_STATUS,
        "assets": {
            name: file_record(staging / name) for name in promoter.ASSET_NAMES
        },
        "sources": {
            promoter.FUSION_EXPORT_MANIFEST: file_record(
                staging / promoter.FUSION_EXPORT_MANIFEST
            )["sha256"],
            **external_sources,
        },
    }
    write_json(manifest_path, manifest)


def build_staging(staging: Path, hashes: dict[str, Any]) -> None:
    staging.mkdir(parents=True)
    payloads = staging_payloads(hashes)
    for name, payload in payloads.items():
        write_json(staging / name, payload)
    write_json(
        staging / promoter.OPENSET_SMOKE,
        {
            "schema": "synthetic-runtime-smoke",
            "status": promoter.STAGING_STATUS,
            "rows": [],
        },
    )
    write_json(
        staging / promoter.FUSION_PROBE,
        {
            "schema": "synthetic-probe",
            "status": promoter.STAGING_STATUS,
            "cases": [],
        },
    )
    refresh_staging_manifests(staging)


def evaluation_report(hashes: dict[str, Any]) -> dict[str, Any]:
    history = historical_metadata()
    stage_one_lengths = [4096, 8192]
    protocol = {
        "version": promoter.EVALUATION_VERSION,
        "gates": copy.deepcopy(promoter.STRICT_V2_GATE_FLOORS),
        "novelty": {"seed": promoter.RELEASE_SEED},
        "historical_gate_redeclaration": copy.deepcopy(history),
        "open_set": {
            "architecture": promoter.STAGED_POLICY_KIND,
            "staged_policy_schema": promoter.STAGED_POLICY_SCHEMA,
            "staged_policy_version": promoter.STAGED_POLICY_VERSION,
            "additive_only": False,
            "changes_closed_label": True,
            "gates_before_classification": True,
            "stage_one_capture_lengths": stage_one_lengths,
        },
    }
    return {
        "schema": promoter.EVALUATOR_SCHEMA,
        "status": "complete",
        "release_evidence": True,
        "evaluation_version": promoter.EVALUATION_VERSION,
        "development_data_loaded": False,
        "retraining_performed": False,
        "recalibration_performed": False,
        "architecture_contract": {
            "additive_only": False,
            "changes_closed_label": True,
            "gates_before_classification": True,
            "staged_policy_schema": promoter.STAGED_POLICY_SCHEMA,
            "staged_policy_version": promoter.STAGED_POLICY_VERSION,
            "staged_policy_kind": promoter.STAGED_POLICY_KIND,
            "stage_one_capture_lengths": stage_one_lengths,
        },
        "candidate": {
            "path": "synthetic/bundle_manifest.json",
            "sha256": hashes["bundle_manifest"],
            "runtime_schema": promoter.RUNTIME_BUNDLE_SCHEMA,
            "runtime_schema_version": promoter.RUNTIME_BUNDLE_SCHEMA_VERSION,
            "runtime_kind": promoter.RUNTIME_BUNDLE_KIND,
            "classes": ["a", "b"],
            "frozen_assets_used": ["synthetic frozen candidate"],
            "components": {
                "bundle_dir": "synthetic/runtime-bundle",
                "bundle_manifest_sha256": hashes["bundle_manifest"],
                "bundle_assets": copy.deepcopy(hashes["bundle_assets"]),
                "fusion_dir": "synthetic/fusion",
                "fusion_directory_sha256": hashes["fusion_directory"],
                "fusion_file_sha256": copy.deepcopy(hashes["fusion_files"]),
                "fusion_seed": hashes["fusion_seed"],
                "staged_dir": "synthetic/staged",
                "staged_artifact_sha256": copy.deepcopy(hashes["staged"]),
                "staged_status": promoter.STAGED_STATUS,
                "staged_novelty_seeds": [20260950, 20260951],
                "prefilter_dir": "synthetic/prefilter",
                "prefilter_set_sha256": hashes["prefilter"],
                "stage_one_capture_lengths": stage_one_lengths,
                "staged_policy_version": promoter.STAGED_POLICY_VERSION,
                "staged_threshold": 0.95,
                "stage_two_threshold": 0.95,
                "composite": {"synthetic": True},
                "source_sha256": {"enforced": {}},
            },
        },
        "gates": release_gates(hashes["bundle_manifest"]),
        "historical_gate_redeclaration": copy.deepcopy(history),
        "all_release_gates_pass": True,
        "provenance": {
            "release_seed": promoter.RELEASE_SEED,
            "evaluation_protocol": protocol,
            "candidate_sha256": hashes["bundle_manifest"],
            "release_intent_sha256": hashes["release_intent"],
            "release_manifest_sha256": hashes["release_manifest"],
        },
    }


class SyntheticFixture:
    def __init__(
        self,
        root: Path,
        *,
        destination_name: str = "assets-v3-release",
        require_git_tracking: bool = False,
    ) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.staging = root / "assets-v3-staging"
        self.destination = root / destination_name
        self.live_v2 = root / "assets"
        self.hashes = candidate_hashes()
        build_staging(self.staging, self.hashes)
        self.report = evaluation_report(self.hashes)
        self.report_path = root / "RELEASE_EVALUATION.json"
        write_json(self.report_path, self.report)
        self.policy = promoter.PromotionPolicy(
            repo_root=root,
            staging_package=self.staging,
            release_package=self.destination,
            live_v2_assets=self.live_v2,
            require_git_tracking=require_git_tracking,
        )

    def write_report(self, report: dict[str, Any]) -> None:
        write_json(self.report_path, report)

    def promote(self) -> dict[str, Any]:
        return promoter.promote(
            self.staging,
            self.report_path,
            self.destination,
            policy=self.policy,
        )


def mutate_path(root: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cursor: dict[str, Any] = root
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value


class PromotionTests(unittest.TestCase):
    def test_promotes_exact_three_statuses_and_binds_source_and_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticFixture(Path(temporary))
            fixture.destination.mkdir()  # Existing empty destinations are safe.
            source_fixture_bytes = {
                name: (fixture.staging / name).read_bytes()
                for name in (promoter.OPENSET_SMOKE, promoter.FUSION_PROBE)
            }
            source_status_bytes = {
                name: (fixture.staging / name).read_bytes()
                for name in promoter.STATUS_ASSET_NAMES
            }
            source_manifest_raw = (
                fixture.staging / promoter.PACKAGE_MANIFEST
            ).read_bytes()
            report_raw = fixture.report_path.read_bytes()

            manifest = fixture.promote()

            self.assertEqual(manifest["status"], promoter.RELEASE_STATUS)
            self.assertEqual(set(manifest["assets"]), set(promoter.ASSET_NAMES))
            for name in promoter.STATUS_ASSET_NAMES:
                payload = json.loads(
                    (fixture.destination / name).read_text(encoding="utf-8")
                )
                self.assertEqual(payload["status"], promoter.RELEASE_STATUS)
                source_payload = json.loads(source_status_bytes[name])
                payload["status"] = promoter.STAGING_STATUS
                self.assertEqual(payload, source_payload)
            for name, expected in source_fixture_bytes.items():
                self.assertEqual((fixture.destination / name).read_bytes(), expected)

            promotion = manifest["promotion"]
            self.assertEqual(
                promotion["source"]["package_manifest_sha256"],
                hashlib.sha256(source_manifest_raw).hexdigest(),
            )
            self.assertEqual(
                promotion["evaluation"]["report_sha256"],
                hashlib.sha256(report_raw).hexdigest(),
            )
            self.assertEqual(
                promotion["evaluation"]["gate_contract"],
                "strict_imported_v2_23_of_23",
            )
            self.assertEqual(promotion["evaluation"]["five_shot_minimum"], 0.85)
            self.assertEqual(
                promotion["evaluation"]["known_false_unknown_maximum"], 0.10
            )
            self.assertEqual(
                promotion["candidate_artifacts"][
                    "runtime_bundle_manifest_sha256"
                ],
                fixture.hashes["bundle_manifest"],
            )
            self.assertFalse(
                promotion["operation"]["sealed_corpus_opened_by_promoter"]
            )

            # Promotion is non-mutating with respect to the verified source.
            for name, expected in source_status_bytes.items():
                self.assertEqual((fixture.staging / name).read_bytes(), expected)

    def test_identical_inputs_produce_byte_identical_packages(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticFixture(Path(temporary), destination_name="release-a")
            fixture.promote()
            destination_b = fixture.root / "release-b"
            policy_b = promoter.PromotionPolicy(
                repo_root=fixture.root,
                staging_package=fixture.staging,
                release_package=destination_b,
                live_v2_assets=fixture.live_v2,
                require_git_tracking=False,
            )
            promoter.promote(
                fixture.staging,
                fixture.report_path,
                destination_b,
                policy=policy_b,
            )
            names_a = sorted(path.name for path in fixture.destination.iterdir())
            names_b = sorted(path.name for path in destination_b.iterdir())
            self.assertEqual(names_a, names_b)
            for name in names_a:
                self.assertEqual(
                    (fixture.destination / name).read_bytes(),
                    (destination_b / name).read_bytes(),
                    name,
                )

    def test_report_contract_failures_are_refused_before_writes(self):
        mutations: list[
            tuple[str, Callable[[dict[str, Any]], None], str]
        ] = [
            (
                "status",
                lambda report: mutate_path(report, ("status",), "failed"),
                "status must be complete",
            ),
            (
                "release evidence",
                lambda report: mutate_path(
                    report, ("release_evidence",), False
                ),
                "not release evidence",
            ),
            (
                "development leakage",
                lambda report: mutate_path(
                    report, ("development_data_loaded",), True
                ),
                "development_data_loaded",
            ),
            (
                "training",
                lambda report: mutate_path(
                    report, ("retraining_performed",), True
                ),
                "retraining_performed",
            ),
            (
                "recalibration",
                lambda report: mutate_path(
                    report, ("recalibration_performed",), True
                ),
                "recalibration_performed",
            ),
            (
                "aggregate pass",
                lambda report: mutate_path(
                    report, ("all_release_gates_pass",), False
                ),
                "does not pass every",
            ),
            (
                "wrong seed",
                lambda report: mutate_path(
                    report, ("provenance", "release_seed"), 20260736
                ),
                "must bind untouched seed",
            ),
            (
                "wrong novelty seed",
                lambda report: mutate_path(
                    report,
                    ("provenance", "evaluation_protocol", "novelty", "seed"),
                    20260736,
                ),
                "novelty protocol",
            ),
            (
                "relaxed embedded five-shot floor",
                lambda report: mutate_path(
                    report,
                    (
                        "provenance",
                        "evaluation_protocol",
                        "gates",
                        "five_shot",
                    ),
                    0.84,
                ),
                "strict imported v2 floors",
            ),
            (
                "active historical relaxation",
                lambda report: (
                    mutate_path(
                        report,
                        (
                            "historical_gate_redeclaration",
                            "active_for_current_protocol",
                        ),
                        True,
                    ),
                    mutate_path(
                        report,
                        (
                            "provenance",
                            "evaluation_protocol",
                            "historical_gate_redeclaration",
                            "active_for_current_protocol",
                        ),
                        True,
                    ),
                ),
                "explicitly inactive",
            ),
            (
                "missing gate",
                lambda report: report["gates"].pop("closed_fine_worst_length"),
                "exactly the 23",
            ),
            (
                "false gate",
                lambda report: mutate_path(
                    report,
                    ("gates", "closed_fine_worst_length", "passes"),
                    False,
                ),
                "strict v2 contract",
            ),
            (
                "relaxed five-shot gate",
                lambda report: (
                    mutate_path(
                        report,
                        (
                            "gates",
                            "five_shot_worst_length_balanced",
                            "threshold",
                        ),
                        0.84,
                    ),
                    mutate_path(
                        report,
                        (
                            "gates",
                            "five_shot_worst_length_balanced",
                            "value",
                        ),
                        0.84,
                    ),
                ),
                "strict v2 contract",
            ),
            (
                "relaxed false-unknown gate",
                lambda report: (
                    mutate_path(
                        report,
                        (
                            "gates",
                            "open_known_false_unknown_worst_length",
                            "threshold",
                        ),
                        0.12,
                    ),
                    mutate_path(
                        report,
                        (
                            "gates",
                            "open_known_false_unknown_worst_length",
                            "value",
                        ),
                        0.12,
                    ),
                ),
                "strict v2 contract",
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticFixture(Path(temporary))
            for name, mutation, message in mutations:
                with self.subTest(name=name):
                    report = copy.deepcopy(fixture.report)
                    mutation(report)
                    fixture.write_report(report)
                    with self.assertRaisesRegex(promoter.PromotionError, message):
                        fixture.promote()
                    self.assertFalse(fixture.destination.exists())

    def test_exact_candidate_hash_mismatches_are_refused(self):
        mutations: list[
            tuple[str, Callable[[dict[str, Any]], None], str]
        ] = [
            (
                "bundle asset",
                lambda report: mutate_path(
                    report,
                    (
                        "candidate",
                        "components",
                        "bundle_assets",
                        "real_center.npy",
                        "sha256",
                    ),
                    digest("different real center"),
                ),
                "fusion asset is from different runtime-bundle bytes",
            ),
            (
                "fusion directory",
                lambda report: mutate_path(
                    report,
                    (
                        "candidate",
                        "components",
                        "fusion_directory_sha256",
                    ),
                    digest("different fusion"),
                ),
                "different fusion artifact",
            ),
            (
                "fusion seed",
                lambda report: mutate_path(
                    report,
                    ("candidate", "components", "fusion_seed"),
                    20260732,
                ),
                "disagree on fusion seed",
            ),
            (
                "staged artifact",
                lambda report: mutate_path(
                    report,
                    (
                        "candidate",
                        "components",
                        "staged_artifact_sha256",
                        "v3_branch_lof_components.npz",
                    ),
                    digest("different staged artifact"),
                ),
                "different staged artifact",
            ),
            (
                "prefilter set",
                lambda report: mutate_path(
                    report,
                    (
                        "candidate",
                        "components",
                        "prefilter_set_sha256",
                    ),
                    digest("different prefilter"),
                ),
                "different prefilter set",
            ),
            (
                "staged policy",
                lambda report: mutate_path(
                    report,
                    ("candidate", "components", "staged_policy_version"),
                    "obsolete-policy",
                ),
                "not v2 composite",
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticFixture(Path(temporary))
            for name, mutation, message in mutations:
                with self.subTest(name=name):
                    report = copy.deepcopy(fixture.report)
                    mutation(report)
                    fixture.write_report(report)
                    with self.assertRaisesRegex(promoter.PromotionError, message):
                        fixture.promote()
                    self.assertFalse(fixture.destination.exists())

    def test_staging_tamper_and_pre_release_leakage_are_refused(self):
        with self.subTest("unmanifested byte tamper"):
            with tempfile.TemporaryDirectory() as temporary:
                fixture = SyntheticFixture(Path(temporary))
                with (fixture.staging / promoter.OPENSET_SMOKE).open("ab") as handle:
                    handle.write(b" ")
                with self.assertRaisesRegex(
                    promoter.PromotionError, "byte count changed"
                ):
                    fixture.promote()

        for key, value, message in (
            ("consumed_test_rows_used", 1, "consumed_test_rows_used"),
            ("sealed_release_data_used", 1, "sealed_release_data_used"),
            ("release_seed_not_spent", 20260731, "release_seed_not_spent"),
            ("staged_validation_all_pass", False, "staged_validation_all_pass"),
        ):
            with self.subTest(key=key):
                with tempfile.TemporaryDirectory() as temporary:
                    fixture = SyntheticFixture(Path(temporary))
                    for name in (
                        promoter.CLASSIFIER_WEIGHTS,
                        promoter.OPENSET_WEIGHTS,
                    ):
                        path = fixture.staging / name
                        payload = json.loads(path.read_text(encoding="utf-8"))
                        payload["provenance"][key] = value
                        write_json(path, payload)
                    refresh_staging_manifests(fixture.staging)
                    with self.assertRaisesRegex(
                        promoter.PromotionError, message
                    ):
                        fixture.promote()
                    self.assertFalse(fixture.destination.exists())

    def test_destination_and_source_scope_are_fail_closed(self):
        with self.subTest("nonempty destination"):
            with tempfile.TemporaryDirectory() as temporary:
                fixture = SyntheticFixture(Path(temporary))
                fixture.destination.mkdir()
                (fixture.destination / "keep.txt").write_text(
                    "do not overwrite", encoding="utf-8"
                )
                with self.assertRaisesRegex(
                    promoter.PromotionError, "must be empty"
                ):
                    fixture.promote()
                self.assertEqual(
                    (fixture.destination / "keep.txt").read_text(encoding="utf-8"),
                    "do not overwrite",
                )

        with self.subTest("live v2 destination"):
            with tempfile.TemporaryDirectory() as temporary:
                fixture = SyntheticFixture(Path(temporary))
                policy = promoter.PromotionPolicy(
                    repo_root=fixture.root,
                    staging_package=fixture.staging,
                    release_package=fixture.live_v2,
                    live_v2_assets=fixture.live_v2,
                    require_git_tracking=False,
                )
                with self.assertRaisesRegex(
                    promoter.PromotionError, "live v2"
                ):
                    promoter.promote(
                        fixture.staging,
                        fixture.report_path,
                        fixture.live_v2,
                        policy=policy,
                    )
                self.assertFalse(fixture.live_v2.exists())

        with self.subTest("staging destination"):
            with tempfile.TemporaryDirectory() as temporary:
                fixture = SyntheticFixture(Path(temporary))
                policy = promoter.PromotionPolicy(
                    repo_root=fixture.root,
                    staging_package=fixture.staging,
                    release_package=fixture.staging,
                    live_v2_assets=fixture.live_v2,
                    require_git_tracking=False,
                )
                with self.assertRaisesRegex(
                    promoter.PromotionError, "staging package"
                ):
                    promoter.promote(
                        fixture.staging,
                        fixture.report_path,
                        fixture.staging,
                        policy=policy,
                    )

        with self.subTest("different source"):
            with tempfile.TemporaryDirectory() as temporary:
                fixture = SyntheticFixture(Path(temporary))
                other = fixture.root / "other-staging"
                other.mkdir()
                with self.assertRaisesRegex(
                    promoter.PromotionError, "not the policy"
                ):
                    promoter.promote(
                        other,
                        fixture.report_path,
                        fixture.destination,
                        policy=fixture.policy,
                    )

    def test_duplicate_json_keys_are_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticFixture(Path(temporary))
            fixture.report_path.write_text(
                '{"schema":2,"schema":2}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(promoter.PromotionError, "repeats key"):
                fixture.promote()
            self.assertFalse(fixture.destination.exists())

    def test_git_tracking_is_required_and_committed_bytes_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticFixture(
                Path(temporary), require_git_tracking=True
            )
            subprocess.run(
                ["git", "init", "-q", str(fixture.root)],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "-C", str(fixture.root), "add", "."],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(fixture.root),
                    "-c",
                    "user.name=Synthetic Test",
                    "-c",
                    "user.email=synthetic@example.invalid",
                    "commit",
                    "-qm",
                    "synthetic tracked staging package",
                ],
                check=True,
                capture_output=True,
            )
            manifest = fixture.promote()
            self.assertEqual(manifest["status"], promoter.RELEASE_STATUS)

    def test_git_dirty_staging_manifest_is_refused_even_if_semantically_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticFixture(
                Path(temporary), require_git_tracking=True
            )
            subprocess.run(
                ["git", "init", "-q", str(fixture.root)],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "-C", str(fixture.root), "add", "."],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(fixture.root),
                    "-c",
                    "user.name=Synthetic Test",
                    "-c",
                    "user.email=synthetic@example.invalid",
                    "commit",
                    "-qm",
                    "synthetic tracked staging package",
                ],
                check=True,
                capture_output=True,
            )
            manifest_path = fixture.staging / promoter.PACKAGE_MANIFEST
            manifest_path.write_text(
                manifest_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                promoter.PromotionError, "differ from their committed HEAD"
            ):
                fixture.promote()
            self.assertFalse(fixture.destination.exists())


if __name__ == "__main__":
    unittest.main()
