"""Fail-closed tests for the deployable dual-fusion staging packager."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("package-v3-staging-assets.py")
SPEC = importlib.util.spec_from_file_location("package_v3_staging_assets", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
packager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packager)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def record(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": packager.sha256(path)}


class SyntheticInputs:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.rejector = root / "rejector-export"
        self.classifier = root / "classifier-export"
        self.openset = root / "openset-export"
        self.destination = root / "dual-package"
        for directory in (self.rejector, self.classifier, self.openset):
            directory.mkdir(parents=True)
        self.report_sha = digest("validation report")
        self.staged = {
            "v3_branch_lof_components.npz": digest("lof"),
            "v3_open_policy_stage_two.npz": digest("stage two"),
            "v3_staged_composite_policy.npz": digest("composite"),
        }
        self.build_role(
            self.rejector,
            role=packager.REJECTOR_ROLE,
            weights_name=packager.REJECTOR_WEIGHTS,
            probe_name=packager.REJECTOR_PROBE,
            bundle_sha=digest("rejector bundle"),
        )
        self.build_role(
            self.classifier,
            role=packager.CLASSIFIER_ROLE,
            weights_name=packager.CLASSIFIER_WEIGHTS,
            probe_name=packager.CLASSIFIER_PROBE,
            bundle_sha=digest("classifier bundle"),
        )
        self.build_openset()

    def build_role(
        self,
        directory: Path,
        *,
        role: str,
        weights_name: str,
        probe_name: str,
        bundle_sha: str,
    ) -> None:
        write_json(
            directory / weights_name,
            {
                "schema": packager.FUSION_WEIGHTS_SCHEMA,
                "schema_version": 1,
                "status": packager.STAGING_STATUS,
                "runtime_role": role,
                "frontend": {
                    "version": "invariant-patch-time-domain-v1",
                    "patch_length": 64,
                    "patch_count": 16,
                    "target_frac": 0.5,
                    "uses_frequency_transform": False,
                },
                "classification": {"classes": ["a", "b"]},
                "provenance": {
                    "source_bundle_manifest_sha256": bundle_sha,
                },
            },
        )
        write_json(
            directory / probe_name,
            {"schema": "synthetic-probe", "cases": []},
        )
        manifest = {
            "schema": packager.FUSION_EXPORT_SCHEMA,
            "schema_version": packager.FUSION_EXPORT_SCHEMA_VERSION,
            "runtime_role": role,
            "source_bundle_manifest_sha256": bundle_sha,
            "emitted": {
                weights_name: record(directory / weights_name),
                probe_name: record(directory / probe_name),
            },
        }
        write_json(directory / packager.FUSION_MANIFEST, manifest)

    def role_state(self, directory: Path, weights: str, probe: str) -> dict:
        manifest_path = directory / packager.FUSION_MANIFEST
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {
            "manifest": manifest,
            "manifest_record": record(manifest_path),
            "weights_record": record(directory / weights),
            "probe_record": record(directory / probe),
        }

    def build_openset(self) -> None:
        rejector = self.role_state(
            self.rejector, packager.REJECTOR_WEIGHTS, packager.REJECTOR_PROBE
        )
        classifier = self.role_state(
            self.classifier,
            packager.CLASSIFIER_WEIGHTS,
            packager.CLASSIFIER_PROBE,
        )
        rejector_bundle = rejector["manifest"][
            "source_bundle_manifest_sha256"
        ]
        classifier_bundle = classifier["manifest"][
            "source_bundle_manifest_sha256"
        ]
        rejector_fusion = digest("rejector fusion")
        classifier_fusion = digest("classifier fusion")
        policy = {
            "schema": packager.OPENSET_SCHEMA,
            "schema_version": 2,
            "status": packager.STAGING_STATUS,
            "provenance": {
                "candidate_id": packager.CANDIDATE_ID,
                "runtime_roles": {
                    "rejector": {
                        "runtime_role": packager.REJECTOR_ROLE,
                        "browser_asset": packager.REJECTOR_WEIGHTS,
                        "browser_asset_sha256": rejector["weights_record"][
                            "sha256"
                        ],
                        "browser_export_manifest_sha256": rejector[
                            "manifest_record"
                        ]["sha256"],
                        "runtime_bundle_manifest_sha256": rejector_bundle,
                        "fusion_directory_sha256": rejector_fusion,
                    },
                    "classifier": {
                        "runtime_role": packager.CLASSIFIER_ROLE,
                        "browser_asset": packager.CLASSIFIER_WEIGHTS,
                        "browser_asset_sha256": classifier["weights_record"][
                            "sha256"
                        ],
                        "browser_export_manifest_sha256": classifier[
                            "manifest_record"
                        ]["sha256"],
                        "runtime_bundle_manifest_sha256": classifier_bundle,
                        "fusion_directory_sha256": classifier_fusion,
                    },
                },
                "staged_validation_report_sha256": self.report_sha,
                "staged_artifact_sha256": self.staged,
            },
        }
        write_json(self.openset / packager.OPENSET_POLICY, policy)
        policy_record = record(self.openset / packager.OPENSET_POLICY)
        binding = {
            "schema": packager.DUAL_BINDING_SCHEMA,
            "schema_version": 1,
            "status": packager.STAGING_STATUS,
            "candidate_id": packager.CANDIDATE_ID,
            "frontend": {
                "version": "invariant-patch-time-domain-v1",
                "patch_length": 64,
                "patch_count": 16,
                "target_frac": 0.5,
                "packed_length": 1024,
                "uses_frequency_transform": False,
            },
            "execution_order": [
                "stage_one_noise_gate",
                "rejector_known_unknown",
                "classifier_known_label",
            ],
            "roles": {
                "rejector": {
                    "asset": packager.REJECTOR_WEIGHTS,
                    "asset_sha256": rejector["weights_record"]["sha256"],
                    "fusion_directory_sha256": rejector_fusion,
                    "runtime_bundle_manifest_sha256": rejector_bundle,
                    "runtime_role": packager.REJECTOR_ROLE,
                    "responsibility": packager.REJECTOR_RESPONSIBILITY,
                },
                "classifier": {
                    "asset": packager.CLASSIFIER_WEIGHTS,
                    "asset_sha256": classifier["weights_record"]["sha256"],
                    "fusion_directory_sha256": classifier_fusion,
                    "runtime_bundle_manifest_sha256": classifier_bundle,
                    "runtime_role": packager.CLASSIFIER_ROLE,
                    "responsibility": packager.CLASSIFIER_RESPONSIBILITY,
                },
            },
            "openset_policy": {
                "asset": packager.OPENSET_POLICY,
                "asset_sha256": policy_record["sha256"],
                "rejector_asset_sha256": rejector["weights_record"]["sha256"],
                "fitted_rejector_runtime_bundle_manifest_sha256": (
                    rejector_bundle
                ),
                "staged_validation_report_sha256": self.report_sha,
                "staged_artifacts_sha256": self.staged,
            },
            "validation": {
                "report_sha256": self.report_sha,
                "role": "validate",
                "status": "development_openset_pass",
                "novelty_seeds": [20260950, 20260951],
            },
            "fail_closed": {
                "role_assets_bound_by_sha256": True,
                "distinct_role_assets": True,
                "role_asset_sha256_must_differ": True,
                "classifier_runs_only_after_rejector_acceptance": True,
                "public_known_label_from_classifier_only": True,
            },
        }
        write_json(self.openset / packager.DUAL_BINDING, binding)
        parity = {
            "schema": "time-domain-openset-parity-v1",
            "schema_version": 3,
            "status": packager.STAGING_STATUS,
            "candidate_id": packager.CANDIDATE_ID,
            "counts": {"accepted": 1, "classifier_executed": 1},
            "rows": [
                {
                    "name": "known",
                    "rejected_stage": None,
                    "classifier_executed": True,
                    "classifier": {"predicted_class_label": "a"},
                },
                {
                    "name": "noise",
                    "rejected_stage": 1,
                    "classifier_executed": False,
                    "classifier": None,
                },
                {
                    "name": "unknown",
                    "rejected_stage": 2,
                    "classifier_executed": False,
                    "classifier": None,
                },
            ],
        }
        write_json(self.openset / packager.OPENSET_PARITY, parity)
        self.refresh_openset_manifest(rejector, classifier)

    def refresh_openset_manifest(
        self,
        rejector: dict | None = None,
        classifier: dict | None = None,
    ) -> None:
        if rejector is None:
            rejector = self.role_state(
                self.rejector,
                packager.REJECTOR_WEIGHTS,
                packager.REJECTOR_PROBE,
            )
        if classifier is None:
            classifier = self.role_state(
                self.classifier,
                packager.CLASSIFIER_WEIGHTS,
                packager.CLASSIFIER_PROBE,
            )
        manifest = {
            "schema": packager.OPENSET_EXPORT_SCHEMA,
            "schema_version": 1,
            "status": packager.STAGING_STATUS,
            "candidate_id": packager.CANDIDATE_ID,
            "outputs": {
                name: record(self.openset / name)
                for name in (
                    packager.OPENSET_POLICY,
                    packager.DUAL_BINDING,
                    packager.OPENSET_PARITY,
                )
            },
            "role_exports": {
                "rejector": {
                    "manifest_sha256": rejector["manifest_record"]["sha256"],
                    "runtime_role": packager.REJECTOR_ROLE,
                    "weights": {
                        "name": packager.REJECTOR_WEIGHTS,
                        **rejector["weights_record"],
                    },
                    "probe": {
                        "name": packager.REJECTOR_PROBE,
                        **rejector["probe_record"],
                    },
                },
                "classifier": {
                    "manifest_sha256": classifier["manifest_record"]["sha256"],
                    "runtime_role": packager.CLASSIFIER_ROLE,
                    "weights": {
                        "name": packager.CLASSIFIER_WEIGHTS,
                        **classifier["weights_record"],
                    },
                    "probe": {
                        "name": packager.CLASSIFIER_PROBE,
                        **classifier["probe_record"],
                    },
                },
            },
        }
        write_json(self.openset / packager.OPENSET_MANIFEST, manifest)

    def package(self) -> dict:
        return packager.package(
            self.rejector,
            self.classifier,
            self.openset,
            self.destination,
        )


class PackageTests(unittest.TestCase):
    def test_materializes_only_four_runtime_assets_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticInputs(Path(temporary))
            manifest = fixture.package()
            self.assertEqual(manifest["schema"], packager.PACKAGE_SCHEMA)
            self.assertEqual(
                set(manifest["assets"]), set(packager.DEPLOYABLE_ASSETS)
            )
            self.assertEqual(
                {path.name for path in fixture.destination.iterdir()},
                {*packager.DEPLOYABLE_ASSETS, packager.PACKAGE_MANIFEST},
            )
            self.assertNotIn(
                packager.OPENSET_PARITY,
                {path.name for path in fixture.destination.iterdir()},
            )
            self.assertFalse(
                manifest["external_evidence"]["parity"]["packaged"]
            )
            self.assertEqual(
                manifest["roles"]["rejector"]["asset"],
                manifest["assets"][packager.REJECTOR_WEIGHTS],
            )
            self.assertEqual(
                manifest["roles"]["classifier"]["asset"],
                manifest["assets"][packager.CLASSIFIER_WEIGHTS],
            )
            for path in fixture.destination.iterdir():
                self.assertLess(
                    path.stat().st_size, packager.MAX_DEPLOYABLE_BYTES
                )

    def test_legacy_single_fusion_manifest_is_refused_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticInputs(Path(temporary))
            path = fixture.openset / packager.OPENSET_MANIFEST
            value = json.loads(path.read_text(encoding="utf-8"))
            value["schema"] = "time-domain-v3-openset-staging-manifest-v1"
            write_json(path, value)
            with self.assertRaisesRegex(packager.PackageError, "dual staging"):
                fixture.package()
            self.assertFalse(fixture.destination.exists())

    def test_swapped_runtime_role_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticInputs(Path(temporary))
            weights = fixture.classifier / packager.CLASSIFIER_WEIGHTS
            value = json.loads(weights.read_text(encoding="utf-8"))
            value["runtime_role"] = packager.REJECTOR_ROLE
            write_json(weights, value)
            manifest_path = fixture.classifier / packager.FUSION_MANIFEST
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["emitted"][packager.CLASSIFIER_WEIGHTS] = record(weights)
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(packager.PackageError, "exact role"):
                fixture.package()
            self.assertFalse(fixture.destination.exists())

    def test_rejected_parity_row_may_not_serialize_classifier_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticInputs(Path(temporary))
            parity_path = fixture.openset / packager.OPENSET_PARITY
            parity = json.loads(parity_path.read_text(encoding="utf-8"))
            parity["rows"][1]["classifier_executed"] = True
            parity["rows"][1]["classifier"] = {
                "predicted_class_label": "leak"
            }
            write_json(parity_path, parity)
            fixture.refresh_openset_manifest()
            with self.assertRaisesRegex(
                packager.PackageError, "rejected parity row"
            ):
                fixture.package()

    def test_oversized_deployable_asset_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticInputs(Path(temporary))
            policy_path = fixture.openset / packager.OPENSET_POLICY
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
            policy["padding"] = "x" * packager.MAX_DEPLOYABLE_BYTES
            write_json(policy_path, policy)
            fixture.refresh_openset_manifest()
            with self.assertRaisesRegex(packager.PackageError, "25 MiB"):
                fixture.package()
            self.assertFalse(fixture.destination.exists())

    def test_nonempty_destination_is_not_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticInputs(Path(temporary))
            fixture.destination.mkdir()
            sentinel = fixture.destination / "mine.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(packager.PackageError, "empty"):
                fixture.package()
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
