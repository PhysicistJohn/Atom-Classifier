"""End-to-end synthetic tests for the dual-fusion v3 release promoter."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from typing import Any


TOOLS = Path(__file__).resolve().parent


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


promoter = load_module(
    "promote_v3_release_assets_under_test",
    TOOLS / "promote-v3-release-assets.py",
)
package_tests = load_module(
    "package_v3_staging_fixture_for_promoter",
    TOOLS / "test_package_v3_staging_assets.py",
)


def digest_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def digest(label: str) -> str:
    return digest_bytes(label.encode("utf-8"))


def json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.write_bytes(json_bytes(payload))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_record(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": digest_bytes(raw)}


def historical_metadata() -> dict[str, Any]:
    return copy.deepcopy(
        promoter._expected_protocol()["historical_gate_redeclaration"]
    )


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


class SyntheticPromotion:
    """Build one internally consistent candidate/package/release-report chain."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True)
        self.release = self.root / "assets-v3-release"
        self.live_v2 = self.root / "assets-v2-live"
        self.legacy_staging = self.root / "assets-v3-staging"
        self.prefilter = self.root / "prefilter"
        self.prefilter.mkdir()
        self.prefilter_bundles = self._build_prefilter()
        prefilter_digest = hashlib.sha256()
        for name in sorted(path.name for path in self.prefilter.iterdir()):
            prefilter_digest.update(f"{name}:".encode("utf-8"))
            prefilter_digest.update(
                self.prefilter_bundles[name[1:]].encode("utf-8")
            )
        self.prefilter_sha = prefilter_digest.hexdigest()

        self.rejector_bundle = self._build_bundle(
            "rejector-bundle", promoter.REJECTOR_ROLE
        )
        self.classifier_bundle = self._build_bundle(
            "classifier-bundle", promoter.CLASSIFIER_ROLE
        )
        (
            self.rejector_fusion,
            self.rejector_fusion_files,
            self.rejector_fusion_sha,
        ) = self._build_fusion("rejector-fusion")
        (
            self.classifier_fusion,
            self.classifier_fusion_files,
            self.classifier_fusion_sha,
        ) = self._build_fusion("classifier-fusion")

        self.inputs = package_tests.SyntheticInputs(
            self.root / "package-chain"
        )
        self._bind_role_export(
            self.inputs.rejector,
            package_tests.packager.REJECTOR_WEIGHTS,
            package_tests.packager.REJECTOR_PROBE,
            self.rejector_bundle["manifest_sha256"],
        )
        self._bind_role_export(
            self.inputs.classifier,
            package_tests.packager.CLASSIFIER_WEIGHTS,
            package_tests.packager.CLASSIFIER_PROBE,
            self.classifier_bundle["manifest_sha256"],
        )
        self.inputs.build_openset()
        self._bind_openset_provenance()
        self.inputs.package()
        self.staging = self.inputs.destination.resolve()
        self.staging_manifest = self.staging / promoter.PACKAGE_MANIFEST
        self.staging_binding = self.staging / promoter.DUAL_BINDING
        self.package_payload = read_json(self.staging_manifest)
        self.binding_payload = read_json(self.staging_binding)

        self.staged = self._build_staged_validation()
        self.frozen_contract = self.root / "frozen-prevalidation.json"
        write_json(
            self.frozen_contract,
            {
                "schema": promoter.PREVALIDATION_CONTRACT_SCHEMA,
                "status": "frozen_before_validation",
                "candidate_id": promoter.CANDIDATE_ID,
            },
        )
        self.validation_evidence = self.root / "validation-evidence.json"
        write_json(
            self.validation_evidence,
            {
                "schema": promoter.VALIDATION_EVIDENCE_SCHEMA,
                "status": "development_openset_pass",
                "candidate_id": promoter.CANDIDATE_ID,
                "candidate_contract": {
                    "path": str(self.frozen_contract),
                    "sha256": digest_bytes(self.frozen_contract.read_bytes()),
                },
            },
        )
        self.candidate = self.root / "release-candidate.json"
        self._write_candidate()
        self.candidate_sha = digest_bytes(self.candidate.read_bytes())
        self.protocol = self._protocol()
        self.release_root = (
            self.root
            / promoter.EXPECTED_RELEASE_ROOT.relative_to(promoter.REPO)
        )
        self._write_release_suite()
        self.report = self.release_root / "RELEASE_EVALUATION.json"
        self._write_report()
        self.policy = promoter.PromotionPolicy(
            repo_root=self.root,
            staging_package=self.staging,
            release_package=self.release,
            live_v2_assets=self.live_v2,
            legacy_v3_staging=self.legacy_staging,
            require_git_tracking=False,
        )

    def _build_prefilter(self) -> dict[str, str]:
        hashes: dict[str, str] = {}
        for length in (4096, 8192, 16384):
            directory = self.prefilter / f"N{length}"
            directory.mkdir()
            for name in promoter.PREFILTER_BUNDLE_FILES:
                path = directory / name
                if name == "meta.json":
                    write_json(
                        path,
                        {
                            "schema": "noise-prefilter-bundle-v1",
                            "capture_length": length,
                        },
                    )
                else:
                    path.write_bytes(f"N{length}:{name}".encode("utf-8"))
            hashes[str(length)] = promoter._prefilter_bundle_sha256(
                directory, length
            )
        return hashes

    def _build_bundle(self, name: str, runtime_role: str) -> dict[str, Any]:
        directory = self.root / name
        directory.mkdir()
        assets: dict[str, dict[str, Any]] = {}
        for asset_name in sorted(promoter.EXPECTED_BUNDLE_ASSETS):
            path = directory / asset_name
            path.write_bytes(f"{name}:{asset_name}".encode("utf-8"))
            assets[asset_name] = file_record(path)
        manifest_path = directory / "bundle_manifest.json"
        write_json(
            manifest_path,
            {
                "schema": promoter.BUNDLE_SCHEMA,
                "schema_version": promoter.BUNDLE_SCHEMA_VERSION,
                "kind": promoter.BUNDLE_KIND,
                "runtime_role": runtime_role,
                "assets": assets,
            },
        )
        return {
            "directory": directory,
            "manifest_path": manifest_path,
            "manifest_sha256": digest_bytes(manifest_path.read_bytes()),
            "assets": assets,
        }

    def _build_fusion(
        self, name: str
    ) -> tuple[Path, dict[str, str], str]:
        directory = self.root / name
        directory.mkdir()
        hashes: dict[str, str] = {}
        for file_name in sorted(promoter.EXPECTED_FUSION_FILES):
            path = directory / file_name
            path.write_bytes(f"{name}:{file_name}".encode("utf-8"))
            hashes[file_name] = digest_bytes(path.read_bytes())
        return directory, hashes, promoter._fusion_directory_sha256(hashes)

    def _bind_role_export(
        self,
        directory: Path,
        weights_name: str,
        probe_name: str,
        bundle_sha: str,
    ) -> None:
        weights_path = directory / weights_name
        weights = read_json(weights_path)
        weights["provenance"]["source_bundle_manifest_sha256"] = bundle_sha
        weights["development_only"] = True
        weights["release_evidence"] = False
        weights["release_blockers"] = ["sealed seed-20260735 evaluation"]
        write_json(weights_path, weights)
        manifest_path = directory / package_tests.packager.FUSION_MANIFEST
        manifest = read_json(manifest_path)
        manifest["source_bundle_manifest_sha256"] = bundle_sha
        manifest["emitted"][weights_name] = file_record(weights_path)
        manifest["emitted"][probe_name] = file_record(directory / probe_name)
        write_json(manifest_path, manifest)

    def _bind_openset_provenance(self) -> None:
        policy_path = (
            self.inputs.openset / package_tests.packager.OPENSET_POLICY
        )
        policy = read_json(policy_path)
        policy["provenance"]["prefilter_set_sha256"] = self.prefilter_sha
        policy["provenance"].update(
            {
                "development_only": True,
                "release_evidence": False,
                "sealed_release_data_used": 0,
                "consumed_test_rows_used": 0,
                "release_seed_not_spent": promoter.RELEASE_SEED,
                "staged_validation_all_pass": True,
                "staged_validation_role": "validate",
                "staged_validation_status": "development_openset_pass",
            }
        )
        binding_path = (
            self.inputs.openset / package_tests.packager.DUAL_BINDING
        )
        binding = read_json(binding_path)
        role_fusions = {
            "rejector": self.rejector_fusion_sha,
            "classifier": self.classifier_fusion_sha,
        }
        for label, fusion_sha in role_fusions.items():
            policy["provenance"]["runtime_roles"][label][
                "fusion_directory_sha256"
            ] = fusion_sha
            binding["roles"][label]["fusion_directory_sha256"] = fusion_sha
        write_json(policy_path, policy)
        binding["openset_policy"]["asset_sha256"] = digest_bytes(
            policy_path.read_bytes()
        )
        write_json(binding_path, binding)
        self.inputs.refresh_openset_manifest()

    def _build_staged_validation(self) -> dict[str, Any]:
        directory = self.root / "staged-validation"
        directory.mkdir()
        contents = {
            "v3_branch_lof_components.npz": b"lof",
            "v3_open_policy_stage_two.npz": b"stage two",
            "v3_staged_composite_policy.npz": b"composite",
        }
        for name, raw in contents.items():
            (directory / name).write_bytes(raw)
        report_path = directory / "staged-validation-report.json"
        report_path.write_bytes(b"validation report")
        return {
            "directory": directory,
            "report_path": report_path,
            "report_sha256": digest_bytes(report_path.read_bytes()),
            "artifact_sha256": {
                name: digest_bytes(raw) for name, raw in contents.items()
            },
        }

    def _bundle_component(self, bundle: dict[str, Any]) -> dict[str, Any]:
        return {
            "directory": str(bundle["directory"]),
            "manifest_path": str(bundle["manifest_path"]),
            "manifest_sha256": bundle["manifest_sha256"],
            "schema": promoter.BUNDLE_SCHEMA,
            "schema_version": promoter.BUNDLE_SCHEMA_VERSION,
            "kind": promoter.BUNDLE_KIND,
            "assets": copy.deepcopy(bundle["assets"]),
        }

    def _fusion_component(
        self,
        role: str,
        directory: Path,
        files: dict[str, str],
        directory_sha: str,
    ) -> dict[str, Any]:
        return {
            "directory": str(directory),
            "directory_sha256": directory_sha,
            "file_sha256": copy.deepcopy(files),
            "seed": 20260730,
            "role": role,
        }

    def _browser_records(self) -> dict[str, dict[str, Any]]:
        assets = self.package_payload["assets"]
        return {
            "classifier": {
                "path": str(self.staging / promoter.CLASSIFIER_WEIGHTS),
                "sha256": assets[promoter.CLASSIFIER_WEIGHTS]["sha256"],
                "schema": promoter.FUSION_SCHEMA,
                "status": promoter.STAGING_STATUS,
            },
            "rejector": {
                "path": str(self.staging / promoter.REJECTOR_WEIGHTS),
                "sha256": assets[promoter.REJECTOR_WEIGHTS]["sha256"],
                "schema": promoter.FUSION_SCHEMA,
                "status": promoter.STAGING_STATUS,
            },
            "openset_policy": {
                "path": str(self.staging / promoter.OPENSET_POLICY),
                "sha256": assets[promoter.OPENSET_POLICY]["sha256"],
                "schema": promoter.OPENSET_SCHEMA,
                "status": promoter.STAGING_STATUS,
            },
        }

    def _write_candidate(self) -> None:
        roles = self.binding_payload["roles"]
        candidate = {
            "schema": promoter.CANDIDATE_SCHEMA,
            "status": "release_candidate_frozen",
            "candidate_id": promoter.CANDIDATE_ID,
            "validation_evidence": {
                "path": str(self.validation_evidence),
                "sha256": digest_bytes(self.validation_evidence.read_bytes()),
            },
            "classifier": {
                "role": "known_class_label",
                "fusion": {
                    "directory": str(self.classifier_fusion),
                    "directory_sha256": roles["classifier"][
                        "fusion_directory_sha256"
                    ],
                    "file_sha256": copy.deepcopy(
                        self.classifier_fusion_files
                    ),
                },
                "runtime_bundle": {
                    key: self.classifier_bundle[key]
                    if key == "manifest_sha256"
                    else str(self.classifier_bundle[key])
                    for key in (
                        "directory",
                        "manifest_path",
                        "manifest_sha256",
                    )
                },
            },
            "rejector": {
                "role": "known_unknown_decision",
                "fusion": {
                    "directory": str(self.rejector_fusion),
                    "directory_sha256": roles["rejector"][
                        "fusion_directory_sha256"
                    ],
                    "file_sha256": copy.deepcopy(self.rejector_fusion_files),
                },
                "runtime_bundle": {
                    key: self.rejector_bundle[key]
                    if key == "manifest_sha256"
                    else str(self.rejector_bundle[key])
                    for key in (
                        "directory",
                        "manifest_path",
                        "manifest_sha256",
                    )
                },
            },
            "staged_validation": {
                "directory": str(self.staged["directory"]),
                "report_path": str(self.staged["report_path"]),
                "report_sha256": self.staged["report_sha256"],
                "artifact_sha256": copy.deepcopy(
                    self.staged["artifact_sha256"]
                ),
            },
            "stage_one_prefilter": {
                "directory": str(self.prefilter),
                "set_sha256": self.prefilter_sha,
                "bundle_sha256": copy.deepcopy(self.prefilter_bundles),
            },
            "browser_assets": self._browser_records(),
            "staging_package_manifest": {
                "path": str(self.staging_manifest),
                "sha256": digest_bytes(self.staging_manifest.read_bytes()),
                "schema": promoter.PACKAGE_SCHEMA,
                "status": promoter.STAGING_STATUS,
            },
            "dual_binding": {
                "path": str(self.staging_binding),
                "sha256": digest_bytes(self.staging_binding.read_bytes()),
                "schema": promoter.BINDING_SCHEMA,
            },
        }
        write_json(self.candidate, candidate)

    def _protocol(self) -> dict[str, Any]:
        return copy.deepcopy(promoter._expected_protocol())

    def _write_release_suite(self) -> None:
        self.release_root.mkdir(parents=True)
        intent = {
            "status": "in_progress",
            "release_seed": promoter.RELEASE_SEED,
            "candidate_path": str(self.candidate),
            "candidate_sha256": self.candidate_sha,
            "evaluation_protocol": copy.deepcopy(self.protocol),
        }
        intent_path = self.release_root / "RELEASE_INTENT.json"
        write_json(intent_path, intent)
        manifest = copy.deepcopy(intent)
        manifest["status"] = "complete"
        manifest["release_intent_sha256"] = digest_bytes(
            intent_path.read_bytes()
        )
        write_json(self.release_root / "RELEASE_MANIFEST.json", manifest)

    def _components(self) -> dict[str, Any]:
        candidate = read_json(self.candidate)
        validation_sha = digest_bytes(self.validation_evidence.read_bytes())
        frozen_sha = digest_bytes(self.frozen_contract.read_bytes())
        return {
            "candidate_contract": {
                "path": str(self.candidate),
                "sha256": self.candidate_sha,
                "schema": promoter.CANDIDATE_SCHEMA,
                "candidate_id": promoter.CANDIDATE_ID,
                "status": "release_candidate_frozen",
            },
            "validation_evidence": {
                "path": str(self.validation_evidence),
                "sha256": validation_sha,
                "schema": promoter.VALIDATION_EVIDENCE_SCHEMA,
                "status": "development_openset_pass",
            },
            "frozen_prevalidation_contract": {
                "path": str(self.frozen_contract),
                "sha256": frozen_sha,
                "schema": promoter.PREVALIDATION_CONTRACT_SCHEMA,
                "status": "frozen_before_validation",
            },
            "classifier_runtime_bundle": self._bundle_component(
                self.classifier_bundle
            ),
            "rejector_runtime_bundle": self._bundle_component(
                self.rejector_bundle
            ),
            "classifier_fusion": self._fusion_component(
                "known_class_label",
                self.classifier_fusion,
                self.classifier_fusion_files,
                self.binding_payload["roles"]["classifier"][
                    "fusion_directory_sha256"
                ],
            ),
            "rejector_fusion": self._fusion_component(
                "known_unknown_decision",
                self.rejector_fusion,
                self.rejector_fusion_files,
                self.binding_payload["roles"]["rejector"][
                    "fusion_directory_sha256"
                ],
            ),
            "staged_validation": {
                "directory": str(self.staged["directory"]),
                "report_path": str(self.staged["report_path"]),
                "report_sha256": self.staged["report_sha256"],
                "status": "development_openset_pass",
                "novelty_seeds": [20260950, 20260951],
                "artifact_sha256": copy.deepcopy(
                    self.staged["artifact_sha256"]
                ),
            },
            "stage_one_prefilter": {
                "directory": str(self.prefilter),
                "set_sha256": self.prefilter_sha,
                "bundle_sha256": copy.deepcopy(self.prefilter_bundles),
            },
            "browser_assets": copy.deepcopy(candidate["browser_assets"]),
            "staging_package_manifest": copy.deepcopy(
                candidate["staging_package_manifest"]
            ),
            "dual_binding": copy.deepcopy(candidate["dual_binding"]),
            "dual_binding_sha256": candidate["dual_binding"]["sha256"],
            "source_sha256": {
                "enforced": {
                    "fit_v3_openset_staged.py": (
                        promoter.EXPECTED_LEDGER_TRANSITION[
                            "current_sha256"
                        ]
                    ),
                },
                "recorded_only": {},
                "post_validation_ledger_transitions": {
                    "fit_v3_openset_staged.py": copy.deepcopy(
                        promoter.EXPECTED_LEDGER_TRANSITION
                    ),
                },
                "evaluator_chain": {},
            },
        }

    def _metric_bodies(self) -> dict[str, Any]:
        lengths = tuple(str(value) for value in promoter.REQUIRED_CAPTURE_LENGTHS)
        closed = {
            length: {
                "accuracy": 0.72,
                "balanced_accuracy": 0.75,
                "mean_pairwise_cosine": 0.78,
                "family": {"accuracy": 0.82},
                "high_snr": {"accuracy": 0.78},
                "low_snr": {"accuracy": 0.72},
                "clean_accuracy": 0.85,
                "clean_subset": {"accuracy": 0.85},
                "impaired_subset": {"accuracy": 0.72},
            }
            for length in lengths
        }
        five = {
            length: {"balanced_accuracy": 0.85} for length in lengths
        }
        opened = {
            length: {
                "known_rows": 10,
                "auroc_overall": 0.72,
                "auroc_noise": 0.80,
                "auroc_chirp": 0.80,
                "known_false_unknown_rate": 0.10,
                "flagged_unknown_noise": 0.10,
                "flagged_unknown_chirp": 0.10,
                "known_gated_fraction": 0.0,
                "known_false_unknown_by_stage": {
                    "rows": 10,
                    "staged_false_unknown_rate": 0.10,
                    "stage_one_gate_rate": 0.0,
                    "stage_two_false_unknown_rate_marginal": 0.10,
                    "rejected_by_stage_two_only": 1,
                },
            }
            for length in lengths
        }
        decisions = {
            length: {
                "rows": 10,
                "unknown": 1,
                "gated_at_stage_one": 0,
                "accepted_rows": 9,
            }
            for length in lengths
        }
        length_rows = [
            {
                "capture_length": length,
                "balanced_accuracy": 0.75,
                "mean_pairwise_cosine": 0.78,
                "paired_to_matched": {
                    "prediction_agreement": 0.80,
                    "embedding_cosine_mean": 0.85,
                },
            }
            for length in promoter.REQUIRED_CAPTURE_LENGTHS
        ]
        scale_rows = [
            {
                "factor": factor,
                "balanced_accuracy": 0.75,
                "mean_pairwise_cosine": 0.78,
                "paired_to_factor1": {
                    "prediction_agreement": 0.75,
                    "embedding_cosine_mean": 0.80,
                },
            }
            for factor in promoter.PHYSICAL_SCALE_FACTORS
        ]
        return {
            "closed_per_length": closed,
            "family_per_length": {
                length: row["family"] for length, row in closed.items()
            },
            "high_snr_per_length": {
                length: row["high_snr"] for length, row in closed.items()
            },
            "low_snr_per_length": {
                length: row["low_snr"] for length, row in closed.items()
            },
            "clean_subset_per_length": {
                length: row["clean_subset"]
                for length, row in closed.items()
            },
            "impaired_subset_per_length": {
                length: row["impaired_subset"]
                for length, row in closed.items()
            },
            "five_shot_predeclared_per_length": five,
            "open_staged_per_length": opened,
            "staged_known_decisions_per_length": decisions,
            "matched_length_sweep": {
                "matched_capture_length": 16384,
                "rows": length_rows,
                "worst_balanced_accuracy": 0.75,
                "worst_mean_pairwise_cosine": 0.78,
                "worst_prediction_agreement_to_matched": 0.80,
                "worst_embedding_cosine_to_matched": 0.85,
            },
            "physical_scale_sweep": {
                "factors": list(promoter.PHYSICAL_SCALE_FACTORS),
                "rows": scale_rows,
                "worst_balanced_accuracy": 0.75,
                "worst_mean_pairwise_cosine": 0.78,
                "worst_prediction_agreement_to_factor1": 0.75,
                "worst_embedding_cosine_to_factor1": 0.80,
            },
        }

    def _write_report(self) -> None:
        metrics = self._metric_bodies()
        report = {
            "schema": promoter.EVALUATOR_SCHEMA,
            "status": "complete",
            "release_evidence": True,
            "evaluation_version": promoter.EVALUATION_VERSION,
            "development_data_loaded": False,
            "retraining_performed": False,
            "recalibration_performed": False,
            "architecture_contract": {
                "intentional_dual_fusion": True,
                "candidate_architecture": (
                    "dual_fusion_classifier8k_rejector4k"
                ),
                "known_label_source": (
                    "classifier_fusion_8k_regularized"
                ),
                "known_unknown_source": (
                    "rejector_fusion_4k_frozen_policy"
                ),
                "closed_gate_source": (
                    "classifier_fusion_8k_regularized"
                ),
                "open_gate_source": (
                    "rejector_fusion_4k_frozen_policy"
                ),
                "gates_before_classification": True,
            },
            "candidate": {
                "path": str(self.candidate),
                "sha256": self.candidate_sha,
                "schema": promoter.CANDIDATE_SCHEMA,
                "candidate_id": promoter.CANDIDATE_ID,
                "classes": ["a", "b"],
                "components": self._components(),
                "frozen_assets_used": ["synthetic dual frozen candidate"],
            },
            **metrics,
            "gates": release_gates(self.candidate_sha),
            "historical_gate_redeclaration": historical_metadata(),
            "all_release_gates_pass": True,
            "provenance": {
                "release_root": str(self.release_root),
                "release_intent_sha256": digest_bytes(
                    (
                        self.release_root / "RELEASE_INTENT.json"
                    ).read_bytes()
                ),
                "release_manifest_sha256": digest_bytes(
                    (
                        self.release_root / "RELEASE_MANIFEST.json"
                    ).read_bytes()
                ),
                "release_seed": promoter.RELEASE_SEED,
                "evaluation_protocol": copy.deepcopy(self.protocol),
                "candidate_sha256": self.candidate_sha,
                "prefix_nesting": {"passes": True},
                "dependency_provenance": {"passes": True},
                "unscored_start_probe": {
                    "scored": False,
                    "passes": True,
                },
                "evaluator_path": str(promoter.EXPECTED_EVALUATOR),
                "evaluator_sha256": promoter.EXPECTED_EVALUATOR_SHA256,
                "v2_evaluator_sha256": (
                    promoter.EXPECTED_V2_EVALUATOR_SHA256
                ),
            },
        }
        self.report_payload = report
        write_json(self.report, report)

    def rewrite_report(self) -> None:
        write_json(self.report, self.report_payload)

    def promote(
        self,
        *,
        destination: Path | None = None,
        policy: Any | None = None,
    ) -> dict[str, Any]:
        return promoter.promote(
            self.staging,
            self.report,
            destination or self.release,
            policy=policy or self.policy,
        )


class PromotionTests(unittest.TestCase):
    def test_promotes_exact_dual_package_and_rebinds_release_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticPromotion(Path(temporary) / "fixture")
            source = {
                name: (fixture.staging / name).read_bytes()
                for name in promoter.STAGING_FILE_NAMES
            }
            manifest = fixture.promote()

            self.assertEqual(
                {path.name for path in fixture.release.iterdir()},
                set(promoter.STAGING_FILE_NAMES),
            )
            self.assertEqual(manifest["status"], promoter.RELEASE_STATUS)
            self.assertEqual(
                set(manifest["assets"]), set(promoter.ASSET_NAMES)
            )
            self.assertEqual(
                source,
                {
                    name: (fixture.staging / name).read_bytes()
                    for name in promoter.STAGING_FILE_NAMES
                },
            )
            for name in promoter.ASSET_NAMES:
                payload = read_json(fixture.release / name)
                self.assertEqual(payload["status"], promoter.RELEASE_STATUS)
                expected = file_record(fixture.release / name)
                self.assertEqual(
                    manifest["assets"][name]["sha256"],
                    expected["sha256"],
                )
                self.assertEqual(
                    manifest["assets"][name]["bytes"], expected["bytes"]
                )
            for name in (
                promoter.REJECTOR_WEIGHTS,
                promoter.CLASSIFIER_WEIGHTS,
            ):
                payload = read_json(fixture.release / name)
                self.assertIs(payload["development_only"], False)
                self.assertIs(payload["release_evidence"], True)
                self.assertEqual(payload["release_blockers"], [])

            binding = read_json(fixture.release / promoter.DUAL_BINDING)
            policy = read_json(fixture.release / promoter.OPENSET_POLICY)
            role_files = {
                "rejector": promoter.REJECTOR_WEIGHTS,
                "classifier": promoter.CLASSIFIER_WEIGHTS,
            }
            for role, name in role_files.items():
                actual = digest_bytes((fixture.release / name).read_bytes())
                self.assertEqual(
                    binding["roles"][role]["asset_sha256"], actual
                )
                self.assertEqual(
                    policy["provenance"]["runtime_roles"][role][
                        "browser_asset_sha256"
                    ],
                    actual,
                )
            self.assertEqual(
                binding["openset_policy"]["asset_sha256"],
                digest_bytes(
                    (fixture.release / promoter.OPENSET_POLICY).read_bytes()
                ),
            )
            promotion = manifest["promotion"]
            self.assertEqual(
                promotion["source"]["package_manifest_sha256"],
                digest_bytes(fixture.staging_manifest.read_bytes()),
            )
            self.assertEqual(
                promotion["evaluation"]["report_sha256"],
                digest_bytes(fixture.report.read_bytes()),
            )
            self.assertEqual(
                promotion["evaluation"]["gate_contract"],
                "strict_imported_v2_23_of_23",
            )
            self.assertEqual(
                promotion["candidate_artifacts"][
                    "post_validation_ledger_transition"
                ],
                promoter.EXPECTED_LEDGER_TRANSITION,
            )
            self.assertEqual(fixture.release.stat().st_mode & 0o777, 0o755)
            for name in promoter.STAGING_FILE_NAMES:
                self.assertEqual(
                    (fixture.release / name).stat().st_mode & 0o777,
                    0o644,
                )

    def test_promotion_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticPromotion(Path(temporary) / "fixture")
            first = fixture.release
            second = fixture.root / "second-release"
            fixture.promote()
            second_policy = promoter.PromotionPolicy(
                repo_root=fixture.root,
                staging_package=fixture.staging,
                release_package=second,
                live_v2_assets=fixture.live_v2,
                legacy_v3_staging=fixture.legacy_staging,
                require_git_tracking=False,
            )
            fixture.promote(destination=second, policy=second_policy)
            self.assertEqual(
                {
                    name: (first / name).read_bytes()
                    for name in promoter.STAGING_FILE_NAMES
                },
                {
                    name: (second / name).read_bytes()
                    for name in promoter.STAGING_FILE_NAMES
                },
            )

    def test_wrong_release_seed_is_refused_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticPromotion(Path(temporary) / "fixture")
            fixture.report_payload["provenance"]["release_seed"] = 20260736
            fixture.rewrite_report()
            with self.assertRaisesRegex(promoter.PromotionError, "20260735"):
                fixture.promote()
            self.assertFalse(fixture.release.exists())

    def test_relaxed_or_falsely_passing_gate_is_refused(self) -> None:
        mutations = (
            (
                "relaxed threshold",
                lambda report: report["gates"][
                    "five_shot_worst_length_balanced"
                ].__setitem__("threshold", 0.84),
                "contract differs",
            ),
            (
                "false pass",
                lambda report: report["gates"][
                    "closed_fine_worst_length"
                ].__setitem__("value", 0.71),
                "contract differs",
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            for index, (label, mutate, message) in enumerate(mutations):
                with self.subTest(label=label):
                    fixture = SyntheticPromotion(
                        Path(temporary) / f"fixture-{index}"
                    )
                    mutate(fixture.report_payload)
                    fixture.rewrite_report()
                    with self.assertRaisesRegex(
                        promoter.PromotionError, message
                    ):
                        fixture.promote()
                    self.assertFalse(fixture.release.exists())

    def test_gate_value_must_be_derived_from_metric_body(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticPromotion(Path(temporary) / "fixture")
            fixture.report_payload["closed_per_length"]["4096"][
                "accuracy"
            ] = 0.0
            fixture.rewrite_report()
            with self.assertRaisesRegex(
                promoter.PromotionError, "contract differs"
            ):
                fixture.promote()
            self.assertFalse(fixture.release.exists())

    def test_protocol_must_equal_the_canonical_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticPromotion(Path(temporary) / "fixture")
            fixture.report_payload["provenance"]["evaluation_protocol"][
                "minimum_target_per_class"
            ] = 79
            fixture.rewrite_report()
            with self.assertRaisesRegex(
                promoter.PromotionError, "exact seed-20260735 fixture"
            ):
                fixture.promote()

    def test_evaluator_identity_is_pinned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticPromotion(Path(temporary) / "fixture")
            fixture.report_payload["provenance"]["evaluator_sha256"] = (
                "0" * 64
            )
            fixture.rewrite_report()
            with self.assertRaisesRegex(
                promoter.PromotionError, "evaluator identity"
            ):
                fixture.promote()

    def test_post_validation_source_transition_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticPromotion(Path(temporary) / "fixture")
            transition = fixture.report_payload["candidate"]["components"][
                "source_sha256"
            ]["post_validation_ledger_transitions"][
                "fit_v3_openset_staged.py"
            ]
            transition["next_clean_novelty_seed"] = 20260953
            fixture.rewrite_report()
            with self.assertRaisesRegex(
                promoter.PromotionError, "ledger transition record differs"
            ):
                fixture.promote()

    def test_report_must_be_the_release_root_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticPromotion(Path(temporary) / "fixture")
            displaced = fixture.root / "copied-evaluation.json"
            displaced.write_bytes(fixture.report.read_bytes())
            with self.assertRaisesRegex(
                promoter.PromotionError, "release_root"
            ):
                promoter.promote(
                    fixture.staging,
                    displaced,
                    fixture.release,
                    policy=fixture.policy,
                )

    def test_symlinked_validation_evidence_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticPromotion(Path(temporary) / "fixture")
            target = fixture.root / "identical-validation.json"
            target.write_bytes(fixture.validation_evidence.read_bytes())
            fixture.validation_evidence.unlink()
            fixture.validation_evidence.symlink_to(target)
            with self.assertRaisesRegex(
                promoter.PromotionError, "symlink"
            ):
                fixture.promote()
            self.assertFalse(fixture.release.exists())

    def test_symlinked_destination_is_never_written_through(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            fixture = SyntheticPromotion(base / "fixture")
            outside = base / "outside"
            outside.mkdir()
            sentinel = outside / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")
            fixture.release.symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(
                promoter.PromotionError, "symlink"
            ):
                fixture.promote()
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
            self.assertEqual({path.name for path in outside.iterdir()}, {"keep.txt"})

    def test_symlinked_destination_parent_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            fixture = SyntheticPromotion(base / "fixture")
            outside = base / "outside"
            outside.mkdir()
            alias = fixture.root / "redirect-parent"
            alias.symlink_to(outside, target_is_directory=True)
            destination = alias / "release"
            policy = promoter.PromotionPolicy(
                repo_root=fixture.root,
                staging_package=fixture.staging,
                release_package=destination,
                live_v2_assets=fixture.live_v2,
                legacy_v3_staging=fixture.legacy_staging,
                require_git_tracking=False,
            )
            with self.assertRaisesRegex(
                promoter.PromotionError, "symlink"
            ):
                fixture.promote(destination=destination, policy=policy)
            self.assertEqual(list(outside.iterdir()), [])

    def test_absolute_external_evidence_path_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticPromotion(Path(temporary) / "fixture")
            package = read_json(fixture.staging_manifest)
            package["external_evidence"]["parity"]["path"] = (
                "/tmp/outside-parity.json"
            )
            write_json(fixture.staging_manifest, package)
            with self.assertRaisesRegex(
                promoter.PromotionError, "package-relative"
            ):
                fixture.promote()

    def test_prefilter_bytes_are_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticPromotion(Path(temporary) / "fixture")
            path = fixture.prefilter / "N4096" / "coefficients.npy"
            path.write_bytes(path.read_bytes() + b"tampered")
            with self.assertRaisesRegex(
                promoter.PromotionError, "prefilter bundle N4096 SHA"
            ):
                fixture.promote()
            self.assertFalse(fixture.release.exists())

    def test_fusion_directory_hash_is_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticPromotion(Path(temporary) / "fixture")
            path = fixture.classifier_fusion / "feature_mean.npy"
            path.write_bytes(path.read_bytes() + b"tampered")
            with self.assertRaisesRegex(
                promoter.PromotionError, "classifier fusion.*differs"
            ):
                fixture.promote()
            self.assertFalse(fixture.release.exists())

    def test_verified_staging_bytes_cannot_be_swapped_on_second_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticPromotion(Path(temporary) / "fixture")
            path = fixture.staging / promoter.CLASSIFIER_WEIGHTS
            verified = path.read_bytes()
            substituted_payload = json.loads(verified.decode("utf-8"))
            substituted_payload["unverified_adversarial_field"] = True
            substituted = json_bytes(substituted_payload)
            original_read = promoter._read_file
            reads = 0

            def interleaved_read(candidate: Path, label: str) -> bytes:
                nonlocal reads
                if candidate == path:
                    reads += 1
                    if reads == 2:
                        return substituted
                return original_read(candidate, label)

            with mock.patch.object(
                promoter, "_read_file", side_effect=interleaved_read
            ):
                manifest = fixture.promote()
            self.assertEqual(reads, 1)
            self.assertEqual(manifest["status"], promoter.RELEASE_STATUS)
            released = read_json(
                fixture.release / promoter.CLASSIFIER_WEIGHTS
            )
            self.assertNotIn("unverified_adversarial_field", released)

    def test_parent_inode_swap_is_refused_and_rolled_back(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticPromotion(Path(temporary) / "fixture")
            original_open = promoter._open_directory_fd
            moved_root = fixture.root.with_name(f"{fixture.root.name}-moved")
            calls = 0

            def swapping_open(path: Path, label: str) -> int:
                nonlocal calls
                if path == fixture.release.parent:
                    calls += 1
                    if calls == 2:
                        fixture.root.rename(moved_root)
                        fixture.root.mkdir()
                return original_open(path, label)

            try:
                with mock.patch.object(
                    promoter,
                    "_open_directory_fd",
                    side_effect=swapping_open,
                ):
                    with self.assertRaisesRegex(
                        promoter.PromotionError, "parent identity changed"
                    ):
                        fixture.promote()
                self.assertFalse(fixture.release.exists())
                self.assertFalse(
                    (moved_root / fixture.release.name).exists()
                )
            finally:
                if fixture.root.exists():
                    fixture.root.rmdir()
                if moved_root.exists():
                    moved_root.rename(fixture.root)

    def test_candidate_sha_mismatch_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticPromotion(Path(temporary) / "fixture")
            fixture.report_payload["candidate"]["sha256"] = digest(
                "different candidate"
            )
            fixture.rewrite_report()
            with self.assertRaisesRegex(
                promoter.PromotionError, "candidate identity"
            ):
                fixture.promote()
            self.assertFalse(fixture.release.exists())

    def test_cross_role_bundle_link_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticPromotion(Path(temporary) / "fixture")
            components = fixture.report_payload["candidate"]["components"]
            components["classifier_runtime_bundle"]["manifest_sha256"] = (
                components["rejector_runtime_bundle"]["manifest_sha256"]
            )
            fixture.rewrite_report()
            with self.assertRaisesRegex(
                promoter.PromotionError, "classifier runtime bundle"
            ):
                fixture.promote()
            self.assertFalse(fixture.release.exists())

    def test_intent_protocol_must_match_evaluated_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticPromotion(Path(temporary) / "fixture")
            intent_path = fixture.release_root / "RELEASE_INTENT.json"
            intent = read_json(intent_path)
            intent["evaluation_protocol"]["gates"]["five_shot"] = 0.84
            write_json(intent_path, intent)
            manifest_path = fixture.release_root / "RELEASE_MANIFEST.json"
            manifest = read_json(manifest_path)
            manifest["release_intent_sha256"] = digest_bytes(
                intent_path.read_bytes()
            )
            write_json(manifest_path, manifest)
            fixture.report_payload["provenance"][
                "release_intent_sha256"
            ] = digest_bytes(intent_path.read_bytes())
            fixture.report_payload["provenance"][
                "release_manifest_sha256"
            ] = digest_bytes(manifest_path.read_bytes())
            fixture.rewrite_report()
            with self.assertRaisesRegex(
                promoter.PromotionError, "candidate chain"
            ):
                fixture.promote()

    def test_legacy_package_schema_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticPromotion(Path(temporary) / "fixture")
            package = read_json(fixture.staging_manifest)
            package["schema"] = (
                "atomos.v3.time-domain-classifier.runtime-package"
            )
            write_json(fixture.staging_manifest, package)
            with self.assertRaisesRegex(
                promoter.PromotionError, "schema/status/candidate"
            ):
                fixture.promote()
            self.assertFalse(fixture.release.exists())

    def test_nonempty_destination_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticPromotion(Path(temporary) / "fixture")
            fixture.release.mkdir()
            sentinel = fixture.release / "mine.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(
                promoter.PromotionError, "absent or empty"
            ):
                fixture.promote()
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
