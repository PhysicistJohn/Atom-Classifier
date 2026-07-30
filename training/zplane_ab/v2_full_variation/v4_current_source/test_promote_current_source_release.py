from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import calibrate_current_source_openset as openset
import export_current_source_browser_runtime as browser_export
import promote_current_source_release as promotion
from test_calibrate_current_source_openset import _fit_policy


def _write(path: Path, value: object) -> None:
    path.write_bytes(openset._json_bytes(value))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence_tree(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    classifier_path = root / browser_export.WEIGHTS_NAME
    classifier = {
        "schema": browser_export.BROWSER_SCHEMA,
        "schema_version": browser_export.BROWSER_SCHEMA_VERSION,
        "status": "staging_not_release",
        "development_only": True,
        "release_evidence": False,
        "numeric_model_payload": [1.25, 2, -3.5],
        "provenance": {
            "source_fusion_dev_metrics_sha256": "b" * 64,
            "source_fusion_artifacts_sha256": {
                "fusion_state_dict.pt": "c" * 64,
            },
        },
    }
    _write(classifier_path, classifier)
    classifier_sha = _sha(classifier_path)
    display_path = root / promotion.DISPLAY_CALIBRATION_NAME
    display = {
        "schema": promotion.display_calibration.CALIBRATION_SCHEMA,
        "schema_version":
            promotion.display_calibration.CALIBRATION_SCHEMA_VERSION,
        "status": "staging_not_release",
        "development_only": True,
        "runtime_role": "conditional_display_calibration",
        "decision_authority": False,
        "calibration_kind": promotion.display_calibration.CALIBRATION_KIND,
        "classifier_asset_sha256": classifier_sha,
        "distance_logit_scale_by_prototype_source": {
            "current": 10.5,
            "historical": 7.8,
        },
    }
    _write(display_path, display)
    display_evidence_path = root / "display-calibration-evidence.json"
    _write(
        display_evidence_path,
        {
            "schema": promotion.display_calibration.EVIDENCE_SCHEMA,
            "schema_version":
                promotion.display_calibration.EVIDENCE_SCHEMA_VERSION,
            "classifier_asset_sha256": classifier_sha,
            "display_calibration_asset_sha256": _sha(display_path),
            "decision_authority": False,
            "argmin_and_open_set_decisions_changed": False,
            "per_prototype_source": {
                "current": {"selection_evaluation_only": {}},
                "historical": {"selection_evaluation_only": {}},
            },
        },
    )

    policy = _fit_policy()
    policy["classifier_binding"][
        "browser_classifier_asset_sha256"
    ] = classifier_sha
    policy["validation_protocol"][
        "classifier_asset_sha256"
    ] = classifier_sha
    policy = openset.validate_policy(policy)
    policy_path = root / openset.POLICY_NAME
    _write(policy_path, policy)
    policy_sha = _sha(policy_path)

    design_report_path = root / openset.DESIGN_REPORT_NAME
    design_report = {
        "schema": openset.REPORT_SCHEMA,
        "schema_version": openset.REPORT_SCHEMA_VERSION,
        "status": "complete",
        "role": "design",
        "policy_sha256": policy_sha,
        "gate_summary": {"all_pass": True},
    }
    _write(design_report_path, design_report)
    design_manifest_path = root / "design-manifest.json"
    _write(
        design_manifest_path,
        {
            "files_sha256": {
                policy_path.name: _sha(policy_path),
                design_report_path.name: _sha(design_report_path),
            }
        },
    )

    ledger_path = root / "validation-ledger.json"
    ledger = {
        "state": "validation_evidence_claimed",
        "policy_sha256": policy_sha,
        "validation_protocol": policy["validation_protocol"],
        "deletion_or_replacement_invalidates_evidence": True,
    }
    _write(ledger_path, ledger)
    ledger_sha = _sha(ledger_path)

    validation_report_path = root / openset.VALIDATION_REPORT_NAME
    validation_report = {
        "schema": openset.REPORT_SCHEMA,
        "schema_version": openset.REPORT_SCHEMA_VERSION,
        "status": "complete",
        "role": "validate",
        "policy_sha256": policy_sha,
        "gate_summary": {"all_pass": True},
        "validation_ledger": {
            "sha256": ledger_sha,
            "state": "validation_evidence_claimed",
            "created_before_validation_scoring": True,
            "absence_or_replacement_invalidates_evidence": True,
        },
    }
    _write(validation_report_path, validation_report)
    validation_manifest_path = root / "validation-manifest.json"
    _write(
        validation_manifest_path,
        {
            "files_sha256": {
                validation_report_path.name: _sha(validation_report_path),
            },
            "validation_ledger_sha256": ledger_sha,
        },
    )
    return {
        "classifier_path": classifier_path,
        "policy_path": policy_path,
        "design_report_path": design_report_path,
        "design_manifest_path": design_manifest_path,
        "validation_report_path": validation_report_path,
        "validation_manifest_path": validation_manifest_path,
        "validation_ledger_path": ledger_path,
        "display_calibration_path": display_path,
        "display_calibration_evidence_path": display_evidence_path,
    }


class ReleasePromotionTests(unittest.TestCase):
    def test_promotion_preserves_numeric_payload_and_rebinds_release_pair(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _evidence_tree(Path(temporary))
            (
                classifier_bytes,
                policy_bytes,
                display_bytes,
                runtime_manifest,
                release_evidence,
            ) = (
                promotion.build_release_pair(**paths)
            )
            classifier = json.loads(classifier_bytes)
            policy = json.loads(policy_bytes)
            display = json.loads(display_bytes)
            self.assertEqual(classifier["status"], "release")
            self.assertIs(classifier["development_only"], False)
            self.assertIs(classifier["release_evidence"], True)
            self.assertEqual(
                classifier["numeric_model_payload"], [1.25, 2, -3.5]
            )
            classifier_sha = hashlib.sha256(classifier_bytes).hexdigest()
            self.assertEqual(
                policy["classifier_binding"][
                    "browser_classifier_asset_sha256"
                ],
                classifier_sha,
            )
            self.assertEqual(
                policy["validation_protocol"]["classifier_asset_sha256"],
                classifier_sha,
            )
            self.assertEqual(
                runtime_manifest["assets"][
                    browser_export.WEIGHTS_NAME
                ]["sha256"],
                classifier_sha,
            )
            self.assertEqual(
                set(runtime_manifest["assets"]),
                {
                    browser_export.WEIGHTS_NAME,
                    openset.POLICY_NAME,
                    promotion.DISPLAY_CALIBRATION_NAME,
                },
            )
            self.assertEqual(
                runtime_manifest["architecture"][
                    "runtime_length_policy_by_prototype_source"
                ],
                policy[
                    "runtime_length_policy_by_prototype_source"
                ],
            )
            self.assertEqual(
                runtime_manifest["architecture"][
                    "runtime_length_policy_by_prototype_source"
                ],
                openset.RUNTIME_LENGTH_POLICY_BY_PROTOTYPE_SOURCE,
            )
            self.assertEqual(display["status"], "release")
            self.assertIs(display["development_only"], False)
            self.assertEqual(
                display["classifier_asset_sha256"], classifier_sha
            )
            self.assertIs(
                runtime_manifest["assets"][
                    promotion.DISPLAY_CALIBRATION_NAME
                ]["required_for_decision"],
                False,
            )
            self.assertIs(release_evidence["numeric_payload_preserved"], True)

    def test_promotion_refuses_failed_validation_and_missing_ledger(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _evidence_tree(Path(temporary))
            report_path = paths["validation_report_path"]
            report = json.loads(report_path.read_text())
            report["gate_summary"]["all_pass"] = False
            _write(report_path, report)
            manifest_path = paths["validation_manifest_path"]
            manifest = json.loads(manifest_path.read_text())
            manifest["files_sha256"][report_path.name] = _sha(report_path)
            _write(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "gates"):
                promotion.build_release_pair(**paths)

            paths = _evidence_tree(Path(temporary) / "second")
            paths["validation_ledger_path"].unlink()
            with self.assertRaisesRegex(ValueError, "cannot read"):
                promotion.build_release_pair(**paths)


if __name__ == "__main__":
    unittest.main()
