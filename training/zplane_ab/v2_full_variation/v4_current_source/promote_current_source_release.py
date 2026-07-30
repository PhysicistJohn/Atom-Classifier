"""Promote validated v4 classifier/policy bytes into a production release pair.

Promotion is impossible unless both design and independent-validation reports
are complete, all gates pass, their manifests reproduce, and the immutable
validation ledger still exists with the expected hash. Numeric model/policy
payloads are preserved exactly; only admission/evidence metadata and the
policy's binding to the newly encoded release classifier are changed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
V2 = HERE.parent
ZPLANE = V2.parent
TRAINING = ZPLANE.parent
for path in (TRAINING, ZPLANE, V2, V2 / "v3_scale", HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import calibrate_current_source_openset as openset  # noqa: E402
import export_current_source_browser_runtime as browser_export  # noqa: E402
import fit_current_source_display_calibration as display_calibration  # noqa: E402


CLASSIFIER_NAME = browser_export.WEIGHTS_NAME
POLICY_NAME = openset.POLICY_NAME
DISPLAY_CALIBRATION_NAME = display_calibration.CALIBRATION_NAME
RUNTIME_MANIFEST_NAME = "runtime-package-manifest.json"
RUNTIME_MANIFEST_SCHEMA = (
    "atomos.v4.time-domain-profile-bank.runtime-package"
)
RELEASE_EVIDENCE_NAME = "release-evidence.json"
RELEASE_EVIDENCE_SCHEMA = (
    "atomos.v4.time-domain-current-source.release-evidence"
)
RELEASE_PROMOTION_SCHEMA = (
    "atomos.v4.time-domain-current-source.validated-release-promotion-v1"
)


def _load_json(path: Path, role: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {role} JSON {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must contain a JSON object")
    return value


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _verify_manifest_file(
    manifest: Mapping[str, Any],
    path: Path,
    *,
    manifest_role: str,
) -> None:
    files = manifest.get("files_sha256")
    if (
        not isinstance(files, Mapping)
        or files.get(path.name) != _sha256(path)
    ):
        raise ValueError(
            f"{manifest_role} manifest does not reproduce {path.name}"
        )


def _numeric_leaves(value: Any, prefix: str = "") -> dict[str, int | float]:
    output: dict[str, int | float] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            output.update(
                _numeric_leaves(item, f"{prefix}.{key}" if prefix else str(key))
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            output.update(_numeric_leaves(item, f"{prefix}[{index}]"))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        output[prefix] = value
    return output


def _promotion_record(
    *,
    validated_classifier_sha256: str,
    design_policy_sha256: str,
    design_report_sha256: str,
    validation_report_sha256: str,
    validation_ledger_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": RELEASE_PROMOTION_SCHEMA,
        "validated_classifier_sha256": validated_classifier_sha256,
        "validated_design_policy_sha256": design_policy_sha256,
        "design_report_sha256": design_report_sha256,
        "independent_validation_report_sha256": validation_report_sha256,
        "immutable_validation_ledger_sha256": validation_ledger_sha256,
        "numeric_payload_preserved": True,
        "uses_frequency_transform": False,
    }


def build_release_pair(
    *,
    classifier_path: Path,
    policy_path: Path,
    design_report_path: Path,
    design_manifest_path: Path,
    validation_report_path: Path,
    validation_manifest_path: Path,
    validation_ledger_path: Path,
    display_calibration_path: Path,
    display_calibration_evidence_path: Path,
) -> tuple[bytes, bytes, bytes, dict[str, Any], dict[str, Any]]:
    """Verify evidence and return deterministic release classifier/policy bytes."""
    classifier = _load_json(classifier_path, "staging classifier")
    raw_policy = _load_json(policy_path, "design policy")
    policy = openset.validate_policy(raw_policy)
    design_report = _load_json(design_report_path, "design report")
    design_manifest = _load_json(design_manifest_path, "design manifest")
    validation_report = _load_json(
        validation_report_path, "independent validation report"
    )
    validation_manifest = _load_json(
        validation_manifest_path, "validation manifest"
    )
    validation_ledger = _load_json(
        validation_ledger_path, "immutable validation ledger"
    )
    display = _load_json(
        display_calibration_path, "staging display calibration"
    )
    display_evidence = _load_json(
        display_calibration_evidence_path,
        "display-calibration fit evidence",
    )

    classifier_sha = _sha256(classifier_path)
    policy_sha = _sha256(policy_path)
    design_report_sha = _sha256(design_report_path)
    validation_report_sha = _sha256(validation_report_path)
    ledger_sha = _sha256(validation_ledger_path)
    display_sha = _sha256(display_calibration_path)
    if (
        classifier.get("schema") != browser_export.BROWSER_SCHEMA
        or classifier.get("schema_version")
        != browser_export.BROWSER_SCHEMA_VERSION
        or classifier.get("status") != "staging_not_release"
        or classifier.get("development_only") is not True
        or classifier.get("release_evidence") is not False
        or policy["classifier_binding"][
            "browser_classifier_asset_sha256"
        ] != classifier_sha
    ):
        raise ValueError("staging classifier/policy binding is invalid")
    if (
        design_report.get("schema") != openset.REPORT_SCHEMA
        or design_report.get("schema_version")
        != openset.REPORT_SCHEMA_VERSION
        or design_report.get("role") != "design"
        or design_report.get("status") != "complete"
        or design_report.get("policy_sha256") != policy_sha
        or design_report.get("gate_summary", {}).get("all_pass") is not True
        or validation_report.get("schema") != openset.REPORT_SCHEMA
        or validation_report.get("schema_version")
        != openset.REPORT_SCHEMA_VERSION
        or validation_report.get("role") != "validate"
        or validation_report.get("status") != "complete"
        or validation_report.get("policy_sha256") != policy_sha
        or validation_report.get("gate_summary", {}).get("all_pass") is not True
    ):
        raise ValueError("design/independent-validation gates are incomplete")
    _verify_manifest_file(
        design_manifest,
        policy_path,
        manifest_role="design",
    )
    _verify_manifest_file(
        design_manifest,
        design_report_path,
        manifest_role="design",
    )
    _verify_manifest_file(
        validation_manifest,
        validation_report_path,
        manifest_role="validation",
    )
    report_ledger = validation_report.get("validation_ledger")
    if (
        not isinstance(report_ledger, Mapping)
        or report_ledger.get("sha256") != ledger_sha
        or report_ledger.get("state") != "validation_evidence_claimed"
        or report_ledger.get("created_before_validation_scoring") is not True
        or report_ledger.get("absence_or_replacement_invalidates_evidence")
        is not True
        or validation_manifest.get("validation_ledger_sha256") != ledger_sha
        or validation_ledger.get("state") != "validation_evidence_claimed"
        or validation_ledger.get("policy_sha256") != policy_sha
        or validation_ledger.get(
            "deletion_or_replacement_invalidates_evidence"
        ) is not True
        or validation_ledger.get("validation_protocol")
        != policy["validation_protocol"]
    ):
        raise ValueError("immutable validation-ledger evidence is invalid")
    expected_display_keys = {
        "schema",
        "schema_version",
        "status",
        "development_only",
        "runtime_role",
        "decision_authority",
        "calibration_kind",
        "classifier_asset_sha256",
        "distance_logit_scale_by_prototype_source",
    }
    display_scales = display.get(
        "distance_logit_scale_by_prototype_source"
    )
    if (
        set(display) != expected_display_keys
        or display.get("schema") != display_calibration.CALIBRATION_SCHEMA
        or display.get("schema_version")
        != display_calibration.CALIBRATION_SCHEMA_VERSION
        or display.get("status") != "staging_not_release"
        or display.get("development_only") is not True
        or display.get("runtime_role")
        != "conditional_display_calibration"
        or display.get("decision_authority") is not False
        or display.get("calibration_kind")
        != display_calibration.CALIBRATION_KIND
        or display.get("classifier_asset_sha256") != classifier_sha
        or not isinstance(display_scales, Mapping)
        or set(display_scales) != set(openset.PROTOTYPE_SOURCE_ROUTES)
        or any(
            isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or not math.isfinite(float(scale))
            or float(scale) <= 0.0
            for scale in display_scales.values()
        )
        or display_evidence.get("schema")
        != display_calibration.EVIDENCE_SCHEMA
        or display_evidence.get("schema_version")
        != display_calibration.EVIDENCE_SCHEMA_VERSION
        or display_evidence.get("classifier_asset_sha256") != classifier_sha
        or display_evidence.get("display_calibration_asset_sha256")
        != display_sha
        or display_evidence.get("decision_authority") is not False
        or display_evidence.get("argmin_and_open_set_decisions_changed")
        is not False
        or display_evidence.get("per_prototype_source") is None
    ):
        raise ValueError("display-calibration artifact/evidence is invalid")

    promotion = _promotion_record(
        validated_classifier_sha256=classifier_sha,
        design_policy_sha256=policy_sha,
        design_report_sha256=design_report_sha,
        validation_report_sha256=validation_report_sha,
        validation_ledger_sha256=ledger_sha,
    )
    release_classifier = deepcopy(classifier)
    release_classifier["status"] = "release"
    release_classifier["development_only"] = False
    release_classifier["release_evidence"] = True
    classifier_provenance = release_classifier.get("provenance")
    if not isinstance(classifier_provenance, dict):
        raise ValueError("classifier provenance is missing")
    classifier_provenance["release_promotion"] = promotion
    if _numeric_leaves(classifier) != _numeric_leaves(release_classifier):
        raise AssertionError("classifier numeric payload changed during promotion")
    release_classifier_bytes = _json_bytes(release_classifier)
    release_classifier_sha = _sha256_bytes(release_classifier_bytes)

    release_policy = deepcopy(policy)
    release_policy["status"] = "release"
    release_policy["development_only"] = False
    release_policy["release_evidence"] = True
    release_policy["classifier_binding"][
        "browser_classifier_asset_sha256"
    ] = release_classifier_sha
    release_policy["validation_protocol"][
        "classifier_asset_sha256"
    ] = release_classifier_sha
    policy_provenance = release_policy.get("provenance")
    if not isinstance(policy_provenance, dict):
        raise ValueError("policy provenance is missing")
    policy_provenance["release_promotion"] = promotion
    if _numeric_leaves(policy) != _numeric_leaves(release_policy):
        raise AssertionError("policy numeric payload changed during promotion")
    release_policy_bytes = _json_bytes(release_policy)
    release_policy_sha = _sha256_bytes(release_policy_bytes)

    release_display = deepcopy(display)
    release_display["status"] = "release"
    release_display["development_only"] = False
    release_display["classifier_asset_sha256"] = release_classifier_sha
    if set(release_display) != expected_display_keys:
        raise AssertionError("release display-calibration keys changed")
    if _numeric_leaves(display) != _numeric_leaves(release_display):
        raise AssertionError(
            "display-calibration numeric payload changed during promotion"
        )
    release_display_bytes = _json_bytes(release_display)
    release_display_sha = _sha256_bytes(release_display_bytes)

    runtime_manifest = {
        "schema": RUNTIME_MANIFEST_SCHEMA,
        "schema_version": 1,
        "status": "release",
        "development_only": False,
        "architecture": {
            "classifier_and_policy_bytes_mutually_bound": True,
            "trusted_prototype_source_required": True,
            "classifier_precedes_route_conditioned_abstention": True,
            "display_calibration_has_decision_authority": False,
            "runtime_length_policy_by_prototype_source": deepcopy(
                release_policy[
                    "runtime_length_policy_by_prototype_source"
                ]
            ),
        },
        "assets": {
            CLASSIFIER_NAME: {
                "path": CLASSIFIER_NAME,
                "sha256": release_classifier_sha,
                "bytes": len(release_classifier_bytes),
                "schema": browser_export.BROWSER_SCHEMA,
                "schema_version": browser_export.BROWSER_SCHEMA_VERSION,
                "runtime_role": "accepted_known_classifier",
                "required_for_decision": True,
            },
            POLICY_NAME: {
                "path": POLICY_NAME,
                "sha256": release_policy_sha,
                "bytes": len(release_policy_bytes),
                "schema": openset.POLICY_SCHEMA,
                "schema_version": openset.POLICY_SCHEMA_VERSION,
                "runtime_role": "external_abstention_policy",
                "required_for_decision": True,
            },
            DISPLAY_CALIBRATION_NAME: {
                "path": DISPLAY_CALIBRATION_NAME,
                "sha256": release_display_sha,
                "bytes": len(release_display_bytes),
                "schema": display_calibration.CALIBRATION_SCHEMA,
                "schema_version":
                    display_calibration.CALIBRATION_SCHEMA_VERSION,
                "runtime_role": "conditional_display_calibration",
                "required_for_decision": False,
            },
        },
    }
    release_evidence = {
        "schema": RELEASE_EVIDENCE_SCHEMA,
        "schema_version": 1,
        "status": "release",
        "development_only": False,
        "release_evidence": True,
        "runtime_package_manifest_sha256":
            _sha256_bytes(_json_bytes(runtime_manifest)),
        "validated_staging_assets": {
            CLASSIFIER_NAME: classifier_sha,
            POLICY_NAME: policy_sha,
            DISPLAY_CALIBRATION_NAME: display_sha,
        },
        "promotion": promotion,
        "display_calibration_evidence": display_evidence,
        "routing": browser_export.TRUSTED_SOURCE_ROUTING,
        "supported_public_class_mask_by_prototype_source": {
            source: list(
                openset.EXPECTED_PUBLIC_CLASS_SUPPORT_BY_PROTOTYPE_SOURCE[
                    source
                ]
            )
            for source in openset.PROTOTYPE_SOURCE_ROUTES
        },
        "uses_frequency_transform": False,
        "numeric_payload_preserved": True,
    }
    return (
        release_classifier_bytes,
        release_policy_bytes,
        release_display_bytes,
        runtime_manifest,
        release_evidence,
    )


def _write_exclusive(path: Path, value: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        view = memoryview(value)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("release write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def promote(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"release output already exists: {output}")
    (
        classifier_bytes,
        policy_bytes,
        display_bytes,
        runtime_manifest,
        release_evidence,
    ) = build_release_pair(
        classifier_path=Path(args.classifier).expanduser().resolve(),
        policy_path=Path(args.policy).expanduser().resolve(),
        design_report_path=Path(args.design_report).expanduser().resolve(),
        design_manifest_path=Path(args.design_manifest).expanduser().resolve(),
        validation_report_path=
            Path(args.validation_report).expanduser().resolve(),
        validation_manifest_path=
            Path(args.validation_manifest).expanduser().resolve(),
        validation_ledger_path=
            Path(args.validation_ledger).expanduser().resolve(),
        display_calibration_path=
            Path(args.display_calibration).expanduser().resolve(),
        display_calibration_evidence_path=
            Path(args.display_calibration_evidence).expanduser().resolve(),
    )
    output.mkdir(parents=True, exist_ok=False)
    classifier_output = output / CLASSIFIER_NAME
    policy_output = output / POLICY_NAME
    display_output = output / DISPLAY_CALIBRATION_NAME
    manifest_output = output / RUNTIME_MANIFEST_NAME
    evidence_output = output / RELEASE_EVIDENCE_NAME
    _write_exclusive(classifier_output, classifier_bytes)
    _write_exclusive(policy_output, policy_bytes)
    _write_exclusive(display_output, display_bytes)
    _write_exclusive(manifest_output, _json_bytes(runtime_manifest))
    _write_exclusive(evidence_output, _json_bytes(release_evidence))
    if (
        _sha256(classifier_output)
        != runtime_manifest["assets"][CLASSIFIER_NAME]["sha256"]
        or _sha256(policy_output)
        != runtime_manifest["assets"][POLICY_NAME]["sha256"]
        or _sha256(display_output)
        != runtime_manifest["assets"][DISPLAY_CALIBRATION_NAME]["sha256"]
    ):
        raise RuntimeError("written release bytes disagree with manifest")
    print(f"[v4 release promotion] wrote {output}", flush=True)
    return runtime_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--design-report", required=True)
    parser.add_argument("--design-manifest", required=True)
    parser.add_argument("--validation-report", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--validation-ledger", required=True)
    parser.add_argument("--display-calibration", required=True)
    parser.add_argument("--display-calibration-evidence", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    promote(build_parser().parse_args())


if __name__ == "__main__":
    main()
