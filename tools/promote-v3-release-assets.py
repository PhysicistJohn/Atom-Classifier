#!/usr/bin/env python3
"""Fail-closed promotion of the tracked v3 staging runtime package.

This tool does not evaluate a model, open a sealed corpus, fit or recalibrate
anything, modify the live v2 assets, or deploy.  It accepts only the exact
tracked staging package and a completed seed-20260735 v3 release-evaluation
report.  Before materializing a distinct release package it verifies:

* every staging byte against the tracked package/export manifests;
* the staging files are committed and unchanged in Git;
* all 23 strict v2 release gates, including five-shot >= 0.85 and known
  false-unknown <= 0.10;
* the report's exact runtime-bundle, fusion, staged-policy, and prefilter
  hashes against provenance embedded in all three cooperating runtime assets;
* no development data was loaded by the evaluator, no training or
  recalibration occurred, and the exported candidate consumed no sealed/test
  rows before the one-shot release evaluation.

Promotion changes only the top-level ``status`` of the fusion, classifier, and
open-set assets from ``staging_not_release`` to ``release``.  The release
package manifest binds the exact source-package and evaluation-report bytes.
No timestamp or absolute path is emitted, so identical inputs produce
byte-identical release packages.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
DEFAULT_STAGING_PACKAGE = REPO / "src/embedding/assets-v3-staging"
DEFAULT_RELEASE_PACKAGE = REPO / "src/embedding/assets-v3-release"
LIVE_V2_ASSETS = REPO / "src/embedding/assets"

PACKAGE_MANIFEST = "runtime-package-manifest.json"
FUSION_EXPORT_MANIFEST = "export-manifest.json"
FUSION_WEIGHTS = "time-domain-fusion-weights-v3.json"
CLASSIFIER_WEIGHTS = "time-domain-classifier-weights-v1.json"
OPENSET_WEIGHTS = "time-domain-openset-weights-v1.json"
OPENSET_SMOKE = "time-domain-openset-smoke-v1.json"
FUSION_PROBE = "time-domain-probe-fixture-v3.json"

ASSET_NAMES = (
    CLASSIFIER_WEIGHTS,
    FUSION_WEIGHTS,
    OPENSET_SMOKE,
    OPENSET_WEIGHTS,
    FUSION_PROBE,
)
STATUS_ASSET_NAMES = (
    FUSION_WEIGHTS,
    CLASSIFIER_WEIGHTS,
    OPENSET_WEIGHTS,
)
STAGING_FILE_NAMES = frozenset(
    (*ASSET_NAMES, PACKAGE_MANIFEST, FUSION_EXPORT_MANIFEST)
)

PACKAGE_SCHEMA = "atomos.v3.time-domain-classifier.runtime-package"
FUSION_EXPORT_SCHEMA = (
    "atomos.v3.time-domain-invariant-fusion.browser-weights.export-manifest"
)
ASSET_CONTRACTS = {
    FUSION_WEIGHTS: (
        "atomos.v3.time-domain-invariant-fusion.browser-weights",
        1,
    ),
    CLASSIFIER_WEIGHTS: (
        "atomos.v3.time-domain-invariant-fusion.browser-decision",
        1,
    ),
    OPENSET_WEIGHTS: ("atomos.v3.time-domain-openset.staged", 2),
}

STAGING_STATUS = "staging_not_release"
RELEASE_STATUS = "release"
RELEASE_SEED = 20260735
HISTORICAL_RELAXED_SEED = 20260734
EVALUATOR_SCHEMA = 2
EVALUATION_VERSION = "time-domain-v3-release-evaluation-v2"
RUNTIME_BUNDLE_SCHEMA = "atomos.v3.time-domain-invariant-fusion.runtime-bundle"
RUNTIME_BUNDLE_KIND = "v3-time-domain-centered-invariant-fusion"
RUNTIME_BUNDLE_SCHEMA_VERSION = 1
STAGED_STATUS = "development_openset_pass"
STAGED_POLICY_SCHEMA = 2
STAGED_POLICY_VERSION = "v3-staged-openset-policy-v2-composite-survivor"
STAGED_POLICY_KIND = (
    "v3_staged_noise_prefilter_then_composite_survivor_lof_geometry"
)

EXPECTED_BUNDLE_ASSETS = frozenset(
    {
        "complex_center.npy",
        "feature_mean.npy",
        "feature_std.npy",
        "fusion_prototypes.npy",
        "fusion_state_dict.pt",
        "probe_fixture.json",
        "real_center.npy",
    }
)
EXPECTED_STAGED_ASSETS = frozenset(
    {
        "v3_branch_lof_components.npz",
        "v3_open_policy_stage_two.npz",
        "v3_staged_composite_policy.npz",
    }
)

# This is the exact imported-v2 gate-floor object embedded by
# evaluate_v3_release_suite.expected_evaluation_protocol.  The historical
# seed-20260734 0.84/0.12 redeclaration is intentionally not accepted here.
STRICT_V2_GATE_FLOORS = {
    "closed_fine": 0.72,
    "closed_family": 0.82,
    "closed_high_snr": 0.78,
    "closed_clean": 0.85,
    "five_shot": 0.85,
    "open_auroc_overall": 0.72,
    "open_auroc_noise": 0.80,
    "open_auroc_chirp": 0.80,
    "open_known_false_unknown_max": 0.10,
    "open_unknown_recall_noise": 0.10,
    "open_unknown_recall_chirp": 0.10,
    "invariant_worst_balanced_accuracy": 0.75,
    "invariant_max_mean_pairwise_cosine": 0.78,
    "length_pair_prediction_agreement": 0.80,
    "length_pair_embedding_cosine": 0.85,
    "scale_pair_prediction_agreement": 0.75,
    "scale_pair_embedding_cosine": 0.80,
}

BOOLEAN_GATE_NAMES = frozenset(
    {
        "prefix_nesting",
        "dependency_provenance",
        "start_probe_excluded",
    }
)
CANDIDATE_GATE_NAME = "candidate_sha_bound"
NUMERIC_GATE_CONTRACT = {
    "closed_fine_worst_length": (0.72, "min"),
    "closed_family_worst_length": (0.82, "min"),
    "closed_high_snr_worst_length": (0.78, "min"),
    "closed_clean_worst_length": (0.85, "min"),
    "five_shot_worst_length_balanced": (0.85, "min"),
    "open_auroc_overall_worst_length": (0.72, "min"),
    "open_auroc_noise_worst_length": (0.80, "min"),
    "open_auroc_chirp_worst_length": (0.80, "min"),
    "open_known_false_unknown_worst_length": (0.10, "max"),
    "open_unknown_recall_noise_worst_length": (0.10, "min"),
    "open_unknown_recall_chirp_worst_length": (0.10, "min"),
    "length_worst_balanced_accuracy": (0.75, "min"),
    "length_max_mean_pairwise_cosine": (0.78, "max"),
    "length_worst_prediction_agreement_to_matched": (0.80, "min"),
    "length_worst_embedding_cosine_to_matched": (0.85, "min"),
    "physical_scale_worst_balanced_accuracy": (0.75, "min"),
    "physical_scale_max_mean_pairwise_cosine": (0.78, "max"),
    "physical_scale_worst_prediction_agreement_to_factor1": (0.75, "min"),
    "physical_scale_worst_embedding_cosine_to_factor1": (0.80, "min"),
}
EXPECTED_GATE_NAMES = frozenset(
    {
        *BOOLEAN_GATE_NAMES,
        CANDIDATE_GATE_NAME,
        *NUMERIC_GATE_CONTRACT,
    }
)
if len(EXPECTED_GATE_NAMES) != 23:  # pragma: no cover - import-time invariant
    raise RuntimeError("the frozen release gate contract must contain 23 gates")

SHA256_RE = re.compile(r"[0-9a-f]{64}")


class PromotionError(RuntimeError):
    """The supplied package/report cannot safely be promoted."""


@dataclass(frozen=True)
class PromotionPolicy:
    """Paths and Git policy that define one allowed promotion channel."""

    repo_root: Path
    staging_package: Path
    release_package: Path
    live_v2_assets: Path
    require_git_tracking: bool = True


@dataclass(frozen=True)
class VerifiedStaging:
    manifest: dict[str, Any]
    manifest_sha256: str
    asset_bytes: dict[str, bytes]
    asset_payloads: dict[str, dict[str, Any]]
    tracked_paths: tuple[Path, ...]


@dataclass(frozen=True)
class VerifiedEvaluation:
    report_sha256: str
    candidate_sha256: str
    candidate_artifacts: dict[str, Any]
    release_intent_sha256: str
    release_manifest_sha256: str


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_exact_int(value: Any) -> bool:
    return type(value) is int


def _is_finite_number(value: Any) -> bool:
    return (
        type(value) in (int, float)
        and math.isfinite(float(value))
    )


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise PromotionError(f"{label} must be a lowercase SHA-256")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PromotionError(f"{label} must be an object")
    return value


def _exact_object_keys(
    value: Mapping[str, Any], expected: set[str] | frozenset[str], label: str
) -> None:
    actual = set(value)
    if actual != set(expected):
        raise PromotionError(
            f"{label} keys differ: expected {sorted(expected)}, got "
            f"{sorted(actual)}"
        )


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PromotionError(f"JSON object repeats key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise PromotionError(f"non-finite JSON constant {value!r} is forbidden")


def _parse_json(raw: bytes, label: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PromotionError(f"{label} is not UTF-8 JSON") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_no_duplicate_object,
            parse_constant=_reject_json_constant,
        )
    except PromotionError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PromotionError(f"{label} is not valid strict JSON") from exc


def _read_regular_file(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise PromotionError(f"{label} must be a regular, non-symlink file")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise PromotionError(f"cannot read {label}: {path}") from exc


def _canonical_json_bytes(payload: Any) -> bytes:
    try:
        text = json.dumps(
            payload,
            indent=1,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:  # pragma: no cover - internal only
        raise PromotionError("cannot serialize deterministic release JSON") from exc
    return f"{text}\n".encode("utf-8")


def _within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _file_record(value: Any, label: str) -> dict[str, Any]:
    record = _mapping(value, label)
    _exact_object_keys(record, {"bytes", "sha256"}, label)
    size = record["bytes"]
    if not _is_exact_int(size) or size <= 0:
        raise PromotionError(f"{label}.bytes must be a positive integer")
    return {
        "bytes": size,
        "sha256": _sha256(record["sha256"], f"{label}.sha256"),
    }


def _hash_map(
    value: Any,
    label: str,
    *,
    exact_names: frozenset[str] | None = None,
) -> dict[str, str]:
    source = _mapping(value, label)
    if exact_names is not None and set(source) != set(exact_names):
        raise PromotionError(
            f"{label} names differ: expected {sorted(exact_names)}, got "
            f"{sorted(source)}"
        )
    if not source:
        raise PromotionError(f"{label} must not be empty")
    return {
        str(name): _sha256(digest, f"{label}[{name!r}]")
        for name, digest in source.items()
    }


def _bundle_asset_records(value: Any) -> dict[str, dict[str, Any]]:
    source = _mapping(value, "report candidate.components.bundle_assets")
    if set(source) != set(EXPECTED_BUNDLE_ASSETS):
        raise PromotionError(
            "report candidate bundle asset names differ from the exact v3 "
            f"runtime bundle: {sorted(source)}"
        )
    return {
        str(name): _file_record(
            record, f"report candidate.components.bundle_assets[{name!r}]"
        )
        for name, record in source.items()
    }


def _run_git(repo: Path, arguments: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise PromotionError("Git is required to verify the staging package") from exc


def _verify_git_tracking(repo: Path, paths: tuple[Path, ...]) -> None:
    top = _run_git(repo, ["rev-parse", "--show-toplevel"])
    if top.returncode != 0:
        raise PromotionError("staging package is not inside a Git worktree")
    try:
        reported_top = Path(top.stdout.strip()).resolve(strict=True)
    except OSError as exc:
        raise PromotionError("Git reported an unusable worktree root") from exc
    if reported_top != repo:
        raise PromotionError(
            f"Git worktree root {reported_top} differs from policy root {repo}"
        )

    relatives: list[str] = []
    for path in paths:
        try:
            relative = path.relative_to(repo)
        except ValueError as exc:
            raise PromotionError(f"tracked source escapes repository: {path}") from exc
        relatives.append(relative.as_posix())

    tracked = _run_git(
        repo, ["ls-files", "--error-unmatch", "--", *sorted(relatives)]
    )
    if tracked.returncode != 0:
        raise PromotionError(
            "every staging package input must be tracked by Git"
        )
    clean = _run_git(repo, ["diff", "--quiet", "HEAD", "--", *sorted(relatives)])
    if clean.returncode == 1:
        raise PromotionError(
            "staging package inputs differ from their committed HEAD bytes"
        )
    if clean.returncode != 0:
        raise PromotionError(
            "cannot prove staging package inputs match committed HEAD bytes"
        )


def _resolved_policy(policy: PromotionPolicy) -> PromotionPolicy:
    try:
        repo = policy.repo_root.expanduser().resolve(strict=True)
    except OSError as exc:
        raise PromotionError("policy repository root does not exist") from exc
    if repo.is_symlink() or not repo.is_dir():
        raise PromotionError("policy repository root must be a regular directory")
    return PromotionPolicy(
        repo_root=repo,
        staging_package=policy.staging_package.expanduser().resolve(),
        release_package=policy.release_package.expanduser().resolve(),
        live_v2_assets=policy.live_v2_assets.expanduser().resolve(),
        require_git_tracking=policy.require_git_tracking,
    )


def _validate_paths(
    staging_package: Path,
    destination: Path,
    policy: PromotionPolicy,
) -> tuple[Path, Path, PromotionPolicy]:
    normalized = _resolved_policy(policy)
    if staging_package.is_symlink():
        raise PromotionError("staging package may not be a symlink")
    staging = staging_package.expanduser().resolve()
    release = destination.expanduser().resolve()
    if staging != normalized.staging_package:
        raise PromotionError(
            "source is not the policy's exact tracked v3 staging package"
        )
    if release != normalized.release_package:
        raise PromotionError(
            "destination is not the policy's exact v3 release package"
        )
    if _within(release, normalized.live_v2_assets):
        raise PromotionError("refusing the live v2 asset directory or a descendant")
    if _within(release, normalized.staging_package):
        raise PromotionError("refusing the staging package or a descendant")
    if _within(normalized.staging_package, release):
        raise PromotionError("release destination may not contain the staging package")
    if staging == release:
        raise PromotionError("release and staging packages must be distinct")
    if release.is_symlink():
        raise PromotionError("release destination may not be a symlink")
    if release.exists():
        if not release.is_dir():
            raise PromotionError("release destination exists and is not a directory")
        try:
            if next(release.iterdir(), None) is not None:
                raise PromotionError("release destination must be empty")
        except OSError as exc:
            raise PromotionError("cannot inspect release destination") from exc
    return staging, release, normalized


def _verify_staging_package(
    staging: Path,
    policy: PromotionPolicy,
) -> VerifiedStaging:
    if staging.is_symlink() or not staging.is_dir():
        raise PromotionError("staging package must be a regular, non-symlink directory")
    try:
        entries = tuple(staging.iterdir())
    except OSError as exc:
        raise PromotionError("cannot enumerate staging package") from exc
    names = {entry.name for entry in entries}
    if names != set(STAGING_FILE_NAMES):
        raise PromotionError(
            "staging package entries differ from the exact tracked package: "
            f"expected {sorted(STAGING_FILE_NAMES)}, got {sorted(names)}"
        )
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            raise PromotionError(
                f"staging entry {entry.name!r} must be a regular file"
            )

    manifest_path = staging / PACKAGE_MANIFEST
    manifest_raw = _read_regular_file(manifest_path, "staging package manifest")
    manifest_value = _parse_json(manifest_raw, "staging package manifest")
    manifest = dict(_mapping(manifest_value, "staging package manifest"))
    if manifest.get("schema") != PACKAGE_SCHEMA:
        raise PromotionError("staging package manifest has an unexpected schema")
    if manifest.get("schema_version") != 1:
        raise PromotionError("staging package manifest schema_version must be 1")
    if manifest.get("status") != STAGING_STATUS:
        raise PromotionError("staging package manifest is not staging_not_release")

    asset_records_source = _mapping(
        manifest.get("assets"), "staging package manifest.assets"
    )
    if set(asset_records_source) != set(ASSET_NAMES):
        raise PromotionError(
            "staging package manifest does not bind the exact five runtime assets"
        )
    asset_records = {
        name: _file_record(
            asset_records_source[name],
            f"staging package manifest.assets[{name!r}]",
        )
        for name in ASSET_NAMES
    }

    asset_bytes: dict[str, bytes] = {}
    for name in ASSET_NAMES:
        raw = _read_regular_file(staging / name, f"staging asset {name}")
        record = asset_records[name]
        if len(raw) != record["bytes"]:
            raise PromotionError(f"staging asset {name} byte count changed")
        if _sha256_bytes(raw) != record["sha256"]:
            raise PromotionError(f"staging asset {name} SHA-256 changed")
        asset_bytes[name] = raw

    sources = _mapping(manifest.get("sources"), "staging package manifest.sources")
    _exact_object_keys(
        sources,
        {
            FUSION_EXPORT_MANIFEST,
            "openset-export-manifest.json",
            "time-domain-openset-parity-v1.json",
        },
        "staging package manifest.sources",
    )
    source_hashes = {
        name: _sha256(value, f"staging package manifest.sources[{name!r}]")
        for name, value in sources.items()
    }

    export_path = staging / FUSION_EXPORT_MANIFEST
    export_raw = _read_regular_file(export_path, "fusion export manifest")
    if _sha256_bytes(export_raw) != source_hashes[FUSION_EXPORT_MANIFEST]:
        raise PromotionError(
            "fusion export manifest differs from the staging source binding"
        )
    export_value = _parse_json(export_raw, "fusion export manifest")
    export_manifest = _mapping(export_value, "fusion export manifest")
    if (
        export_manifest.get("schema") != FUSION_EXPORT_SCHEMA
        or export_manifest.get("schema_version") != 1
    ):
        raise PromotionError("fusion export manifest has an unexpected schema")
    emitted = _mapping(export_manifest.get("emitted"), "fusion export emitted")
    _exact_object_keys(
        emitted,
        {FUSION_WEIGHTS, FUSION_PROBE},
        "fusion export emitted",
    )
    for name in (FUSION_WEIGHTS, FUSION_PROBE):
        if _file_record(
            emitted[name], f"fusion export emitted[{name!r}]"
        ) != asset_records[name]:
            raise PromotionError(
                f"fusion export record for {name} differs from package manifest"
            )

    asset_payloads: dict[str, dict[str, Any]] = {}
    for name, (schema, schema_version) in ASSET_CONTRACTS.items():
        value = _parse_json(asset_bytes[name], f"staging asset {name}")
        payload = dict(_mapping(value, f"staging asset {name}"))
        if payload.get("schema") != schema:
            raise PromotionError(f"staging asset {name} has an unexpected schema")
        if payload.get("schema_version") != schema_version:
            raise PromotionError(
                f"staging asset {name} has an unexpected schema_version"
            )
        if payload.get("status") != STAGING_STATUS:
            raise PromotionError(
                f"staging asset {name} is not staging_not_release"
            )
        asset_payloads[name] = payload

    fusion_payload = asset_payloads[FUSION_WEIGHTS]
    if (
        fusion_payload.get("development_only") is not True
        or fusion_payload.get("release_evidence") is not False
    ):
        raise PromotionError(
            "fusion staging asset does not preserve development-only provenance"
        )

    classifier_provenance = _mapping(
        asset_payloads[CLASSIFIER_WEIGHTS].get("provenance"),
        "classifier provenance",
    )
    openset_provenance = _mapping(
        asset_payloads[OPENSET_WEIGHTS].get("provenance"),
        "open-set provenance",
    )
    if classifier_provenance != openset_provenance:
        raise PromotionError(
            "classifier and open-set assets do not carry identical provenance"
        )
    for key, expected in (
        ("development_only", True),
        ("release_evidence", False),
        ("sealed_release_data_used", 0),
        ("consumed_test_rows_used", 0),
        ("release_seed_not_spent", RELEASE_SEED),
        ("staged_validation_status", STAGED_STATUS),
        ("staged_validation_role", "validate"),
        ("staged_validation_all_pass", True),
    ):
        if classifier_provenance.get(key) != expected:
            raise PromotionError(
                f"staging candidate provenance {key!r} is "
                f"{classifier_provenance.get(key)!r}, expected {expected!r}"
            )
    _sha256(
        classifier_provenance.get("staged_validation_report_sha256"),
        "staging candidate provenance.staged_validation_report_sha256",
    )

    open_contract = _mapping(
        asset_payloads[OPENSET_WEIGHTS].get("contract"), "open-set contract"
    )
    for key, expected in (
        ("additive_only", False),
        ("changes_closed_label", True),
        ("gates_before_classification", True),
    ):
        if open_contract.get(key) is not expected:
            raise PromotionError(f"open-set contract {key!r} is not {expected!r}")
    composite = _mapping(
        asset_payloads[OPENSET_WEIGHTS].get("composite"),
        "open-set composite policy",
    )
    for key, expected in (
        ("schema", STAGED_POLICY_SCHEMA),
        ("kind", STAGED_POLICY_KIND),
        ("policy_version", STAGED_POLICY_VERSION),
    ):
        if composite.get(key) != expected:
            raise PromotionError(
                f"open-set composite policy {key!r} differs from {expected!r}"
            )

    tracked_paths = tuple(sorted((entry.resolve() for entry in entries), key=str))
    if policy.require_git_tracking:
        _verify_git_tracking(policy.repo_root, tracked_paths)

    return VerifiedStaging(
        manifest=manifest,
        manifest_sha256=_sha256_bytes(manifest_raw),
        asset_bytes=asset_bytes,
        asset_payloads=asset_payloads,
        tracked_paths=tracked_paths,
    )


def _verify_historical_metadata(
    report: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    top = report.get("historical_gate_redeclaration")
    nested = protocol.get("historical_gate_redeclaration")
    if top != nested:
        raise PromotionError(
            "report and protocol historical gate metadata differ"
        )
    block = _mapping(top, "historical_gate_redeclaration")
    if (
        block.get("release_seed") != HISTORICAL_RELAXED_SEED
        or block.get("active_for_current_protocol") is not False
        or block.get("current_protocol_gate_source")
        != "imported_v2_gate_floors_unchanged"
    ):
        raise PromotionError(
            "seed-20260734 relaxed gates must remain explicitly inactive history"
        )
    gates = _mapping(
        block.get("gates_redeclared"),
        "historical_gate_redeclaration.gates_redeclared",
    )
    expected = {
        "five_shot_worst_length_balanced": (0.85, 0.84),
        "open_known_false_unknown_worst_length": (0.10, 0.12),
    }
    if set(gates) != set(expected):
        raise PromotionError("historical gate metadata has an unexpected gate set")
    for name, (v2_level, v3_level) in expected.items():
        record = _mapping(gates[name], f"historical gate {name}")
        _exact_object_keys(
            record,
            {"v2_level", "v3_level", "owner_decision"},
            f"historical gate {name}",
        )
        if (
            record.get("v2_level") != v2_level
            or record.get("v3_level") != v3_level
            or not isinstance(record.get("owner_decision"), str)
            or not record["owner_decision"].strip()
        ):
            raise PromotionError(f"historical gate metadata is invalid for {name}")


def _verify_release_gates(
    gates_value: Any,
    candidate_sha256: str,
) -> None:
    gates = _mapping(gates_value, "release report gates")
    if set(gates) != set(EXPECTED_GATE_NAMES):
        raise PromotionError(
            "release report must contain exactly the 23 strict v2 gates; "
            f"got {len(gates)}"
        )

    for name in BOOLEAN_GATE_NAMES:
        gate = _mapping(gates[name], f"release gate {name}")
        if dict(gate) != {"value": True, "expected": True, "passes": True}:
            raise PromotionError(f"release gate {name} did not strictly pass")

    candidate_gate = _mapping(
        gates[CANDIDATE_GATE_NAME],
        f"release gate {CANDIDATE_GATE_NAME}",
    )
    if dict(candidate_gate) != {
        "value": candidate_sha256,
        "expected": candidate_sha256,
        "passes": True,
    }:
        raise PromotionError("candidate_sha_bound gate is not exact and passing")

    for name, (threshold, comparison) in NUMERIC_GATE_CONTRACT.items():
        gate = _mapping(gates[name], f"release gate {name}")
        _exact_object_keys(
            gate,
            {"value", "threshold", "comparison", "passes"},
            f"release gate {name}",
        )
        value = gate.get("value")
        if not _is_finite_number(value):
            raise PromotionError(f"release gate {name} value must be finite")
        if (
            not _is_finite_number(gate.get("threshold"))
            or float(gate["threshold"]) != threshold
            or gate.get("comparison") != comparison
            or gate.get("passes") is not True
        ):
            raise PromotionError(
                f"release gate {name} does not use the strict v2 contract"
            )
        actual = float(value)
        passes = actual >= threshold if comparison == "min" else actual <= threshold
        if not passes:
            raise PromotionError(
                f"release gate {name} claims pass for failing value {actual}"
            )


def _verify_evaluation_report(
    report_path: Path,
    staging: VerifiedStaging,
) -> VerifiedEvaluation:
    report_raw = _read_regular_file(report_path, "sealed evaluation report")
    report_value = _parse_json(report_raw, "sealed evaluation report")
    report = _mapping(report_value, "sealed evaluation report")

    if type(report.get("schema")) is not int or report.get("schema") != EVALUATOR_SCHEMA:
        raise PromotionError("release report evaluator schema must be integer 2")
    if report.get("status") != "complete":
        raise PromotionError("release report status must be complete")
    if report.get("release_evidence") is not True:
        raise PromotionError("release report is not release evidence")
    if report.get("evaluation_version") != EVALUATION_VERSION:
        raise PromotionError("release report evaluation_version is not v2")
    for key, expected in (
        ("development_data_loaded", False),
        ("retraining_performed", False),
        ("recalibration_performed", False),
    ):
        if report.get(key) is not expected:
            raise PromotionError(
                f"release report {key!r} is {report.get(key)!r}, "
                f"expected {expected!r}"
            )
    if report.get("all_release_gates_pass") is not True:
        raise PromotionError("release report does not pass every release gate")

    provenance = _mapping(report.get("provenance"), "release report provenance")
    if (
        not _is_exact_int(provenance.get("release_seed"))
        or provenance.get("release_seed") != RELEASE_SEED
    ):
        raise PromotionError(
            f"release report must bind untouched seed {RELEASE_SEED}"
        )
    protocol = _mapping(
        provenance.get("evaluation_protocol"),
        "release report evaluation protocol",
    )
    if protocol.get("version") != EVALUATION_VERSION:
        raise PromotionError("embedded evaluation protocol has the wrong version")
    protocol_gates = _mapping(
        protocol.get("gates"), "release report protocol gates"
    )
    if dict(protocol_gates) != STRICT_V2_GATE_FLOORS:
        raise PromotionError(
            "embedded protocol gates differ from the strict imported v2 floors"
        )
    novelty = _mapping(
        protocol.get("novelty"), "release report protocol novelty"
    )
    if novelty.get("seed") != RELEASE_SEED:
        raise PromotionError(
            f"embedded novelty protocol must bind seed {RELEASE_SEED}"
        )
    _verify_historical_metadata(report, protocol)

    candidate = _mapping(report.get("candidate"), "release report candidate")
    candidate_sha = _sha256(
        candidate.get("sha256"), "release report candidate.sha256"
    )
    if provenance.get("candidate_sha256") != candidate_sha:
        raise PromotionError(
            "release report provenance candidate SHA differs from candidate"
        )
    for key, expected in (
        ("runtime_schema", RUNTIME_BUNDLE_SCHEMA),
        ("runtime_schema_version", RUNTIME_BUNDLE_SCHEMA_VERSION),
        ("runtime_kind", RUNTIME_BUNDLE_KIND),
    ):
        if candidate.get(key) != expected:
            raise PromotionError(
                f"release report candidate {key!r} differs from {expected!r}"
            )
    frozen_assets = candidate.get("frozen_assets_used")
    if (
        not isinstance(frozen_assets, list)
        or not frozen_assets
        or not all(isinstance(item, str) and item for item in frozen_assets)
    ):
        raise PromotionError("release report does not enumerate frozen assets")

    components = _mapping(
        candidate.get("components"), "release report candidate.components"
    )
    if components.get("bundle_manifest_sha256") != candidate_sha:
        raise PromotionError(
            "candidate component bundle manifest SHA differs from candidate SHA"
        )
    bundle_assets = _bundle_asset_records(components.get("bundle_assets"))
    bundle_asset_hashes = {
        name: record["sha256"] for name, record in bundle_assets.items()
    }
    fusion_directory_sha = _sha256(
        components.get("fusion_directory_sha256"),
        "candidate fusion_directory_sha256",
    )
    fusion_file_hashes = _hash_map(
        components.get("fusion_file_sha256"),
        "candidate fusion_file_sha256",
    )
    if (
        bundle_asset_hashes["fusion_state_dict.pt"]
        not in set(fusion_file_hashes.values())
    ):
        raise PromotionError(
            "runtime bundle fusion state is not bound to a fusion artifact file"
        )
    fusion_seed = components.get("fusion_seed")
    if not _is_exact_int(fusion_seed):
        raise PromotionError("candidate fusion_seed must be an integer")
    staged_hashes = _hash_map(
        components.get("staged_artifact_sha256"),
        "candidate staged_artifact_sha256",
        exact_names=EXPECTED_STAGED_ASSETS,
    )
    if components.get("staged_status") != STAGED_STATUS:
        raise PromotionError("candidate staged artifact did not pass validation")
    prefilter_hash = _sha256(
        components.get("prefilter_set_sha256"),
        "candidate prefilter_set_sha256",
    )
    if components.get("staged_policy_version") != STAGED_POLICY_VERSION:
        raise PromotionError("candidate staged policy version is not v2 composite")

    stage_one_lengths = components.get("stage_one_capture_lengths")
    if (
        not isinstance(stage_one_lengths, list)
        or not stage_one_lengths
        or any(not _is_exact_int(value) or value <= 0 for value in stage_one_lengths)
        or len(stage_one_lengths) != len(set(stage_one_lengths))
    ):
        raise PromotionError("candidate stage-one capture lengths are invalid")
    open_protocol = _mapping(
        protocol.get("open_set"), "release report protocol open_set"
    )
    for key, expected in (
        ("staged_policy_schema", STAGED_POLICY_SCHEMA),
        ("staged_policy_version", STAGED_POLICY_VERSION),
        ("architecture", STAGED_POLICY_KIND),
        ("additive_only", False),
        ("changes_closed_label", True),
        ("gates_before_classification", True),
    ):
        if open_protocol.get(key) != expected:
            raise PromotionError(
                f"release protocol open_set {key!r} differs from {expected!r}"
            )
    if open_protocol.get("stage_one_capture_lengths") != stage_one_lengths:
        raise PromotionError(
            "release protocol and candidate stage-one lengths differ"
        )

    architecture = _mapping(
        report.get("architecture_contract"),
        "release report architecture_contract",
    )
    for key, expected in (
        ("additive_only", False),
        ("changes_closed_label", True),
        ("gates_before_classification", True),
        ("staged_policy_schema", STAGED_POLICY_SCHEMA),
        ("staged_policy_version", STAGED_POLICY_VERSION),
        ("staged_policy_kind", STAGED_POLICY_KIND),
    ):
        if architecture.get(key) != expected:
            raise PromotionError(
                f"release architecture {key!r} differs from {expected!r}"
            )
    if architecture.get("stage_one_capture_lengths") != stage_one_lengths:
        raise PromotionError(
            "release architecture and candidate stage-one lengths differ"
        )

    _verify_release_gates(report.get("gates"), candidate_sha)

    classifier = staging.asset_payloads[CLASSIFIER_WEIGHTS]
    openset = staging.asset_payloads[OPENSET_WEIGHTS]
    fusion = staging.asset_payloads[FUSION_WEIGHTS]
    classifier_provenance = _mapping(
        classifier.get("provenance"), "classifier provenance"
    )
    openset_provenance = _mapping(
        openset.get("provenance"), "open-set provenance"
    )
    fusion_provenance = _mapping(fusion.get("provenance"), "fusion provenance")

    for label, asset_provenance in (
        ("classifier", classifier_provenance),
        ("open-set", openset_provenance),
    ):
        if asset_provenance.get("runtime_bundle_manifest_sha256") != candidate_sha:
            raise PromotionError(
                f"{label} asset is from a different runtime bundle"
            )
        if asset_provenance.get("fusion_directory_sha256") != fusion_directory_sha:
            raise PromotionError(f"{label} asset is from a different fusion artifact")
        asset_staged_hashes = _hash_map(
            asset_provenance.get("staged_artifact_sha256"),
            f"{label} provenance staged_artifact_sha256",
            exact_names=EXPECTED_STAGED_ASSETS,
        )
        if asset_staged_hashes != staged_hashes:
            raise PromotionError(f"{label} asset is from a different staged artifact")
        if asset_provenance.get("prefilter_set_sha256") != prefilter_hash:
            raise PromotionError(f"{label} asset is from a different prefilter set")

    exported_bundle_hashes = _hash_map(
        fusion_provenance.get("source_bundle_assets_sha256"),
        "fusion provenance source_bundle_assets_sha256",
        exact_names=EXPECTED_BUNDLE_ASSETS,
    )
    if exported_bundle_hashes != bundle_asset_hashes:
        raise PromotionError(
            "fusion asset is from different runtime-bundle bytes"
        )
    if fusion_provenance.get("source_bundle_schema") != RUNTIME_BUNDLE_SCHEMA:
        raise PromotionError("fusion asset records an unexpected runtime schema")
    if fusion_provenance.get("assembly_seed") != fusion_seed:
        raise PromotionError("fusion asset and report disagree on fusion seed")

    release_intent_sha = _sha256(
        provenance.get("release_intent_sha256"),
        "release report provenance.release_intent_sha256",
    )
    release_manifest_sha = _sha256(
        provenance.get("release_manifest_sha256"),
        "release report provenance.release_manifest_sha256",
    )
    staged_validation_sha = _sha256(
        classifier_provenance.get("staged_validation_report_sha256"),
        "candidate staged validation report SHA",
    )
    candidate_artifacts = {
        "runtime_bundle_manifest_sha256": candidate_sha,
        "runtime_bundle_assets_sha256": bundle_asset_hashes,
        "fusion_directory_sha256": fusion_directory_sha,
        "fusion_files_sha256": fusion_file_hashes,
        "fusion_seed": fusion_seed,
        "staged_artifacts_sha256": staged_hashes,
        "staged_validation_report_sha256": staged_validation_sha,
        "prefilter_set_sha256": prefilter_hash,
        "stage_one_capture_lengths": list(stage_one_lengths),
        "staged_policy_version": STAGED_POLICY_VERSION,
    }
    return VerifiedEvaluation(
        report_sha256=_sha256_bytes(report_raw),
        candidate_sha256=candidate_sha,
        candidate_artifacts=candidate_artifacts,
        release_intent_sha256=release_intent_sha,
        release_manifest_sha256=release_manifest_sha,
    )


def _release_files(
    staging: VerifiedStaging,
    evaluation: VerifiedEvaluation,
) -> dict[str, bytes]:
    output: dict[str, bytes] = {}
    source_asset_records = _mapping(
        staging.manifest["assets"], "staging package manifest.assets"
    )
    for name in ASSET_NAMES:
        if name in STATUS_ASSET_NAMES:
            payload = copy.deepcopy(staging.asset_payloads[name])
            if payload.get("status") != STAGING_STATUS:  # defensive
                raise PromotionError(f"source asset {name} changed during verification")
            payload["status"] = RELEASE_STATUS
            output[name] = _canonical_json_bytes(payload)
        else:
            output[name] = staging.asset_bytes[name]

    output_records = {
        name: {
            "bytes": len(output[name]),
            "sha256": _sha256_bytes(output[name]),
        }
        for name in ASSET_NAMES
    }
    source_records = {
        name: {
            "bytes": int(source_asset_records[name]["bytes"]),
            "sha256": str(source_asset_records[name]["sha256"]),
        }
        for name in ASSET_NAMES
    }
    manifest = {
        "schema": PACKAGE_SCHEMA,
        "schema_version": 1,
        "status": RELEASE_STATUS,
        "assets": output_records,
        "sources": dict(staging.manifest["sources"]),
        "promotion": {
            "schema": "atomos.v3.time-domain-classifier.release-promotion",
            "schema_version": 1,
            "release_seed": RELEASE_SEED,
            "source": {
                "package_status": STAGING_STATUS,
                "package_manifest_sha256": staging.manifest_sha256,
                "asset_records": source_records,
            },
            "evaluation": {
                "status": "complete",
                "all_release_gates_pass": True,
                "gate_contract": "strict_imported_v2_23_of_23",
                "five_shot_minimum": 0.85,
                "known_false_unknown_maximum": 0.10,
                "report_sha256": evaluation.report_sha256,
                "release_intent_sha256": evaluation.release_intent_sha256,
                "release_manifest_sha256": evaluation.release_manifest_sha256,
                "evaluation_version": EVALUATION_VERSION,
            },
            "candidate_artifacts": evaluation.candidate_artifacts,
            "status_rewrites": {
                name: {"from": STAGING_STATUS, "to": RELEASE_STATUS}
                for name in STATUS_ASSET_NAMES
            },
            "operation": {
                "development_data_loaded": False,
                "training_performed": False,
                "recalibration_performed": False,
                "sealed_corpus_opened_by_promoter": False,
            },
        },
    }
    output[PACKAGE_MANIFEST] = _canonical_json_bytes(manifest)
    return output


def _write_new_file(path: Path, raw: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, 0o644)
    except OSError as exc:
        raise PromotionError(f"cannot materialize release file {path.name}") from exc


def _materialize_release(destination: Path, files: Mapping[str, bytes]) -> None:
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PromotionError("cannot create release package parent") from exc

    try:
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.tmp-",
                dir=destination.parent,
            )
        )
    except OSError as exc:
        raise PromotionError("cannot create temporary release package") from exc

    moved = False
    removed_empty_destination = False
    try:
        # Manifest last: a partially written temporary directory never looks
        # like a complete package even before the atomic directory rename.
        for name in ASSET_NAMES:
            _write_new_file(temporary / name, files[name])
        _write_new_file(temporary / PACKAGE_MANIFEST, files[PACKAGE_MANIFEST])

        if destination.exists():
            if destination.is_symlink() or not destination.is_dir():
                raise PromotionError("release destination changed during promotion")
            if next(destination.iterdir(), None) is not None:
                raise PromotionError("release destination became nonempty")
            destination.rmdir()
            removed_empty_destination = True
        temporary.rename(destination)
        moved = True
    except PromotionError:
        raise
    except OSError as exc:
        raise PromotionError("cannot atomically install release package") from exc
    finally:
        if not moved:
            shutil.rmtree(temporary, ignore_errors=True)
            if removed_empty_destination and not destination.exists():
                try:
                    destination.mkdir()
                except OSError:
                    pass


def _verify_materialized_release(
    destination: Path,
    expected_files: Mapping[str, bytes],
) -> None:
    expected_names = {*ASSET_NAMES, PACKAGE_MANIFEST}
    try:
        entries = tuple(destination.iterdir())
    except OSError as exc:
        raise PromotionError("cannot verify materialized release package") from exc
    if {entry.name for entry in entries} != expected_names:
        raise PromotionError("materialized release package has unexpected entries")
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            raise PromotionError("materialized release contains a non-regular file")
        if _read_regular_file(entry, f"release file {entry.name}") != expected_files[
            entry.name
        ]:
            raise PromotionError(
                f"materialized release file {entry.name} changed while writing"
            )


def promote(
    staging_package: Path,
    evaluation_report: Path,
    destination: Path,
    *,
    policy: PromotionPolicy,
) -> dict[str, Any]:
    """Verify and atomically materialize one release package.

    All validation completes before the destination is touched.  The return
    value is the newly written release package manifest.
    """

    staging_path, destination_path, normalized = _validate_paths(
        Path(staging_package), Path(destination), policy
    )
    evaluation_path = Path(evaluation_report).expanduser()
    if evaluation_path.is_symlink():
        raise PromotionError("sealed evaluation report may not be a symlink")
    evaluation_path = evaluation_path.resolve()
    if _within(evaluation_path, destination_path):
        raise PromotionError("evaluation report may not be inside the destination")

    verified_staging = _verify_staging_package(staging_path, normalized)
    verified_evaluation = _verify_evaluation_report(
        evaluation_path, verified_staging
    )
    files = _release_files(verified_staging, verified_evaluation)
    _materialize_release(destination_path, files)
    _verify_materialized_release(destination_path, files)
    manifest_value = _parse_json(
        files[PACKAGE_MANIFEST], "materialized release package manifest"
    )
    return dict(_mapping(manifest_value, "materialized release package manifest"))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--evaluation-report",
        type=Path,
        required=True,
        help="completed sealed seed-20260735 RELEASE_EVALUATION.json",
    )
    result.add_argument(
        "--staging-package",
        type=Path,
        default=DEFAULT_STAGING_PACKAGE,
        help="must resolve to the exact tracked v3 staging package",
    )
    result.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_RELEASE_PACKAGE,
        help="must resolve to the distinct v3 release package path",
    )
    return result


def main() -> None:
    arguments = parser().parse_args()
    policy = PromotionPolicy(
        repo_root=REPO,
        staging_package=DEFAULT_STAGING_PACKAGE,
        release_package=DEFAULT_RELEASE_PACKAGE,
        live_v2_assets=LIVE_V2_ASSETS,
        require_git_tracking=True,
    )
    try:
        manifest = promote(
            arguments.staging_package,
            arguments.evaluation_report,
            arguments.destination,
            policy=policy,
        )
    except PromotionError as exc:
        raise SystemExit(f"release promotion refused: {exc}") from exc
    print(
        f"promoted {len(manifest['assets'])} verified v3 runtime assets to "
        f"{DEFAULT_RELEASE_PACKAGE}"
    )


if __name__ == "__main__":
    main()
