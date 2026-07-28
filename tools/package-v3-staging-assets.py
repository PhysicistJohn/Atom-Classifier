#!/usr/bin/env python3
"""Build the fail-closed, deployable v3 dual-fusion staging package.

Only four runtime-required JSON assets are copied: the role-bound rejector and
classifier weights, the staged open-set policy, and their dual binding. Probe
and parity fixtures remain training evidence and are hash-bound externally.
Every deployable file must be strictly smaller than 25 MiB.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
ARTIFACT_STAGING = (
    REPO / "training/zplane_ab/v2_full_variation/artifacts/staging"
)
DEFAULT_REJECTOR_EXPORT = (
    ARTIFACT_STAGING / "time_domain_v3_rejector_fusion_dual"
)
DEFAULT_CLASSIFIER_EXPORT = (
    ARTIFACT_STAGING / "time_domain_v3_classifier_fusion_dual"
)
DEFAULT_OPENSET_EXPORT = ARTIFACT_STAGING / "time_domain_v3_openset_dual"
DEFAULT_DESTINATION = REPO / "src/embedding/assets-v3-dual-staging"
LIVE_V2_ASSETS = (REPO / "src/embedding/assets").resolve()
LEGACY_V3_STAGING = (REPO / "src/embedding/assets-v3-staging").resolve()

PACKAGE_SCHEMA = "atomos.v3.time-domain-classifier.dual-runtime-package"
PACKAGE_SCHEMA_VERSION = 1
STAGING_STATUS = "staging_not_release"
CANDIDATE_ID = "v3.3-decoupled-8k-classifier-4k-rejector"
MAX_DEPLOYABLE_BYTES = 25 * 1024 * 1024

FUSION_EXPORT_SCHEMA = (
    "atomos.v3.time-domain-invariant-fusion.browser-weights.export-manifest"
)
FUSION_EXPORT_SCHEMA_VERSION = 2
FUSION_WEIGHTS_SCHEMA = (
    "atomos.v3.time-domain-invariant-fusion.browser-weights"
)
OPENSET_EXPORT_SCHEMA = "time-domain-v3-dual-openset-staging-manifest-v1"
OPENSET_SCHEMA = "atomos.v3.time-domain-openset.staged"
DUAL_BINDING_SCHEMA = "atomos.v3.time-domain-dual-fusion.binding"

REJECTOR_ROLE = "known_unknown_rejector"
CLASSIFIER_ROLE = "accepted_known_classifier"
REJECTOR_RESPONSIBILITY = "known_unknown_only"
CLASSIFIER_RESPONSIBILITY = "accepted_known_label_only"

FUSION_MANIFEST = "export-manifest.json"
OPENSET_MANIFEST = "manifest.json"
REJECTOR_WEIGHTS = "time-domain-v3-rejector-weights.json"
CLASSIFIER_WEIGHTS = "time-domain-v3-classifier-weights.json"
REJECTOR_PROBE = "time-domain-v3-rejector-probe.json"
CLASSIFIER_PROBE = "time-domain-v3-classifier-probe.json"
OPENSET_POLICY = "time-domain-v3-openset-policy.json"
DUAL_BINDING = "time-domain-v3-dual-binding.json"
OPENSET_PARITY = "time-domain-openset-parity-v1.json"
PACKAGE_MANIFEST = "runtime-package-manifest.json"
DEPLOYABLE_ASSETS = (
    REJECTOR_WEIGHTS,
    CLASSIFIER_WEIGHTS,
    OPENSET_POLICY,
    DUAL_BINDING,
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class PackageError(RuntimeError):
    """An input cannot safely become a dual-fusion runtime package."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise PackageError(f"{label} must be a lowercase SHA-256")
    return value


def read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PackageError(f"{path} must be a regular non-symlink file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackageError(f"{path} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise PackageError(f"{path} must contain a JSON object")
    return value


def json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _record(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def _verify_record(
    path: Path,
    value: Any,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"bytes", "sha256"}:
        raise PackageError(f"{label} must contain exactly bytes and sha256")
    size = value.get("bytes")
    if type(size) is not int or size <= 0:
        raise PackageError(f"{label}.bytes must be a positive integer")
    digest = _sha(value.get("sha256"), f"{label}.sha256")
    if path.is_symlink() or not path.is_file():
        raise PackageError(f"{path} must be a regular non-symlink file")
    if path.stat().st_size != size or sha256(path) != digest:
        raise PackageError(f"{label} differs from its recorded bytes")
    return {"bytes": size, "sha256": digest}


def _source_path(path: Path, destination: Path) -> str:
    """A package-parent-relative evidence locator; never serialize absolutes."""
    return Path(
        os.path.relpath(Path(path).resolve(), Path(destination).resolve())
    ).as_posix()


def _asset_record(
    path: Path,
    payload: Mapping[str, Any],
    *,
    runtime_role: str | None = None,
) -> dict[str, Any]:
    record = {
        "path": path.name,
        **_record(path),
        "schema": payload.get("schema"),
        "schema_version": payload.get("schema_version"),
        "status": payload.get("status"),
    }
    if runtime_role is not None:
        record["runtime_role"] = runtime_role
    return record


def _load_role_export(
    directory: Path,
    *,
    role: str,
    weights_name: str,
    probe_name: str,
) -> dict[str, Any]:
    if directory.is_symlink() or not directory.is_dir():
        raise PackageError(f"{role} export must be a non-symlink directory")
    manifest_path = directory / FUSION_MANIFEST
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema") != FUSION_EXPORT_SCHEMA
        or manifest.get("schema_version") != FUSION_EXPORT_SCHEMA_VERSION
        or manifest.get("runtime_role") != role
    ):
        raise PackageError(f"{role} export manifest has the wrong role/schema")
    bundle_sha = _sha(
        manifest.get("source_bundle_manifest_sha256"),
        f"{role} source bundle manifest",
    )
    emitted = manifest.get("emitted")
    if not isinstance(emitted, Mapping) or set(emitted) != {
        weights_name,
        probe_name,
    }:
        raise PackageError(f"{role} export has missing or aliased role files")
    weights_path = directory / weights_name
    probe_path = directory / probe_name
    weights_record = _verify_record(
        weights_path, emitted[weights_name], f"{role} weights"
    )
    probe_record = _verify_record(
        probe_path, emitted[probe_name], f"{role} probe"
    )
    weights = read_json(weights_path)
    if (
        weights.get("schema") != FUSION_WEIGHTS_SCHEMA
        or weights.get("schema_version") != 1
        or weights.get("status") != STAGING_STATUS
        or weights.get("runtime_role") != role
    ):
        raise PackageError(f"{role} weights do not carry their exact role")
    provenance = weights.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("source_bundle_manifest_sha256") != bundle_sha
    ):
        raise PackageError(f"{role} weights lose their source-bundle binding")
    return {
        "directory": directory,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "manifest_record": _record(manifest_path),
        "bundle_sha256": bundle_sha,
        "weights_path": weights_path,
        "weights": weights,
        "weights_record": weights_record,
        "probe_path": probe_path,
        "probe_record": probe_record,
    }


def _verify_binding(
    binding: Mapping[str, Any],
    rejector: Mapping[str, Any],
    classifier: Mapping[str, Any],
    policy_record: Mapping[str, Any],
) -> None:
    if (
        binding.get("schema") != DUAL_BINDING_SCHEMA
        or binding.get("schema_version") != 1
        or binding.get("status") != STAGING_STATUS
        or binding.get("candidate_id") != CANDIDATE_ID
    ):
        raise PackageError("dual binding has the wrong schema/status/candidate")
    if binding.get("execution_order") != [
        "stage_one_noise_gate",
        "rejector_known_unknown",
        "classifier_known_label",
    ]:
        raise PackageError("dual binding execution order changed")
    roles = binding.get("roles")
    if not isinstance(roles, Mapping) or set(roles) != {
        "rejector",
        "classifier",
    }:
        raise PackageError("dual binding must contain exactly two named roles")
    expected = {
        "rejector": (
            roles["rejector"],
            REJECTOR_WEIGHTS,
            REJECTOR_ROLE,
            REJECTOR_RESPONSIBILITY,
            rejector,
        ),
        "classifier": (
            roles["classifier"],
            CLASSIFIER_WEIGHTS,
            CLASSIFIER_ROLE,
            CLASSIFIER_RESPONSIBILITY,
            classifier,
        ),
    }
    for label, (
        value,
        filename,
        role,
        responsibility,
        source,
    ) in expected.items():
        if not isinstance(value, Mapping):
            raise PackageError(f"binding role {label} is not an object")
        if (
            value.get("asset") != filename
            or value.get("runtime_role") != role
            or value.get("responsibility") != responsibility
            or value.get("asset_sha256")
            != source["weights_record"]["sha256"]
            or value.get("runtime_bundle_manifest_sha256")
            != source["bundle_sha256"]
        ):
            raise PackageError(f"binding role {label} is not byte/role bound")
        _sha(
            value.get("fusion_directory_sha256"),
            f"binding {label} fusion directory",
        )
    rejector_role = roles["rejector"]
    classifier_role = roles["classifier"]
    for field in (
        "asset_sha256",
        "runtime_bundle_manifest_sha256",
        "fusion_directory_sha256",
    ):
        if rejector_role[field] == classifier_role[field]:
            raise PackageError(f"dual binding aliases role {field}")

    openset = binding.get("openset_policy")
    if not isinstance(openset, Mapping):
        raise PackageError("binding has no open-set policy")
    if (
        openset.get("asset") != OPENSET_POLICY
        or openset.get("asset_sha256") != policy_record["sha256"]
        or openset.get("rejector_asset_sha256")
        != rejector["weights_record"]["sha256"]
        or openset.get("fitted_rejector_runtime_bundle_manifest_sha256")
        != rejector["bundle_sha256"]
    ):
        raise PackageError("open-set policy is not bound to the rejector role")
    report_sha = _sha(
        openset.get("staged_validation_report_sha256"),
        "binding staged validation report",
    )
    staged_hashes = openset.get("staged_artifacts_sha256")
    expected_staged = {
        "v3_branch_lof_components.npz",
        "v3_open_policy_stage_two.npz",
        "v3_staged_composite_policy.npz",
    }
    if not isinstance(staged_hashes, Mapping) or set(staged_hashes) != expected_staged:
        raise PackageError("binding staged policy asset set is incomplete")
    for name, digest in staged_hashes.items():
        _sha(digest, f"binding staged asset {name}")
    validation = binding.get("validation")
    if (
        not isinstance(validation, Mapping)
        or validation.get("report_sha256") != report_sha
        or validation.get("role") != "validate"
        or validation.get("status") != "development_openset_pass"
        or validation.get("novelty_seeds") != [20260950, 20260951]
    ):
        raise PackageError("binding validation evidence is not the frozen pair")
    fail_closed = binding.get("fail_closed")
    required_flags = {
        "role_assets_bound_by_sha256",
        "distinct_role_assets",
        "role_asset_sha256_must_differ",
        "classifier_runs_only_after_rejector_acceptance",
        "public_known_label_from_classifier_only",
    }
    if (
        not isinstance(fail_closed, Mapping)
        or set(fail_closed) != required_flags
        or any(fail_closed[name] is not True for name in required_flags)
    ):
        raise PackageError("dual binding fail-closed contract is incomplete")


def _load_openset_export(
    directory: Path,
    rejector: Mapping[str, Any],
    classifier: Mapping[str, Any],
) -> dict[str, Any]:
    if directory.is_symlink() or not directory.is_dir():
        raise PackageError("open-set export must be a non-symlink directory")
    manifest_path = directory / OPENSET_MANIFEST
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema") != OPENSET_EXPORT_SCHEMA
        or manifest.get("schema_version") != 1
        or manifest.get("status") != STAGING_STATUS
        or manifest.get("candidate_id") != CANDIDATE_ID
    ):
        raise PackageError("open-set export is not the dual staging schema")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != {
        OPENSET_POLICY,
        DUAL_BINDING,
        OPENSET_PARITY,
    }:
        raise PackageError("open-set export output set differs from dual contract")
    paths = {name: directory / name for name in outputs}
    records = {
        name: _verify_record(paths[name], outputs[name], f"open-set {name}")
        for name in outputs
    }

    role_exports = manifest.get("role_exports")
    if not isinstance(role_exports, Mapping) or set(role_exports) != {
        "rejector",
        "classifier",
    }:
        raise PackageError("open-set manifest lacks exact role exports")
    for label, role, source, weights_name, probe_name in (
        ("rejector", REJECTOR_ROLE, rejector, REJECTOR_WEIGHTS, REJECTOR_PROBE),
        (
            "classifier",
            CLASSIFIER_ROLE,
            classifier,
            CLASSIFIER_WEIGHTS,
            CLASSIFIER_PROBE,
        ),
    ):
        value = role_exports[label]
        if (
            not isinstance(value, Mapping)
            or value.get("runtime_role") != role
            or value.get("manifest_sha256")
            != source["manifest_record"]["sha256"]
            or value.get("weights")
            != {"name": weights_name, **source["weights_record"]}
            or value.get("probe")
            != {"name": probe_name, **source["probe_record"]}
        ):
            raise PackageError(f"open-set manifest {label} role link differs")

    policy = read_json(paths[OPENSET_POLICY])
    if (
        policy.get("schema") != OPENSET_SCHEMA
        or policy.get("schema_version") != 2
        or policy.get("status") != STAGING_STATUS
    ):
        raise PackageError("open-set policy schema/status differs")
    if paths[OPENSET_POLICY].stat().st_size >= MAX_DEPLOYABLE_BYTES:
        raise PackageError("open-set policy is not below the 25 MiB limit")
    provenance = policy.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("candidate_id") != CANDIDATE_ID
    ):
        raise PackageError("open-set policy has no candidate provenance")
    runtime_roles = provenance.get("runtime_roles")
    if not isinstance(runtime_roles, Mapping):
        raise PackageError("open-set policy has no role provenance")
    for label, role, source, filename in (
        ("rejector", REJECTOR_ROLE, rejector, REJECTOR_WEIGHTS),
        ("classifier", CLASSIFIER_ROLE, classifier, CLASSIFIER_WEIGHTS),
    ):
        value = runtime_roles.get(label)
        if (
            not isinstance(value, Mapping)
            or value.get("runtime_role") != role
            or value.get("browser_asset") != filename
            or value.get("browser_asset_sha256")
            != source["weights_record"]["sha256"]
            or value.get("browser_export_manifest_sha256")
            != source["manifest_record"]["sha256"]
            or value.get("runtime_bundle_manifest_sha256")
            != source["bundle_sha256"]
        ):
            raise PackageError(f"open-set policy {label} provenance differs")
        _sha(
            value.get("fusion_directory_sha256"),
            f"open-set policy {label} fusion directory",
        )

    binding = read_json(paths[DUAL_BINDING])
    _verify_binding(binding, rejector, classifier, records[OPENSET_POLICY])
    openset_binding = binding["openset_policy"]
    if (
        provenance.get("staged_validation_report_sha256")
        != openset_binding["staged_validation_report_sha256"]
        or provenance.get("staged_artifact_sha256")
        != openset_binding["staged_artifacts_sha256"]
    ):
        raise PackageError("policy provenance differs from binding evidence pins")
    parity = read_json(paths[OPENSET_PARITY])
    if (
        parity.get("schema") != "time-domain-openset-parity-v1"
        or parity.get("schema_version") != 3
        or parity.get("status") != STAGING_STATUS
        or parity.get("candidate_id") != CANDIDATE_ID
    ):
        raise PackageError("parity evidence is not the dual-role fixture")
    counts = parity.get("counts")
    if (
        not isinstance(counts, Mapping)
        or counts.get("classifier_executed") != counts.get("accepted")
    ):
        raise PackageError("parity does not prove acceptance-gated classification")
    for row in parity.get("rows", []):
        rejected = row.get("rejected_stage")
        classifier_executed = row.get("classifier_executed")
        classifier_result = row.get("classifier")
        if rejected is None:
            if classifier_executed is not True or not isinstance(
                classifier_result, Mapping
            ):
                raise PackageError("accepted parity row did not run classifier")
        elif classifier_executed is not False or classifier_result is not None:
            raise PackageError("rejected parity row serialized classifier output")
    return {
        "directory": directory,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "manifest_record": _record(manifest_path),
        "paths": paths,
        "records": records,
        "policy": policy,
        "binding": binding,
        "parity": parity,
    }


def _materialize(destination: Path, files: Mapping[str, bytes]) -> None:
    if destination.exists():
        if destination.is_symlink() or not destination.is_dir():
            raise PackageError("destination must be absent or an empty directory")
        if next(destination.iterdir(), None) is not None:
            raise PackageError("destination must be empty")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
        )
    )
    moved = False
    removed_empty = False
    try:
        for name in DEPLOYABLE_ASSETS:
            (temporary / name).write_bytes(files[name])
        (temporary / PACKAGE_MANIFEST).write_bytes(files[PACKAGE_MANIFEST])
        if destination.exists():
            destination.rmdir()
            removed_empty = True
        temporary.rename(destination)
        moved = True
    except OSError as exc:
        raise PackageError("cannot atomically install staging package") from exc
    finally:
        if not moved:
            shutil.rmtree(temporary, ignore_errors=True)
            if removed_empty and not destination.exists():
                destination.mkdir()


def package(
    rejector_export: Path,
    classifier_export: Path,
    openset_export: Path,
    destination: Path,
) -> dict[str, Any]:
    destination = Path(destination).resolve()
    if destination in {LIVE_V2_ASSETS, LEGACY_V3_STAGING}:
        raise PackageError("refusing live v2 or legacy single-fusion staging")
    if (
        LIVE_V2_ASSETS in destination.parents
        or LEGACY_V3_STAGING in destination.parents
    ):
        raise PackageError("refusing a descendant of a legacy asset directory")

    rejector = _load_role_export(
        Path(rejector_export).resolve(),
        role=REJECTOR_ROLE,
        weights_name=REJECTOR_WEIGHTS,
        probe_name=REJECTOR_PROBE,
    )
    classifier = _load_role_export(
        Path(classifier_export).resolve(),
        role=CLASSIFIER_ROLE,
        weights_name=CLASSIFIER_WEIGHTS,
        probe_name=CLASSIFIER_PROBE,
    )
    if (
        rejector["weights_record"]["sha256"]
        == classifier["weights_record"]["sha256"]
        or rejector["bundle_sha256"] == classifier["bundle_sha256"]
    ):
        raise PackageError("rejector and classifier role assets are aliased")
    openset = _load_openset_export(
        Path(openset_export).resolve(),
        rejector,
        classifier,
    )

    source_paths = {
        REJECTOR_WEIGHTS: rejector["weights_path"],
        CLASSIFIER_WEIGHTS: classifier["weights_path"],
        OPENSET_POLICY: openset["paths"][OPENSET_POLICY],
        DUAL_BINDING: openset["paths"][DUAL_BINDING],
    }
    for name, path in source_paths.items():
        if path.stat().st_size >= MAX_DEPLOYABLE_BYTES:
            raise PackageError(
                f"deployable asset {name} is not below 25 MiB"
            )

    payloads = {
        REJECTOR_WEIGHTS: rejector["weights"],
        CLASSIFIER_WEIGHTS: classifier["weights"],
        OPENSET_POLICY: openset["policy"],
        DUAL_BINDING: openset["binding"],
    }
    asset_records = {
        REJECTOR_WEIGHTS: _asset_record(
            source_paths[REJECTOR_WEIGHTS],
            payloads[REJECTOR_WEIGHTS],
            runtime_role=REJECTOR_ROLE,
        ),
        CLASSIFIER_WEIGHTS: _asset_record(
            source_paths[CLASSIFIER_WEIGHTS],
            payloads[CLASSIFIER_WEIGHTS],
            runtime_role=CLASSIFIER_ROLE,
        ),
        OPENSET_POLICY: _asset_record(
            source_paths[OPENSET_POLICY], payloads[OPENSET_POLICY]
        ),
        DUAL_BINDING: _asset_record(
            source_paths[DUAL_BINDING], payloads[DUAL_BINDING]
        ),
    }
    binding = openset["binding"]
    manifest = {
        "schema": PACKAGE_SCHEMA,
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "status": STAGING_STATUS,
        "candidate_id": CANDIDATE_ID,
        "architecture": {
            "execution_order": [
                "stage_one_noise_gate",
                "rejector_known_unknown",
                "classifier_known_label",
            ],
            "classifier_runs_only_after_rejector_acceptance": True,
            "public_known_label_from_classifier_only": True,
        },
        "assets": asset_records,
        "roles": {
            "rejector": {
                "runtime_role": REJECTOR_ROLE,
                "responsibility": REJECTOR_RESPONSIBILITY,
                "asset": asset_records[REJECTOR_WEIGHTS],
                "source_bundle_manifest_sha256": rejector["bundle_sha256"],
                "fusion_directory_sha256": binding["roles"]["rejector"][
                    "fusion_directory_sha256"
                ],
            },
            "classifier": {
                "runtime_role": CLASSIFIER_ROLE,
                "responsibility": CLASSIFIER_RESPONSIBILITY,
                "asset": asset_records[CLASSIFIER_WEIGHTS],
                "source_bundle_manifest_sha256": classifier["bundle_sha256"],
                "fusion_directory_sha256": binding["roles"]["classifier"][
                    "fusion_directory_sha256"
                ],
            },
        },
        "openset_policy": {
            **asset_records[OPENSET_POLICY],
            "fitted_rejector_runtime_bundle_manifest_sha256": (
                binding["openset_policy"][
                    "fitted_rejector_runtime_bundle_manifest_sha256"
                ]
            ),
            "staged_validation_report_sha256": binding["openset_policy"][
                "staged_validation_report_sha256"
            ],
            "staged_artifacts_sha256": binding["openset_policy"][
                "staged_artifacts_sha256"
            ],
        },
        "dual_binding": asset_records[DUAL_BINDING],
        "external_evidence": {
            "rejector_export_manifest": {
                "path": _source_path(rejector["manifest_path"], destination),
                **rejector["manifest_record"],
                "schema": FUSION_EXPORT_SCHEMA,
                "schema_version": FUSION_EXPORT_SCHEMA_VERSION,
                "runtime_role": REJECTOR_ROLE,
            },
            "classifier_export_manifest": {
                "path": _source_path(classifier["manifest_path"], destination),
                **classifier["manifest_record"],
                "schema": FUSION_EXPORT_SCHEMA,
                "schema_version": FUSION_EXPORT_SCHEMA_VERSION,
                "runtime_role": CLASSIFIER_ROLE,
            },
            "openset_export_manifest": {
                "path": _source_path(openset["manifest_path"], destination),
                **openset["manifest_record"],
                "schema": OPENSET_EXPORT_SCHEMA,
                "schema_version": 1,
                "status": STAGING_STATUS,
            },
            "rejector_probe": {
                "path": _source_path(rejector["probe_path"], destination),
                **rejector["probe_record"],
            },
            "classifier_probe": {
                "path": _source_path(classifier["probe_path"], destination),
                **classifier["probe_record"],
            },
            "parity": {
                "path": _source_path(
                    openset["paths"][OPENSET_PARITY], destination
                ),
                **openset["records"][OPENSET_PARITY],
                "schema": openset["parity"]["schema"],
                "schema_version": openset["parity"]["schema_version"],
                "status": openset["parity"]["status"],
                "packaged": False,
            },
        },
        "size_contract": {
            "maximum_file_bytes_exclusive": MAX_DEPLOYABLE_BYTES,
            "all_deployable_files_below_limit": True,
        },
    }
    manifest_raw = json_bytes(manifest)
    if len(manifest_raw) >= MAX_DEPLOYABLE_BYTES:
        raise PackageError("runtime package manifest is not below 25 MiB")
    files = {
        name: source_paths[name].read_bytes() for name in DEPLOYABLE_ASSETS
    }
    files[PACKAGE_MANIFEST] = manifest_raw
    _materialize(destination, files)
    expected_names = {*DEPLOYABLE_ASSETS, PACKAGE_MANIFEST}
    if {path.name for path in destination.iterdir()} != expected_names:
        raise PackageError("materialized package has unexpected files")
    for name, raw in files.items():
        path = destination / name
        if path.is_symlink() or path.read_bytes() != raw:
            raise PackageError(f"materialized package changed {name}")
        if path.stat().st_size >= MAX_DEPLOYABLE_BYTES:
            raise PackageError(f"materialized package exceeds size limit: {name}")
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--rejector-export", type=Path, default=DEFAULT_REJECTOR_EXPORT
    )
    result.add_argument(
        "--classifier-export", type=Path, default=DEFAULT_CLASSIFIER_EXPORT
    )
    result.add_argument(
        "--openset-export", type=Path, default=DEFAULT_OPENSET_EXPORT
    )
    result.add_argument(
        "--destination", type=Path, default=DEFAULT_DESTINATION
    )
    return result


def main() -> None:
    arguments = parser().parse_args()
    manifest = package(
        arguments.rejector_export,
        arguments.classifier_export,
        arguments.openset_export,
        arguments.destination,
    )
    print(
        f"packaged {len(manifest['assets'])} verified dual v3 runtime assets "
        f"into {arguments.destination}"
    )


if __name__ == "__main__":
    main()
