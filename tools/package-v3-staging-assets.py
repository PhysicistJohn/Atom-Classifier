#!/usr/bin/env python3
"""Materialize the tracked v3 browser-runtime staging package.

The training exporters deliberately write open-set output below the ignored
``training/**/artifacts`` tree.  Runtime tests and sibling applications must
not depend on that mutable local state.  This tool verifies both exporter
manifests, copies the required model files into the tracked staging package,
derives a compact raw-I/Q smoke fixture covering accept/stage-1/stage-2 at two
lengths, and writes one manifest over every runtime input.

It never changes an asset status and refuses the live v2 asset directory.
Re-running it against identical exporter output is byte deterministic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_OPENSET_EXPORT = (
    REPO
    / "training/zplane_ab/v2_full_variation/artifacts/staging"
    / "time_domain_v3_openset"
)
DEFAULT_DESTINATION = REPO / "src/embedding/assets-v3-staging"
LIVE_V2_ASSETS = (REPO / "src/embedding/assets").resolve()

FUSION_MANIFEST = "export-manifest.json"
FUSION_WEIGHTS = "time-domain-fusion-weights-v3.json"
FUSION_PROBE = "time-domain-probe-fixture-v3.json"
OPENSET_MANIFEST = "manifest.json"
OPENSET_WEIGHTS = "time-domain-openset-weights-v1.json"
CLASSIFIER_WEIGHTS = "time-domain-classifier-weights-v1.json"
OPENSET_PARITY = "time-domain-openset-parity-v1.json"
SMOKE_FIXTURE = "time-domain-openset-smoke-v1.json"
PACKAGE_MANIFEST = "runtime-package-manifest.json"

# Six real raw captures: accepted, stage-1 gated, and stage-2 rejected at both
# fitted lengths.  Probe cases remain in the fixture too, including the N2048
# refusal case.
SMOKE_ROWS = (
    "known-am-N4096",
    "noise-4-N4096",
    "noise-1-N4096",
    "known-am-N8192",
    "noise-2-N8192",
    "noise-1-N8192",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def verify_file(path: Path, metadata: dict[str, Any], source: str) -> None:
    expected_bytes = int(metadata["bytes"])
    expected_sha = str(metadata["sha256"])
    if path.stat().st_size != expected_bytes:
        raise RuntimeError(f"{source}: byte count changed for {path.name}")
    actual_sha = sha256(path)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"{source}: SHA-256 changed for {path.name}: "
            f"{actual_sha} != {expected_sha}"
        )


def require_staging_asset(path: Path, schema: str) -> None:
    value = read_json(path)
    if value.get("schema") != schema:
        raise RuntimeError(f"{path.name} has unexpected schema")
    if value.get("status") != "staging_not_release":
        raise RuntimeError(
            f"{path.name} must remain staging_not_release; promotion is a "
            "separate, release-gated operation"
        )


def compact_fixture(parity: dict[str, Any], source_sha: str) -> dict[str, Any]:
    by_name = {str(row["name"]): row for row in parity["rows"]}
    missing = [name for name in SMOKE_ROWS if name not in by_name]
    if missing:
        raise RuntimeError(f"parity export is missing smoke rows: {missing}")
    rows = [by_name[name] for name in SMOKE_ROWS]
    stages = {row["rejected_stage"] for row in rows}
    lengths = {int(row["capture_length"]) for row in rows}
    if stages != {None, 1, 2} or lengths != {4096, 8192}:
        raise RuntimeError("smoke rows no longer cover both lengths/all decisions")
    counts = {
        "rows": len(rows),
        "known": sum(row["family"] == "known" for row in rows),
        "noise": sum(row["family"] == "noise" for row in rows),
        "chirp": sum(row["family"] == "chirp" for row in rows),
        "probe": 0,
        "gated_stage_one": sum(row["rejected_stage"] == 1 for row in rows),
        "rejected_stage_two": sum(row["rejected_stage"] == 2 for row in rows),
        "accepted": sum(row["rejected_stage"] is None for row in rows),
    }
    output = dict(parity)
    output["schema"] = "time-domain-openset-runtime-smoke-v1"
    output["rows"] = rows
    output["counts"] = counts
    output["fixture_scope"] = (
        "tracked runtime smoke fixture; full exporter parity remains training "
        "evidence and is not required at runtime"
    )
    output["source_parity_sha256"] = source_sha
    return output


def package(openset_export: Path, destination: Path) -> dict[str, Any]:
    openset_export = openset_export.resolve()
    destination = destination.resolve()
    if destination == LIVE_V2_ASSETS or LIVE_V2_ASSETS in destination.parents:
        raise ValueError("refusing to package v3 assets into the live v2 directory")
    destination.mkdir(parents=True, exist_ok=True)

    fusion_manifest_path = destination / FUSION_MANIFEST
    fusion_manifest = read_json(fusion_manifest_path)
    if fusion_manifest.get("schema") != (
        "atomos.v3.time-domain-invariant-fusion.browser-weights.export-manifest"
    ):
        raise RuntimeError("unexpected fusion export manifest")
    for name in (FUSION_WEIGHTS, FUSION_PROBE):
        verify_file(
            destination / name,
            fusion_manifest["emitted"][name],
            FUSION_MANIFEST,
        )

    openset_manifest_path = openset_export / OPENSET_MANIFEST
    openset_manifest = read_json(openset_manifest_path)
    if openset_manifest.get("schema") != "time-domain-v3-openset-staging-manifest-v1":
        raise RuntimeError("unexpected open-set export manifest")
    for name in (CLASSIFIER_WEIGHTS, OPENSET_WEIGHTS, OPENSET_PARITY):
        verify_file(
            openset_export / name,
            openset_manifest["outputs"][name],
            OPENSET_MANIFEST,
        )

    require_staging_asset(
        destination / FUSION_WEIGHTS,
        "atomos.v3.time-domain-invariant-fusion.browser-weights",
    )
    require_staging_asset(
        openset_export / CLASSIFIER_WEIGHTS,
        "atomos.v3.time-domain-invariant-fusion.browser-decision",
    )
    require_staging_asset(
        openset_export / OPENSET_WEIGHTS,
        "atomos.v3.time-domain-openset.staged",
    )

    for name in (CLASSIFIER_WEIGHTS, OPENSET_WEIGHTS):
        source = openset_export / name
        target = destination / name
        temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)

    parity_path = openset_export / OPENSET_PARITY
    parity_sha = sha256(parity_path)
    smoke = compact_fixture(read_json(parity_path), parity_sha)
    write_json(destination / SMOKE_FIXTURE, smoke)

    packaged_names = (
        FUSION_WEIGHTS,
        FUSION_PROBE,
        CLASSIFIER_WEIGHTS,
        OPENSET_WEIGHTS,
        SMOKE_FIXTURE,
    )
    manifest = {
        "schema": "atomos.v3.time-domain-classifier.runtime-package",
        "schema_version": 1,
        "status": "staging_not_release",
        "assets": {
            name: {
                "bytes": (destination / name).stat().st_size,
                "sha256": sha256(destination / name),
            }
            for name in packaged_names
        },
        "sources": {
            FUSION_MANIFEST: sha256(fusion_manifest_path),
            "openset-export-manifest.json": sha256(openset_manifest_path),
            OPENSET_PARITY: parity_sha,
        },
    }
    write_json(destination / PACKAGE_MANIFEST, manifest)
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--openset-export",
        type=Path,
        default=DEFAULT_OPENSET_EXPORT,
    )
    result.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
    )
    return result


def main() -> None:
    arguments = parser().parse_args()
    manifest = package(arguments.openset_export, arguments.destination)
    print(
        f"packaged {len(manifest['assets'])} verified v3 staging assets into "
        f"{arguments.destination}"
    )


if __name__ == "__main__":
    main()
