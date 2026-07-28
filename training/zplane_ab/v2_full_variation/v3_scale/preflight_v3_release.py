"""Preflight for the one-shot v3 sealed release run on seed 20260731.

This script runs WITHOUT any release suite and consumes nothing: no corpus is
read, no seed is drawn, no novelty realization is generated, and nothing is
written under ``training/artifacts/releases``.  It exists so the orchestrator
can prove, immediately before spending the unspent release seed, that every
frozen input of the one-shot run is exactly what the evidence trail says it is.

What it verifies, each as an itemised PASS/FAIL check:

* the candidate fusion artifact's recorded asset SHA-256 values against disk;
* the v3 runtime bundle: schema identity, every asset hash, the binding to the
  exact candidate fusion (``provenance.source_dev_metrics_sha256``), the bound
  frontend source hashes against the current repo files, and a full
  re-execution of the bundle's probe-fixture self-verification (reload from
  bundle-only inputs, replay the closed-form probes, compare embeddings);
* the staged noise prefilter: fit-report schema, the per-length bundle hashes
  and the set hash, the architecture-contract keys (``additive_only`` false,
  ``changes_closed_label`` true, ``gates_before_classification`` true), the
  fitting-seed hygiene, loadability of every per-length model, and that the
  recorded gate floors equal the frozen development floors;
* the frozen open-set policy constants in ``v3_time_domain_openset`` and the
  import-identity discipline of ``fit_v3_openset`` / ``fit_v3_openset_staged``
  (gates imported, never re-typed);
* evaluator sources: the v2 release evaluator still matches its recorded
  release SHA, the v3 release evaluator exists and byte-compiles, and every
  dependency source file hashes cleanly (all recorded in the report);
* the on-disk seed ledger: no release root exists for seed 20260731 and the
  consumed sealed v2 root is intact (its three JSON hashes match section 7 of
  HANDOFF.md), so the schema reference has not drifted;
* the isolated generation source root from HANDOFF section 6: SignalLab and
  Atom-DSP source-tree digests (same walk as the launcher: skip
  ``node_modules``, ``dist`` and ``RELEASE_SOURCE_PROVENANCE.json``, refuse
  symlinks), package-lock hashes, the built ``dist/index.js`` /
  ``dist/index.d.ts`` hashes, and the ``@atomos/dsp`` symlink resolution;
* the launcher / corpus generator / prefix deriver hashes and the Node, npm,
  npx and tsx pins.

At the end it prints every check itemised and then a single GO / NO-GO line.
Exit status 0 means GO, 1 means NO-GO, 2 means the preflight itself was
invoked unusably (for example an output path that already exists).

It also RECONSTRUCTS, and never runs, the exact generation command the
orchestrator should use for seed 20260731; the command is printed and stored
in the report so the one-shot run cannot be improvised.

The Apple Accelerate spurious-IEEE-flag pattern applies here exactly as in
``noise_prefilter._dot``: this platform's BLAS can raise divide/overflow flags
on ordinary finite float64 products, so the smoke check exercises ``_dot``
itself rather than a bare ``@`` under ``PYTHONWARNINGS=error``.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent          # .../v2_full_variation/v3_scale
V2 = HERE.parent                                # .../v2_full_variation
ZPAB = V2.parent                                # .../zplane_ab
TRAINING = ZPAB.parent                          # .../training
REPO = TRAINING.parent                          # repo root
for _path in (TRAINING, ZPAB, V2, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


REPORT_SCHEMA = "v3-release-preflight-v1"

# ---------------------------------------------------------------------------
# frozen expectations
#
# Every hash below is a pin from the evidence trail, not a value computed here:
# HANDOFF sections 2 (seed rules), 6 (generation protocol and dependency
# digests), 7 (consumed sealed v2 root) and the frozen policy modules.  A pin
# that no longer matches disk is a NO-GO, never something to update casually.
# ---------------------------------------------------------------------------

DEFAULT_PINS: dict[str, Any] = {
    "release_seed": 20260731,
    "sealed_seed": 20260729,
    "target_per_class": 192,
    # HANDOFF 6: evaluator SHA recorded in the sealed v2 result.
    "v2_evaluator_sha256": (
        "1b8137b4c222a857a91f340730137fefd3fe17a026d9ba5eb172e7fd774c0541"
    ),
    # HANDOFF 6: frozen generation source hashes.  The launcher pin was
    # deliberately advanced from the sealed-v2 bytes (552ada69...) after the
    # launcher gained RELEASE_EVALUATION_PROTOCOL selection: 'v2' (default)
    # still embeds the byte-identical historical protocol object, and 'v3'
    # embeds the staged protocol from the pinned fixture printed by
    # evaluate_v3_release_suite.py --print-expected-protocol (which itself
    # refuses any suite whose intent protocol differs).  The v2 bytes remain
    # recorded in the consumed seed-20260729 RELEASE_INTENT/MANIFEST.
    "launcher_sha256": (
        "461508ee5b39a3cdfe7b7fce6fd3b229e45ec013477b90edb2373850e9fe969f"
    ),
    "corpus_generator_sha256": (
        "305418a5bc7bd8f9a49799477f3a457b4c07d0c58b637766989fc9557565371b"
    ),
    "prefix_deriver_sha256": (
        "d9383a642d21a59f66f1f9e88fc7ce51ad50893a381985c4f53f2b0da9aec3a3"
    ),
    # HANDOFF 6: pinned runtime.
    "node_version": "v22.23.1",
    "npm_version": "10.9.8",
    "tsx_package": "tsx@4.20.3",
    # HANDOFF 6: pinned dependency digests of the isolated source root.
    "signallab": {
        "source_tree_sha256": (
            "93bec60b8da571bd0cf4a51630bd2b053d95eaf1bf2da503251775545fb873c0"
        ),
        "source_file_count": 81,
        "package_lock_sha256": (
            "5dd71450021febce4acbb9a835a3cadea659e5b4ee088217b9b697bf3f0a5fa6"
        ),
    },
    "atom_dsp": {
        "source_tree_sha256": (
            "80a8363407acfa60ea366a1b6f83401a07797ef7c9776579bc949eab16e04588"
        ),
        "source_file_count": 20,
        "package_lock_sha256": (
            "b1c8198c18a4b25ed605db0781ae57b6050e0123ab0696ca2bdcba112696c0ca"
        ),
        "dist_index_js_sha256": (
            "caef3a8b3cbf4c300f8ceab5619a1db6b4fcebb942d428577f50790a1fb360e7"
        ),
        "dist_index_dts_sha256": (
            "fc6933a4c7e7d9aeaa2ca22043f6e22b2333671c08b81b1cd35b29a7de08c980"
        ),
    },
    # HANDOFF 7: the consumed sealed v2 root, preserved as negative evidence
    # and used by the v3 evaluator only as a read-only schema reference.
    "sealed_root_name": "invariant_fusion_v2_sealed_seed20260729",
    "sealed_json_sha256": {
        "RELEASE_INTENT.json": (
            "62040f8edd81bdda0cd99eb6f1b89af41c7c28d33c62ed2a58d977f591b49d9c"
        ),
        "RELEASE_MANIFEST.json": (
            "a710f2139aab7c52399936be2bb45ba009ff90e73cdbab36848e4391267694d3"
        ),
        "RELEASE_EVALUATION.json": (
            "dfb815d604aeb234480dcbbe1712d08d6bba3a81c53fdb490c0707b8b3ce8e1b"
        ),
    },
    # Frozen open-set policy (v3_time_domain_openset, HANDOFF 15.2/20).
    "frozen_policy": {
        "schema": 1,
        "kind": "v3_known_only_lof_frequency_dispersion_rank_blend",
        "geometry_feature": "across_patch_frequency_dispersion",
        "v2_weight": 0.80,
        "geometry_weight": 0.20,
        "threshold_quantile": 0.95,
    },
    "staged_policy_kind": (
        "v3_staged_noise_prefilter_then_known_only_lof_geometry"
    ),
    # Staged prefilter provenance (HANDOFF 20).
    "prefilter_version": "noise-prefilter-v1",
    "prefilter_lengths": (4096, 8192, 16384),
    "prefilter_fitting_seed": 20261001,
    "prefilter_fitting_namespace": (20261000, 20261999),
    # Bundle schema identity (export_v3_fusion_runtime).
    "bundle_schema": "atomos.v3.time-domain-invariant-fusion.runtime-bundle",
    "bundle_schema_version": 1,
}

# The default candidate, frozen.  Overridable on the CLI so the test suite can
# exercise every check against synthetic trees, never so a different candidate
# can be slipped in silently: the resolved paths are recorded in the report.
DEFAULT_FUSION_DIR = (
    V2
    / "artifacts/invariant_patch/v3_scale/v3_fusion_multilength_seed20260730"
)
DEFAULT_BUNDLE_DIR = (
    V2 / "artifacts/invariant_patch/v3_scale/v3_runtime_bundle_seed20260730"
)
DEFAULT_PREFILTER_ROOT = (
    V2 / "artifacts/invariant_patch/v3_scale/noise_prefilter_fit20261001"
)
DEFAULT_RELEASES_DIR = TRAINING / "artifacts/releases"
DEFAULT_ISOLATED_ROOT = Path(
    "/private/tmp/atomos-release-source-final.RvbW7V/Atom-SignalLab"
)
DEFAULT_EVALUATOR = HERE / "evaluate_v3_release_suite.py"
DEFAULT_NODE_BIN_DIR = Path(
    "/Users/johnelliott/.nvm/versions/node/v22.23.1/bin"
)

# Source files whose hashes travel into the report as evaluator dependencies.
# The two frontend files additionally carry a pin through the runtime bundle's
# ``frontend.source_sha256`` record and are cross-checked there.
DEPENDENCY_SOURCES: tuple[tuple[str, Path], ...] = (
    ("v3_scale/evaluate_v3_release_suite.py", DEFAULT_EVALUATOR),
    (
        "v2_full_variation/evaluate_invariant_release_suite.py",
        V2 / "evaluate_invariant_release_suite.py",
    ),
    ("v2_full_variation/v3_time_domain_openset.py", V2 / "v3_time_domain_openset.py"),
    ("v2_full_variation/invariant_fusion.py", V2 / "invariant_fusion.py"),
    ("v2_full_variation/invariant_patch_cnn.py", V2 / "invariant_patch_cnn.py"),
    ("v2_full_variation/openset_eval.py", V2 / "openset_eval.py"),
    ("v3_scale/noise_prefilter.py", HERE / "noise_prefilter.py"),
    ("v3_scale/pose_degeneracy.py", HERE / "pose_degeneracy.py"),
    ("v3_scale/fit_v3_openset.py", HERE / "fit_v3_openset.py"),
    ("v3_scale/fit_v3_openset_staged.py", HERE / "fit_v3_openset_staged.py"),
    ("v3_scale/assemble_v3_fusion.py", HERE / "assemble_v3_fusion.py"),
    ("v3_scale/export_v3_fusion_runtime.py", HERE / "export_v3_fusion_runtime.py"),
    ("training/time_domain_geometry.py", TRAINING / "time_domain_geometry.py"),
    (
        "training/time_domain_invariant_patch_preprocess.py",
        TRAINING / "time_domain_invariant_patch_preprocess.py",
    ),
)

# The fusion artifact's ``artifacts`` record: JSON key -> (file key, sha key).
FUSION_ASSET_KEYS: tuple[tuple[str, str], ...] = (
    ("state_dict", "state_dict_sha256"),
    ("prototypes", "prototypes_sha256"),
    ("real_center", "real_center_sha256"),
    ("complex_center", "complex_center_sha256"),
    ("feature_mean", "feature_mean_sha256"),
    ("feature_std", "feature_std_sha256"),
)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def source_tree_digest(root: Path) -> tuple[str, int]:
    """The launcher's ``sourceTreeDigest`` walk, byte for byte.

    Entries sorted by name; ``node_modules``, ``dist`` and
    ``RELEASE_SOURCE_PROVENANCE.json`` skipped at every level; symlinks and
    special files refused; each file contributes ``utf8(relative) + 0x00 +
    contents``.  Matching the JavaScript implementation exactly is the point:
    the digest this computes must equal the digest the launcher will compute.
    """
    digest = hashlib.sha256()
    count = 0

    def walk(directory: Path, prefix: str) -> None:
        nonlocal count
        with os.scandir(directory) as iterator:
            entries = sorted(iterator, key=lambda item: item.name)
        for entry in entries:
            if entry.name in (
                "node_modules",
                "dist",
                "RELEASE_SOURCE_PROVENANCE.json",
            ):
                continue
            relative = f"{prefix}/{entry.name}" if prefix else entry.name
            if entry.is_symlink():
                raise RuntimeError(
                    "release dependency source may not contain symlink "
                    f"{relative}"
                )
            if entry.is_dir(follow_symlinks=False):
                walk(Path(entry.path), relative)
            elif entry.is_file(follow_symlinks=False):
                digest.update(relative.encode("utf-8"))
                digest.update(b"\x00")
                digest.update(Path(entry.path).read_bytes())
                count += 1
            else:
                raise RuntimeError(
                    "release dependency source contains unsupported entry "
                    f"{relative}"
                )

    walk(Path(root), "")
    return digest.hexdigest(), count


def _command_version(binary: Path) -> str:
    result = subprocess.run(
        [str(binary), "--version"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{binary} --version failed: {result.stderr.strip()!r}"
        )
    return result.stdout.strip()


Check = tuple[str, bool, str]


def _match(name: str, actual: Any, expected: Any) -> Check:
    ok = actual == expected
    if ok:
        return (name, True, f"{actual}")
    return (name, False, f"expected {expected!r}, observed {actual!r}")


# ---------------------------------------------------------------------------
# check groups; each returns a list of (name, passed, detail)
# ---------------------------------------------------------------------------


def check_fusion_artifact(fusion_dir: Path) -> list[Check]:
    checks: list[Check] = []
    metrics = _read_json(Path(fusion_dir) / "dev_metrics.json")
    artifacts = metrics.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return [("fusion.artifacts_record", False, "no artifacts record")]
    bad: list[str] = []
    for file_key, sha_key in FUSION_ASSET_KEYS:
        filename = str(artifacts[file_key])
        expected = str(artifacts[sha_key])
        actual = _sha256(Path(fusion_dir) / filename)
        if actual != expected:
            bad.append(f"{filename}: expected {expected}, observed {actual}")
    checks.append(
        (
            "fusion.asset_sha256",
            not bad,
            "; ".join(bad) if bad else f"{len(FUSION_ASSET_KEYS)} assets match",
        )
    )
    checks.append(
        _match(
            "fusion.consumed_test_rows_used",
            metrics.get("consumed_test_rows_used"),
            0,
        )
    )
    checks.append(
        _match(
            "fusion.sealed_release_data_used",
            metrics.get("sealed_release_data_used"),
            0,
        )
    )
    return checks


def check_runtime_bundle(
    bundle_dir: Path,
    fusion_dir: Path,
    pins: Mapping[str, Any],
) -> list[Check]:
    checks: list[Check] = []
    bundle_dir = Path(bundle_dir)
    manifest = _read_json(bundle_dir / "bundle_manifest.json")
    checks.append(
        _match("bundle.schema", manifest.get("schema"), pins["bundle_schema"])
    )
    checks.append(
        _match(
            "bundle.schema_version",
            manifest.get("schema_version"),
            pins["bundle_schema_version"],
        )
    )
    assets = manifest.get("assets")
    if not isinstance(assets, Mapping) or not assets:
        checks.append(("bundle.asset_sha256", False, "no assets record"))
    else:
        bad = []
        for name, record in sorted(assets.items()):
            path = bundle_dir / str(name)
            actual = _sha256(path)
            if actual != record.get("sha256"):
                bad.append(f"{name}: expected {record.get('sha256')}, observed {actual}")
            elif int(record.get("bytes", -1)) != path.stat().st_size:
                bad.append(f"{name}: byte count mismatch")
        checks.append(
            (
                "bundle.asset_sha256",
                not bad,
                "; ".join(bad) if bad else f"{len(assets)} assets match",
            )
        )
    provenance = manifest.get("provenance", {})
    dev_metrics_sha = _sha256(Path(fusion_dir) / "dev_metrics.json")
    checks.append(
        _match(
            "bundle.bound_to_candidate_fusion",
            provenance.get("source_dev_metrics_sha256"),
            dev_metrics_sha,
        )
    )
    frontend_pins = manifest.get("frontend", {}).get("source_sha256", {})
    if not frontend_pins:
        checks.append(("bundle.frontend_source_pins", False, "no frontend pins"))
    else:
        bad = []
        for relative, expected in sorted(frontend_pins.items()):
            actual = _sha256(REPO / relative)
            if actual != expected:
                bad.append(f"{relative}: expected {expected}, observed {actual}")
        checks.append(
            (
                "bundle.frontend_source_pins",
                not bad,
                "; ".join(bad)
                if bad
                else f"{len(frontend_pins)} frontend sources match the repo",
            )
        )
    rejection = manifest.get("rejection", {})
    contract = rejection.get("required_contract", {})
    frozen = pins["frozen_policy"]
    contract_ok = (
        rejection.get("state") == "unset"
        and contract.get("policy_schema") == frozen["schema"]
        and contract.get("policy_kind") == frozen["kind"]
        and contract.get("geometry_feature") == frozen["geometry_feature"]
        and float(contract.get("v2_weight", -1)) == frozen["v2_weight"]
        and float(contract.get("geometry_weight", -1)) == frozen["geometry_weight"]
        and float(contract.get("threshold_quantile", -1))
        == frozen["threshold_quantile"]
    )
    checks.append(
        (
            "bundle.rejection_slot_contract",
            contract_ok,
            "state unset; required contract matches the frozen policy"
            if contract_ok
            else f"state={rejection.get('state')!r}, contract={contract!r}",
        )
    )
    return checks


def check_bundle_self_verification(bundle_dir: Path) -> list[Check]:
    """Reload the bundle from its own files and replay the probe fixture."""
    import export_v3_fusion_runtime as exporter

    bundle_dir = Path(bundle_dir)
    manifest = _read_json(bundle_dir / "bundle_manifest.json")
    fixture = _read_json(bundle_dir / exporter.PROBE_NAME)
    expected_names = [str(item["name"]) for item in exporter.PROBE_SPECIFICATIONS]
    fixture_names = [str(case["name"]) for case in fixture.get("cases", ())]
    if fixture_names != expected_names:
        return [
            (
                "bundle.self_verification",
                False,
                f"probe cases {fixture_names} != specifications {expected_names}",
            )
        ]
    cases = fixture["cases"]

    def stack(key: str, dtype: Any) -> np.ndarray:
        return np.asarray(
            [case["expected"][key] for case in cases], dtype=dtype
        )

    packed = np.asarray(
        [
            np.asarray(case["expected"]["packed_iq"], dtype=np.float32).reshape(
                2, -1
            )
            for case in cases
        ],
        dtype=np.float32,
    )
    expected = {
        "packed": packed,
        "raw_features": stack("raw_features", np.float32),
        "standardized_features": stack("standardized_features", np.float32),
        "real_embedding": stack("real_embedding", np.float32),
        "complex_embedding": stack("complex_embedding", np.float32),
        "fused_embedding": stack("fused_embedding", np.float32),
        "prototypes": np.load(
            bundle_dir / "fusion_prototypes.npy", allow_pickle=False
        ),
        "classes": list(manifest["classification"]["classes"]),
    }
    verification = exporter.verify_bundle(
        bundle_dir,
        expected,
        tolerance=float(fixture["tolerance"]),
    )
    worst = float(verification["worst_max_abs_error"])
    return [
        (
            "bundle.self_verification",
            True,
            f"{len(cases)} probe cases replayed from bundle-only inputs, "
            f"worst |error| {worst:.3g} <= {float(fixture['tolerance']):.3g}, "
            "closed labels agree",
        )
    ]


def check_prefilter(prefilter_root: Path, pins: Mapping[str, Any]) -> list[Check]:
    import fit_v3_openset as openset_base
    import noise_prefilter as npf

    checks: list[Check] = []
    prefilter_root = Path(prefilter_root)
    bundles_dir = prefilter_root / "bundles"
    report = _read_json(prefilter_root / "noise_prefilter_fit.json")
    checks.append(
        _match("prefilter.version", report.get("version"), pins["prefilter_version"])
    )
    recorded_set = str(report.get("prefilter_set_sha256", ""))
    actual_set = npf.prefilter_set_sha256(bundles_dir)
    checks.append(_match("prefilter.set_sha256", actual_set, recorded_set))
    recorded_each = report.get("bundle_sha256", {})
    expected_names = {f"N{length}" for length in pins["prefilter_lengths"]}
    bad = []
    if set(recorded_each) != expected_names:
        bad.append(
            f"recorded lengths {sorted(recorded_each)} != {sorted(expected_names)}"
        )
    for name in sorted(recorded_each):
        actual = npf.bundle_sha256(bundles_dir / name)
        if actual != recorded_each[name]:
            bad.append(f"{name}: expected {recorded_each[name]}, observed {actual}")
    checks.append(
        (
            "prefilter.per_length_sha256",
            not bad,
            "; ".join(bad)
            if bad
            else f"{len(recorded_each)} per-length bundles match",
        )
    )
    contract_ok = (
        report.get("additive_only") is False
        and report.get("changes_closed_label") is True
        and report.get("gates_before_classification") is True
        and bool(str(report.get("architecture_contract_change", "")).strip())
    )
    checks.append(
        (
            "prefilter.architecture_contract_keys",
            contract_ok,
            "additive_only false, changes_closed_label true, "
            "gates_before_classification true, statement present"
            if contract_ok
            else {
                key: report.get(key)
                for key in npf.CONTRACT_KEYS_FOR_RELEASE_CLAIM
            },
        )
    )
    seeds = report.get("seeds", {})
    fitting_seed = seeds.get("fitting_seed")
    seed_ok = (
        fitting_seed == pins["prefilter_fitting_seed"]
        and list(seeds.get("fitting_seed_namespace", ()))
        == list(pins["prefilter_fitting_namespace"])
        and seeds.get("release_seed_not_spent") == pins["release_seed"]
        and seeds.get("sealed_seed_not_read") == pins["sealed_seed"]
        and npf.validate_fitting_seed(int(fitting_seed)) == int(fitting_seed)
    )
    checks.append(
        (
            "prefilter.fitting_seed_hygiene",
            seed_ok,
            f"fitting seed {fitting_seed} in namespace "
            f"{list(pins['prefilter_fitting_namespace'])}, disjoint from the "
            "novelty namespace and the release seed"
            if seed_ok
            else f"seeds record {seeds!r}",
        )
    )
    bad = []
    try:
        models = npf.load_prefilter_set(bundles_dir)
    except Exception as error:  # noqa: BLE001 - a refused bundle is a finding
        models = {}
        bad.append(f"load refused: {type(error).__name__}: {error}")
    if models and sorted(models) != sorted(
        int(v) for v in pins["prefilter_lengths"]
    ):
        bad.append(f"loaded lengths {sorted(models)}")
    for length, model in sorted(models.items()):
        if int(model.capture_length) != int(length):
            bad.append(f"N{length}: capture_length {model.capture_length}")
        if not model.has_threshold:
            bad.append(f"N{length}: no operating threshold")
        elif not np.isfinite(float(model.threshold_score)):
            bad.append(f"N{length}: non-finite threshold")
        if tuple(model.feature_names) != tuple(npf.PREFILTER_FEATURES):
            bad.append(f"N{length}: feature names differ from PREFILTER_FEATURES")
    checks.append(
        (
            "prefilter.models_loadable",
            not bad,
            "; ".join(bad)
            if bad
            else f"{len(models)} thresholded per-length models load",
        )
    )
    floors = report.get("gate_floors", {})
    floors_ok = (
        float(floors.get("noise_auroc", -1))
        == float(openset_base.GATE_FLOORS["noise_auroc"])
        and float(floors.get("noise_threshold_recall", -1))
        == float(openset_base.GATE_FLOORS["noise_threshold_recall"])
        and float(floors.get("known_false_unknown_ceiling", -1))
        == float(openset_base.KNOWN_FUR_CEILING)
    )
    checks.append(
        (
            "prefilter.gate_floors_match_frozen",
            floors_ok,
            f"{floors}" if floors_ok else f"recorded {floors!r}",
        )
    )
    return checks


def check_frozen_policy(pins: Mapping[str, Any]) -> list[Check]:
    import fit_v3_openset as openset_base
    import fit_v3_openset_staged as staged
    import v3_time_domain_openset as openset

    frozen = pins["frozen_policy"]
    checks: list[Check] = []
    checks.append(
        _match("policy.schema", openset.FROZEN_POLICY_SCHEMA, frozen["schema"])
    )
    checks.append(_match("policy.kind", openset.FROZEN_POLICY_KIND, frozen["kind"]))
    checks.append(
        _match(
            "policy.geometry_feature",
            openset.FROZEN_GEOMETRY_FEATURE,
            frozen["geometry_feature"],
        )
    )
    checks.append(
        _match("policy.v2_weight", float(openset.FROZEN_V2_WEIGHT), frozen["v2_weight"])
    )
    checks.append(
        _match(
            "policy.geometry_weight",
            float(openset.FROZEN_GEOMETRY_WEIGHT),
            frozen["geometry_weight"],
        )
    )
    checks.append(
        _match(
            "policy.threshold_quantile",
            float(openset.FROZEN_THRESHOLD_QUANTILE),
            frozen["threshold_quantile"],
        )
    )
    identity_ok = (
        staged.GATE_FLOORS is openset_base.GATE_FLOORS
        and staged.KNOWN_FUR_CEILING == openset_base.KNOWN_FUR_CEILING
        and float(openset_base.BRANCH_LOF_RANK_WEIGHT)
        == float(openset.FROZEN_V2_WEIGHT)
        and float(openset_base.GEOMETRY_RANK_WEIGHT)
        == float(openset.FROZEN_GEOMETRY_WEIGHT)
        and float(openset_base.THRESHOLD_QUANTILE)
        == float(openset.FROZEN_THRESHOLD_QUANTILE)
    )
    checks.append(
        (
            "policy.import_identity_discipline",
            identity_ok,
            "staged gates are the fit_v3_openset objects; weights are the "
            "frozen constants"
            if identity_ok
            else "a frozen constant was re-typed somewhere in the chain",
        )
    )
    checks.append(
        _match(
            "policy.staged_kind",
            staged.STAGED_POLICY_KIND,
            pins["staged_policy_kind"],
        )
    )
    ledger_ok = (
        int(openset_base.RELEASE_SEED_NEVER_SPENT_HERE) == pins["release_seed"]
        and int(staged.SEALED_RELEASE_SEED) == pins["sealed_seed"]
        and sorted(staged.SPENT_NOVELTY_SEEDS)
        == [20260938, 20260939, 20260940, 20260941]
    )
    checks.append(
        (
            "policy.seed_ledger_constants",
            ledger_ok,
            f"release seed {pins['release_seed']} unreachable from the fit "
            "modules; spent novelty seeds 20260938-20260941"
            if ledger_ok
            else "module seed ledger no longer matches the HANDOFF ledger",
        )
    )
    return checks


def check_evaluator_sources(
    evaluator_path: Path,
    pins: Mapping[str, Any],
    observed: dict[str, Any],
) -> list[Check]:
    checks: list[Check] = []
    v2_evaluator = V2 / "evaluate_invariant_release_suite.py"
    checks.append(
        _match(
            "sources.v2_evaluator_sha256",
            _sha256(v2_evaluator),
            pins["v2_evaluator_sha256"],
        )
    )
    evaluator_path = Path(evaluator_path)
    if not evaluator_path.is_file():
        checks.append(
            (
                "sources.v3_evaluator_present",
                False,
                f"{evaluator_path} does not exist; build it before preflight",
            )
        )
    else:
        import tempfile

        with tempfile.TemporaryDirectory() as scratch:
            py_compile.compile(
                str(evaluator_path),
                cfile=str(Path(scratch) / "evaluator_check.pyc"),
                doraise=True,
            )
        sha = _sha256(evaluator_path)
        observed["v3_evaluator_sha256"] = sha
        checks.append(
            (
                "sources.v3_evaluator_present",
                True,
                f"byte-compiles; sha256 {sha}",
            )
        )
    hashes: dict[str, str] = {}
    missing = []
    for label, path in DEPENDENCY_SOURCES:
        path = Path(path)
        if label == "v3_scale/evaluate_v3_release_suite.py":
            path = evaluator_path
        if path.is_file():
            hashes[label] = _sha256(path)
        else:
            missing.append(label)
    observed["dependency_source_sha256"] = hashes
    checks.append(
        (
            "sources.dependency_hashes_recorded",
            not missing,
            f"{len(hashes)} dependency sources hashed into the report"
            if not missing
            else f"missing: {missing}",
        )
    )
    v2_floors_ok = False
    try:
        import evaluate_invariant_release_suite as release
        import fit_v3_openset as openset_base

        v2_floors_ok = (
            float(release.GATE_FLOORS["open_auroc_noise"])
            == float(openset_base.GATE_FLOORS["noise_auroc"])
            and float(release.GATE_FLOORS["open_auroc_chirp"])
            == float(openset_base.GATE_FLOORS["chirp_auroc"])
            and float(release.GATE_FLOORS["open_auroc_overall"])
            == float(openset_base.GATE_FLOORS["overall_auroc"])
            and float(release.GATE_FLOORS["open_unknown_recall_noise"])
            == float(openset_base.GATE_FLOORS["noise_threshold_recall"])
            and float(release.GATE_FLOORS["open_unknown_recall_chirp"])
            == float(openset_base.GATE_FLOORS["chirp_threshold_recall"])
            and float(release.GATE_FLOORS["open_known_false_unknown_max"])
            == float(openset_base.KNOWN_FUR_CEILING)
            and len(release.GATE_FLOORS) == 17
        )
        detail = (
            "the 17 sealed gate floors are importable and the open-set subset "
            "equals the development floors"
            if v2_floors_ok
            else f"release.GATE_FLOORS drifted: {release.GATE_FLOORS!r}"
        )
    except Exception as error:  # noqa: BLE001 - reported, not swallowed
        detail = f"{type(error).__name__}: {error}"
    checks.append(("sources.gate_floor_consistency", v2_floors_ok, detail))
    return checks


def check_seed_ledger(releases_dir: Path, pins: Mapping[str, Any]) -> list[Check]:
    checks: list[Check] = []
    releases_dir = Path(releases_dir)
    seed_text = str(pins["release_seed"])
    problems: list[str] = []
    if releases_dir.is_dir():
        for child in sorted(releases_dir.iterdir()):
            if seed_text in child.name:
                problems.append(f"release root {child.name} names the seed")
            intent_path = child / "RELEASE_INTENT.json"
            if child.is_dir() and intent_path.is_file():
                intent = _read_json(intent_path)
                if int(intent.get("release_seed", -1)) == int(pins["release_seed"]):
                    problems.append(
                        f"{child.name}/RELEASE_INTENT.json records the seed"
                    )
    checks.append(
        (
            "ledger.release_seed_unused",
            not problems,
            f"seed {seed_text} appears in no release root under "
            f"{releases_dir}"
            if not problems
            else "; ".join(problems),
        )
    )
    sealed_root = releases_dir / str(pins["sealed_root_name"])
    bad = []
    for name, expected in sorted(pins["sealed_json_sha256"].items()):
        path = sealed_root / name
        if not path.is_file():
            bad.append(f"{name} missing")
            continue
        actual = _sha256(path)
        if actual != expected:
            bad.append(f"{name}: expected {expected}, observed {actual}")
    checks.append(
        (
            "ledger.sealed_v2_reference_intact",
            not bad,
            "consumed sealed v2 root JSONs match HANDOFF section 7"
            if not bad
            else "; ".join(bad),
        )
    )
    return checks


def check_isolated_source(
    isolated_root: Path,
    pins: Mapping[str, Any],
) -> list[Check]:
    checks: list[Check] = []
    signallab_root = Path(isolated_root)
    if not signallab_root.is_dir():
        return [
            (
                "isolated.root_exists",
                False,
                f"{signallab_root} is missing; HANDOFF 6 warns the isolated "
                "root is temporary and must be rebuilt before release",
            )
        ]
    checks.append(("isolated.root_exists", True, str(signallab_root)))
    link = signallab_root / "node_modules" / "@atomos" / "dsp"
    atom_dsp_root = (signallab_root.parent / "Atom-DSP").resolve()
    link_ok = link.is_symlink() and link.resolve() == atom_dsp_root
    checks.append(
        (
            "isolated.atom_dsp_symlink",
            link_ok,
            f"@atomos/dsp resolves to {atom_dsp_root}"
            if link_ok
            else f"@atomos/dsp is not the expected symlink to {atom_dsp_root}",
        )
    )
    signallab_digest, signallab_count = source_tree_digest(signallab_root)
    checks.append(
        _match(
            "isolated.signallab_source_digest",
            (signallab_digest, signallab_count),
            (
                pins["signallab"]["source_tree_sha256"],
                pins["signallab"]["source_file_count"],
            ),
        )
    )
    dsp_digest, dsp_count = source_tree_digest(atom_dsp_root)
    checks.append(
        _match(
            "isolated.atom_dsp_source_digest",
            (dsp_digest, dsp_count),
            (
                pins["atom_dsp"]["source_tree_sha256"],
                pins["atom_dsp"]["source_file_count"],
            ),
        )
    )
    files = {
        "isolated.signallab_package_lock": (
            signallab_root / "package-lock.json",
            pins["signallab"]["package_lock_sha256"],
        ),
        "isolated.atom_dsp_package_lock": (
            atom_dsp_root / "package-lock.json",
            pins["atom_dsp"]["package_lock_sha256"],
        ),
        "isolated.atom_dsp_dist_index_js": (
            atom_dsp_root / "dist" / "index.js",
            pins["atom_dsp"]["dist_index_js_sha256"],
        ),
        "isolated.atom_dsp_dist_index_dts": (
            atom_dsp_root / "dist" / "index.d.ts",
            pins["atom_dsp"]["dist_index_dts_sha256"],
        ),
    }
    for name, (path, expected) in sorted(files.items()):
        if not path.is_file():
            checks.append((name, False, f"{path} missing"))
        else:
            checks.append(_match(name, _sha256(path), expected))
    return checks


def check_generation_sources(pins: Mapping[str, Any]) -> list[Check]:
    checks: list[Check] = []
    launcher = REPO / "tools/generate-signallab-iq-release-suite.mjs"
    generator = REPO / "tools/generate-signallab-iq-corpus.ts"
    deriver = REPO / "tools/derive-signallab-iq-prefix-corpus.mjs"
    checks.append(
        _match("generation.launcher_sha256", _sha256(launcher), pins["launcher_sha256"])
    )
    checks.append(
        _match(
            "generation.corpus_generator_sha256",
            _sha256(generator),
            pins["corpus_generator_sha256"],
        )
    )
    checks.append(
        _match(
            "generation.prefix_deriver_sha256",
            _sha256(deriver),
            pins["prefix_deriver_sha256"],
        )
    )
    text = launcher.read_text(encoding="utf-8")
    pin_ok = (
        pins["tsx_package"] in text
        and pins["node_version"] in text
        and pins["npm_version"] in text
    )
    checks.append(
        (
            "generation.launcher_frozen_pins",
            pin_ok,
            f"launcher source pins {pins['tsx_package']}, Node "
            f"{pins['node_version']}, npm/npx {pins['npm_version']}"
            if pin_ok
            else "launcher no longer pins the recorded runtime",
        )
    )
    return checks


def check_node_runtime(node_bin_dir: Path, pins: Mapping[str, Any]) -> list[Check]:
    checks: list[Check] = []
    node_bin_dir = Path(node_bin_dir)
    if not node_bin_dir.is_dir():
        return [
            (
                "runtime.node_bin_dir",
                False,
                f"{node_bin_dir} is missing; install Node "
                f"{pins['node_version']} via nvm",
            )
        ]
    checks.append(
        _match(
            "runtime.node_version",
            _command_version(node_bin_dir / "node"),
            pins["node_version"],
        )
    )
    checks.append(
        _match(
            "runtime.npm_version",
            _command_version(node_bin_dir / "npm"),
            pins["npm_version"],
        )
    )
    checks.append(
        _match(
            "runtime.npx_version",
            _command_version(node_bin_dir / "npx"),
            pins["npm_version"],
        )
    )
    return checks


def check_python_runtime() -> list[Check]:
    import noise_prefilter as npf
    import torch

    checks: list[Check] = []
    warnings_policy = os.environ.get("PYTHONWARNINGS", "")
    checks.append(
        (
            "runtime.pythonwarnings_error",
            warnings_policy == "error",
            f"PYTHONWARNINGS={warnings_policy!r}"
            + ("" if warnings_policy == "error" else " (must be 'error')"),
        )
    )
    rng = np.random.default_rng(0)
    left = rng.standard_normal((120, 8))
    right = rng.standard_normal((120, 8))
    product = npf._dot(left.T, right)
    dot_ok = product.shape == (8, 8) and bool(np.isfinite(product).all())
    checks.append(
        (
            "runtime.accelerate_dot_smoke",
            dot_ok,
            "noise_prefilter._dot neutralises the Apple Accelerate spurious "
            f"IEEE flags (numpy {np.__version__}, torch {torch.__version__})",
        )
    )
    return checks


# ---------------------------------------------------------------------------
# generation command reconstruction (never executed here)
# ---------------------------------------------------------------------------


def build_generation_command(
    pins: Mapping[str, Any],
    bundle_dir: Path,
    isolated_root: Path,
    node_bin_dir: Path,
) -> dict[str, Any]:
    """The exact one-shot launcher invocation for seed 20260731.

    Reconstructed from the v2 ``RELEASE_INTENT.json`` and the launcher CLI.
    The candidate file is the runtime bundle's manifest: it embeds the SHA-256
    of every other bundle asset, so pinning it pins the entire candidate.
    This function only formats strings; nothing is executed.
    """
    manifest_path = Path(bundle_dir) / "bundle_manifest.json"
    candidate_sha = _sha256(manifest_path)
    release_root = (
        REPO
        / "training/artifacts/releases"
        / f"invariant_fusion_v3_sealed_seed{pins['release_seed']}"
    )
    path_env = (
        f"{node_bin_dir}:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:"
        "/usr/sbin:/sbin"
    )
    environment = {
        "PATH": path_env,
        "RELEASE_ROOT": str(release_root),
        "RELEASE_SEED": str(pins["release_seed"]),
        "RELEASE_TARGET_PER_CLASS": str(pins["target_per_class"]),
        "CANDIDATE_PATH": str(manifest_path),
        "CANDIDATE_SHA256": candidate_sha,
        "SIGNALLAB_ROOT": str(isolated_root),
    }
    assignments = " \\\n  ".join(
        f"{key}={value}" for key, value in environment.items()
    )
    command = (
        f"cd {REPO} && \\\n  env {assignments} \\\n"
        "  node tools/generate-signallab-iq-release-suite.mjs"
    )
    return {
        "not_run_by_preflight": True,
        "environment": environment,
        "command": command,
        "candidate_manifest_sha256": candidate_sha,
        "notes": [
            "RELEASE_LENGTHS defaults to 4096,8192,16384,32768 and the "
            "launcher refuses any other value.",
            "RELEASE_TARGET_PER_CLASS=192 keeps the metadata-only no-alias "
            "scale subset safely above 20 rows per class.",
            "The launcher freezes evaluation_protocol.novelty.seed at "
            "20260729; the v3 evaluator must declare how it derives novelty "
            "seeds before this intent is written.",
        ],
    }


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


def run_preflight(
    args: argparse.Namespace,
    pins: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    pins = copy.deepcopy(DEFAULT_PINS) if pins is None else pins
    observed: dict[str, Any] = {}
    checks: list[dict[str, Any]] = []

    def run_group(label: str, group: Callable[[], list[Check]]) -> None:
        try:
            results = group()
        except Exception as error:  # noqa: BLE001 - a crash is a failed check
            results = [(label, False, f"{type(error).__name__}: {error}")]
        for name, passed, detail in results:
            checks.append(
                {"name": name, "passed": bool(passed), "detail": str(detail)}
            )

    fusion_dir = Path(args.fusion_dir)
    bundle_dir = Path(args.bundle_dir)
    prefilter_root = Path(args.prefilter_root)
    releases_dir = Path(args.releases_dir)
    isolated_root = Path(args.isolated_root)
    evaluator_path = Path(args.evaluator)
    node_bin_dir = Path(args.node_bin_dir)

    run_group("fusion", lambda: check_fusion_artifact(fusion_dir))
    run_group(
        "bundle", lambda: check_runtime_bundle(bundle_dir, fusion_dir, pins)
    )
    run_group(
        "bundle.self_verification",
        lambda: check_bundle_self_verification(bundle_dir),
    )
    run_group("prefilter", lambda: check_prefilter(prefilter_root, pins))
    run_group("policy", lambda: check_frozen_policy(pins))
    run_group(
        "sources",
        lambda: check_evaluator_sources(evaluator_path, pins, observed),
    )
    run_group("ledger", lambda: check_seed_ledger(releases_dir, pins))
    run_group("isolated", lambda: check_isolated_source(isolated_root, pins))
    run_group("generation", lambda: check_generation_sources(pins))
    run_group("runtime.node", lambda: check_node_runtime(node_bin_dir, pins))
    run_group("runtime.python", check_python_runtime)

    try:
        generation = build_generation_command(
            pins, bundle_dir, isolated_root, node_bin_dir
        )
    except Exception as error:  # noqa: BLE001 - reported, not fatal on its own
        generation = {
            "not_run_by_preflight": True,
            "error": f"{type(error).__name__}: {error}",
        }
        checks.append(
            {
                "name": "generation.command_reconstruction",
                "passed": False,
                "detail": generation["error"],
            }
        )

    go = all(item["passed"] for item in checks)
    report = {
        "schema": REPORT_SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "preflight_source_sha256": _sha256(Path(__file__)),
        "release_seed": pins["release_seed"],
        "go": go,
        "checks": checks,
        "pins": copy.deepcopy(dict(pins)),
        "observed": observed,
        "paths": {
            "fusion_dir": str(fusion_dir),
            "bundle_dir": str(bundle_dir),
            "prefilter_root": str(prefilter_root),
            "releases_dir": str(releases_dir),
            "isolated_root": str(isolated_root),
            "evaluator": str(evaluator_path),
            "node_bin_dir": str(node_bin_dir),
        },
        "generation_command": generation,
        "consumes_nothing": {
            "release_seed_drawn": False,
            "novelty_realizations_generated": 0,
            "corpus_rows_read": 0,
            "sealed_labels_read": False,
            "writes_under_releases": False,
        },
    }
    return report


def _print_report(report: Mapping[str, Any], stream=None) -> None:
    stream = sys.stdout if stream is None else stream
    for item in report["checks"]:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"{status}  {item['name']}: {item['detail']}", file=stream)
    generation = report.get("generation_command", {})
    if "command" in generation:
        print(
            "\nGeneration command for the orchestrator (NOT run by "
            "preflight):\n" + generation["command"] + "\n",
            file=stream,
        )
    total = len(report["checks"])
    passed = sum(1 for item in report["checks"] if item["passed"])
    if report["go"]:
        print(
            f"GO: {passed}/{total} checks passed; safe to spend release seed "
            f"{report['release_seed']} once",
            file=stream,
        )
    else:
        print(
            f"NO-GO: {total - passed} of {total} checks FAILED; do not spend "
            f"release seed {report['release_seed']}",
            file=stream,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fusion-dir", default=str(DEFAULT_FUSION_DIR))
    parser.add_argument("--bundle-dir", default=str(DEFAULT_BUNDLE_DIR))
    parser.add_argument("--prefilter-root", default=str(DEFAULT_PREFILTER_ROOT))
    parser.add_argument("--releases-dir", default=str(DEFAULT_RELEASES_DIR))
    parser.add_argument("--isolated-root", default=str(DEFAULT_ISOLATED_ROOT))
    parser.add_argument("--evaluator", default=str(DEFAULT_EVALUATOR))
    parser.add_argument("--node-bin-dir", default=str(DEFAULT_NODE_BIN_DIR))
    parser.add_argument(
        "--output",
        default=None,
        help="optional JSON report path; refused if it already exists",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output is not None:
        output = Path(args.output).expanduser().resolve()
        if output.exists():
            print(
                f"refusing to overwrite existing preflight report {output}",
                file=sys.stderr,
            )
            return 2
        releases_dir = Path(args.releases_dir).resolve()
        if releases_dir == output or releases_dir in output.parents:
            print(
                "refusing to write the preflight report under the release "
                f"tree {releases_dir}",
                file=sys.stderr,
            )
            return 2
    report = run_preflight(args)
    _print_report(report)
    if args.output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return 0 if report["go"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
