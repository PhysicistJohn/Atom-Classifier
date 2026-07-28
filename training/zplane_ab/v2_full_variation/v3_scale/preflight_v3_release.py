"""Preflight for the next one-shot v3 sealed release run on seed 20260735.

This script runs WITHOUT any release suite and consumes nothing: no corpus is
read, no seed is drawn, no novelty realization is generated, and nothing is
written under ``training/artifacts/releases``.  It exists so the orchestrator
can prove, immediately before spending the unspent release seed, that every
frozen input of the one-shot run is exactly what the evidence trail says it is.

Release seeds 20260729, 20260731, 20260733 and 20260734 are consumed and
immutable negative evidence.  The seed-20260734 protocol used predeclared
0.12 known-FUR and 0.84 five-shot levels; this fact remains visible as
explicitly inactive historical metadata.  Seed 20260735 restores the original
strict imported gates: known FUR <= 0.10 and five-shot >= 0.85.

* the stage-1 prefilter set is the TIGHTENED 0.01 enrollment-budget refit
  (``noise_prefilter_fit20261001_budget001``): coefficients bit-identical to
  the 0.02 set, threshold only;
* the next staged artifact must validate the COMPOSITE survivor-score policy
  (stage-2 rank maxed with the stage-1 score's enrollment-survivor rank,
  unknown threshold re-fit at q95 of the composite on enrollment survivors)
  under an explicit policy version, on reserved validation novelty seeds
  20260950/20260951 with design seed 20260949;
* the launcher's pinned v3 protocol fixture must predeclare seed 20260735,
  carry the strict imported gate levels, and preserve the seed-20260734
  redeclaration only in an inactive historical block.

What it verifies, each as an itemised PASS/FAIL check:

* both fusion artifacts independently: 8k classifier and 4k rejector tree
  digests plus every recorded asset SHA-256;
* both role-labelled runtime bundles: schema identity, explicit runtime role,
  every asset hash, exact fusion binding, frontend source hashes, and a full
  replay of each bundle's probe-fixture self-verification;
* the final dual release-candidate manifest: its independent byte pin and
  complete evaluator load, transitively checking the validation evidence,
  pre-validation contract, both role bundles/fusions, validation-locked
  staged policy, canonical prefilter set, all three browser assets, the
  dual-runtime staging package and the deploy-time dual binding;
* the staged noise prefilter: fit-report schema, the per-length bundle hashes
  and the set hash, the architecture-contract keys (``additive_only`` false,
  ``changes_closed_label`` true, ``gates_before_classification`` true), the
  fitting-seed hygiene, loadability of every per-length model, that the
  recorded gate floors equal the frozen development floors, and that every
  per-length operating point declares the tightened 0.01 known-false-positive
  budget;
* the staged validation artifact: policy version, passing validate-role
  status on the declared fresh novelty seeds, its recorded prefilter binding,
  and its stage-2 npz hashes against disk;
* the frozen open-set policy constants in ``v3_time_domain_openset`` and the
  import-identity discipline of ``fit_v3_openset`` / ``fit_v3_openset_staged``
  (gates imported, never re-typed);
* evaluator sources: the v2 release evaluator still matches its recorded
  release SHA, the v3 release evaluator exists and byte-compiles, and every
  dependency source file hashes cleanly (all recorded in the report);
* the on-disk seed ledger: no release root exists for seed 20260735 and all
  four consumed sealed roots are intact, preserved as negative evidence;
* the isolated generation source root from HANDOFF section 6: SignalLab and
  Atom-DSP source-tree digests (same walk as the launcher: skip
  ``node_modules``, ``dist`` and ``RELEASE_SOURCE_PROVENANCE.json``, refuse
  symlinks), package-lock hashes, the built ``dist/index.js`` /
  ``dist/index.d.ts`` hashes, and the ``@atomos/dsp`` symlink resolution;
* the launcher / corpus generator / prefix deriver hashes, the Node, npm,
  npx and tsx pins, and the launcher's pinned v3 protocol fixture for seed
  20260735 (present, hash-pinned, strict, historically transparent, and
  actually referenced by the launcher source).

At the end it prints every check itemised and then a single GO / NO-GO line.
Exit status 0 means GO, 1 means NO-GO, 2 means the preflight itself was
invoked unusably (for example an output path that already exists).

It also RECONSTRUCTS, and never runs, the exact generation command the
orchestrator should use for seed 20260735; the command is printed and stored
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


REPORT_SCHEMA = "v3-release-preflight-v2-dual-fusion"

# ---------------------------------------------------------------------------
# frozen expectations
#
# Every hash below is a pin from the evidence trail, not a value computed here:
# HANDOFF sections 2 (seed rules), 6 (generation protocol and dependency
# digests), 7 (consumed sealed v2 root) and the frozen policy modules.  A pin
# that no longer matches disk is a NO-GO, never something to update casually.
# ---------------------------------------------------------------------------

DEFAULT_PINS: dict[str, Any] = {
    "release_seed": 20260735,
    "sealed_seed": 20260729,
    # Every release seed already consumed by a sealed run.  The launcher and
    # v3 evaluator must both refuse every entry.
    "consumed_release_seeds": (20260729, 20260731, 20260733, 20260734),
    # 20260732 is deliberately skipped: it is already a development MODEL
    # seed (HANDOFF 25 records the namespace hazard explicitly).
    "skipped_model_seed": 20260732,
    "target_per_class": 192,
    # HANDOFF 6: evaluator SHA recorded in the sealed v2 result.
    "v2_evaluator_sha256": (
        "1b8137b4c222a857a91f340730137fefd3fe17a026d9ba5eb172e7fd774c0541"
    ),
    # Exact evaluator bytes for the seed-20260735 protocol.  This evaluator
    # restores the strict v2 floors and preserves the prior redeclaration only
    # as inactive history.
    "v3_evaluator_sha256": (
        "e6875c303c817b553201cc6f2dc317330833cc387b551ead2f99eadde0039149"
    ),
    "candidate_manifest_schema": "time-domain-v3-dual-release-candidate-v1",
    # Filled only after the browser package and dual-binding bytes are final.
    # None is intentionally a hard NO-GO, never a skipped check.
    "candidate_manifest_sha256": None,
    "candidate_id": "v3.3-decoupled-8k-classifier-4k-rejector",
    "staging_package_schema": (
        "atomos.v3.time-domain-classifier.dual-runtime-package"
    ),
    "staging_package_schema_version": 1,
    "dual_binding_schema": "atomos.v3.time-domain-dual-fusion.binding",
    "dual_binding_schema_version": 1,
    "staging_status": "staging_not_release",
    "validation_evidence_sha256": (
        "85d01b9154bb1d0e60c3f42d8d5cf1854dbcbb1366b0c918250e1a4bc8c8880e"
    ),
    "frozen_candidate_contract_sha256": (
        "2209a56df22ff0c3ffd152bf2973f94da2d3488536a9cfcbf3904d077a1ba706"
    ),
    "classifier_fusion_directory_sha256": (
        "ecd894281b4cbac0917304b22ad7be194d41f994dcf2c008ca6ea0643b58667b"
    ),
    "rejector_fusion_directory_sha256": (
        "0f730e0cc2e0bdab015b08fbf6ad7b53c58851104a01616d660636937b0b82b5"
    ),
    "classifier_bundle_manifest_sha256": (
        "db216324da9b1ab8ec90cbe710c2b147b347ae29842cba20424f44c2869b99c8"
    ),
    "rejector_bundle_manifest_sha256": (
        "80ee91f3f8577507a30fdae7adc47de7e7b448d9e563558f2d0becbadb4dee78"
    ),
    "staged_validation_report_sha256": (
        "24241bcb2e47c96e855f2b3e1d213ed5c02ee7c392758af8c62ea41557ea848c"
    ),
    "staged_artifact_sha256": {
        "v3_branch_lof_components.npz": (
            "0a66ca267356bdacf239de5512b9406e010526c990dc38c137cac84342a062a6"
        ),
        "v3_open_policy_stage_two.npz": (
            "9378f6eea893174b4f716f4db5ea6ea2b637d64b04a0e7672bb39b2e50600dd8"
        ),
        "v3_staged_composite_policy.npz": (
            "7a3d5b8b1bdd67053f5f48a16d97cc2eef1fce909bd68f514c89c8bb4a64c133"
        ),
    },
    "prefilter_set_sha256": (
        "4751ae631f879bc2d987d64082c296d7150b7199fab0f1eeb6de84a0f338b5f2"
    ),
    "prefilter_bundle_sha256": {
        "4096": "5ef175f5ddc296ec9ae0ffb8ac6801504fd609d6a852dbd5c4b1bf9a10de0f85",
        "8192": "e0e972be4d1b7e43929b6c45c69d2dada884b49b39e16555998c375edf744557",
        "16384": "3d56108fc016d8fb8ff7f4c18f8fc2fa6a1443d5d32d30cd769d52de087384db",
    },
    # Independent pin for the transparent but inactive seed-20260734 record.
    "historical_gate_redeclaration": {
        "release_seed": 20260734,
        "gates": {
            "open_known_false_unknown_worst_length": {
                "v2_level": 0.10,
                "v3_level": 0.12,
                "v2_floor_key": "open_known_false_unknown_max",
            },
            "five_shot_worst_length_balanced": {
                "v2_level": 0.85,
                "v3_level": 0.84,
                "v2_floor_key": "five_shot",
            },
        },
    },
    # Frozen generation source hash.  The v2 default remains byte-compatible
    # with its historical protocol; v3 reads the seed-20260735 fixture and
    # refuses all four consumed release seeds before generating anything.
    "launcher_sha256": (
        "1bf0dc38e66a60e7a57ae8b8a96d2a2947acd27753314642228a0783f6e276a8"
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
    # The consumed sealed roots, preserved as negative evidence: the v2 root
    # (HANDOFF 7, still the evaluator's read-only schema reference), the v3
    # seed-20260731 root (HANDOFF 25, the frozen 22/23 failure) and the v3
    # seed-20260733 root (frozen 21/23 failure) and seed-20260734 root (frozen
    # 22/23 failure under its historical gate contract).  A drifted hash means
    # the frozen evidence was touched, which is a NO-GO.
    "consumed_sealed_roots": {
        "invariant_fusion_v2_sealed_seed20260729": {
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
        "invariant_fusion_v3_sealed_seed20260731": {
            "RELEASE_INTENT.json": (
                "996040b4cb4963b5a00f82d27dbc62c0609e0f83a0081c78951dd457ff99e786"
            ),
            "RELEASE_MANIFEST.json": (
                "c70831d82568932a75fbdc28a7af3331f28d94b6cf1858030f6eb8294b3c5da0"
            ),
            "RELEASE_EVALUATION.json": (
                "f408748f9c292090ecb8f93dc7b9d13eb95eb573fe0bf678ef9fd76cbfea58e5"
            ),
        },
        "invariant_fusion_v3_sealed_seed20260733": {
            "RELEASE_INTENT.json": (
                "96b890b3a4f3c9031fd4cac9ebe676cf554e71d7005c4a86d2262d56250b418c"
            ),
            "RELEASE_MANIFEST.json": (
                "19076c3d0e758b7c958fafece190b082163f236a68188013902d6a441dbc9dee"
            ),
            "RELEASE_EVALUATION.json": (
                "089da0b4f0f833221069b6553f336279d6cd87c0024fba6327aad14d3856ec8b"
            ),
        },
        "invariant_fusion_v3_sealed_seed20260734": {
            "RELEASE_INTENT.json": (
                "0c0a42ea02daeef99b116981f62483e7112ad748c839fd6921fd2432ee309b20"
            ),
            "RELEASE_MANIFEST.json": (
                "a9e00543bb257659bb25aeebc738fc18a4096ff892f7c3b6c5836d4aa315bf06"
            ),
            "RELEASE_EVALUATION.json": (
                "701f296e55853c892bd169e75edec8ba7c4c4f9e36735d7d6aea1f7b8ca94bc2"
            ),
        },
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
    # Version 2 of the staged policy: the composite survivor score
    # (fit_v3_openset_staged.STAGED_POLICY_VERSION/KIND, HANDOFF 25 follow-up).
    "staged_policy_kind": (
        "v3_staged_noise_prefilter_then_composite_survivor_lof_geometry"
    ),
    "staged_policy_schema": 2,
    # Staged prefilter provenance (HANDOFF 20, tightened per the sealed
    # 20260731 failure): the 0.01 enrollment-budget refit.  Coefficients are
    # bit-identical to the 0.02 set; only the per-length operating points
    # moved.  The fit report predates the seed-20260731 sealed run, so the
    # release seed it recorded as not-spent is 20260731, not this preflight's
    # 20260735; both facts are pinned separately.
    "prefilter_version": "noise-prefilter-v1",
    "prefilter_lengths": (4096, 8192, 16384),
    "prefilter_fitting_seed": 20261001,
    "prefilter_fitting_namespace": (20261000, 20261999),
    "prefilter_known_false_positive_budget": 0.01,
    "prefilter_recorded_release_seed": 20260731,
    # The next staged validation artifact must be a passing validate-role run
    # on the reserved fresh validation novelty seeds, referencing the frozen
    # design seed.  ``staged_policy_version`` is the sibling
    # module's frozen constant and is REQUIRED to match both the module and
    # the artifact; a None pin fails the check loudly rather than skipping.
    "staged_design_novelty_seed": 20260949,
    "staged_validation_novelty_seeds": (20260950, 20260951),
    "staged_policy_version": "v3-staged-openset-policy-v2-composite-survivor",
    # The launcher's pinned v3 expected-protocol fixture for this seed.
    # Pinned after verifying the fixture's evaluation_protocol object is
    # byte-equal to a fresh
    # ``evaluate_v3_release_suite.py --print-expected-protocol 20260735``
    # (composite policy version, staged score axis, q95-on-composite text,
    # strict gates and inactive historical redeclaration) and that the
    # launcher reads exactly this file.
    "v3_protocol_fixture_name": (
        "time-domain-v3-expected-evaluation-protocol-seed20260735.json"
    ),
    "v3_protocol_fixture_sha256": (
        "40ef572beeabd18c38e55fdc5001dbfb0365dcd81dac5ff6ea1b95ceca163f7e"
    ),
    # Bundle schema identity (export_v3_fusion_runtime).
    "bundle_schema": "atomos.v3.time-domain-invariant-fusion.runtime-bundle",
    "bundle_schema_version": 1,
}

# The default dual candidate, frozen.  Overridable on the CLI so tests can
# exercise every check against synthetic trees, never so a different candidate
# can be slipped in silently: the resolved paths are recorded in the report.
DEFAULT_CLASSIFIER_FUSION_DIR = (
    V2
    / "artifacts/invariant_patch/v3_scale/v3_fusion_ml8000reg_seed20260730"
)
DEFAULT_REJECTOR_FUSION_DIR = (
    V2
    / "artifacts/invariant_patch/v3_scale/v3_fusion_multilength_seed20260730"
)
DEFAULT_CLASSIFIER_BUNDLE_DIR = (
    V2
    / "artifacts/invariant_patch/v3_scale/"
    "v3_runtime_bundle_classifier8kreg_dual_seed20260730"
)
DEFAULT_REJECTOR_BUNDLE_DIR = (
    V2
    / "artifacts/invariant_patch/v3_scale/"
    "v3_runtime_bundle_rejector4k_dual_seed20260730"
)
DEFAULT_PREFILTER_ROOT = (
    V2
    / "artifacts/invariant_patch/v3_scale/noise_prefilter_fit20261001_budget001"
)
DEFAULT_STAGED_DIR = (
    V2
    / "artifacts/invariant_patch/v3_scale"
    / "staged_validate_decoupled_rejector4k_classifier8k_budget001_"
    "seeds20260950_20260951"
)
DEFAULT_CANDIDATE_MANIFEST = (
    HERE / "evidence" / "v3_dual_release_candidate.json"
)
# Read-only compatibility aliases for generic helper tests.  The parser and
# preflight harness expose only the explicit role-named arguments.
DEFAULT_FUSION_DIR = DEFAULT_REJECTOR_FUSION_DIR
DEFAULT_BUNDLE_DIR = DEFAULT_REJECTOR_BUNDLE_DIR
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


def check_fusion_artifact(
    fusion_dir: Path,
    *,
    role: str | None = None,
    expected_directory_sha256: str | None = None,
) -> list[Check]:
    prefix = "fusion" if role is None else f"{role}.fusion"
    checks: list[Check] = []
    metrics = _read_json(Path(fusion_dir) / "dev_metrics.json")
    artifacts = metrics.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return [(f"{prefix}.artifacts_record", False, "no artifacts record")]
    bad: list[str] = []
    for file_key, sha_key in FUSION_ASSET_KEYS:
        filename = str(artifacts[file_key])
        expected = str(artifacts[sha_key])
        actual = _sha256(Path(fusion_dir) / filename)
        if actual != expected:
            bad.append(f"{filename}: expected {expected}, observed {actual}")
    checks.append(
        (
            f"{prefix}.asset_sha256",
            not bad,
            "; ".join(bad) if bad else f"{len(FUSION_ASSET_KEYS)} assets match",
        )
    )
    checks.append(
        _match(
            f"{prefix}.consumed_test_rows_used",
            metrics.get("consumed_test_rows_used"),
            0,
        )
    )
    checks.append(
        _match(
            f"{prefix}.sealed_release_data_used",
            metrics.get("sealed_release_data_used"),
            0,
        )
    )
    if expected_directory_sha256 is not None:
        try:
            import fit_v3_openset as openset_base

            observed = openset_base.load_fusion_artifact(
                Path(fusion_dir)
            ).directory_sha256
            checks.append(
                _match(
                    f"{prefix}.directory_sha256",
                    observed,
                    expected_directory_sha256,
                )
            )
        except Exception as error:  # noqa: BLE001
            checks.append(
                (
                    f"{prefix}.directory_sha256",
                    False,
                    f"{type(error).__name__}: {error}",
                )
            )
    return checks


def check_runtime_bundle(
    bundle_dir: Path,
    fusion_dir: Path,
    pins: Mapping[str, Any],
    *,
    role: str | None = None,
    expected_runtime_role: str | None = None,
    expected_manifest_sha256: str | None = None,
) -> list[Check]:
    prefix = "bundle" if role is None else f"{role}.bundle"
    checks: list[Check] = []
    bundle_dir = Path(bundle_dir)
    manifest = _read_json(bundle_dir / "bundle_manifest.json")
    checks.append(
        _match(f"{prefix}.schema", manifest.get("schema"), pins["bundle_schema"])
    )
    checks.append(
        _match(
            f"{prefix}.schema_version",
            manifest.get("schema_version"),
            pins["bundle_schema_version"],
        )
    )
    assets = manifest.get("assets")
    if not isinstance(assets, Mapping) or not assets:
        checks.append((f"{prefix}.asset_sha256", False, "no assets record"))
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
                f"{prefix}.asset_sha256",
                not bad,
                "; ".join(bad) if bad else f"{len(assets)} assets match",
            )
        )
    provenance = manifest.get("provenance", {})
    dev_metrics_sha = _sha256(Path(fusion_dir) / "dev_metrics.json")
    checks.append(
        _match(
            f"{prefix}.bound_to_candidate_fusion",
            provenance.get("source_dev_metrics_sha256"),
            dev_metrics_sha,
        )
    )
    frontend_pins = manifest.get("frontend", {}).get("source_sha256", {})
    if not frontend_pins:
        checks.append((f"{prefix}.frontend_source_pins", False, "no frontend pins"))
    else:
        bad = []
        for relative, expected in sorted(frontend_pins.items()):
            actual = _sha256(REPO / relative)
            if actual != expected:
                bad.append(f"{relative}: expected {expected}, observed {actual}")
        checks.append(
            (
                f"{prefix}.frontend_source_pins",
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
            f"{prefix}.rejection_slot_contract",
            contract_ok,
            "state unset; required contract matches the frozen policy"
            if contract_ok
            else f"state={rejection.get('state')!r}, contract={contract!r}",
        )
    )
    if expected_runtime_role is not None:
        checks.append(
            _match(
                f"{prefix}.runtime_role",
                manifest.get("runtime_role"),
                expected_runtime_role,
            )
        )
    if expected_manifest_sha256 is not None:
        checks.append(
            _match(
                f"{prefix}.manifest_sha256",
                _sha256(bundle_dir / "bundle_manifest.json"),
                expected_manifest_sha256,
            )
        )
    return checks


def check_bundle_self_verification(
    bundle_dir: Path, *, role: str | None = None
) -> list[Check]:
    """Reload the bundle from its own files and replay the probe fixture."""
    name = (
        "bundle.self_verification"
        if role is None
        else f"{role}.bundle.self_verification"
    )
    import export_v3_fusion_runtime as exporter

    bundle_dir = Path(bundle_dir)
    manifest = _read_json(bundle_dir / "bundle_manifest.json")
    fixture = _read_json(bundle_dir / exporter.PROBE_NAME)
    expected_names = [str(item["name"]) for item in exporter.PROBE_SPECIFICATIONS]
    fixture_names = [str(case["name"]) for case in fixture.get("cases", ())]
    if fixture_names != expected_names:
        return [
            (
                name,
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
            name,
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
    checks.append(
        _match(
            "prefilter.pinned_set_sha256",
            actual_set,
            pins.get("prefilter_set_sha256"),
        )
    )
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
    observed_each = {
        str(int(name[1:])): npf.bundle_sha256(bundles_dir / name)
        for name in sorted(recorded_each)
        if str(name).startswith("N")
    }
    checks.append(
        _match(
            "prefilter.pinned_per_length_sha256",
            observed_each,
            pins.get("prefilter_bundle_sha256"),
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
        # The fit predates the seed-20260731 sealed run, so the report
        # records THAT seed as the not-yet-spent release seed; what matters
        # for this preflight is that the record matches the pinned history
        # and that the fitting seed itself is still admissible.
        and seeds.get("release_seed_not_spent")
        == pins["prefilter_recorded_release_seed"]
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
    budget = float(pins["prefilter_known_false_positive_budget"])
    bad = []
    per_length = report.get("per_length", {})
    if sorted(int(key) for key in per_length) != sorted(
        int(value) for value in pins["prefilter_lengths"]
    ):
        bad.append(f"per_length keys {sorted(per_length)}")
    for key in sorted(per_length):
        recorded = per_length[key].get("in_sample_fitting_noise", {}).get(
            "known_false_positive_budget"
        )
        if recorded is None or float(recorded) != budget:
            bad.append(f"N{key}: budget {recorded!r}")
    checks.append(
        (
            "prefilter.tightened_budget",
            not bad,
            f"every per-length operating point declares the {budget} "
            "enrollment known-false-positive budget"
            if not bad
            else "; ".join(bad),
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


def check_staged_artifact(
    staged_dir: Path,
    prefilter_root: Path,
    pins: Mapping[str, Any],
    *,
    rejector_fusion_dir: Path | None = None,
) -> list[Check]:
    """The composite staged validation artifact the sealed run will load.

    Everything here is a property that, if wrong, would make the one-shot run
    either refuse to start or -- worse -- validate a different candidate than
    the one the dev evidence supports: the policy version, the passing
    validate-role status on the declared FRESH novelty seeds, the binding to
    the tightened prefilter set, the stage-2/LOF/composite npz hashes against
    disk, and the composite policy's own loadability (which re-verifies the
    q95 threshold from the stored calibration and refuses any other policy
    version).
    """
    import fit_v3_openset_staged as staged
    import noise_prefilter as npf

    checks: list[Check] = []
    staged_dir = Path(staged_dir)
    report = _read_json(staged_dir / "openset_metrics.json")

    expected_version = pins["staged_policy_version"]
    if not isinstance(expected_version, str) or not expected_version:
        return [
            (
                "staged.policy_version",
                False,
                "pins.staged_policy_version is not pinned; refusing to treat "
                "an unpinned staged artifact as GO",
            )
        ]
    recorded_version = report.get("architecture", {}).get(
        "staged_policy_version"
    )
    version_ok = (
        recorded_version == expected_version
        and staged.STAGED_POLICY_VERSION == expected_version
        and report.get("composite", {}).get("policy_version")
        == expected_version
        and report.get("architecture", {}).get("kind")
        == pins["staged_policy_kind"]
        and int(report.get("architecture", {}).get("schema", -1))
        == int(pins["staged_policy_schema"])
    )
    checks.append(
        (
            "staged.policy_version",
            version_ok,
            f"artifact validates {expected_version} "
            f"(kind {pins['staged_policy_kind']}, schema "
            f"{pins['staged_policy_schema']}); module constant agrees"
            if version_ok
            else (
                f"artifact records {recorded_version!r} / "
                f"{report.get('architecture', {}).get('kind')!r}, module "
                f"{staged.STAGED_POLICY_VERSION!r}, pin {expected_version!r}"
            ),
        )
    )

    status_ok = (
        report.get("status") == "development_openset_pass"
        and report.get("role") == "validate"
        and report.get("gates_are_evidence") is True
        and report.get("release_evidence") is False
        and int(report.get("sealed_release_data_used", -1)) == 0
        and int(report.get("consumed_test_rows_used", -1)) == 0
        and report.get("all_pass") is True
    )
    checks.append(
        (
            "staged.validation_status",
            status_ok,
            "development_openset_pass, role validate, gates are evidence, "
            "zero sealed/consumed rows"
            if status_ok
            else (
                f"status={report.get('status')!r}, role={report.get('role')!r}, "
                f"gates_are_evidence={report.get('gates_are_evidence')!r}"
            ),
        )
    )

    seeds = report.get("seeds", {})
    seeds_ok = (
        sorted(int(seed) for seed in seeds.get("novelty_seeds", ()))
        == sorted(int(seed) for seed in pins["staged_validation_novelty_seeds"])
        and int(seeds.get("design_novelty_seed", -1))
        == int(pins["staged_design_novelty_seed"])
    )
    checks.append(
        (
            "staged.evidence_seeds",
            seeds_ok,
            f"validated on fresh seeds "
            f"{sorted(pins['staged_validation_novelty_seeds'])}, designed on "
            f"{pins['staged_design_novelty_seed']}"
            if seeds_ok
            else f"seeds record {seeds!r}",
        )
    )

    recorded_prefilter = report.get("stage_one", {})
    bundles_dir = Path(prefilter_root) / "bundles"
    prefilter_ok = False
    prefilter_detail = ""
    try:
        actual_set = npf.prefilter_set_sha256(bundles_dir)
        recorded_dir = recorded_prefilter.get("directory")
        prefilter_ok = (
            recorded_dir is not None
            and Path(str(recorded_dir)).resolve() == bundles_dir.resolve()
            and recorded_prefilter.get("set_sha256") == actual_set
        )
        prefilter_detail = (
            f"staged artifact bound to {bundles_dir} (set "
            f"{str(recorded_prefilter.get('set_sha256'))[:16]}...)"
            if prefilter_ok
            else (
                f"artifact records dir={recorded_dir!r} set="
                f"{recorded_prefilter.get('set_sha256')!r}; disk set "
                f"{actual_set}"
            )
        )
    except Exception as error:  # noqa: BLE001 - a refused set is a finding
        prefilter_detail = f"{type(error).__name__}: {error}"
    checks.append(
        ("staged.prefilter_binding", prefilter_ok, prefilter_detail)
    )

    recorded_artifacts = report.get("artifacts", {})
    expected_names = {
        "v3_open_policy_stage_two.npz",
        "v3_branch_lof_components.npz",
        staged.COMPOSITE_POLICY_FILENAME,
    }
    bad = []
    if set(recorded_artifacts) != expected_names:
        bad.append(f"recorded artifacts {sorted(recorded_artifacts)}")
    for name in sorted(recorded_artifacts):
        path = staged_dir / name
        if not path.is_file():
            bad.append(f"{name} missing")
            continue
        actual = _sha256(path)
        if actual != recorded_artifacts[name]:
            bad.append(
                f"{name}: expected {recorded_artifacts[name]}, observed "
                f"{actual}"
            )
    checks.append(
        (
            "staged.npz_sha256",
            not bad,
            f"{len(expected_names)} npz artifacts match the report"
            if not bad
            else "; ".join(bad),
        )
    )

    composite_ok = False
    composite_detail = ""
    try:
        composite = staged.load_composite_policy(
            staged_dir / staged.COMPOSITE_POLICY_FILENAME
        )
        recorded_threshold = report.get("composite", {}).get("threshold")
        composite_ok = (
            recorded_threshold is not None
            and float(recorded_threshold) == float(composite.threshold)
        )
        composite_detail = (
            f"composite policy loads (threshold {composite.threshold:.6f} "
            "re-verified as the frozen quantile of the stored calibration)"
            if composite_ok
            else (
                f"report threshold {recorded_threshold!r} vs npz "
                f"{composite.threshold!r}"
            )
        )
    except Exception as error:  # noqa: BLE001 - a refused policy is a finding
        composite_detail = f"{type(error).__name__}: {error}"
    checks.append(
        ("staged.composite_policy_loads", composite_ok, composite_detail)
    )
    checks.append(
        _match(
            "staged.report_sha256",
            _sha256(staged_dir / "openset_metrics.json"),
            pins.get("staged_validation_report_sha256"),
        )
    )
    checks.append(
        _match(
            "staged.pinned_npz_sha256",
            {
                name: _sha256(staged_dir / name)
                for name in sorted(expected_names)
                if (staged_dir / name).is_file()
            },
            pins.get("staged_artifact_sha256"),
        )
    )
    if rejector_fusion_dir is not None:
        try:
            import fit_v3_openset as openset_base

            artifact_sha = openset_base.load_fusion_artifact(
                Path(rejector_fusion_dir)
            ).directory_sha256
            recorded_sha = report.get("fusion", {}).get("directory_sha256")
            checks.append(
                (
                    "staged.rejector_fusion_binding",
                    recorded_sha == artifact_sha
                    == pins.get("rejector_fusion_directory_sha256"),
                    f"report={recorded_sha}, artifact={artifact_sha}, "
                    f"pin={pins.get('rejector_fusion_directory_sha256')}",
                )
            )
        except Exception as error:  # noqa: BLE001
            checks.append(
                (
                    "staged.rejector_fusion_binding",
                    False,
                    f"{type(error).__name__}: {error}",
                )
            )
    return checks


def check_dual_candidate(
    candidate_manifest: Path,
    classifier_bundle_dir: Path,
    rejector_bundle_dir: Path,
    classifier_fusion_dir: Path,
    rejector_fusion_dir: Path,
    staged_dir: Path,
    prefilter_root: Path,
    pins: Mapping[str, Any],
) -> list[Check]:
    """Load the exact evaluator candidate without reading a release corpus."""
    import evaluate_v3_release_suite as evaluator
    import torch

    checks: list[Check] = []
    path = Path(candidate_manifest).expanduser().resolve()
    manifest = _read_json(path)
    checks.extend(
        [
            _match(
                "candidate.schema",
                manifest.get("schema"),
                pins.get("candidate_manifest_schema"),
            ),
            _match(
                "candidate.status",
                manifest.get("status"),
                "release_candidate_frozen",
            ),
            _match(
                "candidate.id",
                manifest.get("candidate_id"),
                pins.get("candidate_id"),
            ),
        ]
    )
    checks.append(
        (
            "candidate.package_binding_schemas",
            evaluator.STAGING_PACKAGE_SCHEMA
            == pins.get("staging_package_schema")
            and evaluator.STAGING_PACKAGE_SCHEMA_VERSION
            == pins.get("staging_package_schema_version")
            and evaluator.DUAL_BINDING_SCHEMA
            == pins.get("dual_binding_schema")
            and evaluator.DUAL_BINDING_SCHEMA_VERSION
            == pins.get("dual_binding_schema_version")
            and evaluator.STAGING_STATUS == pins.get("staging_status"),
            "dual runtime package and dual binding are schema v1 with "
            "status staging_not_release; legacy single-fusion packages are "
            "inadmissible",
        )
    )
    observed_sha = _sha256(path)
    pinned_sha = pins.get("candidate_manifest_sha256")
    checks.append(
        (
            "candidate.manifest_sha256",
            isinstance(pinned_sha, str)
            and len(pinned_sha) == 64
            and observed_sha == pinned_sha,
            f"observed {observed_sha}; pinned {pinned_sha!r}"
            + (
                ""
                if isinstance(pinned_sha, str) and len(pinned_sha) == 64
                else " (an unpinned final manifest is a hard NO-GO)"
            ),
        )
    )
    try:
        candidate = evaluator.load_candidate(
            candidate_manifest_path=path,
            classifier_bundle_dir=Path(classifier_bundle_dir),
            rejector_bundle_dir=Path(rejector_bundle_dir),
            classifier_fusion_dir=Path(classifier_fusion_dir),
            rejector_fusion_dir=Path(rejector_fusion_dir),
            staged_dir=Path(staged_dir),
            prefilter_dir=Path(prefilter_root) / "bundles",
            device=torch.device("cpu"),
        )
        exact = (
            candidate.candidate_manifest_sha256 == observed_sha
            and candidate.validation_evidence_sha256
            == pins.get("validation_evidence_sha256")
            and candidate.frozen_contract_sha256
            == pins.get("frozen_candidate_contract_sha256")
            and candidate.classifier_fusion_artifact.directory_sha256
            == pins.get("classifier_fusion_directory_sha256")
            and candidate.rejector_fusion_artifact.directory_sha256
            == pins.get("rejector_fusion_directory_sha256")
            and candidate.classifier_bundle_manifest_sha256
            == pins.get("classifier_bundle_manifest_sha256")
            and candidate.rejector_bundle_manifest_sha256
            == pins.get("rejector_bundle_manifest_sha256")
            and candidate.staged_metrics_sha256
            == pins.get("staged_validation_report_sha256")
            and dict(candidate.staged_hashes)
            == dict(pins.get("staged_artifact_sha256", {}))
            and candidate.prefilter_set_sha256
            == pins.get("prefilter_set_sha256")
        )
        checks.append(
            (
                "candidate.complete_binding_chain",
                exact,
                "final manifest, validation evidence, pre-validation "
                "contract, both role bundles/fusions, staged policy, "
                "prefilter, browser assets, staging package and dual binding "
                "all load under evaluator schema 3"
                if exact
                else "candidate loaded but one independent preflight pin differs",
            )
        )
    except Exception as error:  # noqa: BLE001 - refusal is the finding
        checks.append(
            (
                "candidate.complete_binding_chain",
                False,
                f"{type(error).__name__}: {error}",
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
    # One current ledger must describe the complete evidence lifecycle before
    # either the next development draw or the sealed draw is allowed.
    consumed_now = sorted(int(s) for s in pins["consumed_release_seeds"])
    new_seed = int(pins["release_seed"])
    ledger_ok = (
        int(staged.RELEASE_SEED_NEVER_SPENT_HERE) == new_seed
        and sorted(staged.CONSUMED_SEALED_RELEASE_SEEDS)
        == [20260729, 20260731, 20260733, 20260734]
        and int(openset_base.RELEASE_SEED_NEVER_SPENT_HERE)
        in staged.CONSUMED_SEALED_RELEASE_SEEDS
        and consumed_now == [20260729, 20260731, 20260733, 20260734]
        and new_seed not in consumed_now
        and new_seed not in staged.SPENT_NOVELTY_SEEDS
        and new_seed not in staged.CONSUMED_SEALED_RELEASE_SEEDS
        and sorted(staged.SPENT_NOVELTY_SEEDS)
        == list(range(20260938, 20260952))
        and int(staged.PROPOSED_DESIGN_NOVELTY_SEED)
        == int(pins["staged_design_novelty_seed"])
        and tuple(staged.DEFAULT_VALIDATION_NOVELTY_SEEDS)
        == tuple(pins["staged_validation_novelty_seeds"])
        and int(staged.PROPOSED_DESIGN_NOVELTY_SEED)
        in staged.SPENT_NOVELTY_SEEDS
        and set(staged.DEFAULT_VALIDATION_NOVELTY_SEEDS).issubset(
            staged.SPENT_NOVELTY_SEEDS
        )
    )
    checks.append(
        (
            "policy.seed_ledger_constants",
            ledger_ok,
            f"release seed {new_seed} reserved and unreachable from "
            f"development; consumed releases {consumed_now}; spent novelty "
            "seeds 20260938-20260951; spent design and consumed validation "
            f"{pins['staged_design_novelty_seed']}/"
            f"{list(pins['staged_validation_novelty_seeds'])}"
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
        # Pin the exact strict evaluator bytes before a sealed suite exists.
        checks.append(
            _match(
                "sources.v3_evaluator_sha256",
                sha,
                pins["v3_evaluator_sha256"],
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
    for root_name, expected_hashes in sorted(
        pins["consumed_sealed_roots"].items()
    ):
        sealed_root = releases_dir / str(root_name)
        bad = []
        for name, expected in sorted(expected_hashes.items()):
            path = sealed_root / name
            if not path.is_file():
                bad.append(f"{name} missing")
                continue
            actual = _sha256(path)
            if actual != expected:
                bad.append(f"{name}: expected {expected}, observed {actual}")
        checks.append(
            (
                f"ledger.consumed_root_intact.{root_name}",
                not bad,
                "consumed sealed root JSONs match the pinned negative "
                "evidence"
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

    # The launcher's pinned v3 expected-protocol fixture for this seed: it
    # must exist, hash to the pin, predeclare exactly this release seed, and
    # be the file the launcher actually reads.  The sealed evaluator's own
    # intent-equality check remains the final backstop; this catches a stale
    # or missing fixture BEFORE the one-shot run is attempted.
    fixture_name = str(pins["v3_protocol_fixture_name"])
    fixture_path = REPO / "tools" / fixture_name
    expected_fixture_sha = pins["v3_protocol_fixture_sha256"]
    if not fixture_path.is_file():
        checks.append(
            (
                "generation.v3_protocol_fixture",
                False,
                f"{fixture_path} is missing; regenerate it with "
                "evaluate_v3_release_suite.py --print-expected-protocol "
                f"{pins['release_seed']}",
            )
        )
    elif not isinstance(expected_fixture_sha, str):
        checks.append(
            (
                "generation.v3_protocol_fixture",
                False,
                "pins.v3_protocol_fixture_sha256 is not pinned; refusing to "
                "treat an unpinned fixture as GO",
            )
        )
    else:
        wrapper = _read_json(fixture_path)
        protocol = wrapper.get("evaluation_protocol", {})
        fixture_sha = _sha256(fixture_path)
        fixture_ok = (
            fixture_sha == expected_fixture_sha
            and int(wrapper.get("release_seed", -1)) == int(pins["release_seed"])
            and int(protocol.get("novelty", {}).get("seed", -1))
            == int(pins["release_seed"])
            and fixture_name in text
        )
        checks.append(
            (
                "generation.v3_protocol_fixture",
                fixture_ok,
                f"{fixture_name} predeclares seed {pins['release_seed']} and "
                f"hashes to the pin; launcher references it"
                if fixture_ok
                else (
                    f"sha {fixture_sha} vs pin {expected_fixture_sha}; "
                    f"wrapper seed {wrapper.get('release_seed')!r}, novelty "
                    f"seed {protocol.get('novelty', {}).get('seed')!r}, "
                    f"referenced={fixture_name in text}"
                ),
            )
        )
        import evaluate_v3_release_suite as evaluator

        fresh_protocol = evaluator.expected_evaluation_protocol(
            int(pins["release_seed"]),
            tuple(
                sorted(
                    int(length)
                    for length in pins["prefilter_bundle_sha256"]
                )
            ),
        )
        checks.append(
            (
                "generation.v3_protocol_fixture_matches_evaluator",
                protocol == fresh_protocol,
                "fixture evaluation_protocol exactly matches the final "
                "evaluator"
                if protocol == fresh_protocol
                else "fixture evaluation_protocol is stale relative to the "
                "final evaluator",
            )
        )
        checks.extend(_check_fixture_gate_contract(protocol, pins))
    return checks


def _check_fixture_gate_contract(
    protocol: Mapping[str, Any],
    pins: Mapping[str, Any],
) -> list[Check]:
    """Require strict current gates plus inactive seed-20260734 history."""
    import evaluate_invariant_release_suite as release

    expected_history = pins["historical_gate_redeclaration"]
    historical = protocol.get("historical_gate_redeclaration")
    problems: list[str] = []
    if not isinstance(historical, Mapping):
        problems.append(
            "fixture protocol carries no historical_gate_redeclaration block"
        )
        historical = {}
    if historical.get("active_for_current_protocol") is not False:
        problems.append("historical redeclaration is not explicitly inactive")
    if int(historical.get("release_seed", -1)) != int(
        expected_history["release_seed"]
    ):
        problems.append("historical redeclaration names the wrong release seed")
    if (
        historical.get("current_protocol_gate_source")
        != "imported_v2_gate_floors_unchanged"
    ):
        problems.append("current gate source is not the unchanged v2 floors")
    expected = expected_history["gates"]
    redeclared = historical.get("gates_redeclared", {})
    if not isinstance(redeclared, Mapping):
        problems.append("historical gates_redeclared is not an object")
        redeclared = {}
    if sorted(redeclared) != sorted(expected):
        problems.append(
            f"historical gate names {sorted(redeclared)} != "
            f"{sorted(expected)}"
        )
    for gate_name in sorted(expected):
        pin = expected[gate_name]
        record = redeclared.get(gate_name, {})
        if not isinstance(record, Mapping):
            problems.append(f"{gate_name}: record is not an object")
            continue
        if float(record.get("v2_level", -1)) != float(pin["v2_level"]):
            problems.append(
                f"{gate_name}: v2_level {record.get('v2_level')!r} != "
                f"{pin['v2_level']}"
            )
        if float(record.get("v3_level", -1)) != float(pin["v3_level"]):
            problems.append(
                f"{gate_name}: v3_level {record.get('v3_level')!r} != "
                f"{pin['v3_level']}"
            )
        rationale = record.get("owner_decision")
        if (
            not isinstance(rationale, str)
            or "re-declared for the v3 architecture by the owner on "
            "2026-07-28" not in rationale
            or "BEFORE release seed 20260734 was generated"
            not in rationale
        ):
            problems.append(f"{gate_name}: owner_decision rationale missing")
    gates = protocol.get("gates", {})
    for floor_key, v2_level in sorted(release.GATE_FLOORS.items()):
        want = float(v2_level)
        got = gates.get(floor_key)
        if got is None or float(got) != want:
            problems.append(
                f"gates[{floor_key!r}] is {got!r}, expected {want}"
            )
    if sorted(gates) != sorted(release.GATE_FLOORS):
        problems.append(
            "fixture gates keys differ from the imported v2 floors"
        )
    ok = not problems
    return [
        (
            "generation.v3_fixture_gate_contract",
            ok,
            "fixture uses all 17 strict imported v2 gate floors; the "
            "seed-20260734 0.12/0.84 redeclaration is retained only as "
            "explicitly inactive historical metadata"
            if ok
            else "; ".join(problems),
        )
    ]


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
    candidate_manifest: Path,
    isolated_root: Path,
    node_bin_dir: Path,
) -> dict[str, Any]:
    """The exact one-shot launcher invocation for seed 20260735.

    Reconstructed from the consumed release mechanics and re-aimed at the new
    untouched seed.
    The candidate file is the final dual release-candidate manifest: it binds
    both role bundles/fusions, validation evidence, staged/prefilter state,
    browser assets, staging package and the deploy-time dual binding.
    This function only formats strings; nothing is executed.
    """
    manifest_path = Path(candidate_manifest).expanduser().resolve()
    candidate_sha = _sha256(manifest_path)
    pinned_sha = pins.get("candidate_manifest_sha256")
    if not isinstance(pinned_sha, str) or candidate_sha != pinned_sha:
        raise ValueError(
            "final candidate manifest is unpinned or differs from "
            "pins.candidate_manifest_sha256"
        )
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
        "RELEASE_EVALUATION_PROTOCOL": "v3",
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
            "RELEASE_EVALUATION_PROTOCOL=v3 embeds the staged protocol from "
            f"the pinned fixture {pins['v3_protocol_fixture_name']}, which "
            "must predeclare exactly this release seed; the sealed evaluator "
            "refuses any suite whose intent protocol differs from its own "
            "expected object.",
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

    candidate_manifest = Path(args.candidate_manifest)
    classifier_fusion_dir = Path(args.classifier_fusion_dir)
    rejector_fusion_dir = Path(args.rejector_fusion_dir)
    classifier_bundle_dir = Path(args.classifier_bundle_dir)
    rejector_bundle_dir = Path(args.rejector_bundle_dir)
    prefilter_root = Path(args.prefilter_root)
    staged_dir = Path(args.staged_dir)
    releases_dir = Path(args.releases_dir)
    isolated_root = Path(args.isolated_root)
    evaluator_path = Path(args.evaluator)
    node_bin_dir = Path(args.node_bin_dir)

    run_group(
        "classifier.fusion",
        lambda: check_fusion_artifact(
            classifier_fusion_dir,
            role="classifier",
            expected_directory_sha256=pins[
                "classifier_fusion_directory_sha256"
            ],
        ),
    )
    run_group(
        "rejector.fusion",
        lambda: check_fusion_artifact(
            rejector_fusion_dir,
            role="rejector",
            expected_directory_sha256=pins[
                "rejector_fusion_directory_sha256"
            ],
        ),
    )
    run_group(
        "classifier.bundle",
        lambda: check_runtime_bundle(
            classifier_bundle_dir,
            classifier_fusion_dir,
            pins,
            role="classifier",
            expected_runtime_role="accepted_known_classifier",
            expected_manifest_sha256=pins["classifier_bundle_manifest_sha256"],
        ),
    )
    run_group(
        "rejector.bundle",
        lambda: check_runtime_bundle(
            rejector_bundle_dir,
            rejector_fusion_dir,
            pins,
            role="rejector",
            expected_runtime_role="known_unknown_rejector",
            expected_manifest_sha256=pins["rejector_bundle_manifest_sha256"],
        ),
    )
    run_group(
        "classifier.bundle.self_verification",
        lambda: check_bundle_self_verification(
            classifier_bundle_dir, role="classifier"
        ),
    )
    run_group(
        "rejector.bundle.self_verification",
        lambda: check_bundle_self_verification(
            rejector_bundle_dir, role="rejector"
        ),
    )
    run_group("prefilter", lambda: check_prefilter(prefilter_root, pins))
    run_group(
        "staged",
        lambda: check_staged_artifact(
            staged_dir,
            prefilter_root,
            pins,
            rejector_fusion_dir=rejector_fusion_dir,
        ),
    )
    run_group(
        "candidate",
        lambda: check_dual_candidate(
            candidate_manifest,
            classifier_bundle_dir,
            rejector_bundle_dir,
            classifier_fusion_dir,
            rejector_fusion_dir,
            staged_dir,
            prefilter_root,
            pins,
        ),
    )
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
            pins, candidate_manifest, isolated_root, node_bin_dir
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
            "candidate_manifest": str(candidate_manifest),
            "classifier_fusion_dir": str(classifier_fusion_dir),
            "rejector_fusion_dir": str(rejector_fusion_dir),
            "classifier_bundle_dir": str(classifier_bundle_dir),
            "rejector_bundle_dir": str(rejector_bundle_dir),
            "prefilter_root": str(prefilter_root),
            "staged_dir": str(staged_dir),
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
    parser.add_argument(
        "--candidate-manifest", default=str(DEFAULT_CANDIDATE_MANIFEST)
    )
    parser.add_argument(
        "--classifier-fusion-dir", default=str(DEFAULT_CLASSIFIER_FUSION_DIR)
    )
    parser.add_argument(
        "--rejector-fusion-dir", default=str(DEFAULT_REJECTOR_FUSION_DIR)
    )
    parser.add_argument(
        "--classifier-bundle-dir", default=str(DEFAULT_CLASSIFIER_BUNDLE_DIR)
    )
    parser.add_argument(
        "--rejector-bundle-dir", default=str(DEFAULT_REJECTOR_BUNDLE_DIR)
    )
    parser.add_argument("--prefilter-root", default=str(DEFAULT_PREFILTER_ROOT))
    parser.add_argument("--staged-dir", default=str(DEFAULT_STAGED_DIR))
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
