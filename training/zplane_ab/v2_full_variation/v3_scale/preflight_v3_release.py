"""Preflight for the next one-shot v3.4/q97 sealed run on seed 20260736.

This script runs WITHOUT any release suite and consumes nothing: no corpus is
read, no seed is drawn, no novelty realization is generated, and nothing is
written under ``training/artifacts/releases``.  It exists so the orchestrator
can prove, immediately before spending the unspent release seed, that every
frozen input of the one-shot run is exactly what the evidence trail says it is.

Release seeds 20260729, 20260731, 20260733 and 20260734 are consumed and
immutable negative evidence, as is seed 20260735's 22/23 v3.3 failure.  The
seed-20260734 protocol used predeclared 0.12 known-FUR and 0.84 five-shot
levels; this fact remains visible as explicitly inactive historical metadata,
alongside a machine-readable correction that the old cross-seed comparison
was unpaired and that v2 failed eight gates, not six.  Seed 20260736 uses the
original strict imported gates: known FUR <= 0.10 and five-shot >= 0.85.

* the stage-1 prefilter set is the TIGHTENED 0.01 enrollment-budget refit
  (``noise_prefilter_fit20261001_budget001``): coefficients bit-identical to
  the 0.02 set, threshold only;
* the frozen q99 design failure on seed 20260952 and passing q97 design on
  seed 20260955 are independently hash-pinned;
* the staged artifact validates policy-v4's COMPOSITE survivor score with its
  enrollment-only q97 threshold on the single consumed validation draw at
  novelty seeds 20260953/20260954;
* the launcher's pinned evaluator-v4 fixture must predeclare seed 20260736,
  carry the strict imported gates and corrected historical audit.

The validation evidence, candidate contract, browser/package assets, final
evaluator and seed-20260736 fixture are now independently frozen and pinned.
The same checks retain explicit hard-refusal behavior for any missing pin;
the preflight never substitutes old v3.3 bytes or fabricates a digest.

What it verifies, each as an itemised PASS/FAIL check:

* both fusion artifacts independently: 8k classifier and 4k rejector tree
  digests plus every recorded asset SHA-256;
* both role-labelled runtime bundles: schema identity, explicit runtime role,
  every asset hash, exact fusion binding, frontend source hashes, and a full
  replay of each bundle's probe-fixture self-verification;
* the final v3.4/q97 dual release-candidate manifest: its independent byte pin and
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
* the staged validation artifact: policy-v4/q97, passing validate-role
  status on the declared fresh novelty seeds, its recorded prefilter binding,
  and its stage-2 npz hashes against disk;
* the frozen open-set policy constants in ``v3_time_domain_openset`` and the
  import-identity discipline of ``fit_v3_openset`` / ``fit_v3_openset_staged``
  (gates imported, never re-typed);
* evaluator sources: evaluator schema 4 and the q97 evaluation identity, the
  v2 evaluator's recorded hash, the final v4 evaluator byte pin, and every
  dependency source hash;
* the on-disk seed ledger: no release root exists for seed 20260736 and all
  five consumed sealed roots are intact, preserved as negative evidence;
* the isolated generation source root from HANDOFF section 6: SignalLab and
  Atom-DSP source-tree digests (same walk as the launcher: skip
  ``node_modules``, ``dist`` and ``RELEASE_SOURCE_PROVENANCE.json``, refuse
  symlinks), package-lock hashes, the built ``dist/index.js`` /
  ``dist/index.d.ts`` hashes, and the ``@atomos/dsp`` symlink resolution;
* the launcher / corpus generator / prefix deriver hashes, the Node, npm,
  npx and tsx pins, and the launcher's versioned evaluator-v4 protocol fixture
  for seed 20260736 (present, hash-pinned, strict, historically corrected, and
  actually referenced by the launcher source).

At the end it prints every check itemised and then a single GO / NO-GO line.
Exit status 0 means GO, 1 means NO-GO, 2 means the preflight itself was
invoked unusably (for example an output path that already exists).

It also RECONSTRUCTS, and never runs, the exact generation command the
orchestrator should use for seed 20260736; the command is printed and stored
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
import platform
from pathlib import Path
import py_compile
import stat
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


REPORT_SCHEMA = "v3-release-preflight-v3-q97-dual-fusion"

# ---------------------------------------------------------------------------
# frozen expectations
#
# Every hash below is a pin from the evidence trail, not a value computed here:
# HANDOFF sections 2 (seed rules), 6 (generation protocol and dependency
# digests), 7 (consumed sealed v2 root) and the frozen policy modules.  A pin
# that no longer matches disk is a NO-GO, never something to update casually.
# ---------------------------------------------------------------------------

DEFAULT_PINS: dict[str, Any] = {
    "release_seed": 20260736,
    "sealed_seed": 20260729,
    # Every release seed already consumed by a sealed run.  The launcher and
    # v3 evaluator must both refuse every entry.
    "consumed_release_seeds": (
        20260729,
        20260731,
        20260733,
        20260734,
        20260735,
    ),
    "consumed_release_seed_reasons": {
        20260729: "consumed sealed v2 release suite",
        20260731: (
            "consumed sealed v3.0 release suite (HANDOFF 25: 22/23 gates, "
            "known false-unknown failure frozen)"
        ),
        20260733: (
            "consumed sealed v3.2 release suite (21/23 gates; known "
            "false-unknown and five-shot failures frozen)"
        ),
        20260734: (
            "consumed sealed v3.2 release suite under its predeclared "
            "historical gate redeclaration (22/23 gates; five-shot failure "
            "frozen)"
        ),
        20260735: (
            "consumed sealed v3.3 decoupled 8k-classifier/4k-rejector release "
            "suite (22/23 gates; known false-unknown failure frozen)"
        ),
    },
    # 20260732 is deliberately skipped: it is already a development MODEL
    # seed (HANDOFF 25 records the namespace hazard explicitly).
    "skipped_model_seed": 20260732,
    "target_per_class": 192,
    # HANDOFF 6: evaluator SHA recorded in the sealed v2 result.
    "v2_evaluator_sha256": (
        "1b8137b4c222a857a91f340730137fefd3fe17a026d9ba5eb172e7fd774c0541"
    ),
    "evaluator_schema": 4,
    "evaluation_version": (
        "time-domain-v3-release-evaluation-v4-q97-dual-fusion"
    ),
    "v3_evaluator_sha256": (
        "ac728f7f1059260814e74e9825d2ab937ed5cb29136fd4c8e2bf388d4ff454e3"
    ),
    "candidate_manifest_schema": "time-domain-v3-dual-release-candidate-v2",
    "candidate_evidence_schema": (
        "time-domain-v3-decoupled-validation-evidence-v2"
    ),
    "candidate_contract_schema": "time-domain-v3-q97-candidate-v1",
    "candidate_manifest_sha256": (
        "7f824eb734466cb697ee28387a470e8e669e19567542d928130a2b4ad9f59053"
    ),
    "candidate_id": "v3.4-q97-decoupled-8k-classifier-4k-rejector",
    "staging_package_schema": (
        "atomos.v3.time-domain-classifier.dual-runtime-package"
    ),
    "staging_package_schema_version": 1,
    "dual_binding_schema": "atomos.v3.time-domain-dual-fusion.binding",
    "dual_binding_schema_version": 1,
    "staging_status": "staging_not_release",
    "browser_fusion_schema": (
        "atomos.v3.time-domain-invariant-fusion.browser-weights"
    ),
    "browser_fusion_schema_version": 1,
    "browser_openset_schema": "atomos.v3.time-domain-openset.staged",
    "browser_openset_schema_version": 4,
    "parity_schema": "time-domain-openset-parity-v1",
    "parity_schema_version": 4,
    "validation_evidence_sha256": (
        "355f0a201178431a53b9b38bb184650f764e50651e230a768f620fce7778a3c7"
    ),
    "frozen_candidate_contract_sha256": (
        "0be5ebf35dea6e2ab160353f4f42e95445e8bb30ea4742a8c0ea0b9b0a001c7d"
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
        "13b5dc55c150dd24c4b057d1537ee715ec632543284e56ea5c5f105e4afe5ac7"
    ),
    "staged_artifact_sha256": {
        "v3_branch_lof_components.npz": (
            "0a66ca267356bdacf239de5512b9406e010526c990dc38c137cac84342a062a6"
        ),
        "v3_open_policy_stage_two.npz": (
            "9378f6eea893174b4f716f4db5ea6ea2b637d64b04a0e7672bb39b2e50600dd8"
        ),
        "v3_staged_composite_policy.npz": (
            "15554c0fec5f7770c3b0c7a01c5250b61b4ce0b45fed9dfdf7e8785bb328d19e"
        ),
    },
    "browser_asset_sha256": {
        "classifier": (
            "55ab6b0374f1c82491b745c2e6bef7614702c4a3a7e9a68b76e8771f3fba501b"
        ),
        "rejector": (
            "588bb7de6c802a8c44c1ba6c01d6693fb3e39926de918274f16042c15f1863a1"
        ),
        "openset_policy": (
            "4110332985ea13e3d93f83e51c027efc63f4c11380bd6020cfef240c590fffbc"
        ),
    },
    "staging_package_manifest_sha256": (
        "4c27afa382fbdf16da1215d807945710b264d03d876231a2b01af12313f60952"
    ),
    "dual_binding_sha256": (
        "f820d390368266cdf266adbd915559ae3986caad622c67b21352dbbddb7e766f"
    ),
    "prefilter_set_sha256": (
        "4751ae631f879bc2d987d64082c296d7150b7199fab0f1eeb6de84a0f338b5f2"
    ),
    "prefilter_bundle_sha256": {
        "4096": "5ef175f5ddc296ec9ae0ffb8ac6801504fd609d6a852dbd5c4b1bf9a10de0f85",
        "8192": "e0e972be4d1b7e43929b6c45c69d2dada884b49b39e16555998c375edf744557",
        "16384": "3d56108fc016d8fb8ff7f4c18f8fc2fa6a1443d5d32d30cd769d52de087384db",
    },
    # Immutable development evidence.  These pins must never be replaced by a
    # new draw or silently relabelled.
    "q99_design_evidence": {
        "novelty_seed": 20260952,
        "role": "design",
        "status": "design_selection_fail",
        "all_pass": False,
        "gates_are_evidence": False,
        "policy_schema": 3,
        "policy_version": (
            "v3-staged-openset-policy-v3-composite-survivor-q99"
        ),
        "policy_kind": (
            "v3_staged_noise_prefilter_then_q99_composite_survivor_lof_geometry"
        ),
        "threshold_quantile": 0.99,
        "threshold": 0.9945961538794619,
        "source_sha256": (
            "8910a84544955cc61756a0f0bb61a4e9830526dab145eada1048cea80e9b55a8"
        ),
        "report_sha256": (
            "513812bd0255983978a9cc10363df59cbbab278e1c9783068cf566bca8e3ada6"
        ),
        "artifact_sha256": {
            "v3_branch_lof_components.npz": (
                "0a66ca267356bdacf239de5512b9406e010526c990dc38c137cac84342a062a6"
            ),
            "v3_open_policy_stage_two.npz": (
                "9378f6eea893174b4f716f4db5ea6ea2b637d64b04a0e7672bb39b2e50600dd8"
            ),
            "v3_staged_composite_policy.npz": (
                "57344475a456503fec6fec2ea24e752ddf63d5ccd586437f650ca35eca650639"
            ),
        },
        "failed_gate": "chirp_threshold_recall",
        "failed_gate_worst": 0.0033333333333333335,
        "known_false_unknown_worst": 0.019916142557651992,
    },
    "q97_design_evidence": {
        "novelty_seed": 20260955,
        "role": "design",
        "status": "design_selection_pass",
        "all_pass": True,
        "gates_are_evidence": False,
        "policy_schema": 4,
        "policy_version": (
            "v3-staged-openset-policy-v4-composite-survivor-q97"
        ),
        "policy_kind": (
            "v3_staged_noise_prefilter_then_q97_composite_survivor_lof_geometry"
        ),
        "threshold_quantile": 0.97,
        "threshold": 0.9844868317511343,
        "source_sha256": (
            "4cca22059959ec472278f28594bd54a72bbbab04fcf9a294baa81d31fea59be9"
        ),
        "report_sha256": (
            "bb749aadd5395a3fc21ff9453621ad8fa7a7512f9b3c21c29cee933b09adfb0f"
        ),
        "artifact_sha256": {
            "v3_branch_lof_components.npz": (
                "0a66ca267356bdacf239de5512b9406e010526c990dc38c137cac84342a062a6"
            ),
            "v3_open_policy_stage_two.npz": (
                "9378f6eea893174b4f716f4db5ea6ea2b637d64b04a0e7672bb39b2e50600dd8"
            ),
            "v3_staged_composite_policy.npz": (
                "15554c0fec5f7770c3b0c7a01c5250b61b4ce0b45fed9dfdf7e8785bb328d19e"
            ),
        },
        "failed_gate": None,
        "failed_gate_worst": None,
        "known_false_unknown_worst": 0.041928721174004195,
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
    "historical_claim_correction": {
        "schema": "time-domain-v3-historical-claim-correction-v1",
        "applies_to_release_seed": 20260734,
        "original_text_preserved_verbatim": True,
        "original_text_sha256": (
            "cac7c5c4221fffeef59bbe74c2fc1203277e909b7f36511269b3579754a89199"
        ),
        "comparison_design": "cross_seed_unpaired",
        "same_rows_or_same_novelty_draw": False,
        "v2_failed_gate_count": 8,
        "historical_claimed_v2_failed_gate_count": 6,
        "every_identically_measured_axis_claim_supported": False,
        "all_six_gates_claim_supported_as_complete_count": False,
        "corrected_scope": (
            "comparisons to v2 use different sealed release seeds and are "
            "unpaired; the v2 release failed eight gates, not six; therefore "
            "the preserved claim that the candidate beat v2 on every "
            "identically measured axis is unsupported"
        ),
        "affects_current_gate_values_or_pass_fail": False,
    },
    # Frozen generation source hash.  The v2 default remains byte-compatible
    # with its historical protocol; v3 reads only the versioned seed-20260736
    # fixture and refuses all five consumed release seeds before generating.
    "launcher_sha256": (
        "c8fe3ff6239387714a27e1ae79e168535af7e6596ad88692c93b33ff220daf11"
    ),
    "corpus_generator_sha256": (
        "305418a5bc7bd8f9a49799477f3a457b4c07d0c58b637766989fc9557565371b"
    ),
    "prefix_deriver_sha256": (
        "d9383a642d21a59f66f1f9e88fc7ce51ad50893a381985c4f53f2b0da9aec3a3"
    ),
    "python_runtime": {
        "python": "3.9.6",
        "numpy": "2.0.2",
        "torch": "2.8.0",
        "device": "cpu",
        "platform": "darwin-arm64",
    },
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
        "invariant_fusion_v3_sealed_seed20260735": {
            "RELEASE_INTENT.json": (
                "c2cb73fa55783e9348914b28106c1fcb0c54c66ced40c0d3448b6178a6e38f2e"
            ),
            "RELEASE_MANIFEST.json": (
                "a55d4739b7744d58b493a75a441c4d265ec2bb3f69b73b94ca9ee221fd018f03"
            ),
            "RELEASE_EVALUATION.json": (
                "e51aea4c3874effc210d0f010bf757f933977c6009bdad5dcc86f998ab73cfef"
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
    # Version 4 of the staged policy: the q97 composite survivor score.
    "staged_policy_kind": (
        "v3_staged_noise_prefilter_then_q97_composite_survivor_lof_geometry"
    ),
    "staged_policy_schema": 4,
    "staged_policy_threshold_quantile": 0.97,
    # Staged prefilter provenance (HANDOFF 20, tightened per the sealed
    # 20260731 failure): the 0.01 enrollment-budget refit.  Coefficients are
    # bit-identical to the 0.02 set; only the per-length operating points
    # moved.  The fit report predates the seed-20260731 sealed run, so the
    # release seed it recorded as not-spent is 20260731, not this preflight's
    # 20260736; both facts are pinned separately.
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
    "staged_design_novelty_seed": 20260955,
    "staged_validation_novelty_seeds": (20260953, 20260954),
    "staged_policy_version": (
        "v3-staged-openset-policy-v4-composite-survivor-q97"
    ),
    "spent_design_seed_reasons": {
        20260952: (
            "designed the frozen policy-v3/q99 v3.4 candidate: "
            "design_selection_fail; only N4096 chirp threshold recall failed "
            "(1/300 = 0.0033333333333333335 < 0.10), while known "
            "false-unknown was 38/1908 = 0.019916142557651992. The seed is "
            "consumed and may not be redrawn"
        ),
        20260955: (
            "designed the frozen policy-v4/q97 v3.4 candidate: "
            "design_selection_pass; all six unchanged design gates passed. "
            "Known false-unknown was 80/1908 = 0.041928721174004195 and worst "
            "N4096 chirp threshold recall was 57/300 = 0.19. Report sha256 "
            "bb749aadd5395a3fc21ff9453621ad8fa7a7512f9b3c21c29cee933b09adfb0f. "
            "This is development selection, not validation evidence; the "
            "seed is consumed and may not be redrawn"
        ),
    },
    "validation_seed_reasons": {
        20260953: (
            "validated the frozen policy-v4/q97 v3.4 candidate in the one "
            "exact ordered validation draw: development_openset_pass; all six "
            "unchanged validation gates passed across seeds 20260953/20260954 "
            "and prefix lengths [4096, 8192, 16384, 32768]. Known "
            "false-unknown was 80/1908 = 0.041928721174004195, worst noise "
            "AUROC was 0.9576589797344515, and worst chirp threshold recall "
            "was 57/300 = 0.19. Report sha256 "
            "13b5dc55c150dd24c4b057d1537ee715ec632543284e56ea5c5f105e4afe5ac7. "
            "Validation evidence is consumed and may not be reused"
        ),
        20260954: (
            "validated the frozen policy-v4/q97 v3.4 candidate in the one "
            "exact ordered validation draw: development_openset_pass; all six "
            "unchanged validation gates passed across seeds 20260953/20260954 "
            "and prefix lengths [4096, 8192, 16384, 32768]. Known "
            "false-unknown was 80/1908 = 0.041928721174004195, worst noise "
            "AUROC was 0.9576589797344515, and worst chirp threshold recall "
            "was 57/300 = 0.19. Report sha256 "
            "13b5dc55c150dd24c4b057d1537ee715ec632543284e56ea5c5f105e4afe5ac7. "
            "Validation evidence is consumed and may not be reused"
        ),
    },
    # The launcher's pinned v3 expected-protocol fixture for this seed.
    # Pinned after verifying the fixture's evaluation_protocol object is
    # byte-equal to a fresh
    # ``evaluate_v3_release_suite.py --print-expected-protocol 20260736``
    # (policy-v4/q97, staged score axis, strict gates and corrected inactive
    # historical redeclaration) and that the
    # launcher reads exactly this file.
    "v3_protocol_fixture_name": (
        "time-domain-v3-expected-evaluation-protocol-v4-q97-seed20260736.json"
    ),
    "v3_protocol_fixture_sha256": (
        "40d29042f844e2169f1025b75d0a63545f669b25ab7d7f9e46ae630599668fdd"
    ),
    "seed35_preflight_evidence_name": (
        "v3_dual_release_preflight_seed20260735.json"
    ),
    "seed35_preflight_evidence_sha256": (
        "c5b16f13ac1df0d9a8e9e8bc655f452be69f05ef95894d92e4d93901a788c19c"
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
DEFAULT_Q99_DESIGN_DIR = (
    V2
    / "artifacts/invariant_patch/v3_scale/"
    "staged_design_v34_q99_rejector4k_budget001_seed20260952"
)
DEFAULT_Q97_DESIGN_DIR = (
    V2
    / "artifacts/invariant_patch/v3_scale/"
    "staged_design_v34_q97_rejector4k_budget001_seed20260955"
)
DEFAULT_STAGED_DIR = (
    V2
    / "artifacts/invariant_patch/v3_scale"
    / "staged_validate_v34_q97_decoupled_rejector4k_classifier8k_budget001_"
    "seeds20260953_20260954"
)
DEFAULT_CANDIDATE_MANIFEST = (
    HERE / "evidence" / "v3_4_q97_dual_release_candidate.json"
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

# Exact local Python import closure of evaluator-v4, excluding the evaluator
# itself (which has its own independent pin).  These 49 files are an
# admission contract, not a best-effort provenance list.
DEPENDENCY_SOURCE_LABELS: tuple[str, ...] = (
    "training/canonical_probe.py",
    "training/dataset.py",
    "training/invariant_patch_preprocess.py",
    "training/model.py",
    "training/preprocess.py",
    "training/rfgen.py",
    "training/time_domain_geometry.py",
    "training/time_domain_invariant_patch_preprocess.py",
    "training/train.py",
    "training/zplane_ab/common_split.py",
    "training/zplane_ab/train_common.py",
    "training/zplane_ab/zplane_backbone.py",
    "training/zplane_ab/v2_full_variation/assemble_invariant_candidate.py",
    "training/zplane_ab/v2_full_variation/canonical_probe_net.py",
    "training/zplane_ab/v2_full_variation/complex_multiscale_backbone.py",
    "training/zplane_ab/v2_full_variation/denoise_eval.py",
    "training/zplane_ab/v2_full_variation/equalizer_frontend.py",
    "training/zplane_ab/v2_full_variation/evaluate_ab_v2.py",
    "training/zplane_ab/v2_full_variation/evaluate_invariant_release_suite.py",
    "training/zplane_ab/v2_full_variation/full_split.py",
    "training/zplane_ab/v2_full_variation/ground_state_data.py",
    "training/zplane_ab/v2_full_variation/invariant_fusion.py",
    "training/zplane_ab/v2_full_variation/invariant_patch_cnn.py",
    "training/zplane_ab/v2_full_variation/invariant_patch_data.py",
    "training/zplane_ab/v2_full_variation/known_only_patch_openset.py",
    "training/zplane_ab/v2_full_variation/length_aug.py",
    "training/zplane_ab/v2_full_variation/multitask_autoencoder.py",
    "training/zplane_ab/v2_full_variation/native_preprocess.py",
    "training/zplane_ab/v2_full_variation/openset_eval.py",
    "training/zplane_ab/v2_full_variation/pool_cache.py",
    "training/zplane_ab/v2_full_variation/run_bounded_dev.py",
    "training/zplane_ab/v2_full_variation/run_corrected_unet.py",
    "training/zplane_ab/v2_full_variation/run_invariant_cnn_dev.py",
    "training/zplane_ab/v2_full_variation/scalar_transfer.py",
    "training/zplane_ab/v2_full_variation/train_common_v2.py",
    "training/zplane_ab/v2_full_variation/train_transfer.py",
    "training/zplane_ab/v2_full_variation/unet_multitask.py",
    "training/zplane_ab/v2_full_variation/unet_transfer.py",
    "training/zplane_ab/v2_full_variation/v3_time_domain_openset.py",
    "training/zplane_ab/v2_full_variation/vit_backbone.py",
    "training/zplane_ab/v2_full_variation/v3_scale/assemble_v3_fusion.py",
    "training/zplane_ab/v2_full_variation/v3_scale/export_v3_openset_browser_assets.py",
    "training/zplane_ab/v2_full_variation/v3_scale/fit_v3_openset.py",
    "training/zplane_ab/v2_full_variation/v3_scale/fit_v3_openset_staged.py",
    "training/zplane_ab/v2_full_variation/v3_scale/measure_pose_degeneracy.py",
    "training/zplane_ab/v2_full_variation/v3_scale/measure_v3_remaining_gates.py",
    "training/zplane_ab/v2_full_variation/v3_scale/noise_prefilter.py",
    "training/zplane_ab/v2_full_variation/v3_scale/pose_degeneracy.py",
    "training/zplane_ab/v2_full_variation/v3_scale/run_time_domain_dev.py",
)
DEPENDENCY_SOURCES: tuple[tuple[str, Path], ...] = tuple(
    (label, REPO / label) for label in DEPENDENCY_SOURCE_LABELS
)
# Audited current observations are retained to make later final pinning
# reviewable, but they are not admission pins while any member of the
# execution closure is still changing.
KNOWN_CURRENT_DEPENDENCY_SOURCE_SHA256: dict[str, str | None] = {
    "training/canonical_probe.py": "b51387eabb06c740b6a94cefe720546ee2b5d3cbfabde32a82594bb010095acc",
    "training/dataset.py": "8561d3397c9e7d71884cdada681f59d46c8ce05cc7fb7f34f93a1f3ea94fe12c",
    "training/invariant_patch_preprocess.py": "04c66362e6978bc1827791653963d0921bcdde51f93b9dd74e6eb9b4e8217420",
    "training/model.py": "dbe135b9b69dce088f52917b702c1d41f04e97a60a25b41facbc5f9c29b94801",
    "training/preprocess.py": "5369ea8277f64fcd6367a8a363bf7a5a28298663c39315126160ded5bcc0091d",
    "training/rfgen.py": "c5ccb74007328abc3cb43908dd69f032391612feb3f8c848c72e67141e5811ba",
    "training/time_domain_geometry.py": "a9735473f44105d20cffb9d4888ff8cf8e02503746f70b50e6d5d3737138da33",
    "training/time_domain_invariant_patch_preprocess.py": "5cd1787aed0e5fd95adc2e4de56db460d753d0557350aa8f3aff0e393c0bb10a",
    "training/train.py": "0669704ff4933b2ab7f8e35b0b6fb58553a06a8d75c93f54ee928ad33e2e257c",
    "training/zplane_ab/common_split.py": "9e3170631dfc5d1a1b3f534844a48c67c77b48ba49e5265e504414e69ef485db",
    "training/zplane_ab/train_common.py": "05251d47665b8b8a68f9008e6d39c6858f81a68a537249e29730b67e595280e9",
    "training/zplane_ab/zplane_backbone.py": "1c4c840b9f18bc871ca133386a537756ac871cefd71db9e1ee5e8cbfdb8d39bf",
    "training/zplane_ab/v2_full_variation/assemble_invariant_candidate.py": "de2215ffa73cc046bd5de817ca2516ecf396ff9f1767022bb1d99cccc4fb9f09",
    "training/zplane_ab/v2_full_variation/canonical_probe_net.py": "c778c908021347084d0e92f5de0fb1b3adb4e39333aad734edee2fa1cae58e79",
    "training/zplane_ab/v2_full_variation/complex_multiscale_backbone.py": "ff029ff9380a9a6f13d99cf9cdec2cbe0755d175a2ba9f9147c99c8c74cc9350",
    "training/zplane_ab/v2_full_variation/denoise_eval.py": "262c4d8e5f35296466f4c1dd7cfbbdd139ec1e6457158f0c7a1eff7d2c818ccb",
    "training/zplane_ab/v2_full_variation/equalizer_frontend.py": "fc495924b9dee934f6fc4e6fcd5acfa3c09ee86381dcfa5c284442b1e50c8381",
    "training/zplane_ab/v2_full_variation/evaluate_ab_v2.py": "f8ecc75b569cc89fbf91bc94a826b3371cdc97069f02d7f68653f07fd1d5070d",
    "training/zplane_ab/v2_full_variation/evaluate_invariant_release_suite.py": "1b8137b4c222a857a91f340730137fefd3fe17a026d9ba5eb172e7fd774c0541",
    "training/zplane_ab/v2_full_variation/full_split.py": "7ecb6bcd73549107fb3f4423b43509f471e7253711dfea12b6e4a578060417ba",
    "training/zplane_ab/v2_full_variation/ground_state_data.py": "67071dc1690d546830323efde0af0e2285e6efe318eb1aea01241645908081cd",
    "training/zplane_ab/v2_full_variation/invariant_fusion.py": "22a46a75c95cb4aa452bc44228b156b57474b981da20cf7095fff8a129f1c281",
    "training/zplane_ab/v2_full_variation/invariant_patch_cnn.py": "b402e8bb75a5809f4a57e8de231268e9ad18afc26b2cfcbf57d2f0b43804ede4",
    "training/zplane_ab/v2_full_variation/invariant_patch_data.py": "63e6134bddb927149a54c73fb011c580d1423b8b9977bab0e0b88655c3d0a798",
    "training/zplane_ab/v2_full_variation/known_only_patch_openset.py": "4764a9997331bb657787dcff211ab8d7098f98536208c1c0432af8f373763785",
    "training/zplane_ab/v2_full_variation/length_aug.py": "e356b73f5bdee5c3ccd24cfd1473e89216b742789183db7bd9b16456fa1ff428",
    "training/zplane_ab/v2_full_variation/multitask_autoencoder.py": "af4a7b6f44eaaa94a2c312520e0f48ea15f46b115bfb7c95c250c1cf4912f53f",
    "training/zplane_ab/v2_full_variation/native_preprocess.py": "6752cb2c83f5dc6f4b634017d2c3ab16e4da4e85eea83c777378a37dbb9a4227",
    "training/zplane_ab/v2_full_variation/openset_eval.py": "1675b9ce60139e58b0d12dd287552b22d8bc47cfcb264a0ec347def251ed3354",
    "training/zplane_ab/v2_full_variation/pool_cache.py": "59dc10d2ee0c0b6ff1722ab93fa43bda8d0e63257397ae8dca6674d301f40d53",
    "training/zplane_ab/v2_full_variation/run_bounded_dev.py": "8564e66208a1eb5051df4cbe74f61124c208bd3b6209cb909c33ec56bb7e6869",
    "training/zplane_ab/v2_full_variation/run_corrected_unet.py": "73482231d2d5eb3ae73221ed702adf9cd2e0709fd3a788b66d8a067e747977ca",
    "training/zplane_ab/v2_full_variation/run_invariant_cnn_dev.py": "22036917af78e48e00dcfe0922c464d0cab352750c00fcb45d6aece1f4fd9cfb",
    "training/zplane_ab/v2_full_variation/scalar_transfer.py": "2e5d8380d9663127a89f96cf2bfca8cc6d4d4df3f73b1d804eacfcffafa54848",
    "training/zplane_ab/v2_full_variation/train_common_v2.py": "9be6b5d08a3fc5c3aff1e3110c8df440aad6bde9d21ad1059d9ba666439cebca",
    "training/zplane_ab/v2_full_variation/train_transfer.py": "d5e68558657ba96348bef30cf749f199497c7b3c306126f09086b009c51c2592",
    "training/zplane_ab/v2_full_variation/unet_multitask.py": "23178c9d18b7cc9aed95df5288f0543abd6659961c0f68c8fd417daafbd059af",
    "training/zplane_ab/v2_full_variation/unet_transfer.py": "3b9f7e0998d11f83328cd9295a46c54ab656630a3f0c4800f3479d0a99da05a1",
    "training/zplane_ab/v2_full_variation/v3_time_domain_openset.py": "ae1ddb6c14777cd27be591345a23ca567bd1e20accd4b7729804a5453144ea64",
    "training/zplane_ab/v2_full_variation/vit_backbone.py": "ea0255c82d222b2bd8a3d3bf84d4b2b8dfaa49819950193495fcda824a280f9c",
    "training/zplane_ab/v2_full_variation/v3_scale/assemble_v3_fusion.py": "961e493697d1d941395dafe32570262dc09996b53fed9b2a0e82525e43560daa",
    "training/zplane_ab/v2_full_variation/v3_scale/export_v3_openset_browser_assets.py": "13722adf59c49826a7d356d8b3ce538a3da46ef30b0ea6bb0194e53fe40fbdb1",
    "training/zplane_ab/v2_full_variation/v3_scale/fit_v3_openset.py": "970e536087c0c4660424628a69e8bb08955885b337fd6ac560edf4bfaec1fba9",
    "training/zplane_ab/v2_full_variation/v3_scale/fit_v3_openset_staged.py": "da79f85b68ffd5ab7b4bdba8f78e336af698bbeef9bfabdf766f0eb96f23ad88",
    "training/zplane_ab/v2_full_variation/v3_scale/measure_pose_degeneracy.py": "7a532f2420dfd7c77deed1337c2b8d710a8e5546955452b89855aa55ef653565",
    "training/zplane_ab/v2_full_variation/v3_scale/measure_v3_remaining_gates.py": "fd8765614b9cadbbf852845ef11e26e861ab77d510d2a91c87513eea4e6cb405",
    "training/zplane_ab/v2_full_variation/v3_scale/noise_prefilter.py": "435495e5b0d11558a2e0edfd7d6e6112e7cebf716460d3d81ae741d97deaa7b4",
    "training/zplane_ab/v2_full_variation/v3_scale/pose_degeneracy.py": "431499124cd8470276a22356622b54f9c60c63c15e3770f49944bc626091aaf1",
    "training/zplane_ab/v2_full_variation/v3_scale/run_time_domain_dev.py": "9446a23b8d80c86fd7984307d3a2c04530e12f553df5b778967a4fd54dd193c4",
}
DEPENDENCY_SOURCE_SHA256: dict[str, str] = copy.deepcopy(
    KNOWN_CURRENT_DEPENDENCY_SOURCE_SHA256
)
DEFAULT_PINS["dependency_source_sha256"] = copy.deepcopy(
    DEPENDENCY_SOURCE_SHA256
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


def _is_sha256_pin(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _local_import_closure(entrypoint: Path) -> set[Path]:
    """Resolve the evaluator's simple local imports with its sys.path order."""
    import ast

    roots = (HERE, V2, ZPAB, TRAINING)

    def resolve_module(module: str) -> Path | None:
        top_level = module.split(".", 1)[0]
        for root in roots:
            module_file = root / f"{top_level}.py"
            if module_file.is_file():
                return module_file.resolve()
            package_file = root / top_level / "__init__.py"
            if package_file.is_file():
                return package_file.resolve()
        return None

    closure: set[Path] = set()
    pending = [Path(entrypoint).resolve()]
    while pending:
        path = pending.pop()
        if path in closure:
            continue
        closure.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for module in modules:
                resolved = resolve_module(module)
                if resolved is not None and resolved not in closure:
                    pending.append(resolved)
    return closure


def check_future_release_pins(pins: Mapping[str, Any]) -> list[Check]:
    """Make every not-yet-frozen release-chain dependency fail explicitly."""
    flat = {
        "launcher": pins.get("launcher_sha256"),
        "evaluator": pins.get("v3_evaluator_sha256"),
        "candidate_manifest": pins.get("candidate_manifest_sha256"),
        "validation_evidence": pins.get("validation_evidence_sha256"),
        "candidate_contract": pins.get("frozen_candidate_contract_sha256"),
        "staged_validation_report": pins.get(
            "staged_validation_report_sha256"
        ),
        "staging_package_manifest": pins.get(
            "staging_package_manifest_sha256"
        ),
        "dual_binding": pins.get("dual_binding_sha256"),
        "protocol_fixture": pins.get("v3_protocol_fixture_sha256"),
    }
    for name, value in sorted(
        dict(pins.get("staged_artifact_sha256", {})).items()
    ):
        flat[f"staged_validation_artifact.{name}"] = value
    for role, value in sorted(
        dict(pins.get("browser_asset_sha256", {})).items()
    ):
        flat[f"browser_asset.{role}"] = value
    checks: list[Check] = []
    for name, value in sorted(flat.items()):
        ready = _is_sha256_pin(value)
        checks.append(
            (
                f"future_pin.{name}",
                ready,
                f"frozen lowercase SHA-256 {value}"
                if ready
                else (
                    "unbound fail-closed placeholder; freeze the actual "
                    "artifact and replace None with its verified SHA-256"
                ),
            )
        )
    reasons = dict(pins.get("validation_seed_reasons", {}))
    for seed in pins["staged_validation_novelty_seeds"]:
        reason = reasons.get(int(seed))
        ready = isinstance(reason, str) and bool(reason.strip())
        checks.append(
            (
                f"future_pin.validation_seed_reason.{int(seed)}",
                ready,
                reason
                if ready
                else (
                    "unbound fail-closed placeholder; record the exact "
                    "post-validation ledger reason after the one draw"
                ),
            )
        )
    return checks


def check_design_evidence(
    design_dir: Path,
    expected: Mapping[str, Any],
    *,
    label: str,
) -> list[Check]:
    """Verify an immutable design draw without loading any population rows."""
    root = Path(design_dir)
    report_path = root / "openset_metrics.json"
    report = _read_json(report_path)
    checks: list[Check] = []
    checks.append(
        _match(
            f"{label}.report_sha256",
            _sha256(report_path),
            expected["report_sha256"],
        )
    )
    identity = {
        "role": report.get("role"),
        "status": report.get("status"),
        "all_pass": report.get("all_pass"),
        "gates_are_evidence": report.get("gates_are_evidence"),
        "release_evidence": report.get("release_evidence"),
        "sealed_release_data_used": report.get("sealed_release_data_used"),
        "consumed_test_rows_used": report.get("consumed_test_rows_used"),
    }
    expected_identity = {
        "role": expected["role"],
        "status": expected["status"],
        "all_pass": expected["all_pass"],
        "gates_are_evidence": expected["gates_are_evidence"],
        "release_evidence": False,
        "sealed_release_data_used": 0,
        "consumed_test_rows_used": 0,
    }
    checks.append(
        _match(f"{label}.role_status_hygiene", identity, expected_identity)
    )
    seeds = report.get("seeds", {})
    seed_identity = {
        "design_novelty_seed": seeds.get("design_novelty_seed"),
        "novelty_seeds": seeds.get("novelty_seeds"),
        "release_seed_not_spent": seeds.get("release_seed_not_spent"),
    }
    expected_seed = int(expected["novelty_seed"])
    checks.append(
        _match(
            f"{label}.seed_role",
            seed_identity,
            {
                "design_novelty_seed": expected_seed,
                "novelty_seeds": [expected_seed],
                "release_seed_not_spent": 20260736,
            },
        )
    )
    architecture = report.get("architecture", {})
    composite = report.get("composite", {})
    policy_identity = {
        "schema": architecture.get("schema"),
        "version": architecture.get("staged_policy_version"),
        "kind": architecture.get("kind"),
        "composite_schema": composite.get("schema"),
        "composite_version": composite.get("policy_version"),
        "composite_kind": composite.get("kind"),
        "threshold_quantile": composite.get("threshold_quantile"),
        "threshold": composite.get("threshold"),
    }
    expected_policy_identity = {
        "schema": expected["policy_schema"],
        "version": expected["policy_version"],
        "kind": expected["policy_kind"],
        "composite_schema": expected["policy_schema"],
        "composite_version": expected["policy_version"],
        "composite_kind": expected["policy_kind"],
        "threshold_quantile": expected["threshold_quantile"],
        "threshold": expected["threshold"],
    }
    checks.append(
        _match(
            f"{label}.policy_identity",
            policy_identity,
            expected_policy_identity,
        )
    )
    checks.append(
        _match(
            f"{label}.source_sha256",
            report.get("source_sha256", {}).get("fit_v3_openset_staged.py"),
            expected["source_sha256"],
        )
    )
    recorded = report.get("artifacts", {})
    expected_artifacts = dict(expected["artifact_sha256"])
    on_disk = {
        name: _sha256(root / name)
        for name in sorted(expected_artifacts)
        if (root / name).is_file()
    }
    checks.append(
        _match(f"{label}.recorded_artifact_sha256", recorded, expected_artifacts)
    )
    checks.append(
        _match(f"{label}.on_disk_artifact_sha256", on_disk, expected_artifacts)
    )
    gates = report.get("gates", {})
    failed = sorted(
        name
        for name, gate in gates.items()
        if isinstance(gate, Mapping) and gate.get("passes") is False
    )
    expected_failed = (
        [] if expected["failed_gate"] is None else [expected["failed_gate"]]
    )
    checks.append(_match(f"{label}.failed_gates", failed, expected_failed))
    if expected["failed_gate"] is not None:
        checks.append(
            _match(
                f"{label}.failed_gate_worst",
                gates.get(expected["failed_gate"], {}).get("worst"),
                expected["failed_gate_worst"],
            )
        )
    checks.append(
        _match(
            f"{label}.known_false_unknown_worst",
            gates.get("known_false_unknown_rate", {}).get("worst"),
            expected["known_false_unknown_worst"],
        )
    )
    return checks


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
    q97 threshold from the stored calibration and refuses any other policy
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
        and float(
            report.get("composite", {}).get("threshold_quantile", -1)
        )
        == float(pins["staged_policy_threshold_quantile"])
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
            evaluator.EVALUATOR_SCHEMA == pins.get("evaluator_schema")
            and evaluator.EVALUATION_VERSION
            == pins.get("evaluation_version")
            and evaluator.CANDIDATE_ID == pins.get("candidate_id")
            and evaluator.CANDIDATE_MANIFEST_SCHEMA
            == pins.get("candidate_manifest_schema")
            and evaluator.CANDIDATE_EVIDENCE_SCHEMA
            == pins.get("candidate_evidence_schema")
            and evaluator.CANDIDATE_CONTRACT_SCHEMA
            == pins.get("candidate_contract_schema")
            and evaluator.EXPECTED_RELEASE_SEED == pins.get("release_seed")
            and evaluator.EXPECTED_STAGED_POLICY_SCHEMA
            == pins.get("staged_policy_schema")
            and evaluator.EXPECTED_STAGED_POLICY_VERSION
            == pins.get("staged_policy_version")
            and evaluator.EXPECTED_STAGED_POLICY_KIND
            == pins.get("staged_policy_kind")
            and evaluator.STAGING_PACKAGE_SCHEMA
            == pins.get("staging_package_schema")
            and evaluator.STAGING_PACKAGE_SCHEMA_VERSION
            == pins.get("staging_package_schema_version")
            and evaluator.DUAL_BINDING_SCHEMA
            == pins.get("dual_binding_schema")
            and evaluator.DUAL_BINDING_SCHEMA_VERSION
            == pins.get("dual_binding_schema_version")
            and evaluator.STAGING_STATUS == pins.get("staging_status")
            and evaluator.BROWSER_FUSION_SCHEMA
            == pins.get("browser_fusion_schema")
            and evaluator.BROWSER_FUSION_SCHEMA_VERSION
            == pins.get("browser_fusion_schema_version")
            and evaluator.BROWSER_OPENSET_SCHEMA
            == pins.get("browser_openset_schema")
            and evaluator.BROWSER_OPENSET_SCHEMA_VERSION
            == pins.get("browser_openset_schema_version")
            and evaluator.PARITY_SCHEMA == pins.get("parity_schema")
            and evaluator.PARITY_SCHEMA_VERSION
            == pins.get("parity_schema_version"),
            "evaluator schema 4 is frozen for release36/q97; dual runtime "
            "package and binding remain schema v1 staging_not_release; "
            "legacy candidates are inadmissible",
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
            and {
                role: record.get("sha256")
                for role, record in candidate.candidate_manifest.get(
                    "browser_assets", {}
                ).items()
            }
            == dict(pins.get("browser_asset_sha256", {}))
            and candidate.candidate_manifest.get(
                "staging_package_manifest", {}
            ).get("sha256")
            == pins.get("staging_package_manifest_sha256")
            and candidate.candidate_manifest.get("dual_binding", {}).get(
                "sha256"
            )
            == pins.get("dual_binding_sha256")
        )
        checks.append(
            (
                "candidate.complete_binding_chain",
                exact,
                "final manifest, validation evidence, pre-validation "
                "contract, both role bundles/fusions, staged policy, "
                "prefilter, browser assets, staging package and dual binding "
                "all load under evaluator schema 4"
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
            "policy.staged_identity",
            {
                "schema": int(staged.STAGED_POLICY_SCHEMA),
                "version": staged.STAGED_POLICY_VERSION,
                "kind": staged.STAGED_POLICY_KIND,
                "threshold_quantile": float(
                    staged.COMPOSITE_THRESHOLD_QUANTILE
                ),
            },
            {
                "schema": int(pins["staged_policy_schema"]),
                "version": pins["staged_policy_version"],
                "kind": pins["staged_policy_kind"],
                "threshold_quantile": float(
                    pins["staged_policy_threshold_quantile"]
                ),
            },
        )
    )
    # One current ledger must describe the complete evidence lifecycle before
    # either the next development draw or the sealed draw is allowed.
    consumed_now = sorted(int(s) for s in pins["consumed_release_seeds"])
    new_seed = int(pins["release_seed"])
    expected_consumed_reasons = dict(
        pins["consumed_release_seed_reasons"]
    )
    expected_design_reasons = dict(pins["spent_design_seed_reasons"])
    expected_validation_reasons = dict(pins["validation_seed_reasons"])
    validation_reasons_bound = all(
        isinstance(reason, str) and bool(reason.strip())
        for reason in expected_validation_reasons.values()
    )
    expected_spent = set(range(20260938, 20260956))
    observed_spent = {int(seed) for seed in staged.SPENT_NOVELTY_SEEDS}
    ledger_ok = (
        int(staged.RELEASE_SEED_NEVER_SPENT_HERE) == new_seed
        and dict(staged.CONSUMED_SEALED_RELEASE_SEEDS)
        == expected_consumed_reasons
        and int(openset_base.RELEASE_SEED_NEVER_SPENT_HERE)
        in staged.CONSUMED_SEALED_RELEASE_SEEDS
        and consumed_now == sorted(expected_consumed_reasons)
        and new_seed not in consumed_now
        and new_seed not in staged.SPENT_NOVELTY_SEEDS
        and new_seed not in staged.CONSUMED_SEALED_RELEASE_SEEDS
        and observed_spent == expected_spent
        and int(staged.PROPOSED_DESIGN_NOVELTY_SEED)
        == int(pins["staged_design_novelty_seed"])
        and tuple(staged.DEFAULT_VALIDATION_NOVELTY_SEEDS)
        == tuple(pins["staged_validation_novelty_seeds"])
        and int(staged.FIRST_CLEAN_NOVELTY_SEED) == 20260956
        and int(staged.PROPOSED_DESIGN_NOVELTY_SEED)
        in staged.SPENT_NOVELTY_SEEDS
        and set(staged.DEFAULT_VALIDATION_NOVELTY_SEEDS).issubset(
            staged.SPENT_NOVELTY_SEEDS
        )
        and all(
            staged.SPENT_NOVELTY_SEEDS.get(seed) == reason
            for seed, reason in expected_design_reasons.items()
        )
        and validation_reasons_bound
        and all(
            staged.SPENT_NOVELTY_SEEDS.get(seed) == reason
            for seed, reason in expected_validation_reasons.items()
        )
    )
    checks.append(
        (
            "policy.seed_ledger_constants",
            ledger_ok,
            f"release seed {new_seed} reserved and unreachable from "
            f"development; consumed releases {consumed_now}; spent novelty "
            "seeds 20260938-20260955; exact q99/q97 design and validation "
            "roles/reasons bound; spent design and consumed validation "
            f"{pins['staged_design_novelty_seed']}/"
            f"{list(pins['staged_validation_novelty_seeds'])}"
            if ledger_ok
            else (
                "release36 requires exact consumed-release reasons, exact "
                "q99/q97 design reasons, spent validation seeds 20260953/"
                "20260954 with their post-draw reasons, and first-clean "
                "20260956; an unbound validation reason is a hard NO-GO"
            ),
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
        checks.append(
            (
                "sources.v3_evaluator_identity",
                False,
                "evaluator-v4 source is missing",
            )
        )
    else:
        import ast
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
        tree = ast.parse(
            evaluator_path.read_text(encoding="utf-8"),
            filename=str(evaluator_path),
        )
        wanted = {
            "EVALUATOR_SCHEMA",
            "EVALUATION_VERSION",
            "EXPECTED_RELEASE_SEED",
            "CANDIDATE_ID",
            "CANDIDATE_MANIFEST_SCHEMA",
            "CANDIDATE_EVIDENCE_SCHEMA",
            "CANDIDATE_CONTRACT_SCHEMA",
            "EXPECTED_STAGED_POLICY_SCHEMA",
            "EXPECTED_STAGED_POLICY_VERSION",
            "EXPECTED_STAGED_POLICY_KIND",
            "EXPECTED_COMPOSITE_THRESHOLD_QUANTILE",
        }
        literals: dict[str, Any] = {}
        for node in tree.body:
            targets = node.targets if isinstance(node, ast.Assign) else ()
            for target in targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    try:
                        literals[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        literals[target.id] = "<non-literal>"
        expected_identity = {
            "EVALUATOR_SCHEMA": pins["evaluator_schema"],
            "EVALUATION_VERSION": pins["evaluation_version"],
            "EXPECTED_RELEASE_SEED": pins["release_seed"],
            "CANDIDATE_ID": pins["candidate_id"],
            "CANDIDATE_MANIFEST_SCHEMA": pins[
                "candidate_manifest_schema"
            ],
            "CANDIDATE_EVIDENCE_SCHEMA": pins[
                "candidate_evidence_schema"
            ],
            "CANDIDATE_CONTRACT_SCHEMA": pins[
                "candidate_contract_schema"
            ],
            "EXPECTED_STAGED_POLICY_SCHEMA": pins["staged_policy_schema"],
            "EXPECTED_STAGED_POLICY_VERSION": pins["staged_policy_version"],
            "EXPECTED_STAGED_POLICY_KIND": pins["staged_policy_kind"],
            "EXPECTED_COMPOSITE_THRESHOLD_QUANTILE": pins[
                "staged_policy_threshold_quantile"
            ],
        }
        checks.append(
            _match(
                "sources.v3_evaluator_identity",
                literals,
                expected_identity,
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
    expected_hashes = dict(pins.get("dependency_source_sha256", {}))
    labels = tuple(label for label, _ in DEPENDENCY_SOURCES)
    import evaluate_v3_release_suite as v3_evaluator

    evaluator_paths = {
        key: Path(path).resolve()
        for key, path in v3_evaluator.TRANSITIVE_DEPENDENCY_PATHS.items()
    }
    local_paths = {
        key: Path(path).resolve() for key, path in DEPENDENCY_SOURCES
    }
    checks.append(
        _match(
            "sources.evaluator_dependency_path_contract",
            local_paths,
            evaluator_paths,
        )
    )
    checks.append(
        _match(
            "sources.evaluator_dependency_hash_contract",
            expected_hashes,
            dict(v3_evaluator.EXPECTED_TRANSITIVE_DEPENDENCY_SHA256),
        )
    )
    checks.append(
        _match(
            "sources.evaluator_runtime_contract",
            dict(pins["python_runtime"]),
            dict(v3_evaluator.EXPECTED_EVALUATOR_RUNTIME_IDENTITY),
        )
    )
    checks.append(
        _match(
            "sources.dependency_contract_key_set",
            tuple(sorted(expected_hashes)),
            tuple(sorted(labels)),
        )
    )
    hashes: dict[str, str] = {}
    invalid_files: list[str] = []
    for label, path in DEPENDENCY_SOURCES:
        path = Path(path)
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            invalid_files.append(f"{label}: missing")
            continue
        if path.is_symlink() or not stat.S_ISREG(mode):
            invalid_files.append(f"{label}: not a regular non-symlink file")
            continue
        hashes[label] = _sha256(path)
    observed["dependency_source_sha256"] = hashes
    checks.append(
        (
            "sources.dependency_hashes_recorded",
            not invalid_files and len(hashes) == 49,
            "all 49 regular non-symlink dependency files hashed"
            if not invalid_files and len(hashes) == 49
            else "; ".join(invalid_files),
        )
    )
    drift = {
        label: {"expected": expected_hashes.get(label), "observed": actual}
        for label, actual in sorted(hashes.items())
        if not _is_sha256_pin(expected_hashes.get(label))
        or expected_hashes.get(label) != actual
    }
    checks.append(
        (
            "sources.dependency_hashes_pinned",
            not drift and set(hashes) == set(labels),
            "all 49 dependency hashes exactly match their frozen pins"
            if not drift and set(hashes) == set(labels)
            else json.dumps(drift, sort_keys=True),
        )
    )
    if evaluator_path.is_file():
        closure = _local_import_closure(evaluator_path)
        expected_closure = {
            path.resolve() for _, path in DEPENDENCY_SOURCES
        } | {evaluator_path.resolve()}
        missing_from_contract = sorted(
            str(path.relative_to(REPO))
            for path in closure - expected_closure
        )
        excess_contract = sorted(
            str(path.relative_to(REPO))
            for path in expected_closure - closure
        )
        checks.append(
            (
                "sources.static_import_closure",
                not missing_from_contract and not excess_contract,
                "evaluator plus the exact 49-file local import closure"
                if not missing_from_contract and not excess_contract
                else (
                    f"missing={missing_from_contract}; "
                    f"excess={excess_contract}"
                ),
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


def check_seed_ledger(
    releases_dir: Path,
    pins: Mapping[str, Any],
    *,
    require_canonical: bool = True,
) -> list[Check]:
    checks: list[Check] = []
    releases_dir = Path(releases_dir).expanduser()
    seed_text = str(pins["release_seed"])
    problems: list[str] = []
    absolute_releases_dir = Path(
        os.path.abspath(os.fspath(releases_dir))
    )
    symlink_component = any(
        component.is_symlink()
        for component in (
            absolute_releases_dir,
            *absolute_releases_dir.parents,
        )
    )
    root_is_real_dir = (
        releases_dir.is_dir()
        and not releases_dir.is_symlink()
        and (not require_canonical or not symlink_component)
    )
    canonical_path = DEFAULT_RELEASES_DIR
    canonical_ok = (
        not require_canonical
        or (
            int(pins.get("release_seed", -1)) == 20260736
            and releases_dir == canonical_path
        )
    )
    root_ok = root_is_real_dir and canonical_ok
    identity_detail = (
        f"exact canonical non-symlink release ledger {canonical_path}"
        if root_ok
        else (
            f"observed={releases_dir}, expected={canonical_path}, "
            f"is_dir={releases_dir.is_dir()}, "
            f"is_symlink={releases_dir.is_symlink()}, "
            f"has_symlink_component={symlink_component}, "
            f"require_canonical={require_canonical}"
        )
    )
    checks.append(("ledger.releases_dir_identity", root_ok, identity_detail))
    if not root_is_real_dir:
        problems.append(
            f"{releases_dir}: release ledger root must be a real "
            "non-symlink directory"
        )
    elif not canonical_ok:
        problems.append(
            f"{releases_dir}: release-36 ledger must be exactly "
            f"{canonical_path}"
        )
    else:
        pending = [releases_dir]
        while pending:
            directory = pending.pop()
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda item: item.name)
            for entry in entries:
                relative = Path(entry.path).relative_to(releases_dir)
                if entry.is_symlink():
                    problems.append(f"{relative}: symlink is forbidden")
                    continue
                if entry.is_dir(follow_symlinks=False):
                    if seed_text in entry.name:
                        problems.append(f"release path {relative} names the seed")
                    pending.append(Path(entry.path))
                    continue
                if not entry.is_file(follow_symlinks=False):
                    problems.append(f"{relative}: unsupported filesystem entry")
                    continue
                if entry.name != "RELEASE_INTENT.json":
                    continue
                try:
                    intent = _read_json(Path(entry.path))
                except Exception as error:  # noqa: BLE001 - ledger corruption
                    problems.append(
                        f"{relative}: unreadable intent: "
                        f"{type(error).__name__}: {error}"
                    )
                    continue
                if int(intent.get("release_seed", -1)) == int(
                    pins["release_seed"]
                ):
                    problems.append(f"{relative} records the seed")
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
        if not root_ok:
            bad.append("release ledger root identity was refused")
        elif sealed_root.is_symlink() or not sealed_root.is_dir():
            bad.append("consumed root must be a real non-symlink directory")
        for name, expected in sorted(expected_hashes.items()):
            path = sealed_root / name
            if bad and not root_ok:
                continue
            if path.is_symlink() or not path.is_file():
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
    seed35_evidence = (
        HERE / "evidence" / str(pins["seed35_preflight_evidence_name"])
    )
    evidence_regular = (
        seed35_evidence.exists()
        and not seed35_evidence.is_symlink()
        and stat.S_ISREG(seed35_evidence.lstat().st_mode)
    )
    if not evidence_regular:
        checks.append(
            (
                "generation.seed35_go_preflight_evidence",
                False,
                f"{seed35_evidence} is not a regular non-symlink file",
            )
        )
    else:
        evidence_sha = _sha256(seed35_evidence)
        evidence_ok = (
            evidence_sha == pins["seed35_preflight_evidence_sha256"]
        )
        if evidence_ok:
            evidence = _read_json(seed35_evidence)
            evidence_ok = (
                evidence.get("go") is True
                and int(evidence.get("release_seed", -1)) == 20260735
            )
        checks.append(
            (
                "generation.seed35_go_preflight_evidence",
                evidence_ok,
                "seed-20260735 GO preflight evidence is hash-pinned and intact"
                if evidence_ok
                else (
                    f"expected SHA-256 "
                    f"{pins['seed35_preflight_evidence_sha256']}, "
                    f"observed {evidence_sha}"
                ),
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
    if (
        not fixture_path.exists()
        or fixture_path.is_symlink()
        or not stat.S_ISREG(fixture_path.lstat().st_mode)
    ):
        checks.append(
            (
                "generation.v3_protocol_fixture",
                False,
                f"{fixture_path} is missing; regenerate it with "
                "evaluate_v3_release_suite.py --print-expected-protocol "
                f"{pins['release_seed']}",
            )
        )
    elif not _is_sha256_pin(expected_fixture_sha):
        checks.append(
            (
                "generation.v3_protocol_fixture",
                False,
                "pins.v3_protocol_fixture_sha256 is not pinned; refusing to "
                "treat an unpinned fixture as GO",
            )
        )
    else:
        fixture_sha = _sha256(fixture_path)
        if fixture_sha != expected_fixture_sha:
            checks.append(
                (
                    "generation.v3_protocol_fixture",
                    False,
                    f"fixture SHA-256 {fixture_sha} differs from the frozen "
                    f"pin {expected_fixture_sha}; JSON was not parsed",
                )
            )
            return checks
        wrapper = _read_json(fixture_path)
        protocol = wrapper.get("evaluation_protocol", {})
        fixture_ok = (
            int(wrapper.get("release_seed", -1)) == int(pins["release_seed"])
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
    expected_correction = pins["historical_claim_correction"]
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
    correction = historical.get("historical_claim_correction")
    if correction != expected_correction:
        problems.append(
            "historical claim correction does not exactly record the "
            "cross-seed/unpaired comparison, eight v2 failures and unsupported "
            "'every axis' claim"
        )
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
            "seed-20260734 0.12/0.84 redeclaration is inactive and its old "
            "unpaired/eight-gate claim is corrected machine-readably"
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


def _python_runtime_snapshot() -> dict[str, str]:
    import torch

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": str(torch.__version__).split("+", 1)[0],
        "device": "cpu",
        "platform": (
            f"{platform.system().lower()}-{platform.machine().lower()}"
        ),
    }


def check_python_runtime(
    pins: Mapping[str, Any] = DEFAULT_PINS,
    *,
    runtime: Mapping[str, str] | None = None,
    warnings_policy: str | None = None,
) -> list[Check]:
    import noise_prefilter as npf

    checks: list[Check] = []
    runtime = (
        _python_runtime_snapshot() if runtime is None else dict(runtime)
    )
    expected = dict(pins["python_runtime"])
    checks.append(
        _match("runtime.python_contract_key_set", sorted(runtime), sorted(expected))
    )
    for key, value in sorted(expected.items()):
        checks.append(_match(f"runtime.python.{key}", runtime.get(key), value))
    warnings_policy = (
        os.environ.get("PYTHONWARNINGS", "")
        if warnings_policy is None
        else warnings_policy
    )
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
            f"IEEE flags (numpy {runtime.get('numpy')}, "
            f"torch {runtime.get('torch')}, CPU)",
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
    """The exact one-shot launcher invocation for seed 20260736.

    Reconstructed from the consumed release mechanics and re-aimed at the new
    untouched seed.
    The candidate file is the final dual release-candidate manifest: it binds
    both role bundles/fusions, validation evidence, staged/prefilter state,
    browser assets, staging package and the deploy-time dual binding.
    This function only formats strings; nothing is executed.
    """
    manifest_path = Path(candidate_manifest).expanduser().resolve()
    if int(pins.get("release_seed", -1)) != 20260736:
        raise ValueError("release command is reserved for seed 20260736")
    if int(pins.get("target_per_class", -1)) != 192:
        raise ValueError("release-36 requires exactly 192 rows per class")
    canonical_manifest = DEFAULT_CANDIDATE_MANIFEST.resolve()
    if manifest_path != canonical_manifest:
        raise ValueError(
            f"release-36 candidate path must be exactly {canonical_manifest}"
        )
    for name in (
        "candidate_manifest_sha256",
        "v3_protocol_fixture_sha256",
        "launcher_sha256",
    ):
        if not _is_sha256_pin(pins.get(name)):
            raise ValueError(f"{name} is an unbound fail-closed placeholder")
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
    q99_design_dir = Path(args.q99_design_dir)
    q97_design_dir = Path(args.q97_design_dir)
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
        "q99_design",
        lambda: check_design_evidence(
            q99_design_dir,
            pins["q99_design_evidence"],
            label="q99_design",
        ),
    )
    run_group(
        "q97_design",
        lambda: check_design_evidence(
            q97_design_dir,
            pins["q97_design_evidence"],
            label="q97_design",
        ),
    )
    run_group("future_pins", lambda: check_future_release_pins(pins))
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
    run_group("runtime.python", lambda: check_python_runtime(pins))

    generation: dict[str, Any] | None = None
    if all(item["passed"] for item in checks):
        try:
            generation = build_generation_command(
                pins, candidate_manifest, isolated_root, node_bin_dir
            )
            checks.append(
                {
                    "name": "generation.command_reconstruction",
                    "passed": True,
                    "detail": "exact one-shot command reconstructed",
                }
            )
        except Exception as error:  # noqa: BLE001 - itemised fail closed
            checks.append(
                {
                    "name": "generation.command_reconstruction",
                    "passed": False,
                    "detail": f"{type(error).__name__}: {error}",
                }
            )

    go = all(item["passed"] for item in checks) and generation is not None
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
            "q99_design_dir": str(q99_design_dir),
            "q97_design_dir": str(q97_design_dir),
            "staged_dir": str(staged_dir),
            "releases_dir": str(releases_dir),
            "isolated_root": str(isolated_root),
            "evaluator": str(evaluator_path),
            "node_bin_dir": str(node_bin_dir),
        },
        "consumes_nothing": {
            "release_seed_drawn": False,
            "novelty_realizations_generated": 0,
            "corpus_rows_read": 0,
            "sealed_labels_read": False,
            "writes_under_releases": False,
        },
    }
    if go:
        report["generation_command"] = generation
    return report


def _print_report(report: Mapping[str, Any], stream=None) -> None:
    stream = sys.stdout if stream is None else stream
    for item in report["checks"]:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"{status}  {item['name']}: {item['detail']}", file=stream)
    generation = report.get("generation_command") or {}
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
    parser.add_argument(
        "--q99-design-dir", default=str(DEFAULT_Q99_DESIGN_DIR)
    )
    parser.add_argument(
        "--q97-design-dir", default=str(DEFAULT_Q97_DESIGN_DIR)
    )
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
        supplied_releases_dir = Path(args.releases_dir).resolve()
        canonical_releases_dir = DEFAULT_RELEASES_DIR.resolve()
        forbidden_release_roots = {
            supplied_releases_dir,
            canonical_releases_dir,
        }
        if any(
            root == output or root in output.parents
            for root in forbidden_release_roots
        ):
            print(
                "refusing to write the preflight report under the release "
                "tree "
                + ", ".join(
                    str(path) for path in sorted(
                        forbidden_release_roots, key=str
                    )
                ),
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
