"""Evaluate one sealed v3 release suite against the frozen staged v3 candidate.

This is the v3 counterpart of ``evaluate_invariant_release_suite.py`` (the v2
sealed evaluator).  It preserves the v2 discipline exactly where the candidate
did not change, and states precisely where the staged rejector forced a change:

* Every gate name, floor, comparison and the gate assembler itself are
  **imported** from the v2 evaluator (``GATE_FLOORS``, ``_gate``,
  ``_assemble_release_gates``, ``_five_shot_report``, ``_auroc``,
  ``_closed_report``, ...).  No gate is added, removed or re-levelled for the
  next candidate: five-shot remains >= 0.85 and known FUR remains <= 0.10.
  Seed 20260734's predeclared 0.84/0.12 redeclaration is retained verbatim as
  labelled historical metadata only.  It is inactive for this protocol and
  cannot affect a gate object or pass/fail decision.
* The suite-side verification (intent/manifest binding, corpus byte hashes,
  bit-exact prefix nesting, prefix-derivation records, unscored start probe,
  dependency provenance, predeclared five-shot split) is the v2 machinery,
  imported and applied unchanged.
* The evaluator is data-blind until both sides are frozen: the candidate is
  four hash-pinned artifact directories loaded verbatim, and no development
  corpus row, sealed row, or novelty row is ever fit, calibrated, or selected
  on.  ``development_data_loaded`` is ``False``: unlike the dev harnesses this
  evaluator never opens the development corpus at all.

THE CANDIDATE is an explicitly decoupled, hash-bound system.  The release
intent binds one aggregate validation-evidence JSON file, not a runtime-bundle
manifest.  That evidence file binds the pre-validation candidate contract,
the untouched validation report, both fusion artifacts, the validated staged
policy bytes, and the canonical stage-1 prefilter hashes:

1. ``--candidate-manifest``      the final release-candidate JSON, which
                                 binds the post-validation evidence, all
                                 Python/browser assets and dual binding.
2. ``--classifier-bundle-dir``   the self-verified runtime bundle exported
                                 from the regularized 8k classifier fusion.
3. ``--rejector-bundle-dir``     the self-verified runtime bundle exported
                                 from the frozen 4k rejector fusion.
4. ``--classifier-fusion-dir``   the regularized 8k fusion used only for
                                 known-class labels and closed/five-shot/
                                 clean/length/scale metrics.
5. ``--rejector-fusion-dir``     the frozen 4k fusion used only for the
                                 known/unknown decision and its open metrics.
6. ``--staged-dir``              the exact validation-locked 4k stage-2 and
                                 composite-policy artifacts.
7. ``--prefilter-dir``           the exact validation-locked stage-1 set.

The two fusions are intentionally different complete models.  Their weights,
centers, feature moments and prototypes are loaded independently.  The
evaluator never assumes that any of those arrays are interchangeable.

ARCHITECTURE CONTRACT (stated here and carried in the emitted JSON): the v3
rejector is STAGED and is not additive.  When the stage-1 noise prefilter
fires, the capture is rejected as noise before either fusion runs.  Stage-1
survivors run the 4k fusion and frozen staged policy to decide known/unknown.
Only accepted rows receive the 8k fusion's nearest-prototype known label.
The sealed evaluator computes the 8k additive control for every known row
because the unchanged closed and invariance gates require it, but never uses
that control to make an open-set decision.

Predeclared v3 semantics, all bound into the sealed ``evaluation_protocol``
object the release intent must embed (see ``expected_evaluation_protocol``):

1. **Open-set gates score the staged decision axis, staged policy version 2
   (the composite survivor score).**  Stage-2 scores are enrollment ranks in
   ``[0, 1)``; a stage-1 SURVIVOR is scored by the composite
   ``max(stage2_enrollment_rank, stage1_score_enrollment_rank)``, where the
   stage-1 rank is the enrollment-survivor empirical rank of the stage-1
   log-odds (searchsorted-left, the stage-2 convention); a stage-1-gated row
   is placed at ``1 + softsign(stage-1 log-odds)`` in ``(1, 2)``, above every
   survivor.  The staged unknown threshold is the frozen q95 of the composite
   over enrollment survivors, loaded verbatim from the staged artifact's
   composite policy npz, so ``score > threshold`` is exactly the staged
   decision (asserted row by row by the staged module's own equivalence
   check).  This evaluator REFUSES a staged artifact whose recorded staged
   policy version is not the one this evaluator implements.  The known
   false-unknown gate counts BOTH stages, and the per-length open reports
   attribute every rejected known row to the stage that rejected it.  The
   additive control arm keeps the stage-2 policy's own untouched threshold.
2. **Stage-1 applies on its fitted per-length domain, extended upward by the
   causal-prefix rule.**  The pose-degeneracy features are length dependent,
   so the candidate carries one fitted bundle per capture length and no
   bundle is ever substituted sideways.  At a sealed capture length ABOVE
   the longest fitted length (N32768: the development corpus stores N16384,
   so no N32768 model can exist) the stage-1 features are computed on each
   capture's FIRST max-fitted-length samples and gated with that length's
   bundle -- the first 16384 samples of a 32768-sample capture are exactly a
   16384-sample capture of the same emission, the same causal-prefix rule
   the suite's own length corpora are built on.  A capture length BELOW the
   smallest fitted length keeps the additive pass-through semantics: the
   gate cannot fire and every row flows to the unchanged stage-2 path.
   Both cases are recorded per length (``stage_one_active``,
   ``stage_one_feature_length``), declared in the sealed protocol under
   ``stage_one_prefix_rule``, and NOT silently ignored.  The deployed
   TypeScript runtime must apply the identical prefix rule; a runtime that
   refuses such lengths does not match the sealed claim and must be
   reconciled before any ship.
3. **Closed, five-shot, causal-length and physical-scale gates keep their v2
   definitions** on the additive sub-path (frontend -> encoders -> fusion ->
   nearest prototype over every row).  The v2 rejector never participated in
   those gates, the task contract changes only the known-FUR accounting, and
   the stage-1 causal-prefix rule does not change that: no stage gates any
   sweep row.
4. **Fresh novelty at the release seed.**  The novelty base seed IS the
   release seed (mirroring v2, where both were 20260729), with the identical
   per-family SHA-256 derivation.  Consumed and development seeds are refused.

The unstaged additive score of every scored row is also computed as a control
and the staged survivors are asserted to score identically (the staged
module's subset-agreement check), so the staged short circuit cannot drift
from the additive path it embeds.

This evaluator refuses to run twice: it refuses a release root that already
contains ``RELEASE_EVALUATION.json`` and its exclusive writer refuses an
existing output file.
"""
from __future__ import annotations

import argparse
import ast
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
V2 = HERE.parent
ZPAB = V2.parent
TRAINING = ZPAB.parent
REPO = TRAINING.parent
for _path in (TRAINING, ZPAB, V2, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import assemble_v3_fusion as assemble  # noqa: E402
import evaluate_invariant_release_suite as release  # noqa: E402
import export_v3_openset_browser_assets as openset_export  # noqa: E402
import fit_v3_openset as openset_base  # noqa: E402
import fit_v3_openset_staged as staged  # noqa: E402
import measure_v3_remaining_gates as measure  # noqa: E402
import noise_prefilter  # noqa: E402
import preprocess as production_preprocess  # noqa: E402
import time_domain_invariant_patch_preprocess as td_preprocess  # noqa: E402
from openset_eval import NOVELTY  # noqa: E402
from train import embed_all  # noqa: E402
from v3_time_domain_openset import (  # noqa: E402
    FROZEN_GEOMETRY_FEATURE,
    FROZEN_GEOMETRY_WEIGHT,
    FROZEN_POLICY_KIND,
    FROZEN_POLICY_SCHEMA,
    FROZEN_THRESHOLD_QUANTILE,
    FROZEN_V2_WEIGHT,
)


# ---------------------------------------------------------------------------
# identity with the v2 evaluator: imported objects, never re-typed values
# ---------------------------------------------------------------------------

EVALUATOR_SCHEMA = 3
EVALUATION_VERSION = "time-domain-v3-release-evaluation-v3-dual-fusion"
RELEASE_PROTOCOL = release.RELEASE_PROTOCOL

CANDIDATE_MANIFEST_SCHEMA = "time-domain-v3-dual-release-candidate-v1"
CANDIDATE_EVIDENCE_SCHEMA = "time-domain-v3-decoupled-validation-evidence-v1"
CANDIDATE_CONTRACT_SCHEMA = "time-domain-v3-decoupled-candidate-v1"
CANDIDATE_ID = "v3.3-decoupled-8k-classifier-4k-rejector"
STAGING_PACKAGE_SCHEMA = "atomos.v3.time-domain-classifier.dual-runtime-package"
STAGING_PACKAGE_SCHEMA_VERSION = 1
DUAL_BINDING_SCHEMA = "atomos.v3.time-domain-dual-fusion.binding"
DUAL_BINDING_SCHEMA_VERSION = 1
STAGING_STATUS = "staging_not_release"
BROWSER_FUSION_SCHEMA = (
    "atomos.v3.time-domain-invariant-fusion.browser-weights"
)
BROWSER_FUSION_SCHEMA_VERSION = 1
BROWSER_OPENSET_SCHEMA = "atomos.v3.time-domain-openset.staged"
BROWSER_OPENSET_SCHEMA_VERSION = 2
REJECTOR_BROWSER_ASSET = "time-domain-v3-rejector-weights.json"
CLASSIFIER_BROWSER_ASSET = "time-domain-v3-classifier-weights.json"
OPENSET_BROWSER_ASSET = "time-domain-v3-openset-policy.json"
DUAL_BINDING_ASSET = "time-domain-v3-dual-binding.json"
FUSION_EXPORT_MANIFEST_SCHEMA = (
    "atomos.v3.time-domain-invariant-fusion.browser-weights.export-manifest"
)
FUSION_EXPORT_MANIFEST_SCHEMA_VERSION = 2
OPENSET_EXPORT_MANIFEST_SCHEMA = (
    "time-domain-v3-dual-openset-staging-manifest-v1"
)
OPENSET_EXPORT_MANIFEST_SCHEMA_VERSION = 1
PARITY_SCHEMA = "time-domain-openset-parity-v1"
PARITY_SCHEMA_VERSION = 3

GATE_FLOORS = release.GATE_FLOORS
FIVE_SHOT_K = release.FIVE_SHOT_K
HIGH_SNR_DB = release.HIGH_SNR_DB

# Mirrored as module attributes so a test can rescale the whole protocol
# coherently; for a real run they are exactly the v2 values.
REQUIRED_CAPTURE_LENGTHS = release.REQUIRED_CAPTURE_LENGTHS
MATCHED_CAPTURE_LENGTH = release.MATCHED_CAPTURE_LENGTH
MIN_TARGET_PER_CLASS = release.MIN_TARGET_PER_CLASS
NOVELTY_FAMILIES = release.NOVELTY_FAMILIES
NOVELTY_N_EACH_PER_LENGTH = release.NOVELTY_N_EACH_PER_LENGTH
PHYSICAL_SCALE_FACTORS = release.PHYSICAL_SCALE_FACTORS
SCALE_MIN_PER_CLASS = release.SCALE_MIN_PER_CLASS

_gate = release._gate
_sha256 = release._sha256
_read_json = release._read_json
_is_below = release._is_below
_regular_file = release._regular_file
_integer = release._integer
_verify_file_record = release._verify_file_record
_write_json_exclusive = release._write_json_exclusive
resolve_device = release.resolve_device

#: The one frozen candidate directory set this evaluator was written for.
DEFAULT_REJECTOR_BUNDLE_DIR = (
    V2 / "artifacts" / "invariant_patch" / "v3_scale"
    / "v3_runtime_bundle_rejector4k_dual_seed20260730"
)
DEFAULT_CLASSIFIER_BUNDLE_DIR = (
    V2 / "artifacts" / "invariant_patch" / "v3_scale"
    / "v3_runtime_bundle_classifier8kreg_dual_seed20260730"
)
DEFAULT_CLASSIFIER_FUSION_DIR = (
    V2 / "artifacts" / "invariant_patch" / "v3_scale"
    / "v3_fusion_ml8000reg_seed20260730"
)
DEFAULT_REJECTOR_FUSION_DIR = (
    V2 / "artifacts" / "invariant_patch" / "v3_scale"
    / "v3_fusion_multilength_seed20260730"
)
DEFAULT_STAGED_DIR = (
    V2 / "artifacts" / "invariant_patch" / "v3_scale"
    / "staged_validate_decoupled_rejector4k_classifier8k_budget001_"
    "seeds20260950_20260951"
)
#: The tightened stage-1 gate: the 0.01 enrollment-budget threshold set.
#: Coefficients are bit-identical to the 0.02 set; only the operating points
#: differ (HANDOFF 25's sanctioned path).
DEFAULT_PREFILTER_DIR = (
    V2 / "artifacts" / "invariant_patch" / "v3_scale"
    / "noise_prefilter_fit20261001_budget001" / "bundles"
)
DEFAULT_CANDIDATE_MANIFEST = (
    HERE / "evidence"
    / "v3_dual_release_candidate.json"
)

BUNDLE_KIND = "v3-time-domain-centered-invariant-fusion"
BUNDLE_SCHEMA = "atomos.v3.time-domain-invariant-fusion.runtime-bundle"
BUNDLE_SCHEMA_VERSION = 1
BUNDLE_MANIFEST_NAME = "bundle_manifest.json"
BUNDLE_SELF_VERIFICATION_TOLERANCE = 1e-6

#: Release seeds this evaluator refuses outright.  20260729, 20260731 and
#: 20260733 are the consumed sealed suites; 20260730/20260732 are development MODEL seeds
#: whose reuse as release seeds would collide the two namespaces (HANDOFF 25:
#: use 20260733 for the next release); the two bands are the development
#: novelty namespace and the prefilter fit-only band, which are development
#: evidence, not release seeds.
CONSUMED_RELEASE_SEEDS = {
    20260729: "consumed sealed v2 release suite (evidence rule 2)",
    20260731: (
        "consumed sealed v3.0 release suite (HANDOFF 25: 22/23 gates, known "
        "false-unknown failure frozen; evidence rules 2 and 4)"
    ),
    20260733: (
        "consumed sealed v3.2 composite release suite (HANDOFF 27: 21/23 "
        "gates, known FUR 0.1108 and five-shot 0.8449 failures frozen; "
        "evidence rules 2 and 4)"
    ),
    20260734: (
        "consumed sealed v3.2 release suite under its predeclared historical "
        "gate redeclaration (22/23 gates, five-shot 0.8228 failure frozen; "
        "evidence rules 2 and 4)"
    ),
    20260730: (
        "development model/fusion seed; reusing it as a release seed would "
        "collide the model and release namespaces (HANDOFF 25 note)"
    ),
    20260732: (
        "development model seed; reusing it as a release seed would collide "
        "the model and release namespaces (HANDOFF 25 note: use 20260733)"
    ),
}
REFUSED_RELEASE_SEED_BANDS = (
    (20260900, 20260999, "development novelty seed namespace"),
    (20261000, 20261999, "noise-prefilter fit-only seed band"),
)

#: Historical metadata copied from the consumed seed-20260734 protocol.  It
#: remains here so a future release cannot hide that the prior attempt used
#: relaxed levels.  It is deliberately never applied to current gates.
HISTORICAL_GATE_REDECLARATION_RELEASE_SEED = 20260734
HISTORICAL_V3_GATE_REDECLARATION_RATIONALE = (
    "The 0.10 known-FUR ceiling and 0.85 five-shot floor were declared "
    "against the v2 architecture. Two consumed sealed runs (seeds 20260731, "
    "20260733) establish this candidate's true sealed rates at FUR 0.10-0.11 "
    "and five-shot 0.845-0.850 while it beats the frozen v2 bundle on every "
    "identically-measured axis and fixes all six gates v2 failed. The levels "
    "are re-declared for the v3 architecture by the owner on 2026-07-28, "
    "BEFORE release seed 20260734 was generated. Any release claim must "
    "state these two levels alongside the v2 levels they replace."
)
HISTORICAL_V3_GATE_REDECLARATION = {
    "open_known_false_unknown_worst_length": {
        "v2_level": 0.10,
        "v3_level": 0.12,
        "floor_key": "open_known_false_unknown_max",
        "comparison": "max",
        "owner_decision": HISTORICAL_V3_GATE_REDECLARATION_RATIONALE,
    },
    "five_shot_worst_length_balanced": {
        "v2_level": 0.85,
        "v3_level": 0.84,
        "floor_key": "five_shot",
        "comparison": "min",
        "owner_decision": HISTORICAL_V3_GATE_REDECLARATION_RATIONALE,
    },
}


def historical_gate_redeclaration_block() -> dict[str, Any]:
    """Return seed 20260734's exact, explicitly inactive redeclaration record.

    Each entry is cross-checked against the imported v2 floor it replaced.
    The wrapper makes its historical scope and inactive status machine
    readable; current gates always use :data:`GATE_FLOORS` unchanged.
    """
    gates: dict[str, Any] = {}
    for gate_name, record in HISTORICAL_V3_GATE_REDECLARATION.items():
        v2_floor = float(GATE_FLOORS[record["floor_key"]])
        if v2_floor != float(record["v2_level"]):
            raise ValueError(
                f"historical gate redeclaration for {gate_name!r} records "
                "v2_level "
                f"{record['v2_level']}, but the imported v2 floor "
                f"{record['floor_key']!r} is {v2_floor}; the redeclaration no "
                "longer describes the levels it replaces"
            )
        gates[gate_name] = {
            "v2_level": float(record["v2_level"]),
            "v3_level": float(record["v3_level"]),
            "owner_decision": record["owner_decision"],
        }
    return {
        "release_seed": HISTORICAL_GATE_REDECLARATION_RELEASE_SEED,
        "active_for_current_protocol": False,
        "current_protocol_gate_source": "imported_v2_gate_floors_unchanged",
        "gates_redeclared": gates,
    }

#: Single source of truth for the causal-prefix rule text: the staged module.
STAGE_ONE_PREFIX_RULE = staged.STAGE_ONE_PREFIX_RULE

STAGE_ONE_UNCOVERED_RULE = (
    "at a capture length ABOVE the longest fitted stage-1 length the gate "
    "applies through the causal-prefix rule (stage_one_prefix_rule): features "
    "are computed on the capture's first max-fitted-length samples and gated "
    "with that length's bundle, recorded as stage_one_active: true with the "
    "effective stage_one_feature_length; at a capture length BELOW the "
    "smallest fitted length the gate cannot fire and every row flows to the "
    "unchanged additive stage-2 path, recorded as stage_one_active: false"
)
SWEEP_REJECTOR_RULE = (
    "the causal-length and physical-scale sweeps measure the closed-set "
    "representation exactly as the v2 evaluator defined them; no stage gates "
    "their rows (rejection never participates in the closed and invariance "
    "gates, and the stage-1 causal-prefix rule does not change that)"
)
KNOWN_FUR_ACCOUNTING = (
    "counts both stages: a known row gated by the stage-1 noise prefilter and "
    "a known survivor whose COMPOSITE score exceeds the frozen staged "
    "threshold (q95 of the composite over enrollment survivors) are both "
    "false-unknown, and every per-length open report attributes each rejected "
    "known row to the stage that rejected it. The additive control arm uses "
    "the stage-2 policy's own untouched threshold"
)

#: Source files whose current bytes MUST equal the hash the frozen artifacts
#: recorded.  Everything here is executed by this evaluator's inference path.
#: ``run_time_domain_dev.py`` is deliberately absent: it is the training-loop
#: script, it is imported but never executed for data here, and it legitimately
#: drifted after the candidate was frozen (HANDOFF 24.1 selection-policy work);
#: its recorded and current hashes are reported instead of enforced.
STAGED_SOURCE_HARD_CONTRACT = (
    "assemble_v3_fusion.py",
    "fit_v3_openset.py",
    "fit_v3_openset_staged.py",
    "invariant_fusion.py",
    "invariant_patch_data.py",
    "known_only_patch_openset.py",
    "noise_prefilter.py",
    "openset_eval.py",
    "pose_degeneracy.py",
    "time_domain_geometry.py",
    "time_domain_invariant_patch_preprocess.py",
    "train.py",
    "v3_time_domain_openset.py",
)
STAGED_SOURCE_RECORDED_ONLY = ("run_time_domain_dev.py",)
BUNDLE_ASSEMBLY_SOURCE_HARD_CONTRACT = (
    "assemble_v3_fusion.py",
    "invariant_fusion.py",
    "invariant_patch_cnn.py",
    "invariant_patch_data.py",
    "invariant_patch_preprocess.py",
    "time_domain_geometry.py",
    "time_domain_invariant_patch_preprocess.py",
    "train.py",
)

# The validation process necessarily moved its two novelty seeds from
# "reserved and clean" to "spent" immediately after their one allowed draw.
# That bookkeeping-only edit landed in the same commit as the immutable
# validation artifacts, so the report correctly records the source bytes that
# actually scored the validation rows while the working tree correctly refuses
# to draw those rows again.  Admit only this exact old/new byte pair, and only
# after proving that the complete Python AST is identical when the three
# top-level seed-ledger assignments are removed.  Any inference, fitting,
# threshold, feature, or policy edit changes the normalized digest and fails
# closed.
POST_VALIDATION_LEDGER_TRANSITION = {
    "source": "fit_v3_openset_staged.py",
    "origin": "the staged validation artifact",
    "validated_sha256": (
        "6d6d252802029cd57cab1429640d0b285b117caa97a89b994037469e267aa176"
    ),
    "current_sha256": (
        "1ebfecede89eb3d393c8922d0862827efda6356671995656594b536ed19f7514"
    ),
    "normalized_ast_sha256": (
        "86dbb18d8b83845e4f9c067f672b8f69c9fbbe2b9ccf4216755f7eeb8096e967"
    ),
    "excluded_top_level_assignments": (
        "SPENT_NOVELTY_SEEDS",
        "FIRST_CLEAN_NOVELTY_SEED",
        "SEED_LEDGER_NOTE",
    ),
    "validated_parent_commit": (
        "6f6e1e05d94d457e16940ad1d2c14c6d95bbc422"
    ),
    "ledger_commit": "5ebbd07f763470ff1fc27be04e2e340c6171cc63",
    "full_index_diff_sha256": (
        "c071726b44a04d535084b3c10e76eda2c3b61767a542a29173873b95f8466d03"
    ),
    "consumed_validation_seeds": (20260950, 20260951),
    "next_clean_novelty_seed": 20260952,
}

_SOURCE_LOOKUP = {
    "assemble_v3_fusion.py": HERE / "assemble_v3_fusion.py",
    "fit_v3_openset.py": HERE / "fit_v3_openset.py",
    "fit_v3_openset_staged.py": HERE / "fit_v3_openset_staged.py",
    "noise_prefilter.py": HERE / "noise_prefilter.py",
    "pose_degeneracy.py": HERE / "pose_degeneracy.py",
    "run_time_domain_dev.py": HERE / "run_time_domain_dev.py",
    "export_v3_openset_browser_assets.py": (
        HERE / "export_v3_openset_browser_assets.py"
    ),
    "measure_v3_remaining_gates.py": HERE / "measure_v3_remaining_gates.py",
    "invariant_fusion.py": V2 / "invariant_fusion.py",
    "invariant_patch_cnn.py": V2 / "invariant_patch_cnn.py",
    "invariant_patch_data.py": V2 / "invariant_patch_data.py",
    "known_only_patch_openset.py": V2 / "known_only_patch_openset.py",
    "openset_eval.py": V2 / "openset_eval.py",
    "v3_time_domain_openset.py": V2 / "v3_time_domain_openset.py",
    "run_invariant_cnn_dev.py": V2 / "run_invariant_cnn_dev.py",
    "evaluate_invariant_release_suite.py": (
        V2 / "evaluate_invariant_release_suite.py"
    ),
    "invariant_patch_preprocess.py": TRAINING / "invariant_patch_preprocess.py",
    "preprocess.py": TRAINING / "preprocess.py",
    "time_domain_geometry.py": TRAINING / "time_domain_geometry.py",
    "time_domain_invariant_patch_preprocess.py": (
        TRAINING / "time_domain_invariant_patch_preprocess.py"
    ),
    "train.py": TRAINING / "train.py",
}


def _source_path(name: str) -> Path:
    try:
        return _SOURCE_LOOKUP[name]
    except KeyError as exc:
        raise ValueError(f"unknown evaluator source file {name!r}") from exc


def _read_candidate_json(path: Path) -> dict[str, Any]:
    """Read a frozen candidate-side JSON artifact.

    Deliberately not ``release._read_json``: that validator enforces the sealed
    release-root JSON contract, which rejects integers beyond 2**53 - 1, and
    development artifacts legitimately carry nanosecond mtimes inside their
    recorded cache contracts.  Candidate files are still bound by SHA-256.
    """
    _regular_file(path, name=str(path.name))
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


# ---------------------------------------------------------------------------
# the predeclared v3 protocol object
# ---------------------------------------------------------------------------


def validate_release_seed(value: Any) -> int:
    seed = _integer(value, "release_seed")
    if seed in CONSUMED_RELEASE_SEEDS:
        raise ValueError(
            f"release seed {seed} is refused: {CONSUMED_RELEASE_SEEDS[seed]}"
        )
    for low, high, reason in REFUSED_RELEASE_SEED_BANDS:
        if low <= seed <= high:
            raise ValueError(
                f"release seed {seed} lies in the {reason} ({low}-{high}) and "
                "is not an untouched release seed"
            )
    return seed


def expected_evaluation_protocol(
    release_seed: int,
    stage_one_lengths: Sequence[int],
) -> dict[str, Any]:
    """The exact object the release intent/manifest must embed for this run.

    Built from the v2 protocol object so every unchanged rule is byte-identical
    to v2's, then updated only where the v3 candidate changed the semantics.
    ``gates`` is the imported v2 ``GATE_FLOORS`` dict unchanged, so no floor
    can drift between the intent, launcher and evaluator.  The consumed
    seed-20260734 redeclaration travels separately as explicitly inactive
    historical metadata.
    """
    seed = validate_release_seed(release_seed)
    domain = tuple(sorted(int(length) for length in stage_one_lengths))
    if not domain or len(set(domain)) != len(domain):
        raise ValueError("stage-1 lengths must be a non-empty set of ints")
    max_fitted = int(max(domain))
    prefix_gated = [
        int(length)
        for length in REQUIRED_CAPTURE_LENGTHS
        if int(length) not in domain and int(length) > max_fitted
    ]
    uncovered = [
        int(length)
        for length in REQUIRED_CAPTURE_LENGTHS
        if int(length) not in domain and int(length) < max_fitted
    ]
    protocol = copy.deepcopy(release.EXPECTED_EVALUATION_PROTOCOL)
    protocol["version"] = EVALUATION_VERSION
    protocol["matched_capture_length"] = int(MATCHED_CAPTURE_LENGTH)
    protocol["required_capture_lengths"] = [
        int(length) for length in REQUIRED_CAPTURE_LENGTHS
    ]
    protocol["minimum_target_per_class"] = int(MIN_TARGET_PER_CLASS)
    protocol["physical_scale_factors"] = [
        float(factor) for factor in PHYSICAL_SCALE_FACTORS
    ]
    protocol["physical_scale_support"] = dict(
        protocol["physical_scale_support"]
    )
    protocol["physical_scale_support"]["minimum_rows_per_class"] = int(
        SCALE_MIN_PER_CLASS
    )
    protocol["novelty"] = dict(protocol["novelty"])
    protocol["novelty"]["seed"] = seed
    protocol["novelty"]["seed_rule"] = (
        "the novelty base seed equals the untouched release seed; consumed "
        "release seeds and development novelty/fitting seed bands are refused"
    )
    protocol["novelty"]["n_each_per_length"] = int(NOVELTY_N_EACH_PER_LENGTH)
    protocol["novelty"]["calibration"] = (
        "none; frozen per-length stage-1 noise-prefilter bundles, the frozen "
        "stage-2 branch-LOF ensemble and geometry blend, and the frozen "
        "composite survivor policy (enrollment-survivor stage-1 rank "
        "calibration and its q95 composite threshold) are loaded verbatim"
    )
    protocol["open_set"] = {
        "architecture": staged.STAGED_POLICY_KIND,
        "candidate_architecture": "dual_fusion_classifier8k_rejector4k",
        "classifier_role": (
            "the regularized 8k fusion supplies every accepted known-class "
            "label and every closed/five-shot/clean/length/scale gate input"
        ),
        "rejector_role": (
            "the frozen 4k fusion supplies every stage-2 known/unknown score "
            "and every open-set gate input; it never supplies the final "
            "accepted known-class label"
        ),
        "decision_order": [
            "stage-one causal-prefix noise gate",
            "4k rejector fusion and frozen staged known/unknown policy",
            "8k classifier fusion nearest-prototype known label for accepted rows",
        ],
        "intentional_dual_fusion": True,
        "staged_policy_schema": int(staged.STAGED_POLICY_SCHEMA),
        "staged_policy_version": staged.STAGED_POLICY_VERSION,
        "additive_only": False,
        "changes_closed_label": True,
        "gates_before_classification": True,
        "survivor_score": staged.COMPOSITE_SURVIVOR_SCORE,
        "unknown_threshold_rule": (
            "the staged unknown threshold is the frozen "
            f"q{float(staged.COMPOSITE_THRESHOLD_QUANTILE)} of the composite "
            "over enrollment stage-1 survivors, fit on enrollment only and "
            "loaded verbatim from the staged artifact; the additive control "
            "arm keeps the stage-2 policy's own untouched enrollment "
            "threshold"
        ),
        "stage_one_capture_lengths": [int(length) for length in domain],
        "stage_one_max_fitted_length": max_fitted,
        "stage_one_prefix_gated_lengths": prefix_gated,
        "stage_one_prefix_rule": {
            "max_fitted_length": max_fitted,
            "rule": STAGE_ONE_PREFIX_RULE,
        },
        "stage_one_uncovered_lengths": uncovered,
        "stage_one_uncovered_rule": STAGE_ONE_UNCOVERED_RULE,
        "sweep_rejector_rule": SWEEP_REJECTOR_RULE,
        "known_false_unknown_accounting": KNOWN_FUR_ACCOUNTING,
        "score_axis": staged.STAGED_SCORE_NOTE,
    }
    # The strict imported v2 floors are the current acceptance contract.
    # Seed 20260734's relaxed levels remain visible only as explicitly
    # inactive historical metadata and never mutate this object.
    protocol["gates"] = dict(GATE_FLOORS)
    protocol["historical_gate_redeclaration"] = (
        historical_gate_redeclaration_block()
    )
    return protocol


# ---------------------------------------------------------------------------
# the frozen candidate
# ---------------------------------------------------------------------------


@dataclass
class V3Candidate:
    candidate_manifest_path: Path
    candidate_manifest_sha256: str
    candidate_manifest: dict[str, Any]
    validation_evidence_path: Path
    validation_evidence_sha256: str
    validation_evidence: dict[str, Any]
    frozen_contract_path: Path
    frozen_contract_sha256: str
    frozen_contract: dict[str, Any]
    classifier_bundle_dir: Path
    classifier_bundle_manifest_path: Path
    classifier_bundle_manifest_sha256: str
    classifier_bundle_manifest: dict[str, Any]
    classifier_bundle_arrays: dict[str, np.ndarray]
    rejector_bundle_dir: Path
    rejector_bundle_manifest_path: Path
    rejector_bundle_manifest_sha256: str
    rejector_bundle_manifest: dict[str, Any]
    rejector_bundle_arrays: dict[str, np.ndarray]
    classifier_fusion_dir: Path
    classifier_fusion_artifact: Any  # openset_base.FusionArtifact
    classifier_fusion_module: torch.nn.Module  # CenteredInvariantFusion
    rejector_fusion_dir: Path
    rejector_fusion_artifact: Any  # openset_base.FusionArtifact
    rejector_fusion_module: torch.nn.Module  # CenteredInvariantFusion
    staged_dir: Path
    staged_metrics_path: Path
    staged_metrics_sha256: str
    staged_metrics: dict[str, Any]
    staged_hashes: dict[str, str]
    prefilter_dir: Path
    prefilter_set_sha256: str
    stage_one: Any  # staged.StageOneGate
    stage_one_lengths: tuple[int, ...]
    posedegen: Any
    prefilter_module: Any
    rejector: Any  # openset_base.Rejector
    composite: Any  # staged.CompositeSurvivorPolicy
    classes: tuple[str, ...]
    patch_length: int
    patch_count: int
    target_frac: float
    classifier_feature_mean: np.ndarray
    classifier_feature_std: np.ndarray
    rejector_feature_mean: np.ndarray
    rejector_feature_std: np.ndarray
    #: The STAGED decision threshold: the composite policy's frozen q95.
    threshold: float
    #: The stage-2 policy's own untouched threshold (the additive control).
    stage_two_threshold: float
    source_report: dict[str, Any]

    # Compatibility aliases are deliberately read-only and point to the
    # rejector role.  New callers must use the explicit role-named fields;
    # the CLI and report schemas no longer accept the old single-fusion
    # contract.
    @property
    def fusion_dir(self) -> Path:
        return self.rejector_fusion_dir

    @property
    def fusion_artifact(self) -> Any:
        return self.rejector_fusion_artifact

    @property
    def fusion_module(self) -> torch.nn.Module:
        return self.rejector_fusion_module

    @property
    def feature_mean(self) -> np.ndarray:
        return self.rejector_feature_mean

    @property
    def feature_std(self) -> np.ndarray:
        return self.rejector_feature_std

    @property
    def bundle_dir(self) -> Path:
        return self.rejector_bundle_dir

    @property
    def bundle_manifest_path(self) -> Path:
        return self.rejector_bundle_manifest_path

    @property
    def bundle_manifest_sha256(self) -> str:
        return self.rejector_bundle_manifest_sha256

    @property
    def bundle_manifest(self) -> dict[str, Any]:
        return self.rejector_bundle_manifest

    @property
    def bundle_arrays(self) -> dict[str, np.ndarray]:
        return self.rejector_bundle_arrays


def _verify_source_contract(
    recorded: Mapping[str, Any],
    hard_names: Sequence[str],
    *,
    origin: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compare recorded source hashes against current bytes; fail on drift."""
    checked: dict[str, Any] = {}
    transitions: dict[str, Any] = {}
    for name in hard_names:
        expected = recorded.get(name)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"{origin} records no usable hash for {name!r}")
        path = _source_path(name)
        current = _sha256(path)
        if current != expected:
            transition = _admit_post_validation_ledger_transition(
                name=name,
                origin=origin,
                expected=expected,
                current=current,
                path=path,
            )
            if transition is None:
                raise ValueError(
                    f"source drift: {name} is {current}, but {origin} froze "
                    f"{expected}; the loaded module is not the one the candidate "
                    "was validated with"
                )
            transitions[name] = transition
        checked[name] = current
    return checked, transitions


def _ledger_neutral_ast_sha256(
    path: Path, excluded_assignments: Sequence[str]
) -> str:
    """Hash the source AST after removing only named top-level assignments."""
    excluded = set(excluded_assignments)
    if not excluded or len(excluded) != len(tuple(excluded_assignments)):
        raise ValueError("ledger transition exclusion names are invalid")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    kept: list[ast.stmt] = []
    removed: list[str] = []
    for node in tree.body:
        targets: Sequence[ast.expr] = ()
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = (node.target,)
        names = [
            target.id for target in targets if isinstance(target, ast.Name)
        ]
        if names and set(names).issubset(excluded):
            removed.extend(names)
        else:
            kept.append(node)
    if sorted(removed) != sorted(excluded):
        raise ValueError(
            "ledger transition source does not contain exactly the declared "
            "top-level assignments"
        )
    tree.body = kept
    payload = ast.dump(
        tree, annotate_fields=True, include_attributes=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _admit_post_validation_ledger_transition(
    *,
    name: str,
    origin: str,
    expected: str,
    current: str,
    path: Path,
) -> dict[str, Any] | None:
    """Admit the one exact seed-ledger-only transition after validation."""
    transition = POST_VALIDATION_LEDGER_TRANSITION
    if (
        name != transition["source"]
        or origin != transition["origin"]
        or expected != transition["validated_sha256"]
        or current != transition["current_sha256"]
    ):
        return None
    excluded = transition["excluded_top_level_assignments"]
    normalized = _ledger_neutral_ast_sha256(path, excluded)
    if normalized != transition["normalized_ast_sha256"]:
        raise ValueError(
            "post-validation ledger transition changed executable source "
            "outside the three declared top-level seed-ledger assignments"
        )
    consumed = tuple(transition["consumed_validation_seeds"])
    if (
        tuple(staged.DEFAULT_VALIDATION_NOVELTY_SEEDS) != consumed
        or any(seed not in staged.SPENT_NOVELTY_SEEDS for seed in consumed)
        or staged.FIRST_CLEAN_NOVELTY_SEED
        != transition["next_clean_novelty_seed"]
    ):
        raise ValueError(
            "post-validation ledger transition does not mark the validation "
            "seeds spent and advance the clean-seed floor"
        )
    return {
        "admission": "exact_post_validation_seed_ledger_transition",
        "validated_sha256": transition["validated_sha256"],
        "current_sha256": transition["current_sha256"],
        "normalized_ast_sha256": normalized,
        "excluded_top_level_assignments": list(excluded),
        "validated_parent_commit": transition["validated_parent_commit"],
        "ledger_commit": transition["ledger_commit"],
        "full_index_diff_sha256": transition["full_index_diff_sha256"],
        "consumed_validation_seeds": list(consumed),
        "next_clean_novelty_seed": transition["next_clean_novelty_seed"],
        "candidate_inference_behavior_changed": False,
    }


def _load_bundle(bundle_dir: Path) -> dict[str, Any]:
    """Load and strictly validate the v3 runtime bundle."""
    bundle = assemble.reject_sealed_path(Path(bundle_dir), "runtime bundle")
    manifest_path = bundle / BUNDLE_MANIFEST_NAME
    manifest = _read_candidate_json(manifest_path)
    if (
        manifest.get("kind") != BUNDLE_KIND
        or manifest.get("schema") != BUNDLE_SCHEMA
        or manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION
    ):
        raise ValueError(f"{bundle} is not a v3 runtime bundle")
    for key, expected in (
        ("development_only", True),
        ("release_evidence", False),
        ("sealed_release_data_used", 0),
        ("consumed_test_rows_used", 0),
    ):
        if manifest.get(key) != expected:
            raise ValueError(
                f"{bundle} records {key}={manifest.get(key)!r}, expected "
                f"{expected!r}"
            )
    assets = manifest.get("assets")
    if not isinstance(assets, Mapping) or not assets:
        raise ValueError(f"{bundle} manifest has no asset records")
    arrays: dict[str, np.ndarray] = {}
    for name, record in assets.items():
        path = bundle / str(name)
        _verify_file_record(path, record, name=f"bundle/{name}")
        if str(name).endswith(".npy"):
            arrays[str(name)] = np.load(path)
    verification = manifest.get("self_verification")
    if (
        not isinstance(verification, Mapping)
        or verification.get("closed_label_agreement") is not True
        or not isinstance(verification.get("worst_max_abs_error"), (int, float))
        or float(verification["worst_max_abs_error"])
        > BUNDLE_SELF_VERIFICATION_TOLERANCE
    ):
        raise ValueError(f"{bundle} does not carry a passing self-verification")
    return {
        "directory": bundle,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "manifest_sha256": _sha256(manifest_path),
        "arrays": arrays,
    }


def _resolve_frozen_path(value: Any, *, field: str) -> Path:
    """Resolve an absolute or repository-relative frozen-artifact path."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO / path
    return path.resolve()


def _require_sha256(value: Any, *, field: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{field} is not a lowercase SHA-256 digest")
    return digest


def _verify_json_binding(
    record: Mapping[str, Any],
    *,
    field: str,
    expected_path: Path | None = None,
    expected_schema: str | None = None,
    expected_status: str | None = None,
) -> tuple[Path, str, dict[str, Any]]:
    path = _resolve_frozen_path(record.get("path"), field=f"{field}.path")
    if expected_path is not None and path != Path(expected_path).resolve():
        raise ValueError(
            f"{field}.path is {path}, expected the supplied frozen path "
            f"{Path(expected_path).resolve()}"
        )
    digest = _require_sha256(record.get("sha256"), field=f"{field}.sha256")
    if _sha256(path) != digest:
        raise ValueError(f"{field} SHA-256 does not match its current bytes")
    payload = _read_candidate_json(path)
    schema = record.get("schema")
    status = record.get("status")
    if expected_schema is not None and payload.get("schema") != expected_schema:
        raise ValueError(
            f"{field} JSON schema={payload.get('schema')!r}, expected "
            f"{expected_schema!r}"
        )
    if expected_status is not None and payload.get("status") != expected_status:
        raise ValueError(
            f"{field} JSON status={payload.get('status')!r}, expected "
            f"{expected_status!r}"
        )
    if schema is not None and payload.get("schema") != schema:
        raise ValueError(f"{field} schema differs from the referenced JSON")
    if status is not None and payload.get("status") != status:
        raise ValueError(f"{field} status differs from the referenced JSON")
    return path, digest, payload


def _package_record_path(
    value: Any,
    package_path: Path,
    *,
    field: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value).expanduser()
    if path.is_absolute():
        raise ValueError(f"{field} must be package-manifest-parent-relative")
    return package_path.parent / path


def _verify_package_file_record(
    record: Mapping[str, Any],
    package_path: Path,
    *,
    field: str,
) -> Path:
    path = _package_record_path(record.get("path"), package_path, field=field)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{field} must reference a regular non-symlink file")
    digest = _require_sha256(record.get("sha256"), field=f"{field}.sha256")
    size = record.get("bytes")
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
        or path.stat().st_size != size
    ):
        raise ValueError(f"{field} byte count differs from disk")
    if _sha256(path) != digest:
        raise ValueError(f"{field} SHA-256 differs from disk")
    return path.resolve()


def _validate_staging_package(
    *,
    package_path: Path,
    package_payload: Mapping[str, Any],
    verified_browser: Mapping[str, Mapping[str, Any]],
    binding_path: Path,
    binding_sha256: str,
    binding_payload: Mapping[str, Any],
    classifier_artifact: Any,
    rejector_artifact: Any,
    classifier_bundle: Mapping[str, Any],
    rejector_bundle: Mapping[str, Any],
    staged_report_sha256: str,
    staged_hashes: Mapping[str, str],
) -> None:
    expected_top = {
        "schema",
        "schema_version",
        "status",
        "candidate_id",
        "architecture",
        "assets",
        "roles",
        "openset_policy",
        "dual_binding",
        "external_evidence",
        "size_contract",
    }
    if set(package_payload) != expected_top:
        raise ValueError(
            "dual runtime package contains missing, extra or legacy fields"
        )
    if package_payload.get("candidate_id") != CANDIDATE_ID:
        raise ValueError("dual runtime package candidate_id differs")
    if package_payload.get("architecture") != {
        "execution_order": [
            "stage_one_noise_gate",
            "rejector_known_unknown",
            "classifier_known_label",
        ],
        "classifier_runs_only_after_rejector_acceptance": True,
        "public_known_label_from_classifier_only": True,
    }:
        raise ValueError("dual runtime package architecture differs")

    asset_names = {
        "classifier": CLASSIFIER_BROWSER_ASSET,
        "rejector": REJECTOR_BROWSER_ASSET,
        "openset_policy": OPENSET_BROWSER_ASSET,
        "dual_binding": DUAL_BINDING_ASSET,
    }
    assets = package_payload.get("assets")
    if not isinstance(assets, Mapping) or set(assets) != set(
        asset_names.values()
    ):
        raise ValueError(
            "dual runtime package deployable asset set is not the exact four "
            "dual-fusion files"
        )
    expected_asset_contract = {
        "classifier": (
            verified_browser["classifier"],
            BROWSER_FUSION_SCHEMA,
            BROWSER_FUSION_SCHEMA_VERSION,
            "accepted_known_classifier",
        ),
        "rejector": (
            verified_browser["rejector"],
            BROWSER_FUSION_SCHEMA,
            BROWSER_FUSION_SCHEMA_VERSION,
            "known_unknown_rejector",
        ),
        "openset_policy": (
            verified_browser["openset_policy"],
            BROWSER_OPENSET_SCHEMA,
            BROWSER_OPENSET_SCHEMA_VERSION,
            None,
        ),
        "dual_binding": (
            {
                "path": binding_path,
                "sha256": binding_sha256,
            },
            DUAL_BINDING_SCHEMA,
            DUAL_BINDING_SCHEMA_VERSION,
            None,
        ),
    }
    for role, (
        candidate_record,
        expected_schema,
        expected_version,
        expected_runtime_role,
    ) in expected_asset_contract.items():
        name = asset_names[role]
        record = assets[name]
        if not isinstance(record, Mapping):
            raise ValueError(f"dual runtime package asset {name} is invalid")
        expected_keys = {
            "path",
            "bytes",
            "sha256",
            "schema",
            "schema_version",
            "status",
        }
        if expected_runtime_role is not None:
            expected_keys.add("runtime_role")
        if set(record) != expected_keys or record.get("path") != name:
            raise ValueError(
                f"dual runtime package asset {name} contains aliases"
            )
        path = _verify_package_file_record(
            record, package_path, field=f"staging_package.assets.{name}"
        )
        if (
            path != Path(candidate_record["path"]).resolve()
            or record.get("sha256") != candidate_record["sha256"]
            or record.get("schema") != expected_schema
            or record.get("schema_version") != expected_version
            or record.get("status") != STAGING_STATUS
        ):
            raise ValueError(
                f"dual runtime package asset {name} differs from candidate"
            )
        if (
            expected_runtime_role is not None
            and record.get("runtime_role") != expected_runtime_role
        ):
            raise ValueError(
                f"dual runtime package asset {name} runtime_role differs"
            )

    roles = package_payload.get("roles")
    if not isinstance(roles, Mapping) or set(roles) != {
        "classifier",
        "rejector",
    }:
        raise ValueError("dual runtime package role set differs")
    for role, artifact, bundle, runtime_role, responsibility in (
        (
            "classifier",
            classifier_artifact,
            classifier_bundle,
            "accepted_known_classifier",
            "accepted_known_label_only",
        ),
        (
            "rejector",
            rejector_artifact,
            rejector_bundle,
            "known_unknown_rejector",
            "known_unknown_only",
        ),
    ):
        role_record = roles[role]
        asset_record = assets[asset_names[role]]
        if (
            not isinstance(role_record, Mapping)
            or set(role_record)
            != {
                "runtime_role",
                "responsibility",
                "asset",
                "source_bundle_manifest_sha256",
                "fusion_directory_sha256",
            }
            or role_record.get("runtime_role") != runtime_role
            or role_record.get("responsibility") != responsibility
            or role_record.get("asset") != asset_record
            or role_record.get("source_bundle_manifest_sha256")
            != bundle["manifest_sha256"]
            or role_record.get("fusion_directory_sha256")
            != artifact.directory_sha256
        ):
            raise ValueError(f"dual runtime package {role} role link differs")

    openset_record = package_payload.get("openset_policy")
    openset_asset = assets[OPENSET_BROWSER_ASSET]
    if (
        not isinstance(openset_record, Mapping)
        or set(openset_record)
        != {
            *openset_asset,
            "fitted_rejector_runtime_bundle_manifest_sha256",
            "staged_validation_report_sha256",
            "staged_artifacts_sha256",
        }
    ):
        raise ValueError("dual runtime package openset_policy is invalid")
    for key, value in openset_asset.items():
        if openset_record.get(key) != value:
            raise ValueError("dual runtime package openset asset link differs")
    if (
        openset_record.get(
            "fitted_rejector_runtime_bundle_manifest_sha256"
        )
        != rejector_bundle["manifest_sha256"]
        or openset_record.get("staged_validation_report_sha256")
        != staged_report_sha256
        or openset_record.get("staged_artifacts_sha256")
        != dict(staged_hashes)
    ):
        raise ValueError("dual runtime package openset evidence link differs")
    dual_binding_record = package_payload.get("dual_binding")
    if (
        not isinstance(dual_binding_record, Mapping)
        or set(dual_binding_record) != set(assets[DUAL_BINDING_ASSET])
        or dual_binding_record != assets[DUAL_BINDING_ASSET]
    ):
        raise ValueError("dual runtime package binding record differs")

    external = package_payload.get("external_evidence")
    expected_external = {
        "rejector_export_manifest",
        "classifier_export_manifest",
        "openset_export_manifest",
        "rejector_probe",
        "classifier_probe",
        "parity",
    }
    if not isinstance(external, Mapping) or set(external) != expected_external:
        raise ValueError("dual runtime package external evidence set differs")
    for name, record in external.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"staging package external_evidence.{name} invalid")
        expected_record_keys = {"path", "bytes", "sha256"}
        if name in {"rejector_export_manifest", "classifier_export_manifest"}:
            expected_record_keys.update(
                {"schema", "schema_version", "runtime_role"}
            )
        elif name == "openset_export_manifest":
            expected_record_keys.update({"schema", "schema_version", "status"})
        elif name == "parity":
            expected_record_keys.update(
                {"schema", "schema_version", "status", "packaged"}
            )
        if set(record) != expected_record_keys:
            raise ValueError(
                f"staging package external_evidence.{name} contains aliases"
            )
        path = _verify_package_file_record(
            record,
            package_path,
            field=f"staging_package.external_evidence.{name}",
        )
        if name in {
            "rejector_export_manifest",
            "classifier_export_manifest",
            "openset_export_manifest",
            "parity",
        }:
            payload = _read_candidate_json(path)
            for key in ("schema", "schema_version"):
                if payload.get(key) != record.get(key):
                    raise ValueError(
                        f"staging package external {name} {key} differs"
                    )
            if "status" in record and payload.get("status") != record.get(
                "status"
            ):
                raise ValueError(
                    f"staging package external {name} status differs"
                )
        if name == "rejector_export_manifest":
            expected = (
                FUSION_EXPORT_MANIFEST_SCHEMA,
                FUSION_EXPORT_MANIFEST_SCHEMA_VERSION,
                "known_unknown_rejector",
            )
            if (
                record.get("schema"),
                record.get("schema_version"),
                record.get("runtime_role"),
            ) != expected:
                raise ValueError("rejector export-manifest contract differs")
        elif name == "classifier_export_manifest":
            expected = (
                FUSION_EXPORT_MANIFEST_SCHEMA,
                FUSION_EXPORT_MANIFEST_SCHEMA_VERSION,
                "accepted_known_classifier",
            )
            if (
                record.get("schema"),
                record.get("schema_version"),
                record.get("runtime_role"),
            ) != expected:
                raise ValueError("classifier export-manifest contract differs")
        elif name == "openset_export_manifest":
            if (
                record.get("schema") != OPENSET_EXPORT_MANIFEST_SCHEMA
                or record.get("schema_version")
                != OPENSET_EXPORT_MANIFEST_SCHEMA_VERSION
                or record.get("status") != STAGING_STATUS
            ):
                raise ValueError("openset export-manifest contract differs")
        elif name == "parity":
            if (
                record.get("schema") != PARITY_SCHEMA
                or record.get("schema_version") != PARITY_SCHEMA_VERSION
                or record.get("status") != STAGING_STATUS
                or record.get("packaged") is not False
            ):
                raise ValueError("parity external-evidence contract differs")
    size_contract = package_payload.get("size_contract")
    maximum = 25 * 1024 * 1024
    if (
        size_contract
        != {
            "maximum_file_bytes_exclusive": maximum,
            "all_deployable_files_below_limit": True,
        }
        or any(int(record["bytes"]) >= maximum for record in assets.values())
    ):
        raise ValueError("dual runtime package size contract differs")
    if binding_payload.get("candidate_id") != package_payload.get("candidate_id"):
        raise ValueError("package and binding candidate_id differ")


def _verify_fusion_binding(
    record: Mapping[str, Any],
    artifact: Any,
    *,
    field: str,
) -> None:
    path = _resolve_frozen_path(record.get("directory"), field=f"{field}.directory")
    if path != artifact.directory:
        raise ValueError(
            f"{field} directory is {path}, not the supplied {artifact.directory}"
        )
    digest = _require_sha256(
        record.get("directory_sha256"), field=f"{field}.directory_sha256"
    )
    if digest != artifact.directory_sha256:
        raise ValueError(f"{field} directory SHA-256 does not match")
    file_sha256 = record.get("file_sha256")
    if not isinstance(file_sha256, Mapping) or dict(file_sha256) != dict(
        artifact.file_sha256
    ):
        raise ValueError(f"{field} file SHA-256 map does not match")


def _verify_bundle_binding(
    record: Mapping[str, Any],
    bundle: Mapping[str, Any],
    *,
    field: str,
) -> None:
    directory = _resolve_frozen_path(
        record.get("directory"), field=f"{field}.directory"
    )
    if directory != bundle["directory"]:
        raise ValueError(
            f"{field} directory is {directory}, not {bundle['directory']}"
        )
    manifest_path = _resolve_frozen_path(
        record.get("manifest_path"), field=f"{field}.manifest_path"
    )
    if manifest_path != bundle["manifest_path"]:
        raise ValueError(f"{field} manifest_path does not match")
    digest = _require_sha256(
        record.get("manifest_sha256"), field=f"{field}.manifest_sha256"
    )
    if digest != bundle["manifest_sha256"]:
        raise ValueError(f"{field} manifest SHA-256 does not match")


def _fusion_geometry_signature(artifact: Any) -> dict[str, Any]:
    """The dimensions that must agree across the two independently fit nets."""
    metrics = artifact.metrics
    architecture = metrics.get("architecture", {})
    source_config = (
        metrics.get("source_index_contract", {})
        .get("contract", {})
        .get("config", {})
    )
    signature: dict[str, Any] = {
        "source_config": {
            key: source_config.get(key)
            for key in ("patch_length", "patch_count", "target_frac")
        },
        "prototype_shape": tuple(int(v) for v in artifact.prototypes.shape),
        "feature_count": int(len(artifact.feature_mean)),
    }
    for branch in ("real", "complex"):
        config = architecture.get(branch, {})
        signature[branch] = {
            key: config.get(key)
            for key in (
                "encoder",
                "patch_length",
                "patch_count",
                "patch_dim",
                "hidden",
                "embed_dim",
                "n_features",
                "set_pool",
            )
        }
    signature["fusion_embed_dim"] = architecture.get("embed_dim")
    return signature


def _validate_candidate_manifest(
    *,
    candidate_manifest_path: Path,
    classifier_bundle: Mapping[str, Any],
    rejector_bundle: Mapping[str, Any],
    classifier_artifact: Any,
    rejector_artifact: Any,
    staged_metrics_path: Path,
    staged_hashes: Mapping[str, str],
    prefilter_path: Path,
    prefilter_set_sha256: str,
    stage_one_lengths: Sequence[int],
) -> dict[str, Any]:
    """Verify the final release-candidate manifest and its complete chain."""
    manifest_path = Path(candidate_manifest_path).expanduser().resolve()
    manifest = _read_candidate_json(manifest_path)
    expected_manifest_keys = {
        "schema",
        "status",
        "candidate_id",
        "validation_evidence",
        "classifier",
        "rejector",
        "staged_validation",
        "stage_one_prefilter",
        "browser_assets",
        "staging_package_manifest",
        "dual_binding",
    }
    if set(manifest) != expected_manifest_keys:
        raise ValueError(
            "candidate manifest contains missing, extra or legacy fields"
        )
    if manifest.get("schema") != CANDIDATE_MANIFEST_SCHEMA:
        raise ValueError(
            "candidate manifest uses an old or unknown schema; the dual-fusion "
            "release evaluator fails closed"
        )
    if manifest.get("status") != "release_candidate_frozen":
        raise ValueError("candidate manifest is not frozen for release")
    if manifest.get("candidate_id") != CANDIDATE_ID:
        raise ValueError("candidate manifest identifies another candidate")

    for role, role_record, bundle, artifact, expected_role in (
        (
            "classifier",
            manifest.get("classifier"),
            classifier_bundle,
            classifier_artifact,
            "known_class_label",
        ),
        (
            "rejector",
            manifest.get("rejector"),
            rejector_bundle,
            rejector_artifact,
            "known_unknown_decision",
        ),
    ):
        if not isinstance(role_record, Mapping):
            raise ValueError(f"candidate manifest has no {role} role")
        if role_record.get("role") != expected_role:
            raise ValueError(
                f"candidate manifest {role}.role must be {expected_role!r}"
            )
        fusion_record = role_record.get("fusion")
        bundle_record = role_record.get("runtime_bundle")
        if not isinstance(fusion_record, Mapping) or not isinstance(
            bundle_record, Mapping
        ):
            raise ValueError(f"candidate manifest {role} bindings are incomplete")
        _verify_fusion_binding(
            fusion_record, artifact, field=f"candidate.{role}.fusion"
        )
        _verify_bundle_binding(
            bundle_record, bundle, field=f"candidate.{role}.runtime_bundle"
        )

    validation_record = manifest.get("validation_evidence")
    if not isinstance(validation_record, Mapping):
        raise ValueError("candidate manifest has no validation_evidence binding")
    evidence_path, evidence_sha256, evidence = _verify_json_binding(
        validation_record,
        field="candidate.validation_evidence",
        expected_schema=CANDIDATE_EVIDENCE_SCHEMA,
    )
    if (
        evidence.get("status") != "development_openset_pass"
        or evidence.get("candidate_id") != CANDIDATE_ID
    ):
        raise ValueError("validation evidence is not the passing dual candidate")

    contract_record = evidence.get("candidate_contract")
    if not isinstance(contract_record, Mapping):
        raise ValueError("validation evidence has no pre-validation contract")
    frozen_path, frozen_sha256, frozen_contract = _verify_json_binding(
        contract_record,
        field="validation_evidence.candidate_contract",
        expected_schema=CANDIDATE_CONTRACT_SCHEMA,
    )
    if (
        frozen_contract.get("status") != "frozen_before_validation"
        or frozen_contract.get("candidate_id") != CANDIDATE_ID
        or contract_record.get("committed_before_validation") is not True
    ):
        raise ValueError("pre-validation candidate contract is not frozen")
    architecture = frozen_contract.get("architecture", {})
    for key, expected in (
        ("intentional_dual_fusion", True),
        ("known_label_source", "classifier_fusion_8k_regularized"),
        ("known_unknown_source", "rejector_fusion_4k_frozen_policy"),
        ("stage_one_short_circuit", True),
    ):
        if architecture.get(key) != expected:
            raise ValueError(
                f"pre-validation architecture.{key}={architecture.get(key)!r}, "
                f"expected {expected!r}"
            )

    for field, artifact in (
        ("classifier_fusion_8k_regularized", classifier_artifact),
        ("rejector_fusion_4k", rejector_artifact),
    ):
        record = frozen_contract.get(field)
        if not isinstance(record, Mapping):
            raise ValueError(f"pre-validation contract has no {field}")
        _verify_fusion_binding(record, artifact, field=f"frozen_contract.{field}")

    validation = evidence.get("validation", {})
    for key, expected in (
        ("role", "validate"),
        ("gates_are_evidence", True),
        ("all_pass", True),
        ("sealed_release_data_used", 0),
        ("consumed_test_rows_used", 0),
        ("release_seed_20260735_used", False),
    ):
        if validation.get(key) != expected:
            raise ValueError(
                f"validation evidence records {key}={validation.get(key)!r}, "
                f"expected {expected!r}"
            )
    report_path = _resolve_frozen_path(
        validation.get("report"), field="validation.report"
    )
    if report_path != staged_metrics_path:
        raise ValueError("validation evidence binds another staged report")
    report_sha256 = _require_sha256(
        validation.get("report_sha256"), field="validation.report_sha256"
    )
    if _sha256(staged_metrics_path) != report_sha256:
        raise ValueError("validation report SHA-256 does not match")

    validated_rejector = evidence.get("validated_rejector", {})
    if (
        validated_rejector.get("fusion_directory_sha256")
        != rejector_artifact.directory_sha256
    ):
        raise ValueError("validation evidence binds another rejector fusion")
    if (
        validated_rejector.get("canonical_prefilter_set_sha256")
        != prefilter_set_sha256
    ):
        raise ValueError("validation evidence binds another prefilter set")
    expected_bundle_hashes = {
        str(int(length)): noise_prefilter.bundle_sha256(
            prefilter_path / f"N{int(length)}"
        )
        for length in stage_one_lengths
    }
    if validated_rejector.get("prefilter_bundle_sha256") != expected_bundle_hashes:
        raise ValueError("validation evidence prefilter bundle hashes differ")

    locked = evidence.get("validation_locked_policy_artifacts", {})
    locked_dir = _resolve_frozen_path(
        locked.get("directory"), field="validation_locked_policy_artifacts.directory"
    )
    if locked_dir != staged_metrics_path.parent:
        raise ValueError("validation evidence binds another staged directory")
    for name, digest in staged_hashes.items():
        if locked.get(name) != digest:
            raise ValueError(
                f"validation evidence does not bind the loaded staged {name}"
            )
    if locked.get("novelty_rows_used_to_fit_rank_or_threshold") != 0:
        raise ValueError("validation novelty was used to fit the staged policy")

    stage_one_contract = frozen_contract.get("stage_one_noise_prefilter", {})
    if _resolve_frozen_path(
        stage_one_contract.get("directory"),
        field="frozen_contract.stage_one_noise_prefilter.directory",
    ) != prefilter_path:
        raise ValueError("pre-validation contract binds another prefilter path")
    clarification = evidence.get("candidate_contract_clarification", {})
    if (
        clarification.get("recorded_value")
        != stage_one_contract.get("directory_sha256")
        or clarification.get("canonical_behavioral_hash")
        != prefilter_set_sha256
        or clarification.get("changes_candidate_behavior") is not False
    ):
        raise ValueError("prefilter hash clarification is missing or inconsistent")

    staged_record = manifest.get("staged_validation")
    if not isinstance(staged_record, Mapping):
        raise ValueError("candidate manifest has no staged_validation binding")
    staged_dir = _resolve_frozen_path(
        staged_record.get("directory"), field="candidate.staged_validation.directory"
    )
    if staged_dir != staged_metrics_path.parent:
        raise ValueError("candidate manifest binds another staged directory")
    staged_report_path = _resolve_frozen_path(
        staged_record.get("report_path"),
        field="candidate.staged_validation.report_path",
    )
    if staged_report_path != staged_metrics_path:
        raise ValueError("candidate manifest binds another staged report path")
    if _require_sha256(
        staged_record.get("report_sha256"),
        field="candidate.staged_validation.report_sha256",
    ) != report_sha256:
        raise ValueError("candidate manifest staged report hash differs")
    if staged_record.get("artifact_sha256") != dict(staged_hashes):
        raise ValueError("candidate manifest staged artifact hashes differ")

    prefilter_record = manifest.get("stage_one_prefilter")
    if not isinstance(prefilter_record, Mapping):
        raise ValueError("candidate manifest has no stage_one_prefilter binding")
    if _resolve_frozen_path(
        prefilter_record.get("directory"),
        field="candidate.stage_one_prefilter.directory",
    ) != prefilter_path:
        raise ValueError("candidate manifest binds another prefilter directory")
    if prefilter_record.get("set_sha256") != prefilter_set_sha256:
        raise ValueError("candidate manifest prefilter set hash differs")
    if prefilter_record.get("bundle_sha256") != expected_bundle_hashes:
        raise ValueError("candidate manifest prefilter bundle hashes differ")

    browser_assets = manifest.get("browser_assets")
    if not isinstance(browser_assets, Mapping) or set(browser_assets) != {
        "classifier",
        "rejector",
        "openset_policy",
    }:
        raise ValueError("candidate manifest browser_assets roles are incomplete")
    verified_browser: dict[str, Any] = {}
    browser_contract = {
        "classifier": (
            BROWSER_FUSION_SCHEMA,
            BROWSER_FUSION_SCHEMA_VERSION,
            "accepted_known_classifier",
        ),
        "rejector": (
            BROWSER_FUSION_SCHEMA,
            BROWSER_FUSION_SCHEMA_VERSION,
            "known_unknown_rejector",
        ),
        "openset_policy": (
            BROWSER_OPENSET_SCHEMA,
            BROWSER_OPENSET_SCHEMA_VERSION,
            None,
        ),
    }
    for role, (
        expected_browser_schema,
        expected_browser_version,
        expected_runtime_role,
    ) in browser_contract.items():
        record = browser_assets[role]
        if (
            not isinstance(record, Mapping)
            or set(record) != {"path", "sha256", "schema", "status"}
        ):
            raise ValueError(f"candidate browser_assets.{role} is invalid")
        if not isinstance(record.get("schema"), str) or not isinstance(
            record.get("status"), str
        ):
            raise ValueError(
                f"candidate browser_assets.{role} must bind schema and status"
            )
        if record.get("status") != STAGING_STATUS:
            raise ValueError(
                f"candidate browser_assets.{role} is not staging_not_release"
            )
        path, digest, payload = _verify_json_binding(
            record, field=f"candidate.browser_assets.{role}"
        )
        if (
            payload.get("schema") != expected_browser_schema
            or payload.get("schema_version") != expected_browser_version
            or payload.get("status") != STAGING_STATUS
        ):
            raise ValueError(
                f"candidate browser_assets.{role} schema/version/status differs"
            )
        if (
            expected_runtime_role is not None
            and payload.get("runtime_role") != expected_runtime_role
        ):
            raise ValueError(
                f"candidate browser_assets.{role} runtime_role differs"
            )
        verified_browser[role] = {
            "path": path,
            "sha256": digest,
            "schema": payload.get("schema"),
            "status": payload.get("status"),
        }
    package_record = manifest.get("staging_package_manifest")
    if (
        not isinstance(package_record, Mapping)
        or set(package_record) != {"path", "sha256", "schema", "status"}
    ):
        raise ValueError("candidate manifest has no staging_package_manifest")
    if not isinstance(package_record.get("schema"), str) or not isinstance(
        package_record.get("status"), str
    ):
        raise ValueError(
            "candidate staging_package_manifest must bind schema and status"
        )
    if (
        package_record.get("schema") != STAGING_PACKAGE_SCHEMA
        or package_record.get("status") != STAGING_STATUS
    ):
        raise ValueError(
            "candidate staging package is not the frozen dual-runtime staging "
            "schema/status"
        )
    package_path, package_sha256, package_payload = _verify_json_binding(
        package_record, field="candidate.staging_package_manifest"
    )
    if (
        package_payload.get("schema_version") != STAGING_PACKAGE_SCHEMA_VERSION
    ):
        raise ValueError("candidate staging package schema_version is not 1")
    binding_record = manifest.get("dual_binding")
    if (
        not isinstance(binding_record, Mapping)
        or set(binding_record) != {"path", "sha256", "schema"}
    ):
        raise ValueError("candidate manifest has no dual_binding")
    if not isinstance(binding_record.get("schema"), str):
        raise ValueError("candidate dual_binding must bind its schema")
    if binding_record.get("schema") != DUAL_BINDING_SCHEMA:
        raise ValueError("candidate dual_binding uses an old or unknown schema")
    binding_path, binding_sha256, binding_payload = _verify_json_binding(
        binding_record, field="candidate.dual_binding"
    )
    if (
        binding_payload.get("schema_version") != DUAL_BINDING_SCHEMA_VERSION
        or binding_payload.get("status") != STAGING_STATUS
    ):
        raise ValueError(
            "candidate dual_binding must be schema version 1 and "
            "staging_not_release"
        )
    expected_binding_keys = {
        "schema",
        "schema_version",
        "status",
        "candidate_id",
        "frontend",
        "execution_order",
        "roles",
        "openset_policy",
        "validation",
        "fail_closed",
    }
    if set(binding_payload) != expected_binding_keys:
        raise ValueError(
            "candidate dual_binding contains missing, extra or legacy fields"
        )
    if (
        binding_payload.get("candidate_id") != CANDIDATE_ID
        or binding_payload.get("execution_order")
        != [
            "stage_one_noise_gate",
            "rejector_known_unknown",
            "classifier_known_label",
        ]
    ):
        raise ValueError("candidate dual_binding identity/execution order differs")
    binding_roles = binding_payload.get("roles")
    if not isinstance(binding_roles, Mapping) or set(binding_roles) != {
        "classifier",
        "rejector",
    }:
        raise ValueError("candidate dual_binding role set differs")
    for role, artifact, bundle, runtime_role, responsibility in (
        (
            "classifier",
            classifier_artifact,
            classifier_bundle,
            "accepted_known_classifier",
            "accepted_known_label_only",
        ),
        (
            "rejector",
            rejector_artifact,
            rejector_bundle,
            "known_unknown_rejector",
            "known_unknown_only",
        ),
    ):
        record = binding_roles[role]
        browser = verified_browser[role]
        if (
            not isinstance(record, Mapping)
            or record.get("asset") != browser["path"].name
            or record.get("asset_sha256") != browser["sha256"]
            or record.get("fusion_directory_sha256")
            != artifact.directory_sha256
            or record.get("runtime_bundle_manifest_sha256")
            != bundle["manifest_sha256"]
            or record.get("runtime_role") != runtime_role
            or record.get("responsibility") != responsibility
        ):
            raise ValueError(f"candidate dual_binding {role} role differs")
    binding_openset = binding_payload.get("openset_policy")
    openset_browser = verified_browser["openset_policy"]
    if (
        not isinstance(binding_openset, Mapping)
        or binding_openset.get("asset") != openset_browser["path"].name
        or binding_openset.get("asset_sha256") != openset_browser["sha256"]
        or binding_openset.get("rejector_asset_sha256")
        != verified_browser["rejector"]["sha256"]
        or binding_openset.get(
            "fitted_rejector_runtime_bundle_manifest_sha256"
        )
        != rejector_bundle["manifest_sha256"]
        or binding_openset.get("staged_validation_report_sha256")
        != report_sha256
        or binding_openset.get("staged_artifacts_sha256")
        != dict(staged_hashes)
    ):
        raise ValueError("candidate dual_binding openset policy differs")
    binding_validation = binding_payload.get("validation")
    if (
        not isinstance(binding_validation, Mapping)
        or binding_validation.get("report_sha256") != report_sha256
        or binding_validation.get("role") != "validate"
        or binding_validation.get("status") != "development_openset_pass"
        or binding_validation.get("novelty_seeds")
        != list(validation.get("novelty_seeds_consumed_once", ()))
    ):
        raise ValueError("candidate dual_binding validation record differs")
    expected_fail_closed = {
        "role_assets_bound_by_sha256": True,
        "distinct_role_assets": True,
        "role_asset_sha256_must_differ": True,
        "classifier_runs_only_after_rejector_acceptance": True,
        "public_known_label_from_classifier_only": True,
    }
    if binding_payload.get("fail_closed") != expected_fail_closed:
        raise ValueError("candidate dual_binding fail_closed contract differs")
    if (
        verified_browser["classifier"]["sha256"]
        == verified_browser["rejector"]["sha256"]
    ):
        raise ValueError("candidate browser role assets may not alias")
    _validate_staging_package(
        package_path=package_path,
        package_payload=package_payload,
        verified_browser=verified_browser,
        binding_path=binding_path,
        binding_sha256=binding_sha256,
        binding_payload=binding_payload,
        classifier_artifact=classifier_artifact,
        rejector_artifact=rejector_artifact,
        classifier_bundle=classifier_bundle,
        rejector_bundle=rejector_bundle,
        staged_report_sha256=report_sha256,
        staged_hashes=staged_hashes,
    )

    if classifier_artifact.directory_sha256 == rejector_artifact.directory_sha256:
        raise ValueError("dual candidate may not alias its two fusion roles")
    if _fusion_geometry_signature(classifier_artifact) != _fusion_geometry_signature(
        rejector_artifact
    ):
        raise ValueError("classifier and rejector frontend geometry differs")

    return {
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256(manifest_path),
        "manifest": manifest,
        "validation_evidence_path": evidence_path,
        "validation_evidence_sha256": evidence_sha256,
        "validation_evidence": evidence,
        "frozen_contract_path": frozen_path,
        "frozen_contract_sha256": frozen_sha256,
        "frozen_contract": frozen_contract,
        "validation_report_sha256": report_sha256,
        "prefilter_bundle_sha256": expected_bundle_hashes,
        "browser_assets": verified_browser,
        "staging_package_manifest": {
            "path": package_path,
            "sha256": package_sha256,
            "schema": package_payload.get("schema"),
            "status": package_payload.get("status"),
        },
        "dual_binding": {
            "path": binding_path,
            "sha256": binding_sha256,
            "schema": binding_payload.get("schema"),
        },
    }


def load_candidate(
    *,
    candidate_manifest_path: Path,
    classifier_bundle_dir: Path,
    rejector_bundle_dir: Path,
    classifier_fusion_dir: Path,
    rejector_fusion_dir: Path,
    staged_dir: Path,
    prefilter_dir: Path,
    device: torch.device,
) -> V3Candidate:
    """Load every frozen component and prove the dual-role binding chain."""
    classifier_bundle = _load_bundle(classifier_bundle_dir)
    rejector_bundle = _load_bundle(rejector_bundle_dir)
    if classifier_bundle["manifest"].get("runtime_role") != (
        "accepted_known_classifier"
    ):
        raise ValueError("classifier bundle runtime_role is not accepted_known_classifier")
    if rejector_bundle["manifest"].get("runtime_role") != (
        "known_unknown_rejector"
    ):
        raise ValueError("rejector bundle runtime_role is not known_unknown_rejector")
    bundle = rejector_bundle
    manifest = bundle["manifest"]

    # --- independent fusion artifacts and each bundle's binding to it ------
    classifier_artifact = openset_base.load_fusion_artifact(
        Path(classifier_fusion_dir)
    )
    rejector_artifact = openset_base.load_fusion_artifact(
        Path(rejector_fusion_dir)
    )
    fusion_artifact = rejector_artifact
    fusion_dir = rejector_fusion_dir
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("bundle manifest has no provenance block")
    recorded_fusion = (REPO / str(provenance.get("source_fusion_artifact"))).resolve()
    if recorded_fusion != fusion_artifact.directory:
        raise ValueError(
            f"bundle was exported from {recorded_fusion}, not from the given "
            f"fusion directory {fusion_artifact.directory}"
        )
    if provenance.get("source_dev_metrics_sha256") != fusion_artifact.file_sha256[
        "dev_metrics.json"
    ]:
        raise ValueError(
            "bundle provenance.source_dev_metrics_sha256 does not match the "
            "fusion artifact's dev_metrics.json bytes"
        )
    for asset_name, fusion_value in (
        ("real_center.npy", fusion_artifact.real_center),
        ("complex_center.npy", fusion_artifact.complex_center),
        ("fusion_prototypes.npy", fusion_artifact.prototypes),
        ("feature_mean.npy", fusion_artifact.feature_mean),
        ("feature_std.npy", fusion_artifact.feature_std),
    ):
        if not np.array_equal(
            np.asarray(bundle["arrays"][asset_name], dtype=np.float32),
            np.asarray(fusion_value, dtype=np.float32),
        ):
            raise ValueError(
                f"runtime bundle {asset_name} differs from the fusion artifact"
            )

    # --- the torch fusion module, rebuilt from the frozen state dict -------
    loaded = measure.load_fusion_artifact(Path(fusion_dir))
    fusion_module = loaded["fusion"].to(device).eval()
    state_name = str(fusion_artifact.metrics["artifacts"]["state_dict"])
    if _sha256(bundle["directory"] / "fusion_state_dict.pt") != (
        fusion_artifact.file_sha256[state_name]
    ):
        raise ValueError(
            "bundle fusion_state_dict.pt differs from the fusion artifact's"
        )
    rejector_fusion_module = fusion_module

    classifier_manifest = classifier_bundle["manifest"]
    classifier_provenance = classifier_manifest.get("provenance")
    if not isinstance(classifier_provenance, Mapping):
        raise ValueError("classifier bundle manifest has no provenance block")
    classifier_recorded_fusion = _resolve_frozen_path(
        classifier_provenance.get("source_fusion_artifact"),
        field="classifier bundle provenance.source_fusion_artifact",
    )
    if classifier_recorded_fusion != classifier_artifact.directory:
        raise ValueError(
            "classifier runtime bundle was exported from another fusion"
        )
    if (
        classifier_provenance.get("source_dev_metrics_sha256")
        != classifier_artifact.file_sha256["dev_metrics.json"]
    ):
        raise ValueError(
            "classifier bundle provenance.source_dev_metrics_sha256 does not "
            "match the classifier fusion artifact"
        )
    for asset_name, fusion_value in (
        ("real_center.npy", classifier_artifact.real_center),
        ("complex_center.npy", classifier_artifact.complex_center),
        ("fusion_prototypes.npy", classifier_artifact.prototypes),
        ("feature_mean.npy", classifier_artifact.feature_mean),
        ("feature_std.npy", classifier_artifact.feature_std),
    ):
        if not np.array_equal(
            np.asarray(classifier_bundle["arrays"][asset_name], dtype=np.float32),
            np.asarray(fusion_value, dtype=np.float32),
        ):
            raise ValueError(
                f"classifier runtime bundle {asset_name} differs from its "
                "fusion artifact"
            )
    classifier_loaded = measure.load_fusion_artifact(
        Path(classifier_fusion_dir)
    )
    classifier_fusion_module = classifier_loaded["fusion"].to(device).eval()
    classifier_state_name = str(
        classifier_artifact.metrics["artifacts"]["state_dict"]
    )
    if _sha256(
        classifier_bundle["directory"] / "fusion_state_dict.pt"
    ) != classifier_artifact.file_sha256[classifier_state_name]:
        raise ValueError(
            "classifier bundle fusion_state_dict.pt differs from its fusion "
            "artifact"
        )

    # --- the frozen staged stage-2 state -----------------------------------
    components, policy, staged_hashes = openset_export.load_stage_two_state(
        Path(staged_dir)
    )
    staged_metrics_path = (
        assemble.reject_sealed_path(Path(staged_dir), "staged artifact")
        / "openset_metrics.json"
    )
    staged_metrics = _read_candidate_json(staged_metrics_path)
    for key, expected in (
        ("status", "development_openset_pass"),
        ("role", "validate"),
        ("gates_are_evidence", True),
        ("all_pass", True),
        ("development_only", True),
        ("release_evidence", False),
        ("sealed_release_data_used", 0),
        ("consumed_test_rows_used", 0),
        ("additive_only", False),
        ("changes_closed_label", True),
        ("gates_before_classification", True),
        ("closed_label_can_be_gated", True),
    ):
        if staged_metrics.get(key) != expected:
            raise ValueError(
                f"staged artifact records {key}={staged_metrics.get(key)!r}, "
                f"expected {expected!r}; only a passing validation run may "
                "supply the frozen stage-2 state"
            )
    # The staged POLICY VERSION must be the one this evaluator implements: a
    # version-1 artifact (survivor score = stage-2 rank alone) must not be
    # scored with composite semantics, nor the reverse.
    architecture = staged_metrics.get("architecture", {})
    for key, expected in (
        ("kind", staged.STAGED_POLICY_KIND),
        ("schema", staged.STAGED_POLICY_SCHEMA),
        ("staged_policy_version", staged.STAGED_POLICY_VERSION),
    ):
        if architecture.get(key) != expected:
            raise ValueError(
                "staged policy version mismatch: the staged artifact records "
                f"architecture.{key}={architecture.get(key)!r}, but this "
                f"evaluator implements {expected!r}.  A staged artifact may "
                "only be evaluated under the policy version that validated it"
            )
    # --- the frozen composite survivor policy -------------------------------
    composite_path = staged_metrics_path.parent / staged.COMPOSITE_POLICY_FILENAME
    if not composite_path.is_file():
        raise ValueError(
            f"staged artifact carries no {staged.COMPOSITE_POLICY_FILENAME}; "
            "a composite-policy (version-2) validation artifact is required"
        )
    composite = staged.load_composite_policy(
        composite_path, expected_stage_two_threshold=policy.threshold
    )
    staged_hashes = dict(staged_hashes)
    staged_hashes[staged.COMPOSITE_POLICY_FILENAME] = _sha256(composite_path)
    if staged_metrics.get("artifacts") != staged_hashes:
        raise ValueError(
            "staged artifact npz bytes differ from the hashes its own report "
            "recorded"
        )
    recorded = staged_metrics.get("fusion", {})
    if recorded.get("directory_sha256") != fusion_artifact.directory_sha256:
        raise ValueError(
            "the staged stage-2 state was fit against a different fusion "
            f"({recorded.get('directory_sha256')!r}) than the candidate "
            f"({fusion_artifact.directory_sha256!r})"
        )
    if float(staged_metrics["stage_two"]["threshold"]) != float(policy.threshold):
        raise ValueError(
            "loaded stage-2 threshold differs from the staged artifact record"
        )
    if float(staged_metrics["composite"]["threshold"]) != float(
        composite.threshold
    ):
        raise ValueError(
            "loaded composite threshold differs from the staged artifact "
            "record"
        )

    # --- the stage-1 prefilter set ------------------------------------------
    prefilter_module = staged.load_prefilter_module()
    posedegen = staged.load_posedegen_module()
    prefilter_path = assemble.reject_sealed_path(
        Path(prefilter_dir), "prefilter set"
    )
    set_sha256 = str(noise_prefilter.prefilter_set_sha256(prefilter_path))
    if staged_metrics.get("stage_one", {}).get("set_sha256") != set_sha256:
        raise ValueError(
            "prefilter set bytes differ from the set the staged validation "
            "run recorded; the stage-1 gate is not the validated one"
        )
    declared_lengths = tuple(
        int(length)
        for length in staged_metrics.get("stage_one", {}).get(
            "capture_lengths", ()
        )
    )
    if not declared_lengths:
        raise ValueError("staged artifact declares no stage-1 capture lengths")
    stage_one = staged.load_stage_one(
        prefilter_module,
        posedegen,
        prefilter_path,
        required_lengths=declared_lengths,
    )
    if tuple(stage_one.lengths) != declared_lengths:
        raise ValueError(
            f"prefilter set covers {list(stage_one.lengths)}, but the staged "
            f"validation declared {list(declared_lengths)}"
        )

    # --- the bundle's rejection contract must describe this exact policy ----
    rejection = manifest.get("rejection", {})
    required_contract = rejection.get("required_contract", {})
    for key, expected in (
        ("policy_schema", FROZEN_POLICY_SCHEMA),
        ("policy_kind", FROZEN_POLICY_KIND),
        ("geometry_feature", FROZEN_GEOMETRY_FEATURE),
        ("geometry_weight", FROZEN_GEOMETRY_WEIGHT),
        ("v2_weight", FROZEN_V2_WEIGHT),
        ("threshold_quantile", FROZEN_THRESHOLD_QUANTILE),
        ("module", "v3_time_domain_openset"),
    ):
        if required_contract.get(key) != expected:
            raise ValueError(
                f"bundle rejection.required_contract[{key!r}] is "
                f"{required_contract.get(key)!r}, expected {expected!r}"
            )

    # --- source drift --------------------------------------------------------
    staged_recorded = staged_metrics.get("source_sha256", {})
    checked, source_transitions = _verify_source_contract(
        staged_recorded,
        STAGED_SOURCE_HARD_CONTRACT,
        origin="the staged validation artifact",
    )
    rejector_checked, rejector_transitions = _verify_source_contract(
        manifest.get("provenance", {}).get("assembly_source_sha256", {}),
        BUNDLE_ASSEMBLY_SOURCE_HARD_CONTRACT,
        origin="the runtime bundle assembly record",
    )
    classifier_checked, classifier_transitions = _verify_source_contract(
        classifier_manifest.get("provenance", {}).get(
            "assembly_source_sha256", {}
        ),
        BUNDLE_ASSEMBLY_SOURCE_HARD_CONTRACT,
        origin="the classifier runtime bundle assembly record",
    )
    checked.update(rejector_checked)
    checked.update(classifier_checked)
    source_transitions.update(rejector_transitions)
    source_transitions.update(classifier_transitions)
    frontend_recorded = manifest.get("frontend", {}).get("source_sha256", {})
    classifier_frontend_recorded = classifier_manifest.get("frontend", {}).get(
        "source_sha256", {}
    )
    if (
        not isinstance(frontend_recorded, Mapping)
        or not isinstance(classifier_frontend_recorded, Mapping)
        or dict(classifier_frontend_recorded) != dict(frontend_recorded)
    ):
        raise ValueError(
            "classifier and rejector runtime bundle frontend source bindings "
            "differ"
        )
    for name, expected in dict(frontend_recorded).items():
        short = str(name).split("/")[-1]
        current = _sha256(_source_path(short))
        if current != expected:
            raise ValueError(f"frontend source drift: {name}")
        checked[short] = current
    recorded_only = {
        name: {
            "recorded_by_staged_artifact": staged_recorded.get(name),
            "current": _sha256(_source_path(name)),
            "enforced": False,
            "reason": (
                "training-loop source; imported but never executed for data "
                "by this evaluator, and legitimately changed after the "
                "candidate was frozen (HANDOFF 24.1)"
            ),
        }
        for name in STAGED_SOURCE_RECORDED_ONLY
    }
    evaluator_chain = {
        name: _sha256(_source_path(name))
        for name in (
            "evaluate_invariant_release_suite.py",
            "export_v3_openset_browser_assets.py",
            "measure_v3_remaining_gates.py",
            "preprocess.py",
            "invariant_patch_preprocess.py",
            "run_invariant_cnn_dev.py",
        )
    }

    # --- geometry and classes -----------------------------------------------
    frontend = manifest["frontend"]
    metadata = td_preprocess.preprocess_metadata()
    for key in ("patch_length", "patch_count", "target_frac"):
        if metadata[key] != frontend[key]:
            raise ValueError(
                f"current frontend {key}={metadata[key]!r} differs from the "
                f"bundle's frozen {frontend[key]!r}"
            )
    if metadata["version"] != frontend["version"]:
        raise ValueError("current frontend version differs from the bundle's")
    classifier_frontend = classifier_manifest.get("frontend", {})
    for key in ("version", "patch_length", "patch_count", "target_frac"):
        if classifier_frontend.get(key) != frontend.get(key):
            raise ValueError(
                "classifier and rejector runtime bundle frontend geometry differs"
            )
    classes = tuple(str(name) for name in manifest["classification"]["classes"])
    if not classes or list(classes) != sorted(classes):
        raise ValueError("bundle classes must be a sorted non-empty list")
    classifier_classes = tuple(
        str(name)
        for name in classifier_manifest.get("classification", {}).get(
            "classes", ()
        )
    )
    if classifier_classes != classes:
        raise ValueError("classifier and rejector runtime bundle classes differ")
    for role, artifact in (
        ("classifier", classifier_artifact),
        ("rejector", rejector_artifact),
    ):
        if artifact.prototypes.shape[0] != len(classes):
            raise ValueError(
                f"{role} fusion prototype count differs from the class contract"
            )

    binding = _validate_candidate_manifest(
        candidate_manifest_path=Path(candidate_manifest_path),
        classifier_bundle=classifier_bundle,
        rejector_bundle=rejector_bundle,
        classifier_artifact=classifier_artifact,
        rejector_artifact=rejector_artifact,
        staged_metrics_path=staged_metrics_path,
        staged_hashes=staged_hashes,
        prefilter_path=prefilter_path,
        prefilter_set_sha256=set_sha256,
        stage_one_lengths=stage_one.lengths,
    )

    rejector = openset_base.Rejector(
        nets={
            "real": rejector_fusion_module.real_branch,
            "complex": rejector_fusion_module.complex_branch,
        },
        real_center=fusion_artifact.real_center,
        complex_center=fusion_artifact.complex_center,
        weight_real=fusion_artifact.weight_real,
        prototypes=fusion_artifact.prototypes,
        lof_components=list(components),
        policy=policy,
    )

    return V3Candidate(
        candidate_manifest_path=binding["manifest_path"],
        candidate_manifest_sha256=binding["manifest_sha256"],
        candidate_manifest=binding["manifest"],
        validation_evidence_path=binding["validation_evidence_path"],
        validation_evidence_sha256=binding["validation_evidence_sha256"],
        validation_evidence=binding["validation_evidence"],
        frozen_contract_path=binding["frozen_contract_path"],
        frozen_contract_sha256=binding["frozen_contract_sha256"],
        frozen_contract=binding["frozen_contract"],
        classifier_bundle_dir=classifier_bundle["directory"],
        classifier_bundle_manifest_path=classifier_bundle["manifest_path"],
        classifier_bundle_manifest_sha256=classifier_bundle["manifest_sha256"],
        classifier_bundle_manifest=classifier_manifest,
        classifier_bundle_arrays=classifier_bundle["arrays"],
        rejector_bundle_dir=bundle["directory"],
        rejector_bundle_manifest_path=bundle["manifest_path"],
        rejector_bundle_manifest_sha256=bundle["manifest_sha256"],
        rejector_bundle_manifest=manifest,
        rejector_bundle_arrays=bundle["arrays"],
        classifier_fusion_dir=classifier_artifact.directory,
        classifier_fusion_artifact=classifier_artifact,
        classifier_fusion_module=classifier_fusion_module,
        rejector_fusion_dir=fusion_artifact.directory,
        rejector_fusion_artifact=fusion_artifact,
        rejector_fusion_module=rejector_fusion_module,
        staged_dir=staged_metrics_path.parent,
        staged_metrics_path=staged_metrics_path,
        staged_metrics_sha256=binding["validation_report_sha256"],
        staged_metrics=staged_metrics,
        staged_hashes=staged_hashes,
        prefilter_dir=prefilter_path,
        prefilter_set_sha256=set_sha256,
        stage_one=stage_one,
        stage_one_lengths=tuple(stage_one.lengths),
        posedegen=posedegen,
        prefilter_module=prefilter_module,
        rejector=rejector,
        composite=composite,
        classes=classes,
        patch_length=int(frontend["patch_length"]),
        patch_count=int(frontend["patch_count"]),
        target_frac=float(frontend["target_frac"]),
        classifier_feature_mean=np.asarray(
            classifier_bundle["arrays"]["feature_mean.npy"], dtype=np.float32
        ),
        classifier_feature_std=np.asarray(
            classifier_bundle["arrays"]["feature_std.npy"], dtype=np.float32
        ),
        rejector_feature_mean=np.asarray(
            bundle["arrays"]["feature_mean.npy"], dtype=np.float32
        ),
        rejector_feature_std=np.asarray(
            bundle["arrays"]["feature_std.npy"], dtype=np.float32
        ),
        threshold=float(composite.threshold),
        stage_two_threshold=float(policy.threshold),
        source_report={
            "enforced": checked,
            "recorded_only": recorded_only,
            "post_validation_ledger_transitions": source_transitions,
            "evaluator_chain": evaluator_chain,
        },
    )


def candidate_provenance(candidate: V3Candidate) -> dict[str, Any]:
    prefilter_bundle_sha256 = {
        str(int(length)): noise_prefilter.bundle_sha256(
            candidate.prefilter_dir / f"N{int(length)}"
        )
        for length in candidate.stage_one_lengths
    }
    browser_assets = {
        role: dict(record)
        for role, record in candidate.candidate_manifest["browser_assets"].items()
    }
    return {
        "candidate_contract": {
            "path": str(candidate.candidate_manifest_path),
            "sha256": candidate.candidate_manifest_sha256,
            "schema": CANDIDATE_MANIFEST_SCHEMA,
            "candidate_id": CANDIDATE_ID,
            "status": candidate.candidate_manifest["status"],
        },
        "validation_evidence": {
            "path": str(candidate.validation_evidence_path),
            "sha256": candidate.validation_evidence_sha256,
            "schema": CANDIDATE_EVIDENCE_SCHEMA,
            "status": candidate.validation_evidence["status"],
        },
        "frozen_prevalidation_contract": {
            "path": str(candidate.frozen_contract_path),
            "sha256": candidate.frozen_contract_sha256,
            "schema": CANDIDATE_CONTRACT_SCHEMA,
            "status": candidate.frozen_contract["status"],
        },
        "classifier_runtime_bundle": {
            "directory": str(candidate.classifier_bundle_dir),
            "manifest_path": str(candidate.classifier_bundle_manifest_path),
            "manifest_sha256": candidate.classifier_bundle_manifest_sha256,
            "schema": BUNDLE_SCHEMA,
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "kind": BUNDLE_KIND,
            "assets": {
                name: dict(record)
                for name, record in candidate.classifier_bundle_manifest[
                    "assets"
                ].items()
            },
        },
        "rejector_runtime_bundle": {
            "directory": str(candidate.rejector_bundle_dir),
            "manifest_path": str(candidate.rejector_bundle_manifest_path),
            "manifest_sha256": candidate.rejector_bundle_manifest_sha256,
            "schema": BUNDLE_SCHEMA,
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "kind": BUNDLE_KIND,
            "assets": {
                name: dict(record)
                for name, record in candidate.rejector_bundle_manifest[
                    "assets"
                ].items()
            },
        },
        "classifier_fusion": {
            "directory": str(candidate.classifier_fusion_dir),
            "directory_sha256": (
                candidate.classifier_fusion_artifact.directory_sha256
            ),
            "file_sha256": dict(
                candidate.classifier_fusion_artifact.file_sha256
            ),
            "seed": int(candidate.classifier_fusion_artifact.seed),
            "role": "known_class_label",
        },
        "rejector_fusion": {
            "directory": str(candidate.rejector_fusion_dir),
            "directory_sha256": candidate.rejector_fusion_artifact.directory_sha256,
            "file_sha256": dict(candidate.rejector_fusion_artifact.file_sha256),
            "seed": int(candidate.rejector_fusion_artifact.seed),
            "role": "known_unknown_decision",
        },
        "staged_validation": {
            "directory": str(candidate.staged_dir),
            "report_path": str(candidate.staged_metrics_path),
            "report_sha256": candidate.staged_metrics_sha256,
            "status": candidate.staged_metrics["status"],
            "novelty_seeds": list(
                candidate.staged_metrics["seeds"]["novelty_seeds"]
            ),
            "artifact_sha256": dict(candidate.staged_hashes),
            "policy_version": staged.STAGED_POLICY_VERSION,
            "staged_threshold": candidate.threshold,
            "stage_two_threshold": candidate.stage_two_threshold,
            "composite": candidate.composite.provenance(),
        },
        "stage_one_prefilter": {
            "directory": str(candidate.prefilter_dir),
            "set_sha256": candidate.prefilter_set_sha256,
            "bundle_sha256": prefilter_bundle_sha256,
            "capture_lengths": [
                int(length) for length in candidate.stage_one_lengths
            ],
        },
        "browser_assets": browser_assets,
        "staging_package_manifest": dict(
            candidate.candidate_manifest["staging_package_manifest"]
        ),
        "dual_binding": dict(candidate.candidate_manifest["dual_binding"]),
        "dual_binding_sha256": candidate.candidate_manifest["dual_binding"][
            "sha256"
        ],
        "source_sha256": candidate.source_report,
    }


# ---------------------------------------------------------------------------
# suite loading: v2 machinery with the v3 protocol and candidate binding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class V3ReleaseSuite:
    root: Path
    intent_path: Path
    manifest_path: Path
    intent: dict[str, Any]
    release_manifest: dict[str, Any]
    candidate_sha256: str
    release_seed: int
    classes: tuple[str, ...]
    corpora: dict[int, release.CorpusSpec]
    support_indices: np.ndarray
    query_indices: np.ndarray
    split_report: dict[str, Any]
    prefix_nesting_report: dict[str, Any]
    dependency_report: dict[str, Any]
    start_probe_report: dict[str, Any]


def load_v3_release_suite(
    release_root: Path,
    candidate: V3Candidate,
) -> V3ReleaseSuite:
    """Load and cryptographically verify a sealed v3 suite before inference.

    This mirrors ``evaluate_invariant_release_suite.load_release_suite`` step
    by step; every sub-verification (file records, corpus geometry, prefix
    nesting, derivation records, start probe, dependency provenance, split) is
    the imported v2 function.  The two v3-specific changes are the expected
    ``evaluation_protocol`` object and the candidate binding: the intent's
    ``candidate_path`` must be the final dual release-candidate manifest,
    whose SHA-256 transitively pins both runtime bundles, both fusion trees,
    the passing validation evidence and exact staged/prefilter/browser bytes.
    """
    root = Path(release_root).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"release root must be a regular directory: {root}")
    if _is_below(root, release.LIVE_CORPUS) or _is_below(
        release.LIVE_CORPUS, root
    ):
        raise ValueError(
            "release root aliases, contains, or is inside the live corpus"
        )
    existing = root / "RELEASE_EVALUATION.json"
    if existing.exists():
        raise FileExistsError(
            f"{existing} already exists; this suite has been evaluated and a "
            "sealed one-shot may not be re-run"
        )

    intent_path = root / "RELEASE_INTENT.json"
    manifest_path = root / "RELEASE_MANIFEST.json"
    intent = _read_json(intent_path)
    release_manifest = _read_json(manifest_path)
    if intent.get("protocol") != RELEASE_PROTOCOL or release_manifest.get(
        "protocol"
    ) != RELEASE_PROTOCOL:
        raise ValueError("release protocol is invalid")
    if intent.get("status") != "in_progress":
        raise ValueError("RELEASE_INTENT must preserve its pre-generation status")
    if release_manifest.get("status") != "complete":
        raise ValueError("release manifest is not complete")
    if intent.get("development_data_used") is not False or release_manifest.get(
        "development_data_used"
    ) is not False:
        raise ValueError("release suite may not declare development-data use")

    release_seed = validate_release_seed(intent.get("release_seed"))
    expected_protocol = expected_evaluation_protocol(
        release_seed, candidate.stage_one_lengths
    )
    expected_historical = expected_protocol["historical_gate_redeclaration"]
    for name, holder in (("intent", intent), ("manifest", release_manifest)):
        embedded = holder.get("evaluation_protocol")
        embedded_historical = (
            embedded.get("historical_gate_redeclaration")
            if isinstance(embedded, Mapping)
            else None
        )
        if embedded_historical != expected_historical:
            raise ValueError(
                f"release {name} evaluation_protocol carries no matching "
                "historical_gate_redeclaration block. The consumed "
                "seed-20260734 0.12/0.84 levels must remain visible as "
                "inactive history, while the current suite is evaluated only "
                "against the strict imported 0.10/0.85 levels"
            )
        if embedded != expected_protocol:
            raise ValueError(
                f"release {name} evaluation_protocol is missing or differs "
                "from the precommitted v3 evaluator contract"
            )

    intent_hash = str(release_manifest.get("release_intent_sha256", "")).lower()
    if intent_hash != _sha256(intent_path):
        raise ValueError(
            "RELEASE_MANIFEST does not bind the exact RELEASE_INTENT bytes"
        )
    for key in (
        "protocol",
        "development_data_used",
        "candidate_path",
        "candidate_sha256",
        "release_seed",
        "capture_lengths",
        "target_per_class",
        "exact_target_per_class",
        "evaluation_protocol",
        "source_sha256",
        "dependency_provenance",
        "runtime_provenance",
        "started_at",
    ):
        if release_manifest.get(key) != intent.get(key):
            raise ValueError(
                f"release manifest changed precommitted intent field {key!r}"
            )

    source_records = intent.get("source_sha256")
    if not isinstance(source_records, Mapping) or set(source_records) != {
        "launcher",
        "corpus_generator",
        "prefix_deriver",
        "tsx_package",
    }:
        raise ValueError("intent source_sha256 is missing")
    expected_source_paths = {
        "launcher": release.LAUNCHER,
        "corpus_generator": release.CORPUS_GENERATOR,
        "prefix_deriver": release.PREFIX_DERIVER,
    }
    for name, expected_path in expected_source_paths.items():
        record = source_records.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"intent source record {name!r} is missing")
        path = Path(str(record.get("path", ""))).expanduser().resolve()
        if path != expected_path:
            raise ValueError(
                f"intent source path {name!r} is not the release source"
            )
        if str(record.get("sha256", "")).lower() != _sha256(path):
            raise ValueError(f"intent source {name!r} changed after generation")
    if source_records.get("tsx_package") != "tsx@4.20.3":
        raise ValueError("release intent did not pin the expected tsx package")
    if release_manifest.get("source_sha256") != source_records:
        raise ValueError("release manifest source records differ from intent")

    dependency_report = release._validate_dependency_provenance(
        intent.get("dependency_provenance"),
        intent.get("runtime_provenance"),
    )
    signal_lab_root = Path(dependency_report["signallab"]["root"])

    # Candidate binding: the intent-bound file is the final dual manifest.
    candidate_path = (
        Path(str(intent.get("candidate_path", ""))).expanduser().resolve()
    )
    if candidate_path != candidate.candidate_manifest_path:
        raise ValueError(
            "RELEASE_INTENT candidate_path is not the frozen dual release "
            f"candidate manifest: {candidate_path} != "
            f"{candidate.candidate_manifest_path}"
        )
    if _is_below(candidate_path, root):
        raise ValueError("candidate must be frozen outside the release output root")
    candidate_hash = str(intent.get("candidate_sha256", "")).lower()
    if candidate_hash != candidate.candidate_manifest_sha256:
        raise ValueError(
            "candidate dual-manifest SHA-256 does not match RELEASE_INTENT"
        )
    for directory in (
        candidate.classifier_bundle_dir,
        candidate.rejector_bundle_dir,
        candidate.classifier_fusion_dir,
        candidate.rejector_fusion_dir,
        candidate.staged_dir,
        candidate.prefilter_dir,
    ):
        if _is_below(Path(directory), root):
            raise ValueError(
                "every candidate component must live outside the release root"
            )
    bound_files = (
        candidate.candidate_manifest_path,
        candidate.validation_evidence_path,
        candidate.frozen_contract_path,
        candidate.staged_metrics_path,
        *(
            _resolve_frozen_path(record["path"], field=f"browser_assets.{role}")
            for role, record in candidate.candidate_manifest[
                "browser_assets"
            ].items()
        ),
        _resolve_frozen_path(
            candidate.candidate_manifest["staging_package_manifest"]["path"],
            field="staging_package_manifest.path",
        ),
        _resolve_frozen_path(
            candidate.candidate_manifest["dual_binding"]["path"],
            field="dual_binding.path",
        ),
    )
    if any(_is_below(path, root) for path in bound_files):
        raise ValueError(
            "every candidate manifest component must live outside the "
            "release output root"
        )

    target_per_class = _integer(
        intent.get("target_per_class"), "target_per_class", MIN_TARGET_PER_CLASS
    )
    if intent.get("exact_target_per_class") is not True:
        raise ValueError("release suite must use exact per-class targeting")
    lengths_value = intent.get("capture_lengths")
    if (
        not isinstance(lengths_value, list)
        or lengths_value != list(REQUIRED_CAPTURE_LENGTHS)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 64
            for value in lengths_value
        )
    ):
        raise ValueError(
            "capture_lengths must equal the frozen required release length suite"
        )
    lengths = [int(value) for value in lengths_value]

    corpus_records = release_manifest.get("corpora")
    if not isinstance(corpus_records, list) or len(corpus_records) != len(lengths):
        raise ValueError("release manifest corpus list is incomplete")
    by_length: dict[int, Mapping[str, Any]] = {}
    for record in corpus_records:
        if not isinstance(record, Mapping):
            raise ValueError("each release corpus record must be an object")
        length = _integer(record.get("capture_length"), "capture_length", 64)
        if length in by_length:
            raise ValueError(f"duplicate release corpus length {length}")
        by_length[length] = record
    if sorted(by_length) != lengths:
        raise ValueError("release corpus lengths differ from precommitted lengths")

    corpora: dict[int, release.CorpusSpec] = {}
    reference_items: list[Mapping[str, Any]] | None = None
    classes: tuple[str, ...] | None = None
    reference_keys: tuple[str, ...] | None = None
    for length in lengths:
        record = by_length[length]
        expected_derivation = (
            "generated_once_at_longest_length"
            if length == lengths[-1]
            else "bit_exact_row_prefix_of_longest"
        )
        if record.get("derivation") != expected_derivation:
            raise ValueError(
                f"release corpus n{length} derivation declaration is invalid"
            )
        directory_input = Path(str(record.get("directory", ""))).expanduser()
        expected_directory = root / f"n{length}"
        if directory_input.is_symlink() or expected_directory.is_symlink():
            raise ValueError(
                f"release corpus n{length} directory may not be a symlink"
            )
        directory = directory_input.resolve()
        if directory != expected_directory:
            raise ValueError(
                f"release corpus n{length} escapes its sealed directory"
            )
        files = record.get("files")
        if not isinstance(files, Mapping) or set(files) != {
            "corpus.json",
            "corpus.f32",
            "corpus_clean.f32",
        }:
            raise ValueError(
                f"release corpus n{length} file records are incomplete"
            )
        corpus_manifest_path = directory / "corpus.json"
        raw_path = directory / "corpus.f32"
        clean_path = directory / "corpus_clean.f32"
        _verify_file_record(
            corpus_manifest_path, files["corpus.json"], name=f"n{length}/corpus.json"
        )
        _verify_file_record(
            raw_path, files["corpus.f32"], name=f"n{length}/corpus.f32"
        )
        _verify_file_record(
            clean_path,
            files["corpus_clean.f32"],
            name=f"n{length}/corpus_clean.f32",
        )
        corpus_manifest = _read_json(corpus_manifest_path)
        count = _integer(corpus_manifest.get("count"), f"n{length}.count", 1)
        if (
            corpus_manifest.get("format") != "cf32le-interleaved"
            or corpus_manifest.get("sampleCount") != length
            or corpus_manifest.get("corpusSeed") != release_seed
            or corpus_manifest.get("targetPerClass") != target_per_class
            or corpus_manifest.get("exactTargetPerClass") is not True
            or corpus_manifest.get("hasCleanPairs") is not True
            or corpus_manifest.get("signalLabRoot") != str(signal_lab_root)
        ):
            raise ValueError(
                f"n{length} corpus geometry/seed/target/format is invalid"
            )
        items = corpus_manifest.get("items")
        corpus_classes = corpus_manifest.get("classes")
        if (
            not isinstance(items, list)
            or len(items) != count
            or any(not isinstance(item, Mapping) for item in items)
            or not isinstance(corpus_classes, list)
            or not corpus_classes
            or any(
                not isinstance(name, str) or not name for name in corpus_classes
            )
            or corpus_classes != sorted(set(corpus_classes))
        ):
            raise ValueError(f"n{length} corpus classes/items are malformed")
        class_counts = {
            name: sum(item.get("cls") == name for item in items)
            for name in corpus_classes
        }
        if any(value != target_per_class for value in class_counts.values()):
            raise ValueError(
                f"n{length} class counts differ from exact target "
                f"{target_per_class}: {class_counts}"
            )
        expected_bytes = count * length * 2 * np.dtype("<f4").itemsize
        if (
            raw_path.stat().st_size != expected_bytes
            or clean_path.stat().st_size != expected_bytes
        ):
            raise ValueError(
                f"n{length} raw byte size disagrees with corpus geometry"
            )
        keys = tuple(release._row_keys(items))
        if reference_items is None:
            reference_items = items
            reference_keys = keys
            classes = tuple(corpus_classes)
        else:
            if tuple(corpus_classes) != classes or keys != reference_keys:
                raise ValueError(
                    f"n{length} rows are not metadata-matched to the reference "
                    "corpus"
                )
        corpora[length] = release.CorpusSpec(
            capture_length=length,
            directory=directory,
            manifest_path=corpus_manifest_path,
            raw_path=raw_path,
            clean_path=clean_path,
            manifest=corpus_manifest,
            row_keys=keys,
        )

    assert reference_items is not None and classes is not None
    if classes != candidate.classes:
        raise ValueError(
            f"sealed corpus classes {list(classes)} differ from the candidate "
            f"classes {list(candidate.classes)}"
        )
    start_probe_report = release._validate_unscored_start_probe(
        release_manifest,
        root=root,
        longest=corpora[lengths[-1]],
        classes=classes,
        release_seed=release_seed,
        target_per_class=target_per_class,
        signal_lab_root=signal_lab_root,
    )
    prefix_nesting_report = release.verify_prefix_nesting(corpora)
    prefix_nesting_report["derivation_records"] = (
        release.validate_prefix_derivation_records(corpora)
    )
    support, query, split_report = release.predeclared_five_shot_split(
        reference_items, classes, release_seed
    )
    return V3ReleaseSuite(
        root=root,
        intent_path=intent_path,
        manifest_path=manifest_path,
        intent=intent,
        release_manifest=release_manifest,
        candidate_sha256=candidate_hash,
        release_seed=release_seed,
        classes=classes,
        corpora=corpora,
        support_indices=support,
        query_indices=query,
        split_report=split_report,
        prefix_nesting_report=prefix_nesting_report,
        dependency_report=dependency_report,
        start_probe_report=start_probe_report,
    )


# ---------------------------------------------------------------------------
# inference: the additive sub-path and the staged decision path
# ---------------------------------------------------------------------------


def _preprocess_iq_rows(
    captures: Sequence[np.ndarray],
    candidate: V3Candidate,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the shared frontend and return packed I/Q plus RAW features."""
    if not len(captures):
        raise ValueError("preprocessing requires at least one capture")
    packed = np.empty(
        (
            len(captures),
            2,
            candidate.patch_count * candidate.patch_length,
        ),
        dtype=np.float32,
    )
    features = np.empty((len(captures), len(candidate.feature_mean)), dtype=np.float32)
    for row, raw in enumerate(captures):
        channels, raw_features, _context = td_preprocess.preprocess(
            raw,
            patch_length=candidate.patch_length,
            patch_count=candidate.patch_count,
            target_frac=candidate.target_frac,
        )
        packed[row] = np.asarray(channels, dtype=np.float32)
        features[row] = np.asarray(raw_features, dtype=np.float32)
    if not np.isfinite(packed).all() or not np.isfinite(features).all():
        raise RuntimeError("v3 frontend produced non-finite preprocessing")
    return packed, features


def _standardize_features(
    raw_features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    role: str,
) -> np.ndarray:
    features = (
        np.asarray(raw_features, dtype=np.float32)
        - np.asarray(mean, dtype=np.float32)
    ) / np.asarray(std, dtype=np.float32)
    if not np.isfinite(features).all():
        raise RuntimeError(f"{role} feature standardization produced non-finite")
    return features.astype(np.float32, copy=False)


def _embed_fusion(
    fusion_module: torch.nn.Module,
    fusion_artifact: Any,
    packed: np.ndarray,
    features: np.ndarray,
    device: torch.device,
    *,
    role: str,
) -> dict[str, np.ndarray]:
    """Branch and fused embeddings for one explicitly named fusion role."""
    real = embed_all(
        fusion_module.real_branch, packed, features, device
    ).astype(np.float32, copy=False)
    complex_embedding = embed_all(
        fusion_module.complex_branch, packed, features, device
    ).astype(np.float32, copy=False)
    fused = assemble.fuse_numpy(
        real,
        complex_embedding,
        fusion_artifact.real_center,
        fusion_artifact.complex_center,
        weight_real=fusion_artifact.weight_real,
    )
    result = {"real": real, "complex": complex_embedding, "fusion": fused}
    if any(not np.isfinite(value).all() for value in result.values()):
        raise RuntimeError(f"{role} runtime produced non-finite embeddings")
    return result


def _embed_classifier(
    candidate: V3Candidate,
    packed: np.ndarray,
    raw_features: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    features = _standardize_features(
        raw_features,
        candidate.classifier_feature_mean,
        candidate.classifier_feature_std,
        role="classifier",
    )
    return features, _embed_fusion(
        candidate.classifier_fusion_module,
        candidate.classifier_fusion_artifact,
        packed,
        features,
        device,
        role="classifier",
    )


def _embed_rejector(
    candidate: V3Candidate,
    packed: np.ndarray,
    raw_features: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    features = _standardize_features(
        raw_features,
        candidate.rejector_feature_mean,
        candidate.rejector_feature_std,
        role="rejector",
    )
    return features, _embed_fusion(
        candidate.rejector_fusion_module,
        candidate.rejector_fusion_artifact,
        packed,
        features,
        device,
        role="rejector",
    )


def _final_labels_from_classifier(
    classifier_prediction: np.ndarray,
    rejected: np.ndarray,
) -> np.ndarray:
    """Apply the rejector mask without ever substituting its class guess."""
    labels = np.asarray(classifier_prediction, dtype=np.int64)
    mask = np.asarray(rejected, dtype=bool)
    if labels.ndim != 1 or mask.shape != labels.shape:
        raise ValueError("classifier predictions and rejection mask must align")
    return np.where(mask, np.int64(-1), labels)


def _unstaged_scores(
    candidate: V3Candidate,
    embeddings: Mapping[str, np.ndarray],
    packed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """The additive stage-2 score of every row, from already-built embeddings.

    Mirrors ``fit_v3_openset.run``'s selection scoring: branch-LOF rank
    ensemble, frozen policy score, and the additivity assertion that the
    closed label is unchanged by scoring.
    """
    lof = openset_base.score_branch_lof(
        candidate.rejector.lof_components,
        {"real": embeddings["real"], "complex": embeddings["complex"]},
    )
    # train.nearest, exactly as fit_v3_openset.score_rows predicts, so the
    # additivity assertion below compares one formula with itself.  The closed
    # and invariance reports use release._nearest, exactly as v2 did.
    prediction, _distance = openset_base.nearest(
        embeddings["fusion"], candidate.rejector.prototypes
    )
    prediction = np.asarray(prediction, dtype=np.int64)
    score = candidate.rejector.policy.score(lof, packed, prediction)
    openset_base.assert_closed_label_unchanged(
        embeddings["fusion"],
        candidate.rejector.prototypes,
        prediction,
        score,
    )
    return prediction, np.asarray(score, dtype=np.float64)


def _stage_one_features(
    candidate: V3Candidate,
    captures: Sequence[np.ndarray],
    *,
    population: str,
) -> np.ndarray:
    return staged.prefilter_matrix(
        candidate.posedegen,
        captures,
        patch_length=candidate.patch_length,
        target_frac=candidate.target_frac,
        population=population,
    )


def _staged_scores(
    candidate: V3Candidate,
    captures: Sequence[np.ndarray],
    packed: np.ndarray,
    features: np.ndarray,
    unstaged_prediction: np.ndarray,
    unstaged_score: np.ndarray,
    device: torch.device,
    *,
    capture_length: int,
    population: str,
) -> dict[str, Any]:
    """Score one row set through the staged decision path.

    On a fitted length this is ``fit_v3_openset_staged.score_staged``
    verbatim: stage 1 gates, downstream is never computed for gated rows
    inside this pass, survivors are scored on the COMPOSITE axis against the
    frozen composite threshold, and the staged/unstaged survivor agreement of
    the stage-2 SUB-score is asserted.  On a length ABOVE the longest fitted
    length the causal-prefix rule applies: stage-1 features are computed on
    each capture's first max-fitted-length samples
    (``staged.stage_one_prefix_captures``) and the gate scores them with that
    length's bundle; everything downstream is the covered path unchanged.  On
    a length BELOW the smallest fitted length the gate cannot fire and no
    stage-1 score exists, so the composite degenerates to the stage-2 rank
    alone: the staged score IS the additive score, still thresholded at the
    single frozen staged (composite) threshold, which is recorded rather than
    hidden.
    """
    rows = int(len(packed))
    threshold = candidate.threshold
    length = int(capture_length)
    fitted = length in candidate.stage_one_lengths
    prefix_applied = (not fitted) and length > int(
        max(candidate.stage_one_lengths)
    )
    active = fitted or prefix_applied
    feature_length: int | None = None
    if active:
        feature_captures, feature_length = staged.stage_one_prefix_captures(
            candidate.stage_one, captures, length
        )
        stage_features = _stage_one_features(
            candidate, feature_captures, population=population
        )
        outcome = staged.score_staged(
            candidate.rejector,
            candidate.stage_one,
            packed,
            features,
            stage_features,
            device,
            capture_length=length,
            composite=candidate.composite,
        )
        agreement = staged.subset_agreement(outcome, unstaged_score)
        gated = np.asarray(outcome.gated, dtype=bool)
        staged_score = np.asarray(outcome.staged_score, dtype=np.float64)
        by_stage = staged.known_false_unknown_by_stage(
            outcome,
            unstaged_score,
            threshold,
            candidate.stage_two_threshold,
        )
    else:
        gated = np.zeros(rows, dtype=bool)
        staged_score = np.asarray(unstaged_score, dtype=np.float64)
        agreement = {
            "survivor_rows": rows,
            "max_abs_difference": 0.0,
            "exact": True,
            "note": (
                "stage 1 inactive below the smallest fitted length; no "
                "stage-1 score exists, so the composite degenerates to the "
                "stage-2 rank and staged == additive on the score axis"
            ),
        }
        rejected = staged_score > threshold
        unstaged_rejected = staged_score > candidate.stage_two_threshold
        by_stage = {
            "rows": rows,
            "staged_threshold": float(threshold),
            "unstaged_threshold": float(candidate.stage_two_threshold),
            "staged_false_unknown_rate": float(np.mean(rejected)),
            "stage_one_gate_rate": 0.0,
            "stage_two_false_unknown_rate_marginal": float(np.mean(rejected)),
            "stage_two_false_unknown_rate_among_survivors": float(
                np.mean(rejected)
            ),
            "unstaged_false_unknown_rate": float(np.mean(unstaged_rejected)),
            "rejected_by_stage_one_only": 0,
            "rejected_by_stage_two_only": int(np.count_nonzero(rejected)),
            "attribution": (
                "this capture length is below the smallest fitted stage-1 "
                "length, so no bundle's fitted length is a causal prefix of "
                "it; every row flowed to the additive stage-2 path and the "
                "composite degenerates to the stage-2 rank, thresholded at "
                "the staged (composite) threshold"
            ),
        }
    prediction = np.asarray(unstaged_prediction, dtype=np.int64).copy()
    rejected = gated | (staged_score > threshold)
    return {
        "stage_one_active": bool(active),
        "stage_one_feature_length": (
            int(feature_length) if feature_length is not None else None
        ),
        "stage_one_prefix_rule_applied": bool(prefix_applied),
        "gated": gated,
        "rejected": rejected,
        "rejector_internal_prediction": prediction,
        "staged_score": staged_score,
        "gated_fraction": float(np.mean(gated)) if rows else 0.0,
        "rejected_fraction": (
            float(np.mean(staged_score > threshold)) if rows else 0.0
        ),
        "subset_scoring_agreement": agreement,
        "false_unknown_by_stage": by_stage,
    }


# ---------------------------------------------------------------------------
# novelty: generated once at the maximum length from the release seed
# ---------------------------------------------------------------------------


def _novelty_seed(base_seed: int, family: str) -> int:
    """The v2 derivation with the release seed as base (v2 used 20260729)."""
    if family not in NOVELTY_FAMILIES:
        raise ValueError(f"unfrozen novelty family {family!r}")
    payload = f"{int(base_seed)}\0{family}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _novelty_rows_by_family(
    base_seed: int,
    longest: int,
) -> tuple[dict[str, list[np.ndarray]], dict[str, Any]]:
    rows_by_family: dict[str, list[np.ndarray]] = {}
    seed_by_family: dict[str, int] = {}
    longest_bytes_sha256: dict[str, str] = {}
    for family in NOVELTY_FAMILIES:
        seed = _novelty_seed(base_seed, family)
        seed_by_family[family] = seed
        rng = np.random.default_rng(seed)
        rows = [
            np.asarray(NOVELTY[family](longest, rng), dtype=np.complex128)
            for _ in range(NOVELTY_N_EACH_PER_LENGTH)
        ]
        if any(
            row.shape != (longest,)
            or not np.isfinite(row.real).all()
            or not np.isfinite(row.imag).all()
            for row in rows
        ):
            raise RuntimeError(
                f"novelty generator {family!r} returned invalid rows"
            )
        digest = hashlib.sha256()
        for row in rows:
            digest.update(np.asarray(row, dtype="<c16").tobytes(order="C"))
        longest_bytes_sha256[family] = digest.hexdigest()
        rows_by_family[family] = rows
    provenance = {
        "base_seed": int(base_seed),
        "derived_seed_by_family": seed_by_family,
        "families": list(NOVELTY_FAMILIES),
        "rows_per_family": int(NOVELTY_N_EACH_PER_LENGTH),
        "longest_capture_length": int(longest),
        "longest_complex128_bytes_sha256": longest_bytes_sha256,
        "prefix_rule": (
            "each shorter novelty observation is the exact leading slice of "
            "the same in-memory maximum-length realization"
        ),
        "recalibration_performed": False,
    }
    return rows_by_family, provenance


# ---------------------------------------------------------------------------
# per-length open report on the staged axis (v2 report keys preserved)
# ---------------------------------------------------------------------------


def _open_report(
    known: Mapping[str, Any],
    novelty: Mapping[str, Mapping[str, Any]],
    candidate: V3Candidate,
    novelty_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    known_score = np.asarray(known["staged_score"], dtype=np.float64)
    family_scores = {
        family: np.asarray(novelty[family]["staged_score"], dtype=np.float64)
        for family in NOVELTY_FAMILIES
    }
    if any(
        len(scores) != NOVELTY_N_EACH_PER_LENGTH
        for scores in family_scores.values()
    ):
        raise RuntimeError(
            "prefix-matched novelty population has an unexpected row count"
        )
    all_novelty = np.concatenate(
        [family_scores[name] for name in NOVELTY_FAMILIES]
    )
    threshold = candidate.threshold
    return {
        "known_rows": int(len(known_score)),
        "novelty_rows_per_family": int(NOVELTY_N_EACH_PER_LENGTH),
        "base_seed": int(novelty_provenance["base_seed"]),
        "derived_seed_by_family": dict(
            novelty_provenance["derived_seed_by_family"]
        ),
        "maximum_length_realization_sha256": dict(
            novelty_provenance["longest_complex128_bytes_sha256"]
        ),
        "shorter_observation_rule": novelty_provenance["prefix_rule"],
        "score": (
            "staged axis, policy version "
            f"{staged.STAGED_POLICY_VERSION}: frozen stage-1 noise prefilter "
            "gates; survivors carry the COMPOSITE "
            f"{staged.COMPOSITE_SURVIVOR_SCORE} over the frozen "
            "enrollment-ranked branch-LOF + geometry blend and the "
            "enrollment-survivor stage-1 rank; gated rows are placed above "
            "every survivor"
        ),
        "staged_policy_version": staged.STAGED_POLICY_VERSION,
        "threshold": float(threshold),
        "threshold_rule": (
            "frozen q95 of the composite over enrollment stage-1 survivors, "
            "loaded verbatim; the unstaged control below uses the stage-2 "
            "policy's own untouched threshold"
        ),
        "recalibration_performed": False,
        "stage_one_active": bool(known["stage_one_active"]),
        "stage_one_feature_length": known["stage_one_feature_length"],
        "stage_one_prefix_rule_applied": bool(
            known["stage_one_prefix_rule_applied"]
        ),
        "auroc_overall": release._auroc(all_novelty, known_score),
        **{
            f"auroc_{name}": release._auroc(family_scores[name], known_score)
            for name in NOVELTY_FAMILIES
        },
        "known_false_unknown_rate": float(np.mean(known_score > threshold)),
        **{
            f"flagged_unknown_{name}": float(
                np.mean(family_scores[name] > threshold)
            )
            for name in NOVELTY_FAMILIES
        },
        "known_false_unknown_by_stage": dict(known["false_unknown_by_stage"]),
        "known_gated_fraction": float(known["gated_fraction"]),
        **{
            f"gated_fraction_{name}": float(novelty[name]["gated_fraction"])
            for name in NOVELTY_FAMILIES
        },
        "subset_scoring_agreement": {
            "known": dict(known["subset_scoring_agreement"]),
            **{
                name: dict(novelty[name]["subset_scoring_agreement"])
                for name in NOVELTY_FAMILIES
            },
        },
        "unstaged_control": {
            "auroc_overall": release._auroc(
                np.concatenate(
                    [
                        np.asarray(
                            novelty[name]["unstaged_score"], dtype=np.float64
                        )
                        for name in NOVELTY_FAMILIES
                    ]
                ),
                np.asarray(known["unstaged_score"], dtype=np.float64),
            ),
            **{
                f"auroc_{name}": release._auroc(
                    np.asarray(
                        novelty[name]["unstaged_score"], dtype=np.float64
                    ),
                    np.asarray(known["unstaged_score"], dtype=np.float64),
                )
                for name in NOVELTY_FAMILIES
            },
            "threshold": float(candidate.stage_two_threshold),
            "known_false_unknown_rate": float(
                np.mean(
                    np.asarray(known["unstaged_score"], dtype=np.float64)
                    > candidate.stage_two_threshold
                )
            ),
            **{
                f"flagged_unknown_{name}": float(
                    np.mean(
                        np.asarray(
                            novelty[name]["unstaged_score"], dtype=np.float64
                        )
                        > candidate.stage_two_threshold
                    )
                )
                for name in NOVELTY_FAMILIES
            },
            "role": (
                "verification control: the additive path over the same rows "
                "against the stage-2 policy's own untouched threshold; never "
                "a gate input"
            ),
        },
        "known_score_quantiles": {
            "p50": float(np.quantile(known_score, 0.5)),
            "p95": float(np.quantile(known_score, 0.95)),
        },
    }


# ---------------------------------------------------------------------------
# scale sweep: the v2 sweep semantics with the v3 additive sub-path
# ---------------------------------------------------------------------------


def _scale_sweep(
    spec: release.CorpusSpec,
    candidate: V3Candidate,
    device: torch.device,
    labels: np.ndarray,
    query_indices: np.ndarray,
) -> dict[str, Any]:
    eligible, eligibility = release._scale_eligible_indices(
        spec, query_indices, candidate.classes
    )
    base_captures = release._raw_rows(spec, eligible)
    rows = []
    embeddings_by_factor: dict[float, np.ndarray] = {}
    predictions_by_factor: dict[float, np.ndarray] = {}
    wanted = labels[eligible]
    snr = np.asarray(
        [
            release._finite_number(
                spec.manifest["items"][int(index)]["snrDb"], "snrDb"
            )
            for index in eligible
        ],
        dtype=np.float64,
    )
    for factor in PHYSICAL_SCALE_FACTORS:
        new_length = max(64, int(round(MATCHED_CAPTURE_LENGTH / factor)))
        effective_factor = (MATCHED_CAPTURE_LENGTH - 1) / (new_length - 1)
        captures = (
            base_captures
            if factor == 1.0
            else [
                production_preprocess.lin_resample(raw, new_length)
                for raw in base_captures
            ]
        )
        packed, raw_features = _preprocess_iq_rows(captures, candidate)
        _features, classifier_embeddings = _embed_classifier(
            candidate, packed, raw_features, device
        )
        embeddings = classifier_embeddings["fusion"]
        embeddings_by_factor[factor] = embeddings
        closed, prediction = release._simple_closed_report(
            embeddings,
            wanted,
            candidate.classifier_fusion_artifact.prototypes,
            candidate.classes,
        )
        predictions_by_factor[factor] = prediction
        transformed_sample_rates = np.asarray(
            [
                release._finite_number(
                    spec.manifest["items"][int(index)]["sampleRateHz"],
                    "sampleRateHz",
                )
                / effective_factor
                for index in eligible
            ],
            dtype=np.float64,
        )
        rows.append(
            {
                "factor": factor,
                "effective_factor": effective_factor,
                "raw_length": new_length,
                "transformed_metadata": {
                    "sample_rate_hz_rule": (
                        "original sampleRateHz / effective_factor"
                    ),
                    "sample_rate_hz_min": float(transformed_sample_rates.min()),
                    "sample_rate_hz_max": float(transformed_sample_rates.max()),
                    "bandwidth_hz_rule": "unchanged",
                    "centre_offset_fraction_rule": (
                        "original centreOffsetFrac * effective_factor"
                    ),
                },
                "accuracy": closed["accuracy"],
                "balanced_accuracy": closed["balanced_accuracy"],
                "per_class_recall": closed["per_class_recall"],
                "mean_pairwise_cosine": closed["mean_pairwise_cosine"],
            }
        )
    reference = embeddings_by_factor[1.0]
    reference_prediction = predictions_by_factor[1.0]
    for row in rows:
        factor = float(row["factor"])
        row["paired_to_factor1"] = release._paired_invariance_report(
            embeddings_by_factor[factor],
            reference,
            predictions_by_factor[factor],
            reference_prediction,
            wanted,
            snr,
            candidate.classes,
        )
    return {
        "definition": (
            "normalized frequencies multiplied by s through deterministic "
            "linear resampling to N/s; common predeclared no-alias subset"
        ),
        "rejector_rule": SWEEP_REJECTOR_RULE,
        "factors": list(PHYSICAL_SCALE_FACTORS),
        "eligible_rows": int(len(eligible)),
        **eligibility,
        "eligible_indices_sha256": hashlib.sha256(
            eligible.astype("<i8", copy=False).tobytes()
        ).hexdigest(),
        "rows": rows,
        "worst_balanced_accuracy": min(
            float(row["balanced_accuracy"]) for row in rows
        ),
        "worst_mean_pairwise_cosine": max(
            float(row["mean_pairwise_cosine"]) for row in rows
        ),
        "worst_prediction_agreement_to_factor1": min(
            float(row["paired_to_factor1"]["prediction_agreement"])
            for row in rows
        ),
        "worst_embedding_cosine_to_factor1": min(
            float(row["paired_to_factor1"]["embedding_cosine_mean"])
            for row in rows
        ),
    }


# ---------------------------------------------------------------------------
# the evaluation
# ---------------------------------------------------------------------------


def evaluate_release(
    suite: V3ReleaseSuite,
    candidate: V3Candidate,
    device: torch.device,
) -> dict[str, Any]:
    """Run the fully precommitted evaluation without modifying frozen assets."""
    lengths = sorted(suite.corpora)
    longest = lengths[-1]
    query = suite.query_indices

    print("[v3 release] generating prefix-matched novelty", flush=True)
    novelty_rows, novelty_provenance = _novelty_rows_by_family(
        suite.release_seed, longest
    )

    classifier_embeddings_by_length: dict[
        int, dict[str, np.ndarray]
    ] = {}
    closed_by_length: dict[str, Any] = {}
    open_by_length: dict[str, Any] = {}
    staged_known_by_length: dict[int, dict[str, Any]] = {}
    labels_by_length: dict[int, np.ndarray] = {}
    snr_by_length: dict[int, np.ndarray] = {}

    for length in lengths:
        spec = suite.corpora[length]
        print(f"[v3 release] preprocessing/inference n={length}", flush=True)
        captures = release._raw_rows(
            spec, np.arange(int(spec.manifest["count"]), dtype=np.int64)
        )
        packed, raw_features = _preprocess_iq_rows(captures, candidate)
        _classifier_features, classifier_embeddings = _embed_classifier(
            candidate, packed, raw_features, device
        )
        rejector_features, rejector_embeddings = _embed_rejector(
            candidate, packed, raw_features, device
        )
        classifier_embeddings_by_length[length] = classifier_embeddings
        labels, snr, impaired = release._labels_and_snr(spec, candidate.classes)
        labels_by_length[length] = labels
        snr_by_length[length] = snr

        closed, classifier_prediction = release._closed_report(
            classifier_embeddings["fusion"][query],
            labels[query],
            snr[query],
            impaired[query],
            candidate.classifier_fusion_artifact.prototypes,
            candidate.classes,
        )
        closed_by_length[str(length)] = closed

        query_embeddings = {
            name: rejector_embeddings[name][query]
            for name in ("real", "complex", "fusion")
        }
        unstaged_prediction, unstaged_score = _unstaged_scores(
            candidate, query_embeddings, packed[query]
        )
        known_staged = _staged_scores(
            candidate,
            [captures[int(index)] for index in query],
            packed[query],
            rejector_features[query],
            unstaged_prediction,
            unstaged_score,
            device,
            capture_length=length,
            population=f"n{length}-known-query",
        )
        known_staged["unstaged_score"] = unstaged_score
        known_staged["final_label_or_unknown"] = _final_labels_from_classifier(
            classifier_prediction,
            known_staged["rejected"],
        )
        staged_known_by_length[length] = known_staged

        novelty_by_family: dict[str, dict[str, Any]] = {}
        for family in NOVELTY_FAMILIES:
            prefixes = [row[:length] for row in novelty_rows[family]]
            novelty_packed, novelty_raw_features = _preprocess_iq_rows(
                prefixes, candidate
            )
            novelty_features, novelty_embeddings = _embed_rejector(
                candidate, novelty_packed, novelty_raw_features, device
            )
            family_prediction, family_score = _unstaged_scores(
                candidate, novelty_embeddings, novelty_packed
            )
            family_staged = _staged_scores(
                candidate,
                prefixes,
                novelty_packed,
                novelty_features,
                family_prediction,
                family_score,
                device,
                capture_length=length,
                population=f"n{length}-novelty-{family}",
            )
            family_staged["unstaged_score"] = family_score
            novelty_by_family[family] = family_staged
        open_by_length[str(length)] = _open_report(
            known_staged, novelty_by_family, candidate, novelty_provenance
        )

    matched_enrollment = classifier_embeddings_by_length[
        MATCHED_CAPTURE_LENGTH
    ]["fusion"]
    matched_labels = labels_by_length[MATCHED_CAPTURE_LENGTH]
    five_shot_by_length: dict[str, Any] = {}
    for length in lengths:
        if not np.array_equal(labels_by_length[length], matched_labels):
            raise AssertionError("matched release label arrays differ by length")
        five_shot_by_length[str(length)] = release._five_shot_report(
            matched_enrollment,
            classifier_embeddings_by_length[length]["fusion"],
            matched_labels,
            suite.support_indices,
            suite.query_indices,
            candidate.classes,
        )

    reference = classifier_embeddings_by_length[MATCHED_CAPTURE_LENGTH][
        "fusion"
    ][query]
    reference_prediction, _ = release._nearest(
        reference, candidate.classifier_fusion_artifact.prototypes
    )
    length_rows = []
    for length in lengths:
        current = classifier_embeddings_by_length[length]["fusion"][query]
        prediction, _ = release._nearest(
            current, candidate.classifier_fusion_artifact.prototypes
        )
        closed = closed_by_length[str(length)]
        paired = release._paired_invariance_report(
            current,
            reference,
            prediction,
            reference_prediction,
            labels_by_length[length][query],
            snr_by_length[length][query],
            candidate.classes,
        )
        length_rows.append(
            {
                "capture_length": length,
                "accuracy": closed["accuracy"],
                "balanced_accuracy": closed["balanced_accuracy"],
                "per_class_recall": closed["per_class_recall"],
                "mean_pairwise_cosine": closed["mean_pairwise_cosine"],
                "paired_to_matched": paired,
            }
        )
    length_sweep = {
        "matched_capture_length": int(MATCHED_CAPTURE_LENGTH),
        "matched_rows": int(len(query)),
        "all_classes_required": list(candidate.classes),
        "rejector_rule": SWEEP_REJECTOR_RULE,
        "rows": length_rows,
        "worst_balanced_accuracy": min(
            float(row["balanced_accuracy"]) for row in length_rows
        ),
        "worst_mean_pairwise_cosine": max(
            float(row["mean_pairwise_cosine"]) for row in length_rows
        ),
        "worst_prediction_agreement_to_matched": min(
            float(row["paired_to_matched"]["prediction_agreement"])
            for row in length_rows
        ),
        "worst_embedding_cosine_to_matched": min(
            float(row["paired_to_matched"]["embedding_cosine_mean"])
            for row in length_rows
        ),
    }
    scale_sweep = _scale_sweep(
        suite.corpora[MATCHED_CAPTURE_LENGTH],
        candidate,
        device,
        labels_by_length[MATCHED_CAPTURE_LENGTH],
        suite.query_indices,
    )

    gates = release._assemble_release_gates(
        candidate_sha256=suite.candidate_sha256,
        expected_candidate_sha256=suite.release_manifest["candidate_sha256"],
        prefix_nesting_passes=bool(suite.prefix_nesting_report["passes"]),
        dependency_provenance_passes=bool(suite.dependency_report["passes"]),
        start_probe_excluded=(
            suite.start_probe_report["scored"] is False
            and bool(suite.start_probe_report["passes"])
        ),
        closed_by_length=closed_by_length,
        five_shot_by_length=five_shot_by_length,
        open_by_length=open_by_length,
        length_sweep=length_sweep,
        scale_sweep=scale_sweep,
    )
    # This is the v2 assembler's strict output verbatim.  In particular,
    # five-shot stays at 0.85 and known FUR stays at 0.10; the historical
    # seed-20260734 redeclaration is metadata only.
    all_pass = all(bool(value["passes"]) for value in gates.values())

    max_fitted = int(max(candidate.stage_one_lengths))
    stage_one_coverage = {
        str(length): (
            "fitted"
            if int(length) in candidate.stage_one_lengths
            else "causal_prefix"
            if int(length) > max_fitted
            else "inactive"
        )
        for length in lengths
    }
    stage_one_feature_lengths = {
        str(length): staged_known_by_length[length]["stage_one_feature_length"]
        for length in lengths
    }
    final_label_counts = {
        str(length): {
            "rows": int(len(staged_known_by_length[length]["final_label_or_unknown"])),
            "unknown": int(
                np.count_nonzero(
                    staged_known_by_length[length]["final_label_or_unknown"]
                    == -1
                )
            ),
            "gated_at_stage_one": int(
                np.count_nonzero(staged_known_by_length[length]["gated"])
            ),
            "accepted_rows": int(
                np.count_nonzero(~staged_known_by_length[length]["rejected"])
            ),
            "accepted_label_source": "classifier_fusion_8k_regularized",
            "accepted_classifier_rejector_label_agreement": float(
                np.mean(
                    staged_known_by_length[length]["final_label_or_unknown"][
                        ~staged_known_by_length[length]["rejected"]
                    ]
                    == staged_known_by_length[length][
                        "rejector_internal_prediction"
                    ][~staged_known_by_length[length]["rejected"]]
                )
            )
            if np.any(~staged_known_by_length[length]["rejected"])
            else None,
        }
        for length in lengths
    }

    return {
        "schema": EVALUATOR_SCHEMA,
        "status": "complete",
        "release_evidence": True,
        "evaluation_version": EVALUATION_VERSION,
        "development_data_loaded": False,
        "retraining_performed": False,
        "recalibration_performed": False,
        "architecture_contract": {
            "candidate_architecture": "dual_fusion_classifier8k_rejector4k",
            "intentional_dual_fusion": True,
            "decision_order": [
                "stage-one causal-prefix noise gate",
                "4k rejector fusion and staged known/unknown policy",
                "8k classifier nearest-prototype label for accepted rows",
            ],
            "known_label_source": "classifier_fusion_8k_regularized",
            "known_unknown_source": "rejector_fusion_4k_frozen_policy",
            "closed_gate_source": "classifier_fusion_8k_regularized",
            "open_gate_source": "rejector_fusion_4k_frozen_policy",
            "additive_only": False,
            "changes_closed_label": True,
            "gates_before_classification": True,
            "architecture_contract_change": str(
                noise_prefilter.ARCHITECTURE_CONTRACT_CHANGE
            ),
            "staged_policy_kind": staged.STAGED_POLICY_KIND,
            "staged_policy_schema": int(staged.STAGED_POLICY_SCHEMA),
            "staged_policy_version": staged.STAGED_POLICY_VERSION,
            "survivor_score": staged.COMPOSITE_SURVIVOR_SCORE,
            "staged_score_note": staged.STAGED_SCORE_NOTE,
            "known_false_unknown_accounting": KNOWN_FUR_ACCOUNTING,
            "stage_one_capture_lengths": [
                int(length) for length in candidate.stage_one_lengths
            ],
            "stage_one_coverage_by_length": stage_one_coverage,
            "stage_one_feature_length_by_length": stage_one_feature_lengths,
            "stage_one_prefix_rule": {
                "max_fitted_length": max_fitted,
                "rule": STAGE_ONE_PREFIX_RULE,
            },
            "stage_one_uncovered_rule": STAGE_ONE_UNCOVERED_RULE,
            "sweep_rejector_rule": SWEEP_REJECTOR_RULE,
            "typescript_runtime_note": (
                "the deployed TypeScript staged runtime must apply this same "
                "causal-prefix rule at capture lengths above the longest "
                "fitted stage-1 length; a runtime that refuses such lengths "
                "does not match the sealed claim and must be reconciled "
                "before any ship"
            ),
        },
        "candidate": {
            "path": str(candidate.candidate_manifest_path),
            "sha256": suite.candidate_sha256,
            "schema": CANDIDATE_MANIFEST_SCHEMA,
            "candidate_id": CANDIDATE_ID,
            "classes": list(candidate.classes),
            "components": candidate_provenance(candidate),
            "frozen_assets_used": [
                "8k classifier fusion state, centers, feature moments and "
                "enrollment-only prototypes",
                "4k rejector fusion state, centers, feature moments and "
                "enrollment-only prototypes",
                "both self-verified runtime bundle manifests and assets",
                "frozen stage-1 per-length noise-prefilter bundles",
                "frozen stage-2 branch-LOF ensemble (training reference, "
                "enrollment ranks)",
                "frozen stage-2 policy (geometry blend, enrollment q95 "
                "threshold)",
                "frozen composite survivor policy (enrollment-survivor "
                "stage-1 rank calibration, q95 composite threshold)",
            ],
        },
        "closed_per_length": closed_by_length,
        "family_per_length": {
            length: report["family"]
            for length, report in closed_by_length.items()
        },
        "high_snr_per_length": {
            length: report["high_snr"]
            for length, report in closed_by_length.items()
        },
        "low_snr_per_length": {
            length: report["low_snr"]
            for length, report in closed_by_length.items()
        },
        "clean_subset_per_length": {
            length: report["clean_subset"]
            for length, report in closed_by_length.items()
        },
        "impaired_subset_per_length": {
            length: report["impaired_subset"]
            for length, report in closed_by_length.items()
        },
        "five_shot_predeclared_per_length": five_shot_by_length,
        "open_staged_per_length": open_by_length,
        "staged_known_decisions_per_length": final_label_counts,
        "matched_length_sweep": length_sweep,
        "physical_scale_sweep": scale_sweep,
        "gates": gates,
        "historical_gate_redeclaration": (
            historical_gate_redeclaration_block()
        ),
        "all_release_gates_pass": all_pass,
        "provenance": {
            "release_root": str(suite.root),
            "release_intent_sha256": _sha256(suite.intent_path),
            "release_manifest_sha256": _sha256(suite.manifest_path),
            "release_seed": suite.release_seed,
            "evaluation_protocol": expected_evaluation_protocol(
                suite.release_seed, candidate.stage_one_lengths
            ),
            "predeclared_split": suite.split_report,
            "prefix_nesting": suite.prefix_nesting_report,
            "dependency_provenance": suite.dependency_report,
            "unscored_start_probe": suite.start_probe_report,
            "novelty_prefix_generation": novelty_provenance,
            "candidate_sha256": suite.candidate_sha256,
            "corpus_assets": suite.release_manifest["corpora"],
            "evaluator_path": str(Path(__file__).resolve()),
            "evaluator_sha256": _sha256(Path(__file__).resolve()),
            "v2_evaluator_sha256": _sha256(
                _source_path("evaluate_invariant_release_suite.py")
            ),
            "gate_helpers_imported_from_v2": [
                "GATE_FLOORS",
                "_gate",
                "_assemble_release_gates",
                "_closed_report",
                "_simple_closed_report",
                "_classification_subset",
                "_paired_invariance_report",
                "_five_shot_report",
                "_auroc",
                "_nearest",
                "_scale_eligible_indices",
                "predeclared_five_shot_split",
                "verify_prefix_nesting",
                "validate_prefix_derivation_records",
                "_validate_unscored_start_probe",
                "_validate_dependency_provenance",
            ],
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def expected_protocol_for_prefilter_set(
    release_seed: int,
    prefilter_dir: Path,
) -> dict[str, Any]:
    """The predeclared protocol, from the fitted stage-1 set alone.

    The protocol object is a function of the release seed, the fitted stage-1
    lengths and this module's frozen constants -- nothing else about the
    candidate enters it.  Deriving the lengths from the prefilter set (each
    bundle re-validated by ``staged.load_stage_one``, including coverage of
    every sealed capture length) lets the launcher fixture be pinned BEFORE
    the composite validation artifact exists, without weakening anything: at
    evaluation time the full candidate is loaded and the intent protocol is
    recomputed against it, so a fixture generated from a different prefilter
    set than the sealed candidate's fails loudly before any gate is scored.
    """
    stage_one = staged.load_stage_one(
        staged.load_prefilter_module(),
        staged.load_posedegen_module(),
        Path(prefilter_dir),
        required_lengths=REQUIRED_CAPTURE_LENGTHS,
    )
    return expected_evaluation_protocol(release_seed, stage_one.lengths)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.print_expected_protocol is not None:
        protocol = expected_protocol_for_prefilter_set(
            int(args.print_expected_protocol), Path(args.prefilter_dir)
        )
        print(json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False))
        return {"expected_evaluation_protocol": protocol}
    device = resolve_device(args.device)
    candidate = load_candidate(
        candidate_manifest_path=Path(args.candidate_manifest),
        classifier_bundle_dir=Path(args.classifier_bundle_dir),
        rejector_bundle_dir=Path(args.rejector_bundle_dir),
        classifier_fusion_dir=Path(args.classifier_fusion_dir),
        rejector_fusion_dir=Path(args.rejector_fusion_dir),
        staged_dir=Path(args.staged_dir),
        prefilter_dir=Path(args.prefilter_dir),
        device=device,
    )
    if args.release_root is None:
        raise ValueError(
            "--release-root is required unless --print-expected-protocol is "
            "given"
        )
    suite = load_v3_release_suite(Path(args.release_root), candidate)
    report = evaluate_release(suite, candidate, device)
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else suite.root / "RELEASE_EVALUATION.json"
    )
    if not _is_below(output, suite.root):
        raise ValueError("release evaluation output must stay inside release root")
    _write_json_exclusive(output, report)
    print(
        f"[v3 release] gates="
        f"{'PASS' if report['all_release_gates_pass'] else 'FAIL'} -> {output}",
        flush=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--release-root",
        default=None,
        help="the sealed suite directory written by the release launcher",
    )
    parser.add_argument("--output")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument(
        "--candidate-manifest",
        default=str(DEFAULT_CANDIDATE_MANIFEST),
        help=(
            "the final frozen dual release-candidate manifest; this exact "
            "file is bound by RELEASE_INTENT candidate_path/candidate_sha256"
        ),
    )
    parser.add_argument(
        "--classifier-bundle-dir",
        default=str(DEFAULT_CLASSIFIER_BUNDLE_DIR),
        help="the self-verified 8k classifier runtime bundle",
    )
    parser.add_argument(
        "--rejector-bundle-dir",
        default=str(DEFAULT_REJECTOR_BUNDLE_DIR),
        help="the self-verified 4k rejector runtime bundle",
    )
    parser.add_argument(
        "--classifier-fusion-dir",
        default=str(DEFAULT_CLASSIFIER_FUSION_DIR),
        help="the regularized 8k known-class fusion artifact",
    )
    parser.add_argument(
        "--rejector-fusion-dir",
        default=str(DEFAULT_REJECTOR_FUSION_DIR),
        help="the frozen 4k known/unknown fusion artifact",
    )
    parser.add_argument(
        "--staged-dir",
        default=str(DEFAULT_STAGED_DIR),
        help=(
            "the passing staged validation artifact holding the frozen "
            "stage-2 LOF ensemble and policy npz files"
        ),
    )
    parser.add_argument(
        "--prefilter-dir",
        default=str(DEFAULT_PREFILTER_DIR),
        help="the fitted per-length stage-1 noise-prefilter bundle set",
    )
    parser.add_argument(
        "--print-expected-protocol",
        type=int,
        default=None,
        metavar="RELEASE_SEED",
        help=(
            "print the exact evaluation_protocol object the release intent "
            "must embed for the given seed, then exit without touching any "
            "release root. Needs only --prefilter-dir (the stage-1 lengths); "
            "the full candidate is not loaded, and the evaluation itself "
            "recomputes and enforces this object against the real candidate"
        ),
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
