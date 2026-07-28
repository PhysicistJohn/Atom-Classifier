"""STAGED v3 open-set rejector: a noise prefilter that **gates**, then the v3 path.

HANDOFF section 19.3 option 2, built.  Sections 17 and 19 measured the problem
this module exists for:

* Refitting the frozen additive rejector on v3 fusion embeddings solved chirp
  (AUROC 0.9716, threshold recall 0.72-1.00) and broke noise (AUROC 0.6038,
  threshold recall 0.0000).
* Pose-degeneracy features separate noise well on their own -- 0.9154, 0.9004,
  0.8872, 0.8579 noise AUROC worst-of 2 seeds x 3 lengths -- yet folding them
  into the fixed-weight rank blend lifts fused noise AUROC only to ~0.74 and
  leaves noise threshold recall at 0.01-0.07 against a >= 0.10 gate.

The diagnosis is that the failure is the **operating point**, not the ranking.
The frozen blend must carry the branch-LOF term, which on v3 embeddings ranks
noise as *known*; q95 is then fit on a fused known distribution that term
dominates, so the threshold lands in the wrong place.  A good term cannot rescue
a blend that contains a harmful one, and raising the good term's weight only
trades noise against chirp.

So noise is decided **first**, by a dedicated detector that sets its own
threshold against its **own** score distribution, before the branch-LOF term
enters at all.  Chirp continues to be handled by the existing v3 path, which
already solves it.

    STAGE 1  noise prefilter on pose-degeneracy-style features.
             Fires  -> the capture is rejected as noise and the run
                       short-circuits: no encoder forward, no fusion, no branch
                       LOF, no frozen policy score is computed for that row.
             Passes -> stage 2.

    STAGE 2  the existing :mod:`fit_v3_openset` path, unchanged and imported
             rather than reimplemented.

ARCHITECTURE CONTRACT CHANGE, stated here, in the CLI epilog, and in the emitted
JSON as ``closed_label_can_be_gated: true``
=========================================================================
Every previous rejector in this repository is **strictly additive**: it emits a
score alongside a closed-set label it can never alter, and
``fit_v3_openset.assert_closed_label_unchanged`` proves that row by row.

**This module deliberately breaks that contract.**  A gating prefilter changes
behaviour: when stage 1 fires, the system abstains *before* classification runs,
so there is no closed-set label for that capture at all.  That is the point of
the design -- it is what makes the operating point independent of the branch-LOF
distribution, and it is what makes the compute saving possible -- and it is a
deliberate change, not an oversight.  Consequences that must travel with any
release claim built on this module:

1. The rejector can suppress a closed-set label.  A known signal gated at stage 1
   is not merely flagged, it is **not classified**.  Known false-unknown rate is
   therefore the binding risk, and this module reports it **per stage** so it is
   always visible whether the prefilter or the branch LOF rejected a known row.
2. Stage 2 remains additive on the rows it sees.  Within stage 2 the closed label
   is still recomputed after scoring and asserted bit-identical, by the base
   module's own assertion.  ``closed_label_can_be_gated`` is about stage 1.
3. Because work is skipped, the prefilter can save downstream compute.  That is
   **measured** here -- short-circuited fraction and wall clock against the
   unstaged path on the same rows -- and never asserted.  Feature extraction is
   per-capture NumPy while stage 2 is a batched tensor forward, so the staged
   path may be slower in wall clock even while doing strictly less model work.
   The measurement is reported whichever way it lands.

What is preserved, unchanged and verified
=========================================
* Density is fit on **training** rows only; every empirical rank and every
  operating threshold, in both stages, on **enrollment** rows only.
* The immutable development-selection population is scored, never fit.
* Novelty is scored only, and in a ``--role validate`` run informs nothing.
* The rebuilt fusion is checked against the fusion artifact's stored centers and
  prototypes, and the report records whether the rebuild was bit-identical.
* ``sealed_release_data_used`` 0, ``consumed_test_rows_used`` 0, release seed
  20260736 unreachable, sealed and release paths refused before any data load.
* Gate floors are **imported** from :mod:`fit_v3_openset`, never re-typed, so
  they cannot drift and cannot be quietly lowered.

Stage 2 is fit on the FULL enrollment population, exactly as the unstaged path
fits it, not on the enrollment rows that survive stage 1.  Refitting stage 2 on
survivors is defensible and is deliberately **not** done, because it would
confound the effect of the gate with the effect of a refit; here the only
difference between the staged and unstaged arms is the gate itself, which is
what makes the side-by-side comparison in this report an experiment rather than
an anecdote.

The prefilter itself is neither defined nor fitted here.  It comes from a
sibling module (default import name :data:`PREFILTER_MODULE_DEFAULT`) whose
contract is :data:`NOISE_PREFILTER_CONTRACT`, and this module **loads a fitted
set** of its bundles, one per capture length, via ``--prefilter-dir``.  Fitting
stays in one place because that is where the fitting hygiene lives: coefficients
on training rows, an operating point that is a stated known-false-positive
budget on enrollment rows, and synthetic fitting noise drawn from a seed
namespace disjoint from every novelty seed.  Each of those is re-checked here
from the bundle's own provenance rather than taken on trust, and a bundle whose
metadata does not admit that it gates is refused.

The gate is length-keyed with exactly one substitution, the **causal-prefix
rule** (:data:`STAGE_ONE_PREFIX_RULE`): a capture LONGER than every fitted
length is gated by the longest fitted bundle applied to the capture's first
max-fitted-length samples, which are exactly a capture of that fitted length.
Lengths below the smallest fitted length are refused as before.  Callers that
score such longer captures (the sealed evaluator, the deployed runtime) must
slice via :func:`stage_one_prefix_captures`; this development harness applies
the same rule to its novelty sweep, computing stage-1 features on each
capture's first :meth:`StageOneGate.feature_length_for` samples, so a sweep
length above the fitted set (the sealed suite's N32768) is gated exactly as
the sealed evaluator gates it.  Every other uncovered sweep length is still
refused.

One consequence is worth stating plainly: **no novelty seed took part in
choosing the stage-1 operating point.**  This module does not extract stage-1
features for the training population at all, which is the strongest available
evidence that it cannot have fitted anything on it.  Enrollment stage-1
features ARE extracted, for exactly one purpose introduced by staged policy
version 2 below: the composite survivor rank and its threshold are, like every
other rank and threshold in this chain, calibrated on enrollment only.

STAGED POLICY VERSION 2: THE COMPOSITE SURVIVOR SCORE (HANDOFF 25 follow-up)
============================================================================
The sealed seed-20260731 run (policy version 1) froze one failure: known
false-unknown 0.10237 vs the 0.10 ceiling at N32768, with the stage-1 gate
realising ~2.4x its dev budget.  The v3.1 candidate (same policy, stage-1
budget tightened 0.02 -> 0.01) fixed known FUR on dev (0.0566) and broke the
noise AUROC floor (worst cell 0.7957 vs 0.80, seeds 20260944/20260945,
frozen): tightening the gate pushes noise into stage 2, whose own noise axis
is weak (~0.60 standalone).  Stage-1's continuous log-odds separates that
noise at 0.978 AUROC in-sample and was previously DISCARDED below the gate.

Policy version 2 keeps the tightened 0.01 gate and uses that discarded score
without moving the gate: for stage-1 survivors the open-set score becomes the
COMPOSITE ``max(stage2_enrollment_rank, stage1_score_enrollment_rank)``,
where the stage-1 rank is the empirical enrollment rank of the stage-1
log-odds, fit on ENROLLMENT SURVIVORS only with the same searchsorted-left
convention stage 2 uses.  Gated rows keep the version-1 axis,
``1 + softsign(stage-1 log-odds)`` in (1, 2), above every survivor.  The
staged unknown threshold is re-fit at the SAME frozen quantile (q95) of the
composite over enrollment survivors -- enrollment only, as ever.  The
prefilter coefficients and thresholds, the LOF ensemble, the blend weights
and every stage-2 internal are untouched.

The imports are lazy and loud: an absent or non-conforming module stops the run
before any data is touched rather than silently degrading to the unstaged
rejector, which would emit a report that looks like a staged result and is not
one.

STAGED POLICY VERSION 3 / v3.4: FROZEN q99 DESIGN FAILURE
==========================================================
The sealed seed-20260735 v3.3 run passed 22/23 gates and failed only known
false-unknown.  Policy-v3 changed one operating-point constant and nothing
else: the composite-survivor enrollment quantile rose from q95 to q99.  Its
one design draw, seed 20260952, is frozen and consumed: known false-unknown was
0.019916, but N4096 chirp threshold recall was 0.003333 against the unchanged
0.10 floor.  q99 therefore failed design and may not be redrawn or validated.

STAGED POLICY VERSION 4: PREDECLARED q97 OPERATING POINT
=========================================================
Policy-v4 changes only the failed q99 survivor quantile to q97, derived from
the same enrollment-only calibration.  Stage 1 remains the frozen 1%
enrollment-budget gate; the stage-1 coefficients and thresholds, both frozen
CNN fusions, the survivor score, the LOF ensemble, blend weights, rank
conventions and every release gate remain unchanged.

On the enrollment population, a 1% stage-1 budget followed by a 3% survivor
tail has a nominal union budget of 3.97%, still leaving more than 2.5x
headroom to the unchanged 10% known-false-unknown ceiling.  The frozen
calibration gives q97 threshold 0.9844868317511343, between its q95 and q99
thresholds.  This is design rationale, not a population-shift guarantee:
policy-v4 must still pass every unchanged gate on fresh design and validation
populations before any release run.

This produces development evidence.  It is not release evidence.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
V2 = HERE.parent
ZPAB = V2.parent
TRAINING = ZPAB.parent
for _path in (TRAINING, ZPAB, V2, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import fit_v3_openset as base  # noqa: E402
import run_time_domain_dev as dev  # noqa: E402
import time_domain_geometry as geometry  # noqa: E402
import time_domain_invariant_patch_preprocess as td_preprocess  # noqa: E402
from openset_eval import NOVELTY  # noqa: E402
from v3_time_domain_openset import (  # noqa: E402
    FROZEN_GEOMETRY_FEATURE,
    FROZEN_POLICY_KIND,
    empirical_rank,
)


# ---------------------------------------------------------------------------
# inherited, never redefined
# ---------------------------------------------------------------------------

BRANCH_LOF_RANK_WEIGHT = base.BRANCH_LOF_RANK_WEIGHT
GEOMETRY_RANK_WEIGHT = base.GEOMETRY_RANK_WEIGHT
THRESHOLD_QUANTILE = base.THRESHOLD_QUANTILE
BRANCH_LOF = base.BRANCH_LOF

# Gates: imported objects, not re-typed values, so they cannot drift.
GATE_FLOORS = base.GATE_FLOORS
KNOWN_FUR_CEILING = base.KNOWN_FUR_CEILING

CENTER_SPLIT = base.CENTER_SPLIT
PROTOTYPE_SPLIT = base.PROTOTYPE_SPLIT
METRIC_SPLIT = base.METRIC_SPLIT

NOVELTY_FAMILIES = base.NOVELTY_FAMILIES
DEFAULT_NOVELTY_N = base.DEFAULT_NOVELTY_N
DEFAULT_PREFIX_LENGTHS = base.DEFAULT_PREFIX_LENGTHS
DEFAULT_REPRODUCTION_TOLERANCE = base.DEFAULT_REPRODUCTION_TOLERANCE

reject_sealed_path = base.reject_sealed_path
write_json = base.write_json
validate_prefix_lengths = base.validate_prefix_lengths
load_fusion_artifact = base.load_fusion_artifact
resolve_device = base.resolve_device
fit_branch_lof = base.fit_branch_lof
score_branch_lof = base.score_branch_lof
save_branch_lof = base.save_branch_lof
fit_stage_two_policy = base.fit_policy
assert_closed_label_unchanged = base.assert_closed_label_unchanged
family_metrics = base.family_metrics
gate_summary = base.gate_summary
FusionArtifact = base.FusionArtifact
Populations = base.Populations

_sha256 = base._sha256


# ---------------------------------------------------------------------------
# the novelty seed ledger
# ---------------------------------------------------------------------------

# Spent, with what spent them.  None of these may be drawn again, in either
# role: a seed that has already been looked at cannot validate anything.
SPENT_NOVELTY_SEEDS: dict[int, str] = {
    20260938: "designed the frozen 0.80/0.20 additive policy",
    20260939: (
        "v3 open-set refit validation (HANDOFF 17) and pose-degeneracy feature "
        "scoring (HANDOFF 19.1)"
    ),
    20260940: (
        "v3 open-set refit validation (HANDOFF 17) and pose-degeneracy feature "
        "scoring (HANDOFF 19.1)"
    ),
    20260941: "pose-degeneracy blend-weight design (HANDOFF 19.2)",
    20260942: (
        "validated the staged v3.0 policy (HANDOFF 20.2) and its "
        "causal-prefix-rule revalidation; that policy was then sealed at "
        "release seed 20260731"
    ),
    20260943: (
        "validated the staged v3.0 policy (HANDOFF 20.2) and its "
        "causal-prefix-rule revalidation; that policy was then sealed at "
        "release seed 20260731"
    ),
    20260944: (
        "validated the v3.1 tightened-budget candidate (stage-1 budget 0.01): "
        "frozen FAIL, worst noise AUROC cell 0.7957 vs the 0.80 floor"
    ),
    20260945: (
        "validated the v3.1 tightened-budget candidate (stage-1 budget 0.01): "
        "frozen FAIL, worst noise AUROC cell 0.7957 vs the 0.80 floor"
    ),
    20260946: (
        "designed the v3.2 composite-survivor policy; its choices are frozen"
    ),
    20260947: (
        "validated the v3.2 composite-survivor policy; validation evidence is "
        "consumed and may not be reused"
    ),
    20260948: (
        "validated the v3.2 composite-survivor policy; validation evidence is "
        "consumed and may not be reused"
    ),
    20260949: (
        "designed the 8k-regularized v3.3 composite candidate; frozen "
        "design_selection_fail because chirp failed at N4096/N8192. This "
        "spent design population may inform a documented design refinement "
        "but is never independent validation evidence"
    ),
    20260950: (
        "validated the frozen v3.3 decoupled 8k-classifier/4k-rejector "
        "candidate: development_openset_pass; validation evidence is consumed "
        "and may not be reused"
    ),
    20260951: (
        "validated the frozen v3.3 decoupled 8k-classifier/4k-rejector "
        "candidate: development_openset_pass; validation evidence is consumed "
        "and may not be reused"
    ),
    20260952: (
        "designed the frozen policy-v3/q99 v3.4 candidate: "
        "design_selection_fail; only N4096 chirp threshold recall failed "
        "(1/300 = 0.0033333333333333335 < 0.10), while known false-unknown "
        "was 38/1908 = 0.019916142557651992. The seed is consumed and may not "
        "be redrawn"
    ),
}

# Consumed sealed release suites (evidence rule 2): never re-run, never fit,
# calibrate, select or debug against.  The base module's older constants remain
# refused through its validator as well; this moving ledger is authoritative
# for every sealed attempt made since those modules were frozen.
CONSUMED_SEALED_RELEASE_SEEDS: dict[int, str] = {
    20260729: "consumed sealed v2 release suite",
    20260731: (
        "consumed sealed v3.0 release suite (HANDOFF 25: 22/23 gates, known "
        "false-unknown failure frozen)"
    ),
    20260733: (
        "consumed sealed v3.2 release suite (21/23 gates; known false-unknown "
        "and five-shot failures frozen)"
    ),
    20260734: (
        "consumed sealed v3.2 release suite under its predeclared historical "
        "gate redeclaration (22/23 gates; five-shot failure frozen)"
    ),
    20260735: (
        "consumed sealed v3.3 decoupled 8k-classifier/4k-rejector release "
        "suite (22/23 gates; known false-unknown failure frozen)"
    ),
}
RELEASE_SEED_NEVER_SPENT_HERE = 20260736

# Seeds through 20260952 are consumed.  The policy-v3/q99 design failure is
# frozen and may not be redrawn.  Seeds 20260953/20260954 remain untouched and
# reserved for validation; policy-v4/q97 uses the next unreserved clean seed,
# 20260955, for its one design draw.
FIRST_CLEAN_NOVELTY_SEED = 20260953
PROPOSED_DESIGN_NOVELTY_SEED = 20260955
DEFAULT_VALIDATION_NOVELTY_SEEDS = (20260953, 20260954)
V34_DESIGN_PREFIX_LENGTHS = (4096, 8192, 16384, 32768)

ROLES = ("design", "validate")

SEED_LEDGER_NOTE = (
    "Novelty seeds are consumable evidence. "
    f"{sorted(SPENT_NOVELTY_SEEDS)} are spent and are refused in both roles; "
    f"{sorted(CONSUMED_SEALED_RELEASE_SEEDS)} are consumed sealed suites and "
    f"{RELEASE_SEED_NEVER_SPENT_HERE} is the next untouched release seed, and "
    f"none of them is a development novelty seed. {FIRST_CLEAN_NOVELTY_SEED} "
    "onward are clean except for the explicit reservations below. The frozen "
    "policy-v3/q99 design consumed seed 20260952 and failed only N4096 chirp "
    "threshold recall. The predeclared policy-v4/q97 candidate reserves clean "
    f"design seed {PROPOSED_DESIGN_NOVELTY_SEED} and validation seeds "
    f"{list(DEFAULT_VALIDATION_NOVELTY_SEEDS)}. The design seed is intentionally "
    "not in SPENT_NOVELTY_SEEDS yet. Validation is refused until a ledger-only "
    "transition marks that design seed spent, and drawing either validation "
    "seed during design would violate evidence rule 4."
)


def v34_design_and_validation_commands(
    *,
    fusion_dir: Path,
    prefilter_dir: Path,
    design_output_dir: Path,
    validation_output_dir: Path,
    device: str = "cpu",
    novelty_n: int = DEFAULT_NOVELTY_N,
    prefix_lengths: Sequence[int] = V34_DESIGN_PREFIX_LENGTHS,
) -> dict[str, list[str]]:
    """Return the exact future commands without drawing either population.

    The design command is the only command currently admissible.  The
    validation command is predeclared here, but :func:`validate_seed_plan`
    refuses it until seed 20260955 has been run exactly once and a ledger-only
    transition records that design seed as spent.  Seeds 20260953/20260954
    remain untouched while this command is merely constructed; constructing
    argv reads no data and consumes no seed.
    """
    common = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--fusion-dir",
        str(Path(fusion_dir)),
        "--prefilter-dir",
        str(Path(prefilter_dir)),
        "--device",
        str(device),
        "--novelty-n",
        str(int(novelty_n)),
        "--prefix-lengths",
        *[str(int(length)) for length in prefix_lengths],
    ]
    return {
        "design": [
            *common,
            "--output-dir",
            str(Path(design_output_dir)),
            "--role",
            "design",
            "--design-novelty-seed",
            str(PROPOSED_DESIGN_NOVELTY_SEED),
            "--novelty-seeds",
            str(PROPOSED_DESIGN_NOVELTY_SEED),
        ],
        "validation_after_design_ledger_transition": [
            *common,
            "--output-dir",
            str(Path(validation_output_dir)),
            "--role",
            "validate",
            "--design-novelty-seed",
            str(PROPOSED_DESIGN_NOVELTY_SEED),
            "--novelty-seeds",
            *[str(seed) for seed in DEFAULT_VALIDATION_NOVELTY_SEEDS],
        ],
    }


# ---------------------------------------------------------------------------
# the sibling modules (imported lazily and loudly)
# ---------------------------------------------------------------------------

PREFILTER_MODULE_DEFAULT = "noise_prefilter"
POSEDEGEN_MODULE_DEFAULT = "pose_degeneracy"

#: Module-level names :mod:`noise_prefilter` must expose for this module to be
#: able to stage anything at all.
REQUIRED_PREFILTER_API = (
    "NOISE_PREFILTER_VERSION",
    "PREFILTER_FEATURES",
    "ARCHITECTURE_CONTRACT_CHANGE",
    "CONTRACT_KEYS_FOR_RELEASE_CLAIM",
    "NoisePrefilter",
    "load_prefilter_set",
    "prefilter_set_sha256",
    "compute_saving",
    "validate_fitting_seed",
)

#: What one fitted, thresholded bundle must expose.
REQUIRED_MODEL_API = (
    "capture_length",
    "feature_names",
    "has_threshold",
    "threshold_score",
    "score",
    "decide",
    "metadata",
)

#: The pose-degeneracy extractor the prefilter's features are drawn from.
REQUIRED_POSEDEGEN_API = (
    "POSE_DEGENERACY_VERSION",
    "FEATURE_NAMES",
    "FEATURE_COUNT",
    "pose_degeneracy_features",
)

NOISE_PREFILTER_CONTRACT = (
    "the noise prefilter module must expose:\n"
    "  NOISE_PREFILTER_VERSION: str, non-empty\n"
    "  PREFILTER_FEATURES: the pose-degeneracy feature subset it consumes\n"
    "  ARCHITECTURE_CONTRACT_CHANGE, CONTRACT_KEYS_FOR_RELEASE_CLAIM: the\n"
    "      statement and key set that record that this gate can suppress the\n"
    "      closed label\n"
    "  NoisePrefilter: the fitted detector class\n"
    "  load_prefilter_set(directory) -> {capture_length: NoisePrefilter}\n"
    "  prefilter_set_sha256(directory) -> str\n"
    "  compute_saving(gated) -> dict\n"
    "  validate_fitting_seed(seed) -> int, refusing every ledger seed\n"
    "and each loaded model must expose capture_length, feature_names,\n"
    "has_threshold, threshold_score, score(features, capture_length=...),\n"
    "decide(features, capture_length=...) where True means NOISE, and\n"
    "metadata().  This module LOADS a fitted set; it never fits one, because\n"
    "the fitting protocol -- training-only coefficients, an enrollment-only\n"
    "operating point and a fitting seed drawn from a namespace disjoint from\n"
    "every novelty seed -- belongs to the prefilter module and must stay in one\n"
    "place."
)

POSEDEGEN_CONTRACT = (
    "the pose-degeneracy module must expose POSE_DEGENERACY_VERSION: str, "
    "FEATURE_NAMES: Sequence[str], FEATURE_COUNT: int, and "
    "pose_degeneracy_features(iq, *, min_bandwidth=...) -> float64 vector of "
    "shape (FEATURE_COUNT,) for ONE raw complex I/Q capture.  It is called once "
    "per capture rather than once per batch because the v3 frontend's "
    "resolution floor min_bandwidth depends on the capture length, and the "
    "features must describe the pose the frontend actually adopted."
)

# The frontend calls estimate_band with a length-dependent resolution floor
# (time_domain_geometry.minimum_bandwidth_for_active_span).  Stage-1 features are
# computed with the SAME floor, so they describe the pose the classifier actually
# used rather than a different pose computed under the estimator's library
# default.  A module constant with no flag: a correctness requirement, not a
# tuning knob.
PREFILTER_MATCH_FRONTEND_MIN_BANDWIDTH = True

# Gated rows are placed above every ungated row on a single staged axis, so the
# reported AUROC is the AUROC of the staged decision rather than of a stage read
# in isolation.  Stage-2 scores are empirical ranks in [0, 1); a gated row gets
# 1.0 + a bounded monotone map of its stage-1 log-odds, also in (0, 1), so gated
# rows occupy (1, 2).
STAGE_ONE_SCORE_OFFSET = 1.0

# Surviving rows are scored by stage 2 in a smaller batch than the unstaged arm
# uses.  Scoring is row-independent, so the two must agree; this is the tolerance
# on that assertion and the report records whether it was exactly 0.
SUBSET_SCORING_TOLERANCE = 1e-6

# ---------------------------------------------------------------------------
# STAGED POLICY -- FROZEN CONSTANTS.  Version 4: q97 composite survivor.
#
# Every artifact this module emits states the policy version it validates, and
# the sealed evaluator refuses a staged artifact whose recorded version is not
# the one below.  History, so a reader of version N can find what changed:
#
#   1  "v3_staged_noise_prefilter_then_known_only_lof_geometry": survivors
#      scored by the stage-2 enrollment rank alone; staged threshold = the
#      stage-2 policy's own enrollment q95.  Sealed at release seed 20260731;
#      failure frozen (known FUR 0.10237 vs 0.10 at N32768).  The v3.1
#      variant (same policy, stage-1 budget 0.02 -> 0.01) froze a dev FAIL:
#      noise AUROC worst cell 0.7957 vs 0.80 (seeds 20260944/20260945).
#   2  survivors scored by the COMPOSITE
#      max(stage2_enrollment_rank, stage1_score_enrollment_rank); the staged
#      unknown threshold is re-fit at the SAME frozen quantile on the
#      composite over enrollment stage-1 survivors.  The stage-1 gate (0.01
#      enrollment budget set), prefilter coefficients, LOF ensemble, blend
#      weights and every stage-2 internal are untouched.
#   3  changes ONLY that composite-survivor enrollment quantile from q95 to
#      q99.  Its one design draw (20260952) failed only N4096 chirp threshold
#      recall, 0.003333 < 0.10; known FUR was 0.019916.  Frozen failure.
#   4  (this file) changes ONLY the failed q99 survivor quantile to the
#      predeclared q97.  The 1% stage-1 gate, both CNN fusions, score,
#      calibration population, rank conventions, LOF, blend weights and every
#      gate floor/ceiling remain frozen.
# ---------------------------------------------------------------------------
#: The survivor score, stated once.  ``stage1_score_enrollment_rank`` is the
#: empirical enrollment rank (searchsorted-left over a sorted calibration, the
#: same :func:`v3_time_domain_openset.empirical_rank` convention stage 2 uses)
#: of the stage-1 logistic log-odds, with the calibration fit on ENROLLMENT
#: STAGE-1 SURVIVORS only.
COMPOSITE_SURVIVOR_SCORE = (
    "max(stage2_enrollment_rank, stage1_score_enrollment_rank)"
)

LEGACY_STAGED_POLICY_SCHEMA = 2
LEGACY_STAGED_POLICY_VERSION = (
    "v3-staged-openset-policy-v2-composite-survivor"
)
LEGACY_STAGED_POLICY_KIND = (
    "v3_staged_noise_prefilter_then_composite_survivor_lof_geometry"
)
LEGACY_COMPOSITE_THRESHOLD_QUANTILE = THRESHOLD_QUANTILE
LEGACY_COMPOSITE_RATIONALE = (
    "stage-1's continuous log-odds separates noise from known captures at "
    "0.978 AUROC in-sample but was discarded below the gate; the composite "
    "uses that score without moving the gate. The frozen v3.1 evidence "
    "(seeds 20260944/20260945) showed that tightening the gate alone pushes "
    "noise into stage 2, whose own noise axis is weak (~0.60 AUROC "
    "standalone), costing the noise AUROC floor (worst cell 0.7957 vs 0.80)"
)

FAILED_Q99_STAGED_POLICY_SCHEMA = 3
FAILED_Q99_STAGED_POLICY_VERSION = (
    "v3-staged-openset-policy-v3-composite-survivor-q99"
)
FAILED_Q99_STAGED_POLICY_KIND = (
    "v3_staged_noise_prefilter_then_q99_composite_survivor_lof_geometry"
)
FAILED_Q99_COMPOSITE_THRESHOLD_QUANTILE = 0.99
FAILED_Q99_COMPOSITE_RATIONALE = (
    "Policy-v3 changed only the composite survivor enrollment quantile q95 "
    "-> q99. Its one design draw, seed 20260952, is frozen and consumed: "
    "known false-unknown was 0.019916, but N4096 chirp threshold recall was "
    "0.003333 against the unchanged 0.10 floor. It failed design and may not "
    "be redrawn or validated"
)

#: Policy-v4's only operating-point change.  This literal is predeclared
#: before design seed 20260955 is drawn; it is not searched on selection,
#: novelty, sealed or release data.
COMPOSITE_THRESHOLD_QUANTILE = 0.97
STAGE_ONE_KNOWN_FALSE_POSITIVE_BUDGET = 0.01
SURVIVOR_KNOWN_FALSE_POSITIVE_BUDGET = (
    1.0 - COMPOSITE_THRESHOLD_QUANTILE
)
NOMINAL_ENROLLMENT_FALSE_UNKNOWN_BUDGET = (
    STAGE_ONE_KNOWN_FALSE_POSITIVE_BUDGET
    + (1.0 - STAGE_ONE_KNOWN_FALSE_POSITIVE_BUDGET)
    * SURVIVOR_KNOWN_FALSE_POSITIVE_BUDGET
)
FROZEN_POLICY_V2_Q95_ENROLLMENT_THRESHOLD = 0.9740452030878427
FROZEN_POLICY_V4_Q97_ENROLLMENT_THRESHOLD = 0.9844868317511343
FROZEN_POLICY_V3_Q99_ENROLLMENT_THRESHOLD = 0.9945961538794619
if not (
    FROZEN_POLICY_V2_Q95_ENROLLMENT_THRESHOLD
    < FROZEN_POLICY_V4_Q97_ENROLLMENT_THRESHOLD
    < FROZEN_POLICY_V3_Q99_ENROLLMENT_THRESHOLD
):
    raise AssertionError("frozen enrollment thresholds must satisfy q95 < q97 < q99")

STAGED_POLICY_SCHEMA = 4
STAGED_POLICY_VERSION = (
    "v3-staged-openset-policy-v4-composite-survivor-q97"
)
STAGED_POLICY_KIND = (
    "v3_staged_noise_prefilter_then_q97_composite_survivor_lof_geometry"
)
COMPOSITE_RATIONALE = (
    "The frozen policy-v3/q99 design failed only N4096 chirp threshold recall "
    "(0.003333 < 0.10), with known false-unknown 0.019916. Policy-v4 changes "
    "only that survivor enrollment quantile q99 -> q97, a conservative "
    "enrollment-only balance. The 1% stage-1 budget followed by a 3% survivor "
    "tail has a nominal 3.97% union budget, leaving more than 2.5x headroom "
    "to the unchanged 10% known-FUR ceiling. Stage 1, both CNN fusions, the "
    "survivor score and every gate remain frozen"
)


@dataclass(frozen=True)
class StagedPolicySpec:
    """Exact serialization and operating-point semantics for one version."""

    schema: int
    version: str
    kind: str
    threshold_quantile: float
    rationale: str
    only_policy_change: str


LEGACY_STAGED_POLICY_SPEC = StagedPolicySpec(
    schema=LEGACY_STAGED_POLICY_SCHEMA,
    version=LEGACY_STAGED_POLICY_VERSION,
    kind=LEGACY_STAGED_POLICY_KIND,
    threshold_quantile=LEGACY_COMPOSITE_THRESHOLD_QUANTILE,
    rationale=LEGACY_COMPOSITE_RATIONALE,
    only_policy_change="legacy q95 composite survivor policy",
)
FAILED_Q99_STAGED_POLICY_SPEC = StagedPolicySpec(
    schema=FAILED_Q99_STAGED_POLICY_SCHEMA,
    version=FAILED_Q99_STAGED_POLICY_VERSION,
    kind=FAILED_Q99_STAGED_POLICY_KIND,
    threshold_quantile=FAILED_Q99_COMPOSITE_THRESHOLD_QUANTILE,
    rationale=FAILED_Q99_COMPOSITE_RATIONALE,
    only_policy_change=(
        "composite survivor enrollment threshold quantile q95 -> q99"
    ),
)
STAGED_POLICY_SPEC = StagedPolicySpec(
    schema=STAGED_POLICY_SCHEMA,
    version=STAGED_POLICY_VERSION,
    kind=STAGED_POLICY_KIND,
    threshold_quantile=COMPOSITE_THRESHOLD_QUANTILE,
    rationale=COMPOSITE_RATIONALE,
    only_policy_change=(
        "composite survivor enrollment threshold quantile q99 -> q97"
    ),
)
STAGED_POLICY_SPECS = {
    LEGACY_STAGED_POLICY_VERSION: LEGACY_STAGED_POLICY_SPEC,
    FAILED_Q99_STAGED_POLICY_VERSION: FAILED_Q99_STAGED_POLICY_SPEC,
    STAGED_POLICY_VERSION: STAGED_POLICY_SPEC,
}

STAGED_POLICY_VERSION_HISTORY = {
    "1": (
        "v3_staged_noise_prefilter_then_known_only_lof_geometry: survivor "
        "score = stage-2 enrollment rank; threshold = stage-2 enrollment q95. "
        "Sealed seed 20260731, known-FUR failure frozen; 0.01-budget variant "
        "(v3.1) froze a noise-AUROC dev failure on seeds 20260944/20260945"
    ),
    "2": (
        "composite survivor score max(stage2_enrollment_rank, "
        "stage1_score_enrollment_rank); staged threshold = q95 of the "
        "composite on enrollment stage-1 survivors; gate, prefilter "
        "coefficients, LOF ensemble, blend weights and stage-2 internals "
        "untouched"
    ),
    "3": (
        "v3.4: identical composite score and enrollment-survivor calibration; "
        "only the predeclared threshold quantile changes q95 -> q99. Frozen "
        "design seed 20260952 failed only N4096 chirp threshold recall "
        "(0.003333 < 0.10); known FUR 0.019916"
    ),
    "4": (
        "identical composite score and enrollment-survivor calibration; only "
        "the predeclared threshold quantile changes failed q99 -> q97. The 1% "
        "stage-1 gate, both CNN fusions and every gate remain frozen; nominal "
        "enrollment union budget 3.97%, >2.5x headroom to 10%"
    ),
}

#: The serialized composite policy, alongside the two stage-2 npz files.
COMPOSITE_POLICY_FILENAME = "v3_staged_composite_policy.npz"


def _bounded_rank(scores: np.ndarray) -> np.ndarray:
    """A strictly increasing map from the real line into ``(0, 1)``.

    Softsign, ``x / (1 + |x|)``, rescaled.  Deliberately not the logistic: stage
    1 scores are log-odds and the logistic saturates to exactly 1.0 in float64
    by about x = 37, which would tie rows that are genuinely ordered.  Softsign
    stays strictly increasing out to about 1e16, so the staged axis preserves
    the stage-1 ordering over any range these features produce.
    """
    value = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError("stage-1 scores must be finite")
    return 0.5 * (value / (1.0 + np.abs(value))) + 0.5


def _load_module(name: str, required: Sequence[str], contract: str, what: str):
    """Import ``name`` lazily and refuse it unless it satisfies ``required``."""
    module_name = str(name)
    if not module_name:
        raise ValueError(f"the {what} module name must be non-empty")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"the {what} module {module_name!r} is not importable ({exc}).  It "
            "is required: this staged rejector will not silently degrade to the "
            f"unstaged additive policy.\n{contract}"
        ) from exc
    missing = [item for item in required if not hasattr(module, item)]
    if missing:
        raise RuntimeError(
            f"{what} module {module_name!r} is missing {missing}.\n{contract}"
        )
    return module


@dataclass(frozen=True)
class PoseDegeneracySource:
    """A validated pose-degeneracy extractor."""

    module_name: str
    module: Any
    version: str
    feature_names: tuple[str, ...]
    source_path: str | None
    source_sha256: str | None

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    def provenance(self) -> dict[str, Any]:
        metadata_fn = getattr(self.module, "pose_degeneracy_metadata", None)
        return {
            "module": self.module_name,
            "version": self.version,
            "feature_count": self.feature_count,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "min_bandwidth_matches_frontend": bool(
                PREFILTER_MATCH_FRONTEND_MIN_BANDWIDTH
            ),
            "extractor_metadata": (
                metadata_fn() if callable(metadata_fn) else None
            ),
        }


def load_posedegen_module(
    module_name: str = POSEDEGEN_MODULE_DEFAULT,
) -> PoseDegeneracySource:
    """Import and validate the pose-degeneracy extractor."""
    module = _load_module(
        module_name, REQUIRED_POSEDEGEN_API, POSEDEGEN_CONTRACT, "pose-degeneracy"
    )
    if not callable(getattr(module, "pose_degeneracy_features")):
        raise RuntimeError(
            f"pose-degeneracy module {module_name!r} exposes a non-callable "
            f"pose_degeneracy_features.\n{POSEDEGEN_CONTRACT}"
        )
    raw_names = getattr(module, "FEATURE_NAMES")
    if isinstance(raw_names, (str, bytes)):
        raise RuntimeError(
            f"pose-degeneracy module {module_name!r} exposes FEATURE_NAMES as a "
            f"string, not a sequence.\n{POSEDEGEN_CONTRACT}"
        )
    feature_names = tuple(str(item) for item in raw_names)
    if not feature_names or len(set(feature_names)) != len(feature_names):
        raise RuntimeError(
            f"pose-degeneracy module {module_name!r} must declare a non-empty "
            f"set of unique feature names.\n{POSEDEGEN_CONTRACT}"
        )
    if int(getattr(module, "FEATURE_COUNT")) != len(feature_names):
        raise RuntimeError(
            f"pose-degeneracy module {module_name!r} declares FEATURE_COUNT "
            f"{getattr(module, 'FEATURE_COUNT')!r} for {len(feature_names)} "
            "feature names"
        )
    version = str(getattr(module, "POSE_DEGENERACY_VERSION"))
    if not version:
        raise RuntimeError(
            f"pose-degeneracy module {module_name!r} declares an empty "
            f"POSE_DEGENERACY_VERSION.\n{POSEDEGEN_CONTRACT}"
        )
    source_path = getattr(module, "__file__", None)
    return PoseDegeneracySource(
        module_name=str(module_name),
        module=module,
        version=version,
        feature_names=feature_names,
        source_path=str(source_path) if source_path else None,
        source_sha256=(
            _sha256(Path(source_path))
            if source_path and Path(source_path).is_file()
            else None
        ),
    )


@dataclass(frozen=True)
class NoisePrefilterSource:
    """A validated noise-prefilter module."""

    module_name: str
    module: Any
    version: str
    feature_names: tuple[str, ...]
    source_path: str | None
    source_sha256: str | None

    def provenance(self) -> dict[str, Any]:
        return {
            "module": self.module_name,
            "version": self.version,
            "feature_names": list(self.feature_names),
            "feature_count": len(self.feature_names),
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "architecture_contract_change": str(
                getattr(self.module, "ARCHITECTURE_CONTRACT_CHANGE")
            ),
            "contract_keys_for_release_claim": list(
                getattr(self.module, "CONTRACT_KEYS_FOR_RELEASE_CLAIM")
            ),
            "fitted_here": False,
            "fitted_by": (
                "noise_prefilter.py's own CLI, which owns the training-only "
                "coefficient fit, the enrollment-only operating point and the "
                "fitting-seed ledger"
            ),
        }


def load_prefilter_module(
    module_name: str = PREFILTER_MODULE_DEFAULT,
) -> NoisePrefilterSource:
    """Import and validate the sibling noise-prefilter module.

    Deliberately lazy and deliberately loud.  If the module is missing or does
    not satisfy :data:`REQUIRED_PREFILTER_API`, this raises instead of falling
    back to the unstaged rejector: a silent fallback would emit a report that
    looks like a staged result and is not one.
    """
    module = _load_module(
        module_name,
        REQUIRED_PREFILTER_API,
        NOISE_PREFILTER_CONTRACT,
        "noise prefilter",
    )
    for item in (
        "load_prefilter_set",
        "prefilter_set_sha256",
        "compute_saving",
        "validate_fitting_seed",
    ):
        if not callable(getattr(module, item)):
            raise RuntimeError(
                f"noise prefilter module {module_name!r} exposes a non-callable "
                f"{item}.\n{NOISE_PREFILTER_CONTRACT}"
            )
    raw_names = getattr(module, "PREFILTER_FEATURES")
    if isinstance(raw_names, (str, bytes)):
        raise RuntimeError(
            f"noise prefilter module {module_name!r} exposes PREFILTER_FEATURES "
            f"as a string, not a sequence.\n{NOISE_PREFILTER_CONTRACT}"
        )
    feature_names = tuple(str(item) for item in raw_names)
    if not feature_names or len(set(feature_names)) != len(feature_names):
        raise RuntimeError(
            f"noise prefilter module {module_name!r} must declare a non-empty "
            f"set of unique features.\n{NOISE_PREFILTER_CONTRACT}"
        )
    version = str(getattr(module, "NOISE_PREFILTER_VERSION"))
    if not version:
        raise RuntimeError(
            f"noise prefilter module {module_name!r} declares an empty "
            f"NOISE_PREFILTER_VERSION.\n{NOISE_PREFILTER_CONTRACT}"
        )
    source_path = getattr(module, "__file__", None)
    return NoisePrefilterSource(
        module_name=str(module_name),
        module=module,
        version=version,
        feature_names=feature_names,
        source_path=str(source_path) if source_path else None,
        source_sha256=(
            _sha256(Path(source_path))
            if source_path and Path(source_path).is_file()
            else None
        ),
    )


def frontend_min_bandwidth(
    raw_length: int,
    patch_length: int,
    target_frac: float,
) -> float | None:
    """The resolution floor the v3 frontend itself passes to ``estimate_band``."""
    if not PREFILTER_MATCH_FRONTEND_MIN_BANDWIDTH:
        return None
    return geometry.minimum_bandwidth_for_active_span(
        int(raw_length), int(patch_length), float(target_frac)
    )


def prefilter_row(
    posedegen: PoseDegeneracySource,
    capture: np.ndarray,
    *,
    patch_length: int,
    target_frac: float,
) -> np.ndarray:
    """The full pose-degeneracy vector for one capture, at the frontend's floor.

    The full vector is computed, not the prefilter's seven-feature subset, and
    the model slices it itself.  That way the column order the detector was
    fitted with is chosen by the detector, and a feature-selection change in the
    prefilter cannot silently misalign the columns handed to it.
    """
    minimum = frontend_min_bandwidth(len(capture), patch_length, target_frac)
    kwargs = {} if minimum is None else {"min_bandwidth": float(minimum)}
    return np.asarray(
        posedegen.module.pose_degeneracy_features(capture, **kwargs),
        dtype=np.float64,
    )


def prefilter_matrix(
    posedegen: PoseDegeneracySource,
    captures: Sequence[np.ndarray],
    *,
    patch_length: int,
    target_frac: float,
    population: str,
) -> np.ndarray:
    """Stack :func:`prefilter_row` over an ordered population, and validate it."""
    rows = list(captures)
    if not rows:
        raise ValueError(f"{population}: the prefilter needs at least one row")
    value = np.stack(
        [
            prefilter_row(
                posedegen,
                capture,
                patch_length=patch_length,
                target_frac=target_frac,
            )
            for capture in rows
        ]
    ).astype(np.float64)
    expected = (len(rows), posedegen.feature_count)
    if value.shape != expected:
        raise RuntimeError(
            f"{population}: {posedegen.module_name}.pose_degeneracy_features "
            f"produced shape {value.shape}, expected {expected}.\n"
            f"{POSEDEGEN_CONTRACT}"
        )
    if not np.isfinite(value).all():
        raise RuntimeError(
            f"{population}: {posedegen.module_name}.pose_degeneracy_features "
            "returned a non-finite value"
        )
    return value


def _matrix_sha256(value: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    return hashlib.sha256(canonical.view(np.uint8)).hexdigest()



# ---------------------------------------------------------------------------
# hyperparameter and seed hygiene
# ---------------------------------------------------------------------------


def validate_role(role: Any) -> str:
    value = str(role)
    if value not in ROLES:
        raise ValueError(f"--role must be one of {list(ROLES)}, got {value!r}")
    return value



def _refuse_ledger_seed(seed: int, role_description: str) -> None:
    if seed in SPENT_NOVELTY_SEEDS:
        raise ValueError(
            f"novelty seed {seed} is SPENT ({SPENT_NOVELTY_SEEDS[seed]}) and "
            f"may not be used as {role_description}.  {FIRST_CLEAN_NOVELTY_SEED} "
            "onward are clean."
        )
    if seed in CONSUMED_SEALED_RELEASE_SEEDS:
        raise ValueError(
            f"{seed} is a consumed sealed release suite "
            f"({CONSUMED_SEALED_RELEASE_SEEDS[seed]}) and may not be used for "
            "anything"
        )
    if seed == RELEASE_SEED_NEVER_SPENT_HERE:
        raise ValueError(
            f"{seed} is the unspent release seed and is not a development "
            "novelty seed"
        )


def validate_novelty_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    """Refuse an empty, duplicated, spent, sealed or release novelty seed set."""
    raw_values = tuple(seeds)
    for seed in raw_values:
        if isinstance(seed, (bool, np.bool_)) or not isinstance(
            seed, (int, np.integer)
        ):
            raise ValueError("novelty seeds must be integers, not coerced values")
    values = tuple(int(seed) for seed in raw_values)
    if not values or len(set(values)) != len(values):
        raise ValueError("novelty seeds must be a non-empty set of distinct ints")
    for seed in values:
        _refuse_ledger_seed(seed, "development novelty")
        if seed < FIRST_CLEAN_NOVELTY_SEED:
            raise ValueError(
                f"novelty seed {seed} predates the authoritative clean boundary "
                f"{FIRST_CLEAN_NOVELTY_SEED} and is refused even though it is "
                "not named in the local spent ledger"
            )
    # Belt and braces: the base module's own refusals must also hold.
    base.validate_novelty_seeds(values)
    return values


def validate_seed_plan(
    role: Any,
    design_novelty_seed: Any,
    novelty_seeds: Sequence[int],
) -> tuple[str, int, tuple[int, ...]]:
    """Return ``(role, design_seed, novelty_seeds)`` or refuse to run.

    A design run must draw a fresh design seed.  A later validation run only
    *references* that now-frozen design seed, so it may be present in the
    spent ledger; it never draws that seed again.  The novelty seeds actually
    drawn by validation must remain fresh.
    """
    role_value = validate_role(role)
    if design_novelty_seed is None:
        raise ValueError(
            "--design-novelty-seed is required: the report must record which "
            "novelty seed the staged system was designed on.  Proposed fresh "
            f"seed: {PROPOSED_DESIGN_NOVELTY_SEED}."
        )
    if isinstance(design_novelty_seed, (bool, np.bool_)) or not isinstance(
        design_novelty_seed, (int, np.integer)
    ):
        raise ValueError("--design-novelty-seed must be an integer")
    design = int(design_novelty_seed)
    if role_value == "design":
        _refuse_ledger_seed(design, "a design seed")
        if design != PROPOSED_DESIGN_NOVELTY_SEED:
            raise ValueError(
                f"the current design seed is reserved as "
                f"{PROPOSED_DESIGN_NOVELTY_SEED}; got {design}"
            )
    else:
        if design != PROPOSED_DESIGN_NOVELTY_SEED:
            raise ValueError(
                f"the current design seed is reserved as "
                f"{PROPOSED_DESIGN_NOVELTY_SEED}; got {design}"
            )
        if design not in SPENT_NOVELTY_SEEDS:
            raise ValueError(
                f"design seed {design} is not yet frozen in "
                "SPENT_NOVELTY_SEEDS. Run the predeclared design command "
                "exactly once, freeze its result, then make the ledger-only "
                "transition before validation"
            )
        if design in CONSUMED_SEALED_RELEASE_SEEDS:
            raise ValueError(
                f"{design} is a consumed sealed release suite and cannot be "
                "referenced as a development design seed"
            )
        if design == RELEASE_SEED_NEVER_SPENT_HERE:
            raise ValueError(
                f"{design} is the unspent release seed and cannot be "
                "referenced as a development design seed"
            )
    if design in DEFAULT_VALIDATION_NOVELTY_SEEDS:
        raise ValueError(
            f"design novelty seed {design} is a default validation seed "
            f"{list(DEFAULT_VALIDATION_NOVELTY_SEEDS)}.  Choosing the stage-1 "
            "design there converts validation evidence into training "
            "evidence, which evidence rule 4 forbids."
        )

    seeds = validate_novelty_seeds(novelty_seeds)

    if role_value == "validate":
        if design in seeds:
            raise ValueError(
                f"design novelty seed {design} is also a validation novelty "
                "seed in this run.  The seed a design was chosen on cannot "
                "be the seed that validates it."
            )
        if seeds != DEFAULT_VALIDATION_NOVELTY_SEEDS:
            raise ValueError(
                "a validation run must draw exactly the predeclared reserved "
                f"novelty seeds {list(DEFAULT_VALIDATION_NOVELTY_SEEDS)} in "
                f"that order; got {list(seeds)}"
            )
    else:
        overlap = [
            seed for seed in seeds if seed in DEFAULT_VALIDATION_NOVELTY_SEEDS
        ]
        if overlap:
            raise ValueError(
                f"a design run may not draw novelty at reserved validation "
                f"seeds {overlap}; they stay untouched for validation"
            )
        if design not in seeds:
            raise ValueError(
                f"a design run must draw novelty at its declared design seed "
                f"{design}; got {list(seeds)}"
            )
        if seeds != (design,):
            raise ValueError(
                "a design run must draw exactly its declared design novelty "
                f"seed [{design}]; got {list(seeds)}"
            )
    return role_value, design, seeds


# ---------------------------------------------------------------------------
# STAGE 1: a fitted noise prefilter set, loaded and never re-fit here
# ---------------------------------------------------------------------------

STAGE_ONE_PREFIX_RULE = (
    "at a capture length ABOVE the longest fitted prefilter length, the "
    "stage-1 pose-degeneracy features are computed on the capture's FIRST "
    "max-fitted-length samples and gated with that length's bundle.  The "
    "first N samples of a longer capture are exactly an N-sample capture of "
    "the same emission, so this is the same causal-prefix rule the release "
    "protocol's length suite is built on, not an approximation.  Lengths "
    "BELOW the smallest fitted length keep the existing refusal semantics: "
    "no fitted length is a causal prefix of such a capture, so no bundle may "
    "be substituted"
)


@dataclass(frozen=True)
class StageOneGate:
    """A length-keyed set of fitted prefilter bundles, validated for use here.

    The prefilter is **length dependent** -- the pose-degeneracy features are --
    so the sibling module fits one bundle per capture length and this gate
    refuses to score a length it has no bundle for rather than reusing a
    neighbouring one.  The single exception is the causal-prefix rule
    (:data:`STAGE_ONE_PREFIX_RULE`): a capture LONGER than every fitted length
    is gated by the longest fitted bundle, applied to the capture's first
    max-fitted-length samples -- which are exactly a capture of that fitted
    length.  Callers must hand this gate features computed on that prefix;
    :func:`stage_one_prefix_captures` derives the correctly sliced captures.
    """

    prefilter: NoisePrefilterSource
    posedegen: PoseDegeneracySource
    directory: Path
    models: dict[int, Any]
    set_sha256: str

    @property
    def lengths(self) -> tuple[int, ...]:
        return tuple(sorted(self.models))

    @property
    def max_fitted_length(self) -> int:
        return int(max(self.models))

    def feature_length_for(self, capture_length: int) -> int:
        """The fitted length whose bundle gates a capture of this length.

        A fitted length maps to itself.  A length above the longest fitted
        length maps to the longest fitted length under the causal-prefix rule;
        stage-1 features must then be computed on the capture's first
        ``max_fitted_length`` samples.  Every other uncovered length is
        refused, exactly as before.
        """
        length = int(capture_length)
        if length in self.models:
            return length
        if length > self.max_fitted_length:
            return self.max_fitted_length
        raise KeyError(
            f"no fitted noise prefilter for capture length {length} in "
            f"{self.directory}; available {list(self.lengths)}.  The "
            "pose-degeneracy features are length dependent, so a bundle "
            "fitted at another length may not be substituted, and the "
            "causal-prefix rule applies only ABOVE the longest fitted "
            f"length ({self.max_fitted_length})"
        )

    def model_for(self, capture_length: int) -> Any:
        return self.models[self.feature_length_for(capture_length)]

    def raw(self, features: np.ndarray, capture_length: int) -> np.ndarray:
        """Stage-1 log-odds of noise; higher is more noise-like.

        ``capture_length`` is the capture's true length; it is resolved through
        :meth:`feature_length_for`, so ``features`` MUST have been computed on
        each capture's first ``feature_length_for(capture_length)`` samples.
        """
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.posedegen.feature_count:
            raise ValueError(
                "stage-1 features must be a "
                f"[rows, {self.posedegen.feature_count}] pose-degeneracy "
                f"matrix, got {matrix.shape}"
            )
        if not len(matrix):
            return np.zeros(0, dtype=np.float64)
        feature_length = self.feature_length_for(capture_length)
        value = np.asarray(
            self.models[feature_length].score(
                matrix, capture_length=int(feature_length)
            ),
            dtype=np.float64,
        )
        if value.shape != (len(matrix),):
            raise RuntimeError(
                f"{self.prefilter.module_name} model.score returned shape "
                f"{value.shape}, expected {(len(matrix),)}"
            )
        if not np.isfinite(value).all():
            raise RuntimeError(
                f"{self.prefilter.module_name} model.score returned a "
                "non-finite value"
            )
        return value

    def fires(self, features: np.ndarray, capture_length: int) -> np.ndarray:
        """``True`` where the capture is gated as noise before classification.

        The decision comes from the model's own ``decide``, not from a
        re-derived comparison, so the gate rule (``score >= threshold_score``)
        lives in exactly one place.  It is then cross-checked against the
        threshold this module read, so a change to that rule cannot pass
        silently.
        """
        matrix = np.asarray(features, dtype=np.float64)
        if not len(matrix):
            return np.zeros(0, dtype=bool)
        feature_length = self.feature_length_for(capture_length)
        model = self.models[feature_length]
        decided = np.asarray(
            model.decide(matrix, capture_length=int(feature_length)), dtype=bool
        )
        expected = self.raw(matrix, capture_length) >= float(
            model.threshold_score
        )
        if not np.array_equal(decided, expected):
            raise AssertionError(
                f"{self.prefilter.module_name} model.decide disagrees with "
                "'score >= threshold_score'; the gate rule changed"
            )
        return decided

    def rank(self, raw_scores: np.ndarray) -> np.ndarray:
        """The stage-1 score mapped monotonically into ``(0, 1)``."""
        value = np.asarray(raw_scores, dtype=np.float64)
        if not len(value):
            return np.zeros(0, dtype=np.float64)
        return _bounded_rank(value)

    def provenance(self) -> dict[str, Any]:
        return {
            "directory": str(self.directory),
            "set_sha256": self.set_sha256,
            "capture_lengths": list(self.lengths),
            "fitted_here": False,
            "stage_one_prefix_rule": {
                "max_fitted_length": self.max_fitted_length,
                "rule": STAGE_ONE_PREFIX_RULE,
            },
            "models": {
                str(length): self.models[length].metadata()
                for length in self.lengths
            },
        }


def stage_one_prefix_captures(
    stage_one: StageOneGate,
    captures: Sequence[np.ndarray],
    capture_length: int,
) -> tuple[list[np.ndarray], int]:
    """Slice captures to the stage-1 feature length under the prefix rule.

    Returns ``(captures, feature_length)`` where every returned capture has
    exactly ``feature_length = stage_one.feature_length_for(capture_length)``
    samples: the capture itself at a fitted length, its first
    ``max_fitted_length`` samples above the longest fitted length.  Every
    capture is first verified to actually have ``capture_length`` samples, so
    a mislabeled population cannot be silently truncated.
    """
    length = int(capture_length)
    feature_length = stage_one.feature_length_for(length)
    rows = [np.asarray(row) for row in captures]
    for index, row in enumerate(rows):
        if row.ndim != 1 or len(row) != length:
            raise ValueError(
                f"capture {index} has shape {row.shape}, expected ({length},); "
                "the prefix rule needs the declared capture length to be true"
            )
    if feature_length == length:
        return rows, feature_length
    return [row[:feature_length] for row in rows], feature_length


def load_stage_one(
    prefilter: NoisePrefilterSource,
    posedegen: PoseDegeneracySource,
    directory: Path,
    *,
    required_lengths: Sequence[int],
) -> StageOneGate:
    """Load a fitted prefilter set and refuse it unless it is usable as a gate.

    Everything checked here is a property that, if wrong, would make the emitted
    evidence wrong rather than merely make the run fail:

    * a bundle exists for every capture length this run will score;
    * every bundle carries an operating point, so ``decide`` is defined;
    * every bundle's operating point was chosen on ENROLLMENT, with zero
      training, selection and novelty rows -- read from the bundle's own
      provenance, not assumed;
    * every bundle's fitting seed is refused by the prefilter module's own
      ledger check if it is a novelty, sealed or release seed;
    * every bundle declares the architecture contract change, so a staged run
      cannot be assembled from a bundle that does not admit it gates.
    """
    path = reject_sealed_path(Path(directory), "prefilter set")
    if not path.is_dir():
        raise FileNotFoundError(path)
    models = {
        int(length): model
        for length, model in prefilter.module.load_prefilter_set(path).items()
    }
    if not models:
        raise ValueError(f"no fitted noise prefilter bundles under {path}")

    for length, model in sorted(models.items()):
        missing = [
            item for item in REQUIRED_MODEL_API if not hasattr(model, item)
        ]
        if missing:
            raise RuntimeError(
                f"the N{length} bundle is missing {missing}.\n"
                f"{NOISE_PREFILTER_CONTRACT}"
            )
        if int(model.capture_length) != length:
            raise RuntimeError(
                f"the bundle filed under N{length} was fitted at "
                f"{model.capture_length}"
            )
        if not bool(model.has_threshold):
            raise RuntimeError(
                f"the N{length} bundle has no operating point; a gate without a "
                "threshold cannot decide anything.  Run noise_prefilter.py's "
                "enrollment-only fit_threshold step first"
            )
        unknown = [
            name
            for name in model.feature_names
            if name not in posedegen.feature_names
        ]
        if unknown:
            raise RuntimeError(
                f"the N{length} bundle consumes features {unknown} that "
                f"{posedegen.module_name} does not produce"
            )
        metadata = model.metadata()
        for key, expected in (
            ("additive_only", False),
            ("changes_closed_label", True),
            ("gates_before_classification", True),
        ):
            if metadata.get(key) != expected:
                raise RuntimeError(
                    f"the N{length} bundle records {key}="
                    f"{metadata.get(key)!r}, expected {expected!r}; a staged "
                    "run may not be assembled from a bundle that does not "
                    "declare that it gates"
                )
        threshold_provenance = dict(model.threshold_provenance)
        if threshold_provenance.get("population") != prefilter.module.ENROLLMENT_SPLIT:
            raise RuntimeError(
                f"the N{length} operating point was chosen on "
                f"{threshold_provenance.get('population')!r}, not enrollment"
            )
        if threshold_provenance.get("policy") != prefilter.module.BUDGET_POLICY:
            raise RuntimeError(
                f"the N{length} operating point uses policy "
                f"{threshold_provenance.get('policy')!r}, expected the frozen "
                f"{prefilter.module.BUDGET_POLICY!r}"
            )
        if threshold_provenance.get(
            "known_false_positive_budget"
        ) != STAGE_ONE_KNOWN_FALSE_POSITIVE_BUDGET:
            raise RuntimeError(
                f"the N{length} operating point records known-false-positive "
                f"budget "
                f"{threshold_provenance.get('known_false_positive_budget')!r}, "
                f"expected the frozen "
                f"{STAGE_ONE_KNOWN_FALSE_POSITIVE_BUDGET!r}"
            )
        for key in (
            "training_rows_used_for_threshold",
            "selection_rows_used_for_threshold",
            "novelty_rows_used_for_threshold",
        ):
            if int(threshold_provenance.get(key, -1)) != 0:
                raise RuntimeError(
                    f"the N{length} operating point records {key}="
                    f"{threshold_provenance.get(key)!r}; it must be 0"
                )
        fit_provenance = dict(model.fit_provenance)
        if fit_provenance.get("known_population") != prefilter.module.TRAIN_SPLIT:
            raise RuntimeError(
                f"the N{length} coefficients were fit on "
                f"{fit_provenance.get('known_population')!r}, not training"
            )
        # The prefilter module's own ledger check, reused rather than restated.
        prefilter.module.validate_fitting_seed(fit_provenance["fitting_seed"])

    gate = StageOneGate(
        prefilter=prefilter,
        posedegen=posedegen,
        directory=path,
        models=models,
        set_sha256=str(prefilter.module.prefilter_set_sha256(path)),
    )

    # A required length is usable exactly when the gate itself can resolve it:
    # fitted lengths map to themselves, and a length ABOVE the longest fitted
    # length is gated on its causal prefix (STAGE_ONE_PREFIX_RULE) -- callers,
    # this harness's own sweep included, must then compute stage-1 features on
    # the capture's first feature_length_for(length) samples
    # (stage_one_prefix_captures).  Every other uncovered length is refused.
    missing_lengths = []
    for length in sorted({int(item) for item in required_lengths}):
        try:
            gate.feature_length_for(length)
        except KeyError:
            missing_lengths.append(int(length))
    if missing_lengths:
        raise ValueError(
            f"{path} has no fitted prefilter for capture lengths "
            f"{missing_lengths}; available {sorted(models)}.  The features are "
            "length dependent and a bundle from another length may not stand "
            "in, and the causal-prefix rule applies only ABOVE the longest "
            "fitted length"
        )

    return gate



# ---------------------------------------------------------------------------
# composite survivor policy (version 4; explicit read-only v2/v3 loaders)
# ---------------------------------------------------------------------------


def _rank_vector(value: np.ndarray, name: str) -> np.ndarray:
    """A finite vector of empirical ranks in ``[0, 1)``."""
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector")
    if np.any(result < 0.0) or np.any(result >= 1.0):
        raise ValueError(f"{name} must lie in [0, 1)")
    return result


@dataclass(frozen=True)
class CompositeSurvivorPolicy:
    """The frozen composite survivor score and its enrollment-only threshold.

    ``stage_one_calibration_raw`` is the sorted stage-1 log-odds of the
    ENROLLMENT rows that survive the stage-1 gate; a survivor's stage-1 rank is
    its :func:`v3_time_domain_openset.empirical_rank` against that calibration
    (searchsorted-left over the sorted vector, the same convention as every
    stage-2 rank).  ``composite_calibration_raw`` is the sorted composite of
    those same enrollment survivors, kept so a loader can recompute and verify
    ``threshold`` instead of trusting it.  ``stage_two_threshold`` is the
    stage-2 policy's own untouched enrollment q95, recorded for cross-checks;
    the staged decision threshold is ``threshold``.  ``policy_spec`` travels
    with a loaded artifact so explicitly requested legacy-v2/q95 and
    failed-v3/q99 policies retain their own semantics after this module's
    default moves to policy-v4/q97.
    """

    stage_one_calibration_raw: np.ndarray
    composite_calibration_raw: np.ndarray
    threshold: float
    stage_two_threshold: float
    enrollment_rows: int
    enrollment_gated_rows: int
    enrollment_capture_length: int
    policy_spec: StagedPolicySpec = STAGED_POLICY_SPEC

    @property
    def enrollment_survivor_rows(self) -> int:
        return int(len(self.stage_one_calibration_raw))

    def stage_one_survivor_rank(self, stage_one_raw: np.ndarray) -> np.ndarray:
        """Enrollment-survivor empirical rank of stage-1 log-odds, in [0, 1)."""
        return empirical_rank(
            np.asarray(stage_one_raw, dtype=np.float64),
            self.stage_one_calibration_raw,
        )

    def composite(
        self,
        stage_two_score: np.ndarray,
        stage_one_raw: np.ndarray,
    ) -> np.ndarray:
        """:data:`COMPOSITE_SURVIVOR_SCORE` for survivor rows."""
        two = _rank_vector(stage_two_score, "stage-2 survivor scores")
        raw = np.asarray(stage_one_raw, dtype=np.float64)
        if raw.shape != two.shape:
            raise ValueError(
                "stage-2 scores and stage-1 raw scores are not aligned"
            )
        return np.maximum(two, self.stage_one_survivor_rank(raw))

    def provenance(self) -> dict[str, Any]:
        spec = self.policy_spec
        return {
            "policy_version": spec.version,
            "schema": int(spec.schema),
            "kind": spec.kind,
            "survivor_score": COMPOSITE_SURVIVOR_SCORE,
            "stage_one_rank_convention": (
                "empirical enrollment rank: searchsorted-left over the sorted "
                "enrollment-SURVIVOR stage-1 log-odds, divided by "
                "(survivors + 1); identical to the stage-2 "
                "v3_time_domain_openset.empirical_rank convention"
            ),
            "threshold": float(self.threshold),
            "threshold_quantile": float(spec.threshold_quantile),
            "threshold_population": (
                "enrollment stage-1 survivors only (enrollment only, as every "
                "rank and threshold in this chain)"
            ),
            "training_rows_used_for_threshold": 0,
            "selection_rows_used_for_threshold": 0,
            "novelty_rows_used_for_threshold": 0,
            "release_rows_used_for_threshold": 0,
            "stage_one_known_false_positive_budget": (
                STAGE_ONE_KNOWN_FALSE_POSITIVE_BUDGET
            ),
            "survivor_known_false_positive_budget": (
                1.0 - float(spec.threshold_quantile)
            ),
            "nominal_enrollment_false_unknown_budget": (
                STAGE_ONE_KNOWN_FALSE_POSITIVE_BUDGET
                + (1.0 - STAGE_ONE_KNOWN_FALSE_POSITIVE_BUDGET)
                * (1.0 - float(spec.threshold_quantile))
            ),
            "only_policy_change": spec.only_policy_change,
            "stage_one_changed": False,
            "rejector_cnn_fusion_changed": False,
            "classifier_cnn_fusion_changed": False,
            "gate_floors_and_known_fur_ceiling_unchanged": True,
            "stage_two_threshold_unchanged": float(self.stage_two_threshold),
            "enrollment_rows": int(self.enrollment_rows),
            "enrollment_gated_rows": int(self.enrollment_gated_rows),
            "enrollment_survivor_rows": self.enrollment_survivor_rows,
            "enrollment_stage_one_gate_rate": (
                float(self.enrollment_gated_rows / self.enrollment_rows)
                if self.enrollment_rows
                else 0.0
            ),
            "enrollment_capture_length": int(self.enrollment_capture_length),
            "rationale": spec.rationale,
        }


def fit_composite_policy(
    stage_one: StageOneGate,
    enrollment_features: np.ndarray,
    enrollment_capture_length: int,
    enrollment_stage_two_scores: np.ndarray,
    *,
    stage_two_threshold: float,
) -> CompositeSurvivorPolicy:
    """Fit the composite rank and threshold on ENROLLMENT survivors only.

    The signature is the guarantee, exactly as for the stage-2 fit: only the
    enrollment stage-1 feature matrix and the enrollment stage-2 scores are
    parameters, so no selection row, no novelty row and no release row can
    influence the calibration or the threshold.  Nothing here is searched: the
    q97 quantile is the predeclared frozen policy-v4 constant.
    """
    length = int(enrollment_capture_length)
    if length not in stage_one.models:
        raise ValueError(
            f"the enrollment population's native length N{length} has no "
            f"fitted prefilter bundle in {stage_one.directory}; available "
            f"{list(stage_one.lengths)}.  Enrollment features are computed at "
            "native length, so the causal-prefix rule may not stand in for "
            "this length"
        )
    two = _rank_vector(
        enrollment_stage_two_scores, "enrollment stage-2 scores"
    )
    matrix = np.asarray(enrollment_features, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) != len(two):
        raise ValueError(
            "enrollment stage-1 features and stage-2 scores are not aligned"
        )
    raw = stage_one.raw(matrix, length)
    gated = stage_one.fires(matrix, length)
    survivors = ~gated
    if not survivors.any():
        raise ValueError(
            "the stage-1 gate fired on every enrollment row; a composite "
            "calibration needs at least one enrollment survivor"
        )
    calibration = np.sort(raw[survivors])
    survivor_rank = empirical_rank(raw[survivors], calibration)
    composite = np.maximum(two[survivors], survivor_rank)
    threshold = float(
        np.quantile(composite, COMPOSITE_THRESHOLD_QUANTILE)
    )
    return CompositeSurvivorPolicy(
        stage_one_calibration_raw=calibration,
        composite_calibration_raw=np.sort(composite),
        threshold=threshold,
        stage_two_threshold=float(stage_two_threshold),
        enrollment_rows=int(len(matrix)),
        enrollment_gated_rows=int(np.count_nonzero(gated)),
        enrollment_capture_length=length,
    )


def save_composite_policy(path: Path, policy: CompositeSurvivorPolicy) -> None:
    """Serialize the current composite policy without pickle.

    Historical policy-v2/q95 and failed-policy-v3/q99 objects are read-only
    compatibility objects.  Refusing to re-save either through the v4 writer
    prevents an old threshold from being relabelled with policy-v4/q97
    metadata.
    """
    if policy.policy_spec != STAGED_POLICY_SPEC:
        raise ValueError(
            "save_composite_policy only writes the current "
            f"{STAGED_POLICY_VERSION!r}; legacy artifacts remain loadable only "
            "under their explicitly requested old version"
        )
    np.savez_compressed(
        Path(path),
        schema=np.asarray(policy.policy_spec.schema, dtype=np.int64),
        kind=np.asarray(policy.policy_spec.kind),
        policy_version=np.asarray(policy.policy_spec.version),
        survivor_score=np.asarray(COMPOSITE_SURVIVOR_SCORE),
        threshold_quantile=np.asarray(
            policy.policy_spec.threshold_quantile, dtype=np.float64
        ),
        threshold=np.asarray(policy.threshold, dtype=np.float64),
        stage_two_threshold=np.asarray(
            policy.stage_two_threshold, dtype=np.float64
        ),
        stage_one_calibration_raw=np.asarray(
            policy.stage_one_calibration_raw, dtype=np.float64
        ),
        composite_calibration_raw=np.asarray(
            policy.composite_calibration_raw, dtype=np.float64
        ),
        enrollment_rows=np.asarray(policy.enrollment_rows, dtype=np.int64),
        enrollment_gated_rows=np.asarray(
            policy.enrollment_gated_rows, dtype=np.int64
        ),
        enrollment_capture_length=np.asarray(
            policy.enrollment_capture_length, dtype=np.int64
        ),
        threshold_population=np.asarray(
            "enrollment_stage_one_survivors_only"
        ),
        training_rows_used_for_threshold=np.asarray(0, dtype=np.int64),
        selection_rows_used_for_threshold=np.asarray(0, dtype=np.int64),
        novelty_rows_used_for_threshold=np.asarray(0, dtype=np.int64),
        release_rows_used_for_threshold=np.asarray(0, dtype=np.int64),
        stage_one_known_false_positive_budget=np.asarray(
            STAGE_ONE_KNOWN_FALSE_POSITIVE_BUDGET, dtype=np.float64
        ),
        survivor_known_false_positive_budget=np.asarray(
            SURVIVOR_KNOWN_FALSE_POSITIVE_BUDGET, dtype=np.float64
        ),
        nominal_enrollment_false_unknown_budget=np.asarray(
            NOMINAL_ENROLLMENT_FALSE_UNKNOWN_BUDGET, dtype=np.float64
        ),
        only_policy_change=np.asarray(policy.policy_spec.only_policy_change),
        stage_one_changed=np.asarray(False, dtype=np.bool_),
        rejector_cnn_fusion_changed=np.asarray(False, dtype=np.bool_),
        classifier_cnn_fusion_changed=np.asarray(False, dtype=np.bool_),
        gate_contract_changed=np.asarray(False, dtype=np.bool_),
    )


def _policy_spec_for(version: str) -> StagedPolicySpec:
    try:
        return STAGED_POLICY_SPECS[str(version)]
    except KeyError as exc:
        raise ValueError(
            f"unsupported staged policy version {version!r}; supported "
            f"versions are {sorted(STAGED_POLICY_SPECS)}"
        ) from exc


def _policy_hygiene_expected(
    spec: StagedPolicySpec,
) -> dict[str, str | int | float | bool] | None:
    """Schema-3+ declarations shared by failed q99 and current q97."""
    if spec.schema < FAILED_Q99_STAGED_POLICY_SCHEMA:
        return None
    survivor_budget = 1.0 - float(spec.threshold_quantile)
    return {
        "threshold_population": "enrollment_stage_one_survivors_only",
        "training_rows_used_for_threshold": 0,
        "selection_rows_used_for_threshold": 0,
        "novelty_rows_used_for_threshold": 0,
        "release_rows_used_for_threshold": 0,
        "stage_one_known_false_positive_budget": (
            STAGE_ONE_KNOWN_FALSE_POSITIVE_BUDGET
        ),
        "survivor_known_false_positive_budget": survivor_budget,
        "nominal_enrollment_false_unknown_budget": (
            STAGE_ONE_KNOWN_FALSE_POSITIVE_BUDGET
            + (1.0 - STAGE_ONE_KNOWN_FALSE_POSITIVE_BUDGET)
            * survivor_budget
        ),
        "only_policy_change": spec.only_policy_change,
        "stage_one_changed": False,
        "rejector_cnn_fusion_changed": False,
        "classifier_cnn_fusion_changed": False,
        "gate_contract_changed": False,
    }


def load_composite_policy(
    path: Path,
    *,
    expected_stage_two_threshold: float | None = None,
    expected_policy_version: str = STAGED_POLICY_VERSION,
) -> CompositeSurvivorPolicy:
    """Load and verify a serialized composite policy.

    The default admits only current policy-v4/q97.  Historical v2/q95 and
    failed-v3/q99 artifacts remain inspectable only when the caller explicitly
    requests their exact version; neither can silently acquire q97 semantics.
    In every case the threshold is recomputed from the stored
    enrollment-survivor composite calibration instead of trusting the scalar.
    """
    spec = _policy_spec_for(expected_policy_version)
    file_path = Path(path)
    with np.load(file_path) as payload:
        common_required = [
            "schema",
            "kind",
            "policy_version",
            "survivor_score",
            "threshold_quantile",
            "threshold",
            "stage_two_threshold",
            "stage_one_calibration_raw",
            "composite_calibration_raw",
            "enrollment_rows",
            "enrollment_gated_rows",
            "enrollment_capture_length",
        ]
        missing = [
            name
            for name in common_required
            if name not in payload.files
        ]
        if missing:
            raise ValueError(f"{file_path} is missing {missing}")
        for name, expected in (
            ("schema", spec.schema),
            ("kind", spec.kind),
            ("policy_version", spec.version),
            ("survivor_score", COMPOSITE_SURVIVOR_SCORE),
        ):
            found = payload[name].item()
            found = int(found) if name == "schema" else str(found)
            if found != expected:
                raise ValueError(
                    f"staged policy version mismatch: {file_path} records "
                    f"{name}={found!r}, but the caller explicitly requested "
                    f"{expected!r}.  A staged artifact may only be scored "
                    "with the policy version that produced it"
                )
        hygiene = _policy_hygiene_expected(spec)
        if hygiene is not None:
            versioned_hygiene_required = [
                "threshold_population",
                "training_rows_used_for_threshold",
                "selection_rows_used_for_threshold",
                "novelty_rows_used_for_threshold",
                "release_rows_used_for_threshold",
                "stage_one_known_false_positive_budget",
                "survivor_known_false_positive_budget",
                "nominal_enrollment_false_unknown_budget",
                "only_policy_change",
                "stage_one_changed",
                "rejector_cnn_fusion_changed",
                "classifier_cnn_fusion_changed",
                "gate_contract_changed",
            ]
            missing = [
                name
                for name in versioned_hygiene_required
                if name not in payload.files
            ]
            if missing:
                raise ValueError(f"{file_path} is missing {missing}")
        quantile = float(payload["threshold_quantile"])
        if quantile != float(spec.threshold_quantile):
            raise ValueError(
                f"{file_path} records threshold_quantile={quantile}, expected "
                f"the frozen {float(spec.threshold_quantile)} for "
                f"{spec.version}"
            )
        if hygiene is not None:
            for name, expected in hygiene.items():
                found = payload[name].item()
                if found != expected:
                    raise ValueError(
                        f"{file_path} records {name}={found!r}, expected "
                        f"{expected!r}; {spec.version} calibration hygiene "
                        "may not be weakened"
                    )
        calibration = np.asarray(
            payload["stage_one_calibration_raw"], dtype=np.float64
        )
        composite_calibration = np.asarray(
            payload["composite_calibration_raw"], dtype=np.float64
        )
        policy = CompositeSurvivorPolicy(
            stage_one_calibration_raw=calibration,
            composite_calibration_raw=composite_calibration,
            threshold=float(payload["threshold"]),
            stage_two_threshold=float(payload["stage_two_threshold"]),
            enrollment_rows=int(payload["enrollment_rows"]),
            enrollment_gated_rows=int(payload["enrollment_gated_rows"]),
            enrollment_capture_length=int(
                payload["enrollment_capture_length"]
            ),
            policy_spec=spec,
        )
    for name, vector in (
        ("stage_one_calibration_raw", policy.stage_one_calibration_raw),
        ("composite_calibration_raw", policy.composite_calibration_raw),
    ):
        if (
            vector.ndim != 1
            or not len(vector)
            or not np.isfinite(vector).all()
            or np.any(np.diff(vector) < 0.0)
        ):
            raise ValueError(
                f"{file_path}: {name} must be a non-empty, finite, sorted "
                "vector"
            )
    _rank_vector(
        policy.composite_calibration_raw, "composite_calibration_raw"
    )
    if len(policy.stage_one_calibration_raw) != len(
        policy.composite_calibration_raw
    ):
        raise ValueError(
            f"{file_path}: the two survivor calibrations disagree on the "
            "survivor count"
        )
    if policy.enrollment_rows != policy.enrollment_gated_rows + len(
        policy.stage_one_calibration_raw
    ):
        raise ValueError(
            f"{file_path}: enrollment_rows != gated + survivors"
        )
    expected_threshold = float(
        np.quantile(
            policy.composite_calibration_raw,
            policy.policy_spec.threshold_quantile,
        )
    )
    if (
        not np.isfinite(policy.threshold)
        or policy.threshold != expected_threshold
    ):
        raise ValueError(
            f"{file_path}: stored threshold {policy.threshold!r} does not "
            "equal the frozen quantile of the stored survivor composite "
            f"calibration ({expected_threshold!r})"
        )
    if not 0.0 <= policy.threshold < 1.0:
        raise ValueError(
            f"{file_path}: composite threshold must lie in [0, 1)"
        )
    if expected_stage_two_threshold is not None and float(
        expected_stage_two_threshold
    ) != policy.stage_two_threshold:
        raise ValueError(
            f"{file_path} records stage_two_threshold="
            f"{policy.stage_two_threshold!r}, but the loaded stage-2 policy "
            f"has threshold {float(expected_stage_two_threshold)!r}; the "
            "composite was fit against a different stage-2 state"
        )
    return policy


def rethreshold_legacy_composite_policy(
    legacy: CompositeSurvivorPolicy,
) -> CompositeSurvivorPolicy:
    """Derive policy-v4/q97 from a verified v2/q95 or failed-v3/q99 policy.

    This is deliberately a pure enrollment-calibration transformation: it
    receives no selection, novelty, sealed or release rows, and it preserves
    every stored calibration value and the untouched stage-2 threshold.  It is
    useful both as a migration check and as proof that q97 lies between q95
    and q99 on byte-identical enrollment calibration.
    """
    if legacy.policy_spec not in (
        LEGACY_STAGED_POLICY_SPEC,
        FAILED_Q99_STAGED_POLICY_SPEC,
    ):
        raise ValueError(
            "rethreshold_legacy_composite_policy requires an explicitly "
            "loaded policy-v2/q95 or failed-policy-v3/q99 artifact"
        )
    threshold = float(
        np.quantile(
            legacy.composite_calibration_raw,
            COMPOSITE_THRESHOLD_QUANTILE,
        )
    )
    if (
        legacy.policy_spec == LEGACY_STAGED_POLICY_SPEC
        and threshold < legacy.threshold
    ):
        raise AssertionError(
            "q97 threshold is below the verified q95 threshold"
        )
    if (
        legacy.policy_spec == FAILED_Q99_STAGED_POLICY_SPEC
        and threshold > legacy.threshold
    ):
        raise AssertionError(
            "q97 threshold is above the verified q99 threshold"
        )
    return CompositeSurvivorPolicy(
        stage_one_calibration_raw=np.array(
            legacy.stage_one_calibration_raw, dtype=np.float64, copy=True
        ),
        composite_calibration_raw=np.array(
            legacy.composite_calibration_raw, dtype=np.float64, copy=True
        ),
        threshold=threshold,
        stage_two_threshold=float(legacy.stage_two_threshold),
        enrollment_rows=int(legacy.enrollment_rows),
        enrollment_gated_rows=int(legacy.enrollment_gated_rows),
        enrollment_capture_length=int(legacy.enrollment_capture_length),
        policy_spec=STAGED_POLICY_SPEC,
    )


# ---------------------------------------------------------------------------
# rebuilding the populations, with aligned stage-1 features
# ---------------------------------------------------------------------------

# ``run_time_domain_dev._prepare_data`` preprocesses these three populations, in
# this order, by calling the module-level ``_preprocess_rows`` once per
# population with the raw captures for that population.
PREPARE_DATA_POPULATIONS = (
    ("train", CENTER_SPLIT),
    ("enroll", PROTOTYPE_SPLIT),
    ("selection", METRIC_SPLIT),
)


class _PrefilterRecorder:
    """Extract stage-1 features from captures as they pass through.

    ``_prepare_data`` builds the raw captures (including the multi-length
    training views, whose per-row prefix length is not recoverable from the
    returned arrays), preprocesses them, and drops them.  Stage 1 needs the raw
    enrollment and selection rows.

    Rather than reimplement the capture construction -- which would duplicate the
    training-view expansion and could silently drift out of alignment -- this
    wraps ``_prepare_data``'s own ``_preprocess_rows`` and derives the features
    from exactly the captures it was handed, in exactly its order.  Nothing about
    the preprocessing is altered: the wrapped call is delegated unchanged and its
    result returned unchanged.

    **Only the enrollment and selection populations are extracted.**  Stage 1
    arrives already fitted, so this module has no use for training features --
    and not computing them is the strongest available statement that it cannot
    have fitted anything on them.  Enrollment features exist for exactly one
    purpose: the composite survivor rank and its policy-version threshold are
    calibrated on enrollment, which is where every rank and threshold in this
    chain is calibrated.  The training population is counted, so the
    interception can still be verified.

    :func:`_recorded_prefilter` refuses to continue unless all three populations
    passed through in order and each recorded matrix has one row per packed row
    of its population.
    """

    #: The populations whose stage-1 features this module needs.
    RECORDED_SPLITS = (PROTOTYPE_SPLIT, METRIC_SPLIT)

    def __init__(self, inner: Any, posedegen: PoseDegeneracySource) -> None:
        self._inner = inner
        self._posedegen = posedegen
        self.row_counts: list[int] = []
        self.matrices: dict[str, np.ndarray] = {}
        self.seconds: dict[str, float] = {}
        self.capture_lengths: dict[str, tuple[int, ...]] = {}

    def __call__(
        self,
        captures: Sequence[np.ndarray],
        *,
        patch_length: int,
        patch_count: int,
        target_frac: float,
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        rows = list(captures)
        index = len(self.row_counts)
        self.row_counts.append(len(rows))
        if (
            index < len(PREPARE_DATA_POPULATIONS)
            and PREPARE_DATA_POPULATIONS[index][1] in self.RECORDED_SPLITS
        ):
            split = PREPARE_DATA_POPULATIONS[index][1]
            started = time.perf_counter()
            self.matrices[split] = prefilter_matrix(
                self._posedegen,
                rows,
                patch_length=int(patch_length),
                target_frac=float(target_frac),
                population=PREPARE_DATA_POPULATIONS[index][0],
            )
            self.seconds[split] = time.perf_counter() - started
            self.capture_lengths[split] = tuple(
                sorted({len(row) for row in rows})
            )
        return self._inner(
            rows,
            patch_length=patch_length,
            patch_count=patch_count,
            target_frac=target_frac,
        )


def _recorded_prefilter(
    recorder: _PrefilterRecorder,
    data: Mapping[str, Any],
) -> dict[str, tuple[np.ndarray, int]]:
    """Validate the interception; return ``{split: (features, length)}``."""
    if len(recorder.row_counts) != len(PREPARE_DATA_POPULATIONS):
        raise AssertionError(
            "stage-1 feature interception saw "
            f"{len(recorder.row_counts)} populations, expected "
            f"{len(PREPARE_DATA_POPULATIONS)}; run_time_domain_dev._prepare_data "
            "no longer routes every population through _preprocess_rows"
        )
    recorded: dict[str, tuple[np.ndarray, int]] = {}
    for name, split in PREPARE_DATA_POPULATIONS:
        if split not in recorder.RECORDED_SPLITS:
            continue
        matrix = recorder.matrices.get(split)
        if matrix is None:
            raise AssertionError(
                f"the {split!r} population never reached the stage-1 feature "
                "interception"
            )
        packed = data[f"x{split}"]
        if len(matrix) != len(packed):
            raise AssertionError(
                f"{name}: {len(matrix)} stage-1 feature rows for "
                f"{len(packed)} packed rows; the features would be misaligned"
            )
        lengths = recorder.capture_lengths.get(split, ())
        if len(lengths) != 1:
            raise AssertionError(
                f"the {name} population mixes capture lengths "
                f"{list(lengths)}; the prefilter is length dependent and one "
                "bundle cannot score a mixed population"
            )
        recorded[split] = (matrix, int(lengths[0]))
    return recorded


@dataclass
class RebuiltPopulations:
    """The base rebuild plus enrollment and selection stage-1 features."""

    populations: Populations
    enrollment_features: np.ndarray
    enrollment_capture_length: int
    enrollment_feature_seconds: float
    selection_features: np.ndarray
    selection_capture_length: int
    selection_feature_seconds: float


def rebuild_populations(
    artifact: FusionArtifact,
    device: torch.device,
    posedegen: PoseDegeneracySource,
    *,
    tolerance: float = DEFAULT_REPRODUCTION_TOLERANCE,
) -> RebuiltPopulations:
    """Call the base rebuild unchanged, recording stage-1 features alongside it.

    :func:`fit_v3_openset.rebuild_populations` is invoked, not copied, so the
    bit-identical rebuild check against the fusion artifact, the branch hash
    check, the consumed-test-row assertions and the cache contract check are all
    the base module's own code.  The only addition is the interception that
    derives enrollment and selection stage-1 features from the same raw
    captures.
    """
    recorder = _PrefilterRecorder(dev._preprocess_rows, posedegen)
    dev._preprocess_rows = recorder
    try:
        populations = base.rebuild_populations(
            artifact, device, tolerance=tolerance
        )
    finally:
        dev._preprocess_rows = recorder._inner
    recorded = _recorded_prefilter(recorder, populations.data)
    enrollment_features, enrollment_length = recorded[PROTOTYPE_SPLIT]
    selection_features, selection_length = recorded[METRIC_SPLIT]
    return RebuiltPopulations(
        populations=populations,
        enrollment_features=enrollment_features,
        enrollment_capture_length=enrollment_length,
        enrollment_feature_seconds=float(
            recorder.seconds.get(PROTOTYPE_SPLIT, 0.0)
        ),
        selection_features=selection_features,
        selection_capture_length=selection_length,
        selection_feature_seconds=float(
            recorder.seconds.get(METRIC_SPLIT, 0.0)
        ),
    )


# ---------------------------------------------------------------------------
# development novelty: the base fixtures, plus verified stage-1 features
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrefilterFixture:
    features: np.ndarray
    seconds: float


def generate_prefilter_features(
    posedegen: PoseDegeneracySource,
    seeds: Sequence[int],
    *,
    n_each: int,
    lengths: Sequence[int],
    patch_length: int,
    target_frac: float,
    expected_provenance: Mapping[str, Any],
    families: Sequence[str] = NOVELTY_FAMILIES,
    stage_one: StageOneGate | None = None,
) -> dict[int, dict[str, dict[int, PrefilterFixture]]]:
    """Stage-1 features for the same novelty captures the base module drew.

    The captures are redrawn from the same seeds with the same generator calls in
    the same order, and the per-family capture digests are then compared against
    the provenance :func:`fit_v3_openset.generate_novelty` recorded.  That turns
    "these are the same rows" from an assumption into a check: if the base
    generator ever changes, this raises instead of silently scoring stage 1 on
    different captures from stage 2.

    Redrawing rather than mirroring the base loop is the point.  The base
    module's preprocessing, standardization and provenance stay in one place and
    are not duplicated here, and the redraw costs only the generator, which is
    negligible beside the frontend.

    When ``stage_one`` is given, each observation's stage-1 features are
    computed on its first ``stage_one.feature_length_for(length)`` samples --
    the causal-prefix rule (:data:`STAGE_ONE_PREFIX_RULE`) -- so a sweep
    length above the longest fitted prefilter length is featurized exactly as
    that gate will score it.  At fitted lengths the slice is the whole
    observation and the features are bit-identical to the ``stage_one=None``
    path.  The per-length digests always cover the FULL observation: the
    stage-1/stage-2 same-rows check is about the captures, not the slice.
    """
    seed_values = validate_novelty_seeds(seeds)
    prefix_lengths = validate_prefix_lengths(lengths)
    feature_length_by_length = {
        length: (
            int(stage_one.feature_length_for(length))
            if stage_one is not None
            else int(length)
        )
        for length in prefix_lengths
    }
    for length, feature_length in feature_length_by_length.items():
        if not 0 < feature_length <= int(length):
            raise ValueError(
                f"stage-1 feature length {feature_length} for sweep length "
                f"{length} is not a causal prefix"
            )
    if int(n_each) <= 0:
        raise ValueError("n_each must be positive")
    family_names = tuple(str(name) for name in families)
    unknown = [name for name in family_names if name not in NOVELTY]
    if not family_names or unknown:
        raise ValueError(f"unknown novelty families: {unknown}")

    longest = max(prefix_lengths)
    fixtures: dict[int, dict[str, dict[int, PrefilterFixture]]] = {}
    for seed in seed_values:
        rng = np.random.default_rng(int(seed))
        fixtures[seed] = {}
        for family in family_names:
            generator = NOVELTY[family]
            rows_by_length: dict[int, list[np.ndarray]] = {
                length: [] for length in prefix_lengths
            }
            seconds_by_length: dict[int, float] = {
                length: 0.0 for length in prefix_lengths
            }
            longest_digest = hashlib.sha256()
            prefix_digest = {
                length: hashlib.sha256() for length in prefix_lengths
            }
            for _row in range(int(n_each)):
                raw = generator(longest, rng)
                if len(raw) != longest:
                    raise RuntimeError("novelty generator returned wrong length")
                base._complex_digest(longest_digest, raw)
                for length in prefix_lengths:
                    observation = raw[:length]
                    base._complex_digest(prefix_digest[length], observation)
                    started = time.perf_counter()
                    rows_by_length[length].append(
                        prefilter_row(
                            posedegen,
                            observation[: feature_length_by_length[length]],
                            patch_length=int(patch_length),
                            target_frac=float(target_frac),
                        )
                    )
                    seconds_by_length[length] += time.perf_counter() - started

            recorded = expected_provenance[str(seed)][family]
            if recorded["longest_capture_sha256"] != longest_digest.hexdigest():
                raise AssertionError(
                    f"novelty {seed}/{family}: the redrawn captures do not match "
                    "the digest fit_v3_openset.generate_novelty recorded; "
                    "stage 1 and stage 2 would be scoring different rows"
                )
            for length in prefix_lengths:
                if (
                    recorded["prefix_sha256"][str(length)]
                    != prefix_digest[length].hexdigest()
                ):
                    raise AssertionError(
                        f"novelty {seed}/{family}/N{length}: prefix digest "
                        "differs from the base module's record"
                    )

            fixtures[seed][family] = {}
            for length in prefix_lengths:
                matrix = np.stack(rows_by_length[length]).astype(np.float64)
                if matrix.shape != (int(n_each), posedegen.feature_count):
                    raise RuntimeError(
                        f"novelty {seed}/{family}/N{length}: stage-1 feature "
                        f"shape {matrix.shape} is not aligned with {n_each} rows"
                    )
                if not np.isfinite(matrix).all():
                    raise RuntimeError(
                        f"novelty {seed}/{family}/N{length}: stage-1 features "
                        "contain a non-finite value"
                    )
                fixtures[seed][family][length] = PrefilterFixture(
                    matrix, float(seconds_by_length[length])
                )
    return fixtures


# ---------------------------------------------------------------------------
# staged scoring
# ---------------------------------------------------------------------------

Rejector = base.Rejector

STAGED_SCORE_NOTE = (
    "Stage-2 scores are enrollment empirical ranks in [0, 1).  A stage-1 "
    "survivor's staged score is the COMPOSITE "
    f"{COMPOSITE_SURVIVOR_SCORE}, also in [0, 1), where the stage-1 rank is "
    "the empirical enrollment rank of the stage-1 log-odds fit on ENROLLMENT "
    "stage-1 survivors only (searchsorted-left, the stage-2 convention).  A "
    "row gated at stage 1 has no stage-2 score at all -- that is the short "
    f"circuit -- and is placed at {STAGE_ONE_SCORE_OFFSET} + a bounded "
    "monotone map (softsign) of its stage-1 log-odds, i.e. in (1, 2), above "
    f"every survivor.  The staged unknown threshold is the frozen "
    f"q{COMPOSITE_THRESHOLD_QUANTILE} of the "
    "composite over enrollment survivors, so the staged score is a single "
    "axis on which 'score > staged threshold' is exactly the staged decision "
    "'gated at stage 1, or composite-rejected at stage 2', which is asserted "
    "row by row.  AUROC and threshold recall computed from it are properties "
    "of the staged system, not of either stage read in isolation."
)


@dataclass(frozen=True)
class StagedOutcome:
    """One population scored by the staged system.

    ``stage_two_score`` is the unchanged stage-2 enrollment rank (NaN at gated
    rows: the short circuit).  ``stage_one_survivor_rank`` and
    ``composite_score`` are the version-2 survivor terms, likewise NaN at
    gated rows.  ``staged_score`` is the single decision axis: the composite
    for survivors, ``1 + softsign(stage-1 log-odds)`` for gated rows.
    """

    gated: np.ndarray
    stage_one_raw: np.ndarray
    stage_one_rank: np.ndarray
    stage_two_score: np.ndarray
    stage_one_survivor_rank: np.ndarray
    composite_score: np.ndarray
    staged_score: np.ndarray
    survivor_index: np.ndarray
    feature_seconds: float
    stage_one_seconds: float
    stage_two_seconds: float

    @property
    def rows(self) -> int:
        return int(len(self.gated))

    @property
    def short_circuited(self) -> int:
        return int(np.count_nonzero(self.gated))

    @property
    def staged_seconds(self) -> float:
        return float(
            self.feature_seconds + self.stage_one_seconds + self.stage_two_seconds
        )


def assert_staged_decision_equivalence(
    gated: np.ndarray,
    stage_two_score: np.ndarray,
    composite_score: np.ndarray,
    staged_score: np.ndarray,
    threshold: float,
) -> None:
    """Prove the single staged axis reproduces the two-stage decision exactly.

    ``threshold`` is the staged (composite) threshold.  Survivor decisions are
    made on the composite; the staged axis must equal the composite on
    survivors, sit in (1, 2) on gated rows, and its thresholding must equal
    the two-stage decision row by row.  The composite is also proven to be a
    true max: never below the stage-2 score it contains.
    """
    fired = np.asarray(gated, dtype=bool)
    stage_two = np.asarray(stage_two_score, dtype=np.float64)
    composite = np.asarray(composite_score, dtype=np.float64)
    staged = np.asarray(staged_score, dtype=np.float64)
    if not (len(fired) == len(stage_two) == len(composite) == len(staged)):
        raise AssertionError("staged score vectors are not aligned with rows")
    if not np.isfinite(staged).all():
        raise AssertionError("staged score contains a non-finite value")
    survivors = ~fired
    if np.any(np.isnan(stage_two[survivors])):
        raise AssertionError("a surviving row has no stage-2 score")
    if np.any(np.isnan(composite[survivors])):
        raise AssertionError("a surviving row has no composite score")
    if not np.all(np.isnan(stage_two[fired])):
        raise AssertionError(
            "a gated row carries a stage-2 score; the short circuit did not "
            "happen and the compute saving would be fictional"
        )
    if not np.all(np.isnan(composite[fired])):
        raise AssertionError(
            "a gated row carries a composite score; the short circuit did "
            "not happen"
        )
    if np.any(composite[survivors] < stage_two[survivors]):
        raise AssertionError(
            "a survivor's composite is below its stage-2 score; the "
            "composite is required to be a max"
        )
    if not np.array_equal(staged[survivors], composite[survivors]):
        raise AssertionError(
            "the staged axis does not equal the composite on survivors"
        )
    decision = np.zeros(len(fired), dtype=bool)
    decision[fired] = True
    decision[survivors] = composite[survivors] > float(threshold)
    if not np.array_equal(staged > float(threshold), decision):
        raise AssertionError(
            "the staged score does not reproduce the staged decision"
        )


def score_unstaged(
    rejector: Rejector,
    packed: np.ndarray,
    features: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    """The existing v3 path on every row, unchanged, timed as the control arm."""
    started = time.perf_counter()
    _prediction, score = base.score_rows(rejector, packed, features, device)
    return np.asarray(score, dtype=np.float64), time.perf_counter() - started


def score_staged(
    rejector: Rejector,
    stage_one: StageOneGate,
    packed: np.ndarray,
    features: np.ndarray,
    prefilter_features: np.ndarray,
    device: torch.device,
    *,
    capture_length: int,
    composite: CompositeSurvivorPolicy,
    feature_seconds: float = 0.0,
) -> StagedOutcome:
    """Stage 1 gates; survivors reach the unchanged v3 path, then the composite.

    The short circuit is real: ``base.score_rows`` -- which is the encoder
    forward, the fusion, the branch LOF and the frozen policy -- is called on the
    surviving rows alone, and gated rows never enter it.  A survivor's staged
    score is the composite :data:`COMPOSITE_SURVIVOR_SCORE`; the stage-2
    sub-score is kept unchanged alongside it so the staged/unstaged
    subset-agreement check still proves stage 2 itself did not drift.
    """
    rows = int(len(packed))
    if len(features) != rows or len(prefilter_features) != rows:
        raise ValueError("packed, feature and stage-1 rows are not aligned")

    started = time.perf_counter()
    stage_one_raw = stage_one.raw(prefilter_features, capture_length)
    gated = stage_one.fires(prefilter_features, capture_length)
    stage_one_rank = stage_one.rank(stage_one_raw)
    stage_one_seconds = time.perf_counter() - started

    survivor_index = np.flatnonzero(~gated)
    stage_two_score = np.full(rows, np.nan, dtype=np.float64)
    stage_one_survivor_rank = np.full(rows, np.nan, dtype=np.float64)
    composite_score = np.full(rows, np.nan, dtype=np.float64)
    stage_two_seconds = 0.0
    if len(survivor_index):
        started = time.perf_counter()
        _prediction, survivor_score = base.score_rows(
            rejector,
            np.asarray(packed)[survivor_index],
            np.asarray(features)[survivor_index],
            device,
        )
        stage_two_seconds = time.perf_counter() - started
        stage_two_score[survivor_index] = np.asarray(
            survivor_score, dtype=np.float64
        )
        stage_one_survivor_rank[survivor_index] = (
            composite.stage_one_survivor_rank(stage_one_raw[survivor_index])
        )
        composite_score[survivor_index] = composite.composite(
            stage_two_score[survivor_index],
            stage_one_raw[survivor_index],
        )

    staged_score = np.where(
        gated, STAGE_ONE_SCORE_OFFSET + stage_one_rank, composite_score
    )
    assert_staged_decision_equivalence(
        gated,
        stage_two_score,
        composite_score,
        staged_score,
        composite.threshold,
    )
    return StagedOutcome(
        gated=gated,
        stage_one_raw=stage_one_raw,
        stage_one_rank=stage_one_rank,
        stage_two_score=stage_two_score,
        stage_one_survivor_rank=stage_one_survivor_rank,
        composite_score=composite_score,
        staged_score=staged_score,
        survivor_index=survivor_index,
        feature_seconds=float(feature_seconds),
        stage_one_seconds=float(stage_one_seconds),
        stage_two_seconds=float(stage_two_seconds),
    )


def subset_agreement(
    outcome: StagedOutcome,
    unstaged_score: np.ndarray,
) -> dict[str, Any]:
    """Prove that scoring survivors in a smaller batch changed nothing.

    Stage 2 is row-independent, so a surviving row must receive the same score
    whether it was scored alongside the gated rows or without them.  If that ever
    stopped holding, the staged and unstaged arms would not be comparable and the
    experiment would be worthless, so it is asserted rather than assumed.
    """
    unstaged = np.asarray(unstaged_score, dtype=np.float64)
    if len(unstaged) != outcome.rows:
        raise AssertionError("unstaged scores are not aligned with staged rows")
    index = outcome.survivor_index
    if not len(index):
        return {"survivor_rows": 0, "max_abs_difference": 0.0, "exact": True}
    difference = float(
        np.max(np.abs(outcome.stage_two_score[index] - unstaged[index]))
    )
    if difference > SUBSET_SCORING_TOLERANCE:
        raise AssertionError(
            "stage 2 gave a surviving row a different score when the gated rows "
            f"were removed (max abs difference {difference}); the staged and "
            "unstaged arms are not comparable"
        )
    return {
        "survivor_rows": int(len(index)),
        "max_abs_difference": difference,
        "exact": bool(difference == 0.0),
        "tolerance": float(SUBSET_SCORING_TOLERANCE),
    }


def compute_accounting(
    outcome: StagedOutcome,
    unstaged_seconds: float,
) -> dict[str, Any]:
    """What the gate actually saved, measured on these rows."""
    staged_seconds = outcome.staged_seconds
    unstaged = float(unstaged_seconds)
    return {
        "rows": outcome.rows,
        "short_circuited": outcome.short_circuited,
        "short_circuited_fraction": (
            float(outcome.short_circuited / outcome.rows)
            if outcome.rows
            else 0.0
        ),
        "stage_two_rows_scored_staged": int(len(outcome.survivor_index)),
        "stage_two_rows_scored_unstaged": outcome.rows,
        "stage_one_feature_seconds": outcome.feature_seconds,
        "stage_one_score_seconds": outcome.stage_one_seconds,
        "stage_two_seconds_staged": outcome.stage_two_seconds,
        "staged_total_seconds": staged_seconds,
        "unstaged_total_seconds": unstaged,
        "wall_clock_delta_seconds": float(staged_seconds - unstaged),
        "wall_clock_speedup": (
            float(unstaged / staged_seconds) if staged_seconds > 0.0 else None
        ),
    }


def _sum_compute(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-cell compute accounting without re-deriving any ratio."""
    if not entries:
        raise ValueError("compute aggregation needs at least one cell")
    totals = {
        key: float(sum(float(entry[key]) for entry in entries))
        for key in (
            "stage_one_feature_seconds",
            "stage_one_score_seconds",
            "stage_two_seconds_staged",
            "staged_total_seconds",
            "unstaged_total_seconds",
        )
    }
    counts = {
        key: int(sum(int(entry[key]) for entry in entries))
        for key in (
            "rows",
            "short_circuited",
            "stage_two_rows_scored_staged",
            "stage_two_rows_scored_unstaged",
        )
    }
    totals.update(counts)
    rows = counts["rows"]
    short_circuited = counts["short_circuited"]
    staged_seconds = totals["staged_total_seconds"]
    totals.update(
        {
            "short_circuited_fraction": (
                float(short_circuited / rows) if rows else 0.0
            ),
            "wall_clock_delta_seconds": float(
                staged_seconds - totals["unstaged_total_seconds"]
            ),
            "wall_clock_speedup": (
                float(totals["unstaged_total_seconds"] / staged_seconds)
                if staged_seconds > 0.0
                else None
            ),
        }
    )
    return totals


def known_false_unknown_by_stage(
    outcome: StagedOutcome,
    unstaged_score: np.ndarray,
    staged_threshold: float,
    unstaged_threshold: float,
) -> dict[str, Any]:
    """Attribute every rejected KNOWN row to the stage that rejected it.

    Known false-unknown rate is the binding risk of the staged design, because
    stage 1 does not merely flag a known signal, it suppresses its label.  This
    splits the rate so it is always visible which stage is responsible.

    ``staged_threshold`` is the composite threshold the staged decision uses;
    survivor rejection compares the COMPOSITE against it.  The unstaged control
    is the additive path with no gate and no composite, so it keeps its own
    ``unstaged_threshold`` (the stage-2 policy's enrollment q95).
    """
    rows = outcome.rows
    if not rows:
        raise ValueError("known false-unknown attribution needs at least one row")
    gated = np.asarray(outcome.gated, dtype=bool)
    survivors = ~gated
    stage_two_reject = np.zeros(rows, dtype=bool)
    if survivors.any():
        stage_two_reject[survivors] = (
            outcome.composite_score[survivors] > float(staged_threshold)
        )
    staged_reject = gated | stage_two_reject
    unstaged_value = np.asarray(unstaged_score, dtype=np.float64)
    if (
        unstaged_value.ndim != 1
        or len(unstaged_value) != rows
        or not np.isfinite(unstaged_value).all()
    ):
        raise ValueError(
            "unstaged known scores must be a finite vector aligned with rows"
        )
    unstaged_reject = unstaged_value > float(unstaged_threshold)
    stage_one_total = int(np.count_nonzero(gated))
    stage_two_total = int(np.count_nonzero(stage_two_reject))
    staged_total = int(np.count_nonzero(staged_reject))
    partition_total = stage_one_total + stage_two_total
    if staged_total != partition_total:
        raise AssertionError(
            "known false-unknown attribution is not an exact stage-one/stage-two "
            f"partition: staged={staged_total}, partition={partition_total}"
        )
    return {
        "rows": rows,
        "staged_threshold": float(staged_threshold),
        "unstaged_threshold": float(unstaged_threshold),
        "staged_false_unknown_rate": float(np.mean(staged_reject)),
        "stage_one_gate_rate": float(np.mean(gated)),
        "stage_two_false_unknown_rate_marginal": float(
            np.count_nonzero(stage_two_reject) / rows
        ),
        "stage_two_false_unknown_rate_among_survivors": (
            float(
                np.count_nonzero(stage_two_reject) / np.count_nonzero(survivors)
            )
            if survivors.any()
            else None
        ),
        "unstaged_false_unknown_rate": float(np.mean(unstaged_reject)),
        "staged_rejected_total": staged_total,
        "rejected_by_stage_one_total": stage_one_total,
        "stage_one_also_rejected_by_unstaged_control": int(
            np.count_nonzero(gated & unstaged_reject)
        ),
        "rejected_by_stage_two_only": stage_two_total,
        "attribution_partition_total": partition_total,
        "attribution_partition_exact": True,
        "attribution": (
            "a known row counted here was NOT classified if stage_one gated it; "
            "rejected_by_stage_one_total counts every such row, while "
            "stage_one_also_rejected_by_unstaged_control reports the overlap "
            "with the counterfactual additive control without subtracting it "
            "from the staged partition. The exact staged partition is "
            "rejected_by_stage_one_total + rejected_by_stage_two_only. "
            "Survivor rejection is on the COMPOSITE axis against "
            "staged_threshold; the unstaged control uses the additive axis "
            "against unstaged_threshold"
        ),
    }


def _source_hashes(
    prefilter: NoisePrefilterSource,
    posedegen: PoseDegeneracySource,
) -> dict[str, str]:
    """The base chain, plus this module and both sibling modules."""
    hashes = dict(base._source_hashes())
    hashes["fit_v3_openset_staged.py"] = _sha256(Path(__file__).resolve())
    for item in (prefilter, posedegen):
        if item.source_sha256 is not None:
            hashes[Path(str(item.source_path)).name] = item.source_sha256
    return hashes


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

ARCHITECTURE_CONTRACT_NOTE = (
    "This rejector can decide before the classifier runs. When the stage-1 "
    "noise prefilter fires, the capture is rejected as noise and no closed-set "
    "label is produced for it at all: the encoder forward, the fusion, the "
    "branch LOF and the frozen policy are never computed for that row. Every "
    "earlier rejector in this repository was strictly additive and could only "
    "annotate a label it was forbidden to change. This one can suppress the "
    "label. That is a deliberate architecture change, made because the frozen "
    "additive blend cannot reach the noise gate (HANDOFF 17 and 19), and it "
    "must be stated in any release claim built on this module. Its cost is "
    "known false-unknown rate, which is reported per stage below so it is "
    "always visible whether the prefilter or the branch LOF rejected a known "
    "signal. Stage 2 remains additive on the rows it sees: within stage 2 the "
    "closed label is recomputed after scoring and asserted bit-identical."
)

PREFILTER_SET_PROTOCOL = (
    "--prefilter-dir is a fitted noise-prefilter set, one bundle per capture "
    "length, produced by noise_prefilter.py. This module LOADS it and never "
    "fits it: the coefficients are fit on training rows, the operating point is "
    "a known-false-positive budget chosen on enrollment rows, and the synthetic "
    "fitting noise comes from a fitting-seed namespace disjoint from every "
    "novelty seed. All of that is the prefilter module's contract and is "
    "verified here from each bundle's own provenance rather than assumed. "
    "Consequently NO novelty seed took part in choosing the stage-1 operating "
    "point. --role and --design-novelty-seed still govern the staged system's "
    "own choices -- which bundle, which prefix lengths -- and keep the audit "
    "trail. " + SEED_LEDGER_NOTE
)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output = reject_sealed_path(Path(args.output_dir), "output")

    # Fail fast, before any data is touched: an absent or non-conforming sibling
    # must stop the run, not quietly reduce it to the unstaged path.
    prefilter = load_prefilter_module(
        getattr(args, "prefilter_module", PREFILTER_MODULE_DEFAULT)
    )
    posedegen = load_posedegen_module(
        getattr(args, "posedegen_module", POSEDEGEN_MODULE_DEFAULT)
    )
    role, design_seed, novelty_seeds = validate_seed_plan(
        getattr(args, "role", None),
        getattr(args, "design_novelty_seed", None),
        args.novelty_seeds,
    )
    prefix_lengths = validate_prefix_lengths(args.prefix_lengths)
    if int(args.novelty_n) <= 0:
        raise ValueError("--novelty-n must be positive")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} is not empty")

    artifact = load_fusion_artifact(Path(args.fusion_dir))
    device = resolve_device(args.device, artifact.device)
    print(
        f"[v3 staged] role={role} design_seed={design_seed} "
        f"fusion={artifact.directory.name} "
        f"sha={artifact.directory_sha256[:16]} device={device}",
        flush=True,
    )

    rebuilt = rebuild_populations(
        artifact,
        device,
        posedegen,
        tolerance=float(args.reproduction_tolerance),
    )
    populations = rebuilt.populations
    data = populations.data
    known_length = int(rebuilt.selection_capture_length)
    enrollment_length = int(rebuilt.enrollment_capture_length)
    print(
        f"[v3 staged] rebuilt populations; reproduction "
        f"{populations.reproduction}",
        flush=True,
    )

    # Stage 1 arrives already fitted.  The set must cover every length that will
    # be scored, including the native lengths of the enrollment population (the
    # composite calibration) and the known selection population.  A sweep length
    # above the longest fitted length is covered by the causal-prefix rule (its
    # features are sliced below); every other uncovered sweep length is refused
    # inside load_stage_one.
    stage_one = load_stage_one(
        prefilter,
        posedegen,
        Path(args.prefilter_dir),
        required_lengths=(*prefix_lengths, known_length, enrollment_length),
    )
    # The known selection and enrollment populations' stage-1 features are
    # computed at their NATIVE lengths by rebuild_populations, so the
    # causal-prefix rule may not stand in for a fitted bundle there: those
    # lengths must be fitted exactly, or the gate would score native-length
    # features with a shorter-length model.  (fit_composite_policy re-checks
    # the enrollment length itself.)
    for name, length in (
        ("known selection", known_length),
        ("enrollment", enrollment_length),
    ):
        if int(length) not in stage_one.models:
            raise ValueError(
                f"{stage_one.directory} has no fitted prefilter for the "
                f"{name} population's native length N{length}; available "
                f"{list(stage_one.lengths)}.  {name} features are computed "
                "at native length, so the causal-prefix rule may not stand "
                "in for this length"
            )
    print(
        f"[v3 staged] stage1 {prefilter.module_name}@{prefilter.version} "
        f"set={stage_one.set_sha256[:16]} lengths={list(stage_one.lengths)}",
        flush=True,
    )

    # ---- stage 2: the existing v3 path, fit exactly as the unstaged path ----
    lof_components = fit_branch_lof(populations.branch_embeddings)
    enrollment_lof = score_branch_lof(
        lof_components,
        {
            branch: populations.branch_embeddings[branch][PROTOTYPE_SPLIT]
            for branch in ("real", "complex")
        },
    )
    enrollment_prediction, _distance = base.nearest(
        populations.fused[PROTOTYPE_SPLIT], populations.prototypes
    )
    enrollment_prediction = np.asarray(enrollment_prediction, dtype=np.int64)
    policy = fit_stage_two_policy(
        data["xtr"],
        np.asarray(data["ytr"], dtype=np.int64),
        data["xen"],
        enrollment_prediction,
        enrollment_lof,
        n_classes=int(data["n_classes"]),
    )
    print(
        f"[v3 staged] stage2 threshold={policy.threshold:.6f}",
        flush=True,
    )

    # ---- the composite survivor policy: enrollment only, as ever ----------
    # policy.calibration_scores is the stage-2 score of every enrollment row,
    # in enrollment row order, produced by the stage-2 fit itself.
    enrollment_stage_two_scores = np.asarray(
        policy.calibration_scores, dtype=np.float64
    )
    if len(enrollment_stage_two_scores) != len(data["xen"]):
        raise AssertionError(
            "stage-2 enrollment calibration scores are not aligned with the "
            "enrollment population"
        )
    composite = fit_composite_policy(
        stage_one,
        rebuilt.enrollment_features,
        enrollment_length,
        enrollment_stage_two_scores,
        stage_two_threshold=policy.threshold,
    )
    print(
        f"[v3 staged] composite threshold={composite.threshold:.6f} "
        f"(q{COMPOSITE_THRESHOLD_QUANTILE:.2f} on "
        f"{composite.enrollment_survivor_rows} enrollment survivors; "
        f"enrollment gate rate "
        f"{composite.enrollment_gated_rows / composite.enrollment_rows:.6f})",
        flush=True,
    )

    rejector = Rejector(
        nets=populations.nets,
        real_center=populations.real_center,
        complex_center=populations.complex_center,
        weight_real=artifact.weight_real,
        prototypes=populations.prototypes,
        lof_components=lof_components,
        policy=policy,
    )

    # ---- the immutable selection population: scored, never fit ----
    known_unstaged, known_unstaged_seconds = score_unstaged(
        rejector, data["xva"], data["fva"], device
    )
    known_outcome = score_staged(
        rejector,
        stage_one,
        data["xva"],
        data["fva"],
        rebuilt.selection_features,
        device,
        capture_length=known_length,
        composite=composite,
        feature_seconds=rebuilt.selection_feature_seconds,
    )
    known_agreement = subset_agreement(known_outcome, known_unstaged)
    known_stages = known_false_unknown_by_stage(
        known_outcome, known_unstaged, composite.threshold, policy.threshold
    )
    known_scores = known_outcome.staged_score
    known_fur = float(known_stages["staged_false_unknown_rate"])
    unstaged_known_fur = float(known_stages["unstaged_false_unknown_rate"])
    known_labels = np.asarray(data["yva"], dtype=np.int64)
    print(
        f"[v3 staged] known FUR staged={known_fur:.6f} "
        f"(stage1 {known_stages['stage_one_gate_rate']:.6f} + stage2 "
        f"{known_stages['stage_two_false_unknown_rate_marginal']:.6f}) "
        f"unstaged={unstaged_known_fur:.6f}",
        flush=True,
    )

    # ---- development novelty ----
    fixtures, fixture_provenance = base.generate_novelty(
        novelty_seeds,
        n_each=int(args.novelty_n),
        lengths=prefix_lengths,
        feature_mean=np.asarray(data["fmean"], dtype=np.float32),
        feature_std=np.asarray(data["fstd"], dtype=np.float32),
        patch_length=int(data["cache_contract"]["config"]["patch_length"]),
        patch_count=int(data["cache_contract"]["config"]["patch_count"]),
        target_frac=float(data["cache_contract"]["config"]["target_frac"]),
        progress=True,
    )
    prefilter_fixtures = generate_prefilter_features(
        posedegen,
        novelty_seeds,
        n_each=int(args.novelty_n),
        lengths=prefix_lengths,
        patch_length=int(data["cache_contract"]["config"]["patch_length"]),
        target_frac=float(data["cache_contract"]["config"]["target_frac"]),
        expected_provenance=fixture_provenance,
        stage_one=stage_one,
    )

    by_seed: dict[str, Any] = {}
    staged_rows: list[dict[str, Any]] = []
    unstaged_rows: list[dict[str, Any]] = []
    compute_cells: list[dict[str, Any]] = []
    for seed in novelty_seeds:
        by_length: dict[str, Any] = {}
        for length in prefix_lengths:
            row: dict[str, Any] = {}
            unstaged_row: dict[str, Any] = {}
            staged_family: dict[str, np.ndarray] = {}
            unstaged_family: dict[str, np.ndarray] = {}
            for family in NOVELTY_FAMILIES:
                fixture = fixtures[seed][family][length]
                stage_one_fixture = prefilter_fixtures[seed][family][length]
                unstaged_score, unstaged_seconds = score_unstaged(
                    rejector, fixture.packed, fixture.features, device
                )
                outcome = score_staged(
                    rejector,
                    stage_one,
                    fixture.packed,
                    fixture.features,
                    stage_one_fixture.features,
                    device,
                    capture_length=int(length),
                    composite=composite,
                    feature_seconds=stage_one_fixture.seconds,
                )
                agreement = subset_agreement(outcome, unstaged_score)
                compute = compute_accounting(outcome, unstaged_seconds)
                compute["family"] = family
                compute["novelty_seed"] = int(seed)
                compute["prefix_length"] = int(length)
                compute["prefilter_module_compute_saving"] = (
                    prefilter.module.compute_saving(outcome.gated)
                )
                compute_cells.append(compute)

                staged_family[family] = outcome.staged_score
                unstaged_family[family] = unstaged_score
                row[family] = {
                    **family_metrics(
                        outcome.staged_score, known_scores, composite.threshold
                    ),
                    "gated_at_stage_one_fraction": float(
                        np.mean(outcome.gated)
                    ),
                    "rejected_at_stage_two_fraction": (
                        float(
                            np.count_nonzero(
                                outcome.composite_score[outcome.survivor_index]
                                > composite.threshold
                            )
                            / outcome.rows
                        )
                        if outcome.rows
                        else 0.0
                    ),
                    "stage_one_raw_median": float(
                        np.median(outcome.stage_one_raw)
                    ),
                    "compute": compute,
                    "subset_scoring_agreement": agreement,
                }
                unstaged_row[family] = family_metrics(
                    unstaged_score, known_unstaged, policy.threshold
                )
            row["overall"] = family_metrics(
                np.concatenate(
                    [staged_family[family] for family in NOVELTY_FAMILIES]
                ),
                known_scores,
                composite.threshold,
            )
            unstaged_row["overall"] = family_metrics(
                np.concatenate(
                    [unstaged_family[family] for family in NOVELTY_FAMILIES]
                ),
                known_unstaged,
                policy.threshold,
            )
            row["known_false_unknown_rate"] = known_fur
            row["known_false_unknown_by_stage"] = known_stages
            unstaged_row["known_false_unknown_rate"] = unstaged_known_fur
            by_length[str(length)] = row
            staged_rows.append(row)
            unstaged_rows.append(unstaged_row)
            print(
                f"  [seed {seed} N{length}] staged noise "
                f"{row['noise']['auroc']:.6f}/"
                f"{row['noise']['threshold_recall']:.6f} chirp "
                f"{row['chirp']['auroc']:.6f}/"
                f"{row['chirp']['threshold_recall']:.6f} overall "
                f"{row['overall']['auroc']:.6f} | gated noise "
                f"{row['noise']['gated_at_stage_one_fraction']:.3f} chirp "
                f"{row['chirp']['gated_at_stage_one_fraction']:.3f}",
                flush=True,
            )
        by_seed[str(seed)] = by_length

    summary = gate_summary(staged_rows, known_fur)
    unstaged_summary = gate_summary(unstaged_rows, unstaged_known_fur)
    compute_total = _sum_compute(compute_cells)

    output.mkdir(parents=True, exist_ok=True)
    stage_two_path = output / "v3_open_policy_stage_two.npz"
    np.savez_compressed(stage_two_path, **policy.to_payload())
    lof_path = output / "v3_branch_lof_components.npz"
    save_branch_lof(lof_path, lof_components)
    composite_path = output / COMPOSITE_POLICY_FILENAME
    save_composite_policy(composite_path, composite)
    # Prove the serialization round-trips before the report cites it.
    load_composite_policy(
        composite_path, expected_stage_two_threshold=policy.threshold
    )

    prefix = "design_selection" if role == "design" else "development_openset"
    report = {
        "status": (
            f"{prefix}_pass" if summary["all_pass"] else f"{prefix}_fail"
        ),
        "role": role,
        "evidence_role": (
            "hyperparameter selection on a fresh design novelty seed; the gate "
            "block below is NOT validation evidence and must not be quoted as "
            "a result"
            if role == "design"
            else "validation on untouched development novelty seeds; the gate "
            "block below is the evidence"
        ),
        "gates_are_evidence": bool(role == "validate"),
        "development_only": True,
        "release_evidence": False,
        "sealed_release_data_used": 0,
        "consumed_test_rows_used": 0,
        "sealed_release_paths_read": 0,
        "release_seed_20260736_used": False,
        "consumed_sealed_release_seeds_excluded": sorted(
            CONSUMED_SEALED_RELEASE_SEEDS
        ),
        # ---- the architecture contract change, stated explicitly ----
        "closed_label_can_be_gated": True,
        "closed_label_can_be_gated_note": ARCHITECTURE_CONTRACT_NOTE,
        "additive_only": False,
        "changes_closed_label": True,
        "gates_before_classification": True,
        "architecture_contract_change": str(
            getattr(prefilter.module, "ARCHITECTURE_CONTRACT_CHANGE")
        ),
        "rejector_is_additive_only": False,
        "stage_two_closed_label_altered": False,
        "closed_label_contract": (
            "STAGE 1 IS A BEHAVIOURAL GATE, NOT AN ADDITIVE SCORE: a capture "
            "the noise prefilter fires on is rejected before classification and "
            "receives no closed-set label. STAGE 2 is additive as before: for "
            "every row it sees, the nearest-prototype label is recomputed after "
            "scoring and asserted bit-identical."
        ),
        "architecture": {
            "kind": STAGED_POLICY_KIND,
            "schema": int(STAGED_POLICY_SCHEMA),
            "staged_policy_version": STAGED_POLICY_VERSION,
            "staged_policy_version_history": dict(
                STAGED_POLICY_VERSION_HISTORY
            ),
            "survivor_score": COMPOSITE_SURVIVOR_SCORE,
            "stage_one": (
                "noise prefilter on pose-degeneracy features; coefficients fit "
                "on training rows, operating point a known-false-positive "
                "budget on its OWN enrollment score distribution, one bundle "
                "per capture length"
            ),
            "stage_two": (
                "fit_v3_openset unchanged and imported: branch-LOF rank blended "
                f"with the {FROZEN_GEOMETRY_FEATURE} geometry rank at the frozen "
                f"{BRANCH_LOF_RANK_WEIGHT}/{GEOMETRY_RANK_WEIGHT} weights, "
                f"internally thresholded at the enrollment q{THRESHOLD_QUANTILE}"
            ),
            "composite": (
                f"a stage-1 survivor's staged score is {COMPOSITE_SURVIVOR_SCORE}; "
                "the staged unknown threshold is the frozen "
                f"q{COMPOSITE_THRESHOLD_QUANTILE} of the composite over "
                "enrollment stage-1 survivors (enrollment only). Gated rows "
                "keep the 1 + softsign(stage-1 log-odds) axis above every "
                "survivor"
            ),
            "stage_two_policy_kind": FROZEN_POLICY_KIND,
            "stage_one_fitted_here": False,
            "stage_two_refit_on_survivors": False,
            "stage_two_refit_on_survivors_note": (
                "stage 2 is fit on the FULL enrollment population, exactly as "
                "the unstaged path fits it. Refitting it on stage-1 survivors "
                "would confound the effect of the gate with the effect of a "
                "refit; leaving it identical makes the staged/unstaged "
                "comparison in this report an experiment rather than an anecdote"
            ),
            "why_staged": (
                "the frozen blend must carry the branch-LOF term, which on v3 "
                "embeddings ranks noise as known (HANDOFF 17: noise AUROC "
                "0.6038), so q95 fit on the fused known distribution puts the "
                "operating point in the wrong place. Deciding noise first, "
                "against its own score distribution, removes that dependency"
            ),
            "why_composite": COMPOSITE_RATIONALE,
            "staged_score_note": STAGED_SCORE_NOTE,
        },
        "fusion": {
            "directory": str(artifact.directory),
            "directory_sha256": artifact.directory_sha256,
            "file_sha256": artifact.file_sha256,
            "weight_real": artifact.weight_real,
            "seed": artifact.seed,
            "recorded_device": artifact.device,
            "closed_selection_balanced_accuracy": artifact.metrics[
                "closed_selection"
            ]["balanced_accuracy"],
            "source_branches": {
                name: {
                    "directory": artifact.metrics["source_branches"][name][
                        "directory"
                    ],
                    "sha256": populations.branch_hashes[name],
                }
                for name in ("real", "complex")
            },
            "reproduction": populations.reproduction,
        },
        "fitting_contract": {
            "density_population": "training only",
            "density_rows": int(len(data["xtr"])),
            "density_components": [
                "stage 1: fitted elsewhere on training rows; loaded here",
                "stage 2: per-class time-domain geometry (KnownOnlyGeometry)",
                "stage 2: branch LOF reference population and k-distances",
            ],
            "rank_and_threshold_population": "enrollment only",
            "rank_rows": int(len(data["xen"])),
            "composite_rank_and_threshold_population": (
                "enrollment stage-1 survivors only"
            ),
            "stage_one_fitted_in_this_run": False,
            "stage_one_training_features_computed_here": False,
            "stage_one_enrollment_features_computed_here": True,
            "stage_one_enrollment_features_role": (
                f"the composite survivor rank calibration and its "
                f"q{COMPOSITE_THRESHOLD_QUANTILE} "
                "threshold, and nothing else. Every rank and threshold in "
                "this chain is calibrated on enrollment; the stage-1 gate "
                "itself stays exactly as loaded"
            ),
            "stage_one_features_computed_here": [
                "enrollment (composite rank/threshold calibration only)",
                "selection (scored only)",
                "novelty (scored only)",
            ],
            "selection_population_role": (
                "scored for metrics only; never fit, calibrated, or selected on"
            ),
            "novelty_population_role": (
                "scored for metrics only; never fit, ranked, or thresholded. "
                + (
                    "In this design run the novelty seed is the declared design "
                    "seed and its scores may inform staged-system choices; that "
                    "is why this report is not evidence."
                    if role == "design"
                    else "In this validation run the novelty seeds never inform "
                    "any policy choice."
                )
            ),
            "policy_searched_here": False,
            "sealed_release_rows_loaded": 0,
            "consumed_test_rows_loaded": 0,
            "reproduction_tolerance": float(args.reproduction_tolerance),
        },
        "stage_one": {
            **prefilter.provenance(),
            **stage_one.provenance(),
            "posedegen": posedegen.provenance(),
            "known_capture_length": known_length,
            "selection_features_sha256": _matrix_sha256(
                rebuilt.selection_features
            ),
            "selection_feature_seconds": float(
                rebuilt.selection_feature_seconds
            ),
            "decision": "gated as noise iff score >= threshold_score",
            "selection_protocol": PREFILTER_SET_PROTOCOL,
        },
        "stage_two": {
            "kind": FROZEN_POLICY_KIND,
            "geometry_feature": FROZEN_GEOMETRY_FEATURE,
            "branch_lof_rank_weight": float(BRANCH_LOF_RANK_WEIGHT),
            "geometry_rank_weight": float(GEOMETRY_RANK_WEIGHT),
            "threshold_quantile": float(THRESHOLD_QUANTILE),
            "threshold": float(policy.threshold),
            "threshold_role": (
                "the stage-2 policy's own internal enrollment q95, untouched. "
                "The STAGED decision threshold is the composite block's "
                "threshold; this one still governs the unstaged control arm"
            ),
            "branch_lof_components": [
                {"branch": branch, "weight": weight, "neighbors": neighbors}
                for branch, weight, neighbors in BRANCH_LOF
            ],
            "branch_lof_hyperparameters_source": (
                "inherited from the v2 ensemble and re-fit on v3 embeddings; "
                "not re-searched here"
            ),
            "cannot_change_closed_label": True,
        },
        "composite": {
            **composite.provenance(),
            "enrollment_features_sha256": _matrix_sha256(
                rebuilt.enrollment_features
            ),
            "enrollment_feature_seconds": float(
                rebuilt.enrollment_feature_seconds
            ),
        },
        "seeds": {
            "model_seed": artifact.seed,
            "fusion_artifact_seed": artifact.seed,
            "novelty_seeds": list(novelty_seeds),
            "design_novelty_seed": int(design_seed),
            "spent_novelty_seeds_excluded": sorted(SPENT_NOVELTY_SEEDS),
            "spent_novelty_seed_reasons": {
                str(seed): reason
                for seed, reason in sorted(SPENT_NOVELTY_SEEDS.items())
            },
            "first_clean_novelty_seed": int(FIRST_CLEAN_NOVELTY_SEED),
            "default_validation_novelty_seeds": list(
                DEFAULT_VALIDATION_NOVELTY_SEEDS
            ),
            "consumed_sealed_release_seeds_excluded": {
                str(seed): reason
                for seed, reason in sorted(
                    CONSUMED_SEALED_RELEASE_SEEDS.items()
                )
            },
            "release_seed_not_spent": RELEASE_SEED_NEVER_SPENT_HERE,
            "novelty_rows_used_to_choose_the_stage_one_operating_point": 0,
            "novelty_rows_used_to_choose_the_composite_threshold": 0,
            "ledger": SEED_LEDGER_NOTE,
        },
        "protocol": {
            "novelty_families": list(NOVELTY_FAMILIES),
            "novelty_n_each": int(args.novelty_n),
            "prefix_lengths": list(prefix_lengths),
            "generation": (
                f"each novelty row generated exactly once at N{max(prefix_lengths)}; "
                "shorter lengths are exact raw prefixes of the same capture"
            ),
            "stage_one_features_verified_against_stage_two_captures": True,
            "stage_one_feature_length_by_prefix_length": {
                str(length): int(stage_one.feature_length_for(length))
                for length in prefix_lengths
            },
            "stage_one_causal_prefix_rule_lengths": [
                int(length)
                for length in prefix_lengths
                if int(stage_one.feature_length_for(length)) != int(length)
            ],
            "novelty_frontend": td_preprocess.PREPROCESS_VERSION,
            "known_reference": (
                f"immutable development-selection population at its native "
                f"N{known_length}; known FUR is therefore constant across "
                "novelty prefix lengths"
            ),
            "known_rows": int(len(known_scores)),
            "known_classes": int(data["n_classes"]),
        },
        "known": {
            "n": int(len(known_scores)),
            "false_unknown_rate": known_fur,
            "by_stage": known_stages,
            "closed_selection_accuracy_on_stage_one_survivors": (
                float(
                    np.mean(
                        base.nearest(
                            populations.fused[METRIC_SPLIT][
                                known_outcome.survivor_index
                            ],
                            populations.prototypes,
                        )[0]
                        == known_labels[known_outcome.survivor_index]
                    )
                )
                if len(known_outcome.survivor_index)
                else None
            ),
            "closed_selection_accuracy_all_rows": float(
                np.mean(
                    base.nearest(
                        populations.fused[METRIC_SPLIT], populations.prototypes
                    )[0]
                    == known_labels
                )
            ),
            "score_median": float(np.median(known_scores)),
            "stage_one_raw_median": float(
                np.median(known_outcome.stage_one_raw)
            ),
            "subset_scoring_agreement": known_agreement,
            "compute": compute_accounting(known_outcome, known_unstaged_seconds),
        },
        "novelty": by_seed,
        "fixture_provenance": fixture_provenance,
        **summary,
        "unstaged_control": {
            "description": (
                "the same rows through fit_v3_openset unchanged, with no gate. "
                "The only difference from the staged arm is stage 1, so the "
                "delta is attributable to the gate"
            ),
            "known_false_unknown_rate": unstaged_known_fur,
            **unstaged_summary,
        },
        "compute_saved": {
            "note": (
                "measured, not asserted. Stage-1 feature extraction is "
                "per-capture NumPy while stage 2 is a batched tensor forward, "
                "so the staged path can be slower in wall clock even though it "
                "skips strictly more model work. Both numbers are reported."
            ),
            "novelty_total": compute_total,
            "by_cell": compute_cells,
            "known_selection": compute_accounting(
                known_outcome, known_unstaged_seconds
            ),
        },
        "artifacts": {
            stage_two_path.name: _sha256(stage_two_path),
            lof_path.name: _sha256(lof_path),
            composite_path.name: _sha256(composite_path),
        },
        "source_sha256": _source_hashes(prefilter, posedegen),
        "data_audit": data["data_audit"],
        "source_index_contract": {
            "contract": data["cache_contract"],
            "role": (
                "identifies the raw corpus and exposed train/enroll/selection "
                "indices only; every tensor here was rebuilt with the v3 "
                "time-domain frontend and the v3 fusion"
            ),
        },
        "provenance": {
            "command": [sys.executable, *sys.argv],
            "cwd": str(Path.cwd()),
            "device": str(device),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "wall_clock_s": time.perf_counter() - started,
    }
    report_path = output / "openset_metrics.json"
    write_json(report_path, report)
    print(
        f"[v3 staged] wrote {report_path} status={report['status']}",
        flush=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog=ARCHITECTURE_CONTRACT_NOTE + "  " + PREFILTER_SET_PROTOCOL,
    )
    parser.add_argument(
        "--fusion-dir",
        required=True,
        help="a v3 fusion artifact directory written by assemble_v3_fusion.py",
    )
    parser.add_argument(
        "--prefilter-dir",
        required=True,
        help=(
            "a fitted noise-prefilter set written by noise_prefilter.py: one "
            "N{length} bundle per capture length. Loaded, never re-fit here"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--role",
        required=True,
        choices=ROLES,
        help=(
            "'design' draws novelty at the declared design seed and marks the "
            "report as selection, not evidence; 'validate' draws novelty at "
            "untouched seeds and refuses any spent, sealed, release or design "
            "seed"
        ),
    )
    parser.add_argument(
        "--design-novelty-seed",
        required=True,
        type=int,
        default=None,
        help=(
            "the novelty seed the staged system's own choices were (or will be) "
            "made on. Emitted into the report. A design run requires the "
            "unspent reserved design seed; a validation run requires that "
            "same seed to have been frozen in the spent ledger. Consumed "
            f"sealed seeds {sorted(CONSUMED_SEALED_RELEASE_SEEDS)}, release "
            f"seed {RELEASE_SEED_NEVER_SPENT_HERE}, and reserved validation "
            f"seeds {list(DEFAULT_VALIDATION_NOVELTY_SEEDS)} are never legal "
            "design seeds."
        ),
    )
    parser.add_argument(
        "--prefilter-module",
        default=PREFILTER_MODULE_DEFAULT,
        help="import name of the noise prefilter module",
    )
    parser.add_argument(
        "--posedegen-module",
        default=POSEDEGEN_MODULE_DEFAULT,
        help="import name of the pose-degeneracy feature module",
    )
    parser.add_argument(
        "--device",
        choices=("artifact", "auto", "cpu", "mps"),
        default="artifact",
        help="'artifact' reuses the device recorded in the fusion artifact",
    )
    parser.add_argument(
        "--novelty-seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "novelty seeds to draw. With --role validate this defaults to "
            f"{list(DEFAULT_VALIDATION_NOVELTY_SEEDS)}. With --role design it "
            "must be given explicitly and must exclude that pair."
        ),
    )
    parser.add_argument("--novelty-n", type=int, default=DEFAULT_NOVELTY_N)
    parser.add_argument(
        "--prefix-lengths",
        type=int,
        nargs="+",
        default=list(V34_DESIGN_PREFIX_LENGTHS),
    )
    parser.add_argument(
        "--reproduction-tolerance",
        type=float,
        default=DEFAULT_REPRODUCTION_TOLERANCE,
        help=(
            "maximum tolerated disagreement between the rebuilt fusion and the "
            "centers/prototypes stored in the fusion artifact"
        ),
    )
    return parser


def resolve_novelty_seeds(args: argparse.Namespace) -> list[int]:
    """Apply the role-dependent default for ``--novelty-seeds``.

    A design run has no default: the seeds it draws are a decision, and the
    validation pair must never be the fallback for one.
    """
    if args.novelty_seeds is not None:
        return [int(seed) for seed in args.novelty_seeds]
    if validate_role(args.role) == "design":
        raise ValueError(
            "--role design requires explicit --novelty-seeds; the validation "
            f"pair {list(DEFAULT_VALIDATION_NOVELTY_SEEDS)} is never a design "
            "default"
        )
    return list(DEFAULT_VALIDATION_NOVELTY_SEEDS)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    args.novelty_seeds = resolve_novelty_seeds(args)
    return run(args)


if __name__ == "__main__":
    main()
