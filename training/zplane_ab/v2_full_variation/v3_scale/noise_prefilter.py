"""A staged noise/not-noise PREFILTER over pose-degeneracy features.

Why this module exists
======================

HANDOFF section 19 measured a specific, reproducible shape:

* pose-degeneracy features separate noise well **standalone** -- worst-of 2
  novelty seeds x 3 prefix lengths, ``carrier_phase_coherence`` 0.9154,
  ``log10_relative_lag_magnitude_mean`` 0.9004,
  ``relative_lag_magnitude_decay_slope`` 0.8872,
  ``high_lag_qualified_fraction`` 0.8579 noise AUROC;
* folding them into the frozen fixed-weight rank blend lifts fused noise AUROC
  only to ~0.74 and leaves threshold recall at 0.01-0.07 against a >= 0.10 gate.

The blend must carry the branch-LOF term, and on v3 embeddings that term ranks
**noise as known** (section 17: noise AUROC 0.6038).  A fixed blend containing a
harmful term cannot be rescued by adding a good one; raising the good term's
weight only trades noise against chirp.

**The diagnosis that matters is the operating point, not the ranking.**  The
frozen policy sets its threshold as the q95 of the *fused* known score, and that
fused score is dominated by the branch LOF.  A quantile of the wrong
distribution lands the threshold in the wrong place, which is exactly why a
0.9154 single-feature separation can coexist with 0.01 threshold recall.  A
dedicated detector sets its own threshold against its own score distribution.

So this module is a **binary noise detector** that decides noise *first*, before
the branch-LOF term enters at all.  Chirp is deliberately **not** its job: the
existing v3 path already solves chirp (AUROC 0.9716, threshold recall
0.72-1.00) and continues to handle it downstream, unchanged.

ARCHITECTURE CONTRACT CHANGE, stated plainly
============================================

Today's rejector is strictly **additive**.  ``fit_v3_openset`` asserts the
closed label is bit-identical before and after scoring
(``assert_closed_label_unchanged``), and every emitted report carries
``cannot_change_closed_label: true``.

**A gating prefilter does not satisfy that contract, and is not meant to.**  It
can abstain *before* classification runs, so on a gated row there is no closed
label at all.  That is a deliberate change of behaviour, not an oversight.  It
is recorded here in three places so it cannot be lost:

* the constant :data:`ARCHITECTURE_CONTRACT_CHANGE`;
* ``metadata()["changes_closed_label"] == True`` and
  ``metadata()["additive_only"] == False``, which are written into every
  serialised bundle and must appear in any release claim;
* :data:`CONTRACT_KEYS_FOR_RELEASE_CLAIM`, the exact key set a release report is
  required to carry.

The same property has an upside that must be **measured rather than asserted**:
a gated row never reaches the encoder, the fusion, the prototype distance or the
branch LOF.  :func:`compute_saving` reports the realised gated fraction on
whatever stream it is given.  It reports nothing about cost that was not counted.

Evidence contract
=================

* **Fit on TRAINING only.**  :func:`fit_prefilter` requires an explicit
  ``known_population`` argument and refuses anything but
  :data:`TRAIN_SPLIT`.  Coefficients, the standardiser mean and the
  standardiser scale are all functions of training known rows and of
  fitting-seed synthetic noise.  Nothing else.
* **Threshold on ENROLLMENT only.**  :func:`fit_threshold` requires an explicit
  ``population`` argument and refuses anything but :data:`ENROLLMENT_SPLIT`.
  The operating point is a stated known-false-positive budget, not a quantile
  inherited from some other model's score distribution.
* **The fitting noise comes from a fitting-only seed.**  ``--fitting-seed`` is
  required, has no default, and :func:`validate_fitting_seed` refuses every seed
  in the ledger: the consumed sealed seed 20260729, the unspent release seed
  20260731, the spent novelty seeds 20260938 (policy design), 20260939/20260940
  (feature scoring) and 20260941 (blend weight), **and the entire novelty seed
  namespace** :data:`NOVELTY_SEED_NAMESPACE`, because 20260942 onward are clean
  validation evidence and fitting against one would convert it.  A fitting seed
  must come from the disjoint band :data:`FITTING_SEED_NAMESPACE`.
* **Selection is scored, never fit.**  The immutable selection half is not read
  here at all.

Feature set, and why
====================

Twenty-one pose-degeneracy features exist.  This detector uses **seven**.  The
exclusions are recorded in :data:`DROPPED_FEATURES`, each with its reason, and
:func:`selection_rationale` returns the whole table for the emitted report.

Three rules, all of which were fixed before this detector was fitted:

1. **Measured dead** (HANDOFF 19.1): ``degenerate_full_band_fallback`` (AUROC
   exactly 0.5000 -- the estimator's own degeneracy flag never fires on any
   development or novelty row), ``bandwidth_clipped_to_full`` (0.5009),
   ``zero_magnitude_prefix_fraction`` (0.5066), ``quotient_clip_fraction``
   (0.5144).
2. **Continuous columns only.**  Binary indicators and small-support discrete
   codes are excluded by design: they are near-constant on the fit population,
   a constant column cannot be standardised at all, and a near-constant one
   turns the standardiser into a divide-by-noise.  This rule is stated
   structurally, and it happens to subsume three of the four measured-dead
   features, which is corroboration rather than selection.  It removes
   ``bandwidth_clipped_to_minimum`` and ``log2_selected_base_lag`` as well.
3. **One member per mechanism family.**  Several features are the same
   underlying statistic under a different reduction; keeping both spends model
   capacity on a rotation of one axis.  The retained member of each family and
   the dropped near-duplicates are named individually in
   :data:`DROPPED_FEATURES`.

Model choice, and why
=====================

**Ridge-regularised logistic regression on seven standardised features.**
Fitted by damped Newton/IRLS, which is deterministic: no sampling, no shuffling,
no early stop on a random schedule, a fixed iteration cap and a fixed tolerance.

Why this rather than something larger:

* The whole artifact is 7 means, 7 scales, 7 coefficients, 1 intercept and 1
  threshold.  A human can read it, and the TypeScript port is a dot product and
  a logistic -- no tree walk, no neighbour search, no matrix decomposition at
  inference.
* It gives a monotone score with a probability-shaped output, which is exactly
  what a ``predict_proba`` plus hard-gate contract wants.
* The decision boundary is a hyperplane in a space whose axes are individually
  interpretable, so a failure is attributable to a named feature.

Why not LDA: LDA is the same hyperplane under a shared-covariance Gaussian
assumption, and these features violate it visibly -- ``carrier_phase_coherence``
and ``high_lag_qualified_fraction`` are bounded on [0, 1] and pile up at their
endpoints for coherent emissions, and the noise class is far more dispersed than
the known class.  Logistic regression makes no distributional claim, only that
the log-odds are linear in the standardised features, and it is no larger.

Why not a single-feature threshold rule: ``carrier_phase_coherence`` alone
scored 0.9154, so a one-feature rule is a legitimate baseline and is available
here as ``--features carrier_phase_coherence``.  It is not the default because
the four strong features are not redundant with one another by construction
(phase consistency, magnitude level, magnitude decay and estimator admission are
four different failure modes of the pose estimate), and because a one-feature
rule has no way to trade them off at a fixed known-false-positive budget.

The ridge penalty is **not** cosmetic.  Known rows and white noise are close to
linearly separable in these features, and unregularised logistic regression
diverges on separable data: coefficients run to infinity and the fit depends on
the iteration cap.  :data:`L2_PENALTY` is declared, small, and was **not** tuned
against any validation seed.  It affects the fitted direction slightly and is
therefore reported, not hidden.

Length
======

The features are length-dependent by construction: for white noise the relative
lag magnitude decays as ``1/sqrt(N)``, so the same feature value means different
things at N=4096 and N=16384.  One model is therefore fitted **per capture
length**, and :class:`NoisePrefilter` records the length it was fitted at and
refuses to score a matrix that was not extracted at that length (the caller
declares the length; the module cannot infer it from features).
:func:`save_prefilter_set` / :func:`load_prefilter_set` handle the usual
three-length set as one directory.

The operating point is the point
================================

AUROC is reported because it is comparable to sections 17 and 19.  It is not the
gate.  **Threshold recall is the gate**, at a known-false-positive budget chosen
here and not inherited:
:data:`DEFAULT_KNOWN_FALSE_POSITIVE_BUDGET` is 0.02, argued in its own docstring
against the 0.10 known-false-unknown ceiling and the 0.0466 the existing v3
rejector already spends.  It is a required, configurable argument;
:func:`evaluate` reports the realised known false-positive rate alongside
recall, so an operating point that overspends its budget is visible.

Determinism and serialisation
=============================

The bundle is :data:`BUNDLE_FILES`: five ``.npy`` arrays and one ``meta.json``.
No pickle, no ``npz`` (numpy's zip writer stamps entry timestamps), no
``created_utc``, no elapsed seconds, no host name, no absolute paths.
:func:`bundle_sha256` therefore depends only on the fitted numbers and the
declared contract, and two runs of the same fit on the same inputs produce the
same SHA.

Frequency transforms
====================

Inference here is arithmetic on pose-degeneracy features, which are themselves
FFT-free.  The **offline** fitting-noise generator ``openset_eval.NOVELTY`` does
call ``np.fft``; that is the house novelty generator, it runs only at fit time,
and using a different one would silently change what "noise" means relative to
sections 16-19.  ``metadata()`` states both facts separately rather than
averaging them into one reassuring flag.

This module produces development evidence.  It is not release evidence, and
release seed 20260731 is neither read nor reachable from here.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


_V3_DIR = os.path.dirname(os.path.abspath(__file__))
_V2_DIR = os.path.dirname(_V3_DIR)
_ZPAB_DIR = os.path.dirname(_V2_DIR)
_TRAINING_DIR = os.path.dirname(_ZPAB_DIR)
for _path in (_TRAINING_DIR, _ZPAB_DIR, _V2_DIR, _V3_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import pose_degeneracy as pdg  # noqa: E402


NOISE_PREFILTER_VERSION = "noise-prefilter-v1"
BUNDLE_SCHEMA = "noise-prefilter-bundle-v1"
REPORT_SCHEMA = "noise-prefilter-fit-v1"


# ---------------------------------------------------------------------------
# the architecture contract change
# ---------------------------------------------------------------------------

ARCHITECTURE_CONTRACT_CHANGE = (
    "The v3 open-set rejector is strictly additive and never alters the closed "
    "label. This noise prefilter is a GATE: when it fires, classification does "
    "not run and the row has no closed label at all. That is a deliberate "
    "change to the architecture contract, not an oversight, and it must be "
    "stated in the emitted JSON and in any release claim. It also means a gated "
    "row skips the encoder, the fusion, the prototype distance and the branch "
    "LOF; that saving is measured by compute_saving() and is never asserted."
)

#: The exact keys a report or release claim must carry so the contract change
#: cannot be dropped between this module and a claim made about it.
CONTRACT_KEYS_FOR_RELEASE_CLAIM = (
    "additive_only",
    "changes_closed_label",
    "gates_before_classification",
    "architecture_contract_change",
)


# ---------------------------------------------------------------------------
# populations
# ---------------------------------------------------------------------------

#: The only population coefficients may be fitted on.
TRAIN_SPLIT = "train"
#: The only population the threshold may be chosen on.
ENROLLMENT_SPLIT = "enroll"
#: Scored, never fit; not read by this module at all.
SELECTION_SPLIT = "val"

FITTABLE_POPULATIONS = (TRAIN_SPLIT,)
THRESHOLDABLE_POPULATIONS = (ENROLLMENT_SPLIT,)


# ---------------------------------------------------------------------------
# the seed ledger
# ---------------------------------------------------------------------------

#: Consumed sealed v2 release suite (evidence rule 2).
CONSUMED_SEALED_SEED = 20260729
#: Unspent release seed (evidence rule 3).  Never reachable from here.
RELEASE_SEED_NEVER_SPENT_HERE = 20260731

#: Novelty seeds already spent, with what spent them.
SPENT_NOVELTY_SEEDS: dict = {
    20260938: "original frozen open-set policy design",
    20260939: "v3 refit validation and pose-degeneracy feature scoring",
    20260940: "v3 refit validation and pose-degeneracy feature scoring",
    20260941: "pose-degeneracy blend-weight design",
}

#: The whole novelty-seed namespace, inclusive.  Everything in it is either
#: spent or reserved as clean validation evidence, so none of it may be fit on.
NOVELTY_SEED_NAMESPACE = (20260900, 20260999)
#: Clean validation evidence for this work starts here (HANDOFF 19.3).
CLEAN_VALIDATION_SEEDS_FROM = 20260942

#: A disjoint band reserved for fit-only noise draws.  Structurally separate
#: from the novelty namespace so a fitting seed can never collide with a
#: validation seed, whichever validation seeds get drawn later.
FITTING_SEED_NAMESPACE = (20261000, 20261999)
#: Proposed default for a caller who has no preference.  Deliberately not a
#: default value on the argument: the fitting seed must be declared.
PROPOSED_FITTING_SEED = 20261001


def _in_range(value: int, bounds: Sequence[int]) -> bool:
    return int(bounds[0]) <= int(value) <= int(bounds[1])


def validate_fitting_seed(seed: Any) -> int:
    """Return ``seed`` or refuse it.

    The fitting seed generates the synthetic noise the coefficients are fitted
    against.  It is therefore training evidence, and anything it touches stops
    being validation evidence.  Every ledger seed is refused by name, the entire
    novelty namespace is refused as a block, and the seed is required to live in
    :data:`FITTING_SEED_NAMESPACE`.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        # Deliberately strict.  int("20261001") and int(20261001.5) both
        # succeed, and a seed that silently truncates or parses is a seed whose
        # emitted provenance does not describe the draw that happened.
        raise ValueError(
            f"fitting seed must be an integer, got {type(seed).__name__}"
        )
    value = int(seed)
    if value == CONSUMED_SEALED_SEED:
        raise ValueError(
            f"{value} is the consumed sealed v2 release suite seed and may "
            "never be fitted against"
        )
    if value == RELEASE_SEED_NEVER_SPENT_HERE:
        raise ValueError(
            f"{value} is the unspent release seed and is not a fitting seed"
        )
    if value in SPENT_NOVELTY_SEEDS:
        raise ValueError(
            f"novelty seed {value} is spent ({SPENT_NOVELTY_SEEDS[value]}) and "
            "may not be reused as fitting evidence"
        )
    if _in_range(value, NOVELTY_SEED_NAMESPACE):
        raise ValueError(
            f"{value} lies in the novelty seed namespace "
            f"{NOVELTY_SEED_NAMESPACE[0]}-{NOVELTY_SEED_NAMESPACE[1]}; seeds "
            f"from {CLEAN_VALIDATION_SEEDS_FROM} onward are the clean "
            "validation evidence for this work and fitting against one would "
            "convert it into training evidence"
        )
    if not _in_range(value, FITTING_SEED_NAMESPACE):
        raise ValueError(
            f"fitting seed must lie in the fit-only band "
            f"{FITTING_SEED_NAMESPACE[0]}-{FITTING_SEED_NAMESPACE[1]}, got "
            f"{value}; that band is disjoint from the novelty namespace so a "
            "fitting draw can never collide with a validation draw"
        )
    return value


def validate_validation_seeds(seeds: Sequence[int]) -> tuple:
    """Return the distinct validation novelty seeds, or refuse them.

    Mirror image of :func:`validate_fitting_seed`: refuses the spent ledger, the
    sealed and release seeds, and anything from the fit-only band, and requires
    every seed to be clean (>= :data:`CLEAN_VALIDATION_SEEDS_FROM`).
    """
    values = tuple(int(seed) for seed in seeds)
    if not values or len(set(values)) != len(values):
        raise ValueError("validation seeds must be a non-empty set of distinct ints")
    for value in values:
        if value == CONSUMED_SEALED_SEED:
            raise ValueError(f"{value} is the consumed sealed seed")
        if value == RELEASE_SEED_NEVER_SPENT_HERE:
            raise ValueError(
                f"{value} is the unspent release seed and is not a development "
                "novelty seed"
            )
        if value in SPENT_NOVELTY_SEEDS:
            raise ValueError(
                f"novelty seed {value} is spent "
                f"({SPENT_NOVELTY_SEEDS[value]}) and is no longer clean evidence"
            )
        if _in_range(value, FITTING_SEED_NAMESPACE):
            raise ValueError(
                f"{value} lies in the fit-only band and cannot be validation "
                "evidence"
            )
        if value < CLEAN_VALIDATION_SEEDS_FROM:
            raise ValueError(
                f"{value} predates the clean validation range starting at "
                f"{CLEAN_VALIDATION_SEEDS_FROM}"
            )
    return values


# ---------------------------------------------------------------------------
# feature selection, fixed before fitting
# ---------------------------------------------------------------------------

#: The features this detector consumes, in bundle order.
PREFILTER_FEATURES: tuple = (
    "log10_relative_lag_magnitude_mean",
    "relative_lag_magnitude_decay_slope",
    "bandwidth_log_dispersion",
    "carrier_phase_coherence",
    "high_lag_qualified_fraction",
    "prefix_bandwidth_log_dispersion",
    "peak_normalized_mean_power",
)

#: Why each retained feature is here, one per mechanism family.
FEATURE_RATIONALE: dict = {
    "log10_relative_lag_magnitude_mean": (
        "magnitude family: how much coherent structure exists at all, averaged "
        "over BASE_LAGS. Noise AUROC 0.9004 standalone (HANDOFF 19.1). Retained "
        "over the lag-one element and the max because the mean is the "
        "lowest-variance reduction of the same statistic."
    ),
    "relative_lag_magnitude_decay_slope": (
        "lag-decay family: slope of that log ratio against log2(lag). A "
        "short-memory process collapses with lag, a narrowband emission does "
        "not. Noise AUROC 0.8872. Independent of the level above by "
        "construction (slope versus intercept)."
    ),
    "bandwidth_log_dispersion": (
        "multiscale-agreement family: population standard deviation of log10 "
        "of the per-lag bandwidth estimates. A coherent emission must agree "
        "across scales; a degenerate pose estimate disagrees. This is the "
        "quantity the frontend computes and discards, and is the direct "
        "expression of the HANDOFF 17 mechanism."
    ),
    "carrier_phase_coherence": (
        "phase family: |mean_k exp(i(angle(R[k]) - k angle(R[1])))|. A single "
        "carrier makes every term identically 1. Best standalone noise "
        "separator measured, AUROC 0.9154."
    ),
    "high_lag_qualified_fraction": (
        "estimator-admission family: fraction of higher base lags passing both "
        "of estimate_band's own admission tests. Noise AUROC 0.8579. It says "
        "how much of the estimator's evidence survived its own gate, which no "
        "other retained feature reports."
    ),
    "prefix_bandwidth_log_dispersion": (
        "temporal-stability family: dispersion of log10(bandwidth) re-estimated "
        "on nested prefixes of the same capture. Multiscale agreement "
        "(bandwidth_log_dispersion) and temporal agreement are different "
        "failures; noise fails both, a gated or bursty emission fails only the "
        "second."
    ),
    "peak_normalized_mean_power": (
        "amplitude-shape family: inverse squared crest factor, in (0, 1], "
        "exactly invariant to global scale and phase. It is a shape statistic, "
        "not an absolute level, so it adds no scale dependence; complex "
        "Gaussian noise has a characteristic crest factor that no retained "
        "correlation statistic sees."
    ),
}

#: Every excluded feature and the rule that excluded it.
DROPPED_FEATURES: dict = {
    "degenerate_full_band_fallback": (
        "measured dead: AUROC exactly 0.5000 (HANDOFF 19.1). The estimator's "
        "own degeneracy flag never fires on any development or novelty row. "
        "Also a binary indicator."
    ),
    "bandwidth_clipped_to_full": (
        "measured dead: AUROC 0.5009 (HANDOFF 19.1). Also a binary indicator."
    ),
    "zero_magnitude_prefix_fraction": (
        "measured dead: AUROC 0.5066 (HANDOFF 19.1). Also near-constant."
    ),
    "quotient_clip_fraction": (
        "measured dead: AUROC 0.5144 (HANDOFF 19.1)."
    ),
    "bandwidth_clipped_to_minimum": (
        "binary indicator of the same clipping mechanism whose continuous "
        "expression (bandwidth_log_dispersion) is retained; near-constant on "
        "the fit population, so standardising it divides by noise."
    ),
    "log2_selected_base_lag": (
        "discrete estimator-internal index over five possible base lags; a "
        "small-support ordinal code is the wrong input to a standardised "
        "linear model, and its continuous counterpart "
        "(high_lag_qualified_fraction) is retained."
    ),
    "log10_relative_lag1_magnitude": (
        "near-duplicate: the lag-one element of the same log-ratio statistic "
        "whose mean over BASE_LAGS is retained."
    ),
    "log10_relative_lag_magnitude_max": (
        "near-duplicate: the max reduction of the same log-ratio statistic "
        "whose mean is retained."
    ),
    "log10_lag1_sampling_margin": (
        "near-duplicate: log10_relative_lag1_magnitude minus a deterministic "
        "function of capture length. One model is fitted per length, so the "
        "length offset is a constant absorbed by the intercept and this reduces "
        "to the already-dropped lag-one element."
    ),
    "carrier_phase_coherence_sign_blind": (
        "near-duplicate: the same circular-mean statistic evaluated on doubled "
        "angles; retained sibling is carrier_phase_coherence."
    ),
    "bandwidth_log_range": (
        "near-duplicate: max minus min of the same per-lag log bandwidths "
        "whose population standard deviation is retained, and the less robust "
        "of the two reductions."
    ),
    "prefix_relative_lag1_log_dispersion": (
        "near-duplicate: prefix dispersion of the lag-one magnitude, one of "
        "three prefix-dispersion statistics of which "
        "prefix_bandwidth_log_dispersion is retained."
    ),
    "max_abs_unclipped_quotient": (
        "cap-saturating heavy-tailed tail statistic of the same per-lag "
        "quotient set summarised by the retained bandwidth_log_dispersion and "
        "high_lag_qualified_fraction. A column that saturates at QUOTIENT_CAP "
        "is the worst kind of input to a standardised linear model. This is the "
        "most defensible re-addition candidate if the operating point misses, "
        "and it would need a fresh design seed."
    ),
    "prefix_centre_dispersion": (
        "the CHIRP carrier, not the noise carrier: chirp AUROC 0.9612 against "
        "noise AUROC 0.7230 (HANDOFF 19.1). Chirp is already solved by the "
        "existing v3 path (AUROC 0.9716, threshold recall 0.72-1.00) and is "
        "deliberately left there. A noise prefilter that also chases chirp "
        "reintroduces the very trade-off section 19.2 measured."
    ),
}


def selection_rationale() -> dict:
    """Return the full, serialisable feature-selection table."""
    used = [name for name in PREFILTER_FEATURES]
    dropped = [name for name in pdg.FEATURE_NAMES if name not in set(used)]
    return {
        "source_feature_count": pdg.FEATURE_COUNT,
        "used_feature_count": len(used),
        "used": [
            {"feature": name, "reason": FEATURE_RATIONALE[name]} for name in used
        ],
        "dropped": [
            {"feature": name, "reason": DROPPED_FEATURES[name]} for name in dropped
        ],
        "rules": [
            "measured dead in HANDOFF 19.1",
            "continuous columns only: no binary indicators, no small-support "
            "discrete codes",
            "one member per mechanism family: near-duplicate reductions of the "
            "same statistic are dropped",
            "chirp-carrying features are left to the existing v3 path",
        ],
    }


def _validate_feature_names(names: Sequence[str]) -> tuple:
    values = tuple(str(name) for name in names)
    if not values:
        raise ValueError("at least one feature is required")
    if len(set(values)) != len(values):
        raise ValueError("feature names must be distinct")
    unknown = [name for name in values if name not in pdg.FEATURE_NAMES]
    if unknown:
        raise ValueError(f"unknown pose-degeneracy features: {unknown}")
    return values


def feature_columns(names: Sequence[str] = PREFILTER_FEATURES) -> np.ndarray:
    """Column indices of ``names`` into the frozen pose-degeneracy vector."""
    values = _validate_feature_names(names)
    lookup = {name: index for index, name in enumerate(pdg.FEATURE_NAMES)}
    return np.asarray([lookup[name] for name in values], dtype=np.int64)


def select_features(
    matrix: Any, names: Sequence[str] = PREFILTER_FEATURES
) -> np.ndarray:
    """Slice a full ``(rows, 21)`` pose-degeneracy matrix down to ``names``."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != pdg.FEATURE_COUNT:
        raise ValueError(
            f"expected a (rows, {pdg.FEATURE_COUNT}) pose-degeneracy matrix, "
            f"got shape {values.shape}"
        )
    return np.ascontiguousarray(values[:, feature_columns(names)])


# ---------------------------------------------------------------------------
# fit hyperparameters, all declared
# ---------------------------------------------------------------------------

#: Ridge penalty on the standardised coefficients (the intercept is never
#: penalised).  Known rows and white noise are close to linearly separable in
#: these features and unregularised logistic regression diverges on separable
#: data, so this is a stability requirement, not a tuning knob.  It was NOT
#: selected against any validation seed.  It changes the fitted direction
#: slightly, which is why it is reported in every bundle.
L2_PENALTY = 1e-2

#: Damped Newton/IRLS controls.  Fixed, so the fit is a pure function of inputs.
MAX_NEWTON_ITERATIONS = 100
NEWTON_TOLERANCE = 1e-12
MAX_BACKTRACK_STEPS = 40

#: A standardiser divisor below this is treated as a degenerate column and
#: refused rather than silently amplified.
SCALE_FLOOR = 1e-12

#: Known-false-positive budget for the gate, i.e. the fraction of ENROLLMENT
#: known rows the prefilter is allowed to gate away.
#:
#: Argued, not inherited.  The house ceiling on known false-unknown rate is 0.10
#: (``fit_v3_openset.KNOWN_FUR_CEILING``) and the existing v3 rejector already
#: spends 0.0466 of it (HANDOFF 17).  A staged design spends both budgets on the
#: same rows, so the prefilter must fit inside roughly 0.05 to keep the stack
#: under the ceiling.  0.02 leaves headroom for the downstream stage and for the
#: gap between enrollment and any later population.  It is a required, explicit
#: argument everywhere; this constant is only the documented default.
DEFAULT_KNOWN_FALSE_POSITIVE_BUDGET = 0.02

#: Operating-point policies.
BUDGET_POLICY = "enrollment_known_false_positive_budget"
FIXED_SCORE_POLICY = "fixed_score"
OPERATING_POINT_POLICIES = (BUDGET_POLICY, FIXED_SCORE_POLICY)


def gate_floors() -> dict:
    """Return the noise gates, imported from ``fit_v3_openset``, never re-typed.

    A gate that is copied can be lowered by editing the copy.  This reads the
    live values so it cannot drift, and the import is lazy only because
    ``fit_v3_openset`` pulls in torch.
    """
    import fit_v3_openset as base  # noqa: PLC0415  (lazy: torch)

    return {
        "noise_auroc": float(base.GATE_FLOORS["noise_auroc"]),
        "noise_threshold_recall": float(base.GATE_FLOORS["noise_threshold_recall"]),
        "known_false_unknown_ceiling": float(base.KNOWN_FUR_CEILING),
    }


# ---------------------------------------------------------------------------
# numerics
# ---------------------------------------------------------------------------


def _sigmoid(scores: np.ndarray) -> np.ndarray:
    """Overflow-free logistic, elementwise."""
    values = np.asarray(scores, dtype=np.float64)
    out = np.empty_like(values)
    positive = values >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    out[~positive] = exponential / (1.0 + exponential)
    return out


def _log1p_exp(values: np.ndarray) -> np.ndarray:
    """``log(1 + exp(v))`` without overflow.

    Branches are evaluated on disjoint masks rather than with ``np.where``:
    ``np.where`` evaluates both arms, so the positive arm would overflow on a
    large score and raise under ``PYTHONWARNINGS=error``.
    """
    values = np.asarray(values, dtype=np.float64)
    out = np.empty_like(values)
    positive = values > 0.0
    out[positive] = values[positive] + np.log1p(np.exp(-values[positive]))
    out[~positive] = np.log1p(np.exp(values[~positive]))
    return out


def _dot(left: Any, right: Any) -> np.ndarray:
    """Matrix product with the IEEE flags neutralised and finiteness asserted.

    This is not defensive noise.  On this platform numpy 2.0 links Apple's
    Accelerate BLAS, and Accelerate raises **spurious** IEEE flags on ordinary
    float64 products: ``A.T @ B`` on two finite random ``(120, 8)`` arrays
    reports ``divide by zero encountered in matmul``.  The repository runs under
    ``PYTHONWARNINGS=error``, so that spurious flag would abort a numerically
    perfect fit.

    Simply suppressing the flag would also hide a genuine overflow, so the
    product is checked for finiteness instead: a real numerical failure still
    stops the fit, and a BLAS artifact does not.
    """
    with np.errstate(all="ignore"):
        out = np.asarray(left) @ np.asarray(right)
    if not np.isfinite(out).all():
        raise FloatingPointError("non-finite matrix product")
    return np.asarray(out)


def _solve(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Linear solve with the same treatment, falling back to least squares."""
    with np.errstate(all="ignore"):
        try:
            out = np.linalg.solve(matrix, vector)
        except np.linalg.LinAlgError:
            out = np.linalg.lstsq(matrix, vector, rcond=None)[0]
    if not np.isfinite(out).all():
        raise FloatingPointError("non-finite Newton step")
    return np.asarray(out)


def _check_matrix(matrix: Any, name: str, width: int) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != width:
        raise ValueError(f"{name} must be a (rows, {width}) float matrix")
    if values.shape[0] == 0:
        raise ValueError(f"{name} must have at least one row")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains a non-finite value")
    return np.ascontiguousarray(values)


def _fit_logistic(
    design: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    l2: float,
) -> tuple:
    """Damped Newton/IRLS for weighted ridge logistic regression.

    ``design`` carries the intercept as its final column and that column is not
    penalised.  Returns ``(coefficients, diagnostics)``.  Deterministic: the
    starting point is exactly zero, the step is a linear solve, and the
    backtracking schedule is a fixed halving.
    """
    n_rows, n_columns = design.shape
    penalty = np.ones(n_columns, dtype=np.float64) * float(l2)
    penalty[-1] = 0.0

    def objective(beta: np.ndarray) -> float:
        scores = _dot(design, beta)
        loss = float(np.sum(weights * (_log1p_exp(scores) - labels * scores)))
        return loss + float(np.sum(penalty * beta * beta))

    beta = np.zeros(n_columns, dtype=np.float64)
    value = objective(beta)
    iterations = 0
    converged = False
    for iterations in range(1, MAX_NEWTON_ITERATIONS + 1):
        probabilities = _sigmoid(_dot(design, beta))
        gradient = _dot(design.T, weights * (probabilities - labels))
        gradient = gradient + 2.0 * penalty * beta
        variance = weights * probabilities * (1.0 - probabilities)
        hessian = _dot(design.T, design * variance[:, None])
        hessian = hessian + np.diag(2.0 * penalty)
        # Newton on a strictly convex ridge objective; the diagonal is positive
        # because the penalty is, except on the intercept, whose curvature comes
        # from the data.  A tiny ridge keeps the solve well posed if every
        # probability saturates.
        hessian = hessian + np.eye(n_columns) * 1e-12
        step = _solve(hessian, gradient)
        scale = 1.0
        improved = False
        for _ in range(MAX_BACKTRACK_STEPS):
            candidate = beta - scale * step
            candidate_value = objective(candidate)
            if candidate_value <= value:
                improved = True
                break
            scale *= 0.5
        if not improved:
            converged = True
            break
        movement = float(np.max(np.abs(candidate - beta)))
        beta = candidate
        value = candidate_value
        if movement <= NEWTON_TOLERANCE:
            converged = True
            break
    diagnostics = {
        "iterations": int(iterations),
        "converged": bool(converged),
        "objective": float(value),
        "l2_penalty": float(l2),
        "max_iterations": int(MAX_NEWTON_ITERATIONS),
        "tolerance": float(NEWTON_TOLERANCE),
    }
    return beta, diagnostics


def mann_whitney_auroc(positive: Any, negative: Any) -> float:
    """AUROC that ``positive`` (noise) scores exceed ``negative`` (known).

    Delegates to the shipped ``train.auroc`` so this module cannot disagree with
    sections 16-19 about what an AUROC is.  Lazy import only because ``train``
    pulls in torch.
    """
    from train import auroc  # noqa: PLC0415  (lazy: torch)

    return float(auroc(positive, negative))


# ---------------------------------------------------------------------------
# the detector
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NoisePrefilter:
    """A fitted, serialisable, deterministic noise/not-noise gate.

    ``score`` is the linear log-odds, ``predict_proba`` the logistic of it, and
    ``decide`` the hard gate.  The threshold is stored in **score** space, not
    probability space: probabilities saturate to exactly 1.0 in float64 and a
    quantile of a saturated column is not a usable operating point, while the
    score is unbounded and the two orderings are identical because the logistic
    is strictly increasing.
    """

    feature_names: tuple
    capture_length: int
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    threshold_score: float
    fit_provenance: dict
    threshold_provenance: dict

    # -- shape -------------------------------------------------------------

    def __post_init__(self) -> None:
        names = _validate_feature_names(self.feature_names)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "capture_length", int(self.capture_length))
        if self.capture_length <= 0:
            raise ValueError("capture_length must be positive")
        for field in ("mean", "scale", "coefficients"):
            values = np.ascontiguousarray(
                np.asarray(getattr(self, field), dtype=np.float64)
            )
            if values.shape != (len(names),):
                raise ValueError(f"{field} must have one entry per feature")
            if not np.isfinite(values).all():
                raise ValueError(f"{field} contains a non-finite value")
            object.__setattr__(self, field, values)
        if not np.all(self.scale > SCALE_FLOOR):
            raise ValueError("standardiser scale has a degenerate entry")
        object.__setattr__(self, "intercept", float(self.intercept))
        if not math.isfinite(self.intercept):
            raise ValueError("intercept must be finite")
        threshold = float(self.threshold_score)
        if not (math.isfinite(threshold) or math.isnan(threshold)):
            raise ValueError("threshold_score must be finite or NaN")
        object.__setattr__(self, "threshold_score", threshold)

    @property
    def has_threshold(self) -> bool:
        """Whether an operating point has been chosen on enrollment."""
        return not math.isnan(self.threshold_score)

    @property
    def threshold_probability(self) -> float:
        """The operating point in probability space, for reading only."""
        if not self.has_threshold:
            return float("nan")
        return float(_sigmoid(np.asarray([self.threshold_score]))[0])

    # -- scoring -----------------------------------------------------------

    def standardize(self, features: Any) -> np.ndarray:
        """Standardise a selected or full pose-degeneracy matrix."""
        values = _as_selected(features, self.feature_names, "features")
        return (values - self.mean) / self.scale

    def score(self, features: Any, *, capture_length: int | None = None) -> np.ndarray:
        """Linear log-odds of noise, higher = more noise-like."""
        self._check_length(capture_length)
        return _dot(self.standardize(features), self.coefficients) + self.intercept

    def predict_proba(self, features: Any, *, capture_length: int | None = None):
        """``P(noise)`` per row, in ``[0, 1]``."""
        return _sigmoid(self.score(features, capture_length=capture_length))

    def decide(self, features: Any, *, capture_length: int | None = None):
        """Hard gate: ``True`` means NOISE, abstain, do not classify."""
        if not self.has_threshold:
            raise ValueError(
                "no operating point: fit_threshold() must be run on the "
                "enrollment population before the gate can be used"
            )
        scores = self.score(features, capture_length=capture_length)
        return scores >= self.threshold_score

    def gate(self, features: Any, *, capture_length: int | None = None) -> dict:
        """Score, probability, decision and realised compute saving in one call."""
        scores = self.score(features, capture_length=capture_length)
        gated = scores >= self.threshold_score if self.has_threshold else None
        return {
            "score": scores,
            "probability": _sigmoid(scores),
            "gated_as_noise": gated,
            "compute_saving": compute_saving(gated) if gated is not None else None,
        }

    def _check_length(self, capture_length: int | None) -> None:
        if capture_length is None:
            return
        if int(capture_length) != self.capture_length:
            raise ValueError(
                f"this prefilter was fitted at capture length "
                f"{self.capture_length} and the pose-degeneracy features are "
                f"length dependent; refusing to score length {int(capture_length)}"
            )

    # -- contract ----------------------------------------------------------

    def metadata(self) -> dict:
        """The full, serialisable contract.  Contains no timestamp by design."""
        return {
            "schema": BUNDLE_SCHEMA,
            "version": NOISE_PREFILTER_VERSION,
            "model": "ridge logistic regression on standardised features",
            "capture_length": int(self.capture_length),
            "feature_names": list(self.feature_names),
            "feature_selection": selection_rationale(),
            "extractor": pdg.pose_degeneracy_metadata(),
            "operating_point": dict(self.threshold_provenance),
            "fit": dict(self.fit_provenance),
            "additive_only": False,
            "changes_closed_label": True,
            "gates_before_classification": True,
            "architecture_contract_change": ARCHITECTURE_CONTRACT_CHANGE,
            "decides": "noise vs not-noise only; chirp stays on the v3 path",
            "inference_uses_frequency_transform": False,
            "fitting_noise_generator_uses_frequency_transform": True,
            "uses_absolute_scale": False,
            "invariant_to_global_phase": True,
            "invariant_to_global_scale": True,
            "fit_population": TRAIN_SPLIT,
            "threshold_population": ENROLLMENT_SPLIT,
            "selection_rows_used": 0,
            "consumed_test_rows_used": 0,
            "sealed_release_paths_read": 0,
            "release_seed_not_spent": RELEASE_SEED_NEVER_SPENT_HERE,
        }

    # -- serialisation -----------------------------------------------------

    def save(self, directory: Any) -> Path:
        """Write the plain ``.npy`` / ``.json`` bundle.  No pickle, no npz."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        _write_array(path / "mean.npy", self.mean)
        _write_array(path / "scale.npy", self.scale)
        _write_array(path / "coefficients.npy", self.coefficients)
        _write_array(path / "intercept.npy", np.asarray([self.intercept], dtype=np.float64))
        _write_array(
            path / "threshold_score.npy",
            np.asarray([self.threshold_score], dtype=np.float64),
        )
        meta = self.metadata()
        meta["threshold_score_is_set"] = bool(self.has_threshold)
        meta["threshold_score_repr"] = repr(float(self.threshold_score))
        meta["threshold_probability_repr"] = repr(float(self.threshold_probability))
        _write_json(path / "meta.json", meta)
        return path

    @classmethod
    def load(cls, directory: Any) -> "NoisePrefilter":
        path = Path(directory)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        if meta.get("schema") != BUNDLE_SCHEMA:
            raise ValueError(f"not a {BUNDLE_SCHEMA} bundle: {path}")
        threshold = float(np.load(path / "threshold_score.npy")[0])
        if repr(threshold) != meta["threshold_score_repr"]:
            raise ValueError(
                "bundle integrity: threshold in meta.json disagrees with "
                "threshold_score.npy"
            )
        return cls(
            feature_names=tuple(meta["feature_names"]),
            capture_length=int(meta["capture_length"]),
            mean=np.load(path / "mean.npy"),
            scale=np.load(path / "scale.npy"),
            coefficients=np.load(path / "coefficients.npy"),
            intercept=float(np.load(path / "intercept.npy")[0]),
            threshold_score=threshold,
            fit_provenance=dict(meta["fit"]),
            threshold_provenance=dict(meta["operating_point"]),
        )


#: The exact contents of a bundle directory.
BUNDLE_FILES = (
    "coefficients.npy",
    "intercept.npy",
    "mean.npy",
    "meta.json",
    "scale.npy",
    "threshold_score.npy",
)


def _write_array(path: Path, values: np.ndarray) -> None:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    with Path(path).open("wb") as handle:
        np.save(handle, array, allow_pickle=False)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(
            _jsonable(payload), handle, indent=2, sort_keys=True, allow_nan=False
        )
        handle.write("\n")


def bundle_sha256(directory: Any) -> str:
    """Deterministic SHA over exactly :data:`BUNDLE_FILES`.

    Names and bytes are folded in sorted order with explicit lengths, so a
    rename cannot collide with a content change, and an unexpected extra file in
    the directory is refused rather than silently ignored.
    """
    path = Path(directory)
    present = sorted(entry.name for entry in path.iterdir() if entry.is_file())
    if tuple(present) != BUNDLE_FILES:
        raise ValueError(
            f"bundle must contain exactly {list(BUNDLE_FILES)}, found {present}"
        )
    digest = hashlib.sha256()
    for name in BUNDLE_FILES:
        payload = (path / name).read_bytes()
        digest.update(f"{name}:{len(payload)}:".encode("utf-8"))
        digest.update(payload)
    return digest.hexdigest()


def save_prefilter_set(directory: Any, models: Mapping[int, NoisePrefilter]) -> Path:
    """Write one bundle per capture length under ``directory/N{length}``."""
    if not models:
        raise ValueError("no models to save")
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    for length in sorted(int(key) for key in models):
        model = models[length]
        if int(model.capture_length) != int(length):
            raise ValueError(
                f"model filed under length {length} was fitted at "
                f"{model.capture_length}"
            )
        model.save(path / f"N{int(length)}")
    return path


def load_prefilter_set(directory: Any) -> dict:
    """Load every ``N{length}`` bundle under ``directory``."""
    path = Path(directory)
    models = {}
    for entry in sorted(path.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("N"):
            continue
        model = NoisePrefilter.load(entry)
        models[int(entry.name[1:])] = model
    if not models:
        raise ValueError(f"no prefilter bundles under {path}")
    return models


def prefilter_set_sha256(directory: Any) -> str:
    """Deterministic SHA over a whole length set."""
    path = Path(directory)
    digest = hashlib.sha256()
    names = sorted(
        entry.name
        for entry in path.iterdir()
        if entry.is_dir() and entry.name.startswith("N")
    )
    if not names:
        raise ValueError(f"no prefilter bundles under {path}")
    for name in names:
        digest.update(f"{name}:".encode("utf-8"))
        digest.update(bundle_sha256(path / name).encode("utf-8"))
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# fitting: TRAINING only
# ---------------------------------------------------------------------------


def fit_prefilter(
    known_training_features: Any,
    fitting_noise_features: Any,
    *,
    capture_length: int,
    known_population: str,
    fitting_seed: int,
    feature_names: Sequence[str] = PREFILTER_FEATURES,
    l2: float = L2_PENALTY,
    fitting_noise_provenance: Mapping[str, Any] | None = None,
) -> NoisePrefilter:
    """Fit the coefficients on TRAINING known rows and fitting-seed noise.

    ``known_population`` must be :data:`TRAIN_SPLIT`; anything else is refused,
    so a caller cannot pass enrollment or selection rows by accident.
    ``fitting_seed`` is validated against the ledger even though the caller has
    already drawn the noise, because the seed is what the emitted provenance
    claims and an unchecked claim is worse than none.

    Both inputs are already-selected ``(rows, len(feature_names))`` matrices, or
    full ``(rows, 21)`` pose-degeneracy matrices, which are sliced here.

    Class weights are balanced to 0.5 total mass per class and the standardiser
    is the **class-balanced weighted** mean and standard deviation of the pooled
    fit rows.  Both choices exist for the same reason: neither the fitted
    direction nor the feature scaling may depend on how many synthetic noise
    rows were drawn.  Standardising on the known rows alone was tried and is
    wrong here: the known class is very tight in several of these features, so
    noise lands hundreds of standard deviations out, the ridge term then
    dominates a badly conditioned problem, and the fit collapses onto a large
    intercept with near-zero coefficients.
    """
    if known_population not in FITTABLE_POPULATIONS:
        raise ValueError(
            f"coefficients may only be fitted on {list(FITTABLE_POPULATIONS)}, "
            f"got {known_population!r}"
        )
    seed = validate_fitting_seed(fitting_seed)
    names = _validate_feature_names(feature_names)
    length = int(capture_length)
    if length <= 0:
        raise ValueError("capture_length must be positive")
    penalty = float(l2)
    if not (math.isfinite(penalty) and penalty > 0.0):
        raise ValueError(
            "l2 must be finite and strictly positive; known rows and white "
            "noise are close to separable in these features and an "
            "unregularised fit diverges"
        )

    known = _as_selected(known_training_features, names, "known training features")
    noise = _as_selected(fitting_noise_features, names, "fitting noise features")
    if known.shape[0] < 2 or noise.shape[0] < 2:
        raise ValueError("each class needs at least two rows")
    if known.shape[0] < len(names) + 1:
        raise ValueError(
            f"need at least {len(names) + 1} known training rows for "
            f"{len(names)} features"
        )

    pooled = np.vstack([known, noise])
    labels = np.concatenate(
        [np.zeros(known.shape[0]), np.ones(noise.shape[0])]
    ).astype(np.float64)
    weights = np.concatenate(
        [
            np.full(known.shape[0], 0.5 / known.shape[0]),
            np.full(noise.shape[0], 0.5 / noise.shape[0]),
        ]
    ).astype(np.float64)

    mean = _dot(weights, pooled)
    scale = np.sqrt(_dot(weights, (pooled - mean) ** 2))
    degenerate = [
        name for name, value in zip(names, scale) if not value > SCALE_FLOOR
    ]
    if degenerate:
        raise ValueError(
            f"degenerate fit columns, zero variance, cannot be standardised: "
            f"{degenerate}"
        )

    standardized = (pooled - mean) / scale
    design = np.hstack(
        [standardized, np.ones((standardized.shape[0], 1), dtype=np.float64)]
    )

    beta, diagnostics = _fit_logistic(design, labels, weights, l2=penalty)

    provenance = {
        "known_population": TRAIN_SPLIT,
        "known_rows": int(known.shape[0]),
        "fitting_noise_rows": int(noise.shape[0]),
        "fitting_seed": int(seed),
        "fitting_seed_namespace": list(FITTING_SEED_NAMESPACE),
        "fitting_seed_is_not_a_validation_seed": True,
        "spent_novelty_seeds_refused": sorted(SPENT_NOVELTY_SEEDS),
        "clean_validation_seeds_from": CLEAN_VALIDATION_SEEDS_FROM,
        "standardizer_population": (
            "training known rows plus fitting-seed noise, class-balanced "
            "weighted mean and standard deviation"
        ),
        "class_weighting": "balanced, 0.5 total mass per class",
        "solver": "damped Newton / IRLS, zero start, fixed halving backtrack",
        "enrollment_rows_used_in_fit": 0,
        "selection_rows_used_in_fit": 0,
        "consumed_test_rows_used_in_fit": 0,
        **diagnostics,
    }
    if fitting_noise_provenance is not None:
        provenance["fitting_noise"] = dict(fitting_noise_provenance)

    return NoisePrefilter(
        feature_names=names,
        capture_length=length,
        mean=mean,
        scale=scale,
        coefficients=beta[:-1].copy(),
        intercept=float(beta[-1]),
        threshold_score=float("nan"),
        fit_provenance=provenance,
        threshold_provenance={
            "policy": None,
            "chosen": False,
            "note": "no operating point yet; fit_threshold() on enrollment",
        },
    )


def _as_selected(matrix: Any, names: Sequence[str], label: str) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"{label} must be a 2-D matrix")
    if values.shape[1] == pdg.FEATURE_COUNT and len(names) != pdg.FEATURE_COUNT:
        values = select_features(values, names)
    return _check_matrix(values, label, len(names))


# ---------------------------------------------------------------------------
# operating point: ENROLLMENT only
# ---------------------------------------------------------------------------


def budget_threshold(known_scores: Any, budget: float) -> tuple:
    """Smallest score threshold whose known false-positive rate is <= ``budget``.

    With ``m`` known rows, at most ``k = floor(budget * m)`` may be gated.  The
    threshold is one ULP above the ``k``-th largest known score, so the realised
    rate is ``count(score > d[k]) / m <= k / m <= budget`` even when scores tie.
    Choosing the smallest such threshold maximises noise recall at the budget,
    which is the whole point: an operating point that is conservative by accident
    throws away recall it was allowed to have.
    """
    scores = np.asarray(known_scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        raise ValueError("threshold needs at least one known score")
    if not np.isfinite(scores).all():
        raise ValueError("known scores must be finite")
    value = float(budget)
    if not (math.isfinite(value) and 0.0 <= value < 1.0):
        raise ValueError("budget must be a finite fraction in [0, 1)")
    descending = np.sort(scores)[::-1]
    allowance = int(math.floor(value * scores.size))
    allowance = min(allowance, scores.size - 1)
    threshold = float(np.nextafter(descending[allowance], np.inf))
    realised = float(np.mean(scores >= threshold))
    return threshold, {
        "rows": int(scores.size),
        "allowance": allowance,
        "realised_known_false_positive_rate": realised,
    }


def fit_threshold(
    model: NoisePrefilter,
    enrollment_known_features: Any,
    *,
    population: str,
    budget: float = DEFAULT_KNOWN_FALSE_POSITIVE_BUDGET,
    policy: str = BUDGET_POLICY,
    fixed_score: float | None = None,
) -> NoisePrefilter:
    """Choose the operating point on ENROLLMENT known rows and nothing else.

    This is the step the frozen policy got wrong.  q95 there is a quantile of the
    *fused* known distribution, which the branch LOF dominates, so the threshold
    lands where the branch LOF says the known rows end, not where this detector's
    known rows end.  Here the quantile is taken over this detector's own score
    distribution, at a stated budget.
    """
    if population not in THRESHOLDABLE_POPULATIONS:
        raise ValueError(
            f"the operating point may only be chosen on "
            f"{list(THRESHOLDABLE_POPULATIONS)}, got {population!r}"
        )
    if policy not in OPERATING_POINT_POLICIES:
        raise ValueError(f"policy must be one of {list(OPERATING_POINT_POLICIES)}")

    scores = model.score(
        _as_selected(
            enrollment_known_features, model.feature_names, "enrollment features"
        )
    )
    if policy == FIXED_SCORE_POLICY:
        if fixed_score is None or not math.isfinite(float(fixed_score)):
            raise ValueError("fixed_score policy requires a finite fixed_score")
        threshold = float(fixed_score)
        detail = {
            "rows": int(scores.size),
            "realised_known_false_positive_rate": float(
                np.mean(scores >= threshold)
            ),
        }
        budget_value = None
    else:
        threshold, detail = budget_threshold(scores, budget)
        budget_value = float(budget)

    provenance = {
        "policy": policy,
        "chosen": True,
        "population": ENROLLMENT_SPLIT,
        "known_false_positive_budget": budget_value,
        "threshold_score": float(threshold),
        "threshold_probability": float(
            _sigmoid(np.asarray([threshold], dtype=np.float64))[0]
        ),
        "threshold_space": "linear score (log-odds), not probability, because "
        "probabilities saturate to 1.0 in float64",
        "gate_rule": "gated as NOISE when score >= threshold_score",
        "training_rows_used_for_threshold": 0,
        "selection_rows_used_for_threshold": 0,
        "novelty_rows_used_for_threshold": 0,
        "inherited_from_another_model": False,
        **detail,
    }
    return replace(model, threshold_score=float(threshold), threshold_provenance=provenance)


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def compute_saving(gated: Any) -> dict:
    """Measured downstream saving on whatever stream was actually gated."""
    values = np.asarray(gated, dtype=bool).reshape(-1)
    if values.size == 0:
        raise ValueError("compute_saving needs at least one decision")
    gated_rows = int(np.count_nonzero(values))
    return {
        "rows": int(values.size),
        "gated_rows": gated_rows,
        "gated_fraction": float(gated_rows / values.size),
        "downstream_rows_evaluated": int(values.size - gated_rows),
        "measured": True,
        "counts": "rows that never reach the encoder, fusion, prototype "
        "distance or branch LOF",
    }


def evaluate(
    model: NoisePrefilter,
    known_features: Any,
    noise_features: Any,
    *,
    known_population: str,
) -> dict:
    """Report AUROC and, the thing that actually matters, THRESHOLD RECALL.

    AUROC is included because sections 17 and 19 are quoted in it and the
    comparison must stay possible.  It is not the gate.  The gate is threshold
    recall at a known false-positive rate inside the declared budget, and both
    are reported next to the imported floors.
    """
    known = _as_selected(known_features, model.feature_names, "known features")
    noise = _as_selected(noise_features, model.feature_names, "noise features")
    known_scores = model.score(known)
    noise_scores = model.score(noise)
    floors = gate_floors()

    report: dict = {
        "known_population": str(known_population),
        "known_rows": int(known.shape[0]),
        "noise_rows": int(noise.shape[0]),
        "capture_length": int(model.capture_length),
        "noise_auroc": mann_whitney_auroc(noise_scores, known_scores),
        "noise_auroc_floor": floors["noise_auroc"],
        "noise_threshold_recall_floor": floors["noise_threshold_recall"],
        "known_false_unknown_ceiling": floors["known_false_unknown_ceiling"],
        "auroc_is_not_the_gate": (
            "threshold recall at the declared operating point is the gate"
        ),
    }
    if model.has_threshold:
        gated_noise = noise_scores >= model.threshold_score
        gated_known = known_scores >= model.threshold_score
        report.update(
            {
                "threshold_score": float(model.threshold_score),
                "threshold_probability": float(model.threshold_probability),
                "noise_threshold_recall": float(np.mean(gated_noise)),
                "known_false_positive_rate": float(np.mean(gated_known)),
                "noise_compute_saving": compute_saving(gated_noise),
                "known_compute_saving": compute_saving(gated_known),
            }
        )
        report["passes_noise_auroc"] = bool(
            report["noise_auroc"] >= floors["noise_auroc"]
        )
        report["passes_noise_threshold_recall"] = bool(
            report["noise_threshold_recall"] >= floors["noise_threshold_recall"]
        )
        budget = model.threshold_provenance.get("known_false_positive_budget")
        report["known_false_positive_budget"] = budget
        report["within_known_false_positive_budget"] = (
            None
            if budget is None
            else bool(report["known_false_positive_rate"] <= float(budget))
        )
    else:
        report["noise_threshold_recall"] = None
        report["note"] = "no operating point chosen; recall is undefined"
    return report


# ---------------------------------------------------------------------------
# populations and fitting noise
# ---------------------------------------------------------------------------


def synthesize_fitting_noise(
    *,
    fitting_seed: int,
    n_rows: int,
    lengths: Sequence[int],
    min_bandwidth_mode: str = "frontend",
    progress: bool = False,
) -> tuple:
    """Draw fit-only synthetic noise and extract pose-degeneracy features.

    Same generator, draw protocol and resolution floor as
    ``measure_pose_degeneracy.novelty_features`` -- drawn once at the longest
    length with the shorter observations taken as exact raw prefixes -- so the
    fitting distribution differs from the validation distribution in the **seed
    and nothing else**.  Only the ``noise`` family is drawn; chirp is not this
    detector's job.
    """
    import measure_pose_degeneracy as measure  # noqa: PLC0415  (lazy: torch)
    from openset_eval import NOVELTY  # noqa: PLC0415  (lazy: torch)

    seed = validate_fitting_seed(fitting_seed)
    prefix_lengths = tuple(int(value) for value in lengths)
    if not prefix_lengths or any(value <= 0 for value in prefix_lengths):
        raise ValueError("lengths must be positive")
    if tuple(sorted(set(prefix_lengths))) != prefix_lengths:
        raise ValueError("lengths must be unique and increasing")
    rows = int(n_rows)
    if rows <= 0:
        raise ValueError("n_rows must be positive")

    generator = NOVELTY["noise"]
    longest = max(prefix_lengths)
    rng = np.random.default_rng(seed)
    collected: dict = {length: [] for length in prefix_lengths}
    digest = hashlib.sha256()
    for row in range(rows):
        raw = generator(longest, rng)
        if len(raw) != longest:
            raise RuntimeError("noise generator returned the wrong length")
        digest.update(
            np.ascontiguousarray(np.asarray(raw, dtype=np.complex128)).view(np.uint8)
        )
        for length in prefix_lengths:
            collected[length].append(
                pdg.pose_degeneracy_features(
                    raw[:length],
                    min_bandwidth=measure._min_bandwidth(length, min_bandwidth_mode),
                )
            )
        if progress and (row + 1) % 100 == 0:
            print(f"  [fitting noise] {row + 1}/{rows}", flush=True)

    matrices = {
        length: np.stack(values).astype(np.float64)
        for length, values in collected.items()
    }
    provenance = {
        "generator": "openset_eval.NOVELTY['noise']",
        "family": "noise",
        "chirp_drawn": False,
        "seed": int(seed),
        "seed_role": "fit only",
        "n_rows": rows,
        "generated_once_at": int(longest),
        "shorter_lengths_are_exact_raw_prefixes": True,
        "min_bandwidth_mode": str(min_bandwidth_mode),
        "longest_capture_sha256": digest.hexdigest(),
    }
    return matrices, provenance


def load_known_features(
    split: str,
    *,
    lengths: Sequence[int],
    min_bandwidth_mode: str = "frontend",
    limit: int | None = None,
    progress: bool = False,
) -> tuple:
    """Pose-degeneracy features for development known rows, by length.

    Delegates to ``measure_pose_degeneracy.known_features`` so the extraction
    path, including the all-zero-prefix exclusion rule, is shared rather than
    re-implemented.  Only ``train`` and ``enroll`` are reachable; the immutable
    selection half and the consumed test half are not.
    """
    if split not in (TRAIN_SPLIT, ENROLLMENT_SPLIT):
        raise ValueError(
            f"only {TRAIN_SPLIT!r} and {ENROLLMENT_SPLIT!r} are reachable, got "
            f"{split!r}"
        )
    import measure_pose_degeneracy as measure  # noqa: PLC0415  (lazy: torch)

    return measure.known_features(
        split,
        lengths=lengths,
        min_bandwidth_mode=min_bandwidth_mode,
        limit=limit,
        progress=progress,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_LENGTHS = (4096, 8192, 16384)
DEFAULT_FITTING_NOISE_ROWS = 600
DEFAULT_OUTPUT_DIR = (
    Path(_V2_DIR) / "artifacts" / "invariant_patch" / "v3_scale" / "noise_prefilter"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit the v3 noise prefilter.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--fitting-seed",
        type=int,
        required=True,
        help=(
            "fit-only noise seed; must lie in "
            f"{FITTING_SEED_NAMESPACE[0]}-{FITTING_SEED_NAMESPACE[1]}. "
            f"Proposed: {PROPOSED_FITTING_SEED}. Ledger seeds are refused."
        ),
    )
    parser.add_argument(
        "--known-false-positive-budget",
        type=float,
        required=True,
        help=(
            "fraction of ENROLLMENT known rows the gate may claim; documented "
            f"default {DEFAULT_KNOWN_FALSE_POSITIVE_BUDGET}"
        ),
    )
    parser.add_argument("--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS))
    parser.add_argument("--fitting-noise-rows", type=int, default=DEFAULT_FITTING_NOISE_ROWS)
    parser.add_argument("--l2", type=float, default=L2_PENALTY)
    parser.add_argument(
        "--features", nargs="+", default=list(PREFILTER_FEATURES)
    )
    parser.add_argument(
        "--min-bandwidth-mode", choices=("frontend", "fixed"), default="frontend"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    seed = validate_fitting_seed(args.fitting_seed)

    import fit_v3_openset as base  # noqa: PLC0415  (lazy: torch)

    output_dir = base.reject_sealed_path(Path(args.output_dir), "prefilter output")
    output_dir.mkdir(parents=True, exist_ok=True)

    lengths = tuple(int(value) for value in args.lengths)
    names = _validate_feature_names(args.features)

    train_known, train_audit = load_known_features(
        TRAIN_SPLIT,
        lengths=lengths,
        min_bandwidth_mode=args.min_bandwidth_mode,
        limit=args.limit,
        progress=args.progress,
    )
    enroll_known, enroll_audit = load_known_features(
        ENROLLMENT_SPLIT,
        lengths=lengths,
        min_bandwidth_mode=args.min_bandwidth_mode,
        limit=args.limit,
        progress=args.progress,
    )
    noise, noise_provenance = synthesize_fitting_noise(
        fitting_seed=seed,
        n_rows=int(args.fitting_noise_rows),
        lengths=lengths,
        min_bandwidth_mode=args.min_bandwidth_mode,
        progress=args.progress,
    )

    models: dict = {}
    fit_reports: dict = {}
    for length in lengths:
        model = fit_prefilter(
            train_known[length],
            noise[length],
            capture_length=length,
            known_population=TRAIN_SPLIT,
            fitting_seed=seed,
            feature_names=names,
            l2=float(args.l2),
            fitting_noise_provenance=noise_provenance,
        )
        model = fit_threshold(
            model,
            enroll_known[length],
            population=ENROLLMENT_SPLIT,
            budget=float(args.known_false_positive_budget),
        )
        models[length] = model
        fit_reports[str(length)] = {
            "in_sample_fitting_noise": evaluate(
                model,
                enroll_known[length],
                noise[length],
                known_population=ENROLLMENT_SPLIT,
            ),
            "coefficients": {
                name: float(value)
                for name, value in zip(model.feature_names, model.coefficients)
            },
            "intercept": float(model.intercept),
        }

    bundle_dir = output_dir / "bundles"
    save_prefilter_set(bundle_dir, models)

    payload = {
        "schema": REPORT_SCHEMA,
        "version": NOISE_PREFILTER_VERSION,
        "additive_only": False,
        "changes_closed_label": True,
        "gates_before_classification": True,
        "architecture_contract_change": ARCHITECTURE_CONTRACT_CHANGE,
        "feature_selection": selection_rationale(),
        "gate_floors": gate_floors(),
        "seeds": {
            "fitting_seed": int(seed),
            "fitting_seed_namespace": list(FITTING_SEED_NAMESPACE),
            "spent_novelty_seeds": {
                str(key): value for key, value in sorted(SPENT_NOVELTY_SEEDS.items())
            },
            "clean_validation_seeds_from": CLEAN_VALIDATION_SEEDS_FROM,
            "release_seed_not_spent": RELEASE_SEED_NEVER_SPENT_HERE,
            "sealed_seed_not_read": CONSUMED_SEALED_SEED,
        },
        "populations": {
            "fit": TRAIN_SPLIT,
            "threshold": ENROLLMENT_SPLIT,
            "train_audit": train_audit,
            "enroll_audit": enroll_audit,
            "selection_rows_used": 0,
            "consumed_test_rows_used": 0,
        },
        "fitting_noise": noise_provenance,
        "per_length": fit_reports,
        "bundle_sha256": {
            f"N{length}": bundle_sha256(bundle_dir / f"N{length}")
            for length in sorted(models)
        },
        "prefilter_set_sha256": prefilter_set_sha256(bundle_dir),
        "warning": (
            "The numbers under per_length are IN-SAMPLE for the fitting noise "
            "seed and are NOT validation evidence. Validation requires clean "
            "novelty seeds from "
            f"{CLEAN_VALIDATION_SEEDS_FROM} onward, scored once."
        ),
    }
    _write_json(output_dir / "noise_prefilter_fit.json", payload)
    print(f"wrote {output_dir / 'noise_prefilter_fit.json'}", flush=True)
    return payload


if __name__ == "__main__":  # pragma: no cover
    main()
