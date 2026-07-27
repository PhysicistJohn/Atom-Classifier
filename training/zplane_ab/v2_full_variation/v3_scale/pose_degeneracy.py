"""Degeneracy and confidence of the v3 time-domain pose estimate, as features.

Why this module exists
----------------------

HANDOFF 17 records the v3 open-set failure exactly: refitting the additive
rejector on v3 fusion embeddings solves chirp (AUROC 0.9716, threshold recall
0.72-1.00, against a gate sealed v2 scored 0.0 on) and breaks noise (AUROC
0.8429 -> 0.6038, threshold recall 0.3467 -> 0.0000).

The stated mechanism is that ``time_domain_geometry.estimate_band`` estimates
occupied bandwidth from non-zero-lag autocorrelation quotients and
``time_domain_invariant_patch_preprocess`` then resamples onto the dimensionless
coordinate ``u = n * bandwidth / target_frac``.  White noise has no coherent
autocorrelation, so its bandwidth estimate is degenerate; the normalisation maps
noise into the same canonical geometry as a real emission, and it stops looking
anomalous.  The estimator's own diagnostics already show the tail: measured
bandwidth p05 at N4096 is 0.0137 against a median ratio of 1.0.

The rejector is **additive**.  It never changes the closed label (see
``v3_time_domain_openset`` and ``fit_v3_openset.assert_closed_label_unchanged``),
so it may legitimately use information the invariant front end discards.  This
module exposes the quantity the front end throws away that is most directly
relevant: *how degenerate or low-confidence the pose estimate itself was*.  Those
numbers are already computed inside ``estimate_band`` and then dropped.

What this deliberately does **not** do
--------------------------------------

* It adds **no** FFT, STFT, Welch, filterbank, or frequency grid.  The whole v3
  route exists to be strictly transform-free; every quantity here is a
  ``vdot`` inner product, an ``arccos``, or arithmetic on those.  The unit suite
  asserts this both by source inspection and by executing the extractor with the
  transform APIs patched to raise.
* It introduces **no** absolute-scale dependence.  Every feature is computed
  after the estimator's own peak normalisation, so every feature is exactly
  invariant to a global non-zero amplitude scaling and to a global phase
  rotation.  ``raw_peak`` is available in the estimator context and is
  deliberately **not** used.  HANDOFF 17 option 1 ("give the rejector a
  scale-bearing feature") is a different and larger change; this module is the
  narrower one, and it can be evaluated without touching closed-set invariance.
* It fits nothing, learns nothing, and has no calibration constant derived from
  data.  It is a deterministic function of one capture.

Feature vector
--------------

``pose_degeneracy_features(iq)`` returns a fixed, ordered ``float64`` vector of
``len(FEATURE_NAMES)`` entries.  ``pose_degeneracy_report(iq)`` returns the same
values as a name -> value mapping plus a JSON-serialisable audit context.

======================================  =====================================================
name                                    meaning
======================================  =====================================================
``log10_relative_lag1_magnitude``       ``log10(|R[1]| / mean_power)`` after peak
                                        normalisation.  How much coherent structure exists at
                                        all.  A coherent emission holds an O(1) ratio; white
                                        noise decays as ``1/sqrt(N)``.
``log10_relative_lag_magnitude_mean``   mean of the same log ratio over ``BASE_LAGS``.
``log10_relative_lag_magnitude_max``    max of the same log ratio over ``BASE_LAGS``.
``log10_lag1_sampling_margin``          ``log10(|R[1]|/mean_power)`` minus ``log10`` of the
                                        estimator's own sampling-noise floor
                                        ``MIN_LAG_SIGNAL_TO_SAMPLING_NOISE / sqrt(N - 1)``.
                                        Negative means lag one is itself indistinguishable
                                        from sampling noise.
``relative_lag_magnitude_decay_slope``  least-squares slope of the log ratio against
                                        ``log2(lag)``.  A short-memory process collapses with
                                        lag; a narrowband emission does not.
``bandwidth_log_dispersion``            population standard deviation of ``log10`` of the
                                        per-lag bandwidth estimates ``arccos(q_k)/(pi k)``.
                                        A coherent emission must agree across scales.
``bandwidth_log_range``                 max minus min of the same log bandwidths.
``max_abs_unclipped_quotient``          largest ``|q_k|`` before clipping, capped at
                                        ``QUOTIENT_CAP``.  ``|q| > 1`` is outside the model:
                                        no real occupied width can produce it.
``quotient_clip_fraction``              fraction of the per-lag quotients with ``|q_k| > 1``.
``carrier_phase_coherence``             ``|mean_k exp(i (angle(R[k]) - k angle(R[1])))|`` over
                                        the higher base lags.  A single carrier makes every
                                        term identically 1.
``carrier_phase_coherence_sign_blind``  the same statistic on doubled angles, so a negative
                                        sinc lobe (a legitimate phase flip of pi) does not
                                        count as incoherence.
``degenerate_full_band_fallback``       1.0 when ``estimate_band`` took its explicit
                                        no-carrier fallback to ``(0, full_bandwidth)``.
``bandwidth_clipped_to_minimum``        1.0 when the unclipped width fell below
                                        ``min_bandwidth``.
``bandwidth_clipped_to_full``           1.0 when the unclipped width exceeded
                                        ``full_bandwidth``.
``log2_selected_base_lag``              ``log2`` of the base lag the estimator actually used
                                        (0 for lag one and for the degenerate fallback).
``high_lag_qualified_fraction``         fraction of the higher base lags that passed **both**
                                        of the estimator's admission tests, the unambiguous
                                        arc test ``k * B <= MAX_UNAMBIGUOUS_LAG_ARC`` and the
                                        sampling-noise floor.
``prefix_bandwidth_log_dispersion``     population standard deviation of ``log10(bandwidth)``
                                        re-estimated on nested prefixes of the same capture.
``prefix_centre_dispersion``            ``1 - |mean_p exp(i 2 pi centre_p)|`` over those same
                                        prefixes, in ``[0, 1]``.
``prefix_relative_lag1_log_dispersion`` population standard deviation of
                                        ``log10(|R[1]|/mean_power)`` over those prefixes.
``zero_magnitude_prefix_fraction``      fraction of the nested prefixes that are exactly
                                        all-zero and therefore admit no pose at all.  The
                                        development corpus contains gated windows whose
                                        emission starts after sample 4096; a prefix like that
                                        is the strongest possible statement that the estimate
                                        does not exist, and it is excluded from the three
                                        dispersion statistics above rather than faked.
``peak_normalized_mean_power``          ``mean(|x|^2)`` after peak normalisation, i.e. the
                                        inverse squared crest factor.  It is in ``(0, 1]``,
                                        exactly invariant to global scale and phase, and it is
                                        an amplitude *shape* quantity rather than a multiscale
                                        one; it is included because ``estimate_band`` computes
                                        it, reports it, and derives its numerical floor from
                                        it.
======================================  =====================================================

Fallback conventions, so the schema is total
--------------------------------------------

Every feature is finite for every capture the estimator itself accepts.  When a
statistic has no support (fewer than two valid lags, fewer than two usable
prefixes, or the degenerate fallback firing so that no quotient exists) the
dispersion-style features are ``0.0`` and the magnitude-style features take the
documented log floor.  The ``degenerate_full_band_fallback`` flag is what tells a
consumer that the zeros mean "no support" rather than "perfect agreement".

Captures that ``estimate_band`` rejects (non-1-D, real dtype, fewer than three
samples, NaN/infinity, all-zero) are rejected here too, with the same exception
types, deliberately: this must not silently invent a pose for an invalid input.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Any, Mapping, Sequence

import numpy as np


_V3_DIR = os.path.dirname(os.path.abspath(__file__))
_V2_DIR = os.path.dirname(_V3_DIR)
_ZPAB_DIR = os.path.dirname(_V2_DIR)
_TRAINING_DIR = os.path.dirname(_ZPAB_DIR)
for _path in (_TRAINING_DIR, _ZPAB_DIR, _V2_DIR, _V3_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import time_domain_geometry as geometry  # noqa: E402


POSE_DEGENERACY_VERSION = "pose-degeneracy-v1"

#: Log floor for magnitude ratios.  ``log10`` of anything at or below this is
#: reported as ``LOG_MAGNITUDE_FLOOR``.  ``estimate_band`` declares a capture
#: carrier-free at ``64 * eps ~ 1.4e-14`` of the mean power, and the smallest
#: physically interesting ratio (white noise at N=16384) is ``~8e-3``, so a
#: floor seven orders above machine epsilon discards nothing and keeps the
#: feature exactly reproducible under a global rotation or rescaling.
MAGNITUDE_FLOOR = 1e-9
LOG_MAGNITUDE_FLOOR = math.log10(MAGNITUDE_FLOOR)

#: Log floor for bandwidths.  This one is load-bearing, not cosmetic.  A pure
#: tone drives the lag quotient to ``q = 1``, and ``arccos`` near 1 has
#: square-root conditioning: a rounding difference of ``eps`` in ``q`` moves
#: ``arccos(q)`` by ``~2e-8``.  Without a floor, ``log10`` of that quantity is
#: not reproducible under a global phase rotation, and the dispersion features
#: would report pure numerical noise as multiscale disagreement.  The estimator
#: itself clips any width below ``MIN_BANDWIDTH = 1/512`` anyway, so a floor two
#: hundred times below that clip cannot hide a real occupied width.
BANDWIDTH_FLOOR = 1e-5
LOG_BANDWIDTH_FLOOR = math.log10(BANDWIDTH_FLOOR)

#: ``|q|`` is unbounded above for incoherent input; cap it so one pathological
#: row cannot dominate a downstream rank or distance.
QUOTIENT_CAP = 10.0

#: Nested prefixes used for the stability features.  ``1.0`` (the whole capture)
#: is included so the full-length estimate participates in the dispersion.
PREFIX_FRACTIONS = (0.25, 0.5, 1.0)

#: A prefix shorter than this cannot supply the estimator's own maximum lag plus
#: a usable average, so it is not counted as a prefix observation at all.
MIN_PREFIX_SAMPLES = 2 * geometry.MAX_LAG + 2

FEATURE_NAMES: tuple[str, ...] = (
    "log10_relative_lag1_magnitude",
    "log10_relative_lag_magnitude_mean",
    "log10_relative_lag_magnitude_max",
    "log10_lag1_sampling_margin",
    "relative_lag_magnitude_decay_slope",
    "bandwidth_log_dispersion",
    "bandwidth_log_range",
    "max_abs_unclipped_quotient",
    "quotient_clip_fraction",
    "carrier_phase_coherence",
    "carrier_phase_coherence_sign_blind",
    "degenerate_full_band_fallback",
    "bandwidth_clipped_to_minimum",
    "bandwidth_clipped_to_full",
    "log2_selected_base_lag",
    "high_lag_qualified_fraction",
    "prefix_bandwidth_log_dispersion",
    "prefix_centre_dispersion",
    "prefix_relative_lag1_log_dispersion",
    "zero_magnitude_prefix_fraction",
    "peak_normalized_mean_power",
)

FEATURE_COUNT = len(FEATURE_NAMES)


def feature_names() -> tuple[str, ...]:
    """Return the frozen feature order."""
    return FEATURE_NAMES


# ---------------------------------------------------------------------------
# primitive quantities, all transform-free
# ---------------------------------------------------------------------------


def _nonzero_lag(x: np.ndarray, lag: int) -> complex:
    """Mean ``conj(x[n]) * x[n + lag]``.

    Byte-for-byte the same arithmetic as
    ``time_domain_geometry._nonzero_lag``; a unit test asserts the equality so
    this module cannot drift from the estimator it is describing.
    """
    return complex(np.vdot(x[:-lag], x[lag:]) / (len(x) - lag))


def _lag_table(x: np.ndarray) -> dict:
    lags = sorted(
        set(geometry.BASE_LAGS + tuple(2 * lag for lag in geometry.BASE_LAGS))
    )
    return {lag: _nonzero_lag(x, lag) for lag in lags if lag < len(x)}


def _safe_log10(value: float, floor: float, log_floor: float) -> float:
    magnitude = float(value)
    if not math.isfinite(magnitude) or magnitude <= floor:
        return float(log_floor)
    return float(math.log10(magnitude))


def _population_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(np.asarray(values, dtype=np.float64)))


def _circular_dispersion(cycles: Sequence[float]) -> float:
    """``1 - |mean exp(i 2 pi c)|`` over frequencies expressed in cycles."""
    if len(cycles) < 2:
        return 0.0
    angles = 2.0 * np.pi * np.asarray(cycles, dtype=np.float64)
    resultant = complex(np.mean(np.cos(angles) + 1j * np.sin(angles)))
    return float(np.clip(1.0 - abs(resultant), 0.0, 1.0))


def _sampling_floor(length: int, lag: int) -> float:
    return float(
        geometry.MIN_LAG_SIGNAL_TO_SAMPLING_NOISE / np.sqrt(length - lag)
    )


def _prefix_lengths(length: int) -> tuple[int, ...]:
    candidates = []
    for fraction in PREFIX_FRACTIONS:
        value = int(length * float(fraction))
        if value >= MIN_PREFIX_SAMPLES and value <= length:
            candidates.append(value)
    return tuple(sorted(set(candidates)))


# ---------------------------------------------------------------------------
# the extractor
# ---------------------------------------------------------------------------


def pose_degeneracy_report(
    iq: Any,
    *,
    min_bandwidth: float = geometry.MIN_BANDWIDTH,
    full_bandwidth: float = geometry.FULL_BAND_BW,
) -> dict:
    """Return ``{"features", "values", "context"}`` for one raw capture.

    ``features`` is the ordered ``float64`` vector, ``values`` the same numbers
    keyed by :data:`FEATURE_NAMES`, and ``context`` a JSON-serialisable audit of
    the intermediate quantities, including the estimator's own returned context.
    """
    centre, bandwidth, estimator_context = geometry.estimate_band(
        iq,
        min_bandwidth=min_bandwidth,
        full_bandwidth=full_bandwidth,
    )

    # Repeat the estimator's own peak normalisation rather than importing its
    # private helper, so this module owns every number it reports.  Validation
    # already happened inside ``estimate_band`` above.
    x = np.asarray(iq).astype(np.complex128, copy=False)
    peak = float(np.max(np.abs(x)))
    x = x / peak
    length = int(len(x))
    mean_power = float(np.mean(x.real * x.real + x.imag * x.imag))

    lags = _lag_table(x)
    numerical_floor = (
        geometry._CORRELATION_EPSILON_FACTOR
        * float(np.finfo(np.float64).eps)
        * mean_power
    )

    # --- coherent-structure magnitudes -----------------------------------
    relative: dict = {}
    for lag in geometry.BASE_LAGS:
        if lag in lags:
            relative[lag] = float(abs(lags[lag])) / mean_power
    log_relative = {
        lag: _safe_log10(value, MAGNITUDE_FLOOR, LOG_MAGNITUDE_FLOOR)
        for lag, value in relative.items()
    }
    log_relative_lag1 = log_relative.get(1, LOG_MAGNITUDE_FLOOR)
    log_relative_values = [log_relative[lag] for lag in sorted(log_relative)]
    if log_relative_values:
        log_relative_mean = float(np.mean(log_relative_values))
        log_relative_max = float(np.max(log_relative_values))
    else:
        log_relative_mean = LOG_MAGNITUDE_FLOOR
        log_relative_max = LOG_MAGNITUDE_FLOOR

    sampling_margin = log_relative_lag1 - _safe_log10(
        _sampling_floor(length, 1), MAGNITUDE_FLOOR, LOG_MAGNITUDE_FLOOR
    )

    decay_slope = 0.0
    if len(log_relative) >= 2:
        abscissa = np.asarray(
            [math.log2(lag) for lag in sorted(log_relative)], dtype=np.float64
        )
        ordinate = np.asarray(
            [log_relative[lag] for lag in sorted(log_relative)],
            dtype=np.float64,
        )
        centred = abscissa - abscissa.mean()
        denominator = float(np.dot(centred, centred))
        if denominator > 0.0:
            decay_slope = float(
                np.dot(centred, ordinate - ordinate.mean()) / denominator
            )

    # --- multiscale quotient disagreement --------------------------------
    per_lag_bandwidth: dict = {}
    per_lag_quotient: dict = {}
    for base_lag in geometry.BASE_LAGS:
        doubled = 2 * base_lag
        if base_lag not in lags or doubled not in lags:
            continue
        base_value = lags[base_lag]
        base_magnitude = float(abs(base_value))
        if base_magnitude <= numerical_floor:
            continue
        base_unit = base_value / base_magnitude
        quotient = float(
            (lags[doubled] * np.conj(base_unit * base_unit)).real
            / base_magnitude
        )
        clipped = float(np.clip(quotient, -1.0, 1.0))
        per_lag_quotient[base_lag] = quotient
        per_lag_bandwidth[base_lag] = float(
            np.arccos(clipped) / (np.pi * base_lag)
        )

    log_bandwidths = [
        _safe_log10(value, BANDWIDTH_FLOOR, LOG_BANDWIDTH_FLOOR)
        for _lag, value in sorted(per_lag_bandwidth.items())
    ]
    bandwidth_log_dispersion = _population_std(log_bandwidths)
    bandwidth_log_range = (
        float(max(log_bandwidths) - min(log_bandwidths))
        if len(log_bandwidths) >= 2
        else 0.0
    )

    if per_lag_quotient:
        absolute = [abs(value) for value in per_lag_quotient.values()]
        max_abs_quotient = float(min(max(absolute), QUOTIENT_CAP))
        clip_fraction = float(
            sum(1 for value in absolute if value > 1.0) / len(absolute)
        )
    else:
        max_abs_quotient = 0.0
        clip_fraction = 0.0

    # --- carrier phase agreement across scales ---------------------------
    coherence = 0.0
    coherence_sign_blind = 0.0
    if 1 in lags and float(abs(lags[1])) > numerical_floor:
        reference = float(np.angle(lags[1]))
        residuals = []
        for base_lag in geometry.BASE_LAGS[1:]:
            if base_lag not in lags:
                continue
            if float(abs(lags[base_lag])) <= numerical_floor:
                continue
            residuals.append(
                float(np.angle(lags[base_lag])) - base_lag * reference
            )
        if residuals:
            angles = np.asarray(residuals, dtype=np.float64)
            coherence = float(
                abs(complex(np.mean(np.cos(angles) + 1j * np.sin(angles))))
            )
            doubled = 2.0 * angles
            coherence_sign_blind = float(
                abs(
                    complex(
                        np.mean(np.cos(doubled) + 1j * np.sin(doubled))
                    )
                )
            )
    coherence = float(np.clip(coherence, 0.0, 1.0))
    coherence_sign_blind = float(np.clip(coherence_sign_blind, 0.0, 1.0))

    # --- the estimator's own clip and admission decisions -----------------
    fallback = bool(estimator_context["degenerate_full_band_fallback"])
    selected_lag = estimator_context["selected_base_lag"]
    log2_selected_lag = (
        float(math.log2(int(selected_lag))) if selected_lag else 0.0
    )
    base_lag_one_bandwidth = estimator_context["base_lag_one_bandwidth"]
    higher_lags = geometry.BASE_LAGS[1:]
    qualified = 0
    considered = 0
    for base_lag in higher_lags:
        doubled = 2 * base_lag
        considered += 1
        if doubled not in lags or fallback or base_lag_one_bandwidth is None:
            continue
        relative_magnitude = float(abs(lags[base_lag])) / mean_power
        arc_ok = (
            base_lag * float(base_lag_one_bandwidth)
            <= geometry.MAX_UNAMBIGUOUS_LAG_ARC
        )
        floor_ok = relative_magnitude >= _sampling_floor(length, base_lag)
        if arc_ok and floor_ok:
            qualified += 1
    qualified_fraction = float(qualified / considered) if considered else 0.0

    # --- stability across nested prefixes of the same capture -------------
    prefix_lengths = _prefix_lengths(length)
    prefix_bandwidths = []
    prefix_centres = []
    prefix_relative_lag1 = []
    prefix_audit = []
    zero_prefixes = 0
    for prefix in prefix_lengths:
        window = x[:prefix]
        if float(np.max(np.abs(window))) == 0.0:
            # A gated capture whose emission starts later has no pose on this
            # prefix at all.  ``estimate_band`` refuses it, correctly; counting
            # it is more honest than substituting a fabricated estimate.
            zero_prefixes += 1
            prefix_audit.append(
                {
                    "length": int(prefix),
                    "centre": None,
                    "bandwidth": None,
                    "degenerate_full_band_fallback": None,
                    "zero_magnitude": True,
                }
            )
            continue
        prefix_centre, prefix_bandwidth, prefix_context = geometry.estimate_band(
            window,
            min_bandwidth=min_bandwidth,
            full_bandwidth=full_bandwidth,
        )
        prefix_bandwidths.append(
            _safe_log10(prefix_bandwidth, BANDWIDTH_FLOOR, LOG_BANDWIDTH_FLOOR)
        )
        prefix_centres.append(float(prefix_centre))
        prefix_relative_lag1.append(
            _safe_log10(
                float(prefix_context["relative_lag1_magnitude"]),
                MAGNITUDE_FLOOR,
                LOG_MAGNITUDE_FLOOR,
            )
        )
        prefix_audit.append(
            {
                "length": int(prefix),
                "centre": float(prefix_centre),
                "bandwidth": float(prefix_bandwidth),
                "degenerate_full_band_fallback": bool(
                    prefix_context["degenerate_full_band_fallback"]
                ),
                "zero_magnitude": False,
            }
        )

    values = {
        "log10_relative_lag1_magnitude": float(log_relative_lag1),
        "log10_relative_lag_magnitude_mean": float(log_relative_mean),
        "log10_relative_lag_magnitude_max": float(log_relative_max),
        "log10_lag1_sampling_margin": float(sampling_margin),
        "relative_lag_magnitude_decay_slope": float(decay_slope),
        "bandwidth_log_dispersion": float(bandwidth_log_dispersion),
        "bandwidth_log_range": float(bandwidth_log_range),
        "max_abs_unclipped_quotient": float(max_abs_quotient),
        "quotient_clip_fraction": float(clip_fraction),
        "carrier_phase_coherence": float(coherence),
        "carrier_phase_coherence_sign_blind": float(coherence_sign_blind),
        "degenerate_full_band_fallback": 1.0 if fallback else 0.0,
        "bandwidth_clipped_to_minimum": (
            1.0 if bool(estimator_context["bandwidth_clipped_to_minimum"]) else 0.0
        ),
        "bandwidth_clipped_to_full": (
            1.0 if bool(estimator_context["bandwidth_clipped_to_full"]) else 0.0
        ),
        "log2_selected_base_lag": float(log2_selected_lag),
        "high_lag_qualified_fraction": float(qualified_fraction),
        "prefix_bandwidth_log_dispersion": _population_std(prefix_bandwidths),
        "prefix_centre_dispersion": _circular_dispersion(prefix_centres),
        "prefix_relative_lag1_log_dispersion": _population_std(
            prefix_relative_lag1
        ),
        "zero_magnitude_prefix_fraction": (
            float(zero_prefixes / len(prefix_lengths)) if prefix_lengths else 0.0
        ),
        "peak_normalized_mean_power": float(mean_power),
    }
    if tuple(values) != FEATURE_NAMES:
        raise AssertionError("pose degeneracy feature order drifted from schema")

    features = np.asarray(
        [values[name] for name in FEATURE_NAMES], dtype=np.float64
    )
    if features.shape != (FEATURE_COUNT,) or not np.all(np.isfinite(features)):
        raise AssertionError("pose degeneracy features must be finite")

    context = {
        "version": POSE_DEGENERACY_VERSION,
        "estimator_version": geometry.ESTIMATOR_VERSION,
        "uses_frequency_transform": False,
        "raw_length": length,
        "centre": float(centre),
        "bandwidth": float(bandwidth),
        "mean_power_after_peak_normalization": mean_power,
        "relative_lag_magnitude": {
            str(lag): float(value) for lag, value in sorted(relative.items())
        },
        "per_lag_bandwidth": {
            str(lag): float(value)
            for lag, value in sorted(per_lag_bandwidth.items())
        },
        "per_lag_quotient": {
            str(lag): float(value)
            for lag, value in sorted(per_lag_quotient.items())
        },
        "prefix_observations": prefix_audit,
        "estimator_context": {
            key: value
            for key, value in estimator_context.items()
            # ``raw_peak`` is the only scale-bearing field the estimator
            # returns.  It is excluded so that no consumer of this audit can
            # accidentally reintroduce absolute scale.
            if key != "raw_peak"
        },
    }
    return {"features": features, "values": values, "context": context}


def pose_degeneracy_features(
    iq: Any,
    *,
    min_bandwidth: float = geometry.MIN_BANDWIDTH,
    full_bandwidth: float = geometry.FULL_BAND_BW,
) -> np.ndarray:
    """Return the ordered ``float64`` degeneracy vector for one raw capture."""
    return pose_degeneracy_report(
        iq,
        min_bandwidth=min_bandwidth,
        full_bandwidth=full_bandwidth,
    )["features"]


def pose_degeneracy_matrix(
    captures: Sequence[Any],
    *,
    min_bandwidth: float = geometry.MIN_BANDWIDTH,
    full_bandwidth: float = geometry.FULL_BAND_BW,
) -> np.ndarray:
    """Stack :func:`pose_degeneracy_features` over an ordered set of captures."""
    rows = [
        pose_degeneracy_features(
            capture,
            min_bandwidth=min_bandwidth,
            full_bandwidth=full_bandwidth,
        )
        for capture in captures
    ]
    if not rows:
        return np.empty((0, FEATURE_COUNT), dtype=np.float64)
    return np.stack(rows).astype(np.float64)


def pose_degeneracy_metadata() -> dict:
    """Return the stable, serializable extractor contract."""
    return {
        "version": POSE_DEGENERACY_VERSION,
        "estimator_version": geometry.ESTIMATOR_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "feature_count": FEATURE_COUNT,
        "base_lags": list(geometry.BASE_LAGS),
        "maximum_lag": geometry.MAX_LAG,
        "prefix_fractions": list(PREFIX_FRACTIONS),
        "minimum_prefix_samples": MIN_PREFIX_SAMPLES,
        "quotient_cap": QUOTIENT_CAP,
        "magnitude_floor": MAGNITUDE_FLOOR,
        "bandwidth_floor": BANDWIDTH_FLOOR,
        "uses_frequency_transform": False,
        "learned_calibration": False,
        "uses_absolute_scale": False,
        "invariant_to_global_phase": True,
        "invariant_to_global_scale": True,
        "changes_closed_label": False,
    }


def assert_schema(values: Mapping[str, Any]) -> None:
    """Raise unless ``values`` carries exactly the frozen feature schema."""
    if tuple(values) != FEATURE_NAMES:
        raise ValueError("pose degeneracy schema mismatch")
