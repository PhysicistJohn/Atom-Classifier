"""Deterministic occupied-band geometry from time-domain correlations only.

The production ``hybrid-v3`` detector estimates centre frequency and occupied
bandwidth from a Welch periodogram.  That detector is useful, but it means the
otherwise time-domain invariant-patch path is not end-to-end free of spectral
grids.  This module supplies an isolated alternative based on quotients of
non-zero autocorrelation lags.

For a signal with a rectangular occupied spectrum of width ``B`` centred at
``c`` (both in cycles/sample), its noise-free autocorrelation is

    R[k] = A exp(j 2 pi c k) sinc(B k).

White receiver noise contributes to ``R[0]`` but not, in expectation, to the
non-zero lags.  Consequently

    angle(R[1]) / (2 pi) = c
    Re(R[2] exp(-j 4 pi c)) / abs(R[1]) = cos(pi B).

The second identity is a lag quotient: the unknown signal power and white-noise
attenuation cancel.  The same identity holds at a base lag ``k``:

    Re(R[2k] exp(-j 4 pi c k)) / abs(R[k]) = cos(pi k B).

For narrow signals we use the largest reliable lag from ``{1,2,4,8,16}``.
This raises width sensitivity and rejects short-memory coloured noise created
by a sample-rate converter.  The lag-one estimate selects an unambiguous arc;
broad signals stay at lag one.  This gives a deterministic equivalent occupied
width without an FFT, STFT, filterbank, frequency grid, or learned calibration.
Real RF spectra are not exactly rectangular, so the result is an equivalent
geometry, not a claim to recover a regulatory bandwidth exactly.

The construction has explicit nuisance behaviour away from the Nyquist seam:

* finite non-zero gain and global phase leave both outputs unchanged;
* a carrier shift rotates lag ``k`` by exactly ``k`` times the shift, so centre
  is equivariant and bandwidth is invariant;
* physical sample-rate scaling changes the lag quotient according to the
  normalized bandwidth, without introducing a fixed sample-count feature.

Only the explicit full-band degeneracy fallback breaks exact scale equivariance.
The patch front end supplies a length-dependent resolution floor that keeps the
minimum dimensionless active span fixed; the floor therefore scales with sample
rate rather than leaking raw capture length.  Every clipping decision is
reported in the returned audit context.
"""
from __future__ import annotations

from typing import Any

import numpy as np


ESTIMATOR_VERSION = "time-correlation-v1"
MIN_BANDWIDTH = 1.0 / 512.0
FULL_BAND_BW = 0.95
_CORRELATION_EPSILON_FACTOR = 64.0
BASE_LAGS = (1, 2, 4, 8, 16)
MAX_LAG = 2 * max(BASE_LAGS)
MAX_UNAMBIGUOUS_LAG_ARC = 0.45
MIN_LAG_SIGNAL_TO_SAMPLING_NOISE = 4.0


def _finite_fraction(value: Any, name: str, *, inclusive_zero: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        interval = "[0, 1]" if inclusive_zero else "(0, 1]"
        raise ValueError(f"{name} must be a finite fraction in {interval}") from exc
    lower_ok = result >= 0.0 if inclusive_zero else result > 0.0
    if not np.isfinite(result) or not lower_ok or result > 1.0:
        interval = "[0, 1]" if inclusive_zero else "(0, 1]"
        raise ValueError(f"{name} must be a finite fraction in {interval}")
    return result


def _capture(iq: Any) -> np.ndarray:
    x = np.asarray(iq)
    if x.ndim != 1 or len(x) < 3:
        raise ValueError(
            "expected a complex 1-D I/Q capture with at least three samples, "
            f"got {x.shape}"
        )
    if not np.iscomplexobj(x):
        raise TypeError(f"expected complex I/Q, got dtype={x.dtype}")
    if not np.all(np.isfinite(x.real)) or not np.all(np.isfinite(x.imag)):
        raise ValueError("I/Q capture contains NaN or infinity")
    return x.astype(np.complex128, copy=False)


def _wrap_cycles(value: float) -> float:
    return float((value + 0.5) % 1.0 - 0.5)


def _peak_normalize(x: np.ndarray) -> tuple[np.ndarray, float]:
    peak = float(np.max(np.abs(x)))
    if not np.isfinite(peak):
        raise ValueError("I/Q peak magnitude is not finite")
    if peak == 0.0:
        raise ValueError("I/Q capture has zero magnitude")
    return x / peak, peak


def _nonzero_lag(x: np.ndarray, lag: int) -> complex:
    """Mean ``conj(x[n]) * x[n+lag]`` with no zero padding."""
    return complex(np.vdot(x[:-lag], x[lag:]) / (len(x) - lag))


def minimum_bandwidth_for_active_span(
    raw_length: Any,
    patch_length: Any,
    target_frac: Any,
    *,
    full_bandwidth: float = FULL_BAND_BW,
) -> float:
    """Resolution floor that maps every unresolved capture to one full patch.

    With ``u = n * bandwidth / target_frac``, choosing

    ``bandwidth = (patch_length - 1) * target_frac / (raw_length - 1)``

    gives exactly ``patch_length`` supported integer-``u`` samples.  If physical
    sample rate changes by ``s`` and sample count changes by ``1/s``, this floor
    changes by ``s`` as well.  It therefore avoids both zero padding and a fixed
    FFT-bin/sample-count fingerprint.
    """
    if (
        isinstance(raw_length, (bool, np.bool_))
        or not isinstance(raw_length, (int, np.integer))
        or int(raw_length) < 3
    ):
        raise ValueError("raw_length must be an integer of at least three")
    if (
        isinstance(patch_length, (bool, np.bool_))
        or not isinstance(patch_length, (int, np.integer))
        or int(patch_length) <= 0
    ):
        raise ValueError("patch_length must be a positive integer")
    target = _finite_fraction(target_frac, "target_frac")
    full = _finite_fraction(full_bandwidth, "full_bandwidth")
    required = (int(patch_length) - 1) * target / (int(raw_length) - 1)
    return float(np.clip(required, np.finfo(np.float64).eps, full))


def estimate_band(
    iq: Any,
    *,
    min_bandwidth: float = MIN_BANDWIDTH,
    full_bandwidth: float = FULL_BAND_BW,
) -> tuple[float, float, dict[str, Any]]:
    """Estimate ``(centre, equivalent_bandwidth, audit_context)``.

    ``centre`` is wrapped to ``[-0.5, 0.5)`` and both frequencies are measured
    in cycles/sample.  No absolute signal-power threshold is used.  A capture
    whose first non-zero lag is numerically indistinguishable from zero has no
    stable time-domain carrier direction and therefore fails explicitly to the
    full-band geometry ``(0, full_bandwidth)``.
    """
    minimum = _finite_fraction(min_bandwidth, "min_bandwidth")
    full = _finite_fraction(full_bandwidth, "full_bandwidth")
    if minimum > full:
        raise ValueError("min_bandwidth may not exceed full_bandwidth")

    x, raw_peak = _peak_normalize(_capture(iq))
    mean_power = float(np.mean(x.real * x.real + x.imag * x.imag))
    if not np.isfinite(mean_power) or mean_power <= 0.0:
        raise ValueError("peak-normalized I/Q has invalid or zero power")

    lags = {
        lag: _nonzero_lag(x, lag)
        for lag in sorted(set(BASE_LAGS + tuple(2 * lag for lag in BASE_LAGS)))
        if lag < len(x)
    }
    lag1 = lags[1]
    lag2 = lags[2]
    lag1_magnitude = float(abs(lag1))
    relative_lag1 = lag1_magnitude / mean_power
    numerical_floor = (
        _CORRELATION_EPSILON_FACTOR
        * np.finfo(np.float64).eps
        * mean_power
    )

    if lag1_magnitude <= numerical_floor:
        context = {
            "version": ESTIMATOR_VERSION,
            "raw_length": int(len(x)),
            "raw_peak": raw_peak,
            "mean_power_after_peak_normalization": mean_power,
            "lag1_real": float(lag1.real),
            "lag1_imag": float(lag1.imag),
            "lag2_real": float(lag2.real),
            "lag2_imag": float(lag2.imag),
            "relative_lag1_magnitude": relative_lag1,
            "lag_quotient": None,
            "clipped_lag_quotient": None,
            "unclipped_equivalent_bandwidth": None,
            "base_lag_one_bandwidth": None,
            "selected_base_lag": None,
            "minimum_bandwidth": minimum,
            "full_bandwidth": full,
            "bandwidth_clipped_to_minimum": False,
            "bandwidth_clipped_to_full": False,
            "degenerate_full_band_fallback": True,
            "uses_frequency_transform": False,
        }
        return 0.0, full, context

    carrier_phase = float(np.angle(lag1))
    preliminary_centre = _wrap_cycles(carrier_phase / (2.0 * np.pi))
    unit_carrier = lag1 / lag1_magnitude

    # Demodulate lag two by twice the lag-one carrier phase before taking the
    # real quotient.  This makes a global carrier shift cancel algebraically.
    quotient = float((lag2 * np.conj(unit_carrier * unit_carrier)).real)
    quotient /= lag1_magnitude
    clipped_quotient = float(np.clip(quotient, -1.0, 1.0))
    base_bandwidth = float(np.arccos(clipped_quotient) / np.pi)

    selected_lag = 1
    centre = preliminary_centre
    raw_bandwidth = base_bandwidth
    selected_quotient = quotient
    selected_clipped_quotient = clipped_quotient

    # A higher base lag magnifies a narrow occupied width and rejects noise whose
    # correlation dies before that lag. Descending order is deterministic. The
    # lag-one width keeps k*B inside the monotonic arccos branch, and the
    # sampling-noise guard avoids division by an accidental high-lag correlation.
    for base_lag in reversed(BASE_LAGS[1:]):
        doubled = 2 * base_lag
        if doubled not in lags:
            continue
        relative = float(abs(lags[base_lag]) / mean_power)
        sampling_floor = MIN_LAG_SIGNAL_TO_SAMPLING_NOISE / np.sqrt(
            len(x) - base_lag
        )
        if (
            base_lag * base_bandwidth > MAX_UNAMBIGUOUS_LAG_ARC
            or relative < sampling_floor
        ):
            continue
        base_value = lags[base_lag]
        base_unit = base_value / abs(base_value)
        candidate_quotient = float(
            (
                lags[doubled]
                * np.conj(base_unit * base_unit)
            ).real
            / abs(base_value)
        )
        candidate_clipped = float(np.clip(candidate_quotient, -1.0, 1.0))
        candidate_bandwidth = float(
            np.arccos(candidate_clipped) / (np.pi * base_lag)
        )

        # angle(R[k]) has k circular roots. On the selected positive-sinc arc,
        # the root nearest the lag-one carrier is the unambiguous carrier.
        root_zero = float(np.angle(base_value) / (2.0 * np.pi * base_lag))
        roots = [
            _wrap_cycles(root_zero + branch / base_lag)
            for branch in range(base_lag)
        ]
        centre = min(
            roots,
            key=lambda candidate: abs(
                _wrap_cycles(candidate - preliminary_centre)
            ),
        )
        raw_bandwidth = candidate_bandwidth
        selected_lag = base_lag
        selected_quotient = candidate_quotient
        selected_clipped_quotient = candidate_clipped
        break

    bandwidth = float(np.clip(raw_bandwidth, minimum, full))

    context = {
        "version": ESTIMATOR_VERSION,
        "raw_length": int(len(x)),
        "raw_peak": raw_peak,
        "mean_power_after_peak_normalization": mean_power,
        "lag1_real": float(lag1.real),
        "lag1_imag": float(lag1.imag),
        "lag2_real": float(lag2.real),
        "lag2_imag": float(lag2.imag),
        "relative_lag1_magnitude": relative_lag1,
        "lag_quotient": selected_quotient,
        "clipped_lag_quotient": selected_clipped_quotient,
        "unclipped_equivalent_bandwidth": raw_bandwidth,
        "base_lag_one_bandwidth": base_bandwidth,
        "selected_base_lag": selected_lag,
        "minimum_bandwidth": minimum,
        "full_bandwidth": full,
        "bandwidth_clipped_to_minimum": bool(raw_bandwidth < minimum),
        "bandwidth_clipped_to_full": bool(raw_bandwidth > full),
        "degenerate_full_band_fallback": False,
        "uses_frequency_transform": False,
    }
    return centre, bandwidth, context


def estimator_metadata() -> dict[str, Any]:
    """Return the stable, serializable estimator contract."""
    return {
        "version": ESTIMATOR_VERSION,
        "method": "multiscale non-zero-lag autocorrelation quotient",
        "base_lags": list(BASE_LAGS),
        "maximum_lag": MAX_LAG,
        "max_unambiguous_lag_arc": MAX_UNAMBIGUOUS_LAG_ARC,
        "min_lag_signal_to_sampling_noise": MIN_LAG_SIGNAL_TO_SAMPLING_NOISE,
        "minimum_bandwidth": MIN_BANDWIDTH,
        "full_bandwidth": FULL_BAND_BW,
        "uses_frequency_transform": False,
        "learned_calibration": False,
    }
