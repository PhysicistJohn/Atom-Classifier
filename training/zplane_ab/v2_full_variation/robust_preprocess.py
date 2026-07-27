"""Training-only circular DSP front end for follow-up experiments.

This module deliberately does not replace ``training/preprocess.py``.  It keeps
the production helper implementations for Welch PSD, interpolation, fitting,
channel conversion, and handcrafted features, while fixing two experimental
front-end contracts:

* occupied bands are intervals on a *circle*, so a band crossing the
  ``-0.5/+0.5`` FFT seam must not be mistaken for an almost-full-band signal;
* detection and RMS normalization are invariant to every finite, non-zero
  global input gain instead of depending on absolute epsilon thresholds.

Pair construction
-----------------
Call ``preprocess`` on the impaired member first.  Its returned ``center``,
``bw``, and ``resample_frac`` may then be supplied to the clean member as
``force_center``, ``force_bw``, and ``force_resample_frac``.  This gives both
members one downconversion and one interpolation grid, including the exact
scale-jitter draw.  When both center and bandwidth are forced, the clean member
does not run an unnecessary independent detector.

This is research code.  There is intentionally no TypeScript port and no
production asset may claim parity with it.
"""
from __future__ import annotations

import os
import sys

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.dirname(os.path.dirname(HERE))
if TRAINING not in sys.path:
    sys.path.insert(0, TRAINING)

import preprocess as pp  # noqa: E402


L_OUT = pp.L_OUT
TARGET_FRAC = pp.TARGET_FRAC
NFFT = pp.NFFT
ENERGY_EDGE = pp.ENERGY_EDGE
SMOOTH = pp.SMOOTH
FULL_BAND_BW = 0.95

# The expected spectral-flatness deficit of smoothed white-noise Welch spectra
# scales approximately as 1 / segment_count.  This deliberately conservative
# coefficient was checked at 1, 3, 7, 15, and 63 overlapping segments (capture
# lengths 512 through 16384): the threshold stayed below the least-flat white
# realization while still admitting broad, low-SNR occupied bands once enough
# averages make their contrast measurable.
WHITE_FLATNESS_DEFICIT = 0.30

# A supported occupied band leaves at least 5% of the frequency circle as a
# guard because output bandwidth is capped at FULL_BAND_BW.  Estimating the
# noise floor in the lowest-power *circular contiguous* guard avoids the
# median-contamination failure that affects signals occupying more than half
# the spectrum.  Smoothing makes the guard median close to the noise mean; a
# modest margin suppresses residual noise bins before the 99%-energy scan.
GUARD_FRACTION = 1.0 - FULL_BAND_BW
GUARD_FLOOR_SCALE = 1.30

# If guard-floor subtraction removes every meaningful bin, there is no
# defensible occupied interval; use the explicit full-band fallback.
MIN_EXCESS_FRACTION = 1e-6


def _as_finite_complex_1d(iq: np.ndarray) -> np.ndarray:
    x = np.asarray(iq)
    if x.ndim != 1 or len(x) == 0:
        raise ValueError(f"expected a non-empty 1-D I/Q capture, got shape={x.shape}")
    if not np.iscomplexobj(x):
        raise TypeError(f"expected complex I/Q, got dtype={x.dtype}")
    if not np.all(np.isfinite(x.real)) or not np.all(np.isfinite(x.imag)):
        raise ValueError("I/Q capture contains NaN or infinity")
    return x.astype(np.complex128, copy=False)


def _peak_normalized(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Return ``x/max(|x|)`` without absolute-scale underflow or overflow."""
    peak = float(np.max(np.abs(x)))
    if peak == 0.0:
        return np.zeros_like(x, dtype=np.complex128), 0.0
    if not np.isfinite(peak):
        raise ValueError("I/Q magnitude is not finite")
    return x / peak, peak


def _circular_smooth(psd: np.ndarray, width: int = SMOOTH) -> np.ndarray:
    """Moving average with periodic, rather than zero-filled, FFT boundaries."""
    p = np.asarray(psd, dtype=np.float64)
    if width <= 1:
        return p
    if width > len(p):
        raise ValueError("smoothing width may not exceed PSD length")
    offsets = np.arange(width) - (width // 2)
    return sum((np.roll(p, int(offset)) for offset in offsets), np.zeros_like(p)) / width


def _spectral_flatness(psd: np.ndarray) -> float:
    p = np.asarray(psd, dtype=np.float64)
    mean = float(np.mean(p))
    if mean <= 0.0:
        return 1.0
    # PSD was peak-normalized before this point.  ``tiny`` only defines log(0);
    # it is not an absolute input-power decision threshold.
    tiny = np.finfo(np.float64).tiny
    return float(np.exp(np.mean(np.log(np.maximum(p, tiny)))) / mean)


def _welch_segment_count(length: int, nfft: int) -> int:
    """Number of overlapping segments used by ``pp.welch_psd``."""
    if length < nfft:
        return 1
    return 1 + (length - nfft) // (nfft // 2)


def _white_flatness_threshold(length: int, nfft: int) -> float:
    count = _welch_segment_count(length, nfft)
    return max(0.0, 1.0 - WHITE_FLATNESS_DEFICIT / count)


def _minimum_circular_guard(psd: np.ndarray) -> np.ndarray:
    """Return the lowest-energy supported contiguous guard on the FFT circle."""
    p = np.asarray(psd, dtype=np.float64)
    n = len(p)
    guard_bins = min(
        n,
        max(2 * SMOOTH + 1, int(np.ceil(GUARD_FRACTION * n))),
    )
    extended = np.concatenate([p, p[: guard_bins - 1]])
    window_sums = np.convolve(
        extended,
        np.ones(guard_bins, dtype=np.float64),
        mode="valid",
    )[:n]
    start = int(np.argmin(window_sums))
    return p[(start + np.arange(guard_bins)) % n]


def _shortest_circular_interval(
    weights: np.ndarray,
    retained_fraction: float,
) -> tuple[int, int]:
    """Inclusive unwrapped bin endpoints of the shortest retained-energy arc.

    ``start`` is in ``[0, N)`` and ``end`` is in ``[start, start + N)``.
    A two-pointer scan over one doubled PSD is O(N), deterministic, and avoids
    selecting an arbitrary FFT seam before the occupied interval is known.
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or len(w) == 0 or np.any(w < 0.0):
        raise ValueError("circular interval weights must be a non-empty non-negative vector")
    total = float(w.sum())
    if total <= 0.0:
        return 0, len(w) - 1
    if not 0.0 < retained_fraction <= 1.0:
        raise ValueError("retained_fraction must lie in (0, 1]")

    n = len(w)
    target = retained_fraction * total
    doubled = np.concatenate([w, w])
    end_exclusive = 0
    running = 0.0
    best: tuple[int, float, int, int] | None = None

    for start in range(n):
        if end_exclusive < start:
            end_exclusive = start
            running = 0.0
        while end_exclusive < start + n and running < target:
            running += float(doubled[end_exclusive])
            end_exclusive += 1
        if running >= target:
            width_bins = end_exclusive - start
            # Prefer the least overshoot when equal-width arcs exist.  This
            # removes an otherwise arbitrary dependence on FFT index zero.
            candidate = (
                width_bins,
                running - target,
                start,
                end_exclusive - 1,
            )
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        if end_exclusive > start:
            running -= float(doubled[start])
            if running < 0.0 and running > -1e-12 * total:
                running = 0.0

    if best is None:
        return 0, n - 1
    return best[2], best[3]


def _wrap_cycles(value: float) -> float:
    """Map cycles/sample to the half-open interval [-0.5, 0.5)."""
    return float((value + 0.5) % 1.0 - 0.5)


def estimate_band(iq: np.ndarray, nfft: int = NFFT) -> tuple[float, float]:
    """Circular 99%-energy occupied band as ``(center, fractional_bw)``.

    The input is peak-normalized *before* Welch.  Consequently all finite,
    non-zero global gains produce the same PSD, including gains whose squared
    magnitude would underflow an absolute post-PSD threshold.

    Exact zero and spectrally flat/noise-only captures use the explicit
    ``(0, FULL_BAND_BW)`` fallback.  Its length-aware threshold accounts for
    the number of Welch averages and uses normalized spectral shape, never
    absolute energy.
    """
    x = _as_finite_complex_1d(iq)
    if (
        not isinstance(nfft, (int, np.integer))
        or int(nfft) < 2 * SMOOTH + 1
        or int(nfft) % 2 != 0
    ):
        raise ValueError(
            f"nfft must be an even integer >= {2 * SMOOTH + 1}"
        )
    nfft = int(nfft)
    scaled, peak = _peak_normalized(x)
    if peak == 0.0:
        return 0.0, FULL_BAND_BW

    psd = _circular_smooth(pp.welch_psd(scaled, nfft), SMOOTH)
    psd = np.maximum(psd, 0.0)
    psd_total = float(psd.sum())
    if psd_total <= 0.0:
        return 0.0, FULL_BAND_BW

    if _spectral_flatness(psd) >= _white_flatness_threshold(len(x), nfft):
        return 0.0, FULL_BAND_BW

    guard = _minimum_circular_guard(psd)
    floor = GUARD_FLOOR_SCALE * float(np.median(guard))
    excess = np.clip(psd - floor, 0.0, None)
    excess_fraction = float(excess.sum() / psd_total)
    if excess_fraction < MIN_EXCESS_FRACTION:
        return 0.0, FULL_BAND_BW

    start, end = _shortest_circular_interval(
        excess,
        retained_fraction=1.0 - 2.0 * ENERGY_EDGE,
    )
    # Bins represent their center frequencies.  Match production's convention
    # that one retained bin still has a minimum width of one FFT bin.
    span_bins = end - start
    bw = max(span_bins / nfft, 1.0 / nfft)
    center_unwrapped_index = 0.5 * (start + end)
    center = _wrap_cycles(center_unwrapped_index / nfft - 0.5)
    return center, float(min(bw, FULL_BAND_BW))


def _stable_unit_rms(x: np.ndarray) -> np.ndarray:
    """Unit-RMS normalization with no fixed absolute-power epsilon."""
    z = np.asarray(x, dtype=np.complex128)
    scaled, peak = _peak_normalized(z)
    if peak == 0.0:
        return np.zeros(z.shape, dtype=np.complex64)
    rms_scaled = float(np.sqrt(np.mean(np.abs(scaled) ** 2)))
    if rms_scaled == 0.0 or not np.isfinite(rms_scaled):
        raise ValueError("cannot normalize a non-zero capture with invalid RMS")
    return (scaled / rms_scaled).astype(np.complex64)


def preprocess(
    iq: np.ndarray,
    l_out: int = L_OUT,
    target_frac: float = TARGET_FRAC,
    nfft: int = NFFT,
    scale_jitter: float = 0.0,
    rng: np.random.Generator | None = None,
    force_center: float | None = None,
    force_bw: float | None = None,
    force_resample_frac: float | None = None,
    resample: bool = True,
) -> tuple[np.ndarray, dict]:
    """Circular detect/downconvert/(optional resample)/fit/normalize front end.

    With ``resample=True`` (the default), behavior follows the production
    canonical 1024-sample geometry.  ``resample=False`` is available for direct
    native-bandwidth experiments; callers should normally also choose their
    native ``l_out`` explicitly.
    """
    x_in = _as_finite_complex_1d(iq)
    if not isinstance(l_out, (int, np.integer)) or int(l_out) <= 0:
        raise ValueError("l_out must be a positive integer")
    l_out = int(l_out)
    if not np.isfinite(target_frac) or target_frac <= 0.0:
        raise ValueError("target_frac must be finite and positive")
    if scale_jitter < 0.0 or not np.isfinite(scale_jitter):
        raise ValueError("scale_jitter must be finite and non-negative")
    if force_resample_frac is not None and not resample:
        raise ValueError("force_resample_frac is only meaningful when resample=True")

    if force_center is None or force_bw is None:
        detected_center, detected_bw = estimate_band(x_in, nfft)
    else:
        detected_center, detected_bw = 0.0, FULL_BAND_BW

    center = detected_center if force_center is None else float(force_center)
    bw = detected_bw if force_bw is None else float(force_bw)
    if not np.isfinite(center):
        raise ValueError("center must be finite")
    if not np.isfinite(bw) or bw <= 0.0:
        raise ValueError("bandwidth must be finite and positive")
    center = _wrap_cycles(center)

    # Absolute input gain is intentionally discarded before every subsequent
    # operation.  This also protects downconversion and interpolation from
    # avoidable overflow/underflow at extreme but finite gains.
    x, _ = _peak_normalized(x_in)
    sample = np.arange(len(x), dtype=np.float64)
    x = x * np.exp(-2.0j * np.pi * center * sample)

    resample_frac: float | None = None
    if resample:
        resample_frac = bw if force_resample_frac is None else float(force_resample_frac)
        if not np.isfinite(resample_frac) or resample_frac <= 0.0:
            raise ValueError("resample fraction must be finite and positive")
        if force_resample_frac is None and scale_jitter > 0.0 and rng is not None:
            resample_frac *= float(1.0 + rng.uniform(-scale_jitter, scale_jitter))
        resample_frac = float(np.clip(resample_frac, 1e-3, FULL_BAND_BW))
        new_len = int(max(64, round(len(x) * resample_frac / target_frac)))
        x = pp.lin_resample(x, new_len)

    x = pp.center_fit(x, l_out)
    x = _stable_unit_rms(x)
    return x, {
        "center": center,
        "bw": bw,
        "resample_frac": resample_frac,
        "resampled": bool(resample),
    }


to_channels = pp.to_channels
iq_features = pp.iq_features
features_from_channels = pp.features_from_channels
