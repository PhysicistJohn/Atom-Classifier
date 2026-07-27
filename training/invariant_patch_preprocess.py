"""Experimental invariant patch front-end, isolated from production preprocessing.

The production front-end centre-fits a variable-duration normalized signal into a fixed
window.  That makes signal fill a proxy for the raw capture length.  This module instead
keeps the complete active sequence and samples fixed-length patches from a dimensionless
time coordinate:

    u = n * resample_frac / TARGET_FRAC

One step in ``u`` therefore spans the same occupied-band arc regardless of raw sample
rate/bandwidth geometry.  There is no centre crop, zero padding, or FFT after hybrid-v3
band estimation.

This is a new experimental contract.  It deliberately does not modify or masquerade as
``preprocess.py`` and has no TypeScript production counterpart.
"""
from __future__ import annotations

from typing import Any

import numpy as np

import preprocess as pp


PREPROCESS_VERSION = "invariant-patch-v1"
ESTIMATOR_VERSION = "hybrid-v3"
DEFAULT_PATCH_LENGTH = 64
DEFAULT_PATCH_COUNT = 16
MAX_ACTIVE_LENGTH = 10_000_000
MAX_PACKED_LENGTH = 10_000_000


def _positive_integer(value: Any, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_nfft(value: Any) -> int:
    nfft = _positive_integer(value, "nfft")
    minimum = 2 * int(pp.SMOOTH) + 1
    if nfft < minimum or nfft % 2 != 0 or nfft & (nfft - 1):
        raise ValueError(f"nfft must be a power of two >= {minimum}")
    return nfft


def _finite_fraction(value: Any, name: str) -> float:
    try:
        fraction = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite fraction in (0, 1]") from exc
    if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError(f"{name} must be a finite fraction in (0, 1]")
    return fraction


def _finite_center(value: Any) -> float:
    try:
        center = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("center must be finite") from exc
    if not np.isfinite(center):
        raise ValueError("center must be finite")
    return float((center + 0.5) % 1.0 - 0.5)


def _capture(iq: Any) -> np.ndarray:
    x = np.asarray(iq)
    if x.ndim != 1 or len(x) < 2:
        raise ValueError(
            f"expected a complex 1-D I/Q capture with at least two samples, got {x.shape}"
        )
    if not np.iscomplexobj(x):
        raise TypeError(f"expected complex I/Q, got dtype={x.dtype}")
    if not np.all(np.isfinite(x.real)) or not np.all(np.isfinite(x.imag)):
        raise ValueError("I/Q capture contains NaN or infinity")
    return x.astype(np.complex128, copy=False)


def _peak_normalize(x: np.ndarray) -> tuple[np.ndarray, float]:
    # np.abs(complex128) uses a hypot-like stable magnitude implementation.
    peak = float(np.max(np.abs(x)))
    if not np.isfinite(peak):
        raise ValueError("I/Q peak magnitude is not finite")
    if peak == 0.0:
        raise ValueError("I/Q capture has zero magnitude")
    return x / peak, peak


def _resample_on_u(
    x: np.ndarray,
    resample_frac: float,
    target_frac: float,
) -> tuple[np.ndarray, float]:
    """Interpolate samples located at ``u=n*resample_frac/target_frac`` onto integer u."""
    u_step = resample_frac / target_frac
    u_max = (len(x) - 1) * u_step
    if not np.isfinite(u_max):
        raise ValueError("dimensionless active-sequence extent is not finite")

    # nextafter prevents an exact integer endpoint computed one ulp low from disappearing,
    # while never admitting a genuinely out-of-support sample.
    last_u = int(np.floor(np.nextafter(u_max, np.inf)))
    active_length = last_u + 1
    if active_length < 2:
        raise ValueError(
            "normalized active sequence has fewer than two samples; "
            "increase capture length or resample fraction"
        )
    if active_length > MAX_ACTIVE_LENGTH:
        raise ValueError(
            f"normalized active sequence length {active_length} exceeds "
            f"safety limit {MAX_ACTIVE_LENGTH}"
        )

    u = np.arange(active_length, dtype=np.float64)
    positions = u / u_step
    # The last point can exceed the endpoint only by the one-ulp tolerance above.
    positions = np.clip(positions, 0.0, len(x) - 1.0)
    lower = np.floor(positions).astype(np.int64)
    upper = np.minimum(lower + 1, len(x) - 1)
    fraction = positions - lower
    active = x[lower] * (1.0 - fraction) + x[upper] * fraction
    return np.asarray(active, dtype=np.complex128), float(u_max)


def _rms_normalize(x: np.ndarray) -> tuple[np.ndarray, float]:
    power = float(np.mean(x.real * x.real + x.imag * x.imag))
    if not np.isfinite(power) or power <= 0.0:
        raise ValueError("resampled active sequence has invalid or zero power")
    rms = float(np.sqrt(power))
    normalized = x / rms
    if not np.all(np.isfinite(normalized.real)) or not np.all(
        np.isfinite(normalized.imag)
    ):
        raise ValueError("active-sequence RMS normalization produced non-finite values")
    return normalized, rms


def _even_patch_starts(active_length: int, patch_length: int, patch_count: int) -> np.ndarray:
    maximum = active_length - patch_length
    if maximum < 0:
        return np.zeros(patch_count, dtype=np.int64)
    if patch_count == 1:
        return np.zeros(1, dtype=np.int64)
    # Integer half-up rounding of i*maximum/(patch_count-1): deterministic across NumPy
    # versions, includes both endpoints, and avoids float linspace rounding.
    numerator = np.arange(patch_count, dtype=np.int64) * maximum
    return (2 * numerator + (patch_count - 1)) // (2 * (patch_count - 1))


def _extract_patches(
    active: np.ndarray,
    patch_length: int,
    patch_count: int,
) -> tuple[np.ndarray, np.ndarray, bool, int]:
    active_length = len(active)
    starts = _even_patch_starts(active_length, patch_length, patch_count)
    if active_length < patch_length:
        repeat_count = int(np.ceil(patch_length / active_length))
        one = np.tile(active, repeat_count)[:patch_length]
        patches = np.repeat(one[None, :], patch_count, axis=0)
        return patches, starts, True, repeat_count

    patches = np.stack(
        [active[int(start) : int(start) + patch_length] for start in starts]
    )
    return patches, starts, False, 1


def preprocess(
    iq: Any,
    patch_length: int = DEFAULT_PATCH_LENGTH,
    patch_count: int = DEFAULT_PATCH_COUNT,
    target_frac: float = pp.TARGET_FRAC,
    nfft: int = pp.NFFT,
    force_center: float | None = None,
    force_bw: float | None = None,
    force_resample_frac: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return fixed packed patches, active-sequence features, and audit context.

    Parameters mirror the geometry overrides in :func:`preprocess.preprocess`. When both
    ``force_center`` and ``force_bw`` are supplied, hybrid-v3 detection is bypassed; a
    partial override detects both values and replaces only the supplied component.

    Returns
    -------
    packed_channels
        Float32 array shaped ``[2, patch_count * patch_length]``. Patches are flattened in
        start order, with no separators or inserted zeros.
    raw_features
        The production 12-element ``iq_features`` vector computed from the complete
        RMS-normalized active sequence, before patch extraction or repetition.
    context
        JSON-serializable geometry and extraction metadata.
    """
    if pp.PREPROCESS_VERSION != ESTIMATOR_VERSION:
        raise RuntimeError(
            f"invariant patch v1 requires {ESTIMATOR_VERSION}, "
            f"but preprocess.py reports {pp.PREPROCESS_VERSION!r}"
        )
    x_in = _capture(iq)
    patch_length = _positive_integer(patch_length, "patch_length")
    patch_count = _positive_integer(patch_count, "patch_count")
    packed_length = patch_length * patch_count
    if packed_length > MAX_PACKED_LENGTH:
        raise ValueError(
            f"packed patch length {packed_length} exceeds safety limit "
            f"{MAX_PACKED_LENGTH}"
        )
    target_frac = _finite_fraction(target_frac, "target_frac")
    nfft = _validate_nfft(nfft)

    detected_center: float | None
    detected_bw: float | None
    if force_center is not None and force_bw is not None:
        detected_center = None
        detected_bw = None
    else:
        detected_center, detected_bw = pp.estimate_band(x_in, nfft)

    center = (
        _finite_center(detected_center)
        if force_center is None
        else _finite_center(force_center)
    )
    bw = (
        _finite_fraction(detected_bw, "bandwidth")
        if force_bw is None
        else _finite_fraction(force_bw, "bandwidth")
    )
    resample_frac = (
        bw
        if force_resample_frac is None
        else _finite_fraction(force_resample_frac, "resample_frac")
    )

    normalized_peak, raw_peak = _peak_normalize(x_in)
    sample_index = np.arange(len(normalized_peak), dtype=np.float64)
    baseband = normalized_peak * np.exp(
        -1j * 2.0 * np.pi * center * sample_index
    )
    active, captured_u_end = _resample_on_u(
        baseband,
        resample_frac=resample_frac,
        target_frac=target_frac,
    )
    active, active_rms = _rms_normalize(active)

    raw_features = pp.iq_features(active)
    if raw_features.shape != (pp.N_FEATURES,) or not np.all(
        np.isfinite(raw_features)
    ):
        raise ValueError("active-sequence iq_features are malformed or non-finite")
    raw_features = np.asarray(raw_features, dtype=np.float32)

    patches, starts, repeated, repeat_count = _extract_patches(
        active,
        patch_length=patch_length,
        patch_count=patch_count,
    )
    packed_channels = np.stack(
        (patches.real.reshape(-1), patches.imag.reshape(-1)),
        axis=0,
    ).astype(np.float32)
    expected_shape = (2, patch_count * patch_length)
    if packed_channels.shape != expected_shape or not np.all(
        np.isfinite(packed_channels)
    ):
        raise AssertionError(
            f"packed patch contract violated: got {packed_channels.shape}, "
            f"expected {expected_shape}"
        )

    context: dict[str, Any] = {
        "version": PREPROCESS_VERSION,
        "estimator_version": ESTIMATOR_VERSION,
        "raw_length": int(len(x_in)),
        "raw_peak": raw_peak,
        "detected_center": detected_center,
        "detected_bw": detected_bw,
        "center": center,
        "bw": bw,
        "resample_frac": resample_frac,
        "target_frac": target_frac,
        "nfft": nfft,
        "u_step": resample_frac / target_frac,
        "captured_u_end": captured_u_end,
        "active_u_end": int(len(active) - 1),
        "active_length": int(len(active)),
        "active_rms_before_normalization": active_rms,
        "patch_length": patch_length,
        "patch_count": patch_count,
        "patch_starts": starts.tolist(),
        "patches_repeated_from_short_active": repeated,
        "active_repeat_count": repeat_count,
        "padding_samples": 0,
        "packed_length": int(packed_length),
        "feature_source": "complete RMS-normalized active sequence before patch repetition",
    }
    return packed_channels, raw_features, context
