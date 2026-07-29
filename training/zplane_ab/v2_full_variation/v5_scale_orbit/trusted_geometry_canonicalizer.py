"""Trusted-geometry physical-time canonicalization for the v5 current route.

This component deliberately does one small job.  Given the first 4,096
complex samples of a trusted current acquisition and its measured/native
sample rates, it evaluates that acquisition on the fixed physical-time grid

    t[m] = m / (2 * native_sample_rate_hz),  m = 0, ..., 4095.

The corresponding fractional input index is

    p[m] = m * sample_rate_hz / (2 * native_sample_rate_hz).

The supported input-rate ratio is closed and explicit: ``1 <= Fs/Fnative <=
2``.  Therefore every requested position lies in the causal first 4,096 input
samples.  Extra tail samples are ignored by construction.

Interpolation uses a six-tap quintic Lagrange finite impulse response.  It is a
small algebraic kernel with no FFT, spectral grid, learned state, selected
profile, or class input, and has a direct browser implementation.
"""
from __future__ import annotations

from typing import Any

import numpy as np


CANONICALIZER_VERSION = "trusted-current-physical-time-v1"
TRUST_SCOPE = "trusted-current"
CONSUMED_INPUT_SAMPLES = 4096
OUTPUT_SAMPLES = 4096
TARGET_NATIVE_RATE_MULTIPLIER = 2.0
MIN_SAMPLE_RATE_RATIO = 1.0
MAX_SAMPLE_RATE_RATIO = 2.0
INTERPOLATION = "six-tap quintic Lagrange FIR"
USES_FREQUENCY_TRANSFORM = False
USES_SELECTED_PROFILE_OR_CLASS = False


def _finite_positive_rate(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _validate_capture(iq: Any) -> np.ndarray:
    value = np.asarray(iq)
    if value.ndim != 1 or len(value) < CONSUMED_INPUT_SAMPLES:
        raise ValueError(
            "trusted-current canonicalization requires a complex 1-D "
            f"capture with at least {CONSUMED_INPUT_SAMPLES} samples"
        )
    if not np.iscomplexobj(value):
        raise TypeError("trusted-current canonicalization requires complex I/Q")
    raw_prefix = value[:CONSUMED_INPUT_SAMPLES]
    if not np.all(np.isfinite(raw_prefix.real)) or not np.all(
        np.isfinite(raw_prefix.imag)
    ):
        raise ValueError("causal I/Q prefix contains NaN or infinity")
    with np.errstate(over="ignore", invalid="ignore"):
        prefix = np.asarray(raw_prefix, dtype=np.complex64)
    if not np.all(np.isfinite(prefix.real)) or not np.all(
        np.isfinite(prefix.imag)
    ):
        raise ValueError("causal I/Q prefix exceeds float32 range")
    return prefix


def _rate_contract(
    sample_rate_hz: Any,
    native_sample_rate_hz: Any,
) -> tuple[float, float, float, float]:
    sample_rate = _finite_positive_rate(sample_rate_hz, "sample_rate_hz")
    native_sample_rate = _finite_positive_rate(
        native_sample_rate_hz,
        "native_sample_rate_hz",
    )
    ratio = sample_rate / native_sample_rate
    if (
        not np.isfinite(ratio)
        or ratio < MIN_SAMPLE_RATE_RATIO
        or ratio > MAX_SAMPLE_RATE_RATIO
    ):
        raise ValueError(
            "sample_rate_hz/native_sample_rate_hz must lie in [1, 2]"
        )
    target_sample_rate = TARGET_NATIVE_RATE_MULTIPLIER * native_sample_rate
    if not np.isfinite(target_sample_rate):
        raise ValueError("2 * native_sample_rate_hz must be finite")
    input_step = ratio / TARGET_NATIVE_RATE_MULTIPLIER
    return sample_rate, native_sample_rate, ratio, input_step


def _lagrange6(
    x: np.ndarray,
    position: float,
) -> np.complex64:
    """Evaluate six adjacent samples at one in-bounds fractional position."""
    integer = int(np.floor(position))
    fraction = position - integer
    if fraction == 0.0:
        return np.complex64(x[integer])

    # Use a centred six-point stencil in the interior.  At either edge, use
    # the first/last six available causal-prefix samples.  The same generic
    # Lagrange weights apply to nodes q={0,1,2,3,4,5}.
    if integer < 2:
        start = 0
    elif integer + 3 >= CONSUMED_INPUT_SAMPLES:
        start = CONSUMED_INPUT_SAMPLES - 6
    else:
        start = integer - 2
    q = position - start
    q_minus_1 = q - 1.0
    q_minus_2 = q - 2.0
    q_minus_3 = q - 3.0
    q_minus_4 = q - 4.0
    q_minus_5 = q - 5.0
    w0 = -(
        q_minus_1
        * q_minus_2
        * q_minus_3
        * q_minus_4
        * q_minus_5
    ) / 120.0
    w1 = (
        q * q_minus_2 * q_minus_3 * q_minus_4 * q_minus_5
    ) / 24.0
    w2 = -(
        q * q_minus_1 * q_minus_3 * q_minus_4 * q_minus_5
    ) / 12.0
    w3 = (
        q * q_minus_1 * q_minus_2 * q_minus_4 * q_minus_5
    ) / 12.0
    w4 = -(
        q * q_minus_1 * q_minus_2 * q_minus_3 * q_minus_5
    ) / 24.0
    w5 = (
        q * q_minus_1 * q_minus_2 * q_minus_3 * q_minus_4
    ) / 120.0

    x0 = x[start]
    x1 = x[start + 1]
    x2 = x[start + 2]
    x3 = x[start + 3]
    x4 = x[start + 4]
    x5 = x[start + 5]
    real = (
        w0 * float(x0.real)
        + w1 * float(x1.real)
        + w2 * float(x2.real)
        + w3 * float(x3.real)
        + w4 * float(x4.real)
        + w5 * float(x5.real)
    )
    imaginary = (
        w0 * float(x0.imag)
        + w1 * float(x1.imag)
        + w2 * float(x2.imag)
        + w3 * float(x3.imag)
        + w4 * float(x4.imag)
        + w5 * float(x5.imag)
    )
    return np.complex64(complex(real, imaginary))


def canonicalize(
    iq: Any,
    *,
    sample_rate_hz: Any,
    native_sample_rate_hz: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the fixed 4,096-sample physical-time window and audit context.

    Only the causal first :data:`CONSUMED_INPUT_SAMPLES` samples are validated
    or read.  The returned array is one-dimensional ``complex64`` and owns its
    storage.
    """
    prefix = _validate_capture(iq)
    (
        sample_rate,
        native_sample_rate,
        ratio,
        input_step,
    ) = _rate_contract(sample_rate_hz, native_sample_rate_hz)

    maximum_input_position = (OUTPUT_SAMPLES - 1) * input_step
    if (
        not np.isfinite(maximum_input_position)
        or maximum_input_position < 0.0
        or maximum_input_position > CONSUMED_INPUT_SAMPLES - 1
    ):
        raise AssertionError("validated rate contract escaped the causal prefix")

    output = np.empty(OUTPUT_SAMPLES, dtype=np.complex64)
    for output_index in range(OUTPUT_SAMPLES):
        position = output_index * input_step
        output[output_index] = _lagrange6(prefix, position)
    if not np.all(np.isfinite(output.real)) or not np.all(
        np.isfinite(output.imag)
    ):
        raise AssertionError("canonical interpolation produced a non-finite value")

    context: dict[str, Any] = {
        "version": CANONICALIZER_VERSION,
        "trust_scope": TRUST_SCOPE,
        "input_length": int(len(np.asarray(iq))),
        "consumed_input_samples": CONSUMED_INPUT_SAMPLES,
        "ignored_tail_samples": int(len(np.asarray(iq)) - CONSUMED_INPUT_SAMPLES),
        "output_samples": OUTPUT_SAMPLES,
        "sample_rate_hz": sample_rate,
        "native_sample_rate_hz": native_sample_rate,
        "sample_rate_ratio": ratio,
        "minimum_sample_rate_ratio": MIN_SAMPLE_RATE_RATIO,
        "maximum_sample_rate_ratio": MAX_SAMPLE_RATE_RATIO,
        "target_native_rate_multiplier": TARGET_NATIVE_RATE_MULTIPLIER,
        "target_sample_rate_hz": (
            TARGET_NATIVE_RATE_MULTIPLIER * native_sample_rate
        ),
        "input_samples_per_output": input_step,
        "maximum_input_position": maximum_input_position,
        "input_rounding": "IEEE-754 float32 before interpolation",
        "interpolation": INTERPOLATION,
        "interpolation_taps": 6,
        "uses_frequency_transform": USES_FREQUENCY_TRANSFORM,
        "uses_selected_profile_or_class": USES_SELECTED_PROFILE_OR_CLASS,
    }
    return output, context


def canonicalizer_metadata() -> dict[str, Any]:
    """Return the stable, JSON-serializable v5 component contract."""
    return {
        "version": CANONICALIZER_VERSION,
        "trust_scope": TRUST_SCOPE,
        "consumed_input_samples": CONSUMED_INPUT_SAMPLES,
        "output_samples": OUTPUT_SAMPLES,
        "target_native_rate_multiplier": TARGET_NATIVE_RATE_MULTIPLIER,
        "minimum_sample_rate_ratio": MIN_SAMPLE_RATE_RATIO,
        "maximum_sample_rate_ratio": MAX_SAMPLE_RATE_RATIO,
        "physical_time_grid": "m / (2 * native_sample_rate_hz), m=0..4095",
        "input_position_grid": (
            "m * sample_rate_hz / (2 * native_sample_rate_hz), m=0..4095"
        ),
        "input_rounding": "IEEE-754 float32 before interpolation",
        "interpolation": INTERPOLATION,
        "interpolation_taps": 6,
        "boundary_rule": (
            "centred six-point stencil; first/last six prefix samples at edges"
        ),
        "output_dtype": "complex64",
        "uses_frequency_transform": USES_FREQUENCY_TRANSFORM,
        "uses_selected_profile_or_class": USES_SELECTED_PROFILE_OR_CLASS,
    }
