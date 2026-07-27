"""Generate the hybrid-v3 Python -> TypeScript parity fixture.

This is deliberately separate from ``src/embedding/assets``: those files belong
to models trained with linear-v1 preprocessing and must not be rewritten merely
to make a new front end appear compatible.
"""
from __future__ import annotations

import json
import os

import numpy as np

import magnitude as mag
import preprocess as pp


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(
    ROOT,
    "src",
    "embedding",
    "test-fixtures",
    "iq-preprocess-hybrid-v3.json",
)


def circular_error(a: float, b: float) -> float:
    return abs((a - b + 0.5) % 1.0 - 0.5)


def bandlimited_noise(
    length: int,
    center: float,
    bw: float,
    seed: int,
    snr_db: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    freq = np.fft.fftfreq(length)
    distance = np.abs((freq - center + 0.5) % 1.0 - 0.5)
    occupied = distance <= bw / 2.0
    spectrum = np.zeros(length, dtype=np.complex128)
    count = int(occupied.sum())
    spectrum[occupied] = (
        rng.standard_normal(count) + 1.0j * rng.standard_normal(count)
    )
    signal = np.fft.ifft(spectrum) * np.sqrt(length / count)
    noise_scale = (
        np.sqrt(np.mean(np.abs(signal) ** 2) / 2.0)
        * 10.0 ** (-snr_db / 20.0)
    )
    noise = noise_scale * (
        rng.standard_normal(length) + 1.0j * rng.standard_normal(length)
    )
    return (signal + noise).astype(np.complex64)


def _record(
    name: str,
    iq: np.ndarray,
    *,
    true_center: float | None = None,
    true_bw: float | None = None,
    gains: tuple[float, ...] = (1e-200, 1.0, 1e200),
) -> dict:
    expected, context = pp.preprocess(iq)
    scaled, peak = pp._peak_normalized(pp._as_finite_complex_1d(iq))
    if peak == 0.0:
        linear = (0.0, pp.FULL_BAND_BW)
        circular = (0.0, pp.FULL_BAND_BW)
    else:
        linear = pp.estimate_band_linear_v1(scaled)
        circular = pp.estimate_band_circular_v2(scaled)
    if circular == (0.0, pp.FULL_BAND_BW):
        route = "full-band-fallback"
    elif pp._hybrid_uses_circular(linear, circular, pp.NFFT):
        route = "circular-v2"
    else:
        route = "linear-v1"
    magnitude_shape, magnitude_features = mag.magnitude_from_iq(iq)
    if true_center is not None:
        if circular_error(context["center"], true_center) > 4.0 / pp.NFFT:
            raise AssertionError(f"{name}: inaccurate center {context['center']}")
    if true_bw is not None:
        if abs(context["bw"] - true_bw) > 10.0 / pp.NFFT:
            raise AssertionError(f"{name}: inaccurate bandwidth {context['bw']}")
    return {
        "name": name,
        "input_i": iq.real.astype(float).tolist(),
        "input_q": iq.imag.astype(float).tolist(),
        "gains": list(gains),
        "expected_i": expected.real.astype(float).tolist(),
        "expected_q": expected.imag.astype(float).tolist(),
        "center": float(context["center"]),
        "bw": float(context["bw"]),
        "resample_frac": float(context["resample_frac"]),
        "route": route,
        "linear_center": float(linear[0]),
        "linear_bw": float(linear[1]),
        "circular_center": float(circular[0]),
        "circular_bw": float(circular[1]),
        "magnitude_shape": magnitude_shape.astype(float).tolist(),
        "magnitude_features": magnitude_features.astype(float).tolist(),
        "true_center": true_center,
        "true_bw": true_bw,
    }


def build_fixture() -> dict:
    rng = np.random.default_rng(20260727)
    white = (
        rng.standard_normal(4096) + 1.0j * rng.standard_normal(4096)
    ).astype(np.complex64)
    cases = [
        _record(
            "positive-seam",
            bandlimited_noise(4096, 0.38, 0.30, 101, 20.0),
            true_center=0.38,
            true_bw=0.30,
        ),
        _record(
            "negative-seam",
            bandlimited_noise(4096, -0.40, 0.25, 102, 20.0),
            true_center=-0.40,
            true_bw=0.25,
        ),
        _record(
            "broad-seam-low-snr",
            bandlimited_noise(4096, 0.47, 0.70, 103, 6.0),
            true_center=0.47,
            true_bw=0.70,
        ),
        _record(
            "normal-band",
            bandlimited_noise(4096, 0.20, 0.30, 104, 20.0),
            true_center=0.20,
            true_bw=0.30,
        ),
        _record("white-noise", white),
        _record(
            "exact-zero",
            np.zeros(1024, dtype=np.complex64),
            gains=(1.0,),
        ),
    ]
    return {
        "preprocess_version": pp.PREPROCESS_VERSION,
        "parameters": pp.preprocess_metadata(),
        "cases": cases,
    }


def main() -> None:
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(build_fixture(), f, separators=(",", ":"))
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
