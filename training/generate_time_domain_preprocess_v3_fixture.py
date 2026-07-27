"""Generate synthetic Python -> TypeScript parity evidence for the v3 frontend.

The inputs are analytic deterministic waveforms created in this file. No
training, selection, historical test, or sealed release payload is read.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUTPUT = (
    ROOT
    / "src"
    / "embedding"
    / "test-fixtures"
    / "time-domain-invariant-preprocess-v3.json"
)

import time_domain_geometry as geometry  # noqa: E402
import time_domain_invariant_patch_preprocess as preprocess  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _analytic_capture(
    length: int,
    *,
    center: float,
    width: float,
    physical_scale: float = 1.0,
) -> np.ndarray:
    """Deterministic multitone with scaled AM/PM texture and no transform."""
    sample = np.arange(length, dtype=np.float64)
    offsets = np.array(
        (-0.47, -0.29, -0.11, 0.03, 0.21, 0.44),
        dtype=np.float64,
    )
    amplitudes = np.array(
        (0.71, 0.43, 0.92, 0.58, 0.36, 0.67),
        dtype=np.float64,
    )
    phases = np.array(
        (0.19, -1.07, 0.73, 2.01, -0.41, 1.37),
        dtype=np.float64,
    )
    frequencies = physical_scale * (center + width * offsets)
    carrier = np.sum(
        amplitudes[:, None]
        * np.exp(
            1j
            * (
                2.0 * np.pi * frequencies[:, None] * sample[None, :]
                + phases[:, None]
            )
        ),
        axis=0,
    )
    envelope = (
        0.83
        + 0.13
        * np.cos(
            2.0 * np.pi * physical_scale * 0.0027 * sample + 0.31
        )
        + 0.04
        * np.sin(
            2.0 * np.pi * physical_scale * 0.0061 * sample - 0.57
        )
    )
    phase_texture = np.exp(
        1j
        * (
            0.09
            * np.sin(
                2.0 * np.pi * physical_scale * 0.0019 * sample + 0.17
            )
        )
    )
    return np.asarray(envelope * carrier * phase_texture, dtype=np.complex128)


def _variant(
    iq: np.ndarray,
    *,
    gain: float = 1.0,
    phase: float = 0.0,
    shift: float = 0.0,
) -> np.ndarray:
    sample = np.arange(len(iq), dtype=np.float64)
    return np.asarray(
        gain
        * np.exp(1j * phase)
        * iq
        * np.exp(1j * 2.0 * np.pi * shift * sample),
        dtype=np.complex128,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _record(
    name: str,
    iq: np.ndarray,
    *,
    physical_scale: float | None = None,
    relation: str | None = None,
    nuisance: dict[str, float] | None = None,
    patch_length: int = preprocess.DEFAULT_PATCH_LENGTH,
    patch_count: int = preprocess.DEFAULT_PATCH_COUNT,
    target_frac: float = 0.5,
) -> dict[str, Any]:
    channels, features, context = preprocess.preprocess(
        iq,
        patch_length=patch_length,
        patch_count=patch_count,
        target_frac=target_frac,
    )
    direct_center, direct_bandwidth, direct_context = geometry.estimate_band(
        iq,
        min_bandwidth=geometry.minimum_bandwidth_for_active_span(
            len(iq),
            patch_length,
            target_frac,
        ),
    )
    if direct_center != context["center"] or direct_bandwidth != context["bw"]:
        raise AssertionError(f"{name}: wrapper geometry does not match direct geometry")
    if direct_context != context["geometry_estimator"]:
        raise AssertionError(f"{name}: wrapper audit context does not match geometry")
    return {
        "name": name,
        "physical_scale": physical_scale,
        "relation": relation,
        "nuisance": nuisance or {},
        "options": {
            "patch_length": patch_length,
            "patch_count": patch_count,
            "target_frac": target_frac,
        },
        "input": {
            "in_phase": iq.real.astype(float).tolist(),
            "quadrature": iq.imag.astype(float).tolist(),
        },
        "expected": {
            "center": direct_center,
            "bandwidth": direct_bandwidth,
            "geometry_context": _jsonable(direct_context),
            "packed_in_phase": channels[0].astype(float).tolist(),
            "packed_quadrature": channels[1].astype(float).tolist(),
            "raw_features": features.astype(float).tolist(),
            "context": _jsonable(context),
        },
    }


def build_fixture() -> dict[str, Any]:
    normal = _analytic_capture(1024, center=0.08125, width=0.071)
    scale_rows = [
        (
            scale,
            _analytic_capture(
                int(round(1024 / scale)),
                center=0.078125,
                width=0.0625,
                physical_scale=scale,
            ),
        )
        for scale in (0.5, 1.0, 2.0)
    ]
    impulse = np.zeros(65, dtype=np.complex128)
    impulse[0] = 1.0 + 0.0j

    cases = [
        _record("normal-base", normal, relation="normal"),
        _record(
            "gain-tiny-phase",
            _variant(normal, gain=1e-150, phase=0.731),
            relation="normal",
            nuisance={"gain": 1e-150, "global_phase": 0.731},
        ),
        _record(
            "gain-huge-phase",
            _variant(normal, gain=1e150, phase=-1.171),
            relation="normal",
            nuisance={"gain": 1e150, "global_phase": -1.171},
        ),
        _record(
            "carrier-shift",
            _variant(normal, shift=0.123),
            relation="normal-shift-0.123",
            nuisance={"carrier_shift": 0.123},
        ),
        *[
            _record(
                f"physical-scale-{str(scale).replace('.', 'p')}",
                row,
                physical_scale=scale,
                relation="physical-scale",
            )
            for scale, row in scale_rows
        ],
        _record(
            "positive-nyquist-seam",
            _analytic_capture(777, center=0.487, width=0.083),
            relation="seam",
        ),
        _record(
            "negative-nyquist-seam",
            _analytic_capture(777, center=-0.489, width=0.079),
            relation="seam",
        ),
        _record(
            "short-three",
            _analytic_capture(3, center=0.071, width=0.17),
            relation="short-boundary",
        ),
        _record(
            "short-thirty-three",
            _analytic_capture(33, center=-0.113, width=0.11),
            relation="short-boundary",
        ),
        _record(
            "short-thirty-four",
            _analytic_capture(34, center=0.193, width=0.13),
            relation="short-boundary",
        ),
        _record(
            "short-thirty-five",
            _analytic_capture(35, center=-0.217, width=0.09),
            relation="short-boundary",
        ),
        _record(
            "odd-length-51",
            _analytic_capture(51, center=0.131, width=0.12),
            relation="odd-length",
        ),
        _record(
            "odd-length-1733",
            _analytic_capture(1733, center=-0.137, width=0.043),
            relation="odd-length",
        ),
        _record(
            "narrow-long",
            _analytic_capture(4096, center=0.037, width=0.012),
            relation="high-base-lag",
        ),
        _record(
            "lag-one-degenerate-impulse",
            impulse,
            relation="full-band-fallback",
        ),
        _record(
            "custom-patch-contract",
            _analytic_capture(127, center=0.211, width=0.091),
            relation="custom-options",
            patch_length=31,
            patch_count=3,
            target_frac=0.4,
        ),
    ]
    source_paths = {
        "geometry": HERE / "time_domain_geometry.py",
        "wrapper": HERE / "time_domain_invariant_patch_preprocess.py",
        "patch_core": HERE / "invariant_patch_preprocess.py",
        "feature_core": HERE / "preprocess.py",
        "generator": Path(__file__).resolve(),
    }
    return {
        "fixture_version": "time-domain-invariant-preprocess-parity-v3",
        "synthetic_only": True,
        "reads_held_or_sealed_payload": False,
        "python_source_sha256": {
            name: _sha256(path) for name, path in source_paths.items()
        },
        "metadata": preprocess.preprocess_metadata(),
        "cases": cases,
    }


def main() -> None:
    fixture = build_fixture()
    os.makedirs(OUTPUT.parent, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as destination:
        json.dump(fixture, destination, separators=(",", ":"), allow_nan=False)
        destination.write("\n")
    print(f"wrote {OUTPUT}")
    print(f"sha256={_sha256(OUTPUT)}")
    print(f"cases={len(fixture['cases'])}")


if __name__ == "__main__":
    main()
