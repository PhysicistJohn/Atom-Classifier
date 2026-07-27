"""Build the compact synthetic fixture used by the strict v3 browser audit.

Only analytic waveforms defined below are read. The generator refuses any
external input path and never accesses training, held, or sealed corpora.
"""
from __future__ import annotations

import base64
import hashlib
import json
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
    / "time-domain-v3-browser-parity.json"
)
LENGTHS = (4096, 8192, 16384, 32768)

import time_domain_geometry as geometry  # noqa: E402
import time_domain_invariant_patch_preprocess as preprocess  # noqa: E402


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _analytic_capture(length: int) -> np.ndarray:
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
    frequencies = 0.078125 + 0.0625 * offsets
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
        + 0.13 * np.cos(2.0 * np.pi * 0.0027 * sample + 0.31)
        + 0.04 * np.sin(2.0 * np.pi * 0.0061 * sample - 0.57)
    )
    phase_texture = np.exp(
        1j
        * 0.09
        * np.sin(2.0 * np.pi * 0.0019 * sample + 0.17)
    )
    return np.asarray(envelope * carrier * phase_texture, dtype=np.complex128)


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


def _binary(value: np.ndarray, dtype: str) -> dict[str, Any]:
    array = np.ascontiguousarray(value, dtype=np.dtype(dtype))
    payload = array.tobytes(order="C")
    return {
        "dtype": dtype,
        "shape": list(array.shape),
        "base64": base64.b64encode(payload).decode("ascii"),
        "sha256": _sha256_bytes(payload),
        "byte_length": len(payload),
    }


def _case(length: int) -> dict[str, Any]:
    capture = _analytic_capture(length)
    channels, features, context = preprocess.preprocess(capture)
    raw = np.stack((capture.real, capture.imag), axis=0)
    packed = np.asarray(channels, dtype=np.float32)
    return {
        "name": f"synthetic-n{length}",
        "length": length,
        "input_iq": _binary(raw, "<f8"),
        "expected_packed_iq": _binary(packed, "<f4"),
        "expected_raw_features": _binary(features, "<f4"),
        "expected_context": _jsonable(context),
    }


def build_fixture() -> dict[str, Any]:
    sources = {
        "time_domain_geometry.py": HERE / "time_domain_geometry.py",
        "time_domain_invariant_patch_preprocess.py": (
            HERE / "time_domain_invariant_patch_preprocess.py"
        ),
        "invariant_patch_preprocess.py": HERE / "invariant_patch_preprocess.py",
        "preprocess.py": HERE / "preprocess.py",
        "generate_time_domain_v3_browser_fixture.py": Path(__file__).resolve(),
    }
    return {
        "fixture_version": "time-domain-v3-browser-parity-v1",
        "frontend_version": preprocess.PREPROCESS_VERSION,
        "geometry_version": geometry.ESTIMATOR_VERSION,
        "synthetic_only": True,
        "reads_held_or_sealed_payload": False,
        "input_method": "analytic time-domain multitone; no corpus input",
        "uses_frequency_transform": False,
        "lengths": list(LENGTHS),
        "python_source_sha256": {
            name: _sha256_file(path) for name, path in sources.items()
        },
        "cases": [_case(length) for length in LENGTHS],
    }


def main() -> None:
    fixture = build_fixture()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(fixture, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")
    OUTPUT.write_bytes(serialized)
    print(f"wrote {OUTPUT}")
    print(f"sha256={_sha256_bytes(serialized)}")
    print(f"cases={len(fixture['cases'])}")


if __name__ == "__main__":
    main()
