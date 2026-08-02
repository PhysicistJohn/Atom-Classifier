"""Parity gate: numpy port vs Atom-DSP golden vectors.

Vectors come from tools/gen-dsp-channel-vectors.mjs, evaluated against the
pinned Atom-DSP TypeScript. Hashes must match integer-exactly; floats to
1e-12 relative. Also proves split-invariance and the fading fast path's
grid_step=1 exactness.

Run: .venv-training/bin/python -m pytest tools/test_dsp_channel_parity.py -q
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

import dsp_channel as dc

VECTORS = json.loads(
    (Path(__file__).parent / "dsp-channel-conformance-vectors.json")
    .read_text())

REL = 1e-12


def close(actual, expected):
    return math.isclose(actual, expected, rel_tol=REL, abs_tol=1e-15)


def source_at(index):
    angle = 2 * math.pi * 0.03125 * index
    envelope = 0.5 + 0.25 * math.cos(2 * math.pi * index / 1000)
    return envelope * math.cos(angle), envelope * math.sin(angle)


def test_hash_exact():
    for case in VECTORS["hashCases"]:
        got = int(dc.channel_hash32(case["seed"], case["index"], case["lane"]))
        assert got == case["value"], case


def test_hash_vectorized_matches_scalar():
    cases = VECTORS["hashCases"]
    seeds = {c["seed"] for c in cases}
    for seed in seeds:
        sub = [c for c in cases if c["seed"] == seed]
        for c in sub:
            arr = dc.channel_hash32(seed, np.array([c["index"]] * 3), c["lane"])
            assert (arr == c["value"]).all()


def test_gaussian():
    for case in VECTORS["gaussianCases"]:
        real, imaginary = dc.seeded_complex_gaussian(
            case["seed"], case["sample"], case["lane"])
        assert close(float(real), case["value"][0]), case
        assert close(float(imaginary), case["value"][1]), case


def test_tdl_resolution():
    for case in VECTORS["tdlCases"]:
        got = dc.resolve_tdl_taps(case["profile"],
                                  case["delaySpreadSeconds"],
                                  case["sampleRateHz"])
        expected = case["taps"]
        assert len(got) == len(expected), case
        for g, e in zip(got, expected):
            assert g["delaySamples"] == e["delaySamples"], case
            assert close(g["powerDb"], e["powerDb"]), case


def test_fading():
    for case in VECTORS["fadingCases"]:
        configuration = dc.FadingConfiguration(
            kind=case["kind"], doppler_hz=case["dopplerHz"],
            k_factor_db=case["kFactorDb"])
        real, imaginary = dc.fading_gain_at_time(
            configuration, case["seed"], case["tapIndex"],
            case["timeSeconds"])
        assert close(float(real), case["value"][0]), case
        assert close(float(imaginary), case["value"][1]), case


def test_noise_deviation():
    for case in VECTORS["noiseCases"]:
        got = dc.noise_standard_deviation(case["noiseFloorDbm"],
                                          case["fullScaleDbm"])
        assert close(got, case["value"]), case


def _configuration_from(case) -> dc.ChannelConfiguration:
    raw = case["configuration"]
    fading = None
    if "fading" in raw:
        fading = dc.FadingConfiguration(
            kind=raw["fading"]["kind"],
            doppler_hz=raw["fading"]["dopplerHz"],
            k_factor_db=raw["fading"].get("kFactorDb"))
    return dc.ChannelConfiguration(
        noise_floor_dbm=raw["noiseFloorDbm"],
        seed=raw["seed"], sample_rate_hz=raw["sampleRateHz"],
        full_scale_dbm=raw.get("fullScaleDbm", 0.0),
        taps=tuple(raw["taps"]) if "taps" in raw
        else ({"delaySamples": 0, "powerDb": 0.0},),
        fading=fading)


def test_end_to_end():
    for case in VECTORS["endToEnd"]:
        configuration = _configuration_from(case)
        for index, expected in zip(case["indices"], case["values"]):
            real, imaginary = dc.apply_channel_at_index(
                source_at, index, configuration)
            assert close(real, expected[0]), (case["name"], index)
            assert close(imaginary, expected[1]), (case["name"], index)


def test_row_path_matches_scalar_reference():
    case = next(c for c in VECTORS["endToEnd"] if c["name"] == "rayleigh-fading")
    configuration = _configuration_from(case)
    n = 4096
    row = np.array([complex(*source_at(i)) for i in range(n)])
    got = dc.apply_channel_to_row(row, configuration, fading_grid_step=1)
    for index in (0, 1, 17, 1000, 4095):
        expected = dc.apply_channel_at_index(source_at, index, configuration)
        assert close(got[index].real, expected[0]), index
        assert close(got[index].imag, expected[1]), index


def test_fading_grid_interpolation_error_is_small():
    configuration = dc.FadingConfiguration(kind="rayleigh", doppler_hz=200.0)
    fs = 20e6
    exact = dc.fading_gain_grid(configuration, 42, 0, 0, 200_000, fs, 1)
    fast = dc.fading_gain_grid(configuration, 42, 0, 0, 200_000, fs, 1024)
    # 1024 samples = 51.2 us; at 200 Hz Doppler that is ~1% of a fading
    # cycle, so linear interpolation must track closely.
    assert np.max(np.abs(exact - fast)) < 1e-3


def test_fading_grid_split_invariance():
    configuration = dc.FadingConfiguration(kind="rayleigh", doppler_hz=120.0)
    fs = 20e6
    whole = dc.fading_gain_grid(configuration, 7, 2, 0, 8192, fs, 256)
    part = dc.fading_gain_grid(configuration, 7, 2, 3000, 2000, fs, 256)
    assert np.allclose(whole[3000:5000], part, rtol=0, atol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
