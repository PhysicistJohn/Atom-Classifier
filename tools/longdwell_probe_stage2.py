"""Long-dwell probe corpus, stage 2: common-rate clean/noisy pairs.

Takes stage 1's native-rate clean captures and produces the training corpus:

  1. polyphase resample every row to a common 20 Msps (rational up/down from
     each native rate; scipy.signal.resample_poly, kaiser window) -- this is
     the sanctioned post-synthesis resampler that SignalLab's exact-replay
     firewall deliberately refuses to host;
  2. trim to a fixed common length (DURATION_MS at 20 Msps);
  3. apply a seeded receiver impairment chain to make the NOISY twin:
     AWGN at U(0,30) dB SNR, carrier offset U(-2,2) ppm, Wiener phase noise,
     IQ gain/phase imbalance, DC offsets, 0-3 tap static multipath --
     mirroring the historical corpus nuisance spec;
  4. write clean.f32 + noisy.f32 row-aligned, and manifest.json with the
     drawn impairment parameters and a train/eval split by row.

The CLEAN side is resampled but unimpaired: it is the denoising target.
"""
from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
CORPUS = Path(
    __import__("os").environ.get(
        "CORPUS_DIR",
        HERE.parent / "training/artifacts/longdwell-probe-corpus",
    )
)
TARGET_FS = 20_000_000
SEED = int(__import__("os").environ.get("IMPAIR_SEED", 20260731))
EVAL_PER_PROFILE = int(__import__("os").environ.get("EVAL_PER_PROFILE", 96))


def impair(x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    n = len(x)
    params: dict = {}
    y = x.astype(np.complex128, copy=True)

    # static multipath (0-3 taps, Rayleigh-ish complex gains)
    taps = int(rng.integers(0, 4))
    params["multipathTaps"] = taps
    if taps:
        delays = rng.integers(1, 65, size=taps)
        gains = (rng.normal(size=taps) + 1j * rng.normal(size=taps)) * 0.2
        for delay, gain in zip(delays, gains):
            y[delay:] += gain * y[: n - delay]
        params["multipathDelays"] = delays.tolist()

    # carrier frequency offset: refError U(-2,2) ppm of a drawn centre
    center_hz = 10 ** rng.uniform(np.log10(70e6), np.log10(10e9))
    cfo_cycles_per_sample = rng.uniform(-2, 2) * 1e-6 * center_hz / TARGET_FS
    params["centerHz"] = center_hz
    params["cfoCyclesPerSample"] = cfo_cycles_per_sample
    idx = np.arange(n)
    y *= np.exp(2j * np.pi * cfo_cycles_per_sample * idx)

    # Wiener phase noise scaled by centre frequency
    phase_std = (0.004 + rng.uniform(0, 0.03)) * math.sqrt(center_hz / 1e9)
    params["phaseNoiseStd"] = phase_std
    y *= np.exp(1j * np.cumsum(rng.normal(0, phase_std, size=n)))

    # IQ imbalance + DC
    g = rng.uniform(-0.12, 0.12)
    phi = rng.uniform(-0.30, 0.30)
    dc = rng.uniform(-0.10, 0.10) + 1j * rng.uniform(-0.10, 0.10)
    params.update(iqGainImbalance=g, iqPhaseImbalance=phi,
                  dcInPhase=dc.real, dcQuadrature=dc.imag)
    i = y.real * (1 + g)
    q = y.imag * (1 - g)
    q = q * math.cos(phi) + i * math.sin(phi)
    y = (i + 1j * q) + dc * np.sqrt(np.mean(np.abs(y) ** 2))

    # AWGN at drawn SNR relative to the ACTIVE-sample power (bursty signals'
    # SNR is a property of the emission, not of the silence between bursts)
    power = np.abs(y) ** 2
    active = power > 0.05 * power.max() if power.max() > 0 else power > 0
    signal_power = float(power[active].mean()) if active.any() else 0.0
    snr_db = rng.uniform(-5, 45)
    params["snrDb"] = snr_db
    if signal_power > 0:
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = (rng.normal(size=n) + 1j * rng.normal(size=n)) * math.sqrt(noise_power / 2)
        y += noise
    # unit-ish scale for float32 storage
    peak = np.abs(y).max()
    if peak > 0:
        y /= max(1.0, peak)
    return y.astype(np.complex64), params


def main() -> None:
    stage1 = json.loads((CORPUS / "manifest_stage1.json").read_text())
    raw = np.memmap(CORPUS / "clean_native.f32", dtype=np.complex64, mode="r")
    out_len = int(TARGET_FS * stage1["durationMs"] / 1_000)
    rows = stage1["items"]

    clean_out = np.lib.format.open_memmap(
        CORPUS / "clean.npy", mode="w+", dtype=np.complex64,
        shape=(len(rows), out_len))
    noisy_out = np.lib.format.open_memmap(
        CORPUS / "noisy.npy", mode="w+", dtype=np.complex64,
        shape=(len(rows), out_len))

    manifest_rows = []
    for index, row in enumerate(rows):
        start = row["byteOffset"] // 8
        x = np.asarray(raw[start : start + row["sampleCount"]])
        ratio = Fraction(TARGET_FS, row["sampleRateHz"]).limit_denominator(10_000)
        if ratio != 1:
            x = resample_poly(x, ratio.numerator, ratio.denominator,
                              window=("kaiser", 8.0))
        if len(x) < out_len:
            x = np.pad(x, (0, out_len - len(x)))
        clean = x[:out_len].astype(np.complex64)
        rng = np.random.default_rng(SEED ^ (index * 2_654_435_761 & 0xFFFFFFFF))
        noisy, params = impair(clean, rng)
        clean_out[index] = clean
        noisy_out[index] = noisy
        manifest_rows.append({
            **{k: row[k] for k in ("profile", "cls", "sampleRateHz",
                                    "startSampleIndex")},
            # Diversity provenance from stage 1. Every one of these is absent
            # from a stride-mode (pre-2026-07-31) manifest and most are absent
            # from a stage1-v1 manifest, so they are copied only when present
            # and readers must treat them as optional.
            #
            # contentSha256 is the row's FIXED-PHASE content hash. It is
            # content evidence only for contentClass 'A'; for 'B' and 'C' the
            # generator holds exactly one content realization and a differing
            # hash means a differing time origin, nothing more. Read
            # manifest_stage1.json's diversity block before quoting it.
            **{k: row[k] for k in (
                "offsetJitter", "offsetDrawIndex", "offsetRedraws",
                "carrierPhaseDraw", "carrierPhaseRadians",
                "contentClass", "contentPolicy", "contentSha256",
            ) if k in row},
            "row": index,
            "role": "eval" if (index % stage1["rowsPerProfile"]) < EVAL_PER_PROFILE else "train",
            "impairments": params,
        })
        if (index + 1) % 200 == 0:
            print(f"stage2 {index + 1}/{len(rows)}", flush=True)

    (CORPUS / "manifest.json").write_text(json.dumps({
        "schema": "longdwell-probe-corpus-v1",
        "targetSampleRateHz": TARGET_FS,
        "rowSamples": out_len,
        "durationMs": stage1["durationMs"],
        "hasCleanPairs": True,
        "impairmentSeed": SEED,
        # Stage-1 provenance, forwarded verbatim. `diversity` is the per-profile
        # record of what actually varies row to row and what does NOT: it is the
        # only place the corpus states, per profile, how many content
        # realizations it holds. Do not report accuracy from this corpus without
        # it.
        **{k: stage1[k] for k in (
            "planSource", "offsetMode", "offsetSeed", "offsetSpacing",
            "phaseMode", "contentSpanRows", "minActiveSamples",
            "offsetRejectionPolicy", "allowPhaseOnlyAcknowledged",
            "contentDeficitProfiles", "sourceProvenance", "diversity",
        ) if k in stage1},
        "stage1Schema": stage1.get("schema"),
        "rows": manifest_rows,
    }, indent=1))
    print(f"stage2 complete: {len(rows)} rows x {out_len} samples "
          f"({len(rows) * out_len * 8 * 2 / 1e9:.1f} GB pairs)")


if __name__ == "__main__":
    main()
