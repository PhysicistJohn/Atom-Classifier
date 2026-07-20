"""Reproducible d' experiment behind the push-through claims in DESIGN.md.

Measures the 16-vs-64-QAM separability (d' on the normalized C42 cumulant, the
classic order discriminator) under the levers that push through the information
limit: dwell, timing sync, and equalization. Committed so the DESIGN numbers are
reproducible, not prose-only.

    .venv-training/bin/python training/experiments/dprime.py
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rfgen  # noqa: E402
import recover  # noqa: E402

SPS = 8


def qam(side: int, n_sym: int, rng) -> np.ndarray:
    lv = np.arange(-(side - 1), side, 2, float)
    s = rng.choice(lv, n_sym) + 1j * rng.choice(lv, n_sym)
    return s / np.sqrt(np.mean(np.abs(s) ** 2))


def pulse(sym: np.ndarray, rng) -> np.ndarray:
    up = np.zeros(len(sym) * SPS, complex)
    up[::SPS] = sym
    return np.convolve(up, rfgen.rrc_taps(SPS, 6, 0.3), "same")


def multipath(x: np.ndarray, rng):
    h = np.zeros(9, complex)
    h[0] = 1.0
    h[3] = 0.4 * (rng.standard_normal() + 1j * rng.standard_normal())
    h[6] = 0.22 * (rng.standard_normal() + 1j * rng.standard_normal())
    return np.convolve(x, h, "full")[: len(x)], h


def awgn(x, rng, snr):
    npow = np.mean(np.abs(x) ** 2) / 10 ** (snr / 10)
    return x + (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x))) * np.sqrt(npow / 2)


def c42(z: np.ndarray) -> float:
    z = z - z.mean()
    z = z / (np.sqrt(np.mean(np.abs(z) ** 2)) + 1e-12)
    return float((np.mean(np.abs(z) ** 4) - abs(np.mean(z * z)) ** 2 - 2).real)


def dprime(a, b) -> float:
    a, b = np.array(a), np.array(b)
    return abs(a.mean() - b.mean()) / np.sqrt(0.5 * (a.var() + b.var()) + 1e-12)


def sym_at(x, taps, sync):
    mf = np.convolve(x, taps[::-1], "same")
    if sync:
        idx = np.arange(9 * SPS, len(mf) - 9 * SPS, SPS)
        return mf[idx]
    return mf[9 * SPS : len(mf) - 9 * SPS]


def oracle_eq(x, h, snr):
    N = len(x)
    H = np.fft.fft(h, N)
    npow = np.mean(np.abs(x) ** 2) / 10 ** (snr / 10)
    G = np.conj(H) / (np.abs(H) ** 2 + npow / max(np.mean(np.abs(x) ** 2), 1e-9))
    return np.fft.ifft(np.fft.fft(x) * G)


def main():
    taps = rfgen.rrc_taps(SPS, 6, 0.3)
    print("d'(16 vs 64 QAM) on C42, SNR 25 dB\n")

    print("A. dwell sweep (clean channel, no sync) — d' ~ sqrt(N):")
    for nsym in [128, 512, 2048, 8192]:
        a, b = [], []
        for t in range(40):
            rng = np.random.default_rng(1000 * t + nsym)
            a.append(c42(sym_at(awgn(pulse(qam(4, nsym, rng), rng), rng, 25), taps, False)))
            b.append(c42(sym_at(awgn(pulse(qam(8, nsym, rng), rng), rng, 25), taps, False)))
        print(f"   N={nsym:5d}  d'={dprime(a, b):.2f}")

    print("\nB. timing sync uplift (clean, 2048 sym):")
    for sync, lbl in [(False, "no sync"), (True, "oracle timing")]:
        a, b = [], []
        for t in range(40):
            rng = np.random.default_rng(7 * t + 2)
            a.append(c42(sym_at(awgn(pulse(qam(4, 2048, rng), rng), rng, 25), taps, sync)))
            b.append(c42(sym_at(awgn(pulse(qam(8, 2048, rng), rng), rng, 25), taps, sync)))
        print(f"   {lbl:14s}  d'={dprime(a, b):.2f}")

    print("\nC. multipath: equalization is the lever (2048 sym):")
    for mode in ["no multipath (ref)", "multipath, no eq", "multipath, oracle eq", "multipath, blind CMA+IQ"]:
        a, b = [], []
        for t in range(40):
            rng = np.random.default_rng(11 * t + 5)
            for side, acc in [(4, a), (8, b)]:
                x = pulse(qam(side, 2048, rng), rng)
                if "no multipath" in mode:
                    xx = awgn(x, rng, 25)
                    acc.append(c42(sym_at(xx, taps, True)))
                    continue
                xmp, h = multipath(x, rng)
                xx = awgn(xmp, rng, 25)
                if mode == "multipath, no eq":
                    acc.append(c42(sym_at(xx, taps, True)))
                elif mode == "multipath, oracle eq":
                    acc.append(c42(sym_at(oracle_eq(xx, h, 25), taps, True)))
                else:  # blind CMA (+ IQ comp) via the shipping recover.py
                    acc.append(c42(recover.recover(xx, sps_hint=SPS)["symbols"]))
        print(f"   {mode:26s}  d'={dprime(a, b):.2f}")


if __name__ == "__main__":
    main()
