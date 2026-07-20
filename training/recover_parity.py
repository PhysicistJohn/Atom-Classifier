"""Generate the blind-recovery parity fixture for the TypeScript port.

Builds a handful of deterministic captures (fixed seed) — a synthesized QPSK and
16QAM single-carrier signal with a mild multipath channel + small carrier/timing
offset + AWGN, plus a pure-noise case and a CW tone — runs the exact
`recover.ts` display contract on each (iq_balance -> sps -> T/2 resample ->
cma_fse -> carrier_4th, with residual_isi on the pre-carrier symbols and
estimate_snr on the original I/Q), and writes
`src/embedding/assets/recover-parity-fixture.json`.

The saved `expected` block mirrors `recoverConstellation`'s return exactly (the
carrier-locked symbols), so the TS parity test can compare element-for-element.

Run:  python3 training/recover_parity.py
"""

from __future__ import annotations

import json
import os

import numpy as np

import recover as rc

ASSET_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "embedding", "assets")
OUT = os.path.join(ASSET_DIR, "recover-parity-fixture.json")


def rrc(beta: float, sps: int, span: int) -> np.ndarray:
    """Root-raised-cosine pulse, unit-energy, length span*sps + 1."""
    N = span * sps
    t = (np.arange(N + 1) - N / 2) / sps
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-9:
            h[i] = 1.0 - beta + 4.0 * beta / np.pi
        elif beta > 0 and abs(abs(ti) - 1.0 / (4.0 * beta)) < 1e-9:
            h[i] = (beta / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1.0 - beta)) + 4.0 * beta * ti * np.cos(np.pi * ti * (1.0 + beta))
            den = np.pi * ti * (1.0 - (4.0 * beta * ti) ** 2)
            h[i] = num / den
    return h / np.sqrt(np.sum(h ** 2))


def _frac_delay(x: np.ndarray, d: float) -> np.ndarray:
    """Fractional-sample delay via linear interpolation (small timing offset)."""
    n = len(x)
    idx = np.clip(np.arange(n) - d, 0.0, n - 1.0)
    i0 = np.clip(np.floor(idx).astype(int), 0, n - 2)
    f = idx - i0
    return x[i0] * (1.0 - f) + x[i0 + 1] * f


def single_carrier(order: int, sps: int, nsym: int, snr_db: float, seed: int) -> np.ndarray:
    r = np.random.default_rng(seed)
    if order == 4:  # QPSK
        pts = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2.0)
    elif order == 16:  # 16QAM
        levels = np.array([-3.0, -1.0, 1.0, 3.0])
        pts = np.array([a + 1j * b for a in levels for b in levels])
        pts = pts / np.sqrt(np.mean(np.abs(pts) ** 2))
    else:
        raise ValueError(order)
    syms = pts[r.integers(0, len(pts), nsym)]
    up = np.zeros(nsym * sps, dtype=complex)
    up[::sps] = syms
    tx = np.convolve(up, rrc(0.3, sps, 8))
    # mild sparse multipath channel
    ch = np.array([1.0, 0.0, 0.35 + 0.15j, 0.0, -0.12])
    rxc = np.convolve(tx, ch)
    rxc = _frac_delay(rxc, 0.4)  # small timing offset
    n = np.arange(len(rxc))
    rxc = rxc * np.exp(1j * (2.0 * np.pi * 0.001 * n + 0.5))  # small CFO + phase
    p = np.mean(np.abs(rxc) ** 2)
    npow = p / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(npow / 2.0) * (r.standard_normal(len(rxc)) + 1j * r.standard_normal(len(rxc)))
    return (rxc + noise).astype(np.complex128)


def noise_case(n: int, seed: int) -> np.ndarray:
    r = np.random.default_rng(seed)
    return (r.standard_normal(n) + 1j * r.standard_normal(n)).astype(np.complex128) / np.sqrt(2.0)


def cw_case(n: int, f: float, snr_db: float, seed: int) -> np.ndarray:
    r = np.random.default_rng(seed)
    tone = np.exp(1j * (2.0 * np.pi * f * np.arange(n) + 0.3))
    npow = 1.0 / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(npow / 2.0) * (r.standard_normal(n) + 1j * r.standard_normal(n))
    return (tone + noise).astype(np.complex128)


def recover_display(x: np.ndarray, sps_hint):
    """Exact mirror of recoverConstellation's return contract."""
    xb = rc.iq_balance(x)
    sps = sps_hint or rc.estimate_sps(xb)
    x2 = rc._resample_to(xb, sps, 2.0)
    sym = rc.cma_fse(x2)
    isi = rc.residual_isi(sym)             # pre-carrier (rotation-invariant)
    snr = rc.estimate_snr(x)               # on the original I/Q
    locked = rc.carrier_4th(sym)           # carrier-lock for display
    return locked, int(sps), float(isi), float(snr)


def entry(name: str, x: np.ndarray, sps_hint):
    locked, sps, isi, snr = recover_display(x, sps_hint)
    print(f"  {name:12s} sps={sps:2d} nsym={len(locked):4d} residualIsi={isi:.4f} snrDb={snr:.3f}")
    return {
        "name": name,
        "re": x.real.astype(float).tolist(),
        "im": x.imag.astype(float).tolist(),
        "spsHint": sps_hint,  # int or None
        "expected": {
            "symbolsRe": locked.real.astype(float).tolist(),
            "symbolsIm": locked.imag.astype(float).tolist(),
            "sps": sps,
            "residualIsi": isi,
            "snrDb": snr,
        },
    }


def main() -> None:
    cases = []
    print("recover parity cases:")
    # QPSK: known sps supplied as a hint
    cases.append(entry("qpsk", single_carrier(4, 8, 220, 20.0, seed=11), sps_hint=8))
    # 16QAM: no hint -> exercises estimate_sps
    cases.append(entry("qam16", single_carrier(16, 4, 320, 25.0, seed=22), sps_hint=None))
    # pure complex noise: no line -> estimate_sps falls back to default
    cases.append(entry("noise", noise_case(1500, seed=33), sps_hint=None))
    # CW tone: constant envelope, no cyclostationary line
    cases.append(entry("cw", cw_case(1500, 0.11, 25.0, seed=44), sps_hint=None))

    os.makedirs(ASSET_DIR, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"cases": cases}, f)
    print("wrote", os.path.relpath(OUT))


if __name__ == "__main__":
    main()
