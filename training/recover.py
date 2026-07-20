"""Blind symbol recovery for single-carrier digital signals — the "push through
the information limit" front-end.

A fractionally-spaced (T/2) CMA equalizer inverts the propagation channel AND
absorbs symbol-timing offset in one adaptive filter. The Godard modulus target
R2 is a mild (QAM-ish) constant; CMA is largely constellation-agnostic but not
perfectly so, so the order decision is gated on *order-agnostic* recovery
quality, not on the equalizer cost. Two gate signals, both independent of
constellation order:
  * `estimate_snr` — blind in-band SNR from the PSD noise floor (catches the
    noise-limited regime),
  * `residual_isi` — normalized symbol-autocorrelation over the first few lags
    (catches residual ISI / mis-equalization). Well-equalized iid symbols have
    ~zero symbol autocorrelation regardless of modulation order, so this does not
    covary with the label the way a modulus-dispersion metric does.

Runs at capture ingestion (SDR path), not in the browser render loop — feedback
DSP, kept in Python. Order cumulants downstream are phase-rotation-invariant, so
no carrier recovery is required for classification.
"""

from __future__ import annotations

import numpy as np


def estimate_sps(x: np.ndarray, sps_range=(2, 16), default: int = 8, k_sigma: float = 5.0) -> int:
    """Blind symbol-rate estimate via the cyclostationary line in |x|^2.

    Falls back to `default` when there is no significant line (peak not >= k_sigma
    above the in-band median) — otherwise argmax would confidently return a
    garbage rate for CW / noise / non-linear-digital captures.
    """
    y = np.abs(x) ** 2
    y = y - y.mean()
    n = len(y)
    spec = np.abs(np.fft.rfft(y * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n)  # cycles/sample
    lo, hi = 1.0 / sps_range[1], 1.0 / sps_range[0]
    band = (freqs >= lo) & (freqs <= hi)
    if not band.any():
        return default
    bandspec = spec[band]
    peak = bandspec.max()
    med = np.median(bandspec)
    if peak < k_sigma * (med + 1e-30):
        return default
    fsym = freqs[band][int(np.argmax(bandspec))]
    return int(round(1.0 / fsym)) if fsym > 0 else default


def iq_balance(x: np.ndarray) -> np.ndarray:
    """Blind IQ-imbalance + DC correction via properness restoration.

    A cheap direct-conversion front-end makes the proper (circular) baseband
    IMPROPER: it leaks a mirror-image conj(z) term, so E[z^2] != 0. This is a
    *widely-linear* distortion that the linear CMA equalizer downstream cannot
    invert, and it biases exactly the order cumulants. We remove it with the
    widely-linear correction y = r + c*conj(r) that restores E[y^2]=0 to first
    order (c = -E[r^2]/(2 E|r|^2)) — closed-form, no pilots. DC is removed first.
    """
    r = x - x.mean()
    denom = np.mean(np.abs(r) ** 2) + 1e-12
    c = -np.mean(r * r) / (2.0 * denom)
    return r + c * np.conj(r)


def _resample_to(x: np.ndarray, sps_in: float, sps_out: float) -> np.ndarray:
    """Linear-interpolate from sps_in to sps_out samples/symbol."""
    n = len(x)
    new_n = int(round(n * sps_out / sps_in))
    if new_n < 4:
        return x
    pos = np.arange(new_n) * ((n - 1) / (new_n - 1))
    i0 = np.clip(np.floor(pos).astype(int), 0, n - 2)
    frac = pos - i0
    return x[i0] * (1 - frac) + x[i0 + 1] * frac


def cma_fse(x2: np.ndarray, ntaps: int = 21, mu: float = 2e-3, passes: int = 25,
            R2: float = 1.3) -> np.ndarray:
    """T/2 fractionally-spaced CMA equalizer. x2 is 2 samples/symbol.

    Returns symbol-spaced (1 sps) equalized output. Blind: no constellation or
    channel knowledge. The FSE absorbs timing offset, so no separate timing loop.
    """
    x2 = x2 / (np.sqrt(np.mean(np.abs(x2) ** 2)) + 1e-12)
    nsym = (len(x2) - ntaps) // 2
    if nsym < 8:
        return x2[::2]
    w = np.zeros(ntaps, dtype=complex)
    w[ntaps // 2] = 1.0
    y = np.zeros(nsym, dtype=complex)
    for _ in range(passes):
        for k in range(nsym):
            u = x2[2 * k : 2 * k + ntaps]
            yk = np.dot(w, u[::-1])
            w += mu * yk * (R2 - abs(yk) ** 2) * np.conj(u[::-1])
        # light tap-leakage keeps it from drifting
        w *= 0.9999
    for k in range(nsym):
        y[k] = np.dot(w, x2[2 * k : 2 * k + ntaps][::-1])
    return y


def carrier_4th(y: np.ndarray) -> np.ndarray:
    """Remove carrier phase (mod 90 deg) via a 4th-power estimate — for EVM."""
    phi = np.angle(np.mean(y ** 4)) / 4.0
    return y * np.exp(-1j * phi)


def residual_isi(y: np.ndarray, max_lag: int = 4) -> float:
    """Order-AGNOSTIC recovery-quality proxy: normalized symbol autocorrelation
    over the first `max_lag` lags. Well-equalized iid symbols (any modulation
    order) have ~zero symbol autocorrelation; residual ISI / mis-equalization
    colors the sequence and raises it. Rotation-invariant (a global phase cancels
    in y[k]·conj(y[k-lag])) and does NOT covary with constellation order, unlike a
    modulus-dispersion metric. White noise is uncorrelated, so this isolates ISI
    from the noise floor (which `estimate_snr` covers)."""
    z = y - y.mean()
    r0 = float(np.mean(np.abs(z) ** 2)) + 1e-12
    acc = 0.0
    for lag in range(1, max_lag + 1):
        if lag >= len(z):
            break
        acc += abs(np.mean(z[lag:] * np.conj(z[:-lag]))) ** 2
    return float(np.sqrt(acc) / r0)


def estimate_snr(x: np.ndarray, nfft: int = 512) -> float:
    """Blind in-band SNR estimate (dB) from the PSD noise floor. Order-agnostic
    (measured on the spectrum, not the constellation). Noise power is counted
    over the OCCUPIED bins only, so it is a true in-band SNR."""
    import preprocess as pp
    psd = pp._smooth(pp.welch_psd(x.astype(np.complex128), nfft), pp.SMOOTH)
    floor = pp.NOISE_FLOOR_SCALE * float(np.median(psd))
    inband = psd > floor
    n_in = int(np.count_nonzero(inband))
    sig = float(np.clip(psd - floor, 0.0, None).sum())
    noise = floor * max(n_in, 1)
    if noise <= 0 or sig <= 0:
        return -10.0
    return float(10.0 * np.log10(sig / noise))


def recover(x: np.ndarray, sps_hint: int | None = None) -> dict:
    """Full blind recovery. Returns recovered symbols + order-agnostic diagnostics."""
    xb = iq_balance(x)                           # blind IQ-imbalance + DC removal
    sps = sps_hint or estimate_sps(xb)
    x2 = _resample_to(xb, sps, 2.0)              # to T/2
    sym = cma_fse(x2)
    return {
        "symbols": sym,
        "sps": sps,
        "residual_isi": residual_isi(sym),
        "snr_db": estimate_snr(x),
    }
