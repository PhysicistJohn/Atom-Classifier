"""DSP front-end: detect -> downconvert -> resample -> amplitude-normalise.

Decouples the nuisance geometry (where the signal sits, how wide it is, how
strong it is) from the modulation structure the embedding learns. The measured
centre and occupied bandwidth are returned as *separate* context scalars — they
are deliberately NOT fed into the embedding, because they are what disambiguates
modulation-degenerate protocols (e.g. OFDM variants) at fusion time.

The operations here are intentionally simple and FFT-light (one radix-2 PSD, then
complex multiply + linear-interpolation resample) so the TypeScript inference
port can mirror them exactly. `iq-preprocess.ts` is a line-for-line port and a
parity test guards the two implementations against drift.
"""

from __future__ import annotations

import numpy as np

L_OUT = 1024        # embedding input length (complex samples)
TARGET_FRAC = 0.5   # canonical occupied fractional bandwidth after normalisation
NFFT = 512          # PSD segment length (power of two for the radix-2 port)
ENERGY_EDGE = 0.005 # 99% occupied-bandwidth energy percentile


def _hann(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def welch_psd(x: np.ndarray, nfft: int = NFFT) -> np.ndarray:
    """Averaged periodogram, fftshifted so index 0 is the most-negative freq."""
    win = _hann(nfft)
    hop = nfft // 2
    if len(x) < nfft:
        x = np.pad(x, (0, nfft - len(x)))
    acc = np.zeros(nfft)
    count = 0
    for start in range(0, len(x) - nfft + 1, hop):
        seg = x[start : start + nfft] * win
        acc += np.abs(np.fft.fft(seg)) ** 2
        count += 1
    if count == 0:
        acc += np.abs(np.fft.fft(x[:nfft] * win)) ** 2
        count = 1
    return np.fft.fftshift(acc / count)


# Noise-floor scale on the PSD median. For a narrowband signal most bins are
# noise, and for exponentially-distributed noise power mean ~= median/ln(2), so
# 1.44*median recovers the true white floor and keeps the occupied-bandwidth
# estimate stable down to ~3-6 dB SNR (tuned empirically across classes).
NOISE_FLOOR_SCALE = 1.44
SMOOTH = 5          # PSD moving-average window (bins)


def _smooth(psd: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return psd
    k = np.ones(w) / w
    return np.convolve(psd, k, mode="same")


def estimate_band(x: np.ndarray, nfft: int = NFFT) -> tuple[float, float]:
    """Return (centre_freq, occupied_bw) in cycles/sample; bw is a fraction.

    The occupied band is measured on the *noise-subtracted* PSD so a white noise
    floor (which dominates the energy percentile at low SNR) does not smear the
    estimate across the whole band. The floor is a low percentile of the PSD, so
    it is robust for narrowband signals (mostly guard band) and harmless for
    wideband ones (subtracts a small uniform pedestal).
    """
    psd = _smooth(welch_psd(x, nfft), SMOOTH)
    freqs = (np.arange(nfft) / nfft) - 0.5
    floor = NOISE_FLOOR_SCALE * float(np.median(psd))
    sig = np.clip(psd - floor, 0.0, None)
    total = sig.sum()
    if total < 1e-9:  # essentially all noise -> full band
        return 0.0, 0.95
    cum = np.cumsum(sig) / total
    lo_i = int(np.searchsorted(cum, ENERGY_EDGE))
    hi_i = int(np.searchsorted(cum, 1.0 - ENERGY_EDGE))
    lo_i = min(lo_i, nfft - 1)
    hi_i = min(max(hi_i, lo_i + 1), nfft - 1)
    f_lo = freqs[lo_i]
    f_hi = freqs[hi_i]
    center = 0.5 * (f_lo + f_hi)
    bw = max(f_hi - f_lo, 1.0 / nfft)
    return float(center), float(bw)


def lin_resample(x: np.ndarray, new_len: int) -> np.ndarray:
    """Complex linear-interpolation resampler (portable, deterministic)."""
    n = len(x)
    if new_len == n or n < 2:
        return x.astype(np.complex64)
    # position of each output sample in input coordinates
    pos = np.arange(new_len) * ((n - 1) / (new_len - 1)) if new_len > 1 else np.zeros(1)
    i0 = np.floor(pos).astype(int)
    i0 = np.clip(i0, 0, n - 2)
    frac = pos - i0
    out = x[i0] * (1.0 - frac) + x[i0 + 1] * frac
    return out.astype(np.complex64)


def center_fit(x: np.ndarray, length: int) -> np.ndarray:
    """Centre-crop or symmetric-pad a complex vector to `length`."""
    n = len(x)
    if n == length:
        return x
    if n > length:
        start = (n - length) // 2
        return x[start : start + length]
    pad = length - n
    left = pad // 2
    return np.pad(x, (left, pad - left))


def preprocess(
    iq: np.ndarray,
    l_out: int = L_OUT,
    target_frac: float = TARGET_FRAC,
    nfft: int = NFFT,
    scale_jitter: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """Full front-end. Returns (normalised complex I/Q[l_out], context)."""
    center, bw = estimate_band(iq, nfft)

    # down-convert measured centre to DC (frequency-invariance)
    n = np.arange(len(iq))
    x = iq * np.exp(-1j * 2.0 * np.pi * center * n)

    # resample so occupied bandwidth hits the canonical target (scale-invariance).
    # scale_jitter models bandwidth-estimate error so the embedding tolerates it.
    frac = bw
    if scale_jitter > 0.0 and rng is not None:
        frac = frac * (1.0 + rng.uniform(-scale_jitter, scale_jitter))
    frac = float(np.clip(frac, 1e-3, 0.95))
    ratio = frac / target_frac
    new_len = int(max(64, round(len(x) * ratio)))
    x = lin_resample(x, new_len)

    # fit to canonical length and amplitude-normalise (power-invariance)
    x = center_fit(x, l_out)
    rms = np.sqrt(np.mean(np.abs(x) ** 2) + 1e-12)
    x = (x / rms).astype(np.complex64)

    return x, {"center": center, "bw": bw}


def to_channels(x: np.ndarray) -> np.ndarray:
    """Complex[l] -> real[2, l] (I, Q) for the network."""
    return np.stack([x.real, x.imag], axis=0).astype(np.float32)


N_FEATURES = 12


def iq_features(x: np.ndarray) -> np.ndarray:
    """Phase-rotation-invariant higher-order + instantaneous statistics.

    The conv+pool path cannot cheaply compute phase-invariant higher-order
    cumulants, yet those are exactly the statistics that separate modulation
    *order* (QPSK vs 16QAM vs 64QAM) without symbol-timing/carrier recovery. The
    last two features are instantaneous-frequency spread and amplitude
    coefficient-of-variation, which crack the constant-envelope narrowband
    cluster: CW (flat freq, flat amp) vs AM (flat freq, varying amp) vs FM/GFSK
    (varying freq). Every feature is invariant to a constant phase rotation
    (cumulant magnitudes, or amplitude/derivative-of-phase quantities), so they
    survive the coarse front-end. Computed on the *normalised* I/Q so the TS
    port reproduces them from the same [2, L] the network sees. See Swami &
    Sadler, "Hierarchical digital modulation classification using cumulants".
    """
    z = x.astype(np.complex128)
    z = z - z.mean()
    p = np.sqrt(np.mean(np.abs(z) ** 2)) + 1e-12
    z = z / p  # zero-mean, unit-power
    az2 = np.abs(z) ** 2
    m20 = np.mean(z * z)
    m40 = np.mean(z ** 4)
    m41 = np.mean(z ** 3 * np.conj(z))
    m42 = float(np.mean(az2 ** 2))          # E|z|^4
    m60 = np.mean(z ** 6)
    m63 = float(np.mean(az2 ** 3))          # E|z|^6
    c20 = m20
    c40 = m40 - 3.0 * m20 ** 2
    c41 = m41 - 3.0 * m20                    # m21 = 1
    c42 = m42 - abs(m20) ** 2 - 2.0
    c60 = m60 - 15.0 * m20 * m40 + 30.0 * m20 ** 3
    c63 = m63 - 9.0 * c42 - 6.0
    absz = np.abs(z)
    ifreq = np.diff(np.unwrap(np.angle(z)))  # instantaneous frequency
    cov = float(np.std(absz) / (np.mean(absz) + 1e-9))
    return np.array([
        abs(c20), abs(c40), float(c42.real), abs(c41), abs(c60),
        float(c63), m42, float(np.std(absz)), float(np.max(az2)), float(np.mean(absz)),
        float(np.std(ifreq)), cov,
    ], dtype=np.float32)


def features_from_channels(ch: np.ndarray) -> np.ndarray:
    """iq_features from a [2, L] real array (I, Q) — the inference path."""
    return iq_features(ch[0] + 1j * ch[1])
