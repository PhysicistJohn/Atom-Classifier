"""Magnitude-only representation for the tinySA (scalar) flavor of the classifier.

A tinySA measures a *power spectrum* (power vs frequency), no phase. This derives
the same kind of representation from the SAME SignalLab I/Q corpus — no separate
corpus — by taking |FFT|^2 and reducing it to a scale-invariant spectral-shape
fingerprint + a few magnitude/spectral scalars. At tinySA inference the input is
the swept power spectrum, run through the identical `representation_from_psd`.

Design mirrors the I/Q front-end: detect the occupied band, normalise its SHAPE
to a canonical length (scale-invariance across span/RBW), and keep occupied
bandwidth as a separate feature (width discriminator). The TS port implements the
same `representation_from_psd`; parity-tested against exported fixtures.
"""

from __future__ import annotations

import numpy as np

import preprocess as pp

MAG_LEN = 256          # canonical spectral-shape length
MAG_NFFT = 1024        # PSD resolution for training-from-I/Q
MARGIN = 0.75          # include this fraction of bw of roll-off beyond the band
N_MAG_FEATURES = 8


def lin_resample_real(x: np.ndarray, new_len: int) -> np.ndarray:
    n = len(x)
    if n == new_len or n < 2:
        return np.resize(x.astype(np.float64), new_len) if n < 2 else x.astype(np.float64)
    pos = np.arange(new_len) * ((n - 1) / (new_len - 1))
    i0 = np.clip(np.floor(pos).astype(int), 0, n - 2)
    frac = pos - i0
    return x[i0] * (1 - frac) + x[i0 + 1] * frac


def representation_from_psd(psd: np.ndarray, center: float, bw: float) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a (linear, fftshifted) power spectrum + occupied band to a canonical
    log-shape [MAG_LEN] in [0,1] and N_MAG_FEATURES magnitude/spectral scalars.
    Works identically on a Welch PSD (training) or a tinySA sweep (inference)."""
    nfft = len(psd)
    half = bw * (0.5 + MARGIN)
    lo_i = int(np.clip(round((center - half + 0.5) * nfft), 0, nfft - 1))
    hi_i = int(np.clip(round((center + half + 0.5) * nfft), lo_i + 2, nfft))
    band = np.asarray(psd[lo_i:hi_i], dtype=np.float64) + 1e-12

    # scale-invariant SHAPE: relative-dB above the noise floor, peak-normalised
    logband = 10.0 * np.log10(band)
    logband = np.clip(logband - np.median(logband), 0.0, None)
    peak = float(logband.max()) + 1e-9
    shape = lin_resample_real(logband / peak, MAG_LEN).astype(np.float32)

    # magnitude/spectral features (all derivable from a power spectrum)
    p = band / band.sum()
    xs = np.linspace(0.0, 1.0, len(band))
    centroid = float(np.sum(xs * p))
    spread = float(np.sqrt(np.sum((xs - centroid) ** 2 * p)) + 1e-9)
    skew = float(np.sum((xs - centroid) ** 3 * p) / spread ** 3)
    kurt = float(np.sum((xs - centroid) ** 4 * p) / spread ** 4)
    flatness = float(np.exp(np.mean(np.log(band))) / (np.mean(band) + 1e-12))
    papr = float(band.max() / (band.mean() + 1e-12))
    c0, c1 = int(len(band) * 0.45), int(len(band) * 0.55)
    frac_central = float(band[c0:c1].sum() / band.sum())
    feats = np.array([bw, flatness, np.log1p(papr), spread, skew, kurt, frac_central,
                      len(band) / nfft], dtype=np.float32)
    return shape, feats


def magnitude_from_iq(iq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Training path: complex I/Q -> Welch PSD -> the magnitude representation."""
    psd = pp._smooth(pp.welch_psd(iq.astype(np.complex128), MAG_NFFT), pp.SMOOTH)
    center, bw = pp.estimate_band(iq, MAG_NFFT)
    return representation_from_psd(psd, center, bw)
