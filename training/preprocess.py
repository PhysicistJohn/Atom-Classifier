"""DSP front-end: seam-safe hybrid detect -> downconvert -> resample -> normalise.

Decouples the nuisance geometry (where the signal sits, how wide it is, how
strong it is) from the modulation structure the embedding learns. The measured
centre and occupied bandwidth are returned as *separate* context scalars — they
are deliberately NOT fed into the embedding, because they are what disambiguates
modulation-degenerate protocols (e.g. OFDM variants) at fusion time.

The default ``hybrid-v3`` estimator preserves the measured ``linear-v1``
geometry on ordinary, non-wrapping spectra. It selects the circular estimator
only when that estimator identifies a credible ``-0.5/+0.5`` seam crossing.
This retains the classifier quality of the legacy floor estimator without
reintroducing the seam bug. Detection and RMS normalization are invariant to
every finite non-zero global input gain. The TypeScript inference port mirrors
the contract and parity fixtures guard the two implementations against drift.
"""

from __future__ import annotations

import numpy as np

L_OUT = 1024        # embedding input length (complex samples)
TARGET_FRAC = 0.5   # canonical occupied fractional bandwidth after normalisation
NFFT = 512          # PSD segment length (power of two for the radix-2 port)
ENERGY_EDGE = 0.005 # 99% occupied-bandwidth energy percentile
FULL_BAND_BW = 0.95
PREPROCESS_VERSION = "hybrid-v3"


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


# Retained for backward compatibility with exported v1 model metadata. The
# circular-v2 estimator uses a contiguous low-power guard and
# GUARD_FLOOR_SCALE instead of the global PSD median.
NOISE_FLOOR_SCALE = 1.44
SMOOTH = 5          # PSD moving-average window (bins)

# The expected spectral-flatness deficit of a smoothed white-noise Welch
# spectrum scales approximately as 1 / segment_count. This conservative value
# was verified for capture lengths 512 through 16384.
WHITE_FLATNESS_DEFICIT = 0.30

# A supported occupied band leaves at least 5% of the circle as a guard because
# bandwidth is capped at FULL_BAND_BW. Estimating the floor in the lowest-power
# circular contiguous guard remains valid for bands wider than half the FFT.
GUARD_FRACTION = 1.0 - FULL_BAND_BW
GUARD_FLOOR_SCALE = 1.30
MIN_EXCESS_FRACTION = 1e-6

# Same-distribution development evidence showed that circular-v2's minimum
# guard floor broadened low-SNR, non-wrapping signals and cost 8.0 percentage
# points in a frozen random-CNN ridge probe. Keep linear-v1 unless the circular
# arc crosses the seam and either estimator supplies independent evidence that
# the apparent wrap is credible. These thresholds are serialized as part of the
# preprocessing contract and mirrored exactly in TypeScript.
HYBRID_LEGACY_WIDE_BW = 0.90
HYBRID_CIRCULAR_BROAD_BW = 0.65
HYBRID_SEAM_TOLERANCE_BINS = 1.0


def preprocess_metadata() -> dict:
    """Serializable contract consumed by the TypeScript inference port."""
    return {
        "estimator_version": PREPROCESS_VERSION,
        "l_out": L_OUT,
        "target_frac": TARGET_FRAC,
        "nfft": NFFT,
        "energy_edge": ENERGY_EDGE,
        # Kept so older readers can still parse the common schema.
        "noise_floor_scale": NOISE_FLOOR_SCALE,
        "smooth": SMOOTH,
        "full_band_bw": FULL_BAND_BW,
        "white_flatness_deficit": WHITE_FLATNESS_DEFICIT,
        "guard_floor_scale": GUARD_FLOOR_SCALE,
        "min_excess_fraction": MIN_EXCESS_FRACTION,
        "hybrid_legacy_wide_bw": HYBRID_LEGACY_WIDE_BW,
        "hybrid_circular_broad_bw": HYBRID_CIRCULAR_BROAD_BW,
        "hybrid_seam_tolerance_bins": HYBRID_SEAM_TOLERANCE_BINS,
    }


def _smooth(psd: np.ndarray, w: int) -> np.ndarray:
    """Legacy zero-padded smoothing used by the magnitude representation."""
    if w <= 1:
        return psd
    k = np.ones(w) / w
    return np.convolve(psd, k, mode="same")


def _as_finite_complex_1d(iq: np.ndarray) -> np.ndarray:
    x = np.asarray(iq)
    if x.ndim != 1 or len(x) == 0:
        raise ValueError(f"expected a non-empty 1-D I/Q capture, got shape={x.shape}")
    if not np.iscomplexobj(x):
        raise TypeError(f"expected complex I/Q, got dtype={x.dtype}")
    if not np.all(np.isfinite(x.real)) or not np.all(np.isfinite(x.imag)):
        raise ValueError("I/Q capture contains NaN or infinity")
    return x.astype(np.complex128, copy=False)


def _peak_normalized(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Return ``x / max(abs(x))`` without power underflow or overflow."""
    peak = float(np.max(np.abs(x)))
    if peak == 0.0:
        return np.zeros_like(x, dtype=np.complex128), 0.0
    if not np.isfinite(peak):
        raise ValueError("I/Q magnitude is not finite")
    return x / peak, peak


def _circular_smooth(psd: np.ndarray, width: int = SMOOTH) -> np.ndarray:
    """Moving average with periodic FFT boundaries."""
    p = np.asarray(psd, dtype=np.float64)
    if width <= 1:
        return p
    if width > len(p):
        raise ValueError("smoothing width may not exceed PSD length")
    offsets = np.arange(width) - (width // 2)
    out = np.zeros_like(p)
    for offset in offsets:
        out += np.roll(p, int(offset))
    return out / width


def _spectral_flatness(psd: np.ndarray) -> float:
    p = np.asarray(psd, dtype=np.float64)
    mean = float(np.mean(p))
    if mean <= 0.0:
        return 1.0
    tiny = np.finfo(np.float64).tiny
    return float(np.exp(np.mean(np.log(np.maximum(p, tiny)))) / mean)


def _welch_segment_count(length: int, nfft: int) -> int:
    if length < nfft:
        return 1
    return 1 + (length - nfft) // (nfft // 2)


def _white_flatness_threshold(length: int, nfft: int) -> float:
    count = _welch_segment_count(length, nfft)
    return max(0.0, 1.0 - WHITE_FLATNESS_DEFICIT / count)


def _minimum_circular_guard(psd: np.ndarray) -> np.ndarray:
    """Lowest-energy supported contiguous guard, with deterministic summation."""
    p = np.asarray(psd, dtype=np.float64)
    n = len(p)
    guard_bins = min(
        n,
        max(2 * SMOOTH + 1, int(np.ceil(GUARD_FRACTION * n))),
    )
    best_start = 0
    best_sum = float("inf")
    for start in range(n):
        total = 0.0
        for offset in range(guard_bins):
            total += float(p[(start + offset) % n])
        if total < best_sum:
            best_sum = total
            best_start = start
    return p[(best_start + np.arange(guard_bins)) % n]


def _shortest_circular_interval(
    weights: np.ndarray,
    retained_fraction: float,
) -> tuple[int, int]:
    """Inclusive unwrapped endpoints of the shortest retained-energy arc."""
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or len(w) == 0 or np.any(w < 0.0):
        raise ValueError("circular interval weights must be non-empty and non-negative")
    total = float(w.sum())
    if total <= 0.0:
        return 0, len(w) - 1
    if not 0.0 < retained_fraction <= 1.0:
        raise ValueError("retained_fraction must lie in (0, 1]")

    n = len(w)
    target = retained_fraction * total
    doubled = np.concatenate([w, w])
    end_exclusive = 0
    running = 0.0
    best: tuple[int, float, int, int] | None = None

    for start in range(n):
        if end_exclusive < start:
            end_exclusive = start
            running = 0.0
        while end_exclusive < start + n and running < target:
            running += float(doubled[end_exclusive])
            end_exclusive += 1
        if running >= target:
            candidate = (
                end_exclusive - start,
                running - target,
                start,
                end_exclusive - 1,
            )
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        if end_exclusive > start:
            running -= float(doubled[start])
            if running < 0.0 and running > -1e-12 * total:
                running = 0.0

    if best is None:
        return 0, n - 1
    return best[2], best[3]


def _wrap_cycles(value: float) -> float:
    return float((value + 0.5) % 1.0 - 0.5)


def _validate_nfft(nfft: int) -> int:
    if (
        not isinstance(nfft, (int, np.integer))
        or int(nfft) < 2 * SMOOTH + 1
        or int(nfft) % 2 != 0
        or int(nfft) & (int(nfft) - 1)
    ):
        raise ValueError(f"nfft must be a power of two >= {2 * SMOOTH + 1}")
    return int(nfft)


def estimate_band_linear_v1(
    x: np.ndarray,
    nfft: int = NFFT,
    *,
    normalize_gain: bool = False,
) -> tuple[float, float]:
    """Legacy non-circular estimator, retained as an explicit contract.

    ``normalize_gain=False`` reproduces shipped linear-v1 metadata semantics,
    including its absolute all-noise threshold. Hybrid-v3 supplies an already
    peak-normalized input, so routing is invariant to finite non-zero global
    gain while retaining the same bins at normal capture scales.
    """
    iq = _as_finite_complex_1d(x)
    nfft = _validate_nfft(nfft)
    if normalize_gain:
        iq, peak = _peak_normalized(iq)
        if peak == 0.0:
            return 0.0, FULL_BAND_BW

    psd = _smooth(welch_psd(iq, nfft), SMOOTH)
    floor = NOISE_FLOOR_SCALE * float(np.median(psd))
    sig = np.clip(psd - floor, 0.0, None)
    total = float(sig.sum())
    if total < 1e-9:
        return 0.0, FULL_BAND_BW
    cum = np.cumsum(sig) / total
    lo_i = min(int(np.searchsorted(cum, ENERGY_EDGE)), nfft - 1)
    hi_i = min(
        max(int(np.searchsorted(cum, 1.0 - ENERGY_EDGE)), lo_i + 1),
        nfft - 1,
    )
    f_lo = lo_i / nfft - 0.5
    f_hi = hi_i / nfft - 0.5
    return float(0.5 * (f_lo + f_hi)), float(
        max(f_hi - f_lo, 1.0 / nfft)
    )


def estimate_band_circular_v2(
    x: np.ndarray,
    nfft: int = NFFT,
) -> tuple[float, float]:
    """Return circular ``(centre_freq, occupied_bw)`` in cycles/sample.

    Input is peak-normalized before Welch, making every finite non-zero global
    gain equivalent. Exact zero and statistically flat noise use the explicit
    full-band fallback. The occupied interval is the shortest circular arc
    containing 99% of the guard-floor-subtracted energy.
    """
    iq = _as_finite_complex_1d(x)
    nfft = _validate_nfft(nfft)

    scaled, peak = _peak_normalized(iq)
    if peak == 0.0:
        return 0.0, FULL_BAND_BW

    psd = np.maximum(_circular_smooth(welch_psd(scaled, nfft), SMOOTH), 0.0)
    psd_total = float(psd.sum())
    if psd_total <= 0.0:
        return 0.0, FULL_BAND_BW
    if _spectral_flatness(psd) >= _white_flatness_threshold(len(iq), nfft):
        return 0.0, FULL_BAND_BW

    guard = _minimum_circular_guard(psd)
    floor = GUARD_FLOOR_SCALE * float(np.median(guard))
    excess = np.clip(psd - floor, 0.0, None)
    if float(excess.sum() / psd_total) < MIN_EXCESS_FRACTION:
        return 0.0, FULL_BAND_BW

    start, end = _shortest_circular_interval(
        excess,
        retained_fraction=1.0 - 2.0 * ENERGY_EDGE,
    )
    bw = min(max((end - start) / nfft, 1.0 / nfft), FULL_BAND_BW)
    center = _wrap_cycles(0.5 * (start + end) / nfft - 0.5)
    return center, float(bw)


def _hybrid_uses_circular(
    legacy: tuple[float, float],
    circular: tuple[float, float],
    nfft: int,
) -> bool:
    """Whether circular-v2 has enough evidence to override linear-v1."""
    circular_center, circular_bw = circular
    seam_limit = 0.5 - HYBRID_SEAM_TOLERANCE_BINS / nfft
    crosses_seam = abs(circular_center) + 0.5 * circular_bw >= seam_limit
    return bool(
        crosses_seam
        and (
            legacy[1] >= HYBRID_LEGACY_WIDE_BW
            or circular_bw >= HYBRID_CIRCULAR_BROAD_BW
        )
    )


def estimate_band(x: np.ndarray, nfft: int = NFFT) -> tuple[float, float]:
    """Hybrid-v3 occupied-band estimate.

    Compute both gain-normalized candidates. Preserve linear-v1 on ordinary
    spectra and route to circular-v2 only for a credible seam crossing (or its
    explicit zero/white-noise abstention).
    """
    iq = _as_finite_complex_1d(x)
    nfft = _validate_nfft(nfft)
    scaled, peak = _peak_normalized(iq)
    if peak == 0.0:
        return 0.0, FULL_BAND_BW
    legacy = estimate_band_linear_v1(
        scaled,
        nfft,
        normalize_gain=False,
    )
    circular = estimate_band_circular_v2(scaled, nfft)
    # This is a common estimator abstention, not a circular routing decision.
    # Preserve the explicit zero/white-noise full-band contract before applying
    # the seam-only override.
    if circular == (0.0, FULL_BAND_BW):
        return circular
    return circular if _hybrid_uses_circular(legacy, circular, nfft) else legacy


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
    force_center: float | None = None,
    force_bw: float | None = None,
    force_resample_frac: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Full front-end. Returns (normalised complex I/Q[l_out], context)."""
    x_in = _as_finite_complex_1d(iq)
    if not isinstance(l_out, (int, np.integer)) or int(l_out) <= 0:
        raise ValueError("l_out must be a positive integer")
    l_out = int(l_out)
    if not np.isfinite(target_frac) or target_frac <= 0.0:
        raise ValueError("target_frac must be finite and positive")
    if scale_jitter < 0.0 or not np.isfinite(scale_jitter):
        raise ValueError("scale_jitter must be finite and non-negative")

    if force_center is None or force_bw is None:
        detected_center, detected_bw = estimate_band(x_in, nfft)
    else:
        detected_center, detected_bw = 0.0, FULL_BAND_BW
    center = detected_center if force_center is None else float(force_center)
    bw = detected_bw if force_bw is None else float(force_bw)
    if not np.isfinite(center):
        raise ValueError("center must be finite")
    if not np.isfinite(bw) or bw <= 0.0:
        raise ValueError("bandwidth must be finite and positive")
    center = _wrap_cycles(center)

    # down-convert measured centre to DC (frequency-invariance)
    x, _ = _peak_normalized(x_in)
    n = np.arange(len(x), dtype=np.float64)
    x = x * np.exp(-1j * 2.0 * np.pi * center * n)

    # resample so occupied bandwidth hits the canonical target (scale-invariance).
    # scale_jitter models bandwidth-estimate error so the embedding tolerates it.
    frac = bw if force_resample_frac is None else float(force_resample_frac)
    if force_resample_frac is None and scale_jitter > 0.0 and rng is not None:
        frac = frac * (1.0 + rng.uniform(-scale_jitter, scale_jitter))
    if not np.isfinite(frac) or frac <= 0.0:
        raise ValueError("resample fraction must be finite and positive")
    frac = float(np.clip(frac, 1e-3, FULL_BAND_BW))
    ratio = frac / target_frac
    new_len = int(max(64, round(len(x) * ratio)))
    x = lin_resample(x, new_len)

    # fit to canonical length and amplitude-normalise (power-invariance)
    x = center_fit(x, l_out)
    x, peak = _peak_normalized(x)
    if peak == 0.0:
        x = np.zeros(x.shape, dtype=np.complex64)
    else:
        rms = float(np.sqrt(np.mean(np.abs(x) ** 2)))
        if rms == 0.0 or not np.isfinite(rms):
            raise ValueError("cannot normalize a non-zero capture with invalid RMS")
        x = (x / rms).astype(np.complex64)

    return x, {"center": center, "bw": bw, "resample_frac": frac}


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
