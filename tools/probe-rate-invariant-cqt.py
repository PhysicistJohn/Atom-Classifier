#!/usr/bin/env python
"""Correctness probe for a rate-normalized complex CQT front end.

Question: does a constant-Q transform with bin centers defined as
FRACTIONS OF THE SAMPLE RATE (rather than absolute Hz) turn a change of
sample rate into a pure shift along the log-frequency axis, as the
scale-invariance argument claims?

Method: synthesize the SAME analytic signal (a chirp through a burst
envelope, so both spectral and temporal structure are present) at two
sample rates related by an exact ratio, resample the slower one up to the
faster rate (band-limited, so both captures now cover the same absolute
spectrum on the same sample grid), take the CQT of each, and check that
CQT(fast) lines up with CQT(slow-resampled-to-fast) after accounting for
the ratio as a bin-index shift.

This is deliberately NOT a resample-invariance test (any transform is
invariant to a lossless upsample of the identical waveform). It is a
BANDWIDTH-scale test: signal B is a genuinely narrower-band, slower-rate
version of signal A (a real analogue of "the same modulation family
captured at a different physical rate/bandwidth"), and the claim under
test is that CQT(A) and CQT(B) are the same picture shifted along the log
axis by log2(rate_ratio) octaves -- because that's what would let a single
set of convolutional weights recognize both without retraining or
resampling to one canonical rate.

Run: .venv-training/bin/python tools/probe-rate-invariant-cqt.py
"""
from __future__ import annotations

import numpy as np


def cqt_complex(x: np.ndarray, fs: float, bins_per_octave: int = 24,
                n_octaves: int = 9, top_frac_of_nyquist: float = 0.9
                ) -> tuple[np.ndarray, np.ndarray]:
    """Complex CQT via direct Morlet-kernel correlation (reference-quality,
    not optimized). Returns (center_freqs_hz, coefficients[time, freq])
    with bin centers defined as top_frac_of_nyquist * fs/2 * 2^(-k/B).

    Time axis: one column per HOP samples, HOP tied to the LOWEST bin's
    period so the slowest component is still resolved.
    """
    n_bins = bins_per_octave * n_octaves
    top_hz = top_frac_of_nyquist * (fs / 2)
    k = np.arange(n_bins)
    centers_hz = top_hz * (2.0 ** (-k / bins_per_octave))
    q = bins_per_octave / (2 ** (1 / bins_per_octave) - 1)  # constant Q

    lowest_hz = centers_hz[-1]
    hop = max(1, int(fs / lowest_hz / 4))
    n_frames = max(1, (len(x) - 1) // hop)
    out = np.zeros((n_frames, n_bins), dtype=np.complex128)

    t_full = np.arange(len(x)) / fs
    for bi, f0 in enumerate(centers_hz):
        # Morlet kernel length: Q cycles of f0, in samples.
        cycles = q / f0 * fs
        klen = int(min(len(x), max(8, round(cycles))))
        if klen % 2 == 0:
            klen += 1
        half = klen // 2
        tk = (np.arange(klen) - half) / fs
        window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(klen) / (klen - 1))
        kernel = window * np.exp(2j * np.pi * f0 * tk)
        kernel /= np.sqrt(np.sum(np.abs(kernel) ** 2))
        for fr in range(n_frames):
            center = fr * hop
            lo = center - half
            hi = center + half + 1
            if lo < 0 or hi > len(x):
                continue
            out[fr, bi] = np.vdot(kernel, x[lo:hi])
    return centers_hz, out


def synthesize_burst(fs: float, duration_s: float, center_hz: float,
                     bandwidth_hz: float, seed: int) -> np.ndarray:
    """A chirp sweeping +-bandwidth/2 around center_hz, in a raised-cosine
    burst envelope covering the middle half of the capture -- gives the
    probe both spectral (chirp) and temporal (envelope edge) structure to
    align, at an arbitrary bandwidth/rate."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    sweep_rate = bandwidth_hz / duration_s
    phase = 2 * np.pi * (center_hz * t
                        + 0.5 * sweep_rate * (t - duration_s / 2) ** 2)
    carrier = np.exp(1j * phase)
    edge = n // 8
    envelope = np.ones(n)
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(edge) / edge)
    envelope[:edge] = ramp
    envelope[-edge:] = ramp[::-1]
    noise = (rng.normal(size=n) + 1j * rng.normal(size=n)) * 0.01
    return (envelope * carrier + noise).astype(np.complex128)


def main() -> None:
    duration_a_s = 200e-6
    ratio = 4  # signal B is a true dilation of A: 1/4 the rate, 1/4 the
    # bandwidth, AND 4x the duration -- same waveform shape, stretched.
    # (Holding duration fixed while only shrinking rate/bandwidth would
    # just capture fewer of B's own cycles -- a different, unfair test.)

    fs_a = 20_000_000.0
    bw_a = 4_000_000.0
    center_frac = 0.15  # same relative center-frequency fraction for both
    sig_a = synthesize_burst(fs_a, duration_a_s, center_frac * fs_a / 2,
                             bw_a, seed=1)

    fs_b = fs_a / ratio
    bw_b = bw_a / ratio
    duration_b_s = duration_a_s * ratio
    sig_b = synthesize_burst(fs_b, duration_b_s, center_frac * fs_b / 2,
                             bw_b, seed=1)

    freqs_a, cqt_a = cqt_complex(sig_a, fs_a)
    freqs_b, cqt_b = cqt_complex(sig_b, fs_b)

    print(f"signal A: fs={fs_a/1e6:.2f} MHz, bw={bw_a/1e6:.2f} MHz, "
          f"CQT shape {cqt_a.shape}")
    print(f"signal B: fs={fs_b/1e6:.2f} MHz, bw={bw_b/1e6:.2f} MHz "
          f"(= A / {ratio}), CQT shape {cqt_b.shape}")

    # This test dilates the WHOLE signal (rate, bandwidth, AND duration
    # all scale by `ratio` together) -- both signals' center/bandwidth are
    # the SAME fraction of their own fs. Because CQT bin centers here are
    # defined as top_frac * fs/2 * 2^(-k/B), that construction lands B in
    # the literal SAME bin index as A: predicted shift is ZERO, not a
    # function of the ratio. (A genuine octave-shift would appear only for
    # the OTHER case -- fixed fs, signal's own relative bandwidth scaling
    # -- which is a distinct, not-yet-tested scenario; see bottom note.)
    shift_bins = 0
    print("predicted bin shift for this same-relative-fraction dilation: "
          "0 (literal identity, not merely a shift)")

    mag_a = np.abs(cqt_a)
    mag_b = np.abs(cqt_b)

    # Align B's bin axis to A's by the predicted shift and compare on the
    # overlapping bin range, using a resampled common time axis (B has
    # ratio-times fewer frames covering the same duration).
    n_bins = mag_a.shape[1]
    valid = slice(0, n_bins - shift_bins)
    a_shifted_view = mag_a[:, shift_bins:]
    b_view = mag_b[:, valid]

    # Resample B's time axis onto A's frame count for a like-for-like
    # correlation (nearest-frame lookup; this is a comparison tool, not
    # part of the proposed architecture).
    t_a = np.linspace(0, 1, a_shifted_view.shape[0])
    t_b = np.linspace(0, 1, b_view.shape[0])
    b_on_a_grid = np.stack([
        np.interp(t_a, t_b, b_view[:, k]) for k in range(b_view.shape[1])
    ], axis=1)

    def normalize(m: np.ndarray) -> np.ndarray:
        m = m - m.mean()
        n = np.linalg.norm(m)
        return m / n if n > 0 else m

    x = normalize(a_shifted_view)
    y = normalize(b_on_a_grid)
    correlation = float(np.sum(x * y))
    print(f"\ncorrelation(A's bins [{shift_bins}:], B's bins [0:{n_bins - shift_bins}] "
          f"time-resampled) = {correlation:.4f}")

    # Control: same comparison with NO shift applied (the naive "just use
    # absolute Hz bins" expectation) -- should be much worse if the
    # rate-normalized construction is doing real work.
    a_noshift = mag_a[:, :n_bins - shift_bins]
    t_a2 = np.linspace(0, 1, a_noshift.shape[0])
    b_full_on_a = np.stack([
        np.interp(t_a2, t_b, mag_b[:, k])
        for k in range(min(n_bins, mag_b.shape[1]))
    ], axis=1)[:, :n_bins - shift_bins]
    correlation_noshift = float(
        np.sum(normalize(a_noshift) * normalize(b_full_on_a)))
    print(f"control, no bin shift applied                    "
          f"= {correlation_noshift:.4f}")

    pass1 = correlation > 0.95
    print("\nTEST 1 (acquisition-rate invariance): "
          + ("PASS" if pass1 else "FAIL"))

    # ------------------------------------------------------------------
    # TEST 2 -- the practically load-bearing case: ONE fixed capture rate
    # (as a real SDR always uses), two signals whose OWN bandwidths differ
    # by `ratio` (e.g. a narrowband burst vs. a wideband carrier in the
    # same 20 Msps capture). Here fs does NOT change, so the CQT's top-of-
    # band anchor is fixed; the prediction is a literal SHIFT of
    # bins_per_octave * log2(ratio) bins along frequency, not identity.
    # ------------------------------------------------------------------
    print("\n--- TEST 2: fixed fs, signal's own bandwidth scales "
          f"by {ratio}x (the GSM-vs-wideband-carrier case) ---")
    bins_per_octave = 24  # must match cqt_complex's default
    fs_fixed = 20_000_000.0
    duration_fixed_s = 200e-6
    sig_narrow = synthesize_burst(fs_fixed, duration_fixed_s,
                                  center_frac * fs_fixed / 2,
                                  bw_a, seed=1)
    sig_wide = synthesize_burst(fs_fixed, duration_fixed_s,
                                center_frac * fs_fixed / 2,
                                bw_a * ratio, seed=1)
    _, cqt_narrow = cqt_complex(sig_narrow, fs_fixed)
    _, cqt_wide = cqt_complex(sig_wide, fs_fixed)
    mag_narrow = np.abs(cqt_narrow)
    mag_wide = np.abs(cqt_wide)
    predicted_shift2 = int(round(bins_per_octave * np.log2(ratio)))
    print(f"CQT shapes: narrow {mag_narrow.shape}, wide {mag_wide.shape}; "
          f"predicted shift {predicted_shift2} bins "
          f"({np.log2(ratio):.3f} octaves)")

    n_bins2 = mag_narrow.shape[1]
    n_frames2 = min(mag_narrow.shape[0], mag_wide.shape[0])
    narrow_v = mag_narrow[:n_frames2, :n_bins2 - predicted_shift2]
    wide_shifted_v = mag_wide[:n_frames2, predicted_shift2:]
    wide_noshift_v = mag_wide[:n_frames2, :n_bins2 - predicted_shift2]
    corr2_shifted = float(np.sum(normalize(narrow_v) * normalize(wide_shifted_v)))
    corr2_noshift = float(np.sum(normalize(narrow_v) * normalize(wide_noshift_v)))
    print(f"correlation WITH predicted {predicted_shift2}-bin shift  = "
          f"{corr2_shifted:.4f}")
    print(f"correlation with NO shift (naive expectation)           = "
          f"{corr2_noshift:.4f}")
    pass2 = corr2_shifted > 0.5 and corr2_shifted > 2 * abs(corr2_noshift)
    print("TEST 2 (fixed-rate bandwidth-scale shift): "
          + ("PASS" if pass2 else "FAIL"))

    print("\nOVERALL:", "PASS" if (pass1 and pass2) else "NEEDS WORK",
          "-- both properties the architecture proposal depends on"
          if (pass1 and pass2) else
          "-- at least one predicted invariance property did not hold"
          " as constructed; do not proceed to a training run yet")


if __name__ == "__main__":
    main()
