"""Native-bandwidth preprocessing: identical to training/preprocess.py's
preprocess() EXCEPT it never calls lin_resample. Reuses estimate_band,
center_fit, to_channels, iq_features from preprocess.py verbatim (imported,
not reimplemented) since those steps are pipeline-agnostic.

This is the "no resampling crutch" condition: the neural network sees the
signal's own native occupied-bandwidth fraction and native sample-rate-
relative scale, undisturbed by the canonical-bandwidth normalization step
training/preprocess.py's CNN-oriented pipeline performs.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_V2_DIR = os.path.dirname(os.path.abspath(__file__))
_ZPAB_DIR = os.path.dirname(_V2_DIR)
_TRAINING_DIR = os.path.dirname(_ZPAB_DIR)
for p in (_TRAINING_DIR, _ZPAB_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import preprocess as pp  # noqa: E402 -- imported, never modified

L_OUT_NATIVE = 16384
# == corpus.json's global sampleCount, so center_fit is a structural no-op (the
# n==length branch) for every item and the capture reaches the network intact.
#
# 4096 -> 16384 (2026-07-24), tracking the corpus regeneration. This constant MUST
# equal the corpus sampleCount. If it is left below it, center_fit CROPS, and the
# crop discards exactly the thing the corpus fix exists to provide: at 16384 samples
# a Bluetooth capture spans ~1.3 ms (~2 x 625us slots) and a GSM capture ~4-16 ms
# (several 577us bursts / 4.615ms frames), so cropping to 4096 would throw away
# three quarters of the burst structure and silently reinstate the ceiling.
# Downsampling instead of cropping is NOT a safe alternative here: measured
# fractional bandwidths after the fix run 0.11-0.54 (bluetooth's hopping spreads it
# widest), so a 4x decimation would push several classes past Nyquist and alias.


def preprocess(iq: np.ndarray, l_out: int = L_OUT_NATIVE, nfft: int = pp.NFFT,
               scale_jitter: float = 0.0, rng: np.random.Generator | None = None,
               force_center: float | None = None
               ) -> tuple[np.ndarray, dict]:
    """Full native-bandwidth front-end. Returns (normalised complex I/Q[l_out], context)."""
    center, bw = pp.estimate_band(iq, nfft)
    # force_center exists for CLEAN/IMPAIRED PAIRS. Each member otherwise estimates its
    # own centre, and noise/multipath move the spectral centroid, so the two get
    # downconverted by slightly different amounts. Over a 16384-sample window even a tiny
    # residual frequency error decorrelates them completely: measured own-pair coherence
    # fell from 0.081 (raw) to 0.0007 (after independent preprocessing), i.e. the
    # reconstruction target was destroyed by the front end rather than by the channel.
    # Passing the impaired member's centre keeps the pair phase-aligned.
    if force_center is not None:
        center = force_center

    # down-convert measured centre to DC only -- no resample to a canonical
    # occupied-bandwidth fraction (that step exists solely to make the CNN's
    # fixed-topology conv stack work; it is deliberately absent here).
    n = np.arange(len(iq))
    x = iq * np.exp(-1j * 2.0 * np.pi * center * n)
    # NOTE: no lin_resample call -- no canonical-bandwidth normalization.
    # scale_jitter is accepted for call-signature parity with pp.preprocess
    # but has no effect: there is no resample ratio here to perturb. The
    # role scale_jitter played in v1 (teaching tolerance to bandwidth-estimate
    # error) is instead subsumed by genuine corpus diversity across the many
    # native sample rates / bandwidths, which this module deliberately does
    # not erase.

    x = pp.center_fit(x, l_out)
    rms = np.sqrt(np.mean(np.abs(x) ** 2) + 1e-12)
    x = (x / rms).astype(np.complex64)

    return x, {"center": center, "bw": bw}


to_channels = pp.to_channels
iq_features = pp.iq_features
