"""Open-set evaluation for the v2 pipeline: does the nearest-prototype distance
separate KNOWN signals from NOVELTY (band-limited noise, chirp/LFM)?

WHY THIS EXISTS. Every measurement in the v2 architecture search so far has been
closed-set: "given that a signal is present and is one of the 7 classes, which is
it?" But DESIGN.md treats open-set rejection as a safety requirement, not a nice-
to-have:
  line 23  "reports 'unknown' honestly when a signal is far from every prototype"
  line 68  "the open-set 'unknown' valve is the safety net when it fails"
  line 98  "Open-set novelty (never enrolled; must read as unknown): band-limited
            noise, and a chirp/LFM never shown in training"
  line 133 acceptance gate: Open-set AUROC >= 0.72
Optimizing closed-set accuracy alone can silently degrade that valve, so the
weekend campaign scores both.

This also tests a specific prediction of the corpus-bug diagnosis. Before the fix,
41% of clean bluetooth and 36% of clean dsss windows contained NO emission yet
carried a modulation label. Prototypes built from such windows sit near the noise
region of embedding space, so genuine noise cannot be pushed away from them -- which
predicts poor noise rejection. The shipped model's own report shows exactly that:
flagged_unknown_noise = 0.383 at auroc_noise = 0.826. If the diagnosis is right, a
model trained on the occupancy-gated corpus should reject noise substantially better.

Novelty is generated here rather than added to corpus.json on purpose: novelty must
never be enrollable or trainable, and keeping it out of the corpus makes that
structural rather than a convention someone can forget.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_V2_DIR = os.path.dirname(os.path.abspath(__file__))
_ZPAB_DIR = os.path.dirname(_V2_DIR)
_TRAINING_DIR = os.path.dirname(_ZPAB_DIR)
for p in (_TRAINING_DIR, _ZPAB_DIR, _V2_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import preprocess as pp  # noqa: E402
import native_preprocess as npp  # noqa: E402
from train import embed_all, nearest, auroc  # noqa: E402 -- auroc reused from the shipped evaluator


def _bandlimited_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Complex noise band-limited to a random fraction of the sampled band."""
    frac = rng.uniform(0.05, 0.6)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    X = np.fft.fft(x)
    keep = int(max(4, frac * n / 2))
    mask = np.zeros(n, dtype=bool)
    mask[:keep] = True
    mask[-keep:] = True
    X[~mask] = 0
    y = np.fft.ifft(X)
    # random centre offset so it is not always at DC
    k = np.arange(n)
    y = y * np.exp(1j * 2 * np.pi * rng.uniform(-0.3, 0.3) * k)
    return y


def _chirp(n: int, rng: np.random.Generator) -> np.ndarray:
    """Linear FM sweep, never shown in training."""
    f0 = rng.uniform(-0.35, -0.05)
    f1 = rng.uniform(0.05, 0.35)
    k = np.arange(n)
    phase = 2 * np.pi * (f0 * k + 0.5 * (f1 - f0) / n * k ** 2)
    y = np.exp(1j * phase)
    y = y + (10 ** (-rng.uniform(10, 30) / 20)) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return y


NOVELTY = {"noise": _bandlimited_noise, "chirp": _chirp}


def build_novelty(condition: str, n_each: int = 300, seed: int = 424243,
                  in_len: int | None = None, raw_len: int | None = None):
    """Returns {name: (x[N,2,L], feat[N,F])} preprocessed by the SAME front end the
    condition's known samples went through -- otherwise the AUROC would measure a
    preprocessing difference rather than a novelty difference."""
    module = npp if condition == "native" else pp
    l_out = in_len if in_len is not None else (npp.L_OUT_NATIVE if condition == "native" else pp.L_OUT)
    # Novelty must begin with the same capture geometry as known corpus rows. In the
    # resampled condition the network input is 1024 samples, but the raw corpus capture is
    # 16384 samples; generating only 1024 raw samples changes Welch averaging, resampling,
    # and time support before novelty is even measured.
    n_raw = int(raw_len if raw_len is not None else l_out)
    if n_raw <= 0 or l_out <= 0:
        raise ValueError("raw_len and in_len must be positive")
    rng = np.random.default_rng(seed)
    out = {}
    for name, gen in NOVELTY.items():
        xs, fs = [], []
        for _ in range(n_each):
            raw = gen(n_raw, rng)
            norm, _ = module.preprocess(raw, l_out=l_out)
            xs.append(pp.to_channels(norm))
            fs.append(pp.iq_features(norm))
        out[name] = (np.stack(xs).astype(np.float32), np.stack(fs).astype(np.float32))
    return out


def evaluate_openset(net, dev, data, protos, fmean=None, fstd=None, n_each: int = 300,
                     calibration_data=None, novelty_raw_len: int | None = None):
    """Nearest-prototype distance as the novelty score (higher = more novel), exactly
    the shipped rule in evaluate.py. Returns AUROC per novelty type plus the
    known-side distance distribution needed to set an operating threshold."""
    condition = data.get("condition", "native")
    in_len = data["xva"].shape[-1]
    nov = build_novelty(
        condition, n_each=n_each, in_len=in_len,
        raw_len=novelty_raw_len,
    )

    fmean = data["fmean"] if fmean is None else fmean
    fstd = data["fstd"] if fstd is None else fstd

    emb_va = embed_all(net, data["xva"], data["fva"], dev)
    _, d_known = nearest(emb_va, protos)
    nn_known = d_known.min(axis=1)
    calibration_data = data if calibration_data is None else calibration_data
    emb_cal = embed_all(net, calibration_data["xva"], calibration_data["fva"], dev)
    _, d_cal = nearest(emb_cal, protos)
    nn_cal = d_cal.min(axis=1)

    report = {}
    all_scores = []
    for name, (xn, fn) in nov.items():
        fn_z = (fn - fmean) / fstd          # same normalization the known pools got
        emb_n = embed_all(net, xn, fn_z, dev)
        _, d_nov = nearest(emb_n, protos)
        sc = d_nov.min(axis=1)
        all_scores.append(sc)
        report[f"auroc_{name}"] = round(float(auroc(sc, nn_known)), 4)
        report[f"median_dist_{name}"] = round(float(np.median(sc)), 4)
    report["auroc_overall"] = round(float(auroc(np.concatenate(all_scores), nn_known)), 4)
    report["median_dist_known"] = round(float(np.median(nn_known)), 4)
    # Calibrate on the model-selection split, then report the false-unknown rate on this
    # function's untouched known test rows. With no separate calibration_data the legacy
    # same-split behavior remains available, but corrected runs always pass one.
    thr = float(np.percentile(nn_cal, 95))
    report["threshold_p95_known"] = round(thr, 4)
    report["known_false_unknown_rate"] = round(float((nn_known > thr).mean()), 4)
    report["calibration_false_unknown_rate"] = round(float((nn_cal > thr).mean()), 4)
    report["n_known_test"] = int(len(nn_known))
    report["n_known_calibration"] = int(len(nn_cal))
    report["novelty_raw_length"] = int(
        novelty_raw_len if novelty_raw_len is not None else in_len)
    report["network_input_length"] = int(in_len)
    for name, sc in zip(NOVELTY, all_scores):
        report[f"flagged_unknown_{name}"] = round(float((sc > thr).mean()), 4)
    return report
