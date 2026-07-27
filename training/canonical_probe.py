"""Score the embedding on textbook signals built OUTSIDE the training pipeline,
ACROSS CAPTURE LENGTHS.

WHY THIS EXISTS, AND WHY THE LENGTH SWEEP IS THE WHOLE POINT. On 2026-07-25 a retrain on
the rebuilt corpus scored 0.890 closed-set and 1.000 per-class on cw, yet called a pure
carrier gsm. The first version of this file scored it 0/19 and reported "representational
collapse". That diagnosis was WRONG, and the way it was wrong is the actual lesson.

This file originally hard-coded N=4096 -- copied from src/embedding/embedding-classifier
.test.ts:39, which is itself the OLD corpus's SAMPLE_COUNT. The rebuilt corpus moved
SAMPLE_COUNT to 16384. Sweeping the probe length instead of fixing it:

      N     fill    shipped         retrain
   4096    0.094    16/19  +0.609     0/19  +0.955
   8192    0.188    17/19  +0.645    10/19  +0.880
  16384    0.375     5/19  +0.703    18/19  +0.820
  32768    0.750     2/19  +0.777    10/19  +0.822

Neither model is collapsed. Both are FILL-LOCKED, in mirror image, each competent only
near its own corpus's capture length. preprocess resamples so occupied bandwidth hits
TARGET_FRAC then centre-fits to L_OUT, so the fraction of the 1024-sample network input
that carries signal is linear in raw capture length; for a narrowband emitter whose
measured bandwidth is floor-limited at ~6/512 by the Hann-512 mainlobe, fill is a pure
function of SAMPLE_COUNT. Below ~0.2 fill the conv stack sees mostly zero padding and the
12 scalar features read out 1/fill rather than modulation.

So the "independent oracle" was not independent: it silently inherited the old corpus's
geometry through a hard-coded constant, and then reported the resulting mismatch as a
property of the model. An instrument built to escape in-distribution evaluation carried a
parameter from the very distribution it was meant to escape. That is why this file now
sweeps length and gates on the WORST length, not a chosen one.

The probe is intentionally small, but it is also heavily imbalanced: 15 CW, 2 AM and 2 FM
rows. Raw accuracy is therefore only a diagnostic -- the constant prediction "cw" scores
15/19 = 0.789 without recognizing a single AM or FM signal. The deployability gate is on
balanced accuracy (mean recall over the three represented classes), never raw accuracy.

The underlying defect is upstream, and the generator names it at
tools/generate-signallab-iq-corpus.ts:111 -- captureLengthSamples "FIXED at SAMPLE_COUNT
-- not yet swept; known gap". Every other nuisance was widened and randomised; this one
was moved from one fixed point to another, which is the worst case: a distribution shift
with coverage on neither side. Until capture length is swept in the corpus, NO model will
pass the invariance bar here, and that failure is the correct output rather than something
to tune around.

Run:  .venv-training/bin/python training/canonical_probe.py [assets_dir]
Exits non-zero if the model falls below the bars in GATE.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preprocess as pp  # noqa: E402
from train import np_forward  # noqa: E402

LIVE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "embedding", "assets")

# Realistic capture lengths. A field receiver does not hand the classifier a fixed N, and
# preprocess accepts any length, so a model that only works at one is not deployable --
# it is matched to a corpus constant, which is precisely the failure this file exists to
# surface.
LENGTHS = (4096, 8192, 16384, 32768)

# Bars apply to the WORST length in the sweep, never the best. Gating on the best is how
# the first version of this file certified a fill-locked model as healthy.
#
# `min_valid_bal_acc` is not a ship bar. It is the floor a matched-condition measurement
# must clear before a flat length curve is interpretable at all: a dead representation is
# perfectly invariant. canonical_probe_net applies that validity guard because it knows the
# campaign's matched capture length. This exported-assets probe has no reliable way to infer
# which corpus length produced an arbitrary assets directory, so it reports best_bal instead.
#
# Both shipped/retrained models measured so far fail min_bal_acc. That is the honest state --
# capture length is an unswept nuisance in the corpus, so nothing trained on it can be assumed
# length-invariant. Do NOT relax these bars to make a model pass; fix the data coverage.
GATE = {
    "min_bal_acc": 0.75,
    "min_valid_bal_acc": 0.50,
    "max_mean_pairwise_cos": 0.78,
}


def awgn(s, snr_db, rng):
    if snr_db is None:
        return s
    npow = np.mean(np.abs(s) ** 2) / (10 ** (snr_db / 10))
    return s + np.sqrt(npow / 2) * (rng.standard_normal(len(s)) + 1j * rng.standard_normal(len(s)))


def build_probes(rng, n):
    """Textbook definitions only. Nothing here imports the corpus generator.

    Length is a PARAMETER, never a constant. All rates are in cycles/sample, so the
    signals are the same signals at every n -- only the capture window changes."""
    t = np.arange(n)
    out = []
    for f in (0.0117, 0.05, 0.1, 0.2, -0.15):          # the TS suite's tone, plus offsets
        for snr in (40, 20, 10):
            out.append((f"cw f={f:+.4f} snr={snr}", "cw",
                        awgn(np.exp(2j * np.pi * f * t), snr, rng)))
    for depth in (0.5, 0.9):
        out.append((f"am depth={depth}", "am",
                    awgn((1 + depth * np.cos(2 * np.pi * 0.004 * t)) * np.exp(2j * np.pi * 0.05 * t), 30, rng)))
    for dev in (0.01, 0.03):
        out.append((f"fm dev={dev}", "fm",
                    awgn(np.exp(1j * (2 * np.pi * 0.05 * t + (dev / 0.002) * np.sin(2 * np.pi * 0.002 * t))), 30, rng)))
    return out


def score_at(assets_dir, n, verbose=False):
    """Score every probe at one capture length.

    Returns (raw_accuracy, balanced_accuracy, mean_pairwise_cos, fill, per_class_recall).
    Raw accuracy is retained for continuity, but the gate must never use it because this
    probe contains 15 CW rows and only two rows for each of AM and FM.
    """
    w = json.load(open(os.path.join(assets_dir, "embedding-weights.json")))
    pj = json.load(open(os.path.join(assets_dir, "prototypes.json")))
    protos = np.asarray(pj["prototypes"], np.float32)
    classes, thr = pj["classes"], float(pj["unknown_threshold"])

    embs, wanted, predicted, fill = [], [], [], None
    for label, want, iq in build_probes(np.random.default_rng(7), n):
        x, ctx = pp.preprocess(np.asarray(iq, np.complex128))
        if fill is None:  # signal-carrying fraction of the network input, the key geometry
            fill = min(int(max(64, round(n * ctx["bw"] / pp.TARGET_FRAC))), pp.L_OUT) / pp.L_OUT
        e = np_forward(pp.to_channels(x), w)
        embs.append(e)
        d = np.sum((protos - e) ** 2, axis=1)
        o = np.argsort(d)
        got = "unknown" if d[o[0]] > thr else classes[o[0]]
        wanted.append(want)
        predicted.append(got)
        if verbose:
            top3 = {classes[i]: round(float(d[i]), 3) for i in o[:3]}
            print(f"    {'OK ' if got == want else 'FAIL'} {label:<20} want {want:<3} got {got:<9} {top3}")

    E = np.stack(embs)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    iu = np.triu_indices(len(E), 1)
    wanted = np.asarray(wanted)
    predicted = np.asarray(predicted)
    names = sorted(set(wanted.tolist()))
    per_class = {name: float(np.mean(predicted[wanted == name] == name)) for name in names}
    raw = float(np.mean(predicted == wanted))
    bal = float(np.mean(list(per_class.values())))
    return raw, bal, float((E @ E.T)[iu].mean()), fill, per_class


def score(assets_dir, verbose=True):
    """Sweep capture length. Gates on worst balanced accuracy, never CW-heavy raw accuracy."""
    rows = [(n,) + score_at(assets_dir, n) for n in LENGTHS]
    if verbose:
        print(f"=== {assets_dir} ===")
        print(f"  {'N':>7} {'fill':>6} {'raw':>9} {'balanced':>10} {'pairwise cos':>13}")
        for n, raw, bal, cos, fill, _per_class in rows:
            print(f"  {n:>7} {fill:>6.3f} {raw*100:>8.0f}% {bal*100:>9.0f}% {cos:>+13.3f}")
    worst_bal = min(r[2] for r in rows)
    worst_cos = max(r[3] for r in rows)
    best_bal = max(r[2] for r in rows)
    best_raw = max(r[1] for r in rows)
    if verbose:
        print(f"  worst-length balanced accuracy: {worst_bal*100:.0f}%, cos {worst_cos:+.3f}"
              f"   (best balanced {best_bal*100:.0f}%; best raw {best_raw*100:.0f}% -- not gated)")
        print(f"  gate on WORST balanced accuracy >= {GATE['min_bal_acc']:.2f}, "
              f"cos <= {GATE['max_mean_pairwise_cos']:.2f}")
    return worst_bal, worst_cos, best_bal


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else LIVE
    bal, cos, best = score(d)
    fails = []
    if bal < GATE["min_bal_acc"]:
        fails.append(f"worst capture length balanced accuracy is only {bal*100:.0f}% "
                     f"(best length reaches {best*100:.0f}%) -- model is fill-locked, "
                     "not length-invariant")
    if cos > GATE["max_mean_pairwise_cos"]:
        fails.append(f"probe embeddings degenerate at some length: mean pairwise cos {cos:+.3f}")
    if fails:
        print("\nCANONICAL PROBE FAILED -- do not ship:")
        for f in fails:
            print(f"  - {f}")
        print("\n  Expected until capture length is swept in the corpus generator")
        print("  (tools/generate-signallab-iq-corpus.ts:111 declares it an unswept gap).")
        print("  Fix the corpus; do not relax GATE.")
        return 1
    print("\ncanonical probe PASSED across all capture lengths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
