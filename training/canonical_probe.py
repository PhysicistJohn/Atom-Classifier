"""Score the embedding on textbook signals built OUTSIDE the training pipeline.

WHY THIS EXISTS. On 2026-07-25 a retrain on the corrected corpus scored 0.890 closed-set,
1.000 per-class on cw, and plausible open-set AUROC -- and was broadly non-functional. It
called a pure carrier gsm, and mapped a carrier, an AM tone and wideband FM to nearly the
same direction in embedding space (pairwise cosine 0.925, versus 0.400 for the model it
replaced). Every metric used to approve it was computed on draws from the same generator
the model trained on, so the corpus was simultaneously the optimisation target and the
measuring instrument. No in-distribution metric can see a model that has learned "corpus
cw" instead of "cw" -- the failure only appears on inputs the corpus does not contain.

The collapse also defeats the unknown-threshold, which is why it must be caught here
rather than papered over downstream: collapse pulls unfamiliar input TOWARD the
prototypes (mean nearest-prototype distance fell 0.188 -> 0.108), so open-set rejection
fires LESS on exactly the inputs it should reject most. Tightening the threshold cannot
recover a collapsed space.

So this module shares no code with the corpus generator and no parameters with training.
It builds a carrier, an AM tone and an FM tone from their definitions and asks the only
question the corpus cannot: does the model still know what these are?

Run:  .venv-training/bin/python training/canonical_probe.py [assets_dir]
Exits non-zero if the model falls below the bars in GATE -- run it after any retrain,
before the assets are copied into src/embedding/assets.
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

# Bars calibrated on the two measured models: the shipped one scores 16/19 at cos 0.609,
# the collapsed retrain 0/19 at cos 0.955. The cos bar sits between them with margin on
# both sides -- close enough to 0.609 to catch real degradation, far enough that ordinary
# run-to-run variation in a healthy retrain will not trip it.
GATE = {"min_correct_frac": 0.75, "max_mean_pairwise_cos": 0.78}
N = 4096


def awgn(s, snr_db, rng):
    if snr_db is None:
        return s
    npow = np.mean(np.abs(s) ** 2) / (10 ** (snr_db / 10))
    return s + np.sqrt(npow / 2) * (rng.standard_normal(len(s)) + 1j * rng.standard_normal(len(s)))


def build_probes(rng):
    """Textbook definitions only. Nothing here imports the corpus generator."""
    t = np.arange(N)
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


def score(assets_dir, verbose=True):
    w = json.load(open(os.path.join(assets_dir, "embedding-weights.json")))
    pj = json.load(open(os.path.join(assets_dir, "prototypes.json")))
    protos = np.asarray(pj["prototypes"], np.float32)
    classes, thr = pj["classes"], float(pj["unknown_threshold"])

    probes = build_probes(np.random.default_rng(7))
    embs, hits = [], 0
    if verbose:
        print(f"=== {assets_dir} (threshold {thr:.4f}) ===")
    for label, want, iq in probes:
        x, _ = pp.preprocess(np.asarray(iq, np.complex128))
        e = np_forward(pp.to_channels(x), w)
        embs.append(e)
        d = np.sum((protos - e) ** 2, axis=1)
        o = np.argsort(d)
        got = "unknown" if d[o[0]] > thr else classes[o[0]]
        hits += got == want
        if verbose:
            top3 = {classes[i]: round(float(d[i]), 3) for i in o[:3]}
            print(f"  {'OK ' if got == want else 'FAIL'} {label:<20} want {want:<3} got {got:<9} {top3}")

    E = np.stack(embs)
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    iu = np.triu_indices(len(E), 1)
    cos = float((E @ E.T)[iu].mean())
    frac = hits / len(probes)
    if verbose:
        print(f"  -> {hits}/{len(probes)} correct ({frac*100:.0f}%), "
              f"probe-embedding mean pairwise cos {cos:+.3f}")
        print(f"     gates: correct >= {GATE['min_correct_frac']:.2f}, "
              f"cos <= {GATE['max_mean_pairwise_cos']:.2f}")
    return frac, cos


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else LIVE
    frac, cos = score(d)
    fails = []
    if frac < GATE["min_correct_frac"]:
        fails.append(f"only {frac*100:.0f}% of canonical probes correct")
    if cos > GATE["max_mean_pairwise_cos"]:
        fails.append(f"embedding collapse: probes at mean pairwise cos {cos:+.3f}")
    if fails:
        print("\nCANONICAL PROBE FAILED -- do not ship:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("\ncanonical probe PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
