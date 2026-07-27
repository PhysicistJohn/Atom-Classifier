"""Short-trial hyperparameter sweep, run BEFORE committing to long runs.

WHY (user, 2026-07-24). campaign3's weight settings were a coarse hand-picked grid
(lambda_eq in {0,0.5,1,3}, w_rec in {0,1,3}, margin in {1,2,3}) evaluated inside runs
that each cost 20-60 min. That is an expensive way to search a 1-D axis, and it is not
really architecture research -- it is tuning wearing a campaign's clothes. Short trials
pin the weight cheaply, then only the winners get a long run.

WHAT MOTIVATED IT CONCRETELY. The U-Net reached coherence 0.479 while the physically
achievable ceiling implied by the corpus SNR distribution is 0.878 mean (no equalizer can
beat gamma^2 = SNR/(1+SNR); at -5 dB that is 0.36). So it is at 55% of achievable, not at
a wall -- there is real headroom, and the reconstruction weight is the obvious lever.

THE CAVEAT, HONOURED IN THE DESIGN. Short-trial rankings do not always survive to full
length: cosine LR schedules anneal over the whole budget, and configs converge at
different rates, so a weight that wins at 1k can lose at 10k. This sweep therefore
NARROWS the range rather than picking a single winner -- the top TWO settings per axis go
forward to full-length confirmation, and the sweep result is reported as a range with its
spread, not as a point estimate.

Run:  .venv-training/bin/python training/zplane_ab/v2_full_variation/sweep_hparams.py [--axis eq|unet|oe]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_V2_DIR = os.path.dirname(os.path.abspath(__file__))
_ZPAB_DIR = os.path.dirname(_V2_DIR)
_TRAINING_DIR = os.path.dirname(_ZPAB_DIR)
for p in (_TRAINING_DIR, _ZPAB_DIR, _V2_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import campaign3 as c3  # noqa: E402 -- reuse its runner, so the sweep and the campaign
                        # score identically (same eval, same openset protocol)

ART = os.path.join(_V2_DIR, "artifacts", "sweep")
os.makedirs(ART, exist_ok=True)
RESULTS = os.path.join(ART, "results.jsonl")

# Trial budgets: short enough to sweep an axis in reasonable time, long enough that the
# ranking is not pure noise. The U-Net costs ~3.4 s/episode at 16384 samples, so its
# trials are shorter than the CNN's by necessity, and that is stated in the report.
BUDGET = {"eq": 700, "unet": 400, "oe": 1500}

AXES = {
    # (mode, kwarg-name, values) -- one axis at a time, everything else at its default
    "eq":   ("equalizer", "lambda_eq", [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]),
    "unet": ("unet",      "w_rec",     [0.0, 0.5, 1.0, 2.0, 4.0]),
    "oe":   ("openset",   "margin",    [0.5, 1.0, 2.0, 4.0, 8.0]),
}


def load():
    return [json.loads(x) for x in open(RESULTS)] if os.path.exists(RESULTS) else []


def run_axis(axis: str):
    mode, key, values = AXES[axis]
    ep = BUDGET[axis]
    done = {r["tag"] for r in load()}
    print(f"[sweep] axis={axis} mode={mode} {key} in {values} @ {ep} episodes", flush=True)
    for v in values:
        tag = f"sweep_{axis}_{key}{v}"
        if tag in done:
            print(f"[sweep] skip {tag}", flush=True); continue
        kw = {}
        if mode == "equalizer":
            kw["eq"] = {key: v}
        elif mode == "unet":
            kw["mt"] = {key: v, "w_prof": 0.3, "w_par": 0.3}
        else:
            kw["oe"] = {key: v, "lambda_out": 1.0}
        t0 = time.perf_counter()
        try:
            r = c3.run_one(tag, backbone="cnn", net_params={}, episodes=ep, mode=mode, **kw)
        except Exception as e:
            print(f"[sweep] FAIL {tag}: {e}", flush=True); continue
        r["axis"], r["axis_value"] = axis, v
        with open(RESULTS, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(f"[sweep] {tag}: closed={r['closed_overall']:.4f} chirp={r['auroc_chirp_HELDOUT']:.3f} "
              f"coh={r['final_coherence']} ({time.perf_counter()-t0:.0f}s)", flush=True)


def report():
    rs = load()
    if not rs:
        print("no sweep results yet"); return
    print("\n=== SHORT-TRIAL SWEEP ===")
    print("NOTE: short-trial rankings do not always survive to full length. Use this to")
    print("narrow the range; the top TWO per axis get a full-length confirmation run.\n")
    for axis in AXES:
        sub = [r for r in rs if r.get("axis") == axis]
        if not sub:
            continue
        _, key, _ = AXES[axis]
        print(f"-- axis {axis} ({key}), {sub[0]['episodes']} episodes --")
        print(f"  {'value':>8} {'closed':>8} {'chirp':>8} {'noise':>8} {'coh':>8}")
        for r in sorted(sub, key=lambda x: x["axis_value"]):
            coh = f"{r['final_coherence']:.3f}" if r.get("final_coherence") is not None else "     -"
            print(f"  {r['axis_value']:>8} {r['closed_overall']:>8.4f} {r['auroc_chirp_HELDOUT']:>8.3f} "
                  f"{r['auroc_noise']:>8.3f} {coh:>8}")
        best = sorted(sub, key=lambda x: -x["closed_overall"])[:2]
        spread = max(x["closed_overall"] for x in sub) - min(x["closed_overall"] for x in sub)
        print(f"  -> top two by closed-set: {[b['axis_value'] for b in best]} "
              f"(spread across axis {spread:.4f})")
        if spread < 0.01:
            print("  -> WARNING: spread is within run-to-run noise; this axis may not matter")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", choices=list(AXES) + ["all"], default="all")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if not a.report_only:
        for ax in ([a.axis] if a.axis != "all" else list(AXES)):
            run_axis(ax)
    report()
