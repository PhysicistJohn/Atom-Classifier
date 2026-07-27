"""Priority runner with per-trial isolation, hard timeouts and a liveness watchdog.

WHY THIS REPLACES THE IN-PROCESS CAMPAIGN LOOP. The weekend campaign ran every config
inside one long-lived process. On 2026-07-24 that process stalled at 0% CPU while the
machine was in heavy swap (25.5 of 26.6 GB used), never recovered, and sat idle for two
days producing nothing. The guards it had -- disk space and a wall-clock budget -- could
not see a hang, because the process was still alive and the budget clock was still
sensible. Three runs came out of a whole weekend.

Fixes, all structural rather than hopeful:
  1. Each trial runs in its OWN subprocess with a hard timeout. A hang costs that trial,
     not the night; the runner kills it and moves to the next.
  2. Free memory is checked before each launch, and the runner waits (rather than piling
     on) if the machine is already under pressure -- that pressure is what caused the hang.
  3. Results are appended by the child immediately on completion, so a later crash cannot
     lose earlier work, and the whole list is resumable by tag.
  4. Configs are ordered by INFORMATION VALUE, not by convenience, because the runner may
     be stopped at any point. The controls come first: an experiment whose control never
     ran cannot be interpreted at all.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

_V2 = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(_V2, "artifacts", "campaign3")
RESULTS = os.path.join(ART, "results.jsonl")
PY = "/Users/johnelliott/PersonalGitHub/Atom-Classifier/.venv-training/bin/python"
LOG = os.path.dirname(os.path.abspath(__file__))

# Per-trial ceiling, per MODE -- a single 45 min value was a sizing error: the U-Net runs
# ~3.4 s/episode at 16384 samples, so 700 episodes is ~40 min of training alone, landing
# essentially ON the timeout and killing healthy trials. Budgets below are measured
# rate x episodes, plus prep and eval, plus ~50% headroom.
TIMEOUT_BY_MODE = {"unet": 80 * 60, "equalizer": 60 * 60, "multitask": 80 * 60,
                   "plain": 25 * 60, "openset": 30 * 60}
TIMEOUT_S = 80 * 60
MIN_FREE_PCT = 25


def free_pct() -> int:
    try:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=20).stdout
        for line in out.splitlines():
            if "free percentage" in line:
                return int(line.strip().split(":")[1].strip().rstrip("%"))
    except Exception:
        pass
    return 100


def done_tags():
    if not os.path.exists(RESULTS):
        return set()
    return {json.loads(l)["tag"] for l in open(RESULTS) if l.strip()}


# Ordered by information value. Controls first -- see note 4 above.
PLAN = [
    # The architecture question the user asked for first. full vs CONTROL is the only
    # pair that can say whether the multi-task objective earns its parameters.
    ("c3_unet_CONTROL_classonly", "unet", dict(w_rec=0.0, w_prof=0.0, w_par=0.0), 700),
    ("c3_unet_full",              "unet", dict(w_rec=1.0, w_prof=0.3, w_par=0.3), 700),
    # Does the reconstruction term specifically do the work, or the aux labels?
    ("c3_unet_recon_only",        "unet", dict(w_rec=1.0, w_prof=0.0, w_par=0.0), 700),
    ("c3_unet_params_only",       "unet", dict(w_rec=0.0, w_prof=0.3, w_par=0.3), 700),
    # Equalizer axis: 0.0 already measured (closed 0.7822, chirp 0.892, coh 0.300).
    # Two more points bracket it; 0.25 was mid-flight at coh 0.3905 when the hang hit.
    ("sweep_eq_lambda_eq0.5",     "equalizer", dict(lambda_eq=0.5), 700),
    ("sweep_eq_lambda_eq2.0",     "equalizer", dict(lambda_eq=2.0), 700),
    # Cheap CNN-side runs that still have open questions on this corpus.
    ("c3_cnn_embed64",            "plain", {}, 4000),
    ("c3_oe_m2",                  "openset", dict(lambda_out=1.0, margin=2.0), 4000),
]

CHILD = '''
import sys, json, os, threading, time
sys.path.insert(0, {v2!r})
import torch
import campaign3 as c3

def _heartbeat():
    """Every 30s record wall time and MPS allocation. If a trial wedges, this says
    whether the GPU allocation is still moving (slow) or frozen (wedged) -- the
    distinction the weekend failure could not be diagnosed without."""
    t0 = time.time()
    while True:
        try:
            alloc = torch.mps.current_allocated_memory() / 1e9 if torch.backends.mps.is_available() else -1
        except Exception:
            alloc = -2
        print(f"HEARTBEAT t={{time.time()-t0:.0f}}s mps_alloc={{alloc:.3f}}GB", flush=True)
        time.sleep(30)

threading.Thread(target=_heartbeat, daemon=True).start()
tag, mode, ep = {tag!r}, {mode!r}, {ep!r}
kw = {kw!r}
key = {{"unet": "mt", "multitask": "mt", "equalizer": "eq", "openset": "oe"}}.get(mode)
extra = {{key: kw}} if key else {{}}
bb = "cnn"
r = c3.run_one(tag, backbone=bb, net_params={{"embed_dim": 64}} if tag.endswith("embed64") else {{}},
               episodes=ep, mode=mode, **extra)
with open({res!r}, "a") as f:
    f.write(json.dumps(r) + "\\n")
print("CHILD_OK", tag, round(r["closed_overall"], 4), r["auroc_chirp_HELDOUT"], flush=True)
'''


def main():
    print(f"[prio] start {time.strftime('%H:%M:%S')} free_mem={free_pct()}%", flush=True)
    for tag, mode, kw, ep in PLAN:
        if tag in done_tags():
            print(f"[prio] skip {tag} (done)", flush=True)
            continue
        waited = 0
        while free_pct() < MIN_FREE_PCT and waited < 900:
            print(f"[prio] memory {free_pct()}% < {MIN_FREE_PCT}%, waiting", flush=True)
            time.sleep(60); waited += 60
        code = CHILD.format(v2=_V2, tag=tag, mode=mode, ep=ep, kw=kw, res=RESULTS)
        print(f"[prio] === {tag} ({mode}, {ep} ep) {time.strftime('%H:%M:%S')} "
              f"free={free_pct()}% ===", flush=True)
        t0 = time.time()
        trial_log = os.path.join(ART, f"_{tag}.log")
        try:
            with open(trial_log, "w") as lf:
                p = subprocess.run([PY, "-u", "-c", code], timeout=TIMEOUT_BY_MODE.get(mode, TIMEOUT_S),
                                   stdout=lf, stderr=subprocess.STDOUT, text=True)
            tail = "\n".join(open(trial_log).read().strip().splitlines()[-3:])
            print(f"[prio] {tag} rc={p.returncode} {time.time()-t0:.0f}s (log {trial_log})\n{tail}",
                  flush=True)
        except subprocess.TimeoutExpired:
            # Record WHY it timed out, so a recurring wedge is distinguishable from slowness.
            beats = [l for l in open(trial_log).read().splitlines() if l.startswith("HEARTBEAT")]
            print(f"[prio] *** {tag} TIMED OUT after {TIMEOUT_BY_MODE.get(mode, TIMEOUT_S)}s -- killed, moving on ***",
                  flush=True)
            print(f"[prio] last 3 heartbeats: {beats[-3:] if beats else 'NONE (died before first beat)'}",
                  flush=True)
        except Exception as e:
            print(f"[prio] *** {tag} FAILED: {e} ***", flush=True)
    print(f"[prio] COMPLETE {time.strftime('%H:%M:%S')} -- {len(done_tags())} total runs",
          flush=True)


if __name__ == "__main__":
    main()
