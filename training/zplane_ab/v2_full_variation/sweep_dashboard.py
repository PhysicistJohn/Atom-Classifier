"""Emit the sweep's live state as JSON for an inline dashboard.

Reads only completed-trial results and the live training log, so it never touches the
running process. Everything it reports is measured; nothing is projected except the ETA
the trainer itself prints.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime

V2 = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.join(V2, "artifacts", "sweep", "results.jsonl")
CAMP = os.path.join(V2, "artifacts", "campaign3", "results.jsonl")
LOG = ("/private/tmp/claude-502/-Users-johnelliott-PersonalGitHub/"
       "d0de1806-43d2-4722-96e4-a3efe51af569/scratchpad/sweep.log")

# Measured earlier today on this corpus; used as reference lines on the chart.
REF = {
    "cnn_closed": 0.8899,          # plain 38k production CNN, 4000 ep
    "vit_closed": 0.7767,
    "coh_start": 0.153,            # impaired-row coherence before any training
    "coh_ceiling": 0.878,          # mean SNR/(1+SNR) over the corpus: no equalizer beats this
}
AXIS_KEY = {"eq": "lambda_eq", "unet": "w_rec", "oe": "margin"}


def jload(p):
    return [json.loads(x) for x in open(p) if x.strip()] if os.path.exists(p) else []


def main():
    done = jload(SWEEP)
    curve, cur = [], None
    if os.path.exists(LOG):
        for l in open(LOG):
            m = re.search(r"\[(sweep_\S+)\] ep\s+(\d+)/(\d+) ce ([\d.]+)(?: coh ([\d.]+))?"
                          r".*?elapsed ([\d.]+)s eta ([\d.]+)min", l)
            if m:
                tag, ep, tot, ce, coh, el, eta = m.groups()
                if cur != tag:
                    cur, curve = tag, []
                curve.append({"ep": int(ep), "total": int(tot), "ce": float(ce),
                              "coh": float(coh) if coh else None,
                              "elapsed": float(el), "eta_min": float(eta)})
    alive = bool(subprocess.run(["pgrep", "-f", "sweep_hparams"],
                                capture_output=True, text=True).stdout.strip())
    trials = [{"tag": r["tag"], "axis": r.get("axis"), "value": r.get("axis_value"),
               "closed": r["closed_overall"], "chirp": r["auroc_chirp_HELDOUT"],
               "noise": r["auroc_noise"], "coh": r.get("final_coherence"),
               "episodes": r["episodes"]} for r in done]
    print(json.dumps({
        "now": datetime.now().strftime("%H:%M:%S"),
        "running": alive, "current": cur, "curve": curve,
        "trials": trials, "ref": REF, "axis_key": AXIS_KEY,
        "n_campaign_done": len(jload(CAMP)),
    }))


if __name__ == "__main__":
    main()
