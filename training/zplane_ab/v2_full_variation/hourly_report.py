"""Hourly campaign status with confusion matrices, for unattended monitoring.

Reads only what the campaign has already written to disk (results.jsonl + the live log),
so it never interferes with training and is safe to run at any moment.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime

V2 = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(V2, "artifacts", "campaign3")
RESULTS = os.path.join(ART, "results.jsonl")
LOG = ("/private/tmp/claude-502/-Users-johnelliott-PersonalGitHub/"
       "d0de1806-43d2-4722-96e4-a3efe51af569/scratchpad/campaign3.log")

# Reference points measured earlier today on this same corpus.
BASELINE = {"closed": 0.890, "auroc_overall": 0.44, "chirp": 0.617, "noise": 0.268}


def load():
    if not os.path.exists(RESULTS):
        return []
    return [json.loads(x) for x in open(RESULTS) if x.strip()]


def cm_block(r):
    classes, cm = r["classes"], r["confusion"]
    w = max(len(c[:4]) for c in classes)
    head = "         " + " ".join(f"{c[:4]:>5}" for c in classes)
    lines = [head]
    for i, c in enumerate(classes):
        row = " ".join(f"{cm[i][j]:5d}" for j in range(len(classes)))
        tot = sum(cm[i]) or 1
        lines.append(f"{c[:8]:>8} {row}   n={tot:<5d} recall={cm[i][i]/tot:.3f}")
    return "\n".join(lines)


def main():
    rs = load()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"=== CAMPAIGN STATUS {now} ===")
    alive = subprocess.run(["pgrep", "-f", "campaign3.py"], capture_output=True, text=True).stdout.strip()
    print(f"campaign process: {'RUNNING pid ' + alive.splitlines()[0] if alive else 'NOT RUNNING'}")
    print(f"completed runs: {len(rs)}/21")
    print(f"reference (plain CNN, this corpus): closed={BASELINE['closed']:.3f} "
          f"chirp={BASELINE['chirp']:.3f} noise={BASELINE['noise']:.3f}")
    print()

    if rs:
        print("| tag | mode | ep | closed | clean | impaired | chirp(heldout) | noise | coh | params |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(rs, key=lambda x: -x["joint"]):
            coh = f"{r['final_coherence']:.3f}" if r.get("final_coherence") is not None else "-"
            print(f"| {r['tag']} | {r['mode']} | {r['episodes']} | {r['closed_overall']:.4f} "
                  f"| {r['closed_clean']:.4f} | {r['closed_impaired']:.4f} "
                  f"| {r['auroc_chirp_HELDOUT']:.3f} | {r['auroc_noise']:.3f} | {coh} | {r['param_count']} |")
        print()
        # confusion matrix for the newest run and the best-closed-set run
        newest = rs[-1]
        best = max(rs, key=lambda x: x["closed_overall"])
        shown = []
        for label, r in [("NEWEST", newest), ("BEST closed-set", best)]:
            if r["tag"] in shown:
                continue
            shown.append(r["tag"])
            print(f"--- {label}: {r['tag']}  (closed {r['closed_overall']:.4f}, "
                  f"chirp {r['auroc_chirp_HELDOUT']:.3f}) ---")
            print(cm_block(r))
            print()

    # live progress of whatever is training right now
    if os.path.exists(LOG):
        cur = [l for l in open(LOG) if l.startswith("[c3] === ")]
        prog = [l.rstrip() for l in open(LOG) if " ep " in l and "eta" in l]
        if cur:
            print(f"in progress: {cur[-1].strip()}")
        for l in prog[-3:]:
            print("  " + l.strip())


if __name__ == "__main__":
    main()
