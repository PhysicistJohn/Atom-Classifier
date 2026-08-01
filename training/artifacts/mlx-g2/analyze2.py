#!/usr/bin/env python
"""Confound checks + power analysis."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
CORPUS = Path("/Users/johnelliott/PersonalGitHub/Atom-Classifier/training/"
              "artifacts/longdwell-production-corpus")
CLASSES = ("am", "bluetooth", "cw", "dsss", "fm", "gsm", "ofdm")
DWELLS = ["1ms", "2.5ms", "10ms"]

man = json.loads((CORPUS / "manifest.json").read_text())
row_meta = {r["row"]: r for r in man["rows"]}
div = json.loads((HERE / "diversity_final.json").read_text())
dp = div["per_profile"]
G = json.loads((HERE / "g6_memorization.json").read_text())
seeds = [str(s) for s in G["seeds"]]
proto_rows = set(G["proto_rows"])

ok = {s: {sp: {} for sp in ("eval", "train")} for s in seeds}
for s in seeds:
    for sp in ("eval", "train"):
        for w in DWELLS:
            d = G["results"][s][sp][w]
            ok[s][sp][w] = {r: int(CLASSES[p] == row_meta[r]["cls"])
                            for r, p in zip(d["row"], d["pred"])}

prof_rows = defaultdict(lambda: {"eval": [], "train": []})
for r in man["rows"]:
    prof_rows[r["profile"]][r["role"]].append(r["row"])
profiles = sorted(prof_rows)


def macc(rows_, sp, w):
    if not rows_:
        return float("nan")
    return float(np.mean([np.mean([ok[s][sp][w][r] for r in rows_])
                          for s in seeds]))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


print("=" * 104)
print("CONFOUND A: prototype-DONOR train rows vs non-donor train rows, "
      "WITHIN each donor profile")
print("=" * 104)
donor_prof = defaultdict(set)
for r in proto_rows:
    donor_prof[row_meta[r]["profile"]].add(r)
print(f"  {'profile':<40}{'nDonor':>7}{'nOther':>7}" +
      "".join(f"{w+' don':>10}{w+' oth':>10}" for w in DWELLS))
for p in sorted(donor_prof):
    don = sorted(donor_prof[p])
    oth = [r for r in prof_rows[p]["train"] if r not in donor_prof[p]]
    line = f"  {p:<40}{len(don):>7}{len(oth):>7}"
    for w in DWELLS:
        line += f"{macc(don,'train',w):>10.4f}{macc(oth,'train',w):>10.4f}"
    print(line)

print()
print("=" * 104)
print("CONFOUND B: training exposure per DISTINCT content realization")
print("  (3000 episodes x (k_shot 5 + q_query 5) = 30000 row-draws per class,")
print("   spread over that class's train rows; worst-dur draws excluded)")
print("=" * 104)
ntr_cls = defaultdict(int)
for r in man["rows"]:
    if r["role"] == "train":
        ntr_cls[r["cls"]] += 1
print(f"  {'profile':<40}{'cls':<10}{'Ctrain':>8}{'draws/prof':>12}"
      f"{'draws/realization':>19}")
expo = {}
for p in profiles:
    c = dp[p]["cls"]
    dpr = 30000 * 160 / ntr_cls[c]
    per = dpr / dp[p]["clusters_train"]
    expo[p] = per
    print(f"  {p:<40}{c:<10}{dp[p]['clusters_train']:>8}{dpr:>12.0f}"
          f"{per:>19.1f}")

print()
print("  rank correlation of eval recall with draws-per-realization:")
for w in DWELLS:
    y = np.array([macc(prof_rows[p]["eval"], "eval", w) for p in profiles])
    x = np.log10(np.array([expo[p] for p in profiles]))
    xr = np.argsort(np.argsort(x)).astype(float)
    yr = np.argsort(np.argsort(y)).astype(float)
    print(f"    {w:<6} spearman rho = "
          f"{float(np.corrcoef(xr, yr)[0,1]):+.3f}   pearson r = "
          f"{float(np.corrcoef(x, y)[0,1]):+.3f}")

print()
print("=" * 104)
print("POWER of the only held-out-content test (bluetooth), "
      "seed-pooled counts, Wilson 95% CI")
print("=" * 104)
bt_c = "bluetooth-classic-connected-longdwell"
bt_le = "bluetooth-le-advertising-longdwell"
zero_rows = set()
for v in json.loads((HERE / "zero_rows.json").read_text()).values():
    zero_rows |= set(v)
held = (prof_rows[bt_c]["eval"]
        + [r for r in prof_rows[bt_le]["eval"] if r not in zero_rows])
seen_ctrl = prof_rows[bt_c]["train"] + [r for r in prof_rows[bt_le]["train"]
                                        if r not in zero_rows]
for label, rws, sp in (("HELD-OUT content (eval)", held, "eval"),
                       ("SEEN content (train)", seen_ctrl, "train")):
    for w in DWELLS:
        k = sum(ok[s][sp][w][r] for s in seeds for r in rws)
        n = len(rws) * len(seeds)
        lo, hi = wilson(k, n)
        print(f"  {label:<26}{w:<7} {k}/{n} = {k/n:.4f}  "
              f"95% CI [{lo:.4f}, {hi:.4f}]")
print()
print(f"  held-out-content eval rows: {len(held)} of 3264 "
      f"= {len(held)/3264:.2%} of the eval split, "
      f"covering 1 of 7 classes (bluetooth).")
print("  The other 6 classes (2 of 2 am/cw/fm/dsss/gsm/ofdm groupings, "
      "3101 of 3264 eval rows) are scored entirely on clean content that "
      "is byte-duplicated in the train split.")

print()
print("=" * 104)
print("GSM 1 ms failure is a DUTY-CYCLE effect, not a diversity effect")
print("=" * 104)
clean = np.load(CORPUS / "clean.npy", mmap_mode="r")
for p in [q for q in profiles if dp[q]["cls"] == "gsm"]:
    rid = prof_rows[p]["eval"][0]
    x = np.asarray(clean[rid])
    a = np.abs(x)
    thr = 0.1 * a.max()
    occ = float((a > thr).mean())
    # fraction of 1 ms (20000-sample) windows that are >50% occupied
    w = 20000
    m = (a > thr).astype(np.float32)
    cs = np.concatenate([[0.0], np.cumsum(m)])
    occ1 = (cs[w:] - cs[:-w]) / w
    print(f"  {p:<40} Cdiv={dp[p]['clusters_all']:>3}  "
          f"duty(row)={occ:.3f}  frac of 1ms windows >50% occupied="
          f"{float((occ1 > 0.5).mean()):.3f}  "
          f"eval recall 1ms={macc(prof_rows[p]['eval'],'eval','1ms'):.3f}")
