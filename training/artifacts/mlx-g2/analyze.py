#!/usr/bin/env python
"""Analysis of g6_memorization.json against the content-diversity audit."""
from __future__ import annotations

import json
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
row_cluster = {int(k): v for k, v in div["row_cluster"].items()}
G = json.loads((HERE / "g6_memorization.json").read_text())
seeds = [str(s) for s in G["seeds"]]
proto_rows = set(G["proto_rows"])
zero_rows = set()
for v in json.loads((HERE / "zero_rows.json").read_text()).values():
    zero_rows |= set(v)

# ok[seed][split][dwell][row] = 0/1
ok = {s: {sp: {} for sp in ("eval", "train")} for s in seeds}
for s in seeds:
    for sp in ("eval", "train"):
        for w in DWELLS:
            d = G["results"][s][sp][w]
            ok[s][sp][w] = {r: int(CLASSES[p] == row_meta[r]["cls"])
                            for r, p in zip(d["row"], d["pred"])}

profiles = sorted({r["profile"] for r in man["rows"]})
prof_rows = defaultdict(lambda: {"eval": [], "train": []})
for r in man["rows"]:
    prof_rows[r["profile"]][r["role"]].append(r["row"])


def acc(rows_, s, sp, w):
    if not rows_:
        return float("nan"), 0
    v = [ok[s][sp][w][r] for r in rows_]
    return float(np.mean(v)), len(v)


def macc(rows_, sp, w):
    """mean over seeds"""
    vals = [acc(rows_, s, sp, w)[0] for s in seeds]
    return float(np.mean(vals)), float(np.std(vals)), len(rows_)


print("=" * 118)
print("VALIDATION vs g5_offset_reeval.json (seed 20260740, eval split)")
print("=" * 118)
g5 = json.loads(Path("/Users/johnelliott/PersonalGitHub/Atom-Classifier/"
                     "training/artifacts/mlx-g2/g5_offset_reeval.json"
                     ).read_text())
g5r = g5["results"]["seed20260740"]
for w in DWELLS:
    mine = {}
    for p in profiles:
        a, _ = acc(prof_rows[p]["eval"], "20260740", "eval", w)
        mine[p] = a
    theirs = g5r[w]["profile_recalls"]
    dmax = max(abs(mine[p] - theirs[p]) for p in profiles)
    per_cls = {}
    for c in CLASSES:
        rr = [r for r in ok["20260740"]["eval"][w] if row_meta[r]["cls"] == c]
        per_cls[c] = float(np.mean([ok["20260740"]["eval"][w][r] for r in rr]))
    bal = float(np.mean(list(per_cls.values())))
    print(f"  {w:<6} max |per-profile recall diff| = {dmax:.6f}   "
          f"balanced acc mine={bal:.4f} g5={g5r[w]['balanced_accuracy']:.4f}")

print()
print("=" * 118)
print("1. MEMORIZATION TEST -- per-profile TRAIN vs EVAL accuracy")
print("   (same prototypes, same forward path, independent random window "
       "offsets; mean over 3 seeds)")
print("=" * 118)
hdr = (f"{'profile':<40}{'cls':<10}{'Cdiv':>5}{'held':>5}" +
       "".join(f"{w+' tr':>9}{w+' ev':>9}{'d':>8}" for w in DWELLS))
print(hdr)
print("-" * len(hdr))
table = {}
for p in profiles:
    d = div["per_profile"][p]
    held = d["clusters_eval_only"]
    line = f"{p:<40}{d['cls']:<10}{d['clusters_all']:>5}{held:>5}"
    rec = {"clusters": d["clusters_all"], "held_out": held, "cls": d["cls"]}
    for w in DWELLS:
        tr, trs, _ = macc(prof_rows[p]["train"], "train", w)
        ev, evs, _ = macc(prof_rows[p]["eval"], "eval", w)
        line += f"{tr:>9.4f}{ev:>9.4f}{ev-tr:>+8.4f}"
        rec[w] = {"train": tr, "train_sd": trs, "eval": ev, "eval_sd": evs,
                  "gap": ev - tr}
    table[p] = rec
    print(line)

print()
print("Aggregates (unweighted mean over profiles in the group):")
groups = {
    "single-realization (Cdiv==1, n=20)":
        [p for p in profiles if div["per_profile"][p]["clusters_all"] == 1],
    "gsm 3-cluster (n=5)":
        [p for p in profiles
         if div["per_profile"][p]["cls"] == "gsm"
         and div["per_profile"][p]["clusters_all"] == 3],
    "gsm 12-cluster (n=2)":
        [p for p in profiles
         if div["per_profile"][p]["cls"] == "gsm"
         and div["per_profile"][p]["clusters_all"] == 12],
    "diverse but eval-content-DUPLICATED (n=5)":
        [p for p in profiles
         if div["per_profile"][p]["clusters_all"] > 3
         and div["per_profile"][p]["clusters_eval_only"] == 0
         and div["per_profile"][p]["cls"] != "gsm"],
    "bluetooth: HELD-OUT content (n=2)":
        [p for p in profiles
         if div["per_profile"][p]["clusters_eval_only"] > 0],
}
for gname, ps in groups.items():
    line = f"  {gname:<44} n={len(ps):<3}"
    for w in DWELLS:
        tr = np.mean([table[p][w]["train"] for p in ps])
        ev = np.mean([table[p][w]["eval"] for p in ps])
        line += f"  {w}: tr={tr:.4f} ev={ev:.4f} d={ev-tr:+.4f}"
    print(line)

print()
print("Row-level pooled (all rows, not profile-averaged):")
for w in DWELLS:
    allev = [r for p in profiles for r in prof_rows[p]["eval"]]
    alltr = [r for p in profiles for r in prof_rows[p]["train"]]
    nonproto = [r for r in alltr if r not in proto_rows]
    isproto = [r for r in alltr if r in proto_rows]
    e = macc(allev, "eval", w)
    t = macc(alltr, "train", w)
    np_ = macc(nonproto, "train", w)
    ip = macc(isproto, "train", w)
    print(f"  {w:<6} eval {e[0]:.4f}+-{e[1]:.4f} (n={e[2]})   "
          f"train {t[0]:.4f}+-{t[1]:.4f} (n={t[2]})   "
          f"train-nonProto {np_[0]:.4f} (n={np_[2]})   "
          f"train-protoDonor {ip[0]:.4f} (n={ip[2]})   "
          f"gap(ev-tr) {e[0]-t[0]:+.4f}")

print()
print("=" * 118)
print("2. DIVERSITY vs ACCURACY")
print("=" * 118)
for w in DWELLS:
    x = np.array([np.log10(table[p]["clusters"]) for p in profiles])
    y = np.array([table[p][w]["eval"] for p in profiles])
    xr = np.argsort(np.argsort(x)).astype(float)
    yr = np.argsort(np.argsort(y)).astype(float)
    pear = float(np.corrcoef(x, y)[0, 1])
    spear = float(np.corrcoef(xr, yr)[0, 1])
    print(f"  {w:<6} eval recall vs log10(content clusters): "
          f"pearson r={pear:+.3f}  spearman rho={spear:+.3f}  (n=34 profiles)")
    single = [table[p][w]["eval"] for p in groups[
        "single-realization (Cdiv==1, n=20)"]]
    multi = [table[p][w]["eval"] for p in profiles
             if table[p]["clusters"] > 1]
    print(f"         Cdiv==1 mean recall {np.mean(single):.4f} "
          f"(min {np.min(single):.4f})   Cdiv>1 mean {np.mean(multi):.4f} "
          f"(min {np.min(multi):.4f})")

print()
print("Chance level = 1/7 = 0.1429.  Profiles with content diversity == 1 "
      "and eval recall >> chance:")
for w in ["1ms", "10ms"]:
    fl = [(p, table[p][w]["eval"]) for p in
          groups["single-realization (Cdiv==1, n=20)"]
          if table[p][w]["eval"] > 0.5]
    print(f"  {w}: {len(fl)}/20 profiles above 0.50; "
          f"recalls "
          + ", ".join(f"{p.split('-')[0]}..={v:.2f}" for p, v in fl[:6])
          + f" ... (mean {np.mean([v for _, v in fl]):.4f})")

print()
print("=" * 118)
print("3. HELD-OUT-CONTENT PROXY (bluetooth)")
print("=" * 118)
bt_c = "bluetooth-classic-connected-longdwell"
bt_le = "bluetooth-le-advertising-longdwell"
# BLE: split eval rows by silent vs non-silent, and by content seen in train
le_eval = prof_rows[bt_le]["eval"]
le_train = prof_rows[bt_le]["train"]
tr_clusters = {row_cluster[r] for r in le_train}
le_seen = [r for r in le_eval if row_cluster[r] in tr_clusters]
le_unseen = [r for r in le_eval if row_cluster[r] not in tr_clusters]
le_silent_e = [r for r in le_eval if r in zero_rows]
le_live_e = [r for r in le_eval if r not in zero_rows]
le_silent_t = [r for r in le_train if r in zero_rows]
le_live_t = [r for r in le_train if r not in zero_rows]
print(f"  BLE eval: {len(le_seen)} rows whose content cluster IS in train "
      f"(all of them silent: {len(set(le_seen) & zero_rows)}), "
      f"{len(le_unseen)} content-unseen")
subsets = [
    ("BT-classic TRAIN (content seen in training)", bt_c,
     prof_rows[bt_c]["train"], "train"),
    ("BT-classic EVAL  (content 100% HELD OUT)", bt_c,
     prof_rows[bt_c]["eval"], "eval"),
    ("BLE TRAIN live (non-silent)", bt_le, le_live_t, "train"),
    ("BLE EVAL live (non-silent, content HELD OUT)", bt_le, le_live_e,
     "eval"),
    ("BLE TRAIN silent (all-zero clean)", bt_le, le_silent_t, "train"),
    ("BLE EVAL silent (content = silence, seen in train)", bt_le,
     le_silent_e, "eval"),
]
print(f"  {'subset':<52}{'n':>5}" + "".join(f"{w:>10}" for w in DWELLS))
for label, _p, rws, sp in subsets:
    line = f"  {label:<52}{len(rws):>5}"
    for w in DWELLS:
        a, sd, _ = macc(rws, sp, w)
        line += f"{a:>10.4f}"
    print(line)

print()
print("  Train->eval drop, held-out-content profiles vs the rest:")
for w in DWELLS:
    bt = [p for p in (bt_c, bt_le)]
    rest = [p for p in profiles if p not in bt]
    dbt = np.mean([table[p][w]["gap"] for p in bt])
    dre = np.mean([table[p][w]["gap"] for p in rest])
    print(f"    {w:<6} bluetooth (content held out) {dbt:+.4f}   "
          f"other 32 profiles (content duplicated) {dre:+.4f}   "
          f"difference {dbt-dre:+.4f}")

print()
print("  Where do the held-out-content bluetooth eval rows GO when wrong "
      "(seed-pooled predicted class, 1ms):")
pred_by_row = defaultdict(list)
for s in seeds:
    d = G["results"][s]["eval"]["1ms"]
    for r, p in zip(d["row"], d["pred"]):
        pred_by_row[r].append(CLASSES[p])
for label, rws in (("BT-classic eval (96)", prof_rows[bt_c]["eval"]),
                   ("BLE eval live (67)", le_live_e),
                   ("BLE eval silent (29)", le_silent_e)):
    cnt = defaultdict(int)
    for r in rws:
        for c in pred_by_row[r]:
            cnt[c] += 1
    tot = sum(cnt.values())
    print(f"    {label:<24} " + "  ".join(
        f"{c}={cnt[c]/tot:.3f}" for c in CLASSES if cnt[c]))

print()
print("=" * 118)
print("PER-CLASS recall, train vs eval (mean over 3 seeds)")
print("=" * 118)
cls_rows = defaultdict(lambda: {"eval": [], "train": []})
for r in man["rows"]:
    cls_rows[r["cls"]][r["role"]].append(r["row"])
print(f"  {'class':<11}{'nEv':>6}{'nTr':>6}" +
      "".join(f"{w+' tr':>10}{w+' ev':>10}{'d':>9}" for w in DWELLS))
for c in CLASSES:
    line = (f"  {c:<11}{len(cls_rows[c]['eval']):>6}"
            f"{len(cls_rows[c]['train']):>6}")
    for w in DWELLS:
        tr, _, _ = macc(cls_rows[c]["train"], "train", w)
        ev, _, _ = macc(cls_rows[c]["eval"], "eval", w)
        line += f"{tr:>10.4f}{ev:>10.4f}{ev-tr:>+9.4f}"
    print(line)
for w in DWELLS:
    bt = np.mean([macc(cls_rows[c]["train"], "train", w)[0] for c in CLASSES])
    be = np.mean([macc(cls_rows[c]["eval"], "eval", w)[0] for c in CLASSES])
    print(f"  BALANCED  {w:<6} train {bt:.4f}  eval {be:.4f}  "
          f"gap {be-bt:+.4f}")

json.dump({"table": table, "groups": {k: v for k, v in groups.items()}},
          open(HERE / "analysis_summary.json", "w"), indent=1)
print("\nwrote analysis_summary.json")
