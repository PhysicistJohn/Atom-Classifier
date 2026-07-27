"""Weekend campaign v2 -- on the CORRECTED corpus, scoring closed-set AND open-set.

CONTEXT. The architecture search that preceded this was chasing a corpus bug: the
generator sized Bluetooth captures from the 79 MHz aggregate hop span instead of the
1-2 MHz instantaneous channel, giving 17-23us windows (a Bluetooth slot is 625us), and
41% of clean bluetooth / 36% of clean dsss windows contained no emission at all while
still carrying a class label. Fixing both (instantaneous bandwidth, 16384-sample
captures, occupancy rejection sampling) took the UNCHANGED 38k production CNN from
0.891 to 0.991 closed-set. Architecture was never the binding constraint.

WHAT IS ACTUALLY OPEN. Closed-set is essentially saturated at 0.991. The open-set
valve is not: the 0.991 model classifies band-limited noise as `ofdm` 95% of the time
at a SMALLER prototype distance than correct known signals, AUROC 0.44 overall (0.27
noise) against DESIGN.md's >= 0.72 gate. A first outlier-exposure attempt raised the
seen family (noise 0.27 -> 0.62) but DEGRADED held-out novelty (chirp 0.62 -> 0.26)
for no net gain and -2.2 closed-set: it learned that one family, not novelty.

So this campaign optimizes a JOINT objective and treats chirp as strictly held out.
Configs vary: outlier-family diversity (the leading hypothesis for why OE failed to
generalize), loss form (margin hinge vs entropy/uniformity), margin/lambda, embedding
dim, and backbone. Every run reports closed_overall AND auroc_chirp (generalizing
rejection) AND auroc_noise (seen-family rejection), so a config that games one is
visible immediately.

Safety for unsupervised multi-day operation: disk check + global time budget before
every run, per-run try/except, results appended to results.jsonl immediately, and
REPORT.md regenerated after every run.

Run:  caffeinate -i .venv-training/bin/python training/zplane_ab/v2_full_variation/campaign2.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
import traceback

import numpy as np
import torch

_V2_DIR = os.path.dirname(os.path.abspath(__file__))
_ZPAB_DIR = os.path.dirname(_V2_DIR)
_TRAINING_DIR = os.path.dirname(_ZPAB_DIR)
for p in (_TRAINING_DIR, _ZPAB_DIR, _V2_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from model import Embedding, EMBED_DIM, N_FEATURES  # noqa: E402
from vit_backbone import ViTEmbedding  # noqa: E402
from complex_multiscale_backbone import ComplexMultiScaleEmbedding  # noqa: E402
import train_common_v2 as tcv2  # noqa: E402
from evaluate_ab_v2 import evaluate_v2  # noqa: E402
from openset_eval import evaluate_openset  # noqa: E402
from train_tuned import train_tuned  # noqa: E402
import train_openset as tos  # noqa: E402
from train import embed_all, nearest  # noqa: E402

ARTIFACTS = os.path.join(_V2_DIR, "artifacts", "campaign2")
os.makedirs(ARTIFACTS, exist_ok=True)
RESULTS = os.path.join(ARTIFACTS, "results.jsonl")
REPORT = os.path.join(ARTIFACTS, "REPORT.md")
STATE = os.path.join(ARTIFACTS, "state.json")

MIN_FREE_GB = 4.0
BUDGET_H = 60.0
T0 = time.perf_counter()
DEV = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
_CACHE = {}


def data_for(cond):
    if cond not in _CACHE:
        d = tcv2.prepare_data_v2(condition=cond)
        d["condition"] = cond
        _CACHE[cond] = d
    return _CACHE[cond]


def free_gb():
    return shutil.disk_usage("/").free / 1024 ** 3


def hours():
    return (time.perf_counter() - T0) / 3600.0


def build_net(bb, params, in_len):
    if bb == "cnn":
        return Embedding(params.get("embed_dim", EMBED_DIM), N_FEATURES)
    if bb == "vit":
        p = dict(params)
        p["patch_size"] = p.pop("patch", 16) * (in_len // 1024)
        p["in_len"] = in_len
        return ViTEmbedding(**p)
    if bb == "complexlstm":
        return ComplexMultiScaleEmbedding(**params)
    raise ValueError(bb)


def cm_and_leak(net, data, protos):
    emb = embed_all(net, data["xva"], data["fva"], torch.device("cpu"))
    pred, _ = nearest(emb, protos)
    true, n = data["yva"], data["n_classes"]
    m = np.zeros((n, n), dtype=int)
    for t, p in zip(true, pred):
        m[t, p] += 1
    return m.tolist()


def run_one(tag, backbone, net_params, episodes, condition="native", openset=False, oe=None):
    data = data_for(condition)
    net = build_net(backbone, net_params, data["input_length"])
    n_params = sum(p.numel() for p in net.parameters())
    eval_every = max(200, episodes // 6)
    log_every = max(100, episodes // 10)
    if openset:
        net, hist, wall = tos.train_openset(net, data, DEV, episodes=episodes, eval_every=eval_every,
                                            log_tag=tag, log_every=log_every, **(oe or {}))
    else:
        net, hist, wall = train_tuned(net, data, DEV, episodes=episodes, eval_every=eval_every,
                                      log_tag=tag, lr=7e-4, weight_decay=5e-4, warmup_frac=0.05,
                                      label_smoothing=0.05, log_every=log_every)
    net = net.to("cpu")
    rep, protos = evaluate_v2(net, torch.device("cpu"), data)
    osr = evaluate_openset(net, torch.device("cpu"), data, protos, n_each=300)
    out_dir = os.path.join(ARTIFACTS, tag)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(out_dir, "state_dict.pt"))
    np.save(os.path.join(out_dir, "prototypes.npy"), protos)
    r = {
        "tag": tag, "backbone": backbone, "condition": condition, "episodes": episodes,
        "openset_training": openset, "oe": oe or {}, "net_params": {k: str(v) for k, v in net_params.items()},
        "param_count": n_params, "wall_s": round(wall, 1),
        "closed_overall": round(rep["closed_overall"], 4),
        "closed_clean": round(rep["closed_clean"], 4),
        "closed_impaired": round(rep["closed_impaired"], 4),
        "per_class_clean": rep["per_class_clean"],
        "auroc_overall": osr["auroc_overall"], "auroc_noise": osr["auroc_noise"],
        "auroc_chirp_HELDOUT": osr["auroc_chirp"],
        "flagged_unknown_noise": osr["flagged_unknown_noise"],
        "flagged_unknown_chirp": osr["flagged_unknown_chirp"],
        "confusion_overall": cm_and_leak(net, data, protos),
        "classes": data["classes"],
    }
    # joint score: closed-set must stay high, rejection must GENERALIZE (chirp is the
    # family never trained on). Deliberately weights held-out chirp above seen noise.
    r["joint_score"] = round(r["closed_overall"] + 0.5 * r["auroc_chirp_HELDOUT"] + 0.25 * r["auroc_noise"], 4)
    return r


def log(r):
    with open(RESULTS, "a") as f:
        f.write(json.dumps(r) + "\n")


def load():
    if not os.path.exists(RESULTS):
        return []
    return [json.loads(x) for x in open(RESULTS) if x.strip()]


def write_report():
    rs = load()
    L = ["# Weekend campaign v2 -- corrected corpus, closed-set + open-set\n",
         f"elapsed {hours():.2f}h | {len(rs)} runs | {free_gb():.1f}GB free\n",
         "\nBaseline for comparison (corrected corpus, plain closed-set CNN): "
         "closed_overall 0.991, auroc_overall 0.442, auroc_chirp 0.617, auroc_noise 0.268\n",
         "\n| tag | bb | ep | OE | closed | auroc_all | auroc_chirp(heldout) | auroc_noise | joint | params | s |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rs, key=lambda x: -x["joint_score"]):
        L.append(f"| {r['tag']} | {r['backbone']} | {r['episodes']} | {'Y' if r['openset_training'] else 'n'} "
                 f"| {r['closed_overall']:.4f} | {r['auroc_overall']:.3f} | {r['auroc_chirp_HELDOUT']:.3f} "
                 f"| {r['auroc_noise']:.3f} | {r['joint_score']:.3f} | {r['param_count']} | {r['wall_s']:.0f} |")
    open(REPORT, "w").write("\n".join(L) + "\n")


def safe(tag, **kw):
    if free_gb() < MIN_FREE_GB:
        print(f"[c2] HALT low disk {free_gb():.1f}GB", flush=True)
        raise SystemExit(1)
    if hours() > BUDGET_H:
        print("[c2] budget reached", flush=True)
        return None
    if tag in {r["tag"] for r in load()}:
        print(f"[c2] skip {tag} (done)", flush=True)
        return None
    print(f"[c2] === {tag} === ({hours():.2f}h, {free_gb():.1f}GB)", flush=True)
    try:
        r = run_one(tag, **kw)
        log(r); write_report()
        print(f"[c2] {tag}: closed={r['closed_overall']:.4f} chirp={r['auroc_chirp_HELDOUT']:.3f} "
              f"noise={r['auroc_noise']:.3f} joint={r['joint_score']:.3f}", flush=True)
        return r
    except Exception as e:
        print(f"[c2] FAIL {tag}: {e}", flush=True)
        traceback.print_exc()
        open(os.path.join(ARTIFACTS, "errors.log"), "a").write(f"{tag}: {e}\n{traceback.format_exc()}\n---\n")
        return None


SCREEN = 1500

# The central hypothesis under test: OE failed to generalize because the outlier pool
# was a single family. `outlier_families` widens it. Also varying loss strength/margin,
# capacity, and backbone.
PHASE1 = [
    # -- reference points on the corrected corpus --
    dict(tag="c2_cnn_plain", backbone="cnn", net_params={}, openset=False),
    dict(tag="c2_vit_plain", backbone="vit", net_params=dict(embed_dim=64, depth=4, num_heads=4), openset=False),
    dict(tag="c2_cx_plain", backbone="complexlstm", net_params={}, openset=False),
    # -- outlier exposure: strength sweep --
    dict(tag="c2_cnn_oe_lo", backbone="cnn", net_params={}, openset=True, oe=dict(lambda_out=0.25, margin=0.30)),
    dict(tag="c2_cnn_oe_hi", backbone="cnn", net_params={}, openset=True, oe=dict(lambda_out=4.0, margin=0.30)),
    # -- margin sweep: is 0.30 too easy to satisfy trivially? --
    dict(tag="c2_cnn_oe_m10", backbone="cnn", net_params={}, openset=True, oe=dict(lambda_out=1.0, margin=1.0)),
    dict(tag="c2_cnn_oe_m20", backbone="cnn", net_params={}, openset=True, oe=dict(lambda_out=1.0, margin=2.0)),
    # -- more outliers per episode (weak gradient signal was a candidate cause) --
    dict(tag="c2_cnn_oe_manyout", backbone="cnn", net_params={}, openset=True,
         oe=dict(lambda_out=1.0, margin=1.0, n_out_per_ep=30)),
    # -- capacity: does a wider embedding leave room for an 'unknown' region? --
    dict(tag="c2_cnn_embed64", backbone="cnn", net_params=dict(embed_dim=64), openset=False),
    dict(tag="c2_cnn_embed64_oe", backbone="cnn", net_params=dict(embed_dim=64), openset=True,
         oe=dict(lambda_out=1.0, margin=1.0)),
    # -- other backbones under OE --
    dict(tag="c2_vit_oe", backbone="vit", net_params=dict(embed_dim=64, depth=4, num_heads=4), openset=True,
         oe=dict(lambda_out=1.0, margin=1.0)),
    dict(tag="c2_cx_oe", backbone="complexlstm", net_params={}, openset=True, oe=dict(lambda_out=1.0, margin=1.0)),
]


def main():
    print(f"[c2] start dev={DEV} disk={free_gb():.1f}GB budget={BUDGET_H}h", flush=True)
    json.dump({"phase": "1", "h": hours()}, open(STATE, "w"))
    for c in PHASE1:
        safe(c["tag"], backbone=c["backbone"], net_params=c["net_params"], episodes=SCREEN,
             openset=c.get("openset", False), oe=c.get("oe"))
    json.dump({"phase": "1done", "h": hours()}, open(STATE, "w"))

    rs = [r for r in load() if r["tag"].startswith("c2_")]
    top = sorted(rs, key=lambda r: -r["joint_score"])[:4]
    print(f"[c2] phase2 deep: {[t['tag'] for t in top]}", flush=True)
    by = {c["tag"]: c for c in PHASE1}
    for t in top:
        c = by.get(t["tag"])
        if not c:
            continue
        safe(t["tag"].replace("c2_", "c2deep_"), backbone=c["backbone"], net_params=c["net_params"],
             episodes=6000, openset=c.get("openset", False), oe=c.get("oe"))
    json.dump({"phase": "done", "h": hours()}, open(STATE, "w"))
    write_report()
    print(f"[c2] COMPLETE {hours():.2f}h", flush=True)


if __name__ == "__main__":
    main()
