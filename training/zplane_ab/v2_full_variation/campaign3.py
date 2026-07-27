"""FINAL weekend campaign, on the field-realistic corpus.

WHAT CHANGED UNDER THIS CAMPAIGN vs everything before it. The corpus was rebuilt three
times today after three separate defects were found and fixed:
  1. Bluetooth captures were sized from the 79 MHz aggregate hop span instead of the
     1-2 MHz instantaneous channel -> 17-23us windows against a 625us slot. Burst
     structure was absent from the data, not merely hard to learn.
  2. 41% of clean bluetooth and 36% of clean dsss windows contained NO emission but
     still carried a class label -- pure noise wearing a modulation label.
  3. Sample rate was derived from each signal's own bandwidth, so fs carried 0.859 bits
     about the class (31% of class entropy). Accuracy was 0.99+ wherever fs narrowed the
     candidate set and 0.967 in the one band where all 7 classes coexist.
Fixes: instantaneous bandwidth, 16384-sample captures, occupancy rejection sampling,
full-range fs sweep, and a physically-derived nuisance distribution grounded in the
actual radio envelope (200 MSPS / 10 GHz / 0 to -110 dBm). Residual I(class; fs) is
0.207 bits and is irreducible -- dsss cannot be sampled below its own 25.3 MHz width.

The honest baseline on that corpus is 0.890 (the unchanged 38k production CNN), against
0.991 on the shortcut-laden version. 0.890 is the number that means anything.

WHAT THIS CAMPAIGN SEARCHES.
  A. The learned equalizer front end (user's proposal). The corpus now stores clean/
     impaired PAIRS, so the front end is supervised by coherence against the emission
     each capture came from, not just by classification error. lambda_eq=0 is included
     as the capacity control -- without it, any gain could be explained by added depth.
  B. Open-set rejection, which is measured but NOT yet solved: the closed-set-optimal
     model classifies band-limited noise as `ofdm` 95% of the time at higher confidence
     than it classifies real signals. Chirp is held out from all outlier training so the
     score measures generalizing rejection rather than memorizing one novelty family.
  C. Backbone comparison, now that it is being run on data where the answer is not
     predetermined by a corpus artifact.

Every run reports closed-set AND held-out-novelty AUROC, so a config that wins one by
sacrificing the other is visible rather than hidden in an aggregate.
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
from equalizer_frontend import EqualizedClassifier  # noqa: E402
from multitask_autoencoder import MultiTaskAutoencoder  # noqa: E402
from unet_multitask import ComplexUNetMultiTask  # noqa: E402
import train_common_v2 as tcv2  # noqa: E402
from evaluate_ab_v2 import evaluate_v2  # noqa: E402
from openset_eval import evaluate_openset  # noqa: E402
from train_tuned import train_tuned  # noqa: E402
from train_equalizer import train_equalizer  # noqa: E402
from train_multitask import train_multitask  # noqa: E402
import train_openset as tos  # noqa: E402
from train import embed_all, nearest  # noqa: E402

ART = os.path.join(_V2_DIR, "artifacts", "campaign3")
os.makedirs(ART, exist_ok=True)
RESULTS, REPORT, STATE = (os.path.join(ART, x) for x in ("results.jsonl", "REPORT.md", "state.json"))
MIN_FREE_GB, BUDGET_H = 4.0, 58.0
T0 = time.perf_counter()
DEV = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
_C = {}


def data_for(c):
    """Memory-mapped pools when the cache exists, otherwise build from scratch.

    Each isolated trial subprocess previously re-ran prepare_data_v2: ~80 s and a ~9 GB
    peak, once per trial. That churn is the likeliest driver of the swap exhaustion behind
    the weekend stall, so the containment fix was making its own suspected cause worse.
    np.load(mmap_mode="r") hands every child a read-only view of one shared page-cached
    copy -- seconds to open, and one copy of the data no matter how many trials run."""
    if c not in _C:
        try:
            import pool_cache
            d = pool_cache.load(c)
            print(f"[data] {c}: memory-mapped pools (xtr {d['xtr'].shape})", flush=True)
        except Exception as e:
            print(f"[data] pool cache unavailable ({e}); building in-process", flush=True)
            d = tcv2.prepare_data_v2(condition=c)
        d["condition"] = c
        _C[c] = d
    return _C[c]


def free_gb(): return shutil.disk_usage("/").free / 1024 ** 3
def hours(): return (time.perf_counter() - T0) / 3600.0


def make_backbone(bb, params, in_len):
    if bb == "cnn":
        return Embedding(params.get("embed_dim", EMBED_DIM), N_FEATURES)
    if bb == "vit":
        p = dict(params); p["patch_size"] = p.pop("patch", 16) * (in_len // 1024); p["in_len"] = in_len
        return ViTEmbedding(**p)
    if bb == "complexlstm":
        return ComplexMultiScaleEmbedding(**params)
    raise ValueError(bb)


def run_one(tag, backbone, net_params, episodes, mode="plain", eq=None, oe=None, mt=None, condition="native"):
    data = data_for(condition)
    if mode == "multitask":
        net = MultiTaskAutoencoder(**(mt or {}).get("arch", {}))
    elif mode == "unet":
        net = ComplexUNetMultiTask(**(mt or {}).get("arch", {}))
    else:
        core = make_backbone(backbone, net_params, data["input_length"])
        net = EqualizedClassifier(core, **(eq or {}).get("arch", {})) if mode == "equalizer" else core
    n_params = sum(p.numel() for p in net.parameters())
    ee, le = max(200, episodes // 6), max(100, episodes // 10)
    if mode in ("multitask", "unet"):
        m = dict(mt or {}); m.pop("arch", None)
        net, hist, wall = train_multitask(net, data, DEV, episodes, ee, log_tag=tag, log_every=le, **m)
    elif mode == "equalizer":
        net, hist, wall = train_equalizer(net, data, DEV, episodes, ee, log_tag=tag,
                                          lambda_eq=(eq or {}).get("lambda_eq", 1.0), log_every=le)
    elif mode == "openset":
        net, hist, wall = tos.train_openset(net, data, DEV, episodes=episodes, eval_every=ee,
                                            log_tag=tag, log_every=le, **(oe or {}))
    else:
        net, hist, wall = train_tuned(net, data, DEV, episodes=episodes, eval_every=ee, log_tag=tag,
                                      lr=7e-4, weight_decay=5e-4, warmup_frac=0.05,
                                      label_smoothing=0.05, log_every=le)
    # Evaluation device (2026-07-24). This was pinned to CPU so numbers stayed
    # device-controlled, which was cheap for the 38k CNN but not for the equalizer /
    # U-Net at 16384 samples: scoring 4197 val + 4197 enroll + 600 novelty captures
    # through complex FIR convs took ~23 min per trial, MORE than the 25 min of
    # training, and pushed the 16-trial sweep from ~4.4 h to ~11 h. Every trial in a
    # sweep is scored on the same device, so within-axis RANKING -- the only thing the
    # sweep is for -- is unaffected. Cross-run comparisons against the CPU-scored
    # c3_cnn (0.890) / c3_vit (0.777) baselines are the ones to treat with care; the
    # winners get a full-length confirmation run where the device can be pinned again.
    EVAL_DEV = DEV
    net = net.to(EVAL_DEV)
    rep, protos = evaluate_v2(net, EVAL_DEV, data)
    osr = evaluate_openset(net, EVAL_DEV, data, protos, n_each=300)
    emb = embed_all(net, data["xva"], data["fva"], EVAL_DEV)
    pred, _ = nearest(emb, protos)
    n = data["n_classes"]; cm = np.zeros((n, n), dtype=int)
    for t, p_ in zip(data["yva"], pred):
        cm[t, p_] += 1
    net = net.to("cpu")   # checkpoints stay device-agnostic on disk
    od = os.path.join(ART, tag); os.makedirs(od, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(od, "state_dict.pt"))
    np.save(os.path.join(od, "prototypes.npy"), protos)
    r = dict(tag=tag, backbone=backbone, mode=mode, episodes=episodes, param_count=n_params,
             wall_s=round(wall, 1), eq=eq or {}, oe=oe or {},
             closed_overall=round(rep["closed_overall"], 4),
             closed_clean=round(rep["closed_clean"], 4),
             closed_impaired=round(rep["closed_impaired"], 4),
             per_class_clean=rep["per_class_clean"],
             auroc_overall=osr["auroc_overall"], auroc_noise=osr["auroc_noise"],
             auroc_chirp_HELDOUT=osr["auroc_chirp"],
             final_coherence=(hist.get("coherence") or [None])[-1],
             confusion=cm.tolist(), classes=data["classes"])
    r["joint"] = round(r["closed_overall"] + 0.5 * r["auroc_chirp_HELDOUT"] + 0.25 * r["auroc_noise"], 4)
    return r


def load(): return [json.loads(x) for x in open(RESULTS)] if os.path.exists(RESULTS) else []


def write_report():
    rs = load()
    L = ["# Weekend campaign 3 -- field-realistic corpus\n",
         f"elapsed {hours():.2f}h | {len(rs)} runs | {free_gb():.1f}GB free\n",
         "\nReference: unchanged 38k production CNN on this corpus = **closed 0.890**, "
         "auroc_overall 0.44 (open-set is NOT solved; see campaign docstring)\n",
         "\n| tag | bb | mode | ep | closed | clean | impaired | auroc_all | auroc_chirp(heldout) | auroc_noise | coh | joint | params | s |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rs, key=lambda x: -x["joint"]):
        coh = f"{r['final_coherence']:.3f}" if r.get("final_coherence") is not None else "-"
        L.append(f"| {r['tag']} | {r['backbone']} | {r['mode']} | {r['episodes']} | {r['closed_overall']:.4f} "
                 f"| {r['closed_clean']:.4f} | {r['closed_impaired']:.4f} | {r['auroc_overall']:.3f} "
                 f"| {r['auroc_chirp_HELDOUT']:.3f} | {r['auroc_noise']:.3f} | {coh} | {r['joint']:.3f} "
                 f"| {r['param_count']} | {r['wall_s']:.0f} |")
    open(REPORT, "w").write("\n".join(L) + "\n")


def safe(tag, **kw):
    if free_gb() < MIN_FREE_GB:
        print(f"[c3] HALT low disk {free_gb():.1f}GB", flush=True); raise SystemExit(1)
    if hours() > BUDGET_H:
        print("[c3] budget reached", flush=True); return None
    if tag in {r["tag"] for r in load()}:
        print(f"[c3] skip {tag}", flush=True); return None
    print(f"[c3] === {tag} === ({hours():.2f}h, {free_gb():.1f}GB)", flush=True)
    try:
        r = run_one(tag, **kw);
        with open(RESULTS, "a") as f: f.write(json.dumps(r) + "\n")
        write_report()
        print(f"[c3] {tag}: closed={r['closed_overall']:.4f} chirp={r['auroc_chirp_HELDOUT']:.3f} "
              f"joint={r['joint']:.3f}", flush=True)
        return r
    except Exception as e:
        print(f"[c3] FAIL {tag}: {e}", flush=True); traceback.print_exc()
        open(os.path.join(ART, "errors.log"), "a").write(f"{tag}: {e}\n{traceback.format_exc()}\n---\n")
        return None


# Equalizer runs cost ~2.1 s/episode at 16384 samples, so they get a smaller screen
# budget than the cheap CNN runs. Sized so the whole phase-1 sweep fits comfortably.
CHEAP, EQ, UNET = 4000, 1200, 900   # U-Net is ~800k params over 16384 samples: fewer episodes per screen

PHASE1 = [
    # --- complex U-Net (user's design): skips carry payload around the bottleneck so the
    # bottleneck is free to carry modulation structure + channel parameters for the heads ---
    dict(tag="c3_unet_full", backbone="cnn", net_params={}, episodes=UNET, mode="unet",
         mt=dict(w_rec=1.0, w_prof=0.3, w_par=0.3)),
    dict(tag="c3_unet_CONTROL_classonly", backbone="cnn", net_params={}, episodes=UNET, mode="unet",
         mt=dict(w_rec=0.0, w_prof=0.0, w_par=0.0)),
    dict(tag="c3_unet_recon_only", backbone="cnn", net_params={}, episodes=UNET, mode="unet",
         mt=dict(w_rec=1.0, w_prof=0.0, w_par=0.0)),
    dict(tag="c3_unet_recon_heavy", backbone="cnn", net_params={}, episodes=UNET, mode="unet",
         mt=dict(w_rec=3.0, w_prof=0.3, w_par=0.3)),
    # controls on the honest corpus
    dict(tag="c3_cnn", backbone="cnn", net_params={}, episodes=CHEAP, mode="plain"),
    dict(tag="c3_vit", backbone="vit", net_params=dict(embed_dim=64, depth=4, num_heads=4),
         episodes=CHEAP, mode="plain"),
    dict(tag="c3_cx", backbone="complexlstm", net_params={}, episodes=CHEAP, mode="plain"),
    # THE capacity control: equalizer present, coherence loss OFF. Any gain from the
    # lambda_eq>0 runs must beat THIS, not the plain CNN, or it is just extra depth.
    dict(tag="c3_eq_lambda0_CONTROL", backbone="cnn", net_params={}, episodes=EQ,
         mode="equalizer", eq=dict(lambda_eq=0.0)),
    # equalizer strength sweep
    dict(tag="c3_eq_lambda05", backbone="cnn", net_params={}, episodes=EQ, mode="equalizer",
         eq=dict(lambda_eq=0.5)),
    dict(tag="c3_eq_lambda1", backbone="cnn", net_params={}, episodes=EQ, mode="equalizer",
         eq=dict(lambda_eq=1.0)),
    dict(tag="c3_eq_lambda3", backbone="cnn", net_params={}, episodes=EQ, mode="equalizer",
         eq=dict(lambda_eq=3.0)),
    # bigger equalizer
    dict(tag="c3_eq_wide", backbone="cnn", net_params={}, episodes=EQ, mode="equalizer",
         eq=dict(lambda_eq=1.0, arch=dict(eq_width=32, eq_taps=65, eq_blocks=4))),
    # open-set attempts (chirp always held out)
    dict(tag="c3_oe_m1", backbone="cnn", net_params={}, episodes=CHEAP, mode="openset",
         oe=dict(lambda_out=1.0, margin=1.0)),
    dict(tag="c3_oe_m3", backbone="cnn", net_params={}, episodes=CHEAP, mode="openset",
         oe=dict(lambda_out=1.0, margin=3.0)),
    dict(tag="c3_oe_strong", backbone="cnn", net_params={}, episodes=CHEAP, mode="openset",
         oe=dict(lambda_out=5.0, margin=2.0, n_out_per_ep=30)),
    # --- multi-task encoder/decoder (user's design): latent must name the class,
    # predict the physical parameters, AND decode back to the clean emission ---
    dict(tag="c3_mt_full", backbone="cnn", net_params={}, episodes=EQ, mode="multitask",
         mt=dict(w_rec=1.0, w_prof=0.3, w_par=0.3)),
    # ablations that attribute any gain to a specific term
    dict(tag="c3_mt_CONTROL_classonly", backbone="cnn", net_params={}, episodes=EQ, mode="multitask",
         mt=dict(w_rec=0.0, w_prof=0.0, w_par=0.0)),
    dict(tag="c3_mt_no_recon", backbone="cnn", net_params={}, episodes=EQ, mode="multitask",
         mt=dict(w_rec=0.0, w_prof=0.3, w_par=0.3)),
    dict(tag="c3_mt_no_params", backbone="cnn", net_params={}, episodes=EQ, mode="multitask",
         mt=dict(w_rec=1.0, w_prof=0.3, w_par=0.0)),
    dict(tag="c3_mt_recon_heavy", backbone="cnn", net_params={}, episodes=EQ, mode="multitask",
         mt=dict(w_rec=3.0, w_prof=0.3, w_par=0.3)),
    # capacity
    dict(tag="c3_cnn_embed64", backbone="cnn", net_params=dict(embed_dim=64), episodes=CHEAP, mode="plain"),
]


def main():
    print(f"[c3] start dev={DEV} disk={free_gb():.1f}GB budget={BUDGET_H}h", flush=True)
    json.dump({"phase": 1, "h": hours()}, open(STATE, "w"))
    for c in PHASE1:
        safe(c["tag"], backbone=c["backbone"], net_params=c["net_params"], episodes=c["episodes"],
             mode=c["mode"], eq=c.get("eq"), oe=c.get("oe"), mt=c.get("mt"))
    json.dump({"phase": "1done", "h": hours()}, open(STATE, "w"))

    rs = [r for r in load() if r["tag"].startswith("c3_")]
    top = sorted(rs, key=lambda r: -r["joint"])[:3]
    print(f"[c3] deep: {[t['tag'] for t in top]}", flush=True)
    by = {c["tag"]: c for c in PHASE1}
    for t in top:
        c = by.get(t["tag"])
        if not c: continue
        deep = 12000 if c["mode"] not in ("equalizer", "multitask", "unet") else 3000
        safe(t["tag"].replace("c3_", "c3deep_"), backbone=c["backbone"], net_params=c["net_params"],
             episodes=deep, mode=c["mode"], eq=c.get("eq"), oe=c.get("oe"), mt=c.get("mt"))
    json.dump({"phase": "done", "h": hours()}, open(STATE, "w"))
    write_report()
    print(f"[c3] COMPLETE {hours():.2f}h", flush=True)


if __name__ == "__main__":
    main()
