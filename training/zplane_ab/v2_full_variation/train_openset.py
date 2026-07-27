"""Episodic prototypical training + OUTLIER EXPOSURE, so the embedding learns a
"none of the above" region instead of mapping every input onto one of the 7
prototypes.

WHY. Closed-set training alone produced a model that is 0.991 accurate on known
signals and simultaneously classifies band-limited noise as `ofdm` 95% of the time
at a SMALLER prototype distance (0.0025) than correctly-classified real signals
(0.0063). Open-set AUROC 0.44 overall / 0.27 on noise -- below chance, against
DESIGN.md's >= 0.72 acceptance gate. Prototypical cross-entropy only shapes
distances BETWEEN the 7 classes; nothing in it says "and everything else should be
far away", so low-information inputs collapse onto whichever prototype they most
resemble. Noise landing on OFDM is not even a mistake in the usual sense: OFDM is a
sum of many subcarriers and is genuinely noise-like, so the two are close in every
statistic the network uses. It still has to be REJECTED rather than confidently
labelled, which is what the open-set valve is for.

METHOD. Standard outlier exposure adapted to a metric/prototypical head: each
episode draws a small batch of auxiliary outliers and adds a hinge that pushes their
nearest-prototype distance beyond a margin:

    L_out = mean( max(0, margin - min_c ||f(x_out) - p_c||^2 )^2 )
    L     = L_proto + lambda_out * L_out

This adds no prototype and no class, so DESIGN.md's "novelty is never enrolled"
constraint is preserved structurally -- the taxonomy stays at 7.

HONEST EVALUATION. Training on the same novelty family used for scoring would be
self-fulfilling. Outliers here are drawn ONLY from the noise family (plus optional
zero/near-silent windows, which is what an idle capture actually looks like);
chirp/LFM is never trained on and is scored as genuinely held-out novelty. Report
both: auroc_noise is the seen-family number, auroc_chirp is the one that says
whether rejection generalizes.
"""
from __future__ import annotations

import copy
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

import dataset as ds  # noqa: E402
import preprocess as pp  # noqa: E402
import native_preprocess as npp  # noqa: E402
from train import embed_all, prototypes_from, nearest, sq_dist  # noqa: E402
from openset_eval import _bandlimited_noise  # noqa: E402 -- the TRAIN-side family only

K_SHOT = 5
Q_QUERY = 5


def build_outlier_pool(condition: str, in_len: int, fmean, fstd, n: int = 600, seed: int = 991):
    """Auxiliary outliers: band-limited noise plus near-silent/idle windows.
    Preprocessed through the same front end as the known pools."""
    module = npp if condition == "native" else pp
    rng = np.random.default_rng(seed)
    xs, fs = [], []
    for i in range(n):
        if i % 4 == 0:
            # idle capture: tiny noise floor only. This is literally what a window
            # with no emission looks like, and it is the case the occupancy gate
            # removed from the labelled corpus -- it belongs here instead.
            raw = (1e-3) * (rng.standard_normal(in_len) + 1j * rng.standard_normal(in_len))
        else:
            raw = _bandlimited_noise(in_len, rng)
        norm, _ = module.preprocess(raw, l_out=in_len)
        xs.append(pp.to_channels(norm))
        fs.append(pp.iq_features(norm))
    x = np.stack(xs).astype(np.float32)
    f = ((np.stack(fs).astype(np.float32) - fmean) / fstd).astype(np.float32)
    return x, f


def train_openset(net, data, dev, episodes: int, eval_every: int, log_tag: str = "",
                   lr: float = 7e-4, weight_decay: float = 5e-4, warmup_frac: float = 0.05,
                   label_smoothing: float = 0.05, lambda_out: float = 1.0, margin: float = 0.30,
                   n_out_per_ep: int = 10, log_every: int | None = None):
    n_classes = data["n_classes"]
    rng = data["rng"]
    xtr, ftr, ytr, idx_by_class = data["xtr"], data["ftr"], data["ytr"], data["idx_by_class"]
    xen, fen, yen = data["xen"], data["fen"], data["yen"]
    xva, fva, yva = data["xva"], data["fva"], data["yva"]
    n_way = n_classes

    in_len = xtr.shape[-1]
    xout, fout = build_outlier_pool(data.get("condition", "native"), in_len, data["fmean"], data["fstd"])
    xout_t = torch.from_numpy(xout)
    fout_t = torch.from_numpy(fout)
    print(f"[{log_tag}] outlier pool: {xout.shape[0]} (band-limited noise + idle windows); "
          f"chirp is NOT trained on and stays held-out novelty", flush=True)

    net = net.to(dev)
    scale = torch.nn.Parameter(torch.tensor(10.0, device=dev))
    opt = torch.optim.Adam(list(net.parameters()) + [scale], lr=lr, weight_decay=weight_decay)
    warmup = max(1, int(episodes * warmup_frac))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(episodes - warmup, 1))
    out_rng = np.random.default_rng(4242)

    def eval_clean():
        protos = prototypes_from(embed_all(net, xen, fen, dev), yen, n_classes)
        pred, _ = nearest(embed_all(net, xva, fva, dev), protos)
        return float((pred == yva).mean())

    print(f"[{log_tag}] OPENSET training {episodes} ep ... lambda_out={lambda_out} margin={margin} "
          f"n_out/ep={n_out_per_ep}", flush=True)
    running_ce = running_out = 0.0
    best_acc, best_state = -1.0, None
    history = {"loss_ce": [], "loss_out": [], "val_acc": []}
    t0 = time.perf_counter()
    for ep in range(episodes):
        net.train()
        if ep < warmup:
            for g in opt.param_groups:
                g["lr"] = lr * (ep + 1) / warmup
        sx, sf, sl, qx, qf, ql = ds.sample_episode(xtr, ftr, idx_by_class, rng, n_way, K_SHOT, Q_QUERY)
        xb = torch.from_numpy(np.concatenate([sx, qx])).to(dev)
        fb = torch.from_numpy(np.concatenate([sf, qf])).to(dev)
        emb = net(xb, fb)
        se, qe = emb[: len(sx)], emb[len(sx):]
        sl_t = torch.from_numpy(sl).to(dev)
        protos = torch.stack([se[sl_t == c].mean(0) for c in range(n_way)])
        logits = -sq_dist(qe, protos) * scale
        loss_ce = torch.nn.functional.cross_entropy(logits, torch.from_numpy(ql).to(dev),
                                                     label_smoothing=label_smoothing)

        pick = out_rng.choice(len(xout), size=n_out_per_ep, replace=False)
        eo = net(xout_t[pick].to(dev), fout_t[pick].to(dev))
        d_out = sq_dist(eo, protos)                      # (n_out, n_way)
        min_d = d_out.min(dim=1).values
        loss_out = torch.clamp(margin - min_d, min=0.0).pow(2).mean()

        loss = loss_ce + lambda_out * loss_out
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep >= warmup:
            cosine.step()
        running_ce += loss_ce.item()
        running_out += loss_out.item()

        log_n = log_every or max(1, min(500, episodes))
        if (ep + 1) % log_n == 0:
            el = time.perf_counter() - t0
            print(f"  [{log_tag}] ep {ep+1:5d}/{episodes} ce {running_ce/log_n:.4f} out {running_out/log_n:.4f} "
                  f"elapsed {el:.0f}s eta {(el/(ep+1))*(episodes-ep-1)/60:.1f}min", flush=True)
            history["loss_ce"].append(running_ce / log_n)
            history["loss_out"].append(running_out / log_n)
            running_ce = running_out = 0.0
        if (ep + 1) % eval_every == 0 or (ep + 1) == episodes:
            acc = eval_clean()
            history["val_acc"].append({"episode": ep + 1, "acc": acc})
            tag = ""
            if acc > best_acc:
                best_acc, best_state = acc, copy.deepcopy(net.state_dict())
                tag = "  <- best"
            print(f"    [{log_tag}] [val(clean) {acc:.3f}]{tag}", flush=True)
    wall = time.perf_counter() - t0
    if best_state is not None:
        net.load_state_dict(best_state)
    history["best_val_acc"] = best_acc
    history["wall_clock_s"] = wall
    history["episodes"] = episodes
    return net, history, wall
