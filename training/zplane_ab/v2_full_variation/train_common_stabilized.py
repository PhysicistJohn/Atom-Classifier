"""Stabilized training loop for the z-plane backbone ONLY -- a new, additive
module, not a modification of ../train_common.py (which stays untouched as
the shared harness both v1 and v2's CNN/original-zplane runs depend on).

WHY THIS EXISTS: the first v2 z-plane run (zplane_resampled, 10,000 episodes)
collapsed -- val accuracy peaked at 0.716 by episode 1666, then crashed to
~0.29-0.33 for the remaining 83% of training and never recovered, while loss
rose from ~1.0 to ~1.7 and plateaued. No NaN (so not the same torch.abs()
singular-gradient bug already fixed), which points at a slower drift into a
bad-but-finite regime. The most likely suspect given the architecture: nothing
pulls pole_radius_logit back down, so Adam is free to push kernel pole radii
toward RHO_MAX (0.9922, i.e. very high-Q near-unit-circle resonances) if doing
so reduces training loss short-term; a near-resonant kernel massively amplifies
whatever content sits near its resonant frequency, compounding across 3 layers
of FFT/IFFT, and could plausibly blow up intermediate activation magnitude
into a badly-scaled (but finite, gradient-clipped) regime that wrecks the
downstream embedding without ever NaN-ing. Separately, the learned distance
`scale` parameter (unconstrained, multiplies -sq_dist before cross-entropy) is
a known instability point in prototypical-network-style training generally:
if embeddings spread out unusually far (plausible if the backbone above is
already misbehaving), scale can grow to compensate, sharpening the softmax and
amplifying gradient noise, in a feedback loop.

Three targeted, minimal changes vs ../train_common.py's train(), all
opt-outable/defaulted to reproduce the original behavior for anything that
isn't the z-plane backbone:
  1. Two LR groups: backbone (pole/zero/gain/thresh/mix/lift/proj) at a LOWER
     LR than before, fc1/fc2 head at the original LR -- the head isn't
     implicated, no reason to slow it down too.
  2. `scale` gets its OWN param group with much higher weight_decay than the
     rest of the network, specifically to counter unbounded growth.
  3. Tighter grad-norm clipping than the original 1.0.
Plus: diagnostic logging every eval checkpoint (scale value, pole-radius
max/mean, mix-matrix Frobenius norm mean) so a future run's health can be
checked directly instead of inferred after the fact from a collapsed loss
curve.
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
from train import embed_all, prototypes_from, nearest, sq_dist  # noqa: E402
from zplane_backbone import RHO_MAX  # noqa: E402

K_SHOT = 5
Q_QUERY = 5
HEAD_LR = 1e-3
BACKBONE_LR = 3e-4       # ~3x lower than the original shared 1e-3 -- fix #1
WD = 2e-4                # unchanged base weight decay for net params
SCALE_WD = 1e-2          # 50x the base WD, applied ONLY to `scale` -- fix #2
CLIP_GRAD_STABILIZED = 0.5   # tighter than the original 1.0 -- fix #3


def _backbone_diagnostics(net) -> dict:
    with torch.no_grad():
        radius = RHO_MAX * torch.sigmoid(net.backbone.pole_radius_logit)
        mix_norm = torch.sqrt(net.backbone.mix_re ** 2 + net.backbone.mix_im ** 2).mean(dim=(1, 2))
        return {
            "pole_radius_max": round(float(radius.max()), 4),
            "pole_radius_mean": round(float(radius.mean()), 4),
            "mix_frobenius_mean_per_layer": [round(float(x), 4) for x in mix_norm],
        }


def train_stabilized(net, data, dev, episodes: int, eval_every: int, log_tag: str = ""):
    """Same episodic prototypical-CE loop as ../train_common.py's train(), plus
    the 3 fixes above and per-checkpoint diagnostics. Returns
    (net_with_best_state_loaded, history_dict, wall_clock_seconds)."""
    n_classes = data["n_classes"]
    rng = data["rng"]
    xtr, ftr, ytr, idx_by_class = data["xtr"], data["ftr"], data["ytr"], data["idx_by_class"]
    xen, fen, yen = data["xen"], data["fen"], data["yen"]
    xva, fva, yva = data["xva"], data["fva"], data["yva"]
    n_way = n_classes

    net = net.to(dev)
    scale = torch.nn.Parameter(torch.tensor(10.0, device=dev))

    backbone_params = list(net.backbone.parameters())
    head_params = list(net.fc1.parameters()) + list(net.fc2.parameters())
    opt = torch.optim.Adam([
        {"params": backbone_params, "lr": BACKBONE_LR, "weight_decay": WD},
        {"params": head_params, "lr": HEAD_LR, "weight_decay": WD},
        {"params": [scale], "lr": HEAD_LR, "weight_decay": SCALE_WD},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(episodes, 1))

    def eval_clean():
        protos = prototypes_from(embed_all(net, xen, fen, dev), yen, n_classes)
        pred, _ = nearest(embed_all(net, xva, fva, dev), protos)
        return float((pred == yva).mean())

    print(f"[{log_tag}] STABILIZED training {episodes} episodes ({n_way}-way {K_SHOT}-shot) ... "
          f"backbone_lr={BACKBONE_LR} head_lr={HEAD_LR} scale_wd={SCALE_WD} clip={CLIP_GRAD_STABILIZED}")
    running, best_acc, best_state = 0.0, -1.0, None
    history = {"loss": [], "val_acc": [], "diagnostics": []}
    t0 = time.perf_counter()
    for ep in range(episodes):
        net.train()
        sx, sf, sl, qx, qf, ql = ds.sample_episode(xtr, ftr, idx_by_class, rng, n_way, K_SHOT, Q_QUERY)
        xb = torch.from_numpy(np.concatenate([sx, qx])).to(dev)
        fb = torch.from_numpy(np.concatenate([sf, qf])).to(dev)
        emb = net(xb, fb)
        se, qe = emb[: len(sx)], emb[len(sx):]
        sl_t = torch.from_numpy(sl).to(dev)
        protos = torch.stack([se[sl_t == c].mean(0) for c in range(n_way)])
        logits = -sq_dist(qe, protos) * scale
        loss = torch.nn.functional.cross_entropy(logits, torch.from_numpy(ql).to(dev))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP_GRAD_STABILIZED)
        opt.step()
        sched.step()
        running += loss.item()
        if (ep + 1) % max(1, min(500, episodes)) == 0:
            n = max(1, min(500, episodes))
            print(f"  [{log_tag}] ep {ep+1:5d}  loss {running/n:.4f}  scale {float(scale.detach()):.3f}")
            history["loss"].append(running / n)
            running = 0.0
        if (ep + 1) % eval_every == 0 or (ep + 1) == episodes:
            acc = eval_clean()
            diag = _backbone_diagnostics(net)
            diag["scale"] = round(float(scale.detach()), 4)
            history["val_acc"].append({"episode": ep + 1, "acc": acc})
            history["diagnostics"].append({"episode": ep + 1, **diag})
            tag = ""
            if acc > best_acc:
                best_acc, best_state = acc, copy.deepcopy(net.state_dict())
                tag = "  <- best"
            print(f"    [{log_tag}] [val(clean) {acc:.3f}]{tag}  diag={diag}")
    wall = time.perf_counter() - t0
    if best_state is not None:
        net.load_state_dict(best_state)
    history["best_val_acc"] = best_acc
    history["wall_clock_s"] = wall
    history["episodes"] = episodes
    return net, history, wall
