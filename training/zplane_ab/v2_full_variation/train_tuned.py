"""Plain prototypical-CE training (same math as ../train_common.py's train(),
no margin/repulsion term) but with the standard hyperparameters exposed as
knobs instead of hardcoded: learning rate, weight decay, a linear LR warmup
(standard for transformers -- ../train_common.py has none, straight cosine
decay from step 0), and label smoothing on the cross-entropy.
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

K_SHOT = 5
Q_QUERY = 5


def train_tuned(net, data, dev, episodes: int, eval_every: int, log_tag: str = "",
                 lr: float = 1e-3, weight_decay: float = 2e-4, warmup_frac: float = 0.05,
                 label_smoothing: float = 0.0, clip_grad: float | None = None, log_every: int | None = None):
    n_classes = data["n_classes"]
    rng = data["rng"]
    xtr, ftr, ytr, idx_by_class = data["xtr"], data["ftr"], data["ytr"], data["idx_by_class"]
    xen, fen, yen = data["xen"], data["fen"], data["yen"]
    xva, fva, yva = data["xva"], data["fva"], data["yva"]
    n_way = n_classes

    net = net.to(dev)
    scale = torch.nn.Parameter(torch.tensor(10.0, device=dev))
    opt = torch.optim.Adam(list(net.parameters()) + [scale], lr=lr, weight_decay=weight_decay)
    warmup_episodes = max(1, int(episodes * warmup_frac))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(episodes - warmup_episodes, 1))

    def eval_clean():
        protos = prototypes_from(embed_all(net, xen, fen, dev), yen, n_classes)
        pred, _ = nearest(embed_all(net, xva, fva, dev), protos)
        return float((pred == yva).mean())

    print(f"[{log_tag}] TUNED training {episodes} episodes ({n_way}-way {K_SHOT}-shot) ... "
          f"lr={lr} wd={weight_decay} warmup_frac={warmup_frac} ({warmup_episodes} ep) "
          f"label_smoothing={label_smoothing} clip_grad={clip_grad}")
    running, best_acc, best_state = 0.0, -1.0, None
    history = {"loss": [], "val_acc": [], "lr": []}
    t0 = time.perf_counter()
    for ep in range(episodes):
        net.train()
        if ep < warmup_episodes:
            lr_now = lr * (ep + 1) / warmup_episodes
            for g in opt.param_groups:
                g["lr"] = lr_now
        sx, sf, sl, qx, qf, ql = ds.sample_episode(xtr, ftr, idx_by_class, rng, n_way, K_SHOT, Q_QUERY)
        xb = torch.from_numpy(np.concatenate([sx, qx])).to(dev)
        fb = torch.from_numpy(np.concatenate([sf, qf])).to(dev)
        emb = net(xb, fb)
        se, qe = emb[: len(sx)], emb[len(sx):]
        sl_t = torch.from_numpy(sl).to(dev)
        protos = torch.stack([se[sl_t == c].mean(0) for c in range(n_way)])
        logits = -sq_dist(qe, protos) * scale
        loss = torch.nn.functional.cross_entropy(logits, torch.from_numpy(ql).to(dev),
                                                  label_smoothing=label_smoothing)
        opt.zero_grad()
        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip_grad)
        opt.step()
        if ep >= warmup_episodes:
            cosine.step()
        running += loss.item()
        log_n = log_every if log_every is not None else max(1, min(500, episodes))
        if (ep + 1) % log_n == 0:
            cur_lr = opt.param_groups[0]["lr"]
            elapsed = time.perf_counter() - t0
            rate = elapsed / (ep + 1)
            eta_s = rate * (episodes - (ep + 1))
            print(f"  [{log_tag}] ep {ep+1:5d}/{episodes}  loss {running/log_n:.4f}  lr {cur_lr:.2e}  "
                  f"elapsed {elapsed:.1f}s  {rate:.3f}s/ep  eta {eta_s/60:.1f}min", flush=True)
            history["loss"].append(running / log_n)
            history["lr"].append(cur_lr)
            running = 0.0
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
