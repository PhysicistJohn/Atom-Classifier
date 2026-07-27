"""Per-SAMPLE contrastive repulsion, not just prototype-mean repulsion (see
train_margin.py's docstring for why the mean-only version -- tested in
iterate_margin.py's margin_run1 -- left the bluetooth/gsm/dsss confusion
counts essentially unchanged: pushing prototype MEANS apart does nothing if
the underlying per-sample embedding DISTRIBUTIONS for those classes overlap,
which is consistent with what the confusion matrix showed).

This version takes every pair of classes in the triad and, within each
episode, computes a hinge-margin penalty over EVERY individual embedding pair
across the two classes (all support+query embeddings, not just their means):
max(0, MARGIN - dist(x_i, x_j))^2 for x_i in class i's embeddings, x_j in
class j's. With K_SHOT=Q_QUERY=5 that's 10 embeddings/class, so 100 pairs per
triad class-pair, 3 pairs = 300 extra pairwise terms/episode -- trivial cost.
This directly discourages individual samples from landing near a different
class's samples, which prototype-mean repulsion cannot do.
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
from train_margin import pairwise_sq_dist  # noqa: E402 -- reused for the eval-time triad-distance diagnostic

K_SHOT = 5
Q_QUERY = 5
LR = 1e-3
WD = 2e-4
MARGIN = 0.75
LAMBDA_REPEL = 2.0


def sample_repulsion_loss(all_emb: torch.Tensor, all_labels: np.ndarray, classes: list,
                           triad: set, margin: float, device) -> torch.Tensor:
    """all_emb: (N, D) every support+query embedding in the episode (both
    classes' full sample sets, not means). all_labels: (N,) int class index
    per embedding (support labels + query labels concatenated, matching
    all_emb's row order)."""
    triad_idx = [i for i, c in enumerate(classes) if c in triad]
    total = torch.tensor(0.0, device=device)
    n_pairs_total = 0
    for a_pos in range(len(triad_idx)):
        for b_pos in range(a_pos + 1, len(triad_idx)):
            ci, cj = triad_idx[a_pos], triad_idx[b_pos]
            xi = all_emb[all_labels == ci]
            xj = all_emb[all_labels == cj]
            if xi.shape[0] == 0 or xj.shape[0] == 0:
                continue
            diff = xi.unsqueeze(1) - xj.unsqueeze(0)          # (ni, nj, D)
            d2 = (diff ** 2).sum(-1)                           # (ni, nj)
            hinge = torch.clamp(margin - d2, min=0.0) ** 2
            total = total + hinge.sum()
            n_pairs_total += xi.shape[0] * xj.shape[0]
    if n_pairs_total == 0:
        return total
    return total / n_pairs_total


def train_with_sample_margin(net, data, dev, episodes: int, eval_every: int, log_tag: str = "",
                              triad: set = frozenset({"bluetooth", "gsm", "dsss"}),
                              margin: float = MARGIN, lambda_repel: float = LAMBDA_REPEL):
    n_classes = data["n_classes"]
    classes = data["classes"]
    rng = data["rng"]
    xtr, ftr, ytr, idx_by_class = data["xtr"], data["ftr"], data["ytr"], data["idx_by_class"]
    xen, fen, yen = data["xen"], data["fen"], data["yen"]
    xva, fva, yva = data["xva"], data["fva"], data["yva"]
    n_way = n_classes

    net = net.to(dev)
    scale = torch.nn.Parameter(torch.tensor(10.0, device=dev))
    opt = torch.optim.Adam(list(net.parameters()) + [scale], lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(episodes, 1))

    def eval_clean():
        protos = prototypes_from(embed_all(net, xen, fen, dev), yen, n_classes)
        pred, _ = nearest(embed_all(net, xva, fva, dev), protos)
        return float((pred == yva).mean())

    print(f"[{log_tag}] SAMPLE-MARGIN training {episodes} episodes ({n_way}-way {K_SHOT}-shot) ... "
          f"margin={margin} lambda_repel={lambda_repel} triad={sorted(triad)}")
    running_ce, running_rep, best_acc, best_state = 0.0, 0.0, -1.0, None
    history = {"loss_ce": [], "loss_repel": [], "val_acc": []}
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
        loss_ce = torch.nn.functional.cross_entropy(logits, torch.from_numpy(ql).to(dev))

        all_labels = np.concatenate([sl, ql])
        loss_rep = sample_repulsion_loss(emb, all_labels, classes, triad, margin, dev)
        loss = loss_ce + lambda_repel * loss_rep
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        running_ce += loss_ce.item()
        running_rep += loss_rep.item()
        if (ep + 1) % max(1, min(500, episodes)) == 0:
            n = max(1, min(500, episodes))
            print(f"  [{log_tag}] ep {ep+1:5d}  ce {running_ce/n:.4f}  repel {running_rep/n:.4f}")
            history["loss_ce"].append(running_ce / n)
            history["loss_repel"].append(running_rep / n)
            running_ce, running_rep = 0.0, 0.0
        if (ep + 1) % eval_every == 0 or (ep + 1) == episodes:
            acc = eval_clean()
            history["val_acc"].append({"episode": ep + 1, "acc": acc})
            tag = ""
            if acc > best_acc:
                best_acc, best_state = acc, copy.deepcopy(net.state_dict())
                tag = "  <- best"
            print(f"    [{log_tag}] [val(clean) {acc:.3f}]{tag}")
    wall = time.perf_counter() - t0
    if best_state is not None:
        net.load_state_dict(best_state)
    history["best_val_acc"] = best_acc
    history["wall_clock_s"] = wall
    history["episodes"] = episodes
    return net, history, wall
