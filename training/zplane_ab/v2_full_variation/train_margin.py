"""Episodic prototypical training + an explicit PROTOTYPE-REPULSION margin
term, on top of the existing cross-entropy-over-distances loss (see
../train_common.py's train() for the unmodified baseline this extends).

WHY: iterate_cnn.py's vit_run1 -> vit_run2 showed that plain accuracy pressure
does not keep confusable classes apart -- giving the ViT enough training to fix
bluetooth's own internal (clean-vs-impaired) cohesion caused its prototype to
start absorbing gsm (236/614 -> bluetooth) and dsss (213/573 -> bluetooth)
samples. Cross-entropy over distances only rewards "closest to your own
prototype," it has no term rewarding "and stay away from other prototypes,"
so one class's cluster can drift into and swallow its neighbors as a side
effect of training, with nothing to stop it.

Fix: every episode, after computing the 7 class prototypes, add a hinge-margin
repulsion term over every prototype PAIR: max(0, MARGIN - squared_dist(p_i,
p_j))^2, so pairs closer than MARGIN are explicitly pushed apart. The
bluetooth/gsm/dsss triad (the specific cluster identified from the confusion
matrices) gets TRIAD_WEIGHT x the base weight, since that is the actual
problem being targeted -- the other 18 pairs still get a (smaller) baseline
repulsion so fixing this triad doesn't let some other pair drift together.

Embeddings are L2-normalized (unit norm) by both model.Embedding and
ZPlaneEmbedding/ViTEmbedding's forward(), so squared distance between two unit
vectors ranges [0, 4]; MARGIN defaults to 0.5 (modest, not "as far apart as
possible" -- overly large margins on unit-norm embeddings ask for near-
antipodal prototypes for every one of 21 pairs simultaneously, which is
geometrically overconstrained in a 32-d space and would fight the
classification loss instead of complementing it).
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
LR = 1e-3
WD = 2e-4
MARGIN = 0.5
TRIAD_WEIGHT = 3.0
LAMBDA_REPEL = 0.5


def pairwise_sq_dist(protos: torch.Tensor) -> torch.Tensor:
    """protos: (C, D) -> (C, C) squared euclidean distance matrix."""
    diff = protos.unsqueeze(0) - protos.unsqueeze(1)  # (C, C, D)
    return (diff ** 2).sum(-1)


def repulsion_loss(protos: torch.Tensor, classes: list, triad: set,
                    margin: float = MARGIN, triad_weight: float = TRIAD_WEIGHT) -> torch.Tensor:
    C = protos.shape[0]
    d2 = pairwise_sq_dist(protos)
    hinge = torch.clamp(margin - d2, min=0.0) ** 2
    weight = torch.ones(C, C, device=protos.device)
    triad_idx = [i for i, c in enumerate(classes) if c in triad]
    for i in triad_idx:
        for j in triad_idx:
            if i != j:
                weight[i, j] = triad_weight
    mask = 1.0 - torch.eye(C, device=protos.device)  # exclude self-pairs (i==j, always 0 anyway)
    weighted = hinge * weight * mask
    n_pairs = C * (C - 1)
    return weighted.sum() / n_pairs


def train_with_margin(net, data, dev, episodes: int, eval_every: int, log_tag: str = "",
                       triad: set = frozenset({"bluetooth", "gsm", "dsss"}),
                       margin: float = MARGIN, triad_weight: float = TRIAD_WEIGHT,
                       lambda_repel: float = LAMBDA_REPEL):
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

    print(f"[{log_tag}] MARGIN training {episodes} episodes ({n_way}-way {K_SHOT}-shot) ... "
          f"margin={margin} triad_weight={triad_weight} lambda_repel={lambda_repel} triad={sorted(triad)}")
    running_ce, running_rep, best_acc, best_state = 0.0, 0.0, -1.0, None
    history = {"loss_ce": [], "loss_repel": [], "val_acc": [], "min_triad_dist": []}
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
        loss_rep = repulsion_loss(protos, classes, triad, margin=margin, triad_weight=triad_weight)
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
            with torch.no_grad():
                d2 = pairwise_sq_dist(protos).sqrt()
                triad_idx = [i for i, c in enumerate(classes) if c in triad]
                triad_pairs = [d2[i, j].item() for i in triad_idx for j in triad_idx if i != j]
                min_triad_dist = min(triad_pairs) if triad_pairs else float("nan")
            history["val_acc"].append({"episode": ep + 1, "acc": acc})
            history["min_triad_dist"].append({"episode": ep + 1, "dist": min_triad_dist})
            tag = ""
            if acc > best_acc:
                best_acc, best_state = acc, copy.deepcopy(net.state_dict())
                tag = "  <- best"
            print(f"    [{log_tag}] [val(clean) {acc:.3f}]{tag}  min_triad_proto_dist={min_triad_dist:.3f}")
    wall = time.perf_counter() - t0
    if best_state is not None:
        net.load_state_dict(best_state)
    history["best_val_acc"] = best_acc
    history["wall_clock_s"] = wall
    history["episodes"] = episodes
    return net, history, wall
