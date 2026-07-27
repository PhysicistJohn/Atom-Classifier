"""Shared episodic training harness for BOTH the fresh-CNN-baseline retrain and
the z-plane run. Factoring this out (rather than duplicating the loop in two
files) guarantees the two runs use the exact same optimizer setup, episode
sampling calls, and eval cadence — differing only in which `nn.Module` is
plugged in — which is the actual apples-to-apples property this A/B needs.

This mirrors training/train_signallab.py's main() loop faithfully (same
Adam/CosineAnnealingLR/scale-parameter setup, same build_pool scale-jitter
choices, same eval-on-clean-val/keep-best pattern); it is a new file, not a
modification of train_signallab.py.
"""

from __future__ import annotations

import copy
import os
import sys
import time

import numpy as np
import torch

_ZPAB_DIR = os.path.dirname(os.path.abspath(__file__))
_TRAINING_DIR = os.path.dirname(_ZPAB_DIR)
for p in (_TRAINING_DIR, _ZPAB_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import dataset as ds  # noqa: E402
from train import embed_all, prototypes_from, nearest, sq_dist  # noqa: E402
from common_split import load_and_split, build_pool  # noqa: E402

K_SHOT = 5
Q_QUERY = 5
LR = 1e-3
WD = 2e-4


def count_params(net) -> int:
    return sum(p.numel() for p in net.parameters())


def prepare_data(rng_seed_note=""):
    """Load corpus + split (verbatim, shared), build train/enroll/val pools,
    exactly matching train_signallab.py's build_pool scale-jitter choices
    (0.08 train, 0.0 enroll+val). Returns everything both training scripts
    need, including the still-advancing rng so subsequent episode sampling in
    the training loop continues the same deterministic sequence."""
    iq, y, impaired, classes, tr_idx, en_idx, va_idx, rng = load_and_split()
    n_classes = len(classes)
    print(f"corpus {iq.shape} | impaired {int(impaired.sum())}/{len(impaired)} | classes {classes}")
    print("building pools ...")
    xtr, ftr, ytr = build_pool(iq, y, tr_idx, rng, scale_jitter=0.08)
    fmean = ftr.mean(0)
    fstd = ftr.std(0) + 1e-6
    ftr = (ftr - fmean) / fstd
    idx_by_class = ds.class_indices(ytr, n_classes)
    xen, fen, yen = build_pool(iq, y, en_idx, rng, scale_jitter=0.0)
    fen = (fen - fmean) / fstd
    xva, fva, yva = build_pool(iq, y, va_idx, rng, scale_jitter=0.0)
    fva = (fva - fmean) / fstd
    print(f"  train {xtr.shape}  enroll {xen.shape}  val {xva.shape}")
    return dict(
        iq=iq, y=y, impaired=impaired, classes=classes, n_classes=n_classes,
        tr_idx=tr_idx, en_idx=en_idx, va_idx=va_idx, rng=rng,
        xtr=xtr, ftr=ftr, ytr=ytr, idx_by_class=idx_by_class,
        xen=xen, fen=fen, yen=yen, xva=xva, fva=fva, yva=yva,
        fmean=fmean, fstd=fstd,
    )


def train(net, data, dev, episodes: int, eval_every: int, clip_grad: float | None = None,
          log_tag: str = ""):
    """Runs the episodic prototypical-CE training loop. Returns
    (net_with_best_state_loaded, history_dict, wall_clock_seconds)."""
    n_classes = data["n_classes"]
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

    print(f"[{log_tag}] training {episodes} episodes ({n_way}-way {K_SHOT}-shot) ...")
    running, best_acc, best_state = 0.0, -1.0, None
    history = {"loss": [], "val_acc": []}
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
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip_grad)
        opt.step()
        sched.step()
        running += loss.item()
        if (ep + 1) % max(1, min(500, episodes)) == 0:
            n = max(1, min(500, episodes))
            print(f"  [{log_tag}] ep {ep+1:5d}  loss {running/n:.4f}")
            history["loss"].append(running / n)
            running = 0.0
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


def timing_probe(net_factory, data, dev, n_probe: int = 50, clip_grad: float | None = None) -> float:
    """Runs n_probe episodes and returns average ms/episode (excludes model
    construction time)."""
    net = net_factory().to(dev)
    scale = torch.nn.Parameter(torch.tensor(10.0, device=dev))
    opt = torch.optim.Adam(list(net.parameters()) + [scale], lr=LR, weight_decay=WD)
    n_classes = data["n_classes"]
    rng = np.random.default_rng(12345)  # separate probe rng, does not disturb the real training rng
    xtr, ftr, idx_by_class = data["xtr"], data["ftr"], data["idx_by_class"]
    t0 = time.perf_counter()
    for _ in range(n_probe):
        net.train()
        sx, sf, sl, qx, qf, ql = ds.sample_episode(xtr, ftr, idx_by_class, rng, n_classes, K_SHOT, Q_QUERY)
        xb = torch.from_numpy(np.concatenate([sx, qx])).to(dev)
        fb = torch.from_numpy(np.concatenate([sf, qf])).to(dev)
        emb = net(xb, fb)
        se, qe = emb[: len(sx)], emb[len(sx):]
        sl_t = torch.from_numpy(sl).to(dev)
        protos = torch.stack([se[sl_t == c].mean(0) for c in range(n_classes)])
        logits = -sq_dist(qe, protos) * scale
        loss = torch.nn.functional.cross_entropy(logits, torch.from_numpy(ql).to(dev))
        opt.zero_grad()
        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip_grad)
        opt.step()
    wall = time.perf_counter() - t0
    return (wall / n_probe) * 1000.0


def infer_latency_ms_per_1k(net, data, dev, n_samples: int = 256) -> float:
    """embed_all inference latency, normalized to ms per 1000 samples, measured
    on the validation pool (repeats/truncates to n_samples)."""
    xva, fva = data["xva"], data["fva"]
    reps = max(1, -(-n_samples // len(xva)))
    x = np.tile(xva, (reps, 1, 1))[:n_samples]
    f = np.tile(fva, (reps, 1))[:n_samples]
    net.eval()
    # warm-up
    embed_all(net, x[:min(32, len(x))], f[:min(32, len(x))], dev)
    t0 = time.perf_counter()
    embed_all(net, x, f, dev)
    wall = time.perf_counter() - t0
    seconds_per_sample = wall / len(x)
    return seconds_per_sample * 1000.0 * 1000.0  # ms/sample * 1000 samples = ms per 1k samples
