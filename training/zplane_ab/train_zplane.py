"""Trains ZPlaneEmbedding (the torch-ported z-plane rational-kernel operator
backbone) under the exact same harness as train_cnn_baseline_fresh.py --
same split (common_split), same episodic prototypical-CE loop, same
Adam/CosineAnnealingLR/scale-parameter setup, same eval-on-clean-val/keep-best
pattern -- with `net = ZPlaneEmbedding(...)` in place of `Embedding(...)`,
plus gradient clipping (clip_grad_norm_=1.0) as a cheap safety net for a
complex-valued network with no prior track record on this task.

Run:  .venv-training/bin/python training/zplane_ab/train_zplane.py [--episodes N] [--eval-every N] [--probe]
                                                                     [--width W] [--layers L] [--sections K]
                                                                     [--warm-radius R]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_ZPAB_DIR = os.path.dirname(os.path.abspath(__file__))
_TRAINING_DIR = os.path.dirname(_ZPAB_DIR)
for p in (_TRAINING_DIR, _ZPAB_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from zplane_backbone import ZPlaneEmbedding  # noqa: E402
import train_common as tc  # noqa: E402

OUT_DIR = os.path.join(_ZPAB_DIR, "artifacts", "zplane")
CLIP_GRAD = 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=None, help="override episode budget")
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--probe", action="store_true", help="run a 50-episode timing probe and exit")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--sections", type=int, default=8)
    ap.add_argument("--warm-radius", type=float, default=0.5)
    args = ap.parse_args()

    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"device: {dev}")

    data = tc.prepare_data()
    n_classes = data["n_classes"]

    def net_factory():
        return ZPlaneEmbedding(width=args.width, layers=args.layers, sections=args.sections,
                                warm_radius=args.warm_radius)

    if args.probe:
        ms = tc.timing_probe(net_factory, data, dev, n_probe=50, clip_grad=CLIP_GRAD)
        print(f"[zplane] timing probe: {ms:.2f} ms/episode (width={args.width} layers={args.layers} sections={args.sections})")
        return

    episodes = args.episodes if args.episodes is not None else 1500
    eval_every = args.eval_every if args.eval_every is not None else max(200, episodes // 6)

    net = net_factory()
    n_params = tc.count_params(net)
    print(f"[zplane] param count: {n_params}  (width={args.width} layers={args.layers} sections={args.sections})")

    net, history, wall = tc.train(net, data, dev, episodes=episodes, eval_every=eval_every,
                                   clip_grad=CLIP_GRAD, log_tag="zplane")

    from train import embed_all, prototypes_from, nearest  # noqa: E402
    emb_en = embed_all(net, data["xen"], data["fen"], dev)
    protos = prototypes_from(emb_en, data["yen"], n_classes)
    emb_va = embed_all(net, data["xva"], data["fva"], dev)
    pred, dists = nearest(emb_va, protos)
    closed_acc = float((pred == data["yva"]).mean())
    classes = data["classes"]
    per_class = {classes[c]: round(float((pred[data["yva"] == c] == c).mean()), 3) for c in range(n_classes)}
    print(f"[zplane] clean closed-set accuracy: {closed_acc:.3f}")
    print("[zplane] per class:", per_class)

    latency = tc.infer_latency_ms_per_1k(net, data, dev)
    print(f"[zplane] embed_all latency: {latency:.3f} ms/1k samples")

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(OUT_DIR, "state_dict.pt"))
    np.save(os.path.join(OUT_DIR, "prototypes.npy"), protos)
    manifest = {
        "backbone": "zplane_operator",
        "config": {"width": args.width, "layers": args.layers, "sections": args.sections,
                    "warm_radius": args.warm_radius, "clip_grad": CLIP_GRAD},
        "seed": 20260721,
        "classes": classes,
        "episodes_requested": episodes,
        "eval_every": eval_every,
        "param_count": n_params,
        "wall_clock_train_s": wall,
        "infer_latency_ms_per_1k": latency,
        "clean_closed_set_accuracy": closed_acc,
        "per_class": per_class,
        "best_val_acc_during_training": history["best_val_acc"],
        "history": history,
    }
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("done. artifacts ->", os.path.relpath(OUT_DIR))


if __name__ == "__main__":
    main()
