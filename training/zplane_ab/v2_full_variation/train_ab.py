"""The single parameterized v2 training entry point -- replaces what would
otherwise be 4 near-duplicate scripts (one per backbone x condition cell).

Run:
  .venv-training/bin/python training/zplane_ab/v2_full_variation/train_ab.py \
      --backbone {cnn,zplane} --condition {resampled,native} \
      --episodes 10000 [--eval-every N] [--width 64 --layers 3 --sections 8 --warm-radius 0.5]

Both backbones train on the SAME device in v2 (unlike v1's CNN-forced-CPU) --
symmetric device choice keeps wall_clock_train_s comparable within a
condition. Final accuracy/inference-latency measurement in evaluate_ab_v2.py
still hard-forces CPU for both, exactly as v1's evaluate_ab.py did, so THOSE
numbers stay device-controlled regardless of what device training used.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

_V2_DIR = os.path.dirname(os.path.abspath(__file__))
_ZPAB_DIR = os.path.dirname(_V2_DIR)
_TRAINING_DIR = os.path.dirname(_ZPAB_DIR)
for p in (_TRAINING_DIR, _ZPAB_DIR, _V2_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from model import Embedding, EMBED_DIM, N_FEATURES  # noqa: E402 -- imported, never modified
from zplane_backbone import ZPlaneEmbedding  # noqa: E402 -- imported, never modified
import train_common_v2 as tcv2  # noqa: E402

ARTIFACTS = os.path.join(_V2_DIR, "artifacts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=["cnn", "zplane"], required=True)
    ap.add_argument("--condition", choices=["resampled", "native"], required=True)
    ap.add_argument("--episodes", type=int, default=10000)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--sections", type=int, default=8)
    ap.add_argument("--warm-radius", type=float, default=0.5)
    args = ap.parse_args()

    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"[{args.backbone}-{args.condition}] training device: {dev}")

    data = tcv2.prepare_data_v2(condition=args.condition)
    n_classes = data["n_classes"]

    if args.backbone == "cnn":
        def net_factory():
            return Embedding(EMBED_DIM, N_FEATURES)
        clip_grad = None
    else:
        def net_factory():
            return ZPlaneEmbedding(width=args.width, layers=args.layers, sections=args.sections,
                                    warm_radius=args.warm_radius)
        clip_grad = 1.0

    net = net_factory()
    n_params = tcv2.count_params(net)
    print(f"[{args.backbone}-{args.condition}] param count: {n_params}  input_length={data['input_length']}")

    eval_every = args.eval_every if args.eval_every is not None else max(200, args.episodes // 6)
    net, history, wall = tcv2.train(net, data, dev, episodes=args.episodes, eval_every=eval_every,
                                     clip_grad=clip_grad, log_tag=f"{args.backbone}-{args.condition}")

    print(f"[{args.backbone}-{args.condition}] episodes_actually_completed: {history['episodes']}")

    # tc.train() leaves net on the training device (mps or cpu); move to CPU
    # explicitly before measuring CPU latency -- otherwise embed_all's inputs
    # (moved to cpu by infer_latency_ms_per_1k) mismatch the net's own
    # still-mps parameters.
    net = net.to("cpu")
    latency = tcv2.infer_latency_ms_per_1k(net, data, torch.device("cpu"))  # CPU, matching v1 convention
    print(f"[{args.backbone}-{args.condition}] embed_all latency (CPU): {latency:.3f} ms/1k samples")

    out_dir = os.path.join(ARTIFACTS, f"{args.backbone}_{args.condition}")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(out_dir, "state_dict.pt"))

    manifest = {
        "backbone": args.backbone,
        "condition": args.condition,
        "config": ({"width": args.width, "layers": args.layers, "sections": args.sections,
                     "warm_radius": args.warm_radius, "clip_grad": clip_grad}
                    if args.backbone == "zplane" else {}),
        "seed": tcv2.__dict__.get("SEED_V2", None) or __import__("full_split").SEED_V2,
        "classes": data["classes"],
        "input_length": data["input_length"],
        "episodes_requested": args.episodes,
        "episodes_actually_completed": history["episodes"],
        "eval_every": eval_every,
        "param_count": n_params,
        "wall_clock_train_s": wall,
        "infer_latency_ms_per_1k_cpu": latency,
        "training_device": str(dev),
        "best_val_acc_during_training": history["best_val_acc"],
        "history": history,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("done. artifacts ->", os.path.relpath(out_dir))


if __name__ == "__main__":
    main()
