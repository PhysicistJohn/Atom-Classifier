"""Re-run the z-plane backbone with train_common_stabilized's 3 fixes (lower
backbone LR, higher weight-decay on the learned distance-scale, tighter grad
clipping) after the original v2 zplane_resampled run collapsed after episode
1666. Writes to a SEPARATE artifacts dir (zplane_{condition}_stabilized) so
the original collapsed run's artifacts/manifest are preserved untouched as
evidence.

Run:
  .venv-training/bin/python training/zplane_ab/v2_full_variation/train_zplane_stabilized.py \
      --condition {resampled,native} --episodes 10000 [--eval-every N]
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

from zplane_backbone import ZPlaneEmbedding  # noqa: E402 -- imported, never modified
import train_common_v2 as tcv2  # noqa: E402 -- prepare_data_v2/count_params/infer_latency reused unchanged
from train_common_stabilized import train_stabilized, BACKBONE_LR, HEAD_LR, SCALE_WD, CLIP_GRAD_STABILIZED  # noqa: E402

ARTIFACTS = os.path.join(_V2_DIR, "artifacts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["resampled", "native"], required=True)
    ap.add_argument("--episodes", type=int, default=10000)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--sections", type=int, default=8)
    ap.add_argument("--warm-radius", type=float, default=0.5)
    args = ap.parse_args()

    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"[zplane-{args.condition}-stabilized] training device: {dev}")

    data = tcv2.prepare_data_v2(condition=args.condition)

    net = ZPlaneEmbedding(width=args.width, layers=args.layers, sections=args.sections,
                          warm_radius=args.warm_radius)
    n_params = tcv2.count_params(net)
    print(f"[zplane-{args.condition}-stabilized] param count: {n_params}  input_length={data['input_length']}")

    eval_every = args.eval_every if args.eval_every is not None else max(200, args.episodes // 6)
    net, history, wall = train_stabilized(net, data, dev, episodes=args.episodes, eval_every=eval_every,
                                           log_tag=f"zplane-{args.condition}-stabilized")

    print(f"[zplane-{args.condition}-stabilized] episodes_actually_completed: {history['episodes']}")

    net = net.to("cpu")
    latency = tcv2.infer_latency_ms_per_1k(net, data, torch.device("cpu"))
    print(f"[zplane-{args.condition}-stabilized] embed_all latency (CPU): {latency:.3f} ms/1k samples")

    out_dir = os.path.join(ARTIFACTS, f"zplane_{args.condition}_stabilized")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(out_dir, "state_dict.pt"))

    manifest = {
        "backbone": "zplane",
        "condition": args.condition,
        "variant": "stabilized",
        "config": {"width": args.width, "layers": args.layers, "sections": args.sections,
                    "warm_radius": args.warm_radius,
                    "backbone_lr": BACKBONE_LR, "head_lr": HEAD_LR, "scale_wd": SCALE_WD,
                    "clip_grad": CLIP_GRAD_STABILIZED},
        "seed": __import__("full_split").SEED_V2,
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
