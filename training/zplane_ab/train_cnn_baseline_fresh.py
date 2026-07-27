"""Fresh retrain of the PRODUCTION CNN backbone (model.Embedding, unmodified,
imported not copied) under the shared zplane_ab harness, so the A/B baseline
number comes from the identical harness/split/eval protocol as the z-plane
run — not from the possibly-stale shipped model-manifest.json.

Differences from training/train_signallab.py's main(), and ONLY these:
  (a) split comes from common_split.load_and_split() instead of being inlined,
  (b) all outputs are redirected to training/zplane_ab/artifacts/cnn_baseline/
      instead of src/embedding/assets/ (nothing under src/embedding/assets/ or
      any other production path is touched by this script),
  (c) device is forced to CPU explicitly (task's environment is CPU-only via
      .venv-training; production's train_signallab.py auto-detects MPS).

Same EPISODES/K_SHOT/Q_QUERY/Adam/CosineAnnealingLR/scale-parameter setup as
production, same build_pool scale-jitter choices (0.08 train / 0.0 enroll+val).

Run:  .venv-training/bin/python training/zplane_ab/train_cnn_baseline_fresh.py [--episodes N] [--eval-every N] [--probe]
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

from model import Embedding, EMBED_DIM, N_FEATURES  # noqa: E402 -- imported, never modified
import train_common as tc  # noqa: E402

OUT_DIR = os.path.join(_ZPAB_DIR, "artifacts", "cnn_baseline")
PRODUCTION_EPISODES = 6000  # train_signallab.py's EPISODES, for reference/labeling


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=None, help="override episode budget")
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--probe", action="store_true", help="run a 50-episode timing probe and exit")
    args = ap.parse_args()

    dev = torch.device("cpu")  # forced per reproducibility plan §3.5
    print(f"device: {dev} (forced CPU)")

    data = tc.prepare_data()
    n_classes = data["n_classes"]

    def net_factory():
        return Embedding(EMBED_DIM, N_FEATURES)  # in_channels defaults to 2 (I,Q) -- production config

    if args.probe:
        ms = tc.timing_probe(net_factory, data, dev, n_probe=50)
        print(f"[cnn] timing probe: {ms:.2f} ms/episode")
        return

    episodes = args.episodes if args.episodes is not None else PRODUCTION_EPISODES
    eval_every = args.eval_every if args.eval_every is not None else max(200, episodes // 6)

    net = net_factory()
    n_params = tc.count_params(net)
    print(f"[cnn] param count: {n_params}")

    net, history, wall = tc.train(net, data, dev, episodes=episodes, eval_every=eval_every,
                                   clip_grad=None, log_tag="cnn")

    # final enroll + val on clean pools (identical metric to production)
    from train import embed_all, prototypes_from, nearest  # noqa: E402
    emb_en = embed_all(net, data["xen"], data["fen"], dev)
    protos = prototypes_from(emb_en, data["yen"], n_classes)
    emb_va = embed_all(net, data["xva"], data["fva"], dev)
    pred, dists = nearest(emb_va, protos)
    closed_acc = float((pred == data["yva"]).mean())
    classes = data["classes"]
    per_class = {classes[c]: round(float((pred[data["yva"] == c] == c).mean()), 3) for c in range(n_classes)}
    print(f"[cnn] clean closed-set accuracy: {closed_acc:.3f}")
    print("[cnn] per class:", per_class)

    latency = tc.infer_latency_ms_per_1k(net, data, dev)
    print(f"[cnn] embed_all latency: {latency:.3f} ms/1k samples")

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(OUT_DIR, "state_dict.pt"))
    np.save(os.path.join(OUT_DIR, "prototypes.npy"), protos)
    manifest = {
        "backbone": "cnn_production_fresh",
        "seed": tc.__dict__.get("SEED", 20260721),
        "classes": classes,
        "episodes_requested": episodes,
        "episodes_production_reference": PRODUCTION_EPISODES,
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
