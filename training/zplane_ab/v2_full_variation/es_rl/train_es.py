"""ES/RL training entry point: evolves EVERY parameter of ZPlaneEmbedding
(~56,256 params -- lift/mix/proj/kernel-bank/thresh AND the fc1/fc2
classification head, no frozen subset) via sep-CMA-ES (sep_cma_es.py),
gradient-free, mirroring cma.py's train_operator loop pattern (fixed
validation batch drawn once from a disjoint seed, common-random-numbers per
generation, keep best-ever mean against the held-out validation batch).

Reuses full_split.py / native_preprocess.py / train_common_v2.prepare_data_v2
verbatim, unmodified -- same impairment-inclusive split, same data pipeline as
every other cell in this A/B, for a fair comparison.

Run (smoke test, seconds):
  .venv-training/bin/python training/zplane_ab/v2_full_variation/es_rl/train_es.py \
      --condition resampled --smoke

Run (full):
  .venv-training/bin/python training/zplane_ab/v2_full_variation/es_rl/train_es.py \
      --condition resampled --generations 120 --popsize 48 --batch 8 --sigma0 1.0 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.utils as nn_utils

_ES_DIR = os.path.dirname(os.path.abspath(__file__))
_V2_DIR = os.path.dirname(_ES_DIR)
_ZPAB_DIR = os.path.dirname(_V2_DIR)
_TRAINING_DIR = os.path.dirname(_ZPAB_DIR)
for p in (_TRAINING_DIR, _ZPAB_DIR, _V2_DIR, _ES_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import dataset as ds  # noqa: E402
from zplane_backbone import ZPlaneEmbedding  # noqa: E402 -- imported, never modified
import train_common_v2 as tcv2  # noqa: E402 -- imported, never modified

from sep_cma_es import SepCMAES, build_d0  # noqa: E402
from fitness import episode_fitness  # noqa: E402

K_SHOT = 5
Q_QUERY = 5
ARTIFACTS = os.path.join(_ES_DIR, "artifacts")


def device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def train_operator_es(net_factory, data, dev, generations: int, batch: int, popsize: int,
                       sigma0: float, seed: int, val_batch: int | None = None, log_every: int = 1):
    n_way = data["n_classes"]
    xtr, ftr, idx_by_class = data["xtr"], data["ftr"], data["idx_by_class"]
    xva, fva, yva = data["xva"], data["fva"], data["yva"]
    idx_by_class_va = ds.class_indices(yva, n_way)
    val_batch = val_batch or batch

    net = net_factory().to(dev)
    x0 = nn_utils.parameters_to_vector(net.parameters()).detach().cpu().double().numpy()
    d0 = build_d0(net)
    es = SepCMAES(x0, d0, sigma0=sigma0, popsize=popsize, seed=seed)
    print(f"[es] n_params={es.n}  lambda={es.lam}  mu={es.mu}  mueff={es.mueff:.2f}  "
          f"cc={es.cc:.3g}  cs={es.cs:.3g}  c1={es.c1:.3g}  cmu={es.cmu:.3g}  damps={es.damps:.3g}")

    val_rng = np.random.default_rng(seed + 10_000)  # fixed, disjoint (real held-out va_idx), never redrawn
    val_episodes = [ds.sample_episode(xva, fva, idx_by_class_va, val_rng, n_way, K_SHOT, Q_QUERY)
                    for _ in range(val_batch)]

    warm_ce, warm_acc = episode_fitness(net, dev, val_episodes, n_way)  # generation-0 reference point
    print(f"[es] warm-start baseline: val_ce={warm_ce:.4f} val_acc={warm_acc:.4f}")

    gen_rng = np.random.default_rng(seed + 1)
    history = {
        "val_ce": [], "val_acc": [], "sigma": [], "d_min": [], "d_max": [],
        "warm_start_ce": warm_ce, "warm_start_acc": warm_acc,
    }
    best_vec, best_ce = None, np.inf
    t0 = time.perf_counter()

    for g in range(generations):
        train_episodes = [ds.sample_episode(xtr, ftr, idx_by_class, gen_rng, n_way, K_SHOT, Q_QUERY)
                           for _ in range(batch)]  # CRN: same batch scores whole generation
        candidates = es.ask()
        fitnesses = []
        for x in candidates:
            nn_utils.vector_to_parameters(torch.from_numpy(x).float().to(dev), net.parameters())
            ce, _ = episode_fitness(net, dev, train_episodes, n_way)
            fitnesses.append(ce)
        es.tell(candidates, fitnesses)

        nn_utils.vector_to_parameters(torch.from_numpy(es.mean).float().to(dev), net.parameters())
        val_ce, val_acc = episode_fitness(net, dev, val_episodes, n_way)
        for k, v in [("val_ce", val_ce), ("val_acc", val_acc), ("sigma", es.sigma),
                     ("d_min", float(es.D.min())), ("d_max", float(es.D.max()))]:
            history[k].append(v)
        if val_ce < best_ce:
            best_ce, best_vec = val_ce, es.mean.copy()
        if (g + 1) % log_every == 0 or (g + 1) == generations:
            elapsed = time.perf_counter() - t0
            print(f"[es] gen {g+1:4d}/{generations}  val_ce={val_ce:.4f}  val_acc={val_acc:.4f}"
                  f"  sigma={es.sigma:.4g}  D[min,max]=[{es.D.min():.3g},{es.D.max():.3g}]"
                  f"  elapsed={elapsed:.1f}s")

    wall = time.perf_counter() - t0
    final_vec = best_vec if best_vec is not None else es.mean
    nn_utils.vector_to_parameters(torch.from_numpy(final_vec).float().to(dev), net.parameters())
    history["best_val_ce"] = best_ce
    history["best_val_acc_at_best_ce"] = None
    # locate the val_acc recorded at the generation where best_ce was achieved
    if best_ce in history["val_ce"]:
        idx = history["val_ce"].index(best_ce)
        history["best_val_acc_at_best_ce"] = history["val_acc"][idx]
    history["wall_clock_s"] = wall
    history["generations"] = generations
    return net, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["resampled", "native"], required=True)
    ap.add_argument("--generations", type=int, default=120)
    ap.add_argument("--popsize", type=int, default=48)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--val-batch", type=int, default=None)
    ap.add_argument("--sigma0", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--sections", type=int, default=8)
    ap.add_argument("--warm-radius", type=float, default=0.5)
    ap.add_argument("--smoke", action="store_true", help="override to generations=5, popsize=16, batch=4")
    ap.add_argument("--log-every", type=int, default=1)
    args = ap.parse_args()

    if args.smoke:
        args.generations, args.popsize, args.batch = 5, 16, 4

    dev = device()
    print(f"[es-{args.condition}] training device: {dev}")

    data = tcv2.prepare_data_v2(condition=args.condition)

    def net_factory():
        return ZPlaneEmbedding(width=args.width, layers=args.layers, sections=args.sections,
                                warm_radius=args.warm_radius)

    n_params = tcv2.count_params(net_factory())
    print(f"[es-{args.condition}] param count: {n_params}  input_length={data['input_length']}")

    net, history = train_operator_es(
        net_factory, data, dev,
        generations=args.generations, batch=args.batch, popsize=args.popsize,
        sigma0=args.sigma0, seed=args.seed, val_batch=args.val_batch, log_every=args.log_every,
    )

    print(f"[es-{args.condition}] done. wall_clock={history['wall_clock_s']:.1f}s  "
          f"warm_start_ce={history['warm_start_ce']:.4f}  best_val_ce={history['best_val_ce']:.4f}  "
          f"warm_start_acc={history['warm_start_acc']:.4f}  "
          f"best_val_acc_at_best_ce={history['best_val_acc_at_best_ce']}")

    # measure CPU inference latency the same way train_ab.py does, for a
    # directly-comparable manifest field
    net_cpu = net.to("cpu")
    latency = tcv2.infer_latency_ms_per_1k(net_cpu, data, torch.device("cpu"))
    print(f"[es-{args.condition}] embed_all latency (CPU): {latency:.3f} ms/1k samples")

    out_dir = os.path.join(ARTIFACTS, f"zplane_es_{args.condition}")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(net_cpu.state_dict(), os.path.join(out_dir, "state_dict.pt"))

    manifest = {
        "backbone": "zplane_es",
        "condition": args.condition,
        "algorithm": "sep-CMA-ES (diagonal covariance, from-scratch, mirrored sampling)",
        "config": {"width": args.width, "layers": args.layers, "sections": args.sections,
                    "warm_radius": args.warm_radius},
        "seed": args.seed,
        "generations": args.generations,
        "popsize": args.popsize,
        "batch": args.batch,
        "val_batch": args.val_batch or args.batch,
        "sigma0": args.sigma0,
        "classes": data["classes"],
        "input_length": data["input_length"],
        "param_count": n_params,
        "wall_clock_train_s": history["wall_clock_s"],
        "infer_latency_ms_per_1k_cpu": latency,
        "training_device": str(dev),
        "warm_start_ce": history["warm_start_ce"],
        "warm_start_acc": history["warm_start_acc"],
        "best_val_ce": history["best_val_ce"],
        "best_val_acc_at_best_ce": history["best_val_acc_at_best_ce"],
        "smoke": bool(args.smoke),
        "history": history,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("done. artifacts ->", os.path.relpath(out_dir, start=_TRAINING_DIR))


if __name__ == "__main__":
    main()
