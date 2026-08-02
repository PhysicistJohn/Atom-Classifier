#!/usr/bin/env python
"""Optuna study over the MLX v7 trainer (short-run proxy objective).

Closes the old G5/task-12 gap: hyperparameter search wired directly to
v7_trainer_mlx.py. Each trial is a 600-episode bf16 run on the production
corpus with a trial-unique training seed; the objective is a dwell-weighted
final balanced accuracy (weighted toward 1 ms, where the headroom is).

Variance measurement is built in: every BASELINE_EVERY-th trial re-runs the
fixed production configuration under a new seed. The spread of those
baseline trials is the training-variance estimate; the TPE search runs on
the same noise floor, so "better than baseline" can be judged against it.

Usage:
  CORPUS_DIR=training/artifacts/longdwell-production-corpus-v2 \
  .venv-training/bin/python tools/optuna-mlx-study.py \
    --hours 19 --out-dir training/artifacts/optuna-mlx-20260802

Summarize an existing study:
  .venv-training/bin/python tools/optuna-mlx-study.py \
    --out-dir training/artifacts/optuna-mlx-20260802 --summarize
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import optuna

REPO = Path(__file__).resolve().parent.parent
TRAINER = REPO / "training/zplane_ab/v2_full_variation/v7_probe/v7_trainer_mlx.py"
PYTHON = REPO / ".venv-training/bin/python"
INIT_CKPT = REPO / "training/artifacts/mlx-g2/step0.safetensors"

EPISODES = 600
BASELINE_EVERY = 8
SEED_BASE = 20270000
DWELL_WEIGHTS = {"1ms": 0.5, "2.5ms": 0.3, "10ms": 0.2}
BASELINE = {"lr": 2e-3, "recon_weight": 0.3, "conf_weight": 0.1,
            "mask_frac": 0.5, "worst_dur_weight": 0.15}
TRIAL_TIMEOUT_S = 40 * 60


def run_trial(params: dict, seed: int, out_json: Path, log: Path) -> dict:
    cmd = [str(PYTHON), str(TRAINER),
           "--episodes", str(EPISODES), "--eval-every", str(EPISODES),
           "--seed", str(seed), "--dtype", "bf16",
           "--ckpt-blocks", "3", "--batch-chunks", "5",
           "--log-every", "200",
           "--lr", str(params["lr"]),
           "--recon-weight", str(params["recon_weight"]),
           "--conf-weight", str(params["conf_weight"]),
           "--mask-frac", str(params["mask_frac"]),
           "--worst-dur-weight", str(params["worst_dur_weight"]),
           "--init-checkpoint", str(INIT_CKPT),
           "--out", str(out_json)]
    with open(log, "ab") as lf:
        subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                       timeout=TRIAL_TIMEOUT_S, check=True,
                       env={**os.environ})
    return json.loads(out_json.read_text())


def objective_value(final: dict) -> float:
    return sum(w * final[d]["balanced_accuracy"]
               for d, w in DWELL_WEIGHTS.items())


def make_objective(out_dir: Path):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "lr": trial.suggest_float("lr", 5e-4, 8e-3, log=True),
            "recon_weight": trial.suggest_float("recon_weight", 0.0, 0.8),
            "conf_weight": trial.suggest_float("conf_weight", 0.0, 0.3),
            "mask_frac": trial.suggest_float("mask_frac", 0.3, 0.7),
            "worst_dur_weight": trial.suggest_float("worst_dur_weight",
                                                    0.0, 0.3),
        }
        seed = SEED_BASE + trial.number
        trial.set_user_attr("seed", seed)
        is_baseline = all(
            abs(params[k] - BASELINE[k]) < 1e-12 for k in BASELINE)
        trial.set_user_attr("baseline", is_baseline)
        out_json = out_dir / f"trial{trial.number:04d}.json"
        log = out_dir / f"trial{trial.number:04d}.log"
        t0 = time.time()
        result = run_trial(params, seed, out_json, log)
        final = result["final"]
        for d in DWELL_WEIGHTS:
            trial.set_user_attr(f"bal_{d}", final[d]["balanced_accuracy"])
            trial.set_user_attr(f"minprof_{d}",
                                final[d]["min_profile_recall"])
        trial.set_user_attr("skipped_steps", result.get("skipped_steps"))
        trial.set_user_attr("wall_s", time.time() - t0)
        val = objective_value(final)
        print(f"trial {trial.number:3d} {'BASELINE' if is_baseline else '  '}"
              f" obj={val:.4f} "
              + " ".join(f"{d}={final[d]['balanced_accuracy']:.4f}"
                         for d in DWELL_WEIGHTS)
              + f" ({time.time()-t0:.0f}s)", flush=True)
        return val
    return objective


def enqueue_baseline_cb(study: optuna.Study,
                        trial: optuna.trial.FrozenTrial) -> None:
    if (trial.number + 1) % BASELINE_EVERY == 0:
        study.enqueue_trial(BASELINE, skip_if_exists=False)


def summarize(study: optuna.Study) -> None:
    done = [t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE]
    base = [t for t in done if t.user_attrs.get("baseline")]
    print(f"{len(done)} complete trials ({len(base)} baseline replicates)")
    if base:
        vals = [t.value for t in base]
        print(f"baseline objective: mean={statistics.mean(vals):.4f} "
              f"std={statistics.stdev(vals) if len(vals) > 1 else 0:.4f} "
              f"range=[{min(vals):.4f}, {max(vals):.4f}] n={len(vals)}")
        for d in DWELL_WEIGHTS:
            bv = [t.user_attrs[f"bal_{d}"] for t in base]
            print(f"  baseline {d}: mean={statistics.mean(bv):.4f} "
                  f"std={statistics.stdev(bv) if len(bv) > 1 else 0:.4f}")
    if done:
        best = study.best_trial
        print(f"best: trial {best.number} obj={best.value:.4f} "
              f"params={best.params} seed={best.user_attrs.get('seed')}")
        top = sorted(done, key=lambda t: t.value, reverse=True)[:5]
        for t in top:
            print(f"  #{t.number:3d} obj={t.value:.4f} "
                  f"{'BASELINE' if t.user_attrs.get('baseline') else ''} "
                  f"lr={t.params['lr']:.2e} rw={t.params['recon_weight']:.2f}"
                  f" cw={t.params['conf_weight']:.2f}"
                  f" mf={t.params['mask_frac']:.2f}"
                  f" wdw={t.params['worst_dur_weight']:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hours", type=float, default=19.0)
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name="dacs-v7-mlx-short600",
        storage=f"sqlite:///{out_dir}/study.db",
        direction="maximize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=20260802,
                                           n_startup_trials=12))
    if args.summarize:
        summarize(study)
        return
    if not any(t.user_attrs.get("baseline") for t in study.trials):
        study.enqueue_trial(BASELINE)
    study.optimize(make_objective(out_dir),
                   timeout=args.hours * 3600,
                   callbacks=[enqueue_baseline_cb],
                   catch=(subprocess.TimeoutExpired,
                          subprocess.CalledProcessError))
    summarize(study)


if __name__ == "__main__":
    main()
