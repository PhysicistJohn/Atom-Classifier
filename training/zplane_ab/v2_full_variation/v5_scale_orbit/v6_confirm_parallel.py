"""Parallel held-out-seed confirmation for the top v6 study configurations.

Why this exists separately from the study's own confirm stage: that stage runs
its (top-K x seeds) grid sequentially inside one process, which at ~26 minutes
per 24k-episode run would take nearly four hours for nine runs.  This launcher
runs the grid concurrently instead, so it fits in roughly one hour.

Why confirmation matters at all: the study reports the maximum over ~100+ noisy
trials, and a maximum over noise is upward-biased.  The v5 study's own spread at
fixed budget was 0.4531..0.7434, so part of any "best" value is selection luck.
Re-running the top configurations on seeds never used during the search
separates real quality from a seed lottery win.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = Path("/Users/johnelliott/PersonalGitHub/Atom-Classifier")
PY = ROOT / ".venv-training/bin/python"
SINGLE = HERE / "v6_single_run.py"

CONFIRM_SEEDS = (20260901, 20260902, 20260903)
AUX_KEYS = {
    "worst_length_weight",
    "worst_length_profiles",
    "worst_length_ramp_frac",
}


def top_configs(journal: Path, study_name: str, k: int) -> list[dict]:
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
    from optuna.trial import TrialState as S

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(
        study_name=study_name,
        storage=JournalStorage(JournalFileBackend(str(journal))),
    )
    done = [t for t in study.trials if t.state == S.COMPLETE]
    done.sort(key=lambda t: t.value, reverse=True)
    return [
        {"number": t.number, "value": t.value, "params": dict(t.params)}
        for t in done[:k]
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--study-name", default="v6_worst_length_broad")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--episodes", type=int, default=24000)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--concurrency", type=int, default=7)
    p.add_argument("--current-corpus",
                   default="training/artifacts/signallab-current-scale-train-v5-seed20264101-r64")
    p.add_argument("--selection-corpus",
                   default="training/artifacts/signallab-current-scale-dev-v5-seed20262904-r64-identity-firewalled")
    cli = p.parse_args()

    journal = Path(cli.study_dir) / "journal.log"
    out = Path(cli.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    configs = top_configs(journal, cli.study_name, cli.top_k)
    if not configs:
        print("[confirm] no completed trials to confirm", flush=True)
        return
    print(f"[confirm] top-{len(configs)} configs:", flush=True)
    for c in configs:
        print(f"   #{c['number']}: study value {c['value']:.4f}", flush=True)

    jobs = []
    for rank, cfg in enumerate(configs):
        for seed in CONFIRM_SEEDS:
            jobs.append({
                "label": f"confirm_r{rank}_t{cfg['number']}_s{seed}",
                "seed": seed,
                "rank": rank,
                "source_trial": cfg["number"],
                "study_value": cfg["value"],
                "params": cfg["params"],
            })
    (out / "confirm_plan.json").write_text(json.dumps(jobs, indent=2))
    print(f"[confirm] {len(jobs)} runs, concurrency {cli.concurrency}", flush=True)

    env = dict(os.environ,
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               VECLIB_MAXIMUM_THREADS="1")
    running: list[tuple[subprocess.Popen, dict]] = []
    queue = list(jobs)
    started = time.perf_counter()

    while queue or running:
        while queue and len(running) < cli.concurrency:
            job = queue.pop(0)
            cmd = [
                str(PY), str(SINGLE),
                "--current-corpus", cli.current_corpus,
                "--selection-corpus", cli.selection_corpus,
                "--output-dir", str(out),
                "--label", job["label"],
                "--seed", str(job["seed"]),
                "--episodes", str(cli.episodes),
                "--eval-every", str(cli.eval_every),
                "--params-json", json.dumps(job["params"]),
            ]
            log = open(out / f"{job['label']}.log", "w")
            proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
            running.append((proc, job))
            print(f"[confirm] launched {job['label']} pid={proc.pid}", flush=True)
        time.sleep(10)
        still = []
        for proc, job in running:
            if proc.poll() is None:
                still.append((proc, job))
            else:
                print(f"[confirm] finished {job['label']} rc={proc.returncode} "
                      f"({(time.perf_counter()-started)/60:.1f}m elapsed)", flush=True)
        running = still

    # Aggregate
    results = []
    for job in jobs:
        f = out / f"{job['label']}.json"
        if f.exists():
            d = json.loads(f.read_text())
            d.update(rank=job["rank"], source_trial=job["source_trial"],
                     study_value=job["study_value"])
            results.append(d)
    by_rank: dict[int, list[float]] = {}
    for r in results:
        if r.get("status") == "complete":
            by_rank.setdefault(r["rank"], []).append(r["bottleneck"])
    summary = {
        "schema": "v6-confirm-summary-v1",
        "seeds": list(CONFIRM_SEEDS),
        "episodes": cli.episodes,
        "configs": configs,
        "results": results,
        "by_rank": {
            str(k): {
                "n": len(v),
                "mean": sum(v) / len(v),
                "min": min(v),
                "max": max(v),
                "values": v,
                "study_value": next(c["value"] for i, c in enumerate(configs) if i == k),
                "optimism_gap": next(c["value"] for i, c in enumerate(configs) if i == k) - sum(v) / len(v),
            }
            for k, v in by_rank.items()
        },
    }
    (out / "confirm_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\n[confirm] === SUMMARY ===", flush=True)
    for rank, stats in sorted(summary["by_rank"].items()):
        print(f"  rank {rank} (trial #{configs[int(rank)]['number']}): "
              f"study={stats['study_value']:.4f}  held-out mean={stats['mean']:.4f} "
              f"(min {stats['min']:.4f} max {stats['max']:.4f}, n={stats['n']})  "
              f"optimism gap={stats['optimism_gap']:+.4f}", flush=True)


if __name__ == "__main__":
    main()
