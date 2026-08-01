"""Attribution ablation for the worst-length / profile-balanced auxiliary.

Runs at 8,000 episodes with the v5 frozen control hyperparameters, so results
are directly comparable to the existing 51-run v5 baseline distribution at the
same budget (control 0.6814, mean 0.6815, max 0.7434).  Only the auxiliary
weight varies, which makes any movement attributable to the new loss term
rather than to compute, hyperparameters, or search.

Each configuration is repeated over seeds because a single run of this metric
is noisy -- the v5 baseline spread at fixed budget spans 0.4531 to 0.7434.

Writes ``ablation_progress.json`` after every run so progress can be monitored
while it executes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

import v6_harness as H

WEIGHTS = (0.0, 0.05, 0.15, 0.40)
SEEDS = (20260740, 20260741)
EPISODES = 8_000
EVAL_EVERY = 500
RAMP_FRAC = 0.1

# The v5 baseline distribution at this exact budget, from the completed
# 51-run study.  Used only as a reference for interpreting effect size.
BASELINE_REFERENCE = {
    "control": 0.6813909278362462,
    "n_runs": 51,
    "mean": 0.6815,
    "median": 0.6913,
    "max": 0.7434072856861842,
    "min": 0.4531,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-corpus", required=True)
    parser.add_argument("--selection-corpus", required=True)
    parser.add_argument("--historical-corpus", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profiles", type=int, default=12)
    cli = parser.parse_args()

    root = Path(cli.output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    ctx = H.StudyContext(
        current_corpus=cli.current_corpus,
        selection_corpus=cli.selection_corpus,
        historical_corpus=cli.historical_corpus,
        study_root=root,
    )

    records: list[dict] = []
    progress = root / "ablation_progress.json"
    total = len(WEIGHTS) * len(SEEDS)
    started = time.perf_counter()

    for weight in WEIGHTS:
        for seed in SEEDS:
            label = f"w{weight:g}_s{seed}"
            print(f"\n[ablation] === {label} "
                  f"({len(records) + 1}/{total}) ===", flush=True)
            entry: dict = {
                "label": label,
                "weight": weight,
                "seed": seed,
                "episodes": EPISODES,
                "profiles_per_episode": cli.profiles,
                "ramp_frac": RAMP_FRAC if weight > 0 else 0.0,
            }
            try:
                t0 = time.perf_counter()
                result = H.run_trial(
                    dict(H.CONTROL_PARAMS),
                    ctx=ctx,
                    output_dir=root / label,
                    seed=seed,
                    episodes=EPISODES,
                    eval_every=EVAL_EVERY,
                    worst_length_weight=weight,
                    worst_length_profiles=cli.profiles,
                    worst_length_ramp_frac=RAMP_FRAC,
                )
                entry.update(
                    status="complete",
                    wall_s=time.perf_counter() - t0,
                    bottleneck=H.bottleneck_of(result),
                    best_episode=result["training"]["best_episode"],
                    terms=H.terms_of(result),
                    trajectory=[
                        {
                            "episode": e["episode"],
                            "bottleneck": e["checkpoint_score"][0],
                        }
                        for e in result["training"]["evaluation"]
                    ],
                )
                print(f"[ablation] {label} bottleneck="
                      f"{entry['bottleneck']:.4f} "
                      f"best_ep={entry['best_episode']} "
                      f"({entry['wall_s']:.0f}s)", flush=True)
            except Exception as exc:  # noqa: BLE001
                entry.update(status="failed", error=str(exc),
                             traceback=traceback.format_exc())
                print(f"[ablation] {label} FAILED: {exc}", flush=True)

            records.append(entry)
            progress.write_text(json.dumps({
                "schema": "v6-ablation-progress-v1",
                "completed": len(records),
                "total": total,
                "elapsed_s": time.perf_counter() - started,
                "baseline_reference": BASELINE_REFERENCE,
                "records": records,
            }, indent=2, default=str))

    # Aggregate by weight
    summary: dict = {
        "schema": "v6-ablation-summary-v1",
        "episodes": EPISODES,
        "profiles_per_episode": cli.profiles,
        "ramp_frac": RAMP_FRAC,
        "baseline_reference": BASELINE_REFERENCE,
        "by_weight": {},
        "records": records,
    }
    for weight in WEIGHTS:
        values = [
            r["bottleneck"] for r in records
            if r["weight"] == weight and r["status"] == "complete"
        ]
        if values:
            summary["by_weight"][str(weight)] = {
                "n": len(values),
                "mean": sum(values) / len(values),
                "max": max(values),
                "min": min(values),
                "values": values,
            }
    (root / "ablation_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    print("\n[ablation] === SUMMARY ===", flush=True)
    print(f"  v5 baseline @8k: control={BASELINE_REFERENCE['control']:.4f} "
          f"mean={BASELINE_REFERENCE['mean']:.4f} "
          f"max={BASELINE_REFERENCE['max']:.4f} (n=51)")
    for weight, stats in summary["by_weight"].items():
        print(f"  weight={weight:<6} n={stats['n']} "
              f"mean={stats['mean']:.4f} max={stats['max']:.4f} "
              f"values={[f'{v:.4f}' for v in stats['values']]}")


if __name__ == "__main__":
    main()
