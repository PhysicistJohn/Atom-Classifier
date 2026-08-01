"""Run one named v6 configuration and write a compact result JSON.

Used for parallel A/B reference runs (for example baseline vs auxiliary at the
full 24k budget) without going through the Optuna study machinery.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

import v6_harness as H


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--current-corpus", required=True)
    p.add_argument("--selection-corpus", required=True)
    p.add_argument("--historical-corpus", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--seed", type=int, default=20260740)
    p.add_argument("--episodes", type=int, default=24000)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--weight", type=float, default=0.0)
    p.add_argument("--profiles", type=int, default=12)
    p.add_argument("--ramp-frac", type=float, default=0.1)
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--params-json", default=None,
        help="JSON dict of runner hyperparameters overriding CONTROL_PARAMS; "
             "worst_length_* keys inside it override the flags above",
    )
    cli = p.parse_args()

    params = dict(H.CONTROL_PARAMS)
    weight, profiles, ramp = cli.weight, cli.profiles, cli.ramp_frac
    if cli.params_json:
        supplied = json.loads(cli.params_json)
        weight = float(supplied.pop("worst_length_weight", weight))
        profiles = int(supplied.pop("worst_length_profiles", profiles))
        ramp = float(supplied.pop("worst_length_ramp_frac", ramp))
        params.update(supplied)

    root = Path(cli.output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    ctx = H.StudyContext(
        current_corpus=cli.current_corpus,
        selection_corpus=cli.selection_corpus,
        historical_corpus=cli.historical_corpus,
        study_root=root,
    )
    out = root / f"{cli.label}.json"
    t0 = time.perf_counter()
    try:
        result = H.run_trial(
            dict(params, device=cli.device),
            ctx=ctx,
            output_dir=root / cli.label,
            seed=cli.seed,
            episodes=cli.episodes,
            eval_every=cli.eval_every,
            worst_length_weight=weight,
            worst_length_profiles=profiles,
            worst_length_ramp_frac=ramp,
        )
        payload = {
            "label": cli.label,
            "status": "complete",
            "seed": cli.seed,
            "episodes": cli.episodes,
            "weight": weight,
            "profiles": profiles,
            "ramp_frac": ramp,
            "params": params,
            "wall_s": time.perf_counter() - t0,
            "bottleneck": H.bottleneck_of(result),
            "best_episode": result["training"]["best_episode"],
            "terms": H.terms_of(result),
            "trajectory": [
                {"episode": e["episode"], "bottleneck": e["checkpoint_score"][0]}
                for e in result["training"]["evaluation"]
            ],
            "aux": result["worst_length_auxiliary"],
        }
        print(f"[{cli.label}] bottleneck={payload['bottleneck']:.4f} "
              f"best_ep={payload['best_episode']}/{cli.episodes}", flush=True)
    except Exception as exc:  # noqa: BLE001
        payload = {
            "label": cli.label, "status": "failed", "error": str(exc),
            "traceback": traceback.format_exc(),
            "wall_s": time.perf_counter() - t0,
        }
        print(f"[{cli.label}] FAILED: {exc}", flush=True)
    out.write_text(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
