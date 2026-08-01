"""Broad Optuna study for the v6 worst-length / profile-balanced classifier.

Experimental design
-------------------
The v5 study tuned 8 knobs at a fixed 8,000-episode budget and topped out at a
0.7434 bottleneck.  Post-hoc analysis found three independent causes, all of
which this study addresses and, critically, *attributes*:

1. Budget.  39 of 51 v5 runs selected the final episode as best and 44 were
   still improving on the last step -- the runs were cut off mid-climb.
2. Objective mismatch.  The scored term is the *worst* runtime prefix, but the
   episodic loss draws one prefix uniformly, optimising the *mean*.
3. Sampling.  Episodes are class-balanced while historical profile
   multiplicity is 1..24 per class, so rare modes are badly under-exposed.

Stages
------
* ``control_8k``   -- v5 defaults, v5 budget, auxiliary off.  Reproduces the
  known 0.6814 reference and confirms the harness is faithful.
* ``control_24k``  -- v5 defaults, 3x budget, auxiliary off.  Isolates the
  effect of budget alone.
* ``control_24k_aux`` -- v5 defaults, 3x budget, auxiliary on.  Isolates the
  effect of the new loss on top of budget.
* ``search``       -- broad 11-parameter TPE search at the fixed 24k budget,
  median-pruned, bounded by wall clock rather than trial count.
* ``confirm``      -- top configurations re-run on held-out seeds.

Why the budget is fixed at 24,000 rather than searched
------------------------------------------------------
Pruning compares intermediate values at equal steps.  The learning rate follows
a cosine schedule spanning the whole run, so a trial with a longer budget is
systematically *behind* a short-budget trial at any shared step purely because
it has annealed less.  Searching ``episodes`` while pruning would therefore
penalise long schedules for a reason unrelated to their final quality.  Fixing
the budget makes pruning valid; the ``confirm`` stage then re-runs the winner at
larger budgets to test for remaining headroom.

Why the study seed is fixed
---------------------------
Selecting the maximum over many trials of a noisy metric is upward-biased --
the v5 study's own spread at fixed budget was 0.4531..0.7434.  Holding the seed
constant makes trials comparable to each other; the ``confirm`` stage then
re-evaluates the top configurations on seeds never used during the search, so
the reported final number is not a seed lottery win.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback
from typing import Any

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

import v6_harness as H

STUDY_EPISODES = 24_000
STUDY_EVAL_EVERY = 1_000
STUDY_SEED = 20260740
CONFIRM_SEEDS = (20260901, 20260902, 20260903)

DEFAULT_STUDY_NAME = "v6_worst_length_broad"

# Broad ranges.  Every v5 range that produced a boundary-pinned optimum has
# been widened well past where the v5 top-10 clustered:
#   lr            v5 1e-5..1e-2   -> top trials at 4e-3..9e-3
#   warmup_frac   v5 0.01..0.10   -> all top-10 in the top quartile (pinned)
#   k_shot        v5 2..10        -> top trials at 9..10 (pinned)
#   q_query       v5 2..10        -> top trials at 8..10 (pinned)
SEARCH_SPACE: dict[str, Any] = {
    "lr": ("float_log", 1e-5, 3e-2),
    "weight_decay": ("float_log", 1e-7, 3e-2),
    "dropout": ("float", 0.0, 0.6),
    "hidden": ("categorical", [48, 64, 96, 128, 160, 192, 256, 320, 384]),
    "k_shot": ("int", 2, 16),
    "q_query": ("int", 2, 16),
    "warmup_frac": ("float", 0.005, 0.30),
    "label_smoothing": ("float", 0.0, 0.25),
    # Auxiliary knobs, narrowed by the 24k A/B measured before this study.
    # At seed 20260740 with v5 control hyperparameters and a 24k budget:
    #     w=0.00 -> 0.6570   (baseline)
    #     w=0.15 -> 0.7120   (+0.0550)
    #     w=0.40 -> 0.6832   (+0.0262)
    # and at 8k, w=0.40 was -0.0084 against its matched-seed baseline.  The
    # response is an inverted U peaking near 0.15, so the range is centred
    # there rather than left at (0, 1).  The lower bound stays well above zero
    # because BOTH auxiliary arms beat the baseline at the full budget; TPE can
    # still back off to a near-negligible 0.02 if a given region disagrees.
    "worst_length_weight": ("float", 0.02, 0.60),
    "worst_length_profiles": ("int", 4, 37),
    "worst_length_ramp_frac": ("float", 0.0, 0.5),
}


def _suggest(trial: optuna.trial.Trial) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, spec in SEARCH_SPACE.items():
        kind = spec[0]
        if kind == "float":
            out[name] = trial.suggest_float(name, spec[1], spec[2])
        elif kind == "float_log":
            out[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
        elif kind == "int":
            out[name] = trial.suggest_int(name, spec[1], spec[2])
        elif kind == "categorical":
            out[name] = trial.suggest_categorical(name, spec[1])
        else:  # pragma: no cover - guarded by the literal table above
            raise ValueError(f"unknown search-space kind {kind!r}")
    return out


def _split(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate runner hyperparameters from auxiliary knobs."""
    aux_keys = {
        "worst_length_weight",
        "worst_length_profiles",
        "worst_length_ramp_frac",
    }
    runner = {k: v for k, v in params.items() if k not in aux_keys}
    aux = {k: params[k] for k in aux_keys if k in params}
    return runner, aux


def _write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str))


def _run_named(
    label: str,
    params: dict[str, Any],
    *,
    ctx: H.StudyContext,
    root: Path,
    seed: int,
    episodes: int,
    eval_every: int,
    device: str = "auto",
) -> dict[str, Any]:
    """Run one un-pruned, named reference configuration."""
    runner_params, aux = _split(params)
    runner_params["device"] = device
    print(f"\n[v6] === {label} (episodes={episodes}, seed={seed}) ===",
          flush=True)
    t0 = time.perf_counter()
    result = H.run_trial(
        runner_params,
        ctx=ctx,
        output_dir=root / label,
        seed=seed,
        episodes=episodes,
        eval_every=eval_every,
        worst_length_weight=aux.get("worst_length_weight", 0.0),
        worst_length_profiles=int(aux.get("worst_length_profiles", 12)),
        worst_length_ramp_frac=aux.get("worst_length_ramp_frac", 0.1),
        trial=None,
    )
    entry = {
        "label": label,
        "seed": seed,
        "episodes": episodes,
        "params": params,
        "bottleneck": H.bottleneck_of(result),
        "best_episode": result["training"]["best_episode"],
        "terms": H.terms_of(result),
        "wall_s": time.perf_counter() - t0,
        "trajectory": [
            {"episode": e["episode"], "bottleneck": e["checkpoint_score"][0]}
            for e in result["training"]["evaluation"]
        ],
    }
    print(f"[v6] {label} bottleneck={entry['bottleneck']:.4f} "
          f"best_ep={entry['best_episode']}/{episodes} "
          f"({entry['wall_s']:.0f}s)", flush=True)
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description="v6 broad Optuna study")
    parser.add_argument("--current-corpus", required=True)
    parser.add_argument("--selection-corpus", required=True)
    parser.add_argument("--historical-corpus", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--study-name", default=DEFAULT_STUDY_NAME)
    parser.add_argument("--trials", type=int, default=400,
                        help="upper bound; the wall-clock budget usually binds")
    parser.add_argument("--search-hours", type=float, default=8.5,
                        help="wall-clock budget for the search stage")
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--role", choices=["all", "controls", "search", "confirm"],
        default="all",
        help="stage this process runs; parallel workers use 'search'",
    )
    parser.add_argument(
        "--storage", default=None,
        help="journal file for multi-process studies (SQLite is not safe "
             "under concurrent writes); defaults to sqlite:///study.db",
    )
    parser.add_argument(
        "--sampler-seed", type=int, default=None,
        help="MUST differ per parallel worker -- workers sharing a sampler "
             "seed draw identical startup trials and waste the budget",
    )
    parser.add_argument("--device", default="auto")
    cli = parser.parse_args()

    root = Path(cli.output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    ctx = H.StudyContext(
        current_corpus=cli.current_corpus,
        selection_corpus=cli.selection_corpus,
        historical_corpus=cli.historical_corpus,
        study_root=root,
    )
    state_path = root / (
        "study_state.json" if cli.role in {"all", "controls"}
        else f"worker_state_{cli.sampler_seed or 0}.json"
    )
    state: dict[str, Any] = {
        "schema": "v6-study-state-v1",
        "stage": "starting",
        "started_wall": time.time(),
        "controls": {},
        "search": {
            "episodes": STUDY_EPISODES,
            "seed": STUDY_SEED,
            "budget_hours": cli.search_hours,
        },
        "confirm": [],
    }
    if state_path.exists() and cli.resume:
        try:
            state.update(json.loads(state_path.read_text()))
        except Exception:  # noqa: BLE001
            pass
    _write(state_path, state)

    # ---------------- Stage 1: attribution controls ----------------
    if not cli.skip_controls and cli.role in {"all", "controls"}:
        state["stage"] = "controls"
        _write(state_path, state)
        control_specs = [
            ("control_8k", dict(H.CONTROL_PARAMS,
                                worst_length_weight=0.0), 8_000, 500),
            ("control_24k", dict(H.CONTROL_PARAMS,
                                 worst_length_weight=0.0),
             STUDY_EPISODES, STUDY_EVAL_EVERY),
            ("control_24k_aux", dict(H.CONTROL_PARAMS,
                                     worst_length_weight=0.15,
                                     worst_length_profiles=12,
                                     worst_length_ramp_frac=0.1),
             STUDY_EPISODES, STUDY_EVAL_EVERY),
        ]
        for label, params, episodes, eval_every in control_specs:
            if label in state["controls"]:
                continue
            try:
                state["controls"][label] = _run_named(
                    label, params, ctx=ctx, root=root, seed=STUDY_SEED,
                    episodes=episodes, eval_every=eval_every,
                    device=cli.device,
                )
            except Exception as exc:  # noqa: BLE001
                state["controls"][label] = {
                    "label": label, "status": "failed", "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                print(f"[v6] {label} FAILED: {exc}", flush=True)
            _write(state_path, state)

    # ---------------- Stage 2: broad search ----------------
    state["stage"] = "search"
    _write(state_path, state)
    if cli.storage:
        from optuna.storages import JournalStorage
        from optuna.storages.journal import JournalFileBackend

        storage: Any = JournalStorage(JournalFileBackend(cli.storage))
    else:
        storage = f"sqlite:///{root / 'study.db'}"
    study = optuna.create_study(
        study_name=cli.study_name,
        storage=storage,
        direction="maximize",
        sampler=TPESampler(
            multivariate=True,
            group=True,
            n_startup_trials=25,   # broad random coverage before TPE narrows
            # Distinct per worker.  A shared sampler seed makes every worker
            # draw the same startup trials, which silently wastes the budget
            # on duplicates.
            seed=(cli.sampler_seed
                  if cli.sampler_seed is not None else STUDY_SEED),
        ),
        # Do not prune until enough trials exist to form a meaningful median,
        # and never before the metric has had time to move -- the v5 data shows
        # this score is still rising late in training.
        pruner=MedianPruner(
            n_startup_trials=12,
            n_warmup_steps=8_000,
            interval_steps=STUDY_EVAL_EVERY * 4,
        ),
        load_if_exists=True,
    )

    def objective(trial: optuna.trial.Trial) -> float:
        params = _suggest(trial)
        runner_params, aux = _split(params)
        runner_params["device"] = cli.device
        trial_dir = root / f"trial_{trial.number:04d}"
        t0 = time.perf_counter()
        result = H.run_trial(
            runner_params,
            ctx=ctx,
            output_dir=trial_dir,
            seed=STUDY_SEED,
            episodes=STUDY_EPISODES,
            eval_every=STUDY_EVAL_EVERY,
            worst_length_weight=aux["worst_length_weight"],
            worst_length_profiles=int(aux["worst_length_profiles"]),
            worst_length_ramp_frac=aux["worst_length_ramp_frac"],
            trial=trial,
        )
        bottleneck = H.bottleneck_of(result)
        terms = H.terms_of(result)
        trial.set_user_attr("terms", terms)
        trial.set_user_attr("best_episode", result["training"]["best_episode"])
        trial.set_user_attr("wall_s", time.perf_counter() - t0)
        binding = min(
            (v, k) for k, v in terms.items()
            if k != "current_pooled_accuracy"
        )[1]
        trial.set_user_attr("binding_term", binding)
        print(f"[v6] trial {trial.number} bottleneck={bottleneck:.4f} "
              f"binding={binding} "
              f"best_ep={result['training']['best_episode']}", flush=True)
        return bottleneck

    def _progress(study_: optuna.study.Study, _trial: Any) -> None:
        finished = [t for t in study_.trials
                    if t.state == optuna.trial.TrialState.COMPLETE]
        pruned = [t for t in study_.trials
                  if t.state == optuna.trial.TrialState.PRUNED]
        state["search"].update(
            trials_total=len(study_.trials),
            trials_complete=len(finished),
            trials_pruned=len(pruned),
            best_value=(study_.best_value if finished else None),
            best_number=(study_.best_trial.number if finished else None),
            best_params=(study_.best_trial.params if finished else None),
            elapsed_s=time.time() - state["started_wall"],
        )
        _write(state_path, state)

    if cli.role in {"all", "search"}:
        study.optimize(
            objective,
            n_trials=cli.trials,
            timeout=cli.search_hours * 3600.0,
            gc_after_trial=True,
            callbacks=[_progress],
            catch=(RuntimeError, ValueError),
        )

    if cli.role == "search":
        state["stage"] = "worker_done"
        _write(state_path, state)
        print("[v6] search worker finished", flush=True)
        return

    # ---------------- Stage 3: held-out seed confirmation ----------------
    state["stage"] = "confirm"
    _write(state_path, state)
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    ranked = sorted(completed, key=lambda t: t.value, reverse=True)[:3]
    for rank, trial in enumerate(ranked):
        for seed in CONFIRM_SEEDS:
            label = f"confirm_r{rank}_t{trial.number}_s{seed}"
            try:
                entry = _run_named(
                    label, dict(trial.params), ctx=ctx, root=root, seed=seed,
                    episodes=STUDY_EPISODES, eval_every=STUDY_EVAL_EVERY,
                    device=cli.device,
                )
                entry["source_trial"] = trial.number
                entry["study_value"] = trial.value
                state["confirm"].append(entry)
            except Exception as exc:  # noqa: BLE001
                state["confirm"].append({
                    "label": label, "status": "failed", "error": str(exc),
                })
                print(f"[v6] {label} FAILED: {exc}", flush=True)
            _write(state_path, state)

    state["stage"] = "done"
    _write(state_path, state)
    print("\n[v6] === COMPLETE ===", flush=True)
    if completed:
        print(f"  best trial {study.best_trial.number}: {study.best_value:.4f}")
    for label, entry in state["controls"].items():
        if "bottleneck" in entry:
            print(f"  {label}: {entry['bottleneck']:.4f}")


if __name__ == "__main__":
    main()
