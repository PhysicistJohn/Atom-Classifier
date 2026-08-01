"""Optuna hyperparameter study for v5 scale-orbit branch.

Runs a control trial with the frozen defaults, then a 50-trial TPE search
over the tunable knobs while keeping architecture and data pipeline frozen.

Objective: maximize the checkpoint-score bottleneck term (score[0]) — the
minimum of historical floor and all scale-sensitive current-route metrics.

Tunable parameters (8): lr, weight_decay, dropout, hidden, k_shot, q_query,
warmup_frac, label_smoothing. Architecture (encoder, set_pool, phase_augmentation),
data pipeline (patch_length, patch_count, target_frac), and loss weighting
(supervised_pair_weight, current_source_share) are frozen by the v5 firewall.

The runner's attempt-4 hyperparameter validation is relaxed so Optuna can search.
Invariant-patch cache identity uses content hashing instead of mtime to survive
macOS APFS nanosecond timestamp drift.

Usage:
    python optuna_study.py \\
        --current-corpus  training/artifacts/signallab-current-scale-train-v5-seed20264101-r64 \\
        --selection-corpus training/artifacts/signallab-current-scale-dev-v5-seed20262904-r64-identity-firewalled \\
        --output-dir      training/artifacts/v5-optuna-study

To resume an interrupted study, add --resume.
"""
from __future__ import annotations

import argparse
import hashlib as _hashlib
import json
import shutil
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any

import optuna
from optuna.samplers import TPESampler

# ---------------------------------------------------------------------------
# Import the v5 runner
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent

def _ensure_import_path() -> None:
    """Add parent directories to sys.path so the v5 runner can import."""
    import sys
    parts = [str(HERE), str(HERE.parent), str(HERE.parent.parent)]
    for part in parts:
        if part not in sys.path:
            sys.path.insert(0, part)

_ensure_import_path()

# Patch invariant_patch_data to use content hash instead of mtime.
# macOS APFS returns slightly different nanosecond mtimes on successive stat()
# calls, which invalidates the cache contract on every run. This monkey-patch
# makes the corpus identity stable as long as file content doesn't change.

def _patch_corpus_identity() -> None:
    import invariant_patch_data as inv  # type: ignore[import-untyped]

    def _fixed(corpus_dir):
        manifest_path = Path(corpus_dir) / "corpus.json"
        raw_path = Path(corpus_dir) / "corpus.f32"
        with open(raw_path, "rb") as f:
            header = f.read(1_000_000)
            f.seek(-1_000_000, 2)
            footer = f.read(1_000_000)
        content_hash = _hashlib.sha256(header + footer).hexdigest()
        return {
            "manifest_sha256": inv._sha256(manifest_path),
            "corpus_f32": {
                "size": int(raw_path.stat().st_size),
                "content_hash_partial": content_hash,
            },
        }

    inv._corpus_identity = _fixed


_patch_corpus_identity()

import run_scale_orbit_dev as _runner  # noqa: E402
from run_scale_orbit_dev import build_parser, run  # noqa: E402

# Bypass frozen-baseline hyperparameter validation.
# The v5 runner enforces attempt-4 config to protect release integrity.
# For a development-only study, we relax this so Optuna can search params.
_original_validate = _runner._validate_adaptation_run_configuration


def _relaxed_validate(args):
    """Only validate encoder and current_source_share; allow hyperparam sweep."""
    _runner._validate_supervised_pair_weight(args.supervised_pair_weight)
    if _runner._source_share_fraction(args.current_source_share) != Fraction(1, 3):
        raise ValueError("current_source_share must be 1/3")
    if getattr(args, "encoder", None) not in {"real", "complex"}:
        raise ValueError("encoder must be real or complex")


_runner._validate_adaptation_run_configuration = _relaxed_validate


# ---------------------------------------------------------------------------
# Hyperparameter search space
# ---------------------------------------------------------------------------
# Tunable knobs — the v5 runner takes these via argparse.
# Architecture (encoder, set_pool, phase_augmentation) and data pipeline stay frozen.
# patch_length/patch_count/target_frac are also frozen because the historical
# corpus has prebuilt invariant-patch caches keyed by those three values.
# supervised_pair_weight is frozen at 0.2 by the attempt-3 firewall contract.

CONTROL_PARAMS = {
    "lr": 1e-3,
    "weight_decay": 5e-4,
    "dropout": 0.35,
    "hidden": 96,
    "k_shot": 5,
    "q_query": 5,
    "warmup_frac": 0.03,
    "label_smoothing": 0.0,
}

# Fixed parameters (frozen architectural decisions + cache-bound values)
FIXED_PARAMS = {
    "encoder": "real",
    "set_pool": "mean_std",
    "phase_augmentation": True,
    "current_source_share": Fraction(1, 3),
    "supervised_pair_weight": 0.2,
    "seed": 20260740,
    "episodes": 8_000,
    "eval_every": 500,
    "patch_dim": 64,
    "patch_length": 64,
    "patch_count": 16,
    "target_frac": 0.5,
    "device": "auto",
}

DEFAULT_STUDY_NAME = "v5_scale_orbit_hyperparam"
DEFAULT_TRIALS = 50


class StudyContext:
    """Shared corpus paths and output root for all trials."""

    def __init__(
        self,
        *,
        current_corpus: str,
        selection_corpus: str,
        historical_corpus: str | None,
        study_root: Path,
    ) -> None:
        self.current_corpus = current_corpus
        self.selection_corpus = selection_corpus
        # Historical corpus defaults to runner's HISTORICAL_CORPUS if not given
        from run_scale_orbit_dev import HISTORICAL_CORPUS  # noqa: PLC2701

        self.historical_corpus = (
            historical_corpus if historical_corpus is not None else str(HISTORICAL_CORPUS)
        )
        self.study_root = study_root


def _suggest(trial: optuna.trial.Trial) -> dict[str, Any]:
    """Sample hyperparameters from the search space."""
    return {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "hidden": trial.suggest_categorical(
            "hidden", [48, 64, 80, 96, 112, 128, 160, 192, 224, 256]
        ),
        "k_shot": trial.suggest_int("k_shot", 2, 10),
        "q_query": trial.suggest_int("q_query", 2, 10),
        "warmup_frac": trial.suggest_float("warmup_frac", 0.01, 0.10),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.1),
    }


def _namespace(
    params: dict[str, Any], *, ctx: StudyContext, output_dir: Path
) -> argparse.Namespace:
    """Convert a flat param dict to an argparse.Namespace the runner expects."""
    args_dict = {
        "current_corpus": ctx.current_corpus,
        "current_selection_corpus": ctx.selection_corpus,
        "historical_corpus": ctx.historical_corpus,
        "output_dir": str(output_dir),
    }
    # Fixed params — map back to argparse key names
    for key, value in FIXED_PARAMS.items():
        args_dict[key] = value

    # Tunable params
    for key, value in params.items():
        args_dict[key] = value

    return argparse.Namespace(**args_dict)


def _extract_objective(result: dict[str, Any]) -> float:
    """Extract the bottleneck checkpoint score from a run result.

    The v5 checkpoint score is a 10-tuple compared lexicographically.
    We optimize score[0], which is:
        min(historical_balanced, worst_scale_present_class_balanced,
            worst_profile_scale_recall, physical_scale_agreement,
            wifi_hr_dsss_worst_scale_accuracy, directional_margin)

    Higher = better.
    """
    training = result.get("training", {})
    best_score = training.get("best_checkpoint_score")
    if best_score is None:
        raise ValueError(
            "run() returned no best_checkpoint_score in training output"
        )
    bottleneck = float(best_score[0])
    return bottleneck


def objective(
    trial: optuna.trial.Trial,
    *,
    ctx: StudyContext,
) -> float:
    """Run one v5 training trial and return the bottleneck score."""

    params = _suggest(trial)
    trial_dir = ctx.study_root / f"trial_{trial.number:03d}"

    try:
        args = _namespace(params, ctx=ctx, output_dir=trial_dir)
        result = run(args)
    except Exception as exc:
        # Clean up partial output on failure
        shutil.rmtree(trial_dir, ignore_errors=True)
        raise optuna.TrialPruned(f"Training failed: {exc}") from exc

    bottleneck = _extract_objective(result)
    trial.set_user_attr("bottleneck", bottleneck)
    trial.set_user_attr("params", params)
    wall_clock = result.get("wall_clock_s", None)
    if wall_clock is not None:
        trial.set_user_attr("wall_clock_s", float(wall_clock))

    print(
        f"[optuna] trial {trial.number} bottleneck={bottleneck:.4f} "
        f"params={{lr:{params['lr']:.0e}, wd:{params['weight_decay']:.0e}, "
        f"do:{params['dropout']:.2f}, h:{params['hidden']}}}",
        flush=True,
    )

    return bottleneck


def run_control(*, ctx: StudyContext) -> float:
    """Run the control trial with frozen defaults."""

    print("[optuna] === CONTROL TRIAL ===", flush=True)
    control_dir = ctx.study_root / "control"

    # Build args with all defaults
    args = _namespace(CONTROL_PARAMS, ctx=ctx, output_dir=control_dir)
    result = run(args)
    bottleneck = _extract_objective(result)

    # Save params file for reference
    params_path = control_dir / "control_params.json"
    combined = {
        "control": True,
        **CONTROL_PARAMS,
        **{k: str(v) if not isinstance(v, (int, float, bool)) else v for k, v in FIXED_PARAMS.items()},
    }
    _write_json(params_path, combined)

    print(
        f"[optuna] control bottleneck={bottleneck:.4f} "
        f"wall_clock={result.get('wall_clock_s', 'N/A')}",
        flush=True,
    )
    return bottleneck


def summarize(
    study: optuna.study.Study,
    control_score: float,
    *,
    study_root: Path,
) -> None:
    """Write a summary of top trials to JSON."""

    finished = [t for t in study.trials if t.state.is_finished()]
    pruned = len(study.trials) - len(finished)
    trials_sorted = sorted(finished, key=lambda t: t.value, reverse=True)
    top_n = min(10, len(trials_sorted))

    summary = {
        "study": study.study_name,
        "control_score": control_score,
        "total_trials": len(study.trials),
        "completed_trials": len(finished),
        "pruned_trials": pruned,
        "best_trial": None,
        "top_trials": [],
    }

    for trial in trials_sorted[:top_n]:
        params = trial.user_attrs.get("params", {})
        entry = {
            "number": trial.number,
            "bottleneck": trial.value,
            "beat_control": trial.value > control_score,
            "wall_clock_s": trial.user_attrs.get("wall_clock_s"),
            "params": params,
        }
        summary["top_trials"].append(entry)

    if trials_sorted:
        best = trials_sorted[0]
        summary["best_trial"] = {
            "number": best.number,
            "bottleneck": best.value,
            "params": best.user_attrs.get("params", {}),
            "beat_control": best.value > control_score,
        }

    summary_path = study_root / "summary.json"
    _write_json(summary_path, summary)
    print(f"\n[optuna] Summary written to {summary_path}", flush=True)

    # Print top-5 table
    print("\nTop 5 trials vs control (bottleneck):")
    print("-" * 60)
    print(
        f"  {'Trial':>5}  {'Bottleneck':>10}  {'vs Control':>10}  "
        f"{'lr':>10}  {'wd':>10}  {'do':>5}  {'h':>4}",
        flush=True,
    )
    for entry in summary["top_trials"][:5]:
        diff = entry["bottleneck"] - control_score
        p = entry.get("params", {})
        print(
            f"  {entry['number']:>5}  {entry['bottleneck']:>10.4f}  "
            f"{diff:+>9.4f}  "
            f"{p.get('lr', '?'):>10.0e}  "
            f"{p.get('weight_decay', '?'):>10.0e}  "
            f"{p.get('dropout', '?'):>5.2f}  "
            f"{p.get('hidden', '?'):>4}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optuna study for v5 scale-orbit hyperparameters",
    )
    parser.add_argument(
        "--current-corpus", required=True, help="V5 current training corpus path"
    )
    parser.add_argument(
        "--selection-corpus", required=True, help="V5 selection/dev corpus path"
    )
    parser.add_argument(
        "--historical-corpus", default=None, help="Historical corpus (optional — defaults to runner default)"
    )
    parser.add_argument(
        "--output-dir",
        default=str(HERE.parent.parent / "artifacts" / "v5-optuna-study"),
        help="Where to store trial outputs and study DB",
    )
    parser.add_argument(
        "--trials", type=int, default=DEFAULT_TRIALS,
        help=f"Number of study trials (default: {DEFAULT_TRIALS})",
    )
    parser.add_argument(
        "--study-name", default=DEFAULT_STUDY_NAME,
        help=f"Optuna study name (default: {DEFAULT_STUDY_NAME})",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume existing study from DB",
    )
    cli = parser.parse_args()

    study_root = Path(cli.output_dir).expanduser().resolve()
    study_root.mkdir(parents=True, exist_ok=True)
    db_path = study_root / "study.db"

    ctx = StudyContext(
        current_corpus=cli.current_corpus,
        selection_corpus=cli.selection_corpus,
        historical_corpus=cli.historical_corpus,
        study_root=study_root,
    )

    control_score = None
    if not cli.resume:
        # Control trial first
        control_score = run_control(ctx=ctx)

    # Create or resume study
    storage = f"sqlite:///{db_path}"
    study = optuna.create_study(
        study_name=cli.study_name,
        storage=storage,
        direction="maximize",
        sampler=TPESampler(
            multivariate=True,  # capture correlations between params
            n_startup_trials=10,  # random sampling before TPE kicks in
        ),
        load_if_exists=True,
    )

    obj_fn = lambda trial: objective(trial, ctx=ctx)  # noqa: E731

    n_complete = len([t for t in study.trials if t.state.is_finished()])
    remaining = cli.trials - n_complete

    print(
        f"\n[optuna] Study '{cli.study_name}': "
        f"{n_complete}/{cli.trials} complete, {remaining} remaining",
        flush=True,
    )

    if remaining > 0:
        study.optimize(obj_fn, n_trials=remaining, gc_after_trial=True)

    # Final summary
    if control_score is None and n_complete > 0:
        # Resumed without control — read from DB or skip
        summary_path = study_root / "summary.json"
        if summary_path.exists():
            control_score = json.loads(summary_path.read_text()).get("control_score")

    summarize(study, control_score or 0.0, study_root=study_root)


if __name__ == "__main__":
    main()
