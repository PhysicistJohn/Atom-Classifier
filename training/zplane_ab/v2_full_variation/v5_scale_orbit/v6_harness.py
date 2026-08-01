"""Shared harness for v6 (worst-length) development runs and Optuna studies.

Centralises the setup that both the ablation runner and the study need:

* import-path wiring for the v3/v4/v5 module tree
* content-hash corpus identity (macOS APFS returns drifting nanosecond mtimes,
  which invalidates the v5 cache contract on every run)
* relaxed frozen-baseline hyperparameter validation, so a development study may
  search knobs the release firewall pins
* installation of the worst-length / profile-balanced auxiliary
* an ``_evaluate`` hook that streams intermediate checkpoint scores to an Optuna
  trial so unpromising runs can be pruned early

Nothing here modifies a frozen v5 source file.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib as _hashlib
from fractions import Fraction
from pathlib import Path
import sys
from typing import Any

HERE = Path(__file__).resolve().parent


def _ensure_import_path() -> None:
    for part in (str(HERE), str(HERE.parent), str(HERE.parent.parent)):
        if part not in sys.path:
            sys.path.insert(0, part)


_ensure_import_path()


def _patch_corpus_identity() -> None:
    """Use partial content hashing instead of mtime for corpus identity."""
    import invariant_patch_data as inv  # type: ignore[import-untyped]

    def _fixed(corpus_dir):
        manifest_path = Path(corpus_dir) / "corpus.json"
        raw_path = Path(corpus_dir) / "corpus.f32"
        with open(raw_path, "rb") as handle:
            header = handle.read(1_000_000)
            handle.seek(-1_000_000, 2)
            footer = handle.read(1_000_000)
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

import run_scale_orbit_dev as runner  # noqa: E402
from run_scale_orbit_dev import run  # noqa: E402
import worst_length_adaptation as worst_length  # noqa: E402

_original_validate = runner._validate_adaptation_run_configuration


def _relaxed_validate(args: argparse.Namespace) -> None:
    """Keep the architectural invariants; allow the hyperparameter sweep."""
    runner._validate_supervised_pair_weight(args.supervised_pair_weight)
    if runner._source_share_fraction(args.current_source_share) != Fraction(1, 3):
        raise ValueError("current_source_share must be 1/3")
    if getattr(args, "encoder", None) not in {"real", "complex"}:
        raise ValueError("encoder must be real or complex")


runner._validate_adaptation_run_configuration = _relaxed_validate


# Frozen architecture / data-pipeline decisions.  patch_length, patch_count and
# target_frac stay fixed because the historical corpus carries prebuilt
# invariant-patch caches keyed by those three values.
FIXED_PARAMS: dict[str, Any] = {
    "encoder": "real",
    "set_pool": "mean_std",
    "phase_augmentation": True,
    "current_source_share": Fraction(1, 3),
    "supervised_pair_weight": 0.2,
    "patch_dim": 64,
    "patch_length": 64,
    "patch_count": 16,
    "target_frac": 0.5,
    "device": "auto",
}

# v5 frozen defaults -- the control arm.
CONTROL_PARAMS: dict[str, Any] = {
    "lr": 1e-3,
    "weight_decay": 5e-4,
    "dropout": 0.35,
    "hidden": 96,
    "k_shot": 5,
    "q_query": 5,
    "warmup_frac": 0.03,
    "label_smoothing": 0.0,
}


class StudyContext:
    """Corpus paths and output root shared by every run."""

    def __init__(
        self,
        *,
        current_corpus: str,
        selection_corpus: str,
        historical_corpus: str | None,
        study_root: Path,
    ) -> None:
        from run_scale_orbit_dev import HISTORICAL_CORPUS  # noqa: PLC2701

        self.current_corpus = current_corpus
        self.selection_corpus = selection_corpus
        self.historical_corpus = (
            historical_corpus
            if historical_corpus is not None
            else str(HISTORICAL_CORPUS)
        )
        self.study_root = study_root


def build_namespace(
    params: dict[str, Any],
    *,
    ctx: StudyContext,
    output_dir: Path,
) -> argparse.Namespace:
    """Flat parameter dict -> the Namespace the v5 runner expects."""
    args: dict[str, Any] = {
        "current_corpus": ctx.current_corpus,
        "current_selection_corpus": ctx.selection_corpus,
        "historical_corpus": ctx.historical_corpus,
        "output_dir": str(output_dir),
    }
    args.update(FIXED_PARAMS)
    for key, value in params.items():
        if key in {"worst_length_weight", "worst_length_profiles"}:
            continue  # auxiliary knobs, not runner arguments
        args[key] = value
    return argparse.Namespace(**args)


@contextlib.contextmanager
def eval_reporting(trial: Any, *, eval_every: int, episodes: int):
    """Stream intermediate checkpoint scores to an Optuna trial.

    ``_evaluate`` is called once per evaluation point during training and once
    more after the best state is reloaded.  Only the in-training calls are
    reported; the trailing call is ignored so the final metric never triggers a
    prune decision after the work is already paid for.
    """
    import optuna

    original = runner._evaluate
    expected = max(1, episodes // max(1, eval_every))
    state = {"calls": 0}

    def _hooked(net, data, device):
        reports, prototypes = original(net, data, device)
        state["calls"] += 1
        if trial is not None and state["calls"] <= expected:
            step = state["calls"] * eval_every
            bottleneck = float(reports["checkpoint_score"][0])
            trial.report(bottleneck, step)
            if trial.should_prune():
                raise optuna.TrialPruned(
                    f"pruned at episode {step} with bottleneck {bottleneck:.4f}"
                )
        return reports, prototypes

    runner._evaluate = _hooked
    try:
        yield state
    finally:
        runner._evaluate = original


def run_trial(
    params: dict[str, Any],
    *,
    ctx: StudyContext,
    output_dir: Path,
    seed: int,
    episodes: int,
    eval_every: int,
    worst_length_weight: float,
    worst_length_profiles: int,
    worst_length_ramp_frac: float = 0.1,
    trial: Any = None,
) -> dict[str, Any]:
    """Execute one training run with the worst-length auxiliary installed."""
    full = dict(params)
    full["seed"] = int(seed)
    full["episodes"] = int(episodes)
    full["eval_every"] = int(eval_every)

    worst_length.configure(
        weight=worst_length_weight,
        profiles_per_episode=worst_length_profiles,
        seed=int(seed),
        ramp_episodes=int(round(episodes * float(worst_length_ramp_frac))),
    )
    if worst_length_weight > 0.0:
        worst_length.install()
    else:
        worst_length.uninstall()

    args = build_namespace(full, ctx=ctx, output_dir=output_dir)
    try:
        with eval_reporting(trial, eval_every=eval_every, episodes=episodes):
            result = run(args)
    finally:
        worst_length.uninstall()

    result["worst_length_auxiliary"] = worst_length.metadata()
    return result


def bottleneck_of(result: dict[str, Any]) -> float:
    """Return the binding checkpoint term (score[0]) of a completed run."""
    best = result.get("training", {}).get("best_checkpoint_score")
    if best is None:
        raise ValueError("run() returned no best_checkpoint_score")
    return float(best[0])


def terms_of(result: dict[str, Any]) -> dict[str, float]:
    """Return every scored term, for diagnosing which one binds."""
    final = result.get("final_evaluation", {})
    current = final.get("current", {})
    historical = final.get("historical", {})
    return {
        "historical_worst_length": float(
            historical.get("worst_length_balanced_accuracy", float("nan"))
        ),
        "current_worst_scale_present_class": float(
            current.get("worst_scale_present_class_balanced_accuracy", float("nan"))
        ),
        "current_worst_profile_scale_recall": float(
            current.get("worst_profile_scale_recall", float("nan"))
        ),
        "current_physical_scale_agreement": float(
            current.get("physical_scale_prediction_agreement_rate", float("nan"))
        ),
        "current_wifi_hr_dsss_worst_scale": float(
            current.get("wifi_hr_dsss_worst_scale_accuracy", float("nan"))
        ),
        "one_minus_directional_confusion": 1.0
        - float(current.get("worst_directional_dsss_bluetooth_confusion_rate", 0.0)),
        "current_pooled_accuracy": float(
            current.get("pooled", {}).get("accuracy", float("nan"))
        ),
    }
