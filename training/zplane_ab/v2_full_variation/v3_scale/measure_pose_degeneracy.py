"""Does pose degeneracy actually separate noise?  Measure it before building on it.

HANDOFF 17 states a mechanism: the v3 front end estimates occupied bandwidth from
non-zero-lag autocorrelation, white noise has no coherent autocorrelation, the
bandwidth estimate is therefore degenerate, and normalising by it maps noise into
the same canonical geometry as a real emission so it stops looking anomalous.
``pose_degeneracy.py`` exposes that degeneracy as features.  This module is the
falsification test for the mechanism, run *before* anything is built on top of it.

Protocol
--------

* **Known population**: raw development rows, training split only, from
  ``invariant_patch_data.load``.  Enrollment is scored as a secondary population
  so the reader can see the numbers are not a property of one split; nothing is
  fitted on either, and the immutable selection half and the consumed test half
  are never touched.
* **Novelty**: generated fresh from ``openset_eval.NOVELTY`` at the untouched
  development seeds 20260939 and 20260940, drawn once at the longest length with
  the shorter observations taken as exact raw prefixes, exactly as
  ``fit_v3_openset.generate_novelty`` does it.  Seed validation is imported from
  ``fit_v3_openset`` so the design seed 20260938 and the unspent release seed
  20260731 are both refused structurally.
* **Statistic**: for each feature independently, the Mann-Whitney AUROC that the
  novelty value exceeds the known value, computed by the shipped ``train.auroc``.
  A single feature can separate in either direction, so the oriented value
  ``max(a, 1 - a)`` is reported alongside the raw value and the direction; a
  feature whose direction is inconsistent across cells is marked as such and its
  oriented worst is withheld.
* **Worst-of**: over 2 novelty seeds x 3 prefix lengths, matching the house
  convention for open-set evidence in HANDOFF 16 and 17.

Nothing here fits, calibrates, selects, or thresholds anything.  The feature set
is frozen in ``pose_degeneracy.py`` before this module is run and must not be
revised in response to these numbers: seeds 20260939 and 20260940 are validation
evidence, and 20260938 is spent.  If the features do not separate noise, that
refutes the mechanism and is the result.

This is development evidence only.  Release seed 20260731 is not read, not
accepted, and not reachable from here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


_V3_DIR = os.path.dirname(os.path.abspath(__file__))
_V2_DIR = os.path.dirname(_V3_DIR)
_ZPAB_DIR = os.path.dirname(_V2_DIR)
_TRAINING_DIR = os.path.dirname(_ZPAB_DIR)
for _path in (_TRAINING_DIR, _ZPAB_DIR, _V2_DIR, _V3_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import fit_v3_openset as openset  # noqa: E402
import invariant_patch_data as invariant_data  # noqa: E402
import pose_degeneracy as pdg  # noqa: E402
import run_time_domain_dev as dev  # noqa: E402
import time_domain_geometry as geometry  # noqa: E402
import time_domain_invariant_patch_preprocess as td_preprocess  # noqa: E402
from openset_eval import NOVELTY  # noqa: E402
from train import auroc  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    Path(_V2_DIR) / "artifacts" / "invariant_patch" / "v3_scale" / "pose_degeneracy_dev"
)
DEFAULT_LENGTHS = openset.DEFAULT_PREFIX_LENGTHS
DEFAULT_NOVELTY_SEEDS = openset.DEFAULT_NOVELTY_SEEDS
DEFAULT_NOVELTY_N = openset.DEFAULT_NOVELTY_N
KNOWN_SPLITS = ("train", "enroll")

#: The v3 front end does not call ``estimate_band`` with the module default
#: floor; it derives a length-dependent resolution floor first.  Measuring with
#: the same floor keeps the clip flags describing the pose the fusion embedding
#: was actually built from.
PATCH_LENGTH = td_preprocess.DEFAULT_PATCH_LENGTH
PATCH_COUNT = td_preprocess.DEFAULT_PATCH_COUNT
TARGET_FRAC = 0.5


def _frontend_min_bandwidth(length: int) -> float:
    return geometry.minimum_bandwidth_for_active_span(
        int(length), PATCH_LENGTH, TARGET_FRAC
    )


def _min_bandwidth(length: int, mode: str) -> float:
    if mode == "frontend":
        return _frontend_min_bandwidth(length)
    if mode == "fixed":
        return geometry.MIN_BANDWIDTH
    raise ValueError(f"unknown min-bandwidth mode: {mode}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


# ---------------------------------------------------------------------------
# populations
# ---------------------------------------------------------------------------


def known_features(
    split: str,
    *,
    lengths: Sequence[int],
    min_bandwidth_mode: str,
    limit: int | None = None,
    progress: bool = False,
) -> tuple[dict, dict]:
    """Pose-degeneracy features for the known development rows, by length.

    Rows whose prefix at a given length is exactly all-zero are excluded at that
    length, by the same pre-inference rule ``run_time_domain_dev._length_audit``
    already applies; the estimator refuses an all-zero capture and inventing one
    would be worse than reporting the exclusion.
    """
    if split not in KNOWN_SPLITS:
        raise ValueError(f"unknown split: {split}")
    key = {"train": "tr_idx", "enroll": "en_idx"}[split]
    source = invariant_data.load(build_if_missing=False)
    manifest = dev._load_manifest()
    raw = dev._raw_memmap(manifest)
    indices = np.asarray(source[key], dtype=np.int64)
    if limit is not None:
        indices = indices[: int(limit)]

    by_length = {int(length): [] for length in lengths}
    excluded = {int(length): 0 for length in lengths}
    for position, index in enumerate(indices):
        capture = dev._complex_row(raw, int(index))
        for length in by_length:
            window = capture[:length]
            if len(window) != length or float(np.max(np.abs(window))) == 0.0:
                excluded[length] += 1
                continue
            by_length[length].append(
                pdg.pose_degeneracy_features(
                    window,
                    min_bandwidth=_min_bandwidth(length, min_bandwidth_mode),
                )
            )
        if progress and (position + 1) % 500 == 0:
            print(f"  [known {split}] {position + 1}/{len(indices)}", flush=True)

    matrices = {
        length: (
            np.stack(rows).astype(np.float64)
            if rows
            else np.empty((0, pdg.FEATURE_COUNT), dtype=np.float64)
        )
        for length, rows in by_length.items()
    }
    audit = {
        "split": split,
        "rows_considered": int(len(indices)),
        "rows_scored": {
            str(length): int(matrix.shape[0]) for length, matrix in matrices.items()
        },
        "rows_excluded_all_zero_prefix": {
            str(length): int(count) for length, count in excluded.items()
        },
        "consumed_test_rows_used": 0,
        "selection_rows_used": 0,
    }
    return matrices, audit


def novelty_features(
    seeds: Sequence[int],
    *,
    n_each: int,
    lengths: Sequence[int],
    min_bandwidth_mode: str,
    progress: bool = False,
) -> tuple[dict, dict]:
    """Pose-degeneracy features for generated novelty, by seed/family/length."""
    seed_values = openset.validate_novelty_seeds(seeds)
    prefix_lengths = openset.validate_prefix_lengths(lengths)
    if int(n_each) <= 0:
        raise ValueError("n_each must be positive")
    longest = max(prefix_lengths)

    matrices: dict = {}
    provenance: dict = {}
    for seed in seed_values:
        rng = np.random.default_rng(int(seed))
        matrices[seed] = {}
        provenance[str(seed)] = {}
        for family in openset.NOVELTY_FAMILIES:
            generator = NOVELTY[family]
            rows = {int(length): [] for length in prefix_lengths}
            longest_digest = hashlib.sha256()
            for row in range(int(n_each)):
                raw = generator(longest, rng)
                if len(raw) != longest:
                    raise RuntimeError("novelty generator returned wrong length")
                openset._complex_digest(longest_digest, raw)
                for length in rows:
                    rows[length].append(
                        pdg.pose_degeneracy_features(
                            raw[:length],
                            min_bandwidth=_min_bandwidth(
                                length, min_bandwidth_mode
                            ),
                        )
                    )
                if progress and (row + 1) % 100 == 0:
                    print(
                        f"  [novelty {seed}/{family}] {row + 1}/{n_each}",
                        flush=True,
                    )
            matrices[seed][family] = {
                length: np.stack(values).astype(np.float64)
                for length, values in rows.items()
            }
            provenance[str(seed)][family] = {
                "n": int(n_each),
                "generated_once_at": int(longest),
                "longest_capture_sha256": longest_digest.hexdigest(),
            }
    return matrices, provenance


# ---------------------------------------------------------------------------
# separation
# ---------------------------------------------------------------------------


def feature_auroc(known: np.ndarray, novel: np.ndarray) -> np.ndarray:
    """Per-feature AUROC that the novelty value exceeds the known value."""
    known = np.asarray(known, dtype=np.float64)
    novel = np.asarray(novel, dtype=np.float64)
    if known.ndim != 2 or novel.ndim != 2 or known.shape[1] != novel.shape[1]:
        raise ValueError("known and novel matrices must share a feature axis")
    return np.asarray(
        [
            auroc(novel[:, column], known[:, column])
            for column in range(known.shape[1])
        ],
        dtype=np.float64,
    )


def oriented(value: float) -> float:
    """AUROC of the better of the two directions, in ``[0.5, 1]``."""
    return float(max(float(value), 1.0 - float(value)))


def direction(value: float) -> str:
    """Which tail of the feature the novelty sits in."""
    if float(value) > 0.5:
        return "novelty_high"
    if float(value) < 0.5:
        return "novelty_low"
    return "none"


def summarize_cells(cells: Sequence[Mapping[str, Any]]) -> dict:
    """Collapse per-cell AUROCs into the reportable worst-of summary."""
    if not cells:
        raise ValueError("no cells to summarize")
    summary: dict = {}
    for index, name in enumerate(pdg.FEATURE_NAMES):
        values = [float(cell["auroc"][index]) for cell in cells]
        directions = {direction(value) for value in values}
        consistent = len(directions - {"none"}) <= 1
        entry = {
            "feature": name,
            "auroc_min": float(min(values)),
            "auroc_max": float(max(values)),
            "auroc_mean": float(np.mean(values)),
            "oriented_worst": float(min(oriented(value) for value in values)),
            "oriented_mean": float(np.mean([oriented(value) for value in values])),
            "direction_consistent": bool(consistent),
            "direction": (
                sorted(directions - {"none"})[0]
                if consistent and directions - {"none"}
                else "mixed"
            ),
            "cells": {
                str(cell["cell"]): float(cell["auroc"][index]) for cell in cells
            },
        }
        if not consistent:
            # A feature that points one way at one seed/length and the other way
            # at another is not a usable rejector input, whatever its magnitude.
            entry["oriented_worst"] = None
        summary[name] = entry
    return summary


def ranked(summary: Mapping[str, Any]) -> list:
    def key(item):
        value = item[1]["oriented_worst"]
        return (-1.0 if value is None else -float(value), item[0])

    return [name for name, _entry in sorted(summary.items(), key=key)]


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


def _source_hashes() -> dict:
    paths = {
        "training/time_domain_geometry.py": Path(_TRAINING_DIR)
        / "time_domain_geometry.py",
        "v3_scale/pose_degeneracy.py": Path(_V3_DIR) / "pose_degeneracy.py",
        "v3_scale/measure_pose_degeneracy.py": Path(_V3_DIR)
        / "measure_pose_degeneracy.py",
        "v2_full_variation/openset_eval.py": Path(_V2_DIR) / "openset_eval.py",
    }
    return {name: _sha256(path) for name, path in paths.items()}


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    lengths = openset.validate_prefix_lengths(args.lengths)
    seeds = openset.validate_novelty_seeds(args.novelty_seeds)
    output_dir = Path(args.output_dir).resolve()
    openset.reject_sealed_path(output_dir, "pose degeneracy output")

    print(f"pose degeneracy: {pdg.POSE_DEGENERACY_VERSION}", flush=True)
    print(f"features        : {pdg.FEATURE_COUNT}", flush=True)
    print(f"lengths         : {list(lengths)}", flush=True)
    print(f"novelty seeds   : {list(seeds)}", flush=True)
    print(f"min-bandwidth   : {args.min_bandwidth_mode}", flush=True)

    known: dict = {}
    known_audit: dict = {}
    for split in args.known_splits:
        print(f"[known] {split}", flush=True)
        matrices, audit = known_features(
            split,
            lengths=lengths,
            min_bandwidth_mode=args.min_bandwidth_mode,
            limit=args.known_limit,
            progress=args.progress,
        )
        known[split] = matrices
        known_audit[split] = audit

    print("[novelty] generating", flush=True)
    novelty, novelty_provenance = novelty_features(
        seeds,
        n_each=args.novelty_n,
        lengths=lengths,
        min_bandwidth_mode=args.min_bandwidth_mode,
        progress=args.progress,
    )

    report: dict = {}
    for split in args.known_splits:
        split_report: dict = {}
        for family in openset.NOVELTY_FAMILIES:
            cells = []
            for seed in seeds:
                for length in lengths:
                    known_matrix = known[split][int(length)]
                    novel_matrix = novelty[seed][family][int(length)]
                    cells.append(
                        {
                            "cell": f"seed{seed}_N{length}",
                            "seed": int(seed),
                            "length": int(length),
                            "known_rows": int(known_matrix.shape[0]),
                            "novel_rows": int(novel_matrix.shape[0]),
                            "auroc": feature_auroc(known_matrix, novel_matrix),
                        }
                    )
            summary = summarize_cells(cells)
            split_report[family] = {
                "per_feature": summary,
                "ranking_by_oriented_worst": ranked(summary),
                "cells": [
                    {
                        key: value
                        for key, value in cell.items()
                        if key != "auroc"
                    }
                    for cell in cells
                ],
            }
        report[split] = split_report

    payload = {
        "schema": "pose-degeneracy-dev-separation-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": float(time.time() - started),
        "extractor": pdg.pose_degeneracy_metadata(),
        "estimator": geometry.estimator_metadata(),
        "protocol": {
            "known_population": "development raw rows, "
            + ", ".join(args.known_splits),
            "novelty_generator": "openset_eval.NOVELTY, drawn once at the "
            "longest length, shorter observations are exact raw prefixes",
            "statistic": "train.auroc, novelty positive against known negative",
            "worst_of": f"{len(seeds)} novelty seeds x {len(lengths)} lengths",
            "min_bandwidth_mode": args.min_bandwidth_mode,
            "patch_length": PATCH_LENGTH,
            "patch_count": PATCH_COUNT,
            "target_frac": TARGET_FRAC,
        },
        "guardrails": {
            "fit_performed": False,
            "threshold_fitted": False,
            "feature_set_frozen_before_measurement": True,
            "sealed_release_paths_read": 0,
            "consumed_test_rows_used": 0,
            "selection_rows_used": 0,
            "release_seed_20260731_used": False,
            "design_novelty_seed_excluded": openset.DESIGN_NOVELTY_SEED,
            "cannot_change_closed_label": True,
            "uses_absolute_scale": False,
            "uses_frequency_transform": False,
        },
        "known_audit": known_audit,
        "novelty_provenance": novelty_provenance,
        "novelty_n_each": int(args.novelty_n),
        "seeds": {
            "novelty_seeds": [int(seed) for seed in seeds],
            "release_seed_not_spent": openset.RELEASE_SEED_NEVER_SPENT_HERE,
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "source_sha256": _source_hashes(),
        "separation": report,
    }

    write_json(output_dir / "pose_degeneracy_separation.json", payload)
    _print_tables(payload, args.known_splits)
    print(f"\nwrote {output_dir / 'pose_degeneracy_separation.json'}", flush=True)
    return payload


def _print_tables(payload: Mapping[str, Any], splits: Sequence[str]) -> None:
    for split in splits:
        for family in openset.NOVELTY_FAMILIES:
            block = payload["separation"][split][family]
            print(f"\n=== known {split} vs {family} ===", flush=True)
            print(
                f"{'feature':44s} {'worst':>7s} {'mean':>7s} "
                f"{'min':>7s} {'max':>7s}  direction",
                flush=True,
            )
            for name in block["ranking_by_oriented_worst"]:
                entry = block["per_feature"][name]
                worst = entry["oriented_worst"]
                print(
                    f"{name:44s} "
                    f"{'  mixed' if worst is None else format(worst, '7.4f')} "
                    f"{entry['oriented_mean']:7.4f} "
                    f"{entry['auroc_min']:7.4f} {entry['auroc_max']:7.4f}  "
                    f"{entry['direction']}",
                    flush=True,
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--novelty-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_NOVELTY_SEEDS),
    )
    parser.add_argument("--novelty-n", type=int, default=DEFAULT_NOVELTY_N)
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS)
    )
    parser.add_argument(
        "--known-splits",
        nargs="+",
        choices=list(KNOWN_SPLITS),
        default=["train", "enroll"],
    )
    parser.add_argument("--known-limit", type=int, default=None)
    parser.add_argument(
        "--min-bandwidth-mode",
        choices=("frontend", "fixed"),
        default="frontend",
        help="frontend mirrors the length-dependent resolution floor the v3 "
        "preprocessor actually uses; fixed uses the estimator default",
    )
    parser.add_argument("--progress", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
