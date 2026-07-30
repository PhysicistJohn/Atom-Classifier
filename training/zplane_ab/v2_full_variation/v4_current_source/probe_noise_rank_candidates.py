"""Development-only probe for pooled-rank and fourth-cumulant noise scores.

This probe may use only already-consumed design seeds 20263001/20263004.  It
never reads the held scale corpus or reserved validation seeds.  Every
candidate threshold is refit from the frozen policy's enrollment calibration
rows; selection and consumed novelty are score-only diagnostics.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
V2 = HERE.parent
ZPLANE = V2.parent
TRAINING = ZPLANE.parent
for path in (TRAINING, ZPLANE, V2, V2 / "v3_scale", HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import calibrate_current_source_openset as openset  # noqa: E402


PROBE_SCHEMA = "atomos.v4.development.noise-rank-candidate-probe"
PROBE_SCHEMA_VERSION = 1
ALLOWED_CONSUMED_SEEDS = (20_263_001, 20_263_004)
CANDIDATES = (
    "baseline",
    "pooled_stage_one_max",
    "c40_max",
    "c42_max",
    "cumulant_joint_max",
    "pooled_stage_one_c42_max",
    "pooled_stage_one_cumulant_joint_max",
    "pooled_stage_one_c42_joint_max",
)


def normalized_fourth_cumulants(
    rows: Sequence[np.ndarray],
) -> dict[str, np.ndarray]:
    """Return scale/phase-invariant centered fourth-cumulant magnitudes."""
    c40_values: list[float] = []
    c42_values: list[float] = []
    for raw in rows:
        value = np.asarray(raw, dtype=np.complex128)
        if (
            value.ndim != 1
            or len(value) < 4
            or not np.isfinite(value.real).all()
            or not np.isfinite(value.imag).all()
        ):
            raise ValueError("fourth-cumulant input is invalid")
        centered = value - np.mean(value)
        power = float(np.mean(np.abs(centered) ** 2))
        if not math.isfinite(power) or power <= 0.0:
            c40_values.append(0.0)
            c42_values.append(0.0)
            continue
        moment20 = complex(np.mean(centered ** 2))
        cumulant40 = complex(
            np.mean(centered ** 4) - 3.0 * moment20 * moment20
        )
        cumulant42 = float(
            np.mean(np.abs(centered) ** 4)
            - abs(moment20) ** 2
            - 2.0 * power * power
        )
        denominator = power * power
        c40_values.append(float(abs(cumulant40) / denominator))
        c42_values.append(float(abs(cumulant42) / denominator))
    c40 = np.asarray(c40_values, dtype=np.float64)
    c42 = np.asarray(c42_values, dtype=np.float64)
    if (
        not np.isfinite(c40).all()
        or not np.isfinite(c42).all()
        or np.any(c40 < 0.0)
        or np.any(c42 < 0.0)
    ):
        raise RuntimeError("fourth-cumulant score is invalid")
    return {
        # Noise is near zero cumulant, so a larger *negative* value is more
        # noise-like for the existing upper-tail empirical-rank convention.
        "c40": -c40,
        "c42": -c42,
        "joint": -np.sqrt(c40 * c40 + c42 * c42),
    }


def _class_rank(
    values: np.ndarray,
    prediction: np.ndarray,
    row: Mapping[str, Any],
) -> np.ndarray:
    return openset._class_rank(
        values,
        prediction,
        row["score_rank_calibration_by_predicted_class_sorted"],
        row["pooled_score_rank_calibration_sorted"],
    )


def _fit_thresholds(
    composite: np.ndarray,
    prediction: np.ndarray,
    groups: np.ndarray,
    *,
    budget: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    pooled, pooled_detail = openset.grouped_budget_threshold(
        composite,
        groups,
        pooled_budget=budget,
        group_budget=budget,
    )
    values: list[float] = []
    details: list[Mapping[str, Any]] = []
    for class_index in range(len(openset.PUBLIC_CLASSES)):
        mask = prediction == class_index
        if not np.any(mask):
            values.append(pooled)
            details.append(
                {"fallback": "pooled_composite_threshold", "rows": 0}
            )
            continue
        threshold, detail = openset.grouped_budget_threshold(
            composite[mask],
            groups[mask],
            pooled_budget=budget,
            group_budget=budget,
        )
        values.append(threshold)
        details.append(detail)
    thresholds = np.asarray(values, dtype=np.float64)
    rejected = composite >= thresholds[prediction]
    return thresholds, {
        "pooled": pooled,
        "pooled_detail": pooled_detail,
        "by_class": values,
        "detail_by_class": details,
        "rejected_rows": int(np.sum(rejected)),
        "rejection_rate": float(np.mean(rejected)),
    }


def _candidate_composites(
    *,
    baseline: np.ndarray,
    pooled_stage_one: np.ndarray,
    c40_rank: np.ndarray,
    c42_rank: np.ndarray,
    joint_rank: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "baseline": baseline,
        "pooled_stage_one_max": np.maximum(
            baseline, pooled_stage_one
        ),
        "c40_max": np.maximum(baseline, c40_rank),
        "c42_max": np.maximum(baseline, c42_rank),
        "cumulant_joint_max": np.maximum(baseline, joint_rank),
        "pooled_stage_one_c42_max": np.maximum.reduce(
            (baseline, pooled_stage_one, c42_rank)
        ),
        "pooled_stage_one_cumulant_joint_max": np.maximum.reduce(
            (baseline, pooled_stage_one, joint_rank)
        ),
        "pooled_stage_one_c42_joint_max": np.maximum.reduce(
            (baseline, pooled_stage_one, c42_rank, joint_rank)
        ),
    }


def _fit_candidate_contracts(
    context: openset.DevelopmentContext,
    policy: Mapping[str, Any],
) -> dict[tuple[str, int], dict[str, Any]]:
    enrollment_raw = openset._combined_enrollment_raw(
        context.historical, context.current
    )
    enrollment_records = openset._combined_enrollment_records(
        context.historical, context.current
    )
    contracts: dict[tuple[str, int], dict[str, Any]] = {}
    for length in openset.RUNTIME_INPUT_LENGTHS:
        records = enrollment_records[length]
        route = np.asarray(
            [str(row.source) for row in records], dtype=object
        )
        cumulants = normalized_fourth_cumulants(enrollment_raw[length])
        for source in openset.PROTOTYPE_SOURCE_ROUTES:
            source_mask = route == source
            route_policy = policy["per_prototype_source"][source][
                "per_length"
            ][str(length)]
            calibration = route_policy["composition"][
                "threshold_calibration_rows"
            ]
            prediction = np.asarray(
                calibration["predicted_public_class_index"],
                dtype=np.int64,
            )
            groups = np.asarray(
                calibration["source_profile_group"], dtype=object
            )
            expected_groups = np.asarray(
                [
                    f"{row.source}:{row.profile_id}"
                    for row in np.asarray(records, dtype=object)[source_mask]
                ],
                dtype=object,
            )
            if not np.array_equal(groups, expected_groups):
                raise RuntimeError(
                    f"{source} N{length} enrollment order changed"
                )
            stage_one_score = np.asarray(
                calibration["stage_one_score"], dtype=np.float64
            )
            phase_score = np.asarray(
                calibration["instantaneous_phase_linearity_score"],
                dtype=np.float64,
            )
            distance = np.asarray(
                calibration[
                    "leave_one_base_identity_out_winning_distance"
                ],
                dtype=np.float64,
            )
            class_stage_one = _class_rank(
                stage_one_score, prediction, route_policy["stage_one"]
            )
            pooled_stage_one = openset.empirical_rank(
                stage_one_score,
                np.asarray(
                    route_policy["stage_one"][
                        "pooled_score_rank_calibration_sorted"
                    ],
                    dtype=np.float64,
                ),
            )
            distance_rank = openset._class_rank(
                distance,
                prediction,
                route_policy["stage_two"][
                    "predicted_class_distance_calibration_sorted"
                ],
                route_policy["stage_two"][
                    "pooled_distance_calibration_sorted"
                ],
            )
            phase_rank = _class_rank(
                phase_score, prediction, route_policy["stage_chirp"]
            )
            selected_cumulants = {
                name: values[source_mask]
                for name, values in cumulants.items()
            }
            cumulant_calibration = {
                name: np.sort(values)
                for name, values in selected_cumulants.items()
            }
            cumulant_ranks = {
                name: openset.empirical_rank(
                    selected_cumulants[name],
                    cumulant_calibration[name],
                )
                for name in selected_cumulants
            }
            baseline = np.maximum.reduce(
                (class_stage_one, distance_rank, phase_rank)
            )
            composites = _candidate_composites(
                baseline=baseline,
                pooled_stage_one=pooled_stage_one,
                c40_rank=cumulant_ranks["c40"],
                c42_rank=cumulant_ranks["c42"],
                joint_rank=cumulant_ranks["joint"],
            )
            budget = float(
                route_policy["composition"]["known_false_positive_budget"]
            )
            thresholds = {}
            threshold_audit = {}
            for name, composite in composites.items():
                values, audit = _fit_thresholds(
                    composite, prediction, groups, budget=budget
                )
                thresholds[name] = values
                threshold_audit[name] = audit
            baseline_threshold = np.asarray(
                route_policy["composition"][
                    "threshold_by_predicted_public_class"
                ],
                dtype=np.float64,
            )
            if not np.array_equal(
                thresholds["baseline"], baseline_threshold
            ):
                raise RuntimeError(
                    f"{source} N{length} baseline threshold did not reproduce"
                )
            contracts[(source, length)] = {
                "route_policy": route_policy,
                "cumulant_calibration": cumulant_calibration,
                "thresholds": thresholds,
                "threshold_audit": threshold_audit,
            }
    return contracts


def _runtime_composites(
    scores: openset.PopulationScores,
    raw: Sequence[np.ndarray],
    contract: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    route_policy = contract["route_policy"]
    pooled_stage_one = openset.empirical_rank(
        scores.stage_one_score,
        np.asarray(
            route_policy["stage_one"][
                "pooled_score_rank_calibration_sorted"
            ],
            dtype=np.float64,
        ),
    )
    cumulants = normalized_fourth_cumulants(raw)
    cumulant_ranks = {
        name: openset.empirical_rank(
            values,
            np.asarray(
                contract["cumulant_calibration"][name], dtype=np.float64
            ),
        )
        for name, values in cumulants.items()
    }
    return _candidate_composites(
        baseline=scores.composite_score,
        pooled_stage_one=pooled_stage_one,
        c40_rank=cumulant_ranks["c40"],
        c42_rank=cumulant_ranks["c42"],
        joint_rank=cumulant_ranks["joint"],
    )


def _decisions(
    scores: openset.PopulationScores,
    composites: Mapping[str, np.ndarray],
    contract: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    return {
        name: composite
        >= np.asarray(contract["thresholds"][name], dtype=np.float64)[
            scores.prediction
        ]
        for name, composite in composites.items()
    }


def _selection_metrics(
    context: openset.DevelopmentContext,
    policy: Mapping[str, Any],
    contracts: Mapping[tuple[str, int], Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[tuple[str, int, str], np.ndarray]]:
    cell_rows: dict[str, dict[str, list[dict[str, Any]]]] = {
        candidate: {"class": [], "profile": []}
        for candidate in CANDIDATES
    }
    pooled: dict[str, dict[str, list[np.ndarray]]] = {
        candidate: {
            "rejected": [],
            "closed_correct": [],
        }
        for candidate in CANDIDATES
    }
    per_length: dict[str, dict[str, list[np.ndarray]]] = {
        candidate: defaultdict(list) for candidate in CANDIDATES
    }
    known_scores: dict[tuple[str, int, str], np.ndarray] = {}
    for source, corpus in (
        ("historical", context.historical),
        ("current", context.current),
    ):
        raw_by_length = openset._raw_role_views(corpus, "selection")
        records_by_length = openset._role_records(corpus, "selection")
        prepared = context.data[f"{source}_selection"]["by_length"]
        for length in openset.RUNTIME_INPUT_LENGTHS:
            raw = raw_by_length[length]
            records = records_by_length[length]
            bucket = prepared[length]
            scores = openset.score_prepared_population(
                context,
                policy,
                raw,
                bucket["x"],
                bucket["f"],
                length=length,
                prototype_source=source,
            )
            contract = contracts[(source, length)]
            composites = _runtime_composites(scores, raw, contract)
            decisions = _decisions(scores, composites, contract)
            truth = np.asarray(bucket["y"], dtype=np.int64)
            closed_correct = scores.prediction == truth
            groups = np.asarray(
                [
                    f"{row.source}:{row.profile_id}"
                    for row in records
                ],
                dtype=object,
            )
            for candidate in CANDIDATES:
                rejected = decisions[candidate]
                if candidate == "baseline" and not np.array_equal(
                    rejected, scores.rejected
                ):
                    raise RuntimeError(
                        f"{source} N{length} baseline decision did not reproduce"
                    )
                pooled[candidate]["rejected"].append(rejected)
                pooled[candidate]["closed_correct"].append(closed_correct)
                per_length[candidate][str(length)].append(
                    rejected[closed_correct]
                )
                known_scores[
                    (source, length, candidate)
                ] = composites[candidate]
                for class_index, class_name in enumerate(
                    openset.PUBLIC_CLASSES
                ):
                    mask = truth == class_index
                    if not np.any(mask):
                        continue
                    correct_mask = mask & closed_correct
                    cell_rows[candidate]["class"].append(
                        {
                            "source": source,
                            "length": length,
                            "name": class_name,
                            "rows": int(np.sum(mask)),
                            "closed_correct": int(np.sum(correct_mask)),
                            "rejected_closed_correct": int(
                                np.sum(rejected[correct_mask])
                            ),
                        }
                    )
                for group in sorted(set(str(value) for value in groups)):
                    mask = groups == group
                    correct_mask = mask & closed_correct
                    cell_rows[candidate]["profile"].append(
                        {
                            "source": source,
                            "length": length,
                            "name": group,
                            "rows": int(np.sum(mask)),
                            "closed_correct": int(np.sum(correct_mask)),
                            "rejected_closed_correct": int(
                                np.sum(rejected[correct_mask])
                            ),
                        }
                    )
    report = {}
    for candidate in CANDIDATES:
        rejected = np.concatenate(pooled[candidate]["rejected"])
        closed_correct = np.concatenate(
            pooled[candidate]["closed_correct"]
        )
        supported_cells = {}
        for section in ("class", "profile"):
            eligible = [
                row
                for row in cell_rows[candidate][section]
                if row["closed_correct"]
                >= openset.KNOWN_CELL_MIN_CLOSED_CORRECT_ROWS
            ]
            worst = max(
                eligible,
                key=lambda row: (
                    row["rejected_closed_correct"] / row["closed_correct"]
                ),
            )
            supported_cells[section] = {
                **worst,
                "damage": float(
                    worst["rejected_closed_correct"]
                    / worst["closed_correct"]
                ),
                "eligible_cells": len(eligible),
            }
        length_damage = {
            length: float(
                np.mean(np.concatenate(values))
            )
            for length, values in per_length[candidate].items()
        }
        wifi_cells = [
            row
            for row in cell_rows[candidate]["profile"]
            if row["name"] == openset.WIFI_HR_DSSS_PROFILE
        ]
        report[candidate] = {
            "rows": int(len(rejected)),
            "unconditional_rejection_rate": float(np.mean(rejected)),
            "closed_correct_rows": int(np.sum(closed_correct)),
            "pooled_damage_given_closed_correct": float(
                np.mean(rejected[closed_correct])
            ),
            "per_length_damage_given_closed_correct": length_damage,
            "worst_length_damage_given_closed_correct":
                max(length_damage.values()),
            "worst_supported_class_cell": supported_cells["class"],
            "worst_supported_profile_cell": supported_cells["profile"],
            "wifi_hr_dsss": [
                {
                    **row,
                    "damage": float(
                        row["rejected_closed_correct"]
                        / row["closed_correct"]
                    ),
                }
                for row in wifi_cells
            ],
        }
    return report, known_scores


def _noise_metrics(
    context: openset.DevelopmentContext,
    policy: Mapping[str, Any],
    contracts: Mapping[tuple[str, int], Mapping[str, Any]],
    known_scores: Mapping[tuple[str, int, str], np.ndarray],
    *,
    seeds: Sequence[int],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for seed in seeds:
        generated, provenance = openset.generate_novelty_raw(
            seed,
            n_each=openset.DEFAULT_NOVELTY_ROWS,
            lengths=openset.RUNTIME_INPUT_LENGTHS,
        )
        seed_report: dict[str, Any] = {
            "noise_provenance": provenance["noise"],
            "routes": {},
        }
        for source in openset.PROTOTYPE_SOURCE_ROUTES:
            route_report: dict[str, Any] = {}
            for length in openset.RUNTIME_INPUT_LENGTHS:
                raw = generated["noise"][length]
                scores = openset.score_raw_population(
                    context,
                    policy,
                    raw,
                    length=length,
                    prototype_source=source,
                )
                contract = contracts[(source, length)]
                composites = _runtime_composites(scores, raw, contract)
                decisions = _decisions(scores, composites, contract)
                route_report[str(length)] = {
                    candidate: {
                        "unknown_recall": float(
                            np.mean(decisions[candidate])
                        ),
                        "auroc_vs_route_known": float(
                            openset.auroc(
                                composites[candidate],
                                known_scores[
                                    (source, length, candidate)
                                ],
                            )
                        ),
                    }
                    for candidate in CANDIDATES
                }
            seed_report["routes"][source] = route_report
        report[str(seed)] = seed_report
    for candidate in CANDIDATES:
        cells = [
            row[candidate]
            for seed in seeds
            for seed_report in (report[str(seed)],)
            for route in seed_report["routes"].values()
            for row in route.values()
        ]
        report.setdefault("summary", {})[candidate] = {
            "worst_unknown_recall": min(
                row["unknown_recall"] for row in cells
            ),
            "worst_auroc": min(
                row["auroc_vs_route_known"] for row in cells
            ),
        }
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    seeds = tuple(int(seed) for seed in args.seeds)
    if seeds != ALLOWED_CONSUMED_SEEDS:
        raise ValueError(
            "this development probe is frozen to consumed seeds "
            f"{ALLOWED_CONSUMED_SEEDS}"
        )
    output = openset.validate_empty_output(args.output_dir)
    policy_path = openset.reject_sensitive_path(
        args.policy, "development policy input"
    )
    policy = openset.validate_policy(openset._load_json(policy_path))
    context = openset.load_development_context(
        args.fusion,
        args.historical_corpus,
        args.current_corpus,
        device_name=args.device,
    )
    openset._validate_policy_corpus_binding(policy, context)
    contracts = _fit_candidate_contracts(context, policy)
    selection, known_scores = _selection_metrics(
        context, policy, contracts
    )
    noise = _noise_metrics(
        context,
        policy,
        contracts,
        known_scores,
        seeds=seeds,
    )
    payload = {
        "schema": PROBE_SCHEMA,
        "schema_version": PROBE_SCHEMA_VERSION,
        "development_only": True,
        "release_evidence": False,
        "reads_held_or_reserved_validation_data": False,
        "consumed_design_seeds": list(seeds),
        "policy_sha256": openset._sha256(policy_path),
        "candidate_contract": {
            "candidates": list(CANDIDATES),
            "threshold_population": "enrollment only",
            "threshold_budget": 0.03,
            "threshold_group_safety": (
                "pooled and every source/profile group"
            ),
            "fourth_cumulants": {
                "centering": "subtract complex sample mean",
                "c40": "|E[z^4] - 3 E[z^2]^2| / E[|z|^2]^2",
                "c42": (
                    "|E[|z|^4] - |E[z^2]|^2 - "
                    "2 E[|z|^2]^2| / E[|z|^2]^2"
                ),
                "joint": "sqrt(c40^2 + c42^2)",
                "rank_direction": (
                    "negative magnitude; upper tail is more Gaussian/noise-like"
                ),
                "uses_frequency_transform": False,
                "global_scale_invariant": True,
                "global_phase_invariant": True,
                "per_runtime_length_calibration": True,
            },
        },
        "threshold_fit": {
            f"{source}/N{length}": contract["threshold_audit"]
            for (source, length), contract in contracts.items()
        },
        "selection": selection,
        "noise": noise,
    }
    output.mkdir(parents=True, exist_ok=True)
    openset._write_json(output / "noise-rank-candidate-probe.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fusion", required=True)
    parser.add_argument("--historical-corpus", required=True)
    parser.add_argument("--current-corpus", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(ALLOWED_CONSUMED_SEEDS),
    )
    return parser


def main() -> None:
    payload = run(build_parser().parse_args())
    for candidate, row in payload["noise"]["summary"].items():
        print(
            f"{candidate}: recall={row['worst_unknown_recall']:.6f} "
            f"auroc={row['worst_auroc']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
