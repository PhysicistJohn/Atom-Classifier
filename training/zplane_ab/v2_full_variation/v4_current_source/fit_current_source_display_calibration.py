"""Fit display-only route logit scales from identity-LOO enrollment distances.

This step is independent of open-set design/validation. It reads no held scale
corpus, selection rows never fit parameters, and it consumes no novelty seed.
The resulting runtime JSON has no decision authority: multiplying every routed
distance by one positive scalar preserves the strict argmin exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


HERE = Path(__file__).resolve().parent
V2 = HERE.parent
ZPLANE = V2.parent
TRAINING = ZPLANE.parent
for path in (TRAINING, ZPLANE, V2, V2 / "v3_scale", HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import calibrate_current_source_openset as openset  # noqa: E402


CALIBRATION_NAME = "time-domain-profile-bank-display-calibration-v4.json"
EVIDENCE_NAME = "display-calibration-evidence.json"
CALIBRATION_SCHEMA = (
    "atomos.v4.time-domain-profile-bank.display-calibration"
)
CALIBRATION_SCHEMA_VERSION = 1
CALIBRATION_KIND = (
    "per-prototype-source-positive-distance-logit-scale-v1"
)
EVIDENCE_SCHEMA = (
    "atomos.v4.time-domain-profile-bank.display-calibration-evidence"
)
EVIDENCE_SCHEMA_VERSION = 1
REFERENCE_SCALE = {
    "current": 10.501238027734903,
    "historical": 8.655378601595453,
}
REFERENCE_SCALE_ABSOLUTE_TOLERANCE = 5e-10
ECE_BINS = 15


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _validate_rows(
    distances: np.ndarray,
    truth: np.ndarray,
    support: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(distances, dtype=np.float64)
    labels = np.asarray(truth, dtype=np.int64).reshape(-1)
    mask = np.asarray(support, dtype=bool)
    if (
        values.ndim != 2
        or values.shape != (len(labels), len(openset.PUBLIC_CLASSES))
        or not len(labels)
        or mask.shape != (len(openset.PUBLIC_CLASSES),)
        or not np.any(mask)
        or not np.isfinite(values[:, mask]).all()
        or not np.isinf(values[:, ~mask]).all()
        or np.any(labels < 0)
        or np.any(labels >= len(openset.PUBLIC_CLASSES))
        or not np.all(mask[labels])
    ):
        raise ValueError("display-calibration rows are invalid")
    return values[:, mask], labels, np.flatnonzero(mask)


def _truth_local_indices(
    labels: np.ndarray,
    supported_indices: np.ndarray,
) -> np.ndarray:
    lookup = {
        int(class_index): local
        for local, class_index in enumerate(supported_indices)
    }
    return np.asarray([lookup[int(label)] for label in labels], dtype=np.int64)


def profile_balanced_weights(groups: np.ndarray) -> np.ndarray:
    values = np.asarray(groups, dtype=object).reshape(-1)
    if (
        not len(values)
        or any(not isinstance(value, str) or not value for value in values)
    ):
        raise ValueError("profile groups must be nonempty strings")
    unique = sorted(set(str(value) for value in values))
    weights = np.zeros(len(values), dtype=np.float64)
    for group in unique:
        selected = values == group
        count = int(np.sum(selected))
        if not count:
            raise AssertionError("profile weight group disappeared")
        weights[selected] = 1.0 / (len(unique) * count)
    if abs(float(np.sum(weights)) - 1.0) > 1e-14:
        raise AssertionError("profile-balanced weights do not sum to one")
    return weights


def _normalized_weights(
    rows: int,
    sample_weight: np.ndarray | None,
) -> np.ndarray:
    if sample_weight is None:
        return np.full(rows, 1.0 / rows, dtype=np.float64)
    value = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    if (
        value.shape != (rows,)
        or not np.isfinite(value).all()
        or np.any(value <= 0.0)
        or not float(np.sum(value)) > 0.0
    ):
        raise ValueError("sample weights must be aligned positive finite values")
    return value / np.sum(value)


def _softmax_probabilities(
    supported_distances: np.ndarray,
    scale: float,
) -> np.ndarray:
    value = float(scale)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            "distance-logit scale must be nonnegative finite during fitting"
        )
    logits = -value * np.asarray(supported_distances, dtype=np.float64)
    logits -= np.max(logits, axis=1, keepdims=True)
    exponential = np.exp(logits)
    return exponential / np.sum(exponential, axis=1, keepdims=True)


def categorical_nll(
    distances: np.ndarray,
    truth: np.ndarray,
    support: np.ndarray,
    scale: float,
    sample_weight: np.ndarray | None = None,
) -> float:
    values, labels, supported = _validate_rows(distances, truth, support)
    local_truth = _truth_local_indices(labels, supported)
    probabilities = _softmax_probabilities(values, scale)
    selected = probabilities[np.arange(len(labels)), local_truth]
    weights = _normalized_weights(len(labels), sample_weight)
    return float(-np.dot(weights, np.log(selected)))


def _nll_derivative(
    supported_distances: np.ndarray,
    local_truth: np.ndarray,
    scale: float,
    sample_weight: np.ndarray,
) -> float:
    probabilities = _softmax_probabilities(supported_distances, scale)
    true_distance = supported_distances[
        np.arange(len(local_truth)), local_truth
    ]
    expected_distance = np.sum(
        probabilities * supported_distances, axis=1
    )
    return float(np.dot(sample_weight, true_distance - expected_distance))


def fit_positive_distance_logit_scale(
    distances: np.ndarray,
    truth: np.ndarray,
    support: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> tuple[float, dict[str, Any]]:
    """Deterministic convex-NLL derivative root by bracketed bisection."""
    values, labels, supported = _validate_rows(distances, truth, support)
    local_truth = _truth_local_indices(labels, supported)
    weights = _normalized_weights(len(labels), sample_weight)
    lower = 0.0
    derivative_lower = _nll_derivative(
        values, local_truth, lower, weights
    )
    if derivative_lower >= 0.0:
        raise ValueError("positive temperature scale has no interior optimum")
    upper = 1.0
    derivative_upper = _nll_derivative(
        values, local_truth, upper, weights
    )
    doublings = 0
    while derivative_upper < 0.0 and doublings < 60:
        upper *= 2.0
        derivative_upper = _nll_derivative(
            values, local_truth, upper, weights
        )
        doublings += 1
    if derivative_upper < 0.0 or not math.isfinite(derivative_upper):
        raise RuntimeError("could not bracket display-calibration NLL optimum")
    iterations = 200
    for _ in range(iterations):
        middle = 0.5 * (lower + upper)
        derivative_middle = _nll_derivative(
            values, local_truth, middle, weights
        )
        if derivative_middle < 0.0:
            lower = middle
        else:
            upper = middle
    scale = 0.5 * (lower + upper)
    return scale, {
        "optimizer": "deterministic-bracketed-bisection-of-convex-nll-derivative",
        "lower_initial": 0.0,
        "upper_initial": 1.0,
        "upper_doublings": doublings,
        "bisection_iterations": iterations,
        "final_bracket": [lower, upper],
        "derivative_at_scale":
            _nll_derivative(values, local_truth, scale, weights),
    }


def display_metrics(
    distances: np.ndarray,
    truth: np.ndarray,
    support: np.ndarray,
    scale: float,
    sample_weight: np.ndarray | None = None,
) -> dict[str, Any]:
    values, labels, supported = _validate_rows(distances, truth, support)
    local_truth = _truth_local_indices(labels, supported)
    weights = _normalized_weights(len(labels), sample_weight)
    probability = _softmax_probabilities(values, scale)
    local_prediction = np.argmin(values, axis=1).astype(np.int64)
    prediction = supported[local_prediction]
    confidence = probability[
        np.arange(len(probability)), local_prediction
    ]
    correct = prediction == labels
    ece = 0.0
    bin_rows: list[dict[str, Any]] = []
    for bin_index in range(ECE_BINS):
        lower = bin_index / ECE_BINS
        upper = (bin_index + 1) / ECE_BINS
        selected = (
            (confidence >= lower)
            & (
                confidence <= upper
                if bin_index == ECE_BINS - 1
                else confidence < upper
            )
        )
        count = int(np.sum(selected))
        if not count:
            continue
        bin_weights = weights[selected]
        bin_mass = float(np.sum(bin_weights))
        accuracy = float(
            np.dot(bin_weights, correct[selected]) / bin_mass
        )
        mean_confidence = float(
            np.dot(bin_weights, confidence[selected]) / bin_mass
        )
        ece += bin_mass * abs(accuracy - mean_confidence)
        bin_rows.append(
            {
                "bin_index": bin_index,
                "rows": count,
                "weight_mass": bin_mass,
                "accuracy": accuracy,
                "mean_confidence": mean_confidence,
            }
        )
    selected_probability = probability[
        np.arange(len(labels)), local_truth
    ]
    return {
        "rows": int(len(labels)),
        "nll": float(-np.dot(weights, np.log(selected_probability))),
        "ece_15_equal_width": float(ece),
        "mean_winning_confidence": float(np.dot(weights, confidence)),
        "closed_accuracy": float(np.dot(weights, correct)),
        "prediction_sha256": hashlib.sha256(
            np.ascontiguousarray(prediction.astype(np.int64)).view(np.uint8)
        ).hexdigest(),
        "ece_nonempty_bins": bin_rows,
    }


def _enrollment_rows_by_route(
    context: openset.DevelopmentContext,
) -> dict[
    str,
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    distance_rows = openset._enrollment_distance_inputs(context)
    lengths = np.asarray(context.data["enrollment_lengths"], dtype=np.int64)
    truth = np.asarray(context.data["yen"], dtype=np.int64)
    if truth.shape != lengths.shape:
        raise RuntimeError("enrollment truth/length alignment changed")
    output: dict[
        str,
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ] = {}
    for source in openset.PROTOTYPE_SOURCE_ROUTES:
        distance_parts: list[np.ndarray] = []
        truth_parts: list[np.ndarray] = []
        group_parts: list[np.ndarray] = []
        for length in openset.RUNTIME_INPUT_LENGTHS:
            row = distance_rows[length]
            source_mask = np.asarray(
                row["prototype_sources"], dtype=object
            ) == source
            length_truth = truth[lengths == length]
            if len(length_truth) != len(source_mask):
                raise RuntimeError("enrollment distance/truth alignment changed")
            distance_parts.append(
                row["leave_one_base_identity_out_class_distances"][
                    source_mask
                ]
            )
            truth_parts.append(length_truth[source_mask])
            group_parts.append(
                np.asarray(row["groups"], dtype=object)[source_mask]
            )
        support = np.asarray(
            openset.EXPECTED_PUBLIC_CLASS_SUPPORT_BY_PROTOTYPE_SOURCE[source],
            dtype=bool,
        )
        output[source] = (
            np.concatenate(distance_parts),
            np.concatenate(truth_parts),
            support,
            np.concatenate(group_parts),
        )
    return output


def _selection_rows_by_route(
    context: openset.DevelopmentContext,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    output: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for source in openset.PROTOTYPE_SOURCE_ROUTES:
        prepared = context.data[f"{source}_selection"]["by_length"]
        distance_parts: list[np.ndarray] = []
        truth_parts: list[np.ndarray] = []
        for length in sorted(prepared):
            bucket = prepared[length]
            embedding = openset._embed(context, bucket["x"], bucket["f"])
            distance, _closest, support = openset._public_distances(
                context,
                embedding,
                prototype_source=source,
            )
            distance_parts.append(distance)
            truth_parts.append(np.asarray(bucket["y"], dtype=np.int64))
        expected_support = np.asarray(
            openset.EXPECTED_PUBLIC_CLASS_SUPPORT_BY_PROTOTYPE_SOURCE[source],
            dtype=bool,
        )
        if not np.array_equal(support, expected_support):
            raise RuntimeError("selection route support changed")
        output[source] = (
            np.concatenate(distance_parts),
            np.concatenate(truth_parts),
            expected_support,
        )
    return output


def fit_display_calibration(
    *,
    fusion: str | Path,
    historical_corpus: str | Path,
    current_corpus: str | Path,
    classifier_asset: str | Path,
    device_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    context = openset.load_development_context(
        fusion,
        historical_corpus,
        current_corpus,
        device_name=device_name,
    )
    binding = openset.build_classifier_binding(
        context.fusion, classifier_asset
    )
    enrollment = _enrollment_rows_by_route(context)
    selection = _selection_rows_by_route(context)
    scales: dict[str, float] = {}
    evidence_by_route: dict[str, Any] = {}
    for source in openset.PROTOTYPE_SOURCE_ROUTES:
        fit_distance, fit_truth, support, fit_groups = enrollment[source]
        fit_weights = profile_balanced_weights(fit_groups)
        scale, optimizer = fit_positive_distance_logit_scale(
            fit_distance,
            fit_truth,
            support,
            sample_weight=fit_weights,
        )
        reference = REFERENCE_SCALE[source]
        if abs(scale - reference) > REFERENCE_SCALE_ABSOLUTE_TOLERANCE:
            raise RuntimeError(
                f"{source} independently fitted scale {scale:.17g} differs "
                f"from preflight reference {reference:.17g}"
            )
        # Persist the independently fitted root, not a copied reference value.
        scales[source] = float(scale)
        selection_distance, selection_truth, selection_support = (
            selection[source]
        )
        evidence_by_route[source] = {
            "supported_public_class_mask": support.tolist(),
            "fit_population":
                "matching-route enrollment views only; every view sharing the "
                "query base identity excluded from its own source/profile "
                "centroid; all other enrollment-only centroids unchanged",
            "fit_weighting":
                "equal route-profile mass; each row in source/profile g has "
                "weight 1 / (number_of_route_profiles * rows_in_g)",
            "fit_rows": int(len(fit_truth)),
            "fit_route_profiles": int(len(set(fit_groups.tolist()))),
            "fit_weight_sum": float(np.sum(fit_weights)),
            "selection_rows_used_to_fit": 0,
            "novelty_rows_used_to_fit": 0,
            "held_scale_rows_used_to_fit_or_evaluate": 0,
            "open_set_design_or_validation_seed_consumed": False,
            "objective":
                "profile-balanced categorical NLL of "
                "exp(-scale*squared_distance) over supported routed public "
                "classes; no intercept or class weights",
            "optimizer": optimizer,
            "reference_scale": reference,
            "reference_absolute_error": float(abs(scale - reference)),
            "enrollment_identity_loo_metrics": {
                "before_scale_1": display_metrics(
                    fit_distance,
                    fit_truth,
                    support,
                    1.0,
                    sample_weight=fit_weights,
                ),
                "after_fitted_scale": display_metrics(
                    fit_distance,
                    fit_truth,
                    support,
                    scale,
                    sample_weight=fit_weights,
                ),
            },
            "selection_evaluation_only": {
                "before_scale_1": display_metrics(
                    selection_distance,
                    selection_truth,
                    selection_support,
                    1.0,
                ),
                "after_fitted_scale": display_metrics(
                    selection_distance,
                    selection_truth,
                    selection_support,
                    scale,
                ),
            },
        }
    artifact = {
        "schema": CALIBRATION_SCHEMA,
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "status": "staging_not_release",
        "development_only": True,
        "runtime_role": "conditional_display_calibration",
        "decision_authority": False,
        "calibration_kind": CALIBRATION_KIND,
        "classifier_asset_sha256":
            binding["browser_classifier_asset_sha256"],
        "distance_logit_scale_by_prototype_source": {
            "current": scales["current"],
            "historical": scales["historical"],
        },
    }
    artifact_bytes = _json_bytes(artifact)
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "complete_development_evidence",
        "development_only": True,
        "release_evidence": False,
        "classifier_asset_sha256":
            binding["browser_classifier_asset_sha256"],
        "display_calibration_asset_sha256":
            _sha256_bytes(artifact_bytes),
        "decision_authority": False,
        "argmin_and_open_set_decisions_changed": False,
        "calibration_kind": CALIBRATION_KIND,
        "uses_frequency_transform": False,
        "source_sha256": {
            **openset._source_hashes(),
            Path(__file__).name: _sha256(Path(__file__).resolve()),
        },
        "per_prototype_source": evidence_by_route,
    }
    return artifact, evidence


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = _json_bytes(payload)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"display-calibration output exists: {output}")
    artifact, evidence = fit_display_calibration(
        fusion=args.fusion,
        historical_corpus=args.historical_corpus,
        current_corpus=args.current_corpus,
        classifier_asset=args.classifier_asset,
        device_name=args.device,
    )
    output.mkdir(parents=True, exist_ok=False)
    _write_exclusive(output / CALIBRATION_NAME, artifact)
    _write_exclusive(output / EVIDENCE_NAME, evidence)
    if (
        _sha256(output / CALIBRATION_NAME)
        != evidence["display_calibration_asset_sha256"]
    ):
        raise RuntimeError("written display calibration differs from evidence")
    print(f"[v4 display calibration] wrote {output}", flush=True)
    return evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fusion", required=True)
    parser.add_argument(
        "--historical-corpus", default=str(openset.branch_runner.HISTORICAL_CORPUS)
    )
    parser.add_argument("--current-corpus", required=True)
    parser.add_argument("--classifier-asset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
