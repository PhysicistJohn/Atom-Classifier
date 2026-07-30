"""Generate deterministic Python expectations for the v4 browser open-set path.

This script never reads validation data and consumes no design/validation seed.
It operates only on an already-frozen classifier JSON and already-frozen policy
JSON, then emits compact routed embedding/pose cases for the TypeScript runtime.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
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
import export_current_source_browser_runtime as browser_export  # noqa: E402


FIXTURE_SCHEMA = "atomos.v4.time-domain-profile-bank.openset-browser-parity"
FIXTURE_SCHEMA_VERSION = 4


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _source_distances(
    embedding: np.ndarray,
    *,
    bank: np.ndarray,
    labels: np.ndarray,
    groups: list[Mapping[str, Any]],
    prototype_source: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = np.asarray(embedding, dtype=np.float64).reshape(1, -1)
    prototype_bank = np.asarray(bank, dtype=np.float64)
    if value.shape[1] != prototype_bank.shape[1]:
        raise ValueError("parity embedding and prototype bank dimensions differ")
    source = np.asarray(
        [str(group.get("source")) for group in groups], dtype=object
    )
    admitted = source == prototype_source
    squared = ((value[:, None, :] - prototype_bank[None, :, :]) ** 2).sum(
        axis=-1
    )
    distances = np.full((1, len(openset.PUBLIC_CLASSES)), np.inf)
    closest = np.full((1, len(openset.PUBLIC_CLASSES)), -1, dtype=np.int64)
    support = np.zeros(len(openset.PUBLIC_CLASSES), dtype=bool)
    for class_index in range(len(openset.PUBLIC_CLASSES)):
        positions = np.flatnonzero(admitted & (labels == class_index))
        if not len(positions):
            continue
        support[class_index] = True
        local = int(np.argmin(squared[0, positions]))
        closest[0, class_index] = int(positions[local])
        distances[0, class_index] = float(squared[0, positions[local]])
    expected = np.asarray(
        openset.EXPECTED_PUBLIC_CLASS_SUPPORT_BY_PROTOTYPE_SOURCE[
            prototype_source
        ],
        dtype=bool,
    )
    if not np.array_equal(support, expected):
        raise ValueError(
            f"classifier {prototype_source} support differs from policy"
        )
    return distances, closest, support


def _pose_score(
    policy: Mapping[str, Any],
    pose_features: np.ndarray,
    length: int,
) -> float:
    row = policy["stage_one"]["parameters_by_length"][str(length)]
    mean = np.asarray(row["mean"], dtype=np.float64)
    scale = np.asarray(row["scale"], dtype=np.float64)
    coefficient = np.asarray(row["coefficients"], dtype=np.float64)
    features = np.asarray(pose_features, dtype=np.float64)
    if features.shape != mean.shape:
        raise ValueError("pose parity feature width changed")
    return float(
        float(row["intercept"])
        + np.dot(coefficient, (features - mean) / scale)
    )


def _case(
    *,
    case_id: str,
    policy: Mapping[str, Any],
    embedding: np.ndarray,
    pose_features: np.ndarray,
    prototype_source: str,
    length: int,
    bank: np.ndarray,
    labels: np.ndarray,
    groups: list[Mapping[str, Any]],
) -> dict[str, Any]:
    distances, closest, support = _source_distances(
        embedding,
        bank=bank,
        labels=labels,
        groups=groups,
        prototype_source=prototype_source,
    )
    pose_score = _pose_score(policy, pose_features, length)
    phase_linearity_score = (
        int(hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:2], 16)
        % 3
    ) / 2.0
    scored = openset._score_from_distances(
        policy,
        length=length,
        prototype_source=prototype_source,
        stage_one_score=np.asarray([pose_score], dtype=np.float64),
        phase_linearity_score=np.asarray(
            [phase_linearity_score], dtype=np.float64
        ),
        class_distances=distances,
    )
    prediction = int(scored.prediction[0])
    if not support[prediction]:
        raise AssertionError("unsupported class won Python parity case")
    return {
        "id": case_id,
        "prototype_source": prototype_source,
        "runtime_input_length": int(length),
        "embedding": np.asarray(embedding, dtype=np.float64).tolist(),
        "pose_features": np.asarray(
            pose_features, dtype=np.float64
        ).tolist(),
        "expected": {
            "observation_runtime_input_length": int(length),
            "predicted_public_class_index": prediction,
            "predicted_public_class": openset.PUBLIC_CLASSES[prediction],
            "winning_squared_distance":
                float(scored.winning_distance[0]),
            "winning_prototype_index": int(closest[0, prediction]),
            "supported_public_class_mask": support.tolist(),
            "squared_public_class_distances": [
                float(value) if math_is_finite(value) else None
                for value in distances[0]
            ],
            "closest_prototype_indices": closest[0].tolist(),
            "stage_one_score": float(scored.stage_one_score[0]),
            "stage_one_predicted_class_rank":
                float(scored.stage_one_predicted_class_rank[0]),
            "stage_one_pooled_rank":
                float(scored.stage_one_pooled_rank[0]),
            "stage_one_rank": float(scored.stage_one_rank[0]),
            "stage_two_rank": float(scored.stage_two_rank[0]),
            "instantaneous_phase_linearity_score":
                float(scored.phase_linearity_score[0]),
            "instantaneous_phase_linearity_rank":
                float(scored.phase_linearity_rank[0]),
            "composite_rank": float(scored.composite_score[0]),
            "threshold": float(
                policy["per_prototype_source"][prototype_source][
                    "per_length"
                ][str(length)]["composition"][
                    "threshold_by_predicted_public_class"
                ][prediction]
            ),
            "rejected": bool(scored.rejected[0]),
        },
    }


def math_is_finite(value: Any) -> bool:
    return bool(np.isfinite(float(value)))


def _f32le_base64(values: np.ndarray) -> tuple[str, str]:
    raw = np.asarray(values, dtype="<f4").tobytes(order="C")
    return (
        base64.b64encode(raw).decode("ascii"),
        hashlib.sha256(raw).hexdigest(),
    )


def _phase_linearity_raw_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for length in openset.RUNTIME_INPUT_LENGTHS:
        sample = np.arange(length, dtype=np.float64)
        normalized = sample / (length - 1)
        envelope = 0.55 + 0.35 * np.cos(
            2.0 * np.pi * 3.0 * normalized + 0.4
        )
        phase = 2.0 * np.pi * (
            0.17 * sample + 0.31 * sample * normalized
        )
        chirp = np.asarray(
            envelope * np.exp(1j * phase), dtype=np.complex64
        )
        transformed_chirp = np.asarray(
            1e4
            * chirp.astype(np.complex128)
            * np.exp(1j * (0.71 + 0.013 * sample)),
            dtype=np.complex64,
        )
        time_scaled_chirp = np.asarray(
            np.exp(
                1j
                * 2.0
                * np.pi
                * (
                    -0.11 * sample
                    + 0.23 * sample * normalized
                )
            ),
            dtype=np.complex64,
        )
        cw = np.asarray(
            np.exp(1j * (0.2 * sample + 0.3)), dtype=np.complex64
        )
        deterministic_noise = 0.01 * (
            np.sin(sample * 1.618033988749895 + 0.2)
            + 1j * np.cos(sample * 1.4142135623730951 - 0.1)
        )
        noisy_cw = np.asarray(
            cw.astype(np.complex128) + deterministic_noise,
            dtype=np.complex64,
        )
        jitter = 0.015 * np.sin(2.0 * np.pi * sample / 701.0)
        robust = np.asarray(
            chirp.astype(np.complex128) * np.exp(1j * jitter)
            + deterministic_noise,
            dtype=np.complex64,
        )
        gap_start = int(round(length * 0.34))
        robust[gap_start:gap_start + 160] = 0.0
        insufficient = np.zeros(length, dtype=np.complex64)
        insufficient[:100] = chirp[:100]
        nonlinear_phase = (
            0.2 * sample
            + 0.7 * np.sin(2.0 * np.pi * sample / 97.0)
            + 0.15 * np.sin(2.0 * np.pi * sample / 431.0)
        )
        nonlinear_known_like = np.asarray(
            envelope * np.exp(1j * nonlinear_phase),
            dtype=np.complex64,
        )
        length_values = (
            (
                "lfm",
                chirp,
                {"score_minimum": 0.999999999},
            ),
            (
                "scaled-phase-carrier-lfm",
                transformed_chirp,
                {
                    "score_minimum": 0.999999999,
                    "reference_case": f"N{length}-lfm",
                    "maximum_reference_delta": 1e-9,
                },
            ),
            (
                "time-scaled-lfm",
                time_scaled_chirp,
                {"score_minimum": 0.999999999},
            ),
            (
                "cw",
                cw,
                {"score_minimum": 0.0, "score_maximum": 0.0},
            ),
            (
                "noisy-cw",
                noisy_cw,
                {"score_maximum": 0.9999},
            ),
            (
                "gapped-envelope-noise-jitter-lfm",
                robust,
                {"score_minimum": 0.95},
            ),
            (
                "insufficient-active-coverage",
                insufficient,
                {"score_minimum": 0.0, "score_maximum": 0.0},
            ),
            (
                "nonlinear-known-like-phase",
                nonlinear_known_like,
                {"score_maximum": 0.9999},
            ),
        )
        for case_name, row, property_contract in length_values:
            in_phase = row.real.astype("<f4", copy=False)
            quadrature = row.imag.astype("<f4", copy=False)
            encoded_i, sha_i = _f32le_base64(in_phase)
            encoded_q, sha_q = _f32le_base64(quadrature)
            score = openset.instantaneous_phase_linearity_chirp_score(
                row
            )
            minimum = property_contract.get("score_minimum")
            maximum = property_contract.get("score_maximum")
            if (
                minimum is not None
                and score < float(minimum)
            ) or (
                maximum is not None
                and score > float(maximum)
            ):
                raise AssertionError(
                    f"N{length}-{case_name} violates its property contract"
                )
            cases.append(
                {
                    "id": f"N{length}-{case_name}",
                    "runtime_input_length": int(length),
                    "sample_encoding": "float32-little-endian-base64",
                    "in_phase_f32le_base64": encoded_i,
                    "quadrature_f32le_base64": encoded_q,
                    "in_phase_sha256": sha_i,
                    "quadrature_sha256": sha_q,
                    "expected_score": score,
                    "property_contract": property_contract,
                }
            )
    scores = {
        str(row["id"]): float(row["expected_score"]) for row in cases
    }
    for row in cases:
        contract = row["property_contract"]
        reference = contract.get("reference_case")
        if (
            reference is not None
            and abs(
                float(row["expected_score"]) - scores[str(reference)]
            )
            > float(contract["maximum_reference_delta"])
        ):
            raise AssertionError(
                f"{row['id']} violates its reference property contract"
            )
    return cases


def build_fixture(
    classifier_path: Path,
    policy_path: Path,
) -> dict[str, Any]:
    classifier = _load_json(classifier_path)
    policy = openset.validate_policy(_load_json(policy_path))
    classifier_sha = _sha256(classifier_path)
    policy_sha = _sha256(policy_path)
    if (
        policy["classifier_binding"]["browser_classifier_asset_sha256"]
        != classifier_sha
    ):
        raise ValueError("policy is bound to another classifier byte stream")
    if (
        classifier.get("schema") != browser_export.BROWSER_SCHEMA
        or classifier.get("schema_version")
        != browser_export.BROWSER_SCHEMA_VERSION
    ):
        raise ValueError("classifier browser schema changed")
    classification = classifier.get("classification")
    if not isinstance(classification, Mapping):
        raise ValueError("classifier classification object is missing")
    if classification.get("routing") != browser_export.TRUSTED_SOURCE_ROUTING:
        raise ValueError("classifier routing differs from the trusted contract")
    bank = np.asarray(classification["prototype_bank"], dtype=np.float64)
    labels = np.asarray(
        classification["prototype_public_class_indices"], dtype=np.int64
    )
    raw_groups = classification["prototype_groups"]
    if (
        bank.ndim != 2
        or labels.shape != (len(bank),)
        or not isinstance(raw_groups, list)
        or len(raw_groups) != len(bank)
        or not all(isinstance(group, Mapping) for group in raw_groups)
    ):
        raise ValueError("classifier prototype arrays/groups do not align")
    groups = list(raw_groups)

    cases: list[dict[str, Any]] = []
    for prototype_source in openset.PROTOTYPE_SOURCE_ROUTES:
        support = openset.EXPECTED_PUBLIC_CLASS_SUPPORT_BY_PROTOTYPE_SOURCE[
            prototype_source
        ]
        for length in openset._runtime_policy_lengths(prototype_source):
            parameters = policy["stage_one"]["parameters_by_length"][
                str(length)
            ]
            pose_features = np.asarray(parameters["mean"], dtype=np.float64)
            for class_index, supported in enumerate(support):
                if not supported:
                    continue
                positions = [
                    index
                    for index, group in enumerate(groups)
                    if group["source"] == prototype_source
                    and int(labels[index]) == class_index
                ]
                if not positions:
                    raise ValueError(
                        f"no {prototype_source} prototype for class "
                        f"{openset.PUBLIC_CLASSES[class_index]}"
                    )
                prototype_index = positions[0]
                cases.append(
                    _case(
                        case_id=(
                            f"{prototype_source}-N{length}-"
                            f"{openset.PUBLIC_CLASSES[class_index]}"
                        ),
                        policy=policy,
                        embedding=bank[prototype_index],
                        pose_features=pose_features,
                        prototype_source=prototype_source,
                        length=length,
                        bank=bank,
                        labels=labels,
                        groups=groups,
                    )
                )

    # Route-leak sentinels: embeddings equal to unsupported historical analog
    # prototypes must still score only the current BT/DSSS/GSM/OFDM bank.
    for class_index in (0, 2, 4):
        historical = next(
            index
            for index, group in enumerate(groups)
            if group["source"] == "historical"
            and int(labels[index]) == class_index
        )
        length = openset.RUNTIME_INPUT_LENGTHS[0]
        pose_features = np.asarray(
            policy["stage_one"]["parameters_by_length"][str(length)]["mean"],
            dtype=np.float64,
        )
        cases.append(
            _case(
                case_id=(
                    "current-route-leak-sentinel-historical-"
                    f"{openset.PUBLIC_CLASSES[class_index]}"
                ),
                policy=policy,
                embedding=bank[historical],
                pose_features=pose_features,
                prototype_source="current",
                length=length,
                bank=bank,
                labels=labels,
                groups=groups,
            )
        )

    rank_calibration = [0.0, 1.0, 1.0, 2.0]
    return {
        "schema": FIXTURE_SCHEMA,
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "classifier_sha256": classifier_sha,
        "policy_sha256": policy_sha,
        "uses_frequency_transform": False,
        "tolerance": {
            "absolute": 1e-10,
            "relative": 1e-10,
        },
        "rank_edge_cases": [
            {
                "calibration": rank_calibration,
                "query": 1.0,
                "expected_less_count": 1,
                "expected_rank": 1.0 / 5.0,
            },
            {
                "calibration": rank_calibration,
                "query": 1.5,
                "expected_less_count": 3,
                "expected_rank": 3.0 / 5.0,
            },
        ],
        "threshold_comparison": {
            "rule": "reject iff composite_rank >= threshold",
            "equality_rejects": True,
        },
        "exact_zero": {
            "prototype_source": "current",
            "runtime_input_length": 4_096,
            "expected_disposition": "unknown",
            "expected_reason": "exact_no_signal",
            "closed_decision": None,
        },
        "phase_linearity_raw_cases": _phase_linearity_raw_cases(),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite parity fixture {output}")
    payload = build_fixture(
        Path(args.classifier).resolve(),
        Path(args.policy).resolve(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    with output.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
    print(f"wrote {output} ({len(payload['cases'])} routed cases)")


if __name__ == "__main__":
    main()
