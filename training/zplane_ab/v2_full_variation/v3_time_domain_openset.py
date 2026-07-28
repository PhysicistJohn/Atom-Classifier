"""Compact, known-only time-domain rejection for invariant I/Q patches.

This module is an isolated v3 development experiment.  It never changes the
closed-set label.  Training captures fit class-conditional reference geometry;
disjoint enrollment captures calibrate empirical ranks and the final operating
threshold.  Novelty examples may be used by a caller to select one of the
explicit policies below, but are never accepted by :meth:`fit`.

The statistics operate on the fixed-u packed I/Q patches and use no FFT.  They
are symmetric over the patch set and invariant to a global complex phase.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


EPSILON = 1e-8
PATCH_COUNT = 16
PATCH_LENGTH = 64
LAGS = (1, 2, 4, 8, 16, 32)
FROZEN_POLICY_SCHEMA = 1
FROZEN_POLICY_KIND = "v3_known_only_lof_frequency_dispersion_rank_blend"
FROZEN_GEOMETRY_FEATURE = "across_patch_frequency_dispersion"
FROZEN_V2_WEIGHT = 0.80
FROZEN_GEOMETRY_WEIGHT = 0.20
FROZEN_THRESHOLD_QUANTILE = 0.95


def _validate_packed(
    packed: np.ndarray,
    *,
    patch_count: int,
    patch_length: int,
) -> np.ndarray:
    value = np.asarray(packed, dtype=np.float32)
    if value.ndim != 3 or value.shape[1:] != (
        2,
        patch_count * patch_length,
    ):
        raise ValueError(
            "packed must have shape "
            f"[rows, 2, {patch_count * patch_length}]"
        )
    if not len(value) or not np.isfinite(value).all():
        raise ValueError("packed must be non-empty and finite")
    return (
        value[:, 0].reshape(len(value), patch_count, patch_length).astype(
            np.float64
        )
        + 1j
        * value[:, 1].reshape(len(value), patch_count, patch_length).astype(
            np.float64
        )
    )


def _pool(
    columns: list[np.ndarray],
    names: list[str],
    prefix: str,
    value: np.ndarray,
    reducers: Sequence[str] = ("mean", "std", "q10", "q90"),
) -> None:
    current = np.asarray(value, dtype=np.float64)
    for reducer in reducers:
        if reducer == "mean":
            reduced = current.mean(axis=1)
        elif reducer == "std":
            reduced = current.std(axis=1)
        elif reducer == "q10":
            reduced = np.quantile(current, 0.1, axis=1)
        elif reducer == "q90":
            reduced = np.quantile(current, 0.9, axis=1)
        elif reducer == "max":
            reduced = current.max(axis=1)
        else:  # pragma: no cover - reducers are compile-time constants
            raise ValueError(f"unknown reducer {reducer!r}")
        columns.append(reduced)
        names.append(f"{prefix}_{reducer}")


def time_domain_geometry(
    packed: np.ndarray,
    *,
    patch_count: int = PATCH_COUNT,
    patch_length: int = PATCH_LENGTH,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return phase-invariant, patch-order-invariant time-domain geometry.

    The descriptors cover two physically interpretable regimes:

    * stochastic/noise-like captures: amplitude shape and lagged coherence;
    * swept carriers: locally linear instantaneous frequency together with
      across-patch frequency dispersion.
    """
    z = _validate_packed(
        packed,
        patch_count=patch_count,
        patch_length=patch_length,
    )
    amplitude = np.abs(z)
    power = amplitude * amplitude
    mean_amplitude = amplitude.mean(axis=2)
    rms = np.sqrt(power.mean(axis=2))
    centered_amplitude = amplitude - mean_amplitude[:, :, None]
    columns: list[np.ndarray] = []
    names: list[str] = []

    _pool(
        columns,
        names,
        "amplitude_cv",
        amplitude.std(axis=2) / np.maximum(mean_amplitude, EPSILON),
    )
    _pool(
        columns,
        names,
        "amplitude_kurtosis",
        (centered_amplitude**4).mean(axis=2)
        / np.maximum((centered_amplitude**2).mean(axis=2) ** 2, EPSILON),
    )
    _pool(
        columns,
        names,
        "peak_to_rms",
        amplitude.max(axis=2) / np.maximum(rms, EPSILON),
        ("mean", "std", "q90"),
    )
    _pool(
        columns,
        names,
        "pseudocovariance",
        np.abs((z * z).mean(axis=2))
        / np.maximum(power.mean(axis=2), EPSILON),
    )

    for lag in LAGS:
        first, second = z[:, :, :-lag], z[:, :, lag:]
        denominator = np.sqrt(
            (np.abs(first) ** 2).mean(axis=2)
            * (np.abs(second) ** 2).mean(axis=2)
        )
        correlation = (first * np.conj(second)).mean(axis=2) / np.maximum(
            denominator, EPSILON
        )
        _pool(
            columns,
            names,
            f"coherence_lag{lag}",
            np.clip(np.abs(correlation), 0.0, 1.0),
            ("mean", "std", "q10", "q90"),
        )

    increments = np.angle(z[:, :, 1:] * np.conj(z[:, :, :-1]))
    increment_phasor = np.exp(1j * increments)
    increment_coherence = np.abs(increment_phasor.mean(axis=2))
    _pool(
        columns,
        names,
        "increment_coherence",
        np.clip(increment_coherence, 0.0, 1.0),
    )
    patch_frequency = np.angle(increment_phasor.mean(axis=2))
    columns.append(
        1.0 - np.abs(np.exp(1j * patch_frequency).mean(axis=1))
    )
    names.append("across_patch_frequency_dispersion")

    curvature = np.angle(
        np.exp(1j * (increments[:, :, 1:] - increments[:, :, :-1]))
    )
    curvature_phasor = np.exp(1j * curvature)
    curvature_coherence = np.abs(curvature_phasor.mean(axis=2))
    _pool(
        columns,
        names,
        "curvature_coherence",
        np.clip(curvature_coherence, 0.0, 1.0),
    )
    curvature_mean = np.abs(np.angle(curvature_phasor.mean(axis=2)))
    _pool(
        columns,
        names,
        "curvature_magnitude",
        curvature_mean,
        ("mean", "std", "q90", "max"),
    )

    # A chirp has a nearly affine instantaneous frequency in each local patch.
    # Least-squares slope/R² needs only fixed scalar reductions and is easy to
    # reproduce in a browser worker.
    position = np.arange(increments.shape[2], dtype=np.float64)
    position -= position.mean()
    denominator = float(np.sum(position * position))
    unwrapped = np.unwrap(increments, axis=2)
    centered = unwrapped - unwrapped.mean(axis=2, keepdims=True)
    slope = np.einsum("npt,t->np", centered, position) / denominator
    fitted = slope[:, :, None] * position[None, None, :]
    residual = centered - fitted
    total_energy = np.sum(centered * centered, axis=2)
    residual_energy = np.sum(residual * residual, axis=2)
    linearity = np.where(
        total_energy > EPSILON,
        1.0 - residual_energy / np.maximum(total_energy, EPSILON),
        0.0,
    )
    _pool(
        columns,
        names,
        "frequency_slope_magnitude",
        np.abs(slope),
        ("mean", "std", "q90", "max"),
    )
    _pool(
        columns,
        names,
        "frequency_linearity",
        np.clip(linearity, 0.0, 1.0),
    )
    _pool(
        columns,
        names,
        "frequency_residual",
        np.sqrt(residual_energy / increments.shape[2]),
        ("mean", "std", "q90"),
    )

    result = np.stack(columns, axis=1).astype(np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("time-domain geometry produced a non-finite value")
    return result, tuple(names)


def empirical_rank(
    value: np.ndarray,
    calibration: np.ndarray,
) -> np.ndarray:
    """Strict empirical upper-tail rank in ``[0, 1)``."""
    query = np.asarray(value, dtype=np.float64)
    ordered = np.sort(np.asarray(calibration, dtype=np.float64))
    if (
        query.ndim != 1
        or ordered.ndim != 1
        or not len(ordered)
        or not np.isfinite(query).all()
        or not np.isfinite(ordered).all()
    ):
        raise ValueError("value and calibration must be finite vectors")
    return np.searchsorted(ordered, query, side="left") / (
        len(ordered) + 1.0
    )


def _labels(value: np.ndarray, rows: int, n_classes: int) -> np.ndarray:
    result = np.asarray(value)
    if (
        result.ndim != 1
        or len(result) != rows
        or not np.issubdtype(result.dtype, np.integer)
    ):
        raise ValueError("labels must be an integer vector aligned with rows")
    result = result.astype(np.int64, copy=False)
    if result.min() < 0 or result.max() >= n_classes:
        raise ValueError("labels lie outside fitted classes")
    return result


@dataclass(frozen=True)
class KnownOnlyGeometry:
    """Per-class standardized scalar deviations fitted without novelty."""

    class_mean: np.ndarray
    class_scale: np.ndarray
    feature_names: tuple[str, ...]

    @classmethod
    def fit(
        cls,
        training_packed: np.ndarray,
        training_labels: np.ndarray,
        *,
        n_classes: int,
    ) -> "KnownOnlyGeometry":
        statistics, names = time_domain_geometry(training_packed)
        labels = _labels(training_labels, len(statistics), n_classes)
        means, scales = [], []
        for class_index in range(n_classes):
            current = statistics[labels == class_index].astype(np.float64)
            if len(current) < 2:
                raise ValueError(f"class {class_index} needs at least two rows")
            means.append(current.mean(axis=0))
            scales.append(np.maximum(current.std(axis=0), 1e-5))
        return cls(np.stack(means), np.stack(scales), names)

    def deviations(
        self,
        packed: np.ndarray,
        predicted_classes: np.ndarray,
    ) -> np.ndarray:
        statistics, names = time_domain_geometry(packed)
        if names != self.feature_names:
            raise RuntimeError("time-domain feature schema changed")
        predicted = _labels(
            predicted_classes,
            len(statistics),
            len(self.class_mean),
        )
        return np.abs(
            (
                statistics.astype(np.float64) - self.class_mean[predicted]
            )
            / self.class_scale[predicted]
        )

    def indices(self, names: Iterable[str]) -> np.ndarray:
        lookup = {name: index for index, name in enumerate(self.feature_names)}
        requested = tuple(names)
        if not requested:
            raise ValueError("at least one feature is required")
        try:
            return np.asarray([lookup[name] for name in requested], dtype=np.int64)
        except KeyError as exc:
            raise ValueError(f"unknown time-domain feature {exc.args[0]!r}") from exc


def _finite_vector(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a non-empty finite vector")
    return result


def _rank_score(value: np.ndarray, name: str) -> np.ndarray:
    result = _finite_vector(value, name)
    if np.any(result < 0.0) or np.any(result >= 1.0):
        raise ValueError(f"{name} must lie in [0, 1)")
    return result


@dataclass(frozen=True)
class FrozenV3OpenSet:
    """The immutable seed-1-selected v3 score calibrated on known enrollment.

    ``v2_scores`` are the existing weighted real-k2/complex-k64 LOF ranks.
    This class adds exactly one time-domain nonconformity rank and then ranks
    the frozen 0.80/0.20 blend against enrollment once more.  No method accepts
    novelty data, and predicted closed labels are inputs rather than outputs.
    """

    geometry: KnownOnlyGeometry
    geometry_feature_index: int
    geometry_calibration_raw: np.ndarray
    combined_calibration_raw: np.ndarray
    calibration_scores: np.ndarray
    threshold: float

    @classmethod
    def fit(
        cls,
        training_packed: np.ndarray,
        training_labels: np.ndarray,
        enrollment_packed: np.ndarray,
        enrollment_predicted_classes: np.ndarray,
        enrollment_v2_scores: np.ndarray,
        *,
        n_classes: int,
    ) -> "FrozenV3OpenSet":
        """Fit class geometry on training and both ranks on enrollment only."""
        geometry = KnownOnlyGeometry.fit(
            training_packed,
            training_labels,
            n_classes=n_classes,
        )
        feature_index = int(
            geometry.indices((FROZEN_GEOMETRY_FEATURE,))[0]
        )
        enrollment_deviation = geometry.deviations(
            enrollment_packed,
            enrollment_predicted_classes,
        )[:, feature_index]
        geometry_calibration = np.sort(
            _finite_vector(enrollment_deviation, "enrollment geometry")
        )
        geometry_rank = empirical_rank(
            enrollment_deviation,
            geometry_calibration,
        )
        v2_rank = _rank_score(enrollment_v2_scores, "enrollment_v2_scores")
        if len(v2_rank) != len(geometry_rank):
            raise ValueError("enrollment score and packed row counts differ")
        combined_raw = (
            FROZEN_V2_WEIGHT * v2_rank
            + FROZEN_GEOMETRY_WEIGHT * geometry_rank
        )
        combined_calibration = np.sort(combined_raw)
        calibration_scores = empirical_rank(
            combined_raw,
            combined_calibration,
        )
        threshold = float(
            np.quantile(calibration_scores, FROZEN_THRESHOLD_QUANTILE)
        )
        return cls(
            geometry=geometry,
            geometry_feature_index=feature_index,
            geometry_calibration_raw=geometry_calibration,
            combined_calibration_raw=combined_calibration,
            calibration_scores=calibration_scores,
            threshold=threshold,
        )

    def score(
        self,
        v2_scores: np.ndarray,
        packed: np.ndarray,
        predicted_classes: np.ndarray,
    ) -> np.ndarray:
        """Return the final enrollment-ranked score without changing labels."""
        v2_rank = _rank_score(v2_scores, "v2_scores")
        deviation = self.geometry.deviations(
            packed,
            predicted_classes,
        )
        if len(v2_rank) != len(deviation):
            raise ValueError("v2 score and packed row counts differ")
        geometry_rank = empirical_rank(
            deviation[:, self.geometry_feature_index],
            self.geometry_calibration_raw,
        )
        combined_raw = (
            FROZEN_V2_WEIGHT * v2_rank
            + FROZEN_GEOMETRY_WEIGHT * geometry_rank
        )
        return empirical_rank(combined_raw, self.combined_calibration_raw)

    def to_payload(self) -> dict[str, np.ndarray]:
        """Return an allow-pickle-free ``np.savez`` payload."""
        return {
            "schema": np.asarray(FROZEN_POLICY_SCHEMA, dtype=np.int64),
            "kind": np.asarray(FROZEN_POLICY_KIND),
            "geometry_feature": np.asarray(FROZEN_GEOMETRY_FEATURE),
            "v2_weight": np.asarray(FROZEN_V2_WEIGHT, dtype=np.float64),
            "geometry_weight": np.asarray(
                FROZEN_GEOMETRY_WEIGHT,
                dtype=np.float64,
            ),
            "threshold_quantile": np.asarray(
                FROZEN_THRESHOLD_QUANTILE,
                dtype=np.float64,
            ),
            "threshold": np.asarray(self.threshold, dtype=np.float64),
            "geometry_feature_index": np.asarray(
                self.geometry_feature_index,
                dtype=np.int64,
            ),
            "feature_names": np.asarray(self.geometry.feature_names),
            "class_mean": np.asarray(
                self.geometry.class_mean,
                dtype=np.float64,
            ),
            "class_scale": np.asarray(
                self.geometry.class_scale,
                dtype=np.float64,
            ),
            "geometry_calibration_raw": np.asarray(
                self.geometry_calibration_raw,
                dtype=np.float64,
            ),
            "combined_calibration_raw": np.asarray(
                self.combined_calibration_raw,
                dtype=np.float64,
            ),
            "calibration_scores": np.asarray(
                self.calibration_scores,
                dtype=np.float64,
            ),
        }

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, np.ndarray],
    ) -> "FrozenV3OpenSet":
        """Validate and restore an allow-pickle-free serialized policy."""
        required = {
            "schema",
            "kind",
            "geometry_feature",
            "v2_weight",
            "geometry_weight",
            "threshold_quantile",
            "threshold",
            "geometry_feature_index",
            "feature_names",
            "class_mean",
            "class_scale",
            "geometry_calibration_raw",
            "combined_calibration_raw",
            "calibration_scores",
        }
        if set(payload) != required:
            raise ValueError(
                "serialized policy keys differ: "
                f"missing={sorted(required - set(payload))}, "
                f"extra={sorted(set(payload) - required)}"
            )
        scalar_contract = {
            "schema": FROZEN_POLICY_SCHEMA,
            "kind": FROZEN_POLICY_KIND,
            "geometry_feature": FROZEN_GEOMETRY_FEATURE,
            "v2_weight": FROZEN_V2_WEIGHT,
            "geometry_weight": FROZEN_GEOMETRY_WEIGHT,
            "threshold_quantile": FROZEN_THRESHOLD_QUANTILE,
        }
        for name, expected in scalar_contract.items():
            value = np.asarray(payload[name])
            if value.ndim != 0 or value.item() != expected:
                raise ValueError(f"serialized {name} does not match policy")

        names_array = np.asarray(payload["feature_names"])
        if names_array.ndim != 1 or not len(names_array):
            raise ValueError("serialized feature_names must be a vector")
        names = tuple(str(value) for value in names_array.tolist())
        mean = np.asarray(payload["class_mean"], dtype=np.float64)
        scale = np.asarray(payload["class_scale"], dtype=np.float64)
        if (
            mean.ndim != 2
            or mean.shape != scale.shape
            or mean.shape[1] != len(names)
            or not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or np.any(scale <= 0.0)
        ):
            raise ValueError("serialized class geometry is invalid")
        feature_index = int(np.asarray(payload["geometry_feature_index"]).item())
        if (
            feature_index < 0
            or feature_index >= len(names)
            or names[feature_index] != FROZEN_GEOMETRY_FEATURE
        ):
            raise ValueError("serialized geometry feature index is invalid")
        geometry_calibration = np.asarray(
            payload["geometry_calibration_raw"],
            dtype=np.float64,
        )
        combined_calibration = np.asarray(
            payload["combined_calibration_raw"],
            dtype=np.float64,
        )
        calibration_scores = _rank_score(
            payload["calibration_scores"],
            "calibration_scores",
        )
        for name, value in (
            ("geometry_calibration_raw", geometry_calibration),
            ("combined_calibration_raw", combined_calibration),
        ):
            _finite_vector(value, name)
            if np.any(value[1:] < value[:-1]):
                raise ValueError(f"serialized {name} must be sorted")
        if not (
            len(geometry_calibration)
            == len(combined_calibration)
            == len(calibration_scores)
        ):
            raise ValueError("serialized calibration lengths differ")
        derived_calibration_scores = np.searchsorted(
            combined_calibration,
            combined_calibration,
            side="left",
        ) / (len(combined_calibration) + 1.0)
        if not np.array_equal(
            np.sort(calibration_scores), derived_calibration_scores
        ):
            raise ValueError(
                "serialized calibration_scores are not the empirical "
                "self-ranks of combined_calibration_raw"
            )
        threshold = float(np.asarray(payload["threshold"]).item())
        expected_threshold = float(
            np.quantile(
                derived_calibration_scores,
                FROZEN_THRESHOLD_QUANTILE,
            )
        )
        if (
            not np.isfinite(threshold)
            or threshold != expected_threshold
        ):
            raise ValueError("serialized threshold is invalid")
        return cls(
            geometry=KnownOnlyGeometry(mean, scale, names),
            geometry_feature_index=feature_index,
            geometry_calibration_raw=geometry_calibration,
            combined_calibration_raw=combined_calibration,
            calibration_scores=calibration_scores,
            threshold=threshold,
        )


def aggregate_deviation(
    deviations: np.ndarray,
    indices: np.ndarray,
    *,
    reducer: str,
) -> np.ndarray:
    """Reduce selected standardized deviations to one raw nonconformity."""
    value = np.asarray(deviations, dtype=np.float64)
    selected = value[:, np.asarray(indices, dtype=np.int64)]
    if selected.ndim != 2 or not selected.shape[1]:
        raise ValueError("selected deviations must be a non-empty matrix")
    if reducer == "max":
        return selected.max(axis=1)
    if reducer == "rms":
        return np.sqrt(np.mean(selected * selected, axis=1))
    if reducer == "mean":
        return selected.mean(axis=1)
    raise ValueError("reducer must be max, rms, or mean")


__all__ = [
    "FROZEN_GEOMETRY_FEATURE",
    "FROZEN_GEOMETRY_WEIGHT",
    "FROZEN_POLICY_KIND",
    "FROZEN_POLICY_SCHEMA",
    "FROZEN_THRESHOLD_QUANTILE",
    "FROZEN_V2_WEIGHT",
    "FrozenV3OpenSet",
    "KnownOnlyGeometry",
    "aggregate_deviation",
    "empirical_rank",
    "time_domain_geometry",
]
