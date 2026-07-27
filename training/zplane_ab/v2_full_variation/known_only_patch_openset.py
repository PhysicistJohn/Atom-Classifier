"""Known-only density rejection for the invariant-patch classifier.

This scorer is deliberately separate from closed-set classification.  The
authoritative classifier supplies ``predicted_classes`` unchanged.  Novelty is
then measured by two complementary, known-only density estimates:

* one-nearest-neighbour distance to training embeddings; and
* predicted-class Mahalanobis distance over deterministic, phase-invariant
  patch statistics.

Training captures fit both reference populations and covariance matrices.
Disjoint enrollment captures calibrate empirical ranks and the operating
threshold.  Selection or novelty captures are never used by :meth:`fit`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_LAGS = (1, 2, 4, 8, 16, 24, 32)
_EPSILON = 1e-8


def _finite_matrix(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError(f"{name} must be a non-empty matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _labels(value: np.ndarray, rows: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or len(array) != rows:
        raise ValueError(f"{name} must be a length-{rows} vector")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
            raise ValueError(f"{name} must contain integer class indices")
    return array.astype(np.int64, copy=False)


def _packed_complex(
    packed: np.ndarray,
    *,
    patch_count: int,
) -> np.ndarray:
    value = np.asarray(packed, dtype=np.float32)
    if (
        value.ndim != 3
        or value.shape[1] != 2
        or not value.shape[0]
        or value.shape[2] % patch_count
    ):
        raise ValueError(
            "packed must have shape [rows, 2, patch_count * patch_length]"
        )
    if not np.isfinite(value).all():
        raise ValueError("packed must contain only finite values")
    patch_length = value.shape[2] // patch_count
    if patch_length <= max(_LAGS):
        raise ValueError(f"patch length must exceed {max(_LAGS)}")
    return (
        value[:, 0].reshape(len(value), patch_count, patch_length).astype(np.float64)
        + 1j
        * value[:, 1].reshape(len(value), patch_count, patch_length).astype(np.float64)
    )


def patch_statistics(
    packed: np.ndarray,
    *,
    patch_count: int = 16,
) -> np.ndarray:
    """Return symmetric, global-phase-invariant patch statistics.

    No FFT is used.  Every patch-level quantity is reduced by mean, standard
    deviation, and fixed quantiles over the patch set, so patch order cannot
    affect the result.  Normalized correlation values are clipped to their
    mathematical ranges to prevent near-silent patches from harming covariance
    conditioning.
    """
    z = _packed_complex(packed, patch_count=patch_count)
    amplitude = np.abs(z)
    power = amplitude * amplitude
    columns: list[np.ndarray] = []

    def pool(value: np.ndarray, *, lower: float | None = None,
             upper: float | None = None) -> None:
        current = np.asarray(value, dtype=np.float64)
        if lower is not None or upper is not None:
            current = np.clip(current, lower, upper)
        columns.extend(
            (
                current.mean(axis=1),
                current.std(axis=1),
                np.quantile(current, 0.1, axis=1),
                np.quantile(current, 0.9, axis=1),
            )
        )

    patch_power = power.mean(axis=2)
    pool(amplitude.mean(axis=2), lower=0.0, upper=1e3)
    pool(amplitude.std(axis=2), lower=0.0, upper=1e3)
    pool(
        power.max(axis=2) / np.maximum(patch_power, _EPSILON),
        lower=0.0,
        upper=float(z.shape[2]),
    )
    pool(
        np.abs((z * z).mean(axis=2)) / np.maximum(patch_power, _EPSILON),
        lower=0.0,
        upper=1.0,
    )

    for lag in _LAGS:
        first, second = z[:, :, :-lag], z[:, :, lag:]
        denominator = np.sqrt(
            (np.abs(first) ** 2).mean(axis=2)
            * (np.abs(second) ** 2).mean(axis=2)
        )
        correlation = (first * np.conj(second)).mean(axis=2) / np.maximum(
            denominator, _EPSILON
        )
        correlation = np.where(
            np.abs(correlation) > 1.0,
            correlation / np.maximum(np.abs(correlation), 1.0),
            correlation,
        )
        pool(np.abs(correlation), lower=0.0, upper=1.0)
        pool(correlation.real, lower=-1.0, upper=1.0)
        pool(correlation.imag, lower=-1.0, upper=1.0)

        first_amp, second_amp = np.abs(first), np.abs(second)
        centered_first = first_amp - first_amp.mean(axis=2, keepdims=True)
        centered_second = second_amp - second_amp.mean(axis=2, keepdims=True)
        amplitude_correlation = (
            (centered_first * centered_second).mean(axis=2)
            / np.maximum(
                first_amp.std(axis=2) * second_amp.std(axis=2),
                _EPSILON,
            )
        )
        pool(amplitude_correlation, lower=-1.0, upper=1.0)

    increments = np.angle(z[:, :, 1:] * np.conj(z[:, :, :-1]))
    pool(
        np.abs(np.mean(np.exp(1j * increments), axis=2)),
        lower=0.0,
        upper=1.0,
    )
    increment_difference = np.angle(
        np.exp(1j * (increments[:, :, 1:] - increments[:, :, :-1]))
    )
    pool(
        np.abs(np.mean(np.exp(1j * increment_difference), axis=2)),
        lower=0.0,
        upper=1.0,
    )
    pool(np.mean(np.abs(increment_difference), axis=2), lower=0.0, upper=np.pi)

    patch_mean = z.mean(axis=2)
    columns.extend(
        (
            np.abs(patch_mean.mean(axis=1)),
            np.abs(patch_mean).std(axis=1),
            power.mean(axis=(1, 2)),
            power.std(axis=(1, 2)),
        )
    )
    result = np.stack(columns, axis=1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("patch statistics produced a non-finite value")
    return result


def _standardizer(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = reference.mean(axis=0)
    scale = reference.std(axis=0)
    scale = np.maximum(scale, 1e-5)
    return mean, scale


def _standardize(
    value: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return (value - mean) / scale


def _nearest_distance(
    queries: np.ndarray,
    reference: np.ndarray,
    *,
    chunk_size: int = 256,
) -> np.ndarray:
    reference_norm = np.sum(reference * reference, axis=1)
    output = []
    for start in range(0, len(queries), chunk_size):
        query = queries[start : start + chunk_size]
        distance = (
            np.sum(query * query, axis=1)[:, None]
            + reference_norm[None, :]
            - 2.0 * np.einsum("ni,mi->nm", query, reference)
        )
        output.append(np.maximum(distance.min(axis=1), 0.0))
    return np.concatenate(output)


def _nearest_neighbors(
    queries: np.ndarray,
    reference: np.ndarray,
    neighbors: int,
    *,
    exclude_self: bool = False,
    chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    if neighbors <= 0 or neighbors >= len(reference):
        raise ValueError("neighbors must lie in [1, reference rows)")
    if exclude_self and len(queries) != len(reference):
        raise ValueError("exclude_self requires queries and reference to align")
    reference_norm = np.sum(reference * reference, axis=1)
    all_indices, all_distances = [], []
    for start in range(0, len(queries), chunk_size):
        query = queries[start : start + chunk_size]
        distance = (
            np.sum(query * query, axis=1)[:, None]
            + reference_norm[None, :]
            - 2.0 * np.einsum("ni,mi->nm", query, reference)
        )
        distance = np.maximum(distance, 0.0)
        if exclude_self:
            row = np.arange(len(query))
            distance[row, row + start] = np.inf
        indices = np.argpartition(distance, neighbors - 1, axis=1)[
            :, :neighbors
        ]
        selected = np.take_along_axis(distance, indices, axis=1)
        order = np.argsort(selected, axis=1)
        all_indices.append(np.take_along_axis(indices, order, axis=1))
        all_distances.append(np.take_along_axis(selected, order, axis=1))
    return np.concatenate(all_indices), np.concatenate(all_distances)


def _empirical_rank(value: np.ndarray, calibration: np.ndarray) -> np.ndarray:
    ordered = np.sort(calibration)
    return np.searchsorted(ordered, value, side="left") / (len(ordered) + 1.0)


@dataclass(frozen=True)
class KnownOnlyLOFOpenSet:
    """Local-outlier-factor score fit on training and ranked on enrollment."""

    embedding_reference: np.ndarray
    embedding_mean: np.ndarray
    embedding_scale: np.ndarray
    reference_k_distance: np.ndarray
    reference_local_density: np.ndarray
    calibration: np.ndarray
    calibration_scores: np.ndarray
    neighbors: int

    @classmethod
    def fit(
        cls,
        training_embeddings: np.ndarray,
        enrollment_embeddings: np.ndarray,
        *,
        neighbors: int = 5,
    ) -> "KnownOnlyLOFOpenSet":
        """Fit LOF without selection rows, novelty examples, or class labels."""
        training = _finite_matrix(training_embeddings, "training_embeddings")
        enrollment = _finite_matrix(
            enrollment_embeddings, "enrollment_embeddings"
        )
        if training.shape[1] != enrollment.shape[1]:
            raise ValueError("training and enrollment embedding dimensions differ")
        if (
            isinstance(neighbors, (bool, np.bool_))
            or not isinstance(neighbors, (int, np.integer))
        ):
            raise ValueError("neighbors must be an integer")
        neighbors = int(neighbors)
        if neighbors <= 0 or neighbors >= len(training):
            raise ValueError("neighbors must lie in [1, training rows)")

        mean, scale = _standardizer(training)
        reference = _standardize(training, mean, scale)
        enrollment_z = _standardize(enrollment, mean, scale)
        reference_indices, reference_distances = _nearest_neighbors(
            reference,
            reference,
            neighbors,
            exclude_self=True,
        )
        k_distance = reference_distances[:, -1]
        reference_reachability = np.maximum(
            reference_distances,
            k_distance[reference_indices],
        )
        reference_density = 1.0 / (
            reference_reachability.mean(axis=1) + 1e-12
        )

        def raw_score(query: np.ndarray) -> np.ndarray:
            indices, distances = _nearest_neighbors(
                query, reference, neighbors
            )
            density = 1.0 / (
                np.maximum(distances, k_distance[indices]).mean(axis=1) + 1e-12
            )
            return reference_density[indices].mean(axis=1) / density

        calibration_raw = raw_score(enrollment_z)
        calibration_scores = _empirical_rank(
            calibration_raw, calibration_raw
        )
        return cls(
            reference,
            mean,
            scale,
            k_distance,
            reference_density,
            np.sort(calibration_raw),
            calibration_scores,
            neighbors,
        )

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Return enrollment-ranked LOF nonconformity in ``[0, 1)``."""
        embedding = _finite_matrix(embeddings, "embeddings")
        if embedding.shape[1] != self.embedding_mean.shape[0]:
            raise ValueError("embedding dimension does not match fitted model")
        standardized = _standardize(
            embedding, self.embedding_mean, self.embedding_scale
        )
        indices, distances = _nearest_neighbors(
            standardized,
            self.embedding_reference,
            self.neighbors,
        )
        density = 1.0 / (
            np.maximum(
                distances,
                self.reference_k_distance[indices],
            ).mean(axis=1)
            + 1e-12
        )
        raw = (
            self.reference_local_density[indices].mean(axis=1) / density
        )
        return _empirical_rank(raw, self.calibration)


@dataclass(frozen=True)
class KnownOnlyPatchOpenSet:
    """Known-only density score that cannot change closed class predictions."""

    embedding_reference: np.ndarray
    embedding_mean: np.ndarray
    embedding_scale: np.ndarray
    embedding_calibration: np.ndarray
    statistic_mean: np.ndarray
    statistic_scale: np.ndarray
    class_statistic_means: np.ndarray
    class_statistic_precisions: np.ndarray
    statistic_calibration: tuple[np.ndarray, ...]
    calibration_scores: np.ndarray
    patch_count: int
    shrinkage: float

    @classmethod
    def fit(
        cls,
        training_embeddings: np.ndarray,
        training_packed: np.ndarray,
        training_labels: np.ndarray,
        enrollment_embeddings: np.ndarray,
        enrollment_packed: np.ndarray,
        enrollment_labels: np.ndarray,
        *,
        n_classes: int | None = None,
        patch_count: int = 16,
        shrinkage: float = 1e-4,
    ) -> "KnownOnlyPatchOpenSet":
        """Fit references on training and empirical ranks on enrollment only."""
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("shrinkage must lie in [0, 1]")
        train_embedding = _finite_matrix(
            training_embeddings, "training_embeddings"
        )
        enroll_embedding = _finite_matrix(
            enrollment_embeddings, "enrollment_embeddings"
        )
        if train_embedding.shape[1] != enroll_embedding.shape[1]:
            raise ValueError("training and enrollment embedding dimensions differ")
        train_y = _labels(training_labels, len(train_embedding), "training_labels")
        enroll_y = _labels(
            enrollment_labels, len(enroll_embedding), "enrollment_labels"
        )
        inferred = int(max(train_y.max(), enroll_y.max())) + 1
        n_classes = inferred if n_classes is None else int(n_classes)
        if (
            n_classes <= 0
            or train_y.min() < 0
            or enroll_y.min() < 0
            or train_y.max() >= n_classes
            or enroll_y.max() >= n_classes
        ):
            raise ValueError("labels lie outside [0, n_classes)")

        train_statistic = patch_statistics(
            training_packed, patch_count=patch_count
        ).astype(np.float64)
        enroll_statistic = patch_statistics(
            enrollment_packed, patch_count=patch_count
        ).astype(np.float64)
        if len(train_statistic) != len(train_embedding):
            raise ValueError("training packed and embedding row counts differ")
        if len(enroll_statistic) != len(enroll_embedding):
            raise ValueError("enrollment packed and embedding row counts differ")

        embedding_mean, embedding_scale = _standardizer(train_embedding)
        train_embedding_z = _standardize(
            train_embedding, embedding_mean, embedding_scale
        )
        enroll_embedding_z = _standardize(
            enroll_embedding, embedding_mean, embedding_scale
        )
        embedding_calibration = _nearest_distance(
            enroll_embedding_z, train_embedding_z
        )

        statistic_mean, statistic_scale = _standardizer(train_statistic)
        train_statistic_z = _standardize(
            train_statistic, statistic_mean, statistic_scale
        )
        enroll_statistic_z = _standardize(
            enroll_statistic, statistic_mean, statistic_scale
        )

        means, precisions, calibration = [], [], []
        for class_index in range(n_classes):
            class_train = train_statistic_z[train_y == class_index]
            class_enroll = enroll_statistic_z[enroll_y == class_index]
            if not len(class_train) or not len(class_enroll):
                raise ValueError(
                    f"class {class_index} is absent from training or enrollment"
                )
            mean = class_train.mean(axis=0)
            centered = class_train - mean
            covariance = np.einsum(
                "ni,nj->ij", centered, centered
            ) / max(len(centered) - 1, 1)
            diagonal = np.diag(np.diag(covariance))
            covariance = (
                (1.0 - shrinkage) * covariance + shrinkage * diagonal
            )
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            floor = max(float(eigenvalues[-1]) * 1e-6, 1e-6)
            precision = np.einsum(
                "ik,k,jk->ij",
                eigenvectors,
                1.0 / np.maximum(eigenvalues, floor),
                eigenvectors,
            )
            difference = class_enroll - mean
            distance = np.einsum(
                "ni,ij,nj->n", difference, precision, difference
            )
            means.append(mean)
            precisions.append(precision)
            calibration.append(np.sort(distance))

        mean_array = np.stack(means)
        precision_array = np.stack(precisions)
        statistic_calibration = tuple(calibration)
        enrollment_statistic_distance = np.empty(len(enroll_y))
        for class_index in range(n_classes):
            rows = enroll_y == class_index
            difference = enroll_statistic_z[rows] - mean_array[class_index]
            enrollment_statistic_distance[rows] = np.einsum(
                "ni,ij,nj->n",
                difference,
                precision_array[class_index],
                difference,
            )
        embedding_rank = _empirical_rank(
            embedding_calibration, embedding_calibration
        )
        statistic_rank = np.empty(len(enroll_y))
        for class_index in range(n_classes):
            rows = enroll_y == class_index
            statistic_rank[rows] = _empirical_rank(
                enrollment_statistic_distance[rows],
                statistic_calibration[class_index],
            )
        calibration_scores = np.sqrt(embedding_rank * statistic_rank)

        return cls(
            train_embedding_z,
            embedding_mean,
            embedding_scale,
            np.sort(embedding_calibration),
            statistic_mean,
            statistic_scale,
            mean_array,
            precision_array,
            statistic_calibration,
            calibration_scores,
            int(patch_count),
            float(shrinkage),
        )

    @property
    def n_classes(self) -> int:
        return int(self.class_statistic_means.shape[0])

    def component_scores(
        self,
        embeddings: np.ndarray,
        packed: np.ndarray,
        predicted_classes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return enrollment-calibrated embedding and statistic ranks."""
        embedding = _finite_matrix(embeddings, "embeddings")
        predicted = _labels(
            predicted_classes, len(embedding), "predicted_classes"
        )
        if predicted.min() < 0 or predicted.max() >= self.n_classes:
            raise ValueError("predicted_classes lie outside fitted classes")
        if embedding.shape[1] != self.embedding_mean.shape[0]:
            raise ValueError("embedding dimension does not match fitted model")
        statistic = patch_statistics(
            packed, patch_count=self.patch_count
        ).astype(np.float64)
        if len(statistic) != len(embedding):
            raise ValueError("packed and embedding row counts differ")

        embedding_z = _standardize(
            embedding, self.embedding_mean, self.embedding_scale
        )
        embedding_distance = _nearest_distance(
            embedding_z, self.embedding_reference
        )
        embedding_rank = _empirical_rank(
            embedding_distance, self.embedding_calibration
        )

        statistic_z = _standardize(
            statistic, self.statistic_mean, self.statistic_scale
        )
        statistic_distance = np.empty(len(statistic_z))
        statistic_rank = np.empty(len(statistic_z))
        for class_index in range(self.n_classes):
            rows = predicted == class_index
            difference = (
                statistic_z[rows] - self.class_statistic_means[class_index]
            )
            statistic_distance[rows] = np.einsum(
                "ni,ij,nj->n",
                difference,
                self.class_statistic_precisions[class_index],
                difference,
            )
            statistic_rank[rows] = _empirical_rank(
                statistic_distance[rows],
                self.statistic_calibration[class_index],
            )
        return embedding_rank, statistic_rank

    def score(
        self,
        embeddings: np.ndarray,
        packed: np.ndarray,
        predicted_classes: np.ndarray,
    ) -> np.ndarray:
        """Return fused nonconformity in ``[0, 1)``; higher is more unknown."""
        embedding_rank, statistic_rank = self.component_scores(
            embeddings, packed, predicted_classes
        )
        return np.sqrt(embedding_rank * statistic_rank)


__all__ = [
    "KnownOnlyLOFOpenSet",
    "KnownOnlyPatchOpenSet",
    "patch_statistics",
]
