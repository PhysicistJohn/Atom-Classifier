"""Known-only open-set score for an existing prototype embedding model.

No novelty examples are used. ``ClassRadiusOpenSet`` keeps the deployed
nearest-prototype class decision unchanged and replaces the raw nearest
distance used for rejection with a class-conditional empirical radius:

1. build the ordinary class prototypes from enrollment embeddings;
2. measure an independent known calibration population against its true-class
   prototype; and
3. rank a query's predicted-class distance in that class's calibration
   distribution.

The returned value is conformal-style nonconformity in ``[0, 1)``: larger
means less like known calibration captures. Classes with naturally broad
embedding clouds therefore no longer dominate the global unknown threshold.
The calibration population must contain known captures only and should be
disjoint from enrollment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _embeddings(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError(f"{name} must be a non-empty [rows, dimensions] array")
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


def _squared_distances(
    embeddings: np.ndarray,
    prototypes: np.ndarray,
) -> np.ndarray:
    # This is the same expression and float32 coordinate system used by
    # train.nearest, so scoring cannot silently change the closed prediction.
    difference = embeddings[:, None, :] - prototypes[None, :, :]
    return np.sum(difference * difference, axis=-1)


@dataclass(frozen=True)
class ClassRadiusOpenSet:
    """Nearest-prototype classifier with a known-only conditional radius score."""

    prototypes: np.ndarray
    calibration_distances: tuple[np.ndarray, ...]

    @classmethod
    def fit(
        cls,
        enrollment_embeddings: np.ndarray,
        enrollment_labels: np.ndarray,
        calibration_embeddings: np.ndarray,
        calibration_labels: np.ndarray,
        *,
        n_classes: int | None = None,
    ) -> "ClassRadiusOpenSet":
        """Fit prototypes and per-class radii without novelty examples.

        ``calibration_embeddings`` should be the training population or another
        known-only population independent of enrollment. Every class must occur
        in both populations.
        """
        enrollment = _embeddings(enrollment_embeddings, "enrollment_embeddings")
        calibration = _embeddings(calibration_embeddings, "calibration_embeddings")
        if enrollment.shape[1] != calibration.shape[1]:
            raise ValueError("enrollment and calibration embedding dimensions differ")
        enrollment_y = _labels(
            enrollment_labels, len(enrollment), "enrollment_labels"
        )
        calibration_y = _labels(
            calibration_labels, len(calibration), "calibration_labels"
        )

        inferred = int(max(enrollment_y.max(), calibration_y.max())) + 1
        n_classes = inferred if n_classes is None else int(n_classes)
        if n_classes <= 0:
            raise ValueError("n_classes must be positive")
        if (
            enrollment_y.min() < 0
            or calibration_y.min() < 0
            or enrollment_y.max() >= n_classes
            or calibration_y.max() >= n_classes
        ):
            raise ValueError("labels lie outside [0, n_classes)")

        prototypes = []
        radii = []
        for class_index in range(n_classes):
            enrolled = enrollment[enrollment_y == class_index]
            calibrated = calibration[calibration_y == class_index]
            if not len(enrolled) or not len(calibrated):
                raise ValueError(
                    f"class {class_index} is absent from enrollment or calibration"
                )
            prototype = enrolled.mean(axis=0, dtype=np.float32)
            distance = np.sum(
                (calibrated - prototype[None, :]) ** 2,
                axis=1,
            )
            prototypes.append(prototype)
            radii.append(np.sort(distance.astype(np.float32, copy=False)))

        prototype_array = np.stack(prototypes).astype(np.float32, copy=False)
        prototype_array.setflags(write=False)
        frozen_radii = []
        for radius in radii:
            radius.setflags(write=False)
            frozen_radii.append(radius)
        return cls(prototype_array, tuple(frozen_radii))

    @property
    def n_classes(self) -> int:
        return int(self.prototypes.shape[0])

    @property
    def embedding_dim(self) -> int:
        return int(self.prototypes.shape[1])

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Return the unchanged nearest-enrollment-prototype class decision."""
        values = _embeddings(embeddings, "embeddings")
        if values.shape[1] != self.embedding_dim:
            raise ValueError("embedding dimension does not match fitted prototypes")
        return _squared_distances(values, self.prototypes).argmin(axis=1)

    def score(
        self,
        embeddings: np.ndarray,
        predicted_classes: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return class-conditional nonconformity; higher is more unknown.

        Passing ``predicted_classes`` lets a caller prove that an existing
        authoritative class decision is retained. If omitted, the same ordinary
        nearest-prototype decision is computed internally.
        """
        values = _embeddings(embeddings, "embeddings")
        if values.shape[1] != self.embedding_dim:
            raise ValueError("embedding dimension does not match fitted prototypes")
        distances = _squared_distances(values, self.prototypes)
        if predicted_classes is None:
            predicted = distances.argmin(axis=1)
        else:
            predicted = _labels(
                predicted_classes, len(values), "predicted_classes"
            )
            if predicted.min() < 0 or predicted.max() >= self.n_classes:
                raise ValueError("predicted_classes lie outside fitted classes")

        selected = distances[np.arange(len(values)), predicted]
        scores = np.empty(len(values), dtype=np.float64)
        for class_index, calibration in enumerate(self.calibration_distances):
            rows = predicted == class_index
            # Strictly smaller calibration distances implement 1-p where
            # p=(1 + #{calibration >= query})/(n+1). Ties stay conservative.
            count = np.searchsorted(
                calibration,
                selected[rows],
                side="left",
            )
            scores[rows] = count / (len(calibration) + 1.0)
        return scores

    def predict_and_score(
        self,
        embeddings: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        predicted = self.predict(embeddings)
        return predicted, self.score(embeddings, predicted)


def known_quantile_threshold(
    known_scores: np.ndarray,
    *,
    false_unknown_rate: float = 0.05,
) -> float:
    """Fit an operating threshold from a separate known-only population."""
    values = np.asarray(known_scores, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("known_scores must be a non-empty finite vector")
    if not 0.0 < false_unknown_rate < 1.0:
        raise ValueError("false_unknown_rate must lie strictly between zero and one")
    return float(
        np.quantile(
            values,
            1.0 - false_unknown_rate,
            method="higher",
        )
    )


__all__ = ["ClassRadiusOpenSet", "known_quantile_threshold"]
