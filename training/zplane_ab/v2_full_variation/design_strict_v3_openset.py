"""Bounded development-only open-set design for the strict v3 frontend.

This file is isolated from the failed v2-pose open-set policy and from every
release artifact.  It uses the frozen strict real/complex checkpoints and the
FFT-free data prepared by :func:`v3_scale.run_time_domain_dev._prepare_data`.

Leakage contract
----------------

* training only: branch centers, LOF density, class geometry;
* enrollment only: prototypes, LOF ranks, geometry ranks, final rank/cutoff;
* development seed 20260938 only: bounded policy selection;
* development seeds 20260941/20260942: one-shot replication after serialization.

Every novelty row is generated once at N32768 and N4096/N8192/N16384 are exact
prefixes.  Novelty never fits a center, density, class statistic, empirical
rank, or threshold.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
V3_SCALE = HERE / "v3_scale"
TRAINING = HERE.parent.parent
REPO = TRAINING.parent
for path in (TRAINING, HERE.parent, HERE, V3_SCALE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from assemble_invariant_candidate import (  # noqa: E402
    _embed_all,
    _fuse_numpy,
    _reload_lof_ensemble,
    _save_lof_ensemble,
    _score_lof_ensemble,
)
import invariant_patch_data  # noqa: E402
from invariant_patch_cnn import (  # noqa: E402
    InvariantPatchCNN,
    InvariantPatchConfig,
)
from known_only_patch_openset import KnownOnlyLOFOpenSet  # noqa: E402
from openset_eval import NOVELTY  # noqa: E402
from run_time_domain_dev import (  # noqa: E402
    _load_manifest,
    _prepare_data,
)
from train import auroc, nearest, prototypes_from  # noqa: E402
import time_domain_invariant_patch_preprocess as strict_preprocess  # noqa: E402
from v3_time_domain_openset import (  # noqa: E402
    KnownOnlyGeometry,
    aggregate_deviation,
    empirical_rank,
)


SCHEMA = 1
POLICY_KIND = "strict_v3_known_only_lof_geometry_rank_blend"
MODEL_SEED = 20260730
DESIGN_NOVELTY_SEED = 20260938
REPLICATION_NOVELTY_SEEDS = (20260941, 20260942)
CAPTURE_LENGTHS = (4096, 8192, 16384, 32768)
NOVELTY_FAMILIES = ("noise", "chirp")
NOVELTY_N_EACH = 300
PATCH_LENGTH = 64
PATCH_COUNT = 16
TARGET_FRAC = 0.5

ALPHA_REAL = 0.2
ALPHA_COMPLEX = 0.2
FUSION_WEIGHT_REAL = 0.6
LOF_REAL_NEIGHBORS = 2
LOF_COMPLEX_NEIGHBORS = 64
LOF_WEIGHT_REAL = 0.4
LOF_WEIGHT_COMPLEX = 0.6

# Both score components remain strictly positive.  This grid and order are
# frozen before any replication seed is generated.
BASE_WEIGHT_GRID = (0.25, 0.50, 0.75, 0.90)
# q95 is mandatory when any q95 candidate clears every design gate.  A lower
# quantile can be selected only when every earlier quantile lacks a full pass
# and immutable selection FUR remains within the predeclared 0.10 budget.
THRESHOLD_QUANTILE_SEQUENCE = (0.95, 0.925, 0.90)

GATES = {
    "overall_auroc": 0.72,
    "noise_auroc": 0.80,
    "chirp_auroc": 0.80,
    "noise_threshold_recall": 0.10,
    "chirp_threshold_recall": 0.10,
}
KNOWN_FUR_CEILING = 0.10

REAL_SOURCE = (
    HERE
    / "artifacts"
    / "invariant_patch"
    / "v3_scale"
    / "timecorr_real_4000_seed20260730_v3a"
)
COMPLEX_SOURCE = (
    HERE
    / "artifacts"
    / "invariant_patch"
    / "v3_scale"
    / "timecorr_complex_4000_seed20260730_run1"
)
DEFAULT_OUTPUT = (
    HERE
    / "artifacts"
    / "invariant_patch"
    / "strict_v3_openset_design_seed20260938_replication_20260941_20260942"
)


@dataclass(frozen=True)
class GeometrySpec:
    name: str
    feature_names: tuple[str, ...]
    reducer: str


INDIVIDUAL_FEATURES = (
    "amplitude_cv_mean",
    "amplitude_cv_q90",
    "amplitude_kurtosis_mean",
    "amplitude_kurtosis_q90",
    "peak_to_rms_mean",
    "peak_to_rms_q90",
    "pseudocovariance_mean",
    "coherence_lag1_mean",
    "coherence_lag1_q10",
    "coherence_lag2_mean",
    "coherence_lag2_q10",
    "coherence_lag4_mean",
    "coherence_lag4_q10",
    "coherence_lag8_mean",
    "coherence_lag8_q10",
    "coherence_lag16_mean",
    "coherence_lag16_q10",
    "coherence_lag32_mean",
    "coherence_lag32_q10",
    "increment_coherence_mean",
    "increment_coherence_q10",
    "across_patch_frequency_dispersion",
    "curvature_coherence_mean",
    "curvature_coherence_q10",
    "curvature_magnitude_mean",
    "curvature_magnitude_q90",
    "frequency_slope_magnitude_mean",
    "frequency_slope_magnitude_q90",
    "frequency_linearity_mean",
    "frequency_linearity_q90",
    "frequency_residual_mean",
    "frequency_residual_q90",
)

AMPLITUDE_GROUP = (
    "amplitude_cv_mean",
    "amplitude_cv_q90",
    "amplitude_kurtosis_mean",
    "amplitude_kurtosis_q90",
    "peak_to_rms_mean",
    "peak_to_rms_q90",
    "pseudocovariance_mean",
)
SHORT_COHERENCE_GROUP = (
    "coherence_lag1_mean",
    "coherence_lag1_q10",
    "coherence_lag2_mean",
    "coherence_lag2_q10",
    "coherence_lag4_mean",
    "coherence_lag4_q10",
    "increment_coherence_mean",
    "increment_coherence_q10",
)
LONG_COHERENCE_GROUP = (
    "coherence_lag8_mean",
    "coherence_lag8_q10",
    "coherence_lag16_mean",
    "coherence_lag16_q10",
    "coherence_lag32_mean",
    "coherence_lag32_q10",
)
CHIRP_DYNAMICS_GROUP = (
    "across_patch_frequency_dispersion",
    "curvature_coherence_mean",
    "curvature_coherence_q10",
    "curvature_magnitude_mean",
    "curvature_magnitude_q90",
    "frequency_slope_magnitude_mean",
    "frequency_slope_magnitude_q90",
    "frequency_linearity_mean",
    "frequency_linearity_q90",
    "frequency_residual_mean",
    "frequency_residual_q90",
)

GEOMETRY_SPECS = tuple(
    GeometrySpec(f"scalar:{name}", (name,), "max")
    for name in INDIVIDUAL_FEATURES
) + tuple(
    GeometrySpec(f"group:{name}:{reducer}", features, reducer)
    for name, features in (
        ("amplitude", AMPLITUDE_GROUP),
        ("short_coherence", SHORT_COHERENCE_GROUP),
        ("long_coherence", LONG_COHERENCE_GROUP),
        ("chirp_dynamics", CHIRP_DYNAMICS_GROUP),
    )
    for reducer in ("max", "rms")
)


@dataclass
class NoveltyFixture:
    packed: np.ndarray
    features: np.ndarray


@dataclass
class ScoredFixture:
    real_embedding: np.ndarray
    complex_embedding: np.ndarray
    fused_embedding: np.ndarray
    prediction: np.ndarray
    base_score: np.ndarray
    deviations: np.ndarray


@dataclass(frozen=True)
class StrictV3OpenSetPolicy:
    """One serialized, known-only calibrated strict-v3 policy."""

    geometry: KnownOnlyGeometry
    spec: GeometrySpec
    feature_indices: np.ndarray
    base_weight: float
    geometry_calibration_raw: np.ndarray
    combined_calibration_raw: np.ndarray
    calibration_scores: np.ndarray
    threshold_quantile: float
    threshold: float

    @classmethod
    def calibrate(
        cls,
        geometry: KnownOnlyGeometry,
        spec: GeometrySpec,
        enrollment_deviations: np.ndarray,
        enrollment_base_score: np.ndarray,
        *,
        base_weight: float,
        threshold_quantile: float,
    ) -> "StrictV3OpenSetPolicy":
        """Calibrate ranks/cutoff using enrollment arrays only."""
        if base_weight not in BASE_WEIGHT_GRID:
            raise ValueError("base_weight is outside the frozen grid")
        if threshold_quantile not in THRESHOLD_QUANTILE_SEQUENCE:
            raise ValueError("threshold_quantile is outside the frozen sequence")
        indices = geometry.indices(spec.feature_names)
        raw_geometry = aggregate_deviation(
            enrollment_deviations,
            indices,
            reducer=spec.reducer,
        )
        geometry_calibration = np.sort(_finite_vector(raw_geometry, "geometry"))
        geometry_rank = empirical_rank(raw_geometry, geometry_calibration)
        base = _rank_vector(enrollment_base_score, "enrollment_base_score")
        if len(base) != len(geometry_rank):
            raise ValueError("enrollment base/geometry row counts differ")
        combined_raw = (
            base_weight * base + (1.0 - base_weight) * geometry_rank
        )
        combined_calibration = np.sort(combined_raw)
        calibration_scores = empirical_rank(
            combined_raw,
            combined_calibration,
        )
        threshold = float(
            np.quantile(calibration_scores, threshold_quantile)
        )
        return cls(
            geometry=geometry,
            spec=spec,
            feature_indices=indices,
            base_weight=float(base_weight),
            geometry_calibration_raw=geometry_calibration,
            combined_calibration_raw=combined_calibration,
            calibration_scores=calibration_scores,
            threshold_quantile=float(threshold_quantile),
            threshold=threshold,
        )

    def score_from_deviations(
        self,
        base_score: np.ndarray,
        deviations: np.ndarray,
    ) -> np.ndarray:
        base = _rank_vector(base_score, "base_score")
        raw_geometry = aggregate_deviation(
            deviations,
            self.feature_indices,
            reducer=self.spec.reducer,
        )
        if len(base) != len(raw_geometry):
            raise ValueError("base/geometry row counts differ")
        geometry_rank = empirical_rank(
            raw_geometry,
            self.geometry_calibration_raw,
        )
        combined_raw = (
            self.base_weight * base
            + (1.0 - self.base_weight) * geometry_rank
        )
        return empirical_rank(combined_raw, self.combined_calibration_raw)

    def score(
        self,
        base_score: np.ndarray,
        packed: np.ndarray,
        predicted_classes: np.ndarray,
    ) -> np.ndarray:
        deviations = self.geometry.deviations(
            packed,
            predicted_classes,
        )
        return self.score_from_deviations(base_score, deviations)

    def to_payload(self) -> dict[str, np.ndarray]:
        return {
            "schema": np.asarray(SCHEMA, dtype=np.int64),
            "kind": np.asarray(POLICY_KIND),
            "spec_name": np.asarray(self.spec.name),
            "spec_feature_names": np.asarray(self.spec.feature_names),
            "spec_reducer": np.asarray(self.spec.reducer),
            "base_weight": np.asarray(self.base_weight, dtype=np.float64),
            "geometry_weight": np.asarray(
                1.0 - self.base_weight,
                dtype=np.float64,
            ),
            "threshold_quantile": np.asarray(
                self.threshold_quantile,
                dtype=np.float64,
            ),
            "threshold": np.asarray(self.threshold, dtype=np.float64),
            "geometry_feature_indices": np.asarray(
                self.feature_indices,
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
        payload: Mapping[str, np.ndarray],
    ) -> "StrictV3OpenSetPolicy":
        required = {
            "schema",
            "kind",
            "spec_name",
            "spec_feature_names",
            "spec_reducer",
            "base_weight",
            "geometry_weight",
            "threshold_quantile",
            "threshold",
            "geometry_feature_indices",
            "feature_names",
            "class_mean",
            "class_scale",
            "geometry_calibration_raw",
            "combined_calibration_raw",
            "calibration_scores",
        }
        if set(payload) != required:
            raise ValueError("serialized strict policy keys differ")
        if int(_scalar(payload["schema"])) != SCHEMA:
            raise ValueError("serialized strict policy schema differs")
        if str(_scalar(payload["kind"])) != POLICY_KIND:
            raise ValueError("serialized strict policy kind differs")
        spec_name = str(_scalar(payload["spec_name"]))
        try:
            canonical_spec = next(
                spec for spec in GEOMETRY_SPECS if spec.name == spec_name
            )
        except StopIteration as exc:
            raise ValueError("serialized geometry spec is outside grid") from exc
        serialized_features = tuple(
            str(value)
            for value in np.asarray(payload["spec_feature_names"]).tolist()
        )
        reducer = str(_scalar(payload["spec_reducer"]))
        if (
            serialized_features != canonical_spec.feature_names
            or reducer != canonical_spec.reducer
        ):
            raise ValueError("serialized geometry spec contract differs")
        base_weight = float(_scalar(payload["base_weight"]))
        geometry_weight = float(_scalar(payload["geometry_weight"]))
        quantile = float(_scalar(payload["threshold_quantile"]))
        if (
            base_weight not in BASE_WEIGHT_GRID
            or geometry_weight != 1.0 - base_weight
            or quantile not in THRESHOLD_QUANTILE_SEQUENCE
        ):
            raise ValueError("serialized blend/quantile is outside grid")

        feature_names = tuple(
            str(value)
            for value in np.asarray(payload["feature_names"]).tolist()
        )
        mean = np.asarray(payload["class_mean"], dtype=np.float64)
        scale = np.asarray(payload["class_scale"], dtype=np.float64)
        if (
            mean.ndim != 2
            or mean.shape != scale.shape
            or mean.shape[1] != len(feature_names)
            or not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or np.any(scale <= 0.0)
        ):
            raise ValueError("serialized class geometry is invalid")
        geometry = KnownOnlyGeometry(mean, scale, feature_names)
        indices = np.asarray(
            payload["geometry_feature_indices"],
            dtype=np.int64,
        )
        np.testing.assert_array_equal(
            indices,
            geometry.indices(canonical_spec.feature_names),
        )
        geometry_calibration = _sorted_vector(
            payload["geometry_calibration_raw"],
            "geometry_calibration_raw",
        )
        combined_calibration = _sorted_vector(
            payload["combined_calibration_raw"],
            "combined_calibration_raw",
        )
        calibration_scores = _rank_vector(
            payload["calibration_scores"],
            "calibration_scores",
        )
        if not (
            len(geometry_calibration)
            == len(combined_calibration)
            == len(calibration_scores)
        ):
            raise ValueError("serialized calibration lengths differ")
        threshold = float(_scalar(payload["threshold"]))
        expected = float(np.quantile(calibration_scores, quantile))
        if threshold != expected:
            raise ValueError("serialized threshold does not replay")
        return cls(
            geometry=geometry,
            spec=canonical_spec,
            feature_indices=indices,
            base_weight=base_weight,
            geometry_calibration_raw=geometry_calibration,
            combined_calibration_raw=combined_calibration,
            calibration_scores=calibration_scores,
            threshold_quantile=quantile,
            threshold=threshold,
        )


def _scalar(value: np.ndarray) -> Any:
    result = np.asarray(value)
    if result.ndim != 0:
        raise ValueError("serialized scalar has non-scalar shape")
    return result.item()


def _finite_vector(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a non-empty finite vector")
    return result


def _sorted_vector(value: np.ndarray, name: str) -> np.ndarray:
    result = _finite_vector(value, name)
    if np.any(result[1:] < result[:-1]):
        raise ValueError(f"{name} must be sorted")
    return result


def _rank_vector(value: np.ndarray, name: str) -> np.ndarray:
    result = _finite_vector(value, name)
    if np.any(result < 0.0) or np.any(result >= 1.0):
        raise ValueError(f"{name} must lie in [0, 1)")
    return result


def _assert_development_path(path: Path, name: str) -> Path:
    resolved = path.expanduser().resolve()
    lowered = tuple(part.lower() for part in resolved.parts)
    if "releases" in lowered or any("sealed" in part for part in lowered):
        raise ValueError(f"{name} refuses release or sealed paths")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            _jsonable(payload),
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ("git", "rev-parse", "HEAD"),
            cwd=REPO,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_state(path: Path) -> Mapping[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover
        return torch.load(path, map_location="cpu")


def _load_strict_models(
    device: torch.device,
) -> tuple[InvariantPatchCNN, InvariantPatchCNN, dict[str, Any]]:
    real_metrics_path = REAL_SOURCE / "dev_metrics.json"
    complex_metrics_path = COMPLEX_SOURCE / "dev_metrics.json"
    real_metrics = _load_json(real_metrics_path)
    complex_metrics = _load_json(complex_metrics_path)
    for name, value in (("real", real_metrics), ("complex", complex_metrics)):
        if (
            value.get("status") != "complete"
            or value.get("development_only") is not True
            or value.get("release_evidence") is not False
            or value.get("sealed_release_data_used") != 0
            or value.get("consumed_test_rows_used") != 0
        ):
            raise ValueError(f"{name} strict artifact provenance is invalid")
        frontend = value.get("frontend", {})
        if (
            frontend.get("version") != "invariant-patch-time-domain-v1"
            or frontend.get("uses_frequency_transform") is not False
        ):
            raise ValueError(f"{name} strict artifact frontend is invalid")

    real = InvariantPatchCNN(
        InvariantPatchConfig(**dict(real_metrics["architecture"]))
    )
    complex_model = InvariantPatchCNN(
        InvariantPatchConfig(**dict(complex_metrics["architecture"]))
    )
    real_state_path = REAL_SOURCE / real_metrics["artifacts"]["state_dict"]
    complex_state_path = (
        COMPLEX_SOURCE / complex_metrics["artifacts"]["state_dict"]
    )
    if (
        _sha256(real_state_path)
        != real_metrics["artifacts"]["state_dict_sha256"]
        or _sha256(complex_state_path)
        != complex_metrics["artifacts"]["state_dict_sha256"]
    ):
        raise ValueError("strict checkpoint hash mismatch")
    real.load_state_dict(_load_state(real_state_path), strict=True)
    complex_model.load_state_dict(
        _load_state(complex_state_path),
        strict=True,
    )
    provenance = {
        "real": {
            "directory": str(REAL_SOURCE),
            "dev_metrics_sha256": _sha256(real_metrics_path),
            "state_dict_sha256": _sha256(real_state_path),
        },
        "complex": {
            "directory": str(COMPLEX_SOURCE),
            "dev_metrics_sha256": _sha256(complex_metrics_path),
            "state_dict_sha256": _sha256(complex_state_path),
        },
    }
    return real.to(device).eval(), complex_model.to(device).eval(), provenance


def _prepare_strict_data() -> tuple[
    dict[str, Any],
    dict[str, list[dict[str, Any]]],
    dict[str, Any],
]:
    source = invariant_patch_data.load(
        patch_length=PATCH_LENGTH,
        patch_count=PATCH_COUNT,
        target_frac=TARGET_FRAC,
        model_seed=MODEL_SEED,
        build_if_missing=False,
    )
    if source["data_audit"].get("consumed_test_rows_exposed") != 0:
        raise RuntimeError("data loader exposed consumed test rows")
    forbidden = [name for name in source if "test" in name.lower()]
    if forbidden:
        raise RuntimeError(f"data loader exposed test-like keys: {forbidden}")
    return _prepare_data(
        source,
        _load_manifest(),
        patch_length=PATCH_LENGTH,
        patch_count=PATCH_COUNT,
        target_frac=TARGET_FRAC,
        train_lengths=None,
    )


def _embed_splits(
    real: InvariantPatchCNN,
    complex_model: InvariantPatchCNN,
    data: Mapping[str, Any],
    device: torch.device,
) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {"real": {}, "complex": {}}
    for branch, model in (("real", real), ("complex", complex_model)):
        for split in ("tr", "en", "va"):
            result[branch][split] = _embed_all(
                model,
                data[f"x{split}"],
                data[f"f{split}"],
                device,
                batch=256,
            )
        print(f"[strict-open] embedded {branch} splits", flush=True)
    return result


def _fit_known_only(
    real: InvariantPatchCNN,
    complex_model: InvariantPatchCNN,
    data: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    embedding = _embed_splits(real, complex_model, data, device)
    real_center = embedding["real"]["tr"].mean(axis=0).astype(np.float32)
    complex_center = embedding["complex"]["tr"].mean(axis=0).astype(
        np.float32
    )
    embedding["fusion"] = {
        split: _fuse_numpy(
            embedding["real"][split],
            embedding["complex"][split],
            real_center,
            complex_center,
        )
        for split in ("tr", "en", "va")
    }
    prototypes = prototypes_from(
        embedding["fusion"]["en"],
        np.asarray(data["yen"], dtype=np.int64),
        int(data["n_classes"]),
    )
    prediction = {
        split: nearest(embedding["fusion"][split], prototypes)[0]
        for split in ("en", "va")
    }
    components = [
        (
            "real",
            LOF_WEIGHT_REAL,
            KnownOnlyLOFOpenSet.fit(
                embedding["real"]["tr"],
                embedding["real"]["en"],
                neighbors=LOF_REAL_NEIGHBORS,
            ),
        ),
        (
            "complex",
            LOF_WEIGHT_COMPLEX,
            KnownOnlyLOFOpenSet.fit(
                embedding["complex"]["tr"],
                embedding["complex"]["en"],
                neighbors=LOF_COMPLEX_NEIGHBORS,
            ),
        ),
    ]
    base_score = {
        split: _score_lof_ensemble(
            components,
            {
                branch: embedding[branch][split]
                for branch in ("real", "complex", "fusion")
            },
        )
        for split in ("en", "va")
    }
    geometry = KnownOnlyGeometry.fit(
        data["xtr"],
        data["ytr"],
        n_classes=int(data["n_classes"]),
    )
    deviation = {
        split: geometry.deviations(
            data[f"x{split}"],
            prediction[split],
        )
        for split in ("en", "va")
    }
    closed_accuracy = float(
        np.mean(prediction["va"] == np.asarray(data["yva"]))
    )
    return {
        "embedding": embedding,
        "real_center": real_center,
        "complex_center": complex_center,
        "prototypes": prototypes,
        "prediction": prediction,
        "lof_components": components,
        "base_score": base_score,
        "geometry": geometry,
        "deviation": deviation,
        "closed_accuracy": closed_accuracy,
    }


def _update_complex_digest(digest: Any, value: np.ndarray) -> None:
    canonical = np.ascontiguousarray(
        np.asarray(value, dtype=np.complex128)
    )
    digest.update(canonical.view(np.uint8))


def _build_novelty(
    seed: int,
    data: Mapping[str, Any],
    *,
    n_each: int,
) -> tuple[dict[str, dict[int, NoveltyFixture]], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    maximum = max(CAPTURE_LENGTHS)
    fmean = np.asarray(data["fmean"], dtype=np.float32)
    fstd = np.asarray(data["fstd"], dtype=np.float32)
    output: dict[str, dict[int, NoveltyFixture]] = {}
    provenance: dict[str, Any] = {}
    print(f"[strict-open] generating novelty seed {seed}", flush=True)
    for family in NOVELTY_FAMILIES:
        packed_rows: dict[int, list[np.ndarray]] = {
            length: [] for length in CAPTURE_LENGTHS
        }
        feature_rows: dict[int, list[np.ndarray]] = {
            length: [] for length in CAPTURE_LENGTHS
        }
        maximum_digest = hashlib.sha256()
        prefix_digests = {
            length: hashlib.sha256() for length in CAPTURE_LENGTHS
        }
        for row in range(n_each):
            raw = NOVELTY[family](maximum, rng)
            _update_complex_digest(maximum_digest, raw)
            for length in CAPTURE_LENGTHS:
                prefix = raw[:length]
                _update_complex_digest(prefix_digests[length], prefix)
                packed, raw_features, context = strict_preprocess.preprocess(
                    prefix,
                    patch_length=PATCH_LENGTH,
                    patch_count=PATCH_COUNT,
                    target_frac=TARGET_FRAC,
                )
                if context.get("uses_frequency_transform") is not False:
                    raise RuntimeError("strict novelty preprocessing used FFT")
                packed_rows[length].append(packed)
                feature_rows[length].append(raw_features)
            if (row + 1) % 100 == 0:
                print(f"  {family}: {row + 1}/{n_each}", flush=True)
        output[family] = {}
        for length in CAPTURE_LENGTHS:
            packed = np.stack(packed_rows[length]).astype(np.float32)
            raw_features = np.stack(feature_rows[length]).astype(np.float32)
            output[family][length] = NoveltyFixture(
                packed=packed,
                features=((raw_features - fmean) / fstd).astype(np.float32),
            )
        provenance[family] = {
            "n": n_each,
            "maximum_sha256": maximum_digest.hexdigest(),
            "prefix_sha256": {
                str(length): prefix_digests[length].hexdigest()
                for length in CAPTURE_LENGTHS
            },
        }
    return output, provenance


def _score_fixture(
    fixture: NoveltyFixture,
    *,
    real: InvariantPatchCNN,
    complex_model: InvariantPatchCNN,
    real_center: np.ndarray,
    complex_center: np.ndarray,
    prototypes: np.ndarray,
    lof_components: list[tuple[str, float, KnownOnlyLOFOpenSet]],
    geometry: KnownOnlyGeometry,
    device: torch.device,
) -> ScoredFixture:
    real_embedding = _embed_all(
        real,
        fixture.packed,
        fixture.features,
        device,
        batch=256,
    )
    complex_embedding = _embed_all(
        complex_model,
        fixture.packed,
        fixture.features,
        device,
        batch=256,
    )
    fused = _fuse_numpy(
        real_embedding,
        complex_embedding,
        real_center,
        complex_center,
    )
    prediction = nearest(fused, prototypes)[0]
    base = _score_lof_ensemble(
        lof_components,
        {
            "real": real_embedding,
            "complex": complex_embedding,
            "fusion": fused,
        },
    )
    deviations = geometry.deviations(fixture.packed, prediction)
    return ScoredFixture(
        real_embedding,
        complex_embedding,
        fused,
        prediction,
        base,
        deviations,
    )


def _score_fixture_set(
    fixtures: Mapping[str, Mapping[int, NoveltyFixture]],
    **kwargs: Any,
) -> dict[str, dict[int, ScoredFixture]]:
    return {
        family: {
            length: _score_fixture(fixture, **kwargs)
            for length, fixture in by_length.items()
        }
        for family, by_length in fixtures.items()
    }


def _metric(
    novel: np.ndarray,
    known: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    return {
        "auroc": float(auroc(novel, known)),
        "threshold_recall": float(np.mean(novel > threshold)),
        "median": float(np.median(novel)),
    }


def _evaluate_policy(
    policy: StrictV3OpenSetPolicy,
    *,
    known_base: np.ndarray,
    known_deviations: np.ndarray,
    novelty: Mapping[str, Mapping[int, ScoredFixture]],
) -> dict[str, Any]:
    known = policy.score_from_deviations(
        known_base,
        known_deviations,
    )
    rows: dict[str, Any] = {}
    for length in CAPTURE_LENGTHS:
        scores = {
            family: policy.score_from_deviations(
                novelty[family][length].base_score,
                novelty[family][length].deviations,
            )
            for family in NOVELTY_FAMILIES
        }
        row = {
            family: _metric(scores[family], known, policy.threshold)
            for family in NOVELTY_FAMILIES
        }
        row["overall"] = _metric(
            np.concatenate([scores[name] for name in NOVELTY_FAMILIES]),
            known,
            policy.threshold,
        )
        rows[str(length)] = row
    return {
        "known_fur": float(np.mean(known > policy.threshold)),
        "known_score_median": float(np.median(known)),
        "rows": rows,
    }


def _gate_values(report: Mapping[str, Any]) -> dict[str, float]:
    rows = report["rows"].values()
    return {
        "overall_auroc": min(row["overall"]["auroc"] for row in rows),
        "noise_auroc": min(row["noise"]["auroc"] for row in rows),
        "chirp_auroc": min(row["chirp"]["auroc"] for row in rows),
        "noise_threshold_recall": min(
            row["noise"]["threshold_recall"] for row in rows
        ),
        "chirp_threshold_recall": min(
            row["chirp"]["threshold_recall"] for row in rows
        ),
    }


def _selection_summary(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    values = _gate_values(report)
    gates = {
        name: {
            "value": float(values[name]),
            "floor": float(floor),
            "passes": bool(values[name] >= floor),
        }
        for name, floor in GATES.items()
    }
    gates["known_fur"] = {
        "value": float(report["known_fur"]),
        "ceiling": KNOWN_FUR_CEILING,
        "passes": bool(report["known_fur"] <= KNOWN_FUR_CEILING),
    }
    normalized = [
        values[name] / floor for name, floor in GATES.items()
    ] + [KNOWN_FUR_CEILING / max(float(report["known_fur"]), 1e-12)]
    return {
        "values": values,
        "gates": gates,
        "all_pass": bool(all(item["passes"] for item in gates.values())),
        "gate_count": int(sum(item["passes"] for item in gates.values())),
        "minimum_normalized_margin": float(min(normalized)),
        "mean_overall_auroc": float(
            np.mean(
                [row["overall"]["auroc"] for row in report["rows"].values()]
            )
        ),
    }


def _passing_objective(summary: Mapping[str, Any]) -> tuple[float, ...]:
    values = summary["values"]
    return (
        min(values["noise_auroc"], values["chirp_auroc"]),
        min(
            values["noise_threshold_recall"],
            values["chirp_threshold_recall"],
        ),
        values["overall_auroc"],
        summary["mean_overall_auroc"],
        -summary["gates"]["known_fur"]["value"],
    )


def _failure_objective(summary: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(summary["gate_count"]),
        summary["minimum_normalized_margin"],
        *_passing_objective(summary),
    )


def _choose_candidate(
    by_quantile: Mapping[float, Sequence[dict[str, Any]]],
) -> tuple[dict[str, Any], str]:
    """Apply q95-first preference to already evaluated candidates."""
    for quantile in THRESHOLD_QUANTILE_SEQUENCE:
        passing = [
            candidate
            for candidate in by_quantile[quantile]
            if candidate["summary"]["all_pass"]
        ]
        if passing:
            return (
                max(
                    passing,
                    key=lambda item: _passing_objective(item["summary"]),
                ),
                (
                    "first preferred quantile with an all-gate candidate: "
                    f"{quantile}"
                ),
            )
    return (
        max(
            by_quantile[THRESHOLD_QUANTILE_SEQUENCE[0]],
            key=lambda item: _failure_objective(item["summary"]),
        ),
        (
            "no quantile had an all-gate candidate; fail closed to the best "
            "q95 candidate under the predeclared normalized gate objective"
        ),
    )


def select_policy(
    geometry: KnownOnlyGeometry,
    enrollment_deviations: np.ndarray,
    enrollment_base: np.ndarray,
    known_deviations: np.ndarray,
    known_base: np.ndarray,
    novelty: Mapping[str, Mapping[int, ScoredFixture]],
) -> tuple[StrictV3OpenSetPolicy, dict[str, Any]]:
    """Evaluate the frozen grid and apply the predeclared quantile preference."""
    by_quantile: dict[float, list[dict[str, Any]]] = {
        quantile: [] for quantile in THRESHOLD_QUANTILE_SEQUENCE
    }
    evaluated = 0
    for quantile in THRESHOLD_QUANTILE_SEQUENCE:
        for spec in GEOMETRY_SPECS:
            for base_weight in BASE_WEIGHT_GRID:
                evaluated += 1
                policy = StrictV3OpenSetPolicy.calibrate(
                    geometry,
                    spec,
                    enrollment_deviations,
                    enrollment_base,
                    base_weight=base_weight,
                    threshold_quantile=quantile,
                )
                report = _evaluate_policy(
                    policy,
                    known_base=known_base,
                    known_deviations=known_deviations,
                    novelty=novelty,
                )
                summary = _selection_summary(report)
                by_quantile[quantile].append(
                    {
                        "policy": policy,
                        "report": report,
                        "summary": summary,
                        "candidate": {
                            "spec": spec.name,
                            "feature_names": list(spec.feature_names),
                            "reducer": spec.reducer,
                            "base_weight": base_weight,
                            "geometry_weight": 1.0 - base_weight,
                            "threshold_quantile": quantile,
                            "threshold": policy.threshold,
                        },
                    }
                )

    selected, selection_reason = _choose_candidate(by_quantile)

    diagnostics = []
    for quantile in THRESHOLD_QUANTILE_SEQUENCE:
        ordered = sorted(
            by_quantile[quantile],
            key=lambda item: (
                item["summary"]["all_pass"],
                _passing_objective(item["summary"])
                if item["summary"]["all_pass"]
                else _failure_objective(item["summary"]),
            ),
            reverse=True,
        )
        diagnostics.append(
            {
                "quantile": quantile,
                "all_gate_candidate_count": int(
                    sum(item["summary"]["all_pass"] for item in ordered)
                ),
                "top_candidates": [
                    {
                        **item["candidate"],
                        "summary": item["summary"],
                    }
                    for item in ordered[:10]
                ],
            }
        )
    return selected["policy"], {
        "selection_reason": selection_reason,
        "selected_candidate": selected["candidate"],
        "selected_design_report": selected["report"],
        "selected_summary": selected["summary"],
        "grid": {
            "geometry_specs": [
                {
                    "name": spec.name,
                    "feature_names": list(spec.feature_names),
                    "reducer": spec.reducer,
                }
                for spec in GEOMETRY_SPECS
            ],
            "base_weights": list(BASE_WEIGHT_GRID),
            "threshold_quantile_preference": list(
                THRESHOLD_QUANTILE_SEQUENCE
            ),
            "candidate_count": evaluated,
            "fit_or_calibration_uses_novelty": False,
        },
        "quantile_diagnostics": diagnostics,
    }


def _baseline_report(
    known_base: np.ndarray,
    enrollment_base: np.ndarray,
    novelty: Mapping[str, Mapping[int, ScoredFixture]],
) -> dict[str, Any]:
    threshold = float(np.quantile(enrollment_base, 0.95))
    rows = {}
    for length in CAPTURE_LENGTHS:
        family = {
            name: _metric(
                novelty[name][length].base_score,
                known_base,
                threshold,
            )
            for name in NOVELTY_FAMILIES
        }
        family["overall"] = _metric(
            np.concatenate(
                [
                    novelty[name][length].base_score
                    for name in NOVELTY_FAMILIES
                ]
            ),
            known_base,
            threshold,
        )
        rows[str(length)] = family
    return {
        "threshold": threshold,
        "known_fur": float(np.mean(known_base > threshold)),
        "rows": rows,
    }


def _save_classifier_state(path: Path, known: Mapping[str, Any], data: Mapping[str, Any]) -> None:
    np.savez_compressed(
        path,
        schema=np.asarray(SCHEMA, dtype=np.int64),
        kind=np.asarray("strict_v3_centered_fusion_state"),
        alpha_real=np.asarray(ALPHA_REAL, dtype=np.float64),
        alpha_complex=np.asarray(ALPHA_COMPLEX, dtype=np.float64),
        fusion_weight_real=np.asarray(FUSION_WEIGHT_REAL, dtype=np.float64),
        real_center=np.asarray(known["real_center"], dtype=np.float32),
        complex_center=np.asarray(known["complex_center"], dtype=np.float32),
        prototypes=np.asarray(known["prototypes"], dtype=np.float32),
        feature_mean=np.asarray(data["fmean"], dtype=np.float32),
        feature_std=np.asarray(data["fstd"], dtype=np.float32),
        classes=np.asarray(data["classes"]),
    )


def _load_policy(path: Path) -> StrictV3OpenSetPolicy:
    with np.load(path, allow_pickle=False) as stored:
        return StrictV3OpenSetPolicy.from_payload(
            {name: stored[name] for name in stored.files}
        )


def _serialization_replay(
    policy_path: Path,
    lof_path: Path,
    state_path: Path,
    policy: StrictV3OpenSetPolicy,
    known: Mapping[str, Any],
    design: Mapping[str, Mapping[int, ScoredFixture]],
) -> tuple[StrictV3OpenSetPolicy, list[tuple[str, float, KnownOnlyLOFOpenSet]], dict[str, Any]]:
    restored_policy = _load_policy(policy_path)
    restored_lof, restored_base_threshold = _reload_lof_ensemble(lof_path)
    with np.load(state_path, allow_pickle=False) as state:
        state_payload = {name: state[name] for name in state.files}
    for name, expected in (
        ("real_center", known["real_center"]),
        ("complex_center", known["complex_center"]),
        ("prototypes", known["prototypes"]),
    ):
        np.testing.assert_array_equal(state_payload[name], expected)

    base_known = _score_lof_ensemble(
        restored_lof,
        {
            branch: known["embedding"][branch]["va"]
            for branch in ("real", "complex", "fusion")
        },
    )
    np.testing.assert_array_equal(base_known, known["base_score"]["va"])
    original_known = policy.score_from_deviations(
        known["base_score"]["va"],
        known["deviation"]["va"],
    )
    replay_known = restored_policy.score_from_deviations(
        base_known,
        known["deviation"]["va"],
    )
    np.testing.assert_array_equal(replay_known, original_known)

    max_base_error = 0.0
    max_policy_error = 0.0
    for family in NOVELTY_FAMILIES:
        for length in CAPTURE_LENGTHS:
            scored = design[family][length]
            replay_base = _score_lof_ensemble(
                restored_lof,
                {
                    "real": scored.real_embedding,
                    "complex": scored.complex_embedding,
                    "fusion": scored.fused_embedding,
                },
            )
            max_base_error = max(
                max_base_error,
                float(np.max(np.abs(replay_base - scored.base_score))),
            )
            original = policy.score_from_deviations(
                scored.base_score,
                scored.deviations,
            )
            replay = restored_policy.score_from_deviations(
                replay_base,
                scored.deviations,
            )
            max_policy_error = max(
                max_policy_error,
                float(np.max(np.abs(replay - original))),
            )
    if max_base_error != 0.0 or max_policy_error != 0.0:
        raise RuntimeError("serialized strict open-set replay is not exact")
    return restored_policy, restored_lof, {
        "exact": True,
        "max_base_score_error": max_base_error,
        "max_policy_score_error": max_policy_error,
        "restored_base_threshold": restored_base_threshold,
        "policy_threshold": restored_policy.threshold,
    }


def _replication_gates(
    reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    values = {name: [] for name in GATES}
    fur = []
    for report in reports.values():
        fur.append(float(report["known_fur"]))
        current = _gate_values(report)
        for name in GATES:
            values[name].append(current[name])
    gates = {
        name: {
            "worst": float(min(values[name])),
            "floor": float(floor),
            "passes": bool(min(values[name]) >= floor),
        }
        for name, floor in GATES.items()
    }
    gates["known_fur"] = {
        "worst": float(max(fur)),
        "ceiling": KNOWN_FUR_CEILING,
        "passes": bool(max(fur) <= KNOWN_FUR_CEILING),
    }
    return {
        "gates": gates,
        "all_pass": bool(all(item["passes"] for item in gates.values())),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output = _assert_development_path(args.output_dir, "output")
    _assert_development_path(REAL_SOURCE, "real source")
    _assert_development_path(COMPLEX_SOURCE, "complex source")
    if args.device.lower() != "cpu":
        raise ValueError("strict open-set design is pinned to CPU")
    if args.novelty_n != NOVELTY_N_EACH:
        raise ValueError(f"novelty_n is frozen at {NOVELTY_N_EACH}")
    if tuple(args.replication_seeds) != REPLICATION_NOVELTY_SEEDS:
        raise ValueError(
            f"replication seeds are frozen at {REPLICATION_NOVELTY_SEEDS}"
        )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    data, _contexts, training_view_audit = _prepare_strict_data()
    real, complex_model, model_provenance = _load_strict_models(device)
    known = _fit_known_only(real, complex_model, data, device)
    print(
        f"[strict-open] fused closed accuracy={known['closed_accuracy']:.6f}",
        flush=True,
    )

    design_fixture, design_fixture_provenance = _build_novelty(
        DESIGN_NOVELTY_SEED,
        data,
        n_each=args.novelty_n,
    )
    score_kwargs = {
        "real": real,
        "complex_model": complex_model,
        "real_center": known["real_center"],
        "complex_center": known["complex_center"],
        "prototypes": known["prototypes"],
        "lof_components": known["lof_components"],
        "geometry": known["geometry"],
        "device": device,
    }
    design_scored = _score_fixture_set(design_fixture, **score_kwargs)
    baseline_design = _baseline_report(
        known["base_score"]["va"],
        known["base_score"]["en"],
        design_scored,
    )
    selected_policy, selection = select_policy(
        known["geometry"],
        known["deviation"]["en"],
        known["base_score"]["en"],
        known["deviation"]["va"],
        known["base_score"]["va"],
        design_scored,
    )
    print(
        "[strict-open] selected "
        f"{selected_policy.spec.name}, base={selected_policy.base_weight}, "
        f"q={selected_policy.threshold_quantile}, "
        f"design_pass={selection['selected_summary']['all_pass']}",
        flush=True,
    )

    policy_path = output / "strict_v3_open_policy.npz"
    lof_path = output / "strict_v3_lof_components.npz"
    classifier_path = output / "strict_v3_classifier_state.npz"
    np.savez_compressed(policy_path, **selected_policy.to_payload())
    base_threshold = float(np.quantile(known["base_score"]["en"], 0.95))
    _save_lof_ensemble(
        lof_path,
        known["lof_components"],
        base_threshold,
    )
    _save_classifier_state(classifier_path, known, data)

    frozen_manifest = {
        "status": "frozen_before_replication",
        "development_only": True,
        "release_evidence": False,
        "sealed_release_paths_read": 0,
        "consumed_test_rows_used": 0,
        "design_seed": DESIGN_NOVELTY_SEED,
        "replication_seeds_not_generated_at_freeze": list(
            args.replication_seeds
        ),
        "policy": selection["selected_candidate"],
        "selection_reason": selection["selection_reason"],
        "design_summary": selection["selected_summary"],
        "artifact_hashes": {
            policy_path.name: _sha256(policy_path),
            lof_path.name: _sha256(lof_path),
            classifier_path.name: _sha256(classifier_path),
        },
        "source_sha256": _sha256(Path(__file__).resolve()),
    }
    frozen_path = output / "DESIGN_FROZEN_BEFORE_REPLICATION.json"
    _atomic_json(frozen_path, frozen_manifest)

    restored_policy, restored_lof, replay = _serialization_replay(
        policy_path,
        lof_path,
        classifier_path,
        selected_policy,
        known,
        design_scored,
    )
    replay_path = output / "SERIALIZATION_REPLAY.json"
    _atomic_json(replay_path, replay)

    # Replication generation starts only after the design is frozen and every
    # serialized score exactly replays.
    replication_reports: dict[str, Any] = {}
    replication_fixture_provenance: dict[str, Any] = {}
    replication_baselines: dict[str, Any] = {}
    replay_score_kwargs = {
        **score_kwargs,
        "lof_components": restored_lof,
    }
    for seed in args.replication_seeds:
        fixtures, provenance = _build_novelty(
            int(seed),
            data,
            n_each=args.novelty_n,
        )
        scored = _score_fixture_set(fixtures, **replay_score_kwargs)
        report = _evaluate_policy(
            restored_policy,
            known_base=known["base_score"]["va"],
            known_deviations=known["deviation"]["va"],
            novelty=scored,
        )
        replication_reports[str(seed)] = report
        replication_baselines[str(seed)] = _baseline_report(
            known["base_score"]["va"],
            known["base_score"]["en"],
            scored,
        )
        replication_fixture_provenance[str(seed)] = provenance
        values = _gate_values(report)
        print(
            f"[strict-open] replication {seed}: "
            f"overall={values['overall_auroc']:.6f}, "
            f"noise={values['noise_auroc']:.6f}, "
            f"chirp={values['chirp_auroc']:.6f}, "
            f"recalls={values['noise_threshold_recall']:.6f}/"
            f"{values['chirp_threshold_recall']:.6f}",
            flush=True,
        )

    replication_gates = _replication_gates(replication_reports)
    source_paths = {
        "design_strict_v3_openset.py": Path(__file__).resolve(),
        "test_strict_v3_openset.py": (
            HERE / "test_strict_v3_openset.py"
        ).resolve(),
        "v3_time_domain_openset.py": (
            HERE / "v3_time_domain_openset.py"
        ).resolve(),
        "v3_scale/run_time_domain_dev.py": (
            V3_SCALE / "run_time_domain_dev.py"
        ).resolve(),
        "training/time_domain_geometry.py": (
            TRAINING / "time_domain_geometry.py"
        ).resolve(),
        "training/time_domain_invariant_patch_preprocess.py": (
            TRAINING / "time_domain_invariant_patch_preprocess.py"
        ).resolve(),
    }
    report = {
        "status": (
            "development_replication_pass"
            if replication_gates["all_pass"]
            else "development_replication_fail"
        ),
        "development_only": True,
        "release_evidence": False,
        "sealed_release_paths_read": 0,
        "consumed_test_rows_used": 0,
        "frontend": strict_preprocess.preprocess_metadata(),
        "closed_selection_accuracy": known["closed_accuracy"],
        "policy_selection": selection,
        "design": {
            "seed": DESIGN_NOVELTY_SEED,
            "fixture_provenance": design_fixture_provenance,
            "lof_baseline": baseline_design,
        },
        "replication": {
            "seeds": list(args.replication_seeds),
            "reports": replication_reports,
            "lof_baselines": replication_baselines,
            "fixture_provenance": replication_fixture_provenance,
            **replication_gates,
        },
        "serialization_replay": replay,
        "training_view_audit": training_view_audit,
        "known_only_contract": {
            "centers": "training only",
            "lof_density": "training only",
            "class_geometry": "training only",
            "prototypes": "enrollment only",
            "lof_ranks": "enrollment only",
            "geometry_ranks": "enrollment only",
            "final_rank_and_threshold": "enrollment only",
            "novelty_in_fit_or_calibration": False,
            "cannot_change_closed_label": True,
        },
        "provenance": {
            "command": [sys.executable, *sys.argv],
            "cwd": str(Path.cwd()),
            "device": str(device),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "git_head": _git_head(),
            "model_sources": model_provenance,
            "data_audit": data["data_audit"],
            "source_index_contract": data["cache_contract"],
            "source_hashes": {
                name: _sha256(path)
                for name, path in source_paths.items()
            },
            "artifact_hashes_before_report": {
                path.name: _sha256(path)
                for path in sorted(output.iterdir())
                if path.is_file()
            },
            "wall_clock_s": time.perf_counter() - started,
        },
    }
    report_path = output / "development_report.json"
    _atomic_json(report_path, report)
    report_sha = _sha256(report_path)
    _atomic_json(
        output / "REPORT_SHA256.json",
        {"file": report_path.name, "sha256": report_sha},
    )
    print(
        f"[strict-open] {report['status']} report_sha256={report_sha}",
        flush=True,
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--novelty-n", type=int, default=NOVELTY_N_EACH)
    parser.add_argument(
        "--replication-seeds",
        type=int,
        nargs="+",
        default=list(REPLICATION_NOVELTY_SEEDS),
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected": result["policy_selection"][
                    "selected_candidate"
                ],
                "replication_gates": result["replication"]["gates"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
