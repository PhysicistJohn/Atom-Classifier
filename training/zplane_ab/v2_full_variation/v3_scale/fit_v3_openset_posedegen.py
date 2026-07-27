"""v3 open-set rejector variant with a **pose-degeneracy** rank term.

This mirrors :mod:`fit_v3_openset` exactly, with one change: the additive
rejector blends a *third* rank, computed from features that describe how
**degenerate or confident the pose estimate itself was**.

Why that is legitimate.  The v3 frontend
(:mod:`time_domain_geometry` + :mod:`time_domain_invariant_patch_preprocess`)
estimates an occupied bandwidth from multiscale non-zero-lag autocorrelation
quotients and then resamples onto the dimensionless time coordinate
``u = n * bandwidth / target_frac``.  White noise has no coherent
autocorrelation, so its bandwidth estimate is degenerate; the normalisation then
maps noise into the same canonical geometry as a real emission and it stops
looking anomalous.  HANDOFF section 17 measured exactly that: refitting the
rejector on v3 fusion embeddings solved chirp (AUROC 0.9716, threshold recall
0.72-1.00) and broke noise (AUROC 0.8429 -> 0.6038, threshold recall
0.3467 -> 0.0000).

The rejector is **additive** and never changes the closed-set label, so it may
legitimately consume information the invariant frontend discards.  The
degeneracy/confidence of the pose estimate is already computed inside
``time_domain_geometry.estimate_band`` and thrown away by the classifier path.
Feeding it to the rejector reintroduces **no** scale dependence into the
classifier: the closed label is recomputed after scoring and asserted
bit-identical, exactly as in :mod:`fit_v3_openset`.

**The blend weight is a hyperparameter and this module refuses to guess it.**
The existing 0.80/0.20 split was chosen on novelty design seed 20260938, which
is spent.  Choosing a new weight by looking at results on the validation seeds
20260939/20260940 would convert the validation evidence into training evidence,
which evidence rule 4 forbids.  Therefore:

* ``--posedegen-weight`` is **required** and has no default.  There is no
  built-in value, because any built-in value would be a tuning decision this
  module has no evidence budget for.
* ``--design-novelty-seed`` is **required** and is emitted into the report.  A
  defensible weight must come from a **fresh** design novelty seed; 20260941 is
  proposed (:data:`PROPOSED_DESIGN_NOVELTY_SEED`) and is unused elsewhere in the
  repository.  The run is refused if the declared design seed is a reserved
  validation seed, the spent design seed 20260938, the consumed sealed seed
  20260729, or the unspent release seed 20260731.
* ``--role design`` draws novelty at the design seed and marks its report as
  hyperparameter selection, not evidence.  ``--role validate`` draws novelty at
  untouched validation seeds and refuses any overlap with the design seed.
  20260939/20260940 stay untouched for validation.

The pose-degeneracy **features** themselves are not defined here.  They come
from a sibling module (default import name :data:`POSEDEGEN_MODULE_DEFAULT`,
overridable with ``--posedegen-module``) whose contract is
:data:`REQUIRED_POSEDEGEN_API`.  The import is lazy and, if the module is absent
or does not satisfy the contract, this module fails with an explicit message
rather than silently degrading to the two-term policy.

Everything else is inherited unchanged from :mod:`fit_v3_openset`, by import
rather than by copy wherever the base function is usable as-is: gate floors,
the sealed/release path refusal, the fusion artifact hash validation, the
train-only density contract, the enrollment-only rank and q95 threshold
contract, the closed-label invariance assertion, the bit-identical rebuild
verification against the fusion artifact, and the full provenance block.

This produces development evidence.  It is not release evidence.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
V2 = HERE.parent
ZPAB = V2.parent
TRAINING = ZPAB.parent
for _path in (TRAINING, ZPAB, V2, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import assemble_v3_fusion as assemble  # noqa: E402
import fit_v3_openset as base  # noqa: E402
import invariant_patch_data as invariant_data  # noqa: E402
import run_time_domain_dev as dev  # noqa: E402
import time_domain_geometry as geometry  # noqa: E402
import time_domain_invariant_patch_preprocess as td_preprocess  # noqa: E402
import run_invariant_cnn_dev as runner  # noqa: E402
from known_only_patch_openset import KnownOnlyLOFOpenSet  # noqa: E402
from openset_eval import NOVELTY  # noqa: E402
from train import embed_all, nearest  # noqa: E402
from v3_time_domain_openset import (  # noqa: E402
    FROZEN_GEOMETRY_FEATURE,
    FROZEN_POLICY_KIND,
    FROZEN_THRESHOLD_QUANTILE,
    KnownOnlyGeometry,
    aggregate_deviation,
    empirical_rank,
)


# ---------------------------------------------------------------------------
# inherited, never redefined
# ---------------------------------------------------------------------------

# The two existing rank weights keep their frozen relative proportions.  They
# are imported from the base module, which imports them from the frozen policy.
BRANCH_LOF_RANK_WEIGHT = base.BRANCH_LOF_RANK_WEIGHT
GEOMETRY_RANK_WEIGHT = base.GEOMETRY_RANK_WEIGHT
THRESHOLD_QUANTILE = base.THRESHOLD_QUANTILE
BRANCH_LOF = base.BRANCH_LOF

# Gates: imported, not re-typed, so they cannot drift and cannot be lowered.
GATE_FLOORS = base.GATE_FLOORS
KNOWN_FUR_CEILING = base.KNOWN_FUR_CEILING

CENTER_SPLIT = base.CENTER_SPLIT
PROTOTYPE_SPLIT = base.PROTOTYPE_SPLIT
METRIC_SPLIT = base.METRIC_SPLIT

NOVELTY_FAMILIES = base.NOVELTY_FAMILIES
DEFAULT_NOVELTY_N = base.DEFAULT_NOVELTY_N
DEFAULT_PREFIX_LENGTHS = base.DEFAULT_PREFIX_LENGTHS
DEFAULT_REPRODUCTION_TOLERANCE = base.DEFAULT_REPRODUCTION_TOLERANCE

reject_sealed_path = base.reject_sealed_path
write_json = base.write_json
validate_prefix_lengths = base.validate_prefix_lengths
fit_branch_lof = base.fit_branch_lof
score_branch_lof = base.score_branch_lof
save_branch_lof = base.save_branch_lof
assert_closed_label_unchanged = base.assert_closed_label_unchanged
family_metrics = base.family_metrics
gate_summary = base.gate_summary
load_fusion_artifact = base.load_fusion_artifact
resolve_device = base.resolve_device
FusionArtifact = base.FusionArtifact

_sha256 = base._sha256

if abs((BRANCH_LOF_RANK_WEIGHT + GEOMETRY_RANK_WEIGHT) - 1.0) > 1e-12:
    raise RuntimeError(
        "the two inherited rank weights must remain a convex pair; the "
        "pose-degeneracy term is layered on top of that pair, not mixed into it"
    )


# ---------------------------------------------------------------------------
# seeds
# ---------------------------------------------------------------------------

# 20260938 designed the frozen 0.80/0.20 blend and is SPENT.
SPENT_DESIGN_NOVELTY_SEED = base.DESIGN_NOVELTY_SEED
# 20260939 / 20260940 are the validation evidence and must never be fit against.
RESERVED_VALIDATION_SEEDS = base.DEFAULT_NOVELTY_SEEDS
# 20260731 is the unspent release seed.  It cannot be reached from this module.
RELEASE_SEED_NEVER_SPENT_HERE = base.RELEASE_SEED_NEVER_SPENT_HERE
# 20260729 is the consumed sealed v2 release suite.
SEALED_RELEASE_SEED = 20260729
# The fresh design seed this module proposes for choosing --posedegen-weight.
PROPOSED_DESIGN_NOVELTY_SEED = 20260941

ROLES = ("design", "validate")


# ---------------------------------------------------------------------------
# the pose-degeneracy feature module (sibling task; imported lazily)
# ---------------------------------------------------------------------------

POSEDEGEN_MODULE_DEFAULT = "pose_degeneracy"

REQUIRED_POSEDEGEN_API = (
    "POSE_DEGENERACY_VERSION",
    "FEATURE_NAMES",
    "pose_degeneracy_features",
)

POSEDEGEN_CONTRACT = (
    "the pose-degeneracy module must expose:\n"
    "  POSE_DEGENERACY_VERSION: str\n"
    "  FEATURE_NAMES: Sequence[str], non-empty and unique\n"
    "  pose_degeneracy_features(iq, *, min_bandwidth=...) -> float64 vector of\n"
    "      shape (len(FEATURE_NAMES),) for ONE raw complex I/Q capture.\n"
    "It is called once per capture rather than once per batch because the v3\n"
    "frontend's resolution floor min_bandwidth depends on the capture length, "
    "and\n"
    "the features must describe the pose the frontend actually adopted."
)

# The frontend calls estimate_band with a length-dependent resolution floor
# (time_domain_geometry.minimum_bandwidth_for_active_span).  The degeneracy
# features are computed with the SAME floor, so 'bandwidth_clipped_to_minimum'
# and friends describe the pose the classifier actually used rather than a
# different pose computed under the estimator's library default.  A module
# constant with no flag: it is a correctness requirement, not a tuning knob.
POSEDEGEN_MATCH_FRONTEND_MIN_BANDWIDTH = True

# The reduction from the per-feature standardized deviations to one scalar
# nonconformity.  Deliberately a module constant with no command-line flag, for
# the same reason BRANCH_LOF is: an operator must not be able to search it.
POSEDEGEN_REDUCER = "rms"

POSEDEGEN_POLICY_SCHEMA = 1
POSEDEGEN_POLICY_KIND = (
    "v3_known_only_lof_geometry_posedegeneracy_rank_blend"
)


@dataclass(frozen=True)
class PoseDegeneracySource:
    """A validated pose-degeneracy feature module."""

    module_name: str
    module: Any
    version: str
    feature_names: tuple[str, ...]
    source_path: str | None
    source_sha256: str | None

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    def provenance(self) -> dict[str, Any]:
        metadata_fn = getattr(self.module, "pose_degeneracy_metadata", None)
        return {
            "module": self.module_name,
            "version": self.version,
            "feature_names": list(self.feature_names),
            "feature_count": self.feature_count,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "reducer": POSEDEGEN_REDUCER,
            "min_bandwidth_matches_frontend": bool(
                POSEDEGEN_MATCH_FRONTEND_MIN_BANDWIDTH
            ),
            "extractor_metadata": (
                metadata_fn() if callable(metadata_fn) else None
            ),
            "consumed_by": "the additive rejector only",
            "reaches_closed_classifier": False,
        }


def load_posedegen_module(
    module_name: str = POSEDEGEN_MODULE_DEFAULT,
) -> PoseDegeneracySource:
    """Import and validate the sibling pose-degeneracy feature module.

    The import is deliberately lazy and deliberately loud.  If the module is
    missing or does not satisfy :data:`REQUIRED_POSEDEGEN_API`, this raises
    instead of falling back to the two-term policy: a silent fallback would emit
    a report that looks like a pose-degeneracy result and is not one.
    """
    name = str(module_name)
    if not name:
        raise ValueError("--posedegen-module must be a non-empty module name")
    try:
        module = importlib.import_module(name)
    except ImportError as exc:
        raise RuntimeError(
            f"the pose-degeneracy feature module {name!r} is not importable "
            f"({exc}).  It is written by the sibling task and is required: this "
            "variant will not silently degrade to the two-term policy.\n"
            f"{POSEDEGEN_CONTRACT}"
        ) from exc

    missing = [item for item in REQUIRED_POSEDEGEN_API if not hasattr(module, item)]
    if missing:
        raise RuntimeError(
            f"pose-degeneracy module {name!r} is missing {missing}.\n"
            f"{POSEDEGEN_CONTRACT}"
        )
    if not callable(getattr(module, "pose_degeneracy_features")):
        raise RuntimeError(
            f"pose-degeneracy module {name!r} exposes a non-callable "
            "pose_degeneracy_features.\n"
            f"{POSEDEGEN_CONTRACT}"
        )
    raw_names = getattr(module, "FEATURE_NAMES")
    if isinstance(raw_names, (str, bytes)):
        raise RuntimeError(
            f"pose-degeneracy module {name!r} exposes FEATURE_NAMES as a "
            "string, not a sequence.\n"
            f"{POSEDEGEN_CONTRACT}"
        )
    feature_names = tuple(str(item) for item in raw_names)
    if not feature_names or len(set(feature_names)) != len(feature_names):
        raise RuntimeError(
            f"pose-degeneracy module {name!r} must declare a non-empty set of "
            "unique feature names.\n"
            f"{POSEDEGEN_CONTRACT}"
        )
    version = str(getattr(module, "POSE_DEGENERACY_VERSION"))
    if not version:
        raise RuntimeError(
            f"pose-degeneracy module {name!r} declares an empty "
            "POSE_DEGENERACY_VERSION.\n"
            f"{POSEDEGEN_CONTRACT}"
        )

    source_path = getattr(module, "__file__", None)
    source_sha256 = (
        _sha256(Path(source_path)) if source_path and Path(source_path).is_file()
        else None
    )
    return PoseDegeneracySource(
        module_name=name,
        module=module,
        version=version,
        feature_names=feature_names,
        source_path=str(source_path) if source_path else None,
        source_sha256=source_sha256,
    )


def frontend_min_bandwidth(
    raw_length: int,
    patch_length: int,
    target_frac: float,
) -> float | None:
    """The resolution floor the v3 frontend itself passes to ``estimate_band``."""
    if not POSEDEGEN_MATCH_FRONTEND_MIN_BANDWIDTH:
        return None
    return geometry.minimum_bandwidth_for_active_span(
        int(raw_length), int(patch_length), float(target_frac)
    )


def pose_row(
    source: PoseDegeneracySource,
    capture: np.ndarray,
    *,
    patch_length: int,
    target_frac: float,
) -> np.ndarray:
    """Pose-degeneracy features for one raw capture, at the frontend's floor."""
    minimum = frontend_min_bandwidth(len(capture), patch_length, target_frac)
    kwargs = {} if minimum is None else {"min_bandwidth": float(minimum)}
    return np.asarray(
        source.module.pose_degeneracy_features(capture, **kwargs),
        dtype=np.float64,
    )


def pose_matrix(
    source: PoseDegeneracySource,
    captures: Sequence[np.ndarray],
    *,
    patch_length: int,
    target_frac: float,
    population: str,
) -> np.ndarray:
    """Stack :func:`pose_row` over an ordered population, and validate it."""
    rows = list(captures)
    if not rows:
        raise ValueError(f"{population}: pose-degeneracy needs at least one row")
    value = np.stack(
        [
            pose_row(
                source,
                capture,
                patch_length=patch_length,
                target_frac=target_frac,
            )
            for capture in rows
        ]
    ).astype(np.float64)
    expected = (len(rows), source.feature_count)
    if value.shape != expected:
        raise RuntimeError(
            f"{population}: {source.module_name}.pose_degeneracy_features "
            f"produced shape {value.shape}, expected {expected}.\n"
            f"{POSEDEGEN_CONTRACT}"
        )
    if not np.isfinite(value).all():
        raise RuntimeError(
            f"{population}: {source.module_name}.pose_degeneracy_features "
            "returned a non-finite value"
        )
    return value


def _matrix_sha256(value: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    return hashlib.sha256(canonical.view(np.uint8)).hexdigest()


# ---------------------------------------------------------------------------
# hyperparameter and seed hygiene
# ---------------------------------------------------------------------------


def validate_role(role: Any) -> str:
    value = str(role)
    if value not in ROLES:
        raise ValueError(f"--role must be one of {list(ROLES)}, got {value!r}")
    return value


def validate_posedegen_weight(weight: Any) -> float:
    """The blend weight is a hyperparameter: explicit, finite, and in [0, 1)."""
    if weight is None:
        raise ValueError(
            "--posedegen-weight is required and has no default.  A default "
            "would be an undocumented tuning decision; a defensible value must "
            "come from a fresh design novelty seed (proposed "
            f"{PROPOSED_DESIGN_NOVELTY_SEED}), never from "
            f"{list(RESERVED_VALIDATION_SEEDS)}."
        )
    try:
        value = float(weight)
    except (TypeError, ValueError) as exc:
        raise ValueError("--posedegen-weight must be a finite float") from exc
    if not np.isfinite(value) or value < 0.0 or value >= 1.0:
        raise ValueError(
            f"--posedegen-weight must lie in [0, 1), got {value!r}.  At 0.0 the "
            "policy is bit-identical to the frozen two-term blend."
        )
    return value


def validate_seed_plan(
    role: Any,
    design_novelty_seed: Any,
    novelty_seeds: Sequence[int],
) -> tuple[str, int, tuple[int, ...]]:
    """Return ``(role, design_seed, novelty_seeds)`` or refuse to run.

    Evidence rule 4 in one function.  The declared design seed is the seed the
    blend weight was chosen on; it may never be a reserved validation seed, the
    spent design seed, the consumed sealed seed, or the unspent release seed.
    """
    role_value = validate_role(role)
    if design_novelty_seed is None:
        raise ValueError(
            "--design-novelty-seed is required: the report must record which "
            "novelty seed the blend weight was chosen on.  Proposed fresh "
            f"seed: {PROPOSED_DESIGN_NOVELTY_SEED}."
        )
    if isinstance(design_novelty_seed, (bool, np.bool_)) or not isinstance(
        design_novelty_seed, (int, np.integer)
    ):
        raise ValueError("--design-novelty-seed must be an integer")
    design = int(design_novelty_seed)

    if design in RESERVED_VALIDATION_SEEDS:
        raise ValueError(
            f"design novelty seed {design} is a reserved validation seed "
            f"{list(RESERVED_VALIDATION_SEEDS)}.  Choosing a blend weight there "
            "converts validation evidence into training evidence, which "
            "evidence rule 4 forbids.  Use a fresh design seed, e.g. "
            f"{PROPOSED_DESIGN_NOVELTY_SEED}."
        )
    if design == SPENT_DESIGN_NOVELTY_SEED:
        raise ValueError(
            f"novelty seed {design} already designed the frozen 0.80/0.20 blend "
            "and is spent; a new hyperparameter may not be chosen on it"
        )
    if design == SEALED_RELEASE_SEED:
        raise ValueError(
            f"{design} is the consumed sealed release suite and may not be used "
            "for anything"
        )
    if design == RELEASE_SEED_NEVER_SPENT_HERE:
        raise ValueError(
            f"{design} is the unspent release seed and is not a development "
            "design seed"
        )

    seeds = base.validate_novelty_seeds(novelty_seeds)
    if SEALED_RELEASE_SEED in seeds:
        raise ValueError(
            f"{SEALED_RELEASE_SEED} is the consumed sealed release suite and "
            "may not be drawn as development novelty"
        )

    if role_value == "validate":
        if design in seeds:
            raise ValueError(
                f"design novelty seed {design} is also a validation novelty "
                "seed in this run.  The seed a weight was chosen on cannot be "
                "the seed that validates it."
            )
    else:
        overlap = [seed for seed in seeds if seed in RESERVED_VALIDATION_SEEDS]
        if overlap:
            raise ValueError(
                f"a design run may not draw novelty at reserved validation "
                f"seeds {overlap}; they stay untouched for validation"
            )
        if design not in seeds:
            raise ValueError(
                f"a design run must draw novelty at its declared design seed "
                f"{design}; got {list(seeds)}"
            )
    return role_value, design, seeds


# ---------------------------------------------------------------------------
# small validators, mirroring the frozen module's private ones
# ---------------------------------------------------------------------------


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


def _label_vector(value: np.ndarray, rows: int, n_classes: int) -> np.ndarray:
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


# ---------------------------------------------------------------------------
# the pose-degeneracy density (TRAINING only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KnownOnlyPoseDegeneracy:
    """Per-class standardized pose-degeneracy deviations, fit without novelty.

    Deliberately the same shape as
    :class:`v3_time_domain_openset.KnownOnlyGeometry`: class-conditional mean and
    scale fit on **training** rows, then an absolute standardized deviation from
    the *predicted* class.  Conditioning on the predicted class matters because
    genuinely different emissions occupy genuinely different bandwidths, so a
    global standardization would call a narrow known class degenerate.
    """

    class_mean: np.ndarray
    class_scale: np.ndarray
    feature_names: tuple[str, ...]

    @classmethod
    def fit(
        cls,
        training_pose: np.ndarray,
        training_labels: np.ndarray,
        *,
        n_classes: int,
        feature_names: Sequence[str],
    ) -> "KnownOnlyPoseDegeneracy":
        names = tuple(str(item) for item in feature_names)
        statistics = np.asarray(training_pose, dtype=np.float64)
        if statistics.ndim != 2 or statistics.shape[1] != len(names):
            raise ValueError(
                "training pose-degeneracy features must be a "
                f"[rows, {len(names)}] matrix"
            )
        if not len(statistics) or not np.isfinite(statistics).all():
            raise ValueError("training pose-degeneracy features must be finite")
        labels = _label_vector(training_labels, len(statistics), int(n_classes))
        means, scales = [], []
        for class_index in range(int(n_classes)):
            current = statistics[labels == class_index]
            if len(current) < 2:
                raise ValueError(f"class {class_index} needs at least two rows")
            means.append(current.mean(axis=0))
            scales.append(np.maximum(current.std(axis=0), 1e-5))
        return cls(np.stack(means), np.stack(scales), names)

    def deviations(
        self,
        pose: np.ndarray,
        predicted_classes: np.ndarray,
    ) -> np.ndarray:
        statistics = np.asarray(pose, dtype=np.float64)
        if statistics.ndim != 2 or statistics.shape[1] != len(self.feature_names):
            raise RuntimeError("pose-degeneracy feature schema changed")
        if not len(statistics) or not np.isfinite(statistics).all():
            raise ValueError("pose-degeneracy features must be finite")
        predicted = _label_vector(
            predicted_classes, len(statistics), len(self.class_mean)
        )
        return np.abs(
            (statistics - self.class_mean[predicted]) / self.class_scale[predicted]
        )

    def nonconformity(
        self,
        pose: np.ndarray,
        predicted_classes: np.ndarray,
    ) -> np.ndarray:
        """Reduce the standardized deviations to one scalar per row."""
        deviations = self.deviations(pose, predicted_classes)
        return aggregate_deviation(
            deviations,
            np.arange(len(self.feature_names), dtype=np.int64),
            reducer=POSEDEGEN_REDUCER,
        )


def blend(
    branch_lof_rank: np.ndarray,
    geometry_rank: np.ndarray,
    posedegen_rank: np.ndarray,
    posedegen_weight: float,
) -> np.ndarray:
    """``(1 - w) * frozen_blend + w * posedegen_rank``.

    The frozen 0.80/0.20 proportions between the two existing terms are
    preserved exactly, and at ``w == 0`` the result is bit-identical to
    :class:`v3_time_domain_openset.FrozenV3OpenSet`'s combined raw score.
    """
    weight = validate_posedegen_weight(posedegen_weight)
    frozen_blend = (
        BRANCH_LOF_RANK_WEIGHT * branch_lof_rank
        + GEOMETRY_RANK_WEIGHT * geometry_rank
    )
    return (1.0 - weight) * frozen_blend + weight * posedegen_rank


@dataclass(frozen=True)
class PoseDegenOpenSet:
    """The frozen v3 policy plus one pose-degeneracy rank.

    Density (class geometry, pose-degeneracy class statistics) is fit on
    **training** rows.  Every empirical rank and the q95 operating threshold are
    fit on **enrollment** rows.  No method accepts novelty data, and predicted
    closed labels are inputs rather than outputs.
    """

    geometry: KnownOnlyGeometry
    geometry_feature_index: int
    geometry_calibration_raw: np.ndarray
    pose: KnownOnlyPoseDegeneracy
    pose_calibration_raw: np.ndarray
    combined_calibration_raw: np.ndarray
    calibration_scores: np.ndarray
    threshold: float
    posedegen_weight: float
    posedegen_version: str

    @classmethod
    def fit(
        cls,
        training_packed: np.ndarray,
        training_labels: np.ndarray,
        training_pose: np.ndarray,
        enrollment_packed: np.ndarray,
        enrollment_predicted_classes: np.ndarray,
        enrollment_branch_lof_ranks: np.ndarray,
        enrollment_pose: np.ndarray,
        *,
        n_classes: int,
        posedegen_weight: float,
        posedegen_feature_names: Sequence[str],
        posedegen_version: str,
    ) -> "PoseDegenOpenSet":
        weight = validate_posedegen_weight(posedegen_weight)

        geometry = KnownOnlyGeometry.fit(
            training_packed, training_labels, n_classes=int(n_classes)
        )
        feature_index = int(geometry.indices((FROZEN_GEOMETRY_FEATURE,))[0])
        enrollment_deviation = geometry.deviations(
            enrollment_packed, enrollment_predicted_classes
        )[:, feature_index]
        geometry_calibration = np.sort(
            _finite_vector(enrollment_deviation, "enrollment geometry")
        )
        geometry_rank = empirical_rank(enrollment_deviation, geometry_calibration)

        pose = KnownOnlyPoseDegeneracy.fit(
            training_pose,
            training_labels,
            n_classes=int(n_classes),
            feature_names=posedegen_feature_names,
        )
        enrollment_pose_raw = pose.nonconformity(
            enrollment_pose, enrollment_predicted_classes
        )
        pose_calibration = np.sort(
            _finite_vector(enrollment_pose_raw, "enrollment pose degeneracy")
        )
        pose_rank = empirical_rank(enrollment_pose_raw, pose_calibration)

        lof_rank = _rank_score(
            enrollment_branch_lof_ranks, "enrollment_branch_lof_ranks"
        )
        if not (len(lof_rank) == len(geometry_rank) == len(pose_rank)):
            raise ValueError(
                "enrollment branch-LOF, packed and pose-degeneracy row counts "
                "differ"
            )

        combined_raw = blend(lof_rank, geometry_rank, pose_rank, weight)
        combined_calibration = np.sort(combined_raw)
        calibration_scores = empirical_rank(combined_raw, combined_calibration)
        threshold = float(
            np.quantile(calibration_scores, FROZEN_THRESHOLD_QUANTILE)
        )
        return cls(
            geometry=geometry,
            geometry_feature_index=feature_index,
            geometry_calibration_raw=geometry_calibration,
            pose=pose,
            pose_calibration_raw=pose_calibration,
            combined_calibration_raw=combined_calibration,
            calibration_scores=calibration_scores,
            threshold=threshold,
            posedegen_weight=weight,
            posedegen_version=str(posedegen_version),
        )

    def score(
        self,
        branch_lof_ranks: np.ndarray,
        packed: np.ndarray,
        pose: np.ndarray,
        predicted_classes: np.ndarray,
    ) -> np.ndarray:
        """Return the enrollment-ranked score.  No label is produced."""
        lof_rank = _rank_score(branch_lof_ranks, "branch_lof_ranks")
        deviation = self.geometry.deviations(packed, predicted_classes)
        pose_raw = self.pose.nonconformity(pose, predicted_classes)
        if not (len(lof_rank) == len(deviation) == len(pose_raw)):
            raise ValueError(
                "branch-LOF, packed and pose-degeneracy row counts differ"
            )
        geometry_rank = empirical_rank(
            deviation[:, self.geometry_feature_index],
            self.geometry_calibration_raw,
        )
        pose_rank = empirical_rank(pose_raw, self.pose_calibration_raw)
        combined_raw = blend(
            lof_rank, geometry_rank, pose_rank, self.posedegen_weight
        )
        return empirical_rank(combined_raw, self.combined_calibration_raw)

    def to_payload(self) -> dict[str, np.ndarray]:
        """Return an allow-pickle-free ``np.savez`` payload."""
        return {
            "schema": np.asarray(POSEDEGEN_POLICY_SCHEMA, dtype=np.int64),
            "kind": np.asarray(POSEDEGEN_POLICY_KIND),
            "base_policy_kind": np.asarray(FROZEN_POLICY_KIND),
            "geometry_feature": np.asarray(FROZEN_GEOMETRY_FEATURE),
            "branch_lof_rank_weight": np.asarray(
                BRANCH_LOF_RANK_WEIGHT, dtype=np.float64
            ),
            "geometry_weight": np.asarray(
                GEOMETRY_RANK_WEIGHT, dtype=np.float64
            ),
            "posedegen_weight": np.asarray(
                self.posedegen_weight, dtype=np.float64
            ),
            "posedegen_reducer": np.asarray(POSEDEGEN_REDUCER),
            "posedegen_version": np.asarray(self.posedegen_version),
            "threshold_quantile": np.asarray(
                FROZEN_THRESHOLD_QUANTILE, dtype=np.float64
            ),
            "threshold": np.asarray(self.threshold, dtype=np.float64),
            "geometry_feature_index": np.asarray(
                self.geometry_feature_index, dtype=np.int64
            ),
            "feature_names": np.asarray(self.geometry.feature_names),
            "class_mean": np.asarray(self.geometry.class_mean, dtype=np.float64),
            "class_scale": np.asarray(
                self.geometry.class_scale, dtype=np.float64
            ),
            "pose_feature_names": np.asarray(self.pose.feature_names),
            "pose_class_mean": np.asarray(
                self.pose.class_mean, dtype=np.float64
            ),
            "pose_class_scale": np.asarray(
                self.pose.class_scale, dtype=np.float64
            ),
            "geometry_calibration_raw": np.asarray(
                self.geometry_calibration_raw, dtype=np.float64
            ),
            "pose_calibration_raw": np.asarray(
                self.pose_calibration_raw, dtype=np.float64
            ),
            "combined_calibration_raw": np.asarray(
                self.combined_calibration_raw, dtype=np.float64
            ),
            "calibration_scores": np.asarray(
                self.calibration_scores, dtype=np.float64
            ),
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, np.ndarray],
    ) -> "PoseDegenOpenSet":
        """Validate and restore an allow-pickle-free serialized policy."""
        required = set(
            cls(
                geometry=KnownOnlyGeometry(
                    np.zeros((1, 1)), np.ones((1, 1)), ("x",)
                ),
                geometry_feature_index=0,
                geometry_calibration_raw=np.zeros(1),
                pose=KnownOnlyPoseDegeneracy(
                    np.zeros((1, 1)), np.ones((1, 1)), ("p",)
                ),
                pose_calibration_raw=np.zeros(1),
                combined_calibration_raw=np.zeros(1),
                calibration_scores=np.zeros(1),
                threshold=0.0,
                posedegen_weight=0.0,
                posedegen_version="probe",
            ).to_payload()
        )
        if set(payload) != required:
            raise ValueError(
                "serialized policy keys differ: "
                f"missing={sorted(required - set(payload))}, "
                f"extra={sorted(set(payload) - required)}"
            )
        scalar_contract = {
            "schema": POSEDEGEN_POLICY_SCHEMA,
            "kind": POSEDEGEN_POLICY_KIND,
            "base_policy_kind": FROZEN_POLICY_KIND,
            "geometry_feature": FROZEN_GEOMETRY_FEATURE,
            "branch_lof_rank_weight": BRANCH_LOF_RANK_WEIGHT,
            "geometry_weight": GEOMETRY_RANK_WEIGHT,
            "posedegen_reducer": POSEDEGEN_REDUCER,
            "threshold_quantile": FROZEN_THRESHOLD_QUANTILE,
        }
        for name, expected in scalar_contract.items():
            value = np.asarray(payload[name])
            if value.ndim != 0 or value.item() != expected:
                raise ValueError(f"serialized {name} does not match policy")

        weight = validate_posedegen_weight(
            float(np.asarray(payload["posedegen_weight"]).item())
        )
        version = str(np.asarray(payload["posedegen_version"]).item())
        if not version:
            raise ValueError("serialized posedegen_version is empty")

        def _names(key: str) -> tuple[str, ...]:
            array = np.asarray(payload[key])
            if array.ndim != 1 or not len(array):
                raise ValueError(f"serialized {key} must be a vector")
            return tuple(str(item) for item in array.tolist())

        def _class_arrays(
            mean_key: str, scale_key: str, names: tuple[str, ...]
        ) -> tuple[np.ndarray, np.ndarray]:
            mean = np.asarray(payload[mean_key], dtype=np.float64)
            scale = np.asarray(payload[scale_key], dtype=np.float64)
            if (
                mean.ndim != 2
                or mean.shape != scale.shape
                or mean.shape[1] != len(names)
                or not np.isfinite(mean).all()
                or not np.isfinite(scale).all()
                or np.any(scale <= 0.0)
            ):
                raise ValueError(f"serialized {mean_key}/{scale_key} is invalid")
            return mean, scale

        names = _names("feature_names")
        mean, scale = _class_arrays("class_mean", "class_scale", names)
        pose_names = _names("pose_feature_names")
        pose_mean, pose_scale = _class_arrays(
            "pose_class_mean", "pose_class_scale", pose_names
        )
        if len(pose_mean) != len(mean):
            raise ValueError("serialized class counts differ")

        feature_index = int(np.asarray(payload["geometry_feature_index"]).item())
        if (
            feature_index < 0
            or feature_index >= len(names)
            or names[feature_index] != FROZEN_GEOMETRY_FEATURE
        ):
            raise ValueError("serialized geometry feature index is invalid")

        calibrations = {
            key: np.asarray(payload[key], dtype=np.float64)
            for key in (
                "geometry_calibration_raw",
                "pose_calibration_raw",
                "combined_calibration_raw",
            )
        }
        for key, value in calibrations.items():
            _finite_vector(value, key)
            if np.any(value[1:] < value[:-1]):
                raise ValueError(f"serialized {key} must be sorted")
        calibration_scores = _rank_score(
            payload["calibration_scores"], "calibration_scores"
        )
        if len({len(value) for value in calibrations.values()} | {
            len(calibration_scores)
        }) != 1:
            raise ValueError("serialized calibration lengths differ")

        threshold = float(np.asarray(payload["threshold"]).item())
        expected_threshold = float(
            np.quantile(calibration_scores, FROZEN_THRESHOLD_QUANTILE)
        )
        if not np.isfinite(threshold) or threshold != expected_threshold:
            raise ValueError("serialized threshold is invalid")

        return cls(
            geometry=KnownOnlyGeometry(mean, scale, names),
            geometry_feature_index=feature_index,
            geometry_calibration_raw=calibrations["geometry_calibration_raw"],
            pose=KnownOnlyPoseDegeneracy(pose_mean, pose_scale, pose_names),
            pose_calibration_raw=calibrations["pose_calibration_raw"],
            combined_calibration_raw=calibrations["combined_calibration_raw"],
            calibration_scores=calibration_scores,
            threshold=threshold,
            posedegen_weight=weight,
            posedegen_version=version,
        )


def fit_policy(
    training_packed: np.ndarray,
    training_labels: np.ndarray,
    training_pose: np.ndarray,
    enrollment_packed: np.ndarray,
    enrollment_predicted_classes: np.ndarray,
    enrollment_branch_lof_ranks: np.ndarray,
    enrollment_pose: np.ndarray,
    *,
    n_classes: int,
    posedegen_weight: float,
    posedegen_feature_names: Sequence[str],
    posedegen_version: str,
) -> PoseDegenOpenSet:
    """Fit the rejector: density on TRAINING, ranks/threshold on ENROLLMENT.

    As in :func:`fit_v3_openset.fit_policy` the signature is the guarantee: only
    training and enrollment arrays are parameters, so no selection row, no
    novelty row and no release row can influence the class geometry, the
    pose-degeneracy statistics, any empirical rank, or the q95 threshold.
    """
    return PoseDegenOpenSet.fit(
        training_packed,
        training_labels,
        training_pose,
        enrollment_packed,
        enrollment_predicted_classes,
        enrollment_branch_lof_ranks,
        enrollment_pose,
        n_classes=int(n_classes),
        posedegen_weight=posedegen_weight,
        posedegen_feature_names=posedegen_feature_names,
        posedegen_version=posedegen_version,
    )


# ---------------------------------------------------------------------------
# rebuilding the populations through the v3 fusion
# ---------------------------------------------------------------------------

# ``run_time_domain_dev._prepare_data`` preprocesses these three populations, in
# this order, by calling the module-level ``_preprocess_rows`` once per
# population with the raw captures for that population.
PREPARE_DATA_POPULATIONS = (
    ("train", CENTER_SPLIT),
    ("enroll", PROTOTYPE_SPLIT),
    ("selection", METRIC_SPLIT),
)


class _PoseRecorder:
    """Compute pose-degeneracy features from the captures as they are preprocessed.

    ``_prepare_data`` builds the raw captures (including the multi-length
    training views, whose per-row prefix length is not recoverable from the
    returned arrays), preprocesses them, and drops them.  The pose-degeneracy
    features need those same raw rows.

    Rather than reimplement the capture construction -- which would duplicate
    the training-view expansion and could silently drift out of alignment --
    this wraps ``_prepare_data``'s own ``_preprocess_rows`` and derives the
    features from exactly the captures it was handed, in exactly its order.
    Nothing about the preprocessing is altered: the wrapped call is delegated
    unchanged and its result returned unchanged.

    The interception is verified afterwards by :func:`_recorded_pose`, which
    refuses to continue unless one matrix was recorded per population, in order,
    with a row count matching that population's packed rows.  If
    ``_prepare_data`` ever stops routing through ``_preprocess_rows``, the run
    fails loudly instead of scoring a misaligned feature.
    """

    def __init__(
        self,
        inner: Any,
        source: PoseDegeneracySource,
        *,
        target_frac: float,
    ) -> None:
        self._inner = inner
        self._source = source
        self._target_frac = float(target_frac)
        self.matrices: list[np.ndarray] = []

    def __call__(
        self,
        captures: Sequence[np.ndarray],
        *,
        patch_length: int,
        patch_count: int,
        target_frac: float,
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        rows = list(captures)
        index = len(self.matrices)
        population = (
            PREPARE_DATA_POPULATIONS[index][0]
            if index < len(PREPARE_DATA_POPULATIONS)
            else f"unexpected population {index}"
        )
        self.matrices.append(
            pose_matrix(
                self._source,
                rows,
                patch_length=int(patch_length),
                target_frac=float(target_frac),
                population=population,
            )
        )
        return self._inner(
            rows,
            patch_length=patch_length,
            patch_count=patch_count,
            target_frac=target_frac,
        )


def _recorded_pose(
    recorder: _PoseRecorder,
    data: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Validate the recorded matrices and key them by split."""
    if len(recorder.matrices) != len(PREPARE_DATA_POPULATIONS):
        raise AssertionError(
            "pose-degeneracy interception recorded "
            f"{len(recorder.matrices)} populations, expected "
            f"{len(PREPARE_DATA_POPULATIONS)}; run_time_domain_dev._prepare_data "
            "no longer routes every population through _preprocess_rows"
        )
    result: dict[str, np.ndarray] = {}
    for (name, split), matrix in zip(
        PREPARE_DATA_POPULATIONS, recorder.matrices
    ):
        packed = data[f"x{split}"]
        if len(matrix) != len(packed):
            raise AssertionError(
                f"{name}: {len(matrix)} pose-degeneracy rows for {len(packed)} "
                "packed rows; the features would be misaligned"
            )
        result[split] = matrix
    return result


@dataclass
class Populations:
    """Train / enrollment / selection rebuilt through the v3 fusion.

    Identical to :class:`fit_v3_openset.Populations` with one addition:
    ``pose`` holds the per-row pose-degeneracy features, keyed by split.  Those
    describe how degenerate or confident ``estimate_band``'s pose was for each
    capture -- information the invariant frontend computes and then discards.
    It is the only new information this variant consumes, and it reaches the
    additive rejector only.
    """

    data: dict[str, Any]
    nets: dict[str, torch.nn.Module]
    branch_embeddings: dict[str, dict[str, np.ndarray]]
    fused: dict[str, np.ndarray]
    real_center: np.ndarray
    complex_center: np.ndarray
    prototypes: np.ndarray
    reproduction: dict[str, float]
    branch_hashes: dict[str, dict[str, str]]
    pose: dict[str, np.ndarray]


def rebuild_populations(
    artifact: FusionArtifact,
    device: torch.device,
    posedegen: PoseDegeneracySource,
    *,
    tolerance: float = DEFAULT_REPRODUCTION_TOLERANCE,
) -> Populations:
    """Rebuild every split through the fusion, then prove it is the same fusion.

    This mirrors :func:`fit_v3_openset.rebuild_populations` step for step; the
    only difference is that the raw captures ``run_time_domain_dev._prepare_data``
    preprocesses are also passed to the pose-degeneracy extractor, so every
    packed row has an aligned degeneracy vector.  The reproduction check against
    the stored centers and prototypes is unchanged, so a run still refuses to
    continue unless the rebuilt fusion is the fusion the artifact recorded.
    """
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("reproduction tolerance must be positive and finite")

    branches = {
        name: assemble.load_branch(
            Path(artifact.metrics["source_branches"][name]["directory"]), name
        )
        for name in ("real", "complex")
    }
    assemble._assert_branches_match(branches["real"], branches["complex"])
    for name, branch in branches.items():
        recorded = artifact.metrics["source_branches"][name]["sha256"]
        if branch["hashes"] != recorded:
            raise RuntimeError(
                f"{name} branch hashes differ from the fusion artifact record"
            )

    config = branches["real"]["net"].cfg
    contract = branches["real"]["metrics"]["source_index_contract"]["contract"]
    target_frac = float(contract["config"]["target_frac"])

    source = invariant_data.load(
        patch_length=config.patch_length,
        patch_count=config.patch_count,
        target_frac=target_frac,
        model_seed=artifact.seed,
        build_if_missing=False,
    )
    if source["data_audit"].get("consumed_test_rows_exposed") != 0:
        raise AssertionError("data loader exposed consumed test rows")
    forbidden = [key for key in source if "test" in key.lower()]
    if forbidden:
        raise AssertionError(f"data loader exposed test-like keys: {forbidden}")
    if source["cache_contract"] != contract:
        raise RuntimeError("current cache contract differs from branch source")

    manifest = dev._load_manifest()
    training_views_enabled = bool(
        branches["real"]["metrics"].get("training_views", {}).get("enabled")
    )
    recorder = _PoseRecorder(
        dev._preprocess_rows, posedegen, target_frac=target_frac
    )
    dev._preprocess_rows = recorder
    try:
        data, _contexts, _audit = dev._prepare_data(
            source,
            manifest,
            patch_length=config.patch_length,
            patch_count=config.patch_count,
            target_frac=target_frac,
            train_lengths=dev.LENGTHS if training_views_enabled else None,
        )
    finally:
        dev._preprocess_rows = recorder._inner
    pose = _recorded_pose(recorder, data)

    runner.seed_everything(artifact.seed)
    nets = {
        name: branch["net"].to(device).eval()
        for name, branch in branches.items()
    }
    branch_embeddings = {
        name: {
            split: embed_all(
                net, data[f"x{split}"], data[f"f{split}"], device
            ).astype(np.float32, copy=False)
            for split in (CENTER_SPLIT, PROTOTYPE_SPLIT, METRIC_SPLIT)
        }
        for name, net in nets.items()
    }
    for name, splits in branch_embeddings.items():
        for split, value in splits.items():
            if not np.isfinite(value).all():
                raise RuntimeError(f"{name} branch produced non-finite {split}")

    # Density contract: centers are fit on training rows and nothing else.
    real_center = assemble.fit_center(branch_embeddings["real"])
    complex_center = assemble.fit_center(branch_embeddings["complex"])
    fused = {
        split: assemble.fuse_numpy(
            branch_embeddings["real"][split],
            branch_embeddings["complex"][split],
            real_center,
            complex_center,
            weight_real=artifact.weight_real,
        )
        for split in (CENTER_SPLIT, PROTOTYPE_SPLIT, METRIC_SPLIT)
    }
    # Rank contract: prototypes are fit on enrollment rows and nothing else.
    prototypes = assemble.fit_prototypes(
        fused, np.asarray(data["yen"], dtype=np.int64), int(data["n_classes"])
    )

    reproduction = {
        "real_center_max_abs_error": float(
            np.max(np.abs(real_center - artifact.real_center))
        ),
        "complex_center_max_abs_error": float(
            np.max(np.abs(complex_center - artifact.complex_center))
        ),
        "prototypes_max_abs_error": float(
            np.max(np.abs(prototypes - artifact.prototypes))
        ),
        "feature_mean_max_abs_error": float(
            np.max(
                np.abs(
                    np.asarray(data["fmean"], dtype=np.float32)
                    - artifact.feature_mean
                )
            )
        ),
        "feature_std_max_abs_error": float(
            np.max(
                np.abs(
                    np.asarray(data["fstd"], dtype=np.float32)
                    - artifact.feature_std
                )
            )
        ),
    }
    worst = max(reproduction.values())
    if worst > tolerance:
        raise RuntimeError(
            f"v3 fusion did not reproduce from {artifact.directory}: "
            f"{reproduction}"
        )
    reproduction["tolerance"] = float(tolerance)
    reproduction["bit_identical_to_artifact"] = bool(worst == 0.0)

    return Populations(
        data=data,
        nets=nets,
        branch_embeddings=branch_embeddings,
        fused=fused,
        real_center=real_center,
        complex_center=complex_center,
        prototypes=prototypes,
        reproduction=reproduction,
        branch_hashes={
            name: dict(branch["hashes"]) for name, branch in branches.items()
        },
        pose=pose,
    )


# ---------------------------------------------------------------------------
# development novelty
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NoveltyFixture:
    packed: np.ndarray
    features: np.ndarray
    pose: np.ndarray


def generate_novelty(
    seeds: Sequence[int],
    *,
    n_each: int,
    lengths: Sequence[int],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    patch_length: int,
    patch_count: int,
    target_frac: float,
    posedegen: PoseDegeneracySource,
    families: Sequence[str] = NOVELTY_FAMILIES,
    progress: bool = False,
) -> tuple[dict[int, dict[str, dict[int, NoveltyFixture]]], dict[str, Any]]:
    """Generate dev novelty once at the longest prefix, then observe prefixes.

    Mirrors :func:`fit_v3_openset.generate_novelty` exactly, and additionally
    derives the pose-degeneracy features from each observation at the same
    length-dependent resolution floor the frontend used for that observation,
    so a prefix sweep exercises the degeneracy features as well as the packed
    patches.
    """
    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values or len(set(seed_values)) != len(seed_values):
        raise ValueError("novelty seeds must be a non-empty set of distinct ints")
    prefix_lengths = validate_prefix_lengths(lengths)
    if int(n_each) <= 0:
        raise ValueError("n_each must be positive")
    family_names = tuple(str(name) for name in families)
    unknown = [name for name in family_names if name not in NOVELTY]
    if not family_names or unknown:
        raise ValueError(f"unknown novelty families: {unknown}")
    mean = np.asarray(feature_mean, dtype=np.float32)
    std = np.asarray(feature_std, dtype=np.float32)
    if mean.shape != std.shape or not np.all(std > 0.0):
        raise ValueError("feature standardization arrays are invalid")

    longest = max(prefix_lengths)
    fixtures: dict[int, dict[str, dict[int, NoveltyFixture]]] = {}
    provenance: dict[str, Any] = {}
    for seed in seed_values:
        rng = np.random.default_rng(int(seed))
        fixtures[seed] = {}
        seed_provenance: dict[str, Any] = {}
        for family in family_names:
            generator = NOVELTY[family]
            packed_by_length: dict[int, list[np.ndarray]] = {
                length: [] for length in prefix_lengths
            }
            feature_by_length: dict[int, list[np.ndarray]] = {
                length: [] for length in prefix_lengths
            }
            pose_by_length: dict[int, list[np.ndarray]] = {
                length: [] for length in prefix_lengths
            }
            longest_digest = hashlib.sha256()
            prefix_digest = {
                length: hashlib.sha256() for length in prefix_lengths
            }
            for row in range(int(n_each)):
                raw = generator(longest, rng)
                if len(raw) != longest:
                    raise RuntimeError("novelty generator returned wrong length")
                base._complex_digest(longest_digest, raw)
                for length in prefix_lengths:
                    observation = raw[:length]
                    base._complex_digest(prefix_digest[length], observation)
                    packed, raw_features, _context = td_preprocess.preprocess(
                        observation,
                        patch_length=int(patch_length),
                        patch_count=int(patch_count),
                        target_frac=float(target_frac),
                    )
                    packed_by_length[length].append(packed)
                    feature_by_length[length].append(raw_features)
                    pose_by_length[length].append(
                        pose_row(
                            posedegen,
                            observation,
                            patch_length=int(patch_length),
                            target_frac=float(target_frac),
                        )
                    )
                if progress and (row + 1) % 100 == 0:
                    print(
                        f"  [novelty {seed}/{family}] {row + 1}/{n_each}",
                        flush=True,
                    )
            fixtures[seed][family] = {}
            for length in prefix_lengths:
                packed = np.stack(packed_by_length[length]).astype(np.float32)
                raw_features = np.stack(feature_by_length[length]).astype(
                    np.float32
                )
                pose = np.stack(pose_by_length[length]).astype(np.float64)
                if pose.shape != (len(packed), posedegen.feature_count):
                    raise RuntimeError(
                        f"novelty {seed}/{family}/N{length}: pose-degeneracy "
                        f"shape {pose.shape} is not aligned with {len(packed)} "
                        "packed rows"
                    )
                if not np.isfinite(pose).all():
                    raise RuntimeError(
                        f"novelty {seed}/{family}/N{length}: pose-degeneracy "
                        "produced a non-finite value"
                    )
                fixtures[seed][family][length] = NoveltyFixture(
                    packed,
                    ((raw_features - mean) / std).astype(np.float32),
                    pose,
                )
            seed_provenance[family] = {
                "n": int(n_each),
                "generated_once_at": int(longest),
                "longest_capture_sha256": longest_digest.hexdigest(),
                "prefix_sha256": {
                    str(length): prefix_digest[length].hexdigest()
                    for length in prefix_lengths
                },
            }
        provenance[str(seed)] = seed_provenance
    return fixtures, provenance


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


@dataclass
class Rejector:
    """Everything needed to score a row without refitting anything."""

    nets: dict[str, torch.nn.Module]
    real_center: np.ndarray
    complex_center: np.ndarray
    weight_real: float
    prototypes: np.ndarray
    lof_components: list[tuple[str, float, KnownOnlyLOFOpenSet]]
    policy: PoseDegenOpenSet


def score_rows(
    rejector: Rejector,
    packed: np.ndarray,
    features: np.ndarray,
    pose: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(closed_prediction, rejection_score)`` for raw packed rows."""
    real_embedding = embed_all(
        rejector.nets["real"], packed, features, device
    ).astype(np.float32, copy=False)
    complex_embedding = embed_all(
        rejector.nets["complex"], packed, features, device
    ).astype(np.float32, copy=False)
    fused = assemble.fuse_numpy(
        real_embedding,
        complex_embedding,
        rejector.real_center,
        rejector.complex_center,
        weight_real=rejector.weight_real,
    )
    prediction, _distance = nearest(fused, rejector.prototypes)
    prediction = np.asarray(prediction, dtype=np.int64)
    lof_rank = score_branch_lof(
        rejector.lof_components,
        {"real": real_embedding, "complex": complex_embedding},
    )
    score = rejector.policy.score(lof_rank, packed, pose, prediction)
    assert_closed_label_unchanged(fused, rejector.prototypes, prediction, score)
    return prediction, score


def _source_hashes(posedegen: PoseDegeneracySource) -> dict[str, str]:
    """Base chain plus this variant and the pose-degeneracy feature module."""
    hashes = dict(base._source_hashes())
    hashes["fit_v3_openset_posedegen.py"] = _sha256(Path(__file__).resolve())
    if posedegen.source_sha256 is not None:
        hashes[Path(str(posedegen.source_path)).name] = posedegen.source_sha256
    return hashes


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

WEIGHT_SELECTION_PROTOCOL = (
    "The pose-degeneracy blend weight is a hyperparameter with no default. The "
    "existing 0.80/0.20 split was chosen on novelty design seed 20260938, which "
    "is spent, so it carries no authority over a third term. A defensible "
    "weight must be chosen on a FRESH design novelty seed (proposed "
    f"{PROPOSED_DESIGN_NOVELTY_SEED}) with --role design, and only then "
    "validated with --role validate on the untouched seeds "
    f"{list(RESERVED_VALIDATION_SEEDS)}. Those two seeds are the validation "
    "evidence: choosing or adjusting the weight against them would convert "
    "validation evidence into training evidence, which evidence rule 4 forbids."
)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output = reject_sealed_path(Path(args.output_dir), "output")

    # Fail fast, before any data is touched: an absent or non-conforming
    # pose-degeneracy module must stop the run, not quietly reduce it to the
    # frozen two-term policy.
    posedegen = load_posedegen_module(
        getattr(args, "posedegen_module", POSEDEGEN_MODULE_DEFAULT)
    )
    weight = validate_posedegen_weight(getattr(args, "posedegen_weight", None))
    role, design_seed, novelty_seeds = validate_seed_plan(
        getattr(args, "role", None),
        getattr(args, "design_novelty_seed", None),
        args.novelty_seeds,
    )
    prefix_lengths = validate_prefix_lengths(args.prefix_lengths)
    if int(args.novelty_n) <= 0:
        raise ValueError("--novelty-n must be positive")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} is not empty")

    artifact = load_fusion_artifact(Path(args.fusion_dir))
    device = resolve_device(args.device, artifact.device)
    print(
        f"[v3 openset posedegen] role={role} weight={weight:.6f} "
        f"design_seed={design_seed} fusion={artifact.directory.name} "
        f"sha={artifact.directory_sha256[:16]} device={device}",
        flush=True,
    )

    populations = rebuild_populations(
        artifact,
        device,
        posedegen,
        tolerance=float(args.reproduction_tolerance),
    )
    data = populations.data
    pose = populations.pose
    print(
        "[v3 openset posedegen] rebuilt populations; reproduction "
        f"{populations.reproduction}",
        flush=True,
    )
    print(
        "[v3 openset posedegen] pose-degeneracy features "
        f"{posedegen.module_name}@{posedegen.version} "
        f"({posedegen.feature_count} features)",
        flush=True,
    )

    # Density on TRAINING rows only.
    lof_components = fit_branch_lof(populations.branch_embeddings)

    enrollment_lof = score_branch_lof(
        lof_components,
        {
            branch: populations.branch_embeddings[branch][PROTOTYPE_SPLIT]
            for branch in ("real", "complex")
        },
    )
    enrollment_prediction, _distance = nearest(
        populations.fused[PROTOTYPE_SPLIT], populations.prototypes
    )
    enrollment_prediction = np.asarray(enrollment_prediction, dtype=np.int64)

    # Ranks and the q95 threshold on ENROLLMENT rows only.
    policy = fit_policy(
        data["xtr"],
        np.asarray(data["ytr"], dtype=np.int64),
        pose[CENTER_SPLIT],
        data["xen"],
        enrollment_prediction,
        enrollment_lof,
        pose[PROTOTYPE_SPLIT],
        n_classes=int(data["n_classes"]),
        posedegen_weight=weight,
        posedegen_feature_names=posedegen.feature_names,
        posedegen_version=posedegen.version,
    )

    rejector = Rejector(
        nets=populations.nets,
        real_center=populations.real_center,
        complex_center=populations.complex_center,
        weight_real=artifact.weight_real,
        prototypes=populations.prototypes,
        lof_components=lof_components,
        policy=policy,
    )

    # The immutable selection population is scored, never fit.
    selection_lof = score_branch_lof(
        lof_components,
        {
            branch: populations.branch_embeddings[branch][METRIC_SPLIT]
            for branch in ("real", "complex")
        },
    )
    selection_prediction, _distance = nearest(
        populations.fused[METRIC_SPLIT], populations.prototypes
    )
    selection_prediction = np.asarray(selection_prediction, dtype=np.int64)
    known_scores = policy.score(
        selection_lof, data["xva"], pose[METRIC_SPLIT], selection_prediction
    )
    assert_closed_label_unchanged(
        populations.fused[METRIC_SPLIT],
        populations.prototypes,
        selection_prediction,
        known_scores,
    )
    known_labels = np.asarray(data["yva"], dtype=np.int64)
    known_fur = float(np.mean(known_scores > policy.threshold))
    print(
        f"[v3 openset posedegen] threshold={policy.threshold:.6f} "
        f"known FUR={known_fur:.6f}",
        flush=True,
    )

    fixtures, fixture_provenance = generate_novelty(
        novelty_seeds,
        n_each=int(args.novelty_n),
        lengths=prefix_lengths,
        feature_mean=np.asarray(data["fmean"], dtype=np.float32),
        feature_std=np.asarray(data["fstd"], dtype=np.float32),
        patch_length=int(data["cache_contract"]["config"]["patch_length"]),
        patch_count=int(data["cache_contract"]["config"]["patch_count"]),
        target_frac=float(data["cache_contract"]["config"]["target_frac"]),
        posedegen=posedegen,
        progress=True,
    )

    by_seed: dict[str, Any] = {}
    flat_rows: list[dict[str, Any]] = []
    for seed in novelty_seeds:
        by_length: dict[str, Any] = {}
        for length in prefix_lengths:
            row: dict[str, Any] = {}
            family_scores: dict[str, np.ndarray] = {}
            for family in NOVELTY_FAMILIES:
                fixture = fixtures[seed][family][length]
                _prediction, score = score_rows(
                    rejector,
                    fixture.packed,
                    fixture.features,
                    fixture.pose,
                    device,
                )
                family_scores[family] = score
                row[family] = family_metrics(
                    score, known_scores, policy.threshold
                )
            row["overall"] = family_metrics(
                np.concatenate(
                    [family_scores[family] for family in NOVELTY_FAMILIES]
                ),
                known_scores,
                policy.threshold,
            )
            row["known_false_unknown_rate"] = known_fur
            by_length[str(length)] = row
            flat_rows.append(row)
            print(
                f"  [seed {seed} N{length}] noise "
                f"{row['noise']['auroc']:.6f}/"
                f"{row['noise']['threshold_recall']:.6f} chirp "
                f"{row['chirp']['auroc']:.6f}/"
                f"{row['chirp']['threshold_recall']:.6f} overall "
                f"{row['overall']['auroc']:.6f}",
                flush=True,
            )
        by_seed[str(seed)] = by_length

    summary = gate_summary(flat_rows, known_fur)

    output.mkdir(parents=True, exist_ok=True)
    policy_path = output / "v3_open_policy_posedegen.npz"
    np.savez_compressed(policy_path, **policy.to_payload())
    lof_path = output / "v3_branch_lof_components.npz"
    save_branch_lof(lof_path, lof_components)

    prefix = "design_selection" if role == "design" else "development_openset"
    report = {
        "status": (
            f"{prefix}_pass" if summary["all_pass"] else f"{prefix}_fail"
        ),
        "role": role,
        "evidence_role": (
            "hyperparameter selection on a fresh design novelty seed; the gate "
            "block below is NOT validation evidence and must not be quoted as "
            "a result"
            if role == "design"
            else "validation on untouched development novelty seeds; the gate "
            "block below is the evidence"
        ),
        "gates_are_evidence": bool(role == "validate"),
        "development_only": True,
        "release_evidence": False,
        "sealed_release_data_used": 0,
        "consumed_test_rows_used": 0,
        "sealed_release_paths_read": 0,
        "release_seed_20260731_used": False,
        "closed_label_altered": False,
        "closed_label_contract": (
            "the rejector is additive only; the nearest-prototype label is "
            "recomputed after scoring and asserted bit-identical for every "
            "scored population, known and novel. The pose-degeneracy features "
            "are consumed by the rejector alone and never reach the encoder, "
            "the fusion, or the prototype distance, so no scale dependence is "
            "reintroduced into the closed-set classifier."
        ),
        "fusion": {
            "directory": str(artifact.directory),
            "directory_sha256": artifact.directory_sha256,
            "file_sha256": artifact.file_sha256,
            "weight_real": artifact.weight_real,
            "seed": artifact.seed,
            "recorded_device": artifact.device,
            "closed_selection_balanced_accuracy": artifact.metrics[
                "closed_selection"
            ]["balanced_accuracy"],
            "source_branches": {
                name: {
                    "directory": artifact.metrics["source_branches"][name][
                        "directory"
                    ],
                    "sha256": populations.branch_hashes[name],
                }
                for name in ("real", "complex")
            },
            "reproduction": populations.reproduction,
        },
        "fitting_contract": {
            "density_population": "training only",
            "density_rows": int(len(data["xtr"])),
            "density_components": [
                "per-class time-domain geometry (KnownOnlyGeometry)",
                "per-class pose-degeneracy statistics (KnownOnlyPoseDegeneracy)",
                "branch LOF reference population and k-distances",
            ],
            "rank_and_threshold_population": "enrollment only",
            "rank_rows": int(len(data["xen"])),
            "selection_population_role": (
                "scored for metrics only; never fit, calibrated, or selected on"
            ),
            "novelty_population_role": (
                "scored for metrics only; never fit, ranked, or thresholded. "
                + (
                    "In this design run the novelty seed is the declared design "
                    "seed and its scores may inform the choice of "
                    "--posedegen-weight; that is why this report is not "
                    "evidence."
                    if role == "design"
                    else "In this validation run the novelty seeds never inform "
                    "any policy choice."
                )
            ),
            "policy_searched_here": False,
            "sealed_release_rows_loaded": 0,
            "consumed_test_rows_loaded": 0,
            "reproduction_tolerance": float(args.reproduction_tolerance),
        },
        "policy": {
            "schema": int(POSEDEGEN_POLICY_SCHEMA),
            "kind": POSEDEGEN_POLICY_KIND,
            "base_policy_kind": FROZEN_POLICY_KIND,
            "geometry_feature": FROZEN_GEOMETRY_FEATURE,
            "branch_lof_rank_weight": float(BRANCH_LOF_RANK_WEIGHT),
            "geometry_rank_weight": float(GEOMETRY_RANK_WEIGHT),
            "posedegen_rank_weight": float(policy.posedegen_weight),
            "blend": (
                "(1 - posedegen_weight) * (branch_lof_rank_weight * lof_rank + "
                "geometry_rank_weight * geometry_rank) + posedegen_weight * "
                "posedegen_rank, then ranked against enrollment once more"
            ),
            "frozen_pair_proportions_preserved": True,
            "reduces_to_frozen_policy_at_weight_zero": True,
            "posedegen_reducer": POSEDEGEN_REDUCER,
            "posedegen_reducer_source": (
                "module constant with no command-line flag, for the same reason "
                "the branch LOF shape has none: an operator must not be able to "
                "search it"
            ),
            "threshold_quantile": float(THRESHOLD_QUANTILE),
            "threshold": float(policy.threshold),
            "branch_lof_components": [
                {"branch": branch, "weight": weight_, "neighbors": neighbors}
                for branch, weight_, neighbors in BRANCH_LOF
            ],
            "branch_lof_hyperparameters_source": (
                "inherited from the v2 ensemble and re-fit on v3 embeddings; "
                "not re-searched here"
            ),
            "cannot_change_closed_label": True,
        },
        "hyperparameter": {
            "posedegen_weight": float(weight),
            "posedegen_weight_has_default": False,
            "design_novelty_seed": int(design_seed),
            "design_novelty_seed_is_fresh": bool(
                design_seed not in RESERVED_VALIDATION_SEEDS
                and design_seed != SPENT_DESIGN_NOVELTY_SEED
            ),
            "proposed_design_novelty_seed": int(PROPOSED_DESIGN_NOVELTY_SEED),
            "spent_design_novelty_seed": int(SPENT_DESIGN_NOVELTY_SEED),
            "reserved_validation_seeds": list(RESERVED_VALIDATION_SEEDS),
            "selection_protocol": WEIGHT_SELECTION_PROTOCOL,
        },
        "posedegen": {
            **posedegen.provenance(),
            "population_sha256": {
                "training": _matrix_sha256(pose[CENTER_SPLIT]),
                "enrollment": _matrix_sha256(pose[PROTOTYPE_SPLIT]),
                "selection": _matrix_sha256(pose[METRIC_SPLIT]),
            },
            "rationale": (
                "estimate_band already computes the degeneracy and confidence "
                "of the pose it returns and the invariant frontend discards it; "
                "the additive rejector may read it because it cannot change the "
                "closed label"
            ),
        },
        "seeds": {
            "model_seed": artifact.seed,
            "fusion_artifact_seed": artifact.seed,
            "novelty_seeds": list(novelty_seeds),
            "design_novelty_seed": int(design_seed),
            "spent_design_novelty_seed_excluded": SPENT_DESIGN_NOVELTY_SEED,
            "reserved_validation_seeds": list(RESERVED_VALIDATION_SEEDS),
            "sealed_release_seed_excluded": SEALED_RELEASE_SEED,
            "release_seed_not_spent": RELEASE_SEED_NEVER_SPENT_HERE,
        },
        "protocol": {
            "novelty_families": list(NOVELTY_FAMILIES),
            "novelty_n_each": int(args.novelty_n),
            "prefix_lengths": list(prefix_lengths),
            "generation": (
                f"each novelty row generated exactly once at N{max(prefix_lengths)}; "
                "shorter lengths are exact raw prefixes of the same capture"
            ),
            "novelty_frontend": td_preprocess.PREPROCESS_VERSION,
            "known_reference": (
                "immutable development-selection population at its native "
                "N16384; known FUR is therefore constant across novelty "
                "prefix lengths"
            ),
            "known_rows": int(len(known_scores)),
            "known_classes": int(data["n_classes"]),
        },
        "known": {
            "n": int(len(known_scores)),
            "false_unknown_rate": known_fur,
            "closed_selection_accuracy": float(
                np.mean(selection_prediction == known_labels)
            ),
            "score_median": float(np.median(known_scores)),
        },
        "novelty": by_seed,
        "fixture_provenance": fixture_provenance,
        **summary,
        "artifacts": {
            policy_path.name: _sha256(policy_path),
            lof_path.name: _sha256(lof_path),
        },
        "source_sha256": _source_hashes(posedegen),
        "data_audit": data["data_audit"],
        "source_index_contract": {
            "contract": data["cache_contract"],
            "role": (
                "identifies the raw corpus and exposed train/enroll/selection "
                "indices only; every tensor here was rebuilt with the v3 "
                "time-domain frontend and the v3 fusion"
            ),
        },
        "provenance": {
            "command": [sys.executable, *sys.argv],
            "cwd": str(Path.cwd()),
            "device": str(device),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "wall_clock_s": time.perf_counter() - started,
    }
    report_path = output / "openset_metrics.json"
    write_json(report_path, report)
    print(
        f"[v3 openset posedegen] wrote {report_path} status={report['status']}",
        flush=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog=WEIGHT_SELECTION_PROTOCOL,
    )
    parser.add_argument(
        "--fusion-dir",
        required=True,
        help="a v3 fusion artifact directory written by assemble_v3_fusion.py",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--role",
        required=True,
        choices=ROLES,
        help=(
            "'design' draws novelty at the declared design seed and marks the "
            "report as hyperparameter selection, not evidence; 'validate' draws "
            "novelty at untouched validation seeds and refuses any overlap with "
            "the design seed"
        ),
    )
    parser.add_argument(
        "--posedegen-weight",
        required=True,
        type=float,
        default=None,
        help=(
            "blend weight on the pose-degeneracy rank, in [0, 1). REQUIRED with "
            "no default: a default would be an undocumented tuning decision. "
            "Choose it on a fresh design novelty seed (proposed "
            f"{PROPOSED_DESIGN_NOVELTY_SEED}), never on "
            f"{list(RESERVED_VALIDATION_SEEDS)}."
        ),
    )
    parser.add_argument(
        "--design-novelty-seed",
        required=True,
        type=int,
        default=None,
        help=(
            "the novelty seed --posedegen-weight was (or will be) chosen on. "
            "Emitted into the report. Refused if it is a reserved validation "
            f"seed {list(RESERVED_VALIDATION_SEEDS)}, the spent design seed "
            f"{SPENT_DESIGN_NOVELTY_SEED}, the sealed seed "
            f"{SEALED_RELEASE_SEED}, or the release seed "
            f"{RELEASE_SEED_NEVER_SPENT_HERE}."
        ),
    )
    parser.add_argument(
        "--posedegen-module",
        default=POSEDEGEN_MODULE_DEFAULT,
        help=(
            "import name of the pose-degeneracy feature module written by the "
            "sibling task"
        ),
    )
    parser.add_argument(
        "--device",
        choices=("artifact", "auto", "cpu", "mps"),
        default="artifact",
        help="'artifact' reuses the device recorded in the fusion artifact",
    )
    parser.add_argument(
        "--novelty-seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "novelty seeds to draw. With --role validate this defaults to the "
            f"untouched pair {list(RESERVED_VALIDATION_SEEDS)}. With --role "
            "design it must be given explicitly and must exclude that pair."
        ),
    )
    parser.add_argument("--novelty-n", type=int, default=DEFAULT_NOVELTY_N)
    parser.add_argument(
        "--prefix-lengths",
        type=int,
        nargs="+",
        default=list(DEFAULT_PREFIX_LENGTHS),
    )
    parser.add_argument(
        "--reproduction-tolerance",
        type=float,
        default=DEFAULT_REPRODUCTION_TOLERANCE,
        help=(
            "maximum tolerated disagreement between the rebuilt fusion and the "
            "centers/prototypes stored in the fusion artifact"
        ),
    )
    return parser


def resolve_novelty_seeds(args: argparse.Namespace) -> list[int]:
    """Apply the role-dependent default for ``--novelty-seeds``.

    A design run has no default: the seeds it draws are a decision, and the
    validation pair must never be the fallback for one.
    """
    if args.novelty_seeds is not None:
        return [int(seed) for seed in args.novelty_seeds]
    if validate_role(args.role) == "design":
        raise ValueError(
            "--role design requires explicit --novelty-seeds; the validation "
            f"pair {list(RESERVED_VALIDATION_SEEDS)} is never a design default"
        )
    return list(RESERVED_VALIDATION_SEEDS)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    args.novelty_seeds = resolve_novelty_seeds(args)
    return run(args)


if __name__ == "__main__":
    main()
