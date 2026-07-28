"""Export the v3 STAGED open-set decision layer for the browser, with parity.

What this produces (staging only, never release):

``artifacts/staging/time_domain_v3_openset/``
  ``time-domain-openset-weights-v1.json``
      Stage 1: the three per-length noise-prefilter bundles (the tightened
      0.01-budget set ``noise_prefilter_fit20261001_budget001``) converted to
      plain JSON -- 7 means, 7 scales, 7 coefficients, an intercept and a
      score-space threshold per capture length.
      Stage 2: the fitted branch-LOF ensemble and the frozen stage-2 policy
      from the staged artifact (``v3_branch_lof_components.npz`` /
      ``v3_open_policy_stage_two.npz``), including the single frozen geometry
      feature column, both enrollment calibrations and the stage-2 q95.
      COMPOSITE (staged policy version 2): the survivor composite policy
      (``v3_staged_composite_policy.npz``) -- the enrollment-survivor stage-1
      calibration, the composite calibration, the composite q95 threshold and
      the explicit ``policy_version`` the TypeScript runtime refuses to run
      without.
  ``time-domain-classifier-weights-v1.json``
      The decision-layer half of the v3 runtime bundle
      (``v3_runtime_bundle_seed20260730``): feature standardization, fusion
      centers/constants, class order and fused prototypes.  Encoder weights
      are the sibling exporter's job and are deliberately NOT here.
  ``time-domain-openset-parity-v1.json``
      Known + noise + chirp rows pushed through the STAGED Python path with
      every per-stage intermediate and the final decision, INCLUDING which
      stage rejected each row.  This is the fixture the TypeScript suite
      asserts against.
  ``manifest.json``
      SHA-256 of every input consumed and every output written.

Evidence hygiene:

- Novelty parity rows are drawn at :data:`PARITY_NOVELTY_SEED`, which lies
  outside both the novelty-seed namespace (20260900-20260999) and the
  prefilter fit-only band (20261000-20261999).  Parity fixtures are not
  evaluation evidence and must never be allowed to look like it.
- Known parity rows come from the development SELECTION split (scored, never
  fit).  No sealed or consumed path is read; ``assemble.reject_sealed_path``
  guards every directory argument.
- Release seed 20260731 is neither used nor reachable from here.

Determinism: JSON is written with ``sort_keys=True``, no timestamps, no
absolute paths in the payloads, and float repr is Python's shortest
round-trip, so byte-identical inputs give byte-identical outputs.

Self-checks before anything is written:

- every runtime-bundle asset SHA-256 is re-verified against its manifest;
- the fusion artifact's centers/prototypes/feature stats must be exactly the
  runtime bundle's (they are the same fit);
- the staged stage-2 score assembled from intermediates here must EXACTLY
  equal ``fit_v3_openset.score_rows`` on every surviving row, so the fixture
  cannot drift from the real Python path;
- the closed label of every survivor is asserted unchanged by the rejector
  (``assert_closed_label_unchanged`` runs inside ``score_rows``).

Run with the training venv:

    .venv-training/bin/python \
        training/zplane_ab/v2_full_variation/v3_scale/export_v3_openset_browser_assets.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
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
import fit_v3_openset_staged as staged  # noqa: E402
import invariant_patch_data as invariant_data  # noqa: E402
import noise_prefilter  # noqa: E402
import pose_degeneracy as pdg  # noqa: E402
import run_time_domain_dev as dev  # noqa: E402
import time_domain_geometry as geometry  # noqa: E402
import time_domain_invariant_patch_preprocess as td_preprocess  # noqa: E402
import known_only_patch_openset as kop  # noqa: E402
from known_only_patch_openset import KnownOnlyLOFOpenSet  # noqa: E402
from openset_eval import NOVELTY  # noqa: E402
from train import embed_all, nearest  # noqa: E402
from v3_time_domain_openset import (  # noqa: E402
    FROZEN_GEOMETRY_FEATURE,
    FROZEN_GEOMETRY_WEIGHT,
    FROZEN_THRESHOLD_QUANTILE,
    FROZEN_V2_WEIGHT,
    FrozenV3OpenSet,
    empirical_rank,
)
from v3_time_domain_openset import (  # noqa: E402
    time_domain_geometry as stage_two_geometry,
)


OPENSET_SCHEMA = "atomos.v3.time-domain-openset.staged"
# Version 2: the composite survivor score (staged policy version 2).  The
# TypeScript runtime refuses any other schema version, exactly as it refuses
# any other staged policy version.
OPENSET_SCHEMA_VERSION = 2
CLASSIFIER_SCHEMA = "atomos.v3.time-domain-invariant-fusion.browser-decision"
CLASSIFIER_SCHEMA_VERSION = 1
PARITY_SCHEMA = "time-domain-openset-parity-v1"

WEIGHTS_NAME = "time-domain-openset-weights-v1.json"
CLASSIFIER_WEIGHTS_NAME = "time-domain-classifier-weights-v1.json"
PARITY_NAME = "time-domain-openset-parity-v1.json"
MANIFEST_NAME = "manifest.json"

#: Parity probe only.  Outside the novelty namespace (20260900-20260999) and
#: the prefilter fit-only band (20261000-20261999); it is not evaluation
#: evidence and is recorded as such in the fixture.
PARITY_NOVELTY_SEED = 20269101
NOVELTY_ROWS_PER_FAMILY = 6
FIXTURE_LENGTHS = (4096, 8192)

DEFAULT_BUNDLE = (
    V2 / "artifacts" / "invariant_patch" / "v3_scale"
    / "v3_runtime_bundle_seed20260730"
)
DEFAULT_FUSION = (
    V2 / "artifacts" / "invariant_patch" / "v3_scale"
    / "v3_fusion_multilength_seed20260730"
)
DEFAULT_STAGED = (
    V2 / "artifacts" / "invariant_patch" / "v3_scale"
    / "staged_validate_composite_budget001_seed20260730"
)
DEFAULT_PREFILTERS = (
    V2 / "artifacts" / "invariant_patch" / "v3_scale"
    / "noise_prefilter_fit20261001_budget001" / "bundles"
)
DEFAULT_OUTPUT = V2 / "artifacts" / "staging" / "time_domain_v3_openset"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, Path):
        raise TypeError("paths must not leak into deterministic payloads")
    return value


def write_json(path: Path, payload: Any) -> None:
    temporary = Path(path).with_name(f".{Path(path).name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            _jsonable(payload), handle, indent=1, sort_keys=True, allow_nan=False
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_parity_seed(seed: int) -> int:
    value = int(seed)
    staged._refuse_ledger_seed(value, "a parity fixture draw")
    low, high = noise_prefilter.NOVELTY_SEED_NAMESPACE
    if low <= value <= high:
        raise ValueError(
            f"parity seed {value} lies in the novelty namespace {low}-{high}; "
            "a parity fixture must not consume evaluation evidence"
        )
    low, high = noise_prefilter.FITTING_SEED_NAMESPACE
    if low <= value <= high:
        raise ValueError(
            f"parity seed {value} lies in the prefilter fit-only band "
            f"{low}-{high}; a parity fixture must not look like a fitting draw"
        )
    return value


# ---------------------------------------------------------------------------
# loading and cross-checking the fitted state
# ---------------------------------------------------------------------------


def load_bundle(bundle_dir: Path) -> dict[str, Any]:
    """Load the v3 runtime bundle, verifying every asset SHA-256."""
    bundle = assemble.reject_sealed_path(Path(bundle_dir), "runtime bundle")
    manifest_path = bundle / "bundle_manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("kind") != "v3-time-domain-centered-invariant-fusion":
        raise ValueError(f"{bundle} is not a v3 runtime bundle")
    if manifest.get("development_only") is not True:
        raise RuntimeError(f"{bundle} must be development_only")
    arrays: dict[str, np.ndarray] = {}
    for name, meta in manifest["assets"].items():
        path = bundle / name
        digest = _sha256(path)
        if digest != meta["sha256"]:
            raise RuntimeError(f"{path} does not match its recorded SHA-256")
        if name.endswith(".npy"):
            arrays[name] = np.load(path)
    with (bundle / "probe_fixture.json").open(encoding="utf-8") as handle:
        probe = json.load(handle)
    return {
        "directory": bundle,
        "manifest": manifest,
        "manifest_sha256": _sha256(manifest_path),
        "arrays": arrays,
        "probe": probe,
    }


def load_stage_two_state(
    staged_dir: Path,
) -> tuple[list[tuple[str, float, KnownOnlyLOFOpenSet]], FrozenV3OpenSet, dict[str, str]]:
    """Rebuild the fitted LOF ensemble and frozen policy from the staged npz.

    The composite survivor policy is deliberately NOT loaded here: this
    3-tuple contract is shared with ``evaluate_v3_release_suite.load_candidate``,
    which performs its own composite load and version refusal.  This module's
    :func:`load_composite_state` is the exporter-side counterpart.
    """
    directory = assemble.reject_sealed_path(Path(staged_dir), "staged artifact")
    lof_path = directory / "v3_branch_lof_components.npz"
    policy_path = directory / "v3_open_policy_stage_two.npz"
    hashes = {
        "v3_branch_lof_components.npz": _sha256(lof_path),
        "v3_open_policy_stage_two.npz": _sha256(policy_path),
    }

    with np.load(lof_path) as payload:
        if str(payload["kind"]) != "v3_branch_lof_rank_ensemble":
            raise ValueError(f"{lof_path} is not a v3 branch LOF ensemble")
        count = int(payload["component_count"])
        components: list[tuple[str, float, KnownOnlyLOFOpenSet]] = []
        for index in range(count):
            prefix = f"component_{index}_"
            scorer = KnownOnlyLOFOpenSet(
                embedding_reference=np.asarray(
                    payload[f"{prefix}embedding_reference"], dtype=np.float64
                ),
                embedding_mean=np.asarray(
                    payload[f"{prefix}embedding_mean"], dtype=np.float64
                ),
                embedding_scale=np.asarray(
                    payload[f"{prefix}embedding_scale"], dtype=np.float64
                ),
                reference_k_distance=np.asarray(
                    payload[f"{prefix}reference_k_distance"], dtype=np.float64
                ),
                reference_local_density=np.asarray(
                    payload[f"{prefix}reference_local_density"], dtype=np.float64
                ),
                calibration=np.asarray(
                    payload[f"{prefix}calibration"], dtype=np.float64
                ),
                calibration_scores=np.asarray(
                    payload[f"{prefix}calibration_scores"], dtype=np.float64
                ),
                neighbors=int(payload[f"{prefix}neighbors"]),
            )
            components.append(
                (
                    str(payload[f"{prefix}branch_source"]),
                    float(payload[f"{prefix}weight"]),
                    scorer,
                )
            )

    expected = [
        (branch, weight, neighbors) for branch, weight, neighbors in base.BRANCH_LOF
    ]
    found = [
        (branch, weight, scorer.neighbors) for branch, weight, scorer in components
    ]
    if expected != found:
        raise RuntimeError(
            f"staged LOF ensemble {found} differs from the inherited shape "
            f"{expected}"
        )

    with np.load(policy_path) as payload:
        policy = FrozenV3OpenSet.from_payload({key: payload[key] for key in payload.files})
    return components, policy, hashes


def load_composite_state(
    staged_dir: Path,
    policy: FrozenV3OpenSet,
) -> tuple[Any, str]:
    """Load and verify the composite survivor policy from the staged npz.

    The composite loader enforces the staged policy version and recomputes
    the q95 threshold from the stored calibration; the stage-2 threshold
    cross-check proves the composite was fit against exactly this stage-2
    state.  Returns ``(composite, sha256)``.
    """
    directory = assemble.reject_sealed_path(Path(staged_dir), "staged artifact")
    composite_path = directory / staged.COMPOSITE_POLICY_FILENAME
    composite = staged.load_composite_policy(
        composite_path,
        expected_stage_two_threshold=float(policy.threshold),
    )
    return composite, _sha256(composite_path)


def compute_branch_lof_raw(
    scorer: KnownOnlyLOFOpenSet, embeddings: np.ndarray
) -> np.ndarray:
    """The raw (pre-rank) LOF value, mirroring ``KnownOnlyLOFOpenSet.score``."""
    standardized = (
        np.asarray(embeddings, dtype=np.float64) - scorer.embedding_mean
    ) / scorer.embedding_scale
    indices, distances = kop._nearest_neighbors(
        standardized, scorer.embedding_reference, scorer.neighbors
    )
    density = 1.0 / (
        np.maximum(distances, scorer.reference_k_distance[indices]).mean(axis=1)
        + 1e-12
    )
    return scorer.reference_local_density[indices].mean(axis=1) / density


# ---------------------------------------------------------------------------
# weight conversion
# ---------------------------------------------------------------------------


def prefilter_model_payload(model: noise_prefilter.NoisePrefilter) -> dict[str, Any]:
    if not model.has_threshold:
        raise ValueError("prefilter bundle has no fitted operating point")
    if tuple(model.feature_names) != noise_prefilter.PREFILTER_FEATURES:
        raise ValueError(
            "prefilter feature order differs from the frozen PREFILTER_FEATURES"
        )
    return {
        "capture_length": int(model.capture_length),
        "feature_names": list(model.feature_names),
        "mean": model.mean,
        "scale": model.scale,
        "coefficients": model.coefficients,
        "intercept": float(model.intercept),
        "threshold_score": float(model.threshold_score),
    }


def composite_policy_payload(composite: Any) -> dict[str, Any]:
    """The `v3_staged_composite_policy.npz` payload as plain JSON.

    Field names keep the npz spelling exactly, so the TypeScript loader's
    validation (policy version refusal, sorted calibrations, recomputed q95)
    mirrors ``fit_v3_openset_staged.load_composite_policy`` key for key.
    """
    return {
        "schema": int(staged.STAGED_POLICY_SCHEMA),
        "kind": staged.STAGED_POLICY_KIND,
        "policy_version": staged.STAGED_POLICY_VERSION,
        "survivor_score": staged.COMPOSITE_SURVIVOR_SCORE,
        "threshold_quantile": float(staged.COMPOSITE_THRESHOLD_QUANTILE),
        "threshold": float(composite.threshold),
        "stage_two_threshold": float(composite.stage_two_threshold),
        "stage_one_calibration_raw": composite.stage_one_calibration_raw,
        "composite_calibration_raw": composite.composite_calibration_raw,
        "enrollment_rows": int(composite.enrollment_rows),
        "enrollment_gated_rows": int(composite.enrollment_gated_rows),
        "enrollment_capture_length": int(composite.enrollment_capture_length),
    }


def openset_weights_payload(
    prefilters: Mapping[int, noise_prefilter.NoisePrefilter],
    prefilter_set_sha: str,
    components: Sequence[tuple[str, float, KnownOnlyLOFOpenSet]],
    policy: FrozenV3OpenSet,
    composite: Any,
    frontend: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    feature_index = policy.geometry_feature_index
    return {
        "schema": OPENSET_SCHEMA,
        "schema_version": OPENSET_SCHEMA_VERSION,
        "status": "staging_not_release",
        "contract": {
            "additive_only": False,
            "changes_closed_label": True,
            "gates_before_classification": True,
            "architecture_contract_change": (
                noise_prefilter.ARCHITECTURE_CONTRACT_CHANGE
            ),
        },
        "frontend": dict(frontend),
        "stage_one": {
            "kind": noise_prefilter.NOISE_PREFILTER_VERSION,
            "feature_names": list(noise_prefilter.PREFILTER_FEATURES),
            "match_frontend_min_bandwidth": bool(
                staged.PREFILTER_MATCH_FRONTEND_MIN_BANDWIDTH
            ),
            "set_sha256": prefilter_set_sha,
            "models": {
                str(length): prefilter_model_payload(prefilters[length])
                for length in sorted(prefilters)
            },
        },
        "stage_two": {
            "kind": staged.STAGED_POLICY_KIND,
            "lof_components": [
                {
                    "branch": branch,
                    "weight": float(weight),
                    "neighbors": int(scorer.neighbors),
                    "mean": scorer.embedding_mean,
                    "scale": scorer.embedding_scale,
                    "reference": scorer.embedding_reference,
                    "reference_k_distance": scorer.reference_k_distance,
                    "reference_local_density": scorer.reference_local_density,
                    "calibration": scorer.calibration,
                }
                for branch, weight, scorer in components
            ],
            "policy": {
                "branch_lof_rank_weight": float(FROZEN_V2_WEIGHT),
                "geometry_weight": float(FROZEN_GEOMETRY_WEIGHT),
                "threshold_quantile": float(FROZEN_THRESHOLD_QUANTILE),
                "threshold": float(policy.threshold),
                "geometry_feature": FROZEN_GEOMETRY_FEATURE,
                "class_geometry_mean": policy.geometry.class_mean[:, feature_index],
                "class_geometry_scale": policy.geometry.class_scale[:, feature_index],
                "geometry_calibration": policy.geometry_calibration_raw,
                "combined_calibration": policy.combined_calibration_raw,
            },
        },
        "composite": composite_policy_payload(composite),
        "provenance": dict(provenance),
    }


def classifier_weights_payload(
    bundle: Mapping[str, Any], provenance: Mapping[str, Any]
) -> dict[str, Any]:
    manifest = bundle["manifest"]
    arrays = bundle["arrays"]
    frontend = manifest["frontend"]
    fusion = manifest["fusion"]
    return {
        "schema": CLASSIFIER_SCHEMA,
        "schema_version": CLASSIFIER_SCHEMA_VERSION,
        "status": "staging_not_release",
        "frontend": {
            "version": str(frontend["version"]),
            "patch_length": int(frontend["patch_length"]),
            "patch_count": int(frontend["patch_count"]),
            "target_frac": float(frontend["target_frac"]),
            "packed_length": int(frontend["packed_length"]),
            "uses_frequency_transform": False,
        },
        "feature_standardization": {
            "mean": arrays["feature_mean.npy"],
            "std": arrays["feature_std.npy"],
        },
        "fusion": {
            "real_center": arrays["real_center.npy"],
            "complex_center": arrays["complex_center.npy"],
            "alpha_real": float(fusion["alpha_real"]),
            "alpha_complex": float(fusion["alpha_complex"]),
            "weight_real": float(fusion["weight_real"]),
            "eps": float(fusion["eps"]),
        },
        "classification": {
            "classes": list(manifest["classification"]["classes"]),
            "prototypes": arrays["fusion_prototypes.npy"],
            "distance": "squared_euclidean",
        },
        "provenance": dict(provenance),
    }


# ---------------------------------------------------------------------------
# the staged parity path (one row at a time, mirrored from Python exactly)
# ---------------------------------------------------------------------------


def stage_one_row(
    prefilters: Mapping[int, noise_prefilter.NoisePrefilter],
    capture: np.ndarray,
    *,
    patch_length: int,
    target_frac: float,
) -> dict[str, Any]:
    length = int(len(capture))
    model = prefilters[length]
    minimum = geometry.minimum_bandwidth_for_active_span(
        length, patch_length, target_frac
    )
    features = pdg.pose_degeneracy_features(capture, min_bandwidth=minimum)
    selected = noise_prefilter.select_features(
        features[None, :], model.feature_names
    )[0]
    score = float(model.score(selected[None, :], capture_length=length)[0])
    gated = bool(model.decide(selected[None, :], capture_length=length)[0])
    rank = float(staged._bounded_rank(np.asarray([score]))[0])
    if gated != (score >= float(model.threshold_score)):
        raise AssertionError("stage-1 gate rule disagreement")
    return {
        "features": selected,
        "score": score,
        "rank": rank,
        "gated": gated,
        "threshold_score": float(model.threshold_score),
    }


def stage_two_row(
    rejector: base.Rejector,
    components: Sequence[tuple[str, float, KnownOnlyLOFOpenSet]],
    policy: FrozenV3OpenSet,
    packed: np.ndarray,
    standardized: np.ndarray,
    classes: Sequence[str],
    device: torch.device,
) -> dict[str, Any]:
    real_embedding = embed_all(
        rejector.nets["real"], packed[None], standardized[None], device
    ).astype(np.float32, copy=False)
    complex_embedding = embed_all(
        rejector.nets["complex"], packed[None], standardized[None], device
    ).astype(np.float32, copy=False)
    fused = assemble.fuse_numpy(
        real_embedding,
        complex_embedding,
        rejector.real_center,
        rejector.complex_center,
        weight_real=rejector.weight_real,
    )
    prediction, distances = nearest(fused, rejector.prototypes)
    predicted = int(prediction[0])

    lof_scores = []
    weighted = 0.0
    for branch, weight, scorer in components:
        embedding = real_embedding if branch == "real" else complex_embedding
        raw = float(compute_branch_lof_raw(scorer, embedding)[0])
        rank = float(scorer.score(np.asarray(embedding, dtype=np.float64))[0])
        weighted += weight * rank
        lof_scores.append(
            {
                "branch": branch,
                "weight": float(weight),
                "neighbors": int(scorer.neighbors),
                "raw": raw,
                "rank": rank,
            }
        )

    statistics, names = stage_two_geometry(packed[None])
    feature_index = policy.geometry_feature_index
    if names[feature_index] != FROZEN_GEOMETRY_FEATURE:
        raise AssertionError("frozen geometry feature index drifted")
    statistic = float(statistics[0, feature_index])
    deviation = float(
        policy.geometry.deviations(packed[None], np.asarray([predicted]))[
            0, feature_index
        ]
    )
    geometry_rank = float(
        empirical_rank(
            np.asarray([deviation]), policy.geometry_calibration_raw
        )[0]
    )
    combined = float(FROZEN_V2_WEIGHT * weighted + FROZEN_GEOMETRY_WEIGHT * geometry_rank)
    score = float(
        empirical_rank(np.asarray([combined]), policy.combined_calibration_raw)[0]
    )

    # The intermediates above must reproduce the REAL Python path exactly.
    check_prediction, check_score = base.score_rows(
        rejector, packed[None], standardized[None], device
    )
    if int(check_prediction[0]) != predicted:
        raise AssertionError("decomposed prediction differs from score_rows")
    if float(check_score[0]) != score:
        raise AssertionError(
            f"decomposed stage-2 score {score} differs from score_rows "
            f"{float(check_score[0])}"
        )

    return {
        "real_embedding": real_embedding[0],
        "complex_embedding": complex_embedding[0],
        "fused_embedding": fused[0],
        "predicted_class_index": predicted,
        "predicted_class_label": str(classes[predicted]),
        "squared_prototype_distances": distances[0],
        "lof": lof_scores,
        "lof_rank": float(weighted),
        "geometry_statistic": statistic,
        "geometry_deviation": deviation,
        "geometry_rank": geometry_rank,
        "combined_raw": combined,
        "score": score,
        # The stage-2 policy's own enrollment q95, recorded for cross-checks;
        # under staged policy version 2 the survivor DECISION compares the
        # COMPOSITE against the composite threshold.
        "threshold": float(policy.threshold),
    }


def staged_parity_row(
    *,
    family: str,
    name: str,
    capture: np.ndarray,
    prefilters: Mapping[int, noise_prefilter.NoisePrefilter],
    rejector: base.Rejector,
    components: Sequence[tuple[str, float, KnownOnlyLOFOpenSet]],
    policy: FrozenV3OpenSet,
    composite: Any,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    classes: Sequence[str],
    patch_length: int,
    patch_count: int,
    target_frac: float,
    device: torch.device,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    capture = np.asarray(capture, dtype=np.complex128)
    stage_one = stage_one_row(
        prefilters,
        capture,
        patch_length=patch_length,
        target_frac=target_frac,
    )
    packed, raw_features, _context = td_preprocess.preprocess(
        capture,
        patch_length=patch_length,
        patch_count=patch_count,
        target_frac=target_frac,
    )
    packed = np.asarray(packed, dtype=np.float32)
    raw_features = np.asarray(raw_features, dtype=np.float32)
    standardized = ((raw_features - feature_mean) / feature_std).astype(np.float32)

    if stage_one["gated"]:
        stage_two = None
        stage_one_survivor_rank = None
        composite_score = None
        staged_score = float(staged.STAGE_ONE_SCORE_OFFSET + stage_one["rank"])
        rejected_stage: int | None = 1
        decision = "noise"
    else:
        stage_two = stage_two_row(
            rejector,
            components,
            policy,
            packed,
            standardized,
            classes,
            device,
        )
        # Staged policy version 2: the survivor decision is the COMPOSITE
        # against the composite q95 threshold.  Both terms come from the
        # loaded CompositeSurvivorPolicy itself, so the fixture cannot drift
        # from the real Python path.
        stage_one_raw = np.asarray([stage_one["score"]], dtype=np.float64)
        stage_one_survivor_rank = float(
            composite.stage_one_survivor_rank(stage_one_raw)[0]
        )
        composite_score = float(
            composite.composite(
                np.asarray([stage_two["score"]], dtype=np.float64),
                stage_one_raw,
            )[0]
        )
        if composite_score != max(stage_two["score"], stage_one_survivor_rank):
            raise AssertionError(
                "decomposed composite differs from CompositeSurvivorPolicy"
            )
        staged_score = composite_score
        rejected = bool(composite_score > composite.threshold)
        rejected_stage = 2 if rejected else None
        decision = (
            "unknown" if rejected else stage_two["predicted_class_label"]
        )

    row = {
        "family": family,
        "name": name,
        "capture_length": int(len(capture)),
        "iq": {
            "in_phase": np.real(capture),
            "quadrature": np.imag(capture),
        },
        "stage_one": stage_one,
        "packed_iq": {
            "in_phase": packed[0],
            "quadrature": packed[1],
        },
        "raw_features": raw_features,
        "standardized_features": standardized,
        "stage_two": stage_two,
        "stage_one_survivor_rank": stage_one_survivor_rank,
        "composite_score": composite_score,
        "staged_threshold": float(composite.threshold),
        "staged_score": staged_score,
        "rejected_stage": rejected_stage,
        "decision_label": decision,
    }
    if extra:
        row.update(dict(extra))
    return row


# ---------------------------------------------------------------------------
# fixture populations
# ---------------------------------------------------------------------------


def select_known_captures(
    seed: int, classes: Sequence[str]
) -> list[dict[str, Any]]:
    """One SELECTION-split capture per class, deterministic, nonzero prefix."""
    source = invariant_data.load(
        patch_length=64,
        patch_count=16,
        target_frac=0.5,
        model_seed=seed,
        build_if_missing=False,
    )
    if list(source["classes"]) != list(classes):
        raise RuntimeError(
            "cache class order differs from the runtime bundle class order"
        )
    manifest = dev._load_manifest()
    raw = dev._raw_memmap(manifest)
    labels = np.asarray(source["yva"], dtype=np.int64)
    indices = np.asarray(source["va_idx"], dtype=np.int64)
    minimum_length = min(FIXTURE_LENGTHS)
    selected: list[dict[str, Any]] = []
    for class_index, class_name in enumerate(classes):
        positions = np.flatnonzero(labels == class_index)
        chosen = None
        for position in positions:
            capture = dev._complex_row(raw, int(indices[position]))
            if float(np.max(np.abs(capture[:minimum_length]))) > 0.0:
                chosen = (int(position), capture)
                break
        if chosen is None:
            raise RuntimeError(
                f"no selection row of class {class_name} has a nonzero "
                f"{minimum_length}-sample prefix"
            )
        position, capture = chosen
        selected.append(
            {
                "class_index": class_index,
                "class_label": str(class_name),
                "selection_position": position,
                "corpus_index": int(indices[position]),
                "capture": capture,
            }
        )
    del raw
    return selected


def draw_novelty_captures(seed: int) -> dict[str, list[np.ndarray]]:
    """Noise and chirp parity rows at the fixture's longest length."""
    validate_parity_seed(seed)
    longest = max(FIXTURE_LENGTHS)
    rng = np.random.default_rng(int(seed))
    captures: dict[str, list[np.ndarray]] = {}
    for family in ("noise", "chirp"):
        generator = NOVELTY[family]
        captures[family] = [
            np.asarray(generator(longest, rng), dtype=np.complex128)
            for _ in range(NOVELTY_ROWS_PER_FAMILY)
        ]
    return captures


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--fusion-dir", type=Path, default=DEFAULT_FUSION)
    parser.add_argument("--staged-dir", type=Path, default=DEFAULT_STAGED)
    parser.add_argument("--prefilter-dir", type=Path, default=DEFAULT_PREFILTERS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing staging output directory's files",
    )
    parser.add_argument(
        "--parity-seed", type=int, default=PARITY_NOVELTY_SEED
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = assemble.reject_sealed_path(Path(args.output_dir), "output")
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{output} is not empty; pass --overwrite to replace its files"
        )
    output.mkdir(parents=True, exist_ok=True)

    bundle = load_bundle(Path(args.bundle_dir))
    manifest = bundle["manifest"]
    classes = list(manifest["classification"]["classes"])
    frontend_meta = manifest["frontend"]
    patch_length = int(frontend_meta["patch_length"])
    patch_count = int(frontend_meta["patch_count"])
    target_frac = float(frontend_meta["target_frac"])

    prefilter_dir = assemble.reject_sealed_path(
        Path(args.prefilter_dir), "prefilter bundles"
    )
    prefilters = noise_prefilter.load_prefilter_set(prefilter_dir)
    prefilter_sha = noise_prefilter.prefilter_set_sha256(prefilter_dir)
    for length in FIXTURE_LENGTHS:
        if length not in prefilters:
            raise RuntimeError(f"no prefilter bundle for capture length {length}")

    components, policy, staged_hashes = load_stage_two_state(
        Path(args.staged_dir)
    )
    composite, composite_sha = load_composite_state(
        Path(args.staged_dir), policy
    )
    staged_hashes = dict(staged_hashes)
    staged_hashes[staged.COMPOSITE_POLICY_FILENAME] = composite_sha
    staged_metrics_path = Path(args.staged_dir) / "openset_metrics.json"
    with staged_metrics_path.open(encoding="utf-8") as handle:
        staged_metrics = json.load(handle)
    recorded_prefilter_dir = staged_metrics.get("stage_one", {}).get("directory")
    if recorded_prefilter_dir is not None and (
        Path(recorded_prefilter_dir).resolve() != prefilter_dir.resolve()
    ):
        raise RuntimeError(
            f"staged artifact used prefilters {recorded_prefilter_dir}, not "
            f"{prefilter_dir}"
        )
    recorded_set_sha = staged_metrics.get("stage_one", {}).get("set_sha256")
    if recorded_set_sha is not None and recorded_set_sha != prefilter_sha:
        raise RuntimeError(
            "prefilter set SHA differs from the staged artifact's record"
        )

    artifact = base.load_fusion_artifact(Path(args.fusion_dir))
    recorded_fusion = staged_metrics.get("fusion", {})
    recorded_fusion_dir = (
        recorded_fusion.get("directory")
        if isinstance(recorded_fusion, Mapping)
        else recorded_fusion
    )
    if recorded_fusion_dir is not None and (
        Path(str(recorded_fusion_dir)).resolve() != artifact.directory.resolve()
    ):
        raise RuntimeError(
            f"staged artifact was fit against {recorded_fusion_dir}, not "
            f"{artifact.directory}"
        )

    # The runtime bundle and the fusion artifact must carry the same fit.
    for bundle_name, fusion_value in (
        ("real_center.npy", artifact.real_center),
        ("complex_center.npy", artifact.complex_center),
        ("fusion_prototypes.npy", artifact.prototypes),
        ("feature_mean.npy", artifact.feature_mean),
        ("feature_std.npy", artifact.feature_std),
    ):
        if not np.array_equal(
            np.asarray(bundle["arrays"][bundle_name], dtype=np.float32),
            np.asarray(fusion_value, dtype=np.float32),
        ):
            raise RuntimeError(
                f"runtime bundle {bundle_name} differs from the fusion artifact"
            )

    device = torch.device("cpu")
    branches = {
        name: assemble.load_branch(
            Path(artifact.metrics["source_branches"][name]["directory"]), name
        )
        for name in ("real", "complex")
    }
    for name, branch in branches.items():
        recorded = artifact.metrics["source_branches"][name]["sha256"]
        if branch["hashes"] != recorded:
            raise RuntimeError(
                f"{name} branch hashes differ from the fusion artifact record"
            )
    nets = {name: branch["net"].to(device).eval() for name, branch in branches.items()}

    rejector = base.Rejector(
        nets=nets,
        real_center=artifact.real_center,
        complex_center=artifact.complex_center,
        weight_real=artifact.weight_real,
        prototypes=artifact.prototypes,
        lof_components=list(components),
        policy=policy,
    )

    feature_mean = np.asarray(bundle["arrays"]["feature_mean.npy"], dtype=np.float32)
    feature_std = np.asarray(bundle["arrays"]["feature_std.npy"], dtype=np.float32)

    frontend_payload = {
        "patch_length": patch_length,
        "patch_count": patch_count,
        "target_frac": target_frac,
        "packed_length": patch_length * patch_count,
    }

    provenance = {
        "runtime_bundle": bundle["directory"].name,
        "runtime_bundle_manifest_sha256": bundle["manifest_sha256"],
        "fusion_artifact": artifact.directory.name,
        "fusion_directory_sha256": artifact.directory_sha256,
        "staged_artifact": Path(args.staged_dir).name,
        "staged_artifact_sha256": staged_hashes,
        "prefilter_bundles": prefilter_dir.parent.name + "/" + prefilter_dir.name,
        "prefilter_set_sha256": prefilter_sha,
        "exporter": Path(__file__).name,
        "exporter_sha256": _sha256(Path(__file__)),
        "development_only": True,
        "release_evidence": False,
        "sealed_release_data_used": 0,
        "consumed_test_rows_used": 0,
        "release_seed_not_spent": base.RELEASE_SEED_NEVER_SPENT_HERE,
    }

    weights = openset_weights_payload(
        prefilters,
        prefilter_sha,
        components,
        policy,
        composite,
        frontend_payload,
        provenance,
    )
    classifier_weights = classifier_weights_payload(bundle, provenance)

    # ---- parity rows ------------------------------------------------------
    known = select_known_captures(artifact.seed, classes)
    novelty = draw_novelty_captures(int(args.parity_seed))

    def process(family: str, name: str, capture: np.ndarray, extra=None):
        return staged_parity_row(
            family=family,
            name=name,
            capture=capture,
            prefilters=prefilters,
            rejector=rejector,
            components=components,
            policy=policy,
            composite=composite,
            feature_mean=feature_mean,
            feature_std=feature_std,
            classes=classes,
            patch_length=patch_length,
            patch_count=patch_count,
            target_frac=target_frac,
            device=device,
            extra=extra,
        )

    rows: list[dict[str, Any]] = []
    for length in FIXTURE_LENGTHS:
        for item in known:
            rows.append(
                process(
                    "known",
                    f"known-{item['class_label']}-N{length}",
                    item["capture"][:length],
                    extra={
                        "class_label": item["class_label"],
                        "class_index": item["class_index"],
                        "corpus_index": item["corpus_index"],
                    },
                )
            )
        for family in ("noise", "chirp"):
            for index, capture in enumerate(novelty[family]):
                rows.append(
                    process(
                        family,
                        f"{family}-{index}-N{length}",
                        capture[:length],
                    )
                )

    # Probe cases from THE parity anchor, re-scored through the staged path.
    probe_rows: list[dict[str, Any]] = []
    probe_cases: list[dict[str, Any]] = []
    for case in bundle["probe"]["cases"]:
        expected = case["expected"]
        probe_cases.append(
            {
                "name": case["name"],
                "length": int(case["length"]),
                "raw": case["raw"],
                "expected": {
                    "raw_features": expected["raw_features"],
                    "standardized_features": expected["standardized_features"],
                    "real_embedding": expected["real_embedding"],
                    "complex_embedding": expected["complex_embedding"],
                    "fused_embedding": expected["fused_embedding"],
                    "squared_prototype_distances": expected[
                        "squared_prototype_distances"
                    ],
                    "closed_winner_index": int(expected["closed_winner_index"]),
                    "closed_label": str(expected["closed_label"]),
                },
            }
        )
        if int(case["length"]) not in prefilters:
            continue
        capture = np.asarray(
            case["raw"]["in_phase"], dtype=np.float64
        ) + 1j * np.asarray(case["raw"]["quadrature"], dtype=np.float64)
        row = process("probe", str(case["name"]), capture)
        if not row["stage_one"]["gated"]:
            if row["stage_two"]["predicted_class_label"] != expected["closed_label"]:
                raise RuntimeError(
                    f"probe case {case['name']} closed label diverged from the "
                    "runtime bundle probe fixture"
                )
        probe_rows.append(row)

    counts = {
        "rows": len(rows) + len(probe_rows),
        "known": sum(1 for row in rows if row["family"] == "known"),
        "noise": sum(1 for row in rows if row["family"] == "noise"),
        "chirp": sum(1 for row in rows if row["family"] == "chirp"),
        "probe": len(probe_rows),
        "gated_stage_one": sum(
            1 for row in rows + probe_rows if row["rejected_stage"] == 1
        ),
        "rejected_stage_two": sum(
            1 for row in rows + probe_rows if row["rejected_stage"] == 2
        ),
        "accepted": sum(
            1 for row in rows + probe_rows if row["rejected_stage"] is None
        ),
    }

    parity = {
        "schema": PARITY_SCHEMA,
        "schema_version": 2,
        "status": "staging_not_release",
        "contract": weights["contract"],
        "classes": classes,
        "frontend": frontend_payload,
        "fixture_lengths": list(FIXTURE_LENGTHS),
        "parity_novelty_seed": int(args.parity_seed),
        "parity_seed_is_evaluation_evidence": False,
        "known_rows_population": "development selection split (scored, never fit)",
        "counts": counts,
        "policy_version": staged.STAGED_POLICY_VERSION,
        "survivor_score": staged.COMPOSITE_SURVIVOR_SCORE,
        "staged_threshold": float(composite.threshold),
        "stage_two_threshold": float(policy.threshold),
        "tolerance": {
            "stage_one_features_abs": 1e-8,
            "stage_one_score_abs": 1e-8,
            "embedding_abs": 1e-5,
            "rank_abs": 1e-9,
            "raw_lof_relative": 1e-9,
            "decisions": "exact, including which stage rejected each row",
        },
        "staged_score_note": staged.STAGED_SCORE_NOTE,
        "rows": rows + probe_rows,
        "probe_cases": probe_cases,
        "provenance": provenance,
    }

    write_json(output / WEIGHTS_NAME, weights)
    write_json(output / CLASSIFIER_WEIGHTS_NAME, classifier_weights)
    write_json(output / PARITY_NAME, parity)

    output_manifest = {
        "schema": "time-domain-v3-openset-staging-manifest-v1",
        "outputs": {
            name: {
                "bytes": (output / name).stat().st_size,
                "sha256": _sha256(output / name),
            }
            for name in (WEIGHTS_NAME, CLASSIFIER_WEIGHTS_NAME, PARITY_NAME)
        },
        "inputs": provenance,
        "counts": counts,
    }
    write_json(output / MANIFEST_NAME, output_manifest)
    print(
        f"[v3 openset export] wrote {output}: rows={counts['rows']} "
        f"gated={counts['gated_stage_one']} stage2={counts['rejected_stage_two']} "
        f"accepted={counts['accepted']}",
        flush=True,
    )
    return output_manifest


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
