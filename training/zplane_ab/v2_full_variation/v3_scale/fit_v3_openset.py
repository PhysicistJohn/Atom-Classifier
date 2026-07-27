"""Fit and score the additive open-set rejector against a **v3 fusion** artifact.

This is the replacement for ``run_v3_openset_replication.py`` that HANDOFF
sections 10.5 and 15.3.1 require.  The old runner hard-codes the v2 runtime
bundle ``invariant_fusion_runtime_bundle_v2_seed20260727`` and therefore fits the
rejector on v2 embeddings produced by the FFT-bearing v2 frontend.  This module
takes ``--fusion-dir`` instead: every embedding it fits, calibrates and scores is
rebuilt through the same loaders, the same frontend and the same fusion maths
that produced the fusion artifact, by calling ``assemble_v3_fusion``'s own
functions rather than re-deriving them.

Fitting contract, unchanged from v2 and not relaxed:

- The known-only **density** (per-class time-domain geometry, and the branch LOF
  reference population) is fit on **training rows only**.
- Empirical **ranks** and the q95 operating **threshold** are fit on
  **enrollment rows only**.
- The immutable development-selection population is scored, never fit.
- Development novelty is generated fresh at untouched seeds and is scored only.
  It never reaches a fit, a rank, a threshold, or a policy choice; the policy is
  already frozen in :mod:`v3_time_domain_openset` and is imported, not searched.
- The consumed historical test half and every sealed release suite are neither
  loaded nor referenced.  Release and sealed paths are refused before any data
  is touched.  Release seed 20260731 is not used here and cannot be reached from
  this module.

**On ``FROZEN_V2_WEIGHT``.**  The name is historical.  Reading
``v3_time_domain_openset.FrozenV3OpenSet.fit`` and ``.score``, the value is used
only as the blend weight on a caller-supplied rank vector that is validated by
``_rank_score`` to lie in ``[0, 1)``; nothing about v2, its frontend, its
embeddings, or its LOF neighbour counts is bound anywhere in that module.  It is
therefore honestly a *branch LOF rank weight*, and this module imports it under
the alias :data:`BRANCH_LOF_RANK_WEIGHT` and feeds it ranks computed from **v3**
branch embeddings.  The constant is not renamed at its source because
``FrozenV3OpenSet.to_payload`` / ``from_payload`` serialize the key ``v2_weight``
and a rename would silently break every already-written policy file.

What *is* genuinely inherited from v2, and is stated rather than hidden: the LOF
ensemble shape (real branch, k=2, weight 0.40; complex branch, k=64, weight 0.60)
was selected during the v2 campaign.  It is re-fit here on v3 embeddings but its
hyperparameters are **not** re-searched, because re-searching them on dev would
be a fresh selection this module has no budget for.  They are module constants
with no command-line flag so they cannot be tuned by an operator.

The rejector is **additive only**: it never changes the closed-set label.  That
is asserted, not assumed, by :func:`assert_closed_label_unchanged`.

This produces development evidence.  It is not release evidence.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
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
import invariant_patch_data as invariant_data  # noqa: E402
import run_invariant_cnn_dev as runner  # noqa: E402
import run_time_domain_dev as dev  # noqa: E402
import time_domain_invariant_patch_preprocess as td_preprocess  # noqa: E402
from known_only_patch_openset import KnownOnlyLOFOpenSet  # noqa: E402
from openset_eval import NOVELTY  # noqa: E402
from train import auroc, embed_all, nearest  # noqa: E402
from v3_time_domain_openset import (  # noqa: E402
    FROZEN_GEOMETRY_FEATURE,
    FROZEN_GEOMETRY_WEIGHT,
    FROZEN_POLICY_KIND,
    FROZEN_POLICY_SCHEMA,
    FROZEN_THRESHOLD_QUANTILE,
    FROZEN_V2_WEIGHT,
    FrozenV3OpenSet,
)


# ---------------------------------------------------------------------------
# frozen policy, imported and aliased -- never redefined here
# ---------------------------------------------------------------------------

# See the module docstring: FROZEN_V2_WEIGHT is a branch-LOF rank weight, not a
# binding to the v2 stack.
BRANCH_LOF_RANK_WEIGHT = FROZEN_V2_WEIGHT
GEOMETRY_RANK_WEIGHT = FROZEN_GEOMETRY_WEIGHT
THRESHOLD_QUANTILE = FROZEN_THRESHOLD_QUANTILE

# Inherited v2 LOF ensemble shape.  Deliberately not exposed as CLI flags.
BRANCH_LOF = (
    ("real", 0.40, 2),
    ("complex", 0.60, 64),
)

# Gate floors copied verbatim from run_v3_openset_replication._gate_summary.
# Reported, never enforced by weakening: a gate is not lowered after a result.
GATE_FLOORS = {
    "overall_auroc": 0.72,
    "noise_auroc": 0.80,
    "chirp_auroc": 0.80,
    "noise_threshold_recall": 0.10,
    "chirp_threshold_recall": 0.10,
}
KNOWN_FUR_CEILING = 0.10

# HANDOFF 10.2 names 20260939 and 20260940 as untouched development novelty
# seeds.  20260938 designed the policy and must not be reused as evidence.
DESIGN_NOVELTY_SEED = 20260938
DEFAULT_NOVELTY_SEEDS = (20260939, 20260940)
NOVELTY_FAMILIES = ("noise", "chirp")
DEFAULT_NOVELTY_N = 300
# The development corpus stores N=16384.  A longer observation may not be
# synthesized, so the prefix sweep tops out there.
DEFAULT_PREFIX_LENGTHS = (4096, 8192, 16384)

CENTER_SPLIT = assemble.CENTER_SPLIT      # "tr", training
PROTOTYPE_SPLIT = assemble.PROTOTYPE_SPLIT  # "en", enrollment
METRIC_SPLIT = assemble.METRIC_SPLIT      # "va", immutable selection

DEFAULT_REPRODUCTION_TOLERANCE = 1e-4

RELEASE_SEED_NEVER_SPENT_HERE = 20260731


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def reject_sealed_path(path: Path, role: str) -> Path:
    """Refuse release or sealed locations, for reading or for writing."""
    return assemble.reject_sealed_path(Path(path), role)


def _sha256(path: Path) -> str:
    return dev._sha256(Path(path))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    temporary = Path(path).with_name(f".{Path(path).name}.tmp-{os.getpid()}")
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


def validate_novelty_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    """Refuse an empty, duplicated, or policy-designing novelty seed set."""
    values = tuple(int(seed) for seed in seeds)
    if not values or len(set(values)) != len(values):
        raise ValueError("novelty seeds must be a non-empty set of distinct ints")
    if DESIGN_NOVELTY_SEED in values:
        raise ValueError(
            f"novelty seed {DESIGN_NOVELTY_SEED} designed the frozen policy and "
            "may not be reused as replication evidence"
        )
    if RELEASE_SEED_NEVER_SPENT_HERE in values:
        raise ValueError(
            f"{RELEASE_SEED_NEVER_SPENT_HERE} is an unspent release seed and is "
            "not a development novelty seed"
        )
    return values


def validate_prefix_lengths(lengths: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(length) for length in lengths)
    if (
        not values
        or any(length <= 0 for length in values)
        or tuple(sorted(set(values))) != values
    ):
        raise ValueError("prefix lengths must be unique, positive and increasing")
    return values


# ---------------------------------------------------------------------------
# the v3 fusion artifact
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FusionArtifact:
    """A validated, hash-checked v3 fusion directory."""

    directory: Path
    metrics: dict[str, Any]
    file_sha256: dict[str, str]
    directory_sha256: str
    weight_real: float
    seed: int
    device: str
    real_center: np.ndarray
    complex_center: np.ndarray
    prototypes: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray


def load_fusion_artifact(fusion_dir: Path) -> FusionArtifact:
    """Load, validate and hash a v3 fusion artifact directory.

    Refuses anything that is not a development-only, FFT-free v3 fusion whose
    recorded artifact SHA-256 values match the bytes on disk.
    """
    directory = reject_sealed_path(Path(fusion_dir), "fusion")
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    metrics_path = directory / "dev_metrics.json"
    with metrics_path.open(encoding="utf-8") as handle:
        metrics = json.load(handle)

    if str(metrics.get("encoder")) != "fusion":
        raise ValueError(
            f"{directory} holds encoder {metrics.get('encoder')!r}, not 'fusion'"
        )
    for key, expected in (
        ("development_only", True),
        ("release_evidence", False),
        ("sealed_release_data_used", 0),
        ("consumed_test_rows_used", 0),
    ):
        if metrics.get(key) != expected:
            raise RuntimeError(
                f"{directory} records {key}={metrics.get(key)!r}, expected "
                f"{expected!r}"
            )
    if metrics.get("data_audit", {}).get("consumed_test_rows_exposed") != 0:
        raise RuntimeError(f"{directory} provenance exposed consumed test rows")
    frontend = metrics.get("frontend", {})
    if frontend.get("version") != td_preprocess.PREPROCESS_VERSION:
        raise RuntimeError(
            f"{directory} frontend is {frontend.get('version')!r}, expected "
            f"{td_preprocess.PREPROCESS_VERSION!r}"
        )
    if frontend.get("uses_frequency_transform") is not False:
        raise RuntimeError(f"{directory} does not declare an FFT-free frontend")

    artifacts = metrics["artifacts"]
    file_sha256 = {"dev_metrics.json": _sha256(metrics_path)}
    arrays: dict[str, np.ndarray] = {}
    for name in (
        "state_dict",
        "prototypes",
        "real_center",
        "complex_center",
        "feature_mean",
        "feature_std",
    ):
        path = directory / str(artifacts[name])
        digest = _sha256(path)
        if digest != artifacts[f"{name}_sha256"]:
            raise RuntimeError(f"{path} does not match its recorded SHA-256")
        file_sha256[path.name] = digest
        if path.suffix == ".npy":
            arrays[name] = np.load(path).astype(np.float32)

    directory_sha256 = hashlib.sha256(
        "\n".join(
            f"{name}:{file_sha256[name]}" for name in sorted(file_sha256)
        ).encode("utf-8")
    ).hexdigest()

    return FusionArtifact(
        directory=directory,
        metrics=metrics,
        file_sha256=file_sha256,
        directory_sha256=directory_sha256,
        weight_real=assemble.validate_branch_weight(
            metrics["fusion"]["weight_real"]
        ),
        seed=int(metrics["seed"]),
        device=str(metrics["device"]),
        real_center=arrays["real_center"],
        complex_center=arrays["complex_center"],
        prototypes=arrays["prototypes"],
        feature_mean=arrays["feature_mean"],
        feature_std=arrays["feature_std"],
    )


def resolve_device(name: str, artifact_device: str) -> torch.device:
    """``artifact`` reuses the device the fusion artifact was scored on."""
    return runner.resolve_device(
        str(artifact_device) if name == "artifact" else name
    )


# ---------------------------------------------------------------------------
# rebuilding the populations through the v3 fusion
# ---------------------------------------------------------------------------


@dataclass
class Populations:
    """Train / enrollment / selection rebuilt through the v3 fusion."""

    data: dict[str, Any]
    nets: dict[str, torch.nn.Module]
    branch_embeddings: dict[str, dict[str, np.ndarray]]
    fused: dict[str, np.ndarray]
    real_center: np.ndarray
    complex_center: np.ndarray
    prototypes: np.ndarray
    reproduction: dict[str, float]
    branch_hashes: dict[str, dict[str, str]]


def rebuild_populations(
    artifact: FusionArtifact,
    device: torch.device,
    *,
    tolerance: float = DEFAULT_REPRODUCTION_TOLERANCE,
) -> Populations:
    """Rebuild every split through the fusion, then prove it is the same fusion.

    Only ``assemble_v3_fusion``'s own loaders, fitting functions and fusion maths
    are used, so the embeddings are the ones the fusion artifact was scored with.
    Reproduction of the stored centers and prototypes is verified rather than
    assumed.
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
    data, _contexts, _audit = dev._prepare_data(
        source,
        manifest,
        patch_length=config.patch_length,
        patch_count=config.patch_count,
        target_frac=target_frac,
        train_lengths=dev.LENGTHS if training_views_enabled else None,
    )

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
    )


# ---------------------------------------------------------------------------
# branch LOF density (training reference, enrollment ranks)
# ---------------------------------------------------------------------------


def fit_branch_lof(
    branch_embeddings: Mapping[str, Mapping[str, np.ndarray]],
) -> list[tuple[str, float, KnownOnlyLOFOpenSet]]:
    """Fit the branch LOF ensemble: TRAINING reference, ENROLLMENT ranks.

    Only ``[CENTER_SPLIT]`` and ``[PROTOTYPE_SPLIT]`` of each branch are read.
    ``KnownOnlyLOFOpenSet.fit`` builds its reference population and k-distances
    from the training rows and calibrates its empirical rank on enrollment rows;
    selection and novelty are structurally unreachable from here.
    """
    components: list[tuple[str, float, KnownOnlyLOFOpenSet]] = []
    for branch, weight, neighbors in BRANCH_LOF:
        if branch not in branch_embeddings:
            raise KeyError(f"missing {branch!r} branch embeddings")
        splits = branch_embeddings[branch]
        for required in (CENTER_SPLIT, PROTOTYPE_SPLIT):
            if required not in splits:
                raise KeyError(
                    f"{branch!r} branch is missing the {required!r} split"
                )
        components.append(
            (
                branch,
                float(weight),
                KnownOnlyLOFOpenSet.fit(
                    np.asarray(splits[CENTER_SPLIT], dtype=np.float64),
                    np.asarray(splits[PROTOTYPE_SPLIT], dtype=np.float64),
                    neighbors=int(neighbors),
                ),
            )
        )
    return components


def score_branch_lof(
    components: Sequence[tuple[str, float, KnownOnlyLOFOpenSet]],
    embedding_by_branch: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Weighted rank ensemble, identical in form to the v2 assembler's."""
    if not components:
        raise ValueError("LOF ensemble must contain at least one component")
    total = float(sum(weight for _, weight, _ in components))
    if (
        not np.isfinite(total)
        or abs(total - 1.0) > 1e-12
        or any(weight <= 0.0 for _, weight, _ in components)
    ):
        raise ValueError("LOF ensemble weights must be positive and sum to one")
    output: np.ndarray | None = None
    for branch, weight, scorer in components:
        if branch not in embedding_by_branch:
            raise KeyError(f"missing {branch!r} embedding for LOF ensemble")
        component = weight * scorer.score(
            np.asarray(embedding_by_branch[branch], dtype=np.float64)
        )
        output = component if output is None else output + component
    assert output is not None
    if not np.isfinite(output).all():
        raise RuntimeError("LOF ensemble produced a non-finite score")
    return output


def save_branch_lof(
    path: Path,
    components: Sequence[tuple[str, float, KnownOnlyLOFOpenSet]],
) -> None:
    """Serialize the fitted branch LOF ensemble without pickle."""
    payload: dict[str, np.ndarray] = {
        "kind": np.asarray("v3_branch_lof_rank_ensemble"),
        "component_count": np.asarray(len(components), dtype=np.int64),
    }
    for index, (branch, weight, scorer) in enumerate(components):
        prefix = f"component_{index}_"
        payload[f"{prefix}branch_source"] = np.asarray(branch)
        payload[f"{prefix}weight"] = np.asarray(weight, dtype=np.float64)
        payload[f"{prefix}neighbors"] = np.asarray(
            scorer.neighbors, dtype=np.int64
        )
        for name in (
            "embedding_reference",
            "embedding_mean",
            "embedding_scale",
            "reference_k_distance",
            "reference_local_density",
            "calibration",
            "calibration_scores",
        ):
            payload[f"{prefix}{name}"] = np.asarray(
                getattr(scorer, name), dtype=np.float64
            )
    np.savez_compressed(path, **payload)


# ---------------------------------------------------------------------------
# the frozen policy, fit on train + enrollment only
# ---------------------------------------------------------------------------


def fit_policy(
    training_packed: np.ndarray,
    training_labels: np.ndarray,
    enrollment_packed: np.ndarray,
    enrollment_predicted_classes: np.ndarray,
    enrollment_branch_lof_ranks: np.ndarray,
    *,
    n_classes: int,
) -> FrozenV3OpenSet:
    """Fit the frozen rejector: density on TRAINING, ranks/threshold on ENROLLMENT.

    The signature is the whole guarantee.  Only training and enrollment arrays
    are parameters, so no selection row, no novelty row and no release row can
    influence the class geometry, either empirical rank, or the q95 threshold.
    The blend weights and quantile are imported from
    :mod:`v3_time_domain_openset`; nothing is searched here.
    """
    return FrozenV3OpenSet.fit(
        training_packed,
        training_labels,
        enrollment_packed,
        enrollment_predicted_classes,
        enrollment_branch_lof_ranks,
        n_classes=int(n_classes),
    )


def assert_closed_label_unchanged(
    fused_embeddings: np.ndarray,
    prototypes: np.ndarray,
    prediction_before: np.ndarray,
    rejector_scores: np.ndarray,
) -> np.ndarray:
    """Prove the rejector is additive: the closed label is bit-identical.

    Recomputes the nearest-prototype label *after* the rejector has run and
    refuses to continue unless it equals the label computed before.  Also refuses
    a score vector that is misaligned or non-finite, since a rejector that cannot
    be aligned to rows cannot be additive either.
    """
    scores = np.asarray(rejector_scores, dtype=np.float64)
    before = np.asarray(prediction_before, dtype=np.int64)
    if scores.ndim != 1 or len(scores) != len(before):
        raise AssertionError("rejector scores are not aligned with rows")
    if not np.isfinite(scores).all():
        raise AssertionError("rejector produced a non-finite score")
    after, _distance = nearest(fused_embeddings, prototypes)
    after = np.asarray(after, dtype=np.int64)
    if not np.array_equal(before, after):
        raise AssertionError(
            "the open-set rejector altered the closed-set label; it is "
            "required to be additive only"
        )
    return after


# ---------------------------------------------------------------------------
# development novelty
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NoveltyFixture:
    packed: np.ndarray
    features: np.ndarray


def _complex_digest(digest: "hashlib._Hash", value: np.ndarray) -> None:
    canonical = np.ascontiguousarray(np.asarray(value, dtype=np.complex128))
    digest.update(canonical.view(np.uint8))


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
    families: Sequence[str] = NOVELTY_FAMILIES,
    progress: bool = False,
) -> tuple[dict[int, dict[str, dict[int, NoveltyFixture]]], dict[str, Any]]:
    """Generate dev novelty once at the longest prefix, then observe prefixes.

    Every row is drawn once at ``max(lengths)`` and the shorter observations are
    exact raw prefixes of that same capture, so a length effect cannot be
    confounded with a fresh draw.  Each observation is preprocessed with the
    **v3 time-domain frontend**, not the v2 FFT frontend, and standardized with
    the fusion artifact's own training feature statistics.
    """
    seed_values = validate_novelty_seeds(seeds)
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
            longest_digest = hashlib.sha256()
            prefix_digest = {
                length: hashlib.sha256() for length in prefix_lengths
            }
            for row in range(int(n_each)):
                raw = generator(longest, rng)
                if len(raw) != longest:
                    raise RuntimeError("novelty generator returned wrong length")
                _complex_digest(longest_digest, raw)
                for length in prefix_lengths:
                    observation = raw[:length]
                    _complex_digest(prefix_digest[length], observation)
                    packed, raw_features, _context = td_preprocess.preprocess(
                        observation,
                        patch_length=int(patch_length),
                        patch_count=int(patch_count),
                        target_frac=float(target_frac),
                    )
                    packed_by_length[length].append(packed)
                    feature_by_length[length].append(raw_features)
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
                fixtures[seed][family][length] = NoveltyFixture(
                    packed,
                    ((raw_features - mean) / std).astype(np.float32),
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
    policy: FrozenV3OpenSet


def score_rows(
    rejector: Rejector,
    packed: np.ndarray,
    features: np.ndarray,
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
    score = rejector.policy.score(lof_rank, packed, prediction)
    assert_closed_label_unchanged(fused, rejector.prototypes, prediction, score)
    return prediction, score


def family_metrics(
    novelty_scores: np.ndarray,
    known_scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    return {
        "auroc": float(auroc(novelty_scores, known_scores)),
        "threshold_recall": float(np.mean(novelty_scores > threshold)),
        "score_median": float(np.median(novelty_scores)),
    }


def gate_summary(rows: Sequence[Mapping[str, Any]], known_fur: float) -> dict[str, Any]:
    """Report the frozen gates.  Floors are constants and are never lowered."""
    if not rows:
        raise ValueError("gate summary requires at least one scored row")
    values: dict[str, list[float]] = {name: [] for name in GATE_FLOORS}
    for row in rows:
        values["overall_auroc"].append(float(row["overall"]["auroc"]))
        for family in NOVELTY_FAMILIES:
            values[f"{family}_auroc"].append(float(row[family]["auroc"]))
            values[f"{family}_threshold_recall"].append(
                float(row[family]["threshold_recall"])
            )
    gates = {
        name: {
            "worst": float(min(values[name])),
            "floor": float(floor),
            "passes": bool(min(values[name]) >= floor),
        }
        for name, floor in GATE_FLOORS.items()
    }
    gates["known_false_unknown_rate"] = {
        "worst": float(known_fur),
        "ceiling": float(KNOWN_FUR_CEILING),
        "passes": bool(known_fur <= KNOWN_FUR_CEILING),
    }
    return {
        "gates": gates,
        "all_pass": bool(all(item["passes"] for item in gates.values())),
    }


def _source_hashes() -> dict[str, str]:
    paths = {
        "fit_v3_openset.py": Path(__file__).resolve(),
        "assemble_v3_fusion.py": HERE / "assemble_v3_fusion.py",
        "run_time_domain_dev.py": HERE / "run_time_domain_dev.py",
        "v3_time_domain_openset.py": V2 / "v3_time_domain_openset.py",
        "known_only_patch_openset.py": V2 / "known_only_patch_openset.py",
        "openset_eval.py": V2 / "openset_eval.py",
        "invariant_patch_data.py": V2 / "invariant_patch_data.py",
        "invariant_fusion.py": V2 / "invariant_fusion.py",
        "time_domain_geometry.py": TRAINING / "time_domain_geometry.py",
        "time_domain_invariant_patch_preprocess.py": (
            TRAINING / "time_domain_invariant_patch_preprocess.py"
        ),
        "train.py": TRAINING / "train.py",
    }
    return {name: _sha256(path) for name, path in paths.items()}


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output = reject_sealed_path(Path(args.output_dir), "output")
    novelty_seeds = validate_novelty_seeds(args.novelty_seeds)
    prefix_lengths = validate_prefix_lengths(args.prefix_lengths)
    if int(args.novelty_n) <= 0:
        raise ValueError("--novelty-n must be positive")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} is not empty")

    artifact = load_fusion_artifact(Path(args.fusion_dir))
    device = resolve_device(args.device, artifact.device)
    print(
        f"[v3 openset] fusion={artifact.directory.name} "
        f"sha={artifact.directory_sha256[:16]} device={device}",
        flush=True,
    )

    populations = rebuild_populations(
        artifact, device, tolerance=float(args.reproduction_tolerance)
    )
    data = populations.data
    print(
        "[v3 openset] rebuilt populations; reproduction "
        f"{populations.reproduction}",
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
        data["xen"],
        enrollment_prediction,
        enrollment_lof,
        n_classes=int(data["n_classes"]),
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
        selection_lof, data["xva"], selection_prediction
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
        f"[v3 openset] threshold={policy.threshold:.6f} known FUR={known_fur:.6f}",
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
                    rejector, fixture.packed, fixture.features, device
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
    policy_path = output / "v3_open_policy.npz"
    np.savez_compressed(policy_path, **policy.to_payload())
    lof_path = output / "v3_branch_lof_components.npz"
    save_branch_lof(lof_path, lof_components)

    report = {
        "status": (
            "development_openset_pass"
            if summary["all_pass"]
            else "development_openset_fail"
        ),
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
            "scored population, known and novel"
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
                "branch LOF reference population and k-distances",
            ],
            "rank_and_threshold_population": "enrollment only",
            "rank_rows": int(len(data["xen"])),
            "selection_population_role": (
                "scored for metrics only; never fit, calibrated, or selected on"
            ),
            "novelty_population_role": (
                "scored for metrics only; never fit, ranked, thresholded, or "
                "used to choose a policy"
            ),
            "policy_searched_here": False,
            "sealed_release_rows_loaded": 0,
            "consumed_test_rows_loaded": 0,
            "reproduction_tolerance": float(args.reproduction_tolerance),
        },
        "policy": {
            "schema": int(FROZEN_POLICY_SCHEMA),
            "kind": FROZEN_POLICY_KIND,
            "geometry_feature": FROZEN_GEOMETRY_FEATURE,
            "branch_lof_rank_weight": float(BRANCH_LOF_RANK_WEIGHT),
            "branch_lof_rank_weight_source_constant": "FROZEN_V2_WEIGHT",
            "branch_lof_rank_weight_note": (
                "FROZEN_V2_WEIGHT is named for the v2 LOF but binds nothing "
                "about v2: v3_time_domain_openset uses it only as the blend "
                "weight on a caller-supplied rank vector in [0, 1). Here it "
                "weights ranks computed from v3 fusion branch embeddings."
            ),
            "geometry_rank_weight": float(GEOMETRY_RANK_WEIGHT),
            "threshold_quantile": float(THRESHOLD_QUANTILE),
            "threshold": float(policy.threshold),
            "branch_lof_components": [
                {"branch": branch, "weight": weight, "neighbors": neighbors}
                for branch, weight, neighbors in BRANCH_LOF
            ],
            "branch_lof_hyperparameters_source": (
                "inherited from the v2 ensemble and re-fit on v3 embeddings; "
                "not re-searched here"
            ),
            "cannot_change_closed_label": True,
        },
        "seeds": {
            "model_seed": artifact.seed,
            "fusion_artifact_seed": artifact.seed,
            "novelty_seeds": list(novelty_seeds),
            "design_novelty_seed_excluded": DESIGN_NOVELTY_SEED,
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
        "source_sha256": _source_hashes(),
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
        f"[v3 openset] wrote {report_path} status={report['status']}",
        flush=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--fusion-dir",
        required=True,
        help="a v3 fusion artifact directory written by assemble_v3_fusion.py",
    )
    parser.add_argument("--output-dir", required=True)
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
        default=list(DEFAULT_NOVELTY_SEEDS),
        help="untouched development novelty seeds (HANDOFF 10.2)",
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


if __name__ == "__main__":
    run(build_parser().parse_args())
