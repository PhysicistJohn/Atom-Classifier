"""Development-only replication of the frozen v3 open-set add-on.

The statistic and weights in :mod:`v3_time_domain_openset` were selected on
development novelty seed 20260938.  This runner does not search or tune them.
It evaluates untouched development novelty seeds 20260939 and 20260940,
generated once at N32768 and observed through exact prefixes, against:

* the primary model checkpoint (seed 20260727); and
* the independent model checkpoint (seed 20260728).

Only the leak-safe train/enrollment/selection arrays exposed by
``invariant_patch_data`` are loaded.  Paths containing a ``releases`` component
are rejected before any input or output access.
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
from typing import Any, Mapping

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
TRAINING = HERE.parent.parent
REPO = TRAINING.parent
for path in (TRAINING, HERE.parent, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import invariant_patch_preprocess as invariant_preprocess  # noqa: E402
from assemble_invariant_candidate import (  # noqa: E402
    _embed_all,
    _fuse_numpy,
    _save_lof_ensemble,
    _score_lof_ensemble,
)
from invariant_patch_cnn import (  # noqa: E402
    InvariantPatchCNN,
    InvariantPatchConfig,
)
import invariant_patch_data  # noqa: E402
from known_only_patch_openset import KnownOnlyLOFOpenSet  # noqa: E402
from openset_eval import NOVELTY  # noqa: E402
from train import auroc, nearest, prototypes_from  # noqa: E402
from v3_time_domain_openset import (  # noqa: E402
    FROZEN_GEOMETRY_FEATURE,
    FROZEN_GEOMETRY_WEIGHT,
    FROZEN_POLICY_KIND,
    FROZEN_THRESHOLD_QUANTILE,
    FROZEN_V2_WEIGHT,
    FrozenV3OpenSet,
)


PRIMARY_SEED = 20260727
REPLICATION_SEED = 20260728
DESIGN_NOVELTY_SEED = 20260938
REPLICATION_NOVELTY_SEEDS = (20260939, 20260940)
CAPTURE_LENGTHS = (4096, 8192, 16384, 32768)
NOVELTY_FAMILIES = ("noise", "chirp")
NOVELTY_N_EACH = 300
PATCH_LENGTH = 64
PATCH_COUNT = 16
TARGET_FRAC = 0.5
V2_REAL_WEIGHT = 0.4
V2_COMPLEX_WEIGHT = 0.6
V2_REAL_NEIGHBORS = 2
V2_COMPLEX_NEIGHBORS = 64
V2_THRESHOLD_QUANTILE = 0.95

PRIMARY_BUNDLE = (
    HERE
    / "artifacts"
    / "invariant_patch"
    / "invariant_fusion_runtime_bundle_v2_seed20260727"
    / "runtime_bundle.pt"
)
REPLICATION_SOURCE = (
    HERE
    / "artifacts"
    / "invariant_patch"
    / "invariant_patch_replication4000_seed20260728"
)
DEFAULT_OUTPUT = (
    HERE
    / "artifacts"
    / "invariant_patch"
    / "v3_openset_fixed_policy_replication_20260939_20260940"
)


@dataclass
class NoveltyFixture:
    packed: np.ndarray
    features: np.ndarray


@dataclass
class ModelContext:
    name: str
    model_seed: int
    real: InvariantPatchCNN
    complex: InvariantPatchCNN
    real_center: np.ndarray
    complex_center: np.ndarray
    prototypes: np.ndarray
    lof_components: list[tuple[str, float, KnownOnlyLOFOpenSet]]
    v2_threshold: float
    v3: FrozenV3OpenSet
    known_v2: np.ndarray
    known_v3: np.ndarray
    known_prediction: np.ndarray
    closed_accuracy: float


def _assert_development_path(path: Path, name: str) -> Path:
    resolved = path.expanduser().resolve()
    if "releases" in resolved.parts:
        raise ValueError(f"{name} must not be inside a releases path")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _load_state(path: Path) -> Mapping[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with old Torch
        return torch.load(path, map_location="cpu")


def _models_for_seed(
    model_seed: int,
    primary_bundle: Mapping[str, Any],
    device: torch.device,
) -> tuple[InvariantPatchCNN, InvariantPatchCNN, dict[str, str]]:
    real = InvariantPatchCNN(
        InvariantPatchConfig(**dict(primary_bundle["real_config"]))
    )
    complex_model = InvariantPatchCNN(
        InvariantPatchConfig(**dict(primary_bundle["complex_config"]))
    )
    if model_seed == PRIMARY_SEED:
        real.load_state_dict(primary_bundle["real_state_dict"], strict=True)
        complex_model.load_state_dict(
            primary_bundle["complex_state_dict"],
            strict=True,
        )
        hashes = {"runtime_bundle.pt": _sha256(PRIMARY_BUNDLE)}
    elif model_seed == REPLICATION_SEED:
        real_path = REPLICATION_SOURCE / "real_state_dict.pt"
        complex_path = REPLICATION_SOURCE / "complex_state_dict.pt"
        real.load_state_dict(_load_state(real_path), strict=True)
        complex_model.load_state_dict(_load_state(complex_path), strict=True)
        hashes = {
            "real_state_dict.pt": _sha256(real_path),
            "complex_state_dict.pt": _sha256(complex_path),
            "dev_metrics.json": _sha256(
                REPLICATION_SOURCE / "dev_metrics.json"
            ),
        }
    else:  # pragma: no cover - callers use compile-time seeds
        raise ValueError(f"unsupported model seed {model_seed}")
    return real.to(device).eval(), complex_model.to(device).eval(), hashes


def _branch_embeddings(
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
        print(f"  embedded {branch} train/enrollment/selection", flush=True)
    return result


def _build_model_context(
    model_seed: int,
    primary_bundle: Mapping[str, Any],
    data: Mapping[str, Any],
    device: torch.device,
) -> tuple[ModelContext, dict[str, str]]:
    name = f"model_seed_{model_seed}"
    print(f"[{name}] loading", flush=True)
    real, complex_model, source_hashes = _models_for_seed(
        model_seed,
        primary_bundle,
        device,
    )
    embeddings = _branch_embeddings(
        real,
        complex_model,
        data,
        device,
    )
    real_center = embeddings["real"]["tr"].mean(axis=0).astype(np.float32)
    complex_center = embeddings["complex"]["tr"].mean(axis=0).astype(
        np.float32
    )
    embeddings["fusion"] = {
        split: _fuse_numpy(
            embeddings["real"][split],
            embeddings["complex"][split],
            real_center,
            complex_center,
        )
        for split in ("tr", "en", "va")
    }
    prototypes = prototypes_from(
        embeddings["fusion"]["en"],
        np.asarray(data["yen"], dtype=np.int64),
        int(data["n_classes"]),
    )
    prediction = {
        split: nearest(embeddings["fusion"][split], prototypes)[0]
        for split in ("en", "va")
    }
    real_lof = KnownOnlyLOFOpenSet.fit(
        embeddings["real"]["tr"],
        embeddings["real"]["en"],
        neighbors=V2_REAL_NEIGHBORS,
    )
    complex_lof = KnownOnlyLOFOpenSet.fit(
        embeddings["complex"]["tr"],
        embeddings["complex"]["en"],
        neighbors=V2_COMPLEX_NEIGHBORS,
    )
    components = [
        ("real", V2_REAL_WEIGHT, real_lof),
        ("complex", V2_COMPLEX_WEIGHT, complex_lof),
    ]
    v2_scores = {
        split: _score_lof_ensemble(
            components,
            {
                branch: embeddings[branch][split]
                for branch in ("real", "complex", "fusion")
            },
        )
        for split in ("en", "va")
    }
    v2_threshold = float(
        np.quantile(v2_scores["en"], V2_THRESHOLD_QUANTILE)
    )
    if model_seed == PRIMARY_SEED:
        replay_errors = {
            "real_center": float(
                np.max(
                    np.abs(
                        real_center
                        - np.asarray(primary_bundle["real_center"])
                    )
                )
            ),
            "complex_center": float(
                np.max(
                    np.abs(
                        complex_center
                        - np.asarray(primary_bundle["complex_center"])
                    )
                )
            ),
            "prototypes": float(
                np.max(
                    np.abs(
                        prototypes
                        - np.asarray(primary_bundle["prototypes"])
                    )
                )
            ),
        }
        if max(replay_errors.values()) > 2e-6:
            raise RuntimeError(
                f"primary model replay mismatch: {replay_errors}"
            )
        expected_threshold = float(primary_bundle["unknown_threshold"])
        if abs(v2_threshold - expected_threshold) > 1e-12:
            raise RuntimeError(
                "primary v2 threshold replay mismatch: "
                f"{v2_threshold} != {expected_threshold}"
            )
    v3 = FrozenV3OpenSet.fit(
        data["xtr"],
        data["ytr"],
        data["xen"],
        prediction["en"],
        v2_scores["en"],
        n_classes=int(data["n_classes"]),
    )
    known_v3 = v3.score(
        v2_scores["va"],
        data["xva"],
        prediction["va"],
    )
    context = ModelContext(
        name=name,
        model_seed=model_seed,
        real=real,
        complex=complex_model,
        real_center=real_center,
        complex_center=complex_center,
        prototypes=prototypes,
        lof_components=components,
        v2_threshold=v2_threshold,
        v3=v3,
        known_v2=v2_scores["va"],
        known_v3=known_v3,
        known_prediction=prediction["va"],
        closed_accuracy=float(
            np.mean(prediction["va"] == np.asarray(data["yva"]))
        ),
    )
    print(
        f"[{name}] fitted known-only rejectors: "
        f"v2 FUR={np.mean(context.known_v2 > context.v2_threshold):.6f}, "
        f"v3 FUR={np.mean(context.known_v3 > context.v3.threshold):.6f}",
        flush=True,
    )
    return context, source_hashes


def _update_complex_digest(
    digest: "hashlib._Hash",
    value: np.ndarray,
) -> None:
    canonical = np.ascontiguousarray(
        np.asarray(value, dtype=np.complex128)
    )
    digest.update(canonical.view(np.uint8))


def _build_novelty_fixtures(
    seeds: tuple[int, ...],
    n_each: int,
    data: Mapping[str, Any],
) -> tuple[
    dict[int, dict[str, dict[int, NoveltyFixture]]],
    dict[str, Any],
]:
    fixtures: dict[int, dict[str, dict[int, NoveltyFixture]]] = {}
    provenance: dict[str, Any] = {}
    fmean = np.asarray(data["fmean"], dtype=np.float32)
    fstd = np.asarray(data["fstd"], dtype=np.float32)
    max_length = max(CAPTURE_LENGTHS)

    for seed in seeds:
        print(f"[novelty seed {seed}] generating N{max_length} once", flush=True)
        rng = np.random.default_rng(seed)
        fixtures[seed] = {}
        seed_provenance: dict[str, Any] = {}
        for family in NOVELTY_FAMILIES:
            generator = NOVELTY[family]
            packed_by_length: dict[int, list[np.ndarray]] = {
                length: [] for length in CAPTURE_LENGTHS
            }
            feature_by_length: dict[int, list[np.ndarray]] = {
                length: [] for length in CAPTURE_LENGTHS
            }
            max_digest = hashlib.sha256()
            prefix_digest = {
                length: hashlib.sha256() for length in CAPTURE_LENGTHS
            }
            for row in range(n_each):
                raw_max = generator(max_length, rng)
                if len(raw_max) != max_length:
                    raise RuntimeError("novelty generator returned wrong length")
                _update_complex_digest(max_digest, raw_max)
                for length in CAPTURE_LENGTHS:
                    raw_prefix = raw_max[:length]
                    _update_complex_digest(prefix_digest[length], raw_prefix)
                    packed, raw_features, _ = invariant_preprocess.preprocess(
                        raw_prefix,
                        patch_length=PATCH_LENGTH,
                        patch_count=PATCH_COUNT,
                        target_frac=TARGET_FRAC,
                    )
                    packed_by_length[length].append(packed)
                    feature_by_length[length].append(raw_features)
                if (row + 1) % 100 == 0:
                    print(
                        f"  {family}: {row + 1}/{n_each}",
                        flush=True,
                    )
            fixtures[seed][family] = {}
            for length in CAPTURE_LENGTHS:
                packed = np.stack(packed_by_length[length]).astype(np.float32)
                raw_features = np.stack(
                    feature_by_length[length]
                ).astype(np.float32)
                features = ((raw_features - fmean) / fstd).astype(np.float32)
                fixtures[seed][family][length] = NoveltyFixture(
                    packed,
                    features,
                )
            seed_provenance[family] = {
                "max_capture_sha256": max_digest.hexdigest(),
                "prefix_sha256": {
                    str(length): prefix_digest[length].hexdigest()
                    for length in CAPTURE_LENGTHS
                },
                "n": n_each,
            }
        provenance[str(seed)] = seed_provenance
    return fixtures, provenance


def _novelty_scores(
    context: ModelContext,
    fixture: NoveltyFixture,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    real_embedding = _embed_all(
        context.real,
        fixture.packed,
        fixture.features,
        device,
        batch=256,
    )
    complex_embedding = _embed_all(
        context.complex,
        fixture.packed,
        fixture.features,
        device,
        batch=256,
    )
    fusion_embedding = _fuse_numpy(
        real_embedding,
        complex_embedding,
        context.real_center,
        context.complex_center,
    )
    prediction = nearest(fusion_embedding, context.prototypes)[0]
    v2_score = _score_lof_ensemble(
        context.lof_components,
        {
            "real": real_embedding,
            "complex": complex_embedding,
            "fusion": fusion_embedding,
        },
    )
    v3_score = context.v3.score(
        v2_score,
        fixture.packed,
        prediction,
    )
    return v2_score, v3_score


def _family_metrics(
    novelty: np.ndarray,
    known: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    return {
        "auroc": float(auroc(novelty, known)),
        "threshold_recall": float(np.mean(novelty > threshold)),
        "score_median": float(np.median(novelty)),
    }


def _evaluate_context(
    context: ModelContext,
    fixtures: Mapping[int, Mapping[str, Mapping[int, NoveltyFixture]]],
    device: torch.device,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "model_seed": context.model_seed,
        "closed_selection_accuracy": context.closed_accuracy,
        "known_selection_n": int(len(context.known_v2)),
        "known_selection_is_fixed_across_novelty_lengths": True,
        "known_fur": {
            "v2": float(
                np.mean(context.known_v2 > context.v2_threshold)
            ),
            "v3": float(
                np.mean(context.known_v3 > context.v3.threshold)
            ),
        },
        "threshold": {
            "v2": context.v2_threshold,
            "v3": context.v3.threshold,
        },
        "seeds": {},
    }
    for seed, by_family in fixtures.items():
        seed_rows: dict[str, Any] = {}
        print(f"[{context.name}] scoring novelty seed {seed}", flush=True)
        for length in CAPTURE_LENGTHS:
            family_score: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for family in NOVELTY_FAMILIES:
                family_score[family] = _novelty_scores(
                    context,
                    by_family[family][length],
                    device,
                )
            row = {"v2": {}, "v3": {}}
            for version, position in (("v2", 0), ("v3", 1)):
                known = getattr(context, f"known_{version}")
                threshold = (
                    context.v2_threshold
                    if version == "v2"
                    else context.v3.threshold
                )
                for family in NOVELTY_FAMILIES:
                    row[version][family] = _family_metrics(
                        family_score[family][position],
                        known,
                        threshold,
                    )
                all_novel = np.concatenate(
                    [
                        family_score[family][position]
                        for family in NOVELTY_FAMILIES
                    ]
                )
                row[version]["overall"] = _family_metrics(
                    all_novel,
                    known,
                    threshold,
                )
                row[version]["known_fur"] = result["known_fur"][version]
            seed_rows[str(length)] = row
            print(
                f"  N{length}: v3 noise "
                f"{row['v3']['noise']['auroc']:.6f}/"
                f"{row['v3']['noise']['threshold_recall']:.6f}, chirp "
                f"{row['v3']['chirp']['auroc']:.6f}/"
                f"{row['v3']['chirp']['threshold_recall']:.6f}",
                flush=True,
            )
        result["seeds"][str(seed)] = seed_rows
    return result


def _gate_summary(models: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    floors = {
        "overall_auroc": 0.72,
        "noise_auroc": 0.80,
        "chirp_auroc": 0.80,
        "noise_threshold_recall": 0.10,
        "chirp_threshold_recall": 0.10,
    }
    values: dict[str, list[float]] = {name: [] for name in floors}
    fur: list[float] = []
    for model in models.values():
        fur.append(float(model["known_fur"]["v3"]))
        for seed in model["seeds"].values():
            for row in seed.values():
                values["overall_auroc"].append(
                    float(row["v3"]["overall"]["auroc"])
                )
                for family in NOVELTY_FAMILIES:
                    values[f"{family}_auroc"].append(
                        float(row["v3"][family]["auroc"])
                    )
                    values[f"{family}_threshold_recall"].append(
                        float(
                            row["v3"][family]["threshold_recall"]
                        )
                    )
    gates = {
        name: {
            "worst": float(min(current)),
            "floor": floor,
            "passes": bool(min(current) >= floor),
        }
        for name, (floor, current) in (
            (name, (floors[name], values[name])) for name in floors
        )
    }
    gates["known_fur"] = {
        "worst": float(max(fur)),
        "ceiling": 0.10,
        "passes": bool(max(fur) <= 0.10),
    }
    return {
        "gates": gates,
        "all_pass": bool(all(item["passes"] for item in gates.values())),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    primary_path = _assert_development_path(PRIMARY_BUNDLE, "primary bundle")
    replication_path = _assert_development_path(
        REPLICATION_SOURCE,
        "replication source",
    )
    output = _assert_development_path(args.output_dir, "output directory")
    if not primary_path.is_file() or not replication_path.is_dir():
        raise FileNotFoundError("model source is missing")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    if tuple(args.novelty_seeds) != REPLICATION_NOVELTY_SEEDS:
        raise ValueError(
            "replication novelty seeds are frozen at "
            f"{REPLICATION_NOVELTY_SEEDS}"
        )
    if args.novelty_n != NOVELTY_N_EACH:
        raise ValueError(f"novelty_n is frozen at {NOVELTY_N_EACH}")

    device = torch.device(args.device)
    primary_bundle = _load_state(primary_path)
    data = invariant_patch_data.load(
        patch_length=PATCH_LENGTH,
        patch_count=PATCH_COUNT,
        target_frac=TARGET_FRAC,
        model_seed=PRIMARY_SEED,
        build_if_missing=False,
    )
    if data["data_audit"].get("consumed_test_rows_exposed") != 0:
        raise RuntimeError("data loader exposed consumed test rows")
    forbidden = [name for name in data if "test" in name.lower()]
    if forbidden:
        raise RuntimeError(f"data loader exposed test-like keys: {forbidden}")

    contexts: dict[str, ModelContext] = {}
    model_hashes: dict[str, dict[str, str]] = {}
    for model_seed in (PRIMARY_SEED, REPLICATION_SEED):
        context, hashes = _build_model_context(
            model_seed,
            primary_bundle,
            data,
            device,
        )
        contexts[context.name] = context
        model_hashes[context.name] = hashes
        policy_path = output / f"{context.name}_v3_open_policy.npz"
        np.savez_compressed(policy_path, **context.v3.to_payload())
        lof_path = output / f"{context.name}_v2_lof_components.npz"
        _save_lof_ensemble(
            lof_path,
            context.lof_components,
            context.v2_threshold,
        )

    fixtures, fixture_provenance = _build_novelty_fixtures(
        tuple(args.novelty_seeds),
        args.novelty_n,
        data,
    )
    model_reports = {
        name: _evaluate_context(context, fixtures, device)
        for name, context in contexts.items()
    }
    gate_summary = _gate_summary(model_reports)

    artifact_hashes = {
        path.name: _sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file()
    }
    source_paths = {
        "run_v3_openset_replication.py": Path(__file__).resolve(),
        "v3_time_domain_openset.py": (
            HERE / "v3_time_domain_openset.py"
        ).resolve(),
        "known_only_patch_openset.py": (
            HERE / "known_only_patch_openset.py"
        ).resolve(),
        "openset_eval.py": (HERE / "openset_eval.py").resolve(),
        "invariant_patch_data.py": (
            HERE / "invariant_patch_data.py"
        ).resolve(),
        "training/invariant_patch_preprocess.py": (
            TRAINING / "invariant_patch_preprocess.py"
        ).resolve(),
    }
    report = {
        "status": (
            "development_replication_pass"
            if gate_summary["all_pass"]
            else "development_replication_fail"
        ),
        "development_only": True,
        "release_evidence": False,
        "consumed_test_rows_used": 0,
        "sealed_release_paths_read": 0,
        "policy": {
            "kind": FROZEN_POLICY_KIND,
            "design_novelty_seed_not_reused_for_replication": (
                DESIGN_NOVELTY_SEED
            ),
            "geometry_feature": FROZEN_GEOMETRY_FEATURE,
            "v2_weight": FROZEN_V2_WEIGHT,
            "geometry_weight": FROZEN_GEOMETRY_WEIGHT,
            "threshold_quantile": FROZEN_THRESHOLD_QUANTILE,
            "v2_components": [
                {
                    "branch": "real",
                    "neighbors": V2_REAL_NEIGHBORS,
                    "weight": V2_REAL_WEIGHT,
                },
                {
                    "branch": "complex",
                    "neighbors": V2_COMPLEX_NEIGHBORS,
                    "weight": V2_COMPLEX_WEIGHT,
                },
            ],
            "fit_population": "training only",
            "rank_and_threshold_population": "enrollment only",
            "novelty_used_for_fit_rank_or_threshold": False,
            "cannot_change_closed_label": True,
        },
        "replication_protocol": {
            "novelty_seeds": list(args.novelty_seeds),
            "novelty_n_each": args.novelty_n,
            "families": list(NOVELTY_FAMILIES),
            "capture_lengths": list(CAPTURE_LENGTHS),
            "generation": (
                "each novelty row generated exactly once at N32768; "
                "N4096/N8192/N16384 are exact raw prefixes"
            ),
            "known_auroc_reference": (
                "fixed immutable development-selection population at its "
                "native N16384; therefore known FUR is model-specific and "
                "constant across novelty prefix lengths"
            ),
            "models": [PRIMARY_SEED, REPLICATION_SEED],
        },
        "fixture_provenance": fixture_provenance,
        "models": model_reports,
        **gate_summary,
        "provenance": {
            "command": [sys.executable, *sys.argv],
            "cwd": str(Path.cwd()),
            "device": str(device),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "git_head": _git_head(),
            "cache_contract": data["cache_contract"],
            "data_audit": data["data_audit"],
            "model_source_hashes": model_hashes,
            "source_hashes": {
                name: _sha256(path)
                for name, path in source_paths.items()
            },
            "artifact_hashes_before_report": artifact_hashes,
            "wall_clock_s": time.perf_counter() - started,
        },
    }
    report_path = output / "development_replication_report.json"
    _atomic_json(report_path, report)
    report["provenance"]["report_sha256"] = _sha256(report_path)
    # The detached report hash refers to the first immutable report bytes.  A
    # tiny sidecar binds it without creating a self-referential JSON hash.
    _atomic_json(
        output / "REPORT_SHA256.json",
        {
            "file": report_path.name,
            "sha256": report["provenance"]["report_sha256"],
        },
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--novelty-seeds",
        type=int,
        nargs="+",
        default=list(REPLICATION_NOVELTY_SEEDS),
    )
    parser.add_argument("--novelty-n", type=int, default=NOVELTY_N_EACH)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "all_pass": result["all_pass"],
                "gates": result["gates"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
