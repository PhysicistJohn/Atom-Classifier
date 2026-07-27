"""Assemble and validate the selection-only invariant classifier candidate.

The candidate uses centered late fusion for its authoritative closed-set label
and a positive weighted ensemble of enrollment-ranked branch LOF scores for
rejection.  Centers and LOF references are fit on training rows only; enrollment
rows calibrate prototypes, ranks, and the rejection threshold.  The immutable
development-selection population is used only for metrics and bounded
hyperparameter choice.  The already-consumed test half is neither loaded nor
serialized.

This produces a development bundle for runtime/parity work, not release
evidence.  A newly sealed synthetic corpus and real SDR captures remain required
before release claims.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import itertools
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ZPAB = HERE.parent
TRAINING = ZPAB.parent
REPO = TRAINING.parent
for path in (str(TRAINING), str(ZPAB), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import invariant_patch_data as invariant_data  # noqa: E402
from invariant_fusion import CenteredInvariantFusion  # noqa: E402
from invariant_patch_cnn import (  # noqa: E402
    InvariantPatchCNN,
    InvariantPatchConfig,
)
from known_only_patch_openset import KnownOnlyLOFOpenSet  # noqa: E402
from run_invariant_cnn_dev import (  # noqa: E402
    build_development_novelty,
    canonical_length_sweep,
    physical_scale_sweep,
    resolve_device,
    seed_everything,
)
from train import auroc, nearest, prototypes_from  # noqa: E402


DEFAULT_SOURCE = (
    HERE
    / "artifacts"
    / "invariant_patch"
    / "invariant_patch_screen4000_seed20260727"
)
DEFAULT_OUTPUT = (
    HERE
    / "artifacts"
    / "invariant_patch"
    / "invariant_fusion_runtime_bundle_v2_seed20260727"
)
DEFAULT_SEED = 20260727
NOVELTY_SEED = 20260938
FEWSHOT_SEED = 424242
HIGH_SNR_DB = 18.0
ALPHA_REAL = 0.2
ALPHA_COMPLEX = 0.2
WEIGHT_REAL = 0.6
LOF_NEIGHBOR_GRID = (1, 2, 3, 5, 8, 10, 16, 32, 64)
LOF_WEIGHT_GRID = tuple(float(value) for value in np.linspace(0.025, 0.975, 39))
UNKNOWN_QUANTILE = 0.95
FEWSHOT_K = 5
FEWSHOT_TRIALS = 100

GATES = {
    "closed_fine": 0.72,
    "closed_family": 0.82,
    "closed_high_snr": 0.78,
    "fewshot_loo_k5_resolvable": 0.85,
    "open_auroc_overall": 0.72,
    "open_auroc_noise": 0.80,
    "open_auroc_chirp": 0.80,
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
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
    if isinstance(value, torch.device):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
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
    os.replace(temporary, path)


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


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with old PyTorch
        return torch.load(path, map_location="cpu")


def _embed_all(
    net: torch.nn.Module,
    waveforms: np.ndarray,
    features: np.ndarray,
    device: torch.device,
    *,
    batch: int = 256,
) -> np.ndarray:
    """Inference helper that never wraps read-only memmaps with Torch tensors."""
    net.eval()
    output: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(waveforms), batch):
            x = torch.from_numpy(
                np.array(waveforms[start : start + batch], copy=True)
            ).to(device)
            feat = torch.from_numpy(
                np.array(features[start : start + batch], copy=True)
            ).to(device)
            output.append(net(x, feat).detach().cpu().numpy())
    result = np.concatenate(output).astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise RuntimeError("model produced a non-finite embedding")
    return result


def _safe_unit(value: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    norm = np.sqrt(np.sum(value * value, axis=1, keepdims=True))
    output = value / np.maximum(norm, np.float32(eps))
    cancelled = norm[:, 0] <= eps
    if cancelled.any():
        output[cancelled] = 0.0
        output[cancelled, 0] = 1.0
    return output.astype(np.float32, copy=False)


def _fuse_numpy(
    real: np.ndarray,
    complex_embedding: np.ndarray,
    real_center: np.ndarray,
    complex_center: np.ndarray,
) -> np.ndarray:
    centered_real = _safe_unit(
        real - np.float32(ALPHA_REAL) * np.asarray(real_center, np.float32)
    )
    centered_complex = _safe_unit(
        complex_embedding
        - np.float32(ALPHA_COMPLEX) * np.asarray(complex_center, np.float32)
    )
    fused = np.concatenate(
        (
            np.sqrt(np.float32(WEIGHT_REAL)) * centered_real,
            np.sqrt(np.float32(1.0 - WEIGHT_REAL)) * centered_complex,
        ),
        axis=1,
    )
    return _safe_unit(fused)


def _balanced_accuracy(
    prediction: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
) -> float:
    return float(
        np.mean(
            [
                np.mean(prediction[labels == class_index] == class_index)
                for class_index in range(n_classes)
            ]
        )
    )


def _closed_report(
    embeddings: np.ndarray,
    prototypes: np.ndarray,
    data: dict[str, Any],
    snr_db: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    labels = np.asarray(data["yva"], dtype=np.int64)
    impaired = np.asarray(data["imp_va"], dtype=bool)
    classes = list(data["classes"])
    prediction, distances = nearest(embeddings, prototypes)
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for truth, guess in zip(labels, prediction):
        confusion[int(truth), int(guess)] += 1
    recalls = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)
    high = snr_db >= HIGH_SNR_DB
    high_classes = np.unique(labels[high])
    high_balanced = float(
        np.mean(
            [
                np.mean(prediction[high & (labels == c)] == c)
                for c in high_classes
            ]
        )
    )
    accuracy = float(np.mean(prediction == labels))
    balanced = float(np.mean(recalls))
    return (
        {
            "n_selection": int(len(labels)),
            "accuracy": accuracy,
            "balanced_accuracy": balanced,
            "clean_accuracy": float(
                np.mean(prediction[~impaired] == labels[~impaired])
            ),
            "impaired_accuracy": float(
                np.mean(prediction[impaired] == labels[impaired])
            ),
            "per_class": {
                classes[index]: float(recalls[index])
                for index in range(len(classes))
            },
            "confusion": confusion.tolist(),
            "nearest_distance_median": float(np.median(distances.min(axis=1))),
            "high_snr": {
                "definition": f"manifest snrDb >= {HIGH_SNR_DB:g}",
                "threshold_db": HIGH_SNR_DB,
                "n": int(high.sum()),
                "accuracy": float(
                    np.mean(prediction[high] == labels[high])
                ),
                "balanced_accuracy": high_balanced,
            },
            "family": {
                "accuracy": accuracy,
                "balanced_accuracy": balanced,
                "mapping": {
                    class_name: class_name for class_name in classes
                },
                "reason": (
                    "the corrected seven-label corpus already uses broad RF "
                    "families and contains no finer order labels to merge"
                ),
            },
        },
        prediction,
    )


def _fewshot_report(
    enrollment: np.ndarray,
    selection: np.ndarray,
    data: dict[str, Any],
    full_prototypes: np.ndarray,
    per_class_closed: dict[str, float],
    *,
    seed: int,
    trials: int,
    k: int,
) -> dict[str, Any]:
    labels_en = np.asarray(data["yen"], dtype=np.int64)
    labels_va = np.asarray(data["yva"], dtype=np.int64)
    classes = list(data["classes"])
    n_classes = len(classes)
    pools = [np.where(labels_en == c)[0] for c in range(n_classes)]
    if any(len(pool) < k for pool in pools):
        raise RuntimeError("an enrollment class has fewer than k examples")

    rng = np.random.default_rng(seed)
    simultaneous = []
    for _ in range(trials):
        prototypes = np.stack(
            [
                enrollment[rng.choice(pool, size=k, replace=False)].mean(axis=0)
                for pool in pools
            ]
        )
        prediction, _ = nearest(selection, prototypes)
        simultaneous.append(
            _balanced_accuracy(prediction, labels_va, n_classes)
        )

    per_class: dict[str, float] = {}
    per_class_trials: dict[str, list[float]] = {}
    for class_index, class_name in enumerate(classes):
        base = np.delete(full_prototypes, class_index, axis=0)
        query = selection[labels_va == class_index]
        recalls = []
        for _ in range(trials):
            chosen = rng.choice(
                pools[class_index], size=k, replace=False
            )
            shot = enrollment[chosen].mean(axis=0)
            prediction, _ = nearest(
                query, np.vstack((base, shot[None, :]))
            )
            recalls.append(float(np.mean(prediction == len(base))))
        per_class[class_name] = float(np.mean(recalls))
        per_class_trials[class_name] = recalls

    resolvable = [
        class_name
        for class_name in classes
        if per_class_closed[class_name] >= 0.8
    ]
    simultaneous_array = np.asarray(simultaneous)
    return {
        "claim_scope": (
            "prototype re-enrollment for labels seen by the encoder; this is "
            "not unseen-class representation-transfer evidence"
        ),
        "shot_source": "disjoint enrollment population",
        "query_source": "immutable development-selection population",
        "seed": seed,
        "k": k,
        "trials": trials,
        "simultaneous_7way": {
            "balanced_accuracy_mean": float(simultaneous_array.mean()),
            "balanced_accuracy_p05": float(
                np.quantile(simultaneous_array, 0.05)
            ),
            "balanced_accuracy_min": float(simultaneous_array.min()),
            "balanced_accuracy_std": float(simultaneous_array.std()),
        },
        "leave_one_prototype_out": {
            "per_class_recall_mean": per_class,
            "all_class_mean": float(np.mean(list(per_class.values()))),
            "resolvable_definition": "full-enrollment per-class recall >= 0.8",
            "resolvable_classes": resolvable,
            "resolvable_mean": float(
                np.mean([per_class[name] for name in resolvable])
            ),
        },
    }


def _calibrate_temperature(
    enrollment: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
) -> dict[str, Any]:
    """Fit a distance-softmax temperature by enrollment leave-one-out NLL."""
    value = np.asarray(enrollment, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64)
    sums = np.stack([value[target == c].sum(axis=0) for c in range(n_classes)])
    counts = np.bincount(target, minlength=n_classes).astype(np.float64)
    if np.any(counts <= 1):
        raise RuntimeError("temperature calibration needs two rows per class")
    prototypes = np.broadcast_to(
        sums[None, :, :] / counts[None, :, None],
        (len(value), n_classes, value.shape[1]),
    ).copy()
    row = np.arange(len(value))
    prototypes[row, target] = (
        sums[target] - value
    ) / (counts[target, None] - 1.0)
    distances = np.sum((value[:, None, :] - prototypes) ** 2, axis=-1)
    temperatures = np.geomspace(1e-3, 2.0, 512)
    losses = []
    for temperature in temperatures:
        logits = -distances / temperature
        logits -= logits.max(axis=1, keepdims=True)
        log_normalizer = np.log(np.exp(logits).sum(axis=1))
        losses.append(
            float(np.mean(-logits[row, target] + log_normalizer))
        )
    best = int(np.argmin(losses))
    prediction = distances.argmin(axis=1)
    return {
        "temperature": float(temperatures[best]),
        "leave_one_out_nll": float(losses[best]),
        "leave_one_out_accuracy": float(np.mean(prediction == target)),
        "fit_population": "enrollment only",
        "grid_min": float(temperatures[0]),
        "grid_max": float(temperatures[-1]),
        "grid_size": int(len(temperatures)),
    }


def _open_report(
    embeddings: dict[str, dict[str, np.ndarray]],
    novelty_embeddings: dict[str, dict[str, np.ndarray]],
) -> tuple[
    dict[str, Any],
    list[tuple[str, float, KnownOnlyLOFOpenSet]],
]:
    """Bounded development selection of a monotone LOF-rank ensemble.

    Every component density is fit on training embeddings only and converted to
    a rank using enrollment only.  The fixed selection/novelty fixture selects
    branch, neighborhood size, and a positive convex weight; it never fits a
    density or operating threshold.
    """
    component_scores: dict[
        tuple[str, int], dict[str, np.ndarray]
    ] = {}
    component_scorers: dict[tuple[str, int], KnownOnlyLOFOpenSet] = {}
    individual: dict[str, Any] = {}
    novelty_names = tuple(sorted(next(iter(novelty_embeddings.values()))))
    for branch in ("real", "complex", "fusion"):
        for neighbors in LOF_NEIGHBOR_GRID:
            key = (branch, neighbors)
            scorer = KnownOnlyLOFOpenSet.fit(
                embeddings[branch]["tr"],
                embeddings[branch]["en"],
                neighbors=neighbors,
            )
            component_scorers[key] = scorer
            scores = {
                "enrollment": scorer.score(embeddings[branch]["en"]),
                "known": scorer.score(embeddings[branch]["va"]),
                **{
                    novelty_name: scorer.score(
                        novelty_embeddings[branch][novelty_name]
                    )
                    for novelty_name in novelty_names
                },
            }
            component_scores[key] = scores
            threshold = float(
                np.quantile(scores["enrollment"], UNKNOWN_QUANTILE)
            )
            all_novel = np.concatenate(
                [scores[name] for name in novelty_names]
            )
            individual[f"{branch}_k{neighbors}"] = {
                "branch": branch,
                "neighbors": neighbors,
                "auroc_overall": float(
                    auroc(all_novel, scores["known"])
                ),
                "known_false_unknown_rate": float(
                    np.mean(scores["known"] > threshold)
                ),
                **{
                    f"auroc_{name}": float(
                        auroc(scores[name], scores["known"])
                    )
                    for name in novelty_names
                },
            }

    best: dict[str, Any] | None = None
    evaluated = 0
    # The real and complex branches make complementary errors.  A broader
    # development screen also included same-branch and fused-score pairs; none
    # beat a real/complex pair.  The frozen replay grid therefore crosses every
    # real-k component with every complex-k component.
    real_keys = [
        key for key in component_scores if key[0] == "real"
    ]
    complex_keys = [
        key for key in component_scores if key[0] == "complex"
    ]
    for first, second in itertools.product(real_keys, complex_keys):
        for weight_first in LOF_WEIGHT_GRID:
            evaluated += 1
            weight_second = 1.0 - weight_first
            combined = {
                population: (
                    weight_first * component_scores[first][population]
                    + weight_second * component_scores[second][population]
                )
                for population in (
                    "enrollment",
                    "known",
                    *novelty_names,
                )
            }
            family_aurocs = {
                name: float(auroc(combined[name], combined["known"]))
                for name in novelty_names
            }
            all_novel = np.concatenate(
                [combined[name] for name in novelty_names]
            )
            overall = float(auroc(all_novel, combined["known"]))
            objective = (min(family_aurocs.values()), overall)
            if best is None or objective > best["objective"]:
                threshold = float(
                    np.quantile(
                        combined["enrollment"], UNKNOWN_QUANTILE
                    )
                )
                best = {
                    "objective": objective,
                    "first": first,
                    "second": second,
                    "weight_first": float(weight_first),
                    "weight_second": float(weight_second),
                    "scores": combined,
                    "family_aurocs": family_aurocs,
                    "overall": overall,
                    "threshold": threshold,
                }
    if best is None:  # pragma: no cover - grids are compile-time non-empty
        raise RuntimeError("open-set policy grid is empty")

    selected_components = [
        (
            best["first"][0],
            best["weight_first"],
            component_scorers[best["first"]],
        ),
        (
            best["second"][0],
            best["weight_second"],
            component_scorers[best["second"]],
        ),
    ]
    selected_scores = best["scores"]
    selected = {
        "kind": "positive_weighted_enrollment_ranked_lof",
        "components": [
            {
                "branch": branch,
                "neighbors": scorer.neighbors,
                "weight": weight,
            }
            for branch, weight, scorer in selected_components
        ],
        "strictly_monotone_in_each_component": True,
        "fit_population": "training embeddings only",
        "calibration_population": "disjoint enrollment embeddings only",
        "selection_or_novelty_used_in_density_or_threshold_fit": False,
        "selection_or_novelty_used_for_hyperparameter_choice": True,
        "threshold_quantile": UNKNOWN_QUANTILE,
        "threshold": best["threshold"],
        "n_known_selection": int(len(selected_scores["known"])),
        "n_novel": int(
            sum(len(selected_scores[name]) for name in novelty_names)
        ),
        "auroc_overall": best["overall"],
        "known_false_unknown_rate": float(
            np.mean(selected_scores["known"] > best["threshold"])
        ),
        **{
            f"auroc_{name}": best["family_aurocs"][name]
            for name in novelty_names
        },
        **{
            f"flagged_unknown_{name}": float(
                np.mean(selected_scores[name] > best["threshold"])
            )
            for name in novelty_names
        },
    }
    return (
        {
            "development_fixture": True,
            "selected_policy": selected,
            "selection_rule": (
                "maximize minimum fixed-family AUROC, then overall AUROC; "
                "deterministic first candidate breaks an exact tie"
            ),
            "search": {
                "branches": ["real", "complex", "fusion"],
                "neighbors": list(LOF_NEIGHBOR_GRID),
                "weights": list(LOF_WEIGHT_GRID),
                "combination": (
                    "all real-k by complex-k ranked-score pairs; positive "
                    "convex weighted sum"
                ),
                "broader_screen_note": (
                    "a preceding development screen also covered fused and "
                    "same-branch pairs; they did not beat a real/complex pair"
                ),
                "component_count": int(len(component_scores)),
                "ensemble_candidates_evaluated": int(evaluated),
                "individual_component_diagnostics": individual,
            },
        },
        selected_components,
    )


def _manifest_snr(data: dict[str, Any]) -> np.ndarray:
    corpus = TRAINING / "artifacts" / "signallab-corpus" / "corpus.json"
    with corpus.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    selection_ids = np.asarray(data["va_idx"], dtype=np.int64)
    if (
        selection_ids.ndim != 1
        or np.any(selection_ids < 0)
        or np.any(selection_ids >= len(manifest["items"]))
    ):
        raise RuntimeError("selection corpus identifiers are invalid")
    try:
        snr = np.asarray(
            [
                float(manifest["items"][int(index)]["snrDb"])
                for index in selection_ids
            ],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("selection manifest has invalid snrDb metadata") from exc
    if not np.isfinite(snr).all():
        raise RuntimeError("selection manifest has non-finite snrDb metadata")
    return snr


def _scorer_arrays(
    scorer: KnownOnlyLOFOpenSet,
) -> dict[str, np.ndarray]:
    return {
        "embedding_reference": scorer.embedding_reference,
        "embedding_mean": scorer.embedding_mean,
        "embedding_scale": scorer.embedding_scale,
        "reference_k_distance": scorer.reference_k_distance,
        "reference_local_density": scorer.reference_local_density,
        "sorted_calibration_raw": scorer.calibration,
        "calibration_rank_scores": scorer.calibration_scores,
    }


def _save_lof_ensemble(
    path: Path,
    components: list[tuple[str, float, KnownOnlyLOFOpenSet]],
    threshold: float,
) -> None:
    payload: dict[str, np.ndarray] = {
        "schema": np.asarray(2, dtype=np.int64),
        "kind": np.asarray("known_only_lof_rank_ensemble"),
        "component_count": np.asarray(len(components), dtype=np.int64),
        "threshold_quantile": np.asarray(
            UNKNOWN_QUANTILE, dtype=np.float64
        ),
        "threshold": np.asarray(threshold, dtype=np.float64),
    }
    for index, (branch, weight, scorer) in enumerate(components):
        prefix = f"component_{index}_"
        payload[f"{prefix}branch_source"] = np.asarray(branch)
        payload[f"{prefix}weight"] = np.asarray(weight, dtype=np.float64)
        payload[f"{prefix}neighbors"] = np.asarray(
            scorer.neighbors, dtype=np.int64
        )
        for name, value in _scorer_arrays(scorer).items():
            payload[f"{prefix}{name}"] = value
    np.savez_compressed(path, **payload)


def _score_lof_ensemble(
    components: list[tuple[str, float, KnownOnlyLOFOpenSet]],
    embedding_by_branch: dict[str, np.ndarray],
) -> np.ndarray:
    if not components:
        raise ValueError("LOF ensemble must contain at least one component")
    total_weight = float(sum(weight for _, weight, _ in components))
    if (
        not np.isfinite(total_weight)
        or abs(total_weight - 1.0) > 1e-12
        or any(weight <= 0.0 for _, weight, _ in components)
    ):
        raise ValueError("LOF ensemble weights must be positive and sum to one")
    output: np.ndarray | None = None
    for branch, weight, scorer in components:
        if branch not in embedding_by_branch:
            raise KeyError(f"missing {branch!r} embedding for LOF ensemble")
        component = weight * scorer.score(embedding_by_branch[branch])
        output = component if output is None else output + component
    assert output is not None
    if not np.isfinite(output).all():
        raise RuntimeError("LOF ensemble produced a non-finite score")
    return output


def _tensor_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in module.state_dict().items()
    }


def _lof_tensor_bundle(
    branch: str,
    scorer: KnownOnlyLOFOpenSet,
) -> dict[str, Any]:
    return {
        "kind": "known_only_lof",
        "branch_source": branch,
        "embedding_reference": torch.from_numpy(
            np.asarray(scorer.embedding_reference)
        ),
        "embedding_mean": torch.from_numpy(np.asarray(scorer.embedding_mean)),
        "embedding_scale": torch.from_numpy(np.asarray(scorer.embedding_scale)),
        "reference_k_distance": torch.from_numpy(
            np.asarray(scorer.reference_k_distance)
        ),
        "reference_local_density": torch.from_numpy(
            np.asarray(scorer.reference_local_density)
        ),
        "sorted_calibration_raw": torch.from_numpy(
            np.asarray(scorer.calibration)
        ),
        "calibration_rank_scores": torch.from_numpy(
            np.asarray(scorer.calibration_scores)
        ),
        "neighbors": int(scorer.neighbors),
        "selection_or_novelty_used_in_fit": False,
    }


def _lof_ensemble_tensor_bundle(
    components: list[tuple[str, float, KnownOnlyLOFOpenSet]],
    threshold: float,
) -> dict[str, Any]:
    return {
        "kind": "known_only_lof_rank_ensemble",
        "combination": "positive convex weighted sum of enrollment-ranked LOF",
        "strictly_monotone_in_each_component": True,
        "components": [
            {
                **_lof_tensor_bundle(branch, scorer),
                "weight": float(weight),
            }
            for branch, weight, scorer in components
        ],
        "threshold": float(threshold),
        "threshold_quantile": float(UNKNOWN_QUANTILE),
        "threshold_fit_population": "enrollment only",
        "selection_or_novelty_used_in_density_or_threshold_fit": False,
        "selection_or_novelty_used_for_hyperparameter_choice": True,
        "cannot_change_fused_closed_label": True,
    }


def _array_schema(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    return {
        "shape": [int(size) for size in array.shape],
        "dtype": array.dtype.str,
    }


def _reload_lof_ensemble(
    path: Path,
) -> tuple[
    list[tuple[str, float, KnownOnlyLOFOpenSet]],
    float,
]:
    with np.load(path, allow_pickle=False) as bundle:
        if int(bundle["schema"]) != 2:
            raise ValueError("unsupported LOF ensemble schema")
        count = int(bundle["component_count"])
        components = []
        for index in range(count):
            prefix = f"component_{index}_"
            components.append(
                (
                    str(bundle[f"{prefix}branch_source"]),
                    float(bundle[f"{prefix}weight"]),
                    KnownOnlyLOFOpenSet(
                        embedding_reference=bundle[
                            f"{prefix}embedding_reference"
                        ],
                        embedding_mean=bundle[
                            f"{prefix}embedding_mean"
                        ],
                        embedding_scale=bundle[
                            f"{prefix}embedding_scale"
                        ],
                        reference_k_distance=bundle[
                            f"{prefix}reference_k_distance"
                        ],
                        reference_local_density=bundle[
                            f"{prefix}reference_local_density"
                        ],
                        calibration=bundle[
                            f"{prefix}sorted_calibration_raw"
                        ],
                        calibration_scores=bundle[
                            f"{prefix}calibration_rank_scores"
                        ],
                        neighbors=int(bundle[f"{prefix}neighbors"]),
                    ),
                )
            )
        return components, float(bundle["threshold"])


def validate_runtime_bundle(bundle: Mapping[str, Any]) -> Mapping[str, Any]:
    """Fail closed on an incomplete, non-finite, or inconsistent runtime bundle."""
    if not isinstance(bundle, Mapping):
        raise TypeError("runtime bundle must be a mapping")
    required = {
        "schema",
        "kind",
        "real_config",
        "complex_config",
        "real_state_dict",
        "complex_state_dict",
        "real_center",
        "complex_center",
        "alpha_real",
        "alpha_complex",
        "weight_real",
        "eps",
        "feature_mean",
        "feature_std",
        "classes",
        "prototypes",
        "unknown_threshold",
        "temperature",
        "open_set",
        "development_only",
        "provenance",
    }
    missing = sorted(required - set(bundle))
    if missing:
        raise ValueError(f"runtime bundle is missing keys: {missing}")
    if bundle["schema"] != 2:
        raise ValueError("runtime bundle schema must be 2")
    if bundle["kind"] != "invariant_centered_fusion_classifier":
        raise ValueError("runtime bundle kind is invalid")

    real_config = InvariantPatchConfig(**dict(bundle["real_config"])).validate()
    complex_config = InvariantPatchConfig(
        **dict(bundle["complex_config"])
    ).validate()
    if real_config.encoder != "real" or complex_config.encoder != "complex":
        raise ValueError("runtime bundle branch encoders are invalid")
    real_model = InvariantPatchCNN(real_config)
    complex_model = InvariantPatchCNN(complex_config)
    for branch_name in ("real", "complex"):
        state = bundle[f"{branch_name}_state_dict"]
        if not isinstance(state, Mapping) or not state:
            raise ValueError(f"{branch_name}_state_dict must be a mapping")
        for tensor_name, value in state.items():
            if (
                not isinstance(tensor_name, str)
                or not isinstance(value, torch.Tensor)
                or (
                    (value.is_floating_point() or value.is_complex())
                    and not bool(torch.isfinite(value).all())
                )
            ):
                raise ValueError(
                    f"{branch_name}_state_dict entry {tensor_name!r} is invalid"
                )
    real_model.load_state_dict(bundle["real_state_dict"], strict=True)
    complex_model.load_state_dict(bundle["complex_state_dict"], strict=True)

    def finite_tensor(name: str, shape: tuple[int, ...]) -> torch.Tensor:
        value = bundle[name]
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != shape
            or not value.is_floating_point()
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"{name} must be a finite floating tensor {shape}")
        return value

    embed_dim = real_config.embed_dim
    if complex_config.embed_dim != embed_dim:
        raise ValueError("runtime bundle branch embedding dimensions differ")
    finite_tensor("real_center", (embed_dim,))
    finite_tensor("complex_center", (embed_dim,))
    finite_tensor("feature_mean", (real_config.n_features,))
    feature_std = finite_tensor("feature_std", (real_config.n_features,))
    if not bool((feature_std > 0).all()):
        raise ValueError("feature_std must be positive")

    classes = bundle["classes"]
    if (
        not isinstance(classes, list)
        or not classes
        or any(not isinstance(name, str) or not name for name in classes)
        or len(set(classes)) != len(classes)
    ):
        raise ValueError("classes must be a non-empty unique string list")
    finite_tensor("prototypes", (len(classes), 2 * embed_dim))

    for name in ("alpha_real", "alpha_complex", "weight_real", "eps",
                 "unknown_threshold", "temperature"):
        try:
            value = float(bundle[name])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be finite") from exc
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0.0 <= float(bundle["weight_real"]) <= 1.0:
        raise ValueError("weight_real must lie in [0, 1]")
    if float(bundle["eps"]) <= 0.0 or float(bundle["temperature"]) <= 0.0:
        raise ValueError("eps and temperature must be positive")

    open_set = bundle["open_set"]
    if (
        not isinstance(open_set, Mapping)
        or open_set.get("kind") != "known_only_lof_rank_ensemble"
        or open_set.get("cannot_change_fused_closed_label") is not True
    ):
        raise ValueError("runtime bundle open_set policy is invalid")
    components = open_set.get("components")
    if not isinstance(components, list) or len(components) < 2:
        raise ValueError("open_set must contain at least two LOF components")
    weights = []
    for index, component in enumerate(components):
        if not isinstance(component, Mapping):
            raise ValueError(f"open_set component {index} is invalid")
        branch = component.get("branch_source")
        dimension = {
            "real": real_config.embed_dim,
            "complex": complex_config.embed_dim,
            "fusion": 2 * embed_dim,
        }.get(branch)
        if dimension is None:
            raise ValueError(f"open_set component {index} branch is invalid")
        reference = component.get("embedding_reference")
        if (
            not isinstance(reference, torch.Tensor)
            or reference.ndim != 2
            or reference.shape[1] != dimension
            or not reference.is_floating_point()
            or not bool(torch.isfinite(reference).all())
        ):
            raise ValueError(
                f"open_set component {index} reference is invalid"
            )
        rows = int(reference.shape[0])
        for name, shape in (
            ("embedding_mean", (dimension,)),
            ("embedding_scale", (dimension,)),
            ("reference_k_distance", (rows,)),
            ("reference_local_density", (rows,)),
        ):
            value = component.get(name)
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != shape
                or not value.is_floating_point()
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(
                    f"open_set component {index} {name} is invalid"
                )
        if not bool((component["embedding_scale"] > 0).all()):
            raise ValueError(
                f"open_set component {index} embedding_scale is not positive"
            )
        calibration = component.get("sorted_calibration_raw")
        ranks = component.get("calibration_rank_scores")
        if (
            not isinstance(calibration, torch.Tensor)
            or not isinstance(ranks, torch.Tensor)
            or calibration.ndim != 1
            or ranks.shape != calibration.shape
            or not bool(torch.isfinite(calibration).all())
            or not bool(torch.isfinite(ranks).all())
            or not bool((calibration[1:] >= calibration[:-1]).all())
        ):
            raise ValueError(
                f"open_set component {index} calibration is invalid"
            )
        neighbors = component.get("neighbors")
        if (
            isinstance(neighbors, bool)
            or not isinstance(neighbors, int)
            or not 0 < neighbors < rows
        ):
            raise ValueError(
                f"open_set component {index} neighbors is invalid"
            )
        weight = float(component.get("weight", float("nan")))
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError(
                f"open_set component {index} weight is invalid"
            )
        weights.append(weight)
    if abs(sum(weights) - 1.0) > 1e-12:
        raise ValueError("open_set component weights must sum to one")
    if float(open_set.get("threshold", float("nan"))) != float(
        bundle["unknown_threshold"]
    ):
        raise ValueError("unknown_threshold disagrees with open_set threshold")
    if bundle["development_only"] is not True:
        raise ValueError("candidate runtime bundle must be development-only")
    provenance = bundle["provenance"]
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("release_evidence") is not False
        or provenance.get("consumed_test_rows_used") != 0
    ):
        raise ValueError("runtime bundle provenance is not leak-safe")
    return bundle


def load_runtime_bundle(path: Path) -> Mapping[str, Any]:
    """Load a trusted weights-only runtime bundle and validate its schema."""
    try:
        bundle = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with old PyTorch
        bundle = torch.load(path, map_location="cpu")
    return validate_runtime_bundle(bundle)


def _source_hashes(source: Path) -> dict[str, str]:
    paths = {
        "source_dev_metrics": source / "dev_metrics.json",
        "source_real_state_dict": source / "real_state_dict.pt",
        "source_complex_state_dict": source / "complex_state_dict.pt",
        "training/preprocess.py": TRAINING / "preprocess.py",
        "training/train.py": TRAINING / "train.py",
        "training/model.py": TRAINING / "model.py",
        "training/canonical_probe.py": TRAINING / "canonical_probe.py",
        "training/invariant_patch_preprocess.py": (
            TRAINING / "invariant_patch_preprocess.py"
        ),
        "invariant_patch_data.py": HERE / "invariant_patch_data.py",
        "invariant_patch_cnn.py": HERE / "invariant_patch_cnn.py",
        "invariant_fusion.py": HERE / "invariant_fusion.py",
        "known_only_patch_openset.py": HERE / "known_only_patch_openset.py",
        "canonical_probe_net.py": HERE / "canonical_probe_net.py",
        "openset_eval.py": HERE / "openset_eval.py",
        "complex_multiscale_backbone.py": (
            HERE / "complex_multiscale_backbone.py"
        ),
        "unet_transfer.py": HERE / "unet_transfer.py",
        "run_invariant_cnn_dev.py": HERE / "run_invariant_cnn_dev.py",
        "assemble_invariant_candidate.py": Path(__file__).resolve(),
    }
    return {name: _sha256(path) for name, path in paths.items()}


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} is not empty")
    device = resolve_device(args.device)
    seed_everything(args.seed)

    with (source / "dev_metrics.json").open(encoding="utf-8") as handle:
        source_metrics = json.load(handle)
    with (source / "DEVELOPMENT_ONLY.json").open(encoding="utf-8") as handle:
        source_audit = json.load(handle)
    if (
        source_audit.get("consumed_test_rows_used") != 0
        or source_audit.get("data_audit", {}).get(
            "consumed_test_rows_exposed"
        )
        != 0
    ):
        raise RuntimeError("source checkpoint provenance exposed consumed test rows")

    architectures = {
        name: source_metrics["models"][name]["architecture"]
        for name in ("real", "complex")
    }
    real = InvariantPatchCNN(
        InvariantPatchConfig(**architectures["real"])
    )
    complex_branch = InvariantPatchCNN(
        InvariantPatchConfig(**architectures["complex"])
    )
    real.load_state_dict(
        _load_state(source / "real_state_dict.pt"), strict=True
    )
    complex_branch.load_state_dict(
        _load_state(source / "complex_state_dict.pt"), strict=True
    )
    real = real.to(device).eval()
    complex_branch = complex_branch.to(device).eval()

    config = real.cfg
    data = invariant_data.load(
        patch_length=config.patch_length,
        patch_count=config.patch_count,
        target_frac=float(
            source_metrics["cache_contract"]["config"]["target_frac"]
        ),
        model_seed=args.seed,
        build_if_missing=False,
    )
    if data["cache_contract"] != source_metrics["cache_contract"]:
        raise RuntimeError("current cache contract differs from checkpoint source")
    if data["data_audit"].get("consumed_test_rows_exposed") != 0:
        raise RuntimeError("data loader exposed consumed test rows")
    forbidden = [key for key in data if "test" in key.lower()]
    if forbidden:
        raise RuntimeError(f"data loader exposed test-like keys: {forbidden}")

    nets = {"real": real, "complex": complex_branch}
    embeddings: dict[str, dict[str, np.ndarray]] = {
        name: {
            split: _embed_all(
                net,
                data[f"x{split}"],
                data[f"f{split}"],
                device,
            )
            for split in ("tr", "en", "va")
        }
        for name, net in nets.items()
    }
    real_center = embeddings["real"]["tr"].mean(axis=0).astype(np.float32)
    complex_center = embeddings["complex"]["tr"].mean(axis=0).astype(np.float32)
    fusion = CenteredInvariantFusion(
        real,
        complex_branch,
        real_center,
        complex_center,
        alpha_real=ALPHA_REAL,
        alpha_complex=ALPHA_COMPLEX,
        weight_real=WEIGHT_REAL,
    ).to(device).eval()
    embeddings["fusion"] = {
        split: _fuse_numpy(
            embeddings["real"][split],
            embeddings["complex"][split],
            real_center,
            complex_center,
        )
        for split in ("tr", "en", "va")
    }
    direct_probe = _embed_all(
        fusion, data["xva"][:64], data["fva"][:64], device
    )
    fusion_assembly_error = float(
        np.max(np.abs(direct_probe - embeddings["fusion"]["va"][:64]))
    )
    if fusion_assembly_error > 2e-6:
        raise RuntimeError(
            f"NumPy fusion assembly mismatch: {fusion_assembly_error}"
        )

    prototypes = {
        name: prototypes_from(
            value["en"], data["yen"], data["n_classes"]
        )
        for name, value in embeddings.items()
    }
    temperature = _calibrate_temperature(
        embeddings["fusion"]["en"],
        data["yen"],
        data["n_classes"],
    )
    snr_db = _manifest_snr(data)
    closed, fused_prediction = _closed_report(
        embeddings["fusion"]["va"],
        prototypes["fusion"],
        data,
        snr_db,
    )
    fewshot = _fewshot_report(
        embeddings["fusion"]["en"],
        embeddings["fusion"]["va"],
        data,
        prototypes["fusion"],
        closed["per_class"],
        seed=args.fewshot_seed,
        trials=args.fewshot_trials,
        k=args.fewshot_k,
    )

    novelty = build_development_novelty(
        n_each=args.novelty_n,
        seed=args.novelty_seed,
        patch_length=config.patch_length,
        patch_count=config.patch_count,
        target_frac=float(data["cache_contract"]["config"]["target_frac"]),
    )
    standardized_novelty = {
        name: (
            waveforms,
            (
                (raw_features - np.asarray(data["fmean"]))
                / np.asarray(data["fstd"])
            ).astype(np.float32),
        )
        for name, (waveforms, raw_features) in novelty.items()
    }
    novelty_embeddings: dict[str, dict[str, np.ndarray]] = {
        name: {} for name in ("real", "complex", "fusion")
    }
    for novelty_name, (waveforms, features) in standardized_novelty.items():
        novelty_embeddings["real"][novelty_name] = _embed_all(
            real, waveforms, features, device
        )
        novelty_embeddings["complex"][novelty_name] = _embed_all(
            complex_branch, waveforms, features, device
        )
        novelty_embeddings["fusion"][novelty_name] = _fuse_numpy(
            novelty_embeddings["real"][novelty_name],
            novelty_embeddings["complex"][novelty_name],
            real_center,
            complex_center,
        )
    open_set, selected_components = _open_report(
        embeddings, novelty_embeddings
    )

    length = canonical_length_sweep(
        fusion,
        prototypes["fusion"],
        list(data["classes"]),
        data,
        device,
    )
    scale = physical_scale_sweep(
        fusion,
        prototypes["fusion"],
        list(data["classes"]),
        data,
        device,
    )

    selected_open = open_set["selected_policy"]
    gate_values = {
        "closed_fine": closed["accuracy"],
        "closed_family": closed["family"]["accuracy"],
        "closed_high_snr": closed["high_snr"]["accuracy"],
        "fewshot_loo_k5_resolvable": (
            fewshot["leave_one_prototype_out"]["resolvable_mean"]
        ),
        "open_auroc_overall": selected_open["auroc_overall"],
        "open_auroc_noise": selected_open["auroc_noise"],
        "open_auroc_chirp": selected_open["auroc_chirp"],
    }
    gates = {
        name: {
            "value": float(gate_values[name]),
            "floor": float(floor),
            "passes": bool(gate_values[name] >= floor),
        }
        for name, floor in GATES.items()
    }
    gates["canonical_length"] = {
        "passes": bool(length["passes_gate"]),
        "worst_balanced_accuracy": float(length["worst_bal"]),
        "max_allowed_mean_pairwise_cosine": float(
            length["gate"]["max_mean_pairwise_cos"]
        ),
        "worst_mean_pairwise_cosine": float(length["worst_cos"]),
    }
    gates["canonical_scale"] = {
        "passes": bool(scale["passes_gate"]),
        "worst_balanced_accuracy": float(scale["worst_bal"]),
        "max_allowed_mean_pairwise_cosine": float(
            scale["gate"]["max_mean_pairwise_cos"]
        ),
        "worst_mean_pairwise_cosine": float(scale["worst_cos"]),
    }
    selection_gates_pass = all(
        bool(item["passes"]) for item in gates.values()
    )

    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "candidate_state_dict.pt"
    prototypes_path = output / "fused_prototypes.npy"
    feature_path = output / "feature_standardization.npz"
    lof_path = output / "open_rejector_lof_rank_ensemble.npz"
    runtime_path = output / "runtime_bundle.pt"
    torch.save(_tensor_state(fusion), state_path)
    np.save(prototypes_path, prototypes["fusion"])
    np.savez_compressed(
        feature_path,
        mean=np.asarray(data["fmean"], dtype=np.float32),
        std=np.asarray(data["fstd"], dtype=np.float32),
    )
    unknown_threshold = float(selected_open["threshold"])
    _save_lof_ensemble(
        lof_path,
        selected_components,
        unknown_threshold,
    )
    source_hashes = _source_hashes(source)
    runtime_bundle = {
        "schema": 2,
        "kind": "invariant_centered_fusion_classifier",
        "real_config": architectures["real"],
        "complex_config": architectures["complex"],
        "real_state_dict": _tensor_state(real),
        "complex_state_dict": _tensor_state(complex_branch),
        "real_center": torch.from_numpy(real_center),
        "complex_center": torch.from_numpy(complex_center),
        "alpha_real": float(ALPHA_REAL),
        "alpha_complex": float(ALPHA_COMPLEX),
        "weight_real": float(WEIGHT_REAL),
        "eps": float(fusion.eps.detach().cpu()),
        "feature_mean": torch.from_numpy(
            np.asarray(data["fmean"], dtype=np.float32)
        ),
        "feature_std": torch.from_numpy(
            np.asarray(data["fstd"], dtype=np.float32)
        ),
        "classes": list(data["classes"]),
        "prototypes": torch.from_numpy(prototypes["fusion"]),
        "unknown_threshold": unknown_threshold,
        "temperature": float(temperature["temperature"]),
        "open_set": _lof_ensemble_tensor_bundle(
            selected_components,
            unknown_threshold,
        ),
        "development_only": True,
        "provenance": {
            "release_evidence": False,
            "consumed_test_rows_used": 0,
            "source_dir": str(source),
            "source_hashes": source_hashes,
            "cache_contract": data["cache_contract"],
            "center_fit_population": "training only",
            "prototype_population": "enrollment only",
            "temperature_calibration": temperature,
            "unknown_calibration_population": "enrollment only",
            "authoritative_label": "centered fusion nearest prototype",
            "rejection_policy": (
                "fused label plus positive weighted real/complex "
                "enrollment-ranked LOF"
            ),
        },
    }
    torch.save(runtime_bundle, runtime_path)
    reloaded_runtime = load_runtime_bundle(runtime_path)
    reloaded_real = InvariantPatchCNN(
        InvariantPatchConfig(**reloaded_runtime["real_config"])
    )
    reloaded_complex = InvariantPatchCNN(
        InvariantPatchConfig(**reloaded_runtime["complex_config"])
    )
    reloaded_real.load_state_dict(
        reloaded_runtime["real_state_dict"], strict=True
    )
    reloaded_complex.load_state_dict(
        reloaded_runtime["complex_state_dict"], strict=True
    )
    reloaded_fusion = CenteredInvariantFusion(
        reloaded_real,
        reloaded_complex,
        reloaded_runtime["real_center"],
        reloaded_runtime["complex_center"],
        alpha_real=reloaded_runtime["alpha_real"],
        alpha_complex=reloaded_runtime["alpha_complex"],
        weight_real=reloaded_runtime["weight_real"],
        eps=reloaded_runtime["eps"],
    ).eval()
    runtime_reload_embedding = _embed_all(
        reloaded_fusion,
        data["xva"][:16],
        data["fva"][:16],
        torch.device("cpu"),
    )
    runtime_reload_error = float(
        np.max(
            np.abs(
                runtime_reload_embedding
                - embeddings["fusion"]["va"][:16]
            )
        )
    )
    if runtime_reload_error > 1e-4:
        raise RuntimeError(
            f"serialized runtime model mismatch: {runtime_reload_error}"
        )

    reloaded_components, reloaded_threshold = _reload_lof_ensemble(
        lof_path
    )
    original_probe_score = _score_lof_ensemble(
        selected_components,
        {
            branch: embeddings[branch]["va"][:64]
            for branch in ("real", "complex", "fusion")
        },
    )
    reloaded_probe_score = _score_lof_ensemble(
        reloaded_components,
        {
            branch: embeddings[branch]["va"][:64]
            for branch in ("real", "complex", "fusion")
        },
    )
    lof_reload_error = float(
        np.max(np.abs(reloaded_probe_score - original_probe_score))
    )
    if lof_reload_error != 0.0 or reloaded_threshold != unknown_threshold:
        raise RuntimeError(f"serialized LOF mismatch: {lof_reload_error}")

    artifact_hashes = {
        path.name: _sha256(path)
        for path in (
            state_path,
            prototypes_path,
            feature_path,
            lof_path,
            runtime_path,
        )
    }
    bundle_schema = {
        "schema": 2,
        "kind": "invariant-centered-fusion-with-lof-rank-ensemble",
        "authoritative_label": {
            "embedding": "centered_fusion",
            "prototype_asset": prototypes_path.name,
            "distance": "squared_euclidean",
        },
        "fusion": {
            "state_dict_asset": state_path.name,
            "trusted_runtime_bundle_asset": runtime_path.name,
            "alpha_real": ALPHA_REAL,
            "alpha_complex": ALPHA_COMPLEX,
            "weight_real": WEIGHT_REAL,
            "centers": "state_dict buffers real_center and complex_center",
            "center_fit_population": "training only",
            "real_branch": architectures["real"],
            "complex_branch": architectures["complex"],
        },
        "preprocess": {
            "contract": data["cache_contract"],
            "feature_standardization_asset": feature_path.name,
        },
        "rejection": {
            "policy": "fused_label_lof_rank_ensemble_rejector",
            "asset": lof_path.name,
            "score": (
                "positive convex weighted sum of enrollment-ranked "
                "local outlier factors"
            ),
            "components": [
                {
                    "branch_source": branch,
                    "weight": weight,
                    "neighbors": scorer.neighbors,
                    "array_schema": {
                        name: _array_schema(value)
                        for name, value in _scorer_arrays(scorer).items()
                    },
                }
                for branch, weight, scorer in selected_components
            ],
            "strictly_monotone_in_each_component": True,
            "threshold_quantile": UNKNOWN_QUANTILE,
            "threshold": unknown_threshold,
            "density_fit_population": "training only",
            "rank_and_threshold_population": "enrollment only",
            "selection_or_novelty_used_for_hyperparameter_choice": True,
            "label_is_unchanged_before_optional_unknown_abstention": True,
        },
        "temperature_calibration": temperature,
        "classes": list(data["classes"]),
        "n_features": int(data["n_features"]),
        "packed_length": int(data["input_length"]),
        "artifact_sha256": artifact_hashes,
    }
    _write_json(output / "bundle_manifest.json", bundle_schema)

    report = {
        "status": "complete",
        "development_only": True,
        "release_evidence": False,
        "consumed_test_rows_used": 0,
        "candidate": bundle_schema,
        "closed_selection": closed,
        "fewshot_selection": fewshot,
        "open_set_development": open_set,
        "canonical_length_development": length,
        "canonical_scale_development": scale,
        "gates": gates,
        "all_classifier_selection_gates_pass": selection_gates_pass,
        "ship_status": {
            "ready_for_runtime_parity": selection_gates_pass,
            "release_ready": False,
            "remaining_required": [
                "Python-TypeScript end-to-end parity <= 1e-4",
                "newly sealed synthetic length/scale release population",
                "real SDR validation",
            ],
        },
        "validation": {
            "numpy_vs_module_fusion_max_abs": fusion_assembly_error,
            "serialized_runtime_embedding_max_abs": runtime_reload_error,
            "serialized_lof_score_max_abs": lof_reload_error,
            "fused_prediction_sha256": hashlib.sha256(
                np.asarray(fused_prediction, dtype="<i8").tobytes()
            ).hexdigest(),
        },
        "provenance": {
            "source_dir": source,
            "source_hashes": source_hashes,
            "artifact_hashes": artifact_hashes,
            "cache_contract": data["cache_contract"],
            "data_audit": data["data_audit"],
            "git_head": _git_head(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": device,
            "seed": args.seed,
            "novelty_seed": args.novelty_seed,
            "fewshot_seed": args.fewshot_seed,
        },
    }
    _write_json(output / "candidate_dev_report.json", report)
    _write_json(
        output / "DEVELOPMENT_ONLY.json",
        {
            "development_only": True,
            "release_evidence": False,
            "consumed_test_rows_used": 0,
            "reason": (
                "assembly and validation use train, enrollment, and the "
                "immutable selection half only"
            ),
            "remaining_release_evidence": report["ship_status"][
                "remaining_required"
            ],
        },
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps"), default="cpu"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--novelty-seed", type=int, default=NOVELTY_SEED)
    parser.add_argument("--novelty-n", type=int, default=300)
    parser.add_argument("--fewshot-seed", type=int, default=FEWSHOT_SEED)
    parser.add_argument("--fewshot-k", type=int, default=FEWSHOT_K)
    parser.add_argument("--fewshot-trials", type=int, default=FEWSHOT_TRIALS)
    return parser


if __name__ == "__main__":
    result = run(build_parser().parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "all_classifier_selection_gates_pass": result[
                    "all_classifier_selection_gates_pass"
                ],
                "closed_accuracy": result["closed_selection"]["accuracy"],
                "open_auroc": result["open_set_development"][
                    "selected_policy"
                ]["auroc_overall"],
                "output": str(DEFAULT_OUTPUT),
            },
            indent=2,
        )
    )
