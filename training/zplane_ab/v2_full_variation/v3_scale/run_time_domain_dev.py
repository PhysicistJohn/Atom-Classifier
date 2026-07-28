"""Development-only retraining and population scale audit for time-domain pose.

This experiment is intentionally isolated from the frozen v2 candidate and from
all release-suite directories.  It reconstructs only the existing train,
enrollment, and immutable development-selection rows exposed by
``invariant_patch_data``.  Every model input uses
``time_domain_invariant_patch_preprocess``; the occupied-band pose therefore
contains no FFT, Welch transform, STFT, or frequency grid.

Unlike the historical canonical scale probe, the scale audit operates on the
metadata-safe subset of the complete raw development-selection population.  It
uses the same deterministic linear-resampling definition as the release gate,
including common no-alias eligibility and paired prediction/embedding metrics.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
V2 = HERE.parent
TRAINING = V2.parent.parent
for path in (TRAINING, V2.parent, V2):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import invariant_patch_data as invariant_data  # noqa: E402
import preprocess as production_preprocess  # noqa: E402
import run_invariant_cnn_dev as runner  # noqa: E402
import time_domain_invariant_patch_preprocess as td_preprocess  # noqa: E402
from invariant_patch_cnn import InvariantPatchCNN, InvariantPatchConfig  # noqa: E402
from train import embed_all, nearest, prototypes_from  # noqa: E402


CORPUS = TRAINING / "artifacts" / "signallab-corpus"
DEFAULT_OUTPUT_ROOT = (
    V2 / "artifacts" / "invariant_patch" / "v3_scale_time_domain_dev"
)
SCALE_FACTORS = (0.5, 0.75, 1.0, 1.5, 2.0)
LENGTHS = (4096, 8192, 16384)
SCALE_EDGE_MARGIN = 1.0 / 512.0
SCALE_MIN_PER_CLASS = 20
HIGH_SNR_DB = 18.0

# --- checkpoint selection policy (HANDOFF 23.1/23.4) -----------------------
#
# The historical loop kept whichever eval checkpoint maximised closed-set
# selection balanced accuracy, the one criterion that trades directly against
# novelty detection.  ``closed`` preserves that behaviour bit-for-bit.
# ``closed-with-floor`` still maximises the same closed-set score, but only
# among checkpoints whose novelty proxy has not fallen more than
# ``NOVELTY_FLOOR_FRACTION`` below its running maximum at the time the
# checkpoint was evaluated (prefix-max semantics: a later rise in the proxy
# never retroactively disqualifies an earlier checkpoint).
SELECTION_POLICIES = ("closed", "closed-with-floor")

# Pre-declared constant, fixed BEFORE any comparison run and not fitted to any
# observed training curve or novelty result.  Do not adjust it after seeing a
# run; that would repeat the gate-weakening failure mode rule 4 forbids.
NOVELTY_FLOOR_FRACTION = 0.25

# The proxy driving the floor.  HANDOFF 22.3 identified the 8k chirp collapse
# mechanism as the encoder mapping every input, in-class or not, ever more
# confidently onto the learned class manifold, at which point the branch LOF
# no longer sees out-of-class rows as outliers.  That collapse is visible on
# the KNOWN populations alone as dimensional collapse of the embedding cloud,
# so no noise/chirp generation and no novelty seed is needed during training.
# Effective rank (the exponential of the Shannon entropy of the centered
# variance spectrum) of the held-out selection embeddings measures how many
# directions the representation still spans; when it falls, the off-manifold
# room that novelty rejection depends on shrinks.  It is deterministic,
# consumes no RNG, and reuses embeddings already computed at every eval.
# The nearest-prototype distance statistics are recorded alongside it for
# visibility but do not drive the floor, because they also shrink under
# legitimate closed-set improvement.
NOVELTY_PROXY_KEY = "selection_embedding_effective_rank"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
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


def _load_manifest() -> dict[str, Any]:
    with (CORPUS / "corpus.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if int(manifest["sampleCount"]) != 16384:
        raise ValueError(
            "development scale experiment expects the historical N=16384 corpus"
        )
    return manifest


def _raw_memmap(manifest: dict[str, Any]) -> np.memmap:
    return np.memmap(
        CORPUS / "corpus.f32",
        dtype="<f4",
        mode="r",
        shape=(
            int(manifest["count"]),
            int(manifest["sampleCount"]),
            2,
        ),
    )


def _complex_row(raw: np.memmap, index: int) -> np.ndarray:
    pair = np.asarray(raw[int(index)], dtype=np.float64)
    return pair[:, 0] + 1j * pair[:, 1]


def _nonzero_prefix_mask(
    captures: Sequence[np.ndarray],
    length: int,
) -> np.ndarray:
    if length <= 0:
        raise ValueError("prefix length must be positive")
    mask = []
    for capture in captures:
        value = np.asarray(capture)
        if value.ndim != 1 or len(value) < length:
            raise ValueError("capture is shorter than the requested prefix")
        mask.append(float(np.max(np.abs(value[:length]))) > 0.0)
    return np.asarray(mask, dtype=np.bool_)


def _preprocess_rows(
    captures: Sequence[np.ndarray],
    *,
    patch_length: int,
    patch_count: int,
    target_frac: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    packed = np.empty(
        (len(captures), 2, patch_length * patch_count),
        dtype=np.float32,
    )
    features = np.empty((len(captures), 12), dtype=np.float32)
    contexts: list[dict[str, Any]] = []
    for row, capture in enumerate(captures):
        packed[row], features[row], context = td_preprocess.preprocess(
            capture,
            patch_length=patch_length,
            patch_count=patch_count,
            target_frac=target_frac,
        )
        contexts.append(context)
    return packed, features, contexts


def _prepare_data(
    source: dict[str, Any],
    manifest: dict[str, Any],
    *,
    patch_length: int,
    patch_count: int,
    target_frac: float,
    train_lengths: tuple[int, ...] | None = None,
) -> tuple[
    dict[str, Any],
    dict[str, list[dict[str, Any]]],
    dict[str, Any],
]:
    raw = _raw_memmap(manifest)
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    contexts: dict[str, list[dict[str, Any]]] = {}
    training_view_audit: dict[str, Any] = {
        "enabled": train_lengths is not None,
        "source": "exposed training rows only",
    }
    for name, key in (
        ("train", "tr_idx"),
        ("enroll", "en_idx"),
        ("selection", "va_idx"),
    ):
        corpus_indices = np.asarray(source[key], dtype=np.int64)
        captures = [_complex_row(raw, int(index)) for index in corpus_indices]
        if name == "train" and train_lengths is not None:
            lengths = tuple(int(length) for length in train_lengths)
            if (
                not lengths
                or any(length <= 0 for length in lengths)
                or tuple(sorted(set(lengths))) != lengths
                or max(lengths) > int(manifest["sampleCount"])
            ):
                raise ValueError(
                    "train_lengths must be unique increasing supported lengths"
                )
            eligible_mask = _nonzero_prefix_mask(captures, min(lengths))
            source_positions = np.flatnonzero(eligible_mask)
            excluded_positions = np.flatnonzero(~eligible_mask)
            expanded_captures: list[np.ndarray] = []
            expanded_positions: list[int] = []
            for position in source_positions:
                for length in lengths:
                    expanded_captures.append(captures[int(position)][:length])
                    expanded_positions.append(int(position))
            captures = expanded_captures
            position_array = np.asarray(expanded_positions, dtype=np.int64)
            arrays_ytr = np.asarray(source["ytr"])[position_array]
            arrays_imp_tr = np.asarray(source["imp_tr"])[position_array]
            arrays_tr_idx = corpus_indices[position_array]
            training_view_audit.update(
                {
                    "lengths": list(lengths),
                    "common_nonzero_minimum_prefix_rule": (
                        f"max(abs(iq[:{min(lengths)}])) > 0"
                    ),
                    "base_rows_total": int(len(corpus_indices)),
                    "base_rows_eligible": int(len(source_positions)),
                    "base_rows_excluded_all_zero_minimum_prefix": int(
                        len(excluded_positions)
                    ),
                    "expanded_views": int(len(captures)),
                    "views_per_eligible_base_row": len(lengths),
                    "expanded_view_order": (
                        "base-row major, then ascending prefix length"
                    ),
                    "eligible_corpus_indices_sha256": hashlib.sha256(
                        corpus_indices[source_positions]
                        .astype("<i8", copy=False)
                        .tobytes()
                    ).hexdigest(),
                    "excluded_by_class": {
                        class_name: int(
                            np.sum(
                                np.asarray(source["ytr"])[excluded_positions]
                                == class_index
                            )
                        )
                        for class_index, class_name in enumerate(source["classes"])
                    },
                    "episode_sampler_rng_contract": (
                        "numpy.default_rng(model_seed); choose K+Q distinct "
                        "eligible base rows per class without replacement, "
                        "then choose one uniform view per selected base"
                    ),
                }
            )
        packed, features, rows = _preprocess_rows(
            captures,
            patch_length=patch_length,
            patch_count=patch_count,
            target_frac=target_frac,
        )
        arrays[name] = packed, features
        contexts[name] = rows
        print(f"[v3 td] preprocessed {name}: {len(captures)}", flush=True)
    del raw

    feature_mean = (
        arrays["train"][1].astype(np.float64).mean(axis=0).astype(np.float32)
    )
    feature_std = (
        arrays["train"][1].astype(np.float64).std(axis=0) + 1e-6
    ).astype(np.float32)

    def standardized(name: str) -> np.ndarray:
        return (
            (arrays[name][1] - feature_mean) / feature_std
        ).astype(np.float32)

    data = {
        **source,
        "xtr": arrays["train"][0],
        "ftr": standardized("train"),
        "xen": arrays["enroll"][0],
        "fen": standardized("enroll"),
        "xva": arrays["selection"][0],
        "fva": standardized("selection"),
        "fmean": feature_mean,
        "fstd": feature_std,
        "condition": td_preprocess.PREPROCESS_VERSION,
    }
    if train_lengths is not None:
        base_labels = np.asarray(source["ytr"])[source_positions]
        view_positions = np.arange(
            len(source_positions) * len(lengths), dtype=np.int64
        ).reshape(len(source_positions), len(lengths))
        data.update(
            {
                "ytr": arrays_ytr.astype(np.int64, copy=False),
                "imp_tr": arrays_imp_tr.astype(np.bool_, copy=False),
                "tr_idx": arrays_tr_idx.astype(np.int64, copy=False),
                "idx_by_class": [
                    np.where(arrays_ytr == class_index)[0]
                    for class_index in range(int(source["n_classes"]))
                ],
                "multiview_view_positions": view_positions,
                "multiview_base_labels": base_labels.astype(
                    np.int64, copy=False
                ),
                "multiview_base_indices_by_class": [
                    np.where(base_labels == class_index)[0]
                    for class_index in range(int(source["n_classes"]))
                ],
            }
        )
    return data, contexts, training_view_audit


def _sample_multiview_episode(
    base_by_class: Sequence[np.ndarray],
    view_positions: np.ndarray,
    rng: np.random.Generator,
    *,
    per_class: int,
    include_secondary: bool,
    secondary_rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Choose distinct base rows first, then one or two views of each base."""
    if per_class <= 0:
        raise ValueError("per_class must be positive")
    view_map = np.asarray(view_positions, dtype=np.int64)
    if view_map.ndim != 2 or view_map.shape[1] < 2:
        raise ValueError("view_positions must have at least two views")
    classes = [np.asarray(rows, dtype=np.int64) for rows in base_by_class]
    if not classes or any(len(rows) < per_class for rows in classes):
        raise ValueError("each class must contain per_class distinct base rows")
    selected_base = np.concatenate(
        [rng.choice(rows, per_class, replace=False) for rows in classes]
    )
    selected_view = rng.integers(
        0, view_map.shape[1], size=len(selected_base)
    )
    result = {
        "base": selected_base,
        "view": selected_view,
        "primary": view_map[selected_base, selected_view],
    }
    if include_secondary:
        pair_generator = secondary_rng if secondary_rng is not None else rng
        offset = pair_generator.integers(
            1, view_map.shape[1], size=len(selected_base)
        )
        secondary_view = (selected_view + offset) % view_map.shape[1]
        result["secondary_view"] = secondary_view
        result["secondary"] = view_map[selected_base, secondary_view]
    return result


def _effective_rank(embeddings: np.ndarray) -> float:
    """Exponential of the entropy of the centered variance spectrum.

    A cloud spread isotropically over ``k`` directions scores ``k``; a cloud
    collapsed onto one line scores 1; a fully degenerate cloud (every row
    identical, zero variance) scores 0 by convention.  The SVD runs inside a
    narrow ``errstate`` with an explicit finiteness check because this
    platform's Accelerate BLAS raises spurious IEEE flags on clean float64
    factorizations, which ``PYTHONWARNINGS=error`` would turn into an abort
    (same treatment as ``noise_prefilter._dot``).
    """
    value = np.asarray(embeddings, dtype=np.float64)
    if value.ndim != 2 or len(value) < 2:
        raise ValueError("effective rank requires a (rows>=2, dims) matrix")
    if not np.isfinite(value).all():
        raise ValueError("embeddings contain a non-finite value")
    centered = value - value.mean(axis=0, keepdims=True)
    with np.errstate(all="ignore"):
        singular = np.linalg.svd(centered, compute_uv=False)
        power = singular * singular
        total = float(np.sum(power))
        if total <= 0.0:
            return 0.0
        shares = power / total
        shares = shares[shares > 0.0]
        result = float(np.exp(-np.sum(shares * np.log(shares))))
    if not np.isfinite(result):
        raise FloatingPointError("non-finite effective rank")
    return result


def _novelty_proxies(
    selection_embeddings: np.ndarray,
    enrollment_embeddings: np.ndarray,
    squared_distances: np.ndarray,
) -> dict[str, float]:
    """Seed-free novelty statistics computed on known dev populations only.

    No noise or chirp is generated and no novelty seed is consumed; every
    input already exists at each eval checkpoint.  ``squared_distances`` is
    the (rows, classes) matrix ``nearest`` returns for the selection
    population against the enrollment prototypes.
    """
    squared = np.asarray(squared_distances, dtype=np.float64)
    if squared.ndim != 2:
        raise ValueError("squared_distances must be a (rows, classes) matrix")
    if not np.isfinite(squared).all():
        raise ValueError("squared_distances contain a non-finite value")
    distance = np.sqrt(np.maximum(squared.min(axis=1), 0.0))
    return {
        NOVELTY_PROXY_KEY: _effective_rank(selection_embeddings),
        "enrollment_embedding_effective_rank": _effective_rank(
            enrollment_embeddings
        ),
        "selection_nearest_prototype_distance_mean": float(distance.mean()),
        "selection_nearest_prototype_distance_p05": float(
            np.quantile(distance, 0.05)
        ),
    }


class _CheckpointSelector:
    """Online checkpoint choice under a declared selection policy.

    Both trackers always run so every emitted history shows whether the two
    policies would have chosen different checkpoints:

    * ``closed`` reproduces the historical criterion exactly: keep the
      strictly greatest ``(balanced_accuracy, accuracy)`` tuple.
    * ``closed-with-floor`` keeps the same criterion but only over
      checkpoints whose novelty proxy is at least
      ``(1 - floor_fraction) * running_max`` at the time they are evaluated.
      The first checkpoint is always eligible, so a selection always exists.
    """

    def __init__(
        self,
        policy: str,
        *,
        floor_fraction: float = NOVELTY_FLOOR_FRACTION,
        proxy_key: str = NOVELTY_PROXY_KEY,
    ) -> None:
        if policy not in SELECTION_POLICIES:
            raise ValueError(f"unknown selection policy: {policy!r}")
        fraction = float(floor_fraction)
        if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
            raise ValueError("floor_fraction must lie strictly inside (0, 1)")
        self.policy = policy
        self.floor_fraction = fraction
        self.proxy_key = str(proxy_key)
        self.closed_best: tuple[float, float] = (-1.0, -1.0)
        self.closed_best_episode: int | None = None
        self.floor_best: tuple[float, float] = (-1.0, -1.0)
        self.floor_best_episode: int | None = None
        self.running_proxy_max: float | None = None

    def observe(
        self,
        episode: int,
        *,
        accuracy: float,
        balanced: float,
        proxies: dict[str, float],
    ) -> tuple[bool, bool]:
        """Record one eval checkpoint.

        Returns ``(selected_improved, floor_eligible)`` where
        ``selected_improved`` means the ACTIVE policy wants this checkpoint's
        weights saved as the new best.
        """
        score = (float(balanced), float(accuracy))
        closed_improved = score > self.closed_best
        if closed_improved:
            self.closed_best = score
            self.closed_best_episode = int(episode)
        proxy = float(proxies[self.proxy_key])
        if not np.isfinite(proxy):
            raise ValueError("novelty proxy must be finite")
        if self.running_proxy_max is None or proxy > self.running_proxy_max:
            self.running_proxy_max = proxy
        eligible = proxy >= (
            (1.0 - self.floor_fraction) * self.running_proxy_max
        )
        floor_improved = eligible and score > self.floor_best
        if floor_improved:
            self.floor_best = score
            self.floor_best_episode = int(episode)
        if self.policy == "closed":
            return closed_improved, eligible
        return floor_improved, eligible

    @property
    def selected_best(self) -> tuple[float, float]:
        return self.closed_best if self.policy == "closed" else self.floor_best

    @property
    def selected_best_episode(self) -> int | None:
        if self.policy == "closed":
            return self.closed_best_episode
        return self.floor_best_episode

    def summary(self) -> dict[str, Any]:
        return {
            "selection_policy": self.policy,
            "novelty_proxy_key": self.proxy_key,
            "novelty_floor_fraction": self.floor_fraction,
            "novelty_floor_contract": (
                "a checkpoint is floor-eligible when its novelty proxy is at "
                "least (1 - fraction) * the running proxy maximum at the time "
                "it is evaluated; the fraction is a pre-declared constant, "
                "never fitted to a result; proxies are computed on known "
                "train/enroll/selection populations only and consume no "
                "novelty seed and no RNG"
            ),
            "selected_checkpoint_episode": self.selected_best_episode,
            "closed_best_episode": self.closed_best_episode,
            "closed_best_balanced_accuracy": self.closed_best[0],
            "floor_best_episode": self.floor_best_episode,
            "floor_best_balanced_accuracy": self.floor_best[0],
        }


def _train_model_multiview(
    net: InvariantPatchCNN,
    data: dict[str, Any],
    device: torch.device,
    *,
    episodes: int,
    eval_every: int,
    seed: int,
    lr: float,
    weight_decay: float,
    warmup_frac: float,
    label_smoothing: float,
    phase_augmentation: bool,
    consistency_weight: float,
    selection_policy: str = "closed",
    novelty_floor_fraction: float = NOVELTY_FLOOR_FRACTION,
) -> tuple[InvariantPatchCNN, dict[str, Any]]:
    """Train on distinct base rows, selecting one prefix view per episode.

    A second, different prefix of each selected base row is embedded only when
    ``consistency_weight`` is positive.  It never enters the support/query
    classification sets, so one physical capture cannot masquerade as two
    independent prototypical examples.

    Novelty proxies are recorded at every eval checkpoint regardless of
    ``selection_policy``; the default ``closed`` policy consumes the same RNG
    stream and keeps the same checkpoint as the historical loop.
    """
    if episodes <= 0 or eval_every <= 0:
        raise ValueError("episodes and eval_every must be positive")
    if not np.isfinite(consistency_weight) or consistency_weight < 0.0:
        raise ValueError("consistency_weight must be finite and non-negative")
    view_positions = np.asarray(
        data["multiview_view_positions"], dtype=np.int64
    )
    if view_positions.ndim != 2 or view_positions.shape[1] < 2:
        raise ValueError("multiview training requires at least two views")
    base_by_class = [
        np.asarray(rows, dtype=np.int64)
        for rows in data["multiview_base_indices_by_class"]
    ]
    needed = runner.K_SHOT + runner.Q_QUERY
    if any(len(rows) < needed for rows in base_by_class):
        raise ValueError("a class has too few distinct base rows")

    rng = np.random.default_rng(seed)
    secondary_rng = np.random.default_rng(seed + 1_000_003)
    net = net.to(device)
    log_scale = torch.nn.Parameter(
        torch.tensor(math.log(10.0), dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(
        [
            {"params": list(net.parameters()), "weight_decay": weight_decay},
            {"params": [log_scale], "weight_decay": 0.0},
        ],
        lr=lr,
    )
    warmup = max(1, int(episodes * warmup_frac))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(episodes - warmup, 1)
    )

    def evaluate_selection() -> tuple[float, float, dict[str, float]]:
        enrollment = embed_all(net, data["xen"], data["fen"], device)
        prototypes = prototypes_from(
            enrollment, data["yen"], data["n_classes"]
        )
        selection = embed_all(net, data["xva"], data["fva"], device)
        prediction, squared_distances = nearest(selection, prototypes)
        recalls = [
            np.mean(prediction[data["yva"] == class_index] == class_index)
            for class_index in range(data["n_classes"])
        ]
        proxies = _novelty_proxies(selection, enrollment, squared_distances)
        return (
            float(np.mean(prediction == data["yva"])),
            float(np.mean(recalls)),
            proxies,
        )

    history: dict[str, Any] = {
        "loss": [],
        "loss_ce": [],
        "loss_consistency": [],
        "selection": [],
        "logit_scale": [],
    }
    selector = _CheckpointSelector(
        selection_policy, floor_fraction=novelty_floor_fraction
    )
    best_state: tuple[dict[str, torch.Tensor], float] | None = None
    running = {"total": 0.0, "ce": 0.0, "consistency": 0.0}
    started = time.perf_counter()

    for episode in range(episodes):
        net.train()
        if episode < warmup:
            for group in optimizer.param_groups:
                group["lr"] = lr * (episode + 1) / warmup

        use_consistency = consistency_weight > 0.0
        episode_sample = _sample_multiview_episode(
            base_by_class,
            view_positions,
            rng,
            per_class=needed,
            include_secondary=use_consistency,
            secondary_rng=secondary_rng,
        )
        primary_positions = episode_sample["primary"]
        step = needed
        support_positions = np.concatenate(
            [
                primary_positions[
                    class_index * step : class_index * step + runner.K_SHOT
                ]
                for class_index in range(data["n_classes"])
            ]
        )
        query_positions = np.concatenate(
            [
                primary_positions[
                    class_index * step + runner.K_SHOT :
                    (class_index + 1) * step
                ]
                for class_index in range(data["n_classes"])
            ]
        )
        primary_order = np.concatenate((support_positions, query_positions))
        support_labels = np.repeat(
            np.arange(data["n_classes"]), runner.K_SHOT
        )
        query_labels = np.repeat(
            np.arange(data["n_classes"]), runner.Q_QUERY
        )

        if use_consistency:
            secondary_by_class = episode_sample["secondary"]
            secondary_support = np.concatenate(
                [
                    secondary_by_class[
                        class_index * step :
                        class_index * step + runner.K_SHOT
                    ]
                    for class_index in range(data["n_classes"])
                ]
            )
            secondary_query = np.concatenate(
                [
                    secondary_by_class[
                        class_index * step + runner.K_SHOT :
                        (class_index + 1) * step
                    ]
                    for class_index in range(data["n_classes"])
                ]
            )
            secondary_order = np.concatenate(
                (secondary_support, secondary_query)
            )
            all_positions = np.concatenate(
                (primary_order, secondary_order)
            )
        else:
            all_positions = primary_order

        x = torch.from_numpy(np.asarray(data["xtr"])[all_positions]).to(device)
        features = torch.from_numpy(
            np.asarray(data["ftr"])[all_positions]
        ).to(device)
        if phase_augmentation:
            x = runner._phase_augment(x)
        all_embeddings = net(x, features)
        primary_embeddings = all_embeddings[: len(primary_order)]
        support_embeddings = primary_embeddings[: len(support_positions)]
        query_embeddings = primary_embeddings[len(support_positions) :]
        support_labels_t = torch.from_numpy(support_labels).to(device)
        prototypes = torch.stack(
            [
                support_embeddings[support_labels_t == class_index].mean(0)
                for class_index in range(data["n_classes"])
            ]
        )
        logits = (
            -runner.sq_dist(query_embeddings, prototypes)
            * log_scale.exp().clamp(1e-3, 100.0)
        )
        loss_ce = F.cross_entropy(
            logits,
            torch.from_numpy(query_labels).to(device),
            label_smoothing=label_smoothing,
        )
        if use_consistency:
            secondary_embeddings = all_embeddings[len(primary_order) :]
            loss_consistency = (
                1.0
                - torch.sum(
                    primary_embeddings * secondary_embeddings, dim=-1
                )
            ).mean()
        else:
            loss_consistency = primary_embeddings.new_zeros(())
        loss = loss_ce + consistency_weight * loss_consistency
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if episode >= warmup:
            scheduler.step()
        running["total"] += float(loss.detach().cpu())
        running["ce"] += float(loss_ce.detach().cpu())
        running["consistency"] += float(loss_consistency.detach().cpu())

        log_every = max(1, min(250, episodes))
        if (episode + 1) % log_every == 0:
            means = {
                name: value / log_every for name, value in running.items()
            }
            running = {"total": 0.0, "ce": 0.0, "consistency": 0.0}
            history["loss"].append(
                {"episode": episode + 1, "value": means["total"]}
            )
            history["loss_ce"].append(
                {"episode": episode + 1, "value": means["ce"]}
            )
            history["loss_consistency"].append(
                {
                    "episode": episode + 1,
                    "value": means["consistency"],
                }
            )
            history["logit_scale"].append(
                {
                    "episode": episode + 1,
                    "value": float(log_scale.detach().exp().cpu()),
                }
            )
            elapsed = time.perf_counter() - started
            eta = elapsed / (episode + 1) * (episodes - episode - 1) / 60.0
            print(
                f"  [{net.cfg.encoder}/multiview] ep {episode + 1:5d}/"
                f"{episodes} loss {means['total']:.4f} "
                f"ce {means['ce']:.4f} con {means['consistency']:.4f} "
                f"eta {eta:.1f}m",
                flush=True,
            )

        if (episode + 1) % eval_every == 0 or episode + 1 == episodes:
            accuracy, balanced, proxies = evaluate_selection()
            selected_improved, floor_eligible = selector.observe(
                episode + 1,
                accuracy=accuracy,
                balanced=balanced,
                proxies=proxies,
            )
            history["selection"].append(
                {
                    "episode": episode + 1,
                    "accuracy": accuracy,
                    "balanced_accuracy": balanced,
                    "novelty_proxies": proxies,
                    "novelty_floor_eligible": floor_eligible,
                }
            )
            marker = ""
            if selected_improved:
                best_state = (
                    copy.deepcopy(net.state_dict()),
                    float(log_scale.detach().cpu()),
                )
                marker = " <- best"
            print(
                f"    [{net.cfg.encoder}/multiview] selection "
                f"acc {accuracy:.4f} bal {balanced:.4f}{marker}",
                flush=True,
            )
            print(
                f"    [{net.cfg.encoder}/multiview] novelty proxy "
                f"erank {proxies[NOVELTY_PROXY_KEY]:.3f} "
                f"protodist {proxies['selection_nearest_prototype_distance_mean']:.4f} "
                f"floor_eligible {floor_eligible}",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("training completed without a selection checkpoint")
    net.load_state_dict(best_state[0], strict=True)
    history.update(selector.summary())
    history.update(
        {
            "episodes": episodes,
            "best_selection_balanced_accuracy": selector.selected_best[0],
            "best_selection_accuracy": selector.selected_best[1],
            "final_logit_scale": float(math.exp(best_state[1])),
            "wall_clock_s": time.perf_counter() - started,
            "view_consistency_weight": float(consistency_weight),
            "sampler_contract": (
                "numpy.default_rng(seed); per episode choose K+Q distinct base "
                "rows per class without replacement; choose one uniform prefix "
                "view per base; independent numpy.default_rng(seed+1000003) "
                "chooses an optional uniform nonzero cyclic view offset that "
                "enters only cosine consistency"
            ),
        }
    )
    return net.eval(), history


def _mean_pairwise_cosine(embeddings: np.ndarray) -> float:
    value = np.asarray(embeddings, dtype=np.float64)
    value = value / np.maximum(
        np.linalg.norm(value, axis=1, keepdims=True), 1e-12
    )
    upper = np.triu_indices(len(value), 1)
    # Accelerate-backed matrix multiplication can surface stale floating-point
    # status flags under warnings-as-errors on macOS.  The explicit contraction
    # is equivalent here and keeps genuine non-finite checks meaningful.
    gram = np.einsum("nd,md->nm", value, value, optimize=False)
    return float(np.mean(gram[upper]))


def _classification_report(
    embeddings: np.ndarray,
    labels: np.ndarray,
    prototypes: np.ndarray,
    classes: Sequence[str],
    *,
    impaired: np.ndarray | None = None,
    snr_db: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    prediction, _distance = nearest(embeddings, prototypes)
    recalls = {
        name: float(np.mean(prediction[labels == index] == index))
        for index, name in enumerate(classes)
    }
    result: dict[str, Any] = {
        "n": int(len(labels)),
        "accuracy": float(np.mean(prediction == labels)),
        "balanced_accuracy": float(np.mean(list(recalls.values()))),
        "per_class_recall": recalls,
        "mean_pairwise_cosine": _mean_pairwise_cosine(embeddings),
    }
    if impaired is not None:
        result["clean_accuracy"] = float(
            np.mean(prediction[~impaired] == labels[~impaired])
        )
        result["impaired_accuracy"] = float(
            np.mean(prediction[impaired] == labels[impaired])
        )
    if snr_db is not None:
        high = snr_db >= HIGH_SNR_DB
        result["high_snr_accuracy"] = float(
            np.mean(prediction[high] == labels[high])
        )
    return result, prediction


def _paired_report(
    embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    prediction: np.ndarray,
    reference_prediction: np.ndarray,
    labels: np.ndarray,
    classes: Sequence[str],
) -> dict[str, Any]:
    current = np.asarray(embeddings, dtype=np.float64)
    reference = np.asarray(reference_embeddings, dtype=np.float64)
    current /= np.maximum(
        np.linalg.norm(current, axis=1, keepdims=True), 1e-12
    )
    reference /= np.maximum(
        np.linalg.norm(reference, axis=1, keepdims=True), 1e-12
    )
    cosine = np.clip(np.sum(current * reference, axis=1), -1.0, 1.0)
    agreement = prediction == reference_prediction
    return {
        "prediction_agreement": float(np.mean(agreement)),
        "embedding_cosine_mean": float(np.mean(cosine)),
        "embedding_cosine_p05": float(np.quantile(cosine, 0.05)),
        "per_class": {
            name: {
                "n": int(np.sum(labels == index)),
                "prediction_agreement": float(
                    np.mean(agreement[labels == index])
                ),
                "embedding_cosine_mean": float(
                    np.mean(cosine[labels == index])
                ),
            }
            for index, name in enumerate(classes)
        },
    }


def _metadata_arrays(
    manifest: dict[str, Any],
    indices: np.ndarray,
    classes: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    class_index = {name: index for index, name in enumerate(classes)}
    labels = np.asarray(
        [
            class_index[str(manifest["items"][int(index)]["cls"])]
            for index in indices
        ],
        dtype=np.int64,
    )
    impaired = np.asarray(
        [
            bool(manifest["items"][int(index)]["impaired"])
            for index in indices
        ],
        dtype=np.bool_,
    )
    snr = np.asarray(
        [
            float(manifest["items"][int(index)]["snrDb"])
            for index in indices
        ],
        dtype=np.float64,
    )
    return labels, impaired, snr


def _scale_eligible(
    manifest: dict[str, Any],
    indices: np.ndarray,
    classes: Sequence[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    base_length = int(manifest["sampleCount"])
    effective = {
        factor: (
            (base_length - 1)
            / (max(64, int(round(base_length / factor))) - 1)
        )
        for factor in SCALE_FACTORS
    }
    maximum = max(effective.values())
    included: list[int] = []
    counts = {name: 0 for name in classes}
    excluded = {name: 0 for name in classes}
    for index in np.asarray(indices, dtype=np.int64):
        item = manifest["items"][int(index)]
        center = abs(float(item["centreOffsetFrac"]))
        bandwidth = float(item["bandwidthHz"]) / float(item["sampleRateHz"])
        class_name = str(item["cls"])
        if maximum * (center + 0.5 * bandwidth) < 0.5 - SCALE_EDGE_MARGIN:
            included.append(int(index))
            counts[class_name] += 1
        else:
            excluded[class_name] += 1
    if any(value < SCALE_MIN_PER_CLASS for value in counts.values()):
        raise ValueError(f"scale eligibility has too few rows: {counts}")
    output = np.asarray(included, dtype=np.int64)
    return output, {
        "eligible_by_class": counts,
        "excluded_by_class": excluded,
        "effective_scale_by_requested_factor": {
            str(key): value for key, value in effective.items()
        },
        "eligible_indices_sha256": hashlib.sha256(
            output.astype("<i8", copy=False).tobytes()
        ).hexdigest(),
    }


def _embed_raw(
    net: InvariantPatchCNN,
    captures: Sequence[np.ndarray],
    data: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    config = net.cfg
    packed, raw_features, contexts = _preprocess_rows(
        captures,
        patch_length=config.patch_length,
        patch_count=config.patch_count,
        target_frac=float(data["cache_contract"]["config"]["target_frac"]),
    )
    features = (
        (raw_features - np.asarray(data["fmean"]))
        / np.asarray(data["fstd"])
    ).astype(np.float32)
    return embed_all(net, packed, features, device), contexts


def _scale_audit(
    net: InvariantPatchCNN,
    prototypes: np.ndarray,
    data: dict[str, Any],
    manifest: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    eligible, eligibility = _scale_eligible(
        manifest,
        np.asarray(data["va_idx"], dtype=np.int64),
        data["classes"],
    )
    labels, _impaired, _snr = _metadata_arrays(
        manifest, eligible, data["classes"]
    )
    raw = _raw_memmap(manifest)
    base = [_complex_row(raw, int(index)) for index in eligible]
    del raw
    base_length = int(manifest["sampleCount"])
    rows: list[dict[str, Any]] = []
    embeddings_by_factor: dict[float, np.ndarray] = {}
    predictions_by_factor: dict[float, np.ndarray] = {}
    for factor in SCALE_FACTORS:
        new_length = max(64, int(round(base_length / factor)))
        effective = (base_length - 1) / (new_length - 1)
        captures = (
            base
            if factor == 1.0
            else [
                production_preprocess.lin_resample(capture, new_length)
                for capture in base
            ]
        )
        embeddings, contexts = _embed_raw(net, captures, data, device)
        report, prediction = _classification_report(
            embeddings,
            labels,
            prototypes,
            data["classes"],
        )
        embeddings_by_factor[factor] = embeddings
        predictions_by_factor[factor] = prediction
        rows.append(
            {
                "factor": factor,
                "effective_factor": effective,
                "raw_length": new_length,
                **report,
                "estimated_bandwidth_median": float(
                    np.median([row["bw"] for row in contexts])
                ),
            }
        )
        print(
            f"[v3 td] scale {factor:g}: bal={report['balanced_accuracy']:.4f}",
            flush=True,
        )
    reference = embeddings_by_factor[1.0]
    reference_prediction = predictions_by_factor[1.0]
    for row in rows:
        factor = float(row["factor"])
        row["paired_to_factor1"] = _paired_report(
            embeddings_by_factor[factor],
            reference,
            predictions_by_factor[factor],
            reference_prediction,
            labels,
            data["classes"],
        )
    return {
        "definition": (
            "normalized frequencies multiplied by s through deterministic "
            "linear resampling to N/s; common metadata-safe no-alias subset"
        ),
        "development_only": True,
        "eligible_rows": int(len(eligible)),
        **eligibility,
        "rows": rows,
        "worst_balanced_accuracy": min(
            float(row["balanced_accuracy"]) for row in rows
        ),
        "worst_mean_pairwise_cosine": max(
            float(row["mean_pairwise_cosine"]) for row in rows
        ),
        "worst_prediction_agreement_to_factor1": min(
            float(row["paired_to_factor1"]["prediction_agreement"])
            for row in rows
        ),
        "worst_embedding_cosine_to_factor1": min(
            float(row["paired_to_factor1"]["embedding_cosine_mean"])
            for row in rows
        ),
    }


def _length_audit(
    net: InvariantPatchCNN,
    prototypes: np.ndarray,
    data: dict[str, Any],
    manifest: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    all_indices = np.asarray(data["va_idx"], dtype=np.int64)
    raw = _raw_memmap(manifest)
    minimum_length = min(LENGTHS)
    indices_list: list[int] = []
    longest: list[np.ndarray] = []
    included = {name: 0 for name in data["classes"]}
    excluded = {name: 0 for name in data["classes"]}
    for index in all_indices:
        capture = _complex_row(raw, int(index))
        class_name = str(manifest["items"][int(index)]["cls"])
        # The historical development generator allowed a signal to begin after
        # N=4096.  Such an all-zero prefix has no RF pose to classify and is not
        # comparable to the release generator's predeclared occupied-start rule.
        if float(np.max(np.abs(capture[:minimum_length]))) == 0.0:
            excluded[class_name] += 1
            continue
        indices_list.append(int(index))
        longest.append(capture)
        included[class_name] += 1
    del raw
    if any(value < SCALE_MIN_PER_CLASS for value in included.values()):
        raise ValueError(f"length eligibility has too few rows: {included}")
    indices = np.asarray(indices_list, dtype=np.int64)
    labels, _impaired, _snr = _metadata_arrays(
        manifest, indices, data["classes"]
    )
    rows = []
    embeddings_by_length: dict[int, np.ndarray] = {}
    predictions_by_length: dict[int, np.ndarray] = {}
    for length in LENGTHS:
        captures = [capture[:length] for capture in longest]
        embeddings, _contexts = _embed_raw(net, captures, data, device)
        report, prediction = _classification_report(
            embeddings,
            labels,
            prototypes,
            data["classes"],
        )
        embeddings_by_length[length] = embeddings
        predictions_by_length[length] = prediction
        rows.append({"length": length, **report})
        print(
            f"[v3 td] length {length}: bal={report['balanced_accuracy']:.4f}",
            flush=True,
        )
    reference = embeddings_by_length[max(LENGTHS)]
    reference_prediction = predictions_by_length[max(LENGTHS)]
    for row in rows:
        length = int(row["length"])
        row["paired_to_n16384"] = _paired_report(
            embeddings_by_length[length],
            reference,
            predictions_by_length[length],
            reference_prediction,
            labels,
            data["classes"],
        )
    return {
        "definition": "exact observed/clean row prefixes from development corpus",
        "development_only": True,
        "eligible_rows": int(len(indices)),
        "eligible_by_class": included,
        "excluded_all_zero_n4096_prefix_by_class": excluded,
        "eligible_indices_sha256": hashlib.sha256(
            indices.astype("<i8", copy=False).tobytes()
        ).hexdigest(),
        "rows": rows,
        "worst_balanced_accuracy": min(
            float(row["balanced_accuracy"]) for row in rows
        ),
        "worst_prediction_agreement_to_n16384": min(
            float(row["paired_to_n16384"]["prediction_agreement"])
            for row in rows
        ),
        "worst_embedding_cosine_to_n16384": min(
            float(row["paired_to_n16384"]["embedding_cosine_mean"])
            for row in rows
        ),
        "n32768_status": (
            "untested on this historical development corpus; it stores only "
            "N=16384. A longer observation may not be synthesized by padding "
            "or extrapolation."
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(args.output_dir).expanduser().resolve()
    lowered_parts = {part.lower() for part in output.parts}
    if "releases" in lowered_parts or any(
        "sealed" in part for part in lowered_parts
    ):
        raise ValueError(
            "development experiment refuses release or sealed output paths"
        )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} is not empty")
    if args.selection_policy not in SELECTION_POLICIES:
        raise ValueError(
            f"unknown selection policy: {args.selection_policy!r}"
        )
    floor_fraction = float(args.novelty_floor_fraction)
    if not np.isfinite(floor_fraction) or not 0.0 < floor_fraction < 1.0:
        raise ValueError(
            "--novelty-floor-fraction must lie strictly inside (0, 1)"
        )
    if args.selection_policy != "closed" and not args.multilength_train:
        raise ValueError(
            "--selection-policy closed-with-floor requires "
            "--multilength-train; the shared single-length trainer records "
            "no novelty proxies"
        )
    output.mkdir(parents=True, exist_ok=True)
    source = invariant_data.load(
        patch_length=args.patch_length,
        patch_count=args.patch_count,
        target_frac=args.target_frac,
        model_seed=args.seed,
        build_if_missing=False,
    )
    if source["data_audit"].get("consumed_test_rows_exposed") != 0:
        raise AssertionError("data loader exposed consumed test rows")
    manifest = _load_manifest()
    data, contexts, training_view_audit = _prepare_data(
        source,
        manifest,
        patch_length=args.patch_length,
        patch_count=args.patch_count,
        target_frac=args.target_frac,
        train_lengths=LENGTHS if args.multilength_train else None,
    )
    device = runner.resolve_device(args.device)
    runner.seed_everything(args.seed)
    config = InvariantPatchConfig(
        patch_length=args.patch_length,
        patch_count=args.patch_count,
        encoder=args.encoder,
        patch_dim=args.patch_dim,
        hidden=args.hidden,
        set_pool=args.set_pool,
        dropout=args.dropout,
    )
    net = InvariantPatchCNN(config)
    if args.multilength_train:
        net, history = _train_model_multiview(
            net,
            data,
            device,
            episodes=args.episodes,
            eval_every=args.eval_every,
            seed=args.seed,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_frac=args.warmup_frac,
            label_smoothing=args.label_smoothing,
            phase_augmentation=args.phase_augmentation,
            consistency_weight=args.view_consistency_weight,
            selection_policy=args.selection_policy,
            novelty_floor_fraction=floor_fraction,
        )
    else:
        if args.view_consistency_weight != 0.0:
            raise ValueError(
                "view consistency requires --multilength-train"
            )
        net, history = runner.train_model(
            net,
            data,
            device,
            episodes=args.episodes,
            eval_every=args.eval_every,
            seed=args.seed,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_frac=args.warmup_frac,
            label_smoothing=args.label_smoothing,
            phase_augmentation=args.phase_augmentation,
        )
    enrollment = embed_all(net, data["xen"], data["fen"], device)
    prototypes = prototypes_from(
        enrollment, data["yen"], data["n_classes"]
    )
    selection = embed_all(net, data["xva"], data["fva"], device)
    labels = np.asarray(data["yva"], dtype=np.int64)
    selection_indices = np.asarray(data["va_idx"], dtype=np.int64)
    _metadata_labels, impaired, snr = _metadata_arrays(
        manifest, selection_indices, data["classes"]
    )
    if not np.array_equal(labels, _metadata_labels):
        raise AssertionError("selection labels disagree with manifest")
    closed, _prediction = _classification_report(
        selection,
        labels,
        prototypes,
        data["classes"],
        impaired=impaired,
        snr_db=snr,
    )
    print(
        f"[v3 td] closed bal={closed['balanced_accuracy']:.4f} "
        f"clean={closed['clean_accuracy']:.4f}",
        flush=True,
    )
    scale = _scale_audit(net, prototypes, data, manifest, device)
    length = _length_audit(net, prototypes, data, manifest, device)
    state_path = output / f"{args.encoder}_state_dict.pt"
    prototypes_path = output / f"{args.encoder}_prototypes.npy"
    torch.save(net.state_dict(), state_path)
    np.save(prototypes_path, prototypes)
    result = {
        "status": "complete",
        "development_only": True,
        "release_evidence": False,
        "sealed_release_data_used": 0,
        "consumed_test_rows_used": 0,
        "frontend": td_preprocess.preprocess_metadata(),
        "encoder": args.encoder,
        "architecture": net.config(),
        "parameter_count": int(sum(p.numel() for p in net.parameters())),
        "training": history,
        "training_views": training_view_audit,
        "closed_selection": closed,
        "population_physical_scale": scale,
        "population_causal_length": length,
        "geometry_summary": {
            name: {
                "count": len(rows),
                "bandwidth_median": float(
                    np.median([row["bw"] for row in rows])
                ),
                "center_median": float(
                    np.median([row["center"] for row in rows])
                ),
            }
            for name, rows in contexts.items()
        },
        "artifacts": {
            "state_dict": state_path.name,
            "state_dict_sha256": _sha256(state_path),
            "prototypes": prototypes_path.name,
            "prototypes_sha256": _sha256(prototypes_path),
        },
        "source_sha256": {
            "run_time_domain_dev.py": _sha256(Path(__file__)),
            "time_domain_geometry.py": _sha256(TRAINING / "time_domain_geometry.py"),
            "time_domain_invariant_patch_preprocess.py": _sha256(
                TRAINING / "time_domain_invariant_patch_preprocess.py"
            ),
            "invariant_patch_preprocess.py": _sha256(
                TRAINING / "invariant_patch_preprocess.py"
            ),
        },
        "data_audit": data["data_audit"],
        "source_index_contract": {
            "contract": data["cache_contract"],
            "role": (
                "identifies the raw corpus and exposed train/enroll/selection "
                "indices only; every tensor in this run was rebuilt with the "
                "time-domain v3 frontend"
            ),
        },
        "device": str(device),
        "wall_clock_s": time.perf_counter() - started,
    }
    _write_json(output / "dev_metrics.json", result)
    print(f"[v3 td] wrote {output}", flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--encoder", choices=("real", "complex"), default="real")
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--patch-length", type=int, default=64)
    parser.add_argument("--patch-count", type=int, default=16)
    parser.add_argument("--target-frac", type=float, default=0.5)
    parser.add_argument("--patch-dim", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--set-pool", choices=("mean", "mean_std"), default="mean_std")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--warmup-frac", type=float, default=0.03)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--multilength-train",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "expand eligible train rows into exact 4096/8192/16384 prefix views; "
            "enrollment and all prototypes remain N=16384"
        ),
    )
    parser.add_argument(
        "--view-consistency-weight",
        type=float,
        default=0.0,
        help=(
            "bounded cosine consistency weight for a second, different prefix "
            "of each sampled base row; requires --multilength-train"
        ),
    )
    parser.add_argument(
        "--phase-augmentation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--selection-policy",
        choices=SELECTION_POLICIES,
        default="closed",
        help=(
            "checkpoint selection criterion: 'closed' is the historical "
            "closed-set balanced-accuracy maximum (default, bit-identical); "
            "'closed-with-floor' is the same maximum restricted to "
            "checkpoints whose novelty proxy has not degraded more than "
            "--novelty-floor-fraction from its running best; requires "
            "--multilength-train"
        ),
    )
    parser.add_argument(
        "--novelty-floor-fraction",
        type=float,
        default=NOVELTY_FLOOR_FRACTION,
        help=(
            "maximum tolerated fractional decline of the novelty proxy from "
            "its running best under closed-with-floor; the default is a "
            "pre-declared constant, never fitted to a result"
        ),
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
