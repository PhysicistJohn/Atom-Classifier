"""Data pools and episodic samplers for prototypical / contrastive training.

A pool is a bank of independent impaired+preprocessed realisations per class.
Because every realisation draws its own impairments, any two samples of one class
are a valid positive pair and any cross-class samples are negatives — exactly the
structure the prototypical loss consumes.
"""

from __future__ import annotations

import numpy as np

import rfgen
import preprocess as pp


def build_pool(
    classes: list[str],
    per_class: int,
    rng: np.random.Generator,
    snr_range=(0.0, 30.0),
    scale_jitter: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Generate `per_class` impaired, preprocessed realisations of each class.

    Returns (channels[N,2,L], features[N,F], labels[N], classes).
    """
    xs, fs, ys = [], [], []
    for ci, cls in enumerate(classes):
        for _ in range(per_class):
            iq, _ = rfgen.synth(cls, rng, snr_range=snr_range)
            norm, _ctx = pp.preprocess(iq, scale_jitter=scale_jitter, rng=rng)
            xs.append(pp.to_channels(norm))
            fs.append(pp.iq_features(norm))
            ys.append(ci)
    return (
        np.stack(xs).astype(np.float32),
        np.stack(fs).astype(np.float32),
        np.array(ys, dtype=np.int64),
        classes,
    )


def class_indices(y: np.ndarray, n_classes: int) -> list[np.ndarray]:
    return [np.where(y == c)[0] for c in range(n_classes)]


def sample_episode(
    x: np.ndarray,
    feat: np.ndarray,
    idx_by_class: list[np.ndarray],
    rng: np.random.Generator,
    n_way: int,
    k_shot: int,
    q_query: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return support/query channels+features+labels.

    (sup_x[NK,2,L], sup_f[NK,F], sup_lbl, qry_x[NQ,2,L], qry_f[NQ,F], qry_lbl).
    Labels are episode-local (0..n_way-1). Prototypes come from the support set,
    queries are scored against them — the training-time mirror of few-shot
    enrollment + nearest-prototype inference.
    """
    n_classes = len(idx_by_class)
    chosen = rng.choice(n_classes, size=n_way, replace=False)
    sx, sf, sl, qx, qf, ql = [], [], [], [], [], []
    for local, c in enumerate(chosen):
        pool = idx_by_class[c]
        pick = rng.choice(pool, size=k_shot + q_query, replace=False)
        sx.append(x[pick[:k_shot]]); sf.append(feat[pick[:k_shot]])
        qx.append(x[pick[k_shot:]]); qf.append(feat[pick[k_shot:]])
        sl += [local] * k_shot
        ql += [local] * q_query
    return (
        np.concatenate(sx).astype(np.float32),
        np.concatenate(sf).astype(np.float32),
        np.array(sl, dtype=np.int64),
        np.concatenate(qx).astype(np.float32),
        np.concatenate(qf).astype(np.float32),
        np.array(ql, dtype=np.int64),
    )
