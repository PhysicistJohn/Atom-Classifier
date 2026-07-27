"""Single source of truth for the corpus load + rng-ordered split, shared by
BOTH the fresh CNN-baseline retrain and the z-plane run, so the comparison is
against the identical held-out data.

Copied verbatim (not reimplemented) from training/train_signallab.py's
load_corpus() and the split block in main() (lines ~45-92 at the time this was
written) — see that file for the authoritative original. Do NOT reorder the
RNG calls here relative to train_signallab.py's own order
(torch.manual_seed -> np.random.seed -> np.random.default_rng -> load_corpus
[no RNG use] -> rng.permutation) — reordering silently produces a different
split, which would break the apples-to-apples comparison this experiment
depends on.

training/train_signallab.py, training/model.py, training/preprocess.py,
training/dataset.py, training/train.py, training/evaluate.py and the corpus
files under training/artifacts/signallab-corpus/ are all read-only references
here: imported/read, never modified.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

_TRAINING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # -> training/
if _TRAINING_DIR not in sys.path:
    sys.path.insert(0, _TRAINING_DIR)

import preprocess as pp  # noqa: E402

SEED = 20260721
CORPUS = os.path.join(_TRAINING_DIR, "artifacts", "signallab-corpus")


def load_corpus():
    """Verbatim copy of train_signallab.py's load_corpus()."""
    manifest = json.load(open(os.path.join(CORPUS, "corpus.json")))
    n = manifest["count"]
    L = manifest["sampleCount"]
    raw = np.fromfile(os.path.join(CORPUS, "corpus.f32"), dtype="<f4").reshape(n, L, 2)
    iq = (raw[:, :, 0] + 1j * raw[:, :, 1]).astype(np.complex64)  # complex64: half the RAM of complex128, ample precision for cf32 source
    classes = sorted(manifest["classes"])
    cindex = {c: i for i, c in enumerate(classes)}
    y = np.array([cindex[it["cls"]] for it in manifest["items"]], dtype=np.int64)
    impaired = np.array([bool(it.get("impaired", False)) for it in manifest["items"]])
    return iq, y, impaired, classes


def load_and_split():
    """Deterministic corpus load + split, RNG order matching train_signallab.py
    exactly: torch.manual_seed, np.random.seed, np.random.default_rng, THEN
    load_corpus() (no RNG use inside), THEN the permutation. Returns
    (iq, y, impaired, classes, tr_idx, en_idx, va_idx)."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    iq, y, impaired, classes = load_corpus()

    clean_pos = rng.permutation(np.where(~impaired)[0])
    imp_pos = np.where(impaired)[0]
    n_en = int(0.30 * len(clean_pos))
    n_va = int(0.30 * len(clean_pos))
    en_idx = clean_pos[:n_en]
    va_idx = clean_pos[n_en:n_en + n_va]
    tr_idx = np.concatenate([clean_pos[n_en + n_va:], imp_pos])

    return iq, y, impaired, classes, tr_idx, en_idx, va_idx, rng


def build_pool(iq, y, idx, rng, scale_jitter=0.0):
    """Verbatim copy of train_signallab.py's build_pool()."""
    xs, fs, ys = [], [], []
    for i in idx:
        norm, _ = pp.preprocess(iq[i], scale_jitter=scale_jitter, rng=rng)
        xs.append(pp.to_channels(norm))
        fs.append(pp.iq_features(norm))
        ys.append(y[i])
    return (np.stack(xs).astype(np.float32), np.stack(fs).astype(np.float32),
            np.array(ys, dtype=np.int64))
