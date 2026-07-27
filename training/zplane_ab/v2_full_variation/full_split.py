"""v2 split: carves BOTH the clean and impaired populations into
enroll/validation/train thirds (v1's common_split.py only ever carved the
clean population, and dumped 100% of impaired items straight into training,
so the validation set was clean-only and channel impairment was never
actually measured -- see the design-plan finding this file exists to fix).

SEED_V2 is deliberately the SAME LITERAL as common_split.SEED (20260721).
This is intentional, not a copy/paste bug -- read the note below.

    torch.manual_seed(SEED_V2)
    np.random.seed(SEED_V2)
    rng = np.random.default_rng(SEED_V2)
    ...
    clean_pos = rng.permutation(np.where(~impaired)[0])   # same call, same position as v1
    imp_pos   = rng.permutation(np.where(impaired)[0])    # v1 NEVER makes this call

Because `clean_pos`'s permutation is the same RNG call in the same position as
common_split.load_and_split()'s, the clean population's raw permutation
output would coincide with v1's if nothing else diverged. But v2's very next
call, `rng.permutation(imp_pos)`, is a call v1 never makes -- so all RNG state
downstream of it (scale-jitter draws in build_pool_v2, every episode-sampling
draw in train()) diverges from v1 from that point forward. v2 is a
deliberately different, independently-seeded split, not a superset of v1's.

Imports common_split.load_corpus() (unmodified) rather than re-reading
corpus.json, so both v1 and v2 are guaranteed to agree on what "the corpus"
even is.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

_V2_DIR = os.path.dirname(os.path.abspath(__file__))          # -> v2_full_variation/
_ZPAB_DIR = os.path.dirname(_V2_DIR)                            # -> zplane_ab/
_TRAINING_DIR = os.path.dirname(_ZPAB_DIR)                      # -> training/
for p in (_TRAINING_DIR, _ZPAB_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from common_split import load_corpus, CORPUS  # noqa: E402 -- imported, not copied

SEED_V2 = 20260721  # same literal as common_split.SEED, intentionally -- see module docstring


def load_clean_pairs():
    """The clean emission each impaired capture came from, same order as corpus.f32.
    Written by the generator as corpus_clean.f32 (manifest flag hasCleanPairs). This is
    the supervision target for a learned equalizer front end -- see
    equalizer_frontend.py. Returns None if the corpus predates paired output."""
    manifest = json.load(open(os.path.join(CORPUS, "corpus.json")))
    path = os.path.join(CORPUS, "corpus_clean.f32")
    if not manifest.get("hasCleanPairs") or not os.path.exists(path):
        return None
    n, L = manifest["count"], manifest["sampleCount"]
    raw = np.fromfile(path, dtype="<f4").reshape(n, L, 2)
    return (raw[:, :, 0] + 1j * raw[:, :, 1]).astype(np.complex64)


def load_and_split_v2():
    """Returns (iq, y, impaired, classes, tr_idx, en_idx, va_idx, rng), with
    BOTH clean and impaired populations independently permuted and carved
    30/30/40 (enroll/val/train), so tr/en/va each contain a proportional mix
    of clean and impaired items."""
    torch.manual_seed(SEED_V2)
    np.random.seed(SEED_V2)
    rng = np.random.default_rng(SEED_V2)

    iq, y, impaired, classes = load_corpus()

    clean_pos = rng.permutation(np.where(~impaired)[0])   # 459 items
    imp_pos = rng.permutation(np.where(impaired)[0])      # 1352 items -- v1 never permutes this; v2 must

    def carve(pos):
        n_en = int(0.30 * len(pos))
        n_va = int(0.30 * len(pos))
        return pos[:n_en], pos[n_en:n_en + n_va], pos[n_en + n_va:]

    clean_en, clean_va, clean_tr = carve(clean_pos)   # 137 / 137 / 185
    imp_en, imp_va, imp_tr = carve(imp_pos)            # 405 / 405 / 542

    en_idx = np.concatenate([clean_en, imp_en])
    va_idx = np.concatenate([clean_va, imp_va])
    tr_idx = np.concatenate([clean_tr, imp_tr])

    return iq, y, impaired, classes, tr_idx, en_idx, va_idx, rng


def build_pool_v2(iq, y, impaired, idx, rng, preprocess_module, scale_jitter=0.0):
    """Like common_split.build_pool but (a) parameterized over which
    preprocessing module to use (pp.py resampled, or native_preprocess.py
    native -- both expose the same preprocess/to_channels/iq_features
    surface) and (b) also returns a per-item impaired boolean array so
    downstream eval can split clean vs impaired accuracy."""
    xs, fs, ys, imp_flags = [], [], [], []
    for i in idx:
        norm, _ = preprocess_module.preprocess(iq[i], scale_jitter=scale_jitter, rng=rng)
        xs.append(preprocess_module.to_channels(norm))
        fs.append(preprocess_module.iq_features(norm))
        ys.append(y[i])
        imp_flags.append(bool(impaired[i]))
    return (np.stack(xs).astype(np.float32), np.stack(fs).astype(np.float32),
            np.array(ys, dtype=np.int64), np.array(imp_flags, dtype=bool))
