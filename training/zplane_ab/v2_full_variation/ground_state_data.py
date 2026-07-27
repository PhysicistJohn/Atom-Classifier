"""Auditable data preparation for the corrected transfer-U-Net run.

This layer deliberately leaves the historical pool cache untouched.  It:

* excludes captures whose nominal occupied band crosses the FFT seam, because the current
  non-circular estimator misreads those rows as almost full-band;
* recomputes feature standardization after that exclusion; and
* divides the old validation pool into disjoint model-selection and final-test halves,
  stratified by class and impairment status.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.dirname(os.path.dirname(HERE))
CORPUS = os.path.join(TRAINING, "artifacts", "signallab-corpus")
MANIFEST = os.path.join(CORPUS, "corpus.json")


def manifest_fingerprint() -> str:
    h = hashlib.sha256()
    with open(MANIFEST, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def file_fingerprint(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _manifest():
    with open(MANIFEST) as f:
        return json.load(f)


def edge_safe_corpus_mask(manifest=None) -> np.ndarray:
    """Rows whose nominal occupied band does not cross the ±Nyquist seam."""
    manifest = manifest or _manifest()
    safe = []
    for item in manifest["items"]:
        center = abs(float(item.get("centreOffsetFrac", 0.0)))
        frac_bw = float(item["bandwidthHz"]) / float(item["sampleRateHz"])
        safe.append(center + 0.5 * frac_bw < 0.5)
    return np.asarray(safe, dtype=bool)


def _selection_test_rows(y, impaired, seed: int):
    """Balanced deterministic half-split of a validation pool."""
    y = np.asarray(y)
    impaired = np.asarray(impaired, dtype=bool)
    rng = np.random.default_rng(seed)
    select, test = [], []
    for cls in np.unique(y):
        for imp in (False, True):
            rows = np.where((y == cls) & (impaired == imp))[0]
            rows = rng.permutation(rows)
            cut = len(rows) // 2
            select.extend(rows[:cut])
            test.extend(rows[cut:])
    select = np.sort(np.asarray(select, dtype=np.int64))
    test = np.sort(np.asarray(test, dtype=np.int64))
    if len(np.intersect1d(select, test)) or len(select) + len(test) != len(y):
        raise AssertionError("selection/test split must be disjoint and exhaustive")
    return select, test


def prepare_ground_state(data, seed: int = 20260727):
    """Return a filtered copy plus split metadata; never mutates the cached dict."""
    import dataset as ds
    import preprocess as pp

    out = copy.copy(data)
    safe = edge_safe_corpus_mask()
    excluded = {}

    manifest = _manifest()
    raw = np.memmap(
        os.path.join(CORPUS, "corpus.f32"),
        dtype="<f4",
        mode="r",
        shape=(int(manifest["count"]), int(manifest["sampleCount"]), 2),
    )

    for tag in ("tr", "en", "va"):
        corpus_idx = np.asarray(data[f"{tag}_idx"], dtype=np.int64)
        keep = safe[corpus_idx]
        kept_idx = corpus_idx[keep]
        y_before = np.asarray(data[f"y{tag}"])
        excluded[tag] = {
            str(data["classes"][c]): int(np.sum((~keep) & (y_before == c)))
            for c in range(data["n_classes"])
        }

        # Rebuild every retained waveform with the current, non-jittered preprocessing
        # implementation. The historical resampled training cache was created with
        # scale_jitter=0.08, while CropStream(mode="off") uses no jitter; reusing that cache
        # made the alleged no-augmentation control differ by up to ~3.0 in waveform space.
        # Enrolment, selection, test, and crop-off training now share one exact transform.
        xs, fs = [], []
        for corpus_row in kept_idx:
            pair = np.asarray(raw[int(corpus_row)], dtype=np.float64)
            iq = pair[:, 0] + 1j * pair[:, 1]
            norm, _ctx = pp.preprocess(iq, scale_jitter=0.0)
            xs.append(pp.to_channels(norm))
            fs.append(pp.iq_features(norm))
        out[f"x{tag}"] = np.stack(xs).astype(np.float32)
        out[f"f{tag}"] = np.stack(fs).astype(np.float32)
        out[f"y{tag}"] = np.asarray(data[f"y{tag}"])[keep]
        out[f"imp_{tag}"] = np.asarray(data[f"imp_{tag}"])[keep]
        out[f"{tag}_idx"] = kept_idx
    del raw

    # Refit on the actual corrected training population, then apply once everywhere.
    fmean = out["ftr"].mean(axis=0).astype(np.float32)
    fstd = (out["ftr"].std(axis=0) + 1e-6).astype(np.float32)
    for tag in ("tr", "en", "va"):
        out[f"f{tag}"] = ((out[f"f{tag}"] - fmean) / fstd).astype(np.float32)
    out["fmean"], out["fstd"] = fmean, fstd
    out["idx_by_class"] = ds.class_indices(out["ytr"], out["n_classes"])
    out["rng"] = np.random.default_rng(seed)

    select, test = _selection_test_rows(out["yva"], out["imp_va"], seed + 1)
    out["selection_rows"] = select
    out["test_rows"] = test
    out["ground_state"] = {
        "seed": int(seed),
        "manifest_sha256": manifest_fingerprint(),
        "preprocess_sha256": file_fingerprint(pp.__file__),
        "preprocess_policy": "current resampled preprocess; scale_jitter=0.0 in every split",
        "edge_rule": "abs(centreOffsetFrac) + bandwidthHz/sampleRateHz/2 < 0.5",
        "excluded_by_split_class": excluded,
        "counts": {
            "train": int(len(out["ytr"])),
            "enroll": int(len(out["yen"])),
            "selection": int(len(select)),
            "test": int(len(test)),
        },
    }
    return out


def validation_subset(data, rows):
    """Shallow copy whose validation fields contain only ``rows``."""
    out = copy.copy(data)
    rows = np.asarray(rows, dtype=np.int64)
    for prefix in ("x", "f", "y"):
        out[f"{prefix}va"] = np.asarray(data[f"{prefix}va"])[rows]
    out["imp_va"] = np.asarray(data["imp_va"])[rows]
    out["va_idx"] = np.asarray(data["va_idx"])[rows]
    return out


def selection_data(data):
    return validation_subset(data, data["selection_rows"])


def test_data(data):
    return validation_subset(data, data["test_rows"])
