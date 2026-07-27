"""Preprocess the corpus ONCE to disk; children memory-map it instead of rebuilding.

THE PROBLEM THIS ACTUALLY FIXES. run_priority isolates each trial in its own subprocess so
a hang cannot cost the whole night. But isolation means no shared state, so every child
re-ran prepare_data_v2: ~80 s of preprocessing and a ~9 GB peak, eight times over. That
repeated allocation churn is the most likely reason the machine went into heavy swap
(25.5 of 26.6 GB) before the weekend stall -- i.e. the containment fix was aggravating the
suspected cause. Timeouts survive the symptom; this removes it.

With the pools on disk as .npy, np.load(mmap_mode="r") gives each child a read-only view
backed by the page cache. The OS shares those pages across every process, so N children
cost roughly one copy rather than N, and startup is a few seconds instead of eighty.

Episode sampling indexes the mapped arrays and torch.from_numpy copies each small batch,
so training touches only the pages it needs. Nothing downstream changes.
"""
from __future__ import annotations

import json
import hashlib
import os
import sys

import numpy as np

_V2 = os.path.dirname(os.path.abspath(__file__))
if _V2 not in sys.path:
    sys.path.insert(0, _V2)
_TRAINING = os.path.dirname(os.path.dirname(_V2))
for p in (_TRAINING, os.path.dirname(_V2)):
    if p not in sys.path:
        sys.path.insert(0, p)

CACHE = os.path.join(_TRAINING, "artifacts", "signallab-corpus", "_pools")
CORPUS = os.path.dirname(CACHE)
CACHE_SCHEMA = 3
ARRAYS = ["xtr", "ftr", "ytr", "xen", "fen", "yen", "xva", "fva", "yva",
          "imp_tr", "imp_en", "imp_va", "fmean", "fstd", "tr_idx", "en_idx", "va_idx"]


class CacheUnavailableError(RuntimeError):
    """The requested preprocessing contract has not been materialized yet."""


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _corpus_identity() -> dict:
    manifest = os.path.join(CORPUS, "corpus.json")
    files = {}
    for name in ("corpus.f32", "corpus_clean.f32"):
        path = os.path.join(CORPUS, name)
        if os.path.exists(path):
            stat = os.stat(path)
            files[name] = {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}
    return {"manifest_sha256": _sha256(manifest), "binary_stats": files}


def _cache_contract(condition: str) -> dict:
    """Content/implementation identity for arrays produced by one transform."""
    import preprocess as pp
    import train_common_v2 as tcv2

    if condition not in tcv2.CONDITIONS:
        raise ValueError(f"unknown preprocessing condition {condition!r}")
    module = tcv2.CONDITIONS[condition]
    sources = {"training/preprocess.py": _sha256(pp.__file__)}
    if os.path.abspath(module.__file__) != os.path.abspath(pp.__file__):
        sources[os.path.basename(module.__file__)] = _sha256(module.__file__)
    payload = {
        "schema": CACHE_SCHEMA,
        "condition": condition,
        "preprocess_version": getattr(pp, "PREPROCESS_VERSION", "unversioned"),
        "source_sha256": sources,
        "corpus": _corpus_identity(),
        "pool_policy": {"train_scale_jitter": 0.08, "eval_scale_jitter": 0.0},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["id"] = hashlib.sha256(encoded).hexdigest()[:16]
    return payload


def _prefix(condition: str, contract: dict) -> str:
    return f"{condition}__{contract['id']}"


def _meta_path(condition: str, contract: dict) -> str:
    return os.path.join(CACHE, f"{_prefix(condition, contract)}_meta.json")


def _array_path(condition: str, key: str, contract: dict) -> str:
    return os.path.join(CACHE, f"{_prefix(condition, contract)}_{key}.npy")


def build(condition: str = "native"):
    """Build the on-disk pool cache for a condition. Idempotent."""
    import train_common_v2 as tcv2
    contract = _cache_contract(condition)
    os.makedirs(CACHE, exist_ok=True)
    meta_path = _meta_path(condition, contract)
    if os.path.exists(meta_path):
        print(f"[pools] {condition} contract {contract['id']} already cached")
        return
    print(
        f"[pools] building {condition} contract {contract['id']} (one time, ~80s)",
        flush=True,
    )
    d = tcv2.prepare_data_v2(condition=condition)
    for k in ARRAYS:
        np.save(_array_path(condition, k, contract), np.asarray(d[k]))
    meta = {"classes": d["classes"], "n_classes": d["n_classes"],
            "input_length": int(d["input_length"]), "condition": condition,
            "cache_contract": contract}
    # Metadata is the completion marker and is deliberately written last.
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    total = sum(os.path.getsize(_array_path(condition, k, contract)) for k in ARRAYS)
    print(
        f"[pools] cached {condition} contract {contract['id']}: "
        f"{total/1e9:.2f} GB on disk",
        flush=True,
    )


def load(
    condition: str = "native",
    seed: int = 20260721,
    *,
    build_if_missing: bool = True,
):
    """Return the same dict prepare_data_v2 returns, but with the big arrays memory-mapped.

    idx_by_class and rng are rebuilt here (cheap) rather than cached, so episode sampling
    behaves identically to the non-cached path."""
    import dataset as ds
    contract = _cache_contract(condition)
    meta_path = _meta_path(condition, contract)
    if not os.path.exists(meta_path):
        if not build_if_missing:
            raise CacheUnavailableError(
                f"{condition} pool contract {contract['id']} is not built; "
                f"run pool_cache.py {condition}"
            )
        build(condition)
    with open(meta_path) as f:
        meta = json.load(f)
    if meta.get("cache_contract") != contract:
        raise RuntimeError(
            f"pool cache contract mismatch for {condition}: "
            f"expected {contract['id']}; refusing stale arrays"
        )
    d = dict(meta)
    for k in ARRAYS:
        path = _array_path(condition, k, contract)
        # big per-sample arrays are mapped; small vectors load normally
        big = k in ("xtr", "ftr", "xen", "fen", "xva", "fva")
        d[k] = np.load(path, mmap_mode="r") if big else np.load(path)
    d["idx_by_class"] = ds.class_indices(d["ytr"], d["n_classes"])
    d["rng"] = np.random.default_rng(seed)
    return d


if __name__ == "__main__":
    cond = sys.argv[1] if len(sys.argv) > 1 else "native"
    build(cond)
    d = load(cond)
    print(f"[pools] loaded: xtr {d['xtr'].shape} mmap={isinstance(d['xtr'], np.memmap)} "
          f"input_length={d['input_length']} classes={d['classes']}")
