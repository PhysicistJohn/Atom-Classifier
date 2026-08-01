#!/usr/bin/env python
"""Per-row content-realization audit of the production corpus CLEAN waveforms.

SHA-256 of every clean row (full 400000 complex64 samples, exact bytes).
Reports, per profile:
  * distinct realizations overall / in train rows / in eval rows
  * how many eval rows carry a realization ALSO present in the train rows
    ("seen content") vs one that appears only in eval rows ("unseen content")
  * the realization id per row (for downstream slicing)

Output: content_audit.json
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CORPUS = Path("/Users/johnelliott/PersonalGitHub/Atom-Classifier/training/"
              "artifacts/longdwell-production-corpus")
OUT = Path(__file__).with_name("content_audit.json")

t0 = time.perf_counter()
manifest = json.loads((CORPUS / "manifest.json").read_text())
rows = manifest["rows"]
n = len(rows)
clean = np.load(CORPUS / "clean.npy", mmap_mode="r")
print(f"clean {clean.shape} {clean.dtype}", flush=True)
assert clean.shape[0] == n

BATCH = 32
hashes = [None] * n
for i0 in range(0, n, BATCH):
    blk = np.ascontiguousarray(clean[i0:i0 + BATCH])
    for j in range(blk.shape[0]):
        hashes[i0 + j] = hashlib.sha256(blk[j].tobytes()).hexdigest()
    del blk
    if (i0 // BATCH) % 32 == 0:
        el = time.perf_counter() - t0
        print(f"  {i0+BATCH}/{n}  {el:.0f}s", flush=True)
print(f"hashed {n} rows in {time.perf_counter()-t0:.0f}s", flush=True)

by_prof = defaultdict(list)
for r, h in zip(rows, hashes):
    by_prof[r["profile"]].append((r["row"], r["role"], h))

report = {}
row_realization = {}
for prof in sorted(by_prof):
    recs = by_prof[prof]
    allh = [h for _, _, h in recs]
    trh = [h for _, role, h in recs if role == "train"]
    evh = [h for _, role, h in recs if role == "eval"]
    order = [h for h, _ in Counter(allh).most_common()]
    rid = {h: k for k, h in enumerate(order)}
    for row, _role, h in recs:
        row_realization[row] = rid[h]
    train_set = set(trh)
    seen = sum(1 for h in evh if h in train_set)
    report[prof] = {
        "n_rows": len(recs),
        "distinct_all": len(set(allh)),
        "distinct_train": len(train_set),
        "distinct_eval": len(set(evh)),
        "eval_rows_content_seen_in_train": seen,
        "eval_rows_content_unseen": len(evh) - seen,
        "distinct_eval_only": len(set(evh) - train_set),
        "realization_counts_all": [c for _, c in Counter(allh).most_common()],
        "realization_counts_train": [Counter(trh)[h] for h in order],
        "realization_counts_eval": [Counter(evh)[h] for h in order],
    }
    print(f"{prof:<42} distinct all={len(set(allh)):>3} "
          f"train={len(train_set):>3} eval={len(set(evh)):>3} "
          f"evalSeenInTrain={seen}/{len(evh)}", flush=True)

OUT.write_text(json.dumps(
    {"corpus": str(CORPUS), "n_rows": n,
     "per_profile": report,
     "row_realization": row_realization,
     "row_hash": {r["row"]: h for r, h in zip(rows, hashes)},
     "elapsed_s": time.perf_counter() - t0}, indent=1))
print(f"wrote {OUT}", flush=True)
