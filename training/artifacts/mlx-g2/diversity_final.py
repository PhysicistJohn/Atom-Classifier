#!/usr/bin/env python
"""Final content-diversity table: phase-invariant clustering with the
all-zero (silent) rows pulled out into their own degenerate cluster."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
CORPUS = Path("/Users/johnelliott/PersonalGitHub/Atom-Classifier/training/"
              "artifacts/longdwell-production-corpus")
THRESH = 0.999


def clusters(S, thresh):
    n = S.shape[0]
    lab = -np.ones(n, dtype=np.int64)
    k = 0
    for i in range(n):
        if lab[i] >= 0:
            continue
        stack, lab[i] = [i], k
        while stack:
            j = stack.pop()
            nb = np.nonzero((S[j] >= thresh) & (lab < 0))[0]
            lab[nb] = k
            stack.extend(nb.tolist())
        k += 1
    return lab


manifest = json.loads((CORPUS / "manifest.json").read_text())
rows = manifest["rows"]
clean = np.load(CORPUS / "clean.npy", mmap_mode="r")
by = defaultdict(list)
for r in rows:
    by[r["profile"]].append(r)

out, row_cluster = {}, {}
hdr = (f"{'profile':<42}{'cls':<10}{'C_all':>6}{'C_tr':>6}{'C_ev':>6}"
       f"{'evOnly':>7}{'zeroTr':>7}{'zeroEv':>7}{'meanRho':>9}{'evMaxRho':>9}")
print(hdr)
print("-" * len(hdr))
for prof in sorted(by):
    recs = sorted(by[prof], key=lambda r: r["row"])
    ids = [r["row"] for r in recs]
    X = np.ascontiguousarray(clean[ids]).astype(np.complex64)
    nrm = np.linalg.norm(X, axis=1)
    zero = nrm == 0
    Xn = X / np.where(nrm == 0, 1.0, nrm)[:, None]
    G = np.abs(Xn @ Xn.conj().T).astype(np.float64)
    np.fill_diagonal(G, 1.0)
    del X, Xn
    nz = np.nonzero(~zero)[0]
    lab = np.full(len(ids), -1, dtype=np.int64)
    if len(nz):
        lab[nz] = clusters(G[np.ix_(nz, nz)], THRESH)
    n_nz_clus = int(lab[nz].max() + 1) if len(nz) else 0
    lab[zero] = n_nz_clus                       # silence == one cluster
    for r, L in zip(recs, lab):
        row_cluster[r["row"]] = int(L)
    tr = np.array([r["role"] == "train" for r in recs])
    ev = ~tr
    ctr, cev = set(lab[tr].tolist()), set(lab[ev].tolist())
    nzoff = G[np.ix_(nz, nz)][np.triu_indices(len(nz), k=1)] if len(nz) > 1 \
        else np.array([1.0])
    Gce = G[np.ix_(ev, tr)]
    rec = {
        "cls": recs[0]["cls"],
        "clusters_all": int(len(set(lab.tolist()))),
        "clusters_nonzero": n_nz_clus,
        "clusters_train": len(ctr),
        "clusters_eval": len(cev),
        "clusters_eval_only": len(cev - ctr),
        "eval_rows_in_train_cluster": int(sum(1 for L in lab[ev] if L in ctr)),
        "zero_rows_train": int(zero[tr].sum()),
        "zero_rows_eval": int(zero[ev].sum()),
        "mean_offdiag_absrho_nonzero": float(nzoff.mean()),
        "eval_max_rho_to_any_train_mean": float(Gce.max(axis=1).mean()),
        "cluster_sizes": np.bincount(lab).tolist(),
    }
    out[prof] = rec
    print(f"{prof:<42}{rec['cls']:<10}{rec['clusters_all']:>6}"
          f"{rec['clusters_train']:>6}{rec['clusters_eval']:>6}"
          f"{rec['clusters_eval_only']:>7}{rec['zero_rows_train']:>7}"
          f"{rec['zero_rows_eval']:>7}"
          f"{rec['mean_offdiag_absrho_nonzero']:>9.4f}"
          f"{rec['eval_max_rho_to_any_train_mean']:>9.4f}", flush=True)
    del G

(HERE / "diversity_final.json").write_text(json.dumps(
    {"thresh": THRESH, "per_profile": out, "row_cluster": row_cluster},
    indent=1))
print("\nwrote diversity_final.json")
