"""Shared evaluator for BOTH trained nets (CNN baseline and z-plane), working
directly on the torch net objects (not a numpy/export path -- there is no
shipped z-plane export and none is needed for this experiment).

Reuses train.py's embed_all/prototypes_from/nearest UNCHANGED (imported, never
modified/copied) -- the distance/classification math is identical for both
backbones; only the embedding function differs, which is exactly the point of
the A/B. Does NOT reuse evaluate.py, which is confirmed stale/incompatible
with the current 7-class signallab corpus (see the classifier-pipeline
survey): it synthesizes its "known" test set via rfgen, which has no
bluetooth/gsm/dsss-as-signallab-defines-it modulator, and evaluates against a
9-class rfgen taxonomy that doesn't match this corpus at all.

Metrics computed, all against the SAME held-out split from common_split:
  - closed_fine (clean closed-set accuracy) + per_class
  - leave-one-class-out few-shot recall at K=1 and K=5, sourcing the "K shots"
    from the IMPAIRED training pool (xtr/ytr) instead of rfgen.synth (this
    corpus has no synthetic generator for signallab's classes) -- structurally
    mirrors evaluate.py's LOO loop (delete a class's enrolled prototype,
    re-enroll it from K fresh embeddings, measure recall of that class's own
    held-out validation embeddings against the re-enrolled + remaining bank).

Run:  .venv-training/bin/python training/zplane_ab/evaluate_ab.py
(invoked as a library by report_ab.py normally; can also run standalone to
regenerate both eval-report-*.json from the saved state_dicts.)
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

_ZPAB_DIR = os.path.dirname(os.path.abspath(__file__))
_TRAINING_DIR = os.path.dirname(_ZPAB_DIR)
for p in (_TRAINING_DIR, _ZPAB_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from model import Embedding, EMBED_DIM, N_FEATURES  # noqa: E402
from zplane_backbone import ZPlaneEmbedding  # noqa: E402
from train import embed_all, prototypes_from, nearest  # noqa: E402
import train_common as tc  # noqa: E402

ARTIFACTS = os.path.join(_ZPAB_DIR, "artifacts")
LOO_SEED = 424242  # independent stream for few-shot draw ordering, does not touch training rng
K_LIST = [1, 5]
N_LOO_TRIALS = 20  # average recall over this many independent K-shot draws (impaired pool is small per class)


def evaluate(net, dev, data, loo_seed: int = LOO_SEED, n_trials: int = N_LOO_TRIALS):
    classes = data["classes"]
    n_classes = data["n_classes"]
    xen, fen, yen = data["xen"], data["fen"], data["yen"]
    xva, fva, yva = data["xva"], data["fva"], data["yva"]
    xtr, ftr, ytr = data["xtr"], data["ftr"], data["ytr"]

    emb_en = embed_all(net, xen, fen, dev)
    protos = prototypes_from(emb_en, yen, n_classes)
    emb_va = embed_all(net, xva, fva, dev)
    pred, dists = nearest(emb_va, protos)

    report = {}
    report["closed_fine"] = float((pred == yva).mean())
    report["per_class"] = {classes[c]: round(float((pred[yva == c] == c).mean()), 3) for c in range(n_classes)}

    # embed the impaired training pool once (shot source for LOO few-shot)
    emb_tr = embed_all(net, xtr, ftr, dev)
    tr_idx_by_class = [np.where(ytr == c)[0] for c in range(n_classes)]

    rng = np.random.default_rng(loo_seed)
    fewshot = {}
    loo_by_k = {k: [] for k in K_LIST}
    for c in range(n_classes):
        base = np.delete(protos, c, axis=0)
        held_test = emb_va[yva == c]
        pool_idx = tr_idx_by_class[c]
        for K in K_LIST:
            recalls = []
            for _ in range(n_trials):
                if len(pool_idx) >= K:
                    pick = rng.choice(pool_idx, size=K, replace=False)
                else:
                    pick = rng.choice(pool_idx, size=K, replace=True)
                shot_proto = emb_tr[pick].mean(0)
                ext = np.vstack([base, shot_proto[None, :]])
                qp, _ = nearest(held_test, ext)
                recalls.append(float((qp == len(base)).mean()))
            recall = float(np.mean(recalls))
            fewshot[f"loo_{classes[c]}_k{K}"] = round(recall, 3)
            loo_by_k[K].append(recall)
    for K in K_LIST:
        fewshot[f"loo_k{K}_mean"] = round(float(np.mean(loo_by_k[K])), 3)
    report["fewshot"] = fewshot

    return report, protos


def load_cnn_net(manifest, state_dict_path):
    net = Embedding(EMBED_DIM, N_FEATURES)
    net.load_state_dict(torch.load(state_dict_path, map_location="cpu"))
    net.eval()
    return net


def load_zplane_net(manifest, state_dict_path):
    cfg = manifest["config"]
    net = ZPlaneEmbedding(width=cfg["width"], layers=cfg["layers"], sections=cfg["sections"],
                           warm_radius=cfg["warm_radius"])
    net.load_state_dict(torch.load(state_dict_path, map_location="cpu"))
    net.eval()
    return net


def main():
    dev = torch.device("cpu")
    data = tc.prepare_data()

    results = {}
    for name, out_dir, loader in [
        ("cnn", os.path.join(ARTIFACTS, "cnn_baseline"), load_cnn_net),
        ("zplane", os.path.join(ARTIFACTS, "zplane"), load_zplane_net),
    ]:
        manifest_path = os.path.join(out_dir, "manifest.json")
        state_path = os.path.join(out_dir, "state_dict.pt")
        if not (os.path.exists(manifest_path) and os.path.exists(state_path)):
            print(f"[{name}] skipped -- no trained artifacts at {out_dir}")
            continue
        manifest = json.load(open(manifest_path))
        net = loader(manifest, state_path)
        report, protos = evaluate(net, dev, data)
        report["param_count"] = manifest["param_count"]
        report["wall_clock_train_s"] = manifest["wall_clock_train_s"]
        report["episodes"] = manifest["episodes_requested"]
        report["infer_latency_ms_per_1k"] = manifest["infer_latency_ms_per_1k"]
        report["classes"] = data["classes"]
        results[name] = report
        out_path = os.path.join(ARTIFACTS, f"eval-report-{name}.json")
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[{name}] closed_fine={report['closed_fine']:.3f}  loo_k1_mean={report['fewshot'].get('loo_k1_mean')}  loo_k5_mean={report['fewshot'].get('loo_k5_mean')}")
        print(f"  wrote {out_path}")

    return results


if __name__ == "__main__":
    main()
