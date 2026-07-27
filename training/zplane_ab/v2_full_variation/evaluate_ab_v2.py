"""v2 evaluator: for each of the 4 trained (backbone, condition) cells,
computes closed-set accuracy split by clean/impaired/overall (the actual fix
for the v1 eval-methodology bug -- v1's validation pool was clean-only), plus
per-class-clean/per-class-impaired accuracy, plus few-shot leave-one-class-out
recall at K=1,5, ALSO split by clean/impaired/overall query subset.

Reuses train.py's embed_all/prototypes_from/nearest UNCHANGED (imported, never
modified) -- identical distance/classification math to v1's evaluate_ab.py;
only the data (split + preprocessing condition) differs.

Run:  .venv-training/bin/python training/zplane_ab/v2_full_variation/evaluate_ab_v2.py
(invoked as a library by report_ab_v2.py normally; can also run standalone to
regenerate all 4 eval-report-*.json files from the saved state_dicts.)
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

_V2_DIR = os.path.dirname(os.path.abspath(__file__))
_ZPAB_DIR = os.path.dirname(_V2_DIR)
_TRAINING_DIR = os.path.dirname(_ZPAB_DIR)
for p in (_TRAINING_DIR, _ZPAB_DIR, _V2_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from model import Embedding, EMBED_DIM, N_FEATURES  # noqa: E402
from zplane_backbone import ZPlaneEmbedding  # noqa: E402
from train import embed_all, prototypes_from, nearest  # noqa: E402
import train_common_v2 as tcv2  # noqa: E402

ARTIFACTS = os.path.join(_V2_DIR, "artifacts")
LOO_SEED = 424242  # independent stream for few-shot draw ordering, does not touch training rng
K_LIST = [1, 5]
N_LOO_TRIALS = 20

CELLS = [
    ("cnn", "resampled"),
    ("cnn", "native"),
    ("zplane", "resampled"),
    ("zplane", "native"),
]


def evaluate_v2(net, dev, data, loo_seed: int = LOO_SEED, n_trials: int = N_LOO_TRIALS,
                include_fewshot: bool = True):
    classes = data["classes"]
    n_classes = data["n_classes"]
    xen, fen, yen = data["xen"], data["fen"], data["yen"]
    xva, fva, yva, imp_va = data["xva"], data["fva"], data["yva"], data["imp_va"]
    xtr, ftr, ytr = data["xtr"], data["ftr"], data["ytr"]

    emb_en = embed_all(net, xen, fen, dev)
    protos = prototypes_from(emb_en, yen, n_classes)
    emb_va = embed_all(net, xva, fva, dev)
    pred, dists = nearest(emb_va, protos)

    clean_mask = ~imp_va
    imp_mask = imp_va

    report = {
        "closed_overall": float((pred == yva).mean()),
        "closed_clean": float((pred[clean_mask] == yva[clean_mask]).mean()) if clean_mask.sum() else None,
        "closed_impaired": float((pred[imp_mask] == yva[imp_mask]).mean()) if imp_mask.sum() else None,
        "n_va_clean": int(clean_mask.sum()),
        "n_va_impaired": int(imp_mask.sum()),
        "per_class_clean": {
            classes[c]: round(float((pred[clean_mask & (yva == c)] == c).mean()), 3)
            for c in range(n_classes) if (clean_mask & (yva == c)).sum() > 0
        },
        "per_class_impaired": {
            classes[c]: round(float((pred[imp_mask & (yva == c)] == c).mean()), 3)
            for c in range(n_classes) if (imp_mask & (yva == c)).sum() > 0
        },
    }
    if not include_fewshot:
        report["fewshot"] = {
            "status": "omitted",
            "reason": "historical LOO re-enrolment is not an encoder-held-out transfer test",
        }
        return report, protos

    # Historical few-shot re-enrolment diagnostic -- same mechanism as v1's evaluate_ab.py
    # (draw K shots from
    # the training pool per class, re-enroll, measure recall) -- but now ALSO
    # split the held-out query set by clean/impaired so recall-under-
    # impairment is visible, not just an aggregate.
    # IMPORTANT: the encoder was trained on every one of these classes. This is therefore
    # held-class prototype replacement, not evidence of transfer to an unseen class.
    emb_tr = embed_all(net, xtr, ftr, dev)
    tr_idx_by_class = [np.where(ytr == c)[0] for c in range(n_classes)]
    rng = np.random.default_rng(loo_seed)
    fewshot = {}
    loo_agg = {K: {"clean": [], "impaired": [], "overall": []} for K in K_LIST}
    overall_mask = np.ones_like(clean_mask)
    for c in range(n_classes):
        base = np.delete(protos, c, axis=0)
        pool_idx = tr_idx_by_class[c]
        for K in K_LIST:
            for label, mask in [("clean", clean_mask), ("impaired", imp_mask), ("overall", overall_mask)]:
                held_test = emb_va[(yva == c) & mask]
                if len(held_test) == 0:
                    continue
                recalls = []
                for _ in range(n_trials):
                    pick = rng.choice(pool_idx, size=K, replace=(len(pool_idx) < K))
                    shot_proto = emb_tr[pick].mean(0)
                    ext = np.vstack([base, shot_proto[None, :]])
                    qp, _ = nearest(held_test, ext)
                    recalls.append(float((qp == len(base)).mean()))
                recall = float(np.mean(recalls))
                fewshot[f"loo_{classes[c]}_k{K}_{label}"] = round(recall, 3)
                loo_agg[K][label].append(recall)
    for K in K_LIST:
        for label in ("clean", "impaired", "overall"):
            vals = loo_agg[K][label]
            if vals:
                fewshot[f"loo_k{K}_{label}_mean"] = round(float(np.mean(vals)), 3)
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


LOADERS = {"cnn": load_cnn_net, "zplane": load_zplane_net}


def main():
    dev = torch.device("cpu")
    data_by_condition = {}
    for condition in ("resampled", "native"):
        data_by_condition[condition] = tcv2.prepare_data_v2(condition=condition)

    results = {}
    for backbone, condition in CELLS:
        name = f"{backbone}_{condition}"
        out_dir = os.path.join(ARTIFACTS, name)
        manifest_path = os.path.join(out_dir, "manifest.json")
        state_path = os.path.join(out_dir, "state_dict.pt")
        if not (os.path.exists(manifest_path) and os.path.exists(state_path)):
            print(f"[{name}] skipped -- no trained artifacts at {out_dir}")
            continue
        manifest = json.load(open(manifest_path))
        net = LOADERS[backbone](manifest, state_path)
        data = data_by_condition[condition]
        report, protos = evaluate_v2(net, dev, data)
        report["backbone"] = backbone
        report["condition"] = condition
        report["input_length"] = manifest["input_length"]
        report["param_count"] = manifest["param_count"]
        report["wall_clock_train_s"] = manifest["wall_clock_train_s"]
        report["episodes"] = manifest["episodes_requested"]
        report["episodes_actually_completed"] = manifest.get("episodes_actually_completed")
        report["infer_latency_ms_per_1k_cpu"] = manifest["infer_latency_ms_per_1k_cpu"]
        report["training_device"] = manifest.get("training_device")
        report["classes"] = data["classes"]
        np.save(os.path.join(out_dir, "prototypes.npy"), protos)
        results[name] = report
        out_path = os.path.join(ARTIFACTS, f"eval-report-{name}.json")
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[{name}] closed_overall={report['closed_overall']:.3f}  "
              f"closed_clean={report['closed_clean']:.3f}  closed_impaired={report['closed_impaired']:.3f}  "
              f"loo_k1_overall={report['fewshot'].get('loo_k1_overall_mean')}  "
              f"loo_k5_overall={report['fewshot'].get('loo_k5_overall_mean')}")
        print(f"  wrote {out_path}")

    return results


if __name__ == "__main__":
    main()
