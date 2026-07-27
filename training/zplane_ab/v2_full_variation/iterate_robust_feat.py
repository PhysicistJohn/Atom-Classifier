"""Same fast-iterate harness as iterate_cnn.py, but with ROBUST (median/MAD)
feature normalization instead of train_common_v2's mean/std z-scoring.

WHY: direct measurement (see conversation) found gsm's iq_features have severe
outliers -- a handful of samples with feature-vector norms in the hundreds of
thousands (median ~30), traced to 6th-order-cumulant blowup on near-silent
burst-edge captures. Since normalization uses the GLOBAL mean/std across all
classes, those few outliers inflate fstd for every dimension, compressing the
usable signal for every class's features toward noise -- a plausible reason
three different loss formulations (plain CE, mean-repulsion weak/strong,
sample-repulsion) all converged to nearly identical bluetooth/gsm/dsss
confusion counts: the discriminative feature signal may have been getting
crushed by normalization before the network ever saw it.

Fix: median/MAD normalization (robust to outliers by construction) instead of
mean/std, computed on the SAME train split (never touching enroll/val), then
clip to +-8 MAD as a hard backstop against any remaining extreme values.

Run:
  .venv-training/bin/python training/zplane_ab/v2_full_variation/iterate_robust_feat.py \
      --tag robust_run1 --episodes 3000 --backbone vit
"""
from __future__ import annotations

import argparse
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
from vit_backbone import ViTEmbedding  # noqa: E402
import train_common_v2 as tcv2  # noqa: E402
from evaluate_ab_v2 import evaluate_v2  # noqa: E402
from train import embed_all, nearest  # noqa: E402
from train_common import train as train_plain  # noqa: E402

ARTIFACTS = os.path.join(_V2_DIR, "artifacts", "iterate_robust_feat")
MAD_CLIP = 8.0


def robust_renormalize(data: dict) -> dict:
    """Undo train_common_v2's mean/std z-scoring (already applied to
    ftr/fen/fva using the train split's mean/std) and replace it with
    median/MAD normalization computed on the SAME raw feature values.
    prepare_data_v2 only returns the ALREADY-z-scored features plus fmean/
    fstd, so reconstruct the raw features first: raw = z*fstd + fmean."""
    fmean, fstd = data["fmean"], data["fstd"]
    raw_tr = data["ftr"] * fstd + fmean
    raw_en = data["fen"] * fstd + fmean
    raw_va = data["fva"] * fstd + fmean

    med = np.median(raw_tr, axis=0)
    mad = np.median(np.abs(raw_tr - med), axis=0) * 1.4826 + 1e-6  # 1.4826 -> consistent with std for normal data

    def norm(x):
        z = (x - med) / mad
        return np.clip(z, -MAD_CLIP, MAD_CLIP).astype(np.float32)

    data["ftr"] = norm(raw_tr)
    data["fen"] = norm(raw_en)
    data["fva"] = norm(raw_va)
    data["_robust_med"] = med
    data["_robust_mad"] = mad
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--backbone", choices=["cnn", "vit"], default="vit")
    ap.add_argument("--vit-patch", type=int, default=16)
    ap.add_argument("--vit-embed-dim", type=int, default=64)
    ap.add_argument("--vit-depth", type=int, default=4)
    ap.add_argument("--vit-heads", type=int, default=4)
    args = ap.parse_args()

    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"[robust-{args.tag}] device={dev} episodes={args.episodes}")

    data = tcv2.prepare_data_v2(condition="resampled")
    classes = data["classes"]
    n_classes = data["n_classes"]

    before_outlier = float(np.abs(data["ftr"]).max())
    data = robust_renormalize(data)
    after_outlier = float(np.abs(data["ftr"]).max())
    print(f"[robust-{args.tag}] max |z-scored feature| before={before_outlier:.1f} after robust norm={after_outlier:.2f}")

    if args.backbone == "cnn":
        net = Embedding(EMBED_DIM, N_FEATURES)
    else:
        net = ViTEmbedding(patch_size=args.vit_patch, embed_dim=args.vit_embed_dim,
                            depth=args.vit_depth, num_heads=args.vit_heads)
    n_params = tcv2.count_params(net)
    print(f"[robust-{args.tag}] backbone={args.backbone} param_count={n_params}")

    eval_every = args.eval_every if args.eval_every is not None else max(100, args.episodes // 6)
    net, history, wall = train_plain(net, data, dev, episodes=args.episodes, eval_every=eval_every,
                                      clip_grad=None, log_tag=f"robust-{args.tag}")
    print(f"[robust-{args.tag}] episodes_actually_completed={history['episodes']} wall={wall:.1f}s")

    net = net.to("cpu")
    report, protos = evaluate_v2(net, torch.device("cpu"), data)
    report["param_count"] = n_params
    report["episodes"] = args.episodes
    report["wall_clock_train_s"] = wall
    report["backbone"] = args.backbone
    report["feature_norm"] = "robust_median_mad"
    print(f"[robust-{args.tag}] closed_overall={report['closed_overall']:.3f} "
          f"closed_clean={report['closed_clean']:.3f} closed_impaired={report['closed_impaired']:.3f}")
    print(f"[robust-{args.tag}] per_class_clean={report['per_class_clean']}")
    print(f"[robust-{args.tag}] per_class_impaired={report['per_class_impaired']}")

    emb_va = embed_all(net, data["xva"], data["fva"], torch.device("cpu"))
    pred, _ = nearest(emb_va, protos)
    true = data["yva"]
    imp = data["imp_va"]
    n = n_classes

    def cm(mask):
        m = np.zeros((n, n), dtype=int)
        for t, p in zip(true[mask], pred[mask]):
            m[t, p] += 1
        return m

    print(f"[robust-{args.tag}] -- CONFUSION MATRIX (OVERALL, rows=true, cols=pred) --")
    print("        " + " ".join(f"{c[:4]:>5}" for c in classes))
    m_overall = cm(np.ones_like(imp))
    for i, c in enumerate(classes):
        row = " ".join(f"{m_overall[i, j]:5d}" for j in range(n))
        print(f"{c[:7]:>7} {row}   (n={m_overall[i].sum()})")

    out_dir = os.path.join(ARTIFACTS, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(out_dir, "state_dict.pt"))
    np.save(os.path.join(out_dir, "prototypes.npy"), protos)
    report["confusion_overall"] = m_overall.tolist()
    report["confusion_clean"] = cm(~imp).tolist()
    report["confusion_impaired"] = cm(imp).tolist()
    report["classes"] = classes
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[robust-{args.tag}] done. -> {os.path.relpath(out_dir)}")


if __name__ == "__main__":
    main()
