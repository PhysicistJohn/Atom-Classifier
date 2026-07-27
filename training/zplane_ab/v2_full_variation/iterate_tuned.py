"""Fast-iterate + confusion-matrix harness using train_tuned (configurable
LR/weight-decay/warmup/label-smoothing), for hyperparameter sweeps on top of
plain cross-entropy (no margin/repulsion term).

Run:
  .venv-training/bin/python training/zplane_ab/v2_full_variation/iterate_tuned.py \
      --tag tuned_run1 --episodes 5000 --backbone vit --condition native \
      --lr 5e-4 --wd 1e-3 --warmup-frac 0.05 --label-smoothing 0.05
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
from vit2d_backbone import ViT2DEmbedding  # noqa: E402
from complex_multiscale_backbone import ComplexMultiScaleEmbedding  # noqa: E402
import train_common_v2 as tcv2  # noqa: E402
from evaluate_ab_v2 import evaluate_v2  # noqa: E402
from train import embed_all, nearest  # noqa: E402
from train_tuned import train_tuned  # noqa: E402

ARTIFACTS = os.path.join(_V2_DIR, "artifacts", "iterate_tuned")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--backbone", choices=["cnn", "vit", "vit2d", "complexlstm"], default="vit")
    ap.add_argument("--cx-width", type=int, default=32)
    ap.add_argument("--cx-real-width", type=int, default=16)
    ap.add_argument("--cx-kernel", type=int, default=5)
    ap.add_argument("--cx-blocks", type=int, default=3)
    ap.add_argument("--cx-lstm-hidden", type=int, default=24)
    ap.add_argument("--cx-lstm-downsample", type=int, default=8)
    ap.add_argument("--cx-rnn-type", choices=["lstm", "gru"], default="lstm")
    ap.add_argument("--cx-unidirectional", action="store_true")
    ap.add_argument("--cx-no-rnn", action="store_true")
    ap.add_argument("--condition", choices=["resampled", "native"], default="native")
    ap.add_argument("--vit-patch", type=int, default=16)
    ap.add_argument("--vit-embed-dim", type=int, default=64)
    ap.add_argument("--vit-depth", type=int, default=4)
    ap.add_argument("--vit-heads", type=int, default=4)
    ap.add_argument("--vit2d-nfft", type=int, default=64)
    ap.add_argument("--vit2d-hop", type=int, default=8)
    ap.add_argument("--vit2d-patch-h", type=int, default=8)
    ap.add_argument("--vit2d-patch-w", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=2e-4)
    ap.add_argument("--log-every", type=int, default=None)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--clip-grad", type=float, default=None)
    args = ap.parse_args()

    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"[tuned-{args.tag}] device={dev} episodes={args.episodes} condition={args.condition}")

    data = tcv2.prepare_data_v2(condition=args.condition)
    classes = data["classes"]
    n_classes = data["n_classes"]

    in_len = data["input_length"]
    if args.backbone == "cnn":
        net = Embedding(EMBED_DIM, N_FEATURES)
    elif args.backbone == "vit":
        patch_size = args.vit_patch * (in_len // 1024)
        net = ViTEmbedding(patch_size=patch_size, embed_dim=args.vit_embed_dim,
                            depth=args.vit_depth, num_heads=args.vit_heads, in_len=in_len)
    elif args.backbone == "vit2d":
        net = ViT2DEmbedding(n_fft=args.vit2d_nfft, hop_length=args.vit2d_hop,
                             patch_h=args.vit2d_patch_h, patch_w=args.vit2d_patch_w,
                             embed_dim=args.vit_embed_dim, depth=args.vit_depth, num_heads=args.vit_heads)
    else:
        net = ComplexMultiScaleEmbedding(complex_width=args.cx_width, real_width=args.cx_real_width,
                                          kernel_size=args.cx_kernel, n_blocks=args.cx_blocks,
                                          lstm_hidden=args.cx_lstm_hidden, lstm_downsample=args.cx_lstm_downsample,
                                          rnn_type=args.cx_rnn_type, rnn_bidirectional=not args.cx_unidirectional,
                                          use_rnn=not args.cx_no_rnn)
    n_params = tcv2.count_params(net)
    print(f"[tuned-{args.tag}] backbone={args.backbone} param_count={n_params}")

    eval_every = args.eval_every if args.eval_every is not None else max(100, args.episodes // 8)
    net, history, wall = train_tuned(net, data, dev, episodes=args.episodes, eval_every=eval_every,
                                      log_tag=f"tuned-{args.tag}", lr=args.lr, weight_decay=args.wd,
                                      warmup_frac=args.warmup_frac, label_smoothing=args.label_smoothing,
                                      clip_grad=args.clip_grad, log_every=args.log_every)
    print(f"[tuned-{args.tag}] episodes_actually_completed={history['episodes']} wall={wall:.1f}s")

    net = net.to("cpu")
    report, protos = evaluate_v2(net, torch.device("cpu"), data)
    report["param_count"] = n_params
    report["episodes"] = args.episodes
    report["wall_clock_train_s"] = wall
    report["backbone"] = args.backbone
    report["condition"] = args.condition
    report["hp"] = {"lr": args.lr, "wd": args.wd, "warmup_frac": args.warmup_frac,
                     "label_smoothing": args.label_smoothing, "clip_grad": args.clip_grad}
    print(f"[tuned-{args.tag}] closed_overall={report['closed_overall']:.3f} "
          f"closed_clean={report['closed_clean']:.3f} closed_impaired={report['closed_impaired']:.3f}")
    print(f"[tuned-{args.tag}] per_class_clean={report['per_class_clean']}")
    print(f"[tuned-{args.tag}] per_class_impaired={report['per_class_impaired']}")

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

    print(f"[tuned-{args.tag}] -- CONFUSION MATRIX (OVERALL, rows=true, cols=pred) --")
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
    print(f"[tuned-{args.tag}] done. -> {os.path.relpath(out_dir)}")


if __name__ == "__main__":
    main()
