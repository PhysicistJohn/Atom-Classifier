"""Confusion matrices per dwell from a trained v7 checkpoint.

Loads a checkpoint saved by v7_trainer.py, rebuilds class prototypes from the
train split, classifies every eval row at each dwell, and renders one
row-normalized confusion matrix per dwell (counts annotated), plus a JSON
with the raw matrices. Also reports the top confusion pairs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
from v7_trainer import (CLASSES, DWELLS, V7Net, rms_normalize,  # noqa: E402
                        spectrogram)

CORPUS = Path(os.environ.get(
    "CORPUS_DIR",
    HERE.parent.parent.parent / "artifacts/longdwell-production-corpus",
))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-prefix", default=str(HERE / "confusion"))
    ap.add_argument("--proto-rows", type=int, default=64)
    cli = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(cli.checkpoint, map_location=device, weights_only=False)
    cfg = ck.get("config", {})
    net = V7Net(width=cfg.get("width", 48),
                embed_dim=cfg.get("embed_dim", 128)).to(device)
    net.load_state_dict(ck["state_dict"])
    net.eval()

    manifest = json.loads((CORPUS / "manifest.json").read_text())
    noisy = np.load(CORPUS / "noisy.npy", mmap_mode="r")
    train_by_class = {c: [] for c in CLASSES}
    eval_rows = []
    for row in manifest["rows"]:
        if row["role"] == "train":
            train_by_class[row["cls"]].append(row["row"])
        else:
            eval_rows.append(row)

    result = {"checkpoint": cli.checkpoint, "dwells": {}}
    fig, axes = plt.subplots(1, len(DWELLS), figsize=(4.6 * len(DWELLS), 4.4))
    plt.rcParams.update({"font.family": "serif"})

    with torch.no_grad():
        for ax, (wname, length) in zip(axes, DWELLS.items()):
            protos_acc = torch.zeros(len(CLASSES), net.embed.out_features,
                                     device=device)
            protos_n = torch.zeros(len(CLASSES), device=device)
            for ci, cls in enumerate(CLASSES):
                ids = train_by_class[cls][: cli.proto_rows]
                for start in range(0, len(ids), 48):
                    chunk = ids[start : start + 48]
                    xb = np.stack([noisy[r, :length] for r in chunk])
                    zb, _, _ = net.encode(spectrogram(rms_normalize(
                        torch.from_numpy(xb).to(device))))
                    protos_acc[ci] += zb.sum(dim=0)
                    protos_n[ci] += len(chunk)
            protos = protos_acc / protos_n.unsqueeze(1)

            cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
            for start in range(0, len(eval_rows), 48):
                chunk = eval_rows[start : start + 48]
                xb = np.stack([noisy[r["row"], :length] for r in chunk])
                zb, _, _ = net.encode(spectrogram(rms_normalize(
                    torch.from_numpy(xb).to(device))))
                pred = (-torch.cdist(zb, protos) ** 2).argmax(dim=1)
                for r, pi in zip(chunk, pred.tolist()):
                    cm[CLASSES.index(r["cls"]), pi] += 1

            result["dwells"][wname] = cm.tolist()
            norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
            ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
            for i in range(len(CLASSES)):
                for j in range(len(CLASSES)):
                    if cm[i, j]:
                        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                                fontsize=7,
                                color="white" if norm[i, j] > 0.6 else "black")
            ax.set_xticks(range(len(CLASSES)))
            ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(CLASSES)))
            ax.set_yticklabels(CLASSES, fontsize=7)
            ax.set_title(f"{wname} dwell", fontsize=10)
            ax.set_xlabel("predicted", fontsize=8)
            if wname == list(DWELLS)[0]:
                ax.set_ylabel("true", fontsize=8)

            off = [(CLASSES[i], CLASSES[j], int(cm[i, j]))
                   for i in range(len(CLASSES)) for j in range(len(CLASSES))
                   if i != j and cm[i, j]]
            off.sort(key=lambda t: -t[2])
            print(f"[{wname}] top confusions: "
                  + (", ".join(f"{a}->{b}:{n}" for a, b, n in off[:4])
                     or "none"), flush=True)

    fig.tight_layout()
    fig.savefig(f"{cli.out_prefix}.png", dpi=190)
    Path(f"{cli.out_prefix}.json").write_text(json.dumps(result, indent=1))
    print(f"wrote {cli.out_prefix}.png / .json", flush=True)


if __name__ == "__main__":
    main()
