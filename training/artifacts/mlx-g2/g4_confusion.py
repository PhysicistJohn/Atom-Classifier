"""Per-dwell 7x7 confusion matrices for the MLX bf16 G4 final checkpoint.

Mirrors v7_trainer_mlx.main().evaluate() exactly:
  * prototypes = mean over the FIRST 64 train rows per class, read from the
    fp16 train cache (the precision asymmetry), offset 0, sliced [:length];
  * query rows = the FULL eval split (3264 rows), complex64, sliced [:length];
  * chunk_n = 48 with zero-slot padding (per-sample ops => padding is inert);
  * pred = argmin ||z - proto||^2 (argmax of -cdist_sq);
  * net compute dtype = bf16 via nn.Module.set_dtype, fp32 front end
    (spectrogram / rms_normalize) cast at model input, z/conf back to fp32;
  * TF32 left at the MLX default (ON) -- the run's own training config.

Beyond evaluate(): keeps the full per-row prediction vector, so it emits the
confusion matrices, the per-dwell misclassified-row SETS (and their pairwise
intersections / differences), per-profile recalls and the off-diagonal
confusion pairs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

TRAINER_DIR = Path("/Users/johnelliott/PersonalGitHub/Atom-Classifier/training/"
                   "zplane_ab/v2_full_variation/v7_probe")
OUT_DIR = Path("/Users/johnelliott/PersonalGitHub/Atom-Classifier/training/"
               "artifacts/mlx-g2")
CKPT = OUT_DIR / "g4_bf16.safetensors"

sys.path.insert(0, str(TRAINER_DIR))

import mlx.core as mx          # noqa: E402
import mlx.nn as nn            # noqa: E402  (kept for parity of import order)
from functools import partial  # noqa: E402

from v7_trainer_mlx import (   # noqa: E402
    CLASSES, CORPUS, DWELLS, GB, TrainState, load_init_checkpoint,
    measure_matmul_precision, params_sha256,
)
from v7_model_mlx import cdist_sq, rms_normalize, spectrogram  # noqa: E402

CHUNK_N = 48
N_CLS = len(CLASSES)


def main() -> None:
    rel, tf32 = measure_matmul_precision()
    print(f"mlx {mx.__version__} device={mx.default_device()} "
          f"TF32 {'ON' if tf32 else 'off'} (rel {rel:.2e}) dtype=bf16",
          flush=True)
    try:
        mx.set_wired_limit(int(22 * GB))
    except Exception as e:
        print(f"WARNING set_wired_limit: {e!r}", flush=True)
    mx.set_cache_limit(2 * GB)

    # ---------------- corpus split (identical construction) ----------------
    manifest = json.loads((CORPUS / "manifest.json").read_text())
    rows = manifest["rows"]
    train_by_class: dict[str, list[int]] = {c: [] for c in CLASSES}
    eval_rows: list[dict] = []
    for row in rows:
        if row["role"] == "train":
            train_by_class[row["cls"]].append(row["row"])
        else:
            eval_rows.append(row)
    n_eval = len(eval_rows)
    max_len = max(DWELLS.values())
    print(f"eval rows {n_eval}  train {sum(len(v) for v in train_by_class.values())}",
          flush=True)

    noisy = np.load(CORPUS / "noisy.npy", mmap_mode="r")

    # eval queries: complex64, first max_len samples (all dwells slice this)
    eval_store = mx.zeros((n_eval, max_len), dtype=mx.complex64)
    eval_ids = [r["row"] for r in eval_rows]
    step = 256
    for i0 in range(0, n_eval, step):
        ids = eval_ids[i0:i0 + step]
        eval_store[i0:i0 + len(ids)] = mx.array(
            np.ascontiguousarray(noisy[ids, :max_len]))
        mx.eval(eval_store)

    # prototype source: fp16 I/Q planes, first 64 train rows per class
    proto_ids = {c: train_by_class[c][:64] for c in CLASSES}
    flat_ids = [r for c in CLASSES for r in proto_ids[c]]
    tpos = {r: i for i, r in enumerate(flat_ids)}
    train_iq = mx.zeros((len(flat_ids), 2, max_len), dtype=mx.float16)
    for i0 in range(0, len(flat_ids), step):
        ids = flat_ids[i0:i0 + step]
        rc = np.asarray(noisy[ids, :max_len])
        train_iq[i0:i0 + len(ids)] = mx.array(
            np.stack((rc.real, rc.imag), axis=1).astype(np.float16))
        mx.eval(train_iq)
    del noisy

    # ---------------- model: load bf16 checkpoint ----------------
    sidecar = json.loads((CKPT.parent / (CKPT.name + ".json")).read_text())
    cfg = sidecar["config"]
    model = TrainState(width=cfg["width"], embed_dim=cfg["embed_dim"])
    kind, _opt_state, _sc = load_init_checkpoint(model, str(CKPT))
    mx.eval(model.parameters())
    model.net.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    got, want = params_sha256(model), sidecar.get("params_sha256")
    assert kind == "native" and got == want, f"params hash {got} != {want}"
    print(f"checkpoint ep {sidecar['episode']} hash {got[:16]} OK", flush=True)

    def to_c(a):
        return a.astype(mx.bfloat16)

    def to_f(a):
        return a.astype(mx.float32)

    @partial(mx.compile, inputs=[model.state])
    def eval_forward(xb):
        z, conf, _ = model.net.encode(to_c(spectrogram(rms_normalize(xb))))
        return to_f(z), to_f(conf)

    # ---------------- per-dwell evaluation ----------------
    y_true = np.array([CLASSES.index(r["cls"]) for r in eval_rows], dtype=int)
    results: dict[str, dict] = {}
    preds_by_dwell: dict[str, np.ndarray] = {}

    for wname, length in DWELLS.items():
        mx.clear_cache()
        protos_list = []
        for cls in CLASSES:
            ids = proto_ids[cls]
            z_parts = []
            for s0 in range(0, len(ids), CHUNK_N):
                chunk = ids[s0:s0 + CHUNK_N]
                slots = np.array([tpos[r] for r in chunk], dtype=np.int32)
                pad = np.concatenate(
                    [slots, np.zeros(CHUNK_N - len(chunk), dtype=np.int32)])
                seg = mx.take(train_iq, mx.array(pad), axis=0)[
                    :, :, :length].astype(mx.float32)
                xb = seg[:, 0] + 1j * seg[:, 1]
                zb, _ = eval_forward(xb)
                z_parts.append(zb[:len(chunk)])
            zc_all = mx.concatenate(z_parts, axis=0)
            protos_list.append(mx.sum(zc_all, axis=0) / len(ids))
        protos = mx.stack(protos_list, axis=0)
        mx.eval(protos)

        pred_all = np.zeros(n_eval, dtype=int)
        conf_all = np.zeros(n_eval, dtype=np.float32)
        for s0 in range(0, n_eval, CHUNK_N):
            n = min(CHUNK_N, n_eval - s0)
            slots = np.arange(s0, s0 + n, dtype=np.int32)
            pad = np.concatenate(
                [slots, np.zeros(CHUNK_N - n, dtype=np.int32)])
            xb = mx.take(eval_store, mx.array(pad), axis=0)[:, :length]
            zb, confb = eval_forward(xb)
            pred = mx.argmax(-cdist_sq(zb[:n], protos), axis=1)
            pred_all[s0:s0 + n] = np.asarray(pred)
            conf_all[s0:s0 + n] = np.asarray(confb[:n])
        preds_by_dwell[wname] = pred_all

        cm = np.zeros((N_CLS, N_CLS), dtype=int)
        for t, p in zip(y_true, pred_all):
            cm[t, p] += 1
        recalls = {c: float(cm[i, i] / cm[i].sum()) for i, c in enumerate(CLASSES)}
        bal = float(np.mean(list(recalls.values())))

        per_profile: dict[str, list[int]] = {}
        for r, t, p in zip(eval_rows, y_true, pred_all):
            per_profile.setdefault(r["profile"], []).append(int(t == p))
        prof = {k: (sum(v) / len(v), len(v)) for k, v in per_profile.items()}
        worst = sorted(prof.items(), key=lambda kv: (kv[1][0], kv[0]))[:6]

        offdiag = sorted(
            ((int(cm[i, j]), CLASSES[i], CLASSES[j])
             for i in range(N_CLS) for j in range(N_CLS) if i != j and cm[i, j]),
            reverse=True)[:4]

        results[wname] = {
            "confusion_matrix": cm.tolist(),
            "balanced_accuracy": bal,
            "overall_accuracy": float((pred_all == y_true).mean()),
            "per_class_recall": recalls,
            "min_profile_recall": min(v[0] for v in prof.values()),
            "worst_profiles": [
                {"profile": k, "recall": v[0], "n": v[1]} for k, v in worst],
            "top_offdiag": [
                {"true": t, "pred": p, "count": n} for n, t, p in offdiag],
            "n_errors": int((pred_all != y_true).sum()),
            "answered_frac": float((conf_all > 0).mean()),
            "answered_accuracy": float(
                (pred_all == y_true)[conf_all > 0].mean())
            if (conf_all > 0).any() else 0.0,
        }
        print(f"{wname}: bal {bal:.4f} acc {results[wname]['overall_accuracy']:.4f} "
              f"errors {results[wname]['n_errors']} "
              f"minCell {results[wname]['min_profile_recall']:.4f}", flush=True)

    # ---------------- error-set overlap ----------------
    err = {w: {int(eval_rows[i]["row"])
               for i in np.flatnonzero(preds_by_dwell[w] != y_true)}
           for w in DWELLS}
    e1, e25, e10 = err["1ms"], err["2.5ms"], err["10ms"]
    overlap = {
        "sizes": {w: len(err[w]) for w in DWELLS},
        "inter_1ms_2.5ms": len(e1 & e25),
        "inter_1ms_10ms": len(e1 & e10),
        "inter_2.5ms_10ms": len(e25 & e10),
        "inter_all_three": len(e1 & e25 & e10),
        "created_by_10ms_not_in_1ms": len(e10 - e1),
        "fixed_by_10ms_in_1ms_only": len(e1 - e10),
        "created_by_2.5ms_not_in_1ms": len(e25 - e1),
        "fixed_by_2.5ms": len(e1 - e25),
        "10ms_subset_of_1ms": e10 <= e1,
        "jaccard_1ms_10ms": len(e1 & e10) / max(1, len(e1 | e10)),
        "error_rows": {w: sorted(err[w]) for w in DWELLS},
        "rows_10ms_only": sorted(e10 - e1),
        "rows_10ms_only_detail": [
            {"row": int(r["row"]), "cls": r["cls"], "profile": r["profile"],
             "pred_10ms": CLASSES[preds_by_dwell["10ms"][i]],
             "pred_1ms": CLASSES[preds_by_dwell["1ms"][i]]}
            for i, r in enumerate(eval_rows) if int(r["row"]) in (e10 - e1)],
    }

    out = {
        "schema": "g4-confusion-v1",
        "checkpoint": str(CKPT),
        "episode": sidecar["episode"],
        "params_sha256": got,
        "n_eval_rows": n_eval,
        "classes": list(CLASSES),
        "dtype": "bf16",
        "tf32": tf32,
        "per_dwell": results,
        "error_overlap": overlap,
    }
    (OUT_DIR / "g4_confusion.json").write_text(json.dumps(out, indent=2))
    print("wrote g4_confusion.json", flush=True)

    make_figure(results)
    print("wrote g4_confusion.png", flush=True)


# ---------------------------------------------------------------- figure
def make_figure(results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
        "mathtext.fontset": "dejavuserif",
        "axes.linewidth": 0.7,
        "savefig.facecolor": "white",
    })
    labels = [c.upper() if c in ("am", "cw", "fm") else
              ("BT" if c == "bluetooth" else c.upper()) for c in CLASSES]
    cmap = LinearSegmentedColormap.from_list(
        "paperblue", ["#ffffff", "#dbe9f5", "#8fb6d8", "#4878a8", "#1f3f66"])

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 5.15),
                             gridspec_kw={"wspace": 0.26})
    for ax, wname in zip(axes, DWELLS):
        cm = np.array(results[wname]["confusion_matrix"], dtype=float)
        rown = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(rown, cmap=cmap, vmin=0.0, vmax=1.0,
                       interpolation="nearest")
        for i in range(N_CLS):
            for j in range(N_CLS):
                v, c = rown[i, j], int(cm[i, j])
                if c == 0:
                    ax.text(j, i, "·", ha="center", va="center",
                            fontsize=8, color="#c8c8c8")
                    continue
                col = "white" if v > 0.55 else "#1a1a1a"
                ax.text(j, i, f"{v*100:.1f}", ha="center", va="center",
                        fontsize=8.6, color=col)
                ax.text(j, i + 0.30, f"({c})", ha="center", va="center",
                        fontsize=6.6, color=col,
                        alpha=0.85 if v > 0.55 else 0.7)
        ax.set_xticks(range(N_CLS)); ax.set_yticks(range(N_CLS))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xticks(np.arange(-0.5, N_CLS, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, N_CLS, 1), minor=True)
        ax.grid(which="minor", color="#e6e6e6", lw=0.6)
        ax.tick_params(which="minor", length=0)
        ax.tick_params(which="major", length=2.5, width=0.7)
        ax.set_xlabel("predicted", fontsize=10.5)
        if ax is axes[0]:
            ax.set_ylabel("true", fontsize=10.5)
        r = results[wname]
        ax.set_title(f"{wname} dwell", fontsize=12, pad=20)
        ax.text(0.5, 1.055,
                f"balanced acc {r['balanced_accuracy']:.3f}   "
                f"min-cell {r['min_profile_recall']:.3f}   "
                f"{r['n_errors']}/3264 errors",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8.8, color="#444444")
    cb = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.02)
    cb.set_label("row-normalised rate", fontsize=9.5)
    cb.ax.tick_params(labelsize=8.5, length=2.5, width=0.7)
    cb.outline.set_linewidth(0.7)
    fig.suptitle("DACS per-dwell confusion, full eval split "
                 "(3264 rows, cell values are row-% with counts)",
                 fontsize=11.5, y=0.99)
    fig.savefig(OUT_DIR / "g4_confusion.png", dpi=300, bbox_inches="tight",
                facecolor="white")


if __name__ == "__main__":
    main()
