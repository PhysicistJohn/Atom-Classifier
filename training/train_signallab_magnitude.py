"""Magnitude-only (tinySA) flavor of the SignalLab-trained classifier.

Same architecture, same corpus, same 7 classes as the I/Q flavor — but the input
is the log power-spectrum SHAPE (magnitude, no phase), so it runs on a scalar
spectrum analyzer. Enroll/validate on the clean subset (matches a clean sweep),
train on the impaired subset for robustness.

Run:  .venv-training/bin/python training/train_signallab_magnitude.py
Exports src/embedding/assets/magnitude-{weights,prototypes}.json (+ parity fixture).
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import magnitude as mag  # noqa: E402
from model import Embedding, EMBED_DIM  # noqa: E402
from train import np_conv1d, embed_all, prototypes_from, nearest, sq_dist  # noqa: E402
from train_signallab import load_corpus  # noqa: E402

SEED = 20260722
ASSET_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "embedding", "assets")
EPISODES = 6000
EVAL_EVERY = 1000
K_SHOT, Q_QUERY = 5, 5


def np_forward_mag(x1l, feat_raw, weights) -> np.ndarray:
    a = x1l.astype(np.float32)
    for cv in weights["convs"]:
        a = np.maximum(np_conv1d(a, cv["weight"], cv["bias"], cv["in"], cv["out"], cv["k"], cv["stride"], cv["pad"]), 0.0)
    pooled = np.concatenate([a.mean(axis=1), a.std(axis=1)])
    fz = (np.asarray(feat_raw, np.float32) - np.asarray(weights["feat_mean"], np.float32)) / np.asarray(weights["feat_std"], np.float32)
    h = np.concatenate([pooled, fz])
    fc1, fc2 = weights["fc1"], weights["fc2"]
    h = np.maximum(np.asarray(fc1["weight"], np.float32).reshape(fc1["out"], fc1["in"]) @ h + np.asarray(fc1["bias"], np.float32), 0.0)
    z = np.asarray(fc2["weight"], np.float32).reshape(fc2["out"], fc2["in"]) @ h + np.asarray(fc2["bias"], np.float32)
    return z / (np.linalg.norm(z) + 1e-12)


def build_pool(iq, y, idx):
    xs, fs, ys = [], [], []
    for i in idx:
        shape, feat = mag.magnitude_from_iq(iq[i])
        xs.append(shape[None, :])
        fs.append(feat)
        ys.append(y[i])
    return np.stack(xs).astype(np.float32), np.stack(fs).astype(np.float32), np.array(ys, np.int64)


def sample_episode(x, feat, idx_by_class, rng, n_way, k, q):
    chosen = rng.choice(len(idx_by_class), size=n_way, replace=False)
    sx, sf, sl, qx, qf, ql = [], [], [], [], [], []
    for local, c in enumerate(chosen):
        pick = rng.choice(idx_by_class[c], size=k + q, replace=False)
        sx.append(x[pick[:k]]); sf.append(feat[pick[:k]]); sl += [local] * k
        qx.append(x[pick[k:]]); qf.append(feat[pick[k:]]); ql += [local] * q
    return (np.concatenate(sx), np.concatenate(sf), np.array(sl, np.int64),
            np.concatenate(qx), np.concatenate(qf), np.array(ql, np.int64))


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); rng = np.random.default_rng(SEED)
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    iq, y, impaired, classes = load_corpus()
    n_classes = len(classes)
    print(f"device: {dev} | corpus {iq.shape} | classes {classes} | MAG_LEN {mag.MAG_LEN}")

    clean_pos = rng.permutation(np.where(~impaired)[0])
    imp_pos = np.where(impaired)[0]
    n_en = int(0.30 * len(clean_pos)); n_va = int(0.30 * len(clean_pos))
    en_idx, va_idx = clean_pos[:n_en], clean_pos[n_en:n_en + n_va]
    tr_idx = np.concatenate([clean_pos[n_en + n_va:], imp_pos])

    print("building magnitude pools ...")
    xtr, ftr, ytr = build_pool(iq, y, tr_idx)
    fmean = ftr.mean(0); fstd = ftr.std(0) + 1e-6
    ftr = (ftr - fmean) / fstd
    idx_by_class = [np.where(ytr == c)[0] for c in range(n_classes)]
    xen, fen, yen = build_pool(iq, y, en_idx); fen = (fen - fmean) / fstd
    xva, fva, yva = build_pool(iq, y, va_idx); fva = (fva - fmean) / fstd
    print(f"  train {xtr.shape}  enroll {xen.shape}  val {xva.shape}")

    net = Embedding(EMBED_DIM, mag.N_MAG_FEATURES, in_channels=1).to(dev)
    scale = torch.nn.Parameter(torch.tensor(10.0, device=dev))
    opt = torch.optim.Adam(list(net.parameters()) + [scale], lr=1e-3, weight_decay=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPISODES)

    def eval_clean():
        protos = prototypes_from(embed_all(net, xen, fen, dev), yen, n_classes)
        pred, _ = nearest(embed_all(net, xva, fva, dev), protos)
        return float((pred == yva).mean())

    print(f"training {EPISODES} episodes ...")
    running, best_acc, best_state = 0.0, -1.0, None
    for ep in range(EPISODES):
        net.train()
        sx, sf, sl, qx, qf, ql = sample_episode(xtr, ftr, idx_by_class, rng, n_classes, K_SHOT, Q_QUERY)
        xb = torch.from_numpy(np.concatenate([sx, qx])).to(dev)
        fb = torch.from_numpy(np.concatenate([sf, qf])).to(dev)
        emb = net(xb, fb)
        se, qe = emb[: len(sx)], emb[len(sx):]
        protos = torch.stack([se[torch.from_numpy(sl).to(dev) == c].mean(0) for c in range(n_classes)])
        loss = torch.nn.functional.cross_entropy(-sq_dist(qe, protos) * scale, torch.from_numpy(ql).to(dev))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        running += loss.item()
        if (ep + 1) % 500 == 0:
            print(f"  ep {ep+1:5d}  loss {running/500:.4f}"); running = 0.0
        if (ep + 1) % EVAL_EVERY == 0:
            acc = eval_clean(); tag = ""
            if acc > best_acc:
                best_acc, best_state = acc, copy.deepcopy(net.state_dict()); tag = "  <- best"
            print(f"    [val(clean) {acc:.3f}]{tag}")
    if best_state is not None:
        net.load_state_dict(best_state)

    emb_en = embed_all(net, xen, fen, dev)
    protos = prototypes_from(emb_en, yen, n_classes)
    emb_va = embed_all(net, xva, fva, dev)
    pred, dists = nearest(emb_va, protos)
    closed = float((pred == yva).mean())
    nn_known = dists.min(1)
    thr = float(max(np.quantile(nn_known, 0.99) * 1.6, float(nn_known.max()) * 1.25))
    print(f"clean closed-set accuracy: {closed:.3f}")
    print("per class:", {classes[c]: round(float((pred[yva == c] == c).mean()), 3) for c in range(n_classes)})

    best_T, best_nll = 1.0, math.inf
    for T in np.linspace(0.02, 2.0, 100):
        lg = -dists / T; lg -= lg.max(1, keepdims=True); p = np.exp(lg); p /= p.sum(1, keepdims=True)
        nll = -np.log(p[np.arange(len(yva)), yva] + 1e-12).mean()
        if nll < best_nll: best_nll, best_T = nll, float(T)

    weights = net.export_weights()
    weights["input_len"] = mag.MAG_LEN
    weights["feat_mean"] = fmean.astype(float).tolist()
    weights["feat_std"] = fstd.astype(float).tolist()
    weights["magnitude"] = {"mag_len": mag.MAG_LEN, "mag_nfft": mag.MAG_NFFT, "margin": mag.MARGIN,
                            "smooth": pp_smooth(), "noise_floor_scale": pp_nfs(), "energy_edge": pp_edge()}
    net.eval()
    with torch.no_grad():
        probe = xva[:12]
        temb = net(torch.from_numpy(probe).to(dev), torch.from_numpy(fva[:12]).to(dev)).cpu().numpy()
    nemb = np.stack([np_forward_mag(probe[i], (fva[i] * fstd + fmean), weights) for i in range(len(probe))])
    fold = float(np.abs(temb - nemb).max())
    print(f"BN-fold parity: max|dz| = {fold:.2e}"); assert fold < 1e-4

    os.makedirs(ASSET_DIR, exist_ok=True)
    json.dump({"input_len": mag.MAG_LEN, "embed_dim": EMBED_DIM, "n_features": mag.N_MAG_FEATURES, **weights},
              open(os.path.join(ASSET_DIR, "magnitude-weights.json"), "w"))
    json.dump({"classes": classes, "embed_dim": EMBED_DIM, "prototypes": protos.astype(float).tolist(),
               "unknown_threshold": thr, "temperature": best_T},
              open(os.path.join(ASSET_DIR, "magnitude-prototypes.json"), "w"))
    # parity fixture: raw PSD-derived shape+feat -> embedding, for the TS port
    fix = []
    for c in [0, 1, 2]:
        gi = int(np.where(y == c)[0][0])
        shape, feat = mag.magnitude_from_iq(iq[gi])
        fix.append({"class": classes[c], "shape": shape.astype(float).tolist(),
                    "features": feat.astype(float).tolist(),
                    "embedding": np_forward_mag(shape[None, :], feat, weights).astype(float).tolist()})
    json.dump({"forward": fix}, open(os.path.join(ASSET_DIR, "magnitude-parity-fixture.json"), "w"))
    print("done. magnitude assets ->", os.path.relpath(ASSET_DIR))


def pp_smooth():
    import preprocess as pp; return pp.SMOOTH
def pp_nfs():
    import preprocess as pp; return pp.NOISE_FLOOR_SCALE
def pp_edge():
    import preprocess as pp; return pp.ENERGY_EDGE


if __name__ == "__main__":
    main()
