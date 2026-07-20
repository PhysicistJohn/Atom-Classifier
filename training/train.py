"""Train the metric embedding on the Apple GPU (MPS) and export the assets.

Prototypical episodic training: each episode enrolls prototypes from a support
set and classifies queries by nearest prototype, which is exactly the inference
rule — so few-shot enrollment is what the loss optimises, not an afterthought.

After training we:
  * enroll final prototypes for the KNOWN classes,
  * calibrate a fusion temperature and an open-set "unknown" distance threshold,
  * fold BatchNorm into the convs and verify a pure-numpy forward pass matches
    the torch eval output (this is the contract the TypeScript port must meet),
  * export weights JSON, prototypes JSON, a manifest, parity fixtures, and ONNX.

Run:  .venv-training/bin/python training/train.py
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
import rfgen  # noqa: E402
import preprocess as pp  # noqa: E402
import dataset as ds  # noqa: E402
from model import Embedding, INPUT_LEN, EMBED_DIM, N_FEATURES  # noqa: E402

SEED = 20260720
ASSET_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "embedding", "assets")
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

SNR_TRAIN = (2.0, 30.0)
POOL_PER_CLASS = 1200
VAL_PER_CLASS = 150
EPISODES = 8000
EVAL_EVERY = 1000
N_WAY = len(rfgen.KNOWN)
K_SHOT = 5
Q_QUERY = 5


def device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def sq_dist(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Squared euclidean [Q,N] without cdist (MPS-safe)."""
    qn = (q * q).sum(1, keepdim=True)
    pn = (p * p).sum(1)
    return (qn + pn.unsqueeze(0) - 2.0 * (q @ p.t())).clamp_min(0.0)


def supcon_loss(emb: torch.Tensor, labels: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """Supervised contrastive loss over unit embeddings; tightens same-class
    clusters and widens class gaps, which improves both closed-set margins and
    open-set separation (novel points sit relatively farther from tight protos).
    """
    sim = (emb @ emb.t()) / tau
    n = emb.shape[0]
    eye = torch.eye(n, device=emb.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e9)                      # drop self-similarity
    same = (labels[:, None] == labels[None, :]) & ~eye
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = same.float()
    denom = pos.sum(1).clamp_min(1.0)
    return -(pos * logp).sum(1).div(denom).mean()


# --------------------------------------------------------------------------
# pure-numpy forward (the reference the TS port must match; also validates fold)
# --------------------------------------------------------------------------
def np_conv1d(x, w_flat, bias, cin, cout, k, stride, pad):
    L = x.shape[1]
    xp = np.zeros((cin, L + 2 * pad), dtype=np.float32)
    xp[:, pad : pad + L] = x
    Lout = (L + 2 * pad - k) // stride + 1
    w = np.asarray(w_flat, dtype=np.float32).reshape(cout, cin, k)
    cols = np.stack([xp[:, t * stride : t * stride + k] for t in range(Lout)], axis=0)
    out = np.tensordot(cols, w, axes=([1, 2], [1, 2])).T  # [cout, Lout]
    out += np.asarray(bias, dtype=np.float32)[:, None]
    return out


def np_forward(x2l, weights) -> np.ndarray:
    a = x2l.astype(np.float32)
    for cv in weights["convs"]:
        a = np_conv1d(a, cv["weight"], cv["bias"], cv["in"], cv["out"], cv["k"], cv["stride"], cv["pad"])
        a = np.maximum(a, 0.0)
    pooled = np.concatenate([a.mean(axis=1), a.std(axis=1)]) if weights.get("pool") == "mean_std" else a.mean(axis=1)
    feat = pp.features_from_channels(x2l)
    feat = (feat - np.asarray(weights["feat_mean"], np.float32)) / np.asarray(weights["feat_std"], np.float32)
    h = np.concatenate([pooled, feat.astype(np.float32)])
    fc1, fc2 = weights["fc1"], weights["fc2"]
    w1 = np.asarray(fc1["weight"], np.float32).reshape(fc1["out"], fc1["in"])
    h = np.maximum(w1 @ h + np.asarray(fc1["bias"], np.float32), 0.0)
    w2 = np.asarray(fc2["weight"], np.float32).reshape(fc2["out"], fc2["in"])
    h = w2 @ h + np.asarray(fc2["bias"], np.float32)
    return h / (np.linalg.norm(h) + 1e-12)


# --------------------------------------------------------------------------
def embed_all(net, x, feat, dev, batch=256) -> np.ndarray:
    net.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(x[i : i + batch]).to(dev)
            fb = torch.from_numpy(feat[i : i + batch]).to(dev)
            outs.append(net(xb, fb).cpu().numpy())
    return np.concatenate(outs)


def prototypes_from(emb, y, n_classes) -> np.ndarray:
    return np.stack([emb[y == c].mean(0) for c in range(n_classes)]).astype(np.float32)


def nearest(emb, protos):
    d = ((emb[:, None, :] - protos[None, :, :]) ** 2).sum(-1)
    return d.argmin(1), d


def auroc(pos_scores, neg_scores) -> float:
    """AUROC that `pos` (novel) scores exceed `neg` (known). Higher = more novel."""
    labels = np.r_[np.ones(len(pos_scores)), np.zeros(len(neg_scores))]
    scores = np.r_[pos_scores, neg_scores]
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    dev = device()
    print(f"device: {dev}  | classes(known): {rfgen.KNOWN}")

    print("building pools ...")
    xtr, ftr, ytr, _ = ds.build_pool(rfgen.KNOWN, POOL_PER_CLASS, rng, SNR_TRAIN)
    fmean = ftr.mean(0)
    fstd = ftr.std(0) + 1e-6
    ftr = (ftr - fmean) / fstd
    idx_by_class = ds.class_indices(ytr, N_WAY)

    xen, fen, yen, _ = ds.build_pool(rfgen.KNOWN, VAL_PER_CLASS, rng, SNR_TRAIN)
    fen = (fen - fmean) / fstd
    xval, fval, yval, _ = ds.build_pool(rfgen.KNOWN, VAL_PER_CLASS, rng, SNR_TRAIN)
    fval = (fval - fmean) / fstd
    print(f"  train pool {xtr.shape}  feat {ftr.shape}")

    net = Embedding(EMBED_DIM, N_FEATURES).to(dev)
    scale = torch.nn.Parameter(torch.tensor(10.0, device=dev))
    opt = torch.optim.Adam(list(net.parameters()) + [scale], lr=1e-3, weight_decay=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPISODES)

    def eval_closed():
        emb_en = embed_all(net, xen, fen, dev)
        protos = prototypes_from(emb_en, yen, N_WAY)
        emb_v = embed_all(net, xval, fval, dev)
        pred, _ = nearest(emb_v, protos)
        return float((pred == yval).mean())

    print(f"training {EPISODES} episodes ({N_WAY}-way {K_SHOT}-shot) ...")
    running, best_acc, best_state = 0.0, -1.0, None
    for ep in range(EPISODES):
        net.train()
        sx, sf, sl, qx, qf, ql = ds.sample_episode(xtr, ftr, idx_by_class, rng, N_WAY, K_SHOT, Q_QUERY)
        xb = torch.from_numpy(np.concatenate([sx, qx])).to(dev)
        fb = torch.from_numpy(np.concatenate([sf, qf])).to(dev)
        sup_lt = torch.from_numpy(sl).to(dev)
        qry_lt = torch.from_numpy(ql).to(dev)

        emb = net(xb, fb)
        se, qe = emb[: len(sx)], emb[len(sx) :]
        protos = torch.stack([se[sup_lt == c].mean(0) for c in range(N_WAY)])
        logits = -sq_dist(qe, protos) * scale
        # SupCon auxiliary was evaluated and did not improve closed-set or
        # open-set on this corpus; prototypical CE alone is kept for clarity.
        loss = torch.nn.functional.cross_entropy(logits, qry_lt)

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        running += loss.item()
        if (ep + 1) % 500 == 0:
            print(f"  ep {ep+1:5d}  loss {running/500:.4f}  scale {scale.item():.2f}")
            running = 0.0
        if (ep + 1) % EVAL_EVERY == 0:
            acc = eval_closed()
            tag = ""
            if acc > best_acc:
                best_acc, best_state = acc, copy.deepcopy(net.state_dict())
                tag = "  <- best"
            print(f"    [val closed-set {acc:.3f}]{tag}")

    if best_state is not None:
        net.load_state_dict(best_state)
    print(f"restored best model (val closed-set {best_acc:.3f})")

    # ---- enroll final prototypes + full validation ----
    emb_en = embed_all(net, xen, fen, dev)
    protos = prototypes_from(emb_en, yen, N_WAY)
    emb_val = embed_all(net, xval, fval, dev)
    pred, dists = nearest(emb_val, protos)
    closed_acc = float((pred == yval).mean())
    nn_dist_known = dists.min(1)
    print(f"closed-set accuracy: {closed_acc:.3f}")

    # ---- open-set threshold from novel pool ----
    xnov, fnov, ynov, _ = ds.build_pool(rfgen.NOVEL, VAL_PER_CLASS, rng, SNR_TRAIN)
    fnov = (fnov - fmean) / fstd
    emb_nov = embed_all(net, xnov, fnov, dev)
    _, dnov = nearest(emb_nov, protos)
    nn_dist_novel = dnov.min(1)
    open_auroc = auroc(nn_dist_novel, nn_dist_known)
    thr = float(np.quantile(nn_dist_known, 0.95))
    print(f"open-set AUROC: {open_auroc:.3f}  | unknown-threshold(95%): {thr:.4f}")

    # ---- fusion temperature: grid-search min NLL of val labels ----
    best_T, best_nll = 1.0, math.inf
    for T in np.linspace(0.02, 2.0, 100):
        logits = -dists / T
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(1, keepdims=True)
        nll = -np.log(p[np.arange(len(yval)), yval] + 1e-12).mean()
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    print(f"fusion temperature: {best_T:.3f}  (val NLL {best_nll:.3f})")

    # ---- fold + verify numpy reference matches torch eval ----
    weights = net.export_weights()
    weights["feat_mean"] = fmean.astype(float).tolist()
    weights["feat_std"] = fstd.astype(float).tolist()
    net.eval()
    with torch.no_grad():
        probe_x = xval[:16]
        probe_f = torch.from_numpy(fval[:16]).to(dev)
        torch_emb = net(torch.from_numpy(probe_x).to(dev), probe_f).cpu().numpy()
    np_emb = np.stack([np_forward(probe_x[i], weights) for i in range(len(probe_x))])
    fold_err = float(np.abs(torch_emb - np_emb).max())
    print(f"BN-fold parity (torch vs numpy): max|dz| = {fold_err:.2e}")
    assert fold_err < 1e-4, "BN fold mismatch — export would be wrong"

    # ---- parity fixtures for the TS tests ----
    fwd_fixture = [
        {"input": xval[i].ravel().astype(float).tolist(), "embedding": np_emb[i].astype(float).tolist()}
        for i in range(8)
    ]
    pre_fixture = []
    for cls in ["qpsk", "ofdm", "gfsk"]:
        iq, _ = rfgen.synth(cls, rng, snr_range=(10.0, 20.0))
        norm, ctx = pp.preprocess(iq, scale_jitter=0.0)
        pre_fixture.append({
            "class": cls,
            "iq": np.stack([iq.real, iq.imag], 0).ravel().astype(float).tolist(),
            "iq_len": int(len(iq)),
            "expected": pp.to_channels(norm).ravel().astype(float).tolist(),
            "features": pp.iq_features(norm).astype(float).tolist(),
            "center": ctx["center"], "bw": ctx["bw"],
        })

    # ---- write assets ----
    os.makedirs(ASSET_DIR, exist_ok=True)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    with open(os.path.join(ASSET_DIR, "embedding-weights.json"), "w") as f:
        json.dump({
            "input_len": INPUT_LEN, "embed_dim": EMBED_DIM, "n_features": N_FEATURES,
            "preprocess": {"l_out": pp.L_OUT, "target_frac": pp.TARGET_FRAC, "nfft": pp.NFFT,
                           "energy_edge": pp.ENERGY_EDGE, "noise_floor_scale": pp.NOISE_FLOOR_SCALE, "smooth": pp.SMOOTH},
            **weights,
        }, f)
    with open(os.path.join(ASSET_DIR, "prototypes.json"), "w") as f:
        json.dump({
            "classes": rfgen.KNOWN, "embed_dim": EMBED_DIM,
            "prototypes": protos.astype(float).tolist(),
            "unknown_threshold": thr, "temperature": best_T,
        }, f)
    with open(os.path.join(ASSET_DIR, "parity-fixture.json"), "w") as f:
        json.dump({"forward": fwd_fixture, "preprocess": pre_fixture}, f)
    manifest = {
        "seed": SEED, "known": rfgen.KNOWN, "fewshot": rfgen.FEWSHOT, "novel": rfgen.NOVEL,
        "episodes": EPISODES, "snr_train": list(SNR_TRAIN),
        "metrics": {"closed_set_accuracy": closed_acc, "open_set_auroc": open_auroc, "best_val_accuracy": best_acc},
        "unknown_threshold": thr, "temperature": best_T,
    }
    with open(os.path.join(ASSET_DIR, "model-manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # ---- ONNX export (alternative onnxruntime-web path) ----
    try:
        net.eval().to("cpu")
        dummy_x = torch.from_numpy(xval[:1])
        dummy_f = torch.from_numpy(fval[:1])
        torch.onnx.export(
            net, (dummy_x, dummy_f), os.path.join(ARTIFACT_DIR, "embedding.onnx"),
            input_names=["iq", "features"], output_names=["embedding"],
            dynamic_axes={"iq": {0: "batch"}, "features": {0: "batch"}, "embedding": {0: "batch"}},
            opset_version=17,
        )
        print("exported ONNX")
    except Exception as e:  # noqa: BLE001
        print(f"ONNX export skipped: {e}")

    print("done. assets ->", os.path.relpath(ASSET_DIR))
    return manifest


if __name__ == "__main__":
    main()
