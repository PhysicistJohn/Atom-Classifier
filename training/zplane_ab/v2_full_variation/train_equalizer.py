"""Joint training: equalizer front end + classifier, one objective.

    L = L_proto(classification) + lambda_eq * (1 - coherence(equalized, clean_pair))

The coherence term is what makes this different from simply adding depth to the
classifier: the front end gets a direct, physically-meaningful target (the emission
this capture came from) instead of only a gradient trickling back through the
classification loss. See equalizer_frontend.py for why coherence rather than MSE.

lambda_eq=0 reduces this to the plain classifier with extra layers, which is the
control the campaign needs -- otherwise any gain could just be added capacity.
"""
from __future__ import annotations

import copy
import os
import sys
import time

import numpy as np
import torch

_V2_DIR = os.path.dirname(os.path.abspath(__file__))
_ZPAB_DIR = os.path.dirname(_V2_DIR)
_TRAINING_DIR = os.path.dirname(_ZPAB_DIR)
for p in (_TRAINING_DIR, _ZPAB_DIR, _V2_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import dataset as ds  # noqa: E402
import preprocess as pp  # noqa: E402
import native_preprocess as npp  # noqa: E402
from train import embed_all, prototypes_from, nearest, sq_dist  # noqa: E402
from equalizer_frontend import coherence  # noqa: E402
from full_split import load_clean_pairs, CORPUS  # noqa: E402
from common_split import load_corpus  # noqa: E402

K_SHOT, Q_QUERY = 5, 5


def build_clean_targets(condition, tr_idx, in_len):
    """Preprocess the clean pair of every TRAINING item through the same front end as
    the impaired one, so coherence compares like with like."""
    cache = os.path.join(CORPUS, f"_clean_targets_{condition}_{in_len}_{len(tr_idx)}.npy")
    if os.path.exists(cache):
        return np.load(cache)   # shared with train_multitask; see its build_targets note
    clean = load_clean_pairs()
    if clean is None:
        return None
    _raw_imp, _, _, _ = load_corpus()
    module = npp if condition == "native" else pp
    out = np.zeros((len(tr_idx), in_len), dtype=np.complex64)
    for j, i in enumerate(tr_idx):
        _, ctx_imp = module.preprocess(_raw_imp[i], l_out=in_len)
        norm, _ = module.preprocess(clean[i], l_out=in_len, force_center=ctx_imp["center"])
        out[j] = norm
    del clean, _raw_imp   # ~3.7 GB; only the preprocessed training subset is needed from here
    np.save(cache, out)
    return out


def train_equalizer(net, data, dev, episodes, eval_every, log_tag="", lr=7e-4,
                    weight_decay=5e-4, warmup_frac=0.05, label_smoothing=0.05,
                    lambda_eq=1.0, log_every=None):
    n_classes = data["n_classes"]
    rng = data["rng"]
    xtr, ftr, idx_by_class = data["xtr"], data["ftr"], data["idx_by_class"]
    xen, fen, yen = data["xen"], data["fen"], data["yen"]
    xva, fva, yva = data["xva"], data["fva"], data["yva"]
    in_len = xtr.shape[-1]

    ctr = build_clean_targets(data.get("condition", "native"), data["tr_idx"], in_len)
    if ctr is None:
        raise RuntimeError("corpus has no clean pairs; regenerate with hasCleanPairs")
    ctr_t = torch.from_numpy(ctr)
    print(f"[{log_tag}] clean targets: {ctr.shape}, lambda_eq={lambda_eq}", flush=True)

    net = net.to(dev)
    scale = torch.nn.Parameter(torch.tensor(10.0, device=dev))
    opt = torch.optim.Adam(list(net.parameters()) + [scale], lr=lr, weight_decay=weight_decay)
    warm = max(1, int(episodes * warmup_frac))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(episodes - warm, 1))

    def eval_clean():
        protos = prototypes_from(embed_all(net, xen, fen, dev), yen, n_classes)
        pred, _ = nearest(embed_all(net, xva, fva, dev), protos)
        return float((pred == yva).mean())

    run_ce = run_coh = 0.0
    best_acc, best_state = -1.0, None
    history = {"loss_ce": [], "coherence": [], "val_acc": []}
    t0 = time.perf_counter()
    for ep in range(episodes):
        net.train()
        if ep < warm:
            for g in opt.param_groups:
                g["lr"] = lr * (ep + 1) / warm
        # sample_episode returns positions into the training pool, which index ctr too
        sx, sf, sl, qx, qf, ql, spos, qpos = ds.sample_episode(
            xtr, ftr, idx_by_class, rng, n_classes, K_SHOT, Q_QUERY, return_positions=True) \
            if "return_positions" in ds.sample_episode.__code__.co_varnames else (None,) * 8
        if sx is None:
            # harness build without position return: reconstruct by sampling positions here
            pos = np.concatenate([rng.choice(idx_by_class[c], K_SHOT + Q_QUERY, replace=False)
                                  for c in range(n_classes)])
            sup = np.concatenate([pos[c * (K_SHOT + Q_QUERY): c * (K_SHOT + Q_QUERY) + K_SHOT]
                                  for c in range(n_classes)])
            qry = np.concatenate([pos[c * (K_SHOT + Q_QUERY) + K_SHOT: (c + 1) * (K_SHOT + Q_QUERY)]
                                  for c in range(n_classes)])
            sl = np.repeat(np.arange(n_classes), K_SHOT)
            ql = np.repeat(np.arange(n_classes), Q_QUERY)
            allpos = np.concatenate([sup, qry])
            xb = torch.from_numpy(xtr[allpos]).to(dev)
            fb = torch.from_numpy(ftr[allpos]).to(dev)
            cb = ctr_t[allpos].to(dev)
            n_sup = len(sup)

        emb = net(xb, fb)
        se, qe = emb[:n_sup], emb[n_sup:]
        sl_t = torch.from_numpy(sl).to(dev)
        protos = torch.stack([se[sl_t == c].mean(0) for c in range(n_classes)])
        logits = -sq_dist(qe, protos) * scale
        loss_ce = torch.nn.functional.cross_entropy(logits, torch.from_numpy(ql).to(dev),
                                                     label_smoothing=label_smoothing)
        coh = coherence(net.last_equalized, cb).mean()
        loss = loss_ce + lambda_eq * (1.0 - coh)

        opt.zero_grad(); loss.backward(); opt.step()
        if ep >= warm:
            cosine.step()
        run_ce += loss_ce.item(); run_coh += coh.item()

        ln = log_every or max(1, min(500, episodes))
        if (ep + 1) % ln == 0:
            el = time.perf_counter() - t0
            print(f"  [{log_tag}] ep {ep+1:5d}/{episodes} ce {run_ce/ln:.4f} coh {run_coh/ln:.4f} "
                  f"elapsed {el:.0f}s eta {(el/(ep+1))*(episodes-ep-1)/60:.1f}min", flush=True)
            history["loss_ce"].append(run_ce / ln); history["coherence"].append(run_coh / ln)
            run_ce = run_coh = 0.0
        if (ep + 1) % eval_every == 0 or (ep + 1) == episodes:
            acc = eval_clean()
            history["val_acc"].append({"episode": ep + 1, "acc": acc})
            tag = ""
            if acc > best_acc:
                best_acc, best_state = acc, copy.deepcopy(net.state_dict()); tag = "  <- best"
            print(f"    [{log_tag}] [val(clean) {acc:.3f}]{tag}", flush=True)
    wall = time.perf_counter() - t0
    if best_state is not None:
        net.load_state_dict(best_state)
    history["best_val_acc"] = best_acc; history["wall_clock_s"] = wall; history["episodes"] = episodes
    return net, history, wall
