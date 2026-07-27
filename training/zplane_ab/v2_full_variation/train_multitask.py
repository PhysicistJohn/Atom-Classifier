"""Joint training for MultiTaskAutoencoder:

    L =        L_proto(class)                       # the task
      + w_rec  (1 - coherence(recon, clean_pair))   # latent must describe the emission
      + w_prof CE(profile_logits, 37-way profile)   # finer label already in the manifest
      + w_par  SmoothL1(params, physical params)    # latent must represent the nuisances

Every weight can be zeroed independently, which is what makes the campaign able to
attribute any gain: w_rec=w_prof=w_par=0 is the same encoder trained on classification
alone, so a win from the full objective cannot be explained by added capacity.

Parameter targets come from the corpus manifest, which the generator rewrite populated
with the physical draw for every realization. They are standardized (zero mean, unit
variance over the training split) so no single target dominates the gradient by unit
choice -- clockErrorPpm spans +/-40 while centreOffsetFrac spans +/-0.35.
"""
from __future__ import annotations

import copy
import json
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
from multitask_autoencoder import PARAM_KEYS  # noqa: E402
import denoise_eval as de  # noqa: E402 -- derotate_onto / best_freq_offset for pair alignment

K_SHOT, Q_QUERY = 5, 5

# Derotate the clean target onto the impaired member when the pair is built. Off restores
# the bin-accurate-only pairing so the two can be compared as a controlled A/B; the cache
# filename carries the flag so the arms cannot silently share a cache.
FINE_ALIGN = os.environ.get("FINE_ALIGN", "1") == "1"


def build_targets(condition, tr_idx, in_len, want_clean=True):
    """Clean reconstruction targets + profile ids + standardized physical parameters,
    all aligned to the TRAINING pool ordering."""
    manifest = json.load(open(os.path.join(CORPUS, "corpus.json")))
    items = manifest["items"]
    profiles = sorted({it["profile"] for it in items})
    pid = {p: i for i, p in enumerate(profiles)}

    prof = np.array([pid[items[i]["profile"]] for i in tr_idx], dtype=np.int64)
    raw = []
    for i in tr_idx:
        it = items[i]
        raw.append([
            it.get("snrDb", 0.0), it.get("centreOffsetFrac", 0.0), it.get("clockErrorPpm", 0.0),
            it.get("multipathTaps", 0.0),
            np.log10(max(it.get("centerHz", 1e6), 1.0)),
            np.log10(max(it["bandwidthHz"] / it["sampleRateHz"], 1e-9)),
        ])
    par = np.asarray(raw, dtype=np.float32)
    mu, sd = par.mean(0), par.std(0) + 1e-6
    par = (par - mu) / sd

    ctr = None
    # Building the clean targets preprocesses 5600 x 16384 samples TWICE (once to get the
    # impaired member's centre, once for the clean member at that centre) -- ~10 min. The
    # result depends only on (condition, tr_idx, in_len, corpus), all fixed for a campaign,
    # so cache it. Several equalizer/multitask runs otherwise repeat the same work.
    suffix = "_fine" if FINE_ALIGN else ""
    cache = os.path.join(CORPUS, f"_clean_targets_{condition}_{in_len}_{len(tr_idx)}{suffix}.npy")
    if want_clean and os.path.exists(cache):
        ctr = np.load(cache)
        return ctr, prof, par, len(profiles), (mu, sd)
    if want_clean:
        clean = load_clean_pairs()
        if clean is None:
            raise RuntimeError("corpus has no clean pairs; regenerate with hasCleanPairs")
        _raw_imp, _, _, _ = load_corpus()
        module = npp if condition == "native" else pp
        ctr = np.zeros((len(tr_idx), in_len), dtype=np.complex64)
        for j, i in enumerate(tr_idx):
            # centre estimated from the IMPAIRED member, applied to both -> pair stays aligned
            imp_norm, ctx_imp = module.preprocess(_raw_imp[i], l_out=in_len)
            norm, _ = module.preprocess(clean[i], l_out=in_len, force_center=ctx_imp["center"])
            if FINE_ALIGN:
                # force_center only aligns the pair to ONE PSD BIN. estimate_band works from a
                # 512-point PSD, so its quantization is 1.95e-3 cycles/sample, while a 16384-
                # sample window tolerates 6.1e-5 -- 32x finer. Measured on near-noise-free rows
                # (denoise_eval reachability probe, snr >= 36 dB where rho/(1+rho) = 0.9999):
                # the pair only reaches gamma^2 0.388 as built, and ONE fine derotation lifts it
                # to 0.644. That 0.26 is not noise and not channel, so it is not work a denoiser
                # should be learning to undo -- it is target corruption, and training against it
                # spends capacity chasing a carrier offset the front end failed to remove.
                norm = de.derotate_onto(norm[None, :], imp_norm[None, :])[0]
            ctr[j] = norm
        del clean, _raw_imp
        np.save(cache, ctr)
    return ctr, prof, par, len(profiles), (mu, sd)


def train_multitask(net, data, dev, episodes, eval_every, log_tag="", lr=7e-4,
                    weight_decay=5e-4, warmup_frac=0.05, label_smoothing=0.05,
                    w_rec=1.0, w_prof=0.3, w_par=0.3, log_every=None, crop=None):
    """crop: an optional length_aug.CropStream. When None every path below is unchanged
    and the precomputed pools are used exactly as before -- the control arm passes
    crop=CropStream(cfg=CropConfig(mode="off")) instead of None, so ON and OFF differ in
    the crop draw and in nothing else (same memmap reads, same preprocess call)."""
    n_classes = data["n_classes"]
    rng = data["rng"]
    xtr, ftr, idx_by_class = data["xtr"], data["ftr"], data["idx_by_class"]
    xen, fen, yen = data["xen"], data["fen"], data["yen"]
    xva, fva, yva = data["xva"], data["fva"], data["yva"]
    in_len = xtr.shape[-1]

    need_clean = w_rec > 0
    ctr, prof, par, n_prof, _ = build_targets(data.get("condition", "native"),
                                              data["tr_idx"], in_len, want_clean=need_clean)
    ctr_t = torch.from_numpy(ctr) if ctr is not None else None
    prof_t = torch.from_numpy(prof)
    par_t = torch.from_numpy(par)
    print(f"[{log_tag}] targets: profiles={n_prof} params={len(PARAM_KEYS)} "
          f"w_rec={w_rec} w_prof={w_prof} w_par={w_par}", flush=True)

    net = net.to(dev)
    scale = torch.nn.Parameter(torch.tensor(10.0, device=dev))
    opt = torch.optim.Adam(list(net.parameters()) + [scale], lr=lr, weight_decay=weight_decay)
    warm = max(1, int(episodes * warmup_frac))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(episodes - warm, 1))

    def eval_clean():
        protos = prototypes_from(embed_all(net, xen, fen, dev), yen, n_classes)
        pred, _ = nearest(embed_all(net, xva, fva, dev), protos)
        return float((pred == yva).mean())

    acc_ce = acc_coh = acc_prof = acc_par = 0.0
    best_acc, best_state = -1.0, None
    history = {"loss_ce": [], "coherence": [], "loss_prof": [], "loss_par": [], "val_acc": []}
    t0 = time.perf_counter()
    for ep in range(episodes):
        net.train()
        if ep < warm:
            for g in opt.param_groups:
                g["lr"] = lr * (ep + 1) / warm

        pos = np.concatenate([rng.choice(idx_by_class[c], K_SHOT + Q_QUERY, replace=False)
                              for c in range(n_classes)])
        step = K_SHOT + Q_QUERY
        sup = np.concatenate([pos[c * step: c * step + K_SHOT] for c in range(n_classes)])
        qry = np.concatenate([pos[c * step + K_SHOT: (c + 1) * step] for c in range(n_classes)])
        allpos = np.concatenate([sup, qry])
        sl = np.repeat(np.arange(n_classes), K_SHOT)
        ql = np.repeat(np.arange(n_classes), Q_QUERY)

        if crop is None:
            xb = torch.from_numpy(xtr[allpos]).to(dev)
            fb = torch.from_numpy(ftr[allpos]).to(dev)
            cb_np = None
        else:
            # Re-drawn every episode, so an item is seen at a different capture length each
            # time. Cropping in the pool build instead would freeze one length per item for
            # the whole run and buy almost nothing.
            xb_np, fb_np, cb_np = crop.batch(allpos)
            xb = torch.from_numpy(xb_np).to(dev)
            fb = torch.from_numpy(fb_np).to(dev)
        emb = net(xb, fb)
        se, qe = emb[: len(sup)], emb[len(sup):]
        sl_t = torch.from_numpy(sl).to(dev)
        protos = torch.stack([se[sl_t == c].mean(0) for c in range(n_classes)])
        logits = -sq_dist(qe, protos) * scale
        loss_ce = torch.nn.functional.cross_entropy(logits, torch.from_numpy(ql).to(dev),
                                                     label_smoothing=label_smoothing)
        loss = loss_ce
        coh_v = 0.0
        if w_rec > 0:
            # The cropped target comes from the SAME window as the cropped input; using the
            # cached full-length target here would pair a cropped input with an uncropped
            # target and silently destroy the reconstruction objective.
            cb = (torch.from_numpy(cb_np) if cb_np is not None else ctr_t[allpos]).to(dev)
            coh = coherence(net.last_recon, cb).mean()
            loss = loss + w_rec * (1.0 - coh)
            coh_v = coh.item()
        lp = 0.0
        if w_prof > 0:
            lprof = torch.nn.functional.cross_entropy(net.last_profile_logits, prof_t[allpos].to(dev))
            loss = loss + w_prof * lprof
            lp = lprof.item()
        la = 0.0
        if w_par > 0:
            lpar = torch.nn.functional.smooth_l1_loss(net.last_params, par_t[allpos].to(dev))
            loss = loss + w_par * lpar
            la = lpar.item()

        opt.zero_grad(); loss.backward(); opt.step()
        if ep >= warm:
            cosine.step()
        acc_ce += loss_ce.item(); acc_coh += coh_v; acc_prof += lp; acc_par += la

        ln = log_every or max(1, min(500, episodes))
        if (ep + 1) % ln == 0:
            el = time.perf_counter() - t0
            print(f"  [{log_tag}] ep {ep+1:5d}/{episodes} ce {acc_ce/ln:.4f} coh {acc_coh/ln:.4f} "
                  f"prof {acc_prof/ln:.4f} par {acc_par/ln:.4f} eta "
                  f"{(el/(ep+1))*(episodes-ep-1)/60:.1f}min", flush=True)
            history["loss_ce"].append(acc_ce / ln); history["coherence"].append(acc_coh / ln)
            history["loss_prof"].append(acc_prof / ln); history["loss_par"].append(acc_par / ln)
            acc_ce = acc_coh = acc_prof = acc_par = 0.0
        if (ep + 1) % eval_every == 0 or (ep + 1) == episodes:
            a = eval_clean()
            history["val_acc"].append({"episode": ep + 1, "acc": a})
            tag = ""
            if a > best_acc:
                best_acc, best_state = a, copy.deepcopy(net.state_dict()); tag = "  <- best"
            print(f"    [{log_tag}] [val(clean) {a:.3f}]{tag}", flush=True)
    wall = time.perf_counter() - t0
    if best_state is not None:
        net.load_state_dict(best_state)
    history["best_val_acc"] = best_acc; history["wall_clock_s"] = wall; history["episodes"] = episodes
    return net, history, wall
