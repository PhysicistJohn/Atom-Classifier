"""Recon-driven training for the transfer campaign: online capture-length cropping,
pair-aligned reconstruction targets, and no auxiliary label heads.

    L = L_proto(class) + w_rec * (1 - coherence(recon, clean_pair))          [impaired rows]

WHY NO AUX HEADS. campaign3's own 2x2 on the multitask objective, 700 episodes each:
    c3_cnn (reference)          closed 0.8899   chirpAUROC 0.440
    c3_unet_CONTROL_classonly   closed 0.5876   chirpAUROC 0.454   coherence 0.0
    c3_unet_full                closed 0.5942   chirpAUROC 0.495   coherence 0.473
    c3_unet_recon_only          closed 0.6183   chirpAUROC 0.834   coherence 0.474
recon_only beats full on BOTH axes, so the profile/parameter heads are cancelling the
reconstruction term rather than adding to it. w_prof and w_par are therefore not exposed
here at all. (That 2x2 was run on a network whose bottleneck was 97-100% dead, so it
measured the decoder's shallow path rather than the objective -- which is a reason to
re-confirm it later, not a reason to re-introduce heads that lost.)

THREE DIFFERENCES FROM train_multitask.py, each because a measurement said so.

 1. THE CROP IS ONLINE, ONCE PER EPISODE.  `crop_cfg`
    length_aug.CropStream re-draws capture length and offset from the memory-mapped raw
    corpus on every batch. Applying a crop in pool_cache.py instead would freeze one random
    length per item for the whole run -- it would look applied and buy almost nothing.
    Measured cost: 1.8 ms per clean/impaired pair, i.e. ~0.13 s on a 70-sample episode
    against ~4.5 s of U-Net training.
    THE CONTROL ARM GOES THROUGH THE SAME CODE PATH. crop_cfg=CropConfig(mode="off") yields
    the full-length capture, preprocessed identically to the cached pool (asserted
    bit-close in length_aug._selftest). So the ON/OFF comparison differs in the crop and in
    nothing else -- not in the data path, not in the RNG consumption pattern, not in the
    preprocessing call.

 2. THE TARGET IS ROTATED INTO THE INPUT-CARRIER COORDINATE.  `align_pair`
    The corpus applies the same receiver-side centre nuisance to both pair members, so a
    common coarse centre estimate cannot create or remove their relative offset.  The
    remaining relative carrier is a real link impairment applied only to the impaired
    member.  `derotate_onto(clean, impaired)` deliberately puts that carrier into the clean
    target: the decoder is trained to remove noise and multipath while preserving the
    input's carrier, not to perform carrier recovery.  Measured near noise-free coherence
    rises from 0.237 to 0.547 because this is a narrower task, not because PSD quantisation
    was repaired.
    This policy is explicit and auditable. A convolutional decoder with a finite receptive
    field is not a reliable full-window carrier estimator; a product that promises CFO
    correction needs a separate global carrier-recovery stage. denoise_eval scores both the
    declared input-carrier-aligned target and the original clean-coordinate target.

 3. CLEAN ROWS DO NOT GET A RECONSTRUCTION LOSS.  `mask_clean_recon`
    25.2% of the native training pool is CLEAN, and for a clean row the impaired member IS
    the clean member -- so its coherence loss is minimized exactly by recon = input. A
    quarter of every batch was teaching the identity, and the trained model's measured
    coh(recon, input) = 0.92 is what that looks like. Those rows are kept for the class
    loss and dropped from the reconstruction term.

WHAT IS REPORTED, and why the old number was not enough: `final_coherence` in campaign3 is
a POOL average whose identity-passthrough value is 0.4624, because clean rows score exactly
1.0000 for free. The honest quantity is coherence on impaired rows against the passthrough
baseline on the same rows, which is what history["coherence_impaired"] tracks and what
denoise_eval reports at the end of the trial.
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

import native_preprocess as npp  # noqa: E402
import preprocess as pp  # noqa: E402
from train import embed_all, prototypes_from, nearest, sq_dist  # noqa: E402
from equalizer_frontend import coherence  # noqa: E402
from length_aug import CropConfig, CropStream  # noqa: E402
from denoise_eval import derotate_onto  # noqa: E402

K_SHOT, Q_QUERY = 5, 5      # identical to train_multitask, so batch size is comparable


def standardize_online_features(features, data):
    """Put online-crop features in the same coordinate system as eval features.

    ``prepare_data_v2`` fits ``fmean``/``fstd`` on the training pool and applies that
    transform to all cached train/enrol/validation features.  ``CropStream`` necessarily
    recomputes features after each crop, so it returns them in raw units.  Feeding those raw
    values directly to the network while evaluating with standardized values is a train/eval
    distribution mismatch, not augmentation.
    """
    features = np.asarray(features, dtype=np.float32)
    mean = np.asarray(data["fmean"], dtype=np.float32)
    std = np.asarray(data["fstd"], dtype=np.float32)
    if features.shape[-1] != mean.shape[0] or mean.shape != std.shape:
        raise ValueError(
            f"feature standardizer shape mismatch: features={features.shape}, "
            f"mean={mean.shape}, std={std.shape}"
        )
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
        raise ValueError("feature standardizer must contain finite means and positive stds")
    return ((features - mean) / std).astype(np.float32, copy=False)


def make_stream(data, condition="native", crop_cfg=None, align_pair=True, seed=0,
                want_clean=True):
    module = npp if condition == "native" else pp
    align = (lambda c, i: derotate_onto(c, i)) if align_pair else None
    return CropStream(data["tr_idx"], module, crop_cfg or CropConfig(mode="off"),
                      seed=seed, want_clean=want_clean, align_fn=align)


def embed_transfer(net, x, feat, dev, batch=256, embedding_only=False):
    """Batch embedding helper with an optional decoder-free U-Net path."""
    if not embedding_only or not hasattr(net, "forward_embedding"):
        return embed_all(net, x, feat, dev, batch=batch)
    net.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(np.asarray(x[i:i + batch])).to(dev)
            fb = torch.from_numpy(np.asarray(feat[i:i + batch])).to(dev)
            outs.append(net.forward_embedding(xb, fb).cpu().numpy())
    return np.concatenate(outs)


def train_transfer(net, data, dev, episodes, eval_every, log_tag="", lr=7e-4,
                   weight_decay=5e-4, warmup_frac=0.05, label_smoothing=0.05,
                   w_rec=1.0, crop_cfg=None, align_pair=True, mask_clean_recon=True,
                   excess_loss=False, log_every=None, seed=0, condition="native"):
    n_classes = data["n_classes"]
    rng = data["rng"]
    idx_by_class = data["idx_by_class"]
    xen, fen, yen = data["xen"], data["fen"], data["yen"]
    xva, fva, yva = data["xva"], data["fva"], data["yva"]
    selection_rows = data.get("selection_rows")
    if selection_rows is not None:
        selection_rows = np.asarray(selection_rows, dtype=np.int64)
        x_select = np.asarray(xva)[selection_rows]
        f_select = np.asarray(fva)[selection_rows]
        y_select = np.asarray(yva)[selection_rows]
    else:
        x_select, f_select, y_select = xva, fva, yva
    imp_tr = np.asarray(data["imp_tr"], dtype=bool)

    stream = make_stream(data, condition, crop_cfg, align_pair, seed, want_clean=w_rec > 0)
    cached_crop_off = (
        w_rec <= 0
        and getattr(crop_cfg, "mode", "off") == "off"
        and all(key in data for key in ("xtr", "ftr"))
    )
    print(f"[{log_tag}] w_rec={w_rec} crop={getattr(crop_cfg, 'mode', 'off')} "
          f"align_pair={align_pair} mask_clean_recon={mask_clean_recon} "
          f"excess_loss={excess_loss} cached_crop_off={cached_crop_off} "
          f"arch={getattr(net, 'config', lambda: {})()}", flush=True)

    net = net.to(dev)
    # Positive temperature, tracked separately from the inference model.  The previous free
    # scalar could cross zero, received weight decay, and was neither restored with the best
    # checkpoint nor recorded, making an exact training run impossible to reproduce.
    log_scale = torch.nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32, device=dev))
    opt = torch.optim.Adam(
        [
            {"params": list(net.parameters()), "weight_decay": weight_decay},
            {"params": [log_scale], "weight_decay": 0.0},
        ],
        lr=lr,
    )
    warm = max(1, int(episodes * warmup_frac))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(episodes - warm, 1))
    # Classification-only *training* can skip the decoder.  Selection evaluation only
    # reads embeddings regardless of the training objective, so it should always use the
    # numerically identical encoder-only path when available.
    training_embedding_only = w_rec <= 0 and hasattr(net, "forward_embedding")
    evaluation_embedding_only = hasattr(net, "forward_embedding")

    def eval_clean():
        protos = prototypes_from(
            embed_transfer(net, xen, fen, dev,
                           embedding_only=evaluation_embedding_only),
            yen,
            n_classes,
        )
        pred, _ = nearest(
            embed_transfer(net, x_select, f_select, dev,
                           embedding_only=evaluation_embedding_only),
            protos,
        )
        return float((pred == y_select).mean())

    acc = dict(ce=0.0, coh=0.0, pass_=0.0, n=0)
    best_acc, best_state = -1.0, None
    history = {"loss_ce": [], "coherence_impaired": [], "passthrough_impaired": [],
               "val_acc": [], "crop_len_median": [], "crop_len_p10": [],
               "logit_scale": [], "selection_n": int(len(y_select))}
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

        if cached_crop_off:
            # Ground-state tests assert that CropStream(mode="off") and these corrected,
            # non-jittered tensors are identical.  Classification-only training has no
            # paired target to construct, so re-running the same FFT/resampler 70 times per
            # episode only wastes time.  Cropped or reconstruction-bearing arms always
            # remain on the live stream above.
            Xn = np.asarray(data["xtr"])[allpos]
            Fn = np.asarray(data["ftr"])[allpos]
            Cn = None
            stream.last_lengths = [stream.n_samples] * len(allpos)
        else:
            Xn, Fn, Cn = stream.batch(allpos)
            # CropStream recomputes iq_features from the newly cropped waveform. Normalize
            # them with the SAME training-set statistics used by xen/fen and xva/fva.
            # Campaign 4 omitted this line, trained on raw feature units, and evaluated on
            # z-scored units.
            Fn = standardize_online_features(Fn, data)
        xb = torch.from_numpy(Xn).to(dev)
        fb = torch.from_numpy(Fn).to(dev)
        emb = net.forward_embedding(xb, fb) if training_embedding_only else net(xb, fb)
        se, qe = emb[: len(sup)], emb[len(sup):]
        sl_t = torch.from_numpy(sl).to(dev)
        protos = torch.stack([se[sl_t == c].mean(0) for c in range(n_classes)])
        scale = log_scale.exp().clamp(1e-3, 100.0)
        logits = -sq_dist(qe, protos) * scale
        loss_ce = torch.nn.functional.cross_entropy(logits, torch.from_numpy(ql).to(dev),
                                                     label_smoothing=label_smoothing)
        loss = loss_ce
        coh_v = pass_v = float("nan")
        if w_rec > 0 and Cn is not None:
            cb = torch.from_numpy(Cn).to(dev)
            inp = torch.complex(xb[:, 0, :], xb[:, 1, :])
            keep = torch.from_numpy(imp_tr[allpos]).to(dev) if mask_clean_recon \
                else torch.ones(len(allpos), dtype=torch.bool, device=dev)
            if keep.any():
                # Mask the LOSS, never the complex tensor. Boolean-mask indexing a complex
                # tensor inside the autograd graph -- net.last_recon[keep] -- is what the
                # MPS backward kernel cannot handle: it raises
                #   "Expected supportedFloatingType(scalar_type) ... but got false"
                # 3 s into every trial. Computing coherence on all rows and weighting the
                # scalar terms is the same quantity (both are means over the kept rows) and
                # keeps every masked op on real dtypes.
                w = keep.to(emb.dtype)
                wsum = w.sum().clamp_min(1.0)
                coh = coherence(net.last_recon, cb)
                base = coherence(inp, cb).detach()
                if excess_loss:
                    # T3: excess over passthrough, which is 0 for the identity and 1 for a
                    # perfect denoiser at every SNR. Makes the training objective the same
                    # quantity denoise_eval reports, instead of one dominated by the rows
                    # where the target is least reachable.
                    term = 1.0 - ((coh - base) / (1.0 - base).clamp_min(1e-3)).clamp(-1.0, 1.0)
                    loss = loss + w_rec * (term * w).sum() / wsum
                else:
                    loss = loss + w_rec * ((1.0 - coh) * w).sum() / wsum
                coh_v = ((coh.detach() * w).sum() / wsum).item()
                pass_v = ((base * w).sum() / wsum).item()

        opt.zero_grad(); loss.backward(); opt.step()
        if ep >= warm:
            cosine.step()
        acc["ce"] += loss_ce.item(); acc["n"] += 1
        if not np.isnan(coh_v):
            acc["coh"] += coh_v; acc["pass_"] += pass_v

        ln = log_every or max(1, min(500, episodes))
        if (ep + 1) % ln == 0:
            el = time.perf_counter() - t0
            n = max(acc["n"], 1)
            L = np.asarray(stream.last_lengths)
            print(f"  [{log_tag}] ep {ep+1:5d}/{episodes} ce {acc['ce']/n:.4f} "
                  f"coh(imp) {acc['coh']/n:.4f} vs passthrough {acc['pass_']/n:.4f} "
                  f"cropN med {int(np.median(L))} p10 {int(np.percentile(L, 10))} "
                  f"eta {(el/(ep+1))*(episodes-ep-1)/60:.1f}min", flush=True)
            history["loss_ce"].append(acc["ce"] / n)
            history["coherence_impaired"].append(acc["coh"] / n)
            history["passthrough_impaired"].append(acc["pass_"] / n)
            history["crop_len_median"].append(int(np.median(L)))
            history["crop_len_p10"].append(int(np.percentile(L, 10)))
            history["logit_scale"].append(float(log_scale.detach().exp().cpu()))
            acc = dict(ce=0.0, coh=0.0, pass_=0.0, n=0)
        if (ep + 1) % eval_every == 0 or (ep + 1) == episodes:
            a = eval_clean()
            history["val_acc"].append({"episode": ep + 1, "acc": a})
            tag = ""
            if a > best_acc:
                best_acc = a
                best_state = (copy.deepcopy(net.state_dict()),
                              float(log_scale.detach().cpu()))
                tag = "  <- best"
            print(f"    [{log_tag}] [selection {a:.3f}]{tag}", flush=True)
    wall = time.perf_counter() - t0
    if best_state is not None:
        net.load_state_dict(best_state[0])
        with torch.no_grad():
            log_scale.fill_(best_state[1])
    history["best_val_acc"] = best_acc
    history["wall_clock_s"] = wall
    history["episodes"] = episodes
    history["final_logit_scale"] = float(log_scale.detach().exp().cpu())
    # The coherence pair, always together. A bare coherence number has no scale: the
    # identity map scores 0.4624 on the full pool and 0.2813 on impaired rows.
    history["final_coherence_impaired"] = (history["coherence_impaired"] or [None])[-1]
    history["final_passthrough_impaired"] = (history["passthrough_impaired"] or [None])[-1]
    return net, history, wall


if __name__ == "__main__":
    # Smoke test only. CPU, 3 episodes, and a DELIBERATELY TINY net (base=2, depth=2): the
    # thing under test is the training loop -- crop stream, pair alignment, clean-row
    # masking, the coherence/passthrough pair -- not the architecture, and a full-size U-Net
    # forward+backward at L=16384 costs ~100 s per episode on CPU while a GPU trial is
    # already competing for the machine. Never allocates on MPS.
    import pool_cache
    from unet_transfer import TransferUNet
    torch.set_num_threads(2)
    d = pool_cache.load("native")
    d["condition"] = "native"
    net = TransferUNet(base=2, depth=2, taps=3, magnorm=True, skip_dropout=0.5,
                       feat_dropout=0.5)
    net, h, w = train_transfer(net, d, torch.device("cpu"), episodes=3, eval_every=1000,
                               log_tag="smoke", log_every=1,
                               crop_cfg=CropConfig(mode="loguniform"))
    print(f"[ok] smoke {w:.1f}s  ce {h['loss_ce']} coh {h['coherence_impaired']} "
          f"passthrough {h['passthrough_impaired']} cropN {h['crop_len_median']}")
