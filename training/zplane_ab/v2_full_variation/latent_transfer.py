"""Does the U-Net bottleneck actually TRANSFER? -- a probe built so it can fail.

WHAT THIS MEASURES, AND WHY IT IS NOT THE CAMPAIGN'S closed_overall.
campaign3's closed_overall trains the encoder and the (prototype) classifier on the SAME
7 labels, then scores those same 7 labels. A number like that cannot distinguish "the
latent is a good general description of the emission" from "the encoder memorised the 7
decision boundaries". The deliverable claims the first. This file measures the first.

PROTOCOL (identical for every model scored, including the controls):
  1. FREEZE the encoder. No gradient ever reaches it here.
  2. Extract a latent on the ENROLL pool (fit) and the VAL pool (eval). Both are held out
     of encoder training -- full_split carves enroll/val/train 30/30/40 and only `xtr`
     ever receives a gradient. Fitting the head on enroll rather than on xtr is the
     stricter choice: nothing the head sees was ever a training input to the encoder.
  3. Train ONLY a light head (nearest-centroid / linear / small MLP) on the fit latents.
  4. Score on the eval latents, on label sets the encoder was NEVER trained against.

THE FOUR TAPS. "The latent" is ambiguous in this architecture and the ambiguity matters,
because ComplexUNetMultiTask.forward concatenates the 12 HANDCRAFTED iq_features into the
trunk input (unet_multitask.py:113). A probe on the trunk or on the embedding is therefore
partly a probe on preprocess.iq_features, not on anything the U-Net learned.
    bott   -- [|bottleneck|.mean, |bottleneck|.std]. PURE learned signal, 256-d for the
              default U-Net. This is the headline tap; claims about "the latent" mean this.
    trunk  -- the 128-d hidden layer. Learned features PLUS the 12 handcrafted ones.
    embed  -- the 32-d unit-norm embedding the harness contract returns.
    feat12 -- CONTROL: the 12 handcrafted features alone, zero learned parameters.
`feat12` is the reason `trunk`/`embed` are not sufficient evidence on their own.

THE CONTROLS ARE THE POINT. Each one is a way for the claim to lose:
  C1  random_init   -- the SAME architecture, freshly random-initialised, same protocol.
                       Random 1-D conv stacks are genuinely strong feature extractors, so
                       this is not a straw man. MANDATORY.
  C2  cnn reference -- the 38k production CNN (c3_cnn, closed 0.8899), same protocol.
  C3  feat12        -- handcrafted features, no network at all.
  C4  class_prior   -- for the profile label sets: predict the most common profile GIVEN
                       THE TRUE 7-WAY CLASS. Beats a lot of "37-way transfer" numbers,
                       because 5 of the 7 classes have only one profile.
  C5  shuffled      -- the trained latent with the fit labels permuted. Must collapse to
                       chance. If it does not, the probe itself is broken and every other
                       number in the report is void.

THE LABEL SETS. profile37 alone is NOT sufficient evidence of transfer: profile is nearly
determined by class (am/cw/fm/dsss each have exactly 1 profile), so a perfect 7-way
classifier scores ~0.72 accuracy on it while learning nothing new. The load-bearing sets
are the WITHIN-CLASS ones, where the 7-way class label carries exactly zero information:
    ofdm24  -- 24 OFDM profiles (lte-* / nr-* / wifi*/ custom-*) among ofdm-class items.
    gsm7    -- 7 GSM burst types among gsm-class items.
    bt2     -- classic vs LE among bluetooth-class items.
    novel5  -- leave-one-class-out 5-shot recall: can a class the encoder never saw as a
               class be added from 5 examples? This is transfer to a label set of size 1.
!! profile37 IS NOT A TRANSFER SET FOR ANY MODEL TRAINED WITH w_prof > 0. train_multitask
   supervises net.profile_head on exactly these 37 labels (train_multitask.py:162), using
   the same `sorted(set(...))` id order this file uses. For c3_unet_full / c3_unet_recon_
   heavy / c3_mt_* that number is IN-distribution and must be read as memorisation, not
   transfer. It is only a transfer set for w_prof == 0 runs (c3_cnn, c3_unet_recon_only,
   c3_unet_CONTROL_classonly). ofdm24 / gsm7 / bt2 / novel5 stay valid for every model:
   nothing is ever trained on the WITHIN-class profile distinction.
CAVEAT ON class7 UNDER --per-profile-cap. The cap stratifies by PROFILE, which rebalances
the class marginal (ofdm has 24 profiles, cw has 1). Under a cap, read class7's `bal_acc`,
not `acc`, and do not compare either directly to campaign3's closed_overall -- that number
is on the full, unbalanced val pool with prototypes enrolled from the full enroll pool.
Run with --per-profile-cap omitted if you need a like-for-like closed-set comparison.

WHAT WOULD FALSIFY "THE LATENT TRANSFERS" -- state these before looking at the numbers.
ALL SIX ARE NOW EVALUATED MECHANICALLY BY verdict(). An earlier version claimed that and did
not: `transfers` was F1 AND F2 AND F6 only, F3 was computed but never gated, F4 was exported
as a bare number, and F5 was not computed at all -- so a model could return transfers=True
while its only "transfer" was the 7-way class label re-expressed, which is the exact failure
this file exists to catch.
  F1  bott(trained) does not beat bott(random_init) on ofdm24 AND gsm7 by more than the
      paired-bootstrap 95% CI. => U-Net training did not shape the bottleneck; report it.
  F2  bott(trained) does not beat feat12, same test. => 12 handcrafted numbers carry it.
  F3  profile37 <= class_prior, PAIRED-BOOTSTRAPPED like F1/F2 rather than compared as bare
      point estimates. => "transfer" is the 7-way class label re-expressed.
  F4  mlp - linear > MLP_GAP_MAX while linear does not beat random_init. => the information
      is present but not accessible to a LIGHT head. The deliverable says "light head", so
      this is a qualified FAILURE, not a pass.
  F5  bott(trained) loses to the reference model on every label set. Needs the reference's
      report; pass `reference=` (a previously written latent_transfer.json) or F5 is
      recorded as None and, being unevaluated, cannot contribute a pass.
  F6  shuffled control does not fall to chance => probe broken, discard the report; AND the
      probe-POWER control fails => the probe is too weak to see a signal that is present,
      which would make every reported ABSENCE meaningless. The shuffled half alone is close
      to vacuous (a linear head on permuted labels degenerates to the prior by
      construction), and it only tests for optimism while every conclusion here is a claim
      of absence. Both halves must pass.

MULTIPLICITY AND SEED NOISE, both of which used to bias toward declaring transfer:
  * F1/F2 previously used any() over {ofdm24, gsm7} -- two one-sided tests, roughly double
    the nominal false-positive rate, in a file that bills itself as conservative. They now
    require ALL transfer sets that have enough rows to be scored.
  * Every trained head is fitted at HEAD_SEEDS (3 seeds) and the per-row correctness is
    averaged before the paired bootstrap. Measured head-init spread on the decisive set is
    ~0.013 in accuracy, which is the same order as the margins being tested, and the
    bootstrap does not model it.

METRIC. Every flag and every paired bootstrap is on `acc`, because the paired bootstrap
needs a per-row correctness vector and only acc has one. `bal_acc` is reported beside it
everywhere (report_markdown prints "acc/bal"). Under --per-profile-cap the two differ a lot
on class7 (the cap balances by PROFILE, not by class): acc 0.758 vs bal_acc 0.276 for the
same latent. Say which one you mean; the two halves of this report used to disagree.

Also reported: latent dimensionality and participation-ratio effective rank per tap; the
metadata oracle (below); and the class_prior control. A collapsed bottleneck (the failure
mode this codebase has previously misdiagnosed three times) shows up as effective rank near
1 regardless of accuracy.

A CEILING, NOT JUST A CHANCE LEVEL. "ofdm24: 0.078 vs feat12 0.106 vs chance 0.042" cannot
distinguish "the latent is poor" from "ofdm24 is unlearnable". `metadata_oracle_baseline`
fits a lookup table on (sampleRateHz, bandwidthHz) straight from corpus.json -- information
no model has to learn, and an upper bound on any fs/bw-only predictor. On ofdm24 it reaches
~0.35 against chance 0.042, so there IS headroom. Note also that 57 of 109 (fs,bw) pairs map
to more than one OFDM profile (the LTE ETM variants and the four NB-IoT profiles are
identical in both), so part of ofdm24 is not separable in principle and even the oracle is
far below 1.0.

USE FROM A CAMPAIGN HARNESS. run_transfer_campaign.py calls this per trial. ALWAYS pass an
explicit per_profile_cap: the uncapped path is ~30 min PER ENCODER PASS for the U-Net on
CPU, and there are four passes, i.e. ~2 h per trial.
    from latent_transfer import latent_transfer_report
    r["latent_transfer"] = latent_transfer_report(net, data, EVAL_DEV, tag=tag,
                                                  per_profile_cap=20, reference=ref_json)
Put it after `net = net.to(EVAL_DEV)` and before `net.to("cpu")`.

STANDALONE:
    .venv-training/bin/python training/zplane_ab/v2_full_variation/latent_transfer.py \
        --ckpt .../artifacts/campaign3/c3_unet_recon_only/state_dict.pt --arch unet \
        --per-profile-cap 40 --device cpu

COST. Extraction is the whole cost and it is encoder-only: the decoder is never run, which
is ~2.8x cheaper than a full forward (measured 0.224 s vs 0.63 s per 16384-sample capture
on 4 CPU threads). Everything after extraction operates on <=268-d vectors and is free, so
every tap x head x label-set combination is amortised over ONE pass. --per-profile-cap
stratifies the subsample by profile rather than uniformly, which is what buys statistical
power on the 83-item OFDM profiles.

MEASURED SIZES AND COST (native condition, 2026-07-26).
  --per-profile-cap 20 -> fit 735 / eval 732 rows. Per label set (fit/eval/chance):
      class7 735/732/0.143   profile37 735/732/0.027   ofdm24 475/472/0.042
      gsm7   140/140/0.143   bt2 40/40/0.500  <- bt2 is too thin to conclude anything from.
    CNN 34 s total; U-Net ~25 min total (4 encoder passes: trained+random x fit+eval).
  no cap -> fit 4197 / eval 4197. CNN a few minutes; U-Net ~30 min PER PASS on CPU. Run the
    uncapped U-Net only on the device the campaign is already using.
The eval-set size is what sets the paired-bootstrap CI width, so "not separated" at cap 20
on gsm7/bt2 means UNDERPOWERED, not "equal". ofdm24 at ~470 rows is the set to trust.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_V2_DIR = os.path.dirname(os.path.abspath(__file__))
_ZPAB_DIR = os.path.dirname(_V2_DIR)
_TRAINING_DIR = os.path.dirname(_ZPAB_DIR)
for p in (_TRAINING_DIR, _ZPAB_DIR, _V2_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from model import Embedding, EMBED_DIM, N_FEATURES  # noqa: E402
from complex_multiscale_backbone import ComplexModReLU  # noqa: E402
from unet_multitask import ComplexUNetMultiTask  # noqa: E402
from common_split import CORPUS  # noqa: E402

SEED = 20260727
BOOT = 1000          # bootstrap resamples for accuracy CIs
K_SHOT_NOVEL = 5     # shots for the leave-one-class-out novel-class probe
N_NOVEL_TRIALS = 20
# Head-init seeds. One seed was an uncontrolled noise source of the same magnitude as the
# margins being tested (measured spread 0.013 on ofdm24, 0.021 on gsm7, against an F1 CI
# lower bound of 0.0085). Correctness is averaged over these before any bootstrap.
HEAD_SEEDS = (0, 1, 2)
MLP_GAP_MAX = 0.05   # F4: acc(mlp) - acc(linear) above this means a light head is NOT enough
# F6b: the probe must detect a signal this weak when one is genuinely present. Injected as
# a label-correlated direction of `alpha` latent standard deviations.
POWER_ALPHA = 0.25
POWER_MIN_LIFT = 0.10


# --------------------------------------------------------------------------- labels
def profile_table():
    """(profiles, pid, prof_of_item, cls_of_profile). Profile ordering is `sorted(set(...))`,
    the SAME convention train_multitask.build_targets uses, so profile ids here and the ids
    the w_prof head was trained against are the same integers."""
    manifest = json.load(open(os.path.join(CORPUS, "corpus.json")))
    items = manifest["items"]
    profiles = sorted({it["profile"] for it in items})
    pid = {p: i for i, p in enumerate(profiles)}
    prof_of_item = np.array([pid[it["profile"]] for it in items], dtype=np.int64)
    cls_of_profile = {}
    for it in items:
        cls_of_profile.setdefault(it["profile"], it["cls"])
    return profiles, pid, prof_of_item, cls_of_profile


def build_label_sets(classes, profiles, cls_of_profile):
    """Definitions shared by fit and eval so the dense remaps agree.

    Each entry: (class_name_or_None, [profile ids in that class], dense remap array).
    class_name None means 'all items'."""
    sets = {}
    sets["class7"] = dict(kind="class", n=len(classes), names=list(classes))
    sets["profile37"] = dict(kind="profile", restrict=None, n=len(profiles), names=list(profiles),
                             members=list(range(len(profiles))))
    for cname in classes:
        members = [i for i, p in enumerate(profiles) if cls_of_profile[p] == cname]
        if len(members) < 2:
            continue                                     # nothing to discriminate
        key = {"ofdm": "ofdm24", "gsm": "gsm7", "bluetooth": "bt2"}.get(cname, f"{cname}_prof")
        sets[key] = dict(kind="profile", restrict=cname, n=len(members),
                         names=[profiles[m] for m in members], members=members)
    return sets


# --------------------------------------------------------------------------- subsample
def build_probe_data(data, per_profile_cap=None, seed=SEED, verbose=True):
    """Choose the fit (enroll) and eval (val) rows ONCE, so every model scored is scored on
    exactly the same captures. That alignment is what makes the paired bootstrap against
    the random-init control valid."""
    profiles, _pid, prof_of_item, cls_of_profile = profile_table()
    classes = list(data["classes"])
    rng = np.random.default_rng(seed)

    def select(split):
        idx = np.asarray(data[f"{split}_idx"])
        prof = prof_of_item[idx]
        if per_profile_cap is None:
            sel = np.arange(len(idx))
        else:
            keep = []
            for p in np.unique(prof):
                rows = np.where(prof == p)[0]
                if len(rows) > per_profile_cap:
                    rows = rng.choice(rows, per_profile_cap, replace=False)
                keep.append(rows)
            sel = np.sort(np.concatenate(keep))
        return sel, prof[sel]

    sel_en, prof_en = select("en")
    sel_va, prof_va = select("va")
    out = dict(
        sel_en=sel_en, sel_va=sel_va,
        y7_en=np.asarray(data["yen"])[sel_en], y7_va=np.asarray(data["yva"])[sel_va],
        prof_en=prof_en, prof_va=prof_va,
        classes=classes, profiles=profiles,
        label_sets=build_label_sets(classes, profiles, cls_of_profile),
        n_fit=len(sel_en), n_eval=len(sel_va),
    )
    # fit/eval aliases: enroll is the fit split, val is the eval split. Everything
    # downstream speaks fit/eval so the split choice lives in exactly one place.
    out["y7_fit"] = out["y7_fit_cls"] = out["y7_en"]
    out["y7_eval"] = out["y7_eval_cls"] = out["y7_va"]
    out["prof_fit"], out["prof_eval"] = out["prof_en"], out["prof_va"]
    if verbose:
        print(f"[lt] probe rows: fit(enroll)={len(sel_en)} eval(val)={len(sel_va)} "
              f"cap={per_profile_cap}", flush=True)
    return out


def labels_for(pd_, name, split):
    """(labels, row-mask, n_classes) for one label set on one split."""
    spec = pd_["label_sets"][name]
    y7 = pd_[f"y7_{split}"]
    prof = pd_[f"prof_{split}"]
    if spec["kind"] == "class":
        return y7, np.ones(len(y7), dtype=bool), spec["n"]
    members = spec["members"]
    remap = -np.ones(len(pd_["profiles"]), dtype=np.int64)
    remap[np.asarray(members)] = np.arange(len(members))
    lab = remap[prof]
    mask = lab >= 0
    return lab, mask, spec["n"]


# --------------------------------------------------------------------------- taps
EXTRA_TAPS = False   # see bott4 below; off by default because 1024-d overfits a small fit split


def _chunk_pool(mag, k=4):
    """mean+std within k equal time chunks of the bottleneck sequence."""
    B, C, L = mag.shape
    m = mag[..., : (L // k) * k].reshape(B, C, k, L // k)
    return torch.cat([m.mean(-1).reshape(B, -1), m.std(-1).reshape(B, -1)], dim=-1)


def _unet_taps(net, xb, fb):
    """Re-implements ComplexUNetMultiTask.forward's ENCODER half only -- the decoder is
    ~2/3 of the cost and contributes nothing to any tap. Kept honest by verify_taps().

    `bott` is mean+std of |bottleneck| over the WHOLE time axis, which is what the model's
    own heads read (unet_multitask.py:112-113). That global pooling throws away all
    temporal structure at L/16. `bott4` (enable with EXTRA_TAPS) pools in 4 time chunks
    instead, and exists to separate two very different failures: if bott fails and bott4
    succeeds, the bottleneck HAS the information and the global pooling is destroying it --
    a head-side fix. If both fail, the bottleneck never encoded it."""
    z = torch.complex(xb[:, 0, :], xb[:, 1, :]).unsqueeze(1)
    v = z
    for blk in net.downs:
        v = blk(v)
        v = v[..., ::2]
    v = net.bottleneck(v)
    mag = v.abs()
    bott = torch.cat([mag.mean(-1), mag.std(-1)], dim=-1)
    h = net.trunk(torch.cat([bott, fb], dim=-1))
    emb = F.normalize(net.embed_head(h), dim=-1)
    out = {"bott": bott, "trunk": h, "embed": emb}
    if EXTRA_TAPS:
        out["bott4"] = _chunk_pool(mag, 4)
    return out


def _cnn_taps(net, xb, fb):
    x = xb
    for b in net.blocks:
        x = b(x)
    mean, std = x.mean(dim=-1), x.std(dim=-1, unbiased=False)
    bott = torch.cat([mean, std], dim=-1)
    h = F.relu(net.fc1(torch.cat([bott, fb], dim=-1)))
    emb = F.normalize(net.fc2(h), dim=-1)
    return {"bott": bott, "trunk": h, "embed": emb}


def _generic_taps(net, xb, fb):
    return {"embed": net(xb, fb)}


def tap_fn_for(net):
    if isinstance(net, ComplexUNetMultiTask):
        return _unet_taps
    if isinstance(net, Embedding):
        return _cnn_taps
    # unet_transfer.TransferUNet exposes the SAME encoder attribute names but is a distinct
    # class, so this isinstance chain fell through to _generic_taps and silently dropped the
    # 'bott' tap. Every campaign4 trial then reported verdict {"transfers": null, "why": "no
    # tap 'bott'"} -- the campaign's primary measurement, absent, with no error raised.
    # Duck-type on the encoder surface instead. verify_taps() asserts the reimplementation
    # reproduces forward()'s embedding, so a structural divergence fails loudly rather than
    # producing a confident wrong number.
    if all(hasattr(net, a) for a in ("downs", "bottleneck", "trunk", "embed_head")):
        return _unet_taps
    return _generic_taps


def verify_taps(net, tap_fn, in_len, dev):
    """The tap functions duplicate forward(). If unet_multitask.py changes and this file
    does not, every number below is silently wrong. So: assert the 'embed' tap reproduces
    the model's own forward() output, on two random captures, every time we extract."""
    net.eval()
    xb = torch.randn(2, 2, in_len, device=dev)
    fb = torch.randn(2, N_FEATURES, device=dev)
    with torch.no_grad():
        ref = net(xb, fb)
        got = tap_fn(net, xb, fb)["embed"]
    ok = torch.allclose(ref, got, atol=1e-4)
    if not ok:
        raise RuntimeError("latent_transfer tap functions have drifted from forward(); "
                           f"max |delta| = {(ref - got).abs().max().item():.3e}")


@torch.no_grad()
def extract_latents(net, data, dev, pd_, batch=16, verbose=True):
    """tap -> (Z_fit, Z_eval) as float32 numpy. One encoder pass per split."""
    net = net.to(dev).eval()
    tap_fn = tap_fn_for(net)
    verify_taps(net, tap_fn, int(data["input_length"]), dev)
    out = {}
    for split, xk, fk, selk in (("fit", "xen", "fen", "sel_en"), ("eval", "xva", "fva", "sel_va")):
        sel = pd_[selk]
        x, f = data[xk], data[fk]
        acc, t0 = {}, time.perf_counter()
        for i in range(0, len(sel), batch):
            rows = sel[i:i + batch]
            xb = torch.from_numpy(np.asarray(x[rows])).to(dev)
            fb = torch.from_numpy(np.asarray(f[rows])).to(dev)
            for k, v in tap_fn(net, xb, fb).items():
                acc.setdefault(k, []).append(v.float().cpu().numpy())
        for k, v in acc.items():
            out.setdefault(k, {})[split] = np.concatenate(v).astype(np.float32)
        if verbose:
            print(f"[lt]   {split}: {len(sel)} captures in {time.perf_counter()-t0:.1f}s",
                  flush=True)
    # feat12 CONTROL: the handcrafted features, no network involved at all.
    out["feat12"] = {"fit": np.asarray(data["fen"])[pd_["sel_en"]].astype(np.float32),
                     "eval": np.asarray(data["fva"])[pd_["sel_va"]].astype(np.float32)}
    return out


# --------------------------------------------------------------------------- heads
def _standardize(Ztr, Zva):
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6
    return (Ztr - mu) / sd, (Zva - mu) / sd


def head_proto(Ztr, ytr, Zva, C, **_):
    """Nearest class centroid on L2-normalised latents. ZERO trained parameters -- the
    lightest head there is, and the same decision rule the campaign's own evaluator uses.

    KNOWN BIAS, disclosed rather than corrected, because correcting it would stop this
    matching train.nearest(): rows are L2-normalised but the class CENTROIDS are not
    renormalised after averaging, and squared Euclidean distance is used. A class with
    larger intra-class spread has a shorter centroid and is therefore systematically
    favoured. novel_class_recall inherits it with a twist -- the k=5 novel centroid averages
    5 unit vectors while the base centroids average ~100, so the novel centroid's norm is
    systematically larger and the novel-class test is biased CONSERVATIVE (against detecting
    the novel class). That direction is the safe one for a probe built to fail."""
    A = Ztr / (np.linalg.norm(Ztr, axis=1, keepdims=True) + 1e-9)
    B = Zva / (np.linalg.norm(Zva, axis=1, keepdims=True) + 1e-9)
    protos = np.stack([A[ytr == c].mean(0) if (ytr == c).any() else np.zeros(A.shape[1])
                       for c in range(C)]).astype(np.float32)
    d = ((B[:, None, :] - protos[None, :, :]) ** 2).sum(-1)
    return d.argmin(1), 0


def _torch_head(module, Ztr, ytr, Zva, epochs, lr, wd, seed, batch=None):
    torch.manual_seed(seed)
    Xtr, Xva = _standardize(Ztr, Zva)
    xt = torch.from_numpy(Xtr); yt = torch.from_numpy(ytr.astype(np.int64))
    xv = torch.from_numpy(Xva)
    opt = torch.optim.Adam(module.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = len(xt)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        module.train()
        if batch is None or batch >= n:
            opt.zero_grad()
            F.cross_entropy(module(xt), yt).backward()
            opt.step()
        else:
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, batch):
                b = perm[i:i + batch]
                opt.zero_grad()
                F.cross_entropy(module(xt[b]), yt[b]).backward()
                opt.step()
        sched.step()
    module.eval()
    with torch.no_grad():
        pred = module(xv).argmax(1).numpy()
    return pred, sum(p.numel() for p in module.parameters())


def head_linear(Ztr, ytr, Zva, C, seed=0, **_):
    return _torch_head(nn.Linear(Ztr.shape[1], C), Ztr, ytr, Zva,
                       epochs=400, lr=1e-2, wd=1e-3, seed=seed)


def head_mlp(Ztr, ytr, Zva, C, seed=0, hidden=256, **_):
    m = nn.Sequential(nn.Linear(Ztr.shape[1], hidden), nn.ReLU(), nn.Dropout(0.1),
                      nn.Linear(hidden, C))
    return _torch_head(m, Ztr, ytr, Zva, epochs=300, lr=3e-3, wd=1e-3, seed=seed, batch=256)


HEADS = {"proto": head_proto, "linear": head_linear, "mlp": head_mlp}


# --------------------------------------------------------------------------- metrics
def metrics(pred, y, C, seed=SEED, boot=BOOT):
    correct = (pred == y).astype(np.float64)
    acc = float(correct.mean())
    per = [float((pred[y == c] == c).mean()) for c in range(C) if (y == c).any()]
    rng = np.random.default_rng(seed)
    bs = correct[rng.integers(0, len(correct), size=(boot, len(correct)))].mean(1)
    counts = np.bincount(y, minlength=C)
    return dict(acc=round(acc, 4), bal_acc=round(float(np.mean(per)), 4),
                ci95=[round(float(np.quantile(bs, 0.025)), 4), round(float(np.quantile(bs, 0.975)), 4)],
                n=int(len(y)), n_classes=int(C),
                chance=round(1.0 / C, 4),
                majority=round(float(counts.max() / counts.sum()), 4),
                correct=correct)          # kept for the paired bootstrap; stripped on export


def eval_head(hname, Zf, yf, Ze, ye, C, seed=SEED, head_seeds=HEAD_SEEDS):
    """Fit one head at several inits and average the PER-ROW correctness before scoring.

    head_proto has no trainable parameters, so it is fitted once; the torch heads are not
    bit-reproducible on this machine (CPU BLAS threading moves accuracy ~0.007, comparable
    to the margins under test) and are genuinely init-sensitive, so they are averaged.

    The returned `correct` is a per-row value in [0,1] rather than {0,1}. paired_delta and
    the bootstrap are means over rows either way, so this is a strict improvement: it keeps
    the pairing while removing head-init variance from the comparison."""
    seeds = (seed,) if hname == "proto" else tuple(seed + s for s in head_seeds)
    cors, bals, npar = [], [], 0
    for s in seeds:
        pred, npar = HEADS[hname](Zf, yf, Ze, C, seed=s)
        cors.append((pred == ye).astype(np.float64))
        bals.append(float(np.mean([float((pred[ye == c] == c).mean())
                                   for c in range(C) if (ye == c).any()])))
    correct = np.mean(cors, axis=0)
    m = metrics_from_correct(correct, ye, C, seed=seed)
    m["bal_acc"] = round(float(np.mean(bals)), 4)
    m["head_params"] = int(npar)
    m["n_head_seeds"] = len(seeds)
    m["acc_seed_spread"] = round(float(np.max([c.mean() for c in cors]) -
                                       np.min([c.mean() for c in cors])), 4)
    return m


def metrics_from_correct(correct, y, C, seed=SEED, boot=BOOT):
    """Same shape as metrics(), but from an already-computed per-row correctness vector."""
    correct = np.asarray(correct, dtype=np.float64)
    rng = np.random.default_rng(seed)
    bs = correct[rng.integers(0, len(correct), size=(boot, len(correct)))].mean(1)
    counts = np.bincount(y, minlength=C)
    return dict(acc=round(float(correct.mean()), 4), bal_acc=None,
                ci95=[round(float(np.quantile(bs, 0.025)), 4),
                      round(float(np.quantile(bs, 0.975)), 4)],
                n=int(len(y)), n_classes=int(C), chance=round(1.0 / C, 4),
                majority=round(float(counts.max() / max(counts.sum(), 1)), 4),
                correct=correct)


def paired_delta(correct_a, correct_b, seed=SEED, boot=BOOT):
    """95% CI on acc(a) - acc(b) over the SAME eval captures. If this CI contains 0 the two
    models are not distinguishable at this sample size -- which is the F1 falsification."""
    d = correct_a - correct_b
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(boot, len(d)))].mean(1)
    lo, hi = float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))
    return dict(delta=round(float(d.mean()), 4), ci95=[round(lo, 4), round(hi, 4)],
                separated=bool(lo > 0.0))


def effective_rank(Z):
    """Participation ratio of the covariance spectrum: (sum l)^2 / sum l^2. Equals D for an
    isotropic latent and 1 for a fully collapsed one. Independent of any head, so it fails
    even when accuracy happens to look fine."""
    Zc = Z - Z.mean(0)
    lam = np.linalg.svd(Zc, compute_uv=False) ** 2
    s = lam.sum()
    return round(float(s * s / (np.square(lam).sum() + 1e-30)), 2) if s > 0 else 0.0


# --------------------------------------------------------------------------- controls
def class_prior_baseline(pd_, name):
    """C4. Predict the most frequent profile GIVEN THE TRUE 7-WAY CLASS, fit on the fit
    split. A latent that does not beat this has not learned anything the 7-way class label
    did not already say."""
    lab_f, m_f, C = labels_for(pd_, name, "fit")
    lab_e, m_e, _ = labels_for(pd_, name, "eval")
    y_f, y_e = pd_["y7_fit_cls"][m_f], pd_["y7_eval_cls"][m_e]
    lab_f, lab_e = lab_f[m_f], lab_e[m_e]
    table = {}
    for c in np.unique(y_f):
        v = lab_f[y_f == c]
        table[int(c)] = int(np.bincount(v, minlength=C).argmax()) if len(v) else 0
    fallback = int(np.bincount(lab_f, minlength=C).argmax())
    pred = np.array([table.get(int(c), fallback) for c in y_e], dtype=np.int64)
    return metrics(pred, lab_e, C)


def metadata_oracle_baseline(pd_, name, data):
    """A CEILING, not a chance level. Predict the label from (sampleRateHz, bandwidthHz)
    read straight out of corpus.json -- an oracle lookup table fitted on the fit rows and
    scored on the eval rows.

    Why this is the right ceiling for the within-class profile sets: an OFDM profile is
    largely a choice of numerology, and numerology is mostly visible as (fs, bw). Anything
    the latent could plausibly recover about the profile is at or below what those two
    numbers already determine. Without it, "0.078 vs chance 0.042" cannot distinguish a poor
    latent from an unlearnable label set. Unseen (fs,bw) combinations fall back to the modal
    label, so this is achievable, not an information-theoretic bound."""
    manifest = json.load(open(os.path.join(CORPUS, "corpus.json")))
    items = manifest["items"]

    def key_of(i):
        it = items[i]
        return (round(float(it["sampleRateHz"]), 3), round(float(it["bandwidthHz"]), 3))

    lab_f, m_f, C = labels_for(pd_, name, "fit")
    lab_e, m_e, _ = labels_for(pd_, name, "eval")
    idx_f = np.asarray(data["en_idx"])[pd_["sel_en"]][m_f]
    idx_e = np.asarray(data["va_idx"])[pd_["sel_va"]][m_e]
    lab_f, lab_e = lab_f[m_f], lab_e[m_e]
    table: dict[tuple, np.ndarray] = {}
    for i, l in zip(idx_f, lab_f):
        table.setdefault(key_of(int(i)), np.zeros(C, dtype=np.int64))[l] += 1
    fallback = int(np.bincount(lab_f, minlength=C).argmax())
    pred = np.array([int(table[k].argmax()) if (k := key_of(int(i))) in table else fallback
                     for i in idx_e], dtype=np.int64)
    out = metrics(pred, lab_e, C)
    out["distinct_keys_fit"] = len(table)
    out["ambiguous_keys"] = int(sum(1 for v in table.values() if (v > 0).sum() > 1))
    return out


def probe_power_control(Zf, lab_f, Ze, lab_e, C, alpha=POWER_ALPHA, seed=SEED):
    """F6b. Can this probe SEE a weak signal that is genuinely there?

    The shuffled-label control only tests for optimism: it asks whether the head invents
    structure that is not present. Every conclusion this file draws is the opposite kind of
    claim -- an ABSENCE -- and nothing in the report used to control for a probe too weak,
    at this dimensionality and this fit-set size, to detect a signal that is present.

    So: take the REAL latent, add a label-correlated direction of `alpha` latent standard
    deviations, and re-run the same head with the same hyperparameters on the same rows. If
    the accuracy does not lift materially, the probe is underpowered at this operating point
    and every 'the latent does not carry X' statement in the report is unsupported."""
    rng = np.random.default_rng(seed)
    D = Zf.shape[1]
    R = rng.normal(size=(C, D)).astype(np.float32)
    R /= (np.linalg.norm(R, axis=1, keepdims=True) + 1e-9)
    sd = Zf.std(0, keepdims=True) + 1e-6
    base = eval_head("linear", Zf, lab_f, Ze, lab_e, C, seed=seed)
    Zf2 = Zf + alpha * sd * R[lab_f]
    Ze2 = Ze + alpha * sd * R[lab_e]
    lift = eval_head("linear", Zf2, lab_f, Ze2, lab_e, C, seed=seed)
    return dict(alpha=alpha, acc_base=base["acc"], acc_with_weak_signal=lift["acc"],
                lift=round(lift["acc"] - base["acc"], 4),
                min_lift_required=POWER_MIN_LIFT,
                detects_weak_signal=bool(lift["acc"] - base["acc"] >= POWER_MIN_LIFT))


def randomize_(net, seed=0):
    """C1. Fresh random init of an architecturally identical network. Walks the module tree
    calling reset_parameters(), and re-applies ComplexModReLU's constant init (-2.0) which
    has no reset_parameters of its own -- so the result matches what the constructor would
    have produced, not a half-reset hybrid."""
    torch.manual_seed(seed)
    for m in net.modules():
        if isinstance(m, ComplexModReLU):
            with torch.no_grad():
                m.thresh_raw.fill_(-2.0)
        elif hasattr(m, "reset_parameters"):
            m.reset_parameters()
    return net


def novel_class_recall(Zfit, y7_fit, Zeval, y7_eval, n_classes, k=K_SHOT_NOVEL,
                       trials=N_NOVEL_TRIALS, seed=SEED):
    """Leave-one-class-out K-shot recall on the frozen latent. For each class c: build
    centroids for the other 6 from the fit split, add a centroid from k random shots of c,
    and ask what fraction of c's EVAL items land on it. Chance is 1/n_classes.

    PRECISION MATTERS HERE. This is leave-one-class-out for the HEAD, not for the encoder:
    the encoder was trained 7-way on all of them. Do not describe a good number here as
    "enrolling a class the encoder never saw" -- it saw all seven. The claim it supports is
    narrower: the frozen latent supports adding a label from 5 examples with no fitting."""
    A = Zfit / (np.linalg.norm(Zfit, axis=1, keepdims=True) + 1e-9)
    B = Zeval / (np.linalg.norm(Zeval, axis=1, keepdims=True) + 1e-9)
    rng = np.random.default_rng(seed)
    protos = np.stack([A[y7_fit == c].mean(0) for c in range(n_classes)])
    per = {}
    for c in range(n_classes):
        base = np.delete(protos, c, axis=0)
        pool = np.where(y7_fit == c)[0]
        q = B[y7_eval == c]
        if len(q) == 0 or len(pool) == 0:
            continue
        rec = []
        for _ in range(trials):
            pick = rng.choice(pool, size=k, replace=len(pool) < k)
            ext = np.vstack([base, A[pick].mean(0)[None, :]])
            d = ((q[:, None, :] - ext[None, :, :]) ** 2).sum(-1)
            rec.append(float((d.argmin(1) == len(base)).mean()))
        per[c] = round(float(np.mean(rec)), 4)
    return dict(per_class=per, mean=round(float(np.mean(list(per.values()))), 4) if per else None,
                chance=round(1.0 / n_classes, 4), k=k)


# --------------------------------------------------------------------------- probe
def probe_latents(latents, pd_, heads=("proto", "linear", "mlp"),
                  label_sets=None, seed=SEED, verbose=True):
    """All taps x heads x label sets on one already-extracted latent set. Cheap: everything
    here operates on <=268-d vectors."""
    label_sets = label_sets or list(pd_["label_sets"].keys())
    res = {}
    for tap, Z in latents.items():
        Zf, Ze = Z["fit"], Z["eval"]
        res[tap] = {"dim": int(Zf.shape[1]), "eff_rank_eval": effective_rank(Ze), "sets": {}}
        for name in label_sets:
            lab_f, m_f, C = labels_for(pd_, name, "fit")
            lab_e, m_e, _ = labels_for(pd_, name, "eval")
            if m_f.sum() < 2 * C or m_e.sum() < C:
                continue
            entry = {}
            for hname in heads:
                entry[hname] = eval_head(hname, Zf[m_f], lab_f[m_f], Ze[m_e], lab_e[m_e],
                                         C, seed=seed)
            res[tap]["sets"][name] = entry
        # C5: probe sanity. Fit the strongest head on PERMUTED class labels; must be chance.
        lab_f, m_f, C = labels_for(pd_, "class7", "fit")
        lab_e, m_e, _ = labels_for(pd_, "class7", "eval")
        sh = np.random.default_rng(seed).permutation(lab_f[m_f])
        pred, _ = HEADS["linear"](Zf[m_f], sh, Ze[m_e], C, seed=seed)
        res[tap]["shuffled_control"] = metrics(pred, lab_e[m_e], C, seed=seed)
        # C6: probe POWER. Run it on the decisive within-class set when there is one, since
        # that is the operating point (C=24, few hundred fit rows) the conclusions live at.
        pw_set = next((s for s in ("ofdm24", "gsm7", "class7") if s in res[tap]["sets"]),
                      "class7")
        pl_f, pm_f, pC = labels_for(pd_, pw_set, "fit")
        pl_e, pm_e, _ = labels_for(pd_, pw_set, "eval")
        res[tap]["power_control"] = dict(
            label_set=pw_set,
            **probe_power_control(Zf[pm_f], pl_f[pm_f], Ze[pm_e], pl_e[pm_e], pC, seed=seed))
        res[tap]["novel5"] = novel_class_recall(Zf, pd_["y7_fit_cls"], Ze, pd_["y7_eval_cls"],
                                                len(pd_["classes"]), seed=seed)
        if verbose:
            s = res[tap]["sets"]
            head = ", ".join(f"{k}={s[k]['linear']['bal_acc']:.3f}" for k in s)
            print(f"[lt]   {tap:7s} d={res[tap]['dim']:4d} effrank={res[tap]['eff_rank_eval']:6.2f} "
                  f"| linear bal_acc: {head}", flush=True)
    return res


def _strip(o):
    """Drop the per-item correctness vectors before JSON export (they exist only for the
    paired bootstrap)."""
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if k != "correct"}
    if isinstance(o, list):
        return [_strip(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def latent_transfer_report(net, data, dev, tag="", per_profile_cap=None,
                           heads=("proto", "linear", "mlp"), label_sets=None,
                           include_random_control=True, batch=16, seed=SEED,
                           probe_data=None, verbose=True, reference=None):
    """THE campaign-harness entry point. Freeze `net`, probe its latents, and run the
    random-init control on an architecturally identical network.

    Returns a JSON-serialisable dict. Read `verdict` first: it is the mechanical evaluation
    of F1-F6 above, not a summary."""
    t0 = time.perf_counter()
    pd_ = probe_data or build_probe_data(data, per_profile_cap=per_profile_cap, seed=seed,
                                         verbose=verbose)
    if verbose:
        print(f"[lt] === {tag or type(net).__name__} === trained model", flush=True)
    lat = extract_latents(net, data, dev, pd_, batch=batch, verbose=verbose)
    trained = probe_latents(lat, pd_, heads=heads, label_sets=label_sets, seed=seed,
                            verbose=verbose)

    rand = None
    if include_random_control:
        if verbose:
            print(f"[lt] === {tag} === CONTROL 1: random-init encoder", flush=True)
        rnet = randomize_(copy.deepcopy(net).to("cpu"), seed=seed % (2 ** 31))
        rlat = extract_latents(rnet, data, dev, pd_, batch=batch, verbose=verbose)
        rlat.pop("feat12", None)          # identical by construction; no control value
        rand = probe_latents(rlat, pd_, heads=heads, label_sets=label_sets, seed=seed,
                             verbose=verbose)
        del rnet, rlat

    names = label_sets or list(pd_["label_sets"].keys())
    prior, oracle = {}, {}
    for name in names:
        spec = pd_["label_sets"][name]
        if spec["kind"] == "profile" and spec.get("restrict") is None:
            prior[name] = class_prior_baseline(pd_, name)
        if spec["kind"] == "profile":
            try:
                oracle[name] = metadata_oracle_baseline(pd_, name, data)
            except Exception as e:               # missing idx arrays -> report, do not crash
                oracle[name] = {"error": str(e)}

    out = dict(tag=tag, n_fit=pd_["n_fit"], n_eval=pd_["n_eval"],
               per_profile_cap=per_profile_cap, seed=seed, head_seeds=list(HEAD_SEEDS),
               taps=trained, random_init_control=rand, class_prior_control=prior,
               metadata_oracle=oracle,
               reference_tag=(reference or {}).get("tag"),
               wall_s=round(time.perf_counter() - t0, 1))
    out["verdict"] = verdict(out, reference=reference)
    return _strip(out)


def verdict(out, primary_tap="bott", primary_head="linear",
            transfer_sets=("ofdm24", "gsm7"), reference=None):
    """Mechanical evaluation of ALL SIX falsification criteria. Every field is a way to LOSE.

    `transfers` is True only if F1..F6 all pass, where an UNEVALUATED criterion (F5 with no
    reference supplied) counts as not passed. It is deliberately conservative: a False here
    with a high closed_overall means the campaign number was measuring memorisation.

    Everything flagged here is on `acc`, which is the only metric with a per-row correctness
    vector to bootstrap. bal_acc sits beside it in every row; they differ substantially under
    --per-profile-cap and the two halves of this report used to quietly use different ones."""
    t = out["taps"].get(primary_tap)
    if t is None:
        return {"transfers": None, "why": f"no tap '{primary_tap}'"}
    r = (out.get("random_init_control") or {}).get(primary_tap)
    feat = out["taps"].get("feat12")
    v = {"primary_tap": primary_tap, "primary_head": primary_head,
         "metric_for_all_flags": "acc", "transfer_sets": list(transfer_sets), "per_set": {}}

    # F6a: with the fit labels permuted the head must not beat the trivial baseline. Stated
    # as "the LOWER bound of the shuffled accuracy does not exceed max(chance, majority)".
    # On its own this is close to vacuous -- a linear head on permuted labels degenerates to
    # the prior by construction -- so F6b below carries the real weight.
    sh = t.get("shuffled_control", {})
    v["F6a_no_false_signal"] = bool(sh) and sh["ci95"][0] <= max(sh["chance"], sh["majority"])
    v["shuffled_acc"] = sh.get("acc")
    # F6b: the probe must SEE a weak signal that is genuinely present. Without this, every
    # claim of absence in the report is unsupported -- an underpowered probe reports the
    # same "no transfer" as a genuinely empty latent.
    pw = t.get("power_control", {})
    v["F6b_probe_detects_weak_signal"] = bool(pw.get("detects_weak_signal"))
    v["power_control"] = pw
    v["F6_probe_sane"] = bool(v["F6a_no_false_signal"] and v["F6b_probe_detects_weak_signal"])
    # Deltas are computed for EVERY label set (class7 included -- "how much of closed_overall
    # survives a frozen encoder + light head" is worth seeing), but only `transfer_sets`
    # drive the pass/fail flags, because class7 is the set the encoder was trained on and a
    # win there is not evidence of transfer.
    beats_random, beats_feat, light_enough = [], [], []
    oracles = out.get("metadata_oracle") or {}
    for name in list(t["sets"].keys()):
        e = t["sets"].get(name)
        if not e:
            continue
        row = {"acc": e[primary_head]["acc"], "bal_acc": e[primary_head]["bal_acc"],
               "chance": e[primary_head]["chance"],
               "acc_seed_spread": e[primary_head].get("acc_seed_spread"),
               "decides_verdict": name in transfer_sets}
        # a CEILING beside the chance level, so "poor latent" and "unlearnable set" are
        # distinguishable rather than conflated.
        if isinstance(oracles.get(name), dict) and "acc" in oracles[name]:
            row["metadata_oracle_acc"] = oracles[name]["acc"]
        if r and name in r["sets"]:
            row["vs_random_init"] = paired_delta(e[primary_head]["correct"],
                                                 r["sets"][name][primary_head]["correct"])
            if name in transfer_sets:
                beats_random.append(row["vs_random_init"]["separated"])
        if feat and name in feat["sets"]:
            row["vs_feat12"] = paired_delta(e[primary_head]["correct"],
                                            feat["sets"][name][primary_head]["correct"])
            if name in transfer_sets:
                beats_feat.append(row["vs_feat12"]["separated"])
        # F4: is a LIGHT head enough, or does only the MLP see it? A big MLP-over-linear gap
        # is only damning when the linear head ALSO fails to beat random init -- that is the
        # "information present but not lightly accessible" case the deliverable rules out.
        if "mlp" in e and "linear" in e:
            gap = round(e["mlp"]["acc"] - e["linear"]["acc"], 4)
            row["head_capacity_gap_mlp_minus_linear"] = gap
            if name in transfer_sets:
                lin_wins = bool(row.get("vs_random_init", {}).get("separated"))
                light_enough.append(bool(gap <= MLP_GAP_MAX or lin_wins))
        v["per_set"][name] = row
    # ALL transfer sets, not any(): two one-sided tests with no multiplicity correction
    # roughly doubles the false-positive rate, and the error direction was toward declaring
    # transfer. A set with too few rows to be scored simply is not in t["sets"] and so does
    # not contribute -- which is why n_eval per set is printed.
    v["F1_beats_random_init"] = bool(beats_random) and all(beats_random)
    v["F2_beats_feat12"] = bool(beats_feat) and all(beats_feat)
    v["F4_light_head_is_enough"] = bool(light_enough) and all(light_enough)
    v["F4_max_gap_allowed"] = MLP_GAP_MAX

    # F3, now paired-bootstrapped on the same eval rows rather than compared as bare point
    # estimates. The point-estimate version certified a margin whose CI straddles zero.
    p37 = t["sets"].get("profile37")
    cp = (out.get("class_prior_control") or {}).get("profile37")
    if p37 and cp:
        pdl = paired_delta(p37[primary_head]["correct"], cp["correct"])
        v["F3_profile37_beats_class_prior"] = bool(pdl["separated"])
        v["profile37_vs_class_prior"] = pdl
        v["profile37_acc"] = p37[primary_head]["acc"]
        v["profile37_class_prior_acc"] = cp["acc"]
    else:
        # profile37 not scored (e.g. --label-sets restricted). Do not silently pass.
        v["F3_profile37_beats_class_prior"] = None

    # F5: does the trained latent lose to the reference model on EVERY label set? Needs the
    # reference's own report -- a single run cannot compute it, which is why the earlier
    # version simply omitted it while claiming it was enforced.
    if reference:
        rt = (reference.get("taps") or {}).get(primary_tap, {}).get("sets", {})
        cmp_ = {}
        for name, row in v["per_set"].items():
            ref = rt.get(name, {}).get(primary_head, {}).get("acc")
            if ref is not None:
                cmp_[name] = {"this": row["acc"], "reference": ref,
                              "wins": bool(row["acc"] > ref)}
        v["F5_vs_reference"] = cmp_
        v["F5_not_dominated_by_reference"] = bool(cmp_) and any(c["wins"] for c in cmp_.values())
    else:
        v["F5_not_dominated_by_reference"] = None
        v["F5_note"] = ("no reference report supplied; F5 is UNEVALUATED and therefore "
                        "cannot contribute a pass. Pass reference=<the CNN's "
                        "latent_transfer.json> to evaluate it.")
    # How much of the model's headline accuracy is carried by the 12 handcrafted features
    # that forward() concatenates in, rather than by anything the encoder learned? `trunk`
    # and `embed` see feat12; `bott` does not. If trunk_trained is close to trunk_RANDOM,
    # the trained encoder is contributing little and the reported closed_overall is largely
    # a readout of preprocess.iq_features.
    tr, em = out["taps"].get("trunk"), out["taps"].get("embed")
    rtr = (out.get("random_init_control") or {}).get("trunk")
    leak = {}
    for name in list(t["sets"].keys()):
        row = {}
        for lbl, blk in (("bott", t), ("trunk", tr), ("embed", em), ("trunk_RANDOM", rtr),
                         ("feat12", feat)):
            e = (blk or {}).get("sets", {}).get(name, {}).get(primary_head)
            if e:
                row[lbl] = e["acc"]
        if row:
            leak[name] = row
    v["feat_leakage_acc_by_tap"] = leak
    v["eff_rank"] = {k: out["taps"][k]["eff_rank_eval"] for k in out["taps"]}
    if r:
        v["eff_rank_random_init"] = {k: (out["random_init_control"] or {})[k]["eff_rank_eval"]
                                     for k in (out["random_init_control"] or {})}
    v["novel5_trained"] = t["novel5"]["mean"]
    v["novel5_random_init"] = (r or {}).get("novel5", {}).get("mean")
    # All six. `is True` matters: an UNEVALUATED criterion is None, and None must not pass.
    checks = {k: v.get(k) for k in ("F1_beats_random_init", "F2_beats_feat12",
                                    "F3_profile37_beats_class_prior", "F4_light_head_is_enough",
                                    "F5_not_dominated_by_reference", "F6_probe_sane")}
    v["criteria"] = checks
    v["criteria_unevaluated"] = [k for k, x in checks.items() if x is None]
    v["transfers"] = all(x is True for x in checks.values())
    return v


# --------------------------------------------------------------------------- reporting
def report_markdown(rep):
    """Compact table. Every trained row is printed next to its random-init counterpart,
    because a trained number on its own is not interpretable."""
    L = [f"### latent transfer -- {rep['tag']}  (fit={rep['n_fit']} eval={rep['n_eval']}, "
         f"cap={rep['per_profile_cap']}, {rep['wall_s']}s)", "",
         "Cells are **acc/bal_acc**. Every flag in the verdict is on `acc`; `bal_acc` is",
         "printed beside it because under a per-profile cap the two differ a lot and this",
         "report used to quote one while the verdict used the other.", ""]
    sets = sorted({s for t in rep["taps"].values() for s in t["sets"]})
    L += ["| tap | model | dim | effrank | head | " + " | ".join(sets) + " | novel5 |",
          "|---|---|---|---|---|" + "---|" * (len(sets) + 1)]
    for tap in rep["taps"]:
        for who, blk in (("trained", rep["taps"]),
                         ("random", rep.get("random_init_control") or {})):
            t = blk.get(tap)
            if t is None:
                continue
            for h in ("proto", "linear", "mlp"):
                cells = []
                for s in sets:
                    e = t["sets"].get(s, {}).get(h)
                    cells.append(f"{e['acc']:.3f}/{e['bal_acc']:.3f}" if e else "-")
                if all(c == "-" for c in cells):
                    continue
                nv = t["novel5"]["mean"] if h == "proto" else None
                L.append(f"| {tap} | {who} | {t['dim']} | {t['eff_rank_eval']} | {h} | "
                         + " | ".join(cells) + f" | {nv if nv is not None else '-'} |")
    # BASELINE AND CEILING ROWS, in the same table units.
    orc = rep.get("metadata_oracle") or {}
    cells = []
    for s in sets:
        e = orc.get(s)
        cells.append(f"{e['acc']:.3f}/{e['bal_acc']:.3f}" if isinstance(e, dict) and "acc" in e
                     else "-")
    if any(c != "-" for c in cells):
        L.append("| CEILING | fs+bw oracle | - | - | lookup | " + " | ".join(cells) + " | - |")
    cp = rep.get("class_prior_control", {}).get("profile37")
    if cp:
        L += ["", f"CONTROL class_prior (predict profile from the TRUE 7-way class): "
                  f"acc={cp['acc']:.3f} bal_acc={cp['bal_acc']:.3f}"]
    pw = (rep.get("verdict") or {}).get("power_control") or {}
    if pw:
        L += ["", f"CONTROL probe power on `{pw.get('label_set')}`: injecting a "
                  f"{pw.get('alpha')}-sd label direction moves acc "
                  f"{pw.get('acc_base')} -> {pw.get('acc_with_weak_signal')} "
                  f"(lift {pw.get('lift')}, need >= {pw.get('min_lift_required')}). "
                  f"Without this passing, every ABSENCE reported above is unsupported."]
    L += ["", "```", json.dumps(rep["verdict"], indent=1), "```"]
    return "\n".join(L)


# --------------------------------------------------------------------------- self-test
def selftest():
    """The probe must return ~1.0 for a latent that trivially encodes the label and ~chance
    for pure noise. If it cannot do BOTH, no number this file produces means anything.
    Runs in under a second; `--selftest` should be run whenever this file is edited."""
    # The RuntimeWarnings this used to emit ("overflow/invalid in matmul") are spurious
    # Accelerate-BLAS noise, NOT a numeric problem: both operands are float64 and the output
    # has no NaN or inf (max |value| ~4). An earlier comment here blamed "the float32 BLAS
    # path", which was simply wrong. Silenced rather than explained away, and asserted.
    np.seterr(over="ignore", invalid="ignore", divide="ignore")
    rng = np.random.default_rng(0)
    C, D, n = 8, 32, 800
    y = rng.integers(0, C, n)
    onehot = np.eye(C)[y]
    W = rng.normal(size=(C, D))
    good = (onehot @ W + 0.10 * rng.normal(size=(n, D))).astype(np.float32)
    assert np.isfinite(good).all(), "synthetic latent is not finite -- the warning was real"
    junk = rng.normal(size=(n, D)).astype(np.float32)
    half = n // 2
    ok = True
    for name, Z in (("informative", good), ("noise", junk)):
        for h in ("proto", "linear", "mlp"):
            pred, _ = HEADS[h](Z[:half], y[:half], Z[half:], C, seed=0)
            a = float((pred == y[half:]).mean())
            want = a > 0.90 if name == "informative" else a < 0.25
            ok &= want
            print(f"[selftest] {name:12s} {h:6s} acc={a:.3f} {'ok' if want else 'FAIL'}")

    # THE REGIME THAT ACTUALLY MATTERS. The two cases above are noiseless-separable and
    # pure-noise; neither is anywhere near where every reported failure lives (C=24, a few
    # hundred fit rows, 256 dimensions, accuracy ~0.1). A probe that passes those two and
    # is blind here would certify "no transfer" for a latent that has some. So: a WEAK but
    # real signal at the operating point must be detected.
    C2, D2, n2 = 24, 256, 950
    y2 = rng.integers(0, C2, n2)
    base = rng.normal(size=(n2, D2))
    W2 = rng.normal(size=(C2, D2)); W2 /= np.linalg.norm(W2, axis=1, keepdims=True)
    weak = (base + POWER_ALPHA * np.eye(C2)[y2] @ W2 * np.sqrt(D2)).astype(np.float32)
    h2 = n2 // 2
    m_null = eval_head("linear", base.astype(np.float32)[:h2], y2[:h2],
                       base.astype(np.float32)[h2:], y2[h2:], C2)
    m_weak = eval_head("linear", weak[:h2], y2[:h2], weak[h2:], y2[h2:], C2)
    want = m_weak["acc"] - m_null["acc"] >= POWER_MIN_LIFT
    ok &= want
    print(f"[selftest] weak signal @ operating point (C={C2}, d={D2}, n_fit={h2}): "
          f"null {m_null['acc']:.3f} -> weak {m_weak['acc']:.3f} "
          f"(lift {m_weak['acc']-m_null['acc']:+.3f}, need >= {POWER_MIN_LIFT}) "
          f"{'ok' if want else 'FAIL -- probe underpowered, absence claims are void'}")
    er_iso = effective_rank(rng.normal(size=(500, 64)).astype(np.float32))
    er_col = effective_rank((rng.normal(size=(500, 1)) @ rng.normal(size=(1, 64))).astype(np.float32))
    print(f"[selftest] eff_rank isotropic={er_iso:.1f} (want ~64)  collapsed={er_col:.2f} (want ~1)")
    ok &= er_iso > 40 and er_col < 1.5
    print("[selftest]", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# --------------------------------------------------------------------------- __main__
def load_checkpoint(path, arch, arch_kwargs=None):
    sd = torch.load(path, map_location="cpu")
    if arch == "unet":
        net = ComplexUNetMultiTask(**(arch_kwargs or {}))
    elif arch == "cnn":
        net = Embedding(EMBED_DIM, N_FEATURES, **(arch_kwargs or {}))
    else:
        raise ValueError(f"unknown arch {arch!r}")
    net.load_state_dict(sd)
    return net.eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true",
                    help="validate the probe machinery on synthetic latents and exit")
    ap.add_argument("--ckpt", help="state_dict.pt to score")
    ap.add_argument("--arch", choices=["unet", "cnn"])
    ap.add_argument("--condition", default="native")
    ap.add_argument("--device", default="cpu",
                    help="cpu (default). mps only if you know no GPU trial is running.")
    ap.add_argument("--per-profile-cap", type=int, default=None,
                    help="stratified subsample: at most this many captures per profile per "
                         "split. None = all 4197. Extraction is ~0.22 s/capture per model "
                         "on 4 CPU threads for the U-Net.")
    ap.add_argument("--heads", default="proto,linear,mlp")
    ap.add_argument("--label-sets", default=None)
    ap.add_argument("--no-random-control", action="store_true",
                    help="skip CONTROL 1. Only for debugging -- a report without it is not "
                         "evidence of anything.")
    ap.add_argument("--extra-taps", action="store_true",
                    help="also probe bott4 (4-chunk temporal pooling of the bottleneck) -- "
                         "separates 'pooling destroyed it' from 'bottleneck never had it'")
    ap.add_argument("--reference", default=None,
                    help="a previously written latent_transfer.json (e.g. the CNN's) to "
                         "evaluate F5 against. Without it F5 is UNEVALUATED and `transfers` "
                         "cannot be True.")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    if a.selftest:
        raise SystemExit(selftest())
    if not (a.ckpt and a.arch):
        ap.error("--ckpt and --arch are required unless --selftest")
    global EXTRA_TAPS
    EXTRA_TAPS = a.extra_taps
    import pool_cache
    data = pool_cache.load(a.condition)
    data["condition"] = a.condition
    net = load_checkpoint(a.ckpt, a.arch)
    rep = latent_transfer_report(
        net, data, torch.device(a.device),
        tag=os.path.basename(os.path.dirname(a.ckpt)) or a.arch,
        per_profile_cap=a.per_profile_cap,
        heads=tuple(a.heads.split(",")),
        label_sets=a.label_sets.split(",") if a.label_sets else None,
        include_random_control=not a.no_random_control,
        batch=a.batch,
        reference=json.load(open(a.reference)) if a.reference else None)
    out = a.out or os.path.join(os.path.dirname(a.ckpt), "latent_transfer.json")
    json.dump(rep, open(out, "w"), indent=1)
    print(report_markdown(rep))
    open(os.path.splitext(out)[0] + ".md", "w").write(report_markdown(rep) + "\n")
    print(f"[lt] wrote {out}  ({rep['wall_s']}s)")


if __name__ == "__main__":
    main()
