"""Overnight campaign for the transfer-learnable + denoising U-Net.

STRUCTURE IS COPIED FROM run_priority.py ON PURPOSE, and for a reason with a date on it.
On 2026-07-24 an in-process campaign loop stalled at 0% CPU while the machine was in heavy
swap (25.5 of 26.6 GB), never recovered, and sat idle for two days -- three runs out of a
whole weekend. Its guards (disk space, a wall-clock budget) could not see a hang, because
the process was alive and the clock was sensible. The same four structural fixes apply here:

  1. Each trial runs in its OWN subprocess with a hard timeout. A hang costs that trial,
     not the night.
  2. Free memory is checked before each launch and the runner WAITS rather than piling on.
  3. The child appends its result row on completion, so a later crash cannot lose earlier
     work, and the plan is resumable by tag.
  4. Trials are ordered by INFORMATION VALUE, not convenience, because the runner may be
     stopped at any point and an experiment whose control never ran cannot be interpreted.

STRICT SERIALIZATION. subprocess.run blocks, so exactly one trial exists at a time; on top
of that `wait_for_exclusive_gpu()` refuses to launch while any OTHER training process is
alive, and a lockfile prevents two copies of this runner from overlapping. MPS is not
safely shareable on this machine and the campaign's own timings assume it is not shared.

--------------------------------------------------------------------------------------
THE GRID, AND WHY IT IS THIS AND NOT SOMETHING BIGGER
--------------------------------------------------------------------------------------
Every trial is RECON-DRIVEN WITH NO AUX LABEL HEADS (w_prof = w_par = 0). campaign3's own
2x2 at 700 episodes each: recon_only closed 0.6183 / chirpAUROC 0.834 beat full 0.5942 /
0.495 on BOTH axes, so the profile and parameter heads were cancelling the reconstruction
term. They are not in the grid.

The grid is a 2x2 plus two points on the reconstruction weight:

              crop OFF                     crop ON
  arch BASE   t1 (the reference point)     t3
  arch FIXED  t2                           t4  (+ t5 w_rec=3, t6 w_rec=0.3)

  * ARCH BASE is today's network, unchanged. It is in the grid as a CONTROL, even though
    campaign3 already ran something like it, because the measurement stack changed
    underneath (pair-aligned targets, clean-row masking, and a different set of reported
    metrics). Comparing a new arch against an old campaign's number is exactly the
    cross-run confound this codebase keeps getting caught by.
  * ARCH FIXED bundles the encoder audit's top three ranked changes, which the audit
    explicitly says to do together because #1 makes #2 and #3 measurable at all:
      1. magnorm -- the bottleneck is 100.0% exactly zero at init and 97.3% zero after 700
         episodes of training, so the deep half of every U-Net trial in campaign3 was
         switched off before training began. Measured at init: dead fraction 1.000 -> 0.017,
         encoder gradient norms ~8e-2 -> ~1e2, phase equivariance preserved to 1e-6.
      2. skip_dropout=0.5 -- zeroing the ENTIRE bottleneck changes reconstruction coherence
         by exactly 0.0000 on the trained checkpoint. The decoder does not use it.
      3. feat_dropout=0.5 -- a RANDOMLY INITIALISED U-Net with the same 12 handcrafted
         features concatenated into the trunk reaches 0.482 against the trained one's 0.528.
    These are bundled, not swept, because six trials cannot resolve three binary factors and
    a 1-of-3 result would be uninterpretable either way. If FIXED wins, the follow-up
    campaign ablates within it; if it loses, none of the three was the blocker.
  * CROP ON/OFF IS THE EXPLICIT CONTROL. Without the OFF arm there is no way to claim the
    augmentation did anything, and the OFF arm goes through the SAME CropStream code path
    (mode="off") so the two differ in the crop and in nothing else.
  * The two extra trials go on w_rec rather than on more architecture, because w_rec is the
    axis that trades the two halves of the deliverable against each other (a transferable
    latent vs. a denoising output) and nothing measured so far pins it.

Not in the grid, deliberately: antialiased decimation and the ACF head readout (both
implemented in unet_transfer.py). The aliasing finding is MEDIUM confidence -- it measures
energy folded for an unfiltered capture, an upper bound on the damage, not the residual
after the learned 7-tap convs -- and the ACF readout targets a per-class failure (fm -> cw)
that may simply disappear once the encoder is alive. Both are cheap to add to a follow-up
once t2 says whether the encoder fix worked.

--------------------------------------------------------------------------------------
WALL-CLOCK ARITHMETIC (target ~7 h on one Mac)
--------------------------------------------------------------------------------------
Measured, not assumed. c3_unet_recon_only ran 700 episodes in 3151 s of training wall
(results.jsonl) inside a 3310 s child (heartbeat log), so:

    U-Net training, recon-driven, L=16384, batch 70   4.50 s/episode   [measured]
    + online crop stream (1.8 ms/pair x 70)           0.13 s/episode   [measured]
    + pair CFO alignment, one 2^18 FFT per pair       0.22 s/episode   [measured]
    ------------------------------------------------------------------
                                                      4.85 s/episode
    x 700 episodes                                    3395 s
    + evaluate_v2 + open-set + confusion on MPS        160 s  [measured 3310-3151]
    + latent_transfer, cap 20 (4 encoder passes,
      ~2934 rows, plus 3-seed heads)                  ~240 s
    + denoise_eval, 256 val rows, CPU, aligned and
      unaligned targets                                ~60 s
    + canonical probe sweep (19 probes x 4 lengths)     ~5 s
    + encoder-health and bottleneck-ablation probes    ~10 s
    + startup, pool mmap, checkpoint write             ~30 s
    ------------------------------------------------------------------
    per trial                                        ~3900 s = 65 min
    x 6 trials                                      ~23400 s = 6.5 h

That leaves ~30 min of slack in a 7 h window, which is why the grid is six trials and not
seven, and why EPISODES is 700 rather than the 900 campaign3 nominally asked for.

EQUAL EPISODE COUNTS ACROSS ALL SIX CELLS IS DELIBERATE, and so is not spending trials on a
budget sweep. campaign3 gave the U-Net 700 episodes against the CNN's 4000 at the same batch
size and then compared the two, which is a live confound in its conclusions -- but "the gap
is just budget" is already refuted by a control sitting in campaign3's own results.jsonl:
sweep_eq_lambda_eq0.5 ran 700 episodes, identical pool, identical batch of 70, same eval
device, with 89,008 parameters against the U-Net's 807,467, and reached train CE 0.8633 /
closed 0.7798 against the U-Net's 1.2052 / 0.6183 -- and fm 0.763 against 0.052. A 9x
smaller model fits its own training episodes BETTER at identical step count. That is
capacity allocation and architecture, not step count, which is why the trials here buy
architecture rather than episodes. Nothing in this campaign compares across budgets.

TIMEOUT is 95 min per trial (1.46x the estimate). A trial that hits it is killed and the
campaign moves on, with the last heartbeats printed so a wedge is distinguishable from
mere slowness.

--------------------------------------------------------------------------------------
WHAT EVERY ROW RECORDS
--------------------------------------------------------------------------------------
closed_overall / closed_clean / closed_impaired / per-class, chirp+noise AUROC;
latent-transfer accuracy on the bottleneck WITH its random-init control, the feat12
control, and the alternate within-class label set (gsm7 beside ofdm24), each with a paired
bootstrap; denoise coherence on IMPAIRED ROWS ONLY with the identity-passthrough baseline
on the same rows, the per-sample rho/(1+rho) reference, and the fraction of the remaining
gap closed; the low-SNR-bin coherence; and the canonical-probe length sweep's WORST length.
Every one of those has a matched baseline in the same row, because a bare number for any of
them has already been misread once in this project.

Run:  see the nohup + caffeinate line at the bottom of this docstring.
    nohup caffeinate -i .venv-training/bin/python -u \\
        training/zplane_ab/v2_full_variation/run_transfer_campaign.py \\
        > training/zplane_ab/v2_full_variation/artifacts/campaign4/runner.log 2>&1 &
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

_V2 = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(_V2, "artifacts", "campaign4")
RESULTS = os.path.join(ART, "results.jsonl")
LOCK = os.path.join(ART, "runner.lock")
PY = "/Users/johnelliott/PersonalGitHub/Atom-Classifier/.venv-training/bin/python"
# The CNN's latent-transfer report, if it exists, supplies F5 (does the U-Net latent lose to
# the production model on every label set?). Without it F5 is UNEVALUATED and `transfers`
# cannot come back True -- which is the intended conservative behaviour, not a bug.
REFERENCE_LT = os.path.join(_V2, "artifacts", "campaign3", "c3_cnn", "latent_transfer.json")

EPISODES = 700
TIMEOUT_S = 95 * 60
MIN_FREE_PCT = 25
MAX_WAIT_S = 3600          # how long to wait for memory / for another trainer to finish

# --- the two architecture cells -------------------------------------------------------
ARCH_BASE = {}                                    # today's network, unchanged
ARCH_FIXED = dict(magnorm=True, skip_dropout=0.5, feat_dropout=0.5)

CROP_ON = dict(mode="loguniform", guard_frac=0.35)
CROP_OFF = dict(mode="off")

# Ordered by information value; controls first. Stopping after any prefix still leaves
# something interpretable:
#   after 2 -> the architecture effect at crop OFF
#   after 3 -> the crop effect at the BASE architecture
#   after 4 -> the complete 2x2, which is the campaign's main claim
#   after 6 -> the reconstruction-weight axis on the best cell
# REORDERED 04:42 on measured evidence, not on the plan as drafted. t1 (base) came back at
# closed 0.4139 and 0/19 on the canonical probe at EVERY length -- it is simply a bad model.
# t2 (FIXED) came back at 0.6733, the best U-Net in either campaign, and 19/19 at fill 1.0.
# That makes the base arm's crop cell nearly uninformative, while FIXED+crop is the actual
# deliverable: whether length augmentation broadens a model that is otherwise fill-locked at
# the top. The crop effect is attributable within the FIXED arm alone (t2 vs t4), so moving
# t4 ahead of t3 costs no comparison and buys the headline cell ~75 minutes earlier -- which
# matters because the runner may not reach the end of the list before morning.
PLAN = [
    ("t1_base_cropoff",      ARCH_BASE,  CROP_OFF, 1.0),
    ("t2_fixed_cropoff",     ARCH_FIXED, CROP_OFF, 1.0),
    ("t4_fixed_cropon",      ARCH_FIXED, CROP_ON,  1.0),
    ("t5_fixed_cropon_wrec3", ARCH_FIXED, CROP_ON, 3.0),
    ("t3_base_cropon",       ARCH_BASE,  CROP_ON,  1.0),
    ("t6_fixed_cropon_wrec03", ARCH_FIXED, CROP_ON, 0.3),
]


# ======================================================================================
# the trial itself -- imported and called by the child, never run in the parent
# ======================================================================================
def run_trial(tag, arch, crop, w_rec, episodes=EPISODES, condition="native",
              lt_cap=20, denoise_n=256, seed=0, device=None):
    """Train one configuration and score it on everything the deliverable is defined by.

    Returns a JSON-serialisable row. Runs in a child process; the parent never imports the
    heavy stack, which is what keeps a wedged trial from wedging the runner."""
    import numpy as np
    import torch

    for p in (os.path.dirname(os.path.dirname(_V2)), os.path.dirname(_V2), _V2):
        if p not in sys.path:
            sys.path.insert(0, p)

    import pool_cache
    import canonical_probe_net as cpn
    import denoise_eval as de
    from equalizer_frontend import coherence as tcoh
    from evaluate_ab_v2 import evaluate_v2
    from openset_eval import evaluate_openset
    from latent_transfer import latent_transfer_report
    from length_aug import CropConfig
    from train import embed_all, nearest
    from train_transfer import train_transfer
    from unet_transfer import TransferUNet, encoder_health

    DEV = (torch.device(device) if device else
           (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")))
    data = pool_cache.load(condition)
    data["condition"] = condition
    t_start = time.perf_counter()

    net = TransferUNet(**arch)
    n_params = sum(p.numel() for p in net.parameters())
    # Encoder health BEFORE training, on real rows. This is the diagnostic that would have
    # caught the dead-bottleneck defect in campaign3 on day one: a bottleneck that is 100%
    # exactly zero at initialisation cannot be trained, and nothing in the old harness
    # looked. Recorded per trial so it can never regress silently again.
    # A SPREAD of rows, not xtr[:8]. The audit's own census used xtr[:8] and those eight
    # rows happen to leave a handful of bottleneck elements alive at ~3.7e-09, which turns
    # an exactly-zero result into a merely-tiny one -- and exact zeros are the argument that
    # survives Adam, since Adam's per-parameter normalisation nullifies a small gradient but
    # not an absent one. Fixed stride so the sample is deterministic across trials.
    hrows = np.linspace(0, len(data["xtr"]) - 1, 12).astype(int)
    xb0 = torch.from_numpy(np.asarray(data["xtr"][hrows], dtype=np.float32))
    fb0 = torch.from_numpy(np.asarray(data["ftr"][hrows], dtype=np.float32))
    health_init = encoder_health(net, xb0, fb0)

    cfg = CropConfig(**crop)
    net, hist, wall = train_transfer(
        net, data, DEV, episodes=episodes, eval_every=max(100, episodes // 6),
        log_tag=tag, log_every=max(50, episodes // 14), w_rec=w_rec, crop_cfg=cfg,
        align_pair=True, mask_clean_recon=True, seed=seed, condition=condition)

    # ---- closed set / open set, scored on the SAME device for every trial in the grid ---
    # campaign3 scored the U-Net rows on MPS and its CNN baselines on CPU, and then compared
    # them; here every cell is scored on DEV, so within-campaign ranking is device-clean.
    EVAL_DEV = DEV
    net = net.to(EVAL_DEV)
    rep, protos = evaluate_v2(net, EVAL_DEV, data)
    osr = evaluate_openset(net, EVAL_DEV, data, protos, n_each=300)
    emb = embed_all(net, data["xva"], data["fva"], EVAL_DEV)
    pred, _ = nearest(emb, protos)
    n = data["n_classes"]
    cm = np.zeros((n, n), dtype=int)
    for t, p_ in zip(data["yva"], pred):
        cm[t, p_] += 1

    health_final = encoder_health(net.to("cpu"), xb0, fb0)

    # ---- canonical probe length sweep (worst length is what is gated) -------------------
    probe = cpn.sweep(net, protos, list(data["classes"]), torch.device("cpu"),
                      condition=condition, in_len=int(data["input_length"]))

    # ---- latent transfer, with every control -------------------------------------------
    ref = json.load(open(REFERENCE_LT)) if os.path.exists(REFERENCE_LT) else None
    lt = latent_transfer_report(net.to(EVAL_DEV), data, EVAL_DEV, tag=tag,
                                per_profile_cap=lt_cap, reference=ref, verbose=True)
    net = net.to("cpu")

    od = os.path.join(ART, tag)
    os.makedirs(od, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(od, "state_dict.pt"))
    np.save(os.path.join(od, "prototypes.npy"), protos)
    json.dump(lt, open(os.path.join(od, "latent_transfer.json"), "w"), indent=1)

    # ---- denoising, on HELD-OUT val rows, impaired rows only ----------------------------
    # Scored against BOTH targets. The model trained against the ALIGNED target, so that is
    # the honest number for it; the UNALIGNED one is kept because aligning the pair narrows
    # the task (a global derotation is outside a conv stack's function class) and that
    # choice must stay auditable rather than buried in a preprocessing step.
    dn = {}
    imp, cln_unaligned, meta = de.load_eval_rows("val", denoise_n, condition, seed=seed,
                                                 align_target=False)
    # ONE forward pass, two targets. The rows are identical between the two scorings -- only
    # the target differs -- so re-running the model for each would just burn CPU.
    recon = de.run_model(os.path.join(od, "state_dict.pt"), imp, meta["feats"],
                         arch=arch) if w_rec > 0 else None
    cln_aligned = np.stack([de.derotate_onto(cln_unaligned[j], imp[j]).astype(np.complex128)
                            for j in range(len(cln_unaligned))])
    for name, cln in (("aligned", cln_aligned), ("unaligned", cln_unaligned)):
        res = de.score_denoising(cln, imp, recon, meta["snr_db"], meta["frac_bw"],
                                 meta["impaired"], with_oracle_fir=True)
        a = res["agg"]
        dn[name] = {
            "n_impaired": res["n_impaired"],
            "model": a.get("model", {}).get("impaired_mean"),
            "passthrough": a["passthrough"]["impaired_mean"],
            "delta_vs_passthrough": a.get("model", {}).get("delta_vs_passthrough"),
            "win_rate": a.get("model", {}).get("win_rate_vs_passthrough"),
            "frac_gap_closed": a.get("model", {}).get("frac_gap_closed"),
            "snr_expectation_rho_over_1plusrho":
                float(np.mean(res["snr_expectation"][res["impaired_mask"]])),
            "oracle_fir_65": a.get("oracle_fir_65", {}).get("impaired_mean"),
            "best_lowpass": max((a[k]["impaired_mean"] for k in a if k.startswith("lowpass")),
                                default=None),
            "coh_recon_vs_input": (float(res["coh_recon_vs_input"][res["impaired_mask"]].mean())
                                   if "coh_recon_vs_input" in res else None),
            "low_snr": res.get("low_snr"),
            "by_snr": res.get("by_snr"),
            "high_snr_passthrough": res.get("high_snr_passthrough"),
            "verdict": de.verdict(res),
        }
    # Is the bottleneck load-bearing for the DECODER? On c3_unet_recon_only, zeroing the
    # entire bottleneck changed reconstruction coherence by EXACTLY 0.0000 -- the deployed
    # "U-Net" was a full-resolution FIR filter with an ornamental bottleneck. Standing
    # per-trial metric, because that is precisely the kind of thing that regresses silently.
    if w_rec > 0:
        k = min(24, len(imp))
        xb = torch.from_numpy(np.stack([imp[:k].real, imp[:k].imag], 1).astype(np.float32))
        fb = torch.from_numpy(np.asarray(meta["feats"][:k], dtype=np.float32))
        cb = torch.from_numpy(cln_aligned[:k].astype(np.complex64))
        with torch.no_grad():
            net.eval()
            net(xb, fb)
            coh_full = float(tcoh(net.last_recon, cb).mean())
            v, skips = net.encode(xb)
            u = torch.zeros_like(v)
            for blk, skip in zip(net.ups, reversed(skips)):
                u = net._up(u, skip.shape[-1])
                u = blk(torch.cat([u, skip], dim=1))
            coh_zero = float(tcoh(net.out(u).squeeze(1), cb).mean())
        dn["bottleneck_ablation"] = {
            "n": k, "coh_full": coh_full, "coh_bottleneck_zeroed": coh_zero,
            "delta": coh_zero - coh_full,
            "note": "delta ~ 0.0000 means the decoder ignores the bottleneck entirely; "
                    "that was exactly the case for c3_unet_recon_only."}

    v = lt.get("verdict", {})
    ps = v.get("per_set", {})
    row = dict(
        tag=tag, arch=arch, arch_effective=net.config(), crop=crop, w_rec=w_rec,
        episodes=episodes, condition=condition, param_count=n_params,
        pair_align="cfo_derotate_2^18", mask_clean_recon=True,
        aux_heads="none (w_prof=w_par=0)",
        wall_s=round(wall, 1), total_s=round(time.perf_counter() - t_start, 1),
        eval_device=str(EVAL_DEV),
        # --- closed set
        closed_overall=round(rep["closed_overall"], 4),
        closed_clean=round(rep["closed_clean"], 4),
        closed_impaired=round(rep["closed_impaired"], 4),
        per_class_clean=rep["per_class_clean"],
        auroc_chirp_HELDOUT=osr["auroc_chirp"], auroc_noise=osr["auroc_noise"],
        auroc_overall=osr["auroc_overall"],
        confusion=cm.tolist(), classes=list(data["classes"]),
        # --- latent transfer, always beside its controls
        latent_transfer=dict(
            transfers=v.get("transfers"),
            criteria=v.get("criteria"), unevaluated=v.get("criteria_unevaluated"),
            bott_eff_rank=(v.get("eff_rank") or {}).get("bott"),
            bott_eff_rank_random_init=(v.get("eff_rank_random_init") or {}).get("bott"),
            ofdm24=ps.get("ofdm24"), gsm7=ps.get("gsm7"), class7=ps.get("class7"),
            profile37_vs_class_prior=v.get("profile37_vs_class_prior"),
            novel5=v.get("novel5_trained"), novel5_random_init=v.get("novel5_random_init"),
            power_control=v.get("power_control"),
            json="latent_transfer.json"),
        # --- denoising, always beside passthrough
        denoise=dn,
        # --- length invariance
        canonical_probe=dict(
            worst_correct=probe["worst_correct"], worst_bal=probe["worst_bal"],
            best_correct=probe["best_correct"], worst_cos=probe["worst_cos"],
            majority_baseline=probe["majority_baseline"],
            passes_gate=probe["passes_gate"], rows=probe["rows"]),
        # --- did the encoder ever wake up?
        encoder_health_init=health_init, encoder_health_final=health_final,
        train_history={k: hist[k] for k in
                       ("loss_ce", "coherence_impaired", "passthrough_impaired",
                        "val_acc", "crop_len_median", "crop_len_p10", "best_val_acc")},
    )
    return row


# ======================================================================================
# the supervisor
# ======================================================================================
def free_pct() -> int:
    try:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True,
                             timeout=20).stdout
        for line in out.splitlines():
            if "free percentage" in line:
                return int(line.strip().split(":")[1].strip().rstrip("%"))
    except Exception:
        pass
    return 100


def other_trainers() -> list[str]:
    """Any OTHER training process alive? The campaign's timings assume sole ownership of
    MPS, and this machine has hung under memory pressure with two large jobs resident."""
    try:
        out = subprocess.run(["ps", "-Ao", "pid=,command="], capture_output=True, text=True,
                             timeout=20).stdout
    except Exception:
        return []
    me = str(os.getpid())
    hits = []
    for line in out.splitlines():
        pid, _, cmd = line.strip().partition(" ")
        if pid in (me, str(os.getppid())):
            continue
        if any(k in cmd for k in ("campaign3", "run_priority", "train_multitask",
                                  "train_equalizer", "campaign_runner")):
            hits.append(f"{pid} {cmd[:90]}")
    return hits


def wait_for_exclusive_gpu(max_wait=MAX_WAIT_S):
    waited = 0
    while waited < max_wait:
        busy = other_trainers()
        if not busy:
            return True
        print(f"[tc] waiting for exclusive GPU: {len(busy)} other trainer(s) alive", flush=True)
        for b in busy[:3]:
            print(f"[tc]    {b}", flush=True)
        time.sleep(120); waited += 120
    print(f"[tc] *** still not exclusive after {max_wait}s -- proceeding anyway, timings "
          f"and memory headroom will be off ***", flush=True)
    return False


def done_tags():
    if not os.path.exists(RESULTS):
        return set()
    return {json.loads(l)["tag"] for l in open(RESULTS) if l.strip()}


CHILD = '''
import sys, json, os, threading, time
sys.path.insert(0, {v2!r})
import torch
import run_transfer_campaign as rc

def _heartbeat():
    """Every 30 s: wall time and MPS allocation. If a trial wedges, this is what says
    whether the GPU allocation is still moving (slow) or frozen (wedged) -- the distinction
    the 2026-07-24 stall could not be diagnosed without."""
    t0 = time.time()
    while True:
        try:
            a = torch.mps.current_allocated_memory()/1e9 if torch.backends.mps.is_available() else -1
        except Exception:
            a = -2
        print(f"HEARTBEAT t={{time.time()-t0:.0f}}s mps_alloc={{a:.3f}}GB", flush=True)
        time.sleep(30)

threading.Thread(target=_heartbeat, daemon=True).start()
r = rc.run_trial({tag!r}, {arch!r}, {crop!r}, {w_rec!r}, episodes={ep!r})
with open({res!r}, "a") as f:
    f.write(json.dumps(r) + "\\n")
lt = r["latent_transfer"]; dn = r["denoise"]["aligned"]
print("CHILD_OK", r["tag"],
      "closed", round(r["closed_overall"], 4),
      "| probe worst", r["canonical_probe"]["worst_correct"],
      "| bott effrank", lt["bott_eff_rank"], "vs rand", lt["bott_eff_rank_random_init"],
      "| coh", dn["model"], "vs passthrough", dn["passthrough"], flush=True)
'''


def write_report():
    rows = [json.loads(l) for l in open(RESULTS)] if os.path.exists(RESULTS) else []
    L = ["# campaign 4 -- transfer-learnable + denoising U-Net\n",
         f"{len(rows)} trials | {EPISODES} episodes each | recon-driven, no aux heads\n",
         "\nEvery column is printed beside its baseline. A bare coherence, a bare latent",
         "accuracy and a bare probe score have each already been misread once in this",
         "project; none of them means anything alone.\n",
         "\n| tag | arch | crop | w_rec | closed | clean | chirp | probe worst (maj 0.79) |"
         " bott effrank (rand) | ofdm24 acc (rand / feat12) | coh imp (passthru / oracleFIR) |"
         " lowSNR d | transfers |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    def num(x, fmt="{:.4f}", plus=False):
        if x is None:
            return "-"
        return ("{:+.4f}" if plus else fmt).format(x)

    for r in rows:
        lt, dn = r["latent_transfer"], r["denoise"]["aligned"]
        o = lt.get("ofdm24") or {}
        vr = (o.get("vs_random_init") or {}).get("delta")
        vf = (o.get("vs_feat12") or {}).get("delta")
        ls = (dn.get("low_snr") or {}).get("delta")
        L.append(" | ".join([
            "", r["tag"], "FIXED" if r["arch"] else "base", r["crop"]["mode"], str(r["w_rec"]),
            num(r["closed_overall"]), num(r["closed_clean"]),
            num(r["auroc_chirp_HELDOUT"], "{:.3f}"),
            num(r["canonical_probe"]["worst_correct"], "{:.3f}"),
            f"{lt.get('bott_eff_rank')} ({lt.get('bott_eff_rank_random_init')})",
            f"{num(o.get('acc'))} ({num(vr, plus=True)} / {num(vf, plus=True)})",
            f"{num(dn.get('model'))} ({num(dn.get('passthrough'))} / "
            f"{num(dn.get('oracle_fir_65'))})",
            num(ls, plus=True), str(lt.get("transfers")), ""]))
    open(os.path.join(ART, "REPORT.md"), "w").write("\n".join(L) + "\n")


def main():
    os.makedirs(ART, exist_ok=True)
    if os.path.exists(LOCK):
        try:
            pid = int(open(LOCK).read().strip())
            os.kill(pid, 0)
            print(f"[tc] another runner is alive (pid {pid}); refusing to start a second. "
                  f"Delete {LOCK} if that pid is stale.", flush=True)
            return 1
        except (ValueError, ProcessLookupError, PermissionError):
            print(f"[tc] stale lock at {LOCK}, taking it over", flush=True)
    open(LOCK, "w").write(str(os.getpid()))
    try:
        print(f"[tc] start {time.strftime('%Y-%m-%d %H:%M:%S')} free_mem={free_pct()}% "
              f"episodes={EPISODES} trials={len(PLAN)} "
              f"budget~{len(PLAN) * 65 / 60:.1f}h", flush=True)
        wait_for_exclusive_gpu()
        for tag, arch, crop, w_rec in PLAN:
            if tag in done_tags():
                print(f"[tc] skip {tag} (already in results.jsonl)", flush=True)
                continue
            waited = 0
            while free_pct() < MIN_FREE_PCT and waited < MAX_WAIT_S:
                print(f"[tc] memory {free_pct()}% < {MIN_FREE_PCT}%, waiting", flush=True)
                time.sleep(60); waited += 60
            code = CHILD.format(v2=_V2, tag=tag, arch=arch, crop=crop, w_rec=w_rec,
                                ep=EPISODES, res=RESULTS)
            print(f"[tc] === {tag} arch={'FIXED' if arch else 'base'} crop={crop['mode']} "
                  f"w_rec={w_rec} === {time.strftime('%H:%M:%S')} free={free_pct()}%",
                  flush=True)
            t0 = time.time()
            trial_log = os.path.join(ART, f"_{tag}.log")
            try:
                with open(trial_log, "w") as lf:
                    p = subprocess.run([PY, "-u", "-c", code], timeout=TIMEOUT_S,
                                       stdout=lf, stderr=subprocess.STDOUT, text=True)
                tail = "\n".join(open(trial_log).read().strip().splitlines()[-4:])
                print(f"[tc] {tag} rc={p.returncode} {time.time()-t0:.0f}s "
                      f"(log {trial_log})\n{tail}", flush=True)
            except subprocess.TimeoutExpired:
                beats = [l for l in open(trial_log).read().splitlines()
                         if l.startswith("HEARTBEAT")]
                print(f"[tc] *** {tag} TIMED OUT after {TIMEOUT_S}s -- killed, moving on ***",
                      flush=True)
                print(f"[tc] last 3 heartbeats: "
                      f"{beats[-3:] if beats else 'NONE (died before first beat)'}", flush=True)
            except Exception as e:
                print(f"[tc] *** {tag} FAILED: {e} ***", flush=True)
            try:
                write_report()
            except Exception as e:
                print(f"[tc] (report write failed: {e})", flush=True)
        print(f"[tc] COMPLETE {time.strftime('%H:%M:%S')} -- {len(done_tags())} trials in "
              f"{RESULTS}", flush=True)
        return 0
    finally:
        try:
            os.remove(LOCK)
        except OSError:
            pass


def smoke():
    """End-to-end shape check of ONE trial: 2 episodes, a deliberately tiny net, CPU only.

    This exists because a 6.5 h campaign that dies in the reporting code at minute 62 of
    trial 1 is the expensive failure. It exercises every stage the real trial does --
    training loop, evaluate_v2, open-set, canonical probe, latent transfer with its
    controls, denoise scoring on both targets, the bottleneck ablation, and the row/report
    assembly -- and it never allocates on MPS, so it is safe to run while a GPU trial owns
    the device."""
    import torch
    torch.set_num_threads(2)
    t0 = time.time()
    r = run_trial("smoke", dict(base=2, depth=2, taps=3, magnorm=True, skip_dropout=0.5,
                                feat_dropout=0.5),
                  dict(mode="loguniform", guard_frac=0.35), 1.0,
                  episodes=2, lt_cap=2, denoise_n=32, device="cpu")
    os.makedirs(ART, exist_ok=True)
    json.dump(r, open(os.path.join(ART, "_smoke_row.json"), "w"), indent=1, default=str)
    keys = ("closed_overall", "canonical_probe", "latent_transfer", "denoise",
            "encoder_health_init", "encoder_health_final")
    missing = [k for k in keys if k not in r]
    print(f"[smoke] {time.time()-t0:.0f}s  missing={missing or 'none'}")
    print(f"[smoke] closed {r['closed_overall']}  probe worst {r['canonical_probe']['worst_correct']}"
          f"  bott dead init {r['encoder_health_init']['bottleneck_dead_frac']:.3f}"
          f" -> final {r['encoder_health_final']['bottleneck_dead_frac']:.3f}")
    d = r["denoise"]["aligned"]
    print(f"[smoke] denoise aligned: model {d['model']} passthrough {d['passthrough']} "
          f"oracleFIR {d['oracle_fir_65']}")
    print(f"[smoke] denoise unaligned passthrough {r['denoise']['unaligned']['passthrough']}")
    print(f"[smoke] bottleneck ablation {r['denoise']['bottleneck_ablation']}")
    print(f"[smoke] latent transfers={r['latent_transfer']['transfers']} "
          f"unevaluated={r['latent_transfer']['unevaluated']}")
    return 0 if not missing else 1


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        raise SystemExit(smoke())
    raise SystemExit(main())
