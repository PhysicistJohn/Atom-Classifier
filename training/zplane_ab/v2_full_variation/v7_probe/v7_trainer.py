"""v7 production trainer.

Architecture
------------
Spectro-temporal U-Net-style encoder over an STFT stack, with a structured
head set on the bottleneck:

  * prototype embedding  -- episodic classification (the head every model since
    v3 has used; keeps enrollment/open-set machinery viable)
  * reconstruction decoder -- masked+denoising spectrogram reconstruction
    against the CLEAN pair. Silence is a first-class training target: the
    correct output for an empty window is the noise floor, so the silence
    statistics of bursty classes are learned rather than excluded.
  * confidence head -- predicts whether the episodic prediction is correct;
    trained with BCE against realized correctness. This is the dwell-policy
    signal (classify at short dwell when confident, escalate otherwise).

Losses
------
  total = episodic_CE
        + recon_weight * masked_denoise_reconstruction_L1   (vs clean pair)
        + worst_dur_weight(ramp) * worst-case-over-duration episodic CE
        + conf_weight * confidence_BCE

The worst-duration term mirrors the v6 worst-length auxiliary validated at
+0.055 on the v5 contract: each auxiliary draw evaluates the SAME rows at all
three dwells and takes the max CE over dwells before averaging over rows.

Evaluation
----------
Per-dwell (1/2.5/10 ms) per-class recall and balanced accuracy, plus the
min-over-(profile x dwell) recall cells at n>=64 -- the production analogue of
the v5 worst-cell terms, finally readable because the cells are big enough.
Confidence head is scored by escalation curve: accuracy among the fraction the
model would answer at 1 ms.

The eval PROTOCOL -- which rows the mid-run passes score and where inside each
capture the window starts (queries AND prototypes) -- is frozen once at startup
by build_eval_plan() on a dedicated RNG stream, so it is identical across every
checkpoint of a run and across both trainers, and it no longer aliases the
burst timelines of the bursty profiles.  See build_eval_plan() below and the
--eval-offset-mode / --eval-subsample-mode / --eval-offset-seed flags.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
import mmap as _mmap
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
CORPUS = Path(os.environ.get(
    "CORPUS_DIR",
    HERE.parent.parent.parent / "artifacts/longdwell-production-corpus",
))

TARGET_FS = 20_000_000
DWELLS = {"1ms": 20_000, "2.5ms": 50_000, "10ms": 200_000}
NFFT = 64
HOP = 32
CLASSES = ("am", "bluetooth", "cw", "dsss", "fm", "gsm", "ofdm")


# --- BEGIN SHARED EVAL PROTOCOL (byte-identical in v7_trainer.py) ----------
# The two trainers carry this block verbatim; the guard is
#   diff <(sed -n '/BEGIN SHARED EVAL PROTOCOL/,/END SHARED EVAL PROTOCOL/p' \
#            v7_trainer.py) \
#        <(sed -n '/BEGIN SHARED EVAL PROTOCOL/,/END SHARED EVAL PROTOCOL/p' \
#            v7_trainer_mlx.py)
# which must print nothing.
# Why this exists (measured: artifacts/mlx-g2/subsample_bias_test.json).  Eval
# rows inside a profile block are a fixed-stride slide across ONE long capture
# (GSM: startSampleIndex = 26000*j) and the GSM burst timeline has period
# 78000 = 3*26000, so the old mid-run subsample eval_rows[::3] locked onto a
# SINGLE burst phase -- GSM 1 ms recall 0.9955 on [::3] vs 0.4390 on the full
# split.  Worse: because every eval window started at capture offset 0, even
# the FULL split only ever sampled THREE discrete burst phases.  The protocol
# was measuring a phase, not a distribution.  Training was never affected (it
# draws a fresh random offset per row per episode) -- this is a measurement
# bug, not a learning bug.
#
# The fix, drawn ONCE at startup from a dedicated generator so every
# checkpoint of a run -- and both trainers -- score exactly the same thing:
#   * mid-run subsample: a seeded random subset, same size as [::3]
#   * eval windows: a per-row seeded random start offset (row -> offset
#     table) for query rows AND for prototype rows
# Nothing here touches the training streams (rng, rng2), so episode
# composition stays bit-identical to a pre-fix run.
EVAL_STREAM_KEY = 0xE7A1
EVAL_SUBSAMPLE_DIV = 3


def build_eval_plan(eval_seed, eval_rows, all_train, row_samples, max_dwell,
                    offset_mode="random", subsample_mode="random"):
    """Freeze the eval protocol at startup.

    Returns (mid_rows, eval_offsets, proto_offsets, meta); the two offset
    maps are row_id -> start sample.

    Offsets are drawn in [0, row_samples - max_dwell] so ONE offset per row
    is valid at EVERY dwell and the windows NEST: 2.5 ms starts at the same
    sample as 1 ms and strictly extends it, 10 ms extends 2.5 ms.  The
    escalation semantics -- a longer dwell sees a superset of what the
    shorter one saw -- are therefore preserved exactly as under offset 0,
    while burst phase becomes continuously sampled instead of pinned to 3
    discrete values.

    Prototypes get the same treatment (their own offsets from the same
    generator and the same range, drawn after the query offsets): a
    prototype built at one fixed capture offset is a phase-locked class
    centroid, i.e. the identical bias sitting on the reference side of the
    comparison.  Fixing only the query side would leave half the aliasing in
    place.

    Draw order is part of the contract and is deliberately unconditional --
    subsample indices, then eval-row offsets in split order, then
    prototype-row offsets in all_train order -- so the stream position never
    depends on the modes, and the random subsample is therefore the SAME set
    of rows whether offsets are random or zero.
    """
    if row_samples < max_dwell:
        raise SystemExit(f"row_samples {row_samples} < max dwell {max_dwell}")
    erng = np.random.default_rng([int(eval_seed), EVAL_STREAM_KEY])
    n_eval = len(eval_rows)
    n_sub = -(-n_eval // EVAL_SUBSAMPLE_DIV)      # == len(eval_rows[::3])
    hi = row_samples - max_dwell                  # inclusive upper bound
    rand_sub = np.sort(erng.choice(n_eval, size=n_sub, replace=False))
    rand_eval_off = erng.integers(0, hi + 1, size=n_eval)
    rand_proto_off = erng.integers(0, hi + 1, size=len(all_train))

    sub_idx = (np.arange(0, n_eval, EVAL_SUBSAMPLE_DIV)
               if subsample_mode == "stride" else rand_sub)
    mid_rows = [eval_rows[int(i)] for i in sub_idx]
    if offset_mode == "zero":
        eval_offsets = {r["row"]: 0 for r in eval_rows}
        proto_offsets = {int(rid): 0 for rid in all_train}
    else:
        eval_offsets = {r["row"]: int(o)
                        for r, o in zip(eval_rows, rand_eval_off)}
        proto_offsets = {int(rid): int(o)
                         for rid, o in zip(all_train, rand_proto_off)}
    eo = np.asarray([eval_offsets[r["row"]] for r in eval_rows],
                    dtype=np.int64)
    po = np.asarray([proto_offsets[int(r)] for r in all_train],
                    dtype=np.int64)
    h = hashlib.sha256()
    h.update(f"{offset_mode}|{subsample_mode}|{hi}|".encode())
    h.update(np.asarray(sub_idx, dtype=np.int64).tobytes())
    h.update(eo.tobytes())
    h.update(po.tobytes())
    meta = {
        "eval_seed": int(eval_seed),
        "stream_key": EVAL_STREAM_KEY,
        "offset_mode": offset_mode,
        "subsample_mode": subsample_mode,
        "offset_range": [0, int(hi)],
        "max_dwell": int(max_dwell),
        "n_eval_rows": n_eval,
        "n_mid_rows": len(mid_rows),
        "n_proto_rows": len(all_train),
        "eval_offset_mean": float(eo.mean()) if n_eval else 0.0,
        "eval_offset_min": int(eo.min()) if n_eval else 0,
        "eval_offset_max": int(eo.max()) if n_eval else 0,
        "plan_sha256": h.hexdigest(),
    }
    return mid_rows, eval_offsets, proto_offsets, meta
# --- END SHARED EVAL PROTOCOL ----------------------------------------------


def spectrogram(x: torch.Tensor) -> torch.Tensor:
    frames = x.unfold(-1, NFFT, HOP)
    window = torch.hann_window(NFFT, device=x.device, dtype=torch.float32)
    spec = torch.fft.fft(frames * window, dim=-1)[..., : NFFT // 2 + 1]
    return torch.stack((spec.real, spec.imag, torch.log1p(spec.abs())), dim=1)


def rms_normalize(x: torch.Tensor) -> torch.Tensor:
    rms = x.abs().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-9)
    return x / rms


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=1)
        self.norm = nn.GroupNorm(8, cout)

    def forward(self, x):
        return F.silu(self.norm(self.conv(x)))


class UpBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm = nn.GroupNorm(8, cout)

    def forward(self, x, size):
        x = F.interpolate(x, size=size, mode="nearest")
        return F.silu(self.norm(self.conv(x)))


class V7Net(nn.Module):
    """Encoder-decoder with structured latent heads."""

    def __init__(self, width: int = 48, embed_dim: int = 128):
        super().__init__()
        w = width
        self.stem = Block(3, w, (1, 1))
        self.d1 = Block(w, w * 2, (2, 2))
        self.d2 = Block(w * 2, w * 2, (2, 1))
        self.d3 = Block(w * 2, w * 4, (2, 2))
        self.d4 = Block(w * 4, w * 4, (2, 1))
        self.d5 = Block(w * 4, w * 4, (2, 2))
        self.embed = nn.Linear(w * 4 * 2, embed_dim)
        self.conf = nn.Sequential(
            nn.Linear(w * 4 * 2, 64), nn.SiLU(), nn.Linear(64, 1))
        # light decoder: three ups from the bottleneck back to the spec grid
        self.u1 = UpBlock(w * 4, w * 4)
        self.u2 = UpBlock(w * 4, w * 2)
        self.u3 = UpBlock(w * 2, w)
        self.out = nn.Conv2d(w, 3, 3, padding=1)

    def encode(self, spec):
        h = self.stem(spec)
        h = self.d1(h); h = self.d2(h); h = self.d3(h)
        h = self.d4(h); h = self.d5(h)
        pooled = torch.cat((h.mean(dim=(2, 3)), h.amax(dim=(2, 3))), dim=1)
        z = F.normalize(self.embed(pooled), dim=-1) * 4.0
        return z, self.conf(pooled).squeeze(-1), h

    def reconstruct(self, bottleneck, spec_shape):
        _, _, t, f = spec_shape
        h = self.u1(bottleneck, (max(1, t // 8), max(1, f // 4)))
        h = self.u2(h, (max(1, t // 4), max(1, f // 2)))
        h = self.u3(h, (t, f))
        return self.out(h)


def worst_duration_ce(net, rows_np, targets, log_scale, protos, device,
                      dwells=(20_000, 50_000, 200_000)):
    """Max episodic CE over dwells per row, mean over rows. Prototypes and
    logit scale detached -- same discipline as the v6 auxiliary."""
    losses = []
    for length in dwells:
        x = rms_normalize(torch.from_numpy(rows_np[:, :length]).to(device))
        z, _, _ = net.encode(spectrogram(x))
        logits = -torch.cdist(z, protos.detach()) ** 2 \
            * log_scale.detach().exp().clamp(1e-3, 100)
        losses.append(F.cross_entropy(logits, targets, reduction="none"))
    return torch.stack(losses, dim=1).max(dim=1).values.mean()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=6000)
    ap.add_argument("--eval-every", type=int, default=1500)
    ap.add_argument("--k-shot", type=int, default=5)
    ap.add_argument("--q-query", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--recon-weight", type=float, default=0.3)
    ap.add_argument("--mask-frac", type=float, default=0.5)
    ap.add_argument("--worst-dur-weight", type=float, default=0.15)
    ap.add_argument("--worst-dur-ramp-frac", type=float, default=0.1)
    ap.add_argument("--worst-dur-rows", type=int, default=14)
    ap.add_argument("--conf-weight", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20260740)
    ap.add_argument("--out", default=str(HERE / "v7_result.json"))
    ap.add_argument("--save-checkpoint", default=None)
    ap.add_argument("--init-checkpoint", default=None,
                    help="warm-start weights/log_scale (wedge recovery)")
    ap.add_argument("--start-episode", type=int, default=0)
    ap.add_argument("--eval-offset-mode", choices=("zero", "random"),
                    default="random",
                    help="eval window start offsets: 'random' (default) "
                         "draws a per-row offset ONCE at startup from the "
                         "dedicated eval stream in [0, rowSamples-maxDwell] "
                         "(nested across dwells, applied to query AND "
                         "prototype rows); 'zero' restores the legacy "
                         "offset-0 protocol, which samples only 3 discrete "
                         "burst phases -- kept for comparison only")
    ap.add_argument("--eval-subsample-mode", choices=("stride", "random"),
                    default="random",
                    help="mid-run eval subsample: 'random' (default) is a "
                         "seeded random 1/3 subset fixed at startup; "
                         "'stride' restores the legacy eval_rows[::3], "
                         "which aliases the GSM burst period -- comparison "
                         "only")
    ap.add_argument("--eval-offset-seed", type=int, default=-1,
                    help="seed for the dedicated eval-protocol stream "
                         "(-1 = use --seed); vary it to measure the "
                         "protocol's own sampling uncertainty")
    cli = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # resumed runs jump the stream so episodes don't repeat the consumed draws
    rng = (np.random.default_rng([cli.seed, cli.start_episode])
           if cli.start_episode else np.random.default_rng(cli.seed))
    torch.manual_seed(cli.seed + cli.start_episode)

    manifest = json.loads((CORPUS / "manifest.json").read_text())
    noisy = np.load(CORPUS / "noisy.npy", mmap_mode="r")
    clean = np.load(CORPUS / "clean.npy", mmap_mode="r")
    row_samples = manifest["rowSamples"]
    rows = manifest["rows"]
    train_by_class: dict[str, list[int]] = {c: [] for c in CLASSES}
    class_index = {c: i for i, c in enumerate(CLASSES)}
    eval_rows: list[dict] = []
    for row in rows:
        if row["role"] == "train":
            train_by_class[row["cls"]].append(row["row"])
        else:
            eval_rows.append(row)
    row_cls = {r["row"]: r["cls"] for r in rows}
    all_train = [r for ids in train_by_class.values() for r in ids]
    print(f"train {len(all_train)}  eval {len(eval_rows)}  "
          f"profiles {len({r['profile'] for r in rows})}", flush=True)

    # eval protocol frozen ONCE, here, on its own stream (see build_eval_plan)
    eval_seed = cli.seed if cli.eval_offset_seed < 0 else cli.eval_offset_seed
    mid_eval_rows, eval_offsets, proto_offsets, eval_plan = build_eval_plan(
        eval_seed, eval_rows, all_train, row_samples, max(DWELLS.values()),
        offset_mode=cli.eval_offset_mode,
        subsample_mode=cli.eval_subsample_mode)
    print(f"eval plan: offsets={cli.eval_offset_mode} "
          f"subsample={cli.eval_subsample_mode} seed={eval_seed} "
          f"mid-rows={len(mid_eval_rows)}/{len(eval_rows)} "
          f"offset range {eval_plan['offset_range']} "
          f"(mean {eval_plan['eval_offset_mean']:.0f}) "
          f"sha {eval_plan['plan_sha256'][:16]}...", flush=True)

    # RAM cache: training rows as float16 I/Q planes. Two concurrent arms on
    # the 56 GB memmaps thrash the page cache into ~15% CPU utilisation; the
    # train split at half precision is ~11 GB and fits in memory. Episodes
    # then never touch disk; eval still streams from the memmaps.
    t_cache = time.perf_counter()
    train_pos = {rid: slot for slot, rid in enumerate(all_train)}
    noisy_ram = np.empty((len(all_train), 2, row_samples), dtype=np.float16)
    clean_ram = np.empty((len(all_train), 2, 50_000), dtype=np.float16)
    for slot, rid in enumerate(all_train):
        nrow = np.asarray(noisy[rid])
        noisy_ram[slot, 0] = nrow.real.astype(np.float16)
        noisy_ram[slot, 1] = nrow.imag.astype(np.float16)
        crow = np.asarray(clean[rid, :50_000])
        clean_ram[slot, 0] = crow.real.astype(np.float16)
        clean_ram[slot, 1] = crow.imag.astype(np.float16)
    print(f"RAM cache loaded: "
          f"{(noisy_ram.nbytes + clean_ram.nbytes)/1e9:.1f} GB in "
          f"{time.perf_counter()-t_cache:.0f}s", flush=True)

    # Pin the cache: the kernel compressed it out from under three runs
    # while ~20 GB of one-shot memmap reads polluted the page cache. 48 GB
    # machine — 10 GB wired is the intended trade.
    _libc = ctypes.CDLL(None, use_errno=True)
    for _arr in (noisy_ram, clean_ram):
        if _libc.mlock(ctypes.c_void_p(_arr.ctypes.data),
                       ctypes.c_size_t(_arr.nbytes)):
            print(f"mlock failed (errno {ctypes.get_errno()}) — "
                  "cache stays evictable", flush=True)
            break
    else:
        print("RAM cache pinned (mlock)", flush=True)

    def drop_memmap_pages() -> None:
        """Purge one-shot corpus reads from the page cache so they never
        compete with the pinned episode cache."""
        for _m in (noisy, clean):
            _mm = getattr(_m, "_mmap", None)
            if _mm is not None:
                try:
                    _mm.madvise(_mmap.MADV_DONTNEED)
                except (AttributeError, OSError, ValueError):
                    return

    drop_memmap_pages()

    def train_window(rid: int, offset: int, length: int) -> np.ndarray:
        slot = train_pos[rid]
        seg = noisy_ram[slot, :, offset : offset + length].astype(np.float32)
        return (seg[0] + 1j * seg[1]).astype(np.complex64)

    def train_clean(rid: int, offset: int, length: int) -> np.ndarray:
        slot = train_pos[rid]
        seg = clean_ram[slot, :, : length].astype(np.float32)
        return (seg[0] + 1j * seg[1]).astype(np.complex64)

    net = V7Net(width=cli.width, embed_dim=cli.embed_dim).to(device)
    print(f"params: {sum(p.numel() for p in net.parameters()):,}", flush=True)
    log_scale = nn.Parameter(torch.tensor(math.log(10.0), device=device))
    opt = torch.optim.AdamW(
        [{"params": net.parameters(), "weight_decay": cli.weight_decay},
         {"params": [log_scale], "weight_decay": 0.0}], lr=cli.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cli.episodes)
    ramp_eps = max(1, int(cli.episodes * cli.worst_dur_ramp_frac))

    if cli.init_checkpoint:
        ck = torch.load(cli.init_checkpoint, map_location=device,
                        weights_only=False)
        net.load_state_dict(ck["state_dict"])
        with torch.no_grad():
            log_scale.copy_(torch.tensor(float(ck["log_scale"])))
        for _ in range(cli.start_episode):  # continue the cosine schedule
            sched.step()
        print(f"resumed {cli.init_checkpoint} "
              f"(saved ep {ck.get('episode', '?')}) at ep {cli.start_episode}; "
              f"optimizer moments start fresh", flush=True)

    dwell_lengths = list(DWELLS.values())
    history = []
    t0 = time.perf_counter()

    for ep in range(cli.start_episode, cli.episodes):
        net.train()
        length = dwell_lengths[int(rng.integers(0, len(dwell_lengths)))]
        picks: list[int] = []
        for cls in CLASSES:
            ids = train_by_class[cls]
            chosen = rng.choice(len(ids), size=cli.k_shot + cli.q_query,
                                replace=len(ids) < cli.k_shot + cli.q_query)
            picks.extend(ids[int(i)] for i in chosen)
        offsets = rng.integers(0, row_samples - length + 1, size=len(picks))
        noisy_np = np.stack([train_window(r, int(o), length)
                             for r, o in zip(picks, offsets)])
        clean_np = np.stack([train_clean(r, 0, min(length, 50_000))
                             for r in picks])
        x = rms_normalize(torch.from_numpy(noisy_np).to(device))
        theta = torch.from_numpy(
            rng.uniform(0, 2 * np.pi, size=(x.shape[0], 1)).astype(np.float32)
        ).to(device)
        rot = torch.polar(torch.ones_like(theta), theta)
        x = x * rot
        # clean target gets the SAME normalization and rotation so the decoder
        # learns denoising, not gain/phase bookkeeping. Reconstruction is
        # scored on a cropped slice (<=50k samples): a regularizer does not
        # need the full 10 ms window, and the crop bounds decoder cost/IO.
        recon_len = min(length, 50_000)
        c = rms_normalize(torch.from_numpy(
            clean_np[:, :recon_len]).to(device)) * rot

        spec = spectrogram(x)
        spec_clean = spectrogram(c)
        # masked+denoising input: zero a random fraction of time frames
        if cli.mask_frac > 0:
            keep = torch.rand(spec.shape[0], 1, spec.shape[2], 1,
                              device=device) > cli.mask_frac * rng.random()
            spec_in = spec * keep
        else:
            spec_in = spec
        z, conf_logit, bottleneck = net.encode(spec_in)

        per = cli.k_shot + cli.q_query
        zc = z.view(len(CLASSES), per, -1)
        protos = zc[:, : cli.k_shot].mean(dim=1)
        query = zc[:, cli.k_shot :].reshape(-1, z.shape[-1])
        logits = -torch.cdist(query, protos) ** 2 \
            * log_scale.exp().clamp(1e-3, 100)
        target = torch.arange(len(CLASSES), device=device
                              ).repeat_interleave(cli.q_query)
        ce = F.cross_entropy(logits, target)

        recon_frames = spec_clean.shape[2]
        recon = net.reconstruct(bottleneck, (spec.shape[0], spec.shape[1],
                                             recon_frames, spec.shape[3]))
        recon_loss = F.l1_loss(recon, spec_clean)

        query_conf = conf_logit.view(len(CLASSES), per)[:, cli.k_shot:].reshape(-1)
        correct = (logits.argmax(dim=1) == target).float()
        conf_loss = F.binary_cross_entropy_with_logits(query_conf, correct)

        loss = ce + cli.recon_weight * recon_loss + cli.conf_weight * conf_loss

        wd_weight = cli.worst_dur_weight * min(1.0, ep / ramp_eps)
        if wd_weight > 0 and ep % 4 == 0:
            wd_weight *= 4.0
            wd_ids = [all_train[int(i)] for i in
                      rng.choice(len(all_train), size=cli.worst_dur_rows,
                                 replace=False)]
            wd_np = np.stack([train_window(r, 0, 200_000) for r in wd_ids])
            wd_targets = torch.tensor(
                [class_index[row_cls[r]] for r in wd_ids], device=device)
            loss = loss + wd_weight * worst_duration_ce(
                net, wd_np, wd_targets, log_scale, protos, device)

        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

        if (ep + 1) % 200 == 0:
            print(f"ep {ep+1}/{cli.episodes} loss={float(loss):.4f} "
                  f"ce={float(ce):.4f} recon={float(recon_loss):.4f} "
                  f"({(time.perf_counter()-t0)/(ep+1)*1000:.0f} ms/ep)",
                  flush=True)

        if (ep + 1) % cli.eval_every == 0 or ep + 1 == cli.episodes:
            net.eval()
            final_pass = ep + 1 == cli.episodes
            # Mid-run evals stream a 1/3 subsample: a full eval pushes ~7 GB
            # through the page cache and macOS answers by swapping out the
            # training RAM cache (measured: 1.8 s/ep -> ~13 s/ep afterwards).
            # The subsample is the frozen seeded-random one (same rows at
            # every checkpoint), not eval_rows[::3] -- see build_eval_plan.
            rows_for_eval = eval_rows if final_pass else mid_eval_rows
            report = evaluate(net, noisy, train_by_class, rows_for_eval,
                              device, train_window_fn=train_window,
                              eval_offsets=eval_offsets,
                              proto_offsets=proto_offsets,
                              eval_plan=eval_plan)
            report["eval_rows_used"] = len(rows_for_eval)
            history.append({"episode": ep + 1, **report})
            line = " | ".join(
                f"{w}: bal={report[w]['balanced_accuracy']:.3f} "
                f"minCell={report[w]['min_profile_recall']:.3f}"
                for w in DWELLS)
            print(f"[eval ep {ep+1}] {line} | "
                  f"esc@1ms: {report['escalation']['answered_frac']:.2f} answered "
                  f"@acc {report['escalation']['answered_accuracy']:.3f}",
                  flush=True)
            if cli.save_checkpoint:
                # rolling save so an MPS wedge can't lose the whole run
                torch.save({"state_dict": net.state_dict(),
                            "log_scale": float(log_scale.detach().cpu()),
                            "config": vars(cli), "episode": ep + 1},
                           cli.save_checkpoint + ".partial")
            # Both observed wedges (2/2) blocked in the first post-eval
            # training step (copy_and_sync -> waitUntilCompleted). Drain the
            # stream and drop eval allocations before resuming episodes.
            gc.collect()
            if device.type == "mps":
                torch.mps.synchronize()
                torch.mps.empty_cache()
            drop_memmap_pages()  # evals stream the corpus; purge the debris

    result = {
        "schema": "v7-trainer-v1",
        "corpus": str(CORPUS),
        "config": vars(cli),
        "eval_plan": eval_plan,
        "params": sum(p.numel() for p in net.parameters()),
        "wall_s": time.perf_counter() - t0,
        "history": history,
        "final": history[-1] if history else None,
    }
    Path(cli.out).write_text(json.dumps(result, indent=2))
    if cli.save_checkpoint:
        torch.save({"state_dict": net.state_dict(),
                    "log_scale": float(log_scale.detach().cpu()),
                    "config": vars(cli)}, cli.save_checkpoint)
    print(f"wrote {cli.out}", flush=True)


def evaluate(net, noisy, train_by_class, eval_rows, device,
             train_window_fn=None, eval_offsets=None, proto_offsets=None,
             eval_plan=None):
    """Per-dwell metrics.  ``eval_offsets`` / ``proto_offsets`` are the
    frozen row_id -> window-start tables from build_eval_plan(); passing
    None reproduces the legacy offset-0 protocol."""
    if train_window_fn is None:
        train_window_fn = lambda r, o, L: np.asarray(noisy[r, o : o + L])

    def q_off(r):    # query row -> window start (0 = legacy protocol)
        return 0 if eval_offsets is None else eval_offsets[r["row"]]

    def p_off(rid):  # prototype row -> window start
        return 0 if proto_offsets is None else proto_offsets[int(rid)]

    report: dict = {}
    conf_records = []
    with torch.no_grad():
        for wname, length in DWELLS.items():
            protos_acc = torch.zeros(len(CLASSES),
                                     net.embed.out_features, device=device)
            protos_n = torch.zeros(len(CLASSES), device=device)
            for ci, cls in enumerate(CLASSES):
                ids = train_by_class[cls][:64]   # capped: eval cost, not accuracy
                for start in range(0, len(ids), 48):
                    chunk = ids[start : start + 48]
                    xb = np.stack([train_window_fn(r, p_off(r), length)
                                   for r in chunk])
                    zb, _, _ = net.encode(spectrogram(rms_normalize(
                        torch.from_numpy(xb).to(device))))
                    protos_acc[ci] += zb.sum(dim=0)
                    protos_n[ci] += len(chunk)
            protos = protos_acc / protos_n.unsqueeze(1)
            per_class_correct = {c: 0 for c in CLASSES}
            per_class_total = {c: 0 for c in CLASSES}
            per_profile: dict[str, list[int]] = {}
            for start in range(0, len(eval_rows), 48):
                chunk = eval_rows[start : start + 48]
                xb = np.stack([noisy[r["row"], q_off(r):q_off(r) + length]
                               for r in chunk])
                zb, confb, _ = net.encode(spectrogram(rms_normalize(
                    torch.from_numpy(xb).to(device))))
                pred = (-torch.cdist(zb, protos) ** 2).argmax(dim=1)
                for r, p, cf in zip(chunk, pred.tolist(), confb.tolist()):
                    ok = CLASSES[p] == r["cls"]
                    per_class_total[r["cls"]] += 1
                    per_class_correct[r["cls"]] += ok
                    per_profile.setdefault(r["profile"], []).append(int(ok))
                    if wname == "1ms":
                        conf_records.append((cf, int(ok)))
            recalls = {c: per_class_correct[c] / max(1, per_class_total[c])
                       for c in CLASSES}
            profile_recalls = {p: sum(v) / len(v)
                               for p, v in per_profile.items()}
            report[wname] = {
                "balanced_accuracy": float(np.mean(list(recalls.values()))),
                "per_class": recalls,
                "min_profile_recall": min(profile_recalls.values()),
                "worst_profiles": sorted(profile_recalls.items(),
                                         key=lambda kv: kv[1])[:3],
            }
        # escalation curve at 1ms: answer when sigmoid(conf) > 0.5
        answered = [(ok) for cf, ok in conf_records if cf > 0]
        report["escalation"] = {
            "answered_frac": len(answered) / max(1, len(conf_records)),
            "answered_accuracy": (sum(answered) / len(answered))
            if answered else 0.0,
        }
    if eval_plan is not None:
        report["eval_protocol"] = {
            k: eval_plan[k] for k in
            ("offset_mode", "subsample_mode", "eval_seed", "plan_sha256")}
    return report


if __name__ == "__main__":
    main()
