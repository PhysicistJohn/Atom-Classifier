"""v7 gating probe: does a full-window spectro-temporal encoder fix GSM@4096?

Hypothesis under test (from the v6 root-cause session, 2026-07-30): the patch
frontend (16x64 samples, 1024-sample span) cannot represent the time-frequency
structure that separates the bursty classes -- GSM's TDMA framing vs Bluetooth's
hopping -- so historical worst-length balanced accuracy asymptotes at ~0.74 no
matter the hyperparameters (75+ runs across two studies).

This probe swaps ONLY the representation: complex window -> single-resolution
STFT stack (Re, Im, log-magnitude) -> small fully-convolutional 2D encoder with
a receptive field spanning the entire window -> the SAME episodic prototype
head used by every model since v3.  Training draws random time-OFFSET windows
(the time-orbit) with empties included; evaluation uses causal prefixes exactly
like the frozen contract.

Success criterion (pre-registered): mean GSM recall on selection rows at the
4096-sample prefix >= 0.75, vs 0.493 for the patch models (mean over 24 v6
trials).  Secondary: bluetooth@4096 (patch baseline 0.557) and per-length
balanced accuracy.

This is a representation experiment, not a contract run: historical corpus
only, no scale orbit, no firewall, dev-only.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
V5 = HERE.parent / "v5_scale_orbit"
for p in (str(V5), str(V5.parent), str(V5.parent.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import v6_harness as H  # noqa: E402  (import-path + corpus identity patch)
import run_scale_orbit_dev as R  # noqa: E402
import current_source_data as corpus_data  # noqa: E402

LENGTHS = (4096, 8192, 16384)
NFFT = 64
HOP = 32
CLASSES = ("am", "bluetooth", "cw", "dsss", "fm", "gsm", "ofdm")


def spectrogram(x: torch.Tensor) -> torch.Tensor:
    """Complex [B, T] -> [B, 3, frames, NFFT//2+1] (Re, Im, log-mag)."""
    frames = x.unfold(-1, NFFT, HOP)                     # [B, F, NFFT] complex
    window = torch.hann_window(NFFT, device=x.device, dtype=torch.float32)
    spec = torch.fft.fft(frames * window, dim=-1)[..., : NFFT // 2 + 1]
    mag = spec.abs()
    return torch.stack(
        (spec.real, spec.imag, torch.log1p(mag)), dim=1
    )  # [B, 3, F, bins]


class Block(nn.Module):
    def __init__(self, cin: int, cout: int, stride: tuple[int, int]):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=1)
        self.norm = nn.GroupNorm(8, cout)

    def forward(self, x):
        return F.silu(self.norm(self.conv(x)))


class SpectroTemporalEncoder(nn.Module):
    """Small fully-convolutional (time) encoder; global receptive field."""

    def __init__(self, embed_dim: int = 128, width: int = 48):
        super().__init__()
        w = width
        self.stem = Block(3, w, (1, 1))
        self.stages = nn.ModuleList([
            Block(w, w * 2, (2, 2)),
            Block(w * 2, w * 2, (2, 1)),
            Block(w * 2, w * 4, (2, 2)),
            Block(w * 4, w * 4, (2, 1)),
            Block(w * 4, w * 4, (2, 2)),
        ])
        self.head = nn.Linear(w * 4 * 2, embed_dim)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        h = self.stem(spec)
        for stage in self.stages:
            h = stage(h)
        # time-mean+max pool -> length-invariant summary, then embed
        pooled = torch.cat(
            (h.mean(dim=(2, 3)), h.amax(dim=(2, 3))), dim=1
        )
        z = self.head(pooled)
        return F.normalize(z, dim=-1) * 4.0


def load_rows():
    corpus, _ = corpus_data.load_historical_exposed(
        R.HISTORICAL_CORPUS,
        patch_length=64, patch_count=16, target_frac=0.5, seed=20260740,
    )
    enroll = list(corpus.rows_by_role["enrollment"])
    select = list(corpus.rows_by_role["selection"])
    return corpus, enroll, select


def raw_window(corpus, row, length: int, offset: int) -> np.ndarray:
    full = np.asarray(corpus.prefix(row, 16384))
    return full[offset : offset + length]


def batch_to_device(windows: list[np.ndarray], device,
                    normalize: bool = True) -> torch.Tensor:
    arr = np.stack(windows).astype(np.complex64)
    x = torch.from_numpy(arr).to(device)
    if normalize:
        rms = x.abs().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-9)
        x = x / rms
    return x


def phase_rotate(x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    theta = torch.from_numpy(
        rng.uniform(0, 2 * np.pi, size=(x.shape[0], 1)).astype(np.float32)
    ).to(x.device)
    return x * torch.polar(torch.ones_like(theta), theta)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=14000)
    ap.add_argument("--k-shot", type=int, default=5)
    ap.add_argument("--q-query", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260740)
    ap.add_argument("--out", default=str(HERE / "probe_result.json"))
    cli = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rng = np.random.default_rng(cli.seed)
    torch.manual_seed(cli.seed)

    corpus, enroll, select = load_rows()
    by_class: dict[str, list] = {c: [] for c in CLASSES}
    for row in enroll:
        by_class[row.class_name].append(row)
    print(f"enrollment rows: {len(enroll)}  selection rows: {len(select)}",
          flush=True)

    net = SpectroTemporalEncoder().to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"encoder params: {n_params:,}", flush=True)
    log_scale = nn.Parameter(
        torch.tensor(math.log(10.0), device=device))
    opt = torch.optim.AdamW(
        [{"params": net.parameters(), "weight_decay": 1e-4},
         {"params": [log_scale], "weight_decay": 0.0}], lr=cli.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cli.episodes)

    history = []
    t0 = time.perf_counter()
    for ep in range(cli.episodes):
        net.train()
        # episode: per class, k+q rows; per row a random offset window with a
        # random length -- the time orbit, empties included (no rejection)
        length = int(rng.choice(LENGTHS))
        windows, labels = [], []
        for ci, cls in enumerate(CLASSES):
            rows = by_class[cls]
            picks = rng.choice(len(rows), size=cli.k_shot + cli.q_query,
                               replace=len(rows) < cli.k_shot + cli.q_query)
            for pi in picks:
                row = rows[int(pi)]
                offset = int(rng.integers(0, 16384 - length + 1))
                windows.append(raw_window(corpus, row, length, offset))
                labels.append(ci)
        x = batch_to_device(windows, device)
        x = phase_rotate(x, rng)
        spec = spectrogram(x)
        z = net(spec)
        per = cli.k_shot + cli.q_query
        z = z.view(len(CLASSES), per, -1)
        support = z[:, : cli.k_shot]
        query = z[:, cli.k_shot :].reshape(-1, z.shape[-1])
        protos = support.mean(dim=1)
        logits = -torch.cdist(query, protos) ** 2 * log_scale.exp().clamp(1e-3, 100)
        target = torch.arange(len(CLASSES), device=device
                              ).repeat_interleave(cli.q_query)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

        if (ep + 1) % 200 == 0:
            print(f"ep {ep+1}/{cli.episodes} loss={float(loss):.4f} "
                  f"({(time.perf_counter()-t0)/(ep+1)*1000:.0f} ms/ep)",
                  flush=True)

        if (ep + 1) % cli.eval_every == 0 or ep + 1 == cli.episodes:
            net.eval()
            report = {}
            with torch.no_grad():
                for L in LENGTHS:
                    # enrollment prototypes at this prefix length
                    proto_acc = torch.zeros(len(CLASSES), 128, device=device)
                    proto_n = torch.zeros(len(CLASSES), device=device)
                    for ci, cls in enumerate(CLASSES):
                        rows = by_class[cls]
                        for start in range(0, len(rows), 128):
                            chunk = rows[start : start + 128]
                            x = batch_to_device(
                                [raw_window(corpus, r, L, 0) for r in chunk],
                                device)
                            z = net(spectrogram(x))
                            proto_acc[ci] += z.sum(dim=0)
                            proto_n[ci] += len(chunk)
                    protos = proto_acc / proto_n.unsqueeze(1)
                    correct = {c: 0 for c in CLASSES}
                    total = {c: 0 for c in CLASSES}
                    for start in range(0, len(select), 128):
                        chunk = select[start : start + 128]
                        x = batch_to_device(
                            [raw_window(corpus, r, L, 0) for r in chunk],
                            device)
                        z = net(spectrogram(x))
                        pred = (-torch.cdist(z, protos) ** 2).argmax(dim=1)
                        for r, p in zip(chunk, pred.tolist()):
                            total[r.class_name] += 1
                            if CLASSES[p] == r.class_name:
                                correct[r.class_name] += 1
                    recalls = {c: correct[c] / max(1, total[c]) for c in CLASSES}
                    report[str(L)] = {
                        "balanced_accuracy":
                            float(np.mean(list(recalls.values()))),
                        "per_class": recalls,
                    }
            entry = {"episode": ep + 1, **report}
            history.append(entry)
            g4 = report["4096"]["per_class"]["gsm"]
            b4 = report["4096"]["per_class"]["bluetooth"]
            print(f"[eval ep {ep+1}] "
                  + " | ".join(
                      f"{L}: bal={report[str(L)]['balanced_accuracy']:.4f} "
                      f"gsm={report[str(L)]['per_class']['gsm']:.4f}"
                      for L in LENGTHS)
                  + f"  (patch-baseline gsm@4096=0.493, bt@4096=0.557; "
                    f"probe bt@4096={b4:.4f})", flush=True)

    result = {
        "schema": "v7-frontend-probe-v1",
        "params": n_params,
        "episodes": cli.episodes,
        "seed": cli.seed,
        "nfft": NFFT, "hop": HOP,
        "wall_s": time.perf_counter() - t0,
        "history": history,
        "final": history[-1] if history else None,
        "patch_baseline_reference": {
            "gsm_4096_mean_24_trials": 0.493,
            "bluetooth_4096_mean_24_trials": 0.557,
            "balanced_4096_best_ever": 0.7434,
        },
        "success_criterion": "gsm@4096 recall >= 0.75",
    }
    Path(cli.out).write_text(json.dumps(result, indent=2))
    print(f"\nwrote {cli.out}", flush=True)


if __name__ == "__main__":
    main()
