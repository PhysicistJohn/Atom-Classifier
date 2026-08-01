"""Probe v3: the data-attribution experiment.

SAME spectro-temporal encoder architecture that scored balanced 0.50 @4096 /
0.68 @16384 on the duration-starved corpus (probe v2), trained on the new
long-dwell paired corpus where every row spans 20 ms of physical time at a
common 20 Msps.  Windows are drawn by DURATION -- {1 ms, 2.5 ms, 10 ms} --
with random time offsets (empties included, silence statistics intact).

Pre-registered success criteria (vs probe v2 on the old corpus):
  primary   : gsm AND bluetooth recall >= 0.85 at the 10 ms window
              (old corpus, full 16384 window: gsm 0.69, bt 0.80)
  secondary : balanced accuracy at 1 ms >= old 4096-window balanced (0.50)
              -- short windows should not get WORSE with better data
  tertiary  : monotone improvement with window duration per bursty class

Training input is the NOISY side of each pair (the realistic receiver view).
The clean side is reserved for the future reconstruction objective and is not
used here.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
CORPUS = Path(
    __import__("os").environ.get(
        "CORPUS_DIR",
        HERE.parent.parent.parent / "artifacts/longdwell-probe-corpus",
    )
)

TARGET_FS = 20_000_000
WINDOW_SAMPLES = {"1ms": 20_000, "2.5ms": 50_000, "10ms": 200_000}
NFFT = 64
HOP = 32
CLASSES = ("am", "bluetooth", "cw", "dsss", "fm", "gsm", "ofdm")


def spectrogram(x: torch.Tensor) -> torch.Tensor:
    frames = x.unfold(-1, NFFT, HOP)
    window = torch.hann_window(NFFT, device=x.device, dtype=torch.float32)
    spec = torch.fft.fft(frames * window, dim=-1)[..., : NFFT // 2 + 1]
    return torch.stack((spec.real, spec.imag, torch.log1p(spec.abs())), dim=1)


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=1)
        self.norm = nn.GroupNorm(8, cout)

    def forward(self, x):
        return F.silu(self.norm(self.conv(x)))


class SpectroTemporalEncoder(nn.Module):
    """Identical topology to probe v2 (1.0M params)."""

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

    def forward(self, spec):
        h = self.stem(spec)
        for stage in self.stages:
            h = stage(h)
        pooled = torch.cat((h.mean(dim=(2, 3)), h.amax(dim=(2, 3))), dim=1)
        return F.normalize(self.head(pooled), dim=-1) * 4.0


def rms_normalize(x: torch.Tensor) -> torch.Tensor:
    rms = x.abs().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-9)
    return x / rms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=9000)
    ap.add_argument("--k-shot", type=int, default=5)
    ap.add_argument("--q-query", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--eval-every", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=20260740)
    ap.add_argument("--out", default=str(HERE / "probe_v3_result.json"))
    cli = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rng = np.random.default_rng(cli.seed)
    torch.manual_seed(cli.seed)

    manifest = json.loads((CORPUS / "manifest.json").read_text())
    noisy = np.load(CORPUS / "noisy.npy", mmap_mode="r")
    row_samples = manifest["rowSamples"]
    rows = manifest["rows"]
    train_by_class: dict[str, list[int]] = {c: [] for c in CLASSES}
    eval_rows: list[dict] = []
    for row in rows:
        if row["role"] == "train":
            train_by_class[row["cls"]].append(row["row"])
        else:
            eval_rows.append(row)
    print(f"train rows: {sum(len(v) for v in train_by_class.values())}  "
          f"eval rows: {len(eval_rows)}", flush=True)
    for cls in CLASSES:
        assert train_by_class[cls], f"no train rows for {cls}"

    net = SpectroTemporalEncoder().to(device)
    print(f"params: {sum(p.numel() for p in net.parameters()):,}", flush=True)
    log_scale = nn.Parameter(torch.tensor(math.log(10.0), device=device))
    opt = torch.optim.AdamW(
        [{"params": net.parameters(), "weight_decay": 1e-4},
         {"params": [log_scale], "weight_decay": 0.0}], lr=cli.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cli.episodes)

    window_keys = list(WINDOW_SAMPLES)
    history = []
    t0 = time.perf_counter()

    def batch_windows(row_ids: list[int], length: int) -> torch.Tensor:
        out = np.empty((len(row_ids), length), dtype=np.complex64)
        for slot, rid in enumerate(row_ids):
            offset = int(rng.integers(0, row_samples - length + 1))
            out[slot] = noisy[rid, offset : offset + length]
        return torch.from_numpy(out).to(device)

    for ep in range(cli.episodes):
        net.train()
        length = WINDOW_SAMPLES[window_keys[int(rng.integers(0, len(window_keys)))]]
        picks: list[int] = []
        for cls in CLASSES:
            ids = train_by_class[cls]
            chosen = rng.choice(len(ids), size=cli.k_shot + cli.q_query,
                                replace=len(ids) < cli.k_shot + cli.q_query)
            picks.extend(ids[int(i)] for i in chosen)
        x = rms_normalize(batch_windows(picks, length))
        theta = torch.from_numpy(
            rng.uniform(0, 2 * np.pi, size=(x.shape[0], 1)).astype(np.float32)
        ).to(device)
        x = x * torch.polar(torch.ones_like(theta), theta)
        z = net(spectrogram(x))
        per = cli.k_shot + cli.q_query
        z = z.view(len(CLASSES), per, -1)
        protos = z[:, : cli.k_shot].mean(dim=1)
        query = z[:, cli.k_shot :].reshape(-1, z.shape[-1])
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
                for wname, length in WINDOW_SAMPLES.items():
                    # class prototypes from train rows, causal-prefix windows
                    protos_acc = torch.zeros(len(CLASSES), 128, device=device)
                    protos_n = torch.zeros(len(CLASSES), device=device)
                    for ci, cls in enumerate(CLASSES):
                        ids = train_by_class[cls]
                        for start in range(0, len(ids), 48):
                            chunk = ids[start : start + 48]
                            xb = np.stack([
                                noisy[rid, :length] for rid in chunk
                            ]).astype(np.complex64)
                            zb = net(spectrogram(rms_normalize(
                                torch.from_numpy(xb).to(device))))
                            protos_acc[ci] += zb.sum(dim=0)
                            protos_n[ci] += len(chunk)
                    protos = protos_acc / protos_n.unsqueeze(1)
                    correct = {c: 0 for c in CLASSES}
                    total = {c: 0 for c in CLASSES}
                    for start in range(0, len(eval_rows), 48):
                        chunk = eval_rows[start : start + 48]
                        xb = np.stack([
                            noisy[r["row"], :length] for r in chunk
                        ]).astype(np.complex64)
                        zb = net(spectrogram(rms_normalize(
                            torch.from_numpy(xb).to(device))))
                        pred = (-torch.cdist(zb, protos) ** 2).argmax(dim=1)
                        for r, p in zip(chunk, pred.tolist()):
                            total[r["cls"]] += 1
                            if CLASSES[p] == r["cls"]:
                                correct[r["cls"]] += 1
                    recalls = {c: correct[c] / max(1, total[c]) for c in CLASSES}
                    report[wname] = {
                        "balanced_accuracy": float(np.mean(list(recalls.values()))),
                        "per_class": recalls,
                    }
            history.append({"episode": ep + 1, **report})
            line = " | ".join(
                f"{w}: bal={report[w]['balanced_accuracy']:.3f} "
                f"gsm={report[w]['per_class']['gsm']:.3f} "
                f"bt={report[w]['per_class']['bluetooth']:.3f}"
                for w in window_keys)
            print(f"[eval ep {ep+1}] {line}", flush=True)

    result = {
        "schema": "v7-probe-v3-longdwell-v1",
        "corpus": str(CORPUS),
        "episodes": cli.episodes,
        "seed": cli.seed,
        "wall_s": time.perf_counter() - t0,
        "history": history,
        "final": history[-1] if history else None,
        "old_corpus_reference": {
            "probe_v2_balanced_4096": 0.5032,
            "probe_v2_balanced_16384": 0.6817,
            "probe_v2_gsm_16384": 0.6948,
            "probe_v2_bt_16384": 0.8022,
            "patch_best_balanced_4096": 0.7434,
        },
        "success_criteria": {
            "primary": "gsm AND bluetooth recall >= 0.85 at 10ms",
            "secondary": "balanced@1ms >= 0.50",
            "tertiary": "monotone recall vs duration for gsm/bt/dsss",
        },
    }
    Path(cli.out).write_text(json.dumps(result, indent=2))
    print(f"wrote {cli.out}", flush=True)


if __name__ == "__main__":
    main()
