#!/usr/bin/env python
"""Classify NeptuneSDR captures with a DACS checkpoint.

Builds (or loads cached) class prototypes from the production corpus train
split — the same enrollment path as tools/eval-dacs-checkpoint.py — then
scores each capture at every dwell that fits, reporting the nearest class,
the distance margin to the runner-up, and the confidence logit.

Prototypes are cached next to the checkpoint
(<ckpt>.prototypes-seed<seed>.npz) so repeat runs need no corpus.

Usage:
  .venv-training/bin/python tools/classify-neptune-capture.py \
    --checkpoint <ckpt.safetensors> --captures training/artifacts/ota-<date>/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
TRAINER_DIR = REPO / "training/zplane_ab/v2_full_variation/v7_probe"
DEFAULT_CORPUS = REPO / "training/artifacts/longdwell-production-corpus-v2"

sys.path.insert(0, str(TRAINER_DIR))
import mlx.core as mx                                          # noqa: E402
import v7_trainer_mlx as T                                     # noqa: E402
from v7_model_mlx import spectrogram, rms_normalize, cdist_sq  # noqa: E402

CLASSES = T.CLASSES
DWELLS = T.DWELLS
MAX_DWELL = max(DWELLS.values())
CHUNK_N = 48


def build_model(ckpt: Path):
    model = T.TrainState(width=48, embed_dim=128)
    kind, _opt, sidecar = T.load_init_checkpoint(model, str(ckpt))
    mx.eval(model.parameters())
    model.net.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    ep = sidecar["episode"] if sidecar else "?"
    print(f"checkpoint {ckpt.name} kind={kind} episode={ep}", flush=True)

    def forward(xb):
        z, conf, _ = model.net.encode(
            spectrogram(rms_normalize(xb)).astype(mx.bfloat16))
        return z.astype(mx.float32), conf.astype(mx.float32)
    return forward


def build_prototypes(forward, proto_corpus: Path, seed: int,
                     rows_per_class: int) -> dict[str, np.ndarray]:
    man = json.loads((proto_corpus / "manifest.json").read_text())
    row_samples = man["rowSamples"]
    train_by_class: dict[str, list[int]] = {c: [] for c in CLASSES}
    for row in man["rows"]:
        if row["role"] == "train":
            train_by_class[row["cls"]].append(row["row"])
    all_train = [r for ids in train_by_class.values() for r in ids]
    plan = T.build_eval_plan(seed, [], all_train, row_samples, MAX_DWELL,
                             offset_mode="random")
    poff = plan[2]
    noisy = np.load(proto_corpus / "noisy.npy", mmap_mode="r")
    protos: dict[str, np.ndarray] = {}
    for wname, length in DWELLS.items():
        plist = []
        for cls in CLASSES:
            ids = train_by_class[cls][:rows_per_class]
            z_parts = []
            for s0 in range(0, len(ids), CHUNK_N):
                chunk = ids[s0:s0 + CHUNK_N]
                segs = np.stack([
                    np.asarray(noisy[r][poff[int(r)]:poff[int(r)] + length])
                    for r in chunk])
                zb, _ = forward(mx.array(segs.astype(np.complex64)))
                z_parts.append(zb)
            zc = mx.concatenate(z_parts, axis=0)
            plist.append(mx.sum(zc, axis=0) / len(ids))
        protos[wname] = np.asarray(mx.stack(plist, axis=0))
        print(f"  prototypes[{wname}] built from "
              f"{rows_per_class}/class train rows", flush=True)
    return protos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--captures", required=True,
                    help="directory of <name>.iq.npy files, or one file")
    ap.add_argument("--proto-corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("--proto-seed", type=int, default=20260740)
    ap.add_argument("--proto-rows-per-class", type=int, default=64)
    ap.add_argument("--windows", type=int, default=5,
                    help="number of window positions per capture per dwell")
    ap.add_argument("--lowpass-hz", type=float, default=None,
                    help="FIR-lowpass each capture to +/- this cutoff before "
                         "classification (channelize a single station, "
                         "matching the single-signal corpus condition)")
    ap.add_argument("--shift-hz", type=float, default=0.0,
                    help="frequency-shift the capture by this much before "
                         "the lowpass (center an off-tune station)")
    ap.add_argument("--add-noise-snr-db", type=float, default=None,
                    help="after channelizing, add full-band AWGN at this "
                         "SNR relative to the remaining signal power "
                         "(reproduces the corpus noise condition)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    forward = build_model(ckpt)

    cache = ckpt.with_suffix(
        f".prototypes-seed{args.proto_seed}.npz")
    if cache.exists():
        protos = {k: v for k, v in np.load(cache).items()}
        print(f"loaded prototype cache {cache.name}", flush=True)
    else:
        protos = build_prototypes(forward, Path(args.proto_corpus),
                                  args.proto_seed,
                                  args.proto_rows_per_class)
        np.savez(cache, **protos)
        print(f"cached prototypes -> {cache.name}", flush=True)

    cap_path = Path(args.captures)
    files = sorted(cap_path.glob("*.iq.npy")) if cap_path.is_dir() \
        else [cap_path]
    if not files:
        raise SystemExit(f"no *.iq.npy under {cap_path}")

    report = []
    for f in files:
        iq = np.load(f).astype(np.complex64)
        if args.shift_hz:
            rate = 20e6
            iq = (iq * np.exp(-2j * np.pi * args.shift_hz
                              * np.arange(len(iq)) / rate)
                  ).astype(np.complex64)
        if args.lowpass_hz:
            from scipy.signal import firwin, fftconvolve
            taps = firwin(1025, args.lowpass_hz, fs=20e6)
            iq = fftconvolve(iq, taps, mode="same").astype(np.complex64)
        if args.add_noise_snr_db is not None:
            sig_p = float(np.mean(np.abs(iq) ** 2))
            noise_p = sig_p / (10 ** (args.add_noise_snr_db / 10))
            rng = np.random.default_rng(20260802)
            iq = (iq + np.sqrt(noise_p / 2)
                  * (rng.standard_normal(len(iq))
                     + 1j * rng.standard_normal(len(iq)))
                  ).astype(np.complex64)
        side = {}
        sj = f.with_name(f.name.replace(".iq.npy", ".json"))
        if sj.exists():
            side = json.loads(sj.read_text())
        entry = {"file": f.name, "sidecar": side, "dwells": {}}
        print(f"\n{f.name}  n={len(iq)}"
              + (f"  {side.get('centerHz', 0)/1e6:.3f} MHz" if side else ""),
              flush=True)
        for wname, length in DWELLS.items():
            if len(iq) < length:
                continue
            n_pos = min(args.windows, max(1, len(iq) - length + 1))
            starts = np.linspace(0, len(iq) - length, n_pos).astype(int)
            xb = mx.array(np.stack([iq[s:s + length] for s in starts]))
            zb, confb = forward(xb)
            d = np.asarray(cdist_sq(zb, mx.array(protos[wname])))
            votes = []
            for i in range(len(starts)):
                order = np.argsort(d[i])
                votes.append({
                    "start": int(starts[i]),
                    "pred": CLASSES[int(order[0])],
                    "margin": float(d[i][order[1]] - d[i][order[0]]),
                    "runner_up": CLASSES[int(order[1])],
                    "conf": float(np.asarray(confb)[i]),
                    "dist": {c: float(d[i][j])
                             for j, c in enumerate(CLASSES)},
                })
            counts: dict[str, int] = {}
            for v in votes:
                counts[v["pred"]] = counts.get(v["pred"], 0) + 1
            top = max(counts.items(), key=lambda kv: kv[1])
            entry["dwells"][wname] = {
                "majority": top[0], "votes": counts, "windows": votes}
            print(f"  {wname:<6} -> {top[0]:<10} votes={counts}  "
                  f"margin(med)={np.median([v['margin'] for v in votes]):.3f}"
                  f"  conf(med)={np.median([v['conf'] for v in votes]):.2f}",
                  flush=True)
        report.append(entry)

    if args.out:
        Path(args.out).write_text(json.dumps({
            "checkpoint": str(ckpt),
            "proto_corpus": args.proto_corpus,
            "proto_seed": args.proto_seed,
            "captures": report,
        }, indent=1))
        print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
