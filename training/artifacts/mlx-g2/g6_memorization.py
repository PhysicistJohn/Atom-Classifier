#!/usr/bin/env python
"""G6: what does the content-diversity deficit cost the MODEL?

Scores the frozen G4 bf16 ep3000 checkpoint under the CORRECTED eval
protocol (v7_trainer_mlx.build_eval_plan, random per-row window offsets,
prototypes offset the same way) on BOTH halves of the split:

  * the 3264 EVAL rows      -- must reproduce g5_offset_reeval.json exactly
  * the 5440 TRAIN rows     -- same forward path, same prototypes, but the
                               query offsets come from an INDEPENDENT plan
                               (different seed) so a prototype row is never
                               scored on the identical window it donated

Prototypes always come from the train rows (first 64 ids/class), verbatim
g5/evaluate() path: fp16 train cache, proto offsets, chunk_n = 48.

Per-row predictions are dumped so downstream slicing (by profile, by
content-realization cluster, by prototype membership) is exact.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

assert "--no-tf32" not in sys.argv

TRAINER = Path("/Users/johnelliott/PersonalGitHub/Atom-Classifier/training/"
               "zplane_ab/v2_full_variation/v7_probe/v7_trainer_mlx.py")
HERE = Path(__file__).parent
OUT = HERE / "g6_memorization.json"

sys.path.insert(0, str(TRAINER.parent))
import mlx.core as mx                                          # noqa: E402
import v7_trainer_mlx as T                                     # noqa: E402
from v7_model_mlx import spectrogram, rms_normalize, cdist_sq  # noqa: E402

CLASSES = T.CLASSES
DWELLS = T.DWELLS
CORPUS = T.CORPUS
MAX_DWELL = max(DWELLS.values())
CKPT = Path("/Users/johnelliott/PersonalGitHub/Atom-Classifier/training/"
            "artifacts/mlx-g2/g4_bf16.safetensors")
GB = 1 << 30
CHUNK_N = 48
SEEDS = [20260740, 20260741, 20260742]
TRAIN_QUERY_SEED_OFFSET = 500_000     # independent stream for train queries


def main() -> None:
    t_boot = time.perf_counter()
    rel, tf32_on = T.measure_matmul_precision()
    print(f"mlx {mx.__version__} device={mx.default_device()} "
          f"TF32 {'ON' if tf32_on else 'off'} ({rel:.2e}) bf16", flush=True)
    assert tf32_on
    try:
        mx.set_wired_limit(int(30 * GB))
    except Exception as e:
        print(f"WARNING set_wired_limit: {e!r}", flush=True)
    mx.set_cache_limit(2 * GB)

    manifest = json.loads((CORPUS / "manifest.json").read_text())
    row_samples = manifest["rowSamples"]
    rows = manifest["rows"]
    train_by_class = {c: [] for c in CLASSES}
    eval_rows, train_rows = [], []
    for row in rows:
        if row["role"] == "train":
            train_by_class[row["cls"]].append(row["row"])
            train_rows.append(row)
        else:
            eval_rows.append(row)
    all_train = [r for ids in train_by_class.values() for r in ids]
    n_train, n_eval = len(all_train), len(eval_rows)
    train_pos = {rid: slot for slot, rid in enumerate(all_train)}
    # query-store slot maps
    eval_qpos = {r["row"]: i for i, r in enumerate(eval_rows)}
    train_qpos = {r["row"]: i for i, r in enumerate(train_rows)}
    proto_ids = {c: train_by_class[c][:64] for c in CLASSES}
    proto_set = {rid for ids in proto_ids.values() for rid in ids}
    print(f"train {n_train} eval {n_eval} protoRows {len(proto_set)} "
          f"rowSamples {row_samples} maxDwell {MAX_DWELL}", flush=True)

    # ---------------- model ------------------------------------------------
    model = T.TrainState(width=48, embed_dim=128)
    kind, _opt, sidecar = T.load_init_checkpoint(model, str(CKPT))
    mx.eval(model.parameters())
    pre_hash = T.params_sha256(model)
    model.net.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    ep = sidecar["episode"] if sidecar else "?"
    print(f"ckpt {CKPT.name} kind={kind} ep={ep} "
          f"hashOK={bool(sidecar and pre_hash == sidecar.get('params_sha256'))}",
          flush=True)

    def eval_forward(xb):
        z, conf, _ = model.net.encode(
            spectrogram(rms_normalize(xb)).astype(mx.bfloat16))
        return z.astype(mx.float32), conf.astype(mx.float32)

    # ---------------- resident train fp16 cache (prototype source) ---------
    noisy = np.load(CORPUS / "noisy.npy", mmap_mode="r")
    rows_per_chunk = max(1, (512 << 20) // (row_samples * 8))
    t0 = time.perf_counter()
    train_iq = mx.zeros((n_train, 2, row_samples), dtype=mx.float16)
    for i0 in range(0, n_train, rows_per_chunk):
        ids = all_train[i0:i0 + rows_per_chunk]
        rc = np.asarray(noisy[ids])
        planes = np.stack((rc.real, rc.imag), axis=1).astype(np.float16)
        train_iq[i0:i0 + len(ids)] = mx.array(planes)
        mx.eval(train_iq)
        del rc, planes
    print(f"train fp16 cache {train_iq.nbytes/1e9:.2f} GB in "
          f"{time.perf_counter()-t0:.0f}s", flush=True)

    def load_windows(row_list, off_map):
        """complex64 (n, MAX_DWELL) windows at the frozen per-row offsets."""
        n = len(row_list)
        store = mx.zeros((n, MAX_DWELL), dtype=mx.complex64)
        for i0 in range(0, n, rows_per_chunk):
            sub = row_list[i0:i0 + rows_per_chunk]
            raw = np.asarray(noisy[[r["row"] for r in sub]])
            buf = np.empty((len(sub), MAX_DWELL), dtype=np.complex64)
            for j, r in enumerate(sub):
                o = off_map[r["row"]]
                buf[j] = raw[j, o:o + MAX_DWELL]
            store[i0:i0 + len(sub)] = mx.array(buf)
            mx.eval(store)
            del raw, buf
        return store

    def score(store, qpos, rows_q, protos, length):
        n = len(rows_q)
        preds = np.empty(n, dtype=np.int32)
        confs = np.empty(n, dtype=np.float64)
        for s0 in range(0, n, CHUNK_N):
            chunk = rows_q[s0:s0 + CHUNK_N]
            slots = np.array([qpos[r["row"]] for r in chunk], dtype=np.int32)
            pad = np.concatenate(
                [slots, np.zeros(CHUNK_N - len(chunk), dtype=np.int32)])
            xb = mx.take(store, mx.array(pad), axis=0)[:, :length]
            zb, confb = eval_forward(xb)
            pred = mx.argmax(-cdist_sq(zb[:len(chunk)], protos), axis=1)
            preds[s0:s0 + len(chunk)] = np.asarray(pred)
            confs[s0:s0 + len(chunk)] = np.asarray(confb[:len(chunk)])
        return preds, confs

    results = {}
    for seed in SEEDS:
        _mid, eval_off, proto_off, meta = T.build_eval_plan(
            seed, eval_rows, all_train, row_samples, MAX_DWELL,
            offset_mode="random")
        _m2, train_q_off, _p2, meta2 = T.build_eval_plan(
            seed + TRAIN_QUERY_SEED_OFFSET, train_rows, all_train,
            row_samples, MAX_DWELL, offset_mode="random")
        print(f"\n[seed {seed}] evalPlan {meta['plan_sha256'][:12]} "
              f"trainQueryPlan {meta2['plan_sha256'][:12]}", flush=True)
        # sanity: a prototype row's query window differs from its proto window
        same = sum(1 for rid in proto_set
                   if train_q_off[rid] == proto_off[rid])
        print(f"  proto rows whose query offset == proto offset: "
              f"{same}/{len(proto_set)}", flush=True)

        t1 = time.perf_counter()
        eval_store = load_windows(eval_rows, eval_off)
        train_store = load_windows(train_rows, train_q_off)
        print(f"  query stores {(eval_store.nbytes+train_store.nbytes)/1e9:.2f}"
              f" GB in {time.perf_counter()-t1:.0f}s", flush=True)

        res = {"eval": {}, "train": {}, "meta": meta, "meta_train_q": meta2}
        for wname, length in DWELLS.items():
            mx.clear_cache()
            protos_list = []
            for cls in CLASSES:
                ids = proto_ids[cls]
                z_parts = []
                for s0 in range(0, len(ids), CHUNK_N):
                    ch = ids[s0:s0 + CHUNK_N]
                    slots = np.array([train_pos[r] for r in ch],
                                     dtype=np.int32)
                    pad = np.concatenate(
                        [slots, np.zeros(CHUNK_N - len(ch), dtype=np.int32)])
                    g = mx.take(train_iq, mx.array(pad), axis=0)
                    offs = np.concatenate(
                        [np.array([proto_off[int(r)] for r in ch],
                                  dtype=np.int64),
                         np.zeros(CHUNK_N - len(ch), dtype=np.int64)])
                    idx = mx.array((offs[:, None, None]
                                    + np.arange(length)[None, None, :]
                                    ).astype(np.int32))
                    seg = mx.take_along_axis(g, idx, axis=2).astype(mx.float32)
                    xb = seg[:, 0] + 1j * seg[:, 1]
                    zb, _ = eval_forward(xb)
                    z_parts.append(zb[:len(ch)])
                zc = mx.concatenate(z_parts, axis=0)
                protos_list.append(mx.sum(zc, axis=0) / len(ids))
            protos = mx.stack(protos_list, axis=0)
            mx.eval(protos)

            pe, ce = score(eval_store, eval_qpos, eval_rows, protos, length)
            pt, ct = score(train_store, train_qpos, train_rows, protos, length)
            res["eval"][wname] = {
                "row": [r["row"] for r in eval_rows],
                "pred": pe.tolist(), "conf": ce.tolist()}
            res["train"][wname] = {
                "row": [r["row"] for r in train_rows],
                "pred": pt.tolist(), "conf": ct.tolist()}
            acc_e = float(np.mean([CLASSES[p] == r["cls"]
                                   for p, r in zip(pe, eval_rows)]))
            acc_t = float(np.mean([CLASSES[p] == r["cls"]
                                   for p, r in zip(pt, train_rows)]))
            print(f"  {wname:<6} raw acc  eval={acc_e:.4f}  train={acc_t:.4f}",
                  flush=True)
        results[str(seed)] = res
        del eval_store, train_store
        mx.clear_cache()

    OUT.write_text(json.dumps({
        "checkpoint": str(CKPT), "episode": ep, "seeds": SEEDS,
        "train_query_seed_offset": TRAIN_QUERY_SEED_OFFSET,
        "proto_rows": sorted(proto_set),
        "results": results,
        "run_s": time.perf_counter() - t_boot}, indent=None))
    print(f"\nwrote {OUT} ({time.perf_counter()-t_boot:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
