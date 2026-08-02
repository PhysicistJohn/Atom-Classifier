#!/usr/bin/env python
"""Evaluate a DACS checkpoint under the corrected multi-seed offset protocol.

Parameterized descendant of training/artifacts/mlx-g2/g5_offset_reeval.py
(the script that established the corrected numbers). The forward path is the
trainer's own evaluate() semantics verbatim: bf16 net, TF32 default ON,
prototypes from an fp16 train cache, queries from a complex64 store, argmin
cdist_sq. Offset plans come from build_eval_plan imported FROM the trainer.

New relative to g5: the prototype (enrollment) corpus and the query corpus
may differ. That is the held-out scenario: enroll prototypes from the
production train split, query rows the model has never seen in any form.

Usage:
  .venv-training/bin/python tools/eval-dacs-checkpoint.py \
    --checkpoint <ckpt.safetensors> \
    --query-corpus training/artifacts/<holdout-dir> --query-role all \
    --proto-corpus training/artifacts/longdwell-production-corpus-v2 \
    --out <result.json>

Defaults reproduce the standard five-seed eval on the production corpus
eval split.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# the run's own config is bf16 with TF32 default ON; do not pin it off here
assert "--no-tf32" not in sys.argv

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
GB = 1 << 30
CHUNK_N = 48


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--query-corpus", default=str(DEFAULT_CORPUS),
                    help="corpus whose rows are scored")
    ap.add_argument("--proto-corpus", default=None,
                    help="corpus whose TRAIN rows build the prototypes "
                         "(default: the query corpus)")
    ap.add_argument("--query-role", choices=("eval", "all"), default="eval",
                    help="'all' scores every row (use for a holdout corpus "
                         "with no train split)")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[20260740, 20260741, 20260742, 20260743,
                             20260744])
    ap.add_argument("--legacy-configs", action="store_true",
                    help="also score offset0 and mixedProtoZero")
    ap.add_argument("--proto-rows-per-class", type=int, default=64)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def metrics(eval_rows, preds, confs) -> dict:
    per_class_correct = {c: 0 for c in CLASSES}
    per_class_total = {c: 0 for c in CLASSES}
    per_profile: dict[str, list[int]] = {}
    conf_records = []
    for i, r in enumerate(eval_rows):
        ok = int(CLASSES[preds[i]] == r["cls"])
        per_class_total[r["cls"]] += 1
        per_class_correct[r["cls"]] += ok
        per_profile.setdefault(r["profile"], []).append(ok)
        conf_records.append((confs[i], ok))
    recalls = {c: per_class_correct[c] / max(1, per_class_total[c])
               for c in CLASSES if per_class_total[c]}
    errors = {c: per_class_total[c] - per_class_correct[c] for c in CLASSES}
    prof = {p: sum(v) / len(v) for p, v in per_profile.items()}
    worst_p = min(prof.items(), key=lambda kv: kv[1])
    answered = [ok for cf, ok in conf_records if cf > 0]
    return {
        "n": len(eval_rows),
        "balanced_accuracy": float(np.mean(list(recalls.values()))),
        "accuracy": sum(per_class_correct.values()) / max(1, len(eval_rows)),
        "per_class": recalls,
        "per_class_total": per_class_total,
        "per_class_errors": errors,
        "errors_total": int(sum(errors.values())),
        "min_profile_recall": worst_p[1],
        "min_profile": worst_p[0],
        "n_profiles": len(prof),
        "profile_recalls": prof,
        "worst_profiles": sorted(prof.items(), key=lambda kv: kv[1])[:5],
        "escalation": {
            "answered_frac": len(answered) / max(1, len(conf_records)),
            "answered_accuracy": (sum(answered) / len(answered))
            if answered else 0.0,
        },
    }


def load_manifest(corpus: Path) -> dict:
    return json.loads((corpus / "manifest.json").read_text())


def main() -> None:
    args = parse_args()
    t_boot = time.perf_counter()
    query_corpus = Path(args.query_corpus)
    proto_corpus = Path(args.proto_corpus) if args.proto_corpus \
        else query_corpus
    ckpt = Path(args.checkpoint)

    rel, tf32_on = T.measure_matmul_precision()
    print(f"mlx {mx.__version__}  device={mx.default_device()}  "
          f"TF32 {'ON' if tf32_on else 'off'} (rel err {rel:.2e})  "
          f"dtype=bf16", flush=True)
    try:
        mx.set_wired_limit(int(22 * GB))
    except Exception as e:
        print(f"WARNING set_wired_limit: {e!r}", flush=True)
    mx.set_cache_limit(2 * GB)

    # ---------------- rows ------------------------------------------------
    qman = load_manifest(query_corpus)
    pman = load_manifest(proto_corpus)
    row_samples = qman["rowSamples"]
    assert pman["rowSamples"] == row_samples, "rowSamples mismatch"

    if args.query_role == "all":
        eval_rows = list(qman["rows"])
    else:
        eval_rows = [r for r in qman["rows"] if r["role"] != "train"]
    train_by_class: dict[str, list[int]] = {c: [] for c in CLASSES}
    for row in pman["rows"]:
        if row["role"] == "train":
            train_by_class[row["cls"]].append(row["row"])
    all_train = [r for ids in train_by_class.values() for r in ids]
    assert all_train, f"no train rows in {proto_corpus}"
    train_pos = {rid: slot for slot, rid in enumerate(all_train)}
    eval_pos = {r["row"]: slot for slot, r in enumerate(eval_rows)}
    n_train, n_eval = len(all_train), len(eval_rows)
    print(f"query {query_corpus.name} role={args.query_role} n={n_eval}  "
          f"proto {proto_corpus.name} train n={n_train}  "
          f"rowSamples {row_samples}", flush=True)

    # ---------------- eval plans (trainer's own code) ---------------------
    plans: dict[str, tuple] = {}
    plan_meta: dict[str, dict] = {}
    zero_plan = T.build_eval_plan(args.seeds[0], eval_rows, all_train,
                                  row_samples, MAX_DWELL, offset_mode="zero")
    for s in args.seeds:
        p = T.build_eval_plan(s, eval_rows, all_train, row_samples,
                              MAX_DWELL, offset_mode="random")
        plans[f"seed{s}"] = (p[1], p[2])
        plan_meta[f"seed{s}"] = p[3]
    if args.legacy_configs:
        plans["offset0"] = (zero_plan[1], zero_plan[2])
        plan_meta["offset0"] = zero_plan[3]
        plans["mixedProtoZero"] = (plans[f"seed{args.seeds[0]}"][0],
                                   zero_plan[2])
        plan_meta["mixedProtoZero"] = {
            **plan_meta[f"seed{args.seeds[0]}"], "proto_offset_mode": "zero"}
    for k, m in plan_meta.items():
        print(f"  {k:<16} sha {m['plan_sha256'][:16]} "
              f"offsets [{m['eval_offset_min']}, {m['eval_offset_max']}] "
              f"mean {m['eval_offset_mean']:.0f}", flush=True)

    # ---------------- resident stores -------------------------------------
    t_conv = time.perf_counter()
    rows_per_chunk = max(1, (512 << 20) // (row_samples * 8))
    qnoisy = np.load(query_corpus / "noisy.npy", mmap_mode="r")
    eval_store = mx.zeros((n_eval, row_samples), dtype=mx.complex64)
    eval_ids = [r["row"] for r in eval_rows]
    for i0 in range(0, n_eval, rows_per_chunk):
        ids = eval_ids[i0:i0 + rows_per_chunk]
        chunk = np.ascontiguousarray(qnoisy[ids])
        eval_store[i0:i0 + len(ids)] = mx.array(chunk)
        mx.eval(eval_store)
        del chunk
    del qnoisy
    pnoisy = np.load(proto_corpus / "noisy.npy", mmap_mode="r")
    train_iq = mx.zeros((n_train, 2, row_samples), dtype=mx.float16)
    for i0 in range(0, n_train, rows_per_chunk):
        ids = all_train[i0:i0 + rows_per_chunk]
        rc = np.asarray(pnoisy[ids])
        planes = np.stack((rc.real, rc.imag), axis=1).astype(np.float16)
        train_iq[i0:i0 + len(ids)] = mx.array(planes)
        mx.eval(train_iq)
        del rc, planes
    del pnoisy
    print(f"resident: query {eval_store.nbytes/1e9:.2f} GB c64 + proto "
          f"{train_iq.nbytes/1e9:.2f} GB fp16 in "
          f"{time.perf_counter()-t_conv:.0f}s", flush=True)

    # ---------------- model ------------------------------------------------
    model = T.TrainState(width=48, embed_dim=128)
    kind, _opt, sidecar = T.load_init_checkpoint(model, str(ckpt))
    mx.eval(model.parameters())
    pre_hash = T.params_sha256(model)
    model.net.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    ep = sidecar["episode"] if sidecar else "?"
    hash_ok = bool(sidecar and pre_hash == sidecar.get("params_sha256"))
    print(f"checkpoint {ckpt.name} kind={kind} episode={ep} "
          f"sidecar hash match={hash_ok}", flush=True)

    def eval_forward(xb):
        z, conf, _ = model.net.encode(
            spectrogram(rms_normalize(xb)).astype(mx.bfloat16))
        return z.astype(mx.float32), conf.astype(mx.float32)

    # ---------------- the evaluation ---------------------------------------
    results: dict[str, dict] = {}
    correct_rows: dict[str, dict[str, list[int]]] = {}
    for name, (eoff, poff) in plans.items():
        results[name] = {}
        correct_rows[name] = {}
        t_cfg = time.perf_counter()
        for wname, length in DWELLS.items():
            mx.clear_cache()
            protos_list = []
            for cls in CLASSES:
                ids = train_by_class[cls][:args.proto_rows_per_class]
                z_parts = []
                for s0 in range(0, len(ids), CHUNK_N):
                    chunk = ids[s0:s0 + CHUNK_N]
                    slots_np = np.array([train_pos[r] for r in chunk],
                                        dtype=np.int32)
                    pad = np.concatenate(
                        [slots_np,
                         np.zeros(CHUNK_N - len(chunk), dtype=np.int32)])
                    g = mx.take(train_iq, mx.array(pad), axis=0)
                    offs = np.concatenate(
                        [np.array([poff[int(r)] for r in chunk],
                                  dtype=np.int64),
                         np.zeros(CHUNK_N - len(chunk), dtype=np.int64)])
                    if not offs.any():
                        seg = g[:, :, :length].astype(mx.float32)
                    else:
                        idx = mx.array((offs[:, None, None]
                                        + np.arange(length)[None, None, :]
                                        ).astype(np.int32))
                        seg = mx.take_along_axis(g, idx, axis=2).astype(
                            mx.float32)
                    xb = seg[:, 0] + 1j * seg[:, 1]
                    zb, _ = eval_forward(xb)
                    z_parts.append(zb[:len(chunk)])
                zc_all = mx.concatenate(z_parts, axis=0)
                protos_list.append(mx.sum(zc_all, axis=0) / len(ids))
            protos = mx.stack(protos_list, axis=0)
            mx.eval(protos)

            preds = np.empty(n_eval, dtype=np.int32)
            confs = np.empty(n_eval, dtype=np.float64)
            for s0 in range(0, n_eval, CHUNK_N):
                chunk = eval_rows[s0:s0 + CHUNK_N]
                slots_np = np.array([eval_pos[r["row"]] for r in chunk],
                                    dtype=np.int32)
                pad = np.concatenate(
                    [slots_np,
                     np.zeros(CHUNK_N - len(chunk), dtype=np.int32)])
                xb_full = mx.take(eval_store, mx.array(pad), axis=0)
                offs = np.concatenate(
                    [np.array([eoff[r["row"]] for r in chunk],
                              dtype=np.int64),
                     np.zeros(CHUNK_N - len(chunk), dtype=np.int64)])
                if not offs.any():
                    xb = xb_full[:, :length]
                else:
                    idx = mx.array((offs[:, None]
                                    + np.arange(length)[None, :]
                                    ).astype(np.int32))
                    xb = mx.take_along_axis(xb_full, idx, axis=1)
                zb, confb = eval_forward(xb)
                pred = mx.argmax(-cdist_sq(zb[:len(chunk)], protos), axis=1)
                preds[s0:s0 + len(chunk)] = np.asarray(pred)
                confs[s0:s0 + len(chunk)] = np.asarray(confb[:len(chunk)])

            results[name][wname] = metrics(eval_rows, preds, confs)
            correct_rows[name][wname] = [
                int(CLASSES[preds[i]] == eval_rows[i]["cls"])
                for i in range(n_eval)]
        print(f"  [{name}] {time.perf_counter()-t_cfg:.0f}s  " + "  ".join(
            f"{w} bal={results[name][w]['balanced_accuracy']:.4f} "
            f"minCell={results[name][w]['min_profile_recall']:.4f}"
            for w in DWELLS), flush=True)

    # ---------------- report -----------------------------------------------
    seed_keys = [f"seed{s}" for s in args.seeds]

    def spread(getter):
        out = {}
        for w in DWELLS:
            v = np.array([getter(results[k][w]) for k in seed_keys],
                         dtype=np.float64)
            out[w] = {"mean": float(v.mean()), "min": float(v.min()),
                      "max": float(v.max()),
                      "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                      "values": [float(x) for x in v]}
        return out

    present = [c for c in CLASSES
               if results[seed_keys[0]]["1ms"]["per_class_total"][c]]
    summary = {
        "balanced_accuracy": spread(lambda r: r["balanced_accuracy"]),
        "min_profile_recall": spread(lambda r: r["min_profile_recall"]),
        "errors_total": spread(lambda r: r["errors_total"]),
        **{f"recall_{c}": spread(lambda r, c=c: r["per_class"][c])
           for c in present},
    }

    print("\n" + "=" * 78)
    print(f"{ckpt.name} on {query_corpus.name} "
          f"({n_eval} rows, prototypes from {proto_corpus.name})")
    print("=" * 78)
    for key, label in (("balanced_accuracy", "balanced accuracy"),
                       ("min_profile_recall", "min-profile recall"),
                       ("errors_total", "error count")):
        print(f"  {label}:")
        for w in DWELLS:
            s = summary[key][w]
            fmt = "{:.0f}" if key == "errors_total" else "{:.4f}"
            print(f"    {w:<6} " + fmt.format(s['mean'])
                  + " [" + fmt.format(s['min']) + ", "
                  + fmt.format(s['max']) + "]")
    print("\nper-class recall (mean over seeds):")
    for w in DWELLS:
        print(f"  {w}: " + "  ".join(
            f"{c}={summary[f'recall_{c}'][w]['mean']:.3f}" for c in present))

    out = {
        "checkpoint": str(ckpt),
        "episode": ep,
        "sidecar_hash_match": hash_ok,
        "query_corpus": str(query_corpus),
        "query_role": args.query_role,
        "proto_corpus": str(proto_corpus),
        "proto_rows_per_class": args.proto_rows_per_class,
        "protocol": {
            "row_samples": row_samples,
            "max_dwell": MAX_DWELL,
            "offset_range": [0, row_samples - MAX_DWELL],
            "seeds": args.seeds,
            "plan_meta": plan_meta,
        },
        "results": results,
        "summary_over_seeds": summary,
        "correct_by_config_dwell": correct_rows,
        "run_s": time.perf_counter() - t_boot,
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out} ({time.perf_counter()-t_boot:.0f}s total)",
          flush=True)


if __name__ == "__main__":
    main()
