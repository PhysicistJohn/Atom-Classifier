#!/usr/bin/env python
"""Re-evaluate the final G4 checkpoint under the CORRECTED eval protocol.

The old protocol started every eval window at capture offset 0, so the burst
phase of every bursty profile was pinned to `startSampleIndex mod period` --
three discrete values across the split for GSM (period 78000 = 3 x the 26000
row stride).  The corrected protocol draws a per-row window offset ONCE from
a dedicated seeded stream, in [0, rowSamples - maxDwell] so one offset is
valid at all three dwells and the windows nest (10 ms strictly extends 2.5 ms
strictly extends 1 ms).  Prototype rows get the same treatment.

This script:
  * imports build_eval_plan FROM the trainers (so the plan under test is the
    trainer's own code, not a copy) and asserts torch and MLX agree;
  * reproduces the trainer's evaluate() forward path verbatim -- bf16 net,
    TF32 default ON, prototypes from the fp16 TRAIN cache (the precision
    asymmetry), queries from the complex64 eval store, argmin cdist_sq;
  * scores the FULL 3264-row eval split at 1/2.5/10 ms under
      offset0        legacy protocol (all offsets 0)
      seed<N> x 5    corrected protocol, five different offset seeds
      mixedProtoZero corrected queries but legacy offset-0 prototypes
                     (isolates how much of the aliasing lives on the
                     reference side of the comparison)
  * reports balanced accuracy, min-cell (min per-profile) recall, per-class
    recall and error counts, and the spread over the five seeds.

Usage:
  .venv-training/bin/python g5_offset_reeval.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# deliberately NOT setting MLX_ENABLE_TF32: the run's own config is bf16 with
# TF32 default ON (the trainer only pins it off when --no-tf32 is in argv)
assert "--no-tf32" not in sys.argv

TRAINER = Path("/Users/johnelliott/PersonalGitHub/Atom-Classifier/training/"
               "zplane_ab/v2_full_variation/v7_probe/v7_trainer_mlx.py")
ART = Path("/Users/johnelliott/PersonalGitHub/Atom-Classifier/training/"
           "artifacts/mlx-g2")
CKPT = ART / "g4_bf16.safetensors"
OUT = ART / "g5_offset_reeval.json"

sys.path.insert(0, str(TRAINER.parent))
import mlx.core as mx                                          # noqa: E402
import v7_trainer_mlx as T                                     # noqa: E402
from v7_model_mlx import spectrogram, rms_normalize, cdist_sq  # noqa: E402

CLASSES = T.CLASSES
DWELLS = T.DWELLS
CORPUS = T.CORPUS
MAX_DWELL = max(DWELLS.values())
GB = 1 << 30
CHUNK_N = 48
SEEDS = [20260740, 20260741, 20260742, 20260743, 20260744]
GSM_BURST_PERIOD = 78_000     # measured in subsample_bias_test.py


def metrics(eval_rows, preds, confs, idxs) -> dict:
    per_class_correct = {c: 0 for c in CLASSES}
    per_class_total = {c: 0 for c in CLASSES}
    per_profile: dict[str, list[int]] = {}
    conf_records = []
    for i in idxs:
        r = eval_rows[i]
        ok = int(CLASSES[preds[i]] == r["cls"])
        per_class_total[r["cls"]] += 1
        per_class_correct[r["cls"]] += ok
        per_profile.setdefault(r["profile"], []).append(ok)
        conf_records.append((confs[i], ok))
    recalls = {c: per_class_correct[c] / max(1, per_class_total[c])
               for c in CLASSES}
    errors = {c: per_class_total[c] - per_class_correct[c] for c in CLASSES}
    prof = {p: sum(v) / len(v) for p, v in per_profile.items()}
    worst_p = min(prof.items(), key=lambda kv: kv[1])
    answered = [ok for cf, ok in conf_records if cf > 0]
    return {
        "n": len(idxs),
        "balanced_accuracy": float(np.mean(list(recalls.values()))),
        "accuracy": sum(per_class_correct.values()) / max(1, len(idxs)),
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


def main() -> None:
    t_boot = time.perf_counter()
    rel, tf32_on = T.measure_matmul_precision()
    print(f"mlx {mx.__version__}  device={mx.default_device()}  "
          f"TF32 {'ON' if tf32_on else 'off'} (rel err {rel:.2e})  "
          f"dtype=bf16", flush=True)
    assert tf32_on, "expected TF32 ON (the run's own config)"
    try:
        mx.set_wired_limit(int(22 * GB))
    except Exception as e:
        print(f"WARNING set_wired_limit: {e!r}", flush=True)
    mx.set_cache_limit(2 * GB)

    # ---------------- corpus split (identical construction) ---------------
    manifest = json.loads((CORPUS / "manifest.json").read_text())
    row_samples = manifest["rowSamples"]
    rows = manifest["rows"]
    train_by_class: dict[str, list[int]] = {c: [] for c in CLASSES}
    eval_rows: list[dict] = []
    for row in rows:
        if row["role"] == "train":
            train_by_class[row["cls"]].append(row["row"])
        else:
            eval_rows.append(row)
    all_train = [r for ids in train_by_class.values() for r in ids]
    n_train, n_eval = len(all_train), len(eval_rows)
    train_pos = {rid: slot for slot, rid in enumerate(all_train)}
    eval_pos = {r["row"]: slot for slot, r in enumerate(eval_rows)}
    print(f"train {n_train}  eval {n_eval}  rowSamples {row_samples}  "
          f"maxDwell {MAX_DWELL}", flush=True)

    # ---------------- eval plans (from the TRAINERS' own code) ------------
    import v7_trainer as TT                                    # noqa: E402
    plans: dict[str, tuple] = {}
    plan_meta: dict[str, dict] = {}
    zero_plan = T.build_eval_plan(SEEDS[0], eval_rows, all_train, row_samples,
                                  MAX_DWELL, offset_mode="zero")
    plans["offset0"] = (zero_plan[1], zero_plan[2])
    plan_meta["offset0"] = zero_plan[3]
    for s in SEEDS:
        p_mlx = T.build_eval_plan(s, eval_rows, all_train, row_samples,
                                  MAX_DWELL, offset_mode="random")
        p_tor = TT.build_eval_plan(s, eval_rows, all_train, row_samples,
                                   MAX_DWELL, offset_mode="random")
        assert p_mlx[3]["plan_sha256"] == p_tor[3]["plan_sha256"], (
            f"torch/MLX eval plans differ at seed {s}")
        assert p_mlx[1] == p_tor[1] and p_mlx[2] == p_tor[2]
        plans[f"seed{s}"] = (p_mlx[1], p_mlx[2])
        plan_meta[f"seed{s}"] = p_mlx[3]
    # corrected queries, legacy offset-0 prototypes
    plans["mixedProtoZero"] = (plans[f"seed{SEEDS[0]}"][0], zero_plan[2])
    plan_meta["mixedProtoZero"] = {
        **plan_meta[f"seed{SEEDS[0]}"], "proto_offset_mode": "zero"}
    print("eval plans (torch == MLX verified):", flush=True)
    for k, m in plan_meta.items():
        print(f"  {k:<16} sha {m['plan_sha256'][:16]} "
              f"offsets [{m['eval_offset_min']}, {m['eval_offset_max']}] "
              f"mean {m['eval_offset_mean']:.0f}", flush=True)

    # burst-phase coverage: how many distinct GSM burst phases does each
    # protocol actually visit?  (offset 0 -> ssi mod 78000, three values)
    gsm_i = [i for i, r in enumerate(eval_rows) if r["cls"] == "gsm"]
    phase_cov = {}
    for name, (eoff, _p) in plans.items():
        ph = {(eval_rows[i]["startSampleIndex"] + eoff[eval_rows[i]["row"]])
              % GSM_BURST_PERIOD for i in gsm_i}
        phase_cov[name] = len(ph)
    print("distinct GSM burst phases sampled (of 672 gsm rows): "
          + "  ".join(f"{k}={v}" for k, v in phase_cov.items()), flush=True)

    # ---------------- resident stores -------------------------------------
    noisy = np.load(CORPUS / "noisy.npy", mmap_mode="r")
    rows_per_chunk = max(1, (512 << 20) // (row_samples * 8))
    t_conv = time.perf_counter()
    eval_store = mx.zeros((n_eval, row_samples), dtype=mx.complex64)
    eval_ids = [r["row"] for r in eval_rows]
    for i0 in range(0, n_eval, rows_per_chunk):
        ids = eval_ids[i0:i0 + rows_per_chunk]
        chunk = np.ascontiguousarray(noisy[ids])
        eval_store[i0:i0 + len(ids)] = mx.array(chunk)
        mx.eval(eval_store)
        del chunk
    train_iq = mx.zeros((n_train, 2, row_samples), dtype=mx.float16)
    for i0 in range(0, n_train, rows_per_chunk):
        ids = all_train[i0:i0 + rows_per_chunk]
        rc = np.asarray(noisy[ids])
        planes = np.stack((rc.real, rc.imag), axis=1).astype(np.float16)
        train_iq[i0:i0 + len(ids)] = mx.array(planes)
        mx.eval(train_iq)
        del rc, planes
    del noisy
    print(f"resident: eval {eval_store.nbytes/1e9:.2f} GB c64 + train "
          f"{train_iq.nbytes/1e9:.2f} GB fp16 in "
          f"{time.perf_counter()-t_conv:.0f}s", flush=True)

    # ---------------- model: final checkpoint, bf16 -----------------------
    model = T.TrainState(width=48, embed_dim=128)
    kind, _opt, sidecar = T.load_init_checkpoint(model, str(CKPT))
    mx.eval(model.parameters())
    pre_hash = T.params_sha256(model)
    model.net.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    ep = sidecar["episode"] if sidecar else "?"
    hash_ok = bool(sidecar and pre_hash == sidecar.get("params_sha256"))
    print(f"checkpoint {CKPT.name} kind={kind} episode={ep} "
          f"sidecar hash match={hash_ok}", flush=True)

    def eval_forward(xb):
        z, conf, _ = model.net.encode(
            spectrogram(rms_normalize(xb)).astype(mx.bfloat16))
        return z.astype(mx.float32), conf.astype(mx.float32)

    # ---------------- the evaluation --------------------------------------
    results: dict[str, dict] = {}
    correct_rows: dict[str, dict[str, list[int]]] = {}
    all_idx = list(range(n_eval))
    for name, (eoff, poff) in plans.items():
        results[name] = {}
        correct_rows[name] = {}
        t_cfg = time.perf_counter()
        for wname, length in DWELLS.items():
            mx.clear_cache()
            # prototypes -- verbatim trainer path (fp16 train cache), with
            # the frozen per-row prototype offsets
            protos_list = []
            for cls in CLASSES:
                ids = train_by_class[cls][:64]
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

            results[name][wname] = metrics(eval_rows, preds, confs, all_idx)
            correct_rows[name][wname] = [
                int(CLASSES[preds[i]] == eval_rows[i]["cls"])
                for i in range(n_eval)]
        print(f"  [{name}] {time.perf_counter()-t_cfg:.0f}s  " + "  ".join(
            f"{w} bal={results[name][w]['balanced_accuracy']:.4f} "
            f"minCell={results[name][w]['min_profile_recall']:.4f}"
            for w in DWELLS), flush=True)

    # ---------------- report ----------------------------------------------
    seed_keys = [f"seed{s}" for s in SEEDS]

    def spread(getter):
        out = {}
        for w in DWELLS:
            v = np.array([getter(results[k][w]) for k in seed_keys],
                         dtype=np.float64)
            out[w] = {"mean": float(v.mean()), "min": float(v.min()),
                      "max": float(v.max()), "std": float(v.std(ddof=1)),
                      "values": [float(x) for x in v]}
        return out

    summary = {
        "balanced_accuracy": spread(lambda r: r["balanced_accuracy"]),
        "min_profile_recall": spread(lambda r: r["min_profile_recall"]),
        "errors_total": spread(lambda r: r["errors_total"]),
        **{f"recall_{c}": spread(lambda r, c=c: r["per_class"][c])
           for c in CLASSES},
    }

    print("\n" + "=" * 92)
    print("FULL SPLIT (3264 rows), final G4 bf16 ep3000 weights")
    print("=" * 92)
    print(f"{'config':<16}" + "".join(
        f"{w+' bal':>12}{w+' minCell':>15}{w+' err':>10}" for w in DWELLS))
    for name in results:
        line = f"{name:<16}"
        for w in DWELLS:
            r = results[name][w]
            line += (f"{r['balanced_accuracy']:>12.4f}"
                     f"{r['min_profile_recall']:>15.4f}"
                     f"{r['errors_total']:>10d}")
        print(line)

    print("\nCORRECTED (5 offset seeds): mean [min, max]  vs  legacy offset0")
    for key, label in (("balanced_accuracy", "balanced accuracy"),
                       ("min_profile_recall", "min-cell recall"),
                       ("errors_total", "error count")):
        print(f"  {label}:")
        for w in DWELLS:
            s = summary[key][w]
            z = results["offset0"][w][key]
            fmt = "{:.0f}" if key == "errors_total" else "{:.4f}"
            print(f"    {w:<6} corrected " + fmt.format(s['mean'])
                  + " [" + fmt.format(s['min']) + ", "
                  + fmt.format(s['max']) + "]"
                  + f"  (+/-{(s['max']-s['min'])/2:.4g})"
                  + "   offset0 " + fmt.format(z))

    print("\nPER-CLASS RECALL (offset0 vs corrected mean [min, max]):")
    for w in DWELLS:
        print(f"  {w}:")
        for c in CLASSES:
            s = summary[f"recall_{c}"][w]
            z = results["offset0"][w]["per_class"][c]
            print(f"    {c:<10} offset0 {z:.4f}   corrected {s['mean']:.4f} "
                  f"[{s['min']:.4f}, {s['max']:.4f}]   d {s['mean']-z:+.4f}")

    print("\nPER-CLASS ERROR COUNTS (offset0 -> corrected seed"
          f"{SEEDS[0]}), n per class:")
    for w in DWELLS:
        z = results["offset0"][w]
        c1 = results[f"seed{SEEDS[0]}"][w]
        print(f"  {w}: " + "  ".join(
            f"{c}={z['per_class_errors'][c]}->{c1['per_class_errors'][c]}"
            f"/{z['per_class_total'][c]}" for c in CLASSES))

    print("\nWORST PROFILES:")
    for w in DWELLS:
        print(f"  {w}:")
        for name in ("offset0", f"seed{SEEDS[0]}", "mixedProtoZero"):
            print(f"    {name:<16} " + "  ".join(
                f"{p}={v:.3f}" for p, v in results[name][w]["worst_profiles"]))

    print("\nESCALATION @1ms (answered frac / answered acc):")
    for name in results:
        e = results[name]["1ms"]["escalation"]
        print(f"  {name:<16} {e['answered_frac']:.3f} / "
              f"{e['answered_accuracy']:.4f}")

    out = {
        "checkpoint": str(CKPT),
        "episode": ep,
        "sidecar_hash_match": hash_ok,
        "protocol": {
            "row_samples": row_samples,
            "max_dwell": MAX_DWELL,
            "offset_range": [0, row_samples - MAX_DWELL],
            "seeds": SEEDS,
            "plan_meta": plan_meta,
            "gsm_burst_phases_sampled": phase_cov,
        },
        "results": results,
        "summary_over_seeds": summary,
        "correct_by_config_dwell": correct_rows,
        "run_s": time.perf_counter() - t_boot,
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT} ({time.perf_counter()-t_boot:.0f}s total)",
          flush=True)


if __name__ == "__main__":
    main()
