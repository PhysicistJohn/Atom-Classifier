#!/usr/bin/env python
"""Is eval_rows[::3] systematically easier than the full eval split?

Loads the FINAL g4 bf16 checkpoint (so the weights are frozen and any
"the model changed between ep2000 and ep3000" explanation is excluded by
construction) and reproduces the trainer's evaluate() exactly:

  * bf16 net (model.net.set_dtype(mx.bfloat16)), TF32 default ON
  * prototypes from the fp16 TRAIN cache, offset 0, first 64 ids/class,
    chunk_n = 48 with zero padding -- the precision-asymmetry path
  * queries from the complex64 eval store, argmin cdist_sq to prototypes

Per-row predictions are computed ONCE over the full 3264-row eval split
(3264 % 48 == 0, so the trainer's chunking is padding-free there and the
per-row results are identical to what evaluate() sees).  Every subset
metric -- full, [0::3], [1::3], [2::3] -- is then derived from that same
per-row correctness vector, which makes the comparison exact rather than
approximate.  A faithful re-run of the chunked loop on eval_rows[::3]
(which DOES pad its last chunk) is done as a control to prove the model
is batch-composition independent.

Also dumps the composition forensics: per-profile / per-class row counts
per stride phase, and the impairment (SNR, multipath, CFO, phase noise,
IQ imbalance) distributions per phase.

Usage:
  .venv-training/bin/python subsample_bias_test.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# NOTE: deliberately do NOT set MLX_ENABLE_TF32 -- the run's own config is
# bf16 with TF32 default ON.  The trainer's module-top sys.argv peek only
# pins TF32 off when "--no-tf32" is in argv, which it is not here.
assert "--no-tf32" not in sys.argv

TRAINER = Path("/Users/johnelliott/PersonalGitHub/Atom-Classifier/training/"
               "zplane_ab/v2_full_variation/v7_probe/v7_trainer_mlx.py")
ART = Path("/Users/johnelliott/PersonalGitHub/Atom-Classifier/training/"
           "artifacts/mlx-g2")
CKPT = ART / "g4_bf16.safetensors"

sys.path.insert(0, str(TRAINER.parent))
import mlx.core as mx                                        # noqa: E402
import v7_trainer_mlx as T                                   # noqa: E402
from v7_model_mlx import spectrogram, rms_normalize           # noqa: E402
from v7_model_mlx import cdist_sq                             # noqa: E402

CLASSES = T.CLASSES
DWELLS = T.DWELLS
CORPUS = T.CORPUS
GB = 1 << 30
CHUNK_N = 48


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

    # ---------------- corpus (identical split construction) --------------
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
    print(f"train {n_train}  eval {n_eval}  "
          f"profiles {len({r['profile'] for r in rows})}", flush=True)

    # ---------------- composition / impairment forensics -----------------
    composition_report(eval_rows)

    # ---------------- resident stores (eval c64 + train fp16) ------------
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

    # ---------------- model: final checkpoint, bf16 ----------------------
    model = T.TrainState(width=48, embed_dim=128)
    kind, _opt, sidecar = T.load_init_checkpoint(model, str(CKPT))
    mx.eval(model.parameters())
    model.net.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    ep = sidecar["episode"] if sidecar else "?"
    print(f"checkpoint {CKPT.name} kind={kind} episode={ep} "
          f"params_sha256={T.params_sha256(model)[:16]}...", flush=True)
    if sidecar and sidecar.get("params_sha256"):
        # hash is over the as-loaded (pre-cast) params; recompute pre-cast
        m2 = T.TrainState(width=48, embed_dim=128)
        T.load_init_checkpoint(m2, str(CKPT))
        mx.eval(m2.parameters())
        got = T.params_sha256(m2)
        print(f"  sidecar hash match: {got == sidecar['params_sha256']}",
              flush=True)
        del m2

    def to_c(a):
        return a.astype(mx.bfloat16)

    def to_f(a):
        return a.astype(mx.float32)

    def eval_forward(xb):
        z, conf, _ = model.net.encode(to_c(spectrogram(rms_normalize(xb))))
        return to_f(z), to_f(conf)

    # ---------------- the evaluation ------------------------------------
    subsets = {
        "full": list(range(n_eval)),
        "[0::3]": list(range(0, n_eval, 3)),
        "[1::3]": list(range(1, n_eval, 3)),
        "[2::3]": list(range(2, n_eval, 3)),
    }
    results: dict[str, dict] = {k: {} for k in subsets}
    control: dict[str, dict] = {}
    correct_by_dwell: dict[str, np.ndarray] = {}

    for wname, length in DWELLS.items():
        mx.clear_cache()
        t_d = time.perf_counter()

        # prototypes: fp16 train cache, offset 0 -- verbatim trainer path
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
                seg = mx.take(train_iq, mx.array(pad), axis=0)[
                    :, :, :length].astype(mx.float32)
                xb = seg[:, 0] + 1j * seg[:, 1]
                zb, _ = eval_forward(xb)
                z_parts.append(zb[:len(chunk)])
            zc_all = mx.concatenate(z_parts, axis=0)
            protos_list.append(mx.sum(zc_all, axis=0) / len(ids))
        protos = mx.stack(protos_list, axis=0)
        mx.eval(protos)

        # per-row predictions over the FULL split (3264 % 48 == 0 -> the
        # trainer's chunk loop is padding-free here)
        preds = np.empty(n_eval, dtype=np.int32)
        confs = np.empty(n_eval, dtype=np.float64)
        for s0 in range(0, n_eval, CHUNK_N):
            chunk = eval_rows[s0:s0 + CHUNK_N]
            slots_np = np.array([eval_pos[r["row"]] for r in chunk],
                                dtype=np.int32)
            pad = np.concatenate(
                [slots_np, np.zeros(CHUNK_N - len(chunk), dtype=np.int32)])
            xb = mx.take(eval_store, mx.array(pad), axis=0)[:, :length]
            zb, confb = eval_forward(xb)
            pred = mx.argmax(-cdist_sq(zb[:len(chunk)], protos), axis=1)
            preds[s0:s0 + len(chunk)] = np.asarray(pred)
            confs[s0:s0 + len(chunk)] = np.asarray(confb[:len(chunk)])

        correct_by_dwell[wname] = np.array(
            [int(CLASSES[preds[i]] == eval_rows[i]["cls"])
             for i in range(n_eval)], dtype=np.int8)

        for name, idxs in subsets.items():
            results[name][wname] = metrics(eval_rows, preds, confs, idxs)

        # control: faithful chunked re-run on eval_rows[::3] (last chunk
        # padded 1088 % 48 = 32) to prove batch-composition independence
        sub = eval_rows[::3]
        cpred = np.empty(len(sub), dtype=np.int32)
        for s0 in range(0, len(sub), CHUNK_N):
            chunk = sub[s0:s0 + CHUNK_N]
            slots_np = np.array([eval_pos[r["row"]] for r in chunk],
                                dtype=np.int32)
            pad = np.concatenate(
                [slots_np, np.zeros(CHUNK_N - len(chunk), dtype=np.int32)])
            xb = mx.take(eval_store, mx.array(pad), axis=0)[:, :length]
            zb, _cf = eval_forward(xb)
            pred = mx.argmax(-cdist_sq(zb[:len(chunk)], protos), axis=1)
            cpred[s0:s0 + len(chunk)] = np.asarray(pred)
        agree = int((cpred == preds[0::3]).sum())
        control[wname] = {"n": len(sub), "agree": agree,
                          "identical": agree == len(sub)}
        print(f"  [{wname}] done in {time.perf_counter()-t_d:.1f}s; "
              f"padded-chunk control agrees {agree}/{len(sub)}", flush=True)

    # ---------------- report ---------------------------------------------
    print("\n" + "=" * 78)
    print("BALANCED ACCURACY / MIN-PROFILE RECALL  (final ep3000 weights)")
    print("=" * 78)
    hdr = f"{'subset':<10}{'n':>6}"
    for w in DWELLS:
        hdr += f"{w+' bal':>13}{w+' minP':>13}"
    print(hdr)
    for name in subsets:
        line = f"{name:<10}{len(subsets[name]):>6}"
        for w in DWELLS:
            r = results[name][w]
            line += f"{r['balanced_accuracy']:>13.4f}{r['min_profile_recall']:>13.4f}"
        print(line)

    print("\nDELTA vs full (subset - full):")
    for name in ("[0::3]", "[1::3]", "[2::3]"):
        line = f"{name:<10}{'':>6}"
        for w in DWELLS:
            db = (results[name][w]["balanced_accuracy"]
                  - results["full"][w]["balanced_accuracy"])
            dm = (results[name][w]["min_profile_recall"]
                  - results["full"][w]["min_profile_recall"])
            line += f"{db:>+13.4f}{dm:>+13.4f}"
        print(line)

    print("\nPER-CLASS RECALL, full vs [0::3]:")
    for w in DWELLS:
        print(f"  {w}:")
        for c in CLASSES:
            f_ = results["full"][w]["per_class"][c]
            s_ = results["[0::3]"][w]["per_class"][c]
            print(f"    {c:<10} full {f_:.4f}   [0::3] {s_:.4f}   "
                  f"d {s_-f_:+.4f}")

    print("\nWORST PROFILES, full vs each stride phase:")
    for w in DWELLS:
        print(f"  {w}:")
        for name in subsets:
            wp = results[name][w]["worst_profiles"]
            print(f"    {name:<8} " + "  ".join(
                f"{p}={v:.3f}" for p, v in wp))

    print("\nESCALATION @1ms (answered frac / answered acc):")
    for name in subsets:
        e = results[name]["1ms"]["escalation"]
        print(f"  {name:<8} {e['answered_frac']:.3f} / "
              f"{e['answered_accuracy']:.4f}")

    print("\nPADDED-CHUNK CONTROL (faithful evaluate() chunking on [::3]):")
    for w, c in control.items():
        print(f"  {w}: {c['agree']}/{c['n']} identical -> {c['identical']}")

    mechanism_report(eval_rows, correct_by_dwell)

    out = {
        "checkpoint": str(CKPT),
        "episode": ep,
        "results": results,
        "control": control,
        "correct_by_dwell": {w: v.tolist()
                             for w, v in correct_by_dwell.items()},
        "run_s": time.perf_counter() - t_boot,
    }
    (ART / "subsample_bias_test.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {ART / 'subsample_bias_test.json'} "
          f"({time.perf_counter()-t_boot:.0f}s total)", flush=True)


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
    prof = {p: sum(v) / len(v) for p, v in per_profile.items()}
    answered = [ok for cf, ok in conf_records if cf > 0]
    return {
        "n": len(idxs),
        "balanced_accuracy": float(np.mean(list(recalls.values()))),
        "per_class": recalls,
        "per_class_total": per_class_total,
        "min_profile_recall": min(prof.values()),
        "n_profiles": len(prof),
        "profile_recalls": prof,
        "worst_profiles": sorted(prof.items(), key=lambda kv: kv[1])[:5],
        "escalation": {
            "answered_frac": len(answered) / max(1, len(conf_records)),
            "answered_accuracy": (sum(answered) / len(answered))
            if answered else 0.0,
        },
    }


def mechanism_report(eval_rows, correct_by_dwell) -> None:
    """Why does the stride phase matter?  Row order inside a profile block
    is a fixed-stride slide across ONE long capture: row j of a block starts
    at j * stride samples.  Taking [k::3] triples that stride, i.e. it
    ALIASES the burst/frame cycle of every bursty profile."""
    print("\n" + "=" * 78)
    print("MECHANISM: stride-phase aliasing of the capture offset")
    print("=" * 78)
    n = len(eval_rows)
    # per-profile block bounds + startSampleIndex stride
    blocks: dict[str, list[int]] = {}
    for i, r in enumerate(eval_rows):
        blocks.setdefault(r["profile"], []).append(i)
    print(f"{'profile':<38}{'cls':<11}{'ssi stride':>12}"
          f"{'x3 stride':>12}")
    for p, idxs in blocks.items():
        ssi = [eval_rows[i]["startSampleIndex"] for i in idxs]
        d = sorted({b - a for a, b in zip(ssi, ssi[1:])})
        s = d[0] if len(d) == 1 else f"{d[:3]}"
        if eval_rows[idxs[0]]["cls"] in ("gsm", "bluetooth"):
            print(f"{p:<38}{eval_rows[idxs[0]]['cls']:<11}{str(s):>12}"
                  f"{str(s*3) if isinstance(s,int) else '?':>12}")

    for wname in DWELLS:
        ok = correct_by_dwell[wname]
        print(f"\n[{wname}] accuracy by position-in-block mod 3, by class:")
        print(f"  {'class':<11}{'full':>9}{'pos%3=0':>10}{'pos%3=1':>10}"
              f"{'pos%3=2':>10}")
        for c in CLASSES:
            sel = np.array([i for i in range(n)
                            if eval_rows[i]["cls"] == c])
            pos = np.array([i - blocks[eval_rows[i]["profile"]][0]
                            for i in sel])
            line = f"  {c:<11}{ok[sel].mean():>9.4f}"
            for k in range(3):
                m = pos % 3 == k
                line += f"{ok[sel][m].mean():>10.4f}"
            print(line)

    # GSM detail: per-block, the 96 outcomes laid out mod 3
    print("\n[1ms] GSM per-profile accuracy by position-in-block mod 3:")
    ok = correct_by_dwell["1ms"]
    print(f"  {'profile':<38}{'all96':>8}{'p0(32)':>9}{'p1(32)':>9}"
          f"{'p2(32)':>9}")
    for p, idxs in blocks.items():
        if eval_rows[idxs[0]]["cls"] != "gsm":
            continue
        a = ok[np.array(idxs)]
        print(f"  {p:<38}{a.mean():>8.3f}{a[0::3].mean():>9.3f}"
              f"{a[1::3].mean():>9.3f}{a[2::3].mean():>9.3f}")

    # is it the SIGNAL or the impairments?  measure clean-envelope duty
    # cycle inside the 1 ms window for the gsm rows
    print("\n[1ms] GSM burst occupancy in the 20000-sample window "
          "(clean.npy envelope, thr = 10% of row peak):")
    clean = np.load(CORPUS / "clean.npy", mmap_mode="r")
    gsm_idx = [i for i in range(n) if eval_rows[i]["cls"] == "gsm"]
    duty = np.empty(len(gsm_idx))
    for j, i in enumerate(gsm_idx):
        seg = np.asarray(clean[eval_rows[i]["row"], :DWELLS["1ms"]])
        e = np.abs(seg)
        duty[j] = float((e > 0.1 * e.max()).mean()) if e.max() > 0 else 0.0
    del clean
    pos = np.array([i - blocks[eval_rows[i]["profile"]][0]
                    for i in gsm_idx])
    okg = ok[np.array(gsm_idx)]
    print(f"  {'group':<12}{'n':>6}{'mean duty':>12}{'p50 duty':>11}"
          f"{'acc':>9}")
    print(f"  {'all gsm':<12}{len(duty):>6}{duty.mean():>12.4f}"
          f"{np.median(duty):>11.4f}{okg.mean():>9.4f}")
    for k in range(3):
        m = pos % 3 == k
        print(f"  {'pos%3='+str(k):<12}{m.sum():>6}{duty[m].mean():>12.4f}"
              f"{np.median(duty[m]):>11.4f}{okg[m].mean():>9.4f}")
    lo, hi = duty <= np.median(duty), duty > np.median(duty)
    print(f"  duty<=med  {lo.sum():>5}{'':>12}{'':>11}{okg[lo].mean():>9.4f}")
    print(f"  duty>med   {hi.sum():>5}{'':>12}{'':>11}{okg[hi].mean():>9.4f}")
    for k in range(3):
        m = pos % 3 == k
        print(f"  pos%3={k} duty deciles: "
              + " ".join(f"{np.percentile(duty[m], q):.3f}"
                         for q in (0, 25, 50, 75, 100)))

    # PROOF: the burst timeline inside a row is a pure function of
    # startSampleIndex mod 78000, and 78000 == 3 * the 26000-sample row
    # stride -- so the GSM rows cycle through exactly THREE burst phases
    # with period 3 in the block index.
    print("\n[proof] gsm-normal-burst: first clean burst onsets per row "
          "(row index j, ssi = 26000*j):")
    clean = np.load(CORPUS / "clean.npy", mmap_mode="r")
    blk = [i for i in range(n)
           if eval_rows[i]["profile"] == "gsm-normal-burst"]
    onsets_by_phase: dict[int, set] = {0: set(), 1: set(), 2: set()}
    print(f"  {'j':>3}{'ssi':>10}{'ssi%78000':>11}{'onsets<=200k':>36}"
          f"{'1ms':>5}{'2.5ms':>7}{'10ms':>6}")
    for j in list(range(9)) + [30, 93, 95]:
        i = blk[j]
        x = np.abs(np.asarray(clean[eval_rows[i]["row"], :200_000]))
        on = x > 0.1 * x.max()
        ons = [int(o) for o in
               (np.flatnonzero(np.diff(on.astype(np.int8)) == 1) + 1)
               if o > 50]
        # merge onsets closer than 1000 samples (intra-burst dropouts)
        merged = [ons[0]] if ons else []
        for o in ons[1:]:
            if o - merged[-1] > 1000:
                merged.append(o)
        onsets_by_phase[j % 3].add(tuple(merged[:3]))
        ssi = eval_rows[i]["startSampleIndex"]
        hits = [any(o < L for o in merged) for L in DWELLS.values()]
        print(f"  {j:>3}{ssi:>10}{ssi % 78000:>11}"
              f"{str(merged[:3]):>36}"
              + "".join(f"{'HIT' if h else '-':>6}" for h in hits))
    del clean
    for k in range(3):
        print(f"  phase j%3={k}: {len(onsets_by_phase[k])} distinct onset "
              f"pattern(s) -> {sorted(onsets_by_phase[k])}")
    print("  => burst timeline is a function of ssi mod 78000 = "
          "3 x row stride; [::3] locks onto ONE burst phase.")

    # ep2000 subsampled report vs these ep3000 weights on the same subset
    hist = json.loads((ART / "g4_bf16.json").read_text()).get("history", [])
    print("\n[ep2000 vs ep3000] trainer's reported evals vs THIS run:")
    for h in hist:
        print(f"  reported ep{h['episode']:<5} "
              f"({h.get('eval_rows_used')} rows): " + "  ".join(
                  f"{w} {h[w]['balanced_accuracy']:.3f}/"
                  f"{h[w]['min_profile_recall']:.3f}" for w in DWELLS))
    print("=" * 78, flush=True)


def composition_report(eval_rows) -> None:
    """Does striding by 3 interact with the row ordering?"""
    print("\n" + "-" * 78)
    print("COMPOSITION FORENSICS: eval split vs stride phases")
    print("-" * 78)
    n = len(eval_rows)
    profs = [r["profile"] for r in eval_rows]
    # contiguous profile blocks?
    blocks = []
    cur, start = profs[0], 0
    for i, p in enumerate(profs):
        if p != cur:
            blocks.append((cur, start, i))
            cur, start = p, i
    blocks.append((cur, start, n))
    sizes = sorted({b[2] - b[1] for b in blocks})
    print(f"rows {n}; profile blocks {len(blocks)}; block sizes {sizes} "
          f"(contiguous: {len(blocks) == len({p for p in profs})})")
    print(f"block size % 3 == 0 for all blocks: "
          f"{all((b[2]-b[1]) % 3 == 0 for b in blocks)}  "
          f"block starts % 3: "
          f"{sorted({b[1] % 3 for b in blocks})}")

    phases = {f"[{k}::3]": list(range(k, n, 3)) for k in range(3)}
    phases = {"full": list(range(n)), **phases}
    # per-profile counts
    print("\nper-profile row counts:")
    for name, idxs in phases.items():
        cnt: dict[str, int] = {}
        for i in idxs:
            cnt[eval_rows[i]["profile"]] = cnt.get(
                eval_rows[i]["profile"], 0) + 1
        vals = sorted(set(cnt.values()))
        print(f"  {name:<8} n={len(idxs):<5} profiles={len(cnt):<3} "
              f"counts/profile={vals}")
    print("\nper-class row counts:")
    for name, idxs in phases.items():
        cnt = {}
        for i in idxs:
            cnt[eval_rows[i]["cls"]] = cnt.get(eval_rows[i]["cls"], 0) + 1
        print(f"  {name:<8} " + "  ".join(f"{c}={cnt.get(c,0)}"
                                          for c in CLASSES))

    keys = ["snrDb", "multipathTaps", "cfoCyclesPerSample", "phaseNoiseStd",
            "iqGainImbalance", "iqPhaseImbalance"]
    print("\nimpairment distributions (mean / p10 / p50 / min):")
    for k in keys:
        v = np.array([abs(r["impairments"][k]) if k == "cfoCyclesPerSample"
                      else r["impairments"][k] for r in eval_rows],
                     dtype=np.float64)
        print(f"  {k}:")
        for name, idxs in phases.items():
            s = v[np.array(idxs)]
            print(f"    {name:<8} mean {s.mean():>10.4g}  p10 "
                  f"{np.percentile(s,10):>10.4g}  p50 "
                  f"{np.percentile(s,50):>10.4g}  min {s.min():>10.4g}")
    # hardest-decile SNR count (low SNR = hard rows)
    snr = np.array([r["impairments"]["snrDb"] for r in eval_rows])
    thr = np.percentile(snr, 10)
    print(f"\nrows below the split-wide SNR p10 ({thr:.2f} dB):")
    for name, idxs in phases.items():
        s = snr[np.array(idxs)]
        print(f"  {name:<8} {(s <= thr).sum():>4} / {len(idxs)} "
              f"= {(s<=thr).mean()*100:.2f}%")
    print("-" * 78 + "\n", flush=True)


if __name__ == "__main__":
    main()
