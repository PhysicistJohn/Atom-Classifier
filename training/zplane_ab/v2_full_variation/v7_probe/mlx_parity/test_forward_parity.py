"""G1 forward-parity harness: converted ep-2000 checkpoint, MLX vs torch CPU.

Per docs/mlx-port-plan.md (G1) and the frozen reference v7_trainer.py:

  1. Convert v7_ab_recon_on.ep2000.pt with convert.py (verify mode: includes
     the torch->mlx->torch bitwise roundtrip), load into V7NetMLX with
     strict=True and the conf.<i>.* -> conf.layers.<i>.* remap.
  2. Frozen real batches: the first 6 eval rows per class in manifest order
     (42 rows), windows at each dwell (20k/50k/200k samples from sample 0),
     pipeline rms_normalize -> spectrogram -> encode on BOTH frameworks
     (torch on CPU).  Gates: z atol 1e-4, conf atol 1e-4, recon (at the
     clean/cropped grid, <=50k samples) atol 1e-3.
  3. Full 1 ms eval-split protocol on MLX: per-class prototypes from the
     first-64 TRAIN rows read through an fp16 round-trip (simulating the
     trainer RAM cache: planes .astype(float16) -> float32 -> complex64);
     query rows full complex64 from noisy.npy; prediction = argmax of
     -cdist_sq.  Torch side computed identically (CPU, same fp16-roundtrip
     prototypes).  Gates: argmax agreement >= 99.9%; confusion-matrix delta
     reported (max abs per cell).  Plan sub-check: fp16-path prototypes must
     DIFFER from full-precision-path prototypes, and the fp16 ones must
     match torch's fp16 ones.
  4. Reports per-check max errors, agreement %, MLX eval-pass wall time and
     mx.get_peak_memory().

Run:
  .venv-training/bin/python test_forward_parity.py
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

# CRITICAL: MLX 0.32.0 defaults fp32 GPU matmul to TF32 on this M5 Max,
# which broke 12 G0 parity checks until disabled.  Must be set BEFORE any
# mlx import; asserted at startup below.
os.environ.setdefault("MLX_ENABLE_TF32", "0")

import numpy as np
import mlx.core as mx

HERE = Path(__file__).resolve().parent
V7_DIR = HERE.parent
sys.path.insert(0, str(V7_DIR))

from v7_model_mlx import (  # noqa: E402
    V7NetMLX, spectrogram, rms_normalize, cdist_sq,
)

CORPUS = Path(os.environ.get(
    "CORPUS_DIR",
    V7_DIR.parent.parent.parent / "artifacts/longdwell-production-corpus",
))
CKPT_PT = V7_DIR / "v7_ab_recon_on.ep2000.pt"
CKPT_ST = HERE / "v7_ab_recon_on.ep2000.safetensors"

CLASSES = ("am", "bluetooth", "cw", "dsss", "fm", "gsm", "ofdm")
DWELLS = {"1ms": 20_000, "2.5ms": 50_000, "10ms": 200_000}
NFFT, HOP = 64, 32
PROTO_CAP = 64      # first-64 train rows per class (trainer eval protocol)
CHUNK = 48          # trainer eval chunking

RESULTS: list[tuple[str, bool, float, float, str]] = []


def record(name: str, err: float, tol: float, note: str = "") -> None:
    ok = bool(err <= tol)
    RESULTS.append((name, ok, err, tol, note))
    print(f"[{'PASS' if ok else 'FAIL'}] {name:48s} "
          f"max|err|={err:.3e}  tol={tol:.0e}  {note}", flush=True)


def frames_of(n: int) -> int:
    return (n - NFFT) // HOP + 1


def assert_non_tf32() -> None:
    """Tiny fp32 GPU matmul vs numpy float64 reference; rel err < 1e-5."""
    rng = np.random.default_rng(123)
    a = rng.standard_normal((64, 128)).astype(np.float32)
    b = rng.standard_normal((128, 7)).astype(np.float32)
    ref = a.astype(np.float64) @ b.astype(np.float64)
    out = np.asarray(mx.array(a) @ mx.array(b)).astype(np.float64)
    rel = float(np.abs(out - ref).max() / np.abs(ref).max())
    assert rel < 1e-5, (
        f"TF32 appears ACTIVE: fp32 matmul rel err {rel:.3e} >= 1e-5. "
        f"MLX_ENABLE_TF32={os.environ.get('MLX_ENABLE_TF32')!r}")
    print(f"non-TF32 asserted: fp32 matmul rel err {rel:.3e} < 1e-5", flush=True)


# --------------------------------------------------------------- step 1: convert
def convert_and_load() -> V7NetMLX:
    spec = importlib.util.spec_from_file_location("convert", HERE / "convert.py")
    convert = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(convert)
    # verify = torch2mlx + mlx2torch + bitwise roundtrip assertion (G1 item)
    convert.verify(str(CKPT_PT), str(CKPT_ST))

    weights, meta = mx.load(str(CKPT_ST), return_metadata=True)
    remapped = []
    for k, v in weights.items():
        if k.startswith("conf."):
            _, rest = k.split(".", 1)
            k = f"conf.layers.{rest}"
        remapped.append((k, v))
    cfg = json.loads(meta["config"])
    net = V7NetMLX(width=int(cfg["width"]), embed_dim=int(cfg["embed_dim"]))
    net.load_weights(remapped, strict=True)
    net.eval()
    print(f"loaded MLX net (width={cfg['width']}, embed_dim={cfg['embed_dim']}, "
          f"episode={meta.get('episode')})", flush=True)
    return net


def load_torch_net():
    import torch
    import v7_trainer

    ck = torch.load(str(CKPT_PT), map_location="cpu", weights_only=False)
    cfg = ck["config"]
    net = v7_trainer.V7Net(width=int(cfg["width"]),
                           embed_dim=int(cfg["embed_dim"]))
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net


# ------------------------------------------------------- step 2: frozen batches
def pick_frozen_rows(rows: list[dict]) -> list[int]:
    """First 6 eval rows per class in manifest order, concatenated in
    CLASSES order -> 42 deterministic row ids."""
    per: dict[str, list[int]] = {c: [] for c in CLASSES}
    for r in rows:
        if r["role"] != "train" and len(per[r["cls"]]) < 6:
            per[r["cls"]].append(r["row"])
    picked = [rid for c in CLASSES for rid in per[c]]
    assert len(picked) == 42 and all(len(per[c]) == 6 for c in CLASSES), \
        f"frozen-batch selection broke: {[len(per[c]) for c in CLASSES]}"
    return picked


def test_frozen_batches(net_mlx: V7NetMLX, net_t, noisy) -> None:
    import torch
    import v7_trainer

    rows42 = pick_frozen_rows_cached
    for wname, length in DWELLS.items():
        xb = np.stack([np.asarray(noisy[r, :length]) for r in rows42])
        assert xb.dtype == np.complex64, xb.dtype
        t_clean = frames_of(min(length, 50_000))  # clean/cropped recon grid

        # torch CPU reference
        with torch.no_grad():
            xt = v7_trainer.rms_normalize(torch.from_numpy(xb))
            spec_t = v7_trainer.spectrogram(xt)
            z_t, conf_t, h_t = net_t.encode(spec_t)
            recon_t = net_t.reconstruct(
                h_t, (xb.shape[0], 3, t_clean, spec_t.shape[3]))
        z_t = z_t.numpy(); conf_t = conf_t.numpy()
        recon_t = np.ascontiguousarray(
            recon_t.numpy().transpose(0, 2, 3, 1))          # NCHW -> NHWC

        # MLX
        spec_m = spectrogram(rms_normalize(mx.array(xb)))
        z_m, conf_m, h_m = net_mlx.encode(spec_m)
        recon_m = net_mlx.reconstruct(h_m, t_clean, spec_m.shape[2])
        mx.eval(z_m, conf_m, recon_m)

        record(f"frozen {wname} z",
               float(np.abs(np.asarray(z_m) - z_t).max()), 1e-4,
               f"B=42 T={spec_t.shape[2]}")
        record(f"frozen {wname} conf",
               float(np.abs(np.asarray(conf_m) - conf_t).max()), 1e-4)
        r_m = np.asarray(recon_m)
        if r_m.shape != recon_t.shape:
            record(f"frozen {wname} recon", math.inf, 1e-3,
                   f"SHAPE {r_m.shape} vs {recon_t.shape}")
        else:
            record(f"frozen {wname} recon",
                   float(np.abs(r_m - recon_t).max()), 1e-3,
                   f"grid t={t_clean}")


# --------------------------------------------- step 3: full 1 ms eval protocol
def fp16_window(noisy, rid: int, length: int) -> np.ndarray:
    """Simulate the trainer RAM cache: planes .astype(float16) -> float32
    -> complex64 (v7_trainer noisy_ram + train_window path)."""
    seg = np.asarray(noisy[rid, :length])
    re = seg.real.astype(np.float16).astype(np.float32)
    im = seg.imag.astype(np.float16).astype(np.float32)
    return (re + 1j * im).astype(np.complex64)


def build_protos_mlx(net: V7NetMLX, noisy, train_by_class, length: int,
                     window_fn) -> np.ndarray:
    protos_acc = np.zeros((len(CLASSES), net.embed.weight.shape[0]),
                          dtype=np.float32)
    protos_n = np.zeros(len(CLASSES), dtype=np.float32)
    for ci, cls in enumerate(CLASSES):
        ids = train_by_class[cls][:PROTO_CAP]
        for start in range(0, len(ids), CHUNK):
            chunk = ids[start:start + CHUNK]
            xb = np.stack([window_fn(noisy, r, length) for r in chunk])
            zb, _, _ = net.encode(spectrogram(rms_normalize(mx.array(xb))))
            mx.eval(zb)
            protos_acc[ci] += np.asarray(mx.sum(zb, axis=0))
            protos_n[ci] += len(chunk)
    return protos_acc / protos_n[:, None]


def build_protos_torch(net_t, noisy, train_by_class, length: int,
                       window_fn) -> np.ndarray:
    import torch
    import v7_trainer

    protos_acc = np.zeros((len(CLASSES), net_t.embed.out_features),
                          dtype=np.float32)
    protos_n = np.zeros(len(CLASSES), dtype=np.float32)
    with torch.no_grad():
        for ci, cls in enumerate(CLASSES):
            ids = train_by_class[cls][:PROTO_CAP]
            for start in range(0, len(ids), CHUNK):
                chunk = ids[start:start + CHUNK]
                xb = np.stack([window_fn(noisy, r, length) for r in chunk])
                zb, _, _ = net_t.encode(v7_trainer.spectrogram(
                    v7_trainer.rms_normalize(torch.from_numpy(xb))))
                protos_acc[ci] += zb.sum(dim=0).numpy()
                protos_n[ci] += len(chunk)
    return protos_acc / protos_n[:, None]


def test_eval_protocol(net_mlx: V7NetMLX, net_t, noisy,
                       train_by_class, eval_rows) -> None:
    import torch

    length = DWELLS["1ms"]
    cls_idx = {c: i for i, c in enumerate(CLASSES)}
    targets = np.array([cls_idx[r["cls"]] for r in eval_rows])

    # ---- MLX eval pass (timed; peak memory scoped to this pass) ----------
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    protos_mlx = build_protos_mlx(net_mlx, noisy, train_by_class, length,
                                  fp16_window)
    protos_mx = mx.array(protos_mlx)
    preds_mlx = np.empty(len(eval_rows), dtype=np.int64)
    for start in range(0, len(eval_rows), CHUNK):
        chunk = eval_rows[start:start + CHUNK]
        xb = np.stack([np.asarray(noisy[r["row"], :length]) for r in chunk])
        zb, _, _ = net_mlx.encode(spectrogram(rms_normalize(mx.array(xb))))
        pred = mx.argmax(-cdist_sq(zb, protos_mx), axis=1)
        mx.eval(pred)
        preds_mlx[start:start + len(chunk)] = np.asarray(pred)
    mlx_wall = time.perf_counter() - t0
    mlx_peak = mx.get_peak_memory()
    print(f"MLX eval pass: {mlx_wall:.2f} s wall "
          f"({len(eval_rows)} queries + {len(CLASSES)}x{PROTO_CAP} proto rows), "
          f"peak memory {mlx_peak / 1e9:.3f} GB ({mlx_peak} bytes)", flush=True)

    # ---- plan sub-check: fp16-cache path vs full-precision path ----------
    full_window = lambda n, r, L: np.asarray(n[r, :L])
    protos_mlx_fp32 = build_protos_mlx(net_mlx, noisy, train_by_class,
                                       length, full_window)
    d_paths = float(np.abs(protos_mlx - protos_mlx_fp32).max())
    record("protos fp16 path != fp32 path (must differ)",
           float(d_paths == 0.0), 0.0, f"max|d|={d_paths:.3e}")

    # ---- torch side, computed identically (CPU, fp16-roundtrip protos) ---
    protos_t = build_protos_torch(net_t, noisy, train_by_class, length,
                                  fp16_window)
    record("protos fp16 path mlx vs torch",
           float(np.abs(protos_mlx - protos_t).max()), 1e-4)

    import v7_trainer
    protos_tt = torch.from_numpy(protos_t)
    preds_t = np.empty(len(eval_rows), dtype=np.int64)
    with torch.no_grad():
        for start in range(0, len(eval_rows), CHUNK):
            chunk = eval_rows[start:start + CHUNK]
            xb = np.stack([np.asarray(noisy[r["row"], :length])
                           for r in chunk])
            zb, _, _ = net_t.encode(v7_trainer.spectrogram(
                v7_trainer.rms_normalize(torch.from_numpy(xb))))
            pred = (-torch.cdist(zb, protos_tt) ** 2).argmax(dim=1)
            preds_t[start:start + len(chunk)] = pred.numpy()

    # ---- agreement + confusion delta -------------------------------------
    agree = float((preds_mlx == preds_t).mean())
    n_dis = int((preds_mlx != preds_t).sum())
    record("1ms eval argmax agreement >= 99.9%",
           1.0 - agree, 1.0 - 0.999,
           f"agreement={agree * 100:.4f}% ({n_dis}/{len(eval_rows)} differ)")

    conf_mlx = np.zeros((7, 7), dtype=np.int64)
    conf_t = np.zeros((7, 7), dtype=np.int64)
    np.add.at(conf_mlx, (targets, preds_mlx), 1)
    np.add.at(conf_t, (targets, preds_t), 1)
    delta = np.abs(conf_mlx - conf_t)
    print("confusion delta (|mlx - torch| per cell, rows=true cls "
          f"{CLASSES}):\n{delta}", flush=True)
    record("confusion-matrix delta (max abs per cell)",
           float(delta.max()), 2.0,  # plan G1: <= 2/cell
           f"sum|delta|={int(delta.sum())}")

    # informational: balanced accuracy both sides
    for tag, p in (("mlx", preds_mlx), ("torch", preds_t)):
        rec = [float((p[targets == i] == i).mean()) for i in range(7)]
        print(f"  {tag:5s} 1ms balanced accuracy = {np.mean(rec):.4f}  "
              f"per-class {[round(r, 3) for r in rec]}", flush=True)

    if n_dis:
        idx = np.nonzero(preds_mlx != preds_t)[0][:20]
        print("  first disagreements (eval-split idx, true, mlx, torch):",
              [(int(i), CLASSES[targets[i]], CLASSES[preds_mlx[i]],
                CLASSES[preds_t[i]]) for i in idx], flush=True)


# ------------------------------------------------------------------------ main
pick_frozen_rows_cached: list[int] = []


def main() -> int:
    global pick_frozen_rows_cached
    print(f"mlx {mx.__version__}  device={mx.default_device()}  "
          f"MLX_ENABLE_TF32={os.environ.get('MLX_ENABLE_TF32')!r}", flush=True)
    assert_non_tf32()

    import torch
    torch.manual_seed(0)
    print(f"torch {torch.__version__} (CPU inference)", flush=True)

    manifest = json.loads((CORPUS / "manifest.json").read_text())
    noisy = np.load(CORPUS / "noisy.npy", mmap_mode="r")
    rows = manifest["rows"]
    train_by_class: dict[str, list[int]] = {c: [] for c in CLASSES}
    eval_rows: list[dict] = []
    for r in rows:
        if r["role"] == "train":
            train_by_class[r["cls"]].append(r["row"])
        else:
            eval_rows.append(r)
    print(f"corpus: train {sum(len(v) for v in train_by_class.values())}  "
          f"eval {len(eval_rows)}", flush=True)

    net_mlx = convert_and_load()
    net_t = load_torch_net()
    pick_frozen_rows_cached = pick_frozen_rows(rows)

    t_frozen = time.perf_counter()
    test_frozen_batches(net_mlx, net_t, noisy)
    print(f"frozen-batch checks: {time.perf_counter() - t_frozen:.1f} s "
          f"(incl. torch CPU at T=6249)", flush=True)

    test_eval_protocol(net_mlx, net_t, noisy, train_by_class, eval_rows)

    n_pass = sum(1 for _, ok, *_ in RESULTS if ok)
    print("\n=== G1 summary ===")
    for name, ok, err, tol, note in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name:48s} "
              f"err={err:.3e} tol={tol:.0e} {note}")
    print(f"{n_pass}/{len(RESULTS)} passed")
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
