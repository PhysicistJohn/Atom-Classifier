"""G0 unit-parity harness: MLX primitives vs frozen torch fixtures.

Loads the .npz fixtures produced by gen_torch_fixtures.py (torch 2.13.0 CPU,
fp32/complex64) and asserts the MLX counterparts in v7_model_mlx.py match
within the per-test tolerances fixed by the plan (docs/mlx-port-plan.md, G0).

Fixture layouts are torch-native NCHW; MLX is NHWC — this harness owns the
transposes (fixture .transpose(0, 2, 3, 1) to compare against MLX output).

Run:
  .venv-training/bin/python test_unit_parity.py
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
import tempfile
from pathlib import Path

# G0-critical: MLX 0.32.0 defaults fp32 GPU matmul to TF32 (10-bit mantissa)
# on this M5 Max — measured 2.7e-2 abs error on a 64x128x7 fp32 matmul vs
# 4.6e-6 with TF32 off.  With the default ON, cdist_sq missed the 1e-4
# relative gate (1.05e-4) and full-model z/conf/recon missed theirs by
# 2-18x.  The fidelity contract (docs/mlx-port-plan.md section 1) requires
# full fp32; the trainer port must set this too, BEFORE mlx initializes.
os.environ.setdefault("MLX_ENABLE_TF32", "0")

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

HERE = Path(__file__).resolve().parent
V7_DIR = HERE.parent
sys.path.insert(0, str(V7_DIR))

from v7_model_mlx import (  # noqa: E402
    V7NetMLX, spectrogram, rms_normalize, nearest_resize, cdist_sq,
)

FIX = HERE / "fixtures"
DWELL_KEYS = ("1ms", "2p5ms", "10ms")
DWELL_N = {"1ms": 20_000, "2p5ms": 50_000, "10ms": 200_000}

RESULTS: list[tuple[str, bool, float, float, str]] = []


def record(name: str, err: float, tol: float, note: str = "") -> None:
    ok = bool(err <= tol)
    RESULTS.append((name, ok, err, tol, note))
    print(f"[{'PASS' if ok else 'FAIL'}] {name:44s} "
          f"max|err|={err:.3e}  tol={tol:.0e}  {note}")


def nchw_to_nhwc(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a.transpose(0, 2, 3, 1))


# ---------------------------------------------------------------- 1. spectrogram
def test_spectrogram() -> None:
    d = np.load(FIX / "spectrogram.npz")
    for key in DWELL_KEYS:
        x = mx.array(d[f"input_{key}"])
        out = np.asarray(spectrogram(x))                    # NHWC [2, T, 33, 3]
        ref = nchw_to_nhwc(d[f"output_{key}"])              # NCHW -> NHWC
        if out.shape != ref.shape:
            record(f"spectrogram {key}", math.inf, 1e-4,
                   f"SHAPE {out.shape} vs {ref.shape}")
            continue
        record(f"spectrogram {key}", float(np.abs(out - ref).max()), 1e-4,
               f"T={ref.shape[1]}")


# -------------------------------------------------------------- 2. rms_normalize
def test_rms_normalize() -> None:
    d = np.load(FIX / "rms_normalize.npz")
    out = np.asarray(rms_normalize(mx.array(d["input"])))
    err = float(np.abs(out - d["output"]).max())            # complex |diff|
    record("rms_normalize (incl 1e-12 row)", err, 1e-6)


# ------------------------------------------------------------- 3. nearest_resize
def test_nearest_resize() -> None:
    d = np.load(FIX / "nearest_resize.npz")
    names = sorted(k[3:] for k in d.files if k.startswith("in_"))
    worst = 0.0
    bad = []
    for name in names:
        x = mx.array(nchw_to_nhwc(d[f"in_{name}"]))
        ref = nchw_to_nhwc(d[f"out_{name}"])
        out = np.asarray(nearest_resize(x, (ref.shape[1], ref.shape[2])))
        if out.shape != ref.shape:
            bad.append(f"{name}:shape{out.shape}")
            worst = math.inf
            continue
        e = float(np.abs(out - ref).max())
        worst = max(worst, e)
        if e != 0.0:
            bad.append(f"{name}:{e:.1e}")
    note = f"{len(names)} cases exact" if not bad else " ".join(bad)
    record("nearest_resize (10 cases, exact)", worst, 0.0, note)


# ------------------------------------------------------------------ 4. GroupNorm
def test_groupnorm() -> None:
    d = np.load(FIX / "groupnorm.npz")
    gn = nn.GroupNorm(8, 48, pytorch_compatible=True)
    gn.weight = mx.array(d["weight"])
    gn.bias = mx.array(d["bias"])
    out = np.asarray(gn(mx.array(nchw_to_nhwc(d["input"]))))
    ref = nchw_to_nhwc(d["output"])
    record("groupnorm pytorch_compatible+affine",
           float(np.abs(out - ref).max()), 1e-5)


# -------------------------------------------------------------------- 5. cdist^2
def test_cdist_sq() -> None:
    d = np.load(FIX / "cdist_sq.npz")
    for n in (35, 48, 14):
        q = mx.array(d[f"q_{n}x7"])
        p = mx.array(d[f"p_{n}x7"])
        out = -np.asarray(cdist_sq(q, p))                   # fixture is -cdist^2
        ref = d[f"out_{n}x7"]
        err = float(np.abs(out - ref).max())
        # tolerance is relative on the 64-scale values: 1e-4 * 64
        record(f"cdist_sq {n}x7", err, 1e-4 * 64.0,
               f"rel={err / 64.0:.2e}")


# ------------------------------------------------------------- 6. normalize * 4
def test_normalize_scale() -> None:
    d = np.load(FIX / "normalize_scale.npz")
    z = mx.array(d["input"])
    out = z / mx.maximum(mx.linalg.norm(z, axis=-1, keepdims=True),
                         1e-12) * 4.0
    record("normalize*4", float(np.abs(np.asarray(out) - d["output"]).max()),
           1e-6)


# --------------------------------------------------------------------- 7. pooling
def test_pooling() -> None:
    d = np.load(FIX / "pooling.npz")
    x = mx.array(nchw_to_nhwc(d["input"]))                  # [3, 13, 5, 192]
    mean = np.asarray(mx.mean(x, axis=(1, 2)))
    amax = np.asarray(mx.max(x, axis=(1, 2)))
    record("pooling mean (1,2)",
           float(np.abs(mean - d["mean_out"]).max()), 1e-6)
    record("pooling amax (1,2)",
           float(np.abs(amax - d["amax_out"]).max()), 1e-6)


# ---------------------------------------------------------------------- 8. losses
def test_losses() -> None:
    d = np.load(FIX / "losses.npz")
    cd = np.load(FIX / "cdist_sq.npz")

    # temperature path: scale = clamp(exp(log_scale), 1e-3, 100).  The scale
    # VALUE is compared at 1e-5 (mlx GPU exp is 1 ulp above torch CPU exp:
    # 10.000001 vs 10.0 — transcendental kernel difference, 9.5e-7 abs).
    # The multiply itself is compared with the fixture's own scale so the
    # 1-ulp exp delta is not amplified by the 640-magnitude logits.
    scale = mx.clip(mx.exp(mx.array(math.log(10.0))), 1e-3, 100)
    record("temperature clamp+exp scale value",
           float(abs(float(scale) - float(d["ce_scale"]))), 1e-5,
           f"mlx={float(scale)!r} torch={float(d['ce_scale'])!r}")
    logits_t = mx.array(cd["out_35x7"]) * mx.array(d["ce_scale"])
    record("temperature-scaled logits (fixture scale)",
           float(np.abs(np.asarray(logits_t) - d["ce_logits"]).max()), 1e-5)

    ce = nn.losses.cross_entropy(
        mx.array(d["ce_logits"]), mx.array(d["ce_target"].astype(np.int32)),
        reduction="mean")
    record("cross_entropy", float(abs(float(ce) - float(d["ce_out"]))), 1e-5,
           f"ref={float(d['ce_out']):.6f}")

    bce = nn.losses.binary_cross_entropy(
        mx.array(d["bce_logits"]), mx.array(d["bce_targets"]),
        with_logits=True, reduction="mean")
    record("bce_with_logits", float(abs(float(bce) - float(d["bce_out"]))),
           1e-5, f"ref={float(d['bce_out']):.6f}")

    l1 = nn.losses.l1_loss(mx.array(d["l1_a"]), mx.array(d["l1_b"]),
                           reduction="mean")
    record("l1_loss", float(abs(float(l1) - float(d["l1_out"]))), 1e-5,
           f"ref={float(d['l1_out']):.6f}")


# ------------------------------------------------------- 9. optimizer + schedule
def make_multi_opt(sched) -> optim.MultiOptimizer:
    """Trainer split: wd=0 sub-optimizer selected by filter (log_scale role,
    here p2), wd=1e-4 fallback for the net params (here p1).  Both AdamW
    betas (0.9, 0.999), eps 1e-8, bias_correction=True."""
    return optim.MultiOptimizer(
        [optim.AdamW(learning_rate=sched, betas=[0.9, 0.999], eps=1e-8,
                     weight_decay=0.0, bias_correction=True),
         optim.AdamW(learning_rate=sched, betas=[0.9, 0.999], eps=1e-8,
                     weight_decay=1e-4, bias_correction=True)],
        [lambda path, g: "p2" in path])


def test_optimizer_steps() -> None:
    d = np.load(FIX / "optimizer.npz")
    sched = optim.cosine_decay(float(d["lr0"]), int(d["t_max"]), end=0.0)
    opt = make_multi_opt(sched)
    params = {"p1": mx.array(d["p1_init"]), "p2": mx.array(d["p2_init"])}
    opt.init(params)

    worst_p = 0.0
    worst_lr = 0.0
    per_step = []
    for i in range(5):
        grads = {"p1": mx.array(d["g1_steps"][i]),
                 "p2": mx.array(d["g2_steps"][i])}
        params = opt.apply_gradients(grads, params)
        mx.eval(params, opt.state)
        e1 = float(np.abs(np.asarray(params["p1"]) - d["p1_steps"][i]).max())
        e2 = float(np.abs(np.asarray(params["p2"]) - d["p2_steps"][i]).max())
        # realized lr used at this step (schedule evaluated pre-increment)
        lr_used = float(opt.optimizers[0].learning_rate)
        lr_used_fb = float(opt.optimizers[1].learning_rate)
        elr = max(abs(lr_used - d["lr_at_step"][i]),
                  abs(lr_used_fb - d["lr_at_step"][i]))
        worst_p = max(worst_p, e1, e2)
        worst_lr = max(worst_lr, elr)
        per_step.append(max(e1, e2))
    record("adamw 5-step params (wd routing)", worst_p, 1e-6,
           "per-step " + " ".join(f"{e:.1e}" for e in per_step))
    record("adamw 5-step realized lr", worst_lr, 1e-9)
    steps = (int(opt.optimizers[0].step.item()),
             int(opt.optimizers[1].step.item()))
    record("adamw both step counters == 5", float(steps != (5, 5)), 0.0,
           f"steps={steps}")


def test_lr_sequence() -> None:
    d = np.load(FIX / "optimizer.npz")
    t_max = int(d["t_max"])
    sched = optim.cosine_decay(float(d["lr0"]), t_max, end=0.0)
    lr_mlx = np.asarray(sched(mx.array(np.arange(t_max)))).astype(np.float64)
    if lr_mlx.shape != (t_max,):                            # vectorized failed
        lr_mlx = np.array([float(sched(mx.array(i))) for i in range(t_max)])
    diff = np.abs(lr_mlx - d["lr_seq"])
    record("realized-lr sequence (3000)", float(diff.max()), 1e-9,
           f"argmax={int(diff.argmax())}")


def test_lr_save_restore() -> None:
    d = np.load(FIX / "optimizer.npz")
    t_max = int(d["t_max"])
    k = 7                                                    # steps pre-save
    sched = optim.cosine_decay(float(d["lr0"]), t_max, end=0.0)

    opt = make_multi_opt(sched)
    params = {"p1": mx.array(d["p1_init"]), "p2": mx.array(d["p2_init"])}
    opt.init(params)
    rng = np.random.default_rng(7)
    for _ in range(k):
        grads = {"p1": mx.array(rng.standard_normal((4, 3)).astype(np.float32)),
                 "p2": mx.array(rng.standard_normal((5,)).astype(np.float32))}
        params = opt.apply_gradients(grads, params)
        mx.eval(params, opt.state)

    with tempfile.TemporaryDirectory() as td:
        st_path = str(Path(td) / "opt_state.safetensors")
        flat = tree_flatten(opt.state)
        mx.save_safetensors(st_path, dict(flat))

        # continue the original for the reference next step
        g_next = {"p1": mx.array(rng.standard_normal((4, 3)).astype(np.float32)),
                  "p2": mx.array(rng.standard_normal((5,)).astype(np.float32))}
        params_a = opt.apply_gradients(g_next, dict(params))
        mx.eval(params_a, opt.state)
        lr_ref = float(opt.optimizers[0].learning_rate)

        # rebuild -> init -> assign state (plan-mandated resume sequence)
        opt2 = make_multi_opt(optim.cosine_decay(float(d["lr0"]), t_max,
                                                 end=0.0))
        params2 = {"p1": mx.array(d["p1_init"]), "p2": mx.array(d["p2_init"])}
        opt2.init(params2)
        loaded = mx.load(st_path)
        opt2.state = tree_unflatten(list(loaded.items()))
        steps = (int(opt2.optimizers[0].step.item()),
                 int(opt2.optimizers[1].step.item()))

        params_b = opt2.apply_gradients(g_next, dict(params))
        mx.eval(params_b, opt2.state)
        lr_restored = float(opt2.optimizers[0].learning_rate)

    e_lr_fix = abs(lr_restored - float(d["lr_seq"][k]))
    e_lr_ab = abs(lr_restored - lr_ref)
    e_par = max(float(np.abs(np.asarray(params_a[n])
                             - np.asarray(params_b[n])).max())
                for n in ("p1", "p2"))
    record("lr after save/restore vs lr_seq[7]", e_lr_fix, 1e-9,
           f"restored steps={steps}")
    record("save/restore next-step params bitwise", e_par, 0.0,
           f"lr(A)-lr(B)={e_lr_ab:.1e}")


# ---------------------------------------------------- 10. full-model forward
def load_converted_net() -> V7NetMLX:
    spec = importlib.util.spec_from_file_location("convert", HERE / "convert.py")
    convert = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(convert)

    st_path = HERE / "fixtures" / "v7net_seed.converted.safetensors"
    convert.torch2mlx(str(FIX / "v7net_seed.pt"), str(st_path))

    weights, meta = mx.load(str(st_path), return_metadata=True)
    remapped = []
    for k, v in weights.items():
        if k.startswith("conf."):
            head, rest = k.split(".", 1)
            k = f"conf.layers.{rest}"
        remapped.append((k, v))
    net = V7NetMLX(width=48, embed_dim=128)
    net.load_weights(remapped, strict=True)
    net.eval()
    return net


def test_full_model() -> None:
    spec_d = np.load(FIX / "spectrogram.npz")
    fwd = np.load(FIX / "model_forward.npz")
    net = load_converted_net()

    peak_10ms = None
    t_clean_2p5 = None
    for key in DWELL_KEYS:
        x = mx.array(spec_d[f"input_{key}"])
        if key == "10ms":
            mx.reset_peak_memory()
        spec = spectrogram(rms_normalize(x))
        z, conf, h = net.encode(spec)
        t_spec = spec.shape[1]
        if key == "2p5ms":
            t_clean_2p5 = t_spec
        t_clean = t_clean_2p5 if DWELL_N[key] > 50_000 else t_spec
        recon = net.reconstruct(h, t_clean, spec.shape[2])
        mx.eval(z, conf, h, recon)
        if key == "10ms":
            peak_10ms = mx.get_peak_memory()

        for name, out_mx, ref, tol in (
                ("z", z, fwd[f"z_{key}"], 1e-4),
                ("conf", conf, fwd[f"conf_{key}"], 1e-4),
                ("recon", recon, nchw_to_nhwc(fwd[f"recon_{key}"]), 1e-3)):
            out = np.asarray(out_mx)
            if out.shape != ref.shape:
                record(f"model {key} {name}", math.inf, tol,
                       f"SHAPE {out.shape} vs {ref.shape}")
                continue
            record(f"model {key} {name}",
                   float(np.abs(out - ref).max()), tol)
        # h is informational (not a gated comparison in the G0 list)
        h_ref = nchw_to_nhwc(fwd[f"h_{key}"])
        h_np = np.asarray(h)
        h_err = (float(np.abs(h_np - h_ref).max())
                 if h_np.shape == h_ref.shape else math.inf)
        print(f"       model {key} h (info): max|err|={h_err:.3e} "
              f"shape={h_np.shape}")

    print(f"       peak memory after 10ms (T=6249) full forward: "
          f"{peak_10ms / 1e9:.3f} GB ({peak_10ms} bytes)")
    globals()["PEAK_10MS"] = peak_10ms


# ------------------------------------------------------------------------ main
def main() -> int:
    print(f"mlx {mx.__version__}  device={mx.default_device()}")
    test_spectrogram()
    test_rms_normalize()
    test_nearest_resize()
    test_groupnorm()
    test_cdist_sq()
    test_normalize_scale()
    test_pooling()
    test_losses()
    test_optimizer_steps()
    test_lr_sequence()
    test_lr_save_restore()
    test_full_model()

    n_pass = sum(1 for _, ok, *_ in RESULTS if ok)
    print("\n=== G0 summary ===")
    for name, ok, err, tol, note in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name:44s} "
              f"err={err:.3e} tol={tol:.0e} {note}")
    print(f"{n_pass}/{len(RESULTS)} passed")
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
