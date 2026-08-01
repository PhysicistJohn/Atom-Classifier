"""Generate torch-side parity fixtures for the MLX port (G0/G1 inputs).

Dumps float32/complex64 numpy .npz fixtures plus one .pt checkpoint from the
canonical torch trainer (v7_trainer.py). All tensors are produced on CPU with
fixed seeds so the fixtures are reproducible bit-for-bit.

Run:
  .venv-training/bin/python gen_torch_fixtures.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
V7_DIR = HERE.parent
sys.path.insert(0, str(V7_DIR))

from v7_trainer import spectrogram, rms_normalize, V7Net, DWELLS, NFFT, HOP  # noqa: E402

FIX = HERE / "fixtures"
FIX.mkdir(parents=True, exist_ok=True)

torch.set_default_dtype(torch.float32)
manifest: dict[str, dict[str, tuple]] = {}


def save_npz(name: str, **arrays: np.ndarray) -> None:
    path = FIX / f"{name}.npz"
    np.savez(path, **arrays)
    manifest[path.name] = {k: tuple(v.shape) for k, v in arrays.items()}


def complex_rand(rng: np.random.Generator, shape) -> np.ndarray:
    return (rng.standard_normal(shape).astype(np.float32)
            + 1j * rng.standard_normal(shape).astype(np.float32)
            ).astype(np.complex64)


# ---------------------------------------------------------------- 1. spectrogram
rng = np.random.default_rng(101)
spec_arrays = {}
spec_T = {}
for wname, length in DWELLS.items():
    key = wname.replace(".", "p")  # 2.5ms -> 2p5ms (npz-key friendly)
    x_np = complex_rand(rng, (2, length))
    spec = spectrogram(torch.from_numpy(x_np))
    spec_arrays[f"input_{key}"] = x_np
    spec_arrays[f"output_{key}"] = spec.numpy().astype(np.float32)
    spec_T[wname] = spec.shape[2]
save_npz("spectrogram", **spec_arrays)

# -------------------------------------------------------------- 2. rms_normalize
rng = np.random.default_rng(102)
rn_in = complex_rand(rng, (4, 5000))
rn_in[3] = (rn_in[3] / np.abs(rn_in[3]).max() * 1e-12).astype(np.complex64)
rn_out = rms_normalize(torch.from_numpy(rn_in)).numpy().astype(np.complex64)
save_npz("rms_normalize", input=rn_in, output=rn_out)

# ------------------------------------------------------------- 3. nearest resize
# Decoder chains (UpBlock does F.interpolate(mode="nearest") before conv).
# Encoder time downsampling: 5 stride-2 convs, out = (in-1)//2 + 1 each;
# freq 33 -> 5 after three stride-2 freq convs. Decoder targets use the CLEAN
# spec T (recon crop min(length, 50000)); for 10ms that is the 2.5ms T=1561.
def enc_t(t: int) -> int:
    for _ in range(5):
        t = (t - 1) // 2 + 1
    return t


F_FREQ = 33
resize_cases = []  # (name, (h_in, w_in), (h_out, w_out))
for wname, length in DWELLS.items():
    key = wname.replace(".", "p")
    t_spec = spec_T[wname]
    t_clean = spec_T["2.5ms"] if length > 50_000 else t_spec
    bott = (enc_t(t_spec), 5)
    targets = [(max(1, t_clean // 8), max(1, F_FREQ // 4)),
               (max(1, t_clean // 4), max(1, F_FREQ // 2)),
               (t_clean, F_FREQ)]
    src = bott
    for ui, tgt in enumerate(targets, start=1):
        resize_cases.append((f"{key}_u{ui}", src, tgt))
        src = tgt
resize_cases.append(("odd", (7, 5), (13, 9)))

rng = np.random.default_rng(103)
rz_arrays = {}
for name, (hi, wi), (ho, wo) in resize_cases:
    x_np = rng.standard_normal((2, 4, hi, wi)).astype(np.float32)
    y = F.interpolate(torch.from_numpy(x_np), size=(ho, wo), mode="nearest")
    rz_arrays[f"in_{name}"] = x_np
    rz_arrays[f"out_{name}"] = y.numpy().astype(np.float32)
save_npz("nearest_resize", **rz_arrays)

# ------------------------------------------------------------------ 4. GroupNorm
rng = np.random.default_rng(104)
gn = nn.GroupNorm(8, 48)
with torch.no_grad():
    gn.weight.copy_(torch.from_numpy(
        rng.standard_normal(48).astype(np.float32)))
    gn.bias.copy_(torch.from_numpy(
        rng.standard_normal(48).astype(np.float32)))
gn_in = rng.standard_normal((2, 48, 37, 33)).astype(np.float32)
with torch.no_grad():
    gn_out = gn(torch.from_numpy(gn_in)).numpy().astype(np.float32)
save_npz("groupnorm", input=gn_in, output=gn_out,
         weight=gn.weight.detach().numpy().astype(np.float32),
         bias=gn.bias.detach().numpy().astype(np.float32))

# -------------------------------------------------------------------- 5. cdist^2
rng = np.random.default_rng(105)
cd_arrays = {}
for n_query in (35, 48, 14):
    q = F.normalize(torch.from_numpy(
        rng.standard_normal((n_query, 128)).astype(np.float32)), dim=-1) * 4.0
    p = F.normalize(torch.from_numpy(
        rng.standard_normal((7, 128)).astype(np.float32)), dim=-1) * 4.0
    out = -torch.cdist(q, p) ** 2
    cd_arrays[f"q_{n_query}x7"] = q.numpy().astype(np.float32)
    cd_arrays[f"p_{n_query}x7"] = p.numpy().astype(np.float32)
    cd_arrays[f"out_{n_query}x7"] = out.numpy().astype(np.float32)
save_npz("cdist_sq", **cd_arrays)

# ------------------------------------------------------------- 6. normalize·4
rng = np.random.default_rng(106)
nz_in = rng.standard_normal((8, 128)).astype(np.float32)
nz_out = (F.normalize(torch.from_numpy(nz_in), dim=-1) * 4.0
          ).numpy().astype(np.float32)
save_npz("normalize_scale", input=nz_in, output=nz_out)

# --------------------------------------------------------------------- 7. pooling
rng = np.random.default_rng(107)
pool_in = rng.standard_normal((3, 192, 13, 5)).astype(np.float32)
pt = torch.from_numpy(pool_in)
save_npz("pooling", input=pool_in,
         mean_out=pt.mean(dim=(2, 3)).numpy().astype(np.float32),
         amax_out=pt.amax(dim=(2, 3)).numpy().astype(np.float32))

# ---------------------------------------------------------------------- 8. losses
# Episodic CE on fixture 5's 35x7 with the trainer's temperature.
log_scale = torch.tensor(math.log(10.0))
scale = log_scale.exp().clamp(1e-3, 100)
ce_logits = torch.from_numpy(cd_arrays["out_35x7"]) * scale
ce_target = torch.arange(7).repeat_interleave(5)
ce = F.cross_entropy(ce_logits, ce_target)

rng = np.random.default_rng(108)
bce_logits = rng.standard_normal(35).astype(np.float32) * 3.0
bce_targets = rng.integers(0, 2, size=35).astype(np.float32)
bce = F.binary_cross_entropy_with_logits(
    torch.from_numpy(bce_logits), torch.from_numpy(bce_targets))

l1_a = rng.standard_normal((2, 3, 100, 33)).astype(np.float32)
l1_b = rng.standard_normal((2, 3, 100, 33)).astype(np.float32)
l1 = F.l1_loss(torch.from_numpy(l1_a), torch.from_numpy(l1_b))

save_npz("losses",
         ce_logits=ce_logits.numpy().astype(np.float32),
         ce_target=ce_target.numpy().astype(np.int64),
         ce_scale=np.float32(scale.item()),
         ce_out=np.float32(ce.item()),
         bce_logits=bce_logits, bce_targets=bce_targets,
         bce_out=np.float32(bce.item()),
         l1_a=l1_a, l1_b=l1_b, l1_out=np.float32(l1.item()))

# ------------------------------------------------------------------ 9. optimizer
# Two-tensor model mirroring the trainer's group split:
#   group0: p1, weight_decay=1e-4 ; group1: p2, weight_decay=0.0
# lr 2e-3, CosineAnnealingLR(T_max=3000), sched.step() AFTER each opt.step().
rng = np.random.default_rng(109)
p1 = nn.Parameter(torch.from_numpy(
    rng.standard_normal((4, 3)).astype(np.float32)))
p2 = nn.Parameter(torch.from_numpy(
    rng.standard_normal((5,)).astype(np.float32)))
p1_init = p1.detach().numpy().copy()
p2_init = p2.detach().numpy().copy()
opt = torch.optim.AdamW(
    [{"params": [p1], "weight_decay": 1e-4},
     {"params": [p2], "weight_decay": 0.0}], lr=2e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=3000)

g1_steps = rng.standard_normal((5, 4, 3)).astype(np.float32)
g2_steps = rng.standard_normal((5, 5)).astype(np.float32)
p1_steps = np.empty((5, 4, 3), dtype=np.float32)
p2_steps = np.empty((5, 5), dtype=np.float32)
lr_at_step = np.empty(5, dtype=np.float64)
for i in range(5):
    lr_at_step[i] = opt.param_groups[0]["lr"]
    p1.grad = torch.from_numpy(g1_steps[i]).clone()
    p2.grad = torch.from_numpy(g2_steps[i]).clone()
    opt.step()
    sched.step()
    p1_steps[i] = p1.detach().numpy()
    p2_steps[i] = p2.detach().numpy()

# Full realized-lr sequence over 3000 episodes (fresh opt+sched, no steps on
# params needed -- CosineAnnealingLR depends only on step count).
opt2 = torch.optim.AdamW(
    [{"params": [nn.Parameter(torch.zeros(1))], "weight_decay": 1e-4}],
    lr=2e-3)
sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=3000)
lr_seq = np.empty(3000, dtype=np.float64)
for i in range(3000):
    lr_seq[i] = opt2.param_groups[0]["lr"]
    sched2.step()

save_npz("optimizer",
         p1_init=p1_init, p2_init=p2_init,
         g1_steps=g1_steps, g2_steps=g2_steps,
         p1_steps=p1_steps, p2_steps=p2_steps,
         lr_at_step=lr_at_step, lr_seq=lr_seq,
         lr0=np.float64(2e-3), wd_group0=np.float64(1e-4),
         wd_group1=np.float64(0.0), t_max=np.int64(3000))

# ----------------------------------------------------------------- 10. full model
torch.manual_seed(20260740)
net = V7Net(width=48, embed_dim=128)
net.eval()
param_count = sum(p.numel() for p in net.parameters())
torch.save({"state_dict": net.state_dict(),
            "log_scale": float(math.log(10.0)),
            "config": {"width": 48, "embed_dim": 128}},
           FIX / "v7net_seed.pt")
manifest["v7net_seed.pt"] = {
    "state_dict": (len(net.state_dict()),),
    "log_scale": (), "config": ()}

fwd_arrays = {}
with torch.no_grad():
    for wname, length in DWELLS.items():
        key = wname.replace(".", "p")
        x = torch.from_numpy(spec_arrays[f"input_{key}"])
        spec = spectrogram(rms_normalize(x))
        z, conf_logit, h = net.encode(spec)
        t_clean = spec_T["2.5ms"] if length > 50_000 else spec_T[wname]
        recon = net.reconstruct(h, (spec.shape[0], spec.shape[1],
                                    t_clean, spec.shape[3]))
        fwd_arrays[f"z_{key}"] = z.numpy().astype(np.float32)
        fwd_arrays[f"conf_{key}"] = conf_logit.numpy().astype(np.float32)
        fwd_arrays[f"h_{key}"] = h.numpy().astype(np.float32)
        fwd_arrays[f"recon_{key}"] = recon.numpy().astype(np.float32)
save_npz("model_forward", **fwd_arrays)

# -------------------------------------------------------------------- manifest
print("=== fixture manifest ===")
for fname in sorted(manifest):
    print(fname)
    for k, shp in manifest[fname].items():
        print(f"  {k}: {shp}")
print("=== key facts ===")
print(f"spec_T: { {k: v for k, v in spec_T.items()} }")
print(f"enc_t per dwell: "
      f"{ {k: enc_t(v) for k, v in spec_T.items()} }")
print(f"resize cases: {[(n, s, t) for n, s, t in resize_cases]}")
print(f"V7Net params: {param_count:,}")
print(f"ce_out={ce.item():.8f} bce_out={bce.item():.8f} "
      f"l1_out={l1.item():.8f}")
print(f"lr_seq[0]={lr_seq[0]!r} lr_seq[1]={lr_seq[1]!r} "
      f"lr_seq[-1]={lr_seq[-1]!r}")
print(f"torch={torch.__version__}")
