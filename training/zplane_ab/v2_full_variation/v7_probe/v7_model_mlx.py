"""MLX model/primitives for the v7 trainer port (NO training loop).

Mirrors ``v7_trainer.py`` (torch-MPS, frozen reference) per
``docs/mlx-port-plan.md``.  Layout is NHWC internally: spectrograms are
``[B, T, 33, 3]`` with H = time (T) and W = freq (33).  Attribute names
mirror the torch module tree exactly (stem/d1..d5/embed/conf/u1..u3/out;
Block/UpBlock expose .conv/.norm) so safetensors keys map mechanically:

  torch ``<blk>.conv.weight`` [O, I, kH, kW]  ->  mlx ``<blk>.conv.weight``
  [O, kH, kW, I]  (permute (0, 2, 3, 1); biases and all GroupNorm/Linear
  params copy through unchanged).  torch ``conf.<i>.*`` -> mlx
  ``conf.layers.<i>.*`` (MLX Sequential nests under ``layers``).

Only live-verified mlx 0.32.0 APIs are used.
"""
from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.nn.utils import checkpoint as _nn_checkpoint

NFFT = 64
HOP = 32
NBINS = NFFT // 2 + 1  # 33; slice of the FULL complex FFT (not rfft)

# Periodic Hann: w[n] = 0.5 - 0.5*cos(2*pi*n/N), n = 0..N-1, computed in
# fp32 mirroring torch.hann_window's op order (arange.mul_(2*pi/N).cos_()
# .mul_(-0.5).add_(0.5)).  Bit-exactness vs torch is unattainable (torch's
# SLEEF fp32 cos is itself ~1 ulp off correctly-rounded on some bins); the
# measured max |d| is 5.96e-8 on 7/64 bins — verified in __main__ at 1e-6,
# far inside the G0 gate tolerance (atol 1e-5 fp32).
_HANN = (np.cos(np.arange(NFFT, dtype=np.float32)
                * np.float32(2.0 * np.pi / NFFT))
         * np.float32(-0.5) + np.float32(0.5)).astype(np.float32)
_HANN_MX = mx.array(_HANN)

# Precomputed [T, 64] int32 framing-index matrices, cached per input length.
_FRAME_IDX: dict[int, mx.array] = {}


def _frame_indices(n: int) -> mx.array:
    idx = _FRAME_IDX.get(n)
    if idx is None:
        t = (n - NFFT) // HOP + 1
        m = (np.arange(t, dtype=np.int32)[:, None] * HOP
             + np.arange(NFFT, dtype=np.int32)[None, :])
        idx = mx.array(m)
        _FRAME_IDX[n] = idx
    return idx


def spectrogram(x: mx.array) -> mx.array:
    """complex64 [B, N] -> float32 NHWC [B, T, 33, 3].

    Frames via index gather (no pad/center), periodic Hann-64, full complex
    FFT (default/backward norm) truncated to the first 33 bins, then
    (real, imag, log1p|.|) stacked on the channel (last) axis.
    """
    frames = mx.take(x, _frame_indices(x.shape[-1]), axis=-1)  # [B, T, 64]
    spec = mx.fft.fft(frames * _HANN_MX, axis=-1)[..., :NBINS]  # [B, T, 33]
    return mx.stack(
        (mx.real(spec), mx.imag(spec), mx.log1p(mx.abs(spec))), axis=-1)


def rms_normalize(x: mx.array) -> mx.array:
    """Per-row RMS over the last axis of |x|^2, clamped at 1e-9."""
    rms = mx.sqrt(mx.mean(mx.abs(x) ** 2, axis=-1, keepdims=True))
    return x / mx.maximum(rms, 1e-9)


def nearest_resize(x: mx.array, size: tuple[int, int]) -> mx.array:
    """Exact torch F.interpolate(mode='nearest') on NHWC [B, H, W, C].

    src = floor(arange(out) * in / out) per spatial axis, via integer
    arithmetic (exact; no fp rounding hazard).
    """
    h_out, w_out = size
    h_in, w_in = x.shape[1], x.shape[2]
    if h_out != h_in:
        idx_h = mx.array(
            ((np.arange(h_out, dtype=np.int64) * h_in) // h_out)
            .astype(np.int32))
        x = mx.take(x, idx_h, axis=1)
    if w_out != w_in:
        idx_w = mx.array(
            ((np.arange(w_out, dtype=np.int64) * w_in) // w_out)
            .astype(np.int32))
        x = mx.take(x, idx_w, axis=2)
    return x


class Block(nn.Module):
    """Conv3x3(stride=(sT, sF), pad=1, bias) -> GroupNorm(8, pytorch) -> SiLU.

    NHWC: H is TIME, W is FREQ; stride tuples keep the torch (time, freq)
    order.
    """

    def __init__(self, cin: int, cout: int, stride: tuple[int, int]):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=1,
                              bias=True)
        self.norm = nn.GroupNorm(8, cout, pytorch_compatible=True)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.silu(self.norm(self.conv(x)))


class UpBlock(nn.Module):
    """nearest_resize FIRST, then Conv3x3(stride=1, pad=1) -> GN(8) -> SiLU."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=1, padding=1, bias=True)
        self.norm = nn.GroupNorm(8, cout, pytorch_compatible=True)

    def __call__(self, x: mx.array, size: tuple[int, int]) -> mx.array:
        x = nearest_resize(x, size)
        return nn.silu(self.norm(self.conv(x)))


class V7NetMLX(nn.Module):
    """Encoder-decoder with structured latent heads (torch V7Net mirror)."""

    def __init__(self, width: int = 48, embed_dim: int = 128):
        super().__init__()
        w = width
        self.stem = Block(3, w, (1, 1))
        self.d1 = Block(w, w * 2, (2, 2))
        self.d2 = Block(w * 2, w * 2, (2, 1))
        self.d3 = Block(w * 2, w * 4, (2, 2))
        self.d4 = Block(w * 4, w * 4, (2, 1))
        self.d5 = Block(w * 4, w * 4, (2, 2))
        self.embed = nn.Linear(w * 4 * 2, embed_dim)
        self.conf = nn.Sequential(
            nn.Linear(w * 4 * 2, 64), nn.SiLU(), nn.Linear(64, 1))
        # light decoder: three ups from the bottleneck back to the spec grid
        self.u1 = UpBlock(w * 4, w * 4)
        self.u2 = UpBlock(w * 4, w * 2)
        self.u3 = UpBlock(w * 2, w)
        self.out = nn.Conv2d(w, 3, 3, stride=1, padding=1, bias=True)

    def encode(self, spec: mx.array, ckpt_blocks: int = 0
               ) -> tuple[mx.array, mx.array, mx.array]:
        """spec NHWC [B, T, 33, 3] -> (z [B, E], conf_logit [B], h NHWC).

        ``ckpt_blocks``: gradient-checkpoint the first N of (stem, d1..d5)
        via ``mlx.nn.utils.checkpoint`` (inputs/outputs saved, intermediates
        recomputed in backward; outputs and grads bitwise identical to the
        plain path -- micro-verified, and it composes with ``mx.compile``).
        Default 0 keeps the original forward exactly; forward-only callers
        (eval) never need it.
        """
        h = spec
        for i, blk in enumerate((self.stem, self.d1, self.d2, self.d3,
                                 self.d4, self.d5)):
            h = _nn_checkpoint(blk)(h) if i < ckpt_blocks else blk(h)
        pooled = mx.concatenate(
            (mx.mean(h, axis=(1, 2)), mx.max(h, axis=(1, 2))), axis=1)
        z = self.embed(pooled)
        z = z / mx.maximum(
            mx.linalg.norm(z, axis=-1, keepdims=True), 1e-12) * 4.0
        return z, mx.squeeze(self.conf(pooled), axis=-1), h

    def reconstruct(self, bottleneck: mx.array, t: int, f: int,
                    ckpt: bool = False) -> mx.array:
        """bottleneck NHWC -> NHWC [B, t, f, 3]; out conv has NO norm/act.

        ``ckpt=True`` gradient-checkpoints each UpBlock and the out conv
        (sizes are trace-time constants, so they ride in the closures).
        Needed at the T=6249 signature: the decoder convs' backward
        explicit-GEMM unfolds (u3: ~12.5 GB grad-weight im2col at B=70)
        dominate peak memory and are untouched by encoder checkpointing.
        Bitwise-identical outputs/grads, as with ``encode``.
        """
        sizes = ((max(1, t // 8), max(1, f // 4)),
                 (max(1, t // 4), max(1, f // 2)), (t, f))
        h = bottleneck
        if ckpt:
            for blk, size in zip((self.u1, self.u2, self.u3), sizes):
                h = _nn_checkpoint(blk, lambda x, b=blk, s=size: b(x, s))(h)
            return _nn_checkpoint(self.out)(h)
        for blk, size in zip((self.u1, self.u2, self.u3), sizes):
            h = blk(h, size)
        return self.out(h)


def cdist_sq(a: mx.array, b: mx.array) -> mx.array:
    """Squared euclidean distances [Na, Nb] via matmul expansion, clamped >=0."""
    a2 = mx.sum(a * a, axis=-1, keepdims=True)          # [Na, 1]
    b2 = mx.sum(b * b, axis=-1, keepdims=True).T        # [1, Nb]
    return mx.maximum(a2 + b2 - 2.0 * (a @ b.T), 0.0)


if __name__ == "__main__":
    from mlx.utils import tree_flatten

    # --- Hann parity vs torch.hann_window(64) (periodic) ---
    try:
        import torch
        th = torch.hann_window(NFFT, dtype=torch.float32).numpy()
        d = np.abs(th - _HANN).max()
        assert d <= 1e-6, f"hann mismatch, max |d| = {d}"
        print(f"hann parity vs torch: max|d| = {d:.3e} "
              f"(1-ulp; bit-exact impossible, see comment)")
    except ImportError:
        print("torch unavailable; hann parity check skipped")

    net = V7NetMLX(width=48, embed_dim=128)
    n_params = sum(v.size for _, v in tree_flatten(net.parameters()))
    print(f"params: {n_params:,}")

    # --- smoke forward at T=624 (1 ms dwell, N=20_000) ---
    rng = np.random.default_rng(0)
    B, N = 4, 20_000
    x_np = (rng.standard_normal((B, N)) + 1j * rng.standard_normal((B, N))
            ).astype(np.complex64)
    x = rms_normalize(mx.array(x_np))
    spec = spectrogram(x)
    z, conf_logit, h = net.encode(spec)
    t, f = spec.shape[1], spec.shape[2]
    recon = net.reconstruct(h, t, f)
    protos = z[:3]  # arbitrary rows just to exercise cdist_sq
    d2 = cdist_sq(z, protos)
    mx.eval(spec, z, conf_logit, h, recon, d2)
    print(f"spec  {spec.shape} {spec.dtype}")
    print(f"z     {z.shape} {z.dtype}  |z| = "
          f"{np.asarray(mx.linalg.norm(z, axis=-1)).round(4).tolist()}")
    print(f"conf  {conf_logit.shape}")
    print(f"h     {h.shape}")
    print(f"recon {recon.shape}")
    print(f"cdist_sq {d2.shape} min={float(d2.min()):.3e}")
    log_scale = mx.array(math.log(10.0))
    print(f"log_scale exp clamp: "
          f"{float(mx.clip(mx.exp(log_scale), 1e-3, 100)):.4f}")
