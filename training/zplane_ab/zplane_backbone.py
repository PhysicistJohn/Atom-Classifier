"""Torch port of the z-plane rational-kernel neural operator (Atom-Neural-RL's
zplane.py / operator.py, reimplemented from scratch in differentiable torch —
NOT imported cross-repo, per the task's hard boundary) as a drop-in backbone
for Atom-Classifier's Embedding head.

Only the conv trunk of model.py's Embedding is replaced. The pool -> concat
feat -> fc1 -> ReLU -> Dropout -> fc2 -> normalize tail is reproduced with
identical shapes/hyperparameters (copied from model.py, not imported, since
model.py hardcodes ConvBlocks into its own forward()) so this is provably the
same head, only the backbone differs.

Key invariants ported from the operator math survey (see module docstring
tests at the bottom):
  1. pole/zero radius reparametrized via RHO_MAX*sigmoid(.) / ZERO_SCALE*sigmoid(.)
     so poles are structurally always < RHO_MAX < 1 and zeros structurally < 2.0 -
     never clamped post-hoc.
  2. modReLU threshold b = -softplus(u) <= 0 always, structurally.
  3. warm-start: pole==zero at every (layer,width,section) slot at init, so
     H(z) == 1 (exact cancellation) while logits sit at the responsive middle
     of the sigmoid (not the saturated -12.0 "identity packing" dead-gradient
     trap the survey flags).
  4. No additive bias anywhere in the operator (lift/mix/project are bare
     matmuls, modReLU's b is a magnitude-only offset) - preserves global phase
     equivariance.
  5. FFT convention: natural bin order, norm="backward" (torch defaults), no
     fftshift - matches numpy's np.fft.fft bin convention used by the survey's
     response_on_grid().
  6. Kernel response is a closed-form function of the current parameters,
     recomputed fresh every forward call (never cached across calls).
"""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_TRAINING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # -> training/
if _TRAINING_DIR not in sys.path:
    sys.path.insert(0, _TRAINING_DIR)

from model import EMBED_DIM, N_FEATURES  # noqa: E402 -- constants only, never import Embedding itself

HIDDEN = 96
DROPOUT = 0.2
CFINAL = 64          # must match CNN's final channel count so 2*CFINAL+N_FEATURES == 140 stays identical

RHO_MAX = 1.0 - 2.0 ** -7   # 0.9921875 -- constant copied, not imported from atom_neural_rl
ZERO_SCALE = 2.0


def _logit(p: float) -> float:
    p = min(max(p, 1e-9), 1.0 - 1e-9)
    return math.log(p / (1.0 - p))


class ZPlaneOperatorBackbone(nn.Module):
    """Complex rational-kernel spectral operator, W parallel diagonal kernels
    per layer + a full (W,W) complex mixing matrix, acting on a single complex
    I/Q channel. Trainable state is entirely real-valued nn.Parameter tensors;
    complex tensors are built on-the-fly in forward() via torch.complex /
    torch.polar and never stored as leaf parameters (keeps optimizer state
    ordinary real Adam, no complex-tensor optimizer edge cases)."""

    def __init__(self, width: int = 64, layers: int = 3, sections: int = 8,
                 in_channels: int = 1, warm_radius: float = 0.5):
        super().__init__()
        L, W, K = layers, width, sections
        self.L, self.W, self.K, self.Cin = L, W, K, in_channels

        angles0 = 2.0 * math.pi * torch.arange(K, dtype=torch.float32) / max(K, 1)  # (K,)
        pole_r0 = _logit(warm_radius / RHO_MAX)
        zero_r0 = _logit(warm_radius / ZERO_SCALE)

        def rep(v):
            return v.expand(L, W, K).clone()

        self.pole_radius_logit = nn.Parameter(rep(torch.full((K,), pole_r0)))
        self.pole_angle = nn.Parameter(rep(angles0))
        self.zero_radius_logit = nn.Parameter(rep(torch.full((K,), zero_r0)))
        self.zero_angle = nn.Parameter(rep(angles0))
        self.gain_re = nn.Parameter(torch.ones(L, W))          # gain = 1+0j at init, unconstrained
        self.gain_im = nn.Parameter(torch.zeros(L, W))
        self.thresh_raw = nn.Parameter(torch.full((L, W), -6.0))  # b = -softplus(-6) ~ -0.0025

        # lift / mix / project -- real+imag stored separately, standard small-Gaussian
        # complex init (trained end-to-end from scratch, no pretrained backbone to preserve).
        self.lift_re = nn.Parameter(torch.randn(W, in_channels) / math.sqrt(in_channels))
        self.lift_im = nn.Parameter(torch.randn(W, in_channels) / math.sqrt(in_channels))
        # Stability fix (found empirically during this experiment, see zplane_ab
        # run log): a naive std=1/sqrt(fan_in) applied independently to the real
        # AND imaginary parts of a (W,W) complex matrix double-counts variance
        # (total complex variance is 2/fan_in, not 1/fan_in), and for a complex
        # Ginibre matrix of width 64 that alone puts the operator (spectral) norm
        # at ~2.75 -- i.e. the per-layer mixing map amplifies signal energy by
        # ~2.75x on every layer BEFORE any training even starts, since modReLU is
        # only ever non-expansive (it can shrink magnitude, never compensate for
        # energy the mix/spectral sum injects). Compounded across L layers and
        # then across training steps (Adam has no reason to shrink it back down),
        # this reliably overflows to inf/NaN within a few hundred episodes (empirically
        # confirmed: the first full run NaN'd by episode ~500). MIX_INIT_SCALE
        # brings the initial operator norm down to ~0.7 (still a real, trainable
        # matrix -- Adam is free to grow it back if the task rewards it -- just a
        # safe non-exploding starting point). Applied to proj too since it has the
        # same (W,W) unconstrained-matrix shape, even though it is only applied once
        # (not compounded across layers) so is lower-risk.
        MIX_INIT_SCALE = 0.25
        self.mix_re = nn.Parameter(torch.randn(L, W, W) * MIX_INIT_SCALE / math.sqrt(W))
        self.mix_im = nn.Parameter(torch.randn(L, W, W) * MIX_INIT_SCALE / math.sqrt(W))
        self.proj_re = nn.Parameter(torch.randn(W, W) * MIX_INIT_SCALE / math.sqrt(W))
        self.proj_im = nn.Parameter(torch.randn(W, W) * MIX_INIT_SCALE / math.sqrt(W))

    def _kernel_response(self, n: int) -> torch.Tensor:
        """Closed-form H(e^{jw}) for all (layer,width) kernels at once, evaluated
        fresh from the current parameters. Returns complex tensor (L, W, n)."""
        radius_p = RHO_MAX * torch.sigmoid(self.pole_radius_logit)     # (L,W,K), always < RHO_MAX < 1
        radius_z = ZERO_SCALE * torch.sigmoid(self.zero_radius_logit)  # (L,W,K)
        pole = torch.polar(radius_p, self.pole_angle)                  # (L,W,K) complex
        zero = torch.polar(radius_z, self.zero_angle)                  # (L,W,K) complex
        gain = torch.complex(self.gain_re, self.gain_im)               # (L,W) complex

        omega = 2.0 * math.pi * torch.arange(n, device=pole.device, dtype=torch.float32) / n
        zinv = torch.polar(torch.ones_like(omega), -omega)             # exp(-i*omega), natural bin order
        num = torch.prod(1 - zero.unsqueeze(-1) * zinv, dim=-2)        # (L,W,n)
        den = torch.prod(1 - pole.unsqueeze(-1) * zinv, dim=-2)        # (L,W,n)
        return gain.unsqueeze(-1) * num / den                          # (L,W,n)

    @staticmethod
    def _modrelu(z: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Masked/where-guarded -- no 0/0 NaN in the forward VALUE, exact identity
        at b=0, non-expansive.

        Stability fix (found empirically during this experiment -- confirmed via
        torch.autograd.set_detect_anomaly(True), which pointed exactly at this
        function's `torch.abs`): plain `torch.abs(z)` on a complex tensor has a
        mathematically well-known singular BACKWARD at z=0 (gradient formula
        z/|z| is 0/0) even when the forward VALUE is masked/guarded downstream --
        PyTorch computes the local vjp for the `abs` op itself before any
        downstream masking is applied, so an exact-zero activation (routinely
        produced by this very modReLU, since clamp(mag+b, min=0) floors to
        precisely 0.0 whenever mag<|b|) reliably poisons the whole backward pass
        with NaN within the first few hundred training steps -- this is exactly
        the gotcha the operator survey flags ("autograd of complex .abs() at z=0
        is a known singular point"). Fix: compute magnitude as
        sqrt(re^2+im^2+EPS) instead of torch.abs -- still exactly phase-invariant
        (re^2+im^2 = |z|^2 regardless of phase, so modReLU's phase-equivariance
        proof is unaffected), smooth everywhere (no singularity), and
        numerically indistinguishable from true |z| for any element not itself
        within ~1e-6 of exactly zero.
        """
        mag = torch.sqrt(z.real ** 2 + z.imag ** 2 + 1e-12)
        scale = torch.clamp(mag + b, min=0.0)
        return scale / mag * z

    def forward(self, iq_complex: torch.Tensor) -> torch.Tensor:
        """iq_complex: (B, Cin, N) complex64 -> (B, W, N) complex64."""
        B, _, N = iq_complex.shape
        lift = torch.complex(self.lift_re, self.lift_im)               # (W, Cin)
        v = torch.einsum("wc,bcn->bwn", lift, iq_complex)               # (B,W,N)
        H = self._kernel_response(N)                                   # (L,W,N), fresh every call
        for l in range(self.L):
            Vf = torch.fft.fft(v, dim=-1)                               # (B,W,N)
            spectral = torch.fft.ifft(H[l].unsqueeze(0) * Vf, dim=-1)    # (B,W,N)
            mix = torch.complex(self.mix_re[l], self.mix_im[l])
            mixed = torch.einsum("ij,bjn->bin", mix, v)                 # (B,W,N)
            pre = mixed + spectral
            b = -F.softplus(self.thresh_raw[l]).unsqueeze(0).unsqueeze(-1)  # (1,W,1), b<=0
            v = self._modrelu(pre, b)
        proj = torch.complex(self.proj_re, self.proj_im)
        return torch.einsum("ij,bjn->bin", proj, v)                     # (B, W, N) complex


class ZPlaneEmbedding(nn.Module):
    """Drop-in replacement for model.Embedding: same forward(x, feat) contract,
    same fc1/fc2/normalize tail (identical in/out dims + hyperparams), only the
    conv trunk is swapped for the z-plane operator backbone."""

    def __init__(self, width: int = 64, layers: int = 3, sections: int = 8,
                 embed_dim: int = EMBED_DIM, n_features: int = N_FEATURES,
                 warm_radius: float = 0.5):
        super().__init__()
        self.backbone = ZPlaneOperatorBackbone(width=width, layers=layers, sections=sections,
                                                in_channels=1, warm_radius=warm_radius)
        self.fc1 = nn.Linear(2 * width + n_features, HIDDEN)   # identical shape/hyperparams to model.py's fc1
        self.drop = nn.Dropout(DROPOUT)
        self.fc2 = nn.Linear(HIDDEN, embed_dim)                # identical shape to model.py's fc2
        self.embed_dim = embed_dim
        self.n_features = n_features

    def forward(self, x: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """x: float32 (B,2,1024), feat: float32 (B,12) -- SAME signature as model.Embedding.forward."""
        z = torch.complex(x[:, 0, :], x[:, 1, :]).unsqueeze(1)   # (B,1,1024) complex64
        v = self.backbone(z)                        # (B, W, N) complex
        mean = v.abs().mean(dim=-1)                  # (B, W) nonnegative, phase-invariant pool
        std = v.abs().std(dim=-1, unbiased=False)     # (B, W) matches CNN's population-std choice
        h = torch.cat([mean, std, feat], dim=-1)      # (B, 2W+12)
        h = F.relu(self.fc1(h))
        h = self.drop(h)
        h = self.fc2(h)
        return F.normalize(h, dim=-1)


if __name__ == "__main__":
    # Cheap sanity checks per the design plan's three invariants -- run before
    # any real training.
    torch.manual_seed(0)
    bb = ZPlaneOperatorBackbone(width=4, layers=2, sections=8, in_channels=1, warm_radius=0.5)

    # 1) pole/zero radii structurally bounded for arbitrary parameter values.
    with torch.no_grad():
        bb.pole_radius_logit.uniform_(-50, 50)
        bb.zero_radius_logit.uniform_(-50, 50)
        radius_p = RHO_MAX * torch.sigmoid(bb.pole_radius_logit)
        radius_z = ZERO_SCALE * torch.sigmoid(bb.zero_radius_logit)
        # sigmoid saturates to exactly 1.0 in float32 for large |x|, so the bound
        # is <= (not strict <) at float32 precision; the *reparametrization*
        # (RHO_MAX*sigmoid(x) with sigmoid's true range (0,1)) still guarantees
        # radius < RHO_MAX < 1 mathematically for any finite real x.
        assert float(radius_p.max()) <= RHO_MAX
        assert float(radius_z.max()) <= ZERO_SCALE
        bb.thresh_raw.uniform_(-50, 50)
        b = -F.softplus(bb.thresh_raw)
        assert float(b.max()) <= 0.0
    print("[ok] pole/zero radius + threshold sign invariants hold for arbitrary params")

    # 2) warm-start: H ~= 1 at init (fresh instance, un-perturbed params).
    bb2 = ZPlaneOperatorBackbone(width=4, layers=2, sections=8, in_channels=1, warm_radius=0.5)
    with torch.no_grad():
        H = bb2._kernel_response(1024)
        mag_err = float((H.abs() - 1.0).abs().max())
        ang_err = float(H.angle().abs().max())
    print(f"[warm-start] max|H|-1| = {mag_err:.2e}, max|angle(H)| = {ang_err:.2e}")
    assert mag_err < 1e-4 and ang_err < 1e-4, "warm-start should give H(z) ~= 1 (identity) at init"

    # 3) modReLU: identity at b=0, phase-equivariant, non-expansive.
    z = torch.randn(5, 3, dtype=torch.cfloat)
    b0 = torch.zeros(5, 3)
    out0 = ZPlaneOperatorBackbone._modrelu(z, b0)
    assert torch.allclose(out0, z, atol=1e-6), "modReLU(z,0) should equal z exactly"
    phi = torch.tensor(0.7)
    rot = torch.complex(torch.cos(phi), torch.sin(phi))
    b_neg = -torch.rand(5, 3)
    lhs = ZPlaneOperatorBackbone._modrelu(rot * z, b_neg)
    rhs = rot * ZPlaneOperatorBackbone._modrelu(z, b_neg)
    assert torch.allclose(lhs, rhs, atol=1e-5), "modReLU should be phase-equivariant"
    out_neg = ZPlaneOperatorBackbone._modrelu(z, b_neg)
    assert bool((out_neg.abs() <= z.abs() + 1e-6).all()), "modReLU should be non-expansive"
    print("[ok] modReLU identity-at-zero / phase-equivariance / non-expansiveness hold")

    # 4) end-to-end shape check through ZPlaneEmbedding.
    net = ZPlaneEmbedding(width=8, layers=2, sections=8)
    xb = torch.randn(3, 2, 1024)
    fb = torch.randn(3, N_FEATURES)
    out = net(xb, fb)
    assert out.shape == (3, EMBED_DIM)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(3), atol=1e-4)
    print(f"[ok] ZPlaneEmbedding forward shape {tuple(out.shape)}, unit-norm ok")
    print("all sanity checks passed")
