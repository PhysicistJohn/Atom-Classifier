"""A learned equalizer front end, trained jointly with the classifier.

IDEA (user, 2026-07-24): rather than making the classifier implicitly invariant to
channel and receiver impairments, put an explicit front end in front of it whose job
is to undo them, and train the two together.

WHY IT IS WELL POSED HERE. The corpus generator now emits PAIRS: the impaired capture
the classifier sees, and the clean emission it came from, at the same start index and
the same profile. So the front end has a direct supervision target, which is a far
stronger signal than backpropagating classification error alone. This is the same
structure the z-plane operator in Atom-Neural-RL was designed around -- that operator
was built for equalization and trained on a coherence reward -- except here it is
trained by gradient descent against an explicit target rather than by CMA-ES against
a black-box reward.

THE LOSS IS COHERENCE, NOT MSE. Reproducing the clean waveform sample-for-sample would
require the front end to also undo global scale and global phase, which are exactly the
nuisances the downstream pipeline already removes (RMS normalization, and the
embedding's phase-invariant features). Penalizing them would waste capacity fighting
the rest of the system. Coherence

    gamma^2 = |<y, x_clean>|^2 / (||y||^2 ||x_clean||^2)

is invariant to both by construction, so the front end is scored only on recovering
STRUCTURE. This is the same quantity Atom-Neural-RL settled on after its weighted-sum
reward proved hackable; the lesson recorded there was to find the invariance-correct
core quantity rather than patch a proxy, and it applies unchanged.

The front end is deliberately small and complex-valued: impairments (multipath, CFO,
IQ imbalance, phase noise) act linearly-ish on the complex envelope, so a complex FIR
stack is the natural function class, and phase equivariance means it cannot cheat by
memorizing an absolute phase.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_V2_DIR = os.path.dirname(os.path.abspath(__file__))
_ZPAB_DIR = os.path.dirname(_V2_DIR)
_TRAINING_DIR = os.path.dirname(_ZPAB_DIR)
for p in (_TRAINING_DIR, _ZPAB_DIR, _V2_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from complex_multiscale_backbone import ComplexConv1d, ComplexModReLU  # noqa: E402 -- reused


class ComplexEqualizer(nn.Module):
    """Complex FIR stack with a residual path. Residual matters: at high SNR with no
    multipath the identity is already the right answer, so the network should have to
    learn a correction rather than re-synthesize the signal from scratch."""

    def __init__(self, width: int = 16, taps: int = 33, n_blocks: int = 3, dilations=(1, 2, 4)):
        super().__init__()
        self.lift = ComplexConv1d(1, width, kernel_size=1)
        self.blocks = nn.ModuleList()
        self.acts = nn.ModuleList()
        for i in range(n_blocks):
            self.blocks.append(ComplexConv1d(width, width, kernel_size=taps,
                                              dilation=dilations[i % len(dilations)]))
            self.acts.append(ComplexModReLU(width))
        self.project = ComplexConv1d(width, 1, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, 1, L) complex -> (B, 1, L) complex."""
        v = self.lift(z)
        for blk, act in zip(self.blocks, self.acts):
            v = act(blk(v) + v)
        return z + self.project(v)          # residual around the whole stack


def coherence(y: torch.Tensor, x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """|<y,x>|^2 / (||y||^2 ||x||^2) per item. Invariant to the global scale and
    global phase of either argument. Returns (B,) in [0,1]."""
    num = (y * x.conj()).sum(dim=-1).abs() ** 2
    den = (y.abs() ** 2).sum(dim=-1) * (x.abs() ** 2).sum(dim=-1) + eps
    return num / den


class EqualizedClassifier(nn.Module):
    """Equalizer front end + any existing backbone, same forward(x, feat) contract as
    every other model in this A/B so it drops into the unchanged harness.

    forward() returns embeddings only. The equalized waveform needed for the coherence
    term is exposed via `last_equalized`, set on every forward pass -- keeping the
    signature identical avoids touching the shared training/eval code."""

    def __init__(self, backbone: nn.Module, eq_width: int = 16, eq_taps: int = 33,
                 eq_blocks: int = 3):
        super().__init__()
        self.equalizer = ComplexEqualizer(width=eq_width, taps=eq_taps, n_blocks=eq_blocks)
        self.backbone = backbone
        self.last_equalized: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        z = torch.complex(x[:, 0, :], x[:, 1, :]).unsqueeze(1)      # (B,1,L)
        y = self.equalizer(z)                                        # (B,1,L)
        self.last_equalized = y.squeeze(1)                           # (B,L) for the coherence term
        # hand the equalized waveform to the backbone in its expected real 2-channel form
        xe = torch.stack([y.squeeze(1).real, y.squeeze(1).imag], dim=1)
        # renormalize: the equalizer may change scale, and everything downstream assumes
        # unit-RMS input (the corpus front end guarantees it, so preserve the invariant)
        rms = torch.sqrt((xe ** 2).mean(dim=(1, 2), keepdim=True) + 1e-12)
        return self.backbone(xe / rms, feat)


if __name__ == "__main__":
    torch.manual_seed(0)
    from model import Embedding, EMBED_DIM, N_FEATURES
    net = EqualizedClassifier(Embedding(EMBED_DIM, N_FEATURES))
    n = sum(p.numel() for p in net.parameters())
    eqn = sum(p.numel() for p in net.equalizer.parameters())
    xb = torch.randn(4, 2, 4096); fb = torch.randn(4, N_FEATURES)
    out = net(xb, fb)
    assert out.shape == (4, EMBED_DIM)
    assert torch.allclose(out.norm(dim=-1), torch.ones(4), atol=1e-4)
    print(f"[ok] EqualizedClassifier total={n} (equalizer={eqn}), out {tuple(out.shape)}, unit-norm")

    # coherence sanity: identical up to scale+phase -> 1.0; orthogonal -> ~0
    a = torch.randn(3, 512, dtype=torch.cfloat)
    rot = torch.complex(torch.tensor(0.3), torch.tensor(0.7))
    assert torch.allclose(coherence(a * rot * 2.5, a), torch.ones(3), atol=1e-5)
    b = torch.randn(3, 512, dtype=torch.cfloat)
    print(f"[ok] coherence: scale+phase invariant = {coherence(a*rot*2.5,a).mean():.6f}, "
          f"independent = {coherence(b,a).mean():.4f}")

    # the equalizer must be phase-equivariant end to end (no absolute-phase memorization)
    eq = ComplexEqualizer(width=4, taps=9, n_blocks=2)
    z = torch.randn(2, 1, 256, dtype=torch.cfloat)
    phi = torch.tensor(1.1); r = torch.complex(torch.cos(phi), torch.sin(phi))
    assert torch.allclose(eq(r * z), r * eq(z), atol=1e-4)
    print("[ok] equalizer is phase-equivariant end to end")
