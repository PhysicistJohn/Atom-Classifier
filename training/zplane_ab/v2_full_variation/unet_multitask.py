"""Complex-valued U-Net: equalize/denoise to the clean emission, and classify from the
bottleneck.

WHY A U-NET SPECIFICALLY (user, 2026-07-24). Equalization needs several time scales at
once, and the corpus was built to demand exactly that: multipath taps out to 64 samples
(fine, local), against CFO / clock drift / tuning offset that act across the whole 16384
window (coarse, global). A U-Net's encoder gives the coarse levels a receptive field that
covers the window while the fine levels keep sample resolution, and the skips let the
decoder emit a sharp waveform instead of a blurred one.

AND IT REMOVES THE BOTTLENECK PARADOX. multitask_autoencoder.py had to use a SEQUENCE
latent because a small latent vector cannot hold the emission's payload entropy, so
reconstruction from a compact code is impossible. With skips the payload flows AROUND the
bottleneck, so the bottleneck no longer has to store it -- which frees it to carry what
the classifier actually wants (modulation structure, channel parameters) rather than
bits. That is a cleaner split than the sequence-latent compromise.

THE TENSION, AND WHAT KEEPS IT HONEST. Skips also weaken the pressure that forces the
bottleneck to be a complete description: a decoder can lean on skips and let the
bottleneck go lazy. Two things counter it here, both deliberate:
  1. The class / profile / parameter heads read the BOTTLENECK ONLY. Nothing downstream
     of a skip reaches them, so a lazy bottleneck costs classification accuracy directly.
  2. The reconstruction target is the CLEAN pair while the input is the IMPAIRED capture,
     so copying the input through the skips is NOT optimal -- the network is scored on
     having actually removed the channel (measured gap: coherence 0.153 on impaired rows).
`w_rec=0` collapses this to a plain classifier of the same size, which is the control the
campaign uses to prove any gain is the objective and not the extra parameters.
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

from model import EMBED_DIM, N_FEATURES  # noqa: E402
from complex_multiscale_backbone import ComplexConv1d, ComplexModReLU  # noqa: E402
from multitask_autoencoder import PARAM_KEYS  # noqa: E402 -- same target set


class CBlock(nn.Module):
    """Complex conv -> phase-equivariant activation, twice."""

    def __init__(self, cin: int, cout: int, taps: int = 7):
        super().__init__()
        self.c1, self.a1 = ComplexConv1d(cin, cout, taps), ComplexModReLU(cout)
        self.c2, self.a2 = ComplexConv1d(cout, cout, taps), ComplexModReLU(cout)

    def forward(self, z):
        return self.a2(self.c2(self.a1(self.c1(z))))


class ComplexUNetMultiTask(nn.Module):
    """forward(x, feat) -> embedding (harness contract). Reconstruction, profile logits
    and parameter predictions are exposed as attributes, as in multitask_autoencoder."""

    def __init__(self, base: int = 8, depth: int = 4, taps: int = 7,
                 embed_dim: int = EMBED_DIM, n_features: int = N_FEATURES,
                 n_profiles: int = 37, hidden: int = 128):
        super().__init__()
        self.depth = depth
        chans = [base * (2 ** i) for i in range(depth)]          # e.g. 16,32,64,128
        self.downs = nn.ModuleList()
        cin = 1
        for c in chans:
            self.downs.append(CBlock(cin, c, taps)); cin = c
        self.bottleneck = CBlock(chans[-1], chans[-1] * 2, taps)

        self.ups = nn.ModuleList()
        rev = list(reversed(chans))
        cin = chans[-1] * 2
        for c in rev:
            # input is upsampled features CONCATENATED with the matching skip
            self.ups.append(CBlock(cin + c, c, taps)); cin = c
        self.out = ComplexConv1d(chans[0], 1, kernel_size=1)

        # heads read the BOTTLENECK ONLY -- see the module docstring
        bott = chans[-1] * 2
        self.trunk = nn.Sequential(nn.Linear(2 * bott + n_features, hidden), nn.ReLU(), nn.Dropout(0.1))
        self.embed_head = nn.Linear(hidden, embed_dim)
        self.profile_head = nn.Linear(hidden, n_profiles)
        self.param_head = nn.Linear(hidden, len(PARAM_KEYS))
        self.last_recon = None
        self.last_profile_logits = None
        self.last_params = None

    @staticmethod
    def _up(v, size):
        re = F.interpolate(v.real, size=size, mode="nearest")
        im = F.interpolate(v.imag, size=size, mode="nearest")
        return torch.complex(re, im)

    def forward(self, x: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        z = torch.complex(x[:, 0, :], x[:, 1, :]).unsqueeze(1)      # (B,1,L)
        skips = []
        v = z
        for blk in self.downs:
            v = blk(v)
            skips.append(v)
            v = v[..., ::2]                                          # complex decimate by 2
        v = self.bottleneck(v)

        mag = v.abs()
        g = torch.cat([mag.mean(-1), mag.std(-1), feat], dim=-1)
        h = self.trunk(g)
        self.last_profile_logits = self.profile_head(h)
        self.last_params = self.param_head(h)
        emb = F.normalize(self.embed_head(h), dim=-1)

        for blk, skip in zip(self.ups, reversed(skips)):
            v = self._up(v, skip.shape[-1])
            v = blk(torch.cat([v, skip], dim=1))
        self.last_recon = self.out(v).squeeze(1)
        return emb


if __name__ == "__main__":
    torch.manual_seed(0)
    net = ComplexUNetMultiTask()
    n = sum(p.numel() for p in net.parameters())
    xb = torch.randn(2, 2, 4096); fb = torch.randn(2, N_FEATURES)
    out = net(xb, fb)
    assert out.shape == (2, EMBED_DIM)
    assert torch.allclose(out.norm(dim=-1), torch.ones(2), atol=1e-4)
    assert net.last_recon.shape == (2, 4096), net.last_recon.shape
    assert net.last_profile_logits.shape == (2, 37) and net.last_params.shape == (2, len(PARAM_KEYS))
    print(f"[ok] ComplexUNetMultiTask params={n}, recon {tuple(net.last_recon.shape)}, unit-norm embedding")

    # phase equivariance must survive the skips: rotating the capture rotates the
    # reconstruction identically, and leaves the (magnitude-pooled) embedding unchanged.
    small = ComplexUNetMultiTask(base=4, depth=2, taps=5).eval()  # eval(): dropout off, else the two passes differ randomly
    x2 = torch.randn(2, 2, 512); f2 = torch.randn(2, N_FEATURES)
    phi = torch.tensor(0.77)
    zr = torch.complex(torch.cos(phi), torch.sin(phi))
    zc = torch.complex(x2[:, 0, :], x2[:, 1, :]) * zr
    with torch.no_grad():
        e1 = small(x2, f2); r1 = small.last_recon.clone()
        e2 = small(torch.stack([zc.real, zc.imag], dim=1), f2); r2 = small.last_recon
    assert torch.allclose(r2, zr * r1, atol=1e-4), "reconstruction must be phase-equivariant"
    assert torch.allclose(e1, e2, atol=1e-5), "embedding must be phase-INVARIANT"
    print("[ok] recon is phase-equivariant and the embedding is phase-invariant through the skips")
