"""Supervised encoder/decoder: encode to a latent that must simultaneously (a) name the
class, (b) predict the physical capture parameters, and (c) decode back to the CLEAN
emission.

RATIONALE (user, 2026-07-24). A sequential equalizer->classifier only ever supervises
the latent with "which class". If instead the latent has to reconstruct the clean signal
AND report the parameters, it is forced to be a complete description of the capture, and
the classifier reads the class-carrying part of a representation that already knows what
the channel did to it. Explicitly REPRESENTING the nuisances is a stronger construction
than hoping the network learns invariance to them implicitly.

This is only possible because the corpus rebuild wrote the parameters into the manifest:
snrDb, centreOffsetFrac, clockErrorPpm, multipathTaps, centerHz, rxPowerDbm, plus the
37-way profile under the 7-way class. Those are the regression/aux targets below.

THE INFORMATION-THEORETIC CONSTRAINT, AND HOW IT IS HANDLED. A small latent VECTOR
cannot reconstruct the clean emission: the emission carries random payload (thousands of
bits) that no 128-dim code can hold, so a vector-bottleneck autoencoder would be
optimizing an impossible objective and would degenerate. The latent here is therefore a
SEQUENCE (C, L/S) that keeps time structure and can carry payload, and a POOLED global
vector is taken off that same latent for the class and parameter heads. Reconstruction
uses the sequence; classification and parameters use the pooled summary. That is the
disentanglement the design wants, without asking for the impossible.

Reconstruction is scored by coherence (scale- and phase-invariant), not MSE, for the
same reason as in equalizer_frontend.py: absolute scale and absolute phase are nuisances
the rest of the pipeline already removes, so penalizing them wastes capacity.
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

# Regression targets read straight from corpus.json. Each is standardized before use;
# see PARAM_STATS in train_multitask.py.
PARAM_KEYS = ["snrDb", "centreOffsetFrac", "clockErrorPpm", "multipathTaps",
              "log10_centerHz", "log10_fracBW"]


class Encoder(nn.Module):
    """Complex conv stack with strided downsampling -> latent sequence (B, C, L/stride)."""

    def __init__(self, width: int = 32, n_down: int = 4, taps: int = 15):
        super().__init__()
        self.lift = ComplexConv1d(1, width, kernel_size=1)
        self.downs = nn.ModuleList()
        self.acts = nn.ModuleList()
        for _ in range(n_down):
            self.downs.append(ComplexConv1d(width, width, kernel_size=taps))
            self.acts.append(ComplexModReLU(width))
        self.stride = 2 ** n_down
        self.width = width

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        v = self.lift(z)
        for conv, act in zip(self.downs, self.acts):
            v = act(conv(v))
            # complex strided decimation by 2 (slicing keeps it exactly linear/aliasing
            # behaviour identical to a real decimator; the conv above is the anti-alias)
            v = v[..., ::2]
        return v                                   # (B, width, L/stride) complex


class Decoder(nn.Module):
    """Latent sequence -> reconstructed clean emission at full rate."""

    def __init__(self, width: int = 32, n_up: int = 4, taps: int = 15):
        super().__init__()
        self.ups = nn.ModuleList()
        self.acts = nn.ModuleList()
        for _ in range(n_up):
            self.ups.append(ComplexConv1d(width, width, kernel_size=taps))
            self.acts.append(ComplexModReLU(width))
        self.project = ComplexConv1d(width, 1, kernel_size=1)

    def forward(self, v: torch.Tensor, out_len: int) -> torch.Tensor:
        for conv, act in zip(self.ups, self.acts):
            # nearest-neighbour upsample on real and imaginary parts, then filter
            re = F.interpolate(v.real, scale_factor=2, mode="nearest")
            im = F.interpolate(v.imag, scale_factor=2, mode="nearest")
            v = act(conv(torch.complex(re, im)))
        y = self.project(v)                        # (B, 1, ~L) complex
        if y.shape[-1] != out_len:                 # guard against off-by-one from odd lengths
            re = F.interpolate(y.real, size=out_len, mode="nearest")
            im = F.interpolate(y.imag, size=out_len, mode="nearest")
            y = torch.complex(re, im)
        return y


class MultiTaskAutoencoder(nn.Module):
    """forward(x, feat) -> embedding, keeping the harness contract. Reconstruction,
    profile logits and parameter predictions are exposed as attributes set during
    forward so the shared training/eval code needs no signature change."""

    def __init__(self, width: int = 32, n_down: int = 4, taps: int = 15,
                 embed_dim: int = EMBED_DIM, n_features: int = N_FEATURES,
                 n_profiles: int = 37, hidden: int = 128):
        super().__init__()
        self.encoder = Encoder(width, n_down, taps)
        self.decoder = Decoder(width, n_down, taps)
        # pooled global summary of the latent SEQUENCE -> the vector the heads read
        pooled = 2 * width + n_features
        self.trunk = nn.Sequential(nn.Linear(pooled, hidden), nn.ReLU(), nn.Dropout(0.1))
        self.embed_head = nn.Linear(hidden, embed_dim)          # -> prototypical embedding
        self.profile_head = nn.Linear(hidden, n_profiles)       # -> 37-way sub-profile
        self.param_head = nn.Linear(hidden, len(PARAM_KEYS))    # -> physical parameters
        self.last_recon: torch.Tensor | None = None
        self.last_profile_logits: torch.Tensor | None = None
        self.last_params: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        z = torch.complex(x[:, 0, :], x[:, 1, :]).unsqueeze(1)          # (B,1,L)
        lat = self.encoder(z)                                            # (B,C,L/S)
        self.last_recon = self.decoder(lat, out_len=z.shape[-1]).squeeze(1)

        mag = lat.abs()
        g = torch.cat([mag.mean(-1), mag.std(-1), feat], dim=-1)
        h = self.trunk(g)
        self.last_profile_logits = self.profile_head(h)
        self.last_params = self.param_head(h)
        return F.normalize(self.embed_head(h), dim=-1)


if __name__ == "__main__":
    torch.manual_seed(0)
    net = MultiTaskAutoencoder()
    n = sum(p.numel() for p in net.parameters())
    xb = torch.randn(4, 2, 4096); fb = torch.randn(4, N_FEATURES)
    out = net(xb, fb)
    assert out.shape == (4, EMBED_DIM)
    assert torch.allclose(out.norm(dim=-1), torch.ones(4), atol=1e-4)
    assert net.last_recon.shape == (4, 4096), net.last_recon.shape
    assert net.last_profile_logits.shape == (4, 37)
    assert net.last_params.shape == (4, len(PARAM_KEYS))
    print(f"[ok] MultiTaskAutoencoder params={n}")
    print(f"     latent seq {tuple(net.encoder(torch.complex(xb[:,0,:],xb[:,1,:]).unsqueeze(1)).shape)} "
          f"(stride {net.encoder.stride}), recon {tuple(net.last_recon.shape)}")

    # the encoder must stay phase-equivariant: a global phase rotation of the capture is
    # a nuisance, and the latent should rotate with it rather than encode it as content
    enc = Encoder(width=4, n_down=2, taps=7)
    z = torch.randn(2, 1, 256, dtype=torch.cfloat)
    phi = torch.tensor(0.9); r = torch.complex(torch.cos(phi), torch.sin(phi))
    assert torch.allclose(enc(r * z), r * enc(z), atol=1e-4)
    print("[ok] encoder is phase-equivariant end to end")
