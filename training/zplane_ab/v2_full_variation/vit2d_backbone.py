"""A 2D Vision Transformer over a time-frequency (spectrogram) view of the
I/Q signal, instead of vit_backbone.py's 1D patchify over raw time samples.

MOTIVATION: 9 tests across loss shaping (plain CE, mean-repulsion weak/strong,
per-sample repulsion), feature normalization (standard vs robust median/MAD),
preprocessing condition (resampled vs native), training length (1k->750k
episodes), and hyperparameters (LR/WD/warmup/label-smoothing) all converged on
the SAME bluetooth/gsm/dsss confusion, plateauing around 0.87 overall -- a
representational ceiling in the raw-time-domain-I/Q-plus-cumulant-features
input, not a fixable training artifact (see conversation). Bluetooth, GSM
bursts, and DSSS chip sequences are exactly the kind of signals whose
distinguishing structure is burst TIMING and duty cycle -- easy to see in a
spectrogram, hard for a 1D conv/attention stack over the raw waveform to
recover on its own.

The spectrogram is computed INSIDE the model's forward() from the exact same
(B, 2, L) preprocessed I/Q tensor every other backbone in this A/B consumes
(model.Embedding, ZPlaneEmbedding, ViTEmbedding) -- so this plugs into the
identical data pipeline (full_split, preprocess/native_preprocess,
train_common(_v2/tuned/margin), evaluate_ab_v2) with zero changes anywhere
else. Only the front-end differs; same fc1/fc2/normalize tail as every other
backbone here.
"""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_TRAINING_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _TRAINING_DIR not in sys.path:
    sys.path.insert(0, _TRAINING_DIR)

from model import EMBED_DIM, N_FEATURES  # noqa: E402
from vit_backbone import TransformerBlock, HIDDEN, DROPOUT  # noqa: E402 -- reused, not duplicated


def iq_to_log_spectrogram(x: torch.Tensor, n_fft: int, hop_length: int) -> torch.Tensor:
    """x: (B, 2, L) real (I, Q channels) -> (B, 1, F, T) real log-magnitude
    spectrogram, F=n_fft (full complex spectrum, not one-sided -- the input
    is genuinely complex I/Q, not a real audio signal, so negative
    frequencies carry independent information and must be kept)."""
    z = torch.complex(x[:, 0, :], x[:, 1, :])  # (B, L) complex
    window = torch.hann_window(n_fft, device=x.device, dtype=x.real.dtype if hasattr(x, "real") else x.dtype)
    spec = torch.stft(z, n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
                       window=window, center=True, return_complex=True, onesided=False)  # (B, F, T) complex
    mag = spec.abs()
    log_mag = torch.log1p(mag)
    # per-sample normalize so absolute scale doesn't dominate (RMS-normalized
    # upstream already controls waveform scale, but STFT + log reshapes that)
    mean = log_mag.mean(dim=(1, 2), keepdim=True)
    std = log_mag.std(dim=(1, 2), keepdim=True) + 1e-6
    log_mag = (log_mag - mean) / std
    return log_mag.unsqueeze(1)  # (B, 1, F, T)


class PatchEmbed2D(nn.Module):
    """Conv2d patch embedding, standard ViT front-end. Pads F/T up to a
    multiple of the patch size first (spectrogram dims depend on n_fft/hop/L
    and won't always divide evenly)."""

    def __init__(self, in_ch: int, embed_dim: int, patch_h: int, patch_w: int):
        super().__init__()
        self.patch_h, self.patch_w = patch_h, patch_w
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w))

    def forward(self, x: torch.Tensor):
        B, C, Fdim, T = x.shape
        pad_f = (self.patch_h - Fdim % self.patch_h) % self.patch_h
        pad_t = (self.patch_w - T % self.patch_w) % self.patch_w
        if pad_f or pad_t:
            x = F.pad(x, (0, pad_t, 0, pad_f))
        x = self.proj(x)                          # (B, embed_dim, F', T')
        n_h, n_w = x.shape[-2], x.shape[-1]
        x = x.flatten(2).transpose(1, 2)            # (B, n_h*n_w, embed_dim)
        return x, n_h, n_w


class ViT2DEmbedding(nn.Module):
    """Drop-in replacement for model.Embedding / ZPlaneEmbedding / ViTEmbedding:
    same forward(x, feat) contract, same fc1/fc2/normalize tail."""

    def __init__(self, n_fft: int = 64, hop_length: int = 8, patch_h: int = 8, patch_w: int = 8,
                 embed_dim: int = 64, depth: int = 4, num_heads: int = 4, mlp_ratio: float = 2.0,
                 dropout: float = 0.1, max_patches: int = 700,
                 out_embed_dim: int = EMBED_DIM, n_features: int = N_FEATURES):
        super().__init__()
        self.n_fft, self.hop_length = n_fft, hop_length
        self.patch_embed = PatchEmbed2D(1, embed_dim, patch_h, patch_w)
        self.embed_dim = embed_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # generous upper bound on patch count (spectrogram size is fixed once
        # n_fft/hop/input-length are fixed, but keep this robust rather than
        # computing it analytically)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.fc1 = nn.Linear(embed_dim + n_features, HIDDEN)
        self.drop = nn.Dropout(DROPOUT)
        self.fc2 = nn.Linear(HIDDEN, out_embed_dim)

    def forward(self, x: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        spec = iq_to_log_spectrogram(x, self.n_fft, self.hop_length)  # (B, 1, F, T)
        patches, n_h, n_w = self.patch_embed(spec)                     # (B, n_patches, embed_dim)
        n_patches = patches.shape[1]
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, patches], dim=1)
        assert n_patches + 1 <= self.pos_embed.shape[1], (
            f"spectrogram produced {n_patches} patches ({n_h}x{n_w}), exceeds max_patches={self.pos_embed.shape[1]-1} "
            f"-- increase max_patches or patch_h/patch_w")
        h = h + self.pos_embed[:, : n_patches + 1]
        h = self.pos_drop(h)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        cls_out = h[:, 0]

        h = torch.cat([cls_out, feat], dim=-1)
        h = F.relu(self.fc1(h))
        h = self.drop(h)
        h = self.fc2(h)
        return F.normalize(h, dim=-1)


if __name__ == "__main__":
    torch.manual_seed(0)
    for in_len in (1024, 4096):
        net = ViT2DEmbedding(n_fft=64, hop_length=8)
        n_params = sum(p.numel() for p in net.parameters())
        xb = torch.randn(5, 2, in_len)
        fb = torch.randn(5, N_FEATURES)
        out = net(xb, fb)
        assert out.shape == (5, EMBED_DIM)
        assert torch.allclose(out.norm(dim=-1), torch.ones(5), atol=1e-4)
        spec = iq_to_log_spectrogram(xb, 64, 8)
        print(f"[ok] in_len={in_len}: spectrogram shape {tuple(spec.shape)}, param_count={n_params}, "
              f"forward shape {tuple(out.shape)}, unit-norm ok")
