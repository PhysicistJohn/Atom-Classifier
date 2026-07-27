"""A small Vision-Transformer-style backbone over I/Q, as an alternative to
model.py's CNN trunk, for the bluetooth-vs-gsm/cw confusion that oversampling
alone didn't fix (see iterate_cnn.py run1-3: boosting bluetooth 3x-5x left it
flat at ~0.4-0.6 regardless of weight, always leaking into the same targets --
a feature-overlap signature, not a data-quantity one). Since the corpus is
synthetic and can be regenerated at any size (see generate-signallab-iq-corpus.ts's
TARGET_PER_CLASS env override), the usual "ViT needs more data than a CNN"
objection is addressed by scaling the corpus up rather than avoiding attention.

Standard ViT recipe adapted to a 1D dual-channel (I,Q) signal instead of a 2D
image: patchify the 1024-sample sequence into non-overlapping patches, linearly
embed each patch, prepend a learnable CLS token, add learnable positional
embeddings, run a small pre-norm Transformer encoder, take the CLS token's
final state as the sequence representation. Same downstream tail (concat hand
features -> fc1 -> ReLU -> Dropout -> fc2 -> L2-normalize) as model.py's
Embedding and zplane_backbone.py's ZPlaneEmbedding, so it plugs into the exact
same prototype/few-shot harness (train_common(_v2).train, evaluate_ab_v2) with
no other code changes.
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

from model import EMBED_DIM, N_FEATURES  # noqa: E402 -- constants only, never import Embedding itself

INPUT_LEN = 1024
HIDDEN = 96
DROPOUT = 0.2


class TransformerBlock(nn.Module):
    """Pre-norm transformer encoder block: MHSA then MLP, both residual."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEmbedding(nn.Module):
    """Drop-in replacement for model.Embedding / zplane_backbone.ZPlaneEmbedding:
    same forward(x, feat) contract, same fc1/fc2/normalize tail."""

    def __init__(self, patch_size: int = 16, embed_dim: int = 64, depth: int = 4,
                 num_heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.1,
                 in_len: int = INPUT_LEN, out_embed_dim: int = EMBED_DIM, n_features: int = N_FEATURES):
        super().__init__()
        assert in_len % patch_size == 0, "in_len must be divisible by patch_size"
        self.patch_size = patch_size
        self.n_patches = in_len // patch_size
        self.embed_dim = embed_dim

        # patch embedding: flatten (I,Q) over each patch -> linear projection
        # (equivalent to a Conv1d(2, embed_dim, kernel=patch_size, stride=patch_size))
        self.patch_embed = nn.Conv1d(2, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
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
        self.embed_dim_out = out_embed_dim
        self.n_features = n_features

    def forward(self, x: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """x: float32 (B,2,in_len), feat: float32 (B,n_features) -- SAME
        signature as model.Embedding.forward / ZPlaneEmbedding.forward."""
        B = x.shape[0]
        patches = self.patch_embed(x)                     # (B, embed_dim, n_patches)
        patches = patches.transpose(1, 2)                  # (B, n_patches, embed_dim)
        cls = self.cls_token.expand(B, -1, -1)              # (B, 1, embed_dim)
        h = torch.cat([cls, patches], dim=1)                # (B, n_patches+1, embed_dim)
        h = h + self.pos_embed
        h = self.pos_drop(h)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        cls_out = h[:, 0]                                   # (B, embed_dim) -- CLS token's final state

        h = torch.cat([cls_out, feat], dim=-1)
        h = F.relu(self.fc1(h))
        h = self.drop(h)
        h = self.fc2(h)
        return F.normalize(h, dim=-1)


if __name__ == "__main__":
    torch.manual_seed(0)
    net = ViTEmbedding()
    n_params = sum(p.numel() for p in net.parameters())
    print(f"[ok] ViTEmbedding param count: {n_params}")
    xb = torch.randn(5, 2, INPUT_LEN)
    fb = torch.randn(5, N_FEATURES)
    out = net(xb, fb)
    assert out.shape == (5, EMBED_DIM)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(5), atol=1e-4)
    print(f"[ok] forward shape {tuple(out.shape)}, unit-norm ok")
