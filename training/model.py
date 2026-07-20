"""Compact 1D-CNN metric embedding over normalised complex I/Q.

Input is [B, 2, L] (I, Q channels); output is an L2-normalised D-dim embedding.
Four strided conv blocks (Conv->BN->ReLU) downsample the time axis, global
average pooling removes the remaining length dependence, and a two-layer head
projects to the embedding. BatchNorm is folded into the preceding convolution at
export time so the TypeScript inference port only has to implement plain
conv/relu/linear — no BN bookkeeping, no runtime statistics.

~50k parameters; the exported asset is a few hundred KB.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_LEN = 1024
EMBED_DIM = 32
N_FEATURES = 12  # phase-invariant cumulant/amplitude/inst-freq features (preprocess.iq_features)
# (in, out, kernel, stride, pad)
CONV_SPEC = [
    (2, 16, 7, 2, 3),
    (16, 32, 5, 2, 2),
    (32, 64, 3, 2, 1),
    (64, 64, 3, 2, 1),
]


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k, s, p):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm1d(cout)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class Embedding(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM, n_features: int = N_FEATURES,
                 hidden: int = 96, dropout: float = 0.2):
        super().__init__()
        self.blocks = nn.ModuleList([ConvBlock(*spec) for spec in CONV_SPEC])
        cfinal = CONV_SPEC[-1][1]
        # mean+std pooling: the std branch exposes the amplitude/feature
        # *distribution* the head needs to separate QAM orders (3 vs 9 amplitude
        # levels) and constant- vs varying-envelope classes (CW vs AM). The
        # cumulant feature vector is concatenated in alongside it.
        self.fc1 = nn.Linear(2 * cfinal + n_features, hidden)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, embed_dim)
        self.embed_dim = embed_dim
        self.n_features = n_features
        self.pool = "mean_std"

    def forward(self, x, feat):
        for b in self.blocks:
            x = b(x)
        mean = x.mean(dim=-1)
        std = x.std(dim=-1, unbiased=False)  # population std -> matches numpy/TS
        h = torch.cat([mean, std, feat], dim=-1)
        h = F.relu(self.fc1(h))
        h = self.drop(h)
        h = self.fc2(h)
        return F.normalize(h, dim=-1)  # unit-norm embedding

    # ---- export: fold BN into conv, dump plain weights ----
    @torch.no_grad()
    def export_weights(self) -> dict:
        self.eval()
        convs = []
        for blk, spec in zip(self.blocks, CONV_SPEC):
            w = blk.conv.weight.detach().cpu().numpy()          # [out,in,k]
            bn = blk.bn
            gamma = bn.weight.detach().cpu().numpy()
            beta = bn.bias.detach().cpu().numpy()
            mean = bn.running_mean.detach().cpu().numpy()
            var = bn.running_var.detach().cpu().numpy()
            eps = bn.eps
            scale = gamma / np.sqrt(var + eps)
            w_folded = w * scale[:, None, None]
            b_folded = beta - mean * scale
            convs.append(
                {
                    "in": spec[0], "out": spec[1], "k": spec[2],
                    "stride": spec[3], "pad": spec[4],
                    "weight": w_folded.astype(np.float32).ravel().tolist(),
                    "bias": b_folded.astype(np.float32).tolist(),
                }
            )
        return {
            "input_len": INPUT_LEN,
            "embed_dim": self.embed_dim,
            "n_features": self.n_features,
            "pool": self.pool,
            "convs": convs,
            "fc1": {
                "in": self.fc1.in_features, "out": self.fc1.out_features,
                "weight": self.fc1.weight.detach().cpu().numpy().astype(np.float32).ravel().tolist(),
                "bias": self.fc1.bias.detach().cpu().numpy().astype(np.float32).tolist(),
            },
            "fc2": {
                "in": self.fc2.in_features, "out": self.fc2.out_features,
                "weight": self.fc2.weight.detach().cpu().numpy().astype(np.float32).ravel().tolist(),
                "bias": self.fc2.bias.detach().cpu().numpy().astype(np.float32).tolist(),
            },
        }
