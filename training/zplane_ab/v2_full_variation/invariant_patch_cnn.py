"""Length- and bandwidth-scale-invariant patch-set embeddings for complex I/Q.

The companion front end maps raw sample index ``n`` to the dimensionless
coordinate ``u = n * occupied_bandwidth / TARGET_FRAC`` and then extracts a
fixed number of fixed-span patches.  This module applies one shared encoder to
every patch and aggregates patch representations with symmetric statistics.
Consequently:

* changing capture length changes which observations estimate the set
  statistics, never the network geometry;
* changing sample rate or occupied bandwidth is removed before this module;
* permuting the patches cannot change the embedding.

Two deliberately comparable encoders are exposed. ``real`` is the smallest
credible control and closely follows the shipped CNN. ``complex`` is
phase-equivariant internally and turns its final complex activations into
global-phase-invariant magnitude and autocorrelation statistics.  We keep both
because equivariance is a mathematical property, while classification quality
is an empirical question.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from complex_multiscale_backbone import ComplexConv1d, ComplexModReLU
from model import EMBED_DIM, N_FEATURES
from unet_transfer import ComplexMagNorm


@dataclass(frozen=True)
class InvariantPatchConfig:
    patch_length: int = 64
    patch_count: int = 16
    encoder: str = "real"
    patch_dim: int = 64
    hidden: int = 96
    embed_dim: int = EMBED_DIM
    n_features: int = N_FEATURES
    set_pool: str = "mean_std"
    dropout: float = 0.2

    def validate(self) -> "InvariantPatchConfig":
        if self.patch_length < 16:
            raise ValueError("patch_length must be at least 16")
        if self.patch_count <= 0:
            raise ValueError("patch_count must be positive")
        if self.encoder not in {"real", "complex"}:
            raise ValueError("encoder must be 'real' or 'complex'")
        if self.patch_dim <= 0 or self.hidden <= 0 or self.embed_dim <= 0:
            raise ValueError("patch_dim, hidden, and embed_dim must be positive")
        if self.n_features < 0:
            raise ValueError("n_features cannot be negative")
        if self.set_pool not in {"mean", "mean_std"}:
            raise ValueError("set_pool must be 'mean' or 'mean_std'")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        return self


class _RealBlock(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int):
        super().__init__()
        self.conv = nn.Conv1d(
            cin, cout, kernel, stride=2, padding=kernel // 2, bias=False
        )
        self.bn = nn.BatchNorm1d(cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class _RealPatchEncoder(nn.Module):
    output_dim = 2 * 64

    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(
            _RealBlock(2, 24, 7),
            _RealBlock(24, 48, 5),
            _RealBlock(48, 64, 3),
            _RealBlock(64, 64, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.blocks(x)
        return torch.cat(
            (y.mean(dim=-1), y.std(dim=-1, unbiased=False)), dim=-1
        )


def _complex_lowpass_decimate(z: torch.Tensor) -> torch.Tensor:
    """Complex-linear [1, 2, 1]/4 antialiasing followed by decimation by two."""
    channels = z.shape[1]
    kernel = z.real.new_tensor((0.25, 0.5, 0.25)).view(1, 1, 3)
    kernel = kernel.expand(channels, 1, 3)
    re = F.conv1d(F.pad(z.real, (1, 1), mode="replicate"), kernel, groups=channels)
    im = F.conv1d(F.pad(z.imag, (1, 1), mode="replicate"), kernel, groups=channels)
    return torch.complex(re[..., ::2], im[..., ::2])


class _ComplexStage(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int):
        super().__init__()
        self.conv = ComplexConv1d(cin, cout, kernel)
        self.norm = ComplexMagNorm()
        self.act = ComplexModReLU(cout)
        # A near-zero threshold avoids the dead-at-initialization failure measured
        # in the original U-Net while retaining a learnable phase-equivariant gate.
        with torch.no_grad():
            self.act.thresh_raw.fill_(-8.0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return _complex_lowpass_decimate(self.act(self.norm(self.conv(z))))


def _complex_patch_statistics(
    z: torch.Tensor, lags: tuple[int, ...] = (1, 2, 4)
) -> torch.Tensor:
    """Global-phase-invariant magnitude and complex autocorrelation statistics."""
    mag = z.abs()
    out = [mag.mean(dim=-1), mag.std(dim=-1, unbiased=False)]
    length = z.shape[-1]
    for lag in lags:
        if lag >= length:
            corr = torch.zeros(
                z.shape[0], z.shape[1], dtype=z.dtype, device=z.device
            )
        else:
            corr = (z[..., :-lag] * z[..., lag:].conj()).mean(dim=-1)
        out.extend((corr.real, corr.imag))
    return torch.cat(out, dim=-1)


class _ComplexPatchEncoder(nn.Module):
    channels = 48
    lags = (1, 2, 4)
    output_dim = channels * (2 + 2 * len(lags))

    def __init__(self):
        super().__init__()
        self.stages = nn.Sequential(
            _ComplexStage(1, 16, 7),
            _ComplexStage(16, 32, 5),
            _ComplexStage(32, self.channels, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.complex(x[:, 0], x[:, 1]).unsqueeze(1)
        return _complex_patch_statistics(self.stages(z), self.lags)


class InvariantPatchCNN(nn.Module):
    """Shared patch encoder plus permutation-invariant set pooling."""

    def __init__(self, config: InvariantPatchConfig | None = None, **overrides):
        super().__init__()
        if config is not None and overrides:
            raise ValueError("pass either config or keyword overrides, not both")
        self.cfg = (config or InvariantPatchConfig(**overrides)).validate()
        if self.cfg.encoder == "real":
            self.patch_encoder = _RealPatchEncoder()
        else:
            self.patch_encoder = _ComplexPatchEncoder()
        self.patch_projection = nn.Sequential(
            nn.Linear(self.patch_encoder.output_dim, self.cfg.patch_dim),
            nn.ReLU(),
        )
        set_dim = self.cfg.patch_dim * (
            2 if self.cfg.set_pool == "mean_std" else 1
        )
        self.fc1 = nn.Linear(set_dim + self.cfg.n_features, self.cfg.hidden)
        self.dropout = nn.Dropout(self.cfg.dropout)
        self.fc2 = nn.Linear(self.cfg.hidden, self.cfg.embed_dim)

    @property
    def packed_length(self) -> int:
        return self.cfg.patch_count * self.cfg.patch_length

    def config(self) -> dict:
        return asdict(self.cfg)

    def _unpack(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 2:
            raise ValueError(f"expected I/Q tensor [B,2,L], got {tuple(x.shape)}")
        if x.shape[2] != self.packed_length:
            raise ValueError(
                f"expected packed length {self.packed_length}, got {x.shape[2]}"
            )
        batch = x.shape[0]
        # The front end packs each channel as [patch0 | patch1 | ...].
        return (
            x.reshape(batch, 2, self.cfg.patch_count, self.cfg.patch_length)
            .permute(0, 2, 1, 3)
            .reshape(batch * self.cfg.patch_count, 2, self.cfg.patch_length)
        )

    def patch_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        encoded = self.patch_projection(self.patch_encoder(self._unpack(x)))
        return encoded.reshape(batch, self.cfg.patch_count, self.cfg.patch_dim)

    def forward(self, x: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        if feat.ndim != 2 or feat.shape != (x.shape[0], self.cfg.n_features):
            raise ValueError(
                f"expected features [{x.shape[0]},{self.cfg.n_features}], "
                f"got {tuple(feat.shape)}"
            )
        patches = self.patch_embeddings(x)
        pooled = patches.mean(dim=1)
        if self.cfg.set_pool == "mean_std":
            pooled = torch.cat(
                (pooled, patches.std(dim=1, unbiased=False)), dim=-1
            )
        hidden = F.relu(self.fc1(torch.cat((pooled, feat), dim=-1)))
        embedding = self.fc2(self.dropout(hidden))
        return F.normalize(embedding, dim=-1)


def _selftest() -> None:
    torch.manual_seed(4)
    for encoder in ("real", "complex"):
        net = InvariantPatchCNN(encoder=encoder).eval()
        x = torch.randn(3, 2, net.packed_length)
        feat = torch.randn(3, N_FEATURES)
        with torch.no_grad():
            y = net(x, feat)
        assert y.shape == (3, EMBED_DIM)
        assert torch.allclose(y.norm(dim=-1), torch.ones(3), atol=1e-6)

        # Patch order is a set property, not a learned augmentation.
        patches = x.reshape(3, 2, net.cfg.patch_count, net.cfg.patch_length)
        perm = torch.randperm(net.cfg.patch_count)
        xp = patches[:, :, perm].reshape_as(x)
        with torch.no_grad():
            yp = net(xp, feat)
        assert torch.allclose(y, yp, atol=2e-6, rtol=1e-6)

        if encoder == "complex":
            phi = 1.234
            z = torch.complex(x[:, 0], x[:, 1]) * torch.exp(
                torch.tensor(1j * phi)
            )
            xr = torch.stack((z.real, z.imag), dim=1)
            with torch.no_grad():
                yr = net(xr, feat)
            assert torch.allclose(y, yr, atol=3e-5, rtol=3e-5)
    print("invariant_patch_cnn selftest: ok")


if __name__ == "__main__":
    _selftest()
