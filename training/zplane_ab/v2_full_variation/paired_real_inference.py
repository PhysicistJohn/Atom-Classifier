"""ONNX-friendly paired-real inference for :class:`unet_transfer.TransferUNet`.

The trained model uses PyTorch complex tensors internally.  That is convenient
for expressing the RF-equivariant architecture, but ``aten::complex`` is not an
ONNX operator.  This module represents every complex activation as a
``(real, imaginary)`` tensor pair and implements the same arithmetic entirely
with real-valued PyTorch operators.

The parameter/module names deliberately mirror ``TransferUNet``.  Consequently
the original checkpoint state dict loads with ``strict=True``; there is no
conversion or numerically lossy weight rewrite.

Inference output contract
-------------------------
``forward(iq, features)`` returns four real tensors:

* unit-normalized 32-dimensional embedding;
* pooled 256-dimensional bottleneck latent (for the corrected mean/std model);
* denoised real channel;
* denoised imaginary channel.

The denoised channels are in the normalized 1024-sample preprocessing frame.
They are not an inverse-resampled raw capture.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


ACF_LAGS = (1, 2, 4, 8, 16, 32)
N_PARAM_OUTPUTS = 6


class PairedRealComplexConv1d(nn.Module):
    """Complex convolution represented by two ordinary real convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv_re = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=pad,
            bias=False,
        )
        self.conv_im = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=pad,
            bias=False,
        )

    def forward(
        self,
        real: torch.Tensor,
        imaginary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out_real = self.conv_re(real) - self.conv_im(imaginary)
        out_imaginary = self.conv_re(imaginary) + self.conv_im(real)
        return out_real, out_imaginary


class PairedRealMagNorm(nn.Module):
    """Per-(batch, channel) complex RMS normalization over time."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        real: torch.Tensor,
        imaginary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rms = torch.sqrt(
            (real.square() + imaginary.square()).mean(dim=-1, keepdim=True)
            + self.eps
        )
        return real / rms, imaginary / rms


class PairedRealModReLU(nn.Module):
    """The exact magnitude gate used by ``ComplexModReLU``, without complex dtype."""

    def __init__(self, channels: int):
        super().__init__()
        self.thresh_raw = nn.Parameter(torch.full((channels,), -2.0))

    def forward(
        self,
        real: torch.Tensor,
        imaginary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bias = -F.softplus(self.thresh_raw).view(1, -1, 1)
        magnitude = torch.sqrt(real.square() + imaginary.square() + 1e-12)
        scale = torch.clamp(magnitude + bias, min=0.0) / magnitude
        return scale * real, scale * imaginary


class PairedRealCBlock(nn.Module):
    """Paired-real complex conv -> optional MagNorm -> modReLU, twice."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        taps: int = 7,
        magnorm: bool = False,
        modrelu_init: float = -2.0,
    ):
        super().__init__()
        self.c1 = PairedRealComplexConv1d(in_channels, out_channels, taps)
        self.a1 = PairedRealModReLU(out_channels)
        self.c2 = PairedRealComplexConv1d(out_channels, out_channels, taps)
        self.a2 = PairedRealModReLU(out_channels)
        self.n1 = PairedRealMagNorm() if magnorm else None
        self.n2 = PairedRealMagNorm() if magnorm else None
        if modrelu_init != -2.0:
            with torch.no_grad():
                self.a1.thresh_raw.fill_(modrelu_init)
                self.a2.thresh_raw.fill_(modrelu_init)

    def forward(
        self,
        real: torch.Tensor,
        imaginary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        real, imaginary = self.c1(real, imaginary)
        if self.n1 is not None:
            real, imaginary = self.n1(real, imaginary)
        real, imaginary = self.a1(real, imaginary)
        real, imaginary = self.c2(real, imaginary)
        if self.n2 is not None:
            real, imaginary = self.n2(real, imaginary)
        return self.a2(real, imaginary)


def _lowpass121(x: torch.Tensor) -> torch.Tensor:
    """Fixed [1, 2, 1]/4 low-pass on one real component."""
    channels = x.shape[1]
    kernel = x.new_tensor([0.25, 0.5, 0.25]).view(1, 1, 3)
    kernel = kernel.expand(channels, 1, 3)
    return F.conv1d(F.pad(x, (1, 1), mode="replicate"), kernel, groups=channels)


def _chunk_pool(magnitude: torch.Tensor, chunks: int = 4) -> torch.Tensor:
    batch, channels, length = magnitude.shape
    usable = (length // chunks) * chunks
    cells = magnitude[..., :usable].reshape(
        batch,
        channels,
        chunks,
        length // chunks,
    )
    return torch.cat(
        [
            cells.mean(dim=-1).reshape(batch, -1),
            cells.std(dim=-1).reshape(batch, -1),
        ],
        dim=-1,
    )


def _acf_features(
    real: torch.Tensor,
    imaginary: torch.Tensor,
    lags: tuple[int, ...] = ACF_LAGS,
) -> torch.Tensor:
    """Magnitude of mean(v[t] * conj(v[t+k])) for each channel and lag."""
    rows = []
    length = real.shape[-1]
    for lag in lags:
        if lag >= length:
            rows.append(real.new_zeros(real.shape[0], real.shape[1]))
            continue
        product_real = (
            real[..., :-lag] * real[..., lag:]
            + imaginary[..., :-lag] * imaginary[..., lag:]
        )
        product_imaginary = (
            imaginary[..., :-lag] * real[..., lag:]
            - real[..., :-lag] * imaginary[..., lag:]
        )
        mean_real = product_real.mean(dim=-1)
        mean_imaginary = product_imaginary.mean(dim=-1)
        rows.append(torch.sqrt(mean_real.square() + mean_imaginary.square()))
    return torch.cat(rows, dim=-1)


class PairedRealTransferUNet(nn.Module):
    """Real-operator inference twin of ``TransferUNet``.

    All training-only dropout flags remain part of the reconstructable
    architecture, but have no effect in ``eval`` mode.  Exporters must call
    ``eval()`` before tracing.
    """

    def __init__(
        self,
        base: int = 8,
        depth: int = 4,
        taps: int = 7,
        embed_dim: int = 32,
        n_features: int = 12,
        n_profiles: int = 37,
        hidden: int = 128,
        magnorm: bool = False,
        modrelu_init: float = -2.0,
        skip_dropout: float = 0.0,
        feat_dropout: float = 0.0,
        antialias: bool = False,
        head_pool: str = "meanstd",
        residual_recon: bool = False,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be positive")
        if head_pool not in {"meanstd", "chunk4", "acf"}:
            raise ValueError(f"unsupported head_pool {head_pool!r}")
        self.depth = int(depth)
        self.skip_dropout = float(skip_dropout)
        self.feat_dropout = float(feat_dropout)
        self.antialias = bool(antialias)
        self.head_pool = head_pool
        self.residual_recon = bool(residual_recon)

        channels = [base * (2**index) for index in range(depth)]
        self.downs = nn.ModuleList()
        in_channels = 1
        for out_channels in channels:
            self.downs.append(
                PairedRealCBlock(
                    in_channels,
                    out_channels,
                    taps,
                    magnorm,
                    modrelu_init,
                )
            )
            in_channels = out_channels
        self.bottleneck = PairedRealCBlock(
            channels[-1],
            channels[-1] * 2,
            taps,
            magnorm,
            modrelu_init,
        )

        self.ups = nn.ModuleList()
        in_channels = channels[-1] * 2
        for out_channels in reversed(channels):
            self.ups.append(
                PairedRealCBlock(
                    in_channels + out_channels,
                    out_channels,
                    taps,
                    magnorm,
                    modrelu_init,
                )
            )
            in_channels = out_channels
        self.out = PairedRealComplexConv1d(channels[0], 1, kernel_size=1)
        if self.residual_recon:
            nn.init.zeros_(self.out.conv_re.weight)
            nn.init.zeros_(self.out.conv_im.weight)

        bottleneck_channels = channels[-1] * 2
        pool_dim = {
            "meanstd": 2 * bottleneck_channels,
            "chunk4": 8 * bottleneck_channels,
            "acf": (2 + len(ACF_LAGS)) * bottleneck_channels,
        }[head_pool]
        self.pool_dim = pool_dim
        self.trunk = nn.Sequential(
            nn.Linear(pool_dim + n_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.embed_head = nn.Linear(hidden, embed_dim)
        # Retained so the original checkpoint loads strictly. These two heads are
        # not outputs of the deployment inference contract.
        self.profile_head = nn.Linear(hidden, n_profiles)
        self.param_head = nn.Linear(hidden, N_PARAM_OUTPUTS)

    def load_transfer_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
    ) -> None:
        """Load an original ``TransferUNet`` state dict with a strict key check."""
        incompatible = self.load_state_dict(state_dict, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise AssertionError(
                "strict state load unexpectedly reported incompatible keys: "
                f"{incompatible}"
            )

    def encode(
        self,
        iq: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        list[tuple[torch.Tensor, torch.Tensor]],
    ]:
        # Shape is a deployment contract, not data-dependent graph control flow.
        # Skip the Python-side assertion while an exporter is tracing so the
        # static channel dimension does not produce a TracerWarning.
        if not torch.jit.is_tracing() and (iq.ndim != 3 or iq.shape[1] != 2):
            raise ValueError(f"iq must have shape [batch, 2, length], got {iq.shape}")
        real = iq[:, 0, :].unsqueeze(1)
        imaginary = iq[:, 1, :].unsqueeze(1)
        skips = []
        for block in self.downs:
            real, imaginary = block(real, imaginary)
            skips.append((real, imaginary))
            if self.antialias:
                real = _lowpass121(real)
                imaginary = _lowpass121(imaginary)
            real = real[..., ::2]
            imaginary = imaginary[..., ::2]
        real, imaginary = self.bottleneck(real, imaginary)
        return real, imaginary, skips

    def pool_bottleneck(
        self,
        real: torch.Tensor,
        imaginary: torch.Tensor,
    ) -> torch.Tensor:
        magnitude = torch.sqrt(real.square() + imaginary.square())
        if self.head_pool == "meanstd":
            # Deliberately leave correction at PyTorch's default (sample std,
            # denominator L-1). TransferUNet uses the same default.
            return torch.cat(
                [magnitude.mean(dim=-1), magnitude.std(dim=-1)],
                dim=-1,
            )
        if self.head_pool == "chunk4":
            return _chunk_pool(magnitude, 4)
        return torch.cat(
            [
                magnitude.mean(dim=-1),
                magnitude.std(dim=-1),
                _acf_features(real, imaginary),
            ],
            dim=-1,
        )

    @staticmethod
    def _upsample(x: torch.Tensor, size: int) -> torch.Tensor:
        return F.interpolate(x, size=size, mode="nearest")

    def forward(
        self,
        iq: torch.Tensor,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        real, imaginary, skips = self.encode(iq)
        bottleneck_pool = self.pool_bottleneck(real, imaginary)
        hidden = self.trunk(torch.cat([bottleneck_pool, features], dim=-1))
        embedding = F.normalize(self.embed_head(hidden), dim=-1)

        for block, (skip_real, skip_imaginary) in zip(
            self.ups,
            reversed(skips),
        ):
            real = self._upsample(real, skip_real.shape[-1])
            imaginary = self._upsample(imaginary, skip_imaginary.shape[-1])
            real = torch.cat([real, skip_real], dim=1)
            imaginary = torch.cat([imaginary, skip_imaginary], dim=1)
            real, imaginary = block(real, imaginary)

        correction_real, correction_imaginary = self.out(real, imaginary)
        correction_real = correction_real.squeeze(1)
        correction_imaginary = correction_imaginary.squeeze(1)
        if self.residual_recon:
            reconstruction_real = iq[:, 0, :] + correction_real
            reconstruction_imaginary = iq[:, 1, :] + correction_imaginary
        else:
            reconstruction_real = correction_real
            reconstruction_imaginary = correction_imaginary
        return (
            embedding,
            bottleneck_pool,
            reconstruction_real,
            reconstruction_imaginary,
        )


def make_paired_real_model(
    architecture: Mapping[str, object],
    state_dict: Mapping[str, torch.Tensor],
) -> PairedRealTransferUNet:
    """Construct, strictly load, and freeze a paired-real inference model."""
    model = PairedRealTransferUNet(**dict(architecture))
    model.load_transfer_state_dict(state_dict)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
