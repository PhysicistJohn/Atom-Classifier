"""Paired-real inference twin for the invariant patch CNN and centered fusion.

The training model expresses its complex branch with PyTorch complex tensors.
This module implements the identical inference graph using only real-valued
operators and a ``(real, imaginary)`` activation pair.  Parameter-bearing
module names intentionally match :mod:`invariant_patch_cnn`, so original
branch checkpoints and complete :class:`invariant_fusion.CenteredInvariantFusion`
state dictionaries load with strict key checks and without rewriting weights.

This file is an inference boundary only.  It does not load data, estimate
centers, fit prototypes, or choose fusion hyperparameters.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from invariant_patch_cnn import InvariantPatchConfig


class PairedRealComplexConv1d(nn.Module):
    """Complex convolution implemented as four ordinary real convolutions."""

    def __init__(self, cin: int, cout: int, kernel: int):
        super().__init__()
        padding = kernel // 2
        self.conv_re = nn.Conv1d(
            cin, cout, kernel, padding=padding, bias=False
        )
        self.conv_im = nn.Conv1d(
            cin, cout, kernel, padding=padding, bias=False
        )

    def forward(
        self, real: torch.Tensor, imaginary: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output_real = self.conv_re(real) - self.conv_im(imaginary)
        output_imaginary = self.conv_re(imaginary) + self.conv_im(real)
        return output_real, output_imaginary


class PairedRealMagNorm(nn.Module):
    """Per-(batch, channel) complex RMS normalization over time."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(
        self, real: torch.Tensor, imaginary: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rms = torch.sqrt(
            (real.square() + imaginary.square()).mean(dim=-1, keepdim=True)
            + self.eps
        )
        return real / rms, imaginary / rms


class PairedRealModReLU(nn.Module):
    """Magnitude gate equivalent to ``ComplexModReLU`` without complex dtype."""

    def __init__(self, channels: int):
        super().__init__()
        self.thresh_raw = nn.Parameter(torch.full((channels,), -8.0))

    def forward(
        self, real: torch.Tensor, imaginary: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bias = -F.softplus(self.thresh_raw).view(1, -1, 1)
        magnitude = torch.sqrt(real.square() + imaginary.square() + 1e-12)
        scale = torch.clamp(magnitude + bias, min=0.0) / magnitude
        return scale * real, scale * imaginary


def _lowpass_decimate(x: torch.Tensor) -> torch.Tensor:
    """Replicate-padded fixed [1, 2, 1]/4 filter followed by decimation."""
    channels = x.shape[1]
    kernel = x.new_tensor((0.25, 0.5, 0.25)).view(1, 1, 3)
    kernel = kernel.expand(channels, 1, 3)
    filtered = F.conv1d(
        F.pad(x, (1, 1), mode="replicate"),
        kernel,
        groups=channels,
    )
    return filtered[..., ::2]


class _PairedRealComplexStage(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int):
        super().__init__()
        self.conv = PairedRealComplexConv1d(cin, cout, kernel)
        self.norm = PairedRealMagNorm()
        self.act = PairedRealModReLU(cout)

    def forward(
        self, real: torch.Tensor, imaginary: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        real, imaginary = self.conv(real, imaginary)
        real, imaginary = self.norm(real, imaginary)
        real, imaginary = self.act(real, imaginary)
        return _lowpass_decimate(real), _lowpass_decimate(imaginary)


def _complex_patch_statistics(
    real: torch.Tensor,
    imaginary: torch.Tensor,
    lags: tuple[int, ...] = (1, 2, 4),
    *,
    expected_length: int,
) -> torch.Tensor:
    magnitude = torch.sqrt(real.square() + imaginary.square())
    output = [
        magnitude.mean(dim=-1),
        magnitude.std(dim=-1, unbiased=False),
    ]
    for lag in lags:
        if lag >= expected_length:
            correlation_real = real.new_zeros(real.shape[:2])
            correlation_imaginary = imaginary.new_zeros(imaginary.shape[:2])
        else:
            # (a + ib) * conj(c + id) = (ac + bd) + i(bc - ad)
            correlation_real = (
                real[..., :-lag] * real[..., lag:]
                + imaginary[..., :-lag] * imaginary[..., lag:]
            ).mean(dim=-1)
            correlation_imaginary = (
                imaginary[..., :-lag] * real[..., lag:]
                - real[..., :-lag] * imaginary[..., lag:]
            ).mean(dim=-1)
        output.extend((correlation_real, correlation_imaginary))
    return torch.cat(output, dim=-1)


class _PairedRealComplexPatchEncoder(nn.Module):
    output_dim = 48 * 8
    lags = (1, 2, 4)

    def __init__(self, patch_length: int):
        super().__init__()
        expected_length = int(patch_length)
        for _ in range(3):
            expected_length = (expected_length + 1) // 2
        self.expected_length = expected_length
        self.stages = nn.ModuleList(
            (
                _PairedRealComplexStage(1, 16, 7),
                _PairedRealComplexStage(16, 32, 5),
                _PairedRealComplexStage(32, 48, 3),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real = x[:, 0, :].unsqueeze(1)
        imaginary = x[:, 1, :].unsqueeze(1)
        for stage in self.stages:
            real, imaginary = stage(real, imaginary)
        return _complex_patch_statistics(
            real,
            imaginary,
            self.lags,
            expected_length=self.expected_length,
        )


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
        encoded = self.blocks(x)
        return torch.cat(
            (
                encoded.mean(dim=-1),
                encoded.std(dim=-1, unbiased=False),
            ),
            dim=-1,
        )


class PairedRealInvariantPatchCNN(nn.Module):
    """Real-operator inference twin of ``InvariantPatchCNN``."""

    def __init__(
        self,
        config: InvariantPatchConfig | Mapping[str, Any] | None = None,
        **overrides: Any,
    ):
        super().__init__()
        if config is not None and overrides:
            raise ValueError("pass either config or keyword overrides, not both")
        if isinstance(config, Mapping):
            config = InvariantPatchConfig(**dict(config))
        self.cfg = (config or InvariantPatchConfig(**overrides)).validate()
        if self.cfg.encoder == "real":
            self.patch_encoder = _RealPatchEncoder()
        else:
            self.patch_encoder = _PairedRealComplexPatchEncoder(
                self.cfg.patch_length
            )
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

    def config(self) -> dict[str, Any]:
        return asdict(self.cfg)

    def load_branch_state_dict(
        self, state_dict: Mapping[str, torch.Tensor]
    ) -> None:
        incompatible = self.load_state_dict(state_dict, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise AssertionError(
                "strict branch state load unexpectedly reported incompatible "
                f"keys: {incompatible}"
            )

    def _unpack(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_tracing():
            if x.ndim != 3 or x.shape[1] != 2:
                raise ValueError(
                    f"expected I/Q tensor [B,2,L], got {tuple(x.shape)}"
                )
            if x.shape[2] != self.packed_length:
                raise ValueError(
                    f"expected packed length {self.packed_length}, got {x.shape[2]}"
                )
        batch = x.shape[0]
        return (
            x.reshape(
                batch,
                2,
                self.cfg.patch_count,
                self.cfg.patch_length,
            )
            .permute(0, 2, 1, 3)
            .reshape(
                batch * self.cfg.patch_count,
                2,
                self.cfg.patch_length,
            )
        )

    def forward(self, x: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_tracing() and (
            feat.ndim != 2
            or feat.shape != (x.shape[0], self.cfg.n_features)
        ):
            raise ValueError(
                f"expected features [{x.shape[0]},{self.cfg.n_features}], "
                f"got {tuple(feat.shape)}"
            )
        batch = x.shape[0]
        patches = self.patch_projection(self.patch_encoder(self._unpack(x)))
        patches = patches.reshape(
            batch, self.cfg.patch_count, self.cfg.patch_dim
        )
        pooled = patches.mean(dim=1)
        if self.cfg.set_pool == "mean_std":
            pooled = torch.cat(
                (pooled, patches.std(dim=1, unbiased=False)),
                dim=-1,
            )
        hidden = F.relu(self.fc1(torch.cat((pooled, feat), dim=-1)))
        embedding = self.fc2(self.dropout(hidden))
        return F.normalize(embedding, dim=-1)


def _safe_unit(x: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    norm_squared = x.square().sum(dim=-1, keepdim=True)
    normalized = x * torch.rsqrt(norm_squared.clamp_min(eps.square()))
    fallback = torch.cat(
        (
            torch.ones_like(x[..., :1]),
            torch.zeros_like(x[..., 1:]),
        ),
        dim=-1,
    )
    return torch.where(norm_squared > eps.square(), normalized, fallback)


class PairedRealCenteredInvariantFusion(nn.Module):
    """Paired-real twin of ``CenteredInvariantFusion``."""

    def __init__(
        self,
        real_branch: PairedRealInvariantPatchCNN,
        complex_branch: PairedRealInvariantPatchCNN,
        real_center: Any,
        complex_center: Any,
        *,
        alpha_real: float = 1.0,
        alpha_complex: float = 1.0,
        weight_real: float = 0.5,
        eps: float = 1e-12,
    ):
        super().__init__()
        if real_branch.cfg.encoder != "real":
            raise ValueError("real_branch must use encoder='real'")
        if complex_branch.cfg.encoder != "complex":
            raise ValueError("complex_branch must use encoder='complex'")
        for field in ("patch_length", "patch_count", "n_features", "embed_dim"):
            if getattr(real_branch.cfg, field) != getattr(
                complex_branch.cfg, field
            ):
                raise ValueError(f"branch {field} mismatch")
        if not math.isfinite(float(weight_real)) or not 0.0 <= float(
            weight_real
        ) <= 1.0:
            raise ValueError("weight_real must lie in [0, 1]")
        if not math.isfinite(float(eps)) or float(eps) <= 0.0:
            raise ValueError("eps must be positive")
        for name, value in (
            ("alpha_real", alpha_real),
            ("alpha_complex", alpha_complex),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")

        reference = next(real_branch.parameters())
        dtype, device = reference.dtype, reference.device
        self.real_branch = real_branch
        self.complex_branch = complex_branch
        embed_dim = real_branch.cfg.embed_dim
        centers = {}
        for name, value in (
            ("real_center", real_center),
            ("complex_center", complex_center),
        ):
            tensor = torch.as_tensor(value, dtype=dtype, device=device)
            if tensor.shape != (embed_dim,) or not bool(
                torch.isfinite(tensor).all()
            ):
                raise ValueError(f"{name} must be a finite vector [{embed_dim}]")
            centers[name] = tensor.detach().clone()
        self.register_buffer("real_center", centers["real_center"])
        self.register_buffer("complex_center", centers["complex_center"])
        self.register_buffer(
            "alpha_real",
            torch.tensor(alpha_real, dtype=dtype, device=device),
        )
        self.register_buffer(
            "alpha_complex",
            torch.tensor(alpha_complex, dtype=dtype, device=device),
        )
        self.register_buffer(
            "weight_real",
            torch.tensor(weight_real, dtype=dtype, device=device),
        )
        self.register_buffer(
            "eps",
            torch.tensor(eps, dtype=dtype, device=device),
        )

    @property
    def packed_length(self) -> int:
        return self.real_branch.packed_length

    @property
    def embed_dim(self) -> int:
        return 2 * self.real_branch.cfg.embed_dim

    def forward(self, x: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        real = _safe_unit(
            self.real_branch(x, feat)
            - self.alpha_real * self.real_center.unsqueeze(0),
            self.eps,
        )
        complex_embedding = _safe_unit(
            self.complex_branch(x, feat)
            - self.alpha_complex * self.complex_center.unsqueeze(0),
            self.eps,
        )
        fused = torch.cat(
            (
                torch.sqrt(self.weight_real) * real,
                torch.sqrt(1.0 - self.weight_real) * complex_embedding,
            ),
            dim=-1,
        )
        return _safe_unit(fused, self.eps)


def make_paired_real_branch(
    config: InvariantPatchConfig | Mapping[str, Any],
    state_dict: Mapping[str, torch.Tensor],
) -> PairedRealInvariantPatchCNN:
    model = PairedRealInvariantPatchCNN(config)
    model.load_branch_state_dict(state_dict)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def make_paired_real_fusion(
    real_config: InvariantPatchConfig | Mapping[str, Any],
    complex_config: InvariantPatchConfig | Mapping[str, Any],
    state_dict: Mapping[str, torch.Tensor],
) -> PairedRealCenteredInvariantFusion:
    """Construct and strictly load a complete paired-real fusion state dict."""
    real = PairedRealInvariantPatchCNN(real_config)
    complex_branch = PairedRealInvariantPatchCNN(complex_config)
    embed_dim = real.cfg.embed_dim
    model = PairedRealCenteredInvariantFusion(
        real,
        complex_branch,
        torch.zeros(embed_dim),
        torch.zeros(embed_dim),
    )
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise AssertionError(
            "strict fusion state load unexpectedly reported incompatible "
            f"keys: {incompatible}"
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
