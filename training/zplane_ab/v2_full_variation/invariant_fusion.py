"""Auditable centered late fusion for the real and complex invariant CNNs.

The branch centers must be estimated from training embeddings only.  They are
stored as buffers so a fusion checkpoint contains the exact centering state
used at inference.  Fusion deliberately has no learned parameters:

    r = unit(real(x)    - alpha_real    * center_real)
    c = unit(complex(x) - alpha_complex * center_complex)
    y = unit([sqrt(weight_real) * r,
              sqrt(1 - weight_real) * c])

The final normalization is mathematically redundant for ordinary finite branch
outputs, but makes the unit-output contract explicit and robust to rounding.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from invariant_patch_cnn import InvariantPatchCNN


def _branch_dtype_and_device(branch: nn.Module) -> tuple[torch.dtype, torch.device]:
    try:
        parameter = next(branch.parameters())
    except StopIteration as exc:  # pragma: no cover - InvariantPatchCNN has parameters
        raise ValueError("fusion branches must contain parameters") from exc
    if not parameter.is_floating_point():
        raise ValueError("fusion branch parameters must use a floating dtype")
    return parameter.dtype, parameter.device


def _finite_scalar(name: str, value: float) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be a finite scalar")
    return scalar


def _center_tensor(
    name: str,
    value: Any,
    *,
    embed_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    try:
        center = torch.as_tensor(value, dtype=dtype, device=device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must be a finite vector [{embed_dim}]") from exc
    if center.ndim != 1 or center.shape[0] != embed_dim:
        raise ValueError(
            f"{name} must have shape [{embed_dim}], got {tuple(center.shape)}"
        )
    if not bool(torch.isfinite(center).all()):
        raise ValueError(f"{name} must contain only finite values")
    return center.detach().clone()


def _safe_unit(x: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """Normalize rows, using the first basis vector for an exact cancellation."""
    norm_sq = x.square().sum(dim=-1, keepdim=True)
    normalized = x * torch.rsqrt(norm_sq.clamp_min(eps.square()))
    fallback = torch.cat(
        (torch.ones_like(x[..., :1]), torch.zeros_like(x[..., 1:])), dim=-1
    )
    return torch.where(norm_sq > eps.square(), normalized, fallback)


class CenteredInvariantFusion(nn.Module):
    """Center and late-fuse one real and one complex ``InvariantPatchCNN``.

    ``real_center`` and ``complex_center`` are inference assets, not statistics
    computed by this module.  Callers are responsible for deriving them from
    training rows only.
    """

    def __init__(
        self,
        real_branch: InvariantPatchCNN,
        complex_branch: InvariantPatchCNN,
        real_center: Any,
        complex_center: Any,
        *,
        alpha_real: float = 1.0,
        alpha_complex: float = 1.0,
        weight_real: float = 0.5,
        eps: float = 1e-12,
    ):
        super().__init__()
        if not isinstance(real_branch, InvariantPatchCNN):
            raise TypeError("real_branch must be an InvariantPatchCNN")
        if not isinstance(complex_branch, InvariantPatchCNN):
            raise TypeError("complex_branch must be an InvariantPatchCNN")
        if real_branch.cfg.encoder != "real":
            raise ValueError("real_branch must use encoder='real'")
        if complex_branch.cfg.encoder != "complex":
            raise ValueError("complex_branch must use encoder='complex'")

        geometry_fields = ("patch_length", "patch_count", "n_features")
        for field in geometry_fields:
            real_value = getattr(real_branch.cfg, field)
            complex_value = getattr(complex_branch.cfg, field)
            if real_value != complex_value:
                raise ValueError(
                    f"branch {field} mismatch: {real_value} != {complex_value}"
                )
        if real_branch.cfg.embed_dim != complex_branch.cfg.embed_dim:
            raise ValueError(
                "branch embed_dim mismatch: "
                f"{real_branch.cfg.embed_dim} != {complex_branch.cfg.embed_dim}"
            )

        real_dtype, real_device = _branch_dtype_and_device(real_branch)
        complex_dtype, complex_device = _branch_dtype_and_device(complex_branch)
        if real_dtype != complex_dtype:
            raise ValueError(
                f"branch dtype mismatch: {real_dtype} != {complex_dtype}"
            )
        if real_device != complex_device:
            raise ValueError(
                f"branch device mismatch: {real_device} != {complex_device}"
            )

        alpha_real_value = _finite_scalar("alpha_real", alpha_real)
        alpha_complex_value = _finite_scalar("alpha_complex", alpha_complex)
        weight_real_value = _finite_scalar("weight_real", weight_real)
        eps_value = _finite_scalar("eps", eps)
        if not 0.0 <= weight_real_value <= 1.0:
            raise ValueError("weight_real must lie in [0, 1]")
        if eps_value <= 0.0:
            raise ValueError("eps must be positive")

        self.real_branch = real_branch
        self.complex_branch = complex_branch
        embed_dim = real_branch.cfg.embed_dim
        self.register_buffer(
            "real_center",
            _center_tensor(
                "real_center",
                real_center,
                embed_dim=embed_dim,
                dtype=real_dtype,
                device=real_device,
            ),
        )
        self.register_buffer(
            "complex_center",
            _center_tensor(
                "complex_center",
                complex_center,
                embed_dim=embed_dim,
                dtype=real_dtype,
                device=real_device,
            ),
        )
        # Scalar buffers make the complete inference rule state-dict resident.
        self.register_buffer(
            "alpha_real",
            torch.tensor(alpha_real_value, dtype=real_dtype, device=real_device),
        )
        self.register_buffer(
            "alpha_complex",
            torch.tensor(alpha_complex_value, dtype=real_dtype, device=real_device),
        )
        self.register_buffer(
            "weight_real",
            torch.tensor(weight_real_value, dtype=real_dtype, device=real_device),
        )
        self.register_buffer(
            "eps", torch.tensor(eps_value, dtype=real_dtype, device=real_device)
        )

    @property
    def packed_length(self) -> int:
        return self.real_branch.packed_length

    @property
    def embed_dim(self) -> int:
        return 2 * self.real_branch.cfg.embed_dim

    def config(self) -> dict[str, Any]:
        return {
            "kind": "centered_invariant_fusion",
            "alpha_real": float(self.alpha_real.detach().cpu()),
            "alpha_complex": float(self.alpha_complex.detach().cpu()),
            "weight_real": float(self.weight_real.detach().cpu()),
            "eps": float(self.eps.detach().cpu()),
            "embed_dim": self.embed_dim,
            "real": self.real_branch.config(),
            "complex": self.complex_branch.config(),
        }

    def parameter_counts(self) -> dict[str, int]:
        real = sum(parameter.numel() for parameter in self.real_branch.parameters())
        complex_count = sum(
            parameter.numel() for parameter in self.complex_branch.parameters()
        )
        return {
            "real": int(real),
            "complex": int(complex_count),
            "fusion": 0,
            "total": int(real + complex_count),
            "trainable": int(
                sum(
                    parameter.numel()
                    for parameter in self.parameters()
                    if parameter.requires_grad
                )
            ),
        }

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ):
        """Load only finite, structurally valid inference state."""
        for name, value in state_dict.items():
            if isinstance(value, torch.Tensor) and (
                value.is_floating_point() or value.is_complex()
            ):
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(f"state_dict tensor {name!r} is non-finite")
        embed_dim = self.real_branch.cfg.embed_dim
        for name in ("real_center", "complex_center"):
            if name in state_dict and tuple(state_dict[name].shape) != (embed_dim,):
                raise ValueError(
                    f"state_dict {name} must have shape [{embed_dim}], "
                    f"got {tuple(state_dict[name].shape)}"
                )
        if "weight_real" in state_dict:
            weight = _finite_scalar("state_dict weight_real", state_dict["weight_real"])
            if not 0.0 <= weight <= 1.0:
                raise ValueError("state_dict weight_real must lie in [0, 1]")
        if "eps" in state_dict:
            loaded_eps = _finite_scalar("state_dict eps", state_dict["eps"])
            if loaded_eps <= 0.0:
                raise ValueError("state_dict eps must be positive")
        for name in ("alpha_real", "alpha_complex"):
            if name in state_dict:
                _finite_scalar(f"state_dict {name}", state_dict[name])
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    @staticmethod
    def _validate_forward_tensor(name: str, value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        # Tensor-to-bool checks are intentionally eager-only.  The exported graph
        # remains pure tensor math while normal Python inference fails closed.
        if not torch.jit.is_tracing() and not torch.jit.is_scripting():
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must contain only finite values")

    def forward(self, x: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        self._validate_forward_tensor("x", x)
        self._validate_forward_tensor("feat", feat)
        real = self.real_branch(x, feat)
        complex_embedding = self.complex_branch(x, feat)
        if not torch.jit.is_tracing() and not torch.jit.is_scripting():
            expected_shape = (x.shape[0], self.real_branch.cfg.embed_dim)
            if tuple(real.shape) != expected_shape:
                raise RuntimeError(
                    f"real branch returned {tuple(real.shape)}, "
                    f"expected {expected_shape}"
                )
            if tuple(complex_embedding.shape) != expected_shape:
                raise RuntimeError(
                    "complex branch returned "
                    f"{tuple(complex_embedding.shape)}, expected {expected_shape}"
                )
            if not bool(torch.isfinite(real).all()):
                raise RuntimeError("real branch returned non-finite values")
            if not bool(torch.isfinite(complex_embedding).all()):
                raise RuntimeError("complex branch returned non-finite values")

        real = _safe_unit(
            real - self.alpha_real * self.real_center.unsqueeze(0), self.eps
        )
        complex_embedding = _safe_unit(
            complex_embedding
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


# Concise alias for callers that do not need to emphasize centering in the name.
InvariantFusion = CenteredInvariantFusion
