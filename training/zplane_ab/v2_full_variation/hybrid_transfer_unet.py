"""Production-plausible hybrid of the transfer U-Net and incumbent I/Q CNN.

The corrected :class:`unet_transfer.TransferUNet` is useful as a denoiser and
as a latent probe, but its classification head has not matched the compact
real-I/Q :class:`model.Embedding` incumbent.  This module composes the two
without weakening either contract:

* the U-Net still runs, reconstructs I/Q, and exposes its pooled latent and
  auxiliary heads;
* the proven compact CNN is the embedding head returned to metric-learning
  callers; and
* ``classify_source="original"`` is an exact safe control.  Loading an
  incumbent ``Embedding.state_dict()`` makes the returned embedding identical
  to that incumbent in evaluation mode.

``classify_source="reconstruction"`` is the bounded experimental arm.  It
feeds the U-Net reconstruction to the same CNN while retaining the original
preprocessor features.  With the default residual decoder, the final complex
projection is zero-initialized, so the reconstruction is exactly the input and
this mode also starts exactly at the incumbent output.  It can then learn
whether denoising helps classification without paying an initialization
regression.

The public ``forward(x, feat) -> unit embedding`` contract and the
``last_recon``/latent attributes match the existing training and evaluation
harnesses.  The component names are intentionally explicit: a complete hybrid
checkpoint uses ``unet.*`` and ``classifier.*`` keys, while helper methods load
raw component checkpoints strictly.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from typing import Any, Literal

import torch
import torch.nn as nn


_HERE = os.path.dirname(os.path.abspath(__file__))
_ZPAB_DIR = os.path.dirname(_HERE)
_TRAINING_DIR = os.path.dirname(_ZPAB_DIR)
for _path in (_TRAINING_DIR, _ZPAB_DIR, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from model import EMBED_DIM, N_FEATURES, Embedding  # noqa: E402
from unet_transfer import TransferUNet  # noqa: E402


ClassificationSource = Literal["original", "reconstruction"]
_VALID_SOURCES = frozenset(("original", "reconstruction"))


def _unwrap_state_dict(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Unwrap the common ``{"state_dict": ...}`` checkpoint envelope."""
    state: Mapping[str, Any] = payload
    while (
        "state_dict" in state
        and isinstance(state["state_dict"], Mapping)
    ):
        state = state["state_dict"]
    return state


def _component_state_dict(
    payload: Mapping[str, Any],
    component: str,
) -> dict[str, Any]:
    """Extract a raw component state from raw, DataParallel, or hybrid keys."""
    state = dict(_unwrap_state_dict(payload))
    if not state:
        raise ValueError("checkpoint state_dict is empty")

    # A complete hybrid checkpoint may be wrapped by DataParallel or another
    # parent model.  Prefer extracting the explicitly named component.
    for prefix in (
        f"module.{component}.",
        f"model.{component}.",
        f"{component}.",
    ):
        selected = {
            key[len(prefix):]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if selected:
            return selected

    # Raw component checkpoints commonly have only a DataParallel/model
    # prefix.  Strip it only when every key shares it, so malformed mixed
    # checkpoints still fail loudly in ``load_state_dict(strict=True)``.
    for prefix in ("module.", "model."):
        if all(key.startswith(prefix) for key in state):
            state = {key[len(prefix):]: value for key, value in state.items()}
    return state


class HybridTransferUNet(nn.Module):
    """Transfer U-Net denoiser plus exact-load incumbent CNN classifier.

    Defaults reproduce the corrected residual-identity U-Net architecture used
    by ``run_corrected_unet.py`` and the shipping compact CNN architecture from
    ``model.py``.

    Parameters
    ----------
    classify_source:
        ``"original"`` returns the CNN embedding of ``x`` while still running
        the U-Net for reconstruction/latent outputs. ``"reconstruction"``
        classifies the reconstructed I/Q.
    normalize_reconstruction:
        Optionally restore unit complex RMS before classifying a learned
        reconstruction. It defaults off because the U-Net target is already in
        that coordinate system and disabling it preserves exact identity
        parity at initialization.
    """

    def __init__(
        self,
        *,
        base: int = 8,
        depth: int = 4,
        taps: int = 7,
        embed_dim: int = EMBED_DIM,
        n_features: int = N_FEATURES,
        n_profiles: int = 37,
        hidden: int = 128,
        magnorm: bool = True,
        modrelu_init: float = -2.0,
        skip_dropout: float = 0.5,
        feat_dropout: float = 0.5,
        antialias: bool = False,
        head_pool: str = "meanstd",
        residual_recon: bool = True,
        classifier_hidden: int = 96,
        classifier_dropout: float = 0.2,
        classify_source: ClassificationSource = "original",
        normalize_reconstruction: bool = False,
        reconstruction_eps: float = 1e-12,
    ):
        super().__init__()
        if classify_source not in _VALID_SOURCES:
            raise ValueError(
                f"classify_source must be one of {sorted(_VALID_SOURCES)}, "
                f"got {classify_source!r}"
            )
        if reconstruction_eps <= 0:
            raise ValueError("reconstruction_eps must be positive")

        self.unet = TransferUNet(
            base=base,
            depth=depth,
            taps=taps,
            embed_dim=embed_dim,
            n_features=n_features,
            n_profiles=n_profiles,
            hidden=hidden,
            magnorm=magnorm,
            modrelu_init=modrelu_init,
            skip_dropout=skip_dropout,
            feat_dropout=feat_dropout,
            antialias=antialias,
            head_pool=head_pool,
            residual_recon=residual_recon,
        )
        self.classifier = Embedding(
            embed_dim=embed_dim,
            n_features=n_features,
            hidden=classifier_hidden,
            dropout=classifier_dropout,
            in_channels=2,
        )
        # State dicts intentionally contain tensors only. Keep the complete
        # constructor contract alongside them for an exact run manifest.
        self._unet_architecture = {
            "base": int(base),
            "depth": int(depth),
            "taps": int(taps),
            "embed_dim": int(embed_dim),
            "n_features": int(n_features),
            "n_profiles": int(n_profiles),
            "hidden": int(hidden),
            "magnorm": bool(magnorm),
            "modrelu_init": float(modrelu_init),
            "skip_dropout": float(skip_dropout),
            "feat_dropout": float(feat_dropout),
            "antialias": bool(antialias),
            "head_pool": str(head_pool),
            "residual_recon": bool(residual_recon),
        }
        self._classifier_architecture = {
            "embed_dim": int(embed_dim),
            "n_features": int(n_features),
            "hidden": int(classifier_hidden),
            "dropout": float(classifier_dropout),
            "in_channels": 2,
            "pool": self.classifier.pool,
        }
        self._classify_source: ClassificationSource = classify_source
        self.normalize_reconstruction = bool(normalize_reconstruction)
        self.reconstruction_eps = float(reconstruction_eps)
        self.last_unet_embedding: torch.Tensor | None = None

    @property
    def classify_source(self) -> ClassificationSource:
        return self._classify_source

    def set_classify_source(
        self,
        source: ClassificationSource,
    ) -> "HybridTransferUNet":
        """Switch the classifier route without rebuilding or reloading weights."""
        if source not in _VALID_SOURCES:
            raise ValueError(
                f"classify_source must be one of {sorted(_VALID_SOURCES)}, "
                f"got {source!r}"
            )
        self._classify_source = source
        return self

    # Proxy the surfaces consumed by the transfer training/evaluation code.
    @property
    def last_recon(self) -> torch.Tensor | None:
        return self.unet.last_recon

    @property
    def last_bottleneck_pool(self) -> torch.Tensor | None:
        return self.unet.last_bottleneck_pool

    @property
    def last_latent(self) -> torch.Tensor | None:
        """Alias for the pooled U-Net bottleneck used by latent probes."""
        return self.unet.last_bottleneck_pool

    @property
    def latent(self) -> torch.Tensor | None:
        """Compatibility alias for evaluators that call the pooled latent ``latent``."""
        return self.unet.last_bottleneck_pool

    @property
    def last_profile_logits(self) -> torch.Tensor | None:
        return self.unet.last_profile_logits

    @property
    def last_params(self) -> torch.Tensor | None:
        return self.unet.last_params

    def config(self) -> dict[str, Any]:
        """Serializable model description for bounded-run manifests."""
        return {
            "kind": "hybrid_transfer_unet",
            "classify_source": self.classify_source,
            "normalize_reconstruction": self.normalize_reconstruction,
            "reconstruction_eps": self.reconstruction_eps,
            "unet": dict(self._unet_architecture),
            "classifier": dict(self._classifier_architecture),
        }

    def parameter_counts(self) -> dict[str, int]:
        """Return auditable component and total parameter counts."""
        unet = sum(parameter.numel() for parameter in self.unet.parameters())
        classifier = sum(
            parameter.numel() for parameter in self.classifier.parameters()
        )
        return {
            "unet": int(unet),
            "classifier": int(classifier),
            "total": int(unet + classifier),
            "trainable": int(
                sum(parameter.numel() for parameter in self.parameters()
                    if parameter.requires_grad)
            ),
        }

    def load_incumbent_state_dict(
        self,
        payload: Mapping[str, Any],
        *,
        strict: bool = True,
    ):
        """Load a raw ``Embedding`` or complete hybrid checkpoint into the CNN."""
        return self.classifier.load_state_dict(
            _component_state_dict(payload, "classifier"),
            strict=strict,
        )

    def load_unet_state_dict(
        self,
        payload: Mapping[str, Any],
        *,
        strict: bool = True,
    ):
        """Load a raw ``TransferUNet`` or complete hybrid checkpoint into the U-Net."""
        return self.unet.load_state_dict(
            _component_state_dict(payload, "unet"),
            strict=strict,
        )

    def _reconstruction_channels(self) -> torch.Tensor:
        reconstruction = self.last_recon
        if reconstruction is None:
            raise RuntimeError("U-Net did not expose last_recon")
        if self.normalize_reconstruction:
            rms = torch.sqrt(
                (
                    reconstruction.real.square()
                    + reconstruction.imag.square()
                ).mean(dim=-1, keepdim=True)
                + self.reconstruction_eps
            )
            reconstruction = reconstruction / rms
        return torch.stack(
            [reconstruction.real, reconstruction.imag],
            dim=1,
        )

    def forward_classifier_only(
        self,
        x: torch.Tensor,
        feat: torch.Tensor,
    ) -> torch.Tensor:
        """Fast production control that deliberately does not update U-Net attributes."""
        return self.classifier(x, feat)

    def forward_embedding(
        self,
        x: torch.Tensor,
        feat: torch.Tensor,
    ) -> torch.Tensor:
        """Decoder-free embedding path when classifying original I/Q.

        In reconstruction mode the decoder is part of the classifier route and
        cannot be skipped, so this method remains numerically identical to
        :meth:`forward` by executing the full U-Net.
        """
        if self.classify_source == "reconstruction":
            return self.forward(x, feat)

        # Run the authoritative classifier first. In training mode this also
        # preserves its dropout RNG behavior relative to a standalone
        # incumbent; U-Net-only random draws happen afterwards.
        embedding = self.classifier(x, feat)
        self.last_unet_embedding = self.unet.forward_embedding(x, feat)
        return embedding

    def forward(
        self,
        x: torch.Tensor,
        feat: torch.Tensor,
    ) -> torch.Tensor:
        """Return the unit CNN embedding and update every U-Net output surface."""
        if self.classify_source == "original":
            # Keeping the CNN first makes the safe arm an exact control even
            # when both branches have training-time dropout.
            embedding = self.classifier(x, feat)
            self.last_unet_embedding = self.unet(x, feat)
            return embedding

        self.last_unet_embedding = self.unet(x, feat)
        return self.classifier(self._reconstruction_channels(), feat)


__all__ = ["ClassificationSource", "HybridTransferUNet"]
