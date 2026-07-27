"""Strict time-domain geometry path for invariant patch preprocessing.

This module composes :mod:`time_domain_geometry` with the already-verified
fixed-``u`` patch extractor.  Geometry is supplied through the extractor's
forced path, which bypasses the production Welch detector.  The resulting
runtime path contains no FFT/STFT/Welch call while preserving the existing
packed-channel and 12-feature model interfaces.

It is deliberately a new experimental contract.  Existing hybrid-v3 caches and
shipped assets remain untouched.
"""
from __future__ import annotations

from typing import Any

import numpy as np

import invariant_patch_preprocess as patch_core
import preprocess as production_preprocess
import time_domain_geometry as geometry


PREPROCESS_VERSION = "invariant-patch-time-domain-v1"
ESTIMATOR_VERSION = geometry.ESTIMATOR_VERSION
DEFAULT_PATCH_LENGTH = patch_core.DEFAULT_PATCH_LENGTH
DEFAULT_PATCH_COUNT = patch_core.DEFAULT_PATCH_COUNT


def preprocess(
    iq: Any,
    patch_length: int = DEFAULT_PATCH_LENGTH,
    patch_count: int = DEFAULT_PATCH_COUNT,
    target_frac: float = production_preprocess.TARGET_FRAC,
) -> tuple[Any, Any, dict[str, Any]]:
    """Return packed patches, invariant features, and no-transform geometry.

    The output arrays have exactly the same shapes and dtypes as
    ``invariant_patch_preprocess.preprocess``.  The context replaces the
    hybrid-v3 detector fields with a complete time-domain estimator audit.
    """
    capture = np.asarray(iq)
    if capture.ndim != 1:
        # Preserve the estimator's more useful capture-shape error.
        centre, bandwidth, geometry_context = geometry.estimate_band(iq)
    else:
        minimum_bandwidth = geometry.minimum_bandwidth_for_active_span(
            len(capture),
            patch_length,
            target_frac,
        )
        centre, bandwidth, geometry_context = geometry.estimate_band(
            iq,
            min_bandwidth=minimum_bandwidth,
        )
    channels, features, context = patch_core.preprocess(
        iq,
        patch_length=patch_length,
        patch_count=patch_count,
        target_frac=target_frac,
        force_center=centre,
        force_bw=bandwidth,
        force_resample_frac=bandwidth,
    )
    context.update(
        {
            "version": PREPROCESS_VERSION,
            "estimator_version": ESTIMATOR_VERSION,
            "detected_center": centre,
            "detected_bw": bandwidth,
            "geometry_estimator": geometry_context,
            # The forced core path validates its legacy nfft argument for API
            # compatibility, but it neither reads nor transforms frequency data.
            "nfft": None,
            "uses_frequency_transform": False,
        }
    )
    return channels, features, context


def preprocess_metadata() -> dict[str, Any]:
    return {
        "version": PREPROCESS_VERSION,
        "patch_length": DEFAULT_PATCH_LENGTH,
        "patch_count": DEFAULT_PATCH_COUNT,
        "target_frac": production_preprocess.TARGET_FRAC,
        "geometry": geometry.estimator_metadata(),
        "uses_frequency_transform": False,
    }
