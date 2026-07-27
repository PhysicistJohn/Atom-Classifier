"""Export the frozen invariant-fusion candidate to a data-blind JSON staging bundle.

The input is a trusted local ``runtime_bundle.pt`` assembled by the
selection-only candidate pipeline.  This module never imports a corpus/data
loader and never fits or retrains anything.  It strictly reconstructs the
native complex model, strictly loads the paired-real inference twin, checks
their outputs, and writes browser-oriented weights plus a deterministic parity
fixture into a fresh staging directory.

The output is deliberately staging-only.  In particular, this exporter refuses
to write below ``src/`` and refuses to overwrite a non-empty directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
TRAINING = HERE.parents[1]
REPO = TRAINING.parent
for candidate in (TRAINING, HERE):
    value = str(candidate)
    if value not in os.sys.path:
        os.sys.path.insert(0, value)

from invariant_fusion import CenteredInvariantFusion  # noqa: E402
from invariant_paired_real_inference import make_paired_real_fusion  # noqa: E402
from invariant_patch_cnn import (  # noqa: E402
    InvariantPatchCNN,
    InvariantPatchConfig,
)
import invariant_patch_preprocess as invariant_frontend  # noqa: E402
import preprocess as canonical_preprocess  # noqa: E402


SCHEMA_ID = "atomos.invariant-fusion.paired-real"
SCHEMA_VERSION = 3
SOURCE_BUNDLE_SCHEMA = 2
WEIGHTS_NAME = "invariant-fusion-weights-v3.json"
FIXTURE_NAME = "invariant-fusion-parity-v3.json"
E2E_FIXTURE_NAME = "invariant-fusion-e2e-parity-v3.json"
MANIFEST_NAME = "manifest.json"
PARITY_LIMIT = 2e-6
E2E_PARITY_LIMIT = 1e-4

DEFAULT_BUNDLE = (
    HERE
    / "artifacts"
    / "invariant_patch"
    / "invariant_fusion_runtime_bundle_v2_seed20260727"
    / "runtime_bundle.pt"
)
DEFAULT_OUTPUT = (
    HERE
    / "artifacts"
    / "staging"
    / "invariant_fusion_paired_real_v3"
)

FEATURE_NAMES = [
    "abs_c20",
    "abs_c40",
    "c42_real",
    "abs_c41",
    "abs_c60",
    "c63",
    "m42",
    "amplitude_std",
    "max_amplitude_squared",
    "mean_amplitude",
    "instantaneous_frequency_std",
    "amplitude_coefficient_of_variation",
]

EXPECTED_REJECTION = {
    "real": {"weight": 0.40, "neighbors": 2},
    "complex": {"weight": 0.60, "neighbors": 64},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            _jsonable(payload),
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    os.replace(temporary, path)


def _is_below(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _validate_output_path(bundle_path: Path, output_dir: Path) -> None:
    """Reject live-code/assets, source-bundle descendants, and overwrites."""
    source_root = (REPO / "src").resolve()
    if _is_below(output_dir, source_root):
        raise ValueError(f"refusing to write staging output below live src/: {output_dir}")
    if _is_below(output_dir, bundle_path.parent):
        raise ValueError(
            "refusing to write staging output inside the source bundle directory"
        )
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"staging output is not a directory: {output_dir}")
        contents = list(output_dir.iterdir())
        if contents:
            preview = ", ".join(sorted(item.name for item in contents[:5]))
            raise FileExistsError(
                f"refusing to overwrite non-empty staging directory "
                f"{output_dir} (contains {preview})"
            )


def _finite_scalar(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite scalar")
    return result


def _tensor(
    name: str,
    value: Any,
    *,
    shape: tuple[int, ...] | None = None,
    positive: bool = False,
) -> torch.Tensor:
    try:
        result = torch.as_tensor(value).detach().cpu()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must be a finite tensor") from exc
    if shape is not None and tuple(result.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(result.shape)}")
    if not (result.is_floating_point() or result.is_complex()):
        raise ValueError(f"{name} must use a floating dtype")
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} contains NaN or infinity")
    if positive and not bool((result > 0).all()):
        raise ValueError(f"{name} must be strictly positive")
    return result


def _state_dict(name: str, value: Any) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty state dict")
    output: dict[str, torch.Tensor] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, torch.Tensor):
            raise ValueError(f"{name} must map string keys to tensors")
        tensor = item.detach().cpu()
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
            torch.isfinite(tensor).all()
        ):
            raise ValueError(f"{name}[{key!r}] is non-finite")
        output[key] = tensor
    return output


def _load_trusted_bundle(path: Path) -> dict[str, Any]:
    # This is intentionally a trusted-local-artifact boundary.  It is not an
    # untrusted upload parser, and nested state dicts require normal torch.load.
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict):
        raise ValueError("runtime bundle must be a mapping")
    return bundle


def _open_component(
    value: Any,
    *,
    embed_dim: int,
    expected_branch: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("each rejection component must be a mapping")
    branch = value.get("branch", value.get("branch_source"))
    if branch not in {"real", "complex"}:
        raise ValueError("rejection component branch must be real or complex")
    if expected_branch is not None and branch != expected_branch:
        raise ValueError(f"expected {expected_branch} rejection component")
    expected = EXPECTED_REJECTION[branch]
    neighbors = int(value.get("neighbors", -1))
    if neighbors != expected["neighbors"]:
        raise ValueError(
            f"{branch} rejection component must use k={expected['neighbors']}"
        )
    weight = _finite_scalar(f"{branch} rejection weight", value.get("weight"))
    if not math.isclose(weight, expected["weight"], abs_tol=1e-12, rel_tol=0.0):
        raise ValueError(
            f"{branch} rejection component must use weight {expected['weight']}"
        )

    reference = _tensor(
        f"{branch} embedding_reference", value.get("embedding_reference")
    ).to(torch.float64)
    if reference.ndim != 2 or reference.shape[1] != embed_dim:
        raise ValueError(
            f"{branch} embedding_reference must have shape [rows,{embed_dim}]"
        )
    rows = int(reference.shape[0])
    if neighbors <= 0 or neighbors >= rows:
        raise ValueError(f"{branch} rejection neighbors are invalid")
    mean = _tensor(
        f"{branch} embedding_mean",
        value.get("embedding_mean"),
        shape=(embed_dim,),
    ).to(torch.float64)
    scale = _tensor(
        f"{branch} embedding_scale",
        value.get("embedding_scale"),
        shape=(embed_dim,),
        positive=True,
    ).to(torch.float64)
    k_distance = _tensor(
        f"{branch} reference_k_distance",
        value.get("reference_k_distance"),
        shape=(rows,),
    ).to(torch.float64)
    local_density = _tensor(
        f"{branch} reference_local_density",
        value.get("reference_local_density"),
        shape=(rows,),
        positive=True,
    ).to(torch.float64)
    calibration_value = value.get(
        "sorted_calibration_raw", value.get("calibration")
    )
    calibration = _tensor(
        f"{branch} sorted_calibration_raw", calibration_value
    ).to(torch.float64)
    if calibration.ndim != 1 or not len(calibration):
        raise ValueError(f"{branch} calibration must be a non-empty vector")
    if not bool((calibration[1:] >= calibration[:-1]).all()):
        raise ValueError(f"{branch} calibration must be sorted")
    if not bool((k_distance >= 0).all()):
        raise ValueError(f"{branch} k-distance must be non-negative")
    if value.get("selection_or_novelty_used_in_fit") is not False:
        raise ValueError(
            f"{branch} LOF density/rank references must exclude selection "
            "and novelty rows"
        )
    return {
        "branch": branch,
        "weight": weight,
        "neighbors": neighbors,
        "reference": reference.numpy(),
        "mean": mean.numpy(),
        "scale": scale.numpy(),
        "reference_k_distance": k_distance.numpy(),
        "reference_local_density": local_density.numpy(),
        "calibration": calibration.numpy(),
    }


def _normalize_open_set(
    value: Any,
    *,
    embed_dim: int,
    source_unknown_threshold: float,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("open_set must be a mapping")
    if value.get("kind") != "known_only_lof_rank_ensemble":
        raise ValueError("v2 open_set kind must be known_only_lof_rank_ensemble")
    components_value = value.get("components")
    if not isinstance(components_value, (list, tuple)) or len(components_value) != 2:
        raise ValueError("v2 open_set must contain exactly two weighted components")
    components = [_open_component(item, embed_dim=embed_dim) for item in components_value]
    by_branch = {item["branch"]: item for item in components}
    if set(by_branch) != set(EXPECTED_REJECTION):
        raise ValueError("v2 open_set must contain real and complex components")
    components = [by_branch["real"], by_branch["complex"]]
    if not math.isclose(
        sum(item["weight"] for item in components),
        1.0,
        abs_tol=1e-12,
        rel_tol=0.0,
    ):
        raise ValueError("rejection component weights must sum to one")

    threshold = _finite_scalar("open_set threshold", value.get("threshold"))
    if not 0.0 <= threshold < 1.0:
        raise ValueError("open_set threshold must lie in [0,1)")
    if not math.isclose(
        threshold, source_unknown_threshold, abs_tol=1e-12, rel_tol=0.0
    ):
        raise ValueError(
            "bundle unknown_threshold must equal the weighted LOF-rank threshold"
        )
    leak_flag = value.get(
        "selection_or_novelty_used_in_density_or_threshold_fit",
        value.get("selection_or_novelty_used_in_fit"),
    )
    if leak_flag is not False:
        raise ValueError(
            "open_set density and threshold fitting must exclude selection "
            "and novelty rows"
        )
    if value.get("cannot_change_fused_closed_label") is not True:
        raise ValueError("LOF rejection must not change the fused closed label")
    return {
        "kind": "weighted_known_only_lof",
        "policy": "fused_label_weighted_branch_lof_rejector",
        "components": components,
        "threshold": threshold,
        "threshold_quantile": _finite_scalar(
            "open_set threshold_quantile", value.get("threshold_quantile")
        ),
    }


def _combined_state(
    real_state: Mapping[str, torch.Tensor],
    complex_state: Mapping[str, torch.Tensor],
    bundle: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    output = {
        **{f"real_branch.{key}": value for key, value in real_state.items()},
        **{
            f"complex_branch.{key}": value
            for key, value in complex_state.items()
        },
        "real_center": bundle["real_center"],
        "complex_center": bundle["complex_center"],
        "alpha_real": torch.tensor(bundle["alpha_real"], dtype=torch.float32),
        "alpha_complex": torch.tensor(
            bundle["alpha_complex"], dtype=torch.float32
        ),
        "weight_real": torch.tensor(bundle["weight_real"], dtype=torch.float32),
        "eps": torch.tensor(bundle["eps"], dtype=torch.float32),
    }
    return output


def _validate_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema",
        "kind",
        "real_config",
        "complex_config",
        "real_state_dict",
        "complex_state_dict",
        "real_center",
        "complex_center",
        "alpha_real",
        "alpha_complex",
        "weight_real",
        "eps",
        "feature_mean",
        "feature_std",
        "classes",
        "prototypes",
        "unknown_threshold",
        "temperature",
        "open_set",
        "development_only",
        "provenance",
    }
    missing = sorted(required.difference(bundle))
    if missing:
        raise ValueError(f"runtime bundle is missing keys: {missing}")
    if int(bundle["schema"]) != SOURCE_BUNDLE_SCHEMA:
        raise ValueError(
            f"exporter binds only frozen runtime bundle schema "
            f"{SOURCE_BUNDLE_SCHEMA}"
        )
    if bundle["kind"] != "invariant_centered_fusion_classifier":
        raise ValueError(f"unexpected bundle kind {bundle['kind']!r}")
    if bundle["development_only"] is not True:
        raise ValueError("this exporter currently requires a development-only bundle")
    provenance = bundle["provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError("provenance must be a mapping")
    if provenance.get("release_evidence") is not False:
        raise ValueError("development bundle cannot claim release evidence")
    if provenance.get("consumed_test_rows_used") != 0:
        raise ValueError("bundle provenance consumed held-out test rows")
    if provenance.get("center_fit_population") != "training only":
        raise ValueError("fusion centers must be fit on training only")
    if provenance.get("prototype_population") != "enrollment only":
        raise ValueError("fused prototypes must be recomputed from enrollment only")
    if provenance.get("unknown_calibration_population") != "enrollment only":
        raise ValueError("unknown threshold must be calibrated on enrollment only")
    cache_contract = provenance.get("cache_contract")
    if not isinstance(cache_contract, Mapping):
        raise ValueError("provenance must contain the invariant frontend contract")
    if (
        cache_contract.get("schema") != 1
        or cache_contract.get("frontend") != "invariant-patch-v1"
    ):
        raise ValueError("bundle frontend contract is not invariant-patch-v1")
    source_hashes = cache_contract.get("source_sha256")
    if not isinstance(source_hashes, Mapping):
        raise ValueError("frontend contract is missing source hashes")
    current_frontend_sources = {
        "training/invariant_patch_preprocess.py": (
            TRAINING / "invariant_patch_preprocess.py"
        ),
        "training/preprocess.py": TRAINING / "preprocess.py",
    }
    for name, path in current_frontend_sources.items():
        if source_hashes.get(name) != _sha256(path):
            raise ValueError(
                f"runtime bundle frontend source hash does not match {name}"
            )

    try:
        real_config = InvariantPatchConfig(**dict(bundle["real_config"])).validate()
        complex_config = InvariantPatchConfig(
            **dict(bundle["complex_config"])
        ).validate()
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid invariant branch config") from exc
    if real_config.encoder != "real" or complex_config.encoder != "complex":
        raise ValueError("bundle branches must be ordered real then complex")
    for field in ("patch_length", "patch_count", "n_features", "embed_dim"):
        if getattr(real_config, field) != getattr(complex_config, field):
            raise ValueError(f"branch config mismatch for {field}")

    embed_dim = int(real_config.embed_dim)
    real_state = _state_dict("real_state_dict", bundle["real_state_dict"])
    complex_state = _state_dict(
        "complex_state_dict", bundle["complex_state_dict"]
    )
    real_center = _tensor(
        "real_center", bundle["real_center"], shape=(embed_dim,)
    ).to(torch.float32)
    complex_center = _tensor(
        "complex_center", bundle["complex_center"], shape=(embed_dim,)
    ).to(torch.float32)
    alpha_real = _finite_scalar("alpha_real", bundle["alpha_real"])
    alpha_complex = _finite_scalar("alpha_complex", bundle["alpha_complex"])
    weight_real = _finite_scalar("weight_real", bundle["weight_real"])
    eps = _finite_scalar("eps", bundle["eps"])
    if not 0.0 <= weight_real <= 1.0:
        raise ValueError("weight_real must lie in [0,1]")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    feature_mean = _tensor(
        "feature_mean",
        bundle["feature_mean"],
        shape=(real_config.n_features,),
    ).to(torch.float32)
    feature_std = _tensor(
        "feature_std",
        bundle["feature_std"],
        shape=(real_config.n_features,),
        positive=True,
    ).to(torch.float32)
    classes = bundle["classes"]
    if (
        not isinstance(classes, (list, tuple))
        or not classes
        or any(not isinstance(name, str) or not name for name in classes)
        or len(set(classes)) != len(classes)
    ):
        raise ValueError("classes must be a non-empty unique string list")
    classes = list(classes)
    prototypes = _tensor(
        "prototypes",
        bundle["prototypes"],
        shape=(len(classes), 2 * embed_dim),
    ).to(torch.float32)
    temperature = _finite_scalar("temperature", bundle["temperature"])
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    source_unknown_threshold = _finite_scalar(
        "unknown_threshold", bundle["unknown_threshold"]
    )
    open_set = _normalize_open_set(
        bundle["open_set"],
        embed_dim=embed_dim,
        source_unknown_threshold=source_unknown_threshold,
    )

    real = InvariantPatchCNN(real_config)
    complex_branch = InvariantPatchCNN(complex_config)
    real.load_state_dict(real_state, strict=True)
    complex_branch.load_state_dict(complex_state, strict=True)
    reference = CenteredInvariantFusion(
        real,
        complex_branch,
        real_center,
        complex_center,
        alpha_real=alpha_real,
        alpha_complex=alpha_complex,
        weight_real=weight_real,
        eps=eps,
    ).eval()
    combined = _combined_state(
        real_state,
        complex_state,
        {
            "real_center": real_center,
            "complex_center": complex_center,
            "alpha_real": alpha_real,
            "alpha_complex": alpha_complex,
            "weight_real": weight_real,
            "eps": eps,
        },
    )
    reference.load_state_dict(combined, strict=True)
    paired = make_paired_real_fusion(
        real_config,
        complex_config,
        reference.state_dict(),
    )
    return {
        "real_config": real_config,
        "complex_config": complex_config,
        "real_state_dict": real_state,
        "complex_state_dict": complex_state,
        "reference": reference,
        "paired": paired,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "classes": classes,
        "prototypes": prototypes,
        # No prototype-distance rejection threshold exists.  The source field
        # is the weighted LOF rank threshold and is exported under rejection.
        "classification_unknown_threshold": None,
        "temperature": temperature,
        "open_set": open_set,
        "development_only": True,
        "provenance": dict(provenance),
    }


def _linear(weight: torch.Tensor, bias: torch.Tensor) -> dict[str, Any]:
    weight = weight.detach().cpu().to(torch.float32)
    bias = bias.detach().cpu().to(torch.float32)
    if weight.ndim != 2 or bias.shape != (weight.shape[0],):
        raise ValueError("malformed linear layer")
    return {
        "in": int(weight.shape[1]),
        "out": int(weight.shape[0]),
        "weight": weight.reshape(-1).tolist(),
        "bias": bias.tolist(),
    }


def _real_asset(validated: Mapping[str, Any]) -> dict[str, Any]:
    net = validated["reference"].real_branch
    blocks = []
    for block in net.patch_encoder.blocks:
        conv, bn = block.conv, block.bn
        scale = bn.weight.detach() / torch.sqrt(
            bn.running_var.detach() + float(bn.eps)
        )
        weight = conv.weight.detach() * scale[:, None, None]
        bias = bn.bias.detach() - bn.running_mean.detach() * scale
        blocks.append(
            {
                "in_channels": int(conv.in_channels),
                "out_channels": int(conv.out_channels),
                "kernel": int(conv.kernel_size[0]),
                "stride": int(conv.stride[0]),
                "padding": int(conv.padding[0]),
                "batch_norm_folded": True,
                "batch_norm_eps": float(bn.eps),
                "weight": weight.to(torch.float32).reshape(-1).tolist(),
                "bias": bias.to(torch.float32).tolist(),
            }
        )
    return {
        "config": net.config(),
        "blocks": blocks,
        "patch_projection": _linear(
            net.patch_projection[0].weight, net.patch_projection[0].bias
        ),
        "fc1": _linear(net.fc1.weight, net.fc1.bias),
        "fc2": _linear(net.fc2.weight, net.fc2.bias),
    }


def _complex_asset(validated: Mapping[str, Any]) -> dict[str, Any]:
    net = validated["reference"].complex_branch
    stages = []
    for stage in net.patch_encoder.stages:
        convolution = stage.conv
        stages.append(
            {
                "in_channels": int(convolution.conv_re.in_channels),
                "out_channels": int(convolution.conv_re.out_channels),
                "kernel": int(convolution.conv_re.kernel_size[0]),
                "padding": int(convolution.conv_re.padding[0]),
                "weight_real": (
                    convolution.conv_re.weight.detach()
                    .cpu()
                    .to(torch.float32)
                    .reshape(-1)
                    .tolist()
                ),
                "weight_imag": (
                    convolution.conv_im.weight.detach()
                    .cpu()
                    .to(torch.float32)
                    .reshape(-1)
                    .tolist()
                ),
                "threshold_raw": (
                    stage.act.thresh_raw.detach()
                    .cpu()
                    .to(torch.float32)
                    .tolist()
                ),
            }
        )
    return {
        "config": net.config(),
        "stages": stages,
        "mag_norm_eps": 1e-6,
        "modrelu_magnitude_eps": 1e-12,
        "lowpass": [0.25, 0.5, 0.25],
        "lags": [1, 2, 4],
        "patch_projection": _linear(
            net.patch_projection[0].weight, net.patch_projection[0].bias
        ),
        "fc1": _linear(net.fc1.weight, net.fc1.bias),
        "fc2": _linear(net.fc2.weight, net.fc2.bias),
    }


def _deterministic_inputs(
    packed_length: int,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sample = torch.arange(packed_length, dtype=torch.float32)
    phase0 = 2.0 * torch.pi * (
        0.071 * sample + 0.000021 * sample.square()
    )
    envelope0 = 0.72 + 0.21 * torch.cos(2.0 * torch.pi * 0.009 * sample)
    z0 = torch.polar(envelope0, phase0)

    phase1 = 2.0 * torch.pi * (
        -0.187 * sample
        + 0.035 * torch.sin(2.0 * torch.pi * sample / 79.0)
    )
    envelope1 = 0.35 + 0.65 * (
        (sample.remainder(137.0) < 91.0).to(torch.float32)
    )
    z1 = torch.polar(envelope1, phase1)

    z2 = (
        0.75 * torch.polar(torch.ones_like(sample), 2.0 * torch.pi * 0.11 * sample)
        + 0.35
        * torch.polar(
            torch.ones_like(sample),
            2.0 * torch.pi * (-0.233 * sample + 0.000013 * sample.square()),
        )
    )
    z = torch.stack((z0, z1, z2))
    rms = torch.sqrt((z.real.square() + z.imag.square()).mean(dim=1, keepdim=True))
    z = z / rms
    iq = torch.stack((z.real, z.imag), dim=1).to(torch.float32)

    width = len(feature_mean)
    desired = torch.stack(
        (
            torch.linspace(-0.75, 0.75, width),
            torch.sin(torch.arange(width, dtype=torch.float32) * 0.61),
            torch.zeros(width),
        )
    )
    raw = feature_mean.unsqueeze(0) + desired * feature_std.unsqueeze(0)
    standardized = (raw - feature_mean.unsqueeze(0)) / feature_std.unsqueeze(0)
    return iq, raw.to(torch.float32), standardized.to(torch.float32)


def _lof_raw_rank(
    embeddings: np.ndarray,
    component: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    embedding = np.asarray(embeddings, dtype=np.float64)
    reference = np.asarray(component["reference"], dtype=np.float64)
    standardized = (embedding - component["mean"]) / component["scale"]
    reference_norm = np.sum(reference * reference, axis=1)
    distance = (
        np.sum(standardized * standardized, axis=1)[:, None]
        + reference_norm[None, :]
        - 2.0 * np.einsum("ni,mi->nm", standardized, reference)
    )
    distance = np.maximum(distance, 0.0)
    neighbors = int(component["neighbors"])
    indices = np.argpartition(distance, neighbors - 1, axis=1)[:, :neighbors]
    selected = np.take_along_axis(distance, indices, axis=1)
    reachability = np.maximum(
        selected,
        np.asarray(component["reference_k_distance"])[indices],
    )
    density = 1.0 / (reachability.mean(axis=1) + 1e-12)
    raw = (
        np.asarray(component["reference_local_density"])[indices].mean(axis=1)
        / density
    )
    calibration = np.asarray(component["calibration"], dtype=np.float64)
    rank = (
        np.searchsorted(calibration, raw, side="left")
        / (len(calibration) + 1.0)
    )
    return raw, rank


@torch.no_grad()
def _fixture(validated: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    reference = validated["reference"].eval()
    paired = validated["paired"].eval()
    iq, raw_features, features = _deterministic_inputs(
        reference.packed_length,
        validated["feature_mean"],
        validated["feature_std"],
    )
    expected_real = reference.real_branch(iq, features)
    expected_complex = reference.complex_branch(iq, features)
    expected_fused = reference(iq, features)
    actual_real = paired.real_branch(iq, features)
    actual_complex = paired.complex_branch(iq, features)
    actual_fused = paired(iq, features)
    errors = {
        "real_embedding": float((expected_real - actual_real).abs().max()),
        "complex_embedding": float(
            (expected_complex - actual_complex).abs().max()
        ),
        "fused_embedding": float((expected_fused - actual_fused).abs().max()),
    }
    if max(errors.values()) > PARITY_LIMIT:
        raise AssertionError(f"native/paired-real parity failed: {errors}")

    prototypes = validated["prototypes"]
    distances = (
        expected_fused[:, None, :] - prototypes[None, :, :]
    ).square().sum(dim=-1)
    winner = distances.argmin(dim=1)
    logits = -distances / float(validated["temperature"])
    posterior = torch.softmax(logits, dim=1)

    component_fixture = []
    weighted_score = np.zeros(len(iq), dtype=np.float64)
    branch_embeddings = {
        "real": expected_real.detach().cpu().numpy(),
        "complex": expected_complex.detach().cpu().numpy(),
    }
    for component in validated["open_set"]["components"]:
        raw, rank = _lof_raw_rank(
            branch_embeddings[component["branch"]], component
        )
        weighted_score += float(component["weight"]) * rank
        component_fixture.append(
            {
                "branch": component["branch"],
                "weight": component["weight"],
                "neighbors": component["neighbors"],
                "raw": raw.tolist(),
                "rank": rank.tolist(),
            }
        )
    threshold = float(validated["open_set"]["threshold"])
    abstain = weighted_score > threshold
    closed_labels = [validated["classes"][int(index)] for index in winner]
    final_labels = [
        "unknown" if rejected else label
        for label, rejected in zip(closed_labels, abstain)
    ]
    fixture = {
        "schema": f"{SCHEMA_ID}.parity",
        "schema_version": SCHEMA_VERSION,
        "case_names": ["chirped-envelope", "bursty-fm", "two-tone"],
        "input": {
            "packed_iq": {
                "shape": list(iq.shape),
                "values": iq.reshape(-1).tolist(),
            },
            "raw_features": {
                "shape": list(raw_features.shape),
                "values": raw_features.reshape(-1).tolist(),
            },
        },
        "expected": {
            "standardized_features": features.tolist(),
            "real_embedding": expected_real.tolist(),
            "complex_embedding": expected_complex.tolist(),
            "fused_embedding": expected_fused.tolist(),
            "squared_prototype_distances": distances.tolist(),
            "closed_winner_index": winner.tolist(),
            "closed_label": closed_labels,
            "posterior": posterior.tolist(),
            "rejection_components": component_fixture,
            "weighted_rejection_score": weighted_score.tolist(),
            "rejection_threshold": threshold,
            "abstain": abstain.tolist(),
            "final_label": final_labels,
        },
        "native_vs_paired_real_max_abs_error": errors,
        "parity_limit": PARITY_LIMIT,
    }
    return fixture, errors


def _raw_capture_cases() -> list[dict[str, Any]]:
    """Deterministic, data-blind captures spanning physical scale and an FFT seam."""
    specifications = (
        ("physical-scale-0p125", 32768, 0.125, 0.13, None),
        ("physical-scale-0p25", 16384, 0.25, 0.13, None),
        ("physical-scale-0p5", 8192, 0.5, 0.13, None),
        ("physical-scale-1p0", 4096, 1.0, 0.13, None),
        ("physical-scale-2p0", 2048, 2.0, 0.13, None),
        ("hybrid-seam-crossing", 4096, 1.0, 0.48, None),
        (
            "nextafter-short-repeat",
            51,
            1.0,
            0.0,
            {
                "force_center": 0.0,
                "force_bw": 0.29,
                "force_resample_frac": 0.29,
            },
        ),
        (
            "non-endpoint-u-grid",
            777,
            1.0,
            0.0,
            {
                "force_center": 0.0,
                "force_bw": 0.19,
                "force_resample_frac": 0.19,
            },
        ),
    )
    cases: list[dict[str, Any]] = []
    for name, raw_length, scale, center, overrides in specifications:
        sample = np.arange(raw_length, dtype=np.float64)
        physical = sample * scale
        envelope = 0.72 + 0.18 * np.cos(2.0 * np.pi * 0.0031 * physical)
        primary_phase = 2.0 * np.pi * (
            center * sample
            + 0.055 * physical
            + 0.000004 * np.square(physical)
        )
        secondary_phase = 2.0 * np.pi * (
            center * sample
            - 0.095 * physical
            + 0.013 * np.sin(2.0 * np.pi * 0.001 * physical)
        )
        capture = (
            envelope * np.exp(1j * primary_phase)
            + 0.28 * np.exp(1j * secondary_phase)
        ).astype(np.complex128)
        cases.append(
            {
                "name": name,
                "raw_length": raw_length,
                "physical_scale": scale,
                "nominal_center": center,
                "preprocess_overrides": overrides or {},
                "capture": capture,
            }
        )
    return cases


@torch.no_grad()
def _end_to_end_fixture(validated: Mapping[str, Any]) -> dict[str, Any]:
    """Raw capture -> invariant frontend -> model -> weighted LOF fixture."""
    reference = validated["reference"].eval()
    patch_length = int(reference.real_branch.cfg.patch_length)
    patch_count = int(reference.real_branch.cfg.patch_count)
    contract = validated["provenance"]["cache_contract"]
    target_frac = float(contract["config"]["target_frac"])
    feature_mean = validated["feature_mean"].numpy()
    feature_std = validated["feature_std"].numpy()
    prototypes = validated["prototypes"]
    temperature = float(validated["temperature"])
    threshold = float(validated["open_set"]["threshold"])
    output_cases = []

    for specification in _raw_capture_cases():
        capture = specification["capture"]
        packed, raw_features, context = invariant_frontend.preprocess(
            capture,
            patch_length=patch_length,
            patch_count=patch_count,
            target_frac=target_frac,
            **specification["preprocess_overrides"],
        )
        # Match invariant_patch_data._standardize_features: float32 subtract
        # and divide with float32 stored moments.
        standardized = (
            (raw_features.astype(np.float32) - feature_mean)
            / feature_std
        ).astype(np.float32)
        iq = torch.from_numpy(packed).unsqueeze(0)
        features = torch.from_numpy(standardized).unsqueeze(0)
        real_embedding = reference.real_branch(iq, features)
        complex_embedding = reference.complex_branch(iq, features)
        fused_embedding = reference(iq, features)
        distances = (
            fused_embedding[:, None, :] - prototypes[None, :, :]
        ).square().sum(dim=-1)
        winner = int(distances.argmin(dim=1).item())
        posterior = torch.softmax(-distances / temperature, dim=1)

        component_rows = []
        rejection_score = 0.0
        branch_embeddings = {
            "real": real_embedding.detach().cpu().numpy(),
            "complex": complex_embedding.detach().cpu().numpy(),
        }
        for component in validated["open_set"]["components"]:
            raw, rank = _lof_raw_rank(
                branch_embeddings[component["branch"]], component
            )
            raw_value = float(raw[0])
            rank_value = float(rank[0])
            rejection_score += float(component["weight"]) * rank_value
            component_rows.append(
                {
                    "branch": component["branch"],
                    "weight": component["weight"],
                    "neighbors": component["neighbors"],
                    "raw": raw_value,
                    "rank": rank_value,
                }
            )
        closed_label = validated["classes"][winner]
        abstain = bool(rejection_score > threshold)
        output_cases.append(
            {
                "name": specification["name"],
                "physical_scale": specification["physical_scale"],
                "nominal_center": specification["nominal_center"],
                "preprocess_overrides": specification[
                    "preprocess_overrides"
                ],
                "raw": {
                    "length": specification["raw_length"],
                    "in_phase": capture.real.tolist(),
                    "quadrature": capture.imag.tolist(),
                },
                "expected": {
                    "context": context,
                    "packed_iq": {
                        "shape": list(packed.shape),
                        "values": packed.reshape(-1).tolist(),
                    },
                    "raw_features": raw_features.tolist(),
                    "standardized_features": standardized.tolist(),
                    "real_embedding": real_embedding[0].tolist(),
                    "complex_embedding": complex_embedding[0].tolist(),
                    "fused_embedding": fused_embedding[0].tolist(),
                    "squared_prototype_distances": distances[0].tolist(),
                    "closed_winner_index": winner,
                    "closed_label": closed_label,
                    "posterior": posterior[0].tolist(),
                    "rejection_components": component_rows,
                    "weighted_rejection_score": rejection_score,
                    "rejection_threshold": threshold,
                    "abstain": abstain,
                    "final_label": "unknown" if abstain else closed_label,
                },
            }
        )
    return {
        "schema": f"{SCHEMA_ID}.end-to-end-parity",
        "schema_version": SCHEMA_VERSION,
        "frontend": {
            "version": invariant_frontend.PREPROCESS_VERSION,
            "estimator_version": invariant_frontend.ESTIMATOR_VERSION,
            "contract": contract,
            "source_sha256": {
                "training/invariant_patch_preprocess.py": _sha256(
                    TRAINING / "invariant_patch_preprocess.py"
                ),
                "training/preprocess.py": _sha256(TRAINING / "preprocess.py"),
            },
        },
        "cases": output_cases,
        "parity_limit": E2E_PARITY_LIMIT,
        "data_or_corpus_loaded": False,
        "held_out_rows_loaded": 0,
    }


def _weights_payload(
    validated: Mapping[str, Any],
    *,
    source_sha256: str,
) -> dict[str, Any]:
    reference = validated["reference"]
    open_set = validated["open_set"]
    rejection_components = []
    for component in open_set["components"]:
        rejection_components.append(
            {
                "branch": component["branch"],
                "weight": component["weight"],
                "neighbors": component["neighbors"],
                "reference": component["reference"].tolist(),
                "mean": component["mean"].tolist(),
                "scale": component["scale"].tolist(),
                "reference_k_distance": component[
                    "reference_k_distance"
                ].tolist(),
                "reference_local_density": component[
                    "reference_local_density"
                ].tolist(),
                "calibration": component["calibration"].tolist(),
            }
        )
    source_contract = validated["provenance"]["cache_contract"]
    preprocess_contract = {
        **source_contract,
        "estimator_version": invariant_frontend.ESTIMATOR_VERSION,
        "nfft": int(canonical_preprocess.NFFT),
        "feature_count": int(canonical_preprocess.N_FEATURES),
        "feature_names": FEATURE_NAMES,
        "packed_dtype": "float32",
        "raw_feature_dtype": "float32",
        "interpolation": (
            "linear interpolation from captured u coordinates onto every "
            "integer u through floor(nextafter(captured_u_end,+inf))"
        ),
        "short_active": "cyclic repeat before fixed-length patch extraction",
    }
    return {
        "schema": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "status": "staging_not_release",
        "provisional": True,
        "packed_length": reference.packed_length,
        "real": _real_asset(validated),
        "complex": _complex_asset(validated),
        "fusion": {
            "real_center": reference.real_center.tolist(),
            "complex_center": reference.complex_center.tolist(),
            "alpha_real": float(reference.alpha_real),
            "alpha_complex": float(reference.alpha_complex),
            "weight_real": float(reference.weight_real),
            "eps": float(reference.eps),
        },
        "feature_standardization": {
            "mean": validated["feature_mean"].tolist(),
            "std": validated["feature_std"].tolist(),
        },
        "classification": {
            "classes": validated["classes"],
            "prototypes": validated["prototypes"].tolist(),
            "unknown_threshold": validated[
                "classification_unknown_threshold"
            ],
            "temperature": validated["temperature"],
            "posterior": "softmax(-squared_euclidean / temperature)",
        },
        "rejection": {
            "kind": open_set["kind"],
            "policy": open_set["policy"],
            "components": rejection_components,
            "threshold": open_set["threshold"],
            "threshold_quantile": open_set["threshold_quantile"],
            "score": "sum(component.weight * component.enrollment_rank)",
            "comparison": "unknown iff score > threshold",
            "closed_label_changed_before_abstention": False,
        },
        "preprocess": preprocess_contract,
        "provenance": {
            **validated["provenance"],
            "source_runtime_bundle_sha256": source_sha256,
        },
        "release_blockers": [
            "provisional candidate; frozen v2 source hash must be rechecked",
            "newly sealed synthetic length/scale release population is absent",
            "real SDR validation is absent",
            "browser-worker latency and memory are not measured",
        ],
    }


def export_staging(bundle_path: Path, output_dir: Path) -> dict[str, Any]:
    bundle_path = bundle_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not bundle_path.is_file():
        raise FileNotFoundError(bundle_path)
    _validate_output_path(bundle_path, output_dir)
    validated = _validate_bundle(_load_trusted_bundle(bundle_path))
    source_sha = _sha256(bundle_path)
    fixture, parity_errors = _fixture(validated)
    end_to_end_fixture = _end_to_end_fixture(validated)
    weights = _weights_payload(validated, source_sha256=source_sha)

    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / WEIGHTS_NAME
    fixture_path = output_dir / FIXTURE_NAME
    end_to_end_fixture_path = output_dir / E2E_FIXTURE_NAME
    _write_json(weights_path, weights)
    _write_json(fixture_path, fixture)
    _write_json(end_to_end_fixture_path, end_to_end_fixture)
    manifest = {
        "schema": f"{SCHEMA_ID}.manifest",
        "schema_version": SCHEMA_VERSION,
        "status": "staging_not_release",
        "provisional": True,
        "source": {
            "runtime_bundle": str(bundle_path),
            "runtime_bundle_sha256": source_sha,
        },
        "weights": {
            "file": WEIGHTS_NAME,
            "bytes": weights_path.stat().st_size,
            "sha256": _sha256(weights_path),
        },
        "parity_fixture": {
            "file": FIXTURE_NAME,
            "bytes": fixture_path.stat().st_size,
            "sha256": _sha256(fixture_path),
        },
        "end_to_end_parity_fixture": {
            "file": E2E_FIXTURE_NAME,
            "bytes": end_to_end_fixture_path.stat().st_size,
            "sha256": _sha256(end_to_end_fixture_path),
            "case_count": len(end_to_end_fixture["cases"]),
            "raw_lengths": [
                case["raw"]["length"] for case in end_to_end_fixture["cases"]
            ],
            "physical_scales": [
                case["physical_scale"]
                for case in end_to_end_fixture["cases"]
            ],
        },
        "validation": {
            "native_vs_paired_real_max_abs_error": parity_errors,
            "limit": PARITY_LIMIT,
            "strict_native_state_load": True,
            "strict_paired_real_state_load": True,
            "data_or_corpus_loaded": False,
            "held_out_rows_loaded": 0,
            "end_to_end_fixture_generated": True,
            "frontend_source_hashes_match_training_contract": True,
        },
        "release_blockers": weights["release_blockers"],
    }
    _write_json(output_dir / MANIFEST_NAME, manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = export_staging(args.bundle, args.output_dir)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "provisional": manifest["provisional"],
                "output_dir": str(args.output_dir.expanduser().resolve()),
                "validation": manifest["validation"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
