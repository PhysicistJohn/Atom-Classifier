"""Shared, fail-closed helpers for the strict time-domain v3 candidate.

This module deliberately contains no release evaluator and performs no model
selection.  It binds one real and one complex development checkpoint to the
FFT-free time-domain frontend, rebuilds only the exposed train/enrollment/
selection tensors, and derives the fixed centered-fusion assets used by both
candidate assembly and independent audits.

The historical development corpus enforced occupancy at N=16384.  The release
protocol instead probes occupancy at N=4096 before generating a longest capture.
``release_compatible_occupancy`` exposes that predeclared population mismatch
without changing a model, prototype, threshold, or gate.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
TRAINING = HERE.parent.parent
REPO = TRAINING.parent

import invariant_patch_data  # noqa: E402
from assemble_invariant_candidate import _embed_all, _fuse_numpy  # noqa: E402
from invariant_fusion import CenteredInvariantFusion  # noqa: E402
from invariant_patch_cnn import (  # noqa: E402
    InvariantPatchCNN,
    InvariantPatchConfig,
)
from train import prototypes_from  # noqa: E402
from v3_scale import run_time_domain_dev as time_domain_dev  # noqa: E402


STRICT_FRONTEND_VERSION = "invariant-patch-time-domain-v1"
STRICT_GEOMETRY_VERSION = "time-correlation-v1"
SHORTEST_RELEASE_LENGTH = 4096
OCCUPANCY_MIN_POWER = 1e-9
ALPHA_REAL = 0.2
ALPHA_COMPLEX = 0.2
WEIGHT_REAL = 0.6
EPSILON = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_state(path: Path) -> Mapping[str, torch.Tensor]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older Torch
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"checkpoint is not a non-empty state mapping: {path}")
    for name, tensor in value.items():
        if (
            not isinstance(name, str)
            or not isinstance(tensor, torch.Tensor)
            or (
                (tensor.is_floating_point() or tensor.is_complex())
                and not bool(torch.isfinite(tensor).all())
            )
        ):
            raise ValueError(f"checkpoint entry {name!r} is invalid: {path}")
    return value


@dataclass(frozen=True)
class BranchArtifact:
    """One validated strict-v3 branch and its immutable source records."""

    directory: Path
    encoder: str
    report: dict[str, Any]
    config: InvariantPatchConfig
    state_path: Path
    state_sha256: str
    report_sha256: str


@dataclass
class StrictFusionContext:
    """Loaded branches plus train-only/enrollment-only fusion assets."""

    real: InvariantPatchCNN
    complex: InvariantPatchCNN
    fusion: CenteredInvariantFusion
    embeddings: dict[str, dict[str, np.ndarray]]
    centers: dict[str, np.ndarray]
    prototypes: np.ndarray


def _architecture_config(
    report: Mapping[str, Any],
    expected_encoder: str,
) -> InvariantPatchConfig:
    architecture = report.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError("branch report has no architecture mapping")
    fields = (
        "patch_length",
        "patch_count",
        "encoder",
        "patch_dim",
        "hidden",
        "set_pool",
        "dropout",
    )
    try:
        config = InvariantPatchConfig(
            **{name: architecture[name] for name in fields}
        ).validate()
    except KeyError as exc:
        raise ValueError(
            f"branch architecture omits {exc.args[0]!r}"
        ) from exc
    if config.encoder != expected_encoder:
        raise ValueError(
            f"expected encoder {expected_encoder!r}, got {config.encoder!r}"
        )
    return config


def load_branch_artifact(
    directory: Path,
    expected_encoder: str,
) -> BranchArtifact:
    """Validate a development branch before any model tensor is loaded."""
    root = Path(directory).expanduser().resolve()
    if expected_encoder not in {"real", "complex"}:
        raise ValueError("expected_encoder must be 'real' or 'complex'")
    if not root.is_dir():
        raise FileNotFoundError(root)
    if "releases" in root.parts or any(
        "sealed" in part.lower() for part in root.parts
    ):
        raise ValueError("strict-v3 branch must be development-only")
    report_path = root / "dev_metrics.json"
    state_path = root / f"{expected_encoder}_state_dict.pt"
    with report_path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    if (
        report.get("development_only") is not True
        or report.get("release_evidence") is not False
        or report.get("consumed_test_rows_used") != 0
        or report.get("sealed_release_data_used") != 0
        or report.get("data_audit", {}).get(
            "consumed_test_rows_exposed"
        )
        != 0
    ):
        raise ValueError("branch provenance is not development-only/leak-safe")
    frontend = report.get("frontend")
    if (
        not isinstance(frontend, Mapping)
        or frontend.get("version") != STRICT_FRONTEND_VERSION
        or frontend.get("uses_frequency_transform") is not False
        or frontend.get("geometry", {}).get("version")
        != STRICT_GEOMETRY_VERSION
        or frontend.get("geometry", {}).get(
            "uses_frequency_transform"
        )
        is not False
    ):
        raise ValueError("branch did not use the strict time-domain frontend")
    config = _architecture_config(report, expected_encoder)
    state_hash = sha256(state_path)
    artifact_record = report.get("artifacts", {})
    if artifact_record.get("state_dict_sha256") != state_hash:
        raise ValueError("branch state hash disagrees with its report")
    if artifact_record.get("state_dict") != state_path.name:
        raise ValueError("branch state filename disagrees with its report")
    return BranchArtifact(
        directory=root,
        encoder=expected_encoder,
        report=dict(report),
        config=config,
        state_path=state_path,
        state_sha256=state_hash,
        report_sha256=sha256(report_path),
    )


def prepare_strict_data(
    *,
    model_seed: int,
    patch_length: int = 64,
    patch_count: int = 16,
    target_frac: float = 0.5,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rebuild strict tensors from exposed indices, never a held-out split."""
    source = invariant_patch_data.load(
        patch_length=patch_length,
        patch_count=patch_count,
        target_frac=target_frac,
        model_seed=model_seed,
        build_if_missing=False,
    )
    if source["data_audit"].get("consumed_test_rows_exposed") != 0:
        raise RuntimeError("data loader exposed consumed test rows")
    forbidden = sorted(name for name in source if "test" in name.lower())
    if forbidden:
        raise RuntimeError(f"data loader exposed test-like keys: {forbidden}")
    manifest = time_domain_dev._load_manifest()
    data, contexts, training_views = time_domain_dev._prepare_data(
        source,
        manifest,
        patch_length=patch_length,
        patch_count=patch_count,
        target_frac=target_frac,
    )
    if training_views.get("enabled") is not False:
        raise AssertionError("shipping branch unexpectedly used multi-length views")
    data["strict_contexts"] = contexts
    return data, manifest


def _load_model(
    artifact: BranchArtifact,
    device: torch.device,
) -> InvariantPatchCNN:
    model = InvariantPatchCNN(artifact.config)
    model.load_state_dict(_load_state(artifact.state_path), strict=True)
    return model.to(device).eval()


def fit_fusion_context(
    real_artifact: BranchArtifact,
    complex_artifact: BranchArtifact,
    data: Mapping[str, Any],
    device: torch.device,
) -> StrictFusionContext:
    """Fit only train centers and enrollment prototypes for fixed fusion."""
    geometry_fields = ("patch_length", "patch_count", "n_features", "embed_dim")
    for field in geometry_fields:
        if getattr(real_artifact.config, field) != getattr(
            complex_artifact.config, field
        ):
            raise ValueError(f"branch {field} differs")
    real = _load_model(real_artifact, device)
    complex_model = _load_model(complex_artifact, device)
    branches = {"real": real, "complex": complex_model}
    embeddings: dict[str, dict[str, np.ndarray]] = {
        name: {
            split: _embed_all(
                model,
                np.asarray(data[f"x{split}"]),
                np.asarray(data[f"f{split}"]),
                device,
            )
            for split in ("tr", "en", "va")
        }
        for name, model in branches.items()
    }
    centers = {
        name: values["tr"].astype(np.float64).mean(axis=0).astype(np.float32)
        for name, values in embeddings.items()
    }
    fusion = CenteredInvariantFusion(
        real,
        complex_model,
        centers["real"],
        centers["complex"],
        alpha_real=ALPHA_REAL,
        alpha_complex=ALPHA_COMPLEX,
        weight_real=WEIGHT_REAL,
        eps=EPSILON,
    ).to(device).eval()
    embeddings["fusion"] = {
        split: _fuse_numpy(
            embeddings["real"][split],
            embeddings["complex"][split],
            centers["real"],
            centers["complex"],
        )
        for split in ("tr", "en", "va")
    }
    direct = _embed_all(
        fusion,
        np.asarray(data["xva"])[:64],
        np.asarray(data["fva"])[:64],
        device,
    )
    error = float(
        np.max(np.abs(direct - embeddings["fusion"]["va"][: len(direct)]))
    )
    if error > 2e-6:
        raise RuntimeError(f"NumPy/module fusion mismatch: {error}")
    prototypes = prototypes_from(
        embeddings["fusion"]["en"],
        np.asarray(data["yen"], dtype=np.int64),
        int(data["n_classes"]),
    )
    return StrictFusionContext(
        real=real,
        complex=complex_model,
        fusion=fusion,
        embeddings=embeddings,
        centers=centers,
        prototypes=prototypes,
    )


def prefix_mean_power(
    clean: np.ndarray,
    indices: np.ndarray,
    *,
    length: int = SHORTEST_RELEASE_LENGTH,
    block_rows: int = 128,
) -> np.ndarray:
    """Compute clean prefix power deterministically without a giant copy."""
    rows = np.asarray(indices, dtype=np.int64)
    if (
        rows.ndim != 1
        or np.any(rows < 0)
        or length <= 0
        or block_rows <= 0
        or np.asarray(clean).ndim != 3
        or np.asarray(clean).shape[2] != 2
        or length > np.asarray(clean).shape[1]
        or (len(rows) and int(rows.max()) >= np.asarray(clean).shape[0])
    ):
        raise ValueError("clean prefix power inputs are invalid")
    output = np.empty(len(rows), dtype=np.float64)
    for start in range(0, len(rows), block_rows):
        chosen = rows[start : start + block_rows]
        value = np.asarray(clean[chosen, :length], dtype=np.float64)
        output[start : start + len(chosen)] = np.mean(
            np.sum(value * value, axis=2),
            axis=1,
        )
    if not np.isfinite(output).all() or np.any(output < 0.0):
        raise RuntimeError("clean prefix power is non-finite or negative")
    return output


def release_compatible_occupancy(
    manifest: Mapping[str, Any],
    indices: np.ndarray,
    classes: list[str],
    *,
    corpus_dir: Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select the development rows satisfying the release N=4096 start rule."""
    corpus = (
        TRAINING / "artifacts" / "signallab-corpus"
        if corpus_dir is None
        else Path(corpus_dir).expanduser().resolve()
    )
    rows = np.asarray(indices, dtype=np.int64)
    count = int(manifest["count"])
    sample_count = int(manifest["sampleCount"])
    clean_path = corpus / "corpus_clean.f32"
    expected_bytes = count * sample_count * 2 * np.dtype("<f4").itemsize
    if clean_path.stat().st_size != expected_bytes:
        raise ValueError("clean corpus size disagrees with manifest")
    clean = np.memmap(
        clean_path,
        dtype="<f4",
        mode="r",
        shape=(count, sample_count, 2),
    )
    power = prefix_mean_power(clean, rows)
    mask = power > OCCUPANCY_MIN_POWER
    selected = rows[mask]
    class_set = set(classes)
    included = {name: 0 for name in classes}
    excluded = {name: 0 for name in classes}
    excluded_impairment = {"clean": 0, "impaired": 0}
    for corpus_index, keep in zip(rows, mask):
        item = manifest["items"][int(corpus_index)]
        class_name = str(item["cls"])
        if class_name not in class_set:
            raise ValueError(f"manifest class {class_name!r} is unknown")
        if keep:
            included[class_name] += 1
        else:
            excluded[class_name] += 1
            excluded_impairment[
                "impaired" if bool(item["impaired"]) else "clean"
            ] += 1
    return selected, {
        "definition": (
            "historical development rows whose stored clean N=4096 prefix "
            "has mean complex power > 1e-9, matching the predeclared release "
            "start-probe occupancy rule"
        ),
        "selection_uses_labels": False,
        "selection_uses_model_outputs": False,
        "shortest_length": SHORTEST_RELEASE_LENGTH,
        "occupancy_min_power": OCCUPANCY_MIN_POWER,
        "eligible_rows": int(mask.sum()),
        "excluded_rows": int((~mask).sum()),
        "eligible_by_class": included,
        "excluded_by_class": excluded,
        "excluded_by_impairment": excluded_impairment,
        "eligible_indices_sha256": hashlib.sha256(
            selected.astype("<i8", copy=False).tobytes()
        ).hexdigest(),
        "maximum_excluded_clean_prefix_power": (
            float(power[~mask].max()) if (~mask).any() else None
        ),
        "minimum_included_clean_prefix_power": (
            float(power[mask].min()) if mask.any() else None
        ),
        "clean_corpus_sha256": sha256(clean_path),
    }


__all__ = [
    "ALPHA_COMPLEX",
    "ALPHA_REAL",
    "BranchArtifact",
    "EPSILON",
    "OCCUPANCY_MIN_POWER",
    "SHORTEST_RELEASE_LENGTH",
    "STRICT_FRONTEND_VERSION",
    "STRICT_GEOMETRY_VERSION",
    "StrictFusionContext",
    "WEIGHT_REAL",
    "fit_fusion_context",
    "load_branch_artifact",
    "prefix_mean_power",
    "prepare_strict_data",
    "release_compatible_occupancy",
    "sha256",
]
