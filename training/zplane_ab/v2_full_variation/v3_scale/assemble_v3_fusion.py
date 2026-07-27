"""Assemble the v3 FFT-free real/complex centered fusion from two trained branches.

This is the v3 replacement for ``assemble_invariant_candidate.py``.  It reuses
the v2 fusion maths (``invariant_fusion.CenteredInvariantFusion``) and the v2
fitting contract verbatim, and replaces only the frontend binding: every tensor
scored here is rebuilt with ``time_domain_invariant_patch_preprocess``, so the
occupied-band pose contains no FFT, Welch transform, STFT, or frequency grid.

Fitting contract, copied from the v2 assembly and not relaxed:

- Branch centers are estimated from **training rows only**.
- Fused prototypes are estimated from **enrollment rows only**.
- The immutable development-selection population is scored, never fit.
- The consumed historical test half and every sealed release suite are neither
  loaded nor referenced.  Release and sealed paths are refused before any data
  is touched.

The reported populations are exactly the ones ``run_time_domain_dev.py`` scores,
computed by calling that module's own functions, so a fusion ``dev_metrics.json``
is directly comparable to a single-branch one.

This produces development evidence for a v3 candidate.  It is not release
evidence: a new untouched release seed and its sealed suite remain required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
V2 = HERE.parent
ZPAB = V2.parent
TRAINING = ZPAB.parent
for _path in (TRAINING, ZPAB, V2, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import invariant_patch_data as invariant_data  # noqa: E402
import run_invariant_cnn_dev as runner  # noqa: E402
import run_time_domain_dev as dev  # noqa: E402
import time_domain_invariant_patch_preprocess as td_preprocess  # noqa: E402
from invariant_fusion import CenteredInvariantFusion  # noqa: E402
from invariant_patch_cnn import (  # noqa: E402
    InvariantPatchCNN,
    InvariantPatchConfig,
)
from train import embed_all, prototypes_from  # noqa: E402


# Frozen v2 fusion constants.  See assemble_invariant_candidate.ALPHA_REAL /
# ALPHA_COMPLEX; partial centering, not full mean removal.  They are duplicated
# rather than imported because importing the v2 assembler drags in the entire
# hybrid-v3 open-set stack that v3 replaces.
ALPHA_REAL = 0.2
ALPHA_COMPLEX = 0.2
DEFAULT_WEIGHT_REAL = 0.5
FUSION_EPS = 1e-12

# Maximum tolerated disagreement between the NumPy fusion used for metrics and
# the Torch module that will be serialized.  Same value as the v2 assembler.
FUSION_ASSEMBLY_TOLERANCE = 2e-6
# Maximum tolerated disagreement between each branch's stored prototypes and the
# prototypes recomputed here from enrollment rows.  Loose enough to absorb
# CPU/MPS reduction-order differences, tight enough to catch a wrong checkpoint,
# a wrong split, or a changed frontend.
DEFAULT_PROTOTYPE_TOLERANCE = 1e-4

CENTER_SPLIT = "tr"
PROTOTYPE_SPLIT = "en"
METRIC_SPLIT = "va"

DEFAULT_OUTPUT_ROOT = (
    V2 / "artifacts" / "invariant_patch" / "v3_scale"
)

# The exact top-level schema emitted by run_time_domain_dev.run.  A fusion
# artifact must be readable by anything that reads a branch artifact.
REQUIRED_DEV_METRICS_KEYS = frozenset(
    {
        "architecture",
        "artifacts",
        "closed_selection",
        "consumed_test_rows_used",
        "data_audit",
        "development_only",
        "device",
        "encoder",
        "frontend",
        "geometry_summary",
        "parameter_count",
        "population_causal_length",
        "population_physical_scale",
        "release_evidence",
        "sealed_release_data_used",
        "source_index_contract",
        "source_sha256",
        "status",
        "training",
        "training_views",
        "wall_clock_s",
    }
)


# ---------------------------------------------------------------------------
# path and argument discipline
# ---------------------------------------------------------------------------


def reject_sealed_path(path: Path, role: str) -> Path:
    """Refuse any release or sealed location, for reading or for writing."""
    resolved = Path(path).expanduser().resolve()
    lowered = {part.lower() for part in resolved.parts}
    if "releases" in lowered or any("sealed" in part for part in lowered):
        raise ValueError(
            f"development assembly refuses release or sealed {role} paths: "
            f"{resolved}"
        )
    return resolved


def validate_branch_weight(value: Any) -> float:
    """Validate the real-branch weight; the complex weight is ``1 - value``."""
    try:
        weight = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("branch weight must be a finite scalar") from exc
    if not np.isfinite(weight):
        raise ValueError("branch weight must be a finite scalar")
    if not 0.0 <= weight <= 1.0:
        raise ValueError("branch weight must lie in [0, 1]")
    return weight


# ---------------------------------------------------------------------------
# fusion maths, mirroring invariant_fusion.CenteredInvariantFusion
# ---------------------------------------------------------------------------


def _safe_unit(value: np.ndarray, eps: float = FUSION_EPS) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    norm = np.sqrt(np.sum(value * value, axis=1, keepdims=True))
    output = value / np.maximum(norm, np.float32(eps))
    cancelled = norm[:, 0] <= eps
    if cancelled.any():
        output[cancelled] = 0.0
        output[cancelled, 0] = 1.0
    return output.astype(np.float32, copy=False)


def fuse_numpy(
    real: np.ndarray,
    complex_embedding: np.ndarray,
    real_center: np.ndarray,
    complex_center: np.ndarray,
    *,
    weight_real: float,
    alpha_real: float = ALPHA_REAL,
    alpha_complex: float = ALPHA_COMPLEX,
) -> np.ndarray:
    """Centered weighted concatenation, identical to the Torch fusion module."""
    weight = validate_branch_weight(weight_real)
    centered_real = _safe_unit(
        np.asarray(real, dtype=np.float32)
        - np.float32(alpha_real) * np.asarray(real_center, np.float32)
    )
    centered_complex = _safe_unit(
        np.asarray(complex_embedding, dtype=np.float32)
        - np.float32(alpha_complex) * np.asarray(complex_center, np.float32)
    )
    fused = np.concatenate(
        (
            np.sqrt(np.float32(weight)) * centered_real,
            np.sqrt(np.float32(1.0 - weight)) * centered_complex,
        ),
        axis=1,
    )
    return _safe_unit(fused)


class FusionEmbedder(torch.nn.Module):
    """Present the fused embedding under the single-branch inference interface.

    ``run_time_domain_dev`` audits accept any module with a ``cfg`` carrying the
    patch geometry.  Delegating to the real branch's config lets the audits run
    unmodified, so fusion numbers come from the same code path as branch numbers.
    """

    def __init__(self, fusion: CenteredInvariantFusion):
        super().__init__()
        if not isinstance(fusion, CenteredInvariantFusion):
            raise TypeError("fusion must be a CenteredInvariantFusion")
        self.fusion = fusion

    @property
    def cfg(self) -> InvariantPatchConfig:
        return self.fusion.real_branch.cfg

    def forward(self, x: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        return self.fusion(x, feat)


# ---------------------------------------------------------------------------
# the fitting contract
# ---------------------------------------------------------------------------


def fit_center(branch_embeddings: Mapping[str, np.ndarray]) -> np.ndarray:
    """Estimate one branch center from TRAINING rows only.

    Only ``branch_embeddings[CENTER_SPLIT]`` is read.  Enrollment, selection and
    anything else present in the mapping cannot influence the result.
    """
    if CENTER_SPLIT not in branch_embeddings:
        raise KeyError(
            f"center fitting requires the {CENTER_SPLIT!r} (training) split"
        )
    train = np.asarray(branch_embeddings[CENTER_SPLIT], dtype=np.float64)
    if train.ndim != 2 or train.shape[0] == 0:
        raise ValueError("training embeddings must be a non-empty [N, D] array")
    if not np.isfinite(train).all():
        raise ValueError("training embeddings must be finite")
    return train.mean(axis=0).astype(np.float32)


def fit_prototypes(
    fused_embeddings: Mapping[str, np.ndarray],
    enroll_labels: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Estimate fused prototypes from ENROLLMENT rows only."""
    if PROTOTYPE_SPLIT not in fused_embeddings:
        raise KeyError(
            f"prototype fitting requires the {PROTOTYPE_SPLIT!r} "
            "(enrollment) split"
        )
    enrollment = np.asarray(
        fused_embeddings[PROTOTYPE_SPLIT], dtype=np.float32
    )
    labels = np.asarray(enroll_labels, dtype=np.int64)
    if enrollment.ndim != 2 or len(enrollment) != len(labels):
        raise ValueError("enrollment embeddings and labels disagree in length")
    if not np.isfinite(enrollment).all():
        raise ValueError("enrollment embeddings must be finite")
    counts = np.bincount(labels, minlength=int(n_classes))
    if int(n_classes) <= 0 or np.any(counts[: int(n_classes)] == 0):
        raise ValueError("every class needs at least one enrollment row")
    return prototypes_from(enrollment, labels, int(n_classes))


def validate_result_schema(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Fail closed unless the artifact matches the branch runner's schema."""
    missing = sorted(REQUIRED_DEV_METRICS_KEYS - set(payload))
    if missing:
        raise ValueError(f"dev_metrics.json is missing required keys: {missing}")
    for key, expected in (
        ("development_only", True),
        ("release_evidence", False),
        ("sealed_release_data_used", 0),
        ("consumed_test_rows_used", 0),
    ):
        if payload[key] != expected:
            raise ValueError(
                f"dev_metrics.json {key} must be {expected!r}, "
                f"got {payload[key]!r}"
            )
    return payload


# ---------------------------------------------------------------------------
# branch loading
# ---------------------------------------------------------------------------


def load_branch(
    directory: Path,
    encoder: str,
) -> dict[str, Any]:
    """Load one trained branch from its directory: state_dict plus prototypes."""
    if encoder not in {"real", "complex"}:
        raise ValueError("encoder must be 'real' or 'complex'")
    source = reject_sealed_path(Path(directory), "source")
    if not source.is_dir():
        raise FileNotFoundError(source)

    metrics_path = source / "dev_metrics.json"
    with metrics_path.open(encoding="utf-8") as handle:
        metrics = json.load(handle)

    if str(metrics.get("encoder")) != encoder:
        raise ValueError(
            f"{source} holds encoder {metrics.get('encoder')!r}, "
            f"expected {encoder!r}"
        )
    if metrics.get("development_only") is not True:
        raise RuntimeError(f"{source} is not marked development_only")
    if (
        metrics.get("sealed_release_data_used") != 0
        or metrics.get("consumed_test_rows_used") != 0
        or metrics.get("data_audit", {}).get("consumed_test_rows_exposed") != 0
    ):
        raise RuntimeError(f"{source} provenance exposed sealed or consumed rows")

    frontend = metrics.get("frontend", {})
    if frontend.get("version") != td_preprocess.PREPROCESS_VERSION:
        raise RuntimeError(
            f"{source} was trained on frontend {frontend.get('version')!r}, "
            f"expected {td_preprocess.PREPROCESS_VERSION!r}"
        )
    if frontend.get("uses_frequency_transform") is not False:
        raise RuntimeError(f"{source} does not declare an FFT-free frontend")

    state_path = source / str(metrics["artifacts"]["state_dict"])
    prototypes_path = source / str(metrics["artifacts"]["prototypes"])
    state_sha = dev._sha256(state_path)
    prototypes_sha = dev._sha256(prototypes_path)
    if state_sha != metrics["artifacts"]["state_dict_sha256"]:
        raise RuntimeError(f"{state_path} does not match its recorded SHA-256")
    if prototypes_sha != metrics["artifacts"]["prototypes_sha256"]:
        raise RuntimeError(
            f"{prototypes_path} does not match its recorded SHA-256"
        )

    config = InvariantPatchConfig(**metrics["architecture"]).validate()
    if config.encoder != encoder:
        raise ValueError(f"{source} architecture encoder is not {encoder!r}")
    net = InvariantPatchCNN(config)
    try:
        state = torch.load(state_path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with old PyTorch
        state = torch.load(state_path, map_location="cpu")
    net.load_state_dict(state, strict=True)

    return {
        "directory": source,
        "encoder": encoder,
        "metrics": metrics,
        "net": net.eval(),
        "prototypes": np.load(prototypes_path).astype(np.float32),
        "hashes": {
            "dev_metrics.json": dev._sha256(metrics_path),
            str(state_path.name): state_sha,
            str(prototypes_path.name): prototypes_sha,
        },
    }


def _assert_branches_match(real: dict[str, Any], complex_branch: dict[str, Any]) -> None:
    """Refuse to fuse branches that are not budget, split and frontend matched."""
    real_metrics = real["metrics"]
    complex_metrics = complex_branch["metrics"]

    for field in ("patch_length", "patch_count", "embed_dim", "n_features"):
        left = real_metrics["architecture"][field]
        right = complex_metrics["architecture"][field]
        if left != right:
            raise ValueError(f"branch {field} mismatch: {left} != {right}")
    if real_metrics["frontend"] != complex_metrics["frontend"]:
        raise ValueError("branch frontend metadata mismatch")
    if (
        real_metrics["source_index_contract"]["contract"]
        != complex_metrics["source_index_contract"]["contract"]
    ):
        raise ValueError("branch source index contract mismatch")

    real_views = real_metrics.get("training_views", {})
    complex_views = complex_metrics.get("training_views", {})
    if bool(real_views.get("enabled")) != bool(complex_views.get("enabled")):
        raise ValueError("branch multilength training flags disagree")
    if real_views.get("lengths") != complex_views.get("lengths"):
        raise ValueError("branch multilength training lengths disagree")
    if real_views.get("eligible_corpus_indices_sha256") != complex_views.get(
        "eligible_corpus_indices_sha256"
    ):
        raise ValueError("branch training populations disagree")

    real_episodes = real_metrics.get("training", {}).get("episodes")
    complex_episodes = complex_metrics.get("training", {}).get("episodes")
    if real_episodes != complex_episodes:
        raise ValueError(
            f"branch episode budgets disagree: {real_episodes} != "
            f"{complex_episodes}"
        )


def _source_hashes() -> dict[str, str]:
    paths = {
        "assemble_v3_fusion.py": Path(__file__).resolve(),
        "run_time_domain_dev.py": HERE / "run_time_domain_dev.py",
        "invariant_fusion.py": V2 / "invariant_fusion.py",
        "invariant_patch_cnn.py": V2 / "invariant_patch_cnn.py",
        "invariant_patch_data.py": V2 / "invariant_patch_data.py",
        "run_invariant_cnn_dev.py": V2 / "run_invariant_cnn_dev.py",
        "time_domain_geometry.py": TRAINING / "time_domain_geometry.py",
        "time_domain_invariant_patch_preprocess.py": (
            TRAINING / "time_domain_invariant_patch_preprocess.py"
        ),
        "invariant_patch_preprocess.py": (
            TRAINING / "invariant_patch_preprocess.py"
        ),
        "train.py": TRAINING / "train.py",
    }
    return {name: dev._sha256(path) for name, path in paths.items()}


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output = reject_sealed_path(Path(args.output_dir), "output")
    real_dir = reject_sealed_path(Path(args.real_dir), "source")
    complex_dir = reject_sealed_path(Path(args.complex_dir), "source")
    weight_real = validate_branch_weight(args.branch_weight)
    prototype_tolerance = float(args.prototype_tolerance)
    if not np.isfinite(prototype_tolerance) or prototype_tolerance <= 0.0:
        raise ValueError("prototype tolerance must be positive and finite")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} is not empty")

    branches = {
        "real": load_branch(real_dir, "real"),
        "complex": load_branch(complex_dir, "complex"),
    }
    _assert_branches_match(branches["real"], branches["complex"])

    config = branches["real"]["net"].cfg
    contract = branches["real"]["metrics"]["source_index_contract"]["contract"]
    target_frac = float(contract["config"]["target_frac"])

    source = invariant_data.load(
        patch_length=config.patch_length,
        patch_count=config.patch_count,
        target_frac=target_frac,
        model_seed=args.seed,
        build_if_missing=False,
    )
    if source["data_audit"].get("consumed_test_rows_exposed") != 0:
        raise AssertionError("data loader exposed consumed test rows")
    forbidden = [key for key in source if "test" in key.lower()]
    if forbidden:
        raise AssertionError(f"data loader exposed test-like keys: {forbidden}")
    if source["cache_contract"] != contract:
        raise RuntimeError("current cache contract differs from branch source")

    manifest = dev._load_manifest()
    training_views_enabled = bool(
        branches["real"]["metrics"].get("training_views", {}).get("enabled")
    )
    train_lengths = dev.LENGTHS if training_views_enabled else None
    data, contexts, training_view_audit = dev._prepare_data(
        source,
        manifest,
        patch_length=config.patch_length,
        patch_count=config.patch_count,
        target_frac=target_frac,
        train_lengths=train_lengths,
    )

    device = runner.resolve_device(args.device)
    runner.seed_everything(args.seed)
    nets = {
        name: branch["net"].to(device).eval()
        for name, branch in branches.items()
    }

    embeddings: dict[str, dict[str, np.ndarray]] = {
        name: {
            split: embed_all(
                net, data[f"x{split}"], data[f"f{split}"], device
            ).astype(np.float32, copy=False)
            for split in (CENTER_SPLIT, PROTOTYPE_SPLIT, METRIC_SPLIT)
        }
        for name, net in nets.items()
    }
    for name, splits in embeddings.items():
        for split, value in splits.items():
            if not np.isfinite(value).all():
                raise RuntimeError(f"{name} branch produced non-finite {split}")

    # Contract point 1: centers come from training rows and nothing else.
    real_center = fit_center(embeddings["real"])
    complex_center = fit_center(embeddings["complex"])

    # Each branch's own enrollment prototypes must reproduce, or the checkpoint,
    # the split, or the frontend is not the one that produced the artifact.
    prototype_errors = {}
    for name, branch in branches.items():
        recomputed = prototypes_from(
            embeddings[name][PROTOTYPE_SPLIT],
            np.asarray(data["yen"], dtype=np.int64),
            int(data["n_classes"]),
        )
        error = float(np.max(np.abs(recomputed - branch["prototypes"])))
        prototype_errors[name] = error
        if error > prototype_tolerance:
            raise RuntimeError(
                f"{name} branch prototypes did not reproduce: {error:g} > "
                f"{prototype_tolerance:g}"
            )

    fusion = (
        CenteredInvariantFusion(
            nets["real"],
            nets["complex"],
            real_center,
            complex_center,
            alpha_real=ALPHA_REAL,
            alpha_complex=ALPHA_COMPLEX,
            weight_real=weight_real,
            eps=FUSION_EPS,
        )
        .to(device)
        .eval()
    )
    embedder = FusionEmbedder(fusion).to(device).eval()

    fused = {
        split: fuse_numpy(
            embeddings["real"][split],
            embeddings["complex"][split],
            real_center,
            complex_center,
            weight_real=weight_real,
        )
        for split in (CENTER_SPLIT, PROTOTYPE_SPLIT, METRIC_SPLIT)
    }
    probe = embed_all(
        embedder,
        data[f"x{METRIC_SPLIT}"][:64],
        data[f"f{METRIC_SPLIT}"][:64],
        device,
    )
    fusion_assembly_error = float(
        np.max(np.abs(probe - fused[METRIC_SPLIT][:64]))
    )
    if fusion_assembly_error > FUSION_ASSEMBLY_TOLERANCE:
        raise RuntimeError(
            f"NumPy fusion assembly mismatch: {fusion_assembly_error}"
        )

    # Contract point 2: prototypes come from enrollment rows and nothing else.
    prototypes = fit_prototypes(
        fused, np.asarray(data["yen"], dtype=np.int64), int(data["n_classes"])
    )

    labels = np.asarray(data["yva"], dtype=np.int64)
    selection_indices = np.asarray(data["va_idx"], dtype=np.int64)
    metadata_labels, impaired, snr = dev._metadata_arrays(
        manifest, selection_indices, data["classes"]
    )
    if not np.array_equal(labels, metadata_labels):
        raise AssertionError("selection labels disagree with manifest")

    closed, _prediction = dev._classification_report(
        fused[METRIC_SPLIT],
        labels,
        prototypes,
        data["classes"],
        impaired=impaired,
        snr_db=snr,
    )
    print(
        f"[v3 fusion] closed bal={closed['balanced_accuracy']:.4f} "
        f"clean={closed['clean_accuracy']:.4f}",
        flush=True,
    )
    scale = dev._scale_audit(embedder, prototypes, data, manifest, device)
    length = dev._length_audit(embedder, prototypes, data, manifest, device)

    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "fusion_state_dict.pt"
    prototypes_path = output / "fusion_prototypes.npy"
    # Plain .npy rather than .npz: a zip container embeds a modification time,
    # which would make the recorded artifact SHA-256 differ between two
    # otherwise identical assemblies.
    real_center_path = output / "real_center.npy"
    complex_center_path = output / "complex_center.npy"
    feature_mean_path = output / "feature_mean.npy"
    feature_std_path = output / "feature_std.npy"
    torch.save(
        {
            key: value.detach().cpu()
            for key, value in fusion.state_dict().items()
        },
        state_path,
    )
    np.save(prototypes_path, prototypes)
    np.save(real_center_path, real_center)
    np.save(complex_center_path, complex_center)
    np.save(feature_mean_path, np.asarray(data["fmean"], dtype=np.float32))
    np.save(feature_std_path, np.asarray(data["fstd"], dtype=np.float32))

    branch_provenance = {
        name: {
            "directory": str(branch["directory"]),
            "encoder": branch["encoder"],
            "sha256": branch["hashes"],
            "architecture": branch["metrics"]["architecture"],
            "parameter_count": branch["metrics"]["parameter_count"],
            "episodes": branch["metrics"].get("training", {}).get("episodes"),
            "training_views_enabled": bool(
                branch["metrics"].get("training_views", {}).get("enabled")
            ),
            "closed_selection_balanced_accuracy": branch["metrics"][
                "closed_selection"
            ]["balanced_accuracy"],
            "prototype_reproduction_max_abs_error": prototype_errors[name],
        }
        for name, branch in branches.items()
    }

    result = {
        "status": "complete",
        "development_only": True,
        "release_evidence": False,
        "sealed_release_data_used": 0,
        "consumed_test_rows_used": 0,
        "frontend": td_preprocess.preprocess_metadata(),
        "encoder": "fusion",
        "architecture": fusion.config(),
        "parameter_count": int(fusion.parameter_counts()["total"]),
        "training": {
            "performed_here": False,
            "role": (
                "no gradient step is taken by this assembly; both branches are "
                "loaded frozen from their development artifacts"
            ),
            "episodes": branches["real"]["metrics"]
            .get("training", {})
            .get("episodes"),
            "branch_history_summary": {
                name: {
                    key: branch["metrics"].get("training", {}).get(key)
                    for key in (
                        "episodes",
                        "best_selection_accuracy",
                        "best_selection_balanced_accuracy",
                        "final_logit_scale",
                        "view_consistency_weight",
                    )
                }
                for name, branch in branches.items()
            },
        },
        "training_views": {
            **training_view_audit,
            "asserted_identical_across_branches": True,
            "branch_recorded_views": {
                name: branch["metrics"].get("training_views", {})
                for name, branch in branches.items()
            },
        },
        "source_branches": branch_provenance,
        "fusion": {
            "kind": "centered_invariant_fusion",
            "rule": (
                "unit(concat(sqrt(w) * unit(real - alpha_real * real_center), "
                "sqrt(1 - w) * unit(complex - alpha_complex * "
                "complex_center)))"
            ),
            "weight_real": weight_real,
            "weight_complex": float(1.0 - weight_real),
            "alpha_real": float(ALPHA_REAL),
            "alpha_complex": float(ALPHA_COMPLEX),
            "eps": float(FUSION_EPS),
            "numpy_vs_module_max_abs_error": fusion_assembly_error,
            "numpy_vs_module_tolerance": float(FUSION_ASSEMBLY_TOLERANCE),
        },
        "fitting_contract": {
            "center_fit_population": "training only",
            "center_rows": int(len(embeddings["real"][CENTER_SPLIT])),
            "prototype_population": "enrollment only",
            "prototype_rows": int(len(fused[PROTOTYPE_SPLIT])),
            "selection_population_role": (
                "scored for metrics only; never fit, calibrated, or selected on"
            ),
            "feature_standardization_population": (
                "training raw features only, rebuilt by run_time_domain_dev"
            ),
            "sealed_release_rows_loaded": 0,
            "consumed_test_rows_loaded": 0,
            "branch_prototype_reproduction_tolerance": prototype_tolerance,
        },
        "closed_selection": closed,
        "population_physical_scale": scale,
        "population_causal_length": length,
        "geometry_summary": {
            name: {
                "count": len(rows),
                "bandwidth_median": float(
                    np.median([row["bw"] for row in rows])
                ),
                "center_median": float(
                    np.median([row["center"] for row in rows])
                ),
            }
            for name, rows in contexts.items()
        },
        "artifacts": {
            "state_dict": state_path.name,
            "state_dict_sha256": dev._sha256(state_path),
            "prototypes": prototypes_path.name,
            "prototypes_sha256": dev._sha256(prototypes_path),
            "real_center": real_center_path.name,
            "real_center_sha256": dev._sha256(real_center_path),
            "complex_center": complex_center_path.name,
            "complex_center_sha256": dev._sha256(complex_center_path),
            "feature_mean": feature_mean_path.name,
            "feature_mean_sha256": dev._sha256(feature_mean_path),
            "feature_std": feature_std_path.name,
            "feature_std_sha256": dev._sha256(feature_std_path),
        },
        "source_sha256": _source_hashes(),
        "data_audit": data["data_audit"],
        "source_index_contract": {
            "contract": data["cache_contract"],
            "role": (
                "identifies the raw corpus and exposed train/enroll/selection "
                "indices only; every tensor in this assembly was rebuilt with "
                "the time-domain v3 frontend"
            ),
        },
        "device": str(device),
        "seed": int(args.seed),
        "wall_clock_s": time.perf_counter() - started,
    }
    validate_result_schema(result)
    dev._write_json(output / "dev_metrics.json", result)
    print(f"[v3 fusion] wrote {output}", flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--complex-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--branch-weight",
        type=float,
        default=DEFAULT_WEIGHT_REAL,
        help=(
            "real-branch fusion weight in [0, 1]; the complex branch receives "
            "1 - this value"
        ),
    )
    parser.add_argument(
        "--prototype-tolerance",
        type=float,
        default=DEFAULT_PROTOTYPE_TOLERANCE,
        help=(
            "maximum tolerated disagreement between each branch's stored "
            "prototypes and the prototypes recomputed from enrollment rows"
        ),
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
