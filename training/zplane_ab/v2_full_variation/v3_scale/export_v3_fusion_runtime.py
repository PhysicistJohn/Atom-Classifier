"""Export a v3 time-domain invariant fusion to a deterministic runtime bundle.

This is the v3 replacement for ``export_invariant_fusion.py``.  It reads one
fusion directory produced by ``assemble_v3_fusion.py`` and writes a fresh,
self-contained runtime bundle.

What is deliberately different from the v2 exporter:

- The bundle declares its own schema id.  It never claims ``invariant-patch-v1``
  or ``hybrid-v3``; those names belong to the FFT-bearing v2 frontend and to the
  hybrid detector that v3 replaces, and reusing them would let a v2 consumer
  silently mis-preprocess a v3 model.
- The bundle binds the SHA-256 of ``training/time_domain_geometry.py`` and
  ``training/time_domain_invariant_patch_preprocess.py``.  Those two files *are*
  the v3 frontend; if either moves, every packed patch this model was fit on
  changes meaning, so the export refuses rather than shipping a stale binding.
- Nothing is fitted, calibrated, selected, or retrained here.  Centers,
  prototypes and feature moments are copied byte-for-byte from the assembly and
  verified against its recorded hashes.
- No corpus, split loader, sealed release suite, or consumed test row is opened.
  The probe set is closed-form synthetic and data-blind.

Determinism.  The bundle contains no timestamp, no wall clock, no absolute host
path, and no ``.npz``.  A ``.npz`` is a zip container that records a per-member
modification time, which would make the bundle SHA-256 differ between two
otherwise identical exports; arrays are therefore written as plain ``.npy`` and
metadata as ``sort_keys=True`` JSON.  Exporting the same source twice must
produce byte-identical files, and the unit suite asserts exactly that.

Open-set rejection.  HANDOFF 10.5 requires the rejector to be refit against the
final v3 embeddings.  Until that happens the bundle carries an explicit unset
rejection slot: the field exists, states that it is unset, states why, and
records the contract a future policy must satisfy.  A fitted policy may be
attached with ``--rejector-dir``; it is accepted only if its provenance binds it
to *this* fusion assembly.

This produces a development bundle.  It is not release evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
V2 = HERE.parent
ZPAB = V2.parent
TRAINING = ZPAB.parent
REPO = TRAINING.parent
for _path in (TRAINING, ZPAB, V2, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import time_domain_geometry as geometry  # noqa: E402
import time_domain_invariant_patch_preprocess as td_preprocess  # noqa: E402
import v3_time_domain_openset as openset  # noqa: E402
from invariant_fusion import CenteredInvariantFusion  # noqa: E402
from invariant_patch_cnn import (  # noqa: E402
    InvariantPatchCNN,
    InvariantPatchConfig,
)


# The v3 bundle's own identity.  Not "invariant-patch-v1", not "hybrid-v3".
SCHEMA_ID = "atomos.v3.time-domain-invariant-fusion.runtime-bundle"
SCHEMA_VERSION = 1

# Schema names a v3 bundle must never adopt.  The first two are the v2/hybrid
# frontend contracts; the third is the v2 paired-real staging export.
FORBIDDEN_SCHEMA_IDS = frozenset(
    {
        "invariant-patch-v1",
        "hybrid-v3",
        "atomos.invariant-fusion.paired-real",
        "invariant-centered-fusion-with-lof-rank-ensemble",
    }
)

# The two source files that define the FFT-free frontend.  Their hashes are
# bound into the bundle and checked against the assembly record.
BOUND_FRONTEND_SOURCES = {
    "training/time_domain_geometry.py": TRAINING / "time_domain_geometry.py",
    "training/time_domain_invariant_patch_preprocess.py": (
        TRAINING / "time_domain_invariant_patch_preprocess.py"
    ),
}
# How ``assemble_v3_fusion._source_hashes`` keys the same two files.
ASSEMBLY_SOURCE_KEYS = {
    "training/time_domain_geometry.py": "time_domain_geometry.py",
    "training/time_domain_invariant_patch_preprocess.py": (
        "time_domain_invariant_patch_preprocess.py"
    ),
}

MANIFEST_NAME = "bundle_manifest.json"
PROBE_NAME = "probe_fixture.json"
REJECTOR_DIRNAME = "rejector"
REJECTOR_PROVENANCE_NAME = "provenance.json"

# Assets copied from the assembly, mapped to their ``artifacts`` record keys.
COPIED_ASSETS = (
    ("state_dict", "fusion_state_dict.pt"),
    ("prototypes", "fusion_prototypes.npy"),
    ("real_center", "real_center.npy"),
    ("complex_center", "complex_center.npy"),
    ("feature_mean", "feature_mean.npy"),
    ("feature_std", "feature_std.npy"),
)

# Maximum tolerated disagreement between the embeddings computed from the source
# assembly and the embeddings recomputed after reloading the written bundle.
# Both runs are float32 CPU evaluations of the same weights in the same process,
# so the honest expectation is exact equality; the tolerance exists only to
# absorb a future reduction-order change, not to hide a wrong checkpoint.
SELF_VERIFICATION_TOLERANCE = 1e-6

# Standardization epsilon used by run_time_domain_dev when fitting feature_std.
FEATURE_STANDARDIZATION_EPSILON = 1e-6

# Closed-form, data-blind probe captures.  Fixed formulas, no RNG, no seed, no
# corpus row.  Lengths and physical scales are chosen to exercise the property
# v3 exists to fix: identical content observed at different sample rates.
PROBE_SPECIFICATIONS: tuple[dict[str, Any], ...] = (
    {"name": "two-tone-scale-1x", "length": 4096, "physical_scale": 1.0,
     "center": 0.13, "kind": "two_tone"},
    {"name": "two-tone-scale-2x", "length": 2048, "physical_scale": 2.0,
     "center": 0.13, "kind": "two_tone"},
    {"name": "chirped-envelope", "length": 4096, "physical_scale": 1.0,
     "center": 0.07, "kind": "chirp"},
    {"name": "bursty-fm", "length": 4096, "physical_scale": 1.0,
     "center": -0.19, "kind": "bursty_fm"},
)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
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
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"refusing to serialize {type(value).__name__} into a bundle")


def _write_json(path: Path, payload: Any) -> None:
    """Write deterministic JSON: sorted keys, no NaN, trailing newline."""
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


def reject_sealed_path(path: Path, role: str) -> Path:
    """Refuse any release or sealed location, for reading or for writing.

    Mirrors ``assemble_v3_fusion.reject_sealed_path``; duplicated rather than
    imported so the exporter never drags in the corpus/split loader.
    """
    resolved = Path(path).expanduser().resolve()
    lowered = {part.lower() for part in resolved.parts}
    if "releases" in lowered or any("sealed" in part for part in lowered):
        raise ValueError(
            f"runtime export refuses release or sealed {role} paths: {resolved}"
        )
    return resolved


def _repo_relative(path: Path) -> str:
    """Return a host-independent path so the bundle stays machine-portable."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO))
    except ValueError:
        return resolved.name


def sanitized_branch_provenance(branches: Any) -> dict[str, Any]:
    """Copy the branch provenance with host paths reduced to repo-relative.

    ``assemble_v3_fusion`` records absolute branch directories.  Those are
    correct locally and useless anywhere else, so the bundle keeps the
    repo-relative form; every hash and architecture field is passed through
    untouched.
    """
    if not isinstance(branches, Mapping):
        return {}
    output: dict[str, Any] = {}
    for name, record in branches.items():
        if not isinstance(record, Mapping):
            output[str(name)] = record
            continue
        copied = dict(record)
        if "directory" in copied:
            copied["directory"] = _repo_relative(Path(str(copied["directory"])))
        output[str(name)] = copied
    return output


def frontend_source_hashes() -> dict[str, str]:
    """Hash the two files that define the FFT-free v3 frontend."""
    hashes: dict[str, str] = {}
    for name, path in BOUND_FRONTEND_SOURCES.items():
        if not path.is_file():
            raise FileNotFoundError(f"bound frontend source is missing: {path}")
        hashes[name] = _sha256(path)
    return hashes


def validate_schema_identity() -> None:
    """Fail closed if this exporter ever adopts a v2/hybrid schema name."""
    if SCHEMA_ID in FORBIDDEN_SCHEMA_IDS:
        raise RuntimeError(
            "the v3 runtime bundle must declare its own schema id and must not "
            f"claim {SCHEMA_ID!r}"
        )


def class_order(metrics: Mapping[str, Any]) -> list[str]:
    """Recover the label index order without touching the corpus.

    ``run_time_domain_dev`` derives class indices as
    ``sorted(manifest["classes"])`` and keys ``data_audit.class_counts`` by the
    same names, so sorting those keys reproduces the index order exactly.  The
    three populations must agree or the audit is not describing one split.
    """
    counts = metrics.get("data_audit", {}).get("class_counts")
    if not isinstance(counts, Mapping) or not counts:
        raise ValueError("dev_metrics data_audit.class_counts is missing")
    populations = ("train", "enroll", "selection")
    orders = []
    for population in populations:
        rows = counts.get(population)
        if not isinstance(rows, Mapping) or not rows:
            raise ValueError(
                f"data_audit.class_counts.{population} is missing or empty"
            )
        orders.append(sorted(str(name) for name in rows))
    if any(order != orders[0] for order in orders[1:]):
        raise ValueError("data_audit class populations disagree on class names")
    classes = orders[0]
    if len(set(classes)) != len(classes) or any(not name for name in classes):
        raise ValueError("class names must be unique and non-empty")
    return classes


CORPUS_MANIFEST = TRAINING / "artifacts" / "signallab-corpus" / "corpus.json"


def _cross_check_class_order(
    classes: Sequence[str],
    manifest_path: Path | None = None,
) -> None:
    """Optionally confirm the derived order against the corpus manifest.

    Reads only ``corpus.json`` metadata, never the 3.67 GB capture file, and is
    skipped when absent.  The outcome is intentionally not recorded in the
    bundle: a field that depends on whether a local file happens to exist would
    make the bundle SHA host-dependent.
    """
    manifest_path = CORPUS_MANIFEST if manifest_path is None else Path(manifest_path)
    if not manifest_path.is_file():
        return
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    declared = sorted(str(name) for name in manifest.get("classes", []))
    if declared and declared != list(classes):
        raise RuntimeError(
            "derived class order disagrees with the corpus manifest: "
            f"{list(classes)} != {declared}"
        )


# ---------------------------------------------------------------------------
# source assembly loading
# ---------------------------------------------------------------------------


def _require(metrics: Mapping[str, Any], key: str) -> Any:
    if key not in metrics:
        raise ValueError(f"dev_metrics.json is missing {key!r}")
    return metrics[key]


def _validate_provenance(metrics: Mapping[str, Any]) -> None:
    for key, expected in (
        ("development_only", True),
        ("release_evidence", False),
        ("sealed_release_data_used", 0),
        ("consumed_test_rows_used", 0),
        ("status", "complete"),
        ("encoder", "fusion"),
    ):
        if _require(metrics, key) != expected:
            raise RuntimeError(
                f"source assembly {key} must be {expected!r}, got "
                f"{metrics[key]!r}"
            )
    audit = _require(metrics, "data_audit")
    if not isinstance(audit, Mapping):
        raise ValueError("data_audit must be a mapping")
    if audit.get("consumed_test_rows_exposed") != 0:
        raise RuntimeError("source assembly exposed consumed test rows")
    contract = _require(metrics, "fitting_contract")
    for key, expected in (
        ("center_fit_population", "training only"),
        ("prototype_population", "enrollment only"),
        ("sealed_release_rows_loaded", 0),
        ("consumed_test_rows_loaded", 0),
    ):
        if contract.get(key) != expected:
            raise RuntimeError(
                f"fitting_contract.{key} must be {expected!r}, got "
                f"{contract.get(key)!r}"
            )


def _validate_frontend(metrics: Mapping[str, Any], config: InvariantPatchConfig) -> None:
    frontend = _require(metrics, "frontend")
    if not isinstance(frontend, Mapping):
        raise ValueError("frontend metadata must be a mapping")
    if frontend.get("version") != td_preprocess.PREPROCESS_VERSION:
        raise RuntimeError(
            f"assembly frontend {frontend.get('version')!r} is not the current "
            f"{td_preprocess.PREPROCESS_VERSION!r}"
        )
    if frontend.get("uses_frequency_transform") is not False:
        raise RuntimeError("assembly does not declare an FFT-free frontend")
    recorded_geometry = frontend.get("geometry")
    if not isinstance(recorded_geometry, Mapping):
        raise ValueError("frontend geometry metadata is missing")
    if recorded_geometry.get("uses_frequency_transform") is not False:
        raise RuntimeError("assembly geometry claims a frequency transform")
    if recorded_geometry != geometry.estimator_metadata():
        raise RuntimeError(
            "assembly geometry metadata differs from the current estimator"
        )
    if int(frontend.get("patch_length", -1)) != int(config.patch_length):
        raise RuntimeError("frontend patch_length disagrees with the model")
    if int(frontend.get("patch_count", -1)) != int(config.patch_count):
        raise RuntimeError("frontend patch_count disagrees with the model")


def _validate_bound_sources(metrics: Mapping[str, Any]) -> dict[str, str]:
    """Hash the frontend now and require the assembly to have seen the same."""
    live = frontend_source_hashes()
    recorded = _require(metrics, "source_sha256")
    if not isinstance(recorded, Mapping):
        raise ValueError("source_sha256 must be a mapping")
    for name, digest in live.items():
        key = ASSEMBLY_SOURCE_KEYS[name]
        if key not in recorded:
            raise RuntimeError(
                f"assembly did not record a hash for {key}; it cannot be bound"
            )
        if recorded[key] != digest:
            raise RuntimeError(
                f"{name} changed since assembly ({recorded[key]} -> {digest}); "
                "the exported model would claim a frontend it was not fit on"
            )
    return live


def load_fusion_artifact(directory: Path) -> dict[str, Any]:
    """Load and verify one ``assemble_v3_fusion`` output directory."""
    source = reject_sealed_path(Path(directory), "source")
    if not source.is_dir():
        raise FileNotFoundError(source)
    metrics_path = source / "dev_metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    with metrics_path.open(encoding="utf-8") as handle:
        metrics = json.load(handle)

    _validate_provenance(metrics)
    architecture = _require(metrics, "architecture")
    if architecture.get("kind") != "centered_invariant_fusion":
        raise RuntimeError(
            f"expected a centered_invariant_fusion, got "
            f"{architecture.get('kind')!r}"
        )
    real_config = InvariantPatchConfig(**dict(architecture["real"])).validate()
    complex_config = InvariantPatchConfig(
        **dict(architecture["complex"])
    ).validate()
    if real_config.encoder != "real" or complex_config.encoder != "complex":
        raise ValueError("fusion branches must be ordered real then complex")
    for field in ("patch_length", "patch_count", "n_features", "embed_dim"):
        if getattr(real_config, field) != getattr(complex_config, field):
            raise ValueError(f"branch config mismatch for {field}")
    _validate_frontend(metrics, real_config)
    bound_sources = _validate_bound_sources(metrics)

    artifacts = _require(metrics, "artifacts")
    paths: dict[str, Path] = {}
    for key, expected_name in COPIED_ASSETS:
        recorded_name = str(artifacts[key])
        if recorded_name != expected_name:
            raise RuntimeError(
                f"assembly artifact {key} is {recorded_name!r}, expected "
                f"{expected_name!r}"
            )
        path = source / recorded_name
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = _sha256(path)
        if digest != artifacts[f"{key}_sha256"]:
            raise RuntimeError(f"{path} does not match its recorded SHA-256")
        paths[key] = path

    embed_dim = int(real_config.embed_dim)
    real_center = np.load(paths["real_center"], allow_pickle=False)
    complex_center = np.load(paths["complex_center"], allow_pickle=False)
    feature_mean = np.load(paths["feature_mean"], allow_pickle=False)
    feature_std = np.load(paths["feature_std"], allow_pickle=False)
    prototypes = np.load(paths["prototypes"], allow_pickle=False)
    for name, array, shape in (
        ("real_center", real_center, (embed_dim,)),
        ("complex_center", complex_center, (embed_dim,)),
        ("feature_mean", feature_mean, (int(real_config.n_features),)),
        ("feature_std", feature_std, (int(real_config.n_features),)),
    ):
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"{name} must be a finite array of shape {shape}")
    if not np.all(feature_std > 0.0):
        raise ValueError("feature_std must be strictly positive")

    classes = class_order(metrics)
    _cross_check_class_order(classes)
    if prototypes.shape != (len(classes), 2 * embed_dim):
        raise ValueError(
            f"prototypes must have shape {(len(classes), 2 * embed_dim)}, got "
            f"{prototypes.shape}"
        )
    if not np.isfinite(prototypes).all():
        raise ValueError("prototypes must be finite")

    fusion = CenteredInvariantFusion(
        InvariantPatchCNN(real_config),
        InvariantPatchCNN(complex_config),
        real_center,
        complex_center,
        alpha_real=float(architecture["alpha_real"]),
        alpha_complex=float(architecture["alpha_complex"]),
        weight_real=float(architecture["weight_real"]),
        eps=float(architecture["eps"]),
    )
    state = torch.load(paths["state_dict"], map_location="cpu", weights_only=True)
    fusion.load_state_dict(state, strict=True)
    fusion = fusion.to(torch.device("cpu")).eval()

    fusion_record = _require(metrics, "fusion")
    weight_real = float(fusion_record["weight_real"])
    weight_complex = float(fusion_record["weight_complex"])
    if abs(weight_real + weight_complex - 1.0) > 1e-9:
        raise ValueError("recorded branch weights must sum to one")
    if abs(float(fusion.weight_real) - weight_real) > 1e-6:
        raise RuntimeError("serialized weight_real disagrees with dev_metrics")

    contract = _require(metrics, "source_index_contract")["contract"]
    target_frac = float(contract["config"]["target_frac"])
    if abs(float(metrics["frontend"]["target_frac"]) - target_frac) > 1e-12:
        raise RuntimeError("frontend target_frac disagrees with the split contract")

    return {
        "directory": source,
        "metrics": metrics,
        "dev_metrics_sha256": _sha256(metrics_path),
        "paths": paths,
        "real_config": real_config,
        "complex_config": complex_config,
        "fusion": fusion,
        "prototypes": prototypes.astype(np.float32, copy=False),
        "feature_mean": feature_mean.astype(np.float32, copy=False),
        "feature_std": feature_std.astype(np.float32, copy=False),
        "classes": classes,
        "weight_real": weight_real,
        "weight_complex": weight_complex,
        "target_frac": target_frac,
        "bound_frontend_sha256": bound_sources,
    }


# ---------------------------------------------------------------------------
# probe set and embedding reproduction
# ---------------------------------------------------------------------------


def probe_captures(
    specifications: Sequence[Mapping[str, Any]] = PROBE_SPECIFICATIONS,
) -> list[np.ndarray]:
    """Build the fixed, closed-form, data-blind probe captures."""
    captures: list[np.ndarray] = []
    for specification in specifications:
        length = int(specification["length"])
        scale = float(specification["physical_scale"])
        center = float(specification["center"])
        kind = str(specification["kind"])
        sample = np.arange(length, dtype=np.float64)
        physical = sample * scale
        if kind == "two_tone":
            capture = np.exp(
                1j * 2.0 * np.pi * (center * sample + 0.051 * physical)
            ) + 0.37 * np.exp(
                1j * 2.0 * np.pi * (center * sample - 0.113 * physical)
            )
        elif kind == "chirp":
            envelope = 0.71 + 0.19 * np.cos(2.0 * np.pi * 0.0029 * physical)
            capture = envelope * np.exp(
                1j
                * 2.0
                * np.pi
                * (
                    center * sample
                    + 0.043 * physical
                    + 0.0000037 * np.square(physical)
                )
            )
        elif kind == "bursty_fm":
            burst = (np.remainder(sample, 137.0) < 91.0).astype(np.float64)
            envelope = 0.33 + 0.67 * burst
            capture = envelope * np.exp(
                1j
                * 2.0
                * np.pi
                * (
                    center * sample
                    + 0.029 * np.sin(2.0 * np.pi * physical / 83.0)
                )
            )
        else:
            raise ValueError(f"unknown probe kind {kind!r}")
        capture = np.asarray(capture, dtype=np.complex128)
        if not np.isfinite(capture).all():
            raise RuntimeError(f"probe {specification['name']} is not finite")
        captures.append(capture)
    return captures


@torch.no_grad()
def probe_embeddings(
    fusion: CenteredInvariantFusion,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    *,
    patch_length: int,
    patch_count: int,
    n_features: int,
    target_frac: float,
    specifications: Sequence[Mapping[str, Any]] = PROBE_SPECIFICATIONS,
) -> dict[str, Any]:
    """Run the fixed probe set through the v3 frontend and the fusion."""
    captures = probe_captures(specifications)
    rows = len(captures)
    packed = np.empty((rows, 2, patch_length * patch_count), dtype=np.float32)
    raw_features = np.empty((rows, n_features), dtype=np.float32)
    contexts: list[dict[str, Any]] = []
    for row, capture in enumerate(captures):
        packed[row], raw_features[row], context = td_preprocess.preprocess(
            capture,
            patch_length=patch_length,
            patch_count=patch_count,
            target_frac=target_frac,
        )
        contexts.append(context)
    # Exactly run_time_domain_dev's standardization: float32 raw features minus
    # float32 stored moments, divided, cast back to float32.
    features = (
        (raw_features - np.asarray(feature_mean)) / np.asarray(feature_std)
    ).astype(np.float32)
    packed_tensor = torch.from_numpy(packed)
    feature_tensor = torch.from_numpy(features)
    module = fusion.eval()
    real = module.real_branch(packed_tensor, feature_tensor)
    complex_branch = module.complex_branch(packed_tensor, feature_tensor)
    fused = module(packed_tensor, feature_tensor)
    return {
        "captures": captures,
        "contexts": contexts,
        "packed": packed,
        "raw_features": raw_features,
        "standardized_features": features,
        "real_embedding": real.detach().cpu().numpy(),
        "complex_embedding": complex_branch.detach().cpu().numpy(),
        "fused_embedding": fused.detach().cpu().numpy(),
    }


def _closed_set(
    fused: np.ndarray,
    prototypes: np.ndarray,
    classes: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    difference = fused[:, None, :].astype(np.float64) - prototypes[None, :, :]
    distances = np.sum(np.square(difference), axis=-1)
    winner = np.argmin(distances, axis=1)
    labels = [str(classes[int(index)]) for index in winner]
    return distances, winner, labels


# ---------------------------------------------------------------------------
# rejection slot
# ---------------------------------------------------------------------------


def required_rejector_contract() -> dict[str, Any]:
    """The contract any attached v3 rejector policy must satisfy."""
    return {
        "module": "v3_time_domain_openset",
        "policy_schema": int(openset.FROZEN_POLICY_SCHEMA),
        "policy_kind": str(openset.FROZEN_POLICY_KIND),
        "geometry_feature": str(openset.FROZEN_GEOMETRY_FEATURE),
        "v2_weight": float(openset.FROZEN_V2_WEIGHT),
        "geometry_weight": float(openset.FROZEN_GEOMETRY_WEIGHT),
        "threshold_quantile": float(openset.FROZEN_THRESHOLD_QUANTILE),
        "density_fit_population": "training only",
        "rank_and_threshold_population": "enrollment only",
        "must_be_fit_against": (
            "the embeddings of this exact fusion assembly, identified by "
            "provenance.source_dev_metrics_sha256"
        ),
        "cannot_change_closed_label": True,
    }


def unset_rejection_slot() -> dict[str, Any]:
    """The explicit, non-null description of an absent rejector."""
    return {
        "state": "unset",
        "fitted": False,
        "policy": None,
        "asset_directory": None,
        "reason": (
            "the v3 open-set rejector has not been refit against these v3 "
            "embeddings; HANDOFF 10.5 requires that refit before any policy may "
            "be frozen into a bundle"
        ),
        "runtime_behaviour": (
            "closed-set only; a consumer of this bundle must not abstain and "
            "must not substitute a v2 rejector"
        ),
        "cannot_change_closed_label": True,
        "required_contract": required_rejector_contract(),
    }


def load_rejector_policy(
    directory: Path,
    *,
    dev_metrics_sha256: str,
    patch_length: int,
    patch_count: int,
) -> dict[str, Any]:
    """Load a fitted policy from a directory of ``.npy`` files.

    The directory layout is one ``<key>.npy`` per key of
    ``FrozenV3OpenSet.to_payload`` plus a ``provenance.json``.  A ``.npz`` is
    deliberately not accepted: its zip container records member modification
    times and would make the bundle SHA-256 unstable.
    """
    source = reject_sealed_path(Path(directory), "rejector")
    if not source.is_dir():
        raise FileNotFoundError(source)
    provenance_path = source / REJECTOR_PROVENANCE_NAME
    if not provenance_path.is_file():
        raise FileNotFoundError(provenance_path)
    with provenance_path.open(encoding="utf-8") as handle:
        provenance = json.load(handle)
    if not isinstance(provenance, Mapping):
        raise ValueError("rejector provenance must be a mapping")

    bound = provenance.get("fitted_against_fusion_dev_metrics_sha256")
    if bound != dev_metrics_sha256:
        raise RuntimeError(
            "rejector policy was not fit against this fusion assembly "
            f"({bound!r} != {dev_metrics_sha256!r}); HANDOFF 10.5 forbids "
            "attaching a rejector fitted on other embeddings"
        )
    for key, expected in (
        ("density_fit_population", "training only"),
        ("rank_and_threshold_population", "enrollment only"),
        ("selection_or_novelty_used_in_fit", False),
        ("cannot_change_closed_label", True),
    ):
        if provenance.get(key) != expected:
            raise RuntimeError(
                f"rejector provenance {key} must be {expected!r}, got "
                f"{provenance.get(key)!r}"
            )
    if int(provenance.get("patch_length", -1)) != int(patch_length):
        raise RuntimeError("rejector patch_length disagrees with the bundle")
    if int(provenance.get("patch_count", -1)) != int(patch_count):
        raise RuntimeError("rejector patch_count disagrees with the bundle")

    payload: dict[str, np.ndarray] = {}
    for path in sorted(source.glob("*.npy")):
        payload[path.stem] = np.load(path, allow_pickle=False)
    # from_payload performs the full frozen-policy validation, including the
    # weights, the quantile, sorted calibrations and the threshold identity.
    policy = openset.FrozenV3OpenSet.from_payload(payload)
    return {"policy": policy, "provenance": dict(provenance)}


def _write_rejector(
    output_dir: Path,
    loaded: Mapping[str, Any],
) -> dict[str, Any]:
    target = output_dir / REJECTOR_DIRNAME
    target.mkdir(parents=True, exist_ok=False)
    payload = loaded["policy"].to_payload()
    assets: dict[str, dict[str, Any]] = {}
    for key in sorted(payload):
        path = target / f"{key}.npy"
        np.save(path, np.asarray(payload[key]), allow_pickle=False)
        assets[f"{REJECTOR_DIRNAME}/{key}.npy"] = {
            "sha256": _sha256(path),
            "bytes": int(path.stat().st_size),
        }
    provenance_path = target / REJECTOR_PROVENANCE_NAME
    _write_json(provenance_path, loaded["provenance"])
    assets[f"{REJECTOR_DIRNAME}/{REJECTOR_PROVENANCE_NAME}"] = {
        "sha256": _sha256(provenance_path),
        "bytes": int(provenance_path.stat().st_size),
    }
    contract = required_rejector_contract()
    return {
        "slot": {
            "state": "fitted",
            "fitted": True,
            "policy": {
                "kind": str(openset.FROZEN_POLICY_KIND),
                "schema": int(openset.FROZEN_POLICY_SCHEMA),
                "geometry_feature": str(openset.FROZEN_GEOMETRY_FEATURE),
                "v2_weight": float(openset.FROZEN_V2_WEIGHT),
                "geometry_weight": float(openset.FROZEN_GEOMETRY_WEIGHT),
                "threshold_quantile": float(openset.FROZEN_THRESHOLD_QUANTILE),
                "threshold": float(loaded["policy"].threshold),
                "score": (
                    "empirical_rank(v2_weight * v2_rank + geometry_weight * "
                    "geometry_rank, enrollment calibration)"
                ),
                "comparison": "unknown iff score > threshold",
            },
            "asset_directory": REJECTOR_DIRNAME,
            "provenance": dict(loaded["provenance"]),
            "cannot_change_closed_label": True,
            "required_contract": contract,
        },
        "assets": assets,
    }


# ---------------------------------------------------------------------------
# bundle reload and self-verification
# ---------------------------------------------------------------------------


def load_bundle(output_dir: Path) -> dict[str, Any]:
    """Reload a written bundle from disk, using only its own manifest."""
    directory = Path(output_dir).expanduser().resolve()
    with (directory / MANIFEST_NAME).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema") != SCHEMA_ID:
        raise RuntimeError(f"unexpected bundle schema {manifest.get('schema')!r}")
    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise RuntimeError("unexpected bundle schema version")
    architecture = manifest["architecture"]
    real_config = InvariantPatchConfig(**dict(architecture["real"])).validate()
    complex_config = InvariantPatchConfig(
        **dict(architecture["complex"])
    ).validate()
    arrays = {
        name: np.load(directory / filename, allow_pickle=False)
        for name, filename in (
            ("real_center", "real_center.npy"),
            ("complex_center", "complex_center.npy"),
            ("feature_mean", "feature_mean.npy"),
            ("feature_std", "feature_std.npy"),
            ("prototypes", "fusion_prototypes.npy"),
        )
    }
    fusion = CenteredInvariantFusion(
        InvariantPatchCNN(real_config),
        InvariantPatchCNN(complex_config),
        arrays["real_center"],
        arrays["complex_center"],
        alpha_real=float(architecture["alpha_real"]),
        alpha_complex=float(architecture["alpha_complex"]),
        weight_real=float(architecture["weight_real"]),
        eps=float(architecture["eps"]),
    )
    state = torch.load(
        directory / "fusion_state_dict.pt",
        map_location="cpu",
        weights_only=True,
    )
    fusion.load_state_dict(state, strict=True)
    return {
        "manifest": manifest,
        "fusion": fusion.to(torch.device("cpu")).eval(),
        "prototypes": arrays["prototypes"].astype(np.float32, copy=False),
        "feature_mean": arrays["feature_mean"].astype(np.float32, copy=False),
        "feature_std": arrays["feature_std"].astype(np.float32, copy=False),
        "classes": list(manifest["classification"]["classes"]),
        "target_frac": float(manifest["frontend"]["target_frac"]),
        "real_config": real_config,
    }


def verify_bundle(
    output_dir: Path,
    expected: Mapping[str, Any],
    *,
    specifications: Sequence[Mapping[str, Any]] = PROBE_SPECIFICATIONS,
    tolerance: float = SELF_VERIFICATION_TOLERANCE,
) -> dict[str, Any]:
    """Reload the written bundle and reproduce the probe embeddings.

    Returns the per-quantity max absolute errors.  Raises if any exceeds the
    stated tolerance, so an unusable bundle can never be reported as exported.
    """
    reloaded = load_bundle(output_dir)
    config = reloaded["real_config"]
    actual = probe_embeddings(
        reloaded["fusion"],
        reloaded["feature_mean"],
        reloaded["feature_std"],
        patch_length=int(config.patch_length),
        patch_count=int(config.patch_count),
        n_features=int(config.n_features),
        target_frac=float(reloaded["target_frac"]),
        specifications=specifications,
    )
    errors: dict[str, float] = {}
    for key in (
        "packed",
        "raw_features",
        "standardized_features",
        "real_embedding",
        "complex_embedding",
        "fused_embedding",
    ):
        errors[key] = float(
            np.max(np.abs(np.asarray(actual[key]) - np.asarray(expected[key])))
        )
    _, expected_winner, _ = _closed_set(
        np.asarray(expected["fused_embedding"]),
        np.asarray(expected["prototypes"]),
        expected["classes"],
    )
    _, actual_winner, actual_labels = _closed_set(
        np.asarray(actual["fused_embedding"]),
        reloaded["prototypes"],
        reloaded["classes"],
    )
    label_agreement = bool(np.array_equal(expected_winner, actual_winner))
    worst = max(errors.values())
    if worst > tolerance or not label_agreement:
        raise AssertionError(
            f"reloaded bundle did not reproduce the probe set: errors={errors}, "
            f"closed_label_agreement={label_agreement}"
        )
    return {
        "max_abs_error": errors,
        "worst_max_abs_error": worst,
        "tolerance": float(tolerance),
        "closed_label_agreement": label_agreement,
        "reloaded_closed_labels": actual_labels,
        "probe_case_count": len(specifications),
        "reload_path_is_bundle_only": True,
    }


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def _validate_output_path(source_dir: Path, output_dir: Path) -> Path:
    output = reject_sealed_path(Path(output_dir), "output")
    source_root = (REPO / "src").resolve()
    if _is_below(output, source_root):
        raise ValueError(f"refusing to write a bundle below live src/: {output}")
    if _is_below(output, Path(source_dir).resolve()):
        raise ValueError("refusing to write the bundle inside the source assembly")
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f"bundle output is not a directory: {output}")
        contents = sorted(item.name for item in output.iterdir())
        if contents:
            raise FileExistsError(
                f"refusing to overwrite non-empty bundle directory {output} "
                f"(contains {', '.join(contents[:5])})"
            )
    return output


def export_bundle(
    source_dir: Path,
    output_dir: Path,
    *,
    rejector_dir: Path | None = None,
    specifications: Sequence[Mapping[str, Any]] = PROBE_SPECIFICATIONS,
    tolerance: float = SELF_VERIFICATION_TOLERANCE,
) -> dict[str, Any]:
    """Write the deterministic v3 runtime bundle and return its manifest."""
    validate_schema_identity()
    source = reject_sealed_path(Path(source_dir), "source")
    output = _validate_output_path(source, output_dir)
    loaded = load_fusion_artifact(source)

    config = loaded["real_config"]
    probe = probe_embeddings(
        loaded["fusion"],
        loaded["feature_mean"],
        loaded["feature_std"],
        patch_length=int(config.patch_length),
        patch_count=int(config.patch_count),
        n_features=int(config.n_features),
        target_frac=loaded["target_frac"],
        specifications=specifications,
    )
    distances, winner, labels = _closed_set(
        probe["fused_embedding"], loaded["prototypes"], loaded["classes"]
    )

    output.mkdir(parents=True, exist_ok=True)
    assets: dict[str, dict[str, Any]] = {}
    for key, name in COPIED_ASSETS:
        destination = output / name
        # Byte copy, not re-serialization: the bundle asset then carries the
        # same SHA-256 the assembly recorded, and copying cannot perturb a
        # single float.
        shutil.copyfile(loaded["paths"][key], destination)
        digest = _sha256(destination)
        if digest != loaded["metrics"]["artifacts"][f"{key}_sha256"]:
            raise RuntimeError(f"copied {name} does not match the source hash")
        assets[name] = {"sha256": digest, "bytes": int(destination.stat().st_size)}

    if rejector_dir is None:
        rejection = unset_rejection_slot()
    else:
        attached = _write_rejector(
            output,
            load_rejector_policy(
                Path(rejector_dir),
                dev_metrics_sha256=loaded["dev_metrics_sha256"],
                patch_length=int(config.patch_length),
                patch_count=int(config.patch_count),
            ),
        )
        rejection = attached["slot"]
        assets.update(attached["assets"])

    metrics = loaded["metrics"]
    packed_length = int(config.patch_length) * int(config.patch_count)
    manifest: dict[str, Any] = {
        "schema": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "kind": "v3-time-domain-centered-invariant-fusion",
        "status": "development_bundle_not_release",
        "development_only": True,
        "release_evidence": False,
        "sealed_release_data_used": 0,
        "consumed_test_rows_used": 0,
        "not_compatible_with_schema_ids": sorted(FORBIDDEN_SCHEMA_IDS),
        "frontend": {
            "version": td_preprocess.PREPROCESS_VERSION,
            "estimator_version": td_preprocess.ESTIMATOR_VERSION,
            "uses_frequency_transform": False,
            "patch_length": int(config.patch_length),
            "patch_count": int(config.patch_count),
            "packed_length": packed_length,
            "packed_dtype": "float32",
            "feature_count": int(config.n_features),
            "raw_feature_dtype": "float32",
            "target_frac": loaded["target_frac"],
            "module_metadata": td_preprocess.preprocess_metadata(),
            "source_sha256": loaded["bound_frontend_sha256"],
            "recorded_in_assembly": metrics["frontend"],
        },
        "architecture": metrics["architecture"],
        "parameter_count": int(metrics["parameter_count"]),
        "fusion": {
            "kind": "centered_invariant_fusion",
            "rule": metrics["fusion"]["rule"],
            "weight_real": loaded["weight_real"],
            "weight_complex": loaded["weight_complex"],
            "alpha_real": float(metrics["fusion"]["alpha_real"]),
            "alpha_complex": float(metrics["fusion"]["alpha_complex"]),
            "eps": float(metrics["fusion"]["eps"]),
            "state_dict_asset": "fusion_state_dict.pt",
        },
        "centers": {
            "fit_population": "training only",
            "real_asset": "real_center.npy",
            "complex_asset": "complex_center.npy",
            "embed_dim": int(config.embed_dim),
        },
        "feature_standardization": {
            "fit_population": "training raw features only",
            "epsilon": FEATURE_STANDARDIZATION_EPSILON,
            "rule": "(raw_float32 - mean) / std, cast to float32",
            "mean_asset": "feature_mean.npy",
            "std_asset": "feature_std.npy",
            "feature_count": int(config.n_features),
        },
        "classification": {
            "classes": loaded["classes"],
            "class_order_rule": (
                "sorted(corpus manifest classes); recovered from "
                "dev_metrics data_audit.class_counts without opening the corpus"
            ),
            "prototype_asset": "fusion_prototypes.npy",
            "prototype_population": "enrollment only",
            "embedding": "centered_fusion",
            "distance": "squared_euclidean",
            "label": "argmin over prototypes",
        },
        "rejection": rejection,
        "assets": assets,
        "probe_fixture": {
            "file": PROBE_NAME,
            "case_names": [str(item["name"]) for item in specifications],
            "case_count": len(specifications),
        },
        "provenance": {
            "source_fusion_artifact": _repo_relative(source),
            "source_dev_metrics_sha256": loaded["dev_metrics_sha256"],
            "source_branches": sanitized_branch_provenance(
                metrics.get("source_branches")
            ),
            "assembly_seed": int(metrics.get("seed", -1)),
            "assembly_source_sha256": metrics["source_sha256"],
            "fitting_contract": metrics["fitting_contract"],
            "closed_selection_balanced_accuracy": float(
                metrics["closed_selection"]["balanced_accuracy"]
            ),
            "exporter": Path(__file__).name,
            "nothing_fitted_here": True,
            "corpus_or_split_loaded": False,
            "sealed_or_consumed_rows_loaded": 0,
        },
        "determinism": {
            "timestamps_recorded": False,
            "npz_containers_used": False,
            "json_sort_keys": True,
            "array_format": "npy",
            "note": (
                "a .npz zip container records member modification times, which "
                "would make the bundle SHA-256 unstable across identical exports"
            ),
        },
        "release_blockers": [
            "open-set rejector is not refit against these v3 embeddings",
            "N4096 clean accuracy and worst five-shot balanced are unmeasured",
            "TypeScript encoder/fusion/prototype/rejection parity is unported",
            "no untouched release seed has been spent on a sealed suite",
        ],
    }

    _write_json(output / MANIFEST_NAME, manifest)
    verification = verify_bundle(
        output,
        {
            **probe,
            "prototypes": loaded["prototypes"],
            "classes": loaded["classes"],
        },
        specifications=specifications,
        tolerance=tolerance,
    )

    fixture = {
        "schema": f"{SCHEMA_ID}.probe-fixture",
        "schema_version": SCHEMA_VERSION,
        "tolerance": float(tolerance),
        "self_verification": verification,
        "frontend_source_sha256": loaded["bound_frontend_sha256"],
        "data_or_corpus_loaded": False,
        "cases": [
            {
                "name": str(specification["name"]),
                "length": int(specification["length"]),
                "physical_scale": float(specification["physical_scale"]),
                "kind": str(specification["kind"]),
                "raw": {
                    "in_phase": probe["captures"][index].real.tolist(),
                    "quadrature": probe["captures"][index].imag.tolist(),
                },
                "expected": {
                    "context": probe["contexts"][index],
                    "packed_iq": probe["packed"][index].reshape(-1).tolist(),
                    "raw_features": probe["raw_features"][index].tolist(),
                    "standardized_features": probe["standardized_features"][
                        index
                    ].tolist(),
                    "real_embedding": probe["real_embedding"][index].tolist(),
                    "complex_embedding": probe["complex_embedding"][
                        index
                    ].tolist(),
                    "fused_embedding": probe["fused_embedding"][index].tolist(),
                    "squared_prototype_distances": distances[index].tolist(),
                    "closed_winner_index": int(winner[index]),
                    "closed_label": labels[index],
                },
            }
            for index, specification in enumerate(specifications)
        ],
    }
    fixture_path = output / PROBE_NAME
    _write_json(fixture_path, fixture)

    manifest["assets"][PROBE_NAME] = {
        "sha256": _sha256(fixture_path),
        "bytes": int(fixture_path.stat().st_size),
    }
    manifest["probe_fixture"]["sha256"] = _sha256(fixture_path)
    manifest["self_verification"] = verification
    _write_json(output / MANIFEST_NAME, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source-dir",
        required=True,
        help="a v3 fusion directory written by assemble_v3_fusion.py",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="empty or absent directory to receive the runtime bundle",
    )
    parser.add_argument(
        "--rejector-dir",
        default=None,
        help=(
            "optional directory of frozen-policy .npy files plus provenance.json; "
            "omit until the rejector is refit against these v3 embeddings"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = export_bundle(
        Path(args.source_dir),
        Path(args.output_dir),
        rejector_dir=None if args.rejector_dir is None else Path(args.rejector_dir),
    )
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "status": manifest["status"],
                "output_dir": str(Path(args.output_dir).expanduser().resolve()),
                "classes": manifest["classification"]["classes"],
                "rejection_state": manifest["rejection"]["state"],
                "self_verification": manifest["self_verification"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
