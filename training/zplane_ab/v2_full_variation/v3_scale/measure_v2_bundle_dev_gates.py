"""Score the frozen v2 runtime bundle through the v3 dev closed-set gate protocol.

Why this exists (HANDOFF section 21.3)
--------------------------------------

``measure_v3_remaining_gates.py`` scored the v3 fusion at ~0.756 worst five-shot
balanced on the development protocol.  v2's sealed number is 0.8495, but it was
produced by the release evaluator on the sealed suite: support and query drawn
from the same sealed corpus.  The development protocol cannot do that (five-shot
support *is* a prototype fit and the selection population is scored, never fit),
so its primary variant draws support from enrollment against selection queries,
a strictly harder cross-population task.  The two numbers therefore measure
different things, and section 18's "regression" claim was withdrawn in 21.2.

This module produces the missing control: the frozen v2 runtime bundle
(``invariant_fusion_runtime_bundle_v2_seed20260727``) scored through the EXACT
development protocol the v3 number came from.

* If v2 also lands near 0.75 on dev, there is no v3 five-shot regression; the
  0.85 bar is a sealed-population figure that does not transfer to dev.
* If v2 lands near 0.85 on dev, the regression is real and v3-specific.

Either way this is strictly more informative than lowering the gate or spending
release seed 20260731, both of which remain forbidden.


Comparability by construction
-----------------------------

Everything protocol-shaped is imported from ``measure_v3_remaining_gates`` (and
through it from ``evaluate_invariant_release_suite``) rather than re-implemented:

* ``eligible_population`` -- the all-zero-N4096-prefix length-eligibility rule;
* ``dev_row_identity_fields`` / ``dev_five_shot_split`` -- the digest support draw;
* ``closed_clean_report`` / ``five_shot_variant`` -- the scoring functions;
* ``assemble_gates`` -- gate assembly at the release ``GATE_FLOORS``;
* ``DEV_LENGTHS`` and the matched capture length.

The one legitimate difference is the model being scored, and with it the
frontend: the v2 bundle is valid only over ``invariant_patch_preprocess``
(invariant-patch-v1 with hybrid-v3 FFT band estimation), while the v3
measurement uses the FFT-free time-domain frontend.  That difference and every
other one are stated machine-readably in the report's ``comparability`` block.

**The eligibility trap, handled explicitly.**  The v2 and v3 frontends would
not agree on which rows are "length eligible" if each derived eligibility from
its own preprocessing.  Eligibility is therefore computed ONCE, by the same
``eligible_population`` function object the v3 measurement calls, over the same
cached corpus index sets (``invariant_patch_data`` splits are model-seed
independent), on the raw captures BEFORE any frontend runs.  The resulting
index-set SHA-256 digests are recorded, and when ``--v3-report`` points at an
existing v3 ``remaining_gates.json`` the digests are asserted equal to the ones
that report recorded, failing closed on any mismatch.  With ``--v3-report`` the
five-shot split seed also defaults to the v3 report's, which makes the support
draws identical row for row; that identity is likewise asserted via the
recorded ``support_positions_sha256``.

Deviations this module adds on top of the v3 measurement's own documented
deviations from the sealed protocol:

1. **Frontend.**  Rows are preprocessed with ``invariant_patch_preprocess``
   (FFT-bearing hybrid-v3 band estimation), the only frontend the v2 bundle is
   valid over.  Fused embeddings are 64-dimensional (two centered, unit-
   normalized 32-d branches) rather than the v3 fusion's dimensionality.
2. **Frozen assets.**  Prototypes (clean gate), branch centers, and the
   12-feature standardization are loaded verbatim from ``runtime_bundle.pt``,
   whose SHA-256 must equal the recorded
   ``b94ac51b15c1813c…`` before any tensor is read.  Nothing is refit.
3. **Reproduction check.**  The v2 bundle records no per-length dev audit, so
   the v3 script's length-audit replay is replaced by three fail-closed checks:
   the pinned bundle hash; a replay of the bundle's centers and prototypes from
   the cached training/enrollment arrays (inference only, tolerance 2e-6, the
   same bound ``run_v3_openset_replication`` used); and agreement between
   raw-derived N16384 embeddings and cache-derived embeddings on the eligible
   selection rows, which proves the raw-prefix path equals the cache path the
   bundle was fit against.

This is development evidence, not release evidence.  No sealed or release path
is read or written, the consumed historical test half is never loaded, and the
report records ``sealed_release_data_used: 0`` and ``consumed_test_rows_used:
0``.  Release seed 20260731 and consumed seed 20260729 are refused wherever a
seed can be supplied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
V2 = HERE.parent
ZPAB = V2.parent
TRAINING = ZPAB.parent
for _path in (TRAINING, ZPAB, V2, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import assemble_v3_fusion as assemble  # noqa: E402
import evaluate_invariant_release_suite as release  # noqa: E402
import invariant_patch_data as invariant_data  # noqa: E402
import invariant_patch_preprocess as v2_preprocess  # noqa: E402
import measure_v3_remaining_gates as gates  # noqa: E402
import preprocess as pp  # noqa: E402
import run_invariant_cnn_dev as runner  # noqa: E402
import run_time_domain_dev as dev  # noqa: E402
from assemble_invariant_candidate import (  # noqa: E402
    ALPHA_COMPLEX,
    ALPHA_REAL,
    WEIGHT_REAL,
    _embed_all,
    _fuse_numpy,
)
from invariant_patch_cnn import (  # noqa: E402
    InvariantPatchCNN,
    InvariantPatchConfig,
)
from train import prototypes_from  # noqa: E402


MEASUREMENT_VERSION = "v2-bundle-dev-closed-gates-v1"
REPORT_NAME = gates.REPORT_NAME  # remaining_gates.json, same schema

V2_BUNDLE_SEED = 20260727
V2_BUNDLE_KIND = "invariant_centered_fusion_classifier"
V2_BUNDLE_SCHEMA = 2
V2_FRONTEND = "invariant-patch-v1"

# The frozen v2 runtime bundle this module exists to score.  The hash is the
# one recorded in the bundle's own manifest and in HANDOFF; it is verified
# before a single tensor is deserialized.
EXPECTED_BUNDLE_SHA256 = (
    "b94ac51b15c1813ca0f94bf3cc3b28ee5f903206b2885ec14dbaa8fddca031e3"
)
DEFAULT_BUNDLE_DIR = (
    V2
    / "artifacts"
    / "invariant_patch"
    / "invariant_fusion_runtime_bundle_v2_seed20260727"
)
DEFAULT_OUTPUT_DIR = (
    V2
    / "artifacts"
    / "invariant_patch"
    / "v3_scale"
    / "v2_bundle_remaining_gates_seed20260727"
)

# Replay of frozen centers/prototypes from the cached arrays.  Same bound
# run_v3_openset_replication.py used for the identical replay.
DEFAULT_REPLAY_TOLERANCE = 2e-6
# Raw-prefix-at-N16384 embeddings vs cache-derived embeddings.  Loose enough
# for device reduction order, far below any decision margin.
DEFAULT_FRONTEND_TOLERANCE = 2e-3
# Bundle feature standardization vs the cache's; these were fit identically.
FEATURE_STANDARDIZATION_TOLERANCE = 1e-6

# Protocol pieces reused from the v3 measurement BY OBJECT IDENTITY, so the two
# reports cannot drift apart.  Tests assert these are the same functions.
DEV_LENGTHS = gates.DEV_LENGTHS
UNMEASURABLE_RELEASE_LENGTHS = gates.UNMEASURABLE_RELEASE_LENGTHS
FIVE_SHOT_VARIANTS = gates.FIVE_SHOT_VARIANTS
PRIMARY_FIVE_SHOT_VARIANT = gates.PRIMARY_FIVE_SHOT_VARIANT
validate_split_seed = gates.validate_split_seed
validate_tolerance = gates.validate_tolerance
dev_row_identity_fields = gates.dev_row_identity_fields
dev_five_shot_split = gates.dev_five_shot_split
eligible_population = gates.eligible_population
closed_clean_report = gates.closed_clean_report
five_shot_variant = gates.five_shot_variant
assemble_gates = gates.assemble_gates


# ---------------------------------------------------------------------------
# loading and validating the frozen v2 bundle
# ---------------------------------------------------------------------------


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_bundle_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless the payload is the frozen v2 fusion bundle shape.

    Returns the validated, numpy-typed assets.  Nothing here fits, adjusts, or
    recomputes; every returned array is the bundle's own frozen value.
    """
    required = (
        "schema",
        "kind",
        "development_only",
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
        "provenance",
    )
    missing = sorted(name for name in required if name not in payload)
    if missing:
        raise ValueError(f"runtime bundle is missing keys: {missing}")
    if int(payload["schema"]) != V2_BUNDLE_SCHEMA:
        raise ValueError(
            f"expected runtime bundle schema {V2_BUNDLE_SCHEMA}, "
            f"got {payload['schema']!r}"
        )
    if payload["kind"] != V2_BUNDLE_KIND:
        raise ValueError(f"unexpected bundle kind {payload['kind']!r}")
    if payload["development_only"] is not True:
        raise RuntimeError("the v2 bundle must declare development_only")

    provenance = payload["provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError("bundle provenance must be a mapping")
    if provenance.get("release_evidence") is not False:
        raise RuntimeError("the v2 bundle must not claim release evidence")
    if provenance.get("consumed_test_rows_used") != 0:
        raise RuntimeError("the v2 bundle provenance consumed test rows")
    if provenance.get("center_fit_population") != "training only":
        raise RuntimeError("v2 fusion centers must be fit on training only")
    if provenance.get("prototype_population") != "enrollment only":
        raise RuntimeError("v2 prototypes must be fit on enrollment only")
    contract = provenance.get("cache_contract")
    if not isinstance(contract, Mapping) or contract.get("frontend") != V2_FRONTEND:
        raise RuntimeError(
            f"the v2 bundle frontend contract is not {V2_FRONTEND!r}"
        )

    # ``_fuse_numpy`` carries the fusion constants at module level; the bundle
    # must agree with them exactly or the fused geometry is not the bundle's.
    for name, expected in (
        ("alpha_real", ALPHA_REAL),
        ("alpha_complex", ALPHA_COMPLEX),
        ("weight_real", WEIGHT_REAL),
    ):
        recorded = float(payload[name])
        if not math.isclose(recorded, expected, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(
                f"bundle {name} {recorded!r} disagrees with the assembly "
                f"constant {expected!r}; _fuse_numpy would not reproduce it"
            )
    eps = float(payload["eps"])
    if not math.isclose(eps, float(np.float32(1e-12)), rel_tol=0.0, abs_tol=1e-18):
        raise RuntimeError(
            f"bundle eps {eps!r} disagrees with _safe_unit's frozen 1e-12"
        )

    real_config = InvariantPatchConfig(**dict(payload["real_config"])).validate()
    complex_config = InvariantPatchConfig(
        **dict(payload["complex_config"])
    ).validate()
    if real_config.encoder != "real" or complex_config.encoder != "complex":
        raise ValueError("bundle branch configs carry the wrong encoders")
    if real_config.embed_dim != complex_config.embed_dim:
        raise ValueError("bundle branch embed dims disagree")
    embed_dim = int(real_config.embed_dim)
    if real_config.n_features != complex_config.n_features:
        raise ValueError("bundle branch feature widths disagree")
    n_features = int(real_config.n_features)

    classes = [str(name) for name in payload["classes"]]
    if len(classes) != len(set(classes)) or not classes:
        raise ValueError("bundle classes must be unique and non-empty")

    def _array(name: str, shape: tuple[int, ...]) -> np.ndarray:
        value = np.asarray(payload[name], dtype=np.float32)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(
                f"bundle {name} must be finite with shape {shape}, "
                f"got {value.shape}"
            )
        return value

    feature_std = _array("feature_std", (n_features,))
    if not (feature_std > 0.0).all():
        raise ValueError("bundle feature_std must be strictly positive")

    return {
        "real_config": real_config,
        "complex_config": complex_config,
        "embed_dim": embed_dim,
        "n_features": n_features,
        "classes": classes,
        "real_center": _array("real_center", (embed_dim,)),
        "complex_center": _array("complex_center", (embed_dim,)),
        "prototypes": _array("prototypes", (len(classes), 2 * embed_dim)),
        "feature_mean": _array("feature_mean", (n_features,)),
        "feature_std": feature_std,
        "weight_real": float(payload["weight_real"]),
        "cache_contract": json.loads(json.dumps(contract)),
        "provenance": json.loads(
            json.dumps(provenance, default=str)
        ),
    }


def load_v2_bundle(
    directory: Path,
    *,
    expected_sha256: str = EXPECTED_BUNDLE_SHA256,
) -> dict[str, Any]:
    """Load ``runtime_bundle.pt`` read-only, hash-verified, and rebuilt.

    The bundle file's SHA-256 must equal ``expected_sha256`` BEFORE any tensor
    is deserialized.  The branch networks are rebuilt from the recorded configs
    and state dicts and set to eval; nothing in the bundle directory is written.
    """
    source = assemble.reject_sealed_path(Path(directory), "v2 bundle")
    bundle_path = source / "runtime_bundle.pt"
    if not bundle_path.is_file():
        raise FileNotFoundError(bundle_path)
    digest = _file_sha256(bundle_path)
    if digest != expected_sha256:
        raise RuntimeError(
            f"{bundle_path} SHA-256 {digest} does not match the recorded "
            f"{expected_sha256}; refusing to load an unrecognized bundle"
        )
    try:
        payload = torch.load(bundle_path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with old PyTorch
        payload = torch.load(bundle_path, map_location="cpu")
    validated = validate_bundle_payload(payload)

    real = InvariantPatchCNN(validated["real_config"])
    real.load_state_dict(payload["real_state_dict"], strict=True)
    complex_model = InvariantPatchCNN(validated["complex_config"])
    complex_model.load_state_dict(payload["complex_state_dict"], strict=True)
    parameter_count = sum(
        int(p.numel()) for p in real.parameters()
    ) + sum(int(p.numel()) for p in complex_model.parameters())

    return {
        "directory": source,
        "path": bundle_path,
        "sha256": {"runtime_bundle.pt": digest},
        "real": real.eval(),
        "complex": complex_model.eval(),
        "parameter_count": parameter_count,
        **{
            key: validated[key]
            for key in (
                "embed_dim",
                "n_features",
                "classes",
                "real_center",
                "complex_center",
                "prototypes",
                "feature_mean",
                "feature_std",
                "weight_real",
                "cache_contract",
                "provenance",
            )
        },
    }


# ---------------------------------------------------------------------------
# v2 embedding of raw capture prefixes
# ---------------------------------------------------------------------------


def standardize_features(
    raw_features: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
) -> np.ndarray:
    """Standardize raw iq features with the bundle's frozen mean/std."""
    value = np.asarray(raw_features, dtype=np.float32)
    mean = np.asarray(feature_mean, dtype=np.float32)
    std = np.asarray(feature_std, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != mean.shape[0]:
        raise ValueError(
            f"raw features must be [n, {mean.shape[0]}], got {value.shape}"
        )
    if mean.shape != std.shape:
        raise ValueError("feature mean and std disagree in shape")
    result = ((value - mean) / std).astype(np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("standardized features are non-finite")
    return result


def v2_embed_prefixes(
    bundle: Mapping[str, Any],
    captures: Sequence[np.ndarray],
    length: int,
    device: torch.device,
    *,
    batch: int = 256,
) -> np.ndarray:
    """Embed exact causal prefixes through the frozen v2 stack.

    Raw prefix -> ``invariant_patch_preprocess.preprocess`` (hybrid-v3, the v2
    frontend) -> bundle feature standardization -> both frozen branches ->
    ``_fuse_numpy`` with the bundle's frozen centers.  This is exactly the
    inference path ``run_v3_openset_replication`` replayed to zero error
    against the bundle's own recorded assets.
    """
    if length <= 0:
        raise ValueError("prefix length must be positive")
    contract_config = bundle["cache_contract"]["config"]
    packed_rows: list[np.ndarray] = []
    feature_rows: list[np.ndarray] = []
    for capture in captures:
        prefix = np.asarray(capture)[:length]
        if len(prefix) != length:
            raise ValueError("a capture is shorter than the requested prefix")
        packed, raw_features, _context = v2_preprocess.preprocess(
            prefix,
            patch_length=int(contract_config["patch_length"]),
            patch_count=int(contract_config["patch_count"]),
            target_frac=float(contract_config["target_frac"]),
        )
        packed_rows.append(packed)
        feature_rows.append(raw_features)
    packed_all = np.stack(packed_rows).astype(np.float32)
    features = standardize_features(
        np.stack(feature_rows),
        bundle["feature_mean"],
        bundle["feature_std"],
    )
    real_embedding = _embed_all(
        bundle["real"], packed_all, features, device, batch=batch
    )
    complex_embedding = _embed_all(
        bundle["complex"], packed_all, features, device, batch=batch
    )
    fused = _fuse_numpy(
        real_embedding,
        complex_embedding,
        bundle["real_center"],
        bundle["complex_center"],
    )
    if not np.isfinite(fused).all():
        raise RuntimeError(f"v2 fusion produced non-finite n{length} embeddings")
    return fused.astype(np.float32, copy=False)


def v2_embed_cached(
    bundle: Mapping[str, Any],
    packed: np.ndarray,
    features: np.ndarray,
    device: torch.device,
    *,
    batch: int = 256,
) -> np.ndarray:
    """Embed already-preprocessed cache arrays through the frozen v2 stack."""
    real_embedding = _embed_all(bundle["real"], packed, features, device, batch=batch)
    complex_embedding = _embed_all(
        bundle["complex"], packed, features, device, batch=batch
    )
    return _fuse_numpy(
        real_embedding,
        complex_embedding,
        bundle["real_center"],
        bundle["complex_center"],
    ).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# fail-closed reproduction checks
# ---------------------------------------------------------------------------


def replay_compare(
    name: str,
    recomputed: np.ndarray,
    recorded: np.ndarray,
    tolerance: float,
) -> float:
    """Return the max abs deviation; fail closed above ``tolerance``."""
    delta = float(
        np.max(
            np.abs(
                np.asarray(recomputed, dtype=np.float64)
                - np.asarray(recorded, dtype=np.float64)
            )
        )
    )
    if not np.isfinite(delta) or delta > tolerance:
        raise RuntimeError(
            f"v2 bundle replay mismatch for {name}: max abs delta {delta:g} "
            f"exceeds {tolerance:g}; the rebuilt model is not the recorded one"
        )
    return delta


def bundle_replay_check(
    bundle: Mapping[str, Any],
    data: Mapping[str, Any],
    device: torch.device,
    *,
    tolerance: float = DEFAULT_REPLAY_TOLERANCE,
) -> dict[str, Any]:
    """Re-derive the bundle's frozen centers and prototypes from the caches.

    Inference only: training rows are embedded to recompute the branch centers,
    enrollment rows to recompute the fused class prototypes, and both must
    match the bundle's recorded arrays within ``tolerance``.  Nothing is fit or
    stored; a failure aborts the measurement.
    """
    real_train = _embed_all(
        bundle["real"], data["xtr"], data["ftr"], device, batch=256
    )
    complex_train = _embed_all(
        bundle["complex"], data["xtr"], data["ftr"], device, batch=256
    )
    deltas = {
        "real_center": replay_compare(
            "real_center",
            real_train.mean(axis=0).astype(np.float32),
            bundle["real_center"],
            tolerance,
        ),
        "complex_center": replay_compare(
            "complex_center",
            complex_train.mean(axis=0).astype(np.float32),
            bundle["complex_center"],
            tolerance,
        ),
    }
    fused_enrollment = v2_embed_cached(
        bundle, data["xen"], data["fen"], device
    )
    prototypes = prototypes_from(
        fused_enrollment,
        np.asarray(data["yen"], dtype=np.int64),
        len(bundle["classes"]),
    )
    deltas["prototypes"] = replay_compare(
        "prototypes", prototypes, bundle["prototypes"], tolerance
    )
    return {
        "tolerance": float(tolerance),
        "max_abs_deltas": deltas,
        "worst_abs_delta": float(max(deltas.values())),
        "training_rows_embedded": int(len(real_train)),
        "enrollment_rows_embedded": int(len(fused_enrollment)),
        "fit_performed": False,
        "role": (
            "proves the rebuilt branches, centers, and prototypes are the "
            "bundle's own recorded ones before any gate is read"
        ),
    }


def frontend_consistency_check(
    fused_from_raw: np.ndarray,
    fused_from_cache: np.ndarray,
    *,
    tolerance: float = DEFAULT_FRONTEND_TOLERANCE,
) -> dict[str, Any]:
    """Raw N16384 prefixes and the build cache must embed identically.

    The corpus stores exactly 16384 samples per row, so the raw prefix at the
    matched length is the very row the cache preprocessed when the bundle was
    fit.  Agreement here proves the raw-prefix inference path used for all
    three lengths equals the frozen cache path, up to device reduction order.
    """
    raw = np.asarray(fused_from_raw, dtype=np.float64)
    cached = np.asarray(fused_from_cache, dtype=np.float64)
    if raw.shape != cached.shape:
        raise ValueError(
            f"raw and cache embeddings disagree in shape: {raw.shape} "
            f"vs {cached.shape}"
        )
    delta = float(np.max(np.abs(raw - cached)))
    if not np.isfinite(delta) or delta > tolerance:
        raise RuntimeError(
            f"raw-prefix N16384 embeddings deviate from the cache path by "
            f"{delta:g} > {tolerance:g}; the v2 frontend replay is not the "
            "one the bundle was fit against"
        )
    return {
        "tolerance": float(tolerance),
        "max_abs_delta": delta,
        "rows_compared": int(len(raw)),
        "role": (
            "proves the raw-prefix v2 preprocessing path equals the cache "
            "path the bundle was fit against, at the matched capture length"
        ),
    }


def selection_positions(
    corpus_indices: np.ndarray,
    eligible_indices: np.ndarray,
) -> np.ndarray:
    """Positions of the eligible corpus rows inside the cached array order."""
    order = {
        int(index): position
        for position, index in enumerate(np.asarray(corpus_indices, np.int64))
    }
    positions = []
    for index in np.asarray(eligible_indices, dtype=np.int64):
        if int(index) not in order:
            raise ValueError(
                f"eligible corpus index {int(index)} is not in the cached "
                "population; the populations have drifted"
            )
        positions.append(order[int(index)])
    return np.asarray(positions, dtype=np.int64)


# ---------------------------------------------------------------------------
# the optional v3 cross-check and split-seed resolution
# ---------------------------------------------------------------------------


def load_v3_report(path: Path) -> dict[str, Any]:
    """Load a v3 ``remaining_gates.json`` for the comparability cross-check."""
    resolved = assemble.reject_sealed_path(Path(path), "v3 report")
    with resolved.open(encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("measurement") != gates.MEASUREMENT_VERSION:
        raise ValueError(
            f"{resolved} is not a {gates.MEASUREMENT_VERSION} report"
        )
    for key, expected in (
        ("development_only", True),
        ("sealed_release_data_used", 0),
        ("consumed_test_rows_used", 0),
    ):
        if report.get(key) != expected:
            raise RuntimeError(
                f"{resolved} declares {key}={report.get(key)!r}; refusing it"
            )
    report["_path"] = str(resolved)
    return report


def resolve_split_seed(
    explicit: int | None,
    v3_report: Mapping[str, Any] | None,
) -> tuple[int, str]:
    """Explicit seed wins; then the v3 report's; then the bundle model seed."""
    if explicit is not None:
        return validate_split_seed(explicit), "explicit --split-seed"
    if v3_report is not None:
        return (
            validate_split_seed(int(v3_report["protocol"]["split_seed"])),
            "adopted from --v3-report (identical support draw)",
        )
    return validate_split_seed(V2_BUNDLE_SEED), "v2 bundle model seed"


def comparability_block(
    *,
    bundle: Mapping[str, Any],
    selection_sha256: str,
    enrollment_sha256: str,
    split_seed: int,
    split_seed_source: str,
    five_shot_by_variant: Mapping[str, Mapping[str, Any]],
    v3_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """State exactly what is shared with the v3 measurement and what differs.

    When a v3 report is supplied, the shared elements are not merely claimed:
    the eligible index-set digests must match it, and, when the split seed
    matches too, the per-variant support draws must match row for row.  Any
    mismatch fails closed, because a mismatched population makes the whole
    comparison meaningless.
    """
    shared = {
        "eligibility_rule": (
            "measure_v3_remaining_gates.eligible_population, the "
            "run_time_domain_dev all-zero-N4096-prefix rule, applied to the "
            "raw captures BEFORE either frontend runs, over the identical "
            "cached corpus index sets (invariant_patch_data splits are "
            "model-seed independent)"
        ),
        "eligible_selection_indices_sha256": selection_sha256,
        "eligible_enrollment_indices_sha256": enrollment_sha256,
        "scoring_functions": (
            "closed_clean_report / five_shot_variant / assemble_gates imported "
            "from measure_v3_remaining_gates, which imports _nearest, "
            "_classification_subset, _five_shot_report, _gate, and GATE_FLOORS "
            "from evaluate_invariant_release_suite"
        ),
        "five_shot_split_rule": (
            "measure_v3_remaining_gates.dev_five_shot_split: release digest "
            "rule over the same 14 identity fields and SPLIT_SALT"
        ),
        "five_shot_variants": list(FIVE_SHOT_VARIANTS),
        "measured_capture_lengths": list(DEV_LENGTHS),
        "matched_capture_length": int(release.MATCHED_CAPTURE_LENGTH),
        "gate_floors": dict(release.GATE_FLOORS),
        "classes": list(bundle["classes"]),
    }
    differs = {
        "frontend": {
            "this_measurement": {
                "version": v2_preprocess.PREPROCESS_VERSION,
                "estimator_version": v2_preprocess.ESTIMATOR_VERSION,
                "uses_frequency_transform": True,
                "reason": "the only frontend the frozen v2 bundle is valid over",
            },
            "v3_measurement": {
                "version": "time-domain v3 frontend",
                "uses_frequency_transform": False,
            },
        },
        "fused_embedding_dim": {
            "this_measurement": 2 * int(bundle["embed_dim"]),
            "v3_measurement": "the v3 fusion's own dimensionality (see its artifact)",
        },
        "frozen_assets_source": (
            "prototypes (clean gate), branch centers, and feature "
            "standardization come verbatim from runtime_bundle.pt "
            f"(sha256 {bundle['sha256']['runtime_bundle.pt']})"
        ),
        "model_seed": {
            "this_measurement": V2_BUNDLE_SEED,
            "v3_measurement": (
                int(v3_report["seed"]) if v3_report is not None else "per artifact"
            ),
        },
        "reproduction_check": (
            "the v2 bundle records no per-length dev audit, so the v3 length-"
            "audit replay is replaced by the pinned bundle hash, a center/"
            "prototype replay from the cached arrays, and raw-vs-cache "
            "embedding agreement at the matched length"
        ),
    }
    block: dict[str, Any] = {
        "role": (
            "settles HANDOFF 21.3: same dev protocol, same populations, same "
            "scoring; only the scored model (and with it the frontend) differs"
        ),
        "shared_with_v3_measurement": shared,
        "differs_from_v3_measurement": differs,
        "split_seed": int(split_seed),
        "split_seed_source": split_seed_source,
    }
    if v3_report is None:
        block["v3_report_cross_check"] = {
            "performed": False,
            "note": (
                "no --v3-report supplied; the index-set digests above can be "
                "compared to any v3 remaining_gates.json population block "
                "after the fact"
            ),
        }
        return block

    v3_population = v3_report["population"]
    mismatches: list[str] = []
    if v3_population["selection"]["eligible_indices_sha256"] != selection_sha256:
        mismatches.append("selection eligible index set")
    if v3_population["enrollment"]["eligible_indices_sha256"] != enrollment_sha256:
        mismatches.append("enrollment eligible index set")
    if mismatches:
        raise RuntimeError(
            "eligible populations do not match the v3 report "
            f"({', '.join(mismatches)}); the measurements would not be "
            "comparable and nothing should be quoted from this run"
        )

    v3_split_seed = int(v3_report["protocol"]["split_seed"])
    support_check: dict[str, Any] = {
        "v3_split_seed": v3_split_seed,
        "split_seed_matches": v3_split_seed == int(split_seed),
    }
    if support_check["split_seed_matches"]:
        for name in FIVE_SHOT_VARIANTS:
            mine = five_shot_by_variant[name]["split"]["support_positions_sha256"]
            theirs = v3_report["five_shot_per_variant"][name]["split"][
                "support_positions_sha256"
            ]
            if mine != theirs:
                raise RuntimeError(
                    f"five-shot {name} support draw does not match the v3 "
                    "report despite an identical split seed and population; "
                    "the protocol has drifted"
                )
            support_check[f"{name}_support_positions_sha256"] = mine
        support_check["support_draws_identical"] = True
    else:
        support_check["support_draws_identical"] = False
        support_check["note"] = (
            "split seeds differ, so the support draws differ by construction; "
            "v3 measured both its seeds within 0.002 of each other on this "
            "statistic"
        )
    block["v3_report_cross_check"] = {
        "performed": True,
        "v3_report_path": v3_report["_path"],
        "eligible_index_sets_match": True,
        "support_draw": support_check,
        "v3_gates_for_reference": json.loads(json.dumps(v3_report["gates"])),
    }
    return block


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


def _source_hashes() -> dict[str, str]:
    paths = {
        "measure_v2_bundle_dev_gates.py": Path(__file__).resolve(),
        "measure_v3_remaining_gates.py": HERE / "measure_v3_remaining_gates.py",
        "evaluate_invariant_release_suite.py": (
            V2 / "evaluate_invariant_release_suite.py"
        ),
        "assemble_invariant_candidate.py": V2 / "assemble_invariant_candidate.py",
        "invariant_patch_cnn.py": V2 / "invariant_patch_cnn.py",
        "invariant_patch_data.py": V2 / "invariant_patch_data.py",
        "invariant_patch_preprocess.py": TRAINING / "invariant_patch_preprocess.py",
        "preprocess.py": TRAINING / "preprocess.py",
        "train.py": TRAINING / "train.py",
    }
    return {name: dev._sha256(path) for name, path in paths.items()}


def v2_frontend_metadata(contract_config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "version": v2_preprocess.PREPROCESS_VERSION,
        "estimator_version": v2_preprocess.ESTIMATOR_VERSION,
        "uses_frequency_transform": True,
        "source": "training/invariant_patch_preprocess.py",
        "valid_for": "the frozen v2 runtime bundle only",
        "config": {
            "patch_length": int(contract_config["patch_length"]),
            "patch_count": int(contract_config["patch_count"]),
            "target_frac": float(contract_config["target_frac"]),
            "nfft": int(pp.NFFT),
        },
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    bundle_dir = assemble.reject_sealed_path(Path(args.bundle_dir), "v2 bundle")
    output_dir = assemble.reject_sealed_path(Path(args.output_dir), "output")
    replay_tolerance = validate_tolerance(args.replay_tolerance)
    frontend_tolerance = validate_tolerance(args.frontend_tolerance)

    report_path = output_dir / REPORT_NAME
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path}")

    v3_report = (
        load_v3_report(Path(args.v3_report)) if args.v3_report else None
    )
    split_seed, split_seed_source = resolve_split_seed(
        args.split_seed, v3_report
    )

    bundle = load_v2_bundle(bundle_dir)
    contract = bundle["cache_contract"]
    contract_config = contract["config"]

    source = invariant_data.load(
        patch_length=int(contract_config["patch_length"]),
        patch_count=int(contract_config["patch_count"]),
        target_frac=float(contract_config["target_frac"]),
        model_seed=V2_BUNDLE_SEED,
        build_if_missing=False,
    )
    if source["data_audit"].get("consumed_test_rows_exposed") != 0:
        raise AssertionError("data loader exposed consumed test rows")
    forbidden = [key for key in source if "test" in key.lower()]
    if forbidden:
        raise AssertionError(f"data loader exposed test-like keys: {forbidden}")
    if source["cache_contract"] != contract:
        raise RuntimeError(
            "current cache contract differs from the v2 bundle's frozen one"
        )
    classes = list(source["classes"])
    if classes != list(bundle["classes"]):
        raise RuntimeError(
            f"bundle classes {bundle['classes']} disagree with the cache "
            f"classes {classes}"
        )

    standardization_delta = float(
        max(
            np.max(np.abs(bundle["feature_mean"] - np.asarray(source["fmean"]))),
            np.max(np.abs(bundle["feature_std"] - np.asarray(source["fstd"]))),
        )
    )
    if standardization_delta > FEATURE_STANDARDIZATION_TOLERANCE:
        raise RuntimeError(
            "bundle feature standardization deviates from the cache's by "
            f"{standardization_delta:g}; the frozen frontend is not the "
            "cache's frontend"
        )

    manifest = dev._load_manifest()
    identity_fields = dev_row_identity_fields(manifest["items"])
    device = runner.resolve_device(args.device)
    runner.seed_everything(V2_BUNDLE_SEED)

    # Eligibility is computed ONCE, by the same function object the v3
    # measurement calls, on raw captures, before the v2 frontend ever runs.
    raw = dev._raw_memmap(manifest)
    selection = eligible_population(
        manifest, np.asarray(source["va_idx"], dtype=np.int64), classes, raw
    )
    enrollment = eligible_population(
        manifest, np.asarray(source["en_idx"], dtype=np.int64), classes, raw
    )
    del raw

    replay = bundle_replay_check(
        bundle, source, device, tolerance=replay_tolerance
    )
    print(
        f"[v2 gates] bundle replay ok: worst delta "
        f"{replay['worst_abs_delta']:.3g}",
        flush=True,
    )

    # --- gate 1: clean accuracy per capture length -------------------------
    selection_by_length: dict[int, np.ndarray] = {}
    closed_by_length: dict[str, Any] = {}
    for length in DEV_LENGTHS:
        embeddings = v2_embed_prefixes(
            bundle, selection["captures"], length, device
        )
        selection_by_length[length] = embeddings
        closed_by_length[str(length)] = closed_clean_report(
            embeddings,
            selection["labels"],
            selection["impaired"],
            bundle["prototypes"],
            classes,
        )
        print(
            f"[v2 gates] n{length}: "
            f"clean={closed_by_length[str(length)]['clean_accuracy']:.4f} "
            f"clean_bal={closed_by_length[str(length)]['clean_balanced_accuracy']:.4f}",
            flush=True,
        )

    matched = release.MATCHED_CAPTURE_LENGTH
    if matched not in selection_by_length:
        raise RuntimeError(
            f"matched capture length {matched} was not measured; the release "
            "five-shot protocol embeds support there"
        )
    cache_positions = selection_positions(source["va_idx"], selection["indices"])
    fused_from_cache = v2_embed_cached(
        bundle,
        np.asarray(source["xva"])[cache_positions],
        np.asarray(source["fva"])[cache_positions],
        device,
    )
    frontend_check = frontend_consistency_check(
        selection_by_length[matched],
        fused_from_cache,
        tolerance=frontend_tolerance,
    )
    print(
        f"[v2 gates] frontend consistency ok: max delta "
        f"{frontend_check['max_abs_delta']:.3g}",
        flush=True,
    )

    # --- gate 2: five-shot, both variants, identical to the v3 protocol ----
    enrollment_support, enrollment_split = dev_five_shot_split(
        enrollment["items"],
        enrollment["labels"],
        classes,
        split_seed,
        identity_fields=identity_fields,
    )
    enrollment_support_matched = v2_embed_prefixes(
        bundle,
        [enrollment["captures"][int(p)] for p in enrollment_support],
        matched,
        device,
    )
    selection_support, selection_split = dev_five_shot_split(
        selection["items"],
        selection["labels"],
        classes,
        split_seed,
        identity_fields=identity_fields,
    )
    support_mask = np.zeros(len(selection["labels"]), dtype=bool)
    support_mask[selection_support] = True
    selection_query = np.flatnonzero(~support_mask)

    five_shot_by_variant = {
        "enrollment_support": {
            "role": (
                "primary; evidence-rule compliant. Support is enrollment, query "
                "is the whole eligible selection population, disjoint by "
                "population separation rather than by exclusion."
            ),
            "support_population": "enrollment",
            "query_population": "selection",
            "support_rows": int(len(enrollment_support)),
            "query_rows": int(len(selection["labels"])),
            "split": enrollment_split,
            **five_shot_variant(
                enrollment_support_matched,
                enrollment["labels"][enrollment_support],
                selection_by_length,
                selection["labels"],
                classes,
            ),
        },
        "selection_support": {
            "role": (
                "diagnostic only; structural mirror of the sealed within-corpus "
                "partition. This variant fits 35 prototypes on selection rows "
                "and must not be used to select or tune anything."
            ),
            "support_population": "selection",
            "query_population": "selection minus support",
            "support_rows": int(len(selection_support)),
            "query_rows": int(len(selection_query)),
            "split": selection_split,
            **five_shot_variant(
                selection_by_length[matched][selection_support],
                selection["labels"][selection_support],
                {
                    length: value[selection_query]
                    for length, value in selection_by_length.items()
                },
                selection["labels"][selection_query],
                classes,
            ),
        },
    }
    for name, variant in five_shot_by_variant.items():
        print(
            f"[v2 gates] five-shot {name}: worst bal="
            f"{variant['worst_length_balanced_accuracy']:.4f}",
            flush=True,
        )

    gate_block = assemble_gates(closed_by_length, five_shot_by_variant)
    comparability = comparability_block(
        bundle=bundle,
        selection_sha256=selection["indices_sha256"],
        enrollment_sha256=enrollment["indices_sha256"],
        split_seed=split_seed,
        split_seed_source=split_seed_source,
        five_shot_by_variant=five_shot_by_variant,
        v3_report=v3_report,
    )

    result = {
        "status": "complete",
        "measurement": MEASUREMENT_VERSION,
        "development_only": True,
        "release_evidence": False,
        "sealed_release_data_used": 0,
        "consumed_test_rows_used": 0,
        "retraining_performed": False,
        "recalibration_performed": False,
        "fusion_artifact": {
            "kind": "frozen v2 runtime bundle (read-only)",
            "directory": str(bundle_dir),
            "sha256": bundle["sha256"],
            "seed": V2_BUNDLE_SEED,
            "encoder": "fusion",
            "weight_real": float(bundle["weight_real"]),
            "parameter_count": int(bundle["parameter_count"]),
            "fused_embedding_dim": 2 * int(bundle["embed_dim"]),
        },
        "frontend": v2_frontend_metadata(contract_config),
        "protocol": {
            "mirrors": (
                "evaluate_invariant_release_suite.py via "
                "measure_v3_remaining_gates.py"
            ),
            "evaluation_version": release.EVALUATION_VERSION,
            "matched_capture_length": int(matched),
            "measured_capture_lengths": list(DEV_LENGTHS),
            "release_required_capture_lengths": list(
                release.REQUIRED_CAPTURE_LENGTHS
            ),
            "unmeasurable_release_capture_lengths": list(
                UNMEASURABLE_RELEASE_LENGTHS
            ),
            "five_shot": dict(release.EXPECTED_EVALUATION_PROTOCOL["five_shot"]),
            "split_seed": split_seed,
            "split_seed_source": split_seed_source,
            "imported_from_v3_measurement": [
                "eligible_population",
                "dev_row_identity_fields",
                "dev_five_shot_split",
                "closed_clean_report",
                "five_shot_variant",
                "assemble_gates",
                "validate_split_seed",
                "DEV_LENGTHS",
            ],
            "deviations_from_sealed_protocol": (
                "identical to the v3 measurement's list; see "
                "measure_v3_remaining_gates.py points 1-8, which apply "
                "unchanged here"
            ),
        },
        "population": {
            "selection": {
                "role": "scored for both gates; never fit",
                "corpus_rows": int(len(source["va_idx"])),
                "eligible_rows": int(len(selection["labels"])),
                "eligible_by_class": selection["included_by_class"],
                "excluded_all_zero_n4096_prefix_by_class": selection[
                    "excluded_all_zero_n4096_prefix_by_class"
                ],
                "eligible_indices_sha256": selection["indices_sha256"],
                "clean_rows": int(np.sum(~selection["impaired"])),
                "impaired_rows": int(np.sum(selection["impaired"])),
            },
            "enrollment": {
                "role": "supplies the primary five-shot support draw only",
                "corpus_rows": int(len(source["en_idx"])),
                "eligible_rows": int(len(enrollment["labels"])),
                "eligible_by_class": enrollment["included_by_class"],
                "excluded_all_zero_n4096_prefix_by_class": enrollment[
                    "excluded_all_zero_n4096_prefix_by_class"
                ],
                "eligible_indices_sha256": enrollment["indices_sha256"],
            },
            "training": {
                "role": (
                    "never scored; centers, prototypes, and feature "
                    "standardization are frozen in the bundle. The replay "
                    "check re-embeds cached training/enrollment arrays "
                    "(inference only) to prove the rebuilt model is the "
                    "recorded one."
                ),
                "rows_scored": 0,
            },
        },
        "closed_clean_per_length": closed_by_length,
        "five_shot_per_variant": five_shot_by_variant,
        "gates": gate_block,
        "all_measured_gates_pass": all(
            bool(value["passes"]) for value in gate_block.values()
        ),
        "reproduction_check": {
            "bundle_sha256_verified": bundle["sha256"]["runtime_bundle.pt"],
            "expected_bundle_sha256": EXPECTED_BUNDLE_SHA256,
            "feature_standardization_max_abs_delta_vs_cache": (
                standardization_delta
            ),
            "center_prototype_replay": replay,
            "frontend_consistency_n16384": frontend_check,
        },
        "comparability": comparability,
        "frozen_assets_used": [
            "both branch state dicts, configs, and centers",
            "fused prototypes (clean gate)",
            "feature mean/std",
        ],
        "source_sha256": _source_hashes(),
        "data_audit": source["data_audit"],
        "source_index_contract": {
            "contract": contract,
            "role": (
                "identifies the raw corpus and exposed train/enroll/selection "
                "indices; identical to the v2 bundle's frozen contract and to "
                "the contract the v3 measurement recorded"
            ),
        },
        "device": str(device),
        "seed": V2_BUNDLE_SEED,
        "wall_clock_s": time.perf_counter() - started,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    dev._write_json(report_path, result)
    print(f"[v2 gates] wrote {report_path}", flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--bundle-dir",
        default=str(DEFAULT_BUNDLE_DIR),
        help="frozen v2 runtime bundle directory (read-only, hash-verified)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"where to write {REPORT_NAME}; never the bundle directory",
    )
    parser.add_argument(
        "--v3-report",
        default=None,
        help=(
            "path to a v3 remaining_gates.json; when given, eligible index "
            "sets are asserted equal to it, the split seed defaults to its, "
            "and the support draws are asserted identical"
        ),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help=(
            "seed bound into the five-shot digest payload; defaults to the "
            "--v3-report's split seed, else the v2 bundle model seed. "
            "Release seeds are refused."
        ),
    )
    parser.add_argument(
        "--replay-tolerance",
        type=float,
        default=DEFAULT_REPLAY_TOLERANCE,
        help="max deviation when replaying the bundle's centers and prototypes",
    )
    parser.add_argument(
        "--frontend-tolerance",
        type=float,
        default=DEFAULT_FRONTEND_TOLERANCE,
        help=(
            "max deviation between raw-prefix and cache-derived embeddings "
            "at the matched capture length"
        ),
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
