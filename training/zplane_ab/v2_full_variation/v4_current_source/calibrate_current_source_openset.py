"""Calibrate and evaluate an external open-set policy for the v4 classifier.

This module is deliberately separate from the v4 trainer, fusion assembler,
browser classifier exporter, and live assets.  It composes three decisions:

0. reject an exact all-zero capture as ``no_signal`` before pose estimation;
1. fit a new class/profile/base-balanced pose-noise score on the existing
   FFT-free pose-degeneracy features using combined historical/current training
   plus a disjoint fit-only direct time-domain noise draw.  The frozen v3 hard
   gate is not reused and this score does not gate before classification;
2. compute a direct weighted instantaneous-phase linearity statistic which is
   invariant to global phase, carrier, nonzero scale, and runtime length; and
3. rank the pose score against both predicted-class and route-length-pooled
   enrollment, take their maximum, and rank the winning public-class profile-
   bank distance and phase-linearity statistic against matching predicted-
   class enrollment. Their maximum and threshold are calibrated on enrollment
   only.

Training rows fit the underlying fusion and the new stage-one coefficients.
Enrollment rows fit every new rank and threshold in this file.  Historical and
current ``selection`` rows, development novelty, sealed data, and consumed test
rows are structurally absent from the fitting function.  A design run writes a
browser-serialisable external policy plus an evaluation report.  A validation
run loads that policy verbatim and only scores an independently held service
scale corpus plus fresh synthetic data.

Novelty is generated once at 16,384 samples and observed through exact leading
prefixes.  The generators are direct time-domain processes (complex AR noise,
LFM chirp, and exact no-signal); neither inference nor novelty generation uses
an FFT.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
V2 = HERE.parent
ZPLANE = V2.parent
TRAINING = ZPLANE.parent
REPO = TRAINING.parent
V3_SCALE = V2 / "v3_scale"
for path in (TRAINING, ZPLANE, V2, V3_SCALE, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import current_source_data as corpus_data  # noqa: E402
import evaluate_current_scale as scale_eval  # noqa: E402
import export_current_source_browser_runtime as browser_export  # noqa: E402
import noise_prefilter  # noqa: E402
import pose_degeneracy  # noqa: E402
import run_current_source_dev as branch_runner  # noqa: E402
import time_domain_geometry as td_geometry  # noqa: E402
import time_domain_invariant_patch_preprocess as td_preprocess  # noqa: E402
from train import auroc  # noqa: E402


POLICY_SCHEMA = "atomos.v4.time-domain-current-source.external-openset-policy"
POLICY_SCHEMA_VERSION = 5
REPORT_SCHEMA = "atomos.v4.time-domain-current-source.openset-evaluation"
REPORT_SCHEMA_VERSION = 4
MANIFEST_SCHEMA = "atomos.v4.time-domain-current-source.openset-manifest"
MANIFEST_SCHEMA_VERSION = 2
POLICY_NAME = "time-domain-profile-bank-openset-v4.json"
DESIGN_REPORT_NAME = "design-evaluation.json"
VALIDATION_REPORT_NAME = "validation-evaluation.json"
MANIFEST_NAME = "openset-manifest.json"

PUBLIC_CLASSES = browser_export.PUBLIC_CLASSES
RUNTIME_INPUT_LENGTHS = browser_export.RUNTIME_INPUT_LENGTHS
NOVELTY_FAMILIES = ("no_signal", "noise", "chirp")
PROTOTYPE_SOURCE_ROUTES = ("historical", "current")
CURRENT_CANONICAL_RUNTIME_INPUT_LENGTH = 4_096
EXPECTED_PUBLIC_CLASS_SUPPORT_BY_PROTOTYPE_SOURCE = {
    "historical": (True, True, True, True, True, True, True),
    "current": (False, True, False, True, False, True, True),
}
RUNTIME_LENGTH_POLICY_BY_PROTOTYPE_SOURCE = {
    "historical": {
        "kind": "largest_supported_prefix_not_exceeding_valid_sample_count",
        "observation_length_reporting_rule":
            browser_export.RUNTIME_BUCKET_RULE,
    },
    "current": {
        "kind": "fixed_causal_prefix",
        "effective_runtime_input_length":
            CURRENT_CANONICAL_RUNTIME_INPUT_LENGTH,
        "observation_length_reporting_rule":
            browser_export.RUNTIME_BUCKET_RULE,
    },
}

FROZEN_V3_PREFILTER_AUDIT_DIR = (
    V2
    / "artifacts"
    / "invariant_patch"
    / "v3_scale"
    / "noise_prefilter_fit20261001_budget001"
    / "bundles"
)
DEFAULT_DESIGN_SEED = 20_263_001
DEFAULT_VALIDATION_SEEDS = (20_263_002, 20_263_003)
DEFAULT_STAGE_ONE_FIT_SEED = 20_264_001
DEFAULT_NOVELTY_ROWS = 300
DEFAULT_COMPOSITE_KNOWN_BUDGET = 0.03
DEFAULT_HELD_SCALE_EVAL_SEED = 20_262_902
DEFAULT_HELD_SCALE_REALIZATIONS_PER_PROFILE = 24
DEFAULT_HELD_SCALE_CORPUS_RELATIVE = (
    "training/artifacts/"
    "signallab-current-scale-eval-v3-seed20262902-r24"
)
DEFAULT_VALIDATION_LEDGER_RELATIVE = (
    "training/artifacts/"
    "v4-current-source-openset-validation-seeds20263002-20263003.ledger.json"
)

# Development gates only.  They are diagnostics, not release claims.
KNOWN_FALSE_UNKNOWN_CEILING = 0.12
# Cell-level abstention claims need enough *closed-head-correct* observations
# to separate abstention damage from classifier error.  Five-row selection
# profile cells are explicitly underpowered and cannot pass or fail a cell
# gate; the independently held service corpus supplies the release evidence.
KNOWN_CELL_MIN_CLOSED_CORRECT_ROWS = 20
DESIGN_CURRENT_POOLED_CLOSED_ACCURACY_FLOOR = 0.95
DESIGN_CURRENT_EVERY_LENGTH_CLOSED_ACCURACY_FLOOR = 0.95
DESIGN_CURRENT_PRESENT_CLASS_BALANCED_ACCURACY_FLOOR = 0.95
DESIGN_CURRENT_PROFILE_BALANCED_ACCURACY_FLOOR = 0.95
DESIGN_CURRENT_WORST_PROFILE_CLOSED_ACCURACY_FLOOR = 0.70
DESIGN_CURRENT_WIFI_HR_DSSS_ACCURACY_FLOOR = 0.95
HELD_CURRENT_POOLED_CLOSED_ACCURACY_FLOOR = 0.95
HELD_CURRENT_EVERY_LEGAL_CELL_CLOSED_ACCURACY_FLOOR = 0.90
HELD_CURRENT_WORST_CELL_PRESENT_CLASS_BALANCED_ACCURACY_FLOOR = 0.85
HELD_CURRENT_PROFILE_BALANCED_ACCURACY_FLOOR = 0.90
HELD_CURRENT_WORST_PROFILE_POOLED_CLOSED_ACCURACY_FLOOR = 0.70
HELD_CURRENT_WORST_PROFILE_LEGAL_CELL_CLOSED_ACCURACY_FLOOR = 0.50
HELD_CURRENT_WIFI_HR_DSSS_ACCURACY_FLOOR = 0.95
WIFI_HR_DSSS_PROFILE = "current:wifi-hr-dsss-11m"
NOVELTY_AUROC_FLOOR = 0.80
NOVELTY_RECALL_FLOOR = 0.10
LENGTH_SCALE_DECISION_AGREEMENT_FLOOR = 0.95
CURRENT_SELECTION_CAUSAL_AGREEMENT_FLOOR = 1.0
MIN_PAIRED_DISTINCT_RUNTIME_LENGTHS = 2
EXPECTED_DESIGN_CURRENT_SELECTION_STORED_OBSERVATIONS = 1_941
EXPECTED_DESIGN_CURRENT_SELECTION_OBSERVATION_CELLS = 5_766
EXPECTED_DESIGN_CURRENT_SELECTION_TWO_LENGTH_OBSERVATIONS = 57
EXPECTED_DESIGN_CURRENT_SELECTION_THREE_LENGTH_OBSERVATIONS = 1_884
EXPECTED_HELD_PAIR_CELLS = 744
EXPECTED_HELD_OBSERVATION_CELLS = 8_880
EXPECTED_HELD_PROFILE_LEGAL_CELLS = 370
EXPECTED_HELD_PHYSICAL_SCALE_CELLS = 2_232
EXPECTED_HELD_PHYSICAL_SCALE_CELLS_PER_LENGTH = 744
EXPECTED_HELD_CAUSAL_PREFIX_CELLS = 2_976

STAGE_ZERO_KIND = "exact-zero-no-signal-validity-gate"
STAGE_ZERO_DECISION = (
    "reject as no_signal iff max(abs(effective_route_scoped_"
    "inference_prefix_iq)) == 0"
)
STAGE_ZERO_PREFIX_SCOPE = (
    "current:first_4096_samples;historical:largest_supported_prefix_"
    "not_exceeding_valid_sample_count"
)
STAGE_ONE_KIND = (
    "class-profile-base-balanced-pose-degeneracy-noise-score-v1"
)
STAGE_ONE_RANK_KIND = (
    "max-predicted-class-and-route-length-pooled-enrollment-"
    "empirical-rank-v1"
)
STAGE_TWO_KIND = (
    "winning-public-class-profile-distance-enrollment-empirical-rank"
)
STAGE_CHIRP_KIND = (
    "weighted-instantaneous-phase-linearity-enrollment-rank-v1"
)
COMPOSITION_KIND = (
    "max-effective-pose-score-rank-profile-distance-rank-"
    "phase-linearity-rank"
)
CHIRP_ACTIVE_RMS_FLOOR = 0.02
CHIRP_MIN_CONTIGUOUS_RUN = 32
CHIRP_MIN_VALID_ADJACENCIES = 128
CHIRP_MIN_VALID_FRACTION = 0.25
CHIRP_WEIGHT_CEILING = 16.0
CHIRP_PHASE_STD_FLOOR_RAD = 1e-4
RUNTIME_COMPARISON_MARGIN = 1e-8
EMPIRICAL_RANK_RULE = (
    "searchsorted(sorted_enrollment, value, side='left') / "
    "(enrollment_count + 1)"
)

_OLD_SPENT_OR_RESERVED_SEED_RANGES = (
    (20_260_700, 20_260_799, "historical release namespace"),
    (20_260_900, 20_260_999, "v3 development novelty namespace"),
    (20_261_000, 20_261_999, "v3 prefilter fitting namespace"),
)
_V4_NOVELTY_NAMESPACE = (20_263_000, 20_263_999)
_V4_FITTING_NAMESPACE = (20_264_000, 20_264_999)
_UNIT_TEST_NOVELTY_NAMESPACE = (20_265_000, 20_265_999)


def _profile_legal_cell_key(
    source_profile: str,
    length: int,
    scale_factor: float,
) -> str:
    """Return the canonical audit key for one legal profile observation cell."""
    if not source_profile or int(length) not in RUNTIME_INPUT_LENGTHS:
        raise ValueError("profile legal-cell identity is invalid")
    scale = float(scale_factor)
    if scale not in {float(value) for value in scale_eval.SCALE_FACTORS}:
        raise ValueError("profile legal-cell scale is invalid")
    return f"{source_profile}|N={int(length)}|scale={scale:g}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            _jsonable(payload),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(_json_bytes(payload))
    os.replace(temporary, path)


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        with Path(path).open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON object {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def reject_sensitive_path(path: str | Path, role: str) -> Path:
    """Refuse release, sealed, consumed-test, and live-asset paths."""
    resolved = Path(path).expanduser().resolve()
    normalized = [part.lower().replace("_", "-") for part in resolved.parts]
    if (
        any("release" in part for part in normalized)
        or any("sealed" in part for part in normalized)
        or any(
            "consumed-test" in part or "consumedtest" in part
            for part in normalized
        )
    ):
        raise ValueError(
            "v4 open-set development refuses release, sealed, or consumed-test "
            f"{role} paths: {resolved}"
        )
    live_assets = (REPO / "src" / "embedding" / "assets").resolve()
    if resolved == live_assets or live_assets in resolved.parents:
        raise ValueError(
            f"refusing live classifier assets as {role}: {resolved}"
        )
    return resolved


def validate_empty_output(path: str | Path) -> Path:
    output = reject_sensitive_path(path, "output")
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f"output is not a directory: {output}")
        contents = sorted(item.name for item in output.iterdir())
        if contents:
            raise FileExistsError(
                f"refusing to overwrite non-empty output {output}: "
                f"{contents[:5]}"
            )
    return output


def validate_seed(seed: Any) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("novelty seed must be an integer")
    value = int(seed)
    for lower, upper, reason in _OLD_SPENT_OR_RESERVED_SEED_RANGES:
        if lower <= value <= upper:
            raise ValueError(f"novelty seed {value} is unavailable: {reason}")
    lower, upper = _V4_NOVELTY_NAMESPACE
    if not lower <= value <= upper:
        raise ValueError(
            f"v4 novelty seed must lie in the fresh namespace [{lower}, {upper}]"
        )
    return value


def _validate_unit_test_seed(seed: Any) -> int:
    """Validate a generator-only seed that cannot become evaluation evidence."""
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("unit-test novelty seed must be an integer")
    value = int(seed)
    lower, upper = _UNIT_TEST_NOVELTY_NAMESPACE
    if not lower <= value <= upper:
        raise ValueError(
            f"unit-test novelty seed must lie in [{lower}, {upper}]"
        )
    return value


def validate_seed_plan(
    role: str,
    seeds: Sequence[Any],
    *,
    design_seed: Any,
) -> tuple[int, ...]:
    role_value = str(role)
    if role_value not in {"design", "validate"}:
        raise ValueError("role must be 'design' or 'validate'")
    design = validate_seed(design_seed)
    values = tuple(validate_seed(value) for value in seeds)
    if not values or len(values) != len(set(values)):
        raise ValueError("novelty seeds must be nonempty and distinct")
    if role_value == "design":
        if values != (design,):
            raise ValueError(
                "design must draw exactly its declared design novelty seed"
            )
    elif design in values:
        raise ValueError("validation may not redraw the design novelty seed")
    return values


def validate_fitting_seed(seed: Any) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("stage-one fitting seed must be an integer")
    value = int(seed)
    lower, upper = _V4_FITTING_NAMESPACE
    if not lower <= value <= upper:
        raise ValueError(
            f"stage-one fitting seed must lie in [{lower}, {upper}]"
        )
    return value


def validate_prototype_source(value: Any) -> str:
    source = str(value)
    if source not in PROTOTYPE_SOURCE_ROUTES:
        raise ValueError(
            "prototype source must be one of "
            f"{list(PROTOTYPE_SOURCE_ROUTES)}, got {source!r}"
        )
    return source


def _runtime_policy_lengths(prototype_source: Any) -> tuple[int, ...]:
    source = validate_prototype_source(prototype_source)
    return (
        (CURRENT_CANONICAL_RUNTIME_INPUT_LENGTH,)
        if source == "current"
        else tuple(RUNTIME_INPUT_LENGTHS)
    )


def effective_runtime_input_length(
    policy: Mapping[str, Any],
    *,
    prototype_source: Any,
    observation_length: Any,
) -> int:
    """Resolve the fail-closed route-scoped inference prefix."""
    source = validate_prototype_source(prototype_source)
    if (
        isinstance(observation_length, (bool, np.bool_))
        or not isinstance(observation_length, (int, np.integer))
        or int(observation_length) not in RUNTIME_INPUT_LENGTHS
    ):
        raise ValueError("observation length is not a runtime bucket")
    if policy.get("runtime_length_policy_by_prototype_source") != (
        RUNTIME_LENGTH_POLICY_BY_PROTOTYPE_SOURCE
    ):
        raise ValueError("route-scoped runtime-length policy changed")
    observed = int(observation_length)
    return (
        CURRENT_CANONICAL_RUNTIME_INPUT_LENGTH
        if source == "current"
        else observed
    )


def _expected_validation_protocol() -> dict[str, Any]:
    return {
        "held_scale_corpus_relative_path":
            DEFAULT_HELD_SCALE_CORPUS_RELATIVE,
        "held_scale_corpus_schema": scale_eval.CORPUS_SCHEMA,
        "held_scale_eval_seed": DEFAULT_HELD_SCALE_EVAL_SEED,
        "held_scale_realizations_per_profile":
            DEFAULT_HELD_SCALE_REALIZATIONS_PER_PROFILE,
        "held_scale_prototype_source": "current",
        "novelty_seeds": list(DEFAULT_VALIDATION_SEEDS),
        "novelty_n_each_per_family_per_length": DEFAULT_NOVELTY_ROWS,
        "validation_ledger_relative_path":
            DEFAULT_VALIDATION_LEDGER_RELATIVE,
        "single_use_enforcement":
            "atomic_create_exclusive_before_validation_scoring",
    }


def validation_corpus_binding_sha256(
    manifest_sha256: str,
    raw_sha256: str,
) -> str:
    if not _is_sha256(manifest_sha256) or not _is_sha256(raw_sha256):
        raise ValueError("validation corpus commitment hashes are invalid")
    return hashlib.sha256(
        (
            "atomos-v4-held-scale-corpus-v1\0"
            f"{manifest_sha256}\0{raw_sha256}"
        ).encode("ascii")
    ).hexdigest()


def build_validation_protocol(
    *,
    classifier_asset_sha256: str,
    held_scale_manifest_sha256: str,
    held_scale_raw_sha256: str,
) -> dict[str, Any]:
    if not _is_sha256(classifier_asset_sha256):
        raise ValueError("validation protocol classifier hash is invalid")
    manifest = str(held_scale_manifest_sha256)
    raw = str(held_scale_raw_sha256)
    return {
        **_expected_validation_protocol(),
        "classifier_asset_sha256": classifier_asset_sha256,
        "held_scale_manifest_sha256": manifest,
        "held_scale_raw_sha256": raw,
        "held_scale_corpus_binding_sha256":
            validation_corpus_binding_sha256(manifest, raw),
    }


def _validate_validation_protocol_record(
    value: Any,
    *,
    classifier_asset_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("open-set validation protocol must be an object")
    protocol = dict(value)
    for key, expected in _expected_validation_protocol().items():
        if protocol.get(key) != expected:
            raise ValueError(
                f"open-set validation protocol {key} changed"
            )
    manifest = protocol.get("held_scale_manifest_sha256")
    raw = protocol.get("held_scale_raw_sha256")
    if (
        protocol.get("classifier_asset_sha256")
        != classifier_asset_sha256
        or not _is_sha256(manifest)
        or not _is_sha256(raw)
        or protocol.get("held_scale_corpus_binding_sha256")
        != validation_corpus_binding_sha256(str(manifest), str(raw))
        or set(protocol)
        != set(_expected_validation_protocol())
        | {
            "classifier_asset_sha256",
            "held_scale_manifest_sha256",
            "held_scale_raw_sha256",
            "held_scale_corpus_binding_sha256",
        }
    ):
        raise ValueError("open-set validation corpus commitment is invalid")
    return protocol


def _resolve_precommitted_repo_path(relative: str, *, role: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError(f"precommitted {role} path must be repo-relative")
    resolved = (REPO / value).resolve()
    if REPO.resolve() not in resolved.parents:
        raise ValueError(f"precommitted {role} path escapes the repository")
    return reject_sensitive_path(resolved, role)


def validate_validation_protocol(
    policy: Mapping[str, Any],
    *,
    held_scale_corpus: str | Path,
    novelty_seeds: Sequence[int],
    novelty_n: int,
) -> dict[str, Any]:
    """Require the validation inputs precommitted by the design policy."""
    binding = policy.get("classifier_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("open-set classifier binding is missing")
    expected = _validate_validation_protocol_record(
        policy.get("validation_protocol"),
        classifier_asset_sha256=str(
            binding.get("browser_classifier_asset_sha256")
        ),
    )
    expected_held = _resolve_precommitted_repo_path(
        expected["held_scale_corpus_relative_path"],
        role="held scale corpus input",
    )
    observed_held = reject_sensitive_path(
        held_scale_corpus, "held scale corpus input"
    )
    if observed_held != expected_held:
        raise ValueError(
            "held scale corpus path differs from the precommitted validation "
            f"path: {observed_held} != {expected_held}"
        )
    seeds = tuple(validate_seed(seed) for seed in novelty_seeds)
    if seeds != tuple(expected["novelty_seeds"]):
        raise ValueError(
            "validation novelty seeds differ from the precommitted seed plan"
        )
    if (
        isinstance(novelty_n, (bool, np.bool_))
        or not isinstance(novelty_n, (int, np.integer))
        or int(novelty_n)
        != int(expected["novelty_n_each_per_family_per_length"])
    ):
        raise ValueError(
            "validation novelty row count differs from the precommitted plan"
        )
    return expected


def claim_validation_ledger(
    *,
    policy_sha256: str,
    protocol: Mapping[str, Any],
) -> tuple[Path, str]:
    """Atomically and irreversibly claim the one allowed validation run."""
    if not _is_sha256(policy_sha256):
        raise ValueError("validation ledger needs a policy SHA-256")
    classifier_sha = protocol.get("classifier_asset_sha256")
    validated = _validate_validation_protocol_record(
        protocol,
        classifier_asset_sha256=str(classifier_sha),
    )
    ledger = _resolve_precommitted_repo_path(
        str(protocol["validation_ledger_relative_path"]),
        role="validation ledger",
    )
    ledger.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": (
            "atomos.v4.time-domain-current-source."
            "openset-validation-ledger"
        ),
        "schema_version": 1,
        "state": "validation_evidence_claimed",
        "policy_sha256": policy_sha256,
        "validation_protocol": validated,
        "deletion_or_replacement_invalidates_evidence": True,
    }
    encoded = _json_bytes(payload)
    try:
        descriptor = os.open(
            ledger,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        raise RuntimeError(
            "validation evidence has already been claimed; refusing to redraw "
            f"the precommitted seeds: {ledger}"
        ) from exc
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("validation ledger write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    digest = _sha256(ledger)
    if digest != hashlib.sha256(encoded).hexdigest():
        raise RuntimeError("validation ledger bytes changed after creation")
    return ledger, digest


def empirical_rank(value: np.ndarray, calibration: np.ndarray) -> np.ndarray:
    """Strict empirical upper-tail rank, matching the v3 browser convention."""
    query = np.asarray(value, dtype=np.float64)
    ordered = np.asarray(calibration, dtype=np.float64)
    if (
        query.ndim != 1
        or ordered.ndim != 1
        or len(ordered) == 0
        or not np.isfinite(query).all()
        or not np.isfinite(ordered).all()
    ):
        raise ValueError("rank query and calibration must be finite vectors")
    if len(ordered) > 1 and np.any(ordered[1:] < ordered[:-1]):
        raise ValueError("rank calibration must already be sorted")
    return np.searchsorted(ordered, query, side="left") / (
        len(ordered) + 1.0
    )


def budget_threshold(
    known_scores: np.ndarray,
    budget: float,
) -> tuple[float, dict[str, Any]]:
    """Smallest threshold whose enrollment false-positive rate is <= budget."""
    scores = np.asarray(known_scores, dtype=np.float64).reshape(-1)
    value = float(budget)
    if (
        len(scores) == 0
        or not np.isfinite(scores).all()
        or not math.isfinite(value)
        or not 0.0 <= value < 1.0
    ):
        raise ValueError("budget threshold needs finite scores and budget [0,1)")
    descending = np.sort(scores)[::-1]
    allowance = min(int(math.floor(value * len(scores))), len(scores) - 1)
    boundary = float(descending[allowance])
    comparison_margin = RUNTIME_COMPARISON_MARGIN * max(1.0, abs(boundary))
    threshold = float(boundary + comparison_margin)
    realised = float(np.mean(scores >= threshold))
    if realised > value + 1e-15:
        raise AssertionError("budget threshold exceeded its declared budget")
    return threshold, {
        "rows": int(len(scores)),
        "allowance": allowance,
        "calibration_boundary": boundary,
        "runtime_comparison_margin": comparison_margin,
        "realised_known_false_positive_rate": realised,
    }


def grouped_budget_threshold(
    known_scores: np.ndarray,
    groups: Sequence[str],
    *,
    pooled_budget: float,
    group_budget: float,
) -> tuple[float, dict[str, Any]]:
    """Global threshold satisfying pooled and every source/profile budget."""
    scores = np.asarray(known_scores, dtype=np.float64).reshape(-1)
    group_array = np.asarray([str(value) for value in groups], dtype=object)
    if group_array.shape != scores.shape:
        raise ValueError("grouped threshold scores/groups lost alignment")
    pooled_threshold, pooled_detail = budget_threshold(
        scores, pooled_budget
    )
    per_group: dict[str, Any] = {}
    thresholds = [pooled_threshold]
    for group in sorted(set(str(value) for value in group_array)):
        mask = group_array == group
        threshold, detail = budget_threshold(
            scores[mask], group_budget
        )
        thresholds.append(threshold)
        per_group[group] = {
            "threshold_needed": threshold,
            **detail,
        }
    chosen = float(max(thresholds))
    per_group_realised = {
        group: float(np.mean(scores[group_array == group] >= chosen))
        for group in per_group
    }
    if max(per_group_realised.values(), default=0.0) > group_budget + 1e-15:
        raise AssertionError("chosen group threshold exceeded a group budget")
    return chosen, {
        "pooled_budget": float(pooled_budget),
        "group_budget": float(group_budget),
        "pooled_threshold_needed": pooled_threshold,
        "per_group": per_group,
        "chosen_as_maximum_required_threshold": chosen,
        "realised_pooled_false_positive_rate":
            float(np.mean(scores >= chosen)),
        "realised_false_positive_rate_by_group": per_group_realised,
        "worst_group_realised_false_positive_rate":
            max(per_group_realised.values(), default=0.0),
    }


def _derived_family_seed(base_seed: int, family: str) -> int:
    digest = hashlib.sha256(
        f"atomos-v4-openset:{int(base_seed)}:{family}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _complex_digest(rows: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        canonical = np.ascontiguousarray(np.asarray(row, dtype=np.complex64))
        digest.update(canonical.view(np.uint8))
    return digest.hexdigest()


def _noise_row(length: int, rng: np.random.Generator) -> np.ndarray:
    """Time-domain complex AR noise with random correlation and carrier."""
    innovations = (
        rng.standard_normal(length) + 1j * rng.standard_normal(length)
    ) / math.sqrt(2.0)
    rho = float(rng.uniform(0.0, 0.97))
    carrier = float(rng.uniform(-math.pi, math.pi))
    coefficient = rho * np.exp(1j * carrier)
    scale = math.sqrt(max(1.0 - rho * rho, 1e-12))
    output = np.empty(length, dtype=np.complex128)
    output[0] = innovations[0]
    for index in range(1, length):
        output[index] = coefficient * output[index - 1] + scale * innovations[index]
    output *= 10.0 ** float(rng.uniform(-2.0, 2.0))
    output *= np.exp(1j * float(rng.uniform(-math.pi, math.pi)))
    return output.astype(np.complex64)


def _chirp_row(length: int, rng: np.random.Generator) -> np.ndarray:
    """Direct time-domain LFM with random sweep, offset, envelope, and scale."""
    sample = np.arange(length, dtype=np.float64)
    normalized = sample / max(length - 1, 1)
    start = float(rng.uniform(-0.42, 0.42))
    sweep = float(rng.uniform(0.08, 0.80))
    if rng.random() < 0.5:
        sweep = -sweep
    phase = 2.0 * np.pi * (
        start * sample + 0.5 * sweep * sample * normalized
    )
    envelope_rate = int(rng.integers(1, 7))
    envelope_phase = float(rng.uniform(-math.pi, math.pi))
    envelope = 0.65 + 0.35 * np.cos(
        2.0 * np.pi * envelope_rate * normalized + envelope_phase
    )
    output = envelope * np.exp(1j * phase)
    output *= 10.0 ** float(rng.uniform(-2.0, 2.0))
    output *= np.exp(1j * float(rng.uniform(-math.pi, math.pi)))
    return np.asarray(output, dtype=np.complex64)


def instantaneous_phase_linearity_chirp_score(raw: np.ndarray) -> float:
    """Weighted R² of instantaneous phase against normalized time.

    Adjacent products cancel global phase.  RMS-relative admission and
    weights make the result invariant to every finite nonzero global scale.
    Per-run intercept removal also cancels a constant carrier phase increment.
    Normalized time makes R² independent of capture length and nonzero linear
    time scaling.  Near-constant instantaneous phase (CW) is explicitly
    assigned zero instead of being treated as a zero-slope chirp.
    """
    value = np.asarray(raw)
    if value.ndim != 1 or len(value) < CHIRP_MIN_VALID_ADJACENCIES + 1:
        raise ValueError("phase-linearity score input is too short")
    if (
        not np.isfinite(value.real).all()
        or not np.isfinite(value.imag).all()
    ):
        raise ValueError("phase-linearity score input is non-finite")
    complex_value = np.asarray(value, dtype=np.complex128)
    amplitude = np.abs(complex_value)
    rms = float(np.sqrt(np.mean(amplitude * amplitude)))
    if not math.isfinite(rms) or rms <= 0.0:
        return 0.0
    relative_amplitude = amplitude / rms
    adjacent = complex_value[1:] * np.conj(complex_value[:-1])
    valid = (
        (relative_amplitude[:-1] >= CHIRP_ACTIVE_RMS_FLOOR)
        & (relative_amplitude[1:] >= CHIRP_ACTIVE_RMS_FLOOR)
        & (np.abs(adjacent) > 0.0)
    )
    valid_indices = np.flatnonzero(valid)
    if not len(valid_indices):
        return 0.0
    breaks = np.flatnonzero(np.diff(valid_indices) != 1) + 1
    runs = [
        run
        for run in np.split(valid_indices, breaks)
        if len(run) >= CHIRP_MIN_CONTIGUOUS_RUN
    ]
    coverage = sum(len(run) for run in runs)
    minimum_coverage = max(
        CHIRP_MIN_VALID_ADJACENCIES,
        int(CHIRP_MIN_VALID_FRACTION * (len(complex_value) - 1)),
    )
    if coverage < minimum_coverage:
        return 0.0
    weight_sum = 0.0
    xx = 0.0
    yy = 0.0
    xy = 0.0
    denominator = float(len(complex_value) - 1)
    for run in runs:
        phase = np.unwrap(np.angle(adjacent[run]))
        time_coordinate = run.astype(np.float64) / denominator
        weights = np.minimum(
            relative_amplitude[run],
            relative_amplitude[run + 1],
        ) ** 2
        weights = np.minimum(weights, CHIRP_WEIGHT_CEILING)
        run_weight = float(np.sum(weights))
        if run_weight <= 0.0:
            continue
        time_center = float(
            np.sum(weights * time_coordinate) / run_weight
        )
        phase_center = float(np.sum(weights * phase) / run_weight)
        centered_time = time_coordinate - time_center
        centered_phase = phase - phase_center
        weight_sum += run_weight
        xx += float(np.sum(weights * centered_time * centered_time))
        yy += float(np.sum(weights * centered_phase * centered_phase))
        xy += float(np.sum(weights * centered_time * centered_phase))
    if (
        weight_sum <= 0.0
        or xx <= 0.0
        or yy
        <= weight_sum * CHIRP_PHASE_STD_FLOOR_RAD ** 2
    ):
        return 0.0
    score = xy * xy / (xx * yy)
    if not math.isfinite(score):
        raise RuntimeError("phase-linearity score is non-finite")
    return float(min(score, 1.0))


def instantaneous_phase_linearity_scores(
    rows: Sequence[np.ndarray],
) -> np.ndarray:
    values = list(rows)
    if not values:
        raise ValueError("phase-linearity scoring needs at least one row")
    result = np.asarray(
        [
            instantaneous_phase_linearity_chirp_score(
                np.asarray(row, dtype=np.complex64)
            )
            for row in values
        ],
        dtype=np.float64,
    )
    if not np.isfinite(result).all():
        raise RuntimeError("phase-linearity scorer returned non-finite values")
    return result


def generate_novelty_raw(
    seed: Any,
    *,
    n_each: int,
    lengths: Sequence[int] = RUNTIME_INPUT_LENGTHS,
    _unit_test_only: bool = False,
) -> tuple[dict[str, dict[int, list[np.ndarray]]], dict[str, Any]]:
    """Generate each row once at max length, then expose exact raw prefixes."""
    base_seed = (
        _validate_unit_test_seed(seed)
        if _unit_test_only
        else validate_seed(seed)
    )
    prefix_lengths = tuple(sorted(set(int(length) for length in lengths)))
    if (
        prefix_lengths != tuple(length for length in RUNTIME_INPUT_LENGTHS
                                if length in prefix_lengths)
        or not prefix_lengths
    ):
        raise ValueError(
            f"lengths must be a nonempty subset of {list(RUNTIME_INPUT_LENGTHS)}"
        )
    if isinstance(n_each, bool) or int(n_each) <= 0:
        raise ValueError("n_each must be positive")
    longest = max(prefix_lengths)
    output: dict[str, dict[int, list[np.ndarray]]] = {}
    provenance: dict[str, Any] = {}
    for family in NOVELTY_FAMILIES:
        derived = _derived_family_seed(base_seed, family)
        rng = np.random.default_rng(derived)
        if family == "no_signal":
            longest_rows = [
                np.zeros(longest, dtype=np.complex64)
                for _ in range(int(n_each))
            ]
        elif family == "noise":
            longest_rows = [
                _noise_row(longest, rng) for _ in range(int(n_each))
            ]
        elif family == "chirp":
            longest_rows = [
                _chirp_row(longest, rng) for _ in range(int(n_each))
            ]
        else:  # pragma: no cover - family constant is frozen above
            raise AssertionError(family)
        output[family] = {
            length: [
                np.ascontiguousarray(row[:length], dtype=np.complex64)
                for row in longest_rows
            ]
            for length in prefix_lengths
        }
        provenance[family] = {
            "base_seed": base_seed,
            "unit_test_only": bool(_unit_test_only),
            "derived_seed": derived,
            "rows": int(n_each),
            "generated_once_at": longest,
            "longest_complex64_sha256": _complex_digest(longest_rows),
            "prefix_complex64_sha256": {
                str(length): _complex_digest(output[family][length])
                for length in prefix_lengths
            },
            "shorter_observation_rule": "exact leading raw prefix",
            "generator": {
                "no_signal": "exact complex64 zeros",
                "noise": "direct time-domain complex AR(1)",
                "chirp": "direct time-domain linear-FM exponential",
            }[family],
            "uses_frequency_transform": False,
        }
    return output, provenance


@dataclass
class StageOneTemplate:
    directory: Path
    models: dict[int, noise_prefilter.NoisePrefilter]
    set_sha256: str
    bundle_sha256: dict[int, str]
    metadata: dict[int, Mapping[str, Any]]


def load_frozen_v3_stage_one_for_audit(
    directory: str | Path,
) -> StageOneTemplate:
    """Load the rejected v3 bundle only to reproduce its compatibility audit."""
    source = reject_sensitive_path(directory, "stage-one input")
    models = noise_prefilter.load_prefilter_set(source)
    if tuple(sorted(models)) != tuple(RUNTIME_INPUT_LENGTHS):
        raise ValueError(
            "stage-one prefilter lengths must exactly match runtime buckets"
        )
    current_extractor = pose_degeneracy.pose_degeneracy_metadata()
    metadata: dict[int, Mapping[str, Any]] = {}
    digests: dict[int, str] = {}
    for length in RUNTIME_INPUT_LENGTHS:
        bundle = source / f"N{length}"
        meta = _load_json(bundle / "meta.json")
        required = {
            "schema": noise_prefilter.BUNDLE_SCHEMA,
            "version": noise_prefilter.NOISE_PREFILTER_VERSION,
            "capture_length": length,
            "inference_uses_frequency_transform": False,
            "uses_absolute_scale": False,
            "invariant_to_global_phase": True,
            "invariant_to_global_scale": True,
            "fit_population": noise_prefilter.TRAIN_SPLIT,
            "threshold_population": noise_prefilter.ENROLLMENT_SPLIT,
            "selection_rows_used": 0,
            "consumed_test_rows_used": 0,
            "sealed_release_paths_read": 0,
            "gates_before_classification": True,
            "additive_only": False,
        }
        for key, expected in required.items():
            if meta.get(key) != expected:
                raise ValueError(
                    f"stage-one N{length} {key} must be {expected!r}, "
                    f"got {meta.get(key)!r}"
                )
        if meta.get("extractor") != current_extractor:
            raise ValueError(
                f"stage-one N{length} extractor contract differs from current "
                "pose_degeneracy.py"
            )
        if tuple(meta.get("feature_names", ())) != tuple(
            noise_prefilter.PREFILTER_FEATURES
        ):
            raise ValueError(f"stage-one N{length} feature order changed")
        model = models[length]
        if (
            model.capture_length != length
            or tuple(model.feature_names)
            != tuple(noise_prefilter.PREFILTER_FEATURES)
        ):
            raise ValueError(f"stage-one N{length} model shape contract changed")
        metadata[length] = meta
        digests[length] = noise_prefilter.bundle_sha256(bundle)
    return StageOneTemplate(
        directory=source,
        models=models,
        set_sha256=noise_prefilter.prefilter_set_sha256(source),
        bundle_sha256=digests,
        metadata=metadata,
    )


def _pose_feature_row(
    raw: np.ndarray,
    *,
    patch_length: int,
    target_frac: float,
) -> np.ndarray:
    minimum = td_geometry.minimum_bandwidth_for_active_span(
        int(len(raw)), int(patch_length), float(target_frac)
    )
    value = pose_degeneracy.pose_degeneracy_features(
        raw, min_bandwidth=float(minimum)
    )
    result = np.asarray(value, dtype=np.float64)
    if (
        result.shape != (pose_degeneracy.FEATURE_COUNT,)
        or not np.isfinite(result).all()
    ):
        raise RuntimeError("pose-degeneracy extractor returned invalid features")
    return result


def stage_one_scores(
    template: StageOneTemplate,
    rows: Sequence[np.ndarray],
    *,
    length: int,
    patch_length: int,
    target_frac: float,
) -> np.ndarray:
    values = list(rows)
    if not values:
        raise ValueError("stage-one scoring needs at least one row")
    if int(length) not in template.models:
        raise ValueError(f"no stage-one model for length {length}")
    features = np.stack(
        [
            _pose_feature_row(
                np.asarray(row, dtype=np.complex64),
                patch_length=patch_length,
                target_frac=target_frac,
            )
            for row in values
        ]
    )
    return np.asarray(
        template.models[int(length)].score(
            features, capture_length=int(length)
        ),
        dtype=np.float64,
    )


@dataclass
class DevelopmentContext:
    fusion: dict[str, Any]
    historical: corpus_data.RawCorpus
    current: corpus_data.RawCorpus
    data: dict[str, Any]
    preprocessing_audit: dict[str, Any]
    module: torch.nn.Module
    device: torch.device
    patch_length: int
    patch_count: int
    target_frac: float


def _raw_role_views(
    corpus: corpus_data.RawCorpus,
    role: str,
) -> dict[int, list[np.ndarray]]:
    """Rebuild raw views in the exact order used by ``_prepare_data``."""
    buckets: dict[int, list[np.ndarray]] = {}
    for row in corpus.rows_by_role[role]:
        minimum = corpus.prefix(row, corpus_data.MINIMUM_INPUT_LENGTH)
        if float(np.max(np.abs(minimum))) == 0.0:
            if corpus.source == "current":
                raise ValueError(
                    f"current {role} row {row.identity} has an all-zero "
                    "eligible prefix"
                )
            continue
        for length in corpus_data.training_view_lengths(row.valid_sample_count):
            buckets.setdefault(length, []).append(
                np.ascontiguousarray(
                    corpus.prefix(row, length), dtype=np.complex64
                )
            )
    if not buckets:
        raise ValueError(f"{corpus.source} {role} has no eligible raw views")
    return dict(sorted(buckets.items()))


def _role_records(
    corpus: corpus_data.RawCorpus,
    role: str,
) -> dict[int, list[corpus_data.RowRef]]:
    """Return admitted row records in the same per-length order as raw views."""
    buckets: dict[int, list[corpus_data.RowRef]] = {}
    for row in corpus.rows_by_role[role]:
        minimum = corpus.prefix(row, corpus_data.MINIMUM_INPUT_LENGTH)
        if float(np.max(np.abs(minimum))) == 0.0:
            if corpus.source == "current":
                raise ValueError(
                    f"current {role} row {row.identity} has an all-zero "
                    "eligible prefix"
                )
            continue
        for length in corpus_data.training_view_lengths(row.valid_sample_count):
            buckets.setdefault(length, []).append(row)
    return dict(sorted(buckets.items()))


def _combined_enrollment_raw(
    historical: corpus_data.RawCorpus,
    current: corpus_data.RawCorpus,
) -> dict[int, list[np.ndarray]]:
    populations = [
        _raw_role_views(historical, "enrollment"),
        _raw_role_views(current, "enrollment"),
    ]
    return {
        length: [
            row
            for population in populations
            for row in population.get(length, [])
        ]
        for length in RUNTIME_INPUT_LENGTHS
    }


def _combined_enrollment_records(
    historical: corpus_data.RawCorpus,
    current: corpus_data.RawCorpus,
) -> dict[int, list[corpus_data.RowRef]]:
    populations = [
        _role_records(historical, "enrollment"),
        _role_records(current, "enrollment"),
    ]
    return {
        length: [
            row
            for population in populations
            for row in population.get(length, [])
        ]
        for length in RUNTIME_INPUT_LENGTHS
    }


def _combined_training_raw_and_groups(
    historical: corpus_data.RawCorpus,
    current: corpus_data.RawCorpus,
) -> tuple[
    dict[int, list[np.ndarray]],
    dict[int, list[str]],
    dict[int, list[str]],
    dict[int, list[str]],
]:
    raw: dict[int, list[np.ndarray]] = {
        length: [] for length in RUNTIME_INPUT_LENGTHS
    }
    groups: dict[int, list[str]] = {
        length: [] for length in RUNTIME_INPUT_LENGTHS
    }
    classes: dict[int, list[str]] = {
        length: [] for length in RUNTIME_INPUT_LENGTHS
    }
    identities: dict[int, list[str]] = {
        length: [] for length in RUNTIME_INPUT_LENGTHS
    }
    for corpus in (historical, current):
        for row in corpus.rows_by_role["train"]:
            minimum = corpus.prefix(row, corpus_data.MINIMUM_INPUT_LENGTH)
            if float(np.max(np.abs(minimum))) == 0.0:
                continue
            group = f"{row.source}:{row.profile_id}"
            for length in corpus_data.training_view_lengths(
                row.valid_sample_count
            ):
                raw[length].append(
                    np.ascontiguousarray(
                        corpus.prefix(row, length), dtype=np.complex64
                    )
                )
                groups[length].append(group)
                classes[length].append(row.class_name)
                identities[length].append(row.identity)
    if any(not raw[length] for length in RUNTIME_INPUT_LENGTHS):
        raise ValueError("combined training lacks a runtime length")
    return raw, groups, classes, identities


def _pose_feature_matrix(
    rows: Sequence[np.ndarray],
    *,
    patch_length: int,
    target_frac: float,
) -> np.ndarray:
    values = list(rows)
    if not values:
        raise ValueError("pose feature extraction needs at least one row")
    matrix = np.stack(
        [
            _pose_feature_row(
                row,
                patch_length=patch_length,
                target_frac=target_frac,
            )
            for row in values
        ]
    ).astype(np.float64)
    if (
        matrix.shape != (len(values), pose_degeneracy.FEATURE_COUNT)
        or not np.isfinite(matrix).all()
    ):
        raise RuntimeError("pose feature matrix is invalid")
    return matrix


def _balanced_known_weights(
    groups: Sequence[str],
    classes: Sequence[str],
    identities: Sequence[str],
) -> np.ndarray:
    group_values = np.asarray(
        [str(value) for value in groups], dtype=object
    )
    class_values = np.asarray(
        [str(value) for value in classes], dtype=object
    )
    identity_values = np.asarray(
        [str(value) for value in identities], dtype=object
    )
    if (
        group_values.ndim != 1
        or len(group_values) == 0
        or class_values.shape != group_values.shape
        or identity_values.shape != group_values.shape
    ):
        raise ValueError(
            "known class/group/identity weights need aligned nonempty vectors"
        )
    public_classes = sorted(set(str(value) for value in class_values))
    weights = np.zeros(len(group_values), dtype=np.float64)
    for class_name in public_classes:
        class_mask = class_values == class_name
        class_groups = sorted(set(str(value) for value in group_values[class_mask]))
        for group in class_groups:
            group_mask = class_mask & (group_values == group)
            group_identities = sorted(
                set(str(value) for value in identity_values[group_mask])
            )
            for identity in group_identities:
                identity_mask = group_mask & (identity_values == identity)
                weights[identity_mask] = (
                    0.5
                    / len(public_classes)
                    / len(class_groups)
                    / len(group_identities)
                    / int(np.sum(identity_mask))
                )
    if not np.isclose(weights.sum(), 0.5, atol=1e-12):
        raise AssertionError(
            "class/profile/base-balanced known weights lost mass"
        )
    return weights


def _fit_v4_stage_one_model(
    known_features: np.ndarray,
    known_groups: Sequence[str],
    known_classes: Sequence[str],
    known_identities: Sequence[str],
    noise_features: np.ndarray,
    *,
    length: int,
    fitting_seed: int,
) -> noise_prefilter.NoisePrefilter:
    names = tuple(noise_prefilter.PREFILTER_FEATURES)
    known = noise_prefilter.select_features(known_features, names)
    noise = noise_prefilter.select_features(noise_features, names)
    if (
        len(known) != len(known_groups)
        or len(known) != len(known_classes)
        or len(known) != len(known_identities)
    ):
        raise ValueError(
            "known feature/class/group/identity vectors lost alignment"
        )
    known_weights = _balanced_known_weights(
        known_groups, known_classes, known_identities
    )
    noise_weights = np.full(len(noise), 0.5 / len(noise), dtype=np.float64)
    pooled = np.vstack((known, noise)).astype(np.float64)
    labels = np.concatenate(
        (np.zeros(len(known)), np.ones(len(noise)))
    ).astype(np.float64)
    weights = np.concatenate((known_weights, noise_weights))
    mean = noise_prefilter._dot(weights, pooled)
    scale = np.sqrt(
        noise_prefilter._dot(weights, (pooled - mean) ** 2)
    )
    if not np.all(scale > noise_prefilter.SCALE_FLOOR):
        raise ValueError("v4 stage-one feature standardizer is degenerate")
    standardized = (pooled - mean) / scale
    design = np.hstack(
        (
            standardized,
            np.ones((len(standardized), 1), dtype=np.float64),
        )
    )
    beta, diagnostics = noise_prefilter._fit_logistic(
        design,
        labels,
        weights,
        l2=noise_prefilter.L2_PENALTY,
    )
    return noise_prefilter.NoisePrefilter(
        feature_names=names,
        capture_length=int(length),
        mean=mean,
        scale=scale,
        coefficients=beta[:-1],
        intercept=float(beta[-1]),
        threshold_score=float("nan"),
        fit_provenance={
            "schema": "v4-current-source-stage-one-fit-v1",
            "known_population":
                "combined historical+current training views only",
            "known_profile_weighting":
                "0.5 total mass; equal public-class mass; equal source/profile "
                "mass within class; equal base-identity then view mass within "
                "source/profile",
            "known_rows": int(len(known)),
            "known_public_classes":
                sorted(set(str(value) for value in known_classes)),
            "known_source_profile_groups":
                sorted(set(str(value) for value in known_groups)),
            "fitting_noise_population":
                "direct time-domain complex AR noise",
            "fitting_noise_rows": int(len(noise)),
            "fitting_noise_weighting": "0.5 total mass, uniform rows",
            "fitting_seed": int(fitting_seed),
            "fitting_seed_namespace": list(_V4_FITTING_NAMESPACE),
            "fitting_noise_uses_frequency_transform": False,
            "enrollment_rows_used_in_fit": 0,
            "selection_rows_used_in_fit": 0,
            "novelty_design_or_validation_rows_used_in_fit": 0,
            "consumed_test_rows_used_in_fit": 0,
            "v3_frozen_hyperplane_reused": False,
            "v3_reuse_rejection": (
                "current service enrollment diagnostics showed 25.4-27.0% "
                "false gates and zero useful noise recall after safe "
                "recalibration; only the deterministic pose features are reused"
            ),
            **diagnostics,
        },
        threshold_provenance={
            "chosen": False,
            "population": None,
            "note": "fit on v4 enrollment by external policy assembler",
        },
    )


def _stage_one_parameter_sha256(
    *,
    mean: np.ndarray,
    scale: np.ndarray,
    coefficients: np.ndarray,
    intercept: float,
) -> str:
    """Canonical hash for one browser-serialisable pose-score parameter row."""
    digest = hashlib.sha256()
    for name, value in (
        ("mean", mean),
        ("scale", scale),
        ("coefficients", coefficients),
        ("intercept", np.asarray([intercept], dtype=np.float64)),
    ):
        payload = np.ascontiguousarray(
            np.asarray(value, dtype=np.float64)
        ).view(np.uint8)
        digest.update(f"{name}:{len(payload)}:".encode("utf-8"))
        digest.update(payload)
    return digest.hexdigest()


def _stage_one_parameter_set_sha256(
    hashes_by_length: Mapping[int, str],
) -> str:
    digest = hashlib.sha256()
    for length in RUNTIME_INPUT_LENGTHS:
        digest.update(
            f"N{length}:{hashes_by_length[length]}".encode("utf-8")
        )
    return digest.hexdigest()


def fit_v4_stage_one_template(
    historical: corpus_data.RawCorpus,
    current: corpus_data.RawCorpus,
    *,
    patch_length: int,
    target_frac: float,
    fitting_seed: Any = DEFAULT_STAGE_ONE_FIT_SEED,
    fitting_noise_rows: int = 600,
) -> StageOneTemplate:
    """Fit v4 stage-one coefficients on allowed training + fit-only noise."""
    seed = validate_fitting_seed(fitting_seed)
    if isinstance(fitting_noise_rows, bool) or int(fitting_noise_rows) < 2:
        raise ValueError("fitting_noise_rows must be at least two")
    (
        training_raw,
        training_groups,
        training_classes,
        training_identities,
    ) = _combined_training_raw_and_groups(historical, current)
    rng = np.random.default_rng(
        _derived_family_seed(seed, "stage-one-fitting-noise")
    )
    longest_noise = [
        _noise_row(max(RUNTIME_INPUT_LENGTHS), rng)
        for _ in range(int(fitting_noise_rows))
    ]
    models: dict[int, noise_prefilter.NoisePrefilter] = {}
    metadata: dict[int, Mapping[str, Any]] = {}
    bundle_hashes: dict[int, str] = {}
    for length in RUNTIME_INPUT_LENGTHS:
        known_features = _pose_feature_matrix(
            training_raw[length],
            patch_length=patch_length,
            target_frac=target_frac,
        )
        noise_features = _pose_feature_matrix(
            [row[:length] for row in longest_noise],
            patch_length=patch_length,
            target_frac=target_frac,
        )
        model = _fit_v4_stage_one_model(
            known_features,
            training_groups[length],
            training_classes[length],
            training_identities[length],
            noise_features,
            length=length,
            fitting_seed=seed,
        )
        models[length] = model
        metadata[length] = {
            "fit": model.fit_provenance,
            "extractor": pose_degeneracy.pose_degeneracy_metadata(),
        }
        bundle_hashes[length] = _stage_one_parameter_sha256(
            mean=model.mean,
            scale=model.scale,
            coefficients=model.coefficients,
            intercept=model.intercept,
        )
    return StageOneTemplate(
        directory=Path("<v4-in-memory-fit>"),
        models=models,
        set_sha256=_stage_one_parameter_set_sha256(bundle_hashes),
        bundle_sha256=bundle_hashes,
        metadata=metadata,
    )


def load_development_context(
    fusion_dir: str | Path,
    historical_dir: str | Path,
    current_dir: str | Path,
    *,
    device_name: str,
) -> DevelopmentContext:
    """Rebuild the admitted train/enrollment/selection data and fusion."""
    fusion_path = reject_sensitive_path(fusion_dir, "fusion input")
    historical_path = reject_sensitive_path(
        historical_dir, "historical corpus input"
    )
    current_path = reject_sensitive_path(current_dir, "current corpus input")
    if historical_path == current_path:
        raise ValueError("historical and current corpora must differ")
    fusion = browser_export.load_verified_fusion(fusion_path)
    metrics = fusion["metrics"]
    seed = int(
        metrics["run_configuration"]["seed_inherited_from_matched_branches"]
    )
    target_frac = float(metrics["frontend"]["target_frac"])
    patch_length = int(fusion["real_config"]["patch_length"])
    patch_count = int(fusion["real_config"]["patch_count"])

    historical, historical_contract = corpus_data.load_historical_exposed(
        historical_path,
        patch_length=patch_length,
        patch_count=patch_count,
        target_frac=target_frac,
        seed=seed,
    )
    classes = list(historical_contract["classes"])
    if tuple(classes) != PUBLIC_CLASSES:
        raise ValueError(
            f"historical classes must be {list(PUBLIC_CLASSES)}, got {classes}"
        )
    current = corpus_data.load_current_corpus(
        current_path,
        class_index={name: index for index, name in enumerate(classes)},
    )
    data, preprocessing_audit = branch_runner._prepare_data(
        historical,
        current,
        classes=classes,
        patch_length=patch_length,
        patch_count=patch_count,
        target_frac=target_frac,
    )
    rebuilt_audit = {
        "historical": historical.audit,
        "current": current.audit,
        "preprocessing": preprocessing_audit,
        "historical_contract": historical_contract,
        "consumed_test_rows_exposed": 0,
    }
    if rebuilt_audit != metrics.get("data_audit"):
        raise RuntimeError(
            "v4 fusion data audit does not reproduce from the supplied corpora"
        )
    if historical.audit.get("consumed_test_rows_exposed") != 0:
        raise AssertionError("historical loader exposed consumed test rows")

    device = branch_runner.v3_runner.resolve_device(str(device_name))
    branch_runner.v3_runner.seed_everything(seed)
    module = browser_export._load_fusion_module(fusion).to(device).eval()
    return DevelopmentContext(
        fusion=fusion,
        historical=historical,
        current=current,
        data=data,
        preprocessing_audit=preprocessing_audit,
        module=module,
        device=device,
        patch_length=patch_length,
        patch_count=patch_count,
        target_frac=target_frac,
    )


def _embed(
    context: DevelopmentContext,
    packed: np.ndarray,
    features: np.ndarray,
) -> np.ndarray:
    return branch_runner.embed_all(
        context.module, packed, features, context.device
    ).astype(np.float32, copy=False)


def _prototype_route_bank(
    fusion: Mapping[str, Any],
    prototype_source: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
    np.ndarray,
    np.ndarray,
]:
    """Return one trusted-source prototype bank, support mask, and row map."""
    source = validate_prototype_source(prototype_source)
    bank = np.asarray(fusion["prototype_bank"], dtype=np.float32)
    labels = np.asarray(fusion["prototype_labels"], dtype=np.int64)
    groups = list(fusion["prototype_groups"])
    if (
        bank.ndim != 2
        or labels.shape != (len(bank),)
        or len(groups) != len(bank)
    ):
        raise ValueError("profile-bank route inputs lost alignment")
    positions = np.asarray(
        [
            index
            for index, row in enumerate(groups)
            if isinstance(row, Mapping) and row.get("source") == source
        ],
        dtype=np.int64,
    )
    if not len(positions):
        raise ValueError(f"prototype route {source!r} has no bank rows")
    routed_labels = labels[positions]
    support = np.asarray(
        [
            class_index in set(routed_labels.tolist())
            for class_index in range(len(PUBLIC_CLASSES))
        ],
        dtype=bool,
    )
    expected_support = np.asarray(
        EXPECTED_PUBLIC_CLASS_SUPPORT_BY_PROTOTYPE_SOURCE[source],
        dtype=bool,
    )
    if not np.array_equal(support, expected_support):
        raise ValueError(
            f"prototype route {source!r} public-class support changed: "
            f"{support.tolist()}"
        )
    routed_groups = [dict(groups[int(index)]) for index in positions]
    return bank[positions], routed_labels, routed_groups, positions, support


def _public_distances(
    context: DevelopmentContext,
    embeddings: np.ndarray,
    *,
    prototype_source: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = validate_prototype_source(prototype_source)
    _bank, _labels, _groups, _positions, expected_support = (
        _prototype_route_bank(context.fusion, source)
    )
    groups = list(context.fusion["prototype_groups"])
    class_distance, closest, support = (
        browser_export.minimum_source_public_class_distances(
        embeddings,
        context.fusion["prototype_bank"],
        context.fusion["prototype_labels"],
        [
            str(row.get("source"))
            if isinstance(row, Mapping)
            else "<invalid>"
            for row in groups
        ],
        source,
        n_classes=len(PUBLIC_CLASSES),
    )
    )
    if (
        class_distance.shape != (len(embeddings), len(PUBLIC_CLASSES))
        or closest.shape != class_distance.shape
        or support.shape != (len(PUBLIC_CLASSES),)
        or not np.array_equal(support, expected_support)
        or not np.isfinite(class_distance[:, support]).all()
        or not np.isinf(class_distance[:, ~support]).all()
        or not np.all(closest[:, support] >= 0)
        or not np.all(closest[:, ~support] == -1)
    ):
        raise RuntimeError("profile-bank public distances are invalid")
    return (
        class_distance.astype(np.float64),
        closest.astype(np.int64, copy=False),
        support,
    )


def leave_one_base_identity_out_class_distances(
    embeddings: np.ndarray,
    groups: Sequence[tuple[str, str]],
    identities: Sequence[str],
    prototype_bank: np.ndarray,
    prototype_labels: np.ndarray,
    prototype_groups: Sequence[tuple[str, str]],
    supported_public_class_mask: Sequence[bool] | None = None,
) -> np.ndarray:
    """Known calibration distances with complete base-identity exclusion."""
    value = np.asarray(embeddings, dtype=np.float32)
    bank = np.asarray(prototype_bank, dtype=np.float32)
    labels = np.asarray(prototype_labels, dtype=np.int64)
    group_values = [(str(a), str(b)) for a, b in groups]
    identity_values = np.asarray(
        [str(identity) for identity in identities], dtype=object
    )
    metadata_groups = [(str(a), str(b)) for a, b in prototype_groups]
    support = (
        np.asarray(supported_public_class_mask, dtype=bool)
        if supported_public_class_mask is not None
        else np.ones(len(PUBLIC_CLASSES), dtype=bool)
    )
    if (
        value.ndim != 2
        or bank.ndim != 2
        or value.shape[1] != bank.shape[1]
        or len(group_values) != len(value)
        or identity_values.shape != (len(value),)
        or labels.shape != (len(bank),)
        or len(metadata_groups) != len(bank)
        or support.shape != (len(PUBLIC_CLASSES),)
        or not np.any(support)
        or set(labels.tolist())
        != set(np.flatnonzero(support).tolist())
    ):
        raise ValueError("leave-one-base-out inputs have invalid shapes")
    if len(set(metadata_groups)) != len(metadata_groups):
        raise ValueError("prototype source/profile groups repeat")
    group_to_positions: dict[tuple[str, str], np.ndarray] = {
        group: np.asarray(
            [
                index
                for index, current in enumerate(group_values)
                if current == group
            ],
            dtype=np.int64,
        )
        for group in sorted(set(group_values))
    }
    group_to_prototype = {
        group: index for index, group in enumerate(metadata_groups)
    }
    if set(group_to_positions) != set(group_to_prototype):
        raise RuntimeError(
            "reconstructed enrollment groups differ from production bank"
        )
    for group, positions in group_to_positions.items():
        reproduced = value[positions].mean(axis=0)
        stored = bank[group_to_prototype[group]]
        error = float(np.max(np.abs(reproduced - stored)))
        if error > 3e-6:
            raise RuntimeError(
                f"production prototype {group} did not reproduce: {error:g}"
            )

    output = np.empty((len(value), len(PUBLIC_CLASSES)), dtype=np.float64)
    for row_index in range(len(value)):
        group = group_values[row_index]
        group_positions = group_to_positions[group]
        same_identity = (
            identity_values[group_positions] == identity_values[row_index]
        )
        remaining = group_positions[~same_identity]
        own_prototype = group_to_prototype[group]
        adjusted = bank.copy()
        omit_own = len(remaining) == 0
        if not omit_own:
            adjusted[own_prototype] = value[remaining].mean(axis=0)
        prototype_distance = (
            (
                value[row_index][None, :].astype(np.float64)
                - adjusted.astype(np.float64)
            )
            ** 2
        ).sum(axis=1)
        if omit_own:
            prototype_distance[own_prototype] = np.inf
        for class_index in range(len(PUBLIC_CLASSES)):
            if not support[class_index]:
                output[row_index, class_index] = np.inf
                continue
            candidates = prototype_distance[labels == class_index]
            if not len(candidates) or not np.isfinite(candidates).any():
                raise ValueError(
                    "leave-one-base-out bank has no remaining prototype for "
                    f"class {PUBLIC_CLASSES[class_index]!r}"
                )
            output[row_index, class_index] = float(np.min(candidates))
    if (
        not np.isfinite(output[:, support]).all()
        or not np.isinf(output[:, ~support]).all()
    ):
        raise RuntimeError("leave-one-base-out distances are non-finite")
    return output


def _enrollment_distance_inputs(
    context: DevelopmentContext,
) -> dict[int, dict[str, np.ndarray]]:
    embeddings = _embed(context, context.data["xen"], context.data["fen"])
    lengths = np.asarray(context.data["enrollment_lengths"], dtype=np.int64)
    if lengths.shape != (len(embeddings),):
        raise RuntimeError("enrollment length vector lost alignment")
    records_by_length = _combined_enrollment_records(
        context.historical, context.current
    )
    source = np.asarray(context.data["enrollment_sources"], dtype=object)
    profile = np.asarray(context.data["enrollment_profiles"], dtype=object)
    if source.shape != lengths.shape or profile.shape != lengths.shape:
        raise RuntimeError("enrollment source/profile vectors lost alignment")
    for length in RUNTIME_INPUT_LENGTHS:
        mask = lengths == length
        records = records_by_length[length]
        if int(np.sum(mask)) != len(records):
            raise RuntimeError(f"N{length} enrollment records lost alignment")
        expected_source = np.asarray(
            [row.source for row in records], dtype=object
        )
        expected_profile = np.asarray(
            [row.profile_id for row in records], dtype=object
        )
        if (
            not np.array_equal(source[mask], expected_source)
            or not np.array_equal(profile[mask], expected_profile)
        ):
            raise RuntimeError(
                f"N{length} enrollment source/profile ordering changed"
            )

    groups = [
        (str(source[index]), str(profile[index]))
        for index in range(len(embeddings))
    ]
    identities_by_length = {
        length: [row.identity for row in records_by_length[length]]
        for length in RUNTIME_INPUT_LENGTHS
    }
    identities = np.empty(len(embeddings), dtype=object)
    for length in RUNTIME_INPUT_LENGTHS:
        mask = lengths == length
        identities[mask] = np.asarray(
            identities_by_length[length], dtype=object
        )
    observed_sources = set(str(value) for value in source)
    if observed_sources != set(PROTOTYPE_SOURCE_ROUTES):
        raise RuntimeError(
            "combined enrollment source routes changed: "
            f"{sorted(observed_sources)}"
        )
    distances = np.full(
        (len(embeddings), len(PUBLIC_CLASSES)),
        np.inf,
        dtype=np.float64,
    )
    closest = np.full(
        (len(embeddings), len(PUBLIC_CLASSES)), -1, dtype=np.int64
    )
    loo_distances = np.full_like(distances, np.inf)
    support_by_source: dict[str, np.ndarray] = {}
    for prototype_source in PROTOTYPE_SOURCE_ROUTES:
        route_mask = source == prototype_source
        route_positions = np.flatnonzero(route_mask)
        if not len(route_positions):
            raise ValueError(
                f"enrollment has no rows for route {prototype_source!r}"
            )
        current, current_closest, support = _public_distances(
            context,
            embeddings[route_mask],
            prototype_source=prototype_source,
        )
        distances[route_mask] = current
        closest[route_mask] = current_closest
        support_by_source[prototype_source] = support
        bank, labels, routed_metadata, _global, routed_support = (
            _prototype_route_bank(context.fusion, prototype_source)
        )
        if not np.array_equal(support, routed_support):
            raise AssertionError("prototype route support changed mid-evaluation")
        loo_distances[route_mask] = (
            leave_one_base_identity_out_class_distances(
                embeddings[route_mask],
                [groups[int(index)] for index in route_positions],
                [str(identities[int(index)]) for index in route_positions],
                bank,
                labels,
                [
                    (str(row["source"]), str(row["profile"]))
                    for row in routed_metadata
                ],
                supported_public_class_mask=support,
            )
        )

    output: dict[int, dict[str, np.ndarray]] = {}
    for length in RUNTIME_INPUT_LENGTHS:
        mask = lengths == length
        if not np.any(mask):
            raise ValueError(f"combined enrollment has no N{length} rows")
        current = distances[mask]
        current_loo = loo_distances[mask]
        prediction = current.argmin(axis=1).astype(np.int64)
        calibration_prediction = current_loo.argmin(axis=1).astype(np.int64)
        current_sources = np.asarray(source[mask], dtype=object)
        expected_support = np.stack(
            [support_by_source[str(value)] for value in current_sources]
        )
        if (
            not np.isfinite(
                np.where(expected_support, current, 0.0)
            ).all()
            or not np.isinf(current[~expected_support]).all()
            or not np.isfinite(
                np.where(expected_support, current_loo, 0.0)
            ).all()
            or not np.isinf(current_loo[~expected_support]).all()
        ):
            raise RuntimeError(
                f"N{length} routed enrollment distance support is invalid"
            )
        output[length] = {
            "class_distances": current,
            "leave_one_base_identity_out_class_distances": current_loo,
            "prediction": prediction,
            "winning_distance": current[
                np.arange(len(current)), prediction
            ],
            "calibration_winning_distance": current_loo[
                np.arange(len(current_loo)), calibration_prediction
            ],
            "closest_prototype_indices": closest[mask],
            "prototype_sources": current_sources,
            "supported_public_class_mask_by_row": expected_support,
            "groups": np.asarray(
                [
                    f"{row.source}:{row.profile_id}"
                    for row in records_by_length[length]
                ],
                dtype=object,
            ),
            "base_identities": np.asarray(
                [
                    row.identity
                    for row in records_by_length[length]
                ],
                dtype=object,
            ),
        }
    return output


def _class_rank(
    values: np.ndarray,
    predictions: np.ndarray,
    class_calibration: Sequence[Sequence[float]],
    pooled_calibration: Sequence[float],
) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64)
    labels = np.asarray(predictions, dtype=np.int64)
    if scores.shape != labels.shape or scores.ndim != 1:
        raise ValueError("class-rank values and predictions must align")
    output = np.empty(len(scores), dtype=np.float64)
    pooled = np.asarray(pooled_calibration, dtype=np.float64)
    for class_index in range(len(PUBLIC_CLASSES)):
        mask = labels == class_index
        if not np.any(mask):
            continue
        selected = np.asarray(class_calibration[class_index], dtype=np.float64)
        calibration = selected if len(selected) else pooled
        output[mask] = empirical_rank(scores[mask], calibration)
    if not np.isfinite(output).all():
        raise RuntimeError("class-conditional rank produced a non-finite value")
    return output


def _fit_route_length_policy(
    *,
    stage_one: np.ndarray,
    phase_linearity: np.ndarray,
    distances: np.ndarray,
    calibration_distances: np.ndarray,
    enrollment_groups: np.ndarray,
    prototype_source: str,
    composite_budget: float,
) -> dict[str, Any]:
    """Fit one source-route/length policy without a union-head fallback."""
    source = validate_prototype_source(prototype_source)
    stage_one_values = np.asarray(stage_one, dtype=np.float64).reshape(-1)
    phase_linearity_values = np.asarray(
        phase_linearity, dtype=np.float64
    ).reshape(-1)
    production = np.asarray(distances, dtype=np.float64)
    calibration = np.asarray(calibration_distances, dtype=np.float64)
    groups = np.asarray(
        [str(value) for value in enrollment_groups], dtype=object
    )
    if (
        production.ndim != 2
        or production.shape
        != (len(stage_one_values), len(PUBLIC_CLASSES))
        or phase_linearity_values.shape != stage_one_values.shape
        or calibration.shape != production.shape
        or groups.shape != (len(stage_one_values),)
        or len(stage_one_values) == 0
        or not np.isfinite(stage_one_values).all()
        or not np.isfinite(phase_linearity_values).all()
        or np.any(phase_linearity_values < 0.0)
        or np.any(phase_linearity_values > 1.0)
    ):
        raise ValueError(
            f"{source} route enrollment inputs are invalid"
        )
    support = np.asarray(
        [
            np.isfinite(production[:, class_index]).all()
            and np.isfinite(calibration[:, class_index]).all()
            for class_index in range(len(PUBLIC_CLASSES))
        ],
        dtype=bool,
    )
    if (
        not np.any(support)
        or not np.isfinite(production[:, support]).all()
        or not np.isfinite(calibration[:, support]).all()
        or not np.isinf(production[:, ~support]).all()
        or not np.isinf(calibration[:, ~support]).all()
    ):
        raise ValueError(
            f"{source} route public-class support mask is invalid"
        )
    if not np.array_equal(
        support,
        np.asarray(
            EXPECTED_PUBLIC_CLASS_SUPPORT_BY_PROTOTYPE_SOURCE[source],
            dtype=bool,
        ),
    ):
        raise ValueError(f"{source} route public-class support changed")
    production_prediction = production.argmin(axis=1).astype(np.int64)
    prediction = calibration.argmin(axis=1).astype(np.int64)
    if (
        not np.all(support[production_prediction])
        or not np.all(support[prediction])
    ):
        raise AssertionError("unsupported routed class won an argmin")
    production_winning = production[
        np.arange(len(production)), production_prediction
    ]
    calibration_winning = calibration[
        np.arange(len(calibration)), prediction
    ]

    pooled_stage_one = np.sort(stage_one_values)
    stage_one_class_calibration: list[list[float]] = []
    for class_index in range(len(PUBLIC_CLASSES)):
        mask = prediction == class_index
        stage_one_class_calibration.append(
            np.sort(stage_one_values[mask]).tolist()
        )
    stage_one_predicted_class_rank = _class_rank(
        stage_one_values,
        prediction,
        stage_one_class_calibration,
        pooled_stage_one,
    )
    stage_one_pooled_rank = empirical_rank(
        stage_one_values,
        pooled_stage_one,
    )
    stage_one_rank = np.maximum(
        stage_one_predicted_class_rank,
        stage_one_pooled_rank,
    )

    pooled_distance = np.sort(calibration_winning)
    distance_class_calibration: list[list[float]] = []
    class_counts: list[int] = []
    for class_index in range(len(PUBLIC_CLASSES)):
        mask = prediction == class_index
        selected = np.sort(calibration_winning[mask])
        class_counts.append(int(len(selected)))
        distance_class_calibration.append(selected.tolist())
    stage_two_rank = _class_rank(
        calibration_winning,
        prediction,
        distance_class_calibration,
        pooled_distance,
    )
    pooled_phase_linearity = np.sort(phase_linearity_values)
    phase_linearity_class_calibration: list[list[float]] = []
    for class_index in range(len(PUBLIC_CLASSES)):
        mask = prediction == class_index
        phase_linearity_class_calibration.append(
            np.sort(phase_linearity_values[mask]).tolist()
        )
    phase_linearity_rank = _class_rank(
        phase_linearity_values,
        prediction,
        phase_linearity_class_calibration,
        pooled_phase_linearity,
    )
    composite = np.maximum.reduce(
        (stage_one_rank, stage_two_rank, phase_linearity_rank)
    )
    pooled_composite_threshold, pooled_composite_detail = (
        grouped_budget_threshold(
            composite,
            groups,
            pooled_budget=composite_budget,
            group_budget=composite_budget,
        )
    )
    threshold_by_class: list[float] = []
    threshold_detail_by_class: list[dict[str, Any]] = []
    for class_index in range(len(PUBLIC_CLASSES)):
        class_mask = prediction == class_index
        if not np.any(class_mask):
            threshold_by_class.append(pooled_composite_threshold)
            threshold_detail_by_class.append(
                {
                    "fallback": "pooled_composite_threshold",
                    "rows": 0,
                }
            )
            continue
        threshold, detail = grouped_budget_threshold(
            composite[class_mask],
            groups[class_mask],
            pooled_budget=composite_budget,
            group_budget=composite_budget,
        )
        threshold_by_class.append(threshold)
        threshold_detail_by_class.append(detail)
    thresholds = np.asarray(threshold_by_class, dtype=np.float64)[prediction]
    rejected = composite >= thresholds
    return {
        "prototype_source": source,
        "supported_public_class_mask": support.tolist(),
        "enrollment_rows": int(len(stage_one_values)),
        "stage_one": {
            "hard_gate_enabled": False,
            "hard_threshold_fitted": False,
            "score_rank_calibration_by_predicted_class_sorted":
                stage_one_class_calibration,
            "pooled_score_rank_calibration_sorted":
                pooled_stage_one.tolist(),
            "empty_predicted_class_fallback": "pooled_score",
            "predicted_class_rank_conditioning": (
                "prototype_source_route_runtime_length_and_"
                "predicted_public_class"
            ),
            "pooled_rank_conditioning":
                "prototype_source_route_and_runtime_length",
            "pooled_rank_always_evaluated": True,
            "effective_rank_kind": STAGE_ONE_RANK_KIND,
            "effective_rank": (
                "max(predicted_class_rank, route_length_pooled_rank)"
            ),
            "empirical_rank_rule": EMPIRICAL_RANK_RULE,
        },
        "stage_two": {
            "predicted_class_distance_calibration_sorted":
                distance_class_calibration,
            "predicted_class_calibration_counts": class_counts,
            "pooled_distance_calibration_sorted":
                pooled_distance.tolist(),
            "empty_predicted_class_fallback": "pooled_distance",
            "empirical_rank_rule": EMPIRICAL_RANK_RULE,
            "calibration_distance_rule": (
                "leave-one-base-identity-out prediction and winning distance "
                "within the trusted prototype-source route after excluding "
                "every view of the row's base identity from its own profile "
                "centroid"
            ),
            "union_head_used": False,
            "self_included_prediction_change_rate":
                float(np.mean(prediction != production_prediction)),
            "production_bank_self_included_distance_median":
                float(np.median(production_winning)),
            "leave_one_base_identity_out_distance_median":
                float(np.median(calibration_winning)),
        },
        "stage_chirp": {
            "score_rank_calibration_by_predicted_class_sorted":
                phase_linearity_class_calibration,
            "pooled_score_rank_calibration_sorted":
                pooled_phase_linearity.tolist(),
            "empty_predicted_class_fallback": "pooled_score",
            "empirical_rank_rule": EMPIRICAL_RANK_RULE,
            "rank_conditioning": (
                "prototype_source_route_runtime_length_and_"
                "predicted_public_class"
            ),
        },
        "composition": {
            "score": (
                "max(effective_stage_one_rank, stage_two_rank, "
                "phase_linearity_rank)"
            ),
            "threshold_by_predicted_public_class": threshold_by_class,
            "threshold_detail_by_predicted_public_class":
                threshold_detail_by_class,
            "pooled_fallback_threshold": pooled_composite_threshold,
            "pooled_fallback_threshold_detail": pooled_composite_detail,
            "known_false_positive_budget": float(composite_budget),
            "threshold_conditioning": (
                "prototype_source_route_runtime_length_and_"
                "predicted_public_class"
            ),
            "threshold_calibration_rows": {
                "stage_one_score": stage_one_values.tolist(),
                "instantaneous_phase_linearity_score":
                    phase_linearity_values.tolist(),
                "leave_one_base_identity_out_winning_distance":
                    calibration_winning.tolist(),
                "predicted_public_class_index": prediction.tolist(),
                "source_profile_group": groups.astype(str).tolist(),
            },
        },
        "enrollment_attribution": {
            "stage_one_gated": 0,
            "stage_two_gated": int(np.sum(rejected)),
            "accepted": int(np.sum(~rejected)),
            "overall_false_unknown_rate": float(np.mean(rejected)),
        },
    }


def fit_enrollment_policy(
    *,
    stage_one_score_by_length: Mapping[int, np.ndarray],
    phase_linearity_score_by_length: Mapping[int, np.ndarray],
    class_distance_by_length: Mapping[int, np.ndarray],
    calibration_class_distance_by_length: Mapping[int, np.ndarray],
    enrollment_group_by_length: Mapping[int, Sequence[str]],
    enrollment_source_by_length: Mapping[int, Sequence[str]],
    stage_one_template: StageOneTemplate,
    fusion_binding: Mapping[str, Any],
    patch_length: int,
    patch_count: int,
    target_frac: float,
    design_seed: int,
    validation_protocol: Mapping[str, Any],
    composite_budget: float = DEFAULT_COMPOSITE_KNOWN_BUDGET,
) -> dict[str, Any]:
    """Fit all *new* ranks and thresholds from enrollment inputs only.

    The intentionally narrow signature is the split firewall: there is no
    parameter for selection, novelty, release, or consumed-test rows.
    """
    per_prototype_source: dict[str, dict[str, Any]] = {
        source: {
            "supported_public_class_mask": None,
            "per_length": {},
        }
        for source in PROTOTYPE_SOURCE_ROUTES
    }
    total_rows = 0
    total_stage_one_gated = 0
    total_stage_two_gated = 0
    for length in RUNTIME_INPUT_LENGTHS:
        stage_one = np.asarray(
            stage_one_score_by_length[length], dtype=np.float64
        ).reshape(-1)
        phase_linearity = np.asarray(
            phase_linearity_score_by_length[length], dtype=np.float64
        ).reshape(-1)
        distances = np.asarray(
            class_distance_by_length[length], dtype=np.float64
        )
        calibration_distances = np.asarray(
            calibration_class_distance_by_length[length], dtype=np.float64
        )
        enrollment_groups = np.asarray(
            [str(value) for value in enrollment_group_by_length[length]],
            dtype=object,
        )
        enrollment_sources = np.asarray(
            [
                validate_prototype_source(value)
                for value in enrollment_source_by_length[length]
            ],
            dtype=object,
        )
        if (
            distances.ndim != 2
            or distances.shape != (len(stage_one), len(PUBLIC_CLASSES))
            or phase_linearity.shape != stage_one.shape
            or calibration_distances.shape != distances.shape
            or enrollment_groups.shape != (len(stage_one),)
            or enrollment_sources.shape != (len(stage_one),)
            or len(stage_one) == 0
            or not np.isfinite(stage_one).all()
            or not np.isfinite(phase_linearity).all()
            or np.any(phase_linearity < 0.0)
            or np.any(phase_linearity > 1.0)
            or set(str(value) for value in enrollment_sources)
            != set(PROTOTYPE_SOURCE_ROUTES)
        ):
            raise ValueError(f"N{length} enrollment inputs are invalid")
        for prototype_source in PROTOTYPE_SOURCE_ROUTES:
            if length not in _runtime_policy_lengths(prototype_source):
                continue
            route_mask = enrollment_sources == prototype_source
            row = _fit_route_length_policy(
                stage_one=stage_one[route_mask],
                phase_linearity=phase_linearity[route_mask],
                distances=distances[route_mask],
                calibration_distances=calibration_distances[route_mask],
                enrollment_groups=enrollment_groups[route_mask],
                prototype_source=prototype_source,
                composite_budget=composite_budget,
            )
            source_policy = per_prototype_source[prototype_source]
            support = row.pop("supported_public_class_mask")
            if source_policy["supported_public_class_mask"] is None:
                source_policy["supported_public_class_mask"] = support
            elif source_policy["supported_public_class_mask"] != support:
                raise RuntimeError(
                    f"{prototype_source} public-class support changed by length"
                )
            source_policy["per_length"][str(length)] = row
            total_rows += int(row["enrollment_rows"])
            total_stage_two_gated += int(
                row["enrollment_attribution"]["stage_two_gated"]
            )
    if any(
        row["supported_public_class_mask"] is None
        or set(row["per_length"])
        != {
            str(length)
            for length in _runtime_policy_lengths(prototype_source)
        }
        for prototype_source, row in per_prototype_source.items()
    ):
        raise AssertionError("source-conditioned policy fit is incomplete")

    return {
        "schema": POLICY_SCHEMA,
        "schema_version": POLICY_SCHEMA_VERSION,
        "status": "development_external_policy",
        "development_only": True,
        "release_evidence": False,
        "runtime_role": "external_abstention_policy",
        "classifier_runtime_role_required": "accepted_known_classifier",
        "classifier_binding": dict(fusion_binding),
        "public_classes": list(PUBLIC_CLASSES),
        "prototype_source_routes": list(PROTOTYPE_SOURCE_ROUTES),
        "acquisition_routing":
            dict(fusion_binding["classifier_routing"]),
        "runtime_input_lengths": list(RUNTIME_INPUT_LENGTHS),
        "runtime_bucket_rule":
            browser_export.RUNTIME_BUCKET_RULE,
        "runtime_length_policy_by_prototype_source": _jsonable(
            RUNTIME_LENGTH_POLICY_BY_PROTOTYPE_SOURCE
        ),
        "frontend": {
            "version": td_preprocess.PREPROCESS_VERSION,
            "patch_length": int(patch_length),
            "patch_count": int(patch_count),
            "target_frac": float(target_frac),
            "uses_frequency_transform": False,
        },
        "stage_zero": {
            "kind": STAGE_ZERO_KIND,
            "decision": STAGE_ZERO_DECISION,
            "effective_prefix_scope": STAGE_ZERO_PREFIX_SCOPE,
            "runs_before_pose_estimation": True,
            "invariant_to_nonzero_global_scale": True,
            "invariant_to_global_phase": True,
            "learned_parameters": 0,
        },
        "stage_one": {
            "kind": STAGE_ONE_KIND,
            "decision": (
                "never hard-gates; after classification contributes the "
                "maximum of predicted-class-conditional and route-length-"
                "pooled enrollment ranks to the final composite"
            ),
            "effective_rank_kind": STAGE_ONE_RANK_KIND,
            "effective_rank": (
                "max(predicted_class_rank, route_length_pooled_rank)"
            ),
            "pooled_rank_always_evaluated": True,
            "coefficient_artifact_set_sha256":
                stage_one_template.set_sha256,
            "coefficient_artifact_by_length_sha256": {
                str(length): stage_one_template.bundle_sha256[length]
                for length in RUNTIME_INPUT_LENGTHS
            },
            "coefficient_fit_population": (
                "combined historical+current training views, hierarchically "
                "balanced by public class/source-profile/base identity, plus "
                "fit-only direct time-domain AR noise"
            ),
            "coefficient_fit_provenance_by_length": {
                str(length): dict(
                    stage_one_template.models[length].fit_provenance
                )
                for length in RUNTIME_INPUT_LENGTHS
            },
            "v3_frozen_hyperplanes_reused": False,
            "v3_reuse_disposition": (
                "rejected after current enrollment false-gate and noise-recall "
                "audit; deterministic pose features only are reused"
            ),
            "source_thresholds_reused": False,
            "feature_names":
                list(noise_prefilter.PREFILTER_FEATURES),
            "extractor": pose_degeneracy.pose_degeneracy_metadata(),
            "uses_frequency_transform": False,
            "uses_absolute_scale": False,
            "invariant_to_nonzero_global_scale": True,
            "invariant_to_global_phase": True,
            "gates_before_classification": False,
            "parameters_by_length": {
                str(length): {
                    "mean":
                        stage_one_template.models[length].mean.tolist(),
                    "scale":
                        stage_one_template.models[length].scale.tolist(),
                    "coefficients":
                        stage_one_template.models[length].coefficients.tolist(),
                    "intercept":
                        float(stage_one_template.models[length].intercept),
                }
                for length in RUNTIME_INPUT_LENGTHS
            },
        },
        "stage_two": {
            "kind": STAGE_TWO_KIND,
            "input": (
                "winning finite value among the classifier's routed public-"
                "class squared distances; unsupported classes remain infinity"
            ),
            "rank_conditioning": (
                "prototype_source_route_runtime_length_and_"
                "predicted_public_class"
            ),
            "rank_fit_population":
                "matching-source v4 enrollment rows only",
            "calibration_centroid_rule":
                "leave one complete base identity out of its source/profile "
                "centroid; production bank is unchanged for runtime queries",
            "self_included_production_distance_used_for_calibration": False,
            "union_head_used": False,
            "uses_frequency_transform": False,
        },
        "stage_chirp": {
            "kind": STAGE_CHIRP_KIND,
            "input": (
                "weighted R-squared of unwrapped adjacent-product phase "
                "against normalized time over admitted contiguous runs"
            ),
            "active_rms_floor": CHIRP_ACTIVE_RMS_FLOOR,
            "minimum_contiguous_run": CHIRP_MIN_CONTIGUOUS_RUN,
            "minimum_valid_adjacencies": CHIRP_MIN_VALID_ADJACENCIES,
            "minimum_valid_fraction": CHIRP_MIN_VALID_FRACTION,
            "weight": "min(r[n],r[n+1])^2 clipped above at 16",
            "weight_ceiling": CHIRP_WEIGHT_CEILING,
            "per_run_intercept_centering": True,
            "normalized_time": "adjacent_index/(N-1)",
            "cw_phase_standard_deviation_floor_rad":
                CHIRP_PHASE_STD_FLOOR_RAD,
            "score": "clamp(XY^2/(XX*YY),0,1)",
            "rank_conditioning": (
                "prototype_source_route_runtime_length_and_"
                "predicted_public_class"
            ),
            "rank_fit_population":
                "matching-source v4 enrollment rows only",
            "uses_frequency_transform": False,
            "uses_absolute_scale": False,
            "invariant_to_nonzero_global_scale": True,
            "invariant_to_global_phase": True,
            "invariant_to_constant_carrier_phase_increment": True,
            "invariant_to_nonzero_linear_time_scaling": True,
            "length_normalized": True,
        },
        "composition": {
            "kind": COMPOSITION_KIND,
            "stage_one_gate_precedes_classifier": False,
            "score": (
                "max(effective_stage_one_rank, stage_two_rank, "
                "phase_linearity_rank)"
            ),
            "threshold_fit_population":
                "matching-source v4 enrollment rows only",
            "conditioned_on_prototype_source_route": True,
        },
        "per_prototype_source": per_prototype_source,
        "fitting_contract": {
            "underlying_fusion_fit": (
                "training fits encoder/centers/features; enrollment fits "
                "source/profile prototype bank"
            ),
            "stage_one_coefficient_fit_population":
                "combined historical+current training plus fit-only synthetic "
                "noise; hierarchical class/profile/base-balanced known weights",
            "hard_stage_one_threshold_fitted": False,
            "new_stage_one_rank_population":
                "source-route-separated historical/current enrollment",
            "new_stage_one_rank_conditioning": (
                "maximum of predicted-public-class and route-length-pooled "
                "empirical ranks"
            ),
            "new_stage_two_rank_population":
                "source-route-separated historical/current enrollment with "
                "leave-one-base-identity-out calibration distances",
            "new_phase_linearity_rank_population":
                "source-route-separated historical/current enrollment",
            "new_composite_threshold_population":
                "source-route-separated historical/current enrollment",
            "union_head_rows_used": 0,
            "training_rows_directly_read_by_threshold_and_rank_fit_function": 0,
            "selection_rows_used": 0,
            "current_selection_rows_used": 0,
            "novelty_rows_used": 0,
            "sealed_release_rows_used": 0,
            "consumed_historical_test_rows_used": 0,
            "enrollment_rows": total_rows,
            "enrollment_stage_one_gated": total_stage_one_gated,
            "enrollment_stage_two_gated": total_stage_two_gated,
        },
        "design_protocol": {
            "design_novelty_seed": int(design_seed),
            "novelty_used_to_fit_or_calibrate": False,
            "selection_used_to_fit_or_calibrate": False,
            "design_novelty_role": "evaluation and policy audit only",
        },
        "validation_protocol": _validate_validation_protocol_record(
            validation_protocol,
            classifier_asset_sha256=str(
                fusion_binding["browser_classifier_asset_sha256"]
            ),
        ),
        "browser_contract": {
            "json_only": True,
            "external_to_classifier_asset": True,
            "classifier_schema_required": browser_export.BROWSER_SCHEMA,
            "classifier_schema_version_required":
                browser_export.BROWSER_SCHEMA_VERSION,
            "empirical_rank_rule": EMPIRICAL_RANK_RULE,
            "runtime_comparison_margin": RUNTIME_COMPARISON_MARGIN,
            "prototype_source_option_required": True,
            "prototype_source_returned": True,
            "union_head_fallback_forbidden": True,
            "runtime_bucket_examples_by_prototype_source": {
                "historical": {
                    "valid_sample_count_12160": 8_192,
                    "valid_sample_count_16384": 16_384,
                },
                "current": {
                    "valid_sample_count_4096": 4_096,
                    "valid_sample_count_12160": 4_096,
                    "valid_sample_count_16384": 4_096,
                },
            },
            "float_comparison": {
                "stage_zero": "max_abs == 0",
                "stage_one": "disabled; no hard decision",
                "stage_two":
                    "composite >= predicted-class-specific threshold",
                "stage_chirp":
                    "analytic score contributes an enrollment empirical rank",
            },
        },
    }


def build_classifier_binding(
    fusion: Mapping[str, Any],
    classifier_asset: str | Path,
) -> dict[str, Any]:
    """Bind the external policy to a verified fusion and exact browser JSON."""
    metrics_path = Path(fusion["metrics_path"])
    binding: dict[str, Any] = {
        "kind": "v4-development-fusion",
        "source_fusion_schema":
            fusion["metrics"]["run_configuration"]["schema"],
        "source_fusion_dev_metrics_sha256": _sha256(metrics_path),
        "source_fusion_artifacts_sha256": dict(
            sorted(fusion["hashes"].items())
        ),
        "classifier_schema_required": browser_export.BROWSER_SCHEMA,
        "classifier_schema_version_required":
            browser_export.BROWSER_SCHEMA_VERSION,
        "classifier_runtime_role_required": "accepted_known_classifier",
        "browser_frontend": None,
        "browser_classifier_asset_sha256": None,
        "browser_classifier_asset_bound": False,
    }
    asset_path = reject_sensitive_path(
        classifier_asset, "browser classifier input"
    )
    asset = _load_json(asset_path)
    required = {
        "schema": browser_export.BROWSER_SCHEMA,
        "schema_version": browser_export.BROWSER_SCHEMA_VERSION,
        "development_only": True,
        "runtime_role": "accepted_known_classifier",
    }
    for key, expected in required.items():
        if asset.get(key) != expected:
            raise ValueError(
                f"browser classifier {key} must be {expected!r}, "
                f"got {asset.get(key)!r}"
            )
    classification = asset.get("classification")
    if (
        not isinstance(classification, Mapping)
        or tuple(classification.get("classes", ())) != PUBLIC_CLASSES
    ):
        raise ValueError("browser classifier public classes changed")
    routing = classification.get("routing")
    if (
        not isinstance(routing, Mapping)
        or routing != browser_export.TRUSTED_SOURCE_ROUTING
    ):
        raise ValueError(
            "browser classifier trusted acquisition routing changed"
        )
    prototype_bank = classification.get("prototype_bank")
    prototype_labels = classification.get(
        "prototype_public_class_indices"
    )
    prototype_groups = classification.get("prototype_groups")
    if (
        not isinstance(prototype_bank, list)
        or not prototype_bank
        or not isinstance(prototype_labels, list)
        or not isinstance(prototype_groups, list)
        or len(prototype_bank) != len(prototype_labels)
        or len(prototype_bank) != len(prototype_groups)
    ):
        raise ValueError(
            "browser classifier prototype rows, labels, and groups must align"
        )
    support = {
        source: [False] * len(PUBLIC_CLASSES)
        for source in PROTOTYPE_SOURCE_ROUTES
    }
    current_profiles: set[str] = set()
    for row_index, (label, group) in enumerate(
        zip(prototype_labels, prototype_groups)
    ):
        if (
            isinstance(label, (bool, np.bool_))
            or not isinstance(label, (int, np.integer))
            or not 0 <= int(label) < len(PUBLIC_CLASSES)
            or not isinstance(group, Mapping)
        ):
            raise ValueError(
                f"browser classifier prototype row {row_index} is invalid"
            )
        source = str(group.get("source"))
        if source not in support:
            raise ValueError(
                f"browser classifier prototype row {row_index} has "
                f"unsupported source {source!r}"
            )
        public_class = PUBLIC_CLASSES[int(label)]
        if group.get("public_class") != public_class:
            raise ValueError(
                f"browser classifier prototype row {row_index} class mapping "
                "disagrees with its label"
            )
        support[source][int(label)] = True
        if source == "current":
            profile = group.get("profile")
            if (
                not isinstance(profile, str)
                or corpus_data.CURRENT_PROFILE_PUBLIC_CLASS_MAP.get(profile)
                != public_class
            ):
                raise ValueError(
                    f"browser classifier current prototype row {row_index} "
                    "is outside the canonical profile inventory"
                )
            current_profiles.add(profile)
    if current_profiles != set(corpus_data.CURRENT_PROFILE_PUBLIC_CLASS_MAP):
        raise ValueError(
            "browser classifier current prototype groups do not cover the "
            "canonical profile inventory"
        )
    expected_support = {
        source: list(
            EXPECTED_PUBLIC_CLASS_SUPPORT_BY_PROTOTYPE_SOURCE[source]
        )
        for source in PROTOTYPE_SOURCE_ROUTES
    }
    if support != expected_support:
        raise ValueError(
            "browser classifier prototype-source public-class support changed"
        )
    asset_frontend = asset.get("frontend")
    expected_frontend = {
        "version": td_preprocess.PREPROCESS_VERSION,
        "patch_length": int(fusion["real_config"]["patch_length"]),
        "patch_count": int(fusion["real_config"]["patch_count"]),
        "target_frac": float(fusion["metrics"]["frontend"]["target_frac"]),
        "uses_frequency_transform": False,
        "runtime_input_lengths": list(RUNTIME_INPUT_LENGTHS),
        "runtime_bucket_rule": browser_export.RUNTIME_BUCKET_RULE,
    }
    if (
        not isinstance(asset_frontend, Mapping)
        or any(
            asset_frontend.get(key) != expected
            for key, expected in expected_frontend.items()
        )
    ):
        raise ValueError("browser classifier frontend differs from fusion")
    provenance = asset.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("source_fusion_dev_metrics_sha256")
        != binding["source_fusion_dev_metrics_sha256"]
        or provenance.get("source_fusion_artifacts_sha256")
        != binding["source_fusion_artifacts_sha256"]
    ):
        raise ValueError("browser classifier is bound to another fusion")
    rejection = asset.get("rejection")
    if (
        not isinstance(rejection, Mapping)
        or rejection.get("state") != "unset"
        or rejection.get("external_policy_required_for_abstention") is not True
    ):
        raise ValueError(
            "browser classifier must delegate abstention to an external policy"
        )
    binding["kind"] = "v4-browser-classifier-json"
    binding["browser_frontend"] = expected_frontend
    binding["classifier_routing"] = _jsonable(routing)
    binding[
        "supported_public_class_mask_by_prototype_source"
    ] = expected_support
    binding["browser_classifier_asset_sha256"] = _sha256(asset_path)
    binding["browser_classifier_asset_bound"] = True
    return binding


def _finite_sorted_vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if (
        array.ndim != 1
        or len(array) == 0
        or not np.isfinite(array).all()
        or (len(array) > 1 and np.any(array[1:] < array[:-1]))
    ):
        raise ValueError(f"{name} must be a nonempty finite sorted vector")
    return array


def validate_policy(
    value: Mapping[str, Any],
    *,
    expected_fusion_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed on a malformed, non-browser-safe, or misbound policy."""
    if not isinstance(value, Mapping):
        raise ValueError("open-set policy must be an object")
    required = {
        "schema": POLICY_SCHEMA,
        "schema_version": POLICY_SCHEMA_VERSION,
        "status": "development_external_policy",
        "development_only": True,
        "release_evidence": False,
        "runtime_role": "external_abstention_policy",
        "classifier_runtime_role_required": "accepted_known_classifier",
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise ValueError(
                f"open-set policy {key} must be {expected!r}, "
                f"got {value.get(key)!r}"
            )
    if (
        tuple(value.get("public_classes", ())) != PUBLIC_CLASSES
        or tuple(value.get("prototype_source_routes", ()))
        != PROTOTYPE_SOURCE_ROUTES
        or tuple(value.get("runtime_input_lengths", ()))
        != tuple(RUNTIME_INPUT_LENGTHS)
        or value.get("runtime_bucket_rule")
        != browser_export.RUNTIME_BUCKET_RULE
        or value.get("runtime_length_policy_by_prototype_source")
        != RUNTIME_LENGTH_POLICY_BY_PROTOTYPE_SOURCE
    ):
        raise ValueError(
            "open-set policy public classes, prototype routes, lengths, or "
            "bucket rule changed"
        )
    frontend = value.get("frontend")
    frontend_patch_length = (
        frontend.get("patch_length")
        if isinstance(frontend, Mapping)
        else None
    )
    frontend_patch_count = (
        frontend.get("patch_count")
        if isinstance(frontend, Mapping)
        else None
    )
    frontend_target_frac = (
        frontend.get("target_frac")
        if isinstance(frontend, Mapping)
        else None
    )
    if (
        not isinstance(frontend, Mapping)
        or frontend.get("version") != td_preprocess.PREPROCESS_VERSION
        or frontend.get("uses_frequency_transform") is not False
        or isinstance(frontend_patch_length, (bool, np.bool_))
        or not isinstance(frontend_patch_length, (int, np.integer))
        or int(frontend_patch_length) < 16
        or isinstance(frontend_patch_count, (bool, np.bool_))
        or not isinstance(frontend_patch_count, (int, np.integer))
        or int(frontend_patch_count) <= 0
        or isinstance(frontend_target_frac, (bool, np.bool_))
        or not isinstance(
            frontend_target_frac, (int, float, np.integer, np.floating)
        )
        or not math.isfinite(float(frontend_target_frac))
        or not 0.0 < float(frontend_target_frac) <= 1.0
    ):
        raise ValueError("open-set policy frontend contract is invalid")
    stage_zero = value.get("stage_zero")
    if (
        not isinstance(stage_zero, Mapping)
        or stage_zero.get("kind") != STAGE_ZERO_KIND
        or stage_zero.get("decision") != STAGE_ZERO_DECISION
        or stage_zero.get("effective_prefix_scope")
        != STAGE_ZERO_PREFIX_SCOPE
        or stage_zero.get("runs_before_pose_estimation") is not True
        or stage_zero.get("learned_parameters") != 0
    ):
        raise ValueError("open-set stage zero contract changed")
    stage_one = value.get("stage_one")
    if (
        not isinstance(stage_one, Mapping)
        or stage_one.get("kind") != STAGE_ONE_KIND
        or stage_one.get("v3_frozen_hyperplanes_reused") is not False
        or stage_one.get("source_thresholds_reused") is not False
        or stage_one.get("gates_before_classification") is not False
        or stage_one.get("uses_frequency_transform") is not False
        or stage_one.get("uses_absolute_scale") is not False
        or stage_one.get("effective_rank_kind") != STAGE_ONE_RANK_KIND
        or stage_one.get("effective_rank")
        != "max(predicted_class_rank, route_length_pooled_rank)"
        or stage_one.get("pooled_rank_always_evaluated") is not True
        or stage_one.get("extractor")
        != pose_degeneracy.pose_degeneracy_metadata()
    ):
        raise ValueError("open-set stage-one contract is invalid")
    if tuple(stage_one.get("feature_names", ())) != tuple(
        noise_prefilter.PREFILTER_FEATURES
    ):
        raise ValueError("open-set stage-one feature order changed")
    artifact_set_hash = stage_one.get("coefficient_artifact_set_sha256")
    if not _is_sha256(artifact_set_hash):
        raise ValueError("stage-one coefficient artifact hash is invalid")
    artifact_hashes = stage_one.get(
        "coefficient_artifact_by_length_sha256"
    )
    if (
        not isinstance(artifact_hashes, Mapping)
        or set(artifact_hashes)
        != {str(length) for length in RUNTIME_INPUT_LENGTHS}
        or not all(_is_sha256(item) for item in artifact_hashes.values())
    ):
        raise ValueError(
            "stage-one coefficient artifact-by-length hashes are invalid"
        )
    fit_provenance = stage_one.get(
        "coefficient_fit_provenance_by_length"
    )
    if (
        not isinstance(fit_provenance, Mapping)
        or set(fit_provenance)
        != {str(length) for length in RUNTIME_INPUT_LENGTHS}
    ):
        raise ValueError("stage-one fit provenance is incomplete")
    parameters = stage_one.get("parameters_by_length")
    if not isinstance(parameters, Mapping):
        raise ValueError("stage-one parameters_by_length must be an object")
    width = len(noise_prefilter.PREFILTER_FEATURES)
    reconstructed_hashes: dict[int, str] = {}
    fitting_seeds: set[int] = set()
    for length in RUNTIME_INPUT_LENGTHS:
        provenance = fit_provenance[str(length)]
        known_rows = (
            provenance.get("known_rows")
            if isinstance(provenance, Mapping)
            else None
        )
        fitting_noise_rows = (
            provenance.get("fitting_noise_rows")
            if isinstance(provenance, Mapping)
            else None
        )
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("schema")
            != "v4-current-source-stage-one-fit-v1"
            or provenance.get("v3_frozen_hyperplane_reused") is not False
            or provenance.get("known_public_classes")
            != list(PUBLIC_CLASSES)
            or provenance.get("known_profile_weighting")
            != (
                "0.5 total mass; equal public-class mass; equal source/profile "
                "mass within class; equal base-identity then view mass within "
                "source/profile"
            )
            or provenance.get(
                "fitting_noise_uses_frequency_transform"
            ) is not False
            or provenance.get("enrollment_rows_used_in_fit") != 0
            or provenance.get("selection_rows_used_in_fit") != 0
            or provenance.get(
                "novelty_design_or_validation_rows_used_in_fit"
            ) != 0
            or provenance.get("consumed_test_rows_used_in_fit") != 0
            or isinstance(known_rows, (bool, np.bool_))
            or not isinstance(known_rows, (int, np.integer))
            or int(known_rows) <= 0
            or isinstance(fitting_noise_rows, (bool, np.bool_))
            or not isinstance(fitting_noise_rows, (int, np.integer))
            or int(fitting_noise_rows) <= 0
        ):
            raise ValueError(
                f"stage-one N{length} fit provenance is contaminated"
            )
        fitting_seeds.add(
            validate_fitting_seed(provenance.get("fitting_seed"))
        )
        row = parameters.get(str(length))
        if not isinstance(row, Mapping):
            raise ValueError(f"missing stage-one N{length} parameters")
        for name in ("mean", "scale", "coefficients"):
            vector = np.asarray(row.get(name), dtype=np.float64)
            if vector.shape != (width,) or not np.isfinite(vector).all():
                raise ValueError(f"stage-one N{length} {name} is invalid")
            if name == "scale" and not np.all(vector > 0.0):
                raise ValueError(f"stage-one N{length} scale is nonpositive")
        if not math.isfinite(float(row.get("intercept", float("nan")))):
            raise ValueError(f"stage-one N{length} intercept is invalid")
        reconstructed_hashes[length] = _stage_one_parameter_sha256(
            mean=np.asarray(row["mean"], dtype=np.float64),
            scale=np.asarray(row["scale"], dtype=np.float64),
            coefficients=np.asarray(row["coefficients"], dtype=np.float64),
            intercept=float(row["intercept"]),
        )
        if reconstructed_hashes[length] != artifact_hashes[str(length)]:
            raise ValueError(
                f"stage-one N{length} parameter hash does not match"
            )
    if (
        _stage_one_parameter_set_sha256(reconstructed_hashes)
        != artifact_set_hash
    ):
        raise ValueError("stage-one parameter set hash does not match")
    if len(fitting_seeds) != 1:
        raise ValueError("stage-one lengths used different fitting seeds")
    stage_two = value.get("stage_two")
    if (
        not isinstance(stage_two, Mapping)
        or stage_two.get("kind") != STAGE_TWO_KIND
        or stage_two.get(
            "self_included_production_distance_used_for_calibration"
        ) is not False
        or stage_two.get("union_head_used") is not False
        or stage_two.get("uses_frequency_transform") is not False
    ):
        raise ValueError("open-set stage-two contract changed")
    stage_chirp = value.get("stage_chirp")
    if (
        not isinstance(stage_chirp, Mapping)
        or stage_chirp.get("kind") != STAGE_CHIRP_KIND
        or float(
            stage_chirp.get("active_rms_floor", float("nan"))
        ) != CHIRP_ACTIVE_RMS_FLOOR
        or stage_chirp.get("minimum_contiguous_run")
        != CHIRP_MIN_CONTIGUOUS_RUN
        or stage_chirp.get("minimum_valid_adjacencies")
        != CHIRP_MIN_VALID_ADJACENCIES
        or float(
            stage_chirp.get("minimum_valid_fraction", float("nan"))
        ) != CHIRP_MIN_VALID_FRACTION
        or float(
            stage_chirp.get("weight_ceiling", float("nan"))
        ) != CHIRP_WEIGHT_CEILING
        or stage_chirp.get("per_run_intercept_centering") is not True
        or float(
            stage_chirp.get(
                "cw_phase_standard_deviation_floor_rad",
                float("nan"),
            )
        ) != CHIRP_PHASE_STD_FLOOR_RAD
        or stage_chirp.get("score")
        != "clamp(XY^2/(XX*YY),0,1)"
        or stage_chirp.get("uses_frequency_transform") is not False
        or stage_chirp.get("uses_absolute_scale") is not False
        or stage_chirp.get(
            "invariant_to_nonzero_global_scale"
        ) is not True
        or stage_chirp.get("invariant_to_global_phase") is not True
        or stage_chirp.get(
            "invariant_to_constant_carrier_phase_increment"
        ) is not True
        or stage_chirp.get(
            "invariant_to_nonzero_linear_time_scaling"
        ) is not True
        or stage_chirp.get("length_normalized") is not True
    ):
        raise ValueError("open-set phase-linearity stage changed")
    composition_contract = value.get("composition")
    if (
        not isinstance(composition_contract, Mapping)
        or composition_contract.get("kind") != COMPOSITION_KIND
        or composition_contract.get("stage_one_gate_precedes_classifier")
        is not False
        or composition_contract.get(
            "conditioned_on_prototype_source_route"
        ) is not True
    ):
        raise ValueError("open-set composition contract changed")

    fitting = value.get("fitting_contract")
    forbidden = (
        "selection_rows_used",
        "current_selection_rows_used",
        "novelty_rows_used",
        "sealed_release_rows_used",
        "consumed_historical_test_rows_used",
        "union_head_rows_used",
    )
    if (
        not isinstance(fitting, Mapping)
        or any(fitting.get(key) != 0 for key in forbidden)
    ):
        raise ValueError("open-set policy fitting provenance is contaminated")
    provenance = value.get("provenance")
    provenance_binding = value.get("classifier_binding")
    if (
        not isinstance(provenance, Mapping)
        or not isinstance(provenance_binding, Mapping)
        or provenance.get("source_sha256") != _source_hashes()
        or provenance.get("stage_one_v4_fitted_artifact_set_sha256")
        != artifact_set_hash
        or validate_fitting_seed(
            provenance.get("stage_one_fitting_seed")
        ) not in fitting_seeds
        or provenance.get("fusion_dev_metrics_sha256")
        != provenance_binding.get(
            "source_fusion_dev_metrics_sha256"
        )
        or not _is_sha256(
            provenance.get("historical_data_audit_sha256")
        )
        or not _is_sha256(
            provenance.get("current_data_audit_sha256")
        )
        or provenance.get("consumed_test_rows_exposed") != 0
    ):
        raise ValueError("open-set source/data provenance is invalid")

    per_source = value.get("per_prototype_source")
    if (
        not isinstance(per_source, Mapping)
        or set(per_source) != set(PROTOTYPE_SOURCE_ROUTES)
    ):
        raise ValueError("open-set source-route policy is incomplete")
    route_rows: list[tuple[str, int, Mapping[str, Any], np.ndarray]] = []
    for prototype_source in PROTOTYPE_SOURCE_ROUTES:
        source_policy = per_source[prototype_source]
        expected_support = np.asarray(
            EXPECTED_PUBLIC_CLASS_SUPPORT_BY_PROTOTYPE_SOURCE[
                prototype_source
            ],
            dtype=bool,
        )
        if (
            not isinstance(source_policy, Mapping)
            or source_policy.get("supported_public_class_mask")
            != expected_support.tolist()
        ):
            raise ValueError(
                f"open-set {prototype_source} support mask changed"
            )
        per_length = source_policy.get("per_length")
        if (
            not isinstance(per_length, Mapping)
            or set(per_length)
            != {
                str(length)
                for length in _runtime_policy_lengths(prototype_source)
            }
        ):
            raise ValueError(
                f"open-set {prototype_source} per-length policy is incomplete"
            )
        for length in _runtime_policy_lengths(prototype_source):
            route_rows.append(
                (
                    prototype_source,
                    length,
                    per_length[str(length)],
                    expected_support,
                )
            )
    for prototype_source, length, row, supported_mask in route_rows:
        if not isinstance(row, Mapping):
            raise ValueError(
                f"open-set {prototype_source} N{length} policy must be an object"
            )
        if row.get("prototype_source") != prototype_source:
            raise ValueError(
                f"open-set {prototype_source} N{length} route changed"
            )
        stage_one_row = row.get("stage_one")
        stage_two_row = row.get("stage_two")
        stage_chirp_row = row.get("stage_chirp")
        composition = row.get("composition")
        if not all(
            isinstance(item, Mapping)
            for item in (
                stage_one_row,
                stage_two_row,
                stage_chirp_row,
                composition,
            )
        ):
            raise ValueError(f"open-set N{length} stages are incomplete")
        if stage_two_row.get("union_head_used") is not False:
            raise ValueError(
                f"open-set {prototype_source} N{length} used a union head"
            )
        pooled_stage_one = _finite_sorted_vector(
            stage_one_row.get(
                "pooled_score_rank_calibration_sorted"
            ),
            f"N{length} stage-one pooled calibration",
        )
        stage_one_classes = stage_one_row.get(
            "score_rank_calibration_by_predicted_class_sorted"
        )
        if (
            not isinstance(stage_one_classes, list)
            or len(stage_one_classes) != len(PUBLIC_CLASSES)
            or stage_one_row.get("empty_predicted_class_fallback")
            != "pooled_score"
            or stage_one_row.get("predicted_class_rank_conditioning")
            != (
                "prototype_source_route_runtime_length_and_"
                "predicted_public_class"
            )
            or stage_one_row.get("pooled_rank_conditioning")
            != "prototype_source_route_and_runtime_length"
            or stage_one_row.get("pooled_rank_always_evaluated") is not True
            or stage_one_row.get("effective_rank_kind")
            != STAGE_ONE_RANK_KIND
            or stage_one_row.get("effective_rank")
            != "max(predicted_class_rank, route_length_pooled_rank)"
            or stage_one_row.get("empirical_rank_rule")
            != EMPIRICAL_RANK_RULE
        ):
            raise ValueError(
                f"N{length} stage-one class calibration invalid"
            )
        for class_index, calibration in enumerate(stage_one_classes):
            array = np.asarray(calibration, dtype=np.float64)
            if (
                array.ndim != 1
                or not np.isfinite(array).all()
                or (len(array) > 1 and np.any(array[1:] < array[:-1]))
            ):
                raise ValueError(
                    f"N{length} stage-one class {class_index} calibration invalid"
                )
        if not len(pooled_stage_one):
            raise AssertionError("stage-one pooled calibration became empty")
        if (
            stage_one_row.get("hard_gate_enabled") is not False
            or stage_one_row.get("hard_threshold_fitted") is not False
        ):
            raise ValueError(f"N{length} stage-one hard gate must be disabled")
        pooled = _finite_sorted_vector(
            stage_two_row.get(
                "pooled_distance_calibration_sorted"
            ),
            f"N{length} stage-two pooled calibration",
        )
        classes = stage_two_row.get(
            "predicted_class_distance_calibration_sorted"
        )
        if not isinstance(classes, list) or len(classes) != len(PUBLIC_CLASSES):
            raise ValueError(f"N{length} stage-two class calibration invalid")
        for class_index, calibration in enumerate(classes):
            array = np.asarray(calibration, dtype=np.float64)
            if array.ndim != 1 or not np.isfinite(array).all():
                raise ValueError(
                    f"N{length} class {class_index} calibration invalid"
                )
            if len(array) > 1 and np.any(array[1:] < array[:-1]):
                raise ValueError(
                    f"N{length} class {class_index} calibration unsorted"
                )
        if not len(pooled):
            raise AssertionError("pooled validation lost nonempty assertion")
        pooled_phase_curvature = _finite_sorted_vector(
            stage_chirp_row.get(
                "pooled_score_rank_calibration_sorted"
            ),
            f"N{length} phase-linearity pooled calibration",
        )
        phase_curvature_classes = stage_chirp_row.get(
            "score_rank_calibration_by_predicted_class_sorted"
        )
        if (
            not isinstance(phase_curvature_classes, list)
            or len(phase_curvature_classes) != len(PUBLIC_CLASSES)
            or stage_chirp_row.get("empty_predicted_class_fallback")
            != "pooled_score"
            or stage_chirp_row.get("empirical_rank_rule")
            != EMPIRICAL_RANK_RULE
        ):
            raise ValueError(
                f"N{length} phase-linearity class calibration invalid"
            )
        for class_index, calibration in enumerate(
            phase_curvature_classes
        ):
            array = np.asarray(calibration, dtype=np.float64)
            if (
                array.ndim != 1
                or not np.isfinite(array).all()
                or np.any(array < 0.0)
                or np.any(array > 1.0)
                or (len(array) > 1 and np.any(array[1:] < array[:-1]))
            ):
                raise ValueError(
                    f"N{length} phase-linearity class {class_index} "
                    "calibration invalid"
                )
        if np.any(pooled_phase_curvature < 0.0) or np.any(
            pooled_phase_curvature > 1.0
        ):
            raise ValueError(
                f"N{length} phase-linearity pooled calibration escaped [0, 1]"
            )
        thresholds = np.asarray(
            composition.get("threshold_by_predicted_public_class"),
            dtype=np.float64,
        )
        if (
            thresholds.shape != (len(PUBLIC_CLASSES),)
            or not np.isfinite(thresholds).all()
            or np.any(thresholds < 0.0)
            or np.any(thresholds > 1.0)
        ):
            raise ValueError(
                f"N{length} composite class thresholds are invalid"
            )
        enrollment_rows = row.get("enrollment_rows")
        class_counts_raw = stage_two_row.get(
            "predicted_class_calibration_counts"
        )
        if (
            isinstance(enrollment_rows, (bool, np.bool_))
            or not isinstance(enrollment_rows, (int, np.integer))
            or int(enrollment_rows) <= 0
            or not isinstance(class_counts_raw, list)
            or len(class_counts_raw) != len(PUBLIC_CLASSES)
            or any(
                isinstance(count, (bool, np.bool_))
                or not isinstance(count, (int, np.integer))
                or int(count) < 0
                for count in class_counts_raw
            )
        ):
            raise ValueError(f"N{length} calibration counts are invalid")
        class_counts = np.asarray(class_counts_raw, dtype=np.int64)
        if (
            int(np.sum(class_counts)) != int(enrollment_rows)
            or len(pooled_stage_one) != int(enrollment_rows)
            or len(pooled) != int(enrollment_rows)
            or len(pooled_phase_curvature) != int(enrollment_rows)
            or any(
                len(stage_one_classes[index]) != int(class_counts[index])
                or len(classes[index]) != int(class_counts[index])
                or len(phase_curvature_classes[index])
                != int(class_counts[index])
                for index in range(len(PUBLIC_CLASSES))
            )
        ):
            raise ValueError(f"N{length} calibration counts disagree")
        if np.any(class_counts[~supported_mask] != 0):
            raise ValueError(
                f"N{length} unsupported {prototype_source} classes have "
                "calibration rows"
            )

        calibration_rows = composition.get("threshold_calibration_rows")
        if not isinstance(calibration_rows, Mapping):
            raise ValueError(f"N{length} threshold calibration rows missing")
        raw_stage_one = np.asarray(
            calibration_rows.get("stage_one_score"), dtype=np.float64
        )
        raw_distance = np.asarray(
            calibration_rows.get(
                "leave_one_base_identity_out_winning_distance"
            ),
            dtype=np.float64,
        )
        raw_phase_curvature = np.asarray(
            calibration_rows.get(
                "instantaneous_phase_linearity_score"
            ),
            dtype=np.float64,
        )
        raw_prediction_float = np.asarray(
            calibration_rows.get("predicted_public_class_index"),
            dtype=np.float64,
        )
        raw_groups = np.asarray(
            calibration_rows.get("source_profile_group"), dtype=object
        )
        if (
            raw_stage_one.shape != (int(enrollment_rows),)
            or raw_distance.shape != raw_stage_one.shape
            or raw_phase_curvature.shape != raw_stage_one.shape
            or raw_prediction_float.shape != raw_stage_one.shape
            or raw_groups.shape != raw_stage_one.shape
            or not np.isfinite(raw_stage_one).all()
            or not np.isfinite(raw_distance).all()
            or not np.isfinite(raw_phase_curvature).all()
            or np.any(raw_phase_curvature < 0.0)
            or np.any(raw_phase_curvature > 1.0)
            or not np.isfinite(raw_prediction_float).all()
            or np.any(raw_prediction_float != np.floor(raw_prediction_float))
            or np.any(raw_prediction_float < 0)
            or np.any(raw_prediction_float >= len(PUBLIC_CLASSES))
            or any(not isinstance(group, str) or not group for group in raw_groups)
            or any(
                not str(group).startswith(f"{prototype_source}:")
                for group in raw_groups
            )
        ):
            raise ValueError(
                f"N{length} threshold calibration rows are invalid"
            )
        raw_prediction = raw_prediction_float.astype(np.int64)
        if not np.all(supported_mask[raw_prediction]):
            raise ValueError(
                f"N{length} unsupported {prototype_source} class was predicted"
            )
        if (
            not np.array_equal(np.sort(raw_stage_one), pooled_stage_one)
            or not np.array_equal(np.sort(raw_distance), pooled)
            or not np.array_equal(
                np.sort(raw_phase_curvature),
                pooled_phase_curvature,
            )
        ):
            raise ValueError(
                f"N{length} pooled calibration does not reproduce"
            )
        for class_index in range(len(PUBLIC_CLASSES)):
            mask = raw_prediction == class_index
            if (
                not np.array_equal(
                    np.sort(raw_stage_one[mask]),
                    np.asarray(
                        stage_one_classes[class_index], dtype=np.float64
                    ),
                )
                or not np.array_equal(
                    np.sort(raw_distance[mask]),
                    np.asarray(classes[class_index], dtype=np.float64),
                )
                or not np.array_equal(
                    np.sort(raw_phase_curvature[mask]),
                    np.asarray(
                        phase_curvature_classes[class_index],
                        dtype=np.float64,
                    ),
                )
            ):
                raise ValueError(
                    f"N{length} class calibration does not reproduce"
                )
        recomputed_stage_one_predicted_class_rank = _class_rank(
            raw_stage_one,
            raw_prediction,
            stage_one_classes,
            pooled_stage_one,
        )
        recomputed_stage_one_pooled_rank = empirical_rank(
            raw_stage_one,
            pooled_stage_one,
        )
        recomputed_stage_one_rank = np.maximum(
            recomputed_stage_one_predicted_class_rank,
            recomputed_stage_one_pooled_rank,
        )
        recomputed_distance_rank = _class_rank(
            raw_distance,
            raw_prediction,
            classes,
            pooled,
        )
        recomputed_phase_curvature_rank = _class_rank(
            raw_phase_curvature,
            raw_prediction,
            phase_curvature_classes,
            pooled_phase_curvature,
        )
        recomputed_composite = np.maximum.reduce(
            (
                recomputed_stage_one_rank,
                recomputed_distance_rank,
                recomputed_phase_curvature_rank,
            )
        )
        composite_budget = float(
            composition.get("known_false_positive_budget", float("nan"))
        )
        if (
            not math.isfinite(composite_budget)
            or not 0.0 <= composite_budget < 1.0
        ):
            raise ValueError(f"N{length} composite budget is invalid")
        pooled_threshold, pooled_detail = grouped_budget_threshold(
            recomputed_composite,
            [str(value) for value in raw_groups],
            pooled_budget=composite_budget,
            group_budget=composite_budget,
        )
        expected_thresholds: list[float] = []
        expected_details: list[dict[str, Any]] = []
        for class_index in range(len(PUBLIC_CLASSES)):
            mask = raw_prediction == class_index
            if not np.any(mask):
                expected_thresholds.append(pooled_threshold)
                expected_details.append(
                    {
                        "fallback": "pooled_composite_threshold",
                        "rows": 0,
                    }
                )
                continue
            class_threshold, class_detail = grouped_budget_threshold(
                recomputed_composite[mask],
                [str(value) for value in raw_groups[mask]],
                pooled_budget=composite_budget,
                group_budget=composite_budget,
            )
            expected_thresholds.append(class_threshold)
            expected_details.append(class_detail)
        if (
            not np.array_equal(
                thresholds, np.asarray(expected_thresholds, dtype=np.float64)
            )
            or composition.get(
                "threshold_detail_by_predicted_public_class"
            ) != _jsonable(expected_details)
            or float(
                composition.get("pooled_fallback_threshold", float("nan"))
            ) != pooled_threshold
            or composition.get("pooled_fallback_threshold_detail")
            != _jsonable(pooled_detail)
        ):
            raise ValueError(
                f"N{length} composite threshold provenance does not reproduce"
            )

    browser_contract = value.get("browser_contract")
    if (
        not isinstance(browser_contract, Mapping)
        or browser_contract.get("json_only") is not True
        or browser_contract.get("external_to_classifier_asset") is not True
        or browser_contract.get("classifier_schema_required")
        != browser_export.BROWSER_SCHEMA
        or browser_contract.get("classifier_schema_version_required")
        != browser_export.BROWSER_SCHEMA_VERSION
        or browser_contract.get("empirical_rank_rule") != EMPIRICAL_RANK_RULE
        or browser_contract.get("prototype_source_option_required")
        is not True
        or browser_contract.get("prototype_source_returned") is not True
        or browser_contract.get("union_head_fallback_forbidden") is not True
        or float(
            browser_contract.get("runtime_comparison_margin", float("nan"))
        ) != RUNTIME_COMPARISON_MARGIN
        or browser_contract.get(
            "runtime_bucket_examples_by_prototype_source"
        )
        != {
            "historical": {
                "valid_sample_count_12160": 8_192,
                "valid_sample_count_16384": 16_384,
            },
            "current": {
                "valid_sample_count_4096": 4_096,
                "valid_sample_count_12160": 4_096,
                "valid_sample_count_16384": 4_096,
            },
        }
    ):
        raise ValueError("open-set browser contract is invalid")

    binding = value.get("classifier_binding")
    if (
        not isinstance(binding, Mapping)
        or binding.get("source_fusion_schema") != scale_eval.FUSION_SCHEMA
        or not _is_sha256(
            binding.get("source_fusion_dev_metrics_sha256")
        )
        or not isinstance(
            binding.get("source_fusion_artifacts_sha256"), Mapping
        )
        or not binding["source_fusion_artifacts_sha256"]
        or not all(
            isinstance(name, str)
            and name
            and _is_sha256(digest)
            for name, digest in binding[
                "source_fusion_artifacts_sha256"
            ].items()
        )
        or binding.get("classifier_schema_required")
        != browser_export.BROWSER_SCHEMA
        or binding.get("classifier_schema_version_required")
        != browser_export.BROWSER_SCHEMA_VERSION
        or binding.get("classifier_runtime_role_required")
        != "accepted_known_classifier"
        or binding.get("browser_frontend")
        != {
            "version": frontend["version"],
            "patch_length": int(frontend["patch_length"]),
            "patch_count": int(frontend["patch_count"]),
            "target_frac": float(frontend["target_frac"]),
            "uses_frequency_transform": False,
            "runtime_input_lengths": list(RUNTIME_INPUT_LENGTHS),
            "runtime_bucket_rule": browser_export.RUNTIME_BUCKET_RULE,
        }
        or binding.get("browser_classifier_asset_bound") is not True
        or not _is_sha256(
            binding.get("browser_classifier_asset_sha256")
        )
        or not isinstance(binding.get("classifier_routing"), Mapping)
        or value.get("acquisition_routing")
        != binding.get("classifier_routing")
        or binding.get(
            "supported_public_class_mask_by_prototype_source"
        )
        != {
            source: list(
                EXPECTED_PUBLIC_CLASS_SUPPORT_BY_PROTOTYPE_SOURCE[source]
            )
            for source in PROTOTYPE_SOURCE_ROUTES
        }
    ):
        raise ValueError(
            "open-set classifier binding must name an exact browser asset"
        )
    if expected_fusion_binding is not None:
        for key in (
            "source_fusion_schema",
            "source_fusion_dev_metrics_sha256",
            "source_fusion_artifacts_sha256",
            "classifier_schema_required",
            "classifier_schema_version_required",
            "classifier_runtime_role_required",
            "browser_frontend",
            "classifier_routing",
            "supported_public_class_mask_by_prototype_source",
        ):
            if binding.get(key) != expected_fusion_binding.get(key):
                raise ValueError(
                    f"open-set policy is bound to another classifier ({key})"
                )
        expected_browser = expected_fusion_binding.get(
            "browser_classifier_asset_sha256"
        )
        recorded_browser = binding.get("browser_classifier_asset_sha256")
        if (
            binding.get("browser_classifier_asset_bound")
            != expected_fusion_binding.get("browser_classifier_asset_bound")
            or recorded_browser != expected_browser
        ):
            raise ValueError(
                "open-set policy is bound to another browser classifier JSON"
            )
    _validate_validation_protocol_record(
        value.get("validation_protocol"),
        classifier_asset_sha256=str(
            binding["browser_classifier_asset_sha256"]
        ),
    )
    # Round-trip through strict JSON now, rather than discovering NaN or a
    # non-serialisable value only in a browser build.
    json.dumps(_jsonable(value), sort_keys=True, allow_nan=False)
    return dict(value)


def _policy_stage_one_model(
    policy: Mapping[str, Any],
    length: int,
) -> noise_prefilter.NoisePrefilter:
    source = policy["stage_one"]["parameters_by_length"][str(length)]
    return noise_prefilter.NoisePrefilter(
        feature_names=tuple(policy["stage_one"]["feature_names"]),
        capture_length=int(length),
        mean=np.asarray(source["mean"], dtype=np.float64),
        scale=np.asarray(source["scale"], dtype=np.float64),
        coefficients=np.asarray(source["coefficients"], dtype=np.float64),
        intercept=float(source["intercept"]),
        threshold_score=0.0,
        fit_provenance={
            "population": "loaded_external_policy",
            "consumed_test_rows_used_in_fit": 0,
        },
        threshold_provenance={
            "population": None,
            "chosen": False,
            "selection_rows_used_for_threshold": 0,
            "novelty_rows_used_for_threshold": 0,
        },
    )


@dataclass
class PopulationScores:
    final_score: np.ndarray
    rejected: np.ndarray
    stage_zero: np.ndarray
    stage_one: np.ndarray
    stage_two: np.ndarray
    prediction: np.ndarray
    winning_distance: np.ndarray
    stage_one_score: np.ndarray
    stage_one_predicted_class_rank: np.ndarray
    stage_one_pooled_rank: np.ndarray
    stage_one_rank: np.ndarray
    stage_two_rank: np.ndarray
    phase_linearity_score: np.ndarray
    phase_linearity_rank: np.ndarray
    composite_score: np.ndarray


def _empty_population_scores(rows: int) -> PopulationScores:
    return PopulationScores(
        final_score=np.zeros(rows, dtype=np.float64),
        rejected=np.zeros(rows, dtype=bool),
        stage_zero=np.zeros(rows, dtype=bool),
        stage_one=np.zeros(rows, dtype=bool),
        stage_two=np.zeros(rows, dtype=bool),
        prediction=np.full(rows, -1, dtype=np.int64),
        winning_distance=np.full(rows, np.nan, dtype=np.float64),
        stage_one_score=np.full(rows, np.nan, dtype=np.float64),
        stage_one_predicted_class_rank=np.full(
            rows, np.nan, dtype=np.float64
        ),
        stage_one_pooled_rank=np.full(rows, np.nan, dtype=np.float64),
        stage_one_rank=np.full(rows, np.nan, dtype=np.float64),
        stage_two_rank=np.full(rows, np.nan, dtype=np.float64),
        phase_linearity_score=np.full(rows, np.nan, dtype=np.float64),
        phase_linearity_rank=np.full(rows, np.nan, dtype=np.float64),
        composite_score=np.full(rows, np.nan, dtype=np.float64),
    )


def _is_exact_no_signal(row: np.ndarray) -> bool:
    value = np.asarray(row)
    return bool(
        value.ndim == 1
        and len(value) > 0
        and np.isfinite(value.real).all()
        and np.isfinite(value.imag).all()
        and float(np.max(np.abs(value))) == 0.0
    )


def _score_from_distances(
    policy: Mapping[str, Any],
    *,
    length: int,
    prototype_source: str,
    stage_one_score: np.ndarray,
    phase_linearity_score: np.ndarray,
    class_distances: np.ndarray,
) -> PopulationScores:
    """Score nonzero rows from stage-one scores and seven class distances."""
    stage_one_values = np.asarray(stage_one_score, dtype=np.float64)
    phase_linearity_values = np.asarray(
        phase_linearity_score, dtype=np.float64
    )
    distances = np.asarray(class_distances, dtype=np.float64)
    source = validate_prototype_source(prototype_source)
    source_policy = policy["per_prototype_source"][source]
    support = np.asarray(
        source_policy["supported_public_class_mask"], dtype=bool
    )
    if (
        stage_one_values.ndim != 1
        or phase_linearity_values.shape != stage_one_values.shape
        or distances.shape != (len(stage_one_values), len(PUBLIC_CLASSES))
        or not np.isfinite(stage_one_values).all()
        or not np.isfinite(phase_linearity_values).all()
        or np.any(phase_linearity_values < 0.0)
        or np.any(phase_linearity_values > 1.0)
        or support.shape != (len(PUBLIC_CLASSES),)
        or not np.any(support)
        or not np.isfinite(distances[:, support]).all()
        or not np.isinf(distances[:, ~support]).all()
    ):
        raise ValueError("open-set scoring inputs are invalid")
    result = _empty_population_scores(len(stage_one_values))
    row_policy = source_policy["per_length"][str(length)]
    stage_one_policy = row_policy["stage_one"]
    stage_two_policy = row_policy["stage_two"]
    phase_linearity_policy = row_policy["stage_chirp"]
    result.stage_one_score[:] = stage_one_values
    result.phase_linearity_score[:] = phase_linearity_values
    result.stage_one[:] = False
    prediction = distances.argmin(axis=1).astype(np.int64)
    if not np.all(support[prediction]):
        raise AssertionError("unsupported routed class won runtime argmin")
    winning = distances[np.arange(len(distances)), prediction]
    stage_one_predicted_class_rank = _class_rank(
        stage_one_values,
        prediction,
        stage_one_policy[
            "score_rank_calibration_by_predicted_class_sorted"
        ],
        stage_one_policy["pooled_score_rank_calibration_sorted"],
    )
    stage_one_pooled_rank = empirical_rank(
        stage_one_values,
        np.asarray(
            stage_one_policy["pooled_score_rank_calibration_sorted"],
            dtype=np.float64,
        ),
    )
    stage_one_rank = np.maximum(
        stage_one_predicted_class_rank,
        stage_one_pooled_rank,
    )
    stage_two_rank = _class_rank(
        winning,
        prediction,
        stage_two_policy[
            "predicted_class_distance_calibration_sorted"
        ],
        stage_two_policy["pooled_distance_calibration_sorted"],
    )
    phase_linearity_rank = _class_rank(
        phase_linearity_values,
        prediction,
        phase_linearity_policy[
            "score_rank_calibration_by_predicted_class_sorted"
        ],
        phase_linearity_policy[
            "pooled_score_rank_calibration_sorted"
        ],
    )
    composite = np.maximum.reduce(
        (stage_one_rank, stage_two_rank, phase_linearity_rank)
    )
    thresholds = np.asarray(
        row_policy["composition"][
            "threshold_by_predicted_public_class"
        ],
        dtype=np.float64,
    )[prediction]
    stage_two = composite >= thresholds
    result.prediction[:] = prediction
    result.winning_distance[:] = winning
    result.stage_one_predicted_class_rank[:] = (
        stage_one_predicted_class_rank
    )
    result.stage_one_pooled_rank[:] = stage_one_pooled_rank
    result.stage_one_rank[:] = stage_one_rank
    result.stage_two_rank[:] = stage_two_rank
    result.phase_linearity_rank[:] = phase_linearity_rank
    result.composite_score[:] = composite
    result.stage_two[:] = stage_two
    result.final_score[:] = composite
    result.rejected[:] = stage_two
    return result


def _policy_stage_one_scores(
    policy: Mapping[str, Any],
    rows: Sequence[np.ndarray],
    *,
    length: int,
) -> np.ndarray:
    frontend = policy["frontend"]
    model = _policy_stage_one_model(policy, length)
    features = np.stack(
        [
            _pose_feature_row(
                np.asarray(row, dtype=np.complex64),
                patch_length=int(frontend["patch_length"]),
                target_frac=float(frontend["target_frac"]),
            )
            for row in rows
        ]
    )
    return np.asarray(
        model.score(features, capture_length=length), dtype=np.float64
    )


def _copy_scores_into(
    destination: PopulationScores,
    positions: np.ndarray,
    source: PopulationScores,
) -> None:
    for name in PopulationScores.__dataclass_fields__:
        target = getattr(destination, name)
        value = getattr(source, name)
        target[positions] = value


def score_prepared_population(
    context: DevelopmentContext,
    policy: Mapping[str, Any],
    raw_rows: Sequence[np.ndarray],
    packed: np.ndarray,
    features: np.ndarray,
    *,
    length: int,
    prototype_source: str,
) -> PopulationScores:
    """Score admitted known rows whose v4 frontend tensors already exist."""
    rows = list(raw_rows)
    if len(rows) != len(packed) or len(rows) != len(features):
        raise ValueError("prepared known rows lost alignment")
    if any(_is_exact_no_signal(row) for row in rows):
        raise ValueError("prepared known population contains exact no-signal")
    stage_one = _policy_stage_one_scores(
        policy, rows, length=int(length)
    )
    embeddings = _embed(context, packed, features)
    distances, _closest, _support = _public_distances(
        context,
        embeddings,
        prototype_source=prototype_source,
    )
    return _score_from_distances(
        policy,
        length=int(length),
        prototype_source=prototype_source,
        stage_one_score=stage_one,
        phase_linearity_score=instantaneous_phase_linearity_scores(rows),
        class_distances=distances,
    )


def score_raw_population(
    context: DevelopmentContext,
    policy: Mapping[str, Any],
    raw_rows: Sequence[np.ndarray],
    *,
    length: int,
    prototype_source: str,
) -> PopulationScores:
    """Score raw novelty with a logical early exit only at exact no-signal."""
    source = validate_prototype_source(prototype_source)
    rows = [
        np.ascontiguousarray(np.asarray(row, dtype=np.complex64))
        for row in raw_rows
    ]
    if not rows or any(row.shape != (int(length),) for row in rows):
        raise ValueError(f"raw population must contain N{length} vectors")
    if any(
        not np.isfinite(row.real).all() or not np.isfinite(row.imag).all()
        for row in rows
    ):
        raise ValueError("raw population contains non-finite I/Q")
    result = _empty_population_scores(len(rows))
    stage_zero = np.asarray(
        [_is_exact_no_signal(row) for row in rows], dtype=bool
    )
    result.stage_zero[:] = stage_zero
    result.rejected[stage_zero] = True
    result.final_score[stage_zero] = 4.0
    nonzero_positions = np.flatnonzero(~stage_zero)
    if not len(nonzero_positions):
        return result

    nonzero_rows = [rows[int(index)] for index in nonzero_positions]
    stage_one_values = _policy_stage_one_scores(
        policy, nonzero_rows, length=int(length)
    )
    phase_linearity_values = instantaneous_phase_linearity_scores(
        nonzero_rows
    )
    stage_one_gate = np.zeros(len(nonzero_rows), dtype=bool)
    class_distances = np.zeros(
        (len(nonzero_rows), len(PUBLIC_CLASSES)), dtype=np.float64
    )
    classified_local = np.arange(len(nonzero_rows), dtype=np.int64)
    if len(classified_local):
        packed_rows: list[np.ndarray] = []
        raw_features: list[np.ndarray] = []
        for local in classified_local:
            packed, features, _context = td_preprocess.preprocess(
                nonzero_rows[int(local)],
                patch_length=context.patch_length,
                patch_count=context.patch_count,
                target_frac=context.target_frac,
            )
            packed_rows.append(np.asarray(packed, dtype=np.float32))
            raw_features.append(np.asarray(features, dtype=np.float32))
        packed = np.stack(packed_rows).astype(np.float32)
        standardized = (
            (
                np.stack(raw_features).astype(np.float32)
                - context.fusion["feature_mean"]
            )
            / context.fusion["feature_std"]
        ).astype(np.float32)
        embeddings = _embed(context, packed, standardized)
        classified_distances, _closest, _support = _public_distances(
            context,
            embeddings,
            prototype_source=source,
        )
        class_distances[classified_local] = classified_distances
    scored_nonzero = _score_from_distances(
        policy,
        length=int(length),
        prototype_source=source,
        stage_one_score=stage_one_values,
        phase_linearity_score=phase_linearity_values,
        class_distances=class_distances,
    )
    if not np.array_equal(scored_nonzero.stage_one, stage_one_gate):
        raise AssertionError("disabled stage-one gate changed during composition")
    _copy_scores_into(result, nonzero_positions, scored_nonzero)
    # Restore stage-zero state after the generic copy helper touches only
    # nonzero positions.
    result.stage_zero[:] = stage_zero
    result.rejected[stage_zero] = True
    result.final_score[stage_zero] = 4.0
    return result


def score_observed_raw_population(
    context: DevelopmentContext,
    policy: Mapping[str, Any],
    raw_rows: Sequence[np.ndarray],
    *,
    observation_length: int,
    prototype_source: str,
) -> PopulationScores:
    """Score an observation through its route-scoped causal prefix."""
    source = validate_prototype_source(prototype_source)
    observed = int(observation_length)
    rows = [
        np.ascontiguousarray(np.asarray(row, dtype=np.complex64))
        for row in raw_rows
    ]
    if not rows or any(row.shape != (observed,) for row in rows):
        raise ValueError(
            f"observed raw population must contain N{observed} vectors"
        )
    effective = effective_runtime_input_length(
        policy,
        prototype_source=source,
        observation_length=observed,
    )
    effective_rows = (
        rows
        if effective == observed
        else [
            np.ascontiguousarray(row[:effective], dtype=np.complex64)
            for row in rows
        ]
    )
    return score_raw_population(
        context,
        policy,
        effective_rows,
        length=effective,
        prototype_source=source,
    )


def _population_summary(
    scores: PopulationScores,
    *,
    truth: np.ndarray | None = None,
    known_groups: Sequence[str] | None = None,
) -> dict[str, Any]:
    rows = len(scores.rejected)
    if rows == 0:
        raise ValueError("population summary needs at least one row")
    attributed = scores.stage_zero | scores.stage_one | scores.stage_two
    if not np.array_equal(attributed, scores.rejected):
        raise AssertionError("open-set rejection attribution does not conserve")
    summary: dict[str, Any] = {
        "rows": rows,
        "rejected": int(np.sum(scores.rejected)),
        "rejection_rate": float(np.mean(scores.rejected)),
        "attribution": {
            "stage_zero_no_signal": int(np.sum(scores.stage_zero)),
            "stage_one_hard_gate_disabled": int(np.sum(scores.stage_one)),
            "stage_two_composite": int(np.sum(scores.stage_two)),
            "accepted": int(np.sum(~scores.rejected)),
        },
        "final_score": {
            "minimum": float(np.min(scores.final_score)),
            "median": float(np.median(scores.final_score)),
            "maximum": float(np.max(scores.final_score)),
        },
    }
    if truth is not None:
        labels = np.asarray(truth, dtype=np.int64)
        if labels.shape != (rows,):
            raise ValueError("known truth labels lost alignment")
        accepted = ~scores.rejected
        closed_correct = scores.prediction == labels
        closed_correct_rows = int(np.sum(closed_correct))
        rejected_closed_correct = scores.rejected & closed_correct
        summary["accepted_closed_accuracy"] = (
            float(np.mean(scores.prediction[accepted] == labels[accepted]))
            if np.any(accepted)
            else None
        )
        summary["closed_head_accuracy"] = float(np.mean(closed_correct))
        summary["closed_head_correct_rows"] = closed_correct_rows
        summary["closed_head_incorrect_rows"] = rows - closed_correct_rows
        summary["rejected_closed_head_correct_rows"] = int(
            np.sum(rejected_closed_correct)
        )
        summary["abstention_damage_given_closed_head_correct"] = (
            float(np.mean(scores.rejected[closed_correct]))
            if closed_correct_rows
            else None
        )
        summary["acceptance_given_closed_head_correct"] = (
            float(np.mean(accepted[closed_correct]))
            if closed_correct_rows
            else None
        )
        summary["correct_and_accepted_rate"] = float(
            np.mean(closed_correct & accepted)
        )
        per_class: dict[str, Any] = {}
        for class_index, class_name in enumerate(PUBLIC_CLASSES):
            mask = labels == class_index
            if not np.any(mask):
                continue
            class_closed_correct = mask & closed_correct
            class_closed_correct_rows = int(np.sum(class_closed_correct))
            per_class[class_name] = {
                "rows": int(np.sum(mask)),
                "rejected": int(np.sum(scores.rejected[mask])),
                "false_unknown_rate": float(
                    np.mean(scores.rejected[mask])
                ),
                "closed_head_accuracy": float(
                    np.mean(closed_correct[mask])
                ),
                "closed_head_correct_rows": class_closed_correct_rows,
                "closed_head_incorrect_rows": int(
                    np.sum(mask) - class_closed_correct_rows
                ),
                "rejected_closed_head_correct_rows": int(
                    np.sum(scores.rejected[class_closed_correct])
                ),
                "abstention_damage_given_closed_head_correct": (
                    float(np.mean(scores.rejected[class_closed_correct]))
                    if class_closed_correct_rows
                    else None
                ),
                "acceptance_given_closed_head_correct": (
                    float(np.mean(accepted[class_closed_correct]))
                    if class_closed_correct_rows
                    else None
                ),
                "correct_and_accepted_rate": float(
                    np.mean(
                        closed_correct[mask] & accepted[mask]
                    )
                ),
                "accepted_closed_accuracy": (
                    float(
                        np.mean(
                            scores.prediction[mask & accepted]
                            == labels[mask & accepted]
                        )
                    )
                    if np.any(mask & accepted)
                    else None
                ),
            }
        summary["per_true_public_class"] = per_class
    if known_groups is not None:
        group_array = np.asarray(
            [str(value) for value in known_groups], dtype=object
        )
        if group_array.shape != (rows,):
            raise ValueError("known profile groups lost alignment")
        per_group: dict[str, Any] = {}
        for group in sorted(set(str(value) for value in group_array)):
            mask = group_array == group
            row = {
                "rows": int(np.sum(mask)),
                "rejected": int(
                    np.sum(scores.rejected[mask])
                ),
                "false_unknown_rate": float(
                    np.mean(scores.rejected[mask])
                ),
            }
            if truth is not None:
                accepted = ~scores.rejected
                closed_correct = scores.prediction == labels
                group_closed_correct = mask & closed_correct
                group_closed_correct_rows = int(
                    np.sum(group_closed_correct)
                )
                row["closed_head_accuracy"] = float(
                    np.mean(closed_correct[mask])
                )
                row["closed_head_correct_rows"] = (
                    group_closed_correct_rows
                )
                row["closed_head_incorrect_rows"] = int(
                    np.sum(mask) - group_closed_correct_rows
                )
                row["rejected_closed_head_correct_rows"] = int(
                    np.sum(scores.rejected[group_closed_correct])
                )
                row["abstention_damage_given_closed_head_correct"] = (
                    float(np.mean(scores.rejected[group_closed_correct]))
                    if group_closed_correct_rows
                    else None
                )
                row["acceptance_given_closed_head_correct"] = (
                    float(np.mean(accepted[group_closed_correct]))
                    if group_closed_correct_rows
                    else None
                )
                row["correct_and_accepted_rate"] = float(
                    np.mean(
                        closed_correct[mask] & accepted[mask]
                    )
                )
                row["accepted_closed_accuracy"] = (
                    float(
                        np.mean(
                            scores.prediction[mask & accepted]
                            == labels[mask & accepted]
                        )
                    )
                    if np.any(mask & accepted)
                    else None
                )
            per_group[group] = row
        summary["per_source_profile"] = per_group
    return summary


def _current_selection_causal_prefix_agreement(
    records_by_length: Mapping[int, Sequence[corpus_data.RowRef]],
    scores_by_length: Mapping[int, PopulationScores],
) -> dict[str, Any]:
    """Audit identical outcomes across every available length of each row."""
    metadata_by_source_index: dict[int, dict[str, Any]] = {}
    for length, records in records_by_length.items():
        if int(length) not in RUNTIME_INPUT_LENGTHS:
            raise ValueError("current agreement has an invalid observation length")
        for record in records:
            source_index = int(record.source_index)
            expected_lengths = tuple(
                corpus_data.training_view_lengths(record.valid_sample_count)
            )
            metadata = {
                "base_identity": str(record.identity),
                "source_index": source_index,
                "source_profile": f"current:{record.profile_id}",
                "expected_observation_lengths": expected_lengths,
            }
            previous = metadata_by_source_index.setdefault(
                source_index, metadata
            )
            if previous != metadata or int(length) not in expected_lengths:
                raise ValueError(
                    "current selection row metadata changed across lengths"
                )
    if not metadata_by_source_index:
        raise ValueError("current selection agreement has no stored rows")

    expected_cells: dict[str, int] = {}
    for metadata in metadata_by_source_index.values():
        for length in metadata["expected_observation_lengths"]:
            key = (
                f"{metadata['base_identity']}|row={metadata['source_index']}"
                f"|N={int(length)}"
            )
            if key in expected_cells:
                raise ValueError(
                    "current selection expected agreement cell is duplicated"
                )
            expected_cells[key] = 1

    observed_cells: dict[str, int] = {}
    outcome_by_source_index: dict[int, dict[int, str]] = {}
    for length, records in records_by_length.items():
        scores = scores_by_length.get(int(length))
        if scores is None or len(scores.rejected) != len(records):
            raise ValueError(
                "current selection agreement records/scores lost alignment"
            )
        for record, rejected, prediction in zip(
            records,
            scores.rejected,
            scores.prediction,
        ):
            source_index = int(record.source_index)
            metadata = metadata_by_source_index[source_index]
            key = (
                f"{metadata['base_identity']}|row={source_index}"
                f"|N={int(length)}"
            )
            if key in observed_cells:
                raise ValueError(
                    "current selection observed agreement cell is duplicated"
                )
            observed_cells[key] = 1
            outcome = (
                "unknown"
                if bool(rejected)
                else PUBLIC_CLASSES[int(prediction)]
            )
            outcome_by_source_index.setdefault(source_index, {})[
                int(length)
            ] = outcome

    exact_coverage = observed_cells == expected_cells
    observation_rows: list[dict[str, Any]] = []
    for source_index, metadata in sorted(metadata_by_source_index.items()):
        expected_lengths = tuple(metadata["expected_observation_lengths"])
        outcomes = outcome_by_source_index.get(source_index, {})
        observed_lengths = tuple(sorted(outcomes))
        complete = observed_lengths == expected_lengths
        agrees = complete and len(set(outcomes.values())) == 1
        observation_rows.append(
            {
                **metadata,
                "expected_observation_lengths": list(expected_lengths),
                "observed_observation_lengths": list(observed_lengths),
                "outcome_by_observation_length": {
                    str(length): outcomes[length]
                    for length in observed_lengths
                },
                "complete_length_coverage": complete,
                "prediction_or_abstention_agrees": agrees,
            }
        )

    by_profile: dict[str, Any] = {}
    profiles = sorted(
        {row["source_profile"] for row in observation_rows}
    )
    for profile in profiles:
        selected = [
            row
            for row in observation_rows
            if row["source_profile"] == profile
        ]
        profile_expected_cells = sum(
            len(row["expected_observation_lengths"]) for row in selected
        )
        profile_observed_cells = sum(
            len(row["observed_observation_lengths"]) for row in selected
        )
        failing = [
            int(row["source_index"])
            for row in selected
            if not row["prediction_or_abstention_agrees"]
        ]
        by_profile[profile] = {
            "stored_observations": len(selected),
            "base_identities": len(
                {row["base_identity"] for row in selected}
            ),
            "expected_observation_length_cells": profile_expected_cells,
            "observed_observation_length_cells": profile_observed_cells,
            "exact_length_coverage": (
                profile_expected_cells == profile_observed_cells
                and all(row["complete_length_coverage"] for row in selected)
            ),
            "prediction_or_abstention_agreement_rate": float(
                np.mean(
                    [
                        row["prediction_or_abstention_agrees"]
                        for row in selected
                    ]
                )
            ),
            "failing_source_indices": failing,
        }

    agreement = float(
        np.mean(
            [
                row["prediction_or_abstention_agrees"]
                for row in observation_rows
            ]
        )
    )
    return {
        "scope": "current_trusted_exact_profile_route_selection",
        "pairing_unit": "stored_observation_with_base_identity_audit",
        "per_length_keys_are_observation_lengths": True,
        "effective_runtime_input_length":
            CURRENT_CANONICAL_RUNTIME_INPUT_LENGTH,
        "stored_observations": len(observation_rows),
        "base_identities": len(
            {row["base_identity"] for row in observation_rows}
        ),
        "expected_observation_length_cells": len(expected_cells),
        "observed_observation_length_cells": len(observed_cells),
        "expected_base_identity_row_length_cells": dict(
            sorted(expected_cells.items())
        ),
        "observed_base_identity_row_length_cells": dict(
            sorted(observed_cells.items())
        ),
        "exact_expected_observed_cell_coverage": exact_coverage,
        "minimum_distinct_observation_lengths": min(
            len(row["expected_observation_lengths"])
            for row in observation_rows
        ),
        "maximum_distinct_observation_lengths": max(
            len(row["expected_observation_lengths"])
            for row in observation_rows
        ),
        "stored_observations_by_distinct_observation_length_count": {
            str(count): sum(
                1
                for row in observation_rows
                if len(row["expected_observation_lengths"]) == count
            )
            for count in range(
                MIN_PAIRED_DISTINCT_RUNTIME_LENGTHS,
                len(RUNTIME_INPUT_LENGTHS) + 1,
            )
        },
        "prediction_or_abstention_agreement_rate": agreement,
        "by_source_profile": by_profile,
        "failing_source_indices": [
            int(row["source_index"])
            for row in observation_rows
            if not row["prediction_or_abstention_agrees"]
        ],
        "observation_rows": observation_rows,
    }


def evaluate_known_selection(
    context: DevelopmentContext,
    policy: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[int, np.ndarray]]]:
    """Score historical/current selection, never returning data to fitting."""
    source_contract = (
        ("historical", context.historical),
        ("current", context.current),
    )
    report: dict[str, Any] = {}
    combined_scores: dict[int, list[np.ndarray]] = {
        length: [] for length in RUNTIME_INPUT_LENGTHS
    }
    combined_correct_and_accepted: dict[int, list[np.ndarray]] = {
        length: [] for length in RUNTIME_INPUT_LENGTHS
    }
    combined_closed_correct: dict[int, list[np.ndarray]] = {
        length: [] for length in RUNTIME_INPUT_LENGTHS
    }
    combined_rejected_closed_correct: dict[int, list[np.ndarray]] = {
        length: [] for length in RUNTIME_INPUT_LENGTHS
    }
    score_by_source: dict[str, dict[int, np.ndarray]] = {}
    for source_name, corpus in source_contract:
        raw = _raw_role_views(corpus, "selection")
        records = _role_records(corpus, "selection")
        prepared = context.data[f"{source_name}_selection"]["by_length"]
        per_length: dict[str, Any] = {}
        pooled_rejected: list[np.ndarray] = []
        pooled_scores: list[np.ndarray] = []
        pooled_closed_correct: list[np.ndarray] = []
        source_scores_by_length: dict[int, PopulationScores] = {}
        for length in sorted(prepared):
            bucket = prepared[length]
            if len(raw[length]) != len(bucket["y"]):
                raise RuntimeError(
                    f"{source_name} N{length} raw/prepared selection mismatch"
                )
            effective_length = effective_runtime_input_length(
                policy,
                prototype_source=source_name,
                observation_length=length,
            )
            scores = (
                score_observed_raw_population(
                    context,
                    policy,
                    raw[length],
                    observation_length=length,
                    prototype_source=source_name,
                )
                if source_name == "current"
                else score_prepared_population(
                    context,
                    policy,
                    raw[length],
                    bucket["x"],
                    bucket["f"],
                    length=length,
                    prototype_source=source_name,
                )
            )
            source_scores_by_length[int(length)] = scores
            groups = [
                f"{row.source}:{row.profile_id}"
                for row in records[length]
            ]
            length_summary = _population_summary(
                scores,
                truth=bucket["y"],
                known_groups=groups,
            )
            length_summary["observation_runtime_input_length"] = int(
                length
            )
            length_summary["effective_runtime_input_length"] = int(
                effective_length
            )
            per_length[str(length)] = length_summary
            pooled_rejected.append(scores.rejected)
            pooled_scores.append(scores.final_score)
            closed_correct = scores.prediction == bucket["y"]
            pooled_closed_correct.append(closed_correct)
            combined_scores[length].append(scores.final_score)
            combined_correct_and_accepted[length].append(
                closed_correct & ~scores.rejected
            )
            combined_closed_correct[length].append(closed_correct)
            combined_rejected_closed_correct[length].append(
                closed_correct & scores.rejected
            )
        all_rejected = np.concatenate(pooled_rejected)
        all_scores = np.concatenate(pooled_scores)
        all_closed_correct = np.concatenate(pooled_closed_correct)
        report[source_name] = {
            "population_role": (
                "SignalLab service current-source selection; evaluation only"
                if source_name == "current"
                else "historical exposed selection; evaluation only"
            ),
            "runtime_length_policy":
                RUNTIME_LENGTH_POLICY_BY_PROTOTYPE_SOURCE[source_name],
            "per_length": per_length,
            "pooled": {
                "rows": int(len(all_rejected)),
                "rejected": int(np.sum(all_rejected)),
                "false_unknown_rate": float(np.mean(all_rejected)),
                "closed_head_accuracy": float(
                    np.mean(all_closed_correct)
                ),
                "closed_head_correct_rows": int(
                    np.sum(all_closed_correct)
                ),
                "abstention_damage_given_closed_head_correct": (
                    float(np.mean(all_rejected[all_closed_correct]))
                    if np.any(all_closed_correct)
                    else None
                ),
                "final_score_median": float(np.median(all_scores)),
            },
        }
        if source_name == "current":
            report[source_name][
                "causal_prefix_length_agreement"
            ] = _current_selection_causal_prefix_agreement(
                records,
                source_scores_by_length,
            )
        score_by_source[source_name] = {
            int(length): np.asarray(values, dtype=np.float64)
            for length, values in (
                (
                    int(length),
                    pooled_scores[index],
                )
                for index, length in enumerate(sorted(prepared))
            )
        }
    combined = {
        length: np.concatenate(values)
        for length, values in combined_scores.items()
        if values
    }
    report["combined"] = {
        "per_length_false_unknown_rate": {},
        "per_length_correct_and_accepted_rate": {},
        "per_length_closed_head_accuracy": {},
        "per_length_closed_head_correct_rows": {},
        "per_length_rejected_closed_head_correct_rows": {},
        "per_length_abstention_damage_given_closed_head_correct": {},
        "note": "combined score vectors are retained for novelty AUROC only",
    }
    combined_fur: dict[str, float] = {}
    for length in RUNTIME_INPUT_LENGTHS:
        numer = 0
        denom = 0
        for source_name, _corpus in source_contract:
            row = report[source_name]["per_length"].get(str(length))
            if row is not None:
                numer += int(row["rejected"])
                denom += int(row["rows"])
        if denom:
            combined_fur[str(length)] = float(numer / denom)
    report["combined"]["per_length_false_unknown_rate"] = combined_fur
    combined_correct = {
        length: np.concatenate(values)
        for length, values in combined_correct_and_accepted.items()
        if values
    }
    combined_head_correct = {
        length: np.concatenate(values)
        for length, values in combined_closed_correct.items()
        if values
    }
    combined_rejected_head_correct = {
        length: np.concatenate(values)
        for length, values in combined_rejected_closed_correct.items()
        if values
    }
    report["combined"]["per_length_correct_and_accepted_rate"] = {
        str(length): float(np.mean(values))
        for length, values in combined_correct.items()
    }
    report["combined"]["per_length_closed_head_accuracy"] = {
        str(length): float(np.mean(values))
        for length, values in combined_head_correct.items()
    }
    report["combined"]["per_length_closed_head_correct_rows"] = {
        str(length): int(np.sum(values))
        for length, values in combined_head_correct.items()
    }
    report["combined"][
        "per_length_rejected_closed_head_correct_rows"
    ] = {
        str(length): int(np.sum(values))
        for length, values in combined_rejected_head_correct.items()
    }
    report["combined"][
        "per_length_abstention_damage_given_closed_head_correct"
    ] = {
        str(length): (
            float(
                np.sum(combined_rejected_head_correct[length])
                / np.sum(values)
            )
            if np.any(values)
            else None
        )
        for length, values in combined_head_correct.items()
    }
    all_numer = sum(
        int(report[name]["pooled"]["rejected"])
        for name, _corpus in source_contract
    )
    all_denom = sum(
        int(report[name]["pooled"]["rows"])
        for name, _corpus in source_contract
    )
    report["combined"]["pooled_false_unknown_rate"] = float(
        all_numer / all_denom
    )
    report["combined"]["rows"] = all_denom
    pooled_head_correct = np.concatenate(
        list(combined_head_correct.values())
    )
    pooled_rejected_head_correct = np.concatenate(
        list(combined_rejected_head_correct.values())
    )
    report["combined"]["pooled_closed_head_accuracy"] = float(
        np.mean(pooled_head_correct)
    )
    report["combined"]["pooled_closed_head_correct_rows"] = int(
        np.sum(pooled_head_correct)
    )
    report["combined"]["pooled_rejected_closed_head_correct_rows"] = int(
        np.sum(pooled_rejected_head_correct)
    )
    report["combined"][
        "pooled_abstention_damage_given_closed_head_correct"
    ] = (
        float(
            np.sum(pooled_rejected_head_correct)
            / np.sum(pooled_head_correct)
        )
        if np.any(pooled_head_correct)
        else None
    )
    report["combined"]["pooled_correct_and_accepted_rate"] = float(
        np.mean(np.concatenate(list(combined_correct.values())))
    )
    report["gate_cells"] = {
        "true_class": [
            {
                "value": float(class_row["false_unknown_rate"]),
                "source": source_name,
                "length": int(length),
                "name": class_name,
                "rows": int(class_row["rows"]),
                "closed_head_accuracy": float(
                    class_row["closed_head_accuracy"]
                ),
                "closed_head_correct_rows": int(
                    class_row["closed_head_correct_rows"]
                ),
                "rejected_closed_head_correct_rows": int(
                    class_row["rejected_closed_head_correct_rows"]
                ),
                "abstention_damage_given_closed_head_correct":
                    class_row[
                        "abstention_damage_given_closed_head_correct"
                    ],
                "correct_and_accepted_rate": float(
                    class_row["correct_and_accepted_rate"]
                ),
            }
            for source_name, _corpus in source_contract
            for length, length_row in report[source_name]["per_length"].items()
            for class_name, class_row in length_row[
                "per_true_public_class"
            ].items()
        ],
        "source_profile": [
            {
                "value": float(profile_row["false_unknown_rate"]),
                "source": source_name,
                "length": int(length),
                "name": profile_name,
                "rows": int(profile_row["rows"]),
                "closed_head_accuracy": float(
                    profile_row["closed_head_accuracy"]
                ),
                "closed_head_correct_rows": int(
                    profile_row["closed_head_correct_rows"]
                ),
                "rejected_closed_head_correct_rows": int(
                    profile_row["rejected_closed_head_correct_rows"]
                ),
                "abstention_damage_given_closed_head_correct":
                    profile_row[
                        "abstention_damage_given_closed_head_correct"
                    ],
                "correct_and_accepted_rate": float(
                    profile_row["correct_and_accepted_rate"]
                ),
            }
            for source_name, _corpus in source_contract
            for length, length_row in report[source_name]["per_length"].items()
            for profile_name, profile_row in length_row[
                "per_source_profile"
            ].items()
        ],
    }
    return report, score_by_source


def evaluate_held_scale_known(
    context: DevelopmentContext,
    policy: Mapping[str, Any],
    scale_corpus_dir: str | Path,
    current_reference_dir: str | Path,
    *,
    validation_protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[int, np.ndarray]]:
    """Score the separately held service-path phase/receiver/scale corpus."""
    corpus = scale_eval.load_scale_eval_corpus(scale_corpus_dir)
    if (
        corpus.audit.get("manifest_sha256")
        != validation_protocol.get("held_scale_manifest_sha256")
        or corpus.audit.get("raw_sha256")
        != validation_protocol.get("held_scale_raw_sha256")
        or corpus.manifest.get("evalSeed")
        != validation_protocol.get("held_scale_eval_seed")
        or corpus.audit.get("realizations_per_profile")
        != validation_protocol.get(
            "held_scale_realizations_per_profile"
        )
        or validation_corpus_binding_sha256(
            str(corpus.audit.get("manifest_sha256")),
            str(corpus.audit.get("raw_sha256")),
        )
        != validation_protocol.get("held_scale_corpus_binding_sha256")
    ):
        raise ValueError(
            "held scale corpus differs from the precommitted validation "
            "content identity"
        )
    separation = scale_eval.validate_reference_separation(
        corpus,
        current_reference_dir,
        candidate_current_audit=context.fusion["metrics"]["data_audit"][
            "current"
        ],
    )
    class_index = {
        name: index for index, name in enumerate(PUBLIC_CLASSES)
    }
    per_length_scale: dict[str, Any] = {}
    combined_score: dict[int, list[np.ndarray]] = {
        length: [] for length in RUNTIME_INPUT_LENGTHS
    }
    combined_rejected: dict[int, list[np.ndarray]] = {
        length: [] for length in RUNTIME_INPUT_LENGTHS
    }
    combined_correct_and_accepted: dict[int, list[np.ndarray]] = {
        length: [] for length in RUNTIME_INPUT_LENGTHS
    }
    combined_closed_correct: dict[int, list[np.ndarray]] = {
        length: [] for length in RUNTIME_INPUT_LENGTHS
    }
    combined_rejected_closed_correct: dict[int, list[np.ndarray]] = {
        length: [] for length in RUNTIME_INPUT_LENGTHS
    }
    gate_class_cells: list[dict[str, Any]] = []
    gate_profile_cells: list[dict[str, Any]] = []
    decision_by_pair_length_scale: dict[
        tuple[str, int, float], bool
    ] = {}
    score_by_pair_length_scale: dict[
        tuple[str, int, float], float
    ] = {}
    outcome_by_pair_length_scale: dict[
        tuple[str, int, float], str
    ] = {}
    for length in RUNTIME_INPUT_LENGTHS:
        length_report: dict[str, Any] = {}
        for scale in scale_eval.SCALE_FACTORS:
            rows = [
                row
                for row in corpus.rows
                if length in row.available_lengths
                and row.scale_factor == scale
            ]
            if not rows:
                continue
            raw = [corpus.prefix(row, length) for row in rows]
            truth = np.asarray(
                [class_index[row.class_name] for row in rows],
                dtype=np.int64,
            )
            groups = [f"current:{row.profile}" for row in rows]
            scores = score_observed_raw_population(
                context,
                policy,
                raw,
                observation_length=length,
                prototype_source="current",
            )
            summary = _population_summary(
                scores, truth=truth, known_groups=groups
            )
            summary["physical_scale_factor"] = float(scale)
            summary["observation_runtime_input_length"] = int(length)
            summary["effective_runtime_input_length"] = int(
                effective_runtime_input_length(
                    policy,
                    prototype_source="current",
                    observation_length=length,
                )
            )
            summary["source"] = (
                "held AtomizerMeasurementService phase/receiver realization"
            )
            length_report[str(scale)] = summary
            combined_score[length].append(scores.final_score)
            combined_rejected[length].append(scores.rejected)
            closed_correct = scores.prediction == truth
            combined_correct_and_accepted[length].append(
                closed_correct & ~scores.rejected
            )
            combined_closed_correct[length].append(closed_correct)
            combined_rejected_closed_correct[length].append(
                closed_correct & scores.rejected
            )
            for row, rejected, prediction, final_score in zip(
                rows,
                scores.rejected,
                scores.prediction,
                scores.final_score,
            ):
                key = (row.pair_id, length, float(scale))
                if key in decision_by_pair_length_scale:
                    raise RuntimeError(
                        "held scale corpus repeats a pair/length/scale "
                        f"observation: {key}"
                    )
                decision_by_pair_length_scale[key] = bool(rejected)
                score_by_pair_length_scale[key] = float(final_score)
                outcome_by_pair_length_scale[key] = (
                    "unknown"
                    if rejected
                    else PUBLIC_CLASSES[int(prediction)]
                )
            for name, row in summary["per_true_public_class"].items():
                gate_class_cells.append(
                    {
                        "value": float(row["false_unknown_rate"]),
                        "source": "held_service_scale",
                        "length": length,
                        "scale_factor": float(scale),
                        "name": name,
                        "rows": int(row["rows"]),
                        "closed_head_accuracy": float(
                            row["closed_head_accuracy"]
                        ),
                        "closed_head_correct_rows": int(
                            row["closed_head_correct_rows"]
                        ),
                        "rejected_closed_head_correct_rows": int(
                            row["rejected_closed_head_correct_rows"]
                        ),
                        "abstention_damage_given_closed_head_correct":
                            row[
                                "abstention_damage_given_closed_head_correct"
                            ],
                        "correct_and_accepted_rate": float(
                            row["correct_and_accepted_rate"]
                        ),
                    }
                )
            for name, row in summary["per_source_profile"].items():
                gate_profile_cells.append(
                    {
                        "value": float(row["false_unknown_rate"]),
                        "source": "held_service_scale",
                        "length": length,
                        "scale_factor": float(scale),
                        "name": name,
                        "rows": int(row["rows"]),
                        "closed_head_accuracy": float(
                            row["closed_head_accuracy"]
                        ),
                        "closed_head_correct_rows": int(
                            row["closed_head_correct_rows"]
                        ),
                        "rejected_closed_head_correct_rows": int(
                            row["rejected_closed_head_correct_rows"]
                        ),
                        "abstention_damage_given_closed_head_correct":
                            row[
                                "abstention_damage_given_closed_head_correct"
                            ],
                        "correct_and_accepted_rate": float(
                            row["correct_and_accepted_rate"]
                        ),
                    }
                )
        per_length_scale[str(length)] = length_report

    score_output = {
        length: np.concatenate(values)
        for length, values in combined_score.items()
        if values
    }
    rejected_output = {
        length: np.concatenate(values)
        for length, values in combined_rejected.items()
        if values
    }
    correct_output = {
        length: np.concatenate(values)
        for length, values in combined_correct_and_accepted.items()
        if values
    }
    closed_correct_output = {
        length: np.concatenate(values)
        for length, values in combined_closed_correct.items()
        if values
    }
    rejected_closed_correct_output = {
        length: np.concatenate(values)
        for length, values in combined_rejected_closed_correct.items()
        if values
    }
    paired_cells: list[dict[str, Any]] = []
    paired_length_scale_cells: list[dict[str, Any]] = []
    pair_ids = sorted({row.pair_id for row in corpus.rows})
    pair_metadata: dict[str, tuple[str, str]] = {}
    expected_observation_keys: set[tuple[str, int, float]] = set()
    expected_profile_legal_cell_rows: dict[str, int] = {}
    for row in corpus.rows:
        metadata = (str(row.profile), str(row.class_name))
        previous = pair_metadata.setdefault(row.pair_id, metadata)
        if previous != metadata:
            raise RuntimeError(
                f"held pair {row.pair_id!r} changes profile/class"
            )
        for length in row.available_lengths:
            if length in RUNTIME_INPUT_LENGTHS:
                expected_observation_keys.add(
                    (row.pair_id, int(length), float(row.scale_factor))
                )
                profile_cell = _profile_legal_cell_key(
                    f"current:{row.profile}",
                    int(length),
                    float(row.scale_factor),
                )
                expected_profile_legal_cell_rows[profile_cell] = (
                    expected_profile_legal_cell_rows.get(profile_cell, 0) + 1
                )
    observed_observation_keys = set(decision_by_pair_length_scale)
    if observed_observation_keys != expected_observation_keys:
        missing = sorted(expected_observation_keys - observed_observation_keys)
        extra = sorted(observed_observation_keys - expected_observation_keys)
        raise RuntimeError(
            "held pair/length/scale scoring coverage changed: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    observed_profile_legal_cell_rows: dict[str, int] = {}
    for length_text, length_rows in per_length_scale.items():
        length = int(length_text)
        for scale_text, summary in length_rows.items():
            scale = float(scale_text)
            for profile, profile_row in summary["per_source_profile"].items():
                profile_cell = _profile_legal_cell_key(
                    str(profile), length, scale
                )
                if profile_cell in observed_profile_legal_cell_rows:
                    raise RuntimeError(
                        "held summaries repeat a profile/length/scale cell: "
                        f"{profile_cell}"
                    )
                observed_profile_legal_cell_rows[profile_cell] = int(
                    profile_row["rows"]
                )
    if (
        observed_profile_legal_cell_rows
        != expected_profile_legal_cell_rows
        or sum(expected_profile_legal_cell_rows.values())
        != len(expected_observation_keys)
    ):
        missing = sorted(
            set(expected_profile_legal_cell_rows)
            - set(observed_profile_legal_cell_rows)
        )
        extra = sorted(
            set(observed_profile_legal_cell_rows)
            - set(expected_profile_legal_cell_rows)
        )
        mismatched = sorted(
            key
            for key in (
                set(expected_profile_legal_cell_rows)
                & set(observed_profile_legal_cell_rows)
            )
            if expected_profile_legal_cell_rows[key]
            != observed_profile_legal_cell_rows[key]
        )
        raise RuntimeError(
            "held profile legal-cell scoring coverage changed: "
            f"missing={missing[:3]}, extra={extra[:3]}, "
            f"count_mismatch={mismatched[:3]}"
        )

    expected_scales_by_pair_length: dict[
        tuple[str, int], tuple[float, ...]
    ] = {}
    observed_scales_by_pair_length: dict[
        tuple[str, int], tuple[float, ...]
    ] = {}
    for keys, destination in (
        (expected_observation_keys, expected_scales_by_pair_length),
        (observed_observation_keys, observed_scales_by_pair_length),
    ):
        groups: dict[tuple[str, int], set[float]] = {}
        for pair_id, length, scale in keys:
            groups.setdefault((pair_id, length), set()).add(scale)
        destination.update(
            {
                key: tuple(sorted(scales))
                for key, scales in groups.items()
            }
        )
    if (
        observed_scales_by_pair_length
        != expected_scales_by_pair_length
        or any(
            len(scales) < 2
            for scales in expected_scales_by_pair_length.values()
        )
    ):
        raise RuntimeError(
            "held physical-scale legal cell coverage changed"
        )
    paired_cells = []
    for (pair_id, length), scales in sorted(
        expected_scales_by_pair_length.items()
    ):
        profile, true_class = pair_metadata[pair_id]
        keys = [
            (pair_id, length, float(scale)) for scale in scales
        ]
        outcomes = [outcome_by_pair_length_scale[key] for key in keys]
        decisions = [decision_by_pair_length_scale[key] for key in keys]
        scores = [score_by_pair_length_scale[key] for key in keys]
        paired_cells.append(
            {
                "pair_id": pair_id,
                "source_profile": f"current:{profile}",
                "true_public_class": true_class,
                "observation_length": int(length),
                "expected_legal_scale_factors": list(scales),
                "observed_legal_scale_factors": list(scales),
                "distinct_legal_scale_count": len(scales),
                "exact_legal_scale_coverage": True,
                "all_scale_abstention_decisions_agree":
                    len(set(decisions)) == 1,
                "all_scale_predictions_or_abstentions_agree":
                    len(set(outcomes)) == 1,
                "maximum_final_score_spread": float(
                    max(scores) - min(scores)
                ),
            }
        )

    expected_lengths_by_pair_scale: dict[
        tuple[str, float], tuple[int, ...]
    ] = {}
    observed_lengths_by_pair_scale: dict[
        tuple[str, float], tuple[int, ...]
    ] = {}
    for keys, destination in (
        (expected_observation_keys, expected_lengths_by_pair_scale),
        (observed_observation_keys, observed_lengths_by_pair_scale),
    ):
        groups: dict[tuple[str, float], set[int]] = {}
        for pair_id, length, scale in keys:
            groups.setdefault((pair_id, scale), set()).add(length)
        destination.update(
            {
                key: tuple(sorted(lengths))
                for key, lengths in groups.items()
            }
        )
    if (
        observed_lengths_by_pair_scale
        != expected_lengths_by_pair_scale
        or any(
            len(lengths) < MIN_PAIRED_DISTINCT_RUNTIME_LENGTHS
            for lengths in expected_lengths_by_pair_scale.values()
        )
    ):
        raise RuntimeError(
            "held causal-prefix observation-length coverage changed"
        )
    causal_prefix_cells: list[dict[str, Any]] = []
    for (pair_id, scale), lengths in sorted(
        expected_lengths_by_pair_scale.items()
    ):
        profile, true_class = pair_metadata[pair_id]
        keys = [
            (pair_id, int(length), float(scale)) for length in lengths
        ]
        outcomes = [outcome_by_pair_length_scale[key] for key in keys]
        decisions = [decision_by_pair_length_scale[key] for key in keys]
        scores = [score_by_pair_length_scale[key] for key in keys]
        causal_prefix_cells.append(
            {
                "pair_id": pair_id,
                "source_profile": f"current:{profile}",
                "true_public_class": true_class,
                "physical_scale_factor": float(scale),
                "expected_observation_lengths": list(lengths),
                "observed_observation_lengths": list(lengths),
                "distinct_observation_length_count": len(lengths),
                "exact_observation_length_coverage": True,
                "all_length_abstention_decisions_agree":
                    len(set(decisions)) == 1,
                "all_length_predictions_or_abstentions_agree":
                    len(set(outcomes)) == 1,
                "maximum_final_score_spread": float(
                    max(scores) - min(scores)
                ),
            }
        )
    for pair_id in pair_ids:
        profile, true_class = pair_metadata[pair_id]
        all_keys = sorted(
            {
                (row.pair_id, length, float(row.scale_factor))
                for row in corpus.rows
                if row.pair_id == pair_id
                for length in row.available_lengths
                if length in RUNTIME_INPUT_LENGTHS
            },
            key=lambda item: (item[1], item[2]),
        )
        if all(key in decision_by_pair_length_scale for key in all_keys):
            distinct_lengths = sorted({key[1] for key in all_keys})
            distinct_scales = sorted({key[2] for key in all_keys})
            all_decisions = [
                decision_by_pair_length_scale[key] for key in all_keys
            ]
            all_outcomes = [
                outcome_by_pair_length_scale[key] for key in all_keys
            ]
            all_scores = [
                score_by_pair_length_scale[key] for key in all_keys
            ]
            paired_length_scale_cells.append(
                {
                    "pair_id": pair_id,
                    "source_profile": f"current:{profile}",
                    "true_public_class": true_class,
                    "eligible_observations": len(all_keys),
                    "distinct_runtime_lengths": distinct_lengths,
                    "distinct_runtime_length_count": len(distinct_lengths),
                    "distinct_scale_factors": distinct_scales,
                    "distinct_scale_factor_count": len(distinct_scales),
                    "all_length_scale_abstention_decisions_agree":
                        len(set(all_decisions)) == 1,
                    "all_length_scale_predictions_or_abstentions_agree":
                        len(set(all_outcomes)) == 1,
                    "maximum_final_score_spread": float(
                        max(all_scores) - min(all_scores)
                    ),
                }
            )
    paired_by_profile: dict[str, dict[str, Any]] = {}
    for profile in sorted(
        set(row["source_profile"] for row in paired_length_scale_cells)
    ):
        cells = [
            row
            for row in paired_length_scale_cells
            if row["source_profile"] == profile
        ]
        expected = sum(
            1
            for pair_id in pair_ids
            if f"current:{pair_metadata[pair_id][0]}" == profile
        )
        paired_by_profile[profile] = {
            "expected_pair_cells": expected,
            "cells": len(cells),
            "complete_pair_coverage_rate": float(
                len(cells) / expected if expected else 0.0
            ),
            "minimum_distinct_runtime_lengths": min(
                (
                    int(row["distinct_runtime_length_count"])
                    for row in cells
                ),
                default=0,
            ),
            "prediction_or_abstention_agreement_rate": (
                float(
                    np.mean(
                        [
                            row[
                                "all_length_scale_predictions_or_abstentions_"
                                "agree"
                            ]
                            for row in cells
                        ]
                    )
                )
                if cells
                else None
            ),
            "failing_pair_ids": [
                str(row["pair_id"])
                for row in cells
                if not row[
                    "all_length_scale_predictions_or_abstentions_agree"
                ]
            ],
        }
    physical_by_length: dict[str, dict[str, Any]] = {}
    for length in RUNTIME_INPUT_LENGTHS:
        cells = [
            row
            for row in paired_cells
            if int(row["observation_length"]) == length
        ]
        physical_by_length[str(length)] = {
            "expected_cells": sum(
                1
                for _pair_id, expected_length
                in expected_scales_by_pair_length
                if expected_length == length
            ),
            "observed_cells": len(cells),
            "minimum_distinct_legal_scales": min(
                (
                    int(row["distinct_legal_scale_count"])
                    for row in cells
                ),
                default=0,
            ),
            "prediction_or_abstention_agreement_rate": (
                float(
                    np.mean(
                        [
                            row[
                                "all_scale_predictions_or_abstentions_agree"
                            ]
                            for row in cells
                        ]
                    )
                )
                if cells
                else None
            ),
            "failing_pair_ids": [
                str(row["pair_id"])
                for row in cells
                if not row[
                    "all_scale_predictions_or_abstentions_agree"
                ]
            ],
        }
    causal_by_profile: dict[str, dict[str, Any]] = {}
    for profile in sorted(
        {row["source_profile"] for row in causal_prefix_cells}
    ):
        cells = [
            row
            for row in causal_prefix_cells
            if row["source_profile"] == profile
        ]
        expected = sum(
            1
            for pair_id, _scale in expected_lengths_by_pair_scale
            if f"current:{pair_metadata[pair_id][0]}" == profile
        )
        causal_by_profile[profile] = {
            "expected_pair_scale_cells": expected,
            "observed_pair_scale_cells": len(cells),
            "exact_cell_coverage": len(cells) == expected,
            "minimum_distinct_observation_lengths": min(
                (
                    int(row["distinct_observation_length_count"])
                    for row in cells
                ),
                default=0,
            ),
            "prediction_or_abstention_agreement_rate": (
                float(
                    np.mean(
                        [
                            row[
                                "all_length_predictions_or_abstentions_agree"
                            ]
                            for row in cells
                        ]
                    )
                )
                if cells
                else None
            ),
            "failing_pair_scale_cells": [
                {
                    "pair_id": str(row["pair_id"]),
                    "physical_scale_factor": float(
                        row["physical_scale_factor"]
                    ),
                }
                for row in cells
                if not row[
                    "all_length_predictions_or_abstentions_agree"
                ]
            ],
        }
    all_rejected = np.concatenate(list(rejected_output.values()))
    all_closed_correct = np.concatenate(
        list(closed_correct_output.values())
    )
    all_rejected_closed_correct = np.concatenate(
        list(rejected_closed_correct_output.values())
    )
    report = {
        "population_role": (
            "independent held current SignalLab service phase/receiver/physical-"
            "scale validation; never used in training, enrollment calibration, "
            "checkpoint selection, or open-set design"
        ),
        "audit": corpus.audit,
        "reference_separation": separation,
        "per_length_scale": per_length_scale,
        "combined": {
            "rows": int(len(all_rejected)),
            "pooled_false_unknown_rate": float(np.mean(all_rejected)),
            "pooled_correct_and_accepted_rate": float(
                np.mean(np.concatenate(list(correct_output.values())))
            ),
            "pooled_closed_head_accuracy": float(
                np.mean(all_closed_correct)
            ),
            "pooled_closed_head_correct_rows": int(
                np.sum(all_closed_correct)
            ),
            "pooled_rejected_closed_head_correct_rows": int(
                np.sum(all_rejected_closed_correct)
            ),
            "pooled_abstention_damage_given_closed_head_correct": (
                float(
                    np.sum(all_rejected_closed_correct)
                    / np.sum(all_closed_correct)
                )
                if np.any(all_closed_correct)
                else None
            ),
            "per_length_false_unknown_rate": {
                str(length): float(np.mean(values))
                for length, values in rejected_output.items()
            },
            "per_length_correct_and_accepted_rate": {
                str(length): float(np.mean(values))
                for length, values in correct_output.items()
            },
            "per_length_closed_head_accuracy": {
                str(length): float(np.mean(values))
                for length, values in closed_correct_output.items()
            },
            "per_length_closed_head_correct_rows": {
                str(length): int(np.sum(values))
                for length, values in closed_correct_output.items()
            },
            "per_length_rejected_closed_head_correct_rows": {
                str(length): int(np.sum(values))
                for length, values in rejected_closed_correct_output.items()
            },
            "per_length_abstention_damage_given_closed_head_correct": {
                str(length): (
                    float(
                        np.sum(rejected_closed_correct_output[length])
                        / np.sum(values)
                    )
                    if np.any(values)
                    else None
                )
                for length, values in closed_correct_output.items()
            },
        },
        "paired_physical_scale_invariance": {
            "eligibility_rule": (
                "each exact legal pair/observation-length scale set from "
                "the corpus; at least two distinct legal scales"
            ),
            "expected_cells": len(expected_scales_by_pair_length),
            "cells": len(paired_cells),
            "expected_observation_cells": sum(
                len(scales)
                for scales in expected_scales_by_pair_length.values()
            ),
            "observed_observation_cells": sum(
                len(scales)
                for scales in observed_scales_by_pair_length.values()
            ),
            "expected_legal_scales_by_pair_observation_length": {
                f"{pair_id}|N={length}": list(scales)
                for (pair_id, length), scales in sorted(
                    expected_scales_by_pair_length.items()
                )
            },
            "observed_legal_scales_by_pair_observation_length": {
                f"{pair_id}|N={length}": list(scales)
                for (pair_id, length), scales in sorted(
                    observed_scales_by_pair_length.items()
                )
            },
            "exact_eligible_cell_coverage": (
                observed_scales_by_pair_length
                == expected_scales_by_pair_length
            ),
            "minimum_distinct_legal_scales": min(
                (
                    len(scales)
                    for scales in expected_scales_by_pair_length.values()
                ),
                default=0,
            ),
            "abstention_decision_agreement_rate": (
                float(
                    np.mean(
                        [
                            row["all_scale_abstention_decisions_agree"]
                            for row in paired_cells
                        ]
                    )
                )
                if paired_cells
                else None
            ),
            "prediction_or_abstention_agreement_rate": (
                float(
                    np.mean(
                        [
                            row[
                                "all_scale_predictions_or_abstentions_agree"
                            ]
                            for row in paired_cells
                        ]
                    )
                )
                if paired_cells
                else None
            ),
            "by_observation_length": physical_by_length,
            "worst_final_score_spread": (
                max(
                    row["maximum_final_score_spread"]
                    for row in paired_cells
                )
                if paired_cells
                else None
            ),
            "pair_observation_length_cells": paired_cells,
        },
        "causal_prefix_length_invariance": {
            "scope": "current_route_fixed_first_4096",
            "eligibility_rule": (
                "each exact legal pair/physical-scale observation-length "
                "set from the corpus; at least two distinct lengths"
            ),
            "expected_cells": len(expected_lengths_by_pair_scale),
            "cells": len(causal_prefix_cells),
            "expected_observation_cells": sum(
                len(lengths)
                for lengths in expected_lengths_by_pair_scale.values()
            ),
            "observed_observation_cells": sum(
                len(lengths)
                for lengths in observed_lengths_by_pair_scale.values()
            ),
            "expected_observation_lengths_by_pair_scale": {
                f"{pair_id}|scale={scale:g}": list(lengths)
                for (pair_id, scale), lengths in sorted(
                    expected_lengths_by_pair_scale.items()
                )
            },
            "observed_observation_lengths_by_pair_scale": {
                f"{pair_id}|scale={scale:g}": list(lengths)
                for (pair_id, scale), lengths in sorted(
                    observed_lengths_by_pair_scale.items()
                )
            },
            "exact_eligible_cell_coverage": (
                observed_lengths_by_pair_scale
                == expected_lengths_by_pair_scale
            ),
            "minimum_distinct_observation_lengths": min(
                (
                    len(lengths)
                    for lengths in expected_lengths_by_pair_scale.values()
                ),
                default=0,
            ),
            "prediction_or_abstention_agreement_rate": (
                float(
                    np.mean(
                        [
                            row[
                                "all_length_predictions_or_abstentions_agree"
                            ]
                            for row in causal_prefix_cells
                        ]
                    )
                )
                if causal_prefix_cells
                else None
            ),
            "by_source_profile": causal_by_profile,
            "pair_scale_cells": causal_prefix_cells,
        },
        "paired_length_scale_invariance": {
            "expected_pair_cells": len(pair_ids),
            "cells": len(paired_length_scale_cells),
            "expected_observation_cells": len(expected_observation_keys),
            "observed_observation_cells": len(observed_observation_keys),
            "expected_profile_legal_cell_rows":
                expected_profile_legal_cell_rows,
            "observed_profile_legal_cell_rows":
                observed_profile_legal_cell_rows,
            "expected_profile_legal_cells":
                len(expected_profile_legal_cell_rows),
            "observed_profile_legal_cells":
                len(observed_profile_legal_cell_rows),
            "expected_profile_legal_observations": int(
                sum(expected_profile_legal_cell_rows.values())
            ),
            "observed_profile_legal_observations": int(
                sum(observed_profile_legal_cell_rows.values())
            ),
            "minimum_distinct_runtime_lengths_required":
                MIN_PAIRED_DISTINCT_RUNTIME_LENGTHS,
            "distinct_runtime_lengths_minimum": (
                min(
                    row["distinct_runtime_length_count"]
                    for row in paired_length_scale_cells
                )
                if paired_length_scale_cells
                else None
            ),
            "eligible_observations_minimum": (
                min(
                    row["eligible_observations"]
                    for row in paired_length_scale_cells
                )
                if paired_length_scale_cells
                else None
            ),
            "eligible_observations_maximum": (
                max(
                    row["eligible_observations"]
                    for row in paired_length_scale_cells
                )
                if paired_length_scale_cells
                else None
            ),
            "complete_pair_coverage_rate": float(
                len(paired_length_scale_cells) / len(pair_ids)
            ),
            "abstention_decision_agreement_rate": (
                float(
                    np.mean(
                        [
                            row[
                                "all_length_scale_abstention_decisions_agree"
                            ]
                            for row in paired_length_scale_cells
                        ]
                    )
                )
                if paired_length_scale_cells
                else None
            ),
            "prediction_or_abstention_agreement_rate": (
                float(
                    np.mean(
                        [
                            row[
                                "all_length_scale_predictions_or_abstentions_"
                                "agree"
                            ]
                            for row in paired_length_scale_cells
                        ]
                    )
                )
                if paired_length_scale_cells
                else None
            ),
            "worst_final_score_spread": (
                max(
                    row["maximum_final_score_spread"]
                    for row in paired_length_scale_cells
                )
                if paired_length_scale_cells
                else None
            ),
            "by_source_profile": paired_by_profile,
            "pair_cells": paired_length_scale_cells,
        },
        "gate_cells": {
            "true_class": gate_class_cells,
            "source_profile": gate_profile_cells,
        },
    }
    return report, score_output


def evaluate_novelty(
    context: DevelopmentContext,
    policy: Mapping[str, Any],
    known_score_by_source: Mapping[str, Mapping[int, np.ndarray]],
    *,
    seeds: Sequence[int],
    n_each: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reports: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    for seed in seeds:
        novelty, generated = generate_novelty_raw(
            seed, n_each=n_each, lengths=RUNTIME_INPUT_LENGTHS
        )
        provenance[str(seed)] = generated
        seed_report: dict[str, Any] = {}
        for prototype_source in PROTOTYPE_SOURCE_ROUTES:
            if prototype_source not in known_score_by_source:
                raise ValueError(
                    f"novelty evaluation lacks {prototype_source} known scores"
                )
            source_report: dict[str, Any] = {}
            for length in RUNTIME_INPUT_LENGTHS:
                known = np.asarray(
                    known_score_by_source[prototype_source][length],
                    dtype=np.float64,
                )
                length_report: dict[str, Any] = {}
                family_final_scores: list[np.ndarray] = []
                for family in NOVELTY_FAMILIES:
                    scores = score_observed_raw_population(
                        context,
                        policy,
                        novelty[family][length],
                        observation_length=length,
                        prototype_source=prototype_source,
                    )
                    summary = _population_summary(scores)
                    summary.update(
                        {
                            "observation_runtime_input_length": int(length),
                            "effective_runtime_input_length": int(
                                effective_runtime_input_length(
                                    policy,
                                    prototype_source=prototype_source,
                                    observation_length=length,
                                )
                            ),
                            "auroc_vs_route_known": float(
                                auroc(scores.final_score, known)
                            ),
                            "unknown_recall":
                                float(np.mean(scores.rejected)),
                            "source": (
                                "fresh prefix-matched synthetic time-domain "
                                f"through {prototype_source} prototype route"
                            ),
                        }
                    )
                    length_report[family] = summary
                    family_final_scores.append(scores.final_score)
                all_novelty = np.concatenate(family_final_scores)
                length_report["overall"] = {
                    "auroc_vs_route_known": float(
                        auroc(all_novelty, known)
                    )
                }
                source_report[str(length)] = length_report
            seed_report[prototype_source] = source_report
        reports[str(seed)] = seed_report
    return reports, provenance


def _aggregate_closed_head_accuracy(
    summaries: Sequence[Mapping[str, Any]],
    *,
    section: str,
) -> dict[str, dict[str, Any]]:
    totals: dict[str, list[int]] = {}
    for summary in summaries:
        rows = summary.get(section)
        if not isinstance(rows, Mapping) or not rows:
            raise ValueError(f"closed-head summary lacks {section}")
        for name, row in rows.items():
            if not isinstance(row, Mapping):
                raise ValueError(f"closed-head {section} row is invalid")
            count = int(row.get("rows", 0))
            correct = int(row.get("closed_head_correct_rows", -1))
            reported_accuracy = float(
                row.get("closed_head_accuracy", float("nan"))
            )
            if (
                count <= 0
                or correct < 0
                or correct > count
                or not math.isfinite(reported_accuracy)
                or not math.isclose(
                    reported_accuracy,
                    correct / count,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    f"closed-head {section} counts are invalid"
                )
            target = totals.setdefault(str(name), [0, 0])
            target[0] += count
            target[1] += correct
    return {
        name: {
            "rows": counts[0],
            "closed_head_correct_rows": counts[1],
            "closed_head_accuracy": float(counts[1] / counts[0]),
        }
        for name, counts in sorted(totals.items())
    }


def _present_class_balanced_accuracy(
    summary: Mapping[str, Any],
) -> float:
    per_class = summary.get("per_true_public_class")
    if not isinstance(per_class, Mapping) or not per_class:
        raise ValueError("closed-head summary has no present classes")
    values = []
    for row in per_class.values():
        if not isinstance(row, Mapping):
            continue
        count = int(row.get("rows", 0))
        correct = int(row.get("closed_head_correct_rows", -1))
        reported = float(
            row.get("closed_head_accuracy", float("nan"))
        )
        if (
            count <= 0
            or correct < 0
            or correct > count
            or not math.isclose(
                reported,
                correct / count,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("present-class integer counts disagree")
        values.append(float(correct / count))
    if (
        len(values) != len(per_class)
        or not values
        or not all(math.isfinite(value) for value in values)
    ):
        raise ValueError("present-class closed accuracy is invalid")
    return float(np.mean(values))


def _positive_integer_count_map(
    value: Any,
    *,
    label: str,
) -> dict[str, int]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} is missing or empty")
    result: dict[str, int] = {}
    for raw_key, raw_count in value.items():
        if (
            not isinstance(raw_key, str)
            or not raw_key
            or isinstance(raw_count, bool)
            or not isinstance(raw_count, int)
            or raw_count <= 0
        ):
            raise ValueError(f"{label} contains an invalid cell count")
        result[raw_key] = raw_count
    return result


def _integer_closed_head_accuracy(
    row: Mapping[str, Any],
    *,
    label: str,
) -> float:
    count = int(row.get("rows", 0))
    correct = int(row.get("closed_head_correct_rows", -1))
    reported = float(row.get("closed_head_accuracy", float("nan")))
    if (
        count <= 0
        or correct < 0
        or correct > count
        or not math.isfinite(reported)
        or not math.isclose(
            reported,
            correct / count,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(f"{label} integer counts disagree")
    return float(correct / count)


def _classifier_gate_summary(
    known: Mapping[str, Any],
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Build role-aware current-route classifier gates.

    These gates use the routed closed prediction before abstention.  Historical
    correctness is intentionally absent: historical head quality is frozen
    diagnostic context, while release claims concern the current trusted route
    and independent current service corpus.
    """
    if "current" in known and "historical" in known:
        role = "design_current_routed_selection"
        current = known["current"]
        if not isinstance(current, Mapping):
            raise ValueError("design current classifier report is invalid")
        per_length = current.get("per_length")
        pooled = current.get("pooled")
        if (
            not isinstance(per_length, Mapping)
            or set(per_length)
            != {str(length) for length in RUNTIME_INPUT_LENGTHS}
            or not isinstance(pooled, Mapping)
        ):
            raise ValueError(
                "design current classifier length coverage is incomplete"
            )
        summaries = [
            per_length[str(length)] for length in RUNTIME_INPUT_LENGTHS
        ]
        if not all(isinstance(row, Mapping) for row in summaries):
            raise ValueError("design current length summary is invalid")
        per_length_accuracy: dict[str, float] = {}
        for length in RUNTIME_INPUT_LENGTHS:
            row = per_length[str(length)]
            count = int(row["rows"])
            correct = int(row["closed_head_correct_rows"])
            accuracy = float(correct / count)
            if not math.isclose(
                float(row["closed_head_accuracy"]),
                accuracy,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "design current duplicate closed accuracy disagrees "
                    "with integer counts"
                )
            per_length_accuracy[str(length)] = accuracy
        class_rows = _aggregate_closed_head_accuracy(
            summaries, section="per_true_public_class"
        )
        profile_rows = _aggregate_closed_head_accuracy(
            summaries, section="per_source_profile"
        )
        expected_profiles = {
            f"current:{profile}"
            for profile in corpus_data.CURRENT_PROFILES
        }
        expected_classes = set(corpus_data.CURRENT_CLASSES)
        wifi_by_length = {
            str(length): float(
                per_length[str(length)]["per_source_profile"][
                    WIFI_HR_DSSS_PROFILE
                ]["closed_head_accuracy"]
            )
            for length in RUNTIME_INPUT_LENGTHS
            if WIFI_HR_DSSS_PROFILE
            in per_length[str(length)]["per_source_profile"]
        }
        coverage = (
            set(profile_rows) == expected_profiles
            and set(class_rows) == expected_classes
            and set(wifi_by_length)
            == {str(length) for length in RUNTIME_INPUT_LENGTHS}
            and sum(int(row["rows"]) for row in summaries)
            == int(pooled["rows"])
        )
        pooled_rows = sum(int(row["rows"]) for row in summaries)
        pooled_correct = sum(
            int(row["closed_head_correct_rows"]) for row in summaries
        )
        pooled_accuracy = float(pooled_correct / pooled_rows)
        if (
            int(pooled["rows"]) != pooled_rows
            or int(pooled["closed_head_correct_rows"]) != pooled_correct
            or not math.isclose(
                float(pooled["closed_head_accuracy"]),
                pooled_accuracy,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                "design current pooled closed accuracy disagrees with "
                "integer counts"
            )
        pooled_present_balanced = float(
            np.mean(
                [
                    row["closed_head_accuracy"]
                    for row in class_rows.values()
                ]
            )
        )
        pooled_profile_balanced = float(
            np.mean(
                [
                    row["closed_head_accuracy"]
                    for row in profile_rows.values()
                ]
            )
        )
        worst_length = min(
            per_length_accuracy,
            key=lambda key: per_length_accuracy[key],
        )
        worst_wifi_length = min(
            wifi_by_length, key=lambda key: wifi_by_length[key]
        )
        worst_profile = min(
            profile_rows,
            key=lambda key: profile_rows[key]["closed_head_accuracy"],
        )
        return role, {
            "classifier_design_current_route_coverage": {
                "value": coverage,
                "expected_profiles": len(expected_profiles),
                "observed_profiles": len(profile_rows),
                "expected_runtime_lengths":
                    list(RUNTIME_INPUT_LENGTHS),
                "passes": coverage,
            },
            "classifier_design_current_pooled_closed_accuracy": {
                "value": pooled_accuracy,
                "floor":
                    DESIGN_CURRENT_POOLED_CLOSED_ACCURACY_FLOOR,
                "rows": int(pooled["rows"]),
                "passes": (
                    coverage
                    and pooled_accuracy
                    >= DESIGN_CURRENT_POOLED_CLOSED_ACCURACY_FLOOR
                ),
            },
            "classifier_design_current_every_length_closed_accuracy": {
                "value": per_length_accuracy[worst_length],
                "floor":
                    DESIGN_CURRENT_EVERY_LENGTH_CLOSED_ACCURACY_FLOOR,
                "worst_length": int(worst_length),
                "per_length": per_length_accuracy,
                "passes": (
                    coverage
                    and min(per_length_accuracy.values())
                    >= DESIGN_CURRENT_EVERY_LENGTH_CLOSED_ACCURACY_FLOOR
                ),
            },
            "classifier_design_current_pooled_present_class_balanced_accuracy": {
                "value": pooled_present_balanced,
                "floor":
                    DESIGN_CURRENT_PRESENT_CLASS_BALANCED_ACCURACY_FLOOR,
                "per_class": class_rows,
                "passes": (
                    coverage
                    and pooled_present_balanced
                    >= DESIGN_CURRENT_PRESENT_CLASS_BALANCED_ACCURACY_FLOOR
                ),
            },
            "classifier_design_current_pooled_profile_balanced_accuracy": {
                "value": pooled_profile_balanced,
                "floor":
                    DESIGN_CURRENT_PROFILE_BALANCED_ACCURACY_FLOOR,
                "per_profile": profile_rows,
                "worst_profile": worst_profile,
                "passes": (
                    coverage
                    and pooled_profile_balanced
                    >= DESIGN_CURRENT_PROFILE_BALANCED_ACCURACY_FLOOR
                ),
            },
            "classifier_design_current_worst_profile_closed_accuracy": {
                "value":
                    profile_rows[worst_profile]["closed_head_accuracy"],
                "floor":
                    DESIGN_CURRENT_WORST_PROFILE_CLOSED_ACCURACY_FLOOR,
                "worst_profile": worst_profile,
                "rows": profile_rows[worst_profile]["rows"],
                "passes": (
                    coverage
                    and profile_rows[worst_profile][
                        "closed_head_accuracy"
                    ]
                    >=
                    DESIGN_CURRENT_WORST_PROFILE_CLOSED_ACCURACY_FLOOR
                ),
            },
            "classifier_design_current_wifi_hr_dsss_worst_length_accuracy": {
                "value": wifi_by_length[worst_wifi_length],
                "floor":
                    DESIGN_CURRENT_WIFI_HR_DSSS_ACCURACY_FLOOR,
                "worst_length": int(worst_wifi_length),
                "per_length": wifi_by_length,
                "passes": (
                    coverage
                    and min(wifi_by_length.values())
                    >= DESIGN_CURRENT_WIFI_HR_DSSS_ACCURACY_FLOOR
                ),
            },
        }

    if "per_length_scale" not in known:
        raise ValueError("classifier gate role cannot be determined")
    role = "independent_held_current_service"
    per_length_scale = known["per_length_scale"]
    combined = known.get("combined")
    paired = known.get("paired_length_scale_invariance")
    if (
        not isinstance(per_length_scale, Mapping)
        or set(per_length_scale)
        != {str(length) for length in RUNTIME_INPUT_LENGTHS}
        or not isinstance(combined, Mapping)
        or not isinstance(paired, Mapping)
    ):
        raise ValueError("held classifier report coverage is incomplete")
    cell_rows: dict[str, Mapping[str, Any]] = {}
    for length in RUNTIME_INPUT_LENGTHS:
        length_rows = per_length_scale[str(length)]
        if (
            not isinstance(length_rows, Mapping)
            or set(length_rows)
            != {str(scale) for scale in scale_eval.SCALE_FACTORS}
        ):
            raise ValueError(
                f"held classifier N{length} scale coverage is incomplete"
            )
        for scale in scale_eval.SCALE_FACTORS:
            row = length_rows[str(scale)]
            if not isinstance(row, Mapping):
                raise ValueError("held classifier cell is invalid")
            cell_rows[
                f"N{length}/scale={float(scale):g}"
            ] = row
    summaries = list(cell_rows.values())
    profile_rows = _aggregate_closed_head_accuracy(
        summaries, section="per_source_profile"
    )
    expected_profiles = {
        f"current:{profile}"
        for profile in corpus_data.CURRENT_PROFILES
    }
    cell_accuracy = {
        key: _integer_closed_head_accuracy(
            row, label=f"held cell {key}"
        )
        for key, row in cell_rows.items()
    }
    cell_present_balanced = {
        key: _present_class_balanced_accuracy(row)
        for key, row in cell_rows.items()
    }
    profile_cell_accuracy: dict[str, float] = {}
    summary_profile_legal_cell_rows: dict[str, int] = {}
    for length in RUNTIME_INPUT_LENGTHS:
        for scale in scale_eval.SCALE_FACTORS:
            row = per_length_scale[str(length)][str(scale)]
            profiles = row.get("per_source_profile")
            if not isinstance(profiles, Mapping) or not profiles:
                raise ValueError(
                    "held classifier cell has no source profiles"
                )
            for profile, profile_row in profiles.items():
                if not isinstance(profile_row, Mapping):
                    raise ValueError(
                        "held classifier profile cell is invalid"
                    )
                legal_key = _profile_legal_cell_key(
                    str(profile), int(length), float(scale)
                )
                if legal_key in summary_profile_legal_cell_rows:
                    raise ValueError(
                        "held classifier repeats a profile legal cell"
                    )
                summary_profile_legal_cell_rows[legal_key] = int(
                    profile_row["rows"]
                )
                profile_cell_accuracy[legal_key] = (
                    _integer_closed_head_accuracy(
                        profile_row,
                        label=f"held profile cell {legal_key}",
                    )
                )
    expected_profile_legal_cell_rows = _positive_integer_count_map(
        paired.get("expected_profile_legal_cell_rows"),
        label="held expected profile legal-cell map",
    )
    reported_observed_profile_legal_cell_rows = (
        _positive_integer_count_map(
            paired.get("observed_profile_legal_cell_rows"),
            label="held observed profile legal-cell map",
        )
    )
    exact_profile_legal_coverage = (
        expected_profile_legal_cell_rows
        == reported_observed_profile_legal_cell_rows
        == summary_profile_legal_cell_rows
    )
    expected_wifi_cell_rows = {
        key: count
        for key, count in expected_profile_legal_cell_rows.items()
        if key.startswith(f"{WIFI_HR_DSSS_PROFILE}|")
    }
    wifi_cells = {
        key: accuracy
        for key, accuracy in profile_cell_accuracy.items()
        if key.startswith(f"{WIFI_HR_DSSS_PROFILE}|")
    }
    if not expected_wifi_cell_rows or not wifi_cells:
        raise ValueError("held WiFi HR-DSSS legal-cell map is empty")
    expected_global_cells = len(RUNTIME_INPUT_LENGTHS) * len(
        scale_eval.SCALE_FACTORS
    )
    expected_observations = sum(
        expected_profile_legal_cell_rows.values()
    )
    observed_observations = sum(
        summary_profile_legal_cell_rows.values()
    )
    coverage = (
        set(profile_rows) == expected_profiles
        and len(cell_rows) == expected_global_cells
        and exact_profile_legal_coverage
        and set(wifi_cells) == set(expected_wifi_cell_rows)
        and len(expected_profile_legal_cell_rows)
        == EXPECTED_HELD_PROFILE_LEGAL_CELLS
        and expected_observations == EXPECTED_HELD_OBSERVATION_CELLS
        and int(paired.get("expected_pair_cells", -1))
        == EXPECTED_HELD_PAIR_CELLS
        and int(paired.get("cells", -1))
        == EXPECTED_HELD_PAIR_CELLS
        and int(paired.get("expected_profile_legal_cells", -1))
        == len(expected_profile_legal_cell_rows)
        and int(paired.get("observed_profile_legal_cells", -1))
        == len(summary_profile_legal_cell_rows)
        and int(
            paired.get("expected_profile_legal_observations", -1)
        )
        == expected_observations
        and int(
            paired.get("observed_profile_legal_observations", -1)
        )
        == observed_observations
        and int(paired.get("expected_observation_cells", -1))
        == expected_observations
        and int(paired.get("observed_observation_cells", -1))
        == observed_observations
        and float(paired.get("complete_pair_coverage_rate", 0.0)) == 1.0
    )
    pooled_rows = sum(int(row["rows"]) for row in summaries)
    pooled_correct = sum(
        int(row["closed_head_correct_rows"]) for row in summaries
    )
    pooled_accuracy = float(pooled_correct / pooled_rows)
    if (
        int(combined["rows"]) != pooled_rows
        or int(combined["pooled_closed_head_correct_rows"])
        != pooled_correct
        or not math.isclose(
            float(combined["pooled_closed_head_accuracy"]),
            pooled_accuracy,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(
            "held pooled closed accuracy disagrees with integer counts"
        )
    profile_balanced = float(
        np.mean(
            [
                row["closed_head_accuracy"]
                for row in profile_rows.values()
            ]
        )
    )
    worst_cell = min(cell_accuracy, key=lambda key: cell_accuracy[key])
    worst_balanced_cell = min(
        cell_present_balanced,
        key=lambda key: cell_present_balanced[key],
    )
    worst_wifi = min(wifi_cells, key=lambda key: wifi_cells[key])
    worst_profile = min(
        profile_rows,
        key=lambda key: profile_rows[key]["closed_head_accuracy"],
    )
    worst_profile_cell = min(
        profile_cell_accuracy,
        key=lambda key: profile_cell_accuracy[key],
    )
    return role, {
        "classifier_held_current_service_coverage": {
            "value": coverage,
            "expected_profiles": len(expected_profiles),
            "observed_profiles": len(profile_rows),
            "expected_global_length_scale_cells": expected_global_cells,
            "observed_length_scale_cells": len(cell_rows),
            "expected_profile_legal_cells":
                len(expected_profile_legal_cell_rows),
            "observed_profile_legal_cells":
                len(summary_profile_legal_cell_rows),
            "expected_profile_legal_observations":
                expected_observations,
            "observed_profile_legal_observations":
                observed_observations,
            "exact_profile_legal_cell_key_and_count_coverage":
                exact_profile_legal_coverage,
            "expected_wifi_legal_cells":
                len(expected_wifi_cell_rows),
            "observed_wifi_legal_cells": len(wifi_cells),
            "passes": coverage,
        },
        "classifier_held_current_service_pooled_closed_accuracy": {
            "value": pooled_accuracy,
            "floor": HELD_CURRENT_POOLED_CLOSED_ACCURACY_FLOOR,
            "rows": int(combined["rows"]),
            "passes": (
                coverage
                and pooled_accuracy
                >= HELD_CURRENT_POOLED_CLOSED_ACCURACY_FLOOR
            ),
        },
        "classifier_held_current_service_every_legal_cell_closed_accuracy": {
            "value": cell_accuracy[worst_cell],
            "floor":
                HELD_CURRENT_EVERY_LEGAL_CELL_CLOSED_ACCURACY_FLOOR,
            "worst_cell": worst_cell,
            "per_cell": cell_accuracy,
            "passes": (
                coverage
                and min(cell_accuracy.values())
                >= HELD_CURRENT_EVERY_LEGAL_CELL_CLOSED_ACCURACY_FLOOR
            ),
        },
        "classifier_held_current_service_worst_cell_present_class_balanced_accuracy": {
            "value": cell_present_balanced[worst_balanced_cell],
            "floor":
                HELD_CURRENT_WORST_CELL_PRESENT_CLASS_BALANCED_ACCURACY_FLOOR,
            "worst_cell": worst_balanced_cell,
            "per_cell": cell_present_balanced,
            "passes": (
                coverage
                and min(cell_present_balanced.values())
                >=
                HELD_CURRENT_WORST_CELL_PRESENT_CLASS_BALANCED_ACCURACY_FLOOR
            ),
        },
        "classifier_held_current_service_pooled_profile_balanced_accuracy": {
            "value": profile_balanced,
            "floor":
                HELD_CURRENT_PROFILE_BALANCED_ACCURACY_FLOOR,
            "per_profile": profile_rows,
            "worst_profile": worst_profile,
            "passes": (
                coverage
                and profile_balanced
                >= HELD_CURRENT_PROFILE_BALANCED_ACCURACY_FLOOR
            ),
        },
        "classifier_held_current_service_worst_profile_pooled_closed_accuracy": {
            "value":
                profile_rows[worst_profile]["closed_head_accuracy"],
            "floor":
                HELD_CURRENT_WORST_PROFILE_POOLED_CLOSED_ACCURACY_FLOOR,
            "worst_profile": worst_profile,
            "rows": profile_rows[worst_profile]["rows"],
            "passes": (
                coverage
                and profile_rows[worst_profile]["closed_head_accuracy"]
                >=
                HELD_CURRENT_WORST_PROFILE_POOLED_CLOSED_ACCURACY_FLOOR
            ),
        },
        "classifier_held_current_service_worst_profile_legal_cell_closed_accuracy": {
            "value": profile_cell_accuracy[worst_profile_cell],
            "floor":
                HELD_CURRENT_WORST_PROFILE_LEGAL_CELL_CLOSED_ACCURACY_FLOOR,
            "worst_cell": worst_profile_cell,
            "per_profile_cell": profile_cell_accuracy,
            "passes": (
                coverage
                and min(profile_cell_accuracy.values())
                >=
                HELD_CURRENT_WORST_PROFILE_LEGAL_CELL_CLOSED_ACCURACY_FLOOR
            ),
        },
        "classifier_held_current_service_wifi_hr_dsss_worst_legal_cell_accuracy": {
            "value": wifi_cells[worst_wifi],
            "floor":
                HELD_CURRENT_WIFI_HR_DSSS_ACCURACY_FLOOR,
            "worst_cell": worst_wifi,
            "per_cell": wifi_cells,
            "passes": (
                coverage
                and min(wifi_cells.values())
                >= HELD_CURRENT_WIFI_HR_DSSS_ACCURACY_FLOOR
            ),
        },
    }


def _abstention_cell_gate(
    cells: Sequence[Mapping[str, Any]],
    *,
    label_field: str,
    label_output_field: str,
) -> dict[str, Any]:
    """Build a support-aware conditional abstention gate.

    Closed-head mistakes are classifier errors, not abstention errors.  A cell
    therefore contributes only its closed-head-correct rows, and an
    underpowered cell is reported but cannot manufacture a pass or a failure.
    """
    if not cells:
        raise ValueError(f"{label_field} abstention cells are empty")
    checked: list[dict[str, Any]] = []
    for source in cells:
        rows = int(source["rows"])
        correct = int(source["closed_head_correct_rows"])
        rejected_correct = int(
            source["rejected_closed_head_correct_rows"]
        )
        value = source.get(
            "abstention_damage_given_closed_head_correct"
        )
        if (
            rows <= 0
            or correct < 0
            or correct > rows
            or rejected_correct < 0
            or rejected_correct > correct
            or (
                correct == 0
                and value is not None
            )
            or (
                correct > 0
                and (
                    value is None
                    or not math.isfinite(float(value))
                    or not math.isclose(
                        float(value),
                        rejected_correct / correct,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                )
            )
        ):
            raise ValueError(
                f"{label_field} conditional abstention cell is invalid"
            )
        checked.append(
            {
                **dict(source),
                "abstention_damage_given_closed_head_correct": (
                    None if value is None else float(value)
                ),
                "eligible_for_cell_gate": (
                    correct >= KNOWN_CELL_MIN_CLOSED_CORRECT_ROWS
                ),
            }
        )
    eligible = [
        row for row in checked if row["eligible_for_cell_gate"]
    ]
    underpowered = [
        {
            "source": str(row["source"]),
            "length": int(row["length"]),
            label_output_field: str(row[label_field]),
            "rows": int(row["rows"]),
            "closed_head_correct_rows": int(
                row["closed_head_correct_rows"]
            ),
            **(
                {
                    "physical_scale_factor": float(row["scale_factor"])
                }
                if "scale_factor" in row
                else {}
            ),
        }
        for row in checked
        if not row["eligible_for_cell_gate"]
    ]
    common = {
        "ceiling": KNOWN_FALSE_UNKNOWN_CEILING,
        "conditioning": "closed_head_prediction_equals_truth",
        "minimum_closed_head_correct_rows":
            KNOWN_CELL_MIN_CLOSED_CORRECT_ROWS,
        "total_cells": len(checked),
        "eligible_cells": len(eligible),
        "underpowered_cells": underpowered,
    }
    if not eligible:
        return {
            **common,
            "applicable": False,
            "value": None,
            "passes": None,
            "reason": (
                "no cell has the precommitted minimum closed-head-correct "
                "support; this cell claim is not used in all_pass"
            ),
        }
    worst = max(
        eligible,
        key=lambda row: float(
            row["abstention_damage_given_closed_head_correct"]
        ),
    )
    output = {
        **common,
        "applicable": True,
        "value": float(
            worst["abstention_damage_given_closed_head_correct"]
        ),
        "source": str(worst["source"]),
        "length": int(worst["length"]),
        label_output_field: str(worst[label_field]),
        "rows": int(worst["rows"]),
        "closed_head_accuracy": float(worst["closed_head_accuracy"]),
        "closed_head_correct_rows": int(
            worst["closed_head_correct_rows"]
        ),
        "rejected_closed_head_correct_rows": int(
            worst["rejected_closed_head_correct_rows"]
        ),
        "passes": (
            float(
                worst["abstention_damage_given_closed_head_correct"]
            )
            <= KNOWN_FALSE_UNKNOWN_CEILING
        ),
    }
    if "scale_factor" in worst:
        output["physical_scale_factor"] = float(worst["scale_factor"])
    return output


def _gate_summary(
    known: Mapping[str, Any],
    novelty: Mapping[str, Any],
) -> dict[str, Any]:
    combined = known["combined"]
    length_damage = {
        str(length): float(value)
        for length, value in combined[
            "per_length_abstention_damage_given_closed_head_correct"
        ].items()
        if value is not None
    }
    length_correct_rows = {
        str(length): int(value)
        for length, value in combined[
            "per_length_closed_head_correct_rows"
        ].items()
    }
    if (
        not length_damage
        or set(length_damage) != set(length_correct_rows)
        or any(value <= 0 for value in length_correct_rows.values())
    ):
        raise ValueError(
            "known conditional abstention length support is incomplete"
        )
    worst_length = max(
        length_damage, key=lambda length: length_damage[length]
    )
    pooled_damage = combined.get(
        "pooled_abstention_damage_given_closed_head_correct"
    )
    pooled_correct_rows = int(
        combined.get("pooled_closed_head_correct_rows", 0)
    )
    if (
        pooled_damage is None
        or pooled_correct_rows <= 0
        or not math.isfinite(float(pooled_damage))
    ):
        raise ValueError("known pooled conditional abstention is invalid")
    pooled_damage = float(pooled_damage)
    classifier_role, classifier_gates = _classifier_gate_summary(known)
    gates: dict[str, Any] = {
        **classifier_gates,
        "known_abstention_damage_given_closed_head_correct_pooled": {
            "value": pooled_damage,
            "ceiling": KNOWN_FALSE_UNKNOWN_CEILING,
            "closed_head_correct_rows": pooled_correct_rows,
            "conditioning": "closed_head_prediction_equals_truth",
            "passes": pooled_damage <= KNOWN_FALSE_UNKNOWN_CEILING,
        },
        "known_abstention_damage_given_closed_head_correct_worst_length": {
            "value": length_damage[worst_length],
            "ceiling": KNOWN_FALSE_UNKNOWN_CEILING,
            "length": int(worst_length),
            "closed_head_correct_rows":
                length_correct_rows[worst_length],
            "conditioning": "closed_head_prediction_equals_truth",
            "passes": (
                length_damage[worst_length]
                <= KNOWN_FALSE_UNKNOWN_CEILING
            ),
        },
    }
    if classifier_role == "design_current_routed_selection":
        causal = known["current"].get(
            "causal_prefix_length_agreement"
        )
        if not isinstance(causal, Mapping):
            raise ValueError(
                "design current causal-prefix agreement report is missing"
            )
        expected_cells = causal.get(
            "expected_base_identity_row_length_cells"
        )
        observed_cells = causal.get(
            "observed_base_identity_row_length_cells"
        )
        by_profile = causal.get("by_source_profile")
        expected_current_profiles = {
            f"current:{profile}"
            for profile in corpus_data.CURRENT_PROFILES
        }
        if (
            not isinstance(expected_cells, Mapping)
            or not expected_cells
            or observed_cells != expected_cells
            or causal.get(
                "exact_expected_observed_cell_coverage"
            ) is not True
            or not isinstance(by_profile, Mapping)
            or set(by_profile) != expected_current_profiles
            or int(causal.get("stored_observations", -1))
            != EXPECTED_DESIGN_CURRENT_SELECTION_STORED_OBSERVATIONS
            or len(expected_cells)
            != EXPECTED_DESIGN_CURRENT_SELECTION_OBSERVATION_CELLS
            or causal.get(
                "stored_observations_by_distinct_observation_length_count"
            )
            != {
                "2":
                    EXPECTED_DESIGN_CURRENT_SELECTION_TWO_LENGTH_OBSERVATIONS,
                "3":
                    EXPECTED_DESIGN_CURRENT_SELECTION_THREE_LENGTH_OBSERVATIONS,
            }
        ):
            raise ValueError(
                "design current causal-prefix coverage is incomplete"
            )
        pooled_agreement = float(
            causal.get(
                "prediction_or_abstention_agreement_rate",
                float("nan"),
            )
        )
        profile_rows: dict[str, dict[str, Any]] = {}
        for profile, raw_row in sorted(by_profile.items()):
            if not isinstance(raw_row, Mapping):
                raise ValueError(
                    "design current causal-prefix profile row is invalid"
                )
            value = float(
                raw_row.get(
                    "prediction_or_abstention_agreement_rate",
                    float("nan"),
                )
            )
            exact_coverage = (
                raw_row.get("exact_length_coverage") is True
                and int(
                    raw_row.get(
                        "expected_observation_length_cells", -1
                    )
                )
                == int(
                    raw_row.get(
                        "observed_observation_length_cells", -2
                    )
                )
            )
            profile_rows[str(profile)] = {
                "value": value,
                "exact_length_coverage": exact_coverage,
                "stored_observations": int(
                    raw_row.get("stored_observations", 0)
                ),
                "passes": (
                    exact_coverage
                    and math.isfinite(value)
                    and value
                    >= CURRENT_SELECTION_CAUSAL_AGREEMENT_FLOOR
                ),
            }
        worst_profile = min(
            profile_rows, key=lambda name: profile_rows[name]["value"]
        )
        gates[
            "current_selection_causal_prefix_outcome_agreement_pooled"
        ] = {
            "value": pooled_agreement,
            "floor": CURRENT_SELECTION_CAUSAL_AGREEMENT_FLOOR,
            "expected_observation_length_cells": len(expected_cells),
            "observed_observation_length_cells": len(observed_cells),
            "expected_stored_observations":
                EXPECTED_DESIGN_CURRENT_SELECTION_STORED_OBSERVATIONS,
            "exact_coverage": True,
            "passes": (
                math.isfinite(pooled_agreement)
                and pooled_agreement
                >= CURRENT_SELECTION_CAUSAL_AGREEMENT_FLOOR
            ),
        }
        gates[
            "current_selection_causal_prefix_outcome_agreement_"
            "worst_source_profile"
        ] = {
            **profile_rows[worst_profile],
            "source_profile": worst_profile,
            "floor": CURRENT_SELECTION_CAUSAL_AGREEMENT_FLOOR,
            "all_source_profiles": profile_rows,
            "passes": all(
                row["passes"] is True for row in profile_rows.values()
            ),
        }
    if classifier_role == "independent_held_current_service":
        physical = known.get("paired_physical_scale_invariance")
        causal = known.get("causal_prefix_length_invariance")
        if not isinstance(physical, Mapping) or not isinstance(
            causal, Mapping
        ):
            raise ValueError(
                "held route-scoped invariance reports are incomplete"
            )
        physical_expected = physical.get(
            "expected_legal_scales_by_pair_observation_length"
        )
        physical_observed = physical.get(
            "observed_legal_scales_by_pair_observation_length"
        )
        physical_by_length = physical.get("by_observation_length")
        physical_pair_cells = physical.get(
            "pair_observation_length_cells"
        )
        expected_current_profiles = {
            f"current:{profile}"
            for profile in corpus_data.CURRENT_PROFILES
        }
        physical_legal_cell_coverage = False
        if (
            isinstance(physical_expected, Mapping)
            and isinstance(physical_observed, Mapping)
            and isinstance(physical_pair_cells, list)
            and len(physical_pair_cells)
            == EXPECTED_HELD_PHYSICAL_SCALE_CELLS
        ):
            seen_cell_keys: set[str] = set()
            pair_profile: dict[str, str] = {}
            lengths_by_pair: dict[str, set[int]] = {}
            pairs_by_profile: dict[str, set[str]] = {
                profile: set() for profile in expected_current_profiles
            }
            physical_legal_cell_coverage = True
            all_scales = tuple(
                float(scale) for scale in scale_eval.SCALE_FACTORS
            )
            for raw_cell in physical_pair_cells:
                if not isinstance(raw_cell, Mapping):
                    physical_legal_cell_coverage = False
                    break
                pair_id = raw_cell.get("pair_id")
                profile = raw_cell.get("source_profile")
                try:
                    length = int(raw_cell.get("observation_length"))
                    expected_scales = tuple(
                        float(scale)
                        for scale in raw_cell.get(
                            "expected_legal_scale_factors", ()
                        )
                    )
                    observed_scales = tuple(
                        float(scale)
                        for scale in raw_cell.get(
                            "observed_legal_scale_factors", ()
                        )
                    )
                except (TypeError, ValueError):
                    physical_legal_cell_coverage = False
                    break
                if (
                    not isinstance(pair_id, str)
                    or not pair_id
                    or profile not in expected_current_profiles
                    or length not in RUNTIME_INPUT_LENGTHS
                ):
                    physical_legal_cell_coverage = False
                    break
                required_scales = (
                    (1.5, 2.0)
                    if (
                        profile
                        == "current:bluetooth-le-advertising"
                        and length == 16_384
                    )
                    else all_scales
                )
                cell_key = f"{pair_id}|N={length}"
                try:
                    mapped_expected = tuple(
                        float(scale)
                        for scale in physical_expected.get(cell_key, ())
                    )
                    mapped_observed = tuple(
                        float(scale)
                        for scale in physical_observed.get(cell_key, ())
                    )
                except (TypeError, ValueError):
                    physical_legal_cell_coverage = False
                    break
                previous_profile = pair_profile.setdefault(
                    pair_id, str(profile)
                )
                if (
                    cell_key in seen_cell_keys
                    or previous_profile != profile
                    or expected_scales != required_scales
                    or observed_scales != required_scales
                    or mapped_expected != required_scales
                    or mapped_observed != required_scales
                    or raw_cell.get("exact_legal_scale_coverage") is not True
                    or int(
                        raw_cell.get("distinct_legal_scale_count", -1)
                    )
                    != len(required_scales)
                ):
                    physical_legal_cell_coverage = False
                    break
                seen_cell_keys.add(cell_key)
                pairs_by_profile[str(profile)].add(pair_id)
                lengths_by_pair.setdefault(pair_id, set()).add(length)
            if physical_legal_cell_coverage:
                physical_legal_cell_coverage = (
                    seen_cell_keys == set(physical_expected)
                    == set(physical_observed)
                    and set(pair_profile.values())
                    == expected_current_profiles
                    and all(
                        len(pair_ids)
                        == DEFAULT_HELD_SCALE_REALIZATIONS_PER_PROFILE
                        for pair_ids in pairs_by_profile.values()
                    )
                    and all(
                        lengths == set(RUNTIME_INPUT_LENGTHS)
                        for lengths in lengths_by_pair.values()
                    )
                )
        physical_coverage = (
            isinstance(physical_expected, Mapping)
            and bool(physical_expected)
            and physical_observed == physical_expected
            and physical.get("exact_eligible_cell_coverage") is True
            and int(physical.get("expected_cells", -1))
            == int(physical.get("cells", -2))
            == len(physical_expected)
            == EXPECTED_HELD_PHYSICAL_SCALE_CELLS
            and int(
                physical.get("expected_observation_cells", -1)
            )
            == int(
                physical.get("observed_observation_cells", -2)
            )
            == EXPECTED_HELD_OBSERVATION_CELLS
            and int(
                physical.get("minimum_distinct_legal_scales", 0)
            ) >= 2
            and isinstance(physical_by_length, Mapping)
            and set(physical_by_length)
            == {str(length) for length in RUNTIME_INPUT_LENGTHS}
            and physical_legal_cell_coverage
        )
        physical_value = float(
            physical.get(
                "prediction_or_abstention_agreement_rate",
                float("nan"),
            )
        )
        length_rows: dict[str, dict[str, Any]] = {}
        if isinstance(physical_by_length, Mapping):
            for length, raw_row in sorted(physical_by_length.items()):
                if not isinstance(raw_row, Mapping):
                    raise ValueError(
                        "held physical-scale length row is invalid"
                    )
                value = float(
                    raw_row.get(
                        "prediction_or_abstention_agreement_rate",
                        float("nan"),
                    )
                )
                coverage = (
                    int(raw_row.get("expected_cells", -1))
                    == int(raw_row.get("observed_cells", -2))
                    == EXPECTED_HELD_PHYSICAL_SCALE_CELLS_PER_LENGTH
                    and int(
                        raw_row.get(
                            "minimum_distinct_legal_scales", 0
                        )
                    ) >= 2
                )
                length_rows[str(length)] = {
                    "value": value,
                    "exact_coverage": coverage,
                    "expected_cells": int(
                        raw_row.get("expected_cells", 0)
                    ),
                    "passes": (
                        coverage
                        and math.isfinite(value)
                        and value >= LENGTH_SCALE_DECISION_AGREEMENT_FLOOR
                    ),
                }
        worst_length = min(
            length_rows, key=lambda length: length_rows[length]["value"]
        )
        gates["held_physical_scale_outcome_agreement_pooled"] = {
            "value": physical_value,
            "floor": LENGTH_SCALE_DECISION_AGREEMENT_FLOOR,
            "exact_coverage": physical_coverage,
            "passes": (
                physical_coverage
                and math.isfinite(physical_value)
                and physical_value
                >= LENGTH_SCALE_DECISION_AGREEMENT_FLOOR
            ),
        }
        gates[
            "held_physical_scale_outcome_agreement_worst_"
            "observation_length"
        ] = {
            **length_rows[worst_length],
            "observation_length": int(worst_length),
            "floor": LENGTH_SCALE_DECISION_AGREEMENT_FLOOR,
            "all_observation_lengths": length_rows,
            "passes": all(
                row["passes"] is True for row in length_rows.values()
            ),
        }

        causal_expected = causal.get(
            "expected_observation_lengths_by_pair_scale"
        )
        causal_observed = causal.get(
            "observed_observation_lengths_by_pair_scale"
        )
        causal_by_profile = causal.get("by_source_profile")
        expected_current_profiles = {
            f"current:{profile}"
            for profile in corpus_data.CURRENT_PROFILES
        }
        causal_coverage = (
            isinstance(causal_expected, Mapping)
            and bool(causal_expected)
            and causal_observed == causal_expected
            and causal.get("exact_eligible_cell_coverage") is True
            and int(causal.get("expected_cells", -1))
            == int(causal.get("cells", -2))
            == len(causal_expected)
            == EXPECTED_HELD_CAUSAL_PREFIX_CELLS
            and int(causal.get("expected_observation_cells", -1))
            == int(causal.get("observed_observation_cells", -2))
            == EXPECTED_HELD_OBSERVATION_CELLS
            and int(
                causal.get(
                    "minimum_distinct_observation_lengths", 0
                )
            ) >= MIN_PAIRED_DISTINCT_RUNTIME_LENGTHS
            and isinstance(causal_by_profile, Mapping)
            and set(causal_by_profile) == expected_current_profiles
        )
        causal_value = float(
            causal.get(
                "prediction_or_abstention_agreement_rate",
                float("nan"),
            )
        )
        causal_profile_rows: dict[str, dict[str, Any]] = {}
        if isinstance(causal_by_profile, Mapping):
            for profile, raw_row in sorted(causal_by_profile.items()):
                if not isinstance(raw_row, Mapping):
                    raise ValueError(
                        "held causal-prefix profile row is invalid"
                    )
                value = float(
                    raw_row.get(
                        "prediction_or_abstention_agreement_rate",
                        float("nan"),
                    )
                )
                coverage = (
                    raw_row.get("exact_cell_coverage") is True
                    and int(
                        raw_row.get("expected_pair_scale_cells", -1)
                    )
                    == int(
                        raw_row.get("observed_pair_scale_cells", -2)
                    )
                    == (
                        DEFAULT_HELD_SCALE_REALIZATIONS_PER_PROFILE
                        * len(scale_eval.SCALE_FACTORS)
                    )
                    and int(
                        raw_row.get(
                            "minimum_distinct_observation_lengths", 0
                        )
                    ) >= MIN_PAIRED_DISTINCT_RUNTIME_LENGTHS
                )
                causal_profile_rows[str(profile)] = {
                    "value": value,
                    "exact_coverage": coverage,
                    "passes": (
                        coverage
                        and math.isfinite(value)
                        and value
                        >= CURRENT_SELECTION_CAUSAL_AGREEMENT_FLOOR
                    ),
                }
        causal_coverage = causal_coverage and all(
            row["exact_coverage"] is True
            for row in causal_profile_rows.values()
        )
        causal_worst_profile = min(
            causal_profile_rows,
            key=lambda name: causal_profile_rows[name]["value"],
        )
        gates["held_causal_prefix_outcome_agreement_pooled"] = {
            "value": causal_value,
            "floor": CURRENT_SELECTION_CAUSAL_AGREEMENT_FLOOR,
            "exact_coverage": causal_coverage,
            "passes": (
                causal_coverage
                and math.isfinite(causal_value)
                and causal_value
                >= CURRENT_SELECTION_CAUSAL_AGREEMENT_FLOOR
            ),
        }
        gates[
            "held_causal_prefix_outcome_agreement_worst_source_profile"
        ] = {
            **causal_profile_rows[causal_worst_profile],
            "source_profile": causal_worst_profile,
            "floor": CURRENT_SELECTION_CAUSAL_AGREEMENT_FLOOR,
            "all_source_profiles": causal_profile_rows,
            "passes": all(
                row["passes"] is True
                for row in causal_profile_rows.values()
            ),
        }
    length_scale = known.get("paired_length_scale_invariance")
    if isinstance(length_scale, Mapping):
        agreement = length_scale.get(
            "prediction_or_abstention_agreement_rate"
        )
        cells = int(length_scale.get("cells", 0))
        expected_cells = int(length_scale.get("expected_pair_cells", 0))
        coverage = float(
            length_scale.get("complete_pair_coverage_rate", 0.0)
        )
        distinct_lengths_minimum = int(
            length_scale.get("distinct_runtime_lengths_minimum", 0)
        )
        valid_agreement = (
            (
                (
                    cells
                    == expected_cells
                    == EXPECTED_HELD_PAIR_CELLS
                    and int(
                        length_scale.get(
                            "expected_observation_cells", -1
                        )
                    )
                    == int(
                        length_scale.get(
                            "observed_observation_cells", -2
                        )
                    )
                    == EXPECTED_HELD_OBSERVATION_CELLS
                )
                if classifier_role
                == "independent_held_current_service"
                else cells > 0 and cells == expected_cells
            )
            and agreement is not None
            and math.isfinite(float(agreement))
        )
        value = float(agreement) if valid_agreement else 0.0
        gates["known_paired_length_scale_outcome_agreement"] = {
            "value": value,
            "floor": LENGTH_SCALE_DECISION_AGREEMENT_FLOOR,
            "cells": cells,
            "expected_cells": expected_cells,
            "coverage": coverage,
            "distinct_runtime_lengths_minimum": distinct_lengths_minimum,
            "minimum_distinct_runtime_lengths_required":
                MIN_PAIRED_DISTINCT_RUNTIME_LENGTHS,
            "passes": (
                valid_agreement
                and coverage == 1.0
                and distinct_lengths_minimum
                >= MIN_PAIRED_DISTINCT_RUNTIME_LENGTHS
                and value >= LENGTH_SCALE_DECISION_AGREEMENT_FLOOR
            ),
        }
        profile_agreement = length_scale.get("by_source_profile")
        if not isinstance(profile_agreement, Mapping) or not profile_agreement:
            raise ValueError(
                "paired length/scale report lacks per-profile cells"
            )
        expected_current_profiles = {
            f"current:{profile}"
            for profile in corpus_data.CURRENT_PROFILES
        }
        exact_held_profile_inventory = (
            classifier_role != "independent_held_current_service"
            or set(profile_agreement) == expected_current_profiles
        )
        profile_gate_rows: list[dict[str, Any]] = []
        for profile, row in sorted(profile_agreement.items()):
            if not isinstance(row, Mapping):
                raise ValueError(
                    f"paired length/scale profile {profile!r} is invalid"
                )
            profile_value = row.get(
                "prediction_or_abstention_agreement_rate"
            )
            profile_cells_count = int(row.get("cells", 0))
            profile_expected = int(row.get("expected_pair_cells", 0))
            profile_coverage = float(
                row.get("complete_pair_coverage_rate", 0.0)
            )
            profile_min_lengths = int(
                row.get("minimum_distinct_runtime_lengths", 0)
            )
            profile_valid = (
                profile_cells_count > 0
                and profile_cells_count == profile_expected
                and (
                    classifier_role
                    != "independent_held_current_service"
                    or profile_cells_count
                    == DEFAULT_HELD_SCALE_REALIZATIONS_PER_PROFILE
                )
                and profile_value is not None
                and math.isfinite(float(profile_value))
            )
            profile_gate_rows.append(
                {
                    "source_profile": str(profile),
                    "value": (
                        float(profile_value) if profile_valid else 0.0
                    ),
                    "cells": profile_cells_count,
                    "expected_cells": profile_expected,
                    "coverage": profile_coverage,
                    "minimum_distinct_runtime_lengths":
                        profile_min_lengths,
                    "failing_pair_ids": list(
                        row.get("failing_pair_ids", [])
                    ),
                    "passes": (
                        profile_valid
                        and profile_coverage == 1.0
                        and profile_min_lengths
                        >= MIN_PAIRED_DISTINCT_RUNTIME_LENGTHS
                        and float(profile_value)
                        >= LENGTH_SCALE_DECISION_AGREEMENT_FLOOR
                    ),
                }
            )
        worst_profile_agreement = min(
            profile_gate_rows, key=lambda row: float(row["value"])
        )
        gates[
            "known_paired_length_scale_outcome_agreement_worst_source_profile"
        ] = {
            **worst_profile_agreement,
            "floor": LENGTH_SCALE_DECISION_AGREEMENT_FLOOR,
            "exact_source_profile_inventory":
                exact_held_profile_inventory,
            "expected_source_profiles": (
                len(expected_current_profiles)
                if classifier_role
                == "independent_held_current_service"
                else None
            ),
            "observed_source_profiles": len(profile_agreement),
            "all_source_profile_cells": profile_gate_rows,
            "passes": (
                exact_held_profile_inventory
                and worst_profile_agreement["passes"] is True
            ),
        }
    class_cells = list(known["gate_cells"]["true_class"])
    profile_cells = list(known["gate_cells"]["source_profile"])
    gates[
        "known_abstention_damage_given_closed_head_correct_"
        "worst_true_class_cell"
    ] = _abstention_cell_gate(
        class_cells,
        label_field="name",
        label_output_field="true_public_class",
    )
    gates[
        "known_abstention_damage_given_closed_head_correct_"
        "worst_source_profile_cell"
    ] = _abstention_cell_gate(
        profile_cells,
        label_field="name",
        label_output_field="source_profile",
    )
    for family in NOVELTY_FAMILIES:
        aurocs = [
            float(length_row[family]["auroc_vs_route_known"])
            for seed_row in novelty.values()
            for source_row in seed_row.values()
            for length_row in source_row.values()
        ]
        recalls = [
            float(length_row[family]["unknown_recall"])
            for seed_row in novelty.values()
            for source_row in seed_row.values()
            for length_row in source_row.values()
        ]
        gates[f"{family}_auroc_worst_cell"] = {
            "value": min(aurocs),
            "floor": NOVELTY_AUROC_FLOOR,
            "passes": min(aurocs) >= NOVELTY_AUROC_FLOOR,
        }
        gates[f"{family}_unknown_recall_worst_cell"] = {
            "value": min(recalls),
            "floor": NOVELTY_RECALL_FLOOR,
            "passes": min(recalls) >= NOVELTY_RECALL_FLOOR,
        }
    applicable_gates = [
        row
        for row in gates.values()
        if row.get("applicable", True)
    ]
    return {
        "development_diagnostics_only": True,
        "classifier_gate_role": classifier_role,
        "gates": gates,
        "closed_head_classifier_summary": {
            "role": "combined_historical_current_diagnostic",
            "is_classifier_gate_result": False,
            "pooled_accuracy": float(
                combined["pooled_closed_head_accuracy"]
            ),
            "per_length_accuracy": dict(
                combined["per_length_closed_head_accuracy"]
            ),
            "rationale": (
                "current-route closed correctness has separate role-aware "
                "classifier gates; historical correctness remains diagnostic; "
                "abstention gates measure damage conditional on a correct head"
            ),
        },
        "unconditional_known_rejection_diagnostics_not_gates": {
            "pooled_false_unknown_rate": float(
                combined["pooled_false_unknown_rate"]
            ),
            "per_length_false_unknown_rate": dict(
                combined["per_length_false_unknown_rate"]
            ),
        },
        "all_pass": bool(
            applicable_gates
            and all(row["passes"] is True for row in applicable_gates)
        ),
    }


EXECUTED_SOURCE_RELATIVE_PATHS = (
    "training/canonical_probe.py",
    "training/dataset.py",
    "training/invariant_patch_preprocess.py",
    "training/model.py",
    "training/preprocess.py",
    "training/rfgen.py",
    "training/time_domain_geometry.py",
    "training/time_domain_invariant_patch_preprocess.py",
    "training/train.py",
    "training/zplane_ab/__init__.py",
    "training/zplane_ab/common_split.py",
    "training/zplane_ab/train_common.py",
    "training/zplane_ab/v2_full_variation/canonical_probe_net.py",
    "training/zplane_ab/v2_full_variation/__init__.py",
    "training/zplane_ab/v2_full_variation/complex_multiscale_backbone.py",
    "training/zplane_ab/v2_full_variation/denoise_eval.py",
    "training/zplane_ab/v2_full_variation/equalizer_frontend.py",
    "training/zplane_ab/v2_full_variation/evaluate_ab_v2.py",
    "training/zplane_ab/v2_full_variation/full_split.py",
    "training/zplane_ab/v2_full_variation/ground_state_data.py",
    "training/zplane_ab/v2_full_variation/invariant_fusion.py",
    "training/zplane_ab/v2_full_variation/invariant_patch_cnn.py",
    "training/zplane_ab/v2_full_variation/invariant_patch_data.py",
    "training/zplane_ab/v2_full_variation/length_aug.py",
    "training/zplane_ab/v2_full_variation/multitask_autoencoder.py",
    "training/zplane_ab/v2_full_variation/native_preprocess.py",
    "training/zplane_ab/v2_full_variation/openset_eval.py",
    "training/zplane_ab/v2_full_variation/pool_cache.py",
    "training/zplane_ab/v2_full_variation/run_bounded_dev.py",
    "training/zplane_ab/v2_full_variation/run_corrected_unet.py",
    "training/zplane_ab/v2_full_variation/run_invariant_cnn_dev.py",
    "training/zplane_ab/v2_full_variation/scalar_transfer.py",
    "training/zplane_ab/v2_full_variation/train_common_v2.py",
    "training/zplane_ab/v2_full_variation/train_transfer.py",
    "training/zplane_ab/v2_full_variation/unet_transfer.py",
    "training/zplane_ab/v2_full_variation/vit_backbone.py",
    "training/zplane_ab/v2_full_variation/v3_scale/"
    "export_v3_browser_weights.py",
    "training/zplane_ab/v2_full_variation/v3_scale/noise_prefilter.py",
    "training/zplane_ab/v2_full_variation/v3_scale/pose_degeneracy.py",
    "training/zplane_ab/v2_full_variation/v4_current_source/"
    "calibrate_current_source_openset.py",
    "training/zplane_ab/v2_full_variation/v4_current_source/__init__.py",
    "training/zplane_ab/v2_full_variation/v4_current_source/"
    "current_source_data.py",
    "training/zplane_ab/v2_full_variation/v4_current_source/"
    "evaluate_current_scale.py",
    "training/zplane_ab/v2_full_variation/v4_current_source/"
    "export_current_source_browser_runtime.py",
    "training/zplane_ab/v2_full_variation/v4_current_source/"
    "run_current_source_dev.py",
    "training/zplane_ab/zplane_backbone.py",
)


def _executed_source_paths() -> dict[str, Path]:
    sources = {
        name: (REPO / name).resolve()
        for name in EXECUTED_SOURCE_RELATIVE_PATHS
    }
    missing = sorted(name for name, path in sources.items() if not path.is_file())
    if missing:
        raise RuntimeError(
            "executed-source contract names missing files: "
            + ", ".join(missing)
        )
    return sources


def _source_hashes() -> dict[str, str]:
    return {
        name: _sha256(path)
        for name, path in sorted(_executed_source_paths().items())
    }


def _assert_source_snapshot_unchanged(
    expected: Mapping[str, str],
    *,
    operation: str,
) -> None:
    observed = _source_hashes()
    if dict(expected) == observed:
        return
    changed = sorted(
        name
        for name in set(expected) | set(observed)
        if expected.get(name) != observed.get(name)
    )
    raise RuntimeError(
        f"executed source changed during {operation}: {', '.join(changed)}"
    )


def _build_report(
    *,
    role: str,
    policy_sha256: str,
    policy: Mapping[str, Any],
    seeds: Sequence[int],
    known: Mapping[str, Any],
    novelty: Mapping[str, Any],
    novelty_provenance: Mapping[str, Any],
    n_each: int,
    source_snapshot: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema": REPORT_SCHEMA,
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "complete",
        "role": str(role),
        "development_only": True,
        "release_evidence": False,
        "policy_sha256": str(policy_sha256),
        "classifier_binding": policy["classifier_binding"],
        "seed_protocol": {
            "design_novelty_seed":
                int(policy["design_protocol"]["design_novelty_seed"]),
            "novelty_seeds_consumed_once": [int(seed) for seed in seeds],
            "role": str(role),
            "n_each_per_family_per_length": int(n_each),
            "prefix_rule": "generated once at N16384; exact leading prefixes",
            "generators_use_frequency_transform": False,
        },
        "known_evaluation": known,
        "novelty": novelty,
        "novelty_provenance": novelty_provenance,
        "gate_summary": _gate_summary(known, novelty),
        "evaluation_contract": {
            "known_population": (
                "historical/current development selection; score only; not "
                "independent validation"
                if role == "design"
                else "separately held current SignalLab service phase/"
                "receiver/physical-scale corpus"
            ),
            "synthetic_role": "fresh design/validation novelty; score only",
            "policy_fit_population": "enrollment only",
            "selection_rows_used_to_fit_rank_or_threshold": 0,
            "current_selection_rows_used_to_fit_rank_or_threshold": 0,
            "novelty_rows_used_to_fit_rank_or_threshold": 0,
            "sealed_release_rows_used": 0,
            "consumed_historical_test_rows_used": 0,
            "per_length_reporting": True,
            "per_length_keys_are_observation_lengths": True,
            "runtime_length_policy_by_prototype_source":
                RUNTIME_LENGTH_POLICY_BY_PROTOTYPE_SOURCE,
            "closed_correct_and_accepted_gates": False,
            "historical_closed_head_accuracy_is_diagnostic_only": True,
            "current_route_closed_head_classifier_gates": True,
            "classifier_gate_role": (
                "design_current_routed_selection"
                if role == "design"
                else "independent_held_current_service"
            ),
            "abstention_gates_conditioned_on_closed_head_correct": True,
            "underpowered_cell_minimum_closed_head_correct_rows":
                KNOWN_CELL_MIN_CLOSED_CORRECT_ROWS,
            "paired_prediction_or_abstention_invariance_gate": True,
            "uses_frequency_transform_at_inference": False,
            "uses_frequency_transform_in_novelty_generation": False,
        },
        "executed_source_sha256_start": dict(source_snapshot),
        "executed_source_unchanged_before_publication": True,
    }


def _write_manifest(
    output: Path,
    files: Sequence[Path],
    *,
    role: str,
    validation_ledger_sha256: str | None = None,
) -> dict[str, Any]:
    payload = {
        "schema": MANIFEST_SCHEMA,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "development_only",
        "role": str(role),
        "files_sha256": {
            path.name: _sha256(path)
            for path in sorted(files, key=lambda item: item.name)
        },
        "live_assets_written": False,
        "release_evidence": False,
        "validation_ledger_sha256": validation_ledger_sha256,
    }
    if role == "validate" and not _is_sha256(validation_ledger_sha256):
        raise ValueError("validation manifest needs the claimed ledger SHA-256")
    if role == "design" and validation_ledger_sha256 is not None:
        raise ValueError("design manifest cannot claim validation evidence")
    _write_json(output / MANIFEST_NAME, payload)
    return payload


def _validate_policy_corpus_binding(
    policy: Mapping[str, Any],
    context: DevelopmentContext,
) -> None:
    provenance = policy["provenance"]
    expected = {
        "historical_data_audit_sha256":
            corpus_data.sha256_json(context.historical.audit),
        "current_data_audit_sha256":
            corpus_data.sha256_json(context.current.audit),
    }
    for key, digest in expected.items():
        if provenance.get(key) != digest:
            raise ValueError(
                f"open-set policy is bound to another corpus audit ({key})"
            )


def run_design(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    source_snapshot = _source_hashes()
    output = validate_empty_output(args.output_dir)
    seeds = validate_seed_plan(
        "design",
        args.novelty_seeds,
        design_seed=args.design_seed,
    )
    context = load_development_context(
        args.fusion,
        args.historical_corpus,
        args.current_corpus,
        device_name=args.device,
    )
    template = fit_v4_stage_one_template(
        context.historical,
        context.current,
        patch_length=context.patch_length,
        target_frac=context.target_frac,
        fitting_seed=args.stage_one_fitting_seed,
        fitting_noise_rows=args.stage_one_fitting_noise_n,
    )
    enrollment_raw = _combined_enrollment_raw(
        context.historical, context.current
    )
    enrollment_distance = _enrollment_distance_inputs(context)
    stage_one_by_length: dict[int, np.ndarray] = {}
    phase_linearity_by_length: dict[int, np.ndarray] = {}
    distance_by_length: dict[int, np.ndarray] = {}
    calibration_distance_by_length: dict[int, np.ndarray] = {}
    enrollment_group_by_length: dict[int, np.ndarray] = {}
    enrollment_source_by_length: dict[int, np.ndarray] = {}
    for length in RUNTIME_INPUT_LENGTHS:
        stage_one_by_length[length] = stage_one_scores(
            template,
            enrollment_raw[length],
            length=length,
            patch_length=context.patch_length,
            target_frac=context.target_frac,
        )
        phase_linearity_by_length[length] = (
            instantaneous_phase_linearity_scores(
                enrollment_raw[length]
            )
        )
        distance_by_length[length] = enrollment_distance[length][
            "class_distances"
        ]
        calibration_distance_by_length[length] = enrollment_distance[length][
            "leave_one_base_identity_out_class_distances"
        ]
        enrollment_group_by_length[length] = enrollment_distance[length][
            "groups"
        ]
        enrollment_source_by_length[length] = enrollment_distance[length][
            "prototype_sources"
        ]
        if len(stage_one_by_length[length]) != len(
            distance_by_length[length]
        ) or len(phase_linearity_by_length[length]) != len(
            distance_by_length[length]
        ):
            raise RuntimeError(
                f"N{length} raw/model enrollment populations lost alignment"
            )
    binding = build_classifier_binding(
        context.fusion, args.classifier_asset
    )
    validation_protocol = build_validation_protocol(
        classifier_asset_sha256=
            binding["browser_classifier_asset_sha256"],
        held_scale_manifest_sha256=
            args.held_scale_manifest_sha256,
        held_scale_raw_sha256=args.held_scale_raw_sha256,
    )
    policy = fit_enrollment_policy(
        stage_one_score_by_length=stage_one_by_length,
        phase_linearity_score_by_length=phase_linearity_by_length,
        class_distance_by_length=distance_by_length,
        calibration_class_distance_by_length=
            calibration_distance_by_length,
        enrollment_group_by_length=enrollment_group_by_length,
        enrollment_source_by_length=enrollment_source_by_length,
        stage_one_template=template,
        fusion_binding=binding,
        patch_length=context.patch_length,
        patch_count=context.patch_count,
        target_frac=context.target_frac,
        design_seed=seeds[0],
        validation_protocol=validation_protocol,
        composite_budget=args.composite_budget,
    )
    policy["provenance"] = {
        "source_sha256": source_snapshot,
        "stage_one_v4_fitted_artifact_set_sha256": template.set_sha256,
        "stage_one_fitting_seed": int(args.stage_one_fitting_seed),
        "fusion_dev_metrics_sha256":
            binding["source_fusion_dev_metrics_sha256"],
        "historical_data_audit_sha256":
            corpus_data.sha256_json(context.historical.audit),
        "current_data_audit_sha256":
            corpus_data.sha256_json(context.current.audit),
        "consumed_test_rows_exposed": 0,
    }
    policy = validate_policy(
        policy, expected_fusion_binding=binding
    )
    _validate_policy_corpus_binding(policy, context)
    policy_sha = hashlib.sha256(_json_bytes(policy)).hexdigest()
    known, known_scores = evaluate_known_selection(context, policy)
    novelty, novelty_provenance = evaluate_novelty(
        context,
        policy,
        known_scores,
        seeds=seeds,
        n_each=args.novelty_n,
    )
    report = _build_report(
        role="design",
        policy_sha256=policy_sha,
        policy=policy,
        seeds=seeds,
        known=known,
        novelty=novelty,
        novelty_provenance=novelty_provenance,
        n_each=args.novelty_n,
        source_snapshot=source_snapshot,
    )
    _assert_source_snapshot_unchanged(
        source_snapshot, operation="open-set design before policy publication"
    )
    output.mkdir(parents=True, exist_ok=True)
    policy_path = output / POLICY_NAME
    _write_json(policy_path, policy)
    if _sha256(policy_path) != policy_sha:
        raise RuntimeError("written policy bytes differ from in-memory hash")
    report_path = output / DESIGN_REPORT_NAME
    _assert_source_snapshot_unchanged(
        source_snapshot, operation="open-set design before report publication"
    )
    _write_json(report_path, report)
    _assert_source_snapshot_unchanged(
        source_snapshot, operation="open-set design before manifest publication"
    )
    _write_manifest(
        output, (policy_path, report_path), role="design"
    )
    print(
        "[v4 open-set design] "
        f"all-pass={report['gate_summary']['all_pass']} "
        f"wall={time.perf_counter() - started:.1f}s",
        flush=True,
    )
    print(f"[v4 open-set design] wrote {output}", flush=True)
    return report


def run_validate(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    source_snapshot = _source_hashes()
    output = validate_empty_output(args.output_dir)
    policy_path = reject_sensitive_path(args.policy, "policy input")
    raw_policy = _load_json(policy_path)
    design_seed = raw_policy.get("design_protocol", {}).get(
        "design_novelty_seed"
    )
    seeds = validate_seed_plan(
        "validate",
        args.novelty_seeds,
        design_seed=design_seed,
    )
    context = load_development_context(
        args.fusion,
        args.historical_corpus,
        args.current_corpus,
        device_name=args.device,
    )
    binding = build_classifier_binding(
        context.fusion, args.classifier_asset
    )
    policy = validate_policy(
        raw_policy, expected_fusion_binding=binding
    )
    _validate_policy_corpus_binding(policy, context)
    validation_protocol = validate_validation_protocol(
        policy,
        held_scale_corpus=args.held_scale_corpus,
        novelty_seeds=seeds,
        novelty_n=args.novelty_n,
    )
    policy_sha256 = _sha256(policy_path)
    validation_ledger_path, validation_ledger_sha256 = (
        claim_validation_ledger(
            policy_sha256=policy_sha256,
            protocol=validation_protocol,
        )
    )
    auxiliary_selection, auxiliary_scores = evaluate_known_selection(
        context, policy
    )
    known, current_known_scores = evaluate_held_scale_known(
        context,
        policy,
        args.held_scale_corpus,
        args.current_corpus,
        validation_protocol=validation_protocol,
    )
    novelty, novelty_provenance = evaluate_novelty(
        context,
        policy,
        {
            "historical": auxiliary_scores["historical"],
            "current": current_known_scores,
        },
        seeds=seeds,
        n_each=args.novelty_n,
    )
    report = _build_report(
        role="validate",
        policy_sha256=policy_sha256,
        policy=policy,
        seeds=seeds,
        known=known,
        novelty=novelty,
        novelty_provenance=novelty_provenance,
        n_each=args.novelty_n,
        source_snapshot=source_snapshot,
    )
    report["auxiliary_repeated_development_selection"] = {
        "independent_validation": False,
        "used_for_validation_gates": False,
        "note": (
            "these are the same selection rows visible during classifier "
            "checkpointing/design; retained only as a regression diagnostic"
        ),
        "metrics": auxiliary_selection,
    }
    report["validation_ledger"] = {
        "relative_path":
            validation_protocol["validation_ledger_relative_path"],
        "sha256": validation_ledger_sha256,
        "state": "validation_evidence_claimed",
        "created_before_validation_scoring": True,
        "exists_at_publication": validation_ledger_path.is_file(),
        "absence_or_replacement_invalidates_evidence": True,
    }
    _assert_source_snapshot_unchanged(
        source_snapshot,
        operation="open-set validation before report publication",
    )
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / VALIDATION_REPORT_NAME
    _write_json(report_path, report)
    _assert_source_snapshot_unchanged(
        source_snapshot,
        operation="open-set validation before manifest publication",
    )
    _write_manifest(
        output,
        (report_path,),
        role="validate",
        validation_ledger_sha256=validation_ledger_sha256,
    )
    print(
        "[v4 open-set validate] "
        f"all-pass={report['gate_summary']['all_pass']} "
        f"wall={time.perf_counter() - started:.1f}s",
        flush=True,
    )
    print(f"[v4 open-set validate] wrote {output}", flush=True)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.classifier_asset is None:
        raise ValueError(
            "--classifier-asset is required so the external abstention policy "
            "is SHA-bound to the exact browser classifier JSON"
        )
    if args.role == "design":
        if (
            args.held_scale_manifest_sha256 is None
            or args.held_scale_raw_sha256 is None
        ):
            raise ValueError(
                "role=design requires precommitted --held-scale-manifest-"
                "sha256 and --held-scale-raw-sha256"
            )
        return run_design(args)
    if args.role == "validate":
        if args.policy is None:
            raise ValueError("--policy is required for role=validate")
        if args.held_scale_corpus is None:
            raise ValueError(
                "--held-scale-corpus is required for independent validation"
            )
        return run_validate(args)
    raise ValueError("role must be design or validate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--role", choices=("design", "validate"), required=True)
    parser.add_argument("--fusion", required=True)
    parser.add_argument(
        "--historical-corpus",
        default=str(branch_runner.HISTORICAL_CORPUS),
    )
    parser.add_argument("--current-corpus", required=True)
    parser.add_argument(
        "--held-scale-corpus",
        help=(
            "separately reserved current SignalLab service phase/receiver/"
            "physical-scale corpus; required for role=validate and never read "
            "during role=design"
        ),
    )
    parser.add_argument(
        "--held-scale-manifest-sha256",
        help=(
            "precommitted SHA-256 of held scale_eval.json; required for "
            "role=design without reading validation data"
        ),
    )
    parser.add_argument(
        "--held-scale-raw-sha256",
        help=(
            "precommitted SHA-256 of held scale_eval.f32; required for "
            "role=design without reading validation data"
        ),
    )
    parser.add_argument(
        "--classifier-asset",
        help=(
            "exported time-domain-profile-bank-v4.json; required so the "
            "external policy is SHA-bound to those exact browser bytes"
        ),
    )
    parser.add_argument(
        "--policy",
        help=f"frozen {POLICY_NAME}; required only for role=validate",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--design-seed", type=int, default=DEFAULT_DESIGN_SEED)
    parser.add_argument(
        "--novelty-seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            f"design default [{DEFAULT_DESIGN_SEED}]; validation default "
            f"{list(DEFAULT_VALIDATION_SEEDS)}"
        ),
    )
    parser.add_argument(
        "--novelty-n", type=int, default=DEFAULT_NOVELTY_ROWS
    )
    parser.add_argument(
        "--stage-one-fitting-seed",
        type=int,
        default=DEFAULT_STAGE_ONE_FIT_SEED,
    )
    parser.add_argument(
        "--stage-one-fitting-noise-n",
        type=int,
        default=600,
    )
    parser.add_argument(
        "--composite-budget",
        "--survivor-budget",
        dest="composite_budget",
        type=float,
        default=DEFAULT_COMPOSITE_KNOWN_BUDGET,
        help=(
            "known enrollment false-unknown budget for the final composite; "
            "--survivor-budget is retained as a deprecated alias"
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.novelty_seeds is None:
        args.novelty_seeds = (
            [args.design_seed]
            if args.role == "design"
            else list(DEFAULT_VALIDATION_SEEDS)
        )
    if args.role == "validate":
        if (
            args.composite_budget != DEFAULT_COMPOSITE_KNOWN_BUDGET
            or args.stage_one_fitting_seed != DEFAULT_STAGE_ONE_FIT_SEED
            or args.stage_one_fitting_noise_n != 600
        ):
            raise ValueError(
                "validation may not supply or alter fitting budgets; it loads "
                "the frozen policy verbatim"
            )
    run(args)


if __name__ == "__main__":
    main()
