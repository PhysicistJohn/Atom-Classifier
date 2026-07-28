#!/usr/bin/env python3
"""Fail-closed promotion of the v3 dual-fusion staging package.

Promotion reads no corpus and performs no training or calibration. It accepts
only the exact committed dual staging package plus a complete sealed
seed-20260735 evaluator-v3 report. The four JSON assets are rewritten to
``status: release``; the release binding is then rebuilt against the rewritten
role/policy byte hashes before the complete directory is installed atomically.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import secrets
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
DEFAULT_STAGING_PACKAGE = REPO / "src/embedding/assets-v3-dual-staging"
DEFAULT_RELEASE_PACKAGE = REPO / "src/embedding/assets-v3-release"
LIVE_V2_ASSETS = REPO / "src/embedding/assets"
LEGACY_V3_STAGING = REPO / "src/embedding/assets-v3-staging"

PACKAGE_MANIFEST = "runtime-package-manifest.json"
REJECTOR_WEIGHTS = "time-domain-v3-rejector-weights.json"
CLASSIFIER_WEIGHTS = "time-domain-v3-classifier-weights.json"
OPENSET_POLICY = "time-domain-v3-openset-policy.json"
DUAL_BINDING = "time-domain-v3-dual-binding.json"
ASSET_NAMES = (
    REJECTOR_WEIGHTS,
    CLASSIFIER_WEIGHTS,
    OPENSET_POLICY,
    DUAL_BINDING,
)
STAGING_FILE_NAMES = frozenset((*ASSET_NAMES, PACKAGE_MANIFEST))

PACKAGE_SCHEMA = "atomos.v3.time-domain-classifier.dual-runtime-package"
PACKAGE_SCHEMA_VERSION = 1
FUSION_SCHEMA = "atomos.v3.time-domain-invariant-fusion.browser-weights"
OPENSET_SCHEMA = "atomos.v3.time-domain-openset.staged"
BINDING_SCHEMA = "atomos.v3.time-domain-dual-fusion.binding"
CANDIDATE_SCHEMA = "time-domain-v3-dual-release-candidate-v1"
VALIDATION_EVIDENCE_SCHEMA = (
    "time-domain-v3-decoupled-validation-evidence-v1"
)
PREVALIDATION_CONTRACT_SCHEMA = "time-domain-v3-decoupled-candidate-v1"
BUNDLE_SCHEMA = "atomos.v3.time-domain-invariant-fusion.runtime-bundle"
BUNDLE_KIND = "v3-time-domain-centered-invariant-fusion"
BUNDLE_SCHEMA_VERSION = 1

REJECTOR_ROLE = "known_unknown_rejector"
CLASSIFIER_ROLE = "accepted_known_classifier"
REJECTOR_RESPONSIBILITY = "known_unknown_only"
CLASSIFIER_RESPONSIBILITY = "accepted_known_label_only"
CANDIDATE_ID = "v3.3-decoupled-8k-classifier-4k-rejector"

STAGING_STATUS = "staging_not_release"
RELEASE_STATUS = "release"
MAX_DEPLOYABLE_BYTES = 25 * 1024 * 1024
RELEASE_SEED = 20260735
HISTORICAL_RELAXED_SEED = 20260734
EVALUATOR_SCHEMA = 3
EVALUATION_VERSION = "time-domain-v3-release-evaluation-v3-dual-fusion"
EXPECTED_RELEASE_ROOT = (
    REPO
    / "training/artifacts/releases"
    / "invariant_fusion_v3_sealed_seed20260735"
)
EXPECTED_PROTOCOL_FIXTURE = (
    REPO / "tools/time-domain-v3-expected-evaluation-protocol-seed20260735.json"
)
EXPECTED_PROTOCOL_FIXTURE_SHA256 = (
    "40ef572beeabd18c38e55fdc5001dbfb0365dcd81dac5ff6ea1b95ceca163f7e"
)
EXPECTED_EVALUATOR = (
    REPO
    / "training/zplane_ab/v2_full_variation/v3_scale"
    / "evaluate_v3_release_suite.py"
)
EXPECTED_EVALUATOR_SHA256 = (
    "09b3ccd1fa9030f6dc1720b8faf3b1a60e5a8359bb300445bfa34a745c99351a"
)
EXPECTED_V2_EVALUATOR = (
    REPO
    / "training/zplane_ab/v2_full_variation"
    / "evaluate_invariant_release_suite.py"
)
EXPECTED_V2_EVALUATOR_SHA256 = (
    "1b8137b4c222a857a91f340730137fefd3fe17a026d9ba5eb172e7fd774c0541"
)
EXPECTED_STAGED_SOURCE = (
    REPO
    / "training/zplane_ab/v2_full_variation/v3_scale"
    / "fit_v3_openset_staged.py"
)
EXPECTED_LEDGER_TRANSITION = {
    "admission": "exact_post_validation_seed_ledger_transition",
    "validated_sha256": (
        "6d6d252802029cd57cab1429640d0b285b117caa97a89b994037469e267aa176"
    ),
    "current_sha256": (
        "1ebfecede89eb3d393c8922d0862827efda6356671995656594b536ed19f7514"
    ),
    "normalized_ast_sha256": (
        "86dbb18d8b83845e4f9c067f672b8f69c9fbbe2b9ccf4216755f7eeb8096e967"
    ),
    "excluded_top_level_assignments": [
        "SPENT_NOVELTY_SEEDS",
        "FIRST_CLEAN_NOVELTY_SEED",
        "SEED_LEDGER_NOTE",
    ],
    "validated_parent_commit": (
        "6f6e1e05d94d457e16940ad1d2c14c6d95bbc422"
    ),
    "ledger_commit": "5ebbd07f763470ff1fc27be04e2e340c6171cc63",
    "full_index_diff_sha256": (
        "c071726b44a04d535084b3c10e76eda2c3b61767a542a29173873b95f8466d03"
    ),
    "consumed_validation_seeds": [20260950, 20260951],
    "next_clean_novelty_seed": 20260952,
    "candidate_inference_behavior_changed": False,
}
REQUIRED_CAPTURE_LENGTHS = (4096, 8192, 16384, 32768)
PHYSICAL_SCALE_FACTORS = (0.5, 0.75, 1.0, 1.5, 2.0)
PREFILTER_BUNDLE_FILES = (
    "coefficients.npy",
    "intercept.npy",
    "mean.npy",
    "meta.json",
    "scale.npy",
    "threshold_score.npy",
)
EXPECTED_FUSION_FILES = frozenset(
    {
        "complex_center.npy",
        "dev_metrics.json",
        "feature_mean.npy",
        "feature_std.npy",
        "fusion_prototypes.npy",
        "fusion_state_dict.pt",
        "real_center.npy",
    }
)

EXPECTED_BUNDLE_ASSETS = frozenset(
    {
        "complex_center.npy",
        "feature_mean.npy",
        "feature_std.npy",
        "fusion_prototypes.npy",
        "fusion_state_dict.pt",
        "probe_fixture.json",
        "real_center.npy",
    }
)
EXPECTED_STAGED_ASSETS = frozenset(
    {
        "v3_branch_lof_components.npz",
        "v3_open_policy_stage_two.npz",
        "v3_staged_composite_policy.npz",
    }
)

STRICT_V2_GATE_FLOORS = {
    "closed_fine": 0.72,
    "closed_family": 0.82,
    "closed_high_snr": 0.78,
    "closed_clean": 0.85,
    "five_shot": 0.85,
    "open_auroc_overall": 0.72,
    "open_auroc_noise": 0.80,
    "open_auroc_chirp": 0.80,
    "open_known_false_unknown_max": 0.10,
    "open_unknown_recall_noise": 0.10,
    "open_unknown_recall_chirp": 0.10,
    "invariant_worst_balanced_accuracy": 0.75,
    "invariant_max_mean_pairwise_cosine": 0.78,
    "length_pair_prediction_agreement": 0.80,
    "length_pair_embedding_cosine": 0.85,
    "scale_pair_prediction_agreement": 0.75,
    "scale_pair_embedding_cosine": 0.80,
}
BOOLEAN_GATE_NAMES = frozenset(
    {"prefix_nesting", "dependency_provenance", "start_probe_excluded"}
)
CANDIDATE_GATE_NAME = "candidate_sha_bound"
NUMERIC_GATE_CONTRACT = {
    "closed_fine_worst_length": (0.72, "min"),
    "closed_family_worst_length": (0.82, "min"),
    "closed_high_snr_worst_length": (0.78, "min"),
    "closed_clean_worst_length": (0.85, "min"),
    "five_shot_worst_length_balanced": (0.85, "min"),
    "open_auroc_overall_worst_length": (0.72, "min"),
    "open_auroc_noise_worst_length": (0.80, "min"),
    "open_auroc_chirp_worst_length": (0.80, "min"),
    "open_known_false_unknown_worst_length": (0.10, "max"),
    "open_unknown_recall_noise_worst_length": (0.10, "min"),
    "open_unknown_recall_chirp_worst_length": (0.10, "min"),
    "length_worst_balanced_accuracy": (0.75, "min"),
    "length_max_mean_pairwise_cosine": (0.78, "max"),
    "length_worst_prediction_agreement_to_matched": (0.80, "min"),
    "length_worst_embedding_cosine_to_matched": (0.85, "min"),
    "physical_scale_worst_balanced_accuracy": (0.75, "min"),
    "physical_scale_max_mean_pairwise_cosine": (0.78, "max"),
    "physical_scale_worst_prediction_agreement_to_factor1": (0.75, "min"),
    "physical_scale_worst_embedding_cosine_to_factor1": (0.80, "min"),
}
EXPECTED_GATE_NAMES = frozenset(
    {*BOOLEAN_GATE_NAMES, CANDIDATE_GATE_NAME, *NUMERIC_GATE_CONTRACT}
)
if len(EXPECTED_GATE_NAMES) != 23:  # pragma: no cover
    raise RuntimeError("strict release gate contract must contain 23 gates")

REPORT_TOP_KEYS = frozenset(
    {
        "schema",
        "status",
        "release_evidence",
        "evaluation_version",
        "development_data_loaded",
        "retraining_performed",
        "recalibration_performed",
        "architecture_contract",
        "candidate",
        "closed_per_length",
        "family_per_length",
        "high_snr_per_length",
        "low_snr_per_length",
        "clean_subset_per_length",
        "impaired_subset_per_length",
        "five_shot_predeclared_per_length",
        "open_staged_per_length",
        "staged_known_decisions_per_length",
        "matched_length_sweep",
        "physical_scale_sweep",
        "gates",
        "historical_gate_redeclaration",
        "all_release_gates_pass",
        "provenance",
    }
)
CANDIDATE_COMPONENT_KEYS = frozenset(
    {
        "candidate_contract",
        "validation_evidence",
        "frozen_prevalidation_contract",
        "classifier_runtime_bundle",
        "rejector_runtime_bundle",
        "classifier_fusion",
        "rejector_fusion",
        "staged_validation",
        "stage_one_prefilter",
        "browser_assets",
        "staging_package_manifest",
        "dual_binding",
        "dual_binding_sha256",
        "source_sha256",
    }
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class PromotionError(RuntimeError):
    """The supplied package/report cannot safely be promoted."""


@dataclass(frozen=True)
class PromotionPolicy:
    repo_root: Path
    staging_package: Path
    release_package: Path
    live_v2_assets: Path
    legacy_v3_staging: Path = LEGACY_V3_STAGING
    require_git_tracking: bool = True


@dataclass(frozen=True)
class VerifiedStaging:
    manifest: dict[str, Any]
    manifest_raw: bytes
    manifest_sha256: str
    asset_bytes: dict[str, bytes]
    asset_payloads: dict[str, dict[str, Any]]
    asset_records: dict[str, dict[str, Any]]
    tracked_paths: tuple[Path, ...]


@dataclass(frozen=True)
class VerifiedEvaluation:
    report_sha256: str
    candidate_sha256: str
    candidate_artifacts: dict[str, Any]
    release_intent_sha256: str
    release_manifest_sha256: str


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise PromotionError(f"{label} must be a lowercase SHA-256")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PromotionError(f"{label} must be an object")
    return value


def _exact_keys(
    value: Mapping[str, Any], expected: set[str] | frozenset[str], label: str
) -> None:
    if set(value) != set(expected):
        raise PromotionError(
            f"{label} keys differ: expected {sorted(expected)}, got "
            f"{sorted(value)}"
        )


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PromotionError(f"JSON object repeats key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise PromotionError(f"non-finite JSON constant {value!r} is forbidden")


def _parse_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_no_duplicate_object,
            parse_constant=_reject_json_constant,
        )
    except PromotionError:
        raise
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise PromotionError(f"{label} is not strict UTF-8 JSON") from exc
    return dict(_mapping(value, label))


def _read_file(path: Path, label: str) -> bytes:
    """Read one regular file without following any symlink component."""
    absolute = Path(os.path.abspath(os.fspath(path)))
    parts = absolute.parts
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    directory_flags = (
        flags | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = flags | getattr(os, "O_NOFOLLOW", 0)
    directory_fd: int | None = None
    file_fd: int | None = None
    try:
        directory_fd = os.open(absolute.anchor, directory_flags)
        for component in parts[1:-1]:
            next_fd = os.open(
                component,
                directory_flags,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        file_fd = os.open(parts[-1], file_flags, dir_fd=directory_fd)
        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            raise PromotionError(
                f"{label} must be a regular non-symlink file"
            )
        chunks: list[bytes] = []
        while True:
            block = os.read(file_fd, 1 << 20)
            if not block:
                return b"".join(chunks)
            chunks.append(block)
    except PromotionError:
        raise
    except OSError as exc:
        raise PromotionError(f"cannot read {label}") from exc
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if directory_fd is not None:
            os.close(directory_fd)


def _compact_json_bytes(payload: Any) -> bytes:
    try:
        text = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise PromotionError("cannot serialize release asset") from exc
    return f"{text}\n".encode("utf-8")


def _manifest_json_bytes(payload: Any) -> bytes:
    try:
        text = json.dumps(payload, indent=1, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise PromotionError("cannot serialize release manifest") from exc
    return f"{text}\n".encode("utf-8")


def _is_exact_int(value: Any) -> bool:
    return type(value) is int


def _is_finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _file_record(value: Any, label: str) -> dict[str, Any]:
    record = _mapping(value, label)
    size = record.get("bytes")
    if type(size) is not int or size <= 0:
        raise PromotionError(f"{label}.bytes must be a positive integer")
    return {"bytes": size, "sha256": _sha(record.get("sha256"), label)}


def _lexical_path(value: Any, base: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PromotionError(f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return Path(os.path.abspath(os.fspath(path)))


def _require_within(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PromotionError(f"{label} escapes the repository") from exc


def _reject_symlink_chain(
    path: Path,
    root: Path,
    label: str,
    *,
    allow_missing_leaf: bool = False,
) -> None:
    _require_within(path, root, label)
    relative = path.relative_to(root)
    current = root
    if root.is_symlink():
        raise PromotionError(f"{label} crosses a symlink")
    for index, component in enumerate(relative.parts):
        current /= component
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as exc:
            if allow_missing_leaf and index == len(relative.parts) - 1:
                return
            raise PromotionError(f"{label} does not exist") from exc
        if stat.S_ISLNK(mode):
            raise PromotionError(f"{label} crosses a symlink")


def _resolve_path(value: Any, repo: Path, label: str) -> Path:
    path = _lexical_path(value, repo, label)
    _reject_symlink_chain(path, repo, label)
    return path


def _resolve_package_path(
    value: Any,
    package_manifest: Path,
    repo: Path,
    label: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise PromotionError(f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    if path.is_absolute():
        raise PromotionError(f"{label} must be package-relative")
    path = _lexical_path(
        os.fspath(package_manifest.parent / path),
        repo,
        label,
    )
    _reject_symlink_chain(path, repo, label)
    return path


def _verify_file_record(
    record: Mapping[str, Any],
    *,
    path: Path,
    label: str,
) -> tuple[dict[str, Any], bytes]:
    base = _file_record(record, label)
    raw = _read_file(path, label)
    if len(raw) != base["bytes"] or _sha256_bytes(raw) != base["sha256"]:
        raise PromotionError(f"{label} differs from its byte record")
    return base, raw


def _verify_json_record(
    record: Mapping[str, Any],
    *,
    repo: Path,
    label: str,
    schema: str | None = None,
    status: str | None = None,
) -> tuple[Path, str, dict[str, Any]]:
    path = _resolve_path(record.get("path"), repo, f"{label}.path")
    digest = _sha(record.get("sha256"), f"{label}.sha256")
    raw = _read_file(path, label)
    if _sha256_bytes(raw) != digest:
        raise PromotionError(f"{label} SHA differs from current bytes")
    payload = _parse_json(raw, label)
    expected_schema = schema if schema is not None else record.get("schema")
    expected_status = status if status is not None else record.get("status")
    if expected_schema is not None and (
        payload.get("schema") != expected_schema
        or record.get("schema") != expected_schema
    ):
        raise PromotionError(f"{label} schema differs")
    if expected_status is not None and (
        payload.get("status") != expected_status
        or record.get("status") != expected_status
    ):
        raise PromotionError(f"{label} status differs")
    return path, digest, payload


def _run_git(repo: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise PromotionError("Git is required for promotion") from exc


def _verify_git_tracking(repo: Path, paths: tuple[Path, ...]) -> None:
    top = _run_git(repo, ["rev-parse", "--show-toplevel"])
    if top.returncode != 0 or Path(top.stdout.strip()).resolve() != repo:
        raise PromotionError("promotion policy root is not this Git worktree")
    relatives: list[str] = []
    for path in paths:
        try:
            relatives.append(path.relative_to(repo).as_posix())
        except ValueError as exc:
            raise PromotionError("tracked staging input escapes repository") from exc
    tracked = _run_git(
        repo, ["ls-files", "--error-unmatch", "--", *sorted(relatives)]
    )
    if tracked.returncode != 0:
        raise PromotionError("every staging package file must be tracked")
    clean = _run_git(repo, ["diff", "--quiet", "HEAD", "--", *sorted(relatives)])
    if clean.returncode == 1:
        raise PromotionError("staging package differs from committed HEAD")
    if clean.returncode != 0:
        raise PromotionError("cannot prove staging package matches HEAD")


def _normalized_policy(policy: PromotionPolicy) -> PromotionPolicy:
    repo = Path(os.path.abspath(os.fspath(Path(policy.repo_root).expanduser())))
    try:
        resolved_repo = repo.resolve(strict=True)
    except OSError as exc:
        raise PromotionError("promotion policy root does not exist") from exc
    if resolved_repo != repo or repo.is_symlink() or not repo.is_dir():
        raise PromotionError(
            "promotion policy root must be a canonical non-symlink directory"
        )

    def policy_path(
        value: Path,
        label: str,
        *,
        allow_missing_leaf: bool,
    ) -> Path:
        path = _lexical_path(os.fspath(value), repo, label)
        _reject_symlink_chain(
            path,
            repo,
            label,
            allow_missing_leaf=allow_missing_leaf,
        )
        return path

    return PromotionPolicy(
        repo_root=repo,
        staging_package=policy_path(
            Path(policy.staging_package),
            "policy staging package",
            allow_missing_leaf=False,
        ),
        release_package=policy_path(
            Path(policy.release_package),
            "policy release package",
            allow_missing_leaf=True,
        ),
        live_v2_assets=policy_path(
            Path(policy.live_v2_assets),
            "policy live v2 assets",
            allow_missing_leaf=True,
        ),
        legacy_v3_staging=policy_path(
            Path(policy.legacy_v3_staging),
            "policy legacy v3 staging",
            allow_missing_leaf=True,
        ),
        require_git_tracking=policy.require_git_tracking,
    )


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _validate_paths(
    staging: Path, destination: Path, policy: PromotionPolicy
) -> tuple[Path, Path, PromotionPolicy]:
    normalized = _normalized_policy(policy)
    source = _lexical_path(
        os.fspath(staging),
        normalized.repo_root,
        "staging package",
    )
    _reject_symlink_chain(
        source,
        normalized.repo_root,
        "staging package",
    )
    release = _lexical_path(
        os.fspath(destination),
        normalized.repo_root,
        "release destination",
    )
    _reject_symlink_chain(
        release,
        normalized.repo_root,
        "release destination",
        allow_missing_leaf=True,
    )
    if source != normalized.staging_package:
        raise PromotionError("source is not the exact dual staging package")
    if release != normalized.release_package:
        raise PromotionError("destination is not the exact release package")
    for forbidden, label in (
        (normalized.live_v2_assets, "live v2"),
        (normalized.legacy_v3_staging, "legacy single-fusion staging"),
        (normalized.staging_package, "dual staging"),
    ):
        if _is_within(release, forbidden):
            raise PromotionError(f"refusing {label} directory or descendant")
    if source == release or _is_within(source, release):
        raise PromotionError("release destination may not contain staging")
    if release.exists():
        if not release.is_dir() or next(release.iterdir(), None) is not None:
            raise PromotionError("release destination must be absent or empty")
    return source, release, normalized


def _verify_binding(
    binding: Mapping[str, Any],
    asset_records: Mapping[str, Mapping[str, Any]],
    *,
    status: str,
) -> None:
    if (
        binding.get("schema") != BINDING_SCHEMA
        or binding.get("schema_version") != 1
        or binding.get("status") != status
        or binding.get("candidate_id") != CANDIDATE_ID
    ):
        raise PromotionError("dual binding schema/status/candidate differs")
    if binding.get("execution_order") != [
        "stage_one_noise_gate",
        "rejector_known_unknown",
        "classifier_known_label",
    ]:
        raise PromotionError("dual binding execution order differs")
    roles = _mapping(binding.get("roles"), "dual binding roles")
    _exact_keys(roles, {"rejector", "classifier"}, "dual binding roles")
    for label, name, role, responsibility in (
        (
            "rejector",
            REJECTOR_WEIGHTS,
            REJECTOR_ROLE,
            REJECTOR_RESPONSIBILITY,
        ),
        (
            "classifier",
            CLASSIFIER_WEIGHTS,
            CLASSIFIER_ROLE,
            CLASSIFIER_RESPONSIBILITY,
        ),
    ):
        value = _mapping(roles[label], f"dual binding {label}")
        if (
            value.get("asset") != name
            or value.get("asset_sha256") != asset_records[name]["sha256"]
            or value.get("runtime_role") != role
            or value.get("responsibility") != responsibility
        ):
            raise PromotionError(f"dual binding {label} asset/role differs")
        _sha(
            value.get("runtime_bundle_manifest_sha256"),
            f"dual binding {label} bundle",
        )
        _sha(
            value.get("fusion_directory_sha256"),
            f"dual binding {label} fusion",
        )
    for field in (
        "asset_sha256",
        "runtime_bundle_manifest_sha256",
        "fusion_directory_sha256",
    ):
        if roles["rejector"][field] == roles["classifier"][field]:
            raise PromotionError(f"dual binding aliases role {field}")
    openset = _mapping(binding.get("openset_policy"), "binding openset_policy")
    if (
        openset.get("asset") != OPENSET_POLICY
        or openset.get("asset_sha256") != asset_records[OPENSET_POLICY]["sha256"]
        or openset.get("rejector_asset_sha256")
        != asset_records[REJECTOR_WEIGHTS]["sha256"]
        or openset.get("fitted_rejector_runtime_bundle_manifest_sha256")
        != roles["rejector"]["runtime_bundle_manifest_sha256"]
    ):
        raise PromotionError("binding open-set policy is not rejector-bound")
    report_sha = _sha(
        openset.get("staged_validation_report_sha256"),
        "binding validation report",
    )
    staged = _mapping(
        openset.get("staged_artifacts_sha256"), "binding staged artifacts"
    )
    _exact_keys(staged, EXPECTED_STAGED_ASSETS, "binding staged artifacts")
    for name, digest in staged.items():
        _sha(digest, f"binding staged artifact {name}")
    validation = _mapping(binding.get("validation"), "binding validation")
    if dict(validation) != {
        "report_sha256": report_sha,
        "role": "validate",
        "status": "development_openset_pass",
        "novelty_seeds": [20260950, 20260951],
    }:
        raise PromotionError("binding validation record differs")
    expected_flags = {
        "role_assets_bound_by_sha256": True,
        "distinct_role_assets": True,
        "role_asset_sha256_must_differ": True,
        "classifier_runs_only_after_rejector_acceptance": True,
        "public_known_label_from_classifier_only": True,
    }
    if binding.get("fail_closed") != expected_flags:
        raise PromotionError("binding fail_closed flags differ")


def _verify_staging_package(
    staging: Path, policy: PromotionPolicy
) -> VerifiedStaging:
    if staging.is_symlink() or not staging.is_dir():
        raise PromotionError("staging package must be a regular directory")
    entries = tuple(staging.iterdir())
    if {entry.name for entry in entries} != set(STAGING_FILE_NAMES):
        raise PromotionError("staging package is not the exact five-file package")
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise PromotionError("staging package contains a non-regular file")
    manifest_path = staging / PACKAGE_MANIFEST
    manifest_raw = _read_file(manifest_path, "staging package manifest")
    manifest = _parse_json(manifest_raw, "staging package manifest")
    expected_top = {
        "schema",
        "schema_version",
        "status",
        "candidate_id",
        "architecture",
        "assets",
        "roles",
        "openset_policy",
        "dual_binding",
        "external_evidence",
        "size_contract",
    }
    _exact_keys(manifest, expected_top, "staging package manifest")
    if (
        manifest.get("schema") != PACKAGE_SCHEMA
        or manifest.get("schema_version") != PACKAGE_SCHEMA_VERSION
        or manifest.get("status") != STAGING_STATUS
        or manifest.get("candidate_id") != CANDIDATE_ID
    ):
        raise PromotionError("staging package schema/status/candidate differs")
    if manifest.get("architecture") != {
        "execution_order": [
            "stage_one_noise_gate",
            "rejector_known_unknown",
            "classifier_known_label",
        ],
        "classifier_runs_only_after_rejector_acceptance": True,
        "public_known_label_from_classifier_only": True,
    }:
        raise PromotionError("staging package architecture differs")

    assets = _mapping(manifest.get("assets"), "staging package assets")
    _exact_keys(assets, set(ASSET_NAMES), "staging package assets")
    contracts = {
        REJECTOR_WEIGHTS: (FUSION_SCHEMA, 1, REJECTOR_ROLE),
        CLASSIFIER_WEIGHTS: (FUSION_SCHEMA, 1, CLASSIFIER_ROLE),
        OPENSET_POLICY: (OPENSET_SCHEMA, 2, None),
        DUAL_BINDING: (BINDING_SCHEMA, 1, None),
    }
    asset_bytes: dict[str, bytes] = {}
    payloads: dict[str, dict[str, Any]] = {}
    records: dict[str, dict[str, Any]] = {}
    for name, (schema, version, role) in contracts.items():
        record = _mapping(assets[name], f"staging asset record {name}")
        expected_keys = {
            "path",
            "bytes",
            "sha256",
            "schema",
            "schema_version",
            "status",
        }
        if role is not None:
            expected_keys.add("runtime_role")
        _exact_keys(record, expected_keys, f"staging asset record {name}")
        if record.get("path") != name:
            raise PromotionError(f"staging asset {name} path is not package-local")
        path = staging / name
        base, raw = _verify_file_record(
            record,
            path=path,
            label=f"staging {name}",
        )
        if len(raw) >= MAX_DEPLOYABLE_BYTES:
            raise PromotionError(f"staging asset {name} is not below 25 MiB")
        payload = _parse_json(raw, f"staging {name}")
        if (
            record.get("schema") != schema
            or record.get("schema_version") != version
            or record.get("status") != STAGING_STATUS
            or payload.get("schema") != schema
            or payload.get("schema_version") != version
            or payload.get("status") != STAGING_STATUS
        ):
            raise PromotionError(f"staging asset {name} contract differs")
        if role is not None and (
            record.get("runtime_role") != role
            or payload.get("runtime_role") != role
        ):
            raise PromotionError(f"staging asset {name} runtime role differs")
        records[name] = dict(record)
        asset_bytes[name] = raw
        payloads[name] = payload

    _verify_binding(payloads[DUAL_BINDING], records, status=STAGING_STATUS)
    roles = _mapping(manifest.get("roles"), "staging package roles")
    _exact_keys(roles, {"rejector", "classifier"}, "staging package roles")
    binding_roles = payloads[DUAL_BINDING]["roles"]
    for label, name, role, responsibility in (
        (
            "rejector",
            REJECTOR_WEIGHTS,
            REJECTOR_ROLE,
            REJECTOR_RESPONSIBILITY,
        ),
        (
            "classifier",
            CLASSIFIER_WEIGHTS,
            CLASSIFIER_ROLE,
            CLASSIFIER_RESPONSIBILITY,
        ),
    ):
        value = _mapping(roles[label], f"package role {label}")
        if (
            value.get("runtime_role") != role
            or value.get("responsibility") != responsibility
            or value.get("asset") != records[name]
            or value.get("source_bundle_manifest_sha256")
            != binding_roles[label]["runtime_bundle_manifest_sha256"]
            or value.get("fusion_directory_sha256")
            != binding_roles[label]["fusion_directory_sha256"]
        ):
            raise PromotionError(f"package role {label} link differs")
        role_payload = payloads[name]
        blockers = role_payload.get("release_blockers")
        if (
            role_payload.get("development_only") is not True
            or role_payload.get("release_evidence") is not False
            or not isinstance(blockers, list)
            or not blockers
            or not all(
                isinstance(blocker, str) and blocker.strip()
                for blocker in blockers
            )
        ):
            raise PromotionError(
                f"staging role {label} does not retain its release blockers"
            )
    openset_record = _mapping(
        manifest.get("openset_policy"), "package openset_policy"
    )
    for key, value in records[OPENSET_POLICY].items():
        if openset_record.get(key) != value:
            raise PromotionError("package open-set asset link differs")
    binding_openset = payloads[DUAL_BINDING]["openset_policy"]
    for key in (
        "fitted_rejector_runtime_bundle_manifest_sha256",
        "staged_validation_report_sha256",
        "staged_artifacts_sha256",
    ):
        if openset_record.get(key) != binding_openset.get(key):
            raise PromotionError(f"package open-set evidence {key} differs")
    if manifest.get("dual_binding") != records[DUAL_BINDING]:
        raise PromotionError("package dual binding record differs")
    policy_provenance = _mapping(
        payloads[OPENSET_POLICY].get("provenance"),
        "open-set policy provenance",
    )
    if (
        policy_provenance.get("candidate_id") != CANDIDATE_ID
        or policy_provenance.get("development_only") is not True
        or policy_provenance.get("release_evidence") is not False
        or policy_provenance.get("sealed_release_data_used") != 0
        or policy_provenance.get("consumed_test_rows_used") != 0
        or policy_provenance.get("release_seed_not_spent") != RELEASE_SEED
        or policy_provenance.get("staged_validation_all_pass") is not True
        or policy_provenance.get("staged_validation_role") != "validate"
        or policy_provenance.get("staged_validation_status")
        != "development_openset_pass"
        or policy_provenance.get("staged_validation_report_sha256")
        != binding_openset["staged_validation_report_sha256"]
        or policy_provenance.get("staged_artifact_sha256")
        != binding_openset["staged_artifacts_sha256"]
    ):
        raise PromotionError("open-set policy evidence differs from binding")
    runtime_roles = _mapping(
        policy_provenance.get("runtime_roles"),
        "open-set runtime role provenance",
    )
    for label, name, role in (
        ("rejector", REJECTOR_WEIGHTS, REJECTOR_ROLE),
        ("classifier", CLASSIFIER_WEIGHTS, CLASSIFIER_ROLE),
    ):
        value = _mapping(runtime_roles.get(label), f"policy role {label}")
        if (
            value.get("runtime_role") != role
            or value.get("browser_asset") != name
            or value.get("browser_asset_sha256") != records[name]["sha256"]
            or value.get("runtime_bundle_manifest_sha256")
            != binding_roles[label]["runtime_bundle_manifest_sha256"]
            or value.get("fusion_directory_sha256")
            != binding_roles[label]["fusion_directory_sha256"]
        ):
            raise PromotionError(f"open-set policy role {label} differs")

    external = _mapping(
        manifest.get("external_evidence"), "package external_evidence"
    )
    expected_external = {
        "rejector_export_manifest",
        "classifier_export_manifest",
        "openset_export_manifest",
        "rejector_probe",
        "classifier_probe",
        "parity",
    }
    _exact_keys(external, expected_external, "package external_evidence")
    for name, value in external.items():
        record = _mapping(value, f"external evidence {name}")
        path = _resolve_package_path(
            record.get("path"),
            manifest_path,
            policy.repo_root,
            f"external evidence {name}.path",
        )
        _base, external_raw = _verify_file_record(
            record,
            path=path,
            label=f"external evidence {name}",
        )
        if name in {"rejector_probe", "classifier_probe"}:
            _exact_keys(
                record,
                {"path", "bytes", "sha256"},
                f"external evidence {name}",
            )
            continue
        payload = _parse_json(
            external_raw,
            f"external evidence {name}",
        )
        if (
            payload.get("schema") != record.get("schema")
            or payload.get("schema_version") != record.get("schema_version")
        ):
            raise PromotionError(f"external evidence {name} schema differs")
        if "status" in record and payload.get("status") != record.get("status"):
            raise PromotionError(f"external evidence {name} status differs")
        if name == "parity" and record.get("packaged") is not False:
            raise PromotionError("parity must remain external")
    if manifest.get("size_contract") != {
        "maximum_file_bytes_exclusive": MAX_DEPLOYABLE_BYTES,
        "all_deployable_files_below_limit": True,
    }:
        raise PromotionError("package size contract differs")

    tracked = tuple(sorted(entries, key=str))
    if policy.require_git_tracking:
        _verify_git_tracking(policy.repo_root, tracked)
    return VerifiedStaging(
        manifest=manifest,
        manifest_raw=manifest_raw,
        manifest_sha256=_sha256_bytes(manifest_raw),
        asset_bytes=asset_bytes,
        asset_payloads=payloads,
        asset_records=records,
        tracked_paths=tracked,
    )


def _verify_historical_metadata(
    report: Mapping[str, Any], protocol: Mapping[str, Any]
) -> None:
    top = report.get("historical_gate_redeclaration")
    if top != protocol.get("historical_gate_redeclaration"):
        raise PromotionError("report and protocol historical metadata differ")
    block = _mapping(top, "historical_gate_redeclaration")
    if (
        block.get("release_seed") != HISTORICAL_RELAXED_SEED
        or block.get("active_for_current_protocol") is not False
        or block.get("current_protocol_gate_source")
        != "imported_v2_gate_floors_unchanged"
    ):
        raise PromotionError("historical relaxed gates are not explicitly inactive")
    gates = _mapping(block.get("gates_redeclared"), "historical gates")
    expected = {
        "five_shot_worst_length_balanced": (0.85, 0.84),
        "open_known_false_unknown_worst_length": (0.10, 0.12),
    }
    _exact_keys(gates, set(expected), "historical gates")
    for name, (v2_level, v3_level) in expected.items():
        value = _mapping(gates[name], f"historical gate {name}")
        if (
            value.get("v2_level") != v2_level
            or value.get("v3_level") != v3_level
            or not isinstance(value.get("owner_decision"), str)
            or not value["owner_decision"].strip()
        ):
            raise PromotionError(f"historical gate {name} differs")


def _finite_metric(
    value: Any,
    label: str,
    *,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise PromotionError(f"{label} must be a finite number")
    result = float(value)
    if result < minimum or result > maximum:
        raise PromotionError(
            f"{label} must be in [{minimum}, {maximum}]"
        )
    return result


def _length_metric_map(report: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = _mapping(report.get(name), name)
    expected = {str(length) for length in REQUIRED_CAPTURE_LENGTHS}
    _exact_keys(value, expected, name)
    return value


def _sweep_metrics(
    value: Any,
    *,
    label: str,
    coordinate_name: str,
    coordinates: tuple[int | float, ...],
    paired_name: str,
    prediction_summary: str,
    embedding_summary: str,
) -> tuple[float, float, float, float]:
    sweep = _mapping(value, label)
    rows = sweep.get("rows")
    if not isinstance(rows, list) or len(rows) != len(coordinates):
        raise PromotionError(f"{label}.rows has the wrong shape")
    balanced: list[float] = []
    pairwise: list[float] = []
    prediction: list[float] = []
    embedding: list[float] = []
    for index, (row_value, expected_coordinate) in enumerate(
        zip(rows, coordinates)
    ):
        row = _mapping(row_value, f"{label}.rows[{index}]")
        coordinate = row.get(coordinate_name)
        if coordinate != expected_coordinate or isinstance(coordinate, bool):
            raise PromotionError(
                f"{label}.rows[{index}].{coordinate_name} differs"
            )
        balanced.append(
            _finite_metric(
                row.get("balanced_accuracy"),
                f"{label}.rows[{index}].balanced_accuracy",
            )
        )
        pairwise.append(
            _finite_metric(
                row.get("mean_pairwise_cosine"),
                f"{label}.rows[{index}].mean_pairwise_cosine",
                minimum=-1.0,
            )
        )
        paired = _mapping(
            row.get(paired_name),
            f"{label}.rows[{index}].{paired_name}",
        )
        prediction.append(
            _finite_metric(
                paired.get("prediction_agreement"),
                f"{label}.rows[{index}].prediction_agreement",
            )
        )
        embedding.append(
            _finite_metric(
                paired.get("embedding_cosine_mean"),
                f"{label}.rows[{index}].embedding_cosine_mean",
                minimum=-1.0,
            )
        )
    derived = (
        min(balanced),
        max(pairwise),
        min(prediction),
        min(embedding),
    )
    summary_fields = (
        "worst_balanced_accuracy",
        "worst_mean_pairwise_cosine",
        prediction_summary,
        embedding_summary,
    )
    for field, expected in zip(summary_fields, derived):
        actual = _finite_metric(
            sweep.get(field),
            f"{label}.{field}",
            minimum=-1.0 if "cosine" in field else 0.0,
        )
        if actual != expected:
            raise PromotionError(f"{label}.{field} is not row-derived")
    return derived


def _recompute_release_gate_values(
    report: Mapping[str, Any],
) -> dict[str, float]:
    closed = _length_metric_map(report, "closed_per_length")
    five = _length_metric_map(report, "five_shot_predeclared_per_length")
    opened = _length_metric_map(report, "open_staged_per_length")
    decisions = _length_metric_map(
        report, "staged_known_decisions_per_length"
    )
    duplicate_sections = {
        "family_per_length": "family",
        "high_snr_per_length": "high_snr",
        "low_snr_per_length": "low_snr",
        "clean_subset_per_length": "clean_subset",
        "impaired_subset_per_length": "impaired_subset",
    }
    for section, nested_name in duplicate_sections.items():
        duplicated = _length_metric_map(report, section)
        expected = {
            length: _mapping(
                _mapping(closed[length], f"closed_per_length[{length}]").get(
                    nested_name
                ),
                f"closed_per_length[{length}].{nested_name}",
            )
            for length in duplicated
        }
        if dict(duplicated) != expected:
            raise PromotionError(
                f"{section} differs from closed_per_length.{nested_name}"
            )

    closed_fine: list[float] = []
    closed_family: list[float] = []
    closed_high: list[float] = []
    closed_clean: list[float] = []
    five_balanced: list[float] = []
    open_overall: list[float] = []
    open_noise: list[float] = []
    open_chirp: list[float] = []
    open_fur: list[float] = []
    open_noise_recall: list[float] = []
    open_chirp_recall: list[float] = []
    for length in map(str, REQUIRED_CAPTURE_LENGTHS):
        closed_row = _mapping(
            closed[length], f"closed_per_length[{length}]"
        )
        closed_fine.append(
            _finite_metric(
                closed_row.get("accuracy"),
                f"closed_per_length[{length}].accuracy",
            )
        )
        closed_family.append(
            _finite_metric(
                _mapping(
                    closed_row.get("family"),
                    f"closed_per_length[{length}].family",
                ).get("accuracy"),
                f"closed_per_length[{length}].family.accuracy",
            )
        )
        closed_high.append(
            _finite_metric(
                _mapping(
                    closed_row.get("high_snr"),
                    f"closed_per_length[{length}].high_snr",
                ).get("accuracy"),
                f"closed_per_length[{length}].high_snr.accuracy",
            )
        )
        closed_clean.append(
            _finite_metric(
                closed_row.get("clean_accuracy"),
                f"closed_per_length[{length}].clean_accuracy",
            )
        )
        five_balanced.append(
            _finite_metric(
                _mapping(
                    five[length],
                    f"five_shot_predeclared_per_length[{length}]",
                ).get("balanced_accuracy"),
                f"five_shot_predeclared_per_length[{length}]"
                ".balanced_accuracy",
            )
        )
        open_row = _mapping(
            opened[length], f"open_staged_per_length[{length}]"
        )
        open_overall.append(
            _finite_metric(
                open_row.get("auroc_overall"),
                f"open_staged_per_length[{length}].auroc_overall",
            )
        )
        open_noise.append(
            _finite_metric(
                open_row.get("auroc_noise"),
                f"open_staged_per_length[{length}].auroc_noise",
            )
        )
        open_chirp.append(
            _finite_metric(
                open_row.get("auroc_chirp"),
                f"open_staged_per_length[{length}].auroc_chirp",
            )
        )
        fur = _finite_metric(
            open_row.get("known_false_unknown_rate"),
            f"open_staged_per_length[{length}].known_false_unknown_rate",
        )
        open_fur.append(fur)
        open_noise_recall.append(
            _finite_metric(
                open_row.get("flagged_unknown_noise"),
                f"open_staged_per_length[{length}].flagged_unknown_noise",
            )
        )
        open_chirp_recall.append(
            _finite_metric(
                open_row.get("flagged_unknown_chirp"),
                f"open_staged_per_length[{length}].flagged_unknown_chirp",
            )
        )

        decision = _mapping(
            decisions[length],
            f"staged_known_decisions_per_length[{length}]",
        )
        rows = decision.get("rows")
        unknown = decision.get("unknown")
        gated = decision.get("gated_at_stage_one")
        accepted = decision.get("accepted_rows")
        known_rows = open_row.get("known_rows")
        if (
            type(rows) is not int
            or rows <= 0
            or type(unknown) is not int
            or not 0 <= unknown <= rows
            or type(gated) is not int
            or not 0 <= gated <= unknown
            or type(accepted) is not int
            or accepted != rows - unknown
            or known_rows != rows
            or isinstance(known_rows, bool)
            or fur != unknown / rows
        ):
            raise PromotionError(
                f"staged known-decision counts differ at length {length}"
            )
        by_stage = _mapping(
            open_row.get("known_false_unknown_by_stage"),
            f"open_staged_per_length[{length}]"
            ".known_false_unknown_by_stage",
        )
        if (
            by_stage.get("rows") != rows
            or _finite_metric(
                by_stage.get("staged_false_unknown_rate"),
                f"open_staged_per_length[{length}].staged rate",
            )
            != fur
            or _finite_metric(
                by_stage.get("stage_one_gate_rate"),
                f"open_staged_per_length[{length}].stage-one rate",
            )
            != gated / rows
            or _finite_metric(
                by_stage.get("stage_two_false_unknown_rate_marginal"),
                f"open_staged_per_length[{length}].stage-two rate",
            )
            != (unknown - gated) / rows
            or by_stage.get("rejected_by_stage_two_only")
            != unknown - gated
            or _finite_metric(
                open_row.get("known_gated_fraction"),
                f"open_staged_per_length[{length}].known_gated_fraction",
            )
            != gated / rows
        ):
            raise PromotionError(
                f"staged known-decision attribution differs at length {length}"
            )

    length_metrics = _sweep_metrics(
        report.get("matched_length_sweep"),
        label="matched_length_sweep",
        coordinate_name="capture_length",
        coordinates=REQUIRED_CAPTURE_LENGTHS,
        paired_name="paired_to_matched",
        prediction_summary="worst_prediction_agreement_to_matched",
        embedding_summary="worst_embedding_cosine_to_matched",
    )
    length_sweep = _mapping(
        report.get("matched_length_sweep"), "matched_length_sweep"
    )
    if length_sweep.get("matched_capture_length") != 16384:
        raise PromotionError("matched_length_sweep matched length differs")
    scale_metrics = _sweep_metrics(
        report.get("physical_scale_sweep"),
        label="physical_scale_sweep",
        coordinate_name="factor",
        coordinates=PHYSICAL_SCALE_FACTORS,
        paired_name="paired_to_factor1",
        prediction_summary="worst_prediction_agreement_to_factor1",
        embedding_summary="worst_embedding_cosine_to_factor1",
    )
    scale_sweep = _mapping(
        report.get("physical_scale_sweep"), "physical_scale_sweep"
    )
    if scale_sweep.get("factors") != list(PHYSICAL_SCALE_FACTORS):
        raise PromotionError("physical_scale_sweep factors differ")

    return {
        "closed_fine_worst_length": min(closed_fine),
        "closed_family_worst_length": min(closed_family),
        "closed_high_snr_worst_length": min(closed_high),
        "closed_clean_worst_length": min(closed_clean),
        "five_shot_worst_length_balanced": min(five_balanced),
        "open_auroc_overall_worst_length": min(open_overall),
        "open_auroc_noise_worst_length": min(open_noise),
        "open_auroc_chirp_worst_length": min(open_chirp),
        "open_known_false_unknown_worst_length": max(open_fur),
        "open_unknown_recall_noise_worst_length": min(open_noise_recall),
        "open_unknown_recall_chirp_worst_length": min(open_chirp_recall),
        "length_worst_balanced_accuracy": length_metrics[0],
        "length_max_mean_pairwise_cosine": length_metrics[1],
        "length_worst_prediction_agreement_to_matched": length_metrics[2],
        "length_worst_embedding_cosine_to_matched": length_metrics[3],
        "physical_scale_worst_balanced_accuracy": scale_metrics[0],
        "physical_scale_max_mean_pairwise_cosine": scale_metrics[1],
        "physical_scale_worst_prediction_agreement_to_factor1": (
            scale_metrics[2]
        ),
        "physical_scale_worst_embedding_cosine_to_factor1": scale_metrics[3],
    }


def _verify_release_gates(
    value: Any,
    candidate_sha: str,
    recomputed: Mapping[str, float],
) -> None:
    gates = _mapping(value, "release gates")
    _exact_keys(gates, EXPECTED_GATE_NAMES, "release gates")
    for name in BOOLEAN_GATE_NAMES:
        if gates[name] != {"value": True, "expected": True, "passes": True}:
            raise PromotionError(f"boolean gate {name} did not strictly pass")
    if gates[CANDIDATE_GATE_NAME] != {
        "value": candidate_sha,
        "expected": candidate_sha,
        "passes": True,
    }:
        raise PromotionError("candidate_sha_bound gate differs")
    for name, (threshold, comparison) in NUMERIC_GATE_CONTRACT.items():
        gate = _mapping(gates[name], f"release gate {name}")
        _exact_keys(
            gate,
            {"value", "threshold", "comparison", "passes"},
            f"release gate {name}",
        )
        actual = gate.get("value")
        if (
            not _is_finite_number(actual)
            or name not in recomputed
            or float(actual) != float(recomputed[name])
            or gate.get("threshold") != threshold
            or gate.get("comparison") != comparison
            or gate.get("passes") is not True
        ):
            raise PromotionError(f"release gate {name} contract differs")
        passes = (
            float(actual) >= threshold
            if comparison == "min"
            else float(actual) <= threshold
        )
        if not passes:
            raise PromotionError(f"release gate {name} claims a false pass")


def _verify_bound_file_map(
    directory: Path,
    file_sha256: Mapping[str, Any],
    label: str,
) -> dict[str, str]:
    if not file_sha256:
        raise PromotionError(f"{label} file hash map is empty")
    verified: dict[str, str] = {}
    for name, digest_value in file_sha256.items():
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or Path(name).is_absolute()
        ):
            raise PromotionError(f"{label} contains a non-local file name")
        digest = _sha(digest_value, f"{label} {name}")
        path = directory / name
        if _sha256_bytes(_read_file(path, f"{label} {name}")) != digest:
            raise PromotionError(f"{label} file {name} differs")
        verified[name] = digest
    return verified


def _fusion_directory_sha256(file_sha256: Mapping[str, str]) -> str:
    return _sha256_bytes(
        "\n".join(
            f"{name}:{file_sha256[name]}" for name in sorted(file_sha256)
        ).encode("utf-8")
    )


def _prefilter_bundle_sha256(directory: Path, length: int) -> str:
    if not directory.is_dir():
        raise PromotionError(f"prefilter N{length} is not a directory")
    entries = tuple(directory.iterdir())
    if (
        {entry.name for entry in entries} != set(PREFILTER_BUNDLE_FILES)
        or any(entry.is_symlink() or not entry.is_file() for entry in entries)
    ):
        raise PromotionError(f"prefilter N{length} files differ")
    digest = hashlib.sha256()
    for name in PREFILTER_BUNDLE_FILES:
        raw = _read_file(directory / name, f"prefilter N{length}/{name}")
        digest.update(f"{name}:{len(raw)}:".encode("utf-8"))
        digest.update(raw)
        if name == "meta.json":
            meta = _parse_json(raw, f"prefilter N{length}/meta.json")
            if meta.get("capture_length") != length:
                raise PromotionError(
                    f"prefilter N{length} capture_length differs"
                )
    return digest.hexdigest()


def _verify_prefilter_set(
    directory: Path,
    bundle_hashes: Mapping[str, Any],
    set_sha256: str,
) -> dict[str, str]:
    expected_lengths = (4096, 8192, 16384)
    _exact_keys(
        bundle_hashes,
        {str(length) for length in expected_lengths},
        "prefilter bundle hashes",
    )
    if not directory.is_dir():
        raise PromotionError("prefilter directory does not exist")
    entries = tuple(directory.iterdir())
    expected_names = {f"N{length}" for length in expected_lengths}
    if (
        {entry.name for entry in entries} != expected_names
        or any(entry.is_symlink() or not entry.is_dir() for entry in entries)
    ):
        raise PromotionError("prefilter set directories differ")
    actual: dict[str, str] = {}
    set_digest = hashlib.sha256()
    for name in sorted(expected_names):
        length = int(name[1:])
        digest = _prefilter_bundle_sha256(directory / name, length)
        expected = _sha(
            bundle_hashes[str(length)],
            f"prefilter bundle N{length}",
        )
        if digest != expected:
            raise PromotionError(f"prefilter bundle N{length} SHA differs")
        actual[str(length)] = digest
        set_digest.update(f"{name}:".encode("utf-8"))
        set_digest.update(digest.encode("utf-8"))
    if set_digest.hexdigest() != set_sha256:
        raise PromotionError("prefilter set SHA differs from current bytes")
    return actual


def _record_core(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in ("path", "sha256", "schema", "status")
        if key in record
    }


def _expected_protocol() -> dict[str, Any]:
    raw = _read_file(
        EXPECTED_PROTOCOL_FIXTURE,
        "seed-20260735 protocol fixture",
    )
    if _sha256_bytes(raw) != EXPECTED_PROTOCOL_FIXTURE_SHA256:
        raise PromotionError("seed-20260735 protocol fixture bytes drifted")
    wrapper = _parse_json(raw, "seed-20260735 protocol fixture")
    if wrapper.get("release_seed") != RELEASE_SEED:
        raise PromotionError("seed-20260735 protocol fixture seed differs")
    return dict(
        _mapping(
            wrapper.get("evaluation_protocol"),
            "seed-20260735 evaluation_protocol",
        )
    )


def _verify_evaluator_identity(provenance: Mapping[str, Any]) -> None:
    evaluator_raw = _read_file(EXPECTED_EVALUATOR, "final v3 evaluator")
    v2_raw = _read_file(EXPECTED_V2_EVALUATOR, "frozen v2 evaluator")
    if (
        _sha256_bytes(evaluator_raw) != EXPECTED_EVALUATOR_SHA256
        or provenance.get("evaluator_sha256")
        != EXPECTED_EVALUATOR_SHA256
        or provenance.get("evaluator_path") != str(EXPECTED_EVALUATOR)
    ):
        raise PromotionError("release report final evaluator identity differs")
    if (
        _sha256_bytes(v2_raw) != EXPECTED_V2_EVALUATOR_SHA256
        or provenance.get("v2_evaluator_sha256")
        != EXPECTED_V2_EVALUATOR_SHA256
    ):
        raise PromotionError("release report v2 evaluator identity differs")


def _verify_source_transition(value: Any) -> dict[str, Any]:
    source_report = _mapping(value, "candidate source_sha256")
    _exact_keys(
        source_report,
        {
            "enforced",
            "recorded_only",
            "post_validation_ledger_transitions",
            "evaluator_chain",
        },
        "candidate source_sha256",
    )
    current_source = _sha256_bytes(
        _read_file(EXPECTED_STAGED_SOURCE, "current staged source")
    )
    if current_source != EXPECTED_LEDGER_TRANSITION["current_sha256"]:
        raise PromotionError("current staged source differs from ledger pin")
    enforced = _mapping(
        source_report.get("enforced"), "candidate enforced source hashes"
    )
    if (
        enforced.get("fit_v3_openset_staged.py")
        != EXPECTED_LEDGER_TRANSITION["current_sha256"]
    ):
        raise PromotionError("candidate staged source enforcement differs")
    transitions = _mapping(
        source_report.get("post_validation_ledger_transitions"),
        "post-validation ledger transitions",
    )
    if dict(transitions) != {
        "fit_v3_openset_staged.py": EXPECTED_LEDGER_TRANSITION
    }:
        raise PromotionError(
            "post-validation ledger transition record differs"
        )
    return copy.deepcopy(EXPECTED_LEDGER_TRANSITION)


def _verify_evaluation_report(
    report_path: Path,
    staging: VerifiedStaging,
    policy: PromotionPolicy,
) -> VerifiedEvaluation:
    raw = _read_file(report_path, "sealed evaluation report")
    report = _parse_json(raw, "sealed evaluation report")
    _exact_keys(report, REPORT_TOP_KEYS, "sealed evaluation report")
    if (
        report.get("schema") != EVALUATOR_SCHEMA
        or report.get("status") != "complete"
        or report.get("release_evidence") is not True
        or report.get("evaluation_version") != EVALUATION_VERSION
        or report.get("all_release_gates_pass") is not True
    ):
        raise PromotionError("release report is not a complete evaluator-v3 pass")
    for field in (
        "development_data_loaded",
        "retraining_performed",
        "recalibration_performed",
    ):
        if report.get(field) is not False:
            raise PromotionError(f"release report {field} must be false")

    provenance = _mapping(report.get("provenance"), "release provenance")
    if provenance.get("release_seed") != RELEASE_SEED:
        raise PromotionError(f"release report must use seed {RELEASE_SEED}")
    _verify_evaluator_identity(provenance)
    protocol = _mapping(
        provenance.get("evaluation_protocol"), "release evaluation protocol"
    )
    if dict(protocol) != _expected_protocol():
        raise PromotionError(
            "embedded protocol is not the exact seed-20260735 fixture"
        )
    open_protocol = _mapping(protocol.get("open_set"), "protocol open_set")
    if (
        open_protocol.get("intentional_dual_fusion") is not True
        or open_protocol.get("candidate_architecture")
        != "dual_fusion_classifier8k_rejector4k"
    ):
        raise PromotionError("protocol does not evaluate the dual architecture")
    _verify_historical_metadata(report, protocol)
    architecture = _mapping(
        report.get("architecture_contract"), "release architecture contract"
    )
    for field, expected in (
        ("intentional_dual_fusion", True),
        ("candidate_architecture", "dual_fusion_classifier8k_rejector4k"),
        ("known_label_source", "classifier_fusion_8k_regularized"),
        ("known_unknown_source", "rejector_fusion_4k_frozen_policy"),
        ("closed_gate_source", "classifier_fusion_8k_regularized"),
        ("open_gate_source", "rejector_fusion_4k_frozen_policy"),
        ("gates_before_classification", True),
    ):
        if architecture.get(field) != expected:
            raise PromotionError(f"release architecture {field} differs")

    candidate = _mapping(report.get("candidate"), "release candidate")
    candidate_sha = _sha(candidate.get("sha256"), "candidate.sha256")
    if (
        candidate.get("schema") != CANDIDATE_SCHEMA
        or candidate.get("candidate_id") != CANDIDATE_ID
        or provenance.get("candidate_sha256") != candidate_sha
    ):
        raise PromotionError("release report candidate identity differs")
    candidate_path = _resolve_path(
        candidate.get("path"), policy.repo_root, "candidate.path"
    )
    candidate_raw = _read_file(candidate_path, "candidate manifest")
    if _sha256_bytes(candidate_raw) != candidate_sha:
        raise PromotionError("candidate manifest SHA differs from report")
    candidate_manifest = _parse_json(candidate_raw, "candidate manifest")
    expected_candidate_keys = {
        "schema",
        "status",
        "candidate_id",
        "validation_evidence",
        "classifier",
        "rejector",
        "staged_validation",
        "stage_one_prefilter",
        "browser_assets",
        "staging_package_manifest",
        "dual_binding",
    }
    _exact_keys(
        candidate_manifest,
        expected_candidate_keys,
        "candidate manifest",
    )
    if (
        candidate_manifest.get("schema") != CANDIDATE_SCHEMA
        or candidate_manifest.get("status") != "release_candidate_frozen"
        or candidate_manifest.get("candidate_id") != CANDIDATE_ID
    ):
        raise PromotionError("candidate manifest is not the frozen dual candidate")
    classes = candidate.get("classes")
    if (
        not isinstance(classes, list)
        or not classes
        or not all(isinstance(name, str) and name for name in classes)
    ):
        raise PromotionError("release report candidate classes are invalid")
    frozen = candidate.get("frozen_assets_used")
    if (
        not isinstance(frozen, list)
        or not frozen
        or not all(isinstance(item, str) and item for item in frozen)
    ):
        raise PromotionError("release report does not enumerate frozen assets")

    components = _mapping(candidate.get("components"), "candidate components")
    _exact_keys(components, CANDIDATE_COMPONENT_KEYS, "candidate components")
    source_transition = _verify_source_transition(
        components.get("source_sha256")
    )
    contract = _mapping(
        components.get("candidate_contract"), "candidate_contract"
    )
    if (
        _resolve_path(contract.get("path"), policy.repo_root, "contract.path")
        != candidate_path
        or contract.get("sha256") != candidate_sha
        or contract.get("schema") != CANDIDATE_SCHEMA
        or contract.get("candidate_id") != CANDIDATE_ID
        or contract.get("status") != "release_candidate_frozen"
    ):
        raise PromotionError("candidate_contract does not identify candidate bytes")

    validation_record = _mapping(
        components.get("validation_evidence"), "validation_evidence"
    )
    validation_path, validation_sha, validation = _verify_json_record(
        validation_record,
        repo=policy.repo_root,
        label="validation evidence",
        schema=VALIDATION_EVIDENCE_SCHEMA,
        status="development_openset_pass",
    )
    candidate_validation = _mapping(
        candidate_manifest.get("validation_evidence"),
        "manifest validation_evidence",
    )
    if (
        _resolve_path(
            candidate_validation.get("path"),
            policy.repo_root,
            "manifest validation_evidence.path",
        )
        != validation_path
        or candidate_validation.get("sha256") != validation_sha
    ):
        raise PromotionError("candidate validation_evidence link differs")
    frozen_record = _mapping(
        components.get("frozen_prevalidation_contract"),
        "frozen_prevalidation_contract",
    )
    frozen_path, frozen_sha, frozen_payload = _verify_json_record(
        frozen_record,
        repo=policy.repo_root,
        label="frozen prevalidation contract",
        schema=PREVALIDATION_CONTRACT_SCHEMA,
        status="frozen_before_validation",
    )
    if frozen_payload.get("candidate_id") != CANDIDATE_ID:
        raise PromotionError("prevalidation contract candidate_id differs")
    validation_contract = _mapping(
        validation.get("candidate_contract"),
        "validation evidence candidate_contract",
    )
    if (
        _resolve_path(
            validation_contract.get("path"),
            policy.repo_root,
            "validation candidate_contract.path",
        )
        != frozen_path
        or validation_contract.get("sha256") != frozen_sha
    ):
        raise PromotionError("validation evidence does not bind frozen contract")

    binding = staging.asset_payloads[DUAL_BINDING]
    binding_roles = binding["roles"]
    bundle_summaries: dict[str, Any] = {}
    fusion_summaries: dict[str, Any] = {}
    for label, runtime_role, fusion_role in (
        ("classifier", CLASSIFIER_ROLE, "known_class_label"),
        ("rejector", REJECTOR_ROLE, "known_unknown_decision"),
    ):
        bundle = _mapping(
            components.get(f"{label}_runtime_bundle"),
            f"{label}_runtime_bundle",
        )
        if (
            bundle.get("schema") != BUNDLE_SCHEMA
            or bundle.get("schema_version") != BUNDLE_SCHEMA_VERSION
            or bundle.get("kind") != BUNDLE_KIND
            or bundle.get("manifest_sha256")
            != binding_roles[label]["runtime_bundle_manifest_sha256"]
        ):
            raise PromotionError(f"{label} runtime bundle contract differs")
        bundle_dir = _resolve_path(
            bundle.get("directory"), policy.repo_root, f"{label} bundle directory"
        )
        bundle_manifest_path = _resolve_path(
            bundle.get("manifest_path"),
            policy.repo_root,
            f"{label} bundle manifest",
        )
        if bundle_manifest_path.parent != bundle_dir:
            raise PromotionError(f"{label} bundle manifest escapes bundle directory")
        bundle_manifest_raw = _read_file(
            bundle_manifest_path, f"{label} bundle manifest"
        )
        if _sha256_bytes(bundle_manifest_raw) != bundle["manifest_sha256"]:
            raise PromotionError(f"{label} bundle manifest SHA differs")
        bundle_manifest = _parse_json(
            bundle_manifest_raw, f"{label} bundle manifest"
        )
        if (
            bundle_manifest.get("schema") != BUNDLE_SCHEMA
            or bundle_manifest.get("schema_version") != 1
            or bundle_manifest.get("kind") != BUNDLE_KIND
            or bundle_manifest.get("runtime_role") != runtime_role
            or bundle_manifest.get("assets") != bundle.get("assets")
        ):
            raise PromotionError(f"{label} bundle bytes differ from report")
        assets = _mapping(bundle.get("assets"), f"{label} bundle assets")
        _exact_keys(assets, EXPECTED_BUNDLE_ASSETS, f"{label} bundle assets")
        for name, value in assets.items():
            _verify_file_record(
                _mapping(value, f"{label} bundle {name}"),
                path=bundle_dir / name,
                label=f"{label} bundle {name}",
            )
        candidate_role = _mapping(
            candidate_manifest.get(label), f"candidate manifest {label}"
        )
        candidate_bundle = _mapping(
            candidate_role.get("runtime_bundle"),
            f"candidate {label} runtime bundle",
        )
        if (
            candidate_role.get("role") != fusion_role
            or _resolve_path(
                candidate_bundle.get("directory"),
                policy.repo_root,
                f"candidate {label} bundle directory",
            )
            != bundle_dir
            or _resolve_path(
                candidate_bundle.get("manifest_path"),
                policy.repo_root,
                f"candidate {label} bundle manifest",
            )
            != bundle_manifest_path
            or candidate_bundle.get("manifest_sha256")
            != bundle["manifest_sha256"]
        ):
            raise PromotionError(f"candidate manifest {label} bundle differs")
        bundle_summaries[label] = {
            "manifest_sha256": bundle["manifest_sha256"],
            "assets_sha256": {
                name: value["sha256"] for name, value in assets.items()
            },
        }

        fusion = _mapping(
            components.get(f"{label}_fusion"), f"{label}_fusion"
        )
        if (
            fusion.get("role") != fusion_role
            or fusion.get("directory_sha256")
            != binding_roles[label]["fusion_directory_sha256"]
            or not _is_exact_int(fusion.get("seed"))
        ):
            raise PromotionError(f"{label} fusion role/hash differs")
        fusion_dir = _resolve_path(
            fusion.get("directory"), policy.repo_root, f"{label} fusion directory"
        )
        fusion_file_map = _mapping(
            fusion.get("file_sha256"), f"{label} fusion files"
        )
        _exact_keys(
            fusion_file_map,
            EXPECTED_FUSION_FILES,
            f"{label} fusion files",
        )
        files = _verify_bound_file_map(
            fusion_dir,
            fusion_file_map,
            f"{label} fusion",
        )
        if _fusion_directory_sha256(files) != fusion["directory_sha256"]:
            raise PromotionError(
                f"{label} fusion directory SHA differs from current bytes"
            )
        candidate_fusion = _mapping(
            candidate_role.get("fusion"), f"candidate {label} fusion"
        )
        if (
            _resolve_path(
                candidate_fusion.get("directory"),
                policy.repo_root,
                f"candidate {label} fusion directory",
            )
            != fusion_dir
            or candidate_fusion.get("directory_sha256")
            != fusion["directory_sha256"]
            or candidate_fusion.get("file_sha256") != fusion["file_sha256"]
        ):
            raise PromotionError(f"candidate manifest {label} fusion differs")
        fusion_summaries[label] = {
            "directory_sha256": fusion["directory_sha256"],
            "files_sha256": files,
            "seed": fusion["seed"],
        }

    staged = _mapping(
        components.get("staged_validation"), "staged_validation"
    )
    if (
        staged.get("status") != "development_openset_pass"
        or staged.get("novelty_seeds") != [20260950, 20260951]
        or staged.get("report_sha256")
        != binding["openset_policy"]["staged_validation_report_sha256"]
        or staged.get("artifact_sha256")
        != binding["openset_policy"]["staged_artifacts_sha256"]
    ):
        raise PromotionError("staged validation evidence differs from binding")
    staged_dir = _resolve_path(
        staged.get("directory"), policy.repo_root, "staged directory"
    )
    staged_report_path = _resolve_path(
        staged.get("report_path"), policy.repo_root, "staged report"
    )
    if (
        staged_report_path.parent != staged_dir
        or _sha256_bytes(
            _read_file(staged_report_path, "staged validation report")
        )
        != staged["report_sha256"]
    ):
        raise PromotionError("staged validation report bytes differ")
    staged_hashes = _mapping(
        staged.get("artifact_sha256"), "staged artifact hashes"
    )
    _exact_keys(staged_hashes, EXPECTED_STAGED_ASSETS, "staged artifact hashes")
    for name, digest in staged_hashes.items():
        if _sha256_bytes(
            _read_file(staged_dir / name, f"staged artifact {name}")
        ) != _sha(digest, f"staged artifact {name}"):
            raise PromotionError(f"staged artifact {name} differs")
    candidate_staged = _mapping(
        candidate_manifest.get("staged_validation"),
        "candidate staged_validation",
    )
    if (
        _resolve_path(
            candidate_staged.get("directory"),
            policy.repo_root,
            "candidate staged directory",
        )
        != staged_dir
        or _resolve_path(
            candidate_staged.get("report_path"),
            policy.repo_root,
            "candidate staged report",
        )
        != staged_report_path
        or candidate_staged.get("report_sha256") != staged["report_sha256"]
        or candidate_staged.get("artifact_sha256") != staged["artifact_sha256"]
    ):
        raise PromotionError("candidate manifest staged validation differs")

    prefilter = _mapping(
        components.get("stage_one_prefilter"), "stage_one_prefilter"
    )
    prefilter_sha = _sha(prefilter.get("set_sha256"), "prefilter set SHA")
    bundle_hashes = _mapping(
        prefilter.get("bundle_sha256"), "prefilter bundle hashes"
    )
    if (
        not bundle_hashes
        or any(
            not isinstance(length, str)
            or SHA256_RE.fullmatch(str(digest)) is None
            for length, digest in bundle_hashes.items()
        )
        or staging.asset_payloads[OPENSET_POLICY]
        .get("provenance", {})
        .get("prefilter_set_sha256")
        != prefilter_sha
    ):
        raise PromotionError("prefilter evidence differs from open-set policy")
    candidate_prefilter = _mapping(
        candidate_manifest.get("stage_one_prefilter"),
        "candidate stage_one_prefilter",
    )
    prefilter_dir = _resolve_path(
        prefilter.get("directory"),
        policy.repo_root,
        "prefilter directory",
    )
    verified_prefilter_hashes = _verify_prefilter_set(
        prefilter_dir,
        bundle_hashes,
        prefilter_sha,
    )
    if (
        _resolve_path(
            candidate_prefilter.get("directory"),
            policy.repo_root,
            "candidate prefilter directory",
        )
        != prefilter_dir
        or candidate_prefilter.get("set_sha256") != prefilter_sha
        or candidate_prefilter.get("bundle_sha256") != bundle_hashes
    ):
        raise PromotionError("candidate manifest prefilter differs")

    browser = _mapping(components.get("browser_assets"), "browser_assets")
    _exact_keys(
        browser, {"classifier", "rejector", "openset_policy"}, "browser_assets"
    )
    browser_contract = {
        "classifier": (CLASSIFIER_WEIGHTS, FUSION_SCHEMA, CLASSIFIER_ROLE),
        "rejector": (REJECTOR_WEIGHTS, FUSION_SCHEMA, REJECTOR_ROLE),
        "openset_policy": (OPENSET_POLICY, OPENSET_SCHEMA, None),
    }
    browser_hashes: dict[str, str] = {}
    candidate_browser = _mapping(
        candidate_manifest.get("browser_assets"), "candidate browser_assets"
    )
    for label, (name, schema, runtime_role) in browser_contract.items():
        record = _mapping(browser[label], f"browser asset {label}")
        _exact_keys(
            record,
            {"path", "sha256", "schema", "status"},
            f"browser asset {label}",
        )
        path = _resolve_path(
            record.get("path"),
            policy.repo_root,
            f"browser asset {label}.path",
        )
        digest = _sha(
            record.get("sha256"), f"browser asset {label}.sha256"
        )
        payload = staging.asset_payloads[name]
        if (
            path != policy.staging_package / name
            or record.get("schema") != schema
            or record.get("status") != STAGING_STATUS
        ):
            raise PromotionError(
                f"browser asset {label} is outside staging package"
            )
        if digest != staging.asset_records[name]["sha256"]:
            raise PromotionError(f"browser asset {label} SHA differs from package")
        if runtime_role is not None and payload.get("runtime_role") != runtime_role:
            raise PromotionError(f"browser asset {label} runtime role differs")
        if _record_core(record) != _record_core(
            _mapping(candidate_browser.get(label), f"candidate browser {label}")
        ):
            raise PromotionError(f"candidate manifest browser {label} differs")
        browser_hashes[label] = digest
    if browser_hashes["classifier"] == browser_hashes["rejector"]:
        raise PromotionError("browser role assets alias")

    package_record = _mapping(
        components.get("staging_package_manifest"),
        "staging_package_manifest",
    )
    _exact_keys(
        package_record,
        {"path", "sha256", "schema", "status"},
        "staging_package_manifest",
    )
    package_path = _resolve_path(
        package_record.get("path"),
        policy.repo_root,
        "staging package manifest.path",
    )
    package_sha = _sha(
        package_record.get("sha256"), "staging package manifest.sha256"
    )
    if (
        package_path != policy.staging_package / PACKAGE_MANIFEST
        or package_sha != staging.manifest_sha256
        or package_record.get("schema") != PACKAGE_SCHEMA
        or package_record.get("status") != STAGING_STATUS
        or _record_core(package_record)
        != _record_core(
            _mapping(
                candidate_manifest.get("staging_package_manifest"),
                "candidate staging_package_manifest",
            )
        )
    ):
        raise PromotionError("report/candidate staging package differs")

    binding_record = _mapping(components.get("dual_binding"), "dual_binding")
    _exact_keys(
        binding_record,
        {"path", "sha256", "schema"},
        "dual_binding",
    )
    binding_path = _resolve_path(
        binding_record.get("path"),
        policy.repo_root,
        "dual binding.path",
    )
    binding_sha = _sha(
        binding_record.get("sha256"), "dual binding.sha256"
    )
    if (
        binding_path != policy.staging_package / DUAL_BINDING
        or binding_sha != staging.asset_records[DUAL_BINDING]["sha256"]
        or binding_record.get("schema") != BINDING_SCHEMA
        or components.get("dual_binding_sha256") != binding_sha
        or _record_core(binding_record)
        != _record_core(
            _mapping(
                candidate_manifest.get("dual_binding"),
                "candidate dual_binding",
            )
        )
    ):
        raise PromotionError("report/candidate dual binding differs")

    recomputed_gates = _recompute_release_gate_values(report)
    _verify_release_gates(
        report.get("gates"),
        candidate_sha,
        recomputed_gates,
    )
    prefix_nesting = _mapping(
        provenance.get("prefix_nesting"), "prefix_nesting"
    )
    dependencies = _mapping(
        provenance.get("dependency_provenance"),
        "dependency_provenance",
    )
    start_probe = _mapping(
        provenance.get("unscored_start_probe"),
        "unscored_start_probe",
    )
    if (
        prefix_nesting.get("passes") is not True
        or dependencies.get("passes") is not True
        or start_probe.get("scored") is not False
        or start_probe.get("passes") is not True
    ):
        raise PromotionError("boolean gate evidence bodies did not pass")
    release_root = _resolve_path(
        provenance.get("release_root"), policy.repo_root, "release_root"
    )
    expected_release_root = (
        policy.repo_root
        / EXPECTED_RELEASE_ROOT.relative_to(REPO)
    )
    if (
        release_root != expected_release_root
        or not release_root.is_dir()
        or report_path != release_root / "RELEASE_EVALUATION.json"
    ):
        raise PromotionError("release_root must be a regular directory")
    intent_path = release_root / "RELEASE_INTENT.json"
    release_manifest_path = release_root / "RELEASE_MANIFEST.json"
    intent_raw = _read_file(intent_path, "RELEASE_INTENT")
    release_manifest_raw = _read_file(release_manifest_path, "RELEASE_MANIFEST")
    intent_sha = _sha256_bytes(intent_raw)
    release_manifest_sha = _sha256_bytes(release_manifest_raw)
    if (
        provenance.get("release_intent_sha256") != intent_sha
        or provenance.get("release_manifest_sha256") != release_manifest_sha
    ):
        raise PromotionError("release report suite manifest hashes differ")
    intent = _parse_json(intent_raw, "RELEASE_INTENT")
    release_manifest = _parse_json(release_manifest_raw, "RELEASE_MANIFEST")
    if (
        intent.get("release_seed") != RELEASE_SEED
        or release_manifest.get("release_seed") != RELEASE_SEED
        or intent.get("status") != "in_progress"
        or release_manifest.get("status") != "complete"
        or _resolve_path(
            intent.get("candidate_path"), policy.repo_root, "intent candidate"
        )
        != candidate_path
        or _resolve_path(
            release_manifest.get("candidate_path"),
            policy.repo_root,
            "release manifest candidate",
        )
        != candidate_path
        or intent.get("candidate_sha256") != candidate_sha
        or release_manifest.get("candidate_sha256") != candidate_sha
        or release_manifest.get("release_intent_sha256") != intent_sha
        or intent.get("evaluation_protocol") != protocol
        or release_manifest.get("evaluation_protocol") != protocol
    ):
        raise PromotionError("release intent/manifest candidate chain differs")

    artifacts = {
        "candidate_manifest_sha256": candidate_sha,
        "validation_evidence_sha256": validation_sha,
        "frozen_prevalidation_contract_sha256": frozen_sha,
        "runtime_bundles": bundle_summaries,
        "fusions": fusion_summaries,
        "staged_validation_report_sha256": staged["report_sha256"],
        "staged_artifacts_sha256": dict(staged_hashes),
        "prefilter_set_sha256": prefilter_sha,
        "prefilter_bundle_sha256": verified_prefilter_hashes,
        "browser_assets_sha256": browser_hashes,
        "staging_package_manifest_sha256": package_sha,
        "staging_dual_binding_sha256": binding_sha,
        "post_validation_ledger_transition": source_transition,
    }
    return VerifiedEvaluation(
        report_sha256=_sha256_bytes(raw),
        candidate_sha256=candidate_sha,
        candidate_artifacts=artifacts,
        release_intent_sha256=intent_sha,
        release_manifest_sha256=release_manifest_sha,
    )


def _release_files(
    staging: VerifiedStaging, evaluation: VerifiedEvaluation
) -> dict[str, bytes]:
    payloads: dict[str, dict[str, Any]] = {}
    output: dict[str, bytes] = {}
    for name in (REJECTOR_WEIGHTS, CLASSIFIER_WEIGHTS):
        payload = copy.deepcopy(staging.asset_payloads[name])
        payload["status"] = RELEASE_STATUS
        if "development_only" in payload:
            payload["development_only"] = False
        if "release_evidence" in payload:
            payload["release_evidence"] = True
        if "release_blockers" in payload:
            payload["release_blockers"] = []
        raw = _compact_json_bytes(payload)
        if len(raw) >= MAX_DEPLOYABLE_BYTES:
            raise PromotionError(f"release asset {name} is not below 25 MiB")
        payloads[name] = payload
        output[name] = raw
    policy_payload = copy.deepcopy(staging.asset_payloads[OPENSET_POLICY])
    policy_payload["status"] = RELEASE_STATUS
    policy_roles = _mapping(
        _mapping(
            policy_payload.get("provenance"),
            "release open-set provenance",
        ).get("runtime_roles"),
        "release open-set runtime roles",
    )
    for label, name in (
        ("rejector", REJECTOR_WEIGHTS),
        ("classifier", CLASSIFIER_WEIGHTS),
    ):
        role = _mapping(
            policy_roles.get(label),
            f"release open-set role {label}",
        )
        role["browser_asset_sha256"] = _sha256_bytes(output[name])
    policy_raw = _compact_json_bytes(policy_payload)
    if len(policy_raw) >= MAX_DEPLOYABLE_BYTES:
        raise PromotionError(
            f"release asset {OPENSET_POLICY} is not below 25 MiB"
        )
    payloads[OPENSET_POLICY] = policy_payload
    output[OPENSET_POLICY] = policy_raw
    release_records = {
        name: {
            **{
                key: value
                for key, value in staging.asset_records[name].items()
                if key not in {"bytes", "sha256", "status"}
            },
            "bytes": len(output[name]),
            "sha256": _sha256_bytes(output[name]),
            "status": RELEASE_STATUS,
        }
        for name in (REJECTOR_WEIGHTS, CLASSIFIER_WEIGHTS, OPENSET_POLICY)
    }

    binding = copy.deepcopy(staging.asset_payloads[DUAL_BINDING])
    binding["status"] = RELEASE_STATUS
    binding["roles"]["rejector"]["asset_sha256"] = release_records[
        REJECTOR_WEIGHTS
    ]["sha256"]
    binding["roles"]["classifier"]["asset_sha256"] = release_records[
        CLASSIFIER_WEIGHTS
    ]["sha256"]
    binding["openset_policy"]["asset_sha256"] = release_records[
        OPENSET_POLICY
    ]["sha256"]
    binding["openset_policy"]["rejector_asset_sha256"] = release_records[
        REJECTOR_WEIGHTS
    ]["sha256"]
    binding_raw = _compact_json_bytes(binding)
    if len(binding_raw) >= MAX_DEPLOYABLE_BYTES:
        raise PromotionError("release dual binding is not below 25 MiB")
    output[DUAL_BINDING] = binding_raw
    payloads[DUAL_BINDING] = binding
    release_records[DUAL_BINDING] = {
        **{
            key: value
            for key, value in staging.asset_records[DUAL_BINDING].items()
            if key not in {"bytes", "sha256", "status"}
        },
        "bytes": len(binding_raw),
        "sha256": _sha256_bytes(binding_raw),
        "status": RELEASE_STATUS,
    }
    _verify_binding(binding, release_records, status=RELEASE_STATUS)

    manifest = copy.deepcopy(staging.manifest)
    manifest["status"] = RELEASE_STATUS
    manifest["assets"] = release_records
    for label, name in (
        ("rejector", REJECTOR_WEIGHTS),
        ("classifier", CLASSIFIER_WEIGHTS),
    ):
        manifest["roles"][label]["asset"] = release_records[name]
    openset_extra = {
        key: value
        for key, value in manifest["openset_policy"].items()
        if key
        not in {
            "path",
            "bytes",
            "sha256",
            "schema",
            "schema_version",
            "status",
        }
    }
    manifest["openset_policy"] = {
        **release_records[OPENSET_POLICY],
        **openset_extra,
    }
    manifest["dual_binding"] = release_records[DUAL_BINDING]
    manifest["promotion"] = {
        "schema": "atomos.v3.time-domain-classifier.dual-release-promotion",
        "schema_version": 1,
        "release_seed": RELEASE_SEED,
        "source": {
            "package_status": STAGING_STATUS,
            "package_manifest_sha256": staging.manifest_sha256,
            "asset_records": copy.deepcopy(staging.asset_records),
        },
        "evaluation": {
            "status": "complete",
            "release_evidence": True,
            "all_release_gates_pass": True,
            "gate_contract": "strict_imported_v2_23_of_23",
            "five_shot_minimum": 0.85,
            "known_false_unknown_maximum": 0.10,
            "evaluation_version": EVALUATION_VERSION,
            "report_sha256": evaluation.report_sha256,
            "candidate_sha256": evaluation.candidate_sha256,
            "release_intent_sha256": evaluation.release_intent_sha256,
            "release_manifest_sha256": evaluation.release_manifest_sha256,
        },
        "candidate_artifacts": evaluation.candidate_artifacts,
        "status_rewrites": {
            name: {"from": STAGING_STATUS, "to": RELEASE_STATUS}
            for name in ASSET_NAMES
        },
        "binding_rewrites": {
            "rejector_asset_sha256": release_records[REJECTOR_WEIGHTS][
                "sha256"
            ],
            "classifier_asset_sha256": release_records[CLASSIFIER_WEIGHTS][
                "sha256"
            ],
            "openset_policy_asset_sha256": release_records[OPENSET_POLICY][
                "sha256"
            ],
            "release_binding_sha256": release_records[DUAL_BINDING]["sha256"],
        },
        "operation": {
            "development_data_loaded": False,
            "training_performed": False,
            "recalibration_performed": False,
            "sealed_corpus_opened_by_promoter": False,
        },
    }
    manifest_raw = _manifest_json_bytes(manifest)
    if len(manifest_raw) >= MAX_DEPLOYABLE_BYTES:
        raise PromotionError("release package manifest is not below 25 MiB")
    output[PACKAGE_MANIFEST] = manifest_raw
    return output


def _open_directory_fd(path: Path, label: str) -> int:
    absolute = Path(os.path.abspath(os.fspath(path)))
    flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_fd: int | None = None
    try:
        directory_fd = os.open(absolute.anchor, flags)
        for component in absolute.parts[1:]:
            next_fd = os.open(component, flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        return directory_fd
    except OSError as exc:
        if directory_fd is not None:
            os.close(directory_fd)
        raise PromotionError(f"cannot open {label} without symlinks") from exc


def _write_new_file_at(directory_fd: int, name: str, raw: bytes) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_fd: int | None = None
    try:
        file_fd = os.open(name, flags, 0o600, dir_fd=directory_fd)
        view = memoryview(raw)
        while view:
            written = os.write(file_fd, view)
            view = view[written:]
        os.fchmod(file_fd, 0o644)
        os.fsync(file_fd)
    except OSError as exc:
        raise PromotionError(f"cannot materialize {name}") from exc
    finally:
        if file_fd is not None:
            os.close(file_fd)


def _read_file_at(directory_fd: int, name: str, label: str) -> bytes:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_fd: int | None = None
    try:
        file_fd = os.open(name, flags, dir_fd=directory_fd)
        info = os.fstat(file_fd)
        if not stat.S_ISREG(info.st_mode):
            raise PromotionError(f"{label} is not a regular file")
        chunks: list[bytes] = []
        while True:
            block = os.read(file_fd, 1 << 20)
            if not block:
                return b"".join(chunks)
            chunks.append(block)
    except PromotionError:
        raise
    except OSError as exc:
        raise PromotionError(f"cannot read {label}") from exc
    finally:
        if file_fd is not None:
            os.close(file_fd)


def _verify_directory_fd(
    directory_fd: int, files: Mapping[str, bytes], label: str
) -> None:
    expected = set(STAGING_FILE_NAMES)
    try:
        names = set(os.listdir(directory_fd))
    except OSError as exc:
        raise PromotionError(f"cannot inspect {label}") from exc
    if names != expected:
        raise PromotionError(f"{label} has unexpected files")
    for name in expected:
        raw = _read_file_at(directory_fd, name, f"{label} {name}")
        if raw != files[name] or len(raw) >= MAX_DEPLOYABLE_BYTES:
            raise PromotionError(f"{label} {name} differs")


def _materialize(destination: Path, files: Mapping[str, bytes]) -> None:
    parent_fd = _open_directory_fd(
        destination.parent, "release destination parent"
    )
    parent_identity = os.fstat(parent_fd)
    temporary_name = (
        f".{destination.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    )
    temporary_fd: int | None = None
    moved = False
    renamed = False
    removed_empty = False

    def require_canonical_parent() -> None:
        check_fd = _open_directory_fd(
            destination.parent, "canonical release destination parent"
        )
        try:
            current = os.fstat(check_fd)
            if (
                current.st_dev != parent_identity.st_dev
                or current.st_ino != parent_identity.st_ino
            ):
                raise PromotionError(
                    "release destination parent identity changed"
                )
        finally:
            os.close(check_fd)

    try:
        os.mkdir(temporary_name, 0o700, dir_fd=parent_fd)
        directory_flags = (
            os.O_RDONLY
            | os.O_DIRECTORY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        temporary_fd = os.open(
            temporary_name,
            directory_flags,
            dir_fd=parent_fd,
        )
        for name in ASSET_NAMES:
            _write_new_file_at(temporary_fd, name, files[name])
        _write_new_file_at(
            temporary_fd,
            PACKAGE_MANIFEST,
            files[PACKAGE_MANIFEST],
        )
        _verify_directory_fd(temporary_fd, files, "temporary release")
        os.fchmod(temporary_fd, 0o755)
        os.fsync(temporary_fd)
        os.fsync(parent_fd)
        require_canonical_parent()

        try:
            destination_info = os.stat(
                destination.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            destination_info = None
        if destination_info is not None:
            if not stat.S_ISDIR(destination_info.st_mode):
                raise PromotionError("release destination changed")
            destination_fd = os.open(
                destination.name,
                directory_flags,
                dir_fd=parent_fd,
            )
            try:
                if os.listdir(destination_fd):
                    raise PromotionError(
                        "release destination became nonempty"
                    )
            finally:
                os.close(destination_fd)
            try:
                os.rmdir(destination.name, dir_fd=parent_fd)
            except OSError as exc:
                raise PromotionError("release destination changed") from exc
            removed_empty = True
        os.rename(
            temporary_name,
            destination.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        renamed = True
        installed_fd = os.open(
            destination.name,
            directory_flags,
            dir_fd=parent_fd,
        )
        try:
            installed = os.fstat(installed_fd)
            temporary_identity = os.fstat(temporary_fd)
            if (
                installed.st_dev != temporary_identity.st_dev
                or installed.st_ino != temporary_identity.st_ino
            ):
                raise PromotionError(
                    "installed release directory identity differs"
                )
            _verify_directory_fd(installed_fd, files, "installed release")
            os.fsync(installed_fd)
        finally:
            os.close(installed_fd)
        require_canonical_parent()
        os.fsync(parent_fd)
        moved = True
    except PromotionError:
        raise
    except OSError as exc:
        raise PromotionError("cannot atomically install release package") from exc
    finally:
        if not moved:
            cleanup_fd = temporary_fd
            cleanup_name = destination.name if renamed else temporary_name
            if cleanup_fd is None:
                try:
                    cleanup_fd = os.open(
                        cleanup_name,
                        (
                            os.O_RDONLY
                            | os.O_DIRECTORY
                            | getattr(os, "O_CLOEXEC", 0)
                            | getattr(os, "O_NOFOLLOW", 0)
                        ),
                        dir_fd=parent_fd,
                    )
                except OSError:
                    cleanup_fd = None
            if cleanup_fd is not None:
                cleanup_identity = os.fstat(cleanup_fd)
                try:
                    for name in os.listdir(cleanup_fd):
                        os.unlink(name, dir_fd=cleanup_fd)
                finally:
                    os.close(cleanup_fd)
                    if cleanup_fd == temporary_fd:
                        temporary_fd = None
                try:
                    named_identity = os.stat(
                        cleanup_name,
                        dir_fd=parent_fd,
                        follow_symlinks=False,
                    )
                    if (
                        named_identity.st_dev == cleanup_identity.st_dev
                        and named_identity.st_ino == cleanup_identity.st_ino
                    ):
                        os.rmdir(cleanup_name, dir_fd=parent_fd)
                except (FileNotFoundError, OSError):
                    pass
            if removed_empty:
                try:
                    os.mkdir(destination.name, 0o755, dir_fd=parent_fd)
                except FileExistsError:
                    pass
            try:
                os.fsync(parent_fd)
            except OSError:
                pass
        if temporary_fd is not None:
            os.close(temporary_fd)
        os.close(parent_fd)


def promote(
    staging_package: Path,
    evaluation_report: Path,
    destination: Path,
    *,
    policy: PromotionPolicy,
) -> dict[str, Any]:
    staging, release, normalized = _validate_paths(
        Path(staging_package), Path(destination), policy
    )
    report = _lexical_path(
        os.fspath(Path(evaluation_report).expanduser()),
        normalized.repo_root,
        "sealed evaluation report",
    )
    _reject_symlink_chain(
        report,
        normalized.repo_root,
        "sealed evaluation report",
    )
    if _is_within(report, release):
        raise PromotionError("evaluation report may not be inside destination")
    verified_staging = _verify_staging_package(staging, normalized)
    verified_evaluation = _verify_evaluation_report(
        report, verified_staging, normalized
    )
    files = _release_files(verified_staging, verified_evaluation)
    _materialize(release, files)
    return _parse_json(
        files[PACKAGE_MANIFEST], "release package manifest"
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--evaluation-report",
        type=Path,
        required=True,
        help="completed sealed seed-20260735 RELEASE_EVALUATION.json",
    )
    result.add_argument(
        "--staging-package",
        type=Path,
        default=DEFAULT_STAGING_PACKAGE,
    )
    result.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_RELEASE_PACKAGE,
    )
    return result


def main() -> None:
    arguments = parser().parse_args()
    policy = PromotionPolicy(
        repo_root=REPO,
        staging_package=DEFAULT_STAGING_PACKAGE,
        release_package=DEFAULT_RELEASE_PACKAGE,
        live_v2_assets=LIVE_V2_ASSETS,
        legacy_v3_staging=LEGACY_V3_STAGING,
        require_git_tracking=True,
    )
    try:
        manifest = promote(
            arguments.staging_package,
            arguments.evaluation_report,
            arguments.destination,
            policy=policy,
        )
    except PromotionError as exc:
        raise SystemExit(f"release promotion refused: {exc}") from exc
    print(
        f"promoted {len(manifest['assets'])} verified dual v3 assets to "
        f"{arguments.destination}"
    )


if __name__ == "__main__":
    main()
