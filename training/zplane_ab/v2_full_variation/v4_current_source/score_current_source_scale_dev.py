"""One-shot development-only scorer for the precommitted seed-20262903 corpus.

This module deliberately lives outside the open-set policy's frozen executed-
source list.  It consumes the already-fitted schema-5 classifier and policy,
and delegates every model score to the unchanged
``evaluate_held_scale_known`` implementation.  It never fits, selects, or
generates a candidate and it cannot produce validation or release evidence.

The workflow has two separate commands:

1. ``prepare-intent`` validates and immutably binds every input without
   scoring the scale corpus.
2. ``score`` consumes that exact intent once.  It durably creates the
   protocol-fixed score ledger before the first model inference, evaluates the
   frozen candidate, and atomically publishes a development report even when
   a numeric gate misses.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
V2 = HERE.parent
ZPLANE = V2.parent
TRAINING = ZPLANE.parent
REPO = TRAINING.parent
for path in (TRAINING, ZPLANE, V2, V2 / "v3_scale", HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import calibrate_current_source_openset as openset  # noqa: E402
import export_current_source_browser_runtime as browser_export  # noqa: E402


PROTOCOL_RELATIVE_PATH = (
    "training/zplane_ab/v2_full_variation/v4_current_source/"
    "current_scale_dev_protocol_seed20262903.json"
)
PROTOCOL_SHA256 = (
    "6169da9eb5510667c33ce01a487bfeeaef42946d8208d1115bcdfb3810c0e7a4"
)
INTENT_RELATIVE_PATH = (
    "training/artifacts/"
    "v4-current-source-scale-dev-seed20262903.score.intent.json"
)
LEDGER_RELATIVE_PATH = (
    "training/artifacts/"
    "v4-current-source-scale-dev-seed20262903.score.ledger.json"
)
REPORT_OUTPUT_RELATIVE_PATH = (
    ".artifacts/v4-current-source-scale-dev-seed20262903-score"
)
HISTORICAL_CORPUS_RELATIVE_PATH = "training/artifacts/signallab-corpus"
SCALE_CORPUS_RELATIVE_PATH = (
    "training/artifacts/"
    "signallab-current-scale-dev-v3-seed20262903-r24"
)
CURRENT_CORPUS_RELATIVE_PATH = (
    "training/artifacts/"
    "signallab-current-94171ba-lineage-v1-seed20260729-n288-s16384"
)
FUSION_RELATIVE_PATH = (
    ".artifacts/v4-current-source-lineage-share13-fusion-w0.5"
)
CLASSIFIER_RELATIVE_PATH = (
    ".artifacts/v4-current-source-lineage-share13-browser-staging/"
    "time-domain-profile-bank-v4.json"
)
POLICY_RELATIVE_PATH = (
    ".artifacts/v4-openset-schema5-final-independent-design-seed20263001/"
    "time-domain-profile-bank-openset-v4.json"
)
EXPECTED_SCALE_MANIFEST_SHA256 = (
    "b21c94c9be185deb461025396833838df5751266537827c90492b3f42fe7a06c"
)
EXPECTED_SCALE_RAW_SHA256 = (
    "8eb1c2263db197a3b06f71c0af1c203468fe077b422012fdc7f0d7a9ed2e850f"
)
EXPECTED_CLASSIFIER_SHA256 = (
    "c0dc837ddacd0bd3055701719a567e8011392180ad2421c007cc2a20e927d86a"
)
EXPECTED_POLICY_SHA256 = (
    "57f3262bb6c10785a766ef6834b88fc2c8daaeae3ef9d1c9e747658c8bfc3bea"
)
EXPECTED_HISTORICAL_MANIFEST_SHA256 = (
    "956bec65c285eef73be5e6beb46b366af6f414e445cc4a4c2e8fbfb7d9a042a2"
)
EXPECTED_HISTORICAL_RAW_SHA256 = (
    "233cdb636416de2e048809ba0897572e3012ec1c855ff4ca23163e43abdc3be4"
)
EFFECTIVE_EMBED_BATCH_SIZE = 256

INTENT_SCHEMA = "atomos.v4.current-service-scale-development-score-intent"
INTENT_SCHEMA_VERSION = 1
LEDGER_SCHEMA = "atomos.v4.current-service-scale-development-score-ledger"
LEDGER_SCHEMA_VERSION = 1
REPORT_SCHEMA = "atomos.v4.current-service-scale-development-score-report"
REPORT_SCHEMA_VERSION = 1
MANIFEST_SCHEMA = (
    "atomos.v4.current-service-scale-development-score-manifest"
)
MANIFEST_SCHEMA_VERSION = 1
REPORT_NAME = "scale-development-evaluation.json"
MANIFEST_NAME = "scale-development-manifest.json"


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _load_json(path: Path, role: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {role} JSON {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    except OSError as exc:
        raise ValueError(f"cannot hash required input {path}") from exc
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("durable JSON write made no progress")
        view = view[written:]


def _exclusive_durable_json(path: Path, payload: Mapping[str, Any]) -> str:
    """Create immutable evidence once and make the file and directory durable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _json_bytes(dict(payload))
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except FileExistsError as exc:
        raise RuntimeError(
            f"one-shot evidence already exists; refusing reuse: {path}"
        ) from exc
    try:
        _write_all(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    expected = hashlib.sha256(encoded).hexdigest()
    if _sha256(path) != expected:
        raise RuntimeError(f"immutable evidence changed after creation: {path}")
    return expected


def _atomic_durable_json(path: Path, payload: Mapping[str, Any]) -> str:
    """Publish complete JSON atomically without replacing any existing path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _json_bytes(dict(payload))
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary publication path exists: {temporary}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        _write_all(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        # Linking a fully fsynced temporary file gives the destination name
        # atomic visibility while failing if any file or symlink raced into
        # place.  Unlike os.replace, this cannot clobber prior evidence.
        os.link(temporary, path, follow_symlinks=False)
    except FileExistsError as exc:
        raise RuntimeError(
            f"refusing to replace existing publication evidence: {path}"
        ) from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    _fsync_directory(path.parent)
    expected = hashlib.sha256(encoded).hexdigest()
    if _sha256(path) != expected:
        raise RuntimeError(f"atomic publication changed bytes: {path}")
    return expected


def _repo_path(
    value: str | Path,
    *,
    role: str,
    must_exist: bool = True,
    directory: bool | None = None,
) -> Path:
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    # Do not allow an alias to hide a substituted input or evidence target.
    # REPO is derived from this source file's resolved path, so only components
    # below that already-resolved root need inspection.
    lexical = Path(os.path.abspath(raw))
    repo = REPO.resolve()
    try:
        relative = lexical.relative_to(repo)
    except ValueError as exc:
        raise ValueError(
            f"{role} must remain lexically inside the repository: {lexical}"
        ) from exc
    cursor = repo
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"{role} may not use a symlink: {cursor}")
    resolved = openset.reject_sensitive_path(value, role)
    if resolved != repo and repo not in resolved.parents:
        raise ValueError(f"{role} must remain inside the repository: {resolved}")
    if must_exist and not resolved.exists():
        raise ValueError(f"{role} does not exist: {resolved}")
    if must_exist and directory is True and not resolved.is_dir():
        raise ValueError(f"{role} must be a directory: {resolved}")
    if must_exist and directory is False and not resolved.is_file():
        raise ValueError(f"{role} must be a file: {resolved}")
    return resolved


def _assert_path_absent(path: Path, role: str) -> None:
    if os.path.lexists(path):
        raise FileExistsError(
            f"{role} already exists in some form; refusing reuse: {path}"
        )


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"path is outside the repository: {path}") from exc


def _path_from_relative(
    value: Any,
    *,
    role: str,
    must_exist: bool = True,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{role} relative path is invalid")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{role} relative path escapes the repository")
    return _repo_path(
        REPO / relative,
        role=role,
        must_exist=must_exist,
    )


def _require_exact_hash(path: Path, expected: Any, role: str) -> str:
    if not _is_sha256(expected):
        raise ValueError(f"{role} expected SHA-256 is invalid")
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(
            f"{role} SHA-256 changed: expected {expected}, got {observed}"
        )
    return observed


def _validate_empty_output(path: str | Path) -> Path:
    output = _repo_path(
        path,
        role="development score output",
        must_exist=False,
    )
    expected = (REPO / REPORT_OUTPUT_RELATIVE_PATH).resolve()
    if output != expected:
        raise ValueError("score output path differs from the frozen path")
    _assert_path_absent(output, "scale-development report output")
    return output


def _validate_protocol(path: str | Path) -> tuple[Path, dict[str, Any]]:
    protocol_path = _repo_path(path, role="scale-development protocol")
    expected_path = (REPO / PROTOCOL_RELATIVE_PATH).resolve()
    if protocol_path != expected_path:
        raise ValueError(
            "scale-development protocol path differs from the frozen path"
        )
    _require_exact_hash(protocol_path, PROTOCOL_SHA256, "protocol")
    protocol = _load_json(protocol_path, "scale-development protocol")
    generation = protocol.get("generation")
    firewall = protocol.get("fit_firewall")
    gates = protocol.get("gates")
    if (
        protocol.get("schema")
        != "atomos.v4.current-service-scale-development-protocol"
        or protocol.get("schema_version") != 1
        or protocol.get("status")
        != "precommitted_before_generation_or_scoring"
        or protocol.get("development_only") is not True
        or protocol.get("release_evidence") is not False
        or not isinstance(generation, Mapping)
        or not isinstance(firewall, Mapping)
        or not isinstance(gates, Mapping)
        or generation.get("eval_seed") != 20_262_903
        or generation.get("output_relative_path")
        != SCALE_CORPUS_RELATIVE_PATH
        or generation.get("reference_corpus_relative_path")
        != CURRENT_CORPUS_RELATIVE_PATH
        or generation.get("realizations_per_profile") != 24
        or generation.get("profiles") != len(
            openset.corpus_data.CURRENT_PROFILES
        )
        or generation.get("scale_factors")
        != list(openset.scale_eval.SCALE_FACTORS)
        or generation.get("expected_raw_rows") != 2_976
        or generation.get("expected_profile_legal_cells") != 370
        or generation.get("expected_profile_legal_observations") != 8_880
        or generation.get("low_level_waveform_synthesizer_calls") != 0
        or generation.get("frequency_transform_calls") != 0
        or generation.get("post_hoc_array_resampling") is not False
        or firewall.get("training_rows_used") != 0
        or firewall.get("enrollment_rows_used") != 0
        or firewall.get("threshold_or_rank_calibration_rows_used") != 0
        or firewall.get("candidate_selection_after_scoring") is not False
        or firewall.get("held_scale_seed_20262902_read") is not False
        or firewall.get(
            "reserved_validation_seeds_20263002_20263003_read"
        ) is not False
        or firewall.get(
            "candidate_bytes_and_executed_source_frozen_before_first_score"
        )
        is not True
        or firewall.get("score_ledger_relative_path")
        != LEDGER_RELATIVE_PATH
    ):
        raise ValueError("scale-development protocol contract changed")
    expected_gate_contract = {
        "exact_profile_length_scale_key_and_count_coverage": True,
        "exact_eligible_pair_and_observation_coverage": True,
        "pooled_closed_accuracy_floor":
            openset.HELD_CURRENT_POOLED_CLOSED_ACCURACY_FLOOR,
        "every_global_observation_length_scale_closed_accuracy_floor":
            openset.HELD_CURRENT_EVERY_LEGAL_CELL_CLOSED_ACCURACY_FLOOR,
        "worst_global_observation_length_scale_present_class_balanced_"
        "accuracy_floor":
            openset.HELD_CURRENT_WORST_CELL_PRESENT_CLASS_BALANCED_ACCURACY_FLOOR,
        "pooled_profile_balanced_accuracy_floor":
            openset.HELD_CURRENT_PROFILE_BALANCED_ACCURACY_FLOOR,
        "worst_source_profile_pooled_closed_accuracy_floor":
            openset.HELD_CURRENT_WORST_PROFILE_POOLED_CLOSED_ACCURACY_FLOOR,
        "worst_source_profile_observation_length_scale_closed_accuracy_"
        "floor":
            openset.HELD_CURRENT_WORST_PROFILE_LEGAL_CELL_CLOSED_ACCURACY_FLOOR,
        "wifi_hr_dsss_worst_legal_cell_accuracy_floor":
            openset.HELD_CURRENT_WIFI_HR_DSSS_ACCURACY_FLOOR,
        "known_abstention_damage_given_closed_head_correct_pooled_ceiling":
            openset.KNOWN_FALSE_UNKNOWN_CEILING,
        "known_abstention_damage_given_closed_head_correct_worst_"
        "observation_length_ceiling":
            openset.KNOWN_FALSE_UNKNOWN_CEILING,
        "known_abstention_damage_given_closed_head_correct_worst_"
        "supported_profile_cell_ceiling":
            openset.KNOWN_FALSE_UNKNOWN_CEILING,
        "known_abstention_damage_given_closed_head_correct_worst_"
        "supported_true_class_cell_ceiling":
            openset.KNOWN_FALSE_UNKNOWN_CEILING,
        "cell_gate_minimum_closed_head_correct_rows":
            openset.KNOWN_CELL_MIN_CLOSED_CORRECT_ROWS,
        "causal_prefix_outcome_agreement_across_observation_lengths_floor":
            openset.CURRENT_SELECTION_CAUSAL_AGREEMENT_FLOOR,
        "causal_prefix_outcome_agreement_across_observation_lengths_every_"
        "source_profile_floor":
            openset.CURRENT_SELECTION_CAUSAL_AGREEMENT_FLOOR,
        "physical_scale_outcome_agreement_pooled_floor":
            openset.LENGTH_SCALE_DECISION_AGREEMENT_FLOOR,
        "physical_scale_outcome_agreement_every_observation_length_floor":
            openset.LENGTH_SCALE_DECISION_AGREEMENT_FLOOR,
        "length_scale_outcome_agreement_pooled_floor":
            openset.LENGTH_SCALE_DECISION_AGREEMENT_FLOOR,
        "length_scale_outcome_agreement_every_source_profile_floor":
            openset.LENGTH_SCALE_DECISION_AGREEMENT_FLOOR,
    }
    if dict(gates) != expected_gate_contract:
        raise ValueError("scale-development gate contract changed")
    return protocol_path, protocol


def _corpus_file_binding(
    directory: Path,
    *,
    manifest_name: str,
    raw_name: str,
) -> dict[str, Any]:
    manifest = _repo_path(
        directory / manifest_name,
        role=f"{manifest_name} input",
        directory=False,
    )
    raw = _repo_path(
        directory / raw_name,
        role=f"{raw_name} input",
        directory=False,
    )
    return {
        "relative_path": _relative(directory),
        "manifest_relative_path": _relative(manifest),
        "manifest_sha256": _sha256(manifest),
        "raw_relative_path": _relative(raw),
        "raw_sha256": _sha256(raw),
    }


def _validate_generation_inputs(
    protocol: Mapping[str, Any],
    *,
    scale_corpus: str | Path,
    current_corpus: str | Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    openset.scale_eval.ScaleEvalCorpus,
]:
    generation = protocol["generation"]
    scale_path = _repo_path(
        scale_corpus,
        role="scale-development corpus",
        directory=True,
    )
    expected_scale = (
        REPO / str(generation["output_relative_path"])
    ).resolve()
    if scale_path != expected_scale:
        raise ValueError("scale-development corpus path changed")
    scale_binding = _corpus_file_binding(
        scale_path,
        manifest_name="scale_eval.json",
        raw_name="scale_eval.f32",
    )
    corpus = openset.scale_eval.load_scale_eval_corpus(scale_path)
    legal_cell_rows: dict[str, int] = {}
    observation_keys: set[tuple[str, int, float]] = set()
    for row in corpus.rows:
        for length in row.available_lengths:
            if int(length) not in openset.RUNTIME_INPUT_LENGTHS:
                continue
            legal_key = openset._profile_legal_cell_key(
                f"current:{row.profile}",
                int(length),
                float(row.scale_factor),
            )
            legal_cell_rows[legal_key] = legal_cell_rows.get(legal_key, 0) + 1
            observation_key = (
                str(row.pair_id),
                int(length),
                float(row.scale_factor),
            )
            if observation_key in observation_keys:
                raise ValueError(
                    "scale-development corpus repeats a pair/length/scale "
                    f"observation: {observation_key}"
                )
            observation_keys.add(observation_key)
    if (
        scale_binding["manifest_sha256"]
        != EXPECTED_SCALE_MANIFEST_SHA256
        or scale_binding["raw_sha256"] != EXPECTED_SCALE_RAW_SHA256
        or corpus.audit.get("manifest_sha256")
        != scale_binding["manifest_sha256"]
        or corpus.audit.get("raw_sha256") != scale_binding["raw_sha256"]
        or corpus.manifest.get("evalSeed") != generation["eval_seed"]
        or corpus.audit.get("realizations_per_profile")
        != generation["realizations_per_profile"]
        or corpus.audit.get("count") != generation["expected_raw_rows"]
        or corpus.audit.get("pair_count")
        != (
            generation["profiles"]
            * generation["realizations_per_profile"]
        )
        or corpus.audit.get("prefix_lengths")
        != list(openset.RUNTIME_INPUT_LENGTHS)
        or corpus.audit.get("scale_factors")
        != list(openset.scale_eval.SCALE_FACTORS)
        or corpus.audit.get("profiles")
        != list(openset.corpus_data.CURRENT_PROFILES)
        or corpus.audit.get("unique_valid_content_hashes")
        != generation["expected_raw_rows"]
        or any(
            count != generation["realizations_per_profile"]
            for count in corpus.audit.get("pairs_per_profile", {}).values()
        )
        or len(legal_cell_rows)
        != generation["expected_profile_legal_cells"]
        or sum(legal_cell_rows.values())
        != generation["expected_profile_legal_observations"]
        or len(observation_keys)
        != generation["expected_profile_legal_observations"]
    ):
        raise ValueError("scale-development corpus structural audit changed")

    current_path = _repo_path(
        current_corpus,
        role="current reference corpus",
        directory=True,
    )
    expected_current = (
        REPO / str(generation["reference_corpus_relative_path"])
    ).resolve()
    if current_path != expected_current:
        raise ValueError("current reference corpus path changed")
    current_binding = _corpus_file_binding(
        current_path,
        manifest_name="corpus.json",
        raw_name="corpus.f32",
    )
    if (
        current_binding["manifest_relative_path"]
        != generation["reference_manifest_relative_path"]
        or current_binding["manifest_sha256"]
        != generation["reference_manifest_sha256"]
        or current_binding["raw_relative_path"]
        != generation["reference_raw_relative_path"]
        or current_binding["raw_sha256"]
        != generation["reference_raw_sha256"]
    ):
        raise ValueError("current reference corpus bytes changed")

    generator_binding: dict[str, Any] = {}
    for path_key, hash_key, role, output_key in (
        (
            "generator_source_relative_path",
            "generator_source_sha256",
            "scale generator source",
            "source",
        ),
        (
            "generator_bundle_relative_path",
            "generator_bundle_sha256",
            "scale generator bundle",
            "bundle",
        ),
    ):
        pinned = _path_from_relative(generation[path_key], role=role)
        _require_exact_hash(pinned, generation[hash_key], role)
        generator_binding[output_key] = {
            "relative_path": _relative(pinned),
            "sha256": str(generation[hash_key]),
        }
    scale_binding["generator"] = generator_binding
    return scale_binding, current_binding, corpus


def _canonical_reference_separation(
    value: Mapping[str, Any],
    *,
    expected_directory: Path,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("reference-separation report must be an object")
    report = dict(value)
    directory = report.pop("directory", None)
    if (
        not isinstance(directory, (str, Path))
        or Path(directory).resolve() != expected_directory.resolve()
    ):
        raise ValueError("reference-separation directory changed")
    report["directory_relative_path"] = _relative(expected_directory)
    required = {
        "manifest": "corpus.json",
        "raw": "corpus.f32",
        "content_collision_count": 0,
        "cyclic_phase_collision_count": 0,
        "one_shot_receiver_seed_collision_count": 0,
        "candidate_current_audit_match": True,
        "signal_lab_source_match": True,
    }
    if any(report.get(key) != expected for key, expected in required.items()):
        raise ValueError("reference-separation evidence changed")
    if (
        report.get("manifest_sha256")
        != openset.corpus_data.sha256_file(
            expected_directory / "corpus.json"
        )
        or report.get("raw_sha256")
        != openset.corpus_data.sha256_file(
            expected_directory / "corpus.f32"
        )
        or int(report.get("reference_rows", 0)) <= 0
        or int(report.get("reference_content_hashes", 0)) <= 0
    ):
        raise ValueError("reference-separation reference identity changed")
    return report


def _reference_separation_binding(
    corpus: openset.scale_eval.ScaleEvalCorpus,
    *,
    current_corpus: Path,
    context: openset.DevelopmentContext,
) -> dict[str, Any]:
    raw = openset.scale_eval.validate_reference_separation(
        corpus,
        current_corpus,
        candidate_current_audit=context.fusion["metrics"]["data_audit"][
            "current"
        ],
    )
    report = _canonical_reference_separation(
        raw,
        expected_directory=current_corpus,
    )
    return {
        "report": report,
        "sha256": hashlib.sha256(_json_bytes(report)).hexdigest(),
    }


def _candidate_binding(
    *,
    fusion: str | Path,
    historical_corpus: str | Path,
    current_corpus: str | Path,
    classifier_asset: str | Path,
    policy_path: str | Path,
    device: str,
) -> tuple[
    dict[str, Any],
    openset.DevelopmentContext,
    dict[str, Any],
]:
    fusion_path = _repo_path(fusion, role="fusion candidate", directory=True)
    if fusion_path != (REPO / FUSION_RELATIVE_PATH).resolve():
        raise ValueError("fusion path differs from the frozen path")
    historical_path = _repo_path(
        historical_corpus,
        role="historical corpus",
        directory=True,
    )
    if historical_path != (
        REPO / HISTORICAL_CORPUS_RELATIVE_PATH
    ).resolve():
        raise ValueError("historical corpus path differs from the frozen path")
    current_path = _repo_path(
        current_corpus,
        role="current corpus",
        directory=True,
    )
    if current_path != (REPO / CURRENT_CORPUS_RELATIVE_PATH).resolve():
        raise ValueError("current corpus path differs from the frozen path")
    classifier_path = _repo_path(
        classifier_asset,
        role="classifier candidate",
        directory=False,
    )
    if classifier_path != (REPO / CLASSIFIER_RELATIVE_PATH).resolve():
        raise ValueError("classifier path differs from the frozen path")
    policy_file = _repo_path(
        policy_path,
        role="open-set policy candidate",
        directory=False,
    )
    if policy_file != (REPO / POLICY_RELATIVE_PATH).resolve():
        raise ValueError("policy path differs from the frozen path")
    if not isinstance(device, str) or not device:
        raise ValueError("scoring device is empty")
    classifier_sha = _require_exact_hash(
        classifier_path,
        EXPECTED_CLASSIFIER_SHA256,
        "final schema-5 classifier",
    )
    policy_sha = _require_exact_hash(
        policy_file,
        EXPECTED_POLICY_SHA256,
        "final schema-5 policy",
    )
    historical_binding = _corpus_file_binding(
        historical_path,
        manifest_name="corpus.json",
        raw_name="corpus.f32",
    )
    if (
        historical_binding["manifest_sha256"]
        != EXPECTED_HISTORICAL_MANIFEST_SHA256
        or historical_binding["raw_sha256"]
        != EXPECTED_HISTORICAL_RAW_SHA256
    ):
        raise ValueError("historical corpus bytes changed")

    context = openset.load_development_context(
        fusion_path,
        historical_path,
        current_path,
        device_name=device,
    )
    expected_classifier_binding = openset.build_classifier_binding(
        context.fusion,
        classifier_path,
    )
    policy = openset.validate_policy(
        _load_json(policy_file, "schema-5 policy"),
        expected_fusion_binding=expected_classifier_binding,
    )
    openset._validate_policy_corpus_binding(policy, context)
    if policy["classifier_binding"] != expected_classifier_binding:
        raise ValueError("policy/classifier/fusion binding changed")
    metrics_path = _repo_path(
        context.fusion["metrics_path"],
        role="fusion development metrics",
        directory=False,
    )
    expected_metrics_path = fusion_path / "dev_metrics.json"
    if metrics_path != expected_metrics_path:
        raise ValueError("fusion development-metrics path changed")
    metrics_sha = _require_exact_hash(
        metrics_path,
        expected_classifier_binding["source_fusion_dev_metrics_sha256"],
        "fusion development metrics",
    )
    expected_artifacts = expected_classifier_binding[
        "source_fusion_artifacts_sha256"
    ]
    if not isinstance(expected_artifacts, Mapping) or not expected_artifacts:
        raise ValueError("fusion artifact binding is empty")
    fusion_artifacts: dict[str, str] = {}
    for name, expected_sha in sorted(expected_artifacts.items()):
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
        ):
            raise ValueError("fusion artifact binding has an unsafe name")
        artifact = _repo_path(
            fusion_path / name,
            role=f"fusion artifact {name}",
            directory=False,
        )
        fusion_artifacts[name] = _require_exact_hash(
            artifact,
            expected_sha,
            f"fusion artifact {name}",
        )
    return (
        {
            "fusion_relative_path": _relative(fusion_path),
            "fusion_classifier_binding": expected_classifier_binding,
            "fusion_dev_metrics": {
                "relative_path": _relative(metrics_path),
                "sha256": metrics_sha,
            },
            "fusion_artifacts": {
                "directory_relative_path": _relative(fusion_path),
                "files_sha256": fusion_artifacts,
            },
            "historical_corpus": historical_binding,
            "classifier_relative_path": _relative(classifier_path),
            "classifier_sha256": classifier_sha,
            "policy_relative_path": _relative(policy_file),
            "policy_sha256": policy_sha,
        },
        context,
        policy,
    )


def build_intent_payload(
    *,
    protocol_path: str | Path,
    scale_corpus: str | Path,
    fusion: str | Path,
    historical_corpus: str | Path,
    current_corpus: str | Path,
    classifier_asset: str | Path,
    policy_path: str | Path,
    output_dir: str | Path,
    device: str,
) -> tuple[
    dict[str, Any],
    openset.DevelopmentContext,
    dict[str, Any],
]:
    """Validate all non-secret inputs and return deterministic intent bytes."""
    resolved_protocol, protocol = _validate_protocol(protocol_path)
    output = _validate_empty_output(output_dir)
    intent_path = (REPO / INTENT_RELATIVE_PATH).resolve()
    ledger_path = _path_from_relative(
        protocol["fit_firewall"]["score_ledger_relative_path"],
        role="scale-development score ledger",
        must_exist=False,
    )
    if _relative(ledger_path) != LEDGER_RELATIVE_PATH:
        raise ValueError("score ledger path differs from the frozen path")
    # These checks precede scale-corpus parsing, context reconstruction, and
    # model loading.  Presence in any form permanently blocks a redraw.
    _assert_path_absent(ledger_path, "scale-development score ledger")
    _assert_path_absent(output, "scale-development report output")
    if intent_path == ledger_path or output in {
        intent_path.parent,
        ledger_path.parent,
    }:
        raise ValueError("intent, ledger, and report output paths overlap")
    scale_binding, current_binding, scale_eval_corpus = (
        _validate_generation_inputs(
        protocol,
        scale_corpus=scale_corpus,
        current_corpus=current_corpus,
        )
    )
    candidate, context, policy = _candidate_binding(
        fusion=fusion,
        historical_corpus=historical_corpus,
        current_corpus=current_corpus,
        classifier_asset=classifier_asset,
        policy_path=policy_path,
        device=device,
    )
    reference_separation = _reference_separation_binding(
        scale_eval_corpus,
        current_corpus=(
            REPO / current_binding["relative_path"]
        ).resolve(),
        context=context,
    )
    if candidate["policy_sha256"] != EXPECTED_POLICY_SHA256:
        raise AssertionError("candidate policy pin changed")
    source_hashes = openset._source_hashes()
    if policy["provenance"]["source_sha256"] != source_hashes:
        raise ValueError("policy executed-source snapshot is no longer current")
    scorer_relative = _relative(Path(__file__).resolve())
    embed_parameter = inspect.signature(
        openset.branch_runner.embed_all
    ).parameters.get("batch")
    if (
        embed_parameter is None
        or embed_parameter.default != EFFECTIVE_EMBED_BATCH_SIZE
    ):
        raise ValueError("effective embedding batch default changed")
    payload = {
        "schema": INTENT_SCHEMA,
        "schema_version": INTENT_SCHEMA_VERSION,
        "status": "prepared_before_first_model_inference",
        "development_only": True,
        "release_evidence": False,
        "independent_validation_evidence": False,
        "one_shot": True,
        "candidate_selection_after_scoring": False,
        "protocol": {
            "relative_path": _relative(resolved_protocol),
            "sha256": PROTOCOL_SHA256,
        },
        "scale_development_corpus": scale_binding,
        "current_reference_corpus": current_binding,
        "reference_separation": reference_separation,
        "candidate": candidate,
        "executed_source_sha256": source_hashes,
        "scorer_source": {
            "relative_path": scorer_relative,
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "execution": {
            "device": device,
            "effective_embed_batch_size": EFFECTIVE_EMBED_BATCH_SIZE,
            "uses_frequency_transform": False,
            "model_score_entrypoint":
                "calibrate_current_source_openset.evaluate_held_scale_known",
            "novelty_evaluation_performed": False,
            "held_seed_20262902_scored": False,
            "reserved_validation_seeds_20263002_20263003_scored": False,
        },
        "score_ledger_relative_path": _relative(ledger_path),
        "report_output_relative_path": _relative(output),
        "deletion_or_replacement_invalidates_score_evidence": True,
    }
    return payload, context, policy


def prepare_intent(args: argparse.Namespace) -> dict[str, Any]:
    expected_intent = (REPO / INTENT_RELATIVE_PATH).resolve()
    requested_intent = _repo_path(
        args.intent_output,
        role="scale-development score intent",
        must_exist=False,
    )
    if requested_intent != expected_intent:
        raise ValueError("score intent path differs from the frozen path")
    _assert_path_absent(
        requested_intent,
        "scale-development score intent",
    )
    _assert_path_absent(
        REPO / LEDGER_RELATIVE_PATH,
        "scale-development score ledger",
    )
    _assert_path_absent(
        REPO / REPORT_OUTPUT_RELATIVE_PATH,
        "scale-development report output",
    )
    payload, _context, _policy = build_intent_payload(
        protocol_path=args.protocol,
        scale_corpus=args.scale_corpus,
        fusion=args.fusion,
        historical_corpus=args.historical_corpus,
        current_corpus=args.current_corpus,
        classifier_asset=args.classifier_asset,
        policy_path=args.policy,
        output_dir=args.output_dir,
        device=args.device,
    )
    digest = _exclusive_durable_json(requested_intent, payload)
    print(
        "[v4 scale development] prepared immutable intent "
        f"{requested_intent} sha256={digest}",
        flush=True,
    )
    print(
        "[v4 scale development] no model inference was performed",
        flush=True,
    )
    return payload


def _intent_paths(intent: Mapping[str, Any]) -> dict[str, Path]:
    """Parse only repository-relative paths; full equality is checked later."""
    protocol = intent.get("protocol")
    corpus = intent.get("scale_development_corpus")
    current = intent.get("current_reference_corpus")
    candidate = intent.get("candidate")
    scorer = intent.get("scorer_source")
    if not all(
        isinstance(value, Mapping)
        for value in (protocol, corpus, current, candidate, scorer)
    ):
        raise ValueError("score intent bindings are incomplete")
    historical = candidate.get("historical_corpus")
    if not isinstance(historical, Mapping):
        raise ValueError("intent historical corpus binding is invalid")
    return {
        "protocol": _path_from_relative(
            protocol.get("relative_path"), role="intent protocol"
        ),
        "scale_corpus": _path_from_relative(
            corpus.get("relative_path"), role="intent scale corpus"
        ),
        "current_corpus": _path_from_relative(
            current.get("relative_path"), role="intent current corpus"
        ),
        "fusion": _path_from_relative(
            candidate.get("fusion_relative_path"), role="intent fusion"
        ),
        "historical_corpus": _path_from_relative(
            historical.get("relative_path"),
            role="intent historical corpus",
        ),
        "classifier_asset": _path_from_relative(
            candidate.get("classifier_relative_path"),
            role="intent classifier",
        ),
        "policy": _path_from_relative(
            candidate.get("policy_relative_path"), role="intent policy"
        ),
        "scorer": _path_from_relative(
            scorer.get("relative_path"), role="intent scorer source"
        ),
        "ledger": _path_from_relative(
            intent.get("score_ledger_relative_path"),
            role="intent ledger",
            must_exist=False,
        ),
        "output": _repo_path(
            REPO / str(intent.get("report_output_relative_path", "")),
            role="intent report output",
            must_exist=False,
        ),
    }


def _validate_intent(
    intent_path: Path,
) -> tuple[
    dict[str, Any],
    str,
    dict[str, Path],
    openset.DevelopmentContext,
    dict[str, Any],
    dict[str, Any],
]:
    intent = _load_json(intent_path, "scale-development score intent")
    paths = _intent_paths(intent)
    if paths["scorer"] != Path(__file__).resolve():
        raise ValueError("score intent names another scorer source")
    execution = intent.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("score intent execution binding is missing")
    rebuilt, context, policy = build_intent_payload(
        protocol_path=paths["protocol"],
        scale_corpus=paths["scale_corpus"],
        fusion=paths["fusion"],
        historical_corpus=paths["historical_corpus"],
        current_corpus=paths["current_corpus"],
        classifier_asset=paths["classifier_asset"],
        policy_path=paths["policy"],
        output_dir=paths["output"],
        device=str(execution.get("device", "")),
    )
    if intent != rebuilt:
        raise ValueError("score intent is stale, tampered, or non-canonical")
    protocol = _load_json(paths["protocol"], "scale-development protocol")
    return (
        intent,
        _sha256(intent_path),
        paths,
        context,
        policy,
        protocol,
    )


def _development_evaluator_protocol(
    intent: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Translate only the frozen dev binding into the held evaluator's API."""
    corpus = intent["scale_development_corpus"]
    generation = protocol["generation"]
    binding = openset.validation_corpus_binding_sha256(
        corpus["manifest_sha256"],
        corpus["raw_sha256"],
    )
    return {
        "held_scale_manifest_sha256": corpus["manifest_sha256"],
        "held_scale_raw_sha256": corpus["raw_sha256"],
        "held_scale_eval_seed": generation["eval_seed"],
        "held_scale_realizations_per_profile":
            generation["realizations_per_profile"],
        "held_scale_corpus_binding_sha256": binding,
    }


def _claim_score_ledger(
    *,
    path: Path,
    intent: Mapping[str, Any],
    intent_sha256: str,
) -> tuple[dict[str, Any], str]:
    payload = {
        "schema": LEDGER_SCHEMA,
        "schema_version": LEDGER_SCHEMA_VERSION,
        "state": "development_scale_score_claimed",
        "development_only": True,
        "release_evidence": False,
        "independent_validation_evidence": False,
        "one_shot": True,
        "intent_sha256": intent_sha256,
        "protocol_sha256": intent["protocol"]["sha256"],
        "scale_manifest_sha256":
            intent["scale_development_corpus"]["manifest_sha256"],
        "scale_raw_sha256":
            intent["scale_development_corpus"]["raw_sha256"],
        "classifier_sha256": intent["candidate"]["classifier_sha256"],
        "policy_sha256": intent["candidate"]["policy_sha256"],
        "fusion_dev_metrics_sha256":
            intent["candidate"]["fusion_dev_metrics"]["sha256"],
        "fusion_artifacts_sha256":
            intent["candidate"]["fusion_artifacts"]["files_sha256"],
        "generator_source_sha256":
            intent["scale_development_corpus"]["generator"]["source"][
                "sha256"
            ],
        "generator_bundle_sha256":
            intent["scale_development_corpus"]["generator"]["bundle"][
                "sha256"
            ],
        "reference_separation_sha256":
            intent["reference_separation"]["sha256"],
        "executed_source_sha256": intent["executed_source_sha256"],
        "scorer_source_sha256": intent["scorer_source"]["sha256"],
        "created_before_first_model_inference": True,
        "deletion_or_replacement_invalidates_score_evidence": True,
    }
    return payload, _exclusive_durable_json(path, payload)


def _relabel_development_population(
    raw_known: Mapping[str, Any],
) -> dict[str, Any]:
    known = deepcopy(dict(raw_known))
    known["population_role"] = (
        "precommitted seed-20262903 current-service physical-scale "
        "development check; candidate frozen before scoring; not independent "
        "validation or release evidence"
    )
    for length_rows in known.get("per_length_scale", {}).values():
        if not isinstance(length_rows, Mapping):
            continue
        for summary in length_rows.values():
            if isinstance(summary, dict):
                summary["source"] = (
                    "precommitted seed-20262903 development service-scale "
                    "population"
                )
    for cells in known.get("gate_cells", {}).values():
        if not isinstance(cells, Sequence):
            continue
        for cell in cells:
            if isinstance(cell, dict):
                cell["source"] = "development_service_scale_seed20262903"
    return known


def _conditional_abstention_gates(
    known: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    combined = known.get("combined")
    if not isinstance(combined, Mapping):
        raise ValueError("development known combined metrics are missing")
    length_damage = {
        str(length): float(value)
        for length, value in combined.get(
            "per_length_abstention_damage_given_closed_head_correct", {}
        ).items()
        if value is not None
    }
    length_correct = {
        str(length): int(value)
        for length, value in combined.get(
            "per_length_closed_head_correct_rows", {}
        ).items()
    }
    expected_lengths = {
        str(length) for length in openset.RUNTIME_INPUT_LENGTHS
    }
    if (
        set(length_damage) != expected_lengths
        or set(length_correct) != expected_lengths
        or any(value <= 0 for value in length_correct.values())
        or not all(math.isfinite(value) for value in length_damage.values())
    ):
        raise ValueError("conditional abstention length support is incomplete")
    pooled = combined.get(
        "pooled_abstention_damage_given_closed_head_correct"
    )
    pooled_correct = int(
        combined.get("pooled_closed_head_correct_rows", 0)
    )
    if (
        pooled is None
        or pooled_correct <= 0
        or not math.isfinite(float(pooled))
    ):
        raise ValueError("conditional pooled abstention metric is invalid")
    pooled_value = float(pooled)
    worst_length = max(length_damage, key=length_damage.__getitem__)
    class_gate = openset._abstention_cell_gate(
        list(known["gate_cells"]["true_class"]),
        label_field="name",
        label_output_field="true_public_class",
    )
    profile_gate = openset._abstention_cell_gate(
        list(known["gate_cells"]["source_profile"]),
        label_field="name",
        label_output_field="source_profile",
    )
    return {
        "known_abstention_damage_given_closed_head_correct_pooled": {
            "value": pooled_value,
            "ceiling": openset.KNOWN_FALSE_UNKNOWN_CEILING,
            "closed_head_correct_rows": pooled_correct,
            "conditioning": "closed_head_prediction_equals_truth",
            "passes": pooled_value <= openset.KNOWN_FALSE_UNKNOWN_CEILING,
        },
        "known_abstention_damage_given_closed_head_correct_worst_"
        "observation_length": {
            "value": length_damage[worst_length],
            "ceiling": openset.KNOWN_FALSE_UNKNOWN_CEILING,
            "observation_length": int(worst_length),
            "closed_head_correct_rows": length_correct[worst_length],
            "conditioning": "closed_head_prediction_equals_truth",
            "all_observation_lengths": length_damage,
            "passes":
                length_damage[worst_length]
                <= openset.KNOWN_FALSE_UNKNOWN_CEILING,
        },
        "known_abstention_damage_given_closed_head_correct_worst_"
        "supported_true_class_cell": class_gate,
        "known_abstention_damage_given_closed_head_correct_worst_"
        "supported_source_profile_cell": profile_gate,
    }


def _physical_scale_gates(
    known: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], bool]:
    physical = known.get("paired_physical_scale_invariance")
    if not isinstance(physical, Mapping):
        raise ValueError("physical-scale invariance metrics are missing")
    expected = physical.get(
        "expected_legal_scales_by_pair_observation_length"
    )
    observed = physical.get(
        "observed_legal_scales_by_pair_observation_length"
    )
    by_length = physical.get("by_observation_length")
    pair_cells = physical.get("pair_observation_length_cells")
    expected_profiles = {
        f"current:{profile}"
        for profile in openset.corpus_data.CURRENT_PROFILES
    }
    inventory_ok = (
        isinstance(expected, Mapping)
        and bool(expected)
        and observed == expected
        and isinstance(by_length, Mapping)
        and set(by_length)
        == {str(length) for length in openset.RUNTIME_INPUT_LENGTHS}
        and isinstance(pair_cells, list)
    )
    seen: set[str] = set()
    pair_profile: dict[str, str] = {}
    lengths_by_pair: dict[str, set[int]] = {}
    pairs_by_profile: dict[str, set[str]] = {
        profile: set() for profile in expected_profiles
    }
    all_scales = tuple(float(value) for value in openset.scale_eval.SCALE_FACTORS)
    if inventory_ok:
        for cell in pair_cells:
            if not isinstance(cell, Mapping):
                inventory_ok = False
                break
            try:
                pair_id = str(cell["pair_id"])
                profile = str(cell["source_profile"])
                length = int(cell["observation_length"])
                expected_scales = tuple(
                    float(value)
                    for value in cell["expected_legal_scale_factors"]
                )
                observed_scales = tuple(
                    float(value)
                    for value in cell["observed_legal_scale_factors"]
                )
            except (KeyError, TypeError, ValueError):
                inventory_ok = False
                break
            required_scales = (
                (1.5, 2.0)
                if (
                    profile == "current:bluetooth-le-advertising"
                    and length == 16_384
                )
                else all_scales
            )
            key = f"{pair_id}|N={length}"
            previous = pair_profile.setdefault(pair_id, profile)
            if (
                not pair_id
                or profile not in expected_profiles
                or length not in openset.RUNTIME_INPUT_LENGTHS
                or key in seen
                or previous != profile
                or expected_scales != required_scales
                or observed_scales != required_scales
                or tuple(float(value) for value in expected.get(key, ()))
                != required_scales
                or tuple(float(value) for value in observed.get(key, ()))
                != required_scales
                or cell.get("exact_legal_scale_coverage") is not True
                or int(cell.get("distinct_legal_scale_count", -1))
                != len(required_scales)
            ):
                inventory_ok = False
                break
            seen.add(key)
            pairs_by_profile[profile].add(pair_id)
            lengths_by_pair.setdefault(pair_id, set()).add(length)
    if inventory_ok:
        inventory_ok = (
            seen == set(expected) == set(observed)
            and set(pair_profile.values()) == expected_profiles
            and all(
                len(pair_ids)
                == openset.DEFAULT_HELD_SCALE_REALIZATIONS_PER_PROFILE
                for pair_ids in pairs_by_profile.values()
            )
            and all(
                lengths == set(openset.RUNTIME_INPUT_LENGTHS)
                for lengths in lengths_by_pair.values()
            )
        )
    exact = (
        inventory_ok
        and physical.get("exact_eligible_cell_coverage") is True
        and int(physical.get("expected_cells", -1))
        == int(physical.get("cells", -2))
        == len(expected)
        == openset.EXPECTED_HELD_PHYSICAL_SCALE_CELLS
        and int(physical.get("expected_observation_cells", -1))
        == int(physical.get("observed_observation_cells", -2))
        == openset.EXPECTED_HELD_OBSERVATION_CELLS
        and int(physical.get("minimum_distinct_legal_scales", 0)) >= 2
    )
    rows: dict[str, dict[str, Any]] = {}
    if isinstance(by_length, Mapping):
        for length, row in sorted(by_length.items()):
            if not isinstance(row, Mapping):
                raise ValueError("physical-scale length metric is invalid")
            value = float(
                row.get(
                    "prediction_or_abstention_agreement_rate",
                    float("nan"),
                )
            )
            coverage = (
                int(row.get("expected_cells", -1))
                == int(row.get("observed_cells", -2))
                == openset.EXPECTED_HELD_PHYSICAL_SCALE_CELLS_PER_LENGTH
                and int(row.get("minimum_distinct_legal_scales", 0)) >= 2
            )
            rows[str(length)] = {
                "value": value,
                "exact_coverage": coverage,
                "passes": (
                    coverage
                    and math.isfinite(value)
                    and value >= openset.LENGTH_SCALE_DECISION_AGREEMENT_FLOOR
                ),
            }
    if set(rows) != {
        str(length) for length in openset.RUNTIME_INPUT_LENGTHS
    }:
        raise ValueError("physical-scale length metrics are incomplete")
    value = float(
        physical.get(
            "prediction_or_abstention_agreement_rate",
            float("nan"),
        )
    )
    worst = min(rows, key=lambda key: rows[key]["value"])
    return (
        {
            "physical_scale_outcome_agreement_pooled": {
                "value": value,
                "floor": openset.LENGTH_SCALE_DECISION_AGREEMENT_FLOOR,
                "exact_coverage": exact,
                "passes": (
                    exact
                    and math.isfinite(value)
                    and value
                    >= openset.LENGTH_SCALE_DECISION_AGREEMENT_FLOOR
                ),
            },
            "physical_scale_outcome_agreement_every_observation_length": {
                **rows[worst],
                "observation_length": int(worst),
                "floor": openset.LENGTH_SCALE_DECISION_AGREEMENT_FLOOR,
                "all_observation_lengths": rows,
                "passes": exact and all(
                    row["passes"] is True for row in rows.values()
                ),
            },
        },
        exact,
    )


def _causal_prefix_gates(
    known: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], bool]:
    causal = known.get("causal_prefix_length_invariance")
    if not isinstance(causal, Mapping):
        raise ValueError("causal-prefix invariance metrics are missing")
    expected = causal.get("expected_observation_lengths_by_pair_scale")
    observed = causal.get("observed_observation_lengths_by_pair_scale")
    by_profile = causal.get("by_source_profile")
    expected_profiles = {
        f"current:{profile}"
        for profile in openset.corpus_data.CURRENT_PROFILES
    }
    rows: dict[str, dict[str, Any]] = {}
    if isinstance(by_profile, Mapping):
        for profile, row in sorted(by_profile.items()):
            if not isinstance(row, Mapping):
                raise ValueError("causal-prefix profile metric is invalid")
            value = float(
                row.get(
                    "prediction_or_abstention_agreement_rate",
                    float("nan"),
                )
            )
            coverage = (
                row.get("exact_cell_coverage") is True
                and int(row.get("expected_pair_scale_cells", -1))
                == int(row.get("observed_pair_scale_cells", -2))
                == (
                    openset.DEFAULT_HELD_SCALE_REALIZATIONS_PER_PROFILE
                    * len(openset.scale_eval.SCALE_FACTORS)
                )
                and int(
                    row.get("minimum_distinct_observation_lengths", 0)
                )
                >= openset.MIN_PAIRED_DISTINCT_RUNTIME_LENGTHS
            )
            rows[str(profile)] = {
                "value": value,
                "exact_coverage": coverage,
                "passes": (
                    coverage
                    and math.isfinite(value)
                    and value
                    >= openset.CURRENT_SELECTION_CAUSAL_AGREEMENT_FLOOR
                ),
            }
    exact = (
        isinstance(expected, Mapping)
        and bool(expected)
        and observed == expected
        and causal.get("exact_eligible_cell_coverage") is True
        and int(causal.get("expected_cells", -1))
        == int(causal.get("cells", -2))
        == len(expected)
        == openset.EXPECTED_HELD_CAUSAL_PREFIX_CELLS
        and int(causal.get("expected_observation_cells", -1))
        == int(causal.get("observed_observation_cells", -2))
        == openset.EXPECTED_HELD_OBSERVATION_CELLS
        and int(causal.get("minimum_distinct_observation_lengths", 0))
        >= openset.MIN_PAIRED_DISTINCT_RUNTIME_LENGTHS
        and set(rows) == expected_profiles
        and all(row["exact_coverage"] is True for row in rows.values())
    )
    value = float(
        causal.get(
            "prediction_or_abstention_agreement_rate",
            float("nan"),
        )
    )
    if not rows:
        raise ValueError("causal-prefix profile metrics are empty")
    worst = min(rows, key=lambda key: rows[key]["value"])
    return (
        {
            "causal_prefix_outcome_agreement_across_observation_lengths": {
                "value": value,
                "floor": openset.CURRENT_SELECTION_CAUSAL_AGREEMENT_FLOOR,
                "exact_coverage": exact,
                "passes": (
                    exact
                    and math.isfinite(value)
                    and value
                    >= openset.CURRENT_SELECTION_CAUSAL_AGREEMENT_FLOOR
                ),
            },
            "causal_prefix_outcome_agreement_every_source_profile": {
                **rows[worst],
                "source_profile": worst,
                "floor": openset.CURRENT_SELECTION_CAUSAL_AGREEMENT_FLOOR,
                "all_source_profiles": rows,
                "exact_source_profile_inventory":
                    set(rows) == expected_profiles,
                "passes": exact and all(
                    row["passes"] is True for row in rows.values()
                ),
            },
        },
        exact,
    )


def _length_scale_gates(
    known: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], bool]:
    paired = known.get("paired_length_scale_invariance")
    if not isinstance(paired, Mapping):
        raise ValueError("combined length/scale metrics are missing")
    by_profile = paired.get("by_source_profile")
    expected_profiles = {
        f"current:{profile}"
        for profile in openset.corpus_data.CURRENT_PROFILES
    }
    rows: dict[str, dict[str, Any]] = {}
    if isinstance(by_profile, Mapping):
        for profile, row in sorted(by_profile.items()):
            if not isinstance(row, Mapping):
                raise ValueError("combined profile metric is invalid")
            value = float(
                row.get(
                    "prediction_or_abstention_agreement_rate",
                    float("nan"),
                )
            )
            cells = int(row.get("cells", 0))
            expected_cells = int(row.get("expected_pair_cells", 0))
            coverage = float(row.get("complete_pair_coverage_rate", 0.0))
            minimum_lengths = int(
                row.get("minimum_distinct_runtime_lengths", 0)
            )
            exact_profile = (
                cells == expected_cells
                == openset.DEFAULT_HELD_SCALE_REALIZATIONS_PER_PROFILE
                and coverage == 1.0
                and minimum_lengths
                >= openset.MIN_PAIRED_DISTINCT_RUNTIME_LENGTHS
            )
            rows[str(profile)] = {
                "value": value,
                "cells": cells,
                "expected_cells": expected_cells,
                "coverage": coverage,
                "minimum_distinct_runtime_lengths": minimum_lengths,
                "exact_coverage": exact_profile,
                "passes": (
                    exact_profile
                    and math.isfinite(value)
                    and value >= openset.LENGTH_SCALE_DECISION_AGREEMENT_FLOOR
                ),
            }
    value = float(
        paired.get(
            "prediction_or_abstention_agreement_rate",
            float("nan"),
        )
    )
    exact = (
        int(paired.get("expected_pair_cells", -1))
        == int(paired.get("cells", -2))
        == openset.EXPECTED_HELD_PAIR_CELLS
        and int(paired.get("expected_observation_cells", -1))
        == int(paired.get("observed_observation_cells", -2))
        == openset.EXPECTED_HELD_OBSERVATION_CELLS
        and float(paired.get("complete_pair_coverage_rate", 0.0)) == 1.0
        and int(paired.get("distinct_runtime_lengths_minimum", 0))
        >= openset.MIN_PAIRED_DISTINCT_RUNTIME_LENGTHS
        and set(rows) == expected_profiles
        and all(row["exact_coverage"] is True for row in rows.values())
    )
    if not rows:
        raise ValueError("combined profile metrics are empty")
    worst = min(rows, key=lambda key: rows[key]["value"])
    return (
        {
            "length_scale_outcome_agreement_pooled": {
                "value": value,
                "floor": openset.LENGTH_SCALE_DECISION_AGREEMENT_FLOOR,
                "exact_coverage": exact,
                "passes": (
                    exact
                    and math.isfinite(value)
                    and value >= openset.LENGTH_SCALE_DECISION_AGREEMENT_FLOOR
                ),
            },
            "length_scale_outcome_agreement_every_source_profile": {
                **rows[worst],
                "source_profile": worst,
                "floor": openset.LENGTH_SCALE_DECISION_AGREEMENT_FLOOR,
                "all_source_profiles": rows,
                "exact_source_profile_inventory":
                    set(rows) == expected_profiles,
                "passes": exact and all(
                    row["passes"] is True for row in rows.values()
                ),
            },
        },
        exact,
    )


def build_development_gate_summary(
    known: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the 19 precommitted known-only gates; novelty is never invented."""
    role, shared_classifier = openset._classifier_gate_summary(known)
    if role != "independent_held_current_service":
        raise ValueError("shared classifier gate helper selected wrong role")
    shared_coverage = dict(
        shared_classifier.pop("classifier_held_current_service_coverage")
    )
    profile_coverage = (
        shared_coverage.get(
            "exact_profile_legal_cell_key_and_count_coverage"
        )
        is True
        and int(shared_coverage.get("expected_profiles", -1))
        == int(shared_coverage.get("observed_profiles", -2))
        == len(openset.corpus_data.CURRENT_PROFILES)
        and int(
            shared_coverage.get(
                "expected_global_length_scale_cells", -1
            )
        )
        == int(shared_coverage.get("observed_length_scale_cells", -2))
        == (
            len(openset.RUNTIME_INPUT_LENGTHS)
            * len(openset.scale_eval.SCALE_FACTORS)
        )
        and int(
            shared_coverage.get("expected_profile_legal_cells", -1)
        )
        == int(
            shared_coverage.get("observed_profile_legal_cells", -2)
        )
        == openset.EXPECTED_HELD_PROFILE_LEGAL_CELLS
        and int(
            shared_coverage.get("expected_profile_legal_observations", -1)
        )
        == int(
            shared_coverage.get("observed_profile_legal_observations", -2)
        )
        == openset.EXPECTED_HELD_OBSERVATION_CELLS
    )
    physical_gates, physical_exact = _physical_scale_gates(known)
    causal_gates, causal_exact = _causal_prefix_gates(known)
    length_scale_gates, combined_exact = _length_scale_gates(known)
    eligible_coverage = (
        shared_coverage.get("passes") is True
        and physical_exact
        and causal_exact
        and combined_exact
    )
    renamed_classifier = {
        key.replace(
            "classifier_held_current_service_",
            "classifier_development_current_service_",
        ): value
        for key, value in shared_classifier.items()
    }
    gates: dict[str, dict[str, Any]] = {
        "exact_profile_length_scale_key_and_count_coverage": {
            **shared_coverage,
            "value": profile_coverage,
            "passes": profile_coverage,
        },
        "exact_eligible_pair_and_observation_coverage": {
            "value": eligible_coverage,
            "physical_scale_exact": physical_exact,
            "causal_prefix_exact": causal_exact,
            "combined_length_scale_exact": combined_exact,
            "passes": eligible_coverage,
        },
        **renamed_classifier,
        **_conditional_abstention_gates(known),
        **causal_gates,
        **physical_gates,
        **length_scale_gates,
    }
    if len(gates) != 19:
        raise AssertionError(
            f"development gate inventory changed: {len(gates)} != 19"
        )
    applicable = [
        row for row in gates.values() if row.get("applicable", True)
    ]
    return {
        "development_diagnostics_only": True,
        "independent_validation_evidence": False,
        "release_evidence": False,
        "novelty_gates_present": False,
        "classifier_gate_source":
            "calibrate_current_source_openset._classifier_gate_summary",
        "conditional_cell_gate_source":
            "calibrate_current_source_openset._abstention_cell_gate",
        "gates": gates,
        "all_pass": bool(
            applicable
            and all(row.get("passes") is True for row in applicable)
        ),
    }


def _assert_bound_inputs_unchanged(intent: Mapping[str, Any]) -> None:
    paths = _intent_paths(intent)
    checks = (
        (paths["protocol"], intent["protocol"]["sha256"], "protocol"),
        (
            paths["scale_corpus"] / "scale_eval.json",
            intent["scale_development_corpus"]["manifest_sha256"],
            "scale manifest",
        ),
        (
            paths["scale_corpus"] / "scale_eval.f32",
            intent["scale_development_corpus"]["raw_sha256"],
            "scale raw",
        ),
        (
            paths["current_corpus"] / "corpus.json",
            intent["current_reference_corpus"]["manifest_sha256"],
            "current manifest",
        ),
        (
            paths["current_corpus"] / "corpus.f32",
            intent["current_reference_corpus"]["raw_sha256"],
            "current raw",
        ),
        (
            paths["historical_corpus"] / "corpus.json",
            intent["candidate"]["historical_corpus"]["manifest_sha256"],
            "historical manifest",
        ),
        (
            paths["historical_corpus"] / "corpus.f32",
            intent["candidate"]["historical_corpus"]["raw_sha256"],
            "historical raw",
        ),
        (
            paths["classifier_asset"],
            intent["candidate"]["classifier_sha256"],
            "classifier",
        ),
        (paths["policy"], intent["candidate"]["policy_sha256"], "policy"),
        (
            paths["scorer"],
            intent["scorer_source"]["sha256"],
            "scorer source",
        ),
    )
    for path, digest, role in checks:
        _require_exact_hash(path, digest, role)
    generator = intent["scale_development_corpus"].get("generator")
    if not isinstance(generator, Mapping):
        raise ValueError("intent generator binding is missing")
    for key in ("source", "bundle"):
        row = generator.get(key)
        if not isinstance(row, Mapping):
            raise ValueError(f"intent generator {key} binding is invalid")
        path = _path_from_relative(
            row.get("relative_path"),
            role=f"intent generator {key}",
        )
        _require_exact_hash(
            path,
            row.get("sha256"),
            f"generator {key}",
        )
    candidate = intent["candidate"]
    fusion_metrics = candidate.get("fusion_dev_metrics")
    fusion_artifacts = candidate.get("fusion_artifacts")
    fusion_classifier_binding = candidate.get(
        "fusion_classifier_binding"
    )
    if not all(
        isinstance(row, Mapping)
        for row in (
            fusion_metrics,
            fusion_artifacts,
            fusion_classifier_binding,
        )
    ):
        raise ValueError("intent fusion binding is incomplete")
    metrics_path = _path_from_relative(
        fusion_metrics.get("relative_path"),
        role="intent fusion development metrics",
    )
    if metrics_path != paths["fusion"] / "dev_metrics.json":
        raise ValueError("intent fusion metrics path changed")
    metrics_sha = fusion_metrics.get("sha256")
    if (
        metrics_sha
        != fusion_classifier_binding.get(
            "source_fusion_dev_metrics_sha256"
        )
    ):
        raise ValueError("intent fusion metrics bindings disagree")
    _require_exact_hash(
        metrics_path,
        metrics_sha,
        "fusion development metrics",
    )
    files = fusion_artifacts.get("files_sha256")
    classifier_files = fusion_classifier_binding.get(
        "source_fusion_artifacts_sha256"
    )
    if (
        fusion_artifacts.get("directory_relative_path")
        != _relative(paths["fusion"])
        or not isinstance(files, Mapping)
        or not files
        or files != classifier_files
    ):
        raise ValueError("intent fusion artifact maps disagree")
    for name, digest in sorted(files.items()):
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
        ):
            raise ValueError("intent fusion artifact name is unsafe")
        artifact = _repo_path(
            paths["fusion"] / name,
            role=f"intent fusion artifact {name}",
            directory=False,
        )
        _require_exact_hash(
            artifact,
            digest,
            f"fusion artifact {name}",
        )
    if openset._source_hashes() != intent["executed_source_sha256"]:
        raise RuntimeError("frozen executed sources changed during scoring")


def score_once(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    expected_intent = (REPO / INTENT_RELATIVE_PATH).resolve()
    intent_path = _repo_path(
        args.intent,
        role="scale-development score intent",
        directory=False,
    )
    if intent_path != expected_intent:
        raise ValueError("score intent path differs from the frozen path")
    # Refuse a redraw before rebuilding the context or loading model weights.
    # lexists catches regular files, directories, and broken symlinks.
    _assert_path_absent(
        REPO / LEDGER_RELATIVE_PATH,
        "scale-development score ledger",
    )
    _assert_path_absent(
        REPO / REPORT_OUTPUT_RELATIVE_PATH,
        "scale-development report output",
    )
    (
        intent,
        intent_sha,
        paths,
        context,
        policy,
        protocol,
    ) = _validate_intent(intent_path)
    _assert_bound_inputs_unchanged(intent)
    ledger_payload, ledger_sha = _claim_score_ledger(
        path=paths["ledger"],
        intent=intent,
        intent_sha256=intent_sha,
    )

    # This is intentionally the first model-scoring call in this workflow.
    raw_known, _known_scores = openset.evaluate_held_scale_known(
        context,
        policy,
        paths["scale_corpus"],
        paths["current_corpus"],
        validation_protocol=_development_evaluator_protocol(
            intent, protocol
        ),
    )
    observed_separation = _canonical_reference_separation(
        raw_known.get("reference_separation", {}),
        expected_directory=paths["current_corpus"],
    )
    if (
        observed_separation != intent["reference_separation"]["report"]
        or hashlib.sha256(
            _json_bytes(observed_separation)
        ).hexdigest()
        != intent["reference_separation"]["sha256"]
    ):
        raise RuntimeError(
            "reference-separation evidence changed during model evaluation"
        )
    raw_known = dict(raw_known)
    raw_known["reference_separation"] = observed_separation
    known = _relabel_development_population(raw_known)
    gate_summary = build_development_gate_summary(known)
    _assert_bound_inputs_unchanged(intent)
    if (
        not paths["ledger"].is_file()
        or _sha256(paths["ledger"]) != ledger_sha
        or _load_json(paths["ledger"], "score ledger") != ledger_payload
    ):
        raise RuntimeError("one-shot score ledger changed during evaluation")

    report = {
        "schema": REPORT_SCHEMA,
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "complete",
        "development_only": True,
        "release_evidence": False,
        "independent_validation_evidence": False,
        "candidate_selection_allowed": False,
        "candidate_refit_allowed": False,
        "intent": {
            "relative_path": _relative(intent_path),
            "sha256": intent_sha,
        },
        "score_ledger": {
            "relative_path": _relative(paths["ledger"]),
            "sha256": ledger_sha,
            "state": ledger_payload["state"],
            "created_before_first_model_inference": True,
            "exists_at_publication": True,
            "deletion_or_replacement_invalidates_score_evidence": True,
        },
        "candidate": intent["candidate"],
        "scale_development_corpus": intent["scale_development_corpus"],
        "evaluation_contract": {
            "population":
                "precommitted seed-20262903 development scale corpus",
            "model_score_entrypoint":
                "calibrate_current_source_openset.evaluate_held_scale_known",
            "current_route_effective_input":
                "fixed causal first 4096 samples",
            "novelty_evaluation_performed": False,
            "held_seed_20262902_scored": False,
            "reserved_validation_seeds_20263002_20263003_scored": False,
            "uses_frequency_transform_at_inference": False,
        },
        "known_evaluation": known,
        "gate_summary": gate_summary,
        "executed_source_sha256": intent["executed_source_sha256"],
        "scorer_source": intent["scorer_source"],
        "bound_inputs_unchanged_before_publication": True,
    }
    output = _validate_empty_output(paths["output"])
    output.mkdir(parents=True, exist_ok=True)
    _fsync_directory(output.parent)
    report_path = output / REPORT_NAME
    report_sha = _atomic_durable_json(report_path, report)
    _assert_bound_inputs_unchanged(intent)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "development_only",
        "development_only": True,
        "release_evidence": False,
        "independent_validation_evidence": False,
        "all_pass": gate_summary["all_pass"],
        "intent_sha256": intent_sha,
        "score_ledger_sha256": ledger_sha,
        "files_sha256": {REPORT_NAME: report_sha},
        "live_assets_written": False,
    }
    _atomic_durable_json(output / MANIFEST_NAME, manifest)
    print(
        "[v4 scale development] "
        f"all-pass={gate_summary['all_pass']} "
        f"wall={time.perf_counter() - started:.1f}s",
        flush=True,
    )
    print(f"[v4 scale development] wrote {output}", flush=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser(
        "prepare-intent",
        help="validate and immutably bind inputs without scoring",
    )
    prepare.add_argument(
        "--protocol",
        default=str(REPO / PROTOCOL_RELATIVE_PATH),
    )
    prepare.add_argument(
        "--scale-corpus",
        default=str(REPO / SCALE_CORPUS_RELATIVE_PATH),
    )
    prepare.add_argument(
        "--fusion",
        default=str(REPO / FUSION_RELATIVE_PATH),
    )
    prepare.add_argument(
        "--historical-corpus",
        default=str(REPO / HISTORICAL_CORPUS_RELATIVE_PATH),
    )
    prepare.add_argument(
        "--current-corpus",
        default=str(REPO / CURRENT_CORPUS_RELATIVE_PATH),
    )
    prepare.add_argument(
        "--classifier-asset",
        default=str(REPO / CLASSIFIER_RELATIVE_PATH),
    )
    prepare.add_argument(
        "--policy",
        default=str(REPO / POLICY_RELATIVE_PATH),
    )
    prepare.add_argument(
        "--output-dir",
        default=str(REPO / REPORT_OUTPUT_RELATIVE_PATH),
    )
    prepare.add_argument("--device", default="cpu")
    prepare.add_argument(
        "--intent-output",
        default=str(REPO / INTENT_RELATIVE_PATH),
    )
    score = subparsers.add_parser(
        "score",
        help="consume the immutable intent and claim the one allowed score",
    )
    score.add_argument(
        "--intent",
        default=str(REPO / INTENT_RELATIVE_PATH),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare-intent":
        prepare_intent(args)
        return
    if args.command == "score":
        report = score_once(args)
        if report["gate_summary"]["all_pass"] is not True:
            raise SystemExit(2)
        return
    raise AssertionError(f"unknown command {args.command!r}")


if __name__ == "__main__":
    main()
