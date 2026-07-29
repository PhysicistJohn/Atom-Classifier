"""Strict, generation-free validation of the adaptive novelty source freeze.

This validator does not import the generator, instantiate an RNG, open any
training corpus, or create any file.  It reads only the six paths enumerated
by ``EXPECTED_FILE_READ_SEQUENCE`` and reads only the five Git blobs enumerated
by the validation report.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]

FREEZE_RELATIVE_PATH = (
    "training/zplane_ab/v2_full_variation/v5_scale_orbit/"
    "openset_novelty_pre_generation_freeze.json"
)
AMENDMENT_RELATIVE_PATH = (
    "training/zplane_ab/v2_full_variation/v5_scale_orbit/"
    "openset_novelty_adaptive_amendment.json"
)
AMENDMENT_VALIDATOR_RELATIVE_PATH = (
    "training/zplane_ab/v2_full_variation/v5_scale_orbit/"
    "validate_openset_novelty_adaptive_amendment.py"
)
GENERATOR_RELATIVE_PATH = (
    "training/zplane_ab/v2_full_variation/v5_scale_orbit/"
    "generate_openset_novelty_adaptive.py"
)
VALIDATOR_RELATIVE_PATH = (
    "training/zplane_ab/v2_full_variation/v5_scale_orbit/"
    "validate_openset_novelty_pre_generation_freeze.py"
)
OUTPUT_RELATIVE_PATH = (
    "training/artifacts/"
    "signallab-v5-adaptive-novelty-seeds20263005-08-r300-n16384"
)

FREEZE_PATH = REPO / FREEZE_RELATIVE_PATH
AMENDMENT_PATH = REPO / AMENDMENT_RELATIVE_PATH
AMENDMENT_VALIDATOR_PATH = REPO / AMENDMENT_VALIDATOR_RELATIVE_PATH
GENERATOR_PATH = REPO / GENERATOR_RELATIVE_PATH
VALIDATOR_PATH = REPO / VALIDATOR_RELATIVE_PATH
OUTPUT_PATH = REPO / OUTPUT_RELATIVE_PATH

ATTEMPT_2_PREREGISTRATION_COMMIT = (
    "53ae9eb2060a0f65a2ffcf607d98866bb695e65c"
)
ATTEMPT_2_SOURCE_COMMIT = (
    "e14a7f2160e6792a07a77619be7281c55c8c0704"
)
ATTEMPT_2_SOURCE_SUBJECT = "Train v5 with supervised scale pairs"
ATTEMPT_2_SOURCE_TIMESTAMP = 1785334107

GENERATOR_PARENT_COMMIT = "20a26849a1f16f6ecbd9a169f2d74289e2dd39e0"
GENERATOR_COMMIT = "35440690d17c9cc75484c462ea737981cdc77092"
GENERATOR_COMMIT_SUBJECT = "Add fail-closed v5 novelty generator"
GENERATOR_COMMIT_TIMESTAMP = 1785338809

FREEZE_RAW_SHA256 = (
    "8894bea02b07ef9e2c288279f82326da10ba4c36e5b6fc350cdbd1c809b5d148"
)
FREEZE_CANONICAL_JSON_SHA256 = (
    "1686299fedfbd58961fc090fae5fc8da0eafb92d8902f56c9d5095d6da994c73"
)
AMENDMENT_SHA256 = (
    "5622458262d51a255fa70f00f1a8baeb150500d32bb9af65db44c641cf4e1154"
)
AMENDMENT_VALIDATOR_SHA256 = (
    "4be0c364df83a55ea5e49314667a49b9deea91aadab58164651d67dfbc930850"
)
GENERATOR_SHA256 = (
    "9c547a06497669beca3421196d24ede3160286e40c5d5d51b356f456ba75f93a"
)
PYTHON_EXECUTABLE = (
    "/Applications/Xcode.app/Contents/Developer/Library/Frameworks/"
    "Python3.framework/Versions/3.9/bin/python3.9"
)
PYTHON_EXECUTABLE_SHA256 = (
    "271143990bc83af0fb2404a255038f5faafb96df1584ed7f085e5018c0f33ffb"
)

EXPECTED_TOP_LEVEL_KEYS = {
    "append_only",
    "authorization",
    "command_configuration",
    "development_only",
    "first_generation_manifest_contract",
    "generator_source",
    "inventory",
    "lineage",
    "mechanics",
    "parent_bindings",
    "release_evidence",
    "runtime",
    "schema",
    "schema_version",
    "status",
    "transitively_executed_project_generation_helpers",
}

EXPECTED_RUNTIME = {
    "numeric_contract": {
        "chirp_intermediate_complex_dtype": "complex128",
        "chirp_intermediate_real_dtype": "float64",
        "digest_and_file_dtype": "little-endian complex64",
        "noise_recurrence_dtype": "complex128",
        "output_dtype": "complex64",
        "rng_draw_dtype": "float64",
    },
    "numpy_complex128_itemsize": 16,
    "numpy_complex64_itemsize": 8,
    "numpy_default_rng": "numpy.random.default_rng",
    "numpy_default_rng_bit_generator": "PCG64",
    "numpy_float64_itemsize": 8,
    "numpy_version": "2.0.2",
    "platform_machine": "arm64",
    "platform_release": "25.5.0",
    "platform_system": "Darwin",
    "python_executable": PYTHON_EXECUTABLE,
    "python_executable_sha256": PYTHON_EXECUTABLE_SHA256,
    "python_implementation": "CPython",
    "python_version": "3.9.6",
    "system_byteorder": "little",
}

EXPECTED_WORKING_FILE_BINDINGS = (
    (FREEZE_PATH, FREEZE_RAW_SHA256),
    (AMENDMENT_PATH, AMENDMENT_SHA256),
    (AMENDMENT_VALIDATOR_PATH, AMENDMENT_VALIDATOR_SHA256),
    (GENERATOR_PATH, GENERATOR_SHA256),
)
EXPECTED_GENERATION_COMMAND = (
    PYTHON_EXECUTABLE,
    GENERATOR_RELATIVE_PATH,
    "--pre-generation-freeze",
    FREEZE_RELATIVE_PATH,
    "--output-dir",
    OUTPUT_RELATIVE_PATH,
)
EXPECTED_FILE_READ_SEQUENCE = tuple(
    path
    for _phase in ("before_git_proof", "after_git_proof")
    for path in (
        FREEZE_PATH,
        AMENDMENT_PATH,
        AMENDMENT_VALIDATOR_PATH,
        GENERATOR_PATH,
        VALIDATOR_PATH,
        Path(PYTHON_EXECUTABLE),
    )
)


class FreezeValidationError(RuntimeError):
    """Raised when any frozen source, runtime, Git, or absence proof fails."""


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    try:
        encoded = (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FreezeValidationError(
            "freeze document is not canonical-JSON encodable"
        ) from exc
    return _sha256_bytes(encoded)


def _is_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _path_exists_including_broken_symlink(path: Path) -> bool:
    return os.path.lexists(str(path))


def _runtime_from_executable_sha256(
    executable_sha256: str,
) -> dict[str, Any]:
    executable = Path(sys.executable).resolve()
    if str(executable) != PYTHON_EXECUTABLE:
        raise FreezeValidationError(
            "validator is not running in the exact frozen Python executable"
        )
    if getattr(np.random.default_rng, "__name__", None) != "default_rng":
        raise FreezeValidationError("NumPy default_rng symbol changed")
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_executable": str(executable),
        "python_executable_sha256": executable_sha256,
        "numpy_version": np.__version__,
        "numpy_default_rng": "numpy.random.default_rng",
        "numpy_default_rng_bit_generator": np.random.PCG64.__name__,
        "numpy_complex64_itemsize": np.dtype(np.complex64).itemsize,
        "numpy_complex128_itemsize": np.dtype(np.complex128).itemsize,
        "numpy_float64_itemsize": np.dtype(np.float64).itemsize,
        "system_byteorder": sys.byteorder,
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "numeric_contract": {
            "rng_draw_dtype": "float64",
            "noise_recurrence_dtype": "complex128",
            "chirp_intermediate_real_dtype": "float64",
            "chirp_intermediate_complex_dtype": "complex128",
            "output_dtype": "complex64",
            "digest_and_file_dtype": "little-endian complex64",
        },
    }


def validate_freeze_document(
    document: Mapping[str, Any],
    *,
    observed_runtime: Mapping[str, Any],
) -> None:
    """Validate every JSON field without opening any file or invoking Git."""
    if not isinstance(document, Mapping):
        raise FreezeValidationError("freeze must contain one JSON object")
    if set(document) != EXPECTED_TOP_LEVEL_KEYS:
        raise FreezeValidationError("freeze top-level key inventory changed")
    if _canonical_json_sha256(document) != FREEZE_CANONICAL_JSON_SHA256:
        raise FreezeValidationError("freeze field value or structure changed")
    if document.get("runtime") != EXPECTED_RUNTIME:
        raise FreezeValidationError("freeze runtime binding changed")
    if dict(observed_runtime) != EXPECTED_RUNTIME:
        raise FreezeValidationError("observed runtime differs from freeze")
    if document.get("generator_source") != {
        "commit": GENERATOR_COMMIT,
        "commit_subject": GENERATOR_COMMIT_SUBJECT,
        "relative_path": GENERATOR_RELATIVE_PATH,
        "sha256": GENERATOR_SHA256,
    }:
        raise FreezeValidationError("generator source binding changed")
    if document.get("parent_bindings", {}).get("attempt_2_source") != {
        "commit": ATTEMPT_2_SOURCE_COMMIT,
        "commit_subject": ATTEMPT_2_SOURCE_SUBJECT,
    }:
        raise FreezeValidationError("attempt-2 source binding changed")
    command = document.get("command_configuration")
    if not isinstance(command, Mapping) or command.get(
        "pre_generation_freeze_relative_path"
    ) != FREEZE_RELATIVE_PATH or command.get(
        "output_directory_relative_path"
    ) != OUTPUT_RELATIVE_PATH:
        raise FreezeValidationError("freeze command path binding changed")


def _capture_file_snapshot() -> tuple[bytes, dict[str, str], dict[str, Any]]:
    try:
        freeze_bytes = _read_bytes(FREEZE_PATH)
        observed_hashes = {str(FREEZE_PATH): _sha256_bytes(freeze_bytes)}
        for path, _expected in EXPECTED_WORKING_FILE_BINDINGS[1:]:
            observed_hashes[str(path)] = _sha256_bytes(_read_bytes(path))
        observed_hashes[str(VALIDATOR_PATH)] = _sha256_bytes(
            _read_bytes(VALIDATOR_PATH)
        )
        executable = Path(PYTHON_EXECUTABLE)
        executable_sha256 = _sha256_bytes(_read_bytes(executable))
    except OSError as exc:
        raise FreezeValidationError(
            "an enumerated frozen file could not be read"
        ) from exc
    observed_hashes[str(executable)] = executable_sha256

    expected_hashes = {
        str(path): expected
        for path, expected in EXPECTED_WORKING_FILE_BINDINGS
    }
    expected_hashes[PYTHON_EXECUTABLE] = PYTHON_EXECUTABLE_SHA256
    changed = sorted(
        path
        for path, expected in expected_hashes.items()
        if observed_hashes.get(path) != expected
    )
    if changed:
        raise FreezeValidationError(
            f"working file binding changed: {changed}"
        )
    runtime = _runtime_from_executable_sha256(executable_sha256)
    if runtime != EXPECTED_RUNTIME:
        raise FreezeValidationError("observed runtime differs from freeze")
    return freeze_bytes, observed_hashes, runtime


def _git(
    *args: str,
    check: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=REPO,
            check=check,
            capture_output=True,
            text=text,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FreezeValidationError(
            f"required Git proof command failed: {args!r}"
        ) from exc


def _commit_metadata(commit: str) -> tuple[str, tuple[str, ...], str, int]:
    result = _git(
        "show",
        "-s",
        "--format=%H%x00%P%x00%s%x00%ct",
        commit,
    ).stdout.rstrip("\n").split("\x00")
    if len(result) != 4:
        raise FreezeValidationError(f"invalid Git metadata for {commit}")
    observed_commit, parents_text, subject, timestamp_text = result
    parents = tuple(parents_text.split()) if parents_text else ()
    if (
        not _is_commit(observed_commit)
        or any(not _is_commit(parent) for parent in parents)
        or not subject
    ):
        raise FreezeValidationError(f"invalid Git object metadata for {commit}")
    try:
        timestamp = int(timestamp_text)
    except ValueError as exc:
        raise FreezeValidationError(
            f"invalid Git timestamp for {commit}"
        ) from exc
    return observed_commit, parents, subject, timestamp


def _require_ancestor(ancestor: str, descendant: str, label: str) -> None:
    result = _git(
        "merge-base",
        "--is-ancestor",
        ancestor,
        descendant,
        check=False,
    )
    if result.returncode != 0:
        raise FreezeValidationError(f"Git ancestry failed: {label}")


def _verify_git_proof(
    *,
    validator_source_sha256: str,
) -> dict[str, Any]:
    history = _git(
        "log",
        "--format=%H",
        "--",
        FREEZE_RELATIVE_PATH,
    ).stdout.splitlines()
    if len(history) != 1 or not _is_commit(history[0]):
        raise FreezeValidationError(
            "freeze must be committed exactly once before generation"
        )
    freeze_commit = history[0]
    if freeze_commit == GENERATOR_COMMIT:
        raise FreezeValidationError(
            "freeze commit must follow the generator source commit"
        )

    preexisting = _git(
        "cat-file",
        "-e",
        f"{GENERATOR_COMMIT}:{FREEZE_RELATIVE_PATH}",
        check=False,
    )
    if preexisting.returncode == 0:
        raise FreezeValidationError(
            "freeze unexpectedly existed in the generator source commit"
        )

    head = _git("rev-parse", "HEAD").stdout.strip()
    if not _is_commit(head):
        raise FreezeValidationError("HEAD is not an exact commit")

    attempt_2_meta = _commit_metadata(ATTEMPT_2_SOURCE_COMMIT)
    if attempt_2_meta != (
        ATTEMPT_2_SOURCE_COMMIT,
        (ATTEMPT_2_PREREGISTRATION_COMMIT,),
        ATTEMPT_2_SOURCE_SUBJECT,
        ATTEMPT_2_SOURCE_TIMESTAMP,
    ):
        raise FreezeValidationError("attempt-2 source commit object changed")
    generator_meta = _commit_metadata(GENERATOR_COMMIT)
    if generator_meta != (
        GENERATOR_COMMIT,
        (GENERATOR_PARENT_COMMIT,),
        GENERATOR_COMMIT_SUBJECT,
        GENERATOR_COMMIT_TIMESTAMP,
    ):
        raise FreezeValidationError("generator source commit object changed")
    freeze_meta = _commit_metadata(freeze_commit)
    if (
        freeze_meta[0] != freeze_commit
        or not freeze_meta[1]
        or freeze_meta[3] <= GENERATOR_COMMIT_TIMESTAMP
    ):
        raise FreezeValidationError(
            "freeze commit chronology or parent binding is invalid"
        )
    if not (
        ATTEMPT_2_SOURCE_TIMESTAMP
        < GENERATOR_COMMIT_TIMESTAMP
        < freeze_meta[3]
    ):
        raise FreezeValidationError("source/freeze commit chronology changed")

    expected_blobs = (
        (GENERATOR_COMMIT, AMENDMENT_RELATIVE_PATH, AMENDMENT_SHA256),
        (
            GENERATOR_COMMIT,
            AMENDMENT_VALIDATOR_RELATIVE_PATH,
            AMENDMENT_VALIDATOR_SHA256,
        ),
        (GENERATOR_COMMIT, GENERATOR_RELATIVE_PATH, GENERATOR_SHA256),
        (freeze_commit, FREEZE_RELATIVE_PATH, FREEZE_RAW_SHA256),
        (freeze_commit, VALIDATOR_RELATIVE_PATH, validator_source_sha256),
    )
    git_blob_report = []
    for commit, relative_path, expected_sha256 in expected_blobs:
        content = _git(
            "show",
            f"{commit}:{relative_path}",
            text=False,
        ).stdout
        if _sha256_bytes(content) != expected_sha256:
            raise FreezeValidationError(
                f"committed blob changed: {commit}:{relative_path}"
            )
        git_blob_report.append(
            {
                "commit": commit,
                "relative_path": relative_path,
                "sha256": expected_sha256,
            }
        )

    _require_ancestor(
        ATTEMPT_2_SOURCE_COMMIT,
        GENERATOR_COMMIT,
        "attempt-2 source -> generator source",
    )
    _require_ancestor(
        ATTEMPT_2_SOURCE_COMMIT,
        freeze_commit,
        "attempt-2 source -> freeze",
    )
    _require_ancestor(
        GENERATOR_COMMIT,
        freeze_commit,
        "generator source -> freeze",
    )
    _require_ancestor(
        freeze_commit,
        head,
        "committed freeze -> current HEAD",
    )

    tracked_output = _git(
        "ls-tree",
        "-r",
        "--name-only",
        freeze_commit,
        "--",
        OUTPUT_RELATIVE_PATH,
    ).stdout.splitlines()
    if tracked_output:
        raise FreezeValidationError(
            "adaptive novelty output was already tracked at freeze commit"
        )
    output_history = _git(
        "log",
        "--all",
        "--format=%H",
        "--",
        OUTPUT_RELATIVE_PATH,
    ).stdout.splitlines()
    if output_history:
        raise FreezeValidationError(
            "adaptive novelty output has pre-generation Git history"
        )

    return {
        "attempt_2_source_commit": ATTEMPT_2_SOURCE_COMMIT,
        "generator_source_commit": GENERATOR_COMMIT,
        "freeze_commit": freeze_commit,
        "validated_head": head,
        "chronology": {
            "attempt_2_source_timestamp": ATTEMPT_2_SOURCE_TIMESTAMP,
            "generator_source_timestamp": GENERATOR_COMMIT_TIMESTAMP,
            "freeze_timestamp": freeze_meta[3],
            "strictly_increasing": True,
        },
        "ancestry": {
            "attempt_2_to_generator": True,
            "attempt_2_to_freeze": True,
            "generator_to_freeze": True,
            "freeze_to_validated_head": True,
        },
        "git_blobs": git_blob_report,
        "output_absent_from_freeze_tree_and_history": True,
    }


def validate_repository_state() -> dict[str, Any]:
    """Validate the committed freeze and return a read-only proof report."""
    if _path_exists_including_broken_symlink(OUTPUT_PATH):
        raise FreezeValidationError(
            "adaptive novelty output exists before authorized generation"
        )
    first_bytes, first_hashes, runtime = _capture_file_snapshot()
    try:
        document = json.loads(first_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FreezeValidationError("freeze is not strict UTF-8 JSON") from exc
    validate_freeze_document(document, observed_runtime=runtime)

    git_proof = _verify_git_proof(
        validator_source_sha256=first_hashes[str(VALIDATOR_PATH)],
    )

    second_bytes, second_hashes, second_runtime = _capture_file_snapshot()
    if (
        second_bytes != first_bytes
        or second_hashes != first_hashes
        or second_runtime != runtime
    ):
        raise FreezeValidationError(
            "frozen files or runtime changed during validation"
        )
    if _path_exists_including_broken_symlink(OUTPUT_PATH):
        raise FreezeValidationError(
            "adaptive novelty output appeared during validation"
        )

    return {
        "schema": "atomos.v5.scale-orbit.novelty-pre-generation-validation",
        "status": "pass",
        "generation_performed": False,
        "rng_instantiated": False,
        "files_created_or_modified": 0,
        "freeze_relative_path": FREEZE_RELATIVE_PATH,
        "freeze_raw_sha256": FREEZE_RAW_SHA256,
        "freeze_canonical_json_sha256": FREEZE_CANONICAL_JSON_SHA256,
        "output_relative_path": OUTPUT_RELATIVE_PATH,
        "output_absent_before_and_after_validation": True,
        "exact_generation_command": list(EXPECTED_GENERATION_COMMAND),
        "file_reads": [
            str(path.relative_to(REPO))
            if path.is_relative_to(REPO)
            else str(path)
            for path in EXPECTED_FILE_READ_SEQUENCE
        ],
        "git_proof": git_proof,
    }


def main(argv: Sequence[str] | None = None) -> None:
    if argv:
        raise FreezeValidationError("validator accepts no arguments")
    print(json.dumps(validate_repository_state(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1:])
