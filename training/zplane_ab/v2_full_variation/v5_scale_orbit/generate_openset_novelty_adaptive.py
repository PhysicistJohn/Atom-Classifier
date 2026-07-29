"""Deterministic, fail-closed v5 adaptive novelty generation.

This module implements the generator mechanics frozen by
``openset_novelty_adaptive_amendment.json``.  The review helpers are pure,
in-memory, and deliberately limited to tiny arrays.  The production inventory
can only be materialized by :func:`execute_frozen_generation`, which first
requires a later, committed pre-generation freeze binding the final source,
runtime, command, output location, and exact inventory.

Importing this module never generates rows or creates files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
GENERATOR_PATH = Path(__file__).resolve()
GENERATOR_RELATIVE_PATH = str(GENERATOR_PATH.relative_to(REPO))
AMENDMENT_PATH = HERE / "openset_novelty_adaptive_amendment.json"
AMENDMENT_RELATIVE_PATH = str(AMENDMENT_PATH.relative_to(REPO))
AMENDMENT_SHA256 = (
    "5622458262d51a255fa70f00f1a8baeb150500d32bb9af65db44c641cf4e1154"
)
AMENDMENT_VALIDATOR_PATH = (
    HERE / "validate_openset_novelty_adaptive_amendment.py"
)
AMENDMENT_VALIDATOR_RELATIVE_PATH = str(
    AMENDMENT_VALIDATOR_PATH.relative_to(REPO)
)
AMENDMENT_VALIDATOR_SHA256 = (
    "4be0c364df83a55ea5e49314667a49b9deea91aadab58164651d67dfbc930850"
)
ATTEMPT_2_PREREGISTRATION_COMMIT = (
    "53ae9eb2060a0f65a2ffcf607d98866bb695e65c"
)
ATTEMPT_2_SOURCE_COMMIT = (
    "e14a7f2160e6792a07a77619be7281c55c8c0704"
)
ATTEMPT_2_SOURCE_SUBJECT = "Train v5 with supervised scale pairs"

ADAPTIVE_SEEDS = (20263005, 20263006, 20263007, 20263008)
NOVELTY_FAMILIES = ("no_signal", "noise", "chirp")
PROTOTYPE_SOURCE_ROUTES = ("historical", "current")
OBSERVATION_LENGTHS = (4096, 8192, 16384)
MAXIMUM_LENGTH = 16384
ROWS_PER_FAMILY_PER_SEED = 300
LONGEST_ROWS_TOTAL = 3600
EXACT_CELL_COUNT = 72
ROWS_PER_CELL = 300
EXACT_ROUTE_SCORE_COUNT = 21600
OUTPUT_DTYPE_NAME = "complex64"
CANONICAL_COMPLEX_DTYPE = np.dtype("<c8")
RNG_NAME = "numpy.random.default_rng"
RNG_BIT_GENERATOR = "PCG64"
FAMILY_SEED_MESSAGE = "atomos-v4-openset:{base_seed}:{family}"
DIGEST_MANIFEST_FILENAME = "novelty-digest-manifest.json"
PRE_GENERATION_FREEZE_SCHEMA = (
    "atomos.v5.scale-orbit.novelty-pre-generation-source-freeze"
)
PRE_GENERATION_FREEZE_STATUS = (
    "committed_outcome_blind_before_first_adaptive_novelty_generation"
)

_REVIEW_MAX_ROWS = 8
_REVIEW_MAX_LENGTH = 512


class GenerationRefused(RuntimeError):
    """Raised before any production row or output is created."""


class GenerationAuthorization:
    """Opaque base type; only the closure-held mint can instantiate it."""

    __slots__ = ()

    def __new__(cls, *args: Any, **kwargs: Any):
        if cls is GenerationAuthorization:
            raise TypeError(
                "GenerationAuthorization is opaque and cannot be constructed"
            )
        return super().__new__(cls)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return value


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, not {type(value).__name__}")
    return int(value)


def _positive_int(value: Any, name: str) -> int:
    result = _strict_int(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _validate_family(family: Any) -> str:
    if not isinstance(family, str) or family not in NOVELTY_FAMILIES:
        raise ValueError(
            f"family must be exactly one of {list(NOVELTY_FAMILIES)}"
        )
    return family


def derive_family_seed(base_seed: Any, family: Any) -> int:
    """Derive the frozen unsigned 64-bit family seed."""
    seed = _strict_int(base_seed, "base_seed")
    name = _validate_family(family)
    message = FAMILY_SEED_MESSAGE.format(
        base_seed=seed,
        family=name,
    ).encode("utf-8")
    digest = hashlib.sha256(message).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _create_generation_boundary():
    """Create a closure-held capability around all production row mechanics.

    The raw mechanics function and capability object never enter the module
    namespace.  The only capability mint is called after Git/source/freeze
    authorization, while the public pure entrypoint is hard-limited to tiny
    review arrays.
    """
    capability = object()
    mint_sentinel = object()
    revalidation_phases: dict[object, set[str]] = {}

    class Authorized(GenerationAuthorization):
        __slots__ = (
            "freeze_path",
            "freeze_relative_path",
            "freeze_sha256",
            "freeze_commit",
            "source_commit",
            "generator_sha256",
            "output_directory",
            "output_relative_path",
            "runtime",
            "document",
            "_capability",
            "_sealed",
        )

        def __init__(
            self,
            provided_mint_sentinel: object,
            /,
            **values: Any,
        ) -> None:
            if provided_mint_sentinel is not mint_sentinel:
                raise TypeError(
                    "authorization construction requires the closure-held "
                    "mint sentinel"
                )
            expected = set(self.__slots__) - {"_capability", "_sealed"}
            if set(values) != expected:
                raise TypeError("authorization mint field inventory changed")
            for key, value in values.items():
                object.__setattr__(
                    self,
                    key,
                    (
                        _deep_freeze(value)
                        if key in {"runtime", "document"}
                        else value
                    ),
                )
            object.__setattr__(self, "_capability", capability)
            object.__setattr__(self, "_sealed", True)
            revalidation_phases[self] = set()

        def __setattr__(self, name: str, value: Any) -> None:
            if getattr(self, "_sealed", False):
                raise AttributeError("generation authorization is immutable")
            object.__setattr__(self, name, value)

    def noise_row(length: int, rng: np.random.Generator) -> np.ndarray:
        innovations = (
            rng.standard_normal(length) + 1j * rng.standard_normal(length)
        ) / math.sqrt(2.0)
        rho = float(rng.uniform(0.0, 0.97))
        carrier = float(rng.uniform(-math.pi, math.pi))
        coefficient = rho * np.exp(1j * carrier)
        innovation_scale = math.sqrt(max(1.0 - rho * rho, 1e-12))
        output = np.empty(length, dtype=np.complex128)
        output[0] = innovations[0]
        for index in range(1, length):
            output[index] = (
                coefficient * output[index - 1]
                + innovation_scale * innovations[index]
            )
        output *= 10.0 ** float(rng.uniform(-2.0, 2.0))
        output *= np.exp(1j * float(rng.uniform(-math.pi, math.pi)))
        return np.ascontiguousarray(output, dtype=np.complex64)

    def chirp_row(length: int, rng: np.random.Generator) -> np.ndarray:
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
        return np.ascontiguousarray(output, dtype=np.complex64)

    def mechanics(
        base_seed: int,
        family: str,
        *,
        row_count: int,
        length: int,
    ) -> np.ndarray:
        derived_seed = derive_family_seed(base_seed, family)
        if family == "no_signal":
            return np.zeros((row_count, length), dtype=np.complex64)
        rng = np.random.default_rng(derived_seed)
        row_generator = noise_row if family == "noise" else chirp_row
        result = np.stack(
            [row_generator(length, rng) for _ in range(row_count)],
            axis=0,
        )
        if (
            result.shape != (row_count, length)
            or result.dtype != np.dtype(np.complex64)
            or not result.flags.c_contiguous
            or not np.isfinite(result.real).all()
            or not np.isfinite(result.imag).all()
        ):
            raise RuntimeError("generator returned an invalid complex64 matrix")
        return result

    def review(
        base_seed: Any,
        family: Any,
        *,
        row_count: Any,
        length: Any,
    ) -> np.ndarray:
        seed = _strict_int(base_seed, "base_seed")
        name = _validate_family(family)
        rows = _positive_int(row_count, "row_count")
        size = _positive_int(length, "length")
        if rows > _REVIEW_MAX_ROWS or size > _REVIEW_MAX_LENGTH:
            raise GenerationRefused(
                "review generation is limited to at most "
                f"{_REVIEW_MAX_ROWS} rows and length {_REVIEW_MAX_LENGTH}; "
                "production generation requires a committed pre-generation "
                "freeze"
            )
        return mechanics(
            seed,
            name,
            row_count=rows,
            length=size,
        )

    def authorize(
        *,
        pre_generation_freeze: str | Path,
        output_directory: str | Path,
    ) -> GenerationAuthorization:
        values = _verify_generation_authorization_inputs(
            pre_generation_freeze=pre_generation_freeze,
            output_directory=output_directory,
        )
        return Authorized(mint_sentinel, **values)

    def assert_authorized(
        authorization: GenerationAuthorization,
    ) -> None:
        if (
            not isinstance(authorization, Authorized)
            or getattr(authorization, "_capability", None) is not capability
        ):
            raise GenerationRefused(
                "production generation requires an internally minted "
                "authorization capability"
            )

    def production(
        authorization: GenerationAuthorization,
        *,
        base_seed: Any,
        family: Any,
    ) -> np.ndarray:
        assert_authorized(authorization)
        seed = _strict_int(base_seed, "base_seed")
        name = _validate_family(family)
        if seed not in ADAPTIVE_SEEDS:
            raise GenerationRefused(
                "authorized production seed is outside the exact joint "
                "four-seed inventory"
            )
        return mechanics(
            seed,
            name,
            row_count=ROWS_PER_FAMILY_PER_SEED,
            length=MAXIMUM_LENGTH,
        )

    def revalidate(
        authorization: GenerationAuthorization,
        *,
        phase: str,
    ) -> None:
        """Verify every retained binding, then record that exact phase."""
        assert_authorized(authorization)
        if phase not in {
            "immediately_before_generation",
            "immediately_after_generation",
            "immediately_before_atomic_install",
        }:
            raise GenerationRefused("unknown authorization revalidation phase")
        _verify_authorized_revalidation_bindings(
            authorization,
            phase=phase,
        )
        revalidation_phases[authorization].add(phase)

    def require_generation_revalidations(
        authorization: GenerationAuthorization,
    ) -> None:
        assert_authorized(authorization)
        required = {
            "immediately_before_generation",
            "immediately_after_generation",
        }
        if not required.issubset(revalidation_phases[authorization]):
            raise GenerationRefused(
                "digest manifest requires successful before-and-after "
                "authorization revalidation"
            )

    return (
        review,
        authorize,
        assert_authorized,
        production,
        revalidate,
        require_generation_revalidations,
    )


(
    generate_family_rows_for_review,
    authorize_generation,
    _assert_generation_authorized,
    _generate_authorized_family_rows,
    _revalidate_authorized_bindings,
    _require_generation_revalidations,
) = _create_generation_boundary()
del _create_generation_boundary


def canonical_complex64_bytes(rows: np.ndarray) -> bytes:
    """Return canonical row-major little-endian complex64 bytes."""
    value = np.asarray(rows)
    if value.ndim != 2 or value.shape[0] <= 0 or value.shape[1] <= 0:
        raise ValueError("rows must be a nonempty rank-2 matrix")
    if (
        not np.issubdtype(value.dtype, np.complexfloating)
        or not np.isfinite(value.real).all()
        or not np.isfinite(value.imag).all()
    ):
        raise ValueError("rows must contain finite complex values")
    canonical = np.ascontiguousarray(value, dtype=CANONICAL_COMPLEX_DTYPE)
    return canonical.view(np.uint8).tobytes(order="C")


def complex64_sha256(rows: np.ndarray) -> str:
    """Digest canonical complex64 rows in row-major order."""
    return hashlib.sha256(canonical_complex64_bytes(rows)).hexdigest()


def causal_prefix_complex64_sha256(
    longest_rows: np.ndarray,
    observation_length: Any,
) -> str:
    """Digest each row's exact leading causal prefix, then concatenate rows."""
    value = np.asarray(longest_rows)
    length = _positive_int(observation_length, "observation_length")
    if value.ndim != 2 or length > value.shape[1]:
        raise ValueError("observation_length exceeds the longest row")
    return complex64_sha256(value[:, :length])


def family_digest_record(
    longest_rows: np.ndarray,
    *,
    base_seed: Any,
    family: Any,
    observation_lengths: Sequence[int],
) -> dict[str, Any]:
    """Build deterministic longest-row and causal-prefix digest metadata."""
    seed = _strict_int(base_seed, "base_seed")
    name = _validate_family(family)
    value = np.asarray(longest_rows)
    if value.ndim != 2:
        raise ValueError("longest_rows must be rank 2")
    lengths = tuple(_positive_int(item, "observation_length")
                    for item in observation_lengths)
    if not lengths or tuple(sorted(set(lengths))) != lengths:
        raise ValueError("observation_lengths must be unique and increasing")
    if lengths[-1] != value.shape[1]:
        raise ValueError("last observation length must equal row width")
    prefixes = {
        str(length): causal_prefix_complex64_sha256(value, length)
        for length in lengths
    }
    return {
        "base_seed": seed,
        "family": name,
        "derived_seed": derive_family_seed(seed, name),
        "row_count": int(value.shape[0]),
        "generated_once_at_maximum_length": int(value.shape[1]),
        "dtype": OUTPUT_DTYPE_NAME,
        "canonical_byte_order": "little_endian_complex64_interleaved_real_imag",
        "longest_complex64_sha256": complex64_sha256(value),
        "prefix_complex64_sha256": prefixes,
        "shorter_observation_rule":
            "exact leading causal raw prefix of the same longest row",
        "uses_frequency_transform": False,
    }


def exact_cell_inventory() -> list[dict[str, Any]]:
    """Return the exact frozen 72-cell scoring inventory."""
    cells = [
        {
            "seed": seed,
            "prototype_source_route": route,
            "observation_length": length,
            "novelty_family": family,
            "rows": ROWS_PER_CELL,
        }
        for seed in ADAPTIVE_SEEDS
        for route in PROTOTYPE_SOURCE_ROUTES
        for length in OBSERVATION_LENGTHS
        for family in NOVELTY_FAMILIES
    ]
    if len(cells) != EXACT_CELL_COUNT:
        raise AssertionError("frozen novelty cell inventory is inconsistent")
    return cells


def frozen_inventory_contract() -> dict[str, Any]:
    """Return exact immutable production inventory/configuration values."""
    return {
        "seed_registry_allocation": "adaptive_novelty_development",
        "seeds": list(ADAPTIVE_SEEDS),
        "all_four_seeds_consumed_and_reported_jointly": True,
        "seed_cherry_pick_or_omission_allowed": False,
        "families": list(NOVELTY_FAMILIES),
        "rows_per_family_per_seed": ROWS_PER_FAMILY_PER_SEED,
        "generated_once_at_maximum_length": MAXIMUM_LENGTH,
        "dtype": OUTPUT_DTYPE_NAME,
        "canonical_byte_order":
            "little_endian_complex64_interleaved_real_imag",
        "observation_lengths": list(OBSERVATION_LENGTHS),
        "shorter_observation_rule":
            "exact leading causal raw prefix of the same generated N16384 row",
        "longest_rows_total": LONGEST_ROWS_TOTAL,
        "routes": list(PROTOTYPE_SOURCE_ROUTES),
        "exact_cell_key_fields": [
            "seed",
            "prototype_source_route",
            "observation_length",
            "novelty_family",
        ],
        "exact_cell_count": EXACT_CELL_COUNT,
        "rows_per_cell": ROWS_PER_CELL,
        "exact_route_score_count": EXACT_ROUTE_SCORE_COUNT,
        "rng": RNG_NAME,
        "rng_bit_generator": RNG_BIT_GENERATOR,
        "family_seed_derivation": {
            "message_utf8": FAMILY_SEED_MESSAGE,
            "digest": "SHA-256",
            "derived_seed":
                "first 8 digest bytes interpreted as little-endian unsigned integer",
        },
        "uses_frequency_transform": False,
    }


def generator_mechanics_contract() -> dict[str, Any]:
    """Return the exact mechanics implemented by this source."""
    return {
        "no_signal": {
            "generator": "exact complex64 zeros",
            "rng_draws_per_row": 0,
        },
        "noise": {
            "generator": "direct time-domain complex AR(1)",
            "innovations":
                "(standard_normal + 1j * standard_normal) / sqrt(2)",
            "rho": "uniform [0.0, 0.97)",
            "carrier_radians": "uniform [-pi, pi)",
            "coefficient": "rho * exp(1j * carrier)",
            "innovation_scale": "sqrt(max(1 - rho^2, 1e-12))",
            "initial_value": "first innovation",
            "amplitude_multiplier": "10 ** uniform(-2.0, 2.0)",
            "global_phase_multiplier": "exp(1j * uniform(-pi, pi))",
            "output_dtype": OUTPUT_DTYPE_NAME,
        },
        "chirp": {
            "generator": "direct time-domain linear-FM exponential",
            "normalized_time": "sample_index / max(length - 1, 1)",
            "start_cycles_per_sample": "uniform [-0.42, 0.42)",
            "sweep_magnitude": "uniform [0.08, 0.80)",
            "sweep_sign": "negative iff rng.random() < 0.5",
            "phase":
                "2*pi*(start*sample_index + 0.5*sweep*sample_index*normalized_time)",
            "envelope_rate": "integers(1, 7)",
            "envelope_phase": "uniform [-pi, pi)",
            "envelope":
                "0.65 + 0.35*cos(2*pi*envelope_rate*normalized_time + envelope_phase)",
            "amplitude_multiplier": "10 ** uniform(-2.0, 2.0)",
            "global_phase_multiplier": "exp(1j * uniform(-pi, pi))",
            "output_dtype": OUTPUT_DTYPE_NAME,
        },
    }


def runtime_fingerprint() -> dict[str, Any]:
    """Return the runtime values a later freeze must bind exactly."""
    executable = Path(sys.executable).resolve()
    probe = np.random.default_rng(0)
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_executable": str(executable),
        "python_executable_sha256": _sha256_file(executable),
        "numpy_version": np.__version__,
        "numpy_default_rng": RNG_NAME,
        "numpy_default_rng_bit_generator":
            type(probe.bit_generator).__name__,
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
            "output_dtype": OUTPUT_DTYPE_NAME,
            "digest_and_file_dtype": "little-endian complex64",
        },
    }


def _repo_relative_existing_or_future(path: Path, name: str) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO.resolve()))
    except ValueError as exc:
        raise GenerationRefused(f"{name} must be inside the repository") from exc


def required_pre_generation_freeze_inputs(
    *,
    freeze_path: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Describe every value the later authorization document must provide.

    This is a requirements report, not an executable authorization.
    """
    freeze_relative = _repo_relative_existing_or_future(
        Path(freeze_path), "pre-generation freeze"
    )
    output_relative = _repo_relative_existing_or_future(
        Path(output_directory), "output directory"
    )
    return {
        "schema": PRE_GENERATION_FREEZE_SCHEMA,
        "status": PRE_GENERATION_FREEZE_STATUS,
        "must_be_append_only_committed_before_generation": True,
        "must_bind_exact_raw_sha256": [
            AMENDMENT_RELATIVE_PATH,
            AMENDMENT_VALIDATOR_RELATIVE_PATH,
            GENERATOR_RELATIVE_PATH,
            "the later pre-generation freeze itself",
        ],
        "attempt_2_source_commit": ATTEMPT_2_SOURCE_COMMIT,
        "must_prove_attempt_2_source_is_generator_source_ancestor": True,
        "must_prove_attempt_2_source_is_freeze_commit_ancestor": True,
        "must_bind_generator_source_commit_and_subject": True,
        "must_prove_generator_source_commit_is_freeze_commit_ancestor": True,
        "transitively_executed_project_generation_helpers": [],
        "pre_generation_freeze_relative_path": freeze_relative,
        "output_directory_relative_path": output_relative,
        "output_directory_must_not_exist": True,
        "runtime": runtime_fingerprint(),
        "inventory": frozen_inventory_contract(),
        "mechanics": generator_mechanics_contract(),
        "command_configuration": {
            "entrypoint_relative_path": GENERATOR_RELATIVE_PATH,
            "pre_generation_freeze_relative_path": freeze_relative,
            "output_directory_relative_path": output_relative,
            "device": "cpu",
            "embedding_batch_size": 256,
            "generation_has_no_device_or_batch_parameter": True,
            "no_seed_family_row_count_length_or_dtype_override": True,
        },
        "authorization_assertions_required": {
            "owner_authorized_first_adaptive_generation": True,
            "final_generator_source_reviewed": True,
            "generation_started_before_freeze": False,
            "adaptive_novelty_scoring_started_before_freeze": False,
            "sealed_novelty_generated_read_or_scored": False,
            "development_only": True,
            "release_evidence": False,
        },
    }


def _expected_freeze_keys() -> set[str]:
    return {
        "schema",
        "schema_version",
        "status",
        "lineage",
        "development_only",
        "release_evidence",
        "append_only",
        "authorization",
        "parent_bindings",
        "generator_source",
        "transitively_executed_project_generation_helpers",
        "runtime",
        "inventory",
        "mechanics",
        "command_configuration",
        "first_generation_manifest_contract",
    }


def validate_pre_generation_freeze_document(
    document: Mapping[str, Any],
    *,
    freeze_relative_path: str,
    output_relative_path: str,
    observed_generator_sha256: str,
    observed_runtime: Mapping[str, Any],
) -> str:
    """Validate a loaded later freeze without opening or generating rows.

    Returns the bound generator source commit when every exact field matches.
    """
    if set(document) != _expected_freeze_keys():
        raise GenerationRefused("pre-generation freeze key inventory changed")
    expected_header = {
        "schema": PRE_GENERATION_FREEZE_SCHEMA,
        "schema_version": 1,
        "status": PRE_GENERATION_FREEZE_STATUS,
        "lineage": "v5-scale-orbit",
        "development_only": True,
        "release_evidence": False,
        "append_only": True,
    }
    for key, expected in expected_header.items():
        if document.get(key) != expected:
            raise GenerationRefused(
                f"pre-generation freeze {key} must be {expected!r}"
            )
    expected_authorization = {
        "owner_authorized_first_adaptive_generation": True,
        "final_generator_source_reviewed": True,
        "generation_started_before_freeze": False,
        "adaptive_novelty_scoring_started_before_freeze": False,
        "sealed_novelty_generated_read_or_scored": False,
    }
    if document.get("authorization") != expected_authorization:
        raise GenerationRefused("pre-generation authorization changed")
    expected_parents = {
        "attempt_2_source": {
            "commit": ATTEMPT_2_SOURCE_COMMIT,
            "commit_subject": ATTEMPT_2_SOURCE_SUBJECT,
        },
        "adaptive_novelty_amendment": {
            "relative_path": AMENDMENT_RELATIVE_PATH,
            "sha256": AMENDMENT_SHA256,
        },
        "adaptive_novelty_amendment_validator": {
            "relative_path": AMENDMENT_VALIDATOR_RELATIVE_PATH,
            "sha256": AMENDMENT_VALIDATOR_SHA256,
        },
    }
    if document.get("parent_bindings") != expected_parents:
        raise GenerationRefused("pre-generation parent bindings changed")
    source = document.get("generator_source")
    if not isinstance(source, Mapping):
        raise GenerationRefused("pre-generation generator source is missing")
    source_commit = source.get("commit")
    if (
        set(source)
        != {"relative_path", "sha256", "commit", "commit_subject"}
        or source.get("relative_path") != GENERATOR_RELATIVE_PATH
        or source.get("sha256") != observed_generator_sha256
        or not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef"
               for character in source_commit)
        or not isinstance(source.get("commit_subject"), str)
        or not source.get("commit_subject")
    ):
        raise GenerationRefused("pre-generation generator source binding changed")
    if document.get(
        "transitively_executed_project_generation_helpers"
    ) != []:
        raise GenerationRefused(
            "generator must remain self-contained or bind every project helper"
        )
    if document.get("runtime") != observed_runtime:
        raise GenerationRefused("pre-generation runtime binding changed")
    if document.get("inventory") != frozen_inventory_contract():
        raise GenerationRefused("pre-generation inventory binding changed")
    if document.get("mechanics") != generator_mechanics_contract():
        raise GenerationRefused("pre-generation mechanics binding changed")
    expected_command = {
        "entrypoint_relative_path": GENERATOR_RELATIVE_PATH,
        "pre_generation_freeze_relative_path": freeze_relative_path,
        "output_directory_relative_path": output_relative_path,
        "device": "cpu",
        "embedding_batch_size": 256,
        "generation_has_no_device_or_batch_parameter": True,
        "no_seed_family_row_count_length_or_dtype_override": True,
    }
    if document.get("command_configuration") != expected_command:
        raise GenerationRefused(
            "pre-generation command or configuration changed"
        )
    expected_manifest = {
        "filename": DIGEST_MANIFEST_FILENAME,
        "required_before_scoring": True,
        "immutable_after_first_generation": True,
        "binds_pre_generation_source_freeze_sha256": True,
        "binds_each_family_seed_longest_complex64_sha256": True,
        "binds_each_family_seed_length_prefix_complex64_sha256": True,
        "binds_final_generator_source_and_runtime_environment": True,
        "deterministic_regeneration_allowed_only_when_every_digest_matches":
            True,
    }
    if document.get("first_generation_manifest_contract") != expected_manifest:
        raise GenerationRefused("first-generation manifest contract changed")
    return source_commit


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise GenerationRefused(f"{path} must contain one JSON object")
    return value


def _git(
    *args: str,
    check: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=REPO,
        check=check,
        capture_output=True,
        text=text,
    )


def _verify_generation_authorization_inputs(
    *,
    pre_generation_freeze: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Validate a later committed freeze before any production RNG call."""
    freeze_path = Path(pre_generation_freeze).resolve()
    output_path = Path(output_directory).resolve()
    freeze_relative = _repo_relative_existing_or_future(
        freeze_path, "pre-generation freeze"
    )
    output_relative = _repo_relative_existing_or_future(
        output_path, "output directory"
    )
    if not freeze_path.is_file():
        raise GenerationRefused(
            "adaptive novelty generation is forbidden until the later "
            "pre-generation freeze exists"
        )
    if output_path.exists():
        raise GenerationRefused(
            "first-generation output directory already exists; redraw and "
            "overwrite are forbidden"
        )
    if _sha256_file(AMENDMENT_PATH) != AMENDMENT_SHA256:
        raise GenerationRefused("adaptive novelty amendment bytes changed")
    if (
        _sha256_file(AMENDMENT_VALIDATOR_PATH)
        != AMENDMENT_VALIDATOR_SHA256
    ):
        raise GenerationRefused(
            "adaptive novelty amendment validator bytes changed"
        )
    generator_sha256 = _sha256_file(GENERATOR_PATH)
    runtime = runtime_fingerprint()
    if runtime.get("system_byteorder") != "little":
        raise GenerationRefused(
            "production generation requires the frozen little-endian byte order"
        )
    if runtime.get("numpy_default_rng_bit_generator") != RNG_BIT_GENERATOR:
        raise GenerationRefused("NumPy default_rng is not PCG64")
    document = _read_json_object(freeze_path)
    source_commit = validate_pre_generation_freeze_document(
        document,
        freeze_relative_path=freeze_relative,
        output_relative_path=output_relative,
        observed_generator_sha256=generator_sha256,
        observed_runtime=runtime,
    )
    freeze_sha256 = _sha256_file(freeze_path)

    freeze_commit = _git(
        "log", "-1", "--format=%H", "--", freeze_relative
    ).stdout.strip()
    if not freeze_commit:
        raise GenerationRefused(
            "pre-generation freeze must be committed before generation"
        )
    committed_freeze = _git(
        "show", f"{freeze_commit}:{freeze_relative}", text=False
    ).stdout
    if hashlib.sha256(committed_freeze).hexdigest() != freeze_sha256:
        raise GenerationRefused(
            "working pre-generation freeze differs from its committed bytes"
        )
    attempt_2_meta = _git(
        "show",
        "-s",
        "--format=%H%n%P%n%s",
        ATTEMPT_2_SOURCE_COMMIT,
    ).stdout.splitlines()
    if attempt_2_meta != [
        ATTEMPT_2_SOURCE_COMMIT,
        ATTEMPT_2_PREREGISTRATION_COMMIT,
        ATTEMPT_2_SOURCE_SUBJECT,
    ]:
        raise GenerationRefused(
            "bound attempt-2 source commit object or direct parent changed"
        )
    source_meta = _git(
        "show", "-s", "--format=%H%n%s", source_commit
    ).stdout.splitlines()
    if source_meta != [
        source_commit,
        document["generator_source"]["commit_subject"],
    ]:
        raise GenerationRefused("bound generator source commit object changed")
    committed_generator = _git(
        "show", f"{source_commit}:{GENERATOR_RELATIVE_PATH}", text=False
    ).stdout
    if hashlib.sha256(committed_generator).hexdigest() != generator_sha256:
        raise GenerationRefused(
            "working generator differs from the bound committed source"
        )
    ancestry_contracts = (
        (
            ATTEMPT_2_SOURCE_COMMIT,
            source_commit,
            "attempt-2 source is not an ancestor of generator source",
        ),
        (
            ATTEMPT_2_SOURCE_COMMIT,
            freeze_commit,
            "attempt-2 source is not an ancestor of freeze commit",
        ),
        (
            source_commit,
            freeze_commit,
            "generator source commit is not an ancestor of freeze commit",
        ),
    )
    for ancestor, descendant, message in ancestry_contracts:
        ancestry = _git(
            "merge-base",
            "--is-ancestor",
            ancestor,
            descendant,
            check=False,
        )
        if ancestry.returncode != 0:
            raise GenerationRefused(message)
    return {
        "freeze_path": freeze_path,
        "freeze_relative_path": freeze_relative,
        "freeze_sha256": freeze_sha256,
        "freeze_commit": freeze_commit,
        "source_commit": source_commit,
        "generator_sha256": generator_sha256,
        "output_directory": output_path,
        "output_relative_path": output_relative,
        "runtime": runtime,
        "document": document,
    }


def _raw_relative_path(seed: int, family: str) -> str:
    return (
        f"seed-{seed}/{family}.N{MAXIMUM_LENGTH}."
        "complex64-le.raw"
    )


def _write_exclusive(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _verify_authorized_revalidation_bindings(
    authorization: GenerationAuthorization,
    *,
    phase: str,
) -> None:
    """Statelessly recheck current and committed authorization bytes."""
    _assert_generation_authorized(authorization)
    current_bindings = {
        GENERATOR_PATH: authorization.generator_sha256,
        authorization.freeze_path: authorization.freeze_sha256,
        AMENDMENT_PATH: AMENDMENT_SHA256,
        AMENDMENT_VALIDATOR_PATH: AMENDMENT_VALIDATOR_SHA256,
    }
    for path, expected in current_bindings.items():
        if _sha256_file(path) != expected:
            raise GenerationRefused(
                f"{phase}: authorized bytes changed: "
                f"{path.relative_to(REPO)}"
            )
    if runtime_fingerprint() != authorization.runtime:
        raise GenerationRefused(f"{phase}: authorized runtime changed")

    committed_generator = _git(
        "show",
        f"{authorization.source_commit}:{GENERATOR_RELATIVE_PATH}",
        text=False,
    ).stdout
    if (
        hashlib.sha256(committed_generator).hexdigest()
        != authorization.generator_sha256
    ):
        raise GenerationRefused(
            f"{phase}: authorized committed generator blob changed"
        )
    committed_freeze = _git(
        "show",
        (
            f"{authorization.freeze_commit}:"
            f"{authorization.freeze_relative_path}"
        ),
        text=False,
    ).stdout
    if (
        hashlib.sha256(committed_freeze).hexdigest()
        != authorization.freeze_sha256
    ):
        raise GenerationRefused(
            f"{phase}: authorized committed freeze blob changed"
        )
    attempt_2_meta = _git(
        "show",
        "-s",
        "--format=%H%n%P%n%s",
        ATTEMPT_2_SOURCE_COMMIT,
    ).stdout.splitlines()
    if attempt_2_meta != [
        ATTEMPT_2_SOURCE_COMMIT,
        ATTEMPT_2_PREREGISTRATION_COMMIT,
        ATTEMPT_2_SOURCE_SUBJECT,
    ]:
        raise GenerationRefused(
            f"{phase}: attempt-2 source commit binding changed"
        )
    source_meta = _git(
        "show",
        "-s",
        "--format=%H%n%s",
        authorization.source_commit,
    ).stdout.splitlines()
    if source_meta != [
        authorization.source_commit,
        authorization.document["generator_source"]["commit_subject"],
    ]:
        raise GenerationRefused(
            f"{phase}: generator source commit binding changed"
        )
    for ancestor, descendant, message in (
        (
            ATTEMPT_2_SOURCE_COMMIT,
            authorization.source_commit,
            "attempt-2 source is not generator-source ancestor",
        ),
        (
            ATTEMPT_2_SOURCE_COMMIT,
            authorization.freeze_commit,
            "attempt-2 source is not freeze ancestor",
        ),
        (
            authorization.source_commit,
            authorization.freeze_commit,
            "generator source is not freeze ancestor",
        ),
    ):
        if _git(
            "merge-base",
            "--is-ancestor",
            ancestor,
            descendant,
            check=False,
        ).returncode != 0:
            raise GenerationRefused(f"{phase}: {message}")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_digest_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    expected_pairs = [
        (seed, family)
        for seed in ADAPTIVE_SEEDS
        for family in NOVELTY_FAMILIES
    ]
    if len(records) != len(expected_pairs):
        raise ValueError("digest record inventory must contain all 12 pairs")
    validated: list[dict[str, Any]] = []
    expected_byte_count = (
        ROWS_PER_FAMILY_PER_SEED
        * MAXIMUM_LENGTH
        * CANONICAL_COMPLEX_DTYPE.itemsize
    )
    expected_keys = {
        "base_seed",
        "family",
        "derived_seed",
        "row_count",
        "generated_once_at_maximum_length",
        "dtype",
        "canonical_byte_order",
        "longest_complex64_sha256",
        "prefix_complex64_sha256",
        "shorter_observation_rule",
        "uses_frequency_transform",
        "raw_relative_path",
        "raw_byte_count",
        "raw_sha256",
    }
    for row, (seed, family) in zip(records, expected_pairs):
        if set(row) != expected_keys:
            raise ValueError("digest record key inventory changed")
        prefixes = row.get("prefix_complex64_sha256")
        if (
            row.get("base_seed") != seed
            or row.get("family") != family
            or row.get("derived_seed") != derive_family_seed(seed, family)
            or row.get("row_count") != ROWS_PER_FAMILY_PER_SEED
            or row.get("generated_once_at_maximum_length") != MAXIMUM_LENGTH
            or row.get("dtype") != OUTPUT_DTYPE_NAME
            or row.get("canonical_byte_order")
            != "little_endian_complex64_interleaved_real_imag"
            or row.get("shorter_observation_rule")
            != "exact leading causal raw prefix of the same longest row"
            or row.get("uses_frequency_transform") is not False
            or row.get("raw_relative_path") != _raw_relative_path(seed, family)
            or row.get("raw_byte_count") != expected_byte_count
            or not _is_sha256(row.get("longest_complex64_sha256"))
            or row.get("raw_sha256") != row.get("longest_complex64_sha256")
            or not isinstance(prefixes, Mapping)
            or set(prefixes)
            != {str(length) for length in OBSERVATION_LENGTHS}
            or not all(_is_sha256(value) for value in prefixes.values())
            or prefixes[str(MAXIMUM_LENGTH)]
            != row.get("longest_complex64_sha256")
        ):
            raise ValueError(
                f"digest record changed for seed={seed} family={family}"
            )
        validated.append(dict(row))
    return validated


def _build_authorized_digest_manifest(
    authorization: GenerationAuthorization,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Pure manifest assembly from an authorized exact 12-record inventory."""
    _assert_generation_authorized(authorization)
    _require_generation_revalidations(authorization)
    validated_records = _validate_digest_records(records)
    return {
        "schema": "atomos.v5.scale-orbit.adaptive-novelty-digest-manifest",
        "schema_version": 1,
        "status": "immutable_first_generation_complete_before_scoring",
        "lineage": "v5-scale-orbit",
        "development_only": True,
        "release_evidence": False,
        "adaptive_reuse_allowed": True,
        "independent_evidence_eligible": False,
        "pre_generation_source_freeze": {
            "relative_path": authorization.freeze_relative_path,
            "sha256": authorization.freeze_sha256,
            "commit": authorization.freeze_commit,
        },
        "attempt_2_source": {
            "commit": ATTEMPT_2_SOURCE_COMMIT,
            "commit_subject": ATTEMPT_2_SOURCE_SUBJECT,
            "is_generator_source_commit_ancestor": True,
            "is_pre_generation_freeze_commit_ancestor": True,
        },
        "generator_source": {
            "relative_path": GENERATOR_RELATIVE_PATH,
            "sha256": authorization.generator_sha256,
            "commit": authorization.source_commit,
            "is_pre_generation_freeze_commit_ancestor": True,
        },
        "parent_bindings": {
            "adaptive_novelty_amendment": {
                "relative_path": AMENDMENT_RELATIVE_PATH,
                "sha256": AMENDMENT_SHA256,
            },
            "adaptive_novelty_amendment_validator": {
                "relative_path": AMENDMENT_VALIDATOR_RELATIVE_PATH,
                "sha256": AMENDMENT_VALIDATOR_SHA256,
            },
        },
        "authorization_revalidation": {
            "authorized_generator_sha256_retained_without_recomputation":
                authorization.generator_sha256,
            "current_and_committed_generator_sha256_rechecked_immediately_before_generation":
                True,
            "current_and_committed_generator_sha256_rechecked_immediately_after_generation":
                True,
            "current_and_committed_freeze_sha256_rechecked_before_and_after_generation":
                True,
            "attempt_2_generator_and_freeze_ancestry_rechecked_before_and_after_generation":
                True,
        },
        "output_directory_relative_path": authorization.output_relative_path,
        "inventory": frozen_inventory_contract(),
        "mechanics": generator_mechanics_contract(),
        "runtime": _deep_thaw(authorization.runtime),
        "records": validated_records,
        "exact_cell_inventory": exact_cell_inventory(),
        "first_generation_contract": {
            "required_before_scoring": True,
            "immutable_after_first_generation": True,
            "raw_files_contain_only_longest_rows": True,
            "shorter_views_are_exact_causal_prefixes": True,
            "route_does_not_change_or_duplicate_generated_raw_rows": True,
            "deterministic_regeneration_allowed_only_when_every_digest_matches":
                True,
            "redraw_overwrite_or_seed_omission_allowed": False,
        },
        "no_signal_disclosure": {
            "all_rows_are_exact_complex64_zeros": True,
            "rows_repeat_within_and_across_seeds": True,
            "seed_independent_draws_claimed": False,
            "repetition_does_not_reduce_required_seed_cell_coverage": True,
        },
        "scoring_status": "not_started",
        "model_inference_runs": 0,
        "known_or_sealed_corpus_files_read": 0,
    }


def _execute_authorized_generation(
    authorization: GenerationAuthorization,
) -> Path:
    _assert_generation_authorized(authorization)
    target = authorization.output_directory
    if target.exists():
        raise GenerationRefused(
            "first-generation output directory already exists"
        )
    _revalidate_authorized_bindings(
        authorization,
        phase="immediately_before_generation",
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.first-generation-",
            dir=target.parent,
        )
    )
    try:
        records: list[dict[str, Any]] = []
        for seed in ADAPTIVE_SEEDS:
            for family in NOVELTY_FAMILIES:
                rows = _generate_authorized_family_rows(
                    authorization,
                    base_seed=seed,
                    family=family,
                )
                digest = family_digest_record(
                    rows,
                    base_seed=seed,
                    family=family,
                    observation_lengths=OBSERVATION_LENGTHS,
                )
                relative = _raw_relative_path(seed, family)
                raw = canonical_complex64_bytes(rows)
                expected_bytes = (
                    ROWS_PER_FAMILY_PER_SEED
                    * MAXIMUM_LENGTH
                    * CANONICAL_COMPLEX_DTYPE.itemsize
                )
                if len(raw) != expected_bytes:
                    raise RuntimeError("canonical raw byte count changed")
                raw_path = temporary / relative
                _write_exclusive(raw_path, raw)
                raw_sha256 = _sha256_file(raw_path)
                if raw_sha256 != digest["longest_complex64_sha256"]:
                    raise RuntimeError("raw file digest differs from row digest")
                records.append(
                    {
                        **digest,
                        "raw_relative_path": relative,
                        "raw_byte_count": expected_bytes,
                        "raw_sha256": raw_sha256,
                    }
                )
                del raw
                del rows

        _revalidate_authorized_bindings(
            authorization,
            phase="immediately_after_generation",
        )
        manifest = _build_authorized_digest_manifest(
            authorization,
            records,
        )
        manifest_path = temporary / DIGEST_MANIFEST_FILENAME
        _write_exclusive(manifest_path, _canonical_json_bytes(manifest))
        _revalidate_authorized_bindings(
            authorization,
            phase="immediately_before_atomic_install",
        )
        if target.exists():
            raise GenerationRefused(
                "output appeared during generation; refusing atomic install"
            )
        temporary.rename(target)
        return target / DIGEST_MANIFEST_FILENAME
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def execute_frozen_generation(
    *,
    pre_generation_freeze: str | Path,
    output_directory: str | Path,
) -> Path:
    """Authorize first, then materialize the exact joint four-seed inventory."""
    authorization = authorize_generation(
        pre_generation_freeze=pre_generation_freeze,
        output_directory=output_directory,
    )
    return _execute_authorized_generation(authorization)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the exact v5 adaptive novelty inventory only after a "
            "later committed pre-generation freeze authorizes execution."
        )
    )
    parser.add_argument("--pre-generation-freeze", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    manifest = execute_frozen_generation(
        pre_generation_freeze=arguments.pre_generation_freeze,
        output_directory=arguments.output_dir,
    )
    print(str(manifest))


if __name__ == "__main__":
    main()
