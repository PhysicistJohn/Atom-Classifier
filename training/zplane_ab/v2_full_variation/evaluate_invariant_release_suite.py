"""Evaluate one sealed matched-length suite against one frozen runtime bundle.

This evaluator is intentionally data-blind until both sides are frozen:

* ``RELEASE_INTENT.json`` must predeclare the split, novelty, scale, and gate
  protocol before corpus generation;
* ``RELEASE_MANIFEST.json`` must bind that intent, every corpus byte, and the
  exact candidate runtime-bundle SHA-256;
* the candidate's centers, feature standardization, prototypes, weighted LOF
  references/ranks/weights, and unknown threshold are loaded verbatim;
* release rows and fresh novelty are never used to fit, calibrate, select, or
  tune any model or threshold.

The generated corpora share row identities across capture lengths. Five-shot
support is the five lowest predeclared SHA-256 split digests per class; every
other matched row is query. Closed, family, high-SNR, length, physical-scale,
five-shot, and weighted-LOF novelty metrics are all reported without touching
the historical development corpus or its consumed test half.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
TRAINING = HERE.parent.parent
REPO = TRAINING.parent
TOOLS = REPO / "tools"
LIVE_CORPUS = (TRAINING / "artifacts" / "signallab-corpus").resolve()
LAUNCHER = (TOOLS / "generate-signallab-iq-release-suite.mjs").resolve()
CORPUS_GENERATOR = (TOOLS / "generate-signallab-iq-corpus.ts").resolve()
PREFIX_DERIVER = (
    TOOLS / "derive-signallab-iq-prefix-corpus.mjs"
).resolve()
for path in (TRAINING, HERE.parent, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import invariant_patch_preprocess as invariant_preprocess  # noqa: E402
import preprocess as production_preprocess  # noqa: E402
from assemble_invariant_candidate import (  # noqa: E402
    load_runtime_bundle,
)
from invariant_fusion import (  # noqa: E402
    CenteredInvariantFusion,
    _safe_unit as _fusion_safe_unit,
)
from invariant_patch_cnn import (  # noqa: E402
    InvariantPatchCNN,
    InvariantPatchConfig,
)
from known_only_patch_openset import KnownOnlyLOFOpenSet  # noqa: E402
from openset_eval import NOVELTY  # noqa: E402


RELEASE_PROTOCOL = "signallab-matched-length-release-v1"
PREFIX_DERIVATION_PROTOCOL = "signallab-row-prefix-v1"
EVALUATOR_SCHEMA = 1
EVALUATION_VERSION = "invariant-release-evaluation-v1"
MATCHED_CAPTURE_LENGTH = 16384
REQUIRED_CAPTURE_LENGTHS = (4096, 8192, 16384, 32768)
HIGH_SNR_DB = 18.0
FIVE_SHOT_K = 5
SPLIT_VERSION = "sha256-lowest-per-class-v1"
SPLIT_SALT = "atomos-invariant-release-five-shot-v1"
NOVELTY_SEED = 20260729
NOVELTY_N_EACH_PER_LENGTH = 300
NOVELTY_FAMILIES = ("noise", "chirp")
PHYSICAL_SCALE_FACTORS = (0.5, 0.75, 1.0, 1.5, 2.0)
SCALE_EDGE_MARGIN = 1.0 / 512.0
SCALE_MIN_PER_CLASS = 20
MIN_TARGET_PER_CLASS = 80
REQUIRED_NODE_VERSION = "v22.23.1"
REQUIRED_NPM_VERSION = "10.9.8"
SOURCE_TREE_DIGEST_ALGORITHM = (
    "sha256(concat(sorted_posix_relative_path_utf8 + NUL + file_bytes)); "
    "exclude path components node_modules/dist and basename "
    "RELEASE_SOURCE_PROVENANCE.json; reject other non-regular entries"
)
DEPENDENCY_CONTRACT: dict[str, dict[str, Any]] = {
    "signallab": {
        "git_commit": "7c8303a0338b0f7c088737f8df2d935fd04bb033",
        "git_tree": "f0fade7d2dfd8f6ebd1e62caeae7d44d916d6796",
        "package_lock_sha256": (
            "5dd71450021febce4acbb9a835a3cadea659e5b4ee088217b9b697bf3f0a5fa6"
        ),
        "source_tree_sha256": (
            "93bec60b8da571bd0cf4a51630bd2b053d95eaf1bf2da503251775545fb873c0"
        ),
        "source_file_count": 81,
    },
    "atom_dsp": {
        "git_commit": "4bb4d7075ba0e30512480ffbf0cb6e5757ffa218",
        "git_tree": "3d1a8b7a97658a98b8ef36adc7f74f97cf16e9d2",
        "package_lock_sha256": (
            "b1c8198c18a4b25ed605db0781ae57b6050e0123ab0696ca2bdcba112696c0ca"
        ),
        "source_tree_sha256": (
            "80a8363407acfa60ea366a1b6f83401a07797ef7c9776579bc949eab16e04588"
        ),
        "source_file_count": 20,
        "dist_index_js_sha256": (
            "caef3a8b3cbf4c300f8ceab5619a1db6b4fcebb942d428577f50790a1fb360e7"
        ),
        "dist_index_dts_sha256": (
            "fc6933a4c7e7d9aeaa2ca22043f6e22b2333671c08b81b1cd35b29a7de08c980"
        ),
    },
}

GATE_FLOORS = {
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

EXPECTED_EVALUATION_PROTOCOL: dict[str, Any] = {
    "version": EVALUATION_VERSION,
    "matched_capture_length": MATCHED_CAPTURE_LENGTH,
    "required_capture_lengths": list(REQUIRED_CAPTURE_LENGTHS),
    "minimum_target_per_class": MIN_TARGET_PER_CLASS,
    "length_observation_rule": (
        "generate maximum capture length once; every shorter observed and "
        "clean row is a bit-exact cf32le prefix of the corresponding "
        "maximum-length row"
    ),
    "high_snr_db": int(HIGH_SNR_DB),
    "five_shot": {
        "k": FIVE_SHOT_K,
        "split_version": SPLIT_VERSION,
        "hash_payload": (
            "utf8(split_salt + NUL + release_seed_decimal + NUL + "
            "canonical_row_identity_json)"
        ),
        "split_salt": SPLIT_SALT,
        "support_rule": (
            "lowest five lowercase SHA-256 digests per class; "
            "ties by corpus row index"
        ),
        "query_rule": (
            "all remaining rows; identical row indices at every matched "
            "capture length"
        ),
        "prototype_source": (
            "support rows embedded at matched_capture_length; one fixed "
            "five-shot prototype set reused at every capture length"
        ),
    },
    "novelty": {
        "seed": NOVELTY_SEED,
        "families": list(NOVELTY_FAMILIES),
        "n_each_per_length": NOVELTY_N_EACH_PER_LENGTH,
        "generation": (
            "generate each fresh openset_eval.NOVELTY realization once at "
            "maximum capture length; shorter observations are exact prefixes; "
            "one family seed derived by SHA-256 from base seed and family"
        ),
        "seed_derivation": (
            "uint64 little-endian first 8 bytes of "
            "SHA-256(utf8(base seed decimal + NUL + family))"
        ),
        "calibration": (
            "none; frozen weighted-LOF references, enrollment ranks, "
            "weights, and threshold only"
        ),
    },
    "physical_scale_factors": list(PHYSICAL_SCALE_FACTORS),
    "physical_scale_support": {
        "edge_guard_fraction": SCALE_EDGE_MARGIN,
        "minimum_rows_per_class": SCALE_MIN_PER_CLASS,
        "rule": (
            "for every requested factor, effective resampling scale times "
            "(abs(centreOffsetFrac) + bandwidthHz/(2*sampleRateHz)) must be "
            "strictly below 0.5 - edge_guard_fraction"
        ),
        "selection_timing": (
            "metadata-only common subset fixed before preprocessing or inference"
        ),
    },
    "gates": dict(GATE_FLOORS),
}

ROW_IDENTITY_FIELDS = (
    "cls",
    "profile",
    "sampleRateHz",
    "bandwidthHz",
    "impaired",
    "startSampleIndex",
    "centerHz",
    "snrDb",
    "rxPowerDbm",
    "noiseFigureDb",
    "centreOffsetFrac",
    "adcBits",
    "clockErrorPpm",
    "multipathTaps",
    "impairmentSeed",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_tree_digest(root: Path) -> tuple[str, int]:
    """Mirror the launcher's deterministic archive-source digest."""
    base = Path(root)
    if not base.is_dir() or base.is_symlink():
        raise ValueError(f"source-tree root must be a real directory: {base}")
    excluded_directories = {"node_modules", "dist"}
    excluded_file = "RELEASE_SOURCE_PROVENANCE.json"
    files: list[tuple[str, Path]] = []

    def visit(directory: Path, relative_parts: tuple[str, ...]) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise ValueError(f"cannot enumerate source tree {directory}") from exc
        for entry in entries:
            if entry.name in {".", ".."}:
                raise ValueError("source tree contains an invalid path entry")
            relative = relative_parts + (entry.name,)
            path = Path(entry.path)
            if entry.is_symlink():
                raise ValueError(f"source tree contains a symlink: {path}")
            if entry.is_dir(follow_symlinks=False):
                if entry.name not in excluded_directories:
                    visit(path, relative)
                continue
            if not entry.is_file(follow_symlinks=False):
                raise ValueError(
                    f"source tree contains a non-regular entry: {path}"
                )
            if entry.name != excluded_file:
                files.append(("/".join(relative), path))

    visit(base, ())
    files.sort(key=lambda pair: pair[0])
    digest = hashlib.sha256()
    for relative, path in files:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest(), len(files)


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_finite_json(value: Any, *, location: str = "$") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        if abs(value) > 2**53 - 1:
            raise ValueError(f"JSON integer exceeds safe range at {location}")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"JSON number is non-finite at {location}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _validate_finite_json(child, location=f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"JSON object key is not a string at {location}")
            _validate_finite_json(child, location=f"{location}.{key}")
        return
    raise ValueError(f"unsupported JSON value at {location}")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"required JSON must be a regular non-symlink file: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    _validate_finite_json(value)
    return value


def _is_below(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _regular_file(path: Path, *, name: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{name} must be a regular non-symlink file: {path}")


def _finite_number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _row_identity(item: Mapping[str, Any]) -> str:
    missing = [name for name in ROW_IDENTITY_FIELDS if name not in item]
    if missing:
        raise ValueError(f"corpus row is missing identity fields: {missing}")
    identity = {name: item[name] for name in ROW_IDENTITY_FIELDS}
    return json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _row_keys(items: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        hashlib.sha256(_row_identity(item).encode("utf-8")).hexdigest()
        for item in items
    ]


@dataclass(frozen=True)
class CorpusSpec:
    capture_length: int
    directory: Path
    manifest_path: Path
    raw_path: Path
    clean_path: Path
    manifest: dict[str, Any]
    row_keys: tuple[str, ...]


@dataclass(frozen=True)
class ReleaseSuite:
    root: Path
    intent_path: Path
    manifest_path: Path
    intent: dict[str, Any]
    release_manifest: dict[str, Any]
    candidate_path: Path
    candidate_sha256: str
    release_seed: int
    classes: tuple[str, ...]
    corpora: dict[int, CorpusSpec]
    support_indices: np.ndarray
    query_indices: np.ndarray
    split_report: dict[str, Any]
    prefix_nesting_report: dict[str, Any]
    dependency_report: dict[str, Any]
    start_probe_report: dict[str, Any]


def _verify_file_record(path: Path, record: Any, *, name: str) -> None:
    _regular_file(path, name=name)
    if not isinstance(record, Mapping):
        raise ValueError(f"{name} manifest record must be an object")
    expected_size = _integer(record.get("bytes"), f"{name}.bytes")
    expected_hash = str(record.get("sha256", "")).lower()
    if len(expected_hash) != 64 or any(c not in "0123456789abcdef" for c in expected_hash):
        raise ValueError(f"{name}.sha256 must be a lowercase SHA-256 digest")
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"{name} byte count mismatch: expected {expected_size}, got {actual_size}"
        )
    actual_hash = _sha256(path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"{name} SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
        )


def _canonical_recorded_path(value: Any, *, name: str) -> Path:
    raw = Path(str(value)).expanduser()
    canonical = raw.resolve()
    if str(raw) != str(canonical):
        raise ValueError(f"{name} must record a canonical real path")
    return canonical


def _validate_dependency_provenance(
    value: Any,
    runtime_value: Any,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("dependency_provenance must be an object")
    if runtime_value != {
        "node": REQUIRED_NODE_VERSION,
        "npm": REQUIRED_NPM_VERSION,
        "npx": REQUIRED_NPM_VERSION,
    }:
        raise ValueError("runtime_provenance differs from the frozen Node/npm engine")
    if set(value) != {"signallab", "atom_dsp"}:
        raise ValueError("dependency_provenance has unexpected dependency keys")
    signallab_record = value["signallab"]
    atom_dsp_record = value["atom_dsp"]
    if not isinstance(signallab_record, Mapping) or not isinstance(
        atom_dsp_record, Mapping
    ):
        raise ValueError("dependency provenance records must be objects")

    signallab_root = _canonical_recorded_path(
        signallab_record.get("root"),
        name="SignalLab root",
    )
    atom_dsp_root = _canonical_recorded_path(
        atom_dsp_record.get("root"),
        name="Atom-DSP root",
    )
    for name, root in (
        ("SignalLab", signallab_root),
        ("Atom-DSP", atom_dsp_root),
    ):
        if not root.is_dir() or root.is_symlink():
            raise ValueError(f"{name} source root is not a real directory")
    if atom_dsp_root != (signallab_root.parent / "Atom-DSP").resolve():
        raise ValueError("Atom-DSP is not the pinned sibling of SignalLab")

    package_resolution = Path(
        str(atom_dsp_record.get("package_resolution"))
    ).expanduser()
    expected_resolution = (
        signallab_root / "node_modules" / "@atomos" / "dsp"
    )
    if (
        package_resolution != expected_resolution
        or not package_resolution.is_symlink()
        or package_resolution.resolve() != atom_dsp_root
    ):
        raise ValueError(
            "SignalLab @atomos/dsp package resolution is not the pinned "
            "sibling symlink"
        )

    signallab_tree_hash, signallab_count = _source_tree_digest(
        signallab_root
    )
    atom_dsp_tree_hash, atom_dsp_count = _source_tree_digest(atom_dsp_root)
    signallab_lock = signallab_root / "package-lock.json"
    atom_dsp_lock = atom_dsp_root / "package-lock.json"
    dist_js = atom_dsp_root / "dist" / "index.js"
    dist_dts = atom_dsp_root / "dist" / "index.d.ts"
    for path, name in (
        (signallab_lock, "SignalLab package-lock"),
        (atom_dsp_lock, "Atom-DSP package-lock"),
        (dist_js, "Atom-DSP dist/index.js"),
        (dist_dts, "Atom-DSP dist/index.d.ts"),
    ):
        _regular_file(path, name=name)

    expected_signallab = {
        "root": str(signallab_root),
        **DEPENDENCY_CONTRACT["signallab"],
        "observed_source_tree": {
            "sha256": signallab_tree_hash,
            "file_count": signallab_count,
        },
        "package_lock_path": str(signallab_lock),
    }
    expected_atom_dsp = {
        "root": str(atom_dsp_root),
        "package_resolution": str(package_resolution),
        **DEPENDENCY_CONTRACT["atom_dsp"],
        "observed_source_tree": {
            "sha256": atom_dsp_tree_hash,
            "file_count": atom_dsp_count,
        },
        "package_lock_path": str(atom_dsp_lock),
        "dist_index_js": {
            "path": str(dist_js),
            "bytes": dist_js.stat().st_size,
            "sha256": _sha256(dist_js),
        },
        "dist_index_dts": {
            "path": str(dist_dts),
            "bytes": dist_dts.stat().st_size,
            "sha256": _sha256(dist_dts),
        },
    }
    if signallab_record != expected_signallab:
        raise ValueError(
            "SignalLab provenance differs from current pinned source bytes"
        )
    if atom_dsp_record != expected_atom_dsp:
        raise ValueError(
            "Atom-DSP provenance differs from current pinned source/build bytes"
        )
    if _sha256(signallab_lock) != DEPENDENCY_CONTRACT["signallab"][
        "package_lock_sha256"
    ]:
        raise ValueError("SignalLab package-lock digest mismatch")
    if _sha256(atom_dsp_lock) != DEPENDENCY_CONTRACT["atom_dsp"][
        "package_lock_sha256"
    ]:
        raise ValueError("Atom-DSP package-lock digest mismatch")
    if signallab_tree_hash != DEPENDENCY_CONTRACT["signallab"][
        "source_tree_sha256"
    ] or signallab_count != DEPENDENCY_CONTRACT["signallab"][
        "source_file_count"
    ]:
        raise ValueError("SignalLab source-tree contract mismatch")
    if atom_dsp_tree_hash != DEPENDENCY_CONTRACT["atom_dsp"][
        "source_tree_sha256"
    ] or atom_dsp_count != DEPENDENCY_CONTRACT["atom_dsp"][
        "source_file_count"
    ]:
        raise ValueError("Atom-DSP source-tree contract mismatch")
    for path, key in (
        (dist_js, "dist_index_js_sha256"),
        (dist_dts, "dist_index_dts_sha256"),
    ):
        if _sha256(path) != DEPENDENCY_CONTRACT["atom_dsp"][key]:
            raise ValueError(f"Atom-DSP {path.name} build digest mismatch")
    return {
        "source_tree_digest_algorithm": SOURCE_TREE_DIGEST_ALGORITHM,
        "runtime": dict(runtime_value),
        "signallab": expected_signallab,
        "atom_dsp": expected_atom_dsp,
        "passes": True,
    }


def _validate_predeclared_protocol(value: Any) -> None:
    if value != EXPECTED_EVALUATION_PROTOCOL:
        raise ValueError(
            "release evaluation protocol is missing or differs from the "
            "precommitted evaluator contract"
        )


def predeclared_five_shot_split(
    items: Sequence[Mapping[str, Any]],
    classes: Sequence[str],
    release_seed: int,
    *,
    k: int = FIVE_SHOT_K,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return the precommitted support/query partition without reading signals."""
    seed = _integer(release_seed, "release_seed")
    if k != FIVE_SHOT_K:
        raise ValueError(f"five-shot k is frozen at {FIVE_SHOT_K}")
    row_keys = _row_keys(items)
    support_by_class: dict[str, list[int]] = {}
    digest_by_support: dict[str, list[str]] = {}
    all_support: list[int] = []
    for class_name in classes:
        candidates = [
            index
            for index, item in enumerate(items)
            if item.get("cls") == class_name
        ]
        if len(candidates) <= k:
            raise ValueError(
                f"class {class_name!r} has {len(candidates)} rows; "
                f"needs more than {k} for disjoint support/query"
            )
        ranked: list[tuple[str, int]] = []
        for index in candidates:
            payload = (
                SPLIT_SALT
                + "\0"
                + str(seed)
                + "\0"
                + _row_identity(items[index])
            ).encode("utf-8")
            ranked.append((hashlib.sha256(payload).hexdigest(), index))
        ranked.sort(key=lambda pair: (pair[0], pair[1]))
        selected = [index for _digest, index in ranked[:k]]
        support_by_class[class_name] = selected
        digest_by_support[class_name] = [digest for digest, _index in ranked[:k]]
        all_support.extend(selected)

    support = np.asarray(sorted(all_support), dtype=np.int64)
    support_set = set(int(value) for value in support)
    query = np.asarray(
        [index for index in range(len(items)) if index not in support_set],
        dtype=np.int64,
    )
    if len(np.intersect1d(support, query)) or len(support) + len(query) != len(items):
        raise AssertionError("predeclared support/query partition is not disjoint/exhaustive")
    report = {
        "version": SPLIT_VERSION,
        "salt": SPLIT_SALT,
        "release_seed": seed,
        "k_per_class": k,
        "support_by_class": support_by_class,
        "support_digest_by_class": digest_by_support,
        "support_indices_sha256": hashlib.sha256(
            support.astype("<i8", copy=False).tobytes()
        ).hexdigest(),
        "query_indices_sha256": hashlib.sha256(
            query.astype("<i8", copy=False).tobytes()
        ).hexdigest(),
        "row_identity_sequence_sha256": hashlib.sha256(
            "\n".join(row_keys).encode("ascii")
        ).hexdigest(),
    }
    return support, query, report


def verify_prefix_nesting(
    corpora: Mapping[int, CorpusSpec],
    *,
    rows_per_chunk: int = 8,
) -> dict[str, Any]:
    """Verify that every shorter row is a bit-exact prefix of the longest row.

    Separate synthesis at each capture length is not a valid causal length sweep:
    seeded noise, normalization, quantization, or a synthesizer implementation can
    still depend on the requested sample count. This compares stored float32 bit
    patterns row by row for both the observed and clean-pair streams before any
    preprocessing or inference.
    """
    if (
        isinstance(rows_per_chunk, bool)
        or not isinstance(rows_per_chunk, int)
        or rows_per_chunk <= 0
    ):
        raise ValueError("rows_per_chunk must be a positive integer")
    lengths = sorted(corpora)
    if not lengths:
        raise ValueError("prefix verification needs at least one corpus")
    longest_length = lengths[-1]
    longest = corpora[longest_length]
    count = _integer(longest.manifest.get("count"), "longest.count", 1)
    streams = {
        "observed": "raw_path",
        "clean": "clean_path",
    }
    comparisons: list[dict[str, Any]] = []
    for length in lengths[:-1]:
        shorter = corpora[length]
        if int(shorter.manifest.get("count", -1)) != count:
            raise ValueError(
                f"n{length} row count differs from longest n{longest_length}"
            )
        for stream_name, path_attribute in streams.items():
            short_path = getattr(shorter, path_attribute)
            long_path = getattr(longest, path_attribute)
            # Compare 32-bit words, not numeric floats. This catches NaN payload,
            # signed-zero, and any other byte-level change.
            short_words = np.memmap(
                short_path,
                dtype="<u4",
                mode="r",
                shape=(count, length, 2),
            )
            long_words = np.memmap(
                long_path,
                dtype="<u4",
                mode="r",
                shape=(count, longest_length, 2),
            )
            for start in range(0, count, rows_per_chunk):
                stop = min(count, start + rows_per_chunk)
                mismatch = np.argwhere(
                    short_words[start:stop]
                    != long_words[start:stop, :length]
                )
                if mismatch.size:
                    relative_row, sample_index, component = (
                        int(value) for value in mismatch[0]
                    )
                    component_name = "I" if component == 0 else "Q"
                    raise ValueError(
                        f"n{length} {stream_name} is not a bit-exact prefix of "
                        f"n{longest_length}: row {start + relative_row}, sample "
                        f"{sample_index}, component {component_name}"
                    )
            del short_words
            del long_words
            comparisons.append(
                {
                    "capture_length": length,
                    "longest_capture_length": longest_length,
                    "stream": stream_name,
                    "rows_compared": count,
                    "words_compared": count * length * 2,
                    "bit_exact": True,
                }
            )
    return {
        "definition": (
            "every shorter observed and clean cf32 row must be a bit-exact "
            "prefix of the corresponding longest stored row"
        ),
        "longest_capture_length": longest_length,
        "row_count": count,
        "streams": list(streams),
        "comparisons": comparisons,
        "passes": True,
    }


def _actual_file_record(path: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def validate_prefix_derivation_records(
    corpora: Mapping[int, CorpusSpec],
) -> dict[str, Any]:
    """Bind every shorter manifest's derivation record to the longest assets."""
    lengths = sorted(corpora)
    if lengths != list(REQUIRED_CAPTURE_LENGTHS):
        raise ValueError("prefix derivation records need the full frozen length set")
    longest = corpora[lengths[-1]]
    if "prefixDerivation" in longest.manifest:
        raise ValueError("longest corpus must be generated, not prefix-derived")
    source_records = {
        "corpus.json": _actual_file_record(longest.manifest_path),
        "corpus.f32": _actual_file_record(longest.raw_path),
        "corpus_clean.f32": _actual_file_record(longest.clean_path),
    }
    derived = []
    for length in lengths[:-1]:
        spec = corpora[length]
        value = spec.manifest.get("prefixDerivation")
        if not isinstance(value, Mapping):
            raise ValueError(f"n{length} is missing prefixDerivation provenance")
        expected_output = {
            "corpus.f32": _actual_file_record(spec.raw_path),
            "corpus_clean.f32": _actual_file_record(spec.clean_path),
        }
        source_directory = Path(
            str(value.get("sourceDirectory", ""))
        ).expanduser().resolve()
        if (
            value.get("protocol") != PREFIX_DERIVATION_PROTOCOL
            or source_directory != longest.directory.resolve()
            or value.get("sourceSampleCount") != longest.capture_length
            or value.get("targetSampleCount") != length
            or value.get("rowCount") != spec.manifest.get("count")
            or value.get("bytesPerComplexSample") != 8
            or value.get("sourceFiles") != source_records
            or value.get("outputFiles") != expected_output
        ):
            raise ValueError(
                f"n{length} prefixDerivation record does not bind the "
                "longest corpus and derived output bytes"
            )
        derived.append(
            {
                "capture_length": length,
                "source_capture_length": longest.capture_length,
                "protocol": PREFIX_DERIVATION_PROTOCOL,
                "record_valid": True,
            }
        )
    return {
        "source_capture_length": longest.capture_length,
        "derived": derived,
        "passes": True,
    }


def _validate_unscored_start_probe(
    release_manifest: Mapping[str, Any],
    *,
    root: Path,
    longest: CorpusSpec,
    classes: Sequence[str],
    release_seed: int,
    target_per_class: int,
    signal_lab_root: Path,
) -> dict[str, Any]:
    record = release_manifest.get("start_probe")
    expected_purpose = (
        "select one occupied start per row before longest-only nuisance realization"
    )
    if (
        not isinstance(record, Mapping)
        or set(record)
        != {"scored", "purpose", "capture_length", "directory", "files"}
        or record.get("scored") is not False
        or record.get("purpose") != expected_purpose
        or record.get("capture_length") != REQUIRED_CAPTURE_LENGTHS[0]
    ):
        raise ValueError("release start_probe declaration is invalid")
    expected_directory = root / f"_start_probe_n{REQUIRED_CAPTURE_LENGTHS[0]}"
    directory_input = Path(str(record.get("directory", ""))).expanduser()
    if (
        directory_input.is_symlink()
        or expected_directory.is_symlink()
        or directory_input.resolve() != expected_directory
    ):
        raise ValueError("release start_probe directory escapes its sealed path")
    files = record.get("files")
    if not isinstance(files, Mapping) or set(files) != {
        "corpus.json",
        "corpus.f32",
        "corpus_clean.f32",
    }:
        raise ValueError("release start_probe files are incomplete")
    paths = {
        "corpus.json": expected_directory / "corpus.json",
        "corpus.f32": expected_directory / "corpus.f32",
        "corpus_clean.f32": expected_directory / "corpus_clean.f32",
    }
    for name, path in paths.items():
        _verify_file_record(
            path,
            files[name],
            name=f"start_probe/{name}",
        )
    manifest = _read_json(paths["corpus.json"])
    items = manifest.get("items")
    expected_count = target_per_class * len(classes)
    expected_binary_bytes = (
        expected_count * REQUIRED_CAPTURE_LENGTHS[0] * 2 * 4
    )
    if (
        manifest.get("sampleCount") != REQUIRED_CAPTURE_LENGTHS[0]
        or manifest.get("format") != "cf32le-interleaved"
        or manifest.get("corpusSeed") != release_seed
        or manifest.get("targetPerClass") != target_per_class
        or manifest.get("exactTargetPerClass") is not True
        or manifest.get("hasCleanPairs") is not True
        or manifest.get("signalLabRoot") != str(signal_lab_root)
        or manifest.get("classes") != list(classes)
        or manifest.get("count") != expected_count
        or not isinstance(items, list)
        or len(items) != expected_count
        or any(not isinstance(item, Mapping) for item in items)
    ):
        raise ValueError("release start_probe corpus manifest is invalid")
    if (
        paths["corpus.f32"].stat().st_size != expected_binary_bytes
        or paths["corpus_clean.f32"].stat().st_size
        != expected_binary_bytes
    ):
        raise ValueError("release start_probe binary geometry is invalid")
    class_counts = {
        name: sum(item.get("cls") == name for item in items)
        for name in classes
    }
    if any(count != target_per_class for count in class_counts.values()):
        raise ValueError("release start_probe class counts are not exact")
    if any(
        _finite_number(item.get("cleanPower"), "start_probe.cleanPower")
        <= 1e-9
        for item in items
    ):
        raise ValueError("release start_probe contains an unoccupied row")
    longest_items = longest.manifest.get("items")
    start_fields = (
        "cls",
        "profile",
        "sampleRateHz",
        "bandwidthHz",
        "startSampleIndex",
    )
    if (
        not isinstance(longest_items, list)
        or len(longest_items) != len(items)
        or any(
            any(probe.get(field) != final.get(field) for field in start_fields)
            for probe, final in zip(items, longest_items)
        )
    ):
        raise ValueError(
            "longest corpus rows do not preserve the unscored occupied starts"
        )
    if longest.manifest.get("matchedStartsFrom") != str(paths["corpus.json"]):
        raise ValueError("longest corpus does not bind its start-probe manifest")
    return {
        "scored": False,
        "purpose": expected_purpose,
        "capture_length": REQUIRED_CAPTURE_LENGTHS[0],
        "row_count": expected_count,
        "class_counts": class_counts,
        "files": dict(files),
        "matched_longest_rows": True,
        "all_probe_rows_occupied": True,
        "passes": True,
    }


def load_release_suite(
    release_root: Path,
    *,
    candidate_override: Path | None = None,
) -> ReleaseSuite:
    """Load and cryptographically verify a fresh suite without model inference."""
    root = Path(release_root).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"release root must be a regular directory: {root}")
    if _is_below(root, LIVE_CORPUS) or _is_below(LIVE_CORPUS, root):
        raise ValueError("release root aliases, contains, or is inside the live corpus")

    intent_path = root / "RELEASE_INTENT.json"
    manifest_path = root / "RELEASE_MANIFEST.json"
    intent = _read_json(intent_path)
    release_manifest = _read_json(manifest_path)
    if intent.get("protocol") != RELEASE_PROTOCOL or release_manifest.get(
        "protocol"
    ) != RELEASE_PROTOCOL:
        raise ValueError("release protocol is invalid")
    if intent.get("status") != "in_progress":
        raise ValueError("RELEASE_INTENT must preserve its pre-generation status")
    if release_manifest.get("status") != "complete":
        raise ValueError("release manifest is not complete")
    if intent.get("development_data_used") is not False or release_manifest.get(
        "development_data_used"
    ) is not False:
        raise ValueError("release suite may not declare development-data use")
    _validate_predeclared_protocol(intent.get("evaluation_protocol"))
    _validate_predeclared_protocol(release_manifest.get("evaluation_protocol"))

    intent_hash = str(release_manifest.get("release_intent_sha256", "")).lower()
    if intent_hash != _sha256(intent_path):
        raise ValueError("RELEASE_MANIFEST does not bind the exact RELEASE_INTENT bytes")

    for key in (
        "protocol",
        "development_data_used",
        "candidate_path",
        "candidate_sha256",
        "release_seed",
        "capture_lengths",
        "target_per_class",
        "exact_target_per_class",
        "evaluation_protocol",
        "source_sha256",
        "dependency_provenance",
        "runtime_provenance",
        "started_at",
    ):
        if release_manifest.get(key) != intent.get(key):
            raise ValueError(f"release manifest changed precommitted intent field {key!r}")

    source_records = intent.get("source_sha256")
    if (
        not isinstance(source_records, Mapping)
        or set(source_records)
        != {
            "launcher",
            "corpus_generator",
            "prefix_deriver",
            "tsx_package",
        }
    ):
        raise ValueError("intent source_sha256 is missing")
    expected_source_paths = {
        "launcher": LAUNCHER,
        "corpus_generator": CORPUS_GENERATOR,
        "prefix_deriver": PREFIX_DERIVER,
    }
    for name, expected_path in expected_source_paths.items():
        record = source_records.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"intent source record {name!r} is missing")
        path = Path(str(record.get("path", ""))).expanduser().resolve()
        if path != expected_path:
            raise ValueError(f"intent source path {name!r} is not the release source")
        if str(record.get("sha256", "")).lower() != _sha256(path):
            raise ValueError(f"intent source {name!r} changed after generation")
    if source_records.get("tsx_package") != "tsx@4.20.3":
        raise ValueError("release intent did not pin the expected tsx package")
    if release_manifest.get("source_sha256") != source_records:
        raise ValueError("release manifest source records differ from intent")
    dependency_report = _validate_dependency_provenance(
        intent.get("dependency_provenance"),
        intent.get("runtime_provenance"),
    )
    if (
        release_manifest.get("dependency_provenance")
        != intent.get("dependency_provenance")
        or release_manifest.get("runtime_provenance")
        != intent.get("runtime_provenance")
    ):
        raise ValueError(
            "release manifest dependency/runtime provenance differs from intent"
        )
    signal_lab_root = Path(
        dependency_report["signallab"]["root"]
    )

    candidate_path = Path(str(intent.get("candidate_path", ""))).expanduser().resolve()
    _regular_file(candidate_path, name="candidate runtime bundle")
    if _is_below(candidate_path, root):
        raise ValueError("candidate must be frozen outside the release output root")
    if candidate_override is not None:
        override = Path(candidate_override).expanduser().resolve()
        if override != candidate_path:
            raise ValueError("candidate override differs from RELEASE_INTENT candidate_path")
    candidate_hash = str(intent.get("candidate_sha256", "")).lower()
    if len(candidate_hash) != 64 or candidate_hash != _sha256(candidate_path):
        raise ValueError("candidate runtime SHA-256 does not match RELEASE_INTENT")

    release_seed = _integer(intent.get("release_seed"), "release_seed")
    target_per_class = _integer(
        intent.get("target_per_class"),
        "target_per_class",
        MIN_TARGET_PER_CLASS,
    )
    if intent.get("exact_target_per_class") is not True:
        raise ValueError("release suite must use exact per-class targeting")
    lengths_value = intent.get("capture_lengths")
    if (
        not isinstance(lengths_value, list)
        or lengths_value != list(REQUIRED_CAPTURE_LENGTHS)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 64
            for value in lengths_value
        )
    ):
        raise ValueError(
            "capture_lengths must equal the frozen required release length suite"
        )
    lengths = [int(value) for value in lengths_value]

    corpus_records = release_manifest.get("corpora")
    if not isinstance(corpus_records, list) or len(corpus_records) != len(lengths):
        raise ValueError("release manifest corpus list is incomplete")
    by_length: dict[int, Mapping[str, Any]] = {}
    for record in corpus_records:
        if not isinstance(record, Mapping):
            raise ValueError("each release corpus record must be an object")
        length = _integer(record.get("capture_length"), "capture_length", 64)
        if length in by_length:
            raise ValueError(f"duplicate release corpus length {length}")
        by_length[length] = record
    if sorted(by_length) != lengths:
        raise ValueError("release corpus lengths differ from precommitted lengths")

    corpora: dict[int, CorpusSpec] = {}
    reference_items: list[Mapping[str, Any]] | None = None
    classes: tuple[str, ...] | None = None
    reference_keys: tuple[str, ...] | None = None
    for length in lengths:
        record = by_length[length]
        expected_derivation = (
            "generated_once_at_longest_length"
            if length == lengths[-1]
            else "bit_exact_row_prefix_of_longest"
        )
        if record.get("derivation") != expected_derivation:
            raise ValueError(
                f"release corpus n{length} derivation declaration is invalid"
            )
        directory_input = Path(
            str(record.get("directory", ""))
        ).expanduser()
        expected_directory = root / f"n{length}"
        if directory_input.is_symlink() or expected_directory.is_symlink():
            raise ValueError(f"release corpus n{length} directory may not be a symlink")
        directory = directory_input.resolve()
        if directory != expected_directory:
            raise ValueError(f"release corpus n{length} escapes its sealed directory")
        files = record.get("files")
        if not isinstance(files, Mapping) or set(files) != {
            "corpus.json",
            "corpus.f32",
            "corpus_clean.f32",
        }:
            raise ValueError(f"release corpus n{length} file records are incomplete")
        corpus_manifest_path = directory / "corpus.json"
        raw_path = directory / "corpus.f32"
        clean_path = directory / "corpus_clean.f32"
        _verify_file_record(
            corpus_manifest_path,
            files["corpus.json"],
            name=f"n{length}/corpus.json",
        )
        _verify_file_record(
            raw_path,
            files["corpus.f32"],
            name=f"n{length}/corpus.f32",
        )
        _verify_file_record(
            clean_path,
            files["corpus_clean.f32"],
            name=f"n{length}/corpus_clean.f32",
        )
        corpus_manifest = _read_json(corpus_manifest_path)
        count = _integer(corpus_manifest.get("count"), f"n{length}.count", 1)
        if (
            corpus_manifest.get("format") != "cf32le-interleaved"
            or corpus_manifest.get("sampleCount") != length
            or corpus_manifest.get("corpusSeed") != release_seed
            or corpus_manifest.get("targetPerClass") != target_per_class
            or corpus_manifest.get("exactTargetPerClass") is not True
            or corpus_manifest.get("hasCleanPairs") is not True
            or corpus_manifest.get("signalLabRoot") != str(signal_lab_root)
        ):
            raise ValueError(
                f"n{length} corpus geometry/seed/target/format is invalid"
            )
        items = corpus_manifest.get("items")
        corpus_classes = corpus_manifest.get("classes")
        if (
            not isinstance(items, list)
            or len(items) != count
            or any(not isinstance(item, Mapping) for item in items)
            or not isinstance(corpus_classes, list)
            or not corpus_classes
            or any(not isinstance(name, str) or not name for name in corpus_classes)
            or corpus_classes != sorted(set(corpus_classes))
        ):
            raise ValueError(f"n{length} corpus classes/items are malformed")
        class_counts = {
            name: sum(item.get("cls") == name for item in items)
            for name in corpus_classes
        }
        if any(value != target_per_class for value in class_counts.values()):
            raise ValueError(
                f"n{length} class counts differ from exact target "
                f"{target_per_class}: {class_counts}"
            )
        expected_bytes = count * length * 2 * np.dtype("<f4").itemsize
        if raw_path.stat().st_size != expected_bytes or clean_path.stat().st_size != expected_bytes:
            raise ValueError(f"n{length} raw byte size disagrees with corpus geometry")
        keys = tuple(_row_keys(items))
        if reference_items is None:
            reference_items = items
            reference_keys = keys
            classes = tuple(corpus_classes)
        else:
            if tuple(corpus_classes) != classes or keys != reference_keys:
                raise ValueError(
                    f"n{length} rows are not metadata-matched to the reference corpus"
                )
        corpora[length] = CorpusSpec(
            capture_length=length,
            directory=directory,
            manifest_path=corpus_manifest_path,
            raw_path=raw_path,
            clean_path=clean_path,
            manifest=corpus_manifest,
            row_keys=keys,
        )

    assert reference_items is not None and classes is not None
    start_probe_report = _validate_unscored_start_probe(
        release_manifest,
        root=root,
        longest=corpora[lengths[-1]],
        classes=classes,
        release_seed=release_seed,
        target_per_class=target_per_class,
        signal_lab_root=signal_lab_root,
    )
    prefix_nesting_report = verify_prefix_nesting(corpora)
    prefix_nesting_report["derivation_records"] = (
        validate_prefix_derivation_records(corpora)
    )
    support, query, split_report = predeclared_five_shot_split(
        reference_items,
        classes,
        release_seed,
    )
    return ReleaseSuite(
        root=root,
        intent_path=intent_path,
        manifest_path=manifest_path,
        intent=intent,
        release_manifest=release_manifest,
        candidate_path=candidate_path,
        candidate_sha256=candidate_hash,
        release_seed=release_seed,
        classes=classes,
        corpora=corpora,
        support_indices=support,
        query_indices=query,
        split_report=split_report,
        prefix_nesting_report=prefix_nesting_report,
        dependency_report=dependency_report,
        start_probe_report=start_probe_report,
    )


@dataclass
class FrozenRuntime:
    real: InvariantPatchCNN
    complex_branch: InvariantPatchCNN
    fusion: CenteredInvariantFusion
    real_config: InvariantPatchConfig
    complex_config: InvariantPatchConfig
    feature_mean: np.ndarray
    feature_std: np.ndarray
    classes: tuple[str, ...]
    prototypes: np.ndarray
    lof_components: tuple[tuple[str, float, KnownOnlyLOFOpenSet], ...]
    unknown_threshold: float
    target_frac: float
    bundle: Mapping[str, Any]


def _tensor_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def load_frozen_runtime(
    path: Path,
    *,
    expected_sha256: str,
    expected_classes: Sequence[str],
    device: torch.device,
) -> FrozenRuntime:
    """Strictly load schema-2 model and frozen rejection assets."""
    bundle_path = Path(path).expanduser().resolve()
    _regular_file(bundle_path, name="runtime bundle")
    actual_hash = _sha256(bundle_path)
    if actual_hash != expected_sha256:
        raise ValueError(
            f"runtime bundle SHA mismatch: expected {expected_sha256}, got {actual_hash}"
        )
    bundle = load_runtime_bundle(bundle_path)
    if bundle["schema"] != 2 or bundle["kind"] != "invariant_centered_fusion_classifier":
        raise ValueError("release evaluator supports only explicit runtime bundle schema 2")
    if tuple(bundle["classes"]) != tuple(expected_classes):
        raise ValueError("runtime classes differ from sealed release classes")

    provenance = bundle["provenance"]
    source_hashes_all = provenance.get("source_hashes")
    if not isinstance(source_hashes_all, Mapping):
        raise ValueError("runtime source-hash provenance is absent")
    for name, expected_hash in source_hashes_all.items():
        if str(name).startswith("source_"):
            # The two state dicts and dev report have already been embedded and
            # validated inside the frozen bundle; release inference does not read
            # their former source paths.
            continue
        source_path = (
            REPO / str(name)
            if str(name).startswith("training/")
            else HERE / str(name)
        )
        if (
            not source_path.is_file()
            or source_path.is_symlink()
            or _sha256(source_path) != expected_hash
        ):
            raise ValueError(f"runtime source drift: {name}")
    contract = provenance.get("cache_contract")
    if (
        not isinstance(contract, Mapping)
        or contract.get("frontend") != "invariant-patch-v1"
        or contract.get("config", {}).get("patch_length")
        != bundle["real_config"]["patch_length"]
        or contract.get("config", {}).get("patch_count")
        != bundle["real_config"]["patch_count"]
    ):
        raise ValueError("runtime preprocessing contract is absent or inconsistent")
    current_sources = {
        "training/preprocess.py": TRAINING / "preprocess.py",
        "training/invariant_patch_preprocess.py": (
            TRAINING / "invariant_patch_preprocess.py"
        ),
        "training/zplane_ab/v2_full_variation/invariant_patch_data.py": (
            HERE / "invariant_patch_data.py"
        ),
    }
    source_hashes = contract.get("source_sha256")
    if not isinstance(source_hashes, Mapping):
        raise ValueError("runtime preprocessing source hashes are absent")
    for name, source_path in current_sources.items():
        if source_hashes.get(name) != _sha256(source_path):
            raise ValueError(f"runtime preprocessing source drift: {name}")

    real_config = InvariantPatchConfig(**dict(bundle["real_config"])).validate()
    complex_config = InvariantPatchConfig(
        **dict(bundle["complex_config"])
    ).validate()
    real = InvariantPatchCNN(real_config)
    complex_branch = InvariantPatchCNN(complex_config)
    real.load_state_dict(bundle["real_state_dict"], strict=True)
    complex_branch.load_state_dict(bundle["complex_state_dict"], strict=True)
    fusion = CenteredInvariantFusion(
        real,
        complex_branch,
        bundle["real_center"],
        bundle["complex_center"],
        alpha_real=bundle["alpha_real"],
        alpha_complex=bundle["alpha_complex"],
        weight_real=bundle["weight_real"],
        eps=bundle["eps"],
    ).to(device).eval()
    real = fusion.real_branch
    complex_branch = fusion.complex_branch

    open_set = bundle["open_set"]
    components: list[tuple[str, float, KnownOnlyLOFOpenSet]] = []
    branches: set[str] = set()
    for component in open_set["components"]:
        branch = str(component["branch_source"])
        if branch not in {"real", "complex"} or branch in branches:
            raise ValueError("v2 weighted LOF must contain unique real/complex branches")
        branches.add(branch)
        components.append(
            (
                branch,
                float(component["weight"]),
                KnownOnlyLOFOpenSet(
                    embedding_reference=_tensor_numpy(
                        component["embedding_reference"]
                    ),
                    embedding_mean=_tensor_numpy(component["embedding_mean"]),
                    embedding_scale=_tensor_numpy(component["embedding_scale"]),
                    reference_k_distance=_tensor_numpy(
                        component["reference_k_distance"]
                    ),
                    reference_local_density=_tensor_numpy(
                        component["reference_local_density"]
                    ),
                    calibration=_tensor_numpy(
                        component["sorted_calibration_raw"]
                    ),
                    calibration_scores=_tensor_numpy(
                        component["calibration_rank_scores"]
                    ),
                    neighbors=int(component["neighbors"]),
                ),
            )
        )
    if branches != {"real", "complex"}:
        raise ValueError("v2 weighted LOF must contain real and complex components")
    weights = [weight for _branch, weight, _scorer in components]
    if any(weight <= 0.0 for weight in weights) or not math.isclose(
        sum(weights), 1.0, abs_tol=1e-12, rel_tol=0.0
    ):
        raise ValueError("frozen LOF weights must be positive and sum to one")
    if open_set.get("selection_or_novelty_used_in_density_or_threshold_fit") is not False:
        raise ValueError("frozen LOF references/threshold used forbidden populations")

    return FrozenRuntime(
        real=real,
        complex_branch=complex_branch,
        fusion=fusion,
        real_config=real_config,
        complex_config=complex_config,
        feature_mean=_tensor_numpy(bundle["feature_mean"]).astype(np.float32),
        feature_std=_tensor_numpy(bundle["feature_std"]).astype(np.float32),
        classes=tuple(bundle["classes"]),
        prototypes=_tensor_numpy(bundle["prototypes"]).astype(np.float32),
        lof_components=tuple(components),
        unknown_threshold=float(bundle["unknown_threshold"]),
        target_frac=float(contract["config"]["target_frac"]),
        bundle=bundle,
    )


def _embed_preprocessed(
    runtime: FrozenRuntime,
    packed: np.ndarray,
    raw_features: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    features = (
        (np.asarray(raw_features, dtype=np.float32) - runtime.feature_mean)
        / runtime.feature_std
    ).astype(np.float32)
    waveforms = np.asarray(packed, dtype=np.float32)
    output: dict[str, list[np.ndarray]] = {
        "real": [],
        "complex": [],
        "fusion": [],
    }
    with torch.no_grad():
        for start in range(0, len(waveforms), batch_size):
            x = torch.from_numpy(
                np.array(waveforms[start : start + batch_size], copy=True)
            ).to(device)
            f = torch.from_numpy(
                np.array(features[start : start + batch_size], copy=True)
            ).to(device)
            real = runtime.real(x, f)
            complex_embedding = runtime.complex_branch(x, f)
            centered_real = _fusion_safe_unit(
                real
                - runtime.fusion.alpha_real
                * runtime.fusion.real_center.unsqueeze(0),
                runtime.fusion.eps,
            )
            centered_complex = _fusion_safe_unit(
                complex_embedding
                - runtime.fusion.alpha_complex
                * runtime.fusion.complex_center.unsqueeze(0),
                runtime.fusion.eps,
            )
            fused = _fusion_safe_unit(
                torch.cat(
                    (
                        torch.sqrt(runtime.fusion.weight_real)
                        * centered_real,
                        torch.sqrt(1.0 - runtime.fusion.weight_real)
                        * centered_complex,
                    ),
                    dim=-1,
                ),
                runtime.fusion.eps,
            )
            output["real"].append(real.detach().cpu().numpy())
            output["complex"].append(complex_embedding.detach().cpu().numpy())
            output["fusion"].append(fused.detach().cpu().numpy())
    result = {
        name: np.concatenate(parts).astype(np.float32, copy=False)
        for name, parts in output.items()
    }
    if any(not np.isfinite(value).all() for value in result.values()):
        raise RuntimeError("runtime produced non-finite embeddings")
    return result


def _preprocess_iq_rows(
    captures: Sequence[np.ndarray],
    runtime: FrozenRuntime,
) -> tuple[np.ndarray, np.ndarray]:
    packed = np.empty(
        (
            len(captures),
            2,
            runtime.real_config.patch_count * runtime.real_config.patch_length,
        ),
        dtype=np.float32,
    )
    features = np.empty(
        (len(captures), runtime.real_config.n_features),
        dtype=np.float32,
    )
    for row, raw in enumerate(captures):
        packed[row], features[row], _context = invariant_preprocess.preprocess(
            raw,
            patch_length=runtime.real_config.patch_length,
            patch_count=runtime.real_config.patch_count,
            target_frac=runtime.target_frac,
        )
    return packed, features


def _raw_rows(spec: CorpusSpec, indices: np.ndarray) -> list[np.ndarray]:
    count = int(spec.manifest["count"])
    raw = np.memmap(
        spec.raw_path,
        dtype="<f4",
        mode="r",
        shape=(count, spec.capture_length, 2),
    )
    captures = []
    for index in np.asarray(indices, dtype=np.int64):
        pair = np.asarray(raw[int(index)], dtype=np.float64)
        captures.append(pair[:, 0] + 1j * pair[:, 1])
    del raw
    return captures


def _embed_corpus(
    spec: CorpusSpec,
    runtime: FrozenRuntime,
    device: torch.device,
) -> dict[str, np.ndarray]:
    captures = _raw_rows(
        spec,
        np.arange(int(spec.manifest["count"]), dtype=np.int64),
    )
    packed, features = _preprocess_iq_rows(captures, runtime)
    return _embed_preprocessed(runtime, packed, features, device)


def _nearest(
    embeddings: np.ndarray,
    prototypes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(embeddings, dtype=np.float64)
    reference = np.asarray(prototypes, dtype=np.float64)
    # Keep this contraction in NumPy's explicit einsum loop. On macOS,
    # Accelerate-backed ``@`` can report stale floating-point status flags
    # (divide-by-zero/overflow/invalid) even when both finite, unit-scale inputs
    # and the resulting dot products are finite. The explicit contraction is
    # numerically equivalent here and preserves warnings-as-errors as a useful
    # fail-closed check for genuine non-finite values.
    cross = np.einsum(
        "nd,kd->nk",
        value,
        reference,
        optimize=False,
    )
    distance = (
        np.sum(value * value, axis=1)[:, None]
        + np.sum(reference * reference, axis=1)[None, :]
        - 2.0 * cross
    )
    distance = np.maximum(distance, 0.0)
    return np.argmin(distance, axis=1), distance


def _balanced_accuracy(
    prediction: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
) -> float:
    recalls = []
    for class_index in range(n_classes):
        mask = labels == class_index
        if not mask.any():
            raise ValueError(f"evaluation population omits class {class_index}")
        recalls.append(float(np.mean(prediction[mask] == class_index)))
    return float(np.mean(recalls))


def _mean_pairwise_cosine(embeddings: np.ndarray) -> float:
    value = np.asarray(embeddings, dtype=np.float64)
    if len(value) < 2:
        raise ValueError("pairwise cosine needs at least two rows")
    value = value / np.maximum(
        np.linalg.norm(value, axis=1, keepdims=True),
        1e-12,
    )
    total = float(np.sum(np.sum(value, axis=0) ** 2) - len(value))
    return total / (len(value) * (len(value) - 1))


def _classification_subset(
    prediction: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    classes: Sequence[str],
    *,
    require_all_classes: bool,
) -> dict[str, Any]:
    selected_prediction = prediction[mask]
    selected_labels = labels[mask]
    if not len(selected_labels):
        raise ValueError("classification subset is empty")
    present = set(int(value) for value in np.unique(selected_labels))
    expected = set(range(len(classes)))
    if require_all_classes and present != expected:
        missing = [classes[index] for index in sorted(expected - present)]
        raise ValueError(f"classification subset omits classes: {missing}")
    recalls: dict[str, float | None] = {}
    counts: dict[str, int] = {}
    present_recalls = []
    for index, name in enumerate(classes):
        class_mask = selected_labels == index
        count = int(class_mask.sum())
        counts[name] = count
        if count:
            recall = float(
                np.mean(selected_prediction[class_mask] == index)
            )
            recalls[name] = recall
            present_recalls.append(recall)
        else:
            recalls[name] = None
    return {
        "n": int(len(selected_labels)),
        "accuracy": float(
            np.mean(selected_prediction == selected_labels)
        ),
        "balanced_accuracy_present_classes": float(np.mean(present_recalls)),
        "class_counts": counts,
        "per_class_recall": recalls,
        "all_classes_present": present == expected,
    }


def _closed_report(
    embeddings: np.ndarray,
    labels: np.ndarray,
    snr_db: np.ndarray,
    impaired: np.ndarray,
    prototypes: np.ndarray,
    classes: Sequence[str],
) -> tuple[dict[str, Any], np.ndarray]:
    if impaired.shape != labels.shape or impaired.dtype != np.bool_:
        raise ValueError("impaired mask must be a boolean vector matching labels")
    prediction, distance = _nearest(embeddings, prototypes)
    n_classes = len(classes)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for truth, guess in zip(labels, prediction):
        confusion[int(truth), int(guess)] += 1
    recalls = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)
    high = snr_db >= HIGH_SNR_DB
    high_report = _classification_subset(
        prediction,
        labels,
        high,
        classes,
        require_all_classes=True,
    )
    low_report = _classification_subset(
        prediction,
        labels,
        ~high,
        classes,
        require_all_classes=True,
    )
    clean_report = _classification_subset(
        prediction,
        labels,
        ~impaired,
        classes,
        require_all_classes=True,
    )
    impaired_report = _classification_subset(
        prediction,
        labels,
        impaired,
        classes,
        require_all_classes=True,
    )
    accuracy = float(np.mean(prediction == labels))
    balanced = float(np.mean(recalls))
    return (
        {
            "n": int(len(labels)),
            "accuracy": accuracy,
            "balanced_accuracy": balanced,
            "per_class_recall": {
                classes[index]: float(recalls[index])
                for index in range(n_classes)
            },
            "confusion": confusion.tolist(),
            "clean_accuracy": clean_report["accuracy"],
            "impaired_accuracy": impaired_report["accuracy"],
            "clean_subset": clean_report,
            "impaired_subset": impaired_report,
            "family": {
                "accuracy": accuracy,
                "balanced_accuracy": balanced,
                "mapping": {name: name for name in classes},
                "reason": "sealed corpus labels are already broad RF families",
            },
            "high_snr": {
                "definition": f"manifest snrDb >= {HIGH_SNR_DB:g}",
                "threshold_db": HIGH_SNR_DB,
                **high_report,
            },
            "low_snr": {
                "definition": f"manifest snrDb < {HIGH_SNR_DB:g}",
                "threshold_db": HIGH_SNR_DB,
                **low_report,
            },
            "mean_pairwise_cosine": _mean_pairwise_cosine(embeddings),
            "nearest_distance_median": float(
                np.median(np.min(distance, axis=1))
            ),
        },
        prediction,
    )


def _simple_closed_report(
    embeddings: np.ndarray,
    labels: np.ndarray,
    prototypes: np.ndarray,
    classes: Sequence[str],
) -> tuple[dict[str, Any], np.ndarray]:
    prediction, distance = _nearest(embeddings, prototypes)
    recalls = {}
    for index, name in enumerate(classes):
        mask = labels == index
        if not mask.any():
            raise ValueError(f"closed population omits class {name!r}")
        recalls[name] = float(np.mean(prediction[mask] == index))
    return (
        {
            "n": int(len(labels)),
            "accuracy": float(np.mean(prediction == labels)),
            "balanced_accuracy": float(np.mean(list(recalls.values()))),
            "per_class_recall": recalls,
            "mean_pairwise_cosine": _mean_pairwise_cosine(embeddings),
            "nearest_distance_median": float(
                np.median(np.min(distance, axis=1))
            ),
        },
        prediction,
    )


def _paired_invariance_report(
    current_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    current_prediction: np.ndarray,
    reference_prediction: np.ndarray,
    labels: np.ndarray,
    snr_db: np.ndarray,
    classes: Sequence[str],
) -> dict[str, Any]:
    current = np.asarray(current_embeddings, dtype=np.float64)
    reference = np.asarray(reference_embeddings, dtype=np.float64)
    if (
        current.shape != reference.shape
        or current.ndim != 2
        or current_prediction.shape != labels.shape
        or reference_prediction.shape != labels.shape
        or snr_db.shape != labels.shape
    ):
        raise ValueError("paired invariance arrays have incompatible shapes")
    current = current / np.maximum(
        np.linalg.norm(current, axis=1, keepdims=True),
        1e-12,
    )
    reference = reference / np.maximum(
        np.linalg.norm(reference, axis=1, keepdims=True),
        1e-12,
    )
    cosine = np.clip(np.sum(current * reference, axis=1), -1.0, 1.0)
    agreement = current_prediction == reference_prediction

    def subset(mask: np.ndarray) -> dict[str, Any]:
        if not mask.any():
            return {
                "n": 0,
                "prediction_agreement": None,
                "embedding_cosine_mean": None,
                "embedding_cosine_p05": None,
                "per_class": {
                    name: {
                        "n": 0,
                        "prediction_agreement": None,
                        "embedding_cosine_mean": None,
                    }
                    for name in classes
                },
            }
        per_class = {}
        for index, name in enumerate(classes):
            class_mask = mask & (labels == index)
            per_class[name] = {
                "n": int(class_mask.sum()),
                "prediction_agreement": (
                    float(np.mean(agreement[class_mask]))
                    if class_mask.any()
                    else None
                ),
                "embedding_cosine_mean": (
                    float(np.mean(cosine[class_mask]))
                    if class_mask.any()
                    else None
                ),
            }
        return {
            "n": int(mask.sum()),
            "prediction_agreement": float(np.mean(agreement[mask])),
            "embedding_cosine_mean": float(np.mean(cosine[mask])),
            "embedding_cosine_p05": float(np.quantile(cosine[mask], 0.05)),
            "per_class": per_class,
        }

    all_rows = np.ones(len(labels), dtype=bool)
    return {
        **subset(all_rows),
        "low_snr": {
            "definition": f"manifest snrDb < {HIGH_SNR_DB:g}",
            **subset(snr_db < HIGH_SNR_DB),
        },
        "high_snr": {
            "definition": f"manifest snrDb >= {HIGH_SNR_DB:g}",
            **subset(snr_db >= HIGH_SNR_DB),
        },
    }


def _five_shot_report(
    enrollment_embeddings: np.ndarray,
    evaluation_embeddings: np.ndarray,
    labels: np.ndarray,
    support_indices: np.ndarray,
    query_indices: np.ndarray,
    classes: Sequence[str],
) -> dict[str, Any]:
    prototypes = []
    for class_index in range(len(classes)):
        selected = support_indices[labels[support_indices] == class_index]
        if len(selected) != FIVE_SHOT_K:
            raise ValueError(
                f"predeclared support for class {classes[class_index]!r} "
                f"is not exactly {FIVE_SHOT_K}"
            )
        prototypes.append(enrollment_embeddings[selected].mean(axis=0))
    prototype_array = np.stack(prototypes).astype(np.float32)
    query_labels = labels[query_indices]
    prediction, _ = _nearest(
        evaluation_embeddings[query_indices],
        prototype_array,
    )
    recalls = {
        classes[index]: float(
            np.mean(prediction[query_labels == index] == index)
        )
        for index in range(len(classes))
    }
    return {
        "k": FIVE_SHOT_K,
        "support_rows": int(len(support_indices)),
        "query_rows": int(len(query_indices)),
        "accuracy": float(np.mean(prediction == query_labels)),
        "balanced_accuracy": float(np.mean(list(recalls.values()))),
        "per_class_recall": recalls,
        "claim_scope": (
            "prototype re-enrollment for encoder-known broad classes; "
            "no encoder or rejector parameter changes"
        ),
        "prototype_source": (
            f"predeclared support rows embedded at n={MATCHED_CAPTURE_LENGTH}; "
            "the same prototypes are reused unchanged at every query length"
        ),
    }


def _auroc(novelty_scores: np.ndarray, known_scores: np.ndarray) -> float:
    positive = np.asarray(novelty_scores, dtype=np.float64)
    negative = np.asarray(known_scores, dtype=np.float64)
    if (
        positive.ndim != 1
        or negative.ndim != 1
        or not len(positive)
        or not len(negative)
        or not np.isfinite(positive).all()
        or not np.isfinite(negative).all()
    ):
        raise ValueError("AUROC inputs must be finite non-empty vectors")
    values = np.concatenate((positive, negative))
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    n_pos, n_neg = len(positive), len(negative)
    return float(
        (ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0)
        / (n_pos * n_neg)
    )


def _weighted_lof_score(
    runtime: FrozenRuntime,
    embeddings: Mapping[str, np.ndarray],
) -> np.ndarray:
    score: np.ndarray | None = None
    for branch, weight, scorer in runtime.lof_components:
        component = weight * scorer.score(embeddings[branch])
        score = component if score is None else score + component
    if score is None or not np.isfinite(score).all():
        raise RuntimeError("frozen weighted LOF produced invalid scores")
    return score


def _novelty_seed(family: str) -> int:
    if family not in NOVELTY_FAMILIES:
        raise ValueError(f"unfrozen novelty family {family!r}")
    payload = f"{NOVELTY_SEED}\0{family}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _novelty_embeddings_by_length(
    lengths: Sequence[int],
    runtime: FrozenRuntime,
    device: torch.device,
) -> tuple[dict[int, dict[str, dict[str, np.ndarray]]], dict[str, Any]]:
    """Generate each novelty once at max length and embed exact prefixes."""
    ordered_lengths = sorted(set(int(value) for value in lengths))
    if ordered_lengths != list(REQUIRED_CAPTURE_LENGTHS):
        raise ValueError("novelty lengths differ from the frozen release suite")
    longest = ordered_lengths[-1]
    by_length: dict[int, dict[str, dict[str, np.ndarray]]] = {
        length: {} for length in ordered_lengths
    }
    seed_by_family: dict[str, int] = {}
    longest_bytes_sha256: dict[str, str] = {}
    for family in NOVELTY_FAMILIES:
        seed = _novelty_seed(family)
        seed_by_family[family] = seed
        rng = np.random.default_rng(seed)
        longest_rows = [
            np.asarray(NOVELTY[family](longest, rng), dtype=np.complex128)
            for _ in range(NOVELTY_N_EACH_PER_LENGTH)
        ]
        if any(
            row.shape != (longest,)
            or not np.isfinite(row.real).all()
            or not np.isfinite(row.imag).all()
            for row in longest_rows
        ):
            raise RuntimeError(f"novelty generator {family!r} returned invalid rows")
        digest = hashlib.sha256()
        for row in longest_rows:
            digest.update(
                np.asarray(row, dtype="<c16").tobytes(order="C")
            )
        longest_bytes_sha256[family] = digest.hexdigest()
        for length in ordered_lengths:
            prefixes = [row[:length] for row in longest_rows]
            packed, features = _preprocess_iq_rows(prefixes, runtime)
            by_length[length][family] = _embed_preprocessed(
                runtime,
                packed,
                features,
                device,
            )
        del longest_rows
    return by_length, {
        "base_seed": NOVELTY_SEED,
        "derived_seed_by_family": seed_by_family,
        "families": list(NOVELTY_FAMILIES),
        "rows_per_family": NOVELTY_N_EACH_PER_LENGTH,
        "longest_capture_length": longest,
        "longest_complex128_bytes_sha256": longest_bytes_sha256,
        "prefix_rule": (
            "each shorter novelty observation is the exact leading slice of "
            "the same in-memory maximum-length realization"
        ),
        "recalibration_performed": False,
    }


def _open_report(
    known_embeddings: Mapping[str, np.ndarray],
    novelty_embeddings: Mapping[str, Mapping[str, np.ndarray]],
    runtime: FrozenRuntime,
    novelty_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    known_score = _weighted_lof_score(runtime, known_embeddings)
    family_scores = {
        family: _weighted_lof_score(runtime, novelty_embeddings[family])
        for family in NOVELTY_FAMILIES
    }
    if any(
        len(scores) != NOVELTY_N_EACH_PER_LENGTH
        for scores in family_scores.values()
    ):
        raise RuntimeError(
            "prefix-matched novelty population has an unexpected row count"
        )
    all_novelty = np.concatenate(
        [family_scores[name] for name in NOVELTY_FAMILIES]
    )
    threshold = runtime.unknown_threshold
    return {
        "known_rows": int(len(known_score)),
        "novelty_rows_per_family": NOVELTY_N_EACH_PER_LENGTH,
        "base_seed": NOVELTY_SEED,
        "derived_seed_by_family": dict(
            novelty_provenance["derived_seed_by_family"]
        ),
        "maximum_length_realization_sha256": dict(
            novelty_provenance["longest_complex128_bytes_sha256"]
        ),
        "shorter_observation_rule": novelty_provenance["prefix_rule"],
        "score": "frozen positive weighted sum of enrollment-ranked branch LOF",
        "threshold": threshold,
        "recalibration_performed": False,
        "auroc_overall": _auroc(all_novelty, known_score),
        **{
            f"auroc_{name}": _auroc(family_scores[name], known_score)
            for name in NOVELTY_FAMILIES
        },
        "known_false_unknown_rate": float(np.mean(known_score > threshold)),
        **{
            f"flagged_unknown_{name}": float(
                np.mean(family_scores[name] > threshold)
            )
            for name in NOVELTY_FAMILIES
        },
        "known_score_quantiles": {
            "p50": float(np.quantile(known_score, 0.5)),
            "p95": float(np.quantile(known_score, 0.95)),
        },
    }


def _labels_and_snr(
    spec: CorpusSpec,
    classes: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    class_index = {name: index for index, name in enumerate(classes)}
    labels, snr, impaired = [], [], []
    for item in spec.manifest["items"]:
        name = item.get("cls")
        if name not in class_index:
            raise ValueError(f"unknown corpus class {name!r}")
        labels.append(class_index[name])
        snr.append(_finite_number(item.get("snrDb"), "snrDb"))
        value = item.get("impaired")
        if not isinstance(value, bool):
            raise ValueError("corpus impaired field must be boolean")
        impaired.append(value)
    return (
        np.asarray(labels, dtype=np.int64),
        np.asarray(snr, dtype=np.float64),
        np.asarray(impaired, dtype=np.bool_),
    )


def _scale_eligible_indices(
    spec: CorpusSpec,
    query_indices: np.ndarray,
    classes: Sequence[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    effective_by_factor = {
        factor: (
            (MATCHED_CAPTURE_LENGTH - 1)
            / (
                max(64, int(round(MATCHED_CAPTURE_LENGTH / factor)))
                - 1
            )
        )
        for factor in PHYSICAL_SCALE_FACTORS
    }
    max_effective_factor = max(effective_by_factor.values())
    eligible = []
    counts = {name: 0 for name in classes}
    excluded_counts = {name: 0 for name in classes}
    for index in np.asarray(query_indices, dtype=np.int64):
        item = spec.manifest["items"][int(index)]
        center = abs(_finite_number(item["centreOffsetFrac"], "centreOffsetFrac"))
        bandwidth = _finite_number(item["bandwidthHz"], "bandwidthHz") / _finite_number(
            item["sampleRateHz"], "sampleRateHz"
        )
        class_name = str(item["cls"])
        if (
            max_effective_factor * (center + 0.5 * bandwidth)
            < 0.5 - SCALE_EDGE_MARGIN
        ):
            eligible.append(int(index))
            counts[class_name] += 1
        else:
            excluded_counts[class_name] += 1
    if any(value < SCALE_MIN_PER_CLASS for value in counts.values()):
        raise ValueError(
            "common physical-scale eligibility has fewer than "
            f"{SCALE_MIN_PER_CLASS} query rows for a class: {counts}"
        )
    return np.asarray(eligible, dtype=np.int64), {
        "eligible_by_class": counts,
        "excluded_by_class": excluded_counts,
        "excluded_total": int(sum(excluded_counts.values())),
        "exclusion_reason": (
            "transformed occupied support would meet/cross the predeclared "
            "Nyquist guard for at least one factor"
        ),
        "effective_scale_by_requested_factor": {
            str(factor): value
            for factor, value in effective_by_factor.items()
        },
        "maximum_effective_scale": max_effective_factor,
    }


def _scale_sweep(
    spec: CorpusSpec,
    runtime: FrozenRuntime,
    device: torch.device,
    labels: np.ndarray,
    query_indices: np.ndarray,
) -> dict[str, Any]:
    eligible, eligibility = _scale_eligible_indices(
        spec, query_indices, runtime.classes
    )
    base_captures = _raw_rows(spec, eligible)
    rows = []
    embeddings_by_factor: dict[float, np.ndarray] = {}
    predictions_by_factor: dict[float, np.ndarray] = {}
    wanted = labels[eligible]
    snr = np.asarray(
        [
            _finite_number(
                spec.manifest["items"][int(index)]["snrDb"],
                "snrDb",
            )
            for index in eligible
        ],
        dtype=np.float64,
    )
    for factor in PHYSICAL_SCALE_FACTORS:
        new_length = max(
            64, int(round(MATCHED_CAPTURE_LENGTH / factor))
        )
        effective_factor = (
            (MATCHED_CAPTURE_LENGTH - 1) / (new_length - 1)
        )
        captures = (
            base_captures
            if factor == 1.0
            else [
                production_preprocess.lin_resample(raw, new_length)
                for raw in base_captures
            ]
        )
        packed, features = _preprocess_iq_rows(captures, runtime)
        embeddings = _embed_preprocessed(
            runtime, packed, features, device
        )["fusion"]
        embeddings_by_factor[factor] = embeddings
        closed, prediction = _simple_closed_report(
            embeddings,
            wanted,
            runtime.prototypes,
            runtime.classes,
        )
        predictions_by_factor[factor] = prediction
        transformed_sample_rates = np.asarray(
            [
                _finite_number(
                    spec.manifest["items"][int(index)]["sampleRateHz"],
                    "sampleRateHz",
                )
                / effective_factor
                for index in eligible
            ],
            dtype=np.float64,
        )
        rows.append(
            {
                "factor": factor,
                "effective_factor": effective_factor,
                "raw_length": new_length,
                "transformed_metadata": {
                    "sample_rate_hz_rule": (
                        "original sampleRateHz / effective_factor"
                    ),
                    "sample_rate_hz_min": float(
                        transformed_sample_rates.min()
                    ),
                    "sample_rate_hz_max": float(
                        transformed_sample_rates.max()
                    ),
                    "bandwidth_hz_rule": "unchanged",
                    "centre_offset_fraction_rule": (
                        "original centreOffsetFrac * effective_factor"
                    ),
                },
                "accuracy": closed["accuracy"],
                "balanced_accuracy": closed["balanced_accuracy"],
                "per_class_recall": closed["per_class_recall"],
                "mean_pairwise_cosine": closed["mean_pairwise_cosine"],
            }
        )
    reference = embeddings_by_factor[1.0]
    reference_prediction = predictions_by_factor[1.0]
    for row in rows:
        factor = float(row["factor"])
        row["paired_to_factor1"] = _paired_invariance_report(
            embeddings_by_factor[factor],
            reference,
            predictions_by_factor[factor],
            reference_prediction,
            wanted,
            snr,
            runtime.classes,
        )
    return {
        "definition": (
            "normalized frequencies multiplied by s through deterministic "
            "linear resampling to N/s; common predeclared no-alias subset"
        ),
        "factors": list(PHYSICAL_SCALE_FACTORS),
        "eligible_rows": int(len(eligible)),
        **eligibility,
        "eligible_indices_sha256": hashlib.sha256(
            eligible.astype("<i8", copy=False).tobytes()
        ).hexdigest(),
        "rows": rows,
        "worst_balanced_accuracy": min(
            float(row["balanced_accuracy"]) for row in rows
        ),
        "worst_mean_pairwise_cosine": max(
            float(row["mean_pairwise_cosine"]) for row in rows
        ),
        "worst_prediction_agreement_to_factor1": min(
            float(row["paired_to_factor1"]["prediction_agreement"])
            for row in rows
        ),
        "worst_embedding_cosine_to_factor1": min(
            float(row["paired_to_factor1"]["embedding_cosine_mean"])
            for row in rows
        ),
    }


def _gate(value: float, floor: float, *, comparison: str = "min") -> dict[str, Any]:
    if comparison not in {"min", "max"}:
        raise ValueError("gate comparison must be 'min' or 'max'")
    passes = value >= floor if comparison == "min" else value <= floor
    return {
        "value": float(value),
        "threshold": float(floor),
        "comparison": comparison,
        "passes": bool(passes),
    }


def _assemble_release_gates(
    *,
    candidate_sha256: str,
    expected_candidate_sha256: str,
    prefix_nesting_passes: bool,
    dependency_provenance_passes: bool,
    start_probe_excluded: bool,
    closed_by_length: Mapping[str, Mapping[str, Any]],
    five_shot_by_length: Mapping[str, Mapping[str, Any]],
    open_by_length: Mapping[str, Mapping[str, Any]],
    length_sweep: Mapping[str, Any],
    scale_sweep: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Assemble every predeclared gate from already-computed release reports."""
    if not closed_by_length or not five_shot_by_length or not open_by_length:
        raise ValueError("release gates require non-empty per-length reports")
    worst_closed = min(
        float(row["accuracy"]) for row in closed_by_length.values()
    )
    worst_family = min(
        float(row["family"]["accuracy"]) for row in closed_by_length.values()
    )
    worst_high = min(
        float(row["high_snr"]["accuracy"])
        for row in closed_by_length.values()
    )
    worst_clean = min(
        float(row["clean_accuracy"]) for row in closed_by_length.values()
    )
    worst_five = min(
        float(row["balanced_accuracy"])
        for row in five_shot_by_length.values()
    )
    worst_open_overall = min(
        float(row["auroc_overall"]) for row in open_by_length.values()
    )
    worst_open_noise = min(
        float(row["auroc_noise"]) for row in open_by_length.values()
    )
    worst_open_chirp = min(
        float(row["auroc_chirp"]) for row in open_by_length.values()
    )
    worst_unknown_noise = min(
        float(row["flagged_unknown_noise"])
        for row in open_by_length.values()
    )
    worst_unknown_chirp = min(
        float(row["flagged_unknown_chirp"])
        for row in open_by_length.values()
    )
    worst_known_false_unknown = max(
        float(row["known_false_unknown_rate"])
        for row in open_by_length.values()
    )
    return {
        "prefix_nesting": {
            "value": bool(prefix_nesting_passes),
            "expected": True,
            "passes": bool(prefix_nesting_passes),
        },
        "dependency_provenance": {
            "value": bool(dependency_provenance_passes),
            "expected": True,
            "passes": bool(dependency_provenance_passes),
        },
        "start_probe_excluded": {
            "value": bool(start_probe_excluded),
            "expected": True,
            "passes": bool(start_probe_excluded),
        },
        "candidate_sha_bound": {
            "value": candidate_sha256,
            "expected": expected_candidate_sha256,
            "passes": candidate_sha256 == expected_candidate_sha256,
        },
        "closed_fine_worst_length": _gate(
            worst_closed, GATE_FLOORS["closed_fine"]
        ),
        "closed_family_worst_length": _gate(
            worst_family, GATE_FLOORS["closed_family"]
        ),
        "closed_high_snr_worst_length": _gate(
            worst_high, GATE_FLOORS["closed_high_snr"]
        ),
        "closed_clean_worst_length": _gate(
            worst_clean, GATE_FLOORS["closed_clean"]
        ),
        "five_shot_worst_length_balanced": _gate(
            worst_five, GATE_FLOORS["five_shot"]
        ),
        "open_auroc_overall_worst_length": _gate(
            worst_open_overall, GATE_FLOORS["open_auroc_overall"]
        ),
        "open_auroc_noise_worst_length": _gate(
            worst_open_noise, GATE_FLOORS["open_auroc_noise"]
        ),
        "open_auroc_chirp_worst_length": _gate(
            worst_open_chirp, GATE_FLOORS["open_auroc_chirp"]
        ),
        "open_known_false_unknown_worst_length": _gate(
            worst_known_false_unknown,
            GATE_FLOORS["open_known_false_unknown_max"],
            comparison="max",
        ),
        "open_unknown_recall_noise_worst_length": _gate(
            worst_unknown_noise,
            GATE_FLOORS["open_unknown_recall_noise"],
        ),
        "open_unknown_recall_chirp_worst_length": _gate(
            worst_unknown_chirp,
            GATE_FLOORS["open_unknown_recall_chirp"],
        ),
        "length_worst_balanced_accuracy": _gate(
            float(length_sweep["worst_balanced_accuracy"]),
            GATE_FLOORS["invariant_worst_balanced_accuracy"],
        ),
        "length_max_mean_pairwise_cosine": _gate(
            float(length_sweep["worst_mean_pairwise_cosine"]),
            GATE_FLOORS["invariant_max_mean_pairwise_cosine"],
            comparison="max",
        ),
        "length_worst_prediction_agreement_to_matched": _gate(
            float(length_sweep["worst_prediction_agreement_to_matched"]),
            GATE_FLOORS["length_pair_prediction_agreement"],
        ),
        "length_worst_embedding_cosine_to_matched": _gate(
            float(length_sweep["worst_embedding_cosine_to_matched"]),
            GATE_FLOORS["length_pair_embedding_cosine"],
        ),
        "physical_scale_worst_balanced_accuracy": _gate(
            float(scale_sweep["worst_balanced_accuracy"]),
            GATE_FLOORS["invariant_worst_balanced_accuracy"],
        ),
        "physical_scale_max_mean_pairwise_cosine": _gate(
            float(scale_sweep["worst_mean_pairwise_cosine"]),
            GATE_FLOORS["invariant_max_mean_pairwise_cosine"],
            comparison="max",
        ),
        "physical_scale_worst_prediction_agreement_to_factor1": _gate(
            float(scale_sweep["worst_prediction_agreement_to_factor1"]),
            GATE_FLOORS["scale_pair_prediction_agreement"],
        ),
        "physical_scale_worst_embedding_cosine_to_factor1": _gate(
            float(scale_sweep["worst_embedding_cosine_to_factor1"]),
            GATE_FLOORS["scale_pair_embedding_cosine"],
        ),
    }


def _write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite release evaluation: {path}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link publication preserves O_EXCL semantics for the final name;
        # os.replace would silently overwrite a report created after the initial
        # existence check.
        os.link(temporary, path)
        temporary.unlink()
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def evaluate_release(
    suite: ReleaseSuite,
    runtime: FrozenRuntime,
    device: torch.device,
) -> dict[str, Any]:
    """Run the fully precommitted evaluation without modifying frozen assets."""
    embeddings_by_length: dict[int, dict[str, np.ndarray]] = {}
    closed_by_length: dict[str, Any] = {}
    five_shot_by_length: dict[str, Any] = {}
    open_by_length: dict[str, Any] = {}
    labels_by_length: dict[int, np.ndarray] = {}
    snr_by_length: dict[int, np.ndarray] = {}
    print("[release] generating prefix-matched novelty", flush=True)
    novelty_by_length, novelty_provenance = _novelty_embeddings_by_length(
        list(suite.corpora),
        runtime,
        device,
    )
    for length, spec in sorted(suite.corpora.items()):
        print(f"[release] preprocessing/inference n={length}", flush=True)
        embeddings = _embed_corpus(spec, runtime, device)
        embeddings_by_length[length] = embeddings
        labels, snr, impaired = _labels_and_snr(spec, runtime.classes)
        labels_by_length[length] = labels
        snr_by_length[length] = snr
        query = suite.query_indices
        closed, _prediction = _closed_report(
            embeddings["fusion"][query],
            labels[query],
            snr[query],
            impaired[query],
            runtime.prototypes,
            runtime.classes,
        )
        closed_by_length[str(length)] = closed
        known = {
            branch: embeddings[branch][query]
            for branch in ("real", "complex")
        }
        open_by_length[str(length)] = _open_report(
            known,
            novelty_by_length[length],
            runtime,
            novelty_provenance,
        )

    matched_enrollment = embeddings_by_length[MATCHED_CAPTURE_LENGTH][
        "fusion"
    ]
    matched_labels = labels_by_length[MATCHED_CAPTURE_LENGTH]
    for length in sorted(suite.corpora):
        if not np.array_equal(labels_by_length[length], matched_labels):
            raise AssertionError("matched release label arrays differ by length")
        five_shot_by_length[str(length)] = _five_shot_report(
            matched_enrollment,
            embeddings_by_length[length]["fusion"],
            matched_labels,
            suite.support_indices,
            suite.query_indices,
            runtime.classes,
        )

    reference = embeddings_by_length[MATCHED_CAPTURE_LENGTH]["fusion"][
        suite.query_indices
    ]
    reference_prediction, _ = _nearest(reference, runtime.prototypes)
    length_rows = []
    for length in sorted(suite.corpora):
        current = embeddings_by_length[length]["fusion"][suite.query_indices]
        prediction, _ = _nearest(current, runtime.prototypes)
        closed = closed_by_length[str(length)]
        paired = _paired_invariance_report(
            current,
            reference,
            prediction,
            reference_prediction,
            labels_by_length[length][suite.query_indices],
            snr_by_length[length][suite.query_indices],
            runtime.classes,
        )
        length_rows.append(
            {
                "capture_length": length,
                "accuracy": closed["accuracy"],
                "balanced_accuracy": closed["balanced_accuracy"],
                "per_class_recall": closed["per_class_recall"],
                "mean_pairwise_cosine": closed["mean_pairwise_cosine"],
                "paired_to_matched": paired,
            }
        )
    length_sweep = {
        "matched_capture_length": MATCHED_CAPTURE_LENGTH,
        "matched_rows": int(len(suite.query_indices)),
        "all_classes_required": list(runtime.classes),
        "rows": length_rows,
        "worst_balanced_accuracy": min(
            float(row["balanced_accuracy"]) for row in length_rows
        ),
        "worst_mean_pairwise_cosine": max(
            float(row["mean_pairwise_cosine"]) for row in length_rows
        ),
        "worst_prediction_agreement_to_matched": min(
            float(row["paired_to_matched"]["prediction_agreement"])
            for row in length_rows
        ),
        "worst_embedding_cosine_to_matched": min(
            float(row["paired_to_matched"]["embedding_cosine_mean"])
            for row in length_rows
        ),
    }
    scale_sweep = _scale_sweep(
        suite.corpora[MATCHED_CAPTURE_LENGTH],
        runtime,
        device,
        labels_by_length[MATCHED_CAPTURE_LENGTH],
        suite.query_indices,
    )

    gates = _assemble_release_gates(
        candidate_sha256=suite.candidate_sha256,
        expected_candidate_sha256=suite.release_manifest[
            "candidate_sha256"
        ],
        prefix_nesting_passes=bool(
            suite.prefix_nesting_report["passes"]
        ),
        dependency_provenance_passes=bool(
            suite.dependency_report["passes"]
        ),
        start_probe_excluded=(
            suite.start_probe_report["scored"] is False
            and bool(suite.start_probe_report["passes"])
        ),
        closed_by_length=closed_by_length,
        five_shot_by_length=five_shot_by_length,
        open_by_length=open_by_length,
        length_sweep=length_sweep,
        scale_sweep=scale_sweep,
    )
    all_pass = all(bool(value["passes"]) for value in gates.values())
    return {
        "schema": EVALUATOR_SCHEMA,
        "status": "complete",
        "release_evidence": True,
        "development_data_loaded": False,
        "retraining_performed": False,
        "recalibration_performed": False,
        "candidate": {
            "path": str(suite.candidate_path),
            "sha256": suite.candidate_sha256,
            "runtime_schema": int(runtime.bundle["schema"]),
            "runtime_kind": runtime.bundle["kind"],
            "classes": list(runtime.classes),
            "frozen_assets_used": [
                "branch weights",
                "fusion centers/weights",
                "feature mean/std",
                "fused prototypes",
                "weighted LOF references/ranks/weights/threshold",
            ],
        },
        "closed_per_length": closed_by_length,
        "family_per_length": {
            length: report["family"]
            for length, report in closed_by_length.items()
        },
        "high_snr_per_length": {
            length: report["high_snr"]
            for length, report in closed_by_length.items()
        },
        "low_snr_per_length": {
            length: report["low_snr"]
            for length, report in closed_by_length.items()
        },
        "clean_subset_per_length": {
            length: report["clean_subset"]
            for length, report in closed_by_length.items()
        },
        "impaired_subset_per_length": {
            length: report["impaired_subset"]
            for length, report in closed_by_length.items()
        },
        "five_shot_predeclared_per_length": five_shot_by_length,
        "open_weighted_lof_per_length": open_by_length,
        "matched_length_sweep": length_sweep,
        "physical_scale_sweep": scale_sweep,
        "gates": gates,
        "all_release_gates_pass": all_pass,
        "provenance": {
            "release_root": str(suite.root),
            "release_intent_sha256": _sha256(suite.intent_path),
            "release_manifest_sha256": _sha256(suite.manifest_path),
            "release_seed": suite.release_seed,
            "evaluation_protocol": EXPECTED_EVALUATION_PROTOCOL,
            "predeclared_split": suite.split_report,
            "prefix_nesting": suite.prefix_nesting_report,
            "dependency_provenance": suite.dependency_report,
            "unscored_start_probe": suite.start_probe_report,
            "novelty_prefix_generation": novelty_provenance,
            "candidate_sha256": suite.candidate_sha256,
            "corpus_assets": suite.release_manifest["corpora"],
            "evaluator_path": str(Path(__file__).resolve()),
            "evaluator_sha256": _sha256(Path(__file__).resolve()),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
        },
    }


def resolve_device(name: str) -> torch.device:
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return torch.device("mps")
    if name != "cpu":
        raise ValueError("release evaluator device must be cpu or mps")
    return torch.device("cpu")


def run(args: argparse.Namespace) -> dict[str, Any]:
    suite = load_release_suite(
        Path(args.release_root),
        candidate_override=(
            None if args.candidate is None else Path(args.candidate)
        ),
    )
    device = resolve_device(args.device)
    runtime = load_frozen_runtime(
        suite.candidate_path,
        expected_sha256=suite.candidate_sha256,
        expected_classes=suite.classes,
        device=device,
    )
    report = evaluate_release(suite, runtime, device)
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else suite.root / "RELEASE_EVALUATION.json"
    )
    if not _is_below(output, suite.root):
        raise ValueError("release evaluation output must stay inside release root")
    _write_json_exclusive(output, report)
    print(
        f"[release] gates={'PASS' if report['all_release_gates_pass'] else 'FAIL'} "
        f"-> {output}",
        flush=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--release-root", required=True)
    parser.add_argument(
        "--candidate",
        help="optional assertion; must resolve exactly to RELEASE_INTENT candidate_path",
    )
    parser.add_argument("--output")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
