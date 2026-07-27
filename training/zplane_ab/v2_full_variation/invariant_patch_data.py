"""Leak-safe, versioned data cache for the invariant-patch front end.

This module reconstructs the historical v2 corpus split directly from the raw
manifest and ``corpus.f32``.  It intentionally does not use an existing pool as
an input: old pool indices are validation witnesses only.  After nominal
FFT-edge filtering, the v2 validation population is split once more into the
immutable development-selection and consumed-test halves.  Only the selection
half is preprocessed, cached, or returned.

The public ``load`` result follows the training pool convention:

``xtr/ftr/ytr/imp_tr/tr_idx``
    Edge-safe v2 training rows.
``xen/fen/yen/imp_en/en_idx``
    Edge-safe v2 enrollment rows.
``xva/fva/yva/imp_va/va_idx``
    The edge-safe ground-state *selection* half only.

There is deliberately no API, array, metadata field, or cache file containing
consumed-test identifiers or samples.  Aggregate held-out counts are retained
for auditing, with ``consumed_test_rows_exposed`` fixed at zero.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import glob
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable, Mapping

import numpy as np


HERE = Path(__file__).resolve().parent
TRAINING = HERE.parent.parent
CORPUS = TRAINING / "artifacts" / "signallab-corpus"
CACHE = CORPUS / "_invariant_patch_pools"
LEGACY_POOL_CACHE = CORPUS / "_pools"
if str(TRAINING) not in sys.path:
    sys.path.insert(0, str(TRAINING))

PREPROCESS_SOURCE = TRAINING / "preprocess.py"
INVARIANT_PREPROCESS_SOURCE = TRAINING / "invariant_patch_preprocess.py"
DATA_SOURCE = Path(__file__).resolve()

CACHE_SCHEMA = 1
V2_SPLIT_SEED = 20260721
GROUND_SPLIT_SEED = 20260727
DEFAULT_MODEL_SEED = 20260727
SPLIT_POLICY_VERSION = "v2-30-30-40-edge-safe-ground-half-v1"
EDGE_RULE = "abs(centreOffsetFrac) + bandwidthHz/sampleRateHz/2 < 0.5"
FEATURE_STD_EPSILON = 1e-6

ARRAY_KEYS = (
    "xtr",
    "ftr",
    "ytr",
    "imp_tr",
    "tr_idx",
    "xen",
    "fen",
    "yen",
    "imp_en",
    "en_idx",
    "xva",
    "fva",
    "yva",
    "imp_va",
    "va_idx",
    "fmean",
    "fstd",
)


class CacheUnavailableError(RuntimeError):
    """The requested invariant-patch cache contract has not been built."""


@dataclass(frozen=True)
class InvariantPatchDataConfig:
    """Preprocessing parameters whose values define cached tensors."""

    patch_length: int = 64
    patch_count: int = 16
    target_frac: float = 0.5

    def validate(self) -> "InvariantPatchDataConfig":
        for name, value in (
            ("patch_length", self.patch_length),
            ("patch_count", self.patch_count),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        try:
            target = float(self.target_frac)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("target_frac must be finite and in (0, 1]") from exc
        if not np.isfinite(target) or not 0.0 < target <= 1.0:
            raise ValueError("target_frac must be finite and in (0, 1]")
        return InvariantPatchDataConfig(
            patch_length=int(self.patch_length),
            patch_count=int(self.patch_count),
            target_frac=target,
        )

    @property
    def input_length(self) -> int:
        return int(self.patch_length * self.patch_count)


def _sha256(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _split_policy() -> dict[str, Any]:
    """Serializable split definition, including every RNG call that matters."""
    return {
        "version": SPLIT_POLICY_VERSION,
        "v2_split_seed": V2_SPLIT_SEED,
        "v2_rng": "numpy.default_rng",
        "v2_rng_order": [
            "permutation(clean corpus indices)",
            "permutation(impaired corpus indices)",
        ],
        "v2_carve": {
            "enroll": "first floor(0.30*n)",
            "validation": "next floor(0.30*n)",
            "train": "remainder",
            "populations": "clean and impaired independently",
        },
        "edge_rule": EDGE_RULE,
        "ground_split_seed": GROUND_SPLIT_SEED,
        "ground_selection_rng_seed": GROUND_SPLIT_SEED + 1,
        "ground_strata": ["class", "impaired"],
        "ground_selection_rule": (
            "per-stratum permutation; first floor(n/2) to selection; "
            "remainder held out and never serialized"
        ),
        "feature_standardization": {
            "fit": "edge-safe training raw features only",
            "epsilon": FEATURE_STD_EPSILON,
        },
    }


def _default_source_paths() -> dict[str, Path]:
    return {
        "training/preprocess.py": PREPROCESS_SOURCE,
        "training/invariant_patch_preprocess.py": INVARIANT_PREPROCESS_SOURCE,
        "training/zplane_ab/v2_full_variation/invariant_patch_data.py": DATA_SOURCE,
    }


def _corpus_identity(corpus_dir: Path) -> dict[str, Any]:
    manifest_path = corpus_dir / "corpus.json"
    raw_path = corpus_dir / "corpus.f32"
    raw_stat = raw_path.stat()
    return {
        "manifest_sha256": _sha256(manifest_path),
        "corpus_f32": {
            "size": int(raw_stat.st_size),
            "mtime_ns": int(raw_stat.st_mtime_ns),
        },
    }


def cache_contract(
    patch_length: int = 64,
    patch_count: int = 16,
    target_frac: float = 0.5,
    *,
    corpus_dir: os.PathLike[str] | str = CORPUS,
    _source_paths: Mapping[str, os.PathLike[str] | str] | None = None,
) -> dict[str, Any]:
    """Return the complete content/implementation identity of a cache.

    ``_source_paths`` exists only to let tiny unit tests use disposable source
    witnesses.  Production callers always hash the current production,
    invariant-front-end, and data-loader sources.
    """
    config = InvariantPatchDataConfig(
        patch_length=patch_length,
        patch_count=patch_count,
        target_frac=target_frac,
    ).validate()
    corpus_path = Path(corpus_dir).resolve()
    paths = dict(_source_paths or _default_source_paths())
    payload: dict[str, Any] = {
        "schema": CACHE_SCHEMA,
        "frontend": "invariant-patch-v1",
        "config": asdict(config),
        "corpus": _corpus_identity(corpus_path),
        "source_sha256": {
            name: _sha256(Path(path)) for name, path in sorted(paths.items())
        },
        "split_policy": _split_policy(),
    }
    payload["id"] = _json_hash(payload)[:16]
    return payload


def _read_manifest(corpus_dir: Path) -> dict[str, Any]:
    with open(corpus_dir / "corpus.json", encoding="utf-8") as handle:
        manifest = json.load(handle)
    required = ("count", "sampleCount", "classes", "items")
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ValueError(f"corpus manifest is missing {missing}")
    count = int(manifest["count"])
    sample_count = int(manifest["sampleCount"])
    if count <= 0 or sample_count < 2 or len(manifest["items"]) != count:
        raise ValueError("corpus manifest count/sampleCount/items are inconsistent")
    expected_size = count * sample_count * 2 * np.dtype("<f4").itemsize
    actual_size = (corpus_dir / "corpus.f32").stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"corpus.f32 size {actual_size} does not match manifest expectation "
            f"{expected_size}"
        )
    return manifest


def _labels_from_manifest(
    manifest: Mapping[str, Any],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    classes = sorted(str(value) for value in manifest["classes"])
    class_index = {name: index for index, name in enumerate(classes)}
    try:
        labels = np.asarray(
            [class_index[str(item["cls"])] for item in manifest["items"]],
            dtype=np.int64,
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("manifest items contain an unknown or missing class") from exc
    impaired = np.asarray(
        [bool(item.get("impaired", False)) for item in manifest["items"]],
        dtype=bool,
    )
    return classes, labels, impaired


def _edge_safe_mask(manifest: Mapping[str, Any]) -> np.ndarray:
    safe: list[bool] = []
    for item in manifest["items"]:
        try:
            center = abs(float(item.get("centreOffsetFrac", 0.0)))
            sample_rate = float(item["sampleRateHz"])
            bandwidth = float(item["bandwidthHz"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("manifest contains invalid nominal band geometry") from exc
        if (
            not np.isfinite(center)
            or not np.isfinite(sample_rate)
            or not np.isfinite(bandwidth)
            or sample_rate <= 0.0
            or bandwidth < 0.0
        ):
            raise ValueError("manifest contains invalid nominal band geometry")
        safe.append(center + 0.5 * bandwidth / sample_rate < 0.5)
    return np.asarray(safe, dtype=bool)


def _carve_v2(position: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_enroll = int(0.30 * len(position))
    n_validation = int(0.30 * len(position))
    enroll = position[:n_enroll]
    validation = position[n_enroll : n_enroll + n_validation]
    train = position[n_enroll + n_validation :]
    return enroll, validation, train


def _reconstruct_v2_split(
    impaired: np.ndarray,
) -> dict[str, np.ndarray]:
    """Reproduce ``full_split.load_and_split_v2`` index arrays exactly."""
    flags = np.asarray(impaired, dtype=bool)
    if flags.ndim != 1 or len(flags) == 0:
        raise ValueError("impaired must be a non-empty 1-D array")
    rng = np.random.default_rng(V2_SPLIT_SEED)
    clean_position = rng.permutation(np.where(~flags)[0])
    impaired_position = rng.permutation(np.where(flags)[0])
    clean_enroll, clean_validation, clean_train = _carve_v2(clean_position)
    impaired_enroll, impaired_validation, impaired_train = _carve_v2(
        impaired_position
    )
    return {
        "tr": np.concatenate((clean_train, impaired_train)).astype(
            np.int64, copy=False
        ),
        "en": np.concatenate((clean_enroll, impaired_enroll)).astype(
            np.int64, copy=False
        ),
        "va": np.concatenate((clean_validation, impaired_validation)).astype(
            np.int64, copy=False
        ),
    }


def _validate_existing_split_indices(
    split: Mapping[str, np.ndarray],
    legacy_cache_dir: Path,
) -> list[str]:
    """Compare every discoverable historical index witness, without using it."""
    validated: list[str] = []
    if not legacy_cache_dir.is_dir():
        return validated
    for tag in ("tr", "en", "va"):
        pattern = str(legacy_cache_dir / f"*_{tag}_idx.npy")
        for filename in sorted(glob.glob(pattern)):
            reference = np.load(filename, allow_pickle=False)
            expected = np.asarray(split[tag], dtype=np.int64)
            if reference.shape != expected.shape or not np.array_equal(
                reference, expected
            ):
                raise RuntimeError(
                    "raw-corpus v2 split reconstruction disagrees with existing "
                    f"index witness {filename}"
                )
            validated.append(os.path.basename(filename))
    return validated


def _ground_selection_indices(
    validation_indices: np.ndarray,
    labels: np.ndarray,
    impaired: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """Return selection corpus IDs and counts, never held-out corpus IDs."""
    validation = np.asarray(validation_indices, dtype=np.int64)
    y = np.asarray(labels, dtype=np.int64)[validation]
    imp = np.asarray(impaired, dtype=bool)[validation]
    rng = np.random.default_rng(GROUND_SPLIT_SEED + 1)
    selection_rows: list[int] = []
    held_out_count = 0
    for cls in np.unique(y):
        for impaired_value in (False, True):
            rows = np.where((y == cls) & (imp == impaired_value))[0]
            permuted = rng.permutation(rows)
            cut = len(rows) // 2
            selection_rows.extend(int(row) for row in permuted[:cut])
            held_out_count += int(len(permuted) - cut)
    selection_rows_array = np.sort(
        np.asarray(selection_rows, dtype=np.int64)
    )
    if len(np.unique(selection_rows_array)) != len(selection_rows_array):
        raise AssertionError("ground selection rows must be unique")
    if len(selection_rows_array) + held_out_count != len(validation):
        raise AssertionError("ground selection split must be exhaustive")
    return validation[selection_rows_array], {
        "edge_safe_validation": int(len(validation)),
        "selection": int(len(selection_rows_array)),
        "consumed_test_held_out": held_out_count,
        "consumed_test_rows_exposed": 0,
    }


def _class_counts(
    indices: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
) -> dict[str, int]:
    chosen = np.asarray(labels, dtype=np.int64)[np.asarray(indices, dtype=np.int64)]
    return {
        name: int(np.sum(chosen == class_index))
        for class_index, name in enumerate(classes)
    }


def _impairment_counts(
    indices: np.ndarray,
    impaired: np.ndarray,
) -> dict[str, int]:
    chosen = np.asarray(impaired, dtype=bool)[
        np.asarray(indices, dtype=np.int64)
    ]
    return {
        "clean": int(np.sum(~chosen)),
        "impaired": int(np.sum(chosen)),
    }


def _prepare_exposed_indices(
    manifest: Mapping[str, Any],
    labels: np.ndarray,
    impaired: np.ndarray,
    *,
    legacy_cache_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    full = _reconstruct_v2_split(impaired)
    witnesses = _validate_existing_split_indices(full, legacy_cache_dir)
    safe = _edge_safe_mask(manifest)
    safe_full = {
        tag: np.asarray(indices, dtype=np.int64)[safe[indices]]
        for tag, indices in full.items()
    }
    selection, selection_counts = _ground_selection_indices(
        safe_full["va"], labels, impaired
    )
    exposed = {
        "tr": safe_full["tr"],
        "en": safe_full["en"],
        "va": selection,
    }

    all_exposed = np.concatenate(tuple(exposed.values()))
    if len(np.unique(all_exposed)) != len(all_exposed):
        raise AssertionError("exposed train/enroll/selection indices must be disjoint")
    for indices in exposed.values():
        if not np.all(safe[indices]):
            raise AssertionError("an FFT-edge-unsafe row reached an exposed split")

    classes = sorted(str(value) for value in manifest["classes"])
    audit: dict[str, Any] = {
        "split_policy_version": SPLIT_POLICY_VERSION,
        "v2_split_seed": V2_SPLIT_SEED,
        "ground_split_seed": GROUND_SPLIT_SEED,
        "ground_selection_rng_seed": GROUND_SPLIT_SEED + 1,
        "edge_rule": EDGE_RULE,
        "existing_split_index_witnesses_validated": witnesses,
        "counts": {
            "raw_corpus": int(len(labels)),
            "v2_before_edge_filter": {
                tag: int(len(full[tag])) for tag in ("tr", "en", "va")
            },
            "edge_unsafe_excluded": {
                tag: int(len(full[tag]) - len(safe_full[tag]))
                for tag in ("tr", "en", "va")
            },
            "exposed": {
                "train": int(len(exposed["tr"])),
                "enroll": int(len(exposed["en"])),
                "selection": int(len(exposed["va"])),
            },
            **selection_counts,
        },
        "class_counts": {
            "train": _class_counts(exposed["tr"], labels, classes),
            "enroll": _class_counts(exposed["en"], labels, classes),
            "selection": _class_counts(exposed["va"], labels, classes),
        },
        "impairment_counts": {
            "train": _impairment_counts(exposed["tr"], impaired),
            "enroll": _impairment_counts(exposed["en"], impaired),
            "selection": _impairment_counts(exposed["va"], impaired),
        },
        "consumed_test_rows_exposed": 0,
    }
    return exposed, audit


def _preprocess_partition(
    raw: np.memmap,
    indices: np.ndarray,
    config: InvariantPatchDataConfig,
    *,
    preprocess_fn: Callable[..., tuple[np.ndarray, np.ndarray, dict[str, Any]]],
    feature_count: int,
    nfft: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(indices, dtype=np.int64)
    channels = np.empty(
        (len(rows), 2, config.input_length),
        dtype=np.float32,
    )
    features = np.empty((len(rows), feature_count), dtype=np.float32)
    for output_row, corpus_row in enumerate(rows):
        pair = np.asarray(raw[int(corpus_row)], dtype=np.float64)
        iq = pair[:, 0] + 1.0j * pair[:, 1]
        packed, raw_features, _context = preprocess_fn(
            iq,
            patch_length=config.patch_length,
            patch_count=config.patch_count,
            target_frac=config.target_frac,
            nfft=nfft,
        )
        packed_array = np.asarray(packed, dtype=np.float32)
        feature_array = np.asarray(raw_features, dtype=np.float32)
        expected_shape = (2, config.input_length)
        if packed_array.shape != expected_shape or not np.all(
            np.isfinite(packed_array)
        ):
            raise ValueError(
                f"invariant preprocess row {int(corpus_row)} returned channels "
                f"{packed_array.shape}, expected {expected_shape}"
            )
        if feature_array.shape != (feature_count,) or not np.all(
            np.isfinite(feature_array)
        ):
            raise ValueError(
                f"invariant preprocess row {int(corpus_row)} returned malformed "
                "features"
            )
        channels[output_row] = packed_array
        features[output_row] = feature_array
    return channels, features


def _standardize_features(
    train: np.ndarray,
    enroll: np.ndarray,
    selection: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit coordinates on training features only and apply them to all splits."""
    train_array = np.asarray(train, dtype=np.float32)
    enroll_array = np.asarray(enroll, dtype=np.float32)
    selection_array = np.asarray(selection, dtype=np.float32)
    if train_array.ndim != 2 or len(train_array) == 0:
        raise ValueError("training features must be a non-empty 2-D array")
    if (
        enroll_array.ndim != 2
        or selection_array.ndim != 2
        or enroll_array.shape[1:] != train_array.shape[1:]
        or selection_array.shape[1:] != train_array.shape[1:]
    ):
        raise ValueError("all feature splits must have the same feature width")
    mean = train_array.astype(np.float64).mean(axis=0).astype(np.float32)
    std = (
        train_array.astype(np.float64).std(axis=0) + FEATURE_STD_EPSILON
    ).astype(np.float32)

    def apply(values: np.ndarray) -> np.ndarray:
        return ((values - mean) / std).astype(np.float32)

    return apply(train_array), apply(enroll_array), apply(selection_array), mean, std


def _cache_prefix(contract: Mapping[str, Any]) -> str:
    return f"invariant_patch_v1__{contract['id']}"


def _metadata_path(cache_dir: Path, contract: Mapping[str, Any]) -> Path:
    return cache_dir / f"{_cache_prefix(contract)}_meta.json"


def _array_path(
    cache_dir: Path,
    contract: Mapping[str, Any],
    key: str,
) -> Path:
    return cache_dir / f"{_cache_prefix(contract)}_{key}.npy"


def _atomic_save_array(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, np.asarray(value), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _metadata_array_schema(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        key: {
            "shape": [int(value) for value in np.asarray(arrays[key]).shape],
            "dtype": np.asarray(arrays[key]).dtype.str,
        }
        for key in ARRAY_KEYS
    }


def build(
    patch_length: int = 64,
    patch_count: int = 16,
    target_frac: float = 0.5,
    *,
    corpus_dir: os.PathLike[str] | str = CORPUS,
    cache_dir: os.PathLike[str] | str = CACHE,
    legacy_cache_dir: os.PathLike[str] | str = LEGACY_POOL_CACHE,
    _preprocess_fn: (
        Callable[..., tuple[np.ndarray, np.ndarray, dict[str, Any]]] | None
    ) = None,
    _source_paths: Mapping[str, os.PathLike[str] | str] | None = None,
) -> dict[str, Any]:
    """Materialize one cache contract; metadata is the final completion marker."""
    import invariant_patch_preprocess as invariant_preprocess
    import preprocess as production_preprocess

    config = InvariantPatchDataConfig(
        patch_length=patch_length,
        patch_count=patch_count,
        target_frac=target_frac,
    ).validate()
    corpus_path = Path(corpus_dir).resolve()
    cache_path = Path(cache_dir).resolve()
    legacy_path = Path(legacy_cache_dir).resolve()
    contract = cache_contract(
        config.patch_length,
        config.patch_count,
        config.target_frac,
        corpus_dir=corpus_path,
        _source_paths=_source_paths,
    )
    metadata_path = _metadata_path(cache_path, contract)
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as handle:
            existing = json.load(handle)
        if (
            existing.get("complete") is True
            and existing.get("cache_contract") == contract
        ):
            return contract
        raise RuntimeError(f"incompatible cache metadata already exists: {metadata_path}")

    manifest = _read_manifest(corpus_path)
    classes, labels, impaired = _labels_from_manifest(manifest)
    exposed, audit = _prepare_exposed_indices(
        manifest,
        labels,
        impaired,
        legacy_cache_dir=legacy_path,
    )
    raw = np.memmap(
        corpus_path / "corpus.f32",
        dtype="<f4",
        mode="r",
        shape=(int(manifest["count"]), int(manifest["sampleCount"]), 2),
    )
    processor = _preprocess_fn or invariant_preprocess.preprocess
    feature_count = int(production_preprocess.N_FEATURES)
    nfft = int(production_preprocess.NFFT)
    xtr, raw_ftr = _preprocess_partition(
        raw,
        exposed["tr"],
        config,
        preprocess_fn=processor,
        feature_count=feature_count,
        nfft=nfft,
    )
    xen, raw_fen = _preprocess_partition(
        raw,
        exposed["en"],
        config,
        preprocess_fn=processor,
        feature_count=feature_count,
        nfft=nfft,
    )
    xva, raw_fva = _preprocess_partition(
        raw,
        exposed["va"],
        config,
        preprocess_fn=processor,
        feature_count=feature_count,
        nfft=nfft,
    )
    del raw
    ftr, fen, fva, fmean, fstd = _standardize_features(
        raw_ftr,
        raw_fen,
        raw_fva,
    )

    arrays: dict[str, np.ndarray] = {
        "xtr": xtr,
        "ftr": ftr,
        "ytr": labels[exposed["tr"]].astype(np.int64, copy=False),
        "imp_tr": impaired[exposed["tr"]].astype(bool, copy=False),
        "tr_idx": exposed["tr"].astype(np.int64, copy=False),
        "xen": xen,
        "fen": fen,
        "yen": labels[exposed["en"]].astype(np.int64, copy=False),
        "imp_en": impaired[exposed["en"]].astype(bool, copy=False),
        "en_idx": exposed["en"].astype(np.int64, copy=False),
        "xva": xva,
        "fva": fva,
        "yva": labels[exposed["va"]].astype(np.int64, copy=False),
        "imp_va": impaired[exposed["va"]].astype(bool, copy=False),
        "va_idx": exposed["va"].astype(np.int64, copy=False),
        "fmean": fmean,
        "fstd": fstd,
    }

    for key in ARRAY_KEYS:
        _atomic_save_array(_array_path(cache_path, contract, key), arrays[key])

    # Fail closed if the manifest/raw stat or any implementation source changed
    # while preprocessing.  In that case no completion marker is written.
    final_contract = cache_contract(
        config.patch_length,
        config.patch_count,
        config.target_frac,
        corpus_dir=corpus_path,
        _source_paths=_source_paths,
    )
    if final_contract != contract:
        raise RuntimeError(
            "cache inputs changed during build; refusing to write completion metadata"
        )

    metadata = {
        "complete": True,
        "cache_contract": contract,
        "classes": classes,
        "n_classes": int(len(classes)),
        "input_length": config.input_length,
        "n_features": feature_count,
        "array_schema": _metadata_array_schema(arrays),
        "data_audit": audit,
    }
    # This is intentionally last: its presence is the only completion signal.
    _atomic_save_json(metadata_path, metadata)
    return contract


def _nonnegative_seed(value: Any) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
    ):
        raise ValueError("model_seed must be a non-negative integer")
    return int(value)


def load(
    patch_length: int = 64,
    patch_count: int = 16,
    target_frac: float = 0.5,
    model_seed: int = DEFAULT_MODEL_SEED,
    *,
    build_if_missing: bool = True,
    corpus_dir: os.PathLike[str] | str = CORPUS,
    cache_dir: os.PathLike[str] | str = CACHE,
    legacy_cache_dir: os.PathLike[str] | str = LEGACY_POOL_CACHE,
    _preprocess_fn: (
        Callable[..., tuple[np.ndarray, np.ndarray, dict[str, Any]]] | None
    ) = None,
    _source_paths: Mapping[str, os.PathLike[str] | str] | None = None,
) -> dict[str, Any]:
    """Load train/enroll/selection-only invariant-patch data.

    A fresh deterministic NumPy generator is returned for model/episode
    sampling.  ``model_seed`` is intentionally absent from the cache identity:
    it changes sampling, not preprocessed data.
    """
    config = InvariantPatchDataConfig(
        patch_length=patch_length,
        patch_count=patch_count,
        target_frac=target_frac,
    ).validate()
    seed = _nonnegative_seed(model_seed)
    corpus_path = Path(corpus_dir).resolve()
    cache_path = Path(cache_dir).resolve()
    contract = cache_contract(
        config.patch_length,
        config.patch_count,
        config.target_frac,
        corpus_dir=corpus_path,
        _source_paths=_source_paths,
    )
    metadata_path = _metadata_path(cache_path, contract)
    if not metadata_path.exists():
        if not build_if_missing:
            raise CacheUnavailableError(
                f"invariant patch cache contract {contract['id']} is not built"
            )
        build(
            config.patch_length,
            config.patch_count,
            config.target_frac,
            corpus_dir=corpus_path,
            cache_dir=cache_path,
            legacy_cache_dir=legacy_cache_dir,
            _preprocess_fn=_preprocess_fn,
            _source_paths=_source_paths,
        )
    with open(metadata_path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("complete") is not True:
        raise RuntimeError(f"cache completion marker is absent: {metadata_path}")
    if metadata.get("cache_contract") != contract:
        raise RuntimeError(
            f"cache contract mismatch for {contract['id']}; refusing stale arrays"
        )

    arrays: dict[str, np.ndarray] = {}
    schema = metadata.get("array_schema", {})
    for key in ARRAY_KEYS:
        path = _array_path(cache_path, contract, key)
        if not path.exists():
            raise RuntimeError(f"completed cache is missing {path.name}")
        mmap = "r" if key in {"xtr", "ftr", "xen", "fen", "xva", "fva"} else None
        value = np.load(path, mmap_mode=mmap, allow_pickle=False)
        expected = schema.get(key)
        if expected != {
            "shape": [int(size) for size in value.shape],
            "dtype": value.dtype.str,
        }:
            raise RuntimeError(f"cache array schema mismatch for {path.name}")
        arrays[key] = value

    n_classes = int(metadata["n_classes"])
    class_indices = [
        np.where(np.asarray(arrays["ytr"]) == cls)[0]
        for cls in range(n_classes)
    ]
    audit = json.loads(json.dumps(metadata["data_audit"]))
    audit["model_seed"] = seed
    audit["consumed_test_rows_exposed"] = 0

    return {
        **arrays,
        "classes": list(metadata["classes"]),
        "n_classes": n_classes,
        "input_length": int(metadata["input_length"]),
        "n_features": int(metadata["n_features"]),
        "idx_by_class": class_indices,
        "rng": np.random.default_rng(seed),
        "condition": "invariant-patch-v1",
        "cache_contract": contract,
        "data_audit": audit,
    }


if __name__ == "__main__":
    loaded = load()
    print(
        "invariant patch data "
        f"{loaded['cache_contract']['id']}: "
        f"train={len(loaded['ytr'])} enroll={len(loaded['yen'])} "
        f"selection={len(loaded['yva'])} input={loaded['input_length']}"
    )
