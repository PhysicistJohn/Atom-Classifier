"""Leak-safe raw-corpus loading for the v4 current-source development trainer.

The historical side obtains its only row identifiers from
``invariant_patch_data.load``.  That API exposes train, enrollment, and the
development-selection half, but deliberately has no consumed-test row API.

The current side uses a separate fixed-width CF32 corpus.  Each manifest item
must explicitly declare ``cls``, ``splitRole``, and ``validSampleCount``.
Storage after the valid prefix is accepted only when
``zeroPaddedAfterValidSampleCount`` is literally ``true`` and every stored
float in the tail is exactly zero.  A padded tail is storage, never evidence:
no view may extend past ``validSampleCount``.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
V2 = HERE.parent
ZPLANE = V2.parent
TRAINING = ZPLANE.parent
for path in (TRAINING, ZPLANE, V2):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import invariant_patch_data as invariant_data  # noqa: E402


# These are the exact prefix buckets admitted by Atomizer's live classifier.
# A staged 12,160-sample capture therefore supplies an 8,192-sample model
# input.  It is intentionally not a fourth model-input geometry.
RUNTIME_INPUT_LENGTHS = (4_096, 8_192, 16_384)
MINIMUM_INPUT_LENGTH = RUNTIME_INPUT_LENGTHS[0]
ROLE_ALIASES = {
    "train": "train",
    "enroll": "enrollment",
    "enrollment": "enrollment",
    "selection": "selection",
}
ROLES = ("train", "enrollment", "selection")

# Fail closed on the exact SignalLab service/corpus contract used to produce
# current-source evidence.  These values deliberately duplicate the pins in
# tools/generate-current-signallab-service-corpus.ts: the loader must not trust
# a manifest to tell it which generator or source checkout should be accepted.
CURRENT_CORPUS_SCHEMA_VERSION = 1
CURRENT_CORPUS_GENERATOR = "current-signallab-service-corpus-v1"
CURRENT_CORPUS_GENERATOR_PATH = (
    "tools/generate-current-signallab-service-corpus.ts"
)
CURRENT_CORPUS_GENERATOR_BUNDLE_PATH = (
    ".artifacts/current-corpus-generator/"
    "generate-current-signallab-service-corpus.js"
)
CURRENT_CORPUS_GENERATOR_LINEAGE_SCHEMA = (
    "atom-classifier-generator-lineage-v1"
)
EXPECTED_SIGNAL_LAB_GIT_COMMIT = (
    "94171baf31bfa62150d4e9684e1fe33e5a343dc0"
)
EXPECTED_SIGNAL_LAB_GIT_TREE = (
    "ccda4401748953bb6705776694b29b571d4c684d"
)
EXPECTED_SIGNAL_LAB_CONTRACT_SHA256 = (
    "e5ebc468495b027c8b91aceac8aaa9031ae3104b2306641a868fe9c1eaec842f"
)
EXPECTED_SIGNAL_LAB_GENERATOR_BINDING_SHA256 = (
    "c88f13a423cf0c78065e38f2ed380c86d483191df9799182689f45dcd2c8a8a4"
)
EXPECTED_SIGNAL_LAB_CATALOG_SHA256 = (
    "00037e0c22f47a8af433b95b00aa7524026357fe21c9979067d37a832294b9c4"
)
EXPECTED_CURRENT_GENERATOR_SOURCE_SHA256 = (
    "aab9059d0b56a0d74c45da549d9ba7f8214474615ea64ead0e9faf6a9f0658c0"
)
EXPECTED_CURRENT_GENERATOR_BUNDLE_SHA256 = (
    "91c5700a6ba8715328f51350182ba14495c0d86f3bcfb4773703fca49b253a7f"
)
CURRENT_SERVICE_PATH = (
    "AtomizerMeasurementService.selectProfile",
    "AtomizerMeasurementService.configureChannel",
    "MeasurementServiceContinuation(v2)",
    "AtomizerMeasurementService.acquireIq",
    "complexIqMeasurementSchema.parse",
)
CURRENT_RECEIVER_PRESETS = (
    "clean",
    "awgn",
    "multipath",
    "carrier-offset",
    "phase-noise",
    "iq-imbalance",
    "dc-offset",
    "pa-compression",
    "composite",
)
CURRENT_PROFILE_PUBLIC_CLASS_MAP = {
    "bluetooth-classic-connected": "bluetooth",
    "bluetooth-le-advertising": "bluetooth",
    "gsm-16qam-higher-symbol-rate-burst": "gsm",
    "gsm-32qam-higher-symbol-rate-burst": "gsm",
    "gsm-8psk-normal-burst": "gsm",
    "gsm-900-loaded-bcch": "gsm",
    "gsm-aqpsk-normal-burst": "gsm",
    "gsm-normal-burst": "gsm",
    "gsm-qpsk-higher-symbol-rate-burst": "gsm",
    "lte-band3-fdd-20m": "ofdm",
    "lte-band38-tdd-10m": "ofdm",
    "lte-etm1.1": "ofdm",
    "lte-etm3.1": "ofdm",
    "lte-etm3.1a": "ofdm",
    "lte-etm3.1b": "ofdm",
    "lte-nbiot-guard-isolated-component": "ofdm",
    "lte-nbiot-inband-isolated-component": "ofdm",
    "lte-ntm": "ofdm",
    "nr-fr1-tm1.1": "ofdm",
    "nr-fr1-tm3.1": "ofdm",
    "nr-fr1-tm3.1a": "ofdm",
    "nr-fr1-tm3.1b": "ofdm",
    "nr-n3-fdd-20m": "ofdm",
    "nr-n78-tdd-100m": "ofdm",
    "nr-nbiot-inband-isolated-component": "ofdm",
    "wifi-hr-dsss-11m": "dsss",
    "wifi-ofdm-20m": "ofdm",
    "wifi6-he-er-su": "ofdm",
    "wifi6-he-mu": "ofdm",
    "wifi6-he-su": "ofdm",
    "wifi6-he-tb": "ofdm",
}
CURRENT_PROFILES = tuple(sorted(CURRENT_PROFILE_PUBLIC_CLASS_MAP))
CURRENT_CLASSES = tuple(sorted(set(CURRENT_PROFILE_PUBLIC_CLASS_MAP.values())))
HASH_PATTERN = re.compile(r"^[a-f0-9]{64}$")
UUID_V4_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_json_object(path: Path) -> dict[str, Any]:
    def reject_duplicate_pairs(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(
                    f"{path.name} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    def reject_nonfinite(token: str) -> None:
        raise ValueError(
            f"{path.name} contains forbidden non-finite JSON token {token}"
        )

    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=reject_duplicate_pairs,
                parse_constant=reject_nonfinite,
            )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{path.name} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("current corpus manifest must be a JSON object")
    return value


def _strict_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or HASH_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _strict_nonnegative_integer(value: Any, field: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
    ):
        raise ValueError(f"{field} must be a non-negative integer")
    return int(value)


def _strict_integer(value: Any, field: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{field} must be an integer")
    return int(value)


def _require_exact(value: Any, expected: Any, field: str) -> None:
    if value != expected:
        raise ValueError(
            f"{field} changed from the pinned current-corpus contract"
        )


def _generator_lineage_binding(
    *,
    source_path: str,
    source_sha256: str,
    bundle_path: str,
    bundle_sha256: str,
) -> str:
    # Keep this byte layout identical to generatorLineageBindingSha256 in the
    # TypeScript generator.
    payload = (
        CURRENT_CORPUS_GENERATOR_LINEAGE_SCHEMA
        + "\0"
        + source_path
        + "\0"
        + source_sha256
        + "\0"
        + bundle_path
        + "\0"
        + bundle_sha256
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_current_manifest_contract(
    manifest: Mapping[str, Any],
) -> None:
    _require_exact(
        manifest.get("schemaVersion"),
        CURRENT_CORPUS_SCHEMA_VERSION,
        "schemaVersion",
    )
    _require_exact(
        manifest.get("generator"),
        CURRENT_CORPUS_GENERATOR,
        "generator",
    )
    _require_exact(
        manifest.get("generatorPath"),
        CURRENT_CORPUS_GENERATOR_PATH,
        "generatorPath",
    )
    _require_exact(
        manifest.get("servicePath"),
        list(CURRENT_SERVICE_PATH),
        "servicePath",
    )
    _require_exact(
        manifest.get("lowLevelSynthesizerCalls"),
        False,
        "lowLevelSynthesizerCalls",
    )
    _require_exact(manifest.get("format"), "cf32le-interleaved", "format")
    _require_exact(manifest.get("dataFile"), "corpus.f32", "dataFile")
    _require_exact(
        manifest.get("cleanDataFile"),
        "corpus_clean.f32",
        "cleanDataFile",
    )
    _require_exact(manifest.get("hasCleanPairs"), True, "hasCleanPairs")
    _require_exact(
        manifest.get("runtimeInputBuckets"),
        list(RUNTIME_INPUT_LENGTHS),
        "runtimeInputBuckets",
    )
    _require_exact(
        manifest.get("receiverPresets"),
        list(CURRENT_RECEIVER_PRESETS),
        "receiverPresets",
    )
    _require_exact(
        manifest.get("classes"),
        list(CURRENT_CLASSES),
        "classes",
    )
    _require_exact(
        manifest.get("profiles"),
        list(CURRENT_PROFILES),
        "profiles",
    )
    _require_exact(
        manifest.get("profilePublicClassMap"),
        CURRENT_PROFILE_PUBLIC_CLASS_MAP,
        "profilePublicClassMap",
    )

    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("source must be an object")
    pinned_source = {
        "repository": "Atom-SignalLab",
        "gitCommit": EXPECTED_SIGNAL_LAB_GIT_COMMIT,
        "gitTree": EXPECTED_SIGNAL_LAB_GIT_TREE,
        "worktreeClean": True,
        "contractSha256": EXPECTED_SIGNAL_LAB_CONTRACT_SHA256,
        "generatorContractBindingSha256":
            EXPECTED_SIGNAL_LAB_GENERATOR_BINDING_SHA256,
    }
    for field, expected in pinned_source.items():
        _require_exact(source.get(field), expected, f"source.{field}")

    lineage = manifest.get("generatorLineage")
    if not isinstance(lineage, Mapping):
        raise ValueError("generatorLineage must be an object")
    if (
        not EXPECTED_CURRENT_GENERATOR_SOURCE_SHA256
        or not EXPECTED_CURRENT_GENERATOR_BUNDLE_SHA256
    ):
        raise ValueError(
            "current corpus loader generator source/bundle pins are unset"
        )
    pinned_lineage = {
        "schema": CURRENT_CORPUS_GENERATOR_LINEAGE_SCHEMA,
        "repository": "Atom-Classifier",
        "sourcePath": CURRENT_CORPUS_GENERATOR_PATH,
        "sourceSha256": EXPECTED_CURRENT_GENERATOR_SOURCE_SHA256,
        "bundlePath": CURRENT_CORPUS_GENERATOR_BUNDLE_PATH,
        "bundleSha256": EXPECTED_CURRENT_GENERATOR_BUNDLE_SHA256,
    }
    for field, expected in pinned_lineage.items():
        _require_exact(
            lineage.get(field),
            expected,
            f"generatorLineage.{field}",
        )
    binding = _generator_lineage_binding(
        source_path=CURRENT_CORPUS_GENERATOR_PATH,
        source_sha256=EXPECTED_CURRENT_GENERATOR_SOURCE_SHA256,
        bundle_path=CURRENT_CORPUS_GENERATOR_BUNDLE_PATH,
        bundle_sha256=EXPECTED_CURRENT_GENERATOR_BUNDLE_SHA256,
    )
    _require_exact(
        lineage.get("sourceBundleBindingSha256"),
        binding,
        "generatorLineage.sourceBundleBindingSha256",
    )


def _clean_raw_path(
    directory: Path,
    manifest: Mapping[str, Any],
) -> Path:
    filename = manifest.get("cleanDataFile")
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
    ):
        raise ValueError(
            "cleanDataFile must be a plain filename inside the corpus"
        )
    path = (directory / filename).resolve()
    if path.parent != directory or not path.is_file():
        raise ValueError("current corpus clean-pair data file is missing")
    return path


def _validate_measurement_receipt(
    value: Any,
    *,
    item_index: int,
    field: str,
    valid_sample_count: int,
    sample_rate_hz: int,
    content_sha256: str,
    source_content_sha256: str,
    receiver_preset: str,
    replay: str,
    phase_native_sample: int,
    receiver_seed: int | None,
) -> None:
    label = f"items[{item_index}].{field}"
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    for identifier in (
        "measurementId",
        "sessionId",
        "configurationRevision",
    ):
        identifier_value = value.get(identifier)
        if (
            not isinstance(identifier_value, str)
            or UUID_V4_PATTERN.fullmatch(identifier_value) is None
        ):
            raise ValueError(f"{label}.{identifier} must be a UUIDv4")
    _strict_positive_integer(value.get("sequence"), f"{label}.sequence")
    captured_at = value.get("capturedAt")
    if (
        not isinstance(captured_at, str)
        or re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z",
            captured_at,
        )
        is None
    ):
        raise ValueError(f"{label}.capturedAt must be canonical UTC")
    _require_exact(value.get("complete"), True, f"{label}.complete")
    _require_exact(
        value.get("sampleCount"),
        valid_sample_count,
        f"{label}.sampleCount",
    )
    _require_exact(
        value.get("byteLength"),
        valid_sample_count * 8,
        f"{label}.byteLength",
    )
    _require_exact(
        value.get("samplesSha256"),
        content_sha256,
        f"{label}.samplesSha256",
    )
    _require_exact(
        value.get("receiverImpairment"),
        receiver_preset,
        f"{label}.receiverImpairment",
    )
    clean = receiver_preset == "clean"
    receipt_contract = (
        {
            "qualification": "independently-verified-digital-baseband",
            "payloadKind": "native-canonical",
            "representation": "source-preserved-complex-envelope",
            "normalization": "none",
            "channelApplication": "not-applied",
        }
        if clean
        else {
            "qualification": "receiver-impaired-complex-baseband",
            "payloadKind": "receiver-impaired-derived",
            "representation": "normalized-complex-envelope",
            "normalization": "peak-to-0.98",
            "channelApplication": "receiver-impairment-preset",
        }
    )
    for receipt_field, expected in receipt_contract.items():
        _require_exact(
            value.get(receipt_field),
            expected,
            f"{label}.{receipt_field}",
        )

    provenance = value.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{label}.provenance must be an object")
    pinned_provenance = {
        "driverId": "signal-lab",
        "sourceKind": "signal-lab-simulation",
        "execution": "signal-lab-simulation",
        "transport": "signal-lab-measurement-bridge",
        "contractId": "tinysa-signal-lab-atomizer-measurement",
        "contractVersion": 2,
        "contractSha256": EXPECTED_SIGNAL_LAB_CONTRACT_SHA256,
        "generatorContractBindingSha256":
            EXPECTED_SIGNAL_LAB_GENERATOR_BINDING_SHA256,
        "catalogSha256": EXPECTED_SIGNAL_LAB_CATALOG_SHA256,
    }
    for receipt_field, expected in pinned_provenance.items():
        _require_exact(
            provenance.get(receipt_field),
            expected,
            f"{label}.provenance.{receipt_field}",
        )
    _require_exact(
        provenance.get("claims"),
        {
            "usbEmulated": False,
            "firmwareExecuted": False,
            "rfEmitted": False,
        },
        f"{label}.provenance.claims",
    )

    transform = value.get("transformReceipt")
    if not isinstance(transform, Mapping):
        raise ValueError(f"{label}.transformReceipt must be an object")
    _require_exact(
        transform.get("receiptVersion"),
        1,
        f"{label}.transformReceipt.receiptVersion",
    )
    _require_exact(
        transform.get("outputSampleCount"),
        valid_sample_count,
        f"{label}.transformReceipt.outputSampleCount",
    )
    _require_exact(
        transform.get("outputSampleRateHz"),
        sample_rate_hz,
        f"{label}.transformReceipt.outputSampleRateHz",
    )
    _require_exact(
        transform.get("outputSamplesSha256"),
        content_sha256,
        f"{label}.transformReceipt.outputSamplesSha256",
    )
    _require_exact(
        transform.get("sourceSamplesSha256"),
        source_content_sha256,
        f"{label}.transformReceipt.sourceSamplesSha256",
    )
    _require_exact(
        transform.get("sourceSampleRateHz"),
        sample_rate_hz,
        f"{label}.transformReceipt.sourceSampleRateHz",
    )
    _require_exact(
        transform.get("sourceSampleCount"),
        valid_sample_count,
        f"{label}.transformReceipt.sourceSampleCount",
    )
    _require_exact(
        transform.get("outputStartSourceSampleNumerator"),
        str(phase_native_sample),
        f"{label}.transformReceipt.outputStartSourceSampleNumerator",
    )
    _require_exact(
        transform.get("outputStartSourceSampleDenominator"),
        "1",
        f"{label}.transformReceipt.outputStartSourceSampleDenominator",
    )
    expected_boundary = (
        "cyclic-modular" if replay == "cyclic"
        else "one-shot-zero-extended"
    )
    _require_exact(
        transform.get("sourceBoundaryPolicy"),
        expected_boundary,
        f"{label}.transformReceipt.sourceBoundaryPolicy",
    )
    source_period = transform.get("sourcePeriodSamples")
    if replay == "cyclic":
        _strict_positive_integer(
            source_period,
            f"{label}.transformReceipt.sourcePeriodSamples",
        )
    elif replay == "one-shot":
        _require_exact(
            source_period,
            None,
            f"{label}.transformReceipt.sourcePeriodSamples",
        )
    else:
        raise ValueError(f"{label} has unsupported replay {replay!r}")
    source_carrier = _strict_integer(
        transform.get("sourceCarrierOffsetHz"),
        f"{label}.transformReceipt.sourceCarrierOffsetHz",
    )
    _require_exact(
        transform.get("outputCarrierOffsetHz"),
        source_carrier,
        f"{label}.transformReceipt.outputCarrierOffsetHz",
    )
    canonical_sha256 = _strict_sha256(
        value.get("canonicalArtifactSha256"),
        f"{label}.canonicalArtifactSha256",
    )
    _require_exact(
        transform.get("sourceArtifactSha256"),
        canonical_sha256,
        f"{label}.transformReceipt.sourceArtifactSha256",
    )
    operations = transform.get("operations")
    if clean:
        _require_exact(
            operations,
            [],
            f"{label}.transformReceipt.operations",
        )
        if receiver_seed is not None:
            raise ValueError(f"{label} clean receipt cannot have a receiver seed")
    else:
        if (
            not isinstance(operations, list)
            or len(operations) != 1
            or not isinstance(operations[0], Mapping)
        ):
            raise ValueError(
                f"{label}.transformReceipt.operations must contain exactly "
                "one receiver impairment"
            )
        operation = operations[0]
        _require_exact(
            operation.get("kind"),
            "receiver-impairment",
            f"{label}.transformReceipt.operations[0].kind",
        )
        _require_exact(
            operation.get("algorithm"),
            "signal-lab-receiver-impairment-v1",
            f"{label}.transformReceipt.operations[0].algorithm",
        )
        _require_exact(
            operation.get("preset"),
            receiver_preset,
            f"{label}.transformReceipt.operations[0].preset",
        )
        operation_seed = _strict_positive_integer(
            operation.get("seed"),
            f"{label}.transformReceipt.operations[0].seed",
        )
        if receiver_seed != operation_seed:
            raise ValueError(
                f"{label} receiver seed disagrees with impairment receipt"
            )


def _strict_positive_integer(value: Any, field: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
    ):
        raise ValueError(f"{field} must be a positive integer")
    return int(value)


def training_view_lengths(valid_sample_count: int) -> tuple[int, ...]:
    """Return live-runtime prefix lengths that are valid training evidence."""
    valid = _strict_positive_integer(valid_sample_count, "validSampleCount")
    return tuple(length for length in RUNTIME_INPUT_LENGTHS if length <= valid)


def admitted_input_length(valid_sample_count: int) -> int:
    """Return Atomizer's largest admitted prefix for one stored capture."""
    lengths = training_view_lengths(valid_sample_count)
    if not lengths:
        raise ValueError(
            f"validSampleCount={valid_sample_count} is below the live minimum "
            f"{MINIMUM_INPUT_LENGTH}"
        )
    return lengths[-1]


@dataclass(frozen=True)
class RowRef:
    """One independent base row and its immutable split/source identity."""

    source: str
    source_index: int
    role: str
    class_name: str
    label: int
    profile_id: str
    valid_sample_count: int
    storage_sample_count: int
    identity: str
    content_sha256: str
    zero_padded_after_valid: bool


@dataclass
class RawCorpus:
    """A validated fixed-width interleaved-I/Q corpus plus row references."""

    source: str
    directory: Path
    manifest_path: Path
    raw_path: Path
    manifest: dict[str, Any]
    raw: np.memmap
    rows_by_role: dict[str, list[RowRef]]
    audit: dict[str, Any]

    def prefix(self, row: RowRef, length: int) -> np.ndarray:
        """Return one contiguous complex prefix, never padded tail storage."""
        if row.source != self.source:
            raise ValueError(
                f"row source {row.source!r} does not belong to {self.source!r}"
            )
        requested = _strict_positive_integer(length, "view length")
        if requested > row.valid_sample_count:
            raise ValueError(
                f"view length {requested} exceeds validSampleCount "
                f"{row.valid_sample_count} for {row.identity}"
            )
        if requested not in RUNTIME_INPUT_LENGTHS:
            raise ValueError(
                f"view length {requested} is not a live runtime bucket "
                f"{RUNTIME_INPUT_LENGTHS}"
            )
        pair = np.asarray(
            self.raw[row.source_index, :requested],
            dtype=np.float64,
        )
        if pair.shape != (requested, 2):
            raise AssertionError("validated corpus row geometry changed")
        if not np.isfinite(pair).all():
            raise ValueError(f"{row.identity} contains NaN or infinity")
        return pair[:, 0] + 1j * pair[:, 1]


def validate_split_separation(rows: Mapping[str, Sequence[RowRef]]) -> None:
    """Reject any base-row identity that appears in more than one split."""
    missing = [role for role in ROLES if role not in rows]
    if missing:
        raise ValueError(f"split map is missing roles {missing}")
    owners: dict[str, str] = {}
    for role in ROLES:
        for row in rows[role]:
            if row.role != role:
                raise ValueError(
                    f"row {row.identity} declares role {row.role!r}, "
                    f"but was placed in {role!r}"
                )
            previous = owners.get(row.identity)
            if previous is not None and previous != role:
                raise ValueError(
                    f"base row {row.identity!r} crosses split roles "
                    f"{previous!r} and {role!r}"
                )
            owners[row.identity] = role


def validate_content_split_separation(
    rows: Mapping[str, Sequence[RowRef]],
) -> None:
    """Reject byte-identical current valid prefixes assigned across roles."""
    owners: dict[str, str] = {}
    for role in ROLES:
        for row in rows[role]:
            if not row.content_sha256:
                continue
            previous = owners.get(row.content_sha256)
            if previous is not None and previous != role:
                raise ValueError(
                    "byte-identical valid I/Q content crosses split roles "
                    f"{previous!r} and {role!r} "
                    f"(sha256={row.content_sha256})"
                )
            owners[row.content_sha256] = role


def _manifest_path(directory: Path) -> Path:
    candidates = [
        path for path in (directory / "corpus.json", directory / "manifest.json")
        if path.is_file()
    ]
    if len(candidates) != 1:
        raise ValueError(
            "current corpus must contain exactly one of corpus.json or "
            "manifest.json"
        )
    return candidates[0]


def _raw_path(directory: Path, manifest: Mapping[str, Any]) -> Path:
    filename = manifest.get("dataFile", "corpus.f32")
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        raise ValueError("dataFile must be a plain filename inside the corpus")
    path = (directory / filename).resolve()
    if path.parent != directory:
        raise ValueError("dataFile escapes the current corpus directory")
    if not path.is_file():
        raise ValueError(f"current corpus data file is missing: {path.name}")
    return path


def _current_identity(
    item: Mapping[str, Any],
    *,
    class_name: str,
    profile_id: str,
    raw_row: np.ndarray,
    valid_sample_count: int,
) -> str:
    """Prefer an explicit acquisition identity, else hash the valid bytes."""
    for field in ("baseRowId", "captureId", "measurementId"):
        if field not in item:
            continue
        value = item[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string when present")
        return f"current:{field}:{value.strip()}"
    valid = np.asarray(
        raw_row[:valid_sample_count],
        dtype="<f4",
    )
    digest = hashlib.sha256()
    digest.update(class_name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(profile_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(valid.tobytes(order="C"))
    return f"current:content:{digest.hexdigest()}"


def _valid_content_sha256(
    raw_row: np.ndarray,
    valid_sample_count: int,
) -> str:
    valid = np.asarray(raw_row[:valid_sample_count], dtype="<f4")
    return hashlib.sha256(valid.tobytes(order="C")).hexdigest()


def _normalized_role(value: Any) -> str:
    if not isinstance(value, str) or value not in ROLE_ALIASES:
        raise ValueError(
            "splitRole must be one of train, enroll, enrollment, or selection; "
            "test/consumed-test roles are never admitted"
        )
    return ROLE_ALIASES[value]


def _profile_id(item: Mapping[str, Any], row_index: int) -> str:
    """Accept the historical ``profile`` or explicit ``profileId`` spelling."""
    supplied = [
        (field, item[field])
        for field in ("profileId", "profile")
        if field in item
    ]
    if not supplied:
        raise ValueError(
            f"manifest item {row_index} must declare profileId or profile"
        )
    values: list[str] = []
    for field, value in supplied:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"manifest item {row_index} {field} must be a non-empty string"
            )
        values.append(value.strip())
    if len(set(values)) != 1:
        raise ValueError(
            f"manifest item {row_index} profileId/profile disagree"
        )
    return values[0]


def load_current_corpus(
    corpus_dir: str | Path,
    *,
    class_index: Mapping[str, int],
) -> RawCorpus:
    """Load and fully validate a separate current-source corpus."""
    directory = Path(corpus_dir).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"current corpus directory does not exist: {directory}")
    manifest_path = _manifest_path(directory)
    manifest = _strict_json_object(manifest_path)
    _validate_current_manifest_contract(manifest)
    for field in ("count", "sampleCount", "items"):
        if field not in manifest:
            raise ValueError(f"current corpus manifest is missing {field}")
    count = _strict_positive_integer(manifest["count"], "count")
    sample_count = _strict_positive_integer(
        manifest["sampleCount"], "sampleCount"
    )
    if sample_count < MINIMUM_INPUT_LENGTH or sample_count > 65_536:
        raise ValueError("sampleCount is outside the service generator contract")
    target_per_profile = _strict_positive_integer(
        manifest.get("targetPerProfile"),
        "targetPerProfile",
    )
    if target_per_profile % len(CURRENT_RECEIVER_PRESETS) != 0:
        raise ValueError("targetPerProfile must be a positive multiple of nine")
    if count != target_per_profile * len(CURRENT_PROFILES):
        raise ValueError(
            "count must equal targetPerProfile times the 31 fixed profiles"
        )
    _strict_nonnegative_integer(manifest.get("corpusSeed"), "corpusSeed")
    items = manifest["items"]
    if not isinstance(items, list) or len(items) != count:
        raise ValueError("manifest count does not match items")
    raw_path = _raw_path(directory, manifest)
    clean_raw_path = _clean_raw_path(directory, manifest)
    expected_bytes = count * sample_count * 2 * np.dtype("<f4").itemsize
    if raw_path.stat().st_size != expected_bytes:
        raise ValueError(
            f"{raw_path.name} has {raw_path.stat().st_size} bytes; "
            f"expected {expected_bytes}"
        )
    if clean_raw_path.stat().st_size != expected_bytes:
        raise ValueError(
            f"{clean_raw_path.name} has {clean_raw_path.stat().st_size} bytes; "
            f"expected {expected_bytes}"
        )
    declared_raw_sha256 = _strict_sha256(
        manifest.get("dataSha256"),
        "dataSha256",
    )
    declared_clean_sha256 = _strict_sha256(
        manifest.get("cleanDataSha256"),
        "cleanDataSha256",
    )
    actual_raw_sha256 = sha256_file(raw_path)
    actual_clean_sha256 = sha256_file(clean_raw_path)
    if actual_raw_sha256 != declared_raw_sha256:
        raise ValueError("corpus.f32 SHA-256 disagrees with dataSha256")
    if actual_clean_sha256 != declared_clean_sha256:
        raise ValueError(
            "corpus_clean.f32 SHA-256 disagrees with cleanDataSha256"
        )
    raw = np.memmap(
        raw_path,
        dtype="<f4",
        mode="r",
        shape=(count, sample_count, 2),
    )
    clean_raw = np.memmap(
        clean_raw_path,
        dtype="<f4",
        mode="r",
        shape=(count, sample_count, 2),
    )
    rows_by_role = {role: [] for role in ROLES}
    padding_rows = 0
    padding_values_verified = 0
    classes_seen: set[str] = set()
    profile_classes: dict[str, str] = {}
    profile_counts = {profile: 0 for profile in CURRENT_PROFILES}
    profile_role_counts_plain = {
        profile: {role: 0 for role in ROLES}
        for profile in CURRENT_PROFILES
    }
    class_counts = {class_name: 0 for class_name in CURRENT_CLASSES}
    prefix_role_owners: dict[tuple[int, str], str] = {}
    prefix_hashes_by_role: dict[str, dict[int, list[str]]] = {
        role: {length: [] for length in RUNTIME_INPUT_LENGTHS}
        for role in ROLES
    }
    for row_index, item_value in enumerate(items):
        if not isinstance(item_value, dict):
            raise ValueError(f"manifest item {row_index} must be an object")
        item = item_value
        missing = [
            field
            for field in (
                "index",
                "cls",
                "profile",
                "profileId",
                "replay",
                "phaseNativeSample",
                "splitRole",
                "validSampleCount",
                "storageSampleCount",
                "zeroPaddedAfterValidSampleCount",
                "contentSha256",
                "storedSha256",
                "cleanContentSha256",
                "cleanStoredSha256",
                "sampleRateHz",
                "signalBandwidthHz",
                "captureBandwidthHz",
                "receiverPreset",
                "receiverRealizationSeed",
                "measurementReceipt",
                "cleanMeasurementReceipt",
            )
            if field not in item
        ]
        if missing:
            raise ValueError(f"manifest item {row_index} is missing {missing}")
        if (
            isinstance(item["index"], (bool, np.bool_))
            or not isinstance(item["index"], (int, np.integer))
            or int(item["index"]) != row_index
        ):
            raise ValueError(
                f"manifest item {row_index} index must equal its row position"
            )
        class_name = item["cls"]
        if not isinstance(class_name, str) or not class_name:
            raise ValueError(f"manifest item {row_index} cls must be non-empty")
        if class_name not in class_index:
            raise ValueError(
                f"manifest item {row_index} has class {class_name!r}, which "
                "is absent from the historical class contract"
            )
        profile_id = _profile_id(item, row_index)
        expected_class = CURRENT_PROFILE_PUBLIC_CLASS_MAP.get(profile_id)
        if expected_class is None:
            raise ValueError(
                f"manifest item {row_index} has unknown fixed profile "
                f"{profile_id!r}"
            )
        if class_name != expected_class:
            raise ValueError(
                f"manifest item {row_index} maps fixed profile {profile_id!r} "
                f"to {class_name!r}; canonical class is {expected_class!r}"
            )
        previous_profile_class = profile_classes.get(profile_id)
        if (
            previous_profile_class is not None
            and previous_profile_class != class_name
        ):
            raise ValueError(
                f"current profile {profile_id!r} crosses public classes "
                f"{previous_profile_class!r} and {class_name!r}"
            )
        profile_classes[profile_id] = class_name
        role = _normalized_role(item["splitRole"])
        valid = _strict_positive_integer(
            item["validSampleCount"],
            f"items[{row_index}].validSampleCount",
        )
        if valid > sample_count:
            raise ValueError(
                f"manifest item {row_index} validSampleCount {valid} exceeds "
                f"storage sampleCount {sample_count}"
            )
        if item["storageSampleCount"] != sample_count:
            raise ValueError(
                f"manifest item {row_index} storageSampleCount disagrees "
                "with manifest sampleCount"
            )
        if not training_view_lengths(valid):
            raise ValueError(
                f"manifest item {row_index} has no admitted live prefix"
            )
        padding_flag = item.get("zeroPaddedAfterValidSampleCount", False)
        if not isinstance(padding_flag, bool):
            raise ValueError(
                "zeroPaddedAfterValidSampleCount must be a JSON boolean"
            )
        if padding_flag is not (valid < sample_count):
            raise ValueError(
                f"manifest item {row_index} zero-padding flag must equal "
                "(validSampleCount < storageSampleCount)"
            )
        valid_pair = np.asarray(raw[row_index, :valid])
        clean_valid_pair = np.asarray(clean_raw[row_index, :valid])
        if not np.isfinite(valid_pair).all():
            raise ValueError(
                f"manifest item {row_index} valid prefix contains NaN or infinity"
            )
        if not np.isfinite(clean_valid_pair).all():
            raise ValueError(
                f"manifest item {row_index} clean valid prefix contains "
                "NaN or infinity"
            )
        if not np.any(valid_pair[:MINIMUM_INPUT_LENGTH] != 0.0):
            raise ValueError(
                f"manifest item {row_index} has an all-zero live prefix"
            )
        if not np.any(clean_valid_pair[:MINIMUM_INPUT_LENGTH] != 0.0):
            raise ValueError(
                f"manifest item {row_index} clean pair has an all-zero "
                "live prefix"
            )
        if valid < sample_count:
            tail = np.asarray(raw[row_index, valid:])
            clean_tail = np.asarray(clean_raw[row_index, valid:])
            if not np.all(tail == 0.0):
                raise ValueError(
                    f"manifest item {row_index} marked a zero-padded tail, "
                    "but the tail contains a nonzero value"
                )
            if not np.all(clean_tail == 0.0):
                raise ValueError(
                    f"manifest item {row_index} clean pair has a nonzero "
                    "padded tail"
                )
            padding_rows += 1
            padding_values_verified += int(tail.size + clean_tail.size)
        content_sha256 = _valid_content_sha256(
            raw[row_index], valid
        )
        clean_content_sha256 = _valid_content_sha256(
            clean_raw[row_index], valid
        )
        stored_sha256 = hashlib.sha256(
            np.asarray(raw[row_index], dtype="<f4").tobytes(order="C")
        ).hexdigest()
        clean_stored_sha256 = hashlib.sha256(
            np.asarray(clean_raw[row_index], dtype="<f4").tobytes(order="C")
        ).hexdigest()
        item_hashes = {
            "contentSha256": content_sha256,
            "storedSha256": stored_sha256,
            "cleanContentSha256": clean_content_sha256,
            "cleanStoredSha256": clean_stored_sha256,
        }
        for field, expected in item_hashes.items():
            declared = _strict_sha256(
                item.get(field),
                f"items[{row_index}].{field}",
            )
            if declared != expected:
                raise ValueError(
                    f"manifest item {row_index} {field} disagrees with bytes"
                )

        sample_rate_hz = _strict_positive_integer(
            item["sampleRateHz"],
            f"items[{row_index}].sampleRateHz",
        )
        signal_bandwidth_hz = _strict_positive_integer(
            item["signalBandwidthHz"],
            f"items[{row_index}].signalBandwidthHz",
        )
        capture_bandwidth_hz = _strict_positive_integer(
            item["captureBandwidthHz"],
            f"items[{row_index}].captureBandwidthHz",
        )
        if (
            signal_bandwidth_hz > capture_bandwidth_hz
            or capture_bandwidth_hz > sample_rate_hz
        ):
            raise ValueError(
                f"manifest item {row_index} has impossible bandwidth geometry"
            )
        receiver_preset = item["receiverPreset"]
        if receiver_preset not in CURRENT_RECEIVER_PRESETS:
            raise ValueError(
                f"manifest item {row_index} has an unknown receiverPreset"
            )
        replay = item["replay"]
        if replay not in {"cyclic", "one-shot"}:
            raise ValueError(
                f"manifest item {row_index} has an invalid replay contract"
            )
        phase_native_sample = _strict_nonnegative_integer(
            item["phaseNativeSample"],
            f"items[{row_index}].phaseNativeSample",
        )
        receiver_seed_value = item["receiverRealizationSeed"]
        receiver_seed = (
            None
            if receiver_seed_value is None
            else _strict_positive_integer(
                receiver_seed_value,
                f"items[{row_index}].receiverRealizationSeed",
            )
        )
        if (receiver_preset == "clean") is not (receiver_seed is None):
            raise ValueError(
                f"manifest item {row_index} receiver seed/preset disagree"
            )
        _validate_measurement_receipt(
            item["measurementReceipt"],
            item_index=row_index,
            field="measurementReceipt",
            valid_sample_count=valid,
            sample_rate_hz=sample_rate_hz,
            content_sha256=content_sha256,
            source_content_sha256=clean_content_sha256,
            receiver_preset=str(receiver_preset),
            replay=str(replay),
            phase_native_sample=phase_native_sample,
            receiver_seed=receiver_seed,
        )
        _validate_measurement_receipt(
            item["cleanMeasurementReceipt"],
            item_index=row_index,
            field="cleanMeasurementReceipt",
            valid_sample_count=valid,
            sample_rate_hz=sample_rate_hz,
            content_sha256=clean_content_sha256,
            source_content_sha256=clean_content_sha256,
            receiver_preset="clean",
            replay=str(replay),
            phase_native_sample=phase_native_sample,
            receiver_seed=None,
        )

        for length in training_view_lengths(valid):
            prefix_sha256 = hashlib.sha256(
                np.asarray(
                    raw[row_index, :length],
                    dtype="<f4",
                ).tobytes(order="C")
            ).hexdigest()
            owner_key = (length, prefix_sha256)
            previous_owner = prefix_role_owners.get(owner_key)
            if previous_owner is not None and previous_owner != role:
                raise ValueError(
                    f"byte-identical {length}-sample live prefix crosses "
                    f"split roles {previous_owner!r} and {role!r}"
                )
            prefix_role_owners[owner_key] = role
            prefix_hashes_by_role[role][length].append(prefix_sha256)
        identity = _current_identity(
            item,
            class_name=class_name,
            profile_id=profile_id,
            raw_row=raw[row_index],
            valid_sample_count=valid,
        )
        rows_by_role[role].append(
            RowRef(
                source="current",
                source_index=row_index,
                role=role,
                class_name=class_name,
                label=int(class_index[class_name]),
                profile_id=profile_id,
                valid_sample_count=valid,
                storage_sample_count=sample_count,
                identity=identity,
                content_sha256=content_sha256,
                zero_padded_after_valid=padding_flag,
            )
        )
        classes_seen.add(class_name)
        profile_counts[profile_id] += 1
        profile_role_counts_plain[profile_id][role] += 1
        class_counts[class_name] += 1

    for role in ROLES:
        if not rows_by_role[role]:
            raise ValueError(f"current corpus has no {role} rows")
    validate_split_separation(rows_by_role)
    validate_content_split_separation(rows_by_role)

    if classes_seen != set(CURRENT_CLASSES):
        raise ValueError("item classes do not match the four current classes")
    if profile_classes != CURRENT_PROFILE_PUBLIC_CLASS_MAP:
        raise ValueError(
            "item profile inventory/class mapping does not match all 31 "
            "fixed profiles"
        )
    if any(
        value != target_per_profile for value in profile_counts.values()
    ):
        raise ValueError(
            "every fixed profile must have exactly targetPerProfile rows"
        )
    if manifest.get("perClass") != dict(sorted(class_counts.items())):
        raise ValueError("perClass disagrees with item class counts")
    if manifest.get("perProfileRole") != profile_role_counts_plain:
        raise ValueError("perProfileRole disagrees with item split roles")

    role_counts = {
        role: len(rows_by_role[role]) for role in ROLES
    }
    class_role_counts = {
        role: {
            class_name: sum(
                row.class_name == class_name for row in rows_by_role[role]
            )
            for class_name in sorted(classes_seen)
        }
        for role in ROLES
    }
    profile_role_counts = {
        role: {
            f"{row.source}:{row.profile_id}": sum(
                candidate.profile_id == row.profile_id
                for candidate in rows_by_role[role]
            )
            for row in rows_by_role[role]
        }
        for role in ROLES
    }
    identity_hashes = {
        role: sha256_json(sorted(row.identity for row in rows_by_role[role]))
        for role in ROLES
    }
    audit = {
        "source": "current",
        "manifest": manifest_path.name,
        "raw": raw_path.name,
        "manifest_sha256": sha256_file(manifest_path),
        "raw_sha256": actual_raw_sha256,
        "clean_raw": clean_raw_path.name,
        "clean_raw_sha256": actual_clean_sha256,
        "count": count,
        "storage_sample_count": sample_count,
        "roles": role_counts,
        "class_role_counts": class_role_counts,
        "profile_role_counts": profile_role_counts,
        "classes_present": sorted(classes_seen),
        "profiles_present": sorted(
            {row.profile_id for role in ROLES for row in rows_by_role[role]}
        ),
        "profile_public_class_map": dict(sorted(profile_classes.items())),
        "padding_policy": {
            "field": "zeroPaddedAfterValidSampleCount",
            "rows_with_verified_zero_tail": padding_rows,
            "tail_float_values_verified_exact_zero": padding_values_verified,
            "views_never_exceed_valid_sample_count": True,
        },
        "runtime_input_lengths": list(RUNTIME_INPUT_LENGTHS),
        "row_identity_sha256_by_role": identity_hashes,
        "valid_content_sha256_set_by_role": {
            role: sha256_json(
                sorted(row.content_sha256 for row in rows_by_role[role])
            )
            for role in ROLES
        },
        "byte_identical_valid_content_cross_role_count": 0,
        "runtime_prefix_sha256_set_by_role_and_length": {
            role: {
                str(length): sha256_json(
                    sorted(prefix_hashes_by_role[role][length])
                )
                for length in RUNTIME_INPUT_LENGTHS
            }
            for role in ROLES
        },
        "byte_identical_runtime_prefix_cross_role_count": 0,
        "generator_lineage": dict(manifest["generatorLineage"]),
    }
    return RawCorpus(
        source="current",
        directory=directory,
        manifest_path=manifest_path,
        raw_path=raw_path,
        manifest=manifest,
        raw=raw,
        rows_by_role=rows_by_role,
        audit=audit,
    )


def load_historical_exposed(
    corpus_dir: str | Path,
    *,
    patch_length: int,
    patch_count: int,
    target_frac: float,
    seed: int,
) -> tuple[RawCorpus, dict[str, Any]]:
    """Load only the historical rows exposed by invariant_patch_data.load."""
    directory = Path(corpus_dir).expanduser().resolve()
    source = invariant_data.load(
        patch_length=patch_length,
        patch_count=patch_count,
        target_frac=target_frac,
        model_seed=seed,
        build_if_missing=False,
        corpus_dir=directory,
        cache_dir=directory / "_invariant_patch_pools",
        legacy_cache_dir=directory / "_pools",
    )
    if source["data_audit"].get("consumed_test_rows_exposed") != 0:
        raise AssertionError("historical loader exposed consumed-test rows")
    manifest_path = directory / "corpus.json"
    raw_path = directory / "corpus.f32"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    count = _strict_positive_integer(manifest.get("count"), "count")
    sample_count = _strict_positive_integer(
        manifest.get("sampleCount"), "sampleCount"
    )
    if sample_count < RUNTIME_INPUT_LENGTHS[-1]:
        raise ValueError(
            "historical corpus cannot supply the 16,384-sample runtime prefix"
        )
    if not isinstance(manifest.get("items"), list) or len(manifest["items"]) != count:
        raise ValueError("historical corpus manifest count/items mismatch")
    expected_bytes = count * sample_count * 2 * np.dtype("<f4").itemsize
    if raw_path.stat().st_size != expected_bytes:
        raise ValueError("historical corpus.f32 size disagrees with manifest")
    raw = np.memmap(
        raw_path,
        dtype="<f4",
        mode="r",
        shape=(count, sample_count, 2),
    )
    classes = list(source["classes"])
    class_index = {name: index for index, name in enumerate(classes)}
    rows_by_role: dict[str, list[RowRef]] = {role: [] for role in ROLES}
    role_fields = (
        ("train", "tr_idx", "ytr"),
        ("enrollment", "en_idx", "yen"),
        ("selection", "va_idx", "yva"),
    )
    all_indices: list[np.ndarray] = []
    historical_profile_classes: dict[str, str] = {}
    for role, index_field, label_field in role_fields:
        indices = np.asarray(source[index_field], dtype=np.int64)
        labels = np.asarray(source[label_field], dtype=np.int64)
        if indices.ndim != 1 or labels.shape != indices.shape:
            raise ValueError(f"historical {role} index/label shape mismatch")
        if np.any(indices < 0) or np.any(indices >= count):
            raise ValueError(f"historical {role} has an out-of-range row")
        all_indices.append(indices)
        # Shape equality above is the explicit length assertion.  Keep this
        # compatible with the canonical Python 3.9 training environment,
        # which predates zip(strict=True).
        for index, label in zip(indices, labels):
            class_name = classes[int(label)]
            if class_index[class_name] != int(label):
                raise AssertionError("historical class indexing changed")
            item = manifest["items"][int(index)]
            if not isinstance(item, dict):
                raise ValueError(
                    f"historical manifest item {int(index)} is not an object"
                )
            profile_id = _profile_id(item, int(index))
            previous_profile_class = historical_profile_classes.get(profile_id)
            if (
                previous_profile_class is not None
                and previous_profile_class != class_name
            ):
                raise ValueError(
                    f"historical profile {profile_id!r} crosses classes"
                )
            historical_profile_classes[profile_id] = class_name
            rows_by_role[role].append(
                RowRef(
                    source="historical",
                    source_index=int(index),
                    role=role,
                    class_name=class_name,
                    label=int(label),
                    profile_id=profile_id,
                    valid_sample_count=sample_count,
                    storage_sample_count=sample_count,
                    identity=f"historical:corpus-row:{int(index)}",
                    content_sha256="",
                    zero_padded_after_valid=False,
                )
            )
    concatenated = np.concatenate(all_indices)
    if len(np.unique(concatenated)) != len(concatenated):
        raise AssertionError(
            "historical exposed train/enrollment/selection rows overlap"
        )
    validate_split_separation(rows_by_role)
    audit = {
        "source": "historical",
        "manifest_sha256": sha256_file(manifest_path),
        "raw_sha256": sha256_file(raw_path),
        "storage_sample_count": sample_count,
        "roles": {role: len(rows_by_role[role]) for role in ROLES},
        "profile_role_counts": {
            role: {
                f"historical:{row.profile_id}": sum(
                    candidate.profile_id == row.profile_id
                    for candidate in rows_by_role[role]
                )
                for row in rows_by_role[role]
            }
            for role in ROLES
        },
        "profile_public_class_map":
            dict(sorted(historical_profile_classes.items())),
        "exposed_index_sha256_by_role": {
            role: hashlib.sha256(
                np.asarray(
                    [row.source_index for row in rows_by_role[role]],
                    dtype="<i8",
                ).tobytes()
            ).hexdigest()
            for role in ROLES
        },
        "consumed_test_rows_exposed": 0,
        "cache_contract": source["cache_contract"],
        "data_audit": source["data_audit"],
    }
    corpus = RawCorpus(
        source="historical",
        directory=directory,
        manifest_path=manifest_path,
        raw_path=raw_path,
        manifest=manifest,
        raw=raw,
        rows_by_role=rows_by_role,
        audit=audit,
    )
    return corpus, {
        "classes": classes,
        "n_classes": int(source["n_classes"]),
        "cache_contract": source["cache_contract"],
        "data_audit": source["data_audit"],
    }


def rows_by_class(
    rows: Iterable[RowRef],
    n_classes: int,
) -> list[list[RowRef]]:
    result = [[] for _ in range(n_classes)]
    for row in rows:
        if row.label < 0 or row.label >= n_classes:
            raise ValueError(f"row {row.identity} has an invalid label")
        result[row.label].append(row)
    return result
