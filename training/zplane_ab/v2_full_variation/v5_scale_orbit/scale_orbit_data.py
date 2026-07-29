"""Load fresh SignalLab scale families as leak-safe v5 training roles.

The scale generator emits four service-path sample-rate views for every
physical realization.  This loader keeps all four views in the same split by
assigning the split from ``realization_index`` before exposing any row to the
trainer:

* seed 20264101, realizations 0..39: training
* seed 20264101, realizations 40..63: enrollment
* seed 20262904, all realizations: adaptive development selection

Rows sharing a ``pair_id`` also share one training identity.  The episodic
trainer can therefore sample a scale view of an identity without ever treating
the other scale views as independent examples.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
V4 = HERE.parent / "v4_current_source"
for path in (V4,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import current_source_data as current_data  # noqa: E402
import evaluate_current_scale as scale_data  # noqa: E402


SCALE_ORBIT_TRAINING_SEED = 20_264_101
SCALE_ORBIT_SELECTION_SEED = 20_262_904
SCALE_ORBIT_REALIZATIONS_PER_PROFILE = 64
SCALE_ORBIT_TRAINING_SPLIT_COUNTS = {
    "train": 40,
    "enrollment": 24,
    "selection": 0,
}
SCALE_ORBIT_SELECTION_SPLIT_COUNTS = {
    "train": 0,
    "enrollment": 0,
    "selection": 64,
}
SCALE_ORBIT_EXPECTED_FACTORS = (1.0, 1.25, 1.5, 2.0)
SCALE_ORBIT_AUDIT_SCHEMA = (
    "v5-current-service-scale-orbit-composite-audit-v1"
)
SCALE_ORBIT_AMENDMENT_PATH = HERE / "pretraining_amendment.json"
SCALE_ORBIT_IDENTITY_FIREWALL_AMENDMENT_PATH = (
    HERE / "identity_firewall_amendment.json"
)
SCALE_ORBIT_IDENTITY_REJECTION_PATH = (
    HERE / "seed20262904_identity_rejection.json"
)
SCALE_ORBIT_IDENTITY_ACCEPTANCE_PATH = (
    HERE / "seed20262904_identity_firewall_acceptance.json"
)
SCALE_ORBIT_PROTOCOL_PATH = HERE / "recovery_protocol.json"
SCALE_ORBIT_SEED_REGISTRY_PATH = HERE / "seed_registry.json"
SCALE_ORBIT_AMENDMENT_SHA256 = (
    "3e4098d0475477bf1ba26e79992ca17848c16dd77deb44b7562c4452544003fd"
)
SCALE_ORBIT_IDENTITY_FIREWALL_AMENDMENT_SHA256 = (
    "89d714910dc5dfdbfb05282abf752382cf382e88108d0f2b589f62289d452ba4"
)
SCALE_ORBIT_IDENTITY_REJECTION_SHA256 = (
    "9f67115eaa07b97f35bcc3cc2f8c3ebd96a0bc819ab47500b98953aa57d31bc5"
)
SCALE_ORBIT_IDENTITY_ACCEPTANCE_SHA256 = (
    "9da7e5b08bb6d5f15e80e920660315d43efe859b4db160d7e40f7b673c741d9b"
)
SCALE_ORBIT_PROTOCOL_SHA256 = (
    "cc79496ba398e4ce2fcb6875d8b810bcbc1ae7f89cb75271fdf2c65e6ad2afff"
)
SCALE_ORBIT_SEED_REGISTRY_SHA256 = (
    "b832e09e8eba264e21d7845f72c0d52f671aa1f9187d55ddce94fb833347e0cd"
)
SCALE_ORBIT_REJECTED_SELECTION_MANIFEST_SHA256 = (
    "ba273ec50520b6416be40146cf595b147c7b767ad94655bec44482f7edaec750"
)
SCALE_ORBIT_REJECTED_SELECTION_RAW_SHA256 = (
    "14ae949ebcf9a56b13cddb4d9b7bcb3c725e36d15c39e8f56e311b905fe04be6"
)
SCALE_ORBIT_IDENTITY_FIREWALLED_SELECTION_MANIFEST_SHA256: str | None = (
    "7a345bc109e46661427c2f81b46125fb48a00048e9b2b7da77505dc49135d998"
)
SCALE_ORBIT_IDENTITY_FIREWALLED_SELECTION_RAW_SHA256: str | None = (
    "53d4ab20faf031ed2aacd3017d55ada1622d7c51f85b200aa490f3d3854dc928"
)
SCALE_ORBIT_TRAINING_MANIFEST_SHA256 = (
    "c61bce0c32ae8728b0c7212da9dd59bc1689fb6dd370c2a27bf8650a7647a353"
)
SCALE_ORBIT_TRAINING_RAW_SHA256 = (
    "03df2b77e6d48cebb434626a17dfc3862f1d062dfbd7b59d9f35b9ce98eda080"
)
SCALE_ORBIT_IDENTITY_FIREWALLED_SELECTION_DIRECTORY = (
    "training/artifacts/"
    "signallab-current-scale-dev-v5-seed20262904-r64-identity-firewalled"
)
SCALE_ORBIT_REFERENCE_DIRECTORY = (
    "training/artifacts/"
    "signallab-current-94171ba-lineage-v1-seed20260729-n288-s16384"
)
SCALE_ORBIT_REFERENCE_SEED = 20_260_729
SCALE_ORBIT_REFERENCE_COUNT = 8_928
SCALE_ORBIT_REFERENCE_MANIFEST_SHA256 = (
    "5b72b98e7275d408e1e47c9922de756aec6fd48f69eedbc1593ab7c484271bea"
)
SCALE_ORBIT_REFERENCE_RAW_SHA256 = (
    "61ae9300c20af4e296cbb68e36239ae66c3272bb84e20e11d648fb71ca95d8d2"
)
SCALE_ORBIT_FIREWALL_GENERATOR_SOURCE_SHA256 = (
    "53f076da486c73bd3088d0ad2359d399736c0eaa56f5323d350d1ebca57bcd34"
)
SCALE_ORBIT_FIREWALL_GENERATOR_BUNDLE_SHA256 = (
    "621fa092a9513e6f54270120e9005df15962a1f99602b6df9ac493887910fd44"
)
SCALE_ORBIT_TRAINING_GENERATOR_LINEAGE = (
    scale_data.DEFAULT_SCALE_GENERATOR_LINEAGE
)
SCALE_ORBIT_FIREWALL_GENERATOR_LINEAGE = (
    scale_data.ScaleGeneratorLineageBinding(
        source_sha256=SCALE_ORBIT_FIREWALL_GENERATOR_SOURCE_SHA256,
        bundle_sha256=SCALE_ORBIT_FIREWALL_GENERATOR_BUNDLE_SHA256,
    )
)
SCALE_ORBIT_REQUIRED_AUDIT_KEYS = (
    "schema",
    "lineage",
    "amendment_sha256",
    "parent_protocol_sha256",
    "seed_registry_sha256",
    "corpora",
    "roles",
    "rows_by_role",
    "pair_identities_by_role",
    "profile_role_counts",
    "profiles_present",
    "profile_public_class_map",
    "classes_present",
    "public_class_order",
    "scale_factors",
    "runtime_input_lengths",
    "identity_separation",
    "fitting_firewall",
    "development_only",
    "release_evidence",
    "consumed_test_rows_exposed",
)
SCALE_ORBIT_REQUIRED_CORPUS_BINDING_KEYS = (
    "eval_seed",
    "manifest_relative_path",
    "manifest_sha256",
    "raw_relative_path",
    "raw_sha256",
    "registry_allocation",
    "statistical_role",
    "allowed_roles",
)
SCALE_ORBIT_REQUIRED_IDENTITY_AUDIT_KEYS = (
    "applicable_collision_counts",
    "inapplicable_keys_by_replay",
    "prefix_lengths_hashed",
    "paired_views_deduplicated",
    "all_applicable_collision_counts_zero",
)
SCALE_ORBIT_FITTING_FIREWALL = {
    "adaptive_selection_rows_used_for_weight_fit": 0,
    "adaptive_selection_rows_used_for_prototype_or_enrollment_fit": 0,
    "adaptive_selection_rows_used_for_threshold_or_rank_fit": 0,
    "sealed_validation_rows_used_for_any_fit_or_selection": 0,
}


@dataclass(frozen=True)
class ScaleOrbitRowRef:
    """One scale view with the common fields consumed by the v4 trainer."""

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
    pair_id: str
    realization_index: int
    physical_scale_factor: float
    sample_rate_hz: int
    native_sample_rate_hz: int
    capture_bandwidth_hz: int
    population_seed: int
    replay: str
    phase_native_sample: int
    receiver_channel_seed: int
    receiver_seed: int | None


@dataclass
class ScaleOrbitCorpus:
    """Composite train/enrollment/development corpus over two memmaps."""

    source: str
    directory: Path
    manifest_path: Path
    raw_path: Path
    manifest: dict[str, Any]
    raw: np.memmap
    rows_by_role: dict[str, list[ScaleOrbitRowRef]]
    audit: dict[str, Any]
    populations: dict[int, scale_data.ScaleEvalCorpus]

    def prefix(self, row: ScaleOrbitRowRef, length: int) -> np.ndarray:
        if row.source != self.source:
            raise ValueError(
                f"row source {row.source!r} does not belong to {self.source!r}"
            )
        population = self.populations.get(row.population_seed)
        if population is None:
            raise ValueError(
                f"row population seed {row.population_seed} is not bound"
            )
        if (
            isinstance(length, bool)
            or not isinstance(length, (int, np.integer))
            or int(length) not in current_data.RUNTIME_INPUT_LENGTHS
        ):
            raise ValueError(
                "scale-orbit prefix length must be an exact live runtime "
                f"bucket {current_data.RUNTIME_INPUT_LENGTHS}"
            )
        requested = int(length)
        if (
            row.source_index < 0
            or row.source_index >= len(population.rows)
            or requested > row.valid_sample_count
        ):
            raise ValueError(
                f"invalid N{requested} view for scale-orbit row {row.identity}"
            )
        pair = np.asarray(
            population.raw[row.source_index, :requested],
            dtype=np.float64,
        )
        if (
            pair.shape != (requested, 2)
            or not np.isfinite(pair).all()
        ):
            raise ValueError(
                f"invalid N{requested} view for scale-orbit row {row.identity}"
            )
        return pair[:, 0] + 1j * pair[:, 1]


def role_for_training_realization(realization_index: int) -> str:
    """Return seed-20264101's amended train/enrollment role."""
    if (
        isinstance(realization_index, bool)
        or not isinstance(realization_index, (int, np.integer))
    ):
        raise ValueError("realization_index must be an integer")
    index = int(realization_index)
    if not 0 <= index < SCALE_ORBIT_REALIZATIONS_PER_PROFILE:
        raise ValueError(
            "realization_index is outside the preregistered v5 range"
        )
    train_end = SCALE_ORBIT_TRAINING_SPLIT_COUNTS["train"]
    if index < train_end:
        return "train"
    return "enrollment"


def _require_exact_class_map(class_index: Mapping[str, int]) -> None:
    expected = {
        name: index for index, name in enumerate(scale_data.PUBLIC_CLASSES)
    }
    if dict(class_index) != expected:
        raise ValueError(
            "v5 scale-orbit class index/order must equal the frozen public "
            f"class order {list(scale_data.PUBLIC_CLASSES)}"
        )


def _audit_pairs(
    corpus: scale_data.ScaleEvalCorpus,
    *,
    split_counts: Mapping[str, int],
    role_for_index: Any,
) -> dict[str, Any]:
    by_pair: dict[str, list[scale_data.ScaleEvalRow]] = {}
    for row in corpus.rows:
        by_pair.setdefault(row.pair_id, []).append(row)
    expected_pair_count = (
        len(current_data.CURRENT_PROFILES)
        * SCALE_ORBIT_REALIZATIONS_PER_PROFILE
    )
    if len(by_pair) != expected_pair_count:
        raise ValueError(
            f"v5 scale-orbit corpus has {len(by_pair)} pairs; "
            f"expected {expected_pair_count}"
        )

    pairs_by_profile_role = {
        profile: {role: 0 for role in current_data.ROLES}
        for profile in current_data.CURRENT_PROFILES
    }
    realization_indices_by_profile = {
        profile: [] for profile in current_data.CURRENT_PROFILES
    }
    for pair_id, rows in sorted(by_pair.items()):
        first = rows[0]
        observed_factors = tuple(
            sorted(float(row.scale_factor) for row in rows)
        )
        if (
            first.profile not in pairs_by_profile_role
            or len(rows) != len(SCALE_ORBIT_EXPECTED_FACTORS)
            or observed_factors != SCALE_ORBIT_EXPECTED_FACTORS
            or len({row.content_sha256 for row in rows})
                != len(SCALE_ORBIT_EXPECTED_FACTORS)
            or any(
                row.profile != first.profile
                or row.class_name != first.class_name
                or row.realization_index != first.realization_index
                or row.pair_id != pair_id
                or row.replay != first.replay
                or row.phase_native_sample != first.phase_native_sample
                or row.channel_seed != first.channel_seed
                or row.receiver_seed != first.receiver_seed
                or row.native_sample_rate_hz
                    != first.native_sample_rate_hz
                or row.capture_bandwidth_hz
                    != first.capture_bandwidth_hz
                for row in rows
            )
        ):
            raise ValueError(
                f"v5 scale-orbit pair {pair_id!r} lost exact four-view "
                "identity/geometry alignment"
            )
        role = role_for_index(first.realization_index)
        pairs_by_profile_role[first.profile][role] += 1
        realization_indices_by_profile[first.profile].append(
            int(first.realization_index)
        )

    expected_indices = list(range(SCALE_ORBIT_REALIZATIONS_PER_PROFILE))
    for profile, counts in pairs_by_profile_role.items():
        if counts != dict(split_counts):
            raise ValueError(
                f"v5 scale-orbit profile {profile!r} split counts changed: "
                f"{counts}"
            )
        if sorted(realization_indices_by_profile[profile]) != expected_indices:
            raise ValueError(
                f"v5 scale-orbit profile {profile!r} does not contain the "
                "exact realization-index set [0,64)"
            )
    return {
        "pair_count": len(by_pair),
        "pairs_by_profile_role": pairs_by_profile_role,
        "exact_four_scale_views_per_pair": True,
        "exact_realization_index_set_per_profile": True,
        "split_unit": "pair_id",
        "scale_factors": list(SCALE_ORBIT_EXPECTED_FACTORS),
    }


def _selection_role(realization_index: int) -> str:
    if (
        isinstance(realization_index, bool)
        or not isinstance(realization_index, (int, np.integer))
        or not 0 <= int(realization_index)
            < SCALE_ORBIT_REALIZATIONS_PER_PROFILE
    ):
        raise ValueError("selection realization_index is outside [0,64)")
    return "selection"


def _validate_population(
    directory: str | Path,
    *,
    expected_seed: int,
    expected_generator_lineage: scale_data.ScaleGeneratorLineageBinding,
    split_counts: Mapping[str, int],
    role_for_index: Any,
) -> tuple[scale_data.ScaleEvalCorpus, dict[str, Any]]:
    source = Path(directory).expanduser().resolve()
    corpus = scale_data.load_scale_eval_corpus(
        source,
        expected_generator_lineage=expected_generator_lineage,
    )
    if (
        corpus.directory.resolve() != source
        or corpus.manifest_path.resolve().parent != source
        or corpus.raw_path.resolve().parent != source
        or corpus.manifest.get("evalSeed") != expected_seed
        or corpus.audit.get("realizations_per_profile")
            != SCALE_ORBIT_REALIZATIONS_PER_PROFILE
        or corpus.audit.get("unbound_smoke") is not False
        or corpus.audit.get("reference_bound") is not True
        or corpus.audit.get("profiles")
            != list(current_data.CURRENT_PROFILES)
        or corpus.audit.get("profile_public_class_map")
            != dict(sorted(
                current_data.CURRENT_PROFILE_PUBLIC_CLASS_MAP.items()
            ))
        or corpus.audit.get("classes_present")
            != sorted(set(
                current_data.CURRENT_PROFILE_PUBLIC_CLASS_MAP.values()
            ))
        or corpus.audit.get("scale_factors")
            != list(SCALE_ORBIT_EXPECTED_FACTORS)
        or corpus.audit.get("prefix_lengths")
            != list(current_data.RUNTIME_INPUT_LENGTHS)
        or np.ndim(corpus.raw) != 3
        or corpus.raw.shape[0] != len(corpus.rows)
        or corpus.raw.shape[1] < current_data.RUNTIME_INPUT_LENGTHS[-1]
        or corpus.raw.shape[2] != 2
        or [row.index for row in corpus.rows]
            != list(range(len(corpus.rows)))
    ):
        raise ValueError(
            f"scale corpus is not preregistered population {expected_seed}"
        )
    return corpus, _audit_pairs(
        corpus,
        split_counts=split_counts,
        role_for_index=role_for_index,
    )


def _validate_identity_firewalled_selection(
    training: scale_data.ScaleEvalCorpus,
    selection: scale_data.ScaleEvalCorpus,
) -> None:
    """Bind the replacement bytes to the append-only firewall amendment."""
    accepted_manifest_sha256 = (
        SCALE_ORBIT_IDENTITY_FIREWALLED_SELECTION_MANIFEST_SHA256
    )
    accepted_raw_sha256 = SCALE_ORBIT_IDENTITY_FIREWALLED_SELECTION_RAW_SHA256
    if accepted_manifest_sha256 is None or accepted_raw_sha256 is None:
        raise ValueError(
            "identity-firewalled selection exact manifest/raw SHA-256 pins "
            "are not bound"
        )
    for field, value in (
        ("manifest", accepted_manifest_sha256),
        ("raw", accepted_raw_sha256),
    ):
        if (
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(
                f"identity-firewalled selection {field} SHA-256 pin is invalid"
            )
    if (
        accepted_manifest_sha256
        == SCALE_ORBIT_REJECTED_SELECTION_MANIFEST_SHA256
        or accepted_raw_sha256 == SCALE_ORBIT_REJECTED_SELECTION_RAW_SHA256
    ):
        raise ValueError(
            "identity-firewalled selection pins still name quarantined bytes"
        )
    try:
        selection_relative = (
            selection.directory.resolve().relative_to(REPO).as_posix()
        )
    except ValueError as error:
        raise ValueError(
            "identity-firewalled selection corpus is outside the repository"
        ) from error
    if (
        training.audit.get("manifest_sha256")
        != SCALE_ORBIT_TRAINING_MANIFEST_SHA256
        or training.audit.get("raw_sha256")
        != SCALE_ORBIT_TRAINING_RAW_SHA256
        or selection_relative
        != SCALE_ORBIT_IDENTITY_FIREWALLED_SELECTION_DIRECTORY
        or selection.audit.get("manifest_sha256")
        != accepted_manifest_sha256
        or selection.audit.get("raw_sha256") != accepted_raw_sha256
    ):
        raise ValueError(
            "scale-orbit corpora do not match the frozen identity-firewall "
            "recovery boundary"
        )
    generator = selection.manifest.get("generatorLineage")
    held_out = selection.manifest.get("heldOutContract")
    reference = selection.manifest.get("referenceCorpus")
    exclusions = selection.manifest.get("scaleIdentityExclusionCorpora")
    if (
        not isinstance(generator, Mapping)
        or generator.get("sourceSha256")
        != SCALE_ORBIT_FIREWALL_GENERATOR_SOURCE_SHA256
        or generator.get("bundleSha256")
        != SCALE_ORBIT_FIREWALL_GENERATOR_BUNDLE_SHA256
        or not isinstance(held_out, Mapping)
        or held_out.get("referenceBound") is not True
        or held_out.get("scaleIdentityExclusionCount") != 1
        or not isinstance(reference, Mapping)
        or not isinstance(exclusions, list)
        or len(exclusions) != 1
        or not isinstance(exclusions[0], Mapping)
    ):
        raise ValueError(
            "adaptive selection lacks the frozen generator identity firewall"
        )
    if (
        reference.get("manifest") != "corpus.json"
        or reference.get("manifestSha256")
        != SCALE_ORBIT_REFERENCE_MANIFEST_SHA256
        or reference.get("rawSha256") != SCALE_ORBIT_REFERENCE_RAW_SHA256
        or reference.get("count") != SCALE_ORBIT_REFERENCE_COUNT
        or reference.get("corpusSeed") != SCALE_ORBIT_REFERENCE_SEED
        or Path(str(reference.get("directory"))).resolve()
        != (REPO / SCALE_ORBIT_REFERENCE_DIRECTORY).resolve()
    ):
        raise ValueError(
            "adaptive selection reference corpus is not bound to the exact "
            "seed-20260729 reference bytes"
        )
    exclusion = exclusions[0]
    if (
        exclusion.get("evalSeed") != SCALE_ORBIT_TRAINING_SEED
        or exclusion.get("manifest") != "scale_eval.json"
        or exclusion.get("manifestSha256")
        != SCALE_ORBIT_TRAINING_MANIFEST_SHA256
        or exclusion.get("rawSha256") != SCALE_ORBIT_TRAINING_RAW_SHA256
        or exclusion.get("count")
        != (
            len(current_data.CURRENT_PROFILES)
            * SCALE_ORBIT_REALIZATIONS_PER_PROFILE
            * len(SCALE_ORBIT_EXPECTED_FACTORS)
        )
        or Path(str(exclusion.get("directory"))).resolve()
        != training.directory.resolve()
    ):
        raise ValueError(
            "adaptive selection identity exclusion is not bound to the exact "
            "seed-20264101 training corpus"
        )


def _prefix_sha256(
    corpus: scale_data.ScaleEvalCorpus,
    row: scale_data.ScaleEvalRow,
    length: int,
) -> str:
    if (
        length not in row.available_lengths
        or length not in current_data.RUNTIME_INPUT_LENGTHS
        or length > row.valid_sample_count
    ):
        raise ValueError(
            f"row {row.index} does not admit audited prefix {length}"
        )
    values = np.ascontiguousarray(
        corpus.raw[row.index, :length],
        dtype="<f4",
    )
    if values.shape != (length, 2) or not np.isfinite(values).all():
        raise ValueError(f"row {row.index} audited prefix {length} is invalid")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _identity_inventory(
    corpus: scale_data.ScaleEvalCorpus,
    *,
    role_for_index: Any,
) -> dict[str, dict[str, set[Any]]]:
    inventory = {
        role: {
            "pair_id": set(),
            "content_sha256": set(),
            "runtime_prefix_sha256": set(),
            "cyclic_profile_and_phase_native_sample": set(),
            "receiver_realization_channel_seed": set(),
            "receiver_realization_seed_when_non_null": set(),
        }
        for role in current_data.ROLES
    }
    first_by_pair: dict[str, scale_data.ScaleEvalRow] = {}
    for row in corpus.rows:
        role = role_for_index(row.realization_index)
        target = inventory[role]
        target["content_sha256"].add(row.content_sha256)
        for length in row.available_lengths:
            target["runtime_prefix_sha256"].add(
                (
                    int(length),
                    _prefix_sha256(corpus, row, int(length)),
                )
            )
        first_by_pair.setdefault(row.pair_id, row)
    for row in first_by_pair.values():
        role = role_for_index(row.realization_index)
        target = inventory[role]
        target["pair_id"].add(row.pair_id)
        target["receiver_realization_channel_seed"].add(row.channel_seed)
        if row.receiver_seed is not None:
            target["receiver_realization_seed_when_non_null"].add(
                row.receiver_seed
            )
        if row.replay == "cyclic":
            target["cyclic_profile_and_phase_native_sample"].add(
                (row.profile, row.phase_native_sample)
            )
        elif row.replay == "one-shot":
            if row.phase_native_sample != 0:
                raise ValueError(
                    "one-shot scale rows must retain fixed origin phase zero"
                )
        else:
            raise ValueError(f"unsupported replay kind {row.replay!r}")
    return inventory


def _assert_cross_role_separation(
    inventories: Mapping[str, Mapping[str, set[Any]]],
) -> dict[str, Any]:
    roles = tuple(current_data.ROLES)
    internal_keys = (
        "pair_id",
        "content_sha256",
        "runtime_prefix_sha256",
        "cyclic_profile_and_phase_native_sample",
        "receiver_realization_channel_seed",
        "receiver_realization_seed_when_non_null",
    )
    if (
        set(inventories) != set(roles)
        or any(
            set(inventories[role]) != set(internal_keys)
            for role in roles
        )
    ):
        raise ValueError("identity inventory keys do not match the amendment")
    internal_counts: dict[str, int] = {}
    for key in internal_keys:
        collisions: set[Any] = set()
        for left_index, left in enumerate(roles):
            for right in roles[left_index + 1:]:
                collisions.update(
                    inventories[left][key] & inventories[right][key]
                )
        internal_counts[key] = len(collisions)
    applicable_collision_counts = {
        "pair_id": internal_counts["pair_id"],
        "content_sha256": internal_counts["content_sha256"],
        "runtime_prefix_sha256":
            internal_counts["runtime_prefix_sha256"],
        "receiver_realization_channel_seed":
            internal_counts["receiver_realization_channel_seed"],
        "receiver_realization_seed_when_non_null":
            internal_counts["receiver_realization_seed_when_non_null"],
        "cyclic_base_transmitter_identity":
            internal_counts[
                "cyclic_profile_and_phase_native_sample"
            ],
        "cyclic_profile_and_phase_native_sample":
            internal_counts[
                "cyclic_profile_and_phase_native_sample"
            ],
    }
    if any(applicable_collision_counts.values()):
        raise ValueError(
            "v5 scale-orbit train/enrollment/development identity "
            f"collision: {applicable_collision_counts}"
        )
    result = {
        "applicable_collision_counts": applicable_collision_counts,
        "inapplicable_keys_by_replay": {
            "cyclic": ["profile_and_payload_seed"],
            "one-shot": [
                "base_transmitter_identity",
                "profile_and_payload_seed",
                "profile_and_phase_native_sample",
            ],
        },
        "prefix_lengths_hashed": list(current_data.RUNTIME_INPUT_LENGTHS),
        "paired_views_deduplicated": True,
        "all_applicable_collision_counts_zero": True,
    }
    if set(result) != set(SCALE_ORBIT_REQUIRED_IDENTITY_AUDIT_KEYS):
        raise AssertionError("identity audit schema changed")
    return result


def _row_reference(
    row: scale_data.ScaleEvalRow,
    corpus: scale_data.ScaleEvalCorpus,
    *,
    role: str,
    class_index: Mapping[str, int],
    seed: int,
) -> ScaleOrbitRowRef:
    expected_class = current_data.CURRENT_PROFILE_PUBLIC_CLASS_MAP.get(
        row.profile
    )
    if expected_class != row.class_name:
        raise ValueError(f"scale row {row.index} profile/class mapping changed")
    return ScaleOrbitRowRef(
        source="current",
        source_index=int(row.index),
        role=role,
        class_name=row.class_name,
        label=int(class_index[row.class_name]),
        profile_id=row.profile,
        valid_sample_count=int(row.valid_sample_count),
        storage_sample_count=int(corpus.raw.shape[1]),
        identity=f"scale-orbit:{seed}:{row.pair_id}",
        content_sha256=row.content_sha256,
        zero_padded_after_valid=(
            int(row.valid_sample_count) < int(corpus.raw.shape[1])
        ),
        pair_id=row.pair_id,
        realization_index=int(row.realization_index),
        physical_scale_factor=float(row.scale_factor),
        sample_rate_hz=int(row.sample_rate_hz),
        native_sample_rate_hz=int(row.native_sample_rate_hz),
        capture_bandwidth_hz=int(row.capture_bandwidth_hz),
        population_seed=seed,
        replay=row.replay,
        phase_native_sample=int(row.phase_native_sample),
        receiver_channel_seed=int(row.channel_seed),
        receiver_seed=(
            None if row.receiver_seed is None else int(row.receiver_seed)
        ),
    )


def _verified_contract_hashes() -> dict[str, str]:
    observed = {
        "amendment_sha256":
            current_data.sha256_file(SCALE_ORBIT_AMENDMENT_PATH),
        "identity_firewall_amendment_sha256":
            current_data.sha256_file(
                SCALE_ORBIT_IDENTITY_FIREWALL_AMENDMENT_PATH
            ),
        "identity_rejection_sha256":
            current_data.sha256_file(SCALE_ORBIT_IDENTITY_REJECTION_PATH),
        "identity_acceptance_sha256":
            current_data.sha256_file(SCALE_ORBIT_IDENTITY_ACCEPTANCE_PATH),
        "parent_protocol_sha256":
            current_data.sha256_file(SCALE_ORBIT_PROTOCOL_PATH),
        "seed_registry_sha256":
            current_data.sha256_file(SCALE_ORBIT_SEED_REGISTRY_PATH),
    }
    expected = {
        "amendment_sha256": SCALE_ORBIT_AMENDMENT_SHA256,
        "identity_firewall_amendment_sha256":
            SCALE_ORBIT_IDENTITY_FIREWALL_AMENDMENT_SHA256,
        "identity_rejection_sha256":
            SCALE_ORBIT_IDENTITY_REJECTION_SHA256,
        "identity_acceptance_sha256":
            SCALE_ORBIT_IDENTITY_ACCEPTANCE_SHA256,
        "parent_protocol_sha256": SCALE_ORBIT_PROTOCOL_SHA256,
        "seed_registry_sha256": SCALE_ORBIT_SEED_REGISTRY_SHA256,
    }
    if observed != expected:
        raise ValueError(
            "v5 scale-orbit frozen identity-firewall contract hash changed"
        )
    return {
        key: observed[key]
        for key in (
            "amendment_sha256",
            "parent_protocol_sha256",
            "seed_registry_sha256",
        )
    }


def _repository_relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError as error:
        raise ValueError(
            f"v5 scale-orbit artifact is outside the repository: {path}"
        ) from error


def _audit_hash(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} is not a lowercase SHA-256")
    return value


def _corpus_binding(
    corpus: scale_data.ScaleEvalCorpus,
    *,
    seed: int,
) -> dict[str, Any]:
    if seed == SCALE_ORBIT_TRAINING_SEED:
        registry_allocation = "training_scale"
        statistical_role = "training_only"
        allowed_roles = ["train", "enrollment"]
    elif seed == SCALE_ORBIT_SELECTION_SEED:
        registry_allocation = "adaptive_scale_development"
        statistical_role = "adaptive_development_only"
        allowed_roles = ["selection"]
    else:
        raise ValueError(f"unregistered scale-orbit seed {seed}")
    result = {
        "eval_seed": seed,
        "manifest_relative_path":
            _repository_relative_path(corpus.manifest_path),
        "manifest_sha256": _audit_hash(
            corpus.audit.get("manifest_sha256"),
            f"seed {seed} manifest_sha256",
        ),
        "raw_relative_path": _repository_relative_path(corpus.raw_path),
        "raw_sha256": _audit_hash(
            corpus.audit.get("raw_sha256"),
            f"seed {seed} raw_sha256",
        ),
        "registry_allocation": registry_allocation,
        "statistical_role": statistical_role,
        "allowed_roles": allowed_roles,
    }
    if set(result) != set(SCALE_ORBIT_REQUIRED_CORPUS_BINDING_KEYS):
        raise AssertionError("scale-orbit corpus binding schema changed")
    return result


def _composite_audit(
    *,
    training: scale_data.ScaleEvalCorpus,
    selection: scale_data.ScaleEvalCorpus,
    rows_by_role: Mapping[str, list[ScaleOrbitRowRef]],
    separation: Mapping[str, Any],
    contract_hashes: Mapping[str, str],
) -> dict[str, Any]:
    row_counts = {
        role: len(rows_by_role[role]) for role in current_data.ROLES
    }
    pair_counts = {
        role: len({row.identity for row in rows_by_role[role]})
        for role in current_data.ROLES
    }
    profile_role_counts = {
        role: {
            profile: sum(
                row.profile_id == profile for row in rows_by_role[role]
            )
            for profile in current_data.CURRENT_PROFILES
        }
        for role in current_data.ROLES
    }
    profiles_present = sorted({
        row.profile_id
        for role in current_data.ROLES
        for row in rows_by_role[role]
    })
    classes_present = sorted({
        row.class_name
        for role in current_data.ROLES
        for row in rows_by_role[role]
    })
    if (
        profiles_present != list(current_data.CURRENT_PROFILES)
        or classes_present
            != sorted(set(
                current_data.CURRENT_PROFILE_PUBLIC_CLASS_MAP.values()
            ))
    ):
        raise ValueError("composite profile/class inventory changed")
    audit = {
        "schema": SCALE_ORBIT_AUDIT_SCHEMA,
        "lineage": "v5-scale-orbit",
        "amendment_sha256": contract_hashes["amendment_sha256"],
        "parent_protocol_sha256":
            contract_hashes["parent_protocol_sha256"],
        "seed_registry_sha256":
            contract_hashes["seed_registry_sha256"],
        "corpora": {
            str(SCALE_ORBIT_TRAINING_SEED): _corpus_binding(
                training,
                seed=SCALE_ORBIT_TRAINING_SEED,
            ),
            str(SCALE_ORBIT_SELECTION_SEED): _corpus_binding(
                selection,
                seed=SCALE_ORBIT_SELECTION_SEED,
            ),
        },
        "roles": {
            str(SCALE_ORBIT_TRAINING_SEED): ["train", "enrollment"],
            str(SCALE_ORBIT_SELECTION_SEED): ["selection"],
        },
        "rows_by_role": row_counts,
        "pair_identities_by_role": pair_counts,
        "profile_role_counts": profile_role_counts,
        "profiles_present": profiles_present,
        "profile_public_class_map": dict(sorted(
            current_data.CURRENT_PROFILE_PUBLIC_CLASS_MAP.items()
        )),
        "classes_present": classes_present,
        "public_class_order": list(scale_data.PUBLIC_CLASSES),
        "scale_factors": list(SCALE_ORBIT_EXPECTED_FACTORS),
        "runtime_input_lengths":
            list(current_data.RUNTIME_INPUT_LENGTHS),
        "identity_separation": dict(separation),
        "fitting_firewall": dict(SCALE_ORBIT_FITTING_FIREWALL),
        "development_only": True,
        "release_evidence": False,
        "consumed_test_rows_exposed": 0,
    }
    if set(audit) != set(SCALE_ORBIT_REQUIRED_AUDIT_KEYS):
        raise AssertionError("scale-orbit composite audit schema changed")
    return audit


def load_scale_orbit_training_corpus(
    directory: str | Path,
    *,
    selection_directory: str | Path,
    class_index: Mapping[str, int],
) -> ScaleOrbitCorpus:
    """Compose training seed 20264101 with adaptive-dev seed 20262904."""
    _require_exact_class_map(class_index)
    training_directory = Path(directory).expanduser().resolve()
    adaptive_directory = Path(selection_directory).expanduser().resolve()
    if training_directory == adaptive_directory:
        raise ValueError(
            "training/enrollment and adaptive selection require distinct "
            "seed-bound corpus directories"
        )
    contract_hashes = _verified_contract_hashes()
    training, training_audit = _validate_population(
        training_directory,
        expected_seed=SCALE_ORBIT_TRAINING_SEED,
        expected_generator_lineage=SCALE_ORBIT_TRAINING_GENERATOR_LINEAGE,
        split_counts=SCALE_ORBIT_TRAINING_SPLIT_COUNTS,
        role_for_index=role_for_training_realization,
    )
    selection, selection_audit = _validate_population(
        adaptive_directory,
        expected_seed=SCALE_ORBIT_SELECTION_SEED,
        expected_generator_lineage=SCALE_ORBIT_FIREWALL_GENERATOR_LINEAGE,
        split_counts=SCALE_ORBIT_SELECTION_SPLIT_COUNTS,
        role_for_index=_selection_role,
    )
    rows_by_role: dict[str, list[ScaleOrbitRowRef]] = {
        role: [] for role in current_data.ROLES
    }
    for corpus, seed, role_resolver in (
        (
            training,
            SCALE_ORBIT_TRAINING_SEED,
            role_for_training_realization,
        ),
        (selection, SCALE_ORBIT_SELECTION_SEED, _selection_role),
    ):
        for row in corpus.rows:
            role = role_resolver(row.realization_index)
            rows_by_role[role].append(
                _row_reference(
                    row,
                    corpus,
                    role=role,
                    class_index=class_index,
                    seed=seed,
                )
            )
    current_data.validate_split_separation(rows_by_role)
    current_data.validate_content_split_separation(rows_by_role)

    training_inventory = _identity_inventory(
        training,
        role_for_index=role_for_training_realization,
    )
    selection_inventory = _identity_inventory(
        selection,
        role_for_index=_selection_role,
    )
    combined_inventory = {
        role: {
            key: (
                training_inventory[role][key]
                | selection_inventory[role][key]
            )
            for key in training_inventory[role]
        }
        for role in current_data.ROLES
    }
    separation = _assert_cross_role_separation(combined_inventory)
    _validate_identity_firewalled_selection(training, selection)
    expected_training_pairs = (
        len(current_data.CURRENT_PROFILES)
        * SCALE_ORBIT_REALIZATIONS_PER_PROFILE
    )
    if (
        training_audit.get("pair_count") != expected_training_pairs
        or selection_audit.get("pair_count") != expected_training_pairs
    ):
        raise AssertionError("validated scale-orbit pair count changed")
    audit = _composite_audit(
        training=training,
        selection=selection,
        rows_by_role=rows_by_role,
        separation=separation,
        contract_hashes=contract_hashes,
    )

    return ScaleOrbitCorpus(
        source="current",
        directory=training.directory,
        manifest_path=training.manifest_path,
        raw_path=training.raw_path,
        manifest={
            "schema": "v5-current-service-scale-orbit-composite-v1",
            "training_seed": SCALE_ORBIT_TRAINING_SEED,
            "selection_seed": SCALE_ORBIT_SELECTION_SEED,
            "seed_to_roles": audit["roles"],
        },
        raw=training.raw,
        rows_by_role=rows_by_role,
        populations={
            SCALE_ORBIT_TRAINING_SEED: training,
            SCALE_ORBIT_SELECTION_SEED: selection,
        },
        audit=audit,
    )
