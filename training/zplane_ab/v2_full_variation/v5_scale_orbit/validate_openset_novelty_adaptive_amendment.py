"""Validate the append-only v5 adaptive-novelty/full-gate intent.

This validator reads only explicitly bound contract, source, policy, and
development-rationale files.  It never opens a corpus, checkpoint, generated
novelty population, or sealed-validation artifact, and it performs no model
inference.  The bound v4 policy is parsed only to derive its stage-one
coefficient hashes; no prototype or calibration row is consumed as evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]

AMENDMENT_PATH = HERE / "openset_novelty_adaptive_amendment.json"
RECOVERY_PROTOCOL_PATH = HERE / "recovery_protocol.json"
SEED_REGISTRY_PATH = HERE / "seed_registry.json"
ATTEMPT_2_INTENT_PATH = HERE / "scale_consistency_adaptation_attempt_2.json"
IDENTITY_ACCEPTANCE_PATH = (
    HERE / "seed20262904_identity_firewall_acceptance.json"
)
PYTHON_CANONICALIZER_PATH = HERE / "trusted_geometry_canonicalizer.py"
TYPESCRIPT_CANONICALIZER_PATH = (
    REPO
    / "src"
    / "embedding"
    / "trusted-current-geometry-canonicalizer-v5.ts"
)
V4_GENERATOR_RATIONALE_SOURCE_PATH = (
    REPO
    / "training"
    / "zplane_ab"
    / "v2_full_variation"
    / "v4_current_source"
    / "calibrate_current_source_openset.py"
)
V4_DESIGN_RATIONALE_PATH = (
    REPO
    / ".artifacts"
    / "v4-openset-schema5-final-independent-design-seed20263001"
    / "design-evaluation.json"
)
V4_CONFIRMATION_RATIONALE_PATH = (
    REPO
    / ".artifacts"
    / "v4-current-source-lineage-share13-openset-confirmation-"
    "seed20263004-final"
    / "design-evaluation.json"
)
V4_STAGE_ONE_POLICY_PATH = (
    REPO
    / ".artifacts"
    / "v4-openset-schema5-final-independent-design-seed20263001"
    / "time-domain-profile-bank-openset-v4.json"
)

EXPECTED_AMENDMENT_SHA256 = (
    "5622458262d51a255fa70f00f1a8baeb150500d32bb9af65db44c641cf4e1154"
)
EXPECTED_RECOVERY_PROTOCOL_SHA256 = (
    "cc79496ba398e4ce2fcb6875d8b810bcbc1ae7f89cb75271fdf2c65e6ad2afff"
)
EXPECTED_SEED_REGISTRY_SHA256 = (
    "b832e09e8eba264e21d7845f72c0d52f671aa1f9187d55ddce94fb833347e0cd"
)
EXPECTED_ATTEMPT_2_INTENT_SHA256 = (
    "5e0cc7bbd58ebe050bf983f854e154c43595dc804aedee3c1c3849cf5691b0a9"
)
EXPECTED_IDENTITY_ACCEPTANCE_SHA256 = (
    "9da7e5b08bb6d5f15e80e920660315d43efe859b4db160d7e40f7b673c741d9b"
)
EXPECTED_PYTHON_CANONICALIZER_SHA256 = (
    "38f0cc8e4a6f866050392c923299d6c3b393e5964bbc6cb25c7fa131873062c5"
)
EXPECTED_TYPESCRIPT_CANONICALIZER_SHA256 = (
    "cbaedce76fc07c844097880e8d64f4223b68a0e92633b18d065034f0861d109a"
)
EXPECTED_V4_GENERATOR_RATIONALE_SOURCE_SHA256 = (
    "06aa949defe59bb38eeb70f6d5fe8b00a56ac41893a3b73300ac3120d2c75215"
)
EXPECTED_V4_DESIGN_RATIONALE_SHA256 = (
    "c0e64ef2bdb7a8b43297af9a2c3d6dcfa775055dfae2e860d8b073d5794493c0"
)
EXPECTED_V4_CONFIRMATION_RATIONALE_SHA256 = (
    "fcdc78ee55ee3ecfad2f61ae74c5ecd44f80fddb573cd3ee88f0433ca38192fe"
)
EXPECTED_V4_STAGE_ONE_POLICY_SHA256 = (
    "57f3262bb6c10785a766ef6834b88fc2c8daaeae3ef9d1c9e747658c8bfc3bea"
)
EXPECTED_ATTEMPT_2_SOURCE_COMMIT = (
    "e14a7f2160e6792a07a77619be7281c55c8c0704"
)
EXPECTED_ATTEMPT_2_SOURCE_SUBJECT = "Train v5 with supervised scale pairs"

BOUND_FILE_HASHES = {
    AMENDMENT_PATH: EXPECTED_AMENDMENT_SHA256,
    RECOVERY_PROTOCOL_PATH: EXPECTED_RECOVERY_PROTOCOL_SHA256,
    SEED_REGISTRY_PATH: EXPECTED_SEED_REGISTRY_SHA256,
    ATTEMPT_2_INTENT_PATH: EXPECTED_ATTEMPT_2_INTENT_SHA256,
    IDENTITY_ACCEPTANCE_PATH: EXPECTED_IDENTITY_ACCEPTANCE_SHA256,
    PYTHON_CANONICALIZER_PATH: EXPECTED_PYTHON_CANONICALIZER_SHA256,
    TYPESCRIPT_CANONICALIZER_PATH:
        EXPECTED_TYPESCRIPT_CANONICALIZER_SHA256,
    V4_GENERATOR_RATIONALE_SOURCE_PATH:
        EXPECTED_V4_GENERATOR_RATIONALE_SOURCE_SHA256,
    V4_DESIGN_RATIONALE_PATH: EXPECTED_V4_DESIGN_RATIONALE_SHA256,
    V4_CONFIRMATION_RATIONALE_PATH:
        EXPECTED_V4_CONFIRMATION_RATIONALE_SHA256,
    V4_STAGE_ONE_POLICY_PATH: EXPECTED_V4_STAGE_ONE_POLICY_SHA256,
}
ALLOWED_PATHS = frozenset(BOUND_FILE_HASHES)

ADAPTIVE_NOVELTY_SEEDS = [20263005, 20263006, 20263007, 20263008]
SEALED_NOVELTY_SEEDS = [20263002, 20263003]
UNUSED_ADAPTIVE_SCALE_SEEDS = [
    20262905,
    20262906,
    20262907,
    20262908,
]
NOVELTY_FAMILIES = ["no_signal", "noise", "chirp"]
ROUTES = ["historical", "current"]
OBSERVATION_LENGTHS = [4096, 8192, 16384]
PHYSICAL_SCALE_FACTORS = [1.0, 1.25, 1.5, 2.0]

LITERAL_PROFILES = [
    "bluetooth-classic-connected",
    "bluetooth-le-advertising",
    "gsm-16qam-higher-symbol-rate-burst",
    "gsm-32qam-higher-symbol-rate-burst",
    "gsm-8psk-normal-burst",
    "gsm-900-loaded-bcch",
    "gsm-aqpsk-normal-burst",
    "gsm-normal-burst",
    "gsm-qpsk-higher-symbol-rate-burst",
    "lte-band3-fdd-20m",
    "lte-band38-tdd-10m",
    "lte-etm1.1",
    "lte-etm3.1",
    "lte-etm3.1a",
    "lte-etm3.1b",
    "lte-nbiot-guard-isolated-component",
    "lte-nbiot-inband-isolated-component",
    "lte-ntm",
    "nr-fr1-tm1.1",
    "nr-fr1-tm3.1",
    "nr-fr1-tm3.1a",
    "nr-fr1-tm3.1b",
    "nr-n3-fdd-20m",
    "nr-n78-tdd-100m",
    "nr-nbiot-inband-isolated-component",
    "wifi-hr-dsss-11m",
    "wifi-ofdm-20m",
    "wifi6-he-er-su",
    "wifi6-he-mu",
    "wifi6-he-su",
    "wifi6-he-tb",
]

NOVELTY_GATE_NAMES = [
    "no_signal_auroc_worst_cell",
    "no_signal_unknown_recall_worst_cell",
    "noise_auroc_worst_cell",
    "noise_unknown_recall_worst_cell",
    "chirp_auroc_worst_cell",
    "chirp_unknown_recall_worst_cell",
]
NOVELTY_GATE_SPECS = [
    {
        "name": "no_signal_auroc_worst_cell",
        "metric": "auroc",
        "family": "no_signal",
        "comparison": "greater_than_or_equal",
        "threshold": 0.8,
    },
    {
        "name": "no_signal_unknown_recall_worst_cell",
        "metric": "unknown_recall",
        "family": "no_signal",
        "comparison": "greater_than_or_equal",
        "threshold": 0.1,
    },
    {
        "name": "noise_auroc_worst_cell",
        "metric": "auroc",
        "family": "noise",
        "comparison": "greater_than_or_equal",
        "threshold": 0.8,
    },
    {
        "name": "noise_unknown_recall_worst_cell",
        "metric": "unknown_recall",
        "family": "noise",
        "comparison": "greater_than_or_equal",
        "threshold": 0.1,
    },
    {
        "name": "chirp_auroc_worst_cell",
        "metric": "auroc",
        "family": "chirp",
        "comparison": "greater_than_or_equal",
        "threshold": 0.8,
    },
    {
        "name": "chirp_unknown_recall_worst_cell",
        "metric": "unknown_recall",
        "family": "chirp",
        "comparison": "greater_than_or_equal",
        "threshold": 0.1,
    },
]

EXPECTED_TOP_LEVEL_KEYS = {
    "schema",
    "schema_version",
    "status",
    "authored_at",
    "lineage",
    "development_only",
    "release_evidence",
    "append_only",
    "mutates_parent_protocol_registry_or_attempt_2_intent",
    "delegated_authorization",
    "parent_bindings",
    "timing_and_source_boundary",
    "parent_development_rationale_only",
    "adaptive_novelty_inventory",
    "normative_generator_mechanics",
    "route_geometry",
    "known_inventory_and_score_references",
    "fixed_stage_one_inheritance",
    "known_only_open_set_fit_contract",
    "execution_contract",
    "novelty_gate_contract",
    "known_gate_contract",
    "full_gate_contract",
    "unused_adaptive_scale_reserve",
    "sealed_novelty_boundary",
}

# Canonical JSON hashes make every nested statement append-only while the
# explicit semantic checks below prove the important arithmetic and parent
# relationships rather than merely trusting opaque section digests.
EXPECTED_SECTION_SHA256 = {
    "delegated_authorization":
        "6ed50833a8c7aa23e5d1b21564f17057dd3d76d23be8f488ff4d70c2223ab3f9",
    "parent_bindings":
        "b52a8642a0c596e71ea8fb65630152b4d82a17ee36b19b1797c7d002418df532",
    "timing_and_source_boundary":
        "065de950de92fe7ece3406dc122c3949ae3e20c47c154605da8d0bfdc6a7dbd6",
    "parent_development_rationale_only":
        "2f84b670dfdd8731a1e4a222d76792a7a214be3e70f61e9c811a699abdefcb24",
    "adaptive_novelty_inventory":
        "4a9b4596b961b7595267b15bee1c9716383110ec3f738a7e3188e846ea6a7384",
    "normative_generator_mechanics":
        "27a5a16a506546c38ccfe8948cbd8824c2005904bb3f9a8f09c7c99f5390f072",
    "route_geometry":
        "352d52a3d41e1a34b4cde16638a2abd00798715b8fd8e1868d6ffda8537a01da",
    "known_inventory_and_score_references":
        "69ed380a96eea8452decc3d400d9c84ffc13af4f3d56818a9e59fa047fd3af66",
    "fixed_stage_one_inheritance":
        "7a3e78ae7a4c7fde41f03264b215bd2671ac5403657ddbb5ffa6f81ed7b35622",
    "known_only_open_set_fit_contract":
        "61bff36697b32a6f281e718ea0de2061243faa44367f6231efd6cdd68f940cea",
    "execution_contract":
        "a85b641a7db10923a6d5f0827d92d999a3f0948408c5d014a14f1c849e40f141",
    "novelty_gate_contract":
        "0d284ca9216e8ab91c296483cdb1885e8fd38d39737f6ee15e3c27d48284f810",
    "known_gate_contract":
        "98f8130b3bc9c3c6bf28f6e6af73c4012dffc7f1bc9c8af07eee9c80bbb732dc",
    "full_gate_contract":
        "822d839fc7469fdb01cb408c183317c6b0762ffc968e57ef9f91c3678388f849",
    "unused_adaptive_scale_reserve":
        "684bb9d7d515e6635de55c9d382181128a7ac38a3948e4af0af9276a32ed4550",
    "sealed_novelty_boundary":
        "3271c6e5fee9249cf90d75176d3d86c78b8e74a5a3be96097cdac562892b6f03",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved not in {candidate.resolve() for candidate in ALLOWED_PATHS}:
        raise ValueError(f"refusing undeclared evidence path: {resolved}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def _float64_payload(value: Any, name: str) -> bytes:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty float64 vector")
    numbers: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{name} contains a non-numeric value")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{name} contains a non-finite value")
        numbers.append(number)
    return b"".join(struct.pack("<d", number) for number in numbers)


def _derive_stage_one_parameter_hashes(
    policy: Mapping[str, Any],
) -> tuple[dict[str, str], str]:
    stage_one = _mapping(policy.get("stage_one"), "v4 policy stage_one")
    parameters = _mapping(
        stage_one.get("parameters_by_length"),
        "v4 policy stage_one parameters_by_length",
    )
    expected_lengths = [str(length) for length in OBSERVATION_LENGTHS]
    if set(parameters) != set(expected_lengths):
        raise ValueError("v4 policy stage-one parameter lengths changed")
    hashes: dict[str, str] = {}
    for length in expected_lengths:
        row = _mapping(
            parameters.get(length),
            f"v4 policy stage-one N{length} parameters",
        )
        if set(row) != {"mean", "scale", "coefficients", "intercept"}:
            raise ValueError(
                f"v4 policy stage-one N{length} parameter fields changed"
        )
        intercept = row.get("intercept")
        if isinstance(intercept, bool) or not isinstance(
            intercept, (int, float)
        ):
            raise ValueError(
                f"v4 policy stage-one N{length} intercept is non-numeric"
            )
        intercept_number = float(intercept)
        if not math.isfinite(intercept_number):
            raise ValueError(
                f"v4 policy stage-one N{length} intercept is non-finite"
            )
        digest = hashlib.sha256()
        widths: list[int] = []
        for name, field_value in (
            ("mean", row.get("mean")),
            ("scale", row.get("scale")),
            ("coefficients", row.get("coefficients")),
            ("intercept", [intercept_number]),
        ):
            payload = _float64_payload(
                field_value,
                f"v4 policy stage-one N{length} {name}",
            )
            widths.append(len(payload) // 8)
            digest.update(f"{name}:{len(payload)}:".encode("utf-8"))
            digest.update(payload)
        if widths != [7, 7, 7, 1]:
            raise ValueError(
                f"v4 policy stage-one N{length} parameter widths changed"
            )
        hashes[length] = digest.hexdigest()

    recorded_by_length = stage_one.get(
        "coefficient_artifact_by_length_sha256"
    )
    if recorded_by_length != hashes:
        raise ValueError(
            "v4 policy recorded stage-one parameter hashes do not reproduce"
        )
    set_digest = hashlib.sha256()
    for length in expected_lengths:
        set_digest.update(f"N{length}:{hashes[length]}".encode("utf-8"))
    derived_set = set_digest.hexdigest()
    if stage_one.get("coefficient_artifact_set_sha256") != derived_set:
        raise ValueError(
            "v4 policy recorded stage-one parameter-set hash does not reproduce"
        )
    return hashes, derived_set


def _allocation(
    registry: Mapping[str, Any],
    name: str,
) -> Mapping[str, Any]:
    fresh = _mapping(
        registry.get("fresh_allocations"),
        "seed registry fresh_allocations",
    )
    return _mapping(fresh.get(name), f"seed allocation {name}")


def _protocol_known_gate_names(
    protocol: Mapping[str, Any],
) -> list[str]:
    contract = _mapping(
        protocol.get("gate_contract"),
        "recovery protocol gate_contract",
    )
    inherited = contract.get("inherited_gates")
    directional = contract.get("added_directional_confusion_gates")
    if not isinstance(inherited, list) or not isinstance(directional, list):
        raise ValueError("recovery protocol gate inventories are invalid")
    if (
        len(inherited) != 19
        or len(directional) != 2
        or contract.get("inherited_gate_count") != 19
        or contract.get("all_inherited_and_added_gates_required") is not True
        or contract.get("weakening_inherited_gates_forbidden") is not True
    ):
        raise ValueError("recovery protocol known-gate contract changed")
    rows = inherited + directional
    names = [
        str(_mapping(row, "recovery protocol gate").get("name"))
        for row in rows
    ]
    if len(set(names)) != 21:
        raise ValueError("recovery protocol known gate names are not unique")
    return names


def _validate_parent_documents(
    value: Mapping[str, Any],
    protocol: Mapping[str, Any],
    registry: Mapping[str, Any],
    attempt_2: Mapping[str, Any],
    acceptance: Mapping[str, Any],
    stage_one_policy: Mapping[str, Any],
) -> None:
    parent = _mapping(value.get("parent_bindings"), "parent_bindings")
    if (
        parent["attempt_2_source"]["commit"]
        != EXPECTED_ATTEMPT_2_SOURCE_COMMIT
        or parent["attempt_2_source"]["commit_subject"]
        != EXPECTED_ATTEMPT_2_SOURCE_SUBJECT
    ):
        raise ValueError("attempt-2 source commit binding changed")
    if (
        attempt_2.get("schema")
        != "atomos.v5.scale-orbit.post-score-supervised-pair-adaptation-intent"
        or attempt_2.get("adaptation_attempt") != 2
        or attempt_2.get("development_only") is not True
        or attempt_2.get("release_evidence") is not False
        or attempt_2.get("timing_and_validation_boundary", {}).get(
            "sealed_seed_generated_or_read"
        )
        is not False
    ):
        raise ValueError("bound attempt-2 intent contract changed")

    known_gate_names = _protocol_known_gate_names(protocol)
    known_contract = _mapping(
        value.get("known_gate_contract"),
        "known_gate_contract",
    )
    if known_contract.get("exact_gate_names") != known_gate_names:
        raise ValueError("known gate names differ from recovery protocol")

    adaptive = _allocation(registry, "adaptive_novelty_development")
    if (
        adaptive.get("seeds") != ADAPTIVE_NOVELTY_SEEDS
        or adaptive.get("statistical_role") != "adaptive_development_only"
        or adaptive.get("adaptive_reuse_allowed") is not True
        or adaptive.get("independent_evidence_eligible") is not False
        or adaptive.get("generation_allowed_before_candidate_freeze")
        is not True
    ):
        raise ValueError("adaptive novelty registry allocation changed")
    sealed = _allocation(registry, "sealed_novelty_validation")
    if (
        sealed.get("seeds") != SEALED_NOVELTY_SEEDS
        or sealed.get("state_at_declaration") != "sealed_unspent"
        or sealed.get("statistical_role") != "one_shot_validation"
        or sealed.get("adaptive_reuse_allowed") is not False
        or sealed.get("independent_evidence_eligible") is not True
        or sealed.get("generation_allowed_before_candidate_freeze")
        is not False
    ):
        raise ValueError("sealed novelty registry allocation changed")
    adaptive_scale = _allocation(registry, "adaptive_scale_development")
    if adaptive_scale.get("seeds") != [20262904, *UNUSED_ADAPTIVE_SCALE_SEEDS]:
        raise ValueError("adaptive scale reserve allocation changed")
    rules = _mapping(
        registry.get("allocation_rules"),
        "seed registry allocation_rules",
    )
    if (
        rules.get("sealed_seed_generation_before_candidate_freeze_forbidden")
        is not True
        or rules.get(
            "sealed_seed_read_or_score_before_claim_ledger_forbidden"
        )
        is not True
    ):
        raise ValueError("sealed registry firewall changed")

    accepted = _mapping(
        acceptance.get("accepted_replacement_corpus"),
        "identity acceptance accepted_replacement_corpus",
    )
    known = _mapping(
        value.get("known_inventory_and_score_references"),
        "known_inventory_and_score_references",
    )
    corpus = _mapping(known.get("adaptive_known_corpus"), "adaptive_known_corpus")
    acceptance_fields = {
        "eval_seed": "eval_seed",
        "directory": "directory",
        "statistical_role": "statistical_role",
        "manifest": "manifest",
        "manifest_sha256": "manifest_sha256",
        "raw": "raw",
        "raw_sha256": "raw_sha256",
        "row_count": "stored_row_count",
        "profile_count": "profile_count",
        "realizations_per_profile": "identities_per_profile",
        "physical_scale_factors": "physical_scale_factors",
    }
    for accepted_name, intent_name in acceptance_fields.items():
        if accepted.get(accepted_name) != corpus.get(intent_name):
            raise ValueError(
                f"known corpus {intent_name} differs from identity acceptance"
            )
    if (
        acceptance.get("development_only") is not True
        or acceptance.get("release_evidence") is not False
        or acceptance.get("observation_boundary", {}).get(
            "sealed_seed_generated_or_read"
        )
        is not False
    ):
        raise ValueError("identity acceptance evidence role changed")

    stage_one = _mapping(
        value.get("fixed_stage_one_inheritance"),
        "fixed_stage_one_inheritance",
    )
    spent = registry.get("spent_or_quarantined")
    if not isinstance(spent, list):
        raise ValueError("seed registry spent inventory is invalid")
    stage_one_rows = [
        row
        for row in spent
        if isinstance(row, Mapping)
        and row.get("seed") == 20264001
        and row.get("state") == "spent_stage_one_fit"
    ]
    if (
        len(stage_one_rows) != 1
        or stage_one_rows[0].get("evidence_sha256")
        != stage_one.get("source_v4_policy_sha256")
    ):
        raise ValueError(
            "fixed_stage_one_inheritance policy differs from seed registry"
        )
    derived_by_length, derived_set = _derive_stage_one_parameter_hashes(
        stage_one_policy
    )
    if (
        stage_one.get("parameter_sha256_by_length") != derived_by_length
        or stage_one.get("coefficient_artifact_set_sha256") != derived_set
        or stage_one.get(
            "validator_must_derive_and_match_every_parameter_and_set_"
            "sha256_from_bound_policy_bytes"
        )
        is not True
    ):
        raise ValueError(
            "fixed_stage_one_inheritance hashes do not reproduce from policy"
        )


def _validate_inventory_arithmetic(value: Mapping[str, Any]) -> None:
    inventory = _mapping(
        value.get("adaptive_novelty_inventory"),
        "adaptive_novelty_inventory",
    )
    if (
        inventory.get("seeds") != ADAPTIVE_NOVELTY_SEEDS
        or inventory.get("families") != NOVELTY_FAMILIES
        or inventory.get("routes") != ROUTES
        or inventory.get("observation_lengths") != OBSERVATION_LENGTHS
        or inventory.get("rows_per_family_per_seed") != 300
        or inventory.get("generated_once_at_maximum_length") != 16384
        or inventory.get("dtype") != "complex64"
        or inventory.get("all_four_seeds_consumed_and_reported_jointly")
        is not True
        or inventory.get("seed_cherry_pick_or_omission_allowed") is not False
    ):
        raise ValueError("adaptive_novelty_inventory changed")
    digest_manifest = _mapping(
        inventory.get("first_generation_digest_manifest"),
        "adaptive_novelty_inventory.first_generation_digest_manifest",
    )
    if (
        digest_manifest.get("required_before_scoring") is not True
        or digest_manifest.get("immutable_after_first_generation") is not True
        or digest_manifest.get(
            "binds_pre_generation_source_freeze_sha256"
        )
        is not True
        or digest_manifest.get(
            "binds_each_family_seed_longest_complex64_sha256"
        )
        is not True
        or digest_manifest.get(
            "binds_each_family_seed_length_prefix_complex64_sha256"
        )
        is not True
        or digest_manifest.get(
            "deterministic_regeneration_allowed_only_when_every_digest_matches"
        )
        is not True
    ):
        raise ValueError(
            "adaptive_novelty_inventory digest-manifest firewall changed"
        )
    expected_cells = (
        len(ADAPTIVE_NOVELTY_SEEDS)
        * len(NOVELTY_FAMILIES)
        * len(ROUTES)
        * len(OBSERVATION_LENGTHS)
    )
    expected_scores = expected_cells * 300
    expected_longest = (
        len(ADAPTIVE_NOVELTY_SEEDS) * len(NOVELTY_FAMILIES) * 300
    )
    if (
        inventory.get("exact_cell_count") != expected_cells
        or inventory.get("exact_route_score_count") != expected_scores
        or inventory.get("longest_rows_total") != expected_longest
        or inventory.get("rows_per_cell") != 300
    ):
        raise ValueError("adaptive_novelty_inventory arithmetic changed")

    known = _mapping(
        value.get("known_inventory_and_score_references"),
        "known_inventory_and_score_references",
    )
    corpus = _mapping(known.get("adaptive_known_corpus"), "adaptive_known_corpus")
    if (
        corpus.get("literal_profiles") != LITERAL_PROFILES
        or corpus.get("profile_count") != len(LITERAL_PROFILES)
        or corpus.get("identities_per_profile") != 64
        or corpus.get("physical_scale_factors") != PHYSICAL_SCALE_FACTORS
        or corpus.get("observation_lengths") != OBSERVATION_LENGTHS
    ):
        raise ValueError("known_inventory_and_score_references changed")
    expected_stored = len(LITERAL_PROFILES) * 64 * len(
        PHYSICAL_SCALE_FACTORS
    )
    expected_views = len(OBSERVATION_LENGTHS) * len(PHYSICAL_SCALE_FACTORS)
    expected_scored = len(LITERAL_PROFILES) * 64 * expected_views
    expected_cells = (
        len(LITERAL_PROFILES)
        * len(OBSERVATION_LENGTHS)
        * len(PHYSICAL_SCALE_FACTORS)
    )
    if (
        corpus.get("stored_row_count") != expected_stored
        or corpus.get("views_per_identity") != expected_views
        or corpus.get("exact_scored_observation_count") != expected_scored
        or corpus.get("exact_profile_length_scale_cell_count")
        != expected_cells
        or corpus.get("rows_per_profile_length_scale_cell") != 64
        or corpus.get("exact_coverage_required") is not True
    ):
        raise ValueError(
            "known_inventory_and_score_references arithmetic or coverage changed"
        )


def _validate_gate_logic(value: Mapping[str, Any]) -> None:
    novelty = _mapping(
        value.get("novelty_gate_contract"),
        "novelty_gate_contract",
    )
    aggregation = _mapping(
        novelty.get("aggregation"),
        "novelty_gate_contract.aggregation",
    )
    gates = novelty.get("gates")
    if not isinstance(gates, list):
        raise ValueError("novelty gates must be a list")
    names = [
        str(_mapping(gate, "novelty gate").get("name")) for gate in gates
    ]
    if (
        names != NOVELTY_GATE_NAMES
        or len(set(names)) != 6
        or gates != NOVELTY_GATE_SPECS
        or novelty.get("exact_gate_count") != 6
        or aggregation.get("cell_count_per_family") != 24
        or aggregation.get("method") != "minimum_worst_cell"
        or aggregation.get(
            "averaging_across_seeds_routes_lengths_or_families_allowed"
        )
        is not False
        or aggregation.get("pooled_overall_auroc_may_substitute_for_family_gate")
        is not False
        or aggregation.get("missing_or_nonfinite_cell_fails") is not True
    ):
        raise ValueError(
            "novelty_gate_contract worst-cell aggregation changed"
        )
    for gate in gates:
        row = _mapping(gate, "novelty gate")
        expected_threshold = 0.8 if row.get("metric") == "auroc" else 0.1
        if (
            row.get("family") not in NOVELTY_FAMILIES
            or row.get("metric") not in {"auroc", "unknown_recall"}
            or row.get("comparison") != "greater_than_or_equal"
            or row.get("threshold") != expected_threshold
        ):
            raise ValueError(
                "novelty_gate_contract threshold or comparison changed"
            )

    known = _mapping(value.get("known_gate_contract"), "known_gate_contract")
    full = _mapping(value.get("full_gate_contract"), "full_gate_contract")
    if (
        known.get("exact_known_gate_count") != 21
        or known.get("all_known_gates_required") is not True
        or full.get("exact_known_gate_count") != 21
        or full.get("exact_novelty_gate_count") != 6
        or full.get("exact_total_gate_count") != 27
        or full.get("exact_known_inventory_coverage_required") is not True
        or full.get("exact_novelty_inventory_coverage_required") is not True
        or full.get("unexpected_duplicate_or_missing_gate_name_fails")
        is not True
        or full.get("known_21_only_may_report_full_pass") is not False
        or full.get("novelty_6_only_may_report_full_pass") is not False
        or full.get("all_pass_rule")
        != (
            "true iff the exact 21 known gates and exact 6 novelty gates are "
            "present and every one of all 27 gates passes"
        )
    ):
        raise ValueError("full_gate_contract 27-gate all-pass rule changed")


def _validate_firewalls(value: Mapping[str, Any]) -> None:
    timing = _mapping(
        value.get("timing_and_source_boundary"),
        "timing_and_source_boundary",
    )
    expected_pre_generation_bindings = [
        "final v5 novelty generator source",
        "every transitively executed novelty-generation helper",
        "generator command and configuration",
        "generator runtime versions numeric dtypes and RNG implementation",
        "this intent and its validator",
    ]
    expected_candidate_bindings = [
        "pre-generation source freeze",
        "final v5 novelty generator source",
        "final v5 novelty scorer source",
        "final v5 full-gate evaluator source",
        "final open-set calibrator and threshold-rank fitter source",
        "every transitively executed scoring fitting and gate helper",
        "this intent and its validator",
        "selected classifier open-set policy and stage-one coefficient bytes",
        "exact known-fit inventory manifests and content digests",
        "produced open-set threshold and rank bytes",
        "exact route-specific known-reference score-vector bytes",
        "preprocessing and runtime export",
        "generated adaptive novelty provenance and immutable digest manifest",
        (
            "device batch size runtime versions numeric dtypes executed "
            "command and configuration"
        ),
    ]
    expected_result_manifest = {
        "binds_candidate_execution_manifest_sha256": True,
        "binds_novelty_digest_manifest_sha256": True,
        (
            "binds_exact_21600_per_row_final_score_and_reject_outputs_or_"
            "canonical_content_hashes"
        ): True,
        "binds_all_72_cell_metrics_row_counts_and_keys": True,
        "binds_exact_known_and_novelty_coverage_inventories": True,
        (
            "binds_all_21_known_gate_rows_and_all_6_aggregated_novelty_"
            "gate_rows"
        ): True,
        "binds_exact_27_gate_name_inventory": True,
        "binds_overall_all_pass": True,
    }
    if (
        timing.get("attempt_2_source_commit_bound_here") is not True
        or timing.get("amendment_commit_existed_at_authorship") is not False
        or timing.get(
            "later_committed_lineage_must_prove_attempt_2_source_commit_is_an_ancestor"
        )
        is not True
        or timing.get(
            "adaptive_v5_novelty_generation_started_before_this_amendment"
        )
        is not False
        or timing.get(
            "adaptive_v5_novelty_scoring_started_before_this_amendment"
        )
        is not False
        or timing.get(
            "sealed_novelty_generation_or_read_started_before_this_amendment"
        )
        is not False
        or timing.get("v5_novelty_generator_source_sha256_frozen_here")
        is not False
        or timing.get("v5_novelty_scorer_source_sha256_frozen_here")
        is not False
        or timing.get("v5_full_gate_source_sha256_frozen_here") is not False
        or timing.get(
            "generation_before_pre_generation_source_freeze_forbidden"
        )
        is not True
        or timing.get("scoring_before_candidate_execution_manifest_forbidden")
        is not True
        or timing.get("later_result_manifest_contract")
        != expected_result_manifest
        or timing.get(
            "later_immutable_pre_generation_source_freeze_must_bind_exact_sha256_for"
        )
        != expected_pre_generation_bindings
        or timing.get(
            "later_candidate_execution_manifest_before_each_adaptive_score_"
            "must_bind_exact_sha256_for"
        )
        != expected_candidate_bindings
    ):
        raise ValueError("timing_and_source_boundary firewall changed")

    fit = _mapping(
        value.get("known_only_open_set_fit_contract"),
        "known_only_open_set_fit_contract",
    )
    if (
        fit.get("known_false_unknown_composite_budget") != 0.03
        or fit.get("thresholds_and_ranks_frozen_before_adaptive_novelty_scoring")
        is not True
        or fit.get("adaptive_known_selection_rows_used_for_threshold_or_rank_fit")
        != 0
        or fit.get("adaptive_novelty_rows_used_for_threshold_or_rank_fit") != 0
        or fit.get("sealed_rows_used_for_threshold_or_rank_fit") != 0
        or fit.get(
            "later_candidate_manifest_must_bind_exact_known_fit_inventory_and_content_digests"
        )
        is not True
        or fit.get(
            "later_candidate_manifest_must_bind_fitter_source_and_all_executed_helpers"
        )
        is not True
        or fit.get(
            "later_candidate_manifest_must_bind_produced_threshold_and_rank_bytes"
        )
        is not True
    ):
        raise ValueError("known_only_open_set_fit_contract firewall changed")

    known = _mapping(
        value.get("known_inventory_and_score_references"),
        "known_inventory_and_score_references",
    )
    row_firewall = _mapping(
        known.get("fitting_firewall"),
        "known_inventory_and_score_references.fitting_firewall",
    )
    if (
        row_firewall.get(
            "seed20262904_rows_used_for_gradient_or_optimizer_step"
        )
        != 0
        or row_firewall.get(
            "seed20262904_rows_used_for_weight_center_or_feature_moment_fit"
        )
        != 0
        or row_firewall.get(
            "seed20262904_rows_used_for_persistent_prototype_fit"
        )
        != 0
        or row_firewall.get(
            "seed20262904_rows_used_for_open_set_threshold_or_rank_fit"
        )
        != 0
        or row_firewall.get(
            "adaptive_novelty_rows_used_for_any_fit_threshold_or_rank"
        )
        != 0
        or row_firewall.get(
            "adaptive_novelty_outcomes_may_be_used_for_adaptive_development_candidate_selection"
        )
        is not True
        or row_firewall.get(
            "adaptive_candidate_rescoring_must_reuse_the_same_immutable_generated_bytes"
        )
        is not True
        or row_firewall.get(
            "sealed_rows_used_for_any_fit_or_adaptive_selection"
        )
        != 0
    ):
        raise ValueError(
            "known_inventory_and_score_references row-use firewall changed"
        )

    execution = _mapping(value.get("execution_contract"), "execution_contract")
    if (
        execution.get("device") != "cpu"
        or execution.get("embedding_batch_size") != 256
        or execution.get("automatic_device_selection_allowed") is not False
        or execution.get("different_device_or_batch_size_allowed") is not False
        or execution.get("same_device_and_batch_required_for_later_sealed_novelty")
        is not True
    ):
        raise ValueError("execution_contract device or batch changed")

    mechanics = _mapping(
        value.get("normative_generator_mechanics"),
        "normative_generator_mechanics",
    )
    family_seed = _mapping(
        mechanics.get("family_seed_derivation"),
        "normative_generator_mechanics.family_seed_derivation",
    )
    if (
        mechanics.get("uses_frequency_transform") is not False
        or mechanics.get("rng") != "numpy.random.default_rng"
        or family_seed.get("message_utf8")
        != "atomos-v4-openset:{base_seed}:{family}"
        or family_seed.get("digest") != "SHA-256"
        or mechanics.get("no_signal", {}).get("generator")
        != "exact complex64 zeros"
        or mechanics.get("noise", {}).get("generator")
        != "direct time-domain complex AR(1)"
        or mechanics.get("chirp", {}).get("generator")
        != "direct time-domain linear-FM exponential"
        or mechanics.get("noise", {}).get("output_dtype") != "complex64"
        or mechanics.get("chirp", {}).get("output_dtype") != "complex64"
    ):
        raise ValueError("normative_generator_mechanics changed")

    geometry = _mapping(value.get("route_geometry"), "route_geometry")
    current = _mapping(geometry.get("current"), "route_geometry.current")
    historical = _mapping(
        geometry.get("historical"),
        "route_geometry.historical",
    )
    if (
        current.get("sample_rate_hz") != 16_000_000
        or current.get("native_sample_rate_hz") != 8_000_000
        or current.get("sample_rate_ratio") != 2.0
        or current.get("class_independent") is not True
        or current.get("profile_independent") is not True
        or current.get("canonicalizer_consumed_input_samples") != 4096
        or current.get("canonicalizer_output_samples") != 4096
        or current.get(
            "effective_runtime_input_length_for_every_observation_length"
        )
        != 4096
        or current.get("uses_frequency_transform") is not False
        or historical.get("sample_rate_hz") is not None
        or historical.get("native_sample_rate_hz") is not None
        or historical.get("canonicalizer_applied") is not False
        or historical.get("uses_frequency_transform") is not False
        or geometry.get(
            "geometry_frozen_before_adaptive_novelty_generation_or_scoring"
        )
        is not True
    ):
        raise ValueError("route_geometry contract changed")

    stage_one = _mapping(
        value.get("fixed_stage_one_inheritance"),
        "fixed_stage_one_inheritance",
    )
    if (
        stage_one.get("source_v4_policy_sha256")
        != EXPECTED_V4_STAGE_ONE_POLICY_SHA256
        or stage_one.get("coefficient_artifact_set_sha256")
        != "de2909bad49a4af49433f09616fa74e05eb7fa9fe4cd7a3aabc01139f78af74d"
        or stage_one.get("inherited_component")
        != "stage-one coefficients only"
        or stage_one.get("source_v4_policy_sha256_must_be_enforced_exactly_by_full_gate")
        is not True
        or stage_one.get(
            "validator_must_derive_and_match_every_parameter_and_set_"
            "sha256_from_bound_policy_bytes"
        )
        is not True
        or stage_one.get("v4_classifier_reused") is not False
        or stage_one.get("v4_prototype_bank_reused") is not False
        or stage_one.get("v4_thresholds_or_ranks_reused") is not False
        or stage_one.get("v4_gate_results_or_validation_claims_reused")
        is not False
    ):
        raise ValueError(
            "fixed_stage_one_inheritance coefficient-only contract changed"
        )

    rationale = _mapping(
        value.get("parent_development_rationale_only"),
        "parent_development_rationale_only",
    )
    if (
        rationale.get("not_v5_evidence") is not True
        or rationale.get("not_release_evidence") is not True
        or rationale.get("not_independent_validation_evidence") is not True
        or rationale.get("may_not_supply_v5_gate_values") is not True
    ):
        raise ValueError(
            "parent_development_rationale_only evidence firewall changed"
        )

    reserve = _mapping(
        value.get("unused_adaptive_scale_reserve"),
        "unused_adaptive_scale_reserve",
    )
    if (
        reserve.get("seeds") != UNUSED_ADAPTIVE_SCALE_SEEDS
        or reserve.get("state_under_this_amendment")
        != "allocated_unspent_and_untouched"
        or reserve.get(
            "use_requires_prior_owner_authorized_append_only_amendment"
        )
        is not True
    ):
        raise ValueError("unused_adaptive_scale_reserve changed")

    sealed = _mapping(
        value.get("sealed_novelty_boundary"),
        "sealed_novelty_boundary",
    )
    expected_copied_contract = [
        "families and 300-row inventory",
        "maximum-length generation and causal-prefix rule",
        "generator mechanics and bound final source and runtime hashes",
        "historical and current geometry",
        (
            "route-specific metric and routing mechanics, explicitly "
            "excluding known-reference populations"
        ),
        "AUROC 0.80 and unknown-recall 0.10 thresholds",
        "worst-cell aggregation with no averaging",
        "CPU device and embedding batch size 256",
    ]
    expected_sealed_freeze_bindings = [
        "selected classifier bytes",
        "open-set policy and stage-one coefficient bytes",
        "preprocessing and runtime export bytes",
        (
            "generator scorer full-gate calibrator threshold-rank fitter and "
            "every executed helper source bytes"
        ),
        (
            "separately owner-authorized sealed known-reference population "
            "and exact score-vector bytes"
        ),
        "device batch size runtime versions and numeric dtypes",
        "this adaptive amendment and passing adaptive gate artifact",
    ]
    if (
        sealed.get("seeds") != SEALED_NOVELTY_SEEDS
        or sealed.get("state_under_this_amendment") != "sealed_unspent"
        or sealed.get("this_amendment_authorizes_generation_read_or_scoring")
        is not False
        or sealed.get("candidate_freeze_required_before_generation") is not True
        or sealed.get("joint_claim_ledger_required_before_either_seed_generation_or_read")
        is not True
        or sealed.get("both_seeds_scored_jointly_once") is not True
        or sealed.get("redraw_rescore_or_adaptation_after_sealed_outcome_allowed")
        is not False
        or sealed.get(
            "known_reference_population_for_sealed_stage_requires_"
            "separate_owner_authorized_freeze"
        )
        is not True
        or sealed.get(
            "later_sealed_contract_is_separate_and_must_copy_unchanged"
        )
        != expected_copied_contract
        or sealed.get("later_candidate_freeze_must_bind")
        != expected_sealed_freeze_bindings
    ):
        raise ValueError("sealed_novelty_boundary firewall changed")


def validate_documents(
    value: Mapping[str, Any],
    protocol: Mapping[str, Any],
    registry: Mapping[str, Any],
    attempt_2: Mapping[str, Any],
    acceptance: Mapping[str, Any],
    stage_one_policy: Mapping[str, Any],
) -> dict[str, Any]:
    if set(value) != EXPECTED_TOP_LEVEL_KEYS:
        raise ValueError("adaptive novelty amendment top-level fields changed")
    if (
        value.get("schema")
        != "atomos.v5.scale-orbit.openset-novelty-adaptive-amendment"
        or value.get("schema_version") != 1
        or value.get("status")
        != (
            "append_only_outcome_blind_before_any_v5_adaptive_novelty_"
            "generation_or_scoring"
        )
        or value.get("authored_at") != "2026-07-29T07:12:56-07:00"
        or value.get("lineage") != "v5-scale-orbit"
        or value.get("development_only") is not True
        or value.get("release_evidence") is not False
        or value.get("append_only") is not True
        or value.get("mutates_parent_protocol_registry_or_attempt_2_intent")
        is not False
    ):
        raise ValueError("adaptive novelty amendment header changed")

    _validate_parent_documents(
        value,
        protocol,
        registry,
        attempt_2,
        acceptance,
        stage_one_policy,
    )
    _validate_inventory_arithmetic(value)
    _validate_gate_logic(value)
    _validate_firewalls(value)
    for section, expected in EXPECTED_SECTION_SHA256.items():
        if _canonical_sha256(value.get(section)) != expected:
            raise ValueError(f"{section} contract changed")
    return {
        "schema": (
            "atomos.v5.scale-orbit.openset-novelty-adaptive-amendment-"
            "validation"
        ),
        "valid": True,
        "development_only": True,
        "release_evidence": False,
        "adaptive_novelty_seeds": ADAPTIVE_NOVELTY_SEEDS,
        "adaptive_novelty_exact_cells": 72,
        "adaptive_novelty_exact_route_scores": 21_600,
        "known_exact_scored_observations": 23_808,
        "known_exact_profile_length_scale_cells": 372,
        "known_gate_count": 21,
        "novelty_gate_count": 6,
        "full_gate_count": 27,
        "corpus_files_read": 0,
        "checkpoint_files_read": 0,
        "stage_one_policy_files_read": 1,
        "prototype_or_calibration_rows_used_as_evidence": 0,
        "model_inference_runs": 0,
        "sealed_files_read": 0,
    }


def _validate_commit_binding() -> bool:
    completed = subprocess.run(
        [
            "git",
            "show",
            "-s",
            "--format=%H%n%s",
            EXPECTED_ATTEMPT_2_SOURCE_COMMIT,
        ],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout.splitlines() != [
        EXPECTED_ATTEMPT_2_SOURCE_COMMIT,
        EXPECTED_ATTEMPT_2_SOURCE_SUBJECT,
    ]:
        raise ValueError("attempt-2 source commit object changed or is missing")
    amendment_relative_path = str(AMENDMENT_PATH.relative_to(REPO))
    amendment_commit = subprocess.run(
        [
            "git",
            "log",
            "-1",
            "--format=%H",
            "--",
            amendment_relative_path,
        ],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not amendment_commit:
        return False
    committed_blob = subprocess.run(
        [
            "git",
            "show",
            f"{amendment_commit}:{amendment_relative_path}",
        ],
        cwd=REPO,
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(committed_blob).hexdigest() != EXPECTED_AMENDMENT_SHA256:
        raise ValueError(
            "committed amendment blob differs from the validated amendment"
        )
    ancestry = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            EXPECTED_ATTEMPT_2_SOURCE_COMMIT,
            amendment_commit,
        ],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestry.returncode != 0:
        raise ValueError(
            "attempt-2 source commit is not an ancestor of the amendment commit"
        )
    return True


def validate_repository() -> dict[str, Any]:
    changed = [
        str(path.relative_to(REPO))
        for path, expected in BOUND_FILE_HASHES.items()
        if _sha256_file(path) != expected
    ]
    if changed:
        raise ValueError(
            "adaptive novelty amendment bound bytes changed: "
            + ", ".join(changed)
        )
    ancestry_verified = _validate_commit_binding()
    result = validate_documents(
        _read_json(AMENDMENT_PATH),
        _read_json(RECOVERY_PROTOCOL_PATH),
        _read_json(SEED_REGISTRY_PATH),
        _read_json(ATTEMPT_2_INTENT_PATH),
        _read_json(IDENTITY_ACCEPTANCE_PATH),
        _read_json(V4_STAGE_ONE_POLICY_PATH),
    )
    result["attempt_2_commit_ancestry_verified"] = ancestry_verified
    result["attempt_2_commit_ancestry_deferred_until_amendment_commit"] = (
        not ancestry_verified
    )
    return result


if __name__ == "__main__":
    print(
        json.dumps(
            validate_repository(),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
