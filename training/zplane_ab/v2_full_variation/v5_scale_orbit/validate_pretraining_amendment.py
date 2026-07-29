"""Validate the append-only v5 pre-training amendment without corpus access."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
AMENDMENT_PATH = HERE / "pretraining_amendment.json"
PROTOCOL_PATH = HERE / "recovery_protocol.json"
REGISTRY_PATH = HERE / "seed_registry.json"

EXPECTED_PARENT_COMMIT = "7911bb6045c553a19b524acecd7b2f465759fce8"
EXPECTED_PROTOCOL_SHA256 = (
    "cc79496ba398e4ce2fcb6875d8b810bcbc1ae7f89cb75271fdf2c65e6ad2afff"
)
EXPECTED_REGISTRY_SHA256 = (
    "b832e09e8eba264e21d7845f72c0d52f671aa1f9187d55ddce94fb833347e0cd"
)
EXPECTED_ROLE_ALLOCATION = {
    "training_and_enrollment": {
        "seed": 20264101,
        "seed_registry_allocation": "training_scale",
        "seed_registry_statistical_role": "training_only",
        "realizations_per_profile": 64,
        "realization_rules": [
            {"start_inclusive": 0, "end_exclusive": 40, "role": "train"},
            {
                "start_inclusive": 40,
                "end_exclusive": 64,
                "role": "enrollment",
            },
        ],
        "allowed_roles": ["train", "enrollment"],
        "forbidden_roles": ["selection", "validation"],
    },
    "adaptive_selection": {
        "seed": 20262904,
        "seed_registry_allocation": "adaptive_scale_development",
        "seed_registry_statistical_role": "adaptive_development_only",
        "realizations_per_profile": 64,
        "realization_rules": [
            {
                "start_inclusive": 0,
                "end_exclusive": 64,
                "role": "selection",
            }
        ],
        "allowed_roles": ["selection"],
        "forbidden_roles": ["train", "enrollment", "validation"],
        "rows_used_for_weight_fit": 0,
        "rows_used_for_prototype_or_enrollment_fit": 0,
        "rows_used_for_threshold_or_rank_fit": 0,
        "candidate_selection_allowed": True,
    },
    "seed20264101_selection_rows": 0,
    "seed20262904_training_or_enrollment_rows": 0,
    "seed_reassignment": False,
}
EXPECTED_UNCONDITIONAL_KEYS = [
    "pair_id",
    "content_sha256",
    "runtime_prefix_sha256",
    "receiver_realization_channel_seed",
    "receiver_realization_seed_when_non_null",
]
EXPECTED_CYCLIC_KEYS = [
    "base_transmitter_identity",
    "profile_and_phase_native_sample",
    *EXPECTED_UNCONDITIONAL_KEYS,
]
EXPECTED_INAPPLICABLE_ONE_SHOT_KEYS = [
    "base_transmitter_identity",
    "profile_and_payload_seed",
    "profile_and_phase_native_sample",
]
EXPECTED_COLLISION_COUNTS = {
    "pair_id": 0,
    "content_sha256": 0,
    "runtime_prefix_sha256": 0,
    "receiver_realization_channel_seed": 0,
    "receiver_realization_seed_when_non_null": 0,
    "cyclic_base_transmitter_identity": 0,
    "cyclic_profile_and_phase_native_sample": 0,
    "profile_and_payload_seed_when_present": 0,
}
EXPECTED_AUDIT_KEYS = [
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
]
EXPECTED_FIREWALL = {
    "adaptive_selection_rows_used_for_weight_fit": 0,
    "adaptive_selection_rows_used_for_prototype_or_enrollment_fit": 0,
    "adaptive_selection_rows_used_for_threshold_or_rank_fit": 0,
    "sealed_validation_rows_used_for_any_fit_or_selection": 0,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if path not in {AMENDMENT_PATH, PROTOCOL_PATH, REGISTRY_PATH}:
        raise ValueError(f"refusing non-contract JSON path: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def _committed_file_sha256(relative_path: str) -> str:
    result = subprocess.run(
        [
            "git",
            "show",
            f"{EXPECTED_PARENT_COMMIT}:{relative_path}",
        ],
        cwd=REPO,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return hashlib.sha256(result.stdout).hexdigest()


def _validate_parent_bindings(
    amendment: Mapping[str, Any],
    protocol: Mapping[str, Any],
    registry: Mapping[str, Any],
    *,
    verify_git_commit: bool,
) -> None:
    bindings = amendment.get("parent_bindings")
    expected = {
        "protocol_relative_path": str(PROTOCOL_PATH.relative_to(REPO)),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "seed_registry_relative_path": str(REGISTRY_PATH.relative_to(REPO)),
        "seed_registry_sha256": EXPECTED_REGISTRY_SHA256,
        "parent_commit": EXPECTED_PARENT_COMMIT,
        "parent_commit_subject": "preregister v5 scale-orbit recovery lineage",
    }
    if bindings != expected:
        raise ValueError("parent protocol/registry bindings changed")
    if _sha256(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("working parent protocol bytes changed")
    if _sha256(REGISTRY_PATH) != EXPECTED_REGISTRY_SHA256:
        raise ValueError("working seed registry bytes changed")
    if protocol.get("seed_registry_sha256") != EXPECTED_REGISTRY_SHA256:
        raise ValueError("parent protocol no longer pins the exact registry")
    if protocol.get("lineage", {}).get("id") != "v5-scale-orbit":
        raise ValueError("parent lineage changed")
    if registry.get("lineage") != "v5-scale-orbit":
        raise ValueError("seed registry lineage changed")
    if verify_git_commit:
        for relative, expected_sha in (
            (expected["protocol_relative_path"], EXPECTED_PROTOCOL_SHA256),
            (expected["seed_registry_relative_path"], EXPECTED_REGISTRY_SHA256),
        ):
            if _committed_file_sha256(str(relative)) != expected_sha:
                raise ValueError(
                    f"parent commit does not contain pinned {relative}"
                )


def _allocation(
    registry: Mapping[str, Any],
    name: str,
) -> Mapping[str, Any]:
    allocations = registry.get("fresh_allocations")
    if not isinstance(allocations, Mapping):
        raise ValueError("registry fresh allocations are missing")
    value = allocations.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"registry allocation {name!r} is missing")
    return value


def _validate_timing(amendment: Mapping[str, Any]) -> None:
    expected = {
        "seed20264101_generation_started_at": "2026-07-29T04:42:00-07:00",
        "generation_start_observation": (
            "local process-table metadata; no manifest, raw sample, hash, "
            "metric, or generated row was opened"
        ),
        "seed20264101_generation_began_before_amendment": True,
        "seed20264101_generation_was_in_progress_when_amendment_was_authored":
            True,
        "seed20264101_manifest_read_before_amendment": False,
        "seed20264101_raw_samples_read_before_amendment": False,
        "seed20264101_generated_content_hashes_observed_before_amendment": False,
        "seed20264101_generated_metrics_observed_before_amendment": False,
        "v5_model_training_started_before_amendment": False,
        "v5_model_inference_started_before_amendment": False,
        "reason_for_amendment": (
            "static code-to-protocol audit found that one corpus was being "
            "assigned both training and adaptive-selection roles and that "
            "fixed-origin one-shot rows cannot honestly satisfy a universal "
            "phase-identity claim"
        ),
        "amendment_is_response_to_generated_outcomes": False,
    }
    if amendment.get("authored_at") != "2026-07-29T04:50:29-07:00":
        raise ValueError("amendment authorship time changed")
    if amendment.get("timing_and_outcome_blindness") != expected:
        raise ValueError("outcome-blind timing declaration changed")


def _validate_role_allocation(
    amendment: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> None:
    if amendment.get("role_allocation") != EXPECTED_ROLE_ALLOCATION:
        raise ValueError("exact seed/role allocation changed")
    training = _allocation(registry, "training_scale")
    adaptive = _allocation(registry, "adaptive_scale_development")
    if (
        20264101 not in training.get("seeds", [])
        or training.get("statistical_role") != "training_only"
        or 20262904 not in adaptive.get("seeds", [])
        or adaptive.get("statistical_role") != "adaptive_development_only"
    ):
        raise ValueError("amended roles disagree with immutable registry")


def _validate_composite_loader(amendment: Mapping[str, Any]) -> None:
    contract = amendment.get("composite_loader_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("composite loader contract is missing")
    exact_required = {
        "required_input_seeds": [20264101, 20262904],
        "required_seed_to_roles": {
            "20262904": ["selection"],
            "20264101": ["train", "enrollment"],
        },
        "each_manifest_and_raw_file_remains_separately_hash_bound": True,
        "row_reference_must_include_source_seed": True,
        "row_identity_format": "scale-orbit:<eval_seed>:<pair_id>",
        "prefix_dispatch_uses_row_source_seed": True,
        "raw_source_index_is_never_applied_to_the_other_seed_corpus": True,
        "all_four_scale_views_of_a_pair_share_one_role_and_one_identity": True,
        "exact_realization_index_set_required_per_profile_per_seed":
            list(range(64)),
        "exact_public_class_order": [
            "am",
            "bluetooth",
            "cw",
            "dsss",
            "fm",
            "gsm",
            "ofdm",
        ],
        "fixed_profile_public_class_map_must_equal_parent_current_source_map":
            True,
        "training_must_not_start_until_composite_loader_audit_passes": True,
    }
    if contract != exact_required:
        raise ValueError("composite loader fail-closed requirements changed")


def _validate_identity_contract(amendment: Mapping[str, Any]) -> None:
    identity = amendment.get("identity_applicability_and_separation")
    if not isinstance(identity, Mapping):
        raise ValueError("identity applicability contract is missing")
    if identity.get("unconditional_zero_collision_keys") != (
        EXPECTED_UNCONDITIONAL_KEYS
    ):
        raise ValueError("unconditional identity separation weakened")
    cyclic = identity.get("cyclic_replay")
    if not isinstance(cyclic, Mapping) or cyclic != {
        "base_transmitter_identity": "profile_and_phase_native_sample",
        "required_zero_collision_keys": EXPECTED_CYCLIC_KEYS,
        "phase_identity_separation_weakened": False,
    }:
        raise ValueError("cyclic phase/transmitter separation changed")
    payload = identity.get("payload_seed")
    if not isinstance(payload, Mapping) or payload != {
        "required_when_emitted_by_generator": True,
        "key": "profile_and_payload_seed",
        "cross_partition_collision_count_when_present": 0,
        "payload_identity_separation_weakened": False,
    }:
        raise ValueError("payload-seed separation changed")
    one_shot = identity.get("one_shot_replay")
    if not isinstance(one_shot, Mapping):
        raise ValueError("one-shot applicability disclosure is missing")
    expected_one_shot = {
        "phase_native_sample_is_structurally_fixed_to_zero": True,
        "current_generator_emits_payload_seed": False,
        "current_generator_emits_independent_transmitter_waveforms_per_realization":
            False,
        "structurally_inapplicable_identity_keys":
            EXPECTED_INAPPLICABLE_ONE_SHOT_KEYS,
        "forbidden_claims": [
            "one-shot transmitter identity is independent across partitions",
            "one-shot phase identity is held out across partitions",
            "one-shot payload identity is held out across partitions",
        ],
        "permitted_generalization_claim":
            "fixed-waveform receiver-realization generalization only",
        "still_required_zero_collision_keys": EXPECTED_UNCONDITIONAL_KEYS,
    }
    if one_shot != expected_one_shot:
        raise ValueError("one-shot applicability disclosure changed")
    if (
        identity.get("required_cross_partition_collision_counts")
        != EXPECTED_COLLISION_COUNTS
        or identity.get(
            "identity_claim_must_report_applicable_and_inapplicable_counts_separately"
        )
        is not True
        or identity.get(
            "missing_identity_key_may_not_be_reported_as_zero_collisions"
        )
        is not True
        or identity.get(
            "paired_scale_views_are_deduplicated_before_identity_collision_counting"
        )
        is not True
    ):
        raise ValueError("identity collision reporting contract changed")


def _validate_required_audit(amendment: Mapping[str, Any]) -> None:
    audit = amendment.get("required_composite_audit")
    if not isinstance(audit, Mapping):
        raise ValueError("required composite audit is missing")
    if (
        audit.get("exact_top_level_keys") != EXPECTED_AUDIT_KEYS
        or audit.get("fitting_firewall_exact") != EXPECTED_FIREWALL
        or audit.get("development_only") is not True
        or audit.get("release_evidence") is not False
        or audit.get("consumed_test_rows_exposed") != 0
    ):
        raise ValueError("required loader audit or fitting firewall changed")
    expected_per_seed = [
        "eval_seed",
        "manifest_relative_path",
        "manifest_sha256",
        "raw_relative_path",
        "raw_sha256",
        "registry_allocation",
        "statistical_role",
        "allowed_roles",
    ]
    expected_identity = [
        "applicable_collision_counts",
        "inapplicable_keys_by_replay",
        "prefix_lengths_hashed",
        "paired_views_deduplicated",
        "all_applicable_collision_counts_zero",
    ]
    if (
        audit.get("corpus_binding_keys_per_seed") != expected_per_seed
        or audit.get("identity_separation_keys") != expected_identity
    ):
        raise ValueError("required audit keys changed")


def validate_amendment(
    amendment: Mapping[str, Any],
    protocol: Mapping[str, Any],
    registry: Mapping[str, Any],
    *,
    verify_git_commit: bool = True,
) -> dict[str, Any]:
    if (
        amendment.get("schema")
        != "atomos.v5.scale-orbit.pretraining-amendment"
        or amendment.get("schema_version") != 1
        or amendment.get("status")
        != "append_only_outcome_blind_before_any_v5_corpus_read_training_or_inference"
        or amendment.get("development_only") is not True
        or amendment.get("release_evidence") is not False
        or amendment.get("append_only") is not True
        or amendment.get("mutates_parent_protocol_or_registry") is not False
    ):
        raise ValueError("amendment top-level declaration changed")
    _validate_parent_bindings(
        amendment,
        protocol,
        registry,
        verify_git_commit=verify_git_commit,
    )
    _validate_timing(amendment)
    _validate_role_allocation(amendment, registry)
    _validate_composite_loader(amendment)
    _validate_identity_contract(amendment)
    _validate_required_audit(amendment)
    stops = amendment.get("stop_conditions")
    if (
        not isinstance(stops, Mapping)
        or set(stops) != {
            "reject_if_either_seed_or_role_differs",
            "reject_if_any_profile_lacks_exact_realization_index_set",
            "reject_if_pair_views_cross_roles_or_corpora",
            "reject_if_any_applicable_identity_collision_count_is_nonzero",
            "reject_if_one_shot_inapplicable_key_is_claimed_as_independently_held_out",
            "reject_if_adaptive_selection_reaches_any_fit_population",
            "reject_if_composite_prefix_dispatch_is_ambiguous",
            "stop_before_training_on_first_failure",
        }
        or any(value is not True for value in stops.values())
    ):
        raise ValueError("pre-training stop conditions changed")
    return {
        "schema": "atomos.v5.scale-orbit.pretraining-amendment-validation",
        "schema_version": 1,
        "valid": True,
        "parent_commit": EXPECTED_PARENT_COMMIT,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "seed_registry_sha256": EXPECTED_REGISTRY_SHA256,
        "training_seed": 20264101,
        "adaptive_selection_seed": 20262904,
        "applicable_zero_collision_key_count":
            len(EXPECTED_COLLISION_COUNTS),
        "corpus_files_read": 0,
    }


def validate_repository_amendment(
    *,
    verify_git_commit: bool = True,
) -> dict[str, Any]:
    return validate_amendment(
        _read_json(AMENDMENT_PATH),
        _read_json(PROTOCOL_PATH),
        _read_json(REGISTRY_PATH),
        verify_git_commit=verify_git_commit,
    )


if __name__ == "__main__":
    print(
        json.dumps(
            validate_repository_amendment(),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
