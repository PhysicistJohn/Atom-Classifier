"""Validate the append-only v5 adaptive-corpus identity-firewall amendment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
AMENDMENT_PATH = HERE / "identity_firewall_amendment.json"
PARENT_AMENDMENT_PATH = HERE / "pretraining_amendment.json"
REJECTION_PATH = HERE / "seed20262904_identity_rejection.json"
PROTOCOL_PATH = HERE / "recovery_protocol.json"
REGISTRY_PATH = HERE / "seed_registry.json"
GENERATOR_SOURCE_PATH = (
    REPO / "tools" / "generate-current-signallab-scale-eval.ts"
)
GENERATOR_BUNDLE_PATH = (
    REPO
    / ".artifacts"
    / "current-scale-eval-generator"
    / "generate-current-signallab-scale-eval.js"
)

EXPECTED_AMENDMENT_SHA256 = (
    "89d714910dc5dfdbfb05282abf752382cf382e88108d0f2b589f62289d452ba4"
)
EXPECTED_PARENT_AMENDMENT_SHA256 = (
    "3e4098d0475477bf1ba26e79992ca17848c16dd77deb44b7562c4452544003fd"
)
EXPECTED_REJECTION_SHA256 = (
    "9f67115eaa07b97f35bcc3cc2f8c3ebd96a0bc819ab47500b98953aa57d31bc5"
)
EXPECTED_PROTOCOL_SHA256 = (
    "cc79496ba398e4ce2fcb6875d8b810bcbc1ae7f89cb75271fdf2c65e6ad2afff"
)
EXPECTED_REGISTRY_SHA256 = (
    "b832e09e8eba264e21d7845f72c0d52f671aa1f9187d55ddce94fb833347e0cd"
)
EXPECTED_GENERATOR_SOURCE_SHA256 = (
    "53f076da486c73bd3088d0ad2359d399736c0eaa56f5323d350d1ebca57bcd34"
)
EXPECTED_GENERATOR_BUNDLE_SHA256 = (
    "621fa092a9513e6f54270120e9005df15962a1f99602b6df9ac493887910fd44"
)
EXPECTED_GENERATOR_COMMIT = "e2897504692db2bc3f948613ec1043c0dd51d89c"
EXPECTED_OUTPUT_DIRECTORY = (
    "training/artifacts/"
    "signallab-current-scale-dev-v5-seed20262904-r64-identity-firewalled"
)
EXPECTED_EXCLUSION_ENVIRONMENT = {
    "IDENTITY_EXCLUSION_SCALE_CORPUS_DIRS_JSON": (
        "[\"training/artifacts/"
        "signallab-current-scale-train-v5-seed20264101-r64\"]"
    )
}
ZERO_COLLISION_FIELDS = (
    "cross_role_pair_id_collision_count",
    "cross_role_content_sha256_collision_count",
    "cross_role_runtime_prefix_sha256_collision_count",
    "cross_role_receiver_realization_channel_seed_collision_count",
    "cross_role_receiver_realization_seed_when_non_null_collision_count",
    "cross_role_cyclic_profile_and_phase_native_sample_collision_count",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def validate_document(value: Mapping[str, Any]) -> dict[str, Any]:
    if (
        value.get("schema")
        != "atomos.v5.scale-orbit.identity-firewall-amendment"
        or value.get("schema_version") != 1
        or value.get("status")
        != "append_only_before_identity_firewalled_replacement_generation"
        or value.get("append_only") is not True
        or value.get("mutates_parent_amendment_protocol_or_seed_registry")
        is not False
        or value.get("development_only") is not True
        or value.get("release_evidence") is not False
    ):
        raise ValueError("identity-firewall amendment header changed")

    parent = _mapping(value.get("parent_bindings"), "parent_bindings")
    expected_parent = {
        "pretraining_amendment_sha256":
            EXPECTED_PARENT_AMENDMENT_SHA256,
        "recovery_protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "seed_registry_sha256": EXPECTED_REGISTRY_SHA256,
        "identity_rejection_sha256": EXPECTED_REJECTION_SHA256,
    }
    if dict(parent) != expected_parent:
        raise ValueError("identity-firewall parent bindings changed")

    timing = _mapping(
        value.get("timing_and_observation_boundary"),
        "timing_and_observation_boundary",
    )
    if (
        timing.get("first_seed20262904_identity_audit_attempted") is not True
        or timing.get("first_seed20262904_model_training_started") is not False
        or timing.get("first_seed20262904_model_inference_started") is not False
        or timing.get("first_seed20262904_candidate_metrics_observed")
        is not False
        or timing.get("replacement_generation_started_before_this_amendment")
        is not False
        or timing.get("class_predictions_or_model_outcomes_used") is not False
    ):
        raise ValueError("identity-firewall observation boundary changed")

    replacement = _mapping(
        value.get("replacement_adaptive_corpus"),
        "replacement_adaptive_corpus",
    )
    if (
        replacement.get("eval_seed") != 20262904
        or replacement.get("seed_registry_allocation")
        != "adaptive_scale_development"
        or replacement.get("seed_registry_adaptive_reuse_allowed") is not True
        or replacement.get("statistical_role")
        != "adaptive_development_only"
        or replacement.get("roles") != ["selection"]
        or replacement.get("realizations_per_profile") != 64
        or replacement.get("output_directory") != EXPECTED_OUTPUT_DIRECTORY
        or replacement.get("must_not_overwrite_rejected_directory") is not True
        or replacement.get(
            "replacement_manifest_and_raw_hashes_must_differ_from_rejected"
        ) is not True
    ):
        raise ValueError("replacement adaptive-corpus contract changed")

    generator = _mapping(value.get("generator_binding"), "generator_binding")
    if (
        generator.get("source_commit") != EXPECTED_GENERATOR_COMMIT
        or generator.get("source_sha256")
        != EXPECTED_GENERATOR_SOURCE_SHA256
        or generator.get("bundle_sha256")
        != EXPECTED_GENERATOR_BUNDLE_SHA256
        or generator.get("required_identity_exclusion_environment")
        != EXPECTED_EXCLUSION_ENVIRONMENT
    ):
        raise ValueError("identity-firewall generator binding changed")

    firewall = _mapping(value.get("identity_firewall"), "identity_firewall")
    if (
        any(firewall.get(field) != 0 for field in ZERO_COLLISION_FIELDS)
        or firewall.get(
            "generator_must_hash_verify_every_reference_manifest_and_raw_file_before_generation"
        )
        is not True
        or firewall.get("composite_loader_must_pass_before_training")
        is not True
        or firewall.get(
            "one_shot_transmitter_phase_payload_independence_claimed"
        )
        is not False
    ):
        raise ValueError("identity-firewall collision contract weakened")

    unchanged = _mapping(
        value.get("unchanged_contracts"),
        "unchanged_contracts",
    )
    if (
        unchanged.get("gate_threshold_or_comparison_changed") is not False
        or unchanged.get("class_or_profile_removed") is not False
        or unchanged.get("realization_or_scale_count_reduced") is not False
        or unchanged.get("selection_rows_used_for_any_fit") != 0
        or unchanged.get("sealed_seed_generated_or_read") is not False
        or unchanged.get("stop_on_identity_collision") is not True
    ):
        raise ValueError("identity-firewall unchanged contracts drifted")
    return {
        "schema": (
            "atomos.v5.scale-orbit.identity-firewall-amendment-validation"
        ),
        "valid": True,
        "adaptive_seed": 20262904,
        "replacement_output_directory": EXPECTED_OUTPUT_DIRECTORY,
        "zero_collision_field_count": len(ZERO_COLLISION_FIELDS),
        "corpus_files_read": 0,
    }


def validate_repository() -> dict[str, Any]:
    observed = {
        AMENDMENT_PATH: EXPECTED_AMENDMENT_SHA256,
        PARENT_AMENDMENT_PATH: EXPECTED_PARENT_AMENDMENT_SHA256,
        REJECTION_PATH: EXPECTED_REJECTION_SHA256,
        PROTOCOL_PATH: EXPECTED_PROTOCOL_SHA256,
        REGISTRY_PATH: EXPECTED_REGISTRY_SHA256,
        GENERATOR_SOURCE_PATH: EXPECTED_GENERATOR_SOURCE_SHA256,
        GENERATOR_BUNDLE_PATH: EXPECTED_GENERATOR_BUNDLE_SHA256,
    }
    changed = [
        str(path)
        for path, expected in observed.items()
        if sha256_file(path) != expected
    ]
    if changed:
        raise ValueError(
            "identity-firewall bound bytes changed: " + ", ".join(changed)
        )
    with AMENDMENT_PATH.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return validate_document(_mapping(value, "identity-firewall amendment"))


if __name__ == "__main__":
    print(
        json.dumps(
            validate_repository(),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
