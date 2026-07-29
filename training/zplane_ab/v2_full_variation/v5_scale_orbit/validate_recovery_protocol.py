"""Fail-closed validation for the v5 scale-orbit recovery declaration.

This validator reads only the v5 protocol/registry and explicitly bound JSON
evidence.  Quarantined or sealed corpus directories are never opened.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PROTOCOL_PATH = HERE / "recovery_protocol.json"
REGISTRY_PATH = HERE / "seed_registry.json"

PROTOCOL_SCHEMA = "atomos.v5.scale-orbit.recovery-protocol"
REGISTRY_SCHEMA = "atomos.v5.scale-orbit.seed-registry"
REGISTRY_RELATIVE_PATH = (
    "training/zplane_ab/v2_full_variation/v5_scale_orbit/"
    "seed_registry.json"
)
EXPECTED_REGISTRY_SHA256 = (
    "b832e09e8eba264e21d7845f72c0d52f671aa1f9187d55ddce94fb833347e0cd"
)

EXPECTED_ANCESTOR = {
    "report_relative_path":
        ".artifacts/v4-current-source-scale-dev-seed20262903-score/"
        "scale-development-evaluation.json",
    "report_sha256":
        "ef58940c1504ef4e914c21991887dd49f4ead98aeea313ceeaa43f2ad6dbaafa",
    "manifest_relative_path":
        ".artifacts/v4-current-source-scale-dev-seed20262903-score/"
        "scale-development-manifest.json",
    "manifest_sha256":
        "2330868b6f17a06ab28383712521822b922afab74185492a4e62bda346551826",
    "intent_relative_path":
        "training/artifacts/"
        "v4-current-source-scale-dev-seed20262903.score.intent.json",
    "intent_sha256":
        "b79718ac8be1327f7096d491a8aa72a7557683765c925f9e262e39e501453832",
    "ledger_relative_path":
        "training/artifacts/"
        "v4-current-source-scale-dev-seed20262903.score.ledger.json",
    "ledger_sha256":
        "e549fe98fda22f04e90c46f260d8a082f508fc49ed36e96282fbfce0107c35e4",
    "all_pass": False,
    "development_only": True,
    "release_evidence": False,
    "classifier_sha256":
        "c0dc837ddacd0bd3055701719a567e8011392180ad2421c007cc2a20e927d86a",
    "policy_sha256":
        "57f3262bb6c10785a766ef6834b88fc2c8daaeae3ef9d1c9e747658c8bfc3bea",
}

EXPECTED_INHERITED_GATES: dict[str, tuple[str, bool | float]] = {
    "causal_prefix_outcome_agreement_across_observation_lengths":
        ("greater_than_or_equal", 1.0),
    "causal_prefix_outcome_agreement_every_source_profile":
        ("greater_than_or_equal", 1.0),
    "classifier_development_current_service_every_legal_cell_closed_accuracy":
        ("greater_than_or_equal", 0.9),
    "classifier_development_current_service_pooled_closed_accuracy":
        ("greater_than_or_equal", 0.95),
    "classifier_development_current_service_pooled_profile_balanced_accuracy":
        ("greater_than_or_equal", 0.9),
    "classifier_development_current_service_wifi_hr_dsss_worst_legal_cell_accuracy":
        ("greater_than_or_equal", 0.95),
    "classifier_development_current_service_worst_cell_present_class_balanced_accuracy":
        ("greater_than_or_equal", 0.85),
    "classifier_development_current_service_worst_profile_legal_cell_closed_accuracy":
        ("greater_than_or_equal", 0.5),
    "classifier_development_current_service_worst_profile_pooled_closed_accuracy":
        ("greater_than_or_equal", 0.7),
    "exact_eligible_pair_and_observation_coverage": ("equals", True),
    "exact_profile_length_scale_key_and_count_coverage": ("equals", True),
    "known_abstention_damage_given_closed_head_correct_pooled":
        ("less_than_or_equal", 0.12),
    "known_abstention_damage_given_closed_head_correct_worst_observation_length":
        ("less_than_or_equal", 0.12),
    "known_abstention_damage_given_closed_head_correct_worst_supported_source_profile_cell":
        ("less_than_or_equal", 0.12),
    "known_abstention_damage_given_closed_head_correct_worst_supported_true_class_cell":
        ("less_than_or_equal", 0.12),
    "length_scale_outcome_agreement_every_source_profile":
        ("greater_than_or_equal", 0.95),
    "length_scale_outcome_agreement_pooled":
        ("greater_than_or_equal", 0.95),
    "physical_scale_outcome_agreement_every_observation_length":
        ("greater_than_or_equal", 0.95),
    "physical_scale_outcome_agreement_pooled":
        ("greater_than_or_equal", 0.95),
}

EXPECTED_DIRECTIONAL_GATES = {
    "wifi_hr_dsss_to_bluetooth_worst_legal_cell_confusion_rate": {
        "truth": "dsss",
        "source_profile": "current:wifi-hr-dsss-11m",
        "predicted_class": "bluetooth",
        "scope": "every_legal_observation_length_scale_cell",
        "comparison": "less_than_or_equal",
        "threshold": 0.02,
    },
    "bluetooth_to_dsss_worst_legal_cell_confusion_rate": {
        "truth": "bluetooth",
        "source_profile": "every_current_bluetooth_profile",
        "predicted_class": "dsss",
        "scope": "every_legal_observation_length_scale_cell",
        "comparison": "less_than_or_equal",
        "threshold": 0.02,
    },
}

EXPECTED_SPENT = {
    20_262_902: {
        "family": "physical_scale",
        "state": "spent_exposed_design_diagnostic_only",
        "raw_corpus_access_in_v5": "forbidden",
        "quarantined_corpus_relative_path":
            "training/artifacts/"
            "signallab-current-scale-eval-v3-seed20262902-r24",
        "evidence_relative_path":
            ".artifacts/"
            "v4-current-source-lineage-share13-openset-design-seed20263001-"
            "scale-v3-20262902/design-evaluation.json",
        "evidence_sha256":
            "4080810f24b3225e47427b28c2d73d74739f850e237ebe5d41a3c2388a784078",
    },
    20_262_903: {
        "family": "physical_scale",
        "state": "spent_failed_one_shot_development_diagnostic_only",
        "raw_corpus_access_in_v5": "forbidden",
        "quarantined_corpus_relative_path":
            "training/artifacts/"
            "signallab-current-scale-dev-v3-seed20262903-r24",
        "evidence_relative_path":
            ".artifacts/v4-current-source-scale-dev-seed20262903-score/"
            "scale-development-evaluation.json",
        "evidence_sha256":
            "ef58940c1504ef4e914c21991887dd49f4ead98aeea313ceeaa43f2ad6dbaafa",
    },
    20_263_001: {
        "family": "novelty_design",
        "state": "spent_adaptive_design",
        "raw_corpus_access_in_v5":
            "not_applicable_generated_in_memory",
        "evidence_relative_path":
            ".artifacts/"
            "v4-openset-schema5-final-independent-design-seed20263001/"
            "design-evaluation.json",
        "evidence_sha256":
            "c0e64ef2bdb7a8b43297af9a2c3d6dcfa775055dfae2e860d8b073d5794493c0",
    },
    20_263_004: {
        "family": "novelty_design",
        "state": "spent_adaptive_confirmation_design",
        "raw_corpus_access_in_v5":
            "not_applicable_generated_in_memory",
        "evidence_relative_path":
            ".artifacts/"
            "v4-current-source-lineage-share13-openset-confirmation-"
            "seed20263004-final/design-evaluation.json",
        "evidence_sha256":
            "fcdc78ee55ee3ecfad2f61ae74c5ecd44f80fddb573cd3ee88f0433ca38192fe",
    },
    20_264_001: {
        "family": "fitting",
        "state": "spent_stage_one_fit",
        "raw_corpus_access_in_v5": "not_applicable_fit_rng",
        "evidence_relative_path":
            ".artifacts/"
            "v4-openset-schema5-final-independent-design-seed20263001/"
            "time-domain-profile-bank-openset-v4.json",
        "evidence_sha256":
            "57f3262bb6c10785a766ef6834b88fc2c8daaeae3ef9d1c9e747658c8bfc3bea",
    },
}

EXPECTED_ALLOCATIONS = {
    "training_scale": {
        "seeds": list(range(20_264_101, 20_264_109)),
        "state_at_declaration": "allocated_unspent",
        "statistical_role": "training_only",
        "adaptive_reuse_allowed": True,
        "independent_evidence_eligible": False,
        "generation_allowed_before_candidate_freeze": True,
    },
    "adaptive_scale_development": {
        "seeds": list(range(20_262_904, 20_262_909)),
        "state_at_declaration": "allocated_unspent",
        "statistical_role": "adaptive_development_only",
        "adaptive_reuse_allowed": True,
        "independent_evidence_eligible": False,
        "generation_allowed_before_candidate_freeze": True,
    },
    "adaptive_novelty_development": {
        "seeds": list(range(20_263_005, 20_263_009)),
        "state_at_declaration": "allocated_unspent",
        "statistical_role": "adaptive_development_only",
        "adaptive_reuse_allowed": True,
        "independent_evidence_eligible": False,
        "generation_allowed_before_candidate_freeze": True,
    },
    "sealed_known_scale_validation": {
        "seeds": [20_262_920, 20_262_921],
        "state_at_declaration": "sealed_unspent",
        "statistical_role": "one_shot_validation",
        "adaptive_reuse_allowed": False,
        "independent_evidence_eligible": True,
        "generation_allowed_before_candidate_freeze": False,
    },
    "sealed_novelty_validation": {
        "seeds": [20_263_002, 20_263_003],
        "state_at_declaration": "sealed_unspent",
        "statistical_role": "one_shot_validation",
        "adaptive_reuse_allowed": False,
        "independent_evidence_eligible": True,
        "generation_allowed_before_candidate_freeze": False,
    },
}

EXPECTED_IDENTITY_KEYS = [
    "base_transmitter_identity",
    "pair_id",
    "content_sha256",
    "profile_and_payload_seed",
    "profile_and_phase_native_sample",
    "receiver_realization_channel_seed",
    "receiver_realization_seed",
]


def _load_json(path: Path, role: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {role} JSON {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must contain an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ValueError(f"cannot hash bound evidence {path}") from exc
    return digest.hexdigest()


def _mapping(value: Any, role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{role} must be an object")
    return value


def _sequence(value: Any, role: str) -> Sequence[Any]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
    ):
        raise ValueError(f"{role} must be an array")
    return value


def _metadata_evidence_path(relative_value: Any, role: str) -> Path:
    if not isinstance(relative_value, str) or not relative_value:
        raise ValueError(f"{role} path is invalid")
    relative = Path(relative_value)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.suffix != ".json"
        or relative.name in {"corpus.json", "scale_eval.json"}
        or any("signallab" in part.lower() for part in relative.parts)
    ):
        raise ValueError(f"{role} may bind only non-corpus JSON evidence")
    lexical = Path(REPO, relative)
    cursor = REPO.resolve()
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"{role} may not traverse a symlink: {cursor}")
    resolved = lexical.resolve()
    repo = REPO.resolve()
    if repo not in resolved.parents or not resolved.is_file():
        raise ValueError(f"{role} is missing or outside the repository")
    return resolved


def _verify_evidence_hash(relative: Any, expected: Any, role: str) -> Path:
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise ValueError(f"{role} SHA-256 is invalid")
    path = _metadata_evidence_path(relative, role)
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(
            f"{role} SHA-256 changed: expected {expected}, got {observed}"
        )
    return path


def _validate_ancestor(
    protocol: Mapping[str, Any],
    *,
    verify_evidence: bool,
) -> list[Path]:
    lineage = _mapping(protocol.get("lineage"), "lineage")
    if (
        lineage.get("id") != "v5-scale-orbit"
        or lineage.get("ancestor_lineage")
        != "v4-current-source-lineage-share13"
        or lineage.get("ancestor_lineage_terminal") is not True
        or lineage.get("continuation_of_v4_one_shot_claim") is not False
        or lineage.get("post_score_adaptation_disclosed") is not True
        or lineage.get("independence_from_seed20262903") is not False
    ):
        raise ValueError("v5 lineage/adaptation declaration changed")
    ancestor = dict(
        _mapping(
            lineage.get("ancestor_failed_evidence"),
            "ancestor failed evidence",
        )
    )
    if ancestor != EXPECTED_ANCESTOR:
        raise ValueError("ancestor failed-evidence binding changed")
    boundary = _mapping(
        lineage.get("adaptation_boundary"), "adaptation boundary"
    )
    required_boundary = {
        "seed20262903_is_adaptive_design_evidence": True,
        "seed20262903_is_independent_validation_evidence": False,
        "seed20262903_corpus_may_be_rescored": False,
        "seed20262903_raw_rows_may_be_used_for_fit_or_calibration": False,
        "cell_specific_changes_informed_by_the_failed_report_must_be_disclosed":
            True,
        "new_validation_claims_require_fresh_sealed_seeds": True,
    }
    if dict(boundary) != required_boundary:
        raise ValueError("ancestor adaptation boundary changed")
    if not verify_evidence:
        return []
    checked = [
        _verify_evidence_hash(
            ancestor[f"{name}_relative_path"],
            ancestor[f"{name}_sha256"],
            f"ancestor {name}",
        )
        for name in ("report", "manifest", "intent", "ledger")
    ]
    report = _load_json(checked[0], "ancestor failed report")
    manifest = _load_json(checked[1], "ancestor failed manifest")
    candidate = _mapping(report.get("candidate"), "ancestor candidate")
    if (
        report.get("development_only") is not True
        or report.get("release_evidence") is not False
        or _mapping(
            report.get("gate_summary"), "ancestor gate summary"
        ).get("all_pass")
        is not False
        or candidate.get("classifier_sha256")
        != ancestor["classifier_sha256"]
        or candidate.get("policy_sha256") != ancestor["policy_sha256"]
        or manifest.get("development_only") is not True
        or manifest.get("release_evidence") is not False
        or manifest.get("all_pass") is not False
        or _mapping(
            manifest.get("files_sha256"), "ancestor manifest files"
        ).get("scale-development-evaluation.json")
        != ancestor["report_sha256"]
    ):
        raise ValueError("ancestor failure semantics changed")
    return checked


def _validate_seed_registry(
    protocol: Mapping[str, Any],
    registry: Mapping[str, Any],
    *,
    verify_evidence: bool,
) -> list[Path]:
    if (
        registry.get("schema") != REGISTRY_SCHEMA
        or registry.get("schema_version") != 1
        or registry.get("status") != "immutable_allocation_declaration"
        or registry.get("lineage") != "v5-scale-orbit"
        or protocol.get("seed_registry_relative_path")
        != REGISTRY_RELATIVE_PATH
        or protocol.get("seed_registry_sha256")
        != EXPECTED_REGISTRY_SHA256
    ):
        raise ValueError("seed-registry identity changed")
    if _sha256(REGISTRY_PATH) != EXPECTED_REGISTRY_SHA256:
        raise ValueError("seed-registry bytes changed")

    spent_rows = _sequence(
        registry.get("spent_or_quarantined"),
        "spent seed inventory",
    )
    spent: dict[int, Mapping[str, Any]] = {}
    for value in spent_rows:
        row = _mapping(value, "spent seed row")
        seed = row.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed in spent:
            raise ValueError("spent seed inventory has an invalid duplicate")
        spent[seed] = row
    if set(spent) != set(EXPECTED_SPENT):
        raise ValueError("spent seed inventory changed")
    evidence_paths: list[Path] = []
    for seed, expected in EXPECTED_SPENT.items():
        row = spent[seed]
        for key, expected_value in expected.items():
            if row.get(key) != expected_value:
                raise ValueError(
                    f"spent seed {seed} field {key} changed"
                )
        if row.get("independent_evidence_eligible") is not False:
            raise ValueError(f"spent seed {seed} became evidence eligible")
        if verify_evidence:
            evidence_paths.append(
                _verify_evidence_hash(
                    row["evidence_relative_path"],
                    row["evidence_sha256"],
                    f"spent seed {seed} evidence",
                )
            )

    allocations = _mapping(
        registry.get("fresh_allocations"), "fresh allocations"
    )
    if set(allocations) != set(EXPECTED_ALLOCATIONS):
        raise ValueError("fresh allocation roles changed")
    fresh_seeds: list[int] = []
    for role, expected in EXPECTED_ALLOCATIONS.items():
        observed = dict(_mapping(allocations.get(role), role))
        if observed != expected:
            raise ValueError(f"fresh allocation {role} changed")
        fresh_seeds.extend(observed["seeds"])
    if (
        len(fresh_seeds) != len(set(fresh_seeds))
        or set(fresh_seeds) & set(spent)
    ):
        raise ValueError("fresh and spent seed allocations overlap")
    rules = _mapping(
        registry.get("allocation_rules"), "allocation rules"
    )
    if not rules or any(value is not True for value in rules.values()):
        raise ValueError("seed allocation rules must all remain true")
    return evidence_paths


def _validate_data_contract(protocol: Mapping[str, Any]) -> None:
    data = _mapping(protocol.get("data_contract"), "data contract")
    if (
        data.get("physical_scale_factors") != [1.0, 1.25, 1.5, 2.0]
        or data.get("causal_observation_lengths")
        != [4096, 8192, 16384]
        or data.get("evaluator_side_array_resampling") is not False
        or data.get("frequency_transform_at_inference") is not False
        or data.get(
            "paired_scale_views_share_one_latent_transmitter_identity"
        )
        is not True
        or data.get("shorter_observations_are_exact_leading_prefixes")
        is not True
        or data.get("partition_unit") != "base_transmitter_identity"
    ):
        raise ValueError("scale-orbit data contract changed")
    separation = _mapping(
        data.get("identity_separation"), "identity separation"
    )
    if (
        separation.get("train_dev_validation_pairwise_disjoint") is not True
        or separation.get("paired_views_may_not_cross_partitions") is not True
        or separation.get("prefix_families_may_not_cross_partitions")
        is not True
        or separation.get("required_disjoint_identity_keys")
        != EXPECTED_IDENTITY_KEYS
        or separation.get(
            "content_collision_count_between_any_partitions"
        )
        != 0
        or separation.get(
            "receiver_identity_collision_count_between_any_partitions"
        )
        != 0
        or separation.get(
            "reference_training_content_collision_count_in_validation"
        )
        != 0
    ):
        raise ValueError("train/dev/validation identity firewall changed")
    fitting = _mapping(
        data.get("fitting_firewall"), "fitting firewall"
    )
    required_zero = (
        "adaptive_development_rows_used_for_weight_fit",
        "adaptive_development_rows_used_for_threshold_or_rank_fit",
        "sealed_validation_rows_used_for_any_fit_or_selection",
    )
    if (
        any(fitting.get(key) != 0 for key in required_zero)
        or fitting.get("threshold_and_rank_fit_population")
        != "training_or_enrollment_only"
        or fitting.get("candidate_selection_population")
        != "training_and_adaptive_development_only"
    ):
        raise ValueError("fitting firewall changed")


def _validate_gate_contract(protocol: Mapping[str, Any]) -> None:
    contract = _mapping(protocol.get("gate_contract"), "gate contract")
    rows = _sequence(
        contract.get("inherited_gates"), "inherited gates"
    )
    observed: dict[str, tuple[str, bool | float]] = {}
    for value in rows:
        row = _mapping(value, "inherited gate")
        name = row.get("name")
        if not isinstance(name, str) or name in observed:
            raise ValueError("inherited gate name is invalid or duplicated")
        threshold = row.get("threshold")
        if isinstance(threshold, bool):
            typed_threshold: bool | float = threshold
        elif isinstance(threshold, (int, float)):
            typed_threshold = float(threshold)
        else:
            raise ValueError(f"inherited gate {name} threshold is invalid")
        observed[name] = (str(row.get("comparison")), typed_threshold)
    if (
        contract.get("weakening_inherited_gates_forbidden") is not True
        or contract.get("all_inherited_and_added_gates_required") is not True
        or contract.get("inherited_gate_count") != 19
        or contract.get(
            "conditional_cell_minimum_closed_head_correct_rows"
        )
        != 20
        or len(rows) != 19
        or observed != EXPECTED_INHERITED_GATES
    ):
        raise ValueError("the exact inherited 19-gate contract changed")

    directional_rows = _sequence(
        contract.get("added_directional_confusion_gates"),
        "directional confusion gates",
    )
    directional: dict[str, dict[str, Any]] = {}
    for value in directional_rows:
        row = dict(_mapping(value, "directional confusion gate"))
        name = row.pop("name", None)
        if not isinstance(name, str) or name in directional:
            raise ValueError("directional confusion gate is duplicated")
        directional[name] = row
    if directional != EXPECTED_DIRECTIONAL_GATES:
        raise ValueError("DSSS/Bluetooth directional gates changed")


def _validate_validation_sequence(
    protocol: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> None:
    stages = [
        _mapping(value, "validation stage")
        for value in _sequence(
            protocol.get("validation_sequence"), "validation sequence"
        )
    ]
    expected_names = [
        "adaptive_design",
        "candidate_freeze",
        "known_scale_validation_primary",
        "known_scale_validation_confirmation",
        "novelty_validation_joint",
        "release_eligibility",
    ]
    if (
        [stage.get("ordinal") for stage in stages] != list(range(6))
        or [stage.get("name") for stage in stages] != expected_names
    ):
        raise ValueError("validation stage order changed")
    allocations = _mapping(
        registry.get("fresh_allocations"), "fresh allocations"
    )
    if (
        stages[0].get("seed_registry_roles")
        != [
            "training_scale",
            "adaptive_scale_development",
            "adaptive_novelty_development",
        ]
        or stages[2].get("seeds")
        != allocations["sealed_known_scale_validation"]["seeds"][:1]
        or stages[3].get("seeds")
        != allocations["sealed_known_scale_validation"]["seeds"][1:]
        or stages[4].get("seeds")
        != allocations["sealed_novelty_validation"]["seeds"]
    ):
        raise ValueError("validation stages and sealed seed registry disagree")
    for index in (2, 3):
        stage = stages[index]
        if (
            stage.get("candidate_bytes_frozen_before_stage") is not True
            or stage.get(
                "generation_claim_ledger_before_corpus_generation"
            )
            is not True
            or stage.get(
                "score_claim_ledger_before_first_model_inference"
            )
            is not True
            or stage.get("metric_miss_report_must_publish") is not True
            or stage.get("selection_allowed") is not False
            or stage.get("adaptation_allowed_after_stage") is not False
            or stage.get("stop_on_failure") is not True
        ):
            raise ValueError(f"one-shot stage {stage['name']} weakened")
    novelty = stages[4]
    if (
        novelty.get("candidate_bytes_frozen_before_stage") is not True
        or novelty.get(
            "joint_claim_ledger_before_generation_or_first_model_inference"
        )
        is not True
        or novelty.get("metric_miss_report_must_publish") is not True
        or novelty.get("selection_allowed") is not False
        or novelty.get("adaptation_allowed_after_stage") is not False
        or novelty.get("stop_on_failure") is not True
    ):
        raise ValueError("joint novelty one-shot stage weakened")
    if (
        stages[3].get("requires_prior_stage_pass")
        != stages[2]["name"]
        or stages[4].get("requires_prior_stage_pass")
        != stages[3]["name"]
        or stages[5].get("requires_prior_stage_pass")
        != stages[4]["name"]
        or stages[5].get("requires_every_validation_gate_pass")
        is not True
        or stages[5].get(
            "requires_identical_candidate_bytes_across_validation"
        )
        is not True
    ):
        raise ValueError("stop-on-first-failure dependency chain changed")
    failure = _mapping(
        protocol.get("failure_policy"), "failure policy"
    )
    if not failure or any(value is not True for value in failure.values()):
        raise ValueError("failure policy must remain entirely fail-closed")


def bound_non_corpus_evidence_paths(
    protocol: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> list[str]:
    """Return every path the validator is allowed to hash besides v5 JSON."""
    lineage = _mapping(protocol.get("lineage"), "lineage")
    ancestor = _mapping(
        lineage.get("ancestor_failed_evidence"),
        "ancestor failed evidence",
    )
    result = [
        str(ancestor[f"{name}_relative_path"])
        for name in ("report", "manifest", "intent", "ledger")
    ]
    for value in _sequence(
        registry.get("spent_or_quarantined"),
        "spent seed inventory",
    ):
        result.append(
            str(_mapping(value, "spent seed row")["evidence_relative_path"])
        )
    return result


def validate_contract(
    protocol: Mapping[str, Any],
    registry: Mapping[str, Any],
    *,
    verify_evidence: bool = True,
) -> dict[str, Any]:
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("schema_version") != 1
        or protocol.get("status")
        != "precommitted_before_v5_corpus_generation_training_or_scoring"
        or protocol.get("development_only") is not True
        or protocol.get("release_evidence") is not False
    ):
        raise ValueError("recovery protocol identity changed")
    _validate_data_contract(protocol)
    _validate_gate_contract(protocol)
    ancestor_paths = _validate_ancestor(
        protocol, verify_evidence=verify_evidence
    )
    seed_paths = _validate_seed_registry(
        protocol,
        registry,
        verify_evidence=verify_evidence,
    )
    _validate_validation_sequence(protocol, registry)
    allowed_paths = bound_non_corpus_evidence_paths(protocol, registry)
    for index, value in enumerate(allowed_paths):
        _metadata_evidence_path(value, f"allowed evidence path {index}")
    return {
        "schema": "atomos.v5.scale-orbit.recovery-validation",
        "schema_version": 1,
        "valid": True,
        "lineage": "v5-scale-orbit",
        "independence_from_seed20262903": False,
        "inherited_gate_count": len(EXPECTED_INHERITED_GATES),
        "added_directional_gate_count": len(EXPECTED_DIRECTIONAL_GATES),
        "spent_seed_count": len(EXPECTED_SPENT),
        "fresh_seed_count": sum(
            len(value["seeds"])
            for value in EXPECTED_ALLOCATIONS.values()
        ),
        "evidence_files_verified": len(ancestor_paths + seed_paths),
        "corpus_files_read": 0,
    }


def validate_repository_contract() -> dict[str, Any]:
    protocol = _load_json(PROTOCOL_PATH, "v5 recovery protocol")
    registry = _load_json(REGISTRY_PATH, "v5 seed registry")
    return validate_contract(protocol, registry, verify_evidence=True)


def main() -> None:
    print(
        json.dumps(
            validate_repository_contract(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
