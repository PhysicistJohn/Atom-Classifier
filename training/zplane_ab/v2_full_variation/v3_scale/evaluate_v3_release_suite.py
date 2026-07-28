"""Evaluate one sealed v3 release suite against the frozen staged v3 candidate.

This is the v3 counterpart of ``evaluate_invariant_release_suite.py`` (the v2
sealed evaluator).  It preserves the v2 discipline exactly where the candidate
did not change, and states precisely where the staged rejector forced a change:

* Every gate name, floor, comparison and the gate assembler itself are
  **imported** from the v2 evaluator (``GATE_FLOORS``, ``_gate``,
  ``_assemble_release_gates``, ``_five_shot_report``, ``_auroc``,
  ``_closed_report``, ...).  No gate is added, removed or re-levelled for the
  next candidate: five-shot remains >= 0.85 and known FUR remains <= 0.10.
  Seed 20260734's predeclared 0.84/0.12 redeclaration is retained verbatim as
  labelled historical metadata only.  It is inactive for this protocol and
  cannot affect a gate object or pass/fail decision.
* The suite-side verification (intent/manifest binding, corpus byte hashes,
  bit-exact prefix nesting, prefix-derivation records, unscored start probe,
  dependency provenance, predeclared five-shot split) is the v2 machinery,
  imported and applied unchanged.
* The evaluator is data-blind until both sides are frozen: the candidate is
  four hash-pinned artifact directories loaded verbatim, and no development
  corpus row, sealed row, or novelty row is ever fit, calibrated, or selected
  on.  ``development_data_loaded`` is ``False``: unlike the dev harnesses this
  evaluator never opens the development corpus at all.

THE CANDIDATE is an explicitly decoupled, hash-bound system.  The release
intent binds one aggregate validation-evidence JSON file, not a runtime-bundle
manifest.  That evidence file binds the pre-validation candidate contract,
the untouched validation report, both fusion artifacts, the validated staged
policy bytes, and the canonical stage-1 prefilter hashes:

1. ``--candidate-manifest``      the final release-candidate JSON, which
                                 binds the post-validation evidence, all
                                 Python/browser assets and dual binding.
2. ``--classifier-bundle-dir``   the self-verified runtime bundle exported
                                 from the regularized 8k classifier fusion.
3. ``--rejector-bundle-dir``     the self-verified runtime bundle exported
                                 from the frozen 4k rejector fusion.
4. ``--classifier-fusion-dir``   the regularized 8k fusion used only for
                                 known-class labels and closed/five-shot/
                                 clean/length/scale metrics.
5. ``--rejector-fusion-dir``     the frozen 4k fusion used only for the
                                 known/unknown decision and its open metrics.
6. ``--staged-dir``              the exact validation-locked 4k stage-2 and
                                 composite-policy artifacts.
7. ``--prefilter-dir``           the exact validation-locked stage-1 set.

The two fusions are intentionally different complete models.  Their weights,
centers, feature moments and prototypes are loaded independently.  The
evaluator never assumes that any of those arrays are interchangeable.

ARCHITECTURE CONTRACT (stated here and carried in the emitted JSON): the v3
rejector is STAGED and is not additive.  When the stage-1 noise prefilter
fires, the capture is rejected as noise before either fusion runs.  Stage-1
survivors run the 4k fusion and frozen staged policy to decide known/unknown.
Only accepted rows receive the 8k fusion's nearest-prototype known label.
The sealed evaluator computes the 8k additive control for every known row
because the unchanged closed and invariance gates require it, but never uses
that control to make an open-set decision.

Predeclared v3 semantics, all bound into the sealed ``evaluation_protocol``
object the release intent must embed (see ``expected_evaluation_protocol``):

1. **Open-set gates score the staged decision axis, staged policy version 4
   (the q97 composite survivor score).**  Stage-2 scores are enrollment ranks in
   ``[0, 1)``; a stage-1 SURVIVOR is scored by the composite
   ``max(stage2_enrollment_rank, stage1_score_enrollment_rank)``, where the
   stage-1 rank is the enrollment-survivor empirical rank of the stage-1
   log-odds (searchsorted-left, the stage-2 convention); a stage-1-gated row
   is placed at ``1 + softsign(stage-1 log-odds)`` in ``(1, 2)``, above every
   survivor.  The staged unknown threshold is the frozen q97 of the composite
   over enrollment survivors, loaded verbatim from the staged artifact's
   composite policy npz, so ``score > threshold`` is exactly the staged
   decision (asserted row by row by the staged module's own equivalence
   check).  This evaluator REFUSES a staged artifact whose recorded staged
   policy version is not the one this evaluator implements.  The known
   false-unknown gate counts BOTH stages, and the per-length open reports
   attribute every rejected known row to the stage that rejected it.  The
   additive control arm keeps the stage-2 policy's own untouched threshold.
2. **Stage-1 applies on its fitted per-length domain, extended upward by the
   causal-prefix rule.**  The pose-degeneracy features are length dependent,
   so the candidate carries one fitted bundle per capture length and no
   bundle is ever substituted sideways.  At a sealed capture length ABOVE
   the longest fitted length (N32768: the development corpus stores N16384,
   so no N32768 model can exist) the stage-1 features are computed on each
   capture's FIRST max-fitted-length samples and gated with that length's
   bundle -- the first 16384 samples of a 32768-sample capture are exactly a
   16384-sample capture of the same emission, the same causal-prefix rule
   the suite's own length corpora are built on.  A capture length BELOW the
   smallest fitted length keeps the additive pass-through semantics: the
   gate cannot fire and every row flows to the unchanged stage-2 path.
   Both cases are recorded per length (``stage_one_active``,
   ``stage_one_feature_length``), declared in the sealed protocol under
   ``stage_one_prefix_rule``, and NOT silently ignored.  The deployed
   TypeScript runtime must apply the identical prefix rule; a runtime that
   refuses such lengths does not match the sealed claim and must be
   reconciled before any ship.
3. **Closed, five-shot, causal-length and physical-scale gates keep their v2
   definitions** on the additive sub-path (frontend -> encoders -> fusion ->
   nearest prototype over every row).  The v2 rejector never participated in
   those gates, the task contract changes only the known-FUR accounting, and
   the stage-1 causal-prefix rule does not change that: no stage gates any
   sweep row.
4. **Fresh novelty at the release seed.**  The novelty base seed IS the
   release seed (mirroring v2, where both were 20260729), with the identical
   per-family SHA-256 derivation.  Consumed and development seeds are refused.

The unstaged additive score of every scored row is also computed as a control
and the staged survivors are asserted to score identically (the staged
module's subset-agreement check), so the staged short circuit cannot drift
from the additive path it embeds.

This evaluator refuses to run twice: it refuses a release root that already
contains ``RELEASE_EVALUATION.json`` and its exclusive writer refuses an
existing output file.
"""
from __future__ import annotations

import argparse
import ast
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
V2 = HERE.parent
ZPAB = V2.parent
TRAINING = ZPAB.parent
REPO = TRAINING.parent
for _path in (TRAINING, ZPAB, V2, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import assemble_v3_fusion as assemble  # noqa: E402
import evaluate_invariant_release_suite as release  # noqa: E402
import export_v3_openset_browser_assets as openset_export  # noqa: E402
import fit_v3_openset as openset_base  # noqa: E402
import fit_v3_openset_staged as staged  # noqa: E402
import measure_v3_remaining_gates as measure  # noqa: E402
import noise_prefilter  # noqa: E402
import preprocess as production_preprocess  # noqa: E402
import time_domain_invariant_patch_preprocess as td_preprocess  # noqa: E402
from openset_eval import NOVELTY  # noqa: E402
from train import embed_all  # noqa: E402
from v3_time_domain_openset import (  # noqa: E402
    FROZEN_GEOMETRY_FEATURE,
    FROZEN_GEOMETRY_WEIGHT,
    FROZEN_POLICY_KIND,
    FROZEN_POLICY_SCHEMA,
    FROZEN_THRESHOLD_QUANTILE,
    FROZEN_V2_WEIGHT,
)


# ---------------------------------------------------------------------------
# identity with the v2 evaluator: imported objects, never re-typed values
# ---------------------------------------------------------------------------

EVALUATOR_SCHEMA = 4
EVALUATION_VERSION = "time-domain-v3-release-evaluation-v4-q97-dual-fusion"
RELEASE_PROTOCOL = release.RELEASE_PROTOCOL

CANDIDATE_MANIFEST_SCHEMA = "time-domain-v3-dual-release-candidate-v2"
CANDIDATE_EVIDENCE_SCHEMA = "time-domain-v3-decoupled-validation-evidence-v2"
CANDIDATE_CONTRACT_SCHEMA = "time-domain-v3-q97-candidate-v1"
CANDIDATE_ID = "v3.4-q97-decoupled-8k-classifier-4k-rejector"
EXPECTED_RELEASE_SEED = 20260736
CONSUMED_PREVIOUS_RELEASE_SEED = 20260735
DESIGN_NOVELTY_SEED = 20260955
VALIDATION_NOVELTY_SEEDS = (20260953, 20260954)

# Copy the v4 policy identity into the evaluator instead of accepting whatever
# the currently-imported fitting module happens to call "current".  The
# module is cross-checked against these literals before a candidate is loaded,
# and every staged/browser artifact is checked against the same values.
EXPECTED_STAGED_POLICY_SCHEMA = 4
EXPECTED_STAGED_POLICY_VERSION = (
    "v3-staged-openset-policy-v4-composite-survivor-q97"
)
EXPECTED_STAGED_POLICY_KIND = (
    "v3_staged_noise_prefilter_then_q97_composite_survivor_lof_geometry"
)
EXPECTED_COMPOSITE_THRESHOLD_QUANTILE = 0.97
EXPECTED_STAGE_ONE_KNOWN_FALSE_POSITIVE_BUDGET = 0.01
EXPECTED_SURVIVOR_KNOWN_FALSE_POSITIVE_BUDGET = (
    1.0 - EXPECTED_COMPOSITE_THRESHOLD_QUANTILE
)
EXPECTED_NOMINAL_ENROLLMENT_FALSE_UNKNOWN_BUDGET = (
    EXPECTED_STAGE_ONE_KNOWN_FALSE_POSITIVE_BUDGET
    + (1.0 - EXPECTED_STAGE_ONE_KNOWN_FALSE_POSITIVE_BUDGET)
    * EXPECTED_SURVIVOR_KNOWN_FALSE_POSITIVE_BUDGET
)
EXPECTED_POLICY_ONLY_CHANGE = (
    "composite survivor enrollment threshold quantile q99 -> q97"
)
EXPECTED_FROZEN_ARCHITECTURE = {
    "frontend": "invariant-patch-time-domain-v1",
    "execution_order": [
        "stage_one_noise_gate",
        "rejector_known_unknown",
        "classifier_known_label",
    ],
    "intentional_dual_fusion": True,
    "known_label_source": "classifier_fusion_8k_regularized",
    "known_unknown_source": "rejector_fusion_4k_frozen_policy",
    "stage_one_short_circuit": True,
    "open_set_decision_changes_closed_label": True,
    "stage_one_causal_prefix_rule": {
        "4096": 4096,
        "8192": 8192,
        "16384": 16384,
        "32768": 16384,
    },
    "uses_frequency_transform": False,
}
EXPECTED_POLICY_HYGIENE = {
    "threshold_population": (
        "enrollment stage-1 survivors only (enrollment only, as every rank "
        "and threshold in this chain)"
    ),
    "training_rows_used_for_threshold": 0,
    "selection_rows_used_for_threshold": 0,
    "novelty_rows_used_for_threshold": 0,
    "release_rows_used_for_threshold": 0,
    "stage_one_known_false_positive_budget": (
        EXPECTED_STAGE_ONE_KNOWN_FALSE_POSITIVE_BUDGET
    ),
    "survivor_known_false_positive_budget": (
        EXPECTED_SURVIVOR_KNOWN_FALSE_POSITIVE_BUDGET
    ),
    "nominal_enrollment_false_unknown_budget": (
        EXPECTED_NOMINAL_ENROLLMENT_FALSE_UNKNOWN_BUDGET
    ),
    "only_policy_change": EXPECTED_POLICY_ONLY_CHANGE,
    "stage_one_changed": False,
    "rejector_cnn_fusion_changed": False,
    "classifier_cnn_fusion_changed": False,
    "gate_floors_and_known_fur_ceiling_unchanged": True,
}
EXPECTED_SERIALIZED_POLICY_HYGIENE = {
    "threshold_population": "enrollment_stage_one_survivors_only",
    "training_rows_used_for_threshold": 0,
    "selection_rows_used_for_threshold": 0,
    "novelty_rows_used_for_threshold": 0,
    "release_rows_used_for_threshold": 0,
    "stage_one_known_false_positive_budget": (
        EXPECTED_STAGE_ONE_KNOWN_FALSE_POSITIVE_BUDGET
    ),
    "survivor_known_false_positive_budget": (
        EXPECTED_SURVIVOR_KNOWN_FALSE_POSITIVE_BUDGET
    ),
    "nominal_enrollment_false_unknown_budget": (
        EXPECTED_NOMINAL_ENROLLMENT_FALSE_UNKNOWN_BUDGET
    ),
    "only_policy_change": EXPECTED_POLICY_ONLY_CHANGE,
    "stage_one_changed": False,
    "rejector_cnn_fusion_changed": False,
    "classifier_cnn_fusion_changed": False,
    "gate_contract_changed": False,
}
STAGING_PACKAGE_SCHEMA = "atomos.v3.time-domain-classifier.dual-runtime-package"
STAGING_PACKAGE_SCHEMA_VERSION = 1
DUAL_BINDING_SCHEMA = "atomos.v3.time-domain-dual-fusion.binding"
DUAL_BINDING_SCHEMA_VERSION = 1
STAGING_STATUS = "staging_not_release"
BROWSER_FUSION_SCHEMA = (
    "atomos.v3.time-domain-invariant-fusion.browser-weights"
)
BROWSER_FUSION_SCHEMA_VERSION = 1
BROWSER_OPENSET_SCHEMA = "atomos.v3.time-domain-openset.staged"
BROWSER_OPENSET_SCHEMA_VERSION = 4
REJECTOR_BROWSER_ASSET = "time-domain-v3-rejector-weights.json"
CLASSIFIER_BROWSER_ASSET = "time-domain-v3-classifier-weights.json"
OPENSET_BROWSER_ASSET = "time-domain-v3-openset-policy.json"
DUAL_BINDING_ASSET = "time-domain-v3-dual-binding.json"
FUSION_EXPORT_MANIFEST_SCHEMA = (
    "atomos.v3.time-domain-invariant-fusion.browser-weights.export-manifest"
)
FUSION_EXPORT_MANIFEST_SCHEMA_VERSION = 2
OPENSET_EXPORT_MANIFEST_SCHEMA = (
    "time-domain-v3-dual-openset-staging-manifest-v1"
)
OPENSET_EXPORT_MANIFEST_SCHEMA_VERSION = 1
PARITY_SCHEMA = "time-domain-openset-parity-v1"
PARITY_SCHEMA_VERSION = 4

GATE_FLOORS = release.GATE_FLOORS
FIVE_SHOT_K = release.FIVE_SHOT_K
HIGH_SNR_DB = release.HIGH_SNR_DB

# Mirrored as module attributes so a test can rescale the whole protocol
# coherently; for a real run they are exactly the v2 values.
REQUIRED_CAPTURE_LENGTHS = release.REQUIRED_CAPTURE_LENGTHS
MATCHED_CAPTURE_LENGTH = release.MATCHED_CAPTURE_LENGTH
MIN_TARGET_PER_CLASS = release.MIN_TARGET_PER_CLASS
NOVELTY_FAMILIES = release.NOVELTY_FAMILIES
NOVELTY_N_EACH_PER_LENGTH = release.NOVELTY_N_EACH_PER_LENGTH
PHYSICAL_SCALE_FACTORS = release.PHYSICAL_SCALE_FACTORS
SCALE_MIN_PER_CLASS = release.SCALE_MIN_PER_CLASS

_gate = release._gate
_sha256 = release._sha256
_release_read_json = release._read_json
_is_below = release._is_below
_regular_file = release._regular_file
_integer = release._integer
_verify_file_record = release._verify_file_record
_write_json_exclusive = release._write_json_exclusive
resolve_device = release.resolve_device


# JSON's surface syntax distinguishes booleans, integers and numbers, but
# Python deliberately aliases ``bool`` to ``int`` and considers ``4 == 4.0``.
# A sealed admission boundary must not.  These helpers are intentionally used
# before semantic comparisons so a type-confused artifact cannot pass merely
# because Python equality is permissive.
_BOOL_FIELD_NAMES = frozenset(
    {
        "active_for_current_protocol",
        "additive_only",
        "all_pass",
        "all_release_gates_pass",
        "all_six_gates_claim_supported_as_complete_count",
        "candidate_inference_behavior_changed",
        "changes_candidate_behavior",
        "changes_closed_label",
        "classifier_runs_only_after_rejector_acceptance",
        "closed_label_agreement",
        "closed_label_can_be_gated",
        "committed_before_validation",
        "development_data_loaded",
        "development_data_used",
        "development_only",
        "distinct_role_assets",
        "exact",
        "exact_target_per_class",
        "gates_are_evidence",
        "gates_before_classification",
        "gate_contract_changed",
        "gate_floors_and_known_fur_ceiling_unchanged",
        "fitting_seed_is_not_a_validation_seed",
        "has_clean_pairs",
        "intentional_dual_fusion",
        "ledger_only",
        "matched_longest_rows",
        "open_set_decision_changes_closed_label",
        "original_text_preserved_verbatim",
        "packaged",
        "passes",
        "policy_changed",
        "public_known_label_from_classifier_only",
        "recalibration_performed",
        "rejector_cnn_fusion_changed",
        "release_evidence",
        "release_seed_drawn",
        "retraining_performed",
        "role_asset_sha256_must_differ",
        "role_assets_bound_by_sha256",
        "same_rows_or_same_novelty_draw",
        "scored",
        "stage_one_changed",
        "stage_one_prefix_rule_applied",
        "stage_one_short_circuit",
        "training_rows_loaded",
        "writes_under_releases",
    }
)
_BOOL_FIELD_SUFFIXES = (
    "_active",
    "_applied",
    "_bound",
    "_changed",
    "_exact",
    "_frozen",
    "_pass",
    "_passes",
    "_performed",
)
_INTEGER_FIELD_SUFFIXES = (
    "_bytes",
    "_count",
    "_counts",
    "_index",
    "_indices",
    "_length",
    "_lengths",
    "_rows",
    "_seed",
    "_seeds",
)
_INTEGER_FIELD_NAMES = frozenset(
    {
        "bytes_per_complex_sample",
        "clean_validation_seeds_from",
        "known_classes",
        "minimum_target_per_class",
        "novelty_seeds_consumed_once",
        "release_seed_not_spent",
        "sealed_release_data_used",
        "staged_policy_schema",
        "target_per_class",
    }
)


def _snake_field_name(value: str) -> str:
    """Normalize snake/camel JSON field names for scalar type admission."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()


def _field_requires_bool(field: str) -> bool:
    name = _snake_field_name(field)
    return (
        name in _BOOL_FIELD_NAMES
        or name.startswith(("has_", "is_"))
        or name.endswith(_BOOL_FIELD_SUFFIXES)
        or re.fullmatch(r"release_seed_\d+_used", name) is not None
    )


def _field_requires_int(field: str, value: Any) -> bool:
    name = _snake_field_name(field)
    if "sha256" in name or name in _BOOL_FIELD_NAMES:
        return False
    if name == "schema":
        # Several artifact families use a string schema identifier; numeric
        # schemas must nevertheless remain exact JSON integers.
        return type(value) is not str
    if name == "schema_version" or name.endswith("_schema_version"):
        return True
    if name in _INTEGER_FIELD_NAMES or name in {
        "bytes",
        "count",
        "index",
        "length",
        "rows",
        "seed",
    }:
        return True
    if name.endswith("_accuracy_all_rows"):
        return False
    return (
        name.endswith(_INTEGER_FIELD_SUFFIXES)
        or "_rows_used_" in name
        or "_rows_per_" in name
        or name.startswith("minimum_rows_")
        or re.search(
            r"_rows_(?:loaded|used|exposed|evaluated|scored_(?:staged|unstaged))$",
            name,
        )
        is not None
        or re.search(
            r"(?:consumed|spent)_.*release_seeds_excluded$", name
        )
        is not None
        or name in {
            "spent_novelty_seeds_excluded",
            "spent_novelty_seeds_refused",
            "fitting_seed_namespace",
        }
    )


def _validate_json_scalar_types(
    value: Any,
    *,
    field: str = "$",
    collection_field: str | None = None,
    force_int: bool = False,
) -> None:
    """Reject JSON scalar type confusion and non-finite numeric extensions.

    ``json.load`` accepts the non-standard ``NaN``/``Infinity`` tokens by
    default, and Python equality accepts booleans and integral floats in many
    integer slots.  This recursive pass closes both holes for every candidate
    and release JSON document before any field is trusted.
    """
    if isinstance(value, Mapping):
        container = _snake_field_name(collection_field or "")
        integer_count_map = (
            container in {"counts", "row_counts"}
            or container.endswith(("_counts", "_row_counts"))
        )
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{field} contains a non-string JSON key")
            numeric_seed_scalar = (
                container == "seeds"
                and type(item) in (bool, int, float)
            )
            _validate_json_scalar_types(
                item,
                field=f"{field}.{key}",
                collection_field=collection_field if integer_count_map else key,
                force_int=(
                    force_int or integer_count_map or numeric_seed_scalar
                ),
            )
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_scalar_types(
                item,
                field=f"{field}[{index}]",
                collection_field=collection_field,
                force_int=force_int,
            )
        return
    leaf = field.rsplit(".", 1)[-1].split("[", 1)[0]
    semantic_field = collection_field or leaf
    if _field_requires_bool(semantic_field):
        if type(value) is not bool:
            raise ValueError(f"{field} must be a JSON boolean")
        return
    if force_int or _field_requires_int(semantic_field, value):
        if type(value) is not int:
            raise ValueError(f"{field} must be an exact JSON integer")
        return
    if type(value) is float and not math.isfinite(value):
        raise ValueError(f"{field} must be a finite JSON number")
    if not (
        value is None
        or type(value) in (str, bool, int, float)
    ):
        raise ValueError(f"{field} contains a non-JSON scalar")


def _require_exact_json_value(found: Any, expected: Any, *, field: str) -> None:
    """Recursive equality with exact JSON scalar types."""
    if isinstance(expected, Mapping):
        if not isinstance(found, Mapping) or set(found) != set(expected):
            raise ValueError(f"{field} object keys differ")
        for key, expected_item in expected.items():
            _require_exact_json_value(
                found[key], expected_item, field=f"{field}.{key}"
            )
        return
    if isinstance(expected, list):
        if not isinstance(found, list) or len(found) != len(expected):
            raise ValueError(f"{field} list shape differs")
        for index, (found_item, expected_item) in enumerate(
            zip(found, expected)
        ):
            _require_exact_json_value(
                found_item,
                expected_item,
                field=f"{field}[{index}]",
            )
        return
    if type(found) is not type(expected) or found != expected:
        raise ValueError(
            f"{field}={found!r} ({type(found).__name__}), expected "
            f"{expected!r} ({type(expected).__name__})"
        )


def _require_finite_json_float(value: Any, *, field: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{field} must be an exact finite JSON number")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    """The v2 sealed reader plus v4's strict scalar type pass."""
    value = _release_read_json(path)
    _validate_json_scalar_types(value, field=str(path))
    return value

#: The one frozen candidate directory set this evaluator was written for.
DEFAULT_REJECTOR_BUNDLE_DIR = (
    V2 / "artifacts" / "invariant_patch" / "v3_scale"
    / "v3_runtime_bundle_rejector4k_dual_seed20260730"
)
DEFAULT_CLASSIFIER_BUNDLE_DIR = (
    V2 / "artifacts" / "invariant_patch" / "v3_scale"
    / "v3_runtime_bundle_classifier8kreg_dual_seed20260730"
)
DEFAULT_CLASSIFIER_FUSION_DIR = (
    V2 / "artifacts" / "invariant_patch" / "v3_scale"
    / "v3_fusion_ml8000reg_seed20260730"
)
DEFAULT_REJECTOR_FUSION_DIR = (
    V2 / "artifacts" / "invariant_patch" / "v3_scale"
    / "v3_fusion_multilength_seed20260730"
)
DEFAULT_STAGED_DIR = (
    V2 / "artifacts" / "invariant_patch" / "v3_scale"
    / "staged_validate_v34_q97_decoupled_rejector4k_classifier8k_budget001_"
    "seeds20260953_20260954"
)
#: The tightened stage-1 gate: the 0.01 enrollment-budget threshold set.
#: Coefficients are bit-identical to the 0.02 set; only the operating points
#: differ (HANDOFF 25's sanctioned path).
DEFAULT_PREFILTER_DIR = (
    V2 / "artifacts" / "invariant_patch" / "v3_scale"
    / "noise_prefilter_fit20261001_budget001" / "bundles"
)
DEFAULT_CANDIDATE_MANIFEST = (
    HERE / "evidence"
    / "v3_4_q97_dual_release_candidate.json"
)
DEFAULT_CANDIDATE_CONTRACT = (
    HERE / "evidence" / "v3_q97_candidate_contract.json"
)
# Unit tests patch this only for their synthetic mini candidate.  Production
# code has no CLI escape hatch: relocated manifests must still bind the exact
# design55 report/source/commit.
ENFORCE_CANONICAL_Q97_EVIDENCE = True

BUNDLE_KIND = "v3-time-domain-centered-invariant-fusion"
BUNDLE_SCHEMA = "atomos.v3.time-domain-invariant-fusion.runtime-bundle"
BUNDLE_SCHEMA_VERSION = 1
BUNDLE_MANIFEST_NAME = "bundle_manifest.json"
BUNDLE_SELF_VERIFICATION_TOLERANCE = 1e-6

#: Release seeds this evaluator refuses outright.  20260729, 20260731 and
#: 20260733 are the consumed sealed suites; 20260730/20260732 are development MODEL seeds
#: whose reuse as release seeds would collide the two namespaces (HANDOFF 25:
#: use 20260733 for the next release); the two bands are the development
#: novelty namespace and the prefilter fit-only band, which are development
#: evidence, not release seeds.
CONSUMED_RELEASE_SEEDS = {
    20260729: "consumed sealed v2 release suite (evidence rule 2)",
    20260731: (
        "consumed sealed v3.0 release suite (HANDOFF 25: 22/23 gates, known "
        "false-unknown failure frozen; evidence rules 2 and 4)"
    ),
    20260733: (
        "consumed sealed v3.2 composite release suite (HANDOFF 27: 21/23 "
        "gates, known FUR 0.1108 and five-shot 0.8449 failures frozen; "
        "evidence rules 2 and 4)"
    ),
    20260734: (
        "consumed sealed v3.2 release suite under its predeclared historical "
        "gate redeclaration (22/23 gates, five-shot 0.8228 failure frozen; "
        "evidence rules 2 and 4)"
    ),
    20260735: (
        "consumed sealed v3.3 release suite (22/23 gates; sole failure was "
        "known false-unknown 149/1309 = 0.11382734912146678 at N32768 "
        "against the unchanged 0.10 ceiling; immutable negative release "
        "evidence, evidence rules 2 and 4)"
    ),
    20260730: (
        "development model/fusion seed; reusing it as a release seed would "
        "collide the model and release namespaces (HANDOFF 25 note)"
    ),
    20260732: (
        "development model seed; reusing it as a release seed would collide "
        "the model and release namespaces (HANDOFF 25 note: use 20260733)"
    ),
}
REFUSED_RELEASE_SEED_BANDS = (
    (20260900, 20260999, "development novelty seed namespace"),
    (20261000, 20261999, "noise-prefilter fit-only seed band"),
)

#: Historical metadata copied from the consumed seed-20260734 protocol.  It
#: remains here so a future release cannot hide that the prior attempt used
#: relaxed levels.  It is deliberately never applied to current gates.
HISTORICAL_GATE_REDECLARATION_RELEASE_SEED = 20260734
HISTORICAL_V3_GATE_REDECLARATION_RATIONALE = (
    "The 0.10 known-FUR ceiling and 0.85 five-shot floor were declared "
    "against the v2 architecture. Two consumed sealed runs (seeds 20260731, "
    "20260733) establish this candidate's true sealed rates at FUR 0.10-0.11 "
    "and five-shot 0.845-0.850 while it beats the frozen v2 bundle on every "
    "identically-measured axis and fixes all six gates v2 failed. The levels "
    "are re-declared for the v3 architecture by the owner on 2026-07-28, "
    "BEFORE release seed 20260734 was generated. Any release claim must "
    "state these two levels alongside the v2 levels they replace."
)
HISTORICAL_V3_GATE_REDECLARATION = {
    "open_known_false_unknown_worst_length": {
        "v2_level": 0.10,
        "v3_level": 0.12,
        "floor_key": "open_known_false_unknown_max",
        "comparison": "max",
        "owner_decision": HISTORICAL_V3_GATE_REDECLARATION_RATIONALE,
    },
    "five_shot_worst_length_balanced": {
        "v2_level": 0.85,
        "v3_level": 0.84,
        "floor_key": "five_shot",
        "comparison": "min",
        "owner_decision": HISTORICAL_V3_GATE_REDECLARATION_RATIONALE,
    },
}

HISTORICAL_CLAIM_CORRECTION_SCHEMA = (
    "time-domain-v3-historical-claim-correction-v1"
)
HISTORICAL_V3_GATE_REDECLARATION_RATIONALE_SHA256 = (
    "cac7c5c4221fffeef59bbe74c2fc1203277e909b7f36511269b3579754a89199"
)


def historical_claim_correction_block() -> dict[str, Any]:
    """Correct unsupported prose without rewriting the historical evidence.

    The seed-20260734 owner-decision string above is intentionally preserved
    verbatim.  This separately-versioned, machine-readable record is the
    authoritative audit of its comparative claims.
    """
    original_sha256 = hashlib.sha256(
        HISTORICAL_V3_GATE_REDECLARATION_RATIONALE.encode("utf-8")
    ).hexdigest()
    if (
        original_sha256
        != HISTORICAL_V3_GATE_REDECLARATION_RATIONALE_SHA256
    ):
        raise ValueError(
            "the immutable seed-20260734 owner-decision text changed; keep "
            "the original verbatim and amend only the versioned correction"
        )
    return {
        "schema": HISTORICAL_CLAIM_CORRECTION_SCHEMA,
        "applies_to_release_seed": (
            HISTORICAL_GATE_REDECLARATION_RELEASE_SEED
        ),
        "original_text_preserved_verbatim": True,
        "original_text_sha256": original_sha256,
        "comparison_design": "cross_seed_unpaired",
        "same_rows_or_same_novelty_draw": False,
        "v2_failed_gate_count": 8,
        "historical_claimed_v2_failed_gate_count": 6,
        "every_identically_measured_axis_claim_supported": False,
        "all_six_gates_claim_supported_as_complete_count": False,
        "corrected_scope": (
            "comparisons to v2 use different sealed release seeds and are "
            "unpaired; the v2 release failed eight gates, not six; therefore "
            "the preserved claim that the candidate beat v2 on every "
            "identically measured axis is unsupported"
        ),
        "affects_current_gate_values_or_pass_fail": False,
    }


def historical_gate_redeclaration_block() -> dict[str, Any]:
    """Return seed 20260734's exact, explicitly inactive redeclaration record.

    Each entry is cross-checked against the imported v2 floor it replaced.
    The wrapper makes its historical scope and inactive status machine
    readable; current gates always use :data:`GATE_FLOORS` unchanged.
    """
    gates: dict[str, Any] = {}
    for gate_name, record in HISTORICAL_V3_GATE_REDECLARATION.items():
        v2_floor = float(GATE_FLOORS[record["floor_key"]])
        if v2_floor != float(record["v2_level"]):
            raise ValueError(
                f"historical gate redeclaration for {gate_name!r} records "
                "v2_level "
                f"{record['v2_level']}, but the imported v2 floor "
                f"{record['floor_key']!r} is {v2_floor}; the redeclaration no "
                "longer describes the levels it replaces"
            )
        gates[gate_name] = {
            "v2_level": float(record["v2_level"]),
            "v3_level": float(record["v3_level"]),
            "owner_decision": record["owner_decision"],
        }
    return {
        "release_seed": HISTORICAL_GATE_REDECLARATION_RELEASE_SEED,
        "active_for_current_protocol": False,
        "current_protocol_gate_source": "imported_v2_gate_floors_unchanged",
        "gates_redeclared": gates,
        "historical_claim_correction": historical_claim_correction_block(),
    }


def q97_policy_contract_block() -> dict[str, Any]:
    """Stable pre-validation identity for the only policy v4 may evaluate."""
    return {
        "schema": EXPECTED_STAGED_POLICY_SCHEMA,
        "version": EXPECTED_STAGED_POLICY_VERSION,
        "kind": EXPECTED_STAGED_POLICY_KIND,
        "survivor_score": staged.COMPOSITE_SURVIVOR_SCORE,
        "threshold_quantile": EXPECTED_COMPOSITE_THRESHOLD_QUANTILE,
        "only_policy_change": EXPECTED_POLICY_ONLY_CHANGE,
        "calibration_hygiene": dict(EXPECTED_SERIALIZED_POLICY_HYGIENE),
        "design_novelty_seed": DESIGN_NOVELTY_SEED,
        "validation_novelty_seeds": list(VALIDATION_NOVELTY_SEEDS),
        "release_seed_never_spent_in_development": EXPECTED_RELEASE_SEED,
    }


def source_transition_intent_block() -> dict[str, Any]:
    """Stable transition intent that can be frozen before validation.

    Endpoint hashes cannot all exist before validation.  Those live in the
    evaluator's later :func:`ordered_source_transition_contract_block`; this
    block freezes only the phase order, seed effects, and allowed assignment
    surface.
    """
    return {
        "schema": "time-domain-v3-q97-source-transition-intent-v1",
        "source": "fit_v3_openset_staged.py",
        "ordered_phases": ["predesign", "postdesign", "postvalidation"],
        "transitions": [
            {
                "order": 1,
                "transition_id": "predesign_to_postdesign",
                "from_phase": "predesign",
                "to_phase": "postdesign",
                "consumed_novelty_seeds": [DESIGN_NOVELTY_SEED],
                "excluded_top_level_assignments": [
                    "SPENT_NOVELTY_SEEDS",
                    "SEED_LEDGER_NOTE",
                ],
            },
            {
                "order": 2,
                "transition_id": "validation_to_postvalidation",
                "from_phase": "postdesign",
                "to_phase": "postvalidation",
                "consumed_novelty_seeds": list(VALIDATION_NOVELTY_SEEDS),
                "excluded_top_level_assignments": [
                    "SPENT_NOVELTY_SEEDS",
                    "FIRST_CLEAN_NOVELTY_SEED",
                    "SEED_LEDGER_NOTE",
                ],
            },
        ],
        "ledger_only": True,
        "candidate_inference_behavior_changed": False,
    }

#: Single source of truth for the causal-prefix rule text: the staged module.
STAGE_ONE_PREFIX_RULE = staged.STAGE_ONE_PREFIX_RULE

STAGE_ONE_UNCOVERED_RULE = (
    "at a capture length ABOVE the longest fitted stage-1 length the gate "
    "applies through the causal-prefix rule (stage_one_prefix_rule): features "
    "are computed on the capture's first max-fitted-length samples and gated "
    "with that length's bundle, recorded as stage_one_active: true with the "
    "effective stage_one_feature_length; at a capture length BELOW the "
    "smallest fitted length the gate cannot fire and every row flows to the "
    "unchanged additive stage-2 path, recorded as stage_one_active: false"
)
SWEEP_REJECTOR_RULE = (
    "the causal-length and physical-scale sweeps measure the closed-set "
    "representation exactly as the v2 evaluator defined them; no stage gates "
    "their rows (rejection never participates in the closed and invariance "
    "gates, and the stage-1 causal-prefix rule does not change that)"
)
KNOWN_FUR_ACCOUNTING = (
    "counts both stages: a known row gated by the stage-1 noise prefilter and "
    "a known survivor whose COMPOSITE score exceeds the frozen staged "
    "threshold (q97 of the composite over enrollment survivors) are both "
    "false-unknown, and every per-length open report attributes each rejected "
    "known row to the stage that rejected it. The additive control arm uses "
    "the stage-2 policy's own untouched threshold"
)

#: Source files whose current bytes MUST equal the hash the frozen artifacts
#: recorded.  Everything here is executed by this evaluator's inference path.
#: ``run_time_domain_dev.py`` is deliberately absent: it is the training-loop
#: script, it is imported but never executed for data here, and it legitimately
#: drifted after the candidate was frozen (HANDOFF 24.1 selection-policy work);
#: its recorded and current hashes are reported instead of enforced.
STAGED_SOURCE_HARD_CONTRACT = (
    "assemble_v3_fusion.py",
    "fit_v3_openset.py",
    "fit_v3_openset_staged.py",
    "invariant_fusion.py",
    "invariant_patch_data.py",
    "known_only_patch_openset.py",
    "noise_prefilter.py",
    "openset_eval.py",
    "pose_degeneracy.py",
    "time_domain_geometry.py",
    "time_domain_invariant_patch_preprocess.py",
    "train.py",
    "v3_time_domain_openset.py",
)
STAGED_SOURCE_RECORDED_ONLY = ("run_time_domain_dev.py",)
BUNDLE_ASSEMBLY_SOURCE_HARD_CONTRACT = (
    "assemble_v3_fusion.py",
    "invariant_fusion.py",
    "invariant_patch_cnn.py",
    "invariant_patch_data.py",
    "invariant_patch_preprocess.py",
    "time_domain_geometry.py",
    "time_domain_invariant_patch_preprocess.py",
    "train.py",
)

# Immutable v3.3 history.  Evaluator-v4 does not admit this transition; keeping
# the complete record under a versioned name prevents the old evidence from
# being silently rewritten while the v3.4 chain is populated.
LEGACY_V3_3_POST_VALIDATION_LEDGER_TRANSITION = {
    "source": "fit_v3_openset_staged.py",
    "origin": "the staged validation artifact",
    "validated_sha256": (
        "6d6d252802029cd57cab1429640d0b285b117caa97a89b994037469e267aa176"
    ),
    "current_sha256": (
        "1ebfecede89eb3d393c8922d0862827efda6356671995656594b536ed19f7514"
    ),
    "normalized_ast_sha256": (
        "86dbb18d8b83845e4f9c067f672b8f69c9fbbe2b9ccf4216755f7eeb8096e967"
    ),
    "excluded_top_level_assignments": (
        "SPENT_NOVELTY_SEEDS",
        "FIRST_CLEAN_NOVELTY_SEED",
        "SEED_LEDGER_NOTE",
    ),
    "validated_parent_commit": (
        "6f6e1e05d94d457e16940ad1d2c14c6d95bbc422"
    ),
    "ledger_commit": "5ebbd07f763470ff1fc27be04e2e340c6171cc63",
    "full_index_diff_sha256": (
        "c071726b44a04d535084b3c10e76eda2c3b61767a542a29173873b95f8466d03"
    ),
    "consumed_validation_seeds": (20260950, 20260951),
    "next_clean_novelty_seed": 20260952,
}

# Policy-v4 has two necessarily ordered, bookkeeping-only source transitions:
#
#   1. the q97 design artifact records the pre-design source; after its single
#      passing draw, seed 20260955 is marked spent before validation;
#   2. the validation artifact records that post-design source; after its
#      single draw, seeds 20260953/20260954 are marked spent before release.
#
# Unknown endpoints are deliberate placeholders at this pre-validation stage.
# They MUST be replaced by lowercase SHA-256/commit/diff bindings from the
# actual ledger-only commits.  Any attempted source admission while one is
# unbound fails closed.  The ordered verifier can bind either the complete
# predesign -> postdesign -> postvalidation chain or only the validation ->
# postvalidation suffix.
ORDERED_STAGED_SOURCE_TRANSITIONS = (
    {
        "order": 1,
        "transition_id": "predesign_to_postdesign",
        "source": "fit_v3_openset_staged.py",
        "origin": "the q97 design artifact",
        "from_phase": "predesign",
        "to_phase": "postdesign",
        "from_sha256": (
            "4cca22059959ec472278f28594bd54a72bbbab04fcf9a294baa81d31fea59be9"
        ),
        "to_sha256": (
            "c412a6f9d0ea81ca760bceadcfe4eec6ddcbb43dc5174d6bdb6724395a1f17fa"
        ),
        "normalized_ast_sha256": (
            "4618b54c28da283e7ac74f23bbcb43f2d55de6746eb244f80336af8368d09a20"
        ),
        "from_commit": "80298b80e6f56360325457efb49e58d13aa93d61",
        "evidence_commit": "825cf3eace2443aa18899dc192911c060045b068",
        "evidence_report_sha256": (
            "bb749aadd5395a3fc21ff9453621ad8fa7a7512f9b3c21c29cee933b09adfb0f"
        ),
        "to_commit": "4306f67c298376b944bc3be58553983174538bce",
        "full_index_diff_sha256": (
            "9910df1e63ccfb482820ea83f1bec6475069a3dcbc85d134750ebb35fc769fec"
        ),
        "excluded_top_level_assignments": (
            "SPENT_NOVELTY_SEEDS",
            "SEED_LEDGER_NOTE",
        ),
        "consumed_novelty_seeds": (DESIGN_NOVELTY_SEED,),
        "next_clean_novelty_seed": 20260953,
    },
    {
        "order": 2,
        "transition_id": "validation_to_postvalidation",
        "source": "fit_v3_openset_staged.py",
        "origin": "the staged validation artifact",
        "from_phase": "postdesign",
        "to_phase": "postvalidation",
        "from_sha256": None,
        "to_sha256": None,
        "normalized_ast_sha256": None,
        "from_commit": None,
        "evidence_commit": None,
        "evidence_report_sha256": None,
        "to_commit": None,
        "full_index_diff_sha256": None,
        "excluded_top_level_assignments": (
            "SPENT_NOVELTY_SEEDS",
            "FIRST_CLEAN_NOVELTY_SEED",
            "SEED_LEDGER_NOTE",
        ),
        "consumed_novelty_seeds": VALIDATION_NOVELTY_SEEDS,
        "next_clean_novelty_seed": 20260956,
    },
)
_TRANSITION_EVIDENCE_PATHS = {
    "predesign_to_postdesign": (
        "training/zplane_ab/v2_full_variation/artifacts/invariant_patch/"
        "v3_scale/staged_design_v34_q97_rejector4k_budget001_seed20260955/"
        "openset_metrics.json"
    ),
    "validation_to_postvalidation": (
        "training/zplane_ab/v2_full_variation/artifacts/invariant_patch/"
        "v3_scale/staged_validate_v34_q97_decoupled_rejector4k_"
        "classifier8k_budget001_seeds20260953_20260954/openset_metrics.json"
    ),
}
_SOURCE_LOOKUP = {
    "assemble_v3_fusion.py": HERE / "assemble_v3_fusion.py",
    "fit_v3_openset.py": HERE / "fit_v3_openset.py",
    "fit_v3_openset_staged.py": HERE / "fit_v3_openset_staged.py",
    "noise_prefilter.py": HERE / "noise_prefilter.py",
    "pose_degeneracy.py": HERE / "pose_degeneracy.py",
    "run_time_domain_dev.py": HERE / "run_time_domain_dev.py",
    "export_v3_openset_browser_assets.py": (
        HERE / "export_v3_openset_browser_assets.py"
    ),
    "measure_v3_remaining_gates.py": HERE / "measure_v3_remaining_gates.py",
    "invariant_fusion.py": V2 / "invariant_fusion.py",
    "invariant_patch_cnn.py": V2 / "invariant_patch_cnn.py",
    "invariant_patch_data.py": V2 / "invariant_patch_data.py",
    "known_only_patch_openset.py": V2 / "known_only_patch_openset.py",
    "openset_eval.py": V2 / "openset_eval.py",
    "v3_time_domain_openset.py": V2 / "v3_time_domain_openset.py",
    "run_invariant_cnn_dev.py": V2 / "run_invariant_cnn_dev.py",
    "evaluate_invariant_release_suite.py": (
        V2 / "evaluate_invariant_release_suite.py"
    ),
    "invariant_patch_preprocess.py": TRAINING / "invariant_patch_preprocess.py",
    "preprocess.py": TRAINING / "preprocess.py",
    "time_domain_geometry.py": TRAINING / "time_domain_geometry.py",
    "time_domain_invariant_patch_preprocess.py": (
        TRAINING / "time_domain_invariant_patch_preprocess.py"
    ),
    "train.py": TRAINING / "train.py",
}

# Complete transitive Python source surface reached by candidate loading,
# release-suite admission, inference, reporting and the imported v2 helpers.
# The evaluator itself is pinned separately by preflight because a source file
# cannot contain a non-self-referential hash of its own final bytes.
TRANSITIVE_DEPENDENCY_PATHS: dict[str, Path] = {
    **{
        f"training/{name}": TRAINING / name
        for name in (
            "canonical_probe.py",
            "dataset.py",
            "invariant_patch_preprocess.py",
            "model.py",
            "preprocess.py",
            "rfgen.py",
            "time_domain_geometry.py",
            "time_domain_invariant_patch_preprocess.py",
            "train.py",
        )
    },
    **{
        f"training/zplane_ab/{name}": ZPAB / name
        for name in (
            "common_split.py",
            "train_common.py",
            "zplane_backbone.py",
        )
    },
    **{
        f"training/zplane_ab/v2_full_variation/{name}": V2 / name
        for name in (
            "assemble_invariant_candidate.py",
            "canonical_probe_net.py",
            "complex_multiscale_backbone.py",
            "denoise_eval.py",
            "equalizer_frontend.py",
            "evaluate_ab_v2.py",
            "evaluate_invariant_release_suite.py",
            "full_split.py",
            "ground_state_data.py",
            "invariant_fusion.py",
            "invariant_patch_cnn.py",
            "invariant_patch_data.py",
            "known_only_patch_openset.py",
            "length_aug.py",
            "multitask_autoencoder.py",
            "native_preprocess.py",
            "openset_eval.py",
            "pool_cache.py",
            "run_bounded_dev.py",
            "run_corrected_unet.py",
            "run_invariant_cnn_dev.py",
            "scalar_transfer.py",
            "train_common_v2.py",
            "train_transfer.py",
            "unet_multitask.py",
            "unet_transfer.py",
            "v3_time_domain_openset.py",
            "vit_backbone.py",
        )
    },
    **{
        f"training/zplane_ab/v2_full_variation/v3_scale/{name}": HERE / name
        for name in (
            "assemble_v3_fusion.py",
            "export_v3_openset_browser_assets.py",
            "fit_v3_openset.py",
            "fit_v3_openset_staged.py",
            "measure_pose_degeneracy.py",
            "measure_v3_remaining_gates.py",
            "noise_prefilter.py",
            "pose_degeneracy.py",
            "run_time_domain_dev.py",
        )
    },
}
if len(TRANSITIVE_DEPENDENCY_PATHS) != 49:  # pragma: no cover - import guard
    raise RuntimeError("v4 transitive dependency contract must contain 49 files")

# Filled only once every source is final.  Until then a canonical release
# fails closed, while synthetic unit candidates explicitly disable canonical
# evidence and remain usable.
EXPECTED_TRANSITIVE_DEPENDENCY_SHA256: dict[str, str | None] = {
    key: None for key in TRANSITIVE_DEPENDENCY_PATHS
}
EXPECTED_EVALUATOR_RUNTIME_IDENTITY = {
    "python": "3.9.6",
    "numpy": "2.0.2",
    "torch": "2.8.0",
    "device": "cpu",
    "platform": "darwin-arm64",
}


def _current_evaluator_runtime_identity(device: torch.device) -> dict[str, str]:
    torch_version = str(torch.__version__).split("+", 1)[0]
    return {
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "numpy": str(np.__version__),
        "torch": torch_version,
        "device": str(torch.device(device).type),
        "platform": f"{platform.system().lower()}-{platform.machine().lower()}",
    }


def _admit_transitive_execution_contract(
    *,
    canonical_release_candidate: bool,
    device: torch.device,
    dependency_paths: Mapping[str, Path] | None = None,
    expected_sha256: Mapping[str, str | None] | None = None,
    expected_runtime: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Admit the exact 49-file/runtime execution surface, or fail closed."""
    paths = dict(
        TRANSITIVE_DEPENDENCY_PATHS
        if dependency_paths is None
        else dependency_paths
    )
    hashes = dict(
        EXPECTED_TRANSITIVE_DEPENDENCY_SHA256
        if expected_sha256 is None
        else expected_sha256
    )
    runtime = dict(
        EXPECTED_EVALUATOR_RUNTIME_IDENTITY
        if expected_runtime is None
        else expected_runtime
    )
    if set(paths) != set(TRANSITIVE_DEPENDENCY_PATHS):
        raise ValueError("transitive dependency path keys differ from exact 49-file contract")
    if set(hashes) != set(TRANSITIVE_DEPENDENCY_PATHS):
        raise ValueError("transitive dependency hash keys differ from exact 49-file contract")
    _require_exact_json_value(
        runtime,
        EXPECTED_EVALUATOR_RUNTIME_IDENTITY,
        field="evaluator_runtime_identity",
    )
    observed_runtime = _current_evaluator_runtime_identity(device)
    _require_exact_json_value(
        observed_runtime,
        runtime,
        field="observed_evaluator_runtime_identity",
    )
    observed: dict[str, str] = {}
    for key in TRANSITIVE_DEPENDENCY_PATHS:
        path = Path(paths[key])
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                f"transitive dependency {key} must be a regular non-symlink file"
            )
        digest = _sha256(path)
        observed[key] = digest
        expected = hashes[key]
        if canonical_release_candidate:
            configured = _require_sha256(
                expected, field=f"transitive_dependency_sha256.{key}"
            )
            if digest != configured:
                raise ValueError(f"transitive dependency hash mismatch: {key}")
    return {
        "schema": "time-domain-v3-transitive-execution-contract-v1",
        "canonical_enforced": bool(canonical_release_candidate),
        "dependency_file_count": 49,
        "dependency_source_sha256": observed,
        "runtime_identity": observed_runtime,
        "passes": True,
    }


def _source_path(name: str) -> Path:
    try:
        return _SOURCE_LOOKUP[name]
    except KeyError as exc:
        raise ValueError(f"unknown evaluator source file {name!r}") from exc


def _read_candidate_json(path: Path) -> dict[str, Any]:
    """Read a frozen candidate-side JSON artifact.

    Deliberately not ``release._read_json``: that validator enforces the sealed
    release-root JSON contract, which rejects integers beyond 2**53 - 1, and
    development artifacts legitimately carry nanosecond mtimes inside their
    recorded cache contracts.  Candidate files are still bound by SHA-256.
    """
    _regular_file(path, name=str(path.name))
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    _validate_json_scalar_types(value, field=str(path))
    return value


# ---------------------------------------------------------------------------
# the predeclared v3 protocol object
# ---------------------------------------------------------------------------


def validate_release_seed(value: Any) -> int:
    seed = _integer(value, "release_seed")
    if seed in CONSUMED_RELEASE_SEEDS:
        raise ValueError(
            f"release seed {seed} is refused: {CONSUMED_RELEASE_SEEDS[seed]}"
        )
    for low, high, reason in REFUSED_RELEASE_SEED_BANDS:
        if low <= seed <= high:
            raise ValueError(
                f"release seed {seed} lies in the {reason} ({low}-{high}) and "
                "is not an untouched release seed"
            )
    if seed != EXPECTED_RELEASE_SEED:
        raise ValueError(
            f"evaluator schema {EVALUATOR_SCHEMA} is frozen for untouched "
            f"release seed {EXPECTED_RELEASE_SEED}, not {seed}"
        )
    return seed


def _assert_expected_policy_module() -> None:
    """Fail if imported fitting code no longer means the frozen q97 policy."""
    expected = {
        "schema": EXPECTED_STAGED_POLICY_SCHEMA,
        "version": EXPECTED_STAGED_POLICY_VERSION,
        "kind": EXPECTED_STAGED_POLICY_KIND,
        "threshold_quantile": EXPECTED_COMPOSITE_THRESHOLD_QUANTILE,
        "design_novelty_seed": DESIGN_NOVELTY_SEED,
        "validation_novelty_seeds": VALIDATION_NOVELTY_SEEDS,
        "release_seed_never_spent_here": EXPECTED_RELEASE_SEED,
    }
    found = {
        "schema": int(staged.STAGED_POLICY_SCHEMA),
        "version": staged.STAGED_POLICY_VERSION,
        "kind": staged.STAGED_POLICY_KIND,
        "threshold_quantile": float(staged.COMPOSITE_THRESHOLD_QUANTILE),
        "design_novelty_seed": int(staged.PROPOSED_DESIGN_NOVELTY_SEED),
        "validation_novelty_seeds": tuple(
            int(seed) for seed in staged.DEFAULT_VALIDATION_NOVELTY_SEEDS
        ),
        "release_seed_never_spent_here": int(
            staged.RELEASE_SEED_NEVER_SPENT_HERE
        ),
    }
    if found != expected:
        raise ValueError(
            "imported staged policy identity/seed hygiene differs from the "
            f"evaluator-v4 q97 contract: found {found!r}, expected {expected!r}"
        )
    hygiene = staged._policy_hygiene_expected(staged.STAGED_POLICY_SPEC)
    if hygiene != EXPECTED_SERIALIZED_POLICY_HYGIENE:
        raise ValueError(
            "imported staged policy calibration hygiene differs from the "
            "evaluator-v4 q97 contract"
        )


def expected_evaluation_protocol(
    release_seed: int,
    stage_one_lengths: Sequence[int],
) -> dict[str, Any]:
    """The exact object the release intent/manifest must embed for this run.

    Built from the v2 protocol object so every unchanged rule is byte-identical
    to v2's, then updated only where the v3 candidate changed the semantics.
    ``gates`` is the imported v2 ``GATE_FLOORS`` dict unchanged, so no floor
    can drift between the intent, launcher and evaluator.  The consumed
    seed-20260734 redeclaration travels separately as explicitly inactive
    historical metadata.
    """
    _assert_expected_policy_module()
    seed = validate_release_seed(release_seed)
    domain = tuple(sorted(int(length) for length in stage_one_lengths))
    if not domain or len(set(domain)) != len(domain):
        raise ValueError("stage-1 lengths must be a non-empty set of ints")
    max_fitted = int(max(domain))
    prefix_gated = [
        int(length)
        for length in REQUIRED_CAPTURE_LENGTHS
        if int(length) not in domain and int(length) > max_fitted
    ]
    uncovered = [
        int(length)
        for length in REQUIRED_CAPTURE_LENGTHS
        if int(length) not in domain and int(length) < max_fitted
    ]
    protocol = copy.deepcopy(release.EXPECTED_EVALUATION_PROTOCOL)
    protocol["version"] = EVALUATION_VERSION
    protocol["matched_capture_length"] = int(MATCHED_CAPTURE_LENGTH)
    protocol["required_capture_lengths"] = [
        int(length) for length in REQUIRED_CAPTURE_LENGTHS
    ]
    protocol["minimum_target_per_class"] = int(MIN_TARGET_PER_CLASS)
    protocol["physical_scale_factors"] = [
        float(factor) for factor in PHYSICAL_SCALE_FACTORS
    ]
    protocol["physical_scale_support"] = dict(
        protocol["physical_scale_support"]
    )
    protocol["physical_scale_support"]["minimum_rows_per_class"] = int(
        SCALE_MIN_PER_CLASS
    )
    protocol["novelty"] = dict(protocol["novelty"])
    protocol["novelty"]["seed"] = seed
    protocol["novelty"]["seed_rule"] = (
        "the novelty base seed equals the untouched release seed; consumed "
        "release seeds and development novelty/fitting seed bands are refused"
    )
    protocol["novelty"]["n_each_per_length"] = int(NOVELTY_N_EACH_PER_LENGTH)
    protocol["novelty"]["calibration"] = (
        "none; frozen per-length stage-1 noise-prefilter bundles, the frozen "
        "stage-2 branch-LOF ensemble and geometry blend, and the frozen "
        "composite survivor policy (enrollment-survivor stage-1 rank "
        "calibration and its q97 composite threshold) are loaded verbatim"
    )
    protocol["open_set"] = {
        "architecture": EXPECTED_STAGED_POLICY_KIND,
        "candidate_architecture": "dual_fusion_classifier8k_rejector4k",
        "classifier_role": (
            "the regularized 8k fusion supplies every accepted known-class "
            "label and every closed/five-shot/clean/length/scale gate input"
        ),
        "rejector_role": (
            "the frozen 4k fusion supplies every stage-2 known/unknown score "
            "and every open-set gate input; it never supplies the final "
            "accepted known-class label"
        ),
        "decision_order": [
            "stage-one causal-prefix noise gate",
            "4k rejector fusion and frozen staged known/unknown policy",
            "8k classifier fusion nearest-prototype known label for accepted rows",
        ],
        "intentional_dual_fusion": True,
        "staged_policy_schema": EXPECTED_STAGED_POLICY_SCHEMA,
        "staged_policy_version": EXPECTED_STAGED_POLICY_VERSION,
        "staged_policy_kind": EXPECTED_STAGED_POLICY_KIND,
        "composite_threshold_quantile": (
            EXPECTED_COMPOSITE_THRESHOLD_QUANTILE
        ),
        "policy_hygiene": dict(EXPECTED_POLICY_HYGIENE),
        "design_novelty_seed": DESIGN_NOVELTY_SEED,
        "validation_novelty_seeds": list(VALIDATION_NOVELTY_SEEDS),
        "additive_only": False,
        "changes_closed_label": True,
        "open_set_decision_changes_closed_label": True,
        "gates_before_classification": True,
        "survivor_score": staged.COMPOSITE_SURVIVOR_SCORE,
        "unknown_threshold_rule": (
            "the staged unknown threshold is the frozen "
            f"q{EXPECTED_COMPOSITE_THRESHOLD_QUANTILE} of the composite "
            "over enrollment stage-1 survivors, fit on enrollment only and "
            "loaded verbatim from the staged artifact; the additive control "
            "arm keeps the stage-2 policy's own untouched enrollment "
            "threshold"
        ),
        "stage_one_capture_lengths": [int(length) for length in domain],
        "stage_one_max_fitted_length": max_fitted,
        "stage_one_prefix_gated_lengths": prefix_gated,
        "stage_one_prefix_rule": {
            "max_fitted_length": max_fitted,
            "rule": STAGE_ONE_PREFIX_RULE,
        },
        "stage_one_uncovered_lengths": uncovered,
        "stage_one_uncovered_rule": STAGE_ONE_UNCOVERED_RULE,
        "sweep_rejector_rule": SWEEP_REJECTOR_RULE,
        "known_false_unknown_accounting": KNOWN_FUR_ACCOUNTING,
        "score_axis": staged.STAGED_SCORE_NOTE,
    }
    # The strict imported v2 floors are the current acceptance contract.
    # Seed 20260734's relaxed levels remain visible only as explicitly
    # inactive historical metadata and never mutate this object.
    protocol["gates"] = dict(GATE_FLOORS)
    protocol["historical_gate_redeclaration"] = (
        historical_gate_redeclaration_block()
    )
    return protocol


# ---------------------------------------------------------------------------
# the frozen candidate
# ---------------------------------------------------------------------------


@dataclass
class V3Candidate:
    candidate_manifest_path: Path
    candidate_manifest_sha256: str
    candidate_manifest: dict[str, Any]
    validation_evidence_path: Path
    validation_evidence_sha256: str
    validation_evidence: dict[str, Any]
    frozen_contract_path: Path
    frozen_contract_sha256: str
    frozen_contract: dict[str, Any]
    classifier_bundle_dir: Path
    classifier_bundle_manifest_path: Path
    classifier_bundle_manifest_sha256: str
    classifier_bundle_manifest: dict[str, Any]
    classifier_bundle_arrays: dict[str, np.ndarray]
    rejector_bundle_dir: Path
    rejector_bundle_manifest_path: Path
    rejector_bundle_manifest_sha256: str
    rejector_bundle_manifest: dict[str, Any]
    rejector_bundle_arrays: dict[str, np.ndarray]
    classifier_fusion_dir: Path
    classifier_fusion_artifact: Any  # openset_base.FusionArtifact
    classifier_fusion_module: torch.nn.Module  # CenteredInvariantFusion
    rejector_fusion_dir: Path
    rejector_fusion_artifact: Any  # openset_base.FusionArtifact
    rejector_fusion_module: torch.nn.Module  # CenteredInvariantFusion
    staged_dir: Path
    staged_metrics_path: Path
    staged_metrics_sha256: str
    staged_metrics: dict[str, Any]
    staged_hashes: dict[str, str]
    prefilter_dir: Path
    prefilter_set_sha256: str
    stage_one: Any  # staged.StageOneGate
    stage_one_lengths: tuple[int, ...]
    posedegen: Any
    prefilter_module: Any
    rejector: Any  # openset_base.Rejector
    composite: Any  # staged.CompositeSurvivorPolicy
    classes: tuple[str, ...]
    patch_length: int
    patch_count: int
    target_frac: float
    classifier_feature_mean: np.ndarray
    classifier_feature_std: np.ndarray
    rejector_feature_mean: np.ndarray
    rejector_feature_std: np.ndarray
    #: The STAGED decision threshold: the composite policy's frozen q97.
    threshold: float
    #: The stage-2 policy's own untouched threshold (the additive control).
    stage_two_threshold: float
    source_report: dict[str, Any]

    # Compatibility aliases are deliberately read-only and point to the
    # rejector role.  New callers must use the explicit role-named fields;
    # the CLI and report schemas no longer accept the old single-fusion
    # contract.
    @property
    def fusion_dir(self) -> Path:
        return self.rejector_fusion_dir

    @property
    def fusion_artifact(self) -> Any:
        return self.rejector_fusion_artifact

    @property
    def fusion_module(self) -> torch.nn.Module:
        return self.rejector_fusion_module

    @property
    def feature_mean(self) -> np.ndarray:
        return self.rejector_feature_mean

    @property
    def feature_std(self) -> np.ndarray:
        return self.rejector_feature_std

    @property
    def bundle_dir(self) -> Path:
        return self.rejector_bundle_dir

    @property
    def bundle_manifest_path(self) -> Path:
        return self.rejector_bundle_manifest_path

    @property
    def bundle_manifest_sha256(self) -> str:
        return self.rejector_bundle_manifest_sha256

    @property
    def bundle_manifest(self) -> dict[str, Any]:
        return self.rejector_bundle_manifest

    @property
    def bundle_arrays(self) -> dict[str, np.ndarray]:
        return self.rejector_bundle_arrays


def _verify_source_contract(
    recorded: Mapping[str, Any],
    hard_names: Sequence[str],
    *,
    origin: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compare recorded source hashes against current bytes; fail on drift."""
    checked: dict[str, Any] = {}
    transitions: dict[str, Any] = {}
    for name in hard_names:
        expected = recorded.get(name)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"{origin} records no usable hash for {name!r}")
        path = _source_path(name)
        current = _sha256(path)
        if current != expected:
            transition = _admit_ordered_seed_ledger_transitions(
                name=name,
                origin=origin,
                expected=expected,
                current=current,
                path=path,
            )
            if transition is None:
                raise ValueError(
                    f"source drift: {name} is {current}, but {origin} froze "
                    f"{expected}; the loaded module is not the one the candidate "
                    "was validated with"
                )
            transitions[name] = transition
        checked[name] = current
    return checked, transitions


def _ledger_neutral_ast_text_sha256(
    source: str,
    excluded_assignments: Sequence[str],
    *,
    filename: str,
) -> str:
    """Hash source AST after removing only named top-level assignments."""
    excluded = set(excluded_assignments)
    if not excluded or len(excluded) != len(tuple(excluded_assignments)):
        raise ValueError("ledger transition exclusion names are invalid")
    tree = ast.parse(source, filename=filename)
    kept: list[ast.stmt] = []
    removed: list[str] = []
    for node in tree.body:
        targets: Sequence[ast.expr] = ()
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = (node.target,)
        names = [
            target.id for target in targets if isinstance(target, ast.Name)
        ]
        if names and set(names).issubset(excluded):
            removed.extend(names)
        else:
            kept.append(node)
    if sorted(removed) != sorted(excluded):
        raise ValueError(
            "ledger transition source does not contain exactly the declared "
            "top-level assignments"
        )
    tree.body = kept
    payload = ast.dump(
        tree, annotate_fields=True, include_attributes=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ledger_neutral_ast_sha256(
    path: Path, excluded_assignments: Sequence[str]
) -> str:
    return _ledger_neutral_ast_text_sha256(
        path.read_text(encoding="utf-8"),
        excluded_assignments,
        filename=str(path),
    )


def _git_output(arguments: Sequence[str]) -> bytes:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=REPO,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            "q97 source transition Git binding cannot be recomputed: "
            + " ".join(arguments)
        ) from exc


def _git_blob(commit: str, path: str) -> bytes:
    return _git_output(("show", f"{commit}:{path}"))


def _verify_git_transition_binding(transition: Mapping[str, Any]) -> None:
    """Recompute every commit/source/evidence/diff endpoint from Git."""
    transition_id = str(transition["transition_id"])
    source_path = (
        "training/zplane_ab/v2_full_variation/v3_scale/"
        "fit_v3_openset_staged.py"
    )
    evidence_path = _TRANSITION_EVIDENCE_PATHS[transition_id]
    from_commit = _configured_transition_commit(transition, "from_commit")
    evidence_commit = _configured_transition_commit(
        transition, "evidence_commit"
    )
    to_commit = _configured_transition_commit(transition, "to_commit")
    if len({from_commit, evidence_commit, to_commit}) != 3:
        raise ValueError(
            f"q97 source transition {transition_id} commit order aliases "
            "distinct phases"
        )
    for earlier, later in (
        (from_commit, evidence_commit),
        (evidence_commit, to_commit),
    ):
        try:
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", earlier, later],
                cwd=REPO,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError(
                f"q97 source transition {transition_id} commit order differs"
            ) from exc
    try:
        evidence_at_from = subprocess.run(
            ["git", "cat-file", "-e", f"{from_commit}:{evidence_path}"],
            cwd=REPO,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise ValueError(
            f"q97 source transition {transition_id} evidence history "
            "cannot be recomputed"
        ) from exc
    if evidence_at_from.returncode == 0:
        raise ValueError(
            f"q97 source transition {transition_id} evidence already exists "
            "at the from phase"
        )
    source_blobs = {
        "from": _git_blob(from_commit, source_path),
        "evidence": _git_blob(evidence_commit, source_path),
        "to": _git_blob(to_commit, source_path),
    }
    expected_from = _configured_transition_digest(transition, "from_sha256")
    expected_to = _configured_transition_digest(transition, "to_sha256")
    for endpoint, expected in (
        ("from", expected_from),
        ("evidence", expected_from),
        ("to", expected_to),
    ):
        if hashlib.sha256(source_blobs[endpoint]).hexdigest() != expected:
            raise ValueError(
                f"q97 source transition {transition_id} {endpoint} source "
                "blob differs"
            )
    evidence_digest = _configured_transition_digest(
        transition, "evidence_report_sha256"
    )
    for commit, endpoint in (
        (evidence_commit, "evidence"),
        (to_commit, "to"),
    ):
        if hashlib.sha256(_git_blob(commit, evidence_path)).hexdigest() != (
            evidence_digest
        ):
            raise ValueError(
                f"q97 source transition {transition_id} {endpoint} evidence "
                "blob differs"
            )
    diff = _git_output(
        ("diff", from_commit, to_commit, "--", source_path)
    )
    if hashlib.sha256(diff).hexdigest() != _configured_transition_digest(
        transition, "full_index_diff_sha256"
    ):
        raise ValueError(
            f"q97 source transition {transition_id} source diff differs"
        )
    excluded = tuple(transition["excluded_top_level_assignments"])
    expected_ast = _configured_transition_digest(
        transition, "normalized_ast_sha256"
    )
    for endpoint, blob in source_blobs.items():
        try:
            text = blob.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"q97 source transition {transition_id} {endpoint} source "
                "is not UTF-8"
            ) from exc
        if _ledger_neutral_ast_text_sha256(
            text, excluded, filename=f"{commit}:{source_path}"
        ) != expected_ast:
            raise ValueError(
                f"q97 source transition {transition_id} {endpoint} normalized "
                "AST differs"
            )


def ordered_source_transition_contract_block() -> dict[str, Any]:
    """Return the exact, ordered policy-v4 source-transition contract.

    ``None`` values are intentional fail-closed placeholders.  A release
    candidate cannot rely on the transition chain until every endpoint,
    normalized AST, commit and diff digest has been bound.
    """
    return {
        "schema": "time-domain-v3-q97-source-transition-contract-v1",
        "source": "fit_v3_openset_staged.py",
        "ordered_phases": ["predesign", "postdesign", "postvalidation"],
        "transitions": [
            {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in transition.items()
            }
            for transition in ORDERED_STAGED_SOURCE_TRANSITIONS
        ],
        "ledger_only": True,
        "candidate_inference_behavior_changed": False,
    }


def _configured_transition_digest(
    transition: Mapping[str, Any],
    key: str,
) -> str:
    value = transition.get(key)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(
            "q97 source transition contract is unbound: "
            f"{transition.get('transition_id')}.{key} must be an exact "
            "lowercase SHA-256 before release evaluation"
        )
    return value


def _configured_transition_commit(
    transition: Mapping[str, Any],
    key: str,
) -> str:
    value = transition.get(key)
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(
            "q97 source transition contract is unbound: "
            f"{transition.get('transition_id')}.{key} must be an exact "
            "40-character commit before release evaluation"
        )
    return value


def _validated_ordered_source_transitions(
    *, required_through_order: int = 2
) -> tuple[dict[str, Any], ...]:
    transitions = tuple(dict(item) for item in ORDERED_STAGED_SOURCE_TRANSITIONS)
    if len(transitions) != 2:
        raise ValueError("q97 source transition contract must contain two steps")
    previous_to: str | None = None
    expected_ids = (
        "predesign_to_postdesign",
        "validation_to_postvalidation",
    )
    expected_descriptors = (
        {
            "origin": "the q97 design artifact",
            "from_phase": "predesign",
            "to_phase": "postdesign",
            "excluded_top_level_assignments": (
                "SPENT_NOVELTY_SEEDS",
                "SEED_LEDGER_NOTE",
            ),
            "next_clean_novelty_seed": 20260953,
        },
        {
            "origin": "the staged validation artifact",
            "from_phase": "postdesign",
            "to_phase": "postvalidation",
            "excluded_top_level_assignments": (
                "SPENT_NOVELTY_SEEDS",
                "FIRST_CLEAN_NOVELTY_SEED",
                "SEED_LEDGER_NOTE",
            ),
            "next_clean_novelty_seed": 20260956,
        },
    )
    for index, (transition, expected_id) in enumerate(
        zip(transitions, expected_ids), start=1
    ):
        if (
            type(transition.get("order")) is not int
            or transition.get("order") != index
            or transition.get("transition_id") != expected_id
            or transition.get("source") != "fit_v3_openset_staged.py"
        ):
            raise ValueError(
                "q97 source transition contract order/identity is invalid"
            )
        for key, expected in expected_descriptors[index - 1].items():
            found = transition.get(key)
            if key == "excluded_top_level_assignments":
                found = tuple(found or ())
            if found != expected:
                raise ValueError(
                    "q97 source transition contract descriptor differs: "
                    f"{expected_id}.{key}"
                )
        if index <= required_through_order:
            from_sha = _configured_transition_digest(
                transition, "from_sha256"
            )
            to_sha = _configured_transition_digest(transition, "to_sha256")
            _configured_transition_digest(
                transition, "normalized_ast_sha256"
            )
            _configured_transition_digest(
                transition, "full_index_diff_sha256"
            )
            _configured_transition_commit(transition, "from_commit")
            _configured_transition_commit(transition, "evidence_commit")
            _configured_transition_digest(
                transition, "evidence_report_sha256"
            )
            _configured_transition_commit(transition, "to_commit")
            if previous_to is not None and from_sha != previous_to:
                raise ValueError(
                    "q97 source transition chain is not contiguous at "
                    f"{expected_id}"
                )
            if from_sha == to_sha:
                raise ValueError(
                    f"q97 source transition {expected_id} does not change bytes"
                )
            _verify_git_transition_binding(transition)
            previous_to = to_sha
    if (
        any(
            type(seed) is not int
            for transition in transitions
            for seed in transition["consumed_novelty_seeds"]
        )
        or tuple(transitions[0]["consumed_novelty_seeds"]) != (
        DESIGN_NOVELTY_SEED,
        )
    ):
        raise ValueError("q97 design source transition consumes another seed")
    if tuple(transitions[1]["consumed_novelty_seeds"]) != (
        VALIDATION_NOVELTY_SEEDS
    ):
        raise ValueError(
            "q97 validation source transition consumes another seed set"
        )
    return transitions


def _admit_ordered_seed_ledger_transitions(
    *,
    name: str,
    origin: str,
    expected: str,
    current: str,
    path: Path,
) -> dict[str, Any] | None:
    """Admit an exact contiguous suffix of the two-step q97 ledger chain."""
    if name != "fit_v3_openset_staged.py":
        return None
    raw = tuple(ORDERED_STAGED_SOURCE_TRANSITIONS)
    start = next(
        (
            index
            for index, transition in enumerate(raw)
            if transition.get("origin") == origin
            and transition.get("from_sha256") == expected
        ),
        None,
    )
    if start is None:
        return None
    required_through = (
        1
        if start == 0 and raw[0].get("to_sha256") == current
        else 2
    )
    transitions = _validated_ordered_source_transitions(
        required_through_order=required_through
    )
    start = next(
        (
            index
            for index, transition in enumerate(transitions)
            if transition["origin"] == origin
            and transition["from_sha256"] == expected
        ),
        None,
    )
    if start is None:
        return None
    selected: list[dict[str, Any]] = []
    cursor = expected
    for transition in transitions[start:]:
        if transition["from_sha256"] != cursor:
            raise ValueError("q97 source transition chain is not contiguous")
        selected.append(transition)
        cursor = transition["to_sha256"]
        if cursor == current:
            break
    if cursor != current:
        return None
    excluded = tuple(selected[-1]["excluded_top_level_assignments"])
    normalized = _ledger_neutral_ast_sha256(path, excluded)
    if normalized != selected[-1]["normalized_ast_sha256"]:
        raise ValueError(
            "q97 seed-ledger transition changed executable source outside "
            "the declared top-level seed-ledger assignments"
        )
    consumed = tuple(
        seed
        for transition in transitions[: start + len(selected)]
        for seed in transition["consumed_novelty_seeds"]
    )
    final = selected[-1]
    if (
        tuple(staged.DEFAULT_VALIDATION_NOVELTY_SEEDS)
        != VALIDATION_NOVELTY_SEEDS
        or any(seed not in staged.SPENT_NOVELTY_SEEDS for seed in consumed)
        or staged.FIRST_CLEAN_NOVELTY_SEED
        != final["next_clean_novelty_seed"]
    ):
        raise ValueError(
            "q97 seed-ledger transition does not mark the required design/"
            "validation seeds spent and advance the clean-seed floor"
        )
    return {
        "admission": "exact_ordered_q97_seed_ledger_transition_chain",
        "origin": origin,
        "from_sha256": expected,
        "current_sha256": current,
        "normalized_ast_sha256": normalized,
        "excluded_top_level_assignments": list(excluded),
        "ordered_transition_ids": [
            transition["transition_id"] for transition in selected
        ],
        "transition_bindings": [
            {
                key: transition[key]
                for key in (
                    "order",
                    "transition_id",
                    "from_phase",
                    "to_phase",
                    "from_sha256",
                    "to_sha256",
                    "from_commit",
                    "evidence_commit",
                    "evidence_report_sha256",
                    "to_commit",
                    "full_index_diff_sha256",
                )
            }
            for transition in selected
        ],
        "consumed_novelty_seeds": list(consumed),
        "next_clean_novelty_seed": final["next_clean_novelty_seed"],
        "candidate_inference_behavior_changed": False,
    }


def _load_bundle(bundle_dir: Path) -> dict[str, Any]:
    """Load and strictly validate the v3 runtime bundle."""
    bundle = assemble.reject_sealed_path(Path(bundle_dir), "runtime bundle")
    manifest_path = bundle / BUNDLE_MANIFEST_NAME
    manifest = _read_candidate_json(manifest_path)
    if (
        manifest.get("kind") != BUNDLE_KIND
        or manifest.get("schema") != BUNDLE_SCHEMA
        or manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION
    ):
        raise ValueError(f"{bundle} is not a v3 runtime bundle")
    for key, expected in (
        ("development_only", True),
        ("release_evidence", False),
        ("sealed_release_data_used", 0),
        ("consumed_test_rows_used", 0),
    ):
        if manifest.get(key) != expected:
            raise ValueError(
                f"{bundle} records {key}={manifest.get(key)!r}, expected "
                f"{expected!r}"
            )
    assets = manifest.get("assets")
    if not isinstance(assets, Mapping) or not assets:
        raise ValueError(f"{bundle} manifest has no asset records")
    arrays: dict[str, np.ndarray] = {}
    for name, record in assets.items():
        path = bundle / str(name)
        _verify_file_record(path, record, name=f"bundle/{name}")
        if str(name).endswith(".npy"):
            arrays[str(name)] = np.load(path)
    verification = manifest.get("self_verification")
    worst_error = (
        verification.get("worst_max_abs_error")
        if isinstance(verification, Mapping)
        else None
    )
    if (
        not isinstance(verification, Mapping)
        or verification.get("closed_label_agreement") is not True
    ):
        raise ValueError(f"{bundle} does not carry a passing self-verification")
    try:
        worst_error = _require_finite_json_float(
            worst_error,
            field=f"{bundle}.self_verification.worst_max_abs_error",
        )
    except ValueError as exc:
        raise ValueError(
            f"{bundle} does not carry a passing self-verification"
        ) from exc
    if worst_error > BUNDLE_SELF_VERIFICATION_TOLERANCE:
        raise ValueError(f"{bundle} does not carry a passing self-verification")
    return {
        "directory": bundle,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "manifest_sha256": _sha256(manifest_path),
        "arrays": arrays,
    }


def _resolve_frozen_path(value: Any, *, field: str) -> Path:
    """Resolve an absolute or repository-relative frozen-artifact path."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO / path
    return path.resolve()


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} is not a lowercase SHA-256 digest")
    digest = value
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{field} is not a lowercase SHA-256 digest")
    return digest


def _verify_json_binding(
    record: Mapping[str, Any],
    *,
    field: str,
    expected_path: Path | None = None,
    expected_schema: str | None = None,
    expected_status: str | None = None,
) -> tuple[Path, str, dict[str, Any]]:
    path = _resolve_frozen_path(record.get("path"), field=f"{field}.path")
    if expected_path is not None and path != Path(expected_path).resolve():
        raise ValueError(
            f"{field}.path is {path}, expected the supplied frozen path "
            f"{Path(expected_path).resolve()}"
        )
    digest = _require_sha256(record.get("sha256"), field=f"{field}.sha256")
    if _sha256(path) != digest:
        raise ValueError(f"{field} SHA-256 does not match its current bytes")
    payload = _read_candidate_json(path)
    schema = record.get("schema")
    status = record.get("status")
    if expected_schema is not None and payload.get("schema") != expected_schema:
        raise ValueError(
            f"{field} JSON schema={payload.get('schema')!r}, expected "
            f"{expected_schema!r}"
        )
    if expected_status is not None and payload.get("status") != expected_status:
        raise ValueError(
            f"{field} JSON status={payload.get('status')!r}, expected "
            f"{expected_status!r}"
        )
    if schema is not None and payload.get("schema") != schema:
        raise ValueError(f"{field} schema differs from the referenced JSON")
    if status is not None and payload.get("status") != status:
        raise ValueError(f"{field} status differs from the referenced JSON")
    return path, digest, payload


def _package_record_path(
    value: Any,
    package_path: Path,
    *,
    field: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value).expanduser()
    if path.is_absolute():
        raise ValueError(f"{field} must be package-manifest-parent-relative")
    return package_path.parent / path


def _verify_package_file_record(
    record: Mapping[str, Any],
    package_path: Path,
    *,
    field: str,
) -> Path:
    path = _package_record_path(record.get("path"), package_path, field=field)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{field} must reference a regular non-symlink file")
    digest = _require_sha256(record.get("sha256"), field=f"{field}.sha256")
    size = record.get("bytes")
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
        or path.stat().st_size != size
    ):
        raise ValueError(f"{field} byte count differs from disk")
    if _sha256(path) != digest:
        raise ValueError(f"{field} SHA-256 differs from disk")
    return path.resolve()


def _validate_staging_package(
    *,
    package_path: Path,
    package_payload: Mapping[str, Any],
    verified_browser: Mapping[str, Mapping[str, Any]],
    binding_path: Path,
    binding_sha256: str,
    binding_payload: Mapping[str, Any],
    classifier_artifact: Any,
    rejector_artifact: Any,
    classifier_bundle: Mapping[str, Any],
    rejector_bundle: Mapping[str, Any],
    staged_report_sha256: str,
    staged_hashes: Mapping[str, str],
) -> None:
    expected_top = {
        "schema",
        "schema_version",
        "status",
        "candidate_id",
        "architecture",
        "assets",
        "roles",
        "openset_policy",
        "dual_binding",
        "external_evidence",
        "size_contract",
    }
    if set(package_payload) != expected_top:
        raise ValueError(
            "dual runtime package contains missing, extra or legacy fields"
        )
    if package_payload.get("candidate_id") != CANDIDATE_ID:
        raise ValueError("dual runtime package candidate_id differs")
    if package_payload.get("architecture") != {
        "execution_order": [
            "stage_one_noise_gate",
            "rejector_known_unknown",
            "classifier_known_label",
        ],
        "classifier_runs_only_after_rejector_acceptance": True,
        "public_known_label_from_classifier_only": True,
    }:
        raise ValueError("dual runtime package architecture differs")

    asset_names = {
        "classifier": CLASSIFIER_BROWSER_ASSET,
        "rejector": REJECTOR_BROWSER_ASSET,
        "openset_policy": OPENSET_BROWSER_ASSET,
        "dual_binding": DUAL_BINDING_ASSET,
    }
    assets = package_payload.get("assets")
    if not isinstance(assets, Mapping) or set(assets) != set(
        asset_names.values()
    ):
        raise ValueError(
            "dual runtime package deployable asset set is not the exact four "
            "dual-fusion files"
        )
    expected_asset_contract = {
        "classifier": (
            verified_browser["classifier"],
            BROWSER_FUSION_SCHEMA,
            BROWSER_FUSION_SCHEMA_VERSION,
            "accepted_known_classifier",
        ),
        "rejector": (
            verified_browser["rejector"],
            BROWSER_FUSION_SCHEMA,
            BROWSER_FUSION_SCHEMA_VERSION,
            "known_unknown_rejector",
        ),
        "openset_policy": (
            verified_browser["openset_policy"],
            BROWSER_OPENSET_SCHEMA,
            BROWSER_OPENSET_SCHEMA_VERSION,
            None,
        ),
        "dual_binding": (
            {
                "path": binding_path,
                "sha256": binding_sha256,
            },
            DUAL_BINDING_SCHEMA,
            DUAL_BINDING_SCHEMA_VERSION,
            None,
        ),
    }
    for role, (
        candidate_record,
        expected_schema,
        expected_version,
        expected_runtime_role,
    ) in expected_asset_contract.items():
        name = asset_names[role]
        record = assets[name]
        if not isinstance(record, Mapping):
            raise ValueError(f"dual runtime package asset {name} is invalid")
        expected_keys = {
            "path",
            "bytes",
            "sha256",
            "schema",
            "schema_version",
            "status",
        }
        if expected_runtime_role is not None:
            expected_keys.add("runtime_role")
        if set(record) != expected_keys or record.get("path") != name:
            raise ValueError(
                f"dual runtime package asset {name} contains aliases"
            )
        path = _verify_package_file_record(
            record, package_path, field=f"staging_package.assets.{name}"
        )
        if (
            path != Path(candidate_record["path"]).resolve()
            or record.get("sha256") != candidate_record["sha256"]
            or record.get("schema") != expected_schema
            or record.get("schema_version") != expected_version
            or record.get("status") != STAGING_STATUS
        ):
            raise ValueError(
                f"dual runtime package asset {name} differs from candidate"
            )
        if (
            expected_runtime_role is not None
            and record.get("runtime_role") != expected_runtime_role
        ):
            raise ValueError(
                f"dual runtime package asset {name} runtime_role differs"
            )

    roles = package_payload.get("roles")
    if not isinstance(roles, Mapping) or set(roles) != {
        "classifier",
        "rejector",
    }:
        raise ValueError("dual runtime package role set differs")
    for role, artifact, bundle, runtime_role, responsibility in (
        (
            "classifier",
            classifier_artifact,
            classifier_bundle,
            "accepted_known_classifier",
            "accepted_known_label_only",
        ),
        (
            "rejector",
            rejector_artifact,
            rejector_bundle,
            "known_unknown_rejector",
            "known_unknown_only",
        ),
    ):
        role_record = roles[role]
        asset_record = assets[asset_names[role]]
        if (
            not isinstance(role_record, Mapping)
            or set(role_record)
            != {
                "runtime_role",
                "responsibility",
                "asset",
                "source_bundle_manifest_sha256",
                "fusion_directory_sha256",
            }
            or role_record.get("runtime_role") != runtime_role
            or role_record.get("responsibility") != responsibility
            or role_record.get("asset") != asset_record
            or role_record.get("source_bundle_manifest_sha256")
            != bundle["manifest_sha256"]
            or role_record.get("fusion_directory_sha256")
            != artifact.directory_sha256
        ):
            raise ValueError(f"dual runtime package {role} role link differs")

    openset_record = package_payload.get("openset_policy")
    openset_asset = assets[OPENSET_BROWSER_ASSET]
    if (
        not isinstance(openset_record, Mapping)
        or set(openset_record)
        != {
            *openset_asset,
            "fitted_rejector_runtime_bundle_manifest_sha256",
            "staged_validation_report_sha256",
            "staged_artifacts_sha256",
        }
    ):
        raise ValueError("dual runtime package openset_policy is invalid")
    for key, value in openset_asset.items():
        if openset_record.get(key) != value:
            raise ValueError("dual runtime package openset asset link differs")
    if (
        openset_record.get(
            "fitted_rejector_runtime_bundle_manifest_sha256"
        )
        != rejector_bundle["manifest_sha256"]
        or openset_record.get("staged_validation_report_sha256")
        != staged_report_sha256
        or openset_record.get("staged_artifacts_sha256")
        != dict(staged_hashes)
    ):
        raise ValueError("dual runtime package openset evidence link differs")
    dual_binding_record = package_payload.get("dual_binding")
    if (
        not isinstance(dual_binding_record, Mapping)
        or set(dual_binding_record) != set(assets[DUAL_BINDING_ASSET])
        or dual_binding_record != assets[DUAL_BINDING_ASSET]
    ):
        raise ValueError("dual runtime package binding record differs")

    external = package_payload.get("external_evidence")
    expected_external = {
        "rejector_export_manifest",
        "classifier_export_manifest",
        "openset_export_manifest",
        "rejector_probe",
        "classifier_probe",
        "parity",
    }
    if not isinstance(external, Mapping) or set(external) != expected_external:
        raise ValueError("dual runtime package external evidence set differs")
    for name, record in external.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"staging package external_evidence.{name} invalid")
        expected_record_keys = {"path", "bytes", "sha256"}
        if name in {"rejector_export_manifest", "classifier_export_manifest"}:
            expected_record_keys.update(
                {"schema", "schema_version", "runtime_role"}
            )
        elif name == "openset_export_manifest":
            expected_record_keys.update({"schema", "schema_version", "status"})
        elif name == "parity":
            expected_record_keys.update(
                {"schema", "schema_version", "status", "packaged"}
            )
        if set(record) != expected_record_keys:
            raise ValueError(
                f"staging package external_evidence.{name} contains aliases"
            )
        path = _verify_package_file_record(
            record,
            package_path,
            field=f"staging_package.external_evidence.{name}",
        )
        if name in {
            "rejector_export_manifest",
            "classifier_export_manifest",
            "openset_export_manifest",
            "parity",
        }:
            payload = _read_candidate_json(path)
            for key in ("schema", "schema_version"):
                if payload.get(key) != record.get(key):
                    raise ValueError(
                        f"staging package external {name} {key} differs"
                    )
            if "status" in record and payload.get("status") != record.get(
                "status"
            ):
                raise ValueError(
                    f"staging package external {name} status differs"
                )
        if name == "rejector_export_manifest":
            expected = (
                FUSION_EXPORT_MANIFEST_SCHEMA,
                FUSION_EXPORT_MANIFEST_SCHEMA_VERSION,
                "known_unknown_rejector",
            )
            if (
                record.get("schema"),
                record.get("schema_version"),
                record.get("runtime_role"),
            ) != expected:
                raise ValueError("rejector export-manifest contract differs")
        elif name == "classifier_export_manifest":
            expected = (
                FUSION_EXPORT_MANIFEST_SCHEMA,
                FUSION_EXPORT_MANIFEST_SCHEMA_VERSION,
                "accepted_known_classifier",
            )
            if (
                record.get("schema"),
                record.get("schema_version"),
                record.get("runtime_role"),
            ) != expected:
                raise ValueError("classifier export-manifest contract differs")
        elif name == "openset_export_manifest":
            if (
                record.get("schema") != OPENSET_EXPORT_MANIFEST_SCHEMA
                or record.get("schema_version")
                != OPENSET_EXPORT_MANIFEST_SCHEMA_VERSION
                or record.get("status") != STAGING_STATUS
            ):
                raise ValueError("openset export-manifest contract differs")
        elif name == "parity":
            if (
                record.get("schema") != PARITY_SCHEMA
                or record.get("schema_version") != PARITY_SCHEMA_VERSION
                or record.get("status") != STAGING_STATUS
                or record.get("packaged") is not False
            ):
                raise ValueError("parity external-evidence contract differs")
    size_contract = package_payload.get("size_contract")
    maximum = 25 * 1024 * 1024
    if (
        size_contract
        != {
            "maximum_file_bytes_exclusive": maximum,
            "all_deployable_files_below_limit": True,
        }
        or any(int(record["bytes"]) >= maximum for record in assets.values())
    ):
        raise ValueError("dual runtime package size contract differs")
    if binding_payload.get("candidate_id") != package_payload.get("candidate_id"):
        raise ValueError("package and binding candidate_id differ")


def _verify_fusion_binding(
    record: Mapping[str, Any],
    artifact: Any,
    *,
    field: str,
) -> None:
    path = _resolve_frozen_path(record.get("directory"), field=f"{field}.directory")
    if path != artifact.directory:
        raise ValueError(
            f"{field} directory is {path}, not the supplied {artifact.directory}"
        )
    digest = _require_sha256(
        record.get("directory_sha256"), field=f"{field}.directory_sha256"
    )
    if digest != artifact.directory_sha256:
        raise ValueError(f"{field} directory SHA-256 does not match")
    file_sha256 = record.get("file_sha256")
    if not isinstance(file_sha256, Mapping) or dict(file_sha256) != dict(
        artifact.file_sha256
    ):
        raise ValueError(f"{field} file SHA-256 map does not match")


def _verify_bundle_binding(
    record: Mapping[str, Any],
    bundle: Mapping[str, Any],
    *,
    field: str,
) -> None:
    directory = _resolve_frozen_path(
        record.get("directory"), field=f"{field}.directory"
    )
    if directory != bundle["directory"]:
        raise ValueError(
            f"{field} directory is {directory}, not {bundle['directory']}"
        )
    manifest_path = _resolve_frozen_path(
        record.get("manifest_path"), field=f"{field}.manifest_path"
    )
    if manifest_path != bundle["manifest_path"]:
        raise ValueError(f"{field} manifest_path does not match")
    digest = _require_sha256(
        record.get("manifest_sha256"), field=f"{field}.manifest_sha256"
    )
    if digest != bundle["manifest_sha256"]:
        raise ValueError(f"{field} manifest SHA-256 does not match")


def _browser_state_vector(
    state: Mapping[str, torch.Tensor],
    name: str,
    shape: tuple[int, ...],
) -> list[float]:
    """Serialize one frozen float32 tensor exactly as the browser exporter."""
    value = state.get(name)
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != shape
        or value.dtype != torch.float32
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(
            f"runtime bundle state tensor {name} is not finite float32 {shape}"
        )
    return value.reshape(-1).tolist()


def _browser_linear_payload(
    state: Mapping[str, torch.Tensor],
    prefix: str,
    input_width: int,
    output_width: int,
) -> dict[str, Any]:
    return {
        "in": input_width,
        "out": output_width,
        "weight": _browser_state_vector(
            state,
            f"{prefix}.weight",
            (output_width, input_width),
        ),
        "bias": _browser_state_vector(
            state,
            f"{prefix}.bias",
            (output_width,),
        ),
    }


def _browser_block_indices(
    state: Mapping[str, torch.Tensor],
    pattern: str,
    *,
    field: str,
) -> list[int]:
    expression = re.compile(pattern)
    indices = sorted(
        {
            int(match.group(1))
            for name in state
            if (match := expression.fullmatch(name)) is not None
        }
    )
    if not indices or indices != list(range(len(indices))):
        raise ValueError(f"{field} indices are missing or non-contiguous")
    return indices


def _expected_browser_fusion_payload(
    bundle: Mapping[str, Any],
    *,
    expected_runtime_role: str,
) -> dict[str, Any]:
    """Regenerate the complete browser asset from the hash-bound bundle.

    This is deliberately an evaluator-local oracle instead of an import of
    ``export_v3_browser_weights``: importing a new source during the sealed
    run would silently expand the precommitted 49-file execution surface.
    Every emitted tensor is reconstructed from the already SHA-verified state
    dict and arrays, so package hashes cannot substitute for semantic
    admission of frontend, branch, fusion or classifier geometry.
    """
    manifest = bundle["manifest"]
    if manifest.get("runtime_role") != expected_runtime_role:
        raise ValueError("runtime bundle role differs from browser role")
    architecture = manifest.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError("runtime bundle has no browser architecture")
    real_config = architecture.get("real")
    complex_config = architecture.get("complex")
    if not isinstance(real_config, Mapping) or not isinstance(
        complex_config, Mapping
    ):
        raise ValueError("runtime bundle branch architecture is incomplete")
    state = torch.load(
        bundle["directory"] / "fusion_state_dict.pt",
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(state, Mapping):
        raise ValueError("runtime bundle fusion state is not a mapping")

    real_blocks: list[dict[str, Any]] = []
    for index in _browser_block_indices(
        state,
        r"real_branch\.patch_encoder\.blocks\.(\d+)\.conv\.weight",
        field="browser real blocks",
    ):
        prefix = f"real_branch.patch_encoder.blocks.{index}"
        conv = state.get(f"{prefix}.conv.weight")
        if not isinstance(conv, torch.Tensor) or conv.ndim != 3:
            raise ValueError(f"{prefix}.conv.weight is not rank three")
        output_channels, input_channels, kernel = (
            int(value) for value in conv.shape
        )
        real_blocks.append(
            {
                "in_channels": input_channels,
                "out_channels": output_channels,
                "kernel": kernel,
                "stride": 2,
                "padding": kernel // 2,
                "conv_weight": _browser_state_vector(
                    state,
                    f"{prefix}.conv.weight",
                    (output_channels, input_channels, kernel),
                ),
                "batch_norm": {
                    "eps": 1e-5,
                    **{
                        name: _browser_state_vector(
                            state,
                            f"{prefix}.bn.{name}",
                            (output_channels,),
                        )
                        for name in (
                            "weight",
                            "bias",
                            "running_mean",
                            "running_var",
                        )
                    },
                },
            }
        )
    real_patch_dim = int(real_config["patch_dim"])
    real_hidden = int(real_config["hidden"])
    real_embed_dim = int(real_config["embed_dim"])
    real_features = int(real_config["n_features"])
    real_set_width = real_patch_dim * (
        2 if real_config["set_pool"] == "mean_std" else 1
    )
    real = {
        "config": dict(real_config),
        "blocks": real_blocks,
        "patch_projection": _browser_linear_payload(
            state,
            "real_branch.patch_projection.0",
            2 * int(real_blocks[-1]["out_channels"]),
            real_patch_dim,
        ),
        "fc1": _browser_linear_payload(
            state,
            "real_branch.fc1",
            real_set_width + real_features,
            real_hidden,
        ),
        "fc2": _browser_linear_payload(
            state,
            "real_branch.fc2",
            real_hidden,
            real_embed_dim,
        ),
    }

    complex_stages: list[dict[str, Any]] = []
    for index in _browser_block_indices(
        state,
        (
            r"complex_branch\.patch_encoder\.stages\.(\d+)"
            r"\.conv\.conv_re\.weight"
        ),
        field="browser complex stages",
    ):
        prefix = f"complex_branch.patch_encoder.stages.{index}"
        weight = state.get(f"{prefix}.conv.conv_re.weight")
        if not isinstance(weight, torch.Tensor) or weight.ndim != 3:
            raise ValueError(f"{prefix}.conv.conv_re.weight is not rank three")
        output_channels, input_channels, kernel = (
            int(value) for value in weight.shape
        )
        complex_stages.append(
            {
                "in_channels": input_channels,
                "out_channels": output_channels,
                "kernel": kernel,
                "padding": (kernel - 1) // 2,
                "weight_real": _browser_state_vector(
                    state,
                    f"{prefix}.conv.conv_re.weight",
                    (output_channels, input_channels, kernel),
                ),
                "weight_imag": _browser_state_vector(
                    state,
                    f"{prefix}.conv.conv_im.weight",
                    (output_channels, input_channels, kernel),
                ),
                "threshold_raw": _browser_state_vector(
                    state,
                    f"{prefix}.act.thresh_raw",
                    (output_channels,),
                ),
            }
        )
    complex_patch_dim = int(complex_config["patch_dim"])
    complex_hidden = int(complex_config["hidden"])
    complex_embed_dim = int(complex_config["embed_dim"])
    complex_features = int(complex_config["n_features"])
    complex_set_width = complex_patch_dim * (
        2 if complex_config["set_pool"] == "mean_std" else 1
    )
    complex_lags = [1, 2, 4]
    complex = {
        "config": dict(complex_config),
        "stages": complex_stages,
        "mag_norm_eps": 1e-6,
        "modrelu_magnitude_eps": 1e-12,
        "lowpass": [0.25, 0.5, 0.25],
        "lags": complex_lags,
        "patch_projection": _browser_linear_payload(
            state,
            "complex_branch.patch_projection.0",
            int(complex_stages[-1]["out_channels"])
            * (2 + 2 * len(complex_lags)),
            complex_patch_dim,
        ),
        "fc1": _browser_linear_payload(
            state,
            "complex_branch.fc1",
            complex_set_width + complex_features,
            complex_hidden,
        ),
        "fc2": _browser_linear_payload(
            state,
            "complex_branch.fc2",
            complex_hidden,
            complex_embed_dim,
        ),
    }
    if (
        real_config.get("patch_length") != complex_config.get("patch_length")
        or real_config.get("patch_count") != complex_config.get("patch_count")
        or real_embed_dim != complex_embed_dim
        or real_features != complex_features
    ):
        raise ValueError("runtime bundle branch geometry differs")

    arrays = bundle["arrays"]
    classes = list(manifest["classification"]["classes"])
    frontend = manifest["frontend"]
    fusion = manifest["fusion"]
    standardization = manifest["feature_standardization"]
    rejection = manifest["rejection"]
    provenance = manifest["provenance"]
    asset_sha256 = {
        name: record["sha256"]
        for name, record in manifest["assets"].items()
    }
    return {
        "schema": BROWSER_FUSION_SCHEMA,
        "schema_version": BROWSER_FUSION_SCHEMA_VERSION,
        "status": STAGING_STATUS,
        "development_only": True,
        "kind": manifest["kind"],
        "runtime_role": expected_runtime_role,
        "packed_length": int(frontend["packed_length"]),
        "parameter_count": int(manifest["parameter_count"]),
        "frontend": {
            "version": frontend["version"],
            "estimator_version": frontend["estimator_version"],
            "patch_length": int(frontend["patch_length"]),
            "patch_count": int(frontend["patch_count"]),
            "target_frac": float(frontend["target_frac"]),
            "feature_count": int(frontend["feature_count"]),
            "uses_frequency_transform": False,
            "source_sha256": dict(frontend["source_sha256"]),
        },
        "real": real,
        "complex": complex,
        "fusion": {
            "real_center": _browser_state_vector(
                state, "real_center", (real_embed_dim,)
            ),
            "complex_center": _browser_state_vector(
                state, "complex_center", (complex_embed_dim,)
            ),
            "alpha_real": float(state["alpha_real"]),
            "alpha_complex": float(state["alpha_complex"]),
            "weight_real": float(state["weight_real"]),
            "eps": float(state["eps"]),
        },
        "embedding_normalize_eps": 1e-12,
        "feature_standardization": {
            "mean": np.asarray(arrays["feature_mean.npy"]).tolist(),
            "std": np.asarray(arrays["feature_std.npy"]).tolist(),
            "epsilon": float(standardization["epsilon"]),
            "rule": standardization["rule"],
        },
        "classification": {
            "classes": classes,
            "prototypes": np.asarray(
                arrays["fusion_prototypes.npy"]
            ).tolist(),
            "distance": manifest["classification"]["distance"],
            "label": manifest["classification"]["label"],
            "embedding": manifest["classification"]["embedding"],
        },
        "rejection": {
            "state": rejection["state"],
            "runtime_behaviour": rejection["runtime_behaviour"],
            "external_staged_policy_required_for_abstention": True,
        },
        "not_compatible_with_schema_ids": list(
            manifest["not_compatible_with_schema_ids"]
        ),
        "provenance": {
            "source_bundle_schema": manifest["schema"],
            "source_bundle_manifest_sha256": bundle["manifest_sha256"],
            "source_bundle_assets_sha256": asset_sha256,
            "assembly_seed": int(provenance["assembly_seed"]),
            "exporter": "export_v3_browser_weights.py",
            "batch_norm_folded": False,
        },
        "release_blockers": list(manifest["release_blockers"]),
        "release_evidence": False,
    }


def _expected_browser_openset_payload(
    observed: Mapping[str, Any],
    *,
    stage_one: Any,
    prefilter_set_sha256: str,
    components: Sequence[tuple[str, float, Any]],
    policy: Any,
    composite: Any,
    frontend: Mapping[str, Any],
) -> dict[str, Any]:
    """Regenerate all runtime policy semantics from frozen Python state."""
    provenance = observed.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("browser open-set provenance must be an object")
    expected = openset_export.openset_weights_payload(
        stage_one.models,
        prefilter_set_sha256,
        components,
        policy,
        composite,
        {
            "version": str(frontend["version"]),
            "patch_length": int(frontend["patch_length"]),
            "patch_count": int(frontend["patch_count"]),
            "target_frac": float(frontend["target_frac"]),
            "packed_length": int(frontend["packed_length"]),
            "uses_frequency_transform": False,
        },
        provenance,
    )
    # Normalize NumPy arrays/scalars through the exporter's actual JSON path.
    return json.loads(openset_export.json_bytes(expected, compact=True))


def _fusion_geometry_signature(artifact: Any) -> dict[str, Any]:
    """The dimensions that must agree across the two independently fit nets."""
    metrics = artifact.metrics
    architecture = metrics.get("architecture", {})
    source_config = (
        metrics.get("source_index_contract", {})
        .get("contract", {})
        .get("config", {})
    )
    signature: dict[str, Any] = {
        "source_config": {
            key: source_config.get(key)
            for key in ("patch_length", "patch_count", "target_frac")
        },
        "prototype_shape": tuple(int(v) for v in artifact.prototypes.shape),
        "feature_count": int(len(artifact.feature_mean)),
    }
    for branch in ("real", "complex"):
        config = architecture.get(branch, {})
        signature[branch] = {
            key: config.get(key)
            for key in (
                "encoder",
                "patch_length",
                "patch_count",
                "patch_dim",
                "hidden",
                "embed_dim",
                "n_features",
                "set_pool",
            )
        }
    signature["fusion_embed_dim"] = architecture.get("embed_dim")
    return signature


def _validate_frozen_q97_contract(
    frozen_contract: Mapping[str, Any],
    *,
    canonical_release_candidate: bool,
) -> dict[str, Any]:
    """Validate the complete policy-v4 contract frozen before validation."""
    expected_keys = {
        "schema",
        "status",
        "candidate_id",
        "architecture",
        "q97_policy",
        "q97_design_evidence",
        "source_transition_intent",
        "classifier_fusion_8k_regularized",
        "rejector_fusion_4k",
        "stage_one_noise_prefilter",
    }
    if set(frozen_contract) != expected_keys:
        raise ValueError(
            "pre-validation q97 candidate contract contains missing, extra "
            "or legacy fields"
        )
    try:
        _require_exact_json_value(
            frozen_contract.get("q97_policy"),
            q97_policy_contract_block(),
            field="frozen_contract.q97_policy",
        )
    except ValueError:
        raise ValueError("pre-validation q97 policy identity/hygiene differs")
    if (
        frozen_contract.get("source_transition_intent")
        != source_transition_intent_block()
    ):
        raise ValueError(
            "pre-validation q97 source transition order/seed intent differs"
        )
    architecture = frozen_contract.get("architecture")
    try:
        _require_exact_json_value(
            architecture,
            EXPECTED_FROZEN_ARCHITECTURE,
            field="frozen_contract.architecture",
        )
    except ValueError as exc:
        raise ValueError("pre-validation q97 architecture contract differs")
    design_record = frozen_contract.get("q97_design_evidence")
    expected_design_keys = {
        "path",
        "sha256",
        "role",
        "status",
        "novelty_seed",
        "source_sha256",
        "commit",
    }
    if (
        not isinstance(design_record, Mapping)
        or set(design_record) != expected_design_keys
        or design_record.get("role") != "design"
        or design_record.get("status") != "design_selection_pass"
        or design_record.get("novelty_seed") != DESIGN_NOVELTY_SEED
    ):
        raise ValueError("pre-validation q97 design evidence record differs")
    design_path, design_sha256, design_report = _verify_json_binding(
        design_record,
        field="frozen_contract.q97_design_evidence",
        expected_status="design_selection_pass",
    )
    first_transition = ORDERED_STAGED_SOURCE_TRANSITIONS[0]
    if (
        design_report.get("role") != "design"
        or design_report.get("gates_are_evidence") is not False
        or design_report.get("all_pass") is not True
        or design_report.get("release_seed_20260736_used") is not False
        or design_report.get("source_sha256", {}).get(
            "fit_v3_openset_staged.py"
        )
        != design_record.get("source_sha256")
    ):
        raise ValueError("bound q97 design report is not the passing design draw")
    design_seeds = design_report.get("seeds")
    if (
        not isinstance(design_seeds, Mapping)
        or design_seeds.get("novelty_seeds") != [DESIGN_NOVELTY_SEED]
        or design_seeds.get("design_novelty_seed") != DESIGN_NOVELTY_SEED
        or design_seeds.get("release_seed_not_spent")
        != EXPECTED_RELEASE_SEED
    ):
        raise ValueError("bound q97 design report seed hygiene differs")
    design_architecture = design_report.get("architecture")
    if (
        not isinstance(design_architecture, Mapping)
        or design_architecture.get("schema")
        != EXPECTED_STAGED_POLICY_SCHEMA
        or design_architecture.get("kind") != EXPECTED_STAGED_POLICY_KIND
        or design_architecture.get("staged_policy_version")
        != EXPECTED_STAGED_POLICY_VERSION
    ):
        raise ValueError("bound design report uses another staged policy")
    design_composite = design_report.get("composite")
    for key, expected in {
        "policy_version": EXPECTED_STAGED_POLICY_VERSION,
        "schema": EXPECTED_STAGED_POLICY_SCHEMA,
        "kind": EXPECTED_STAGED_POLICY_KIND,
        "threshold_quantile": EXPECTED_COMPOSITE_THRESHOLD_QUANTILE,
        **EXPECTED_POLICY_HYGIENE,
    }.items():
        try:
            _require_exact_json_value(
                design_composite.get(key)
                if isinstance(design_composite, Mapping)
                else None,
                expected,
                field=f"design.composite.{key}",
            )
        except ValueError:
            raise ValueError(
                f"bound q97 design report policy hygiene {key} differs"
            )
    if canonical_release_candidate:
        if (
            design_sha256 != first_transition["evidence_report_sha256"]
            or design_record.get("source_sha256")
            != first_transition["from_sha256"]
            or design_record.get("commit")
            != first_transition["evidence_commit"]
        ):
            raise ValueError(
                "canonical q97 contract does not bind the exact frozen "
                "design55 report/source/commit"
            )
    return {
        "path": design_path,
        "sha256": design_sha256,
        "report": design_report,
    }


def _validate_candidate_manifest(
    *,
    candidate_manifest_path: Path,
    classifier_bundle: Mapping[str, Any],
    rejector_bundle: Mapping[str, Any],
    classifier_artifact: Any,
    rejector_artifact: Any,
    staged_metrics_path: Path,
    staged_hashes: Mapping[str, str],
    prefilter_path: Path,
    prefilter_set_sha256: str,
    stage_one_lengths: Sequence[int],
    stage_one: Any,
    stage_two_components: Sequence[tuple[str, float, Any]],
    stage_two_policy: Any,
    composite_policy: Any,
    frontend: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the final release-candidate manifest and its complete chain."""
    manifest_path = Path(candidate_manifest_path).expanduser().resolve()
    canonical_release_candidate = ENFORCE_CANONICAL_Q97_EVIDENCE
    manifest = _read_candidate_json(manifest_path)
    expected_manifest_keys = {
        "schema",
        "status",
        "candidate_id",
        "validation_evidence",
        "classifier",
        "rejector",
        "staged_validation",
        "stage_one_prefilter",
        "browser_assets",
        "staging_package_manifest",
        "dual_binding",
    }
    if set(manifest) != expected_manifest_keys:
        raise ValueError(
            "candidate manifest contains missing, extra or legacy fields"
        )
    if manifest.get("schema") != CANDIDATE_MANIFEST_SCHEMA:
        raise ValueError(
            "candidate manifest uses an old or unknown schema; the dual-fusion "
            "release evaluator fails closed"
        )
    if manifest.get("status") != "release_candidate_frozen":
        raise ValueError("candidate manifest is not frozen for release")
    if manifest.get("candidate_id") != CANDIDATE_ID:
        raise ValueError("candidate manifest identifies another candidate")

    for role, role_record, bundle, artifact, expected_role in (
        (
            "classifier",
            manifest.get("classifier"),
            classifier_bundle,
            classifier_artifact,
            "known_class_label",
        ),
        (
            "rejector",
            manifest.get("rejector"),
            rejector_bundle,
            rejector_artifact,
            "known_unknown_decision",
        ),
    ):
        if not isinstance(role_record, Mapping):
            raise ValueError(f"candidate manifest has no {role} role")
        if role_record.get("role") != expected_role:
            raise ValueError(
                f"candidate manifest {role}.role must be {expected_role!r}"
            )
        fusion_record = role_record.get("fusion")
        bundle_record = role_record.get("runtime_bundle")
        if not isinstance(fusion_record, Mapping) or not isinstance(
            bundle_record, Mapping
        ):
            raise ValueError(f"candidate manifest {role} bindings are incomplete")
        _verify_fusion_binding(
            fusion_record, artifact, field=f"candidate.{role}.fusion"
        )
        _verify_bundle_binding(
            bundle_record, bundle, field=f"candidate.{role}.runtime_bundle"
        )

    validation_record = manifest.get("validation_evidence")
    if not isinstance(validation_record, Mapping):
        raise ValueError("candidate manifest has no validation_evidence binding")
    evidence_path, evidence_sha256, evidence = _verify_json_binding(
        validation_record,
        field="candidate.validation_evidence",
        expected_schema=CANDIDATE_EVIDENCE_SCHEMA,
    )
    if (
        evidence.get("status") != "development_openset_pass"
        or evidence.get("candidate_id") != CANDIDATE_ID
    ):
        raise ValueError("validation evidence is not the passing dual candidate")
    expected_evidence_keys = {
        "schema",
        "status",
        "candidate_id",
        "candidate_contract",
        "validation",
        "validated_rejector",
        "validation_locked_policy_artifacts",
        "candidate_contract_clarification",
        "source_transition_evidence",
    }
    if set(evidence) != expected_evidence_keys:
        raise ValueError(
            "q97 validation evidence contains missing, extra or legacy fields"
        )
    try:
        _require_exact_json_value(
            evidence.get("source_transition_evidence"),
            ordered_source_transition_contract_block(),
            field="validation_evidence.source_transition_evidence",
        )
    except ValueError:
        raise ValueError(
            "q97 validation evidence source transition endpoints/order differ"
        )

    contract_record = evidence.get("candidate_contract")
    if not isinstance(contract_record, Mapping):
        raise ValueError("validation evidence has no pre-validation contract")
    frozen_path, frozen_sha256, frozen_contract = _verify_json_binding(
        contract_record,
        field="validation_evidence.candidate_contract",
        expected_schema=CANDIDATE_CONTRACT_SCHEMA,
    )
    if (
        frozen_contract.get("status") != "frozen_before_validation"
        or frozen_contract.get("candidate_id") != CANDIDATE_ID
        or contract_record.get("committed_before_validation") is not True
    ):
        raise ValueError("pre-validation candidate contract is not frozen")
    if (
        manifest_path == DEFAULT_CANDIDATE_MANIFEST.resolve()
        and frozen_path != DEFAULT_CANDIDATE_CONTRACT.resolve()
    ):
        raise ValueError(
            "canonical release manifest must bind "
            f"{DEFAULT_CANDIDATE_CONTRACT}"
        )
    design_binding = _validate_frozen_q97_contract(
        frozen_contract,
        canonical_release_candidate=canonical_release_candidate,
    )

    for field, artifact in (
        ("classifier_fusion_8k_regularized", classifier_artifact),
        ("rejector_fusion_4k", rejector_artifact),
    ):
        record = frozen_contract.get(field)
        if not isinstance(record, Mapping):
            raise ValueError(f"pre-validation contract has no {field}")
        _verify_fusion_binding(record, artifact, field=f"frozen_contract.{field}")

    validation = evidence.get("validation", {})
    for key, expected in (
        ("role", "validate"),
        ("gates_are_evidence", True),
        ("all_pass", True),
        ("sealed_release_data_used", 0),
        ("consumed_test_rows_used", 0),
        ("release_seed_20260736_used", False),
    ):
        try:
            _require_exact_json_value(
                validation.get(key),
                expected,
                field=f"validation.{key}",
            )
        except ValueError:
            raise ValueError(
                f"validation evidence records {key}={validation.get(key)!r}, "
                f"expected {expected!r}"
            )
    try:
        _require_exact_json_value(
            validation.get("novelty_seeds_consumed_once"),
            list(VALIDATION_NOVELTY_SEEDS),
            field="validation.novelty_seeds_consumed_once",
        )
    except ValueError:
        raise ValueError("validation evidence binds another novelty seed pair")
    report_path = _resolve_frozen_path(
        validation.get("report"), field="validation.report"
    )
    if report_path != staged_metrics_path:
        raise ValueError("validation evidence binds another staged report")
    report_sha256 = _require_sha256(
        validation.get("report_sha256"), field="validation.report_sha256"
    )
    if _sha256(staged_metrics_path) != report_sha256:
        raise ValueError("validation report SHA-256 does not match")

    validated_rejector = evidence.get("validated_rejector", {})
    if (
        validated_rejector.get("fusion_directory_sha256")
        != rejector_artifact.directory_sha256
    ):
        raise ValueError("validation evidence binds another rejector fusion")
    if (
        validated_rejector.get("canonical_prefilter_set_sha256")
        != prefilter_set_sha256
    ):
        raise ValueError("validation evidence binds another prefilter set")
    expected_bundle_hashes = {
        str(int(length)): noise_prefilter.bundle_sha256(
            prefilter_path / f"N{int(length)}"
        )
        for length in stage_one_lengths
    }
    if validated_rejector.get("prefilter_bundle_sha256") != expected_bundle_hashes:
        raise ValueError("validation evidence prefilter bundle hashes differ")

    locked = evidence.get("validation_locked_policy_artifacts", {})
    locked_dir = _resolve_frozen_path(
        locked.get("directory"), field="validation_locked_policy_artifacts.directory"
    )
    if locked_dir != staged_metrics_path.parent:
        raise ValueError("validation evidence binds another staged directory")
    for name, digest in staged_hashes.items():
        if locked.get(name) != digest:
            raise ValueError(
                f"validation evidence does not bind the loaded staged {name}"
            )
    if locked.get("novelty_rows_used_to_fit_rank_or_threshold") != 0:
        raise ValueError("validation novelty was used to fit the staged policy")

    stage_one_contract = frozen_contract.get("stage_one_noise_prefilter", {})
    if _resolve_frozen_path(
        stage_one_contract.get("directory"),
        field="frozen_contract.stage_one_noise_prefilter.directory",
    ) != prefilter_path:
        raise ValueError("pre-validation contract binds another prefilter path")
    clarification = evidence.get("candidate_contract_clarification", {})
    if (
        clarification.get("recorded_value")
        != stage_one_contract.get("directory_sha256")
        or clarification.get("canonical_behavioral_hash")
        != prefilter_set_sha256
        or clarification.get("changes_candidate_behavior") is not False
    ):
        raise ValueError("prefilter hash clarification is missing or inconsistent")

    staged_record = manifest.get("staged_validation")
    if not isinstance(staged_record, Mapping):
        raise ValueError("candidate manifest has no staged_validation binding")
    staged_dir = _resolve_frozen_path(
        staged_record.get("directory"), field="candidate.staged_validation.directory"
    )
    if staged_dir != staged_metrics_path.parent:
        raise ValueError("candidate manifest binds another staged directory")
    staged_report_path = _resolve_frozen_path(
        staged_record.get("report_path"),
        field="candidate.staged_validation.report_path",
    )
    if staged_report_path != staged_metrics_path:
        raise ValueError("candidate manifest binds another staged report path")
    if _require_sha256(
        staged_record.get("report_sha256"),
        field="candidate.staged_validation.report_sha256",
    ) != report_sha256:
        raise ValueError("candidate manifest staged report hash differs")
    if staged_record.get("artifact_sha256") != dict(staged_hashes):
        raise ValueError("candidate manifest staged artifact hashes differ")

    prefilter_record = manifest.get("stage_one_prefilter")
    if not isinstance(prefilter_record, Mapping):
        raise ValueError("candidate manifest has no stage_one_prefilter binding")
    if _resolve_frozen_path(
        prefilter_record.get("directory"),
        field="candidate.stage_one_prefilter.directory",
    ) != prefilter_path:
        raise ValueError("candidate manifest binds another prefilter directory")
    if prefilter_record.get("set_sha256") != prefilter_set_sha256:
        raise ValueError("candidate manifest prefilter set hash differs")
    if prefilter_record.get("bundle_sha256") != expected_bundle_hashes:
        raise ValueError("candidate manifest prefilter bundle hashes differ")

    browser_assets = manifest.get("browser_assets")
    if not isinstance(browser_assets, Mapping) or set(browser_assets) != {
        "classifier",
        "rejector",
        "openset_policy",
    }:
        raise ValueError("candidate manifest browser_assets roles are incomplete")
    verified_browser: dict[str, Any] = {}
    browser_contract = {
        "classifier": (
            BROWSER_FUSION_SCHEMA,
            BROWSER_FUSION_SCHEMA_VERSION,
            "accepted_known_classifier",
        ),
        "rejector": (
            BROWSER_FUSION_SCHEMA,
            BROWSER_FUSION_SCHEMA_VERSION,
            "known_unknown_rejector",
        ),
        "openset_policy": (
            BROWSER_OPENSET_SCHEMA,
            BROWSER_OPENSET_SCHEMA_VERSION,
            None,
        ),
    }
    for role, (
        expected_browser_schema,
        expected_browser_version,
        expected_runtime_role,
    ) in browser_contract.items():
        record = browser_assets[role]
        if (
            not isinstance(record, Mapping)
            or set(record) != {"path", "sha256", "schema", "status"}
        ):
            raise ValueError(f"candidate browser_assets.{role} is invalid")
        if not isinstance(record.get("schema"), str) or not isinstance(
            record.get("status"), str
        ):
            raise ValueError(
                f"candidate browser_assets.{role} must bind schema and status"
            )
        if record.get("status") != STAGING_STATUS:
            raise ValueError(
                f"candidate browser_assets.{role} is not staging_not_release"
            )
        path, digest, payload = _verify_json_binding(
            record, field=f"candidate.browser_assets.{role}"
        )
        if (
            payload.get("schema") != expected_browser_schema
            or payload.get("schema_version") != expected_browser_version
            or payload.get("status") != STAGING_STATUS
        ):
            raise ValueError(
                f"candidate browser_assets.{role} schema/version/status differs"
            )
        if (
            expected_runtime_role is not None
            and payload.get("runtime_role") != expected_runtime_role
        ):
            raise ValueError(
                f"candidate browser_assets.{role} runtime_role differs"
            )
        if role in {"classifier", "rejector"}:
            expected_payload = _expected_browser_fusion_payload(
                (
                    classifier_bundle
                    if role == "classifier"
                    else rejector_bundle
                ),
                expected_runtime_role=str(expected_runtime_role),
            )
            try:
                _require_exact_json_value(
                    payload,
                    expected_payload,
                    field=f"candidate.browser_assets.{role}.payload",
                )
            except ValueError as exc:
                raise ValueError(
                    f"candidate browser_assets.{role} is not the exact "
                    "runtime-bundle export"
                ) from exc
        if role == "openset_policy":
            browser_contract_block = payload.get("contract")
            for key, expected in (
                ("additive_only", False),
                ("changes_closed_label", True),
                ("gates_before_classification", True),
            ):
                if (
                    not isinstance(browser_contract_block, Mapping)
                    or browser_contract_block.get(key) != expected
                ):
                    raise ValueError(
                        f"browser q97 contract {key} differs from Python"
                    )
            browser_stage_two = payload.get("stage_two")
            if (
                not isinstance(browser_stage_two, Mapping)
                or browser_stage_two.get("kind") != FROZEN_POLICY_KIND
            ):
                raise ValueError("browser q97 stage-two policy kind differs")
            browser_stage_two_policy = browser_stage_two.get("policy")
            try:
                browser_stage_two_quantile = _require_finite_json_float(
                    browser_stage_two_policy.get("threshold_quantile")
                    if isinstance(browser_stage_two_policy, Mapping)
                    else None,
                    field="browser.stage_two.policy.threshold_quantile",
                )
            except ValueError:
                browser_stage_two_quantile = None
            if browser_stage_two_quantile != float(FROZEN_THRESHOLD_QUANTILE):
                raise ValueError(
                    "browser q97 stage-two threshold quantile differs"
                )
            browser_composite = payload.get("composite")
            browser_composite_expected = {
                "schema": EXPECTED_STAGED_POLICY_SCHEMA,
                "kind": EXPECTED_STAGED_POLICY_KIND,
                "policy_version": EXPECTED_STAGED_POLICY_VERSION,
                "survivor_score": staged.COMPOSITE_SURVIVOR_SCORE,
                "threshold_quantile": (
                    EXPECTED_COMPOSITE_THRESHOLD_QUANTILE
                ),
                **EXPECTED_SERIALIZED_POLICY_HYGIENE,
            }
            for key, expected in browser_composite_expected.items():
                if (
                    not isinstance(browser_composite, Mapping)
                    or browser_composite.get(key) != expected
                ):
                    raise ValueError(
                        f"browser q97 composite policy/hygiene {key} differs"
                    )
            expected_payload = _expected_browser_openset_payload(
                payload,
                stage_one=stage_one,
                prefilter_set_sha256=prefilter_set_sha256,
                components=stage_two_components,
                policy=stage_two_policy,
                composite=composite_policy,
                frontend=frontend,
            )
            try:
                _require_exact_json_value(
                    payload,
                    expected_payload,
                    field="candidate.browser_assets.openset_policy.payload",
                )
            except ValueError as exc:
                raise ValueError(
                    "candidate browser_assets.openset_policy is not the exact "
                    "frozen staged-policy export"
                ) from exc
        verified_browser[role] = {
            "path": path,
            "sha256": digest,
            "schema": payload.get("schema"),
            "status": payload.get("status"),
        }
    package_record = manifest.get("staging_package_manifest")
    if (
        not isinstance(package_record, Mapping)
        or set(package_record) != {"path", "sha256", "schema", "status"}
    ):
        raise ValueError("candidate manifest has no staging_package_manifest")
    if not isinstance(package_record.get("schema"), str) or not isinstance(
        package_record.get("status"), str
    ):
        raise ValueError(
            "candidate staging_package_manifest must bind schema and status"
        )
    if (
        package_record.get("schema") != STAGING_PACKAGE_SCHEMA
        or package_record.get("status") != STAGING_STATUS
    ):
        raise ValueError(
            "candidate staging package is not the frozen dual-runtime staging "
            "schema/status"
        )
    package_path, package_sha256, package_payload = _verify_json_binding(
        package_record, field="candidate.staging_package_manifest"
    )
    if (
        package_payload.get("schema_version") != STAGING_PACKAGE_SCHEMA_VERSION
    ):
        raise ValueError("candidate staging package schema_version is not 1")
    binding_record = manifest.get("dual_binding")
    if (
        not isinstance(binding_record, Mapping)
        or set(binding_record) != {"path", "sha256", "schema"}
    ):
        raise ValueError("candidate manifest has no dual_binding")
    if not isinstance(binding_record.get("schema"), str):
        raise ValueError("candidate dual_binding must bind its schema")
    if binding_record.get("schema") != DUAL_BINDING_SCHEMA:
        raise ValueError("candidate dual_binding uses an old or unknown schema")
    binding_path, binding_sha256, binding_payload = _verify_json_binding(
        binding_record, field="candidate.dual_binding"
    )
    if (
        binding_payload.get("schema_version") != DUAL_BINDING_SCHEMA_VERSION
        or binding_payload.get("status") != STAGING_STATUS
    ):
        raise ValueError(
            "candidate dual_binding must be schema version 1 and "
            "staging_not_release"
        )
    expected_binding_keys = {
        "schema",
        "schema_version",
        "status",
        "candidate_id",
        "frontend",
        "execution_order",
        "roles",
        "openset_policy",
        "validation",
        "fail_closed",
    }
    if set(binding_payload) != expected_binding_keys:
        raise ValueError(
            "candidate dual_binding contains missing, extra or legacy fields"
        )
    if (
        binding_payload.get("candidate_id") != CANDIDATE_ID
        or binding_payload.get("execution_order")
        != [
            "stage_one_noise_gate",
            "rejector_known_unknown",
            "classifier_known_label",
        ]
    ):
        raise ValueError("candidate dual_binding identity/execution order differs")
    binding_roles = binding_payload.get("roles")
    if not isinstance(binding_roles, Mapping) or set(binding_roles) != {
        "classifier",
        "rejector",
    }:
        raise ValueError("candidate dual_binding role set differs")
    for role, artifact, bundle, runtime_role, responsibility in (
        (
            "classifier",
            classifier_artifact,
            classifier_bundle,
            "accepted_known_classifier",
            "accepted_known_label_only",
        ),
        (
            "rejector",
            rejector_artifact,
            rejector_bundle,
            "known_unknown_rejector",
            "known_unknown_only",
        ),
    ):
        record = binding_roles[role]
        browser = verified_browser[role]
        if (
            not isinstance(record, Mapping)
            or record.get("asset") != browser["path"].name
            or record.get("asset_sha256") != browser["sha256"]
            or record.get("fusion_directory_sha256")
            != artifact.directory_sha256
            or record.get("runtime_bundle_manifest_sha256")
            != bundle["manifest_sha256"]
            or record.get("runtime_role") != runtime_role
            or record.get("responsibility") != responsibility
        ):
            raise ValueError(f"candidate dual_binding {role} role differs")
    binding_openset = binding_payload.get("openset_policy")
    openset_browser = verified_browser["openset_policy"]
    if (
        not isinstance(binding_openset, Mapping)
        or binding_openset.get("asset") != openset_browser["path"].name
        or binding_openset.get("asset_sha256") != openset_browser["sha256"]
        or binding_openset.get("rejector_asset_sha256")
        != verified_browser["rejector"]["sha256"]
        or binding_openset.get(
            "fitted_rejector_runtime_bundle_manifest_sha256"
        )
        != rejector_bundle["manifest_sha256"]
        or binding_openset.get("staged_validation_report_sha256")
        != report_sha256
        or binding_openset.get("staged_artifacts_sha256")
        != dict(staged_hashes)
    ):
        raise ValueError("candidate dual_binding openset policy differs")
    binding_validation = binding_payload.get("validation")
    if (
        not isinstance(binding_validation, Mapping)
        or binding_validation.get("report_sha256") != report_sha256
        or binding_validation.get("role") != "validate"
        or binding_validation.get("status") != "development_openset_pass"
        or binding_validation.get("novelty_seeds")
        != list(validation.get("novelty_seeds_consumed_once", ()))
        or binding_validation.get("design_novelty_seed")
        != DESIGN_NOVELTY_SEED
        or binding_validation.get("release_seed_not_spent")
        != EXPECTED_RELEASE_SEED
    ):
        raise ValueError("candidate dual_binding validation record differs")
    expected_fail_closed = {
        "role_assets_bound_by_sha256": True,
        "distinct_role_assets": True,
        "role_asset_sha256_must_differ": True,
        "classifier_runs_only_after_rejector_acceptance": True,
        "public_known_label_from_classifier_only": True,
    }
    if binding_payload.get("fail_closed") != expected_fail_closed:
        raise ValueError("candidate dual_binding fail_closed contract differs")
    if (
        verified_browser["classifier"]["sha256"]
        == verified_browser["rejector"]["sha256"]
    ):
        raise ValueError("candidate browser role assets may not alias")
    _validate_staging_package(
        package_path=package_path,
        package_payload=package_payload,
        verified_browser=verified_browser,
        binding_path=binding_path,
        binding_sha256=binding_sha256,
        binding_payload=binding_payload,
        classifier_artifact=classifier_artifact,
        rejector_artifact=rejector_artifact,
        classifier_bundle=classifier_bundle,
        rejector_bundle=rejector_bundle,
        staged_report_sha256=report_sha256,
        staged_hashes=staged_hashes,
    )

    if classifier_artifact.directory_sha256 == rejector_artifact.directory_sha256:
        raise ValueError("dual candidate may not alias its two fusion roles")
    if _fusion_geometry_signature(classifier_artifact) != _fusion_geometry_signature(
        rejector_artifact
    ):
        raise ValueError("classifier and rejector frontend geometry differs")

    return {
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256(manifest_path),
        "manifest": manifest,
        "validation_evidence_path": evidence_path,
        "validation_evidence_sha256": evidence_sha256,
        "validation_evidence": evidence,
        "frozen_contract_path": frozen_path,
        "frozen_contract_sha256": frozen_sha256,
        "frozen_contract": frozen_contract,
        "q97_design_evidence": design_binding,
        "validation_report_sha256": report_sha256,
        "prefilter_bundle_sha256": expected_bundle_hashes,
        "browser_assets": verified_browser,
        "staging_package_manifest": {
            "path": package_path,
            "sha256": package_sha256,
            "schema": package_payload.get("schema"),
            "status": package_payload.get("status"),
        },
        "dual_binding": {
            "path": binding_path,
            "sha256": binding_sha256,
            "schema": binding_payload.get("schema"),
        },
    }


def load_candidate(
    *,
    candidate_manifest_path: Path,
    classifier_bundle_dir: Path,
    rejector_bundle_dir: Path,
    classifier_fusion_dir: Path,
    rejector_fusion_dir: Path,
    staged_dir: Path,
    prefilter_dir: Path,
    device: torch.device,
) -> V3Candidate:
    """Load every frozen component and prove the dual-role binding chain."""
    _assert_expected_policy_module()
    execution_contract = _admit_transitive_execution_contract(
        canonical_release_candidate=ENFORCE_CANONICAL_Q97_EVIDENCE,
        device=device,
    )
    classifier_bundle = _load_bundle(classifier_bundle_dir)
    rejector_bundle = _load_bundle(rejector_bundle_dir)
    if classifier_bundle["manifest"].get("runtime_role") != (
        "accepted_known_classifier"
    ):
        raise ValueError(
            "classifier bundle runtime_role is not accepted_known_classifier"
        )
    if rejector_bundle["manifest"].get("runtime_role") != (
        "known_unknown_rejector"
    ):
        raise ValueError("rejector bundle runtime_role is not known_unknown_rejector")
    bundle = rejector_bundle
    manifest = bundle["manifest"]

    # --- independent fusion artifacts and each bundle's binding to it ------
    classifier_artifact = openset_base.load_fusion_artifact(
        Path(classifier_fusion_dir)
    )
    rejector_artifact = openset_base.load_fusion_artifact(
        Path(rejector_fusion_dir)
    )
    fusion_artifact = rejector_artifact
    fusion_dir = rejector_fusion_dir
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("bundle manifest has no provenance block")
    recorded_fusion = (REPO / str(provenance.get("source_fusion_artifact"))).resolve()
    if recorded_fusion != fusion_artifact.directory:
        raise ValueError(
            f"bundle was exported from {recorded_fusion}, not from the given "
            f"fusion directory {fusion_artifact.directory}"
        )
    if provenance.get("source_dev_metrics_sha256") != fusion_artifact.file_sha256[
        "dev_metrics.json"
    ]:
        raise ValueError(
            "bundle provenance.source_dev_metrics_sha256 does not match the "
            "fusion artifact's dev_metrics.json bytes"
        )
    for asset_name, fusion_value in (
        ("real_center.npy", fusion_artifact.real_center),
        ("complex_center.npy", fusion_artifact.complex_center),
        ("fusion_prototypes.npy", fusion_artifact.prototypes),
        ("feature_mean.npy", fusion_artifact.feature_mean),
        ("feature_std.npy", fusion_artifact.feature_std),
    ):
        if not np.array_equal(
            np.asarray(bundle["arrays"][asset_name], dtype=np.float32),
            np.asarray(fusion_value, dtype=np.float32),
        ):
            raise ValueError(
                f"runtime bundle {asset_name} differs from the fusion artifact"
            )

    # --- the torch fusion module, rebuilt from the frozen state dict -------
    loaded = measure.load_fusion_artifact(Path(fusion_dir))
    fusion_module = loaded["fusion"].to(device).eval()
    state_name = str(fusion_artifact.metrics["artifacts"]["state_dict"])
    if _sha256(bundle["directory"] / "fusion_state_dict.pt") != (
        fusion_artifact.file_sha256[state_name]
    ):
        raise ValueError(
            "bundle fusion_state_dict.pt differs from the fusion artifact's"
        )
    rejector_fusion_module = fusion_module

    classifier_manifest = classifier_bundle["manifest"]
    classifier_provenance = classifier_manifest.get("provenance")
    if not isinstance(classifier_provenance, Mapping):
        raise ValueError("classifier bundle manifest has no provenance block")
    classifier_recorded_fusion = _resolve_frozen_path(
        classifier_provenance.get("source_fusion_artifact"),
        field="classifier bundle provenance.source_fusion_artifact",
    )
    if classifier_recorded_fusion != classifier_artifact.directory:
        raise ValueError(
            "classifier runtime bundle was exported from another fusion"
        )
    if (
        classifier_provenance.get("source_dev_metrics_sha256")
        != classifier_artifact.file_sha256["dev_metrics.json"]
    ):
        raise ValueError(
            "classifier bundle provenance.source_dev_metrics_sha256 does not "
            "match the classifier fusion artifact"
        )
    for asset_name, fusion_value in (
        ("real_center.npy", classifier_artifact.real_center),
        ("complex_center.npy", classifier_artifact.complex_center),
        ("fusion_prototypes.npy", classifier_artifact.prototypes),
        ("feature_mean.npy", classifier_artifact.feature_mean),
        ("feature_std.npy", classifier_artifact.feature_std),
    ):
        if not np.array_equal(
            np.asarray(classifier_bundle["arrays"][asset_name], dtype=np.float32),
            np.asarray(fusion_value, dtype=np.float32),
        ):
            raise ValueError(
                f"classifier runtime bundle {asset_name} differs from its "
                "fusion artifact"
            )
    classifier_loaded = measure.load_fusion_artifact(
        Path(classifier_fusion_dir)
    )
    classifier_fusion_module = classifier_loaded["fusion"].to(device).eval()
    classifier_state_name = str(
        classifier_artifact.metrics["artifacts"]["state_dict"]
    )
    if _sha256(
        classifier_bundle["directory"] / "fusion_state_dict.pt"
    ) != classifier_artifact.file_sha256[classifier_state_name]:
        raise ValueError(
            "classifier bundle fusion_state_dict.pt differs from its fusion "
            "artifact"
        )

    # --- the frozen staged stage-2 state -----------------------------------
    components, policy, staged_hashes = openset_export.load_stage_two_state(
        Path(staged_dir)
    )
    staged_metrics_path = (
        assemble.reject_sealed_path(Path(staged_dir), "staged artifact")
        / "openset_metrics.json"
    )
    staged_metrics = _read_candidate_json(staged_metrics_path)
    for key, expected in (
        ("status", "development_openset_pass"),
        ("role", "validate"),
        ("gates_are_evidence", True),
        ("all_pass", True),
        ("development_only", True),
        ("release_evidence", False),
        ("sealed_release_data_used", 0),
        ("consumed_test_rows_used", 0),
        ("release_seed_20260736_used", False),
        ("additive_only", False),
        ("changes_closed_label", True),
        ("gates_before_classification", True),
        ("closed_label_can_be_gated", True),
    ):
        try:
            _require_exact_json_value(
                staged_metrics.get(key),
                expected,
                field=f"staged_metrics.{key}",
            )
        except ValueError:
            raise ValueError(
                f"staged artifact records {key}={staged_metrics.get(key)!r}, "
                f"expected {expected!r}; only a passing validation run may "
                "supply the frozen stage-2 state"
            )
    # The staged POLICY VERSION must be the one this evaluator implements: a
    # version-1 artifact (survivor score = stage-2 rank alone) must not be
    # scored with composite semantics, nor the reverse.
    architecture = staged_metrics.get("architecture", {})
    for key, expected in (
        ("kind", EXPECTED_STAGED_POLICY_KIND),
        ("schema", EXPECTED_STAGED_POLICY_SCHEMA),
        ("staged_policy_version", EXPECTED_STAGED_POLICY_VERSION),
        ("survivor_score", staged.COMPOSITE_SURVIVOR_SCORE),
    ):
        try:
            _require_exact_json_value(
                architecture.get(key),
                expected,
                field=f"staged_metrics.architecture.{key}",
            )
        except ValueError:
            raise ValueError(
                "staged policy version mismatch: the staged artifact records "
                f"architecture.{key}={architecture.get(key)!r}, but this "
                f"evaluator implements {expected!r}.  A staged artifact may "
                "only be evaluated under the policy version that validated it"
            )
    # --- the frozen composite survivor policy -------------------------------
    composite_path = staged_metrics_path.parent / staged.COMPOSITE_POLICY_FILENAME
    if not composite_path.is_file():
        raise ValueError(
            f"staged artifact carries no {staged.COMPOSITE_POLICY_FILENAME}; "
            "a composite-policy (schema-4/q97) validation artifact is required"
        )
    composite = staged.load_composite_policy(
        composite_path,
        expected_stage_two_threshold=policy.threshold,
        expected_policy_version=EXPECTED_STAGED_POLICY_VERSION,
    )
    staged_hashes = dict(staged_hashes)
    staged_hashes[staged.COMPOSITE_POLICY_FILENAME] = _sha256(composite_path)
    if staged_metrics.get("artifacts") != staged_hashes:
        raise ValueError(
            "staged artifact npz bytes differ from the hashes its own report "
            "recorded"
        )
    recorded = staged_metrics.get("fusion", {})
    if recorded.get("directory_sha256") != fusion_artifact.directory_sha256:
        raise ValueError(
            "the staged stage-2 state was fit against a different fusion "
            f"({recorded.get('directory_sha256')!r}) than the candidate "
            f"({fusion_artifact.directory_sha256!r})"
        )
    staged_stage_two_threshold = _require_finite_json_float(
        staged_metrics["stage_two"]["threshold"],
        field="staged_metrics.stage_two.threshold",
    )
    if staged_stage_two_threshold != float(policy.threshold):
        raise ValueError(
            "loaded stage-2 threshold differs from the staged artifact record"
        )
    staged_composite_threshold = _require_finite_json_float(
        staged_metrics["composite"]["threshold"],
        field="staged_metrics.composite.threshold",
    )
    if staged_composite_threshold != float(composite.threshold):
        raise ValueError(
            "loaded composite threshold differs from the staged artifact "
            "record"
        )
    composite_report = staged_metrics.get("composite")
    if not isinstance(composite_report, Mapping):
        raise ValueError("staged report has no q97 composite policy block")
    composite_expected = {
        "policy_version": EXPECTED_STAGED_POLICY_VERSION,
        "schema": EXPECTED_STAGED_POLICY_SCHEMA,
        "kind": EXPECTED_STAGED_POLICY_KIND,
        "survivor_score": staged.COMPOSITE_SURVIVOR_SCORE,
        "threshold_quantile": EXPECTED_COMPOSITE_THRESHOLD_QUANTILE,
        **EXPECTED_POLICY_HYGIENE,
    }
    for key, expected in composite_expected.items():
        try:
            _require_exact_json_value(
                composite_report.get(key),
                expected,
                field=f"staged_metrics.composite.{key}",
            )
        except ValueError:
            raise ValueError(
                f"staged q97 policy hygiene {key}="
                f"{composite_report.get(key)!r}, expected {expected!r}"
            )
    if composite.policy_spec != staged.STAGED_POLICY_SPEC:
        raise ValueError("loaded composite is not the exact current q97 policy")
    seeds = staged_metrics.get("seeds")
    if not isinstance(seeds, Mapping):
        raise ValueError("staged report has no seed hygiene block")
    for key, expected in (
        ("novelty_seeds", list(VALIDATION_NOVELTY_SEEDS)),
        ("design_novelty_seed", DESIGN_NOVELTY_SEED),
        ("default_validation_novelty_seeds", list(VALIDATION_NOVELTY_SEEDS)),
        ("release_seed_not_spent", EXPECTED_RELEASE_SEED),
        ("novelty_rows_used_to_choose_the_stage_one_operating_point", 0),
        ("novelty_rows_used_to_choose_the_composite_threshold", 0),
    ):
        try:
            _require_exact_json_value(
                seeds.get(key),
                expected,
                field=f"staged_metrics.seeds.{key}",
            )
        except ValueError:
            raise ValueError(
                f"staged seed hygiene {key}={seeds.get(key)!r}, expected "
                f"{expected!r}"
            )

    # --- the stage-1 prefilter set ------------------------------------------
    prefilter_module = staged.load_prefilter_module()
    posedegen = staged.load_posedegen_module()
    prefilter_path = assemble.reject_sealed_path(
        Path(prefilter_dir), "prefilter set"
    )
    set_sha256 = str(noise_prefilter.prefilter_set_sha256(prefilter_path))
    if staged_metrics.get("stage_one", {}).get("set_sha256") != set_sha256:
        raise ValueError(
            "prefilter set bytes differ from the set the staged validation "
            "run recorded; the stage-1 gate is not the validated one"
        )
    declared_lengths = tuple(
        int(length)
        for length in staged_metrics.get("stage_one", {}).get(
            "capture_lengths", ()
        )
    )
    if not declared_lengths:
        raise ValueError("staged artifact declares no stage-1 capture lengths")
    stage_one = staged.load_stage_one(
        prefilter_module,
        posedegen,
        prefilter_path,
        required_lengths=declared_lengths,
    )
    if tuple(stage_one.lengths) != declared_lengths:
        raise ValueError(
            f"prefilter set covers {list(stage_one.lengths)}, but the staged "
            f"validation declared {list(declared_lengths)}"
        )

    # --- the bundle's rejection contract must describe this exact policy ----
    rejection = manifest.get("rejection", {})
    required_contract = rejection.get("required_contract", {})
    for key, expected in (
        ("policy_schema", FROZEN_POLICY_SCHEMA),
        ("policy_kind", FROZEN_POLICY_KIND),
        ("geometry_feature", FROZEN_GEOMETRY_FEATURE),
        ("geometry_weight", FROZEN_GEOMETRY_WEIGHT),
        ("v2_weight", FROZEN_V2_WEIGHT),
        ("threshold_quantile", FROZEN_THRESHOLD_QUANTILE),
        ("module", "v3_time_domain_openset"),
    ):
        try:
            _require_exact_json_value(
                required_contract.get(key),
                expected,
                field=f"bundle.rejection.required_contract.{key}",
            )
        except ValueError:
            raise ValueError(
                f"bundle rejection.required_contract[{key!r}] is "
                f"{required_contract.get(key)!r}, expected {expected!r}"
            )

    # --- source drift --------------------------------------------------------
    staged_recorded = staged_metrics.get("source_sha256", {})
    checked, source_transitions = _verify_source_contract(
        staged_recorded,
        STAGED_SOURCE_HARD_CONTRACT,
        origin="the staged validation artifact",
    )
    rejector_checked, rejector_transitions = _verify_source_contract(
        manifest.get("provenance", {}).get("assembly_source_sha256", {}),
        BUNDLE_ASSEMBLY_SOURCE_HARD_CONTRACT,
        origin="the runtime bundle assembly record",
    )
    classifier_checked, classifier_transitions = _verify_source_contract(
        classifier_manifest.get("provenance", {}).get(
            "assembly_source_sha256", {}
        ),
        BUNDLE_ASSEMBLY_SOURCE_HARD_CONTRACT,
        origin="the classifier runtime bundle assembly record",
    )
    checked.update(rejector_checked)
    checked.update(classifier_checked)
    source_transitions.update(rejector_transitions)
    source_transitions.update(classifier_transitions)
    frontend_recorded = manifest.get("frontend", {}).get("source_sha256", {})
    classifier_frontend_recorded = classifier_manifest.get("frontend", {}).get(
        "source_sha256", {}
    )
    if (
        not isinstance(frontend_recorded, Mapping)
        or not isinstance(classifier_frontend_recorded, Mapping)
        or dict(classifier_frontend_recorded) != dict(frontend_recorded)
    ):
        raise ValueError(
            "classifier and rejector runtime bundle frontend source bindings "
            "differ"
        )
    for name, expected in dict(frontend_recorded).items():
        short = str(name).split("/")[-1]
        current = _sha256(_source_path(short))
        if current != expected:
            raise ValueError(f"frontend source drift: {name}")
        checked[short] = current
    recorded_only = {
        name: {
            "recorded_by_staged_artifact": staged_recorded.get(name),
            "current": _sha256(_source_path(name)),
            "enforced": False,
            "reason": (
                "training-loop source; imported but never executed for data "
                "by this evaluator, and legitimately changed after the "
                "candidate was frozen (HANDOFF 24.1)"
            ),
        }
        for name in STAGED_SOURCE_RECORDED_ONLY
    }
    evaluator_chain = {
        name: _sha256(_source_path(name))
        for name in (
            "evaluate_invariant_release_suite.py",
            "export_v3_openset_browser_assets.py",
            "measure_v3_remaining_gates.py",
            "preprocess.py",
            "invariant_patch_preprocess.py",
            "run_invariant_cnn_dev.py",
        )
    }

    # --- geometry and classes -----------------------------------------------
    frontend = manifest["frontend"]
    metadata = td_preprocess.preprocess_metadata()
    for key in ("patch_length", "patch_count", "target_frac"):
        if metadata[key] != frontend[key]:
            raise ValueError(
                f"current frontend {key}={metadata[key]!r} differs from the "
                f"bundle's frozen {frontend[key]!r}"
            )
    if metadata["version"] != frontend["version"]:
        raise ValueError("current frontend version differs from the bundle's")
    classifier_frontend = classifier_manifest.get("frontend", {})
    for key in ("version", "patch_length", "patch_count", "target_frac"):
        if classifier_frontend.get(key) != frontend.get(key):
            raise ValueError(
                "classifier and rejector runtime bundle frontend geometry differs"
            )
    classes = tuple(str(name) for name in manifest["classification"]["classes"])
    if not classes or list(classes) != sorted(classes):
        raise ValueError("bundle classes must be a sorted non-empty list")
    classifier_classes = tuple(
        str(name)
        for name in classifier_manifest.get("classification", {}).get(
            "classes", ()
        )
    )
    if classifier_classes != classes:
        raise ValueError("classifier and rejector runtime bundle classes differ")
    for role, artifact in (
        ("classifier", classifier_artifact),
        ("rejector", rejector_artifact),
    ):
        if artifact.prototypes.shape[0] != len(classes):
            raise ValueError(
                f"{role} fusion prototype count differs from the class contract"
            )

    binding = _validate_candidate_manifest(
        candidate_manifest_path=Path(candidate_manifest_path),
        classifier_bundle=classifier_bundle,
        rejector_bundle=rejector_bundle,
        classifier_artifact=classifier_artifact,
        rejector_artifact=rejector_artifact,
        staged_metrics_path=staged_metrics_path,
        staged_hashes=staged_hashes,
        prefilter_path=prefilter_path,
        prefilter_set_sha256=set_sha256,
        stage_one_lengths=stage_one.lengths,
        stage_one=stage_one,
        stage_two_components=components,
        stage_two_policy=policy,
        composite_policy=composite,
        frontend=frontend,
    )
    if ENFORCE_CANONICAL_Q97_EVIDENCE:
        (
            design_transition,
            validation_transition,
        ) = _validated_ordered_source_transitions()
        design_source = (
            binding["q97_design_evidence"]["report"]
            .get("source_sha256", {})
            .get("fit_v3_openset_staged.py")
        )
        validation_source = staged_recorded.get(
            "fit_v3_openset_staged.py"
        )
        current_staged_source = _sha256(
            _source_path("fit_v3_openset_staged.py")
        )
        if (
            design_source != design_transition["from_sha256"]
            or validation_source != design_transition["to_sha256"]
            or validation_source != validation_transition["from_sha256"]
            or current_staged_source != validation_transition["to_sha256"]
        ):
            raise ValueError(
                "q97 ordered source transitions do not bind the exact "
                "design55, validation53/54 and postvalidation source bytes"
            )
        validation_excluded = tuple(
            validation_transition["excluded_top_level_assignments"]
        )
        if _ledger_neutral_ast_sha256(
            _source_path("fit_v3_openset_staged.py"),
            validation_excluded,
        ) != validation_transition["normalized_ast_sha256"]:
            raise ValueError(
                "q97 validation ledger transition changes executable source"
            )
        all_consumed = (
            DESIGN_NOVELTY_SEED,
            *VALIDATION_NOVELTY_SEEDS,
        )
        if (
            any(
                seed not in staged.SPENT_NOVELTY_SEEDS
                for seed in all_consumed
            )
            or staged.FIRST_CLEAN_NOVELTY_SEED != 20260956
        ):
            raise ValueError(
                "q97 postvalidation ledger does not mark design55 and "
                "validation53/54 spent or advance FIRST_CLEAN to 20260956"
            )

    rejector = openset_base.Rejector(
        nets={
            "real": rejector_fusion_module.real_branch,
            "complex": rejector_fusion_module.complex_branch,
        },
        real_center=fusion_artifact.real_center,
        complex_center=fusion_artifact.complex_center,
        weight_real=fusion_artifact.weight_real,
        prototypes=fusion_artifact.prototypes,
        lof_components=list(components),
        policy=policy,
    )

    return V3Candidate(
        candidate_manifest_path=binding["manifest_path"],
        candidate_manifest_sha256=binding["manifest_sha256"],
        candidate_manifest=binding["manifest"],
        validation_evidence_path=binding["validation_evidence_path"],
        validation_evidence_sha256=binding["validation_evidence_sha256"],
        validation_evidence=binding["validation_evidence"],
        frozen_contract_path=binding["frozen_contract_path"],
        frozen_contract_sha256=binding["frozen_contract_sha256"],
        frozen_contract=binding["frozen_contract"],
        classifier_bundle_dir=classifier_bundle["directory"],
        classifier_bundle_manifest_path=classifier_bundle["manifest_path"],
        classifier_bundle_manifest_sha256=classifier_bundle["manifest_sha256"],
        classifier_bundle_manifest=classifier_manifest,
        classifier_bundle_arrays=classifier_bundle["arrays"],
        rejector_bundle_dir=bundle["directory"],
        rejector_bundle_manifest_path=bundle["manifest_path"],
        rejector_bundle_manifest_sha256=bundle["manifest_sha256"],
        rejector_bundle_manifest=manifest,
        rejector_bundle_arrays=bundle["arrays"],
        classifier_fusion_dir=classifier_artifact.directory,
        classifier_fusion_artifact=classifier_artifact,
        classifier_fusion_module=classifier_fusion_module,
        rejector_fusion_dir=fusion_artifact.directory,
        rejector_fusion_artifact=fusion_artifact,
        rejector_fusion_module=rejector_fusion_module,
        staged_dir=staged_metrics_path.parent,
        staged_metrics_path=staged_metrics_path,
        staged_metrics_sha256=binding["validation_report_sha256"],
        staged_metrics=staged_metrics,
        staged_hashes=staged_hashes,
        prefilter_dir=prefilter_path,
        prefilter_set_sha256=set_sha256,
        stage_one=stage_one,
        stage_one_lengths=tuple(stage_one.lengths),
        posedegen=posedegen,
        prefilter_module=prefilter_module,
        rejector=rejector,
        composite=composite,
        classes=classes,
        patch_length=int(frontend["patch_length"]),
        patch_count=int(frontend["patch_count"]),
        target_frac=float(frontend["target_frac"]),
        classifier_feature_mean=np.asarray(
            classifier_bundle["arrays"]["feature_mean.npy"], dtype=np.float32
        ),
        classifier_feature_std=np.asarray(
            classifier_bundle["arrays"]["feature_std.npy"], dtype=np.float32
        ),
        rejector_feature_mean=np.asarray(
            bundle["arrays"]["feature_mean.npy"], dtype=np.float32
        ),
        rejector_feature_std=np.asarray(
            bundle["arrays"]["feature_std.npy"], dtype=np.float32
        ),
        threshold=float(composite.threshold),
        stage_two_threshold=float(policy.threshold),
        source_report={
            "enforced": checked,
            "recorded_only": recorded_only,
            "post_validation_ledger_transitions": source_transitions,
            "evaluator_chain": evaluator_chain,
            "transitive_execution_contract": execution_contract,
        },
    )


def candidate_provenance(candidate: V3Candidate) -> dict[str, Any]:
    prefilter_bundle_sha256 = {
        str(int(length)): noise_prefilter.bundle_sha256(
            candidate.prefilter_dir / f"N{int(length)}"
        )
        for length in candidate.stage_one_lengths
    }
    browser_assets = {
        role: dict(record)
        for role, record in candidate.candidate_manifest["browser_assets"].items()
    }
    return {
        "candidate_contract": {
            "path": str(candidate.candidate_manifest_path),
            "sha256": candidate.candidate_manifest_sha256,
            "schema": CANDIDATE_MANIFEST_SCHEMA,
            "candidate_id": CANDIDATE_ID,
            "status": candidate.candidate_manifest["status"],
        },
        "validation_evidence": {
            "path": str(candidate.validation_evidence_path),
            "sha256": candidate.validation_evidence_sha256,
            "schema": CANDIDATE_EVIDENCE_SCHEMA,
            "status": candidate.validation_evidence["status"],
        },
        "frozen_prevalidation_contract": {
            "path": str(candidate.frozen_contract_path),
            "sha256": candidate.frozen_contract_sha256,
            "schema": CANDIDATE_CONTRACT_SCHEMA,
            "status": candidate.frozen_contract["status"],
        },
        "q97_design_evidence": dict(
            candidate.frozen_contract["q97_design_evidence"]
        ),
        "ordered_source_transition_evidence": copy.deepcopy(
            candidate.validation_evidence["source_transition_evidence"]
        ),
        "classifier_runtime_bundle": {
            "directory": str(candidate.classifier_bundle_dir),
            "manifest_path": str(candidate.classifier_bundle_manifest_path),
            "manifest_sha256": candidate.classifier_bundle_manifest_sha256,
            "schema": BUNDLE_SCHEMA,
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "kind": BUNDLE_KIND,
            "assets": {
                name: dict(record)
                for name, record in candidate.classifier_bundle_manifest[
                    "assets"
                ].items()
            },
        },
        "rejector_runtime_bundle": {
            "directory": str(candidate.rejector_bundle_dir),
            "manifest_path": str(candidate.rejector_bundle_manifest_path),
            "manifest_sha256": candidate.rejector_bundle_manifest_sha256,
            "schema": BUNDLE_SCHEMA,
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "kind": BUNDLE_KIND,
            "assets": {
                name: dict(record)
                for name, record in candidate.rejector_bundle_manifest[
                    "assets"
                ].items()
            },
        },
        "classifier_fusion": {
            "directory": str(candidate.classifier_fusion_dir),
            "directory_sha256": (
                candidate.classifier_fusion_artifact.directory_sha256
            ),
            "file_sha256": dict(
                candidate.classifier_fusion_artifact.file_sha256
            ),
            "seed": int(candidate.classifier_fusion_artifact.seed),
            "role": "known_class_label",
        },
        "rejector_fusion": {
            "directory": str(candidate.rejector_fusion_dir),
            "directory_sha256": candidate.rejector_fusion_artifact.directory_sha256,
            "file_sha256": dict(candidate.rejector_fusion_artifact.file_sha256),
            "seed": int(candidate.rejector_fusion_artifact.seed),
            "role": "known_unknown_decision",
        },
        "staged_validation": {
            "directory": str(candidate.staged_dir),
            "report_path": str(candidate.staged_metrics_path),
            "report_sha256": candidate.staged_metrics_sha256,
            "status": candidate.staged_metrics["status"],
            "novelty_seeds": list(
                candidate.staged_metrics["seeds"]["novelty_seeds"]
            ),
            "artifact_sha256": dict(candidate.staged_hashes),
            "policy_version": EXPECTED_STAGED_POLICY_VERSION,
            "staged_threshold": candidate.threshold,
            "stage_two_threshold": candidate.stage_two_threshold,
            "composite": candidate.composite.provenance(),
        },
        "stage_one_prefilter": {
            "directory": str(candidate.prefilter_dir),
            "set_sha256": candidate.prefilter_set_sha256,
            "bundle_sha256": prefilter_bundle_sha256,
            "capture_lengths": [
                int(length) for length in candidate.stage_one_lengths
            ],
        },
        "browser_assets": browser_assets,
        "staging_package_manifest": dict(
            candidate.candidate_manifest["staging_package_manifest"]
        ),
        "dual_binding": dict(candidate.candidate_manifest["dual_binding"]),
        "dual_binding_sha256": candidate.candidate_manifest["dual_binding"][
            "sha256"
        ],
        "source_sha256": candidate.source_report,
    }


# ---------------------------------------------------------------------------
# suite loading: v2 machinery with the v3 protocol and candidate binding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class V3ReleaseSuite:
    root: Path
    intent_path: Path
    manifest_path: Path
    intent: dict[str, Any]
    release_manifest: dict[str, Any]
    candidate_sha256: str
    release_seed: int
    classes: tuple[str, ...]
    corpora: dict[int, release.CorpusSpec]
    support_indices: np.ndarray
    query_indices: np.ndarray
    split_report: dict[str, Any]
    prefix_nesting_report: dict[str, Any]
    dependency_report: dict[str, Any]
    start_probe_report: dict[str, Any]


def load_v3_release_suite(
    release_root: Path,
    candidate: V3Candidate,
) -> V3ReleaseSuite:
    """Load and cryptographically verify a sealed v3 suite before inference.

    This mirrors ``evaluate_invariant_release_suite.load_release_suite`` step
    by step; every sub-verification (file records, corpus geometry, prefix
    nesting, derivation records, start probe, dependency provenance, split) is
    the imported v2 function.  The two v3-specific changes are the expected
    ``evaluation_protocol`` object and the candidate binding: the intent's
    ``candidate_path`` must be the final dual release-candidate manifest,
    whose SHA-256 transitively pins both runtime bundles, both fusion trees,
    the passing validation evidence and exact staged/prefilter/browser bytes.
    """
    root = Path(release_root).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"release root must be a regular directory: {root}")
    if _is_below(root, release.LIVE_CORPUS) or _is_below(
        release.LIVE_CORPUS, root
    ):
        raise ValueError(
            "release root aliases, contains, or is inside the live corpus"
        )
    existing = root / "RELEASE_EVALUATION.json"
    if existing.exists():
        raise FileExistsError(
            f"{existing} already exists; this suite has been evaluated and a "
            "sealed one-shot may not be re-run"
        )

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

    release_seed = validate_release_seed(intent.get("release_seed"))
    expected_protocol = expected_evaluation_protocol(
        release_seed, candidate.stage_one_lengths
    )
    expected_historical = expected_protocol["historical_gate_redeclaration"]
    for name, holder in (("intent", intent), ("manifest", release_manifest)):
        embedded = holder.get("evaluation_protocol")
        embedded_historical = (
            embedded.get("historical_gate_redeclaration")
            if isinstance(embedded, Mapping)
            else None
        )
        try:
            _require_exact_json_value(
                embedded_historical,
                expected_historical,
                field=f"release.{name}.historical_gate_redeclaration",
            )
        except ValueError:
            raise ValueError(
                f"release {name} evaluation_protocol carries no matching "
                "historical_gate_redeclaration block. The consumed "
                "seed-20260734 0.12/0.84 levels must remain visible as "
                "inactive history, while the current suite is evaluated only "
                "against the strict imported 0.10/0.85 levels"
            )
        try:
            _require_exact_json_value(
                embedded,
                expected_protocol,
                field=f"release.{name}.evaluation_protocol",
            )
        except ValueError:
            raise ValueError(
                f"release {name} evaluation_protocol is missing or differs "
                "from the precommitted v3 evaluator contract"
            )

    intent_hash = str(release_manifest.get("release_intent_sha256", "")).lower()
    if intent_hash != _sha256(intent_path):
        raise ValueError(
            "RELEASE_MANIFEST does not bind the exact RELEASE_INTENT bytes"
        )
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
        try:
            _require_exact_json_value(
                release_manifest.get(key),
                intent.get(key),
                field=f"release_manifest.{key}",
            )
        except ValueError:
            raise ValueError(
                f"release manifest changed precommitted intent field {key!r}"
            )

    source_records = intent.get("source_sha256")
    if not isinstance(source_records, Mapping) or set(source_records) != {
        "launcher",
        "corpus_generator",
        "prefix_deriver",
        "tsx_package",
    }:
        raise ValueError("intent source_sha256 is missing")
    expected_source_paths = {
        "launcher": release.LAUNCHER,
        "corpus_generator": release.CORPUS_GENERATOR,
        "prefix_deriver": release.PREFIX_DERIVER,
    }
    for name, expected_path in expected_source_paths.items():
        record = source_records.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"intent source record {name!r} is missing")
        path = Path(str(record.get("path", ""))).expanduser().resolve()
        if path != expected_path:
            raise ValueError(
                f"intent source path {name!r} is not the release source"
            )
        if str(record.get("sha256", "")).lower() != _sha256(path):
            raise ValueError(f"intent source {name!r} changed after generation")
    if source_records.get("tsx_package") != "tsx@4.20.3":
        raise ValueError("release intent did not pin the expected tsx package")
    if release_manifest.get("source_sha256") != source_records:
        raise ValueError("release manifest source records differ from intent")

    dependency_report = release._validate_dependency_provenance(
        intent.get("dependency_provenance"),
        intent.get("runtime_provenance"),
    )
    signal_lab_root = Path(dependency_report["signallab"]["root"])

    # Candidate binding: the intent-bound file is the final dual manifest.
    candidate_path = (
        Path(str(intent.get("candidate_path", ""))).expanduser().resolve()
    )
    if candidate_path != candidate.candidate_manifest_path:
        raise ValueError(
            "RELEASE_INTENT candidate_path is not the frozen dual release "
            f"candidate manifest: {candidate_path} != "
            f"{candidate.candidate_manifest_path}"
        )
    if _is_below(candidate_path, root):
        raise ValueError("candidate must be frozen outside the release output root")
    candidate_hash = str(intent.get("candidate_sha256", "")).lower()
    if candidate_hash != candidate.candidate_manifest_sha256:
        raise ValueError(
            "candidate dual-manifest SHA-256 does not match RELEASE_INTENT"
        )
    for directory in (
        candidate.classifier_bundle_dir,
        candidate.rejector_bundle_dir,
        candidate.classifier_fusion_dir,
        candidate.rejector_fusion_dir,
        candidate.staged_dir,
        candidate.prefilter_dir,
    ):
        if _is_below(Path(directory), root):
            raise ValueError(
                "every candidate component must live outside the release root"
            )
    bound_files = (
        candidate.candidate_manifest_path,
        candidate.validation_evidence_path,
        candidate.frozen_contract_path,
        candidate.staged_metrics_path,
        *(
            _resolve_frozen_path(record["path"], field=f"browser_assets.{role}")
            for role, record in candidate.candidate_manifest[
                "browser_assets"
            ].items()
        ),
        _resolve_frozen_path(
            candidate.candidate_manifest["staging_package_manifest"]["path"],
            field="staging_package_manifest.path",
        ),
        _resolve_frozen_path(
            candidate.candidate_manifest["dual_binding"]["path"],
            field="dual_binding.path",
        ),
    )
    if any(_is_below(path, root) for path in bound_files):
        raise ValueError(
            "every candidate manifest component must live outside the "
            "release output root"
        )

    target_per_class = _integer(
        intent.get("target_per_class"), "target_per_class", MIN_TARGET_PER_CLASS
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

    corpora: dict[int, release.CorpusSpec] = {}
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
        directory_input = Path(str(record.get("directory", ""))).expanduser()
        expected_directory = root / f"n{length}"
        if directory_input.is_symlink() or expected_directory.is_symlink():
            raise ValueError(
                f"release corpus n{length} directory may not be a symlink"
            )
        directory = directory_input.resolve()
        if directory != expected_directory:
            raise ValueError(
                f"release corpus n{length} escapes its sealed directory"
            )
        files = record.get("files")
        if not isinstance(files, Mapping) or set(files) != {
            "corpus.json",
            "corpus.f32",
            "corpus_clean.f32",
        }:
            raise ValueError(
                f"release corpus n{length} file records are incomplete"
            )
        corpus_manifest_path = directory / "corpus.json"
        raw_path = directory / "corpus.f32"
        clean_path = directory / "corpus_clean.f32"
        _verify_file_record(
            corpus_manifest_path, files["corpus.json"], name=f"n{length}/corpus.json"
        )
        _verify_file_record(
            raw_path, files["corpus.f32"], name=f"n{length}/corpus.f32"
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
            or any(
                not isinstance(name, str) or not name for name in corpus_classes
            )
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
        if (
            raw_path.stat().st_size != expected_bytes
            or clean_path.stat().st_size != expected_bytes
        ):
            raise ValueError(
                f"n{length} raw byte size disagrees with corpus geometry"
            )
        keys = tuple(release._row_keys(items))
        if reference_items is None:
            reference_items = items
            reference_keys = keys
            classes = tuple(corpus_classes)
        else:
            if tuple(corpus_classes) != classes or keys != reference_keys:
                raise ValueError(
                    f"n{length} rows are not metadata-matched to the reference "
                    "corpus"
                )
        corpora[length] = release.CorpusSpec(
            capture_length=length,
            directory=directory,
            manifest_path=corpus_manifest_path,
            raw_path=raw_path,
            clean_path=clean_path,
            manifest=corpus_manifest,
            row_keys=keys,
        )

    assert reference_items is not None and classes is not None
    if classes != candidate.classes:
        raise ValueError(
            f"sealed corpus classes {list(classes)} differ from the candidate "
            f"classes {list(candidate.classes)}"
        )
    start_probe_report = release._validate_unscored_start_probe(
        release_manifest,
        root=root,
        longest=corpora[lengths[-1]],
        classes=classes,
        release_seed=release_seed,
        target_per_class=target_per_class,
        signal_lab_root=signal_lab_root,
    )
    prefix_nesting_report = release.verify_prefix_nesting(corpora)
    prefix_nesting_report["derivation_records"] = (
        release.validate_prefix_derivation_records(corpora)
    )
    support, query, split_report = release.predeclared_five_shot_split(
        reference_items, classes, release_seed
    )
    return V3ReleaseSuite(
        root=root,
        intent_path=intent_path,
        manifest_path=manifest_path,
        intent=intent,
        release_manifest=release_manifest,
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


# ---------------------------------------------------------------------------
# inference: the additive sub-path and the staged decision path
# ---------------------------------------------------------------------------


def _preprocess_iq_rows(
    captures: Sequence[np.ndarray],
    candidate: V3Candidate,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the shared frontend and return packed I/Q plus RAW features."""
    if not len(captures):
        raise ValueError("preprocessing requires at least one capture")
    packed = np.empty(
        (
            len(captures),
            2,
            candidate.patch_count * candidate.patch_length,
        ),
        dtype=np.float32,
    )
    features = np.empty((len(captures), len(candidate.feature_mean)), dtype=np.float32)
    for row, raw in enumerate(captures):
        channels, raw_features, _context = td_preprocess.preprocess(
            raw,
            patch_length=candidate.patch_length,
            patch_count=candidate.patch_count,
            target_frac=candidate.target_frac,
        )
        packed[row] = np.asarray(channels, dtype=np.float32)
        features[row] = np.asarray(raw_features, dtype=np.float32)
    if not np.isfinite(packed).all() or not np.isfinite(features).all():
        raise RuntimeError("v3 frontend produced non-finite preprocessing")
    return packed, features


def _standardize_features(
    raw_features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    role: str,
) -> np.ndarray:
    features = (
        np.asarray(raw_features, dtype=np.float32)
        - np.asarray(mean, dtype=np.float32)
    ) / np.asarray(std, dtype=np.float32)
    if not np.isfinite(features).all():
        raise RuntimeError(f"{role} feature standardization produced non-finite")
    return features.astype(np.float32, copy=False)


def _embed_fusion(
    fusion_module: torch.nn.Module,
    fusion_artifact: Any,
    packed: np.ndarray,
    features: np.ndarray,
    device: torch.device,
    *,
    role: str,
) -> dict[str, np.ndarray]:
    """Branch and fused embeddings for one explicitly named fusion role."""
    real = embed_all(
        fusion_module.real_branch, packed, features, device
    ).astype(np.float32, copy=False)
    complex_embedding = embed_all(
        fusion_module.complex_branch, packed, features, device
    ).astype(np.float32, copy=False)
    fused = assemble.fuse_numpy(
        real,
        complex_embedding,
        fusion_artifact.real_center,
        fusion_artifact.complex_center,
        weight_real=fusion_artifact.weight_real,
    )
    result = {"real": real, "complex": complex_embedding, "fusion": fused}
    if any(not np.isfinite(value).all() for value in result.values()):
        raise RuntimeError(f"{role} runtime produced non-finite embeddings")
    return result


def _embed_classifier(
    candidate: V3Candidate,
    packed: np.ndarray,
    raw_features: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    features = _standardize_features(
        raw_features,
        candidate.classifier_feature_mean,
        candidate.classifier_feature_std,
        role="classifier",
    )
    return features, _embed_fusion(
        candidate.classifier_fusion_module,
        candidate.classifier_fusion_artifact,
        packed,
        features,
        device,
        role="classifier",
    )


def _embed_rejector(
    candidate: V3Candidate,
    packed: np.ndarray,
    raw_features: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    features = _standardize_features(
        raw_features,
        candidate.rejector_feature_mean,
        candidate.rejector_feature_std,
        role="rejector",
    )
    return features, _embed_fusion(
        candidate.rejector_fusion_module,
        candidate.rejector_fusion_artifact,
        packed,
        features,
        device,
        role="rejector",
    )


def _final_labels_from_classifier(
    classifier_prediction: np.ndarray,
    rejected: np.ndarray,
) -> np.ndarray:
    """Apply the rejector mask without ever substituting its class guess."""
    labels = np.asarray(classifier_prediction, dtype=np.int64)
    mask = np.asarray(rejected, dtype=bool)
    if labels.ndim != 1 or mask.shape != labels.shape:
        raise ValueError("classifier predictions and rejection mask must align")
    return np.where(mask, np.int64(-1), labels)


def _deployment_classifier_labels(
    candidate: V3Candidate,
    packed: np.ndarray,
    raw_features: np.ndarray,
    rejected: np.ndarray,
    control_prediction: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Execute the literal deployed schedule on accepted rows only.

    The all-row classifier arm remains a separately labelled closed-gate
    control.  This second pass is the deployment proof: rejected rows are
    removed before the classifier call, and accepted predictions must equal
    the corresponding all-row control predictions.
    """
    mask = np.asarray(rejected, dtype=bool)
    control = np.asarray(control_prediction, dtype=np.int64)
    packed_value = np.asarray(packed)
    raw_value = np.asarray(raw_features)
    if (
        mask.ndim != 1
        or control.shape != mask.shape
        or len(packed_value) != len(mask)
        or len(raw_value) != len(mask)
    ):
        raise ValueError("deployment classifier inputs must align")
    accepted = np.flatnonzero(~mask)
    labels = np.full(len(mask), -1, dtype=np.int64)
    if len(accepted):
        _features, embeddings = _embed_classifier(
            candidate,
            packed_value[accepted],
            raw_value[accepted],
            device,
        )
        prediction, _distance = release._nearest(
            embeddings["fusion"],
            candidate.classifier_fusion_artifact.prototypes,
        )
        prediction = np.asarray(prediction, dtype=np.int64)
        if not np.array_equal(prediction, control[accepted]):
            raise AssertionError(
                "accepted-only deployment classifier differs from all-row "
                "closed-control predictions"
            )
        labels[accepted] = prediction
    return labels, {
        "rows": int(len(mask)),
        "rejected_rows_never_submitted": int(np.count_nonzero(mask)),
        "classifier_batch_rows": int(len(accepted)),
        "accepted_rows": int(len(accepted)),
        "accepted_prediction_agreement_with_closed_control": True,
        "schedule": "reject_then_classify_accepted_only",
    }


def _unstaged_scores(
    candidate: V3Candidate,
    embeddings: Mapping[str, np.ndarray],
    packed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """The additive stage-2 score of every row, from already-built embeddings.

    Mirrors ``fit_v3_openset.run``'s selection scoring: branch-LOF rank
    ensemble, frozen policy score, and the additivity assertion that the
    closed label is unchanged by scoring.
    """
    lof = openset_base.score_branch_lof(
        candidate.rejector.lof_components,
        {"real": embeddings["real"], "complex": embeddings["complex"]},
    )
    # train.nearest, exactly as fit_v3_openset.score_rows predicts, so the
    # additivity assertion below compares one formula with itself.  The closed
    # and invariance reports use release._nearest, exactly as v2 did.
    prediction, _distance = openset_base.nearest(
        embeddings["fusion"], candidate.rejector.prototypes
    )
    prediction = np.asarray(prediction, dtype=np.int64)
    score = candidate.rejector.policy.score(lof, packed, prediction)
    openset_base.assert_closed_label_unchanged(
        embeddings["fusion"],
        candidate.rejector.prototypes,
        prediction,
        score,
    )
    return prediction, np.asarray(score, dtype=np.float64)


def _stage_one_features(
    candidate: V3Candidate,
    captures: Sequence[np.ndarray],
    *,
    population: str,
) -> np.ndarray:
    return staged.prefilter_matrix(
        candidate.posedegen,
        captures,
        patch_length=candidate.patch_length,
        target_frac=candidate.target_frac,
        population=population,
    )


def _assert_attribution_conservation(
    report: Mapping[str, Any],
    *,
    gated: np.ndarray,
    staged_score: np.ndarray,
    unstaged_score: np.ndarray,
    staged_threshold: float,
    unstaged_threshold: float,
) -> None:
    """Prove every staged rejection is attributed once, with bounded overlap."""
    gated_mask = np.asarray(gated, dtype=bool)
    staged_value = np.asarray(staged_score, dtype=np.float64)
    unstaged_value = np.asarray(unstaged_score, dtype=np.float64)
    if (
        gated_mask.ndim != 1
        or staged_value.shape != gated_mask.shape
        or unstaged_value.shape != gated_mask.shape
        or not np.isfinite(staged_value).all()
        or not np.isfinite(unstaged_value).all()
        or not len(gated_mask)
    ):
        raise AssertionError("attribution inputs must be aligned finite vectors")
    expected_staged_threshold = _require_finite_json_float(
        staged_threshold, field="staged_threshold"
    )
    expected_unstaged_threshold = _require_finite_json_float(
        unstaged_threshold, field="unstaged_threshold"
    )
    for key, expected in (
        ("staged_threshold", expected_staged_threshold),
        ("unstaged_threshold", expected_unstaged_threshold),
    ):
        found = _require_finite_json_float(
            report.get(key), field=f"attribution.{key}"
        )
        if found != expected:
            raise AssertionError(
                f"attribution.{key}={found!r}, expected exact {expected!r}"
            )
    stage_two = (~gated_mask) & (
        staged_value > expected_staged_threshold
    )
    staged_rejected = gated_mask | stage_two
    control_rejected = unstaged_value > expected_unstaged_threshold
    rows = int(len(gated_mask))
    stage_one_total = int(np.count_nonzero(gated_mask))
    stage_two_total = int(np.count_nonzero(stage_two))
    staged_total = int(np.count_nonzero(staged_rejected))
    control_total = int(np.count_nonzero(control_rejected))
    overlap = int(np.count_nonzero(gated_mask & control_rejected))
    expected_counts = {
        "rows": rows,
        "staged_rejected_total": staged_total,
        "rejected_by_stage_one_total": stage_one_total,
        "stage_one_also_rejected_by_unstaged_control": overlap,
        "rejected_by_stage_two_only": stage_two_total,
        "attribution_partition_total": stage_one_total + stage_two_total,
        "attribution_partition_exact": True,
    }
    for key, expected in expected_counts.items():
        _require_exact_json_value(
            report.get(key), expected, field=f"attribution.{key}"
        )
    if staged_total != stage_one_total + stage_two_total:
        raise AssertionError("stage attribution does not conserve rejections")
    if not (0 <= overlap <= stage_one_total and overlap <= control_total):
        raise AssertionError("stage-one/control overlap is outside exact bounds")
    survivor_total = rows - stage_one_total
    expected_rates = {
        "staged_false_unknown_rate": staged_total / rows,
        "stage_one_gate_rate": stage_one_total / rows,
        "stage_two_false_unknown_rate_marginal": stage_two_total / rows,
        "stage_two_false_unknown_rate_among_survivors": (
            stage_two_total / survivor_total if survivor_total else None
        ),
        "unstaged_false_unknown_rate": control_total / rows,
    }
    for key, expected in expected_rates.items():
        found = report.get(key)
        if expected is None:
            if found is not None:
                raise AssertionError(f"attribution.{key} must be null")
        elif (
            type(found) is not float
            or not math.isfinite(found)
            or found != float(expected)
            or not 0.0 <= found <= 1.0
        ):
            raise AssertionError(
                f"attribution.{key}={found!r}, expected exact {expected!r}"
            )


def _staged_scores(
    candidate: V3Candidate,
    captures: Sequence[np.ndarray],
    packed: np.ndarray,
    features: np.ndarray,
    unstaged_prediction: np.ndarray,
    unstaged_score: np.ndarray,
    device: torch.device,
    *,
    capture_length: int,
    population: str,
) -> dict[str, Any]:
    """Score one row set through the staged decision path.

    On a fitted length this is ``fit_v3_openset_staged.score_staged``
    verbatim: stage 1 gates, downstream is never computed for gated rows
    inside this pass, survivors are scored on the COMPOSITE axis against the
    frozen composite threshold, and the staged/unstaged survivor agreement of
    the stage-2 SUB-score is asserted.  On a length ABOVE the longest fitted
    length the causal-prefix rule applies: stage-1 features are computed on
    each capture's first max-fitted-length samples
    (``staged.stage_one_prefix_captures``) and the gate scores them with that
    length's bundle; everything downstream is the covered path unchanged.  On
    a length BELOW the smallest fitted length the gate cannot fire and no
    stage-1 score exists, so the composite degenerates to the stage-2 rank
    alone: the staged score IS the additive score, still thresholded at the
    single frozen staged (composite) threshold, which is recorded rather than
    hidden.
    """
    rows = int(len(packed))
    threshold = candidate.threshold
    length = int(capture_length)
    fitted = length in candidate.stage_one_lengths
    prefix_applied = (not fitted) and length > int(
        max(candidate.stage_one_lengths)
    )
    active = fitted or prefix_applied
    feature_length: int | None = None
    if active:
        feature_captures, feature_length = staged.stage_one_prefix_captures(
            candidate.stage_one, captures, length
        )
        stage_features = _stage_one_features(
            candidate, feature_captures, population=population
        )
        outcome = staged.score_staged(
            candidate.rejector,
            candidate.stage_one,
            packed,
            features,
            stage_features,
            device,
            capture_length=length,
            composite=candidate.composite,
        )
        agreement = staged.subset_agreement(outcome, unstaged_score)
        gated = np.asarray(outcome.gated, dtype=bool)
        staged_score = np.asarray(outcome.staged_score, dtype=np.float64)
        by_stage = staged.known_false_unknown_by_stage(
            outcome,
            unstaged_score,
            threshold,
            candidate.stage_two_threshold,
        )
    else:
        gated = np.zeros(rows, dtype=bool)
        staged_score = np.asarray(unstaged_score, dtype=np.float64)
        agreement = {
            "survivor_rows": rows,
            "max_abs_difference": 0.0,
            "exact": True,
            "note": (
                "stage 1 inactive below the smallest fitted length; no "
                "stage-1 score exists, so the composite degenerates to the "
                "stage-2 rank and staged == additive on the score axis"
            ),
        }
        rejected = staged_score > threshold
        unstaged_rejected = staged_score > candidate.stage_two_threshold
        by_stage = {
            "rows": rows,
            "staged_threshold": float(threshold),
            "unstaged_threshold": float(candidate.stage_two_threshold),
            "staged_false_unknown_rate": float(np.mean(rejected)),
            "stage_one_gate_rate": 0.0,
            "stage_two_false_unknown_rate_marginal": float(np.mean(rejected)),
            "stage_two_false_unknown_rate_among_survivors": float(
                np.mean(rejected)
            ),
            "unstaged_false_unknown_rate": float(np.mean(unstaged_rejected)),
            "staged_rejected_total": int(np.count_nonzero(rejected)),
            "rejected_by_stage_one_total": 0,
            "stage_one_also_rejected_by_unstaged_control": 0,
            "rejected_by_stage_two_only": int(np.count_nonzero(rejected)),
            "attribution_partition_total": int(np.count_nonzero(rejected)),
            "attribution_partition_exact": True,
            "attribution": (
                "this capture length is below the smallest fitted stage-1 "
                "length, so no bundle's fitted length is a causal prefix of "
                "it; every row flowed to the additive stage-2 path and the "
                "composite degenerates to the stage-2 rank, thresholded at "
                "the staged (composite) threshold"
            ),
        }
    prediction = np.asarray(unstaged_prediction, dtype=np.int64).copy()
    rejected = gated | (staged_score > threshold)
    _assert_attribution_conservation(
        by_stage,
        gated=gated,
        staged_score=staged_score,
        unstaged_score=unstaged_score,
        staged_threshold=threshold,
        unstaged_threshold=candidate.stage_two_threshold,
    )
    return {
        "stage_one_active": bool(active),
        "stage_one_feature_length": (
            int(feature_length) if feature_length is not None else None
        ),
        "stage_one_prefix_rule_applied": bool(prefix_applied),
        "gated": gated,
        "rejected": rejected,
        "rejector_internal_prediction": prediction,
        "staged_score": staged_score,
        "gated_fraction": float(np.mean(gated)) if rows else 0.0,
        "rejected_fraction": (
            float(np.mean(staged_score > threshold)) if rows else 0.0
        ),
        "subset_scoring_agreement": agreement,
        "false_unknown_by_stage": by_stage,
    }


# ---------------------------------------------------------------------------
# novelty: generated once at the maximum length from the release seed
# ---------------------------------------------------------------------------


def _novelty_seed(base_seed: int, family: str) -> int:
    """The v2 derivation with the release seed as base (v2 used 20260729)."""
    if family not in NOVELTY_FAMILIES:
        raise ValueError(f"unfrozen novelty family {family!r}")
    payload = f"{int(base_seed)}\0{family}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _novelty_rows_by_family(
    base_seed: int,
    longest: int,
) -> tuple[dict[str, list[np.ndarray]], dict[str, Any]]:
    rows_by_family: dict[str, list[np.ndarray]] = {}
    seed_by_family: dict[str, int] = {}
    longest_bytes_sha256: dict[str, str] = {}
    for family in NOVELTY_FAMILIES:
        seed = _novelty_seed(base_seed, family)
        seed_by_family[family] = seed
        rng = np.random.default_rng(seed)
        rows = [
            np.asarray(NOVELTY[family](longest, rng), dtype=np.complex128)
            for _ in range(NOVELTY_N_EACH_PER_LENGTH)
        ]
        if any(
            row.shape != (longest,)
            or not np.isfinite(row.real).all()
            or not np.isfinite(row.imag).all()
            for row in rows
        ):
            raise RuntimeError(
                f"novelty generator {family!r} returned invalid rows"
            )
        digest = hashlib.sha256()
        for row in rows:
            digest.update(np.asarray(row, dtype="<c16").tobytes(order="C"))
        longest_bytes_sha256[family] = digest.hexdigest()
        rows_by_family[family] = rows
    provenance = {
        "base_seed": int(base_seed),
        "derived_seed_by_family": seed_by_family,
        "families": list(NOVELTY_FAMILIES),
        "rows_per_family": int(NOVELTY_N_EACH_PER_LENGTH),
        "longest_capture_length": int(longest),
        "longest_complex128_bytes_sha256": longest_bytes_sha256,
        "prefix_rule": (
            "each shorter novelty observation is the exact leading slice of "
            "the same in-memory maximum-length realization"
        ),
        "recalibration_performed": False,
    }
    return rows_by_family, provenance


# ---------------------------------------------------------------------------
# per-length open report on the staged axis (v2 report keys preserved)
# ---------------------------------------------------------------------------


def _open_report(
    known: Mapping[str, Any],
    novelty: Mapping[str, Mapping[str, Any]],
    candidate: V3Candidate,
    novelty_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    known_score = np.asarray(known["staged_score"], dtype=np.float64)
    family_scores = {
        family: np.asarray(novelty[family]["staged_score"], dtype=np.float64)
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
    threshold = candidate.threshold
    return {
        "known_rows": int(len(known_score)),
        "novelty_rows_per_family": int(NOVELTY_N_EACH_PER_LENGTH),
        "base_seed": int(novelty_provenance["base_seed"]),
        "derived_seed_by_family": dict(
            novelty_provenance["derived_seed_by_family"]
        ),
        "maximum_length_realization_sha256": dict(
            novelty_provenance["longest_complex128_bytes_sha256"]
        ),
        "shorter_observation_rule": novelty_provenance["prefix_rule"],
        "score": (
            "staged axis, policy version "
            f"{EXPECTED_STAGED_POLICY_VERSION}: frozen stage-1 noise prefilter "
            "gates; survivors carry the COMPOSITE "
            f"{staged.COMPOSITE_SURVIVOR_SCORE} over the frozen "
            "enrollment-ranked branch-LOF + geometry blend and the "
            "enrollment-survivor stage-1 rank; gated rows are placed above "
            "every survivor"
        ),
        "staged_policy_schema": EXPECTED_STAGED_POLICY_SCHEMA,
        "staged_policy_version": EXPECTED_STAGED_POLICY_VERSION,
        "staged_policy_kind": EXPECTED_STAGED_POLICY_KIND,
        "threshold_quantile": EXPECTED_COMPOSITE_THRESHOLD_QUANTILE,
        "policy_hygiene": dict(EXPECTED_POLICY_HYGIENE),
        "threshold": float(threshold),
        "threshold_rule": (
            "frozen q97 of the composite over enrollment stage-1 survivors, "
            "loaded verbatim; the unstaged control below uses the stage-2 "
            "policy's own untouched threshold"
        ),
        "recalibration_performed": False,
        "stage_one_active": bool(known["stage_one_active"]),
        "stage_one_feature_length": known["stage_one_feature_length"],
        "stage_one_prefix_rule_applied": bool(
            known["stage_one_prefix_rule_applied"]
        ),
        "auroc_overall": release._auroc(all_novelty, known_score),
        **{
            f"auroc_{name}": release._auroc(family_scores[name], known_score)
            for name in NOVELTY_FAMILIES
        },
        "known_false_unknown_rate": float(np.mean(known_score > threshold)),
        **{
            f"flagged_unknown_{name}": float(
                np.mean(family_scores[name] > threshold)
            )
            for name in NOVELTY_FAMILIES
        },
        "known_false_unknown_by_stage": dict(known["false_unknown_by_stage"]),
        "known_gated_fraction": float(known["gated_fraction"]),
        **{
            f"gated_fraction_{name}": float(novelty[name]["gated_fraction"])
            for name in NOVELTY_FAMILIES
        },
        "subset_scoring_agreement": {
            "known": dict(known["subset_scoring_agreement"]),
            **{
                name: dict(novelty[name]["subset_scoring_agreement"])
                for name in NOVELTY_FAMILIES
            },
        },
        "unstaged_control": {
            "auroc_overall": release._auroc(
                np.concatenate(
                    [
                        np.asarray(
                            novelty[name]["unstaged_score"], dtype=np.float64
                        )
                        for name in NOVELTY_FAMILIES
                    ]
                ),
                np.asarray(known["unstaged_score"], dtype=np.float64),
            ),
            **{
                f"auroc_{name}": release._auroc(
                    np.asarray(
                        novelty[name]["unstaged_score"], dtype=np.float64
                    ),
                    np.asarray(known["unstaged_score"], dtype=np.float64),
                )
                for name in NOVELTY_FAMILIES
            },
            "threshold": float(candidate.stage_two_threshold),
            "known_false_unknown_rate": float(
                np.mean(
                    np.asarray(known["unstaged_score"], dtype=np.float64)
                    > candidate.stage_two_threshold
                )
            ),
            **{
                f"flagged_unknown_{name}": float(
                    np.mean(
                        np.asarray(
                            novelty[name]["unstaged_score"], dtype=np.float64
                        )
                        > candidate.stage_two_threshold
                    )
                )
                for name in NOVELTY_FAMILIES
            },
            "role": (
                "verification control: the additive path over the same rows "
                "against the stage-2 policy's own untouched threshold; never "
                "a gate input"
            ),
        },
        "known_score_quantiles": {
            "p50": float(np.quantile(known_score, 0.5)),
            "p95": float(np.quantile(known_score, 0.95)),
        },
    }


# ---------------------------------------------------------------------------
# scale sweep: the v2 sweep semantics with the v3 additive sub-path
# ---------------------------------------------------------------------------


def _scale_sweep(
    spec: release.CorpusSpec,
    candidate: V3Candidate,
    device: torch.device,
    labels: np.ndarray,
    query_indices: np.ndarray,
) -> dict[str, Any]:
    eligible, eligibility = release._scale_eligible_indices(
        spec, query_indices, candidate.classes
    )
    base_captures = release._raw_rows(spec, eligible)
    rows = []
    embeddings_by_factor: dict[float, np.ndarray] = {}
    predictions_by_factor: dict[float, np.ndarray] = {}
    wanted = labels[eligible]
    snr = np.asarray(
        [
            release._finite_number(
                spec.manifest["items"][int(index)]["snrDb"], "snrDb"
            )
            for index in eligible
        ],
        dtype=np.float64,
    )
    for factor in PHYSICAL_SCALE_FACTORS:
        new_length = max(64, int(round(MATCHED_CAPTURE_LENGTH / factor)))
        effective_factor = (MATCHED_CAPTURE_LENGTH - 1) / (new_length - 1)
        captures = (
            base_captures
            if factor == 1.0
            else [
                production_preprocess.lin_resample(raw, new_length)
                for raw in base_captures
            ]
        )
        packed, raw_features = _preprocess_iq_rows(captures, candidate)
        _features, classifier_embeddings = _embed_classifier(
            candidate, packed, raw_features, device
        )
        embeddings = classifier_embeddings["fusion"]
        embeddings_by_factor[factor] = embeddings
        closed, prediction = release._simple_closed_report(
            embeddings,
            wanted,
            candidate.classifier_fusion_artifact.prototypes,
            candidate.classes,
        )
        predictions_by_factor[factor] = prediction
        transformed_sample_rates = np.asarray(
            [
                release._finite_number(
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
                    "sample_rate_hz_min": float(transformed_sample_rates.min()),
                    "sample_rate_hz_max": float(transformed_sample_rates.max()),
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
        row["paired_to_factor1"] = release._paired_invariance_report(
            embeddings_by_factor[factor],
            reference,
            predictions_by_factor[factor],
            reference_prediction,
            wanted,
            snr,
            candidate.classes,
        )
    return {
        "definition": (
            "normalized frequencies multiplied by s through deterministic "
            "linear resampling to N/s; common predeclared no-alias subset"
        ),
        "rejector_rule": SWEEP_REJECTOR_RULE,
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


# ---------------------------------------------------------------------------
# the evaluation
# ---------------------------------------------------------------------------


def evaluate_release(
    suite: V3ReleaseSuite,
    candidate: V3Candidate,
    device: torch.device,
) -> dict[str, Any]:
    """Run the fully precommitted evaluation without modifying frozen assets."""
    lengths = sorted(suite.corpora)
    longest = lengths[-1]
    query = suite.query_indices

    print("[v3 release] generating prefix-matched novelty", flush=True)
    novelty_rows, novelty_provenance = _novelty_rows_by_family(
        suite.release_seed, longest
    )

    classifier_embeddings_by_length: dict[
        int, dict[str, np.ndarray]
    ] = {}
    closed_by_length: dict[str, Any] = {}
    open_by_length: dict[str, Any] = {}
    staged_known_by_length: dict[int, dict[str, Any]] = {}
    labels_by_length: dict[int, np.ndarray] = {}
    snr_by_length: dict[int, np.ndarray] = {}

    for length in lengths:
        spec = suite.corpora[length]
        print(f"[v3 release] preprocessing/inference n={length}", flush=True)
        captures = release._raw_rows(
            spec, np.arange(int(spec.manifest["count"]), dtype=np.int64)
        )
        packed, raw_features = _preprocess_iq_rows(captures, candidate)
        _classifier_features, classifier_embeddings = _embed_classifier(
            candidate, packed, raw_features, device
        )
        rejector_features, rejector_embeddings = _embed_rejector(
            candidate, packed, raw_features, device
        )
        classifier_embeddings_by_length[length] = classifier_embeddings
        labels, snr, impaired = release._labels_and_snr(spec, candidate.classes)
        labels_by_length[length] = labels
        snr_by_length[length] = snr

        closed, classifier_prediction = release._closed_report(
            classifier_embeddings["fusion"][query],
            labels[query],
            snr[query],
            impaired[query],
            candidate.classifier_fusion_artifact.prototypes,
            candidate.classes,
        )
        closed_by_length[str(length)] = closed

        query_embeddings = {
            name: rejector_embeddings[name][query]
            for name in ("real", "complex", "fusion")
        }
        unstaged_prediction, unstaged_score = _unstaged_scores(
            candidate, query_embeddings, packed[query]
        )
        known_staged = _staged_scores(
            candidate,
            [captures[int(index)] for index in query],
            packed[query],
            rejector_features[query],
            unstaged_prediction,
            unstaged_score,
            device,
            capture_length=length,
            population=f"n{length}-known-query",
        )
        known_staged["unstaged_score"] = unstaged_score
        (
            known_staged["final_label_or_unknown"],
            known_staged["deployment_classifier_execution"],
        ) = _deployment_classifier_labels(
            candidate,
            packed[query],
            raw_features[query],
            known_staged["rejected"],
            classifier_prediction,
            device,
        )
        staged_known_by_length[length] = known_staged

        novelty_by_family: dict[str, dict[str, Any]] = {}
        for family in NOVELTY_FAMILIES:
            prefixes = [row[:length] for row in novelty_rows[family]]
            novelty_packed, novelty_raw_features = _preprocess_iq_rows(
                prefixes, candidate
            )
            novelty_features, novelty_embeddings = _embed_rejector(
                candidate, novelty_packed, novelty_raw_features, device
            )
            family_prediction, family_score = _unstaged_scores(
                candidate, novelty_embeddings, novelty_packed
            )
            family_staged = _staged_scores(
                candidate,
                prefixes,
                novelty_packed,
                novelty_features,
                family_prediction,
                family_score,
                device,
                capture_length=length,
                population=f"n{length}-novelty-{family}",
            )
            family_staged["unstaged_score"] = family_score
            novelty_by_family[family] = family_staged
        open_by_length[str(length)] = _open_report(
            known_staged, novelty_by_family, candidate, novelty_provenance
        )

    matched_enrollment = classifier_embeddings_by_length[
        MATCHED_CAPTURE_LENGTH
    ]["fusion"]
    matched_labels = labels_by_length[MATCHED_CAPTURE_LENGTH]
    five_shot_by_length: dict[str, Any] = {}
    for length in lengths:
        if not np.array_equal(labels_by_length[length], matched_labels):
            raise AssertionError("matched release label arrays differ by length")
        five_shot_by_length[str(length)] = release._five_shot_report(
            matched_enrollment,
            classifier_embeddings_by_length[length]["fusion"],
            matched_labels,
            suite.support_indices,
            suite.query_indices,
            candidate.classes,
        )

    reference = classifier_embeddings_by_length[MATCHED_CAPTURE_LENGTH][
        "fusion"
    ][query]
    reference_prediction, _ = release._nearest(
        reference, candidate.classifier_fusion_artifact.prototypes
    )
    length_rows = []
    for length in lengths:
        current = classifier_embeddings_by_length[length]["fusion"][query]
        prediction, _ = release._nearest(
            current, candidate.classifier_fusion_artifact.prototypes
        )
        closed = closed_by_length[str(length)]
        paired = release._paired_invariance_report(
            current,
            reference,
            prediction,
            reference_prediction,
            labels_by_length[length][query],
            snr_by_length[length][query],
            candidate.classes,
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
        "matched_capture_length": int(MATCHED_CAPTURE_LENGTH),
        "matched_rows": int(len(query)),
        "all_classes_required": list(candidate.classes),
        "rejector_rule": SWEEP_REJECTOR_RULE,
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
        candidate,
        device,
        labels_by_length[MATCHED_CAPTURE_LENGTH],
        suite.query_indices,
    )

    gates = release._assemble_release_gates(
        candidate_sha256=suite.candidate_sha256,
        expected_candidate_sha256=suite.release_manifest["candidate_sha256"],
        prefix_nesting_passes=bool(suite.prefix_nesting_report["passes"]),
        dependency_provenance_passes=bool(suite.dependency_report["passes"]),
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
    # This is the v2 assembler's strict output verbatim.  In particular,
    # five-shot stays at 0.85 and known FUR stays at 0.10; the historical
    # seed-20260734 redeclaration is metadata only.
    all_pass = all(bool(value["passes"]) for value in gates.values())

    max_fitted = int(max(candidate.stage_one_lengths))
    stage_one_coverage = {
        str(length): (
            "fitted"
            if int(length) in candidate.stage_one_lengths
            else "causal_prefix"
            if int(length) > max_fitted
            else "inactive"
        )
        for length in lengths
    }
    stage_one_feature_lengths = {
        str(length): staged_known_by_length[length]["stage_one_feature_length"]
        for length in lengths
    }
    final_label_counts = {
        str(length): {
            "rows": int(len(staged_known_by_length[length]["final_label_or_unknown"])),
            "unknown": int(
                np.count_nonzero(
                    staged_known_by_length[length]["final_label_or_unknown"]
                    == -1
                )
            ),
            "gated_at_stage_one": int(
                np.count_nonzero(staged_known_by_length[length]["gated"])
            ),
            "accepted_rows": int(
                np.count_nonzero(~staged_known_by_length[length]["rejected"])
            ),
            "accepted_label_source": "classifier_fusion_8k_regularized",
            "deployment_classifier_execution": dict(
                staged_known_by_length[length][
                    "deployment_classifier_execution"
                ]
            ),
            "accepted_classifier_rejector_label_agreement": float(
                np.mean(
                    staged_known_by_length[length]["final_label_or_unknown"][
                        ~staged_known_by_length[length]["rejected"]
                    ]
                    == staged_known_by_length[length][
                        "rejector_internal_prediction"
                    ][~staged_known_by_length[length]["rejected"]]
                )
            )
            if np.any(~staged_known_by_length[length]["rejected"])
            else None,
        }
        for length in lengths
    }

    return {
        "schema": EVALUATOR_SCHEMA,
        "status": "complete",
        "release_evidence": True,
        "evaluation_version": EVALUATION_VERSION,
        "development_data_loaded": False,
        "retraining_performed": False,
        "recalibration_performed": False,
        "architecture_contract": {
            "candidate_architecture": "dual_fusion_classifier8k_rejector4k",
            "intentional_dual_fusion": True,
            "decision_order": [
                "stage-one causal-prefix noise gate",
                "4k rejector fusion and staged known/unknown policy",
                "8k classifier nearest-prototype label for accepted rows",
            ],
            "known_label_source": "classifier_fusion_8k_regularized",
            "known_unknown_source": "rejector_fusion_4k_frozen_policy",
            "closed_gate_source": "classifier_fusion_8k_regularized",
            "open_gate_source": "rejector_fusion_4k_frozen_policy",
            "additive_only": False,
            "changes_closed_label": True,
            "open_set_decision_changes_closed_label": True,
            "gates_before_classification": True,
            "architecture_contract_change": str(
                noise_prefilter.ARCHITECTURE_CONTRACT_CHANGE
            ),
            "staged_policy_kind": EXPECTED_STAGED_POLICY_KIND,
            "staged_policy_schema": EXPECTED_STAGED_POLICY_SCHEMA,
            "staged_policy_version": EXPECTED_STAGED_POLICY_VERSION,
            "composite_threshold_quantile": (
                EXPECTED_COMPOSITE_THRESHOLD_QUANTILE
            ),
            "policy_hygiene": dict(EXPECTED_POLICY_HYGIENE),
            "survivor_score": staged.COMPOSITE_SURVIVOR_SCORE,
            "staged_score_note": staged.STAGED_SCORE_NOTE,
            "known_false_unknown_accounting": KNOWN_FUR_ACCOUNTING,
            "stage_one_capture_lengths": [
                int(length) for length in candidate.stage_one_lengths
            ],
            "stage_one_coverage_by_length": stage_one_coverage,
            "stage_one_feature_length_by_length": stage_one_feature_lengths,
            "stage_one_prefix_rule": {
                "max_fitted_length": max_fitted,
                "rule": STAGE_ONE_PREFIX_RULE,
            },
            "stage_one_uncovered_rule": STAGE_ONE_UNCOVERED_RULE,
            "sweep_rejector_rule": SWEEP_REJECTOR_RULE,
            "typescript_runtime_note": (
                "the deployed TypeScript staged runtime must apply this same "
                "causal-prefix rule at capture lengths above the longest "
                "fitted stage-1 length; a runtime that refuses such lengths "
                "does not match the sealed claim and must be reconciled "
                "before any ship"
            ),
        },
        "candidate": {
            "path": str(candidate.candidate_manifest_path),
            "sha256": suite.candidate_sha256,
            "schema": CANDIDATE_MANIFEST_SCHEMA,
            "candidate_id": CANDIDATE_ID,
            "classes": list(candidate.classes),
            "components": candidate_provenance(candidate),
            "frozen_assets_used": [
                "8k classifier fusion state, centers, feature moments and "
                "enrollment-only prototypes",
                "4k rejector fusion state, centers, feature moments and "
                "enrollment-only prototypes",
                "both self-verified runtime bundle manifests and assets",
                "frozen stage-1 per-length noise-prefilter bundles",
                "frozen stage-2 branch-LOF ensemble (training reference, "
                "enrollment ranks)",
                "frozen stage-2 policy (geometry blend, enrollment q95 "
                "threshold)",
                "frozen composite survivor policy (enrollment-survivor "
                "stage-1 rank calibration, q97 composite threshold)",
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
        "open_staged_per_length": open_by_length,
        "staged_known_decisions_per_length": final_label_counts,
        "matched_length_sweep": length_sweep,
        "physical_scale_sweep": scale_sweep,
        "gates": gates,
        "historical_gate_redeclaration": (
            historical_gate_redeclaration_block()
        ),
        "all_release_gates_pass": all_pass,
        "provenance": {
            "release_root": str(suite.root),
            "release_intent_sha256": _sha256(suite.intent_path),
            "release_manifest_sha256": _sha256(suite.manifest_path),
            "release_seed": suite.release_seed,
            "evaluation_protocol": expected_evaluation_protocol(
                suite.release_seed, candidate.stage_one_lengths
            ),
            "predeclared_split": suite.split_report,
            "prefix_nesting": suite.prefix_nesting_report,
            "dependency_provenance": suite.dependency_report,
            "unscored_start_probe": suite.start_probe_report,
            "novelty_prefix_generation": novelty_provenance,
            "candidate_sha256": suite.candidate_sha256,
            "corpus_assets": suite.release_manifest["corpora"],
            "evaluator_path": str(Path(__file__).resolve()),
            "evaluator_sha256": _sha256(Path(__file__).resolve()),
            "v2_evaluator_sha256": _sha256(
                _source_path("evaluate_invariant_release_suite.py")
            ),
            "gate_helpers_imported_from_v2": [
                "GATE_FLOORS",
                "_gate",
                "_assemble_release_gates",
                "_closed_report",
                "_simple_closed_report",
                "_classification_subset",
                "_paired_invariance_report",
                "_five_shot_report",
                "_auroc",
                "_nearest",
                "_scale_eligible_indices",
                "predeclared_five_shot_split",
                "verify_prefix_nesting",
                "validate_prefix_derivation_records",
                "_validate_unscored_start_probe",
                "_validate_dependency_provenance",
            ],
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def expected_protocol_for_prefilter_set(
    release_seed: int,
    prefilter_dir: Path,
) -> dict[str, Any]:
    """The predeclared protocol, from the fitted stage-1 set alone.

    The protocol object is a function of the release seed, the fitted stage-1
    lengths and this module's frozen constants -- nothing else about the
    candidate enters it.  Deriving the lengths from the prefilter set (each
    bundle re-validated by ``staged.load_stage_one``, including coverage of
    every sealed capture length) lets the launcher fixture be pinned BEFORE
    the composite validation artifact exists, without weakening anything: at
    evaluation time the full candidate is loaded and the intent protocol is
    recomputed against it, so a fixture generated from a different prefilter
    set than the sealed candidate's fails loudly before any gate is scored.
    """
    stage_one = staged.load_stage_one(
        staged.load_prefilter_module(),
        staged.load_posedegen_module(),
        Path(prefilter_dir),
        required_lengths=REQUIRED_CAPTURE_LENGTHS,
    )
    return expected_evaluation_protocol(release_seed, stage_one.lengths)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.print_expected_protocol is not None:
        protocol = expected_protocol_for_prefilter_set(
            int(args.print_expected_protocol), Path(args.prefilter_dir)
        )
        print(json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False))
        return {"expected_evaluation_protocol": protocol}
    device = resolve_device(args.device)
    candidate = load_candidate(
        candidate_manifest_path=Path(args.candidate_manifest),
        classifier_bundle_dir=Path(args.classifier_bundle_dir),
        rejector_bundle_dir=Path(args.rejector_bundle_dir),
        classifier_fusion_dir=Path(args.classifier_fusion_dir),
        rejector_fusion_dir=Path(args.rejector_fusion_dir),
        staged_dir=Path(args.staged_dir),
        prefilter_dir=Path(args.prefilter_dir),
        device=device,
    )
    if args.release_root is None:
        raise ValueError(
            "--release-root is required unless --print-expected-protocol is "
            "given"
        )
    suite = load_v3_release_suite(Path(args.release_root), candidate)
    report = evaluate_release(suite, candidate, device)
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else suite.root / "RELEASE_EVALUATION.json"
    )
    if not _is_below(output, suite.root):
        raise ValueError("release evaluation output must stay inside release root")
    _write_json_exclusive(output, report)
    print(
        f"[v3 release] gates="
        f"{'PASS' if report['all_release_gates_pass'] else 'FAIL'} -> {output}",
        flush=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--release-root",
        default=None,
        help="the sealed suite directory written by the release launcher",
    )
    parser.add_argument("--output")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument(
        "--candidate-manifest",
        default=str(DEFAULT_CANDIDATE_MANIFEST),
        help=(
            "the final frozen dual release-candidate manifest; this exact "
            "file is bound by RELEASE_INTENT candidate_path/candidate_sha256"
        ),
    )
    parser.add_argument(
        "--classifier-bundle-dir",
        default=str(DEFAULT_CLASSIFIER_BUNDLE_DIR),
        help="the self-verified 8k classifier runtime bundle",
    )
    parser.add_argument(
        "--rejector-bundle-dir",
        default=str(DEFAULT_REJECTOR_BUNDLE_DIR),
        help="the self-verified 4k rejector runtime bundle",
    )
    parser.add_argument(
        "--classifier-fusion-dir",
        default=str(DEFAULT_CLASSIFIER_FUSION_DIR),
        help="the regularized 8k known-class fusion artifact",
    )
    parser.add_argument(
        "--rejector-fusion-dir",
        default=str(DEFAULT_REJECTOR_FUSION_DIR),
        help="the frozen 4k known/unknown fusion artifact",
    )
    parser.add_argument(
        "--staged-dir",
        default=str(DEFAULT_STAGED_DIR),
        help=(
            "the passing staged validation artifact holding the frozen "
            "stage-2 LOF ensemble and policy npz files"
        ),
    )
    parser.add_argument(
        "--prefilter-dir",
        default=str(DEFAULT_PREFILTER_DIR),
        help="the fitted per-length stage-1 noise-prefilter bundle set",
    )
    parser.add_argument(
        "--print-expected-protocol",
        type=int,
        default=None,
        metavar="RELEASE_SEED",
        help=(
            "print the exact evaluation_protocol object the release intent "
            "must embed for the given seed, then exit without touching any "
            "release root. Needs only --prefilter-dir (the stage-1 lengths); "
            "the full candidate is not loaded, and the evaluation itself "
            "recomputes and enforces this object against the real candidate"
        ),
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
