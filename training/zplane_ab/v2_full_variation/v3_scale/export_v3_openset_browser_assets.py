"""Export the v3 STAGED open-set decision layer for the browser, with parity.

What this produces (staging only, never release):

``artifacts/staging/time_domain_v3_openset/``
  ``time-domain-openset-weights-v1.json``
      Stage 1: the three per-length noise-prefilter bundles (the tightened
      0.01-budget set ``noise_prefilter_fit20261001_budget001``) converted to
      plain JSON -- 7 means, 7 scales, 7 coefficients, an intercept and a
      score-space threshold per capture length.
      Stage 2: the fitted branch-LOF ensemble and the frozen stage-2 policy
      from the staged artifact (``v3_branch_lof_components.npz`` /
      ``v3_open_policy_stage_two.npz``), including the single frozen geometry
      feature column, both enrollment calibrations and the stage-2 q95.
      COMPOSITE (staged policy version 2): the survivor composite policy
      (``v3_staged_composite_policy.npz``) -- the enrollment-survivor stage-1
      calibration, the composite calibration, the composite q95 threshold and
      the explicit ``policy_version`` the TypeScript runtime refuses to run
      without.
  ``time-domain-classifier-weights-v1.json``
      The decision-layer half of the v3 runtime bundle
      supplied through ``--bundle-dir``: feature standardization, fusion
      centers/constants, class order and fused prototypes. Encoder weights are
      the sibling exporter's job and are deliberately NOT here.
  ``time-domain-openset-parity-v1.json``
      Known + noise + chirp rows pushed through the STAGED Python path with
      every per-stage intermediate and the final decision, INCLUDING which
      stage rejected each row.  This is the fixture the TypeScript suite
      asserts against.
  ``manifest.json``
      SHA-256 of every input consumed and every output written.

Evidence hygiene:

- Novelty parity rows are drawn at :data:`PARITY_NOVELTY_SEED`, which lies
  outside both the novelty-seed namespace (20260900-20260999) and the
  prefilter fit-only band (20261000-20261999).  Parity fixtures are not
  evaluation evidence and must never be allowed to look like it.
- Known parity rows come from the development SELECTION split (scored, never
  fit).  No sealed or consumed path is read; ``assemble.reject_sealed_path``
  guards every directory argument.
- Release seed 20260731 is neither used nor reachable from here.

Determinism: JSON is written with ``sort_keys=True``, no timestamps, no
absolute paths in the payloads, and float repr is Python's shortest
round-trip, so byte-identical inputs give byte-identical outputs.

Self-checks before anything is written:

- every runtime-bundle asset SHA-256 is re-verified against its manifest;
- the fusion artifact's centers/prototypes/feature stats must be exactly the
  runtime bundle's (they are the same fit);
- the staged stage-2 score assembled from intermediates here must EXACTLY
  equal ``fit_v3_openset.score_rows`` on every surviving row, so the fixture
  cannot drift from the real Python path;
- the closed label of every survivor is asserted unchanged by the rejector
  (``assert_closed_label_unchanged`` runs inside ``score_rows``).

Run with the training venv and explicit, freshly validated inputs:

    .venv-training/bin/python \
        training/zplane_ab/v2_full_variation/v3_scale/export_v3_openset_browser_assets.py \
        --bundle-dir <runtime-bundle> \
        --fusion-dir <fusion-artifact> \
        --staged-dir <passing-validate-artifact> \
        --prefilter-dir <prefilter-bundles> \
        --output-dir <fresh-staging-output>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
ZPAB = V2.parent
TRAINING = ZPAB.parent
for _path in (TRAINING, ZPAB, V2, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import assemble_v3_fusion as assemble  # noqa: E402
import fit_v3_openset as base  # noqa: E402
import fit_v3_openset_staged as staged  # noqa: E402
import invariant_patch_data as invariant_data  # noqa: E402
from invariant_fusion import CenteredInvariantFusion  # noqa: E402
from invariant_patch_cnn import (  # noqa: E402
    InvariantPatchCNN,
    InvariantPatchConfig,
)
import noise_prefilter  # noqa: E402
import pose_degeneracy as pdg  # noqa: E402
import run_time_domain_dev as dev  # noqa: E402
import time_domain_geometry as geometry  # noqa: E402
import time_domain_invariant_patch_preprocess as td_preprocess  # noqa: E402
import known_only_patch_openset as kop  # noqa: E402
from known_only_patch_openset import KnownOnlyLOFOpenSet  # noqa: E402
from openset_eval import NOVELTY  # noqa: E402
from train import embed_all, nearest  # noqa: E402
from v3_time_domain_openset import (  # noqa: E402
    FROZEN_GEOMETRY_FEATURE,
    FROZEN_GEOMETRY_WEIGHT,
    FROZEN_THRESHOLD_QUANTILE,
    FROZEN_V2_WEIGHT,
    FrozenV3OpenSet,
    empirical_rank,
)
from v3_time_domain_openset import (  # noqa: E402
    time_domain_geometry as stage_two_geometry,
)


OPENSET_SCHEMA = "atomos.v3.time-domain-openset.staged"
# Version 2: the composite survivor score (staged policy version 2).  The
# TypeScript runtime refuses any other schema version, exactly as it refuses
# any other staged policy version.
OPENSET_SCHEMA_VERSION = 2
CLASSIFIER_SCHEMA = "atomos.v3.time-domain-invariant-fusion.browser-decision"
CLASSIFIER_SCHEMA_VERSION = 1
PARITY_SCHEMA = "time-domain-openset-parity-v1"

REJECTOR_RUNTIME_ROLE = "known_unknown_rejector"
CLASSIFIER_RUNTIME_ROLE = "accepted_known_classifier"
REJECTOR_WEIGHTS_NAME = "time-domain-v3-rejector-weights.json"
CLASSIFIER_WEIGHTS_NAME = "time-domain-v3-classifier-weights.json"
REJECTOR_PROBE_NAME = "time-domain-v3-rejector-probe.json"
CLASSIFIER_PROBE_NAME = "time-domain-v3-classifier-probe.json"
OPENSET_POLICY_NAME = "time-domain-v3-openset-policy.json"
DUAL_BINDING_NAME = "time-domain-v3-dual-binding.json"
DUAL_BINDING_SCHEMA = "atomos.v3.time-domain-dual-fusion.binding"
DUAL_BINDING_SCHEMA_VERSION = 1
FUSION_EXPORT_MANIFEST_NAME = "export-manifest.json"
FUSION_EXPORT_SCHEMA = (
    "atomos.v3.time-domain-invariant-fusion.browser-weights.export-manifest"
)
FUSION_EXPORT_SCHEMA_VERSION = 2
# Compatibility alias for code that calls the policy payload helper directly.
# A dual export never emits the old anonymous filename.
WEIGHTS_NAME = OPENSET_POLICY_NAME
PARITY_NAME = "time-domain-openset-parity-v1.json"
MANIFEST_NAME = "manifest.json"
OPENSET_MANIFEST_SCHEMA = "time-domain-v3-dual-openset-staging-manifest-v1"
CANDIDATE_ID = "v3.3-decoupled-8k-classifier-4k-rejector"
MAX_OPENSET_POLICY_BYTES = 25 * 1024 * 1024
SHA256_RE = re.compile(r"[0-9a-f]{64}")

#: Parity probe only.  Outside the novelty namespace (20260900-20260999) and
#: the prefilter fit-only band (20261000-20261999); it is not evaluation
#: evidence and is recorded as such in the fixture.
PARITY_NOVELTY_SEED = 20269101
NOVELTY_ROWS_PER_FAMILY = 6
# Full fitted-length parity plus two compact long-capture rows exercising the
# release suite's causal-prefix rule.  The development corpus has no N32768
# known rows, so long parity uses the fixed, non-evidence novelty fixture.
FIXTURE_LENGTHS = (4096, 8192, 16384)
CAUSAL_PREFIX_FIXTURE_LENGTH = 32768
REQUIRED_VALIDATION_PREFIX_LENGTHS = (
    *FIXTURE_LENGTHS,
    CAUSAL_PREFIX_FIXTURE_LENGTH,
)
CAUSAL_PREFIX_GATED_ROW_NAME = "causal-prefix-gated-N32768"
CAUSAL_PREFIX_SURVIVOR_ROW_NAME = "causal-prefix-survivor-N32768"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, Path):
        raise TypeError("paths must not leak into deterministic payloads")
    return value


def json_bytes(payload: Any, *, compact: bool = False) -> bytes:
    options: dict[str, Any] = {
        "sort_keys": True,
        "allow_nan": False,
    }
    if compact:
        options["separators"] = (",", ":")
    else:
        options["indent"] = 1
    return (json.dumps(_jsonable(payload), **options) + "\n").encode("utf-8")


def write_bytes(path: Path, raw: bytes) -> None:
    temporary = Path(path).with_name(f".{Path(path).name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json(path: Path, payload: Any, *, compact: bool = False) -> None:
    write_bytes(path, json_bytes(payload, compact=compact))


def validate_parity_seed(seed: int) -> int:
    value = int(seed)
    staged._refuse_ledger_seed(value, "a parity fixture draw")
    low, high = noise_prefilter.NOVELTY_SEED_NAMESPACE
    if low <= value <= high:
        raise ValueError(
            f"parity seed {value} lies in the novelty namespace {low}-{high}; "
            "a parity fixture must not consume evaluation evidence"
        )
    low, high = noise_prefilter.FITTING_SEED_NAMESPACE
    if low <= value <= high:
        raise ValueError(
            f"parity seed {value} lies in the prefilter fit-only band "
            f"{low}-{high}; a parity fixture must not look like a fitting draw"
        )
    return value


def validate_staged_evidence_report(report: Mapping[str, Any]) -> None:
    """Admit only a passing, strict-gate validation artifact for export.

    Export is a release-boundary operation, not a way to turn a design or a
    failed validation run into runtime bytes.  This checks the report before
    any output directory is created and retains the original development
    gates, including known FUR <= 0.10.
    """
    required_scalars = (
        ("status", "development_openset_pass"),
        ("role", "validate"),
        ("gates_are_evidence", True),
        ("development_only", True),
        ("release_evidence", False),
        ("sealed_release_data_used", 0),
        ("consumed_test_rows_used", 0),
        ("sealed_release_paths_read", 0),
        ("release_seed_20260735_used", False),
        ("all_pass", True),
        ("closed_label_can_be_gated", True),
        ("additive_only", False),
        ("changes_closed_label", True),
        ("gates_before_classification", True),
    )
    mismatched = [
        f"{key}={report.get(key)!r}, expected {expected!r}"
        for key, expected in required_scalars
        if report.get(key) != expected
    ]
    if mismatched:
        raise ValueError(
            "staged artifact is not passing validate-role evidence: "
            + "; ".join(mismatched)
        )

    architecture = report.get("architecture")
    expected_architecture = {
        "kind": staged.STAGED_POLICY_KIND,
        "schema": int(staged.STAGED_POLICY_SCHEMA),
        "staged_policy_version": staged.STAGED_POLICY_VERSION,
    }
    if not isinstance(architecture, Mapping):
        raise ValueError("staged artifact carries no architecture record")
    for key, expected in expected_architecture.items():
        if architecture.get(key) != expected:
            raise ValueError(
                f"staged architecture {key}={architecture.get(key)!r}, "
                f"expected {expected!r}"
            )

    gates = report.get("gates")
    expected_gate_names = set(staged.GATE_FLOORS) | {
        "known_false_unknown_rate"
    }
    if not isinstance(gates, Mapping) or set(gates) != expected_gate_names:
        raise ValueError(
            "staged gate set differs from the frozen development gates: "
            f"{sorted(gates) if isinstance(gates, Mapping) else gates!r}"
        )
    for name, floor in staged.GATE_FLOORS.items():
        gate = gates[name]
        if (
            not isinstance(gate, Mapping)
            or gate.get("passes") is not True
            or float(gate.get("floor", float("nan"))) != float(floor)
        ):
            raise ValueError(
                f"staged gate {name!r} did not pass its frozen floor {floor}"
            )
    known_gate = gates["known_false_unknown_rate"]
    if (
        not isinstance(known_gate, Mapping)
        or known_gate.get("passes") is not True
        or float(known_gate.get("ceiling", float("nan")))
        != float(staged.KNOWN_FUR_CEILING)
        or float(known_gate.get("worst", float("inf")))
        > float(staged.KNOWN_FUR_CEILING)
    ):
        raise ValueError(
            "known false-unknown rate did not pass the strict development "
            f"ceiling {staged.KNOWN_FUR_CEILING}"
        )

    stage_one = report.get("stage_one")
    if not isinstance(stage_one, Mapping):
        raise ValueError("staged artifact carries no stage-one provenance")
    fitted_lengths = tuple(
        sorted(int(length) for length in stage_one.get("capture_lengths", ()))
    )
    if fitted_lengths != FIXTURE_LENGTHS:
        raise ValueError(
            f"staged fitted lengths {list(fitted_lengths)} != "
            f"{list(FIXTURE_LENGTHS)}"
        )

    protocol = report.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("staged artifact carries no protocol record")
    prefix_lengths = tuple(
        int(length) for length in protocol.get("prefix_lengths", ())
    )
    if prefix_lengths != REQUIRED_VALIDATION_PREFIX_LENGTHS:
        raise ValueError(
            f"staged validation prefixes {list(prefix_lengths)} != "
            f"{list(REQUIRED_VALIDATION_PREFIX_LENGTHS)}"
        )
    feature_lengths = protocol.get(
        "stage_one_feature_length_by_prefix_length"
    )
    expected_feature_lengths = {
        str(length): (
            max(FIXTURE_LENGTHS)
            if length == CAUSAL_PREFIX_FIXTURE_LENGTH
            else length
        )
        for length in REQUIRED_VALIDATION_PREFIX_LENGTHS
    }
    if feature_lengths != expected_feature_lengths:
        raise ValueError(
            "staged validation does not declare the exact N32768 -> N16384 "
            f"causal-prefix mapping: {feature_lengths!r}"
        )
    if protocol.get("stage_one_causal_prefix_rule_lengths") != [
        CAUSAL_PREFIX_FIXTURE_LENGTH
    ]:
        raise ValueError(
            "staged validation does not identify N32768 as the sole "
            "causal-prefix length"
        )

    seeds = report.get("seeds")
    if not isinstance(seeds, Mapping):
        raise ValueError("staged artifact carries no seed provenance")
    if seeds.get("design_novelty_seed") != 20260949:
        raise ValueError(
            "staged validation must reference frozen design seed 20260949"
        )
    if seeds.get("novelty_seeds") != [20260950, 20260951]:
        raise ValueError(
            "staged validation must be the one-shot 20260950/20260951 pair"
        )
    if seeds.get("release_seed_not_spent") != 20260735:
        raise ValueError("staged validation does not preserve release seed 20260735")


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def load_browser_fusion_export(
    directory: Path,
    *,
    expected_role: str,
    expected_weights_name: str,
    expected_probe_name: str,
    expected_bundle_manifest_sha256: str,
) -> dict[str, Any]:
    """Verify one role-tagged browser fusion export and its two emitted files."""
    root = assemble.reject_sealed_path(
        Path(directory), f"{expected_role} browser fusion export"
    )
    manifest_path = root / FUSION_EXPORT_MANIFEST_NAME
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if (
        manifest.get("schema") != FUSION_EXPORT_SCHEMA
        or manifest.get("schema_version") != FUSION_EXPORT_SCHEMA_VERSION
    ):
        raise ValueError(f"{root} has an unexpected fusion export schema")
    if manifest.get("runtime_role") != expected_role:
        raise ValueError(
            f"{root} runtime_role={manifest.get('runtime_role')!r}, "
            f"expected {expected_role!r}"
        )
    if (
        manifest.get("source_bundle_manifest_sha256")
        != expected_bundle_manifest_sha256
    ):
        raise ValueError(
            f"{root} is not derived from the expected {expected_role} bundle"
        )
    emitted = manifest.get("emitted")
    expected_names = {expected_weights_name, expected_probe_name}
    if not isinstance(emitted, Mapping) or set(emitted) != expected_names:
        raise ValueError(
            f"{root} emitted files differ from role contract "
            f"{sorted(expected_names)}"
        )
    verified: dict[str, dict[str, Any]] = {}
    for name in sorted(expected_names):
        record = emitted[name]
        if not isinstance(record, Mapping):
            raise ValueError(f"{root}/{name} has no file record")
        digest = _require_sha256(record.get("sha256"), f"{root}/{name} SHA")
        size = record.get("bytes")
        path = root / name
        if type(size) is not int or size <= 0 or path.stat().st_size != size:
            raise ValueError(f"{root}/{name} byte count differs from manifest")
        if _sha256(path) != digest:
            raise ValueError(f"{root}/{name} SHA differs from manifest")
        verified[name] = {"bytes": size, "sha256": digest}

    weights_path = root / expected_weights_name
    with weights_path.open(encoding="utf-8") as handle:
        weights = json.load(handle)
    if (
        weights.get("schema")
        != "atomos.v3.time-domain-invariant-fusion.browser-weights"
        or weights.get("schema_version") != 1
        or weights.get("status") != "staging_not_release"
        or weights.get("runtime_role") != expected_role
    ):
        raise ValueError(f"{weights_path} does not carry its exact runtime role")
    provenance = weights.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("source_bundle_manifest_sha256")
        != expected_bundle_manifest_sha256
    ):
        raise ValueError(f"{weights_path} loses its source bundle binding")
    return {
        "directory": root,
        "manifest": manifest,
        "manifest_sha256": _sha256(manifest_path),
        "weights_name": expected_weights_name,
        "weights_sha256": verified[expected_weights_name]["sha256"],
        "weights_bytes": verified[expected_weights_name]["bytes"],
        "probe_name": expected_probe_name,
        "probe_sha256": verified[expected_probe_name]["sha256"],
        "probe_bytes": verified[expected_probe_name]["bytes"],
        "weights": weights,
    }


def verify_staged_artifact_hashes(
    report: Mapping[str, Any],
    staged_dir: Path,
) -> dict[str, str]:
    """Verify every staged policy byte against its validation report."""
    artifacts = report.get("artifacts")
    expected = {
        "v3_branch_lof_components.npz",
        "v3_open_policy_stage_two.npz",
        staged.COMPOSITE_POLICY_FILENAME,
    }
    if not isinstance(artifacts, Mapping) or set(artifacts) != expected:
        raise ValueError(
            "staged report artifact set is incomplete: "
            f"{sorted(artifacts) if isinstance(artifacts, Mapping) else artifacts!r}"
        )
    verified: dict[str, str] = {}
    for name in sorted(expected):
        path = Path(staged_dir) / name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != artifacts[name]:
            raise RuntimeError(
                f"staged artifact {name} SHA {actual} != report "
                f"{artifacts[name]}"
            )
        verified[name] = actual
    return verified


# ---------------------------------------------------------------------------
# loading and cross-checking the fitted state
# ---------------------------------------------------------------------------


def load_bundle(bundle_dir: Path, *, expected_role: str) -> dict[str, Any]:
    """Load the v3 runtime bundle, verifying every asset SHA-256."""
    bundle = assemble.reject_sealed_path(Path(bundle_dir), "runtime bundle")
    manifest_path = bundle / "bundle_manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("kind") != "v3-time-domain-centered-invariant-fusion":
        raise ValueError(f"{bundle} is not a v3 runtime bundle")
    if manifest.get("runtime_role") != expected_role:
        raise ValueError(
            f"{bundle} runtime_role={manifest.get('runtime_role')!r}, "
            f"expected {expected_role!r}"
        )
    if manifest.get("development_only") is not True:
        raise RuntimeError(f"{bundle} must be development_only")
    rejection = manifest.get("rejection")
    if (
        not isinstance(rejection, Mapping)
        or rejection.get("state") != "unset"
        or rejection.get("external_staged_policy_required_for_abstention")
        is not True
    ):
        raise RuntimeError(
            f"{bundle} does not declare that staged rejection is a separate, "
            "required artifact"
        )
    blockers = manifest.get("release_blockers")
    if not isinstance(blockers, list) or not blockers:
        raise RuntimeError(f"{bundle} carries no release requirements")
    serialized_blockers = " ".join(str(item) for item in blockers).lower()
    stale = ("not refit", "unmeasured", "unported", "no untouched release seed")
    if any(token in serialized_blockers for token in stale):
        raise RuntimeError(f"{bundle} carries stale release blockers")
    arrays: dict[str, np.ndarray] = {}
    for name, meta in manifest["assets"].items():
        path = bundle / name
        digest = _sha256(path)
        if digest != meta["sha256"]:
            raise RuntimeError(f"{path} does not match its recorded SHA-256")
        if name.endswith(".npy"):
            arrays[name] = np.load(path)
    with (bundle / "probe_fixture.json").open(encoding="utf-8") as handle:
        probe = json.load(handle)
    return {
        "directory": bundle,
        "manifest": manifest,
        "manifest_sha256": _sha256(manifest_path),
        "arrays": arrays,
        "probe": probe,
    }


def bind_bundle_to_fusion(
    bundle: Mapping[str, Any],
    artifact: base.FusionArtifact,
) -> None:
    """Prove a role bundle is a byte-exact export of one fusion artifact."""
    manifest = bundle["manifest"]
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("runtime bundle has no provenance block")
    recorded_directory = provenance.get("source_fusion_artifact")
    if not isinstance(recorded_directory, str) or not recorded_directory:
        raise ValueError("runtime bundle has no source_fusion_artifact")
    recorded_path = Path(recorded_directory)
    if not recorded_path.is_absolute():
        recorded_path = TRAINING.parent / recorded_path
    if recorded_path.resolve() != artifact.directory.resolve():
        raise RuntimeError(
            f"runtime bundle was exported from {recorded_path.resolve()}, "
            f"not {artifact.directory.resolve()}"
        )
    if (
        provenance.get("source_dev_metrics_sha256")
        != artifact.file_sha256["dev_metrics.json"]
    ):
        raise RuntimeError(
            "runtime bundle is not bound to the loaded fusion dev_metrics"
        )
    state_name = str(artifact.metrics["artifacts"]["state_dict"])
    if (
        _sha256(bundle["directory"] / "fusion_state_dict.pt")
        != artifact.file_sha256[state_name]
    ):
        raise RuntimeError(
            "runtime bundle fusion_state_dict.pt differs from the fusion artifact"
        )
    for bundle_name, fusion_value in (
        ("real_center.npy", artifact.real_center),
        ("complex_center.npy", artifact.complex_center),
        ("fusion_prototypes.npy", artifact.prototypes),
        ("feature_mean.npy", artifact.feature_mean),
        ("feature_std.npy", artifact.feature_std),
    ):
        if not np.array_equal(
            np.asarray(bundle["arrays"][bundle_name], dtype=np.float32),
            np.asarray(fusion_value, dtype=np.float32),
        ):
            raise RuntimeError(
                f"runtime bundle {bundle_name} differs from the fusion artifact"
            )


def build_bundle_fusion(
    bundle: Mapping[str, Any],
    device: torch.device,
) -> CenteredInvariantFusion:
    """Rebuild inference solely from the verified, self-contained bundle.

    In particular this never follows the historical ``source_branches`` paths
    embedded in a fusion metrics report.  Those training directories are not
    runtime dependencies and need not exist in a clean checkout.
    """
    manifest = bundle["manifest"]
    architecture = manifest.get("architecture")
    if (
        not isinstance(architecture, Mapping)
        or architecture.get("kind") != "centered_invariant_fusion"
    ):
        raise ValueError("runtime bundle has no centered-fusion architecture")
    real = InvariantPatchCNN(
        InvariantPatchConfig(**dict(architecture["real"])).validate()
    )
    complex_net = InvariantPatchCNN(
        InvariantPatchConfig(**dict(architecture["complex"])).validate()
    )
    arrays = bundle["arrays"]
    fusion = CenteredInvariantFusion(
        real,
        complex_net,
        arrays["real_center.npy"],
        arrays["complex_center.npy"],
        alpha_real=float(architecture["alpha_real"]),
        alpha_complex=float(architecture["alpha_complex"]),
        weight_real=float(architecture["weight_real"]),
        eps=float(architecture["eps"]),
    )
    try:
        state = torch.load(
            bundle["directory"] / "fusion_state_dict.pt",
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:  # pragma: no cover - older PyTorch
        state = torch.load(
            bundle["directory"] / "fusion_state_dict.pt",
            map_location="cpu",
        )
    fusion.load_state_dict(state, strict=True)
    fusion = fusion.to(device).eval()
    np.testing.assert_array_equal(
        fusion.real_center.detach().cpu().numpy(),
        np.asarray(arrays["real_center.npy"], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        fusion.complex_center.detach().cpu().numpy(),
        np.asarray(arrays["complex_center.npy"], dtype=np.float32),
    )
    return fusion


def load_stage_two_state(
    staged_dir: Path,
) -> tuple[
    list[tuple[str, float, KnownOnlyLOFOpenSet]],
    FrozenV3OpenSet,
    dict[str, str],
]:
    """Rebuild the fitted LOF ensemble and frozen policy from the staged npz.

    The composite survivor policy is deliberately NOT loaded here: this
    3-tuple contract is shared with ``evaluate_v3_release_suite.load_candidate``,
    which performs its own composite load and version refusal.  This module's
    :func:`load_composite_state` is the exporter-side counterpart.
    """
    directory = assemble.reject_sealed_path(Path(staged_dir), "staged artifact")
    lof_path = directory / "v3_branch_lof_components.npz"
    policy_path = directory / "v3_open_policy_stage_two.npz"
    hashes = {
        "v3_branch_lof_components.npz": _sha256(lof_path),
        "v3_open_policy_stage_two.npz": _sha256(policy_path),
    }

    with np.load(lof_path) as payload:
        if str(payload["kind"]) != "v3_branch_lof_rank_ensemble":
            raise ValueError(f"{lof_path} is not a v3 branch LOF ensemble")
        count = int(payload["component_count"])
        components: list[tuple[str, float, KnownOnlyLOFOpenSet]] = []
        for index in range(count):
            prefix = f"component_{index}_"
            scorer = KnownOnlyLOFOpenSet(
                embedding_reference=np.asarray(
                    payload[f"{prefix}embedding_reference"], dtype=np.float64
                ),
                embedding_mean=np.asarray(
                    payload[f"{prefix}embedding_mean"], dtype=np.float64
                ),
                embedding_scale=np.asarray(
                    payload[f"{prefix}embedding_scale"], dtype=np.float64
                ),
                reference_k_distance=np.asarray(
                    payload[f"{prefix}reference_k_distance"], dtype=np.float64
                ),
                reference_local_density=np.asarray(
                    payload[f"{prefix}reference_local_density"], dtype=np.float64
                ),
                calibration=np.asarray(
                    payload[f"{prefix}calibration"], dtype=np.float64
                ),
                calibration_scores=np.asarray(
                    payload[f"{prefix}calibration_scores"], dtype=np.float64
                ),
                neighbors=int(payload[f"{prefix}neighbors"]),
            )
            components.append(
                (
                    str(payload[f"{prefix}branch_source"]),
                    float(payload[f"{prefix}weight"]),
                    scorer,
                )
            )

    expected = [
        (branch, weight, neighbors) for branch, weight, neighbors in base.BRANCH_LOF
    ]
    found = [
        (branch, weight, scorer.neighbors) for branch, weight, scorer in components
    ]
    if expected != found:
        raise RuntimeError(
            f"staged LOF ensemble {found} differs from the inherited shape "
            f"{expected}"
        )

    with np.load(policy_path) as payload:
        policy = FrozenV3OpenSet.from_payload(
            {key: payload[key] for key in payload.files}
        )
    return components, policy, hashes


def load_composite_state(
    staged_dir: Path,
    policy: FrozenV3OpenSet,
) -> tuple[Any, str]:
    """Load and verify the composite survivor policy from the staged npz.

    The composite loader enforces the staged policy version and recomputes
    the q95 threshold from the stored calibration; the stage-2 threshold
    cross-check proves the composite was fit against exactly this stage-2
    state.  Returns ``(composite, sha256)``.
    """
    directory = assemble.reject_sealed_path(Path(staged_dir), "staged artifact")
    composite_path = directory / staged.COMPOSITE_POLICY_FILENAME
    composite = staged.load_composite_policy(
        composite_path,
        expected_stage_two_threshold=float(policy.threshold),
    )
    return composite, _sha256(composite_path)


def compute_branch_lof_raw(
    scorer: KnownOnlyLOFOpenSet, embeddings: np.ndarray
) -> np.ndarray:
    """The raw (pre-rank) LOF value, mirroring ``KnownOnlyLOFOpenSet.score``."""
    standardized = (
        np.asarray(embeddings, dtype=np.float64) - scorer.embedding_mean
    ) / scorer.embedding_scale
    indices, distances = kop._nearest_neighbors(
        standardized, scorer.embedding_reference, scorer.neighbors
    )
    density = 1.0 / (
        np.maximum(distances, scorer.reference_k_distance[indices]).mean(axis=1)
        + 1e-12
    )
    return scorer.reference_local_density[indices].mean(axis=1) / density


# ---------------------------------------------------------------------------
# weight conversion
# ---------------------------------------------------------------------------


def prefilter_model_payload(model: noise_prefilter.NoisePrefilter) -> dict[str, Any]:
    if not model.has_threshold:
        raise ValueError("prefilter bundle has no fitted operating point")
    if tuple(model.feature_names) != noise_prefilter.PREFILTER_FEATURES:
        raise ValueError(
            "prefilter feature order differs from the frozen PREFILTER_FEATURES"
        )
    return {
        "capture_length": int(model.capture_length),
        "feature_names": list(model.feature_names),
        "mean": model.mean,
        "scale": model.scale,
        "coefficients": model.coefficients,
        "intercept": float(model.intercept),
        "threshold_score": float(model.threshold_score),
    }


def composite_policy_payload(composite: Any) -> dict[str, Any]:
    """The `v3_staged_composite_policy.npz` payload as plain JSON.

    Field names keep the npz spelling exactly, so the TypeScript loader's
    validation (policy version refusal, sorted calibrations, recomputed q95)
    mirrors ``fit_v3_openset_staged.load_composite_policy`` key for key.
    """
    return {
        "schema": int(staged.STAGED_POLICY_SCHEMA),
        "kind": staged.STAGED_POLICY_KIND,
        "policy_version": staged.STAGED_POLICY_VERSION,
        "survivor_score": staged.COMPOSITE_SURVIVOR_SCORE,
        "threshold_quantile": float(staged.COMPOSITE_THRESHOLD_QUANTILE),
        "threshold": float(composite.threshold),
        "stage_two_threshold": float(composite.stage_two_threshold),
        "stage_one_calibration_raw": composite.stage_one_calibration_raw,
        "composite_calibration_raw": composite.composite_calibration_raw,
        "enrollment_rows": int(composite.enrollment_rows),
        "enrollment_gated_rows": int(composite.enrollment_gated_rows),
        "enrollment_capture_length": int(composite.enrollment_capture_length),
    }


def openset_weights_payload(
    prefilters: Mapping[int, noise_prefilter.NoisePrefilter],
    prefilter_set_sha: str,
    components: Sequence[tuple[str, float, KnownOnlyLOFOpenSet]],
    policy: FrozenV3OpenSet,
    composite: Any,
    frontend: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    feature_index = policy.geometry_feature_index
    return {
        "schema": OPENSET_SCHEMA,
        "schema_version": OPENSET_SCHEMA_VERSION,
        "status": "staging_not_release",
        "contract": {
            "additive_only": False,
            "changes_closed_label": True,
            "gates_before_classification": True,
            "architecture_contract_change": (
                noise_prefilter.ARCHITECTURE_CONTRACT_CHANGE
            ),
        },
        "frontend": dict(frontend),
        "stage_one": {
            "kind": noise_prefilter.NOISE_PREFILTER_VERSION,
            "feature_names": list(noise_prefilter.PREFILTER_FEATURES),
            "match_frontend_min_bandwidth": bool(
                staged.PREFILTER_MATCH_FRONTEND_MIN_BANDWIDTH
            ),
            "set_sha256": prefilter_set_sha,
            "models": {
                str(length): prefilter_model_payload(prefilters[length])
                for length in sorted(prefilters)
            },
        },
        "stage_two": {
            "kind": staged.STAGED_POLICY_KIND,
            "lof_components": [
                {
                    "branch": branch,
                    "weight": float(weight),
                    "neighbors": int(scorer.neighbors),
                    "mean": scorer.embedding_mean,
                    "scale": scorer.embedding_scale,
                    "reference": scorer.embedding_reference,
                    "reference_k_distance": scorer.reference_k_distance,
                    "reference_local_density": scorer.reference_local_density,
                    "calibration": scorer.calibration,
                }
                for branch, weight, scorer in components
            ],
            "policy": {
                "branch_lof_rank_weight": float(FROZEN_V2_WEIGHT),
                "geometry_weight": float(FROZEN_GEOMETRY_WEIGHT),
                "threshold_quantile": float(FROZEN_THRESHOLD_QUANTILE),
                "threshold": float(policy.threshold),
                "geometry_feature": FROZEN_GEOMETRY_FEATURE,
                "class_geometry_mean": policy.geometry.class_mean[:, feature_index],
                "class_geometry_scale": policy.geometry.class_scale[:, feature_index],
                "geometry_calibration": policy.geometry_calibration_raw,
                "combined_calibration": policy.combined_calibration_raw,
            },
        },
        "composite": composite_policy_payload(composite),
        "provenance": dict(provenance),
    }


def classifier_weights_payload(
    bundle: Mapping[str, Any], provenance: Mapping[str, Any]
) -> dict[str, Any]:
    manifest = bundle["manifest"]
    arrays = bundle["arrays"]
    frontend = manifest["frontend"]
    fusion = manifest["fusion"]
    return {
        "schema": CLASSIFIER_SCHEMA,
        "schema_version": CLASSIFIER_SCHEMA_VERSION,
        "status": "staging_not_release",
        "frontend": {
            "version": str(frontend["version"]),
            "patch_length": int(frontend["patch_length"]),
            "patch_count": int(frontend["patch_count"]),
            "target_frac": float(frontend["target_frac"]),
            "packed_length": int(frontend["packed_length"]),
            "uses_frequency_transform": False,
        },
        "feature_standardization": {
            "mean": arrays["feature_mean.npy"],
            "std": arrays["feature_std.npy"],
        },
        "fusion": {
            "real_center": arrays["real_center.npy"],
            "complex_center": arrays["complex_center.npy"],
            "alpha_real": float(fusion["alpha_real"]),
            "alpha_complex": float(fusion["alpha_complex"]),
            "weight_real": float(fusion["weight_real"]),
            "eps": float(fusion["eps"]),
        },
        "classification": {
            "classes": list(manifest["classification"]["classes"]),
            "prototypes": arrays["fusion_prototypes.npy"],
            "distance": "squared_euclidean",
        },
        "provenance": dict(provenance),
    }


def dual_binding_payload(
    *,
    frontend: Mapping[str, Any],
    rejector_asset_sha256: str,
    classifier_asset_sha256: str,
    openset_asset_sha256: str,
    rejector_bundle_manifest_sha256: str,
    classifier_bundle_manifest_sha256: str,
    rejector_fusion_directory_sha256: str,
    classifier_fusion_directory_sha256: str,
    staged_validation_report_sha256: str,
    staged_artifacts_sha256: Mapping[str, str],
    novelty_seeds: Sequence[int],
) -> dict[str, Any]:
    """Create the exact fail-closed binding consumed by the dual TS runtime."""
    digests = {
        "rejector asset": rejector_asset_sha256,
        "classifier asset": classifier_asset_sha256,
        "open-set policy": openset_asset_sha256,
        "rejector bundle": rejector_bundle_manifest_sha256,
        "classifier bundle": classifier_bundle_manifest_sha256,
        "rejector fusion": rejector_fusion_directory_sha256,
        "classifier fusion": classifier_fusion_directory_sha256,
        "validation report": staged_validation_report_sha256,
        **{
            f"staged artifact {name}": digest
            for name, digest in staged_artifacts_sha256.items()
        },
    }
    for label, digest in digests.items():
        _require_sha256(digest, label)
    if rejector_asset_sha256 == classifier_asset_sha256:
        raise ValueError("rejector and classifier browser assets must differ")
    if rejector_bundle_manifest_sha256 == classifier_bundle_manifest_sha256:
        raise ValueError("rejector and classifier runtime bundles must differ")
    if rejector_fusion_directory_sha256 == classifier_fusion_directory_sha256:
        raise ValueError("rejector and classifier fusion artifacts must differ")
    expected_staged = {
        "v3_branch_lof_components.npz",
        "v3_open_policy_stage_two.npz",
        staged.COMPOSITE_POLICY_FILENAME,
    }
    if set(staged_artifacts_sha256) != expected_staged:
        raise ValueError("dual binding staged artifact set is incomplete")
    seeds = [int(seed) for seed in novelty_seeds]
    if seeds != [20260950, 20260951]:
        raise ValueError("dual binding requires validation seeds 20260950/20260951")

    return {
        "schema": DUAL_BINDING_SCHEMA,
        "schema_version": DUAL_BINDING_SCHEMA_VERSION,
        "status": "staging_not_release",
        "candidate_id": CANDIDATE_ID,
        "frontend": {
            "version": td_preprocess.PREPROCESS_VERSION,
            "patch_length": int(frontend["patch_length"]),
            "patch_count": int(frontend["patch_count"]),
            "target_frac": float(frontend["target_frac"]),
            "packed_length": int(frontend["packed_length"]),
            "uses_frequency_transform": False,
        },
        "execution_order": [
            "stage_one_noise_gate",
            "rejector_known_unknown",
            "classifier_known_label",
        ],
        "roles": {
            "rejector": {
                "asset": REJECTOR_WEIGHTS_NAME,
                "asset_sha256": rejector_asset_sha256,
                "fusion_directory_sha256": rejector_fusion_directory_sha256,
                "runtime_bundle_manifest_sha256": (
                    rejector_bundle_manifest_sha256
                ),
                "runtime_role": REJECTOR_RUNTIME_ROLE,
                "responsibility": "known_unknown_only",
            },
            "classifier": {
                "asset": CLASSIFIER_WEIGHTS_NAME,
                "asset_sha256": classifier_asset_sha256,
                "fusion_directory_sha256": classifier_fusion_directory_sha256,
                "runtime_bundle_manifest_sha256": (
                    classifier_bundle_manifest_sha256
                ),
                "runtime_role": CLASSIFIER_RUNTIME_ROLE,
                "responsibility": "accepted_known_label_only",
            },
        },
        "openset_policy": {
            "asset": OPENSET_POLICY_NAME,
            "asset_sha256": openset_asset_sha256,
            "rejector_asset_sha256": rejector_asset_sha256,
            "fitted_rejector_runtime_bundle_manifest_sha256": (
                rejector_bundle_manifest_sha256
            ),
            "staged_validation_report_sha256": (
                staged_validation_report_sha256
            ),
            "staged_artifacts_sha256": dict(staged_artifacts_sha256),
        },
        "validation": {
            "report_sha256": staged_validation_report_sha256,
            "role": "validate",
            "status": "development_openset_pass",
            "novelty_seeds": seeds,
        },
        "fail_closed": {
            "role_assets_bound_by_sha256": True,
            "distinct_role_assets": True,
            "role_asset_sha256_must_differ": True,
            "classifier_runs_only_after_rejector_acceptance": True,
            "public_known_label_from_classifier_only": True,
        },
    }


# ---------------------------------------------------------------------------
# the staged parity path (one row at a time, mirrored from Python exactly)
# ---------------------------------------------------------------------------


def stage_one_row(
    prefilters: Mapping[int, noise_prefilter.NoisePrefilter],
    capture: np.ndarray,
    *,
    patch_length: int,
    target_frac: float,
) -> dict[str, Any]:
    length = int(len(capture))
    fitted_lengths = tuple(sorted(int(item) for item in prefilters))
    if not fitted_lengths:
        raise ValueError("stage-one parity needs at least one prefilter")
    if length in prefilters:
        evaluated_length = length
        causal_prefix_applied = False
    elif length > max(fitted_lengths):
        evaluated_length = max(fitted_lengths)
        causal_prefix_applied = True
    else:
        raise KeyError(
            f"no fitted noise prefilter for capture length {length}; "
            f"available {list(fitted_lengths)} and causal-prefix substitution "
            "applies only above the longest fitted length"
        )
    model = prefilters[evaluated_length]
    feature_capture = np.asarray(capture[:evaluated_length])
    if len(feature_capture) != evaluated_length:
        raise ValueError(
            f"capture has {len(capture)} samples but stage one needs an "
            f"N{evaluated_length} causal prefix"
        )
    minimum = geometry.minimum_bandwidth_for_active_span(
        evaluated_length, patch_length, target_frac
    )
    features = pdg.pose_degeneracy_features(
        feature_capture, min_bandwidth=minimum
    )
    selected = noise_prefilter.select_features(
        features[None, :], model.feature_names
    )[0]
    score = float(
        model.score(
            selected[None, :], capture_length=evaluated_length
        )[0]
    )
    gated = bool(
        model.decide(
            selected[None, :], capture_length=evaluated_length
        )[0]
    )
    rank = float(staged._bounded_rank(np.asarray([score]))[0])
    if gated != (score >= float(model.threshold_score)):
        raise AssertionError("stage-1 gate rule disagreement")
    return {
        "capture_length": length,
        "evaluated_length": evaluated_length,
        "causal_prefix_applied": causal_prefix_applied,
        "features": selected,
        "score": score,
        "rank": rank,
        "gated": gated,
        "threshold_score": float(model.threshold_score),
    }


def stage_two_row(
    rejector: base.Rejector,
    components: Sequence[tuple[str, float, KnownOnlyLOFOpenSet]],
    policy: FrozenV3OpenSet,
    packed: np.ndarray,
    standardized: np.ndarray,
    classes: Sequence[str],
    device: torch.device,
) -> dict[str, Any]:
    real_embedding = embed_all(
        rejector.nets["real"], packed[None], standardized[None], device
    ).astype(np.float32, copy=False)
    complex_embedding = embed_all(
        rejector.nets["complex"], packed[None], standardized[None], device
    ).astype(np.float32, copy=False)
    fused = assemble.fuse_numpy(
        real_embedding,
        complex_embedding,
        rejector.real_center,
        rejector.complex_center,
        weight_real=rejector.weight_real,
    )
    prediction, distances = nearest(fused, rejector.prototypes)
    predicted = int(prediction[0])

    lof_scores = []
    weighted = 0.0
    for branch, weight, scorer in components:
        embedding = real_embedding if branch == "real" else complex_embedding
        raw = float(compute_branch_lof_raw(scorer, embedding)[0])
        rank = float(scorer.score(np.asarray(embedding, dtype=np.float64))[0])
        weighted += weight * rank
        lof_scores.append(
            {
                "branch": branch,
                "weight": float(weight),
                "neighbors": int(scorer.neighbors),
                "raw": raw,
                "rank": rank,
            }
        )

    statistics, names = stage_two_geometry(packed[None])
    feature_index = policy.geometry_feature_index
    if names[feature_index] != FROZEN_GEOMETRY_FEATURE:
        raise AssertionError("frozen geometry feature index drifted")
    statistic = float(statistics[0, feature_index])
    deviation = float(
        policy.geometry.deviations(packed[None], np.asarray([predicted]))[
            0, feature_index
        ]
    )
    geometry_rank = float(
        empirical_rank(
            np.asarray([deviation]), policy.geometry_calibration_raw
        )[0]
    )
    combined = float(
        FROZEN_V2_WEIGHT * weighted
        + FROZEN_GEOMETRY_WEIGHT * geometry_rank
    )
    score = float(
        empirical_rank(np.asarray([combined]), policy.combined_calibration_raw)[0]
    )

    # The intermediates above must reproduce the REAL Python path exactly.
    check_prediction, check_score = base.score_rows(
        rejector, packed[None], standardized[None], device
    )
    if int(check_prediction[0]) != predicted:
        raise AssertionError("decomposed prediction differs from score_rows")
    if float(check_score[0]) != score:
        raise AssertionError(
            f"decomposed stage-2 score {score} differs from score_rows "
            f"{float(check_score[0])}"
        )

    return {
        "runtime_role": REJECTOR_RUNTIME_ROLE,
        "real_embedding": real_embedding[0],
        "complex_embedding": complex_embedding[0],
        "fused_embedding": fused[0],
        "predicted_class_index": predicted,
        "predicted_class_label": str(classes[predicted]),
        "squared_prototype_distances": distances[0],
        "lof": lof_scores,
        "lof_rank": float(weighted),
        "geometry_statistic": statistic,
        "geometry_deviation": deviation,
        "geometry_rank": geometry_rank,
        "combined_raw": combined,
        "score": score,
        # The stage-2 policy's own enrollment q95, recorded for cross-checks;
        # under staged policy version 2 the survivor DECISION compares the
        # COMPOSITE against the composite threshold.
        "threshold": float(policy.threshold),
    }


def classifier_row(
    classifier: base.Rejector,
    packed: np.ndarray,
    standardized: np.ndarray,
    classes: Sequence[str],
    device: torch.device,
) -> dict[str, Any]:
    """Run the accepted-known classifier without any rejection side effects."""
    real_embedding = embed_all(
        classifier.nets["real"], packed[None], standardized[None], device
    ).astype(np.float32, copy=False)
    complex_embedding = embed_all(
        classifier.nets["complex"], packed[None], standardized[None], device
    ).astype(np.float32, copy=False)
    fused = assemble.fuse_numpy(
        real_embedding,
        complex_embedding,
        classifier.real_center,
        classifier.complex_center,
        weight_real=classifier.weight_real,
    )
    prediction, distances = nearest(fused, classifier.prototypes)
    predicted = int(prediction[0])
    return {
        "runtime_role": CLASSIFIER_RUNTIME_ROLE,
        "real_embedding": real_embedding[0],
        "complex_embedding": complex_embedding[0],
        "fused_embedding": fused[0],
        "predicted_class_index": predicted,
        "predicted_class_label": str(classes[predicted]),
        "squared_prototype_distances": distances[0],
    }


def staged_parity_row(
    *,
    family: str,
    name: str,
    capture: np.ndarray,
    prefilters: Mapping[int, noise_prefilter.NoisePrefilter],
    rejector: base.Rejector,
    classifier: base.Rejector,
    components: Sequence[tuple[str, float, KnownOnlyLOFOpenSet]],
    policy: FrozenV3OpenSet,
    composite: Any,
    rejector_feature_mean: np.ndarray,
    rejector_feature_std: np.ndarray,
    classifier_feature_mean: np.ndarray,
    classifier_feature_std: np.ndarray,
    classes: Sequence[str],
    patch_length: int,
    patch_count: int,
    target_frac: float,
    device: torch.device,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    capture = np.asarray(capture, dtype=np.complex128)
    stage_one = stage_one_row(
        prefilters,
        capture,
        patch_length=patch_length,
        target_frac=target_frac,
    )
    packed, raw_features, _context = td_preprocess.preprocess(
        capture,
        patch_length=patch_length,
        patch_count=patch_count,
        target_frac=target_frac,
    )
    packed = np.asarray(packed, dtype=np.float32)
    raw_features = np.asarray(raw_features, dtype=np.float32)
    rejector_standardized = (
        (raw_features - rejector_feature_mean) / rejector_feature_std
    ).astype(np.float32)
    classifier_standardized: np.ndarray | None = None
    classifier_result: dict[str, Any] | None = None

    if stage_one["gated"]:
        stage_two = None
        stage_one_survivor_rank = None
        composite_score = None
        staged_score = float(staged.STAGE_ONE_SCORE_OFFSET + stage_one["rank"])
        rejected_stage: int | None = 1
        decision = "noise"
        rejector_winner_index = None
        rejector_winner_label = None
    else:
        stage_two = stage_two_row(
            rejector,
            components,
            policy,
            packed,
            rejector_standardized,
            classes,
            device,
        )
        # Staged policy version 2: the survivor decision is the COMPOSITE
        # against the composite q95 threshold.  Both terms come from the
        # loaded CompositeSurvivorPolicy itself, so the fixture cannot drift
        # from the real Python path.
        stage_one_raw = np.asarray([stage_one["score"]], dtype=np.float64)
        stage_one_survivor_rank = float(
            composite.stage_one_survivor_rank(stage_one_raw)[0]
        )
        composite_score = float(
            composite.composite(
                np.asarray([stage_two["score"]], dtype=np.float64),
                stage_one_raw,
            )[0]
        )
        if composite_score != max(stage_two["score"], stage_one_survivor_rank):
            raise AssertionError(
                "decomposed composite differs from CompositeSurvivorPolicy"
            )
        staged_score = composite_score
        rejected = bool(composite_score > composite.threshold)
        rejected_stage = 2 if rejected else None
        rejector_winner_index = int(stage_two["predicted_class_index"])
        rejector_winner_label = str(stage_two["predicted_class_label"])
        if rejected:
            decision = "unknown"
        else:
            classifier_standardized = (
                (raw_features - classifier_feature_mean)
                / classifier_feature_std
            ).astype(np.float32)
            classifier_result = classifier_row(
                classifier,
                packed,
                classifier_standardized,
                classes,
                device,
            )
            decision = str(classifier_result["predicted_class_label"])

    if (classifier_result is not None) != (rejected_stage is None):
        raise AssertionError(
            "accepted-known classifier execution is not exactly acceptance-gated"
        )

    row = {
        "family": family,
        "name": name,
        "capture_length": int(len(capture)),
        "iq": {
            "in_phase": np.real(capture),
            "quadrature": np.imag(capture),
        },
        "stage_one": stage_one,
        "packed_iq": {
            "in_phase": packed[0],
            "quadrature": packed[1],
        },
        "raw_features": raw_features,
        # Legacy parity readers interpret standardized_features as the staged
        # rejector input.  The explicit role-named field is authoritative.
        "standardized_features": rejector_standardized,
        "rejector_standardized_features": rejector_standardized,
        "classifier_standardized_features": classifier_standardized,
        "stage_two": stage_two,
        "rejector_closed_winner_index": rejector_winner_index,
        "rejector_closed_winner_label": rejector_winner_label,
        "classifier": classifier_result,
        "classifier_executed": classifier_result is not None,
        "stage_one_survivor_rank": stage_one_survivor_rank,
        "composite_score": composite_score,
        "staged_threshold": float(composite.threshold),
        "staged_score": staged_score,
        "rejected_stage": rejected_stage,
        "decision_label": decision,
    }
    if extra:
        row.update(dict(extra))
    return row


# ---------------------------------------------------------------------------
# fixture populations
# ---------------------------------------------------------------------------


def select_known_captures(
    seed: int, classes: Sequence[str]
) -> list[dict[str, Any]]:
    """One SELECTION-split capture per class, deterministic, nonzero prefix."""
    source = invariant_data.load(
        patch_length=64,
        patch_count=16,
        target_frac=0.5,
        model_seed=seed,
        build_if_missing=False,
    )
    if list(source["classes"]) != list(classes):
        raise RuntimeError(
            "cache class order differs from the runtime bundle class order"
        )
    manifest = dev._load_manifest()
    raw = dev._raw_memmap(manifest)
    labels = np.asarray(source["yva"], dtype=np.int64)
    indices = np.asarray(source["va_idx"], dtype=np.int64)
    minimum_length = min(FIXTURE_LENGTHS)
    selected: list[dict[str, Any]] = []
    for class_index, class_name in enumerate(classes):
        positions = np.flatnonzero(labels == class_index)
        chosen = None
        for position in positions:
            capture = dev._complex_row(raw, int(indices[position]))
            if float(np.max(np.abs(capture[:minimum_length]))) > 0.0:
                chosen = (int(position), capture)
                break
        if chosen is None:
            raise RuntimeError(
                f"no selection row of class {class_name} has a nonzero "
                f"{minimum_length}-sample prefix"
            )
        position, capture = chosen
        selected.append(
            {
                "class_index": class_index,
                "class_label": str(class_name),
                "selection_position": position,
                "corpus_index": int(indices[position]),
                "capture": capture,
            }
        )
    del raw
    return selected


def draw_novelty_captures(seed: int) -> dict[str, list[np.ndarray]]:
    """Noise and chirp parity rows at the fixture's longest length."""
    validate_parity_seed(seed)
    longest = CAUSAL_PREFIX_FIXTURE_LENGTH
    rng = np.random.default_rng(int(seed))
    captures: dict[str, list[np.ndarray]] = {}
    for family in ("noise", "chirp"):
        generator = NOVELTY[family]
        captures[family] = [
            np.asarray(generator(longest, rng), dtype=np.complex128)
            for _ in range(NOVELTY_ROWS_PER_FAMILY)
        ]
    return captures


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rejector-bundle-dir", type=Path, required=True)
    parser.add_argument("--classifier-bundle-dir", type=Path, required=True)
    parser.add_argument("--rejector-fusion-dir", type=Path, required=True)
    parser.add_argument("--classifier-fusion-dir", type=Path, required=True)
    parser.add_argument("--rejector-browser-export", type=Path, required=True)
    parser.add_argument("--classifier-browser-export", type=Path, required=True)
    parser.add_argument("--staged-dir", type=Path, required=True)
    parser.add_argument("--prefilter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--parity-seed", type=int, default=PARITY_NOVELTY_SEED
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = assemble.reject_sealed_path(Path(args.output_dir), "output")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"{output} is not empty; refusing to replace staged runtime assets"
        )

    rejector_bundle = load_bundle(
        Path(args.rejector_bundle_dir),
        expected_role=REJECTOR_RUNTIME_ROLE,
    )
    classifier_bundle = load_bundle(
        Path(args.classifier_bundle_dir),
        expected_role=CLASSIFIER_RUNTIME_ROLE,
    )
    if (
        rejector_bundle["manifest_sha256"]
        == classifier_bundle["manifest_sha256"]
    ):
        raise RuntimeError("rejector and classifier runtime bundles are identical")

    rejector_manifest = rejector_bundle["manifest"]
    classifier_manifest = classifier_bundle["manifest"]
    classes = list(rejector_manifest["classification"]["classes"])
    if list(classifier_manifest["classification"]["classes"]) != classes:
        raise RuntimeError("rejector and classifier class orders differ")
    frontend_meta = rejector_manifest["frontend"]
    classifier_frontend = classifier_manifest["frontend"]
    for key in (
        "version",
        "patch_length",
        "patch_count",
        "target_frac",
        "packed_length",
        "uses_frequency_transform",
    ):
        if classifier_frontend.get(key) != frontend_meta.get(key):
            raise RuntimeError(
                f"rejector/classifier frontend field {key!r} differs"
            )
    patch_length = int(frontend_meta["patch_length"])
    patch_count = int(frontend_meta["patch_count"])
    target_frac = float(frontend_meta["target_frac"])

    rejector_browser = load_browser_fusion_export(
        Path(args.rejector_browser_export),
        expected_role=REJECTOR_RUNTIME_ROLE,
        expected_weights_name=REJECTOR_WEIGHTS_NAME,
        expected_probe_name=REJECTOR_PROBE_NAME,
        expected_bundle_manifest_sha256=rejector_bundle["manifest_sha256"],
    )
    classifier_browser = load_browser_fusion_export(
        Path(args.classifier_browser_export),
        expected_role=CLASSIFIER_RUNTIME_ROLE,
        expected_weights_name=CLASSIFIER_WEIGHTS_NAME,
        expected_probe_name=CLASSIFIER_PROBE_NAME,
        expected_bundle_manifest_sha256=classifier_bundle["manifest_sha256"],
    )
    if rejector_browser["weights_sha256"] == classifier_browser["weights_sha256"]:
        raise RuntimeError("role browser assets are anonymously interchangeable")
    for label, browser, bundle in (
        ("rejector", rejector_browser, rejector_bundle),
        ("classifier", classifier_browser, classifier_bundle),
    ):
        recorded_probe = bundle["manifest"]["assets"]["probe_fixture.json"][
            "sha256"
        ]
        if browser["probe_sha256"] != recorded_probe:
            raise RuntimeError(
                f"{label} browser probe is not the exact runtime-bundle probe"
            )

    prefilter_dir = assemble.reject_sealed_path(
        Path(args.prefilter_dir), "prefilter bundles"
    )
    prefilters = noise_prefilter.load_prefilter_set(prefilter_dir)
    prefilter_sha = noise_prefilter.prefilter_set_sha256(prefilter_dir)
    for length in FIXTURE_LENGTHS:
        if length not in prefilters:
            raise RuntimeError(f"no prefilter bundle for capture length {length}")

    staged_dir = assemble.reject_sealed_path(
        Path(args.staged_dir), "staged validation artifact"
    )
    staged_metrics_path = staged_dir / "openset_metrics.json"
    with staged_metrics_path.open(encoding="utf-8") as handle:
        staged_metrics = json.load(handle)
    validate_staged_evidence_report(staged_metrics)
    verified_staged_hashes = verify_staged_artifact_hashes(
        staged_metrics, staged_dir
    )

    components, policy, staged_hashes = load_stage_two_state(
        staged_dir
    )
    composite, composite_sha = load_composite_state(
        staged_dir, policy
    )
    staged_hashes = dict(staged_hashes)
    staged_hashes[staged.COMPOSITE_POLICY_FILENAME] = composite_sha
    if staged_hashes != verified_staged_hashes:
        raise AssertionError(
            "loaded staged assets differ from the report-verified assets"
        )
    recorded_prefilter_dir = staged_metrics.get("stage_one", {}).get("directory")
    if recorded_prefilter_dir is None or (
        Path(recorded_prefilter_dir).resolve() != prefilter_dir.resolve()
    ):
        raise RuntimeError(
            f"staged artifact used prefilters {recorded_prefilter_dir}, not "
            f"{prefilter_dir}"
        )
    recorded_set_sha = staged_metrics.get("stage_one", {}).get("set_sha256")
    if recorded_set_sha != prefilter_sha:
        raise RuntimeError(
            "prefilter set SHA differs from the staged artifact's record"
        )

    rejector_artifact = base.load_fusion_artifact(
        Path(args.rejector_fusion_dir)
    )
    classifier_artifact = base.load_fusion_artifact(
        Path(args.classifier_fusion_dir)
    )
    if (
        rejector_artifact.directory_sha256
        == classifier_artifact.directory_sha256
    ):
        raise RuntimeError("rejector and classifier fusion artifacts are identical")
    recorded_fusion = staged_metrics.get("fusion", {})
    if not isinstance(recorded_fusion, Mapping):
        raise RuntimeError("staged artifact carries no fusion provenance")
    recorded_fusion_dir = recorded_fusion.get("directory")
    if recorded_fusion_dir is None or (
        Path(str(recorded_fusion_dir)).resolve()
        != rejector_artifact.directory.resolve()
    ):
        raise RuntimeError(
            f"staged artifact was fit against {recorded_fusion_dir}, not "
            f"{rejector_artifact.directory}"
        )
    if (
        recorded_fusion.get("directory_sha256")
        != rejector_artifact.directory_sha256
    ):
        raise RuntimeError(
            "staged rejector was not fitted against the loaded rejector fusion"
        )

    bind_bundle_to_fusion(rejector_bundle, rejector_artifact)
    bind_bundle_to_fusion(classifier_bundle, classifier_artifact)
    device = torch.device("cpu")
    rejector_fusion = build_bundle_fusion(rejector_bundle, device)
    classifier_fusion = build_bundle_fusion(classifier_bundle, device)
    rejector = base.Rejector(
        nets={
            "real": rejector_fusion.real_branch,
            "complex": rejector_fusion.complex_branch,
        },
        real_center=rejector_artifact.real_center,
        complex_center=rejector_artifact.complex_center,
        weight_real=rejector_artifact.weight_real,
        prototypes=rejector_artifact.prototypes,
        lof_components=list(components),
        policy=policy,
    )
    classifier = base.Rejector(
        nets={
            "real": classifier_fusion.real_branch,
            "complex": classifier_fusion.complex_branch,
        },
        real_center=classifier_artifact.real_center,
        complex_center=classifier_artifact.complex_center,
        weight_real=classifier_artifact.weight_real,
        prototypes=classifier_artifact.prototypes,
        # These fields are deliberately unreachable from classifier_row.
        lof_components=[],
        policy=policy,
    )

    rejector_feature_mean = np.asarray(
        rejector_bundle["arrays"]["feature_mean.npy"], dtype=np.float32
    )
    rejector_feature_std = np.asarray(
        rejector_bundle["arrays"]["feature_std.npy"], dtype=np.float32
    )
    classifier_feature_mean = np.asarray(
        classifier_bundle["arrays"]["feature_mean.npy"], dtype=np.float32
    )
    classifier_feature_std = np.asarray(
        classifier_bundle["arrays"]["feature_std.npy"], dtype=np.float32
    )

    frontend_payload = {
        "version": str(frontend_meta["version"]),
        "patch_length": patch_length,
        "patch_count": patch_count,
        "target_frac": target_frac,
        "packed_length": patch_length * patch_count,
        "uses_frequency_transform": False,
    }

    staged_validation_sha = _sha256(staged_metrics_path)
    provenance = {
        "candidate_id": CANDIDATE_ID,
        "runtime_roles": {
            "rejector": {
                "runtime_role": REJECTOR_RUNTIME_ROLE,
                "runtime_bundle": rejector_bundle["directory"].name,
                "runtime_bundle_manifest_sha256": (
                    rejector_bundle["manifest_sha256"]
                ),
                "fusion_artifact": rejector_artifact.directory.name,
                "fusion_directory_sha256": (
                    rejector_artifact.directory_sha256
                ),
                "browser_export_manifest_sha256": (
                    rejector_browser["manifest_sha256"]
                ),
                "browser_asset": REJECTOR_WEIGHTS_NAME,
                "browser_asset_sha256": (
                    rejector_browser["weights_sha256"]
                ),
            },
            "classifier": {
                "runtime_role": CLASSIFIER_RUNTIME_ROLE,
                "runtime_bundle": classifier_bundle["directory"].name,
                "runtime_bundle_manifest_sha256": (
                    classifier_bundle["manifest_sha256"]
                ),
                "fusion_artifact": classifier_artifact.directory.name,
                "fusion_directory_sha256": (
                    classifier_artifact.directory_sha256
                ),
                "browser_export_manifest_sha256": (
                    classifier_browser["manifest_sha256"]
                ),
                "browser_asset": CLASSIFIER_WEIGHTS_NAME,
                "browser_asset_sha256": (
                    classifier_browser["weights_sha256"]
                ),
            },
        },
        # Compatibility keys identify the fusion against which the staged
        # known/unknown policy was fitted: always the rejector, never the
        # accepted-known classifier.
        "runtime_bundle": rejector_bundle["directory"].name,
        "runtime_bundle_manifest_sha256": (
            rejector_bundle["manifest_sha256"]
        ),
        "fusion_artifact": rejector_artifact.directory.name,
        "fusion_directory_sha256": rejector_artifact.directory_sha256,
        "staged_artifact": staged_dir.name,
        "staged_artifact_sha256": staged_hashes,
        "staged_validation_report_sha256": staged_validation_sha,
        "staged_validation_status": staged_metrics["status"],
        "staged_validation_role": staged_metrics["role"],
        "staged_validation_all_pass": staged_metrics["all_pass"],
        "prefilter_bundles": prefilter_dir.parent.name + "/" + prefilter_dir.name,
        "prefilter_set_sha256": prefilter_sha,
        "exporter": Path(__file__).name,
        "exporter_sha256": _sha256(Path(__file__)),
        "development_only": True,
        "release_evidence": False,
        "sealed_release_data_used": 0,
        "consumed_test_rows_used": 0,
        "release_seed_not_spent": 20260735,
    }

    weights = openset_weights_payload(
        prefilters,
        prefilter_sha,
        components,
        policy,
        composite,
        frontend_payload,
        provenance,
    )
    # ---- parity rows ------------------------------------------------------
    known = select_known_captures(rejector_artifact.seed, classes)
    novelty = draw_novelty_captures(int(args.parity_seed))

    def process(family: str, name: str, capture: np.ndarray, extra=None):
        return staged_parity_row(
            family=family,
            name=name,
            capture=capture,
            prefilters=prefilters,
            rejector=rejector,
            classifier=classifier,
            components=components,
            policy=policy,
            composite=composite,
            rejector_feature_mean=rejector_feature_mean,
            rejector_feature_std=rejector_feature_std,
            classifier_feature_mean=classifier_feature_mean,
            classifier_feature_std=classifier_feature_std,
            classes=classes,
            patch_length=patch_length,
            patch_count=patch_count,
            target_frac=target_frac,
            device=device,
            extra=extra,
        )

    rows: list[dict[str, Any]] = []
    for length in FIXTURE_LENGTHS:
        for item in known:
            rows.append(
                process(
                    "known",
                    f"known-{item['class_label']}-N{length}",
                    item["capture"][:length],
                    extra={
                        "class_label": item["class_label"],
                        "class_index": item["class_index"],
                        "corpus_index": item["corpus_index"],
                    },
                )
            )
        for family in ("noise", "chirp"):
            for index, capture in enumerate(novelty[family]):
                rows.append(
                    process(
                        family,
                        f"{family}-{index}-N{length}",
                        capture[:length],
                    )
                )

    # Two compact, Python-generated N32768 rows bind the browser runtime to
    # the exact causal-prefix semantics used by validation and release: one
    # row must short-circuit at stage 1 and one must survive into the full
    # N32768 classifier/stage-2 path.  Candidate selection here is fixture
    # coverage only, on the explicitly non-evidence parity seed.
    long_rows: dict[str, dict[str, Any]] = {}
    for family in ("noise", "chirp"):
        for index, capture in enumerate(novelty[family]):
            row = process(
                family,
                f"{family}-{index}-N{CAUSAL_PREFIX_FIXTURE_LENGTH}",
                capture,
            )
            stage_one_row_record = row["stage_one"]
            if (
                stage_one_row_record["capture_length"]
                != CAUSAL_PREFIX_FIXTURE_LENGTH
                or stage_one_row_record["evaluated_length"]
                != max(FIXTURE_LENGTHS)
                or stage_one_row_record["causal_prefix_applied"] is not True
            ):
                raise AssertionError(
                    "N32768 parity row did not apply the N16384 causal prefix"
                )
            decision_path = (
                "gated" if row["rejected_stage"] == 1 else "survivor"
            )
            long_rows.setdefault(decision_path, row)
            if set(long_rows) == {"gated", "survivor"}:
                break
        if set(long_rows) == {"gated", "survivor"}:
            break
    if set(long_rows) != {"gated", "survivor"}:
        raise RuntimeError(
            "parity draw did not produce both a gated and a surviving N32768 "
            "causal-prefix row"
        )
    for key, fixed_name in (
        ("gated", CAUSAL_PREFIX_GATED_ROW_NAME),
        ("survivor", CAUSAL_PREFIX_SURVIVOR_ROW_NAME),
    ):
        row = long_rows[key]
        row["parity_source_name"] = row["name"]
        row["name"] = fixed_name
        rows.append(row)

    # Pair both role probes by exact name, length and raw bytes.  Their
    # expected winners are independently asserted on each reachable role.
    probe_rows: list[dict[str, Any]] = []
    probe_cases: list[dict[str, Any]] = []
    rejector_cases = {
        str(case["name"]): case for case in rejector_bundle["probe"]["cases"]
    }
    classifier_cases = {
        str(case["name"]): case for case in classifier_bundle["probe"]["cases"]
    }
    if set(rejector_cases) != set(classifier_cases):
        raise RuntimeError("rejector and classifier probe case names differ")
    for case_name in rejector_cases:
        case = rejector_cases[case_name]
        classifier_case = classifier_cases[case_name]
        if (
            int(classifier_case["length"]) != int(case["length"])
            or classifier_case["raw"] != case["raw"]
        ):
            raise RuntimeError(
                f"role probe case {case_name!r} does not share exact raw input"
            )
        rejector_expected = case["expected"]
        classifier_expected = classifier_case["expected"]
        expected_fields = (
            "raw_features",
            "standardized_features",
            "real_embedding",
            "complex_embedding",
            "fused_embedding",
            "squared_prototype_distances",
            "closed_winner_index",
            "closed_label",
        )
        role_expected = {
            "rejector": {
                key: (
                    int(rejector_expected[key])
                    if key == "closed_winner_index"
                    else str(rejector_expected[key])
                    if key == "closed_label"
                    else rejector_expected[key]
                )
                for key in expected_fields
            },
            "classifier": {
                key: (
                    int(classifier_expected[key])
                    if key == "closed_winner_index"
                    else str(classifier_expected[key])
                    if key == "closed_label"
                    else classifier_expected[key]
                )
                for key in expected_fields
            },
        }
        probe_cases.append(
            {
                "name": case_name,
                "length": int(case["length"]),
                "raw": case["raw"],
                # Flat expected remains the rejector anchor for legacy parity
                # consumers; role_expected is the dual authoritative record.
                "expected": role_expected["rejector"],
                "role_expected": role_expected,
            }
        )
        if int(case["length"]) not in prefilters:
            continue
        capture = np.asarray(
            case["raw"]["in_phase"], dtype=np.float64
        ) + 1j * np.asarray(case["raw"]["quadrature"], dtype=np.float64)
        row = process("probe", case_name, capture)
        if not row["stage_one"]["gated"]:
            if (
                row["rejector_closed_winner_label"]
                != rejector_expected["closed_label"]
            ):
                raise RuntimeError(
                    f"probe case {case_name} rejector winner diverged from its "
                    "runtime-bundle probe"
                )
            if row["rejected_stage"] is None and (
                row["classifier"]["predicted_class_label"]
                != classifier_expected["closed_label"]
            ):
                raise RuntimeError(
                    f"probe case {case_name} classifier winner diverged from "
                    "its runtime-bundle probe"
                )
        probe_rows.append(row)

    all_rows = rows + probe_rows
    counts = {
        "rows": len(all_rows),
        "known": sum(1 for row in rows if row["family"] == "known"),
        "noise": sum(1 for row in rows if row["family"] == "noise"),
        "chirp": sum(1 for row in rows if row["family"] == "chirp"),
        "probe": len(probe_rows),
        "gated_stage_one": sum(
            1 for row in all_rows if row["rejected_stage"] == 1
        ),
        "rejected_stage_two": sum(
            1 for row in all_rows if row["rejected_stage"] == 2
        ),
        "accepted": sum(
            1 for row in all_rows if row["rejected_stage"] is None
        ),
        "classifier_executed": sum(
            1 for row in all_rows if row["classifier_executed"]
        ),
        "accepted_role_winner_disagreements": sum(
            1
            for row in all_rows
            if row["rejected_stage"] is None
            and row["rejector_closed_winner_label"]
            != row["classifier"]["predicted_class_label"]
        ),
    }
    if counts["classifier_executed"] != counts["accepted"]:
        raise AssertionError("classifier execution count differs from accept count")
    disagreement_rows = [
        row["name"]
        for row in all_rows
        if row["rejected_stage"] is None
        and row["rejector_closed_winner_label"]
        != row["classifier"]["predicted_class_label"]
    ]
    for row in all_rows:
        if row["rejected_stage"] == 1:
            public = {
                "outcome": "noise",
                "label": "noise",
                "knownLabel": None,
                "knownWinnerIndex": None,
                "squaredPrototypeDistances": None,
            }
        elif row["rejected_stage"] == 2:
            public = {
                "outcome": "unknown",
                "label": "unknown",
                "knownLabel": None,
                "knownWinnerIndex": None,
                "squaredPrototypeDistances": None,
            }
        else:
            public = {
                "outcome": "known",
                "label": row["classifier"]["predicted_class_label"],
                "knownLabel": row["classifier"]["predicted_class_label"],
                "knownWinnerIndex": row["classifier"]["predicted_class_index"],
                "squaredPrototypeDistances": row["classifier"][
                    "squared_prototype_distances"
                ],
            }
        row["public_result"] = public

    parity = {
        "schema": PARITY_SCHEMA,
        "schema_version": 3,
        "status": "staging_not_release",
        "candidate_id": CANDIDATE_ID,
        "contract": weights["contract"],
        "classes": classes,
        "frontend": frontend_payload,
        "fixture_lengths": list(REQUIRED_VALIDATION_PREFIX_LENGTHS),
        "causal_prefix_parity": {
            "capture_length": CAUSAL_PREFIX_FIXTURE_LENGTH,
            "evaluated_length": max(FIXTURE_LENGTHS),
            "row_names": [
                CAUSAL_PREFIX_GATED_ROW_NAME,
                CAUSAL_PREFIX_SURVIVOR_ROW_NAME,
            ],
            "covers_gated_and_survivor_paths": True,
        },
        "parity_novelty_seed": int(args.parity_seed),
        "parity_seed_is_evaluation_evidence": False,
        "known_rows_population": "development selection split (scored, never fit)",
        "counts": counts,
        "accepted_role_winner_disagreement_rows": disagreement_rows,
        "role_contract": {
            "rejector": REJECTOR_RUNTIME_ROLE,
            "classifier": CLASSIFIER_RUNTIME_ROLE,
            "classifier_runs_only_after_rejector_acceptance": True,
            "public_known_label_from_classifier_only": True,
        },
        "policy_version": staged.STAGED_POLICY_VERSION,
        "survivor_score": staged.COMPOSITE_SURVIVOR_SCORE,
        "staged_threshold": float(composite.threshold),
        "stage_two_threshold": float(policy.threshold),
        "tolerance": {
            "stage_one_features_abs": 1e-8,
            "stage_one_score_abs": 1e-8,
            "embedding_abs": 1e-5,
            "rank_abs": 1e-9,
            "raw_lof_relative": 1e-9,
            "decisions": "exact, including which stage rejected each row",
        },
        "staged_score_note": staged.STAGED_SCORE_NOTE,
        "rows": all_rows,
        "probe_cases": probe_cases,
        "provenance": provenance,
    }

    policy_raw = json_bytes(weights, compact=True)
    if len(policy_raw) >= MAX_OPENSET_POLICY_BYTES:
        raise RuntimeError(
            f"compact open-set policy is {len(policy_raw)} bytes; must be below "
            f"{MAX_OPENSET_POLICY_BYTES}"
        )
    policy_sha = hashlib.sha256(policy_raw).hexdigest()
    binding = dual_binding_payload(
        frontend=frontend_payload,
        rejector_asset_sha256=rejector_browser["weights_sha256"],
        classifier_asset_sha256=classifier_browser["weights_sha256"],
        openset_asset_sha256=policy_sha,
        rejector_bundle_manifest_sha256=rejector_bundle["manifest_sha256"],
        classifier_bundle_manifest_sha256=classifier_bundle["manifest_sha256"],
        rejector_fusion_directory_sha256=(
            rejector_artifact.directory_sha256
        ),
        classifier_fusion_directory_sha256=(
            classifier_artifact.directory_sha256
        ),
        staged_validation_report_sha256=staged_validation_sha,
        staged_artifacts_sha256=staged_hashes,
        novelty_seeds=staged_metrics["seeds"]["novelty_seeds"],
    )

    # All candidate and parity self-checks have passed.  Only now may the
    # exporter materialize its fresh output directory.
    output.mkdir(parents=True, exist_ok=True)
    write_bytes(output / OPENSET_POLICY_NAME, policy_raw)
    write_json(output / DUAL_BINDING_NAME, binding, compact=True)
    write_json(output / PARITY_NAME, parity)

    output_manifest = {
        "schema": OPENSET_MANIFEST_SCHEMA,
        "schema_version": 1,
        "status": "staging_not_release",
        "candidate_id": CANDIDATE_ID,
        "outputs": {
            name: {
                "bytes": (output / name).stat().st_size,
                "sha256": _sha256(output / name),
            }
            for name in (OPENSET_POLICY_NAME, DUAL_BINDING_NAME, PARITY_NAME)
        },
        "inputs": provenance,
        "role_exports": {
            "rejector": {
                "manifest_sha256": rejector_browser["manifest_sha256"],
                "runtime_role": REJECTOR_RUNTIME_ROLE,
                "weights": {
                    "name": REJECTOR_WEIGHTS_NAME,
                    "bytes": rejector_browser["weights_bytes"],
                    "sha256": rejector_browser["weights_sha256"],
                },
                "probe": {
                    "name": REJECTOR_PROBE_NAME,
                    "bytes": rejector_browser["probe_bytes"],
                    "sha256": rejector_browser["probe_sha256"],
                },
            },
            "classifier": {
                "manifest_sha256": classifier_browser["manifest_sha256"],
                "runtime_role": CLASSIFIER_RUNTIME_ROLE,
                "weights": {
                    "name": CLASSIFIER_WEIGHTS_NAME,
                    "bytes": classifier_browser["weights_bytes"],
                    "sha256": classifier_browser["weights_sha256"],
                },
                "probe": {
                    "name": CLASSIFIER_PROBE_NAME,
                    "bytes": classifier_browser["probe_bytes"],
                    "sha256": classifier_browser["probe_sha256"],
                },
            },
        },
        "size_contract": {
            "asset": OPENSET_POLICY_NAME,
            "maximum_bytes_exclusive": MAX_OPENSET_POLICY_BYTES,
            "actual_bytes": len(policy_raw),
            "compact_json": True,
        },
        "counts": counts,
    }
    write_json(output / MANIFEST_NAME, output_manifest)
    print(
        f"[v3 openset export] wrote {output}: rows={counts['rows']} "
        f"gated={counts['gated_stage_one']} stage2={counts['rejected_stage_two']} "
        f"accepted={counts['accepted']}",
        flush=True,
    )
    return output_manifest


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
