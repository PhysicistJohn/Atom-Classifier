"""Unit suite for the seed-20260736 policy-v4/q97 release preflight.

What is under test is the property that makes the preflight worth running at
all: a pin that does not match disk, a tampered artifact, a missing evaluator,
a wrong runtime or a spent seed must each surface as an itemised FAIL and a
NO-GO, while the real pinned inputs and strict evaluator must pass every check.

Hygiene properties asserted explicitly:

* the preflight consumes nothing: no release root is created, nothing is
  written under ``training/artifacts/releases``, and the report says so in
  machine-readable form;
* the source-tree digest is bit-compatible with the launcher's JavaScript
  walk (same skip list, same separator byte, symlink refusal), locked by an
  independent inline reimplementation rather than by calling the function
  under test;
* the reconstructed generation command is exactly the sealed mechanics
  re-aimed at seed 20260736 with 192 rows per class, and is never executed;
* the q99 failure, q97 design pass, independent validation, candidate,
  browser/package assets and evaluator protocol are immutable hash-pinned
  evidence; replacing any final pin with a placeholder still fails closed;
* an existing report path is refused before any check runs, and a report can
  never be written under the release tree.

Synthetic fixtures live in ``tempfile`` directories.  A handful of
integration tests read the real frozen artifacts in this repository, which is
deliberate: the preflight's one job is to describe this machine.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import unittest
from unittest import mock


HERE = os.path.dirname(os.path.abspath(__file__))
_V2_DIR = os.path.dirname(HERE)
_ZPAB_DIR = os.path.dirname(_V2_DIR)
_TRAINING_DIR = os.path.dirname(_ZPAB_DIR)
for _path in (_TRAINING_DIR, _ZPAB_DIR, _V2_DIR, HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import preflight_v3_release as preflight  # noqa: E402


class TempDirTestCase(unittest.TestCase):
    """``unittest`` on this interpreter predates ``enterContext``."""

    def tmpdir(self) -> Path:
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        return Path(holder.name)


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _by_name(checks: list) -> dict:
    if checks and isinstance(checks[0], dict):
        return {item["name"]: item for item in checks}
    return {
        name: {"name": name, "passed": passed, "detail": detail}
        for name, passed, detail in checks
    }


class SourceTreeDigestTest(TempDirTestCase):
    """The digest must match the launcher's algorithm, not merely be stable."""

    def _tree(self) -> Path:
        root = Path(self.tmpdir())
        _write(root / "b.txt", b"beta")
        _write(root / "a.txt", b"alpha")
        _write(root / "sub" / "c.bin", bytes(range(7)))
        return root

    def test_matches_independent_reimplementation(self) -> None:
        root = self._tree()
        digest, count = preflight.source_tree_digest(root)
        # Independent construction, matching the JavaScript walk: sorted
        # names, path + 0x00 + contents, '/' separators.
        expected = hashlib.sha256()
        for relative in ("a.txt", "b.txt", "sub/c.bin"):
            expected.update(relative.encode("utf-8"))
            expected.update(b"\x00")
            expected.update((root / relative).read_bytes())
        self.assertEqual(digest, expected.hexdigest())
        self.assertEqual(count, 3)

    def test_skip_list_is_invisible(self) -> None:
        root = self._tree()
        before = preflight.source_tree_digest(root)
        _write(root / "node_modules" / "x.js", b"ignored")
        _write(root / "dist" / "index.js", b"ignored")
        _write(root / "RELEASE_SOURCE_PROVENANCE.json", b"{}")
        _write(root / "sub" / "node_modules" / "y.js", b"ignored")
        self.assertEqual(preflight.source_tree_digest(root), before)

    def test_content_and_name_sensitivity(self) -> None:
        root = self._tree()
        before, _ = preflight.source_tree_digest(root)
        (root / "a.txt").write_bytes(b"alphA")
        after, _ = preflight.source_tree_digest(root)
        self.assertNotEqual(before, after)
        (root / "a.txt").rename(root / "a2.txt")
        renamed, _ = preflight.source_tree_digest(root)
        self.assertNotEqual(after, renamed)

    def test_symlink_is_refused(self) -> None:
        root = self._tree()
        os.symlink(root / "a.txt", root / "link.txt")
        with self.assertRaisesRegex(RuntimeError, "symlink"):
            preflight.source_tree_digest(root)


class FusionArtifactCheckTest(TempDirTestCase):
    def _fake_fusion(self) -> Path:
        root = Path(self.tmpdir())
        artifacts: dict = {}
        for key, _sha_key in preflight.FUSION_ASSET_KEYS:
            payload = f"payload-{key}".encode("utf-8")
            filename = f"{key}.bin"
            _write(root / filename, payload)
            artifacts[key] = filename
            artifacts[f"{key}_sha256"] = _sha(payload)
        # The real record uses specific *_sha256 key names; mirror them.
        metrics = {
            "artifacts": {
                file_key: artifacts[file_key]
                for file_key, _ in preflight.FUSION_ASSET_KEYS
            }
            | {
                sha_key: artifacts[f"{file_key}_sha256"]
                for file_key, sha_key in preflight.FUSION_ASSET_KEYS
            },
            "consumed_test_rows_used": 0,
            "sealed_release_data_used": 0,
        }
        _write(
            root / "dev_metrics.json",
            json.dumps(metrics, sort_keys=True).encode("utf-8"),
        )
        return root

    def test_intact_artifact_passes(self) -> None:
        checks = _by_name(preflight.check_fusion_artifact(self._fake_fusion()))
        self.assertTrue(all(item["passed"] for item in checks.values()), checks)
        self.assertIn("fusion.asset_sha256", checks)

    def test_tampered_asset_fails(self) -> None:
        root = self._fake_fusion()
        (root / "state_dict.bin").write_bytes(b"tampered")
        checks = _by_name(preflight.check_fusion_artifact(root))
        self.assertFalse(checks["fusion.asset_sha256"]["passed"])
        self.assertIn("state_dict.bin", checks["fusion.asset_sha256"]["detail"])

    def test_nonzero_hygiene_counter_fails(self) -> None:
        root = self._fake_fusion()
        metrics = json.loads((root / "dev_metrics.json").read_text())
        metrics["consumed_test_rows_used"] = 12
        (root / "dev_metrics.json").write_text(json.dumps(metrics))
        checks = _by_name(preflight.check_fusion_artifact(root))
        self.assertFalse(checks["fusion.consumed_test_rows_used"]["passed"])


class RuntimeBundleCheckTest(TempDirTestCase):
    """Manifest-level bundle checks on a synthetic bundle, no torch needed."""

    def _pins(self) -> dict:
        return copy.deepcopy(preflight.DEFAULT_PINS)

    def _fake_bundle_and_fusion(self) -> tuple[Path, Path, dict]:
        base = Path(self.tmpdir())
        fusion = base / "fusion"
        dev_metrics = b'{"artifacts": {}}'
        _write(fusion / "dev_metrics.json", dev_metrics)
        bundle = base / "bundle"
        asset_payload = b"asset-bytes"
        _write(bundle / "weights.npy", asset_payload)
        pins = self._pins()
        frontend_file = "training/time_domain_geometry.py"
        manifest = {
            "schema": pins["bundle_schema"],
            "schema_version": pins["bundle_schema_version"],
            "assets": {
                "weights.npy": {
                    "sha256": _sha(asset_payload),
                    "bytes": len(asset_payload),
                }
            },
            "provenance": {"source_dev_metrics_sha256": _sha(dev_metrics)},
            "frontend": {
                "source_sha256": {
                    frontend_file: preflight._sha256(
                        preflight.REPO / frontend_file
                    )
                }
            },
            "rejection": {
                "state": "unset",
                "required_contract": {
                    "policy_schema": pins["frozen_policy"]["schema"],
                    "policy_kind": pins["frozen_policy"]["kind"],
                    "geometry_feature": pins["frozen_policy"]["geometry_feature"],
                    "v2_weight": pins["frozen_policy"]["v2_weight"],
                    "geometry_weight": pins["frozen_policy"]["geometry_weight"],
                    "threshold_quantile": pins["frozen_policy"][
                        "threshold_quantile"
                    ],
                },
            },
        }
        _write(
            bundle / "bundle_manifest.json",
            json.dumps(manifest, sort_keys=True).encode("utf-8"),
        )
        return bundle, fusion, pins

    def test_intact_bundle_passes(self) -> None:
        bundle, fusion, pins = self._fake_bundle_and_fusion()
        checks = _by_name(preflight.check_runtime_bundle(bundle, fusion, pins))
        self.assertTrue(all(item["passed"] for item in checks.values()), checks)

    def test_tampered_asset_fails(self) -> None:
        bundle, fusion, pins = self._fake_bundle_and_fusion()
        (bundle / "weights.npy").write_bytes(b"tampered!!")
        checks = _by_name(preflight.check_runtime_bundle(bundle, fusion, pins))
        self.assertFalse(checks["bundle.asset_sha256"]["passed"])

    def test_unbound_fusion_fails(self) -> None:
        bundle, fusion, pins = self._fake_bundle_and_fusion()
        (fusion / "dev_metrics.json").write_bytes(b'{"artifacts": {"x": 1}}')
        checks = _by_name(preflight.check_runtime_bundle(bundle, fusion, pins))
        self.assertFalse(checks["bundle.bound_to_candidate_fusion"]["passed"])

    def test_wrong_schema_fails(self) -> None:
        bundle, fusion, pins = self._fake_bundle_and_fusion()
        manifest = json.loads((bundle / "bundle_manifest.json").read_text())
        manifest["schema"] = "invariant-patch-v1"
        (bundle / "bundle_manifest.json").write_text(json.dumps(manifest))
        checks = _by_name(preflight.check_runtime_bundle(bundle, fusion, pins))
        self.assertFalse(checks["bundle.schema"]["passed"])

    def test_non_unset_rejection_slot_fails(self) -> None:
        bundle, fusion, pins = self._fake_bundle_and_fusion()
        manifest = json.loads((bundle / "bundle_manifest.json").read_text())
        manifest["rejection"]["state"] = "attached"
        (bundle / "bundle_manifest.json").write_text(json.dumps(manifest))
        checks = _by_name(preflight.check_runtime_bundle(bundle, fusion, pins))
        self.assertFalse(checks["bundle.rejection_slot_contract"]["passed"])

    def test_wrong_explicit_runtime_role_fails(self) -> None:
        bundle, fusion, pins = self._fake_bundle_and_fusion()
        manifest_path = bundle / "bundle_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["runtime_role"] = "known_unknown_rejector"
        manifest_path.write_text(json.dumps(manifest))
        checks = _by_name(
            preflight.check_runtime_bundle(
                bundle,
                fusion,
                pins,
                role="classifier",
                expected_runtime_role="accepted_known_classifier",
            )
        )
        self.assertFalse(checks["classifier.bundle.runtime_role"]["passed"])


class DualCandidateAdmissionTest(TempDirTestCase):
    def test_old_single_fusion_schema_and_unpinned_sha_fail_closed(self) -> None:
        root = Path(self.tmpdir())
        manifest = root / "candidate.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema": "atomos.v3.single-fusion-candidate",
                    "status": "release_candidate_frozen",
                    "candidate_id": preflight.DEFAULT_PINS["candidate_id"],
                }
            )
        )
        checks = _by_name(
            preflight.check_dual_candidate(
                manifest,
                root / "classifier_bundle",
                root / "rejector_bundle",
                root / "classifier_fusion",
                root / "rejector_fusion",
                root / "staged",
                root / "prefilter",
                preflight.DEFAULT_PINS,
            )
        )
        self.assertFalse(checks["candidate.schema"]["passed"])
        self.assertFalse(checks["candidate.manifest_sha256"]["passed"])
        self.assertFalse(checks["candidate.complete_binding_chain"]["passed"])
        self.assertTrue(checks["candidate.package_binding_schemas"]["passed"])


@unittest.skipUnless(
    preflight.DEFAULT_BUNDLE_DIR.is_dir(),
    "real runtime bundle not present",
)
class BundleSelfVerificationTest(TempDirTestCase):
    """Probe replay against the real frozen bundle, and a tampered copy."""

    def test_real_bundle_verifies(self) -> None:
        checks = preflight.check_bundle_self_verification(
            preflight.DEFAULT_BUNDLE_DIR
        )
        self.assertEqual(len(checks), 1)
        name, passed, detail = checks[0]
        self.assertEqual(name, "bundle.self_verification")
        self.assertTrue(passed, detail)
        self.assertIn("closed labels agree", detail)

    def _copy_bundle(self) -> Path:
        target = (
            Path(self.tmpdir()) / "bundle"
        )
        shutil.copytree(preflight.DEFAULT_BUNDLE_DIR, target)
        return target

    def test_tampered_expected_embedding_is_refused(self) -> None:
        bundle = self._copy_bundle()
        fixture_path = bundle / "probe_fixture.json"
        fixture = json.loads(fixture_path.read_text())
        fixture["cases"][0]["expected"]["fused_embedding"][0] += 1.0
        fixture_path.write_text(json.dumps(fixture))
        with self.assertRaises(AssertionError):
            preflight.check_bundle_self_verification(bundle)

    def test_probe_case_drift_is_refused(self) -> None:
        bundle = self._copy_bundle()
        fixture_path = bundle / "probe_fixture.json"
        fixture = json.loads(fixture_path.read_text())
        fixture["cases"][0]["name"] = "renamed-case"
        fixture_path.write_text(json.dumps(fixture))
        checks = preflight.check_bundle_self_verification(bundle)
        self.assertEqual(len(checks), 1)
        self.assertFalse(checks[0][1])


@unittest.skipUnless(
    preflight.DEFAULT_PREFILTER_ROOT.is_dir(),
    "real prefilter fit not present",
)
class PrefilterCheckTest(TempDirTestCase):
    def test_real_prefilter_passes(self) -> None:
        checks = _by_name(
            preflight.check_prefilter(
                preflight.DEFAULT_PREFILTER_ROOT, preflight.DEFAULT_PINS
            )
        )
        self.assertTrue(all(item["passed"] for item in checks.values()), checks)
        self.assertIn("prefilter.architecture_contract_keys", checks)
        self.assertIn("prefilter.fitting_seed_hygiene", checks)

    def test_tampered_bundle_fails(self) -> None:
        root = (
            Path(self.tmpdir()) / "prefilter"
        )
        shutil.copytree(preflight.DEFAULT_PREFILTER_ROOT, root)
        target = root / "bundles" / "N4096" / "coefficients.npy"
        payload = bytearray(target.read_bytes())
        payload[-1] ^= 0xFF
        target.write_bytes(bytes(payload))
        checks = _by_name(
            preflight.check_prefilter(root, preflight.DEFAULT_PINS)
        )
        self.assertFalse(checks["prefilter.set_sha256"]["passed"])
        self.assertFalse(checks["prefilter.per_length_sha256"]["passed"])


class FrozenDesignEvidenceTest(TempDirTestCase):
    def test_real_q99_failure_and_q97_pass_are_hash_pinned(self) -> None:
        q99 = _by_name(
            preflight.check_design_evidence(
                preflight.DEFAULT_Q99_DESIGN_DIR,
                preflight.DEFAULT_PINS["q99_design_evidence"],
                label="q99_design",
            )
        )
        q97 = _by_name(
            preflight.check_design_evidence(
                preflight.DEFAULT_Q97_DESIGN_DIR,
                preflight.DEFAULT_PINS["q97_design_evidence"],
                label="q97_design",
            )
        )
        self.assertTrue(all(item["passed"] for item in q99.values()), q99)
        self.assertTrue(all(item["passed"] for item in q97.values()), q97)
        self.assertIn("q99_design.failed_gate_worst", q99)
        self.assertNotIn("q97_design.failed_gate_worst", q97)

    def test_tampered_design_report_fails_its_independent_pin(self) -> None:
        root = Path(self.tmpdir()) / "q97"
        shutil.copytree(preflight.DEFAULT_Q97_DESIGN_DIR, root)
        report_path = root / "openset_metrics.json"
        report = json.loads(report_path.read_text())
        report["status"] = "design_selection_fail"
        report_path.write_text(json.dumps(report))
        checks = _by_name(
            preflight.check_design_evidence(
                root,
                preflight.DEFAULT_PINS["q97_design_evidence"],
                label="q97_design",
            )
        )
        self.assertFalse(checks["q97_design.report_sha256"]["passed"])
        self.assertFalse(checks["q97_design.role_status_hygiene"]["passed"])


class FutureReleasePinsTest(TempDirTestCase):
    def test_final_hashes_and_validation_reasons_are_all_bound(self) -> None:
        checks = _by_name(
            preflight.check_future_release_pins(preflight.DEFAULT_PINS)
        )
        expected = {
            "future_pin.evaluator",
            "future_pin.candidate_manifest",
            "future_pin.validation_evidence",
            "future_pin.candidate_contract",
            "future_pin.staged_validation_report",
            "future_pin.staging_package_manifest",
            "future_pin.dual_binding",
            "future_pin.protocol_fixture",
            "future_pin.browser_asset.classifier",
            "future_pin.browser_asset.rejector",
            "future_pin.browser_asset.openset_policy",
            "future_pin.validation_seed_reason.20260953",
            "future_pin.validation_seed_reason.20260954",
        }
        self.assertTrue(expected.issubset(checks))
        self.assertTrue(
            all(checks[name]["passed"] for name in expected),
            checks,
        )
        unbound = copy.deepcopy(preflight.DEFAULT_PINS)
        unbound["candidate_manifest_sha256"] = None
        unbound["validation_seed_reasons"][20260953] = None
        refused = _by_name(preflight.check_future_release_pins(unbound))
        self.assertFalse(refused["future_pin.candidate_manifest"]["passed"])
        self.assertFalse(
            refused["future_pin.validation_seed_reason.20260953"]["passed"]
        )
        self.assertEqual(
            preflight.DEFAULT_PINS["candidate_manifest_schema"],
            "time-domain-v3-dual-release-candidate-v2",
        )
        self.assertEqual(
            preflight.DEFAULT_PINS["candidate_evidence_schema"],
            "time-domain-v3-decoupled-validation-evidence-v2",
        )
        self.assertEqual(
            preflight.DEFAULT_PINS["candidate_contract_schema"],
            "time-domain-v3-q97-candidate-v1",
        )
        self.assertEqual(
            preflight.DEFAULT_PINS["browser_openset_schema_version"], 4
        )
        self.assertEqual(preflight.DEFAULT_PINS["parity_schema_version"], 4)

    def test_only_exact_lowercase_sha256_is_ready(self) -> None:
        self.assertTrue(preflight._is_sha256_pin("a" * 64))
        self.assertFalse(preflight._is_sha256_pin("A" * 64))
        self.assertFalse(preflight._is_sha256_pin("a" * 63))
        self.assertFalse(preflight._is_sha256_pin(None))


class FrozenPolicyCheckTest(TempDirTestCase):
    def test_q97_constants_and_spent_validation_ledger_pass(
        self,
    ) -> None:
        checks = _by_name(preflight.check_frozen_policy(preflight.DEFAULT_PINS))
        identity_checks = {
            name: item
            for name, item in checks.items()
            if name != "policy.seed_ledger_constants"
        }
        self.assertTrue(
            all(item["passed"] for item in identity_checks.values()),
            checks,
        )
        self.assertTrue(checks["policy.seed_ledger_constants"]["passed"])
        self.assertIn("policy.import_identity_discipline", checks)
        self.assertIn("policy.seed_ledger_constants", checks)
        unbound = copy.deepcopy(preflight.DEFAULT_PINS)
        unbound["validation_seed_reasons"][20260954] = None
        refused = _by_name(preflight.check_frozen_policy(unbound))
        self.assertFalse(refused["policy.seed_ledger_constants"]["passed"])

    def test_drifted_pin_fails(self) -> None:
        pins = copy.deepcopy(preflight.DEFAULT_PINS)
        pins["frozen_policy"]["v2_weight"] = 0.75
        checks = _by_name(preflight.check_frozen_policy(pins))
        self.assertFalse(checks["policy.v2_weight"]["passed"])

    def test_module_drift_fails(self) -> None:
        import v3_time_domain_openset as openset

        original = openset.FROZEN_GEOMETRY_WEIGHT
        openset.FROZEN_GEOMETRY_WEIGHT = 0.5
        try:
            checks = _by_name(
                preflight.check_frozen_policy(preflight.DEFAULT_PINS)
            )
        finally:
            openset.FROZEN_GEOMETRY_WEIGHT = original
        self.assertFalse(checks["policy.geometry_weight"]["passed"])


class EvaluatorSourcesCheckTest(TempDirTestCase):
    def test_working_evaluator_has_exact_schema4_q97_identity_and_final_pin(
        self,
    ) -> None:
        checks = _by_name(
            preflight.check_evaluator_sources(
                preflight.DEFAULT_EVALUATOR,
                preflight.DEFAULT_PINS,
                {},
            )
        )
        self.assertTrue(checks["sources.v3_evaluator_present"]["passed"])
        self.assertTrue(checks["sources.v3_evaluator_identity"]["passed"])
        self.assertTrue(checks["sources.v3_evaluator_sha256"]["passed"])
        self.assertEqual(
            preflight.DEFAULT_PINS["v3_evaluator_sha256"],
            preflight._sha256(preflight.DEFAULT_EVALUATOR),
        )

    def test_missing_v3_evaluator_fails(self) -> None:
        missing = (
            Path(self.tmpdir()) / "no.py"
        )
        observed: dict = {}
        checks = _by_name(
            preflight.check_evaluator_sources(
                missing, preflight.DEFAULT_PINS, observed
            )
        )
        self.assertFalse(checks["sources.v3_evaluator_present"]["passed"])
        self.assertTrue(checks["sources.dependency_hashes_recorded"]["passed"])
        # The pinned v2 evaluator and gate-floor identity still hold.
        self.assertTrue(checks["sources.v2_evaluator_sha256"]["passed"])
        self.assertTrue(checks["sources.gate_floor_consistency"]["passed"])

    def test_stub_v3_evaluator_passes_and_is_hashed(self) -> None:
        stub = (
            Path(self.tmpdir())
            / "evaluate_v3_release_suite.py"
        )
        stub.write_text('"""stub evaluator"""\nGATES = "predeclared"\n')
        observed: dict = {}
        checks = _by_name(
            preflight.check_evaluator_sources(
                stub, preflight.DEFAULT_PINS, observed
            )
        )
        self.assertTrue(checks["sources.v3_evaluator_present"]["passed"])
        # A stub is present and byte-compiles, but it is NOT the pinned
        # evaluator carrying the owner's gate redeclaration.
        self.assertFalse(checks["sources.v3_evaluator_sha256"]["passed"])
        self.assertEqual(
            observed["v3_evaluator_sha256"], preflight._sha256(stub)
        )
        self.assertEqual(len(observed["dependency_source_sha256"]), 49)
        self.assertTrue(
            checks["sources.dependency_hashes_recorded"]["passed"]
        )
        self.assertFalse(checks["sources.static_import_closure"]["passed"])

    def test_exact_49_file_contract_and_static_closure(self) -> None:
        observed: dict = {}
        checks = _by_name(
            preflight.check_evaluator_sources(
                preflight.DEFAULT_EVALUATOR,
                preflight.DEFAULT_PINS,
                observed,
            )
        )
        self.assertEqual(len(preflight.DEPENDENCY_SOURCES), 49)
        self.assertEqual(
            set(preflight.DEFAULT_PINS["dependency_source_sha256"]),
            set(preflight.DEPENDENCY_SOURCE_LABELS),
        )
        self.assertTrue(checks["sources.dependency_contract_key_set"]["passed"])
        self.assertTrue(
            checks["sources.evaluator_dependency_path_contract"]["passed"]
        )
        self.assertTrue(checks["sources.static_import_closure"]["passed"])
        self.assertTrue(checks["sources.dependency_hashes_pinned"]["passed"])
        unbound = copy.deepcopy(preflight.DEFAULT_PINS)
        unbound["dependency_source_sha256"][
            next(iter(unbound["dependency_source_sha256"]))
        ] = None
        refused = _by_name(
            preflight.check_evaluator_sources(
                preflight.DEFAULT_EVALUATOR,
                unbound,
                {},
            )
        )
        self.assertFalse(refused["sources.dependency_hashes_pinned"]["passed"])

    def test_dependency_pin_tamper_and_key_drift_fail(self) -> None:
        pins = copy.deepcopy(preflight.DEFAULT_PINS)
        pins["dependency_source_sha256"][
            "training/zplane_ab/v2_full_variation/v3_scale/"
            "measure_v3_remaining_gates.py"
        ] = "0" * 64
        pins["dependency_source_sha256"]["unexpected.py"] = "0" * 64
        checks = _by_name(
            preflight.check_evaluator_sources(
                preflight.DEFAULT_EVALUATOR, pins, {}
            )
        )
        self.assertFalse(
            checks["sources.dependency_contract_key_set"]["passed"]
        )
        self.assertFalse(checks["sources.dependency_hashes_pinned"]["passed"])

    def test_syntax_error_evaluator_is_refused(self) -> None:
        import py_compile

        stub = (
            Path(self.tmpdir()) / "bad.py"
        )
        stub.write_text("def broken(:\n")
        with self.assertRaises(py_compile.PyCompileError):
            preflight.check_evaluator_sources(
                stub, preflight.DEFAULT_PINS, {}
            )

    def test_drifted_v2_evaluator_pin_fails(self) -> None:
        pins = copy.deepcopy(preflight.DEFAULT_PINS)
        pins["v2_evaluator_sha256"] = "0" * 64
        checks = _by_name(
            preflight.check_evaluator_sources(
                preflight.DEFAULT_EVALUATOR, pins, {}
            )
        )
        self.assertFalse(checks["sources.v2_evaluator_sha256"]["passed"])


class HistoricalCorrectionContractTest(TempDirTestCase):
    def _protocol(self) -> dict:
        import evaluate_invariant_release_suite as release

        rationale = (
            "The levels are re-declared for the v3 architecture by the owner "
            "on 2026-07-28, BEFORE release seed 20260734 was generated."
        )
        expected = preflight.DEFAULT_PINS["historical_gate_redeclaration"]
        gates = {
            name: {
                "v2_level": record["v2_level"],
                "v3_level": record["v3_level"],
                "owner_decision": rationale,
            }
            for name, record in expected["gates"].items()
        }
        return {
            "gates": dict(release.GATE_FLOORS),
            "historical_gate_redeclaration": {
                "release_seed": 20260734,
                "active_for_current_protocol": False,
                "current_protocol_gate_source": (
                    "imported_v2_gate_floors_unchanged"
                ),
                "gates_redeclared": gates,
                "historical_claim_correction": copy.deepcopy(
                    preflight.DEFAULT_PINS["historical_claim_correction"]
                ),
            },
        }

    def test_exact_machine_readable_correction_passes(self) -> None:
        checks = _by_name(
            preflight._check_fixture_gate_contract(
                self._protocol(), preflight.DEFAULT_PINS
            )
        )
        self.assertTrue(
            checks["generation.v3_fixture_gate_contract"]["passed"],
            checks,
        )

    def test_old_every_axis_claim_without_correction_fails(self) -> None:
        protocol = self._protocol()
        del protocol["historical_gate_redeclaration"][
            "historical_claim_correction"
        ]
        checks = _by_name(
            preflight._check_fixture_gate_contract(
                protocol, preflight.DEFAULT_PINS
            )
        )
        self.assertFalse(
            checks["generation.v3_fixture_gate_contract"]["passed"]
        )
        self.assertIn(
            "cross-seed/unpaired",
            checks["generation.v3_fixture_gate_contract"]["detail"],
        )


class SeedLedgerCheckTest(TempDirTestCase):
    FAKE_ROOTS = (
        "fake_sealed_v2_root",
        "fake_sealed_v3_root",
        "fake_sealed_v3_run2_root",
        "fake_sealed_v3_run3_root",
        "fake_sealed_v3_run4_root",
    )

    def _pins_with_fake_sealed(self, releases: Path) -> dict:
        pins = copy.deepcopy(preflight.DEFAULT_PINS)
        pins["consumed_sealed_roots"] = {}
        for root_name, recorded_seed in zip(
            self.FAKE_ROOTS,
            (20260729, 20260731, 20260733, 20260734, 20260735),
        ):
            sealed = releases / root_name
            hashes = {}
            for name in (
                "RELEASE_INTENT.json",
                "RELEASE_MANIFEST.json",
                "RELEASE_EVALUATION.json",
            ):
                payload = json.dumps(
                    {"file": name, "release_seed": recorded_seed}
                ).encode()
                _write(sealed / name, payload)
                hashes[name] = _sha(payload)
            pins["consumed_sealed_roots"][root_name] = hashes
        return pins

    def test_unused_seed_and_intact_references_pass(self) -> None:
        releases = Path(self.tmpdir())
        pins = self._pins_with_fake_sealed(releases)
        checks = _by_name(
            preflight.check_seed_ledger(
                releases, pins, require_canonical=False
            )
        )
        self.assertTrue(checks["ledger.release_seed_unused"]["passed"])
        for root_name in self.FAKE_ROOTS:
            self.assertTrue(
                checks[f"ledger.consumed_root_intact.{root_name}"]["passed"]
            )

    def test_seed_named_root_fails(self) -> None:
        releases = Path(self.tmpdir())
        pins = self._pins_with_fake_sealed(releases)
        (releases / "v3_sealed_seed20260736").mkdir()
        checks = _by_name(
            preflight.check_seed_ledger(
                releases, pins, require_canonical=False
            )
        )
        self.assertFalse(checks["ledger.release_seed_unused"]["passed"])

    def test_intent_recording_seed_fails(self) -> None:
        releases = Path(self.tmpdir())
        pins = self._pins_with_fake_sealed(releases)
        _write(
            releases / "innocuous_name" / "RELEASE_INTENT.json",
            json.dumps({"release_seed": 20260736}).encode(),
        )
        checks = _by_name(
            preflight.check_seed_ledger(
                releases, pins, require_canonical=False
            )
        )
        self.assertFalse(checks["ledger.release_seed_unused"]["passed"])

    def test_nested_intent_recording_seed_fails(self) -> None:
        releases = Path(self.tmpdir())
        pins = self._pins_with_fake_sealed(releases)
        _write(
            releases / "outer" / "nested" / "RELEASE_INTENT.json",
            json.dumps({"release_seed": 20260736}).encode(),
        )
        checks = _by_name(
            preflight.check_seed_ledger(
                releases, pins, require_canonical=False
            )
        )
        self.assertFalse(checks["ledger.release_seed_unused"]["passed"])
        self.assertIn(
            "outer/nested/RELEASE_INTENT.json",
            checks["ledger.release_seed_unused"]["detail"],
        )

    def test_seed_scan_refuses_symlinks_without_following(self) -> None:
        releases = Path(self.tmpdir())
        pins = self._pins_with_fake_sealed(releases)
        outside = Path(self.tmpdir())
        _write(
            outside / "RELEASE_INTENT.json",
            json.dumps({"release_seed": 20260736}).encode(),
        )
        os.symlink(outside, releases / "linked-release")
        checks = _by_name(
            preflight.check_seed_ledger(
                releases, pins, require_canonical=False
            )
        )
        self.assertFalse(checks["ledger.release_seed_unused"]["passed"])
        self.assertIn("symlink is forbidden", checks["ledger.release_seed_unused"]["detail"])

    def test_consumed_v3_root_is_not_flagged_as_seed_use(self) -> None:
        # Consumed roots through 20260735 exist on disk by design; only the
        # NEW seed 20260736 may appear in no release root.
        releases = Path(self.tmpdir())
        pins = self._pins_with_fake_sealed(releases)
        _write(
            releases / "old_sealed" / "RELEASE_INTENT.json",
            json.dumps({"release_seed": 20260733}).encode(),
        )
        checks = _by_name(
            preflight.check_seed_ledger(
                releases, pins, require_canonical=False
            )
        )
        self.assertTrue(checks["ledger.release_seed_unused"]["passed"])

    def test_tampered_sealed_reference_fails(self) -> None:
        releases = Path(self.tmpdir())
        pins = self._pins_with_fake_sealed(releases)
        target = releases / self.FAKE_ROOTS[1] / "RELEASE_EVALUATION.json"
        target.write_bytes(b'{"tampered": true}')
        checks = _by_name(
            preflight.check_seed_ledger(
                releases, pins, require_canonical=False
            )
        )
        self.assertFalse(
            checks[f"ledger.consumed_root_intact.{self.FAKE_ROOTS[1]}"]["passed"]
        )
        self.assertTrue(
            checks[f"ledger.consumed_root_intact.{self.FAKE_ROOTS[0]}"]["passed"]
        )

    def test_symlinked_release_root_is_refused_without_traversal(self) -> None:
        real_releases = Path(self.tmpdir())
        pins = self._pins_with_fake_sealed(real_releases)
        link_holder = Path(self.tmpdir())
        linked_releases = link_holder / "linked-releases"
        os.symlink(real_releases, linked_releases)
        checks = _by_name(
            preflight.check_seed_ledger(
                linked_releases, pins, require_canonical=False
            )
        )
        self.assertFalse(checks["ledger.releases_dir_identity"]["passed"])
        self.assertFalse(checks["ledger.release_seed_unused"]["passed"])

    def test_noncanonical_complete_ledger_is_refused_for_release_36(self) -> None:
        releases = Path(self.tmpdir())
        pins = self._pins_with_fake_sealed(releases)
        checks = _by_name(preflight.check_seed_ledger(releases, pins))
        self.assertFalse(checks["ledger.releases_dir_identity"]["passed"])
        self.assertFalse(checks["ledger.release_seed_unused"]["passed"])

    def test_canonical_ledger_with_symlinked_ancestor_is_refused(self) -> None:
        base = Path(self.tmpdir())
        real_releases = base / "real" / "training" / "artifacts" / "releases"
        pins = self._pins_with_fake_sealed(real_releases)
        linked_repo = base / "linked"
        linked_repo.mkdir()
        os.symlink(base / "real" / "training", linked_repo / "training")
        canonical_via_symlink = (
            linked_repo / "training" / "artifacts" / "releases"
        )
        with mock.patch.object(
            preflight,
            "DEFAULT_RELEASES_DIR",
            canonical_via_symlink,
        ):
            checks = _by_name(
                preflight.check_seed_ledger(canonical_via_symlink, pins)
            )
        self.assertFalse(checks["ledger.releases_dir_identity"]["passed"])
        self.assertFalse(checks["ledger.release_seed_unused"]["passed"])

    @unittest.skipUnless(
        preflight.DEFAULT_RELEASES_DIR.is_dir(), "real release tree missing"
    )
    def test_real_release_tree_passes(self) -> None:
        checks = _by_name(
            preflight.check_seed_ledger(
                preflight.DEFAULT_RELEASES_DIR, preflight.DEFAULT_PINS
            )
        )
        self.assertTrue(all(item["passed"] for item in checks.values()), checks)


class IsolatedSourceCheckTest(TempDirTestCase):
    def _fake_isolated(self) -> tuple[Path, dict]:
        base = Path(self.tmpdir())
        signallab = base / "Atom-SignalLab"
        atom_dsp = base / "Atom-DSP"
        _write(signallab / "src" / "index.ts", b"export const x = 1;\n")
        _write(signallab / "package.json", b"{}")
        signallab_lock = _write(
            signallab / "package-lock.json", b'{"lock": "signallab"}'
        )
        _write(atom_dsp / "src" / "dsp.ts", b"export const y = 2;\n")
        atom_dsp_lock = _write(
            atom_dsp / "package-lock.json", b'{"lock": "dsp"}'
        )
        dist_js = _write(atom_dsp / "dist" / "index.js", b"module.exports = {}")
        dist_dts = _write(atom_dsp / "dist" / "index.d.ts", b"export {};")
        link_parent = signallab / "node_modules" / "@atomos"
        link_parent.mkdir(parents=True)
        os.symlink(atom_dsp, link_parent / "dsp")
        pins = copy.deepcopy(preflight.DEFAULT_PINS)
        sl_digest, sl_count = preflight.source_tree_digest(signallab)
        dsp_digest, dsp_count = preflight.source_tree_digest(atom_dsp)
        pins["signallab"] = {
            "source_tree_sha256": sl_digest,
            "source_file_count": sl_count,
            "package_lock_sha256": _sha(signallab_lock.read_bytes()),
        }
        pins["atom_dsp"] = {
            "source_tree_sha256": dsp_digest,
            "source_file_count": dsp_count,
            "package_lock_sha256": _sha(atom_dsp_lock.read_bytes()),
            "dist_index_js_sha256": _sha(dist_js.read_bytes()),
            "dist_index_dts_sha256": _sha(dist_dts.read_bytes()),
        }
        return signallab, pins

    def test_intact_isolated_root_passes(self) -> None:
        signallab, pins = self._fake_isolated()
        checks = _by_name(preflight.check_isolated_source(signallab, pins))
        self.assertTrue(all(item["passed"] for item in checks.values()), checks)
        self.assertIn("isolated.atom_dsp_symlink", checks)

    def test_missing_root_is_one_failure(self) -> None:
        missing = (
            Path(self.tmpdir()) / "gone"
        )
        checks = preflight.check_isolated_source(
            missing, preflight.DEFAULT_PINS
        )
        self.assertEqual(len(checks), 1)
        self.assertFalse(checks[0][1])
        self.assertIn("temporary", checks[0][2])

    def test_source_drift_fails_digest(self) -> None:
        signallab, pins = self._fake_isolated()
        (signallab / "src" / "index.ts").write_bytes(b"export const x = 2;\n")
        checks = _by_name(preflight.check_isolated_source(signallab, pins))
        self.assertFalse(checks["isolated.signallab_source_digest"]["passed"])

    def test_dist_drift_fails_without_touching_source_digest(self) -> None:
        signallab, pins = self._fake_isolated()
        dist = signallab.parent / "Atom-DSP" / "dist" / "index.js"
        dist.write_bytes(b"module.exports = {tampered: 1}")
        checks = _by_name(preflight.check_isolated_source(signallab, pins))
        self.assertTrue(checks["isolated.atom_dsp_source_digest"]["passed"])
        self.assertFalse(checks["isolated.atom_dsp_dist_index_js"]["passed"])

    def test_wrong_symlink_fails(self) -> None:
        signallab, pins = self._fake_isolated()
        link = signallab / "node_modules" / "@atomos" / "dsp"
        link.unlink()
        elsewhere = signallab.parent / "elsewhere"
        elsewhere.mkdir()
        os.symlink(elsewhere, link)
        checks = _by_name(preflight.check_isolated_source(signallab, pins))
        self.assertFalse(checks["isolated.atom_dsp_symlink"]["passed"])

    @unittest.skipUnless(
        preflight.DEFAULT_ISOLATED_ROOT.is_dir(), "real isolated root missing"
    )
    def test_real_isolated_root_matches_handoff_pins(self) -> None:
        checks = _by_name(
            preflight.check_isolated_source(
                preflight.DEFAULT_ISOLATED_ROOT, preflight.DEFAULT_PINS
            )
        )
        self.assertTrue(all(item["passed"] for item in checks.values()), checks)


class NodeRuntimeCheckTest(TempDirTestCase):
    def _stub_bin(self, node: str, npm: str, npx: str) -> Path:
        bin_dir = Path(self.tmpdir())
        for name, version in (("node", node), ("npm", npm), ("npx", npx)):
            script = bin_dir / name
            script.write_text(f"#!/bin/sh\necho '{version}'\n")
            script.chmod(script.stat().st_mode | stat.S_IXUSR)
        return bin_dir

    def test_pinned_versions_pass(self) -> None:
        bin_dir = self._stub_bin("v22.23.1", "10.9.8", "10.9.8")
        checks = _by_name(
            preflight.check_node_runtime(bin_dir, preflight.DEFAULT_PINS)
        )
        self.assertTrue(all(item["passed"] for item in checks.values()), checks)

    def test_wrong_node_version_fails(self) -> None:
        bin_dir = self._stub_bin("v20.11.0", "10.9.8", "10.9.8")
        checks = _by_name(
            preflight.check_node_runtime(bin_dir, preflight.DEFAULT_PINS)
        )
        self.assertFalse(checks["runtime.node_version"]["passed"])
        self.assertTrue(checks["runtime.npm_version"]["passed"])

    def test_missing_bin_dir_is_one_failure(self) -> None:
        missing = (
            Path(self.tmpdir()) / "none"
        )
        checks = preflight.check_node_runtime(missing, preflight.DEFAULT_PINS)
        self.assertEqual(len(checks), 1)
        self.assertFalse(checks[0][1])


class PythonRuntimeContractTest(unittest.TestCase):
    def test_exact_runtime_passes(self) -> None:
        checks = _by_name(
            preflight.check_python_runtime(
                preflight.DEFAULT_PINS,
                runtime=preflight.DEFAULT_PINS["python_runtime"],
                warnings_policy="error",
            )
        )
        self.assertTrue(all(item["passed"] for item in checks.values()), checks)

    def test_every_runtime_identity_mismatch_fails(self) -> None:
        expected = dict(preflight.DEFAULT_PINS["python_runtime"])
        for key in expected:
            with self.subTest(key=key):
                runtime = dict(expected)
                runtime[key] = f"wrong-{expected[key]}"
                checks = _by_name(
                    preflight.check_python_runtime(
                        preflight.DEFAULT_PINS,
                        runtime=runtime,
                        warnings_policy="error",
                    )
                )
                self.assertFalse(checks[f"runtime.python.{key}"]["passed"])

    def test_extra_runtime_key_and_warning_policy_fail(self) -> None:
        runtime = dict(preflight.DEFAULT_PINS["python_runtime"])
        runtime["extra"] = "forbidden"
        checks = _by_name(
            preflight.check_python_runtime(
                preflight.DEFAULT_PINS,
                runtime=runtime,
                warnings_policy="default",
            )
        )
        self.assertFalse(checks["runtime.python_contract_key_set"]["passed"])
        self.assertFalse(checks["runtime.pythonwarnings_error"]["passed"])


class GenerationSourceContractTest(unittest.TestCase):
    def test_seed35_go_preflight_evidence_is_pinned(self) -> None:
        checks = _by_name(
            preflight.check_generation_sources(preflight.DEFAULT_PINS)
        )
        self.assertTrue(
            checks["generation.seed35_go_preflight_evidence"]["passed"]
        )

    def test_seed35_preflight_pin_tamper_fails(self) -> None:
        pins = copy.deepcopy(preflight.DEFAULT_PINS)
        pins["seed35_preflight_evidence_sha256"] = "0" * 64
        checks = _by_name(preflight.check_generation_sources(pins))
        self.assertFalse(
            checks["generation.seed35_go_preflight_evidence"]["passed"]
        )


class GenerationCommandTest(TempDirTestCase):
    """The reconstructed one-shot command: right values, never executed."""

    def test_prevalidation_placeholders_refuse_command(self) -> None:
        candidate = Path(self.tmpdir()) / "candidate.json"
        candidate.write_text(
            json.dumps(
                {
                    "schema": preflight.DEFAULT_PINS[
                        "candidate_manifest_schema"
                    ],
                    "status": "release_candidate_frozen",
                }
            )
        )
        with self.assertRaisesRegex(
            ValueError, "candidate path must be exactly"
        ):
            preflight.build_generation_command(
                preflight.DEFAULT_PINS,
                candidate,
                preflight.DEFAULT_ISOLATED_ROOT,
                preflight.DEFAULT_NODE_BIN_DIR,
            )

    def test_exact_path_builds_only_with_bound_future_pins(self) -> None:
        command = preflight.build_generation_command(
            preflight.DEFAULT_PINS,
            preflight.DEFAULT_CANDIDATE_MANIFEST,
            preflight.DEFAULT_ISOLATED_ROOT,
            preflight.DEFAULT_NODE_BIN_DIR,
        )
        self.assertTrue(command["not_run_by_preflight"])
        self.assertEqual(
            command["candidate_manifest_sha256"],
            preflight.DEFAULT_PINS["candidate_manifest_sha256"],
        )
        unbound = copy.deepcopy(preflight.DEFAULT_PINS)
        unbound["v3_protocol_fixture_sha256"] = None
        with self.assertRaisesRegex(ValueError, "unbound fail-closed"):
            preflight.build_generation_command(
                unbound,
                preflight.DEFAULT_CANDIDATE_MANIFEST,
                preflight.DEFAULT_ISOLATED_ROOT,
                preflight.DEFAULT_NODE_BIN_DIR,
            )

    def test_module_never_invokes_the_launcher(self) -> None:
        source = Path(preflight.__file__).read_text(encoding="utf-8")
        # subprocess is used exactly once, for --version probes.
        self.assertEqual(source.count("subprocess.run"), 1)
        self.assertIn('"--version"', source)


class HarnessTest(TempDirTestCase):
    """The run harness, the itemised output and the single GO / NO-GO line."""

    def _args(self, **overrides) -> object:
        defaults = {
            "candidate_manifest": str(preflight.DEFAULT_CANDIDATE_MANIFEST),
            "classifier_fusion_dir": str(
                preflight.DEFAULT_CLASSIFIER_FUSION_DIR
            ),
            "rejector_fusion_dir": str(
                preflight.DEFAULT_REJECTOR_FUSION_DIR
            ),
            "classifier_bundle_dir": str(
                preflight.DEFAULT_CLASSIFIER_BUNDLE_DIR
            ),
            "rejector_bundle_dir": str(
                preflight.DEFAULT_REJECTOR_BUNDLE_DIR
            ),
            "prefilter_root": str(preflight.DEFAULT_PREFILTER_ROOT),
            "q99_design_dir": str(preflight.DEFAULT_Q99_DESIGN_DIR),
            "q97_design_dir": str(preflight.DEFAULT_Q97_DESIGN_DIR),
            "releases_dir": str(preflight.DEFAULT_RELEASES_DIR),
            "isolated_root": str(preflight.DEFAULT_ISOLATED_ROOT),
            "evaluator": str(preflight.DEFAULT_EVALUATOR),
            "node_bin_dir": str(preflight.DEFAULT_NODE_BIN_DIR),
            "output": None,
        }
        defaults.update(overrides)
        return preflight.build_parser().parse_args(
            [
                f"--{key.replace('_', '-')}={value}"
                for key, value in defaults.items()
                if value is not None
            ]
        )

    def test_crashing_group_becomes_a_failed_check(self) -> None:
        gone = Path(self.tmpdir()) / "gone"
        args = self._args(
            candidate_manifest=str(gone / "candidate.json"),
            classifier_fusion_dir=str(gone / "classifier_fusion"),
            rejector_fusion_dir=str(gone / "rejector_fusion"),
            classifier_bundle_dir=str(gone / "classifier_bundle"),
            rejector_bundle_dir=str(gone / "rejector_bundle"),
            prefilter_root=str(gone / "prefilter"),
            q99_design_dir=str(gone / "q99"),
            q97_design_dir=str(gone / "q97"),
            releases_dir=str(gone / "releases"),
            isolated_root=str(gone / "isolated"),
            evaluator=str(gone / "eval.py"),
            node_bin_dir=str(gone / "bin"),
        )
        report = preflight.run_preflight(args)
        self.assertFalse(report["go"])
        names = {item["name"] for item in report["checks"]}
        # The crashed groups surface under their group labels.
        self.assertIn("classifier.fusion", names)
        self.assertIn("rejector.bundle", names)
        failed = {
            item["name"] for item in report["checks"] if not item["passed"]
        }
        self.assertIn("classifier.fusion", failed)
        # Groups that need nothing from the missing paths still ran.
        self.assertIn("policy.kind", names)

    def test_print_report_single_go_no_go_line(self) -> None:
        report = {
            "schema": preflight.REPORT_SCHEMA,
            "release_seed": 20260731,
            "go": False,
            "checks": [
                {"name": "a", "passed": True, "detail": "ok"},
                {"name": "b", "passed": False, "detail": "bad"},
            ],
            "generation_command": {},
        }
        stream = io.StringIO()
        preflight._print_report(report, stream=stream)
        lines = [line for line in stream.getvalue().splitlines() if line]
        self.assertEqual(lines[0], "PASS  a: ok")
        self.assertEqual(lines[1], "FAIL  b: bad")
        final = [
            line
            for line in lines
            if line.startswith("GO:") or line.startswith("NO-GO:")
        ]
        self.assertEqual(len(final), 1)
        self.assertEqual(final[0], lines[-1])
        self.assertIn("do not spend release seed 20260731", final[0])

    def test_go_line_when_everything_passes(self) -> None:
        report = {
            "schema": preflight.REPORT_SCHEMA,
            "release_seed": 20260731,
            "go": True,
            "checks": [{"name": "a", "passed": True, "detail": "ok"}],
            "generation_command": {},
        }
        stream = io.StringIO()
        preflight._print_report(report, stream=stream)
        final = stream.getvalue().splitlines()[-1]
        self.assertTrue(final.startswith("GO:"))
        self.assertIn("1/1", final)

    def test_refuses_existing_output_before_running(self) -> None:
        existing = (
            Path(self.tmpdir())
            / "report.json"
        )
        existing.write_text("{}")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            status = preflight.main(["--output", str(existing)])
        self.assertEqual(status, 2)
        self.assertIn("refusing to overwrite", stderr.getvalue())
        self.assertEqual(existing.read_text(), "{}")

    def test_refuses_output_under_release_tree(self) -> None:
        releases = Path(self.tmpdir())
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            status = preflight.main(
                [
                    "--releases-dir",
                    str(releases),
                    "--output",
                    str(releases / "sub" / "report.json"),
                ]
            )
        self.assertEqual(status, 2)
        self.assertIn("release tree", stderr.getvalue())
        self.assertEqual(list(releases.iterdir()), [])

    def test_override_cannot_hide_output_under_canonical_release_tree(
        self,
    ) -> None:
        canonical = Path(self.tmpdir())
        fake_override = Path(self.tmpdir())
        output = canonical / "sub" / "report.json"
        stderr = io.StringIO()
        with mock.patch.object(
            preflight, "DEFAULT_RELEASES_DIR", canonical
        ), contextlib.redirect_stderr(stderr):
            status = preflight.main(
                [
                    "--releases-dir",
                    str(fake_override),
                    "--output",
                    str(output),
                ]
            )
        self.assertEqual(status, 2)
        self.assertIn("release tree", stderr.getvalue())
        self.assertFalse(output.exists())


@unittest.skipUnless(
    preflight.DEFAULT_BUNDLE_DIR.is_dir()
    and preflight.DEFAULT_ISOLATED_ROOT.is_dir()
    and preflight.DEFAULT_NODE_BIN_DIR.is_dir(),
    "full frozen environment not present",
)
class EndToEndTest(TempDirTestCase):
    """The final pre-release environment admits release36 without spending it."""

    def test_full_preflight_is_go_and_consumes_nothing(self) -> None:
        tmp = Path(self.tmpdir())
        output = tmp / "preflight_report.json"
        previous = os.environ.get("PYTHONWARNINGS")
        os.environ["PYTHONWARNINGS"] = "error"
        releases_before = sorted(
            path.name for path in preflight.DEFAULT_RELEASES_DIR.iterdir()
        )
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                status = preflight.main(["--output", str(output)])
        finally:
            if previous is None:
                os.environ.pop("PYTHONWARNINGS", None)
            else:
                os.environ["PYTHONWARNINGS"] = previous
        text = stdout.getvalue()
        self.assertEqual(status, 0, text)
        final = text.splitlines()[-1]
        self.assertTrue(final.startswith("GO:"), final)
        self.assertIn("safe to spend release seed 20260736 once", final)
        # Exactly one GO / NO-GO line, every check itemised above it.
        go_lines = [
            line
            for line in text.splitlines()
            if line.startswith("GO:") or line.startswith("NO-GO:")
        ]
        self.assertEqual(len(go_lines), 1)
        report = json.loads(output.read_text())
        self.assertTrue(report["go"])
        failed = [item["name"] for item in report["checks"] if not item["passed"]]
        self.assertEqual(failed, [])
        self.assertGreaterEqual(len(report["checks"]), 40)
        self.assertEqual(
            report["observed"]["v3_evaluator_sha256"],
            preflight._sha256(preflight.DEFAULT_EVALUATOR),
        )
        self.assertEqual(
            preflight.DEFAULT_PINS["v3_evaluator_sha256"],
            report["observed"]["v3_evaluator_sha256"],
        )
        self.assertIn(
            "training/zplane_ab/v2_full_variation/"
            "evaluate_invariant_release_suite.py",
            report["observed"]["dependency_source_sha256"],
        )
        self.assertEqual(
            report["preflight_source_sha256"],
            preflight._sha256(Path(preflight.__file__)),
        )
        # Consumes nothing, and says so machine-readably.
        consumed = report["consumes_nothing"]
        self.assertFalse(consumed["release_seed_drawn"])
        self.assertEqual(consumed["novelty_realizations_generated"], 0)
        self.assertFalse(consumed["writes_under_releases"])
        releases_after = sorted(
            path.name for path in preflight.DEFAULT_RELEASES_DIR.iterdir()
        )
        self.assertEqual(releases_before, releases_after)
        self.assertIn("generation_command", report)
        self.assertTrue(
            report["generation_command"]["not_run_by_preflight"]
        )
        release_root = (
            preflight.REPO
            / "training/artifacts/releases"
            / "invariant_fusion_v3_sealed_seed20260736"
        )
        self.assertFalse(release_root.exists())


if __name__ == "__main__":
    unittest.main()
