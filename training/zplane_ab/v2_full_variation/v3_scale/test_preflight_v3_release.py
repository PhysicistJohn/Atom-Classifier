"""Unit suite for the seed-20260734 release preflight.

What is under test is the property that makes the preflight worth running at
all: a pin that does not match disk, a tampered artifact, a missing evaluator,
a wrong runtime or a spent seed must each surface as an itemised FAIL and a
NO-GO, while the real frozen candidate on this machine (with the real, pinned
v3 evaluator carrying the owner's gate redeclaration) must pass every check.

Hygiene properties asserted explicitly:

* the preflight consumes nothing: no release root is created, nothing is
  written under ``training/artifacts/releases``, and the report says so in
  machine-readable form;
* the source-tree digest is bit-compatible with the launcher's JavaScript
  walk (same skip list, same separator byte, symlink refusal), locked by an
  independent inline reimplementation rather than by calling the function
  under test;
* the reconstructed generation command is exactly the sealed mechanics
  re-aimed at seed 20260734 with 192 rows per class, and is never executed;
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


class FrozenPolicyCheckTest(TempDirTestCase):
    def test_frozen_constants_pass(self) -> None:
        checks = _by_name(preflight.check_frozen_policy(preflight.DEFAULT_PINS))
        self.assertTrue(all(item["passed"] for item in checks.values()), checks)
        self.assertIn("policy.import_identity_discipline", checks)
        self.assertIn("policy.seed_ledger_constants", checks)

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
        self.assertFalse(checks["sources.dependency_hashes_recorded"]["passed"])
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
        self.assertEqual(
            observed["dependency_source_sha256"][
                "v3_scale/evaluate_v3_release_suite.py"
            ],
            preflight._sha256(stub),
        )
        self.assertTrue(
            checks["sources.dependency_hashes_recorded"]["passed"]
        )

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


class SeedLedgerCheckTest(TempDirTestCase):
    FAKE_ROOTS = (
        "fake_sealed_v2_root",
        "fake_sealed_v3_root",
        "fake_sealed_v3_run2_root",
    )

    def _pins_with_fake_sealed(self, releases: Path) -> dict:
        pins = copy.deepcopy(preflight.DEFAULT_PINS)
        pins["consumed_sealed_roots"] = {}
        for root_name, recorded_seed in zip(
            self.FAKE_ROOTS, (20260729, 20260731, 20260733)
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
        checks = _by_name(preflight.check_seed_ledger(releases, pins))
        self.assertTrue(checks["ledger.release_seed_unused"]["passed"])
        for root_name in self.FAKE_ROOTS:
            self.assertTrue(
                checks[f"ledger.consumed_root_intact.{root_name}"]["passed"]
            )

    def test_seed_named_root_fails(self) -> None:
        releases = Path(self.tmpdir())
        pins = self._pins_with_fake_sealed(releases)
        (releases / "v3_sealed_seed20260734").mkdir()
        checks = _by_name(preflight.check_seed_ledger(releases, pins))
        self.assertFalse(checks["ledger.release_seed_unused"]["passed"])

    def test_intent_recording_seed_fails(self) -> None:
        releases = Path(self.tmpdir())
        pins = self._pins_with_fake_sealed(releases)
        _write(
            releases / "innocuous_name" / "RELEASE_INTENT.json",
            json.dumps({"release_seed": 20260734}).encode(),
        )
        checks = _by_name(preflight.check_seed_ledger(releases, pins))
        self.assertFalse(checks["ledger.release_seed_unused"]["passed"])

    def test_consumed_v3_root_is_not_flagged_as_seed_use(self) -> None:
        # The consumed seed-20260731/20260733 roots exist on disk by design;
        # only the NEW seed 20260734 may appear in no release root.
        releases = Path(self.tmpdir())
        pins = self._pins_with_fake_sealed(releases)
        _write(
            releases / "old_sealed" / "RELEASE_INTENT.json",
            json.dumps({"release_seed": 20260733}).encode(),
        )
        checks = _by_name(preflight.check_seed_ledger(releases, pins))
        self.assertTrue(checks["ledger.release_seed_unused"]["passed"])

    def test_tampered_sealed_reference_fails(self) -> None:
        releases = Path(self.tmpdir())
        pins = self._pins_with_fake_sealed(releases)
        target = releases / self.FAKE_ROOTS[1] / "RELEASE_EVALUATION.json"
        target.write_bytes(b'{"tampered": true}')
        checks = _by_name(preflight.check_seed_ledger(releases, pins))
        self.assertFalse(
            checks[f"ledger.consumed_root_intact.{self.FAKE_ROOTS[1]}"]["passed"]
        )
        self.assertTrue(
            checks[f"ledger.consumed_root_intact.{self.FAKE_ROOTS[0]}"]["passed"]
        )

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


class GenerationCommandTest(TempDirTestCase):
    """The reconstructed one-shot command: right values, never executed."""

    @unittest.skipUnless(
        preflight.DEFAULT_BUNDLE_DIR.is_dir(), "real runtime bundle missing"
    )
    def test_command_matches_sealed_mechanics_for_seed_20260734(self) -> None:
        generation = preflight.build_generation_command(
            preflight.DEFAULT_PINS,
            preflight.DEFAULT_BUNDLE_DIR,
            preflight.DEFAULT_ISOLATED_ROOT,
            preflight.DEFAULT_NODE_BIN_DIR,
        )
        self.assertTrue(generation["not_run_by_preflight"])
        environment = generation["environment"]
        self.assertEqual(environment["RELEASE_SEED"], "20260734")
        self.assertEqual(environment["RELEASE_EVALUATION_PROTOCOL"], "v3")
        self.assertEqual(environment["RELEASE_TARGET_PER_CLASS"], "192")
        self.assertEqual(
            environment["CANDIDATE_PATH"],
            str(preflight.DEFAULT_BUNDLE_DIR / "bundle_manifest.json"),
        )
        self.assertEqual(
            environment["CANDIDATE_SHA256"],
            preflight._sha256(
                preflight.DEFAULT_BUNDLE_DIR / "bundle_manifest.json"
            ),
        )
        self.assertEqual(
            environment["SIGNALLAB_ROOT"],
            str(preflight.DEFAULT_ISOLATED_ROOT),
        )
        self.assertIn(
            "invariant_fusion_v3_sealed_seed20260734",
            environment["RELEASE_ROOT"],
        )
        self.assertIn(
            "node tools/generate-signallab-iq-release-suite.mjs",
            generation["command"],
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
            "fusion_dir": str(preflight.DEFAULT_FUSION_DIR),
            "bundle_dir": str(preflight.DEFAULT_BUNDLE_DIR),
            "prefilter_root": str(preflight.DEFAULT_PREFILTER_ROOT),
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
            fusion_dir=str(gone / "fusion"),
            bundle_dir=str(gone / "bundle"),
            prefilter_root=str(gone / "prefilter"),
            releases_dir=str(gone / "releases"),
            isolated_root=str(gone / "isolated"),
            evaluator=str(gone / "eval.py"),
            node_bin_dir=str(gone / "bin"),
        )
        report = preflight.run_preflight(args)
        self.assertFalse(report["go"])
        names = {item["name"] for item in report["checks"]}
        # The crashed groups surface under their group labels.
        self.assertIn("fusion", names)
        self.assertIn("bundle", names)
        failed = {
            item["name"] for item in report["checks"] if not item["passed"]
        }
        self.assertIn("fusion", failed)
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


@unittest.skipUnless(
    preflight.DEFAULT_BUNDLE_DIR.is_dir()
    and preflight.DEFAULT_ISOLATED_ROOT.is_dir()
    and preflight.DEFAULT_NODE_BIN_DIR.is_dir(),
    "full frozen environment not present",
)
class EndToEndTest(TempDirTestCase):
    """One full run against the real frozen inputs and the REAL evaluator.

    This is exactly the run the orchestrator will perform.  It must be a GO,
    it must write a report exactly once, and it must leave the release tree
    untouched.  The evaluator is no longer stubbed: the seed-20260734
    preflight pins the real evaluator bytes (the ones carrying the owner's
    V3_GATE_REDECLARATION), so a stub would rightly NO-GO.
    """

    def test_full_preflight_is_go_with_real_evaluator(self) -> None:
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
        self.assertIn("safe to spend release seed 20260734", final)
        # Exactly one GO / NO-GO line, every check itemised above it.
        go_lines = [
            line
            for line in text.splitlines()
            if line.startswith("GO:") or line.startswith("NO-GO:")
        ]
        self.assertEqual(len(go_lines), 1)
        report = json.loads(output.read_text())
        self.assertTrue(report["go"])
        self.assertTrue(all(item["passed"] for item in report["checks"]))
        self.assertGreaterEqual(len(report["checks"]), 40)
        # The report carries full provenance: the real evaluator's hash both
        # observed and equal to the redeclaration pin.
        self.assertEqual(
            report["observed"]["v3_evaluator_sha256"],
            preflight._sha256(preflight.DEFAULT_EVALUATOR),
        )
        self.assertEqual(
            report["observed"]["v3_evaluator_sha256"],
            preflight.DEFAULT_PINS["v3_evaluator_sha256"],
        )
        self.assertIn(
            "v2_full_variation/evaluate_invariant_release_suite.py",
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
        # The generation command in the report is the v2 mechanics on the
        # new seed, and was not run: its release root must not exist.
        environment = report["generation_command"]["environment"]
        self.assertEqual(environment["RELEASE_SEED"], "20260734")
        self.assertFalse(Path(environment["RELEASE_ROOT"]).exists())


if __name__ == "__main__":
    unittest.main()
