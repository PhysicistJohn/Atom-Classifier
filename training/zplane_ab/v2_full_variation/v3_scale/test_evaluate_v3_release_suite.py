"""Tests for the sealed v3 release evaluator.

Run under warnings-as-errors, matching the repository convention:

    PYTHONWARNINGS=error \
    PYTHONPATH=training:training/zplane_ab:training/zplane_ab/v2_full_variation:training/zplane_ab/v2_full_variation/v3_scale \
    .venv-training/bin/python -m unittest -v test_evaluate_v3_release_suite

The heavy fixture is a synthetic MINI release suite plus a MINI candidate,
both built in a temporary directory (never under ``training/artifacts/
releases``), with the protocol constants rescaled coherently in both the v3
evaluator and the imported v2 machinery.  The end-to-end evaluation runs once
in ``setUpClass`` and every structural assertion reads that one report:

* full evaluator path: suite verification -> candidate loading -> additive
  and staged inference -> novelty -> sweeps -> gate assembly -> report;
* two-stage known false-unknown accounting (stage 1 gates at the fitted
  length; the length above the max fitted length is gated through the
  causal-prefix rule with the max-fitted-length bundle);
* the refuse-to-rerun guards;
* exact gate identity with the strict v2 floors (known-FUR ceiling 0.10,
  five-shot floor 0.85), plus the consumed seed-20260734 0.12/0.84
  redeclaration retained only as inactive historical metadata.

The mini corpus intentionally covers one fitted stage-1 capture length (256)
and one length above it with no fitted bundle (512), mirroring the real run's
N32768 prefix-gated case.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping
from unittest import mock

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
ZPAB = V2.parent
TRAINING = ZPAB.parent
for _path in (TRAINING, ZPAB, V2, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import evaluate_invariant_release_suite as release  # noqa: E402
import evaluate_v3_release_suite as evaluator  # noqa: E402
import export_v3_browser_weights as browser_export  # noqa: E402
import fit_v3_openset as openset_base  # noqa: E402
import fit_v3_openset_staged as staged  # noqa: E402
import noise_prefilter  # noqa: E402
import pose_degeneracy as pdg  # noqa: E402
import time_domain_geometry as geometry  # noqa: E402
from invariant_fusion import CenteredInvariantFusion  # noqa: E402
from invariant_patch_cnn import InvariantPatchCNN, InvariantPatchConfig  # noqa: E402
from known_only_patch_openset import KnownOnlyLOFOpenSet  # noqa: E402
from v3_time_domain_openset import FrozenV3OpenSet  # noqa: E402


MINI_LENGTHS = (256, 512)
MINI_MATCHED = 512
MINI_STAGE_ONE_LENGTHS = (256,)  # 512 is deliberately uncovered
MINI_TARGET_PER_CLASS = 12
MINI_MIN_TARGET = 8
MINI_NOVELTY_N = 6
MINI_SCALE_FACTORS = (0.5, 1.0, 2.0)
MINI_SCALE_MIN_PER_CLASS = 2
MINI_RELEASE_SEED = 777003
MINI_CLASSES = ("alpha", "beta")
MINI_EMBED = 8
PATCH_LENGTH = 64
PATCH_COUNT = 16
TARGET_FRAC = 0.5
N_FEATURES = 12
FAKE_SIGNALLAB_ROOT = "/nonexistent/mini-signallab-root"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _write_json(path: Path, payload: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _mini_protocol_patches() -> list[Any]:
    """Rescale the protocol coherently in BOTH the v3 and the v2 modules."""
    return [
        mock.patch.object(
            evaluator, "REQUIRED_CAPTURE_LENGTHS", MINI_LENGTHS
        ),
        mock.patch.object(evaluator, "MATCHED_CAPTURE_LENGTH", MINI_MATCHED),
        mock.patch.object(evaluator, "MIN_TARGET_PER_CLASS", MINI_MIN_TARGET),
        mock.patch.object(
            evaluator, "NOVELTY_N_EACH_PER_LENGTH", MINI_NOVELTY_N
        ),
        mock.patch.object(
            evaluator, "PHYSICAL_SCALE_FACTORS", MINI_SCALE_FACTORS
        ),
        mock.patch.object(
            evaluator, "SCALE_MIN_PER_CLASS", MINI_SCALE_MIN_PER_CLASS
        ),
        mock.patch.object(
            evaluator, "EXPECTED_RELEASE_SEED", MINI_RELEASE_SEED
        ),
        mock.patch.object(
            evaluator, "ENFORCE_CANONICAL_Q97_EVIDENCE", False
        ),
        mock.patch.object(
            staged, "RELEASE_SEED_NEVER_SPENT_HERE", MINI_RELEASE_SEED
        ),
        mock.patch.object(release, "REQUIRED_CAPTURE_LENGTHS", MINI_LENGTHS),
        mock.patch.object(release, "MATCHED_CAPTURE_LENGTH", MINI_MATCHED),
        mock.patch.object(
            release, "PHYSICAL_SCALE_FACTORS", MINI_SCALE_FACTORS
        ),
        mock.patch.object(
            release, "SCALE_MIN_PER_CLASS", MINI_SCALE_MIN_PER_CLASS
        ),
        mock.patch.object(
            release,
            "_validate_dependency_provenance",
            _stub_dependency_provenance,
        ),
    ]


def _stub_dependency_provenance(value: Any, runtime_value: Any) -> dict[str, Any]:
    """Test stand-in: the real validator needs the pinned isolated trees."""
    if not isinstance(value, Mapping) or "signallab" not in value:
        raise ValueError("dependency_provenance must be an object")
    return {
        "signallab": {"root": str(value["signallab"]["root"])},
        "atom_dsp": dict(value.get("atom_dsp", {})),
        "runtime": dict(runtime_value),
        "passes": True,
        "stubbed_for_unit_test": True,
    }


def _mini_capture(class_index: int, row: int, length: int) -> np.ndarray:
    """A deterministic tone-like capture: coherent, never all-zero."""
    rng = np.random.default_rng(9000 + class_index * 101 + row)
    n = np.arange(length, dtype=np.float64)
    if class_index == 0:
        tone = np.exp(1j * (2.0 * np.pi * 0.11 * n + 0.3 * row))
    else:
        tone = (1.0 + 0.4 * np.cos(2.0 * np.pi * 0.004 * n)) * np.exp(
            1j * (2.0 * np.pi * 0.23 * n + 0.1 * row)
        )
    noise = 0.05 * (rng.standard_normal(length) + 1j * rng.standard_normal(length))
    return (tone + noise).astype(np.complex128)


def _item(
    class_name: str,
    class_index: int,
    row: int,
    *,
    profile_salt: int,
) -> dict[str, Any]:
    combo = row % 4
    impaired = combo >= 2
    snr = {0: 22.0, 1: 10.0, 2: 25.0, 3: 8.0}[combo]
    return {
        "cls": class_name,
        "profile": f"mini-{class_name}-{row}-s{profile_salt}",
        "sampleRateHz": 1000000.0,
        "bandwidthHz": 50000.0,
        "impaired": impaired,
        "startSampleIndex": 0,
        "centerHz": 100000.0 + 1000.0 * row,
        "snrDb": snr,
        "rxPowerDbm": -50.0 - row,
        "noiseFigureDb": 5.0,
        "centreOffsetFrac": 0.03 + 0.001 * class_index,
        "adcBits": 12,
        "clockErrorPpm": 1.5,
        "multipathTaps": 0,
        "impairmentSeed": 4200 + class_index * 100 + row,
        "cleanPower": 1.0,
    }


def _query_covers_all_slices(
    items: list[dict[str, Any]], query: np.ndarray
) -> bool:
    for class_name in MINI_CLASSES:
        for want_impaired in (False, True):
            for want_high in (False, True):
                if not any(
                    items[int(index)]["cls"] == class_name
                    and items[int(index)]["impaired"] is want_impaired
                    and (items[int(index)]["snrDb"] >= 18.0) is want_high
                    for index in query
                ):
                    return False
    return True


class MiniFixture:
    """One mini candidate plus one mini sealed suite, built in a tmp root."""

    def __init__(self, base: Path):
        self.base = Path(base).resolve()
        self.candidate_root = self.base / "mini_candidate"
        self.suite_root = self.base / "mini_v3_suite"
        self.rejector_bundle_dir = self.candidate_root / "rejector_bundle"
        self.classifier_bundle_dir = self.candidate_root / "classifier_bundle"
        self.rejector_fusion_dir = self.candidate_root / "rejector_fusion"
        self.classifier_fusion_dir = self.candidate_root / "classifier_fusion"
        # Old fixture-local names continue to mean the 4k rejector role.
        self.bundle_dir = self.rejector_bundle_dir
        self.fusion_dir = self.rejector_fusion_dir
        self.staged_dir = self.candidate_root / "staged"
        self.prefilter_dir = self.candidate_root / "prefilters"
        torch.manual_seed(20260730)
        self._build_fusion()
        self._build_classifier_fusion()
        self._build_bundle()
        self._build_classifier_bundle()
        self._build_staged()
        self.build_suite(self.base / "mini_calibration_suite")
        # After the suite: the stage-1 operating point is placed against the
        # actual sealed captures so at least one known QUERY row is gated,
        # which is what makes the two-stage FUR accounting testable at all.
        self._build_prefilters()
        self._build_candidate_chain()
        self.build_suite(self.suite_root)

    def rebind_candidate_copy(self, target: Path) -> Path:
        """Copy the candidate tree and rebind its internal fusion path.

        The bundle manifest records the absolute fusion-artifact path it was
        exported from; a byte-for-byte copy therefore still points at the
        original, which ``load_candidate`` correctly refuses.  Tamper tests
        need a mutable copy that is otherwise self-consistent.
        """
        target = Path(target).resolve()
        shutil.copytree(self.candidate_root, target)
        paths = {
            "classifier_fusion": target / "classifier_fusion",
            "rejector_fusion": target / "rejector_fusion",
            "classifier_bundle": target / "classifier_bundle",
            "rejector_bundle": target / "rejector_bundle",
            "staged": target / "staged",
            "prefilters": target / "prefilters",
            "frozen": target / "frozen_candidate_contract.json",
            "design": target / "q97_design_report.json",
            "evidence": target / "validation_evidence.json",
            "manifest": target / "release_candidate_manifest.json",
            "classifier_browser": target / evaluator.CLASSIFIER_BROWSER_ASSET,
            "rejector_browser": target / evaluator.REJECTOR_BROWSER_ASSET,
            "openset_browser": target / evaluator.OPENSET_BROWSER_ASSET,
            "package": target / "staging_package_manifest.json",
            "binding": target / "time-domain-v3-dual-binding.json",
        }
        for role in ("classifier", "rejector"):
            bundle_manifest_path = paths[f"{role}_bundle"] / "bundle_manifest.json"
            bundle_manifest = json.loads(
                bundle_manifest_path.read_text(encoding="utf-8")
            )
            bundle_manifest["provenance"]["source_fusion_artifact"] = str(
                paths[f"{role}_fusion"]
            )
            _write_json(bundle_manifest_path, bundle_manifest)
            _write_json(
                paths[f"{role}_browser"],
                browser_export.build_payload(
                    {
                        "dir": paths[f"{role}_bundle"],
                        "manifest": bundle_manifest,
                    }
                ),
            )
        staged_metrics_path = paths["staged"] / "openset_metrics.json"
        staged_metrics = json.loads(staged_metrics_path.read_text(encoding="utf-8"))
        staged_metrics["fusion"]["directory"] = str(paths["rejector_fusion"])
        staged_metrics["stage_one"]["directory"] = str(paths["prefilters"])
        _write_json(staged_metrics_path, staged_metrics)

        frozen = json.loads(paths["frozen"].read_text(encoding="utf-8"))
        frozen["classifier_fusion_8k_regularized"]["directory"] = str(
            paths["classifier_fusion"]
        )
        frozen["rejector_fusion_4k"]["directory"] = str(paths["rejector_fusion"])
        frozen["stage_one_noise_prefilter"]["directory"] = str(paths["prefilters"])
        frozen["q97_design_evidence"]["path"] = str(paths["design"])
        frozen["q97_design_evidence"]["sha256"] = _sha256(paths["design"])
        _write_json(paths["frozen"], frozen)

        evidence = json.loads(paths["evidence"].read_text(encoding="utf-8"))
        evidence["candidate_contract"]["path"] = str(paths["frozen"])
        evidence["candidate_contract"]["sha256"] = _sha256(paths["frozen"])
        evidence["validation"]["report"] = str(staged_metrics_path)
        evidence["validation"]["report_sha256"] = _sha256(staged_metrics_path)
        evidence["validation_locked_policy_artifacts"]["directory"] = str(
            paths["staged"]
        )
        _write_json(paths["evidence"], evidence)

        binding = json.loads(paths["binding"].read_text(encoding="utf-8"))
        binding["roles"]["classifier"][
            "runtime_bundle_manifest_sha256"
        ] = _sha256(paths["classifier_bundle"] / "bundle_manifest.json")
        binding["roles"]["classifier"]["asset_sha256"] = _sha256(
            paths["classifier_browser"]
        )
        binding["roles"]["rejector"][
            "runtime_bundle_manifest_sha256"
        ] = _sha256(paths["rejector_bundle"] / "bundle_manifest.json")
        binding["roles"]["rejector"]["asset_sha256"] = _sha256(
            paths["rejector_browser"]
        )
        binding["openset_policy"][
            "fitted_rejector_runtime_bundle_manifest_sha256"
        ] = _sha256(paths["rejector_bundle"] / "bundle_manifest.json")
        binding["openset_policy"]["rejector_asset_sha256"] = _sha256(
            paths["rejector_browser"]
        )
        binding["openset_policy"][
            "staged_validation_report_sha256"
        ] = _sha256(staged_metrics_path)
        binding["validation"]["report_sha256"] = _sha256(staged_metrics_path)
        _write_json(paths["binding"], binding)

        package = json.loads(paths["package"].read_text(encoding="utf-8"))
        binding_asset = package["assets"][evaluator.DUAL_BINDING_ASSET]
        binding_asset.update(_record(paths["binding"]))
        package["dual_binding"] = dict(binding_asset)
        for role in ("classifier", "rejector"):
            asset_name = (
                evaluator.CLASSIFIER_BROWSER_ASSET
                if role == "classifier"
                else evaluator.REJECTOR_BROWSER_ASSET
            )
            asset_record = package["assets"][asset_name]
            asset_record.update(_record(paths[f"{role}_browser"]))
            package["roles"][role]["asset"] = dict(asset_record)
            package["roles"][role][
                "source_bundle_manifest_sha256"
            ] = _sha256(paths[f"{role}_bundle"] / "bundle_manifest.json")
        package["openset_policy"][
            "fitted_rejector_runtime_bundle_manifest_sha256"
        ] = _sha256(paths["rejector_bundle"] / "bundle_manifest.json")
        package["openset_policy"][
            "staged_validation_report_sha256"
        ] = _sha256(staged_metrics_path)
        _write_json(paths["package"], package)

        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        manifest["validation_evidence"] = {
            "path": str(paths["evidence"]),
            "sha256": _sha256(paths["evidence"]),
        }
        for role in ("classifier", "rejector"):
            manifest[role]["fusion"]["directory"] = str(paths[f"{role}_fusion"])
            bundle_manifest_path = paths[f"{role}_bundle"] / "bundle_manifest.json"
            manifest[role]["runtime_bundle"].update(
                {
                    "directory": str(paths[f"{role}_bundle"]),
                    "manifest_path": str(bundle_manifest_path),
                    "manifest_sha256": _sha256(bundle_manifest_path),
                }
            )
        manifest["staged_validation"].update(
            {
                "directory": str(paths["staged"]),
                "report_path": str(staged_metrics_path),
                "report_sha256": _sha256(staged_metrics_path),
            }
        )
        manifest["stage_one_prefilter"]["directory"] = str(paths["prefilters"])
        for role, key in (
            ("classifier", "classifier_browser"),
            ("rejector", "rejector_browser"),
            ("openset_policy", "openset_browser"),
        ):
            manifest["browser_assets"][role]["path"] = str(paths[key])
            manifest["browser_assets"][role]["sha256"] = _sha256(paths[key])
        manifest["staging_package_manifest"].update(
            {
                "path": str(paths["package"]),
                "sha256": _sha256(paths["package"]),
            }
        )
        manifest["dual_binding"]["path"] = str(paths["binding"])
        manifest["dual_binding"]["sha256"] = _sha256(paths["binding"])
        _write_json(paths["manifest"], manifest)
        return target

    # -- the mini fusion artifact -------------------------------------------

    def _build_fusion(self) -> None:
        config = dict(
            patch_length=PATCH_LENGTH,
            patch_count=PATCH_COUNT,
            patch_dim=16,
            hidden=16,
            embed_dim=MINI_EMBED,
            n_features=N_FEATURES,
            set_pool="mean_std",
            dropout=0.2,
        )
        real = InvariantPatchCNN(
            InvariantPatchConfig(encoder="real", **config).validate()
        ).eval()
        complex_branch = InvariantPatchCNN(
            InvariantPatchConfig(encoder="complex", **config).validate()
        ).eval()
        rng = np.random.default_rng(7)
        real_center = rng.normal(0, 0.1, MINI_EMBED).astype(np.float32)
        complex_center = rng.normal(0, 0.1, MINI_EMBED).astype(np.float32)
        fusion = CenteredInvariantFusion(
            real,
            complex_branch,
            real_center,
            complex_center,
            alpha_real=0.2,
            alpha_complex=0.2,
            weight_real=0.5,
            eps=1e-12,
        ).eval()
        self.fusion = fusion
        prototypes = rng.normal(0, 1.0, (len(MINI_CLASSES), 2 * MINI_EMBED))
        prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)
        self.prototypes = prototypes.astype(np.float32)
        self.feature_mean = np.zeros(N_FEATURES, dtype=np.float32)
        self.feature_std = np.ones(N_FEATURES, dtype=np.float32)

        directory = self.fusion_dir
        directory.mkdir(parents=True)
        arrays = {
            "fusion_prototypes.npy": self.prototypes,
            "real_center.npy": real_center,
            "complex_center.npy": complex_center,
            "feature_mean.npy": self.feature_mean,
            "feature_std.npy": self.feature_std,
        }
        for name, value in arrays.items():
            np.save(directory / name, value)
        torch.save(fusion.state_dict(), directory / "fusion_state_dict.pt")
        artifact_names = {
            "state_dict": "fusion_state_dict.pt",
            "prototypes": "fusion_prototypes.npy",
            "real_center": "real_center.npy",
            "complex_center": "complex_center.npy",
            "feature_mean": "feature_mean.npy",
            "feature_std": "feature_std.npy",
        }
        artifacts: dict[str, Any] = {}
        for key, file_name in artifact_names.items():
            artifacts[key] = file_name
            artifacts[f"{key}_sha256"] = _sha256(directory / file_name)
        architecture = {
            "kind": "centered_invariant_fusion",
            "real": dict(config, encoder="real"),
            "complex": dict(config, encoder="complex"),
            "alpha_real": float(fusion.alpha_real),
            "alpha_complex": float(fusion.alpha_complex),
            "weight_real": float(fusion.weight_real),
            "eps": float(fusion.eps),
            "embed_dim": 2 * MINI_EMBED,
        }
        metrics = {
            "encoder": "fusion",
            "development_only": True,
            "release_evidence": False,
            "sealed_release_data_used": 0,
            "consumed_test_rows_used": 0,
            "data_audit": {"consumed_test_rows_exposed": 0},
            "frontend": {
                "version": "invariant-patch-time-domain-v1",
                "uses_frequency_transform": False,
            },
            "architecture": architecture,
            "fusion": {"weight_real": 0.5},
            "artifacts": artifacts,
            "seed": 424242,
            "device": "cpu",
            "closed_selection": {"balanced_accuracy": 0.5},
            "parameter_count": int(
                sum(p.numel() for p in fusion.parameters())
            ),
            "source_index_contract": {
                "contract": {
                    "config": {
                        "patch_length": PATCH_LENGTH,
                        "patch_count": PATCH_COUNT,
                        "target_frac": TARGET_FRAC,
                    }
                }
            },
        }
        _write_json(directory / "dev_metrics.json", metrics)

    def _build_classifier_fusion(self) -> None:
        """Create a deliberately distinct, geometry-compatible classifier."""
        shutil.copytree(self.rejector_fusion_dir, self.classifier_fusion_dir)
        directory = self.classifier_fusion_dir
        prototypes = np.load(directory / "fusion_prototypes.npy")
        np.save(directory / "fusion_prototypes.npy", prototypes[::-1].copy())
        mean = np.load(directory / "feature_mean.npy")
        std = np.load(directory / "feature_std.npy")
        np.save(directory / "feature_mean.npy", mean + np.float32(0.25))
        np.save(directory / "feature_std.npy", std * np.float32(1.5))
        state_path = directory / "fusion_state_dict.pt"
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        first = next(
            name
            for name, value in state.items()
            if "branch." in name
            and name.endswith("weight")
            and torch.is_floating_point(value)
            and value.numel()
        )
        state[first] = state[first].clone()
        state[first].view(-1)[0] += 0.125
        torch.save(state, state_path)
        metrics_path = directory / "dev_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        for key in (
            "state_dict",
            "prototypes",
            "real_center",
            "complex_center",
            "feature_mean",
            "feature_std",
        ):
            name = metrics["artifacts"][key]
            metrics["artifacts"][f"{key}_sha256"] = _sha256(directory / name)
        metrics["seed"] = 434343
        _write_json(metrics_path, metrics)

    # -- the mini runtime bundle --------------------------------------------

    def _build_bundle(self) -> None:
        directory = self.bundle_dir
        directory.mkdir(parents=True)
        for name in (
            "fusion_prototypes.npy",
            "real_center.npy",
            "complex_center.npy",
            "feature_mean.npy",
            "feature_std.npy",
            "fusion_state_dict.pt",
        ):
            shutil.copy2(self.fusion_dir / name, directory / name)
        _write_json(directory / "probe_fixture.json", {"cases": []})
        assets = {
            name: _record(directory / name)
            for name in (
                "fusion_prototypes.npy",
                "real_center.npy",
                "complex_center.npy",
                "feature_mean.npy",
                "feature_std.npy",
                "fusion_state_dict.pt",
                "probe_fixture.json",
            )
        }
        assembly_sources = {
            name: _sha256(evaluator._source_path(name))
            for name in evaluator.BUNDLE_ASSEMBLY_SOURCE_HARD_CONTRACT
        }
        fusion_metrics = json.loads(
            (self.fusion_dir / "dev_metrics.json").read_text(encoding="utf-8")
        )
        manifest = {
            "kind": evaluator.BUNDLE_KIND,
            "schema": evaluator.BUNDLE_SCHEMA,
            "schema_version": evaluator.BUNDLE_SCHEMA_VERSION,
            "runtime_role": "known_unknown_rejector",
            "development_only": True,
            "release_evidence": False,
            "sealed_release_data_used": 0,
            "consumed_test_rows_used": 0,
            "not_compatible_with_schema_ids": [],
            "assets": assets,
            "classification": {
                "classes": list(MINI_CLASSES),
                "distance": "squared_euclidean",
                "label": "argmin over prototypes",
                "embedding": "centered_fusion",
            },
            "frontend": {
                "version": "invariant-patch-time-domain-v1",
                "estimator_version": (
                    evaluator.td_preprocess.ESTIMATOR_VERSION
                ),
                "patch_length": PATCH_LENGTH,
                "patch_count": PATCH_COUNT,
                "target_frac": TARGET_FRAC,
                "packed_length": PATCH_LENGTH * PATCH_COUNT,
                "feature_count": N_FEATURES,
                "uses_frequency_transform": False,
                "source_sha256": {
                    "training/time_domain_geometry.py": _sha256(
                        evaluator._source_path("time_domain_geometry.py")
                    ),
                    "training/time_domain_invariant_patch_preprocess.py": (
                        _sha256(
                            evaluator._source_path(
                                "time_domain_invariant_patch_preprocess.py"
                            )
                        )
                    ),
                },
            },
            "architecture": fusion_metrics["architecture"],
            "parameter_count": fusion_metrics["parameter_count"],
            "fusion": {
                "alpha_real": 0.2,
                "alpha_complex": 0.2,
                "weight_real": 0.5,
                "weight_complex": 0.5,
                "eps": 1e-12,
                "kind": "centered_invariant_fusion",
            },
            "feature_standardization": {
                "epsilon": 1e-8,
                "rule": "(raw_float32 - mean) / std, cast to float32",
            },
            "rejection": {
                "state": "unset",
                "fitted": False,
                "runtime_behaviour": (
                    "known/unknown decisions require the separately validated "
                    "staged v3 policy fitted against this exact rejector fusion"
                ),
                "external_staged_policy_required_for_abstention": True,
                "required_contract": {
                    "policy_schema": 1,
                    "policy_kind": (
                        "v3_known_only_lof_frequency_dispersion_rank_blend"
                    ),
                    "geometry_feature": "across_patch_frequency_dispersion",
                    "geometry_weight": 0.2,
                    "v2_weight": 0.8,
                    "threshold_quantile": 0.95,
                    "module": "v3_time_domain_openset",
                    "cannot_change_closed_label": True,
                    "density_fit_population": "training only",
                    "rank_and_threshold_population": "enrollment only",
                    "scope": "stage-one survivors only",
                    "must_be_fit_against": "this exact fusion assembly",
                    "this_bundle_supplies_known_unknown_decision": True,
                    "this_bundle_supplies_public_known_label": False,
                },
            },
            "self_verification": {
                "closed_label_agreement": True,
                "worst_max_abs_error": 0.0,
                "probe_case_count": 0,
            },
            "provenance": {
                "source_fusion_artifact": str(self.fusion_dir),
                "source_dev_metrics_sha256": _sha256(
                    self.fusion_dir / "dev_metrics.json"
                ),
                "assembly_source_sha256": assembly_sources,
                "assembly_seed": fusion_metrics["seed"],
            },
            "release_blockers": [
                "synthetic mini candidate requires its staged policy",
            ],
        }
        _write_json(directory / "bundle_manifest.json", manifest)

    def _build_classifier_bundle(self) -> None:
        shutil.copytree(self.rejector_bundle_dir, self.classifier_bundle_dir)
        directory = self.classifier_bundle_dir
        for name in (
            "fusion_prototypes.npy",
            "real_center.npy",
            "complex_center.npy",
            "feature_mean.npy",
            "feature_std.npy",
            "fusion_state_dict.pt",
        ):
            shutil.copy2(self.classifier_fusion_dir / name, directory / name)
        manifest_path = directory / "bundle_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name in manifest["assets"]:
            manifest["assets"][name] = _record(directory / name)
        manifest["provenance"]["source_fusion_artifact"] = str(
            self.classifier_fusion_dir
        )
        manifest["runtime_role"] = "accepted_known_classifier"
        manifest["provenance"]["source_dev_metrics_sha256"] = _sha256(
            self.classifier_fusion_dir / "dev_metrics.json"
        )
        manifest["provenance"]["assembly_seed"] = json.loads(
            (
                self.classifier_fusion_dir / "dev_metrics.json"
            ).read_text(encoding="utf-8")
        )["seed"]
        manifest["rejection"]["runtime_behaviour"] = (
            "run only after the role-bound rejector accepts the capture, and "
            "use only this bundle's nearest-prototype winner as the public "
            "known label"
        )
        manifest["rejection"]["required_contract"].update(
            {
                "must_be_fit_against": "known_unknown_rejector",
                "this_bundle_supplies_known_unknown_decision": False,
                "this_bundle_supplies_public_known_label": True,
                "classifier_runs_only_after_rejector_acceptance": True,
                "must_not_be_fit_against_this_classifier": True,
            }
        )
        _write_json(manifest_path, manifest)

    # -- the mini frozen stage-2 state --------------------------------------

    def _build_staged(self) -> None:
        directory = self.staged_dir
        directory.mkdir(parents=True)
        rng = np.random.default_rng(11)

        def packed_rows(count: int, offset: int) -> np.ndarray:
            rows = []
            for index in range(count):
                capture = _mini_capture(index % 2, offset + index, 1024)
                i = capture.real[: PATCH_LENGTH * PATCH_COUNT]
                q = capture.imag[: PATCH_LENGTH * PATCH_COUNT]
                rows.append(np.stack((i, q)).astype(np.float32))
            return np.stack(rows)

        train_packed = packed_rows(80, 0)
        train_labels = np.arange(80, dtype=np.int64) % 2
        enroll_packed = packed_rows(40, 500)
        enroll_labels = np.arange(40, dtype=np.int64) % 2

        components = [
            (
                branch,
                weight,
                KnownOnlyLOFOpenSet.fit(
                    rng.normal(0, 1, (80, MINI_EMBED)),
                    rng.normal(0, 1, (40, MINI_EMBED)),
                    neighbors=neighbors,
                ),
            )
            for branch, weight, neighbors in openset_base.BRANCH_LOF
        ]
        enrollment_lof = openset_base.score_branch_lof(
            components,
            {
                "real": rng.normal(0, 1, (40, MINI_EMBED)),
                "complex": rng.normal(0, 1, (40, MINI_EMBED)),
            },
        )
        policy = FrozenV3OpenSet.fit(
            train_packed,
            train_labels,
            enroll_packed,
            enroll_labels,
            enrollment_lof,
            n_classes=len(MINI_CLASSES),
        )
        self.policy = policy
        self.lof_components = components

        np.savez_compressed(
            directory / "v3_open_policy_stage_two.npz", **policy.to_payload()
        )
        openset_base.save_branch_lof(
            directory / "v3_branch_lof_components.npz", components
        )
        self._write_staged_metrics()

    def _write_staged_metrics(self) -> None:
        directory = self.staged_dir
        fusion_artifact = openset_base.load_fusion_artifact(self.fusion_dir)
        sources = {
            name: _sha256(evaluator._source_path(name))
            for name in (
                evaluator.STAGED_SOURCE_HARD_CONTRACT
                + evaluator.STAGED_SOURCE_RECORDED_ONLY
            )
        }
        metrics = {
            "status": "development_openset_pass",
            "role": "validate",
            "gates_are_evidence": True,
            "all_pass": True,
            "development_only": True,
            "release_evidence": False,
            "sealed_release_data_used": 0,
            "consumed_test_rows_used": 0,
            "release_seed_20260736_used": False,
            "additive_only": False,
            "changes_closed_label": True,
            "gates_before_classification": True,
            "closed_label_can_be_gated": True,
            "artifacts": {
                "v3_open_policy_stage_two.npz": _sha256(
                    directory / "v3_open_policy_stage_two.npz"
                ),
                "v3_branch_lof_components.npz": _sha256(
                    directory / "v3_branch_lof_components.npz"
                ),
                # the composite npz record is filled after the prefilters
                # (and therefore the composite policy) exist
            },
            "architecture": {
                "kind": staged.STAGED_POLICY_KIND,
                "schema": staged.STAGED_POLICY_SCHEMA,
                "staged_policy_version": staged.STAGED_POLICY_VERSION,
                "survivor_score": staged.COMPOSITE_SURVIVOR_SCORE,
            },
            "fusion": {
                "directory": str(self.fusion_dir),
                "directory_sha256": fusion_artifact.directory_sha256,
            },
            "stage_one": {
                "set_sha256": None,  # filled after the prefilters exist
                "capture_lengths": list(MINI_STAGE_ONE_LENGTHS),
                "directory": str(self.prefilter_dir),
            },
            "stage_two": {"threshold": float(self.policy.threshold)},
            "composite": {"threshold": None},  # filled with the prefilters
            "seeds": {
                "novelty_seeds": list(evaluator.VALIDATION_NOVELTY_SEEDS),
                "design_novelty_seed": evaluator.DESIGN_NOVELTY_SEED,
                "default_validation_novelty_seeds": list(
                    evaluator.VALIDATION_NOVELTY_SEEDS
                ),
                "release_seed_not_spent": evaluator.EXPECTED_RELEASE_SEED,
                "novelty_rows_used_to_choose_the_stage_one_operating_point": 0,
                "novelty_rows_used_to_choose_the_composite_threshold": 0,
            },
            "source_sha256": sources,
        }
        self._staged_metrics_template = metrics
        _write_json(directory / "openset_metrics.json", metrics)

    # -- the mini stage-1 prefilter set -------------------------------------

    def _build_prefilters(self) -> None:
        names = noise_prefilter.PREFILTER_FEATURES
        coefficients = np.zeros(len(names), dtype=np.float64)
        coefficients[names.index("carrier_phase_coherence")] = -1.0
        mean = np.full(len(names), 0.5, dtype=np.float64)
        scale = np.ones(len(names), dtype=np.float64)

        # Known stage-1 scores at the covered length, from the very captures
        # the mini corpus holds, so the fixture can place the operating point
        # to gate at least one known QUERY row (exercising two-stage FUR).
        # Support rows never gate anything in the report, so the threshold is
        # anchored to the best-scoring query row specifically.
        length = MINI_STAGE_ONE_LENGTHS[0]
        # The evaluator reads float32 corpus bytes; anchor the operating
        # point against those exact rounded values, not the float64 sources.
        known = [
            self._as_stored(self._corpus_capture(item, length))
            for item in self.items
        ]
        minimum = geometry.minimum_bandwidth_for_active_span(
            length, PATCH_LENGTH, TARGET_FRAC
        )
        features = np.stack(
            [
                pdg.pose_degeneracy_features(capture, min_bandwidth=minimum)
                for capture in known
            ]
        )
        selected = noise_prefilter.select_features(features, names)
        raw = (selected - mean) / scale @ coefficients
        query_scores = raw[np.asarray(self.query_indices, dtype=np.int64)]
        threshold = float(np.max(query_scores) - 1e-9)
        base_model = noise_prefilter.NoisePrefilter(
            feature_names=names,
            capture_length=length,
            mean=mean,
            scale=scale,
            coefficients=coefficients,
            intercept=0.0,
            threshold_score=threshold,
            fit_provenance={
                "known_population": noise_prefilter.TRAIN_SPLIT,
                "fitting_seed": 20261001,
                "fitting_rows": int(len(known)),
            },
            threshold_provenance={
                "population": noise_prefilter.ENROLLMENT_SPLIT,
                "policy": noise_prefilter.BUDGET_POLICY,
                "known_false_positive_budget": (
                    staged.STAGE_ONE_KNOWN_FALSE_POSITIVE_BUDGET
                ),
                "training_rows_used_for_threshold": 0,
                "selection_rows_used_for_threshold": 0,
                "novelty_rows_used_for_threshold": 0,
            },
        )
        noise_prefilter.save_prefilter_set(
            self.prefilter_dir, {length: base_model}
        )
        set_sha = noise_prefilter.prefilter_set_sha256(self.prefilter_dir)
        self._staged_metrics_template["stage_one"]["set_sha256"] = set_sha

        # The composite survivor policy (staged policy version 4), fit with
        # the real fitting code against the real loaded gate.  The synthetic
        # enrollment stage-1 features force at least one gated enrollment row
        # so the calibration is a strict survivor subset.
        gate = staged.load_stage_one(
            staged.load_prefilter_module(),
            staged.load_posedegen_module(),
            self.prefilter_dir,
            required_lengths=MINI_STAGE_ONE_LENGTHS,
        )
        rng = np.random.default_rng(23)
        enroll_rows = len(self.policy.calibration_scores)
        enroll_features = rng.normal(
            0.5, 0.2, (enroll_rows, pdg.FEATURE_COUNT)
        )
        # The fixture model scores 0.5 - carrier_phase_coherence (coefficient
        # -1, mean 0.5, unit scale, zero intercept), so placing the coherence
        # column relative to the anchored threshold places each row's gate
        # outcome deterministically: survivors strictly below, row 0 above.
        coherence_column = pdg.FEATURE_NAMES.index("carrier_phase_coherence")
        enroll_features[:, coherence_column] = (
            0.5 - threshold + rng.uniform(0.01, 0.30, enroll_rows)
        )
        enroll_features[0, coherence_column] = 0.5 - threshold - 5.0
        self.composite = staged.fit_composite_policy(
            gate,
            enroll_features,
            length,
            np.asarray(self.policy.calibration_scores, dtype=np.float64),
            stage_two_threshold=self.policy.threshold,
        )
        composite_path = self.staged_dir / staged.COMPOSITE_POLICY_FILENAME
        staged.save_composite_policy(composite_path, self.composite)
        self._staged_metrics_template["artifacts"][
            staged.COMPOSITE_POLICY_FILENAME
        ] = _sha256(composite_path)
        self._staged_metrics_template["composite"] = self.composite.provenance()
        _write_json(
            self.staged_dir / "openset_metrics.json",
            self._staged_metrics_template,
        )

    def _build_candidate_chain(self) -> None:
        """Freeze the mini equivalent of the real dual release identity."""
        classifier_artifact = openset_base.load_fusion_artifact(
            self.classifier_fusion_dir
        )
        rejector_artifact = openset_base.load_fusion_artifact(
            self.rejector_fusion_dir
        )
        self.frozen_contract_path = (
            self.candidate_root / "frozen_candidate_contract.json"
        )
        self.design_report_path = (
            self.candidate_root / "q97_design_report.json"
        )
        design_report = copy.deepcopy(self._staged_metrics_template)
        design_report.update(
            {
                "status": "design_selection_pass",
                "role": "design",
                "gates_are_evidence": False,
                "all_pass": True,
            }
        )
        design_report["seeds"]["novelty_seeds"] = [
            evaluator.DESIGN_NOVELTY_SEED
        ]
        _write_json(self.design_report_path, design_report)
        generic_prefilter_sha = "7" * 64
        frozen_contract = {
            "schema": evaluator.CANDIDATE_CONTRACT_SCHEMA,
            "status": "frozen_before_validation",
            "candidate_id": evaluator.CANDIDATE_ID,
            "architecture": {
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
            },
            "q97_policy": evaluator.q97_policy_contract_block(),
            "q97_design_evidence": {
                "path": str(self.design_report_path),
                "sha256": _sha256(self.design_report_path),
                "role": "design",
                "status": "design_selection_pass",
                "novelty_seed": evaluator.DESIGN_NOVELTY_SEED,
                "source_sha256": design_report["source_sha256"][
                    "fit_v3_openset_staged.py"
                ],
                "commit": "mini-design-evidence",
            },
            "source_transition_intent": (
                evaluator.source_transition_intent_block()
            ),
            "classifier_fusion_8k_regularized": {
                "directory": str(self.classifier_fusion_dir),
                "directory_sha256": classifier_artifact.directory_sha256,
                "file_sha256": dict(classifier_artifact.file_sha256),
            },
            "rejector_fusion_4k": {
                "directory": str(self.rejector_fusion_dir),
                "directory_sha256": rejector_artifact.directory_sha256,
                "file_sha256": dict(rejector_artifact.file_sha256),
            },
            "stage_one_noise_prefilter": {
                "directory": str(self.prefilter_dir),
                "directory_sha256": generic_prefilter_sha,
            },
        }
        _write_json(self.frozen_contract_path, frozen_contract)

        staged_metrics_path = self.staged_dir / "openset_metrics.json"
        staged_hashes = dict(self._staged_metrics_template["artifacts"])
        prefilter_set_sha = noise_prefilter.prefilter_set_sha256(
            self.prefilter_dir
        )
        bundle_sha256 = {
            str(length): noise_prefilter.bundle_sha256(
                self.prefilter_dir / f"N{length}"
            )
            for length in MINI_STAGE_ONE_LENGTHS
        }
        self.validation_evidence_path = (
            self.candidate_root / "validation_evidence.json"
        )
        evidence = {
            "schema": evaluator.CANDIDATE_EVIDENCE_SCHEMA,
            "status": "development_openset_pass",
            "candidate_id": evaluator.CANDIDATE_ID,
            "candidate_contract": {
                "path": str(self.frozen_contract_path),
                "sha256": _sha256(self.frozen_contract_path),
                "committed_before_validation": True,
                "commit": "mini-pre-validation",
            },
            "validation": {
                "role": "validate",
                "novelty_seeds_consumed_once": list(
                    evaluator.VALIDATION_NOVELTY_SEEDS
                ),
                "report": str(staged_metrics_path),
                "report_sha256": _sha256(staged_metrics_path),
                "gates_are_evidence": True,
                "all_pass": True,
                "sealed_release_data_used": 0,
                "consumed_test_rows_used": 0,
                "release_seed_20260736_used": False,
            },
            "validated_rejector": {
                "fusion_directory_sha256": rejector_artifact.directory_sha256,
                "canonical_prefilter_set_sha256": prefilter_set_sha,
                "prefilter_bundle_sha256": bundle_sha256,
            },
            "validation_locked_policy_artifacts": {
                "directory": str(self.staged_dir),
                **staged_hashes,
                "novelty_rows_used_to_fit_rank_or_threshold": 0,
            },
            "candidate_contract_clarification": {
                "recorded_value": generic_prefilter_sha,
                "canonical_behavioral_hash": prefilter_set_sha,
                "changes_candidate_behavior": False,
            },
            "source_transition_evidence": (
                evaluator.ordered_source_transition_contract_block()
            ),
        }
        _write_json(self.validation_evidence_path, evidence)

        stage_one = staged.load_stage_one(
            staged.load_prefilter_module(),
            staged.load_posedegen_module(),
            self.prefilter_dir,
            required_lengths=MINI_STAGE_ONE_LENGTHS,
        )
        browser_records: dict[str, dict[str, Any]] = {}
        for role, file_name in (
            ("classifier", evaluator.CLASSIFIER_BROWSER_ASSET),
            ("rejector", evaluator.REJECTOR_BROWSER_ASSET),
            ("openset_policy", evaluator.OPENSET_BROWSER_ASSET),
        ):
            path = self.candidate_root / file_name
            schema = (
                evaluator.BROWSER_OPENSET_SCHEMA
                if role == "openset_policy"
                else evaluator.BROWSER_FUSION_SCHEMA
            )
            schema_version = (
                evaluator.BROWSER_OPENSET_SCHEMA_VERSION
                if role == "openset_policy"
                else evaluator.BROWSER_FUSION_SCHEMA_VERSION
            )
            status = evaluator.STAGING_STATUS
            if role == "openset_policy":
                payload = json.loads(
                    evaluator.openset_export.json_bytes(
                        evaluator.openset_export.openset_weights_payload(
                            stage_one.models,
                            noise_prefilter.prefilter_set_sha256(
                                self.prefilter_dir
                            ),
                            self.lof_components,
                            self.policy,
                            self.composite,
                            {
                                "version": (
                                    "invariant-patch-time-domain-v1"
                                ),
                                "patch_length": PATCH_LENGTH,
                                "patch_count": PATCH_COUNT,
                                "target_frac": TARGET_FRAC,
                                "packed_length": (
                                    PATCH_LENGTH * PATCH_COUNT
                                ),
                                "uses_frequency_transform": False,
                            },
                            {"fixture": "mini-candidate"},
                        ),
                        compact=True,
                    )
                )
            else:
                bundle_dir = (
                    self.classifier_bundle_dir
                    if role == "classifier"
                    else self.rejector_bundle_dir
                )
                payload = browser_export.build_payload(
                    {
                        "dir": bundle_dir,
                        "manifest": json.loads(
                            (
                                bundle_dir / "bundle_manifest.json"
                            ).read_text(encoding="utf-8")
                        ),
                    }
                )
            _write_json(path, payload)
            browser_records[role] = {
                "path": str(path),
                "sha256": _sha256(path),
                "schema": schema,
                "status": status,
            }
        package_path = self.candidate_root / "staging_package_manifest.json"
        binding_path = self.candidate_root / "time-domain-v3-dual-binding.json"
        _write_json(
            binding_path,
            {
                "schema": evaluator.DUAL_BINDING_SCHEMA,
                "schema_version": evaluator.DUAL_BINDING_SCHEMA_VERSION,
                "status": evaluator.STAGING_STATUS,
                "candidate_id": evaluator.CANDIDATE_ID,
                "frontend": {
                    "version": "invariant-patch-time-domain-v1",
                    "patch_length": PATCH_LENGTH,
                    "patch_count": PATCH_COUNT,
                    "target_frac": TARGET_FRAC,
                },
                "execution_order": [
                    "stage_one_noise_gate",
                    "rejector_known_unknown",
                    "classifier_known_label",
                ],
                "roles": {
                    "classifier": {
                        "asset": Path(browser_records["classifier"]["path"]).name,
                        "asset_sha256": browser_records["classifier"]["sha256"],
                        "fusion_directory_sha256": (
                            classifier_artifact.directory_sha256
                        ),
                        "runtime_bundle_manifest_sha256": _sha256(
                            self.classifier_bundle_dir / "bundle_manifest.json"
                        ),
                        "runtime_role": "accepted_known_classifier",
                        "responsibility": "accepted_known_label_only",
                    },
                    "rejector": {
                        "asset": Path(browser_records["rejector"]["path"]).name,
                        "asset_sha256": browser_records["rejector"]["sha256"],
                        "fusion_directory_sha256": (
                            rejector_artifact.directory_sha256
                        ),
                        "runtime_bundle_manifest_sha256": _sha256(
                            self.rejector_bundle_dir / "bundle_manifest.json"
                        ),
                        "runtime_role": "known_unknown_rejector",
                        "responsibility": "known_unknown_only",
                    },
                },
                "openset_policy": {
                    "asset": Path(
                        browser_records["openset_policy"]["path"]
                    ).name,
                    "asset_sha256": browser_records["openset_policy"]["sha256"],
                    "rejector_asset_sha256": browser_records["rejector"][
                        "sha256"
                    ],
                    "fitted_rejector_runtime_bundle_manifest_sha256": _sha256(
                        self.rejector_bundle_dir / "bundle_manifest.json"
                    ),
                    "staged_validation_report_sha256": _sha256(
                        staged_metrics_path
                    ),
                    "staged_artifacts_sha256": staged_hashes,
                },
                "validation": {
                    "report_sha256": _sha256(staged_metrics_path),
                    "role": "validate",
                    "status": "development_openset_pass",
                    "novelty_seeds": list(
                        evaluator.VALIDATION_NOVELTY_SEEDS
                    ),
                    "design_novelty_seed": evaluator.DESIGN_NOVELTY_SEED,
                    "release_seed_not_spent": (
                        evaluator.EXPECTED_RELEASE_SEED
                    ),
                },
                "fail_closed": {
                    "role_assets_bound_by_sha256": True,
                    "distinct_role_assets": True,
                    "role_asset_sha256_must_differ": True,
                    "classifier_runs_only_after_rejector_acceptance": True,
                    "public_known_label_from_classifier_only": True,
                },
            },
        )

        def asset_record(
            role: str,
            path: Path,
            *,
            schema: str,
            schema_version: int,
            runtime_role: str | None = None,
        ) -> dict[str, Any]:
            record = {
                "path": path.name,
                **_record(path),
                "schema": schema,
                "schema_version": schema_version,
                "status": evaluator.STAGING_STATUS,
            }
            if runtime_role is not None:
                record["runtime_role"] = runtime_role
            return record

        classifier_path = Path(browser_records["classifier"]["path"])
        rejector_path = Path(browser_records["rejector"]["path"])
        openset_path = Path(browser_records["openset_policy"]["path"])
        package_assets = {
            evaluator.CLASSIFIER_BROWSER_ASSET: asset_record(
                "classifier",
                classifier_path,
                schema=evaluator.BROWSER_FUSION_SCHEMA,
                schema_version=evaluator.BROWSER_FUSION_SCHEMA_VERSION,
                runtime_role="accepted_known_classifier",
            ),
            evaluator.REJECTOR_BROWSER_ASSET: asset_record(
                "rejector",
                rejector_path,
                schema=evaluator.BROWSER_FUSION_SCHEMA,
                schema_version=evaluator.BROWSER_FUSION_SCHEMA_VERSION,
                runtime_role="known_unknown_rejector",
            ),
            evaluator.OPENSET_BROWSER_ASSET: asset_record(
                "openset_policy",
                openset_path,
                schema=evaluator.BROWSER_OPENSET_SCHEMA,
                schema_version=evaluator.BROWSER_OPENSET_SCHEMA_VERSION,
            ),
            evaluator.DUAL_BINDING_ASSET: asset_record(
                "dual_binding",
                binding_path,
                schema=evaluator.DUAL_BINDING_SCHEMA,
                schema_version=evaluator.DUAL_BINDING_SCHEMA_VERSION,
            ),
        }

        external_paths: dict[str, Path] = {}
        for directory_name in (
            "rejector_export",
            "classifier_export",
            "openset_export",
        ):
            (self.candidate_root / directory_name).mkdir()
        external_payloads = {
            "rejector_export_manifest": (
                self.candidate_root / "rejector_export" / "export-manifest.json",
                {
                    "schema": evaluator.FUSION_EXPORT_MANIFEST_SCHEMA,
                    "schema_version": (
                        evaluator.FUSION_EXPORT_MANIFEST_SCHEMA_VERSION
                    ),
                    "runtime_role": "known_unknown_rejector",
                },
            ),
            "classifier_export_manifest": (
                self.candidate_root
                / "classifier_export"
                / "export-manifest.json",
                {
                    "schema": evaluator.FUSION_EXPORT_MANIFEST_SCHEMA,
                    "schema_version": (
                        evaluator.FUSION_EXPORT_MANIFEST_SCHEMA_VERSION
                    ),
                    "runtime_role": "accepted_known_classifier",
                },
            ),
            "openset_export_manifest": (
                self.candidate_root / "openset_export" / "manifest.json",
                {
                    "schema": evaluator.OPENSET_EXPORT_MANIFEST_SCHEMA,
                    "schema_version": (
                        evaluator.OPENSET_EXPORT_MANIFEST_SCHEMA_VERSION
                    ),
                    "status": evaluator.STAGING_STATUS,
                },
            ),
            "rejector_probe": (
                self.candidate_root
                / "rejector_export"
                / "time-domain-v3-rejector-probe.json",
                {"schema": "synthetic-rejector-probe", "cases": []},
            ),
            "classifier_probe": (
                self.candidate_root
                / "classifier_export"
                / "time-domain-v3-classifier-probe.json",
                {"schema": "synthetic-classifier-probe", "cases": []},
            ),
            "parity": (
                self.candidate_root
                / "openset_export"
                / "time-domain-openset-parity-v1.json",
                {
                    "schema": evaluator.PARITY_SCHEMA,
                    "schema_version": evaluator.PARITY_SCHEMA_VERSION,
                    "status": evaluator.STAGING_STATUS,
                },
            ),
        }
        for name, (path, payload) in external_payloads.items():
            _write_json(path, payload)
            external_paths[name] = path

        def external_record(name: str) -> dict[str, Any]:
            path = external_paths[name]
            record = {
                "path": path.relative_to(package_path.parent).as_posix(),
                **_record(path),
            }
            if name == "rejector_export_manifest":
                record.update(
                    {
                        "schema": evaluator.FUSION_EXPORT_MANIFEST_SCHEMA,
                        "schema_version": (
                            evaluator.FUSION_EXPORT_MANIFEST_SCHEMA_VERSION
                        ),
                        "runtime_role": "known_unknown_rejector",
                    }
                )
            elif name == "classifier_export_manifest":
                record.update(
                    {
                        "schema": evaluator.FUSION_EXPORT_MANIFEST_SCHEMA,
                        "schema_version": (
                            evaluator.FUSION_EXPORT_MANIFEST_SCHEMA_VERSION
                        ),
                        "runtime_role": "accepted_known_classifier",
                    }
                )
            elif name == "openset_export_manifest":
                record.update(
                    {
                        "schema": evaluator.OPENSET_EXPORT_MANIFEST_SCHEMA,
                        "schema_version": (
                            evaluator.OPENSET_EXPORT_MANIFEST_SCHEMA_VERSION
                        ),
                        "status": evaluator.STAGING_STATUS,
                    }
                )
            elif name == "parity":
                record.update(
                    {
                        "schema": evaluator.PARITY_SCHEMA,
                        "schema_version": evaluator.PARITY_SCHEMA_VERSION,
                        "status": evaluator.STAGING_STATUS,
                        "packaged": False,
                    }
                )
            return record

        _write_json(
            package_path,
            {
                "schema": evaluator.STAGING_PACKAGE_SCHEMA,
                "schema_version": evaluator.STAGING_PACKAGE_SCHEMA_VERSION,
                "status": evaluator.STAGING_STATUS,
                "candidate_id": evaluator.CANDIDATE_ID,
                "architecture": {
                    "execution_order": [
                        "stage_one_noise_gate",
                        "rejector_known_unknown",
                        "classifier_known_label",
                    ],
                    "classifier_runs_only_after_rejector_acceptance": True,
                    "public_known_label_from_classifier_only": True,
                },
                "assets": package_assets,
                "roles": {
                    "classifier": {
                        "runtime_role": "accepted_known_classifier",
                        "responsibility": "accepted_known_label_only",
                        "asset": package_assets[
                            evaluator.CLASSIFIER_BROWSER_ASSET
                        ],
                        "source_bundle_manifest_sha256": _sha256(
                            self.classifier_bundle_dir / "bundle_manifest.json"
                        ),
                        "fusion_directory_sha256": (
                            classifier_artifact.directory_sha256
                        ),
                    },
                    "rejector": {
                        "runtime_role": "known_unknown_rejector",
                        "responsibility": "known_unknown_only",
                        "asset": package_assets[
                            evaluator.REJECTOR_BROWSER_ASSET
                        ],
                        "source_bundle_manifest_sha256": _sha256(
                            self.rejector_bundle_dir / "bundle_manifest.json"
                        ),
                        "fusion_directory_sha256": (
                            rejector_artifact.directory_sha256
                        ),
                    },
                },
                "openset_policy": {
                    **package_assets[evaluator.OPENSET_BROWSER_ASSET],
                    "fitted_rejector_runtime_bundle_manifest_sha256": _sha256(
                        self.rejector_bundle_dir / "bundle_manifest.json"
                    ),
                    "staged_validation_report_sha256": _sha256(
                        staged_metrics_path
                    ),
                    "staged_artifacts_sha256": staged_hashes,
                },
                "dual_binding": package_assets[evaluator.DUAL_BINDING_ASSET],
                "external_evidence": {
                    name: external_record(name) for name in external_paths
                },
                "size_contract": {
                    "maximum_file_bytes_exclusive": 25 * 1024 * 1024,
                    "all_deployable_files_below_limit": True,
                },
            },
        )

        def bundle_record(directory: Path) -> dict[str, Any]:
            manifest_path = directory / "bundle_manifest.json"
            return {
                "directory": str(directory),
                "manifest_path": str(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
            }

        self.candidate_manifest_path = (
            self.candidate_root / "release_candidate_manifest.json"
        )
        manifest = {
            "schema": evaluator.CANDIDATE_MANIFEST_SCHEMA,
            "status": "release_candidate_frozen",
            "candidate_id": evaluator.CANDIDATE_ID,
            "validation_evidence": {
                "path": str(self.validation_evidence_path),
                "sha256": _sha256(self.validation_evidence_path),
            },
            "classifier": {
                "role": "known_class_label",
                "fusion": {
                    "directory": str(self.classifier_fusion_dir),
                    "directory_sha256": classifier_artifact.directory_sha256,
                    "file_sha256": dict(classifier_artifact.file_sha256),
                },
                "runtime_bundle": bundle_record(self.classifier_bundle_dir),
            },
            "rejector": {
                "role": "known_unknown_decision",
                "fusion": {
                    "directory": str(self.rejector_fusion_dir),
                    "directory_sha256": rejector_artifact.directory_sha256,
                    "file_sha256": dict(rejector_artifact.file_sha256),
                },
                "runtime_bundle": bundle_record(self.rejector_bundle_dir),
            },
            "staged_validation": {
                "directory": str(self.staged_dir),
                "report_path": str(staged_metrics_path),
                "report_sha256": _sha256(staged_metrics_path),
                "artifact_sha256": staged_hashes,
            },
            "stage_one_prefilter": {
                "directory": str(self.prefilter_dir),
                "set_sha256": prefilter_set_sha,
                "bundle_sha256": bundle_sha256,
            },
            "browser_assets": browser_records,
            "staging_package_manifest": {
                "path": str(package_path),
                "sha256": _sha256(package_path),
                "schema": evaluator.STAGING_PACKAGE_SCHEMA,
                "status": evaluator.STAGING_STATUS,
            },
            "dual_binding": {
                "path": str(binding_path),
                "sha256": _sha256(binding_path),
                "schema": evaluator.DUAL_BINDING_SCHEMA,
            },
        }
        _write_json(self.candidate_manifest_path, manifest)

    # -- the mini sealed suite ----------------------------------------------

    @staticmethod
    def _as_stored(capture: np.ndarray) -> np.ndarray:
        """Round-trip a capture through the cf32le stored representation."""
        i = np.asarray(capture.real, dtype=np.float32).astype(np.float64)
        q = np.asarray(capture.imag, dtype=np.float32).astype(np.float64)
        return i + 1j * q

    @staticmethod
    def _corpus_capture(item: Mapping[str, Any], length: int) -> np.ndarray:
        """The deterministic capture a manifest row denotes, at any length.

        Always generated once at the longest suite length and sliced, exactly
        as the sealed corpus derives its shorter rows, so the prefilter
        fixture sees the same prefix bytes the evaluator will read.  Shared
        between the corpus writer and the prefilter fixture.
        """
        class_index = MINI_CLASSES.index(str(item["cls"]))
        row = int(item["impairmentSeed"]) % 100
        return _mini_capture(class_index, row, MINI_LENGTHS[-1])[:length]

    def build_suite(self, root: Path) -> Path:
        """Write a complete, self-consistent sealed mini suite at ``root``."""
        root = Path(root).resolve()
        root.mkdir(parents=True)
        longest = MINI_LENGTHS[-1]

        for salt in range(64):
            items = [
                _item(name, class_index, row, profile_salt=salt)
                for class_index, name in enumerate(MINI_CLASSES)
                for row in range(MINI_TARGET_PER_CLASS)
            ]
            support, query, _report = release.predeclared_five_shot_split(
                items, MINI_CLASSES, MINI_RELEASE_SEED
            )
            if _query_covers_all_slices(items, query):
                break
        else:  # pragma: no cover - the deterministic search always lands
            raise AssertionError("no salt kept every query slice populated")
        self.items = items
        self.support_indices = support
        self.query_indices = query

        observed = np.zeros((len(items), longest, 2), dtype=np.float32)
        clean = np.zeros_like(observed)
        for index, item in enumerate(items):
            capture = self._corpus_capture(item, longest)
            observed[index, :, 0] = capture.real.astype(np.float32)
            observed[index, :, 1] = capture.imag.astype(np.float32)
            clean[index, :, 0] = observed[index, :, 0] * 0.9
            clean[index, :, 1] = observed[index, :, 1] * 0.9

        corpora_records = []
        for length in MINI_LENGTHS:
            directory = root / f"n{length}"
            directory.mkdir()
            observed[:, :length].tofile(directory / "corpus.f32")
            clean[:, :length].tofile(directory / "corpus_clean.f32")
            manifest = {
                "format": "cf32le-interleaved",
                "sampleCount": length,
                "corpusSeed": MINI_RELEASE_SEED,
                "targetPerClass": MINI_TARGET_PER_CLASS,
                "exactTargetPerClass": True,
                "hasCleanPairs": True,
                "signalLabRoot": FAKE_SIGNALLAB_ROOT,
                "classes": list(MINI_CLASSES),
                "count": len(items),
                "items": items,
            }
            if length == longest:
                manifest["matchedStartsFrom"] = str(
                    root / f"_start_probe_n{MINI_LENGTHS[0]}" / "corpus.json"
                )
            _write_json(directory / "corpus.json", manifest)
            corpora_records.append(
                {
                    "capture_length": length,
                    "derivation": (
                        "generated_once_at_longest_length"
                        if length == longest
                        else "bit_exact_row_prefix_of_longest"
                    ),
                    "directory": str(directory),
                    "files": {
                        name: _record(directory / name)
                        for name in (
                            "corpus.json",
                            "corpus.f32",
                            "corpus_clean.f32",
                        )
                    },
                }
            )
        # Shorter manifests bind the longest corpus bytes; written after the
        # longest record exists so byte hashes are final.
        longest_dir = root / f"n{longest}"
        for length in MINI_LENGTHS[:-1]:
            directory = root / f"n{length}"
            manifest = json.loads(
                (directory / "corpus.json").read_text(encoding="utf-8")
            )
            manifest["prefixDerivation"] = {
                "protocol": "signallab-row-prefix-v1",
                "sourceDirectory": str(longest_dir),
                "sourceSampleCount": longest,
                "targetSampleCount": length,
                "rowCount": len(items),
                "bytesPerComplexSample": 8,
                "sourceFiles": {
                    name: _record(longest_dir / name)
                    for name in (
                        "corpus.json",
                        "corpus.f32",
                        "corpus_clean.f32",
                    )
                },
                "outputFiles": {
                    name: _record(directory / name)
                    for name in ("corpus.f32", "corpus_clean.f32")
                },
            }
            _write_json(directory / "corpus.json", manifest)
            for record in corpora_records:
                if record["capture_length"] == length:
                    record["files"]["corpus.json"] = _record(
                        directory / "corpus.json"
                    )

        probe_dir = root / f"_start_probe_n{MINI_LENGTHS[0]}"
        probe_dir.mkdir()
        observed[:, : MINI_LENGTHS[0]].tofile(probe_dir / "corpus.f32")
        clean[:, : MINI_LENGTHS[0]].tofile(probe_dir / "corpus_clean.f32")
        _write_json(
            probe_dir / "corpus.json",
            {
                "format": "cf32le-interleaved",
                "sampleCount": MINI_LENGTHS[0],
                "corpusSeed": MINI_RELEASE_SEED,
                "targetPerClass": MINI_TARGET_PER_CLASS,
                "exactTargetPerClass": True,
                "hasCleanPairs": True,
                "signalLabRoot": FAKE_SIGNALLAB_ROOT,
                "classes": list(MINI_CLASSES),
                "count": len(items),
                "items": items,
            },
        )
        start_probe = {
            "scored": False,
            "purpose": (
                "select one occupied start per row before longest-only "
                "nuisance realization"
            ),
            "capture_length": MINI_LENGTHS[0],
            "directory": str(probe_dir),
            "files": {
                name: _record(probe_dir / name)
                for name in ("corpus.json", "corpus.f32", "corpus_clean.f32")
            },
        }

        protocol = evaluator.expected_evaluation_protocol(
            MINI_RELEASE_SEED, MINI_STAGE_ONE_LENGTHS
        )
        candidate_path = getattr(
            self,
            "candidate_manifest_path",
            self.bundle_dir / "bundle_manifest.json",
        )
        source_records = {
            "launcher": {
                "path": str(release.LAUNCHER),
                "sha256": _sha256(release.LAUNCHER),
            },
            "corpus_generator": {
                "path": str(release.CORPUS_GENERATOR),
                "sha256": _sha256(release.CORPUS_GENERATOR),
            },
            "prefix_deriver": {
                "path": str(release.PREFIX_DERIVER),
                "sha256": _sha256(release.PREFIX_DERIVER),
            },
            "tsx_package": "tsx@4.20.3",
        }
        intent = {
            "protocol": release.RELEASE_PROTOCOL,
            "status": "in_progress",
            "development_data_used": False,
            "candidate_path": str(candidate_path),
            "candidate_sha256": _sha256(candidate_path),
            "release_seed": MINI_RELEASE_SEED,
            "capture_lengths": list(MINI_LENGTHS),
            "target_per_class": MINI_TARGET_PER_CLASS,
            "exact_target_per_class": True,
            "evaluation_protocol": protocol,
            "source_sha256": source_records,
            "dependency_provenance": {
                "signallab": {"root": FAKE_SIGNALLAB_ROOT},
                "atom_dsp": {"root": FAKE_SIGNALLAB_ROOT + "/../Atom-DSP"},
            },
            "runtime_provenance": {
                "node": "v22.23.1",
                "npm": "10.9.8",
                "npx": "10.9.8",
            },
            "started_at": "2026-07-27T00:00:00.000Z",
        }
        _write_json(root / "RELEASE_INTENT.json", intent)
        manifest = dict(intent)
        manifest["status"] = "complete"
        manifest["release_intent_sha256"] = _sha256(
            root / "RELEASE_INTENT.json"
        )
        manifest["corpora"] = corpora_records
        manifest["start_probe"] = start_probe
        _write_json(root / "RELEASE_MANIFEST.json", manifest)
        return root


class MiniSuiteEndToEndTests(unittest.TestCase):
    """One full evaluation of the mini suite; every assertion reads it."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._stack = contextlib.ExitStack()
        cls.addClassCleanup(cls._stack.close)
        cls.tmp = Path(
            cls._stack.enter_context(tempfile.TemporaryDirectory())
        )
        for patcher in _mini_protocol_patches():
            cls._stack.enter_context(patcher)
        cls.fixture = MiniFixture(cls.tmp)
        cls.device = torch.device("cpu")
        cls.candidate = evaluator.load_candidate(
            candidate_manifest_path=cls.fixture.candidate_manifest_path,
            classifier_bundle_dir=cls.fixture.classifier_bundle_dir,
            rejector_bundle_dir=cls.fixture.rejector_bundle_dir,
            classifier_fusion_dir=cls.fixture.classifier_fusion_dir,
            rejector_fusion_dir=cls.fixture.rejector_fusion_dir,
            staged_dir=cls.fixture.staged_dir,
            prefilter_dir=cls.fixture.prefilter_dir,
            device=cls.device,
        )
        cls.suite = evaluator.load_v3_release_suite(
            cls.fixture.suite_root, cls.candidate
        )
        cls.report = evaluator.evaluate_release(
            cls.suite, cls.candidate, cls.device
        )
        cls.report_again = evaluator.evaluate_release(
            cls.suite, cls.candidate, cls.device
        )
        evaluator._write_json_exclusive(
            cls.fixture.suite_root / "RELEASE_EVALUATION.json", cls.report
        )

    # -- structure -----------------------------------------------------------

    def test_report_carries_the_v2_shape_plus_architecture_contract(self):
        report = self.report
        for key in (
            "closed_per_length",
            "family_per_length",
            "high_snr_per_length",
            "low_snr_per_length",
            "clean_subset_per_length",
            "impaired_subset_per_length",
            "five_shot_predeclared_per_length",
            "open_staged_per_length",
            "matched_length_sweep",
            "physical_scale_sweep",
            "gates",
            "all_release_gates_pass",
            "provenance",
            "architecture_contract",
        ):
            self.assertIn(key, report)
        self.assertTrue(report["release_evidence"])
        self.assertEqual(report["schema"], 4)
        self.assertEqual(
            report["evaluation_version"], evaluator.EVALUATION_VERSION
        )
        self.assertFalse(report["development_data_loaded"])
        self.assertFalse(report["retraining_performed"])
        self.assertFalse(report["recalibration_performed"])
        contract = report["architecture_contract"]
        for key, expected in (
            ("additive_only", False),
            ("changes_closed_label", True),
            ("open_set_decision_changes_closed_label", True),
            ("gates_before_classification", True),
        ):
            self.assertEqual(contract[key], expected)
        self.assertEqual(
            contract["architecture_contract_change"],
            noise_prefilter.ARCHITECTURE_CONTRACT_CHANGE,
        )
        self.assertEqual(
            contract["staged_policy_version"],
            evaluator.EXPECTED_STAGED_POLICY_VERSION,
        )
        self.assertEqual(
            contract["staged_policy_schema"],
            evaluator.EXPECTED_STAGED_POLICY_SCHEMA,
        )
        self.assertEqual(
            contract["staged_policy_kind"],
            evaluator.EXPECTED_STAGED_POLICY_KIND,
        )
        self.assertEqual(contract["composite_threshold_quantile"], 0.97)
        self.assertEqual(
            contract["policy_hygiene"], evaluator.EXPECTED_POLICY_HYGIENE
        )
        self.assertEqual(
            contract["survivor_score"], staged.COMPOSITE_SURVIVOR_SCORE
        )
        self.assertEqual(
            contract["stage_one_coverage_by_length"],
            {"256": "fitted", "512": "causal_prefix"},
        )
        self.assertEqual(
            contract["stage_one_feature_length_by_length"],
            {"256": 256, "512": 256},
        )
        self.assertEqual(
            contract["stage_one_prefix_rule"],
            {
                "max_fitted_length": 256,
                "rule": evaluator.STAGE_ONE_PREFIX_RULE,
            },
        )
        for length in MINI_LENGTHS:
            self.assertIn(str(length), report["closed_per_length"])
            self.assertIn(str(length), report["open_staged_per_length"])
            self.assertIn(
                str(length), report["five_shot_predeclared_per_length"]
            )

    def test_gate_set_is_exactly_the_v2_declared_set(self):
        expected = {
            "prefix_nesting",
            "dependency_provenance",
            "start_probe_excluded",
            "candidate_sha_bound",
            "closed_fine_worst_length",
            "closed_family_worst_length",
            "closed_high_snr_worst_length",
            "closed_clean_worst_length",
            "five_shot_worst_length_balanced",
            "open_auroc_overall_worst_length",
            "open_auroc_noise_worst_length",
            "open_auroc_chirp_worst_length",
            "open_known_false_unknown_worst_length",
            "open_unknown_recall_noise_worst_length",
            "open_unknown_recall_chirp_worst_length",
            "length_worst_balanced_accuracy",
            "length_max_mean_pairwise_cosine",
            "length_worst_prediction_agreement_to_matched",
            "length_worst_embedding_cosine_to_matched",
            "physical_scale_worst_balanced_accuracy",
            "physical_scale_max_mean_pairwise_cosine",
            "physical_scale_worst_prediction_agreement_to_factor1",
            "physical_scale_worst_embedding_cosine_to_factor1",
        }
        self.assertEqual(set(self.report["gates"]), expected)
        for name, gate in self.report["gates"].items():
            self.assertIn("passes", gate, name)

    def test_gate_values_and_floors_come_from_the_v2_helpers(self):
        gates = self.report["gates"]
        floors = release.GATE_FLOORS
        self.assertEqual(
            gates["closed_fine_worst_length"]["threshold"],
            floors["closed_fine"],
        )
        self.assertEqual(
            gates["open_auroc_noise_worst_length"]["threshold"],
            floors["open_auroc_noise"],
        )
        self.assertEqual(
            gates["open_known_false_unknown_worst_length"]["comparison"], "max"
        )
        worst_closed = min(
            float(row["accuracy"])
            for row in self.report["closed_per_length"].values()
        )
        self.assertEqual(
            gates["closed_fine_worst_length"]["value"], worst_closed
        )
        worst_fur = max(
            float(row["known_false_unknown_rate"])
            for row in self.report["open_staged_per_length"].values()
        )
        self.assertEqual(
            gates["open_known_false_unknown_worst_length"]["value"], worst_fur
        )

    def test_gate_identity_with_v2_is_exact_and_strict(self):
        """The current emitted gates are the v2 assembly byte for byte."""
        gates = self.report["gates"]
        rebuilt = release._assemble_release_gates(
            candidate_sha256=self.suite.candidate_sha256,
            expected_candidate_sha256=self.suite.release_manifest[
                "candidate_sha256"
            ],
            prefix_nesting_passes=True,
            dependency_provenance_passes=True,
            start_probe_excluded=True,
            closed_by_length=self.report["closed_per_length"],
            five_shot_by_length=self.report[
                "five_shot_predeclared_per_length"
            ],
            open_by_length=self.report["open_staged_per_length"],
            length_sweep=self.report["matched_length_sweep"],
            scale_sweep=self.report["physical_scale_sweep"],
        )
        self.assertEqual(rebuilt, gates)
        self.assertEqual(
            gates["open_known_false_unknown_worst_length"]["threshold"], 0.10
        )
        self.assertEqual(
            gates["five_shot_worst_length_balanced"]["threshold"], 0.85
        )
        for gate in gates.values():
            self.assertNotIn("redeclaration", gate)

    def test_report_and_protocol_carry_inactive_historical_redeclaration(self):
        expected = evaluator.historical_gate_redeclaration_block()
        self.assertEqual(self.report["historical_gate_redeclaration"], expected)
        self.assertEqual(
            self.report["provenance"]["evaluation_protocol"][
                "historical_gate_redeclaration"
            ],
            expected,
        )
        self.assertEqual(
            self.suite.intent["evaluation_protocol"][
                "historical_gate_redeclaration"
            ],
            expected,
        )
        self.assertEqual(expected["release_seed"], 20260734)
        self.assertFalse(expected["active_for_current_protocol"])
        self.assertEqual(
            expected["current_protocol_gate_source"],
            "imported_v2_gate_floors_unchanged",
        )
        for entry in expected["gates_redeclared"].values():
            self.assertEqual(
                entry["owner_decision"],
                evaluator.HISTORICAL_V3_GATE_REDECLARATION_RATIONALE,
            )
        correction = expected["historical_claim_correction"]
        self.assertEqual(
            correction["schema"],
            evaluator.HISTORICAL_CLAIM_CORRECTION_SCHEMA,
        )
        self.assertTrue(correction["original_text_preserved_verbatim"])
        self.assertEqual(correction["comparison_design"], "cross_seed_unpaired")
        self.assertFalse(correction["same_rows_or_same_novelty_draw"])
        self.assertEqual(correction["v2_failed_gate_count"], 8)
        self.assertEqual(
            correction["historical_claimed_v2_failed_gate_count"], 6
        )
        self.assertFalse(
            correction["every_identically_measured_axis_claim_supported"]
        )
        self.assertFalse(
            correction["all_six_gates_claim_supported_as_complete_count"]
        )
        self.assertIn(
            "re-declared for the v3 architecture by the owner on 2026-07-28",
            evaluator.HISTORICAL_V3_GATE_REDECLARATION_RATIONALE,
        )
        self.assertIn(
            "BEFORE release seed 20260734 was generated",
            evaluator.HISTORICAL_V3_GATE_REDECLARATION_RATIONALE,
        )

    # -- the staged decision path -------------------------------------------

    def test_two_stage_known_false_unknown_accounting(self):
        covered = self.report["open_staged_per_length"]["256"]
        self.assertTrue(covered["stage_one_active"])
        by_stage = covered["known_false_unknown_by_stage"]
        self.assertGreater(
            by_stage["stage_one_gate_rate"],
            0.0,
            "the fixture places the stage-1 operating point so at least one "
            "known query row is gated; two-stage accounting is not exercised "
            "otherwise",
        )
        self.assertAlmostEqual(
            by_stage["staged_false_unknown_rate"],
            by_stage["stage_one_gate_rate"]
            + by_stage["stage_two_false_unknown_rate_marginal"],
            places=12,
        )
        self.assertEqual(
            covered["known_false_unknown_rate"],
            by_stage["staged_false_unknown_rate"],
        )
        # A gated known row is always a staged false-unknown, whatever the
        # composite threshold does to the survivors.
        self.assertGreaterEqual(
            covered["known_false_unknown_rate"],
            by_stage["stage_one_gate_rate"],
        )
        # The two arms are decided on their own axes and thresholds.
        self.assertEqual(
            by_stage["staged_threshold"],
            float(self.candidate.threshold),
        )
        self.assertEqual(
            by_stage["unstaged_threshold"],
            float(self.candidate.stage_two_threshold),
        )
        self.assertEqual(
            covered["unstaged_control"]["threshold"],
            float(self.candidate.stage_two_threshold),
        )

    def test_inactive_stage_one_uses_the_exact_active_attribution_schema(self):
        score = np.asarray(
            [
                self.candidate.threshold - 0.01,
                self.candidate.threshold + 0.01,
                self.candidate.threshold + 0.02,
            ],
            dtype=np.float64,
        )
        inactive = evaluator._staged_scores(
            self.candidate,
            [],
            np.zeros((3, 1), dtype=np.float32),
            np.zeros((3, 1), dtype=np.float32),
            np.zeros(3, dtype=np.int64),
            score,
            self.device,
            capture_length=128,
            population="unit-inactive",
        )
        inactive_by_stage = inactive["false_unknown_by_stage"]
        active_by_stage = self.report["open_staged_per_length"]["256"][
            "known_false_unknown_by_stage"
        ]
        self.assertEqual(set(inactive_by_stage), set(active_by_stage))
        self.assertNotIn("rejected_by_stage_one_only", inactive_by_stage)
        self.assertEqual(inactive_by_stage["rejected_by_stage_one_total"], 0)
        self.assertEqual(
            inactive_by_stage["stage_one_also_rejected_by_unstaged_control"],
            0,
        )
        self.assertEqual(inactive_by_stage["staged_rejected_total"], 2)
        self.assertEqual(inactive_by_stage["rejected_by_stage_two_only"], 2)
        self.assertEqual(inactive_by_stage["attribution_partition_total"], 2)
        self.assertTrue(inactive_by_stage["attribution_partition_exact"])

    def test_length_above_max_fitted_is_gated_through_the_causal_prefix(self):
        """N512 has no fitted bundle; the N256 bundle gates its 256-prefixes.

        The suite's N256 rows are bit-exact prefixes of its N512 rows (for
        known rows via the sealed prefix derivation, for novelty rows by
        construction), so the stage-1 decisions MUST be identical at both
        lengths: that is the causal-prefix rule doing exactly what it claims.
        """
        covered = self.report["open_staged_per_length"]["256"]
        prefixed = self.report["open_staged_per_length"]["512"]
        self.assertTrue(prefixed["stage_one_active"])
        self.assertTrue(prefixed["stage_one_prefix_rule_applied"])
        self.assertFalse(covered["stage_one_prefix_rule_applied"])
        self.assertEqual(covered["stage_one_feature_length"], 256)
        self.assertEqual(prefixed["stage_one_feature_length"], 256)
        self.assertGreater(prefixed["known_gated_fraction"], 0.0)
        self.assertEqual(
            prefixed["known_gated_fraction"],
            covered["known_gated_fraction"],
        )
        for family in ("noise", "chirp"):
            self.assertEqual(
                prefixed[f"gated_fraction_{family}"],
                covered[f"gated_fraction_{family}"],
            )
        by_stage = prefixed["known_false_unknown_by_stage"]
        self.assertEqual(
            by_stage["stage_one_gate_rate"],
            covered["known_false_unknown_by_stage"]["stage_one_gate_rate"],
        )
        self.assertAlmostEqual(
            by_stage["staged_false_unknown_rate"],
            by_stage["stage_one_gate_rate"]
            + by_stage["stage_two_false_unknown_rate_marginal"],
            places=12,
        )
        # Stage-1 gating means the staged axis is not the additive control:
        # a gated row is always a staged rejection.
        self.assertGreaterEqual(
            prefixed["known_false_unknown_rate"],
            by_stage["stage_one_gate_rate"],
        )
        self.assertGreaterEqual(
            prefixed["flagged_unknown_noise"],
            prefixed["gated_fraction_noise"],
        )

    def test_staged_survivors_score_identically_to_the_additive_control(self):
        for length in MINI_LENGTHS:
            row = self.report["open_staged_per_length"][str(length)]
            agreement = row["subset_scoring_agreement"]
            for population in ("known", "noise", "chirp"):
                self.assertTrue(
                    agreement[population]["exact"],
                    f"n{length} {population} staged/additive drift",
                )

    def test_open_family_metrics_are_on_the_staged_axis(self):
        """Pin the four family-gate inputs (auroc/recall x noise/chirp) to the
        staged axis.  A novelty row gated at stage 1 is placed in (1, 2), so it
        MUST count toward threshold recall and MUST outrank every ungated known
        row in AUROC; both bounds are derived from the report's own stage
        accounting.  A regression that silently reads the additive control axis
        (which scores these same rows without the gate) cannot satisfy them:
        verified by mutation, the additive noise numbers here are 0.0 recall
        and ~0.46 AUROC against staged 1.0/1.0."""
        for length in MINI_LENGTHS:
            row = self.report["open_staged_per_length"][str(length)]
            gate_rate = row["known_false_unknown_by_stage"][
                "stage_one_gate_rate"
            ]
            for family in ("noise", "chirp"):
                gated = row[f"gated_fraction_{family}"]
                self.assertGreaterEqual(
                    row[f"flagged_unknown_{family}"],
                    gated,
                    f"n{length} {family}: a stage-1-gated novelty row must "
                    "count toward threshold recall",
                )
                self.assertGreaterEqual(
                    row[f"auroc_{family}"],
                    (1.0 - gate_rate) * gated,
                    f"n{length} {family}: every gated novelty row must "
                    "outrank every ungated known row on the staged axis",
                )
        covered = self.report["open_staged_per_length"]["256"]
        control = covered["unstaged_control"]
        # Fixture anchor: at the covered length every novelty row is gated
        # while the additive control flags none of the noise, so the two axes
        # are far apart exactly where the sealed noise gates read.
        self.assertEqual(covered["gated_fraction_noise"], 1.0)
        self.assertGreater(
            covered["flagged_unknown_noise"],
            control["flagged_unknown_noise"],
        )
        self.assertGreater(covered["auroc_noise"], control["auroc_noise"])

    def test_stage_one_gates_noise_at_the_covered_length(self):
        covered = self.report["open_staged_per_length"]["256"]
        self.assertGreater(covered["gated_fraction_noise"], 0.0)
        counts = self.report["staged_known_decisions_per_length"]
        self.assertGreater(counts["256"]["gated_at_stage_one"], 0)
        # Under the causal-prefix rule the N512 known rows gate identically
        # to their bit-exact N256 prefixes.
        self.assertEqual(
            counts["512"]["gated_at_stage_one"],
            counts["256"]["gated_at_stage_one"],
        )
        self.assertEqual(
            counts["256"]["rows"], int(len(self.suite.query_indices))
        )

    def test_classifier_and_rejector_roles_are_independent(self):
        self.assertFalse(
            np.array_equal(
                self.candidate.classifier_feature_mean,
                self.candidate.rejector_feature_mean,
            )
        )
        self.assertFalse(
            np.array_equal(
                self.candidate.classifier_feature_std,
                self.candidate.rejector_feature_std,
            )
        )
        self.assertFalse(
            np.array_equal(
                self.candidate.classifier_fusion_artifact.prototypes,
                self.candidate.rejector_fusion_artifact.prototypes,
            )
        )
        counts = self.report["staged_known_decisions_per_length"]
        for length in MINI_LENGTHS:
            row = counts[str(length)]
            self.assertEqual(
                row["accepted_label_source"],
                "classifier_fusion_8k_regularized",
            )
        classifier_prediction = np.asarray([1, 0, 1], dtype=np.int64)
        rejector_internal_prediction = np.asarray([0, 1, 0], dtype=np.int64)
        rejected = np.asarray([False, True, False])
        final = evaluator._final_labels_from_classifier(
            classifier_prediction, rejected
        )
        np.testing.assert_array_equal(final, [1, -1, 1])
        self.assertFalse(
            np.array_equal(final[~rejected], rejector_internal_prediction[~rejected])
        )

    # -- determinism and provenance -----------------------------------------

    def test_evaluation_is_deterministic(self):
        first = json.dumps(self.report, sort_keys=True, allow_nan=False)
        second = json.dumps(
            self.report_again, sort_keys=True, allow_nan=False
        )
        self.assertEqual(first, second)

    def test_provenance_pins_every_candidate_component(self):
        provenance = self.report["provenance"]
        components = self.report["candidate"]["components"]
        self.assertEqual(
            provenance["candidate_sha256"],
            _sha256(self.fixture.candidate_manifest_path),
        )
        self.assertEqual(
            components["candidate_contract"]["sha256"],
            _sha256(self.fixture.candidate_manifest_path),
        )
        self.assertEqual(
            components["classifier_fusion"]["directory_sha256"],
            self.candidate.classifier_fusion_artifact.directory_sha256,
        )
        self.assertEqual(
            components["rejector_fusion"]["directory_sha256"],
            self.candidate.rejector_fusion_artifact.directory_sha256,
        )
        self.assertNotEqual(
            components["classifier_fusion"]["directory_sha256"],
            components["rejector_fusion"]["directory_sha256"],
        )
        self.assertEqual(
            components["stage_one_prefilter"]["set_sha256"],
            noise_prefilter.prefilter_set_sha256(self.fixture.prefilter_dir),
        )
        self.assertEqual(
            set(components["staged_validation"]["artifact_sha256"]),
            {
                "v3_open_policy_stage_two.npz",
                "v3_branch_lof_components.npz",
                staged.COMPOSITE_POLICY_FILENAME,
            },
        )
        self.assertEqual(
            components["stage_one_prefilter"]["capture_lengths"], [256]
        )
        self.assertEqual(
            components["staged_validation"]["policy_version"],
            staged.STAGED_POLICY_VERSION,
        )
        self.assertEqual(
            components["staged_validation"]["staged_threshold"],
            float(self.fixture.composite.threshold),
        )
        self.assertEqual(
            components["staged_validation"]["stage_two_threshold"],
            float(self.fixture.policy.threshold),
        )
        self.assertEqual(
            components["staged_validation"]["composite"]["threshold"],
            float(self.fixture.composite.threshold),
        )
        self.assertEqual(
            components["dual_binding_sha256"],
            components["dual_binding"]["sha256"],
        )
        self.assertEqual(
            set(components["browser_assets"]),
            {"classifier", "rejector", "openset_policy"},
        )
        self.assertEqual(
            provenance["evaluation_protocol"],
            self.suite.intent["evaluation_protocol"],
        )
        self.assertIn("evaluator_sha256", provenance)
        self.assertIn("v2_evaluator_sha256", provenance)
        enforced = components["source_sha256"]["enforced"]
        for name in evaluator.STAGED_SOURCE_HARD_CONTRACT:
            self.assertIn(name, enforced)
        self.assertIn(
            "run_time_domain_dev.py",
            components["source_sha256"]["recorded_only"],
        )

    def test_novelty_is_derived_from_the_release_seed(self):
        provenance = self.report["provenance"]["novelty_prefix_generation"]
        self.assertEqual(provenance["base_seed"], MINI_RELEASE_SEED)
        self.assertEqual(
            provenance["derived_seed_by_family"]["noise"],
            evaluator._novelty_seed(MINI_RELEASE_SEED, "noise"),
        )
        self.assertEqual(provenance["rows_per_family"], MINI_NOVELTY_N)

    def test_five_shot_uses_the_predeclared_split(self):
        split = self.report["provenance"]["predeclared_split"]
        for class_name in MINI_CLASSES:
            self.assertEqual(
                len(split["support_by_class"][class_name]),
                release.FIVE_SHOT_K,
            )
        for length in MINI_LENGTHS:
            row = self.report["five_shot_predeclared_per_length"][str(length)]
            self.assertEqual(
                row["support_rows"], len(self.suite.support_indices)
            )
            self.assertEqual(row["query_rows"], len(self.suite.query_indices))

    # -- refuse-to-rerun guards ---------------------------------------------

    def test_release_root_with_existing_evaluation_is_refused(self):
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            evaluator.load_v3_release_suite(
                self.fixture.suite_root, self.candidate
            )

    def test_exclusive_writer_refuses_existing_output(self):
        target = self.fixture.suite_root / "RELEASE_EVALUATION.json"
        with self.assertRaises(FileExistsError):
            evaluator._write_json_exclusive(target, {"schema": 1})


class SuiteTamperTests(unittest.TestCase):
    """Adversarial mutations of a fresh mini suite must all be refused."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._stack = contextlib.ExitStack()
        cls.addClassCleanup(cls._stack.close)
        cls.tmp = Path(
            cls._stack.enter_context(tempfile.TemporaryDirectory())
        )
        for patcher in _mini_protocol_patches():
            cls._stack.enter_context(patcher)
        cls.fixture = MiniFixture(cls.tmp)
        cls.candidate = evaluator.load_candidate(
            candidate_manifest_path=cls.fixture.candidate_manifest_path,
            classifier_bundle_dir=cls.fixture.classifier_bundle_dir,
            rejector_bundle_dir=cls.fixture.rejector_bundle_dir,
            classifier_fusion_dir=cls.fixture.classifier_fusion_dir,
            rejector_fusion_dir=cls.fixture.rejector_fusion_dir,
            staged_dir=cls.fixture.staged_dir,
            prefilter_dir=cls.fixture.prefilter_dir,
            device=torch.device("cpu"),
        )

    def _copy_suite(self) -> Path:
        # A byte copy would carry the original root's absolute paths inside
        # the sealed manifests, which the loader correctly refuses; a fresh
        # deterministic build at a new root is the pristine equivalent.
        target = Path(tempfile.mkdtemp(dir=self.tmp)) / "suite"
        return self.fixture.build_suite(target)

    def test_pristine_copy_loads(self):
        suite = evaluator.load_v3_release_suite(
            self._copy_suite(), self.candidate
        )
        self.assertEqual(suite.release_seed, MINI_RELEASE_SEED)
        self.assertEqual(suite.classes, MINI_CLASSES)
        self.assertEqual(
            len(suite.support_indices),
            release.FIVE_SHOT_K * len(MINI_CLASSES),
        )

    def _mutate_intent(self, root: Path, mutate) -> None:
        intent_path = root / "RELEASE_INTENT.json"
        intent = json.loads(intent_path.read_text(encoding="utf-8"))
        mutate(intent)
        _write_json(intent_path, intent)
        manifest_path = root / "RELEASE_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, value in intent.items():
            if key in manifest:
                manifest[key] = value
        manifest["status"] = "complete"
        manifest["release_intent_sha256"] = _sha256(intent_path)
        _write_json(manifest_path, manifest)

    def test_tampered_evaluation_protocol_is_refused(self):
        root = self._copy_suite()

        def mutate(intent):
            intent["evaluation_protocol"]["gates"]["closed_fine"] = 0.1

        self._mutate_intent(root, mutate)
        with self.assertRaisesRegex(ValueError, "evaluation_protocol"):
            evaluator.load_v3_release_suite(root, self.candidate)

    def test_intent_without_historical_redeclaration_is_refused(self):
        root = self._copy_suite()

        def mutate(intent):
            del intent["evaluation_protocol"]["historical_gate_redeclaration"]

        self._mutate_intent(root, mutate)
        with self.assertRaisesRegex(
            ValueError, "no matching historical_gate_redeclaration block"
        ):
            evaluator.load_v3_release_suite(root, self.candidate)

    def test_intent_with_a_tampered_historical_level_is_refused(self):
        root = self._copy_suite()

        def mutate(intent):
            intent["evaluation_protocol"]["historical_gate_redeclaration"][
                "gates_redeclared"
            ]["open_known_false_unknown_worst_length"]["v3_level"] = 0.5

        self._mutate_intent(root, mutate)
        with self.assertRaisesRegex(
            ValueError, "no matching historical_gate_redeclaration block"
        ):
            evaluator.load_v3_release_suite(root, self.candidate)

    def test_intent_with_a_tampered_historical_correction_is_refused(self):
        root = self._copy_suite()

        def mutate(intent):
            correction = intent["evaluation_protocol"][
                "historical_gate_redeclaration"
            ]["historical_claim_correction"]
            correction["v2_failed_gate_count"] = 6

        self._mutate_intent(root, mutate)
        with self.assertRaisesRegex(
            ValueError, "no matching historical_gate_redeclaration block"
        ):
            evaluator.load_v3_release_suite(root, self.candidate)

    def test_wrong_candidate_sha_is_refused(self):
        root = self._copy_suite()

        def mutate(intent):
            intent["candidate_sha256"] = "0" * 64

        self._mutate_intent(root, mutate)
        with self.assertRaisesRegex(ValueError, "candidate"):
            evaluator.load_v3_release_suite(root, self.candidate)

    def test_candidate_path_outside_the_bundle_is_refused(self):
        root = self._copy_suite()
        rogue = self.tmp / "rogue_candidate.json"
        rogue.write_text("{}\n", encoding="utf-8")

        def mutate(intent):
            intent["candidate_path"] = str(rogue)
            intent["candidate_sha256"] = _sha256(rogue)

        self._mutate_intent(root, mutate)
        with self.assertRaisesRegex(ValueError, "dual release candidate"):
            evaluator.load_v3_release_suite(root, self.candidate)

    def test_consumed_release_seed_is_refused(self):
        root = self._copy_suite()

        def mutate(intent):
            intent["release_seed"] = 20260729
            intent["evaluation_protocol"]["novelty"]["seed"] = 20260729

        self._mutate_intent(root, mutate)
        with self.assertRaisesRegex(ValueError, "20260729"):
            evaluator.load_v3_release_suite(root, self.candidate)

    def test_flipped_prefix_bit_is_refused(self):
        root = self._copy_suite()
        # Flip one byte inside the shorter observed stream but keep its
        # manifest record consistent, so only bit-exact prefix nesting can
        # catch it.
        raw_path = root / "n256" / "corpus.f32"
        data = bytearray(raw_path.read_bytes())
        data[100] ^= 0x01
        raw_path.write_bytes(bytes(data))
        record = _record(raw_path)
        for name in ("RELEASE_MANIFEST.json",):
            manifest_path = root / name
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for corpus in manifest["corpora"]:
                if corpus["capture_length"] == 256:
                    corpus["files"]["corpus.f32"] = record
            _write_json(manifest_path, manifest)
        corpus_manifest_path = root / "n256" / "corpus.json"
        corpus_manifest = json.loads(
            corpus_manifest_path.read_text(encoding="utf-8")
        )
        corpus_manifest["prefixDerivation"]["outputFiles"]["corpus.f32"] = (
            record
        )
        _write_json(corpus_manifest_path, corpus_manifest)
        manifest_path = root / "RELEASE_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for corpus in manifest["corpora"]:
            if corpus["capture_length"] == 256:
                corpus["files"]["corpus.json"] = _record(corpus_manifest_path)
        _write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "bit-exact prefix"):
            evaluator.load_v3_release_suite(root, self.candidate)

    def test_corpus_byte_tamper_without_manifest_update_is_refused(self):
        root = self._copy_suite()
        raw_path = root / "n512" / "corpus.f32"
        data = bytearray(raw_path.read_bytes())
        data[16] ^= 0xFF
        raw_path.write_bytes(bytes(data))
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            evaluator.load_v3_release_suite(root, self.candidate)


class CandidateTamperTests(unittest.TestCase):
    """A candidate component that is not the validated one must be refused."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._stack = contextlib.ExitStack()
        cls.addClassCleanup(cls._stack.close)
        cls.tmp = Path(
            cls._stack.enter_context(tempfile.TemporaryDirectory())
        )
        for patcher in _mini_protocol_patches():
            cls._stack.enter_context(patcher)
        cls.fixture = MiniFixture(cls.tmp)

    def _copy_candidate(self) -> Path:
        target = Path(tempfile.mkdtemp(dir=self.tmp)) / "candidate"
        return self.fixture.rebind_candidate_copy(target)

    def _load(self, root: Path):
        return evaluator.load_candidate(
            candidate_manifest_path=root / "release_candidate_manifest.json",
            classifier_bundle_dir=root / "classifier_bundle",
            rejector_bundle_dir=root / "rejector_bundle",
            classifier_fusion_dir=root / "classifier_fusion",
            rejector_fusion_dir=root / "rejector_fusion",
            staged_dir=root / "staged",
            prefilter_dir=root / "prefilters",
            device=torch.device("cpu"),
        )

    @staticmethod
    def _mutate_binding(root: Path, mutate) -> None:
        binding_path = root / "time-domain-v3-dual-binding.json"
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        mutate(binding)
        _write_json(binding_path, binding)
        manifest_path = root / "release_candidate_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["dual_binding"]["sha256"] = _sha256(binding_path)
        _write_json(manifest_path, manifest)

    @staticmethod
    def _mutate_package(root: Path, mutate) -> None:
        package_path = root / "staging_package_manifest.json"
        package = json.loads(package_path.read_text(encoding="utf-8"))
        mutate(package)
        _write_json(package_path, package)
        manifest_path = root / "release_candidate_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["staging_package_manifest"]["sha256"] = _sha256(package_path)
        _write_json(manifest_path, manifest)

    @staticmethod
    def _mutate_browser_asset(root: Path, role: str, mutate) -> None:
        """Mutate one deployable asset and consistently rebind every hash."""
        names = {
            "classifier": evaluator.CLASSIFIER_BROWSER_ASSET,
            "rejector": evaluator.REJECTOR_BROWSER_ASSET,
            "openset_policy": evaluator.OPENSET_BROWSER_ASSET,
        }
        asset_name = names[role]
        asset_path = root / asset_name
        payload = json.loads(asset_path.read_text(encoding="utf-8"))
        mutate(payload)
        _write_json(asset_path, payload)

        binding_path = root / evaluator.DUAL_BINDING_ASSET
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        if role in {"classifier", "rejector"}:
            binding["roles"][role]["asset_sha256"] = _sha256(asset_path)
            if role == "rejector":
                binding["openset_policy"]["rejector_asset_sha256"] = (
                    _sha256(asset_path)
                )
        else:
            binding["openset_policy"]["asset_sha256"] = _sha256(asset_path)
        _write_json(binding_path, binding)

        package_path = root / "staging_package_manifest.json"
        package = json.loads(package_path.read_text(encoding="utf-8"))
        asset_record = package["assets"][asset_name]
        asset_record.update(_record(asset_path))
        if role in {"classifier", "rejector"}:
            package["roles"][role]["asset"] = dict(asset_record)
        else:
            package["openset_policy"].update(asset_record)
        binding_record = package["assets"][evaluator.DUAL_BINDING_ASSET]
        binding_record.update(_record(binding_path))
        package["dual_binding"] = dict(binding_record)
        _write_json(package_path, package)

        manifest_path = root / "release_candidate_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["browser_assets"][role]["sha256"] = _sha256(asset_path)
        manifest["dual_binding"]["sha256"] = _sha256(binding_path)
        manifest["staging_package_manifest"]["sha256"] = _sha256(
            package_path
        )
        _write_json(manifest_path, manifest)

    @staticmethod
    def _mutate_evidence(root: Path, mutate) -> None:
        evidence_path = root / "validation_evidence.json"
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        mutate(evidence)
        _write_json(evidence_path, evidence)
        manifest_path = root / "release_candidate_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["validation_evidence"]["sha256"] = _sha256(evidence_path)
        _write_json(manifest_path, manifest)

    @classmethod
    def _mutate_contract(cls, root: Path, mutate) -> None:
        contract_path = root / "frozen_candidate_contract.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        mutate(contract)
        _write_json(contract_path, contract)
        cls._mutate_evidence(
            root,
            lambda evidence: evidence["candidate_contract"].__setitem__(
                "sha256", _sha256(contract_path)
            ),
        )

    def test_pristine_candidate_loads_and_reports_stage_one_domain(self):
        candidate = self._load(self._copy_candidate())
        self.assertEqual(candidate.stage_one_lengths, MINI_STAGE_ONE_LENGTHS)
        self.assertEqual(candidate.classes, MINI_CLASSES)
        # The STAGED decision threshold is the composite policy's; the
        # stage-2 policy's own threshold is carried for the control arm.
        self.assertEqual(
            candidate.threshold, float(self.fixture.composite.threshold)
        )
        self.assertEqual(
            candidate.stage_two_threshold,
            float(self.fixture.policy.threshold),
        )
        self.assertEqual(
            candidate.composite.enrollment_survivor_rows,
            self.fixture.composite.enrollment_survivor_rows,
        )

    def test_failing_staged_validation_is_refused(self):
        root = self._copy_candidate()
        metrics_path = root / "staged" / "openset_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["status"] = "development_openset_fail"
        metrics["all_pass"] = False
        _write_json(metrics_path, metrics)
        with self.assertRaisesRegex(ValueError, "passing validation"):
            self._load(root)

    def test_design_role_staged_artifact_is_refused(self):
        root = self._copy_candidate()
        metrics_path = root / "staged" / "openset_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["role"] = "design"
        metrics["gates_are_evidence"] = False
        _write_json(metrics_path, metrics)
        with self.assertRaisesRegex(ValueError, "role"):
            self._load(root)

    def test_tampered_prefilter_bytes_are_refused(self):
        root = self._copy_candidate()
        target = root / "prefilters" / "N256" / "coefficients.npy"
        coefficients = np.load(target)
        np.save(target, coefficients * 2.0)
        with self.assertRaisesRegex(ValueError, "prefilter set bytes"):
            self._load(root)

    def test_tampered_policy_npz_is_refused(self):
        root = self._copy_candidate()
        staged_dir = root / "staged"
        payload = self.fixture.policy.to_payload()
        payload["threshold"] = np.asarray(0.123456, dtype=np.float64)
        np.savez_compressed(
            staged_dir / "v3_open_policy_stage_two.npz", **payload
        )
        with self.assertRaises(ValueError):
            self._load(root)

    def test_a_version_one_staged_artifact_is_refused(self):
        """A staged artifact validated under a different policy version must
        never be evaluated with composite semantics."""
        root = self._copy_candidate()
        metrics_path = root / "staged" / "openset_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["architecture"]["staged_policy_version"] = "v1-of-something"
        _write_json(metrics_path, metrics)
        with self.assertRaisesRegex(ValueError, "policy version mismatch"):
            self._load(root)

    def test_a_missing_architecture_block_is_refused(self):
        root = self._copy_candidate()
        metrics_path = root / "staged" / "openset_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        del metrics["architecture"]
        _write_json(metrics_path, metrics)
        with self.assertRaisesRegex(ValueError, "policy version mismatch"):
            self._load(root)

    def test_staged_q97_quantile_mutation_is_refused(self):
        root = self._copy_candidate()
        metrics_path = root / "staged" / "openset_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["composite"]["threshold_quantile"] = 0.99
        _write_json(metrics_path, metrics)
        with self.assertRaisesRegex(ValueError, "threshold_quantile"):
            self._load(root)

    def test_staged_validation_seed_mutation_is_refused(self):
        root = self._copy_candidate()
        metrics_path = root / "staged" / "openset_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["seeds"]["novelty_seeds"] = [20260953, 20260956]
        _write_json(metrics_path, metrics)
        with self.assertRaisesRegex(ValueError, "novelty_seeds"):
            self._load(root)

    def test_a_missing_composite_npz_is_refused(self):
        root = self._copy_candidate()
        (root / "staged" / staged.COMPOSITE_POLICY_FILENAME).unlink()
        with self.assertRaisesRegex(
            ValueError, "composite-policy .schema-4/q97. validation artifact"
        ):
            self._load(root)

    def test_a_tampered_composite_npz_is_refused(self):
        root = self._copy_candidate()
        path = root / "staged" / staged.COMPOSITE_POLICY_FILENAME
        with np.load(path) as payload:
            data = {name: payload[name] for name in payload.files}
        data["threshold"] = np.asarray(0.123456, dtype=np.float64)
        np.savez_compressed(path, **data)
        with self.assertRaisesRegex(ValueError, "frozen quantile"):
            self._load(root)

    def test_a_composite_fit_against_another_stage_two_is_refused(self):
        root = self._copy_candidate()
        path = root / "staged" / staged.COMPOSITE_POLICY_FILENAME
        with np.load(path) as payload:
            data = {name: payload[name] for name in payload.files}
        data["stage_two_threshold"] = np.asarray(0.5, dtype=np.float64)
        np.savez_compressed(path, **data)
        with self.assertRaisesRegex(ValueError, "different stage-2"):
            self._load(root)

    def test_bundle_asset_tamper_is_refused(self):
        root = self._copy_candidate()
        target = root / "rejector_bundle" / "fusion_prototypes.npy"
        prototypes = np.load(target)
        np.save(target, prototypes + 0.5)
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self._load(root)

    def test_bundle_bound_to_a_different_fusion_is_refused(self):
        root = self._copy_candidate()
        manifest_path = root / "rejector_bundle" / "bundle_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["provenance"]["source_dev_metrics_sha256"] = "1" * 64
        _write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "source_dev_metrics_sha256"):
            self._load(root)

    def test_classifier_frontend_source_binding_drift_is_refused(self):
        root = self._copy_candidate()
        manifest_path = root / "classifier_bundle" / "bundle_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_sha256 = manifest["frontend"]["source_sha256"]
        first = sorted(source_sha256)[0]
        source_sha256[first] = "1" * 64
        _write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "frontend source bindings differ"):
            self._load(root)

    def test_classifier_fusion_tamper_is_refused(self):
        root = self._copy_candidate()
        target = root / "classifier_fusion" / "fusion_prototypes.npy"
        np.save(target, np.load(target) + np.float32(0.01))
        with self.assertRaises((ValueError, RuntimeError)):
            self._load(root)

    def test_swapped_fusion_roles_are_refused(self):
        root = self._copy_candidate()
        with self.assertRaises(ValueError):
            evaluator.load_candidate(
                candidate_manifest_path=root / "release_candidate_manifest.json",
                classifier_bundle_dir=root / "classifier_bundle",
                rejector_bundle_dir=root / "rejector_bundle",
                classifier_fusion_dir=root / "rejector_fusion",
                rejector_fusion_dir=root / "classifier_fusion",
                staged_dir=root / "staged",
                prefilter_dir=root / "prefilters",
                device=torch.device("cpu"),
            )

    def test_old_single_fusion_candidate_schema_is_refused(self):
        root = self._copy_candidate()
        path = root / "release_candidate_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["schema"] = "time-domain-v3-release-candidate-v0"
        _write_json(path, manifest)
        with self.assertRaisesRegex(ValueError, "old or unknown schema"):
            self._load(root)

    def test_q97_contract_policy_mutation_is_refused_after_rebinding(self):
        root = self._copy_candidate()
        self._mutate_contract(
            root,
            lambda contract: contract["q97_policy"].__setitem__(
                "threshold_quantile", 0.99
            ),
        )
        with self.assertRaisesRegex(ValueError, "policy identity/hygiene"):
            self._load(root)

    def test_q97_contract_transition_order_mutation_is_refused(self):
        root = self._copy_candidate()
        self._mutate_contract(
            root,
            lambda contract: contract["source_transition_intent"][
                "transitions"
            ][1].__setitem__("order", 1),
        )
        with self.assertRaisesRegex(ValueError, "transition order/seed"):
            self._load(root)

    def test_validation_transition_endpoint_mutation_is_refused(self):
        root = self._copy_candidate()
        self._mutate_evidence(
            root,
            lambda evidence: evidence["source_transition_evidence"][
                "transitions"
            ][0].__setitem__("to_sha256", "f" * 64),
        )
        with self.assertRaisesRegex(ValueError, "endpoints/order"):
            self._load(root)

    def test_validation_evidence_byte_tamper_is_refused(self):
        root = self._copy_candidate()
        path = root / "validation_evidence.json"
        evidence = json.loads(path.read_text(encoding="utf-8"))
        evidence["validation"]["all_pass"] = False
        _write_json(path, evidence)
        with self.assertRaisesRegex(ValueError, "validation_evidence SHA-256"):
            self._load(root)

    def test_dual_binding_byte_tamper_is_refused(self):
        root = self._copy_candidate()
        path = root / "time-domain-v3-dual-binding.json"
        binding = json.loads(path.read_text(encoding="utf-8"))
        binding["tampered"] = True
        _write_json(path, binding)
        with self.assertRaisesRegex(ValueError, "dual_binding SHA-256"):
            self._load(root)

    def test_binding_classifier_asset_cross_link_tamper_is_refused(self):
        root = self._copy_candidate()
        self._mutate_binding(
            root,
            lambda binding: binding["roles"]["classifier"].__setitem__(
                "asset_sha256", "1" * 64
            ),
        )
        with self.assertRaisesRegex(ValueError, "classifier role differs"):
            self._load(root)

    def test_binding_openset_cross_link_tamper_is_refused(self):
        root = self._copy_candidate()
        self._mutate_binding(
            root,
            lambda binding: binding["openset_policy"].__setitem__(
                "rejector_asset_sha256", "2" * 64
            ),
        )
        with self.assertRaisesRegex(ValueError, "openset policy differs"):
            self._load(root)

    def test_binding_candidate_id_tamper_is_refused(self):
        root = self._copy_candidate()
        self._mutate_binding(
            root,
            lambda binding: binding.__setitem__("candidate_id", "wrong"),
        )
        with self.assertRaisesRegex(ValueError, "identity/execution order"):
            self._load(root)

    def test_binding_validation_design_seed_tamper_is_refused(self):
        root = self._copy_candidate()
        self._mutate_binding(
            root,
            lambda binding: binding["validation"].__setitem__(
                "design_novelty_seed", 20260956
            ),
        )
        with self.assertRaisesRegex(ValueError, "validation record differs"):
            self._load(root)

    def test_package_candidate_id_cross_link_tamper_is_refused(self):
        root = self._copy_candidate()
        self._mutate_package(
            root,
            lambda package: package.__setitem__("candidate_id", "wrong"),
        )
        with self.assertRaisesRegex(ValueError, "candidate_id differs"):
            self._load(root)

    def test_package_classifier_asset_hash_tamper_is_refused(self):
        root = self._copy_candidate()
        self._mutate_package(
            root,
            lambda package: package["assets"][
                evaluator.CLASSIFIER_BROWSER_ASSET
            ].__setitem__("sha256", "3" * 64),
        )
        with self.assertRaisesRegex(ValueError, "SHA-256 differs from disk"):
            self._load(root)

    def test_package_classifier_role_cross_link_tamper_is_refused(self):
        root = self._copy_candidate()
        self._mutate_package(
            root,
            lambda package: package["roles"]["classifier"].__setitem__(
                "source_bundle_manifest_sha256", "4" * 64
            ),
        )
        with self.assertRaisesRegex(ValueError, "classifier role link differs"):
            self._load(root)

    def test_package_openset_evidence_cross_link_tamper_is_refused(self):
        root = self._copy_candidate()
        self._mutate_package(
            root,
            lambda package: package["openset_policy"].__setitem__(
                "staged_validation_report_sha256", "5" * 64
            ),
        )
        with self.assertRaisesRegex(ValueError, "openset evidence link differs"):
            self._load(root)

    def test_package_binding_record_cross_link_tamper_is_refused(self):
        root = self._copy_candidate()
        self._mutate_package(
            root,
            lambda package: package["dual_binding"].__setitem__(
                "status", "release"
            ),
        )
        with self.assertRaisesRegex(ValueError, "binding record differs"):
            self._load(root)

    def test_package_legacy_top_level_alias_is_refused(self):
        root = self._copy_candidate()
        self._mutate_package(
            root,
            lambda package: package.__setitem__(
                "single_fusion_weights", "legacy"
            ),
        )
        with self.assertRaisesRegex(ValueError, "extra or legacy fields"):
            self._load(root)

    def test_package_asset_legacy_alias_is_refused(self):
        root = self._copy_candidate()
        self._mutate_package(
            root,
            lambda package: package["assets"][
                evaluator.REJECTOR_BROWSER_ASSET
            ].__setitem__("legacy_role", "classifier"),
        )
        with self.assertRaisesRegex(ValueError, "contains aliases"):
            self._load(root)

    def test_package_role_legacy_alias_is_refused(self):
        root = self._copy_candidate()
        self._mutate_package(
            root,
            lambda package: package["roles"]["rejector"].__setitem__(
                "legacy_classifier", False
            ),
        )
        with self.assertRaisesRegex(ValueError, "rejector role link differs"):
            self._load(root)

    def test_package_openset_legacy_alias_is_refused(self):
        root = self._copy_candidate()
        self._mutate_package(
            root,
            lambda package: package["openset_policy"].__setitem__(
                "single_fusion_threshold", 0.5
            ),
        )
        with self.assertRaisesRegex(ValueError, "openset_policy is invalid"):
            self._load(root)

    def test_package_external_evidence_alias_is_refused(self):
        root = self._copy_candidate()
        self._mutate_package(
            root,
            lambda package: package["external_evidence"][
                "classifier_probe"
            ].__setitem__("packaged", False),
        )
        with self.assertRaisesRegex(ValueError, "contains aliases"):
            self._load(root)

    def test_package_external_evidence_symlink_is_refused(self):
        root = self._copy_candidate()
        probe = (
            root
            / "rejector_export"
            / "time-domain-v3-rejector-probe.json"
        )
        target = root / "saved-rejector-probe.json"
        target.write_bytes(probe.read_bytes())
        probe.unlink()
        probe.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "regular non-symlink file"):
            self._load(root)

    def test_package_deployable_asset_symlink_is_refused(self):
        root = self._copy_candidate()
        asset = root / evaluator.CLASSIFIER_BROWSER_ASSET
        saved_dir = root / "saved"
        saved_dir.mkdir()
        target = saved_dir / evaluator.CLASSIFIER_BROWSER_ASSET
        target.write_bytes(asset.read_bytes())
        asset.unlink()
        asset.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "regular non-symlink file"):
            self._load(root)

    def test_legacy_single_fusion_package_schema_is_refused(self):
        root = self._copy_candidate()
        package_path = root / "staging_package_manifest.json"
        package = json.loads(package_path.read_text(encoding="utf-8"))
        package["schema"] = "atomos.v3.time-domain-classifier.runtime-package"
        _write_json(package_path, package)
        manifest_path = root / "release_candidate_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["staging_package_manifest"].update(
            {
                "sha256": _sha256(package_path),
                "schema": package["schema"],
            }
        )
        _write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "dual-runtime staging"):
            self._load(root)

    def test_browser_role_asset_tamper_is_refused(self):
        root = self._copy_candidate()
        path = root / evaluator.CLASSIFIER_BROWSER_ASSET
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["role"] = "rejector"
        _write_json(path, payload)
        with self.assertRaisesRegex(ValueError, "browser_assets.classifier SHA"):
            self._load(root)

    def test_browser_runtime_role_schema_is_enforced_after_rebinding(self):
        root = self._copy_candidate()
        path = root / evaluator.CLASSIFIER_BROWSER_ASSET
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["runtime_role"] = "known_unknown_rejector"
        _write_json(path, payload)
        manifest_path = root / "release_candidate_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["browser_assets"]["classifier"]["sha256"] = _sha256(path)
        _write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "runtime_role differs"):
            self._load(root)

    def test_browser_frontend_geometry_mutation_is_refused_after_rebinding(self):
        root = self._copy_candidate()
        self._mutate_browser_asset(
            root,
            "classifier",
            lambda payload: payload["frontend"].__setitem__(
                "patch_length", 32
            ),
        )
        with self.assertRaisesRegex(ValueError, "exact runtime-bundle export"):
            self._load(root)

    def test_browser_branch_geometry_mutation_is_refused_after_rebinding(self):
        root = self._copy_candidate()
        self._mutate_browser_asset(
            root,
            "rejector",
            lambda payload: payload["real"]["config"].__setitem__(
                "patch_length", 32
            ),
        )
        with self.assertRaisesRegex(ValueError, "exact runtime-bundle export"):
            self._load(root)

    def test_browser_lof_width_mutation_is_refused_after_rebinding(self):
        root = self._copy_candidate()

        def shrink_real_lof(payload):
            component = next(
                item
                for item in payload["stage_two"]["lof_components"]
                if item["branch"] == "real"
            )
            component["mean"] = component["mean"][:2]
            component["scale"] = component["scale"][:2]
            component["reference"] = [
                row[:2] for row in component["reference"]
            ]

        self._mutate_browser_asset(root, "openset_policy", shrink_real_lof)
        with self.assertRaisesRegex(
            ValueError, "exact frozen staged-policy export"
        ):
            self._load(root)

    def test_browser_openset_schema_version_is_enforced_after_rebinding(self):
        root = self._copy_candidate()
        path = root / evaluator.OPENSET_BROWSER_ASSET
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["schema_version"] = 1
        _write_json(path, payload)
        manifest_path = root / "release_candidate_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["browser_assets"]["openset_policy"]["sha256"] = _sha256(path)
        _write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "schema/version/status differs"):
            self._load(root)

    def test_browser_q97_quantile_mutation_is_refused(self):
        root = self._copy_candidate()
        path = root / evaluator.OPENSET_BROWSER_ASSET
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["composite"]["threshold_quantile"] = 0.99
        _write_json(path, payload)
        manifest_path = root / "release_candidate_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["browser_assets"]["openset_policy"]["sha256"] = _sha256(path)
        _write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "threshold_quantile differs"):
            self._load(root)

    def test_browser_q97_hygiene_mutation_is_refused(self):
        root = self._copy_candidate()
        path = root / evaluator.OPENSET_BROWSER_ASSET
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["composite"]["novelty_rows_used_for_threshold"] = 1
        _write_json(path, payload)
        manifest_path = root / "release_candidate_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["browser_assets"]["openset_policy"]["sha256"] = _sha256(path)
        _write_json(manifest_path, manifest)
        with self.assertRaisesRegex(
            ValueError, "novelty_rows_used_for_threshold differs"
        ):
            self._load(root)

    def test_browser_stage_two_kind_mutation_is_refused(self):
        root = self._copy_candidate()
        path = root / evaluator.OPENSET_BROWSER_ASSET
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["stage_two"]["kind"] = evaluator.EXPECTED_STAGED_POLICY_KIND
        _write_json(path, payload)
        manifest_path = root / "release_candidate_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["browser_assets"]["openset_policy"]["sha256"] = _sha256(path)
        _write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "stage-two policy kind differs"):
            self._load(root)

    def test_candidate_browser_record_alias_is_refused(self):
        root = self._copy_candidate()
        manifest_path = root / "release_candidate_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["browser_assets"]["classifier"]["legacy_role"] = "classifier"
        _write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "browser_assets.classifier"):
            self._load(root)

    def test_candidate_manifest_legacy_alias_is_refused(self):
        root = self._copy_candidate()
        manifest_path = root / "release_candidate_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["single_fusion"] = True
        _write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "extra or legacy fields"):
            self._load(root)

    def test_staged_report_bound_to_classifier_fusion_is_refused(self):
        root = self._copy_candidate()
        metrics_path = root / "staged" / "openset_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        classifier = openset_base.load_fusion_artifact(root / "classifier_fusion")
        metrics["fusion"]["directory_sha256"] = classifier.directory_sha256
        _write_json(metrics_path, metrics)
        with self.assertRaisesRegex(ValueError, "different fusion"):
            self._load(root)

    def test_staged_state_fit_against_another_fusion_is_refused(self):
        root = self._copy_candidate()
        metrics_path = root / "staged" / "openset_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["fusion"]["directory_sha256"] = "2" * 64
        _write_json(metrics_path, metrics)
        with self.assertRaisesRegex(ValueError, "different fusion"):
            self._load(root)

    def test_hard_source_drift_is_refused(self):
        root = self._copy_candidate()
        metrics_path = root / "staged" / "openset_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["source_sha256"]["v3_time_domain_openset.py"] = "3" * 64
        _write_json(metrics_path, metrics)
        with self.assertRaisesRegex(ValueError, "source drift"):
            self._load(root)


class OrderedLedgerTransitionTests(unittest.TestCase):
    """Only the ordered q97 design/validation bookkeeping chain is admissible."""

    def test_exact_full_transition_chain_is_ast_verified_and_reported(self):
        first = evaluator.ORDERED_STAGED_SOURCE_TRANSITIONS[0]
        final = evaluator.ORDERED_STAGED_SOURCE_TRANSITIONS[1]
        checked, transitions = evaluator._verify_source_contract(
            {
                str(first["source"]): str(first["from_sha256"])
            },
            (str(first["source"]),),
            origin=str(first["origin"]),
        )
        name = str(first["source"])
        self.assertEqual(checked[name], final["to_sha256"])
        self.assertEqual(
            transitions[name]["admission"],
            "exact_ordered_q97_seed_ledger_transition_chain",
        )
        self.assertEqual(
            transitions[name]["normalized_ast_sha256"],
            final["normalized_ast_sha256"],
        )
        self.assertFalse(
            transitions[name]["candidate_inference_behavior_changed"]
        )
        self.assertEqual(
            transitions[name]["ordered_transition_ids"],
            [
                "predesign_to_postdesign",
                "validation_to_postvalidation",
            ],
        )
        self.assertEqual(
            transitions[name]["consumed_novelty_seeds"],
            [
                evaluator.DESIGN_NOVELTY_SEED,
                *evaluator.VALIDATION_NOVELTY_SEEDS,
            ],
        )
        self.assertEqual(transitions[name]["next_clean_novelty_seed"], 20260956)

    def test_transition_is_not_accepted_for_another_origin(self):
        transition = evaluator.ORDERED_STAGED_SOURCE_TRANSITIONS[0]
        with self.assertRaisesRegex(ValueError, "source drift"):
            evaluator._verify_source_contract(
                {
                    str(transition["source"]): str(
                        transition["from_sha256"]
                    )
                },
                (str(transition["source"]),),
                origin="an untrusted source record",
            )

    def test_exact_validation_transition_suffix_is_git_verified(self):
        transition = evaluator.ORDERED_STAGED_SOURCE_TRANSITIONS[1]
        checked, transitions = evaluator._verify_source_contract(
            {
                str(transition["source"]): str(
                    transition["from_sha256"]
                )
            },
            (str(transition["source"]),),
            origin=str(transition["origin"]),
        )
        name = str(transition["source"])
        self.assertEqual(checked[name], transition["to_sha256"])
        self.assertEqual(
            transitions[name]["ordered_transition_ids"],
            ["validation_to_postvalidation"],
        )
        self.assertEqual(
            transitions[name]["consumed_novelty_seeds"],
            [
                evaluator.DESIGN_NOVELTY_SEED,
                *evaluator.VALIDATION_NOVELTY_SEEDS,
            ],
        )
        self.assertEqual(transitions[name]["next_clean_novelty_seed"], 20260956)

    def test_transition_order_mutation_is_refused_before_hash_admission(self):
        mutated = copy.deepcopy(evaluator.ORDERED_STAGED_SOURCE_TRANSITIONS)
        mutated[1]["order"] = 1
        with mock.patch.object(
            evaluator, "ORDERED_STAGED_SOURCE_TRANSITIONS", tuple(mutated)
        ):
            with self.assertRaisesRegex(ValueError, "order/identity"):
                evaluator._validated_ordered_source_transitions(
                    required_through_order=1
                )

    def test_transition_seed_mutation_is_refused(self):
        mutated = copy.deepcopy(evaluator.ORDERED_STAGED_SOURCE_TRANSITIONS)
        mutated[0]["consumed_novelty_seeds"] = (20260956,)
        with mock.patch.object(
            evaluator, "ORDERED_STAGED_SOURCE_TRANSITIONS", tuple(mutated)
        ):
            with self.assertRaisesRegex(ValueError, "another seed"):
                evaluator._validated_ordered_source_transitions(
                    required_through_order=1
                )

    def test_transition_commit_phase_alias_is_refused(self):
        transition = copy.deepcopy(
            evaluator.ORDERED_STAGED_SOURCE_TRANSITIONS[0]
        )
        transition["from_commit"] = transition["evidence_commit"]
        with self.assertRaisesRegex(ValueError, "aliases distinct phases"):
            evaluator._verify_git_transition_binding(transition)


class ProtocolAndHelperIdentityTests(unittest.TestCase):
    """The v3 evaluator may not restate anything the v2 evaluator owns."""

    def test_gate_objects_are_the_v2_objects(self):
        self.assertIs(evaluator.GATE_FLOORS, release.GATE_FLOORS)
        self.assertIs(evaluator._gate, release._gate)
        self.assertIs(evaluator._write_json_exclusive, release._write_json_exclusive)
        self.assertIs(evaluator.resolve_device, release.resolve_device)
        self.assertIs(evaluator.FIVE_SHOT_K, release.FIVE_SHOT_K)
        self.assertEqual(evaluator.HIGH_SNR_DB, release.HIGH_SNR_DB)

    def test_q97_browser_policy_and_parity_schemas_are_v4(self):
        self.assertEqual(evaluator.BROWSER_OPENSET_SCHEMA_VERSION, 4)
        self.assertEqual(evaluator.PARITY_SCHEMA_VERSION, 4)
        self.assertTrue(evaluator.ENFORCE_CANONICAL_Q97_EVIDENCE)

    def test_historical_owner_text_mutation_is_refused(self):
        with mock.patch.object(
            evaluator,
            "HISTORICAL_V3_GATE_REDECLARATION_RATIONALE",
            evaluator.HISTORICAL_V3_GATE_REDECLARATION_RATIONALE + " mutated",
        ):
            with self.assertRaisesRegex(ValueError, "original verbatim"):
                evaluator.historical_claim_correction_block()

    def test_imported_policy_identity_mutation_is_refused(self):
        with mock.patch.object(staged, "STAGED_POLICY_SCHEMA", 3):
            with self.assertRaisesRegex(ValueError, "identity/seed hygiene"):
                evaluator.expected_evaluation_protocol(
                    evaluator.EXPECTED_RELEASE_SEED, (4096,)
                )

    def test_expected_protocol_reuses_v2_rules_and_updates_v3_fields(self):
        protocol = evaluator.expected_evaluation_protocol(
            20260736, (4096, 8192, 16384)
        )
        v2 = release.EXPECTED_EVALUATION_PROTOCOL
        self.assertEqual(protocol["version"], evaluator.EVALUATION_VERSION)
        self.assertEqual(protocol["gates"], dict(release.GATE_FLOORS))
        historical = protocol["historical_gate_redeclaration"]
        self.assertEqual(
            historical, evaluator.historical_gate_redeclaration_block()
        )
        self.assertFalse(historical["active_for_current_protocol"])
        self.assertEqual(
            set(historical["gates_redeclared"]),
            {
                "open_known_false_unknown_worst_length",
                "five_shot_worst_length_balanced",
            },
        )
        self.assertEqual(protocol["five_shot"], v2["five_shot"])
        self.assertEqual(protocol["high_snr_db"], v2["high_snr_db"])
        self.assertEqual(
            protocol["length_observation_rule"], v2["length_observation_rule"]
        )
        self.assertEqual(protocol["novelty"]["seed"], 20260736)
        self.assertEqual(
            protocol["novelty"]["seed_derivation"],
            v2["novelty"]["seed_derivation"],
        )
        self.assertEqual(
            protocol["open_set"]["stage_one_capture_lengths"],
            [4096, 8192, 16384],
        )
        self.assertEqual(
            protocol["open_set"]["stage_one_max_fitted_length"], 16384
        )
        self.assertEqual(
            protocol["open_set"]["stage_one_prefix_gated_lengths"], [32768]
        )
        self.assertEqual(
            protocol["open_set"]["stage_one_uncovered_lengths"], []
        )
        self.assertEqual(
            protocol["open_set"]["stage_one_prefix_rule"],
            {
                "max_fitted_length": 16384,
                "rule": evaluator.STAGE_ONE_PREFIX_RULE,
            },
        )
        self.assertIs(
            evaluator.STAGE_ONE_PREFIX_RULE, staged.STAGE_ONE_PREFIX_RULE
        )
        self.assertFalse(protocol["open_set"]["additive_only"])
        self.assertTrue(protocol["open_set"]["changes_closed_label"])
        self.assertTrue(
            protocol["open_set"]["open_set_decision_changes_closed_label"]
        )
        # The composite (staged policy version 4) is predeclared: version,
        # survivor score, threshold rule and the staged score axis text.
        self.assertEqual(
            protocol["open_set"]["staged_policy_schema"],
            evaluator.EXPECTED_STAGED_POLICY_SCHEMA,
        )
        self.assertEqual(
            protocol["open_set"]["staged_policy_version"],
            evaluator.EXPECTED_STAGED_POLICY_VERSION,
        )
        self.assertEqual(
            protocol["open_set"]["survivor_score"],
            staged.COMPOSITE_SURVIVOR_SCORE,
        )
        self.assertIn(
            "enrollment stage-1 survivors",
            protocol["open_set"]["unknown_threshold_rule"],
        )
        self.assertEqual(
            protocol["open_set"]["staged_policy_kind"],
            evaluator.EXPECTED_STAGED_POLICY_KIND,
        )
        self.assertEqual(
            protocol["open_set"]["composite_threshold_quantile"], 0.97
        )
        self.assertEqual(
            protocol["open_set"]["policy_hygiene"],
            evaluator.EXPECTED_POLICY_HYGIENE,
        )
        self.assertEqual(
            protocol["open_set"]["design_novelty_seed"], 20260955
        )
        self.assertEqual(
            protocol["open_set"]["validation_novelty_seeds"],
            [20260953, 20260954],
        )
        self.assertIn("q0.97", protocol["open_set"]["unknown_threshold_rule"])
        self.assertIn("COMPOSITE", protocol["open_set"]["score_axis"])
        self.assertIn(
            "composite survivor policy", protocol["novelty"]["calibration"]
        )
        # The protocol must be a plain JSON object.
        json.dumps(protocol, allow_nan=False)

    def test_expected_protocol_refuses_consumed_and_development_seeds(self):
        for seed in (
            20260729,
            20260730,
            20260731,
            20260732,
            20260733,
            20260734,
            20260735,
            20260942,
            20261001,
        ):
            with self.assertRaises(ValueError):
                evaluator.expected_evaluation_protocol(seed, (4096,))
        evaluator.expected_evaluation_protocol(20260736, (4096,))
        with self.assertRaises(ValueError):
            evaluator.expected_evaluation_protocol(20260737, (4096,))

    def test_novelty_seed_derivation_matches_v2_formula(self):
        for family in release.NOVELTY_FAMILIES:
            self.assertEqual(
                evaluator._novelty_seed(release.NOVELTY_SEED, family),
                release._novelty_seed(family),
            )
        with self.assertRaises(ValueError):
            evaluator._novelty_seed(1, "not-a-family")

    def test_validate_release_seed(self):
        self.assertEqual(evaluator.validate_release_seed(20260736), 20260736)
        for seed in (
            20260729,
            20260730,
            20260731,
            20260732,
            20260733,
            20260734,
            20260735,
            20260900,
            20260999,
            20261000,
            20261999,
        ):
            with self.assertRaises(ValueError):
                evaluator.validate_release_seed(seed)
        with self.assertRaisesRegex(ValueError, "frozen for untouched"):
            evaluator.validate_release_seed(20260737)

    def test_the_consumed_v3_seed_refusal_names_the_frozen_failure(self):
        with self.assertRaisesRegex(ValueError, "HANDOFF 25"):
            evaluator.validate_release_seed(20260731)
        with self.assertRaisesRegex(ValueError, "HANDOFF 27"):
            evaluator.validate_release_seed(20260733)
        with self.assertRaisesRegex(ValueError, "historical"):
            evaluator.validate_release_seed(20260734)
        with self.assertRaisesRegex(ValueError, "149/1309"):
            evaluator.validate_release_seed(20260735)

    def test_default_candidate_directories_are_the_frozen_candidate(self):
        parser = evaluator.build_parser()
        args = parser.parse_args([])
        self.assertEqual(
            Path(args.candidate_manifest).name,
            "v3_4_q97_dual_release_candidate.json",
        )
        self.assertEqual(
            evaluator.DEFAULT_CANDIDATE_CONTRACT.name,
            "v3_q97_candidate_contract.json",
        )
        self.assertEqual(
            evaluator.CANDIDATE_CONTRACT_SCHEMA,
            "time-domain-v3-q97-candidate-v1",
        )
        self.assertEqual(
            evaluator.CANDIDATE_ID,
            "v3.4-q97-decoupled-8k-classifier-4k-rejector",
        )
        self.assertEqual(
            Path(args.classifier_bundle_dir).name,
            "v3_runtime_bundle_classifier8kreg_dual_seed20260730",
        )
        self.assertEqual(
            Path(args.rejector_bundle_dir).name,
            "v3_runtime_bundle_rejector4k_dual_seed20260730",
        )
        self.assertEqual(
            Path(args.classifier_fusion_dir).name,
            "v3_fusion_ml8000reg_seed20260730",
        )
        self.assertEqual(
            Path(args.rejector_fusion_dir).name,
            "v3_fusion_multilength_seed20260730",
        )
        self.assertEqual(
            Path(args.staged_dir).name,
            "staged_validate_v34_q97_decoupled_rejector4k_classifier8k_"
            "budget001_seeds20260953_20260954",
        )
        self.assertEqual(
            Path(args.prefilter_dir).parent.name,
            "noise_prefilter_fit20261001_budget001",
        )
        self.assertEqual(args.device, "cpu")
        self.assertIsNone(args.release_root)

    def test_candidate_json_reader_accepts_large_mtime_integers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "meta.json"
            _write_json(path, {"mtime_ns": 2**60, "ok": True})
            value = evaluator._read_candidate_json(path)
            self.assertEqual(value["mtime_ns"], 2**60)
            with self.assertRaises(ValueError):
                release._read_json(path)


class V4AdmissionHardeningTests(unittest.TestCase):
    def _dependency_fixture(self, root: Path):
        paths = {}
        hashes = {}
        for index, key in enumerate(evaluator.TRANSITIVE_DEPENDENCY_PATHS):
            path = root / f"dependency-{index}.py"
            path.write_text(f"# {key}\n", encoding="utf-8")
            paths[key] = path
            hashes[key] = _sha256(path)
        return paths, hashes

    def test_transitive_dependency_contract_is_the_exact_reviewed_49(self):
        self.assertEqual(len(evaluator.TRANSITIVE_DEPENDENCY_PATHS), 49)
        self.assertIn(
            "training/zplane_ab/v2_full_variation/v3_scale/"
            "measure_v3_remaining_gates.py",
            evaluator.TRANSITIVE_DEPENDENCY_PATHS,
        )
        self.assertEqual(
            set(evaluator.EXPECTED_TRANSITIVE_DEPENDENCY_SHA256),
            set(evaluator.TRANSITIVE_DEPENDENCY_PATHS),
        )

    def test_exact_dependency_and_runtime_contract_admits(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, hashes = self._dependency_fixture(Path(tmp))
            report = evaluator._admit_transitive_execution_contract(
                canonical_release_candidate=True,
                device=torch.device("cpu"),
                dependency_paths=paths,
                expected_sha256=hashes,
            )
        self.assertTrue(report["passes"])
        self.assertEqual(report["dependency_file_count"], 49)
        self.assertEqual(
            report["runtime_identity"],
            evaluator.EXPECTED_EVALUATOR_RUNTIME_IDENTITY,
        )

    def test_dependency_key_hash_symlink_and_runtime_mismatches_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, hashes = self._dependency_fixture(root)
            missing = dict(hashes)
            missing.pop(next(iter(missing)))
            with self.assertRaisesRegex(ValueError, "hash keys"):
                evaluator._admit_transitive_execution_contract(
                    canonical_release_candidate=True,
                    device=torch.device("cpu"),
                    dependency_paths=paths,
                    expected_sha256=missing,
                )
            wrong = dict(hashes)
            wrong[next(iter(wrong))] = "0" * 64
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                evaluator._admit_transitive_execution_contract(
                    canonical_release_candidate=True,
                    device=torch.device("cpu"),
                    dependency_paths=paths,
                    expected_sha256=wrong,
                )
            key = next(iter(paths))
            target = paths[key]
            link = root / "dependency-link.py"
            link.symlink_to(target)
            linked = dict(paths)
            linked[key] = link
            with self.assertRaisesRegex(ValueError, "non-symlink"):
                evaluator._admit_transitive_execution_contract(
                    canonical_release_candidate=True,
                    device=torch.device("cpu"),
                    dependency_paths=linked,
                    expected_sha256=hashes,
                )
            runtime = dict(evaluator.EXPECTED_EVALUATOR_RUNTIME_IDENTITY)
            runtime["python"] = "3.9.7"
            with self.assertRaisesRegex(ValueError, "runtime_identity"):
                evaluator._admit_transitive_execution_contract(
                    canonical_release_candidate=True,
                    device=torch.device("cpu"),
                    dependency_paths=paths,
                    expected_sha256=hashes,
                    expected_runtime=runtime,
                )

    def test_bound_canonical_dependency_hashes_admit_and_none_fails_closed(self):
        report = evaluator._admit_transitive_execution_contract(
            canonical_release_candidate=True,
            device=torch.device("cpu"),
        )
        self.assertTrue(report["passes"])
        self.assertEqual(report["dependency_file_count"], 49)
        self.assertEqual(
            report["dependency_source_sha256"],
            evaluator.EXPECTED_TRANSITIVE_DEPENDENCY_SHA256,
        )

        unbound = dict(evaluator.EXPECTED_TRANSITIVE_DEPENDENCY_SHA256)
        unbound[next(iter(unbound))] = None
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            evaluator._admit_transitive_execution_contract(
                canonical_release_candidate=True,
                device=torch.device("cpu"),
                expected_sha256=unbound,
            )

    def test_json_scalar_admission_rejects_bool_and_integer_aliases_broadly(self):
        boolean_fields = (
            "all_pass",
            "development_only",
            "gates_before_classification",
            "passes",
            "release_seed_20260736_used",
        )
        for field in boolean_fields:
            for alias in (0, 1, 0.0, 1.0, "false", None):
                with self.subTest(field=field, alias=alias):
                    with self.assertRaisesRegex(ValueError, "JSON boolean"):
                        evaluator._validate_json_scalar_types({field: alias})
        integer_fields = (
            "schema_version",
            "release_seed",
            "row_count",
            "capture_length",
        )
        for field in integer_fields:
            for alias in (False, True, 4.0, "4", None):
                with self.subTest(field=field, alias=alias):
                    with self.assertRaisesRegex(ValueError, "JSON integer"):
                        evaluator._validate_json_scalar_types({field: alias})
        for field in (
            "novelty_seeds",
            "novelty_seeds_consumed_once",
            "capture_lengths",
            "row_counts",
        ):
            for alias in (False, True, 4.0, "4", None):
                with self.subTest(field=field, alias=alias):
                    with self.assertRaisesRegex(ValueError, "JSON integer"):
                        evaluator._validate_json_scalar_types(
                            {field: [alias]}
                        )
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaisesRegex(ValueError, "finite JSON number"):
                evaluator._validate_json_scalar_types({"threshold": value})
        for alias in (False, True, 0, "0.0", None, float("nan")):
            with self.subTest(float_alias=alias):
                with self.assertRaisesRegex(ValueError, "finite JSON number"):
                    evaluator._require_finite_json_float(
                        alias, field="threshold"
                    )
        for alias in (False, True, 4.0, "4", None):
            with self.subTest(schema_alias=alias):
                with self.assertRaises(ValueError):
                    evaluator._require_exact_json_value(
                        alias, 4, field="schema"
                    )

    def test_protocol_boundary_admits_only_javascript_integral_float_collapse(
        self,
    ):
        with self.assertRaisesRegex(ValueError, "int.*float"):
            evaluator._require_exact_json_value(
                1,
                1.0,
                field="protocol.physical_scale_factors[2]",
            )
        evaluator._require_exact_json_value(
            {"physical_scale_factors": [0.5, 0.75, 1, 1.5, 2]},
            {"physical_scale_factors": [0.5, 0.75, 1.0, 1.5, 2.0]},
            field="protocol",
            allow_integral_float_collapse=True,
        )
        for found, expected in (
            (True, 1.0),
            (1.0, 1),
            (1, 1.5),
            (2, 1.0),
        ):
            with self.subTest(found=found, expected=expected):
                with self.assertRaises(ValueError):
                    evaluator._require_exact_json_value(
                        found,
                        expected,
                        field="protocol.value",
                        allow_integral_float_collapse=True,
                    )

    def test_dynamic_integer_maps_and_real_integer_names_reject_aliases(self):
        for collection in ("counts", "row_counts"):
            for alias in (False, True, 4.0, "4", None):
                with self.subTest(collection=collection, alias=alias):
                    with self.assertRaisesRegex(ValueError, "JSON integer"):
                        evaluator._validate_json_scalar_types(
                            {collection: {"known": alias}}
                        )
        integer_fields = (
            "targetPerClass",
            "minimum_target_per_class",
            "staged_policy_schema",
            "release_seed_not_spent",
            "clean_validation_seeds_from",
            "known_classes",
            "bytesPerComplexSample",
            "sealed_release_data_used",
            "consumed_test_rows_exposed",
            "downstream_rows_evaluated",
            "stage_two_rows_scored_staged",
        )
        for field in integer_fields:
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "JSON integer"):
                    evaluator._validate_json_scalar_types({field: 4.0})
        for payload in (
            {"seeds": {"known": 4.0}},
            {"seeds": {"known": 4.0, "note": "mixed"}},
            {"seeds": {"known": True, "ledger": "explanatory"}},
        ):
            with self.subTest(payload=payload):
                with self.assertRaisesRegex(ValueError, "JSON integer"):
                    evaluator._validate_json_scalar_types(payload)
        evaluator._validate_json_scalar_types(
            {
                "accepted_role_winner_disagreement_rows": [
                    "N4096/known/0001"
                ]
            }
        )
        for alias in (0, 1.0, False, None):
            with self.subTest(disagreement_row_alias=alias):
                with self.assertRaisesRegex(ValueError, "string row identity"):
                    evaluator._validate_json_scalar_types(
                        {
                            "accepted_role_winner_disagreement_rows": [
                                alias
                            ]
                        }
                    )
        for field in ("knownWinnerIndex", "rejector_closed_winner_index"):
            evaluator._validate_json_scalar_types({field: None})
            evaluator._validate_json_scalar_types({field: 3})
            for alias in (False, 3.0, "3"):
                with self.subTest(nullable_integer=field, alias=alias):
                    with self.assertRaisesRegex(ValueError, "JSON integer"):
                        evaluator._validate_json_scalar_types({field: alias})

    def test_strict_scalar_reader_accepts_real_frozen_json_shapes(self):
        paths = (
            evaluator.V2
            / "artifacts/invariant_patch/v3_scale/"
            "staged_design_v34_q97_rejector4k_budget001_seed20260955/"
            "openset_metrics.json",
            evaluator.V2
            / "artifacts/invariant_patch/v3_scale/"
            "noise_prefilter_fit20261001_budget001/noise_prefilter_fit.json",
            evaluator.DEFAULT_VALIDATION_EVIDENCE,
            evaluator.TRAINING
            / "artifacts/releases/invariant_fusion_v3_sealed_seed20260735/"
            "RELEASE_INTENT.json",
            evaluator.TRAINING
            / "artifacts/releases/invariant_fusion_v3_sealed_seed20260735/"
            "RELEASE_MANIFEST.json",
            evaluator.TRAINING
            / "artifacts/releases/invariant_fusion_v3_sealed_seed20260735/"
            "RELEASE_EVALUATION.json",
            evaluator.REPO
            / "training/zplane_ab/v2_full_variation/artifacts/staging/"
            "time_domain_v3_openset_dual/"
            "time-domain-openset-parity-v1.json",
        )
        for path in paths:
            with self.subTest(path=path):
                self.assertIsInstance(
                    evaluator._read_candidate_json(path),
                    dict,
                )

    def test_canonical_validation_evidence_binds_the_frozen_chain(self):
        evidence = evaluator._read_candidate_json(
            evaluator.DEFAULT_VALIDATION_EVIDENCE
        )
        self.assertEqual(
            evidence["source_transition_evidence"],
            evaluator.ordered_source_transition_contract_block(),
        )
        evaluator._validated_ordered_source_transitions()
        contract = Path(evidence["candidate_contract"]["path"]).resolve()
        self.assertEqual(contract, evaluator.DEFAULT_CANDIDATE_CONTRACT.resolve())
        self.assertEqual(
            evidence["candidate_contract"]["sha256"],
            _sha256(contract),
        )
        self.assertEqual(
            evidence["candidate_contract"]["commit"],
            evaluator.ORDERED_STAGED_SOURCE_TRANSITIONS[1]["from_commit"],
        )
        committed_contract = evaluator._git_blob(
            evidence["candidate_contract"]["commit"],
            evaluator.DEFAULT_CANDIDATE_CONTRACT.relative_to(
                evaluator.REPO
            ).as_posix(),
        )
        self.assertEqual(
            hashlib.sha256(committed_contract).hexdigest(),
            evidence["candidate_contract"]["sha256"],
        )
        report = Path(evidence["validation"]["report"]).resolve()
        self.assertEqual(
            report,
            evaluator.DEFAULT_STAGED_DIR / "openset_metrics.json",
        )
        self.assertEqual(
            evidence["validation"]["report_sha256"],
            _sha256(report),
        )
        report_payload = evaluator._read_candidate_json(report)
        for name, digest in report_payload["artifacts"].items():
            self.assertEqual(
                evidence["validation_locked_policy_artifacts"][name],
                digest,
            )

    def test_each_transition_recomputes_every_configured_endpoint(self):
        digest_fields = (
            "from_sha256",
            "to_sha256",
            "normalized_ast_sha256",
            "full_index_diff_sha256",
            "evidence_report_sha256",
        )
        commit_fields = (
            "from_commit",
            "evidence_commit",
            "to_commit",
        )
        for transition_index in range(2):
            for field in digest_fields + commit_fields:
                mutated = copy.deepcopy(
                    evaluator.ORDERED_STAGED_SOURCE_TRANSITIONS
                )
                mutated[transition_index][field] = (
                    "0" * 64 if field in digest_fields else "0" * 40
                )
                with self.subTest(
                    transition=transition_index + 1,
                    field=field,
                ), mock.patch.object(
                    evaluator,
                    "ORDERED_STAGED_SOURCE_TRANSITIONS",
                    tuple(mutated),
                ):
                    with self.assertRaises(ValueError):
                        evaluator._validated_ordered_source_transitions(
                            required_through_order=transition_index + 1
                        )

    def test_deployment_classifier_spy_sees_only_accepted_rows(self):
        candidate = mock.Mock()
        candidate.classifier_fusion_artifact.prototypes = np.asarray(
            [[0.0], [10.0]], dtype=np.float32
        )
        packed = np.arange(10, dtype=np.float32).reshape(5, 2)
        features = np.arange(15, dtype=np.float32).reshape(5, 3)
        rejected = np.asarray([True, False, True, False, False])
        control = np.asarray([1, 0, 1, 1, 0], dtype=np.int64)
        seen = []

        def spy(_candidate, accepted_packed, accepted_features, _device):
            seen.append(
                (accepted_packed.copy(), accepted_features.copy())
            )
            return accepted_features, {
                "fusion": np.asarray([[0.0], [10.0], [0.0]], dtype=np.float32)
            }

        with mock.patch.object(evaluator, "_embed_classifier", side_effect=spy):
            labels, report = evaluator._deployment_classifier_labels(
                candidate,
                packed,
                features,
                rejected,
                control,
                torch.device("cpu"),
            )
        self.assertEqual(len(seen), 1)
        np.testing.assert_array_equal(seen[0][0], packed[~rejected])
        np.testing.assert_array_equal(seen[0][1], features[~rejected])
        np.testing.assert_array_equal(labels, [-1, 0, -1, 1, 0])
        self.assertEqual(report["classifier_batch_rows"], 3)
        self.assertEqual(report["rejected_rows_never_submitted"], 2)

    def test_attribution_counts_rates_partition_and_overlap_are_exact(self):
        gated = np.asarray([True, False, False, True])
        staged_score = np.asarray([1.1, 1.1, 0.1, 0.2])
        unstaged_score = np.asarray([0.9, 0.1, 0.8, 0.7])
        report = {
            "rows": 4,
            "staged_threshold": 0.5,
            "unstaged_threshold": 0.5,
            "staged_false_unknown_rate": 0.75,
            "stage_one_gate_rate": 0.5,
            "stage_two_false_unknown_rate_marginal": 0.25,
            "stage_two_false_unknown_rate_among_survivors": 0.5,
            "unstaged_false_unknown_rate": 0.75,
            "staged_rejected_total": 3,
            "rejected_by_stage_one_total": 2,
            "stage_one_also_rejected_by_unstaged_control": 2,
            "rejected_by_stage_two_only": 1,
            "attribution_partition_total": 3,
            "attribution_partition_exact": True,
            "attribution": "unit",
        }
        evaluator._assert_attribution_conservation(
            report,
            gated=gated,
            staged_score=staged_score,
            unstaged_score=unstaged_score,
            staged_threshold=0.5,
            unstaged_threshold=0.5,
        )
        for field in (
            "rows",
            "staged_threshold",
            "unstaged_threshold",
            "staged_rejected_total",
            "rejected_by_stage_one_total",
            "stage_one_also_rejected_by_unstaged_control",
            "rejected_by_stage_two_only",
            "attribution_partition_total",
            "attribution_partition_exact",
            "staged_false_unknown_rate",
            "stage_one_gate_rate",
            "stage_two_false_unknown_rate_marginal",
            "stage_two_false_unknown_rate_among_survivors",
            "unstaged_false_unknown_rate",
        ):
            mutated = dict(report)
            mutated[field] = (
                False
                if field == "attribution_partition_exact"
                else 99.0
                if field in {"staged_threshold", "unstaged_threshold"}
                else 99
            )
            with self.subTest(field=field):
                with self.assertRaises((AssertionError, ValueError)):
                    evaluator._assert_attribution_conservation(
                        mutated,
                        gated=gated,
                        staged_score=staged_score,
                        unstaged_score=unstaged_score,
                        staged_threshold=0.5,
                        unstaged_threshold=0.5,
                    )

    def test_frozen_architecture_is_time_domain_no_fft_and_exactly_ordered(self):
        self.assertEqual(
            evaluator.EXPECTED_FROZEN_ARCHITECTURE,
            {
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
            },
        )


if __name__ == "__main__":
    unittest.main()
