"""Fail-closed tests for the deployable dual-fusion staging packager."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("package-v3-staging-assets.py")
SPEC = importlib.util.spec_from_file_location("package_v3_staging_assets", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
packager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packager)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def record(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": packager.sha256(path)}


class SyntheticInputs:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.rejector = root / "rejector-export"
        self.classifier = root / "classifier-export"
        self.openset = root / "openset-export"
        self.destination = root / "dual-package"
        for directory in (self.rejector, self.classifier, self.openset):
            directory.mkdir(parents=True)
        self.report_sha = digest("validation report")
        self.staged = {
            "v3_branch_lof_components.npz": digest("lof"),
            "v3_open_policy_stage_two.npz": digest("stage two"),
            "v3_staged_composite_policy.npz": digest("composite"),
        }
        self.build_role(
            self.rejector,
            role=packager.REJECTOR_ROLE,
            weights_name=packager.REJECTOR_WEIGHTS,
            probe_name=packager.REJECTOR_PROBE,
            bundle_sha=digest("rejector bundle"),
        )
        self.build_role(
            self.classifier,
            role=packager.CLASSIFIER_ROLE,
            weights_name=packager.CLASSIFIER_WEIGHTS,
            probe_name=packager.CLASSIFIER_PROBE,
            bundle_sha=digest("classifier bundle"),
        )
        self.build_openset()

    def build_role(
        self,
        directory: Path,
        *,
        role: str,
        weights_name: str,
        probe_name: str,
        bundle_sha: str,
    ) -> None:
        def linear(inputs: int, outputs: int) -> dict:
            return {
                "in": inputs,
                "out": outputs,
                "weight": [0.01] * (inputs * outputs),
                "bias": [0.0] * outputs,
            }

        common_config = {
            "patch_length": 64,
            "patch_count": 16,
            "patch_dim": 2,
            "hidden": 3,
            "embed_dim": 2,
            "n_features": 12,
            "set_pool": "mean_std",
            "dropout": 0.0,
        }
        real = {
            "config": {**common_config, "encoder": "real"},
            "blocks": [
                {
                    "in_channels": 2,
                    "out_channels": 2,
                    "kernel": 3,
                    "stride": 2,
                    "padding": 1,
                    "conv_weight": [0.01] * 12,
                    "batch_norm": {
                        "eps": 1e-5,
                        "weight": [1.0, 1.0],
                        "bias": [0.0, 0.0],
                        "running_mean": [0.0, 0.0],
                        "running_var": [1.0, 1.0],
                    },
                }
            ],
            "patch_projection": linear(4, 2),
            "fc1": linear(16, 3),
            "fc2": linear(3, 2),
        }
        complex_branch = {
            "config": {**common_config, "encoder": "complex"},
            "stages": [
                {
                    "in_channels": 1,
                    "out_channels": 2,
                    "kernel": 3,
                    "padding": 1,
                    "weight_real": [0.01] * 6,
                    "weight_imag": [0.01] * 6,
                    "threshold_raw": [0.0, 0.0],
                }
            ],
            "mag_norm_eps": 1e-5,
            "modrelu_magnitude_eps": 1e-8,
            "lowpass": [0.25, 0.5, 0.25],
            "lags": [1],
            "patch_projection": linear(8, 2),
            "fc1": linear(16, 3),
            "fc2": linear(3, 2),
        }
        write_json(
            directory / weights_name,
            {
                "schema": packager.FUSION_WEIGHTS_SCHEMA,
                "schema_version": 1,
                "status": packager.STAGING_STATUS,
                "runtime_role": role,
                "packed_length": 1024,
                "frontend": {
                    "version": "invariant-patch-time-domain-v1",
                    "patch_length": 64,
                    "patch_count": 16,
                    "target_frac": 0.5,
                    "feature_count": 12,
                    "uses_frequency_transform": False,
                },
                "real": real,
                "complex": complex_branch,
                "fusion": {
                    "real_center": [0.0, 0.0],
                    "complex_center": [0.0, 0.0],
                    "alpha_real": 1.0,
                    "alpha_complex": 1.0,
                    "weight_real": 0.5,
                    "eps": 1e-8,
                },
                "feature_standardization": {
                    "mean": [0.0] * 12,
                    "std": [1.0] * 12,
                },
                "classification": {
                    "classes": ["a", "b"],
                    "prototypes": [
                        [0.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0, 1.0],
                    ],
                },
                "provenance": {
                    "source_bundle_manifest_sha256": bundle_sha,
                },
            },
        )
        write_json(
            directory / probe_name,
            {"schema": "synthetic-probe", "cases": []},
        )
        manifest = {
            "schema": packager.FUSION_EXPORT_SCHEMA,
            "schema_version": packager.FUSION_EXPORT_SCHEMA_VERSION,
            "runtime_role": role,
            "source_bundle_manifest_sha256": bundle_sha,
            "emitted": {
                weights_name: record(directory / weights_name),
                probe_name: record(directory / probe_name),
            },
        }
        write_json(directory / packager.FUSION_MANIFEST, manifest)

    def role_state(self, directory: Path, weights: str, probe: str) -> dict:
        manifest_path = directory / packager.FUSION_MANIFEST
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {
            "manifest": manifest,
            "manifest_record": record(manifest_path),
            "weights_record": record(directory / weights),
            "probe_record": record(directory / probe),
        }

    def build_openset(self) -> None:
        rejector = self.role_state(
            self.rejector, packager.REJECTOR_WEIGHTS, packager.REJECTOR_PROBE
        )
        classifier = self.role_state(
            self.classifier,
            packager.CLASSIFIER_WEIGHTS,
            packager.CLASSIFIER_PROBE,
        )
        rejector_bundle = rejector["manifest"][
            "source_bundle_manifest_sha256"
        ]
        classifier_bundle = classifier["manifest"][
            "source_bundle_manifest_sha256"
        ]
        rejector_fusion = digest("rejector fusion")
        classifier_fusion = digest("classifier fusion")
        composite_calibration = [0.1, 0.2, 0.3, 0.4]
        combined_calibration = [0.1, 0.2, 0.4, 0.8]
        stage_two_scores = [
            packager.bisect_left(combined_calibration, value)
            / (len(combined_calibration) + 1)
            for value in combined_calibration
        ]
        stage_two_threshold = packager._numpy_linear_quantile(
            stage_two_scores, packager.STAGE_TWO_THRESHOLD_QUANTILE
        )

        def prefilter_model(length: int) -> dict:
            width = len(packager.PREFILTER_FEATURE_NAMES)
            return {
                "capture_length": length,
                "feature_names": packager.PREFILTER_FEATURE_NAMES,
                "mean": [0.0] * width,
                "scale": [1.0] * width,
                "coefficients": [0.1] * width,
                "intercept": 0.0,
                "threshold_score": 1.0,
            }

        def lof_component(branch: str) -> dict:
            shape = packager.EXPECTED_LOF_COMPONENTS[branch]
            rows = shape["neighbors"] + 1
            return {
                "branch": branch,
                "weight": shape["weight"],
                "neighbors": shape["neighbors"],
                "mean": [0.0, 0.0],
                "scale": [1.0, 1.0],
                "reference": [[float(index), 0.0] for index in range(rows)],
                "reference_k_distance": [0.1] * rows,
                "reference_local_density": [1.0] * rows,
                "calibration": [0.1, 0.2, 0.3],
            }

        policy = {
            "schema": packager.OPENSET_SCHEMA,
            "schema_version": packager.OPENSET_SCHEMA_VERSION,
            "status": packager.STAGING_STATUS,
            "contract": {
                "additive_only": False,
                "changes_closed_label": True,
                "gates_before_classification": True,
                "architecture_contract_change": "stage one gates",
            },
            "frontend": {
                "version": packager.FRONTEND_VERSION,
                "patch_length": 64,
                "patch_count": 16,
                "target_frac": 0.5,
                "packed_length": 1024,
                "uses_frequency_transform": False,
            },
            "stage_one": {
                "kind": packager.NOISE_PREFILTER_VERSION,
                "feature_names": packager.PREFILTER_FEATURE_NAMES,
                "match_frontend_min_bandwidth": True,
                "set_sha256": digest("prefilter set"),
                "models": {
                    str(length): prefilter_model(length)
                    for length in packager.REQUIRED_STAGE_ONE_LENGTHS
                },
            },
            "stage_two": {
                "kind": packager.STAGE_TWO_POLICY_KIND,
                "lof_components": [
                    lof_component("real"),
                    lof_component("complex"),
                ],
                "policy": {
                    "branch_lof_rank_weight": 0.8,
                    "geometry_weight": 0.2,
                    "threshold_quantile": (
                        packager.STAGE_TWO_THRESHOLD_QUANTILE
                    ),
                    "threshold": stage_two_threshold,
                    "geometry_feature": packager.FROZEN_GEOMETRY_FEATURE,
                    "class_geometry_mean": [0.0, 0.1],
                    "class_geometry_scale": [1.0, 1.0],
                    "geometry_calibration": [0.1, 0.2, 0.3, 0.4],
                    "combined_calibration": combined_calibration,
                },
            },
            "composite": {
                "schema": packager.STAGED_POLICY_SCHEMA,
                "kind": packager.STAGED_POLICY_KIND,
                "policy_version": packager.STAGED_POLICY_VERSION,
                "survivor_score": packager.COMPOSITE_SURVIVOR_SCORE,
                "threshold_quantile": (
                    packager.COMPOSITE_THRESHOLD_QUANTILE
                ),
                "threshold": packager._numpy_linear_quantile(
                    composite_calibration,
                    packager.COMPOSITE_THRESHOLD_QUANTILE,
                ),
                "stage_two_threshold": stage_two_threshold,
                "stage_one_calibration_raw": [-3.0, -1.0, 0.0, 1.0],
                "composite_calibration_raw": composite_calibration,
                "enrollment_rows": 6,
                "enrollment_gated_rows": 2,
                "enrollment_capture_length": 16384,
                "threshold_population": (
                    "enrollment_stage_one_survivors_only"
                ),
                "training_rows_used_for_threshold": 0,
                "selection_rows_used_for_threshold": 0,
                "novelty_rows_used_for_threshold": 0,
                "release_rows_used_for_threshold": 0,
                "stage_one_known_false_positive_budget": (
                    packager.STAGE_ONE_KNOWN_FALSE_POSITIVE_BUDGET
                ),
                "survivor_known_false_positive_budget": (
                    packager.SURVIVOR_KNOWN_FALSE_POSITIVE_BUDGET
                ),
                "nominal_enrollment_false_unknown_budget": (
                    packager.NOMINAL_ENROLLMENT_FALSE_UNKNOWN_BUDGET
                ),
                "only_policy_change": (
                    packager.COMPOSITE_ONLY_POLICY_CHANGE
                ),
                "stage_one_changed": False,
                "rejector_cnn_fusion_changed": False,
                "classifier_cnn_fusion_changed": False,
                "gate_contract_changed": False,
            },
            "provenance": {
                "candidate_id": packager.CANDIDATE_ID,
                "staged_validation_status": "development_openset_pass",
                "staged_validation_role": "validate",
                "staged_validation_all_pass": True,
                "development_only": True,
                "release_evidence": False,
                "sealed_release_data_used": 0,
                "consumed_test_rows_used": 0,
                "release_seed_not_spent": packager.RELEASE_SEED_NOT_SPENT,
                "runtime_roles": {
                    "rejector": {
                        "runtime_role": packager.REJECTOR_ROLE,
                        "browser_asset": packager.REJECTOR_WEIGHTS,
                        "browser_asset_sha256": rejector["weights_record"][
                            "sha256"
                        ],
                        "browser_export_manifest_sha256": rejector[
                            "manifest_record"
                        ]["sha256"],
                        "runtime_bundle_manifest_sha256": rejector_bundle,
                        "fusion_directory_sha256": rejector_fusion,
                    },
                    "classifier": {
                        "runtime_role": packager.CLASSIFIER_ROLE,
                        "browser_asset": packager.CLASSIFIER_WEIGHTS,
                        "browser_asset_sha256": classifier["weights_record"][
                            "sha256"
                        ],
                        "browser_export_manifest_sha256": classifier[
                            "manifest_record"
                        ]["sha256"],
                        "runtime_bundle_manifest_sha256": classifier_bundle,
                        "fusion_directory_sha256": classifier_fusion,
                    },
                },
                "staged_validation_report_sha256": self.report_sha,
                "staged_artifact_sha256": self.staged,
            },
        }
        write_json(self.openset / packager.OPENSET_POLICY, policy)
        policy_record = record(self.openset / packager.OPENSET_POLICY)
        binding = {
            "schema": packager.DUAL_BINDING_SCHEMA,
            "schema_version": 1,
            "status": packager.STAGING_STATUS,
            "candidate_id": packager.CANDIDATE_ID,
            "frontend": {
                "version": "invariant-patch-time-domain-v1",
                "patch_length": 64,
                "patch_count": 16,
                "target_frac": 0.5,
                "packed_length": 1024,
                "uses_frequency_transform": False,
            },
            "execution_order": [
                "stage_one_noise_gate",
                "rejector_known_unknown",
                "classifier_known_label",
            ],
            "roles": {
                "rejector": {
                    "asset": packager.REJECTOR_WEIGHTS,
                    "asset_sha256": rejector["weights_record"]["sha256"],
                    "fusion_directory_sha256": rejector_fusion,
                    "runtime_bundle_manifest_sha256": rejector_bundle,
                    "runtime_role": packager.REJECTOR_ROLE,
                    "responsibility": packager.REJECTOR_RESPONSIBILITY,
                },
                "classifier": {
                    "asset": packager.CLASSIFIER_WEIGHTS,
                    "asset_sha256": classifier["weights_record"]["sha256"],
                    "fusion_directory_sha256": classifier_fusion,
                    "runtime_bundle_manifest_sha256": classifier_bundle,
                    "runtime_role": packager.CLASSIFIER_ROLE,
                    "responsibility": packager.CLASSIFIER_RESPONSIBILITY,
                },
            },
            "openset_policy": {
                "asset": packager.OPENSET_POLICY,
                "asset_sha256": policy_record["sha256"],
                "rejector_asset_sha256": rejector["weights_record"]["sha256"],
                "fitted_rejector_runtime_bundle_manifest_sha256": (
                    rejector_bundle
                ),
                "staged_validation_report_sha256": self.report_sha,
                "staged_artifacts_sha256": self.staged,
            },
            "validation": {
                "report_sha256": self.report_sha,
                "role": "validate",
                "status": "development_openset_pass",
                "design_novelty_seed": packager.DESIGN_NOVELTY_SEED,
                "novelty_seeds": packager.VALIDATION_NOVELTY_SEEDS,
                "release_seed_not_spent": packager.RELEASE_SEED_NOT_SPENT,
            },
            "fail_closed": {
                "role_assets_bound_by_sha256": True,
                "distinct_role_assets": True,
                "role_asset_sha256_must_differ": True,
                "classifier_runs_only_after_rejector_acceptance": True,
                "public_known_label_from_classifier_only": True,
            },
        }
        write_json(self.openset / packager.DUAL_BINDING, binding)

        def prediction(runtime_role: str) -> dict:
            real_embedding = [0.0, 0.0]
            complex_embedding = [0.0, 0.0]
            fusion = {
                "real_center": [0.0, 0.0],
                "complex_center": [0.0, 0.0],
                "alpha_real": 1.0,
                "alpha_complex": 1.0,
                "weight_real": 0.5,
                "eps": 1e-8,
            }
            fused_embedding = packager._fuse_float32(
                real_embedding, complex_embedding, fusion
            )
            distances = packager._squared_distances_float32(
                fused_embedding,
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0],
                ],
            )
            return {
                "runtime_role": runtime_role,
                "real_embedding": real_embedding,
                "complex_embedding": complex_embedding,
                "fused_embedding": fused_embedding,
                "predicted_class_index": 0,
                "predicted_class_label": "a",
                "squared_prototype_distances": distances,
            }

        def stage_one_row(length: int, features: list[float]) -> dict:
            evaluated_length = 16384 if length == 32768 else length
            model = policy["stage_one"]["models"][str(evaluated_length)]
            score = model["intercept"]
            for feature, mean, scale, coefficient in zip(
                features,
                model["mean"],
                model["scale"],
                model["coefficients"],
            ):
                score += coefficient * ((feature - mean) / scale)
            rank = 0.5 * (score / (1.0 + abs(score))) + 0.5
            return {
                "capture_length": length,
                "evaluated_length": evaluated_length,
                "causal_prefix_applied": length == 32768,
                "features": features,
                "score": score,
                "rank": rank,
                "gated": score >= model["threshold_score"],
                "threshold_score": model["threshold_score"],
            }

        def stage_two_row(raw: float, deviation: float) -> dict:
            lof = []
            weighted = 0.0
            for component in policy["stage_two"]["lof_components"]:
                rank = (
                    packager.bisect_left(component["calibration"], raw)
                    / (len(component["calibration"]) + 1)
                )
                weighted += component["weight"] * rank
                lof.append(
                    {
                        "branch": component["branch"],
                        "weight": component["weight"],
                        "neighbors": component["neighbors"],
                        "raw": raw,
                        "rank": rank,
                    }
                )
            stage_two_policy = policy["stage_two"]["policy"]
            geometry_rank = (
                packager.bisect_left(
                    stage_two_policy["geometry_calibration"], deviation
                )
                / (len(stage_two_policy["geometry_calibration"]) + 1)
            )
            combined = (
                stage_two_policy["branch_lof_rank_weight"] * weighted
                + stage_two_policy["geometry_weight"] * geometry_rank
            )
            score = (
                packager.bisect_left(
                    stage_two_policy["combined_calibration"], combined
                )
                / (len(stage_two_policy["combined_calibration"]) + 1)
            )
            return {
                **prediction(packager.REJECTOR_ROLE),
                "lof": lof,
                "lof_rank": weighted,
                "geometry_statistic": deviation,
                "geometry_deviation": deviation,
                "geometry_rank": geometry_rank,
                "combined_raw": combined,
                "score": score,
                "threshold": stage_two_policy["threshold"],
            }

        def parity_row(
            *,
            family: str,
            name: str,
            length: int,
            stage_one_features: list[float],
            stage_two_raw: float = 0.05,
            stage_two_deviation: float = 0.05,
            known: bool = False,
        ) -> dict:
            first = stage_one_row(length, stage_one_features)
            threshold = policy["composite"]["threshold"]
            if first["gated"]:
                second = None
                survivor_rank = None
                composite_score = None
                staged_score = 1.0 + first["rank"]
                rejected_stage = 1
                decision_label = "noise"
                rejector_index = None
                rejector_label = None
                classifier = None
                public_result = {
                    "outcome": "noise",
                    "label": "noise",
                    "knownLabel": None,
                    "knownWinnerIndex": None,
                    "squaredPrototypeDistances": None,
                }
            else:
                second = stage_two_row(
                    stage_two_raw, stage_two_deviation
                )
                calibration = policy["composite"][
                    "stage_one_calibration_raw"
                ]
                survivor_rank = (
                    packager.bisect_left(calibration, first["score"])
                    / (len(calibration) + 1)
                )
                composite_score = max(second["score"], survivor_rank)
                staged_score = composite_score
                rejected_stage = 2 if composite_score > threshold else None
                rejector_index = second["predicted_class_index"]
                rejector_label = second["predicted_class_label"]
                if rejected_stage == 2:
                    decision_label = "unknown"
                    classifier = None
                    public_result = {
                        "outcome": "unknown",
                        "label": "unknown",
                        "knownLabel": None,
                        "knownWinnerIndex": None,
                        "squaredPrototypeDistances": None,
                    }
                else:
                    classifier = prediction(packager.CLASSIFIER_ROLE)
                    decision_label = classifier["predicted_class_label"]
                    public_result = {
                        "outcome": "known",
                        "label": decision_label,
                        "knownLabel": decision_label,
                        "knownWinnerIndex": classifier[
                            "predicted_class_index"
                        ],
                        "squaredPrototypeDistances": classifier[
                            "squared_prototype_distances"
                        ],
                    }
            feature_vector = [0.0] * packager.TIME_DOMAIN_FEATURE_COUNT
            row = {
                "family": family,
                "name": name,
                "capture_length": length,
                "iq": {
                    "in_phase": [0.1] * length,
                    "quadrature": [0.0] * length,
                },
                "stage_one": first,
                "packed_iq": {
                    "in_phase": [0.1] * 1024,
                    "quadrature": [0.0] * 1024,
                },
                "raw_features": feature_vector,
                "standardized_features": feature_vector,
                "rejector_standardized_features": feature_vector,
                "classifier_standardized_features": (
                    feature_vector if classifier is not None else None
                ),
                "stage_two": second,
                "rejector_closed_winner_index": rejector_index,
                "rejector_closed_winner_label": rejector_label,
                "classifier": classifier,
                "classifier_executed": classifier is not None,
                "stage_one_survivor_rank": survivor_rank,
                "composite_score": composite_score,
                "staged_threshold": threshold,
                "staged_score": staged_score,
                "rejected_stage": rejected_stage,
                "decision_label": decision_label,
                "public_result": public_result,
            }
            if known:
                row.update(
                    {
                        "class_index": 0,
                        "class_label": "a",
                        "corpus_index": 0,
                    }
                )
            if length == 32768:
                row["parity_source_name"] = f"{name}-N16384"
            return row

        parity_rows = [
            parity_row(
                family="known",
                name="known",
                length=4096,
                stage_one_features=[-2.0] * 7,
                known=True,
            ),
            parity_row(
                family="noise",
                name="noise",
                length=8192,
                stage_one_features=[2.0] * 7,
            ),
            parity_row(
                family="chirp",
                name="chirp",
                length=16384,
                stage_one_features=[0.0] * 7,
                stage_two_raw=0.15,
                stage_two_deviation=0.15,
            ),
            parity_row(
                family="noise",
                name="causal-gated",
                length=32768,
                stage_one_features=[2.0] * 7,
            ),
            parity_row(
                family="chirp",
                name="causal-survivor",
                length=32768,
                stage_one_features=[-2.0] * 7,
            ),
        ]
        parity = {
            "schema": "time-domain-openset-parity-v1",
            "schema_version": packager.PARITY_SCHEMA_VERSION,
            "status": packager.STAGING_STATUS,
            "candidate_id": packager.CANDIDATE_ID,
            "contract": policy["contract"],
            "classes": ["a", "b"],
            "frontend": policy["frontend"],
            "policy_schema": packager.STAGED_POLICY_SCHEMA,
            "policy_version": packager.STAGED_POLICY_VERSION,
            "policy_kind": packager.STAGED_POLICY_KIND,
            "design_novelty_seed": packager.DESIGN_NOVELTY_SEED,
            "validation_novelty_seeds": packager.VALIDATION_NOVELTY_SEEDS,
            "release_seed_not_spent": packager.RELEASE_SEED_NOT_SPENT,
            "parity_novelty_seed": packager.PARITY_NOVELTY_SEED,
            "parity_seed_is_evaluation_evidence": False,
            "fixture_lengths": packager.REQUIRED_PARITY_LENGTHS,
            "causal_prefix_parity": {
                "capture_length": 32768,
                "evaluated_length": 16384,
                "row_names": ["causal-gated", "causal-survivor"],
                "covers_gated_and_survivor_paths": True,
            },
            "role_contract": {
                "classifier_runs_only_after_rejector_acceptance": True,
                "public_known_label_from_classifier_only": True,
            },
            "staged_threshold": policy["composite"]["threshold"],
            "stage_two_threshold": policy["stage_two"]["policy"]["threshold"],
            "provenance": policy["provenance"],
            "counts": {
                "rows": 5,
                "known": 1,
                "noise": 2,
                "chirp": 2,
                "probe": 0,
                "gated_stage_one": 2,
                "rejected_stage_two": 1,
                "accepted": 2,
                "classifier_executed": 2,
                "accepted_role_winner_disagreements": 0,
            },
            "accepted_role_winner_disagreement_rows": [],
            "rows": parity_rows,
        }
        write_json(self.openset / packager.OPENSET_PARITY, parity)
        self.refresh_openset_manifest(rejector, classifier)

    def refresh_openset_manifest(
        self,
        rejector: dict | None = None,
        classifier: dict | None = None,
    ) -> None:
        if rejector is None:
            rejector = self.role_state(
                self.rejector,
                packager.REJECTOR_WEIGHTS,
                packager.REJECTOR_PROBE,
            )
        if classifier is None:
            classifier = self.role_state(
                self.classifier,
                packager.CLASSIFIER_WEIGHTS,
                packager.CLASSIFIER_PROBE,
            )
        manifest = {
            "schema": packager.OPENSET_EXPORT_SCHEMA,
            "schema_version": 1,
            "status": packager.STAGING_STATUS,
            "candidate_id": packager.CANDIDATE_ID,
            "outputs": {
                name: record(self.openset / name)
                for name in (
                    packager.OPENSET_POLICY,
                    packager.DUAL_BINDING,
                    packager.OPENSET_PARITY,
                )
            },
            "role_exports": {
                "rejector": {
                    "manifest_sha256": rejector["manifest_record"]["sha256"],
                    "runtime_role": packager.REJECTOR_ROLE,
                    "weights": {
                        "name": packager.REJECTOR_WEIGHTS,
                        **rejector["weights_record"],
                    },
                    "probe": {
                        "name": packager.REJECTOR_PROBE,
                        **rejector["probe_record"],
                    },
                },
                "classifier": {
                    "manifest_sha256": classifier["manifest_record"]["sha256"],
                    "runtime_role": packager.CLASSIFIER_ROLE,
                    "weights": {
                        "name": packager.CLASSIFIER_WEIGHTS,
                        **classifier["weights_record"],
                    },
                    "probe": {
                        "name": packager.CLASSIFIER_PROBE,
                        **classifier["probe_record"],
                    },
                },
            },
        }
        write_json(self.openset / packager.OPENSET_MANIFEST, manifest)

    def package(self) -> dict:
        return packager.package(
            self.rejector,
            self.classifier,
            self.openset,
            self.destination,
        )


class PackageTests(unittest.TestCase):
    def test_rederived_float_allows_one_ulp_without_boundary_crossing(
        self,
    ) -> None:
        observed = math.nextafter(1.0, math.inf)
        packager._require_rederived_float_within_ulps(
            observed,
            1.0,
            "independent arithmetic",
            gate_boundary=2.0,
            rank_boundaries=[-1.0, 0.0, 2.0],
        )

    def test_rederived_float_refuses_meaningful_drift(self) -> None:
        observed = 1.0
        for _ in range(packager.MAX_REDERIVED_FLOAT_ULPS + 1):
            observed = math.nextafter(observed, math.inf)
        with self.assertRaisesRegex(
            packager.PackageError, "IEEE-754 binary64 ULPs"
        ):
            packager._require_rederived_float_within_ulps(
                observed, 1.0, "independent arithmetic"
            )

    def test_rederived_float_refuses_gate_or_rank_boundary_flip(self) -> None:
        with self.subTest(boundary="gate"):
            with self.assertRaisesRegex(packager.PackageError, "gate boundary"):
                packager._require_rederived_float_within_ulps(
                    1.0,
                    math.nextafter(1.0, -math.inf),
                    "independent arithmetic",
                    gate_boundary=1.0,
                )
        with self.subTest(boundary="rank"):
            with self.assertRaisesRegex(
                packager.PackageError, "empirical-rank"
            ):
                packager._require_rederived_float_within_ulps(
                    math.nextafter(1.0, math.inf),
                    1.0,
                    "independent arithmetic",
                    rank_boundaries=[1.0],
                )

    def test_materializes_only_four_runtime_assets_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticInputs(Path(temporary))
            manifest = fixture.package()
            self.assertEqual(manifest["schema"], packager.PACKAGE_SCHEMA)
            self.assertEqual(
                set(manifest["assets"]), set(packager.DEPLOYABLE_ASSETS)
            )
            self.assertEqual(
                {path.name for path in fixture.destination.iterdir()},
                {*packager.DEPLOYABLE_ASSETS, packager.PACKAGE_MANIFEST},
            )
            self.assertNotIn(
                packager.OPENSET_PARITY,
                {path.name for path in fixture.destination.iterdir()},
            )
            self.assertFalse(
                manifest["external_evidence"]["parity"]["packaged"]
            )
            self.assertEqual(
                manifest["roles"]["rejector"]["asset"],
                manifest["assets"][packager.REJECTOR_WEIGHTS],
            )
            self.assertEqual(
                manifest["roles"]["classifier"]["asset"],
                manifest["assets"][packager.CLASSIFIER_WEIGHTS],
            )
            for path in fixture.destination.iterdir():
                self.assertLess(
                    path.stat().st_size, packager.MAX_DEPLOYABLE_BYTES
                )

    def test_legacy_single_fusion_manifest_is_refused_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticInputs(Path(temporary))
            path = fixture.openset / packager.OPENSET_MANIFEST
            value = json.loads(path.read_text(encoding="utf-8"))
            value["schema"] = "time-domain-v3-openset-staging-manifest-v1"
            write_json(path, value)
            with self.assertRaisesRegex(packager.PackageError, "dual staging"):
                fixture.package()
            self.assertFalse(fixture.destination.exists())

    def test_swapped_runtime_role_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticInputs(Path(temporary))
            weights = fixture.classifier / packager.CLASSIFIER_WEIGHTS
            value = json.loads(weights.read_text(encoding="utf-8"))
            value["runtime_role"] = packager.REJECTOR_ROLE
            write_json(weights, value)
            manifest_path = fixture.classifier / packager.FUSION_MANIFEST
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["emitted"][packager.CLASSIFIER_WEIGHTS] = record(weights)
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(packager.PackageError, "exact role"):
                fixture.package()
            self.assertFalse(fixture.destination.exists())

    def test_rejected_parity_row_may_not_serialize_classifier_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticInputs(Path(temporary))
            parity_path = fixture.openset / packager.OPENSET_PARITY
            parity = json.loads(parity_path.read_text(encoding="utf-8"))
            parity["rows"][1]["classifier_executed"] = True
            parity["rows"][1]["classifier"] = {
                "predicted_class_label": "leak"
            }
            write_json(parity_path, parity)
            fixture.refresh_openset_manifest()
            with self.assertRaisesRegex(
                packager.PackageError, "rejected parity row"
            ):
                fixture.package()

    def test_legacy_q95_and_failed_q99_policies_are_refused(self) -> None:
        mutations = (
            (
                2,
                "v3-staged-openset-policy-v2-composite-survivor",
                "v3_staged_noise_prefilter_then_composite_survivor_lof_geometry",
                0.95,
            ),
            (
                3,
                "v3-staged-openset-policy-v3-composite-survivor-q99",
                (
                    "v3_staged_noise_prefilter_then_q99_"
                    "composite_survivor_lof_geometry"
                ),
                0.99,
            ),
        )
        for schema, version, kind, quantile in mutations:
            with self.subTest(schema=schema):
                with tempfile.TemporaryDirectory() as temporary:
                    fixture = SyntheticInputs(Path(temporary))
                    path = fixture.openset / packager.OPENSET_POLICY
                    policy = json.loads(path.read_text(encoding="utf-8"))
                    policy["composite"].update(
                        {
                            "schema": schema,
                            "policy_version": version,
                            "kind": kind,
                            "threshold_quantile": quantile,
                        }
                    )
                    write_json(path, policy)
                    fixture.refresh_openset_manifest()
                    with self.assertRaisesRegex(
                        packager.PackageError, "composite"
                    ):
                        fixture.package()

    def test_schema4_hygiene_mutations_are_refused(self) -> None:
        mutations = {
            "threshold_population": "selection",
            "training_rows_used_for_threshold": 1,
            "selection_rows_used_for_threshold": 1,
            "novelty_rows_used_for_threshold": 1,
            "release_rows_used_for_threshold": 1,
            "stage_one_known_false_positive_budget": 0.02,
            "survivor_known_false_positive_budget": 0.04,
            "nominal_enrollment_false_unknown_budget": 0.05,
            "only_policy_change": "anything else",
            "stage_one_changed": True,
            "rejector_cnn_fusion_changed": True,
            "classifier_cnn_fusion_changed": True,
            "gate_contract_changed": True,
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as temporary:
                    fixture = SyntheticInputs(Path(temporary))
                    path = fixture.openset / packager.OPENSET_POLICY
                    policy = json.loads(path.read_text(encoding="utf-8"))
                    policy["composite"][field] = value
                    write_json(path, policy)
                    fixture.refresh_openset_manifest()
                    with self.assertRaisesRegex(
                        packager.PackageError, field
                    ):
                        fixture.package()

    def test_validation_tuple_design_and_release_are_frozen(self) -> None:
        mutations = {
            "novelty_seeds": list(
                reversed(packager.VALIDATION_NOVELTY_SEEDS)
            ),
            "design_novelty_seed": packager.DESIGN_NOVELTY_SEED + 1,
            "release_seed_not_spent": packager.RELEASE_SEED_NOT_SPENT + 1,
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as temporary:
                    fixture = SyntheticInputs(Path(temporary))
                    binding_path = fixture.openset / packager.DUAL_BINDING
                    binding = json.loads(
                        binding_path.read_text(encoding="utf-8")
                    )
                    binding["validation"][field] = value
                    write_json(binding_path, binding)
                    fixture.refresh_openset_manifest()
                    with self.assertRaisesRegex(
                        packager.PackageError, "validation"
                    ):
                        fixture.package()

    def test_every_runtime_required_openset_structure_is_required(self) -> None:
        removals = (
            ("frontend", "packed_length"),
            ("stage_one", "models"),
            ("stage_one", "models", "4096", "coefficients"),
            ("stage_two", "lof_components"),
            (
                "stage_two",
                "lof_components",
                0,
                "reference_k_distance",
            ),
            ("stage_two", "lof_components", 1, "calibration"),
            ("stage_two", "policy", "geometry_feature"),
            ("stage_two", "policy", "class_geometry_mean"),
            ("stage_two", "policy", "class_geometry_scale"),
            ("stage_two", "policy", "geometry_calibration"),
            ("stage_two", "policy", "combined_calibration"),
        )
        for keys in removals:
            with self.subTest(keys=keys):
                with tempfile.TemporaryDirectory() as temporary:
                    fixture = SyntheticInputs(Path(temporary))
                    path = fixture.openset / packager.OPENSET_POLICY
                    policy = json.loads(path.read_text(encoding="utf-8"))
                    parent = policy
                    for key in keys[:-1]:
                        parent = parent[key]
                    del parent[keys[-1]]
                    write_json(path, policy)
                    fixture.refresh_openset_manifest()
                    with self.assertRaises(packager.PackageError):
                        fixture.package()

    def test_every_ts_required_fusion_structure_is_required(self) -> None:
        removals = (
            ("packed_length",),
            ("real",),
            ("complex",),
            ("fusion",),
            ("feature_standardization",),
            ("classification", "prototypes"),
        )
        for keys in removals:
            with self.subTest(keys=keys):
                with tempfile.TemporaryDirectory() as temporary:
                    fixture = SyntheticInputs(Path(temporary))
                    weights_path = fixture.rejector / packager.REJECTOR_WEIGHTS
                    weights = json.loads(
                        weights_path.read_text(encoding="utf-8")
                    )
                    parent = weights
                    for key in keys[:-1]:
                        parent = parent[key]
                    del parent[keys[-1]]
                    write_json(weights_path, weights)
                    manifest_path = (
                        fixture.rejector / packager.FUSION_MANIFEST
                    )
                    manifest = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                    manifest["emitted"][packager.REJECTOR_WEIGHTS] = record(
                        weights_path
                    )
                    write_json(manifest_path, manifest)
                    with self.assertRaises(packager.PackageError):
                        fixture.package()

    def test_fusion_frontend_is_cross_bound_to_branches_and_top_level(
        self,
    ) -> None:
        mutations = (
            ("frontend", "patch_length", 32),
            ("real", "config", "patch_count", 8),
            ("packed_length", 2048),
        )
        for keys in mutations:
            with self.subTest(keys=keys):
                with tempfile.TemporaryDirectory() as temporary:
                    fixture = SyntheticInputs(Path(temporary))
                    weights_path = fixture.rejector / packager.REJECTOR_WEIGHTS
                    weights = json.loads(
                        weights_path.read_text(encoding="utf-8")
                    )
                    parent = weights
                    for key in keys[:-2]:
                        parent = parent[key]
                    if len(keys) == 2:
                        parent[keys[0]] = keys[1]
                    else:
                        parent[keys[-2]] = keys[-1]
                    write_json(weights_path, weights)
                    manifest_path = (
                        fixture.rejector / packager.FUSION_MANIFEST
                    )
                    manifest = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                    manifest["emitted"][packager.REJECTOR_WEIGHTS] = record(
                        weights_path
                    )
                    write_json(manifest_path, manifest)
                    with self.assertRaises(packager.PackageError):
                        fixture.package()

    def test_lof_widths_are_bound_to_rejector_embedding_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticInputs(Path(temporary))
            path = fixture.openset / packager.OPENSET_POLICY
            policy = json.loads(path.read_text(encoding="utf-8"))
            component = policy["stage_two"]["lof_components"][0]
            component["mean"].append(0.0)
            component["scale"].append(1.0)
            for row in component["reference"]:
                row.append(0.0)
            write_json(path, policy)
            fixture.refresh_openset_manifest()
            with self.assertRaisesRegex(
                packager.PackageError, "width differs from rejector"
            ):
                fixture.package()

    def test_stage_two_q95_is_derived_not_repeated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticInputs(Path(temporary))
            policy_path = fixture.openset / packager.OPENSET_POLICY
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
            policy["stage_two"]["policy"]["threshold"] = 0.123
            policy["composite"]["stage_two_threshold"] = 0.123
            write_json(policy_path, policy)

            binding_path = fixture.openset / packager.DUAL_BINDING
            binding = json.loads(binding_path.read_text(encoding="utf-8"))
            binding["openset_policy"]["asset_sha256"] = record(policy_path)[
                "sha256"
            ]
            write_json(binding_path, binding)

            parity_path = fixture.openset / packager.OPENSET_PARITY
            parity = json.loads(parity_path.read_text(encoding="utf-8"))
            parity["stage_two_threshold"] = 0.123
            write_json(parity_path, parity)
            fixture.refresh_openset_manifest()
            with self.assertRaisesRegex(
                packager.PackageError, "not q95 of enrollment ranks"
            ):
                fixture.package()

    def test_dual_binding_frontend_is_exact_and_policy_bound(self) -> None:
        for field, value in (
            ("patch_length", True),
            ("patch_count", 64.0),
            ("target_frac", 1),
            ("packed_length", 2048),
            ("uses_frequency_transform", 0),
        ):
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as temporary:
                    fixture = SyntheticInputs(Path(temporary))
                    path = fixture.openset / packager.DUAL_BINDING
                    binding = json.loads(path.read_text(encoding="utf-8"))
                    binding["frontend"][field] = value
                    write_json(path, binding)
                    fixture.refresh_openset_manifest()
                    with self.assertRaises(packager.PackageError):
                        fixture.package()

    def test_nested_manifest_record_types_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticInputs(Path(temporary))
            path = fixture.openset / packager.OPENSET_MANIFEST
            manifest = json.loads(path.read_text(encoding="utf-8"))
            record_value = manifest["role_exports"]["rejector"]["weights"]
            record_value["bytes"] = float(record_value["bytes"])
            write_json(path, manifest)
            with self.assertRaisesRegex(packager.PackageError, "weights.bytes"):
                fixture.package()

        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticInputs(Path(temporary))
            path = fixture.rejector / packager.FUSION_MANIFEST
            manifest = json.loads(path.read_text(encoding="utf-8"))
            record_value = manifest["emitted"][packager.REJECTOR_WEIGHTS]
            record_value["bytes"] = float(record_value["bytes"])
            write_json(path, manifest)
            with self.assertRaisesRegex(packager.PackageError, "bytes"):
                fixture.package()

    def test_parity_counts_are_derived_from_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticInputs(Path(temporary))
            path = fixture.openset / packager.OPENSET_PARITY
            parity = json.loads(path.read_text(encoding="utf-8"))
            parity["counts"]["accepted"] = 7
            parity["counts"]["classifier_executed"] = 7
            write_json(path, parity)
            fixture.refresh_openset_manifest()
            with self.assertRaisesRegex(
                packager.PackageError, "counts.accepted"
            ):
                fixture.package()

    def test_parity_rows_require_the_full_runtime_evidence_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticInputs(Path(temporary))
            path = fixture.openset / packager.OPENSET_PARITY
            parity = json.loads(path.read_text(encoding="utf-8"))
            del parity["rows"][0]["raw_features"]
            write_json(path, parity)
            fixture.refresh_openset_manifest()
            with self.assertRaisesRegex(
                packager.PackageError, "full field set"
            ):
                fixture.package()

    def test_parity_row_lengths_cover_the_exact_runtime_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticInputs(Path(temporary))
            path = fixture.openset / packager.OPENSET_PARITY
            parity = json.loads(path.read_text(encoding="utf-8"))
            parity["rows"] = [
                row
                for row in parity["rows"]
                if row["capture_length"] != 8192
            ]
            parity["counts"]["rows"] -= 1
            parity["counts"]["noise"] -= 1
            parity["counts"]["gated_stage_one"] -= 1
            write_json(path, parity)
            fixture.refresh_openset_manifest()
            with self.assertRaisesRegex(
                packager.PackageError, "capture-length coverage"
            ):
                fixture.package()

    def test_parity_decisions_are_derived_from_row_scores(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticInputs(Path(temporary))
            path = fixture.openset / packager.OPENSET_PARITY
            parity = json.loads(path.read_text(encoding="utf-8"))
            parity["rows"][0]["rejected_stage"] = 2
            write_json(path, parity)
            fixture.refresh_openset_manifest()
            with self.assertRaisesRegex(
                packager.PackageError, "rejected_stage"
            ):
                fixture.package()

    def test_parity_geometry_deviation_is_derived_from_statistic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticInputs(Path(temporary))
            path = fixture.openset / packager.OPENSET_PARITY
            parity = json.loads(path.read_text(encoding="utf-8"))
            parity["rows"][0]["stage_two"]["geometry_statistic"] = 999.0
            write_json(path, parity)
            fixture.refresh_openset_manifest()
            with self.assertRaisesRegex(
                packager.PackageError,
                "geometry_deviation from predicted-class geometry",
            ):
                fixture.package()

    def test_parity_fused_embedding_is_derived_from_branch_embeddings(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticInputs(Path(temporary))
            path = fixture.openset / packager.OPENSET_PARITY
            parity = json.loads(path.read_text(encoding="utf-8"))
            parity["rows"][0]["stage_two"]["fused_embedding"][0] += 0.1
            write_json(path, parity)
            fixture.refresh_openset_manifest()
            with self.assertRaisesRegex(
                packager.PackageError,
                "fused_embedding from branch embeddings",
            ):
                fixture.package()

    def test_parity_distances_are_derived_from_fused_embedding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticInputs(Path(temporary))
            path = fixture.openset / packager.OPENSET_PARITY
            parity = json.loads(path.read_text(encoding="utf-8"))
            distances = parity["rows"][0]["classifier"][
                "squared_prototype_distances"
            ]
            distances[1] += 0.1
            parity["rows"][0]["public_result"][
                "squaredPrototypeDistances"
            ] = list(distances)
            write_json(path, parity)
            fixture.refresh_openset_manifest()
            with self.assertRaisesRegex(
                packager.PackageError,
                "squared_prototype_distances from fused embedding",
            ):
                fixture.package()

    def test_parity_standardization_is_derived_for_each_role(self) -> None:
        mutations = (
            (
                "rejector",
                lambda row: (
                    row["rejector_standardized_features"].__setitem__(0, 0.1),
                    row["standardized_features"].__setitem__(0, 0.1),
                ),
                "rejector_standardized_features from raw features",
            ),
            (
                "classifier",
                lambda row: row[
                    "classifier_standardized_features"
                ].__setitem__(0, 0.1),
                "classifier_standardized_features from raw features",
            ),
        )
        for label, mutate, message in mutations:
            with self.subTest(role=label), tempfile.TemporaryDirectory() as temporary:
                fixture = SyntheticInputs(Path(temporary))
                path = fixture.openset / packager.OPENSET_PARITY
                parity = json.loads(path.read_text(encoding="utf-8"))
                mutate(parity["rows"][0])
                write_json(path, parity)
                fixture.refresh_openset_manifest()
                with self.assertRaisesRegex(packager.PackageError, message):
                    fixture.package()

    def test_stage_two_and_composite_thresholds_cannot_be_substituted(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticInputs(Path(temporary))
            path = fixture.openset / packager.OPENSET_POLICY
            policy = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotEqual(
                policy["stage_two"]["policy"]["threshold"],
                policy["composite"]["threshold"],
            )
            policy["composite"]["stage_two_threshold"] = policy["composite"][
                "threshold"
            ]
            write_json(path, policy)
            fixture.refresh_openset_manifest()
            with self.assertRaisesRegex(
                packager.PackageError, "stage_two_threshold"
            ):
                fixture.package()

        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticInputs(Path(temporary))
            path = fixture.openset / packager.OPENSET_POLICY
            policy = json.loads(path.read_text(encoding="utf-8"))
            policy["stage_two"]["policy"]["threshold_quantile"] = policy[
                "composite"
            ]["threshold_quantile"]
            write_json(path, policy)
            fixture.refresh_openset_manifest()
            with self.assertRaisesRegex(
                packager.PackageError, "stage two threshold_quantile"
            ):
                fixture.package()

        for field, source in (
            ("staged_threshold", ("stage_two", "policy", "threshold")),
            ("stage_two_threshold", ("composite", "threshold")),
        ):
            with self.subTest(parity_field=field):
                with tempfile.TemporaryDirectory() as temporary:
                    fixture = SyntheticInputs(Path(temporary))
                    policy = json.loads(
                        (
                            fixture.openset / packager.OPENSET_POLICY
                        ).read_text(encoding="utf-8")
                    )
                    value = policy
                    for key in source:
                        value = value[key]
                    parity_path = fixture.openset / packager.OPENSET_PARITY
                    parity = json.loads(
                        parity_path.read_text(encoding="utf-8")
                    )
                    parity[field] = value
                    write_json(parity_path, parity)
                    fixture.refresh_openset_manifest()
                    with self.assertRaisesRegex(
                        packager.PackageError, field
                    ):
                        fixture.package()

    def test_evidence_scalar_equality_aliases_fail_closed(self) -> None:
        mutations = (
            ("manifest", ("schema_version",), 1.0),
            ("policy", ("schema_version",), 4.0),
            (
                "policy",
                ("provenance", "release_seed_not_spent"),
                float(packager.RELEASE_SEED_NOT_SPENT),
            ),
            (
                "policy",
                ("provenance", "staged_validation_all_pass"),
                1,
            ),
            (
                "policy",
                ("provenance", "release_evidence"),
                0,
            ),
            (
                "policy",
                ("provenance", "sealed_release_data_used"),
                False,
            ),
            (
                "policy",
                ("provenance", "consumed_test_rows_used"),
                0.0,
            ),
            ("policy", ("contract", "additive_only"), 0),
            ("policy", ("stage_two", "policy", "threshold"), 1),
            ("policy", ("composite", "schema"), 4.0),
            (
                "policy",
                ("composite", "training_rows_used_for_threshold"),
                0.0,
            ),
            (
                "policy",
                ("composite", "release_rows_used_for_threshold"),
                False,
            ),
            ("policy", ("composite", "stage_one_changed"), 0),
            ("policy", ("composite", "enrollment_rows"), 6.0),
            (
                "policy",
                ("composite", "enrollment_capture_length"),
                16384.0,
            ),
            (
                "policy",
                ("composite", "stage_one_calibration_raw", 0),
                -3,
            ),
            ("binding", ("schema_version",), 1.0),
            (
                "binding",
                ("validation", "design_novelty_seed"),
                float(packager.DESIGN_NOVELTY_SEED),
            ),
            (
                "binding",
                ("validation", "novelty_seeds", 0),
                float(packager.VALIDATION_NOVELTY_SEEDS[0]),
            ),
            (
                "binding",
                ("validation", "release_seed_not_spent"),
                float(packager.RELEASE_SEED_NOT_SPENT),
            ),
            (
                "binding",
                ("fail_closed", "role_assets_bound_by_sha256"),
                1,
            ),
            ("parity", ("schema_version",), 4.0),
            ("parity", ("policy_schema",), 4.0),
            (
                "parity",
                ("parity_novelty_seed",),
                float(packager.PARITY_NOVELTY_SEED),
            ),
            (
                "parity",
                ("parity_seed_is_evaluation_evidence",),
                0,
            ),
            ("parity", ("fixture_lengths", 0), 4096.0),
            (
                "parity",
                (
                    "causal_prefix_parity",
                    "covers_gated_and_survivor_paths",
                ),
                1,
            ),
            (
                "parity",
                ("design_novelty_seed",),
                float(packager.DESIGN_NOVELTY_SEED),
            ),
            (
                "parity",
                ("validation_novelty_seeds", 0),
                float(packager.VALIDATION_NOVELTY_SEEDS[0]),
            ),
            (
                "parity",
                ("release_seed_not_spent",),
                float(packager.RELEASE_SEED_NOT_SPENT),
            ),
            ("parity", ("counts", "accepted"), 1.0),
            ("parity", ("counts", "classifier_executed"), True),
            ("parity", ("rows", 0, "classifier_executed"), 1),
            ("parity", ("rows", 1, "rejected_stage"), True),
        )
        filenames = {
            "manifest": packager.OPENSET_MANIFEST,
            "policy": packager.OPENSET_POLICY,
            "binding": packager.DUAL_BINDING,
            "parity": packager.OPENSET_PARITY,
        }

        for target, keys, replacement in mutations:
            with self.subTest(target=target, keys=keys):
                with tempfile.TemporaryDirectory() as temporary:
                    fixture = SyntheticInputs(Path(temporary))
                    path = fixture.openset / filenames[target]
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    parent = payload
                    for key in keys[:-1]:
                        parent = parent[key]
                    parent[keys[-1]] = replacement
                    write_json(path, payload)
                    if target != "manifest":
                        fixture.refresh_openset_manifest()
                    with self.assertRaises(packager.PackageError):
                        fixture.package()

    def test_oversized_deployable_asset_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticInputs(Path(temporary))
            policy_path = fixture.openset / packager.OPENSET_POLICY
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
            policy["padding"] = "x" * packager.MAX_DEPLOYABLE_BYTES
            write_json(policy_path, policy)
            fixture.refresh_openset_manifest()
            with self.assertRaisesRegex(packager.PackageError, "25 MiB"):
                fixture.package()
            self.assertFalse(fixture.destination.exists())

    def test_nonempty_destination_is_not_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticInputs(Path(temporary))
            fixture.destination.mkdir()
            sentinel = fixture.destination / "mine.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(packager.PackageError, "empty"):
                fixture.package()
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
