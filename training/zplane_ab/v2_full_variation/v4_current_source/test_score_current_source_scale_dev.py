from __future__ import annotations

from argparse import Namespace
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import score_current_source_scale_dev as scorer


def _read(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _fake_intent() -> dict:
    separation = {"separated": True}
    return {
        "protocol": {"sha256": "1" * 64},
        "scale_development_corpus": {
            "manifest_sha256": "2" * 64,
            "raw_sha256": "3" * 64,
            "generator": {
                "source": {"sha256": "9" * 64},
                "bundle": {"sha256": "a" * 64},
            },
        },
        "candidate": {
            "classifier_sha256": "4" * 64,
            "policy_sha256": "5" * 64,
            "fusion_dev_metrics": {"sha256": "b" * 64},
            "fusion_artifacts": {
                "files_sha256": {"state.pt": "c" * 64}
            },
        },
        "executed_source_sha256": {"source.py": "6" * 64},
        "scorer_source": {
            "relative_path":
                "training/zplane_ab/v2_full_variation/v4_current_source/"
                "score_current_source_scale_dev.py",
            "sha256": "7" * 64,
        },
        "reference_separation": {
            "report": separation,
            "sha256": hashlib.sha256(
                scorer._json_bytes(separation)
            ).hexdigest(),
        },
    }


def _fake_score_paths(root: Path) -> dict[str, Path]:
    return {
        "protocol": root / "protocol.json",
        "scale_corpus": root / "scale",
        "current_corpus": root / "current",
        "fusion": root / "fusion",
        "historical_corpus": root / "historical",
        "classifier_asset": root / "classifier.json",
        "policy": root / "policy.json",
        "scorer": root / "scorer.py",
        "ledger": root / scorer.LEDGER_RELATIVE_PATH,
        "output": root / scorer.REPORT_OUTPUT_RELATIVE_PATH,
    }


def _valid_invariance_known() -> dict:
    profiles = [
        f"current:{profile}"
        for profile in scorer.openset.corpus_data.CURRENT_PROFILES
    ]
    lengths = tuple(scorer.openset.RUNTIME_INPUT_LENGTHS)
    scales = tuple(float(value) for value in scorer.openset.scale_eval.SCALE_FACTORS)
    physical_expected = {}
    physical_cells = []
    causal_expected = {}
    physical_by_length = {}
    causal_by_profile = {}
    combined_by_profile = {}
    observed_count = 0
    for profile in profiles:
        for pair_index in range(24):
            pair_id = f"{profile}|pair={pair_index}"
            for length in lengths:
                legal_scales = (
                    (1.5, 2.0)
                    if (
                        profile == "current:bluetooth-le-advertising"
                        and length == 16_384
                    )
                    else scales
                )
                key = f"{pair_id}|N={length}"
                physical_expected[key] = list(legal_scales)
                observed_count += len(legal_scales)
                physical_cells.append(
                    {
                        "pair_id": pair_id,
                        "source_profile": profile,
                        "observation_length": length,
                        "expected_legal_scale_factors": list(legal_scales),
                        "observed_legal_scale_factors": list(legal_scales),
                        "exact_legal_scale_coverage": True,
                        "distinct_legal_scale_count": len(legal_scales),
                    }
                )
            for scale in scales:
                legal_lengths = (
                    [4_096, 8_192]
                    if (
                        profile == "current:bluetooth-le-advertising"
                        and scale in (1.0, 1.25)
                    )
                    else list(lengths)
                )
                causal_expected[
                    f"{pair_id}|scale={scale:g}"
                ] = legal_lengths
        causal_by_profile[profile] = {
            "expected_pair_scale_cells": 96,
            "observed_pair_scale_cells": 96,
            "exact_cell_coverage": True,
            "minimum_distinct_observation_lengths": 2,
            "prediction_or_abstention_agreement_rate": 1.0,
        }
        combined_by_profile[profile] = {
            "expected_pair_cells": 24,
            "cells": 24,
            "complete_pair_coverage_rate": 1.0,
            "minimum_distinct_runtime_lengths": 2,
            "prediction_or_abstention_agreement_rate": 1.0,
        }
    for length in lengths:
        physical_by_length[str(length)] = {
            "expected_cells": 744,
            "observed_cells": 744,
            "minimum_distinct_legal_scales": 2,
            "prediction_or_abstention_agreement_rate": 1.0,
        }
    self_cell = {
        "source": "development_service_scale_seed20262903",
        "length": 4_096,
        "name": "dsss",
        "rows": 24,
        "closed_head_accuracy": 1.0,
        "closed_head_correct_rows": 24,
        "rejected_closed_head_correct_rows": 0,
        "abstention_damage_given_closed_head_correct": 0.0,
    }
    return {
        "combined": {
            "pooled_abstention_damage_given_closed_head_correct": 0.0,
            "pooled_closed_head_correct_rows": 8_000,
            "per_length_abstention_damage_given_closed_head_correct": {
                str(length): 0.0 for length in lengths
            },
            "per_length_closed_head_correct_rows": {
                str(length): 2_000 for length in lengths
            },
        },
        "gate_cells": {
            "true_class": [dict(self_cell)],
            "source_profile": [
                {**self_cell, "name": "current:wifi-hr-dsss-11m"}
            ],
        },
        "paired_physical_scale_invariance": {
            "expected_cells": 2_232,
            "cells": 2_232,
            "expected_observation_cells": 8_880,
            "observed_observation_cells": 8_880,
            "expected_legal_scales_by_pair_observation_length":
                physical_expected,
            "observed_legal_scales_by_pair_observation_length":
                deepcopy(physical_expected),
            "exact_eligible_cell_coverage": True,
            "minimum_distinct_legal_scales": 2,
            "prediction_or_abstention_agreement_rate": 1.0,
            "by_observation_length": physical_by_length,
            "pair_observation_length_cells": physical_cells,
        },
        "causal_prefix_length_invariance": {
            "expected_cells": 2_976,
            "cells": 2_976,
            "expected_observation_cells": observed_count,
            "observed_observation_cells": observed_count,
            "expected_observation_lengths_by_pair_scale": causal_expected,
            "observed_observation_lengths_by_pair_scale":
                deepcopy(causal_expected),
            "exact_eligible_cell_coverage": True,
            "minimum_distinct_observation_lengths": 2,
            "prediction_or_abstention_agreement_rate": 1.0,
            "by_source_profile": causal_by_profile,
        },
        "paired_length_scale_invariance": {
            "expected_pair_cells": 744,
            "cells": 744,
            "expected_observation_cells": observed_count,
            "observed_observation_cells": observed_count,
            "complete_pair_coverage_rate": 1.0,
            "distinct_runtime_lengths_minimum": 2,
            "prediction_or_abstention_agreement_rate": 1.0,
            "by_source_profile": combined_by_profile,
        },
    }


def _shared_classifier_gates() -> dict:
    return {
        "classifier_held_current_service_coverage": {
            "value": True,
            "expected_profiles": 31,
            "observed_profiles": 31,
            "expected_global_length_scale_cells": 12,
            "observed_length_scale_cells": 12,
            "expected_profile_legal_cells": 370,
            "observed_profile_legal_cells": 370,
            "expected_profile_legal_observations": 8_880,
            "observed_profile_legal_observations": 8_880,
            "exact_profile_legal_cell_key_and_count_coverage": True,
            "passes": True,
        },
        **{
            f"classifier_held_current_service_metric_{index}": {
                "value": 1.0,
                "passes": True,
            }
            for index in range(7)
        },
    }


def _write_bound_file(root: Path, relative: str, payload: bytes) -> dict:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "relative_path": relative,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _filesystem_bound_intent(root: Path) -> tuple[dict, dict[str, Path]]:
    protocol = _write_bound_file(root, "protocol.json", b"protocol")
    scale_manifest = _write_bound_file(
        root, "scale/scale_eval.json", b"scale-manifest"
    )
    scale_raw = _write_bound_file(
        root, "scale/scale_eval.f32", b"scale-raw"
    )
    generator_source = _write_bound_file(
        root, "tools/generator.ts", b"generator-source"
    )
    generator_bundle = _write_bound_file(
        root, ".artifacts/generator.js", b"generator-bundle"
    )
    current_manifest = _write_bound_file(
        root, "current/corpus.json", b"current-manifest"
    )
    current_raw = _write_bound_file(
        root, "current/corpus.f32", b"current-raw"
    )
    historical_manifest = _write_bound_file(
        root, "historical/corpus.json", b"historical-manifest"
    )
    historical_raw = _write_bound_file(
        root, "historical/corpus.f32", b"historical-raw"
    )
    metrics = _write_bound_file(
        root, "fusion/dev_metrics.json", b"fusion-metrics"
    )
    artifact = _write_bound_file(
        root, "fusion/fusion_state_dict.pt", b"fusion-artifact"
    )
    classifier = _write_bound_file(
        root, "classifier.json", b"classifier"
    )
    policy = _write_bound_file(root, "policy.json", b"policy")
    scorer_source = _write_bound_file(root, "scorer.py", b"scorer")
    artifact_files = {
        "fusion_state_dict.pt": artifact["sha256"],
    }
    intent = {
        "protocol": protocol,
        "scale_development_corpus": {
            "relative_path": "scale",
            "manifest_relative_path": scale_manifest["relative_path"],
            "manifest_sha256": scale_manifest["sha256"],
            "raw_relative_path": scale_raw["relative_path"],
            "raw_sha256": scale_raw["sha256"],
            "generator": {
                "source": generator_source,
                "bundle": generator_bundle,
            },
        },
        "current_reference_corpus": {
            "relative_path": "current",
            "manifest_relative_path": current_manifest["relative_path"],
            "manifest_sha256": current_manifest["sha256"],
            "raw_relative_path": current_raw["relative_path"],
            "raw_sha256": current_raw["sha256"],
        },
        "candidate": {
            "fusion_relative_path": "fusion",
            "fusion_classifier_binding": {
                "source_fusion_dev_metrics_sha256": metrics["sha256"],
                "source_fusion_artifacts_sha256": artifact_files,
            },
            "fusion_dev_metrics": metrics,
            "fusion_artifacts": {
                "directory_relative_path": "fusion",
                "files_sha256": artifact_files,
            },
            "historical_corpus": {
                "relative_path": "historical",
                "manifest_relative_path":
                    historical_manifest["relative_path"],
                "manifest_sha256": historical_manifest["sha256"],
                "raw_relative_path": historical_raw["relative_path"],
                "raw_sha256": historical_raw["sha256"],
            },
            "classifier_relative_path": classifier["relative_path"],
            "classifier_sha256": classifier["sha256"],
            "policy_relative_path": policy["relative_path"],
            "policy_sha256": policy["sha256"],
        },
        "scorer_source": scorer_source,
        "executed_source_sha256": {"source.py": "a" * 64},
        "score_ledger_relative_path": scorer.LEDGER_RELATIVE_PATH,
        "report_output_relative_path": scorer.REPORT_OUTPUT_RELATIVE_PATH,
    }
    paths = {
        "classifier": root / classifier["relative_path"],
        "fusion_artifact": root / "fusion/fusion_state_dict.pt",
        "generator_source": root / generator_source["relative_path"],
    }
    return intent, paths


class DurableEvidenceTests(unittest.TestCase):
    def test_exclusive_json_is_single_use_and_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "intent.json"
            payload = {"schema": "test", "value": 1}
            digest = scorer._exclusive_durable_json(path, payload)
            self.assertEqual(
                digest,
                hashlib.sha256(scorer._json_bytes(payload)).hexdigest(),
            )
            self.assertEqual(_read(path), payload)
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                scorer._exclusive_durable_json(path, payload)
            self.assertEqual(_read(path), payload)

    def test_exclusive_json_rejects_broken_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "ledger.json"
            path.symlink_to(root / "missing")
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                scorer._exclusive_durable_json(path, {"state": "claimed"})
            self.assertTrue(path.is_symlink())

    def test_atomic_publication_never_replaces_file_or_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "report.json"
            first = {"all_pass": False}
            scorer._atomic_durable_json(path, first)
            with self.assertRaisesRegex(RuntimeError, "replace"):
                scorer._atomic_durable_json(path, {"all_pass": True})
            self.assertEqual(_read(path), first)
            link = root / "manifest.json"
            link.symlink_to(root / "missing")
            with self.assertRaisesRegex(RuntimeError, "replace"):
                scorer._atomic_durable_json(link, {"schema": "x"})
            self.assertTrue(link.is_symlink())

    def test_atomic_publication_rejects_raced_temporary_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "report.json"
            temporary_path = root / f".report.json.tmp-{os.getpid()}"
            temporary_path.symlink_to(root / "missing")
            with self.assertRaises(FileExistsError):
                scorer._atomic_durable_json(path, {"all_pass": True})
            self.assertFalse(os.path.lexists(path))


class ProtocolAndBindingTests(unittest.TestCase):
    def test_real_protocol_and_generation_bytes_reproduce_without_scoring(
        self,
    ) -> None:
        path, protocol = scorer._validate_protocol(
            scorer.REPO / scorer.PROTOCOL_RELATIVE_PATH
        )
        self.assertEqual(scorer._sha256(path), scorer.PROTOCOL_SHA256)
        scale, current, _corpus = scorer._validate_generation_inputs(
            protocol,
            scale_corpus=(
                scorer.REPO
                / protocol["generation"]["output_relative_path"]
            ),
            current_corpus=(
                scorer.REPO
                / protocol["generation"]["reference_corpus_relative_path"]
            ),
        )
        self.assertEqual(
            scale["manifest_sha256"],
            scorer.EXPECTED_SCALE_MANIFEST_SHA256,
        )
        self.assertEqual(
            scale["raw_sha256"], scorer.EXPECTED_SCALE_RAW_SHA256
        )
        self.assertEqual(
            current["manifest_sha256"],
            protocol["generation"]["reference_manifest_sha256"],
        )

    def test_protocol_hash_and_gate_contract_are_fail_closed(self) -> None:
        with mock.patch.object(scorer, "PROTOCOL_SHA256", "0" * 64):
            with self.assertRaisesRegex(ValueError, "SHA-256 changed"):
                scorer._validate_protocol(
                    scorer.REPO / scorer.PROTOCOL_RELATIVE_PATH
                )

    def test_scorer_is_not_in_frozen_policy_source_inventory(self) -> None:
        self.assertNotIn(
            scorer._relative(Path(scorer.__file__)),
            scorer.openset.EXECUTED_SOURCE_RELATIVE_PATHS,
        )
        policy = _read(
            scorer.REPO
            / ".artifacts/v4-openset-schema5-final-independent-design-"
            "seed20263001/time-domain-profile-bank-openset-v4.json"
        )
        self.assertEqual(
            policy["provenance"]["source_sha256"],
            scorer.openset._source_hashes(),
        )

    def test_real_command_defaults_are_the_frozen_paths(self) -> None:
        prepare = scorer.build_parser().parse_args(["prepare-intent"])
        self.assertEqual(
            Path(prepare.scale_corpus),
            scorer.REPO / scorer.SCALE_CORPUS_RELATIVE_PATH,
        )
        self.assertEqual(
            Path(prepare.fusion),
            scorer.REPO / scorer.FUSION_RELATIVE_PATH,
        )
        self.assertEqual(
            Path(prepare.historical_corpus),
            scorer.REPO / scorer.HISTORICAL_CORPUS_RELATIVE_PATH,
        )
        self.assertEqual(
            Path(prepare.current_corpus),
            scorer.REPO / scorer.CURRENT_CORPUS_RELATIVE_PATH,
        )
        self.assertEqual(
            Path(prepare.classifier_asset),
            scorer.REPO / scorer.CLASSIFIER_RELATIVE_PATH,
        )
        self.assertEqual(
            Path(prepare.policy),
            scorer.REPO / scorer.POLICY_RELATIVE_PATH,
        )
        self.assertEqual(
            Path(prepare.output_dir),
            scorer.REPO / scorer.REPORT_OUTPUT_RELATIVE_PATH,
        )
        score = scorer.build_parser().parse_args(["score"])
        self.assertEqual(
            Path(score.intent),
            scorer.REPO / scorer.INTENT_RELATIVE_PATH,
        )

    def test_dev_adapter_has_only_exact_evaluator_binding(self) -> None:
        intent = {
            "scale_development_corpus": {
                "manifest_sha256": "a" * 64,
                "raw_sha256": "b" * 64,
            }
        }
        protocol = {
            "generation": {
                "eval_seed": 20_262_903,
                "realizations_per_profile": 24,
            }
        }
        value = scorer._development_evaluator_protocol(intent, protocol)
        self.assertEqual(
            set(value),
            {
                "held_scale_manifest_sha256",
                "held_scale_raw_sha256",
                "held_scale_eval_seed",
                "held_scale_realizations_per_profile",
                "held_scale_corpus_binding_sha256",
            },
        )
        self.assertEqual(value["held_scale_eval_seed"], 20_262_903)
        self.assertNotIn("novelty_seeds", value)
        self.assertNotIn("validation_ledger_relative_path", value)

    def test_repo_path_rejects_leaf_and_parent_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            regular = root / "regular"
            regular.write_text("x", encoding="utf-8")
            leaf = root / "leaf"
            leaf.symlink_to(regular)
            parent_target = root / "target"
            parent_target.mkdir()
            parent = root / "parent"
            parent.symlink_to(parent_target, target_is_directory=True)
            with mock.patch.object(scorer, "REPO", root):
                with self.assertRaisesRegex(ValueError, "symlink"):
                    scorer._repo_path(leaf, role="leaf")
                with self.assertRaisesRegex(ValueError, "symlink"):
                    scorer._repo_path(
                        parent / "child",
                        role="parent",
                        must_exist=False,
                    )

    def test_malformed_nested_historical_binding_is_value_error(self) -> None:
        intent = {
            "protocol": {},
            "scale_development_corpus": {},
            "current_reference_corpus": {},
            "candidate": {"historical_corpus": []},
            "scorer_source": {},
        }
        with self.assertRaisesRegex(ValueError, "historical corpus"):
            scorer._intent_paths(intent)

    def test_alternate_historical_path_is_rejected_before_corpus_read(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            fusion = root / scorer.FUSION_RELATIVE_PATH
            alternate = root / "alternate-history"
            fusion.mkdir(parents=True)
            alternate.mkdir()
            with (
                mock.patch.object(scorer, "REPO", root),
                mock.patch.object(
                    scorer, "_corpus_file_binding"
                ) as corpus_read,
            ):
                with self.assertRaisesRegex(
                    ValueError, "historical corpus path"
                ):
                    scorer._candidate_binding(
                        fusion=fusion,
                        historical_corpus=alternate,
                        current_corpus=root / "unused-current",
                        classifier_asset=root / "unused-classifier",
                        policy_path=root / "unused-policy",
                        device="cpu",
                    )
            corpus_read.assert_not_called()

    def test_real_bound_input_assertion_uses_classifier_asset_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            intent, _paths = _filesystem_bound_intent(root)
            with (
                mock.patch.object(scorer, "REPO", root),
                mock.patch.object(
                    scorer.openset,
                    "_source_hashes",
                    return_value=intent["executed_source_sha256"],
                ),
            ):
                scorer._assert_bound_inputs_unchanged(intent)

    def test_bound_input_assertion_detects_fusion_and_generator_tamper(
        self,
    ) -> None:
        for target_name in ("fusion_artifact", "generator_source"):
            with self.subTest(target_name=target_name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                intent, paths = _filesystem_bound_intent(root)
                with (
                    mock.patch.object(scorer, "REPO", root),
                    mock.patch.object(
                        scorer.openset,
                        "_source_hashes",
                        return_value=intent["executed_source_sha256"],
                    ),
                ):
                    scorer._assert_bound_inputs_unchanged(intent)
                    paths[target_name].write_bytes(b"tampered")
                    with self.assertRaisesRegex(
                        ValueError, "SHA-256 changed"
                    ):
                        scorer._assert_bound_inputs_unchanged(intent)


class GateAdversarialTests(unittest.TestCase):
    def test_valid_exact_invariance_inventories_pass(self) -> None:
        known = _valid_invariance_known()
        physical, physical_exact = scorer._physical_scale_gates(known)
        causal, causal_exact = scorer._causal_prefix_gates(known)
        combined, combined_exact = scorer._length_scale_gates(known)
        self.assertTrue(physical_exact)
        self.assertTrue(causal_exact)
        self.assertTrue(combined_exact)
        self.assertTrue(all(row["passes"] for row in physical.values()))
        self.assertTrue(all(row["passes"] for row in causal.values()))
        self.assertTrue(all(row["passes"] for row in combined.values()))

    def test_physical_pooled_and_one_length_metric_misses_are_gates(self) -> None:
        known = _valid_invariance_known()
        known["paired_physical_scale_invariance"][
            "prediction_or_abstention_agreement_rate"
        ] = 0.949
        gates, exact = scorer._physical_scale_gates(known)
        self.assertTrue(exact)
        self.assertFalse(
            gates["physical_scale_outcome_agreement_pooled"]["passes"]
        )
        known = _valid_invariance_known()
        known["paired_physical_scale_invariance"][
            "by_observation_length"
        ]["8192"]["prediction_or_abstention_agreement_rate"] = 0.949
        gates, exact = scorer._physical_scale_gates(known)
        self.assertTrue(exact)
        self.assertFalse(
            gates[
                "physical_scale_outcome_agreement_every_observation_length"
            ]["passes"]
        )

    def test_physical_missing_cell_and_illegal_ble_scales_fail_coverage(
        self,
    ) -> None:
        known = _valid_invariance_known()
        known["paired_physical_scale_invariance"][
            "pair_observation_length_cells"
        ].pop()
        _gates, exact = scorer._physical_scale_gates(known)
        self.assertFalse(exact)
        known = _valid_invariance_known()
        cell = next(
            row
            for row in known["paired_physical_scale_invariance"][
                "pair_observation_length_cells"
            ]
            if (
                row["source_profile"]
                == "current:bluetooth-le-advertising"
                and row["observation_length"] == 16_384
            )
        )
        cell["expected_legal_scale_factors"] = [1.0, 1.25, 1.5, 2.0]
        cell["observed_legal_scale_factors"] = [1.0, 1.25, 1.5, 2.0]
        cell["distinct_legal_scale_count"] = 4
        _gates, exact = scorer._physical_scale_gates(known)
        self.assertFalse(exact)

    def test_causal_profile_omission_and_below_one_fail(self) -> None:
        known = _valid_invariance_known()
        profile = next(
            iter(
                known["causal_prefix_length_invariance"][
                    "by_source_profile"
                ]
            )
        )
        del known["causal_prefix_length_invariance"][
            "by_source_profile"
        ][profile]
        gates, exact = scorer._causal_prefix_gates(known)
        self.assertFalse(exact)
        self.assertFalse(
            gates[
                "causal_prefix_outcome_agreement_every_source_profile"
            ]["passes"]
        )
        known = _valid_invariance_known()
        known["causal_prefix_length_invariance"][
            "prediction_or_abstention_agreement_rate"
        ] = 0.999
        gates, exact = scorer._causal_prefix_gates(known)
        self.assertTrue(exact)
        self.assertFalse(
            gates[
                "causal_prefix_outcome_agreement_across_observation_lengths"
            ]["passes"]
        )
        known = _valid_invariance_known()
        profile = next(
            iter(
                known["causal_prefix_length_invariance"][
                    "by_source_profile"
                ]
            )
        )
        known["causal_prefix_length_invariance"][
            "by_source_profile"
        ][profile]["prediction_or_abstention_agreement_rate"] = 0.999
        gates, exact = scorer._causal_prefix_gates(known)
        self.assertTrue(exact)
        self.assertFalse(
            gates[
                "causal_prefix_outcome_agreement_every_source_profile"
            ]["passes"]
        )

    def test_combined_profile_omission_and_below_floor_fail(self) -> None:
        known = _valid_invariance_known()
        profile = next(
            iter(
                known["paired_length_scale_invariance"][
                    "by_source_profile"
                ]
            )
        )
        del known["paired_length_scale_invariance"][
            "by_source_profile"
        ][profile]
        gates, exact = scorer._length_scale_gates(known)
        self.assertFalse(exact)
        self.assertFalse(
            gates[
                "length_scale_outcome_agreement_every_source_profile"
            ]["passes"]
        )
        known = _valid_invariance_known()
        known["paired_length_scale_invariance"][
            "prediction_or_abstention_agreement_rate"
        ] = 0.949
        gates, exact = scorer._length_scale_gates(known)
        self.assertTrue(exact)
        self.assertFalse(
            gates["length_scale_outcome_agreement_pooled"]["passes"]
        )
        known = _valid_invariance_known()
        profile = next(
            iter(
                known["paired_length_scale_invariance"][
                    "by_source_profile"
                ]
            )
        )
        known["paired_length_scale_invariance"][
            "by_source_profile"
        ][profile]["prediction_or_abstention_agreement_rate"] = 0.949
        gates, exact = scorer._length_scale_gates(known)
        self.assertTrue(exact)
        self.assertFalse(
            gates[
                "length_scale_outcome_agreement_every_source_profile"
            ]["passes"]
        )

    def test_abstention_pooled_length_class_and_profile_fail_independently(
        self,
    ) -> None:
        known = _valid_invariance_known()
        known["combined"][
            "pooled_abstention_damage_given_closed_head_correct"
        ] = 0.121
        gates = scorer._conditional_abstention_gates(known)
        self.assertFalse(
            gates[
                "known_abstention_damage_given_closed_head_correct_pooled"
            ]["passes"]
        )

        known = _valid_invariance_known()
        known["combined"][
            "per_length_abstention_damage_given_closed_head_correct"
        ]["8192"] = 0.121
        gates = scorer._conditional_abstention_gates(known)
        self.assertFalse(
            gates[
                "known_abstention_damage_given_closed_head_correct_worst_"
                "observation_length"
            ]["passes"]
        )

        known = _valid_invariance_known()
        class_cell = known["gate_cells"]["true_class"][0]
        class_cell["closed_head_correct_rows"] = 20
        class_cell["rejected_closed_head_correct_rows"] = 3
        class_cell["abstention_damage_given_closed_head_correct"] = 0.15
        class_cell["closed_head_accuracy"] = 20 / 24
        gates = scorer._conditional_abstention_gates(known)
        self.assertFalse(
            gates[
                "known_abstention_damage_given_closed_head_correct_worst_"
                "supported_true_class_cell"
            ]["passes"]
        )

        known = _valid_invariance_known()
        profile_cell = known["gate_cells"]["source_profile"][0]
        profile_cell["closed_head_correct_rows"] = 20
        profile_cell["rejected_closed_head_correct_rows"] = 3
        profile_cell["abstention_damage_given_closed_head_correct"] = 0.15
        profile_cell["closed_head_accuracy"] = 20 / 24
        gates = scorer._conditional_abstention_gates(known)
        self.assertFalse(
            gates[
                "known_abstention_damage_given_closed_head_correct_worst_"
                "supported_source_profile_cell"
            ]["passes"]
        )

    def test_classifier_threshold_family_and_profile_coverage_fail_closed(
        self,
    ) -> None:
        for source_name in [
            f"classifier_held_current_service_metric_{index}"
            for index in range(7)
        ]:
            shared = _shared_classifier_gates()
            shared[source_name]["passes"] = False
            shared[source_name]["value"] = 0.0
            with self.subTest(source_name=source_name), mock.patch.object(
                scorer.openset,
                "_classifier_gate_summary",
                return_value=(
                    "independent_held_current_service",
                    shared,
                ),
            ):
                summary = scorer.build_development_gate_summary(
                    _valid_invariance_known()
                )
                renamed = source_name.replace(
                    "classifier_held_current_service_",
                    "classifier_development_current_service_",
                )
                self.assertFalse(summary["gates"][renamed]["passes"])
                self.assertFalse(summary["all_pass"])

        shared = _shared_classifier_gates()
        shared["classifier_held_current_service_coverage"][
            "observed_profile_legal_cells"
        ] = 369
        with mock.patch.object(
            scorer.openset,
            "_classifier_gate_summary",
            return_value=("independent_held_current_service", shared),
        ):
            summary = scorer.build_development_gate_summary(
                _valid_invariance_known()
            )
        self.assertFalse(
            summary["gates"][
                "exact_profile_length_scale_key_and_count_coverage"
            ]["passes"]
        )
        self.assertFalse(summary["all_pass"])

    def test_gate_inventory_is_exactly_19_and_contains_no_novelty(self) -> None:
        known = _valid_invariance_known()
        shared = _shared_classifier_gates()
        with mock.patch.object(
            scorer.openset,
            "_classifier_gate_summary",
            return_value=("independent_held_current_service", shared),
        ):
            summary = scorer.build_development_gate_summary(known)
        self.assertEqual(len(summary["gates"]), 19)
        self.assertTrue(summary["all_pass"])
        self.assertFalse(summary["novelty_gates_present"])
        self.assertFalse(
            any("novelty" in name for name in summary["gates"])
        )


class ScoreOrderingTests(unittest.TestCase):
    def _run_patched_score(
        self,
        root: Path,
        *,
        all_pass: bool,
        evaluator_side_effect=None,
        assert_unchanged_side_effect=None,
    ):
        intent_path = root / "intent.json"
        intent_path.write_text("{}\n", encoding="utf-8")
        paths = _fake_score_paths(root)
        intent = _fake_intent()
        gate_summary = {
            "development_diagnostics_only": True,
            "independent_validation_evidence": False,
            "release_evidence": False,
            "novelty_gates_present": False,
            "gates": {
                "metric": {
                    "value": 0.0,
                    "floor": 1.0,
                    "passes": all_pass,
                }
            },
            "all_pass": all_pass,
        }

        def evaluate(*_args, **_kwargs):
            self.assertTrue(paths["ledger"].is_file())
            self.assertEqual(
                _read(paths["ledger"])["state"],
                "development_scale_score_claimed",
            )
            if callable(evaluator_side_effect):
                evaluator_side_effect()
            elif evaluator_side_effect is not None:
                raise evaluator_side_effect
            return (
                {
                    "population_role": "underlying evaluator role",
                    "reference_separation": {},
                    "per_length_scale": {},
                    "gate_cells": {},
                },
                {},
            )

        patches = (
            mock.patch.object(scorer, "REPO", root),
            mock.patch.object(
                scorer, "INTENT_RELATIVE_PATH", "intent.json"
            ),
            mock.patch.object(
                scorer,
                "_validate_intent",
                return_value=(
                    intent,
                    "8" * 64,
                    paths,
                    object(),
                    {},
                    {"generation": {}},
                ),
            ),
            mock.patch.object(
                scorer,
                "_assert_bound_inputs_unchanged",
                side_effect=assert_unchanged_side_effect,
            ),
            mock.patch.object(
                scorer.openset,
                "evaluate_held_scale_known",
                side_effect=evaluate,
            ),
            mock.patch.object(
                scorer,
                "_development_evaluator_protocol",
                return_value={},
            ),
            mock.patch.object(
                scorer,
                "_canonical_reference_separation",
                return_value=intent["reference_separation"]["report"],
            ),
            mock.patch.object(
                scorer,
                "build_development_gate_summary",
                return_value=gate_summary,
            ),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4] as evaluator_mock, patches[5], patches[6], patches[7]:
            result = scorer.score_once(Namespace(intent=str(intent_path)))
        return result, paths, evaluator_mock

    def test_ledger_exists_before_first_inference_and_metric_miss_publishes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            report, paths, evaluator = self._run_patched_score(
                root, all_pass=False
            )
            evaluator.assert_called_once()
            self.assertFalse(report["gate_summary"]["all_pass"])
            published = _read(paths["output"] / scorer.REPORT_NAME)
            manifest = _read(paths["output"] / scorer.MANIFEST_NAME)
            self.assertFalse(published["gate_summary"]["all_pass"])
            self.assertFalse(manifest["all_pass"])
            self.assertFalse(manifest["release_evidence"])
            self.assertTrue(paths["ledger"].is_file())

    def test_existing_ledger_blocks_before_context_validation_or_inference(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            intent = root / "intent.json"
            intent.write_text("{}\n", encoding="utf-8")
            ledger = root / scorer.LEDGER_RELATIVE_PATH
            ledger.parent.mkdir(parents=True)
            ledger.write_text("{}\n", encoding="utf-8")
            with (
                mock.patch.object(scorer, "REPO", root),
                mock.patch.object(
                    scorer, "INTENT_RELATIVE_PATH", "intent.json"
                ),
                mock.patch.object(scorer, "_validate_intent") as validate,
                mock.patch.object(
                    scorer.openset, "evaluate_held_scale_known"
                ) as evaluate,
            ):
                with self.assertRaisesRegex(FileExistsError, "ledger"):
                    scorer.score_once(Namespace(intent=str(intent)))
            validate.assert_not_called()
            evaluate.assert_not_called()

    def test_broken_symlink_at_fixed_ledger_name_blocks_immediately(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            intent = root / "intent.json"
            intent.write_text("{}\n", encoding="utf-8")
            ledger = root / scorer.LEDGER_RELATIVE_PATH
            ledger.parent.mkdir(parents=True)
            ledger.symlink_to(root / "missing-ledger-target")
            with (
                mock.patch.object(scorer, "REPO", root),
                mock.patch.object(
                    scorer, "INTENT_RELATIVE_PATH", "intent.json"
                ),
                mock.patch.object(scorer, "_validate_intent") as validate,
                mock.patch.object(
                    scorer.openset, "evaluate_held_scale_known"
                ) as evaluate,
            ):
                with self.assertRaisesRegex(FileExistsError, "ledger"):
                    scorer.score_once(Namespace(intent=str(intent)))
            validate.assert_not_called()
            evaluate.assert_not_called()
            self.assertTrue(ledger.is_symlink())

    def test_preexisting_output_blocks_before_context_or_inference(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            intent = root / "intent.json"
            intent.write_text("{}\n", encoding="utf-8")
            output = root / scorer.REPORT_OUTPUT_RELATIVE_PATH
            output.mkdir(parents=True)
            with (
                mock.patch.object(scorer, "REPO", root),
                mock.patch.object(
                    scorer, "INTENT_RELATIVE_PATH", "intent.json"
                ),
                mock.patch.object(scorer, "_validate_intent") as validate,
                mock.patch.object(
                    scorer.openset, "evaluate_held_scale_known"
                ) as evaluate,
            ):
                with self.assertRaisesRegex(FileExistsError, "output"):
                    scorer.score_once(Namespace(intent=str(intent)))
            validate.assert_not_called()
            evaluate.assert_not_called()

    def test_inference_failure_retains_consumed_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                self._run_patched_score(
                    root,
                    all_pass=True,
                    evaluator_side_effect=RuntimeError("inference failed"),
                )
            paths = _fake_score_paths(root)
            self.assertTrue(paths["ledger"].is_file())
            self.assertFalse(os.path.lexists(paths["output"]))

    def test_post_score_source_tamper_retains_ledger_and_blocks_report(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with self.assertRaisesRegex(RuntimeError, "source changed"):
                self._run_patched_score(
                    root,
                    all_pass=True,
                    assert_unchanged_side_effect=[
                        None,
                        RuntimeError("source changed"),
                    ],
                )
            paths = _fake_score_paths(root)
            self.assertTrue(paths["ledger"].is_file())
            self.assertFalse(os.path.lexists(paths["output"]))

    def test_post_score_fusion_artifact_tamper_retains_ledger_and_blocks_report(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            artifact = root / "fusion-state.pt"
            artifact.write_bytes(b"bound")
            expected = hashlib.sha256(b"bound").hexdigest()

            def tamper() -> None:
                artifact.write_bytes(b"tampered")

            def assert_fusion_unchanged(_intent) -> None:
                if hashlib.sha256(artifact.read_bytes()).hexdigest() != expected:
                    raise RuntimeError("fusion artifact changed")

            with self.assertRaisesRegex(
                RuntimeError, "fusion artifact changed"
            ):
                self._run_patched_score(
                    root,
                    all_pass=True,
                    evaluator_side_effect=tamper,
                    assert_unchanged_side_effect=assert_fusion_unchanged,
                )
            paths = _fake_score_paths(root)
            self.assertTrue(paths["ledger"].is_file())
            self.assertFalse(os.path.lexists(paths["output"]))

    def test_intent_validation_failure_never_claims_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            intent = root / "intent.json"
            intent.write_text("{}\n", encoding="utf-8")
            with (
                mock.patch.object(scorer, "REPO", root),
                mock.patch.object(
                    scorer, "INTENT_RELATIVE_PATH", "intent.json"
                ),
                mock.patch.object(
                    scorer,
                    "_validate_intent",
                    side_effect=ValueError("tampered intent"),
                ),
                mock.patch.object(
                    scorer.openset, "evaluate_held_scale_known"
                ) as evaluate,
            ):
                with self.assertRaisesRegex(ValueError, "tampered intent"):
                    scorer.score_once(Namespace(intent=str(intent)))
            self.assertFalse(
                os.path.lexists(root / scorer.LEDGER_RELATIVE_PATH)
            )
            evaluate.assert_not_called()

    def test_reference_separation_failure_is_before_ledger_and_inference(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            intent = root / "intent.json"
            intent.write_text("{}\n", encoding="utf-8")
            with (
                mock.patch.object(scorer, "REPO", root),
                mock.patch.object(
                    scorer, "INTENT_RELATIVE_PATH", "intent.json"
                ),
                mock.patch.object(
                    scorer,
                    "_validate_intent",
                    side_effect=ValueError(
                        "reference separation collision"
                    ),
                ),
                mock.patch.object(
                    scorer.openset, "evaluate_held_scale_known"
                ) as evaluate,
            ):
                with self.assertRaisesRegex(
                    ValueError, "reference separation"
                ):
                    scorer.score_once(Namespace(intent=str(intent)))
            self.assertFalse(
                os.path.lexists(root / scorer.LEDGER_RELATIVE_PATH)
            )
            evaluate.assert_not_called()

    def test_prepare_mode_does_not_call_model_evaluator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            intent_path = root / "intent.json"
            payload = {"schema": scorer.INTENT_SCHEMA}
            args = Namespace(
                intent_output=str(intent_path),
                protocol="unused",
                scale_corpus="unused",
                fusion="unused",
                historical_corpus="unused",
                current_corpus="unused",
                classifier_asset="unused",
                policy="unused",
                output_dir="unused",
                device="cpu",
            )
            with (
                mock.patch.object(scorer, "REPO", root),
                mock.patch.object(
                    scorer, "INTENT_RELATIVE_PATH", "intent.json"
                ),
                mock.patch.object(
                    scorer,
                    "build_intent_payload",
                    return_value=(payload, object(), {}),
                ),
                mock.patch.object(
                    scorer.openset, "evaluate_held_scale_known"
                ) as evaluate,
            ):
                result = scorer.prepare_intent(args)
            self.assertEqual(result, payload)
            self.assertEqual(_read(intent_path), payload)
            evaluate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
