from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from training.zplane_ab.v2_full_variation import (
    evaluate_invariant_release_suite as release,
)


def _identity_item(class_name: str, index: int) -> dict[str, object]:
    return {
        "cls": class_name,
        "profile": f"{class_name}-{index}",
        "sampleRateHz": 1_000_000,
        "bandwidthHz": 100_000,
        "impaired": bool(index % 2),
        "startSampleIndex": 100 * index,
        "centerHz": 100_000_000 + index,
        "snrDb": 20.0 if index % 2 else 0.0,
        "rxPowerDbm": -50.0,
        "noiseFigureDb": 5.0,
        "centreOffsetFrac": 0.0,
        "adcBits": None,
        "clockErrorPpm": 0.0,
        "multipathTaps": 0,
        "impairmentSeed": 1234 + index,
    }


def _write_words(path: Path, value: np.ndarray) -> None:
    np.asarray(value, dtype="<u4").tofile(path)


class PredeclaredSplitTests(unittest.TestCase):
    def test_split_is_deterministic_disjoint_and_exactly_five_per_class(self):
        classes = ("a", "b")
        items = [
            *[_identity_item("a", index) for index in range(8)],
            *[_identity_item("b", index) for index in range(8)],
        ]
        first = release.predeclared_five_shot_split(items, classes, 20260728)
        second = release.predeclared_five_shot_split(items, classes, 20260728)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(first[2], second[2])
        self.assertEqual(len(first[0]), 10)
        self.assertEqual(len(first[1]), 6)
        self.assertEqual(len(np.intersect1d(first[0], first[1])), 0)
        for name in classes:
            chosen = first[2]["support_by_class"][name]
            self.assertEqual(len(chosen), 5)
            self.assertTrue(all(items[index]["cls"] == name for index in chosen))

    def test_split_rejects_a_class_without_disjoint_query_rows(self):
        items = [_identity_item("a", index) for index in range(5)]
        with self.assertRaisesRegex(ValueError, "needs more than 5"):
            release.predeclared_five_shot_split(items, ("a",), 1)


class PrefixNestingTests(unittest.TestCase):
    def _corpora(self, root: Path) -> dict[int, release.CorpusSpec]:
        lengths = (4, 8, 12, 16)
        count = 4
        longest_observed = np.arange(
            count * lengths[-1] * 2,
            dtype="<u4",
        ).reshape(count, lengths[-1], 2)
        longest_clean = (
            longest_observed + np.uint32(100_000)
        ).astype("<u4")
        directories = {}
        for length in lengths:
            directory = root / f"n{length}"
            directory.mkdir()
            _write_words(
                directory / "corpus.f32",
                longest_observed[:, :length],
            )
            _write_words(
                directory / "corpus_clean.f32",
                longest_clean[:, :length],
            )
            directories[length] = directory

        longest_directory = directories[lengths[-1]]
        longest_manifest = {"count": count, "sampleCount": lengths[-1]}
        longest_manifest_path = longest_directory / "corpus.json"
        longest_manifest_path.write_text(
            json.dumps(longest_manifest),
            encoding="utf-8",
        )
        source_files = {
            "corpus.json": release._actual_file_record(longest_manifest_path),
            "corpus.f32": release._actual_file_record(
                longest_directory / "corpus.f32"
            ),
            "corpus_clean.f32": release._actual_file_record(
                longest_directory / "corpus_clean.f32"
            ),
        }
        corpora = {}
        for length in lengths:
            directory = directories[length]
            manifest_path = directory / "corpus.json"
            if length == lengths[-1]:
                manifest = longest_manifest
            else:
                manifest = {
                    "count": count,
                    "sampleCount": length,
                    "prefixDerivation": {
                        "protocol": release.PREFIX_DERIVATION_PROTOCOL,
                        "sourceDirectory": str(longest_directory),
                        "sourceSampleCount": lengths[-1],
                        "targetSampleCount": length,
                        "rowCount": count,
                        "bytesPerComplexSample": 8,
                        "sourceFiles": source_files,
                        "outputFiles": {
                            "corpus.f32": release._actual_file_record(
                                directory / "corpus.f32"
                            ),
                            "corpus_clean.f32": release._actual_file_record(
                                directory / "corpus_clean.f32"
                            ),
                        },
                    },
                }
                manifest_path.write_text(
                    json.dumps(manifest),
                    encoding="utf-8",
                )
            corpora[length] = release.CorpusSpec(
                capture_length=length,
                directory=directory,
                manifest_path=manifest_path,
                raw_path=directory / "corpus.f32",
                clean_path=directory / "corpus_clean.f32",
                manifest=manifest,
                row_keys=(),
            )
        return corpora

    def test_verifies_both_streams_and_declared_derivation(self):
        with tempfile.TemporaryDirectory() as temporary:
            corpora = self._corpora(Path(temporary))
            actual = release.verify_prefix_nesting(corpora, rows_per_chunk=2)
            self.assertTrue(actual["passes"])
            self.assertEqual(len(actual["comparisons"]), 6)
            with mock.patch.object(
                release,
                "REQUIRED_CAPTURE_LENGTHS",
                (4, 8, 12, 16),
            ):
                declared = release.validate_prefix_derivation_records(corpora)
            self.assertTrue(declared["passes"])
            self.assertEqual(len(declared["derived"]), 3)

    def test_rejects_one_bit_change_in_either_prefix_stream(self):
        for stream_attribute in ("raw_path", "clean_path"):
            with self.subTest(stream=stream_attribute):
                with tempfile.TemporaryDirectory() as temporary:
                    corpora = self._corpora(Path(temporary))
                    path = getattr(corpora[8], stream_attribute)
                    words = np.memmap(path, dtype="<u4", mode="r+")
                    words[7] ^= np.uint32(1)
                    words.flush()
                    del words
                    with self.assertRaisesRegex(
                        ValueError,
                        "not a bit-exact prefix",
                    ):
                        release.verify_prefix_nesting(corpora)

    def test_rejects_tampered_derivation_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            corpora = self._corpora(Path(temporary))
            corpora[8].manifest["prefixDerivation"]["sourceSampleCount"] = 15
            with mock.patch.object(
                release,
                "REQUIRED_CAPTURE_LENGTHS",
                (4, 8, 12, 16),
            ):
                with self.assertRaisesRegex(ValueError, "does not bind"):
                    release.validate_prefix_derivation_records(corpora)

    def test_start_probe_is_bound_to_longest_rows_but_never_scored(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            probe_directory = root / "_start_probe_n4"
            longest_directory = root / "n16"
            probe_directory.mkdir()
            longest_directory.mkdir()
            classes = ("a", "b")
            items = [
                *[_identity_item("a", index) for index in range(2)],
                *[_identity_item("b", index + 2) for index in range(2)],
            ]
            for item in items:
                item["cleanPower"] = 1.0
            manifest_path = probe_directory / "corpus.json"
            signal_root = root / "Atom-SignalLab"
            probe_manifest = {
                "sampleCount": 4,
                "format": "cf32le-interleaved",
                "corpusSeed": 99,
                "targetPerClass": 2,
                "exactTargetPerClass": True,
                "hasCleanPairs": True,
                "signalLabRoot": str(signal_root),
                "classes": list(classes),
                "count": 4,
                "items": items,
            }
            manifest_path.write_text(
                json.dumps(probe_manifest),
                encoding="utf-8",
            )
            np.zeros((4, 4, 2), dtype="<f4").tofile(
                probe_directory / "corpus.f32"
            )
            np.zeros((4, 4, 2), dtype="<f4").tofile(
                probe_directory / "corpus_clean.f32"
            )
            files = {
                name: release._actual_file_record(probe_directory / name)
                for name in (
                    "corpus.json",
                    "corpus.f32",
                    "corpus_clean.f32",
                )
            }
            longest_manifest = {
                "matchedStartsFrom": str(manifest_path),
                "items": [dict(item) for item in items],
            }
            longest = release.CorpusSpec(
                capture_length=16,
                directory=longest_directory,
                manifest_path=longest_directory / "corpus.json",
                raw_path=longest_directory / "corpus.f32",
                clean_path=longest_directory / "corpus_clean.f32",
                manifest=longest_manifest,
                row_keys=(),
            )
            release_manifest = {
                "start_probe": {
                    "scored": False,
                    "purpose": (
                        "select one occupied start per row before longest-only "
                        "nuisance realization"
                    ),
                    "capture_length": 4,
                    "directory": str(probe_directory),
                    "files": files,
                }
            }
            with mock.patch.object(
                release,
                "REQUIRED_CAPTURE_LENGTHS",
                (4, 8, 12, 16),
            ):
                report = release._validate_unscored_start_probe(
                    release_manifest,
                    root=root,
                    longest=longest,
                    classes=classes,
                    release_seed=99,
                    target_per_class=2,
                    signal_lab_root=signal_root,
                )
                self.assertTrue(report["passes"])
                self.assertFalse(report["scored"])
                release_manifest["start_probe"]["scored"] = True
                with self.assertRaisesRegex(ValueError, "declaration is invalid"):
                    release._validate_unscored_start_probe(
                        release_manifest,
                        root=root,
                        longest=longest,
                        classes=classes,
                        release_seed=99,
                        target_per_class=2,
                        signal_lab_root=signal_root,
                    )


class SourceTreeDigestTests(unittest.TestCase):
    def test_digest_excludes_generated_trees_and_detects_source_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "src").mkdir()
            (root / "node_modules").mkdir()
            (root / "dist").mkdir()
            (root / "src" / "b.ts").write_bytes(b"bravo")
            (root / "a.txt").write_bytes(b"alpha")
            (root / "node_modules" / "ignored.js").write_bytes(b"ignored")
            (root / "dist" / "ignored.js").write_bytes(b"ignored")
            (root / "RELEASE_SOURCE_PROVENANCE.json").write_bytes(b"ignored")

            expected = hashlib.sha256()
            for name, value in (("a.txt", b"alpha"), ("src/b.ts", b"bravo")):
                expected.update(name.encode("utf-8"))
                expected.update(b"\0")
                expected.update(value)
            digest, count = release._source_tree_digest(root)
            self.assertEqual(digest, expected.hexdigest())
            self.assertEqual(count, 2)

            (root / "src" / "b.ts").write_bytes(b"tampered")
            changed, changed_count = release._source_tree_digest(root)
            self.assertNotEqual(changed, digest)
            self.assertEqual(changed_count, count)

    def test_digest_rejects_unexpected_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.write_text("x", encoding="utf-8")
            (root / "alias").symlink_to(target)
            with self.assertRaisesRegex(ValueError, "contains a symlink"):
                release._source_tree_digest(root)


class DependencyProvenanceTests(unittest.TestCase):
    def test_binds_source_locks_built_dist_and_package_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            signallab = base / "Atom-SignalLab"
            atom_dsp = base / "Atom-DSP"
            (signallab / "src").mkdir(parents=True)
            (signallab / "node_modules" / "@atomos").mkdir(parents=True)
            (atom_dsp / "src").mkdir(parents=True)
            (atom_dsp / "dist").mkdir()
            (signallab / "src" / "waveforms.ts").write_bytes(b"signal")
            (atom_dsp / "src" / "index.ts").write_bytes(b"dsp")
            (signallab / "package-lock.json").write_bytes(b"signal-lock")
            (atom_dsp / "package-lock.json").write_bytes(b"dsp-lock")
            (atom_dsp / "dist" / "index.js").write_bytes(b"built-js")
            (atom_dsp / "dist" / "index.d.ts").write_bytes(b"built-dts")
            package_resolution = (
                signallab / "node_modules" / "@atomos" / "dsp"
            )
            package_resolution.symlink_to(atom_dsp, target_is_directory=True)

            signal_hash, signal_count = release._source_tree_digest(signallab)
            dsp_hash, dsp_count = release._source_tree_digest(atom_dsp)
            contract = {
                "signallab": {
                    "git_commit": "1" * 40,
                    "git_tree": "2" * 40,
                    "package_lock_sha256": release._sha256(
                        signallab / "package-lock.json"
                    ),
                    "source_tree_sha256": signal_hash,
                    "source_file_count": signal_count,
                },
                "atom_dsp": {
                    "git_commit": "3" * 40,
                    "git_tree": "4" * 40,
                    "package_lock_sha256": release._sha256(
                        atom_dsp / "package-lock.json"
                    ),
                    "source_tree_sha256": dsp_hash,
                    "source_file_count": dsp_count,
                    "dist_index_js_sha256": release._sha256(
                        atom_dsp / "dist" / "index.js"
                    ),
                    "dist_index_dts_sha256": release._sha256(
                        atom_dsp / "dist" / "index.d.ts"
                    ),
                },
            }
            provenance = {
                "signallab": {
                    "root": str(signallab),
                    **contract["signallab"],
                    "observed_source_tree": {
                        "sha256": signal_hash,
                        "file_count": signal_count,
                    },
                    "package_lock_path": str(
                        signallab / "package-lock.json"
                    ),
                },
                "atom_dsp": {
                    "root": str(atom_dsp),
                    "package_resolution": str(package_resolution),
                    **contract["atom_dsp"],
                    "observed_source_tree": {
                        "sha256": dsp_hash,
                        "file_count": dsp_count,
                    },
                    "package_lock_path": str(
                        atom_dsp / "package-lock.json"
                    ),
                    "dist_index_js": {
                        "path": str(atom_dsp / "dist" / "index.js"),
                        **release._actual_file_record(
                            atom_dsp / "dist" / "index.js"
                        ),
                    },
                    "dist_index_dts": {
                        "path": str(atom_dsp / "dist" / "index.d.ts"),
                        **release._actual_file_record(
                            atom_dsp / "dist" / "index.d.ts"
                        ),
                    },
                },
            }
            runtime = {
                "node": release.REQUIRED_NODE_VERSION,
                "npm": release.REQUIRED_NPM_VERSION,
                "npx": release.REQUIRED_NPM_VERSION,
            }
            with mock.patch.object(release, "DEPENDENCY_CONTRACT", contract):
                report = release._validate_dependency_provenance(
                    provenance,
                    runtime,
                )
                self.assertTrue(report["passes"])
                (signallab / "src" / "waveforms.ts").write_bytes(b"tamper")
                with self.assertRaisesRegex(
                    ValueError,
                    "SignalLab provenance differs",
                ):
                    release._validate_dependency_provenance(
                        provenance,
                        runtime,
                    )


class MetricTests(unittest.TestCase):
    def test_nearest_matches_explicit_squared_euclidean_distance(self):
        embeddings = np.asarray(
            [[0.25, -0.5, 0.75], [-0.125, 0.875, 0.25]],
            dtype=np.float32,
        )
        prototypes = np.asarray(
            [[0.5, -0.25, 0.625], [-0.75, 0.5, 0.125]],
            dtype=np.float32,
        )
        prediction, distance = release._nearest(embeddings, prototypes)
        expected = np.sum(
            (
                embeddings.astype(np.float64)[:, None, :]
                - prototypes.astype(np.float64)[None, :, :]
            )
            ** 2,
            axis=-1,
        )
        np.testing.assert_allclose(distance, expected, rtol=0.0, atol=1e-15)
        np.testing.assert_array_equal(prediction, np.argmin(expected, axis=1))

    def test_closed_report_includes_clean_impaired_and_snr_slices(self):
        classes = ("a", "b")
        prototypes = np.eye(2, dtype=np.float32)
        labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
        embeddings = prototypes[labels]
        snr = np.asarray([0, 20, 0, 20, 0, 20, 0, 20], dtype=np.float64)
        impaired = np.asarray(
            [False, False, True, True, False, False, True, True],
            dtype=np.bool_,
        )
        report, prediction = release._closed_report(
            embeddings,
            labels,
            snr,
            impaired,
            prototypes,
            classes,
        )
        np.testing.assert_array_equal(prediction, labels)
        for key in (
            "accuracy",
            "balanced_accuracy",
            "clean_accuracy",
            "impaired_accuracy",
        ):
            self.assertEqual(report[key], 1.0)
        self.assertEqual(report["low_snr"]["accuracy"], 1.0)
        self.assertEqual(report["high_snr"]["accuracy"], 1.0)

    def test_five_shot_uses_fixed_matched_length_support_embeddings(self):
        classes = ("a", "b")
        labels = np.asarray([0] * 6 + [1] * 6, dtype=np.int64)
        support = np.asarray([0, 1, 2, 3, 4, 6, 7, 8, 9, 10])
        query = np.asarray([5, 11])
        matched = np.vstack(
            (
                np.tile([1.0, 0.0], (6, 1)),
                np.tile([0.0, 1.0], (6, 1)),
            )
        ).astype(np.float32)
        other_length = np.zeros_like(matched)
        other_length[query] = matched[query]
        report = release._five_shot_report(
            matched,
            other_length,
            labels,
            support,
            query,
            classes,
        )
        self.assertEqual(report["accuracy"], 1.0)
        self.assertIn("n=16384", report["prototype_source"])

    def test_paired_report_has_per_class_and_snr_diagnostics(self):
        reference = np.eye(4, dtype=np.float32)
        current = reference.copy()
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        prediction = labels.copy()
        snr = np.asarray([0.0, 20.0, 0.0, 20.0])
        report = release._paired_invariance_report(
            current,
            reference,
            prediction,
            prediction,
            labels,
            snr,
            ("a", "b"),
        )
        self.assertEqual(report["prediction_agreement"], 1.0)
        self.assertEqual(report["embedding_cosine_mean"], 1.0)
        self.assertEqual(report["low_snr"]["n"], 2)
        self.assertEqual(report["high_snr"]["n"], 2)
        self.assertEqual(report["per_class"]["a"]["n"], 2)

    def test_weighted_lof_is_frozen_positive_sum(self):
        class Scorer:
            def __init__(self, offset: float):
                self.offset = offset

            def score(self, value):
                return np.asarray(value)[:, 0] + self.offset

        runtime = SimpleNamespace(
            lof_components=(
                ("real", 0.4, Scorer(1.0)),
                ("complex", 0.6, Scorer(2.0)),
            )
        )
        embeddings = {
            "real": np.asarray([[1.0], [2.0]]),
            "complex": np.asarray([[3.0], [4.0]]),
        }
        score = release._weighted_lof_score(runtime, embeddings)
        np.testing.assert_allclose(score, [3.8, 4.8])

    def test_auroc_handles_ties(self):
        self.assertEqual(
            release._auroc(np.asarray([1.0, 2.0]), np.asarray([0.0, 1.0])),
            0.875,
        )

    def test_gate_assembler_includes_every_operational_and_paired_gate(self):
        closed = {
            "4096": {
                "accuracy": 0.72,
                "clean_accuracy": 0.85,
                "family": {"accuracy": 0.82},
                "high_snr": {"accuracy": 0.78},
            }
        }
        five_shot = {"4096": {"balanced_accuracy": 0.85}}
        open_report = {
            "4096": {
                "auroc_overall": 0.72,
                "auroc_noise": 0.80,
                "auroc_chirp": 0.80,
                "known_false_unknown_rate": 0.10,
                "flagged_unknown_noise": 0.10,
                "flagged_unknown_chirp": 0.10,
            }
        }
        length = {
            "worst_balanced_accuracy": 0.75,
            "worst_mean_pairwise_cosine": 0.78,
            "worst_prediction_agreement_to_matched": 0.80,
            "worst_embedding_cosine_to_matched": 0.85,
        }
        scale = {
            "worst_balanced_accuracy": 0.75,
            "worst_mean_pairwise_cosine": 0.78,
            "worst_prediction_agreement_to_factor1": 0.75,
            "worst_embedding_cosine_to_factor1": 0.80,
        }
        gates = release._assemble_release_gates(
            candidate_sha256="a" * 64,
            expected_candidate_sha256="a" * 64,
            prefix_nesting_passes=True,
            dependency_provenance_passes=True,
            start_probe_excluded=True,
            closed_by_length=closed,
            five_shot_by_length=five_shot,
            open_by_length=open_report,
            length_sweep=length,
            scale_sweep=scale,
        )
        self.assertTrue(all(gate["passes"] for gate in gates.values()))
        expected_names = {
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
        self.assertEqual(set(gates), expected_names)

        open_report["4096"]["known_false_unknown_rate"] = 0.1001
        scale["worst_prediction_agreement_to_factor1"] = 0.7499
        failed = release._assemble_release_gates(
            candidate_sha256="a" * 64,
            expected_candidate_sha256="b" * 64,
            prefix_nesting_passes=False,
            dependency_provenance_passes=False,
            start_probe_excluded=False,
            closed_by_length=closed,
            five_shot_by_length=five_shot,
            open_by_length=open_report,
            length_sweep=length,
            scale_sweep=scale,
        )
        for name in (
            "prefix_nesting",
            "dependency_provenance",
            "start_probe_excluded",
            "candidate_sha_bound",
            "open_known_false_unknown_worst_length",
            "physical_scale_worst_prediction_agreement_to_factor1",
        ):
            self.assertFalse(failed[name]["passes"])


class ScaleAndNoveltyProtocolTests(unittest.TestCase):
    def test_scale_subset_is_metadata_only_and_enforces_minimum_per_class(self):
        items = []
        for class_name in ("a", "b"):
            for index in range(4):
                item = _identity_item(class_name, index)
                item["centreOffsetFrac"] = 0.0 if index < 3 else 0.30
                item["bandwidthHz"] = 100_000
                item["sampleRateHz"] = 1_000_000
                items.append(item)
        spec = SimpleNamespace(manifest={"items": items})
        query = np.arange(len(items), dtype=np.int64)
        with mock.patch.object(release, "SCALE_MIN_PER_CLASS", 2):
            eligible, report = release._scale_eligible_indices(
                spec,
                query,
                ("a", "b"),
            )
        self.assertEqual(len(eligible), 6)
        self.assertEqual(report["eligible_by_class"], {"a": 3, "b": 3})
        self.assertEqual(report["excluded_by_class"], {"a": 1, "b": 1})
        with self.assertRaisesRegex(ValueError, "fewer than 20"):
            release._scale_eligible_indices(spec, query, ("a", "b"))

    def test_novelty_is_generated_once_at_max_length_then_prefixed(self):
        calls: dict[str, list[int]] = {"noise": [], "chirp": []}

        def generator(name):
            def make(length, rng):
                calls[name].append(length)
                offset = 10.0 if name == "noise" else 20.0
                return (
                    np.arange(length, dtype=np.float64)
                    + offset
                    + 1j * rng.standard_normal(length)
                )

            return make

        def fake_preprocess(captures, _runtime):
            return list(captures), np.zeros((len(captures), 1), np.float32)

        def fake_embed(_runtime, packed, _features, _device):
            value = np.asarray(
                [[len(row), row[-1].real] for row in packed],
                dtype=np.float32,
            )
            return {"real": value, "complex": value, "fusion": value}

        with (
            mock.patch.object(release, "REQUIRED_CAPTURE_LENGTHS", (4, 8, 12, 16)),
            mock.patch.object(release, "NOVELTY_N_EACH_PER_LENGTH", 3),
            mock.patch.object(
                release,
                "NOVELTY",
                {"noise": generator("noise"), "chirp": generator("chirp")},
            ),
            mock.patch.object(release, "_preprocess_iq_rows", fake_preprocess),
            mock.patch.object(release, "_embed_preprocessed", fake_embed),
        ):
            embedded, provenance = release._novelty_embeddings_by_length(
                (4, 8, 12, 16),
                SimpleNamespace(),
                torch.device("cpu"),
            )
        self.assertEqual(calls["noise"], [16, 16, 16])
        self.assertEqual(calls["chirp"], [16, 16, 16])
        np.testing.assert_array_equal(
            embedded[4]["noise"]["real"][:, 0],
            np.full(3, 4, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            embedded[16]["noise"]["real"][:, 0],
            np.full(3, 16, dtype=np.float32),
        )
        self.assertIn("leading slice", provenance["prefix_rule"])

    def test_protocol_predeclares_operational_and_paired_gates(self):
        gates = release.EXPECTED_EVALUATION_PROTOCOL["gates"]
        for name in (
            "closed_clean",
            "open_known_false_unknown_max",
            "open_unknown_recall_noise",
            "open_unknown_recall_chirp",
            "length_pair_prediction_agreement",
            "length_pair_embedding_cosine",
            "scale_pair_prediction_agreement",
            "scale_pair_embedding_cosine",
        ):
            self.assertIn(name, gates)
        self.assertEqual(
            release.EXPECTED_EVALUATION_PROTOCOL["required_capture_lengths"],
            [4096, 8192, 16384, 32768],
        )
        self.assertEqual(
            release.EXPECTED_EVALUATION_PROTOCOL[
                "physical_scale_support"
            ]["minimum_rows_per_class"],
            20,
        )


if __name__ == "__main__":
    unittest.main()
