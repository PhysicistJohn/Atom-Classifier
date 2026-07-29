from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import evaluate_current_scale as subject


class CurrentScaleMetricTests(unittest.TestCase):
    def test_fixed_profile_labels_are_independent_of_manifest_claims(self) -> None:
        self.assertEqual(
            subject._fixed_public_class("wifi-hr-dsss-11m"), "dsss"
        )
        self.assertEqual(
            subject._fixed_public_class("bluetooth-le-advertising"),
            "bluetooth",
        )
        with self.assertRaisesRegex(ValueError, "unknown fixed"):
            subject._fixed_public_class("wifi-pretend-profile")

    def test_profile_bank_uses_minimum_distance_per_public_class(self) -> None:
        embeddings = np.asarray(
            [[0.1, 0.0], [9.9, 10.0], [5.1, 5.0]], dtype=np.float32
        )
        prototypes = np.asarray(
            [
                [0.0, 0.0],
                [100.0, 100.0],
                [10.0, 10.0],
                [5.0, 5.0],
            ],
            dtype=np.float32,
        )
        labels = np.asarray([0, 0, 1, 2], dtype=np.int64)
        predicted, distances = subject.predict_from_profile_bank(
            embeddings,
            prototypes,
            labels,
            3,
            prototype_sources=[
                "current", "historical", "current", "current"
            ],
            trusted_source="current",
        )
        np.testing.assert_array_equal(predicted, [0, 1, 2])
        self.assertEqual(distances.shape, (3, 3))
        self.assertAlmostEqual(float(distances[0, 0]), 0.01, places=5)

        current_missing = subject.predict_from_profile_bank(
            embeddings,
            prototypes,
            labels,
            3,
            prototype_sources=[
                "current", "historical", "historical", "historical"
            ],
            trusted_source="current",
        )[1]
        self.assertTrue(np.isinf(current_missing[:, 1:]).all())
        with self.assertRaisesRegex(ValueError, "no rows"):
            subject.predict_from_profile_bank(
                embeddings,
                prototypes,
                labels,
                3,
                prototype_sources=["current"] * 4,
                trusted_source="historical",
            )

    def test_classification_report_separates_present_and_all_public(self) -> None:
        report = subject.classification_report(
            np.asarray([0, 1, 1, 1], dtype=np.int64),
            np.asarray([0, 0, 1, 1], dtype=np.int64),
            ["a", "b", "absent"],
            ["p0", "p0", "p1", "p1"],
        )
        self.assertAlmostEqual(
            report["present_class_balanced_accuracy"], 0.75
        )
        self.assertAlmostEqual(
            report["all_public_class_balanced_accuracy"], 0.5
        )
        self.assertEqual(report["per_class"]["absent"]["recall"], None)
        self.assertAlmostEqual(report["per_profile"]["p0"]["accuracy"], 0.5)
        self.assertAlmostEqual(report["per_profile"]["p1"]["accuracy"], 1.0)

    def test_paired_report_joins_by_identity_not_row_order(self) -> None:
        left = subject.CellInference(
            factor=2.0,
            length=4_096,
            pair_ids=("b", "a"),
            labels=np.asarray([1, 0], dtype=np.int64),
            predictions=np.asarray([1, 0], dtype=np.int64),
            embeddings=np.asarray([[0.0, 1.0], [1.0, 0.0]]),
            profiles=("pb", "pa"),
        )
        right = subject.CellInference(
            factor=1.0,
            length=4_096,
            pair_ids=("a", "b"),
            labels=np.asarray([0, 1], dtype=np.int64),
            predictions=np.asarray([0, 0], dtype=np.int64),
            embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
            profiles=("pa", "pb"),
        )
        report = subject.paired_invariance_report(
            left, right, ["a", "b"]
        )
        self.assertEqual(report["pair_count"], 2)
        self.assertAlmostEqual(report["prediction_agreement"], 0.5)
        self.assertAlmostEqual(report["embedding_cosine_mean"], 1.0)
        self.assertAlmostEqual(
            report["per_class"]["a"]["prediction_agreement"], 1.0
        )
        self.assertAlmostEqual(
            report["per_class"]["b"]["prediction_agreement"], 0.0
        )

    def test_sensitive_paths_are_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "refuses release"):
            subject.reject_sensitive_path(
                "/tmp/releases/current-scale", "output"
            )
        with self.assertRaisesRegex(ValueError, "refuses release"):
            subject.reject_sensitive_path(
                "/tmp/assets-v4-release/current-scale", "output"
            )
        with self.assertRaisesRegex(ValueError, "refuses release"):
            subject.reject_sensitive_path(
                "/tmp/consumed_test/current-scale", "corpus"
            )


class CurrentScaleCorpusTests(unittest.TestCase):
    @staticmethod
    def _sha(value: bytes) -> str:
        return hashlib.sha256(value).hexdigest()

    def test_loader_validates_scale_families_and_ble_padding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            storage = 16_384
            valid_counts = [12_160, 15_200, 16_384, 16_384]
            rows = np.zeros((4, storage, 2), dtype="<f4")
            items = []
            rational = [(1, 1), (5, 4), (3, 2), (2, 1)]
            for index, (factor, valid, ratio) in enumerate(
                zip(subject.SCALE_FACTORS, valid_counts, rational)
            ):
                phase = np.arange(valid, dtype=np.float32)
                rows[index, :valid, 0] = np.sin(
                    phase * np.float32(0.01 + 0.001 * index)
                )
                rows[index, :valid, 1] = np.cos(
                    phase * np.float32(0.01 + 0.001 * index)
                )
                valid_bytes = rows[index, :valid].tobytes(order="C")
                stored_bytes = rows[index].tobytes(order="C")
                content_sha256 = self._sha(valid_bytes)
                items.append(
                    {
                        "index": index,
                        "cls": "bluetooth",
                        "profile": "bluetooth-le-advertising",
                        "profileId": "bluetooth-le-advertising",
                        "replay": "one-shot",
                        "pairId": "pair-0",
                        "realizationIndex": 0,
                        "phaseNativeSample": 0,
                        "receiverPreset": "awgn",
                        "receiverRealizationChannelSeed": 123,
                        "receiverRealizationSeed": 456,
                        "scaleFactor": factor,
                        "scaleFactorKey": f"{factor:g}",
                        "scaleNumerator": ratio[0],
                        "scaleDenominator": ratio[1],
                        "nativeSampleRateHz": 80_000_000,
                        "sampleRateHz": int(80_000_000 * factor),
                        "nativeCarrierOffsetHz": -15_000_000,
                        "signalBandwidthHz": 1_000_000,
                        "captureBandwidthHz": 31_000_000,
                        "validSampleCount": valid,
                        "availablePrefixLengths": [
                            length
                            for length in subject.PREFIX_LENGTHS
                            if length <= valid
                        ],
                        "zeroPaddedAfterValidSampleCount": valid < storage,
                        "contentSha256": content_sha256,
                        "storedSha256": self._sha(stored_bytes),
                        "measurementReceipt": {
                            "samplesSha256": content_sha256,
                            "sampleCount": valid,
                            "sampleRateHz": int(80_000_000 * factor),
                            "nativeSampleRateHz": 80_000_000,
                            "nativeCarrierOffsetHz": -15_000_000,
                            "signalBandwidthHz": 1_000_000,
                            "captureBandwidthHz": 31_000_000,
                            "receiverImpairment": "awgn",
                            "transformReceipt": {
                                "outputSamplesSha256": content_sha256,
                                "outputSampleRateHz": int(
                                    80_000_000 * factor
                                ),
                                "sourceSampleRateHz": 80_000_000,
                                "outputSampleCount": valid,
                                "outputStartSourceSampleNumerator": "0",
                                "outputStartSourceSampleDenominator": "1",
                                "operations": (
                                    []
                                    if factor == 1.0
                                    else [{"kind": "resample"}]
                                ),
                            },
                        },
                    }
                )
            raw_path = directory / "scale_eval.f32"
            raw_path.write_bytes(rows.tobytes(order="C"))
            manifest = {
                "schema": subject.CORPUS_SCHEMA,
                "generator": subject.CORPUS_SCHEMA,
                "generatorPath": subject.SCALE_GENERATOR_PATH,
                "generatorLineage": {
                    "schema": subject.GENERATOR_LINEAGE_SCHEMA,
                    "repository": "Atom-Classifier",
                    "sourcePath": subject.SCALE_GENERATOR_PATH,
                    "sourceSha256":
                        subject.EXPECTED_SCALE_GENERATOR_SOURCE_SHA256,
                    "bundlePath": subject.SCALE_GENERATOR_BUNDLE_PATH,
                    "bundleSha256":
                        subject.EXPECTED_SCALE_GENERATOR_BUNDLE_SHA256,
                    "sourceBundleBindingSha256":
                        subject._generator_lineage_binding(),
                },
                "developmentOnly": True,
                "releaseEvidence": False,
                "sealedReleaseDataUsed": 0,
                "consumedHistoricalTestRowsUsed": 0,
                "lowLevelSynthesizerCalls": False,
                "frequencyTransformsUsed": False,
                "unboundSmoke": True,
                "referenceCorpus": None,
                "heldOutContract": {"referenceBound": False},
                "physicalScaleMethod": (
                    "AtomizerMeasurementService derived output "
                    "sample-rate request"
                ),
                "captureBandwidthRule": (
                    subject.PRODUCTION_CAPTURE_BANDWIDTH_RULE
                ),
                "informativePrefixCenteredRmsFloor": (
                    subject.INFORMATIVE_PREFIX_CENTERED_RMS_FLOOR
                ),
                "preImpairmentPrefixAdmissionRule": (
                    subject.PRE_IMPAIRMENT_PREFIX_ADMISSION_RULE
                ),
                "scaleFamilyContentUniquenessRule": (
                    subject.SCALE_FAMILY_CONTENT_UNIQUENESS_RULE
                ),
                "servicePath": list(subject.EXPECTED_SERVICE_PATH),
                "scaleFactors": [
                    {
                        "key": f"{factor:g}",
                        "value": factor,
                        "numerator": numerator,
                        "denominator": denominator,
                    }
                    for factor, (numerator, denominator) in zip(
                        subject.SCALE_FACTORS, rational
                    )
                ],
                "prefixLengths": list(subject.PREFIX_LENGTHS),
                "storageSampleCount": storage,
                "realizationsPerProfile": 1,
                "count": len(items),
                "dataFile": raw_path.name,
                "dataSha256": self._sha(raw_path.read_bytes()),
                "source": {
                    "repository": "Atom-SignalLab",
                    "gitCommit": subject.EXPECTED_SIGNAL_LAB_GIT_COMMIT,
                    "gitTree": subject.EXPECTED_SIGNAL_LAB_GIT_TREE,
                    "worktreeClean": True,
                    "contractSha256":
                        subject.EXPECTED_SIGNAL_LAB_CONTRACT_SHA256,
                    "generatorContractBindingSha256":
                        subject.EXPECTED_SIGNAL_LAB_GENERATOR_BINDING_SHA256,
                },
                "profiles": ["bluetooth-le-advertising"],
                "classes": ["bluetooth"],
                "items": items,
            }
            (directory / "scale_eval.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            corpus = subject.load_scale_eval_corpus(
                directory, allow_unbound_smoke=True
            )
            self.assertEqual(len(corpus.rows), 4)
            self.assertEqual(corpus.audit["pair_count"], 1)
            self.assertEqual(
                corpus.rows[0].available_lengths, (4_096, 8_192)
            )
            self.assertEqual(
                corpus.rows[1].available_lengths, (4_096, 8_192)
            )
            self.assertEqual(
                corpus.rows[2].available_lengths,
                (4_096, 8_192, 16_384),
            )
            self.assertTrue(
                np.all(corpus.raw[0, valid_counts[0] :] == 0)
            )

            firewall_lineage = subject.ScaleGeneratorLineageBinding(
                source_sha256=(
                    "53f076da486c73bd3088d0ad2359d399736c0eaa56f5323d350d1ebca57bcd34"
                ),
                bundle_sha256=(
                    "621fa092a9513e6f54270120e9005df15962a1f99602b6df9ac493887910fd44"
                ),
            )
            with self.assertRaisesRegex(ValueError, "sourceSha256 drifted"):
                subject.load_scale_eval_corpus(
                    directory,
                    allow_unbound_smoke=True,
                    expected_generator_lineage=firewall_lineage,
                )
            manifest["generatorLineage"].update(
                {
                    "sourceSha256": firewall_lineage.source_sha256,
                    "bundleSha256": firewall_lineage.bundle_sha256,
                    "sourceBundleBindingSha256":
                        subject._generator_lineage_binding(firewall_lineage),
                }
            )
            (directory / "scale_eval.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "sourceSha256 drifted"):
                subject.load_scale_eval_corpus(
                    directory,
                    allow_unbound_smoke=True,
                )
            replacement = subject.load_scale_eval_corpus(
                directory,
                allow_unbound_smoke=True,
                expected_generator_lineage=firewall_lineage,
            )
            self.assertEqual(len(replacement.rows), 4)
            manifest["generatorLineage"].update(
                {
                    "sourceSha256":
                        subject.EXPECTED_SCALE_GENERATOR_SOURCE_SHA256,
                    "bundleSha256":
                        subject.EXPECTED_SCALE_GENERATOR_BUNDLE_SHA256,
                    "sourceBundleBindingSha256":
                        subject._generator_lineage_binding(),
                }
            )
            (directory / "scale_eval.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "not evaluation evidence"):
                subject.load_scale_eval_corpus(directory)

            manifest["generatorLineage"]["sourceSha256"] = "0" * 64
            (directory / "scale_eval.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "sourceSha256 drifted"):
                subject.load_scale_eval_corpus(
                    directory, allow_unbound_smoke=True
                )
            manifest["generatorLineage"]["sourceSha256"] = (
                subject.EXPECTED_SCALE_GENERATOR_SOURCE_SHA256
            )

            manifest["schema"] = (
                "current-signallab-service-scale-eval-v1"
            )
            manifest["generator"] = manifest["schema"]
            (directory / "scale_eval.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "schema must be"):
                subject.load_scale_eval_corpus(
                    directory, allow_unbound_smoke=True
                )
            manifest["schema"] = subject.CORPUS_SCHEMA
            manifest["generator"] = subject.CORPUS_SCHEMA

            manifest["informativePrefixCenteredRmsFloor"] = 0.0
            (directory / "scale_eval.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ValueError, "informative-prefix RMS floor"
            ):
                subject.load_scale_eval_corpus(
                    directory, allow_unbound_smoke=True
                )
            manifest["informativePrefixCenteredRmsFloor"] = (
                subject.INFORMATIVE_PREFIX_CENTERED_RMS_FLOOR
            )

            manifest["preImpairmentPrefixAdmissionRule"] = "weaker"
            (directory / "scale_eval.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ValueError, "pre-impairment prefix admission"
            ):
                subject.load_scale_eval_corpus(
                    directory, allow_unbound_smoke=True
                )
            manifest["preImpairmentPrefixAdmissionRule"] = (
                subject.PRE_IMPAIRMENT_PREFIX_ADMISSION_RULE
            )

            manifest["scaleFamilyContentUniquenessRule"] = "weaker"
            (directory / "scale_eval.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ValueError, "family-content uniqueness"
            ):
                subject.load_scale_eval_corpus(
                    directory, allow_unbound_smoke=True
                )
            manifest["scaleFamilyContentUniquenessRule"] = (
                subject.SCALE_FAMILY_CONTENT_UNIQUENESS_RULE
            )

            items[0]["captureBandwidthHz"] = 1_000_000
            items[0]["measurementReceipt"]["captureBandwidthHz"] = 1_000_000
            (directory / "scale_eval.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "production-exact"):
                subject.load_scale_eval_corpus(
                    directory, allow_unbound_smoke=True
                )

    def test_loader_rejects_duplicate_scale_within_pair(self) -> None:
        # Pair-family uniqueness is also asserted by the complete loader test;
        # exercise the public metric join's duplicate-free assumption directly.
        cell = subject.CellInference(
            factor=1.0,
            length=4_096,
            pair_ids=("same", "same"),
            labels=np.asarray([0, 0]),
            predictions=np.asarray([0, 0]),
            embeddings=np.asarray([[1.0, 0.0], [1.0, 0.0]]),
            profiles=("p", "p"),
        )
        reference = subject.CellInference(
            factor=2.0,
            length=4_096,
            pair_ids=("same",),
            labels=np.asarray([0]),
            predictions=np.asarray([0]),
            embeddings=np.asarray([[1.0, 0.0]]),
            profiles=("p",),
        )
        # The last duplicate would silently win without an explicit guard.
        with self.assertRaisesRegex(ValueError, "duplicate paired identities"):
            subject.paired_invariance_report(cell, reference, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
