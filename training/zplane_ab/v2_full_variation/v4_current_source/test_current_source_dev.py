from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import current_source_data as data  # noqa: E402
from run_current_source_dev import (  # noqa: E402
    _build_sampling_hierarchy,
    _classification_report,
    _jsonable,
    _predict_from_profile_bank,
    _sample_episode,
    _select_hierarchical_bases,
    checkpoint_score,
)


SAMPLE_COUNT = 4_096
CLASS_INDEX = {
    "am": 0,
    "bluetooth": 1,
    "cw": 2,
    "dsss": 3,
    "fm": 4,
    "gsm": 5,
    "ofdm": 6,
}


def _signal(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(SAMPLE_COUNT, 2)).astype("<f4")


def _sha256_bytes(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(value, dtype="<f4").tobytes(order="C")
    ).hexdigest()


def _receipt(
    content_sha256: str,
    receiver_preset: str,
) -> dict:
    return {
        "measurementId": "00000000-0000-4000-8000-000000000001",
        "sessionId": "00000000-0000-4000-8000-000000000002",
        "configurationRevision": "00000000-0000-4000-8000-000000000003",
        "sequence": 1,
        "capturedAt": "2026-07-29T00:00:00.000Z",
        "complete": True,
        "sampleCount": SAMPLE_COUNT,
        "byteLength": SAMPLE_COUNT * 8,
        "samplesSha256": content_sha256,
        "qualification": "independently-verified-digital-baseband",
        "payloadKind": "native-canonical",
        "representation": "source-preserved-complex-envelope",
        "normalization": "none",
        "receiverImpairment": receiver_preset,
        "channelApplication": "not-applied",
        "canonicalArtifactSha256": "a" * 64,
        "provenance": {
            "driverId": "signal-lab",
            "sourceKind": "signal-lab-simulation",
            "execution": "signal-lab-simulation",
            "transport": "signal-lab-measurement-bridge",
            "contractId": "tinysa-signal-lab-atomizer-measurement",
            "contractVersion": 2,
            "contractSha256":
                data.EXPECTED_SIGNAL_LAB_CONTRACT_SHA256,
            "generatorContractBindingSha256":
                data.EXPECTED_SIGNAL_LAB_GENERATOR_BINDING_SHA256,
            "catalogSha256": data.EXPECTED_SIGNAL_LAB_CATALOG_SHA256,
            "claims": {
                "usbEmulated": False,
                "firmwareExecuted": False,
                "rfEmitted": False,
            },
        },
        "transformReceipt": {
            "receiptVersion": 1,
            "outputSampleCount": SAMPLE_COUNT,
            "outputSampleRateHz": 80_000_000,
            "outputSamplesSha256": content_sha256,
            "sourceSampleRateHz": 80_000_000,
            "sourceSamplesSha256": content_sha256,
            "sourceArtifactSha256": "a" * 64,
            "sourceSampleCount": SAMPLE_COUNT,
            "outputStartSourceSampleNumerator": "0",
            "outputStartSourceSampleDenominator": "1",
            "sourceBoundaryPolicy": "one-shot-zero-extended",
            "sourcePeriodSamples": None,
            "sourceCarrierOffsetHz": 0,
            "outputCarrierOffsetHz": 0,
            "operations": [],
        },
    }


def _write_strict_corpus(directory: Path) -> Path:
    directory.mkdir()
    rows: list[np.ndarray] = []
    clean_rows: list[np.ndarray] = []
    items: list[dict] = []
    roles = ("train", "train", "train", "train", "train",
             "enrollment", "enrollment", "selection", "selection")
    per_profile_role: dict[str, dict[str, int]] = {}
    per_class = {class_name: 0 for class_name in data.CURRENT_CLASSES}
    for profile_index, profile in enumerate(data.CURRENT_PROFILES):
        class_name = data.CURRENT_PROFILE_PUBLIC_CLASS_MAP[profile]
        per_profile_role[profile] = {
            role: roles.count(role) for role in data.ROLES
        }
        for repeat, role in enumerate(roles):
            row = _signal(10_000 + profile_index * 100 + repeat)
            # Every synthetic fixture row declares the clean preset, so its
            # audit-only clean pair must be byte-identical to service output.
            clean = row.copy()
            content_sha256 = _sha256_bytes(row)
            clean_sha256 = _sha256_bytes(clean)
            index = len(items)
            rows.append(row)
            clean_rows.append(clean)
            per_class[class_name] += 1
            items.append(
                {
                    "index": index,
                    "cls": class_name,
                    "profile": profile,
                    "profileId": profile,
                    "replay": "one-shot",
                    "phaseNativeSample": 0,
                    "splitRole": role,
                    "validSampleCount": SAMPLE_COUNT,
                    "storageSampleCount": SAMPLE_COUNT,
                    "zeroPaddedAfterValidSampleCount": False,
                    "baseRowId": f"{profile}:{role}:{repeat}",
                    "contentSha256": content_sha256,
                    "storedSha256": content_sha256,
                    "cleanContentSha256": clean_sha256,
                    "cleanStoredSha256": clean_sha256,
                    "sampleRateHz": 80_000_000,
                    "signalBandwidthHz": 1_000_000,
                    "captureBandwidthHz": 63_000_000,
                    "receiverPreset": "clean",
                    "receiverRealizationSeed": None,
                    "measurementReceipt":
                        _receipt(content_sha256, "clean"),
                    "cleanMeasurementReceipt":
                        _receipt(clean_sha256, "clean"),
                }
            )
    raw = np.stack(rows).astype("<f4")
    clean_raw = np.stack(clean_rows).astype("<f4")
    raw.tofile(directory / "corpus.f32")
    clean_raw.tofile(directory / "corpus_clean.f32")
    source_sha256 = data.EXPECTED_CURRENT_GENERATOR_SOURCE_SHA256
    bundle_sha256 = data.EXPECTED_CURRENT_GENERATOR_BUNDLE_SHA256
    manifest = {
        "schemaVersion": data.CURRENT_CORPUS_SCHEMA_VERSION,
        "generator": data.CURRENT_CORPUS_GENERATOR,
        "generatorPath": data.CURRENT_CORPUS_GENERATOR_PATH,
        "generatorLineage": {
            "schema": data.CURRENT_CORPUS_GENERATOR_LINEAGE_SCHEMA,
            "repository": "Atom-Classifier",
            "sourcePath": data.CURRENT_CORPUS_GENERATOR_PATH,
            "sourceSha256": source_sha256,
            "bundlePath": data.CURRENT_CORPUS_GENERATOR_BUNDLE_PATH,
            "bundleSha256": bundle_sha256,
            "sourceBundleBindingSha256":
                data._generator_lineage_binding(
                    source_path=data.CURRENT_CORPUS_GENERATOR_PATH,
                    source_sha256=source_sha256,
                    bundle_path=data.CURRENT_CORPUS_GENERATOR_BUNDLE_PATH,
                    bundle_sha256=bundle_sha256,
                ),
        },
        "servicePath": list(data.CURRENT_SERVICE_PATH),
        "lowLevelSynthesizerCalls": False,
        "corpusSeed": 7,
        "targetPerProfile": 9,
        "count": len(items),
        "sampleCount": SAMPLE_COUNT,
        "format": "cf32le-interleaved",
        "dataFile": "corpus.f32",
        "cleanDataFile": "corpus_clean.f32",
        "dataSha256": _sha256_bytes(raw),
        "cleanDataSha256": _sha256_bytes(clean_raw),
        "hasCleanPairs": True,
        "runtimeInputBuckets": list(data.RUNTIME_INPUT_LENGTHS),
        "receiverPresets": list(data.CURRENT_RECEIVER_PRESETS),
        "source": {
            "repository": "Atom-SignalLab",
            "gitCommit": data.EXPECTED_SIGNAL_LAB_GIT_COMMIT,
            "gitTree": data.EXPECTED_SIGNAL_LAB_GIT_TREE,
            "worktreeClean": True,
            "contractSha256":
                data.EXPECTED_SIGNAL_LAB_CONTRACT_SHA256,
            "generatorContractBindingSha256":
                data.EXPECTED_SIGNAL_LAB_GENERATOR_BINDING_SHA256,
        },
        "classes": list(data.CURRENT_CLASSES),
        "profiles": list(data.CURRENT_PROFILES),
        "profilePublicClassMap":
            dict(data.CURRENT_PROFILE_PUBLIC_CLASS_MAP),
        "perClass": per_class,
        "perProfileRole": per_profile_role,
        "items": items,
    }
    (directory / "corpus.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return directory


def _clone_with_manifest(
    source: Path,
    destination: Path,
    mutate,
) -> Path:
    destination.mkdir()
    for filename in ("corpus.f32", "corpus_clean.f32"):
        os.link(source / filename, destination / filename)
    manifest = json.loads((source / "corpus.json").read_text())
    mutate(manifest)
    (destination / "corpus.json").write_text(json.dumps(manifest))
    return destination


class CurrentSourceDataTests(unittest.TestCase):
    def test_strict_service_corpus_contract_and_cross_role_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clean = _write_strict_corpus(root / "clean")
            loaded = data.load_current_corpus(
                clean, class_index=CLASS_INDEX
            )
            self.assertEqual(len(loaded.audit["profiles_present"]), 31)
            self.assertEqual(
                loaded.audit["profile_public_class_map"]
                ["wifi-hr-dsss-11m"],
                "dsss",
            )
            self.assertEqual(
                loaded.audit["byte_identical_runtime_prefix_cross_role_count"],
                0,
            )

            def cross_identity(manifest: dict) -> None:
                train = next(
                    item for item in manifest["items"]
                    if item["splitRole"] == "train"
                )
                selection = next(
                    item for item in manifest["items"]
                    if item["splitRole"] == "selection"
                )
                selection["baseRowId"] = train["baseRowId"]

            crossed = _clone_with_manifest(
                clean, root / "crossed", cross_identity
            )
            with self.assertRaisesRegex(ValueError, "crosses split roles"):
                data.load_current_corpus(
                    crossed, class_index=CLASS_INDEX
                )

    def test_forged_generator_hash_index_label_and_receipt_fail_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clean = _write_strict_corpus(root / "clean")
            cases = [
                (
                    "generator",
                    lambda manifest: manifest.__setitem__(
                        "generator", "forged-generator"
                    ),
                    "generator changed",
                ),
                (
                    "data-hash",
                    lambda manifest: manifest.__setitem__(
                        "dataSha256", "0" * 64
                    ),
                    "SHA-256 disagrees",
                ),
                (
                    "index",
                    lambda manifest: manifest["items"][0].__setitem__(
                        "index", 17
                    ),
                    "index must equal",
                ),
                (
                    "stored-hash",
                    lambda manifest: manifest["items"][0].__setitem__(
                        "storedSha256", "0" * 64
                    ),
                    "storedSha256 disagrees",
                ),
                (
                    "receipt",
                    lambda manifest: manifest["items"][0]
                    ["measurementReceipt"].__setitem__(
                        "samplesSha256", "0" * 64
                    ),
                    "samplesSha256 changed",
                ),
                (
                    "padding-flag",
                    lambda manifest: manifest["items"][0].__setitem__(
                        "zeroPaddedAfterValidSampleCount", True
                    ),
                    "zero-padding flag",
                ),
            ]
            for name, mutate, error in cases:
                forged = _clone_with_manifest(
                    clean, root / name, mutate
                )
                with self.subTest(name=name):
                    with self.assertRaisesRegex(ValueError, error):
                        data.load_current_corpus(
                            forged, class_index=CLASS_INDEX
                        )

            def relabel_wifi(manifest: dict) -> None:
                wifi = next(
                    item for item in manifest["items"]
                    if item["profile"] == "wifi-hr-dsss-11m"
                )
                wifi["cls"] = "bluetooth"

            mislabeled = _clone_with_manifest(
                clean, root / "wifi-as-bluetooth", relabel_wifi
            )
            with self.assertRaisesRegex(
                ValueError,
                "canonical class is 'dsss'",
            ):
                data.load_current_corpus(
                    mislabeled, class_index=CLASS_INDEX
                )

    def test_runtime_length_admission_is_exact(self) -> None:
        self.assertEqual(
            data.training_view_lengths(12_160),
            (4_096, 8_192),
        )
        self.assertEqual(data.admitted_input_length(12_160), 8_192)
        with self.assertRaisesRegex(ValueError, "below the live minimum"):
            data.admitted_input_length(4_095)

    def test_checkpoint_score_order_is_exactly_predeclared(self) -> None:
        floor_wins = checkpoint_score(
            historical_balanced=0.80,
            current_present_class_balanced=0.70,
            combined_balanced=0.10,
            current_balanced=0.10,
        )
        tempting_combined = checkpoint_score(
            historical_balanced=0.69,
            current_present_class_balanced=0.99,
            combined_balanced=0.99,
            current_balanced=0.99,
        )
        self.assertGreater(floor_wins, tempting_combined)

        self.assertGreater(
            checkpoint_score(
                historical_balanced=0.80,
                current_present_class_balanced=0.70,
                combined_balanced=0.81,
                current_balanced=0.20,
            ),
            checkpoint_score(
                historical_balanced=0.80,
                current_present_class_balanced=0.70,
                combined_balanced=0.80,
                current_balanced=0.99,
            ),
        )
        self.assertGreater(
            checkpoint_score(
                historical_balanced=0.81,
                current_present_class_balanced=0.70,
                combined_balanced=0.80,
                current_balanced=0.20,
            ),
            checkpoint_score(
                historical_balanced=0.80,
                current_present_class_balanced=0.70,
                combined_balanced=0.80,
                current_balanced=0.99,
            ),
        )
        self.assertGreater(
            checkpoint_score(
                historical_balanced=0.80,
                current_present_class_balanced=0.70,
                combined_balanced=0.80,
                current_balanced=0.21,
            ),
            checkpoint_score(
                historical_balanced=0.80,
                current_present_class_balanced=0.70,
                combined_balanced=0.80,
                current_balanced=0.20,
            ),
        )

    def test_episode_sampler_balances_source_profile_and_distinct_bases(
        self,
    ) -> None:
        labels = np.asarray(
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            dtype=np.int64,
        )
        sources = (
            ["historical"] * 4
            + ["current"] * 4
            + ["historical"] * 4
        )
        profiles = (
            ["hist-a", "hist-a", "hist-b", "hist-b"]
            + ["current-a", "current-a", "current-b", "current-b"]
            + ["analog-a", "analog-a", "analog-b", "analog-b"]
        )
        hierarchy = _build_sampling_hierarchy(
            labels, sources, profiles, n_classes=2
        )
        selected = _select_hierarchical_bases(
            hierarchy,
            np.random.default_rng(43),
            per_class=4,
        )
        self.assertEqual(len(np.unique(selected[0])), 4)
        self.assertEqual(len(np.unique(selected[1])), 4)
        self.assertEqual(
            sorted(sources[int(base)] for base in selected[0]),
            ["current", "current", "historical", "historical"],
        )
        self.assertEqual(
            sorted(profiles[int(base)] for base in selected[0]),
            ["current-a", "current-b", "hist-a", "hist-b"],
        )
        self.assertEqual(
            {sources[int(base)] for base in selected[1]},
            {"historical"},
        )
        self.assertEqual(
            sorted(profiles[int(base)] for base in selected[1]),
            ["analog-a", "analog-a", "analog-b", "analog-b"],
        )

        base_views = tuple(
            np.asarray([base * 10, base * 10 + 1])
            for base in range(len(labels))
        )
        support, query, support_labels, query_labels = _sample_episode(
            hierarchy,
            base_views,
            np.random.default_rng(44),
            k_shot=2,
            q_query=2,
        )
        sampled_base_identities = (
            np.concatenate((support, query)) // 10
        ).tolist()
        self.assertEqual(
            len(set(sampled_base_identities)),
            len(sampled_base_identities),
        )
        self.assertEqual(support_labels.tolist(), [0, 0, 1, 1])
        self.assertEqual(query_labels.tolist(), [0, 0, 1, 1])

    def test_exact_one_third_source_apportionment_is_deterministic(self) -> None:
        labels = np.asarray(
            [0] * 12 + [1] * 6,
            dtype=np.int64,
        )
        sources = (
            ["historical"] * 6
            + ["current"] * 6
            + ["historical"] * 6
        )
        profiles = (
            [f"hist-{index % 2}" for index in range(6)]
            + [f"current-{index % 2}" for index in range(6)]
            + [f"only-{index % 2}" for index in range(6)]
        )
        hierarchy = _build_sampling_hierarchy(
            labels, sources, profiles, n_classes=2
        )
        counts: list[int] = []
        for allocation_round in range(3):
            first = _select_hierarchical_bases(
                hierarchy,
                np.random.default_rng(900 + allocation_round),
                per_class=4,
                current_source_share=Fraction(1, 3),
                allocation_round=allocation_round,
            )
            second = _select_hierarchical_bases(
                hierarchy,
                np.random.default_rng(900 + allocation_round),
                per_class=4,
                current_source_share=Fraction(1, 3),
                allocation_round=allocation_round,
            )
            self.assertEqual(
                [value.tolist() for value in first],
                [value.tolist() for value in second],
            )
            counts.append(
                sum(sources[int(base)] == "current" for base in first[0])
            )
            self.assertTrue(
                all(sources[int(base)] == "historical" for base in first[1])
            )
        self.assertEqual(counts, [1, 1, 2])
        self.assertEqual(sum(counts), 4)
        self.assertEqual(
            _jsonable(Fraction(1, 3)),
            {"numerator": 1, "denominator": 3, "value": 1 / 3},
        )
        base_views = tuple(
            np.asarray([base * 10, base * 10 + 1])
            for base in range(len(labels))
        )
        support, query, _, _ = _sample_episode(
            hierarchy,
            base_views,
            np.random.default_rng(999),
            k_shot=2,
            q_query=2,
            current_source_share=Fraction(1, 3),
            allocation_round=2,
        )
        base_identities = (np.concatenate((support, query)) // 10).tolist()
        self.assertEqual(
            len(base_identities),
            len(set(base_identities)),
        )

    def test_profile_selection_metrics_report_the_worst_profile(self) -> None:
        rows = [
            data.RowRef(
                source="current",
                source_index=index,
                role="selection",
                class_name=class_name,
                label=label,
                profile_id=profile,
                valid_sample_count=4_096,
                storage_sample_count=4_096,
                identity=f"row-{index}",
                content_sha256=f"{index:064x}",
                zero_padded_after_valid=False,
            )
            for index, (class_name, label, profile) in enumerate(
                [
                    ("bluetooth", 1, "bluetooth-a"),
                    ("bluetooth", 1, "bluetooth-a"),
                    ("dsss", 3, "wifi-hr-dsss-11m"),
                ]
            )
        ]
        report = _classification_report(
            np.asarray([1, 3, 3]),
            np.asarray([1, 1, 3]),
            list(CLASS_INDEX),
            rows,
        )
        self.assertEqual(report["profile_count"], 2)
        self.assertEqual(report["worst_profile_recall"], 0.5)
        self.assertEqual(
            report["worst_profiles"],
            ["current:bluetooth-a"],
        )
        self.assertEqual(
            report["per_profile"]["current:wifi-hr-dsss-11m"]["recall"],
            1.0,
        )

    def test_public_prediction_is_minimum_over_profile_prototypes(
        self,
    ) -> None:
        prototypes = np.asarray(
            [
                [0.0, 0.0],   # class 0, historical profile
                [10.0, 0.0],  # class 0, current/profile-revision mode
                [5.0, 5.0],   # class 1
            ],
            dtype=np.float32,
        )
        prediction, class_distance = _predict_from_profile_bank(
            np.asarray([[9.5, 0.0]], dtype=np.float32),
            prototypes,
            np.asarray([0, 0, 1]),
            2,
        )
        self.assertEqual(prediction.tolist(), [0])
        self.assertAlmostEqual(float(class_distance[0, 0]), 0.25)
        self.assertGreater(float(class_distance[0, 1]), 0.25)


if __name__ == "__main__":
    unittest.main()
