from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import time_domain_invariant_patch_preprocess as td_preprocess  # noqa: E402
import current_source_data as corpus_data  # noqa: E402
from export_current_source_browser_runtime import (  # noqa: E402
    BROWSER_SCHEMA,
    EXPORT_MANIFEST_NAME,
    PROFILE_BANK_DECISION_RULE,
    PROBE_NAME,
    PUBLIC_CLASSES,
    TRUSTED_SOURCE_ROUTING,
    WEIGHTS_NAME,
    _sha256,
    build_browser_payload,
    load_verified_fusion,
    minimum_public_class_distances,
    minimum_source_public_class_distances,
    reject_sensitive_path,
    run,
    validate_empty_output,
)
from invariant_fusion import CenteredInvariantFusion  # noqa: E402
from invariant_patch_cnn import (  # noqa: E402
    InvariantPatchCNN,
    InvariantPatchConfig,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _write_smoke_fusion(directory: Path) -> dict:
    directory.mkdir()
    torch.manual_seed(20260729)
    real_config = InvariantPatchConfig(
        encoder="real",
        patch_length=64,
        patch_count=16,
        patch_dim=8,
        hidden=12,
        embed_dim=8,
        dropout=0.0,
    ).validate()
    complex_config = InvariantPatchConfig(
        **{
            **real_config.__dict__,
            "encoder": "complex",
        }
    ).validate()
    real = InvariantPatchCNN(real_config)
    complex_branch = InvariantPatchCNN(complex_config)
    real_center = np.linspace(-0.1, 0.1, 8, dtype=np.float32)
    complex_center = np.linspace(0.1, -0.1, 8, dtype=np.float32)
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

    state_path = directory / "fusion_state_dict.pt"
    torch.save(fusion.state_dict(), state_path)
    rng = np.random.default_rng(20260729)
    current_profiles = sorted(
        corpus_data.CURRENT_PROFILE_PUBLIC_CLASS_MAP
    )
    prototype_count = len(PUBLIC_CLASSES) + len(current_profiles)
    prototype_bank = rng.normal(
        size=(prototype_count, 16)
    ).astype(np.float32)
    prototype_bank /= np.maximum(
        np.linalg.norm(prototype_bank, axis=1, keepdims=True),
        np.float32(1e-12),
    )
    labels = np.asarray(
        list(range(len(PUBLIC_CLASSES)))
        + [
            PUBLIC_CLASSES.index(
                corpus_data.CURRENT_PROFILE_PUBLIC_CLASS_MAP[profile]
            )
            for profile in current_profiles
        ],
        dtype=np.int64,
    )
    groups = [
        {
            "source": "historical",
            "profile": f"historical-{PUBLIC_CLASSES[index]}",
            "public_class": PUBLIC_CLASSES[int(label)],
            "enrollment_views": 3,
            "eligible_lengths": [4_096, 8_192, 16_384],
        }
        for index, label in enumerate(labels[: len(PUBLIC_CLASSES)])
    ] + [
        {
            "source": "current",
            "profile": profile,
            "public_class":
                corpus_data.CURRENT_PROFILE_PUBLIC_CLASS_MAP[profile],
            "enrollment_views": 3,
            "eligible_lengths": (
                [4_096, 8_192]
                if profile == "bluetooth-le-advertising"
                else [4_096, 8_192, 16_384]
            ),
        }
        for profile in current_profiles
    ]
    paths = {
        "state_dict": state_path,
        "prototype_bank": directory / "fusion_prototype_bank.npy",
        "prototype_labels": directory / "fusion_prototype_labels.npy",
        "prototype_metadata": directory / "fusion_prototype_bank.json",
        "real_center": directory / "real_center.npy",
        "complex_center": directory / "complex_center.npy",
        "feature_mean": directory / "feature_mean.npy",
        "feature_std": directory / "feature_std.npy",
    }
    np.save(paths["prototype_bank"], prototype_bank)
    np.save(paths["prototype_labels"], labels)
    _write_json(
        paths["prototype_metadata"],
        {
            "classes": list(PUBLIC_CLASSES),
            "decision_rule": (
                "minimum squared distance across source/profile centroids "
                "mapped to each public class"
            ),
            "enrollment_view_rule": (
                "each source/profile centroid pools every eligible "
                "4,096/8,192/16,384 prefix view"
            ),
            "groups": groups,
            "prototype_public_class_indices": labels.tolist(),
        },
    )
    np.save(paths["real_center"], real_center)
    np.save(paths["complex_center"], complex_center)
    np.save(
        paths["feature_mean"],
        np.zeros(real_config.n_features, dtype=np.float32),
    )
    np.save(
        paths["feature_std"],
        np.ones(real_config.n_features, dtype=np.float32),
    )
    artifacts: dict[str, object] = {}
    for key, path in paths.items():
        artifacts[key] = path.name
        artifacts[f"{key}_sha256"] = _sha256(path)
    frontend = td_preprocess.preprocess_metadata()
    metrics = {
        "status": "complete",
        "development_only": True,
        "release_evidence": False,
        "sealed_release_data_used": 0,
        "consumed_historical_test_rows_used": 0,
        "consumed_test_rows_exposed": 0,
        "encoder": "fusion",
        "architecture": fusion.config(),
        "parameter_count": fusion.parameter_counts()["total"],
        "frontend": frontend,
        "run_configuration": {
            "schema": "v4-current-source-development-fusion-v1",
            "seed_inherited_from_matched_branches": 20260729,
        },
        "fitting_contract": {
            "prototype_grouping": "(source, profile)",
            "consumed_historical_test_rows_loaded": 0,
            "sealed_release_rows_loaded": 0,
        },
        "data_audit": {
            "preprocessing": {
                "runtime_bucket_policy": {
                    "training_prefixes": [4_096, 8_192, 16_384],
                    "views_never_exceed_valid_sample_count": True,
                },
            },
        },
        "artifacts": artifacts,
        "source_sha256": {"smoke": "test"},
    }
    _write_json(directory / "dev_metrics.json", metrics)
    return metrics


class V4BrowserExporterTests(unittest.TestCase):
    def test_mapped_distance_reference_returns_exactly_seven_columns(self) -> None:
        embedding = np.asarray([[0.5, -0.5]], dtype=np.float32)
        bank = np.asarray(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [0.5, -0.25],
                [0.25, -0.75],
                [0.7, -0.4],
                [-0.5, -0.5],
                [0.5, 0.5],
                [1.0, -0.5],
                [0.51, -0.49],
            ],
            dtype=np.float32,
        )
        labels = np.asarray([0, 0, 1, 2, 3, 4, 5, 6, 3], dtype=np.int64)
        distance, closest = minimum_public_class_distances(
            embedding, bank, labels
        )
        self.assertEqual(distance.shape, (1, 7))
        self.assertEqual(closest.shape, (1, 7))
        np.testing.assert_allclose(
            distance[0],
            np.asarray(
                [0.5, 0.0625, 0.125, 0.0002, 1.0, 1.0, 0.25],
                dtype=np.float32,
            ),
            rtol=0.0,
            atol=2e-8,
        )
        np.testing.assert_array_equal(
            closest[0], np.asarray([0, 2, 3, 8, 5, 6, 7])
        )

    def test_source_distance_reference_never_falls_back_to_union(self) -> None:
        embedding = np.asarray([[0.0, 0.0]], dtype=np.float32)
        bank = np.asarray(
            [[0.0, 0.0], [0.1, 0.0], [1.0, 1.0]],
            dtype=np.float32,
        )
        labels = np.asarray([0, 1, 2], dtype=np.int64)
        distances, closest, supported = (
            minimum_source_public_class_distances(
                embedding,
                bank,
                labels,
                ["current", "current", "historical"],
                "current",
                n_classes=3,
            )
        )
        np.testing.assert_array_equal(supported, [True, True, False])
        self.assertTrue(np.isinf(distances[0, 2]))
        self.assertEqual(int(closest[0, 2]), -1)
        self.assertEqual(int(np.argmin(distances[0])), 0)

    def test_verified_fusion_builds_profile_bank_browser_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fusion_dir = Path(temporary) / "fusion"
            _write_smoke_fusion(fusion_dir)
            fusion = load_verified_fusion(fusion_dir)
            payload = build_browser_payload(fusion)
            self.assertEqual(payload["schema"], BROWSER_SCHEMA)
            self.assertEqual(
                payload["classification"]["decision_rule"],
                PROFILE_BANK_DECISION_RULE,
            )
            self.assertEqual(
                payload["classification"][
                    "prototype_public_class_indices"
                ][:7],
                [0, 1, 2, 3, 4, 5, 6],
            )
            self.assertEqual(
                payload["classification"]["routing"],
                TRUSTED_SOURCE_ROUTING,
            )
            self.assertEqual(
                payload["frontend"]["runtime_input_lengths"],
                [4_096, 8_192, 16_384],
            )
            self.assertFalse(payload["frontend"]["uses_frequency_transform"])

    def test_full_export_is_deterministic_and_emits_all_bucket_probes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fusion_dir = root / "fusion"
            _write_smoke_fusion(fusion_dir)
            outputs = [root / "browser-a", root / "browser-b"]
            results = []
            for output in outputs:
                args = type(
                    "Args",
                    (),
                    {"fusion": str(fusion_dir), "output": str(output)},
                )()
                results.append(run(args))
            for name in (WEIGHTS_NAME, PROBE_NAME, EXPORT_MANIFEST_NAME):
                self.assertEqual(
                    (outputs[0] / name).read_bytes(),
                    (outputs[1] / name).read_bytes(),
                )
            fixture = results[0]["fixture"]
            self.assertEqual(
                [
                    (case["runtime_input_length"], case["prototype_source"])
                    for case in fixture["cases"]
                ],
                [
                    (length, source)
                    for length in (4_096, 8_192, 16_384)
                    for source in ("current", "historical")
                ],
            )
            self.assertTrue(
                all(
                    len(case["expected"]["squared_public_class_distances"]) == 7
                    for case in fixture["cases"]
                )
            )

    def test_hash_mapping_coverage_and_sensitive_paths_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fusion_dir = Path(temporary) / "fusion"
            metrics = _write_smoke_fusion(fusion_dir)
            labels_path = fusion_dir / "fusion_prototype_labels.npy"
            labels = np.load(labels_path, allow_pickle=False)
            labels[labels == 6] = 5
            np.save(labels_path, labels)
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                load_verified_fusion(fusion_dir)

            changed = copy.deepcopy(metrics)
            changed["artifacts"]["prototype_labels_sha256"] = _sha256(
                labels_path
            )
            metadata_path = fusion_dir / "fusion_prototype_bank.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["prototype_public_class_indices"] = labels.tolist()
            for index, label in enumerate(labels):
                metadata["groups"][index]["public_class"] = PUBLIC_CLASSES[
                    int(label)
                ]
            _write_json(metadata_path, metadata)
            changed["artifacts"]["prototype_metadata_sha256"] = _sha256(
                metadata_path
            )
            _write_json(fusion_dir / "dev_metrics.json", changed)
            with self.assertRaisesRegex(ValueError, "missing public classes"):
                load_verified_fusion(fusion_dir)

        for path in (
            "/tmp/release/candidate",
            "/tmp/sealed/candidate",
            "/tmp/consumed_test/candidate",
        ):
            with self.assertRaisesRegex(ValueError, "refuses"):
                reject_sensitive_path(path, "test")
        with self.assertRaisesRegex(ValueError, "live model asset"):
            validate_empty_output(
                Path(__file__).resolve().parents[4]
                / "src"
                / "embedding"
                / "assets"
            )

    def test_exporter_rejects_self_consistent_current_label_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fusion_dir = Path(temporary) / "fusion"
            metrics = _write_smoke_fusion(fusion_dir)
            metadata_path = fusion_dir / "fusion_prototype_bank.json"
            labels_path = fusion_dir / "fusion_prototype_labels.npy"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            labels = np.load(labels_path, allow_pickle=False)
            wifi_index = next(
                index
                for index, group in enumerate(metadata["groups"])
                if group["source"] == "current"
                and group["profile"] == "wifi-hr-dsss-11m"
            )
            labels[wifi_index] = PUBLIC_CLASSES.index("bluetooth")
            metadata["prototype_public_class_indices"] = labels.tolist()
            metadata["groups"][wifi_index]["public_class"] = "bluetooth"
            np.save(labels_path, labels)
            _write_json(metadata_path, metadata)
            metrics["artifacts"]["prototype_labels_sha256"] = _sha256(
                labels_path
            )
            metrics["artifacts"]["prototype_metadata_sha256"] = _sha256(
                metadata_path
            )
            _write_json(fusion_dir / "dev_metrics.json", metrics)
            with self.assertRaisesRegex(
                ValueError, "wifi-hr-dsss-11m must map to dsss"
            ):
                load_verified_fusion(fusion_dir)

        with tempfile.TemporaryDirectory() as temporary:
            fusion_dir = Path(temporary) / "fusion"
            metrics = _write_smoke_fusion(fusion_dir)
            metadata_path = fusion_dir / "fusion_prototype_bank.json"
            labels_path = fusion_dir / "fusion_prototype_labels.npy"
            bank_path = fusion_dir / "fusion_prototype_bank.npy"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            labels = np.load(labels_path, allow_pickle=False)
            bank = np.load(bank_path, allow_pickle=False)
            missing_index = next(
                index
                for index, group in enumerate(metadata["groups"])
                if group["source"] == "current"
                and group["profile"] == "wifi6-he-tb"
            )
            metadata["groups"].pop(missing_index)
            metadata["prototype_public_class_indices"].pop(missing_index)
            labels = np.delete(labels, missing_index)
            bank = np.delete(bank, missing_index, axis=0)
            np.save(labels_path, labels)
            np.save(bank_path, bank)
            _write_json(metadata_path, metadata)
            for key, path in (
                ("prototype_labels", labels_path),
                ("prototype_bank", bank_path),
                ("prototype_metadata", metadata_path),
            ):
                metrics["artifacts"][f"{key}_sha256"] = _sha256(path)
            _write_json(fusion_dir / "dev_metrics.json", metrics)
            with self.assertRaisesRegex(ValueError, "exact 31-profile"):
                load_verified_fusion(fusion_dir)


if __name__ == "__main__":
    unittest.main()
