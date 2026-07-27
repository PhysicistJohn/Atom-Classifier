from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from training.zplane_ab.v2_full_variation import (
    evaluate_unet_denoiser_release as release,
)


def _item(cls: str, index: int) -> dict[str, object]:
    return {
        "cls": cls,
        "profile": f"{cls}-{index}",
        "sampleRateHz": 1_000_000,
        "bandwidthHz": 100_000,
        "impaired": bool(index % 2),
        "startSampleIndex": index * 100,
        "snrDb": -1.0 if index % 4 == 1 else 24.0,
        "centreOffsetFrac": 0.0,
    }


class FrozenArtifactTests(unittest.TestCase):
    def test_authoring_sources_are_exactly_bound(self):
        self.assertEqual(
            release.sha256_file(release.DEFAULT_PREPROCESS_SOURCE),
            release.EXPECTED_SHA256["preprocess_source"],
        )
        self.assertEqual(
            release.sha256_file(release.DEFAULT_TARGET_SOURCE),
            release.EXPECTED_SHA256["target_source"],
        )

    @unittest.skipUnless(
        all(
            path.exists()
            for path in (
                release.DEFAULT_BUNDLE,
                release.DEFAULT_ONNX,
                release.DEFAULT_STAGING_MANIFEST,
                release.DEFAULT_PARITY_FIXTURE,
            )
        ),
        "local frozen U-Net artifacts are not present",
    )
    def test_preflight_keeps_staging_blocked_but_composite_exact(self):
        report, _session, mean, std = release.verify_frozen_artifacts()
        self.assertEqual(
            report["staging_integration"]["status"],
            "blocked_not_release_ready",
        )
        self.assertFalse(
            report["staging_integration"]["compatible_with_checkpoint"]
        )
        self.assertEqual(report["evaluation_composite"]["status"], "ready")
        self.assertEqual(mean.shape, (12,))
        self.assertEqual(std.shape, (12,))
        self.assertTrue(np.all(std > 0.0))
        self.assertLessEqual(
            max(report["live_onnx_parity_max_abs_error"].values()),
            report["onnx_parity_limit"],
        )

    @unittest.skipUnless(
        release.DEFAULT_BUNDLE.exists(),
        "local frozen U-Net bundle is not present",
    )
    def test_preflight_rejects_preprocessing_drift_before_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            changed = Path(temporary).resolve() / "preprocess.py"
            changed.write_bytes(
                release.DEFAULT_PREPROCESS_SOURCE.read_bytes() + b"\n# drift\n"
            )
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                release.verify_frozen_artifacts(
                    preprocess_source_path=changed
                )


class BootstrapTests(unittest.TestCase):
    def test_paired_bootstrap_is_deterministic(self):
        delta = np.linspace(-0.1, 0.3, 40)
        first = release.paired_bootstrap_ci(
            delta, label="determinism", replicates=500
        )
        second = release.paired_bootstrap_ci(
            delta, label="determinism", replicates=500
        )
        self.assertEqual(first, second)
        self.assertLess(first[0], float(delta.mean()))
        self.assertGreater(first[1], float(delta.mean()))

    def test_constant_paired_delta_has_exact_interval(self):
        delta = np.full(25, 0.125)
        low, high = release.paired_bootstrap_ci(
            delta, label="constant", replicates=200
        )
        self.assertEqual(low, 0.125)
        self.assertEqual(high, 0.125)

    def test_metric_summary_uses_same_row_pairs(self):
        passthrough = np.linspace(0.1, 0.5, 25)
        model = passthrough + 0.05
        summary = release.metric_summary(
            passthrough,
            model,
            np.ones(25, dtype=bool),
            label="paired",
        )
        self.assertEqual(summary["n"], 25)
        self.assertAlmostEqual(summary["paired_delta_mean"], 0.05)
        self.assertAlmostEqual(summary["paired_delta_ci95"][0], 0.05)
        self.assertAlmostEqual(summary["paired_delta_ci95"][1], 0.05)
        self.assertEqual(summary["paired_delta_win_rate"], 1.0)


class CorpusContractTests(unittest.TestCase):
    def _corpus(self) -> dict[str, object]:
        classes = ["a", "b"]
        items = [
            *[_item("a", index) for index in range(80)],
            *[_item("b", index + 80) for index in range(80)],
        ]
        return {
            "sampleCount": release.MATCHED_CAPTURE_LENGTH,
            "format": "cf32le-interleaved",
            "hasCleanPairs": True,
            "exactTargetPerClass": True,
            "targetPerClass": 80,
            "classes": classes,
            "count": len(items),
            "items": items,
        }

    def test_validates_exact_class_and_preprocessing_support(self):
        corpus = self._corpus()
        items, supported = release.validate_corpus_manifest(
            corpus,
            target_per_class=80,
            expected_classes=("a", "b"),
        )
        self.assertEqual(len(items), 160)
        self.assertTrue(bool(np.all(supported)))

        corpus["items"][0]["centreOffsetFrac"] = 0.49
        corpus["items"][0]["bandwidthHz"] = 100_000
        _, supported = release.validate_corpus_manifest(
            corpus,
            target_per_class=80,
            expected_classes=("a", "b"),
        )
        self.assertFalse(bool(supported[0]))

    def test_rejects_non_boolean_impairment_and_class_imbalance(self):
        corpus = self._corpus()
        corpus["items"][0]["impaired"] = 1
        with self.assertRaisesRegex(TypeError, "impaired must be boolean"):
            release.validate_corpus_manifest(
                corpus,
                target_per_class=80,
                expected_classes=("a", "b"),
            )

        corpus = self._corpus()
        corpus["items"][0]["cls"] = "b"
        with self.assertRaisesRegex(ValueError, "per-class counts"):
            release.validate_corpus_manifest(
                corpus,
                target_per_class=80,
                expected_classes=("a", "b"),
            )

    def test_slice_definitions_are_explicit_and_supported_only(self):
        items = [
            {
                "impaired": False,
                "snrDb": 40.0,
            },
            {
                "impaired": True,
                "snrDb": release.LOW_SNR_DB - 0.1,
            },
            {
                "impaired": True,
                "snrDb": release.HIGH_SNR_DB,
            },
            {
                "impaired": True,
                "snrDb": 30.0,
            },
        ]
        masks = release._slice_masks(
            items, np.asarray([True, True, True, False])
        )
        np.testing.assert_array_equal(
            masks["clean"], [True, False, False, False]
        )
        np.testing.assert_array_equal(
            masks["impaired"], [False, True, True, False]
        )
        np.testing.assert_array_equal(
            masks["impaired_low_snr"], [False, True, False, False]
        )
        np.testing.assert_array_equal(
            masks["impaired_high_snr"], [False, False, True, False]
        )

    def test_output_inside_sealed_release_is_rejected_before_artifact_use(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with self.assertRaisesRegex(ValueError, "outside the sealed"):
                release.evaluate_release(
                    root,
                    output_path=root / "denoiser.json",
                )


class StrictJsonTests(unittest.TestCase):
    def test_rejects_duplicate_and_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.json"
            path.write_text('{"a": 1, "a": 2}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate key"):
                release.load_json_strict(path)
            path.write_text('{"a": NaN}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                release.load_json_strict(path)


if __name__ == "__main__":
    unittest.main()
