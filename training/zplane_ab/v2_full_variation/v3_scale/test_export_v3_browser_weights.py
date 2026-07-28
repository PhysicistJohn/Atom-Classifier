"""Focused safety tests for the v3 fusion browser-weight exporter."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import export_v3_browser_weights as subject


def valid_manifest() -> dict:
    return {
        "rejection": {
            "state": "unset",
            "external_staged_policy_required_for_abstention": True,
            "required_contract": {
                "scope": (
                    "stage two on stage-one survivors only; staged policy "
                    "assets are separate"
                )
            },
        },
        "release_blockers": [
            "development bundle, not release evidence",
            "passing validate-role artifact required",
            "passing sealed suite required",
        ],
    }


class ExternalStagedContractTests(unittest.TestCase):
    def test_current_external_contract_is_accepted(self) -> None:
        subject.validate_external_staged_rejection(valid_manifest())

    def test_fitted_legacy_slot_is_refused(self) -> None:
        manifest = valid_manifest()
        manifest["rejection"]["state"] = "fitted"
        with self.assertRaisesRegex(ValueError, "unset"):
            subject.validate_external_staged_rejection(manifest)

    def test_older_not_refit_placeholder_is_refused(self) -> None:
        manifest = valid_manifest()
        manifest["release_blockers"] = [
            "open-set rejector is not refit against these embeddings"
        ]
        with self.assertRaisesRegex(ValueError, "stale"):
            subject.validate_external_staged_rejection(manifest)

    def test_missing_stage_one_survivor_scope_is_refused(self) -> None:
        manifest = valid_manifest()
        manifest["rejection"]["required_contract"]["scope"] = "all rows"
        with self.assertRaisesRegex(ValueError, "stage-one survivors"):
            subject.validate_external_staged_rejection(manifest)


class OutputSafetyTests(unittest.TestCase):
    def test_fresh_or_empty_output_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(
                subject.validate_empty_output(root / "fresh"),
                (root / "fresh").resolve(),
            )
            empty = root / "empty"
            empty.mkdir()
            self.assertEqual(
                subject.validate_empty_output(empty), empty.resolve()
            )

    def test_nonempty_output_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            output.mkdir()
            (output / "old.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                subject.validate_empty_output(output)


if __name__ == "__main__":
    unittest.main()
