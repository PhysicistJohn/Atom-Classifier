from __future__ import annotations

from pathlib import Path
import re
import sys
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import assemble_current_source_fusion as assembler  # noqa: E402
import run_current_source_dev as trainer  # noqa: E402


SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")


class V4ExecutedSourceProvenanceTests(unittest.TestCase):
    def test_trainer_hashes_every_directly_executed_local_source(self) -> None:
        paths = trainer._executed_source_paths()
        self.assertIn("v2/run_invariant_cnn_dev.py", paths)
        self.assertIn("training/train.py", paths)
        self.assertEqual(
            paths["v2/run_invariant_cnn_dev.py"],
            Path(trainer.v3_runner.__file__).resolve(),
        )
        self.assertEqual(
            paths["training/train.py"],
            trainer.TRAINING / "train.py",
        )
        self.assertTrue(all(path.is_file() for path in paths.values()))
        hashes = trainer._source_hashes()
        self.assertEqual(set(hashes), set(paths))
        self.assertTrue(
            all(SHA256_PATTERN.fullmatch(value) for value in hashes.values())
        )

    def test_branch_hash_contract_exactly_matches_trainer_snapshot(self) -> None:
        self.assertEqual(
            assembler._expected_branch_source_hashes(),
            trainer._source_hashes(),
        )
        assembly_hashes = assembler._source_hashes()
        self.assertIn("v2/run_invariant_cnn_dev.py", assembly_hashes)
        self.assertIn("training/train.py", assembly_hashes)

    def test_trainer_rejects_a_mid_run_source_change(self) -> None:
        expected = trainer._source_hashes()
        observed = dict(expected)
        observed["training/train.py"] = "0" * 64
        with self.assertRaisesRegex(
            RuntimeError,
            r"changed during v4 branch training: training/train\.py",
        ):
            trainer._assert_source_snapshot_unchanged(
                expected,
                observed,
                operation="v4 branch training",
            )

    def test_assembler_rejects_added_removed_or_changed_sources(self) -> None:
        expected = assembler._source_hashes()
        observed = dict(expected)
        observed.pop("training/train.py")
        observed["unexpected.py"] = "0" * 64
        with self.assertRaisesRegex(
            RuntimeError,
            "unexpected.py",
        ):
            assembler._assert_source_snapshot_unchanged(
                expected,
                observed,
                operation="v4 fusion assembly",
            )


if __name__ == "__main__":
    unittest.main()
