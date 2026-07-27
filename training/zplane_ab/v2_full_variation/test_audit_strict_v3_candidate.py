from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from audit_strict_v3_candidate import _development_output, _gate


class StrictV3ClosedAuditTest(unittest.TestCase):
    def test_gate_supports_min_and_max_without_rounding(self) -> None:
        self.assertTrue(_gate(0.8, 0.8)["passes"])
        self.assertFalse(_gate(0.799999999, 0.8)["passes"])
        self.assertTrue(_gate(0.78, 0.78, "max")["passes"])
        self.assertFalse(_gate(0.780000001, 0.78, "max")["passes"])
        with self.assertRaises(ValueError):
            _gate(1.0, 0.0, "equal")

    def test_output_refuses_release_and_nonempty_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "release/sealed"):
                _development_output(root / "releases" / "candidate")
            with self.assertRaisesRegex(ValueError, "release/sealed"):
                _development_output(root / "my-sealed-run")
            occupied = root / "occupied"
            occupied.mkdir()
            (occupied / "evidence").write_text("owned", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                _development_output(occupied)
            self.assertEqual(
                _development_output(root / "fresh"),
                (root / "fresh").resolve(),
            )


if __name__ == "__main__":
    unittest.main()
