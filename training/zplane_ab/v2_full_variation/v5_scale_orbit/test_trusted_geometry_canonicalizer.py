from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import trusted_geometry_canonicalizer as canonicalizer  # noqa: E402


REPO = HERE.parents[3]
PARITY_FIXTURE = (
    REPO
    / "src"
    / "embedding"
    / "test-fixtures"
    / "trusted-current-geometry-canonicalizer-v5.json"
)


def _continuous_signal(physical_native_time: np.ndarray) -> np.ndarray:
    t = np.asarray(physical_native_time, dtype=np.float64)
    return (
        0.72 * np.exp(2j * np.pi * 0.071 * t)
        + 0.21 * np.exp(-2j * np.pi * 0.133 * t + 0.2j)
        + (0.08 + 0.03j) * np.cos(2 * np.pi * 0.019 * t)
    ).astype(np.complex64)


def _parity_input(length: int) -> np.ndarray:
    index = np.arange(length, dtype=np.int64)
    in_phase = np.asarray(
        ((17 * index) % 257 - 128) / 64.0,
        dtype=np.float32,
    )
    quadrature = np.asarray(
        ((29 * index + 7) % 263 - 131) / 80.0,
        dtype=np.float32,
    )
    return np.asarray(in_phase + 1j * quadrature, dtype=np.complex64)


def _little_endian_interleaved_sha256(value: np.ndarray) -> str:
    interleaved = np.empty(2 * len(value), dtype="<f4")
    interleaved[0::2] = value.real
    interleaved[1::2] = value.imag
    return hashlib.sha256(interleaved.tobytes()).hexdigest()


class TrustedGeometryCanonicalizerTest(unittest.TestCase):
    def test_contract_and_exact_two_x_copy(self):
        source = _parity_input(canonicalizer.CONSUMED_INPUT_SAMPLES)
        output, context = canonicalizer.canonicalize(
            source,
            sample_rate_hz=16_000_000,
            native_sample_rate_hz=8_000_000,
        )
        self.assertEqual(output.dtype, np.dtype(np.complex64))
        self.assertEqual(output.shape, (canonicalizer.OUTPUT_SAMPLES,))
        self.assertTrue(np.array_equal(output, source))
        self.assertEqual(context["maximum_input_position"], 4095.0)
        self.assertEqual(context["input_samples_per_output"], 1.0)
        self.assertFalse(context["uses_frequency_transform"])
        self.assertFalse(context["uses_selected_profile_or_class"])

        metadata = canonicalizer.canonicalizer_metadata()
        self.assertEqual(
            metadata["version"],
            canonicalizer.CANONICALIZER_VERSION,
        )
        self.assertEqual(metadata["trust_scope"], "trusted-current")
        self.assertEqual(metadata["interpolation_taps"], 6)
        self.assertEqual(metadata["output_dtype"], "complex64")
        self.assertFalse(metadata["uses_frequency_transform"])
        self.assertFalse(metadata["uses_selected_profile_or_class"])

    def test_fractional_index_formula_reconstructs_a_quintic(self):
        # Six-point Lagrange interpolation is exact for degree-five
        # polynomials before the required float32 input/output rounding.
        output_index = np.arange(canonicalizer.OUTPUT_SAMPLES, dtype=np.float64)
        expected_time = output_index / 2.0

        def polynomial(t: np.ndarray) -> np.ndarray:
            u = t / canonicalizer.OUTPUT_SAMPLES
            real = 0.2 + 0.7 * u - 0.3 * u**2 + 0.11 * u**5
            imaginary = -0.4 + 0.2 * u + 0.15 * u**3 - 0.03 * u**4
            return np.asarray(real + 1j * imaginary, dtype=np.complex64)

        expected = polynomial(expected_time)
        for ratio in (1.0, 1.25, 1.5, 1.999_999, 2.0):
            input_time = (
                np.arange(
                    canonicalizer.CONSUMED_INPUT_SAMPLES,
                    dtype=np.float64,
                )
                / ratio
            )
            source = polynomial(input_time)
            output, context = canonicalizer.canonicalize(
                source,
                sample_rate_hz=ratio * 8_000_000,
                native_sample_rate_hz=8_000_000,
            )
            self.assertLess(float(np.max(np.abs(output - expected))), 3e-7)
            self.assertLessEqual(
                float(context["maximum_input_position"]),
                canonicalizer.CONSUMED_INPUT_SAMPLES - 1,
            )

    def test_smooth_continuous_signal_is_scale_invariant(self):
        target_time = (
            np.arange(canonicalizer.OUTPUT_SAMPLES, dtype=np.float64) / 2.0
        )
        expected = _continuous_signal(target_time)
        outputs: list[np.ndarray] = []
        for ratio in (1.0, 1.25, 1.5, 2.0):
            input_time = (
                np.arange(
                    canonicalizer.CONSUMED_INPUT_SAMPLES,
                    dtype=np.float64,
                )
                / ratio
            )
            source = _continuous_signal(input_time)
            output, _context = canonicalizer.canonicalize(
                source,
                sample_rate_hz=ratio * 12_000_000,
                native_sample_rate_hz=12_000_000,
            )
            error = np.abs(output - expected)
            self.assertLess(float(np.max(error)), 0.0012)
            self.assertLess(
                float(np.sqrt(np.mean(np.square(error)))),
                0.00023,
            )
            outputs.append(output)
        for output in outputs:
            self.assertLess(
                float(np.max(np.abs(output - outputs[-1]))),
                0.0012,
            )

    def test_extra_tail_is_bit_exactly_ignored(self):
        rng = np.random.default_rng(20260729)
        prefix = (
            rng.standard_normal(canonicalizer.CONSUMED_INPUT_SAMPLES)
            + 1j
            * rng.standard_normal(canonicalizer.CONSUMED_INPUT_SAMPLES)
        ).astype(np.complex64)
        one = np.concatenate(
            (prefix, np.full(37, 90 + 70j, dtype=np.complex64))
        )
        two = np.concatenate(
            (prefix, np.full(811, -60 - 30j, dtype=np.complex64))
        )
        base, base_context = canonicalizer.canonicalize(
            prefix,
            sample_rate_hz=5_000_000,
            native_sample_rate_hz=4_000_000,
        )
        first, first_context = canonicalizer.canonicalize(
            one,
            sample_rate_hz=5_000_000,
            native_sample_rate_hz=4_000_000,
        )
        second, second_context = canonicalizer.canonicalize(
            two,
            sample_rate_hz=5_000_000,
            native_sample_rate_hz=4_000_000,
        )
        self.assertTrue(np.array_equal(base, first))
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(base_context["ignored_tail_samples"], 0)
        self.assertEqual(first_context["consumed_input_samples"], 4096)
        self.assertEqual(second_context["consumed_input_samples"], 4096)
        self.assertEqual(first_context["ignored_tail_samples"], 37)
        self.assertEqual(second_context["ignored_tail_samples"], 811)

    def test_invalid_inputs_and_metadata_fail_closed(self):
        valid = np.ones(
            canonicalizer.CONSUMED_INPUT_SAMPLES,
            dtype=np.complex64,
        )
        with self.assertRaises(ValueError):
            canonicalizer.canonicalize(
                valid[:-1],
                sample_rate_hz=1,
                native_sample_rate_hz=1,
            )
        with self.assertRaises(TypeError):
            canonicalizer.canonicalize(
                valid.real,
                sample_rate_hz=1,
                native_sample_rate_hz=1,
            )
        malformed = valid.copy()
        malformed[17] = np.complex64(np.nan + 1j)
        with self.assertRaises(ValueError):
            canonicalizer.canonicalize(
                malformed,
                sample_rate_hz=1,
                native_sample_rate_hz=1,
            )
        oversized = valid.astype(np.complex128)
        oversized[17] = complex(np.finfo(np.float64).max, 0.0)
        with self.assertRaises(ValueError):
            canonicalizer.canonicalize(
                oversized,
                sample_rate_hz=1,
                native_sample_rate_hz=1,
            )
        for sample_rate, native_rate in (
            (0, 1),
            (-1, 1),
            (np.nan, 1),
            (np.inf, 1),
            (1, 0),
            (1, np.nan),
            (0.999, 1),
            (2.001, 1),
            (True, 1),
            (1, False),
        ):
            with self.subTest(
                sample_rate=sample_rate,
                native_rate=native_rate,
            ):
                with self.assertRaises(ValueError):
                    canonicalizer.canonicalize(
                        valid,
                        sample_rate_hz=sample_rate,
                        native_sample_rate_hz=native_rate,
                    )
        with self.assertRaises(ValueError):
            canonicalizer.canonicalize(
                valid,
                sample_rate_hz=np.finfo(np.float64).max * 0.9375,
                native_sample_rate_hz=np.finfo(np.float64).max * 0.75,
            )

    def test_module_ast_has_no_frequency_transform_calls_or_imports(self):
        source = Path(canonicalizer.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [
                    alias.name
                    for alias in node.names
                ]
                if any("fft" in name.lower() for name in names):
                    forbidden.extend(names)
            if isinstance(node, ast.Call):
                function = node.func
                pieces: list[str] = []
                while isinstance(function, ast.Attribute):
                    pieces.append(function.attr)
                    function = function.value
                if isinstance(function, ast.Name):
                    pieces.append(function.id)
                call_name = ".".join(reversed(pieces))
                if "fft" in call_name.lower():
                    forbidden.append(call_name)
        self.assertEqual(forbidden, [])

    def test_python_output_matches_shared_browser_parity_digest(self):
        fixture = json.loads(PARITY_FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(
            fixture["schema"],
            "atomos.v5.trusted-current-geometry-canonicalizer.parity",
        )
        self.assertTrue(fixture["synthetic_only"])
        self.assertFalse(fixture["reads_held_or_sealed_payload"])
        source = _parity_input(canonicalizer.CONSUMED_INPUT_SAMPLES)
        for case in fixture["cases"]:
            output, _context = canonicalizer.canonicalize(
                source,
                sample_rate_hz=case["sample_rate_hz"],
                native_sample_rate_hz=case["native_sample_rate_hz"],
            )
            self.assertEqual(
                _little_endian_interleaved_sha256(output),
                case["output_cf32le_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
