"""Unit tests for the STAGED v3 open-set rejector.

These tests never touch the development corpus, the consumed test half, or any
sealed release suite.  Every population is synthetic, small, and constructed in
the test.  No spent, sealed or release novelty seed is used except where the
test exists to prove that such a seed is refused.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fit_v3_openset as base  # noqa: E402
import fit_v3_openset_staged as subject  # noqa: E402
import noise_prefilter as prefilter  # noqa: E402
import pose_degeneracy as pdg  # noqa: E402
import time_domain_geometry as geometry  # noqa: E402
import v3_time_domain_openset as frozen  # noqa: E402
from test_fit_v3_openset import FusionArtifactValidationTests  # noqa: E402


PATCH_COUNT = frozen.PATCH_COUNT
PATCH_LENGTH = frozen.PATCH_LENGTH
PACKED_WIDTH = PATCH_COUNT * PATCH_LENGTH

DESIGN_SEED = subject.PROPOSED_DESIGN_NOVELTY_SEED
VALIDATION_SEED = subject.DEFAULT_VALIDATION_NOVELTY_SEEDS[0]
FITTING_SEED = prefilter.PROPOSED_FITTING_SEED

# Short lengths so the tests stay fast.  The prefilter is length dependent, so
# a bundle is fitted for each of these plus the known-population length.
TEST_LENGTHS = (1024, 2048)
KNOWN_LENGTH = 2048


def _packed(rows: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(rows, 2, PACKED_WIDTH)).astype(np.float32)


def _pose_rows(rows: int, seed: int, *, shift: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(rows, pdg.FEATURE_COUNT)) + float(shift)
    ).astype(np.float64)


def _bundle_set(
    directory: Path,
    *,
    lengths=(*TEST_LENGTHS, KNOWN_LENGTH),
    budget: float = 0.05,
) -> Path:
    """A real, fitted, thresholded prefilter set, built with the real module."""
    models = {}
    for index, length in enumerate(sorted(set(int(item) for item in lengths))):
        known = _pose_rows(120, 1000 + index)
        noise = _pose_rows(120, 2000 + index, shift=4.0)
        model = prefilter.fit_prefilter(
            known,
            noise,
            capture_length=length,
            known_population=prefilter.TRAIN_SPLIT,
            fitting_seed=FITTING_SEED,
        )
        models[length] = prefilter.fit_threshold(
            model,
            _pose_rows(80, 3000 + index),
            population=prefilter.ENROLLMENT_SPLIT,
            budget=budget,
        )
    prefilter.save_prefilter_set(directory, models)
    return Path(directory)


def _sources():
    return (
        subject.load_prefilter_module(),
        subject.load_posedegen_module(),
    )


def _fake_module(name: str, **attributes) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def _cleanup_fakes() -> None:
    for key in list(sys.modules):
        if key.startswith("broken_"):
            del sys.modules[key]


# ---------------------------------------------------------------------------


class SiblingModuleContractTests(unittest.TestCase):
    def tearDown(self) -> None:
        _cleanup_fakes()

    def test_an_absent_prefilter_fails_loudly_and_names_the_contract(self) -> None:
        with self.assertRaises(RuntimeError) as caught:
            subject.load_prefilter_module("no_such_noise_prefilter_module")
        message = str(caught.exception)
        self.assertIn("not importable", message)
        self.assertIn("will not silently degrade", message)
        self.assertIn("load_prefilter_set", message)

    def test_an_absent_posedegen_fails_loudly(self) -> None:
        with self.assertRaises(RuntimeError) as caught:
            subject.load_posedegen_module("no_such_pose_degeneracy_module")
        self.assertIn("not importable", str(caught.exception))
        self.assertIn("pose_degeneracy_features", str(caught.exception))

    def test_the_imports_are_lazy(self) -> None:
        """Importing this module must not require either sibling to exist."""
        source = Path(subject.__file__).read_text(encoding="utf-8")
        head = source.split("# ---", 1)[0]
        self.assertNotIn("import noise_prefilter", head)
        self.assertNotIn("import pose_degeneracy", head)
        self.assertIn("importlib.import_module", source)

    def test_the_real_prefilter_module_satisfies_the_contract(self) -> None:
        source = subject.load_prefilter_module()
        self.assertEqual(source.module_name, subject.PREFILTER_MODULE_DEFAULT)
        self.assertEqual(source.version, prefilter.NOISE_PREFILTER_VERSION)
        self.assertEqual(source.feature_names, tuple(prefilter.PREFILTER_FEATURES))
        self.assertEqual(len(source.source_sha256 or ""), 64)

    def test_the_real_posedegen_module_satisfies_the_contract(self) -> None:
        source = subject.load_posedegen_module()
        self.assertEqual(source.feature_count, pdg.FEATURE_COUNT)
        self.assertEqual(len(source.source_sha256 or ""), 64)

    def test_missing_symbols_are_listed(self) -> None:
        _fake_module("broken_pf", NOISE_PREFILTER_VERSION="x")
        with self.assertRaises(RuntimeError) as caught:
            subject.load_prefilter_module("broken_pf")
        self.assertIn("PREFILTER_FEATURES", str(caught.exception))

    def test_a_non_callable_loader_is_refused(self) -> None:
        _fake_module(
            "broken_call",
            NOISE_PREFILTER_VERSION="x",
            PREFILTER_FEATURES=("a",),
            ARCHITECTURE_CONTRACT_CHANGE="c",
            CONTRACT_KEYS_FOR_RELEASE_CLAIM=("additive_only",),
            NoisePrefilter=object,
            load_prefilter_set="nope",
            prefilter_set_sha256=lambda path: "",
            compute_saving=lambda gated: {},
            validate_fitting_seed=lambda seed: seed,
        )
        with self.assertRaisesRegex(RuntimeError, "non-callable"):
            subject.load_prefilter_module("broken_call")

    def test_a_string_or_empty_feature_list_is_refused(self) -> None:
        common = {
            "NOISE_PREFILTER_VERSION": "x",
            "ARCHITECTURE_CONTRACT_CHANGE": "c",
            "CONTRACT_KEYS_FOR_RELEASE_CLAIM": ("additive_only",),
            "NoisePrefilter": object,
            "load_prefilter_set": lambda path: {},
            "prefilter_set_sha256": lambda path: "",
            "compute_saving": lambda gated: {},
            "validate_fitting_seed": lambda seed: seed,
        }
        _fake_module("broken_str", PREFILTER_FEATURES="abc", **common)
        with self.assertRaisesRegex(RuntimeError, "not a sequence"):
            subject.load_prefilter_module("broken_str")
        _fake_module("broken_dup", PREFILTER_FEATURES=("a", "a"), **common)
        with self.assertRaisesRegex(RuntimeError, "unique"):
            subject.load_prefilter_module("broken_dup")

    def test_an_empty_version_or_module_name_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            subject.load_prefilter_module("")
        with self.assertRaisesRegex(ValueError, "non-empty"):
            subject.load_posedegen_module("")

    def test_a_wrong_width_or_non_finite_feature_vector_is_refused(self) -> None:
        _, posedegen = _sources()
        for factory, pattern in (
            (lambda iq, **_k: np.zeros(3, dtype=np.float64), "expected"),
            (
                lambda iq, **_k: np.full(pdg.FEATURE_COUNT, np.nan),
                "non-finite",
            ),
        ):
            broken = subject.PoseDegeneracySource(
                module_name=posedegen.module_name,
                module=types.SimpleNamespace(pose_degeneracy_features=factory),
                version=posedegen.version,
                feature_names=posedegen.feature_names,
                source_path=None,
                source_sha256=None,
            )
            with self.assertRaisesRegex(RuntimeError, pattern):
                subject.prefilter_matrix(
                    broken,
                    [np.ones(256, dtype=np.complex128)],
                    patch_length=PATCH_LENGTH,
                    target_frac=0.5,
                    population="probe",
                )

    def test_an_empty_population_is_refused(self) -> None:
        _, posedegen = _sources()
        with self.assertRaisesRegex(ValueError, "at least one row"):
            subject.prefilter_matrix(
                posedegen,
                [],
                patch_length=PATCH_LENGTH,
                target_frac=0.5,
                population="probe",
            )

    def test_features_are_computed_at_the_frontend_resolution_floor(self) -> None:
        self.assertTrue(subject.PREFILTER_MATCH_FRONTEND_MIN_BANDWIDTH)
        self.assertEqual(
            subject.frontend_min_bandwidth(4096, PATCH_LENGTH, 0.5),
            geometry.minimum_bandwidth_for_active_span(4096, PATCH_LENGTH, 0.5),
        )

    def test_the_floor_is_passed_through_to_the_extractor(self) -> None:
        seen: list[float] = []

        def features(iq, *, min_bandwidth=None, **_kwargs):
            seen.append(min_bandwidth)
            return np.zeros(pdg.FEATURE_COUNT, dtype=np.float64)

        _, posedegen = _sources()
        probe = subject.PoseDegeneracySource(
            module_name=posedegen.module_name,
            module=types.SimpleNamespace(pose_degeneracy_features=features),
            version=posedegen.version,
            feature_names=posedegen.feature_names,
            source_path=None,
            source_sha256=None,
        )
        subject.prefilter_row(
            probe,
            np.ones(4096, dtype=np.complex128),
            patch_length=PATCH_LENGTH,
            target_frac=0.5,
        )
        self.assertEqual(
            seen,
            [geometry.minimum_bandwidth_for_active_span(4096, PATCH_LENGTH, 0.5)],
        )

    def test_the_full_pose_vector_is_handed_to_the_model(self) -> None:
        """The model slices its own columns; this module must not pre-slice."""
        _, posedegen = _sources()
        matrix = subject.prefilter_matrix(
            posedegen,
            [np.ones(1024, dtype=np.complex128)],
            patch_length=PATCH_LENGTH,
            target_frac=0.5,
            population="probe",
        )
        self.assertEqual(matrix.shape, (1, pdg.FEATURE_COUNT))


class InheritedContractTests(unittest.TestCase):
    """The gates and the frozen policy are imported, never re-typed."""

    def test_gate_floors_are_the_base_objects(self) -> None:
        self.assertIs(subject.GATE_FLOORS, base.GATE_FLOORS)
        self.assertIs(subject.KNOWN_FUR_CEILING, base.KNOWN_FUR_CEILING)

    def test_gate_floors_are_the_frozen_values(self) -> None:
        self.assertEqual(
            subject.GATE_FLOORS,
            {
                "overall_auroc": 0.72,
                "noise_auroc": 0.80,
                "chirp_auroc": 0.80,
                "noise_threshold_recall": 0.10,
                "chirp_threshold_recall": 0.10,
            },
        )
        self.assertEqual(subject.KNOWN_FUR_CEILING, 0.10)

    def test_the_prefilter_module_reads_the_same_live_floors(self) -> None:
        floors = prefilter.gate_floors()
        self.assertEqual(floors["noise_auroc"], subject.GATE_FLOORS["noise_auroc"])
        self.assertEqual(
            floors["noise_threshold_recall"],
            subject.GATE_FLOORS["noise_threshold_recall"],
        )
        self.assertEqual(
            floors["known_false_unknown_ceiling"], subject.KNOWN_FUR_CEILING
        )

    def test_stage_two_weights_and_quantile_are_the_frozen_constants(self) -> None:
        self.assertEqual(subject.BRANCH_LOF_RANK_WEIGHT, frozen.FROZEN_V2_WEIGHT)
        self.assertEqual(
            subject.GEOMETRY_RANK_WEIGHT, frozen.FROZEN_GEOMETRY_WEIGHT
        )
        self.assertEqual(
            subject.THRESHOLD_QUANTILE, frozen.FROZEN_THRESHOLD_QUANTILE
        )

    def test_stage_two_is_the_base_code(self) -> None:
        self.assertIs(subject.fit_stage_two_policy, base.fit_policy)
        self.assertIs(subject.fit_branch_lof, base.fit_branch_lof)
        self.assertIs(
            subject.assert_closed_label_unchanged,
            base.assert_closed_label_unchanged,
        )
        self.assertIs(subject.gate_summary, base.gate_summary)
        self.assertIs(subject.Rejector, base.Rejector)

    def test_the_branch_lof_shape_is_inherited_and_unflagged(self) -> None:
        self.assertIs(subject.BRANCH_LOF, base.BRANCH_LOF)
        flags = {action.dest for action in subject.build_parser()._actions}
        self.assertNotIn("branch_lof", flags)
        self.assertNotIn("neighbors", flags)

    def test_release_and_sealed_paths_are_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "release or sealed"):
            subject.reject_sealed_path(
                Path("/tmp/atom/training/artifacts/releases/staged"), "output"
            )


class SeedLedgerTests(unittest.TestCase):
    def test_the_spent_ledger_is_complete(self) -> None:
        self.assertEqual(
            sorted(subject.SPENT_NOVELTY_SEEDS),
            [20260938, 20260939, 20260940, 20260941],
        )
        self.assertEqual(subject.FIRST_CLEAN_NOVELTY_SEED, 20260942)
        self.assertEqual(subject.SEALED_RELEASE_SEED, 20260729)
        self.assertEqual(subject.RELEASE_SEED_NEVER_SPENT_HERE, 20260731)

    def test_the_ledger_agrees_with_the_prefilter_module(self) -> None:
        self.assertEqual(
            sorted(subject.SPENT_NOVELTY_SEEDS),
            sorted(prefilter.SPENT_NOVELTY_SEEDS),
        )
        self.assertEqual(
            subject.FIRST_CLEAN_NOVELTY_SEED,
            prefilter.CLEAN_VALIDATION_SEEDS_FROM,
        )

    def test_every_spent_seed_is_refused_as_novelty(self) -> None:
        for seed in subject.SPENT_NOVELTY_SEEDS:
            with self.assertRaisesRegex(ValueError, "SPENT"):
                subject.validate_novelty_seeds([seed])

    def test_every_spent_seed_is_refused_as_a_design_seed(self) -> None:
        for seed in subject.SPENT_NOVELTY_SEEDS:
            with self.assertRaisesRegex(ValueError, "SPENT"):
                subject.validate_seed_plan("validate", seed, [VALIDATION_SEED])

    def test_the_sealed_and_release_seeds_are_refused(self) -> None:
        for seed in (20260729, 20260731):
            with self.assertRaises(ValueError):
                subject.validate_novelty_seeds([seed])
            with self.assertRaises(ValueError):
                subject.validate_seed_plan("validate", seed, [VALIDATION_SEED])

    def test_a_clean_seed_is_accepted(self) -> None:
        self.assertEqual(
            subject.validate_novelty_seeds([20260943, 20260944]),
            (20260943, 20260944),
        )

    def test_duplicate_and_empty_seed_sets_are_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "distinct"):
            subject.validate_novelty_seeds([20260943, 20260943])
        with self.assertRaisesRegex(ValueError, "distinct"):
            subject.validate_novelty_seeds([])

    def test_validate_refuses_design_and_novelty_overlap(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot"):
            subject.validate_seed_plan("validate", 20260945, [20260945])

    def test_validate_accepts_the_default_pair(self) -> None:
        role, design, seeds = subject.validate_seed_plan(
            "validate", DESIGN_SEED, subject.DEFAULT_VALIDATION_NOVELTY_SEEDS
        )
        self.assertEqual(role, "validate")
        self.assertEqual(design, DESIGN_SEED)
        self.assertEqual(seeds, subject.DEFAULT_VALIDATION_NOVELTY_SEEDS)

    def test_a_default_validation_seed_may_not_be_the_design_seed(self) -> None:
        with self.assertRaisesRegex(ValueError, "evidence rule 4"):
            subject.validate_seed_plan("validate", VALIDATION_SEED, [20260945])

    def test_design_refuses_the_reserved_validation_seeds(self) -> None:
        with self.assertRaisesRegex(ValueError, "stay untouched"):
            subject.validate_seed_plan(
                "design", DESIGN_SEED, [DESIGN_SEED, VALIDATION_SEED]
            )

    def test_design_must_draw_its_own_declared_seed(self) -> None:
        with self.assertRaisesRegex(ValueError, "must draw novelty"):
            subject.validate_seed_plan("design", DESIGN_SEED, [20260945])
        self.assertEqual(
            subject.validate_seed_plan("design", DESIGN_SEED, [DESIGN_SEED]),
            ("design", DESIGN_SEED, (DESIGN_SEED,)),
        )

    def test_an_absent_or_non_integer_design_seed_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "required"):
            subject.validate_seed_plan("validate", None, [VALIDATION_SEED])
        for value in (True, 2.0, "20260942"):
            with self.assertRaisesRegex(ValueError, "must be an integer"):
                subject.validate_seed_plan("validate", value, [VALIDATION_SEED])

    def test_an_unknown_role_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "--role must be one of"):
            subject.validate_role("ship")

    def test_novelty_seed_defaults_are_role_dependent(self) -> None:
        self.assertEqual(
            subject.resolve_novelty_seeds(
                argparse.Namespace(role="validate", novelty_seeds=None)
            ),
            list(subject.DEFAULT_VALIDATION_NOVELTY_SEEDS),
        )
        with self.assertRaisesRegex(ValueError, "never a design default"):
            subject.resolve_novelty_seeds(
                argparse.Namespace(role="design", novelty_seeds=None)
            )

    def test_the_ledger_note_names_every_spent_seed(self) -> None:
        for seed in subject.SPENT_NOVELTY_SEEDS:
            self.assertIn(str(seed), subject.SEED_LEDGER_NOTE)
        self.assertIn("evidence rule 4", subject.SEED_LEDGER_NOTE)


class ArchitectureContractTests(unittest.TestCase):
    """The behavioural change must be stated where a reader will meet it."""

    def test_the_module_docstring_states_the_change(self) -> None:
        self.assertIn("ARCHITECTURE CONTRACT CHANGE", subject.__doc__)
        self.assertIn("closed_label_can_be_gated", subject.__doc__)

    def test_the_note_says_no_label_is_produced(self) -> None:
        note = subject.ARCHITECTURE_CONTRACT_NOTE
        self.assertIn("no closed-set label", note)
        self.assertIn("suppress", note)
        self.assertIn("known false-unknown", note)

    def test_the_parser_epilog_carries_the_note(self) -> None:
        epilog = subject.build_parser().epilog
        self.assertIn("suppress the label", epilog)
        self.assertIn("--prefilter-dir", epilog)


class StageOneLoadingTests(unittest.TestCase):
    def _load(self, root: Path, **kwargs):
        directory = _bundle_set(root / "prefilter", **kwargs)
        source, posedegen = _sources()
        return subject.load_stage_one(
            source,
            posedegen,
            directory,
            required_lengths=(*TEST_LENGTHS, KNOWN_LENGTH),
        )

    def test_a_fitted_set_loads_and_records_its_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            gate = self._load(Path(root))
            self.assertEqual(
                gate.lengths, tuple(sorted({*TEST_LENGTHS, KNOWN_LENGTH}))
            )
            self.assertEqual(len(gate.set_sha256), 64)
            provenance = gate.provenance()
        self.assertFalse(provenance["fitted_here"])
        for length in provenance["models"]:
            metadata = provenance["models"][length]
            self.assertFalse(metadata["additive_only"])
            self.assertTrue(metadata["changes_closed_label"])
            self.assertTrue(metadata["gates_before_classification"])

    def test_a_missing_length_is_refused_rather_than_substituted(self) -> None:
        """A gap BELOW the longest fitted length has no causal prefix bundle."""
        with tempfile.TemporaryDirectory() as root:
            directory = _bundle_set(
                Path(root) / "prefilter", lengths=(1024, 4096)
            )
            source, posedegen = _sources()
            with self.assertRaisesRegex(ValueError, "no fitted prefilter"):
                subject.load_stage_one(
                    source, posedegen, directory, required_lengths=(1024, 2048)
                )

    def test_a_required_length_above_the_fitted_set_loads_by_prefix_rule(
        self,
    ) -> None:
        """The sealed suite's N32768 case: required above max fitted loads."""
        with tempfile.TemporaryDirectory() as root:
            directory = _bundle_set(Path(root) / "prefilter")
            source, posedegen = _sources()
            gate = subject.load_stage_one(
                source,
                posedegen,
                directory,
                required_lengths=(*TEST_LENGTHS, KNOWN_LENGTH, 4096),
            )
            self.assertEqual(
                gate.lengths, tuple(sorted({*TEST_LENGTHS, KNOWN_LENGTH}))
            )
            self.assertEqual(
                gate.feature_length_for(4096), gate.max_fitted_length
            )

    def test_scoring_below_the_smallest_fitted_length_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            gate = self._load(Path(root))
            with self.assertRaisesRegex(KeyError, "length dependent"):
                gate.raw(_pose_rows(3, 7), 512)

    def test_a_gap_length_between_fitted_lengths_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            gate = self._load(Path(root))
            with self.assertRaisesRegex(KeyError, "causal-prefix"):
                gate.feature_length_for(1536)

    def test_a_length_above_the_longest_fitted_uses_the_prefix_bundle(self) -> None:
        """The causal-prefix rule: above max fitted, gate with the max bundle."""
        with tempfile.TemporaryDirectory() as root:
            gate = self._load(Path(root))
            longest = max(gate.lengths)
            self.assertEqual(gate.max_fitted_length, longest)
            self.assertEqual(gate.feature_length_for(4096), longest)
            self.assertEqual(gate.feature_length_for(longest), longest)
            features = _pose_rows(12, 41, shift=1.0)
            np.testing.assert_array_equal(
                gate.raw(features, 4096), gate.raw(features, longest)
            )
            np.testing.assert_array_equal(
                gate.fires(features, 4096), gate.fires(features, longest)
            )

    def test_prefix_captures_slice_to_the_max_fitted_length(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            gate = self._load(Path(root))
            longest = max(gate.lengths)
            rng = np.random.default_rng(5)
            captures = [
                rng.standard_normal(4096) + 1j * rng.standard_normal(4096)
                for _ in range(3)
            ]
            sliced, feature_length = subject.stage_one_prefix_captures(
                gate, captures, 4096
            )
            self.assertEqual(feature_length, longest)
            for original, prefix in zip(captures, sliced):
                self.assertEqual(len(prefix), longest)
                np.testing.assert_array_equal(prefix, original[:longest])
            # A fitted length passes through unsliced.
            same, same_length = subject.stage_one_prefix_captures(
                gate, [row[:longest] for row in captures], longest
            )
            self.assertEqual(same_length, longest)
            for original, unchanged in zip(captures, same):
                np.testing.assert_array_equal(unchanged, original[:longest])
            # A capture whose true length contradicts the declared one is
            # refused rather than silently truncated.
            with self.assertRaisesRegex(ValueError, "declared capture length"):
                subject.stage_one_prefix_captures(
                    gate, [captures[0][:2000]], 4096
                )

    def test_the_provenance_records_the_prefix_rule(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            gate = self._load(Path(root))
            block = gate.provenance()["stage_one_prefix_rule"]
            self.assertEqual(
                block["max_fitted_length"], max(gate.lengths)
            )
            self.assertEqual(block["rule"], subject.STAGE_ONE_PREFIX_RULE)
            self.assertIn("FIRST", block["rule"])
            self.assertIn("BELOW", block["rule"])

    def test_a_bundle_without_an_operating_point_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "prefilter"
            model = prefilter.fit_prefilter(
                _pose_rows(60, 1),
                _pose_rows(60, 2, shift=4.0),
                capture_length=1024,
                known_population=prefilter.TRAIN_SPLIT,
                fitting_seed=FITTING_SEED,
            )
            prefilter.save_prefilter_set(directory, {1024: model})
            source, posedegen = _sources()
            with self.assertRaisesRegex(RuntimeError, "no operating point"):
                subject.load_stage_one(
                    source, posedegen, directory, required_lengths=(1024,)
                )

    def test_a_release_or_sealed_prefilter_directory_is_refused(self) -> None:
        source, posedegen = _sources()
        with self.assertRaisesRegex(ValueError, "release or sealed"):
            subject.load_stage_one(
                source,
                posedegen,
                Path("/tmp/atom/training/artifacts/releases/prefilter"),
                required_lengths=(1024,),
            )

    def test_an_empty_directory_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            source, posedegen = _sources()
            with self.assertRaises(ValueError):
                subject.load_stage_one(
                    source, posedegen, Path(root), required_lengths=(1024,)
                )

    def test_the_gate_rule_is_cross_checked_against_the_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            gate = self._load(Path(root))
            features = _pose_rows(20, 99, shift=2.0)
            fired = gate.fires(features, TEST_LENGTHS[0])
            expected = gate.raw(features, TEST_LENGTHS[0]) >= float(
                gate.model_for(TEST_LENGTHS[0]).threshold_score
            )
            np.testing.assert_array_equal(fired, expected)

    def test_a_model_whose_decide_disagrees_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            gate = self._load(Path(root))
            model = gate.model_for(TEST_LENGTHS[0])
            tampered = types.SimpleNamespace(
                score=model.score,
                decide=lambda features, capture_length=None: np.zeros(
                    len(np.asarray(features)), dtype=bool
                ),
                threshold_score=-1e9,
                capture_length=model.capture_length,
                feature_names=model.feature_names,
            )
            broken = subject.StageOneGate(
                prefilter=gate.prefilter,
                posedegen=gate.posedegen,
                directory=gate.directory,
                models={TEST_LENGTHS[0]: tampered},
                set_sha256=gate.set_sha256,
            )
            with self.assertRaisesRegex(AssertionError, "gate rule changed"):
                broken.fires(_pose_rows(6, 5, shift=6.0), TEST_LENGTHS[0])

    def test_the_bounded_rank_is_strictly_increasing_and_bounded(self) -> None:
        values = np.array([-1e15, -40.0, -1.0, 0.0, 1.0, 40.0, 1e15])
        ranks = subject._bounded_rank(values)
        self.assertTrue(np.all(np.diff(ranks) > 0.0))
        self.assertTrue(np.all(ranks > 0.0) and np.all(ranks < 1.0))

    def test_the_bounded_rank_does_not_saturate_where_a_logistic_would(self) -> None:
        """Log-odds past ~37 saturate a float64 logistic; softsign must not."""
        ranks = subject._bounded_rank(np.array([40.0, 60.0, 100.0]))
        self.assertTrue(np.all(np.diff(ranks) > 0.0))

    def test_a_non_finite_score_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            subject._bounded_rank(np.array([np.inf]))


def _outcome(
    gated,
    stage_two,
    *,
    feature_seconds=0.0,
    stage_one_seconds=0.0,
    stage_two_seconds=0.0,
) -> "subject.StagedOutcome":
    fired = np.asarray(gated, dtype=bool)
    two = np.asarray(stage_two, dtype=np.float64)
    raw = np.where(fired, 5.0, -5.0)
    rank = subject._bounded_rank(raw)
    staged = np.where(fired, subject.STAGE_ONE_SCORE_OFFSET + rank, two)
    return subject.StagedOutcome(
        gated=fired,
        stage_one_raw=raw,
        stage_one_rank=rank,
        stage_two_score=two,
        staged_score=staged,
        survivor_index=np.flatnonzero(~fired),
        feature_seconds=feature_seconds,
        stage_one_seconds=stage_one_seconds,
        stage_two_seconds=stage_two_seconds,
    )


class StagedDecisionTests(unittest.TestCase):
    """A single staged axis must reproduce the two-stage decision exactly."""

    def test_gated_rows_sit_above_every_ungated_row(self) -> None:
        outcome = _outcome([True, False, False], [np.nan, 0.99, 0.10])
        self.assertTrue(
            outcome.staged_score[0] > outcome.staged_score[1:].max()
        )
        self.assertGreaterEqual(
            outcome.staged_score[0], subject.STAGE_ONE_SCORE_OFFSET
        )

    def test_the_decision_is_equivalent_at_any_threshold(self) -> None:
        gated = np.array([True, False, False, True, False])
        stage_two = np.array([np.nan, 0.99, 0.10, np.nan, 0.50])
        outcome = _outcome(gated, stage_two)
        for threshold in (0.05, 0.5, 0.95, 0.999):
            subject.assert_staged_decision_equivalence(
                gated, stage_two, outcome.staged_score, threshold
            )
            np.testing.assert_array_equal(
                outcome.staged_score > threshold,
                gated | (np.nan_to_num(stage_two, nan=-1.0) > threshold),
            )

    def test_a_gated_row_carrying_a_stage_two_score_is_refused(self) -> None:
        """That would mean the short circuit did not happen."""
        gated = np.array([True, False])
        stage_two = np.array([0.4, 0.2])
        staged = np.array([1.5, 0.2])
        with self.assertRaisesRegex(AssertionError, "short circuit did not"):
            subject.assert_staged_decision_equivalence(
                gated, stage_two, staged, 0.5
            )

    def test_a_survivor_without_a_stage_two_score_is_refused(self) -> None:
        with self.assertRaisesRegex(AssertionError, "no stage-2 score"):
            subject.assert_staged_decision_equivalence(
                np.array([False]), np.array([np.nan]), np.array([0.2]), 0.5
            )

    def test_a_misaligned_or_non_finite_staged_score_is_refused(self) -> None:
        with self.assertRaisesRegex(AssertionError, "not aligned"):
            subject.assert_staged_decision_equivalence(
                np.array([True]), np.array([np.nan]), np.array([1.1, 1.2]), 0.5
            )
        with self.assertRaisesRegex(AssertionError, "non-finite"):
            subject.assert_staged_decision_equivalence(
                np.array([True]), np.array([np.nan]), np.array([np.nan]), 0.5
            )

    def test_a_staged_score_that_breaks_the_decision_is_refused(self) -> None:
        with self.assertRaisesRegex(AssertionError, "does not reproduce"):
            subject.assert_staged_decision_equivalence(
                np.array([True, False]),
                np.array([np.nan, 0.9]),
                np.array([0.1, 0.9]),
                0.5,
            )


class SubsetAgreementTests(unittest.TestCase):
    def test_identical_survivor_scores_are_exact(self) -> None:
        outcome = _outcome([True, False, False], [np.nan, 0.4, 0.6])
        report = subject.subset_agreement(
            outcome, np.array([0.9, 0.4, 0.6])
        )
        self.assertTrue(report["exact"])
        self.assertEqual(report["survivor_rows"], 2)

    def test_a_disagreeing_survivor_is_refused(self) -> None:
        outcome = _outcome([False, False], [0.4, 0.6])
        with self.assertRaisesRegex(AssertionError, "not comparable"):
            subject.subset_agreement(outcome, np.array([0.4, 0.9]))

    def test_a_fully_gated_population_agrees_trivially(self) -> None:
        outcome = _outcome([True, True], [np.nan, np.nan])
        report = subject.subset_agreement(outcome, np.array([0.1, 0.2]))
        self.assertEqual(report["survivor_rows"], 0)
        self.assertTrue(report["exact"])

    def test_misaligned_unstaged_scores_are_refused(self) -> None:
        outcome = _outcome([False], [0.4])
        with self.assertRaisesRegex(AssertionError, "not aligned"):
            subject.subset_agreement(outcome, np.array([0.4, 0.5]))


class ComputeAccountingTests(unittest.TestCase):
    def test_the_short_circuited_fraction_is_measured(self) -> None:
        outcome = _outcome(
            [True, True, False, False],
            [np.nan, np.nan, 0.2, 0.3],
            feature_seconds=0.5,
            stage_one_seconds=0.1,
            stage_two_seconds=1.0,
        )
        report = subject.compute_accounting(outcome, 2.0)
        self.assertEqual(report["rows"], 4)
        self.assertEqual(report["short_circuited"], 2)
        self.assertEqual(report["short_circuited_fraction"], 0.5)
        self.assertEqual(report["stage_two_rows_scored_staged"], 2)
        self.assertEqual(report["stage_two_rows_scored_unstaged"], 4)
        self.assertAlmostEqual(report["staged_total_seconds"], 1.6)
        self.assertAlmostEqual(report["wall_clock_delta_seconds"], -0.4)
        self.assertAlmostEqual(report["wall_clock_speedup"], 2.0 / 1.6)

    def test_a_slower_staged_path_is_reported_not_hidden(self) -> None:
        outcome = _outcome(
            [True, False],
            [np.nan, 0.2],
            feature_seconds=10.0,
            stage_two_seconds=0.1,
        )
        report = subject.compute_accounting(outcome, 0.2)
        self.assertGreater(report["wall_clock_delta_seconds"], 0.0)
        self.assertLess(report["wall_clock_speedup"], 1.0)

    def test_aggregation_sums_rows_and_seconds(self) -> None:
        cells = [
            subject.compute_accounting(
                _outcome([True, False], [np.nan, 0.2], stage_two_seconds=1.0),
                2.0,
            ),
            subject.compute_accounting(
                _outcome([False, False], [0.1, 0.2], stage_two_seconds=2.0),
                2.0,
            ),
        ]
        total = subject._sum_compute(cells)
        self.assertEqual(total["rows"], 4)
        self.assertEqual(total["short_circuited"], 1)
        self.assertEqual(total["short_circuited_fraction"], 0.25)
        self.assertAlmostEqual(total["unstaged_total_seconds"], 4.0)
        self.assertAlmostEqual(total["staged_total_seconds"], 3.0)

    def test_aggregating_nothing_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one cell"):
            subject._sum_compute([])

    def test_the_prefilter_modules_own_saving_agrees(self) -> None:
        outcome = _outcome([True, False, True], [np.nan, 0.2, np.nan])
        mine = subject.compute_accounting(outcome, 1.0)
        theirs = prefilter.compute_saving(outcome.gated)
        self.assertEqual(mine["short_circuited"], theirs["gated_rows"])
        self.assertEqual(
            mine["short_circuited_fraction"], theirs["gated_fraction"]
        )
        self.assertEqual(
            mine["stage_two_rows_scored_staged"],
            theirs["downstream_rows_evaluated"],
        )


class KnownFalseUnknownByStageTests(unittest.TestCase):
    """The binding risk, split so the responsible stage is always visible."""

    def test_each_stage_is_attributed_separately(self) -> None:
        outcome = _outcome(
            [True, False, False, False], [np.nan, 0.99, 0.10, 0.20]
        )
        report = subject.known_false_unknown_by_stage(
            outcome, np.array([0.05, 0.99, 0.10, 0.20]), 0.5
        )
        self.assertEqual(report["rows"], 4)
        self.assertEqual(report["stage_one_gate_rate"], 0.25)
        self.assertEqual(report["stage_two_false_unknown_rate_marginal"], 0.25)
        self.assertAlmostEqual(
            report["stage_two_false_unknown_rate_among_survivors"], 1.0 / 3.0
        )
        self.assertEqual(report["staged_false_unknown_rate"], 0.5)
        self.assertEqual(report["unstaged_false_unknown_rate"], 0.25)
        self.assertEqual(report["rejected_by_stage_one_only"], 1)

    def test_a_prefilter_only_rejection_is_visible(self) -> None:
        outcome = _outcome([True, False], [np.nan, 0.1])
        report = subject.known_false_unknown_by_stage(
            outcome, np.array([0.1, 0.1]), 0.5
        )
        self.assertEqual(report["staged_false_unknown_rate"], 0.5)
        self.assertEqual(report["unstaged_false_unknown_rate"], 0.0)
        self.assertEqual(report["rejected_by_stage_one_only"], 1)
        self.assertEqual(report["rejected_by_stage_two_only"], 0)

    def test_a_fully_gated_population_reports_no_survivor_rate(self) -> None:
        outcome = _outcome([True, True], [np.nan, np.nan])
        report = subject.known_false_unknown_by_stage(
            outcome, np.array([0.1, 0.1]), 0.5
        )
        self.assertEqual(report["staged_false_unknown_rate"], 1.0)
        self.assertIsNone(
            report["stage_two_false_unknown_rate_among_survivors"]
        )

    def test_an_empty_population_is_refused(self) -> None:
        outcome = _outcome([], [])
        with self.assertRaisesRegex(ValueError, "at least one row"):
            subject.known_false_unknown_by_stage(outcome, np.zeros(0), 0.5)


class PrefilterRecorderTests(unittest.TestCase):
    def _recorder(self):
        _, posedegen = _sources()
        calls: list[int] = []

        def inner(captures, *, patch_length, patch_count, target_frac):
            calls.append(len(captures))
            rows = len(captures)
            return (
                np.zeros((rows, 2, PACKED_WIDTH), dtype=np.float32),
                np.zeros((rows, 12), dtype=np.float32),
                [{} for _ in range(rows)],
            )

        recorder = subject._PrefilterRecorder(inner, posedegen)
        recorder.calls = calls
        return recorder

    def _feed(self, recorder, counts, *, length=1024):
        for rows in counts:
            recorder(
                [
                    np.ones(length, dtype=np.complex128)
                    for _ in range(rows)
                ],
                patch_length=PATCH_LENGTH,
                patch_count=PATCH_COUNT,
                target_frac=0.5,
            )

    def test_the_wrapped_call_is_delegated_unchanged(self) -> None:
        recorder = self._recorder()
        packed, features, contexts = recorder(
            [np.ones(1024, dtype=np.complex128) for _ in range(3)],
            patch_length=PATCH_LENGTH,
            patch_count=PATCH_COUNT,
            target_frac=0.5,
        )
        self.assertEqual(recorder.calls, [3])
        self.assertEqual(packed.shape, (3, 2, PACKED_WIDTH))
        self.assertEqual(features.shape, (3, 12))
        self.assertEqual(len(contexts), 3)

    def test_only_the_selection_population_is_extracted(self) -> None:
        """Not computing train/enroll features is the fit-nothing guarantee."""
        recorder = self._recorder()
        self._feed(recorder, (5,))
        self.assertIsNone(recorder.matrix)
        self._feed(recorder, (3,))
        self.assertIsNone(recorder.matrix)
        self._feed(recorder, (2,))
        self.assertIsNotNone(recorder.matrix)
        self.assertEqual(recorder.matrix.shape, (2, pdg.FEATURE_COUNT))

    def test_the_recorded_population_is_keyed_and_measured(self) -> None:
        recorder = self._recorder()
        self._feed(recorder, (5, 3, 2))
        features, length = subject._recorded_prefilter(
            recorder,
            {"xtr": np.zeros(5), "xen": np.zeros(3), "xva": np.zeros(2)},
        )
        self.assertEqual(features.shape, (2, pdg.FEATURE_COUNT))
        self.assertEqual(length, 1024)

    def test_a_missing_population_is_refused(self) -> None:
        recorder = self._recorder()
        self._feed(recorder, (5,))
        with self.assertRaisesRegex(AssertionError, "no longer routes"):
            subject._recorded_prefilter(recorder, {"xtr": np.zeros(5)})

    def test_a_misaligned_population_is_refused(self) -> None:
        recorder = self._recorder()
        self._feed(recorder, (5, 3, 2))
        with self.assertRaisesRegex(AssertionError, "misaligned"):
            subject._recorded_prefilter(
                recorder,
                {"xtr": np.zeros(5), "xen": np.zeros(3), "xva": np.zeros(9)},
            )

    def test_a_mixed_length_selection_population_is_refused(self) -> None:
        recorder = self._recorder()
        self._feed(recorder, (2, 2))
        recorder(
            [np.ones(1024, dtype=np.complex128), np.ones(2048, dtype=np.complex128)],
            patch_length=PATCH_LENGTH,
            patch_count=PATCH_COUNT,
            target_frac=0.5,
        )
        with self.assertRaisesRegex(AssertionError, "mixes capture lengths"):
            subject._recorded_prefilter(
                recorder,
                {"xtr": np.zeros(2), "xen": np.zeros(2), "xva": np.zeros(2)},
            )


class NoveltyFeatureTests(unittest.TestCase):
    def _base_fixtures(self, seeds=(DESIGN_SEED,), lengths=TEST_LENGTHS):
        return base.generate_novelty(
            seeds,
            n_each=2,
            lengths=lengths,
            feature_mean=np.zeros(12, np.float32),
            feature_std=np.ones(12, np.float32),
            patch_length=PATCH_LENGTH,
            patch_count=PATCH_COUNT,
            target_frac=0.5,
        )

    def _features(self, provenance, seeds=(DESIGN_SEED,), lengths=TEST_LENGTHS):
        _, posedegen = _sources()
        return subject.generate_prefilter_features(
            posedegen,
            seeds,
            n_each=2,
            lengths=lengths,
            patch_length=PATCH_LENGTH,
            target_frac=0.5,
            expected_provenance=provenance,
        )

    def test_every_fixture_carries_an_aligned_feature_matrix(self) -> None:
        _fixtures, provenance = self._base_fixtures()
        features = self._features(provenance)
        for family in subject.NOVELTY_FAMILIES:
            for length in TEST_LENGTHS:
                entry = features[DESIGN_SEED][family][length]
                self.assertEqual(entry.features.shape, (2, pdg.FEATURE_COUNT))
                self.assertTrue(np.isfinite(entry.features).all())
                self.assertGreaterEqual(entry.seconds, 0.0)

    def test_the_captures_are_verified_against_the_base_provenance(self) -> None:
        """This is what proves stage 1 and stage 2 scored the same rows."""
        _fixtures, provenance = self._base_fixtures()
        tampered = json.loads(json.dumps(provenance))
        tampered[str(DESIGN_SEED)]["noise"]["longest_capture_sha256"] = "0" * 64
        with self.assertRaisesRegex(AssertionError, "do not match the digest"):
            self._features(tampered)

    def test_a_tampered_prefix_digest_is_caught(self) -> None:
        _fixtures, provenance = self._base_fixtures()
        tampered = json.loads(json.dumps(provenance))
        tampered[str(DESIGN_SEED)]["chirp"]["prefix_sha256"][
            str(TEST_LENGTHS[0])
        ] = "0" * 64
        with self.assertRaisesRegex(AssertionError, "prefix digest"):
            self._features(tampered)

    def test_generation_is_deterministic(self) -> None:
        _fixtures, provenance = self._base_fixtures()
        first = self._features(provenance)
        second = self._features(provenance)
        np.testing.assert_array_equal(
            first[DESIGN_SEED]["noise"][TEST_LENGTHS[0]].features,
            second[DESIGN_SEED]["noise"][TEST_LENGTHS[0]].features,
        )

    def test_noise_and_chirp_features_differ(self) -> None:
        _fixtures, provenance = self._base_fixtures()
        features = self._features(provenance)
        noise = features[DESIGN_SEED]["noise"][TEST_LENGTHS[1]].features
        chirp = features[DESIGN_SEED]["chirp"][TEST_LENGTHS[1]].features
        self.assertFalse(np.allclose(noise.mean(axis=0), chirp.mean(axis=0)))

    def test_a_spent_seed_is_refused_here_too(self) -> None:
        _fixtures, provenance = self._base_fixtures()
        with self.assertRaisesRegex(ValueError, "SPENT"):
            self._features(provenance, seeds=(20260939,))

    def test_prefix_rule_features_equal_the_shorter_lengths_features(
        self,
    ) -> None:
        """The causal-prefix identity, end to end through the sweep.

        With a gate fitted at TEST_LENGTHS, a sweep length of 4096 (above the
        longest fitted length) must produce stage-1 features computed on each
        capture's first max-fitted samples -- which are bit-identical to the
        features of the max-fitted-length observation of the same row, because
        the shorter observation IS that prefix.
        """
        sweep = (*TEST_LENGTHS, 4096)
        _fixtures, provenance = self._base_fixtures(lengths=sweep)
        with tempfile.TemporaryDirectory() as root:
            source, posedegen = _sources()
            gate = subject.load_stage_one(
                source,
                posedegen,
                _bundle_set(Path(root) / "prefilter"),
                required_lengths=sweep,
            )
            features = subject.generate_prefilter_features(
                posedegen,
                (DESIGN_SEED,),
                n_each=2,
                lengths=sweep,
                patch_length=PATCH_LENGTH,
                target_frac=0.5,
                expected_provenance=provenance,
                stage_one=gate,
            )
            # Same rows, no gate: the native-length path.  The gate argument
            # may change ONLY the above-max length; the fitted lengths must be
            # bit-identical between the two calls (the slice there is the
            # whole observation), and the above-max length must differ from
            # its own native-length featurization.
            plain = subject.generate_prefilter_features(
                posedegen,
                (DESIGN_SEED,),
                n_each=2,
                lengths=sweep,
                patch_length=PATCH_LENGTH,
                target_frac=0.5,
                expected_provenance=provenance,
            )
        longest_fitted = max(gate.lengths)
        self.assertLess(longest_fitted, 4096)
        for family in subject.NOVELTY_FAMILIES:
            np.testing.assert_array_equal(
                features[DESIGN_SEED][family][4096].features,
                features[DESIGN_SEED][family][longest_fitted].features,
            )
            for length in TEST_LENGTHS:
                np.testing.assert_array_equal(
                    features[DESIGN_SEED][family][length].features,
                    plain[DESIGN_SEED][family][length].features,
                )
            self.assertFalse(
                np.array_equal(
                    features[DESIGN_SEED][family][4096].features,
                    plain[DESIGN_SEED][family][4096].features,
                ),
                "the prefix rule must change the above-max featurization",
            )


class EndToEndRunTests(unittest.TestCase):
    """Drive ``run`` over a tiny synthetic fusion, without the real corpus.

    The corpus-bound half (``rebuild_populations``) is substituted; everything
    downstream -- the loaded prefilter set, the stage-2 fit, the gate, the short
    circuit, novelty generation with real pose-degeneracy features, the staged
    and unstaged arms, the per-stage known FUR, the compute accounting and the
    written contract -- is the real code path.
    """

    def _rebuilt(self, root: Path) -> "subject.RebuiltPopulations":
        import torch
        from invariant_patch_cnn import InvariantPatchCNN, InvariantPatchConfig
        from train import embed_all

        torch.manual_seed(0)
        common = {
            "patch_length": PATCH_LENGTH,
            "patch_count": PATCH_COUNT,
            "patch_dim": 8,
            "hidden": 12,
            "embed_dim": 6,
            "n_features": 12,
            "dropout": 0.0,
        }
        nets = {
            "real": InvariantPatchCNN(
                InvariantPatchConfig(encoder="real", **common)
            ).eval(),
            "complex": InvariantPatchCNN(
                InvariantPatchConfig(encoder="complex", **common)
            ).eval(),
        }
        device = torch.device("cpu")
        n_classes = 3
        rows = {"tr": 90, "en": 45, "va": 24}
        packed = {
            split: _packed(count, 900 + index)
            for index, (split, count) in enumerate(rows.items())
        }
        features = {
            split: np.random.default_rng(31 + index)
            .normal(size=(count, 12))
            .astype(np.float32)
            for index, (split, count) in enumerate(rows.items())
        }
        branch_embeddings = {
            name: {
                split: embed_all(
                    net, packed[split], features[split], device
                ).astype(np.float32)
                for split in ("tr", "en", "va")
            }
            for name, net in nets.items()
        }
        real_center = subject.base.assemble.fit_center(branch_embeddings["real"])
        complex_center = subject.base.assemble.fit_center(
            branch_embeddings["complex"]
        )
        fused = {
            split: subject.base.assemble.fuse_numpy(
                branch_embeddings["real"][split],
                branch_embeddings["complex"][split],
                real_center,
                complex_center,
                weight_real=0.5,
            )
            for split in ("tr", "en", "va")
        }
        labels = {
            split: (np.arange(count) % n_classes).astype(np.int64)
            for split, count in rows.items()
        }
        prototypes = subject.base.assemble.fit_prototypes(
            fused, labels["en"], n_classes
        )
        data = {
            "xtr": packed["tr"],
            "ytr": labels["tr"],
            "ftr": features["tr"],
            "xen": packed["en"],
            "yen": labels["en"],
            "fen": features["en"],
            "xva": packed["va"],
            "yva": labels["va"],
            "fva": features["va"],
            "n_classes": n_classes,
            "fmean": np.zeros(12, np.float32),
            "fstd": np.ones(12, np.float32),
            "data_audit": {"consumed_test_rows_exposed": 0},
            "cache_contract": {
                "config": {
                    "patch_length": PATCH_LENGTH,
                    "patch_count": PATCH_COUNT,
                    "target_frac": 0.5,
                }
            },
        }
        populations = base.Populations(
            data=data,
            nets=nets,
            branch_embeddings=branch_embeddings,
            fused=fused,
            real_center=real_center,
            complex_center=complex_center,
            prototypes=prototypes,
            reproduction={
                "tolerance": 1e-4,
                "bit_identical_to_artifact": True,
            },
            branch_hashes={"real": {}, "complex": {}},
        )
        return subject.RebuiltPopulations(
            populations=populations,
            # A known population whose stage-1 scores straddle the operating
            # point, so both stages contribute to known FUR and the attribution
            # is exercised rather than trivially zero.
            selection_features=np.vstack(
                [
                    _pose_rows(20, 4242),
                    _pose_rows(4, 4243, shift=6.0),
                ]
            ),
            selection_capture_length=KNOWN_LENGTH,
            selection_feature_seconds=0.25,
        )

    def _args(self, root: Path, **overrides) -> argparse.Namespace:
        fusion_dir = FusionArtifactValidationTests()._artifact(root / "fusion")
        values = {
            "fusion_dir": str(fusion_dir),
            "prefilter_dir": str(_bundle_set(root / "prefilter")),
            "output_dir": str(root / "openset"),
            "role": "validate",
            "design_novelty_seed": DESIGN_SEED,
            "prefilter_module": subject.PREFILTER_MODULE_DEFAULT,
            "posedegen_module": subject.POSEDEGEN_MODULE_DEFAULT,
            "device": "cpu",
            "novelty_seeds": [VALIDATION_SEED],
            "novelty_n": 2,
            "prefix_lengths": list(TEST_LENGTHS),
            "reproduction_tolerance": 1e-4,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def _run(self, root: Path, **overrides) -> dict:
        args = self._args(root, **overrides)
        rebuilt = self._rebuilt(root)
        with mock.patch.object(
            subject, "rebuild_populations", return_value=rebuilt
        ), contextlib.redirect_stdout(io.StringIO()):
            return subject.run(args)

    # -- the architecture contract ----------------------------------------

    def test_the_report_declares_that_the_closed_label_can_be_gated(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            report = self._run(Path(root))
            written = json.loads(
                (Path(root) / "openset" / "openset_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertTrue(written["closed_label_can_be_gated"])
        self.assertFalse(written["rejector_is_additive_only"])
        self.assertFalse(written["additive_only"])
        self.assertTrue(written["changes_closed_label"])
        self.assertTrue(written["gates_before_classification"])
        self.assertFalse(written["stage_two_closed_label_altered"])
        note = written["closed_label_can_be_gated_note"]
        self.assertIn("no closed-set label", note)
        self.assertIn("deliberate", note)
        self.assertIn("STAGE 1 IS A BEHAVIOURAL GATE", written["closed_label_contract"])
        self.assertEqual(report["status"], written["status"])

    def test_the_report_carries_every_contract_key_the_prefilter_requires(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            report = self._run(Path(root))
        for key in prefilter.CONTRACT_KEYS_FOR_RELEASE_CLAIM:
            self.assertIn(key, report)

    def test_the_evidence_contract_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            report = self._run(Path(root))
        self.assertEqual(report["sealed_release_data_used"], 0)
        self.assertEqual(report["consumed_test_rows_used"], 0)
        self.assertEqual(report["sealed_release_paths_read"], 0)
        self.assertFalse(report["release_seed_20260731_used"])
        self.assertTrue(report["development_only"])
        self.assertFalse(report["release_evidence"])
        contract = report["fitting_contract"]
        self.assertEqual(contract["density_population"], "training only")
        self.assertEqual(
            contract["rank_and_threshold_population"], "enrollment only"
        )
        self.assertFalse(contract["policy_searched_here"])
        self.assertFalse(contract["stage_one_fitted_in_this_run"])
        self.assertFalse(contract["stage_one_training_features_computed_here"])
        self.assertFalse(contract["stage_one_enrollment_features_computed_here"])
        self.assertEqual(
            report["seeds"][
                "novelty_rows_used_to_choose_the_stage_one_operating_point"
            ],
            0,
        )

    def test_the_rebuild_check_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            report = self._run(Path(root))
        self.assertTrue(
            report["fusion"]["reproduction"]["bit_identical_to_artifact"]
        )
        self.assertEqual(len(report["fusion"]["directory_sha256"]), 64)

    # -- gates, at every seed and length -----------------------------------

    def test_the_full_gate_set_is_reported_at_every_seed_and_length(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            report = self._run(Path(root))
        self.assertEqual(set(report["novelty"]), {str(VALIDATION_SEED)})
        for length in TEST_LENGTHS:
            row = report["novelty"][str(VALIDATION_SEED)][str(length)]
            for family in subject.NOVELTY_FAMILIES:
                self.assertIn("auroc", row[family])
                self.assertIn("threshold_recall", row[family])
                self.assertIn("gated_at_stage_one_fraction", row[family])
                self.assertIn("rejected_at_stage_two_fraction", row[family])
            self.assertIn("auroc", row["overall"])
            self.assertIn("known_false_unknown_rate", row)
            self.assertIn("known_false_unknown_by_stage", row)
        self.assertEqual(
            set(report["gates"]),
            set(subject.GATE_FLOORS) | {"known_false_unknown_rate"},
        )
        for name, floor in subject.GATE_FLOORS.items():
            self.assertEqual(report["gates"][name]["floor"], floor)
        self.assertEqual(
            report["gates"]["known_false_unknown_rate"]["ceiling"],
            subject.KNOWN_FUR_CEILING,
        )

    def test_a_sweep_length_above_the_fitted_set_is_gated_by_prefix_rule(
        self,
    ) -> None:
        """The sealed suite's N32768 shape, in miniature: sweep 1024/2048/4096
        against a gate fitted at 1024/2048 only.

        Stage 1 at the above-max length must be the max-fitted gate applied to
        each capture's first max-fitted samples -- so its stage-1 outcomes are
        bit-identical to the max-fitted length's cell (same rows, same prefix),
        while stage 2 still scores the full-length capture.  The report must
        say all of this rather than leave it implicit.
        """
        sweep = [*TEST_LENGTHS, 4096]
        with tempfile.TemporaryDirectory() as root:
            report = self._run(Path(root), prefix_lengths=sweep)
        protocol = report["protocol"]
        self.assertEqual(protocol["prefix_lengths"], sweep)
        self.assertEqual(
            protocol["stage_one_feature_length_by_prefix_length"],
            {
                str(TEST_LENGTHS[0]): TEST_LENGTHS[0],
                str(TEST_LENGTHS[1]): TEST_LENGTHS[1],
                "4096": KNOWN_LENGTH,
            },
        )
        self.assertEqual(
            protocol["stage_one_causal_prefix_rule_lengths"], [4096]
        )
        rows = report["novelty"][str(VALIDATION_SEED)]
        long_row = rows["4096"]
        max_fitted_row = rows[str(KNOWN_LENGTH)]
        for family in subject.NOVELTY_FAMILIES:
            self.assertEqual(
                long_row[family]["gated_at_stage_one_fraction"],
                max_fitted_row[family]["gated_at_stage_one_fraction"],
            )
            self.assertEqual(
                long_row[family]["stage_one_raw_median"],
                max_fitted_row[family]["stage_one_raw_median"],
            )
        # The gate summary covers the above-max cells too: worst-of includes
        # them, so a failure there would fail the run's gates.
        self.assertEqual(
            report["gates"]["noise_auroc"]["worst"],
            min(
                rows[str(length)]["noise"]["auroc"] for length in sweep
            ),
        )

    def test_the_unstaged_control_is_reported_beside_the_staged_gates(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            report = self._run(Path(root))
        control = report["unstaged_control"]
        self.assertEqual(set(control["gates"]), set(report["gates"]))
        for name, floor in subject.GATE_FLOORS.items():
            self.assertEqual(control["gates"][name]["floor"], floor)
        self.assertIn("attributable to the gate", control["description"])

    # -- known FUR, per stage ---------------------------------------------

    def test_known_false_unknown_rate_is_reported_per_stage(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            report = self._run(Path(root))
        stages = report["known"]["by_stage"]
        self.assertIn("stage_one_gate_rate", stages)
        self.assertIn("stage_two_false_unknown_rate_marginal", stages)
        self.assertIn("stage_two_false_unknown_rate_among_survivors", stages)
        self.assertIn("unstaged_false_unknown_rate", stages)
        self.assertAlmostEqual(
            stages["staged_false_unknown_rate"],
            report["gates"]["known_false_unknown_rate"]["worst"],
        )
        self.assertLessEqual(
            stages["stage_one_gate_rate"], stages["staged_false_unknown_rate"]
        )

    def test_a_known_row_gated_by_the_prefilter_is_visible_as_such(self) -> None:
        """The four shifted selection rows must be attributed to stage 1."""
        with tempfile.TemporaryDirectory() as root:
            report = self._run(Path(root))
        stages = report["known"]["by_stage"]
        self.assertGreater(stages["stage_one_gate_rate"], 0.0)
        self.assertGreater(stages["rejected_by_stage_one_only"], 0)
        self.assertIsNotNone(
            report["known"]["closed_selection_accuracy_on_stage_one_survivors"]
        )

    # -- compute saving ----------------------------------------------------

    def test_the_compute_saving_is_measured_and_reported(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            report = self._run(Path(root))
        saved = report["compute_saved"]
        self.assertIn("measured, not asserted", saved["note"])
        total = saved["novelty_total"]
        self.assertEqual(
            total["rows"],
            2 * len(TEST_LENGTHS) * len(subject.NOVELTY_FAMILIES),
        )
        self.assertIn("short_circuited_fraction", total)
        self.assertIn("wall_clock_delta_seconds", total)
        self.assertIn("unstaged_total_seconds", total)
        self.assertEqual(
            len(saved["by_cell"]),
            len(TEST_LENGTHS) * len(subject.NOVELTY_FAMILIES),
        )
        for cell in saved["by_cell"]:
            self.assertIn("prefilter_module_compute_saving", cell)
            self.assertEqual(
                cell["short_circuited"],
                cell["prefilter_module_compute_saving"]["gated_rows"],
            )
        self.assertIn("known_selection", saved)

    def test_short_circuited_rows_never_reach_stage_two(self) -> None:
        """The saving is real: gated rows are absent from the stage-2 count."""
        with tempfile.TemporaryDirectory() as root:
            report = self._run(Path(root))
        for cell in report["compute_saved"]["by_cell"]:
            self.assertEqual(
                cell["stage_two_rows_scored_staged"] + cell["short_circuited"],
                cell["rows"],
            )
            self.assertEqual(cell["stage_two_rows_scored_unstaged"], cell["rows"])

    def test_a_heavily_gating_prefilter_short_circuits_novelty_rows(self) -> None:
        """Exercise the short circuit on novelty, not only on known rows."""
        with tempfile.TemporaryDirectory() as root:
            directory = _bundle_set(Path(root) / "permissive", budget=0.90)
            report = self._run(Path(root), prefilter_dir=str(directory))
        total = report["compute_saved"]["novelty_total"]
        self.assertGreater(total["short_circuited"], 0)
        self.assertLess(
            total["stage_two_rows_scored_staged"],
            total["stage_two_rows_scored_unstaged"],
        )
        gated_any = any(
            report["novelty"][str(VALIDATION_SEED)][str(length)][family][
                "gated_at_stage_one_fraction"
            ]
            > 0.0
            for length in TEST_LENGTHS
            for family in subject.NOVELTY_FAMILIES
        )
        self.assertTrue(gated_any)

    def test_subset_scoring_agreement_is_asserted_and_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            report = self._run(Path(root))
        agreement = report["known"]["subset_scoring_agreement"]
        self.assertLessEqual(
            agreement["max_abs_difference"], subject.SUBSET_SCORING_TOLERANCE
        )

    # -- provenance --------------------------------------------------------

    def test_the_stage_one_provenance_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            report = self._run(Path(root))
        stage_one = report["stage_one"]
        self.assertEqual(stage_one["version"], prefilter.NOISE_PREFILTER_VERSION)
        self.assertFalse(stage_one["fitted_here"])
        self.assertEqual(len(stage_one["set_sha256"]), 64)
        self.assertIn(KNOWN_LENGTH, stage_one["capture_lengths"])
        self.assertEqual(stage_one["known_capture_length"], KNOWN_LENGTH)
        self.assertEqual(len(stage_one["selection_features_sha256"]), 64)
        self.assertIn("posedegen", stage_one)
        for metadata in stage_one["models"].values():
            self.assertTrue(metadata["changes_closed_label"])

    def test_the_source_hashes_cover_both_siblings(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            report = self._run(Path(root))
        hashes = report["source_sha256"]
        self.assertIn("fit_v3_openset_staged.py", hashes)
        self.assertIn("fit_v3_openset.py", hashes)
        self.assertIn("noise_prefilter.py", hashes)
        self.assertIn("pose_degeneracy.py", hashes)

    def test_roles_label_their_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            validated = self._run(Path(root))
        self.assertTrue(validated["gates_are_evidence"])
        self.assertTrue(validated["status"].startswith("development_openset_"))
        with tempfile.TemporaryDirectory() as root:
            designed = self._run(
                Path(root), role="design", novelty_seeds=[DESIGN_SEED]
            )
        self.assertFalse(designed["gates_are_evidence"])
        self.assertTrue(designed["status"].startswith("design_selection_"))
        self.assertIn("NOT validation evidence", designed["evidence_role"])

    def test_run_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            first = self._run(Path(root))
        with tempfile.TemporaryDirectory() as root:
            second = self._run(Path(root))
        self.assertEqual(first["gates"], second["gates"])
        self.assertEqual(
            first["novelty"][str(VALIDATION_SEED)][str(TEST_LENGTHS[0])][
                "noise"
            ]["auroc"],
            second["novelty"][str(VALIDATION_SEED)][str(TEST_LENGTHS[0])][
                "noise"
            ]["auroc"],
        )

    # -- refusals ----------------------------------------------------------

    def test_an_absent_prefilter_module_stops_the_run_before_any_output(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError, "not importable"):
                self._run(Path(root), prefilter_module="no_such_prefilter")
            self.assertFalse((Path(root) / "openset").exists())

    def test_an_absent_posedegen_module_stops_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError, "not importable"):
                self._run(Path(root), posedegen_module="no_such_posedegen")
            self.assertFalse((Path(root) / "openset").exists())

    def test_a_spent_novelty_seed_stops_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "SPENT"):
                self._run(Path(root), novelty_seeds=[20260939])
            self.assertFalse((Path(root) / "openset").exists())

    def test_the_release_seed_stops_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "unspent release seed"):
                self._run(Path(root), novelty_seeds=[20260731])

    def test_a_prefilter_set_missing_a_scored_length_stops_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            directory = _bundle_set(
                Path(root) / "partial", lengths=(TEST_LENGTHS[0],)
            )
            with self.assertRaisesRegex(ValueError, "no fitted prefilter"):
                self._run(Path(root), prefilter_dir=str(directory))

    def test_run_refuses_a_non_empty_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "openset"
            output.mkdir(parents=True)
            (output / "existing.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "not empty"):
                self._run(Path(root))

    def test_run_refuses_a_release_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "release or sealed"):
                self._run(
                    Path(root),
                    output_dir="/tmp/atom/training/artifacts/releases/staged",
                )


if __name__ == "__main__":
    unittest.main()
