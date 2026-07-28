"""Unit tests for the STAGED v3 open-set rejector.

These tests never touch the development corpus, the consumed test half, or any
sealed release suite.  Every scored population is synthetic, small, and
constructed in the test.  A serialization-compatibility test reads the tracked
frozen q99 policy NPZ (enrollment calibration only, no population rows) so the
machine constants cannot drift from their evidence.  No spent, sealed or
release novelty seed is used except where a test proves that it is refused.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import inspect
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
# Unreserved clean seed used only by tiny synthetic unit fixtures. It is not
# scored as project evidence and is never written to an artifact.
UNIT_NOVELTY_SEED = 20260956
FITTING_SEED = prefilter.PROPOSED_FITTING_SEED

# Short lengths so the tests stay fast.  The prefilter is length dependent, so
# a bundle is fitted for each of these plus the known-population length.
TEST_LENGTHS = (1024, 2048)
KNOWN_LENGTH = 2048


@contextlib.contextmanager
def _temporarily_unspent_design_seed():
    """Exercise the pre-draw design branch without falsifying the live ledger."""
    ledger = {
        seed: reason
        for seed, reason in subject.SPENT_NOVELTY_SEEDS.items()
        if seed != DESIGN_SEED
    }
    with mock.patch.dict(subject.SPENT_NOVELTY_SEEDS, ledger, clear=True):
        yield


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
    budget: float = subject.STAGE_ONE_KNOWN_FALSE_POSITIVE_BUDGET,
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
            [
                20260938,
                20260939,
                20260940,
                20260941,
                20260942,
                20260943,
                20260944,
                20260945,
                20260946,
                20260947,
                20260948,
                20260949,
                20260950,
                20260951,
                20260952,
            ],
        )
        self.assertEqual(subject.FIRST_CLEAN_NOVELTY_SEED, 20260953)
        self.assertEqual(subject.PROPOSED_DESIGN_NOVELTY_SEED, 20260955)
        self.assertIn(20260952, subject.SPENT_NOVELTY_SEEDS)
        self.assertIn(
            "1/300 = 0.0033333333333333335 < 0.10",
            subject.SPENT_NOVELTY_SEEDS[20260952],
        )
        self.assertIn(
            "38/1908 = 0.019916142557651992",
            subject.SPENT_NOVELTY_SEEDS[20260952],
        )
        self.assertNotIn(20260955, subject.SPENT_NOVELTY_SEEDS)
        self.assertEqual(
            subject.DEFAULT_VALIDATION_NOVELTY_SEEDS, (20260953, 20260954)
        )
        for seed in subject.DEFAULT_VALIDATION_NOVELTY_SEEDS:
            self.assertNotIn(seed, subject.SPENT_NOVELTY_SEEDS)
        self.assertEqual(
            sorted(subject.CONSUMED_SEALED_RELEASE_SEEDS),
            [20260729, 20260731, 20260733, 20260734, 20260735],
        )
        self.assertEqual(subject.RELEASE_SEED_NEVER_SPENT_HERE, 20260736)

    def test_the_ledger_extends_the_prefilter_modules_frozen_one(self) -> None:
        """The prefilter module's ledger froze with the fitted bundles; the
        staged ledger is the moving authority and may only ever grow beyond
        it, never contradict it."""
        for seed in prefilter.SPENT_NOVELTY_SEEDS:
            self.assertIn(seed, subject.SPENT_NOVELTY_SEEDS)
        self.assertGreaterEqual(
            subject.FIRST_CLEAN_NOVELTY_SEED,
            prefilter.CLEAN_VALIDATION_SEEDS_FROM,
        )
        # Every seed between the prefilter's clean mark and the staged clean
        # mark must be accounted for as spent, or it silently leaked.
        for seed in range(
            prefilter.CLEAN_VALIDATION_SEEDS_FROM,
            subject.FIRST_CLEAN_NOVELTY_SEED,
        ):
            self.assertIn(seed, subject.SPENT_NOVELTY_SEEDS)

    def test_every_spent_seed_is_refused_as_novelty(self) -> None:
        for seed in subject.SPENT_NOVELTY_SEEDS:
            with self.assertRaisesRegex(ValueError, "SPENT"):
                subject.validate_novelty_seeds([seed])

    def test_every_spent_seed_is_refused_as_a_design_seed(self) -> None:
        for seed in subject.SPENT_NOVELTY_SEEDS:
            with self.assertRaisesRegex(ValueError, "SPENT"):
                subject.validate_seed_plan("design", seed, [seed])

    def test_validation_may_reference_frozen_design_without_redrawing_it(self):
        with mock.patch.dict(
            subject.SPENT_NOVELTY_SEEDS,
            {DESIGN_SEED: "froze the selected design"},
            clear=True,
        ):
            role, design, seeds = subject.validate_seed_plan(
                "validate",
                DESIGN_SEED,
                subject.DEFAULT_VALIDATION_NOVELTY_SEEDS,
            )
            self.assertEqual(
                (role, design, seeds),
                (
                    "validate",
                    DESIGN_SEED,
                    subject.DEFAULT_VALIDATION_NOVELTY_SEEDS,
                ),
            )
            with self.assertRaisesRegex(ValueError, "SPENT"):
                subject.validate_novelty_seeds([DESIGN_SEED])

    def test_validation_must_draw_exactly_the_predeclared_pair(self) -> None:
        with mock.patch.dict(
            subject.SPENT_NOVELTY_SEEDS,
            {DESIGN_SEED: "froze the selected design"},
            clear=True,
        ):
            for seeds in (
                [subject.DEFAULT_VALIDATION_NOVELTY_SEEDS[0]],
                [*subject.DEFAULT_VALIDATION_NOVELTY_SEEDS, UNIT_NOVELTY_SEED],
                list(reversed(subject.DEFAULT_VALIDATION_NOVELTY_SEEDS)),
                [UNIT_NOVELTY_SEED],
            ):
                with self.assertRaisesRegex(
                    ValueError, "exactly the predeclared reserved"
                ):
                    subject.validate_seed_plan(
                        "validate", DESIGN_SEED, seeds
                    )

    def test_validation_cannot_reference_an_unrelated_spent_seed(self) -> None:
        with self.assertRaisesRegex(ValueError, "design seed is reserved"):
            subject.validate_seed_plan(
                "validate",
                20260946,
                subject.DEFAULT_VALIDATION_NOVELTY_SEEDS,
            )

    def test_the_sealed_and_release_seeds_are_refused(self) -> None:
        for seed in (
            20260729,
            20260731,
            20260733,
            20260734,
            20260735,
            20260736,
        ):
            with self.assertRaises(ValueError):
                subject.validate_novelty_seeds([seed])
            with self.assertRaises(ValueError):
                subject.validate_seed_plan("validate", seed, [VALIDATION_SEED])

    def test_the_consumed_sealed_v3_seed_names_its_consumption(self) -> None:
        with self.assertRaisesRegex(ValueError, "consumed sealed"):
            subject.validate_novelty_seeds([20260731])
        with self.assertRaisesRegex(ValueError, "consumed sealed"):
            subject.validate_novelty_seeds([20260734])
        with self.assertRaisesRegex(ValueError, "consumed sealed"):
            subject.validate_novelty_seeds([20260735])
        with self.assertRaisesRegex(ValueError, "unspent release seed"):
            subject.validate_novelty_seeds([20260736])

    def test_a_clean_seed_is_accepted(self) -> None:
        self.assertEqual(
            subject.validate_novelty_seeds([20260953, 20260954, 20260955]),
            (20260953, 20260954, 20260955),
        )

    def test_novelty_seed_validation_is_typed_and_respects_clean_boundary(
        self,
    ) -> None:
        for value in (True, np.bool_(False), 20260953.0, "20260953"):
            with self.assertRaisesRegex(ValueError, "must be integers"):
                subject.validate_novelty_seeds([value])
        for seed in (1, 20260937):
            with self.assertRaisesRegex(ValueError, "clean boundary"):
                subject.validate_novelty_seeds([seed])

    def test_duplicate_and_empty_seed_sets_are_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "distinct"):
            subject.validate_novelty_seeds([20260950, 20260950])
        with self.assertRaisesRegex(ValueError, "distinct"):
            subject.validate_novelty_seeds([])

    def test_validation_is_refused_before_the_design_ledger_transition(self) -> None:
        with self.assertRaisesRegex(ValueError, "not yet frozen"):
            subject.validate_seed_plan(
                "validate", DESIGN_SEED, subject.DEFAULT_VALIDATION_NOVELTY_SEEDS
            )

    def test_a_default_validation_seed_may_not_be_the_design_seed(self) -> None:
        with self.assertRaisesRegex(ValueError, "design seed is reserved"):
            subject.validate_seed_plan("design", VALIDATION_SEED, [VALIDATION_SEED])

    def test_design_refuses_the_reserved_validation_seeds(self) -> None:
        with _temporarily_unspent_design_seed():
            with self.assertRaisesRegex(ValueError, "reserved validation"):
                subject.validate_seed_plan(
                    "design", DESIGN_SEED, [DESIGN_SEED, VALIDATION_SEED]
                )

    def test_design_must_draw_its_own_declared_seed(self) -> None:
        with _temporarily_unspent_design_seed():
            with self.assertRaisesRegex(ValueError, "must draw novelty"):
                subject.validate_seed_plan(
                    "design", DESIGN_SEED, [UNIT_NOVELTY_SEED]
                )
            with self.assertRaisesRegex(
                ValueError, "exactly its declared design"
            ):
                subject.validate_seed_plan(
                    "design",
                    DESIGN_SEED,
                    [DESIGN_SEED, UNIT_NOVELTY_SEED],
                )
            self.assertEqual(
                subject.validate_seed_plan(
                    "design", DESIGN_SEED, [DESIGN_SEED]
                ),
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
        self.assertIn("intentionally not in SPENT", subject.SEED_LEDGER_NOTE)
        self.assertIn("evidence rule 4", subject.SEED_LEDGER_NOTE)

    def test_future_design_and_validation_commands_are_explicit(self) -> None:
        commands = subject.v34_design_and_validation_commands(
            fusion_dir=Path("/development/frozen-rejector"),
            prefilter_dir=Path("/development/frozen-prefilter"),
            design_output_dir=Path("/development/v34-design"),
            validation_output_dir=Path("/development/v34-validation"),
        )
        design = commands["design"]
        validation = commands["validation_after_design_ledger_transition"]
        self.assertEqual(design[design.index("--role") + 1], "design")
        self.assertEqual(
            design[design.index("--novelty-seeds") + 1 :],
            ["20260955"],
        )
        prefix = design.index("--prefix-lengths")
        output = design.index("--output-dir")
        self.assertEqual(
            design[prefix + 1 : output],
            ["4096", "8192", "16384", "32768"],
        )
        self.assertEqual(design[design.index("--device") + 1], "cpu")
        self.assertEqual(validation[validation.index("--role") + 1], "validate")
        self.assertEqual(
            validation[validation.index("--novelty-seeds") + 1 :],
            ["20260953", "20260954"],
        )
        self.assertNotIn("20260736", design)
        self.assertNotIn("20260736", validation)


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


class CompositeSurvivorPolicyTests(unittest.TestCase):
    """Policy version 4: the q97 composite score, fit on enrollment."""

    def _gate(self, root: Path) -> "subject.StageOneGate":
        source, posedegen = _sources()
        return subject.load_stage_one(
            source,
            posedegen,
            _bundle_set(root / "prefilter"),
            required_lengths=(*TEST_LENGTHS, KNOWN_LENGTH),
        )

    def _enrollment(self, rows: int = 60):
        features = np.vstack(
            [_pose_rows(rows - 3, 71), _pose_rows(3, 72, shift=6.0)]
        )
        rng = np.random.default_rng(73)
        stage_two = rng.uniform(0.0, 0.999, size=rows)
        return features, stage_two

    def test_the_frozen_constants_are_the_bumped_policy_version(self) -> None:
        self.assertEqual(subject.STAGED_POLICY_SCHEMA, 4)
        self.assertEqual(
            subject.STAGED_POLICY_VERSION,
            "v3-staged-openset-policy-v4-composite-survivor-q97",
        )
        self.assertIn("composite_survivor", subject.STAGED_POLICY_KIND)
        self.assertIn("1", subject.STAGED_POLICY_VERSION_HISTORY)
        self.assertIn("2", subject.STAGED_POLICY_VERSION_HISTORY)
        self.assertIn("3", subject.STAGED_POLICY_VERSION_HISTORY)
        self.assertIn("4", subject.STAGED_POLICY_VERSION_HISTORY)
        # Stage 2 stays at imported q95; q99 is frozen read-only; current is q97.
        self.assertIs(
            subject.LEGACY_COMPOSITE_THRESHOLD_QUANTILE,
            subject.THRESHOLD_QUANTILE,
        )
        self.assertEqual(subject.THRESHOLD_QUANTILE, 0.95)
        self.assertEqual(
            subject.FAILED_Q99_COMPOSITE_THRESHOLD_QUANTILE, 0.99
        )
        self.assertEqual(subject.COMPOSITE_THRESHOLD_QUANTILE, 0.97)
        self.assertEqual(
            subject.STAGE_ONE_KNOWN_FALSE_POSITIVE_BUDGET, 0.01
        )
        self.assertAlmostEqual(
            subject.NOMINAL_ENROLLMENT_FALSE_UNKNOWN_BUDGET, 0.0397
        )
        self.assertEqual(
            subject.FROZEN_POLICY_V4_Q97_ENROLLMENT_THRESHOLD,
            0.9844868317511343,
        )
        self.assertLess(
            subject.FROZEN_POLICY_V2_Q95_ENROLLMENT_THRESHOLD,
            subject.FROZEN_POLICY_V4_Q97_ENROLLMENT_THRESHOLD,
        )
        self.assertLess(
            subject.FROZEN_POLICY_V4_Q97_ENROLLMENT_THRESHOLD,
            subject.FROZEN_POLICY_V3_Q99_ENROLLMENT_THRESHOLD,
        )
        self.assertEqual(
            subject.COMPOSITE_SURVIVOR_SCORE,
            "max(stage2_enrollment_rank, stage1_score_enrollment_rank)",
        )

    def test_fit_is_on_enrollment_survivors_only(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            gate = self._gate(Path(root))
            features, stage_two = self._enrollment()
            policy = subject.fit_composite_policy(
                gate,
                features,
                KNOWN_LENGTH,
                stage_two,
                stage_two_threshold=0.9,
            )
            raw = gate.raw(features, KNOWN_LENGTH)
            gated = gate.fires(features, KNOWN_LENGTH)
        survivors = ~gated
        self.assertGreater(int(np.count_nonzero(gated)), 0)
        self.assertEqual(policy.enrollment_rows, len(features))
        self.assertEqual(
            policy.enrollment_gated_rows, int(np.count_nonzero(gated))
        )
        self.assertEqual(
            policy.enrollment_survivor_rows, int(np.count_nonzero(survivors))
        )
        np.testing.assert_array_equal(
            policy.stage_one_calibration_raw, np.sort(raw[survivors])
        )
        # The rank convention is the stage-2 searchsorted-left convention.
        rank = policy.stage_one_survivor_rank(raw[survivors])
        expected = np.searchsorted(
            np.sort(raw[survivors]), raw[survivors], side="left"
        ) / (np.count_nonzero(survivors) + 1.0)
        np.testing.assert_array_equal(rank, expected)
        np.testing.assert_array_equal(
            rank, frozen.empirical_rank(raw[survivors], np.sort(raw[survivors]))
        )
        # The composite is unchanged, and only its threshold moves to q97.
        composite = policy.composite(stage_two[survivors], raw[survivors])
        np.testing.assert_array_equal(
            composite, np.maximum(stage_two[survivors], rank)
        )
        self.assertEqual(
            policy.threshold,
            float(
                np.quantile(
                    composite, subject.COMPOSITE_THRESHOLD_QUANTILE
                )
            ),
        )
        np.testing.assert_array_equal(
            policy.composite_calibration_raw, np.sort(composite)
        )
        provenance = policy.provenance()
        self.assertEqual(
            provenance["threshold_population"],
            "enrollment stage-1 survivors only (enrollment only, as every "
            "rank and threshold in this chain)",
        )
        for population in ("training", "selection", "novelty", "release"):
            self.assertEqual(
                provenance[f"{population}_rows_used_for_threshold"], 0
            )
        self.assertEqual(
            provenance["only_policy_change"],
            "composite survivor enrollment threshold quantile q99 -> q97",
        )
        self.assertFalse(provenance["stage_one_changed"])
        self.assertFalse(provenance["rejector_cnn_fusion_changed"])
        self.assertFalse(provenance["classifier_cnn_fusion_changed"])
        self.assertTrue(
            provenance["gate_floors_and_known_fur_ceiling_unchanged"]
        )
        self.assertGreaterEqual(policy.threshold, 0.0)
        self.assertLess(policy.threshold, 1.0)

    def test_fit_api_cannot_receive_selection_novelty_or_release_rows(self) -> None:
        parameters = set(inspect.signature(subject.fit_composite_policy).parameters)
        self.assertEqual(
            parameters,
            {
                "stage_one",
                "enrollment_features",
                "enrollment_capture_length",
                "enrollment_stage_two_scores",
                "stage_two_threshold",
            },
        )
        self.assertFalse(
            parameters & {"selection", "novelty", "release", "sealed"}
        )

    def test_the_enrollment_length_must_be_fitted_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            gate = self._gate(Path(root))
            features, stage_two = self._enrollment()
            with self.assertRaisesRegex(ValueError, "native length"):
                subject.fit_composite_policy(
                    gate,
                    features,
                    4096,
                    stage_two,
                    stage_two_threshold=0.9,
                )

    def test_a_non_rank_stage_two_vector_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            gate = self._gate(Path(root))
            features, stage_two = self._enrollment()
            stage_two[0] = 1.5
            with self.assertRaisesRegex(ValueError, r"\[0, 1\)"):
                subject.fit_composite_policy(
                    gate,
                    features,
                    KNOWN_LENGTH,
                    stage_two,
                    stage_two_threshold=0.9,
                )

    def test_a_fully_gated_enrollment_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            gate = self._gate(Path(root))
            rows = 8
            features = _pose_rows(rows, 74, shift=9.0)
            if not gate.fires(features, KNOWN_LENGTH).all():
                self.skipTest("fixture did not gate every row")
            with self.assertRaisesRegex(ValueError, "at least one"):
                subject.fit_composite_policy(
                    gate,
                    features,
                    KNOWN_LENGTH,
                    np.full(rows, 0.5),
                    stage_two_threshold=0.9,
                )

    def test_save_load_round_trips_and_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            gate = self._gate(Path(root))
            features, stage_two = self._enrollment()
            policy = subject.fit_composite_policy(
                gate,
                features,
                KNOWN_LENGTH,
                stage_two,
                stage_two_threshold=0.9,
            )
            path = Path(root) / subject.COMPOSITE_POLICY_FILENAME
            subject.save_composite_policy(path, policy)
            loaded = subject.load_composite_policy(
                path, expected_stage_two_threshold=0.9
            )
            self.assertEqual(loaded.threshold, policy.threshold)
            self.assertEqual(loaded.stage_two_threshold, 0.9)
            self.assertEqual(loaded.enrollment_rows, policy.enrollment_rows)
            np.testing.assert_array_equal(
                loaded.stage_one_calibration_raw,
                policy.stage_one_calibration_raw,
            )
            probe = np.linspace(-3.0, 9.0, 7)
            np.testing.assert_array_equal(
                loaded.stage_one_survivor_rank(probe),
                policy.stage_one_survivor_rank(probe),
            )

    def _saved(self, root: Path):
        gate = self._gate(root)
        features, stage_two = self._enrollment()
        policy = subject.fit_composite_policy(
            gate, features, KNOWN_LENGTH, stage_two, stage_two_threshold=0.9
        )
        path = root / subject.COMPOSITE_POLICY_FILENAME
        subject.save_composite_policy(path, policy)
        return path

    def _tamper(self, path: Path, **overrides) -> None:
        with np.load(path) as payload:
            data = {name: payload[name] for name in payload.files}
        data.update(overrides)
        np.savez_compressed(path, **data)

    def _historical_saved(
        self, root: Path, spec: "subject.StagedPolicySpec"
    ) -> Path:
        path = self._saved(root)
        with np.load(path) as payload:
            data = {name: payload[name] for name in payload.files}
        data.update(
            schema=np.asarray(spec.schema, dtype=np.int64),
            kind=np.asarray(spec.kind),
            policy_version=np.asarray(spec.version),
            threshold_quantile=np.asarray(
                spec.threshold_quantile, dtype=np.float64
            ),
            threshold=np.asarray(
                np.quantile(
                    data["composite_calibration_raw"],
                    spec.threshold_quantile,
                ),
                dtype=np.float64,
            ),
        )
        hygiene_names = {
            "threshold_population",
            "training_rows_used_for_threshold",
            "selection_rows_used_for_threshold",
            "novelty_rows_used_for_threshold",
            "release_rows_used_for_threshold",
            "stage_one_known_false_positive_budget",
            "survivor_known_false_positive_budget",
            "nominal_enrollment_false_unknown_budget",
            "only_policy_change",
            "stage_one_changed",
            "rejector_cnn_fusion_changed",
            "classifier_cnn_fusion_changed",
            "gate_contract_changed",
        }
        hygiene = subject._policy_hygiene_expected(spec)
        if hygiene is None:
            for name in hygiene_names:
                data.pop(name, None)
        else:
            data.update(
                {
                    name: np.asarray(value)
                    for name, value in hygiene.items()
                }
            )
        np.savez_compressed(path, **data)
        return path

    def _legacy_saved(self, root: Path) -> Path:
        return self._historical_saved(
            root, subject.LEGACY_STAGED_POLICY_SPEC
        )

    def _failed_q99_saved(self, root: Path) -> Path:
        return self._historical_saved(
            root, subject.FAILED_Q99_STAGED_POLICY_SPEC
        )

    def test_q97_is_between_q95_and_q99_on_identical_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            base_path = Path(root)
            q95_path = self._legacy_saved(base_path / "q95")
            q99_path = self._failed_q99_saved(base_path / "q99")
            q95 = subject.load_composite_policy(
                q95_path,
                expected_policy_version=subject.LEGACY_STAGED_POLICY_VERSION,
            )
            q99 = subject.load_composite_policy(
                q99_path,
                expected_policy_version=(
                    subject.FAILED_Q99_STAGED_POLICY_VERSION
                ),
            )
            q97_from_q95 = subject.rethreshold_legacy_composite_policy(q95)
            q97_from_q99 = subject.rethreshold_legacy_composite_policy(q99)
        self.assertLess(q95.threshold, q97_from_q95.threshold)
        self.assertLess(q97_from_q95.threshold, q99.threshold)
        self.assertEqual(q97_from_q95.threshold, q97_from_q99.threshold)
        self.assertEqual(
            q95.threshold,
            float(
                np.quantile(
                    q95.composite_calibration_raw,
                    subject.LEGACY_COMPOSITE_THRESHOLD_QUANTILE,
                )
            ),
        )
        self.assertEqual(
            q97_from_q95.threshold,
            float(
                np.quantile(
                    q95.composite_calibration_raw,
                    subject.COMPOSITE_THRESHOLD_QUANTILE,
                )
            ),
        )
        self.assertEqual(
            q99.threshold,
            float(
                np.quantile(
                    q95.composite_calibration_raw,
                    subject.FAILED_Q99_COMPOSITE_THRESHOLD_QUANTILE,
                )
            ),
        )
        self.assertEqual(
            q95.stage_one_calibration_raw.tobytes(),
            q99.stage_one_calibration_raw.tobytes(),
        )
        self.assertEqual(
            q95.composite_calibration_raw.tobytes(),
            q99.composite_calibration_raw.tobytes(),
        )
        self.assertEqual(
            q97_from_q95.composite_calibration_raw.tobytes(),
            q99.composite_calibration_raw.tobytes(),
        )
        self.assertEqual(q95.stage_two_threshold, q99.stage_two_threshold)
        self.assertEqual(
            q97_from_q95.stage_two_threshold, q99.stage_two_threshold
        )

    def test_exact_thresholds_are_bound_to_frozen_q99_evidence(self) -> None:
        path = (
            HERE.parent
            / "artifacts"
            / "invariant_patch"
            / "v3_scale"
            / "staged_design_v34_q99_rejector4k_budget001_seed20260952"
            / "v3_staged_composite_policy.npz"
        )
        self.assertTrue(path.is_file(), f"missing frozen q99 evidence: {path}")
        q99 = subject.load_composite_policy(
            path,
            expected_policy_version=subject.FAILED_Q99_STAGED_POLICY_VERSION,
        )
        for quantile, expected in (
            (
                subject.LEGACY_COMPOSITE_THRESHOLD_QUANTILE,
                subject.FROZEN_POLICY_V2_Q95_ENROLLMENT_THRESHOLD,
            ),
            (
                subject.COMPOSITE_THRESHOLD_QUANTILE,
                subject.FROZEN_POLICY_V4_Q97_ENROLLMENT_THRESHOLD,
            ),
            (
                subject.FAILED_Q99_COMPOSITE_THRESHOLD_QUANTILE,
                subject.FROZEN_POLICY_V3_Q99_ENROLLMENT_THRESHOLD,
            ),
        ):
            self.assertEqual(
                float(np.quantile(q99.composite_calibration_raw, quantile)),
                expected,
            )

    def test_old_artifact_loads_only_when_old_version_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = self._legacy_saved(Path(root))
            with self.assertRaisesRegex(ValueError, "policy version mismatch"):
                subject.load_composite_policy(path)
            legacy = subject.load_composite_policy(
                path,
                expected_policy_version=subject.LEGACY_STAGED_POLICY_VERSION,
            )
        self.assertEqual(
            legacy.policy_spec, subject.LEGACY_STAGED_POLICY_SPEC
        )
        self.assertEqual(
            legacy.provenance()["threshold_quantile"], 0.95
        )

    def test_failed_q99_loads_only_when_its_version_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = self._failed_q99_saved(Path(root))
            with self.assertRaisesRegex(ValueError, "policy version mismatch"):
                subject.load_composite_policy(path)
            failed = subject.load_composite_policy(
                path,
                expected_policy_version=(
                    subject.FAILED_Q99_STAGED_POLICY_VERSION
                ),
            )
        self.assertEqual(
            failed.policy_spec, subject.FAILED_Q99_STAGED_POLICY_SPEC
        )
        self.assertEqual(failed.provenance()["threshold_quantile"], 0.99)

    def test_q95_and_q99_historical_specs_cannot_cross_load(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            base_path = Path(root)
            q95_path = self._legacy_saved(base_path / "q95")
            q99_path = self._failed_q99_saved(base_path / "q99")
            for path, wrong_version in (
                (q95_path, subject.FAILED_Q99_STAGED_POLICY_VERSION),
                (q99_path, subject.LEGACY_STAGED_POLICY_VERSION),
            ):
                with self.assertRaisesRegex(
                    ValueError, "policy version mismatch"
                ):
                    subject.load_composite_policy(
                        path, expected_policy_version=wrong_version
                    )

    def test_current_artifact_is_refused_under_historical_versions(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = self._saved(Path(root))
            for version in (
                subject.LEGACY_STAGED_POLICY_VERSION,
                subject.FAILED_Q99_STAGED_POLICY_VERSION,
            ):
                with self.assertRaisesRegex(
                    ValueError, "policy version mismatch"
                ):
                    subject.load_composite_policy(
                        path, expected_policy_version=version
                    )

    def test_a_wrong_policy_version_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = self._saved(Path(root))
            self._tamper(path, policy_version=np.asarray("v1-of-something"))
            with self.assertRaisesRegex(ValueError, "policy version mismatch"):
                subject.load_composite_policy(path)

    def test_a_wrong_schema_or_kind_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = self._saved(Path(root))
            self._tamper(path, schema=np.asarray(1, dtype=np.int64))
            with self.assertRaisesRegex(ValueError, "policy version mismatch"):
                subject.load_composite_policy(path)

    def test_a_tampered_threshold_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = self._saved(Path(root))
            self._tamper(path, threshold=np.asarray(0.123456, dtype=np.float64))
            with self.assertRaisesRegex(ValueError, "frozen quantile"):
                subject.load_composite_policy(path)

    def test_policy_v4_hygiene_counts_cannot_be_tampered(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = self._saved(Path(root))
            self._tamper(
                path,
                selection_rows_used_for_threshold=np.asarray(
                    1, dtype=np.int64
                ),
            )
            with self.assertRaisesRegex(ValueError, "may not be weakened"):
                subject.load_composite_policy(path)

    def test_a_stage_two_threshold_mismatch_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = self._saved(Path(root))
            with self.assertRaisesRegex(ValueError, "different stage-2"):
                subject.load_composite_policy(
                    path, expected_stage_two_threshold=0.5
                )


def _outcome(
    gated,
    stage_two,
    *,
    composite=None,
    feature_seconds=0.0,
    stage_one_seconds=0.0,
    stage_two_seconds=0.0,
) -> "subject.StagedOutcome":
    """A synthetic outcome; by default the composite equals the stage-2 score
    (a valid max whenever the stage-1 survivor rank is below it)."""
    fired = np.asarray(gated, dtype=bool)
    two = np.asarray(stage_two, dtype=np.float64)
    comp = (
        two.copy()
        if composite is None
        else np.asarray(composite, dtype=np.float64)
    )
    raw = np.where(fired, 5.0, -5.0)
    rank = subject._bounded_rank(raw)
    survivor_rank = np.where(fired, np.nan, 0.0)
    staged = np.where(fired, subject.STAGE_ONE_SCORE_OFFSET + rank, comp)
    return subject.StagedOutcome(
        gated=fired,
        stage_one_raw=raw,
        stage_one_rank=rank,
        stage_two_score=two,
        stage_one_survivor_rank=survivor_rank,
        composite_score=comp,
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
                gated,
                stage_two,
                outcome.composite_score,
                outcome.staged_score,
                threshold,
            )
            np.testing.assert_array_equal(
                outcome.staged_score > threshold,
                gated | (np.nan_to_num(stage_two, nan=-1.0) > threshold),
            )

    def test_the_survivor_decision_is_made_on_the_composite(self) -> None:
        """The point of policy version 2: a survivor whose stage-2 rank is
        below the threshold but whose composite is above it IS rejected."""
        gated = np.array([False, False])
        stage_two = np.array([0.40, 0.40])
        composite = np.array([0.90, 0.40])
        outcome = _outcome(gated, stage_two, composite=composite)
        subject.assert_staged_decision_equivalence(
            gated, stage_two, composite, outcome.staged_score, 0.5
        )
        np.testing.assert_array_equal(
            outcome.staged_score > 0.5, np.array([True, False])
        )

    def test_a_composite_below_its_stage_two_score_is_refused(self) -> None:
        gated = np.array([False])
        with self.assertRaisesRegex(AssertionError, "required to be a max"):
            subject.assert_staged_decision_equivalence(
                gated,
                np.array([0.6]),
                np.array([0.4]),
                np.array([0.4]),
                0.5,
            )

    def test_a_gated_row_carrying_a_stage_two_score_is_refused(self) -> None:
        """That would mean the short circuit did not happen."""
        gated = np.array([True, False])
        stage_two = np.array([0.4, 0.2])
        composite = np.array([np.nan, 0.2])
        staged = np.array([1.5, 0.2])
        with self.assertRaisesRegex(AssertionError, "short circuit did not"):
            subject.assert_staged_decision_equivalence(
                gated, stage_two, composite, staged, 0.5
            )

    def test_a_gated_row_carrying_a_composite_score_is_refused(self) -> None:
        gated = np.array([True])
        with self.assertRaisesRegex(AssertionError, "composite score"):
            subject.assert_staged_decision_equivalence(
                gated,
                np.array([np.nan]),
                np.array([0.4]),
                np.array([1.5]),
                0.5,
            )

    def test_a_survivor_without_a_stage_two_score_is_refused(self) -> None:
        with self.assertRaisesRegex(AssertionError, "no stage-2 score"):
            subject.assert_staged_decision_equivalence(
                np.array([False]),
                np.array([np.nan]),
                np.array([0.2]),
                np.array([0.2]),
                0.5,
            )

    def test_a_survivor_without_a_composite_score_is_refused(self) -> None:
        with self.assertRaisesRegex(AssertionError, "no composite score"):
            subject.assert_staged_decision_equivalence(
                np.array([False]),
                np.array([0.2]),
                np.array([np.nan]),
                np.array([0.2]),
                0.5,
            )

    def test_a_misaligned_or_non_finite_staged_score_is_refused(self) -> None:
        with self.assertRaisesRegex(AssertionError, "not aligned"):
            subject.assert_staged_decision_equivalence(
                np.array([True]),
                np.array([np.nan]),
                np.array([np.nan]),
                np.array([1.1, 1.2]),
                0.5,
            )
        with self.assertRaisesRegex(AssertionError, "non-finite"):
            subject.assert_staged_decision_equivalence(
                np.array([True]),
                np.array([np.nan]),
                np.array([np.nan]),
                np.array([np.nan]),
                0.5,
            )

    def test_a_staged_score_that_breaks_the_decision_is_refused(self) -> None:
        with self.assertRaisesRegex(AssertionError, "does not reproduce"):
            subject.assert_staged_decision_equivalence(
                np.array([True, False]),
                np.array([np.nan, 0.9]),
                np.array([np.nan, 0.9]),
                np.array([0.1, 0.9]),
                0.5,
            )

    def test_a_staged_axis_that_leaves_the_composite_is_refused(self) -> None:
        with self.assertRaisesRegex(AssertionError, "does not equal"):
            subject.assert_staged_decision_equivalence(
                np.array([False]),
                np.array([0.2]),
                np.array([0.3]),
                np.array([0.2]),
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
            outcome, np.array([0.05, 0.99, 0.10, 0.20]), 0.5, 0.5
        )
        self.assertEqual(report["rows"], 4)
        self.assertEqual(report["stage_one_gate_rate"], 0.25)
        self.assertEqual(report["stage_two_false_unknown_rate_marginal"], 0.25)
        self.assertAlmostEqual(
            report["stage_two_false_unknown_rate_among_survivors"], 1.0 / 3.0
        )
        self.assertEqual(report["staged_false_unknown_rate"], 0.5)
        self.assertEqual(report["unstaged_false_unknown_rate"], 0.25)
        self.assertEqual(report["staged_rejected_total"], 2)
        self.assertEqual(report["rejected_by_stage_one_total"], 1)
        self.assertEqual(
            report["stage_one_also_rejected_by_unstaged_control"], 0
        )
        self.assertEqual(report["rejected_by_stage_two_only"], 1)
        self.assertEqual(report["attribution_partition_total"], 2)
        self.assertTrue(report["attribution_partition_exact"])

    def test_a_prefilter_only_rejection_is_visible(self) -> None:
        outcome = _outcome([True, False], [np.nan, 0.1])
        report = subject.known_false_unknown_by_stage(
            outcome, np.array([0.1, 0.1]), 0.5, 0.5
        )
        self.assertEqual(report["staged_false_unknown_rate"], 0.5)
        self.assertEqual(report["unstaged_false_unknown_rate"], 0.0)
        self.assertEqual(report["rejected_by_stage_one_total"], 1)
        self.assertEqual(
            report["stage_one_also_rejected_by_unstaged_control"], 0
        )
        self.assertEqual(report["rejected_by_stage_two_only"], 0)
        self.assertEqual(
            report["staged_rejected_total"],
            report["rejected_by_stage_one_total"]
            + report["rejected_by_stage_two_only"],
        )

    def test_stage_one_overlap_with_unstaged_control_is_not_subtracted(self):
        outcome = _outcome([True, False], [np.nan, 0.1])
        report = subject.known_false_unknown_by_stage(
            outcome, np.array([0.9, 0.1]), 0.5, 0.5
        )
        self.assertEqual(report["rejected_by_stage_one_total"], 1)
        self.assertEqual(
            report["stage_one_also_rejected_by_unstaged_control"], 1
        )
        self.assertEqual(report["rejected_by_stage_two_only"], 0)
        self.assertEqual(report["staged_rejected_total"], 1)
        self.assertTrue(report["attribution_partition_exact"])

    def test_a_fully_gated_population_reports_no_survivor_rate(self) -> None:
        outcome = _outcome([True, True], [np.nan, np.nan])
        report = subject.known_false_unknown_by_stage(
            outcome, np.array([0.1, 0.1]), 0.5, 0.5
        )
        self.assertEqual(report["staged_false_unknown_rate"], 1.0)
        self.assertIsNone(
            report["stage_two_false_unknown_rate_among_survivors"]
        )

    def test_an_empty_population_is_refused(self) -> None:
        outcome = _outcome([], [])
        with self.assertRaisesRegex(ValueError, "at least one row"):
            subject.known_false_unknown_by_stage(
                outcome, np.zeros(0), 0.5, 0.5
            )

    def test_survivor_rejection_is_on_the_composite_axis(self) -> None:
        """A survivor below the threshold on stage-2 but above it on the
        composite is a staged rejection; the unstaged control judges the
        additive axis against its own threshold."""
        outcome = _outcome(
            [False, False],
            [0.40, 0.40],
            composite=[0.90, 0.40],
        )
        report = subject.known_false_unknown_by_stage(
            outcome, np.array([0.40, 0.40]), 0.5, 0.5
        )
        self.assertEqual(report["staged_false_unknown_rate"], 0.5)
        self.assertEqual(report["unstaged_false_unknown_rate"], 0.0)
        self.assertEqual(report["staged_threshold"], 0.5)
        self.assertEqual(report["unstaged_threshold"], 0.5)

    def test_the_two_thresholds_act_on_their_own_axes(self) -> None:
        outcome = _outcome([False], [0.6])
        report = subject.known_false_unknown_by_stage(
            outcome, np.array([0.6]), 0.7, 0.5
        )
        # Composite 0.6 <= staged threshold 0.7: kept.  Additive 0.6 >
        # unstaged threshold 0.5: control rejects.
        self.assertEqual(report["staged_false_unknown_rate"], 0.0)
        self.assertEqual(report["unstaged_false_unknown_rate"], 1.0)


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

    def test_the_training_population_is_never_extracted(self) -> None:
        """Not computing training features is the fit-nothing guarantee;
        enrollment and selection are the two calibrate/score populations."""
        recorder = self._recorder()
        self._feed(recorder, (5,))
        self.assertEqual(recorder.matrices, {})
        self._feed(recorder, (3,))
        self.assertEqual(set(recorder.matrices), {"en"})
        self._feed(recorder, (2,))
        self.assertEqual(set(recorder.matrices), {"en", "va"})
        self.assertEqual(
            recorder.matrices["en"].shape, (3, pdg.FEATURE_COUNT)
        )
        self.assertEqual(
            recorder.matrices["va"].shape, (2, pdg.FEATURE_COUNT)
        )

    def test_the_recorded_populations_are_keyed_and_measured(self) -> None:
        recorder = self._recorder()
        self._feed(recorder, (5, 3, 2))
        recorded = subject._recorded_prefilter(
            recorder,
            {"xtr": np.zeros(5), "xen": np.zeros(3), "xva": np.zeros(2)},
        )
        self.assertEqual(set(recorded), {"en", "va"})
        enroll_features, enroll_length = recorded["en"]
        selection_features, selection_length = recorded["va"]
        self.assertEqual(enroll_features.shape, (3, pdg.FEATURE_COUNT))
        self.assertEqual(selection_features.shape, (2, pdg.FEATURE_COUNT))
        self.assertEqual(enroll_length, 1024)
        self.assertEqual(selection_length, 1024)

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
    def _base_fixtures(
        self, seeds=(UNIT_NOVELTY_SEED,), lengths=TEST_LENGTHS
    ):
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

    def _features(
        self, provenance, seeds=(UNIT_NOVELTY_SEED,), lengths=TEST_LENGTHS
    ):
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
                entry = features[UNIT_NOVELTY_SEED][family][length]
                self.assertEqual(entry.features.shape, (2, pdg.FEATURE_COUNT))
                self.assertTrue(np.isfinite(entry.features).all())
                self.assertGreaterEqual(entry.seconds, 0.0)

    def test_the_captures_are_verified_against_the_base_provenance(self) -> None:
        """This is what proves stage 1 and stage 2 scored the same rows."""
        _fixtures, provenance = self._base_fixtures()
        tampered = json.loads(json.dumps(provenance))
        tampered[str(UNIT_NOVELTY_SEED)]["noise"][
            "longest_capture_sha256"
        ] = "0" * 64
        with self.assertRaisesRegex(AssertionError, "do not match the digest"):
            self._features(tampered)

    def test_a_tampered_prefix_digest_is_caught(self) -> None:
        _fixtures, provenance = self._base_fixtures()
        tampered = json.loads(json.dumps(provenance))
        tampered[str(UNIT_NOVELTY_SEED)]["chirp"]["prefix_sha256"][
            str(TEST_LENGTHS[0])
        ] = "0" * 64
        with self.assertRaisesRegex(AssertionError, "prefix digest"):
            self._features(tampered)

    def test_generation_is_deterministic(self) -> None:
        _fixtures, provenance = self._base_fixtures()
        first = self._features(provenance)
        second = self._features(provenance)
        np.testing.assert_array_equal(
            first[UNIT_NOVELTY_SEED]["noise"][TEST_LENGTHS[0]].features,
            second[UNIT_NOVELTY_SEED]["noise"][TEST_LENGTHS[0]].features,
        )

    def test_noise_and_chirp_features_differ(self) -> None:
        _fixtures, provenance = self._base_fixtures()
        features = self._features(provenance)
        noise = features[UNIT_NOVELTY_SEED]["noise"][
            TEST_LENGTHS[1]
        ].features
        chirp = features[UNIT_NOVELTY_SEED]["chirp"][
            TEST_LENGTHS[1]
        ].features
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
                (UNIT_NOVELTY_SEED,),
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
                (UNIT_NOVELTY_SEED,),
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
                features[UNIT_NOVELTY_SEED][family][4096].features,
                features[UNIT_NOVELTY_SEED][family][
                    longest_fitted
                ].features,
            )
            for length in TEST_LENGTHS:
                np.testing.assert_array_equal(
                    features[UNIT_NOVELTY_SEED][family][length].features,
                    plain[UNIT_NOVELTY_SEED][family][length].features,
                )
            self.assertFalse(
                np.array_equal(
                    features[UNIT_NOVELTY_SEED][family][4096].features,
                    plain[UNIT_NOVELTY_SEED][family][4096].features,
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
            # An enrollment population with a couple of stage-1-gated rows, so
            # the composite calibration is fit on a strict survivor subset
            # rather than trivially on everything.
            enrollment_features=np.vstack(
                [
                    _pose_rows(43, 4444),
                    _pose_rows(2, 4445, shift=6.0),
                ]
            ),
            enrollment_capture_length=KNOWN_LENGTH,
            enrollment_feature_seconds=0.20,
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
            "novelty_seeds": [UNIT_NOVELTY_SEED],
            "novelty_n": 2,
            "prefix_lengths": list(TEST_LENGTHS),
            "reproduction_tolerance": 1e-4,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def _run(self, root: Path, **overrides) -> dict:
        args = self._args(root, **overrides)
        rebuilt = self._rebuilt(root)
        post_design_ledger = {
            **subject.SPENT_NOVELTY_SEEDS,
            DESIGN_SEED: "synthetic unit-test design freeze; not evidence",
        }
        active_ledger = (
            post_design_ledger
            if args.role == "validate"
            else dict(subject.SPENT_NOVELTY_SEEDS)
        )
        validation_seed_contract = (
            (UNIT_NOVELTY_SEED,)
            if args.role == "validate"
            else subject.DEFAULT_VALIDATION_NOVELTY_SEEDS
        )
        with mock.patch.dict(
            subject.SPENT_NOVELTY_SEEDS, active_ledger, clear=True
        ), mock.patch.object(
            subject,
            "DEFAULT_VALIDATION_NOVELTY_SEEDS",
            validation_seed_contract,
        ), mock.patch.object(
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
        self.assertFalse(report["release_seed_20260736_used"])
        self.assertEqual(
            report["consumed_sealed_release_seeds_excluded"],
            [20260729, 20260731, 20260733, 20260734, 20260735],
        )
        self.assertTrue(report["development_only"])
        self.assertFalse(report["release_evidence"])
        contract = report["fitting_contract"]
        self.assertEqual(contract["density_population"], "training only")
        self.assertEqual(
            contract["rank_and_threshold_population"], "enrollment only"
        )
        self.assertEqual(
            contract["composite_rank_and_threshold_population"],
            "enrollment stage-1 survivors only",
        )
        self.assertFalse(contract["policy_searched_here"])
        self.assertFalse(contract["stage_one_fitted_in_this_run"])
        self.assertFalse(contract["stage_one_training_features_computed_here"])
        self.assertTrue(contract["stage_one_enrollment_features_computed_here"])
        self.assertIn(
            "composite", contract["stage_one_enrollment_features_role"]
        )
        self.assertEqual(
            report["seeds"][
                "novelty_rows_used_to_choose_the_stage_one_operating_point"
            ],
            0,
        )
        self.assertEqual(
            report["seeds"][
                "novelty_rows_used_to_choose_the_composite_threshold"
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

    # -- the composite policy (staged policy version 4 / q97) ---------------

    def test_the_report_states_the_policy_version_it_validates(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            report = self._run(Path(root))
        architecture = report["architecture"]
        self.assertEqual(architecture["kind"], subject.STAGED_POLICY_KIND)
        self.assertEqual(architecture["schema"], subject.STAGED_POLICY_SCHEMA)
        self.assertEqual(
            architecture["staged_policy_version"],
            subject.STAGED_POLICY_VERSION,
        )
        self.assertEqual(
            architecture["survivor_score"], subject.COMPOSITE_SURVIVOR_SCORE
        )
        self.assertIn("composite", architecture)
        self.assertIn("q99", architecture["why_composite"])
        self.assertIn("q97", architecture["why_composite"])
        self.assertIn("both CNN fusions", architecture["why_composite"])
        self.assertIn("remain frozen", architecture["why_composite"])

    def test_the_composite_block_is_recorded_and_consistent(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            report = self._run(Path(root))
            composite_path = (
                Path(root) / "openset" / subject.COMPOSITE_POLICY_FILENAME
            )
            self.assertTrue(composite_path.is_file())
            loaded = subject.load_composite_policy(
                composite_path,
                expected_stage_two_threshold=report["stage_two"]["threshold"],
            )
        block = report["composite"]
        self.assertEqual(
            block["policy_version"], subject.STAGED_POLICY_VERSION
        )
        self.assertEqual(block["threshold"], loaded.threshold)
        self.assertEqual(
            block["threshold_quantile"],
            float(subject.COMPOSITE_THRESHOLD_QUANTILE),
        )
        self.assertEqual(
            block["threshold_population"],
            "enrollment stage-1 survivors only (enrollment only, as every "
            "rank and threshold in this chain)",
        )
        self.assertEqual(
            block["stage_two_threshold_unchanged"],
            report["stage_two"]["threshold"],
        )
        self.assertEqual(
            block["enrollment_rows"],
            block["enrollment_gated_rows"]
            + block["enrollment_survivor_rows"],
        )
        self.assertGreater(block["enrollment_gated_rows"], 0)
        self.assertEqual(len(block["enrollment_features_sha256"]), 64)
        # The staged decision threshold everywhere is the composite one.
        self.assertEqual(
            report["known"]["by_stage"]["staged_threshold"],
            block["threshold"],
        )
        self.assertEqual(
            report["known"]["by_stage"]["unstaged_threshold"],
            report["stage_two"]["threshold"],
        )

    def test_the_composite_npz_is_a_pinned_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            report = self._run(Path(root))
        self.assertEqual(
            set(report["artifacts"]),
            {
                "v3_open_policy_stage_two.npz",
                "v3_branch_lof_components.npz",
                subject.COMPOSITE_POLICY_FILENAME,
            },
        )
        for digest in report["artifacts"].values():
            self.assertEqual(len(digest), 64)

    # -- gates, at every seed and length -----------------------------------

    def test_the_full_gate_set_is_reported_at_every_seed_and_length(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            report = self._run(Path(root))
        self.assertEqual(set(report["novelty"]), {str(UNIT_NOVELTY_SEED)})
        for length in TEST_LENGTHS:
            row = report["novelty"][str(UNIT_NOVELTY_SEED)][str(length)]
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
        rows = report["novelty"][str(UNIT_NOVELTY_SEED)]
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
        self.assertGreater(stages["rejected_by_stage_one_total"], 0)
        self.assertEqual(
            stages["staged_rejected_total"],
            stages["rejected_by_stage_one_total"]
            + stages["rejected_by_stage_two_only"],
        )
        self.assertTrue(stages["attribution_partition_exact"])
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

    def test_v34_refuses_a_non_frozen_stage_one_budget(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            directory = _bundle_set(
                Path(root) / "wrong-budget", budget=0.90
            )
            with self.assertRaisesRegex(
                RuntimeError, "expected the frozen 0.01"
            ):
                self._run(Path(root), prefilter_dir=str(directory))

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
            with _temporarily_unspent_design_seed():
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
            first["novelty"][str(UNIT_NOVELTY_SEED)][str(TEST_LENGTHS[0])][
                "noise"
            ]["auroc"],
            second["novelty"][str(UNIT_NOVELTY_SEED)][str(TEST_LENGTHS[0])][
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
                self._run(Path(root), novelty_seeds=[20260736])

    def test_a_consumed_sealed_seed_stops_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "consumed sealed"):
                self._run(Path(root), novelty_seeds=[20260735])

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
