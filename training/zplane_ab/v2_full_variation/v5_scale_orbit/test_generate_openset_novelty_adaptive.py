from __future__ import annotations

import copy
from contextlib import contextmanager, ExitStack
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import unittest
from unittest import mock

import numpy as np

import generate_openset_novelty_adaptive as generator


EXPECTED_DERIVED_SEEDS = {
    20263005: {
        "no_signal": 3959093290574385373,
        "noise": 6906390931327956509,
        "chirp": 6697928276738510837,
    },
    20263006: {
        "no_signal": 1962966655383687455,
        "noise": 6325156090276620543,
        "chirp": 12179427047883686052,
    },
    20263007: {
        "no_signal": 9819065926392326122,
        "noise": 919859913731889266,
        "chirp": 9089325103754641002,
    },
    20263008: {
        "no_signal": 7814752606972902720,
        "noise": 12640966810523450853,
        "chirp": 38753718437833902,
    },
}

TINY_GOLDEN = {
    "no_signal": {
        "longest":
            "c76903cde8580d1c809ac5352aab33af5a310ad05126294d66e06db880c463ed",
        "prefix": {
            5: "6edd9f6f9cc92cded36e6c4a580933f9c9f1b90562b46903b806f21902a1a54f",
            11: "44b8aa4d28701168922acf61435ea4bb442f97b0b14ad7a2510ed68874ee2a72",
            17: "c76903cde8580d1c809ac5352aab33af5a310ad05126294d66e06db880c463ed",
        },
    },
    "noise": {
        "longest":
            "e49f63755ba7c8abeefb80461baf695197161192eef0a190282af5a381f6b184",
        "prefix": {
            5: "2d135972ecd0c43035918ab243890a575e5bcf434b882dae69de0c07f1162f63",
            11: "d51fbf0ead543911173c0efffd4fcc865110608a61fb749ac950329627ec87b3",
            17: "e49f63755ba7c8abeefb80461baf695197161192eef0a190282af5a381f6b184",
        },
    },
    "chirp": {
        "longest":
            "cc43367ba0b6c4716ffc775f3bd4e214215b398ee23899cfef9627789ad3ee30",
        "prefix": {
            5: "c6dccc9408148915b3d9b233df79f9441415930b943eb92eda4edabb525e9a94",
            11: "1b650dbc674bee2f34b61dd7382314d6bdb8780275f2e11bc9dcfe48c563d659",
            17: "cc43367ba0b6c4716ffc775f3bd4e214215b398ee23899cfef9627789ad3ee30",
        },
    },
}


class AdaptiveNoveltyGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.amendment = json.loads(
            generator.AMENDMENT_PATH.read_text(encoding="utf-8")
        )
        cls.runtime = generator.runtime_fingerprint()
        cls.generator_sha256 = hashlib.sha256(
            generator.GENERATOR_PATH.read_bytes()
        ).hexdigest()
        cls.freeze_relative = (
            "training/zplane_ab/v2_full_variation/v5_scale_orbit/"
            "future_novelty_pre_generation_freeze.json"
        )
        cls.output_relative = (
            ".artifacts/v5-scale-orbit-adaptive-novelty-seeds-"
            "20263005-20263008"
        )

    def valid_freeze(self) -> dict:
        return {
            "schema": generator.PRE_GENERATION_FREEZE_SCHEMA,
            "schema_version": 1,
            "status": generator.PRE_GENERATION_FREEZE_STATUS,
            "lineage": "v5-scale-orbit",
            "development_only": True,
            "release_evidence": False,
            "append_only": True,
            "authorization": {
                "owner_authorized_first_adaptive_generation": True,
                "final_generator_source_reviewed": True,
                "generation_started_before_freeze": False,
                "adaptive_novelty_scoring_started_before_freeze": False,
                "sealed_novelty_generated_read_or_scored": False,
            },
            "parent_bindings": {
                "attempt_2_source": {
                    "commit": generator.ATTEMPT_2_SOURCE_COMMIT,
                    "commit_subject": generator.ATTEMPT_2_SOURCE_SUBJECT,
                },
                "adaptive_novelty_amendment": {
                    "relative_path": generator.AMENDMENT_RELATIVE_PATH,
                    "sha256": generator.AMENDMENT_SHA256,
                },
                "adaptive_novelty_amendment_validator": {
                    "relative_path":
                        generator.AMENDMENT_VALIDATOR_RELATIVE_PATH,
                    "sha256": generator.AMENDMENT_VALIDATOR_SHA256,
                },
            },
            "generator_source": {
                "relative_path": generator.GENERATOR_RELATIVE_PATH,
                "sha256": self.generator_sha256,
                "commit": "a" * 40,
                "commit_subject": "Freeze final v5 adaptive novelty generator",
            },
            "transitively_executed_project_generation_helpers": [],
            "runtime": self.runtime,
            "inventory": generator.frozen_inventory_contract(),
            "mechanics": generator.generator_mechanics_contract(),
            "command_configuration": {
                "entrypoint_relative_path":
                    generator.GENERATOR_RELATIVE_PATH,
                "pre_generation_freeze_relative_path":
                    self.freeze_relative,
                "output_directory_relative_path": self.output_relative,
                "device": "cpu",
                "embedding_batch_size": 256,
                "generation_has_no_device_or_batch_parameter": True,
                "no_seed_family_row_count_length_or_dtype_override": True,
            },
            "first_generation_manifest_contract": {
                "filename": generator.DIGEST_MANIFEST_FILENAME,
                "required_before_scoring": True,
                "immutable_after_first_generation": True,
                "binds_pre_generation_source_freeze_sha256": True,
                "binds_each_family_seed_longest_complex64_sha256": True,
                "binds_each_family_seed_length_prefix_complex64_sha256": True,
                "binds_final_generator_source_and_runtime_environment": True,
                "deterministic_regeneration_allowed_only_when_every_digest_matches":
                    True,
            },
        }

    @contextmanager
    def mocked_authorization_environment(self):
        freeze_path = generator.REPO / self.freeze_relative
        output_path = generator.REPO / self.output_relative
        document = self.valid_freeze()
        source_commit = document["generator_source"]["commit"]
        source_subject = document["generator_source"]["commit_subject"]
        freeze_commit = "c" * 40
        freeze_bytes = b"exact committed future freeze bytes"
        freeze_sha256 = hashlib.sha256(freeze_bytes).hexdigest()
        generator_bytes = generator.GENERATOR_PATH.read_bytes()
        state = {"failure": None}

        def fake_sha256(path: Path) -> str:
            resolved = path.resolve()
            if resolved == generator.GENERATOR_PATH.resolve():
                if state["failure"] == "current_generator":
                    return "0" * 64
                return self.generator_sha256
            if resolved == freeze_path.resolve():
                if state["failure"] == "current_freeze":
                    return "0" * 64
                return freeze_sha256
            if resolved == generator.AMENDMENT_PATH.resolve():
                return generator.AMENDMENT_SHA256
            if resolved == generator.AMENDMENT_VALIDATOR_PATH.resolve():
                return generator.AMENDMENT_VALIDATOR_SHA256
            raise AssertionError(f"unexpected SHA path {resolved}")

        def completed(
            args: tuple[str, ...],
            *,
            stdout="",
            returncode: int = 0,
        ) -> subprocess.CompletedProcess:
            return subprocess.CompletedProcess(
                ["git", *args],
                returncode,
                stdout=stdout,
                stderr=b"" if isinstance(stdout, bytes) else "",
            )

        def fake_git(*args: str, **kwargs):
            if args == (
                "log",
                "-1",
                "--format=%H",
                "--",
                self.freeze_relative,
            ):
                return completed(args, stdout=freeze_commit)
            if args == (
                "show",
                f"{freeze_commit}:{self.freeze_relative}",
            ):
                content = (
                    b"changed committed freeze"
                    if state["failure"] == "freeze_blob"
                    else freeze_bytes
                )
                return completed(args, stdout=content)
            if args == (
                "show",
                "-s",
                "--format=%H%n%P%n%s",
                generator.ATTEMPT_2_SOURCE_COMMIT,
            ):
                lines = [
                    generator.ATTEMPT_2_SOURCE_COMMIT,
                    generator.ATTEMPT_2_PREREGISTRATION_COMMIT,
                    generator.ATTEMPT_2_SOURCE_SUBJECT,
                ]
                if state["failure"] == "attempt_2_commit":
                    lines[1] = "0" * 40
                return completed(args, stdout="\n".join(lines) + "\n")
            if args == (
                "show",
                "-s",
                "--format=%H%n%s",
                source_commit,
            ):
                lines = [source_commit, source_subject]
                if state["failure"] == "source_commit":
                    lines[1] = "changed subject"
                return completed(args, stdout="\n".join(lines) + "\n")
            if args == (
                "show",
                f"{source_commit}:{generator.GENERATOR_RELATIVE_PATH}",
            ):
                content = (
                    b"changed committed generator"
                    if state["failure"] == "generator_blob"
                    else generator_bytes
                )
                return completed(args, stdout=content)
            if args[:2] == ("merge-base", "--is-ancestor"):
                pair = args[2:]
                failed_pair_by_name = {
                    "attempt_2_to_source": (
                        generator.ATTEMPT_2_SOURCE_COMMIT,
                        source_commit,
                    ),
                    "attempt_2_to_freeze": (
                        generator.ATTEMPT_2_SOURCE_COMMIT,
                        freeze_commit,
                    ),
                    "source_to_freeze": (source_commit, freeze_commit),
                }
                return completed(
                    args,
                    returncode=(
                        1
                        if failed_pair_by_name.get(state["failure"]) == pair
                        else 0
                    ),
                )
            raise AssertionError(f"unexpected git call: {args!r}, {kwargs!r}")

        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(Path, "is_file", return_value=True)
            )
            stack.enter_context(
                mock.patch.object(
                    generator,
                    "_read_json_object",
                    return_value=document,
                )
            )
            stack.enter_context(
                mock.patch.object(
                    generator,
                    "_sha256_file",
                    side_effect=fake_sha256,
                )
            )
            stack.enter_context(
                mock.patch.object(
                    generator,
                    "runtime_fingerprint",
                    return_value=self.runtime,
                )
            )
            stack.enter_context(
                mock.patch.object(
                    generator,
                    "_git",
                    side_effect=fake_git,
                )
            )
            yield {
                "freeze_path": freeze_path,
                "output_path": output_path,
                "document": document,
                "state": state,
                "freeze_sha256": freeze_sha256,
                "freeze_commit": freeze_commit,
                "source_commit": source_commit,
            }

    def synthetic_digest_records(self) -> list[dict]:
        records = []
        byte_count = (
            generator.ROWS_PER_FAMILY_PER_SEED
            * generator.MAXIMUM_LENGTH
            * generator.CANONICAL_COMPLEX_DTYPE.itemsize
        )
        for seed in generator.ADAPTIVE_SEEDS:
            for family in generator.NOVELTY_FAMILIES:
                longest = hashlib.sha256(
                    f"{seed}:{family}:16384".encode()
                ).hexdigest()
                records.append(
                    {
                        "base_seed": seed,
                        "family": family,
                        "derived_seed":
                            generator.derive_family_seed(seed, family),
                        "row_count": generator.ROWS_PER_FAMILY_PER_SEED,
                        "generated_once_at_maximum_length":
                            generator.MAXIMUM_LENGTH,
                        "dtype": generator.OUTPUT_DTYPE_NAME,
                        "canonical_byte_order":
                            "little_endian_complex64_interleaved_real_imag",
                        "longest_complex64_sha256": longest,
                        "prefix_complex64_sha256": {
                            "4096": hashlib.sha256(
                                f"{seed}:{family}:4096".encode()
                            ).hexdigest(),
                            "8192": hashlib.sha256(
                                f"{seed}:{family}:8192".encode()
                            ).hexdigest(),
                            "16384": longest,
                        },
                        "shorter_observation_rule":
                            "exact leading causal raw prefix of the same longest row",
                        "uses_frequency_transform": False,
                        "raw_relative_path":
                            generator._raw_relative_path(seed, family),
                        "raw_byte_count": byte_count,
                        "raw_sha256": longest,
                    }
                )
        return records

    def test_amendment_inventory_and_mechanics_are_implemented_exactly(
        self,
    ) -> None:
        inventory = self.amendment["adaptive_novelty_inventory"]
        frozen = generator.frozen_inventory_contract()
        for key in (
            "seeds",
            "all_four_seeds_consumed_and_reported_jointly",
            "seed_cherry_pick_or_omission_allowed",
            "families",
            "rows_per_family_per_seed",
            "generated_once_at_maximum_length",
            "dtype",
            "observation_lengths",
            "shorter_observation_rule",
            "longest_rows_total",
            "routes",
            "exact_cell_key_fields",
            "exact_cell_count",
            "rows_per_cell",
            "exact_route_score_count",
        ):
            self.assertEqual(frozen[key], inventory[key])

        normative = self.amendment["normative_generator_mechanics"]
        self.assertFalse(normative["uses_frequency_transform"])
        self.assertEqual(frozen["rng"], normative["rng"])
        self.assertEqual(
            frozen["family_seed_derivation"],
            normative["family_seed_derivation"],
        )
        mechanics = generator.generator_mechanics_contract()
        for family in generator.NOVELTY_FAMILIES:
            self.assertEqual(mechanics[family], normative[family])

    def test_exact_joint_seed_route_length_family_cell_inventory(self) -> None:
        cells = generator.exact_cell_inventory()
        self.assertEqual(len(cells), 72)
        self.assertEqual(
            {
                (
                    row["seed"],
                    row["prototype_source_route"],
                    row["observation_length"],
                    row["novelty_family"],
                )
                for row in cells
            },
            {
                (seed, route, length, family)
                for seed in generator.ADAPTIVE_SEEDS
                for route in generator.PROTOTYPE_SOURCE_ROUTES
                for length in generator.OBSERVATION_LENGTHS
                for family in generator.NOVELTY_FAMILIES
            },
        )
        self.assertTrue(all(row["rows"] == 300 for row in cells))

    def test_all_twelve_family_seed_derivations_are_frozen(self) -> None:
        actual = {
            seed: {
                family: generator.derive_family_seed(seed, family)
                for family in generator.NOVELTY_FAMILIES
            }
            for seed in generator.ADAPTIVE_SEEDS
        }
        self.assertEqual(actual, EXPECTED_DERIVED_SEEDS)
        with self.assertRaises(TypeError):
            generator.derive_family_seed(True, "noise")
        with self.assertRaises(ValueError):
            generator.derive_family_seed(20263005, "wifi")

    def test_tiny_generation_is_deterministic_complex64_and_golden(
        self,
    ) -> None:
        for family in generator.NOVELTY_FAMILIES:
            first = generator.generate_family_rows_for_review(
                20263005,
                family,
                row_count=3,
                length=17,
            )
            second = generator.generate_family_rows_for_review(
                20263005,
                family,
                row_count=3,
                length=17,
            )
            self.assertEqual(first.shape, (3, 17))
            self.assertEqual(first.dtype, np.dtype(np.complex64))
            self.assertTrue(first.flags.c_contiguous)
            np.testing.assert_array_equal(first, second)
            self.assertEqual(
                generator.complex64_sha256(first),
                TINY_GOLDEN[family]["longest"],
            )
            for length, expected in TINY_GOLDEN[family]["prefix"].items():
                self.assertEqual(
                    generator.causal_prefix_complex64_sha256(first, length),
                    expected,
                )
        no_signal_other_seed = (
            generator.generate_family_rows_for_review(
                20263006,
                "no_signal",
                row_count=3,
                length=17,
            )
        )
        self.assertEqual(np.count_nonzero(no_signal_other_seed), 0)

    def test_prefix_digests_are_rowwise_exact_causal_prefixes(self) -> None:
        rows = generator.generate_family_rows_for_review(
            20263005,
            "noise",
            row_count=3,
            length=17,
        )
        manual = hashlib.sha256()
        for row in rows:
            prefix = np.ascontiguousarray(
                row[:11],
                dtype=generator.CANONICAL_COMPLEX_DTYPE,
            )
            manual.update(prefix.view(np.uint8))
        self.assertEqual(
            generator.causal_prefix_complex64_sha256(rows, 11),
            manual.hexdigest(),
        )
        record = generator.family_digest_record(
            rows,
            base_seed=20263005,
            family="noise",
            observation_lengths=(5, 11, 17),
        )
        self.assertEqual(
            record["prefix_complex64_sha256"]["11"],
            manual.hexdigest(),
        )
        self.assertEqual(
            record["longest_complex64_sha256"],
            record["prefix_complex64_sha256"]["17"],
        )

    def test_generation_mechanics_are_transform_profile_and_class_free(
        self,
    ) -> None:
        source = generator.GENERATOR_PATH.read_text(encoding="utf-8")
        self.assertNotIn("np.fft", source)
        self.assertNotIn("scipy.fft", source)
        review_source = inspect.getsource(
            generator.generate_family_rows_for_review
        )
        self.assertNotIn("profile", review_source.lower())
        self.assertNotIn("public_class", review_source.lower())
        self.assertNotIn("sample_rate", review_source.lower())

    def test_public_review_helper_refuses_production_dimensions_before_rng(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            generator.GenerationRefused,
            "production generation requires",
        ):
            generator.generate_family_rows_for_review(
                20263005,
                "noise",
                row_count=300,
                length=16384,
            )
        self.assertFalse(
            hasattr(generator, "_generate_family_rows_unchecked")
        )
        self.assertFalse(
            hasattr(generator, "_mint_generation_authorization")
        )
        self.assertFalse(hasattr(generator, "generate_noise_row"))
        self.assertFalse(hasattr(generator, "generate_chirp_row"))

    def test_future_freeze_validation_is_exact_and_outcome_blind(self) -> None:
        valid = self.valid_freeze()
        self.assertEqual(
            generator.validate_pre_generation_freeze_document(
                valid,
                freeze_relative_path=self.freeze_relative,
                output_relative_path=self.output_relative,
                observed_generator_sha256=self.generator_sha256,
                observed_runtime=self.runtime,
            ),
            "a" * 40,
        )

        for mutation, message in (
            (
                lambda value: value["inventory"]["seeds"].pop(),
                "inventory",
            ),
            (
                lambda value: value["runtime"].update(
                    {"numpy_version": "changed"}
                ),
                "runtime",
            ),
            (
                lambda value: value["authorization"].update(
                    {"generation_started_before_freeze": True}
                ),
                "authorization",
            ),
            (
                lambda value: value[
                    "transitively_executed_project_generation_helpers"
                ].append("unbound.py"),
                "self-contained",
            ),
        ):
            changed = copy.deepcopy(valid)
            mutation(changed)
            with self.assertRaisesRegex(
                generator.GenerationRefused,
                message,
            ):
                generator.validate_pre_generation_freeze_document(
                    changed,
                    freeze_relative_path=self.freeze_relative,
                    output_relative_path=self.output_relative,
                    observed_generator_sha256=self.generator_sha256,
                    observed_runtime=self.runtime,
                )

    def test_successful_git_authorization_binds_attempt_2_source_and_blobs(
        self,
    ) -> None:
        with self.mocked_authorization_environment() as environment:
            authorization = generator.authorize_generation(
                pre_generation_freeze=environment["freeze_path"],
                output_directory=environment["output_path"],
            )
            self.assertIsInstance(
                authorization,
                generator.GenerationAuthorization,
            )
            self.assertEqual(
                authorization.source_commit,
                environment["source_commit"],
            )
            self.assertEqual(
                authorization.generator_sha256,
                self.generator_sha256,
            )
            self.assertEqual(
                authorization.freeze_sha256,
                environment["freeze_sha256"],
            )
            self.assertEqual(
                authorization.freeze_commit,
                environment["freeze_commit"],
            )
            generator._revalidate_authorized_bindings(
                authorization,
                phase="immediately_before_generation",
            )

    def test_git_authorization_rejects_commit_blob_and_ancestry_failures(
        self,
    ) -> None:
        cases = (
            ("attempt_2_commit", "attempt-2 source commit object"),
            ("source_commit", "generator source commit object"),
            ("freeze_blob", "working pre-generation freeze differs"),
            ("generator_blob", "working generator differs"),
            (
                "attempt_2_to_source",
                "attempt-2 source is not an ancestor of generator source",
            ),
            (
                "attempt_2_to_freeze",
                "attempt-2 source is not an ancestor of freeze commit",
            ),
            (
                "source_to_freeze",
                "generator source commit is not an ancestor of freeze commit",
            ),
        )
        for failure, message in cases:
            with self.subTest(failure=failure):
                with self.mocked_authorization_environment() as environment:
                    environment["state"]["failure"] = failure
                    with self.assertRaisesRegex(
                        generator.GenerationRefused,
                        message,
                    ):
                        generator.authorize_generation(
                            pre_generation_freeze=environment["freeze_path"],
                            output_directory=environment["output_path"],
                        )

    def test_authorized_source_toctou_is_rechecked_before_and_after(
        self,
    ) -> None:
        for failure, message in (
            ("current_generator", "authorized bytes changed"),
            ("current_freeze", "authorized bytes changed"),
            ("generator_blob", "committed generator blob changed"),
            ("freeze_blob", "committed freeze blob changed"),
            ("attempt_2_commit", "attempt-2 source commit binding changed"),
            (
                "attempt_2_to_source",
                "attempt-2 source is not generator-source ancestor",
            ),
        ):
            with self.subTest(failure=failure):
                with self.mocked_authorization_environment() as environment:
                    authorization = generator.authorize_generation(
                        pre_generation_freeze=environment["freeze_path"],
                        output_directory=environment["output_path"],
                    )
                    environment["state"]["failure"] = failure
                    with self.assertRaisesRegex(
                        generator.GenerationRefused,
                        message,
                    ):
                        generator._revalidate_authorized_bindings(
                            authorization,
                            phase="immediately_after_generation",
                        )

    def test_every_production_row_bypass_requires_opaque_authorization(
        self,
    ) -> None:
        with self.assertRaisesRegex(TypeError, "opaque"):
            generator.GenerationAuthorization()
        forged = object()
        with self.assertRaisesRegex(
            generator.GenerationRefused,
            "internally minted authorization",
        ):
            generator._generate_authorized_family_rows(
                forged,
                base_seed=20263005,
                family="noise",
            )
        with mock.patch.object(generator.tempfile, "mkdtemp") as temporary:
            with self.assertRaisesRegex(
                generator.GenerationRefused,
                "internally minted authorization",
            ):
                generator._execute_authorized_generation(forged)
            temporary.assert_not_called()

        with self.mocked_authorization_environment() as environment:
            authorization = generator.authorize_generation(
                pre_generation_freeze=environment["freeze_path"],
                output_directory=environment["output_path"],
            )
            with self.assertRaisesRegex(
                generator.GenerationRefused,
                "outside the exact joint four-seed inventory",
            ):
                generator._generate_authorized_family_rows(
                    authorization,
                    base_seed=20263009,
                    family="noise",
                )
            with self.assertRaisesRegex(AttributeError, "immutable"):
                authorization.generator_sha256 = "0" * 64
            with self.assertRaises(TypeError):
                authorization.runtime["numpy_version"] = "changed"

        for rows, length in ((300, 17), (3, 16384), (300, 16384)):
            with self.subTest(rows=rows, length=length):
                with self.assertRaises(generator.GenerationRefused):
                    generator.generate_family_rows_for_review(
                        20263005,
                        "noise",
                        row_count=rows,
                        length=length,
                    )
        self.assertFalse(
            hasattr(generator, "_generate_family_rows_unchecked")
        )
        self.assertFalse(hasattr(generator, "generate_noise_row"))
        self.assertFalse(hasattr(generator, "generate_chirp_row"))

    def test_hidden_authorization_subclass_cannot_self_mint(self) -> None:
        hidden_types = [
            value
            for value in generator.GenerationAuthorization.__subclasses__()
            if value.__name__ == "Authorized"
        ]
        self.assertEqual(len(hidden_types), 1)
        hidden_type = hidden_types[0]
        constructor_values = {
            name: None
            for name in hidden_type.__slots__
            if name not in {"_capability", "_sealed"}
        }
        with self.assertRaises(TypeError):
            hidden_type(**constructor_values)
        with self.assertRaisesRegex(TypeError, "closure-held mint sentinel"):
            hidden_type(object(), **constructor_values)

        uninitialized = hidden_type.__new__(hidden_type)
        with self.assertRaisesRegex(
            generator.GenerationRefused,
            "internally minted authorization",
        ):
            generator._generate_authorized_family_rows(
                uninitialized,
                base_seed=20263005,
                family="noise",
            )
        with mock.patch.object(
            generator,
            "_verify_authorized_revalidation_bindings",
        ) as verifier:
            with self.assertRaisesRegex(
                generator.GenerationRefused,
                "internally minted authorization",
            ):
                generator._revalidate_authorized_bindings(
                    uninitialized,
                    phase="immediately_before_generation",
                )
            verifier.assert_not_called()
        self.assertFalse(
            any("mint_sentinel" in name for name in vars(generator))
        )
        self.assertNotIn(
            "mint_sentinel",
            vars(generator.GenerationAuthorization),
        )

    def test_revalidation_has_no_direct_phase_mark_bypass(self) -> None:
        self.assertFalse(
            hasattr(generator, "_mark_generation_revalidated")
        )
        with self.mocked_authorization_environment() as environment:
            authorization = generator.authorize_generation(
                pre_generation_freeze=environment["freeze_path"],
                output_directory=environment["output_path"],
            )
            with self.assertRaisesRegex(
                generator.GenerationRefused,
                "unknown authorization revalidation phase",
            ):
                generator._revalidate_authorized_bindings(
                    authorization,
                    phase="caller_claimed_success",
                )
            environment["state"]["failure"] = "current_generator"
            with self.assertRaisesRegex(
                generator.GenerationRefused,
                "authorized bytes changed",
            ):
                generator._revalidate_authorized_bindings(
                    authorization,
                    phase="immediately_before_generation",
                )
            environment["state"]["failure"] = None
            generator._revalidate_authorized_bindings(
                authorization,
                phase="immediately_after_generation",
            )
            with self.assertRaisesRegex(
                generator.GenerationRefused,
                "before-and-after",
            ):
                generator._build_authorized_digest_manifest(
                    authorization,
                    self.synthetic_digest_records(),
                )

    def test_manifest_cross_binds_authorized_source_freeze_e14_and_records(
        self,
    ) -> None:
        with self.mocked_authorization_environment() as environment:
            authorization = generator.authorize_generation(
                pre_generation_freeze=environment["freeze_path"],
                output_directory=environment["output_path"],
            )
            with self.assertRaisesRegex(
                generator.GenerationRefused,
                "before-and-after",
            ):
                generator._build_authorized_digest_manifest(
                    authorization,
                    self.synthetic_digest_records(),
                )
            generator._revalidate_authorized_bindings(
                authorization,
                phase="immediately_before_generation",
            )
            generator._revalidate_authorized_bindings(
                authorization,
                phase="immediately_after_generation",
            )
            with mock.patch.object(
                generator,
                "_sha256_file",
                side_effect=AssertionError(
                    "manifest must retain authorized SHA without rereading"
                ),
            ):
                manifest = generator._build_authorized_digest_manifest(
                    authorization,
                    self.synthetic_digest_records(),
                )
        self.assertEqual(
            manifest["generator_source"]["sha256"],
            self.generator_sha256,
        )
        self.assertEqual(
            manifest["generator_source"]["commit"],
            environment["source_commit"],
        )
        self.assertEqual(
            manifest["pre_generation_source_freeze"]["sha256"],
            environment["freeze_sha256"],
        )
        self.assertEqual(
            manifest["attempt_2_source"]["commit"],
            generator.ATTEMPT_2_SOURCE_COMMIT,
        )
        self.assertTrue(
            manifest["attempt_2_source"][
                "is_generator_source_commit_ancestor"
            ]
        )
        self.assertEqual(len(manifest["records"]), 12)
        self.assertEqual(len(manifest["exact_cell_inventory"]), 72)
        self.assertEqual(
            manifest["authorization_revalidation"][
                "authorized_generator_sha256_retained_without_recomputation"
            ],
            self.generator_sha256,
        )

        changed = self.synthetic_digest_records()
        changed[0]["raw_sha256"] = "0" * 64
        with self.mocked_authorization_environment() as environment:
            authorization = generator.authorize_generation(
                pre_generation_freeze=environment["freeze_path"],
                output_directory=environment["output_path"],
            )
            generator._revalidate_authorized_bindings(
                authorization,
                phase="immediately_before_generation",
            )
            generator._revalidate_authorized_bindings(
                authorization,
                phase="immediately_after_generation",
            )
            with self.assertRaisesRegex(ValueError, "digest record changed"):
                generator._build_authorized_digest_manifest(
                    authorization,
                    changed,
                )

    def test_execution_refuses_missing_freeze_before_production_generation(
        self,
    ) -> None:
        missing = (
            generator.REPO
            / "training/zplane_ab/v2_full_variation/v5_scale_orbit"
            / "definitely-missing-pre-generation-freeze.json"
        )
        output = (
            generator.REPO
            / ".artifacts/definitely-not-created-v5-novelty-output"
        )
        self.assertFalse(missing.exists())
        self.assertFalse(output.exists())
        with mock.patch.object(
            generator,
            "_execute_authorized_generation",
        ) as execute:
            with self.assertRaisesRegex(
                generator.GenerationRefused,
                "forbidden until",
            ):
                generator.execute_frozen_generation(
                    pre_generation_freeze=missing,
                    output_directory=output,
                )
            execute.assert_not_called()
        self.assertFalse(output.exists())

    def test_requirements_report_binds_source_runtime_command_and_inventory(
        self,
    ) -> None:
        report = generator.required_pre_generation_freeze_inputs(
            freeze_path=generator.REPO / self.freeze_relative,
            output_directory=generator.REPO / self.output_relative,
        )
        self.assertTrue(
            report["must_be_append_only_committed_before_generation"]
        )
        self.assertTrue(
            report[
                "must_prove_generator_source_commit_is_freeze_commit_ancestor"
            ]
        )
        self.assertEqual(
            report["attempt_2_source_commit"],
            generator.ATTEMPT_2_SOURCE_COMMIT,
        )
        self.assertTrue(
            report[
                "must_prove_attempt_2_source_is_generator_source_ancestor"
            ]
        )
        self.assertTrue(
            report["must_prove_attempt_2_source_is_freeze_commit_ancestor"]
        )
        self.assertEqual(
            report["transitively_executed_project_generation_helpers"],
            [],
        )
        self.assertEqual(
            report["inventory"]["seeds"],
            list(generator.ADAPTIVE_SEEDS),
        )
        self.assertEqual(
            report["runtime"]["numpy_default_rng_bit_generator"],
            "PCG64",
        )
        self.assertEqual(
            report["command_configuration"]["device"],
            "cpu",
        )
        self.assertEqual(
            report["command_configuration"]["embedding_batch_size"],
            256,
        )


if __name__ == "__main__":
    unittest.main()
