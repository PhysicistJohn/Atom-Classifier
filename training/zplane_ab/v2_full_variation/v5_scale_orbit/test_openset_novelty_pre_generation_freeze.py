from __future__ import annotations

import ast
import copy
from contextlib import ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import unittest
from unittest import mock

import validate_openset_novelty_pre_generation_freeze as validator


class NoCallDefaultRng:
    __name__ = "default_rng"

    def __call__(self, *args, **kwargs):
        raise AssertionError("pre-generation validation must not instantiate RNG")


class OpensetNoveltyPreGenerationFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.freeze_bytes = validator.FREEZE_PATH.read_bytes()
        cls.document = json.loads(cls.freeze_bytes.decode("utf-8"))
        cls.bound_file_bytes = {
            validator.FREEZE_RELATIVE_PATH: cls.freeze_bytes,
            validator.AMENDMENT_RELATIVE_PATH:
                validator.AMENDMENT_PATH.read_bytes(),
            validator.AMENDMENT_VALIDATOR_RELATIVE_PATH:
                validator.AMENDMENT_VALIDATOR_PATH.read_bytes(),
            validator.GENERATOR_RELATIVE_PATH:
                validator.GENERATOR_PATH.read_bytes(),
            validator.VALIDATOR_RELATIVE_PATH:
                validator.VALIDATOR_PATH.read_bytes(),
        }
        cls.freeze_commit = "f" * 40
        cls.head_commit = "d" * 40
        cls.freeze_timestamp = validator.GENERATOR_COMMIT_TIMESTAMP + 100

    @staticmethod
    def _changed(value):
        if isinstance(value, bool):
            return not value
        if isinstance(value, int):
            return value + 1
        if isinstance(value, str):
            return value + "-mutated"
        if isinstance(value, list):
            return [*value, "__mutated__"]
        if isinstance(value, dict):
            return {"__mutated__": True}
        raise AssertionError(f"unsupported mutation value {value!r}")

    @classmethod
    def _field_paths(cls, value, prefix=()):
        if isinstance(value, dict):
            for key, child in value.items():
                path = (*prefix, key)
                yield path
                yield from cls._field_paths(child, path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                path = (*prefix, index)
                yield path
                yield from cls._field_paths(child, path)

    @classmethod
    def _mutate_at_path(cls, original, path):
        changed = copy.deepcopy(original)
        target = changed
        for component in path[:-1]:
            target = target[component]
        final = path[-1]
        target[final] = cls._changed(target[final])
        return changed

    @staticmethod
    def _completed(args, *, stdout="", returncode=0):
        return subprocess.CompletedProcess(
            ["git", *args],
            returncode,
            stdout=stdout,
            stderr=b"" if isinstance(stdout, bytes) else "",
        )

    def fake_git(self, state):
        freeze_commit = self.freeze_commit
        head_commit = self.head_commit

        def implementation(*args: str, **kwargs):
            if args == (
                "log",
                "--format=%H",
                "--",
                validator.FREEZE_RELATIVE_PATH,
            ):
                if state["failure"] == "freeze_history":
                    return self._completed(args, stdout="")
                if state["failure"] == "freeze_history_multiple":
                    return self._completed(
                        args,
                        stdout=f"{freeze_commit}\n{'e' * 40}\n",
                    )
                return self._completed(args, stdout=f"{freeze_commit}\n")
            if args == (
                "cat-file",
                "-e",
                (
                    f"{validator.GENERATOR_COMMIT}:"
                    f"{validator.FREEZE_RELATIVE_PATH}"
                ),
            ):
                return self._completed(
                    args,
                    returncode=(
                        0 if state["failure"] == "preexisting_freeze" else 128
                    ),
                )
            if args == ("rev-parse", "HEAD"):
                value = "invalid" if state["failure"] == "invalid_head" else head_commit
                return self._completed(args, stdout=f"{value}\n")
            if args[:3] == (
                "show",
                "-s",
                "--format=%H%x00%P%x00%s%x00%ct",
            ):
                commit = args[3]
                if commit == validator.ATTEMPT_2_SOURCE_COMMIT:
                    values = [
                        commit,
                        validator.ATTEMPT_2_PREREGISTRATION_COMMIT,
                        validator.ATTEMPT_2_SOURCE_SUBJECT,
                        str(validator.ATTEMPT_2_SOURCE_TIMESTAMP),
                    ]
                    if state["failure"] == "attempt_2_metadata":
                        values[2] = "changed subject"
                elif commit == validator.GENERATOR_COMMIT:
                    values = [
                        commit,
                        validator.GENERATOR_PARENT_COMMIT,
                        validator.GENERATOR_COMMIT_SUBJECT,
                        str(validator.GENERATOR_COMMIT_TIMESTAMP),
                    ]
                    if state["failure"] == "generator_metadata":
                        values[1] = "0" * 40
                elif commit == freeze_commit:
                    values = [
                        freeze_commit,
                        validator.GENERATOR_COMMIT,
                        "Freeze v5 adaptive novelty generation inputs",
                        str(self.freeze_timestamp),
                    ]
                    if state["failure"] == "freeze_chronology":
                        values[3] = str(validator.GENERATOR_COMMIT_TIMESTAMP)
                else:
                    raise AssertionError(f"unexpected metadata commit {commit}")
                return self._completed(
                    args,
                    stdout="\x00".join(values) + "\n",
                )
            if args[0] == "show" and len(args) == 2:
                commit, relative_path = args[1].split(":", 1)
                expected_commit = (
                    freeze_commit
                    if relative_path in {
                        validator.FREEZE_RELATIVE_PATH,
                        validator.VALIDATOR_RELATIVE_PATH,
                    }
                    else validator.GENERATOR_COMMIT
                )
                if commit != expected_commit:
                    raise AssertionError(
                        f"unexpected blob commit {commit}:{relative_path}"
                    )
                content = self.bound_file_bytes[relative_path]
                if state["failure"] == f"blob:{relative_path}":
                    content = b"changed blob"
                return self._completed(args, stdout=content)
            if args[:2] == ("merge-base", "--is-ancestor"):
                pair = args[2:]
                failed_pairs = {
                    "ancestry_attempt_2_generator": (
                        validator.ATTEMPT_2_SOURCE_COMMIT,
                        validator.GENERATOR_COMMIT,
                    ),
                    "ancestry_attempt_2_freeze": (
                        validator.ATTEMPT_2_SOURCE_COMMIT,
                        freeze_commit,
                    ),
                    "ancestry_generator_freeze": (
                        validator.GENERATOR_COMMIT,
                        freeze_commit,
                    ),
                    "ancestry_freeze_head": (
                        freeze_commit,
                        head_commit,
                    ),
                }
                return self._completed(
                    args,
                    returncode=(
                        1
                        if failed_pairs.get(state["failure"]) == pair
                        else 0
                    ),
                )
            if args == (
                "ls-tree",
                "-r",
                "--name-only",
                freeze_commit,
                "--",
                validator.OUTPUT_RELATIVE_PATH,
            ):
                output = (
                    f"{validator.OUTPUT_RELATIVE_PATH}/unexpected.raw\n"
                    if state["failure"] == "tracked_output"
                    else ""
                )
                return self._completed(args, stdout=output)
            if args == (
                "log",
                "--all",
                "--format=%H",
                "--",
                validator.OUTPUT_RELATIVE_PATH,
            ):
                output = (
                    f"{'b' * 40}\n"
                    if state["failure"] == "output_history"
                    else ""
                )
                return self._completed(args, stdout=output)
            raise AssertionError(f"unexpected Git call {args!r} {kwargs!r}")

        return implementation

    def test_exact_freeze_bytes_document_and_generator_compatibility(self) -> None:
        self.assertEqual(
            hashlib.sha256(self.freeze_bytes).hexdigest(),
            validator.FREEZE_RAW_SHA256,
        )
        validator.validate_freeze_document(
            self.document,
            observed_runtime=validator.EXPECTED_RUNTIME,
        )

        spec = importlib.util.spec_from_file_location(
            "compatibility_generator",
            validator.GENERATOR_PATH,
        )
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        self.assertEqual(
            generator.validate_pre_generation_freeze_document(
                self.document,
                freeze_relative_path=validator.FREEZE_RELATIVE_PATH,
                output_relative_path=validator.OUTPUT_RELATIVE_PATH,
                observed_generator_sha256=validator.GENERATOR_SHA256,
                observed_runtime=validator.EXPECTED_RUNTIME,
            ),
            validator.GENERATOR_COMMIT,
        )

    def test_every_freeze_field_and_container_is_mutation_locked(self) -> None:
        paths = list(self._field_paths(self.document))
        self.assertGreater(len(paths), 130)
        for path in paths:
            with self.subTest(path=path):
                changed = self._mutate_at_path(self.document, path)
                with self.assertRaises(validator.FreezeValidationError):
                    validator.validate_freeze_document(
                        changed,
                        observed_runtime=validator.EXPECTED_RUNTIME,
                    )

        for changed in (
            {**self.document, "unexpected": True},
            {
                key: value
                for key, value in self.document.items()
                if key != "authorization"
            },
        ):
            with self.assertRaises(validator.FreezeValidationError):
                validator.validate_freeze_document(
                    changed,
                    observed_runtime=validator.EXPECTED_RUNTIME,
                )

    def test_runtime_binding_is_exact_without_instantiating_rng(self) -> None:
        with mock.patch.object(
            validator.np.random,
            "default_rng",
            new=NoCallDefaultRng(),
        ):
            observed = validator._runtime_from_executable_sha256(
                validator.PYTHON_EXECUTABLE_SHA256
            )
        self.assertEqual(observed, validator.EXPECTED_RUNTIME)

        changed = copy.deepcopy(validator.EXPECTED_RUNTIME)
        changed["numpy_version"] = "changed"
        with self.assertRaisesRegex(
            validator.FreezeValidationError,
            "observed runtime",
        ):
            validator.validate_freeze_document(
                self.document,
                observed_runtime=changed,
            )

    def test_full_validation_accounts_every_read_and_creates_nothing(
        self,
    ) -> None:
        state = {"failure": None}
        reads = []
        original_read = validator._read_bytes

        def recorded_read(path):
            reads.append(path)
            return original_read(path)

        forbidden_write = AssertionError(
            "pre-generation validator attempted a filesystem mutation"
        )
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    validator,
                    "_git",
                    side_effect=self.fake_git(state),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    validator,
                    "_path_exists_including_broken_symlink",
                    return_value=False,
                )
            )
            stack.enter_context(
                mock.patch.object(
                    validator,
                    "_read_bytes",
                    side_effect=recorded_read,
                )
            )
            stack.enter_context(
                mock.patch.object(
                    validator.np.random,
                    "default_rng",
                    new=NoCallDefaultRng(),
                )
            )
            for method in (
                "write_bytes",
                "write_text",
                "mkdir",
                "touch",
                "unlink",
                "rename",
                "replace",
            ):
                stack.enter_context(
                    mock.patch.object(
                        Path,
                        method,
                        side_effect=forbidden_write,
                    )
                )
            report = validator.validate_repository_state()

        self.assertEqual(report["status"], "pass")
        self.assertFalse(report["generation_performed"])
        self.assertFalse(report["rng_instantiated"])
        self.assertEqual(report["files_created_or_modified"], 0)
        self.assertEqual(
            report["exact_generation_command"],
            list(validator.EXPECTED_GENERATION_COMMAND),
        )
        self.assertEqual(
            report["exact_generation_command"][2::2],
            ["--pre-generation-freeze", "--output-dir"],
        )
        self.assertEqual(
            reads,
            list(validator.EXPECTED_FILE_READ_SEQUENCE),
        )
        self.assertEqual(len(report["git_proof"]["git_blobs"]), 5)
        self.assertTrue(
            report["git_proof"]["ancestry"]["freeze_to_validated_head"]
        )

    def test_git_commit_blob_chronology_and_ancestry_fail_closed(self) -> None:
        failures = [
            "freeze_history",
            "freeze_history_multiple",
            "preexisting_freeze",
            "invalid_head",
            "attempt_2_metadata",
            "generator_metadata",
            "freeze_chronology",
            *[
                f"blob:{relative_path}"
                for relative_path in self.bound_file_bytes
            ],
            "ancestry_attempt_2_generator",
            "ancestry_attempt_2_freeze",
            "ancestry_generator_freeze",
            "ancestry_freeze_head",
            "tracked_output",
            "output_history",
        ]
        for failure in failures:
            with self.subTest(failure=failure):
                state = {"failure": failure}
                with mock.patch.object(
                    validator,
                    "_git",
                    side_effect=self.fake_git(state),
                ), mock.patch.object(
                    validator,
                    "_path_exists_including_broken_symlink",
                    return_value=False,
                ):
                    with self.assertRaises(validator.FreezeValidationError):
                        validator.validate_repository_state()

    def test_working_blob_change_fails_before_git(self) -> None:
        original_read = validator._read_bytes

        def changed_generator(path):
            if path == validator.GENERATOR_PATH:
                return b"changed generator"
            return original_read(path)

        with mock.patch.object(
            validator,
            "_path_exists_including_broken_symlink",
            return_value=False,
        ), mock.patch.object(
            validator,
            "_read_bytes",
            side_effect=changed_generator,
        ), mock.patch.object(
            validator,
            "_git",
        ) as git:
            with self.assertRaisesRegex(
                validator.FreezeValidationError,
                "working file binding changed",
            ):
                validator.validate_repository_state()
            git.assert_not_called()

    def test_output_absence_is_checked_before_and_after_proof(self) -> None:
        with mock.patch.object(
            validator,
            "_path_exists_including_broken_symlink",
            return_value=True,
        ), mock.patch.object(validator, "_read_bytes") as reader:
            with self.assertRaisesRegex(
                validator.FreezeValidationError,
                "output exists",
            ):
                validator.validate_repository_state()
            reader.assert_not_called()

        state = {"failure": None}
        with mock.patch.object(
            validator,
            "_path_exists_including_broken_symlink",
            side_effect=(False, True),
        ), mock.patch.object(
            validator,
            "_git",
            side_effect=self.fake_git(state),
        ):
            with self.assertRaisesRegex(
                validator.FreezeValidationError,
                "output appeared",
            ):
                validator.validate_repository_state()

    def test_validator_source_is_standalone_read_only_and_generation_free(
        self,
    ) -> None:
        source = Path(validator.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertNotIn("generate_openset_novelty_adaptive", imports)
        self.assertNotIn("random", imports)
        self.assertNotIn("tempfile", imports)
        self.assertNotIn("shutil", imports)
        self.assertNotIn("torch", imports)
        self.assertNotIn("np.random.default_rng(", source)
        for forbidden in (
            ".write_bytes(",
            ".write_text(",
            ".mkdir(",
            ".touch(",
            ".unlink(",
            ".rename(",
            "open(\"w",
            "open('w",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
