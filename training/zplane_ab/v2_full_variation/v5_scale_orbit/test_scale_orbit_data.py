from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import scale_orbit_data as data


TEST_PROFILES = (
    "bluetooth-le-advertising",
    "wifi-hr-dsss-11m",
)
TEST_PROFILE_CLASS_MAP = {
    "bluetooth-le-advertising": "bluetooth",
    "wifi-hr-dsss-11m": "dsss",
}
TEST_RUNTIME_LENGTHS = (2, 4)
PUBLIC_CLASS_INDEX = {
    name: index for index, name in enumerate(data.scale_data.PUBLIC_CLASSES)
}


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _fake_population(
    seed: int,
    *,
    marker: int,
) -> data.scale_data.ScaleEvalCorpus:
    row_count = (
        len(TEST_PROFILES)
        * data.SCALE_ORBIT_REALIZATIONS_PER_PROFILE
        * len(data.SCALE_ORBIT_EXPECTED_FACTORS)
    )
    raw = np.empty((row_count, TEST_RUNTIME_LENGTHS[-1], 2), np.float32)
    rows: list[data.scale_data.ScaleEvalRow] = []
    scale_ratios = ((1, 1), (5, 4), (3, 2), (2, 1))
    index = 0
    for profile_offset, profile in enumerate(TEST_PROFILES):
        replay = (
            "one-shot" if profile.startswith("bluetooth-") else "cyclic"
        )
        for realization_index in range(
            data.SCALE_ORBIT_REALIZATIONS_PER_PROFILE
        ):
            pair_id = f"pair-{seed}-{profile}-{realization_index}"
            phase = (
                0
                if replay == "one-shot"
                else seed * 100 + profile_offset * 64 + realization_index
            )
            channel_seed = (
                seed * 1_000 + profile_offset * 100 + realization_index
            )
            receiver_seed = (
                seed * 10_000 + profile_offset * 100 + realization_index
            )
            for scale_index, scale_factor in enumerate(
                data.SCALE_ORBIT_EXPECTED_FACTORS
            ):
                numerator, denominator = scale_ratios[scale_index]
                base = marker * 1_000_000 + index * 32
                raw[index, :, 0] = base + np.arange(
                    TEST_RUNTIME_LENGTHS[-1],
                    dtype=np.float32,
                )
                raw[index, :, 1] = -base - np.arange(
                    TEST_RUNTIME_LENGTHS[-1],
                    dtype=np.float32,
                )
                rows.append(
                    data.scale_data.ScaleEvalRow(
                        index=index,
                        class_name=TEST_PROFILE_CLASS_MAP[profile],
                        profile=profile,
                        replay=replay,
                        pair_id=pair_id,
                        realization_index=realization_index,
                        phase_native_sample=phase,
                        receiver_preset="unit-test",
                        channel_seed=channel_seed,
                        receiver_seed=receiver_seed,
                        scale_factor=scale_factor,
                        scale_key=str(scale_factor),
                        scale_numerator=numerator,
                        scale_denominator=denominator,
                        native_sample_rate_hz=1_000_000,
                        sample_rate_hz=int(1_000_000 * scale_factor),
                        native_carrier_offset_hz=0,
                        signal_bandwidth_hz=100_000,
                        capture_bandwidth_hz=200_000,
                        valid_sample_count=TEST_RUNTIME_LENGTHS[-1],
                        available_lengths=TEST_RUNTIME_LENGTHS,
                        content_sha256=_sha256(
                            f"content:{seed}:{profile}:{realization_index}:"
                            f"{scale_factor}"
                        ),
                        stored_sha256=_sha256(
                            f"stored:{seed}:{profile}:{realization_index}:"
                            f"{scale_factor}"
                        ),
                    )
                )
                index += 1
    directory = (
        data.REPO / ".artifacts" / f"unit-scale-orbit-seed{seed}-{marker}"
    )
    return data.scale_data.ScaleEvalCorpus(
        directory=directory,
        manifest_path=directory / "scale_eval.json",
        raw_path=directory / "scale_eval.f32",
        manifest={"evalSeed": seed},
        raw=raw,  # type: ignore[arg-type]
        rows=rows,
        audit={
            "manifest_sha256": _sha256(f"manifest:{seed}:{marker}"),
            "raw_sha256": _sha256(f"raw:{seed}:{marker}"),
            "realizations_per_profile":
                data.SCALE_ORBIT_REALIZATIONS_PER_PROFILE,
            "unbound_smoke": False,
            "reference_bound": True,
            "profiles": list(TEST_PROFILES),
            "profile_public_class_map":
                dict(sorted(TEST_PROFILE_CLASS_MAP.items())),
            "classes_present": sorted(set(TEST_PROFILE_CLASS_MAP.values())),
            "scale_factors": list(data.SCALE_ORBIT_EXPECTED_FACTORS),
            "prefix_lengths": list(TEST_RUNTIME_LENGTHS),
        },
    )


def _replace_pair_rows(
    corpus: data.scale_data.ScaleEvalCorpus,
    *,
    profile: str,
    target_realization_index: int,
    **changes: object,
) -> None:
    for index, row in enumerate(corpus.rows):
        if (
            row.profile == profile
            and row.realization_index == target_realization_index
        ):
            corpus.rows[index] = replace(row, **changes)


def _empty_identity_inventory() -> dict[str, dict[str, set[object]]]:
    keys = (
        "pair_id",
        "content_sha256",
        "runtime_prefix_sha256",
        "cyclic_profile_and_phase_native_sample",
        "receiver_realization_channel_seed",
        "receiver_realization_seed_when_non_null",
    )
    return {
        role: {key: set() for key in keys}
        for role in data.current_data.ROLES
    }


class ScaleOrbitDataTests(unittest.TestCase):
    def setUp(self) -> None:
        patchers = (
            mock.patch.object(
                data.current_data,
                "CURRENT_PROFILES",
                TEST_PROFILES,
            ),
            mock.patch.object(
                data.current_data,
                "CURRENT_PROFILE_PUBLIC_CLASS_MAP",
                TEST_PROFILE_CLASS_MAP,
            ),
            mock.patch.object(
                data.current_data,
                "RUNTIME_INPUT_LENGTHS",
                TEST_RUNTIME_LENGTHS,
            ),
        )
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def _load(
        self,
        training: data.scale_data.ScaleEvalCorpus | None = None,
        selection: data.scale_data.ScaleEvalCorpus | None = None,
    ) -> data.ScaleOrbitCorpus:
        training = training or _fake_population(
            data.SCALE_ORBIT_TRAINING_SEED,
            marker=1,
        )
        selection = selection or _fake_population(
            data.SCALE_ORBIT_SELECTION_SEED,
            marker=2,
        )
        with (
            mock.patch.object(
                data.scale_data,
                "load_scale_eval_corpus",
                side_effect=[training, selection],
            ) as loader,
            mock.patch.object(
                data,
                "_validate_identity_firewalled_selection",
            ) as firewall,
        ):
            result = data.load_scale_orbit_training_corpus(
                training.directory,
                selection_directory=selection.directory,
                class_index=PUBLIC_CLASS_INDEX,
            )
        firewall.assert_called_once_with(training, selection)
        self.assertEqual(
            [call.args[0] for call in loader.call_args_list],
            [training.directory.resolve(), selection.directory.resolve()],
        )
        self.assertEqual(
            [
                call.kwargs["expected_generator_lineage"]
                for call in loader.call_args_list
            ],
            [
                data.SCALE_ORBIT_TRAINING_GENERATOR_LINEAGE,
                data.SCALE_ORBIT_FIREWALL_GENERATOR_LINEAGE,
            ],
        )
        return result

    def test_replacement_selection_is_bound_to_firewall_generator_and_training(
        self,
    ) -> None:
        training = _fake_population(
            data.SCALE_ORBIT_TRAINING_SEED,
            marker=1,
        )
        selection = _fake_population(
            data.SCALE_ORBIT_SELECTION_SEED,
            marker=2,
        )
        selection.directory = (
            data.REPO
            / data.SCALE_ORBIT_IDENTITY_FIREWALLED_SELECTION_DIRECTORY
        )
        selection.manifest_path = selection.directory / "scale_eval.json"
        selection.raw_path = selection.directory / "scale_eval.f32"
        training.audit["manifest_sha256"] = (
            data.SCALE_ORBIT_TRAINING_MANIFEST_SHA256
        )
        training.audit["raw_sha256"] = data.SCALE_ORBIT_TRAINING_RAW_SHA256
        selection.audit["manifest_sha256"] = (
            data.SCALE_ORBIT_IDENTITY_FIREWALLED_SELECTION_MANIFEST_SHA256
        )
        selection.audit["raw_sha256"] = (
            data.SCALE_ORBIT_IDENTITY_FIREWALLED_SELECTION_RAW_SHA256
        )
        selection.manifest.update(
            {
                "generatorLineage": {
                    "sourceSha256":
                        data.SCALE_ORBIT_FIREWALL_GENERATOR_SOURCE_SHA256,
                    "bundleSha256":
                        data.SCALE_ORBIT_FIREWALL_GENERATOR_BUNDLE_SHA256,
                },
                "heldOutContract": {
                    "referenceBound": True,
                    "scaleIdentityExclusionCount": 1,
                },
                "referenceCorpus": {
                    "directory": str(
                        (
                            data.REPO / data.SCALE_ORBIT_REFERENCE_DIRECTORY
                        ).resolve()
                    ),
                    "manifest": "corpus.json",
                    "manifestSha256":
                        data.SCALE_ORBIT_REFERENCE_MANIFEST_SHA256,
                    "rawSha256": data.SCALE_ORBIT_REFERENCE_RAW_SHA256,
                    "count": data.SCALE_ORBIT_REFERENCE_COUNT,
                    "corpusSeed": data.SCALE_ORBIT_REFERENCE_SEED,
                },
                "scaleIdentityExclusionCorpora": [
                    {
                        "directory": str(training.directory.resolve()),
                        "manifest": "scale_eval.json",
                        "manifestSha256":
                            data.SCALE_ORBIT_TRAINING_MANIFEST_SHA256,
                        "rawSha256":
                            data.SCALE_ORBIT_TRAINING_RAW_SHA256,
                        "count": (
                            len(TEST_PROFILES)
                            * data.SCALE_ORBIT_REALIZATIONS_PER_PROFILE
                            * len(data.SCALE_ORBIT_EXPECTED_FACTORS)
                        ),
                        "evalSeed": data.SCALE_ORBIT_TRAINING_SEED,
                    }
                ],
            }
        )
        data._validate_identity_firewalled_selection(training, selection)

        selection.manifest["referenceCorpus"]["rawSha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "reference corpus"):
            data._validate_identity_firewalled_selection(
                training,
                selection,
            )
        selection.manifest["referenceCorpus"]["rawSha256"] = (
            data.SCALE_ORBIT_REFERENCE_RAW_SHA256
        )

        selection.manifest["generatorLineage"]["sourceSha256"] = "0" * 64
        with self.assertRaisesRegex(
            ValueError,
            "generator identity firewall",
        ):
            data._validate_identity_firewalled_selection(
                training,
                selection,
            )

    def test_replacement_selection_refuses_unbound_or_arbitrary_hashes(
        self,
    ) -> None:
        training = _fake_population(
            data.SCALE_ORBIT_TRAINING_SEED,
            marker=1,
        )
        selection = _fake_population(
            data.SCALE_ORBIT_SELECTION_SEED,
            marker=2,
        )
        training.audit["manifest_sha256"] = (
            data.SCALE_ORBIT_TRAINING_MANIFEST_SHA256
        )
        training.audit["raw_sha256"] = data.SCALE_ORBIT_TRAINING_RAW_SHA256
        selection.directory = (
            data.REPO
            / data.SCALE_ORBIT_IDENTITY_FIREWALLED_SELECTION_DIRECTORY
        )
        selection.manifest_path = selection.directory / "scale_eval.json"
        selection.raw_path = selection.directory / "scale_eval.f32"
        selection.audit["manifest_sha256"] = "3" * 64
        selection.audit["raw_sha256"] = "4" * 64
        selection.manifest.update(
            {
                "generatorLineage": {
                    "sourceSha256":
                        data.SCALE_ORBIT_FIREWALL_GENERATOR_SOURCE_SHA256,
                    "bundleSha256":
                        data.SCALE_ORBIT_FIREWALL_GENERATOR_BUNDLE_SHA256,
                },
                "heldOutContract": {
                    "referenceBound": True,
                    "scaleIdentityExclusionCount": 1,
                },
                "referenceCorpus": {
                    "directory": str(
                        (
                            data.REPO / data.SCALE_ORBIT_REFERENCE_DIRECTORY
                        ).resolve()
                    ),
                    "manifest": "corpus.json",
                    "manifestSha256":
                        data.SCALE_ORBIT_REFERENCE_MANIFEST_SHA256,
                    "rawSha256": data.SCALE_ORBIT_REFERENCE_RAW_SHA256,
                    "count": data.SCALE_ORBIT_REFERENCE_COUNT,
                    "corpusSeed": data.SCALE_ORBIT_REFERENCE_SEED,
                },
                "scaleIdentityExclusionCorpora": [
                    {
                        "directory": str(training.directory.resolve()),
                        "manifest": "scale_eval.json",
                        "manifestSha256":
                            data.SCALE_ORBIT_TRAINING_MANIFEST_SHA256,
                        "rawSha256":
                            data.SCALE_ORBIT_TRAINING_RAW_SHA256,
                        "count": (
                            len(TEST_PROFILES)
                            * data.SCALE_ORBIT_REALIZATIONS_PER_PROFILE
                            * len(data.SCALE_ORBIT_EXPECTED_FACTORS)
                        ),
                        "evalSeed": data.SCALE_ORBIT_TRAINING_SEED,
                    }
                ],
            }
        )

        with (
            mock.patch.object(
                data,
                "SCALE_ORBIT_IDENTITY_FIREWALLED_SELECTION_MANIFEST_SHA256",
                None,
            ),
            mock.patch.object(
                data,
                "SCALE_ORBIT_IDENTITY_FIREWALLED_SELECTION_RAW_SHA256",
                None,
            ),
            mock.patch.object(
                data.scale_data,
                "load_scale_eval_corpus",
                side_effect=[training, selection],
            ),
            self.assertRaisesRegex(ValueError, "pins are not bound"),
        ):
            data.load_scale_orbit_training_corpus(
                training.directory,
                selection_directory=selection.directory,
                class_index=PUBLIC_CLASS_INDEX,
            )

        with (
            mock.patch.object(
                data.scale_data,
                "load_scale_eval_corpus",
                side_effect=[training, selection],
            ),
            mock.patch.object(
                data,
                "SCALE_ORBIT_IDENTITY_FIREWALLED_SELECTION_MANIFEST_SHA256",
                "a" * 64,
            ),
            mock.patch.object(
                data,
                "SCALE_ORBIT_IDENTITY_FIREWALLED_SELECTION_RAW_SHA256",
                "b" * 64,
            ),
            self.assertRaisesRegex(ValueError, "recovery boundary"),
        ):
            data.load_scale_orbit_training_corpus(
                training.directory,
                selection_directory=selection.directory,
                class_index=PUBLIC_CLASS_INDEX,
            )

    def test_append_only_acceptance_hash_and_exact_selection_pins(self) -> None:
        self.assertEqual(
            data.current_data.sha256_file(
                data.SCALE_ORBIT_IDENTITY_ACCEPTANCE_PATH
            ),
            data.SCALE_ORBIT_IDENTITY_ACCEPTANCE_SHA256,
        )
        self.assertEqual(
            data.SCALE_ORBIT_IDENTITY_FIREWALLED_SELECTION_MANIFEST_SHA256,
            "7a345bc109e46661427c2f81b46125fb48a00048e9b2b7da77505dc49135d998",
        )
        self.assertEqual(
            data.SCALE_ORBIT_IDENTITY_FIREWALLED_SELECTION_RAW_SHA256,
            "53d4ab20faf031ed2aacd3017d55ada1622d7c51f85b200aa490f3d3854dc928",
        )

    def test_replacement_selection_refuses_quarantined_hash_pins(self) -> None:
        training = _fake_population(
            data.SCALE_ORBIT_TRAINING_SEED,
            marker=1,
        )
        selection = _fake_population(
            data.SCALE_ORBIT_SELECTION_SEED,
            marker=2,
        )
        with (
            mock.patch.object(
                data,
                "SCALE_ORBIT_IDENTITY_FIREWALLED_SELECTION_MANIFEST_SHA256",
                data.SCALE_ORBIT_REJECTED_SELECTION_MANIFEST_SHA256,
            ),
            mock.patch.object(
                data,
                "SCALE_ORBIT_IDENTITY_FIREWALLED_SELECTION_RAW_SHA256",
                "b" * 64,
            ),
            self.assertRaisesRegex(ValueError, "quarantined bytes"),
        ):
            data._validate_identity_firewalled_selection(training, selection)

    def test_preregistered_split_boundaries_and_counts(self) -> None:
        self.assertEqual(data.role_for_training_realization(0), "train")
        self.assertEqual(data.role_for_training_realization(39), "train")
        self.assertEqual(
            data.role_for_training_realization(40),
            "enrollment",
        )
        self.assertEqual(
            data.role_for_training_realization(63),
            "enrollment",
        )
        observed = {
            role: 0 for role in data.SCALE_ORBIT_TRAINING_SPLIT_COUNTS
        }
        for index in range(data.SCALE_ORBIT_REALIZATIONS_PER_PROFILE):
            observed[data.role_for_training_realization(index)] += 1
        self.assertEqual(
            observed,
            data.SCALE_ORBIT_TRAINING_SPLIT_COUNTS,
        )

    def test_role_resolvers_reject_invalid_indices(self) -> None:
        for value in (-1, 64, True, 1.5, "4"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    data.role_for_training_realization(  # type: ignore[arg-type]
                        value
                    )
                with self.assertRaises(ValueError):
                    data._selection_role(value)  # type: ignore[arg-type]

    def test_composite_loader_emits_exact_amended_audit(self) -> None:
        corpus = self._load()
        self.assertEqual(
            set(corpus.audit),
            set(data.SCALE_ORBIT_REQUIRED_AUDIT_KEYS),
        )
        self.assertEqual(corpus.audit["schema"], data.SCALE_ORBIT_AUDIT_SCHEMA)
        self.assertEqual(
            corpus.audit["roles"],
            {
                str(data.SCALE_ORBIT_TRAINING_SEED):
                    ["train", "enrollment"],
                str(data.SCALE_ORBIT_SELECTION_SEED): ["selection"],
            },
        )
        self.assertEqual(
            corpus.audit["rows_by_role"],
            {"train": 320, "enrollment": 192, "selection": 512},
        )
        self.assertEqual(
            corpus.audit["pair_identities_by_role"],
            {"train": 80, "enrollment": 48, "selection": 128},
        )
        self.assertEqual(
            corpus.audit["profile_role_counts"],
            {
                "train": {
                    profile: 160 for profile in TEST_PROFILES
                },
                "enrollment": {
                    profile: 96 for profile in TEST_PROFILES
                },
                "selection": {
                    profile: 256 for profile in TEST_PROFILES
                },
            },
        )
        self.assertEqual(
            set(corpus.audit["corpora"]),
            {
                str(data.SCALE_ORBIT_TRAINING_SEED),
                str(data.SCALE_ORBIT_SELECTION_SEED),
            },
        )
        for binding in corpus.audit["corpora"].values():
            self.assertEqual(
                set(binding),
                set(data.SCALE_ORBIT_REQUIRED_CORPUS_BINDING_KEYS),
            )
        self.assertEqual(
            corpus.audit["fitting_firewall"],
            data.SCALE_ORBIT_FITTING_FIREWALL,
        )
        self.assertEqual(
            corpus.audit["public_class_order"],
            list(data.scale_data.PUBLIC_CLASSES),
        )
        self.assertEqual(
            corpus.audit["amendment_sha256"],
            data.SCALE_ORBIT_AMENDMENT_SHA256,
        )

    def test_seed_roles_and_row_identities_are_exact(self) -> None:
        corpus = self._load()
        expected_seeds = {
            "train": {data.SCALE_ORBIT_TRAINING_SEED},
            "enrollment": {data.SCALE_ORBIT_TRAINING_SEED},
            "selection": {data.SCALE_ORBIT_SELECTION_SEED},
        }
        for role, rows in corpus.rows_by_role.items():
            self.assertEqual({row.population_seed for row in rows}, expected_seeds[role])
            self.assertTrue(all(row.role == role for row in rows))
            self.assertTrue(
                all(
                    row.identity
                    == (
                        f"scale-orbit:{row.population_seed}:"
                        f"{row.pair_id}"
                    )
                    for row in rows
                )
            )
            identities_by_pair: dict[str, set[str]] = {}
            for row in rows:
                identities_by_pair.setdefault(row.pair_id, set()).add(
                    row.identity
                )
            self.assertTrue(
                all(len(values) == 1 for values in identities_by_pair.values())
            )

    def test_prefix_dispatches_source_index_by_population_seed(self) -> None:
        training = _fake_population(
            data.SCALE_ORBIT_TRAINING_SEED,
            marker=1,
        )
        selection = _fake_population(
            data.SCALE_ORBIT_SELECTION_SEED,
            marker=2,
        )
        corpus = self._load(training, selection)
        training_row = next(
            row
            for row in corpus.rows_by_role["train"]
            if row.source_index == 0
        )
        selection_row = next(
            row
            for row in corpus.rows_by_role["selection"]
            if row.source_index == 0
        )
        self.assertEqual(training_row.source_index, selection_row.source_index)
        np.testing.assert_array_equal(
            corpus.prefix(training_row, 2),
            (
                training.raw[0, :2, 0].astype(np.float64)
                + 1j * training.raw[0, :2, 1].astype(np.float64)
            ),
        )
        np.testing.assert_array_equal(
            corpus.prefix(selection_row, 2),
            (
                selection.raw[0, :2, 0].astype(np.float64)
                + 1j * selection.raw[0, :2, 1].astype(np.float64)
            ),
        )
        self.assertFalse(
            np.array_equal(
                corpus.prefix(training_row, 2),
                corpus.prefix(selection_row, 2),
            )
        )
        with self.assertRaisesRegex(ValueError, "live runtime bucket"):
            corpus.prefix(training_row, 3)
        with self.assertRaisesRegex(ValueError, "not bound"):
            corpus.prefix(
                replace(training_row, population_seed=123),
                2,
            )

    def test_one_shot_transmitter_independence_is_explicitly_inapplicable(
        self,
    ) -> None:
        corpus = self._load()
        identity = corpus.audit["identity_separation"]
        self.assertEqual(
            set(identity),
            set(data.SCALE_ORBIT_REQUIRED_IDENTITY_AUDIT_KEYS),
        )
        self.assertEqual(
            identity["inapplicable_keys_by_replay"],
            {
                "cyclic": ["profile_and_payload_seed"],
                "one-shot": [
                    "base_transmitter_identity",
                    "profile_and_payload_seed",
                    "profile_and_phase_native_sample",
                ],
            },
        )
        self.assertNotIn(
            "profile_and_payload_seed_when_present",
            identity["applicable_collision_counts"],
        )
        self.assertTrue(identity["all_applicable_collision_counts_zero"])
        self.assertEqual(
            set(identity["applicable_collision_counts"].values()),
            {0},
        )

    def test_class_index_order_is_frozen(self) -> None:
        changed = dict(PUBLIC_CLASS_INDEX)
        changed["am"], changed["bluetooth"] = (
            changed["bluetooth"],
            changed["am"],
        )
        with mock.patch.object(
            data.scale_data,
            "load_scale_eval_corpus",
        ) as loader:
            with self.assertRaisesRegex(ValueError, "class index/order"):
                data.load_scale_orbit_training_corpus(
                    data.REPO / ".artifacts" / "train",
                    selection_directory=(
                        data.REPO / ".artifacts" / "selection"
                    ),
                    class_index=changed,
                )
        loader.assert_not_called()

    def test_contract_hash_failure_stops_before_corpus_read(self) -> None:
        with (
            mock.patch.object(
                data.current_data,
                "sha256_file",
                return_value="0" * 64,
            ),
            mock.patch.object(
                data.scale_data,
                "load_scale_eval_corpus",
            ) as loader,
        ):
            with self.assertRaisesRegex(ValueError, "hash changed"):
                data.load_scale_orbit_training_corpus(
                    data.REPO / ".artifacts" / "train",
                    selection_directory=(
                        data.REPO / ".artifacts" / "selection"
                    ),
                    class_index=PUBLIC_CLASS_INDEX,
                )
        loader.assert_not_called()

    def test_training_and_selection_directories_must_be_distinct(self) -> None:
        directory = data.REPO / ".artifacts" / "same"
        with mock.patch.object(
            data.scale_data,
            "load_scale_eval_corpus",
        ) as loader:
            with self.assertRaisesRegex(ValueError, "distinct"):
                data.load_scale_orbit_training_corpus(
                    directory,
                    selection_directory=directory,
                    class_index=PUBLIC_CLASS_INDEX,
                )
        loader.assert_not_called()

    def test_population_seed_mismatch_is_rejected(self) -> None:
        training = _fake_population(
            data.SCALE_ORBIT_TRAINING_SEED,
            marker=1,
        )
        training.manifest["evalSeed"] = data.SCALE_ORBIT_SELECTION_SEED
        selection = _fake_population(
            data.SCALE_ORBIT_SELECTION_SEED,
            marker=2,
        )
        with mock.patch.object(
            data.scale_data,
            "load_scale_eval_corpus",
            side_effect=[training, selection],
        ):
            with self.assertRaisesRegex(ValueError, "preregistered population"):
                data.load_scale_orbit_training_corpus(
                    training.directory,
                    selection_directory=selection.directory,
                    class_index=PUBLIC_CLASS_INDEX,
                )

    def test_pair_metadata_must_align_across_all_four_scale_views(self) -> None:
        mutations = {
            "replay": "cyclic",
            "phase_native_sample": 9,
            "channel_seed": 7,
            "receiver_seed": 11,
            "native_sample_rate_hz": 2_000_000,
            "capture_bandwidth_hz": 300_000,
        }
        for field, changed_value in mutations.items():
            with self.subTest(field=field):
                corpus = _fake_population(
                    data.SCALE_ORBIT_TRAINING_SEED,
                    marker=1,
                )
                corpus.rows[0] = replace(
                    corpus.rows[0],
                    **{field: changed_value},
                )
                with self.assertRaisesRegex(ValueError, "alignment"):
                    data._audit_pairs(
                        corpus,
                        split_counts=data.SCALE_ORBIT_TRAINING_SPLIT_COUNTS,
                        role_for_index=data.role_for_training_realization,
                    )

    def test_pair_scale_view_content_hashes_must_be_distinct(self) -> None:
        corpus = _fake_population(
            data.SCALE_ORBIT_TRAINING_SEED,
            marker=1,
        )
        corpus.rows[1] = replace(
            corpus.rows[1],
            content_sha256=corpus.rows[0].content_sha256,
        )
        with self.assertRaisesRegex(ValueError, "alignment"):
            data._audit_pairs(
                corpus,
                split_counts=data.SCALE_ORBIT_TRAINING_SPLIT_COUNTS,
                role_for_index=data.role_for_training_realization,
            )

    def test_exact_realization_index_set_is_required_per_profile(self) -> None:
        corpus = _fake_population(
            data.SCALE_ORBIT_TRAINING_SEED,
            marker=1,
        )
        _replace_pair_rows(
            corpus,
            profile="wifi-hr-dsss-11m",
            target_realization_index=63,
            realization_index=62,
        )
        with self.assertRaisesRegex(ValueError, "realization-index set"):
            data._audit_pairs(
                corpus,
                split_counts=data.SCALE_ORBIT_TRAINING_SPLIT_COUNTS,
                role_for_index=data.role_for_training_realization,
            )

    def test_each_applicable_cross_role_collision_is_rejected(self) -> None:
        collision_values: dict[str, object] = {
            "pair_id": "pair",
            "content_sha256": "content",
            "runtime_prefix_sha256": "prefix",
            "cyclic_profile_and_phase_native_sample": ("profile", 1),
            "receiver_realization_channel_seed": 1,
            "receiver_realization_seed_when_non_null": 2,
        }
        for key, value in collision_values.items():
            with self.subTest(key=key):
                inventory = _empty_identity_inventory()
                inventory["train"][key].add(value)
                inventory["enrollment"][key].add(value)
                with self.assertRaisesRegex(ValueError, "collision"):
                    data._assert_cross_role_separation(inventory)

    def test_runtime_prefix_collision_is_rejected_by_composite_loader(
        self,
    ) -> None:
        training = _fake_population(
            data.SCALE_ORBIT_TRAINING_SEED,
            marker=1,
        )
        selection = _fake_population(
            data.SCALE_ORBIT_SELECTION_SEED,
            marker=2,
        )
        selection.raw[0] = training.raw[0]
        with mock.patch.object(
            data.scale_data,
            "load_scale_eval_corpus",
            side_effect=[training, selection],
        ):
            with self.assertRaisesRegex(ValueError, "collision"):
                data.load_scale_orbit_training_corpus(
                    training.directory,
                    selection_directory=selection.directory,
                    class_index=PUBLIC_CLASS_INDEX,
                )

    def test_content_collision_is_rejected_by_composite_loader(self) -> None:
        training = _fake_population(
            data.SCALE_ORBIT_TRAINING_SEED,
            marker=1,
        )
        selection = _fake_population(
            data.SCALE_ORBIT_SELECTION_SEED,
            marker=2,
        )
        selection.rows[0] = replace(
            selection.rows[0],
            content_sha256=training.rows[0].content_sha256,
        )
        with mock.patch.object(
            data.scale_data,
            "load_scale_eval_corpus",
            side_effect=[training, selection],
        ):
            with self.assertRaisesRegex(ValueError, "content|collision"):
                data.load_scale_orbit_training_corpus(
                    training.directory,
                    selection_directory=selection.directory,
                    class_index=PUBLIC_CLASS_INDEX,
                )

    def test_pair_and_receiver_identity_collisions_are_wired_to_loader(
        self,
    ) -> None:
        mutations = (
            ("pair_id", lambda row: row.pair_id),
            ("channel_seed", lambda row: row.channel_seed),
            ("receiver_seed", lambda row: row.receiver_seed),
        )
        for field, training_value in mutations:
            with self.subTest(field=field):
                training = _fake_population(
                    data.SCALE_ORBIT_TRAINING_SEED,
                    marker=1,
                )
                selection = _fake_population(
                    data.SCALE_ORBIT_SELECTION_SEED,
                    marker=2,
                )
                _replace_pair_rows(
                    selection,
                    profile="bluetooth-le-advertising",
                    target_realization_index=0,
                    **{field: training_value(training.rows[0])},
                )
                with mock.patch.object(
                    data.scale_data,
                    "load_scale_eval_corpus",
                    side_effect=[training, selection],
                ):
                    with self.assertRaisesRegex(ValueError, "collision"):
                        data.load_scale_orbit_training_corpus(
                            training.directory,
                            selection_directory=selection.directory,
                            class_index=PUBLIC_CLASS_INDEX,
                        )

    def test_cyclic_phase_collision_is_rejected(self) -> None:
        training = _fake_population(
            data.SCALE_ORBIT_TRAINING_SEED,
            marker=1,
        )
        selection = _fake_population(
            data.SCALE_ORBIT_SELECTION_SEED,
            marker=2,
        )
        training_phase = next(
            row.phase_native_sample
            for row in training.rows
            if (
                row.profile == "wifi-hr-dsss-11m"
                and row.realization_index == 0
            )
        )
        _replace_pair_rows(
            selection,
            profile="wifi-hr-dsss-11m",
            target_realization_index=0,
            phase_native_sample=training_phase,
        )
        with mock.patch.object(
            data.scale_data,
            "load_scale_eval_corpus",
            side_effect=[training, selection],
        ):
            with self.assertRaisesRegex(ValueError, "collision"):
                data.load_scale_orbit_training_corpus(
                    training.directory,
                    selection_directory=selection.directory,
                    class_index=PUBLIC_CLASS_INDEX,
                )

    def test_one_shot_nonzero_phase_is_rejected(self) -> None:
        training = _fake_population(
            data.SCALE_ORBIT_TRAINING_SEED,
            marker=1,
        )
        selection = _fake_population(
            data.SCALE_ORBIT_SELECTION_SEED,
            marker=2,
        )
        _replace_pair_rows(
            selection,
            profile="bluetooth-le-advertising",
            target_realization_index=0,
            phase_native_sample=1,
        )
        with mock.patch.object(
            data.scale_data,
            "load_scale_eval_corpus",
            side_effect=[training, selection],
        ):
            with self.assertRaisesRegex(ValueError, "origin phase zero"):
                data.load_scale_orbit_training_corpus(
                    training.directory,
                    selection_directory=selection.directory,
                    class_index=PUBLIC_CLASS_INDEX,
                )


if __name__ == "__main__":
    unittest.main()
