from __future__ import annotations

from dataclasses import replace
import io
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_scale_orbit_dev as runner  # noqa: E402
import scale_orbit_data  # noqa: E402


CLASSES = ("am", "bluetooth", "cw", "dsss", "fm", "gsm", "ofdm")
SCALES = (1.0, 1.25, 1.5, 2.0)


def _supervised_pair_pool() -> tuple[
    dict[str, tuple[runner.SupervisedPairIdentity, ...]],
    int,
]:
    pool: dict[str, tuple[runner.SupervisedPairIdentity, ...]] = {}
    position = 0
    for profile_position, profile in enumerate(
        runner.SUPERVISED_PAIR_PROFILES
    ):
        entries: list[runner.SupervisedPairIdentity] = []
        for realization in range(
            scale_orbit_data.SCALE_ORBIT_TRAINING_SPLIT_COUNTS["train"]
        ):
            positions = tuple(range(position, position + len(SCALES)))
            position += len(SCALES)
            entries.append(
                runner.SupervisedPairIdentity(
                    identity=(
                        "scale-orbit:"
                        f"{scale_orbit_data.SCALE_ORBIT_TRAINING_SEED}:"
                        f"{profile}:{realization}"
                    ),
                    source="current",
                    role="train",
                    population_seed=
                        scale_orbit_data.SCALE_ORBIT_TRAINING_SEED,
                    profile_id=profile,
                    public_class_name=(
                        runner.SUPERVISED_PAIR_PROFILE_PUBLIC_CLASS_MAP[
                            profile
                        ]
                    ),
                    public_class_index=(
                        runner.SUPERVISED_PAIR_PROFILE_CLASS_INDICES[
                            profile_position
                        ]
                    ),
                    scale_factors=SCALES,
                    view_positions=positions,
                )
            )
        pool[profile] = tuple(entries)
    return pool, position


def _paired_positions() -> np.ndarray:
    return np.arange(
        runner.SUPERVISED_PAIR_EXPECTED_PROFILE_COUNT * 2,
        dtype=np.int64,
    ).reshape(runner.SUPERVISED_PAIR_EXPECTED_PROFILE_COUNT, 2)


def _paired_labels() -> np.ndarray:
    return np.asarray(
        runner.SUPERVISED_PAIR_PROFILE_CLASS_INDICES,
        dtype=np.int64,
    )


def _rows() -> tuple[list[scale_orbit_data.ScaleOrbitRowRef], np.ndarray]:
    rows: list[scale_orbit_data.ScaleOrbitRowRef] = []
    labels: list[int] = []
    profiles = (
        ("bluetooth-le-advertising", "bluetooth", 1),
        ("wifi-hr-dsss-11m", "dsss", 3),
    )
    for profile_index, (profile, class_name, label) in enumerate(profiles):
        pair_id = f"pair-{profile}"
        for scale_index, scale in enumerate(SCALES):
            rows.append(
                scale_orbit_data.ScaleOrbitRowRef(
                    source="current",
                    source_index=len(rows),
                    role="selection",
                    class_name=class_name,
                    label=label,
                    profile_id=profile,
                    valid_sample_count=4096,
                    storage_sample_count=16384,
                    identity=f"scale-orbit:20262904:{pair_id}",
                    content_sha256=f"{len(rows):064x}",
                    zero_padded_after_valid=True,
                    pair_id=pair_id,
                    realization_index=profile_index,
                    physical_scale_factor=scale,
                    sample_rate_hz=int(8_000_000 * scale),
                    native_sample_rate_hz=8_000_000,
                    capture_bandwidth_hz=4_000_000,
                    population_seed=20262904,
                    replay="cyclic",
                    phase_native_sample=profile_index,
                    receiver_channel_seed=100 + profile_index,
                    receiver_seed=200 + scale_index,
                )
            )
            labels.append(label)
    return rows, np.asarray(labels, dtype=np.int64)


class ScaleOrbitRunnerTests(unittest.TestCase):
    def test_executed_source_paths_bind_identity_acceptance_and_adaptation(
        self,
    ) -> None:
        paths = runner._executed_source_paths()
        self.assertEqual(
            paths["v5/seed20262904_identity_firewall_acceptance.json"],
            scale_orbit_data.SCALE_ORBIT_IDENTITY_ACCEPTANCE_PATH.resolve(),
        )
        self.assertEqual(
            paths["v4/evaluate_current_scale.py"],
            Path(scale_orbit_data.scale_data.__file__).resolve(),
        )
        self.assertEqual(
            paths["v5/scale_consistency_adaptation_attempt_1.json"],
            runner.ATTEMPT_1_ADAPTATION_PATH.resolve(),
        )
        self.assertEqual(
            paths[
                "v5/scale_consistency_adaptation_attempt_1_miss_report.json"
            ],
            runner.ATTEMPT_1_MISS_REPORT_PATH.resolve(),
        )
        self.assertEqual(
            paths["v5/scale_consistency_adaptation_attempt_2.json"],
            runner.SUPERVISED_PAIR_ADAPTATION_PATH.resolve(),
        )
        self.assertEqual(
            paths[
                "v5/scale_consistency_adaptation_attempt_2_miss_report.json"
            ],
            runner.ATTEMPT_2_MISS_REPORT_PATH.resolve(),
        )
        self.assertEqual(
            paths["v5/scale_consistency_adaptation_attempt_3.json"],
            runner.HARD_VIEW_ADAPTATION_PATH.resolve(),
        )
        expected_source_keys = {
            "v5/run_scale_orbit_dev.py",
            "v5/scale_orbit_data.py",
            "v5/trusted_geometry_canonicalizer.py",
            "v5/identity_firewall_amendment.json",
            "v5/seed20262904_identity_rejection.json",
            "v5/seed20262904_identity_firewall_acceptance.json",
            "v5/scale_consistency_adaptation_attempt_1.json",
            "v5/scale_consistency_adaptation_attempt_1_miss_report.json",
            "v5/scale_consistency_adaptation_attempt_2.json",
            "v5/scale_consistency_adaptation_attempt_2_miss_report.json",
            "v5/scale_consistency_adaptation_attempt_3.json",
            "v4/current_source_data.py",
            "v4/evaluate_current_scale.py",
            "v2/run_invariant_cnn_dev.py",
            "v3/time_domain_invariant_patch_preprocess.py",
            "v3/time_domain_geometry.py",
            "v3/invariant_patch_preprocess.py",
            "v3/invariant_patch_cnn.py",
            "v3/invariant_patch_data.py",
            "v2/complex_multiscale_backbone.py",
            "v2/unet_transfer.py",
            "v2/vit_backbone.py",
            "v2/multitask_autoencoder.py",
            "training/model.py",
            "training/preprocess.py",
            "training/train.py",
        }
        self.assertEqual(set(paths), expected_source_keys)
        expected_dependency_paths = {
            "v2/complex_multiscale_backbone.py":
                runner.V2 / "complex_multiscale_backbone.py",
            "v2/unet_transfer.py": runner.V2 / "unet_transfer.py",
            "v2/vit_backbone.py": runner.V2 / "vit_backbone.py",
            "v2/multitask_autoencoder.py":
                runner.V2 / "multitask_autoencoder.py",
            "training/model.py": runner.TRAINING / "model.py",
            "training/preprocess.py": runner.TRAINING / "preprocess.py",
        }
        for key, expected_path in expected_dependency_paths.items():
            self.assertEqual(paths[key], expected_path.resolve())
        binding = runner._validate_supervised_pair_adaptation_source()
        self.assertEqual(
            binding["sha256"],
            runner.HARD_VIEW_ADAPTATION_SHA256,
        )
        self.assertEqual(
            binding["preregistration_commit"],
            "64e6799a8bf3d38bbbed62d2ae5006c60f79c936",
        )
        self.assertEqual(binding["adaptation_attempt"], 3)
        self.assertEqual(
            binding["attempt_1_intent_sha256"],
            runner.ATTEMPT_1_ADAPTATION_SHA256,
        )
        self.assertEqual(
            binding["attempt_1_miss_report_sha256"],
            "1c0e6d94d418e3954a54aa44396ef5cc9d8603eb2895dd69c0a0c458c254bc9d",
        )
        self.assertEqual(
            binding["attempt_2_intent_sha256"],
            runner.SUPERVISED_PAIR_ADAPTATION_SHA256,
        )
        self.assertEqual(
            binding["attempt_2_miss_report_sha256"],
            runner.ATTEMPT_2_MISS_REPORT_SHA256,
        )
        source_hashes = runner._source_hashes()
        self.assertEqual(
            source_hashes[
                "v5/scale_consistency_adaptation_attempt_1.json"
            ],
            runner.ATTEMPT_1_ADAPTATION_SHA256,
        )
        self.assertEqual(
            source_hashes[
                "v5/scale_consistency_adaptation_attempt_1_miss_report.json"
            ],
            runner.ATTEMPT_1_MISS_REPORT_SHA256,
        )
        self.assertEqual(
            source_hashes[
                "v5/scale_consistency_adaptation_attempt_2.json"
            ],
            runner.SUPERVISED_PAIR_ADAPTATION_SHA256,
        )
        self.assertEqual(
            source_hashes[
                "v5/scale_consistency_adaptation_attempt_2_miss_report.json"
            ],
            runner.ATTEMPT_2_MISS_REPORT_SHA256,
        )
        self.assertEqual(
            source_hashes[
                "v5/scale_consistency_adaptation_attempt_3.json"
            ],
            runner.HARD_VIEW_ADAPTATION_SHA256,
        )

    def test_supervised_pair_pool_binds_exact_profile_classes_and_targets(
        self,
    ) -> None:
        pool, training_view_count = _supervised_pair_pool()
        audit = runner._validate_supervised_pair_pool(
            pool,
            training_view_count=training_view_count,
            classes=CLASSES,
        )
        self.assertEqual(audit["literal_profile_count"], 31)
        self.assertEqual(audit["identity_count"], 31 * 40)
        self.assertEqual(
            audit["addressable_training_scale_views"],
            31 * 40 * 4,
        )
        self.assertEqual(
            audit["literal_profile_public_class_map"],
            runner.SUPERVISED_PAIR_PROFILE_PUBLIC_CLASS_MAP,
        )
        self.assertEqual(
            audit["literal_profile_public_class_index_map"],
            dict(
                zip(
                    runner.SUPERVISED_PAIR_PROFILES,
                    runner.SUPERVISED_PAIR_PROFILE_CLASS_INDICES,
                )
            ),
        )
        self.assertEqual(
            audit["public_class_indices_in_literal_profile_order"],
            list(runner.SUPERVISED_PAIR_PROFILE_CLASS_INDICES),
        )
        self.assertEqual(
            audit["paired_targets_sha256"],
            runner.SUPERVISED_PAIR_TARGET_SHA256,
        )
        self.assertEqual(
            audit["target_construction"],
            "repeat(profile_public_class_indices, 2)",
        )
        self.assertEqual(
            audit["pair_rng"],
            (
                "separate numpy.default_rng(seed xor 0x5CA1E); episodic RNG "
                "state is never passed to supervised-pair sampling"
            ),
        )
        self.assertEqual(
            audit["fitting_firewall"],
            {
                "seed20264101_train_identities_addressable": 31 * 40,
                "seed20264101_enrollment_rows_addressable": 0,
                "seed20262904_selection_rows_addressable": 0,
                "sealed_rows_addressable": 0,
            },
        )
        self.assertEqual(
            runner._supervised_pair_profile_class_indices(CLASSES),
            runner.SUPERVISED_PAIR_PROFILE_CLASS_INDICES,
        )
        self.assertEqual(
            runner._supervised_pair_target_sha256(
                runner.SUPERVISED_PAIR_PROFILE_CLASS_INDICES
            ),
            runner.SUPERVISED_PAIR_TARGET_SHA256,
        )
        with self.assertRaisesRegex(ValueError, "public class order"):
            runner._supervised_pair_profile_class_indices(
                tuple(reversed(CLASSES))
            )

        bad_pool = dict(pool)
        first_profile = runner.SUPERVISED_PAIR_PROFILES[0]
        bad_entries = list(bad_pool[first_profile])
        bad_entries[0] = replace(bad_entries[0], role="enrollment")
        bad_pool[first_profile] = tuple(bad_entries)
        with self.assertRaisesRegex(ValueError, "not an exactly mapped"):
            runner._validate_supervised_pair_pool(
                bad_pool,
                training_view_count=training_view_count,
                classes=CLASSES,
            )

        for field, value in (
            ("public_class_name", "dsss"),
            ("public_class_index", CLASSES.index("dsss")),
        ):
            bad_pool = dict(pool)
            bad_entries = list(bad_pool[first_profile])
            bad_entries[0] = replace(
                bad_entries[0],
                **{field: value},
            )
            bad_pool[first_profile] = tuple(bad_entries)
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    ValueError, "not an exactly mapped"
                ):
                    runner._validate_supervised_pair_pool(
                        bad_pool,
                        training_view_count=training_view_count,
                        classes=CLASSES,
                    )

    def test_pair_rng_is_separate_and_samples_distinct_scales_per_profile(
        self,
    ) -> None:
        pool, training_view_count = _supervised_pair_pool()
        runner._validate_supervised_pair_pool(
            pool,
            training_view_count=training_view_count,
            classes=CLASSES,
        )
        seed = 20_260_740
        episodic_rng = np.random.default_rng(seed)
        episodic_reference = np.random.default_rng(seed)
        pair_rng = np.random.default_rng(
            seed ^ runner.SUPERVISED_PAIR_RNG_XOR
        )
        sampled = runner._sample_supervised_pairs(pool, pair_rng)
        self.assertEqual(
            sampled["positions"].shape,
            (runner.SUPERVISED_PAIR_EXPECTED_PROFILE_COUNT, 2),
        )
        self.assertEqual(
            sampled["profiles"],
            runner.SUPERVISED_PAIR_PROFILES,
        )
        np.testing.assert_array_equal(
            sampled["profile_class_indices"],
            np.asarray(
                runner.SUPERVISED_PAIR_PROFILE_CLASS_INDICES,
                dtype=np.int64,
            ),
        )
        self.assertEqual(
            sampled["paired_targets_sha256"],
            runner.SUPERVISED_PAIR_TARGET_SHA256,
        )
        self.assertTrue(
            np.all(
                sampled["scale_factors"][:, 0]
                != sampled["scale_factors"][:, 1]
            )
        )
        np.testing.assert_array_equal(
            episodic_rng.integers(0, 2**31, size=128),
            episodic_reference.integers(0, 2**31, size=128),
        )

        repeated = runner._sample_supervised_pairs(
            pool,
            np.random.default_rng(
                seed ^ runner.SUPERVISED_PAIR_RNG_XOR
            ),
        )
        np.testing.assert_array_equal(
            sampled["positions"],
            repeated["positions"],
        )
        np.testing.assert_array_equal(
            sampled["scale_factors"],
            repeated["scale_factors"],
        )

        invalid = dict(pool)
        first_profile = runner.SUPERVISED_PAIR_PROFILES[0]
        invalid[first_profile] = tuple(
            replace(entry, public_class_index=CLASSES.index("dsss"))
            for entry in pool[first_profile]
        )
        with self.assertRaisesRegex(ValueError, "invalid mapped entry"):
            runner._sample_supervised_pairs(
                invalid,
                np.random.default_rng(9),
            )

    def test_targets_repeat_each_profile_label_and_never_tile(self) -> None:
        labels = _paired_labels()
        repeated = np.repeat(labels, 2)
        tiled = np.tile(labels, 2)
        self.assertFalse(np.array_equal(repeated, tiled))
        np.testing.assert_array_equal(
            repeated.reshape(-1, 2)[:, 0],
            labels,
        )
        np.testing.assert_array_equal(
            repeated.reshape(-1, 2)[:, 1],
            labels,
        )
        self.assertEqual(
            runner.corpus_data.sha256_json(repeated.tolist()),
            runner.SUPERVISED_PAIR_TARGET_SHA256,
        )
        self.assertNotEqual(
            runner.corpus_data.sha256_json(tiled.tolist()),
            runner.SUPERVISED_PAIR_TARGET_SHA256,
        )

    def test_supervised_pair_cross_entropy_matches_hand_calculation(
        self,
    ) -> None:
        pair_count = runner.SUPERVISED_PAIR_EXPECTED_PROFILE_COUNT * 2
        features = np.linspace(
            -1.5,
            2.0,
            num=pair_count * 3,
            dtype=np.float32,
        ).reshape(pair_count, 3)
        data = {
            "xtr": np.ones((pair_count, 2, 4), dtype=np.float32),
            "ftr": features,
            "n_classes": len(CLASSES),
        }

        class FeatureNet(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(3, 2, bias=False)
                with torch.no_grad():
                    self.linear.weight.copy_(
                        torch.tensor(
                            [[0.4, -0.3, 0.7], [-0.2, 0.8, 0.1]]
                        )
                    )

            def forward(
                self,
                x: torch.Tensor,
                feature_batch: torch.Tensor,
            ) -> torch.Tensor:
                self.observed_batch_shape = tuple(x.shape)
                return self.linear(feature_batch)

        net = FeatureNet()
        prototypes = torch.tensor(
            [
                [-1.0, 0.2],
                [-0.6, -0.4],
                [-0.1, 0.9],
                [0.3, -0.8],
                [0.7, 0.4],
                [1.1, -0.2],
                [1.5, 0.8],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )
        log_scale = torch.tensor(
            float(np.log(2.5)),
            dtype=torch.float32,
            requires_grad=True,
        )
        with mock.patch.object(
            runner.v3_runner,
            "_phase_augment",
            side_effect=lambda values: values,
        ) as phase_augment:
            observed = runner._supervised_pair_cross_entropy_for_pairs(
                net,
                data,
                _paired_positions(),
                _paired_labels(),
                prototypes,
                log_scale,
                torch.device("cpu"),
                phase_augmentation=True,
            )
        embeddings = net(
            torch.from_numpy(data["xtr"]),
            torch.from_numpy(features),
        )
        expected_logits = (
            -runner.sq_dist(embeddings, prototypes.detach())
            * log_scale.detach().exp().clamp(1e-3, 100.0)
        )
        expected_per_view = torch.nn.functional.cross_entropy(
            expected_logits,
            torch.from_numpy(np.repeat(_paired_labels(), 2)),
            label_smoothing=0.0,
            reduction="none",
        )
        expected = torch.amax(
            expected_per_view.reshape(
                runner.SUPERVISED_PAIR_EXPECTED_PROFILE_COUNT,
                2,
            ),
            dim=1,
        ).mean()
        mean_over_all_views = expected_per_view.mean()
        self.assertFalse(
            torch.isclose(expected, mean_over_all_views).item()
        )
        torch.testing.assert_close(observed, expected)
        self.assertEqual(
            net.observed_batch_shape[0],
            pair_count,
        )
        phase_augment.assert_called_once()
        self.assertEqual(
            phase_augment.call_args.args[0].shape[0],
            pair_count,
        )
        total = runner._combine_adaptation_losses(
            torch.tensor(2.0),
            observed,
            weight=0.2,
        )
        torch.testing.assert_close(
            total,
            torch.tensor(2.0) + 0.2 * observed,
        )
        with self.assertRaisesRegex(ValueError, "preregistered 0.2"):
            runner._combine_adaptation_losses(
                torch.tensor(2.0),
                observed,
                weight=0.1,
            )

        self.assertIsNone(prototypes.grad)
        self.assertIsNone(log_scale.grad)
        with self.assertRaisesRegex(ValueError, "must remain enabled"):
            runner._supervised_pair_cross_entropy_for_pairs(
                net,
                data,
                _paired_positions(),
                _paired_labels(),
                prototypes,
                log_scale,
                torch.device("cpu"),
                phase_augmentation=False,
            )

    def test_hard_view_reduction_backpropagates_only_through_each_max(self):
        easy = torch.arange(31, dtype=torch.float32)
        hard = easy + 100.0
        per_view = torch.stack((easy, hard), dim=1).reshape(-1)
        per_view.requires_grad_(True)
        loss = runner._hard_view_supervised_pair_reduction(per_view)
        torch.testing.assert_close(loss, hard.mean())
        loss.backward()
        expected_gradients = torch.zeros(31, 2)
        expected_gradients[:, 1] = 1.0 / 31.0
        torch.testing.assert_close(
            per_view.grad.reshape(31, 2),
            expected_gradients,
        )

    def test_auxiliary_updates_encoder_only_and_preserves_batch_norm(
        self,
    ) -> None:
        torch.manual_seed(11)
        net = runner.InvariantPatchCNN(
            runner.InvariantPatchConfig(
                encoder="real",
                patch_length=16,
                patch_count=1,
                patch_dim=4,
                hidden=4,
                embed_dim=3,
                n_features=2,
                set_pool="mean",
                dropout=0.35,
            )
        )
        net.train()
        batch_norm_modules = [
            module
            for module in net.modules()
            if isinstance(
                module,
                (
                    torch.nn.BatchNorm1d,
                    torch.nn.BatchNorm2d,
                    torch.nn.BatchNorm3d,
                    torch.nn.SyncBatchNorm,
                ),
            )
        ]
        self.assertTrue(batch_norm_modules)
        dropout_modules = [
            module
            for module in net.modules()
            if isinstance(module, torch.nn.Dropout)
        ]
        self.assertTrue(dropout_modules)
        batch_norm_modules[0].eval()
        original_modes = [
            module.training for module in batch_norm_modules
        ]
        original_buffers = [
            {
                name: value.detach().clone()
                for name, value in module.named_buffers(recurse=False)
            }
            for module in batch_norm_modules
        ]
        rng = np.random.default_rng(12)
        pair_count = runner.SUPERVISED_PAIR_EXPECTED_PROFILE_COUNT * 2
        data = {
            "xtr": rng.normal(
                size=(pair_count, 2, net.packed_length)
            ).astype(np.float32),
            "ftr": rng.normal(
                size=(pair_count, net.cfg.n_features)
            ).astype(np.float32),
            "n_classes": len(CLASSES),
        }
        prototypes = torch.randn(
            len(CLASSES),
            net.cfg.embed_dim,
            requires_grad=True,
        )
        log_scale = torch.tensor(
            0.7,
            dtype=torch.float32,
            requires_grad=True,
        )

        loss = runner._supervised_pair_cross_entropy_for_pairs(
            net,
            data,
            _paired_positions(),
            _paired_labels(),
            prototypes,
            log_scale,
            torch.device("cpu"),
            phase_augmentation=True,
        )

        self.assertTrue(loss.requires_grad)
        self.assertEqual(
            [module.training for module in batch_norm_modules],
            original_modes,
        )
        self.assertTrue(all(module.training for module in dropout_modules))
        for module, expected in zip(batch_norm_modules, original_buffers):
            self.assertEqual(set(module._buffers), set(expected))
            for name, value in expected.items():
                torch.testing.assert_close(module._buffers[name], value)
        loss.backward()
        parameter_gradients = [
            parameter.grad
            for parameter in net.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(parameter_gradients)
        self.assertTrue(
            all(
                bool(torch.isfinite(gradient).all())
                for gradient in parameter_gradients
            )
        )
        self.assertTrue(
            any(bool(torch.any(gradient != 0)) for gradient in parameter_gradients)
        )
        self.assertIsNone(prototypes.grad)
        self.assertIsNone(log_scale.grad)

    def test_auxiliary_exception_restores_bn_and_leaves_dropout_live(
        self,
    ) -> None:
        class RaisingNet(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bn = torch.nn.BatchNorm1d(2)
                self.dropout = torch.nn.Dropout(0.4)
                self.mode_seen: tuple[bool, bool] | None = None

            def forward(
                self,
                x: torch.Tensor,
                features: torch.Tensor,
            ) -> torch.Tensor:
                self.mode_seen = (self.bn.training, self.dropout.training)
                self.dropout(self.bn(features))
                raise RuntimeError("deliberate auxiliary failure")

        net = RaisingNet().train()
        original_buffers = {
            name: value.detach().clone()
            for name, value in net.bn.named_buffers(recurse=False)
        }
        pair_count = runner.SUPERVISED_PAIR_EXPECTED_PROFILE_COUNT * 2
        data = {
            "xtr": np.ones((pair_count, 2, 2), dtype=np.float32),
            "ftr": np.ones((pair_count, 2), dtype=np.float32),
            "n_classes": len(CLASSES),
        }
        with self.assertRaisesRegex(RuntimeError, "deliberate auxiliary"):
            runner._supervised_pair_cross_entropy_for_pairs(
                net,
                data,
                _paired_positions(),
                _paired_labels(),
                torch.zeros(len(CLASSES), 2),
                torch.tensor(0.0),
                torch.device("cpu"),
                phase_augmentation=True,
            )
        self.assertEqual(net.mode_seen, (False, True))
        self.assertTrue(net.bn.training)
        self.assertTrue(net.dropout.training)
        for name, value in original_buffers.items():
            torch.testing.assert_close(net.bn._buffers[name], value)

    def test_cli_run_configuration_and_metadata_freeze_attempt_three(self):
        args = runner.build_parser().parse_args(
            [
                "--current-corpus",
                "train",
                "--current-selection-corpus",
                "selection",
                "--output-dir",
                "output",
                "--supervised-pair-weight",
                "0.2",
            ]
        )
        runner._validate_adaptation_run_configuration(args)
        binding = runner._validate_supervised_pair_adaptation_source()
        configuration = runner._run_configuration(
            args,
            device=torch.device("cpu"),
            adaptation_binding=binding,
        )
        self.assertEqual(
            configuration["schema"],
            "v5-scale-orbit-development-training-v3",
        )
        self.assertEqual(
            configuration["adaptation"]["adaptation_attempt"],
            3,
        )
        self.assertEqual(
            configuration["adaptation"]["sha256"],
            runner.HARD_VIEW_ADAPTATION_SHA256,
        )
        self.assertEqual(
            configuration["adaptation"]["supervised_pair_weight"],
            0.2,
        )
        self.assertEqual(
            configuration["adaptation"]["loss_contract"],
            runner.SUPERVISED_PAIR_LOSS_CONTRACT,
        )
        expected_reduction = runner._supervised_pair_reduction_metadata()
        for key, value in expected_reduction.items():
            self.assertEqual(
                configuration["adaptation"][key],
                value,
            )
        stale = dict(configuration)
        stale["schema"] = "v5-scale-orbit-development-training-v2"
        with self.assertRaisesRegex(ValueError, "training-v3"):
            runner._validate_attempt_3_run_configuration_metadata(stale)
        self.assertEqual(
            configuration["randomness"]["supervised_pair_rng"]["seed"],
            args.seed ^ runner.SUPERVISED_PAIR_RNG_XOR,
        )
        firewall = configuration["adaptation"]["fitting_firewall"]
        self.assertTrue(
            firewall[
                "supervised_pairs_from_seed20264101_current_train_role_only"
            ]
        )
        self.assertTrue(
            firewall[
                "seed20262904_selection_metrics_may_retain_the_existing_"
                "development_checkpoint_and_candidate_selection_role"
            ]
        )
        for key in (
            "seed20264101_enrollment_rows_used_for_supervised_pair_auxiliary",
            "seed20262904_selection_rows_used_for_any_gradient_or_optimizer_step",
            "seed20262904_selection_rows_used_for_weight_center_or_feature_moment_fit",
            "seed20262904_selection_rows_used_for_persistent_prototype_fit",
            "seed20262904_selection_rows_used_for_open_set_threshold_or_rank_fit",
            "sealed_rows_used_for_any_fit_checkpoint_or_candidate_selection",
        ):
            self.assertEqual(firewall[key], 0)
        drifted = runner.build_parser().parse_args(
            [
                "--current-corpus",
                "train",
                "--current-selection-corpus",
                "selection",
                "--output-dir",
                "output",
                "--supervised-pair-weight",
                "0.2",
                "--dropout",
                "0.2",
            ]
        )
        with self.assertRaisesRegex(ValueError, "dropout"):
            runner._validate_adaptation_run_configuration(drifted)

        with (
            self.assertRaises(SystemExit),
            mock.patch("sys.stderr", new=io.StringIO()),
        ):
            runner.build_parser().parse_args(
                [
                    "--current-corpus",
                    "train",
                    "--current-selection-corpus",
                    "selection",
                    "--output-dir",
                    "output",
                    "--supervised-pair-weight",
                    "0.2",
                    "--scale-consistency-weight",
                    "0.2",
                ]
            )

    def test_source_mutation_is_rejected_before_any_corpus_data_access(
        self,
    ) -> None:
        args = runner.build_parser().parse_args(
            [
                "--current-corpus",
                "train",
                "--current-selection-corpus",
                "selection",
                "--output-dir",
                "output",
                "--supervised-pair-weight",
                "0.2",
            ]
        )
        real_sha256_file = runner.corpus_data.sha256_file

        def changed_attempt_3(path: Path) -> str:
            if (
                Path(path).resolve()
                == runner.HARD_VIEW_ADAPTATION_PATH.resolve()
            ):
                return "0" * 64
            return real_sha256_file(path)

        with (
            mock.patch.object(
                runner.corpus_data,
                "sha256_file",
                side_effect=changed_attempt_3,
            ),
            mock.patch.object(
                runner.corpus_data,
                "load_historical_exposed",
            ) as historical_loader,
            self.assertRaisesRegex(RuntimeError, "SHA-256 changed"),
        ):
            runner.run(args)
        historical_loader.assert_not_called()

    def test_runner_contains_no_active_attempt_one_loss_or_cli_contract(
        self,
    ) -> None:
        source = Path(runner.__file__).read_text(encoding="utf-8")
        for stale in (
            "SCALE_CONSISTENCY_",
            "ScaleConsistencyIdentity",
            "_scale_consistency_loss_for_pairs",
            "_mean_paired_cosine_distance",
            "F.cosine_similarity",
            "one_minus_cosine_similarity",
            "--scale-consistency-weight",
            '"scale_consistency_weight"',
            "v5-scale-orbit-development-training-v1",
        ):
            with self.subTest(stale=stale):
                self.assertNotIn(stale, source)

    def test_checkpoint_score_is_scale_sensitive_and_fixed_width(self):
        common = {
            "historical_balanced": 0.98,
            "current_worst_scale_present_class_balanced": 0.96,
            "current_worst_profile_scale_recall": 0.75,
            "current_physical_scale_prediction_agreement": 0.97,
            "current_worst_directional_confusion": 0.01,
            "current_pooled_profile_balanced": 0.96,
            "current_pooled_accuracy": 0.97,
            "combined_balanced": 0.97,
        }
        good = runner.checkpoint_score(
            **common,
            current_wifi_hr_dsss_worst_scale_accuracy=0.98,
        )
        bad = runner.checkpoint_score(
            **common,
            current_wifi_hr_dsss_worst_scale_accuracy=0.25,
        )
        self.assertEqual(len(good), runner.CHECKPOINT_SCORE_TERMS)
        self.assertGreater(good, bad)
        self.assertEqual(good[0], 0.75)
        with self.assertRaises(ValueError):
            runner.checkpoint_score(
                **common,
                current_wifi_hr_dsss_worst_scale_accuracy=float("nan"),
            )

    def test_scale_report_exposes_exact_pair_and_directional_metrics(self):
        rows, labels = _rows()
        perfect = runner._classification_report(
            labels,
            labels,
            CLASSES,
            rows,
        )
        self.assertTrue(perfect["exact_scale_pair_coverage"])
        self.assertEqual(perfect["scale_pair_count"], 2)
        self.assertEqual(
            perfect["expected_physical_scale_factors"],
            list(SCALES),
        )
        self.assertEqual(
            perfect["worst_scale_present_class_balanced_accuracy"],
            1.0,
        )
        self.assertEqual(perfect["worst_profile_scale_recall"], 1.0)
        self.assertEqual(
            perfect["wifi_hr_dsss_worst_scale_accuracy"],
            1.0,
        )
        self.assertEqual(
            perfect["physical_scale_prediction_agreement_rate"],
            1.0,
        )
        self.assertEqual(
            perfect["worst_directional_dsss_bluetooth_confusion_rate"],
            0.0,
        )

        confused = labels.copy()
        wifi_scale_125 = next(
            index
            for index, row in enumerate(rows)
            if (
                row.profile_id == "wifi-hr-dsss-11m"
                and row.physical_scale_factor == 1.25
            )
        )
        confused[wifi_scale_125] = CLASSES.index("bluetooth")
        report = runner._classification_report(
            confused,
            labels,
            CLASSES,
            rows,
        )
        self.assertEqual(report["wifi_hr_dsss_worst_scale_accuracy"], 0.0)
        self.assertEqual(
            report[
                "wifi_hr_dsss_to_bluetooth_worst_scale_confusion_rate"
            ],
            1.0,
        )
        self.assertEqual(
            report["physical_scale_prediction_agreement_rate"],
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
