from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import invariant_patch_data as data  # noqa: E402


def _write_synthetic_corpus(root: Path, count: int = 80, length: int = 16):
    classes = ["alpha", "beta"]
    items = []
    raw = np.zeros((count, length, 2), dtype="<f4")
    combinations = (
        ("alpha", False),
        ("alpha", True),
        ("beta", False),
        ("beta", True),
    )
    for row in range(count):
        cls, impaired = combinations[row % len(combinations)]
        unsafe = row % 19 == 0
        items.append(
            {
                "cls": cls,
                "impaired": impaired,
                "sampleRateHz": 100.0,
                "bandwidthHz": 20.0,
                "centreOffsetFrac": 0.45 if unsafe else 0.10,
            }
        )
        raw[row, :, 0] = row + 1
        raw[row, :, 1] = np.arange(length, dtype=np.float32) / (row + 2)
    manifest = {
        "count": count,
        "sampleCount": length,
        "classes": classes,
        "items": items,
    }
    root.mkdir(parents=True)
    (root / "corpus.json").write_text(json.dumps(manifest), encoding="utf-8")
    raw.tofile(root / "corpus.f32")
    return manifest


def _write_source_witnesses(root: Path) -> dict[str, Path]:
    output = {}
    for index, name in enumerate(("production", "invariant", "data")):
        path = root / f"{name}.py"
        path.write_text(f"SOURCE = {index}\n", encoding="utf-8")
        output[f"synthetic/{name}.py"] = path
    return output


def _fake_preprocess(
    iq,
    *,
    patch_length,
    patch_count,
    target_frac,
    nfft,
):
    del target_frac, nfft
    marker = float(np.asarray(iq).real[0])
    packed = np.empty((2, patch_length * patch_count), dtype=np.float32)
    packed[0].fill(marker)
    packed[1].fill(-marker)
    features = np.asarray(
        [marker * (feature + 1) + feature**2 for feature in range(12)],
        dtype=np.float32,
    )
    return packed, features, {"synthetic": True}


def _all_mapping_keys(value):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield key
            yield from _all_mapping_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _all_mapping_keys(nested)


class InvariantPatchDataTest(unittest.TestCase):
    def test_v2_split_reconstruction_has_exact_rng_order(self):
        impaired = np.asarray(
            [False, True, False, True, True, False, False, True] * 4,
            dtype=bool,
        )
        actual = data._reconstruct_v2_split(impaired)

        rng = np.random.default_rng(20260721)
        clean = rng.permutation(np.where(~impaired)[0])
        damaged = rng.permutation(np.where(impaired)[0])

        def carve(position):
            n_enroll = int(0.30 * len(position))
            n_validation = int(0.30 * len(position))
            return (
                position[:n_enroll],
                position[n_enroll : n_enroll + n_validation],
                position[n_enroll + n_validation :],
            )

        clean_en, clean_va, clean_tr = carve(clean)
        imp_en, imp_va, imp_tr = carve(damaged)
        np.testing.assert_array_equal(actual["tr"], np.r_[clean_tr, imp_tr])
        np.testing.assert_array_equal(actual["en"], np.r_[clean_en, imp_en])
        np.testing.assert_array_equal(actual["va"], np.r_[clean_va, imp_va])

    def test_cache_contract_invalidates_on_every_material_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus"
            manifest = _write_synthetic_corpus(corpus)
            sources = _write_source_witnesses(root)
            baseline = data.cache_contract(
                8,
                2,
                0.5,
                corpus_dir=corpus,
                _source_paths=sources,
            )

            config_change = data.cache_contract(
                8,
                3,
                0.5,
                corpus_dir=corpus,
                _source_paths=sources,
            )
            self.assertNotEqual(baseline["id"], config_change["id"])

            manifest["items"][0]["centreOffsetFrac"] = 0.11
            (corpus / "corpus.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            manifest_change = data.cache_contract(
                8,
                2,
                0.5,
                corpus_dir=corpus,
                _source_paths=sources,
            )
            self.assertNotEqual(baseline["id"], manifest_change["id"])

            source_path = next(iter(sources.values()))
            source_path.write_text("SOURCE = 'changed'\n", encoding="utf-8")
            source_change = data.cache_contract(
                8,
                2,
                0.5,
                corpus_dir=corpus,
                _source_paths=sources,
            )
            self.assertNotEqual(manifest_change["id"], source_change["id"])

            with open(corpus / "corpus.f32", "ab") as handle:
                handle.write(b"\0\0\0\0")
            raw_stat_change = data.cache_contract(
                8,
                2,
                0.5,
                corpus_dir=corpus,
                _source_paths=sources,
            )
            self.assertNotEqual(source_change["id"], raw_stat_change["id"])

            with mock.patch.object(
                data,
                "SPLIT_POLICY_VERSION",
                "synthetic-policy-change",
            ):
                policy_change = data.cache_contract(
                    8,
                    2,
                    0.5,
                    corpus_dir=corpus,
                    _source_paths=sources,
                )
            self.assertNotEqual(raw_stat_change["id"], policy_change["id"])

    def test_feature_coordinates_are_fit_on_training_only(self):
        train = np.asarray(
            [[1.0, 5.0, 7.0], [3.0, 5.0, 11.0], [5.0, 5.0, 15.0]],
            dtype=np.float32,
        )
        enroll = np.asarray([[101.0, -50.0, 2.0]], dtype=np.float32)
        selection = np.asarray([[-99.0, 500.0, 20.0]], dtype=np.float32)
        ftr, fen, fva, mean, std = data._standardize_features(
            train,
            enroll,
            selection,
        )
        np.testing.assert_allclose(mean, train.mean(axis=0), atol=1e-7)
        np.testing.assert_allclose(
            std,
            train.astype(np.float64).std(axis=0) + 1e-6,
            atol=1e-7,
        )
        np.testing.assert_allclose(ftr.mean(axis=0), 0.0, atol=2e-7)
        np.testing.assert_allclose(ftr[:, (0, 2)].std(axis=0), 1.0, atol=2e-6)
        np.testing.assert_allclose(fen, (enroll - mean) / std, atol=1e-6)
        np.testing.assert_allclose(fva, (selection - mean) / std, atol=1e-6)

    def test_tiny_cache_never_serializes_or_returns_consumed_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus"
            cache = root / "cache"
            legacy = root / "legacy-empty"
            legacy.mkdir()
            manifest = _write_synthetic_corpus(corpus)
            sources = _write_source_witnesses(root)
            processed_markers = []

            def recording_preprocess(*args, **kwargs):
                processed_markers.append(int(np.asarray(args[0]).real[0]))
                return _fake_preprocess(*args, **kwargs)

            contract = data.build(
                8,
                2,
                0.5,
                corpus_dir=corpus,
                cache_dir=cache,
                legacy_cache_dir=legacy,
                _preprocess_fn=recording_preprocess,
                _source_paths=sources,
            )
            first = data.load(
                8,
                2,
                0.5,
                model_seed=44,
                build_if_missing=False,
                corpus_dir=corpus,
                cache_dir=cache,
                legacy_cache_dir=legacy,
                _source_paths=sources,
            )
            second = data.load(
                8,
                2,
                0.5,
                model_seed=44,
                build_if_missing=False,
                corpus_dir=corpus,
                cache_dir=cache,
                legacy_cache_dir=legacy,
                _source_paths=sources,
            )
            self.assertEqual(first["cache_contract"], contract)
            self.assertEqual(
                first["rng"].integers(0, 2**31),
                second["rng"].integers(0, 2**31),
            )

            classes, labels, impaired = data._labels_from_manifest(manifest)
            del classes, labels
            full = data._reconstruct_v2_split(impaired)
            safe = data._edge_safe_mask(manifest)
            safe_validation = full["va"][safe[full["va"]]]
            exposed_selection = np.asarray(first["va_idx"])
            consumed = np.setdiff1d(
                safe_validation,
                exposed_selection,
                assume_unique=True,
            )
            all_exposed = np.concatenate(
                (
                    np.asarray(first["tr_idx"]),
                    np.asarray(first["en_idx"]),
                    exposed_selection,
                )
            )
            self.assertEqual(len(np.intersect1d(consumed, all_exposed)), 0)
            np.testing.assert_array_equal(
                np.sort(np.asarray(processed_markers, dtype=np.int64) - 1),
                np.sort(all_exposed),
            )
            self.assertEqual(
                len(
                    np.intersect1d(
                        consumed,
                        np.asarray(processed_markers, dtype=np.int64) - 1,
                    )
                ),
                0,
            )

            forbidden_return_keys = {
                "test_idx",
                "test_rows",
                "consumed_test_ids",
                "consumed_test_indices",
            }
            self.assertTrue(forbidden_return_keys.isdisjoint(first))
            self.assertEqual(first["data_audit"]["consumed_test_rows_exposed"], 0)

            metadata_files = list(cache.glob("*_meta.json"))
            self.assertEqual(len(metadata_files), 1)
            metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
            self.assertTrue(forbidden_return_keys.isdisjoint(_all_mapping_keys(metadata)))
            self.assertEqual(
                metadata["data_audit"]["consumed_test_rows_exposed"],
                0,
            )
            self.assertFalse(any("test" in path.stem for path in cache.glob("*.npy")))

            # Completion metadata is written only after every declared array.
            for key in data.ARRAY_KEYS:
                self.assertTrue(data._array_path(cache, contract, key).exists())
            np.testing.assert_allclose(first["ftr"].mean(axis=0), 0.0, atol=2e-6)

    def test_failed_build_has_no_completion_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus"
            cache = root / "cache"
            legacy = root / "legacy-empty"
            legacy.mkdir()
            _write_synthetic_corpus(corpus)
            sources = _write_source_witnesses(root)
            calls = 0

            def fail_during_preprocessing(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise RuntimeError("synthetic interrupted build")
                return _fake_preprocess(*args, **kwargs)

            with self.assertRaisesRegex(RuntimeError, "interrupted build"):
                data.build(
                    8,
                    2,
                    0.5,
                    corpus_dir=corpus,
                    cache_dir=cache,
                    legacy_cache_dir=legacy,
                    _preprocess_fn=fail_during_preprocessing,
                    _source_paths=sources,
                )
            self.assertEqual(list(cache.glob("*_meta.json")), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
