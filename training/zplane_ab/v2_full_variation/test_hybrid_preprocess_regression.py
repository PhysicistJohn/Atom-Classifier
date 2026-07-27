"""Opt-in, read-only same-ID regression for the hybrid-v3 routing policy.

The routine unit suite must not require the 1.8 GB development corpus. Run this
guard explicitly on a corpus host:

    ATOMOS_RUN_CORPUS_REGRESSION=1 .venv-training/bin/python \
      training/zplane_ab/v2_full_variation/test_hybrid_preprocess_regression.py

It evaluates only the immutable development-selection IDs. The complementary
consumed-test population is never preprocessed or scored.
"""
from __future__ import annotations

import json
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.dirname(os.path.dirname(HERE))
if TRAINING not in sys.path:
    sys.path.insert(0, TRAINING)

import preprocess as pp  # noqa: E402


CORPUS = os.path.join(TRAINING, "artifacts", "signallab-corpus")
POOL_PREFIX = os.path.join(
    CORPUS,
    "_pools",
    "resampled__db607416e0c908a1",
)
GROUND_SPLIT_SEED = 20260727


def _selection_ids(manifest: dict) -> np.ndarray:
    classes = manifest["classes"]
    class_index = {name: index for index, name in enumerate(classes)}
    items = manifest["items"]
    labels = np.asarray([class_index[item["cls"]] for item in items])
    impaired = np.asarray([item["impaired"] for item in items], dtype=bool)
    safe = np.asarray(
        [
            abs(float(item.get("centreOffsetFrac", 0.0)))
            + 0.5 * float(item["bandwidthHz"]) / float(item["sampleRateHz"])
            < 0.5
            for item in items
        ],
        dtype=bool,
    )
    validation = np.load(f"{POOL_PREFIX}_va_idx.npy")
    validation = validation[safe[validation]]
    y = labels[validation]
    imp = impaired[validation]

    rng = np.random.default_rng(GROUND_SPLIT_SEED + 1)
    selected: list[int] = []
    for cls in np.unique(y):
        for impairment in (False, True):
            rows = np.where((y == cls) & (imp == impairment))[0]
            rows = rng.permutation(rows)
            selected.extend(rows[: len(rows) // 2])
    return validation[np.sort(np.asarray(selected, dtype=np.int64))]


@unittest.skipUnless(
    os.environ.get("ATOMOS_RUN_CORPUS_REGRESSION") == "1",
    "set ATOMOS_RUN_CORPUS_REGRESSION=1 on a corpus host",
)
class HybridSameIdGeometryRegressionTest(unittest.TestCase):
    def test_hybrid_changes_only_audited_bluetooth_seam_rows(self):
        with open(os.path.join(CORPUS, "corpus.json")) as handle:
            manifest = json.load(handle)
        ids = _selection_ids(manifest)
        self.assertEqual(len(ids), 1908)

        raw = np.memmap(
            os.path.join(CORPUS, "corpus.f32"),
            dtype="<f4",
            mode="r",
            shape=(
                int(manifest["count"]),
                int(manifest["sampleCount"]),
                2,
            ),
        )
        routed: list[int] = []
        for corpus_id in ids:
            pair = np.asarray(raw[int(corpus_id)], dtype=np.float64)
            iq = pair[:, 0] + 1.0j * pair[:, 1]
            scaled, _ = pp._peak_normalized(iq)
            legacy = pp.estimate_band_linear_v1(scaled)
            circular = pp.estimate_band_circular_v2(scaled)
            if pp._hybrid_uses_circular(legacy, circular, pp.NFFT):
                routed.append(int(corpus_id))

        # Frozen by the audit that selected the policy. These are all
        # frequency-hopping Bluetooth rows whose measured spectrum reaches the
        # seam even though the manifest's nominal single-channel band is safe.
        self.assertEqual(len(routed), 30)
        self.assertEqual(
            {manifest["items"][corpus_id]["cls"] for corpus_id in routed},
            {"bluetooth"},
        )

        # Exercise the public dispatcher on both sides of the branch.
        routed_set = set(routed)
        for corpus_id in (routed[0], next(i for i in ids if int(i) not in routed_set)):
            pair = np.asarray(raw[int(corpus_id)], dtype=np.float64)
            iq = pair[:, 0] + 1.0j * pair[:, 1]
            scaled, _ = pp._peak_normalized(iq)
            legacy = pp.estimate_band_linear_v1(scaled)
            circular = pp.estimate_band_circular_v2(scaled)
            expected = circular if int(corpus_id) in routed_set else legacy
            self.assertEqual(pp.estimate_band(iq), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
