"""Measure the two unmeasured closed-set release gates on a v3 fusion artifact.

Sealed v2 (seed 20260729) failed eight gates.  ``assemble_v3_fusion.py`` addressed
three of them (physical-scale balanced accuracy, paired scale cosine, paired scale
prediction agreement).  Three more are open-set and blocked on the rejector refit.
The two that remain are closed-set and were never computed for v3 at all:

======================================  ========  =========  ============
release gate                            v2 value  threshold  v3 status
======================================  ========  =========  ============
``closed_clean_worst_length``            0.8464     0.85      unmeasured
``five_shot_worst_length_balanced``      0.8495     0.85      unmeasured
======================================  ========  =========  ============

Both are knife-edge.  This module computes them on the development populations,
mirroring ``evaluate_invariant_release_suite.py`` rather than re-deriving it: the
protocol constants (``FIVE_SHOT_K``, ``SPLIT_SALT``, ``SPLIT_VERSION``,
``MATCHED_CAPTURE_LENGTH``, ``GATE_FLOORS``, ``ROW_IDENTITY_FIELDS``) and the
scoring functions (``_nearest``, ``_classification_subset``, ``_five_shot_report``,
``_gate``) are imported from the evaluator so they cannot drift.

This is development evidence.  It is not release evidence, it does not touch the
consumed historical test half or any sealed suite, and release seed ``20260731``
is neither read nor accepted.


What the two gates mean, exactly
--------------------------------

``closed_clean_worst_length`` is ``min`` over capture lengths of
``closed_by_length[length]["clean_accuracy"]``, which
``evaluate_invariant_release_suite._closed_report`` defines as the **plain, not
balanced, row accuracy over the rows whose manifest ``impaired`` field is
``false``**, scored with the bundle's frozen prototypes by Euclidean nearest
prototype.  "Clean" is a metadata subset of the observed stream; it is *not* the
corpus ``clean_path`` companion stream.  The gate value the release evaluator
would print is therefore an unweighted accuracy over an unbalanced subset, so
this module reports the clean *balanced* accuracy beside it at every length.

``five_shot_worst_length_balanced`` is ``min`` over capture lengths of
``_five_shot_report(...)["balanced_accuracy"]``.  That report:

* takes ``k = 5`` support rows per class, chosen as the five lowest lowercase
  SHA-256 digests of ``utf8(SPLIT_SALT + NUL + seed_decimal + NUL +
  canonical_row_identity_json)``, ties broken by ascending row index;
* embeds the support rows at ``MATCHED_CAPTURE_LENGTH = 16384`` and averages them
  per class with a **plain mean, no renormalization**, discarding the frozen
  prototypes;
* reuses that one fixed prototype set unchanged at every query length;
* scores every remaining row by Euclidean nearest prototype;
* reports the unweighted mean of the per-class recalls.

No encoder, fusion, center, or feature-standardization parameter changes.


Where the development population cannot reproduce the sealed protocol
---------------------------------------------------------------------

Read this before quoting any number produced here.

1. **Only three of the four sealed capture lengths exist.**  The release suite
   requires 4096/8192/16384/32768.  The historical development corpus stores
   ``sampleCount = 16384`` and a longer observation may not be synthesized by
   padding or extrapolation, so N32768 is unmeasurable, exactly as
   ``run_time_domain_dev._length_audit`` already records.  ``worst over length``
   here is a worst over ``{4096, 8192, 16384}``.  Accuracy rises monotonically
   with length on every v3 artifact measured so far, so excluding the longest
   length is unlikely to move the worst; that is an expectation, not a bound.

2. **Occupied-start rule.**  The release generator predeclares an unscored start
   probe, so every sealed row carries signal inside its first 4096 samples.  The
   historical development generator allowed a signal to begin after sample 4096.
   Rows whose N4096 prefix is exactly all-zero are excluded here by the same
   pre-inference rule ``run_time_domain_dev._length_audit`` uses
   (``max(abs(iq[:4096])) == 0``), applied identically at all three lengths and
   to both the support and query populations.  The sealed suite has no such
   exclusion because it needs none.  The excluded counts are recorded per class.

3. **Class balance.**  The sealed corpus holds exactly ``MIN_TARGET_PER_CLASS``
   rows per class.  The development selection population does not, and its clean
   subset is more skewed still.  ``clean_accuracy`` is population-weighted in
   both, so the dev clean number leans toward the larger classes.  Compare
   ``clean_balanced_accuracy`` to see how much of the gap that explains.

4. **Query set composition.**  In the sealed evaluator every non-support row is
   query, and the closed/clean report is computed over exactly those query rows,
   so the five-shot support rows are absent from the clean gate too.  Here the
   primary five-shot support is drawn from *enrollment*, so no selection row is
   withheld and the clean gate is computed over the whole length-eligible
   selection population.

5. **Five-shot support population -- the substantive deviation.**  The sealed
   protocol draws support and query from the same sealed corpus.  The development
   corpus cannot do that without breaking an evidence rule: five-shot support *is*
   a prototype fit, and the selection population is scored, never fit.  Two
   variants are therefore reported and neither is hidden:

   ``enrollment_support`` (primary, evidence-rule compliant)
       Support is the digest-lowest five length-eligible **enrollment** rows per
       class; query is every length-eligible **selection** row.  The two
       populations are disjoint by construction, so the partition property the
       sealed protocol enforces by exclusion holds here by separation.

   ``selection_support`` (structural mirror, diagnostic only)
       Support and query are both drawn from the length-eligible **selection**
       population and made disjoint by exclusion, which is the sealed evaluator's
       exact structure.  This variant fits 35 prototypes on selection rows.  It is
       reported as a protocol-fidelity check, not as evidence, and must not be
       used to select anything.

   The assembled gate ``five_shot_worst_length_balanced_conservative`` is the
   minimum over both variants and all three lengths.  It can only be stricter
   than either variant alone, never weaker, so reading it as the gate cannot
   launder a failure into a pass.

6. **Row identity for the digest split.**  ``predeclared_five_shot_split`` hashes
   all fifteen ``ROW_IDENTITY_FIELDS``.  The historical development manifest
   carries fourteen of them: ``impairmentSeed`` is absent (and ``cleanPower`` is
   present but is not a release identity field, so it is ignored).  The digest
   ranking here uses the fourteen present release fields in the same canonical
   ``json.dumps(sort_keys=True, separators=(",", ":"))`` form.  The draw is a
   faithful re-implementation of the rule; it is not the draw a sealed run would
   make, and no development draw could be.

7. **Split seed.**  The sealed hash payload binds the release seed.  There is no
   release seed here.  ``--split-seed`` defaults to the fusion artifact's own
   model seed and refuses ``20260731`` (unspent) and ``20260729`` (consumed).

8. **Device.**  The release evaluator scores on one device; this module accepts
   ``--device`` and records it.  Reduction order can move an accuracy in the
   fourth decimal.  Because both gates are decided in the third decimal, a
   knife-edge result should be re-run on ``--device cpu`` before it is believed.

The measurement re-derives the artifact's own recorded per-length overall accuracy
and balanced accuracy as a reproduction check, and fails closed if the eligible
row set does not hash identically to the one the artifact recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
V2 = HERE.parent
ZPAB = V2.parent
TRAINING = ZPAB.parent
for _path in (TRAINING, ZPAB, V2, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import assemble_v3_fusion as assemble  # noqa: E402
import evaluate_invariant_release_suite as release  # noqa: E402
import invariant_patch_data as invariant_data  # noqa: E402
import run_invariant_cnn_dev as runner  # noqa: E402
import run_time_domain_dev as dev  # noqa: E402
import time_domain_invariant_patch_preprocess as td_preprocess  # noqa: E402
from invariant_fusion import CenteredInvariantFusion  # noqa: E402
from invariant_patch_cnn import (  # noqa: E402
    InvariantPatchCNN,
    InvariantPatchConfig,
)


MEASUREMENT_VERSION = "v3-remaining-closed-gates-v1"
REPORT_NAME = "remaining_gates.json"

# Capture lengths this development corpus can actually observe.  The release
# suite additionally requires 32768, which cannot be produced here.
DEV_LENGTHS = dev.LENGTHS
UNMEASURABLE_RELEASE_LENGTHS = tuple(
    length
    for length in release.REQUIRED_CAPTURE_LENGTHS
    if length not in DEV_LENGTHS
)

# Seeds this module must never spend or re-read.
FORBIDDEN_SPLIT_SEEDS = {
    20260731: "unspent release seed reserved for the sealed v3 suite",
    release.NOVELTY_SEED: "consumed sealed v2 release seed",
}

# The one release row-identity field the historical development manifest does
# not carry.  Any other absence means the corpus is not the expected one.
EXPECTED_ABSENT_IDENTITY_FIELDS = ("impairmentSeed",)

# Allowance when re-deriving the artifact's own recorded per-length accuracy.
# Loose enough for CPU/MPS reduction order, far tighter than either gate margin.
DEFAULT_REPRODUCTION_TOLERANCE = 2e-3

FIVE_SHOT_VARIANTS = ("enrollment_support", "selection_support")
PRIMARY_FIVE_SHOT_VARIANT = "enrollment_support"


# ---------------------------------------------------------------------------
# argument and path discipline
# ---------------------------------------------------------------------------


def validate_split_seed(value: Any) -> int:
    """Accept any development seed; refuse the unspent and consumed release seeds."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("split seed must be an integer")
    seed = int(value)
    if seed < 0:
        raise ValueError("split seed must be non-negative")
    if seed in FORBIDDEN_SPLIT_SEEDS:
        raise ValueError(
            f"refusing split seed {seed}: {FORBIDDEN_SPLIT_SEEDS[seed]}"
        )
    return seed


def validate_tolerance(value: Any) -> float:
    tolerance = float(value)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("reproduction tolerance must be positive and finite")
    return tolerance


# ---------------------------------------------------------------------------
# the development five-shot split, mirroring predeclared_five_shot_split
# ---------------------------------------------------------------------------


def dev_row_identity_fields(items: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """Return the release identity fields this corpus actually carries.

    Fails closed unless every row carries the same subset and the only missing
    field is the known-absent ``impairmentSeed``.
    """
    if not items:
        raise ValueError("row identity requires at least one manifest item")
    present = tuple(
        name for name in release.ROW_IDENTITY_FIELDS if name in items[0]
    )
    for item in items:
        if tuple(
            name for name in release.ROW_IDENTITY_FIELDS if name in item
        ) != present:
            raise ValueError(
                "development corpus rows disagree on which release identity "
                "fields they carry"
            )
    absent = tuple(
        name for name in release.ROW_IDENTITY_FIELDS if name not in present
    )
    if absent != EXPECTED_ABSENT_IDENTITY_FIELDS:
        raise ValueError(
            "unexpected release identity fields missing from the development "
            f"corpus: {absent}"
        )
    return present


def dev_row_identity(
    item: Mapping[str, Any],
    fields: Sequence[str],
) -> str:
    """Canonical identity JSON, same encoder settings as ``_row_identity``."""
    missing = [name for name in fields if name not in item]
    if missing:
        raise ValueError(f"corpus row is missing identity fields: {missing}")
    return json.dumps(
        {name: item[name] for name in fields},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def dev_five_shot_split(
    items: Sequence[Mapping[str, Any]],
    labels: np.ndarray,
    classes: Sequence[str],
    split_seed: int,
    *,
    identity_fields: Sequence[str],
    k: int = release.FIVE_SHOT_K,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return support positions by the release digest rule, over ``items``.

    ``items`` are the manifest rows of one already-eligible population, in the
    order their embeddings appear.  Positions are into that population, not into
    the raw corpus.  The rule -- lowest ``k`` SHA-256 digests per class, ties by
    ascending position -- and the hash payload layout are the release ones.
    """
    seed = validate_split_seed(split_seed)
    if k != release.FIVE_SHOT_K:
        raise ValueError(f"five-shot k is frozen at {release.FIVE_SHOT_K}")
    label_array = np.asarray(labels, dtype=np.int64)
    if len(label_array) != len(items):
        raise ValueError("items and labels disagree in length")

    support_by_class: dict[str, list[int]] = {}
    digest_by_class: dict[str, list[str]] = {}
    all_support: list[int] = []
    for class_index, class_name in enumerate(classes):
        candidates = [
            position
            for position in range(len(items))
            if int(label_array[position]) == class_index
        ]
        if len(candidates) <= k:
            raise ValueError(
                f"class {class_name!r} has {len(candidates)} eligible rows; "
                f"needs more than {k} for a disjoint support/query partition"
            )
        ranked: list[tuple[str, int]] = []
        for position in candidates:
            payload = (
                release.SPLIT_SALT
                + "\0"
                + str(seed)
                + "\0"
                + dev_row_identity(items[position], identity_fields)
            ).encode("utf-8")
            ranked.append((hashlib.sha256(payload).hexdigest(), position))
        ranked.sort(key=lambda pair: (pair[0], pair[1]))
        support_by_class[class_name] = [
            position for _digest, position in ranked[:k]
        ]
        digest_by_class[class_name] = [digest for digest, _p in ranked[:k]]
        all_support.extend(support_by_class[class_name])

    support = np.asarray(sorted(all_support), dtype=np.int64)
    if len(set(support.tolist())) != len(support):
        raise AssertionError("support positions are not unique")
    report = {
        "version": release.SPLIT_VERSION,
        "salt": release.SPLIT_SALT,
        "split_seed": seed,
        "k_per_class": k,
        "identity_fields": list(identity_fields),
        "absent_release_identity_fields": list(
            EXPECTED_ABSENT_IDENTITY_FIELDS
        ),
        "support_positions_by_class": support_by_class,
        "support_digest_by_class": digest_by_class,
        "support_positions_sha256": hashlib.sha256(
            support.astype("<i8", copy=False).tobytes()
        ).hexdigest(),
    }
    return support, report


# ---------------------------------------------------------------------------
# loading the frozen fusion artifact
# ---------------------------------------------------------------------------


def load_fusion_artifact(directory: Path) -> dict[str, Any]:
    """Load a v3 fusion directory and rebuild its module from frozen assets.

    Every asset is verified against the SHA-256 the artifact recorded before it
    is used, and the rebuilt module's configuration must equal the recorded one.
    """
    source = assemble.reject_sealed_path(Path(directory), "source")
    if not source.is_dir():
        raise FileNotFoundError(source)

    metrics_path = source / "dev_metrics.json"
    with metrics_path.open(encoding="utf-8") as handle:
        metrics = json.load(handle)

    if str(metrics.get("encoder")) != "fusion":
        raise ValueError(
            f"{source} holds encoder {metrics.get('encoder')!r}, expected 'fusion'"
        )
    for key, expected in (
        ("development_only", True),
        ("release_evidence", False),
        ("sealed_release_data_used", 0),
        ("consumed_test_rows_used", 0),
    ):
        if metrics.get(key) != expected:
            raise RuntimeError(
                f"{source} declares {key}={metrics.get(key)!r}, expected {expected!r}"
            )
    if metrics.get("data_audit", {}).get("consumed_test_rows_exposed") != 0:
        raise RuntimeError(f"{source} provenance exposed consumed test rows")

    frontend = metrics.get("frontend", {})
    if frontend.get("version") != td_preprocess.PREPROCESS_VERSION:
        raise RuntimeError(
            f"{source} was built on frontend {frontend.get('version')!r}, "
            f"expected {td_preprocess.PREPROCESS_VERSION!r}"
        )
    if frontend.get("uses_frequency_transform") is not False:
        raise RuntimeError(f"{source} does not declare an FFT-free frontend")

    artifacts = metrics["artifacts"]
    paths = {
        name: source / str(artifacts[name])
        for name in (
            "state_dict",
            "prototypes",
            "real_center",
            "complex_center",
            "feature_mean",
            "feature_std",
        )
    }
    hashes: dict[str, str] = {"dev_metrics.json": dev._sha256(metrics_path)}
    for name, path in paths.items():
        digest = dev._sha256(path)
        if digest != artifacts[f"{name}_sha256"]:
            raise RuntimeError(f"{path} does not match its recorded SHA-256")
        hashes[path.name] = digest

    architecture = metrics["architecture"]
    if architecture.get("kind") != "centered_invariant_fusion":
        raise ValueError(f"{source} is not a centered invariant fusion")
    real_net = InvariantPatchCNN(
        InvariantPatchConfig(**architecture["real"]).validate()
    )
    complex_net = InvariantPatchCNN(
        InvariantPatchConfig(**architecture["complex"]).validate()
    )
    real_center = np.load(paths["real_center"]).astype(np.float32)
    complex_center = np.load(paths["complex_center"]).astype(np.float32)
    fusion = CenteredInvariantFusion(
        real_net.eval(),
        complex_net.eval(),
        real_center,
        complex_center,
        alpha_real=float(architecture["alpha_real"]),
        alpha_complex=float(architecture["alpha_complex"]),
        weight_real=float(architecture["weight_real"]),
        eps=float(architecture["eps"]),
    )
    try:
        state = torch.load(paths["state_dict"], map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with old PyTorch
        state = torch.load(paths["state_dict"], map_location="cpu")
    fusion.load_state_dict(state, strict=True)
    fusion.eval()

    # The state dict carries the centers as buffers.  They must agree with the
    # separately stored center arrays, or the artifact is internally inconsistent.
    np.testing.assert_array_equal(
        fusion.real_center.detach().cpu().numpy(), real_center
    )
    np.testing.assert_array_equal(
        fusion.complex_center.detach().cpu().numpy(), complex_center
    )
    rebuilt = fusion.config()
    for field in ("alpha_real", "alpha_complex", "weight_real", "eps", "embed_dim"):
        if rebuilt[field] != architecture[field]:
            raise RuntimeError(
                f"rebuilt fusion {field} {rebuilt[field]!r} differs from the "
                f"recorded {architecture[field]!r}"
            )

    return {
        "directory": source,
        "metrics": metrics,
        "fusion": fusion,
        "prototypes": np.load(paths["prototypes"]).astype(np.float32),
        "feature_mean": np.load(paths["feature_mean"]).astype(np.float32),
        "feature_std": np.load(paths["feature_std"]).astype(np.float32),
        "sha256": hashes,
    }


def frozen_inference_assets(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Build the minimal ``data``-shaped mapping ``dev._embed_raw`` reads.

    Feature standardization is loaded verbatim from the artifact, exactly as the
    release evaluator loads ``feature_mean``/``feature_std`` from the bundle. It
    is never recomputed here, so no development population can influence it.
    """
    contract = artifact["metrics"]["source_index_contract"]["contract"]
    return {
        "fmean": np.asarray(artifact["feature_mean"], dtype=np.float32),
        "fstd": np.asarray(artifact["feature_std"], dtype=np.float32),
        "cache_contract": contract,
    }


# ---------------------------------------------------------------------------
# eligible populations
# ---------------------------------------------------------------------------


def eligible_population(
    manifest: Mapping[str, Any],
    corpus_indices: np.ndarray,
    classes: Sequence[str],
    raw: np.memmap,
) -> dict[str, Any]:
    """Apply ``run_time_domain_dev``'s all-zero-N4096-prefix exclusion.

    Returns the surviving corpus indices, their longest captures, labels,
    impaired mask, SNR, manifest rows, and per-class included/excluded counts.
    """
    minimum_length = min(DEV_LENGTHS)
    kept: list[int] = []
    captures: list[np.ndarray] = []
    included = {name: 0 for name in classes}
    excluded = {name: 0 for name in classes}
    for index in np.asarray(corpus_indices, dtype=np.int64):
        capture = dev._complex_row(raw, int(index))
        class_name = str(manifest["items"][int(index)]["cls"])
        if float(np.max(np.abs(capture[:minimum_length]))) == 0.0:
            excluded[class_name] += 1
            continue
        kept.append(int(index))
        captures.append(capture)
        included[class_name] += 1
    if not kept:
        raise ValueError("no rows survived the all-zero N4096 prefix rule")
    indices = np.asarray(kept, dtype=np.int64)
    labels, impaired, snr = dev._metadata_arrays(manifest, indices, classes)
    return {
        "indices": indices,
        "captures": captures,
        "labels": labels,
        "impaired": impaired,
        "snr_db": snr,
        "items": [manifest["items"][int(index)] for index in indices],
        "included_by_class": included,
        "excluded_all_zero_n4096_prefix_by_class": excluded,
        "indices_sha256": hashlib.sha256(
            indices.astype("<i8", copy=False).tobytes()
        ).hexdigest(),
    }


def embed_at_length(
    embedder: torch.nn.Module,
    captures: Sequence[np.ndarray],
    length: int,
    frozen: Mapping[str, Any],
    device: torch.device,
) -> np.ndarray:
    """Embed exact causal prefixes through the frozen fusion."""
    if length <= 0:
        raise ValueError("prefix length must be positive")
    prefixes = [capture[:length] for capture in captures]
    if any(len(prefix) != length for prefix in prefixes):
        raise ValueError("a capture is shorter than the requested prefix")
    embeddings, _contexts = dev._embed_raw(
        embedder, prefixes, dict(frozen), device
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if not np.isfinite(embeddings).all():
        raise RuntimeError(f"frozen fusion produced non-finite n{length} embeddings")
    return embeddings


# ---------------------------------------------------------------------------
# gate 1: clean accuracy per capture length
# ---------------------------------------------------------------------------


def closed_clean_report(
    embeddings: np.ndarray,
    labels: np.ndarray,
    impaired: np.ndarray,
    prototypes: np.ndarray,
    classes: Sequence[str],
) -> dict[str, Any]:
    """Reproduce the release ``_closed_report`` fields the clean gate reads.

    ``clean_accuracy`` here is bit-for-bit the quantity
    ``closed_clean_worst_length`` minimises over: plain accuracy over
    ``~impaired``, frozen prototypes, Euclidean nearest.  The balanced figures
    are additional, because the gate metric is population-weighted and the
    development population is not class-balanced.
    """
    impaired_mask = np.asarray(impaired, dtype=np.bool_)
    label_array = np.asarray(labels, dtype=np.int64)
    if impaired_mask.shape != label_array.shape:
        raise ValueError("impaired mask must match the label vector")
    prediction, _distance = release._nearest(embeddings, prototypes)
    clean = release._classification_subset(
        prediction, label_array, ~impaired_mask, classes, require_all_classes=True
    )
    impaired_report = release._classification_subset(
        prediction, label_array, impaired_mask, classes, require_all_classes=True
    )
    recalls = {
        name: float(np.mean(prediction[label_array == index] == index))
        for index, name in enumerate(classes)
    }
    return {
        "n": int(len(label_array)),
        "accuracy": float(np.mean(prediction == label_array)),
        "balanced_accuracy": float(np.mean(list(recalls.values()))),
        "per_class_recall": recalls,
        # The gate value.
        "clean_accuracy": clean["accuracy"],
        # Reported beside it because the clean subset is class-imbalanced here.
        "clean_balanced_accuracy": clean["balanced_accuracy_present_classes"],
        "clean_subset": clean,
        "impaired_accuracy": impaired_report["accuracy"],
        "impaired_balanced_accuracy": impaired_report[
            "balanced_accuracy_present_classes"
        ],
        "impaired_subset": impaired_report,
        "clean_definition": (
            "manifest impaired == false on the observed stream; identical to "
            "evaluate_invariant_release_suite._closed_report, and not the "
            "corpus clean_path companion stream"
        ),
    }


# ---------------------------------------------------------------------------
# gate 2: five-shot balanced accuracy per capture length
# ---------------------------------------------------------------------------


def five_shot_variant(
    support_matched: np.ndarray,
    support_labels: np.ndarray,
    query_by_length: Mapping[int, np.ndarray],
    query_labels: np.ndarray,
    classes: Sequence[str],
) -> dict[str, Any]:
    """Run ``release._five_shot_report`` at every length for one support draw.

    ``support_matched`` are the support embeddings at ``MATCHED_CAPTURE_LENGTH``.
    The evaluator's function indexes one enrollment array and one evaluation
    array with a shared index space, so support and query rows are concatenated
    into that space: support first, query after.  Only ``support_indices`` reads
    the enrollment array and only ``query_indices`` reads the evaluation array,
    so the padding halves are never scored.
    """
    support = np.asarray(support_matched, dtype=np.float32)
    support_label_array = np.asarray(support_labels, dtype=np.int64)
    query_label_array = np.asarray(query_labels, dtype=np.int64)
    if len(support) != len(support_label_array):
        raise ValueError("support embeddings and labels disagree in length")
    n_support = len(support)
    n_query = len(query_label_array)
    support_indices = np.arange(n_support, dtype=np.int64)
    query_indices = np.arange(n_support, n_support + n_query, dtype=np.int64)
    combined_labels = np.concatenate((support_label_array, query_label_array))

    per_length: dict[str, Any] = {}
    for length in sorted(query_by_length):
        query = np.asarray(query_by_length[length], dtype=np.float32)
        if len(query) != n_query:
            raise ValueError(f"n{length} query rows disagree with the label vector")
        # One shared index space: support rows carry their matched-length
        # embedding, query rows carry their embedding at this length. The
        # evaluator reads support rows only through ``support_indices`` and
        # query rows only through ``query_indices``, so one array serves both
        # of its arguments without either half leaking into the other's role.
        space = np.concatenate((support, query))
        report = release._five_shot_report(
            space,
            space,
            combined_labels,
            support_indices,
            query_indices,
            classes,
        )
        per_length[str(length)] = report
    return {
        "per_length": per_length,
        "worst_length_balanced_accuracy": min(
            float(row["balanced_accuracy"]) for row in per_length.values()
        ),
        "worst_length_accuracy": min(
            float(row["accuracy"]) for row in per_length.values()
        ),
    }


# ---------------------------------------------------------------------------
# reproduction check against the artifact's own length audit
# ---------------------------------------------------------------------------


def reproduction_check(
    metrics: Mapping[str, Any],
    measured: Mapping[str, Mapping[str, Any]],
    population_sha256: str,
    *,
    tolerance: float,
) -> dict[str, Any]:
    """Re-derive the artifact's recorded per-length numbers as a sanity check.

    Fails closed on a population mismatch: if the eligible row set is not the one
    the artifact scored, the clean numbers are not comparable to its length audit
    and nothing downstream should be trusted.  Metric drift is recorded but only
    raises above ``tolerance``, because device reduction order legitimately moves
    the fourth decimal.
    """
    audit = metrics.get("population_causal_length", {})
    recorded_sha = audit.get("eligible_indices_sha256")
    if recorded_sha != population_sha256:
        raise RuntimeError(
            "eligible selection population does not match the artifact's "
            f"recorded length audit: {population_sha256} != {recorded_sha}"
        )
    deltas: dict[str, Any] = {}
    worst = 0.0
    for row in audit.get("rows", []):
        length = str(int(row["length"]))
        if length not in measured:
            continue
        pair = {
            key: abs(float(row[key]) - float(measured[length][key]))
            for key in ("accuracy", "balanced_accuracy")
        }
        deltas[length] = pair
        worst = max(worst, max(pair.values()))
    if not deltas:
        raise RuntimeError("artifact records no comparable length-audit rows")
    if worst > tolerance:
        raise RuntimeError(
            f"re-derived length audit differs by {worst:g} > {tolerance:g}; "
            "the frozen assets or the population are not the recorded ones"
        )
    return {
        "eligible_indices_sha256": population_sha256,
        "matches_artifact_population": True,
        "absolute_deltas_by_length": deltas,
        "worst_absolute_delta": float(worst),
        "tolerance": float(tolerance),
        "role": (
            "confirms the frozen assets, population, and frontend reproduce the "
            "artifact's own recorded closed numbers before any new gate is read"
        ),
    }


# ---------------------------------------------------------------------------
# gate assembly
# ---------------------------------------------------------------------------


def assemble_gates(
    closed_by_length: Mapping[str, Mapping[str, Any]],
    five_shot_by_variant: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Assemble the two remaining closed-set gates at the release thresholds.

    Thresholds come from ``release.GATE_FLOORS`` and are never restated here, so
    they cannot be softened by editing this module.
    """
    if not closed_by_length:
        raise ValueError("clean gate needs at least one length")
    missing = [name for name in FIVE_SHOT_VARIANTS if name not in five_shot_by_variant]
    if missing:
        raise ValueError(f"five-shot variants are missing: {missing}")

    minimum_length = str(min(DEV_LENGTHS))
    clean_floor = release.GATE_FLOORS["closed_clean"]
    five_floor = release.GATE_FLOORS["five_shot"]
    worst_clean = min(
        float(row["clean_accuracy"]) for row in closed_by_length.values()
    )
    per_variant_worst = {
        name: float(five_shot_by_variant[name]["worst_length_balanced_accuracy"])
        for name in FIVE_SHOT_VARIANTS
    }
    gates: dict[str, Any] = {
        f"closed_clean_n{minimum_length}": release._gate(
            float(closed_by_length[minimum_length]["clean_accuracy"]), clean_floor
        ),
        "closed_clean_worst_length": release._gate(worst_clean, clean_floor),
    }
    for name, value in per_variant_worst.items():
        gates[f"five_shot_worst_length_balanced_{name}"] = release._gate(
            value, five_floor
        )
    gates["five_shot_worst_length_balanced_conservative"] = release._gate(
        min(per_variant_worst.values()), five_floor
    )
    gates["five_shot_worst_length_balanced_primary"] = release._gate(
        per_variant_worst[PRIMARY_FIVE_SHOT_VARIANT], five_floor
    )
    return gates


def _source_hashes() -> dict[str, str]:
    paths = {
        "measure_v3_remaining_gates.py": Path(__file__).resolve(),
        "assemble_v3_fusion.py": HERE / "assemble_v3_fusion.py",
        "run_time_domain_dev.py": HERE / "run_time_domain_dev.py",
        "evaluate_invariant_release_suite.py": (
            V2 / "evaluate_invariant_release_suite.py"
        ),
        "invariant_fusion.py": V2 / "invariant_fusion.py",
        "invariant_patch_cnn.py": V2 / "invariant_patch_cnn.py",
        "invariant_patch_data.py": V2 / "invariant_patch_data.py",
        "time_domain_geometry.py": TRAINING / "time_domain_geometry.py",
        "time_domain_invariant_patch_preprocess.py": (
            TRAINING / "time_domain_invariant_patch_preprocess.py"
        ),
        "train.py": TRAINING / "train.py",
    }
    return {name: dev._sha256(path) for name, path in paths.items()}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    fusion_dir = assemble.reject_sealed_path(Path(args.fusion_dir), "source")
    output_dir = assemble.reject_sealed_path(
        Path(args.output_dir) if args.output_dir else fusion_dir, "output"
    )
    tolerance = validate_tolerance(args.reproduction_tolerance)

    artifact = load_fusion_artifact(fusion_dir)
    metrics = artifact["metrics"]
    split_seed = validate_split_seed(
        args.split_seed if args.split_seed is not None else metrics["seed"]
    )

    report_path = output_dir / REPORT_NAME
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path}")

    architecture = metrics["architecture"]["real"]
    contract = metrics["source_index_contract"]["contract"]
    target_frac = float(contract["config"]["target_frac"])
    source = invariant_data.load(
        patch_length=int(architecture["patch_length"]),
        patch_count=int(architecture["patch_count"]),
        target_frac=target_frac,
        model_seed=int(metrics["seed"]),
        build_if_missing=False,
    )
    if source["data_audit"].get("consumed_test_rows_exposed") != 0:
        raise AssertionError("data loader exposed consumed test rows")
    forbidden = [key for key in source if "test" in key.lower()]
    if forbidden:
        raise AssertionError(f"data loader exposed test-like keys: {forbidden}")
    if source["cache_contract"] != contract:
        raise RuntimeError("current cache contract differs from the fusion artifact")

    classes = list(source["classes"])
    manifest = dev._load_manifest()
    identity_fields = dev_row_identity_fields(manifest["items"])

    device = runner.resolve_device(args.device)
    runner.seed_everything(int(metrics["seed"]))
    embedder = assemble.FusionEmbedder(artifact["fusion"]).to(device).eval()
    frozen = frozen_inference_assets(artifact)
    prototypes = artifact["prototypes"]

    raw = dev._raw_memmap(manifest)
    selection = eligible_population(
        manifest, np.asarray(source["va_idx"], dtype=np.int64), classes, raw
    )
    enrollment = eligible_population(
        manifest, np.asarray(source["en_idx"], dtype=np.int64), classes, raw
    )
    del raw

    # --- gate 1 -----------------------------------------------------------
    selection_by_length: dict[int, np.ndarray] = {}
    closed_by_length: dict[str, Any] = {}
    for length in DEV_LENGTHS:
        embeddings = embed_at_length(
            embedder, selection["captures"], length, frozen, device
        )
        selection_by_length[length] = embeddings
        closed_by_length[str(length)] = closed_clean_report(
            embeddings,
            selection["labels"],
            selection["impaired"],
            prototypes,
            classes,
        )
        print(
            f"[v3 gates] n{length}: clean={closed_by_length[str(length)]['clean_accuracy']:.4f} "
            f"clean_bal={closed_by_length[str(length)]['clean_balanced_accuracy']:.4f}",
            flush=True,
        )

    checks = reproduction_check(
        metrics,
        closed_by_length,
        selection["indices_sha256"],
        tolerance=tolerance,
    )

    # --- gate 2 -----------------------------------------------------------
    matched = release.MATCHED_CAPTURE_LENGTH
    if matched not in selection_by_length:
        raise RuntimeError(
            f"matched capture length {matched} was not measured; the release "
            "five-shot protocol embeds support there"
        )

    enrollment_support, enrollment_split = dev_five_shot_split(
        enrollment["items"],
        enrollment["labels"],
        classes,
        split_seed,
        identity_fields=identity_fields,
    )
    enrollment_support_matched = embed_at_length(
        embedder,
        [enrollment["captures"][int(p)] for p in enrollment_support],
        matched,
        frozen,
        device,
    )
    selection_support, selection_split = dev_five_shot_split(
        selection["items"],
        selection["labels"],
        classes,
        split_seed,
        identity_fields=identity_fields,
    )
    support_mask = np.zeros(len(selection["labels"]), dtype=bool)
    support_mask[selection_support] = True
    selection_query = np.flatnonzero(~support_mask)

    five_shot_by_variant = {
        "enrollment_support": {
            "role": (
                "primary; evidence-rule compliant. Support is enrollment, query "
                "is the whole eligible selection population, disjoint by "
                "population separation rather than by exclusion."
            ),
            "support_population": "enrollment",
            "query_population": "selection",
            "support_rows": int(len(enrollment_support)),
            "query_rows": int(len(selection["labels"])),
            "split": enrollment_split,
            **five_shot_variant(
                enrollment_support_matched,
                enrollment["labels"][enrollment_support],
                selection_by_length,
                selection["labels"],
                classes,
            ),
        },
        "selection_support": {
            "role": (
                "diagnostic only; structural mirror of the sealed within-corpus "
                "partition. This variant fits 35 prototypes on selection rows "
                "and must not be used to select or tune anything."
            ),
            "support_population": "selection",
            "query_population": "selection minus support",
            "support_rows": int(len(selection_support)),
            "query_rows": int(len(selection_query)),
            "split": selection_split,
            **five_shot_variant(
                selection_by_length[matched][selection_support],
                selection["labels"][selection_support],
                {
                    length: value[selection_query]
                    for length, value in selection_by_length.items()
                },
                selection["labels"][selection_query],
                classes,
            ),
        },
    }
    for name, variant in five_shot_by_variant.items():
        print(
            f"[v3 gates] five-shot {name}: worst bal="
            f"{variant['worst_length_balanced_accuracy']:.4f}",
            flush=True,
        )

    gates = assemble_gates(closed_by_length, five_shot_by_variant)

    result = {
        "status": "complete",
        "measurement": MEASUREMENT_VERSION,
        "development_only": True,
        "release_evidence": False,
        "sealed_release_data_used": 0,
        "consumed_test_rows_used": 0,
        "retraining_performed": False,
        "recalibration_performed": False,
        "fusion_artifact": {
            "directory": str(fusion_dir),
            "sha256": artifact["sha256"],
            "seed": int(metrics["seed"]),
            "encoder": "fusion",
            "weight_real": float(metrics["architecture"]["weight_real"]),
            "parameter_count": int(metrics["parameter_count"]),
            "recorded_closed_selection_balanced_accuracy": float(
                metrics["closed_selection"]["balanced_accuracy"]
            ),
            "recorded_closed_selection_clean_accuracy": float(
                metrics["closed_selection"]["clean_accuracy"]
            ),
        },
        "frontend": td_preprocess.preprocess_metadata(),
        "protocol": {
            "mirrors": "evaluate_invariant_release_suite.py",
            "evaluation_version": release.EVALUATION_VERSION,
            "matched_capture_length": matched,
            "measured_capture_lengths": list(DEV_LENGTHS),
            "release_required_capture_lengths": list(
                release.REQUIRED_CAPTURE_LENGTHS
            ),
            "unmeasurable_release_capture_lengths": list(
                UNMEASURABLE_RELEASE_LENGTHS
            ),
            "five_shot": dict(release.EXPECTED_EVALUATION_PROTOCOL["five_shot"]),
            "split_seed": split_seed,
            "split_seed_source": (
                "explicit --split-seed"
                if args.split_seed is not None
                else "fusion artifact model seed"
            ),
            "imported_from_evaluator": [
                "FIVE_SHOT_K",
                "SPLIT_SALT",
                "SPLIT_VERSION",
                "MATCHED_CAPTURE_LENGTH",
                "ROW_IDENTITY_FIELDS",
                "GATE_FLOORS",
                "_nearest",
                "_classification_subset",
                "_five_shot_report",
                "_gate",
            ],
            "deviations_from_sealed_protocol": [
                "n32768 is unmeasurable: the historical development corpus "
                "stores sampleCount=16384 and a longer observation may not be "
                "synthesized; worst-over-length covers 3 of the 4 sealed lengths",
                "rows whose N4096 prefix is exactly all-zero are excluded by the "
                "run_time_domain_dev pre-inference rule; the sealed generator's "
                "predeclared start probe makes such rows impossible there",
                "the development populations are not class-balanced, while the "
                "sealed corpus holds exactly MIN_TARGET_PER_CLASS rows per class; "
                "clean_accuracy is population-weighted in both",
                "the primary five-shot variant draws support from enrollment "
                "rather than from the scored population, because five-shot "
                "support is a prototype fit and the selection population is "
                "scored, never fit; the selection_support variant mirrors the "
                "sealed within-corpus partition and is diagnostic only",
                "the digest split hashes 14 of the 15 release row-identity "
                "fields; impairmentSeed is absent from the development manifest",
                "the split payload binds a development seed, not a release seed",
            ],
        },
        "population": {
            "selection": {
                "role": "scored for both gates; never fit",
                "corpus_rows": int(len(source["va_idx"])),
                "eligible_rows": int(len(selection["labels"])),
                "eligible_by_class": selection["included_by_class"],
                "excluded_all_zero_n4096_prefix_by_class": selection[
                    "excluded_all_zero_n4096_prefix_by_class"
                ],
                "eligible_indices_sha256": selection["indices_sha256"],
                "clean_rows": int(np.sum(~selection["impaired"])),
                "impaired_rows": int(np.sum(selection["impaired"])),
            },
            "enrollment": {
                "role": "supplies the primary five-shot support draw only",
                "corpus_rows": int(len(source["en_idx"])),
                "eligible_rows": int(len(enrollment["labels"])),
                "eligible_by_class": enrollment["included_by_class"],
                "excluded_all_zero_n4096_prefix_by_class": enrollment[
                    "excluded_all_zero_n4096_prefix_by_class"
                ],
                "eligible_indices_sha256": enrollment["indices_sha256"],
            },
            "training": {
                "role": (
                    "not loaded here; feature standardization and branch centers "
                    "are read verbatim from the frozen fusion artifact"
                ),
                "rows_scored": 0,
            },
        },
        "closed_clean_per_length": closed_by_length,
        "five_shot_per_variant": five_shot_by_variant,
        "gates": gates,
        "all_measured_gates_pass": all(
            bool(value["passes"]) for value in gates.values()
        ),
        "reproduction_check": checks,
        "frozen_assets_used": [
            "fusion state dict (both branches, centers, alphas, weight, eps)",
            "fused prototypes",
            "feature mean/std",
        ],
        "source_sha256": _source_hashes(),
        "data_audit": source["data_audit"],
        "source_index_contract": {
            "contract": contract,
            "role": (
                "identifies the raw corpus and exposed train/enroll/selection "
                "indices only; every tensor scored here was rebuilt with the "
                "time-domain v3 frontend"
            ),
        },
        "device": str(device),
        "seed": int(metrics["seed"]),
        "wall_clock_s": time.perf_counter() - started,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    dev._write_json(report_path, result)
    print(f"[v3 gates] wrote {report_path}", flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--fusion-dir",
        required=True,
        help="v3 fusion artifact directory produced by assemble_v3_fusion.py",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"where to write {REPORT_NAME}; defaults to --fusion-dir",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help=(
            "seed bound into the five-shot digest payload; defaults to the "
            "fusion artifact's model seed. Release seeds are refused."
        ),
    )
    parser.add_argument(
        "--reproduction-tolerance",
        type=float,
        default=DEFAULT_REPRODUCTION_TOLERANCE,
        help=(
            "maximum tolerated difference when re-deriving the artifact's own "
            "recorded per-length accuracy and balanced accuracy"
        ),
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
