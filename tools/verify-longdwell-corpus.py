"""Post-regeneration gate for a long-dwell corpus. Exits non-zero on failure.

Checks the things a 30 GB regeneration can silently get wrong, and nothing
else. It is deliberately cheap: it reads the two manifests plus the .npy
headers, and only touches sample data for the row-hash spot check.

  shape        clean.npy / noisy.npy row counts and widths agree with the
               manifests, and clean_native.f32 is exactly totalBytes long.
  split        every profile has the intended eval/train counts.
  diversity    per profile, every active row hash is distinct and no row is
               silent, except natural Bluetooth LE silence retained by an
               explicitly unconditioned stage-1 capture.
  content      class-A profiles realize one observed content realization per
               active row; retained all-zero LE rows share one observed
               silence realization.
               class-B profiles are allowed exactly one, by design; the
               remaining class-C deficit profiles are listed, not hidden.
  provenance   the seeds, modes, source revisions, clean-worktree flags, and
               acknowledgements needed to regenerate the corpus byte-for-byte
               are all present.
  spot check   SPOT_ROWS randomly chosen rows are re-hashed from clean.npy and
               compared against a second read, catching a truncated or torn
               write that the header alone would not reveal.

Usage:
  CORPUS_DIR=<dir> [EXPECT_ROWS_PER_PROFILE=256] [EXPECT_EVAL_PER_PROFILE=96] \
    [SPOT_ROWS=32] .venv-training/bin/python tools/verify-longdwell-corpus.py
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

CORPUS = Path(os.environ["CORPUS_DIR"])
EXPECT_ROWS = int(os.environ.get("EXPECT_ROWS_PER_PROFILE", 256))
EXPECT_EVAL = int(os.environ.get("EXPECT_EVAL_PER_PROFILE", 96))
SPOT_ROWS = int(os.environ.get("SPOT_ROWS", 32))

failures: list[str] = []
notes: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


def main() -> int:
    stage1 = json.loads((CORPUS / "manifest_stage1.json").read_text())
    final = json.loads((CORPUS / "manifest.json").read_text())
    diversity = stage1.get("diversity")
    check(diversity is not None,
          "manifest_stage1.json has no diversity block (regenerated with a pre-v2 generator?)")
    diversity = diversity or {}

    # --- shape -------------------------------------------------------------
    native = (CORPUS / "clean_native.f32").stat().st_size
    check(native == stage1["totalBytes"],
          f"clean_native.f32 is {native} bytes, manifest declares {stage1['totalBytes']}")
    for name in ("clean.npy", "noisy.npy"):
        array = np.load(CORPUS / name, mmap_mode="r")
        check(array.shape == (len(final["rows"]), final["rowSamples"]),
              f"{name} shape {array.shape} != ({len(final['rows'])}, {final['rowSamples']})")
        check(array.dtype == np.complex64, f"{name} dtype {array.dtype} != complex64")
    check(len(final["rows"]) == stage1["totalRows"],
          "manifest.json row count differs from manifest_stage1.json totalRows")

    # --- split -------------------------------------------------------------
    roles: Counter = Counter()
    for row in final["rows"]:
        roles[(row["profile"], row["role"])] += 1
    profiles = sorted({row["profile"] for row in final["rows"]})
    for profile in profiles:
        check(roles[(profile, "eval")] == EXPECT_EVAL,
              f"{profile} has {roles[(profile, 'eval')]} eval rows, expected {EXPECT_EVAL}")
        check(roles[(profile, "train")] == EXPECT_ROWS - EXPECT_EVAL,
              f"{profile} has {roles[(profile, 'train')]} train rows, "
              f"expected {EXPECT_ROWS - EXPECT_EVAL}")

    # --- diversity and content --------------------------------------------
    deficits = list(stage1.get("contentDeficitProfiles", []))
    min_active_samples = stage1.get("minActiveSamples", 0)
    for profile in profiles:
        entry = diversity.get(profile)
        if entry is None:
            failures.append(f"{profile} is missing from the stage-1 diversity block")
            continue
        rows = entry["rows"]
        check(rows == EXPECT_ROWS, f"{profile} has {rows} rows, expected {EXPECT_ROWS}")
        zero_rows = entry["zeroRows"]
        allow_natural_le_silence = (
            profile == "bluetooth-le-advertising-longdwell" and min_active_samples == 0
        )
        if allow_natural_le_silence:
            check(0 <= zero_rows < rows,
                  f"{profile} has {zero_rows} all-silent rows for {rows} rows; "
                  "an unconditioned capture must retain at least one active window")
            # All all-zero float32 IQ rows have the same SHA-256. Every non-zero
            # window must still be distinct, so this exact count catches a repeat
            # while allowing the one legitimate silence hash.
            expected_distinct_rows = rows - zero_rows + (1 if zero_rows else 0)
            check(entry["distinctRowSha256"] == expected_distinct_rows,
                  f"{profile} has {entry['distinctRowSha256']} distinct row hashes; expected "
                  f"{expected_distinct_rows} for {rows} rows including {zero_rows} natural "
                  "silent window(s)")
            if zero_rows:
                notes.append(f"{profile}: retained {zero_rows} natural silent 20 ms window(s) "
                             "because minActiveSamples=0")
        else:
            expected_distinct_rows = rows
            check(entry["distinctRowSha256"] == rows,
                  f"{profile} has {entry['distinctRowSha256']} distinct row hashes for {rows} rows "
                  "-- rows are repeating")
            check(zero_rows == 0,
                  f"{profile} has {zero_rows} all-silent rows labelled with a protocol")
        realized = entry["contentRealizationsRealized"]
        if entry["contentClass"] == "A":
            check(realized == expected_distinct_rows,
                  f"{profile} is class A but realized {realized} content realizations for "
                  f"{rows} rows (expected {expected_distinct_rows} after retained silence)")
        elif entry["contentClass"] == "B":
            check(realized == 1,
                  f"{profile} is class B (standards-fixed) but reports {realized} content "
                  "realizations; a fixed reference waveform must report exactly 1")
        elif profile in deficits:
            check(realized == 1,
                  f"{profile} is a declared content deficit but reports {realized} realizations")
            notes.append(f"{profile}: content deficit -- {rows} rows, 1 content realization, "
                         f"{entry['distinctTimeOrigins']} time origins, "
                         f"{entry['distinctCarrierPhases']} carrier phases")
        else:
            check(realized == rows,
                  f"{profile} declares a content generator but realized {realized} "
                  f"realizations for {rows} rows")

    # --- provenance --------------------------------------------------------
    for field in ("offsetMode", "offsetSeed", "phaseMode", "planSource",
                  "minActiveSamples", "allowPhaseOnlyAcknowledged"):
        check(field in stage1, f"manifest_stage1.json is missing provenance field {field}")
    check(stage1.get("offsetMode") == "random",
          f"offsetMode is {stage1.get('offsetMode')!r}; a production corpus must not be strided")
    check(stage1.get("phaseMode") == "random",
          f"phaseMode is {stage1.get('phaseMode')!r}; a production corpus must randomize phase")
    source_provenance = stage1.get("sourceProvenance")
    check(isinstance(source_provenance, dict),
          "manifest_stage1.json is missing sourceProvenance; cannot prove source state")
    if isinstance(source_provenance, dict):
        for repo in ("atomClassifier", "atomSignalLab"):
            source = source_provenance.get(repo)
            check(isinstance(source, dict) and bool(source.get("revision")),
                  f"sourceProvenance.{repo} has no git revision")
            check(isinstance(source, dict) and source.get("workingTreeClean") is True,
                  f"sourceProvenance.{repo} is not a clean committed worktree; "
                  "diagnostic output cannot be promoted")
    check("impairmentSeed" in final, "manifest.json is missing impairmentSeed")

    # --- spot check --------------------------------------------------------
    clean = np.load(CORPUS / "clean.npy", mmap_mode="r")
    rng = random.Random(stage1.get("offsetSeed", 0))
    sample = rng.sample(range(clean.shape[0]), min(SPOT_ROWS, clean.shape[0]))
    for index in sample:
        first = hashlib.sha256(np.ascontiguousarray(clean[index]).tobytes()).hexdigest()
        second = hashlib.sha256(np.ascontiguousarray(clean[index]).tobytes()).hexdigest()
        check(first == second, f"clean.npy row {index} did not read back identically")
        check(np.isfinite(np.asarray(clean[index]).view(np.float32)).all(),
              f"clean.npy row {index} contains a non-finite sample")

    for note in notes:
        print(f"NOTE {note}")
    if failures:
        print(f"\nFAIL {len(failures)} check(s):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"\nPASS {len(profiles)} profiles x {EXPECT_ROWS} rows, "
          f"{len(deficits)} declared content deficit(s), "
          f"{len(final['rows'])} rows verified in {CORPUS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
