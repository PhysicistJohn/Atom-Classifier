# Corpus regeneration runbook

This runbook governs the v7/DACS replacement corpus. It exists to prevent a
multi-hour, 80 GB regeneration from being mistaken for proof that the data is
actually more diverse.

## Non-negotiable safety rules

- Generate into a new directory. The existing
  `training/artifacts/longdwell-production-corpus/` is the measured baseline;
  never overwrite or delete it.
- Do not use `git clean`, `git reset --hard`, or a checkout that discards the
  inherited SignalLab long-dwell work.
- Preserve the `g4_bf16.safetensors` checkpoint and its evidence sidecars.
- Run the proof slice before any production invocation.

## What the current generator proves

`tools/generate-longdwell-probe-corpus.mjs` now defaults to the committed
34-profile plan and 256 rows per profile (8,704 rows). Its seeded offset plan
prevents phase aliasing: every row gets a reproducible, non-overlapping native
start coordinate. All 17 operational-content profiles now have a corpus-only
seeded-content path. `ALLOW_PHASE_ONLY=1` remains an explicit diagnostic
escape hatch for a future unsupported profile; it is neither required nor
appropriate for the 34-profile proof or production invocation.

That is necessary, but it is not complete content diversity. The public
SignalLab synthesis API accepts only `startSampleIndex`; corpus-only generators
are deliberately separate so fixed catalog provenance remains intact.
Dedicated 8-row proofs exist for the GERAN, Wi-Fi, and operational LTE/NR
families. The remaining evidence step is the combined 34-profile × 8-row proof
slice below. The replacement production corpus remains prohibited until that
slice is inspected successfully.

The 17 operational-content profiles are:

- 7 GERAN profiles — seeded corpus-only path, including two exact
  libosmocore xCCH fixtures plus chunking and geometry tests.
- LTE Band 3 FDD and Band 38 TDD — seeded PDSCH-only corpus path.
- NR n3 FDD and n78 TDD — seeded PDSCH-only corpus path, preserving inactive
  TDD intervals.
- All 6 Wi-Fi profiles — seeded corpus-only path with fixed PHY geometry.

The 15 standards-fixed or analytic profiles may vary phase only. The two
Bluetooth long-dwell profiles have index-driven timing/channel diversity, but
their qualified packet payloads remain fixed and that limitation must stay in
the manifest and report. `MIN_ACTIVE_SAMPLES` defaults to `0`: a natural
silent Bluetooth LE row is retained, then surfaced in the report rather than
being silently redrawn away. A positive value creates a conditioned corpus and
must be stated explicitly in any experiment report.

## Required software gate

Before treating the complete proof slice as reproducible evidence, each
corpus-only path must have all of these properties:

1. The default catalog path remains byte-identical and continues to pass its
   independent-oracle suites.
2. Every variable-content row receives a deterministic `contentSeed`, recorded
   in the stage-1 and final manifests.
3. A fixed-phase probe for each seeded row produces eight distinct clean
   content hashes for every operational-content profile.
4. Split and whole-window synthesis remain byte-identical for the same seed,
   profile, and absolute coordinates.
5. The source revision containing the long-dwell profiles and variation path is
   committed before generating a corpus intended to be reproducible. A slice
   made from an uncommitted worktree is diagnostic evidence only and must not
   be promoted as a reproducible corpus.

The implementation boundary and per-family requirements are in
`docs/corpus-content-variation-design.md`.

## Proof slice

After the software gate passes for all 17 operational-content profiles,
generate a real 34-profile × 8-row slice:

```bash
cd /Users/johnelliott/PersonalGitHub/Atom-Classifier
OUT_DIR=training/artifacts/longdwell-corpus-v2-proof \
PLAN_JSON=tools/production_corpus_plan.json \
ROWS_PER_PROFILE=8 \
DURATION_MS=20 \
npm run generate:longdwell-probe-corpus
```

The native stage-1 slice is about 0.94 GB. Do not run stage 2 or training if
the content gate is absent, any expected distinct-content hash collides, or the
source revision has not been committed for a proof intended to be retained.
The stage-1 manifest records `sourceProvenance` for both Atom-Classifier and
Atom-SignalLab; both exact revisions must be present and both
`workingTreeClean` flags must be `true` before the proof is promotable.

Inspect the stage-1 evidence without loading the whole slice into memory:

```bash
CORPUS_DIR=training/artifacts/longdwell-corpus-v2-proof \
OUT_JSON=training/artifacts/longdwell-corpus-v2-proof/diversity-report.json \
node tools/inspect-longdwell-stage1.mjs
```

The inspector deliberately calls them raw hashes: different phases of one
cyclic waveform also hash differently. Archive its JSON result alongside the
proof manifest, then record separately:

- phase offsets and raw row hashes for all 34 profiles;
- content seeds and fixed-phase content hashes for the 17 variable-content
  profiles;
- the expected phase-only classification for the 15 standards-fixed/analytic
  profiles; and
- Bluetooth timing/hash diversity plus its fixed-payload disclosure. In
  particular, report (do not discard or fabricate around) fully silent
  Bluetooth LE rows: a 20 ms capture can legitimately miss a ~30 ms
  advertising-event cadence. The proof report must give the zero-row count
  and active-sample fraction for that profile.

## Production run

Only after the proof passes:

```bash
OUT_DIR=training/artifacts/longdwell-production-corpus-v2 \
PLAN_JSON=tools/production_corpus_plan.json \
ROWS_PER_PROFILE=256 \
DURATION_MS=20 \
npm run generate:longdwell-probe-corpus

CORPUS_DIR=training/artifacts/longdwell-production-corpus-v2 \
.venv-training/bin/python tools/longdwell_probe_stage2.py
```

Verify exactly 8,704 rows, 96 eval and 160 train rows per profile, paired
clean/noisy arrays, manifest provenance, and the diversity report. Then train
MLX bf16 with the established v7 configuration and compare to the corrected
section-4 baseline; do not compare against the invalid pre-aliasing numbers.

## Historical pre-content slice and production cost

Everything in this section was measured on 2026-08-01 from a pre-content
34-profile x 8-row x 20 ms diagnostic slice
(`ALLOW_PHASE_ONLY=1`, `MIN_ACTIVE_SAMPLES=1024`). It remains useful for
capacity planning only: it is not evidence for the current content-diversity
gate, and its positive active-sample threshold conditioned away natural silent
Bluetooth LE rows. The production figures are that run extrapolated x32 and
are labelled as extrapolations, not as measurements.

### Wall time

| stage | measured, 8 rows/profile | production, 256 rows/profile |
| --- | --- | --- |
| stage 1 synthesis | 16.4 s (16.6 s of per-profile work) | **~9 min** (531 s) |
| stage 2 resample + impair | 6.2 s | **~3.5 min** compute; budget 4-8 min for 55.7 GB of writes |
| verifier | ~3 s | ~2 min (manifest scan + 32-row spot check) |
| **total** | ~26 s | **~15 min; budget 25 min** |

The GERAN corpus-content generator dominates stage 1: the 7 GSM profiles are
12.0 s of the slice's 16.6 s, and `gsm-32qam-higher-symbol-rate-burst` alone
extrapolates to 1.8 min. Every other profile is I/O-shaped. Stage 2's slice
write was partly absorbed by the page cache, so its extrapolation is the least
reliable number here; treat 4-8 min as the honest range.

### Disk

| artifact | size |
| --- | --- |
| `clean_native.f32` | 29.97 GB |
| `clean.npy` | 27.85 GB |
| `noisy.npy` | 27.85 GB |
| manifests (stage 1 + final) | ~17 MB |
| **new corpus total** | **85.7 GB** |
| baseline corpus, preserved | 85.7 GB |
| **free space required** | **~86 GB** (172 GB with the baseline in place) |

Checked 2026-08-01: 480 GB free on `/System/Volumes/Data`. `clean_native.f32`
may be deleted after stage 2 AND the verifier both succeed, recovering 30 GB,
but only then: it is the only way to rerun stage 2 without redoing stage 1.

### How the baseline is preserved

`OUT_DIR` for the production run is
`training/artifacts/longdwell-production-corpus-v2`, a NEW directory. The
baseline `training/artifacts/longdwell-production-corpus/` is never named by
any command in this runbook except the read-only baseline measurement in
"Verification when it finishes". The generator additionally refuses to write
into any directory already holding a `manifest_stage1.json` unless
`ALLOW_OVERWRITE=1` is set; never set that flag against the baseline. Nothing
in this runbook deletes anything.

### Determinism gate

Run this only after the two source worktrees are clean and committed. It uses
new temporary directories and preserves every result for inspection; it does
not delete or overwrite a prior corpus. `MIN_ACTIVE_SAMPLES=0` is explicit so
the check covers the unconditioned Bluetooth LE policy used by production.

```bash
cd /Users/johnelliott/PersonalGitHub/Atom-Classifier
for seed in 20260731 20260731 987654321; do
  run_dir="$(mktemp -d /private/tmp/longdwell-determinism.XXXXXX)"
  OFFSET_SEED=$seed OUT_DIR="$run_dir" \
    ROWS_PER_PROFILE=8 PLAN_JSON=tools/production_corpus_plan.json \
    MIN_ACTIVE_SAMPLES=0 \
    npm run generate:longdwell-probe-corpus >"$run_dir/generator.log"
  shasum -a 256 "$run_dir/clean_native.f32"
done
```

The first two hashes must match and the third must differ. Compare
`manifest_stage1.json` after excluding `generatedAt` and `elapsedSeconds`;
all other fields must match for the same source state. Do not require every
raw row hash to be unique in this unconditioned gate: natural all-zero BLE
windows share one raw hash and are reported explicitly.

The previously recorded `44a42a40b790...` / `5865ab2d1fd5...` values came
from the historical conditioned, phase-only diagnostic and are not current
acceptance hashes. The historical stage-2 array comparison remains capacity
evidence only; rerun it after the clean-source stage-1 proof before treating
a new production corpus as reproducible.

## Verification when it finishes

```bash
cd /Users/johnelliott/PersonalGitHub/Atom-Classifier
C=training/artifacts/longdwell-production-corpus-v2

# 1. structural + diversity gate; exits non-zero on failure
CORPUS_DIR=$C .venv-training/bin/python tools/verify-longdwell-corpus.py

# 2. independent re-hash from disk, not from the generator's own accounting
CORPUS_DIR=$C OUT_JSON=$C/stage1-inspection.json \
  node tools/inspect-longdwell-stage1.mjs > /dev/null

# 3. phase-invariant diversity, the measure a bit-hash under-reports
CORPUS_DIR=$C ROWS=32 OUT_JSON=$C/diversity-new.json \
  .venv-training/bin/python tools/measure-corpus-diversity.py

# 4. the same measurement on the read-only baseline, for the before/after table
CORPUS_DIR=training/artifacts/longdwell-production-corpus ROWS=32 \
  OUT_JSON=$C/diversity-baseline.json \
  .venv-training/bin/python tools/measure-corpus-diversity.py
```

`tools/verify-longdwell-corpus.py` asserts: `clean.npy`/`noisy.npy` shape and
dtype agree with both manifests; `clean_native.f32` is exactly `totalBytes`;
96 eval and 160 train rows per profile; every active row hash is distinct;
only unconditioned Bluetooth LE may retain its one repeated all-zero hash;
class-A profiles realize one observed content realization per active row (plus
the one retained-silence realization when applicable); class-B and
declared-deficit profiles report exactly 1;
`offsetMode` and `phaseMode` are both `random`; the provenance fields needed to
reproduce the corpus are present; and a 32-row spot re-read of `clean.npy` is
stable and finite. Declared deficits are printed as `NOTE` lines, never
swallowed. Natural fully silent Bluetooth LE rows are permitted only when
reported, not hidden by redraws. The historical pre-content diagnostic slice
passed structurally with 10 declared deficits and 272 rows; it is not the
current content proof.

`tools/measure-corpus-diversity.py` reports, per profile: distinct SHA-256
(rows as stored), distinct burst-onset patterns (phase-invariant, sensitive to
time origin and scheduling), mean `|rho|` over row pairs, single-link cluster
count at `|rho| >= 0.999` — the honest realization count — and the raw and
phase-invariant pairwise relative RMS differences. The gap between the last two
is exactly the part of the row-to-row difference that is global carrier phase,
which is what made the baseline's FM rows look diverse (256 distinct hashes)
while being one waveform times a scalar (`|rho| = 1.00000` for every pair).

Do not train until step 1 exits 0. When reporting results, quote
`manifest_stage1.json.diversity` alongside the accuracy numbers: a profile that
still holds exactly one content realization measures impairment and
time-origin robustness, not content generalization.
