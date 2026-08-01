/**
 * Long-dwell probe corpus, stage 1: stitched CLEAN captures at native rates.
 *
 * Purpose: the v7 attribution experiment needs duration-complete captures --
 * every row covers >= DURATION_MS of physical time, so burst structure (GSM
 * TDMA frames, Bluetooth hopping/advertising cadence, DSSS packet cadence) is
 * IN the data rather than absent from it. The 2026-07-24 corpus fixed 4096 ->
 * 16384 samples for exactly this reason; this generator completes the move by
 * fixing duration rather than sample count.
 *
 * Native-rate synthesis with block stitching (<= 65,536 samples per call,
 * verified byte-exact for every family used here). Rate normalization and the
 * clean/noisy impairment pairing happen in stage 2 (Python), which is where
 * the polyphase resampler lives.
 *
 * Row phase (2026-07-31): row start offsets are drawn from a seeded RNG, not
 * strided. The original `startSampleIndex = row * samplesPerRow` striding
 * aliased against every protocol's natural period: 20 ms is an exact integer
 * multiple of the LTE/NR 10 ms radio frame and of the Bluetooth BR 625 us
 * slot, and 26,000 native GSM samples share gcd 2,000 with the 6,000-sample
 * TDMA frame. The measured consequence on the 2026-07-31 production corpus:
 * 27 of 34 profiles realized fewer than 10 distinct burst phases across their
 * 96 eval rows (22 of them realized exactly ONE -- every eval row bit-identical),
 * and GSM realized exactly 3, which is why an offset-0 1 ms eval window scored
 * ~1/3 recall on GSM. Randomness everywhere is the rule of ML: the phase is now
 * a seeded draw, recorded per row so the corpus stays exactly reproducible.
 *
 * Output (git-ignored):
 *   <out>/clean_native.f32   concatenated variable-length cf32le rows
 *   <out>/manifest_stage1.json
 *
 * Run:  node tools/generate-longdwell-probe-corpus.mjs
 *
 * Offset knobs (all optional):
 *   OFFSET_MODE=random|stride   default random. `stride` reproduces the exact
 *                               pre-fix startSampleIndex = row * samplesPerRow.
 *   OFFSET_SEED=<uint32>        default 20260731. Per-profile substreams are
 *                               derived from (seed, profile name), so editing
 *                               the plan never shifts another profile's draws.
 *   OFFSET_SPACING=<int >= 1>   default 2. Each row owns a block of
 *                               SPACING * samplesPerRow native samples and is
 *                               placed uniformly inside the leading
 *                               (SPACING - 1) * samplesPerRow of it, so rows
 *                               stay strictly non-overlapping (no train/eval
 *                               leakage) while the phase is free over a
 *                               (SPACING - 1) * DURATION_MS window. SPACING=1
 *                               degenerates to stride (zero jitter room).
 *   DRY_RUN=1                   plan and print offsets only; synthesize nothing
 *                               and write nothing.
 *   DRY_RUN_PROFILE=<name>      profile to print in the dry run (default: first)
 *   DRY_RUN_ROWS=<n>            rows to print (default 10)
 */
import { closeSync, existsSync, mkdirSync, openSync, readFileSync, writeFileSync, writeSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, join, resolve } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const SIGNAL_LAB = resolve(HERE, '../../Atom-SignalLab');
const OUT_DIR = process.env.OUT_DIR
  ? resolve(process.env.OUT_DIR)
  : resolve(HERE, '../training/artifacts/longdwell-probe-corpus');
const DURATION_MS = Number(process.env.DURATION_MS ?? 20);
const ROWS_PER_PROFILE = Number(process.env.ROWS_PER_PROFILE ?? 360);
const MAX_CALL = 65_536;

const OFFSET_MODE = process.env.OFFSET_MODE ?? 'random';
const OFFSET_SEED = Number(process.env.OFFSET_SEED ?? 20_260_731);
const OFFSET_SPACING = Number(process.env.OFFSET_SPACING ?? 2);
const DRY_RUN = process.env.DRY_RUN === '1';
const DRY_RUN_ROWS = Number(process.env.DRY_RUN_ROWS ?? 10);

if (OFFSET_MODE !== 'random' && OFFSET_MODE !== 'stride') {
  throw new Error(`OFFSET_MODE must be 'random' or 'stride', got ${OFFSET_MODE}`);
}
if (!Number.isSafeInteger(OFFSET_SEED) || OFFSET_SEED < 0) {
  throw new Error('OFFSET_SEED must be a non-negative safe integer');
}
if (!Number.isSafeInteger(OFFSET_SPACING) || OFFSET_SPACING < 1) {
  throw new Error('OFFSET_SPACING must be an integer >= 1');
}

const MASK64 = (1n << 64n) - 1n;

/** FNV-1a 64-bit over UTF-8 -- stable across Node versions and platforms. */
function fnv1a64(text) {
  let hash = 0xcbf2_9ce4_8422_2325n;
  const bytes = new TextEncoder().encode(text);
  for (const byte of bytes) {
    hash = (hash ^ BigInt(byte)) * 0x0000_0100_0000_01b3n & MASK64;
  }
  return hash;
}

/**
 * splitmix64. Deterministic, self-contained, and independent per profile: the
 * substream seed is fnv1a64(`${OFFSET_SEED}|${profile}`), so a profile's draws
 * depend only on its own name and the corpus seed -- reordering the plan or
 * adding a profile leaves every other profile's offsets untouched.
 */
function createSplitmix64(seed) {
  let state = seed & MASK64;
  return () => {
    state = state + 0x9e37_79b9_7f4a_7c15n & MASK64;
    let z = state;
    z = (z ^ z >> 30n) * 0xbf58_476d_1ce4_e5b9n & MASK64;
    z = (z ^ z >> 27n) * 0x94d0_49bb_1331_11ebn & MASK64;
    return (z ^ z >> 31n) & MASK64;
  };
}

/** Unbiased uniform integer in [0, boundExclusive) via Lemire rejection. */
function nextBoundedBigInt(next, boundExclusive) {
  if (boundExclusive <= 1n) return 0n;
  const limit = MASK64 - MASK64 % boundExclusive;
  for (;;) {
    const draw = next();
    if (draw < limit) return draw % boundExclusive;
  }
}

/**
 * Row start offsets for one profile.
 *
 * stride: startSampleIndex = row * samplesPerRow (the pre-fix behaviour, kept
 *   bit-for-bit so an old corpus can be reproduced).
 * random: row r owns [r * SPACING * samplesPerRow, (r + 1) * SPACING * samplesPerRow)
 *   and starts at a uniform draw inside the leading (SPACING - 1) * samplesPerRow
 *   of that block. Rows therefore never overlap -- a train row can never share
 *   samples with an eval row -- while the phase modulo any protocol period up to
 *   (SPACING - 1) * DURATION_MS is uniform instead of locked.
 */
function planRowOffsets(profile, samplesPerRow, rows) {
  if (OFFSET_MODE === 'stride') {
    return Array.from({ length: rows }, (_unused, row) => ({
      startSampleIndex: row * samplesPerRow,
      offsetJitter: 0,
    }));
  }
  const substreamSeed = fnv1a64(`${OFFSET_SEED}|${profile}`);
  const next = createSplitmix64(substreamSeed);
  const jitterRange = BigInt((OFFSET_SPACING - 1) * samplesPerRow + 1);
  const blockSamples = OFFSET_SPACING * samplesPerRow;
  return Array.from({ length: rows }, (_unused, row) => {
    const jitter = Number(nextBoundedBigInt(next, jitterRange));
    const startSampleIndex = row * blockSamples + jitter;
    if (!Number.isSafeInteger(startSampleIndex + samplesPerRow)) {
      throw new Error(`${profile} row ${row} start index is not a safe integer`);
    }
    return { startSampleIndex, offsetJitter: jitter };
  });
}

function offsetPlanMeta(profile, samplesPerRow) {
  return {
    offsetMode: OFFSET_MODE,
    offsetSeed: OFFSET_SEED,
    offsetSpacing: OFFSET_MODE === 'random' ? OFFSET_SPACING : 1,
    offsetSubstreamSeed: OFFSET_MODE === 'random'
      ? fnv1a64(`${OFFSET_SEED}|${profile}`).toString()
      : null,
    offsetJitterRangeSamples: OFFSET_MODE === 'random'
      ? (OFFSET_SPACING - 1) * samplesPerRow
      : 0,
  };
}

// Loaded lazily: DRY_RUN only exercises the offset planner, which is pure
// arithmetic, so the dry run must not require the SignalLab TS loader.
const synthesizeAnalyticComplexIq = DRY_RUN
  ? null
  : (await import(pathToFileURL(join(SIGNAL_LAB, 'src/complex-iq.ts')).href))
    .synthesizeAnalyticComplexIq;

/**
 * Profile plan. Native rates come from the SignalLab binding registry /
 * catalog facts; analytic profiles synthesize directly at the target 20 Msps.
 * gsm-32qam-higher-symbol-rate-burst is included deliberately: it is the
 * historical worst cell (recall 0.2778 in the v5 study).
 */
const PLAN = process.env.PLAN_JSON
  ? JSON.parse(readFileSync(resolve(process.env.PLAN_JSON), 'utf8'))
  : [
    { profile: 'cw', cls: 'cw', fs: 20_000_000, bw: 2_300 },
    { profile: 'am', cls: 'am', fs: 20_000_000, bw: 30_000 },
    { profile: 'fm', cls: 'fm', fs: 20_000_000, bw: 200_000 },
    { profile: 'gsm-normal-burst', cls: 'gsm', fs: 1_300_000, bw: 200_000 },
    { profile: 'gsm-32qam-higher-symbol-rate-burst', cls: 'gsm', fs: 1_300_000, bw: 325_000 },
    { profile: 'bluetooth-classic-connected-longdwell', cls: 'bluetooth', fs: 80_000_000, bw: 79_000_000 },
    { profile: 'bluetooth-le-advertising-longdwell', cls: 'bluetooth', fs: 80_000_000, bw: 79_000_000 },
    { profile: 'wifi-hr-dsss-11m', cls: 'dsss', fs: 11_000_000, bw: 11_000_000 },
    { profile: 'lte-etm1.1', cls: 'ofdm', fs: 15_360_000, bw: 10_000_000 },
    { profile: 'wifi-ofdm-20m', cls: 'ofdm', fs: 20_000_000, bw: 20_000_000 },
  ];

// Offset plan first: it is pure arithmetic, so a dry run can print it (and a
// real run can preflight the highest start index against every generator's
// bound) without synthesizing a single sample.
const OFFSET_PLAN = new Map(PLAN.map((spec) => {
  const samplesPerRow = Math.ceil((DURATION_MS / 1_000) * spec.fs);
  return [spec.profile, {
    samplesPerRow,
    meta: offsetPlanMeta(spec.profile, samplesPerRow),
    offsets: planRowOffsets(spec.profile, samplesPerRow, ROWS_PER_PROFILE),
  }];
}));

if (DRY_RUN) {
  const profile = process.env.DRY_RUN_PROFILE ?? PLAN[0].profile;
  const plan = OFFSET_PLAN.get(profile);
  if (plan === undefined) throw new Error(`DRY_RUN_PROFILE ${profile} is not in the plan`);
  const shown = plan.offsets.slice(0, DRY_RUN_ROWS);
  console.log(JSON.stringify({
    dryRun: true,
    profile,
    durationMs: DURATION_MS,
    rowsPerProfile: ROWS_PER_PROFILE,
    samplesPerRow: plan.samplesPerRow,
    ...plan.meta,
    maxStartSampleIndexAcrossPlan: Math.max(...[...OFFSET_PLAN.values()]
      .map((p) => p.offsets[p.offsets.length - 1].startSampleIndex + p.samplesPerRow)),
    firstRows: shown.map((entry, row) => ({ row, ...entry })),
  }, null, 1));
  process.exit(0);
}

if (existsSync(join(OUT_DIR, 'manifest_stage1.json')) && process.env.ALLOW_OVERWRITE !== '1') {
  throw new Error(`${OUT_DIR} already exists; set ALLOW_OVERWRITE=1 to regenerate`);
}

// Preflight: randomized offsets reach further along each generator's timeline
// than striding did (by OFFSET_SPACING x). Probe one sample at the highest
// planned coordinate per profile so a bound rejection costs a second, not hours.
for (const spec of PLAN) {
  const plan = OFFSET_PLAN.get(spec.profile);
  const last = plan.offsets[plan.offsets.length - 1];
  const probeIndex = last.startSampleIndex + plan.samplesPerRow - 1;
  try {
    synthesizeAnalyticComplexIq({
      profile: spec.profile,
      sampleRateHz: spec.fs,
      bandwidthHz: spec.bw,
      sampleCount: 1,
      startSampleIndex: probeIndex,
    });
  } catch (cause) {
    throw new Error(`${spec.profile} rejects the highest planned start index `
      + `${probeIndex} (OFFSET_MODE=${OFFSET_MODE}, OFFSET_SPACING=${OFFSET_SPACING}); `
      + 'lower OFFSET_SPACING or ROWS_PER_PROFILE', { cause });
  }
}

mkdirSync(OUT_DIR, { recursive: true });

const fd = openSync(join(OUT_DIR, 'clean_native.f32'), 'w');
const items = [];
let byteOffset = 0;
const startedAt = Date.now();

for (const spec of PLAN) {
  const plan = OFFSET_PLAN.get(spec.profile);
  const { samplesPerRow } = plan;
  // Row starts come from the seeded offset plan above: a per-row random phase
  // (burst phase, hop pattern, event timing all vary independently) rather than
  // a fixed stride that aliases against the protocol's own period.
  for (let row = 0; row < ROWS_PER_PROFILE; row += 1) {
    const { startSampleIndex, offsetJitter } = plan.offsets[row];
    let written = 0;
    while (written < samplesPerRow) {
      const chunk = Math.min(MAX_CALL, samplesPerRow - written);
      const bytes = synthesizeAnalyticComplexIq({
        profile: spec.profile,
        sampleRateHz: spec.fs,
        bandwidthHz: spec.bw,
        sampleCount: chunk,
        startSampleIndex: startSampleIndex + written,
      });
      writeSync(fd, bytes);
      written += chunk;
    }
    items.push({
      profile: spec.profile,
      cls: spec.cls,
      sampleRateHz: spec.fs,
      bandwidthHz: spec.bw,
      startSampleIndex,
      sampleCount: samplesPerRow,
      byteOffset,
      byteLength: samplesPerRow * 8,
      // Recorded draw: startSampleIndex == row * offsetSpacing * sampleCount
      // + offsetJitter, and offsetJitter is draw number `row` of the profile's
      // splitmix64 substream. The row is reproducible from the seed alone.
      offsetJitter,
      offsetDrawIndex: row,
    });
    byteOffset += samplesPerRow * 8;
  }
  console.log(`${spec.profile}: ${ROWS_PER_PROFILE} rows x ${samplesPerRow} samples `
    + `(${(Date.now() - startedAt) / 1000 | 0}s elapsed)`);
}
closeSync(fd);

writeFileSync(join(OUT_DIR, 'manifest_stage1.json'), JSON.stringify({
  // Schema id deliberately unchanged: every field a stage-1 reader already
  // consumed (durationMs, rowsPerProfile, items[].startSampleIndex/sampleCount/
  // byteOffset) keeps its meaning. The offset block below is purely additive.
  schema: 'longdwell-probe-corpus-stage1-v1',
  durationMs: DURATION_MS,
  rowsPerProfile: ROWS_PER_PROFILE,
  totalRows: items.length,
  totalBytes: byteOffset,
  stitchMaxCallSamples: MAX_CALL,
  offsetMode: OFFSET_MODE,
  offsetSeed: OFFSET_SEED,
  offsetSpacing: OFFSET_MODE === 'random' ? OFFSET_SPACING : 1,
  offsetPlan: Object.fromEntries([...OFFSET_PLAN]
    .map(([profile, plan]) => [profile, plan.meta])),
  note: 'clean native-rate captures; stage 2 resamples to a common rate and '
    + 'applies the receiver impairment chain to produce the noisy pair. Row '
    + 'start offsets are seeded random draws (OFFSET_MODE=random) so no row '
    + 'phase is locked to a protocol period; OFFSET_MODE=stride reproduces the '
    + 'pre-2026-07-31 aliased striding.',
  items,
}, null, 1));
console.log(`wrote ${items.length} rows, ${(byteOffset / 1e9).toFixed(2)} GB to ${OUT_DIR}`);
