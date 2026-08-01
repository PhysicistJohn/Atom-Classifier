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
 * ---------------------------------------------------------------------------
 * ROW DIVERSITY (2026-08-01). Three independent seeded per-row draws, each
 * recorded in the manifest so the corpus documents its own diversity.
 *
 *   1. TIME ORIGIN. The pre-fix `startSampleIndex = row * samplesPerRow`
 *      striding aliased against every protocol's natural period: 20 ms is an
 *      exact integer multiple of the LTE/NR 10 ms radio frame at every native
 *      rate, so 17 of the 29 fixed-catalog profiles had rowSamples an exact
 *      integer multiple of nativePeriodSamples and every row landed on frame
 *      phase 0. Measured consequence on the 2026-07-31 production corpus:
 *      18 profiles emitted a bit-identical clean row in all 256 rows, and
 *      phase-invariantly 20 of 34 profiles held exactly ONE realization.
 *      Start offsets are now seeded uniform draws taken WITHOUT REPLACEMENT
 *      inside each profile's realization space.
 *
 *   2. CARRIER PHASE. A seeded per-row rotation by e^(j*theta). This is what
 *      makes `cw` -- literally the constant 1+0j, for which a time origin
 *      carries no information at all -- produce distinct rows.
 *
 *   3. CONTENT. Only where a knob exists. See CONTENT_CLASS below.
 *
 * WHAT THIS GENERATOR CANNOT DO, STATED UP FRONT. SignalLab's public API is
 * `synthesizeAnalyticComplexIq({profile, sampleRateHz, bandwidthHz,
 * sampleCount, startSampleIndex})`. `startSampleIndex` is the ONLY degree of
 * freedom: no payload, scrambler, PCI, frame-number, scheduler, hop, or
 * content seed. And 29 of the 34 planned profiles are `replay: 'cyclic'` --
 * every catalog generator computes `(startSampleIndex + n) %
 * nativePeriodSamples`, asserted here at run time by `contentFacts()`. For
 * those the timeline holds exactly ONE period of content, forever; a
 * different offset yields a rotation of that period, not a new payload.
 * Random offsets therefore buy TIME ORIGIN diversity, not CONTENT diversity,
 * and this file must never let a caller confuse the two. `contentClass`
 * records the distinction per profile and ALLOW_PHASE_ONLY forces the caller
 * to acknowledge any class-C profile that still lacks a corpus-only content
 * path before a multi-hour run.
 *
 * Output (git-ignored):
 *   <out>/clean_native.f32   concatenated variable-length cf32le rows
 *   <out>/manifest_stage1.json
 *
 * Run:  npm run generate:longdwell-probe-corpus
 *
 * Knobs (all optional):
 *   OUT_DIR=<path>              default training/artifacts/longdwell-probe-corpus
 *   PLAN_JSON=<path>            default tools/production_corpus_plan.json
 *   ROWS_PER_PROFILE=<n>        default 256 (production: 96 eval + 160 train)
 *   DURATION_MS=<n>             default 20
 *   OFFSET_MODE=random|stride   default random. `stride` reproduces the exact
 *                               pre-fix startSampleIndex = row * samplesPerRow
 *                               and forces PHASE_MODE=zero for public
 *                               phase-only synthesis. A declared corpus-only
 *                               recipe still changes content by design.
 *   OFFSET_SEED=<uint32>        default 20260731. Substreams derive from
 *                               (seed, profile, purpose), so editing the plan
 *                               never shifts another profile's draws and
 *                               adding a draw purpose never shifts an
 *                               existing one.
 *   PHASE_MODE=random|zero      default random (zero under OFFSET_MODE=stride)
 *   CONTENT_SPAN_ROWS=<int>=2>  default 8. Timeline window, in row-widths, for
 *                               profiles whose CONTENT varies with
 *                               startSampleIndex (class A). 8 rows = 160 ms,
 *                               spanning >5 BLE advertising events and 256
 *                               Bluetooth BR slots.
 *   OFFSET_SPACING=<int >= 2>   legacy override: forces a uniform
 *                               (SPACING - 1) * samplesPerRow window for every
 *                               profile instead of the protocol-aware window.
 *   ALLOW_PHASE_ONLY=1          required acknowledgement when the plan holds
 *                               any unsupported class-C profile (should vary,
 *                               no corpus-only knob yet).
 *                               Fails closed so a 30 GB regeneration is never
 *                               mistaken for a content-diversity fix.
 *   ALLOW_OVERWRITE=1           permit writing into a populated OUT_DIR
 *   DRY_RUN=1                   plan and print draws only; synthesize nothing
 *   DRY_RUN_PROFILE=<name>      restrict the dry run to one profile
 *   DRY_RUN_ROWS=<n>            rows to print (default 8)
 */
import { closeSync, existsSync, mkdirSync, openSync, readFileSync, writeFileSync, writeSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, join, resolve } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const SIGNAL_LAB = resolve(HERE, '../../Atom-SignalLab');
const OUT_DIR = process.env.OUT_DIR
  ? resolve(process.env.OUT_DIR)
  : resolve(HERE, '../training/artifacts/longdwell-probe-corpus');
const DURATION_MS = Number(process.env.DURATION_MS ?? 20);
// The production design is 34 profiles x 256 rows = 8,704 rows. Keep this
// default aligned with the checked-in plan and the stage-2 split (96 eval +
// 160 train rows per profile); callers can still request a smaller proof
// slice explicitly through ROWS_PER_PROFILE.
const ROWS_PER_PROFILE = Number(process.env.ROWS_PER_PROFILE ?? 256);
const MAX_CALL = 65_536;

const OFFSET_MODE = process.env.OFFSET_MODE ?? 'random';
const OFFSET_SEED = Number(process.env.OFFSET_SEED ?? 20_260_731);
const PHASE_MODE = process.env.PHASE_MODE ?? (OFFSET_MODE === 'stride' ? 'zero' : 'random');
const CONTENT_SPAN_ROWS = Number(process.env.CONTENT_SPAN_ROWS ?? 8);
const OFFSET_SPACING = process.env.OFFSET_SPACING === undefined
  ? null
  : Number(process.env.OFFSET_SPACING);
// Require an explicit acknowledgement for every class-C profile that has not
// yet gained a corpus-only content-seed path, rather than silently emitting a
// 30 GB corpus that cannot answer the memorization question.
const ALLOW_PHASE_ONLY = process.env.ALLOW_PHASE_ONLY === '1';
// A 20 ms capture can legitimately miss a Bluetooth LE advertising event.
// Preserve that observed silence by default and report it in the manifest;
// hiding it would turn the corpus into an undocumented conditioned sample.
// A caller may opt into conditioning with a positive MIN_ACTIVE_SAMPLES. In
// that mode, offsets whose row carries fewer than that many non-zero samples
// are redrawn from the same seeded substream, inside the same row block, and
// the redraw count is recorded per row.
const MIN_ACTIVE_SAMPLES = Number(process.env.MIN_ACTIVE_SAMPLES ?? 0);
const MAX_OFFSET_REDRAWS = Number(process.env.MAX_OFFSET_REDRAWS ?? 64);
const DRY_RUN = process.env.DRY_RUN === '1';
const DRY_RUN_ROWS = Number(process.env.DRY_RUN_ROWS ?? 8);

if (OFFSET_MODE !== 'random' && OFFSET_MODE !== 'stride') {
  throw new Error(`OFFSET_MODE must be 'random' or 'stride', got ${OFFSET_MODE}`);
}
if (PHASE_MODE !== 'random' && PHASE_MODE !== 'zero') {
  throw new Error(`PHASE_MODE must be 'random' or 'zero', got ${PHASE_MODE}`);
}
if (!Number.isSafeInteger(OFFSET_SEED) || OFFSET_SEED < 0) {
  throw new Error('OFFSET_SEED must be a non-negative safe integer');
}
if (!Number.isSafeInteger(ROWS_PER_PROFILE) || ROWS_PER_PROFILE < 1) {
  throw new Error('ROWS_PER_PROFILE must be a positive integer');
}
if (!Number.isSafeInteger(CONTENT_SPAN_ROWS) || CONTENT_SPAN_ROWS < 2) {
  throw new Error('CONTENT_SPAN_ROWS must be an integer >= 2');
}
if (OFFSET_SPACING !== null && (!Number.isSafeInteger(OFFSET_SPACING) || OFFSET_SPACING < 2)) {
  throw new Error('OFFSET_SPACING, when set, must be an integer >= 2');
}
if (!Number.isSafeInteger(MIN_ACTIVE_SAMPLES) || MIN_ACTIVE_SAMPLES < 0) {
  throw new Error('MIN_ACTIVE_SAMPLES must be a non-negative safe integer');
}
if (!Number.isSafeInteger(MAX_OFFSET_REDRAWS) || MAX_OFFSET_REDRAWS < 0) {
  throw new Error('MAX_OFFSET_REDRAWS must be a non-negative safe integer');
}
// cf32le is little-endian by definition; the rotation fast path reinterprets
// the generator's byte buffer as a Float32Array, which is host-endian.
if (new Uint8Array(Uint16Array.of(1).buffer)[0] !== 1) {
  throw new Error('This generator requires a little-endian host (cf32le fast path)');
}

const MASK64 = (1n << 64n) - 1n;
const TWO_POW_64 = 1n << 64n;

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
 * splitmix64. Deterministic, self-contained, and independent per (profile,
 * purpose): the substream seed is fnv1a64(`${OFFSET_SEED}|${profile}|${purpose}`),
 * so a profile's draws depend only on its own name, the corpus seed, and what
 * the draw is FOR. Reordering the plan, adding a profile, or adding a new
 * draw purpose all leave every existing draw untouched.
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

function substreamSeed(profile, purpose) {
  return fnv1a64(`${OFFSET_SEED}|${profile}|${purpose}`);
}

function substream(profile, purpose) {
  return createSplitmix64(substreamSeed(profile, purpose));
}

/** Unbiased uniform integer in [0, boundExclusive) by rejection. */
function nextBoundedBigInt(next, boundExclusive) {
  if (boundExclusive <= 1n) return 0n;
  // Largest multiple of the bound that fits in 2^64; draws at or above it
  // would fold unevenly, so they are rejected.
  const limit = TWO_POW_64 - TWO_POW_64 % boundExclusive;
  for (;;) {
    const draw = next();
    if (draw < limit) return draw % boundExclusive;
  }
}

/** Seeded Fisher-Yates permutation of [0, size). */
function seededPermutation(next, size) {
  const order = Array.from({ length: size }, (_unused, index) => index);
  for (let index = size - 1; index > 0; index -= 1) {
    const swap = Number(nextBoundedBigInt(next, BigInt(index + 1)));
    [order[index], order[swap]] = [order[swap], order[index]];
  }
  return order;
}

/**
 * A lazy stream of distinct integers from [0, space), one per row.
 *
 * Distinctness matters. `wifi-ofdm-20m` has a 1,000-sample content period, so
 * a 256-row profile drawing time origins WITH replacement would collide about
 * 30 times by the birthday bound and silently re-emit the same realization.
 * When the space is no larger than the number of rows, distinctness is
 * arithmetically impossible; `withReplacement` reports that rather than
 * hiding it.
 *
 * STRATIFIED mode additionally guarantees SPREAD. Plain uniform draws from a
 * small space clump: eight uniform draws from fm's 800-sample content period
 * put two of them within a sample of each other, and two 20 ms rows whose
 * time origins differ by one sample of a 25 kHz modulation are 0.999-similar.
 * Row r therefore draws inside stratum `permutation[r]` of `rows` equal
 * strata, so the realization space is enumerated evenly instead of sampled
 * with clumps. The permutation is essential: without it a row's time origin
 * would increase monotonically with its row index, and since the stage-2
 * split is blocked (rows 0..95 eval, 96..255 train) that would hand the eval
 * split the low time origins and the train split the high ones -- a
 * split-correlated artifact worse than the clumping it fixes.
 *
 * Lazy rather than pre-drawn because an explicitly conditioned run may reject
 * its candidate (see MIN_ACTIVE_SAMPLES) and pull another. A rejected
 * stratified draw falls back to the full space, since the stratum may simply
 * be all silence.
 */
function createDrawStream(next, space, rows, stratified) {
  const bound = BigInt(space);
  const withReplacement = space <= rows;
  const seen = new Set();
  const useStrata = stratified && !withReplacement;
  const permutation = useStrata ? seededPermutation(next, rows) : null;

  const fromFullSpace = () => {
    if (withReplacement) return Number(nextBoundedBigInt(next, bound));
    for (;;) {
      const candidate = Number(nextBoundedBigInt(next, bound));
      if (seen.has(candidate)) continue;
      seen.add(candidate);
      return candidate;
    }
  };

  return {
    withReplacement,
    stratified: useStrata,
    /** Unstratified draw, for callers with no row structure (content seeds). */
    next: fromFullSpace,
    draw(row, attempt) {
      if (!useStrata || attempt > 0) return fromFullSpace();
      const stratum = permutation[row];
      const low = Math.floor(stratum * space / rows);
      const high = Math.floor((stratum + 1) * space / rows);
      if (high <= low) return fromFullSpace();
      for (;;) {
        const candidate = low + Number(nextBoundedBigInt(next, BigInt(high - low)));
        if (seen.has(candidate)) continue;
        seen.add(candidate);
        return candidate;
      }
    },
  };
}

// ---------------------------------------------------------------------------
// Content classification
// ---------------------------------------------------------------------------

/**
 * Editorial classification from the 2026-07-31 capability audit. It answers a
 * question no mechanical fact can: given that a profile emits one fixed
 * content realization, is that CORRECT or is it a defect?
 *
 *   A  content genuinely varies with startSampleIndex. The generator composes
 *      an unbounded timeline, so a different offset is a different payload
 *      arrangement -- different slots occupied, different hop channels,
 *      different event timing. Random offsets ARE content variation here.
 *
 *   B  standards-fixed, and the repetition is correct. An LTE E-TM, an NR TM,
 *      an N-TM / NB-IoT component and the cw/am/fm laboratory waveforms are
 *      DEFINED as one repeating reference waveform; a "random payload" E-TM
 *      would not be an E-TM. These get random time origin and random carrier
 *      phase, and nothing else, by design.
 *
 *   C  should vary in the field, but SignalLab exposes no knob. Operational
 *      carriers and Wi-Fi PPDUs carry scheduled user data, GSM bursts carry
 *      per-burst payload, and 802.11 mandates a pseudo-random per-PPDU
 *      scrambler seed that wlan-fixed-iq.ts hardcodes (ERP-OFDM 93, HR/DSSS
 *      0x6c). These are a KNOWN DEFICIT: they get the same treatment as B and
 *      the manifest says so in as many words.
 */
const CONTENT_CLASS = Object.freeze({
  // A -- index-driven content (both Bluetooth long-dwell compositions).
  'bluetooth-classic-connected-longdwell': 'A',
  'bluetooth-le-advertising-longdwell': 'A',

  // B -- standards-fixed reference waveforms; repetition is correct.
  cw: 'B',
  am: 'B',
  fm: 'B',
  'lte-etm1.1': 'B',
  'lte-etm3.1': 'B',
  'lte-etm3.1a': 'B',
  'lte-etm3.1b': 'B',
  'lte-ntm': 'B',
  'lte-nbiot-guard-isolated-component': 'B',
  'lte-nbiot-inband-isolated-component': 'B',
  'nr-fr1-tm1.1': 'B',
  'nr-fr1-tm3.1': 'B',
  'nr-fr1-tm3.1a': 'B',
  'nr-fr1-tm3.1b': 'B',
  'nr-nbiot-inband-isolated-component': 'B',

  // C -- should vary, no knob exists. Known deficit.
  'gsm-900-loaded-bcch': 'C',
  'gsm-normal-burst': 'C',
  'gsm-qpsk-higher-symbol-rate-burst': 'C',
  'gsm-aqpsk-normal-burst': 'C',
  'gsm-8psk-normal-burst': 'C',
  'gsm-16qam-higher-symbol-rate-burst': 'C',
  'gsm-32qam-higher-symbol-rate-burst': 'C',
  'lte-band3-fdd-20m': 'C',
  'lte-band38-tdd-10m': 'C',
  'nr-n3-fdd-20m': 'C',
  'nr-n78-tdd-100m': 'C',
  'wifi-hr-dsss-11m': 'C',
  'wifi-ofdm-20m': 'C',
  'wifi6-he-su': 'C',
  'wifi6-he-er-su': 'C',
  'wifi6-he-mu': 'C',
  'wifi6-he-tb': 'C',
});

const CONTENT_CLASS_NOTE = Object.freeze({
  A: 'content varies with startSampleIndex (unbounded composition), so random '
    + 'offsets are real content variation. Payload BITS still replay the '
    + 'qualified packet vector: diversity is slot occupancy, hop channel and '
    + 'event timing, not new bits.',
  B: 'standards-fixed reference waveform. One repeating realization is CORRECT '
    + 'for this profile; rows vary in time origin and carrier phase only, by '
    + 'design rather than by defect.',
  C: 'Field content should vary. The manifest identifies whether this profile '
    + 'uses a seeded corpus-only generator or remains a phase-only deficit; a '
    + 'profile is never promoted merely because its time origin or carrier phase changes.',
});

/**
 * Corpus-only content generators intentionally live outside SignalLab's
 * catalog/measurement boundary. This table is a capability declaration, not
 * a claim that every class-C profile has been solved: any missing profile
 * remains in CONTENT_DEFICIT_PROFILES and keeps the production run blocked.
 */
const CORPUS_CONTENT_CAPABILITIES = Object.freeze({
  'gsm-900-loaded-bcch': Object.freeze({ recipe: 'geran-corpus-content-v1' }),
  'gsm-normal-burst': Object.freeze({ recipe: 'geran-corpus-content-v1' }),
  'gsm-qpsk-higher-symbol-rate-burst': Object.freeze({ recipe: 'geran-corpus-content-v1' }),
  'gsm-aqpsk-normal-burst': Object.freeze({ recipe: 'geran-corpus-content-v1' }),
  'gsm-8psk-normal-burst': Object.freeze({ recipe: 'geran-corpus-content-v1' }),
  'gsm-16qam-higher-symbol-rate-burst': Object.freeze({ recipe: 'geran-corpus-content-v1' }),
  'gsm-32qam-higher-symbol-rate-burst': Object.freeze({ recipe: 'geran-corpus-content-v1' }),
  'wifi-hr-dsss-11m': Object.freeze({ recipe: 'wlan-corpus-content-v1' }),
  'wifi-ofdm-20m': Object.freeze({ recipe: 'wlan-corpus-content-v1' }),
  'wifi6-he-su': Object.freeze({ recipe: 'wlan-corpus-content-v1' }),
  'wifi6-he-er-su': Object.freeze({ recipe: 'wlan-corpus-content-v1' }),
  'wifi6-he-mu': Object.freeze({ recipe: 'wlan-corpus-content-v1' }),
  'wifi6-he-tb': Object.freeze({ recipe: 'wlan-corpus-content-v1' }),
});

function corpusContentCapability(profile) {
  return CORPUS_CONTENT_CAPABILITIES[profile] ?? null;
}

// ---------------------------------------------------------------------------
// SignalLab bindings. The binding tables are plain data and load in a dry run;
// only the synthesizer is gated, so DRY_RUN never needs a generator call.
// ---------------------------------------------------------------------------
const bindingModule = await import(
  pathToFileURL(join(SIGNAL_LAB, 'src/fixed-digital-profile-binding.ts')).href
);
const FIXED_BINDINGS = bindingModule.FIXED_DIGITAL_PROFILE_BINDINGS;
const UNBOUNDED_BINDINGS = bindingModule.UNBOUNDED_COMPOSITION_PROFILE_BINDINGS;

const synthesizeAnalyticComplexIq = DRY_RUN
  ? null
  : (await import(pathToFileURL(join(SIGNAL_LAB, 'src/complex-iq.ts')).href))
    .synthesizeAnalyticComplexIq;

const synthesizeGeranCorpusContentIq = DRY_RUN
  ? null
  : (await import(pathToFileURL(join(SIGNAL_LAB, 'src/geran-corpus-iq.ts')).href))
    .synthesizeGeranCorpusContentIq;

const synthesizeWlanCorpusContentIq = DRY_RUN
  ? null
  : (await import(pathToFileURL(join(SIGNAL_LAB, 'src/wlan-corpus-iq.ts')).href))
    .synthesizeWlanCorpusContentIq;

/** Select the corpus-only generator when, and only when, a declared capability exists. */
function synthesizeCorpusChunk(spec, sampleCount, startSampleIndex, contentSeed, contentRowIndex) {
  const capability = corpusContentCapability(spec.profile);
  if (capability !== null) {
    if (!Number.isSafeInteger(contentSeed) || contentSeed < 1 || !Number.isSafeInteger(contentRowIndex)) {
      throw new Error(`${spec.profile} corpus-content synthesis requires a valid seed and row index`);
    }
    if (capability.recipe === 'geran-corpus-content-v1') {
      if (synthesizeGeranCorpusContentIq === null) {
        throw new Error(`${spec.profile} declares a GERAN corpus-content capability but its SignalLab generator is unavailable`);
      }
      return synthesizeGeranCorpusContentIq({
        profile: spec.profile,
        sampleRateHz: spec.fs,
        bandwidthHz: spec.bw,
        sampleCount,
        startSampleIndex,
        contentSeed,
        contentRowIndex,
      });
    }
    if (capability.recipe === 'wlan-corpus-content-v1') {
      if (synthesizeWlanCorpusContentIq === null) {
        throw new Error(`${spec.profile} declares a WLAN corpus-content capability but its SignalLab generator is unavailable`);
      }
      return synthesizeWlanCorpusContentIq({
        profile: spec.profile,
        sampleRateHz: spec.fs,
        bandwidthHz: spec.bw,
        sampleCount,
        startSampleIndex,
        contentSeed,
        contentRowIndex,
      });
    }
    throw new Error(`${spec.profile} declares unknown corpus-content recipe ${capability.recipe}`);
  }
  return synthesizeAnalyticComplexIq({
    profile: spec.profile,
    sampleRateHz: spec.fs,
    bandwidthHz: spec.bw,
    sampleCount,
    startSampleIndex,
  });
}

/** The 25 kHz analytic modulation rate shared by the am and fm lab profiles. */
const ANALYTIC_MODULATION_HZ = 25_000;

/**
 * Native content facts, plus a run-time assertion that a profile declared
 * `replay: 'cyclic'` really is cyclic at that period. The assertion is what
 * lets the manifest state `contentRealizationsAvailable: 1` as a measured
 * fact rather than as a reading of the binding table.
 */
function contentFacts(spec) {
  const binding = FIXED_BINDINGS[spec.profile] ?? UNBOUNDED_BINDINGS[spec.profile];
  const replay = binding?.replay ?? 'analytic';
  if (replay === 'cyclic') {
    const period = binding.nativePeriodSamples;
    if (!Number.isSafeInteger(period) || period < 1) {
      throw new Error(`${spec.profile} is cyclic but has no usable nativePeriodSamples`);
    }
    if (synthesizeAnalyticComplexIq !== null) {
      const probe = (index) => createHash('sha256').update(synthesizeAnalyticComplexIq({
        profile: spec.profile,
        sampleRateHz: spec.fs,
        bandwidthHz: spec.bw,
        sampleCount: Math.min(4096, MAX_CALL),
        startSampleIndex: index,
      })).digest('hex');
      const base = period + 12_345;
      if (probe(base) !== probe(base + period)) {
        throw new Error(`${spec.profile} declares a ${period}-sample cyclic period but does `
          + 'not repeat at it; the content-space accounting would be wrong');
      }
    }
    return { replay, contentPeriodSamples: period, contentRealizationsAvailable: 1 };
  }
  if (replay === 'unbounded') {
    return { replay, contentPeriodSamples: null, contentRealizationsAvailable: 'unbounded' };
  }
  // cw/am/fm are closed form: sample = f(t), t = index / fs. cw is the
  // constant 1+0j, so its time origin carries no information whatsoever; am
  // and fm are periodic at the 25 kHz modulation rate.
  if (spec.profile === 'cw') {
    return { replay: 'analytic', contentPeriodSamples: 1, contentRealizationsAvailable: 1 };
  }
  if (spec.fs % ANALYTIC_MODULATION_HZ !== 0) {
    throw new Error(`${spec.profile} sample rate ${spec.fs} is not an integer multiple of the `
      + `${ANALYTIC_MODULATION_HZ} Hz analytic modulation rate; its content period is irrational`);
  }
  return {
    replay: 'analytic',
    contentPeriodSamples: spec.fs / ANALYTIC_MODULATION_HZ,
    contentRealizationsAvailable: 1,
  };
}

/**
 * The window a row's start offset is drawn from, and what that window means.
 *
 * class A: CONTENT_SPAN_ROWS row-widths of unbounded timeline -- every draw is
 *   a different composition realization.
 * class B/C: exactly one content period. That is the COMPLETE space of
 *   distinct realizations this generator can ever produce for the profile;
 *   drawing from a wider window would only re-sample rotations already
 *   reachable inside it. Sampling without replacement therefore enumerates
 *   distinct realizations rather than gambling on the birthday bound.
 */
function offsetWindow(spec, facts, samplesPerRow) {
  if (OFFSET_SPACING !== null) {
    return {
      windowSamples: (OFFSET_SPACING - 1) * samplesPerRow,
      windowBasis: `legacy OFFSET_SPACING=${OFFSET_SPACING}`,
    };
  }
  if (CONTENT_CLASS[spec.profile] === 'A') {
    return {
      windowSamples: CONTENT_SPAN_ROWS * samplesPerRow,
      windowBasis: `content span, ${CONTENT_SPAN_ROWS} rows of unbounded timeline`,
    };
  }
  return {
    windowSamples: facts.contentPeriodSamples,
    windowBasis: `one native content period (${facts.contentPeriodSamples} samples)`,
  };
}

/**
 * Per-row draws for one profile.
 *
 * stride: startSampleIndex = row * samplesPerRow with phase 0 -- the pre-fix
 *   behaviour, kept bit-for-bit so the old corpus can be reproduced.
 * random: row r owns [r * blockSamples, (r + 1) * blockSamples) with
 *   blockSamples = samplesPerRow + windowSamples and starts at a draw inside
 *   the leading windowSamples of it. Rows are therefore strictly
 *   non-overlapping on the timeline -- a train row can never share samples
 *   with an eval row -- while the offset inside the window is uniform and,
 *   whenever the window is larger than the row count, distinct across rows.
 *   A row in an explicitly conditioned run (MIN_ACTIVE_SAMPLES > 0) stays
 *   inside its own block when redrawn, so the non-overlap guarantee survives
 *   rejection.
 *
 * Carrier phases are indexed by row, not by candidate, so a redrawn offset
 * never shifts another row's phase.
 */
function planRowDraws(spec, facts, samplesPerRow, rows) {
  if (OFFSET_MODE === 'stride') {
    return {
      windowSamples: 0,
      windowBasis: 'stride (no window)',
      blockSamples: samplesPerRow,
      withReplacement: false,
      stratified: false,
      offsets: { next: () => 0, draw: () => 0 },
      phases: Array.from({ length: rows }, () => ({
        carrierPhaseDraw: '0',
        carrierPhaseRadians: 0,
      })),
    };
  }
  const { windowSamples, windowBasis } = offsetWindow(spec, facts, samplesPerRow);
  if (!Number.isSafeInteger(windowSamples) || windowSamples < 1) {
    throw new Error(`${spec.profile} resolved an unusable offset window ${windowSamples}`);
  }
  const blockSamples = samplesPerRow + windowSamples;
  if (!Number.isSafeInteger(rows * blockSamples + windowSamples + samplesPerRow)) {
    throw new Error(`${spec.profile} plan exceeds the safe-integer sample coordinate range`);
  }
  // Class-A windows are 8 row-widths of unbounded timeline, so uniform draws
  // are already well spread and stratification would only add structure.
  // Class-B/C windows are exactly one content period -- often tiny (fm 800,
  // wifi-ofdm-20m 1,000 samples) -- so their draws are stratified.
  const offsets = createDrawStream(
    substream(spec.profile, 'offset'),
    windowSamples,
    rows,
    CONTENT_CLASS[spec.profile] !== 'A',
  );
  const nextPhase = substream(spec.profile, 'phase');
  const phases = Array.from({ length: rows }, () => {
    const phaseDraw = PHASE_MODE === 'random' ? nextPhase() : 0n;
    return {
      // Exact reproduction: theta = 2*pi * carrierPhaseDraw / 2^64. The draw
      // is recorded as a decimal string because it does not fit a float64.
      carrierPhaseDraw: phaseDraw.toString(),
      carrierPhaseRadians: 2 * Math.PI * (Number(phaseDraw) / 2 ** 64),
    };
  });
  return {
    windowSamples,
    windowBasis,
    blockSamples,
    withReplacement: offsets.withReplacement,
    stratified: offsets.stratified,
    offsets,
    phases,
  };
}

/**
 * Draw distinct non-zero uint32 seeds up front. Unlike offsets, a seed must
 * remain attached to its logical row even when that row redraws a silent time
 * origin, so this is deliberately not a lazy stream.
 */
function planContentSeeds(profile, rows) {
  const stream = createDrawStream(
    substream(profile, 'content'),
    0xffff_ffff,
    rows,
    false,
  );
  return Array.from({ length: rows }, () => stream.next() + 1);
}

/**
 * Profile plan. Native rates come from the SignalLab binding registry /
 * catalog facts; analytic profiles synthesize directly at the target 20 Msps.
 * gsm-32qam-higher-symbol-rate-burst is included deliberately: it is the
 * historical worst cell (recall 0.2778 in the v5 study).
 */
const PLAN_PATH = process.env.PLAN_JSON
  ? resolve(process.env.PLAN_JSON)
  : join(HERE, 'production_corpus_plan.json');
const PLAN = JSON.parse(readFileSync(PLAN_PATH, 'utf8'));
validatePlan(PLAN);

function validatePlan(plan) {
  if (!Array.isArray(plan) || plan.length === 0) {
    throw new Error('Corpus plan must be a non-empty JSON array');
  }
  const profiles = new Set();
  for (const [index, spec] of plan.entries()) {
    if (typeof spec !== 'object' || spec === null
      || typeof spec.profile !== 'string' || spec.profile.length === 0
      || typeof spec.cls !== 'string' || spec.cls.length === 0
      || !Number.isSafeInteger(spec.fs) || spec.fs < 1
      || !Number.isSafeInteger(spec.bw) || spec.bw < 1 || spec.bw > spec.fs) {
      throw new Error(`Invalid corpus plan entry at index ${index}`);
    }
    if (profiles.has(spec.profile)) {
      throw new Error(`Corpus plan contains duplicate profile ${spec.profile}`);
    }
    if (CONTENT_CLASS[spec.profile] === undefined) {
      throw new Error(`${spec.profile} has no content classification. Classify it as A `
        + '(content varies with startSampleIndex), B (standards-fixed, repetition correct) '
        + 'or C (should vary, no knob) in CONTENT_CLASS before generating.');
    }
    profiles.add(spec.profile);
  }
}

// Draw plan first: pure arithmetic plus one 4096-sample period assertion per
// cyclic profile. A dry run can print it, and a real run can preflight the
// highest start index against every generator's bound, in seconds not hours.
const DRAW_PLAN = new Map(PLAN.map((spec) => {
  const samplesPerRow = Math.ceil((DURATION_MS / 1_000) * spec.fs);
  const facts = contentFacts(spec);
  const contentClass = CONTENT_CLASS[spec.profile];
  const contentCapability = corpusContentCapability(spec.profile);
  const plan = planRowDraws(spec, facts, samplesPerRow, ROWS_PER_PROFILE);
  return [spec.profile, {
    samplesPerRow,
    offsets: plan.offsets,
    phases: plan.phases,
    contentSeeds: contentCapability === null ? null : planContentSeeds(spec.profile, ROWS_PER_PROFILE),
    blockSamples: plan.blockSamples,
    // Highest coordinate any row can reach, redraws included. The preflight
    // probe uses it, so a generator bound rejection costs a second.
    maxStartSampleIndex: (ROWS_PER_PROFILE - 1) * plan.blockSamples
      + Math.max(0, plan.windowSamples - 1),
    meta: {
      contentClass,
      contentPolicy: contentCapability !== null
        ? 'seeded-corpus-content'
        : contentClass === 'A' ? 'timeline-content' : 'phase-only',
      contentPolicyNote: CONTENT_CLASS_NOTE[contentClass],
      contentKnobAvailable: contentClass === 'A' || contentCapability !== null,
      contentDeficit: contentClass === 'C' && contentCapability === null,
      contentRecipe: contentCapability?.recipe ?? null,
      replay: facts.replay,
      contentPeriodSamples: facts.contentPeriodSamples,
      contentRealizationsAvailable: contentCapability !== null
        ? 'seeded-per-row'
        : facts.contentRealizationsAvailable,
      offsetMode: OFFSET_MODE,
      offsetSeed: OFFSET_SEED,
      offsetWindowSamples: plan.windowSamples,
      offsetWindowBasis: plan.windowBasis,
      offsetBlockSamples: plan.blockSamples,
      offsetDistinctPerRow: !plan.withReplacement,
      offsetStratified: plan.stratified,
      offsetSpaceExhausted: plan.withReplacement,
      offsetMinActiveSamples: MIN_ACTIVE_SAMPLES,
      offsetSubstreamSeed: OFFSET_MODE === 'random'
        ? substreamSeed(spec.profile, 'offset').toString() : null,
      phaseMode: PHASE_MODE,
      phaseSubstreamSeed: PHASE_MODE === 'random'
        ? substreamSeed(spec.profile, 'phase').toString() : null,
      contentSubstreamSeed: contentCapability === null
        ? null
        : substreamSeed(spec.profile, 'content').toString(),
    },
  }];
}));

// Fail closed. A 30 GB regeneration must never be mistaken for a content fix.
const CONTENT_DEFICIT_PROFILES = PLAN
  .map((spec) => spec.profile)
  .filter((profile) => CONTENT_CLASS[profile] === 'C' && corpusContentCapability(profile) === null);
if (!DRY_RUN && !ALLOW_PHASE_ONLY && CONTENT_DEFICIT_PROFILES.length > 0) {
  throw new Error(
    `Refusing phase-only generation for ${CONTENT_DEFICIT_PROFILES.length} profile(s) whose `
    + 'field content should vary but for which SignalLab exposes no payload / scrambler / '
    + 'scheduler / frame-number knob. Each will contribute exactly ONE content realization, '
    + `varied only in time origin and carrier phase:\n  ${CONTENT_DEFICIT_PROFILES.join('\n  ')}\n`
    + 'Implement the content-seed path in SignalLab first, or set ALLOW_PHASE_ONLY=1 to '
    + 'acknowledge and proceed. The manifest records the acknowledgement either way.',
  );
}

if (DRY_RUN) {
  const only = process.env.DRY_RUN_PROFILE;
  if (only !== undefined && !DRAW_PLAN.has(only)) {
    throw new Error(`DRY_RUN_PROFILE ${only} is not in the plan`);
  }
  console.log(JSON.stringify({
    dryRun: true,
    planSource: PLAN_PATH,
    durationMs: DURATION_MS,
    rowsPerProfile: ROWS_PER_PROFILE,
    allowPhaseOnlyAcknowledged: ALLOW_PHASE_ONLY,
    contentDeficitProfiles: CONTENT_DEFICIT_PROFILES,
    minActiveSamples: MIN_ACTIVE_SAMPLES,
    nativeBytes: [...DRAW_PLAN.values()]
      .reduce((total, plan) => total + plan.samplesPerRow * 8 * ROWS_PER_PROFILE, 0),
    profiles: Object.fromEntries([...DRAW_PLAN]
      .filter(([profile]) => only === undefined || profile === only)
      .map(([profile, plan]) => [profile, {
        samplesPerRow: plan.samplesPerRow,
        ...plan.meta,
        maxStartSampleIndex: plan.maxStartSampleIndex,
        // FIRST CANDIDATE per row, not necessarily the accepted offset: a row
        // whose window lands in silence pulls another draw at generate time.
        firstCandidateRows: Array.from({ length: Math.min(DRY_RUN_ROWS, ROWS_PER_PROFILE) },
          (_unused, row) => ({
            row,
            candidateStartSampleIndex: row * plan.blockSamples + plan.offsets.draw(row, 0),
            ...plan.phases[row],
            contentSeed: plan.contentSeeds?.[row] ?? null,
          })),
      }])),
  }, null, 1));
  process.exit(0);
}

if (existsSync(join(OUT_DIR, 'manifest_stage1.json')) && process.env.ALLOW_OVERWRITE !== '1') {
  throw new Error(`${OUT_DIR} already exists; set ALLOW_OVERWRITE=1 to regenerate`);
}

// Preflight: randomized offsets reach further along each generator's timeline
// than striding did. Probe one sample at the highest planned coordinate per
// profile so a bound rejection costs a second, not hours.
for (const spec of PLAN) {
  const plan = DRAW_PLAN.get(spec.profile);
  const probeIndex = plan.maxStartSampleIndex + plan.samplesPerRow - 1;
  try {
    synthesizeCorpusChunk(spec, 1, probeIndex, plan.contentSeeds?.[0] ?? null, 0);
  } catch (cause) {
    throw new Error(`${spec.profile} rejects the highest planned start index ${probeIndex} `
      + `(OFFSET_MODE=${OFFSET_MODE}); lower CONTENT_SPAN_ROWS or ROWS_PER_PROFILE`, { cause });
  }
}

/**
 * Rotate one interleaved cf32le block in place by e^(j*theta), accumulating
 * the row's power statistics in the same pass.
 *
 * Safe by construction: SignalLab writes every sample through
 * `writeUnitBoundedCf32le`, which pins it inside the closed unit disk, and a
 * rotation preserves magnitude exactly, so no sample can leave the disk.
 */
function rotateAndMeasure(bytes, cos, sin, stats) {
  const aligned = bytes.byteOffset % 4 === 0;
  const floats = aligned
    ? new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)
    : new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  for (let index = 0; index < floats.length; index += 2) {
    const inPhase = floats[index];
    const quadrature = floats[index + 1];
    const rotatedInPhase = inPhase * cos - quadrature * sin;
    const rotatedQuadrature = inPhase * sin + quadrature * cos;
    floats[index] = rotatedInPhase;
    floats[index + 1] = rotatedQuadrature;
    const power = rotatedInPhase * rotatedInPhase + rotatedQuadrature * rotatedQuadrature;
    stats.sumPower += power;
    if (power > stats.peakPower) stats.peakPower = power;
    if (power > 1e-12) stats.active += 1;
  }
  if (!aligned) bytes.set(new Uint8Array(floats.buffer, 0, bytes.byteLength));
}

mkdirSync(OUT_DIR, { recursive: true });

const fd = openSync(join(OUT_DIR, 'clean_native.f32'), 'w');
const items = [];
let byteOffset = 0;
const startedAt = Date.now();

/**
 * Synthesize one whole row into `buffer` at the given coordinate, rotate it,
 * and return its hashes and power statistics. Rows are assembled in memory
 * (at most 19.7 MB, for nr-n78-tdd-100m) so an offset whose window lands in
 * silence can be rejected before anything reaches the file.
 */
function renderRow(
  spec, buffer, samplesPerRow, startSampleIndex, cos, sin, contentSeed, contentRowIndex,
) {
  // contentSha256 is the unrotated waveform at its selected time origin. It
  // remains useful provenance, but it is NOT content proof because a cyclic
  // waveform at two offsets hashes differently. Seeded corpus profiles also
  // receive a separate fixed-phase contentProbeSha256 below.
  const contentHash = createHash('sha256');
  const rowHash = createHash('sha256');
  const stats = { sumPower: 0, peakPower: 0, active: 0 };
  let written = 0;
  while (written < samplesPerRow) {
    const chunk = Math.min(MAX_CALL, samplesPerRow - written);
    const bytes = synthesizeCorpusChunk(
      spec, chunk, startSampleIndex + written, contentSeed, contentRowIndex,
    );
    contentHash.update(bytes);
    rotateAndMeasure(bytes, cos, sin, stats);
    rowHash.update(bytes);
    buffer.set(bytes, written * 8);
    written += chunk;
  }
  return {
    contentSha256: contentHash.digest('hex'),
    rowSha256: rowHash.digest('hex'),
    rms: Math.sqrt(stats.sumPower / samplesPerRow),
    peak: Math.sqrt(stats.peakPower),
    activeSamples: stats.active,
    activeFraction: stats.active / samplesPerRow,
  };
}

/**
 * A fixed-time-origin, unrotated hash for profiles with a real content seed.
 * This separates payload evidence from phase or selected-time-origin diversity.
 */
function fixedPhaseContentProbe(spec, samplesPerRow, contentSeed, contentRowIndex) {
  if (corpusContentCapability(spec.profile) === null) return null;
  const hash = createHash('sha256');
  let written = 0;
  while (written < samplesPerRow) {
    const chunk = Math.min(MAX_CALL, samplesPerRow - written);
    hash.update(synthesizeCorpusChunk(spec, chunk, written, contentSeed, contentRowIndex));
    written += chunk;
  }
  return hash.digest('hex');
}

for (const spec of PLAN) {
  const plan = DRAW_PLAN.get(spec.profile);
  const { samplesPerRow } = plan;
  const profileStartedAt = Date.now();
  const buffer = new Uint8Array(samplesPerRow * 8);
  for (let row = 0; row < ROWS_PER_PROFILE; row += 1) {
    const { carrierPhaseDraw, carrierPhaseRadians } = plan.phases[row];
    const contentSeed = plan.contentSeeds?.[row] ?? null;
    const contentProbeSha256 = fixedPhaseContentProbe(spec, samplesPerRow, contentSeed, row);
    const cos = Math.cos(carrierPhaseRadians);
    const sin = Math.sin(carrierPhaseRadians);
    let offsetJitter = 0;
    let startSampleIndex = 0;
    let rendered = null;
    let redraws = 0;
    for (;;) {
      offsetJitter = plan.offsets.draw(row, redraws);
      startSampleIndex = row * plan.blockSamples + offsetJitter;
      rendered = renderRow(
        spec, buffer, samplesPerRow, startSampleIndex, cos, sin, contentSeed, row,
      );
      if (rendered.activeSamples >= MIN_ACTIVE_SAMPLES) break;
      redraws += 1;
      if (redraws > MAX_OFFSET_REDRAWS) {
        throw new Error(`${spec.profile} row ${row} could not find an offset carrying `
          + `${MIN_ACTIVE_SAMPLES} active samples in ${MAX_OFFSET_REDRAWS} redraws. Widen `
          + 'CONTENT_SPAN_ROWS, lengthen DURATION_MS, or lower MIN_ACTIVE_SAMPLES.');
      }
    }
    writeSync(fd, buffer);
    items.push({
      profile: spec.profile,
      cls: spec.cls,
      sampleRateHz: spec.fs,
      bandwidthHz: spec.bw,
      startSampleIndex,
      sampleCount: samplesPerRow,
      byteOffset,
      byteLength: samplesPerRow * 8,
      // Recorded draws. startSampleIndex == row * offsetBlockSamples +
      // offsetJitter; offsetJitter is the (row + total prior redraws)-th
      // accepted draw of the profile's `offset` substream and
      // carrierPhaseDraw is draw `row` of its `phase` substream. Every row is
      // reproducible from the seed alone.
      offsetJitter,
      offsetDrawIndex: row,
      offsetRedraws: redraws,
      carrierPhaseDraw,
      carrierPhaseRadians,
      contentClass: plan.meta.contentClass,
      contentPolicy: plan.meta.contentPolicy,
      contentRecipe: plan.meta.contentRecipe,
      contentSeed,
      contentSha256: rendered.contentSha256,
      contentProbeSha256,
      rowSha256: rendered.rowSha256,
      rms: rendered.rms,
      peak: rendered.peak,
      activeFraction: rendered.activeFraction,
    });
    byteOffset += samplesPerRow * 8;
  }
  const rows = items.slice(-ROWS_PER_PROFILE);
  const distinctRow = new Set(rows.map((entry) => entry.rowSha256)).size;
  const distinctContent = new Set(rows.map((entry) => entry.contentSha256)).size;
  const distinctContentProbes = new Set(rows
    .map((entry) => entry.contentProbeSha256)
    .filter((hash) => hash !== null)).size;
  const redraws = rows.reduce((total, entry) => total + entry.offsetRedraws, 0);
  console.log(`${spec.profile}: ${ROWS_PER_PROFILE} rows x ${samplesPerRow} samples, `
    + `class ${plan.meta.contentClass}, distinct row/content/probe sha ${distinctRow}/${distinctContent}/${distinctContentProbes}, `
    + `${redraws} redraws, ${((Date.now() - profileStartedAt) / 1000).toFixed(1)}s `
    + `(${(Date.now() - startedAt) / 1000 | 0}s elapsed)`);
}
closeSync(fd);

// A declared corpus-content source must prove distinct seeded content at a
// fixed time origin before this artifact earns a manifest. This applies to
// both the 8-row proof slice and a production-sized run.
const CONTENT_PROBE_COLLISIONS = PLAN
  .map((spec) => {
    if (corpusContentCapability(spec.profile) === null) return null;
    const rows = items.filter((entry) => entry.profile === spec.profile);
    const distinct = new Set(rows.map((entry) => entry.contentProbeSha256)).size;
    return distinct === rows.length ? null : `${spec.profile} (${distinct}/${rows.length})`;
  })
  .filter((value) => value !== null);
if (CONTENT_PROBE_COLLISIONS.length > 0) {
  throw new Error(
    'Fixed-phase content-probe collision(s) prevent this corpus from serving as content-diversity evidence: '
    + CONTENT_PROBE_COLLISIONS.join(', '),
  );
}

/** Per-profile diversity summary: the corpus documenting its own diversity. */
function summarize(profile) {
  const rows = items.filter((entry) => entry.profile === profile);
  const { meta } = DRAW_PLAN.get(profile);
  const distinctContentSha = new Set(rows.map((entry) => entry.contentSha256)).size;
  const distinctContentProbeSha = new Set(rows
    .map((entry) => entry.contentProbeSha256)
    .filter((hash) => hash !== null)).size;
  const contentSeedCount = new Set(rows
    .map((entry) => entry.contentSeed)
    .filter((seed) => seed !== null)).size;
  const rmsValues = rows.map((entry) => entry.rms).sort((left, right) => left - right);
  return {
    ...meta,
    rows: rows.length,
    distinctRowSha256: new Set(rows.map((entry) => entry.rowSha256)).size,
    distinctContentSha256: distinctContentSha,
    distinctContentProbeSha256: distinctContentProbeSha,
    contentSeedCount,
    distinctTimeOrigins: new Set(rows.map((entry) => (
      meta.contentPeriodSamples === null
        ? entry.startSampleIndex
        : entry.startSampleIndex % meta.contentPeriodSamples
    ))).size,
    distinctCarrierPhases: new Set(rows.map((entry) => entry.carrierPhaseDraw)).size,
    // Only a fixed-phase probe can establish seeded payload diversity. Class
    // A has genuine timeline content; fixed catalog references remain one
    // correct realization by design.
    contentRealizationsRealized: meta.contentPolicy === 'seeded-corpus-content'
      ? distinctContentProbeSha
      : meta.contentClass === 'A' ? distinctContentSha : 1,
    offsetRedrawTotal: rows.reduce((total, entry) => total + entry.offsetRedraws, 0),
    offsetRedrawRows: rows.filter((entry) => entry.offsetRedraws > 0).length,
    zeroRows: rows.filter((entry) => entry.activeFraction === 0).length,
    minActiveFraction: Math.min(...rows.map((entry) => entry.activeFraction)),
    maxActiveFraction: Math.max(...rows.map((entry) => entry.activeFraction)),
    medianRms: rmsValues[rows.length >> 1],
  };
}

writeFileSync(join(OUT_DIR, 'manifest_stage1.json'), JSON.stringify({
  // Schema v3 adds an optional corpus-only content seed, recipe, and
  // fixed-phase probe hash. Existing v2 fields retain their meaning.
  schema: 'longdwell-probe-corpus-stage1-v3',
  generatedAt: new Date().toISOString(),
  planSource: PLAN_PATH,
  durationMs: DURATION_MS,
  rowsPerProfile: ROWS_PER_PROFILE,
  totalRows: items.length,
  totalBytes: byteOffset,
  stitchMaxCallSamples: MAX_CALL,
  offsetMode: OFFSET_MODE,
  offsetSeed: OFFSET_SEED,
  offsetSpacing: OFFSET_SPACING,
  phaseMode: PHASE_MODE,
  contentSpanRows: CONTENT_SPAN_ROWS,
  minActiveSamples: MIN_ACTIVE_SAMPLES,
  maxOffsetRedraws: MAX_OFFSET_REDRAWS,
  offsetRejectionPolicy: MIN_ACTIVE_SAMPLES > 0
    ? `rows carrying fewer than ${MIN_ACTIVE_SAMPLES} non-zero samples were rejected and `
      + 'the offset redrawn inside the same row block. This CONDITIONS the corpus: an '
      + 'unconditional 20 ms capture of a 30 ms Bluetooth LE advertising cadence is silent '
      + 'about 31% of the time, and those rows are excluded here on purpose, because a '
      + 'silent row labelled with a protocol teaches the classifier that silence is that '
      + 'protocol. items[].offsetRedraws records the cost per row.'
    : 'disabled (MIN_ACTIVE_SAMPLES=0): natural silent rows are retained and '
      + 'reported through items[].activeFraction and diversity[profile].zeroRows',
  allowPhaseOnlyAcknowledged: ALLOW_PHASE_ONLY,
  contentDeficitProfiles: CONTENT_DEFICIT_PROFILES,
  elapsedSeconds: (Date.now() - startedAt) / 1000,
  diversity: Object.fromEntries(PLAN.map((spec) => [spec.profile, summarize(spec.profile)])),
  note: 'clean native-rate captures; stage 2 resamples to a common rate and '
    + 'applies the receiver impairment chain to produce the noisy pair. Rows '
    + 'differ in seeded time origin and seeded carrier phase. Profiles with a '
    + 'declared corpus-only recipe also receive a mandatory contentSeed and a '
    + 'fixed-phase contentProbeSha256; the generator rejects collisions in that '
    + 'evidence. Remaining class-C profiles are explicitly listed in '
    + 'contentDeficitProfiles and keep a non-acknowledged production run blocked. '
    + 'diversity[profile] states the exact policy per profile.',
  items,
}, null, 1));
console.log(`wrote ${items.length} rows, ${(byteOffset / 1e9).toFixed(2)} GB to ${OUT_DIR} `
  + `in ${((Date.now() - startedAt) / 60_000).toFixed(1)} min`);
