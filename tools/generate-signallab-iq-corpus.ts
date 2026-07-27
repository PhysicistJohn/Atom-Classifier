/**
 * Generate a labeled complex-I/Q training corpus from Atom-SignalLab's OWN
 * synthesizer — the exact `synthesizeAnalyticComplexIq` path the Atomizer app
 * feeds the classifier. This replaces the mismatched rfgen training data so the
 * embedding learns SignalLab's actual I/Q distribution.
 *
 * The 34 catalog profiles are grouped into I/Q-separable modulation classes that
 * map cleanly onto the app's protocol-leaf taxonomy (bandwidth/band context then
 * disambiguates the OFDM protocols at fusion time):
 *   cw · am · fm · gsm(GERAN) · ofdm(LTE+NR+Wi-Fi-OFDM) · dsss(Wi-Fi HR/DSSS) · bluetooth
 *
 * SignalLab's I/Q is clean and (for continuous signals) deterministic per
 * geometry, so diversity comes from varied capture geometry (fractional
 * occupancy) + moving time windows here, plus light AWGN/CFO added in Python.
 *
 * Output (git-ignored):
 *   training/artifacts/signallab-corpus/corpus.f32   concatenated cf32le blocks
 *   training/artifacts/signallab-corpus/corpus.json  manifest (order matches .f32)
 *
 * Run:  npx tsx tools/generate-signallab-iq-corpus.ts
 */

import {
  closeSync,
  existsSync,
  mkdirSync,
  openSync,
  readFileSync,
  writeFileSync,
  writeSync,
} from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, join, resolve } from 'node:path';

// The normal app checkout keeps Atom-SignalLab beside this repository. Release
// generation may instead point at a clean, isolated source tree. This matters
// when the sibling checkout or one of its file: dependencies is dirty/damaged:
// release evidence must never repair or silently depend on that mutable tree.
const SIGNAL_LAB_ROOT = process.env.SIGNALLAB_ROOT
  ? resolve(process.env.SIGNALLAB_ROOT)
  : resolve(dirname(fileURLToPath(import.meta.url)), '../../Atom-SignalLab');
for (const relative of [
  'src/complex-iq.ts',
  'src/impairments.ts',
  'src/waveforms.ts',
]) {
  if (!existsSync(join(SIGNAL_LAB_ROOT, relative))) {
    throw new Error(
      `SIGNALLAB_ROOT is missing ${relative}: ${SIGNAL_LAB_ROOT}`,
    );
  }
}
const complexIqModule = await import(
  pathToFileURL(join(SIGNAL_LAB_ROOT, 'src/complex-iq.ts')).href
) as typeof import('../../Atom-SignalLab/src/complex-iq.js');
const impairmentsModule = await import(
  pathToFileURL(join(SIGNAL_LAB_ROOT, 'src/impairments.ts')).href
) as typeof import('../../Atom-SignalLab/src/impairments.js');
const waveformsModule = await import(
  pathToFileURL(join(SIGNAL_LAB_ROOT, 'src/waveforms.ts')).href
) as typeof import('../../Atom-SignalLab/src/waveforms.js');
const { synthesizeAnalyticComplexIq } = complexIqModule;
const { synthesizeImpairedComplexIq } = impairmentsModule;
const { waveformCatalog } = waveformsModule;
type ReceiverImpairments = import(
  '../../Atom-SignalLab/src/impairments.js'
).ReceiverImpairments;

const SAMPLE_COUNT = process.env.SAMPLE_COUNT ? parseInt(process.env.SAMPLE_COUNT, 10) : 16384;
const CORPUS_SEED = process.env.CORPUS_SEED
  ? parseInt(process.env.CORPUS_SEED, 10)
  : 20260721;
const ALLOW_OVERWRITE = process.env.ALLOW_OVERWRITE === '1';
const START_STRIDE_SAMPLES = process.env.START_STRIDE_SAMPLES
  ? parseInt(process.env.START_STRIDE_SAMPLES, 10)
  : SAMPLE_COUNT;
// 4096 -> 16384 (2026-07-24). At 4096 the capture was too short in TIME to contain
// the burst structure that distinguishes the bursty digital classes from each other:
// a Bluetooth Classic slot is 625us, a BLE advertising packet 376us, a GSM normal
// burst 577us. At 4096 samples and the high sample rates in the sweep, windows were
// as short as 137us -- less than a quarter of any of those. Burst duration, duty
// cycle and packet structure were therefore not merely hard to extract, they were
// absent from the data, which is the leading explanation for a bluetooth/gsm/dsss
// confusion that survived ~10 modelling interventions across 6 architecture families.
// At 16384 the shortest window in the sweep is ~546us (>=0.87 of a Bluetooth slot,
// ~0.95 of a GSM burst) and the longest is tens of milliseconds (many slots/frames).

// Profiles whose catalog `occupiedBandwidthHz` is an AGGREGATE HOP/CHANNEL SPAN, not
// the instantaneous occupied bandwidth of the emission at any one instant. The catalog
// says so explicitly for Bluetooth Classic: "The 79 MHz field is the aggregate
// edge-to-edge support across the 79 modeled 1 MHz channels ... not instantaneous
// occupied bandwidth." Sizing a capture from the span instead of the channel is what
// put Bluetooth at 175-245 MHz sample rate for a ~1-2 MHz emission: ~180x oversampled,
// 17-23us windows, and (in the resampled pipeline) ~96 real samples zero-padded into
// 1024. Values below are the catalog's own channelWidthHz parameters.
const INSTANTANEOUS_BANDWIDTH_HZ: Record<string, number> = {
  'bluetooth-classic-connected': 1_000_000,  // waveforms.ts: channelWidthHz
  'bluetooth-le-advertising': 2_000_000,     // canonical-timing.ts: channelWidthHz
};
const TARGET_PER_CLASS = process.env.TARGET_PER_CLASS ? parseInt(process.env.TARGET_PER_CLASS, 10) : 260;
const EXACT_TARGET_PER_CLASS = process.env.EXACT_TARGET_PER_CLASS === '1';
const TARGET_FRACS = [0.08, 0.12, 0.18, 0.25, 0.35, 0.45];
// 1 in every CLEAN_EVERY realizations is left clean (for prototype enrollment +
// app-match validation); the rest get SignalLab's seeded receiver impairments.
const CLEAN_EVERY = 4;

function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------------------
// NUISANCE PARAMETER DISTRIBUTION (2026-07-24 rewrite)
//
// This model has to work in the field on arbitrary bandwidths, sample rates and
// centre frequencies, so every capture/receiver/channel parameter that varies out
// there has to vary HERE, and -- critically -- has to be drawn INDEPENDENTLY OF
// CLASS. Any nuisance parameter that correlates with the label is a shortcut the
// network will take instead of learning signal structure, and it evaporates on
// contact with real captures. Measured example from the previous build: sample
// rate carried 0.859 bits about the class (31% of class entropy) purely because
// fs was derived from each profile's own bandwidth; accuracy was 0.99+ in fs bands
// where that narrowed the candidate set and 0.967 in the single band where all 7
// classes coexist.
//
// Two structural rules follow, and both are enforced below:
//   1. Draw every nuisance from its own RNG stream keyed on the realization index,
//      never from a per-profile counter (the old code used `k % CLEAN_EVERY` for
//      clean-vs-impaired while `k % rates.length` chose the sample rate, so
//      impairment status was locked to sample rate).
//   2. Sweep to the edges of what the field actually presents, not to what looks
//      tidy. The old SNR floor of 10 dB meant the model had never seen a marginal
//      capture; real detections happen near 0 dB.
//
// Coverage status per parameter is tabulated in corpus.json's `nuisanceSpec` so a
// reader can see what is swept and what is still fixed without reading this file.
// ---------------------------------------------------------------------------
const NUISANCE_SPEC = {
  snrDb: 'U(-5,45) dB, back-solved to a received power inside the radio envelope (-110..0 dBm); power clamps at the rails',
  carrierFrequencyOffset: 'DERIVED: refError U(-2,2) ppm x centreHz / fs (LO-scaled, was flat +/-0.004)',
  centreOffsetFrac: 'U(-0.35, 0.35) of the band: emission is NOT assumed centred in the capture',
  centerHz: 'log-uniform U(70 MHz, 10 GHz): sets LO-derived CFO and phase noise',
  phaseNoiseStd: 'DERIVED: (0.004 + U(0,0.03)) x sqrt(centreHz/1GHz) rad/sample',
  iqGainImbalance: 'U(-0.12, 0.12)',
  iqPhaseImbalance: 'U(-0.30, 0.30) rad',
  dcInPhase: 'U(-0.10, 0.10)', dcQuadrature: 'U(-0.10, 0.10)',
  multipath: '0-3 taps, delay U(1, 64) samples, Rayleigh-ish complex gains (was <=1 tap, delay<=9)',
  paSaturation: '35% of realizations, knee U(0.45, 0.95)',
  adcBits: '30% of realizations quantized to U{8,10,12,14} bits',
  clockErrorPpm: 'U(-40, 40) ppm applied as a resample stretch',
  impairedProbability: 'independent Bernoulli(0.75), NOT a k%4 pattern locked to sample rate',
  sampleRateHz: 'geometric sweep over the full feasible range per profile (Nyquist+guard .. SR_MAX)',
  captureLengthSamples: 'FIXED at SAMPLE_COUNT -- not yet swept (flat binary format); known gap',
  interference: 'NOT MODELLED -- known gap',
  fadingDoppler: 'NOT MODELLED (multipath is static) -- known gap',
} as const;

/** Derive the achievable SNR from received power, capture bandwidth and noise
 *  figure -- see the RX_POWER_* commentary above for why this is not drawn free. */
function drawLinkBudget(rand: () => number, sampleRateHz: number) {
  const noiseFigureDb = 3 + rand() * 9;                       // 3-12 dB, typical SDR front ends
  const noiseFloorDbm = THERMAL_DBM_PER_HZ + 10 * Math.log10(sampleRateHz) + noiseFigureDb;
  // Draw the SNR uniformly and back-solve the received power, rather than drawing
  // power uniformly. Both are physically consistent -- every realization still has a
  // received power inside the radio's real -110..0 dBm envelope -- but drawing power
  // flat over a 110 dB range puts most of the mass at high SNR once the noise floor
  // is subtracted (measured: median 41 dB, i.e. half the corpus effectively clean),
  // which is the opposite of where classification is hard. Uniform-in-SNR spends the
  // corpus where the decision is actually difficult.
  let snrDb = -5 + rand() * 50;                               // -5 .. 45 dB
  let rxPowerDbm = noiseFloorDbm + snrDb;
  if (rxPowerDbm > RX_POWER_MAX_DBM) { rxPowerDbm = RX_POWER_MAX_DBM; snrDb = rxPowerDbm - noiseFloorDbm; }
  if (rxPowerDbm < RX_POWER_MIN_DBM) { rxPowerDbm = RX_POWER_MIN_DBM; snrDb = rxPowerDbm - noiseFloorDbm; }
  return { rxPowerDbm, noiseFigureDb, noiseFloorDbm, snrDb };
}

function drawImpairments(rand: () => number, sampleRateHz: number, centerHz: number,
                         snrDb: number): ReceiverImpairments {
  // LO-derived terms scale with tuned frequency, so they are computed from the drawn
  // centre frequency rather than being flat across the corpus.
  const refPpm = (rand() - 0.5) * 4;                          // +/-2 ppm reference error
  const cfoHz = refPpm * 1e-6 * centerHz;
  const carrierFrequencyOffset = Math.max(-0.25, Math.min(0.25, cfoHz / sampleRateHz));
  const phaseNoiseStd = (0.004 + rand() * 0.03) * Math.sqrt(centerHz / 1e9);

  const nTaps = Math.floor(rand() * 4); // 0..3
  const multipath = [];
  for (let t = 0; t < nTaps; t++) {
    // delays out to 64 samples: at a few MHz that is microseconds of delay spread,
    // which is what indoor/urban multipath actually looks like. The old cap of 9
    // samples was sub-100ns at high fs and therefore nearly a no-op.
    const mag = Math.sqrt(-2 * Math.log(1 - rand() * 0.999)) * 0.25; // Rayleigh-ish
    const ang = rand() * 2 * Math.PI;
    multipath.push({
      delay: 1 + Math.floor(rand() * 64),
      gainInPhase: Math.min(0.8, mag) * Math.cos(ang),
      gainQuadrature: Math.min(0.8, mag) * Math.sin(ang),
    });
  }
  return {
    snrDb,
    carrierFrequencyOffset,
    phaseNoiseStd,
    iqGainImbalance: (rand() - 0.5) * 0.24,
    iqPhaseImbalance: (rand() - 0.5) * 0.6,
    dcInPhase: (rand() - 0.5) * 0.2,
    dcQuadrature: (rand() - 0.5) * 0.2,
    multipath,
    ...(rand() < 0.35 ? { paSaturation: 0.45 + rand() * 0.5 } : {}),
  };
}

/** Post-synthesis nuisances SignalLab's impairment chain does not cover. Applied
 *  in-place to the interleaved cf32 buffer. */
interface CaptureNuisance { centreOffsetFrac: number; adcBits: number | null; clockErrorPpm: number }

function drawCaptureNuisances(rand: () => number): CaptureNuisance {
  return {
    centreOffsetFrac: (rand() - 0.5) * 0.7,
    clockErrorPpm: (rand() - 0.5) * 80,
    adcBits: rand() < 0.3 ? [8, 10, 12, 14][Math.floor(rand() * 4)]! : null,
  };
}

function applyCaptureNuisances(bytes: Uint8Array, nui: CaptureNuisance): CaptureNuisance {
  const buf = f32view(bytes);
  const n = buf.length / 2;

  // (a) The emission is not centred in the capture. Real tuning is off by an
  // arbitrary fraction of the band, and preprocess.estimate_band has to find it.
  // With everything centred the centre estimator is never exercised.
  const { centreOffsetFrac, clockErrorPpm, adcBits } = nui;
  const w = 2 * Math.PI * centreOffsetFrac;
  for (let i = 0; i < n; i++) {
    const re = buf[2 * i]!, im = buf[2 * i + 1]!;
    const c = Math.cos(w * i), s = Math.sin(w * i);
    buf[2 * i] = re * c - im * s;
    buf[2 * i + 1] = re * s + im * c;
  }

  // (b) Sample-clock error: a real radio's ADC clock is off by tens of ppm, so the
  // symbol rate the model sees is not exactly nominal. Applied as a tiny linear
  // resample stretch (cheap first-order stand-in for a true fractional resampler).
  if (Math.abs(clockErrorPpm) > 1) {
    const stretch = 1 + clockErrorPpm * 1e-6;
    const src = Float32Array.from(buf);
    for (let i = 0; i < n; i++) {
      const p = Math.min(n - 1.001, i * stretch);
      const i0 = Math.floor(p), f = p - i0;
      buf[2 * i] = src[2 * i0]! * (1 - f) + src[2 * (i0 + 1)]! * f;
      buf[2 * i + 1] = src[2 * i0 + 1]! * (1 - f) + src[2 * (i0 + 1) + 1]! * f;
    }
  }

  // (c) Finite ADC word length. Quantization sets a noise floor and clips peaks;
  // a model trained only on float64-clean captures has never seen either.
  if (adcBits !== null) {
    let peak = 0;
    for (let i = 0; i < buf.length; i++) peak = Math.max(peak, Math.abs(buf[i]!));
    if (peak > 0) {
      const levels = Math.pow(2, adcBits - 1) - 1;
      const q = peak / levels;
      for (let i = 0; i < buf.length; i++) {
        buf[i] = Math.max(-levels, Math.min(levels, Math.round(buf[i]! / q))) * q;
      }
    }
  }
  return nui;
}
// Common Atomizer I/Q sample rates — narrowband signals are heavily oversampled
// at these, which is exactly the geometry the app feeds; include it so the
// embedding is robust to it (not just the well-matched occupancy sweep).
const APP_SAMPLE_RATES = [2_000_000, 8_000_000, 30_000_000];
const SR_MIN = 1_000_000;
// Hardware bound (user, 2026-07-24): the radios in use max out at 200 MSPS and
// 10 GHz. Sweeping past what the hardware can present just spends corpus on
// operating points that will never occur.
const SR_MAX = 200_000_000;

// Tuner range, used to scale the LO-derived impairments below. Absolute centre
// frequency never reaches the network (everything is downconverted to baseband),
// but it DOES set how bad the local oscillator is, so it has to be swept: a 1 ppm
// reference is +/-100 Hz at 100 MHz and +/-10 kHz at 10 GHz, and oscillator phase
// noise degrades ~20*log10(fc) as well. Training only at low fc would understate
// both by two orders of magnitude.
const FC_MIN_HZ = 70_000_000;
const FC_MAX_HZ = 10_000_000_000;

// Receiver power envelope (user): roughly 0 dBm maximum to -110 dBm minimum.
// SNR is DERIVED from this rather than drawn independently, because the two are
// physically linked: thermal noise is -174 dBm/Hz, so the noise floor in the
// capture is -174 + 10log10(bandwidth) + noise figure. A -110 dBm emission in a
// 1 MHz channel sits right at that floor (SNR ~ 0 dB); the same emission in a
// 100 MHz capture is 20 dB worse. Drawing SNR uniformly, as before, silently
// asserted that signal power and capture bandwidth are unrelated, which is wrong
// and is exactly the kind of thing that does not survive contact with real captures.
const RX_POWER_MAX_DBM = 0;
const RX_POWER_MIN_DBM = -110;
const THERMAL_DBM_PER_HZ = -174;

type Descriptor = (typeof waveformCatalog)[number];
interface MatchedStartRow {
  profile: string;
  sampleRateHz: number;
  bandwidthHz: number;
  startSampleIndex: number;
}

const MATCH_STARTS_FROM = process.env.MATCH_STARTS_FROM
  ? resolve(process.env.MATCH_STARTS_FROM)
  : null;
const matchedStarts: MatchedStartRow[] | null = MATCH_STARTS_FROM
  ? (JSON.parse(readFileSync(MATCH_STARTS_FROM, 'utf8')).items as MatchedStartRow[])
  : null;

function classOf(d: Descriptor): string {
  if (d.id === 'cw') return 'cw';
  if (d.id === 'am') return 'am';
  if (d.id === 'fm') return 'fm';
  if (d.family === 'geran') return 'gsm';
  if (d.id === 'wifi-hr-dsss-11m') return 'dsss';
  if (d.family === 'bluetooth') return 'bluetooth';
  if (d.family === 'e-utra' || d.family === 'nr' || d.family === 'wlan') return 'ofdm';
  return 'unknown';
}

function clampInt(value: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, Math.round(value)));
}

// group profiles by class
const byClass = new Map<string, Descriptor[]>();
for (const d of waveformCatalog) {
  const cls = classOf(d);
  if (cls === 'unknown') continue;
  (byClass.get(cls) ?? byClass.set(cls, []).get(cls)!).push(d);
}

// Occupancy gate: clean synthesis of an idle window is *exactly* zero, so any
// positive floor separates "emission present" from "idle gap" unambiguously.
const OCCUPANCY_MIN_POWER = 1e-9;
const OCCUPANCY_MAX_TRIES = 400;   // BLE's 20-30ms advertising interval at a 0.16ms stride
let rejectedEmpty = 0;

/** The synthesizers hand back Uint8Array holding raw cf32le bytes, NOT samples.
 *  Every numeric inspection/modification below goes through this float view. */
function f32view(bytes: Uint8Array): Float32Array {
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

function meanPower(bytes: Uint8Array): number {
  const a = f32view(bytes);
  let sum = 0;
  for (let i = 0; i < a.length; i += 2) sum += a[i]! * a[i]! + a[i + 1]! * a[i + 1]!;
  return sum / (a.length / 2);
}

// Stream both binaries to disk as they are produced. Accumulating them in memory and
// Buffer.concat-ing at the end overflows Node's max buffer length at full corpus size
// (2 x ~1.8 GB) and throws ERR_OUT_OF_RANGE.
if (!Number.isSafeInteger(SAMPLE_COUNT) || SAMPLE_COUNT < 64) {
  throw new RangeError('SAMPLE_COUNT must be a safe integer >= 64');
}
if (!Number.isSafeInteger(CORPUS_SEED) || CORPUS_SEED < 0) {
  throw new RangeError('CORPUS_SEED must be a non-negative safe integer');
}
if (!Number.isSafeInteger(START_STRIDE_SAMPLES) || START_STRIDE_SAMPLES <= 0) {
  throw new RangeError('START_STRIDE_SAMPLES must be a positive safe integer');
}
if (!Number.isSafeInteger(TARGET_PER_CLASS) || TARGET_PER_CLASS <= 0) {
  throw new RangeError('TARGET_PER_CLASS must be a positive safe integer');
}

const defaultOutDir = join(
  dirname(fileURLToPath(import.meta.url)),
  '..',
  'training',
  'artifacts',
  'signallab-corpus',
);
const outDirEarly = process.env.OUTPUT_DIR
  ? resolve(process.env.OUTPUT_DIR)
  : defaultOutDir;
mkdirSync(outDirEarly, { recursive: true });
const outputPaths = [
  join(outDirEarly, 'corpus.f32'),
  join(outDirEarly, 'corpus_clean.f32'),
  join(outDirEarly, 'corpus.json'),
];
if (!ALLOW_OVERWRITE) {
  const existing = outputPaths.filter((path) => existsSync(path));
  if (existing.length > 0) {
    throw new Error(
      `refusing to overwrite an existing corpus: ${existing.join(', ')}; `
      + 'choose a fresh OUTPUT_DIR (recommended) or set ALLOW_OVERWRITE=1 explicitly',
    );
  }
}
const fdMain = openSync(outputPaths[0]!, ALLOW_OVERWRITE ? 'w' : 'wx');
const fdClean = openSync(outputPaths[1]!, ALLOW_OVERWRITE ? 'w' : 'wx');
const items: Record<string, unknown>[] = [];
const rand = mulberry32(CORPUS_SEED);

for (const [cls, profiles] of byClass) {
  for (const [profileIndex, d] of profiles.entries()) {
    // Historical training builds used a minimum of eight rows per catalog
    // profile. A sealed evaluation instead needs the declared class population
    // to be exact, otherwise classes with many profiles (especially OFDM) get
    // several times more rows than classes with one profile. The exact mode
    // distributes the remainder deterministically in catalog order.
    const perProfile = EXACT_TARGET_PER_CLASS
      ? Math.floor(TARGET_PER_CLASS / profiles.length)
        + (profileIndex < TARGET_PER_CLASS % profiles.length ? 1 : 0)
      : Math.max(8, Math.round(TARGET_PER_CLASS / profiles.length));
    // sample rates to sweep: the occupancy-matched set + the app's common rates
    // (where they are wide enough to actually carry the signal).
    // Use the INSTANTANEOUS occupied bandwidth to size the capture. For frequency-
    // hopping / multi-channel-agile profiles the catalog's occupiedBandwidthHz is the
    // aggregate span across all channels, which is 30-80x wider than what the emission
    // actually occupies at any instant -- see INSTANTANEOUS_BANDWIDTH_HZ above.
    const occ = INSTANTANEOUS_BANDWIDTH_HZ[d.id] ?? d.occupiedBandwidthHz;

    // SAMPLE-RATE DECORRELATION (2026-07-24).
    // Previously the rate set was occ/TARGET_FRACS, i.e. fs was DETERMINED by the
    // signal's own bandwidth. That made capture parameterization a class label:
    // measured I(class; fs) = 0.859 bits, 31% of class entropy, with dsss never
    // appearing below 30 MHz and am/cw/fm/gsm never above it. A classifier can then
    // read fs instead of the waveform -- accuracy was 0.99+ in bands where fs narrowed
    // the candidate set and 0.967 in the one band where all 7 classes coexist. In the
    // field the operator chooses fs freely, so that shortcut is worthless and the
    // aggregate number is inflated.
    // Fix: sweep each profile across its FULL physically feasible rate range. The only
    // hard constraint is Nyquist plus guard band (fs >= 1.15 * occupied bandwidth); a
    // narrowband emission can legitimately be captured anywhere from just above its own
    // bandwidth up to the widest rate the radio supports (a wideband survey capture that
    // happens to contain a narrow signal is a real and common case).
    // Consequence, deliberately accepted: at high fs a fixed 16384-sample window spans
    // little time (67us at 245 MHz), so burst structure is genuinely unavailable there.
    // That is a true property of the operating point, not an artifact -- so it is swept
    // rather than hidden, and results are reported per fs band.
    const fsFloor = Math.max(SR_MIN, Math.ceil(occ * 1.15));
    const RATES_PER_PROFILE = 10;
    const rates: number[] = [];
    if (fsFloor <= SR_MAX) {
      for (let i = 0; i < RATES_PER_PROFILE; i++) {
        // geometric sweep floor -> SR_MAX so every octave is represented
        const t = RATES_PER_PROFILE === 1 ? 0 : i / (RATES_PER_PROFILE - 1);
        rates.push(clampInt(fsFloor * Math.pow(SR_MAX / fsFloor, t), fsFloor, SR_MAX));
      }
      // keep the app's real capture rates in the mix where they are legal
      for (const r of APP_SAMPLE_RATES) if (r >= fsFloor && r <= SR_MAX) rates.push(r);
    } else {
      rates.push(SR_MAX);
    }
    for (let k = 0; k < perProfile; k++) {
      // stride by a coprime step so a small perProfile still walks the whole rate
      // sweep instead of only its first few entries
      const sampleRateHz = rates[(k * 7) % rates.length]!;
      const bandwidthHz = clampInt(occ * 1.15, 1_000, Math.floor(sampleRateHz * 0.95));
      const impaired = rand() < 0.75;   // independent of k, so independent of sample rate

      // OCCUPANCY REJECTION SAMPLING (2026-07-24).
      // Stepping startSampleIndex by k*SAMPLE_COUNT sweeps burst PHASE, which is
      // wanted, but for low-duty-cycle profiles it lands in the idle gap most of the
      // time and yields a window with NO EMISSION IN IT. Measured on the previous
      // build: 41% of clean bluetooth realizations (80% of bluetooth-le-advertising,
      // which transmits a 376us packet once per 20-30ms) and 36% of clean dsss had
      // *exactly zero* power. Those are pure-noise windows carrying a class label --
      // label noise, and specifically the kind that makes two low-duty classes
      // mutually indistinguishable, since their empty windows are identical. That is
      // the leading explanation for the residual symmetric bluetooth<->dsss confusion
      // (129 and 96) left after the instantaneous-bandwidth fix.
      // This classifier runs downstream of energy detection in the app, so a window
      // with no signal is out of distribution for it, not a hard example.
      // We therefore advance the start index in sub-burst steps until the CLEAN
      // synthesis actually contains energy. Partial bursts still pass (the test is
      // "contains emission", not "contains a whole burst"), so genuine occupancy
      // variation -- partial, full, multiple bursts -- is preserved.
      // ~0.2 ms per step at any sample rate -> 400 retries sweep ~80 ms, which covers
      // BLE's 20-30 ms advertising interval regardless of fs.
      const stride = Math.max(1, Math.floor(sampleRateHz * 0.0002));
      const matchedStart = matchedStarts?.[items.length];
      if (matchedStarts && !matchedStart) {
        throw new Error(
          `MATCH_STARTS_FROM has no row ${items.length}: ${MATCH_STARTS_FROM}`,
        );
      }
      if (
        matchedStart
        && (
          matchedStart.profile !== d.id
          || matchedStart.sampleRateHz !== sampleRateHz
          || matchedStart.bandwidthHz !== bandwidthHz
        )
      ) {
        throw new Error(
          `MATCH_STARTS_FROM row ${items.length} geometry does not match `
          + `${d.id}/${sampleRateHz}/${bandwidthHz}`,
        );
      }
      let startSampleIndex = matchedStart?.startSampleIndex
        ?? k * START_STRIDE_SAMPLES;
      let clean = synthesizeAnalyticComplexIq({ profile: d.id, sampleRateHz, bandwidthHz, sampleCount: SAMPLE_COUNT, startSampleIndex });
      let tries = 0;
      while (
        !matchedStart
        && meanPower(clean) <= OCCUPANCY_MIN_POWER
        && tries < OCCUPANCY_MAX_TRIES
      ) {
        tries += 1;
        startSampleIndex += stride;
        clean = synthesizeAnalyticComplexIq({ profile: d.id, sampleRateHz, bandwidthHz, sampleCount: SAMPLE_COUNT, startSampleIndex });
      }
      if (meanPower(clean) <= OCCUPANCY_MIN_POWER) {
        if (matchedStart) {
          throw new Error(
            `matched start row ${items.length} became empty at length ${SAMPLE_COUNT}`,
          );
        }
        rejectedEmpty += 1;
      }

      const input = { profile: d.id, sampleRateHz, bandwidthHz, sampleCount: SAMPLE_COUNT, startSampleIndex };
      const centerHz = FC_MIN_HZ * Math.pow(FC_MAX_HZ / FC_MIN_HZ, rand());
      const link = drawLinkBudget(rand, sampleRateHz);
      const imp = drawImpairments(rand, sampleRateHz, centerHz, link.snrDb);

      // Emit the PAIR: the impaired capture the classifier sees, and the clean
      // emission it came from. The clean member is the supervision target for a
      // learned equalizer front end -- without stored pairs the front end can only be
      // trained implicitly through classification loss, which is a far weaker signal
      // than "reconstruct the emission this capture came from". Same startSampleIndex
      // for both, so they are genuinely the same emission, differing only in channel.
      // Keep the historical default bitstream reproducible.  A deliberately
      // new corpus seed also changes the impairment-noise realization, and
      // incorporates the global row index so profiles do not share noise.
      const legacyImpairmentSeed = (k + 1) * 2654435761;
      const impairmentSeed = CORPUS_SEED === 20260721
        ? legacyImpairmentSeed
        : Math.imul((items.length + 1) ^ CORPUS_SEED, 2654435761) >>> 0;
      const bytes = impaired
        ? synthesizeImpairedComplexIq(input, imp, impairmentSeed)
        : Uint8Array.from(clean);
      const nuiDraw = drawCaptureNuisances(rand);
      const cleanOut = Uint8Array.from(clean);
      const nui = applyCaptureNuisances(bytes, nuiDraw);
      applyCaptureNuisances(cleanOut, nuiDraw);   // identical receiver-side draw

      writeSync(fdMain, Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength));
      writeSync(fdClean, Buffer.from(cleanOut.buffer, cleanOut.byteOffset, cleanOut.byteLength));
      items.push({
        cls, profile: d.id, sampleRateHz, bandwidthHz, impaired, startSampleIndex,
        cleanPower: meanPower(clean), centerHz: Math.round(centerHz),
        snrDb: Math.round(link.snrDb * 10) / 10, rxPowerDbm: Math.round(link.rxPowerDbm * 10) / 10,
        noiseFigureDb: Math.round(link.noiseFigureDb * 10) / 10,
        centreOffsetFrac: Math.round(nui.centreOffsetFrac * 1000) / 1000,
        adcBits: nui.adcBits, clockErrorPpm: Math.round(nui.clockErrorPpm * 10) / 10,
        multipathTaps: (imp.multipath ?? []).length, impairmentSeed,
      });
    }
  }
}

closeSync(fdMain); closeSync(fdClean);
if (matchedStarts && matchedStarts.length !== items.length) {
  throw new Error(
    `MATCH_STARTS_FROM row count ${matchedStarts.length} does not match ${items.length}`,
  );
}
if (EXACT_TARGET_PER_CLASS) {
  for (const cls of byClass.keys()) {
    const count = items.filter((item) => item.cls === cls).length;
    if (count !== TARGET_PER_CLASS) {
      throw new Error(
        `exact class target failed for ${cls}: ${count} != ${TARGET_PER_CLASS}`,
      );
    }
  }
}
const outDir = outDirEarly;
writeFileSync(outputPaths[2]!, JSON.stringify({
  corpusSeed: CORPUS_SEED,
  signalLabRoot: SIGNAL_LAB_ROOT,
  targetPerClass: TARGET_PER_CLASS,
  exactTargetPerClass: EXACT_TARGET_PER_CLASS,
  startStrideSamples: START_STRIDE_SAMPLES,
  matchedStartsFrom: MATCH_STARTS_FROM,
  sampleCount: SAMPLE_COUNT,
  format: 'cf32le-interleaved',
  classes: [...byClass.keys()].sort(),
  count: items.length,
  nuisanceSpec: NUISANCE_SPEC,
  hardwareBounds: { srMaxHz: SR_MAX, fcMinHz: FC_MIN_HZ, fcMaxHz: FC_MAX_HZ, rxPowerMaxDbm: RX_POWER_MAX_DBM, rxPowerMinDbm: RX_POWER_MIN_DBM },
  hasCleanPairs: true,
  items,
}), { flag: ALLOW_OVERWRITE ? 'w' : 'wx' });

const perClass: Record<string, number> = {};
for (const it of items) perClass[it.cls] = (perClass[it.cls] ?? 0) + 1;
console.log(`wrote ${items.length} realizations (${SAMPLE_COUNT} samples each) to ${outDir}`);
console.log('per class:', perClass);
const stillEmpty = items.filter((it) => it.cleanPower <= OCCUPANCY_MIN_POWER).length;
console.log(`occupancy gate: ${rejectedEmpty} items exhausted ${OCCUPANCY_MAX_TRIES} retries; ${stillEmpty}/${items.length} (${(stillEmpty / items.length * 100).toFixed(2)}%) still have no emission`);
