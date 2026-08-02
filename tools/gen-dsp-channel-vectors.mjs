/**
 * Generate golden conformance vectors for the numpy port of Atom-DSP's
 * propagation-channel model (tools/dsp_channel.py).
 *
 * Atom-DSP's TypeScript is the source of truth; this script evaluates it
 * directly (sibling checkout) and pins outputs to JSON. The Python parity
 * test (tools/test_dsp_channel_parity.py) replays these vectors and must
 * match: integer-exact for the hash, <=1e-12 relative for floats.
 *
 * Run: npm run generate:dsp-channel-vectors
 * Regenerate only when Atom-DSP's channel contract intentionally changes,
 * and record the Atom-DSP revision below.
 */
import { writeFileSync } from 'node:fs';
import { execSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { resolve, dirname } from 'node:path';

import {
  channelHash32,
  seededComplexGaussian,
  resolveTdlTaps,
  fadingGainAtTime,
  noiseStandardDeviation,
  applyChannelAtIndex,
  thermalNoiseFloorDbm,
  TDL_PROFILES,
} from '../../Atom-DSP/src/channel.ts';

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT = resolve(HERE, 'dsp-channel-conformance-vectors.json');

const dspRevision = execSync('git -C ../../Atom-DSP rev-parse HEAD', {
  cwd: HERE, encoding: 'utf8',
}).trim();
const dspDirty = execSync('git -C ../../Atom-DSP status --porcelain', {
  cwd: HERE, encoding: 'utf8',
}).trim().length > 0;

// Deterministic synthetic source: a closed-form function of the absolute
// index, so both languages can evaluate it without shared state.
function sourceAt(index) {
  const angle = 2 * Math.PI * 0.03125 * index;
  const envelope = 0.5 + 0.25 * Math.cos(2 * Math.PI * index / 1000);
  return [envelope * Math.cos(angle), envelope * Math.sin(angle)];
}

const hashCases = [];
for (const [seed, index, lane] of [
  [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
  [20260731, 12345, 0x6e6f6973], [0xffffffff, 0xffffffff, 0xffffffff],
  [123456789, 2 ** 40 + 7, 0x74686574], [42, 2 ** 52, 99],
]) {
  hashCases.push({ seed, index, lane, value: channelHash32(seed, index, lane) });
}

const gaussianCases = [];
for (const [seed, sample, lane] of [
  [1, 0, 0], [20260731, 999983, 0], [7, 2 ** 33 + 11, 0], [7, 123, 5],
]) {
  gaussianCases.push({ seed, sample, lane, value: seededComplexGaussian(seed, sample, lane) });
}

const tdlCases = [];
for (const [profile, ds, fs] of [
  ['tdl-a', 100e-9, 20e6], ['tdl-a', 300e-9, 20e6], ['tdl-b', 30e-9, 20e6],
  ['tdl-b', 1000e-9, 20e6], ['tdl-d', 300e-9, 20e6], ['tdl-a', 0, 20e6],
]) {
  tdlCases.push({ profile, delaySpreadSeconds: ds, sampleRateHz: fs, taps: resolveTdlTaps(profile, ds, fs) });
}

const fadingCases = [];
for (const [kind, kFactorDb, dopplerHz, seed, tapIndex, t] of [
  ['rayleigh', null, 50, 20260731, 0, 0],
  ['rayleigh', null, 50, 20260731, 0, 0.0123],
  ['rayleigh', null, 200, 7, 3, 0.02],
  ['rayleigh', null, 0, 7, 1, 0.5],
  ['rician', 6, 30, 99, 0, 0.011],
  ['rician', 12, 300, 20260731, 5, 0.00042],
]) {
  const configuration = kind === 'rician'
    ? { kind, kFactorDb, dopplerHz }
    : { kind, dopplerHz };
  fadingCases.push({
    kind, kFactorDb, dopplerHz, seed, tapIndex, timeSeconds: t,
    value: fadingGainAtTime(configuration, seed, tapIndex, t),
  });
}

const noiseCases = [
  { noiseFloorDbm: -20, fullScaleDbm: 0 },
  { noiseFloorDbm: -3, fullScaleDbm: 3 },
  { noiseFloorDbm: thermalNoiseFloorDbm(20e6, 5), fullScaleDbm: -30 },
].map((c) => ({ ...c, value: noiseStandardDeviation(c) }));

// End-to-end channel application, including a split-invariance witness:
// indices 5000..5063 evaluated one-by-one must equal the same indices
// evaluated as part of any larger sweep (trivially true here because the
// function is closed-form -- the vector pins the actual numbers).
const endToEnd = [];
for (const [name, configuration] of [
  ['static-multipath', {
    noiseFloorDbm: -25, seed: 20260731, sampleRateHz: 20_000_000,
    taps: resolveTdlTaps('tdl-a', 200e-9, 20e6),
  }],
  ['rayleigh-fading', {
    noiseFloorDbm: -30, seed: 42, sampleRateHz: 20_000_000,
    taps: resolveTdlTaps('tdl-b', 100e-9, 20e6),
    fading: { kind: 'rayleigh', dopplerHz: 120 },
  }],
  ['rician-los', {
    noiseFloorDbm: -35, fullScaleDbm: 0, seed: 7, sampleRateHz: 20_000_000,
    taps: resolveTdlTaps('tdl-d', 300e-9, 20e6),
    fading: { kind: 'rician', kFactorDb: 9, dopplerHz: 40 },
  }],
  ['noise-only', { noiseFloorDbm: -20, seed: 1, sampleRateHz: 20_000_000 }],
]) {
  const indices = [0, 1, 2, 3, 17, 5000, 5001, 5031, 5063, 399_999];
  endToEnd.push({
    name,
    configuration,
    indices,
    values: indices.map((i) => applyChannelAtIndex(sourceAt, i, configuration)),
  });
}

const vectors = {
  schema: 'atom-classifier.dsp-channel-conformance.v1',
  atomDspRevision: dspRevision,
  atomDspWorkingTreeClean: !dspDirty,
  source: 'sourceAt(i) = e(i)*exp(j*2*pi*i/32), e(i) = 0.5 + 0.25*cos(2*pi*i/1000)',
  tdlProfiles: Object.fromEntries(
    Object.entries(TDL_PROFILES).map(([k, taps]) => [k, taps]),
  ),
  hashCases, gaussianCases, tdlCases, fadingCases, noiseCases, endToEnd,
};

writeFileSync(OUT, `${JSON.stringify(vectors, null, 1)}\n`);
console.log(`wrote ${OUT} (Atom-DSP ${dspRevision.slice(0, 12)}${dspDirty ? ' DIRTY' : ''})`);
