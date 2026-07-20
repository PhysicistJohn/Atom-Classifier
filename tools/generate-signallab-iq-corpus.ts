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

import { mkdirSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { synthesizeAnalyticComplexIq } from '../../Atom-SignalLab/src/complex-iq.js';
import { synthesizeImpairedComplexIq, type ReceiverImpairments } from '../../Atom-SignalLab/src/impairments.js';
import { waveformCatalog } from '../../Atom-SignalLab/src/waveforms.js';

const SAMPLE_COUNT = 4096;
const TARGET_PER_CLASS = 260;
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

function drawImpairments(rand: () => number): ReceiverImpairments {
  const multipath = rand() < 0.6
    ? [{ delay: 2 + Math.floor(rand() * 8), gainInPhase: (rand() - 0.5) * 0.6, gainQuadrature: (rand() - 0.5) * 0.6 }]
    : [];
  return {
    snrDb: 10 + rand() * 30, // 10-40 dB
    carrierFrequencyOffset: (rand() - 0.5) * 0.008,
    phaseNoiseStd: rand() * 0.02,
    iqGainImbalance: (rand() - 0.5) * 0.16,
    iqPhaseImbalance: (rand() - 0.5) * 0.2,
    dcInPhase: (rand() - 0.5) * 0.06,
    dcQuadrature: (rand() - 0.5) * 0.06,
    multipath,
    ...(rand() < 0.3 ? { paSaturation: 0.6 + rand() * 0.3 } : {}),
  };
}
// Common Atomizer I/Q sample rates — narrowband signals are heavily oversampled
// at these, which is exactly the geometry the app feeds; include it so the
// embedding is robust to it (not just the well-matched occupancy sweep).
const APP_SAMPLE_RATES = [2_000_000, 8_000_000, 30_000_000];
const SR_MIN = 1_000_000;
const SR_MAX = 245_760_000;

type Descriptor = (typeof waveformCatalog)[number];

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

const blocks: Buffer[] = [];
const items: { cls: string; profile: string; sampleRateHz: number; bandwidthHz: number; impaired: boolean }[] = [];
const rand = mulberry32(20260721);

for (const [cls, profiles] of byClass) {
  const perProfile = Math.max(8, Math.round(TARGET_PER_CLASS / profiles.length));
  for (const d of profiles) {
    // sample rates to sweep: the occupancy-matched set + the app's common rates
    // (where they are wide enough to actually carry the signal).
    const occ = d.occupiedBandwidthHz;
    const rates = [
      ...TARGET_FRACS.map((tf) => clampInt(occ / tf, SR_MIN, SR_MAX)),
      ...APP_SAMPLE_RATES.filter((r) => r >= occ * 1.15 && r <= SR_MAX),
    ];
    for (let k = 0; k < perProfile; k++) {
      const sampleRateHz = rates[k % rates.length]!;
      const bandwidthHz = clampInt(occ * 1.15, 1_000, Math.floor(sampleRateHz * 0.95));
      const input = { profile: d.id, sampleRateHz, bandwidthHz, sampleCount: SAMPLE_COUNT, startSampleIndex: k * SAMPLE_COUNT };
      const impaired = k % CLEAN_EVERY !== 0;
      const bytes = impaired
        ? synthesizeImpairedComplexIq(input, drawImpairments(rand), (k + 1) * 2654435761)
        : synthesizeAnalyticComplexIq(input);
      blocks.push(Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength));
      items.push({ cls, profile: d.id, sampleRateHz, bandwidthHz, impaired });
    }
  }
}

const outDir = join(dirname(fileURLToPath(import.meta.url)), '..', 'training', 'artifacts', 'signallab-corpus');
mkdirSync(outDir, { recursive: true });
writeFileSync(join(outDir, 'corpus.f32'), Buffer.concat(blocks));
writeFileSync(join(outDir, 'corpus.json'), JSON.stringify({
  sampleCount: SAMPLE_COUNT,
  format: 'cf32le-interleaved',
  classes: [...byClass.keys()].sort(),
  count: items.length,
  items,
}));

const perClass: Record<string, number> = {};
for (const it of items) perClass[it.cls] = (perClass[it.cls] ?? 0) + 1;
console.log(`wrote ${items.length} realizations (${SAMPLE_COUNT} samples each) to ${outDir}`);
console.log('per class:', perClass);
