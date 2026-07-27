import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { preprocess, iqFeatures, DEFAULT_PARAMS, estimateBand } from './iq-preprocess.js';

const assets = new URL('./assets/', import.meta.url);
const fixture = JSON.parse(readFileSync(new URL('parity-fixture.json', assets), 'utf8'));
const hybridFixture = JSON.parse(
  readFileSync(
    new URL('./test-fixtures/iq-preprocess-hybrid-v3.json', import.meta.url),
    'utf8',
  ),
) as {
  preprocess_version: string;
  cases: Array<{
    name: string;
    input_i: number[];
    input_q: number[];
    gains: number[];
    expected_i: number[];
    expected_q: number[];
    center: number;
    bw: number;
    resample_frac: number;
    route: 'linear-v1' | 'circular-v2' | 'full-band-fallback';
    linear_center: number;
    linear_bw: number;
    circular_center: number;
    circular_bw: number;
    true_center: number | null;
    true_bw: number | null;
  }>;
};
const LEGACY_PARAMS = { ...DEFAULT_PARAMS, version: 'linear-v1' as const };
const CIRCULAR_PARAMS = { ...DEFAULT_PARAMS, version: 'circular-v2' as const };

function splitComplex(flat: number[], len: number): { re: Float64Array; im: Float64Array } {
  return {
    re: Float64Array.from(flat.slice(0, len)),
    im: Float64Array.from(flat.slice(len, 2 * len)),
  };
}

describe('iq-preprocess linear-v1 asset compatibility', () => {
  for (const entry of fixture.preprocess as Array<{
    class: string;
    iq: number[];
    iq_len: number;
    expected: number[];
    features: number[];
    center: number;
    bw: number;
  }>) {
    it(`matches detect/normalise for ${entry.class}`, () => {
      const { re, im } = splitComplex(entry.iq, entry.iq_len);

      const band = estimateBand(re, im, LEGACY_PARAMS);
      expect(band.center).toBeCloseTo(entry.center, 6);
      expect(band.bw).toBeCloseTo(entry.bw, 6);

      const n = preprocess(re, im, LEGACY_PARAMS);
      const L = DEFAULT_PARAMS.lOut;
      let maxCh = 0;
      for (let k = 0; k < L; k++) {
        maxCh = Math.max(maxCh, Math.abs(n.i[k]! - entry.expected[k]!));
        maxCh = Math.max(maxCh, Math.abs(n.q[k]! - entry.expected[L + k]!));
      }
      expect(maxCh).toBeLessThan(1e-3);
    });

    it(`matches cumulant features for ${entry.class}`, () => {
      const L = DEFAULT_PARAMS.lOut;
      const i = Float64Array.from(entry.expected.slice(0, L));
      const q = Float64Array.from(entry.expected.slice(L, 2 * L));
      const feat = iqFeatures(i, q);
      let maxF = 0;
      for (let f = 0; f < feat.length; f++) {
        maxF = Math.max(maxF, Math.abs(feat[f]! - entry.features[f]!));
      }
      expect(maxF).toBeLessThan(1e-4);
    });
  }
});

describe('iq-preprocess hybrid-v3 Python parity', () => {
  it('declares the hybrid-v3 contract explicitly', () => {
    expect(DEFAULT_PARAMS.version).toBe('hybrid-v3');
    expect(hybridFixture.preprocess_version).toBe('hybrid-v3');
  });

  for (const entry of hybridFixture.cases) {
    it(`matches band, normalized waveform, and gains for ${entry.name}`, () => {
      const baseI = Float64Array.from(entry.input_i);
      const baseQ = Float64Array.from(entry.input_q);
      for (const gain of entry.gains) {
        const re = Float64Array.from(baseI, (value) => value * gain);
        const im = Float64Array.from(baseQ, (value) => value * gain);
        const band = estimateBand(re, im, DEFAULT_PARAMS);
        expect(band.center).toBeCloseTo(entry.center, 12);
        expect(band.bw).toBeCloseTo(entry.bw, 12);

        const normalized = preprocess(re, im, DEFAULT_PARAMS);
        expect(normalized.center).toBeCloseTo(entry.center, 12);
        expect(normalized.bw).toBeCloseTo(entry.bw, 12);
        expect(normalized.resampleFrac).toBeCloseTo(entry.resample_frac, 12);
        let maxError = 0;
        for (let k = 0; k < DEFAULT_PARAMS.lOut; k++) {
          maxError = Math.max(
            maxError,
            Math.abs(normalized.i[k]! - entry.expected_i[k]!),
            Math.abs(normalized.q[k]! - entry.expected_q[k]!),
          );
        }
        expect(maxError).toBeLessThan(3e-6);
      }
    });

    it(`matches candidate routing for ${entry.name}`, () => {
      const baseI = Float64Array.from(entry.input_i);
      const baseQ = Float64Array.from(entry.input_q);
      const linear = estimateBand(baseI, baseQ, LEGACY_PARAMS);
      const circular = estimateBand(baseI, baseQ, CIRCULAR_PARAMS);
      expect(linear.center).toBeCloseTo(entry.linear_center, 12);
      expect(linear.bw).toBeCloseTo(entry.linear_bw, 12);
      expect(circular.center).toBeCloseTo(entry.circular_center, 12);
      expect(circular.bw).toBeCloseTo(entry.circular_bw, 12);
    });
  }

  it('preserves normal-band linear-v1 geometry and waveform', () => {
    const entry = hybridFixture.cases.find((candidate) => candidate.name === 'normal-band')!;
    expect(entry.route).toBe('linear-v1');
    const baseI = Float64Array.from(entry.input_i);
    const baseQ = Float64Array.from(entry.input_q);
    const hybridBand = estimateBand(baseI, baseQ, DEFAULT_PARAMS);
    const linearBand = estimateBand(baseI, baseQ, LEGACY_PARAMS);
    expect(hybridBand).toEqual(linearBand);

    const hybrid = preprocess(baseI, baseQ, DEFAULT_PARAMS);
    const linear = preprocess(baseI, baseQ, LEGACY_PARAMS);
    let worst = 0;
    for (let k = 0; k < DEFAULT_PARAMS.lOut; k++) {
      worst = Math.max(
        worst,
        Math.abs(hybrid.i[k]! - linear.i[k]!),
        Math.abs(hybrid.q[k]! - linear.q[k]!),
      );
    }
    expect(worst).toBeLessThan(3e-6);
  });
});
