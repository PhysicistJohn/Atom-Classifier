import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { preprocess, iqFeatures, DEFAULT_PARAMS, estimateBand } from './iq-preprocess.js';

const assets = new URL('./assets/', import.meta.url);
const fixture = JSON.parse(readFileSync(new URL('parity-fixture.json', assets), 'utf8'));

function splitComplex(flat: number[], len: number): { re: Float64Array; im: Float64Array } {
  return {
    re: Float64Array.from(flat.slice(0, len)),
    im: Float64Array.from(flat.slice(len, 2 * len)),
  };
}

describe('iq-preprocess parity with the Python front-end', () => {
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

      const band = estimateBand(re, im, DEFAULT_PARAMS);
      expect(band.center).toBeCloseTo(entry.center, 6);
      expect(band.bw).toBeCloseTo(entry.bw, 6);

      const n = preprocess(re, im, DEFAULT_PARAMS);
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
