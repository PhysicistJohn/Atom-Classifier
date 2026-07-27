import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { MAG_PARAMS, magnitudeFromIq } from './magnitude-preprocess.js';

const fixture = JSON.parse(
  readFileSync(
    new URL('./test-fixtures/iq-preprocess-hybrid-v3.json', import.meta.url),
    'utf8',
  ),
) as {
  cases: Array<{
    name: string;
    input_i: number[];
    input_q: number[];
    gains: number[];
    magnitude_shape: number[];
    magnitude_features: number[];
  }>;
};

describe('magnitude-from-I/Q hybrid-v3 Python parity', () => {
  for (const entry of fixture.cases) {
    it(`keeps circular spectral geometry and gain invariance for ${entry.name}`, () => {
      for (const gain of entry.gains) {
        const re = Float64Array.from(entry.input_i, (value) => value * gain);
        const im = Float64Array.from(entry.input_q, (value) => value * gain);
        const actual = magnitudeFromIq(re, im, MAG_PARAMS);
        let worst = 0;
        for (let k = 0; k < actual.shape.length; k++) {
          worst = Math.max(worst, Math.abs(actual.shape[k]! - entry.magnitude_shape[k]!));
        }
        for (let k = 0; k < actual.features.length; k++) {
          worst = Math.max(
            worst,
            Math.abs(actual.features[k]! - entry.magnitude_features[k]!),
          );
        }
        expect(worst).toBeLessThan(3e-6);
      }
    });
  }
});
