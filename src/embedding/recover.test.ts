import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { recoverConstellation } from './recover.js';

interface Case {
  name: string;
  re: number[];
  im: number[];
  spsHint: number | null;
  expected: {
    symbolsRe: number[];
    symbolsIm: number[];
    sps: number;
    residualIsi: number;
    snrDb: number;
  };
}

const assets = new URL('./assets/', import.meta.url);
const fixture = JSON.parse(readFileSync(new URL('recover-parity-fixture.json', assets), 'utf8')) as {
  cases: Case[];
};

// The CMA equalizer is a deterministic feedback loop. numpy's np.dot over the
// 21-tap complex vector accumulates in the same sequential order as the TS loop,
// so parity is near machine-exact: measured worst symbol abs error ~5e-15 and
// scalar deltas ~1e-13 across all four cases. We assert 1e-9 on the symbols to
// leave headroom for cross-platform ulp differences in atan2/cos (carrier lock),
// and the task's 1e-3 on the scalar diagnostics (actual ~1e-13). sps is an exact
// integer match.
const SYMBOL_ABS_TOL = 1e-9;
const SCALAR_TOL = 1e-3;

describe('blind symbol recovery — Python parity', () => {
  it('has the expected fixture cases', () => {
    expect(fixture.cases.map((c) => c.name)).toEqual(['qpsk', 'qam16', 'noise', 'cw']);
  });

  for (const c of fixture.cases) {
    it(`recovers "${c.name}" to numpy parity`, () => {
      const re = Float64Array.from(c.re);
      const im = Float64Array.from(c.im);
      const spsHint = c.spsHint === null ? undefined : c.spsHint;
      const out = recoverConstellation(re, im, spsHint);

      // sps is an integer decision — must match exactly.
      expect(out.sps).toBe(c.expected.sps);

      // single-pass scalar diagnostics — tight.
      expect(Math.abs(out.residualIsi - c.expected.residualIsi)).toBeLessThan(SCALAR_TOL);
      expect(Math.abs(out.snrDb - c.expected.snrDb)).toBeLessThan(SCALAR_TOL);

      // recovered constellation — same length, small abs error.
      expect(out.symbolsRe.length).toBe(c.expected.symbolsRe.length);
      expect(out.symbolsIm.length).toBe(c.expected.symbolsIm.length);
      let worst = 0;
      for (let k = 0; k < out.symbolsRe.length; k++) {
        worst = Math.max(worst, Math.abs(out.symbolsRe[k]! - c.expected.symbolsRe[k]!));
        worst = Math.max(worst, Math.abs(out.symbolsIm[k]! - c.expected.symbolsIm[k]!));
      }
      expect(worst).toBeLessThan(SYMBOL_ABS_TOL);
    });
  }
});
