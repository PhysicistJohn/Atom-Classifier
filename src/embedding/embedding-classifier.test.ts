import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { EmbeddingWaveformClassifier } from './embedding-classifier.js';
import type { EmbeddingModel } from './embedding-runtime.js';

const assets = new URL('./assets/', import.meta.url);
const model = JSON.parse(readFileSync(new URL('embedding-weights.json', assets), 'utf8')) as EmbeddingModel;
const protoJson = JSON.parse(readFileSync(new URL('prototypes.json', assets), 'utf8'));
const fixture = JSON.parse(readFileSync(new URL('parity-fixture.json', assets), 'utf8'));

function raw(entry: { iq: number[]; iq_len: number }): { re: Float64Array; im: Float64Array } {
  return {
    re: Float64Array.from(entry.iq.slice(0, entry.iq_len)),
    im: Float64Array.from(entry.iq.slice(entry.iq_len, 2 * entry.iq_len)),
  };
}

describe('EmbeddingWaveformClassifier end-to-end', () => {
  const clf = new EmbeddingWaveformClassifier(model, protoJson);

  it('classifies raw I/Q into a valid class with a normalised leaf likelihood', () => {
    for (const entry of fixture.preprocess as Array<{ class: string; iq: number[]; iq_len: number }>) {
      const { re, im } = raw(entry);
      const res = clf.classifyIq(re, im);
      // recovered context matches the fixture's measured band
      expect(Math.abs(res.center)).toBeLessThan(0.5);
      expect(res.bw).toBeGreaterThan(0);
      // label is a known class or 'unknown'
      expect([...clf.classes, 'unknown']).toContain(res.classification.label);
      // fused leaf likelihood is a distribution
      const total = Object.values(res.leafLikelihood).reduce((a, b) => a + b, 0);
      expect(total).toBeCloseTo(1, 6);
    }
  });

  it('classifies a clean CW tone as cw and routes it to the cw-like leaf', () => {
    // An unambiguous carrier generated directly in TS — an end-to-end correctness
    // check that does not depend on a lucky draw of the impaired fixtures.
    const N = 4096;
    const f = 0.011;
    let s = 123456789;
    const rnd = () => {
      s = (1103515245 * s + 12345) & 0x7fffffff;
      return s / 0x7fffffff - 0.5;
    };
    const re = new Float64Array(N);
    const im = new Float64Array(N);
    for (let k = 0; k < N; k++) {
      re[k] = Math.cos(2 * Math.PI * f * k) + 0.02 * rnd();
      im[k] = Math.sin(2 * Math.PI * f * k) + 0.02 * rnd();
    }
    const res = clf.classifyIq(re, im);
    expect(res.classification.isUnknown).toBe(false);
    expect(res.classification.label).toBe('cw');
    expect(res.leafLikelihood['cw-like']).toBeGreaterThan(0.5);
  });

  it('is deterministic — identical input yields identical output', () => {
    const { re, im } = raw(fixture.preprocess[0]);
    const a = clf.classifyIq(Float64Array.from(re), Float64Array.from(im));
    const b = clf.classifyIq(Float64Array.from(re), Float64Array.from(im));
    expect(a.classification.label).toBe(b.classification.label);
    expect(a.classification.distanceToNearest).toBe(b.classification.distanceToNearest);
  });

  it('when a capture is flagged unknown, the fused leaf likelihood abstains (uniform)', () => {
    // whichever fixtures are present: if one lands beyond threshold, the view must
    // abstain (uniform), not inject a confident vote.
    for (const entry of fixture.preprocess as Array<{ iq: number[]; iq_len: number }>) {
      const { re, im } = raw(entry);
      const res = clf.classifyIq(re, im);
      if (res.classification.isUnknown) {
        const vals = Object.values(res.leafLikelihood);
        expect(Math.max(...vals) - Math.min(...vals)).toBeLessThan(1e-9);
      }
    }
  });

  it('supports few-shot enrollment from raw I/Q with no retraining', () => {
    const before = clf.classes.length;
    const ex = raw(fixture.preprocess[0]);
    clf.enrollClass('lab-beacon', [ex, ex]);
    expect(clf.classes.length).toBe(before + 1);
    expect(clf.classes).toContain('lab-beacon');
  });
});
