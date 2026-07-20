import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { classify, enroll, loadPrototypeSet } from './prototype-classifier.js';

const assets = new URL('./assets/', import.meta.url);
const protoJson = JSON.parse(readFileSync(new URL('prototypes.json', assets), 'utf8'));

function unit(v: number[]): Float64Array {
  const a = Float64Array.from(v);
  let n = 0;
  for (const x of a) n += x * x;
  n = Math.sqrt(n) || 1;
  for (let i = 0; i < a.length; i++) a[i] = a[i]! / n;
  return a;
}

describe('prototype classification, open-set, and few-shot enrollment', () => {
  const set = loadPrototypeSet(protoJson);

  it('assigns each prototype to its own class with high confidence', () => {
    for (let c = 0; c < set.classes.length; c++) {
      const r = classify(set, set.prototypes[c]!);
      expect(r.label).toBe(set.classes[c]);
      expect(r.isUnknown).toBe(false);
      expect(r.distanceToNearest).toBeLessThan(1e-6);
      expect(r.confidence).toBeGreaterThan(0.5);
    }
  });

  it('posterior is a valid distribution', () => {
    const r = classify(set, set.prototypes[0]!);
    const total = Object.values(r.posterior).reduce((a, b) => a + b, 0);
    expect(total).toBeCloseTo(1, 6);
  });

  it('flags a far-from-everything embedding as unknown (open-set)', () => {
    // an embedding maximally far from prototype 0 (its antipode) lands well
    // beyond the calibrated distance threshold
    const far = Float64Array.from(set.prototypes[0]!, (x) => -x);
    const r = classify(set, far);
    expect(r.distanceToNearest).toBeGreaterThan(set.unknownThreshold);
    expect(r.isUnknown).toBe(true);
    expect(r.label).toBe('unknown');
  });

  it('enrolls a new class from examples without disturbing existing ones', () => {
    const before = set.classes.length;
    // a synthetic new-class direction not aligned with any existing prototype
    const dim = set.embedDim;
    const a = unit(Array.from({ length: dim }, (_, i) => Math.sin(i * 1.3) + 0.2));
    const b = unit(Array.from({ length: dim }, (_, i) => Math.sin(i * 1.3) + 0.25));
    const set2 = enroll(set, 'test-mode', [a, b]);
    expect(set2.classes.length).toBe(before + 1);
    expect(set2.classes).toContain('test-mode');
    // original set is untouched (no mutation)
    expect(set.classes.length).toBe(before);
    // a query near the enrolled examples classifies as the new class
    const r = classify(set2, a);
    expect(r.label).toBe('test-mode');
  });

  it('re-enrolling an existing class replaces its prototype in place', () => {
    const name = set.classes[0]!;
    const replacement = unit(Array.from({ length: set.embedDim }, (_, i) => Math.cos(i)));
    const set2 = enroll(set, name, [replacement]);
    expect(set2.classes.length).toBe(set.classes.length);
    const r = classify(set2, replacement);
    expect(r.label).toBe(name);
  });
});
