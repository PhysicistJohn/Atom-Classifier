import { describe, expect, it } from 'vitest';
import { OBSERVABLE_LEAF_CLASSES } from '../observable-classifier-model.js';
import { embeddingEvidenceLikelihood } from './embedding-evidence-fusion.js';
import type { Classification } from './prototype-classifier.js';

function classificationOf(posterior: Record<string, number>, isUnknown = false): Classification {
  const entries = Object.entries(posterior);
  const best = entries.reduce((a, b) => (b[1] > a[1] ? b : a));
  return {
    label: isUnknown ? 'unknown' : best[0],
    index: 0,
    distanceToNearest: isUnknown ? 1 : 0.01,
    isUnknown,
    confidence: best[1],
    posterior,
    distances: [],
  };
}

function sum(r: Record<string, number>): number {
  return Object.values(r).reduce((a, b) => a + b, 0);
}

describe('embedding→Bayesian evidence fusion', () => {
  it('always returns a normalised likelihood over the real leaf taxonomy', () => {
    const r = embeddingEvidenceLikelihood(classificationOf({ ofdm: 1 }), { bandwidthHz: 20e6 });
    expect(Object.keys(r).sort()).toEqual([...OBSERVABLE_LEAF_CLASSES].sort());
    expect(sum(r)).toBeCloseTo(1, 6);
  });

  it('maps GSM and Bluetooth to their own protocol leaves', () => {
    expect(embeddingEvidenceLikelihood(classificationOf({ gsm: 1 }))['gsm-like']!).toBeCloseTo(1, 6);
    expect(embeddingEvidenceLikelihood(classificationOf({ bluetooth: 1 }))['bluetooth-like']!).toBeCloseTo(1, 6);
    expect(embeddingEvidenceLikelihood(classificationOf({ dsss: 1 }))['wifi-hr-dsss-like']!).toBeCloseTo(1, 6);
  });

  it('routes OFDM to cellular vs Wi-Fi by bandwidth', () => {
    const c = classificationOf({ ofdm: 1 });
    const narrow = embeddingEvidenceLikelihood(c, { bandwidthHz: 10e6 }); // LTE-ish, not Wi-Fi
    const lteMass = narrow['lte-fdd-like']! + narrow['lte-tdd-like']!;
    expect(lteMass).toBeGreaterThan(narrow['wifi-ofdm-like']!);

    const wide = embeddingEvidenceLikelihood(c, { bandwidthHz: 40e6 }); // Wi-Fi width
    expect(wide['wifi-ofdm-like']!).toBeGreaterThan(0);
  });

  it('routes an unmapped modulation (no protocol leaf) to unknown-signal', () => {
    const r = embeddingEvidenceLikelihood(classificationOf({ 'some-unmapped-mod': 1 }));
    expect(r['unknown-signal']).toBeCloseTo(1, 6);
  });

  it('abstains (uniform) when the classifier reports open-set unknown', () => {
    const r = embeddingEvidenceLikelihood(classificationOf({ cw: 1 }, true));
    const u = 1 / OBSERVABLE_LEAF_CLASSES.length;
    for (const leaf of OBSERVABLE_LEAF_CLASSES) expect(r[leaf]).toBeCloseTo(u, 9);
  });

  it('maps analog modulations to their analog leaves', () => {
    expect(embeddingEvidenceLikelihood(classificationOf({ cw: 1 }))['cw-like']).toBeCloseTo(1, 6);
    expect(embeddingEvidenceLikelihood(classificationOf({ fm: 1 }))['fm-angle-modulated-like']).toBeCloseTo(1, 6);
  });
});
