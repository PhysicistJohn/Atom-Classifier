import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { refineModulation, type OrderRefinement } from './order-refinement.js';
import { EmbeddingWaveformClassifier } from './embedding-classifier.js';
import type { EmbeddingModel } from './embedding-runtime.js';

describe('hierarchical order refinement', () => {
  it('passes non-order-ambiguous classes through unchanged (incl. bpsk)', () => {
    // bpsk has no order ambiguity and the refiner never emits it, so a confident
    // bpsk call must survive rather than collapse to the generic family.
    for (const label of ['cw', 'am', 'fm', 'ofdm', 'gfsk', 'bpsk', 'unknown']) {
      const r = refineModulation(label);
      expect(r.modulation).toBe(label);
      expect(r.orderResolved).toBe(false);
      expect(r.order).toBeNull();
    }
  });

  it('does not downgrade a confident bpsk even when a stray order result is present', () => {
    const r = refineModulation('bpsk', { deferred: false, order: 'qam64', posterior: { qam64: 0.9 } });
    expect(r.modulation).toBe('bpsk');
    expect(r.orderResolved).toBe(false);
  });

  it('adopts a resolved order for a linear-digital family', () => {
    const refinement: OrderRefinement = {
      deferred: false,
      order: 'qam64',
      posterior: { qpsk: 0.02, qam16: 0.1, qam64: 0.88 },
      quality: 0.2,
      snrDb: 22,
    };
    const r = refineModulation('qam16', refinement); // embedding guessed 16, refiner says 64
    expect(r.orderResolved).toBe(true);
    expect(r.modulation).toBe('qam64');
    expect(r.order).toBe('qam64');
    expect(r.orderConfidence).toBeCloseTo(0.88, 6);
  });

  it('reports family (not the embedding guess) when the order is deferred', () => {
    const deferred: OrderRefinement = { deferred: true, reason: 'recovery-below-gate', snrDb: 5 };
    const r = refineModulation('qam16', deferred);
    expect(r.orderResolved).toBe(false);
    expect(r.modulation).toBe('linear-digital');
    expect(r.order).toBeNull();
    expect(r.deferReason).toBe('recovery-below-gate');
  });

  it('reports family when no refinement is available', () => {
    const r = refineModulation('qpsk');
    expect(r.modulation).toBe('linear-digital');
    expect(r.orderResolved).toBe(false);
    expect(r.deferReason).toBe('no-order-refinement');
  });

  it('flows through the classifier end-to-end', () => {
    const assets = new URL('./assets/', import.meta.url);
    const model = JSON.parse(readFileSync(new URL('embedding-weights.json', assets), 'utf8')) as EmbeddingModel;
    const proto = JSON.parse(readFileSync(new URL('prototypes.json', assets), 'utf8'));
    const clf = new EmbeddingWaveformClassifier(model, proto);
    // a clean CW tone -> passthrough modulation, no order machinery engaged
    const N = 4096;
    const re = new Float64Array(N);
    const im = new Float64Array(N);
    for (let k = 0; k < N; k++) {
      re[k] = Math.cos(2 * Math.PI * 0.011 * k);
      im[k] = Math.sin(2 * Math.PI * 0.011 * k);
    }
    const res = clf.classifyIq(re, im, {
      orderRefinement: { deferred: false, order: 'qam64', posterior: { qam64: 0.9 } },
    });
    // CW is not linear-digital, so the (irrelevant) order refinement is ignored
    expect(res.modulation.modulation).toBe('cw');
    expect(res.modulation.orderResolved).toBe(false);
  });
});
