import { describe, expect, it } from 'vitest';
import { OrderBelief, type OrderEvidence } from './order-accumulator.js';

const CLASSES = ['qpsk', 'qam16', 'qam64'];
// a per-look evidence favouring qam64 over qam16 by a small (realistic) margin
const favours64: OrderEvidence = { logLik: { qpsk: -2, qam16: -0.3, qam64: 0 }, reliability: 1 };

describe('multi-look order accumulation', () => {
  it('a single look leaves the hard pair undecided', () => {
    const b = new OrderBelief(CLASSES);
    b.update(favours64);
    const d = b.decision(0.9);
    expect(d.order).toBeNull(); // margin 0.3 per look -> confidence well below 0.9
    expect(d.looks).toBe(1);
  });

  it('accumulating looks sharpens the posterior to certainty', () => {
    const b = new OrderBelief(CLASSES);
    for (let i = 0; i < 14; i++) b.update(favours64);
    const d = b.decision(0.9);
    expect(d.order).toBe('qam64');
    expect(d.confidence).toBeGreaterThan(0.9);
    expect(d.looks).toBe(14);
  });

  it('confidence increases monotonically with independent looks', () => {
    const b = new OrderBelief(CLASSES);
    let prev = 0;
    for (let i = 0; i < 10; i++) {
      b.update(favours64);
      const c = b.decision().confidence;
      expect(c).toBeGreaterThanOrEqual(prev - 1e-9);
      prev = c;
    }
  });

  it('deferred / zero-reliability looks do not move the belief', () => {
    const b = new OrderBelief(CLASSES);
    for (let i = 0; i < 20; i++) b.update({ ...favours64, reliability: 0 });
    const p = b.posterior();
    expect(p['qam64']).toBeCloseTo(1 / 3, 6); // stays at the uniform prior
    expect(b.decision().effectiveLooks).toBe(0);
  });

  it('correlation discount slows convergence (n_eff < n)', () => {
    const indep = new OrderBelief(CLASSES);
    const corr = new OrderBelief(CLASSES);
    for (let i = 0; i < 8; i++) {
      indep.update(favours64, 1);
      corr.update(favours64, 0.25);
    }
    expect(indep.decision().effectiveLooks).toBeCloseTo(8, 6);
    expect(corr.decision().effectiveLooks).toBeCloseTo(2, 6);
    expect(indep.decision().confidence).toBeGreaterThan(corr.decision().confidence);
  });

  it('honours a non-uniform prior and keeps posterior normalised', () => {
    const b = new OrderBelief(CLASSES, { qpsk: 0.8, qam16: 0.1, qam64: 0.1 });
    const p = b.posterior();
    expect(p['qpsk']).toBeGreaterThan(0.7);
    expect(Object.values(p).reduce((a, x) => a + x, 0)).toBeCloseTo(1, 9);
  });
});
