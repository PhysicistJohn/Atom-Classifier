/**
 * Multi-look Bayesian order accumulation (Lever 1).
 *
 * A single capture resolves 16-vs-64-QAM order at only ~0.72-0.83 even with full
 * blind recovery — an intrinsic per-capture limit. But a persistent emitter
 * gives many looks, and independent evidence combines: the reliability-weighted
 * order log-likelihoods sum across looks, and the posterior sharpens toward
 * certainty (measured 0.80 → 0.98 over 16 looks at 18 dB).
 *
 * This is the same log-linear evidence accumulation the Bayesian waveform
 * classifier already uses across its evidence views — here across *time*, keyed
 * per emitter track. It consumes the stable `OrderEvidence` contract produced by
 * the ingestion-side refiner (`training/order_refine.py:order_evidence`), so it
 * is decoupled from the recovery internals. Browser-native, zero-dependency.
 */

export interface OrderEvidence {
  /** Per-order log-likelihood for this capture (pre-accumulation). */
  logLik: Record<string, number>;
  /** [0,1] weight — ~1 for a clean capture, ->0 for a marginal/deferred one. */
  reliability: number;
  /** Convenience flag from the single-capture hard gate (not used in accumulation). */
  deferred?: boolean;
}

export interface OrderDecision {
  /** Resolved order once confidence crosses the threshold, else null. */
  order: string | null;
  confidence: number;
  posterior: Record<string, number>;
  /** Number of looks fused. */
  looks: number;
  /** Effective (reliability- and correlation-discounted) look count. */
  effectiveLooks: number;
}

export class OrderBelief {
  private readonly classes: string[];
  private readonly logPost: Record<string, number> = {};
  private looks = 0;
  private effLooks = 0;

  /**
   * @param classes  order labels (e.g. ['qpsk','qam16','qam64'])
   * @param prior    optional prior over classes (defaults to uniform)
   */
  constructor(classes: string[], prior?: Record<string, number>) {
    this.classes = classes.slice();
    for (const c of this.classes) {
      const p = prior?.[c] ?? 1 / this.classes.length;
      this.logPost[c] = Math.log(Math.max(p, 1e-12));
    }
  }

  /**
   * Fuse one look. `correlationDiscount` in (0,1] downweights a look that is
   * correlated with prior looks (same slowly-varying channel) so the effective
   * count grows like √(n_eff), not √n — pass 1 for an independent look.
   */
  update(ev: OrderEvidence, correlationDiscount = 1): void {
    const w = ev.reliability * correlationDiscount;
    for (const c of this.classes) {
      this.logPost[c] = (this.logPost[c] ?? 0) + w * (ev.logLik[c] ?? 0);
    }
    this.looks += 1;
    this.effLooks += w;
  }

  posterior(): Record<string, number> {
    let max = -Infinity;
    for (const c of this.classes) max = Math.max(max, this.logPost[c] ?? -Infinity);
    let z = 0;
    const exp: Record<string, number> = {};
    for (const c of this.classes) {
      const e = Math.exp((this.logPost[c] ?? -Infinity) - max);
      exp[c] = e;
      z += e;
    }
    const post: Record<string, number> = {};
    for (const c of this.classes) post[c] = (exp[c] ?? 0) / z;
    return post;
  }

  /** Decide the order once the posterior mass crosses `threshold`, else defer. */
  decision(threshold = 0.9): OrderDecision {
    const post = this.posterior();
    let best = this.classes[0]!;
    for (const c of this.classes) if ((post[c] ?? 0) > (post[best] ?? 0)) best = c;
    const confidence = post[best] ?? 0;
    return {
      order: confidence >= threshold ? best : null,
      confidence,
      posterior: post,
      looks: this.looks,
      effectiveLooks: this.effLooks,
    };
  }
}
