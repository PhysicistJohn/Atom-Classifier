/**
 * Hierarchical modulation-order refinement (browser side of the push-through).
 *
 * The embedding resolves modulation *family* but not linear-digital *order*
 * (16- vs 64-QAM is an information limit at its front-end fidelity). The
 * ingestion pipeline (`training/recover.py` + `training/order_refine.py`) blindly
 * equalizes + synchronizes the capture and classifies the order — BUT only when
 * the recovered constellation supports the call; otherwise it defers. That heavy
 * feedback DSP stays in Python (it can't be made bit-exact in the browser), and
 * its result flows here as data.
 *
 * This module combines the two: when the embedding lands on a linear-digital
 * class and a non-deferred order result is present, the reliable order call wins;
 * when the order is deferred (or absent), the modulation is reported at the
 * family level ("linear-digital, order-unresolved") rather than trusting the
 * embedding's unreliable order guess. Analog / OFDM / FSK families pass through
 * unchanged.
 */

/** Result produced by the ingestion-side quality-gated order refiner. */
export interface OrderRefinement {
  deferred: boolean;
  order?: 'qpsk' | 'qam16' | 'qam64' | string;
  posterior?: Record<string, number>;
  quality?: number;
  snrDb?: number;
  reason?: string;
}

export interface RefinedModulation {
  /** Final modulation label: a resolved order, a family, or the passthrough class. */
  modulation: string;
  /** True only when a specific linear-digital order was resolved. */
  orderResolved: boolean;
  /** The resolved order, when any. */
  order: string | null;
  /** Confidence in the resolved order (order posterior), when resolved. */
  orderConfidence?: number;
  /** Why the order was not resolved, when applicable. */
  deferReason?: string;
}

/**
 * Embedding classes whose order the refiner can actually resolve. BPSK is
 * intentionally excluded: it has no order ambiguity and the refiner never
 * produces it, so a confident 'bpsk' call must pass through unchanged rather
 * than be downgraded to the generic family.
 */
export const LINEAR_DIGITAL_CLASSES = new Set(['qpsk', 'qam16', 'qam64']);

/** Family label used when the order is unresolved. */
export const LINEAR_DIGITAL_FAMILY = 'linear-digital';

/**
 * Combine the embedding's family label with an optional ingestion-side order
 * refinement into a single hierarchical modulation decision.
 */
export function refineModulation(
  embeddingLabel: string,
  refinement?: OrderRefinement,
): RefinedModulation {
  // Non-linear-digital families (analog, OFDM, FSK, unknown) are unaffected.
  if (!LINEAR_DIGITAL_CLASSES.has(embeddingLabel)) {
    return { modulation: embeddingLabel, orderResolved: false, order: null };
  }
  // Linear-digital: prefer the reliable, quality-gated order call.
  if (refinement && !refinement.deferred && refinement.order) {
    return {
      modulation: refinement.order,
      orderResolved: true,
      order: refinement.order,
      orderConfidence: refinement.posterior?.[refinement.order],
    };
  }
  // Deferred or absent: report the family, not the embedding's unreliable guess.
  return {
    modulation: LINEAR_DIGITAL_FAMILY,
    orderResolved: false,
    order: null,
    deferReason: refinement?.reason ?? 'no-order-refinement',
  };
}
