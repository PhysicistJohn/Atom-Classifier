/**
 * Fuse the embedding's modulation-family posterior into the Bayesian model's
 * leaf-class taxonomy as an independent evidence-view likelihood.
 *
 * The embedding names the *modulation*; the leaf taxonomy is *protocol*-shaped.
 * Several protocols are modulation-degenerate — GFSK covers both GSM and
 * Bluetooth; OFDM covers LTE/NR/Wi-Fi — so the mapping resolves them with the
 * measured occupied bandwidth (in Hz, from the SDR detection), exactly the role
 * the band-context support mask already plays in the Bayesian model.
 *
 * The result is a normalised likelihood over `OBSERVABLE_LEAF_CLASSES` that can
 * be multiplied into the posterior as one more censored, independently-admissible
 * view. Two safety behaviours preserve the calibrated-uncertainty ethos:
 *   - open-set abstention: if the classifier returned "unknown" (nearest
 *     prototype beyond threshold) the view is uniform — it injects no vote;
 *   - unmapped modulations (raw PSK/QAM with no named-protocol leaf) route their
 *     mass to `unknown-signal` rather than being forced onto a protocol.
 */

import { OBSERVABLE_LEAF_CLASSES, type ObservableLeafClass } from '../observable-classifier-model.js';
import type { Classification } from './prototype-classifier.js';

interface LeafCandidate {
  leaf: ObservableLeafClass;
  bwHz?: [number, number]; // nominal occupied-bandwidth range that selects this leaf
}

/**
 * Modulation family -> candidate protocol leaves. An empty list means "no named
 * protocol leaf" and routes to unknown-signal. Bandwidth ranges are nominal and
 * only used when a measured bandwidth (Hz) is supplied.
 */
// Classes are the SignalLab-trained taxonomy (cw · am · fm · gsm · ofdm · dsss ·
// bluetooth); each maps to the protocol leaf(s) it is compatible with, and
// bandwidth resolves the modulation-degenerate OFDM protocols.
export const MODULATION_TO_LEAVES: Record<string, LeafCandidate[]> = {
  cw: [{ leaf: 'cw-like' }],
  am: [{ leaf: 'am-dsb-full-carrier-like' }],
  fm: [{ leaf: 'fm-angle-modulated-like' }],
  // GERAN / GSM family (GMSK + higher-order EDGE/EGPRS bursts, ~200 kHz)
  gsm: [{ leaf: 'gsm-like' }],
  // Bluetooth BR/EDR + LE (GFSK, ~1-2 MHz)
  bluetooth: [{ leaf: 'bluetooth-like' }],
  // Wi-Fi HR/DSSS (~22 MHz spread)
  dsss: [{ leaf: 'wifi-hr-dsss-like' }],
  // OFDM is shared by LTE, NR and Wi-Fi — bandwidth + band decide which
  ofdm: [
    { leaf: 'lte-fdd-like', bwHz: [1.3e6, 20e6] },
    { leaf: 'lte-tdd-like', bwHz: [1.3e6, 20e6] },
    { leaf: 'nr-fdd-like', bwHz: [5e6, 100e6] },
    { leaf: 'nr-tdd-like', bwHz: [5e6, 100e6] },
    { leaf: 'wifi-ofdm-like', bwHz: [16e6, 84e6] },
  ],
};

const OUT_OF_RANGE_WEIGHT = 0.05; // measured BW outside a leaf's range: unlikely, not impossible

export interface FusionContext {
  /** Measured occupied bandwidth in Hz (from detection). Enables disambiguation. */
  bandwidthHz?: number;
}

function bandwidthWeight(range: [number, number] | undefined, bwHz: number | undefined): number {
  if (!range || bwHz === undefined) return 1;
  return bwHz >= range[0] && bwHz <= range[1] ? 1 : OUT_OF_RANGE_WEIGHT;
}

/**
 * Map a prototype Classification to a normalised likelihood over the leaf
 * taxonomy. `leafClasses` defaults to the real model taxonomy.
 */
export function embeddingEvidenceLikelihood(
  classification: Classification,
  ctx: FusionContext = {},
  leafClasses: readonly string[] = OBSERVABLE_LEAF_CLASSES,
): Record<string, number> {
  const out: Record<string, number> = {};
  for (const leaf of leafClasses) out[leaf] = 0;

  // open-set abstention: no vote, uniform likelihood
  if (classification.isUnknown) {
    const u = 1 / leafClasses.length;
    for (const leaf of leafClasses) out[leaf] = u;
    return out;
  }

  for (const [mod, pm] of Object.entries(classification.posterior)) {
    if (pm <= 0) continue;
    const candidates = MODULATION_TO_LEAVES[mod];
    if (!candidates || candidates.length === 0) {
      out['unknown-signal'] = (out['unknown-signal'] ?? 0) + pm;
      continue;
    }
    const weights = candidates.map((c) => bandwidthWeight(c.bwHz, ctx.bandwidthHz));
    const wsum = weights.reduce((a, b) => a + b, 0) || 1;
    candidates.forEach((c, i) => {
      out[c.leaf] = (out[c.leaf] ?? 0) + (pm * weights[i]!) / wsum;
    });
  }

  // normalise (guards against leaves absent from `leafClasses`)
  let z = 0;
  for (const leaf of leafClasses) z += out[leaf] ?? 0;
  if (z > 0) for (const leaf of leafClasses) out[leaf] = (out[leaf] ?? 0) / z;
  return out;
}
