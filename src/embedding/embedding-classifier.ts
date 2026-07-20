/**
 * Public facade for the metric-embedding few-shot classifier.
 *
 * Composes the deployable inference path end-to-end: normalise raw complex I/Q →
 * embed → nearest-prototype classify (with open-set unknown) → fuse into the
 * Bayesian leaf taxonomy. Also exposes few-shot enrollment. Zero runtime
 * dependencies; runs wherever the Bayesian classifier runs. Construct it from
 * the parsed exported assets (`assets/embedding-weights.json`,
 * `assets/prototypes.json`).
 */

import { preprocess, type Normalised } from './iq-preprocess.js';
import { embed, preprocessParams, type EmbeddingModel } from './embedding-runtime.js';
import {
  classify,
  enroll,
  loadPrototypeSet,
  type Classification,
  type PrototypeSet,
} from './prototype-classifier.js';
import { embeddingEvidenceLikelihood, type FusionContext } from './embedding-evidence-fusion.js';
import { refineModulation, type OrderRefinement, type RefinedModulation } from './order-refinement.js';

export interface ClassifyOptions extends FusionContext {
  /** Ingestion-side quality-gated order refinement (from recover.py + order_refine.py). */
  orderRefinement?: OrderRefinement;
}

export interface EmbeddingResult {
  classification: Classification;
  /** Hierarchical modulation decision: family, or resolved order when supported. */
  modulation: RefinedModulation;
  /** Measured centre frequency (cycles/sample) and occupied fractional bandwidth. */
  center: number;
  bw: number;
  /** Normalised likelihood over the Bayesian leaf taxonomy (the fused evidence view). */
  leafLikelihood: Record<string, number>;
}

export class EmbeddingWaveformClassifier {
  private readonly model: EmbeddingModel;
  private prototypes: PrototypeSet;

  constructor(modelJson: EmbeddingModel, prototypesJson: Parameters<typeof loadPrototypeSet>[0]) {
    this.model = modelJson;
    this.prototypes = loadPrototypeSet(prototypesJson);
  }

  /** Normalise raw complex I/Q through the DSP front-end. */
  normalise(re: Float64Array, im: Float64Array): Normalised {
    return preprocess(re, im, preprocessParams(this.model));
  }

  /** Embed already-normalised channels. */
  embedNormalised(n: Normalised): Float64Array {
    return embed(this.model, n.i, n.q);
  }

  /** Embed raw complex I/Q (normalise + embed). */
  embedIq(re: Float64Array, im: Float64Array): Float64Array {
    return this.embedNormalised(this.normalise(re, im));
  }

  /**
   * Full classification of a raw complex I/Q capture. Pass the measured
   * bandwidth in Hz (from the SDR detection) in `ctx` to enable
   * modulation→protocol disambiguation in the fused leaf likelihood.
   */
  classifyIq(re: Float64Array, im: Float64Array, opts: ClassifyOptions = {}): EmbeddingResult {
    const n = this.normalise(re, im);
    const emb = this.embedNormalised(n);
    const classification = classify(this.prototypes, emb);
    const leafLikelihood = embeddingEvidenceLikelihood(classification, opts);
    const modulation = refineModulation(classification.label, opts.orderRefinement);
    return { classification, modulation, center: n.center, bw: n.bw, leafLikelihood };
  }

  /** Current enrollable class names (in prototype order). */
  get classes(): readonly string[] {
    return this.prototypes.classes;
  }

  /**
   * Few-shot enrollment: teach a new class (or refine an existing one) from a
   * handful of raw I/Q examples. No retraining; effective immediately.
   */
  enrollClass(name: string, examples: Array<{ re: Float64Array; im: Float64Array }>): void {
    const embeddings = examples.map((e) => this.embedIq(e.re, e.im));
    this.prototypes = enroll(this.prototypes, name, embeddings);
  }
}
