/**
 * Magnitude-only (tinySA / scalar) flavor of the waveform classifier.
 *
 * Same architecture, prototypes, fusion, and 7 classes as the I/Q flavor, but the
 * input is a power spectrum (no phase), so it runs on a scalar spectrum analyzer.
 * Two entry points: `classifyIq` (compute the PSD from complex I/Q — used to
 * validate against the I/Q flavor) and `classifyPsd` (a swept power spectrum
 * straight from a tinySA). Browser-native, zero runtime dependencies.
 */

import { forwardChannels, standardizeFeatures, type EmbeddingModel } from './embedding-runtime.js';
import { magnitudeFromIq, representationFromPsd, type MagnitudeRepresentation } from './magnitude-preprocess.js';
import { classify, loadPrototypeSet, type Classification, type PrototypeSet } from './prototype-classifier.js';
import { embeddingEvidenceLikelihood, type FusionContext } from './embedding-evidence-fusion.js';
import { refineModulation, type RefinedModulation } from './order-refinement.js';

export interface MagnitudeResult {
  classification: Classification;
  modulation: RefinedModulation;
  /** Measured occupied fractional bandwidth (from the feature vector). */
  bw: number;
  leafLikelihood: Record<string, number>;
}

export class MagnitudeWaveformClassifier {
  private readonly model: EmbeddingModel;
  private prototypes: PrototypeSet;

  constructor(modelJson: EmbeddingModel, prototypesJson: Parameters<typeof loadPrototypeSet>[0]) {
    this.model = modelJson;
    this.prototypes = loadPrototypeSet(prototypesJson);
  }

  private embed(rep: MagnitudeRepresentation): Float64Array {
    const feat = standardizeFeatures(this.model, rep.features);
    return forwardChannels(this.model, [rep.shape], feat);
  }

  private finish(rep: MagnitudeRepresentation, ctx: FusionContext): MagnitudeResult {
    const classification = classify(this.prototypes, this.embed(rep));
    return {
      classification,
      modulation: refineModulation(classification.label),
      bw: rep.features[0] ?? 0,
      leafLikelihood: embeddingEvidenceLikelihood(classification, ctx),
    };
  }

  /** Classify from complex I/Q (computes the power spectrum internally). */
  classifyIq(re: Float64Array, im: Float64Array, ctx: FusionContext = {}): MagnitudeResult {
    return this.finish(magnitudeFromIq(re, im), ctx);
  }

  /**
   * Classify from a swept power spectrum (tinySA path). `psd` is linear power,
   * fftshifted; `center` and `bw` are the occupied band in cycles/sample.
   */
  classifyPsd(psd: Float64Array, center: number, bw: number, ctx: FusionContext = {}): MagnitudeResult {
    return this.finish(representationFromPsd(psd, center, bw), ctx);
  }

  get classes(): readonly string[] {
    return this.prototypes.classes;
  }
}
