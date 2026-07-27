/**
 * Magnitude-only (tinySA / scalar) flavor of the waveform classifier.
 *
 * Same architecture, prototypes, fusion, and 7 classes as the I/Q flavor, but the
 * input is a power spectrum (no phase), so it runs on a scalar spectrum analyzer.
 * Two entry points: `classifyIq` (compute the PSD from complex I/Q — used to
 * validate against the I/Q flavor) and `classifyPsd` (a swept power spectrum
 * straight from a tinySA). Browser-native, zero runtime dependencies.
 */

import {
  forwardChannels,
  standardizeFeatures,
  type EmbeddingModel,
} from './embedding-runtime.js';
import {
  MAG_PARAMS,
  magnitudeFromIq,
  representationFromPsd,
  type MagnitudeRepresentation,
} from './magnitude-preprocess.js';
import type { PreprocessParams } from './iq-preprocess.js';
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

  private iqPreprocessParams(): PreprocessParams {
    const metadata = (this.model as EmbeddingModel & {
      magnitude?: {
        estimator_version?: 'linear-v1' | 'circular-v2' | 'hybrid-v3';
        mag_nfft?: number;
        smooth?: number;
        noise_floor_scale?: number;
        energy_edge?: number;
        full_band_bw?: number;
        white_flatness_deficit?: number;
        guard_floor_scale?: number;
        min_excess_fraction?: number;
        hybrid_legacy_wide_bw?: number;
        hybrid_circular_broad_bw?: number;
        hybrid_seam_tolerance_bins?: number;
      };
    }).magnitude;
    return {
      ...MAG_PARAMS,
      // Shipped magnitude models predate estimator_version and must remain v1.
      version: metadata?.estimator_version ?? 'linear-v1',
      nfft: metadata?.mag_nfft ?? MAG_PARAMS.nfft,
      smooth: metadata?.smooth ?? MAG_PARAMS.smooth,
      noiseFloorScale: metadata?.noise_floor_scale ?? MAG_PARAMS.noiseFloorScale,
      energyEdge: metadata?.energy_edge ?? MAG_PARAMS.energyEdge,
      fullBandBw: metadata?.full_band_bw,
      whiteFlatnessDeficit: metadata?.white_flatness_deficit,
      guardFloorScale: metadata?.guard_floor_scale,
      minExcessFraction: metadata?.min_excess_fraction,
      hybridLegacyWideBw: metadata?.hybrid_legacy_wide_bw,
      hybridCircularBroadBw: metadata?.hybrid_circular_broad_bw,
      hybridSeamToleranceBins: metadata?.hybrid_seam_tolerance_bins,
    };
  }

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
    return this.finish(magnitudeFromIq(re, im, this.iqPreprocessParams()), ctx);
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
