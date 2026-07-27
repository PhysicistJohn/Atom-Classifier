/** End-to-end raw-capture composition for the invariant fusion classifier. */

import {
  classifyInvariantFusion,
  type InvariantClassification,
  type InvariantFusionAsset,
} from './invariant-fusion-runtime.js';
import {
  preprocessInvariantPatches,
  type InvariantPatchContext,
  type InvariantPatchPreprocessOptions,
} from './invariant-patch-preprocess.js';

const INVARIANT_FEATURE_NAMES = [
  'abs_c20',
  'abs_c40',
  'c42_real',
  'abs_c41',
  'abs_c60',
  'c63',
  'm42',
  'amplitude_std',
  'max_amplitude_squared',
  'mean_amplitude',
  'instantaneous_frequency_std',
  'amplitude_coefficient_of_variation',
] as const;

export interface InvariantRawClassification {
  preprocessing: InvariantPatchContext;
  packedInPhase: Float64Array;
  packedQuadrature: Float64Array;
  rawFeatures: Float64Array;
  classification: InvariantClassification;
}

interface InvariantPreprocessContract {
  schema: number;
  frontend: string;
  estimator_version: string;
  nfft: number;
  feature_count: number;
  feature_names: string[];
  packed_dtype: string;
  raw_feature_dtype: string;
  config: {
    patch_length: number;
    patch_count: number;
    target_frac: number;
  };
}

function preprocessContract(asset: InvariantFusionAsset): InvariantPreprocessContract {
  const value = asset.preprocess;
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new TypeError('asset.preprocess must contain the invariant frontend contract');
  }
  const contract = value as Partial<InvariantPreprocessContract>;
  const config = contract.config;
  if (
    contract.schema !== 1
    || contract.frontend !== 'invariant-patch-v1'
    || contract.estimator_version !== 'hybrid-v3'
    || contract.nfft !== 512
    || contract.feature_count !== asset.real.config.n_features
    || !Array.isArray(contract.feature_names)
    || contract.feature_names.length !== INVARIANT_FEATURE_NAMES.length
    || contract.feature_names.some(
      (name, index) => name !== INVARIANT_FEATURE_NAMES[index],
    )
    || contract.packed_dtype !== 'float32'
    || contract.raw_feature_dtype !== 'float32'
    || typeof config !== 'object'
    || config === null
    || config.patch_length !== asset.real.config.patch_length
    || config.patch_count !== asset.real.config.patch_count
    || typeof config.target_frac !== 'number'
    || !Number.isFinite(config.target_frac)
    || config.target_frac !== 0.5
  ) {
    throw new RangeError('asset.preprocess is incompatible with the model geometry');
  }
  return contract as InvariantPreprocessContract;
}

/**
 * Run the complete deployed path from an arbitrary-length raw capture through
 * hybrid-v3 estimation, invariant patch extraction, fusion, and weighted LOF.
 */
export function classifyInvariantRawCapture(
  asset: InvariantFusionAsset,
  inPhase: Float64Array,
  quadrature: Float64Array,
  overrides: Pick<
    InvariantPatchPreprocessOptions,
    'forceCenter' | 'forceBw' | 'forceResampleFrac'
  > = {},
): InvariantRawClassification {
  const contract = preprocessContract(asset);
  const preprocessing = preprocessInvariantPatches(inPhase, quadrature, {
    patchLength: contract.config.patch_length,
    patchCount: contract.config.patch_count,
    targetFrac: contract.config.target_frac,
    nfft: contract.nfft,
    ...overrides,
  });
  const classification = classifyInvariantFusion(
    asset,
    preprocessing.packedInPhase,
    preprocessing.packedQuadrature,
    preprocessing.rawFeatures,
  );
  return {
    preprocessing: preprocessing.context,
    packedInPhase: preprocessing.packedInPhase,
    packedQuadrature: preprocessing.packedQuadrature,
    rawFeatures: preprocessing.rawFeatures,
    classification,
  };
}
