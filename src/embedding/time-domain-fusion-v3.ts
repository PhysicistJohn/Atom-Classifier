/**
 * Pure-TypeScript v3 centered invariant fusion and prototype classification.
 *
 * Ports the inference rule of
 * `training/zplane_ab/v2_full_variation/invariant_fusion.py`
 * (CenteredInvariantFusion) and the assembly maths of
 * `training/zplane_ab/v2_full_variation/v3_scale/assemble_v3_fusion.py`:
 *
 *   fused = unit(concat(sqrt(w)     * unit(real    - alpha_real    * real_center),
 *                       sqrt(1 - w) * unit(complex - alpha_complex * complex_center)))
 *
 * followed by squared-Euclidean nearest-prototype classification. Raw scalar
 * features are standardized with the training-only moments using NumPy's
 * float32 arithmetic (replicated via `Math.fround`), exactly as
 * `run_time_domain_dev.py` and `export_v3_fusion_runtime.py` do.
 *
 * Consumes the `atomos.v3.time-domain-invariant-fusion.browser-weights` asset
 * written by `v3_scale/export_v3_browser_weights.py`. The source bundle's
 * rejection slot is unset ("closed-set only; a consumer of this bundle must
 * not abstain"), so this runtime exposes no abstention path: the nearest
 * prototype is always the label.
 */

import {
  forwardTimeDomainComplexBranchV3,
  forwardTimeDomainRealBranchV3,
  validateTimeDomainComplexBranchV3,
  validateTimeDomainRealBranchV3,
  type TimeDomainComplexBranchV3,
  type TimeDomainRealBranchV3,
} from './time-domain-encoder-v3.js';
import {
  admitTimeDomainAssetStatusV3,
  type TimeDomainAssetLoadOptionsV3,
  type TimeDomainAssetStatusV3,
} from './time-domain-asset-status-v3.js';

export const TIME_DOMAIN_FUSION_V3_SCHEMA =
  'atomos.v3.time-domain-invariant-fusion.browser-weights' as const;
export const TIME_DOMAIN_FUSION_V3_SCHEMA_VERSION = 1 as const;

/**
 * Explicit responsibilities used by the decoupled v3.3 runtime. The field is
 * optional at this low-level loader so the frozen legacy single-fusion asset
 * remains loadable; the dual-fusion composition requires one exact role on
 * each of its two assets.
 */
export const TIME_DOMAIN_FUSION_REJECTOR_ROLE_V3 =
  'known_unknown_rejector' as const;
export const TIME_DOMAIN_FUSION_CLASSIFIER_ROLE_V3 =
  'accepted_known_classifier' as const;
export type TimeDomainFusionRuntimeRoleV3 =
  | typeof TIME_DOMAIN_FUSION_REJECTOR_ROLE_V3
  | typeof TIME_DOMAIN_FUSION_CLASSIFIER_ROLE_V3;

/**
 * Schema identifiers the v3 runtime must refuse: the v2/hybrid frontend
 * contracts and the v2 paired-real staging export are silently wrong for the
 * time-domain frontend.
 */
export const TIME_DOMAIN_FUSION_V3_FORBIDDEN_SCHEMAS = Object.freeze([
  'atomos.invariant-fusion.paired-real',
  'hybrid-v3',
  'invariant-centered-fusion-with-lof-rank-ensemble',
  'invariant-patch-v1',
]);

export interface CenteredFusionV3 {
  real_center: number[];
  complex_center: number[];
  alpha_real: number;
  alpha_complex: number;
  weight_real: number;
  eps: number;
}

export interface FeatureStandardizationV3 {
  mean: number[];
  std: number[];
}

export interface TimeDomainFusionFrontendV3 {
  version: 'invariant-patch-time-domain-v1';
  patch_length: number;
  patch_count: number;
  target_frac: number;
  feature_count: number;
  uses_frequency_transform: false;
}

export interface TimeDomainClassificationV3 {
  classes: string[];
  /** One row per class, each `2 * embed_dim` wide. */
  prototypes: number[][];
}

export interface TimeDomainFusionAssetV3 {
  schema: typeof TIME_DOMAIN_FUSION_V3_SCHEMA;
  schema_version: typeof TIME_DOMAIN_FUSION_V3_SCHEMA_VERSION;
  status: TimeDomainAssetStatusV3;
  /**
   * Present on role-bound dual-fusion exports. Optional only for backwards
   * compatibility with the frozen legacy single-fusion browser package.
   */
  runtime_role?: TimeDomainFusionRuntimeRoleV3;
  packed_length: number;
  /**
   * The historical low-level loader ignored this already-exported metadata.
   * It is retained now so the dual composition can bind `target_frac` and the
   * explicit no-frequency-transform claim across both fusion roles.
   */
  frontend?: TimeDomainFusionFrontendV3;
  real: TimeDomainRealBranchV3;
  complex: TimeDomainComplexBranchV3;
  fusion: CenteredFusionV3;
  feature_standardization: FeatureStandardizationV3;
  classification: TimeDomainClassificationV3;
  /** Export provenance is retained for package-level identity inspection. */
  provenance?: unknown;
}

export interface TimeDomainForwardV3 {
  standardizedFeatures: Float64Array;
  realEmbedding: Float64Array;
  complexEmbedding: Float64Array;
  fusedEmbedding: Float64Array;
}

export interface TimeDomainClassificationResultV3 extends TimeDomainForwardV3 {
  squaredPrototypeDistances: Float64Array;
  closedWinnerIndex: number;
  /** Nearest-prototype label. There is no abstention path in this bundle. */
  closedLabel: string;
}

type UnknownRecord = Record<string, unknown>;

function record(value: unknown, path: string): UnknownRecord {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new TypeError(`${path} must be an object`);
  }
  return value as UnknownRecord;
}

function finite(value: unknown, path: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`${path} must be a finite number`);
  }
  return value;
}

function finiteVector(value: unknown, path: string, length: number): number[] {
  if (!Array.isArray(value)) throw new TypeError(`${path} must be an array`);
  if (value.length !== length) {
    throw new RangeError(`${path} must contain ${length} values`);
  }
  for (let index = 0; index < value.length; index++) {
    finite(value[index], `${path}[${index}]`);
  }
  return value as number[];
}

/**
 * Validate an untrusted JSON value from the exporter and return it typed.
 * Cross-checks branch geometry, fusion dimensions, and prototype shapes.
 */
export function loadTimeDomainFusionAssetV3(
  value: unknown,
  options: TimeDomainAssetLoadOptionsV3 = {},
): TimeDomainFusionAssetV3 {
  const asset = record(value, 'asset');
  if (TIME_DOMAIN_FUSION_V3_FORBIDDEN_SCHEMAS.includes(String(asset.schema))) {
    throw new RangeError(
      `schema ${String(asset.schema)} is a v2/hybrid contract and is invalid `
      + 'for the v3 time-domain runtime',
    );
  }
  if (asset.schema !== TIME_DOMAIN_FUSION_V3_SCHEMA) {
    throw new RangeError(`unsupported schema ${String(asset.schema)}`);
  }
  if (asset.schema_version !== TIME_DOMAIN_FUSION_V3_SCHEMA_VERSION) {
    throw new RangeError(
      `unsupported schema version ${String(asset.schema_version)}`,
    );
  }
  const status = admitTimeDomainAssetStatusV3(asset.status, 'asset.status', options);
  let runtimeRole: TimeDomainFusionRuntimeRoleV3 | undefined;
  if (asset.runtime_role !== undefined) {
    if (
      asset.runtime_role !== TIME_DOMAIN_FUSION_REJECTOR_ROLE_V3
      && asset.runtime_role !== TIME_DOMAIN_FUSION_CLASSIFIER_ROLE_V3
    ) {
      throw new RangeError(
        'asset.runtime_role must be known_unknown_rejector or '
        + 'accepted_known_classifier',
      );
    }
    runtimeRole = asset.runtime_role;
  }
  const real = validateTimeDomainRealBranchV3(asset.real);
  const complex = validateTimeDomainComplexBranchV3(asset.complex);
  for (const field of [
    'patch_length',
    'patch_count',
    'embed_dim',
    'n_features',
  ] as const) {
    if (real.config[field] !== complex.config[field]) {
      throw new RangeError(`real/complex config mismatch for ${field}`);
    }
  }
  const packedLength = finite(asset.packed_length, 'asset.packed_length');
  if (packedLength !== real.config.patch_length * real.config.patch_count) {
    throw new RangeError('asset.packed_length does not match patch geometry');
  }
  let frontend: TimeDomainFusionFrontendV3 | undefined;
  if (asset.frontend !== undefined) {
    const frontendRaw = record(asset.frontend, 'asset.frontend');
    if (frontendRaw.version !== 'invariant-patch-time-domain-v1') {
      throw new RangeError(
        'asset.frontend.version must be invariant-patch-time-domain-v1',
      );
    }
    if (frontendRaw.uses_frequency_transform !== false) {
      throw new RangeError(
        'asset.frontend must declare uses_frequency_transform: false',
      );
    }
    const patchLength = finite(
      frontendRaw.patch_length,
      'asset.frontend.patch_length',
    );
    const patchCount = finite(
      frontendRaw.patch_count,
      'asset.frontend.patch_count',
    );
    const targetFrac = finite(
      frontendRaw.target_frac,
      'asset.frontend.target_frac',
    );
    const featureCount = finite(
      frontendRaw.feature_count,
      'asset.frontend.feature_count',
    );
    if (
      !Number.isInteger(patchLength)
      || !Number.isInteger(patchCount)
      || !Number.isInteger(featureCount)
      || patchLength !== real.config.patch_length
      || patchCount !== real.config.patch_count
      || featureCount !== real.config.n_features
      || !(targetFrac > 0 && targetFrac <= 1)
    ) {
      throw new RangeError(
        'asset.frontend metadata does not match the branch geometry',
      );
    }
    frontend = {
      version: 'invariant-patch-time-domain-v1',
      patch_length: patchLength,
      patch_count: patchCount,
      target_frac: targetFrac,
      feature_count: featureCount,
      uses_frequency_transform: false,
    };
  }

  const embedDim = real.config.embed_dim;
  const fusionRaw = record(asset.fusion, 'asset.fusion');
  const fusion: CenteredFusionV3 = {
    real_center: finiteVector(
      fusionRaw.real_center,
      'asset.fusion.real_center',
      embedDim,
    ),
    complex_center: finiteVector(
      fusionRaw.complex_center,
      'asset.fusion.complex_center',
      embedDim,
    ),
    alpha_real: finite(fusionRaw.alpha_real, 'asset.fusion.alpha_real'),
    alpha_complex: finite(fusionRaw.alpha_complex, 'asset.fusion.alpha_complex'),
    weight_real: finite(fusionRaw.weight_real, 'asset.fusion.weight_real'),
    eps: finite(fusionRaw.eps, 'asset.fusion.eps'),
  };
  if (fusion.weight_real < 0 || fusion.weight_real > 1) {
    throw new RangeError('asset.fusion.weight_real must lie in [0,1]');
  }
  if (fusion.eps <= 0) throw new RangeError('asset.fusion.eps must be positive');

  const featureRaw = record(
    asset.feature_standardization,
    'asset.feature_standardization',
  );
  const mean = finiteVector(
    featureRaw.mean,
    'asset.feature_standardization.mean',
    real.config.n_features,
  );
  const std = finiteVector(
    featureRaw.std,
    'asset.feature_standardization.std',
    real.config.n_features,
  );
  for (let index = 0; index < std.length; index++) {
    if (std[index]! <= 0) {
      throw new RangeError(
        `asset.feature_standardization.std[${index}] must be positive`,
      );
    }
  }

  const classificationRaw = record(asset.classification, 'asset.classification');
  if (
    !Array.isArray(classificationRaw.classes)
    || classificationRaw.classes.length === 0
    || classificationRaw.classes.some(
      (name) => typeof name !== 'string' || name.length === 0,
    )
    || new Set(classificationRaw.classes).size
      !== classificationRaw.classes.length
  ) {
    throw new RangeError(
      'asset.classification.classes must be unique non-empty strings',
    );
  }
  const classes = classificationRaw.classes as string[];
  if (
    !Array.isArray(classificationRaw.prototypes)
    || classificationRaw.prototypes.length !== classes.length
  ) {
    throw new RangeError(
      'asset.classification.prototypes must have one row per class',
    );
  }
  const prototypes = classificationRaw.prototypes.map((prototype, index) =>
    finiteVector(
      prototype,
      `asset.classification.prototypes[${index}]`,
      2 * embedDim,
    ));

  return {
    schema: TIME_DOMAIN_FUSION_V3_SCHEMA,
    schema_version: TIME_DOMAIN_FUSION_V3_SCHEMA_VERSION,
    status,
    ...(runtimeRole === undefined ? {} : { runtime_role: runtimeRole }),
    packed_length: packedLength,
    ...(frontend === undefined ? {} : { frontend }),
    real,
    complex,
    fusion,
    feature_standardization: { mean, std },
    classification: { classes, prototypes },
    ...(asset.provenance === undefined ? {} : { provenance: asset.provenance }),
  };
}

/**
 * Standardize raw float32 features with the training-only moments.
 *
 * Replicates NumPy float32 elementwise arithmetic exactly: the subtraction is
 * rounded to float32, then the division is rounded to float32, matching
 * `((raw_features - feature_mean) / feature_std).astype(np.float32)` for the
 * float32 operands the bundle stores.
 */
export function standardizeTimeDomainFeaturesV3(
  standardization: FeatureStandardizationV3,
  rawFeatures: ArrayLike<number>,
): Float64Array {
  const width = standardization.mean.length;
  if (rawFeatures.length !== width) {
    throw new RangeError(`rawFeatures must have length ${width}`);
  }
  const output = new Float64Array(width);
  for (let index = 0; index < width; index++) {
    const raw = rawFeatures[index]!;
    if (!Number.isFinite(raw)) {
      throw new RangeError(`rawFeatures[${index}] must be finite`);
    }
    const centered = Math.fround(
      Math.fround(raw) - standardization.mean[index]!,
    );
    output[index] = Math.fround(centered / standardization.std[index]!);
  }
  return output;
}

/**
 * `_safe_unit`: rows with squared norm above `eps^2` are scaled to unit
 * length; an exact cancellation falls back to the first basis vector.
 */
function safeUnit(value: Float64Array, eps: number): Float64Array {
  let normSquared = 0;
  for (const item of value) normSquared += item * item;
  if (normSquared <= eps * eps) {
    const fallback = new Float64Array(value.length);
    fallback[0] = 1;
    return fallback;
  }
  const inverseNorm = 1 / Math.sqrt(normSquared);
  const output = new Float64Array(value.length);
  for (let index = 0; index < value.length; index++) {
    output[index] = value[index]! * inverseNorm;
  }
  return output;
}

/** The centered weighted-concatenation fuse rule. */
export function fuseTimeDomainEmbeddingsV3(
  fusion: CenteredFusionV3,
  realEmbedding: ArrayLike<number>,
  complexEmbedding: ArrayLike<number>,
): Float64Array {
  const embedDim = fusion.real_center.length;
  if (
    realEmbedding.length !== embedDim
    || complexEmbedding.length !== embedDim
  ) {
    throw new RangeError(`branch embeddings must have length ${embedDim}`);
  }
  const centeredReal = new Float64Array(embedDim);
  const centeredComplex = new Float64Array(embedDim);
  for (let index = 0; index < embedDim; index++) {
    centeredReal[index] = (
      realEmbedding[index]! - fusion.alpha_real * fusion.real_center[index]!
    );
    centeredComplex[index] = (
      complexEmbedding[index]!
      - fusion.alpha_complex * fusion.complex_center[index]!
    );
  }
  const normalizedReal = safeUnit(centeredReal, fusion.eps);
  const normalizedComplex = safeUnit(centeredComplex, fusion.eps);
  const realScale = Math.sqrt(fusion.weight_real);
  const complexScale = Math.sqrt(1 - fusion.weight_real);
  const fused = new Float64Array(2 * embedDim);
  for (let index = 0; index < embedDim; index++) {
    fused[index] = realScale * normalizedReal[index]!;
    fused[embedDim + index] = complexScale * normalizedComplex[index]!;
  }
  return safeUnit(fused, fusion.eps);
}

/** Squared Euclidean distance from one embedding to every prototype row. */
export function squaredPrototypeDistancesV3(
  embedding: ArrayLike<number>,
  prototypes: number[][],
): Float64Array {
  const output = new Float64Array(prototypes.length);
  for (let classIndex = 0; classIndex < prototypes.length; classIndex++) {
    const prototype = prototypes[classIndex]!;
    if (prototype.length !== embedding.length) {
      throw new RangeError(
        `prototype ${classIndex} width ${prototype.length} does not match `
        + `embedding width ${embedding.length}`,
      );
    }
    let distance = 0;
    for (let dimension = 0; dimension < embedding.length; dimension++) {
      const delta = embedding[dimension]! - prototype[dimension]!;
      distance += delta * delta;
    }
    output[classIndex] = distance;
  }
  return output;
}

/** Run standardization, both branches, and the fuse rule for one capture. */
export function forwardTimeDomainFusionV3(
  asset: TimeDomainFusionAssetV3,
  inPhase: ArrayLike<number>,
  quadrature: ArrayLike<number>,
  rawFeatures: ArrayLike<number>,
): TimeDomainForwardV3 {
  const standardizedFeatures = standardizeTimeDomainFeaturesV3(
    asset.feature_standardization,
    rawFeatures,
  );
  const realEmbedding = forwardTimeDomainRealBranchV3(
    asset.real,
    inPhase,
    quadrature,
    standardizedFeatures,
  );
  const complexEmbedding = forwardTimeDomainComplexBranchV3(
    asset.complex,
    inPhase,
    quadrature,
    standardizedFeatures,
  );
  return {
    standardizedFeatures,
    realEmbedding,
    complexEmbedding,
    fusedEmbedding: fuseTimeDomainEmbeddingsV3(
      asset.fusion,
      realEmbedding,
      complexEmbedding,
    ),
  };
}

/**
 * Classify one packed capture: nearest fused prototype under squared
 * Euclidean distance. Closed-set only; the bundle's rejection slot is unset
 * and this runtime must not abstain or substitute a v2 rejector.
 */
export function classifyTimeDomainFusionV3(
  asset: TimeDomainFusionAssetV3,
  inPhase: ArrayLike<number>,
  quadrature: ArrayLike<number>,
  rawFeatures: ArrayLike<number>,
): TimeDomainClassificationResultV3 {
  const forward = forwardTimeDomainFusionV3(
    asset,
    inPhase,
    quadrature,
    rawFeatures,
  );
  const distances = squaredPrototypeDistancesV3(
    forward.fusedEmbedding,
    asset.classification.prototypes,
  );
  let winner = 0;
  for (let index = 1; index < distances.length; index++) {
    if (distances[index]! < distances[winner]!) winner = index;
  }
  return {
    ...forward,
    squaredPrototypeDistances: distances,
    closedWinnerIndex: winner,
    closedLabel: asset.classification.classes[winner]!,
  };
}
