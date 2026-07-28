/**
 * Full v3 time-domain classification pipeline for the browser runtime:
 *
 *   raw I/Q
 *     -> STAGE-1 noise gate (pose-degeneracy prefilter; fires BEFORE anything
 *        downstream is computed -- gated rows never reach the frontend or the
 *        encoders)
 *     -> strict time-domain invariant-patch preprocess (FFT-free)
 *     -> feature standardization (training-only moments, float32 semantics)
 *     -> real + complex branch encoders (sibling encoder modules)
 *     -> centered invariant fusion (alpha 0.2, weight_real 0.5, eps 1e-12)
 *     -> nearest fused prototype (squared euclidean argmin)
 *     -> STAGE-2 frozen rejector (branch LOF ranks 0.80 / geometry 0.20, q95)
 *     -> decision
 *
 * This ports the decision layer of the v3 runtime bundle
 * (`atomos.v3.time-domain-invariant-fusion.runtime-bundle`, exported by
 * `v3_scale/export_v3_fusion_runtime.py`) plus the staged rejector of
 * `v3_scale/fit_v3_openset_staged.py`. It is NOT compatible with the frozen
 * hybrid-v3/Welch frontend or the v2 `invariant-patch-v1` assets, and it
 * introduces no frequency transform anywhere.
 *
 * The branch encoder forward passes are the sibling port
 * (`./time-domain-encoder-v3.js`). This module defines the interface it codes
 * against and fails loudly when that module is absent; the decision layer is
 * fully testable with injected encoders in the meantime.
 */

import {
  preprocessTimeDomainInvariantPatches,
  type TimeDomainInvariantPatchResult,
} from './time-domain-invariant-patch-preprocess-v3.js';
import {
  TimeDomainOpenSetV3,
  loadTimeDomainOpenSetAssetV3,
  stagedArchitectureContract,
  type StagedArchitectureContract,
  type StagedOpenSetDecision,
  type StagedRejectionStage,
  type StageOneEvaluation,
  type TimeDomainOpenSetAssetV3,
} from './time-domain-openset-v3.js';
import {
  admitTimeDomainAssetStatusV3,
  type TimeDomainAssetAdmissionV3,
  type TimeDomainAssetLoadOptionsV3,
  type TimeDomainAssetStatusV3,
} from './time-domain-asset-status-v3.js';
import {
  loadTimeDomainFusionAssetV3,
  type TimeDomainFusionAssetV3,
} from './time-domain-fusion-v3.js';

export const TIME_DOMAIN_CLASSIFIER_SCHEMA =
  'atomos.v3.time-domain-invariant-fusion.browser-decision' as const;
export const TIME_DOMAIN_CLASSIFIER_SCHEMA_VERSION = 1 as const;

/**
 * Schema ids this runtime must refuse
 * (bundle manifest `not_compatible_with_schema_ids`).
 */
export const TIME_DOMAIN_CLASSIFIER_INCOMPATIBLE_SCHEMAS = [
  'atomos.invariant-fusion.paired-real',
  'hybrid-v3',
  'invariant-centered-fusion-with-lof-rank-ensemble',
  'invariant-patch-v1',
] as const;

// ---------------------------------------------------------------------------
// the sibling encoder contract
// ---------------------------------------------------------------------------

/**
 * One branch encoder forward pass: packed float32-quantized I/Q channels of
 * `packed_length` samples plus the standardized 12-feature vector, returning
 * the unit-normalized `embed_dim` embedding. Implemented by the sibling
 * encoder port; injected here so the decision layer has no hard dependency.
 */
export type TimeDomainBranchEncoderV3 = (
  packedInPhase: ArrayLike<number>,
  packedQuadrature: ArrayLike<number>,
  standardizedFeatures: Float64Array,
) => Float64Array;

export interface TimeDomainEncoderPairV3 {
  real: TimeDomainBranchEncoderV3;
  complex: TimeDomainBranchEncoderV3;
}

/** Module specifier of the sibling encoder port. */
export const TIME_DOMAIN_ENCODER_MODULE_V3 = './time-domain-encoder-v3.js';

/**
 * Load the sibling encoder module dynamically, failing loudly when it is not
 * present. The expected module shape is
 * `createTimeDomainEncodersV3(encoderAsset: unknown): TimeDomainEncoderPairV3`.
 */
export async function loadTimeDomainEncodersV3(
  encoderAsset: unknown,
): Promise<TimeDomainEncoderPairV3> {
  let module: Record<string, unknown>;
  try {
    module = (await import(
      /* @vite-ignore */ TIME_DOMAIN_ENCODER_MODULE_V3
    )) as Record<string, unknown>;
  } catch (cause) {
    throw new Error(
      `the sibling v3 encoder module ${TIME_DOMAIN_ENCODER_MODULE_V3} is not `
      + 'present. The time-domain v3 decision layer refuses to run without '
      + 'the real encoder forward passes; inject encoders explicitly via '
      + 'createTimeDomainClassifierV3 if you are testing the decision layer '
      + 'alone.',
      { cause },
    );
  }
  const factory = module['createTimeDomainEncodersV3'];
  if (typeof factory !== 'function') {
    throw new Error(
      `${TIME_DOMAIN_ENCODER_MODULE_V3} does not export `
      + 'createTimeDomainEncodersV3(encoderAsset); the sibling encoder '
      + 'contract changed and this module must be updated with it.',
    );
  }
  const encoders = factory(encoderAsset) as TimeDomainEncoderPairV3;
  if (
    typeof encoders !== 'object'
    || encoders === null
    || typeof encoders.real !== 'function'
    || typeof encoders.complex !== 'function'
  ) {
    throw new Error(
      'createTimeDomainEncodersV3 must return { real, complex } branch '
      + 'encoder functions',
    );
  }
  return encoders;
}

// ---------------------------------------------------------------------------
// decision-layer asset
// ---------------------------------------------------------------------------

export interface TimeDomainClassifierAssetV3 {
  schema: typeof TIME_DOMAIN_CLASSIFIER_SCHEMA;
  schema_version: typeof TIME_DOMAIN_CLASSIFIER_SCHEMA_VERSION;
  status: TimeDomainAssetStatusV3;
  frontend: {
    version: 'invariant-patch-time-domain-v1';
    patch_length: number;
    patch_count: number;
    target_frac: number;
    packed_length: number;
    uses_frequency_transform: false;
  };
  feature_standardization: {
    mean: number[];
    std: number[];
  };
  fusion: {
    real_center: number[];
    complex_center: number[];
    alpha_real: number;
    alpha_complex: number;
    weight_real: number;
    eps: number;
  };
  classification: {
    classes: string[];
    /** Fused-space prototypes, shape [classes, 2 * embed_dim]. */
    prototypes: number[][];
    distance: 'squared_euclidean';
  };
  provenance?: unknown;
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

function finiteVector(value: unknown, path: string, length?: number): number[] {
  if (!Array.isArray(value)) throw new TypeError(`${path} must be an array`);
  if (length !== undefined && value.length !== length) {
    throw new RangeError(`${path} must contain ${length} values`);
  }
  for (let index = 0; index < value.length; index++) {
    finite(value[index], `${path}[${index}]`);
  }
  return value as number[];
}

/** Validate an untrusted JSON value into a typed classifier decision asset. */
export function loadTimeDomainClassifierAssetV3(
  value: unknown,
  options: TimeDomainAssetLoadOptionsV3 = {},
): TimeDomainClassifierAssetV3 {
  const asset = record(value, 'asset');
  if (
    TIME_DOMAIN_CLASSIFIER_INCOMPATIBLE_SCHEMAS.includes(
      asset.schema as (typeof TIME_DOMAIN_CLASSIFIER_INCOMPATIBLE_SCHEMAS)[number],
    )
  ) {
    throw new RangeError(
      `asset schema ${String(asset.schema)} is explicitly incompatible with `
      + 'the v3 time-domain runtime',
    );
  }
  if (asset.schema !== TIME_DOMAIN_CLASSIFIER_SCHEMA) {
    throw new RangeError(`unsupported classifier schema ${String(asset.schema)}`);
  }
  if (asset.schema_version !== TIME_DOMAIN_CLASSIFIER_SCHEMA_VERSION) {
    throw new RangeError(
      `unsupported classifier schema version ${String(asset.schema_version)}`,
    );
  }
  admitTimeDomainAssetStatusV3(asset.status, 'asset.status', options);
  const frontend = record(asset.frontend, 'asset.frontend');
  if (frontend.version !== 'invariant-patch-time-domain-v1') {
    throw new RangeError(
      'asset.frontend.version must be invariant-patch-time-domain-v1',
    );
  }
  if (frontend.uses_frequency_transform !== false) {
    throw new RangeError('asset.frontend must declare uses_frequency_transform: false');
  }
  const patchLength = finite(frontend.patch_length, 'asset.frontend.patch_length');
  const patchCount = finite(frontend.patch_count, 'asset.frontend.patch_count');
  const targetFrac = finite(frontend.target_frac, 'asset.frontend.target_frac');
  const packedLength = finite(frontend.packed_length, 'asset.frontend.packed_length');
  if (
    !Number.isInteger(patchLength) || patchLength <= 0
    || !Number.isInteger(patchCount) || patchCount <= 0
    || !(targetFrac > 0 && targetFrac <= 1)
    || packedLength !== patchLength * patchCount
  ) {
    throw new RangeError('asset.frontend geometry is invalid');
  }

  const standardization = record(
    asset.feature_standardization,
    'asset.feature_standardization',
  );
  const mean = finiteVector(standardization.mean, 'asset.feature_standardization.mean');
  const std = finiteVector(
    standardization.std,
    'asset.feature_standardization.std',
    mean.length,
  );
  if (std.some((entry) => entry <= 0)) {
    throw new RangeError('asset.feature_standardization.std must be positive');
  }

  const fusion = record(asset.fusion, 'asset.fusion');
  const realCenter = finiteVector(fusion.real_center, 'asset.fusion.real_center');
  const complexCenter = finiteVector(
    fusion.complex_center,
    'asset.fusion.complex_center',
    realCenter.length,
  );
  const weightReal = finite(fusion.weight_real, 'asset.fusion.weight_real');
  const eps = finite(fusion.eps, 'asset.fusion.eps');
  if (weightReal < 0 || weightReal > 1 || eps <= 0) {
    throw new RangeError('asset.fusion weight/eps are invalid');
  }
  finite(fusion.alpha_real, 'asset.fusion.alpha_real');
  finite(fusion.alpha_complex, 'asset.fusion.alpha_complex');

  const classification = record(asset.classification, 'asset.classification');
  if (classification.distance !== 'squared_euclidean') {
    throw new RangeError('asset.classification.distance must be squared_euclidean');
  }
  if (
    !Array.isArray(classification.classes)
    || classification.classes.length === 0
    || classification.classes.some(
      (name) => typeof name !== 'string' || name.length === 0,
    )
    || new Set(classification.classes).size !== classification.classes.length
  ) {
    throw new RangeError('asset.classification.classes must be unique non-empty strings');
  }
  if (
    !Array.isArray(classification.prototypes)
    || classification.prototypes.length !== classification.classes.length
  ) {
    throw new RangeError('asset.classification.prototypes must have one row per class');
  }
  const embedDim = realCenter.length;
  classification.prototypes.forEach((prototype, index) => {
    finiteVector(
      prototype,
      `asset.classification.prototypes[${index}]`,
      2 * embedDim,
    );
  });
  return asset as unknown as TimeDomainClassifierAssetV3;
}

// ---------------------------------------------------------------------------
// forward math (fusion + nearest prototype)
// ---------------------------------------------------------------------------

/**
 * Standardize the raw 12-feature vector with float32 semantics: the Python
 * cache subtracts and divides in NumPy float32, so both intermediate results
 * are rounded to float32 here too.
 */
export function standardizeTimeDomainFeatures(
  asset: TimeDomainClassifierAssetV3,
  rawFeatures: ArrayLike<number>,
): Float64Array {
  const { mean, std } = asset.feature_standardization;
  if (rawFeatures.length !== mean.length) {
    throw new RangeError(`rawFeatures must have length ${mean.length}`);
  }
  const output = new Float64Array(mean.length);
  for (let index = 0; index < mean.length; index++) {
    const value = rawFeatures[index]!;
    if (!Number.isFinite(value)) {
      throw new RangeError(`rawFeatures[${index}] must be finite`);
    }
    const centered = Math.fround(value - mean[index]!);
    output[index] = Math.fround(centered / std[index]!);
  }
  return output;
}

function safeUnit(value: Float64Array, eps: number): Float64Array {
  let normSquared = 0;
  for (const item of value) normSquared += item * item;
  const norm = Math.sqrt(normSquared);
  if (norm <= eps) {
    const fallback = new Float64Array(value.length);
    fallback[0] = 1;
    return fallback;
  }
  const output = new Float64Array(value.length);
  for (let index = 0; index < value.length; index++) {
    output[index] = value[index]! / norm;
  }
  return output;
}

/**
 * Centered invariant fusion
 * (`assemble_v3_fusion.fuse_numpy`):
 * `unit(concat(sqrt(w) * unit(real - a_r * c_r), sqrt(1-w) * unit(cx - a_c * c_c)))`.
 */
export function fuseTimeDomainEmbeddings(
  asset: TimeDomainClassifierAssetV3,
  realEmbedding: ArrayLike<number>,
  complexEmbedding: ArrayLike<number>,
): Float64Array {
  const fusion = asset.fusion;
  const embedDim = fusion.real_center.length;
  if (realEmbedding.length !== embedDim || complexEmbedding.length !== embedDim) {
    throw new RangeError(`branch embeddings must have ${embedDim} dimensions`);
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
  const unitReal = safeUnit(centeredReal, fusion.eps);
  const unitComplex = safeUnit(centeredComplex, fusion.eps);
  const fused = new Float64Array(2 * embedDim);
  const realScale = Math.sqrt(fusion.weight_real);
  const complexScale = Math.sqrt(1 - fusion.weight_real);
  for (let index = 0; index < embedDim; index++) {
    fused[index] = realScale * unitReal[index]!;
    fused[embedDim + index] = complexScale * unitComplex[index]!;
  }
  return safeUnit(fused, fusion.eps);
}

export interface TimeDomainNearestPrototype {
  winnerIndex: number;
  label: string;
  squaredDistances: Float64Array;
}

/** Nearest fused prototype by squared euclidean distance (`train.nearest`). */
export function nearestTimeDomainPrototype(
  asset: TimeDomainClassifierAssetV3,
  fusedEmbedding: Float64Array,
): TimeDomainNearestPrototype {
  const prototypes = asset.classification.prototypes;
  const squaredDistances = new Float64Array(prototypes.length);
  let winnerIndex = 0;
  for (let classIndex = 0; classIndex < prototypes.length; classIndex++) {
    const prototype = prototypes[classIndex]!;
    let distance = 0;
    for (let dimension = 0; dimension < fusedEmbedding.length; dimension++) {
      const delta = fusedEmbedding[dimension]! - prototype[dimension]!;
      distance += delta * delta;
    }
    squaredDistances[classIndex] = distance;
    if (distance < squaredDistances[winnerIndex]!) winnerIndex = classIndex;
  }
  return {
    winnerIndex,
    label: asset.classification.classes[winnerIndex]!,
    squaredDistances,
  };
}

// ---------------------------------------------------------------------------
// the composed classifier
// ---------------------------------------------------------------------------

export interface TimeDomainClassificationV3 {
  /**
   * `noise` when stage 1 gated the row, `unknown` when stage 2 rejected it,
   * otherwise the closed nearest-prototype class label.
   */
  label: string | 'noise' | 'unknown';
  /**
   * The closed nearest-prototype label. Null exactly when stage 1 gated the
   * row: classification never ran, so no closed label exists at all. Stage 2
   * NEVER alters this label; rejection only replaces the reported `label`
   * with `unknown`.
   */
  closedLabel: string | null;
  closedWinnerIndex: number | null;
  squaredPrototypeDistances: Float64Array | null;
  /** 1 = stage-1 noise gate, 2 = stage-2 unknown, null = accepted. */
  rejectedStage: StagedRejectionStage;
  openSet: StagedOpenSetDecision;
  /** Null exactly when stage 1 gated the row (nothing downstream ran). */
  forward: {
    preprocess: TimeDomainInvariantPatchResult;
    standardizedFeatures: Float64Array;
    realEmbedding: Float64Array;
    complexEmbedding: Float64Array;
    fusedEmbedding: Float64Array;
  } | null;
  /** The staged architecture contract, restated on every decision. */
  contract: StagedArchitectureContract;
}

export interface TimeDomainClassifierV3Options {
  classifierAsset: TimeDomainClassifierAssetV3;
  opensetAsset: TimeDomainOpenSetAssetV3;
  encoders: TimeDomainEncoderPairV3;
}

function assertExactVector(
  left: readonly number[],
  right: readonly number[],
  path: string,
): void {
  if (
    left.length !== right.length
    || left.some((value, index) => value !== right[index])
  ) {
    throw new RangeError(
      `${path} differs between the classifier and encoder assets`,
    );
  }
}

/**
 * Prove that the independently exported encoder, classifier, and open-set
 * files describe one deployable model. Shared arrays are exact. Scalar
 * constants allow only one float32 round-trip because the encoder export
 * preserves state-dict float32 while the decision export records the
 * equivalent source constants (for example 0.20000000298 versus 0.2).
 */
export function assertTimeDomainAssetSetV3(
  classifier: TimeDomainClassifierAssetV3,
  openset: TimeDomainOpenSetAssetV3,
  encoder: TimeDomainFusionAssetV3,
): void {
  if (
    classifier.status !== openset.status
    || classifier.status !== encoder.status
  ) {
    throw new RangeError('v3 asset statuses do not match');
  }
  const frontend = classifier.frontend;
  const realConfig = encoder.real.config;
  const complexConfig = encoder.complex.config;
  if (
    frontend.packed_length !== encoder.packed_length
    || frontend.patch_length !== realConfig.patch_length
    || frontend.patch_length !== complexConfig.patch_length
    || frontend.patch_count !== realConfig.patch_count
    || frontend.patch_count !== complexConfig.patch_count
    || frontend.patch_length !== openset.frontend.patch_length
    || frontend.patch_count !== openset.frontend.patch_count
    || frontend.target_frac !== openset.frontend.target_frac
    || frontend.packed_length !== openset.frontend.packed_length
  ) {
    throw new RangeError('v3 asset frontend geometries do not match');
  }
  assertExactVector(
    classifier.feature_standardization.mean,
    encoder.feature_standardization.mean,
    'feature_standardization.mean',
  );
  assertExactVector(
    classifier.feature_standardization.std,
    encoder.feature_standardization.std,
    'feature_standardization.std',
  );
  assertExactVector(
    classifier.fusion.real_center,
    encoder.fusion.real_center,
    'fusion.real_center',
  );
  assertExactVector(
    classifier.fusion.complex_center,
    encoder.fusion.complex_center,
    'fusion.complex_center',
  );
  for (const field of [
    'alpha_real',
    'alpha_complex',
    'weight_real',
    'eps',
  ] as const) {
    const left = classifier.fusion[field];
    const right = encoder.fusion[field];
    const tolerance = Math.max(1e-15, 1e-7 * Math.max(Math.abs(left), Math.abs(right)));
    if (Math.abs(left - right) > tolerance) {
      throw new RangeError(
        `fusion.${field} differs between the classifier and encoder assets`,
      );
    }
  }
  if (
    classifier.classification.classes.length
      !== encoder.classification.classes.length
    || classifier.classification.classes.some(
      (value, index) => value !== encoder.classification.classes[index],
    )
  ) {
    throw new RangeError('classification classes differ across v3 assets');
  }
  classifier.classification.prototypes.forEach((prototype, index) => {
    assertExactVector(
      prototype,
      encoder.classification.prototypes[index] ?? [],
      `classification.prototypes[${index}]`,
    );
  });
}

export class TimeDomainClassifierV3 {
  readonly asset: TimeDomainClassifierAssetV3;
  readonly openSet: TimeDomainOpenSetV3;
  readonly encoders: TimeDomainEncoderPairV3;

  constructor(options: TimeDomainClassifierV3Options) {
    this.asset = options.classifierAsset;
    this.openSet = new TimeDomainOpenSetV3(options.opensetAsset);
    this.encoders = options.encoders;
    if (this.asset.status !== options.opensetAsset.status) {
      throw new RangeError('classifier and open-set asset statuses do not match');
    }
    const classifierFrontend = this.asset.frontend;
    const opensetFrontend = options.opensetAsset.frontend;
    if (
      classifierFrontend.patch_length !== opensetFrontend.patch_length
      || classifierFrontend.patch_count !== opensetFrontend.patch_count
      || classifierFrontend.target_frac !== opensetFrontend.target_frac
    ) {
      throw new RangeError(
        'classifier and open-set assets disagree about the frontend geometry',
      );
    }
    const classCount = this.asset.classification.classes.length;
    if (this.openSet.stageTwo.classCount !== classCount) {
      throw new RangeError(
        'classifier and open-set assets disagree about the class count',
      );
    }
  }

  /**
   * Classify one raw complex capture.
   *
   * Stage 1 runs FIRST. When it fires, the method returns immediately: the
   * frontend, the encoders, the fusion, the prototype distance, and the
   * branch LOF are all skipped (`gates_before_classification`).
   */
  classify(
    inPhase: ArrayLike<number>,
    quadrature: ArrayLike<number>,
  ): TimeDomainClassificationV3 {
    const stageOne: StageOneEvaluation = this.openSet.evaluateStageOne(
      inPhase,
      quadrature,
    );
    if (stageOne.gated) {
      return {
        label: 'noise',
        closedLabel: null,
        closedWinnerIndex: null,
        squaredPrototypeDistances: null,
        rejectedStage: 1,
        openSet: this.openSet.gatedDecision(stageOne),
        forward: null,
        contract: stagedArchitectureContract(),
      };
    }

    const frontend = this.asset.frontend;
    const preprocess = preprocessTimeDomainInvariantPatches(inPhase, quadrature, {
      patchLength: frontend.patch_length,
      patchCount: frontend.patch_count,
      targetFrac: frontend.target_frac,
    });
    const standardizedFeatures = standardizeTimeDomainFeatures(
      this.asset,
      preprocess.rawFeatures,
    );
    const realEmbedding = this.encoders.real(
      preprocess.packedInPhase,
      preprocess.packedQuadrature,
      standardizedFeatures,
    );
    const complexEmbedding = this.encoders.complex(
      preprocess.packedInPhase,
      preprocess.packedQuadrature,
      standardizedFeatures,
    );
    const fusedEmbedding = fuseTimeDomainEmbeddings(
      this.asset,
      realEmbedding,
      complexEmbedding,
    );
    const nearest = nearestTimeDomainPrototype(this.asset, fusedEmbedding);
    const openSet = this.openSet.finishSurvivor(stageOne, {
      realEmbedding,
      complexEmbedding,
      packedInPhase: preprocess.packedInPhase,
      packedQuadrature: preprocess.packedQuadrature,
      predictedClassIndex: nearest.winnerIndex,
    });
    const rejected = openSet.rejectedStage === 2;
    return {
      label: rejected ? 'unknown' : nearest.label,
      closedLabel: nearest.label,
      closedWinnerIndex: nearest.winnerIndex,
      squaredPrototypeDistances: nearest.squaredDistances,
      rejectedStage: openSet.rejectedStage,
      openSet,
      forward: {
        preprocess,
        standardizedFeatures,
        realEmbedding,
        complexEmbedding,
        fusedEmbedding,
      },
      contract: stagedArchitectureContract(),
    };
  }
}

/**
 * Convenience factory: validate both JSON assets and load the sibling
 * encoder module (failing loudly when it is absent).
 */
export async function createTimeDomainClassifierV3(inputs: {
  classifierAsset: unknown;
  opensetAsset: unknown;
  encoderAsset: unknown;
  /**
   * Production admission refuses every `staging_not_release` asset. The
   * default remains staging for development tools and existing callers.
   */
  admission?: TimeDomainAssetAdmissionV3;
  /** Test seam: inject decision-layer encoders instead of the sibling port. */
  encoders?: TimeDomainEncoderPairV3;
}): Promise<TimeDomainClassifierV3> {
  const loadOptions = { admission: inputs.admission ?? 'staging' } as const;
  const classifierAsset = loadTimeDomainClassifierAssetV3(
    inputs.classifierAsset,
    loadOptions,
  );
  const opensetAsset = loadTimeDomainOpenSetAssetV3(
    inputs.opensetAsset,
    loadOptions,
  );
  const encoderAsset = loadTimeDomainFusionAssetV3(
    inputs.encoderAsset,
    loadOptions,
  );
  assertTimeDomainAssetSetV3(classifierAsset, opensetAsset, encoderAsset);
  const encoders = (
    inputs.encoders ?? (await loadTimeDomainEncodersV3(encoderAsset))
  );
  return new TimeDomainClassifierV3({ classifierAsset, opensetAsset, encoders });
}
