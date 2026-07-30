/**
 * Route-conditioned, FFT-free open-set wrapper for the v4 profile-bank head.
 *
 * The external policy is byte-bound to one classifier JSON. Every nonzero
 * decision is conditioned on exactly `(prototype source, runtime length,
 * predicted public class)`. There is no union-bank or unsupported-class
 * fallback.
 */

import {
  classifyTimeDomainProfileBankV4,
  loadTimeDomainProfileBankAssetV4,
  minimumSourcePublicClassSquaredDistancesV4,
  selectTimeDomainRuntimeInputLengthV4,
  TIME_DOMAIN_PROFILE_BANK_DECISION_RULE_V4,
  TIME_DOMAIN_PROFILE_BANK_RUNTIME_ROLE_V4,
  TIME_DOMAIN_PROFILE_BANK_SCHEMA_V4,
  TIME_DOMAIN_PROFILE_BANK_SCHEMA_VERSION_V4,
  TIME_DOMAIN_PUBLIC_CLASSES_V4,
  TIME_DOMAIN_RUNTIME_BUCKET_RULE_V4,
  TIME_DOMAIN_RUNTIME_INPUT_LENGTHS_V4,
  type TimeDomainProfileBankAssetV4,
  type TimeDomainProfileBankDecisionV4,
  type TimeDomainRuntimeInputLengthV4,
} from './time-domain-profile-bank-classifier-v4.js';
import {
  empiricalRank,
  frontendMinimumBandwidth,
  PREFILTER_FEATURE_NAMES,
  prefilterPoseDegeneracyFeatures,
} from './time-domain-openset-v3.js';
import type { TimeDomainAssetLoadOptionsV3 } from './time-domain-asset-status-v3.js';
import type { TimeDomainAssetAdmissionV3 } from './time-domain-asset-status-v3.js';
import {
  TIME_DOMAIN_TRUSTED_SOURCE_ROUTING_V4,
  type TimeDomainPrototypeSourceV4,
  type TimeDomainPublicClassV4,
  type TimeDomainTrustedSourceRoutingV4,
} from './time-domain-profile-routing-v4.js';

export const TIME_DOMAIN_PROFILE_BANK_OPENSET_SCHEMA_V4 =
  'atomos.v4.time-domain-current-source.external-openset-policy' as const;
export const TIME_DOMAIN_PROFILE_BANK_OPENSET_SCHEMA_VERSION_V4 = 5 as const;
export const TIME_DOMAIN_PROFILE_BANK_OPENSET_STATUS_V4 =
  'development_external_policy' as const;
export const TIME_DOMAIN_PROFILE_BANK_OPENSET_RELEASE_STATUS_V4 =
  'release' as const;
export const TIME_DOMAIN_PROFILE_BANK_OPENSET_RUNTIME_ROLE_V4 =
  'external_abstention_policy' as const;
export const TIME_DOMAIN_PROFILE_BANK_OPENSET_STAGE_ZERO_KIND_V4 =
  'exact-zero-no-signal-validity-gate' as const;
export const TIME_DOMAIN_PROFILE_BANK_OPENSET_STAGE_ZERO_DECISION_V4 =
  'reject as no_signal iff max(abs(effective_route_scoped_inference_prefix_iq)) == 0' as const;
export const TIME_DOMAIN_PROFILE_BANK_OPENSET_STAGE_ZERO_PREFIX_SCOPE_V4 =
  'current:first_4096_samples;historical:largest_supported_prefix_not_exceeding_valid_sample_count' as const;
export const TIME_DOMAIN_PROFILE_BANK_OPENSET_STAGE_ONE_KIND_V4 =
  'class-profile-base-balanced-pose-degeneracy-noise-score-v1' as const;
export const TIME_DOMAIN_PROFILE_BANK_OPENSET_STAGE_ONE_RANK_KIND_V4 =
  'max-predicted-class-and-route-length-pooled-enrollment-empirical-rank-v1' as const;
export const TIME_DOMAIN_PROFILE_BANK_OPENSET_STAGE_TWO_KIND_V4 =
  'winning-public-class-profile-distance-enrollment-empirical-rank' as const;
export const TIME_DOMAIN_PROFILE_BANK_OPENSET_STAGE_CHIRP_KIND_V4 =
  'weighted-instantaneous-phase-linearity-enrollment-rank-v1' as const;
export const TIME_DOMAIN_PROFILE_BANK_OPENSET_COMPOSITION_KIND_V4 =
  'max-effective-pose-score-rank-profile-distance-rank-phase-linearity-rank' as const;
export const TIME_DOMAIN_PROFILE_BANK_EMPIRICAL_RANK_RULE_V4:
  "searchsorted(sorted_enrollment, value, side='left') / (enrollment_count + 1)" =
  "searchsorted(sorted_enrollment, value, side='left') / (enrollment_count + 1)";

const PROTOTYPE_SOURCES = ['historical', 'current'] as const;
export const TIME_DOMAIN_PROFILE_BANK_OPENSET_CURRENT_RUNTIME_LENGTH_V4 =
  4_096 as const;
export const TIME_DOMAIN_PROFILE_BANK_OPENSET_RUNTIME_LENGTH_POLICY_V4 =
  Object.freeze({
    historical: Object.freeze({
      kind: 'largest_supported_prefix_not_exceeding_valid_sample_count',
      observation_length_reporting_rule:
        TIME_DOMAIN_RUNTIME_BUCKET_RULE_V4,
    }),
    current: Object.freeze({
      kind: 'fixed_causal_prefix',
      effective_runtime_input_length:
        TIME_DOMAIN_PROFILE_BANK_OPENSET_CURRENT_RUNTIME_LENGTH_V4,
      observation_length_reporting_rule:
        TIME_DOMAIN_RUNTIME_BUCKET_RULE_V4,
    }),
  } as const);
const SHA256_PATTERN = /^[0-9a-f]{64}$/;
export const PHASE_LINEARITY_ACTIVE_RMS_FLOOR_V4 = 0.02 as const;
export const PHASE_LINEARITY_MIN_CONTIGUOUS_RUN_V4 = 32 as const;
export const PHASE_LINEARITY_MIN_VALID_ADJACENCIES_V4 = 128 as const;
export const PHASE_LINEARITY_MIN_VALID_FRACTION_V4 = 0.25 as const;
export const PHASE_LINEARITY_WEIGHT_CEILING_V4 = 16 as const;
export const PHASE_LINEARITY_PHASE_STD_FLOOR_RAD_V4 = 1e-4 as const;
const EXPECTED_SUPPORT: Readonly<
  Record<TimeDomainPrototypeSourceV4, readonly boolean[]>
> = Object.freeze({
  historical: Object.freeze([true, true, true, true, true, true, true]),
  current: Object.freeze([false, true, false, true, false, true, true]),
});

interface StageOneParametersV4 {
  mean: number[];
  scale: number[];
  coefficients: number[];
  intercept: number;
}

interface RankCalibrationV4 {
  byPredictedClass: number[][];
  pooled: number[];
}

interface RouteLengthPolicyV4 {
  enrollmentRows: number;
  stageOne: RankCalibrationV4;
  stageTwo: RankCalibrationV4;
  stageChirp: RankCalibrationV4;
  thresholdByPredictedClass: number[];
}

interface RoutePolicyV4 {
  supportedPublicClassMask: readonly boolean[];
  perLength: Readonly<
    Partial<
      Record<`${TimeDomainRuntimeInputLengthV4}`, RouteLengthPolicyV4>
    >
  >;
}

export interface TimeDomainProfileBankOpenSetPolicyV4 {
  schema: typeof TIME_DOMAIN_PROFILE_BANK_OPENSET_SCHEMA_V4;
  schemaVersion: typeof TIME_DOMAIN_PROFILE_BANK_OPENSET_SCHEMA_VERSION_V4;
  classifierAssetSha256: string;
  sourceFusionArtifactsSha256: Readonly<Record<string, string>>;
  frontend: {
    version: 'invariant-patch-time-domain-v1';
    patchLength: number;
    patchCount: number;
    targetFrac: number;
    usesFrequencyTransform: false;
  };
  stageOneParameters: Readonly<
    Record<`${TimeDomainRuntimeInputLengthV4}`, StageOneParametersV4>
  >;
  runtimeLengthPolicyByPrototypeSource:
    typeof TIME_DOMAIN_PROFILE_BANK_OPENSET_RUNTIME_LENGTH_POLICY_V4;
  routes: Readonly<Record<TimeDomainPrototypeSourceV4, RoutePolicyV4>>;
  acquisitionRouting: Readonly<TimeDomainTrustedSourceRoutingV4>;
}

export interface LoadBoundTimeDomainOpenSetBundleV4Options
  extends TimeDomainAssetLoadOptionsV3 {
  /** Expected raw-byte hash from the deployment manifest. Required. */
  expectedClassifierSha256: string;
  /** Expected raw-byte hash from the deployment manifest. Required. */
  expectedPolicySha256: string;
}

export interface BoundTimeDomainOpenSetBundleV4 {
  classifier: TimeDomainProfileBankAssetV4;
  policy: TimeDomainProfileBankOpenSetPolicyV4;
  artifactHashes: Readonly<{
    classifierSha256: string;
    policySha256: string;
  }>;
}

export interface TimeDomainOpenSetRankV4 {
  /** `lessCount / denominator`, using strict lower-bound tie behavior. */
  value: number;
  lessCount: number;
  calibrationCount: number;
  denominator: number;
  calibrationScope:
    | 'predicted_class'
    | 'pooled_fallback'
    | 'route_length_pooled';
}

export interface TimeDomainOpenSetClosedDecisionV4 {
  prototypeSource: TimeDomainPrototypeSourceV4;
  runtimeInputLength: TimeDomainRuntimeInputLengthV4;
  predictedPublicClassIndex: number;
  predictedPublicClass: TimeDomainPublicClassV4;
  winningSquaredDistance: number;
  winningPrototypeIndex: number;
  squaredPublicClassDistances: Float64Array;
  closestPrototypeIndices: Int32Array;
  supportedPublicClassMask: readonly boolean[];
}

interface TimeDomainOpenSetScoredBaseV4 {
  stageZero: false;
  prototypeSource: TimeDomainPrototypeSourceV4;
  /** Largest supported observation bucket before route-scoped canonicalizing. */
  observationInputLength: TimeDomainRuntimeInputLengthV4;
  /** Effective inference prefix after applying the trusted-source policy. */
  runtimeInputLength: TimeDomainRuntimeInputLengthV4;
  bucketKey: string;
  closedDecision: TimeDomainOpenSetClosedDecisionV4;
  /**
   * FFT-free frontend bandwidth estimate. Raw-I/Q classification supplies it;
   * compact embedding-only parity scoring cannot reconstruct it.
   */
  occupiedBandwidthFraction: number | null;
  stageOneScore: number;
  stageOnePredictedClassRank: TimeDomainOpenSetRankV4;
  stageOnePooledRank: TimeDomainOpenSetRankV4;
  /** Effective maximum of class-conditioned and route-length-pooled ranks. */
  stageOneRank: TimeDomainOpenSetRankV4;
  stageTwoRank: TimeDomainOpenSetRankV4;
  instantaneousPhaseLinearityScore: number;
  instantaneousPhaseLinearityRank: TimeDomainOpenSetRankV4;
  compositeRank: number;
  threshold: number;
  artifactHashes: BoundTimeDomainOpenSetBundleV4['artifactHashes'];
}

export interface TimeDomainOpenSetAcceptedV4
  extends TimeDomainOpenSetScoredBaseV4 {
  disposition: 'accepted_known';
  reason: 'below_route_length_class_threshold';
}

export interface TimeDomainOpenSetRejectedV4
  extends TimeDomainOpenSetScoredBaseV4 {
  disposition: 'unknown';
  reason: 'at_or_above_route_length_class_threshold';
}

export interface TimeDomainOpenSetExactNoSignalV4 {
  disposition: 'unknown';
  reason: 'exact_no_signal';
  stageZero: true;
  prototypeSource: TimeDomainPrototypeSourceV4;
  observationInputLength: TimeDomainRuntimeInputLengthV4;
  runtimeInputLength: TimeDomainRuntimeInputLengthV4;
  bucketKey: string;
  closedDecision: null;
  /** No pose/bandwidth estimator runs for exact no-signal. */
  occupiedBandwidthFraction: null;
  artifactHashes: BoundTimeDomainOpenSetBundleV4['artifactHashes'];
}

export type TimeDomainOpenSetDecisionV4 =
  | TimeDomainOpenSetAcceptedV4
  | TimeDomainOpenSetRejectedV4
  | TimeDomainOpenSetExactNoSignalV4;

export interface TimeDomainOpenSetClassifyOptionsV4 {
  /** Trusted provenance selected by the controller. Never inferred from I/Q. */
  prototypeSource: TimeDomainPrototypeSourceV4;
  validSampleCount?: number;
}

export interface TimeDomainOpenSetEmbeddingOptionsV4 {
  /** Trusted provenance selected by the controller. Never inferred. */
  prototypeSource: TimeDomainPrototypeSourceV4;
  runtimeInputLength: TimeDomainRuntimeInputLengthV4;
  /** Precomputed raw-I/Q phase-linearity score for compact parity only. */
  instantaneousPhaseLinearityScore: number;
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

function positiveInteger(value: unknown, path: string): number {
  const result = finite(value, path);
  if (!Number.isSafeInteger(result) || result <= 0) {
    throw new RangeError(`${path} must be a positive safe integer`);
  }
  return result;
}

function exactSha256(value: unknown, path: string): string {
  if (typeof value !== 'string' || !SHA256_PATTERN.test(value)) {
    throw new RangeError(`${path} must be a lowercase SHA-256 digest`);
  }
  return value;
}

function exactKeys(value: UnknownRecord, keys: readonly string[], path: string): void {
  const observed = Object.keys(value).sort();
  const expected = [...keys].sort();
  if (
    observed.length !== expected.length
    || observed.some((key, index) => key !== expected[index])
  ) {
    throw new RangeError(`${path} must contain exactly ${JSON.stringify(expected)}`);
  }
}

function exactArray(
  value: unknown,
  expected: readonly unknown[],
  path: string,
): void {
  if (
    !Array.isArray(value)
    || value.length !== expected.length
    || value.some((entry, index) => entry !== expected[index])
  ) {
    throw new RangeError(`${path} differs from the canonical runtime order`);
  }
}

function finiteVector(
  value: unknown,
  path: string,
  length: number,
  options: { positive?: boolean; nonnegative?: boolean } = {},
): number[] {
  if (!Array.isArray(value) || value.length !== length) {
    throw new RangeError(`${path} must contain ${length} values`);
  }
  return value.map((entry, index) => {
    const number = finite(entry, `${path}[${index}]`);
    if (options.positive === true && number <= 0) {
      throw new RangeError(`${path}[${index}] must be positive`);
    }
    if (options.nonnegative === true && number < 0) {
      throw new RangeError(`${path}[${index}] must be nonnegative`);
    }
    return number;
  });
}

function sortedCalibration(
  value: unknown,
  path: string,
  options: { allowEmpty?: boolean; nonnegative?: boolean } = {},
): number[] {
  if (!Array.isArray(value) || (value.length === 0 && options.allowEmpty !== true)) {
    throw new RangeError(`${path} must be a nonempty calibration array`);
  }
  const result: number[] = [];
  let previous = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < value.length; index++) {
    const number = finite(value[index], `${path}[${index}]`);
    if (number < previous) {
      throw new RangeError(`${path} must be sorted ascending`);
    }
    if (options.nonnegative === true && number < 0) {
      throw new RangeError(`${path} distances must be nonnegative`);
    }
    previous = number;
    result.push(number);
  }
  return result;
}

function admittedSource(value: unknown): TimeDomainPrototypeSourceV4 {
  if (value !== 'current' && value !== 'historical') {
    throw new RangeError('prototypeSource must be current or historical');
  }
  return value;
}

function effectiveRuntimeInputLength(
  policy: TimeDomainProfileBankOpenSetPolicyV4,
  source: TimeDomainPrototypeSourceV4,
  observationLength: TimeDomainRuntimeInputLengthV4,
): TimeDomainRuntimeInputLengthV4 {
  if (
    policy.runtimeLengthPolicyByPrototypeSource
      !== TIME_DOMAIN_PROFILE_BANK_OPENSET_RUNTIME_LENGTH_POLICY_V4
  ) {
    throw new RangeError('runtime-length policy object is not admitted');
  }
  return source === 'current'
    ? TIME_DOMAIN_PROFILE_BANK_OPENSET_CURRENT_RUNTIME_LENGTH_V4
    : observationLength;
}

function deepJsonEqual(left: unknown, right: unknown): boolean {
  if (left === right) return true;
  if (Array.isArray(left) || Array.isArray(right)) {
    return (
      Array.isArray(left)
      && Array.isArray(right)
      && left.length === right.length
      && left.every((entry, index) => deepJsonEqual(entry, right[index]))
    );
  }
  if (
    typeof left === 'object'
    && left !== null
    && typeof right === 'object'
    && right !== null
  ) {
    const leftRecord = left as UnknownRecord;
    const rightRecord = right as UnknownRecord;
    const leftKeys = Object.keys(leftRecord).sort();
    const rightKeys = Object.keys(rightRecord).sort();
    return (
      leftKeys.length === rightKeys.length
      && leftKeys.every((key, index) =>
        key === rightKeys[index]
        && deepJsonEqual(leftRecord[key], rightRecord[key]))
    );
  }
  return false;
}

function bytes(value: Uint8Array, path: string): Uint8Array<ArrayBuffer> {
  if (!(value instanceof Uint8Array) || value.length === 0) {
    throw new TypeError(`${path} must be a nonempty Uint8Array`);
  }
  return Uint8Array.from(value);
}

async function sha256Hex(value: Uint8Array): Promise<string> {
  const cryptoObject = globalThis.crypto;
  if (cryptoObject?.subtle === undefined) {
    throw new Error('Web Crypto SHA-256 is unavailable');
  }
  const digest = await cryptoObject.subtle.digest('SHA-256', bytes(value, 'asset bytes'));
  return [...new Uint8Array(digest)]
    .map((entry) => entry.toString(16).padStart(2, '0'))
    .join('');
}

function parseJsonBytes(value: Uint8Array, path: string): unknown {
  let text: string;
  try {
    text = new TextDecoder('utf-8', { fatal: true }).decode(bytes(value, path));
  } catch (error) {
    throw new TypeError(`${path} is not valid UTF-8`, { cause: error });
  }
  try {
    return JSON.parse(text) as unknown;
  } catch (error) {
    throw new TypeError(`${path} is not valid JSON`, { cause: error });
  }
}

function sourceSupportFromClassifier(
  classifier: TimeDomainProfileBankAssetV4,
): Record<TimeDomainPrototypeSourceV4, boolean[]> {
  const result: Record<TimeDomainPrototypeSourceV4, boolean[]> = {
    historical: Array<boolean>(TIME_DOMAIN_PUBLIC_CLASSES_V4.length).fill(false),
    current: Array<boolean>(TIME_DOMAIN_PUBLIC_CLASSES_V4.length).fill(false),
  };
  for (let index = 0; index < classifier.classification.prototype_groups.length; index++) {
    const group = classifier.classification.prototype_groups[index]!;
    const classIndex =
      classifier.classification.prototype_public_class_indices[index]!;
    result[group.source][classIndex] = true;
  }
  for (const source of PROTOTYPE_SOURCES) {
    if (!deepJsonEqual(result[source], EXPECTED_SUPPORT[source])) {
      throw new RangeError(
        `classifier ${source} source support differs from the v4 contract`,
      );
    }
  }
  return result;
}

function loadStageOneParameters(
  value: unknown,
): Readonly<Record<`${TimeDomainRuntimeInputLengthV4}`, StageOneParametersV4>> {
  const parameters = record(value, 'policy.stage_one.parameters_by_length');
  exactKeys(
    parameters,
    TIME_DOMAIN_RUNTIME_INPUT_LENGTHS_V4.map(String),
    'policy.stage_one.parameters_by_length',
  );
  const output = {} as Record<
    `${TimeDomainRuntimeInputLengthV4}`,
    StageOneParametersV4
  >;
  for (const length of TIME_DOMAIN_RUNTIME_INPUT_LENGTHS_V4) {
    const path = `policy.stage_one.parameters_by_length.${length}`;
    const row = record(parameters[String(length)], path);
    const width = PREFILTER_FEATURE_NAMES.length;
    output[String(length) as `${TimeDomainRuntimeInputLengthV4}`] = {
      mean: finiteVector(row.mean, `${path}.mean`, width),
      scale: finiteVector(row.scale, `${path}.scale`, width, { positive: true }),
      coefficients: finiteVector(row.coefficients, `${path}.coefficients`, width),
      intercept: finite(row.intercept, `${path}.intercept`),
    };
  }
  return Object.freeze(output);
}

function loadRankCalibrations(
  stage: UnknownRecord,
  path: string,
  classKey: string,
  pooledKey: string,
  fallbackKey: string,
  fallbackValue: string,
  options: { nonnegative?: boolean } = {},
): RankCalibrationV4 {
  if (stage[fallbackKey] !== fallbackValue) {
    throw new RangeError(`${path}.${fallbackKey} must be ${fallbackValue}`);
  }
  const rawByClass = stage[classKey];
  if (
    !Array.isArray(rawByClass)
    || rawByClass.length !== TIME_DOMAIN_PUBLIC_CLASSES_V4.length
  ) {
    throw new RangeError(`${path}.${classKey} must have seven class arrays`);
  }
  return {
    byPredictedClass: rawByClass.map((entry, classIndex) =>
      sortedCalibration(entry, `${path}.${classKey}[${classIndex}]`, {
        allowEmpty: true,
        nonnegative: options.nonnegative,
      })),
    pooled: sortedCalibration(stage[pooledKey], `${path}.${pooledKey}`, {
      nonnegative: options.nonnegative,
    }),
  };
}

function loadRouteLengthPolicy(
  value: unknown,
  source: TimeDomainPrototypeSourceV4,
  length: TimeDomainRuntimeInputLengthV4,
  support: readonly boolean[],
): RouteLengthPolicyV4 {
  const path = `policy.per_prototype_source.${source}.per_length.${length}`;
  const row = record(value, path);
  if (row.prototype_source !== source) {
    throw new RangeError(`${path}.prototype_source must be ${source}`);
  }
  const enrollmentRows = positiveInteger(row.enrollment_rows, `${path}.enrollment_rows`);
  const stageOneRaw = record(row.stage_one, `${path}.stage_one`);
  if (
    stageOneRaw.hard_gate_enabled !== false
    || stageOneRaw.hard_threshold_fitted !== false
    || stageOneRaw.predicted_class_rank_conditioning
      !== 'prototype_source_route_runtime_length_and_predicted_public_class'
    || stageOneRaw.pooled_rank_conditioning
      !== 'prototype_source_route_and_runtime_length'
    || stageOneRaw.pooled_rank_always_evaluated !== true
    || stageOneRaw.effective_rank_kind
      !== TIME_DOMAIN_PROFILE_BANK_OPENSET_STAGE_ONE_RANK_KIND_V4
    || stageOneRaw.effective_rank
      !== 'max(predicted_class_rank, route_length_pooled_rank)'
    || stageOneRaw.empirical_rank_rule
      !== TIME_DOMAIN_PROFILE_BANK_EMPIRICAL_RANK_RULE_V4
  ) {
    throw new RangeError(`${path}.stage_one rank contract changed`);
  }
  const stageOne = loadRankCalibrations(
    stageOneRaw,
    `${path}.stage_one`,
    'score_rank_calibration_by_predicted_class_sorted',
    'pooled_score_rank_calibration_sorted',
    'empty_predicted_class_fallback',
    'pooled_score',
  );
  const stageTwoRaw = record(row.stage_two, `${path}.stage_two`);
  if (
    stageTwoRaw.empty_predicted_class_fallback !== 'pooled_distance'
    || stageTwoRaw.empirical_rank_rule
      !== TIME_DOMAIN_PROFILE_BANK_EMPIRICAL_RANK_RULE_V4
    || stageTwoRaw.union_head_used !== false
  ) {
    throw new RangeError(`${path}.stage_two contract changed`);
  }
  const stageTwo = loadRankCalibrations(
    stageTwoRaw,
    `${path}.stage_two`,
    'predicted_class_distance_calibration_sorted',
    'pooled_distance_calibration_sorted',
    'empty_predicted_class_fallback',
    'pooled_distance',
    { nonnegative: true },
  );
  const stageChirpRaw = record(row.stage_chirp, `${path}.stage_chirp`);
  if (
    stageChirpRaw.empirical_rank_rule
      !== TIME_DOMAIN_PROFILE_BANK_EMPIRICAL_RANK_RULE_V4
  ) {
    throw new RangeError(`${path}.stage_chirp empirical-rank rule changed`);
  }
  const stageChirp = loadRankCalibrations(
    stageChirpRaw,
    `${path}.stage_chirp`,
    'score_rank_calibration_by_predicted_class_sorted',
    'pooled_score_rank_calibration_sorted',
    'empty_predicted_class_fallback',
    'pooled_score',
    { nonnegative: true },
  );
  if (
    stageChirp.pooled.some((value) => value > 1)
    || stageChirp.byPredictedClass.some((values) =>
      values.some((value) => value > 1))
  ) {
    throw new RangeError(`${path}.stage_chirp scores must lie in [0, 1]`);
  }
  const rawCounts = stageTwoRaw.predicted_class_calibration_counts;
  if (
    !Array.isArray(rawCounts)
    || rawCounts.length !== TIME_DOMAIN_PUBLIC_CLASSES_V4.length
  ) {
    throw new RangeError(`${path}.stage_two calibration counts are invalid`);
  }
  const counts = rawCounts.map((entry, classIndex) => {
    const count = finite(
      entry,
      `${path}.stage_two.predicted_class_calibration_counts[${classIndex}]`,
    );
    if (!Number.isSafeInteger(count) || count < 0) {
      throw new RangeError(`${path}.stage_two calibration count must be an integer`);
    }
    return count;
  });
  if (
    counts.reduce((sum, count) => sum + count, 0) !== enrollmentRows
    || counts.some((count, classIndex) =>
      stageOne.byPredictedClass[classIndex]!.length !== count
      || stageTwo.byPredictedClass[classIndex]!.length !== count
      || stageChirp.byPredictedClass[classIndex]!.length !== count
      || (!support[classIndex] && count !== 0))
  ) {
    throw new RangeError(`${path} calibration counts disagree with route support`);
  }
  const composition = record(row.composition, `${path}.composition`);
  const thresholds = finiteVector(
    composition.threshold_by_predicted_public_class,
    `${path}.composition.threshold_by_predicted_public_class`,
    TIME_DOMAIN_PUBLIC_CLASSES_V4.length,
    { nonnegative: true },
  );
  if (thresholds.some((threshold) => threshold > 1)) {
    throw new RangeError(`${path} thresholds must lie in [0, 1]`);
  }
  return {
    enrollmentRows,
    stageOne,
    stageTwo,
    stageChirp,
    thresholdByPredictedClass: thresholds,
  };
}

function loadRoutePolicies(
  value: unknown,
): Readonly<Record<TimeDomainPrototypeSourceV4, RoutePolicyV4>> {
  const routes = record(value, 'policy.per_prototype_source');
  exactKeys(routes, PROTOTYPE_SOURCES, 'policy.per_prototype_source');
  const output = {} as Record<TimeDomainPrototypeSourceV4, RoutePolicyV4>;
  for (const source of PROTOTYPE_SOURCES) {
    const path = `policy.per_prototype_source.${source}`;
    const sourceRow = record(routes[source], path);
    exactArray(
      sourceRow.supported_public_class_mask,
      EXPECTED_SUPPORT[source],
      `${path}.supported_public_class_mask`,
    );
    const perLengthRaw = record(sourceRow.per_length, `${path}.per_length`);
    const reachableLengths = source === 'current'
      ? [TIME_DOMAIN_PROFILE_BANK_OPENSET_CURRENT_RUNTIME_LENGTH_V4]
      : TIME_DOMAIN_RUNTIME_INPUT_LENGTHS_V4;
    exactKeys(
      perLengthRaw,
      reachableLengths.map(String),
      `${path}.per_length`,
    );
    const perLength = {} as Record<
      `${TimeDomainRuntimeInputLengthV4}`,
      RouteLengthPolicyV4
    >;
    for (const length of reachableLengths) {
      perLength[String(length) as `${TimeDomainRuntimeInputLengthV4}`] =
        loadRouteLengthPolicy(
          perLengthRaw[String(length)],
          source,
          length,
          EXPECTED_SUPPORT[source],
        );
    }
    output[source] = {
      supportedPublicClassMask: EXPECTED_SUPPORT[source],
      perLength: Object.freeze(perLength),
    };
  }
  return Object.freeze(output);
}

function loadBoundPolicy(
  value: unknown,
  classifier: TimeDomainProfileBankAssetV4,
  classifierSha256: string,
  admission: TimeDomainAssetAdmissionV3,
): TimeDomainProfileBankOpenSetPolicyV4 {
  const policy = record(value, 'policy');
  const expectedStatus = admission === 'production'
    ? TIME_DOMAIN_PROFILE_BANK_OPENSET_RELEASE_STATUS_V4
    : TIME_DOMAIN_PROFILE_BANK_OPENSET_STATUS_V4;
  const expectedDevelopmentOnly = admission !== 'production';
  const expectedReleaseEvidence = admission === 'production';
  if (
    policy.schema !== TIME_DOMAIN_PROFILE_BANK_OPENSET_SCHEMA_V4
    || policy.schema_version !== TIME_DOMAIN_PROFILE_BANK_OPENSET_SCHEMA_VERSION_V4
    || policy.status !== expectedStatus
    || policy.development_only !== expectedDevelopmentOnly
    || policy.release_evidence !== expectedReleaseEvidence
    || policy.runtime_role !== TIME_DOMAIN_PROFILE_BANK_OPENSET_RUNTIME_ROLE_V4
    || policy.classifier_runtime_role_required
      !== TIME_DOMAIN_PROFILE_BANK_RUNTIME_ROLE_V4
  ) {
    throw new RangeError(
      `unsupported v4 open-set policy for ${admission} admission`,
    );
  }
  exactArray(policy.public_classes, TIME_DOMAIN_PUBLIC_CLASSES_V4, 'policy.public_classes');
  exactArray(policy.prototype_source_routes, PROTOTYPE_SOURCES, 'policy.prototype_source_routes');
  exactArray(
    policy.runtime_input_lengths,
    TIME_DOMAIN_RUNTIME_INPUT_LENGTHS_V4,
    'policy.runtime_input_lengths',
  );
  if (policy.runtime_bucket_rule !== TIME_DOMAIN_RUNTIME_BUCKET_RULE_V4) {
    throw new RangeError('policy runtime bucket rule changed');
  }
  if (!deepJsonEqual(
    policy.runtime_length_policy_by_prototype_source,
    TIME_DOMAIN_PROFILE_BANK_OPENSET_RUNTIME_LENGTH_POLICY_V4,
  )) {
    throw new RangeError('policy route-scoped runtime-length policy changed');
  }

  const frontend = record(policy.frontend, 'policy.frontend');
  const patchLength = positiveInteger(frontend.patch_length, 'policy.frontend.patch_length');
  const patchCount = positiveInteger(frontend.patch_count, 'policy.frontend.patch_count');
  const targetFrac = finite(frontend.target_frac, 'policy.frontend.target_frac');
  if (
    frontend.version !== 'invariant-patch-time-domain-v1'
    || frontend.uses_frequency_transform !== false
    || targetFrac <= 0
    || targetFrac > 1
    || patchLength !== classifier.frontend.patch_length
    || patchCount !== classifier.frontend.patch_count
    || targetFrac !== classifier.frontend.target_frac
  ) {
    throw new RangeError('policy frontend does not match the classifier frontend');
  }

  const stageZero = record(policy.stage_zero, 'policy.stage_zero');
  if (
    stageZero.kind !== TIME_DOMAIN_PROFILE_BANK_OPENSET_STAGE_ZERO_KIND_V4
    || stageZero.decision
      !== TIME_DOMAIN_PROFILE_BANK_OPENSET_STAGE_ZERO_DECISION_V4
    || stageZero.effective_prefix_scope
      !== TIME_DOMAIN_PROFILE_BANK_OPENSET_STAGE_ZERO_PREFIX_SCOPE_V4
    || stageZero.runs_before_pose_estimation !== true
    || stageZero.learned_parameters !== 0
  ) {
    throw new RangeError('policy exact-zero stage changed');
  }
  const stageOne = record(policy.stage_one, 'policy.stage_one');
  if (
    stageOne.kind !== TIME_DOMAIN_PROFILE_BANK_OPENSET_STAGE_ONE_KIND_V4
    || stageOne.v3_frozen_hyperplanes_reused !== false
    || stageOne.source_thresholds_reused !== false
    || stageOne.gates_before_classification !== false
    || stageOne.uses_frequency_transform !== false
    || stageOne.uses_absolute_scale !== false
    || stageOne.effective_rank_kind
      !== TIME_DOMAIN_PROFILE_BANK_OPENSET_STAGE_ONE_RANK_KIND_V4
    || stageOne.effective_rank
      !== 'max(predicted_class_rank, route_length_pooled_rank)'
    || stageOne.pooled_rank_always_evaluated !== true
  ) {
    throw new RangeError('policy pose-score stage changed');
  }
  exactArray(
    stageOne.feature_names,
    PREFILTER_FEATURE_NAMES,
    'policy.stage_one.feature_names',
  );
  const stageTwo = record(policy.stage_two, 'policy.stage_two');
  if (
    stageTwo.kind !== TIME_DOMAIN_PROFILE_BANK_OPENSET_STAGE_TWO_KIND_V4
    || stageTwo.self_included_production_distance_used_for_calibration !== false
    || stageTwo.union_head_used !== false
    || stageTwo.uses_frequency_transform !== false
  ) {
    throw new RangeError('policy profile-distance stage changed');
  }
  const stageChirp = record(policy.stage_chirp, 'policy.stage_chirp');
  if (
    stageChirp.kind !== TIME_DOMAIN_PROFILE_BANK_OPENSET_STAGE_CHIRP_KIND_V4
    || stageChirp.active_rms_floor !== PHASE_LINEARITY_ACTIVE_RMS_FLOOR_V4
    || stageChirp.minimum_contiguous_run
      !== PHASE_LINEARITY_MIN_CONTIGUOUS_RUN_V4
    || stageChirp.minimum_valid_adjacencies
      !== PHASE_LINEARITY_MIN_VALID_ADJACENCIES_V4
    || stageChirp.minimum_valid_fraction
      !== PHASE_LINEARITY_MIN_VALID_FRACTION_V4
    || stageChirp.weight_ceiling !== PHASE_LINEARITY_WEIGHT_CEILING_V4
    || stageChirp.per_run_intercept_centering !== true
    || stageChirp.cw_phase_standard_deviation_floor_rad
      !== PHASE_LINEARITY_PHASE_STD_FLOOR_RAD_V4
    || stageChirp.score !== 'clamp(XY^2/(XX*YY),0,1)'
    || stageChirp.uses_frequency_transform !== false
    || stageChirp.uses_absolute_scale !== false
    || stageChirp.invariant_to_nonzero_global_scale !== true
    || stageChirp.invariant_to_global_phase !== true
    || stageChirp.invariant_to_constant_carrier_phase_increment !== true
    || stageChirp.invariant_to_nonzero_linear_time_scaling !== true
    || stageChirp.length_normalized !== true
  ) {
    throw new RangeError('policy instantaneous-phase-linearity stage changed');
  }
  const composition = record(policy.composition, 'policy.composition');
  if (
    composition.kind !== TIME_DOMAIN_PROFILE_BANK_OPENSET_COMPOSITION_KIND_V4
    || composition.stage_one_gate_precedes_classifier !== false
    || composition.conditioned_on_prototype_source_route !== true
  ) {
    throw new RangeError('policy open-set composition changed');
  }

  const binding = record(policy.classifier_binding, 'policy.classifier_binding');
  const boundClassifierSha = exactSha256(
    binding.browser_classifier_asset_sha256,
    'policy.classifier_binding.browser_classifier_asset_sha256',
  );
  if (
    boundClassifierSha !== classifierSha256
    || binding.browser_classifier_asset_bound !== true
    || binding.classifier_schema_required !== TIME_DOMAIN_PROFILE_BANK_SCHEMA_V4
    || binding.classifier_schema_version_required
      !== TIME_DOMAIN_PROFILE_BANK_SCHEMA_VERSION_V4
    || binding.classifier_runtime_role_required
      !== TIME_DOMAIN_PROFILE_BANK_RUNTIME_ROLE_V4
    || !deepJsonEqual(
      binding.classifier_routing,
      classifier.classification.routing,
    )
    || !deepJsonEqual(
      binding.classifier_routing,
      TIME_DOMAIN_TRUSTED_SOURCE_ROUTING_V4,
    )
  ) {
    throw new RangeError('policy is not bound to these classifier bytes/routing');
  }
  const sourceSupport = sourceSupportFromClassifier(classifier);
  if (
    !deepJsonEqual(
      binding.supported_public_class_mask_by_prototype_source,
      sourceSupport,
    )
    || !deepJsonEqual(policy.acquisition_routing, binding.classifier_routing)
  ) {
    throw new RangeError('policy route support does not match the classifier');
  }
  const browserFrontend = record(
    binding.browser_frontend,
    'policy.classifier_binding.browser_frontend',
  );
  if (
    browserFrontend.version !== classifier.frontend.version
    || browserFrontend.patch_length !== classifier.frontend.patch_length
    || browserFrontend.patch_count !== classifier.frontend.patch_count
    || browserFrontend.target_frac !== classifier.frontend.target_frac
    || browserFrontend.uses_frequency_transform !== false
    || !deepJsonEqual(
      browserFrontend.runtime_input_lengths,
      TIME_DOMAIN_RUNTIME_INPUT_LENGTHS_V4,
    )
    || browserFrontend.runtime_bucket_rule !== TIME_DOMAIN_RUNTIME_BUCKET_RULE_V4
  ) {
    throw new RangeError('policy classifier-binding frontend changed');
  }

  const sourceArtifacts = record(
    binding.source_fusion_artifacts_sha256,
    'policy.classifier_binding.source_fusion_artifacts_sha256',
  );
  if (Object.keys(sourceArtifacts).length === 0) {
    throw new RangeError('policy source-fusion artifact ledger is empty');
  }
  for (const [name, digest] of Object.entries(sourceArtifacts)) {
    if (name.length === 0) {
      throw new RangeError('policy source-fusion artifact name is empty');
    }
    exactSha256(digest, `policy source-fusion artifact ${name}`);
  }
  const provenance = record(classifier.provenance, 'classifier.provenance');
  if (
    !deepJsonEqual(
      sourceArtifacts,
      provenance.source_fusion_artifacts_sha256,
    )
    || binding.source_fusion_dev_metrics_sha256
      !== provenance.source_fusion_dev_metrics_sha256
  ) {
    throw new RangeError('policy source-fusion artifact ledger does not match');
  }
  const validationProtocol = record(
    policy.validation_protocol,
    'policy.validation_protocol',
  );
  if (validationProtocol.classifier_asset_sha256 !== classifierSha256) {
    throw new RangeError('policy validation commitment names another classifier');
  }

  return {
    schema: TIME_DOMAIN_PROFILE_BANK_OPENSET_SCHEMA_V4,
    schemaVersion: TIME_DOMAIN_PROFILE_BANK_OPENSET_SCHEMA_VERSION_V4,
    classifierAssetSha256: classifierSha256,
    sourceFusionArtifactsSha256:
      sourceArtifacts as Readonly<Record<string, string>>,
    frontend: {
      version: 'invariant-patch-time-domain-v1',
      patchLength,
      patchCount,
      targetFrac,
      usesFrequencyTransform: false,
    },
    stageOneParameters: loadStageOneParameters(stageOne.parameters_by_length),
    runtimeLengthPolicyByPrototypeSource:
      TIME_DOMAIN_PROFILE_BANK_OPENSET_RUNTIME_LENGTH_POLICY_V4,
    routes: loadRoutePolicies(policy.per_prototype_source),
    acquisitionRouting: classifier.classification.routing,
  };
}

/**
 * Load and mutually bind one raw classifier JSON and one raw open-set policy.
 *
 * Both expected hashes are mandatory so selecting the wrong policy file fails
 * before inference even if that file is internally well formed.
 */
export async function loadBoundTimeDomainOpenSetBundleV4(
  classifierBytesValue: Uint8Array,
  policyBytesValue: Uint8Array,
  options: LoadBoundTimeDomainOpenSetBundleV4Options,
): Promise<BoundTimeDomainOpenSetBundleV4> {
  const expectedClassifierSha256 = exactSha256(
    options?.expectedClassifierSha256,
    'options.expectedClassifierSha256',
  );
  const expectedPolicySha256 = exactSha256(
    options?.expectedPolicySha256,
    'options.expectedPolicySha256',
  );
  const admission = options.admission ?? 'staging';
  const classifierBytes = bytes(classifierBytesValue, 'classifierBytes');
  const policyBytes = bytes(policyBytesValue, 'policyBytes');
  const [classifierSha256, policySha256] = await Promise.all([
    sha256Hex(classifierBytes),
    sha256Hex(policyBytes),
  ]);
  if (classifierSha256 !== expectedClassifierSha256) {
    throw new RangeError('classifier raw-byte SHA-256 mismatch');
  }
  if (policySha256 !== expectedPolicySha256) {
    throw new RangeError('open-set policy raw-byte SHA-256 mismatch');
  }
  const classifier = loadTimeDomainProfileBankAssetV4(
    parseJsonBytes(classifierBytes, 'classifierBytes'),
    { admission: options.admission },
  );
  const policy = loadBoundPolicy(
    parseJsonBytes(policyBytes, 'policyBytes'),
    classifier,
    classifierSha256,
    admission,
  );
  return Object.freeze({
    classifier,
    policy,
    artifactHashes: Object.freeze({
      classifierSha256,
      policySha256,
    }),
  });
}

function rankDetail(
  byClass: readonly number[][],
  pooled: readonly number[],
  classIndex: number,
  value: number,
): TimeDomainOpenSetRankV4 {
  const selected = byClass[classIndex]!;
  return rankAgainstCalibration(
    selected.length > 0 ? selected : pooled,
    value,
    selected.length > 0 ? 'predicted_class' : 'pooled_fallback',
  );
}

function rankAgainstCalibration(
  calibration: readonly number[],
  value: number,
  calibrationScope: TimeDomainOpenSetRankV4['calibrationScope'],
): TimeDomainOpenSetRankV4 {
  if (!Number.isFinite(value)) {
    throw new RangeError('rank query must be finite');
  }
  let low = 0;
  let high = calibration.length;
  while (low < high) {
    const middle = low + Math.floor((high - low) / 2);
    if (calibration[middle]! < value) low = middle + 1;
    else high = middle;
  }
  const rank = empiricalRank(calibration, value);
  if (rank !== low / (calibration.length + 1)) {
    throw new Error('empirical-rank implementation disagreement');
  }
  return {
    value: rank,
    lessCount: low,
    calibrationCount: calibration.length,
    denominator: calibration.length + 1,
    calibrationScope,
  };
}

function stageOneScore(
  parameters: StageOneParametersV4,
  poseFeatures: ArrayLike<number>,
): number {
  if (poseFeatures.length !== PREFILTER_FEATURE_NAMES.length) {
    throw new RangeError(
      `poseFeatures must contain ${PREFILTER_FEATURE_NAMES.length} values`,
    );
  }
  let score = parameters.intercept;
  for (let index = 0; index < poseFeatures.length; index++) {
    const value = poseFeatures[index]!;
    if (!Number.isFinite(value)) {
      throw new RangeError(`poseFeatures[${index}] must be finite`);
    }
    score += parameters.coefficients[index]!
      * ((value - parameters.mean[index]!) / parameters.scale[index]!);
  }
  if (!Number.isFinite(score)) {
    throw new RangeError('stage-one score is nonfinite');
  }
  return score;
}

function positiveModulo(value: number, modulus: number): number {
  return ((value % modulus) + modulus) % modulus;
}

/**
 * FFT-free weighted R² of adjacent-product phase against normalized time.
 *
 * This mirrors NumPy's default unwrap rule exactly. Adjacent products cancel
 * global phase, RMS-relative weights cancel nonzero global scale, per-run
 * centering cancels carrier, and R² is invariant to linear time scaling.
 */
export function instantaneousPhaseLinearityScoreV4(
  inPhase: ArrayLike<number>,
  quadrature: ArrayLike<number>,
): number {
  const length = inPhase.length;
  if (
    length !== quadrature.length
    || length < PHASE_LINEARITY_MIN_VALID_ADJACENCIES_V4 + 1
  ) {
    throw new RangeError('phase-linearity I/Q input is too short or misaligned');
  }
  const amplitude = new Float64Array(length);
  let amplitudeSquaredSum = 0;
  for (let index = 0; index < length; index++) {
    const i = inPhase[index]!;
    const q = quadrature[index]!;
    if (!Number.isFinite(i) || !Number.isFinite(q)) {
      throw new RangeError(`phase-linearity I/Q is nonfinite at sample ${index}`);
    }
    const magnitude = Math.hypot(i, q);
    amplitude[index] = magnitude;
    amplitudeSquaredSum += magnitude * magnitude;
  }
  const rms = Math.sqrt(amplitudeSquaredSum / length);
  if (!Number.isFinite(rms) || rms <= 0) return 0;

  const adjacencyCount = length - 1;
  const phase = new Float64Array(adjacencyCount);
  const valid = new Uint8Array(adjacencyCount);
  const relativeAmplitude = new Float64Array(length);
  for (let index = 0; index < length; index++) {
    relativeAmplitude[index] = amplitude[index]! / rms;
  }
  for (let index = 0; index < adjacencyCount; index++) {
    const i0 = inPhase[index]!;
    const q0 = quadrature[index]!;
    const i1 = inPhase[index + 1]!;
    const q1 = quadrature[index + 1]!;
    const productReal = i1 * i0 + q1 * q0;
    const productImaginary = q1 * i0 - i1 * q0;
    phase[index] = Math.atan2(productImaginary, productReal);
    if (
      relativeAmplitude[index]! >= PHASE_LINEARITY_ACTIVE_RMS_FLOOR_V4
      && relativeAmplitude[index + 1]! >= PHASE_LINEARITY_ACTIVE_RMS_FLOOR_V4
      && Math.hypot(productReal, productImaginary) > 0
    ) {
      valid[index] = 1;
    }
  }

  const runs: Array<readonly [number, number]> = [];
  let runStart = -1;
  let coverage = 0;
  for (let index = 0; index <= adjacencyCount; index++) {
    if (index < adjacencyCount && valid[index] === 1) {
      if (runStart < 0) runStart = index;
      continue;
    }
    if (runStart >= 0) {
      const runLength = index - runStart;
      if (runLength >= PHASE_LINEARITY_MIN_CONTIGUOUS_RUN_V4) {
        runs.push([runStart, index]);
        coverage += runLength;
      }
      runStart = -1;
    }
  }
  const minimumCoverage = Math.max(
    PHASE_LINEARITY_MIN_VALID_ADJACENCIES_V4,
    Math.floor(PHASE_LINEARITY_MIN_VALID_FRACTION_V4 * adjacencyCount),
  );
  if (coverage < minimumCoverage) return 0;

  let weightSum = 0;
  let xx = 0;
  let yy = 0;
  let xy = 0;
  const twoPi = 2 * Math.PI;
  for (const [start, end] of runs) {
    const runLength = end - start;
    const unwrapped = new Float64Array(runLength);
    unwrapped[0] = phase[start]!;
    let cumulativeCorrection = 0;
    for (let offset = 1; offset < runLength; offset++) {
      const raw = phase[start + offset]!;
      const previousRaw = phase[start + offset - 1]!;
      const delta = raw - previousRaw;
      let deltaModulo = positiveModulo(delta + Math.PI, twoPi) - Math.PI;
      if (deltaModulo === -Math.PI && delta > 0) deltaModulo = Math.PI;
      let correction = deltaModulo - delta;
      if (Math.abs(delta) < Math.PI) correction = 0;
      cumulativeCorrection += correction;
      unwrapped[offset] = raw + cumulativeCorrection;
    }

    let runWeight = 0;
    let weightedTime = 0;
    let weightedPhase = 0;
    for (let offset = 0; offset < runLength; offset++) {
      const index = start + offset;
      const relative = Math.min(
        relativeAmplitude[index]!,
        relativeAmplitude[index + 1]!,
      );
      const weight = Math.min(
        relative * relative,
        PHASE_LINEARITY_WEIGHT_CEILING_V4,
      );
      const time = index / adjacencyCount;
      runWeight += weight;
      weightedTime += weight * time;
      weightedPhase += weight * unwrapped[offset]!;
    }
    if (runWeight <= 0) continue;
    const timeCenter = weightedTime / runWeight;
    const phaseCenter = weightedPhase / runWeight;
    weightSum += runWeight;
    for (let offset = 0; offset < runLength; offset++) {
      const index = start + offset;
      const relative = Math.min(
        relativeAmplitude[index]!,
        relativeAmplitude[index + 1]!,
      );
      const weight = Math.min(
        relative * relative,
        PHASE_LINEARITY_WEIGHT_CEILING_V4,
      );
      const centeredTime = index / adjacencyCount - timeCenter;
      const centeredPhase = unwrapped[offset]! - phaseCenter;
      xx += weight * centeredTime * centeredTime;
      yy += weight * centeredPhase * centeredPhase;
      xy += weight * centeredTime * centeredPhase;
    }
  }
  if (
    weightSum <= 0
    || xx <= 0
    || yy <= weightSum * PHASE_LINEARITY_PHASE_STD_FLOOR_RAD_V4 ** 2
  ) {
    return 0;
  }
  const score = (xy * xy) / (xx * yy);
  if (!Number.isFinite(score)) {
    throw new RangeError('phase-linearity score is nonfinite');
  }
  return Math.min(Math.max(score, 0), 1);
}

function closedDecisionFromDistances(
  prototypeSource: TimeDomainPrototypeSourceV4,
  runtimeInputLength: TimeDomainRuntimeInputLengthV4,
  squaredPublicClassDistances: Float64Array,
  closestPrototypeIndices: Int32Array,
  supportedPublicClassMask: readonly boolean[],
): TimeDomainOpenSetClosedDecisionV4 {
  let winner = -1;
  for (let classIndex = 0; classIndex < TIME_DOMAIN_PUBLIC_CLASSES_V4.length; classIndex++) {
    const supported = supportedPublicClassMask[classIndex] === true;
    const distance = squaredPublicClassDistances[classIndex]!;
    const closest = closestPrototypeIndices[classIndex]!;
    if (supported) {
      if (!Number.isFinite(distance) || distance < 0 || closest < 0) {
        throw new RangeError('supported routed distance is invalid');
      }
      if (
        winner < 0
        || distance < squaredPublicClassDistances[winner]!
      ) {
        winner = classIndex;
      }
    } else if (distance !== Number.POSITIVE_INFINITY || closest !== -1) {
      throw new RangeError('unsupported routed class must remain Infinity/-1');
    }
  }
  if (winner < 0 || !EXPECTED_SUPPORT[prototypeSource][winner]) {
    throw new RangeError('selected route has no supported winning class');
  }
  return {
    prototypeSource,
    runtimeInputLength,
    predictedPublicClassIndex: winner,
    predictedPublicClass: TIME_DOMAIN_PUBLIC_CLASSES_V4[winner]!,
    winningSquaredDistance: squaredPublicClassDistances[winner]!,
    winningPrototypeIndex: closestPrototypeIndices[winner]!,
    squaredPublicClassDistances,
    closestPrototypeIndices,
    supportedPublicClassMask: Object.freeze([...supportedPublicClassMask]),
  };
}

function closedDecisionFromClassifier(
  decision: TimeDomainProfileBankDecisionV4,
): TimeDomainOpenSetClosedDecisionV4 {
  const closed = closedDecisionFromDistances(
    decision.prototypeSource,
    decision.runtimeInputLength,
    decision.squaredPublicClassDistances,
    decision.closestPrototypeIndices,
    decision.supportedPublicClassMask,
  );
  if (
    closed.predictedPublicClassIndex !== decision.closedWinnerIndex
    || closed.predictedPublicClass !== decision.closedLabel
    || closed.winningPrototypeIndex !== decision.winningPrototypeIndex
  ) {
    throw new Error('classifier closed decision disagrees with routed argmin');
  }
  return closed;
}

function scoreClosedDecision(
  bundle: BoundTimeDomainOpenSetBundleV4,
  closedDecision: TimeDomainOpenSetClosedDecisionV4,
  poseFeatures: ArrayLike<number>,
  instantaneousPhaseLinearityScore: number,
  observationInputLength: TimeDomainRuntimeInputLengthV4 =
    closedDecision.runtimeInputLength,
  occupiedBandwidthFraction: number | null = null,
): TimeDomainOpenSetAcceptedV4 | TimeDomainOpenSetRejectedV4 {
  const source = admittedSource(closedDecision.prototypeSource);
  const length = closedDecision.runtimeInputLength;
  if (
    effectiveRuntimeInputLength(
      bundle.policy,
      source,
      observationInputLength,
    ) !== length
  ) {
    throw new RangeError(
      'closed decision violates the route-scoped runtime-length policy',
    );
  }
  const classIndex = closedDecision.predictedPublicClassIndex;
  if (
    !EXPECTED_SUPPORT[source][classIndex]
    || !bundle.policy.routes[source].supportedPublicClassMask[classIndex]
  ) {
    throw new RangeError('unsupported predicted class for selected route');
  }
  const lengthKey = String(length) as `${TimeDomainRuntimeInputLengthV4}`;
  const routePolicy = bundle.policy.routes[source].perLength[lengthKey];
  const parameters = bundle.policy.stageOneParameters[lengthKey];
  if (routePolicy === undefined || parameters === undefined) {
    throw new RangeError(`missing open-set bucket ${source}/N${length}`);
  }
  const poseScore = stageOneScore(parameters, poseFeatures);
  const stageOnePredictedClassRank = rankDetail(
    routePolicy.stageOne.byPredictedClass,
    routePolicy.stageOne.pooled,
    classIndex,
    poseScore,
  );
  const stageOnePooledRank = rankAgainstCalibration(
    routePolicy.stageOne.pooled,
    poseScore,
    'route_length_pooled',
  );
  const stageOneRank =
    stageOnePooledRank.value > stageOnePredictedClassRank.value
      ? stageOnePooledRank
      : stageOnePredictedClassRank;
  const stageTwoRank = rankDetail(
    routePolicy.stageTwo.byPredictedClass,
    routePolicy.stageTwo.pooled,
    classIndex,
    closedDecision.winningSquaredDistance,
  );
  if (
    !Number.isFinite(instantaneousPhaseLinearityScore)
    || instantaneousPhaseLinearityScore < 0
    || instantaneousPhaseLinearityScore > 1
  ) {
    throw new RangeError(
      'instantaneousPhaseLinearityScore must lie in [0, 1]',
    );
  }
  const instantaneousPhaseLinearityRank = rankDetail(
    routePolicy.stageChirp.byPredictedClass,
    routePolicy.stageChirp.pooled,
    classIndex,
    instantaneousPhaseLinearityScore,
  );
  const compositeRank = Math.max(
    stageOneRank.value,
    stageTwoRank.value,
    instantaneousPhaseLinearityRank.value,
  );
  const threshold = routePolicy.thresholdByPredictedClass[classIndex]!;
  const rejected = compositeRank >= threshold;
  if (
    occupiedBandwidthFraction !== null
    && (
      !Number.isFinite(occupiedBandwidthFraction)
      || occupiedBandwidthFraction < 0
      || occupiedBandwidthFraction > 1
    )
  ) {
    throw new RangeError('occupiedBandwidthFraction must lie in [0, 1]');
  }
  const common: TimeDomainOpenSetScoredBaseV4 = {
    stageZero: false,
    prototypeSource: source,
    observationInputLength,
    runtimeInputLength: length,
    bucketKey: `${source}/N${length}/${closedDecision.predictedPublicClass}`,
    closedDecision,
    occupiedBandwidthFraction,
    stageOneScore: poseScore,
    stageOnePredictedClassRank,
    stageOnePooledRank,
    stageOneRank,
    stageTwoRank,
    instantaneousPhaseLinearityScore,
    instantaneousPhaseLinearityRank,
    compositeRank,
    threshold,
    artifactHashes: bundle.artifactHashes,
  };
  return rejected
    ? {
        ...common,
        disposition: 'unknown',
        reason: 'at_or_above_route_length_class_threshold',
      }
    : {
        ...common,
        disposition: 'accepted_known',
        reason: 'below_route_length_class_threshold',
      };
}

/**
 * Score a precomputed fused embedding and pose descriptor.
 *
 * This compact parity surface still computes distances from the selected
 * classifier bank itself; callers cannot inject a seven-distance union result.
 */
export function scoreTimeDomainOpenSetEmbeddingV4(
  bundle: BoundTimeDomainOpenSetBundleV4,
  fusedEmbedding: ArrayLike<number>,
  poseFeatures: ArrayLike<number>,
  options: TimeDomainOpenSetEmbeddingOptionsV4,
): TimeDomainOpenSetAcceptedV4 | TimeDomainOpenSetRejectedV4 {
  const source = admittedSource(options?.prototypeSource);
  const length = options?.runtimeInputLength;
  if (!TIME_DOMAIN_RUNTIME_INPUT_LENGTHS_V4.includes(length)) {
    throw new RangeError('runtimeInputLength must be 4096, 8192, or 16384');
  }
  if (
    effectiveRuntimeInputLength(bundle.policy, source, length) !== length
  ) {
    throw new RangeError(
      'compact scoring runtimeInputLength is unreachable for this route',
    );
  }
  const distances = minimumSourcePublicClassSquaredDistancesV4(
    fusedEmbedding,
    bundle.classifier.classification.prototype_bank,
    bundle.classifier.classification.prototype_public_class_indices,
    bundle.classifier.classification.prototype_groups,
    source,
  );
  return scoreClosedDecision(
    bundle,
    closedDecisionFromDistances(
      source,
      length,
      distances.squaredPublicClassDistances,
      distances.closestPrototypeIndices,
      distances.supportedPublicClassMask,
    ),
    poseFeatures,
    options.instantaneousPhaseLinearityScore,
    length,
  );
}

function selectedPrefix(
  inPhase: ArrayLike<number>,
  quadrature: ArrayLike<number>,
  length: number,
): { inPhase: Float64Array; quadrature: Float64Array; exactZero: boolean } {
  const prefixInPhase = new Float64Array(length);
  const prefixQuadrature = new Float64Array(length);
  let exactZero = true;
  for (let index = 0; index < length; index++) {
    const i = inPhase[index]!;
    const q = quadrature[index]!;
    if (!Number.isFinite(i) || !Number.isFinite(q)) {
      throw new RangeError(`selected I/Q prefix is nonfinite at sample ${index}`);
    }
    prefixInPhase[index] = i;
    prefixQuadrature[index] = q;
    if (i !== 0 || q !== 0) exactZero = false;
  }
  return {
    inPhase: prefixInPhase,
    quadrature: prefixQuadrature,
    exactZero,
  };
}

/**
 * Classify raw I/Q with exact-zero stage zero and route-conditioned abstention.
 */
export function classifyTimeDomainOpenSetV4(
  bundle: BoundTimeDomainOpenSetBundleV4,
  inPhase: ArrayLike<number>,
  quadrature: ArrayLike<number>,
  options: TimeDomainOpenSetClassifyOptionsV4,
): TimeDomainOpenSetDecisionV4 {
  if (inPhase.length !== quadrature.length) {
    throw new RangeError('inPhase and quadrature lengths must match');
  }
  const source = admittedSource(options?.prototypeSource);
  const validSampleCount = options?.validSampleCount ?? inPhase.length;
  if (
    !Number.isSafeInteger(validSampleCount)
    || validSampleCount <= 0
    || validSampleCount > inPhase.length
  ) {
    throw new RangeError(
      'validSampleCount must be a positive integer no larger than the arrays',
    );
  }
  const observationInputLength =
    selectTimeDomainRuntimeInputLengthV4(validSampleCount);
  const length = effectiveRuntimeInputLength(
    bundle.policy,
    source,
    observationInputLength,
  );
  const prefix = selectedPrefix(inPhase, quadrature, length);
  if (prefix.exactZero) {
    return {
      disposition: 'unknown',
      reason: 'exact_no_signal',
      stageZero: true,
      prototypeSource: source,
      observationInputLength,
      runtimeInputLength: length,
      bucketKey: `${source}/N${length}/exact_no_signal`,
      closedDecision: null,
      occupiedBandwidthFraction: null,
      artifactHashes: bundle.artifactHashes,
    };
  }
  const poseFeatures = prefilterPoseDegeneracyFeatures(
    prefix.inPhase,
    prefix.quadrature,
    {
      minBandwidth: frontendMinimumBandwidth(
        length,
        bundle.policy.frontend.patchLength,
        bundle.policy.frontend.targetFrac,
      ),
    },
  );
  const instantaneousPhaseLinearityScore =
    instantaneousPhaseLinearityScoreV4(
      prefix.inPhase,
      prefix.quadrature,
    );
  const closed = classifyTimeDomainProfileBankV4(
    bundle.classifier,
    inPhase,
    quadrature,
    { prototypeSource: source, validSampleCount: length },
  );
  return scoreClosedDecision(
    bundle,
    closedDecisionFromClassifier(closed),
    poseFeatures,
    instantaneousPhaseLinearityScore,
    observationInputLength,
    closed.preprocess.context.bw,
  );
}

/** Compile-time/runtime audit marker: this wrapper never uses a frequency transform. */
export const TIME_DOMAIN_PROFILE_BANK_OPENSET_USES_FREQUENCY_TRANSFORM_V4 =
  false as const;
export const TIME_DOMAIN_PROFILE_BANK_OPENSET_DISTANCE_RULE_V4 =
  TIME_DOMAIN_PROFILE_BANK_DECISION_RULE_V4;
