/**
 * FFT-free v4 current-source classifier runtime.
 *
 * The v4 head is a bank of enrollment centroids, one per `(source, profile)`.
 * Prototype rows are mapped onto the seven public classes.  Classification
 * therefore returns one distance per public class:
 *
 *   class_distance[c] = min(
 *     squared_euclidean(fused_embedding, prototype[p])
 *     for p where prototype_public_class_indices[p] === c
 *   )
 *
 * This is intentionally a new schema.  A v3 one-prototype-per-class asset
 * cannot be loaded by accident.  The branch encoders and centered-fusion
 * arithmetic reuse the already parity-tested v3 primitives because the v4
 * trainer has not changed those equations; only the prototype head and raw
 * runtime-length policy changed.
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
  fuseTimeDomainEmbeddingsV3,
  standardizeTimeDomainFeaturesV3,
  type CenteredFusionV3,
  type FeatureStandardizationV3,
} from './time-domain-fusion-v3.js';
import {
  admitTimeDomainAssetStatusV3,
  type TimeDomainAssetLoadOptionsV3,
  type TimeDomainAssetStatusV3,
} from './time-domain-asset-status-v3.js';
import {
  preprocessTimeDomainInvariantPatches,
  TIME_DOMAIN_FEATURE_COUNT,
  type TimeDomainInvariantPatchResult,
} from './time-domain-invariant-patch-preprocess-v3.js';
import {
  TIME_DOMAIN_CURRENT_PROFILE_INVENTORY_V4,
  TIME_DOMAIN_CURRENT_PROFILE_PUBLIC_CLASS_V4,
  TIME_DOMAIN_SIGNAL_LAB_PROFILE_ROUTING_RULE_V4,
  TIME_DOMAIN_TRUSTED_SOURCE_ROUTING_KIND_V4,
  type TimeDomainPrototypeSourceV4,
  type TimeDomainPublicClassV4,
  type TimeDomainTrustedSourceRoutingV4,
} from './time-domain-profile-routing-v4.js';

export const TIME_DOMAIN_PROFILE_BANK_SCHEMA_V4 =
  'atomos.v4.time-domain-current-source-profile-bank.browser-weights' as const;
export const TIME_DOMAIN_PROFILE_BANK_SCHEMA_VERSION_V4 = 2 as const;

export const TIME_DOMAIN_PUBLIC_CLASSES_V4 = Object.freeze([
  'am',
  'bluetooth',
  'cw',
  'dsss',
  'fm',
  'gsm',
  'ofdm',
] as const);

export const TIME_DOMAIN_RUNTIME_INPUT_LENGTHS_V4 = Object.freeze([
  4_096,
  8_192,
  16_384,
] as const);

export const TIME_DOMAIN_RUNTIME_BUCKET_RULE_V4 =
  'largest_supported_prefix_not_exceeding_valid_sample_count' as const;
export const TIME_DOMAIN_PROFILE_BANK_DECISION_RULE_V4 =
  'minimum_squared_distance_per_public_class' as const;
export const TIME_DOMAIN_PROFILE_BANK_RUNTIME_ROLE_V4 =
  'accepted_known_classifier' as const;

export type TimeDomainRuntimeInputLengthV4 =
  (typeof TIME_DOMAIN_RUNTIME_INPUT_LENGTHS_V4)[number];

export interface TimeDomainProfileBankFrontendV4 {
  version: 'invariant-patch-time-domain-v1';
  patch_length: number;
  patch_count: number;
  target_frac: number;
  feature_count: typeof TIME_DOMAIN_FEATURE_COUNT;
  uses_frequency_transform: false;
  runtime_input_lengths: [
    4_096,
    8_192,
    16_384,
  ];
  runtime_bucket_rule: typeof TIME_DOMAIN_RUNTIME_BUCKET_RULE_V4;
}

export interface TimeDomainProfileGroupV4 {
  source: TimeDomainPrototypeSourceV4;
  profile: string;
  public_class: TimeDomainPublicClassV4;
  enrollment_views: number;
  eligible_lengths: TimeDomainRuntimeInputLengthV4[];
}

export interface TimeDomainProfileBankHeadV4 {
  classes: TimeDomainPublicClassV4[];
  /** One fused-space centroid per source/profile enrollment group. */
  prototype_bank: number[][];
  /** Public-class index for every row of `prototype_bank`. */
  prototype_public_class_indices: number[];
  /** Auditable metadata in exactly the same row order as the bank. */
  prototype_groups: TimeDomainProfileGroupV4[];
  distance: 'squared_euclidean';
  decision_rule: typeof TIME_DOMAIN_PROFILE_BANK_DECISION_RULE_V4;
  routing: TimeDomainTrustedSourceRoutingV4;
}

export interface TimeDomainProfileBankAssetV4 {
  schema: typeof TIME_DOMAIN_PROFILE_BANK_SCHEMA_V4;
  schema_version: typeof TIME_DOMAIN_PROFILE_BANK_SCHEMA_VERSION_V4;
  status: TimeDomainAssetStatusV3;
  development_only: boolean;
  runtime_role: typeof TIME_DOMAIN_PROFILE_BANK_RUNTIME_ROLE_V4;
  packed_length: number;
  frontend: TimeDomainProfileBankFrontendV4;
  real: TimeDomainRealBranchV3;
  complex: TimeDomainComplexBranchV3;
  fusion: CenteredFusionV3;
  feature_standardization: FeatureStandardizationV3;
  classification: TimeDomainProfileBankHeadV4;
  rejection: {
    state: 'unset';
    runtime_behaviour: 'closed_set_only_no_abstention';
    external_policy_required_for_abstention: true;
  };
  provenance?: unknown;
}

export interface TimeDomainProfileBankForwardV4 {
  standardizedFeatures: Float64Array;
  realEmbedding: Float64Array;
  complexEmbedding: Float64Array;
  fusedEmbedding: Float64Array;
}

export interface TimeDomainProfileBankPackedDecisionV4
  extends TimeDomainProfileBankForwardV4 {
  /** Exactly seven distances, in `TIME_DOMAIN_PUBLIC_CLASSES_V4` order. */
  squaredPublicClassDistances: Float64Array;
  /** Prototype row supplying each of the seven class minima. */
  closestPrototypeIndices: Int32Array;
  /** False for public classes unavailable in the selected source bank. */
  supportedPublicClassMask: readonly boolean[];
  publicClassLabels: readonly TimeDomainPublicClassV4[];
  prototypeSource: TimeDomainPrototypeSourceV4;
  closedWinnerIndex: number;
  closedLabel: TimeDomainPublicClassV4;
  /** Bank row that supplied the winning class minimum. */
  winningPrototypeIndex: number;
}

export interface TimeDomainProfileBankDecisionV4
  extends TimeDomainProfileBankPackedDecisionV4 {
  runtimeInputLength: TimeDomainRuntimeInputLengthV4;
  validSampleCount: number;
  preprocess: TimeDomainInvariantPatchResult;
}

export interface TimeDomainProfileBankClassifyOptionsV4 {
  /** Trusted acquisition provenance; union-bank inference is forbidden. */
  prototypeSource: TimeDomainPrototypeSourceV4;
  /**
   * Number of live samples at the start of the supplied arrays.  This is
   * required when a transport pads (for example) a 12,160-sample Bluetooth
   * capture to 16,384 values.
   */
  validSampleCount?: number;
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

function finiteVector(value: unknown, path: string, length: number): number[] {
  if (!Array.isArray(value) || value.length !== length) {
    throw new RangeError(`${path} must contain ${length} values`);
  }
  for (let index = 0; index < value.length; index++) {
    finite(value[index], `${path}[${index}]`);
  }
  return value as number[];
}

function exactPublicClasses(value: unknown, path: string): TimeDomainPublicClassV4[] {
  if (
    !Array.isArray(value)
    || value.length !== TIME_DOMAIN_PUBLIC_CLASSES_V4.length
    || value.some(
      (entry, index) => entry !== TIME_DOMAIN_PUBLIC_CLASSES_V4[index],
    )
  ) {
    throw new RangeError(
      `${path} must be the canonical seven public classes in runtime order`,
    );
  }
  return [...TIME_DOMAIN_PUBLIC_CLASSES_V4];
}

function exactRuntimeLengths(
  value: unknown,
  path: string,
): [4_096, 8_192, 16_384] {
  if (
    !Array.isArray(value)
    || value.length !== TIME_DOMAIN_RUNTIME_INPUT_LENGTHS_V4.length
    || value.some(
      (entry, index) => entry !== TIME_DOMAIN_RUNTIME_INPUT_LENGTHS_V4[index],
    )
  ) {
    throw new RangeError(`${path} must be exactly [4096,8192,16384]`);
  }
  return [4_096, 8_192, 16_384];
}

function loadFrontend(
  value: unknown,
  real: TimeDomainRealBranchV3,
  complex: TimeDomainComplexBranchV3,
): TimeDomainProfileBankFrontendV4 {
  const frontend = record(value, 'asset.frontend');
  if (frontend.version !== 'invariant-patch-time-domain-v1') {
    throw new RangeError(
      'asset.frontend.version must be invariant-patch-time-domain-v1',
    );
  }
  if (frontend.uses_frequency_transform !== false) {
    throw new RangeError(
      'asset.frontend must declare uses_frequency_transform: false',
    );
  }
  if (frontend.runtime_bucket_rule !== TIME_DOMAIN_RUNTIME_BUCKET_RULE_V4) {
    throw new RangeError(
      `asset.frontend.runtime_bucket_rule must be `
      + TIME_DOMAIN_RUNTIME_BUCKET_RULE_V4,
    );
  }
  const patchLength = positiveInteger(
    frontend.patch_length,
    'asset.frontend.patch_length',
  );
  const patchCount = positiveInteger(
    frontend.patch_count,
    'asset.frontend.patch_count',
  );
  const targetFrac = finite(frontend.target_frac, 'asset.frontend.target_frac');
  const featureCount = positiveInteger(
    frontend.feature_count,
    'asset.frontend.feature_count',
  );
  if (
    patchLength !== real.config.patch_length
    || patchLength !== complex.config.patch_length
    || patchCount !== real.config.patch_count
    || patchCount !== complex.config.patch_count
    || featureCount !== TIME_DOMAIN_FEATURE_COUNT
    || featureCount !== real.config.n_features
    || featureCount !== complex.config.n_features
    || !(targetFrac > 0 && targetFrac <= 1)
  ) {
    throw new RangeError(
      'asset.frontend metadata does not match branch geometry',
    );
  }
  return {
    version: 'invariant-patch-time-domain-v1',
    patch_length: patchLength,
    patch_count: patchCount,
    target_frac: targetFrac,
    feature_count: TIME_DOMAIN_FEATURE_COUNT,
    uses_frequency_transform: false,
    runtime_input_lengths: exactRuntimeLengths(
      frontend.runtime_input_lengths,
      'asset.frontend.runtime_input_lengths',
    ),
    runtime_bucket_rule: TIME_DOMAIN_RUNTIME_BUCKET_RULE_V4,
  };
}

function loadProfileGroups(
  value: unknown,
  labels: number[],
): TimeDomainProfileGroupV4[] {
  if (!Array.isArray(value) || value.length !== labels.length) {
    throw new RangeError(
      'asset.classification.prototype_groups must have one row per prototype',
    );
  }
  const seen = new Set<string>();
  const seenCurrentProfiles = new Set<string>();
  const groups: TimeDomainProfileGroupV4[] = value.map((entry, index) => {
    const path = `asset.classification.prototype_groups[${index}]`;
    const group = record(entry, path);
    if (
      (group.source !== 'current' && group.source !== 'historical')
      || typeof group.profile !== 'string'
      || group.profile.length === 0
    ) {
      throw new RangeError(
        `${path}.source must be current or historical and profile must be non-empty`,
      );
    }
    const key = `${group.source}\u0000${group.profile}`;
    if (seen.has(key)) {
      throw new RangeError(`${path} duplicates a source/profile group`);
    }
    seen.add(key);
    const publicClass = TIME_DOMAIN_PUBLIC_CLASSES_V4[labels[index]!]!;
    if (group.public_class !== publicClass) {
      throw new RangeError(
        `${path}.public_class disagrees with prototype_public_class_indices`,
      );
    }
    if (group.source === 'current') {
      const expected = (
        TIME_DOMAIN_CURRENT_PROFILE_PUBLIC_CLASS_V4 as Readonly<
          Record<string, TimeDomainPublicClassV4 | undefined>
        >
      )[group.profile];
      if (expected === undefined) {
        throw new RangeError(
          `${path}.profile is not in the canonical current profile inventory`,
        );
      }
      if (publicClass !== expected) {
        throw new RangeError(
          `${path} current profile ${group.profile} must map to ${expected}`,
        );
      }
      seenCurrentProfiles.add(group.profile);
    }
    const enrollmentViews = positiveInteger(
      group.enrollment_views,
      `${path}.enrollment_views`,
    );
    if (!Array.isArray(group.eligible_lengths) || group.eligible_lengths.length === 0) {
      throw new RangeError(`${path}.eligible_lengths must be non-empty`);
    }
    const eligibleLengths: TimeDomainRuntimeInputLengthV4[] = [];
    let previous = 0;
    for (let lengthIndex = 0; lengthIndex < group.eligible_lengths.length; lengthIndex++) {
      const length = group.eligible_lengths[lengthIndex];
      if (
        !TIME_DOMAIN_RUNTIME_INPUT_LENGTHS_V4.includes(
          length as TimeDomainRuntimeInputLengthV4,
        )
        || (length as number) <= previous
      ) {
        throw new RangeError(
          `${path}.eligible_lengths must be a sorted unique subset of `
          + '[4096,8192,16384]',
        );
      }
      previous = length as number;
      eligibleLengths.push(length as TimeDomainRuntimeInputLengthV4);
    }
    return {
      source: group.source as TimeDomainPrototypeSourceV4,
      profile: group.profile,
      public_class: publicClass,
      enrollment_views: enrollmentViews,
      eligible_lengths: eligibleLengths,
    };
  });
  const missingCurrentProfiles = TIME_DOMAIN_CURRENT_PROFILE_INVENTORY_V4.filter(
    (profile) => !seenCurrentProfiles.has(profile),
  );
  if (
    seenCurrentProfiles.size !== TIME_DOMAIN_CURRENT_PROFILE_INVENTORY_V4.length
    || missingCurrentProfiles.length !== 0
  ) {
    throw new RangeError(
      'asset.classification current prototype groups must contain the exact '
      + `31-profile inventory; missing ${missingCurrentProfiles.join(', ')}`,
    );
  }
  return groups;
}

function loadTrustedSourceRouting(
  value: unknown,
): TimeDomainTrustedSourceRoutingV4 {
  const routing = record(value, 'asset.classification.routing');
  const sourceMap = record(
    routing.acquisition_source_map,
    'asset.classification.routing.acquisition_source_map',
  );
  if (
    routing.kind !== TIME_DOMAIN_TRUSTED_SOURCE_ROUTING_KIND_V4
    || routing.default_untagged !== 'historical'
    || sourceMap['signal-lab'] !== 'selected-profile-conditioned'
    || sourceMap['physical-sdr'] !== 'historical'
    || sourceMap.untagged !== 'historical'
    || routing.signal_lab_selected_profile_rule
      !== TIME_DOMAIN_SIGNAL_LAB_PROFILE_ROUTING_RULE_V4
    || !Array.isArray(routing.current_profile_inventory)
    || routing.current_profile_inventory.length
      !== TIME_DOMAIN_CURRENT_PROFILE_INVENTORY_V4.length
    || routing.current_profile_inventory.some(
      (profile, index) =>
        profile !== TIME_DOMAIN_CURRENT_PROFILE_INVENTORY_V4[index],
    )
  ) {
    throw new RangeError(
      'asset.classification.routing must match the trusted '
      + 'selected-profile-conditioned routing contract',
    );
  }
  return {
    kind: TIME_DOMAIN_TRUSTED_SOURCE_ROUTING_KIND_V4,
    default_untagged: 'historical',
    acquisition_source_map: {
      'signal-lab': 'selected-profile-conditioned',
      'physical-sdr': 'historical',
      untagged: 'historical',
    },
    signal_lab_selected_profile_rule:
      TIME_DOMAIN_SIGNAL_LAB_PROFILE_ROUTING_RULE_V4,
    current_profile_inventory: [
      ...TIME_DOMAIN_CURRENT_PROFILE_INVENTORY_V4,
    ],
  };
}

/**
 * Validate untrusted v4 browser JSON.  All geometry and profile-bank mappings
 * are checked before a single forward pass can execute.
 */
export function loadTimeDomainProfileBankAssetV4(
  value: unknown,
  options: TimeDomainAssetLoadOptionsV3 = {},
): TimeDomainProfileBankAssetV4 {
  const asset = record(value, 'asset');
  if (asset.schema !== TIME_DOMAIN_PROFILE_BANK_SCHEMA_V4) {
    throw new RangeError(`unsupported v4 classifier schema ${String(asset.schema)}`);
  }
  if (asset.schema_version !== TIME_DOMAIN_PROFILE_BANK_SCHEMA_VERSION_V4) {
    throw new RangeError(
      `unsupported v4 classifier schema version ${String(asset.schema_version)}`,
    );
  }
  const status = admitTimeDomainAssetStatusV3(asset.status, 'asset.status', options);
  if (typeof asset.development_only !== 'boolean') {
    throw new TypeError('asset.development_only must be boolean');
  }
  if (
    (status === 'staging_not_release' && asset.development_only !== true)
    || (status === 'release' && asset.development_only !== false)
  ) {
    throw new RangeError(
      'asset.development_only must match the admitted staging/release status',
    );
  }
  if (asset.runtime_role !== TIME_DOMAIN_PROFILE_BANK_RUNTIME_ROLE_V4) {
    throw new RangeError(
      `asset.runtime_role must be ${TIME_DOMAIN_PROFILE_BANK_RUNTIME_ROLE_V4}`,
    );
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
  const packedLength = positiveInteger(asset.packed_length, 'asset.packed_length');
  if (packedLength !== real.config.patch_length * real.config.patch_count) {
    throw new RangeError('asset.packed_length does not match patch geometry');
  }
  const frontend = loadFrontend(asset.frontend, real, complex);

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
  if (fusion.weight_real < 0 || fusion.weight_real > 1 || fusion.eps <= 0) {
    throw new RangeError('asset.fusion weight/eps are invalid');
  }

  const standardizationRaw = record(
    asset.feature_standardization,
    'asset.feature_standardization',
  );
  const mean = finiteVector(
    standardizationRaw.mean,
    'asset.feature_standardization.mean',
    TIME_DOMAIN_FEATURE_COUNT,
  );
  const std = finiteVector(
    standardizationRaw.std,
    'asset.feature_standardization.std',
    TIME_DOMAIN_FEATURE_COUNT,
  );
  if (std.some((entry) => entry <= 0)) {
    throw new RangeError('asset.feature_standardization.std must be positive');
  }

  const classificationRaw = record(
    asset.classification,
    'asset.classification',
  );
  const classes = exactPublicClasses(
    classificationRaw.classes,
    'asset.classification.classes',
  );
  if (classificationRaw.distance !== 'squared_euclidean') {
    throw new RangeError(
      'asset.classification.distance must be squared_euclidean',
    );
  }
  if (
    classificationRaw.decision_rule
    !== TIME_DOMAIN_PROFILE_BANK_DECISION_RULE_V4
  ) {
    throw new RangeError(
      'asset.classification.decision_rule must be '
      + TIME_DOMAIN_PROFILE_BANK_DECISION_RULE_V4,
    );
  }
  if (
    !Array.isArray(classificationRaw.prototype_bank)
    || classificationRaw.prototype_bank.length < classes.length
  ) {
    throw new RangeError(
      'asset.classification.prototype_bank must contain at least seven rows',
    );
  }
  const prototypeBank = classificationRaw.prototype_bank.map(
    (prototype, index) => finiteVector(
      prototype,
      `asset.classification.prototype_bank[${index}]`,
      2 * embedDim,
    ),
  );
  if (
    !Array.isArray(classificationRaw.prototype_public_class_indices)
    || classificationRaw.prototype_public_class_indices.length
      !== prototypeBank.length
  ) {
    throw new RangeError(
      'asset.classification.prototype_public_class_indices must have one '
      + 'entry per prototype',
    );
  }
  const labels = classificationRaw.prototype_public_class_indices.map(
    (label, index) => {
      if (
        typeof label !== 'number'
        || !Number.isSafeInteger(label)
        || label < 0
        || label >= classes.length
      ) {
        throw new RangeError(
          `asset.classification.prototype_public_class_indices[${index}] `
          + 'is invalid',
        );
      }
      return label;
    },
  );
  for (let classIndex = 0; classIndex < classes.length; classIndex++) {
    if (!labels.includes(classIndex)) {
      throw new RangeError(
        `asset.classification prototype bank has no row for class ${classIndex}`,
      );
    }
  }
  const prototypeGroups = loadProfileGroups(
    classificationRaw.prototype_groups,
    labels,
  );
  const routing = loadTrustedSourceRouting(classificationRaw.routing);
  const rejectionRaw = record(asset.rejection, 'asset.rejection');
  if (
    rejectionRaw.state !== 'unset'
    || rejectionRaw.runtime_behaviour !== 'closed_set_only_no_abstention'
    || rejectionRaw.external_policy_required_for_abstention !== true
  ) {
    throw new RangeError(
      'asset.rejection must declare the external abstention-policy contract',
    );
  }

  return {
    schema: TIME_DOMAIN_PROFILE_BANK_SCHEMA_V4,
    schema_version: TIME_DOMAIN_PROFILE_BANK_SCHEMA_VERSION_V4,
    status,
    development_only: asset.development_only,
    runtime_role: TIME_DOMAIN_PROFILE_BANK_RUNTIME_ROLE_V4,
    packed_length: packedLength,
    frontend,
    real,
    complex,
    fusion,
    feature_standardization: { mean, std },
    classification: {
      classes,
      prototype_bank: prototypeBank,
      prototype_public_class_indices: labels,
      prototype_groups: prototypeGroups,
      distance: 'squared_euclidean',
      decision_rule: TIME_DOMAIN_PROFILE_BANK_DECISION_RULE_V4,
      routing,
    },
    rejection: {
      state: 'unset',
      runtime_behaviour: 'closed_set_only_no_abstention',
      external_policy_required_for_abstention: true,
    },
    ...(asset.provenance === undefined ? {} : { provenance: asset.provenance }),
  };
}

/** Run feature standardization, both branch encoders, and centered fusion. */
export function forwardTimeDomainProfileBankV4(
  asset: TimeDomainProfileBankAssetV4,
  packedInPhase: ArrayLike<number>,
  packedQuadrature: ArrayLike<number>,
  rawFeatures: ArrayLike<number>,
): TimeDomainProfileBankForwardV4 {
  if (
    packedInPhase.length !== asset.packed_length
    || packedQuadrature.length !== asset.packed_length
  ) {
    throw new RangeError(
      `packed I/Q channels must each have length ${asset.packed_length}`,
    );
  }
  const standardizedFeatures = standardizeTimeDomainFeaturesV3(
    asset.feature_standardization,
    rawFeatures,
  );
  const realEmbedding = forwardTimeDomainRealBranchV3(
    asset.real,
    packedInPhase,
    packedQuadrature,
    standardizedFeatures,
  );
  const complexEmbedding = forwardTimeDomainComplexBranchV3(
    asset.complex,
    packedInPhase,
    packedQuadrature,
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

export interface TimeDomainPublicClassDistanceResultV4 {
  squaredPublicClassDistances: Float64Array;
  closestPrototypeIndices: Int32Array;
}

export interface TimeDomainSourcePublicClassDistanceResultV4
  extends TimeDomainPublicClassDistanceResultV4 {
  supportedPublicClassMask: readonly boolean[];
  prototypeSource: TimeDomainPrototypeSourceV4;
}

/**
 * Collapse a source/profile prototype bank to exactly seven public distances.
 * Strict `<` comparisons preserve NumPy `min`/`argmin` first-row tie behavior.
 */
export function minimumPublicClassSquaredDistancesV4(
  embedding: ArrayLike<number>,
  prototypeBank: number[][],
  prototypePublicClassIndices: number[],
): TimeDomainPublicClassDistanceResultV4 {
  if (
    prototypeBank.length !== prototypePublicClassIndices.length
    || prototypeBank.length === 0
  ) {
    throw new RangeError('prototype bank rows and labels must align');
  }
  const distances = new Float64Array(TIME_DOMAIN_PUBLIC_CLASSES_V4.length);
  distances.fill(Number.POSITIVE_INFINITY);
  const closest = new Int32Array(TIME_DOMAIN_PUBLIC_CLASSES_V4.length);
  closest.fill(-1);
  for (let prototypeIndex = 0; prototypeIndex < prototypeBank.length; prototypeIndex++) {
    const prototype = prototypeBank[prototypeIndex]!;
    const classIndex = prototypePublicClassIndices[prototypeIndex]!;
    if (
      !Number.isSafeInteger(classIndex)
      || classIndex < 0
      || classIndex >= distances.length
      || prototype.length !== embedding.length
    ) {
      throw new RangeError(`prototype bank row ${prototypeIndex} is invalid`);
    }
    let distance = 0;
    for (let dimension = 0; dimension < embedding.length; dimension++) {
      const value = embedding[dimension]!;
      const prototypeValue = prototype[dimension]!;
      if (!Number.isFinite(value) || !Number.isFinite(prototypeValue)) {
        throw new RangeError('embedding and prototype bank must be finite');
      }
      const delta = value - prototypeValue;
      distance += delta * delta;
    }
    if (distance < distances[classIndex]!) {
      distances[classIndex] = distance;
      closest[classIndex] = prototypeIndex;
    }
  }
  for (let classIndex = 0; classIndex < distances.length; classIndex++) {
    if (
      !Number.isFinite(distances[classIndex]!)
      || closest[classIndex]! < 0
    ) {
      throw new RangeError(`prototype bank has no row for class ${classIndex}`);
    }
  }
  return {
    squaredPublicClassDistances: distances,
    closestPrototypeIndices: closest,
  };
}

function admittedPrototypeSourceV4(
  value: unknown,
): TimeDomainPrototypeSourceV4 {
  if (value !== 'current' && value !== 'historical') {
    throw new RangeError('prototypeSource must be current or historical');
  }
  return value;
}

/**
 * Compute class minima from exactly one trusted prototype source.
 *
 * Unsupported classes remain `Infinity` with closest index `-1`; they can
 * never win.  There is deliberately no union-bank fallback.
 */
export function minimumSourcePublicClassSquaredDistancesV4(
  embedding: ArrayLike<number>,
  prototypeBank: number[][],
  prototypePublicClassIndices: number[],
  prototypeGroups: TimeDomainProfileGroupV4[],
  prototypeSourceValue: unknown,
): TimeDomainSourcePublicClassDistanceResultV4 {
  const prototypeSource = admittedPrototypeSourceV4(prototypeSourceValue);
  if (
    prototypeBank.length !== prototypePublicClassIndices.length
    || prototypeBank.length !== prototypeGroups.length
    || prototypeBank.length === 0
  ) {
    throw new RangeError(
      'prototype bank rows, labels, and source groups must align',
    );
  }
  const distances = new Float64Array(TIME_DOMAIN_PUBLIC_CLASSES_V4.length);
  distances.fill(Number.POSITIVE_INFINITY);
  const closest = new Int32Array(TIME_DOMAIN_PUBLIC_CLASSES_V4.length);
  closest.fill(-1);
  const supported = Array<boolean>(
    TIME_DOMAIN_PUBLIC_CLASSES_V4.length,
  ).fill(false);
  let admittedRows = 0;
  for (
    let prototypeIndex = 0;
    prototypeIndex < prototypeBank.length;
    prototypeIndex++
  ) {
    if (prototypeGroups[prototypeIndex]!.source !== prototypeSource) continue;
    admittedRows++;
    const prototype = prototypeBank[prototypeIndex]!;
    const classIndex = prototypePublicClassIndices[prototypeIndex]!;
    if (
      !Number.isSafeInteger(classIndex)
      || classIndex < 0
      || classIndex >= distances.length
      || prototype.length !== embedding.length
    ) {
      throw new RangeError(`prototype bank row ${prototypeIndex} is invalid`);
    }
    let distance = 0;
    for (let dimension = 0; dimension < embedding.length; dimension++) {
      const value = embedding[dimension]!;
      const prototypeValue = prototype[dimension]!;
      if (!Number.isFinite(value) || !Number.isFinite(prototypeValue)) {
        throw new RangeError('embedding and prototype bank must be finite');
      }
      const delta = value - prototypeValue;
      distance += delta * delta;
    }
    supported[classIndex] = true;
    if (distance < distances[classIndex]!) {
      distances[classIndex] = distance;
      closest[classIndex] = prototypeIndex;
    }
  }
  if (admittedRows === 0 || !supported.some(Boolean)) {
    throw new RangeError(
      `prototype bank has no rows for requested source ${prototypeSource}`,
    );
  }
  return {
    squaredPublicClassDistances: distances,
    closestPrototypeIndices: closest,
    supportedPublicClassMask: Object.freeze(supported),
    prototypeSource,
  };
}

/** Classify an already-preprocessed packed capture. */
export function classifyPackedTimeDomainProfileBankV4(
  asset: TimeDomainProfileBankAssetV4,
  packedInPhase: ArrayLike<number>,
  packedQuadrature: ArrayLike<number>,
  rawFeatures: ArrayLike<number>,
  options: Pick<TimeDomainProfileBankClassifyOptionsV4, 'prototypeSource'>,
): TimeDomainProfileBankPackedDecisionV4 {
  const forward = forwardTimeDomainProfileBankV4(
    asset,
    packedInPhase,
    packedQuadrature,
    rawFeatures,
  );
  const classDistances = minimumSourcePublicClassSquaredDistancesV4(
    forward.fusedEmbedding,
    asset.classification.prototype_bank,
    asset.classification.prototype_public_class_indices,
    asset.classification.prototype_groups,
    options?.prototypeSource,
  );
  let winner = -1;
  for (
    let classIndex = 0;
    classIndex < classDistances.squaredPublicClassDistances.length;
    classIndex++
  ) {
    if (
      classDistances.supportedPublicClassMask[classIndex]
      && (
        winner < 0
        || classDistances.squaredPublicClassDistances[classIndex]!
          < classDistances.squaredPublicClassDistances[winner]!
      )
    ) {
      winner = classIndex;
    }
  }
  if (winner < 0) {
    throw new RangeError('selected prototype source supports no public class');
  }
  return {
    ...forward,
    squaredPublicClassDistances: classDistances.squaredPublicClassDistances,
    closestPrototypeIndices: classDistances.closestPrototypeIndices,
    supportedPublicClassMask: classDistances.supportedPublicClassMask,
    publicClassLabels: TIME_DOMAIN_PUBLIC_CLASSES_V4,
    prototypeSource: classDistances.prototypeSource,
    closedWinnerIndex: winner,
    closedLabel: TIME_DOMAIN_PUBLIC_CLASSES_V4[winner]!,
    winningPrototypeIndex: classDistances.closestPrototypeIndices[winner]!,
  };
}

/** Select the exact runtime-prefix bucket used by v4 training/evaluation. */
export function selectTimeDomainRuntimeInputLengthV4(
  validSampleCount: number,
): TimeDomainRuntimeInputLengthV4 {
  if (!Number.isSafeInteger(validSampleCount) || validSampleCount <= 0) {
    throw new RangeError('validSampleCount must be a positive safe integer');
  }
  for (
    let index = TIME_DOMAIN_RUNTIME_INPUT_LENGTHS_V4.length - 1;
    index >= 0;
    index--
  ) {
    const candidate = TIME_DOMAIN_RUNTIME_INPUT_LENGTHS_V4[index]!;
    if (candidate <= validSampleCount) return candidate;
  }
  throw new RangeError(
    `v4 classifier requires at least ${TIME_DOMAIN_RUNTIME_INPUT_LENGTHS_V4[0]} `
    + 'valid samples',
  );
}

function prefix(
  values: ArrayLike<number>,
  length: number,
  path: string,
): Float64Array {
  if (values.length < length) {
    throw new RangeError(`${path} has fewer than ${length} values`);
  }
  const output = new Float64Array(length);
  for (let index = 0; index < length; index++) {
    const value = values[index]!;
    if (!Number.isFinite(value)) {
      throw new RangeError(`${path}[${index}] must be finite`);
    }
    output[index] = value;
  }
  return output;
}

/**
 * Classify raw I/Q using the exact prefix-bucket policy trained by v4.
 *
 * Inputs longer than 16,384 samples deliberately use only the first 16,384;
 * inputs between buckets use the largest complete supported prefix.  The
 * frontend remains the FFT-free autocorrelation/lag-quotient path.
 */
export function classifyTimeDomainProfileBankV4(
  asset: TimeDomainProfileBankAssetV4,
  inPhase: ArrayLike<number>,
  quadrature: ArrayLike<number>,
  options: TimeDomainProfileBankClassifyOptionsV4,
): TimeDomainProfileBankDecisionV4 {
  if (inPhase.length !== quadrature.length) {
    throw new RangeError('inPhase and quadrature lengths must match');
  }
  const prototypeSource = admittedPrototypeSourceV4(
    options?.prototypeSource,
  );
  const validSampleCount = options?.validSampleCount ?? inPhase.length;
  if (
    !Number.isSafeInteger(validSampleCount)
    || validSampleCount <= 0
    || validSampleCount > inPhase.length
  ) {
    throw new RangeError(
      'validSampleCount must be a positive integer no larger than the I/Q arrays',
    );
  }
  const runtimeInputLength = selectTimeDomainRuntimeInputLengthV4(
    validSampleCount,
  );
  const inPhasePrefix = prefix(inPhase, runtimeInputLength, 'inPhase');
  const quadraturePrefix = prefix(
    quadrature,
    runtimeInputLength,
    'quadrature',
  );
  const preprocess = preprocessTimeDomainInvariantPatches(
    inPhasePrefix,
    quadraturePrefix,
    {
      patchLength: asset.frontend.patch_length,
      patchCount: asset.frontend.patch_count,
      targetFrac: asset.frontend.target_frac,
    },
  );
  if (
    preprocess.context.uses_frequency_transform !== false
    || preprocess.packedInPhase.length !== asset.packed_length
    || preprocess.packedQuadrature.length !== asset.packed_length
    || preprocess.rawFeatures.length !== TIME_DOMAIN_FEATURE_COUNT
  ) {
    throw new Error('v4 FFT-free preprocess violated the bound asset geometry');
  }
  return {
    ...classifyPackedTimeDomainProfileBankV4(
      asset,
      preprocess.packedInPhase,
      preprocess.packedQuadrature,
      preprocess.rawFeatures,
      { prototypeSource },
    ),
    runtimeInputLength,
    validSampleCount,
    preprocess,
  };
}
