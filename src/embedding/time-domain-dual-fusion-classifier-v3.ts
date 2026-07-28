/**
 * Explicit decoupled v3.4/q97 time-domain classifier:
 *
 *   raw I/Q
 *     -> stage-1 noise gate
 *     -> one invariant-patch preprocess for survivors
 *     -> frozen 4k rejector fusion + staged open-set policy
 *     -> regularized 8k classifier fusion, only for accepted survivors
 *
 * The two fusion assets are intentionally different models with different
 * responsibilities. The rejector's nearest-prototype winner is fitted input
 * to the open-set class geometry; it is never the public known-class answer.
 * The classifier fusion alone supplies a known label, and it is not evaluated
 * for noise or unknown decisions.
 *
 * File hashes supplied to this module are named `preverifiedAssetSha256`
 * deliberately. A parsed JavaScript object has no unique byte serialization,
 * so a package/fetch boundary must hash the exact bytes before parsing. This
 * runtime validates those preverified digests against the role-bound binding.
 */

import {
  preprocessTimeDomainInvariantPatches,
  TIME_DOMAIN_FEATURE_COUNT,
  type TimeDomainInvariantPatchResult,
} from './time-domain-invariant-patch-preprocess-v3.js';
import {
  classifyTimeDomainFusionV3,
  loadTimeDomainFusionAssetV3,
  TIME_DOMAIN_FUSION_CLASSIFIER_ROLE_V3,
  TIME_DOMAIN_FUSION_REJECTOR_ROLE_V3,
  type TimeDomainClassificationResultV3,
  type TimeDomainFusionAssetV3,
  type TimeDomainFusionRuntimeRoleV3,
} from './time-domain-fusion-v3.js';
import {
  loadTimeDomainOpenSetAssetV3,
  stagedArchitectureContract,
  TimeDomainOpenSetV3,
  type StagedArchitectureContract,
  type StagedOpenSetDecision,
  type StageOneEvaluation,
  type TimeDomainOpenSetAssetV3,
} from './time-domain-openset-v3.js';
import {
  admitTimeDomainAssetStatusV3,
  type TimeDomainAssetAdmissionV3,
  type TimeDomainAssetLoadOptionsV3,
  type TimeDomainAssetStatusV3,
} from './time-domain-asset-status-v3.js';

export const TIME_DOMAIN_DUAL_FUSION_BINDING_SCHEMA =
  'atomos.v3.time-domain-dual-fusion.binding' as const;
export const TIME_DOMAIN_DUAL_FUSION_BINDING_SCHEMA_VERSION = 1 as const;
export const TIME_DOMAIN_DUAL_FUSION_CANDIDATE_ID =
  'v3.4-q97-decoupled-8k-classifier-4k-rejector' as const;
export const TIME_DOMAIN_DUAL_FUSION_DESIGN_SEED = 20260955 as const;
export const TIME_DOMAIN_DUAL_FUSION_VALIDATION_SEEDS =
  Object.freeze([20260953, 20260954] as const);
export const TIME_DOMAIN_DUAL_FUSION_RELEASE_SEED = 20260736 as const;

export const TIME_DOMAIN_DUAL_FUSION_REJECTOR_ASSET =
  'time-domain-v3-rejector-weights.json' as const;
export const TIME_DOMAIN_DUAL_FUSION_CLASSIFIER_ASSET =
  'time-domain-v3-classifier-weights.json' as const;
export const TIME_DOMAIN_DUAL_FUSION_OPENSET_ASSET =
  'time-domain-v3-openset-policy.json' as const;
export const TIME_DOMAIN_DUAL_FUSION_BINDING_ASSET =
  'time-domain-v3-dual-binding.json' as const;

export const TIME_DOMAIN_DUAL_FUSION_EXECUTION_ORDER = Object.freeze([
  'stage_one_noise_gate',
  'rejector_known_unknown',
  'classifier_known_label',
] as const);

export const TIME_DOMAIN_DUAL_FUSION_REJECTOR_RESPONSIBILITY =
  'known_unknown_only' as const;
export const TIME_DOMAIN_DUAL_FUSION_CLASSIFIER_RESPONSIBILITY =
  'accepted_known_label_only' as const;

export const TIME_DOMAIN_DUAL_FUSION_STAGED_ARTIFACTS = Object.freeze([
  'v3_branch_lof_components.npz',
  'v3_open_policy_stage_two.npz',
  'v3_staged_composite_policy.npz',
] as const);

export interface TimeDomainDualFusionFrontendBindingV3 {
  version: 'invariant-patch-time-domain-v1';
  patch_length: number;
  patch_count: number;
  target_frac: number;
  packed_length: number;
  uses_frequency_transform: false;
}

export interface TimeDomainDualFusionRoleBindingV3 {
  asset:
    | typeof TIME_DOMAIN_DUAL_FUSION_REJECTOR_ASSET
    | typeof TIME_DOMAIN_DUAL_FUSION_CLASSIFIER_ASSET;
  asset_sha256: string;
  fusion_directory_sha256: string;
  runtime_bundle_manifest_sha256: string;
  runtime_role: TimeDomainFusionRuntimeRoleV3;
  responsibility:
    | typeof TIME_DOMAIN_DUAL_FUSION_REJECTOR_RESPONSIBILITY
    | typeof TIME_DOMAIN_DUAL_FUSION_CLASSIFIER_RESPONSIBILITY;
}

export interface TimeDomainDualFusionOpenSetBindingV3 {
  asset: typeof TIME_DOMAIN_DUAL_FUSION_OPENSET_ASSET;
  asset_sha256: string;
  rejector_asset_sha256: string;
  fitted_rejector_runtime_bundle_manifest_sha256: string;
  staged_validation_report_sha256: string;
  staged_artifacts_sha256: Record<
    (typeof TIME_DOMAIN_DUAL_FUSION_STAGED_ARTIFACTS)[number],
    string
  >;
}

export interface TimeDomainDualFusionValidationBindingV3 {
  report_sha256: string;
  role: 'validate';
  status: 'development_openset_pass';
  design_novelty_seed: typeof TIME_DOMAIN_DUAL_FUSION_DESIGN_SEED;
  novelty_seeds: [20260953, 20260954];
  release_seed_not_spent: typeof TIME_DOMAIN_DUAL_FUSION_RELEASE_SEED;
}

export interface TimeDomainDualFusionFailClosedBindingV3 {
  role_assets_bound_by_sha256: true;
  distinct_role_assets: true;
  role_asset_sha256_must_differ: true;
  classifier_runs_only_after_rejector_acceptance: true;
  public_known_label_from_classifier_only: true;
}

export interface TimeDomainDualFusionBindingV3 {
  schema: typeof TIME_DOMAIN_DUAL_FUSION_BINDING_SCHEMA;
  schema_version: typeof TIME_DOMAIN_DUAL_FUSION_BINDING_SCHEMA_VERSION;
  status: TimeDomainAssetStatusV3;
  candidate_id: typeof TIME_DOMAIN_DUAL_FUSION_CANDIDATE_ID;
  frontend: TimeDomainDualFusionFrontendBindingV3;
  execution_order: [
    'stage_one_noise_gate',
    'rejector_known_unknown',
    'classifier_known_label',
  ];
  roles: {
    rejector: TimeDomainDualFusionRoleBindingV3 & {
      asset: typeof TIME_DOMAIN_DUAL_FUSION_REJECTOR_ASSET;
      runtime_role: typeof TIME_DOMAIN_FUSION_REJECTOR_ROLE_V3;
      responsibility:
        typeof TIME_DOMAIN_DUAL_FUSION_REJECTOR_RESPONSIBILITY;
    };
    classifier: TimeDomainDualFusionRoleBindingV3 & {
      asset: typeof TIME_DOMAIN_DUAL_FUSION_CLASSIFIER_ASSET;
      runtime_role: typeof TIME_DOMAIN_FUSION_CLASSIFIER_ROLE_V3;
      responsibility:
        typeof TIME_DOMAIN_DUAL_FUSION_CLASSIFIER_RESPONSIBILITY;
    };
  };
  openset_policy: TimeDomainDualFusionOpenSetBindingV3;
  validation: TimeDomainDualFusionValidationBindingV3;
  fail_closed: TimeDomainDualFusionFailClosedBindingV3;
}

type UnknownRecord = Record<string, unknown>;

function record(value: unknown, path: string): UnknownRecord {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new TypeError(`${path} must be an object`);
  }
  return value as UnknownRecord;
}

function sha256(value: unknown, path: string): string {
  if (typeof value !== 'string' || !/^[0-9a-f]{64}$/.test(value)) {
    throw new RangeError(`${path} must be a lowercase SHA-256`);
  }
  return value;
}

function positiveInteger(value: unknown, path: string): number {
  if (
    typeof value !== 'number'
    || !Number.isSafeInteger(value)
    || value <= 0
  ) {
    throw new RangeError(`${path} must be a positive safe integer`);
  }
  return value;
}

function exactLiteral(
  value: unknown,
  expected: string | number | boolean,
  path: string,
): void {
  if (value !== expected) {
    throw new RangeError(`${path} must be ${JSON.stringify(expected)}`);
  }
}

function loadRoleBinding(
  value: unknown,
  path: string,
  expected: {
    asset:
      | typeof TIME_DOMAIN_DUAL_FUSION_REJECTOR_ASSET
      | typeof TIME_DOMAIN_DUAL_FUSION_CLASSIFIER_ASSET;
    runtimeRole: TimeDomainFusionRuntimeRoleV3;
    responsibility:
      | typeof TIME_DOMAIN_DUAL_FUSION_REJECTOR_RESPONSIBILITY
      | typeof TIME_DOMAIN_DUAL_FUSION_CLASSIFIER_RESPONSIBILITY;
  },
): TimeDomainDualFusionRoleBindingV3 {
  const role = record(value, path);
  exactLiteral(role.asset, expected.asset, `${path}.asset`);
  exactLiteral(role.runtime_role, expected.runtimeRole, `${path}.runtime_role`);
  exactLiteral(
    role.responsibility,
    expected.responsibility,
    `${path}.responsibility`,
  );
  return {
    asset: expected.asset,
    asset_sha256: sha256(role.asset_sha256, `${path}.asset_sha256`),
    fusion_directory_sha256: sha256(
      role.fusion_directory_sha256,
      `${path}.fusion_directory_sha256`,
    ),
    runtime_bundle_manifest_sha256: sha256(
      role.runtime_bundle_manifest_sha256,
      `${path}.runtime_bundle_manifest_sha256`,
    ),
    runtime_role: expected.runtimeRole,
    responsibility: expected.responsibility,
  };
}

/**
 * Validate an untrusted role binding. Required execution semantics are exact;
 * evidence hashes may vary from candidate to candidate but are always
 * lowercase SHA-256 values.
 */
export function loadTimeDomainDualFusionBindingV3(
  value: unknown,
  options: TimeDomainAssetLoadOptionsV3 = {},
): TimeDomainDualFusionBindingV3 {
  const binding = record(value, 'binding');
  exactLiteral(
    binding.schema,
    TIME_DOMAIN_DUAL_FUSION_BINDING_SCHEMA,
    'binding.schema',
  );
  exactLiteral(
    binding.schema_version,
    TIME_DOMAIN_DUAL_FUSION_BINDING_SCHEMA_VERSION,
    'binding.schema_version',
  );
  const status = admitTimeDomainAssetStatusV3(
    binding.status,
    'binding.status',
    options,
  );
  exactLiteral(
    binding.candidate_id,
    TIME_DOMAIN_DUAL_FUSION_CANDIDATE_ID,
    'binding.candidate_id',
  );
  const candidateId = TIME_DOMAIN_DUAL_FUSION_CANDIDATE_ID;

  const frontendRaw = record(binding.frontend, 'binding.frontend');
  exactLiteral(
    frontendRaw.version,
    'invariant-patch-time-domain-v1',
    'binding.frontend.version',
  );
  exactLiteral(
    frontendRaw.uses_frequency_transform,
    false,
    'binding.frontend.uses_frequency_transform',
  );
  const patchLength = positiveInteger(
    frontendRaw.patch_length,
    'binding.frontend.patch_length',
  );
  const patchCount = positiveInteger(
    frontendRaw.patch_count,
    'binding.frontend.patch_count',
  );
  const packedLength = positiveInteger(
    frontendRaw.packed_length,
    'binding.frontend.packed_length',
  );
  if (packedLength !== patchLength * patchCount) {
    throw new RangeError(
      'binding.frontend.packed_length must equal patch_length * patch_count',
    );
  }
  if (
    typeof frontendRaw.target_frac !== 'number'
    || !Number.isFinite(frontendRaw.target_frac)
    || frontendRaw.target_frac <= 0
    || frontendRaw.target_frac > 1
  ) {
    throw new RangeError('binding.frontend.target_frac must lie in (0, 1]');
  }

  if (
    !Array.isArray(binding.execution_order)
    || binding.execution_order.length
      !== TIME_DOMAIN_DUAL_FUSION_EXECUTION_ORDER.length
    || binding.execution_order.some(
      (entry, index) =>
        entry !== TIME_DOMAIN_DUAL_FUSION_EXECUTION_ORDER[index],
    )
  ) {
    throw new RangeError(
      'binding.execution_order must be stage_one_noise_gate, '
      + 'rejector_known_unknown, classifier_known_label',
    );
  }

  const rolesRaw = record(binding.roles, 'binding.roles');
  const rejector = loadRoleBinding(
    rolesRaw.rejector,
    'binding.roles.rejector',
    {
      asset: TIME_DOMAIN_DUAL_FUSION_REJECTOR_ASSET,
      runtimeRole: TIME_DOMAIN_FUSION_REJECTOR_ROLE_V3,
      responsibility: TIME_DOMAIN_DUAL_FUSION_REJECTOR_RESPONSIBILITY,
    },
  ) as TimeDomainDualFusionBindingV3['roles']['rejector'];
  const classifier = loadRoleBinding(
    rolesRaw.classifier,
    'binding.roles.classifier',
    {
      asset: TIME_DOMAIN_DUAL_FUSION_CLASSIFIER_ASSET,
      runtimeRole: TIME_DOMAIN_FUSION_CLASSIFIER_ROLE_V3,
      responsibility: TIME_DOMAIN_DUAL_FUSION_CLASSIFIER_RESPONSIBILITY,
    },
  ) as TimeDomainDualFusionBindingV3['roles']['classifier'];

  const opensetRaw = record(
    binding.openset_policy,
    'binding.openset_policy',
  );
  exactLiteral(
    opensetRaw.asset,
    TIME_DOMAIN_DUAL_FUSION_OPENSET_ASSET,
    'binding.openset_policy.asset',
  );
  const stagedRaw = record(
    opensetRaw.staged_artifacts_sha256,
    'binding.openset_policy.staged_artifacts_sha256',
  );
  const stagedArtifacts = Object.fromEntries(
    TIME_DOMAIN_DUAL_FUSION_STAGED_ARTIFACTS.map((name) => [
      name,
      sha256(
        stagedRaw[name],
        `binding.openset_policy.staged_artifacts_sha256.${name}`,
      ),
    ]),
  ) as TimeDomainDualFusionOpenSetBindingV3['staged_artifacts_sha256'];
  const openset: TimeDomainDualFusionOpenSetBindingV3 = {
    asset: TIME_DOMAIN_DUAL_FUSION_OPENSET_ASSET,
    asset_sha256: sha256(
      opensetRaw.asset_sha256,
      'binding.openset_policy.asset_sha256',
    ),
    rejector_asset_sha256: sha256(
      opensetRaw.rejector_asset_sha256,
      'binding.openset_policy.rejector_asset_sha256',
    ),
    fitted_rejector_runtime_bundle_manifest_sha256: sha256(
      opensetRaw.fitted_rejector_runtime_bundle_manifest_sha256,
      'binding.openset_policy.'
        + 'fitted_rejector_runtime_bundle_manifest_sha256',
    ),
    staged_validation_report_sha256: sha256(
      opensetRaw.staged_validation_report_sha256,
      'binding.openset_policy.staged_validation_report_sha256',
    ),
    staged_artifacts_sha256: stagedArtifacts,
  };

  const validationRaw = record(binding.validation, 'binding.validation');
  exactLiteral(validationRaw.role, 'validate', 'binding.validation.role');
  exactLiteral(
    validationRaw.status,
    'development_openset_pass',
    'binding.validation.status',
  );
  exactLiteral(
    validationRaw.design_novelty_seed,
    TIME_DOMAIN_DUAL_FUSION_DESIGN_SEED,
    'binding.validation.design_novelty_seed',
  );
  if (
    !Array.isArray(validationRaw.novelty_seeds)
    || validationRaw.novelty_seeds.length
      !== TIME_DOMAIN_DUAL_FUSION_VALIDATION_SEEDS.length
    || validationRaw.novelty_seeds.some(
      (seed, index) =>
        seed !== TIME_DOMAIN_DUAL_FUSION_VALIDATION_SEEDS[index],
    )
  ) {
    throw new RangeError(
      'binding.validation.novelty_seeds must be the frozen one-shot '
      + 'validation pair [20260953, 20260954]',
    );
  }
  exactLiteral(
    validationRaw.release_seed_not_spent,
    TIME_DOMAIN_DUAL_FUSION_RELEASE_SEED,
    'binding.validation.release_seed_not_spent',
  );
  const validation: TimeDomainDualFusionValidationBindingV3 = {
    report_sha256: sha256(
      validationRaw.report_sha256,
      'binding.validation.report_sha256',
    ),
    role: 'validate',
    status: 'development_openset_pass',
    design_novelty_seed: TIME_DOMAIN_DUAL_FUSION_DESIGN_SEED,
    novelty_seeds: [...TIME_DOMAIN_DUAL_FUSION_VALIDATION_SEEDS],
    release_seed_not_spent: TIME_DOMAIN_DUAL_FUSION_RELEASE_SEED,
  };

  const failClosedRaw = record(
    binding.fail_closed,
    'binding.fail_closed',
  );
  for (const name of [
    'role_assets_bound_by_sha256',
    'distinct_role_assets',
    'role_asset_sha256_must_differ',
    'classifier_runs_only_after_rejector_acceptance',
    'public_known_label_from_classifier_only',
  ] as const) {
    exactLiteral(failClosedRaw[name], true, `binding.fail_closed.${name}`);
  }

  if (rejector.asset_sha256 === classifier.asset_sha256) {
    throw new RangeError('rejector and classifier asset SHA-256 must differ');
  }
  if (
    rejector.fusion_directory_sha256
    === classifier.fusion_directory_sha256
  ) {
    throw new RangeError(
      'rejector and classifier fusion-directory SHA-256 must differ',
    );
  }
  if (
    rejector.runtime_bundle_manifest_sha256
    === classifier.runtime_bundle_manifest_sha256
  ) {
    throw new RangeError(
      'rejector and classifier runtime-bundle SHA-256 must differ',
    );
  }
  if (openset.rejector_asset_sha256 !== rejector.asset_sha256) {
    throw new RangeError(
      'open-set policy is not bound to the rejector fusion asset SHA-256',
    );
  }
  if (
    openset.fitted_rejector_runtime_bundle_manifest_sha256
    !== rejector.runtime_bundle_manifest_sha256
  ) {
    throw new RangeError(
      'open-set policy was not fitted against the bound rejector runtime '
      + 'bundle',
    );
  }
  if (
    openset.staged_validation_report_sha256 !== validation.report_sha256
  ) {
    throw new RangeError(
      'open-set staged validation report does not match validation.report_sha256',
    );
  }

  return {
    schema: TIME_DOMAIN_DUAL_FUSION_BINDING_SCHEMA,
    schema_version: TIME_DOMAIN_DUAL_FUSION_BINDING_SCHEMA_VERSION,
    status,
    candidate_id: candidateId,
    frontend: {
      version: 'invariant-patch-time-domain-v1',
      patch_length: patchLength,
      patch_count: patchCount,
      target_frac: frontendRaw.target_frac,
      packed_length: packedLength,
      uses_frequency_transform: false,
    },
    execution_order: [...TIME_DOMAIN_DUAL_FUSION_EXECUTION_ORDER],
    roles: { rejector, classifier },
    openset_policy: openset,
    validation,
    fail_closed: {
      role_assets_bound_by_sha256: true,
      distinct_role_assets: true,
      role_asset_sha256_must_differ: true,
      classifier_runs_only_after_rejector_acceptance: true,
      public_known_label_from_classifier_only: true,
    },
  };
}

export interface PreverifiedTimeDomainAssetV3<T> {
  asset: T;
  /**
   * SHA-256 of the exact source bytes, verified before JSON parsing. This
   * module compares the value with the binding; it does not hash the object.
   */
  preverifiedAssetSha256: string;
}

function assertMatchingDigest(
  supplied: string,
  expected: string,
  path: string,
): void {
  sha256(supplied, `${path}.preverifiedAssetSha256`);
  if (supplied !== expected) {
    throw new RangeError(
      `${path} preverified asset SHA-256 does not match its bound role`,
    );
  }
}

function assertSameStringVector(
  left: readonly string[],
  right: readonly string[],
  path: string,
): void {
  if (
    left.length !== right.length
    || left.some((entry, index) => entry !== right[index])
  ) {
    throw new RangeError(`${path} must match exactly, including order`);
  }
}

function assertLofDimensions(
  openset: TimeDomainOpenSetAssetV3,
  rejector: TimeDomainFusionAssetV3,
): void {
  for (const branch of ['real', 'complex'] as const) {
    const component = openset.stage_two.lof_components.find(
      (candidate) => candidate.branch === branch,
    );
    if (component === undefined) {
      throw new RangeError(`open-set policy has no ${branch} LOF component`);
    }
    const width = rejector[branch].config.embed_dim;
    if (
      component.mean.length !== width
      || component.scale.length !== width
      || component.reference.some((row) => row.length !== width)
    ) {
      throw new RangeError(
        `open-set ${branch} LOF dimensions do not match the rejector fusion`,
      );
    }
  }
}

function assertFusionSourceBinding(
  asset: TimeDomainFusionAssetV3,
  role: TimeDomainDualFusionRoleBindingV3,
  path: string,
): void {
  const provenance = record(asset.provenance, `${path}.provenance`);
  if (
    provenance.source_bundle_manifest_sha256
    !== role.runtime_bundle_manifest_sha256
  ) {
    throw new RangeError(
      `${path} provenance does not match its bound runtime-bundle SHA-256`,
    );
  }
}

function assertOpenSetSourceBinding(
  asset: TimeDomainOpenSetAssetV3,
  binding: TimeDomainDualFusionBindingV3,
): void {
  const provenance = record(asset.provenance, 'opensetPolicy.provenance');
  if (
    provenance.candidate_id !== binding.candidate_id
    || provenance.runtime_bundle_manifest_sha256
      !== binding.roles.rejector.runtime_bundle_manifest_sha256
    || provenance.fusion_directory_sha256
      !== binding.roles.rejector.fusion_directory_sha256
    || provenance.staged_validation_report_sha256
      !== binding.openset_policy.staged_validation_report_sha256
  ) {
    throw new RangeError(
      'open-set provenance is not fitted and validated against the bound '
      + 'rejector role',
    );
  }
  const artifacts = record(
    provenance.staged_artifact_sha256,
    'opensetPolicy.provenance.staged_artifact_sha256',
  );
  for (const name of TIME_DOMAIN_DUAL_FUSION_STAGED_ARTIFACTS) {
    if (
      artifacts[name]
      !== binding.openset_policy.staged_artifacts_sha256[name]
    ) {
      throw new RangeError(
        `open-set provenance for ${name} does not match the binding`,
      );
    }
  }
}

/**
 * Fail-startup compatibility proof for the independently exported binding,
 * rejector fusion, classifier fusion, and staged open-set policy.
 *
 * This deliberately does NOT require the two fusion models' standardization,
 * centers, prototypes, weights, or embedding widths to match: their
 * intentional difference is the reason this composition exists.
 */
export function assertTimeDomainDualFusionAssetSetV3(
  binding: TimeDomainDualFusionBindingV3,
  rejector: PreverifiedTimeDomainAssetV3<TimeDomainFusionAssetV3>,
  classifier: PreverifiedTimeDomainAssetV3<TimeDomainFusionAssetV3>,
  openset: PreverifiedTimeDomainAssetV3<TimeDomainOpenSetAssetV3>,
): void {
  // Public constructors may be called from plain JavaScript, where a
  // TypeScript annotation is no validation. Re-run the binding loader even
  // for a typed value so no caller can bypass its semantic checks.
  loadTimeDomainDualFusionBindingV3(binding, {
    admission: binding.status === 'release' ? 'production' : 'staging',
  });
  assertMatchingDigest(
    rejector.preverifiedAssetSha256,
    binding.roles.rejector.asset_sha256,
    'rejectorFusion',
  );
  assertMatchingDigest(
    classifier.preverifiedAssetSha256,
    binding.roles.classifier.asset_sha256,
    'classifierFusion',
  );
  assertMatchingDigest(
    openset.preverifiedAssetSha256,
    binding.openset_policy.asset_sha256,
    'opensetPolicy',
  );

  if (
    binding.status !== rejector.asset.status
    || binding.status !== classifier.asset.status
    || binding.status !== openset.asset.status
  ) {
    throw new RangeError(
      'binding, rejector, classifier, and open-set statuses must match',
    );
  }
  if (
    rejector.asset.runtime_role !== TIME_DOMAIN_FUSION_REJECTOR_ROLE_V3
  ) {
    throw new RangeError(
      'rejector fusion asset runtime_role must be known_unknown_rejector',
    );
  }
  if (
    classifier.asset.runtime_role !== TIME_DOMAIN_FUSION_CLASSIFIER_ROLE_V3
  ) {
    throw new RangeError(
      'classifier fusion asset runtime_role must be accepted_known_classifier',
    );
  }
  assertFusionSourceBinding(
    rejector.asset,
    binding.roles.rejector,
    'rejectorFusion',
  );
  assertFusionSourceBinding(
    classifier.asset,
    binding.roles.classifier,
    'classifierFusion',
  );
  assertOpenSetSourceBinding(openset.asset, binding);

  const frontend = binding.frontend;
  for (const [name, fusion] of [
    ['rejector', rejector.asset],
    ['classifier', classifier.asset],
  ] as const) {
    if (
      fusion.packed_length !== frontend.packed_length
      || fusion.real.config.patch_length !== frontend.patch_length
      || fusion.complex.config.patch_length !== frontend.patch_length
      || fusion.real.config.patch_count !== frontend.patch_count
      || fusion.complex.config.patch_count !== frontend.patch_count
    ) {
      throw new RangeError(
        `${name} fusion frontend geometry does not match the binding`,
      );
    }
    if (
      fusion.frontend === undefined
      || fusion.frontend.version !== frontend.version
      || fusion.frontend.patch_length !== frontend.patch_length
      || fusion.frontend.patch_count !== frontend.patch_count
      || fusion.frontend.target_frac !== frontend.target_frac
      || fusion.frontend.feature_count !== TIME_DOMAIN_FEATURE_COUNT
      || fusion.frontend.uses_frequency_transform !== false
    ) {
      throw new RangeError(
        `${name} fusion frontend metadata does not match the binding`,
      );
    }
    if (
      fusion.real.config.n_features !== TIME_DOMAIN_FEATURE_COUNT
      || fusion.complex.config.n_features !== TIME_DOMAIN_FEATURE_COUNT
    ) {
      throw new RangeError(
        `${name} fusion feature width must be ${TIME_DOMAIN_FEATURE_COUNT}`,
      );
    }
  }
  if (
    openset.asset.frontend.patch_length !== frontend.patch_length
    || openset.asset.frontend.patch_count !== frontend.patch_count
    || openset.asset.frontend.target_frac !== frontend.target_frac
    || openset.asset.frontend.packed_length !== frontend.packed_length
  ) {
    throw new RangeError(
      'open-set frontend geometry does not match the binding',
    );
  }

  assertSameStringVector(
    rejector.asset.classification.classes,
    classifier.asset.classification.classes,
    'rejector/classifier class labels',
  );
  const openClassCount =
    openset.asset.stage_two.policy.class_geometry_mean.length;
  if (openClassCount !== rejector.asset.classification.classes.length) {
    throw new RangeError(
      'open-set class count does not match the rejector fusion classes',
    );
  }
  assertLofDimensions(openset.asset, rejector.asset);
}

export interface TimeDomainDualFusionOpenSetRuntimeV3 {
  readonly asset: TimeDomainOpenSetAssetV3;
  readonly stageTwo: { readonly classCount: number };
  evaluateStageOne(
    inPhase: ArrayLike<number>,
    quadrature: ArrayLike<number>,
  ): StageOneEvaluation;
  gatedDecision(stageOne: StageOneEvaluation): StagedOpenSetDecision;
  finishSurvivor(
    stageOne: StageOneEvaluation,
    survivor: {
      realEmbedding: ArrayLike<number>;
      complexEmbedding: ArrayLike<number>;
      packedInPhase: ArrayLike<number>;
      packedQuadrature: ArrayLike<number>;
      predictedClassIndex: number;
    },
  ): StagedOpenSetDecision;
}

export interface TimeDomainDualFusionRuntimeDependenciesV3 {
  /** Test seam; production uses the real staged open-set implementation. */
  openSet?: TimeDomainDualFusionOpenSetRuntimeV3;
  /** Test seam; production preprocesses exactly once with the frozen frontend. */
  preprocess?: typeof preprocessTimeDomainInvariantPatches;
  /** Test seams make execution order observable without large CNN fixtures. */
  rejectorClassify?: typeof classifyTimeDomainFusionV3;
  classifierClassify?: typeof classifyTimeDomainFusionV3;
}

export interface TimeDomainDualFusionClassifierV3Options {
  binding: TimeDomainDualFusionBindingV3;
  rejectorFusion: PreverifiedTimeDomainAssetV3<TimeDomainFusionAssetV3>;
  classifierFusion: PreverifiedTimeDomainAssetV3<TimeDomainFusionAssetV3>;
  opensetPolicy: PreverifiedTimeDomainAssetV3<TimeDomainOpenSetAssetV3>;
  dependencies?: TimeDomainDualFusionRuntimeDependenciesV3;
}

export const TIME_DOMAIN_DUAL_FUSION_ARCHITECTURE_CONTRACT = Object.freeze({
  intentional_dual_fusion: true,
  stage_one_first: true,
  preprocess_once_for_survivors: true,
  known_unknown_source: TIME_DOMAIN_FUSION_REJECTOR_ROLE_V3,
  known_label_source: TIME_DOMAIN_FUSION_CLASSIFIER_ROLE_V3,
  classifier_runs_only_after_rejector_acceptance: true,
  uses_frequency_transform: false,
} as const);

export interface TimeDomainDualFusionDecisionBaseV3 {
  candidateId: string;
  label: string;
  knownLabel: string | null;
  knownWinnerIndex: number | null;
  squaredPrototypeDistances: Float64Array | null;
  openSet: StagedOpenSetDecision;
  stagedContract: StagedArchitectureContract;
  architectureContract:
    typeof TIME_DOMAIN_DUAL_FUSION_ARCHITECTURE_CONTRACT;
}

export interface TimeDomainDualFusionNoiseDecisionV3
  extends TimeDomainDualFusionDecisionBaseV3 {
  outcome: 'noise';
  label: 'noise';
  knownLabel: null;
  knownWinnerIndex: null;
  squaredPrototypeDistances: null;
  forward: null;
}

export interface TimeDomainDualFusionSurvivorForwardV3 {
  preprocess: TimeDomainInvariantPatchResult;
  /** Frozen 4k output used only by the staged rejector. */
  rejectorForward: TimeDomainClassificationResultV3;
  /** Null for unknown; populated only after rejector acceptance. */
  classifierForward: TimeDomainClassificationResultV3 | null;
}

export interface TimeDomainDualFusionUnknownDecisionV3
  extends TimeDomainDualFusionDecisionBaseV3 {
  outcome: 'unknown';
  label: 'unknown';
  knownLabel: null;
  knownWinnerIndex: null;
  squaredPrototypeDistances: null;
  forward: TimeDomainDualFusionSurvivorForwardV3 & {
    classifierForward: null;
  };
}

export interface TimeDomainDualFusionKnownDecisionV3
  extends TimeDomainDualFusionDecisionBaseV3 {
  outcome: 'known';
  knownLabel: string;
  knownWinnerIndex: number;
  squaredPrototypeDistances: Float64Array;
  forward: TimeDomainDualFusionSurvivorForwardV3 & {
    classifierForward: TimeDomainClassificationResultV3;
  };
}

export type TimeDomainDualFusionDecisionV3 =
  | TimeDomainDualFusionNoiseDecisionV3
  | TimeDomainDualFusionUnknownDecisionV3
  | TimeDomainDualFusionKnownDecisionV3;

function assertGatedOpenSetDecision(
  decision: StagedOpenSetDecision,
  stageOne: StageOneEvaluation,
): void {
  if (
    decision.stageOne !== stageOne
    || !decision.gated
    || decision.rejectedStage !== 1
    || decision.stageTwo !== null
  ) {
    throw new Error(
      'open-set runtime returned an inconsistent stage-1 gated decision',
    );
  }
}

function assertSurvivorOpenSetDecision(
  decision: StagedOpenSetDecision,
  stageOne: StageOneEvaluation,
): void {
  if (
    decision.stageOne !== stageOne
    || decision.gated
    || decision.stageTwo === null
    || (decision.rejectedStage !== null && decision.rejectedStage !== 2)
  ) {
    throw new Error(
      'open-set runtime returned an inconsistent survivor decision',
    );
  }
}

function assertPreprocessOutput(
  result: TimeDomainInvariantPatchResult,
  frontend: TimeDomainDualFusionFrontendBindingV3,
): void {
  if (
    result.packedInPhase.length !== frontend.packed_length
    || result.packedQuadrature.length !== frontend.packed_length
    || result.rawFeatures.length !== TIME_DOMAIN_FEATURE_COUNT
    || result.context.patch_length !== frontend.patch_length
    || result.context.patch_count !== frontend.patch_count
    || result.context.target_frac !== frontend.target_frac
    || result.context.packed_length !== frontend.packed_length
    || result.context.uses_frequency_transform !== false
  ) {
    throw new Error(
      'invariant-patch preprocess output does not match the bound frontend',
    );
  }
}

function assertFusionClassificationOutput(
  asset: TimeDomainFusionAssetV3,
  result: TimeDomainClassificationResultV3,
  path: string,
): void {
  const classCount = asset.classification.classes.length;
  if (
    result.standardizedFeatures.length !== TIME_DOMAIN_FEATURE_COUNT
    || result.realEmbedding.length !== asset.real.config.embed_dim
    || result.complexEmbedding.length !== asset.complex.config.embed_dim
    || result.fusedEmbedding.length !== 2 * asset.real.config.embed_dim
    || result.squaredPrototypeDistances.length !== classCount
    || !Number.isInteger(result.closedWinnerIndex)
    || result.closedWinnerIndex < 0
    || result.closedWinnerIndex >= classCount
    || result.closedLabel
      !== asset.classification.classes[result.closedWinnerIndex]
  ) {
    throw new Error(`${path} returned an output inconsistent with its asset`);
  }
}

/** Runtime for the frozen 4k-rejector / regularized-8k-classifier contract. */
export class TimeDomainDualFusionClassifierV3 {
  readonly binding: TimeDomainDualFusionBindingV3;
  readonly rejectorFusion: TimeDomainFusionAssetV3;
  readonly classifierFusion: TimeDomainFusionAssetV3;
  readonly openSet: TimeDomainDualFusionOpenSetRuntimeV3;

  private readonly preprocess: typeof preprocessTimeDomainInvariantPatches;
  private readonly rejectorClassify: typeof classifyTimeDomainFusionV3;
  private readonly classifierClassify: typeof classifyTimeDomainFusionV3;

  constructor(options: TimeDomainDualFusionClassifierV3Options) {
    assertTimeDomainDualFusionAssetSetV3(
      options.binding,
      options.rejectorFusion,
      options.classifierFusion,
      options.opensetPolicy,
    );
    this.binding = options.binding;
    this.rejectorFusion = options.rejectorFusion.asset;
    this.classifierFusion = options.classifierFusion.asset;
    this.openSet = options.dependencies?.openSet
      ?? new TimeDomainOpenSetV3(options.opensetPolicy.asset);
    if (this.openSet.asset !== options.opensetPolicy.asset) {
      throw new RangeError(
        'injected open-set runtime must wrap the bound open-set asset',
      );
    }
    if (
      this.openSet.stageTwo.classCount
      !== this.rejectorFusion.classification.classes.length
    ) {
      throw new RangeError(
        'open-set runtime class count does not match the rejector fusion',
      );
    }
    this.preprocess = options.dependencies?.preprocess
      ?? preprocessTimeDomainInvariantPatches;
    this.rejectorClassify = options.dependencies?.rejectorClassify
      ?? classifyTimeDomainFusionV3;
    this.classifierClassify = options.dependencies?.classifierClassify
      ?? classifyTimeDomainFusionV3;
  }

  classify(
    inPhase: ArrayLike<number>,
    quadrature: ArrayLike<number>,
  ): TimeDomainDualFusionDecisionV3 {
    // This call is intentionally the first operation on the capture.
    const stageOne = this.openSet.evaluateStageOne(inPhase, quadrature);
    if (stageOne.gated) {
      const openSet = this.openSet.gatedDecision(stageOne);
      assertGatedOpenSetDecision(openSet, stageOne);
      return {
        outcome: 'noise',
        candidateId: this.binding.candidate_id,
        label: 'noise',
        knownLabel: null,
        knownWinnerIndex: null,
        squaredPrototypeDistances: null,
        openSet,
        forward: null,
        stagedContract: stagedArchitectureContract(),
        architectureContract: TIME_DOMAIN_DUAL_FUSION_ARCHITECTURE_CONTRACT,
      };
    }

    const frontend = this.binding.frontend;
    const preprocess = this.preprocess(inPhase, quadrature, {
      patchLength: frontend.patch_length,
      patchCount: frontend.patch_count,
      targetFrac: frontend.target_frac,
    });
    assertPreprocessOutput(preprocess, frontend);
    const rejectorForward = this.rejectorClassify(
      this.rejectorFusion,
      preprocess.packedInPhase,
      preprocess.packedQuadrature,
      preprocess.rawFeatures,
    );
    assertFusionClassificationOutput(
      this.rejectorFusion,
      rejectorForward,
      'rejector fusion',
    );
    const openSet = this.openSet.finishSurvivor(stageOne, {
      // All stage-2 model inputs come from the frozen rejector, never the
      // accepted-label classifier.
      realEmbedding: rejectorForward.realEmbedding,
      complexEmbedding: rejectorForward.complexEmbedding,
      packedInPhase: preprocess.packedInPhase,
      packedQuadrature: preprocess.packedQuadrature,
      predictedClassIndex: rejectorForward.closedWinnerIndex,
    });
    assertSurvivorOpenSetDecision(openSet, stageOne);
    if (openSet.rejectedStage === 2) {
      return {
        outcome: 'unknown',
        candidateId: this.binding.candidate_id,
        label: 'unknown',
        knownLabel: null,
        knownWinnerIndex: null,
        squaredPrototypeDistances: null,
        openSet,
        forward: {
          preprocess,
          rejectorForward,
          classifierForward: null,
        },
        stagedContract: stagedArchitectureContract(),
        architectureContract: TIME_DOMAIN_DUAL_FUSION_ARCHITECTURE_CONTRACT,
      };
    }

    // The regularized classifier is intentionally the final computation and
    // is unreachable for either rejection path.
    const classifierForward = this.classifierClassify(
      this.classifierFusion,
      preprocess.packedInPhase,
      preprocess.packedQuadrature,
      preprocess.rawFeatures,
    );
    assertFusionClassificationOutput(
      this.classifierFusion,
      classifierForward,
      'classifier fusion',
    );
    return {
      outcome: 'known',
      candidateId: this.binding.candidate_id,
      label: classifierForward.closedLabel,
      knownLabel: classifierForward.closedLabel,
      knownWinnerIndex: classifierForward.closedWinnerIndex,
      squaredPrototypeDistances:
        classifierForward.squaredPrototypeDistances,
      openSet,
      forward: {
        preprocess,
        rejectorForward,
        classifierForward,
      },
      stagedContract: stagedArchitectureContract(),
      architectureContract: TIME_DOMAIN_DUAL_FUSION_ARCHITECTURE_CONTRACT,
    };
  }
}

/** Load all four untrusted JSON values and compose the role-bound runtime. */
export function createTimeDomainDualFusionClassifierV3(inputs: {
  bindingAsset: unknown;
  rejectorFusion: PreverifiedTimeDomainAssetV3<unknown>;
  classifierFusion: PreverifiedTimeDomainAssetV3<unknown>;
  opensetPolicy: PreverifiedTimeDomainAssetV3<unknown>;
  admission?: TimeDomainAssetAdmissionV3;
  dependencies?: TimeDomainDualFusionRuntimeDependenciesV3;
}): TimeDomainDualFusionClassifierV3 {
  const loadOptions = { admission: inputs.admission ?? 'staging' } as const;
  const binding = loadTimeDomainDualFusionBindingV3(
    inputs.bindingAsset,
    loadOptions,
  );
  const rejectorFusion = {
    asset: loadTimeDomainFusionAssetV3(
      inputs.rejectorFusion.asset,
      loadOptions,
    ),
    preverifiedAssetSha256:
      inputs.rejectorFusion.preverifiedAssetSha256,
  };
  const classifierFusion = {
    asset: loadTimeDomainFusionAssetV3(
      inputs.classifierFusion.asset,
      loadOptions,
    ),
    preverifiedAssetSha256:
      inputs.classifierFusion.preverifiedAssetSha256,
  };
  const opensetPolicy = {
    asset: loadTimeDomainOpenSetAssetV3(
      inputs.opensetPolicy.asset,
      loadOptions,
    ),
    preverifiedAssetSha256:
      inputs.opensetPolicy.preverifiedAssetSha256,
  };
  return new TimeDomainDualFusionClassifierV3({
    binding,
    rejectorFusion,
    classifierFusion,
    opensetPolicy,
    dependencies: inputs.dependencies,
  });
}
