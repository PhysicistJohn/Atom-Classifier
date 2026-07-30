import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  TIME_DOMAIN_PROFILE_BANK_DECISION_RULE_V4,
  TIME_DOMAIN_PROFILE_BANK_RUNTIME_ROLE_V4,
  TIME_DOMAIN_PROFILE_BANK_SCHEMA_V4,
  TIME_DOMAIN_PROFILE_BANK_SCHEMA_VERSION_V4,
  TIME_DOMAIN_PUBLIC_CLASSES_V4,
  TIME_DOMAIN_RUNTIME_BUCKET_RULE_V4,
} from './time-domain-profile-bank-classifier-v4.js';
import {
  classifyTimeDomainOpenSetV4,
  instantaneousPhaseLinearityScoreV4,
  loadBoundTimeDomainOpenSetBundleV4,
  scoreTimeDomainOpenSetEmbeddingV4,
  TIME_DOMAIN_PROFILE_BANK_EMPIRICAL_RANK_RULE_V4,
  TIME_DOMAIN_PROFILE_BANK_OPENSET_COMPOSITION_KIND_V4,
  TIME_DOMAIN_PROFILE_BANK_OPENSET_CURRENT_RUNTIME_LENGTH_V4,
  TIME_DOMAIN_PROFILE_BANK_OPENSET_RUNTIME_LENGTH_POLICY_V4,
  TIME_DOMAIN_PROFILE_BANK_OPENSET_SCHEMA_V4,
  TIME_DOMAIN_PROFILE_BANK_OPENSET_SCHEMA_VERSION_V4,
  TIME_DOMAIN_PROFILE_BANK_OPENSET_STAGE_ONE_KIND_V4,
  TIME_DOMAIN_PROFILE_BANK_OPENSET_STAGE_ONE_RANK_KIND_V4,
  TIME_DOMAIN_PROFILE_BANK_OPENSET_STAGE_CHIRP_KIND_V4,
  TIME_DOMAIN_PROFILE_BANK_OPENSET_STAGE_TWO_KIND_V4,
  TIME_DOMAIN_PROFILE_BANK_OPENSET_STAGE_ZERO_DECISION_V4,
  TIME_DOMAIN_PROFILE_BANK_OPENSET_STAGE_ZERO_KIND_V4,
  TIME_DOMAIN_PROFILE_BANK_OPENSET_STAGE_ZERO_PREFIX_SCOPE_V4,
} from './time-domain-profile-bank-openset-v4.js';
import {
  TIME_DOMAIN_CURRENT_PROFILE_INVENTORY_V4,
  TIME_DOMAIN_CURRENT_PROFILE_PUBLIC_CLASS_V4,
  TIME_DOMAIN_TRUSTED_SOURCE_ROUTING_V4,
} from './time-domain-profile-routing-v4.js';
import { PREFILTER_FEATURE_NAMES } from './time-domain-openset-v3.js';

const legacyRaw = JSON.parse(
  readFileSync(
    new URL(
      './assets-v3-staging/time-domain-fusion-weights-v3.json',
      import.meta.url,
    ),
    'utf8',
  ),
) as Record<string, unknown>;

const ARTIFACT_DIGEST = 'a'.repeat(64);
const METRICS_DIGEST = 'b'.repeat(64);

function classifierRaw(): Record<string, unknown> {
  const legacyClassification = legacyRaw.classification as {
    prototypes: number[][];
  };
  const prototypeBank = legacyClassification.prototypes.map((row) => [...row]);
  const labels = [0, 1, 2, 3, 4, 5, 6];
  const groups: Array<Record<string, unknown>> = labels.map(
    (label, index) => ({
      source: 'historical',
      profile: `historical-${TIME_DOMAIN_PUBLIC_CLASSES_V4[index]}`,
      public_class: TIME_DOMAIN_PUBLIC_CLASSES_V4[label],
      enrollment_views: 3,
      eligible_lengths: [4_096, 8_192, 16_384],
    }),
  );
  for (const profile of TIME_DOMAIN_CURRENT_PROFILE_INVENTORY_V4) {
    const publicClass = TIME_DOMAIN_CURRENT_PROFILE_PUBLIC_CLASS_V4[
      profile as keyof typeof TIME_DOMAIN_CURRENT_PROFILE_PUBLIC_CLASS_V4
    ];
    const label = TIME_DOMAIN_PUBLIC_CLASSES_V4.indexOf(publicClass);
    prototypeBank.push(
      legacyClassification.prototypes[label]!.map((value, index) =>
        value + (index === 0 ? 0.001 : 0)),
    );
    labels.push(label);
    groups.push({
      source: 'current',
      profile,
      public_class: publicClass,
      enrollment_views: 3,
      eligible_lengths: [4_096, 8_192, 16_384],
    });
  }
  const frontend = legacyRaw.frontend as Record<string, unknown>;
  return {
    schema: TIME_DOMAIN_PROFILE_BANK_SCHEMA_V4,
    schema_version: TIME_DOMAIN_PROFILE_BANK_SCHEMA_VERSION_V4,
    status: 'staging_not_release',
    development_only: true,
    runtime_role: TIME_DOMAIN_PROFILE_BANK_RUNTIME_ROLE_V4,
    packed_length: legacyRaw.packed_length,
    frontend: {
      version: frontend.version,
      patch_length: frontend.patch_length,
      patch_count: frontend.patch_count,
      target_frac: frontend.target_frac,
      feature_count: frontend.feature_count,
      uses_frequency_transform: false,
      runtime_input_lengths: [4_096, 8_192, 16_384],
      runtime_bucket_rule: TIME_DOMAIN_RUNTIME_BUCKET_RULE_V4,
    },
    real: legacyRaw.real,
    complex: legacyRaw.complex,
    fusion: legacyRaw.fusion,
    feature_standardization: legacyRaw.feature_standardization,
    classification: {
      classes: [...TIME_DOMAIN_PUBLIC_CLASSES_V4],
      prototype_bank: prototypeBank,
      prototype_public_class_indices: labels,
      prototype_groups: groups,
      distance: 'squared_euclidean',
      decision_rule: TIME_DOMAIN_PROFILE_BANK_DECISION_RULE_V4,
      routing: TIME_DOMAIN_TRUSTED_SOURCE_ROUTING_V4,
    },
    rejection: {
      state: 'unset',
      runtime_behaviour: 'closed_set_only_no_abstention',
      external_policy_required_for_abstention: true,
    },
    provenance: {
      source_fusion_dev_metrics_sha256: METRICS_DIGEST,
      source_fusion_artifacts_sha256: {
        'fusion_prototype_bank.npy': ARTIFACT_DIGEST,
      },
    },
  };
}

function encode(value: unknown): Uint8Array {
  return new TextEncoder().encode(JSON.stringify(value));
}

function sha256(value: Uint8Array): string {
  return createHash('sha256').update(value).digest('hex');
}

function routeLengthPolicy(
  source: 'historical' | 'current',
  threshold: number,
): Record<string, unknown> {
  const support = source === 'historical'
    ? [true, true, true, true, true, true, true]
    : [false, true, false, true, false, true, true];
  const counts = support.map((value) => value ? 1 : 0);
  const stageOneByClass = support.map((value) => value ? [0] : []);
  const stageTwoByClass = support.map((value) => value ? [0] : []);
  return {
    prototype_source: source,
    enrollment_rows: counts.reduce<number>((sum, value) => sum + value, 0),
    stage_one: {
      hard_gate_enabled: false,
      hard_threshold_fitted: false,
      score_rank_calibration_by_predicted_class_sorted: stageOneByClass,
      pooled_score_rank_calibration_sorted: support.filter(Boolean).map(() => 0),
      empty_predicted_class_fallback: 'pooled_score',
      predicted_class_rank_conditioning:
        'prototype_source_route_runtime_length_and_predicted_public_class',
      pooled_rank_conditioning:
        'prototype_source_route_and_runtime_length',
      pooled_rank_always_evaluated: true,
      effective_rank_kind:
        TIME_DOMAIN_PROFILE_BANK_OPENSET_STAGE_ONE_RANK_KIND_V4,
      effective_rank:
        'max(predicted_class_rank, route_length_pooled_rank)',
      empirical_rank_rule: TIME_DOMAIN_PROFILE_BANK_EMPIRICAL_RANK_RULE_V4,
    },
    stage_two: {
      predicted_class_distance_calibration_sorted: stageTwoByClass,
      predicted_class_calibration_counts: counts,
      pooled_distance_calibration_sorted: support.filter(Boolean).map(() => 0),
      empty_predicted_class_fallback: 'pooled_distance',
      empirical_rank_rule: TIME_DOMAIN_PROFILE_BANK_EMPIRICAL_RANK_RULE_V4,
      union_head_used: false,
    },
    stage_chirp: {
      score_rank_calibration_by_predicted_class_sorted: stageOneByClass,
      pooled_score_rank_calibration_sorted: support.filter(Boolean).map(() => 0),
      empty_predicted_class_fallback: 'pooled_score',
      empirical_rank_rule: TIME_DOMAIN_PROFILE_BANK_EMPIRICAL_RANK_RULE_V4,
    },
    composition: {
      threshold_by_predicted_public_class:
        TIME_DOMAIN_PUBLIC_CLASSES_V4.map(() => threshold),
    },
  };
}

function policyRaw(
  classifierSha256: string,
  threshold = 0.5,
): Record<string, unknown> {
  const frontend = legacyRaw.frontend as Record<string, unknown>;
  const parameters = Object.fromEntries(
    [4_096, 8_192, 16_384].map((length) => [
      String(length),
      {
        mean: PREFILTER_FEATURE_NAMES.map(() => 0),
        scale: PREFILTER_FEATURE_NAMES.map(() => 1),
        coefficients: PREFILTER_FEATURE_NAMES.map(() => 0),
        intercept: 0,
      },
    ]),
  );
  const route = (source: 'historical' | 'current') => ({
    supported_public_class_mask: source === 'historical'
      ? [true, true, true, true, true, true, true]
      : [false, true, false, true, false, true, true],
    per_length: Object.fromEntries(
      (
        source === 'current'
          ? [TIME_DOMAIN_PROFILE_BANK_OPENSET_CURRENT_RUNTIME_LENGTH_V4]
          : [4_096, 8_192, 16_384]
      ).map((length) => [
        String(length),
        routeLengthPolicy(source, threshold),
      ]),
    ),
  });
  return {
    schema: TIME_DOMAIN_PROFILE_BANK_OPENSET_SCHEMA_V4,
    schema_version: TIME_DOMAIN_PROFILE_BANK_OPENSET_SCHEMA_VERSION_V4,
    status: 'development_external_policy',
    development_only: true,
    release_evidence: false,
    runtime_role: 'external_abstention_policy',
    classifier_runtime_role_required: TIME_DOMAIN_PROFILE_BANK_RUNTIME_ROLE_V4,
    classifier_binding: {
      source_fusion_dev_metrics_sha256: METRICS_DIGEST,
      source_fusion_artifacts_sha256: {
        'fusion_prototype_bank.npy': ARTIFACT_DIGEST,
      },
      classifier_schema_required: TIME_DOMAIN_PROFILE_BANK_SCHEMA_V4,
      classifier_schema_version_required:
        TIME_DOMAIN_PROFILE_BANK_SCHEMA_VERSION_V4,
      classifier_runtime_role_required:
        TIME_DOMAIN_PROFILE_BANK_RUNTIME_ROLE_V4,
      browser_frontend: {
        version: frontend.version,
        patch_length: frontend.patch_length,
        patch_count: frontend.patch_count,
        target_frac: frontend.target_frac,
        uses_frequency_transform: false,
        runtime_input_lengths: [4_096, 8_192, 16_384],
        runtime_bucket_rule: TIME_DOMAIN_RUNTIME_BUCKET_RULE_V4,
      },
      browser_classifier_asset_sha256: classifierSha256,
      browser_classifier_asset_bound: true,
      classifier_routing: TIME_DOMAIN_TRUSTED_SOURCE_ROUTING_V4,
      supported_public_class_mask_by_prototype_source: {
        historical: [true, true, true, true, true, true, true],
        current: [false, true, false, true, false, true, true],
      },
    },
    public_classes: [...TIME_DOMAIN_PUBLIC_CLASSES_V4],
    prototype_source_routes: ['historical', 'current'],
    acquisition_routing: TIME_DOMAIN_TRUSTED_SOURCE_ROUTING_V4,
    runtime_input_lengths: [4_096, 8_192, 16_384],
    runtime_bucket_rule: TIME_DOMAIN_RUNTIME_BUCKET_RULE_V4,
    runtime_length_policy_by_prototype_source: {
      historical: {
        ...TIME_DOMAIN_PROFILE_BANK_OPENSET_RUNTIME_LENGTH_POLICY_V4
          .historical,
      },
      current: {
        ...TIME_DOMAIN_PROFILE_BANK_OPENSET_RUNTIME_LENGTH_POLICY_V4.current,
      },
    },
    frontend: {
      version: frontend.version,
      patch_length: frontend.patch_length,
      patch_count: frontend.patch_count,
      target_frac: frontend.target_frac,
      uses_frequency_transform: false,
    },
    stage_zero: {
      kind: TIME_DOMAIN_PROFILE_BANK_OPENSET_STAGE_ZERO_KIND_V4,
      decision: TIME_DOMAIN_PROFILE_BANK_OPENSET_STAGE_ZERO_DECISION_V4,
      effective_prefix_scope:
        TIME_DOMAIN_PROFILE_BANK_OPENSET_STAGE_ZERO_PREFIX_SCOPE_V4,
      runs_before_pose_estimation: true,
      learned_parameters: 0,
    },
    stage_one: {
      kind: TIME_DOMAIN_PROFILE_BANK_OPENSET_STAGE_ONE_KIND_V4,
      v3_frozen_hyperplanes_reused: false,
      source_thresholds_reused: false,
      gates_before_classification: false,
      uses_frequency_transform: false,
      uses_absolute_scale: false,
      effective_rank_kind:
        TIME_DOMAIN_PROFILE_BANK_OPENSET_STAGE_ONE_RANK_KIND_V4,
      effective_rank:
        'max(predicted_class_rank, route_length_pooled_rank)',
      pooled_rank_always_evaluated: true,
      feature_names: [...PREFILTER_FEATURE_NAMES],
      parameters_by_length: parameters,
    },
    stage_two: {
      kind: TIME_DOMAIN_PROFILE_BANK_OPENSET_STAGE_TWO_KIND_V4,
      self_included_production_distance_used_for_calibration: false,
      union_head_used: false,
      uses_frequency_transform: false,
    },
    stage_chirp: {
      kind: TIME_DOMAIN_PROFILE_BANK_OPENSET_STAGE_CHIRP_KIND_V4,
      active_rms_floor: 0.02,
      minimum_contiguous_run: 32,
      minimum_valid_adjacencies: 128,
      minimum_valid_fraction: 0.25,
      weight: 'min(r[n],r[n+1])^2 clipped above at 16',
      weight_ceiling: 16,
      per_run_intercept_centering: true,
      normalized_time: 'adjacent_index/(N-1)',
      cw_phase_standard_deviation_floor_rad: 1e-4,
      score: 'clamp(XY^2/(XX*YY),0,1)',
      uses_frequency_transform: false,
      uses_absolute_scale: false,
      invariant_to_nonzero_global_scale: true,
      invariant_to_global_phase: true,
      invariant_to_constant_carrier_phase_increment: true,
      invariant_to_nonzero_linear_time_scaling: true,
      length_normalized: true,
    },
    composition: {
      kind: TIME_DOMAIN_PROFILE_BANK_OPENSET_COMPOSITION_KIND_V4,
      stage_one_gate_precedes_classifier: false,
      conditioned_on_prototype_source_route: true,
    },
    per_prototype_source: {
      historical: route('historical'),
      current: route('current'),
    },
    validation_protocol: {
      classifier_asset_sha256: classifierSha256,
    },
  };
}

async function bundle(threshold = 0.5) {
  const classifierBytes = encode(classifierRaw());
  const classifierSha = sha256(classifierBytes);
  const policyBytes = encode(policyRaw(classifierSha, threshold));
  return loadBoundTimeDomainOpenSetBundleV4(
    classifierBytes,
    policyBytes,
    {
      expectedClassifierSha256: classifierSha,
      expectedPolicySha256: sha256(policyBytes),
    },
  );
}

describe('bound v4 route-conditioned open-set loader', () => {
  it('binds exact raw classifier/policy bytes and every reachable bucket', async () => {
    const loaded = await bundle();
    expect(loaded.artifactHashes.classifierSha256).toMatch(/^[0-9a-f]{64}$/);
    expect(loaded.artifactHashes.policySha256).toMatch(/^[0-9a-f]{64}$/);
    expect(loaded.policy.routes.current.supportedPublicClassMask).toEqual(
      [false, true, false, true, false, true, true],
    );
    expect(loaded.policy.routes.historical.supportedPublicClassMask).toEqual(
      [true, true, true, true, true, true, true],
    );
  });

  it('admits only the matching staging or production status pair', async () => {
    const releasedClassifier = classifierRaw();
    releasedClassifier.status = 'release';
    releasedClassifier.development_only = false;
    const classifierBytes = encode(releasedClassifier);
    const classifierSha = sha256(classifierBytes);
    const releasedPolicy = policyRaw(classifierSha);
    releasedPolicy.status = 'release';
    releasedPolicy.development_only = false;
    releasedPolicy.release_evidence = true;
    const policyBytes = encode(releasedPolicy);
    await expect(loadBoundTimeDomainOpenSetBundleV4(
      classifierBytes,
      policyBytes,
      {
        admission: 'production',
        expectedClassifierSha256: classifierSha,
        expectedPolicySha256: sha256(policyBytes),
      },
    )).resolves.toMatchObject({
      classifier: { status: 'release', development_only: false },
    });
    await expect(loadBoundTimeDomainOpenSetBundleV4(
      classifierBytes,
      policyBytes,
      {
        admission: 'staging',
        expectedClassifierSha256: classifierSha,
        expectedPolicySha256: sha256(policyBytes),
      },
    )).rejects.toThrow(/staging_not_release/);
  });

  it('rejects wrong hashes, changed classifier bytes, and route-ledger tampering', async () => {
    const classifierBytes = encode(classifierRaw());
    const classifierSha = sha256(classifierBytes);
    const rawPolicy = policyRaw(classifierSha);
    const policyBytes = encode(rawPolicy);
    await expect(loadBoundTimeDomainOpenSetBundleV4(
      classifierBytes,
      policyBytes,
      {
        expectedClassifierSha256: classifierSha,
        expectedPolicySha256: '0'.repeat(64),
      },
    )).rejects.toThrow(/policy raw-byte SHA-256 mismatch/);

    const changedClassifier = classifierRaw();
    const head = changedClassifier.classification as {
      prototype_bank: number[][];
    };
    head.prototype_bank[0]![0]! += 0.25;
    const changedBytes = encode(changedClassifier);
    await expect(loadBoundTimeDomainOpenSetBundleV4(
      changedBytes,
      policyBytes,
      {
        expectedClassifierSha256: sha256(changedBytes),
        expectedPolicySha256: sha256(policyBytes),
      },
    )).rejects.toThrow(/not bound to these classifier bytes/);

    const tampered = policyRaw(classifierSha);
    const routes = tampered.per_prototype_source as {
      current: { supported_public_class_mask: boolean[] };
    };
    routes.current.supported_public_class_mask[0] = true;
    const tamperedBytes = encode(tampered);
    await expect(loadBoundTimeDomainOpenSetBundleV4(
      classifierBytes,
      tamperedBytes,
      {
        expectedClassifierSha256: classifierSha,
        expectedPolicySha256: sha256(tamperedBytes),
      },
    )).rejects.toThrow(/supported_public_class_mask/);
  });

  it('rejects a policy missing the pooled stage-one rank contract', async () => {
    const classifierBytes = encode(classifierRaw());
    const classifierSha = sha256(classifierBytes);
    const rawPolicy = policyRaw(classifierSha);
    const routes = rawPolicy.per_prototype_source as {
      historical: {
        per_length: Record<string, {
          stage_one: Record<string, unknown>;
        }>;
      };
    };
    delete routes.historical.per_length['4096']!.stage_one
      .pooled_rank_always_evaluated;
    const policyBytes = encode(rawPolicy);
    await expect(loadBoundTimeDomainOpenSetBundleV4(
      classifierBytes,
      policyBytes,
      {
        expectedClassifierSha256: classifierSha,
        expectedPolicySha256: sha256(policyBytes),
      },
    )).rejects.toThrow(/stage_one rank contract changed/);
  });

  it('rejects runtime-length policy tampering', async () => {
    const classifierBytes = encode(classifierRaw());
    const classifierSha = sha256(classifierBytes);
    const rawPolicy = policyRaw(classifierSha);
    const runtimePolicy =
      rawPolicy.runtime_length_policy_by_prototype_source as {
        current: { effective_runtime_input_length: number };
      };
    runtimePolicy.current.effective_runtime_input_length = 8_192;
    const policyBytes = encode(rawPolicy);
    await expect(loadBoundTimeDomainOpenSetBundleV4(
      classifierBytes,
      policyBytes,
      {
        expectedClassifierSha256: classifierSha,
        expectedPolicySha256: sha256(policyBytes),
      },
    )).rejects.toThrow(/route-scoped runtime-length policy changed/);
  });

  it('rejects ambiguous stage-zero prefix scope', async () => {
    const classifierBytes = encode(classifierRaw());
    const classifierSha = sha256(classifierBytes);
    const rawPolicy = policyRaw(classifierSha);
    const stageZero = rawPolicy.stage_zero as Record<string, unknown>;
    stageZero.decision = 'reject iff max(abs(iq)) == 0';
    const policyBytes = encode(rawPolicy);
    await expect(loadBoundTimeDomainOpenSetBundleV4(
      classifierBytes,
      policyBytes,
      {
        expectedClassifierSha256: classifierSha,
        expectedPolicySha256: sha256(policyBytes),
      },
    )).rejects.toThrow(/exact-zero stage changed/);
  });
});

describe('v4 route/length/class open-set decisions', () => {
  it('uses strict lower-bound ties and accepts below threshold', async () => {
    const loaded = await bundle(0.5);
    const groupIndex = loaded.classifier.classification.prototype_groups.findIndex(
      (group) =>
        group.source === 'current'
        && group.profile === 'wifi-hr-dsss-11m',
    );
    const embedding =
      loaded.classifier.classification.prototype_bank[groupIndex]!;
    const decision = scoreTimeDomainOpenSetEmbeddingV4(
      loaded,
      embedding,
      PREFILTER_FEATURE_NAMES.map(() => 123),
      {
        prototypeSource: 'current',
        runtimeInputLength: 4_096,
        instantaneousPhaseLinearityScore: 0,
      },
    );
    expect(decision.disposition).toBe('accepted_known');
    expect(decision.closedDecision.predictedPublicClass).toBe('dsss');
    expect(decision.closedDecision.supportedPublicClassMask).toEqual(
      [false, true, false, true, false, true, true],
    );
    expect(decision.stageOneRank).toMatchObject({
      value: 0,
      lessCount: 0,
      calibrationCount: 1,
      denominator: 2,
      calibrationScope: 'predicted_class',
    });
    expect(decision.stageTwoRank.value).toBe(0);
    expect(decision.bucketKey).toBe('current/N4096/dsss');
  });

  it('rejects exact threshold equality with no runtime epsilon', async () => {
    const loaded = await bundle(0);
    const embedding =
      loaded.classifier.classification.prototype_bank[0]!;
    const decision = scoreTimeDomainOpenSetEmbeddingV4(
      loaded,
      embedding,
      PREFILTER_FEATURE_NAMES.map(() => 0),
      {
        prototypeSource: 'historical',
        runtimeInputLength: 8_192,
        instantaneousPhaseLinearityScore: 0,
      },
    );
    expect(decision.compositeRank).toBe(0);
    expect(decision.threshold).toBe(0);
    expect(decision.disposition).toBe('unknown');
    expect(decision.reason).toBe(
      'at_or_above_route_length_class_threshold',
    );
  });

  it('allows the enrollment-ranked phase-linearity score to reject', async () => {
    const loaded = await bundle(0.5);
    const embedding =
      loaded.classifier.classification.prototype_bank[0]!;
    const decision = scoreTimeDomainOpenSetEmbeddingV4(
      loaded,
      embedding,
      PREFILTER_FEATURE_NAMES.map(() => 0),
      {
        prototypeSource: 'historical',
        runtimeInputLength: 4_096,
        instantaneousPhaseLinearityScore: 1,
      },
    );
    expect(decision.stageOneRank.value).toBe(0);
    expect(decision.stageTwoRank.value).toBe(0);
    expect(decision.instantaneousPhaseLinearityRank.value).toBe(0.5);
    expect(decision.instantaneousPhaseLinearityScore).toBe(1);
    expect(decision.compositeRank).toBe(0.5);
    expect(decision.disposition).toBe('unknown');
  });

  it('uses the route-length-pooled pose rank when it exceeds the class rank', async () => {
    const classifierBytes = encode(classifierRaw());
    const classifierSha = sha256(classifierBytes);
    const rawPolicy = policyRaw(classifierSha, 0.5);
    const stageOne = rawPolicy.stage_one as {
      parameters_by_length: Record<string, { intercept: number }>;
    };
    stageOne.parameters_by_length['4096']!.intercept = 0.5;
    const routes = rawPolicy.per_prototype_source as {
      historical: {
        per_length: Record<string, {
          stage_one: {
            score_rank_calibration_by_predicted_class_sorted: number[][];
          };
        }>;
      };
    };
    routes.historical.per_length['4096']!.stage_one
      .score_rank_calibration_by_predicted_class_sorted[0] = [1];
    const policyBytes = encode(rawPolicy);
    const loaded = await loadBoundTimeDomainOpenSetBundleV4(
      classifierBytes,
      policyBytes,
      {
        expectedClassifierSha256: classifierSha,
        expectedPolicySha256: sha256(policyBytes),
      },
    );
    const decision = scoreTimeDomainOpenSetEmbeddingV4(
      loaded,
      loaded.classifier.classification.prototype_bank[0]!,
      PREFILTER_FEATURE_NAMES.map(() => 0),
      {
        prototypeSource: 'historical',
        runtimeInputLength: 4_096,
        instantaneousPhaseLinearityScore: 0,
      },
    );
    expect(decision.stageOnePredictedClassRank.value).toBe(0);
    expect(decision.stageOnePooledRank).toMatchObject({
      value: 7 / 8,
      calibrationCount: 7,
      calibrationScope: 'route_length_pooled',
    });
    expect(decision.stageOneRank).toEqual(decision.stageOnePooledRank);
    expect(decision.compositeRank).toBe(7 / 8);
    expect(decision.disposition).toBe('unknown');
  });

  it('rejects nonfinite compact inputs and an absent trusted route', async () => {
    const loaded = await bundle();
    const embedding =
      loaded.classifier.classification.prototype_bank[0]!.slice();
    embedding[0] = Number.NaN;
    expect(() => scoreTimeDomainOpenSetEmbeddingV4(
      loaded,
      embedding,
      PREFILTER_FEATURE_NAMES.map(() => 0),
      {
        prototypeSource: 'historical',
        runtimeInputLength: 4_096,
        instantaneousPhaseLinearityScore: 0,
      },
    )).toThrow(/finite/);
    expect(() => scoreTimeDomainOpenSetEmbeddingV4(
      loaded,
      loaded.classifier.classification.prototype_bank[0]!,
      PREFILTER_FEATURE_NAMES.map(() => 0),
      {
        prototypeSource: 'union' as 'historical',
        runtimeInputLength: 4_096,
        instantaneousPhaseLinearityScore: 0,
      },
    )).toThrow(/current or historical/);
    expect(() => scoreTimeDomainOpenSetEmbeddingV4(
      loaded,
      loaded.classifier.classification.prototype_bank[7]!,
      PREFILTER_FEATURE_NAMES.map(() => 0),
      {
        prototypeSource: 'current',
        runtimeInputLength: 8_192,
        instantaneousPhaseLinearityScore: 0,
      },
    )).toThrow(/unreachable for this route/);
  });

  it('runs exact-zero before pose/classification and honors length endpoints', async () => {
    const loaded = await bundle();
    const zero4096 = new Float64Array(4_096);
    const exact = classifyTimeDomainOpenSetV4(
      loaded,
      zero4096,
      zero4096,
      { prototypeSource: 'current' },
    );
    expect(exact).toMatchObject({
      disposition: 'unknown',
      reason: 'exact_no_signal',
      stageZero: true,
      prototypeSource: 'current',
      runtimeInputLength: 4_096,
      closedDecision: null,
    });
    expect(() => classifyTimeDomainOpenSetV4(
      loaded,
      new Float64Array(4_095),
      new Float64Array(4_095),
      { prototypeSource: 'historical' },
    )).toThrow(/at least 4096/);
    const zero12160 = new Float64Array(12_160);
    expect(classifyTimeDomainOpenSetV4(
      loaded,
      zero12160,
      zero12160,
      { prototypeSource: 'historical' },
    ).runtimeInputLength).toBe(8_192);
    expect(classifyTimeDomainOpenSetV4(
      loaded,
      zero12160,
      zero12160,
      { prototypeSource: 'current' },
    )).toMatchObject({
      observationInputLength: 8_192,
      runtimeInputLength: 4_096,
    });
    const zero16384 = new Float64Array(16_384);
    expect(classifyTimeDomainOpenSetV4(
      loaded,
      zero16384,
      zero16384,
      { prototypeSource: 'historical' },
    ).runtimeInputLength).toBe(16_384);
    zero4096[7] = Number.NaN;
    expect(() => classifyTimeDomainOpenSetV4(
      loaded,
      zero4096,
      new Float64Array(4_096),
      { prototypeSource: 'historical' },
    )).toThrow(/nonfinite/);
  });

  it('makes current raw decisions causal and length invariant', async () => {
    const loaded = await bundle();
    const inPhase = new Float64Array(16_384);
    const quadrature = new Float64Array(16_384);
    const changedTailI = new Float64Array(16_384);
    const changedTailQ = new Float64Array(16_384);
    for (let index = 0; index < inPhase.length; index++) {
      const phase = 2 * Math.PI * 0.071 * index;
      inPhase[index] = Math.cos(phase);
      quadrature[index] = Math.sin(phase);
      const tailPhase = 2 * Math.PI * 0.193 * index + 0.4;
      changedTailI[index] = index < 4_096
        ? inPhase[index]!
        : 7 * Math.cos(tailPhase);
      changedTailQ[index] = index < 4_096
        ? quadrature[index]!
        : 7 * Math.sin(tailPhase);
    }
    const decisions = [4_096, 8_192, 16_384].map((validSampleCount) =>
      classifyTimeDomainOpenSetV4(
        loaded,
        inPhase,
        quadrature,
        { prototypeSource: 'current', validSampleCount },
      ));
    expect(decisions.map((row) => row.observationInputLength)).toEqual(
      [4_096, 8_192, 16_384],
    );
    for (const decision of decisions) {
      expect(decision.runtimeInputLength).toBe(4_096);
      expect(decision.stageZero).toBe(false);
    }
    const scored = decisions.filter((row) => !row.stageZero);
    if (scored.length !== 3) throw new Error('tone cannot be exact zero');
    const projection = (row: typeof scored[number]) => ({
      disposition: row.disposition,
      prediction: row.closedDecision.predictedPublicClass,
      composite: row.compositeRank,
      threshold: row.threshold,
    });
    expect(scored.map(projection)).toEqual([
      projection(scored[0]!),
      projection(scored[0]!),
      projection(scored[0]!),
    ]);

    const changedTail = classifyTimeDomainOpenSetV4(
      loaded,
      changedTailI,
      changedTailQ,
      { prototypeSource: 'current', validSampleCount: 16_384 },
    );
    expect(changedTail.stageZero).toBe(false);
    if (changedTail.stageZero) throw new Error('first 4K is nonzero');
    expect(projection(changedTail)).toEqual(projection(scored[0]!));

    const zeroPrefixI = new Float64Array(16_384);
    const zeroPrefixQ = new Float64Array(16_384);
    for (let index = 4_096; index < zeroPrefixI.length; index++) {
      zeroPrefixI[index] = 1;
    }
    expect(classifyTimeDomainOpenSetV4(
      loaded,
      zeroPrefixI,
      zeroPrefixQ,
      { prototypeSource: 'current' },
    )).toMatchObject({
      stageZero: true,
      reason: 'exact_no_signal',
      observationInputLength: 16_384,
      runtimeInputLength: 4_096,
    });
    expect(classifyTimeDomainOpenSetV4(
      loaded,
      zeroPrefixI,
      zeroPrefixQ,
      { prototypeSource: 'historical' },
    ).stageZero).toBe(false);

    expect([4_096, 8_192, 16_384].map((validSampleCount) =>
      classifyTimeDomainOpenSetV4(
        loaded,
        inPhase,
        quadrature,
        { prototypeSource: 'historical', validSampleCount },
      ).runtimeInputLength)).toEqual([4_096, 8_192, 16_384]);
  });

  it('retains the raw FFT-free occupied-bandwidth estimate', async () => {
    const loaded = await bundle();
    const inPhase = new Float64Array(4_096);
    const quadrature = new Float64Array(4_096);
    for (let index = 0; index < inPhase.length; index++) {
      const phase = 2 * Math.PI * 0.071 * index;
      inPhase[index] = Math.cos(phase);
      quadrature[index] = Math.sin(phase);
    }
    const decision = classifyTimeDomainOpenSetV4(
      loaded,
      inPhase,
      quadrature,
      { prototypeSource: 'historical' },
    );
    expect(decision.stageZero).toBe(false);
    if (decision.stageZero) throw new Error('sinusoid cannot be exact zero');
    expect(decision.occupiedBandwidthFraction).not.toBeNull();
    expect(decision.occupiedBandwidthFraction).toBeGreaterThanOrEqual(0);
    expect(decision.occupiedBandwidthFraction).toBeLessThanOrEqual(1);
  });
});

function linearFm(length = 16_384): {
  inPhase: Float64Array;
  quadrature: Float64Array;
} {
  const inPhase = new Float64Array(length);
  const quadrature = new Float64Array(length);
  for (let index = 0; index < length; index++) {
    const normalized = index / (length - 1);
    const envelope =
      0.55 + 0.35 * Math.cos(2 * Math.PI * 3 * normalized + 0.4);
    const phase =
      2 * Math.PI * (0.17 * index + 0.31 * index * normalized);
    inPhase[index] = envelope * Math.cos(phase);
    quadrature[index] = envelope * Math.sin(phase);
  }
  return { inPhase, quadrature };
}

describe('v4 instantaneous-phase-linearity invariant', () => {
  it('is prefix, scale, global-phase, carrier, and time-scale invariant', () => {
    const raw = linearFm();
    const scores: number[] = [];
    for (const length of [4_096, 8_192, 16_384]) {
      const inPhase = raw.inPhase.slice(0, length);
      const quadrature = raw.quadrature.slice(0, length);
      const transformedI = new Float64Array(length);
      const transformedQ = new Float64Array(length);
      for (let index = 0; index < length; index++) {
        const rotation = 0.71 + 0.013 * index;
        const cosine = Math.cos(rotation);
        const sine = Math.sin(rotation);
        transformedI[index] =
          1e4 * (inPhase[index]! * cosine - quadrature[index]! * sine);
        transformedQ[index] =
          1e4 * (inPhase[index]! * sine + quadrature[index]! * cosine);
      }
      const first = instantaneousPhaseLinearityScoreV4(
        inPhase,
        quadrature,
      );
      const second = instantaneousPhaseLinearityScoreV4(
        transformedI,
        transformedQ,
      );
      expect(first).toBeGreaterThan(0.999999999);
      expect(Math.abs(first - second)).toBeLessThan(1e-10);
      scores.push(first);

      const scaledTimeI = new Float64Array(length);
      const scaledTimeQ = new Float64Array(length);
      for (let index = 0; index < length; index++) {
        const normalized = index / (length - 1);
        const phase =
          2 * Math.PI * (-0.11 * index + 0.23 * index * normalized);
        scaledTimeI[index] = Math.cos(phase);
        scaledTimeQ[index] = Math.sin(phase);
      }
      expect(
        instantaneousPhaseLinearityScoreV4(scaledTimeI, scaledTimeQ),
      ).toBeGreaterThan(0.999999999);
    }
    expect(Math.max(...scores) - Math.min(...scores)).toBeLessThan(1e-10);
  });

  it('returns zero for CW and remains robust to envelope gaps/noise/jitter', () => {
    const length = 16_384;
    const cwI = new Float64Array(length);
    const cwQ = new Float64Array(length);
    for (let index = 0; index < length; index++) {
      const phase = 0.2 * index + 0.3;
      cwI[index] = Math.cos(phase);
      cwQ[index] = Math.sin(phase);
    }
    expect(instantaneousPhaseLinearityScoreV4(cwI, cwQ)).toBe(0);

    const raw = linearFm(length);
    let state = 0x6d2b79f5;
    const random = (): number => {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      return state / 0x1_0000_0000 - 0.5;
    };
    for (let index = 0; index < length; index++) {
      if (index >= 5_000 && index < 5_160) {
        raw.inPhase[index] = 0;
        raw.quadrature[index] = 0;
        continue;
      }
      const jitter = 0.015 * Math.sin(2 * Math.PI * index / 701);
      const cosine = Math.cos(jitter);
      const sine = Math.sin(jitter);
      const i = raw.inPhase[index]!;
      const q = raw.quadrature[index]!;
      raw.inPhase[index] = i * cosine - q * sine + 0.01 * random();
      raw.quadrature[index] = i * sine + q * cosine + 0.01 * random();
    }
    expect(
      instantaneousPhaseLinearityScoreV4(raw.inPhase, raw.quadrature),
    ).toBeGreaterThan(0.95);
  });
});

interface RawPhaseParityFixture {
  schema: string;
  schema_version: number;
  uses_frequency_transform: false;
  tolerance: {
    absolute: number;
    relative: number;
  };
  phase_linearity_raw_cases: Array<{
    id: string;
    runtime_input_length: 4_096 | 8_192 | 16_384;
    sample_encoding: 'float32-little-endian-base64';
    in_phase_f32le_base64: string;
    quadrature_f32le_base64: string;
    in_phase_sha256: string;
    quadrature_sha256: string;
    expected_score: number;
    property_contract: {
      score_minimum?: number;
      score_maximum?: number;
      reference_case?: string;
      maximum_reference_delta?: number;
    };
  }>;
  cases: unknown[];
}

function decodeFixtureF32Le(
  encoded: string,
  expectedLength: number,
  expectedSha256: string,
): Float32Array {
  const raw = Buffer.from(encoded, 'base64');
  expect(raw.byteLength).toBe(
    expectedLength * Float32Array.BYTES_PER_ELEMENT,
  );
  expect(createHash('sha256').update(raw).digest('hex')).toBe(
    expectedSha256,
  );
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const result = new Float32Array(expectedLength);
  for (let index = 0; index < expectedLength; index++) {
    result[index] = view.getFloat32(
      index * Float32Array.BYTES_PER_ELEMENT,
      true,
    );
  }
  return result;
}

describe('v4 durable Python raw-I/Q parity fixture', () => {
  it('covers every runtime length, invariance transform, and score edge', () => {
    const fixture = JSON.parse(
      readFileSync(
        new URL(
          './test-fixtures/time-domain-profile-bank-v4-openset-parity.json',
          import.meta.url,
        ),
        'utf8',
      ),
    ) as RawPhaseParityFixture;
    expect(fixture.schema).toBe(
      'atomos.v4.time-domain-profile-bank.openset-browser-parity',
    );
    expect(fixture.schema_version).toBe(4);
    expect(fixture.uses_frequency_transform).toBe(false);
    expect(fixture.cases.length).toBeGreaterThan(0);
    expect(fixture.phase_linearity_raw_cases).toHaveLength(24);
    expect(
      new Set(
        fixture.phase_linearity_raw_cases.map(
          (entry) => entry.runtime_input_length,
        ),
      ),
    ).toEqual(new Set([4_096, 8_192, 16_384]));

    const observedById = new Map<string, number>();
    for (const entry of fixture.phase_linearity_raw_cases) {
      expect(entry.sample_encoding).toBe(
        'float32-little-endian-base64',
      );
      const inPhase = decodeFixtureF32Le(
        entry.in_phase_f32le_base64,
        entry.runtime_input_length,
        entry.in_phase_sha256,
      );
      const quadrature = decodeFixtureF32Le(
        entry.quadrature_f32le_base64,
        entry.runtime_input_length,
        entry.quadrature_sha256,
      );
      const observed = instantaneousPhaseLinearityScoreV4(
        inPhase,
        quadrature,
      );
      const tolerance = fixture.tolerance.absolute
        + fixture.tolerance.relative * Math.abs(entry.expected_score);
      expect(Math.abs(observed - entry.expected_score)).toBeLessThanOrEqual(
        tolerance,
      );
      if (entry.property_contract.score_minimum !== undefined) {
        expect(observed).toBeGreaterThanOrEqual(
          entry.property_contract.score_minimum,
        );
      }
      if (entry.property_contract.score_maximum !== undefined) {
        expect(observed).toBeLessThanOrEqual(
          entry.property_contract.score_maximum,
        );
      }
      expect(observedById.has(entry.id)).toBe(false);
      observedById.set(entry.id, observed);
    }
    for (const entry of fixture.phase_linearity_raw_cases) {
      const referenceId = entry.property_contract.reference_case;
      if (referenceId === undefined) continue;
      const reference = observedById.get(referenceId);
      const observed = observedById.get(entry.id);
      expect(reference).toBeDefined();
      expect(observed).toBeDefined();
      expect(
        Math.abs(observed! - reference!),
      ).toBeLessThanOrEqual(
        entry.property_contract.maximum_reference_delta!,
      );
    }
  });
});
