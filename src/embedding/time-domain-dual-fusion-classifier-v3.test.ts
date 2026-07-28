import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  assertTimeDomainDualFusionAssetSetV3,
  createTimeDomainDualFusionClassifierV3,
  loadTimeDomainDualFusionBindingV3,
  TimeDomainDualFusionClassifierV3,
  TIME_DOMAIN_DUAL_FUSION_ARCHITECTURE_CONTRACT,
  TIME_DOMAIN_DUAL_FUSION_BINDING_SCHEMA,
  type TimeDomainDualFusionBindingV3,
  type TimeDomainDualFusionOpenSetRuntimeV3,
} from './time-domain-dual-fusion-classifier-v3.js';
import {
  TIME_DOMAIN_FUSION_CLASSIFIER_ROLE_V3,
  TIME_DOMAIN_FUSION_REJECTOR_ROLE_V3,
  TIME_DOMAIN_FUSION_V3_SCHEMA,
  TIME_DOMAIN_FUSION_V3_SCHEMA_VERSION,
  type TimeDomainClassificationResultV3,
  type TimeDomainFusionAssetV3,
  type TimeDomainFusionRuntimeRoleV3,
} from './time-domain-fusion-v3.js';
import {
  stagedArchitectureContract,
  TIME_DOMAIN_OPENSET_SCHEMA,
  TIME_DOMAIN_OPENSET_SCHEMA_VERSION,
  type StagedOpenSetDecision,
  type StageOneEvaluation,
  type TimeDomainOpenSetAssetV3,
} from './time-domain-openset-v3.js';
import type {
  TimeDomainInvariantPatchResult,
} from './time-domain-invariant-patch-preprocess-v3.js';

const REJECTOR_ASSET_SHA = '1'.repeat(64);
const CLASSIFIER_ASSET_SHA = '2'.repeat(64);
const OPENSET_ASSET_SHA = '3'.repeat(64);
const REJECTOR_DIRECTORY_SHA = '4'.repeat(64);
const CLASSIFIER_DIRECTORY_SHA = '5'.repeat(64);
const REJECTOR_RUNTIME_SHA = '6'.repeat(64);
const CLASSIFIER_RUNTIME_SHA = '7'.repeat(64);
const VALIDATION_REPORT_SHA = '8'.repeat(64);

function rawBinding(
  status: 'staging_not_release' | 'release' = 'staging_not_release',
): Record<string, unknown> {
  return {
    schema: TIME_DOMAIN_DUAL_FUSION_BINDING_SCHEMA,
    schema_version: 1,
    status,
    candidate_id: 'v3.4-q97-decoupled-8k-classifier-4k-rejector',
    frontend: {
      version: 'invariant-patch-time-domain-v1',
      patch_length: 2,
      patch_count: 2,
      target_frac: 0.5,
      packed_length: 4,
      uses_frequency_transform: false,
    },
    execution_order: [
      'stage_one_noise_gate',
      'rejector_known_unknown',
      'classifier_known_label',
    ],
    roles: {
      rejector: {
        asset: 'time-domain-v3-rejector-weights.json',
        asset_sha256: REJECTOR_ASSET_SHA,
        fusion_directory_sha256: REJECTOR_DIRECTORY_SHA,
        runtime_bundle_manifest_sha256: REJECTOR_RUNTIME_SHA,
        runtime_role: TIME_DOMAIN_FUSION_REJECTOR_ROLE_V3,
        responsibility: 'known_unknown_only',
      },
      classifier: {
        asset: 'time-domain-v3-classifier-weights.json',
        asset_sha256: CLASSIFIER_ASSET_SHA,
        fusion_directory_sha256: CLASSIFIER_DIRECTORY_SHA,
        runtime_bundle_manifest_sha256: CLASSIFIER_RUNTIME_SHA,
        runtime_role: TIME_DOMAIN_FUSION_CLASSIFIER_ROLE_V3,
        responsibility: 'accepted_known_label_only',
      },
    },
    openset_policy: {
      asset: 'time-domain-v3-openset-policy.json',
      asset_sha256: OPENSET_ASSET_SHA,
      rejector_asset_sha256: REJECTOR_ASSET_SHA,
      fitted_rejector_runtime_bundle_manifest_sha256: REJECTOR_RUNTIME_SHA,
      staged_validation_report_sha256: VALIDATION_REPORT_SHA,
      staged_artifacts_sha256: {
        'v3_branch_lof_components.npz': '9'.repeat(64),
        'v3_open_policy_stage_two.npz': 'a'.repeat(64),
        'v3_staged_composite_policy.npz': 'b'.repeat(64),
      },
    },
    validation: {
      report_sha256: VALIDATION_REPORT_SHA,
      role: 'validate',
      status: 'development_openset_pass',
      design_novelty_seed: 20260955,
      novelty_seeds: [20260953, 20260954],
      release_seed_not_spent: 20260736,
    },
    fail_closed: {
      role_assets_bound_by_sha256: true,
      distinct_role_assets: true,
      role_asset_sha256_must_differ: true,
      classifier_runs_only_after_rejector_acceptance: true,
      public_known_label_from_classifier_only: true,
    },
  };
}

function binding(): TimeDomainDualFusionBindingV3 {
  return loadTimeDomainDualFusionBindingV3(rawBinding());
}

function fusionAsset(options: {
  role: TimeDomainFusionRuntimeRoleV3;
  status?: 'staging_not_release' | 'release';
  classes?: string[];
  embedDim?: number;
  featureCount?: number;
  patchLength?: number;
  patchCount?: number;
}): TimeDomainFusionAssetV3 {
  const {
    role,
    status = 'staging_not_release',
    classes = ['alpha', 'beta'],
    embedDim = 2,
    featureCount = 12,
    patchLength = 2,
    patchCount = 2,
  } = options;
  const config = {
    patch_length: patchLength,
    patch_count: patchCount,
    embed_dim: embedDim,
    n_features: featureCount,
  };
  return {
    schema: TIME_DOMAIN_FUSION_V3_SCHEMA,
    schema_version: TIME_DOMAIN_FUSION_V3_SCHEMA_VERSION,
    status,
    runtime_role: role,
    packed_length: patchLength * patchCount,
    frontend: {
      version: 'invariant-patch-time-domain-v1',
      patch_length: patchLength,
      patch_count: patchCount,
      target_frac: 0.5,
      feature_count: featureCount,
      uses_frequency_transform: false,
    },
    real: { config: { ...config, encoder: 'real' } },
    complex: { config: { ...config, encoder: 'complex' } },
    fusion: {
      real_center: new Array<number>(embedDim).fill(0),
      complex_center: new Array<number>(embedDim).fill(0),
      alpha_real: role === TIME_DOMAIN_FUSION_REJECTOR_ROLE_V3 ? 0.2 : 0.4,
      alpha_complex: 0.2,
      weight_real: 0.5,
      eps: 1e-12,
    },
    feature_standardization: {
      mean: new Array<number>(featureCount).fill(0),
      std: new Array<number>(featureCount).fill(1),
    },
    classification: {
      classes,
      prototypes: classes.map(() =>
        new Array<number>(2 * embedDim).fill(0)),
    },
    provenance: {
      source_bundle_manifest_sha256:
        role === TIME_DOMAIN_FUSION_REJECTOR_ROLE_V3
          ? REJECTOR_RUNTIME_SHA
          : CLASSIFIER_RUNTIME_SHA,
    },
  } as unknown as TimeDomainFusionAssetV3;
}

function opensetAsset(options: {
  status?: 'staging_not_release' | 'release';
  classCount?: number;
  rejectorEmbedDim?: number;
  patchLength?: number;
  patchCount?: number;
  targetFrac?: number;
} = {}): TimeDomainOpenSetAssetV3 {
  const {
    status = 'staging_not_release',
    classCount = 2,
    rejectorEmbedDim = 2,
    patchLength = 2,
    patchCount = 2,
    targetFrac = 0.5,
  } = options;
  const component = (branch: 'real' | 'complex') => ({
    branch,
    mean: new Array<number>(rejectorEmbedDim).fill(0),
    scale: new Array<number>(rejectorEmbedDim).fill(1),
    reference: [
      new Array<number>(rejectorEmbedDim).fill(0),
      new Array<number>(rejectorEmbedDim).fill(1),
    ],
  });
  return {
    schema: TIME_DOMAIN_OPENSET_SCHEMA,
    schema_version: TIME_DOMAIN_OPENSET_SCHEMA_VERSION,
    status,
    contract: stagedArchitectureContract(),
    frontend: {
      patch_length: patchLength,
      patch_count: patchCount,
      target_frac: targetFrac,
      packed_length: patchLength * patchCount,
    },
    stage_two: {
      lof_components: [component('real'), component('complex')],
      policy: {
        class_geometry_mean: new Array<number>(classCount).fill(0),
      },
    },
    provenance: {
      candidate_id: 'v3.4-q97-decoupled-8k-classifier-4k-rejector',
      runtime_bundle_manifest_sha256: REJECTOR_RUNTIME_SHA,
      fusion_directory_sha256: REJECTOR_DIRECTORY_SHA,
      staged_validation_report_sha256: VALIDATION_REPORT_SHA,
      staged_artifact_sha256: {
        'v3_branch_lof_components.npz': '9'.repeat(64),
        'v3_open_policy_stage_two.npz': 'a'.repeat(64),
        'v3_staged_composite_policy.npz': 'b'.repeat(64),
      },
    },
  } as unknown as TimeDomainOpenSetAssetV3;
}

function stageOne(gated: boolean): StageOneEvaluation {
  return {
    features: new Float64Array(7),
    score: gated ? 2 : -2,
    rank: gated ? 0.9 : 0.1,
    gated,
    thresholdScore: 0,
    captureLength: 4,
    evaluatedLength: 4,
    causalPrefixApplied: false,
  };
}

function openDecision(
  evaluation: StageOneEvaluation,
  rejectedStage: 1 | 2 | null,
): StagedOpenSetDecision {
  return {
    gated: rejectedStage === 1,
    rejectedStage,
    stageOne: evaluation,
    stageTwo: rejectedStage === 1
      ? null
      : ({ score: rejectedStage === 2 ? 0.99 : 0.1 } as never),
    stageOneSurvivorRank: rejectedStage === 1 ? null : 0.1,
    compositeScore: rejectedStage === 1
      ? null
      : (rejectedStage === 2 ? 0.99 : 0.1),
    stagedScore: rejectedStage === 1 ? 1.9 : 0.1,
    threshold: 0.95,
    contract: stagedArchitectureContract(),
  };
}

function preprocessResult(): TimeDomainInvariantPatchResult {
  return {
    packedInPhase: Float64Array.from([1, 2, 3, 4]),
    packedQuadrature: Float64Array.from([5, 6, 7, 8]),
    rawFeatures: Float64Array.from({ length: 12 }, (_, index) => index),
    context: {
      version: 'invariant-patch-time-domain-v1',
      patch_length: 2,
      patch_count: 2,
      target_frac: 0.5,
      packed_length: 4,
      uses_frequency_transform: false,
    },
  } as TimeDomainInvariantPatchResult;
}

function classificationResult(options: {
  label: string;
  winner: number;
  real: number[];
  complex: number[];
  distance: number[];
}): TimeDomainClassificationResultV3 {
  return {
    standardizedFeatures: new Float64Array(12),
    realEmbedding: Float64Array.from(options.real),
    complexEmbedding: Float64Array.from(options.complex),
    fusedEmbedding: Float64Array.from([
      ...options.real,
      ...options.complex,
    ]),
    squaredPrototypeDistances: Float64Array.from(options.distance),
    closedWinnerIndex: options.winner,
    closedLabel: options.label,
  };
}

function runtimeHarness(options: {
  rejectedStage: 1 | 2 | null;
  inconsistentSurvivor?: boolean;
}): {
  classifier: TimeDomainDualFusionClassifierV3;
  calls: string[];
  survivorInputs: Array<{
    realEmbedding: ArrayLike<number>;
    complexEmbedding: ArrayLike<number>;
    packedInPhase: ArrayLike<number>;
    packedQuadrature: ArrayLike<number>;
    predictedClassIndex: number;
  }>;
  preprocess: TimeDomainInvariantPatchResult;
  rejector: TimeDomainClassificationResultV3;
  classifierResult: TimeDomainClassificationResultV3;
} {
  const calls: string[] = [];
  const survivorInputs: Array<{
    realEmbedding: ArrayLike<number>;
    complexEmbedding: ArrayLike<number>;
    packedInPhase: ArrayLike<number>;
    packedQuadrature: ArrayLike<number>;
    predictedClassIndex: number;
  }> = [];
  const preprocess = preprocessResult();
  const rejector = classificationResult({
    label: 'alpha',
    winner: 0,
    real: [10, 11],
    complex: [12, 13],
    distance: [0.1, 0.8],
  });
  const classifierResult = classificationResult({
    label: 'beta',
    winner: 1,
    real: [20, 21, 22],
    complex: [23, 24, 25],
    distance: [0.7, 0.05],
  });
  let evaluation: StageOneEvaluation;
  const openAsset = opensetAsset();
  const openSet: TimeDomainDualFusionOpenSetRuntimeV3 = {
    asset: openAsset,
    stageTwo: { classCount: 2 },
    evaluateStageOne: () => {
      calls.push('stage1');
      evaluation = stageOne(options.rejectedStage === 1);
      return evaluation;
    },
    gatedDecision: (received) => openDecision(received, 1),
    finishSurvivor: (received, survivor) => {
      calls.push('open');
      survivorInputs.push(survivor);
      if (options.inconsistentSurvivor) {
        return openDecision(received, 1);
      }
      return openDecision(received, options.rejectedStage);
    },
  };
  const classifier = new TimeDomainDualFusionClassifierV3({
    binding: binding(),
    rejectorFusion: {
      asset: fusionAsset({
        role: TIME_DOMAIN_FUSION_REJECTOR_ROLE_V3,
        embedDim: 2,
      }),
      preverifiedAssetSha256: REJECTOR_ASSET_SHA,
    },
    classifierFusion: {
      // A different embedding width and fusion parameters are intentional.
      asset: fusionAsset({
        role: TIME_DOMAIN_FUSION_CLASSIFIER_ROLE_V3,
        embedDim: 3,
      }),
      preverifiedAssetSha256: CLASSIFIER_ASSET_SHA,
    },
    opensetPolicy: {
      asset: openAsset,
      preverifiedAssetSha256: OPENSET_ASSET_SHA,
    },
    dependencies: {
      openSet,
      preprocess: () => {
        calls.push('preprocess');
        return preprocess;
      },
      rejectorClassify: () => {
        calls.push('rejector4k');
        return rejector;
      },
      classifierClassify: () => {
        calls.push('classifier8k');
        return classifierResult;
      },
    },
  });
  return {
    classifier,
    calls,
    survivorInputs,
    preprocess,
    rejector,
    classifierResult,
  };
}

describe('dual-fusion role binding', () => {
  it('loads the exact frozen execution and responsibility contract', () => {
    const loaded = binding();
    expect(loaded.execution_order).toEqual([
      'stage_one_noise_gate',
      'rejector_known_unknown',
      'classifier_known_label',
    ]);
    expect(loaded.roles.rejector.runtime_role).toBe(
      TIME_DOMAIN_FUSION_REJECTOR_ROLE_V3,
    );
    expect(loaded.roles.classifier.runtime_role).toBe(
      TIME_DOMAIN_FUSION_CLASSIFIER_ROLE_V3,
    );
    expect(
      loaded.openset_policy
        .fitted_rejector_runtime_bundle_manifest_sha256,
    ).toBe(loaded.roles.rejector.runtime_bundle_manifest_sha256);
  });

  it.each([
    ['malformed hash', (raw: Record<string, any>) => {
      raw.roles.rejector.asset_sha256 = 'NOT-A-HASH';
    }],
    ['another candidate id', (raw: Record<string, any>) => {
      raw.candidate_id = 'v3.4-different-candidate';
    }],
    ['changed execution order', (raw: Record<string, any>) => {
      raw.execution_order.reverse();
    }],
    ['swapped runtime roles', (raw: Record<string, any>) => {
      raw.roles.rejector.runtime_role =
        TIME_DOMAIN_FUSION_CLASSIFIER_ROLE_V3;
    }],
    ['false fail-closed declaration', (raw: Record<string, any>) => {
      raw.fail_closed.public_known_label_from_classifier_only = false;
    }],
    ['same role asset identity', (raw: Record<string, any>) => {
      raw.roles.classifier.asset_sha256 = REJECTOR_ASSET_SHA;
    }],
    ['same role source identity', (raw: Record<string, any>) => {
      raw.roles.classifier.runtime_bundle_manifest_sha256 =
        REJECTOR_RUNTIME_SHA;
    }],
    ['open set fitted to another runtime', (raw: Record<string, any>) => {
      raw.openset_policy.fitted_rejector_runtime_bundle_manifest_sha256 =
        CLASSIFIER_RUNTIME_SHA;
    }],
    ['open set bound to another asset', (raw: Record<string, any>) => {
      raw.openset_policy.rejector_asset_sha256 = CLASSIFIER_ASSET_SHA;
    }],
    ['validation report mismatch', (raw: Record<string, any>) => {
      raw.validation.report_sha256 = 'c'.repeat(64);
    }],
    ['different validation seeds', (raw: Record<string, any>) => {
      raw.validation.novelty_seeds = [20260954, 20260953];
    }],
    ['different design seed', (raw: Record<string, any>) => {
      raw.validation.design_novelty_seed = 20260956;
    }],
    ['different release seed', (raw: Record<string, any>) => {
      raw.validation.release_seed_not_spent = 20260737;
    }],
    ['frequency-transform frontend', (raw: Record<string, any>) => {
      raw.frontend.uses_frequency_transform = true;
    }],
  ])('rejects %s', (_name, mutate) => {
    const raw = rawBinding() as Record<string, any>;
    mutate(raw);
    expect(() => loadTimeDomainDualFusionBindingV3(raw)).toThrow();
  });

  it('admits staging and production bindings only in their own channel', () => {
    expect(loadTimeDomainDualFusionBindingV3(rawBinding()).status).toBe(
      'staging_not_release',
    );
    expect(() =>
      loadTimeDomainDualFusionBindingV3(rawBinding(), {
        admission: 'production',
      })).toThrow(/must be release for production admission/);
    expect(loadTimeDomainDualFusionBindingV3(rawBinding('release'), {
      admission: 'production',
    }).status).toBe('release');
  });
});

describe('dual-fusion asset-set compatibility', () => {
  function coherent() {
    return {
      binding: binding(),
      rejector: {
        asset: fusionAsset({
          role: TIME_DOMAIN_FUSION_REJECTOR_ROLE_V3,
          embedDim: 2,
        }),
        preverifiedAssetSha256: REJECTOR_ASSET_SHA,
      },
      classifier: {
        asset: fusionAsset({
          role: TIME_DOMAIN_FUSION_CLASSIFIER_ROLE_V3,
          embedDim: 3,
        }),
        preverifiedAssetSha256: CLASSIFIER_ASSET_SHA,
      },
      openset: {
        asset: opensetAsset({ rejectorEmbedDim: 2 }),
        preverifiedAssetSha256: OPENSET_ASSET_SHA,
      },
    };
  }

  it('accepts intentionally different rejector and classifier internals', () => {
    const assets = coherent();
    expect(() =>
      assertTimeDomainDualFusionAssetSetV3(
        assets.binding,
        assets.rejector,
        assets.classifier,
        assets.openset,
      )).not.toThrow();
    expect(assets.rejector.asset.real.config.embed_dim).toBe(2);
    expect(assets.classifier.asset.real.config.embed_dim).toBe(3);
    expect(assets.rejector.asset.fusion.alpha_real).not.toBe(
      assets.classifier.asset.fusion.alpha_real,
    );
  });

  it.each([
    ['digest mismatch', (assets: ReturnType<typeof coherent>) => {
      assets.rejector.preverifiedAssetSha256 = 'd'.repeat(64);
    }],
    ['role swap', (assets: ReturnType<typeof coherent>) => {
      assets.rejector.asset.runtime_role =
        TIME_DOMAIN_FUSION_CLASSIFIER_ROLE_V3;
    }],
    ['mixed status', (assets: ReturnType<typeof coherent>) => {
      assets.classifier.asset.status = 'release';
    }],
    ['source provenance mismatch', (assets: ReturnType<typeof coherent>) => {
      (
        assets.classifier.asset.provenance as Record<string, unknown>
      ).source_bundle_manifest_sha256 = REJECTOR_RUNTIME_SHA;
    }],
    ['open fitted-source mismatch', (assets: ReturnType<typeof coherent>) => {
      (
        assets.openset.asset.provenance as Record<string, unknown>
      ).fusion_directory_sha256 = CLASSIFIER_DIRECTORY_SHA;
    }],
    ['open candidate mismatch', (assets: ReturnType<typeof coherent>) => {
      (
        assets.openset.asset.provenance as Record<string, unknown>
      ).candidate_id = 'another-candidate';
    }],
    ['frontend mismatch', (assets: ReturnType<typeof coherent>) => {
      assets.classifier.asset.packed_length = 5;
    }],
    ['feature width mismatch', (assets: ReturnType<typeof coherent>) => {
      assets.rejector.asset.real.config.n_features = 11;
    }],
    ['class order mismatch', (assets: ReturnType<typeof coherent>) => {
      assets.classifier.asset.classification.classes = ['beta', 'alpha'];
    }],
    ['open class count mismatch', (assets: ReturnType<typeof coherent>) => {
      assets.openset.asset.stage_two.policy.class_geometry_mean = [0];
    }],
    ['LOF dimension mismatch', (assets: ReturnType<typeof coherent>) => {
      assets.openset.asset.stage_two.lof_components[0]!.mean = [0];
    }],
  ])('refuses %s before inference', (_name, mutate) => {
    const assets = coherent();
    mutate(assets);
    expect(() =>
      assertTimeDomainDualFusionAssetSetV3(
        assets.binding,
        assets.rejector,
        assets.classifier,
        assets.openset,
      )).toThrow();
  });
});

describe('dual-fusion execution order and public result', () => {
  it('short-circuits stage-1 noise before preprocessing or either fusion', () => {
    const harness = runtimeHarness({ rejectedStage: 1 });
    const result = harness.classifier.classify([1, 2, 3, 4], [0, 0, 0, 0]);
    expect(harness.calls).toEqual(['stage1']);
    expect(result.outcome).toBe('noise');
    expect(result.knownLabel).toBeNull();
    expect(result.squaredPrototypeDistances).toBeNull();
    expect(result.forward).toBeNull();
  });

  it('uses the 4k fusion for rejection and never runs 8k for unknowns', () => {
    const harness = runtimeHarness({ rejectedStage: 2 });
    const result = harness.classifier.classify([1, 2, 3, 4], [0, 0, 0, 0]);
    expect(harness.calls).toEqual([
      'stage1',
      'preprocess',
      'rejector4k',
      'open',
    ]);
    expect(result.outcome).toBe('unknown');
    expect(result.label).toBe('unknown');
    expect(result.knownLabel).toBeNull();
    expect(result.knownWinnerIndex).toBeNull();
    expect(result.squaredPrototypeDistances).toBeNull();
    expect(result.forward!.classifierForward).toBeNull();
    expect(result.forward!.rejectorForward).toBe(harness.rejector);

    const survivor = harness.survivorInputs[0]!;
    expect(survivor.realEmbedding).toBe(harness.rejector.realEmbedding);
    expect(survivor.complexEmbedding).toBe(harness.rejector.complexEmbedding);
    expect(survivor.predictedClassIndex).toBe(
      harness.rejector.closedWinnerIndex,
    );
    expect(survivor.packedInPhase).toBe(harness.preprocess.packedInPhase);
    expect(survivor.packedQuadrature).toBe(
      harness.preprocess.packedQuadrature,
    );
  });

  it('runs 8k only after acceptance and returns only its known label', () => {
    const harness = runtimeHarness({ rejectedStage: null });
    const result = harness.classifier.classify([1, 2, 3, 4], [0, 0, 0, 0]);
    expect(harness.calls).toEqual([
      'stage1',
      'preprocess',
      'rejector4k',
      'open',
      'classifier8k',
    ]);
    expect(harness.rejector.closedLabel).toBe('alpha');
    expect(harness.classifierResult.closedLabel).toBe('beta');
    expect(result.outcome).toBe('known');
    expect(result.label).toBe('beta');
    expect(result.knownLabel).toBe('beta');
    expect(result.knownWinnerIndex).toBe(1);
    expect(result.squaredPrototypeDistances).toBe(
      harness.classifierResult.squaredPrototypeDistances,
    );
    expect(result.forward!.classifierForward).toBe(
      harness.classifierResult,
    );
    expect(result.architectureContract).toBe(
      TIME_DOMAIN_DUAL_FUSION_ARCHITECTURE_CONTRACT,
    );
  });

  it('fails closed on an inconsistent survivor decision before 8k runs', () => {
    const harness = runtimeHarness({
      rejectedStage: null,
      inconsistentSurvivor: true,
    });
    expect(() =>
      harness.classifier.classify(
        [1, 2, 3, 4],
        [0, 0, 0, 0],
      )).toThrow(/inconsistent survivor decision/);
    expect(harness.calls).toEqual([
      'stage1',
      'preprocess',
      'rejector4k',
      'open',
    ]);
  });
});

describe('dual-fusion public factory', () => {
  it('refuses legacy q95 JSON before composing the q97 runtime', () => {
    const staging = new URL('./assets-v3-staging/', import.meta.url);
    const fusionRaw = JSON.parse(readFileSync(
      new URL('time-domain-fusion-weights-v3.json', staging),
      'utf8',
    )) as Record<string, any>;
    const opensetRaw = JSON.parse(readFileSync(
      new URL('time-domain-openset-weights-v1.json', staging),
      'utf8',
    )) as Record<string, any>;
    const raw = rawBinding();
    raw.frontend = {
      version: 'invariant-patch-time-domain-v1',
      patch_length: 64,
      patch_count: 16,
      target_frac: 0.5,
      packed_length: 1024,
      uses_frequency_transform: false,
    };
    const rejectorRaw = {
      ...fusionRaw,
      runtime_role: TIME_DOMAIN_FUSION_REJECTOR_ROLE_V3,
      provenance: {
        ...fusionRaw.provenance,
        source_bundle_manifest_sha256: REJECTOR_RUNTIME_SHA,
      },
    };
    const classifierRaw = {
      ...fusionRaw,
      runtime_role: TIME_DOMAIN_FUSION_CLASSIFIER_ROLE_V3,
      provenance: {
        ...fusionRaw.provenance,
        source_bundle_manifest_sha256: CLASSIFIER_RUNTIME_SHA,
      },
    };
    const roleBoundOpenSetRaw = {
      ...opensetRaw,
      provenance: {
        ...opensetRaw.provenance,
        candidate_id: 'v3.4-q97-decoupled-8k-classifier-4k-rejector',
        runtime_bundle_manifest_sha256: REJECTOR_RUNTIME_SHA,
        fusion_directory_sha256: REJECTOR_DIRECTORY_SHA,
        staged_validation_report_sha256: VALIDATION_REPORT_SHA,
        staged_artifact_sha256: {
          'v3_branch_lof_components.npz': '9'.repeat(64),
          'v3_open_policy_stage_two.npz': 'a'.repeat(64),
          'v3_staged_composite_policy.npz': 'b'.repeat(64),
        },
      },
    };
    expect(() =>
      createTimeDomainDualFusionClassifierV3({
        bindingAsset: raw,
        rejectorFusion: {
          asset: rejectorRaw,
          preverifiedAssetSha256: REJECTOR_ASSET_SHA,
        },
        classifierFusion: {
          asset: classifierRaw,
          preverifiedAssetSha256: CLASSIFIER_ASSET_SHA,
        },
        opensetPolicy: {
          asset: roleBoundOpenSetRaw,
          preverifiedAssetSha256: OPENSET_ASSET_SHA,
        },
      })).toThrow(/unsupported open-set schema version|staged policy version/);
  });
});
