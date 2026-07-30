import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  classifyTimeDomainProfileBankV4,
  loadTimeDomainProfileBankAssetV4,
  minimumPublicClassSquaredDistancesV4,
  selectTimeDomainRuntimeInputLengthV4,
  TIME_DOMAIN_PROFILE_BANK_DECISION_RULE_V4,
  TIME_DOMAIN_PROFILE_BANK_RUNTIME_ROLE_V4,
  TIME_DOMAIN_PROFILE_BANK_SCHEMA_V4,
  TIME_DOMAIN_PROFILE_BANK_SCHEMA_VERSION_V4,
  TIME_DOMAIN_PUBLIC_CLASSES_V4,
  TIME_DOMAIN_RUNTIME_BUCKET_RULE_V4,
  type TimeDomainProfileBankAssetV4,
} from './time-domain-profile-bank-classifier-v4.js';
import {
  isTimeDomainCurrentProfileV4,
  prototypeSourceForAcquisitionV4,
  TIME_DOMAIN_CURRENT_PROFILE_INVENTORY_V4,
  TIME_DOMAIN_CURRENT_PROFILE_PUBLIC_CLASS_V4,
  TIME_DOMAIN_TRUSTED_SOURCE_ROUTING_V4,
} from './time-domain-profile-routing-v4.js';

const legacyRaw = JSON.parse(
  readFileSync(
    new URL(
      './assets-v3-staging/time-domain-fusion-weights-v3.json',
      import.meta.url,
    ),
    'utf8',
  ),
) as Record<string, unknown>;

function smokeAssetRaw(): Record<string, unknown> {
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
      eligible_lengths: profile === 'bluetooth-le-advertising'
        ? [4_096, 8_192]
        : [4_096, 8_192, 16_384],
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
    provenance: { smoke_from_v3_encoder_weights: true },
  };
}

function loadSmokeAsset(): TimeDomainProfileBankAssetV4 {
  return loadTimeDomainProfileBankAssetV4(smokeAssetRaw());
}

describe('v4 profile-bank asset schema', () => {
  it('admits a mapped multi-prototype bank and preserves canonical output order', () => {
    const asset = loadSmokeAsset();
    expect(asset.classification.classes).toEqual(
      [...TIME_DOMAIN_PUBLIC_CLASSES_V4],
    );
    expect(asset.classification.prototype_bank).toHaveLength(38);
    expect(asset.classification.prototype_public_class_indices.slice(0, 7))
      .toEqual([0, 1, 2, 3, 4, 5, 6]);
    expect(asset.classification.routing).toEqual(
      TIME_DOMAIN_TRUSTED_SOURCE_ROUTING_V4,
    );
    expect(asset.rejection).toEqual({
      state: 'unset',
      runtime_behaviour: 'closed_set_only_no_abstention',
      external_policy_required_for_abstention: true,
    });
    expect(asset.frontend).toMatchObject({
      uses_frequency_transform: false,
      runtime_input_lengths: [4_096, 8_192, 16_384],
      runtime_bucket_rule:
        'largest_supported_prefix_not_exceeding_valid_sample_count',
    });
  });

  it('cannot silently load v3, reorder labels, omit coverage, or enable an FFT', () => {
    expect(() => loadTimeDomainProfileBankAssetV4(legacyRaw)).toThrow(/schema/);

    const reordered = smokeAssetRaw();
    const reorderedHead = reordered.classification as Record<string, unknown>;
    reorderedHead.classes = [
      'bluetooth', 'am', 'cw', 'dsss', 'fm', 'gsm', 'ofdm',
    ];
    expect(() => loadTimeDomainProfileBankAssetV4(reordered)).toThrow(
      /canonical seven/,
    );

    const uncovered = smokeAssetRaw();
    const uncoveredHead = uncovered.classification as Record<string, unknown>;
    const uncoveredLabels = (
      uncoveredHead.prototype_public_class_indices as number[]
    ).map((label) => label === 6 ? 5 : label);
    uncoveredHead.prototype_public_class_indices = uncoveredLabels;
    uncoveredHead.prototype_groups = (
      uncoveredHead.prototype_groups as Array<Record<string, unknown>>
    ).map((group, index) => ({
      ...group,
      public_class: TIME_DOMAIN_PUBLIC_CLASSES_V4[uncoveredLabels[index]!],
    }));
    expect(() => loadTimeDomainProfileBankAssetV4(uncovered)).toThrow(
      /no row for class 6/,
    );

    const transformed = smokeAssetRaw();
    (transformed.frontend as Record<string, unknown>).uses_frequency_transform =
      true;
    expect(() => loadTimeDomainProfileBankAssetV4(transformed)).toThrow(
      /uses_frequency_transform/,
    );
  });

  it('rejects row-map and profile-metadata disagreement', () => {
    const raw = smokeAssetRaw();
    const head = raw.classification as Record<string, unknown>;
    const groups = head.prototype_groups as Array<Record<string, unknown>>;
    groups[2] = { ...groups[2], public_class: 'fm' };
    expect(() => loadTimeDomainProfileBankAssetV4(raw)).toThrow(
      /disagrees with prototype_public_class_indices/,
    );
  });

  it('independently enforces the exact current inventory and WiFi DSSS map', () => {
    const tampered = smokeAssetRaw();
    const tamperedHead = tampered.classification as Record<string, unknown>;
    const tamperedGroups = (
      tamperedHead.prototype_groups as Array<Record<string, unknown>>
    );
    const wifiIndex = tamperedGroups.findIndex(
      (group) =>
        group.source === 'current'
        && group.profile === 'wifi-hr-dsss-11m',
    );
    expect(wifiIndex).toBeGreaterThanOrEqual(0);
    (tamperedHead.prototype_public_class_indices as number[])[wifiIndex] = 1;
    tamperedGroups[wifiIndex] = {
      ...tamperedGroups[wifiIndex],
      public_class: 'bluetooth',
    };
    expect(() => loadTimeDomainProfileBankAssetV4(tampered)).toThrow(
      /wifi-hr-dsss-11m must map to dsss/,
    );

    const incomplete = smokeAssetRaw();
    const incompleteHead = incomplete.classification as Record<string, unknown>;
    const incompleteGroups = (
      incompleteHead.prototype_groups as Array<Record<string, unknown>>
    );
    const missingIndex = incompleteGroups.findIndex(
      (group) => group.profile === 'wifi6-he-tb',
    );
    incompleteGroups.splice(missingIndex, 1);
    (incompleteHead.prototype_bank as number[][]).splice(missingIndex, 1);
    (
      incompleteHead.prototype_public_class_indices as number[]
    ).splice(missingIndex, 1);
    expect(() => loadTimeDomainProfileBankAssetV4(incomplete)).toThrow(
      /exact 31-profile inventory/,
    );

    const aliasedSource = smokeAssetRaw();
    const aliasedGroups = (
      (aliasedSource.classification as Record<string, unknown>)
        .prototype_groups as Array<Record<string, unknown>>
    );
    const currentIndex = aliasedGroups.findIndex(
      (group) => group.source === 'current',
    );
    aliasedGroups[currentIndex] = {
      ...aliasedGroups[currentIndex],
      source: 'current-signallab',
    };
    expect(() => loadTimeDomainProfileBankAssetV4(aliasedSource)).toThrow(
      /source must be current or historical/,
    );
  });

  it('rejects a changed selected-profile routing contract', () => {
    const raw = smokeAssetRaw();
    const head = raw.classification as Record<string, unknown>;
    head.routing = {
      ...TIME_DOMAIN_TRUSTED_SOURCE_ROUTING_V4,
      current_profile_inventory: [
        ...TIME_DOMAIN_CURRENT_PROFILE_INVENTORY_V4.slice(1),
      ],
    };
    expect(() => loadTimeDomainProfileBankAssetV4(raw)).toThrow(
      /selected-profile-conditioned routing contract/,
    );
  });
});

describe('v4 trusted acquisition routing', () => {
  it('routes only canonical SignalLab profiles to current prototypes', () => {
    expect(isTimeDomainCurrentProfileV4('wifi-hr-dsss-11m')).toBe(true);
    expect(
      prototypeSourceForAcquisitionV4(
        'signal-lab',
        'wifi-hr-dsss-11m',
      ),
    ).toBe('current');
    expect(prototypeSourceForAcquisitionV4('signal-lab', 'am')).toBe(
      'historical',
    );
    expect(
      prototypeSourceForAcquisitionV4('signal-lab', 'custom-wifi'),
    ).toBe('historical');
    expect(prototypeSourceForAcquisitionV4('physical-sdr')).toBe(
      'historical',
    );
    expect(prototypeSourceForAcquisitionV4('untagged')).toBe('historical');
  });
});

describe('v4 profile-bank public-class distance parity', () => {
  const fixture = JSON.parse(
    readFileSync(
      new URL(
        './test-fixtures/time-domain-profile-bank-v4-distance-parity.json',
        import.meta.url,
      ),
      'utf8',
    ),
  ) as {
    schema: string;
    embedding: number[];
    prototype_bank: number[][];
    prototype_public_class_indices: number[];
    expected_squared_public_class_distances: number[];
    expected_closest_prototype_indices: number[];
    expected_winner_index: number;
  };

  it('matches the NumPy grouped-minimum reference and returns exactly 7 distances', () => {
    expect(fixture.schema).toBe(
      'atomos.v4.time-domain-current-source-profile-bank.distance-parity',
    );
    const result = minimumPublicClassSquaredDistancesV4(
      fixture.embedding,
      fixture.prototype_bank,
      fixture.prototype_public_class_indices,
    );
    expect([...result.squaredPublicClassDistances]).toHaveLength(7);
    result.squaredPublicClassDistances.forEach((value, index) => {
      expect(value).toBeCloseTo(
        fixture.expected_squared_public_class_distances[index]!,
        14,
      );
    });
    expect([...result.closestPrototypeIndices]).toEqual(
      fixture.expected_closest_prototype_indices,
    );
    let winner = 0;
    for (let index = 1; index < result.squaredPublicClassDistances.length; index++) {
      if (
        result.squaredPublicClassDistances[index]!
        < result.squaredPublicClassDistances[winner]!
      ) {
        winner = index;
      }
    }
    expect(winner).toBe(fixture.expected_winner_index);
  });
});

describe('v4 raw runtime length buckets', () => {
  it('selects the largest supported prefix and refuses undersized captures', () => {
    expect(selectTimeDomainRuntimeInputLengthV4(4_096)).toBe(4_096);
    expect(selectTimeDomainRuntimeInputLengthV4(8_191)).toBe(4_096);
    expect(selectTimeDomainRuntimeInputLengthV4(8_192)).toBe(8_192);
    expect(selectTimeDomainRuntimeInputLengthV4(12_160)).toBe(8_192);
    expect(selectTimeDomainRuntimeInputLengthV4(16_384)).toBe(16_384);
    expect(selectTimeDomainRuntimeInputLengthV4(1_000_000)).toBe(16_384);
    expect(() => selectTimeDomainRuntimeInputLengthV4(4_095)).toThrow(
      /at least 4096/,
    );
  });

  it('honors validSampleCount for a padded 12,160-sample capture', () => {
    const asset = loadSmokeAsset();
    const length = 16_384;
    const inPhase = new Float64Array(length);
    const quadrature = new Float64Array(length);
    for (let index = 0; index < 8_192; index++) {
      const phase = 2 * Math.PI * (
        0.071 * index + 0.0000007 * index * index
      );
      inPhase[index] = Math.cos(phase);
      quadrature[index] = Math.sin(phase);
    }
    // Poison only the padded tail.  The trained 12,160 policy must choose the
    // 8,192 prefix, so these values cannot affect preprocessing or inference.
    inPhase.fill(1e9, 8_192);
    quadrature.fill(-1e9, 8_192);
    const decision = classifyTimeDomainProfileBankV4(
      asset,
      inPhase,
      quadrature,
      { validSampleCount: 12_160, prototypeSource: 'current' },
    );
    expect(decision.runtimeInputLength).toBe(8_192);
    expect(decision.validSampleCount).toBe(12_160);
    expect(decision.preprocess.context.uses_frequency_transform).toBe(false);
    expect(decision.squaredPublicClassDistances).toHaveLength(7);
    expect(decision.publicClassLabels).toEqual(
      TIME_DOMAIN_PUBLIC_CLASSES_V4,
    );
    expect(decision.closedLabel).toBe(
      TIME_DOMAIN_PUBLIC_CLASSES_V4[decision.closedWinnerIndex],
    );
    expect(decision.winningPrototypeIndex).toBeGreaterThanOrEqual(0);
    expect(decision.prototypeSource).toBe('current');
    expect(decision.supportedPublicClassMask).toEqual([
      false, true, false, true, false, true, true,
    ]);
    expect(decision.closestPrototypeIndices[0]).toBe(-1);
    expect(decision.squaredPublicClassDistances[0]).toBe(
      Number.POSITIVE_INFINITY,
    );
  });
});
