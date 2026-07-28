import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  TimeDomainClassifierV3,
  createTimeDomainClassifierV3,
  fuseTimeDomainEmbeddings,
  loadTimeDomainClassifierAssetV3,
  loadTimeDomainEncodersV3,
  nearestTimeDomainPrototype,
  standardizeTimeDomainFeatures,
  type TimeDomainEncoderPairV3,
} from './time-domain-classifier-v3.js';
import {
  loadTimeDomainOpenSetAssetV3,
} from './time-domain-openset-v3.js';

interface FixtureRow {
  family: 'known' | 'noise' | 'chirp' | 'probe';
  name: string;
  capture_length: number;
  iq: { in_phase: number[]; quadrature: number[] };
  stage_one: { gated: boolean };
  raw_features: number[];
  standardized_features: number[];
  stage_two: {
    real_embedding: number[];
    complex_embedding: number[];
    fused_embedding: number[];
    predicted_class_index: number;
    predicted_class_label: string;
    squared_prototype_distances: number[];
    score: number;
  } | null;
  stage_one_survivor_rank: number | null;
  composite_score: number | null;
  staged_threshold: number;
  staged_score: number;
  rejected_stage: 1 | 2 | null;
  decision_label: string;
}

interface ProbeCase {
  name: string;
  length: number;
  raw: { in_phase: number[]; quadrature: number[] };
  expected: {
    raw_features: number[];
    standardized_features: number[];
    real_embedding: number[];
    complex_embedding: number[];
    fused_embedding: number[];
    squared_prototype_distances: number[];
    closed_winner_index: number;
    closed_label: string;
  };
}

interface ParityFixture {
  classes: string[];
  frontend: { patch_length: number; patch_count: number; target_frac: number };
  rows: FixtureRow[];
  probe_cases: ProbeCase[];
}

// Tests consume the tracked runtime package, never mutable gitignored training
// output. The compact fixture preserves both fitted lengths and all decisions.
const staging = new URL('./assets-v3-staging/', import.meta.url);
const classifierAssetRaw = JSON.parse(
  readFileSync(new URL('time-domain-classifier-weights-v1.json', staging), 'utf8'),
) as unknown;
const opensetAssetRaw = JSON.parse(
  readFileSync(new URL('time-domain-openset-weights-v1.json', staging), 'utf8'),
) as unknown;
const encoderAssetRaw = JSON.parse(
  readFileSync(
    new URL('time-domain-fusion-weights-v3.json', staging),
    'utf8',
  ),
) as unknown;
const fixture = JSON.parse(
  readFileSync(new URL('time-domain-openset-smoke-v1.json', staging), 'utf8'),
) as ParityFixture;

const classifierAsset = loadTimeDomainClassifierAssetV3(classifierAssetRaw);
const opensetAsset = loadTimeDomainOpenSetAssetV3(opensetAssetRaw);

function maxAbsDelta(actual: ArrayLike<number>, expected: number[]): number {
  expect(actual.length).toBe(expected.length);
  let worst = 0;
  for (let index = 0; index < expected.length; index++) {
    worst = Math.max(worst, Math.abs(actual[index]! - expected[index]!));
  }
  return worst;
}

/** Encoders that replay the fixture's Python branch embeddings for one row. */
function fixtureEncoders(row: FixtureRow): TimeDomainEncoderPairV3 {
  if (row.stage_two === null) {
    const refuse = (): Float64Array => {
      throw new Error(
        `encoder was invoked for gated row ${row.name}; the stage-1 short `
        + 'circuit must prevent any encoder work',
      );
    };
    return { real: refuse, complex: refuse };
  }
  const expected = row.stage_two;
  return {
    real: () => Float64Array.from(expected.real_embedding),
    complex: () => Float64Array.from(expected.complex_embedding),
  };
}

describe('classifier asset', () => {
  it('validates and matches the openset asset geometry', () => {
    expect(classifierAsset.classification.classes).toEqual(fixture.classes);
    expect(classifierAsset.frontend.patch_length).toBe(
      opensetAsset.frontend.patch_length,
    );
    expect(classifierAsset.frontend.uses_frequency_transform).toBe(false);
  });

  it('refuses the explicitly incompatible schema ids', () => {
    for (const schema of [
      'atomos.invariant-fusion.paired-real',
      'hybrid-v3',
      'invariant-patch-v1',
    ]) {
      expect(() =>
        loadTimeDomainClassifierAssetV3({ ...(classifierAssetRaw as object), schema }))
        .toThrow(/incompatible|unsupported/);
    }
  });

  it('admits a coherent release-status asset set in production mode', async () => {
    const classifierRelease = {
      ...(classifierAssetRaw as Record<string, unknown>),
      status: 'release',
    };
    const opensetRelease = {
      ...(opensetAssetRaw as Record<string, unknown>),
      status: 'release',
    };
    const encoderRelease = {
      ...(encoderAssetRaw as Record<string, unknown>),
      status: 'release',
    };
    const classifier = await createTimeDomainClassifierV3({
      classifierAsset: classifierRelease,
      opensetAsset: opensetRelease,
      encoderAsset: encoderRelease,
      admission: 'production',
      encoders: {
        real: () => new Float64Array(32),
        complex: () => new Float64Array(32),
      },
    });
    expect(classifier.asset.status).toBe('release');
    expect(classifier.openSet.asset.status).toBe('release');
  });

  it('production admission fails closed on the tracked staging package', async () => {
    await expect(createTimeDomainClassifierV3({
      classifierAsset: classifierAssetRaw,
      opensetAsset: opensetAssetRaw,
      encoderAsset: encoderAssetRaw,
      admission: 'production',
    })).rejects.toThrow(/must be release for production admission/);
  });

  it('refuses a mixed candidate before any encoder can run', async () => {
    const raw = classifierAssetRaw as {
      fusion: { weight_real: number };
    };
    const mismatched = {
      ...(classifierAssetRaw as Record<string, unknown>),
      fusion: {
        ...raw.fusion,
        weight_real: raw.fusion.weight_real + 0.01,
      },
    };
    await expect(createTimeDomainClassifierV3({
      classifierAsset: mismatched,
      opensetAsset: opensetAssetRaw,
      encoderAsset: encoderAssetRaw,
      encoders: {
        real: () => {
          throw new Error('must not run');
        },
        complex: () => {
          throw new Error('must not run');
        },
      },
    })).rejects.toThrow(/fusion\.weight_real differs/);
  });
});

describe('fusion and nearest-prototype parity against the bundle probe fixture', () => {
  it('reproduces standardization, fusion, and the closed label for every case', () => {
    for (const probe of fixture.probe_cases) {
      const standardized = standardizeTimeDomainFeatures(
        classifierAsset,
        probe.expected.raw_features,
      );
      expect(
        maxAbsDelta(standardized, probe.expected.standardized_features),
        `${probe.name} standardized features`,
      ).toBeLessThanOrEqual(1e-6);
      const fused = fuseTimeDomainEmbeddings(
        classifierAsset,
        probe.expected.real_embedding,
        probe.expected.complex_embedding,
      );
      expect(
        maxAbsDelta(fused, probe.expected.fused_embedding),
        `${probe.name} fused embedding`,
      ).toBeLessThanOrEqual(1e-6);
      const nearest = nearestTimeDomainPrototype(classifierAsset, fused);
      expect(nearest.winnerIndex, `${probe.name} winner`).toBe(
        probe.expected.closed_winner_index,
      );
      expect(nearest.label).toBe(probe.expected.closed_label);
      expect(
        maxAbsDelta(
          nearest.squaredDistances,
          probe.expected.squared_prototype_distances,
        ),
        `${probe.name} prototype distances`,
      ).toBeLessThanOrEqual(1e-5);
    }
  });
});

describe('full staged classification', () => {
  it('reproduces every fixture decision, including which stage rejected', () => {
    for (const row of fixture.rows) {
      const classifier = new TimeDomainClassifierV3({
        classifierAsset,
        opensetAsset,
        encoders: fixtureEncoders(row),
      });
      const decision = classifier.classify(row.iq.in_phase, row.iq.quadrature);
      expect(decision.rejectedStage, `${row.name} rejected stage`).toBe(
        row.rejected_stage,
      );
      expect(decision.label, `${row.name} label`).toBe(row.decision_label);
      if (row.rejected_stage === 1) {
        // Gated: no closed label exists and nothing downstream ran.
        expect(decision.closedLabel).toBeNull();
        expect(decision.forward).toBeNull();
        expect(decision.openSet.stageTwo).toBeNull();
        expect(decision.label).toBe('noise');
      } else {
        const expected = row.stage_two!;
        // The closed label is never altered by rejection.
        expect(decision.closedLabel).toBe(expected.predicted_class_label);
        expect(decision.closedWinnerIndex).toBe(expected.predicted_class_index);
        expect(
          maxAbsDelta(
            decision.forward!.standardizedFeatures,
            row.standardized_features,
          ),
          `${row.name} standardized features`,
        ).toBeLessThanOrEqual(1e-6);
        expect(
          maxAbsDelta(decision.forward!.fusedEmbedding, expected.fused_embedding),
          `${row.name} fused embedding`,
        ).toBeLessThanOrEqual(1e-6);
        if (row.rejected_stage === 2) {
          expect(decision.label).toBe('unknown');
        } else {
          expect(decision.label).toBe(decision.closedLabel);
        }
      }
      // The architecture contract keys ride on every decision.
      expect(decision.contract.additive_only).toBe(false);
      expect(decision.contract.changes_closed_label).toBe(true);
      expect(decision.contract.gates_before_classification).toBe(true);
      expect(decision.openSet.contract).toEqual(decision.contract);
    }
  });

  it('never touches the encoders for gated rows', () => {
    const gatedRow = fixture.rows.find((row) => row.rejected_stage === 1)!;
    let encoderCalls = 0;
    const classifier = new TimeDomainClassifierV3({
      classifierAsset,
      opensetAsset,
      encoders: {
        real: () => {
          encoderCalls += 1;
          return new Float64Array(32);
        },
        complex: () => {
          encoderCalls += 1;
          return new Float64Array(32);
        },
      },
    });
    const decision = classifier.classify(
      gatedRow.iq.in_phase,
      gatedRow.iq.quadrature,
    );
    expect(decision.rejectedStage).toBe(1);
    expect(encoderCalls).toBe(0);
  });

  it('fails loudly for a capture length without a fitted prefilter', () => {
    const shortProbe = fixture.probe_cases.find((probe) => probe.length === 2048);
    expect(shortProbe).toBeDefined();
    const classifier = new TimeDomainClassifierV3({
      classifierAsset,
      opensetAsset,
      encoders: {
        real: () => new Float64Array(32),
        complex: () => new Float64Array(32),
      },
    });
    expect(() =>
      classifier.classify(
        shortProbe!.raw.in_phase,
        shortProbe!.raw.quadrature,
      )).toThrow(/no fitted noise prefilter/);
  });
});

describe('sibling encoder module contract', () => {
  it('loads the sibling module and returns working branch encoders', async () => {
    // The sibling port has landed, so this is now a genuine integration test:
    // the factory must accept the real staging asset and the returned closures
    // must produce unit-norm 32-d embeddings on real fixture input.
    const encoders = await loadTimeDomainEncodersV3(encoderAssetRaw);
    expect(typeof encoders.real).toBe('function');
    expect(typeof encoders.complex).toBe('function');
    const fixture = JSON.parse(
      readFileSync(
        new URL('assets-v3-staging/time-domain-probe-fixture-v3.json', import.meta.url),
        'utf8',
      ),
    ) as {
      cases: Array<{
        expected: { packed_iq: number[]; standardized_features: number[] };
      }>;
    };
    const probe = fixture.cases[0]!.expected;
    const packedLength = probe.packed_iq.length / 2;
    for (const branch of [encoders.real, encoders.complex]) {
      const embedding = branch(
        probe.packed_iq.slice(0, packedLength),
        probe.packed_iq.slice(packedLength),
        Float64Array.from(probe.standardized_features),
      );
      expect(embedding).toHaveLength(32);
      let norm = 0;
      for (const value of embedding) norm += value * value;
      expect(Math.sqrt(norm)).toBeCloseTo(1, 9);
    }
  });

  it('rejects an invalid encoder asset loudly', async () => {
    await expect(loadTimeDomainEncodersV3(null)).rejects.toThrow(
      /requires the v3 fusion weights asset/,
    );
  });

  it('createTimeDomainClassifierV3 accepts injected decision-layer encoders', async () => {
    const classifier = await createTimeDomainClassifierV3({
      classifierAsset: classifierAssetRaw,
      opensetAsset: opensetAssetRaw,
      encoderAsset: encoderAssetRaw,
      encoders: {
        real: () => new Float64Array(32),
        complex: () => new Float64Array(32),
      },
    });
    expect(classifier).toBeInstanceOf(TimeDomainClassifierV3);
  });
});
