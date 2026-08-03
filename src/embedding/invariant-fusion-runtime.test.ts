import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  classifyInvariantFusion,
  forwardInvariantFusion,
  loadInvariantFusionAsset,
  type InvariantFusionAsset,
} from './invariant-fusion-runtime.js';

interface ParityFixture {
  case_names: string[];
  input: {
    packed_iq: {
      shape: [number, 2, number];
      values: number[];
    };
    raw_features: {
      shape: [number, number];
      values: number[];
    };
  };
  expected: {
    standardized_features: number[][];
    real_embedding: number[][];
    complex_embedding: number[][];
    fused_embedding: number[][];
    squared_prototype_distances: number[][];
    closed_winner_index: number[];
    closed_label: string[];
    posterior: number[][];
    rejection_components: Array<{
      branch: 'real' | 'complex';
      weight: number;
      neighbors: number;
      raw: number[];
      rank: number[];
    }>;
    weighted_rejection_score: number[];
    abstain: boolean[];
    final_label: string[];
  };
}

const staging = process.env.ATOMOS_INVARIANT_FUSION_STAGING_URL
  ? new URL(process.env.ATOMOS_INVARIANT_FUSION_STAGING_URL)
  : new URL(
    '../../training/zplane_ab/v2_full_variation/artifacts/staging/'
      + 'invariant_fusion_paired_real_v3/',
    import.meta.url,
  );
// Git-ignored training outputs; absent in a fresh clone (CI). Skip, don't fail.
const stagingAvailable = existsSync(
  new URL('invariant-fusion-weights-v3.json', staging),
);
const rawAsset = stagingAvailable
  ? JSON.parse(
    readFileSync(new URL('invariant-fusion-weights-v3.json', staging), 'utf8'),
  ) as unknown
  : null;
const fixture = (stagingAvailable
  ? JSON.parse(
    readFileSync(new URL('invariant-fusion-parity-v3.json', staging), 'utf8'),
  )
  : { rows: [] }) as ParityFixture;
const asset = stagingAvailable
  ? loadInvariantFusionAsset(rawAsset)
  : (null as never);

function inputs(row: number): {
  inPhase: number[];
  quadrature: number[];
  rawFeatures: number[];
} {
  const [batch, channels, length] = fixture.input.packed_iq.shape;
  expect(row).toBeLessThan(batch);
  const iqOffset = row * channels * length;
  const featureWidth = fixture.input.raw_features.shape[1];
  return {
    inPhase: fixture.input.packed_iq.values.slice(iqOffset, iqOffset + length),
    quadrature: fixture.input.packed_iq.values.slice(
      iqOffset + length,
      iqOffset + 2 * length,
    ),
    rawFeatures: fixture.input.raw_features.values.slice(
      row * featureWidth,
      (row + 1) * featureWidth,
    ),
  };
}

function maxError(actual: ArrayLike<number>, expected: number[]): number {
  expect(actual.length).toBe(expected.length);
  let maximum = 0;
  for (let index = 0; index < actual.length; index++) {
    maximum = Math.max(maximum, Math.abs(actual[index]! - expected[index]!));
  }
  return maximum;
}

describe.skipIf(!stagingAvailable)('invariant paired-real TypeScript runtime', () => {
  it('matches Python branch, fusion, classifier, and weighted-LOF fixtures', () => {
    let modelError = 0;
    let classifierError = 0;
    let lofRawError = 0;
    let lofRankError = 0;
    for (let row = 0; row < fixture.case_names.length; row++) {
      const input = inputs(row);
      const forward = forwardInvariantFusion(
        asset,
        input.inPhase,
        input.quadrature,
        input.rawFeatures,
      );
      const result = classifyInvariantFusion(
        asset,
        input.inPhase,
        input.quadrature,
        input.rawFeatures,
      );
      modelError = Math.max(
        modelError,
        maxError(
          forward.standardizedFeatures,
          fixture.expected.standardized_features[row]!,
        ),
        maxError(result.realEmbedding, fixture.expected.real_embedding[row]!),
        maxError(
          result.complexEmbedding,
          fixture.expected.complex_embedding[row]!,
        ),
        maxError(result.fusedEmbedding, fixture.expected.fused_embedding[row]!),
      );
      classifierError = Math.max(
        classifierError,
        maxError(
          result.squaredPrototypeDistances,
          fixture.expected.squared_prototype_distances[row]!,
        ),
        maxError(result.posterior, fixture.expected.posterior[row]!),
        Math.abs(
          result.rejectionScore
            - fixture.expected.weighted_rejection_score[row]!,
        ),
      );
      expect(result.closedWinnerIndex).toBe(
        fixture.expected.closed_winner_index[row],
      );
      expect(result.closedLabel).toBe(fixture.expected.closed_label[row]);
      expect(result.isUnknown).toBe(fixture.expected.abstain[row]);
      expect(result.label).toBe(fixture.expected.final_label[row]);
      for (
        let componentIndex = 0;
        componentIndex < result.rejectionComponents.length;
        componentIndex++
      ) {
        const actual = result.rejectionComponents[componentIndex]!;
        const expected = fixture.expected.rejection_components[componentIndex]!;
        expect(actual.branch).toBe(expected.branch);
        expect(actual.neighbors).toBe(expected.neighbors);
        expect(actual.weight).toBe(expected.weight);
        lofRawError = Math.max(
          lofRawError,
          Math.abs(actual.raw - expected.raw[row]!),
        );
        lofRankError = Math.max(
          lofRankError,
          Math.abs(actual.rank - expected.rank[row]!),
        );
      }
    }
    expect(modelError).toBeLessThan(1e-4);
    expect(classifierError).toBeLessThan(1e-4);
    expect(lofRawError).toBeLessThan(1e-4);
    expect(lofRankError).toBeLessThan(1e-4);
  });

  it('rejects malformed assets and non-finite capture values', () => {
    const malformed = structuredClone(rawAsset) as {
      rejection: { components: Array<{ weight: number }> };
    };
    malformed.rejection.components[0]!.weight = 0.1;
    expect(() => loadInvariantFusionAsset(malformed)).toThrow(/sum to one/);

    const input = inputs(0);
    input.inPhase[3] = Number.NaN;
    expect(() =>
      forwardInvariantFusion(
        asset,
        input.inPhase,
        input.quadrature,
        input.rawFeatures,
      )).toThrow(/finite/);
  });

  it('keeps the fused closed winner immutable before optional abstention', () => {
    const input = inputs(0);
    const result = classifyInvariantFusion(
      asset,
      input.inPhase,
      input.quadrature,
      input.rawFeatures,
    );
    const noAbstention = structuredClone(asset) as InvariantFusionAsset;
    noAbstention.rejection.threshold = 0.999999;
    const accepted = classifyInvariantFusion(
      noAbstention,
      input.inPhase,
      input.quadrature,
      input.rawFeatures,
    );
    expect(accepted.closedWinnerIndex).toBe(result.closedWinnerIndex);
    expect(accepted.closedLabel).toBe(result.closedLabel);
    expect(accepted.label).toBe(accepted.closedLabel);
  });
});
