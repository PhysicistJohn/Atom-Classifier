import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { classifyInvariantRawCapture } from './invariant-classifier-runtime.js';
import {
  loadInvariantFusionAsset,
  standardizeInvariantFeatures,
} from './invariant-fusion-runtime.js';
import { preprocessInvariantPatches } from './invariant-patch-preprocess.js';

interface EndToEndCase {
  name: string;
  physical_scale: number;
  preprocess_overrides: {
    force_center?: number;
    force_bw?: number;
    force_resample_frac?: number;
  };
  raw: {
    length: number;
    in_phase: number[];
    quadrature: number[];
  };
  expected: {
    context: Record<string, unknown>;
    packed_iq: { shape: [2, number]; values: number[] };
    raw_features: number[];
    standardized_features: number[];
    real_embedding: number[];
    complex_embedding: number[];
    fused_embedding: number[];
    squared_prototype_distances: number[];
    closed_winner_index: number;
    closed_label: string;
    posterior: number[];
    rejection_components: Array<{
      branch: 'real' | 'complex';
      weight: number;
      neighbors: number;
      raw: number;
      rank: number;
    }>;
    weighted_rejection_score: number;
    abstain: boolean;
    final_label: string;
  };
}

const staging = new URL(
  '../../training/zplane_ab/v2_full_variation/artifacts/staging/'
    + 'invariant_fusion_paired_real_v3/',
  import.meta.url,
);
const asset = loadInvariantFusionAsset(
  JSON.parse(
    readFileSync(new URL('invariant-fusion-weights-v3.json', staging), 'utf8'),
  ) as unknown,
);
const fixture = JSON.parse(
  readFileSync(
    new URL('invariant-fusion-e2e-parity-v3.json', staging),
    'utf8',
  ),
) as { cases: EndToEndCase[]; parity_limit: number };

function maxError(actual: ArrayLike<number>, expected: number[]): number {
  expect(actual.length).toBe(expected.length);
  let maximum = 0;
  for (let index = 0; index < actual.length; index++) {
    maximum = Math.max(maximum, Math.abs(actual[index]! - expected[index]!));
  }
  return maximum;
}

describe('raw-capture invariant fusion runtime', () => {
  it('matches Python end to end across raw lengths, physical scales, and an FFT seam', () => {
    const errors = {
      packed: 0,
      rawFeatures: 0,
      standardizedFeatures: 0,
      realEmbedding: 0,
      complexEmbedding: 0,
      fusedEmbedding: 0,
      prototypeDistance: 0,
      posterior: 0,
      lofRaw: 0,
      lofRank: 0,
      weightedRejection: 0,
      context: 0,
      detectedCenter: 0,
      detectedBandwidth: 0,
    };
    const activeLengths: number[] = [];
    for (const entry of fixture.cases) {
      const result = classifyInvariantRawCapture(
        asset,
        Float64Array.from(entry.raw.in_phase),
        Float64Array.from(entry.raw.quadrature),
        {
          forceCenter: entry.preprocess_overrides.force_center,
          forceBw: entry.preprocess_overrides.force_bw,
          forceResampleFrac:
            entry.preprocess_overrides.force_resample_frac,
        },
      );
      const expected = entry.expected;
      const packedLength = asset.packed_length;
      errors.packed = Math.max(
        errors.packed,
        maxError(
          result.packedInPhase,
          expected.packed_iq.values.slice(0, packedLength),
        ),
        maxError(
          result.packedQuadrature,
          expected.packed_iq.values.slice(packedLength),
        ),
      );
      errors.rawFeatures = Math.max(
        errors.rawFeatures,
        maxError(result.rawFeatures, expected.raw_features),
      );
      errors.standardizedFeatures = Math.max(
        errors.standardizedFeatures,
        maxError(
          standardizeInvariantFeatures(asset, result.rawFeatures),
          expected.standardized_features,
        ),
      );
      const classification = result.classification;
      errors.realEmbedding = Math.max(
        errors.realEmbedding,
        maxError(classification.realEmbedding, expected.real_embedding),
      );
      errors.complexEmbedding = Math.max(
        errors.complexEmbedding,
        maxError(classification.complexEmbedding, expected.complex_embedding),
      );
      errors.fusedEmbedding = Math.max(
        errors.fusedEmbedding,
        maxError(classification.fusedEmbedding, expected.fused_embedding),
      );
      errors.prototypeDistance = Math.max(
        errors.prototypeDistance,
        maxError(
          classification.squaredPrototypeDistances,
          expected.squared_prototype_distances,
        ),
      );
      errors.posterior = Math.max(
        errors.posterior,
        maxError(classification.posterior, expected.posterior),
      );
      for (
        let componentIndex = 0;
        componentIndex < classification.rejectionComponents.length;
        componentIndex++
      ) {
        const actual = classification.rejectionComponents[componentIndex]!;
        const component = expected.rejection_components[componentIndex]!;
        expect(actual.branch).toBe(component.branch);
        expect(actual.weight).toBe(component.weight);
        expect(actual.neighbors).toBe(component.neighbors);
        errors.lofRaw = Math.max(
          errors.lofRaw,
          Math.abs(actual.raw - component.raw),
        );
        errors.lofRank = Math.max(
          errors.lofRank,
          Math.abs(actual.rank - component.rank),
        );
      }
      errors.weightedRejection = Math.max(
        errors.weightedRejection,
        Math.abs(
          classification.rejectionScore
            - expected.weighted_rejection_score,
        ),
      );
      expect(classification.closedWinnerIndex).toBe(
        expected.closed_winner_index,
      );
      expect(classification.closedLabel).toBe(expected.closed_label);
      expect(classification.isUnknown).toBe(expected.abstain);
      expect(classification.label).toBe(expected.final_label);

      const exactContextKeys = [
        'version',
        'estimator_version',
        'raw_length',
        'nfft',
        'active_u_end',
        'active_length',
        'patch_length',
        'patch_count',
        'patch_starts',
        'patches_repeated_from_short_active',
        'active_repeat_count',
        'padding_samples',
        'packed_length',
        'feature_source',
      ] as const;
      for (const key of exactContextKeys) {
        expect(result.preprocessing[key]).toEqual(expected.context[key]);
      }
      const numericContextKeys = [
        'raw_peak',
        'center',
        'bw',
        'resample_frac',
        'target_frac',
        'u_step',
        'captured_u_end',
        'active_rms_before_normalization',
      ] as const;
      const detectedCenter = result.preprocessing.detected_center;
      const expectedDetectedCenter = expected.context.detected_center;
      if (detectedCenter === null || expectedDetectedCenter === null) {
        expect(detectedCenter).toBe(expectedDetectedCenter);
      } else {
        errors.detectedCenter = Math.max(
          errors.detectedCenter,
          Math.abs(detectedCenter - (expectedDetectedCenter as number)),
        );
      }
      const detectedBandwidth = result.preprocessing.detected_bw;
      const expectedDetectedBandwidth = expected.context.detected_bw;
      if (detectedBandwidth === null || expectedDetectedBandwidth === null) {
        expect(detectedBandwidth).toBe(expectedDetectedBandwidth);
      } else {
        errors.detectedBandwidth = Math.max(
          errors.detectedBandwidth,
          Math.abs(
            detectedBandwidth - (expectedDetectedBandwidth as number),
          ),
        );
      }
      for (const key of numericContextKeys) {
        const actual = result.preprocessing[key];
        const wanted = expected.context[key];
        if (actual === null || wanted === null) {
          expect(actual).toBe(wanted);
        } else {
          errors.context = Math.max(
            errors.context,
            Math.abs(actual - (wanted as number)),
          );
        }
      }
      if (
        entry.name === 'physical-scale-0p5'
        || entry.name === 'physical-scale-1p0'
        || entry.name === 'physical-scale-2p0'
      ) {
        activeLengths.push(result.preprocessing.active_length);
      }
    }
    expect(fixture.cases.map((entry) => entry.raw.length)).toEqual([
      32768,
      16384,
      8192,
      4096,
      2048,
      4096,
      51,
      777,
    ]);
    expect(Math.max(...activeLengths) / Math.min(...activeLengths)).toBeLessThan(
      1.1,
    );
    for (const error of Object.values(errors)) {
      expect(error).toBeLessThan(fixture.parity_limit);
    }
  });

  it('fails closed and deterministically repeats a genuinely short active sequence', () => {
    const zero = new Float64Array(128);
    expect(() => preprocessInvariantPatches(zero, zero)).toThrow(/zero magnitude/);

    const inPhase = new Float64Array(128);
    const quadrature = new Float64Array(128);
    for (let index = 0; index < inPhase.length; index++) {
      inPhase[index] = Math.cos(2 * Math.PI * 0.1 * index);
      quadrature[index] = Math.sin(2 * Math.PI * 0.1 * index);
    }
    const result = preprocessInvariantPatches(inPhase, quadrature, {
      forceCenter: 0.1,
      forceBw: 0.01,
      forceResampleFrac: 0.01,
    });
    expect(result.context.active_length).toBe(3);
    expect(result.context.patches_repeated_from_short_active).toBe(true);
    expect(result.context.active_repeat_count).toBe(22);
    expect(result.context.patch_starts).toEqual(new Array(16).fill(0));
    for (let patch = 1; patch < 16; patch++) {
      expect(
        Array.from(result.packedInPhase.slice(patch * 64, (patch + 1) * 64)),
      ).toEqual(Array.from(result.packedInPhase.slice(0, 64)));
    }
  });

  it('rejects preprocessing ABI drift before touching a raw capture', () => {
    const preprocess = asset.preprocess as Record<string, unknown>;
    const badNfft = {
      ...asset,
      preprocess: { ...preprocess, nfft: 1024 },
    };
    const signal = Float64Array.of(1, 1);
    const zero = new Float64Array(2);
    expect(() =>
      classifyInvariantRawCapture(badNfft, signal, zero),
    ).toThrow(/incompatible with the model geometry/);

    const featureNames = preprocess.feature_names as string[];
    const swappedNames = featureNames.slice();
    [swappedNames[0], swappedNames[1]] = [swappedNames[1]!, swappedNames[0]!];
    const badFeatureOrder = {
      ...asset,
      preprocess: { ...preprocess, feature_names: swappedNames },
    };
    expect(() =>
      classifyInvariantRawCapture(badFeatureOrder, signal, zero),
    ).toThrow(/incompatible with the model geometry/);
  });
});
