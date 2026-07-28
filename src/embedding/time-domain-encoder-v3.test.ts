import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  forwardTimeDomainComplexBranchV3,
  forwardTimeDomainRealBranchV3,
} from './time-domain-encoder-v3.js';
import { loadTimeDomainFusionAssetV3 } from './time-domain-fusion-v3.js';

/**
 * Structural properties of the v3 patch-CNN port, checked with the real
 * exported weights: patch-set permutation invariance, global-phase invariance
 * of the complex branch, unit output norm, and loader rejection of v2/hybrid
 * assets. Numerical parity against Python lives in
 * `time-domain-fusion-v3.test.ts`.
 */

const staging = new URL('./assets-v3-staging/', import.meta.url);
const asset = loadTimeDomainFusionAssetV3(
  JSON.parse(
    readFileSync(new URL('time-domain-fusion-weights-v3.json', staging), 'utf8'),
  ),
);

function deterministicInput(): {
  inPhase: Float64Array;
  quadrature: Float64Array;
  features: Float64Array;
} {
  const packedLength = asset.packed_length;
  const inPhase = new Float64Array(packedLength);
  const quadrature = new Float64Array(packedLength);
  for (let index = 0; index < packedLength; index++) {
    inPhase[index] = Math.sin(0.37 * index) + 0.21 * Math.cos(1.7 * index);
    quadrature[index] = Math.cos(0.53 * index) - 0.13 * Math.sin(2.3 * index);
  }
  const features = new Float64Array(asset.real.config.n_features);
  for (let index = 0; index < features.length; index++) {
    features[index] = Math.sin(1.1 * index + 0.4);
  }
  return { inPhase, quadrature, features };
}

function permutePatches(
  packed: Float64Array,
  patchLength: number,
  order: number[],
): Float64Array {
  const output = new Float64Array(packed.length);
  for (let target = 0; target < order.length; target++) {
    const source = order[target]!;
    for (let index = 0; index < patchLength; index++) {
      output[target * patchLength + index] = packed[source * patchLength + index]!;
    }
  }
  return output;
}

describe('time-domain v3 encoder invariances', () => {
  const { inPhase, quadrature, features } = deterministicInput();
  const { patch_length: patchLength, patch_count: patchCount } =
    asset.real.config;
  // A fixed derangement of the 16 patches.
  const order = Array.from(
    { length: patchCount },
    (_, index) => (index + 5) % patchCount,
  );

  it('produces unit-norm embeddings from both branches', () => {
    for (const embedding of [
      forwardTimeDomainRealBranchV3(asset.real, inPhase, quadrature, features),
      forwardTimeDomainComplexBranchV3(
        asset.complex,
        inPhase,
        quadrature,
        features,
      ),
    ]) {
      let norm = 0;
      for (const value of embedding) norm += value * value;
      expect(Math.abs(Math.sqrt(norm) - 1)).toBeLessThan(1e-12);
    }
  });

  it('is exactly invariant to patch permutation up to float64 rounding', () => {
    const permutedInPhase = permutePatches(inPhase, patchLength, order);
    const permutedQuadrature = permutePatches(quadrature, patchLength, order);
    const real = forwardTimeDomainRealBranchV3(
      asset.real,
      inPhase,
      quadrature,
      features,
    );
    const realPermuted = forwardTimeDomainRealBranchV3(
      asset.real,
      permutedInPhase,
      permutedQuadrature,
      features,
    );
    const complex = forwardTimeDomainComplexBranchV3(
      asset.complex,
      inPhase,
      quadrature,
      features,
    );
    const complexPermuted = forwardTimeDomainComplexBranchV3(
      asset.complex,
      permutedInPhase,
      permutedQuadrature,
      features,
    );
    for (let index = 0; index < real.length; index++) {
      expect(Math.abs(real[index]! - realPermuted[index]!)).toBeLessThan(1e-12);
      expect(Math.abs(complex[index]! - complexPermuted[index]!)).toBeLessThan(
        1e-12,
      );
    }
  });

  it('keeps the complex branch invariant to a global phase rotation', () => {
    const phase = 1.234;
    const cos = Math.cos(phase);
    const sin = Math.sin(phase);
    const rotatedInPhase = new Float64Array(inPhase.length);
    const rotatedQuadrature = new Float64Array(inPhase.length);
    for (let index = 0; index < inPhase.length; index++) {
      rotatedInPhase[index] = cos * inPhase[index]! - sin * quadrature[index]!;
      rotatedQuadrature[index] = sin * inPhase[index]! + cos * quadrature[index]!;
    }
    const embedding = forwardTimeDomainComplexBranchV3(
      asset.complex,
      inPhase,
      quadrature,
      features,
    );
    const rotated = forwardTimeDomainComplexBranchV3(
      asset.complex,
      rotatedInPhase,
      rotatedQuadrature,
      features,
    );
    for (let index = 0; index < embedding.length; index++) {
      expect(Math.abs(embedding[index]! - rotated[index]!)).toBeLessThan(1e-9);
    }
  });
});

describe('time-domain v3 asset loader hygiene', () => {
  it('rejects the v2 paired-real schema by name', () => {
    expect(() =>
      loadTimeDomainFusionAssetV3({
        schema: 'atomos.invariant-fusion.paired-real',
        schema_version: 1,
      })).toThrow(/v2\/hybrid contract/);
  });

  it('rejects every schema id the bundle declares incompatible', () => {
    for (const schema of [
      'hybrid-v3',
      'invariant-centered-fusion-with-lof-rank-ensemble',
      'invariant-patch-v1',
    ]) {
      expect(() =>
        loadTimeDomainFusionAssetV3({ schema, schema_version: 1 })).toThrow(
        /v2\/hybrid contract/,
      );
    }
  });

  it('rejects malformed geometry', () => {
    const raw = JSON.parse(
      readFileSync(
        new URL('time-domain-fusion-weights-v3.json', staging),
        'utf8',
      ),
    ) as { packed_length: number };
    raw.packed_length = 512;
    expect(() => loadTimeDomainFusionAssetV3(raw)).toThrow(
      /packed_length does not match/,
    );
  });

  it('validates the shipped staging asset geometry', () => {
    expect(asset.packed_length).toBe(1024);
    expect(asset.real.config.embed_dim).toBe(32);
    expect(asset.classification.classes).toEqual([
      'am',
      'bluetooth',
      'cw',
      'dsss',
      'fm',
      'gsm',
      'ofdm',
    ]);
    expect(asset.classification.prototypes).toHaveLength(7);
    expect(asset.classification.prototypes[0]).toHaveLength(64);
  });
});
