import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { preprocessTimeDomainInvariantPatches } from './time-domain-invariant-patch-preprocess-v3.js';
import {
  classifyTimeDomainFusionV3,
  loadTimeDomainFusionAssetV3,
  standardizeTimeDomainFeaturesV3,
} from './time-domain-fusion-v3.js';

/**
 * Per-stage parity against the v3 runtime bundle's Python-generated probe
 * fixture (self-verified in Python at max_abs_error 0.0). The staging copy is
 * byte-identical to the bundle's `probe_fixture.json`; its SHA-256 is
 * asserted below against the export manifest so the anchor cannot drift.
 *
 * TypeScript computes in float64 while the torch reference computed in
 * float32, so intermediate accumulations differ by float32 rounding noise;
 * every stage must still land within the fixture's own stated tolerance.
 */

interface ProbeCase {
  name: string;
  length: number;
  physical_scale: number;
  kind: string;
  raw: { in_phase: number[]; quadrature: number[] };
  expected: {
    context: { patch_length: number; patch_count: number; target_frac: number };
    packed_iq: number[];
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

interface ProbeFixture {
  schema: string;
  schema_version: number;
  tolerance: number;
  cases: ProbeCase[];
}

interface ExportManifest {
  emitted: Record<string, { sha256: string; bytes: number }>;
  probe_fixture_is_byte_identical_to_bundle: boolean;
}

const staging = new URL('./assets-v3-staging/', import.meta.url);
const fixtureBytes = readFileSync(
  new URL('time-domain-probe-fixture-v3.json', staging),
);
const fixture = JSON.parse(fixtureBytes.toString('utf8')) as ProbeFixture;
const exportManifest = JSON.parse(
  readFileSync(new URL('export-manifest.json', staging), 'utf8'),
) as ExportManifest;
const asset = loadTimeDomainFusionAssetV3(
  JSON.parse(
    readFileSync(new URL('time-domain-fusion-weights-v3.json', staging), 'utf8'),
  ),
);

const TOLERANCE = fixture.tolerance;

function maxAbsError(actual: ArrayLike<number>, expected: number[]): number {
  expect(actual.length).toBe(expected.length);
  let worst = 0;
  for (let index = 0; index < expected.length; index++) {
    worst = Math.max(worst, Math.abs(actual[index]! - expected[index]!));
  }
  return worst;
}

const stageWorst = new Map<string, number>();

function recordStage(stage: string, error: number): void {
  stageWorst.set(stage, Math.max(stageWorst.get(stage) ?? 0, error));
}

describe('time-domain v3 probe fixture provenance', () => {
  it('is the schema-tagged v3 fixture with a stated tolerance', () => {
    expect(fixture.schema).toBe(
      'atomos.v3.time-domain-invariant-fusion.runtime-bundle.probe-fixture',
    );
    expect(fixture.schema_version).toBe(1);
    expect(fixture.tolerance).toBeGreaterThan(0);
    expect(fixture.cases).toHaveLength(4);
  });

  it('is byte-identical to the SHA recorded by the exporter', () => {
    const digest = createHash('sha256').update(fixtureBytes).digest('hex');
    expect(exportManifest.probe_fixture_is_byte_identical_to_bundle).toBe(true);
    expect(digest).toBe(
      exportManifest.emitted['time-domain-probe-fixture-v3.json']!.sha256,
    );
  });
});

describe('time-domain v3 end-to-end parity, every stage, every case', () => {
  for (const probeCase of fixture.cases) {
    describe(probeCase.name, () => {
      const packedLength = asset.packed_length;
      const result = preprocessTimeDomainInvariantPatches(
        probeCase.raw.in_phase,
        probeCase.raw.quadrature,
        {
          patchLength: probeCase.expected.context.patch_length,
          patchCount: probeCase.expected.context.patch_count,
          targetFrac: probeCase.expected.context.target_frac,
        },
      );
      const classification = classifyTimeDomainFusionV3(
        asset,
        result.packedInPhase,
        result.packedQuadrature,
        result.rawFeatures,
      );

      it('packed', () => {
        const error = Math.max(
          maxAbsError(
            result.packedInPhase,
            probeCase.expected.packed_iq.slice(0, packedLength),
          ),
          maxAbsError(
            result.packedQuadrature,
            probeCase.expected.packed_iq.slice(packedLength),
          ),
        );
        recordStage('packed', error);
        expect(error).toBeLessThanOrEqual(TOLERANCE);
      });

      it('raw_features', () => {
        const error = maxAbsError(
          result.rawFeatures,
          probeCase.expected.raw_features,
        );
        recordStage('raw_features', error);
        expect(error).toBeLessThanOrEqual(TOLERANCE);
      });

      it('standardized_features', () => {
        const error = maxAbsError(
          classification.standardizedFeatures,
          probeCase.expected.standardized_features,
        );
        recordStage('standardized_features', error);
        expect(error).toBeLessThanOrEqual(TOLERANCE);
      });

      it('real_embedding', () => {
        const error = maxAbsError(
          classification.realEmbedding,
          probeCase.expected.real_embedding,
        );
        recordStage('real_embedding', error);
        expect(error).toBeLessThanOrEqual(TOLERANCE);
      });

      it('complex_embedding', () => {
        const error = maxAbsError(
          classification.complexEmbedding,
          probeCase.expected.complex_embedding,
        );
        recordStage('complex_embedding', error);
        expect(error).toBeLessThanOrEqual(TOLERANCE);
      });

      it('fused_embedding', () => {
        const error = maxAbsError(
          classification.fusedEmbedding,
          probeCase.expected.fused_embedding,
        );
        recordStage('fused_embedding', error);
        expect(error).toBeLessThanOrEqual(TOLERANCE);
      });

      it('squared_prototype_distances', () => {
        const error = maxAbsError(
          classification.squaredPrototypeDistances,
          probeCase.expected.squared_prototype_distances,
        );
        recordStage('squared_prototype_distances', error);
        expect(error).toBeLessThanOrEqual(TOLERANCE);
      });

      it('label', () => {
        expect(classification.closedWinnerIndex).toBe(
          probeCase.expected.closed_winner_index,
        );
        expect(classification.closedLabel).toBe(probeCase.expected.closed_label);
      });
    });
  }

  it('reports the worst per-stage error across all cases', () => {
    const summary = Object.fromEntries(
      [...stageWorst.entries()].map(([stage, error]) => [
        stage,
        error.toExponential(3),
      ]),
    );
    // eslint-disable-next-line no-console
    console.info('time-domain v3 parity worst-per-stage max_abs_error:', summary);
    for (const [, error] of stageWorst) {
      expect(error).toBeLessThanOrEqual(TOLERANCE);
    }
  });
});

describe('time-domain v3 standardization float32 replication', () => {
  it('matches the fixture when fed the fixture raw features directly', () => {
    for (const probeCase of fixture.cases) {
      const standardized = standardizeTimeDomainFeaturesV3(
        asset.feature_standardization,
        probeCase.expected.raw_features,
      );
      expect(
        maxAbsError(standardized, probeCase.expected.standardized_features),
      ).toBe(0);
    }
  });
});
