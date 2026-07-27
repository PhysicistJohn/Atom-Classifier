import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  estimateTimeDomainBand,
  minimumTimeDomainBandwidthForActiveSpan,
  timeDomainGeometryMetadata,
} from './time-domain-geometry-v3.js';
import {
  preprocessTimeDomainInvariantPatches,
  timeDomainInvariantPatchMetadata,
} from './time-domain-invariant-patch-preprocess-v3.js';

interface FixtureCase {
  name: string;
  physical_scale: number | null;
  relation: string | null;
  nuisance: {
    gain?: number;
    global_phase?: number;
    carrier_shift?: number;
  };
  options: {
    patch_length: number;
    patch_count: number;
    target_frac: number;
  };
  input: {
    in_phase: number[];
    quadrature: number[];
  };
  expected: {
    center: number;
    bandwidth: number;
    geometry_context: Record<string, unknown>;
    packed_in_phase: number[];
    packed_quadrature: number[];
    raw_features: number[];
    context: Record<string, unknown>;
  };
}

interface ParityFixture {
  fixture_version: string;
  synthetic_only: boolean;
  reads_held_or_sealed_payload: boolean;
  python_source_sha256: Record<string, string>;
  metadata: Record<string, unknown>;
  cases: FixtureCase[];
}

const fixture = JSON.parse(
  readFileSync(
    new URL(
      './test-fixtures/time-domain-invariant-preprocess-v3.json',
      import.meta.url,
    ),
    'utf8',
  ),
) as ParityFixture;

const root = new URL('../../', import.meta.url);
const pythonSources = {
  geometry: new URL('training/time_domain_geometry.py', root),
  wrapper: new URL(
    'training/time_domain_invariant_patch_preprocess.py',
    root,
  ),
  patch_core: new URL('training/invariant_patch_preprocess.py', root),
  feature_core: new URL('training/preprocess.py', root),
  generator: new URL(
    'training/generate_time_domain_preprocess_v3_fixture.py',
    root,
  ),
} as const;

function sha256(url: URL): string {
  return createHash('sha256').update(readFileSync(url)).digest('hex');
}

function maxAbsoluteError(
  actual: ArrayLike<number>,
  expected: number[],
): number {
  expect(actual.length).toBe(expected.length);
  let maximum = 0;
  for (let index = 0; index < actual.length; index++) {
    maximum = Math.max(
      maximum,
      Math.abs(actual[index]! - expected[index]!),
    );
  }
  return maximum;
}

function scaledNumericError(actual: number, expected: number): number {
  return Math.abs(actual - expected) / Math.max(1, Math.abs(expected));
}

function compareJsonContract(
  actual: unknown,
  expected: unknown,
  path: string,
): number {
  if (typeof expected === 'number') {
    expect(typeof actual, path).toBe('number');
    return scaledNumericError(actual as number, expected);
  }
  if (expected === null || typeof expected !== 'object') {
    expect(actual, path).toEqual(expected);
    return 0;
  }
  if (Array.isArray(expected)) {
    expect(Array.isArray(actual), path).toBe(true);
    const actualItems = actual as unknown[];
    expect(actualItems.length, path).toBe(expected.length);
    let maximum = 0;
    for (let index = 0; index < expected.length; index++) {
      maximum = Math.max(
        maximum,
        compareJsonContract(
          actualItems[index],
          expected[index],
          `${path}[${index}]`,
        ),
      );
    }
    return maximum;
  }
  expect(actual !== null && typeof actual === 'object', path).toBe(true);
  const actualRecord = actual as Record<string, unknown>;
  const expectedRecord = expected as Record<string, unknown>;
  expect(Object.keys(actualRecord).sort(), path).toEqual(
    Object.keys(expectedRecord).sort(),
  );
  let maximum = 0;
  for (const [key, value] of Object.entries(expectedRecord)) {
    maximum = Math.max(
      maximum,
      compareJsonContract(actualRecord[key], value, `${path}.${key}`),
    );
  }
  return maximum;
}

function byName(name: string): FixtureCase {
  const row = fixture.cases.find((entry) => entry.name === name);
  if (row === undefined) throw new Error(`missing fixture case ${name}`);
  return row;
}

describe('strict time-domain v3 Python/TypeScript parity', () => {
  it('binds the synthetic fixture to the exact Python source contract', () => {
    expect(fixture.fixture_version).toBe(
      'time-domain-invariant-preprocess-parity-v3',
    );
    expect(fixture.synthetic_only).toBe(true);
    expect(fixture.reads_held_or_sealed_payload).toBe(false);
    for (const [name, url] of Object.entries(pythonSources)) {
      expect(sha256(url), name).toBe(fixture.python_source_sha256[name]);
    }
    expect(timeDomainInvariantPatchMetadata()).toEqual(fixture.metadata);
    expect(timeDomainGeometryMetadata()).toEqual(fixture.metadata.geometry);
  });

  it('matches geometry, packed patches, features, and audit context', () => {
    const maximumErrors = {
      center: 0,
      bandwidth: 0,
      geometryContext: 0,
      packed: 0,
      features: 0,
      preprocessContext: 0,
    };
    for (const entry of fixture.cases) {
      const inPhase = Float64Array.from(entry.input.in_phase);
      const quadrature = Float64Array.from(entry.input.quadrature);
      const options = {
        patchLength: entry.options.patch_length,
        patchCount: entry.options.patch_count,
        targetFrac: entry.options.target_frac,
      };
      const minimumBandwidth = minimumTimeDomainBandwidthForActiveSpan(
        inPhase.length,
        options.patchLength,
        options.targetFrac,
      );
      const geometry = estimateTimeDomainBand(inPhase, quadrature, {
        minBandwidth: minimumBandwidth,
      });
      const result = preprocessTimeDomainInvariantPatches(
        inPhase,
        quadrature,
        options,
      );
      maximumErrors.center = Math.max(
        maximumErrors.center,
        Math.abs(geometry.center - entry.expected.center),
      );
      maximumErrors.bandwidth = Math.max(
        maximumErrors.bandwidth,
        Math.abs(geometry.bandwidth - entry.expected.bandwidth),
      );
      maximumErrors.geometryContext = Math.max(
        maximumErrors.geometryContext,
        compareJsonContract(
          geometry.context,
          entry.expected.geometry_context,
          `${entry.name}.geometry`,
        ),
      );
      maximumErrors.packed = Math.max(
        maximumErrors.packed,
        maxAbsoluteError(
          result.packedInPhase,
          entry.expected.packed_in_phase,
        ),
        maxAbsoluteError(
          result.packedQuadrature,
          entry.expected.packed_quadrature,
        ),
      );
      maximumErrors.features = Math.max(
        maximumErrors.features,
        maxAbsoluteError(result.rawFeatures, entry.expected.raw_features),
      );
      maximumErrors.preprocessContext = Math.max(
        maximumErrors.preprocessContext,
        compareJsonContract(
          result.context,
          entry.expected.context,
          `${entry.name}.preprocess`,
        ),
      );
    }
    expect(maximumErrors.center).toBeLessThan(2e-12);
    expect(maximumErrors.bandwidth).toBeLessThan(2e-12);
    expect(maximumErrors.geometryContext).toBeLessThan(2e-12);
    expect(maximumErrors.packed).toBeLessThan(3e-5);
    expect(maximumErrors.features).toBeLessThan(3e-4);
    expect(maximumErrors.preprocessContext).toBeLessThan(2e-12);
  });

  it('preserves gain/phase invariance and carrier-shift equivariance', () => {
    const baseEntry = byName('normal-base');
    const base = preprocessTimeDomainInvariantPatches(
      Float64Array.from(baseEntry.input.in_phase),
      Float64Array.from(baseEntry.input.quadrature),
    );
    for (const name of ['gain-tiny-phase', 'gain-huge-phase']) {
      const entry = byName(name);
      const result = preprocessTimeDomainInvariantPatches(
        Float64Array.from(entry.input.in_phase),
        Float64Array.from(entry.input.quadrature),
      );
      expect(result.context.center).toBeCloseTo(base.context.center, 12);
      expect(result.context.bw).toBeCloseTo(base.context.bw, 12);
      expect(
        maxAbsoluteError(result.rawFeatures, Array.from(base.rawFeatures)),
      ).toBeLessThan(3e-5);
      const phase = entry.nuisance.global_phase!;
      let worstEquivariance = 0;
      for (let index = 0; index < base.packedInPhase.length; index++) {
        const expectedReal = (
          base.packedInPhase[index]! * Math.cos(phase)
          - base.packedQuadrature[index]! * Math.sin(phase)
        );
        const expectedImaginary = (
          base.packedInPhase[index]! * Math.sin(phase)
          + base.packedQuadrature[index]! * Math.cos(phase)
        );
        worstEquivariance = Math.max(
          worstEquivariance,
          Math.abs(result.packedInPhase[index]! - expectedReal),
          Math.abs(result.packedQuadrature[index]! - expectedImaginary),
        );
      }
      expect(worstEquivariance).toBeLessThan(3e-6);
    }

    const shiftedEntry = byName('carrier-shift');
    const shifted = preprocessTimeDomainInvariantPatches(
      Float64Array.from(shiftedEntry.input.in_phase),
      Float64Array.from(shiftedEntry.input.quadrature),
    );
    const shift = shiftedEntry.nuisance.carrier_shift!;
    const expectedCenter = (
      base.context.center + shift
      - Math.floor(base.context.center + shift + 0.5)
    );
    expect(shifted.context.center).toBeCloseTo(expectedCenter, 12);
    expect(shifted.context.bw).toBeCloseTo(base.context.bw, 12);
    expect(
      maxAbsoluteError(
        shifted.packedInPhase,
        Array.from(base.packedInPhase),
      ),
    ).toBeLessThan(3e-6);
    expect(
      maxAbsoluteError(
        shifted.packedQuadrature,
        Array.from(base.packedQuadrature),
      ),
    ).toBeLessThan(3e-6);
  });

  it('obeys the physical scale law and handles both Nyquist seams', () => {
    const scaled = fixture.cases
      .filter((entry) => entry.relation === 'physical-scale')
      .map((entry) => {
        const result = preprocessTimeDomainInvariantPatches(
          Float64Array.from(entry.input.in_phase),
          Float64Array.from(entry.input.quadrature),
        );
        return {
          center: result.context.center / entry.physical_scale!,
          bandwidth: result.context.bw / entry.physical_scale!,
          activeLength: result.context.active_length,
        };
      });
    const reference = scaled[1]!;
    for (const row of scaled) {
      expect(Math.abs(row.center - reference.center)).toBeLessThan(8e-6);
      expect(Math.abs(row.bandwidth - reference.bandwidth)).toBeLessThan(2e-4);
      expect(Math.abs(row.activeLength - reference.activeLength)).toBeLessThanOrEqual(1);
    }
    for (const name of [
      'positive-nyquist-seam',
      'negative-nyquist-seam',
    ]) {
      const entry = byName(name);
      const result = preprocessTimeDomainInvariantPatches(
        Float64Array.from(entry.input.in_phase),
        Float64Array.from(entry.input.quadrature),
      );
      expect(result.context.center).toBeGreaterThanOrEqual(-0.5);
      expect(result.context.center).toBeLessThan(0.5);
      expect(result.context.bw).toBeGreaterThan(0);
      expect(Number.isFinite(result.context.bw)).toBe(true);
    }
  });

  it('covers short-repeat, exact patch boundary, and full-band fallback paths', () => {
    for (const name of [
      'short-three',
      'short-thirty-three',
      'short-thirty-four',
    ]) {
      const entry = byName(name);
      const result = preprocessTimeDomainInvariantPatches(
        Float64Array.from(entry.input.in_phase),
        Float64Array.from(entry.input.quadrature),
      );
      expect(result.context.active_length).toBeLessThan(64);
      expect(result.context.patches_repeated_from_short_active).toBe(true);
      expect(result.context.padding_samples).toBe(0);
    }
    const boundary = byName('short-thirty-five');
    const boundaryResult = preprocessTimeDomainInvariantPatches(
      Float64Array.from(boundary.input.in_phase),
      Float64Array.from(boundary.input.quadrature),
    );
    expect(boundaryResult.context.active_length).toBe(64);
    expect(boundaryResult.context.patches_repeated_from_short_active).toBe(false);

    const fallback = byName('lag-one-degenerate-impulse');
    const fallbackResult = preprocessTimeDomainInvariantPatches(
      Float64Array.from(fallback.input.in_phase),
      Float64Array.from(fallback.input.quadrature),
    );
    expect(
      fallbackResult.context.geometry_estimator.degenerate_full_band_fallback,
    ).toBe(true);
    expect(fallbackResult.context.center).toBe(0);
    expect(fallbackResult.context.bw).toBe(0.95);
  });

  it('fails closed for zero, non-finite, malformed, and unsafe inputs', () => {
    const zero = new Float64Array(64);
    expect(() =>
      preprocessTimeDomainInvariantPatches(zero, zero),
    ).toThrow(/zero magnitude/);
    expect(() =>
      preprocessTimeDomainInvariantPatches(
        Float64Array.of(1, 1),
        Float64Array.of(0, 0),
      ),
    ).toThrow(/at least three/);
    expect(() =>
      preprocessTimeDomainInvariantPatches(
        Float64Array.of(1, 1, 1),
        Float64Array.of(0, 0),
      ),
    ).toThrow(/equal I\/Q/);
    expect(() =>
      preprocessTimeDomainInvariantPatches(
        Float64Array.of(1, Number.NaN, 1),
        Float64Array.of(0, 0, 0),
      ),
    ).toThrow(/NaN or infinity/);
    expect(() =>
      estimateTimeDomainBand(
        Float64Array.of(1, 1, 1),
        Float64Array.of(0, Number.POSITIVE_INFINITY, 0),
      ),
    ).toThrow(/NaN or infinity/);
    const validReal = Float64Array.of(1, 0, -1, 0);
    const validImaginary = Float64Array.of(0, 1, 0, -1);
    expect(() =>
      preprocessTimeDomainInvariantPatches(validReal, validImaginary, {
        patchLength: 0,
      }),
    ).toThrow(/positive integer/);
    expect(() =>
      preprocessTimeDomainInvariantPatches(validReal, validImaginary, {
        patchCount: 10_000_001,
      }),
    ).toThrow(/safety limit/);
    expect(() =>
      preprocessTimeDomainInvariantPatches(validReal, validImaginary, {
        targetFrac: Number.NaN,
      }),
    ).toThrow(/finite fraction/);
    expect(() =>
      estimateTimeDomainBand(validReal, validImaginary, {
        minBandwidth: 0.8,
        fullBandwidth: 0.7,
      }),
    ).toThrow(/may not exceed/);
  });

  it('is deterministic and has no executable transform dependency', () => {
    const entry = byName('odd-length-51');
    const inPhase = Float64Array.from(entry.input.in_phase);
    const quadrature = Float64Array.from(entry.input.quadrature);
    const first = preprocessTimeDomainInvariantPatches(inPhase, quadrature);
    const second = preprocessTimeDomainInvariantPatches(inPhase, quadrature);
    expect(first.packedInPhase).toEqual(second.packedInPhase);
    expect(first.packedQuadrature).toEqual(second.packedQuadrature);
    expect(first.rawFeatures).toEqual(second.rawFeatures);
    expect(first.context).toEqual(second.context);

    const source = [
      readFileSync(
        new URL('./time-domain-geometry-v3.ts', import.meta.url),
        'utf8',
      ),
      readFileSync(
        new URL(
          './time-domain-invariant-patch-preprocess-v3.ts',
          import.meta.url,
        ),
        'utf8',
      ),
    ].join('\n');
    expect(source).not.toMatch(/\b(?:fft|stft|welch)\s*\(/iu);
    expect(source).not.toContain("from './iq-preprocess.js'");
    expect(first.context.uses_frequency_transform).toBe(false);
    expect(first.context.geometry_estimator.uses_frequency_transform).toBe(false);
  });
});
