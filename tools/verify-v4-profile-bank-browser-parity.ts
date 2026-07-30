/**
 * Verify a Python-exported v4 browser asset and probe fixture in Node.
 *
 * Usage after `export_current_source_browser_runtime.py`:
 *
 *   npx tsup tools/verify-v4-profile-bank-browser-parity.ts \
 *     --format esm --platform node --out-dir .artifacts/v4-parity \
 *     --no-splitting --treeshake --silent
 *   node .artifacts/v4-parity/verify-v4-profile-bank-browser-parity.js \
 *     /path/to/time-domain-profile-bank-v4.json \
 *     /path/to/time-domain-profile-bank-probe-v4.json
 */

import { readFileSync } from 'node:fs';
import {
  classifyTimeDomainProfileBankV4,
  loadTimeDomainProfileBankAssetV4,
  TIME_DOMAIN_PUBLIC_CLASSES_V4,
} from '../src/embedding/time-domain-profile-bank-classifier-v4.js';
import type {
  TimeDomainPrototypeSourceV4,
} from '../src/embedding/time-domain-profile-routing-v4.js';

const PROBE_SCHEMA =
  'atomos.v4.time-domain-current-source-profile-bank.probe-fixture';

type UnknownRecord = Record<string, unknown>;

function record(value: unknown, path: string): UnknownRecord {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new TypeError(`${path} must be an object`);
  }
  return value as UnknownRecord;
}

function finite(value: unknown, path: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`${path} must be finite`);
  }
  return value;
}

function integer(value: unknown, path: string): number {
  const result = finite(value, path);
  if (!Number.isSafeInteger(result)) {
    throw new RangeError(`${path} must be a safe integer`);
  }
  return result;
}

function finiteVector(value: unknown, path: string): number[] {
  if (!Array.isArray(value)) {
    throw new TypeError(`${path} must be an array`);
  }
  return value.map((entry, index) => finite(entry, `${path}[${index}]`));
}

function nullableFiniteVector(
  value: unknown,
  path: string,
): Array<number | null> {
  if (!Array.isArray(value)) {
    throw new TypeError(`${path} must be an array`);
  }
  return value.map((entry, index) =>
    entry === null ? null : finite(entry, `${path}[${index}]`));
}

function booleanVector(value: unknown, path: string): boolean[] {
  if (!Array.isArray(value)) {
    throw new TypeError(`${path} must be an array`);
  }
  return value.map((entry, index) => {
    if (typeof entry !== 'boolean') {
      throw new TypeError(`${path}[${index}] must be boolean`);
    }
    return entry;
  });
}

function integerVector(value: unknown, path: string): number[] {
  if (!Array.isArray(value)) {
    throw new TypeError(`${path} must be an array`);
  }
  return value.map((entry, index) => integer(entry, `${path}[${index}]`));
}

function maxAbsError(
  actual: ArrayLike<number>,
  expectedValue: unknown,
  path: string,
): number {
  const expected = nullableFiniteVector(expectedValue, path);
  if (actual.length !== expected.length) {
    throw new RangeError(
      `${path} length mismatch: ${actual.length} != ${expected.length}`,
    );
  }
  let worst = 0;
  for (let index = 0; index < expected.length; index++) {
    if (expected[index] === null) {
      if (actual[index] !== Number.POSITIVE_INFINITY) {
        throw new RangeError(
          `${path}[${index}] must be Infinity for an unsupported class`,
        );
      }
      continue;
    }
    const error = Math.abs(actual[index]! - expected[index]!);
    if (!Number.isFinite(error)) {
      throw new RangeError(`${path}[${index}] produced a non-finite error`);
    }
    worst = Math.max(worst, error);
  }
  return worst;
}

function readJson(path: string): unknown {
  return JSON.parse(readFileSync(path, 'utf8')) as unknown;
}

export interface V4ParitySummary {
  cases: number;
  tolerance: number;
  worstByStage: Record<string, number>;
}

/** Run every Python reference case through the TypeScript browser runtime. */
export function verifyV4ProfileBankBrowserParity(
  rawAsset: unknown,
  rawFixture: unknown,
): V4ParitySummary {
  const asset = loadTimeDomainProfileBankAssetV4(rawAsset);
  const fixture = record(rawFixture, 'fixture');
  if (fixture.schema !== PROBE_SCHEMA || fixture.schema_version !== 2) {
    throw new RangeError('fixture schema/version is not the v4 profile probe');
  }
  if (fixture.uses_frequency_transform !== false) {
    throw new RangeError('fixture must declare uses_frequency_transform: false');
  }
  const tolerance = finite(fixture.tolerance, 'fixture.tolerance');
  if (tolerance <= 0) throw new RangeError('fixture.tolerance must be positive');
  if (
    !Array.isArray(fixture.classes)
    || fixture.classes.length !== TIME_DOMAIN_PUBLIC_CLASSES_V4.length
    || fixture.classes.some(
      (entry, index) => entry !== TIME_DOMAIN_PUBLIC_CLASSES_V4[index],
    )
  ) {
    throw new RangeError('fixture classes are not the canonical seven classes');
  }
  if (!Array.isArray(fixture.cases) || fixture.cases.length === 0) {
    throw new RangeError('fixture.cases must be non-empty');
  }

  const worstByStage: Record<string, number> = {};
  const measure = (
    stage: string,
    actual: ArrayLike<number>,
    expected: unknown,
    path: string,
  ): void => {
    const error = maxAbsError(actual, expected, path);
    worstByStage[stage] = Math.max(worstByStage[stage] ?? 0, error);
    if (error > tolerance) {
      throw new Error(
        `${path} parity error ${error} exceeds tolerance ${tolerance}`,
      );
    }
  };

  fixture.cases.forEach((caseValue, caseIndex) => {
    const path = `fixture.cases[${caseIndex}]`;
    const probe = record(caseValue, path);
    const raw = record(probe.raw, `${path}.raw`);
    const expected = record(probe.expected, `${path}.expected`);
    const inPhase = finiteVector(raw.in_phase, `${path}.raw.in_phase`);
    const quadrature = finiteVector(
      raw.quadrature,
      `${path}.raw.quadrature`,
    );
    const validSampleCount = integer(
      probe.valid_sample_count,
      `${path}.valid_sample_count`,
    );
    const prototypeSource = probe.prototype_source;
    if (
      prototypeSource !== 'current'
      && prototypeSource !== 'historical'
    ) {
      throw new RangeError(`${path}.prototype_source is invalid`);
    }
    const result = classifyTimeDomainProfileBankV4(
      asset,
      inPhase,
      quadrature,
      {
        validSampleCount,
        prototypeSource: prototypeSource as TimeDomainPrototypeSourceV4,
      },
    );
    if (
      result.runtimeInputLength
      !== integer(
        probe.runtime_input_length,
        `${path}.runtime_input_length`,
      )
    ) {
      throw new Error(`${path} selected the wrong runtime input bucket`);
    }
    measure(
      'packed_in_phase',
      result.preprocess.packedInPhase,
      expected.packed_in_phase,
      `${path}.expected.packed_in_phase`,
    );
    measure(
      'packed_quadrature',
      result.preprocess.packedQuadrature,
      expected.packed_quadrature,
      `${path}.expected.packed_quadrature`,
    );
    measure(
      'raw_features',
      result.preprocess.rawFeatures,
      expected.raw_features,
      `${path}.expected.raw_features`,
    );
    measure(
      'standardized_features',
      result.standardizedFeatures,
      expected.standardized_features,
      `${path}.expected.standardized_features`,
    );
    measure(
      'real_embedding',
      result.realEmbedding,
      expected.real_embedding,
      `${path}.expected.real_embedding`,
    );
    measure(
      'complex_embedding',
      result.complexEmbedding,
      expected.complex_embedding,
      `${path}.expected.complex_embedding`,
    );
    measure(
      'fused_embedding',
      result.fusedEmbedding,
      expected.fused_embedding,
      `${path}.expected.fused_embedding`,
    );
    measure(
      'squared_public_class_distances',
      result.squaredPublicClassDistances,
      expected.squared_public_class_distances,
      `${path}.expected.squared_public_class_distances`,
    );
    const expectedClosest = integerVector(
      expected.closest_prototype_indices,
      `${path}.expected.closest_prototype_indices`,
    );
    if (
      expectedClosest.length !== result.closestPrototypeIndices.length
      || expectedClosest.some(
        (entry, index) => entry !== result.closestPrototypeIndices[index],
      )
    ) {
      throw new Error(`${path} closest prototype indices differ`);
    }
    if (
      result.closedWinnerIndex
        !== integer(
          expected.closed_winner_index,
          `${path}.expected.closed_winner_index`,
        )
      || result.closedLabel !== expected.closed_label
    ) {
      throw new Error(`${path} closed winner/label differs`);
    }
    const expectedSupported = booleanVector(
      expected.supported_public_class_mask,
      `${path}.expected.supported_public_class_mask`,
    );
    if (
      expectedSupported.length !== result.supportedPublicClassMask.length
      || expectedSupported.some(
        (entry, index) => entry !== result.supportedPublicClassMask[index],
      )
      || result.prototypeSource !== prototypeSource
      || expected.prototype_source !== prototypeSource
    ) {
      throw new Error(`${path} trusted source/support mask differs`);
    }
  });
  return {
    cases: fixture.cases.length,
    tolerance,
    worstByStage,
  };
}

function main(): void {
  const [, , assetPath, fixturePath] = process.argv;
  if (assetPath === undefined || fixturePath === undefined) {
    throw new Error(
      'usage: verify-v4-profile-bank-browser-parity.js '
      + '<browser-weights.json> <probe-fixture.json>',
    );
  }
  const summary = verifyV4ProfileBankBrowserParity(
    readJson(assetPath),
    readJson(fixturePath),
  );
  // eslint-disable-next-line no-console
  console.info(JSON.stringify(summary, null, 2));
}

if (import.meta.url === `file://${process.argv[1]}`) main();
