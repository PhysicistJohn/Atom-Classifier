import type { TimeDomainInvariantPatchResult } from '../src/embedding/time-domain-invariant-patch-preprocess-v3.js';

export const TIME_DOMAIN_V3_AUDIT_LENGTHS = [
  4096,
  8192,
  16384,
  32768,
] as const;

export interface EncodedNumericArray {
  dtype: '<f8' | '<f4';
  shape: number[];
  base64: string;
  sha256: string;
  byte_length: number;
}

export interface TimeDomainV3BrowserCase {
  name: string;
  length: number;
  input_iq: EncodedNumericArray;
  expected_packed_iq: EncodedNumericArray;
  expected_raw_features: EncodedNumericArray;
  expected_context: Record<string, unknown>;
}

export interface TimeDomainV3BrowserFixture {
  fixture_version: 'time-domain-v3-browser-parity-v1';
  frontend_version: 'invariant-patch-time-domain-v1';
  geometry_version: 'time-correlation-v1';
  synthetic_only: true;
  reads_held_or_sealed_payload: false;
  input_method: string;
  uses_frequency_transform: false;
  lengths: number[];
  python_source_sha256: Record<string, string>;
  cases: TimeDomainV3BrowserCase[];
}

export interface DecodedTimeDomainV3Case {
  inPhase: Float64Array;
  quadrature: Float64Array;
  expectedPackedInPhase: Float64Array;
  expectedPackedQuadrature: Float64Array;
  expectedRawFeatures: Float64Array;
}

export interface TimeDomainV3ParityErrors {
  packed: number;
  rawFeatures: number;
  context: number;
}

function product(shape: number[]): number {
  let result = 1;
  for (const dimension of shape) {
    if (!Number.isInteger(dimension) || dimension <= 0) {
      throw new RangeError('encoded array shape must contain positive integers');
    }
    result *= dimension;
  }
  if (!Number.isSafeInteger(result)) {
    throw new RangeError('encoded array shape exceeds safe integer range');
  }
  return result;
}

export function decodeBase64(value: string): Uint8Array {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index++) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

function decodeFloat64(encoded: EncodedNumericArray): Float64Array {
  if (encoded.dtype !== '<f8') {
    throw new RangeError(`expected <f8, got ${encoded.dtype}`);
  }
  const bytes = decodeBase64(encoded.base64);
  const count = product(encoded.shape);
  if (bytes.byteLength !== count * 8 || bytes.byteLength !== encoded.byte_length) {
    throw new RangeError('encoded Float64 payload has inconsistent geometry');
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const values = new Float64Array(count);
  for (let index = 0; index < count; index++) {
    values[index] = view.getFloat64(index * 8, true);
  }
  return values;
}

function decodeFloat32AsFloat64(
  encoded: EncodedNumericArray,
): Float64Array {
  if (encoded.dtype !== '<f4') {
    throw new RangeError(`expected <f4, got ${encoded.dtype}`);
  }
  const bytes = decodeBase64(encoded.base64);
  const count = product(encoded.shape);
  if (bytes.byteLength !== count * 4 || bytes.byteLength !== encoded.byte_length) {
    throw new RangeError('encoded Float32 payload has inconsistent geometry');
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const values = new Float64Array(count);
  for (let index = 0; index < count; index++) {
    values[index] = view.getFloat32(index * 4, true);
  }
  return values;
}

export function decodeTimeDomainV3Case(
  entry: TimeDomainV3BrowserCase,
): DecodedTimeDomainV3Case {
  if (
    entry.input_iq.shape.length !== 2
    || entry.input_iq.shape[0] !== 2
    || entry.input_iq.shape[1] !== entry.length
  ) {
    throw new RangeError(`${entry.name}: malformed input geometry`);
  }
  const raw = decodeFloat64(entry.input_iq);
  const packed = decodeFloat32AsFloat64(entry.expected_packed_iq);
  const features = decodeFloat32AsFloat64(entry.expected_raw_features);
  const packedLength = entry.expected_packed_iq.shape[1];
  if (
    entry.expected_packed_iq.shape.length !== 2
    || entry.expected_packed_iq.shape[0] !== 2
    || packedLength === undefined
    || packed.length !== 2 * packedLength
  ) {
    throw new RangeError(`${entry.name}: malformed expected packed geometry`);
  }
  if (features.length !== 12) {
    throw new RangeError(`${entry.name}: malformed expected feature geometry`);
  }
  return {
    inPhase: raw.slice(0, entry.length),
    quadrature: raw.slice(entry.length),
    expectedPackedInPhase: packed.slice(0, packedLength),
    expectedPackedQuadrature: packed.slice(packedLength),
    expectedRawFeatures: features,
  };
}

function maxAbsoluteError(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
): number {
  if (actual.length !== expected.length) {
    throw new RangeError(
      `numeric parity length mismatch ${actual.length} != ${expected.length}`,
    );
  }
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

function contextError(
  actual: unknown,
  expected: unknown,
  path: string,
): number {
  if (typeof expected === 'number') {
    if (typeof actual !== 'number' || !Number.isFinite(actual)) {
      throw new TypeError(`${path}: expected a finite numeric value`);
    }
    return scaledNumericError(actual, expected);
  }
  if (expected === null || typeof expected !== 'object') {
    if (actual !== expected) {
      throw new Error(
        `${path}: exact contract mismatch ${String(actual)} != ${String(expected)}`,
      );
    }
    return 0;
  }
  if (Array.isArray(expected)) {
    if (!Array.isArray(actual) || actual.length !== expected.length) {
      throw new Error(`${path}: array contract mismatch`);
    }
    let maximum = 0;
    for (let index = 0; index < expected.length; index++) {
      maximum = Math.max(
        maximum,
        contextError(actual[index], expected[index], `${path}[${index}]`),
      );
    }
    return maximum;
  }
  if (actual === null || typeof actual !== 'object' || Array.isArray(actual)) {
    throw new Error(`${path}: object contract mismatch`);
  }
  const actualRecord = actual as Record<string, unknown>;
  const expectedRecord = expected as Record<string, unknown>;
  const actualKeys = Object.keys(actualRecord).sort();
  const expectedKeys = Object.keys(expectedRecord).sort();
  if (JSON.stringify(actualKeys) !== JSON.stringify(expectedKeys)) {
    throw new Error(`${path}: object keys differ`);
  }
  let maximum = 0;
  for (const [key, value] of Object.entries(expectedRecord)) {
    maximum = Math.max(
      maximum,
      contextError(actualRecord[key], value, `${path}.${key}`),
    );
  }
  return maximum;
}

export function compareTimeDomainV3Result(
  result: TimeDomainInvariantPatchResult,
  decoded: DecodedTimeDomainV3Case,
  expectedContext: Record<string, unknown>,
): TimeDomainV3ParityErrors {
  return {
    packed: Math.max(
      maxAbsoluteError(
        result.packedInPhase,
        decoded.expectedPackedInPhase,
      ),
      maxAbsoluteError(
        result.packedQuadrature,
        decoded.expectedPackedQuadrature,
      ),
    ),
    rawFeatures: maxAbsoluteError(
      result.rawFeatures,
      decoded.expectedRawFeatures,
    ),
    context: contextError(result.context, expectedContext, 'context'),
  };
}

export function assertTimeDomainV3Fixture(
  value: unknown,
): asserts value is TimeDomainV3BrowserFixture {
  if (value === null || typeof value !== 'object') {
    throw new TypeError('strict v3 fixture must be an object');
  }
  const fixture = value as Partial<TimeDomainV3BrowserFixture>;
  if (
    fixture.fixture_version !== 'time-domain-v3-browser-parity-v1'
    || fixture.frontend_version !== 'invariant-patch-time-domain-v1'
    || fixture.geometry_version !== 'time-correlation-v1'
  ) {
    throw new Error('strict v3 fixture contract/version mismatch');
  }
  if (
    fixture.synthetic_only !== true
    || fixture.reads_held_or_sealed_payload !== false
    || fixture.uses_frequency_transform !== false
  ) {
    throw new Error(
      'strict v3 runtime audit refuses non-synthetic, held, sealed, or transform data',
    );
  }
  if (
    !Array.isArray(fixture.lengths)
    || JSON.stringify(fixture.lengths)
      !== JSON.stringify(TIME_DOMAIN_V3_AUDIT_LENGTHS)
    || !Array.isArray(fixture.cases)
    || fixture.cases.length !== TIME_DOMAIN_V3_AUDIT_LENGTHS.length
  ) {
    throw new Error('strict v3 fixture must contain only the four audit lengths');
  }
  for (let index = 0; index < TIME_DOMAIN_V3_AUDIT_LENGTHS.length; index++) {
    if (
      fixture.cases[index]?.length !== TIME_DOMAIN_V3_AUDIT_LENGTHS[index]
    ) {
      throw new Error('strict v3 fixture case order/length mismatch');
    }
  }
}

export async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const copied = new Uint8Array(bytes.byteLength);
  copied.set(bytes);
  const digest = await crypto.subtle.digest('SHA-256', copied);
  return Array.from(new Uint8Array(digest), (value) =>
    value.toString(16).padStart(2, '0')).join('');
}

export async function verifyTimeDomainV3PayloadHashes(
  fixture: TimeDomainV3BrowserFixture,
): Promise<void> {
  for (const entry of fixture.cases) {
    for (const encoded of [
      entry.input_iq,
      entry.expected_packed_iq,
      entry.expected_raw_features,
    ]) {
      const bytes = decodeBase64(encoded.base64);
      if (
        bytes.byteLength !== encoded.byte_length
        || await sha256Hex(bytes) !== encoded.sha256
      ) {
        throw new Error(`${entry.name}: encoded payload hash mismatch`);
      }
    }
  }
}

export function percentile(
  values: number[],
  fraction: number,
): number {
  if (values.length === 0 || fraction < 0 || fraction > 1) {
    throw new RangeError('invalid percentile request');
  }
  const sorted = values.slice().sort((left, right) => left - right);
  const index = Math.max(
    0,
    Math.min(sorted.length - 1, Math.ceil(fraction * sorted.length) - 1),
  );
  return sorted[index]!;
}

export function outputsExactlyEqual(
  left: TimeDomainInvariantPatchResult,
  right: TimeDomainInvariantPatchResult,
): boolean {
  if (
    left.packedInPhase.length !== right.packedInPhase.length
    || left.packedQuadrature.length !== right.packedQuadrature.length
    || left.rawFeatures.length !== right.rawFeatures.length
  ) {
    return false;
  }
  for (let index = 0; index < left.packedInPhase.length; index++) {
    if (
      !Object.is(left.packedInPhase[index], right.packedInPhase[index])
      || !Object.is(
        left.packedQuadrature[index],
        right.packedQuadrature[index],
      )
    ) {
      return false;
    }
  }
  for (let index = 0; index < left.rawFeatures.length; index++) {
    if (!Object.is(left.rawFeatures[index], right.rawFeatures[index])) {
      return false;
    }
  }
  return JSON.stringify(left.context) === JSON.stringify(right.context);
}

export async function outputDigest(
  result: TimeDomainInvariantPatchResult,
): Promise<string> {
  const context = new TextEncoder().encode(JSON.stringify(result.context));
  const numericCount = (
    result.packedInPhase.length
    + result.packedQuadrature.length
    + result.rawFeatures.length
  );
  const bytes = new Uint8Array(numericCount * 4 + context.length);
  const view = new DataView(bytes.buffer);
  let offset = 0;
  for (const values of [
    result.packedInPhase,
    result.packedQuadrature,
    result.rawFeatures,
  ]) {
    for (let index = 0; index < values.length; index++) {
      view.setFloat32(offset, values[index]!, true);
      offset += 4;
    }
  }
  bytes.set(context, offset);
  return sha256Hex(bytes);
}
