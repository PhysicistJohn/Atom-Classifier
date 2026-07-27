/**
 * Strict no-transform invariant-patch frontend.
 *
 * This ports `training/time_domain_invariant_patch_preprocess.py` and the
 * relevant fixed-u core from `training/invariant_patch_preprocess.py` into a
 * new runtime contract. It does not import or mutate the frozen hybrid-v3
 * frontend.
 */

import {
  estimateTimeDomainBand,
  minimumTimeDomainBandwidthForActiveSpan,
  timeDomainGeometryMetadata,
  TIME_DOMAIN_GEOMETRY_VERSION,
  type TimeDomainGeometryContext,
  type TimeDomainGeometryMetadata,
} from './time-domain-geometry-v3.js';

export const TIME_DOMAIN_INVARIANT_PATCH_PREPROCESS_VERSION =
  'invariant-patch-time-domain-v1' as const;
export const DEFAULT_TIME_DOMAIN_PATCH_LENGTH = 64;
export const DEFAULT_TIME_DOMAIN_PATCH_COUNT = 16;
export const DEFAULT_TIME_DOMAIN_TARGET_FRAC = 0.5;
export const MAX_TIME_DOMAIN_ACTIVE_LENGTH = 10_000_000;
export const MAX_TIME_DOMAIN_PACKED_LENGTH = 10_000_000;
export const TIME_DOMAIN_FEATURE_COUNT = 12;

const FEATURE_SOURCE =
  'complete RMS-normalized active sequence before patch repetition';

export interface TimeDomainInvariantPatchOptions {
  patchLength?: number;
  patchCount?: number;
  targetFrac?: number;
}

export interface TimeDomainInvariantPatchContext {
  version: typeof TIME_DOMAIN_INVARIANT_PATCH_PREPROCESS_VERSION;
  estimator_version: typeof TIME_DOMAIN_GEOMETRY_VERSION;
  raw_length: number;
  raw_peak: number;
  detected_center: number;
  detected_bw: number;
  center: number;
  bw: number;
  resample_frac: number;
  target_frac: number;
  nfft: null;
  u_step: number;
  captured_u_end: number;
  active_u_end: number;
  active_length: number;
  active_rms_before_normalization: number;
  patch_length: number;
  patch_count: number;
  patch_starts: number[];
  patches_repeated_from_short_active: boolean;
  active_repeat_count: number;
  padding_samples: 0;
  packed_length: number;
  feature_source: typeof FEATURE_SOURCE;
  geometry_estimator: TimeDomainGeometryContext;
  uses_frequency_transform: false;
}

export interface TimeDomainInvariantPatchResult {
  /** Float32-quantized values carried in JavaScript numbers. */
  packedInPhase: Float64Array;
  /** Float32-quantized values carried in JavaScript numbers. */
  packedQuadrature: Float64Array;
  /** Float32-quantized 12-feature vector carried in JavaScript numbers. */
  rawFeatures: Float64Array;
  context: TimeDomainInvariantPatchContext;
}

export interface TimeDomainInvariantPatchMetadata {
  version: typeof TIME_DOMAIN_INVARIANT_PATCH_PREPROCESS_VERSION;
  patch_length: number;
  patch_count: number;
  target_frac: number;
  geometry: TimeDomainGeometryMetadata;
  uses_frequency_transform: false;
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value <= 0) {
    throw new RangeError(`${name} must be a positive integer`);
  }
  return value;
}

function finiteFraction(value: number, name: string): number {
  if (!Number.isFinite(value) || value <= 0 || value > 1) {
    throw new RangeError(`${name} must be a finite fraction in (0, 1]`);
  }
  return value;
}

function wrapCycles(value: number): number {
  if (!Number.isFinite(value)) {
    throw new RangeError('center must be finite');
  }
  return value - Math.floor(value + 0.5);
}

function nextUp(value: number): number {
  if (Number.isNaN(value) || value === Number.POSITIVE_INFINITY) return value;
  if (value === 0) return Number.MIN_VALUE;
  const storage = new ArrayBuffer(8);
  const view = new DataView(storage);
  view.setFloat64(0, value, false);
  let bits = view.getBigUint64(0, false);
  bits = value > 0 ? bits + 1n : bits - 1n;
  view.setBigUint64(0, bits, false);
  return view.getFloat64(0, false);
}

function peakNormalize(
  inPhase: ArrayLike<number>,
  quadrature: ArrayLike<number>,
): {
  inPhase: Float64Array;
  quadrature: Float64Array;
  peak: number;
} {
  if (inPhase.length < 2 || inPhase.length !== quadrature.length) {
    throw new RangeError(
      'expected equal I/Q channels with at least two samples',
    );
  }
  let peak = 0;
  for (let index = 0; index < inPhase.length; index++) {
    const real = inPhase[index]!;
    const imaginary = quadrature[index]!;
    if (!Number.isFinite(real) || !Number.isFinite(imaginary)) {
      throw new RangeError('I/Q capture contains NaN or infinity');
    }
    peak = Math.max(peak, Math.hypot(real, imaginary));
  }
  if (!Number.isFinite(peak)) {
    throw new RangeError('I/Q peak magnitude is not finite');
  }
  if (peak === 0) {
    throw new RangeError('I/Q capture has zero magnitude');
  }
  const real = new Float64Array(inPhase.length);
  const imaginary = new Float64Array(inPhase.length);
  for (let index = 0; index < inPhase.length; index++) {
    real[index] = inPhase[index]! / peak;
    imaginary[index] = quadrature[index]! / peak;
  }
  return { inPhase: real, quadrature: imaginary, peak };
}

function baseband(
  inPhase: Float64Array,
  quadrature: Float64Array,
  center: number,
): {
  inPhase: Float64Array;
  quadrature: Float64Array;
} {
  const real = new Float64Array(inPhase.length);
  const imaginary = new Float64Array(inPhase.length);
  for (let index = 0; index < inPhase.length; index++) {
    const phase = -2 * Math.PI * center * index;
    const cosine = Math.cos(phase);
    const sine = Math.sin(phase);
    real[index] = (
      inPhase[index]! * cosine - quadrature[index]! * sine
    );
    imaginary[index] = (
      inPhase[index]! * sine + quadrature[index]! * cosine
    );
  }
  return { inPhase: real, quadrature: imaginary };
}

function resampleOnU(
  inPhase: Float64Array,
  quadrature: Float64Array,
  resampleFrac: number,
  targetFrac: number,
): {
  inPhase: Float64Array;
  quadrature: Float64Array;
  capturedUEnd: number;
} {
  const uStep = resampleFrac / targetFrac;
  const capturedUEnd = (inPhase.length - 1) * uStep;
  if (!Number.isFinite(capturedUEnd)) {
    throw new RangeError(
      'dimensionless active-sequence extent is not finite',
    );
  }
  const lastU = Math.floor(nextUp(capturedUEnd));
  const activeLength = lastU + 1;
  if (activeLength < 2) {
    throw new RangeError(
      'normalized active sequence has fewer than two samples; '
      + 'increase capture length or resample fraction',
    );
  }
  if (activeLength > MAX_TIME_DOMAIN_ACTIVE_LENGTH) {
    throw new RangeError(
      `normalized active sequence length ${activeLength} exceeds safety limit `
      + `${MAX_TIME_DOMAIN_ACTIVE_LENGTH}`,
    );
  }

  const real = new Float64Array(activeLength);
  const imaginary = new Float64Array(activeLength);
  const maximumPosition = inPhase.length - 1;
  for (let outputIndex = 0; outputIndex < activeLength; outputIndex++) {
    const position = Math.min(
      maximumPosition,
      Math.max(0, outputIndex / uStep),
    );
    const lower = Math.floor(position);
    const upper = Math.min(lower + 1, maximumPosition);
    const fraction = position - lower;
    real[outputIndex] = (
      inPhase[lower]! * (1 - fraction)
      + inPhase[upper]! * fraction
    );
    imaginary[outputIndex] = (
      quadrature[lower]! * (1 - fraction)
      + quadrature[upper]! * fraction
    );
  }
  return { inPhase: real, quadrature: imaginary, capturedUEnd };
}

function rmsNormalize(
  inPhase: Float64Array,
  quadrature: Float64Array,
): {
  inPhase: Float64Array;
  quadrature: Float64Array;
  rms: number;
} {
  let power = 0;
  for (let index = 0; index < inPhase.length; index++) {
    power += (
      inPhase[index]! * inPhase[index]!
      + quadrature[index]! * quadrature[index]!
    );
  }
  power /= inPhase.length;
  if (!Number.isFinite(power) || power <= 0) {
    throw new RangeError(
      'resampled active sequence has invalid or zero power',
    );
  }
  const rms = Math.sqrt(power);
  const real = new Float64Array(inPhase.length);
  const imaginary = new Float64Array(inPhase.length);
  for (let index = 0; index < inPhase.length; index++) {
    real[index] = inPhase[index]! / rms;
    imaginary[index] = quadrature[index]! / rms;
    if (!Number.isFinite(real[index]) || !Number.isFinite(imaginary[index])) {
      throw new RangeError(
        'active-sequence RMS normalization produced non-finite values',
      );
    }
  }
  return { inPhase: real, quadrature: imaginary, rms };
}

function unwrapPhase(phase: Float64Array): Float64Array {
  const result = new Float64Array(phase.length);
  if (phase.length === 0) return result;
  result[0] = phase[0]!;
  let correction = 0;
  const twoPi = 2 * Math.PI;
  for (let index = 1; index < phase.length; index++) {
    const delta = phase[index]! - phase[index - 1]!;
    const shifted = delta + Math.PI;
    let reduced = (
      shifted - twoPi * Math.floor(shifted / twoPi) - Math.PI
    );
    if (reduced === -Math.PI && delta > 0) reduced = Math.PI;
    let phaseCorrection = reduced - delta;
    if (Math.abs(delta) < Math.PI) phaseCorrection = 0;
    correction += phaseCorrection;
    result[index] = phase[index]! + correction;
  }
  return result;
}

/**
 * Mirror of `preprocess.iq_features`, kept local so this strict runtime has no
 * dependency on a frequency-domain frontend module.
 */
function invariantIqFeatures(
  inPhase: Float64Array,
  quadrature: Float64Array,
): Float64Array {
  const length = inPhase.length;
  let meanReal = 0;
  let meanImaginary = 0;
  for (let index = 0; index < length; index++) {
    meanReal += inPhase[index]!;
    meanImaginary += quadrature[index]!;
  }
  meanReal /= length;
  meanImaginary /= length;

  const real = new Float64Array(length);
  const imaginary = new Float64Array(length);
  let power = 0;
  for (let index = 0; index < length; index++) {
    const centeredReal = inPhase[index]! - meanReal;
    const centeredImaginary = quadrature[index]! - meanImaginary;
    real[index] = centeredReal;
    imaginary[index] = centeredImaginary;
    power += (
      centeredReal * centeredReal
      + centeredImaginary * centeredImaginary
    );
  }
  const normalization = Math.sqrt(power / length) + 1e-12;
  for (let index = 0; index < length; index++) {
    real[index] = real[index]! / normalization;
    imaginary[index] = imaginary[index]! / normalization;
  }

  let m20Real = 0;
  let m20Imaginary = 0;
  let m40Real = 0;
  let m40Imaginary = 0;
  let m41Real = 0;
  let m41Imaginary = 0;
  let m42 = 0;
  let m60Real = 0;
  let m60Imaginary = 0;
  let m63 = 0;
  let sumMagnitude = 0;
  let sumMagnitudeSquared = 0;
  let maximumMagnitudeSquared = 0;
  for (let index = 0; index < length; index++) {
    const a = real[index]!;
    const b = imaginary[index]!;
    const magnitudeSquared = a * a + b * b;
    const squaredReal = a * a - b * b;
    const squaredImaginary = 2 * a * b;
    m20Real += squaredReal;
    m20Imaginary += squaredImaginary;
    const fourthReal = (
      squaredReal * squaredReal - squaredImaginary * squaredImaginary
    );
    const fourthImaginary = 2 * squaredReal * squaredImaginary;
    m40Real += fourthReal;
    m40Imaginary += fourthImaginary;
    m41Real += squaredReal * magnitudeSquared;
    m41Imaginary += squaredImaginary * magnitudeSquared;
    m42 += magnitudeSquared * magnitudeSquared;
    m60Real += (
      fourthReal * squaredReal - fourthImaginary * squaredImaginary
    );
    m60Imaginary += (
      fourthReal * squaredImaginary + fourthImaginary * squaredReal
    );
    m63 += magnitudeSquared * magnitudeSquared * magnitudeSquared;
    sumMagnitude += Math.sqrt(magnitudeSquared);
    sumMagnitudeSquared += magnitudeSquared;
    maximumMagnitudeSquared = Math.max(
      maximumMagnitudeSquared,
      magnitudeSquared,
    );
  }
  m20Real /= length;
  m20Imaginary /= length;
  m40Real /= length;
  m40Imaginary /= length;
  m41Real /= length;
  m41Imaginary /= length;
  m42 /= length;
  m60Real /= length;
  m60Imaginary /= length;
  m63 /= length;

  const complexMagnitude = (realPart: number, imaginaryPart: number): number =>
    Math.sqrt(realPart * realPart + imaginaryPart * imaginaryPart);
  const c20 = complexMagnitude(m20Real, m20Imaginary);
  const m20SquaredReal = (
    m20Real * m20Real - m20Imaginary * m20Imaginary
  );
  const m20SquaredImaginary = 2 * m20Real * m20Imaginary;
  const c40 = complexMagnitude(
    m40Real - 3 * m20SquaredReal,
    m40Imaginary - 3 * m20SquaredImaginary,
  );
  const c41 = complexMagnitude(
    m41Real - 3 * m20Real,
    m41Imaginary - 3 * m20Imaginary,
  );
  const c42 = (
    m42 - (m20Real * m20Real + m20Imaginary * m20Imaginary) - 2
  );
  const m20TimesM40Real = (
    m20Real * m40Real - m20Imaginary * m40Imaginary
  );
  const m20TimesM40Imaginary = (
    m20Real * m40Imaginary + m20Imaginary * m40Real
  );
  const m20CubedReal = (
    m20SquaredReal * m20Real - m20SquaredImaginary * m20Imaginary
  );
  const m20CubedImaginary = (
    m20SquaredReal * m20Imaginary + m20SquaredImaginary * m20Real
  );
  const c60 = complexMagnitude(
    m60Real - 15 * m20TimesM40Real + 30 * m20CubedReal,
    m60Imaginary - 15 * m20TimesM40Imaginary + 30 * m20CubedImaginary,
  );
  const c63 = m63 - 9 * c42 - 6;
  const meanMagnitude = sumMagnitude / length;
  const varianceMagnitude = (
    sumMagnitudeSquared / length - meanMagnitude * meanMagnitude
  );
  const standardDeviationMagnitude = Math.sqrt(
    Math.max(varianceMagnitude, 0),
  );

  const phase = new Float64Array(length);
  for (let index = 0; index < length; index++) {
    phase[index] = Math.atan2(imaginary[index]!, real[index]!);
  }
  const unwrapped = unwrapPhase(phase);
  let frequencyMean = 0;
  for (let index = 1; index < length; index++) {
    frequencyMean += unwrapped[index]! - unwrapped[index - 1]!;
  }
  frequencyMean /= length - 1;
  let frequencyVariance = 0;
  for (let index = 1; index < length; index++) {
    const centered = (
      unwrapped[index]! - unwrapped[index - 1]! - frequencyMean
    );
    frequencyVariance += centered * centered;
  }
  frequencyVariance /= length - 1;
  const standardDeviationFrequency = Math.sqrt(
    Math.max(frequencyVariance, 0),
  );
  const coefficientOfVariation = (
    standardDeviationMagnitude / (meanMagnitude + 1e-9)
  );

  return Float64Array.from([
    c20,
    c40,
    c42,
    c41,
    c60,
    c63,
    m42,
    standardDeviationMagnitude,
    maximumMagnitudeSquared,
    meanMagnitude,
    standardDeviationFrequency,
    coefficientOfVariation,
  ]);
}

function evenPatchStarts(
  activeLength: number,
  patchLength: number,
  patchCount: number,
): number[] {
  const maximum = activeLength - patchLength;
  if (maximum < 0 || patchCount === 1) {
    return new Array<number>(patchCount).fill(0);
  }
  const denominator = 2 * (patchCount - 1);
  const starts = new Array<number>(patchCount);
  for (let index = 0; index < patchCount; index++) {
    starts[index] = Math.floor(
      (2 * index * maximum + patchCount - 1) / denominator,
    );
  }
  return starts;
}

function extractPatches(
  inPhase: Float64Array,
  quadrature: Float64Array,
  patchLength: number,
  patchCount: number,
): {
  packedInPhase: Float64Array;
  packedQuadrature: Float64Array;
  starts: number[];
  repeated: boolean;
  repeatCount: number;
} {
  const starts = evenPatchStarts(
    inPhase.length,
    patchLength,
    patchCount,
  );
  const packedLength = patchLength * patchCount;
  const packedInPhase = new Float64Array(packedLength);
  const packedQuadrature = new Float64Array(packedLength);
  const repeated = inPhase.length < patchLength;
  const repeatCount = repeated
    ? Math.ceil(patchLength / inPhase.length)
    : 1;
  for (let patch = 0; patch < patchCount; patch++) {
    const start = starts[patch]!;
    const outputOffset = patch * patchLength;
    for (let index = 0; index < patchLength; index++) {
      const source = repeated ? index % inPhase.length : start + index;
      packedInPhase[outputOffset + index] = Math.fround(inPhase[source]!);
      packedQuadrature[outputOffset + index] = Math.fround(
        quadrature[source]!,
      );
    }
  }
  return {
    packedInPhase,
    packedQuadrature,
    starts,
    repeated,
    repeatCount,
  };
}

/** Convert one raw complex capture into strict time-domain invariant patches. */
export function preprocessTimeDomainInvariantPatches(
  inPhase: ArrayLike<number>,
  quadrature: ArrayLike<number>,
  options: TimeDomainInvariantPatchOptions = {},
): TimeDomainInvariantPatchResult {
  const patchLength = positiveInteger(
    options.patchLength ?? DEFAULT_TIME_DOMAIN_PATCH_LENGTH,
    'patchLength',
  );
  const targetFrac = finiteFraction(
    options.targetFrac ?? DEFAULT_TIME_DOMAIN_TARGET_FRAC,
    'targetFrac',
  );
  const minimumBandwidth = minimumTimeDomainBandwidthForActiveSpan(
    inPhase.length,
    patchLength,
    targetFrac,
  );
  const geometry = estimateTimeDomainBand(inPhase, quadrature, {
    minBandwidth: minimumBandwidth,
  });

  const patchCount = positiveInteger(
    options.patchCount ?? DEFAULT_TIME_DOMAIN_PATCH_COUNT,
    'patchCount',
  );
  const packedLength = patchLength * patchCount;
  if (
    !Number.isSafeInteger(packedLength)
    || packedLength > MAX_TIME_DOMAIN_PACKED_LENGTH
  ) {
    throw new RangeError(
      `packed patch length ${packedLength} exceeds safety limit `
      + `${MAX_TIME_DOMAIN_PACKED_LENGTH}`,
    );
  }
  const center = wrapCycles(geometry.center);
  const bandwidth = finiteFraction(geometry.bandwidth, 'bandwidth');
  const resampleFrac = bandwidth;

  const normalized = peakNormalize(inPhase, quadrature);
  const converted = baseband(
    normalized.inPhase,
    normalized.quadrature,
    center,
  );
  const resampled = resampleOnU(
    converted.inPhase,
    converted.quadrature,
    resampleFrac,
    targetFrac,
  );
  const active = rmsNormalize(
    resampled.inPhase,
    resampled.quadrature,
  );
  const computedFeatures = invariantIqFeatures(
    active.inPhase,
    active.quadrature,
  );
  if (computedFeatures.length !== TIME_DOMAIN_FEATURE_COUNT) {
    throw new Error(
      'active-sequence iqFeatures returned malformed geometry',
    );
  }
  const rawFeatures = new Float64Array(computedFeatures.length);
  for (let index = 0; index < computedFeatures.length; index++) {
    if (!Number.isFinite(computedFeatures[index])) {
      throw new RangeError(
        'active-sequence iqFeatures returned non-finite values',
      );
    }
    rawFeatures[index] = Math.fround(computedFeatures[index]!);
  }
  const patches = extractPatches(
    active.inPhase,
    active.quadrature,
    patchLength,
    patchCount,
  );

  return {
    packedInPhase: patches.packedInPhase,
    packedQuadrature: patches.packedQuadrature,
    rawFeatures,
    context: {
      version: TIME_DOMAIN_INVARIANT_PATCH_PREPROCESS_VERSION,
      estimator_version: TIME_DOMAIN_GEOMETRY_VERSION,
      raw_length: inPhase.length,
      raw_peak: normalized.peak,
      detected_center: geometry.center,
      detected_bw: geometry.bandwidth,
      center,
      bw: bandwidth,
      resample_frac: resampleFrac,
      target_frac: targetFrac,
      nfft: null,
      u_step: resampleFrac / targetFrac,
      captured_u_end: resampled.capturedUEnd,
      active_u_end: active.inPhase.length - 1,
      active_length: active.inPhase.length,
      active_rms_before_normalization: active.rms,
      patch_length: patchLength,
      patch_count: patchCount,
      patch_starts: patches.starts,
      patches_repeated_from_short_active: patches.repeated,
      active_repeat_count: patches.repeatCount,
      padding_samples: 0,
      packed_length: packedLength,
      feature_source: FEATURE_SOURCE,
      geometry_estimator: geometry.context,
      uses_frequency_transform: false,
    },
  };
}

export function timeDomainInvariantPatchMetadata():
TimeDomainInvariantPatchMetadata {
  return {
    version: TIME_DOMAIN_INVARIANT_PATCH_PREPROCESS_VERSION,
    patch_length: DEFAULT_TIME_DOMAIN_PATCH_LENGTH,
    patch_count: DEFAULT_TIME_DOMAIN_PATCH_COUNT,
    target_frac: DEFAULT_TIME_DOMAIN_TARGET_FRAC,
    geometry: timeDomainGeometryMetadata(),
    uses_frequency_transform: false,
  };
}
