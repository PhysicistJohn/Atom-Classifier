/**
 * TypeScript port of `training/invariant_patch_preprocess.py`.
 *
 * Unlike the production center-fit frontend, this keeps the full detected
 * active sequence in the dimensionless coordinate
 * `u = sample * occupiedBandwidth / targetBandwidth`, then extracts evenly
 * spaced fixed-size patches. No crop, zero padding, or post-estimator FFT is
 * performed.
 */

import {
  DEFAULT_PARAMS,
  estimateBand,
  iqFeatures,
  type PreprocessParams,
} from './iq-preprocess.js';

export const INVARIANT_PATCH_PREPROCESS_VERSION = 'invariant-patch-v1' as const;
export const INVARIANT_PATCH_ESTIMATOR_VERSION = 'hybrid-v3' as const;
export const DEFAULT_INVARIANT_PATCH_LENGTH = 64;
export const DEFAULT_INVARIANT_PATCH_COUNT = 16;
export const MAX_INVARIANT_ACTIVE_LENGTH = 10_000_000;
export const MAX_INVARIANT_PACKED_LENGTH = 10_000_000;

export interface InvariantPatchPreprocessOptions {
  patchLength?: number;
  patchCount?: number;
  targetFrac?: number;
  nfft?: number;
  forceCenter?: number;
  forceBw?: number;
  forceResampleFrac?: number;
  /** Advanced estimator overrides; version is required to remain hybrid-v3. */
  estimatorParams?: PreprocessParams;
}

export interface InvariantPatchContext {
  version: typeof INVARIANT_PATCH_PREPROCESS_VERSION;
  estimator_version: typeof INVARIANT_PATCH_ESTIMATOR_VERSION;
  raw_length: number;
  raw_peak: number;
  detected_center: number | null;
  detected_bw: number | null;
  center: number;
  bw: number;
  resample_frac: number;
  target_frac: number;
  nfft: number;
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
  feature_source: string;
}

export interface InvariantPatchPreprocessResult {
  packedInPhase: Float64Array;
  packedQuadrature: Float64Array;
  /** Float32-quantized production iqFeatures values, carried in JS numbers. */
  rawFeatures: Float64Array;
  context: InvariantPatchContext;
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value <= 0) {
    throw new RangeError(`${name} must be a positive integer`);
  }
  return value;
}

function finiteFraction(value: number, name: string): number {
  if (!Number.isFinite(value) || value <= 0 || value > 1) {
    throw new RangeError(`${name} must be a finite fraction in (0,1]`);
  }
  return value;
}

function wrapCycles(value: number): number {
  if (!Number.isFinite(value)) throw new RangeError('center must be finite');
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
  inPhase: Float64Array,
  quadrature: Float64Array,
): { inPhase: Float64Array; quadrature: Float64Array; peak: number } {
  if (inPhase.length < 2 || inPhase.length !== quadrature.length) {
    throw new RangeError('expected equal I/Q channels with at least two samples');
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
  if (!Number.isFinite(peak)) throw new RangeError('I/Q peak magnitude is not finite');
  if (peak === 0) throw new RangeError('I/Q capture has zero magnitude');
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
): { inPhase: Float64Array; quadrature: Float64Array } {
  const real = new Float64Array(inPhase.length);
  const imaginary = new Float64Array(inPhase.length);
  for (let index = 0; index < inPhase.length; index++) {
    const phase = -2 * Math.PI * center * index;
    const cosine = Math.cos(phase);
    const sine = Math.sin(phase);
    real[index] = inPhase[index]! * cosine - quadrature[index]! * sine;
    imaginary[index] = inPhase[index]! * sine + quadrature[index]! * cosine;
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
    throw new RangeError('dimensionless active-sequence extent is not finite');
  }
  const lastU = Math.floor(nextUp(capturedUEnd));
  const activeLength = lastU + 1;
  if (activeLength < 2) {
    throw new RangeError(
      'normalized active sequence has fewer than two samples; '
      + 'increase capture length or resample fraction',
    );
  }
  if (activeLength > MAX_INVARIANT_ACTIVE_LENGTH) {
    throw new RangeError(
      `normalized active sequence length ${activeLength} exceeds safety limit `
      + `${MAX_INVARIANT_ACTIVE_LENGTH}`,
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
      inPhase[lower]! * (1 - fraction) + inPhase[upper]! * fraction
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
): { inPhase: Float64Array; quadrature: Float64Array; rms: number } {
  let power = 0;
  for (let index = 0; index < inPhase.length; index++) {
    power += (
      inPhase[index]! * inPhase[index]!
      + quadrature[index]! * quadrature[index]!
    );
  }
  power /= inPhase.length;
  if (!Number.isFinite(power) || power <= 0) {
    throw new RangeError('resampled active sequence has invalid or zero power');
  }
  const rms = Math.sqrt(power);
  const real = new Float64Array(inPhase.length);
  const imaginary = new Float64Array(inPhase.length);
  for (let index = 0; index < inPhase.length; index++) {
    real[index] = inPhase[index]! / rms;
    imaginary[index] = quadrature[index]! / rms;
    if (!Number.isFinite(real[index]) || !Number.isFinite(imaginary[index])) {
      throw new RangeError('active-sequence RMS normalization produced non-finite values');
    }
  }
  return { inPhase: real, quadrature: imaginary, rms };
}

function evenPatchStarts(
  activeLength: number,
  patchLength: number,
  patchCount: number,
): number[] {
  const maximum = activeLength - patchLength;
  if (maximum < 0 || patchCount === 1) return new Array<number>(patchCount).fill(0);
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
  const starts = evenPatchStarts(inPhase.length, patchLength, patchCount);
  const packedLength = patchLength * patchCount;
  const packedInPhase = new Float64Array(packedLength);
  const packedQuadrature = new Float64Array(packedLength);
  const repeated = inPhase.length < patchLength;
  const repeatCount = repeated ? Math.ceil(patchLength / inPhase.length) : 1;
  for (let patch = 0; patch < patchCount; patch++) {
    const start = starts[patch]!;
    const outputOffset = patch * patchLength;
    for (let index = 0; index < patchLength; index++) {
      const source = repeated ? index % inPhase.length : start + index;
      // Python's final `.astype(np.float32)` is part of the packed ABI.
      packedInPhase[outputOffset + index] = Math.fround(inPhase[source]!);
      packedQuadrature[outputOffset + index] = Math.fround(quadrature[source]!);
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

/** Convert one raw complex capture to invariant patches and raw model features. */
export function preprocessInvariantPatches(
  inPhase: Float64Array,
  quadrature: Float64Array,
  options: InvariantPatchPreprocessOptions = {},
): InvariantPatchPreprocessResult {
  const patchLength = positiveInteger(
    options.patchLength ?? DEFAULT_INVARIANT_PATCH_LENGTH,
    'patchLength',
  );
  const patchCount = positiveInteger(
    options.patchCount ?? DEFAULT_INVARIANT_PATCH_COUNT,
    'patchCount',
  );
  const packedLength = patchLength * patchCount;
  if (packedLength > MAX_INVARIANT_PACKED_LENGTH) {
    throw new RangeError(
      `packed patch length ${packedLength} exceeds safety limit `
      + `${MAX_INVARIANT_PACKED_LENGTH}`,
    );
  }
  const targetFrac = finiteFraction(
    options.targetFrac ?? DEFAULT_PARAMS.targetFrac,
    'targetFrac',
  );
  const estimatorBase = options.estimatorParams ?? DEFAULT_PARAMS;
  if ((estimatorBase.version ?? 'linear-v1') !== INVARIANT_PATCH_ESTIMATOR_VERSION) {
    throw new RangeError(
      `invariant patch v1 requires ${INVARIANT_PATCH_ESTIMATOR_VERSION}`,
    );
  }
  const nfft = positiveInteger(options.nfft ?? estimatorBase.nfft, 'nfft');
  if (
    nfft < 2 * estimatorBase.smooth + 1
    || nfft % 2 !== 0
    || (nfft & (nfft - 1)) !== 0
  ) {
    throw new RangeError(
      `nfft must be a power of two >= ${2 * estimatorBase.smooth + 1}`,
    );
  }
  const estimatorParams: PreprocessParams = {
    ...estimatorBase,
    version: INVARIANT_PATCH_ESTIMATOR_VERSION,
    nfft,
    targetFrac,
  };

  if (inPhase.length < 2 || inPhase.length !== quadrature.length) {
    throw new RangeError('expected equal I/Q channels with at least two samples');
  }
  let detectedCenter: number | null = null;
  let detectedBw: number | null = null;
  if (options.forceCenter === undefined || options.forceBw === undefined) {
    const detected = estimateBand(inPhase, quadrature, estimatorParams);
    detectedCenter = detected.center;
    detectedBw = detected.bw;
  }
  const center = wrapCycles(options.forceCenter ?? detectedCenter!);
  const bw = finiteFraction(options.forceBw ?? detectedBw!, 'bandwidth');
  const resampleFrac = finiteFraction(
    options.forceResampleFrac ?? bw,
    'resampleFrac',
  );

  const peak = peakNormalize(inPhase, quadrature);
  const converted = baseband(peak.inPhase, peak.quadrature, center);
  const resampled = resampleOnU(
    converted.inPhase,
    converted.quadrature,
    resampleFrac,
    targetFrac,
  );
  const active = rmsNormalize(resampled.inPhase, resampled.quadrature);
  const computedFeatures = iqFeatures(active.inPhase, active.quadrature);
  if (computedFeatures.length !== 12) {
    throw new Error('active-sequence iqFeatures returned malformed geometry');
  }
  const rawFeatures = new Float64Array(computedFeatures.length);
  for (let index = 0; index < computedFeatures.length; index++) {
    if (!Number.isFinite(computedFeatures[index])) {
      throw new RangeError('active-sequence iqFeatures returned non-finite values');
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
      version: INVARIANT_PATCH_PREPROCESS_VERSION,
      estimator_version: INVARIANT_PATCH_ESTIMATOR_VERSION,
      raw_length: inPhase.length,
      raw_peak: peak.peak,
      detected_center: detectedCenter,
      detected_bw: detectedBw,
      center,
      bw,
      resample_frac: resampleFrac,
      target_frac: targetFrac,
      nfft,
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
      feature_source: 'complete RMS-normalized active sequence before patch repetition',
    },
  };
}
