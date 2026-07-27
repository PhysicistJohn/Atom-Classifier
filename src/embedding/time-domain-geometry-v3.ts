/**
 * Exact runtime port of `training/time_domain_geometry.py`.
 *
 * This is deliberately isolated from the frozen hybrid-v3/Welch frontend. It
 * estimates occupied-band pose only from non-zero time-domain autocorrelation
 * lags. No frequency transform, spectral grid, filterbank, or learned
 * calibration is used.
 */

export const TIME_DOMAIN_GEOMETRY_VERSION = 'time-correlation-v1' as const;
export const TIME_DOMAIN_MIN_BANDWIDTH = 1 / 512;
export const TIME_DOMAIN_FULL_BANDWIDTH = 0.95;
export const TIME_DOMAIN_BASE_LAGS = [1, 2, 4, 8, 16] as const;
export const TIME_DOMAIN_MAX_LAG = 32;
export const TIME_DOMAIN_MAX_UNAMBIGUOUS_LAG_ARC = 0.45;
export const TIME_DOMAIN_MIN_LAG_SIGNAL_TO_SAMPLING_NOISE = 4;

const CORRELATION_EPSILON_FACTOR = 64;

interface ComplexValue {
  real: number;
  imaginary: number;
}

export interface TimeDomainGeometryContext {
  version: typeof TIME_DOMAIN_GEOMETRY_VERSION;
  raw_length: number;
  raw_peak: number;
  mean_power_after_peak_normalization: number;
  lag1_real: number;
  lag1_imag: number;
  lag2_real: number;
  lag2_imag: number;
  relative_lag1_magnitude: number;
  lag_quotient: number | null;
  clipped_lag_quotient: number | null;
  unclipped_equivalent_bandwidth: number | null;
  base_lag_one_bandwidth: number | null;
  selected_base_lag: number | null;
  minimum_bandwidth: number;
  full_bandwidth: number;
  bandwidth_clipped_to_minimum: boolean;
  bandwidth_clipped_to_full: boolean;
  degenerate_full_band_fallback: boolean;
  uses_frequency_transform: false;
}

export interface TimeDomainBandEstimate {
  center: number;
  bandwidth: number;
  context: TimeDomainGeometryContext;
}

export interface TimeDomainGeometryOptions {
  minBandwidth?: number;
  fullBandwidth?: number;
}

export interface TimeDomainGeometryMetadata {
  version: typeof TIME_DOMAIN_GEOMETRY_VERSION;
  method: 'multiscale non-zero-lag autocorrelation quotient';
  base_lags: number[];
  maximum_lag: number;
  max_unambiguous_lag_arc: number;
  min_lag_signal_to_sampling_noise: number;
  minimum_bandwidth: number;
  full_bandwidth: number;
  uses_frequency_transform: false;
  learned_calibration: false;
}

function finiteFraction(
  value: number,
  name: string,
  inclusiveZero = false,
): number {
  const lowerOk = inclusiveZero ? value >= 0 : value > 0;
  if (!Number.isFinite(value) || !lowerOk || value > 1) {
    const interval = inclusiveZero ? '[0, 1]' : '(0, 1]';
    throw new RangeError(`${name} must be a finite fraction in ${interval}`);
  }
  return value;
}

function validateCapture(
  inPhase: ArrayLike<number>,
  quadrature: ArrayLike<number>,
): void {
  if (inPhase.length < 3 || inPhase.length !== quadrature.length) {
    throw new RangeError(
      'expected equal I/Q channels with at least three samples',
    );
  }
  for (let index = 0; index < inPhase.length; index++) {
    if (
      !Number.isFinite(inPhase[index])
      || !Number.isFinite(quadrature[index])
    ) {
      throw new RangeError('I/Q capture contains NaN or infinity');
    }
  }
}

function wrapCycles(value: number): number {
  return value - Math.floor(value + 0.5);
}

function peakNormalize(
  inPhase: ArrayLike<number>,
  quadrature: ArrayLike<number>,
): {
  inPhase: Float64Array;
  quadrature: Float64Array;
  peak: number;
} {
  let peak = 0;
  for (let index = 0; index < inPhase.length; index++) {
    peak = Math.max(
      peak,
      Math.hypot(inPhase[index]!, quadrature[index]!),
    );
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

function nonzeroLag(
  inPhase: Float64Array,
  quadrature: Float64Array,
  lag: number,
): ComplexValue {
  let real = 0;
  let imaginary = 0;
  const count = inPhase.length - lag;
  for (let index = 0; index < count; index++) {
    const leftReal = inPhase[index]!;
    const leftImaginary = quadrature[index]!;
    const rightReal = inPhase[index + lag]!;
    const rightImaginary = quadrature[index + lag]!;
    real += (
      leftReal * rightReal
      + leftImaginary * rightImaginary
    );
    imaginary += (
      leftReal * rightImaginary
      - leftImaginary * rightReal
    );
  }
  return {
    real: real / count,
    imaginary: imaginary / count,
  };
}

function magnitude(value: ComplexValue): number {
  return Math.hypot(value.real, value.imaginary);
}

function quotientAfterCarrierRemoval(
  doubled: ComplexValue,
  base: ComplexValue,
): number {
  const baseMagnitude = magnitude(base);
  const unitReal = base.real / baseMagnitude;
  const unitImaginary = base.imaginary / baseMagnitude;
  const squaredReal = unitReal * unitReal - unitImaginary * unitImaginary;
  const squaredImaginary = 2 * unitReal * unitImaginary;
  // Re(doubled * conj(unitCarrier ** 2)).
  return (
    doubled.real * squaredReal
    + doubled.imaginary * squaredImaginary
  ) / baseMagnitude;
}

function clip(value: number, lower: number, upper: number): number {
  return Math.min(upper, Math.max(lower, value));
}

/** Mirror of `minimum_bandwidth_for_active_span`. */
export function minimumTimeDomainBandwidthForActiveSpan(
  rawLength: number,
  patchLength: number,
  targetFrac: number,
  fullBandwidth = TIME_DOMAIN_FULL_BANDWIDTH,
): number {
  if (!Number.isInteger(rawLength) || rawLength < 3) {
    throw new RangeError('rawLength must be an integer of at least three');
  }
  if (!Number.isInteger(patchLength) || patchLength <= 0) {
    throw new RangeError('patchLength must be a positive integer');
  }
  const target = finiteFraction(targetFrac, 'targetFrac');
  const full = finiteFraction(fullBandwidth, 'fullBandwidth');
  const required = (patchLength - 1) * target / (rawLength - 1);
  return clip(required, Number.EPSILON, full);
}

/**
 * Estimate center frequency and equivalent occupied bandwidth from time-domain
 * autocorrelation lags only.
 */
export function estimateTimeDomainBand(
  inPhase: ArrayLike<number>,
  quadrature: ArrayLike<number>,
  options: TimeDomainGeometryOptions = {},
): TimeDomainBandEstimate {
  const minimum = finiteFraction(
    options.minBandwidth ?? TIME_DOMAIN_MIN_BANDWIDTH,
    'minBandwidth',
  );
  const full = finiteFraction(
    options.fullBandwidth ?? TIME_DOMAIN_FULL_BANDWIDTH,
    'fullBandwidth',
  );
  if (minimum > full) {
    throw new RangeError('minBandwidth may not exceed fullBandwidth');
  }
  validateCapture(inPhase, quadrature);
  const normalized = peakNormalize(inPhase, quadrature);

  let meanPower = 0;
  for (let index = 0; index < normalized.inPhase.length; index++) {
    meanPower += (
      normalized.inPhase[index]! * normalized.inPhase[index]!
      + normalized.quadrature[index]! * normalized.quadrature[index]!
    );
  }
  meanPower /= normalized.inPhase.length;
  if (!Number.isFinite(meanPower) || meanPower <= 0) {
    throw new RangeError('peak-normalized I/Q has invalid or zero power');
  }

  const lags = new Map<number, ComplexValue>();
  for (const lag of [1, 2, 4, 8, 16, 32]) {
    if (lag < normalized.inPhase.length) {
      lags.set(
        lag,
        nonzeroLag(normalized.inPhase, normalized.quadrature, lag),
      );
    }
  }
  const lag1 = lags.get(1)!;
  const lag2 = lags.get(2)!;
  const lag1Magnitude = magnitude(lag1);
  const relativeLag1 = lag1Magnitude / meanPower;
  const numericalFloor = (
    CORRELATION_EPSILON_FACTOR * Number.EPSILON * meanPower
  );

  if (lag1Magnitude <= numericalFloor) {
    return {
      center: 0,
      bandwidth: full,
      context: {
        version: TIME_DOMAIN_GEOMETRY_VERSION,
        raw_length: normalized.inPhase.length,
        raw_peak: normalized.peak,
        mean_power_after_peak_normalization: meanPower,
        lag1_real: lag1.real,
        lag1_imag: lag1.imaginary,
        lag2_real: lag2.real,
        lag2_imag: lag2.imaginary,
        relative_lag1_magnitude: relativeLag1,
        lag_quotient: null,
        clipped_lag_quotient: null,
        unclipped_equivalent_bandwidth: null,
        base_lag_one_bandwidth: null,
        selected_base_lag: null,
        minimum_bandwidth: minimum,
        full_bandwidth: full,
        bandwidth_clipped_to_minimum: false,
        bandwidth_clipped_to_full: false,
        degenerate_full_band_fallback: true,
        uses_frequency_transform: false,
      },
    };
  }

  const preliminaryCenter = wrapCycles(
    Math.atan2(lag1.imaginary, lag1.real) / (2 * Math.PI),
  );
  const lagOneQuotient = quotientAfterCarrierRemoval(lag2, lag1);
  const lagOneClipped = clip(lagOneQuotient, -1, 1);
  const baseBandwidth = Math.acos(lagOneClipped) / Math.PI;

  let selectedLag = 1;
  let center = preliminaryCenter;
  let rawBandwidth = baseBandwidth;
  let selectedQuotient = lagOneQuotient;
  let selectedClippedQuotient = lagOneClipped;

  for (const baseLag of [16, 8, 4, 2]) {
    const doubled = lags.get(2 * baseLag);
    const base = lags.get(baseLag);
    if (doubled === undefined || base === undefined) continue;
    const baseMagnitude = magnitude(base);
    const relative = baseMagnitude / meanPower;
    const samplingFloor = (
      TIME_DOMAIN_MIN_LAG_SIGNAL_TO_SAMPLING_NOISE
      / Math.sqrt(normalized.inPhase.length - baseLag)
    );
    if (
      baseLag * baseBandwidth > TIME_DOMAIN_MAX_UNAMBIGUOUS_LAG_ARC
      || relative < samplingFloor
    ) {
      continue;
    }

    const candidateQuotient = quotientAfterCarrierRemoval(doubled, base);
    const candidateClipped = clip(candidateQuotient, -1, 1);
    const candidateBandwidth = (
      Math.acos(candidateClipped) / (Math.PI * baseLag)
    );
    const rootZero = (
      Math.atan2(base.imaginary, base.real) / (2 * Math.PI * baseLag)
    );
    let bestRoot = wrapCycles(rootZero);
    let bestDistance = Math.abs(
      wrapCycles(bestRoot - preliminaryCenter),
    );
    for (let branch = 1; branch < baseLag; branch++) {
      const candidate = wrapCycles(rootZero + branch / baseLag);
      const distance = Math.abs(
        wrapCycles(candidate - preliminaryCenter),
      );
      if (distance < bestDistance) {
        bestRoot = candidate;
        bestDistance = distance;
      }
    }
    center = bestRoot;
    rawBandwidth = candidateBandwidth;
    selectedLag = baseLag;
    selectedQuotient = candidateQuotient;
    selectedClippedQuotient = candidateClipped;
    break;
  }

  const bandwidth = clip(rawBandwidth, minimum, full);
  return {
    center,
    bandwidth,
    context: {
      version: TIME_DOMAIN_GEOMETRY_VERSION,
      raw_length: normalized.inPhase.length,
      raw_peak: normalized.peak,
      mean_power_after_peak_normalization: meanPower,
      lag1_real: lag1.real,
      lag1_imag: lag1.imaginary,
      lag2_real: lag2.real,
      lag2_imag: lag2.imaginary,
      relative_lag1_magnitude: relativeLag1,
      lag_quotient: selectedQuotient,
      clipped_lag_quotient: selectedClippedQuotient,
      unclipped_equivalent_bandwidth: rawBandwidth,
      base_lag_one_bandwidth: baseBandwidth,
      selected_base_lag: selectedLag,
      minimum_bandwidth: minimum,
      full_bandwidth: full,
      bandwidth_clipped_to_minimum: rawBandwidth < minimum,
      bandwidth_clipped_to_full: rawBandwidth > full,
      degenerate_full_band_fallback: false,
      uses_frequency_transform: false,
    },
  };
}

export function timeDomainGeometryMetadata(): TimeDomainGeometryMetadata {
  return {
    version: TIME_DOMAIN_GEOMETRY_VERSION,
    method: 'multiscale non-zero-lag autocorrelation quotient',
    base_lags: [...TIME_DOMAIN_BASE_LAGS],
    maximum_lag: TIME_DOMAIN_MAX_LAG,
    max_unambiguous_lag_arc: TIME_DOMAIN_MAX_UNAMBIGUOUS_LAG_ARC,
    min_lag_signal_to_sampling_noise:
      TIME_DOMAIN_MIN_LAG_SIGNAL_TO_SAMPLING_NOISE,
    minimum_bandwidth: TIME_DOMAIN_MIN_BANDWIDTH,
    full_bandwidth: TIME_DOMAIN_FULL_BANDWIDTH,
    uses_frequency_transform: false,
    learned_calibration: false,
  };
}
