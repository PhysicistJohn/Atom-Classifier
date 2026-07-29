/**
 * Trusted-current physical-time canonicalization for the isolated v5 lineage.
 *
 * The component reads only the causal first 4,096 complex input samples and
 * emits exactly 4,096 samples on the grid
 *
 *   t[m] = m / (2 * nativeSampleRateHz), m = 0, ..., 4095.
 *
 * Its fractional input position is therefore
 *
 *   p[m] = m * sampleRateHz / (2 * nativeSampleRateHz).
 *
 * The accepted rate ratio is [1, 2], so every position is within the consumed
 * prefix. Interpolation is a six-tap quintic Lagrange FIR implemented only with
 * scalar arithmetic. There is no FFT, learned state, profile, or class input.
 */

export const TRUSTED_CURRENT_CANONICALIZER_VERSION =
  'trusted-current-physical-time-v1' as const;
export const TRUSTED_CURRENT_CANONICALIZER_SCOPE =
  'trusted-current' as const;
export const TRUSTED_CURRENT_CONSUMED_INPUT_SAMPLES = 4096;
export const TRUSTED_CURRENT_OUTPUT_SAMPLES = 4096;
export const TRUSTED_CURRENT_TARGET_NATIVE_RATE_MULTIPLIER = 2;
export const TRUSTED_CURRENT_MIN_SAMPLE_RATE_RATIO = 1;
export const TRUSTED_CURRENT_MAX_SAMPLE_RATE_RATIO = 2;
export const TRUSTED_CURRENT_INTERPOLATION =
  'six-tap quintic Lagrange FIR' as const;

export interface TrustedCurrentGeometry {
  sampleRateHz: number;
  nativeSampleRateHz: number;
}

export interface TrustedCurrentCanonicalizationContext {
  version: typeof TRUSTED_CURRENT_CANONICALIZER_VERSION;
  trust_scope: typeof TRUSTED_CURRENT_CANONICALIZER_SCOPE;
  input_length: number;
  consumed_input_samples: number;
  ignored_tail_samples: number;
  output_samples: number;
  sample_rate_hz: number;
  native_sample_rate_hz: number;
  sample_rate_ratio: number;
  minimum_sample_rate_ratio: number;
  maximum_sample_rate_ratio: number;
  target_native_rate_multiplier: number;
  target_sample_rate_hz: number;
  input_samples_per_output: number;
  maximum_input_position: number;
  input_rounding: 'IEEE-754 float32 before interpolation';
  interpolation: typeof TRUSTED_CURRENT_INTERPOLATION;
  interpolation_taps: 6;
  uses_frequency_transform: false;
  uses_selected_profile_or_class: false;
}

export interface TrustedCurrentCanonicalWindow {
  inPhase: Float32Array;
  quadrature: Float32Array;
  context: TrustedCurrentCanonicalizationContext;
}

export interface TrustedCurrentCanonicalizerMetadata {
  version: typeof TRUSTED_CURRENT_CANONICALIZER_VERSION;
  trust_scope: typeof TRUSTED_CURRENT_CANONICALIZER_SCOPE;
  consumed_input_samples: number;
  output_samples: number;
  target_native_rate_multiplier: number;
  minimum_sample_rate_ratio: number;
  maximum_sample_rate_ratio: number;
  physical_time_grid: string;
  input_position_grid: string;
  input_rounding: 'IEEE-754 float32 before interpolation';
  interpolation: typeof TRUSTED_CURRENT_INTERPOLATION;
  interpolation_taps: 6;
  boundary_rule: string;
  output_dtype: 'complex64';
  uses_frequency_transform: false;
  uses_selected_profile_or_class: false;
}

function validateRate(value: number, name: string): number {
  if (!Number.isFinite(value) || value <= 0) {
    throw new RangeError(`${name} must be finite and positive`);
  }
  return value;
}

function validatedPrefix(
  inPhase: ArrayLike<number>,
  quadrature: ArrayLike<number>,
): {
  inPhase: Float32Array;
  quadrature: Float32Array;
} {
  if (
    inPhase.length < TRUSTED_CURRENT_CONSUMED_INPUT_SAMPLES
    || quadrature.length !== inPhase.length
  ) {
    throw new RangeError(
      'trusted-current canonicalization requires equal I/Q channels '
      + `with at least ${TRUSTED_CURRENT_CONSUMED_INPUT_SAMPLES} samples`,
    );
  }
  const prefixInPhase = new Float32Array(
    TRUSTED_CURRENT_CONSUMED_INPUT_SAMPLES,
  );
  const prefixQuadrature = new Float32Array(
    TRUSTED_CURRENT_CONSUMED_INPUT_SAMPLES,
  );
  for (
    let index = 0;
    index < TRUSTED_CURRENT_CONSUMED_INPUT_SAMPLES;
    index++
  ) {
    const real = inPhase[index];
    const imaginary = quadrature[index];
    if (
      !Number.isFinite(real)
      || !Number.isFinite(imaginary)
    ) {
      throw new RangeError('causal I/Q prefix contains NaN or infinity');
    }
    const roundedReal = Math.fround(real!);
    const roundedImaginary = Math.fround(imaginary!);
    if (
      !Number.isFinite(roundedReal)
      || !Number.isFinite(roundedImaginary)
    ) {
      throw new RangeError('causal I/Q prefix exceeds float32 range');
    }
    prefixInPhase[index] = roundedReal;
    prefixQuadrature[index] = roundedImaginary;
  }
  return {
    inPhase: prefixInPhase,
    quadrature: prefixQuadrature,
  };
}

/**
 * Canonicalize one trusted-current I/Q prefix onto the fixed 2x-native grid.
 *
 * Samples beyond index 4,095 are deliberately neither validated nor read.
 */
export function canonicalizeTrustedCurrentGeometry(
  inPhase: ArrayLike<number>,
  quadrature: ArrayLike<number>,
  geometry: TrustedCurrentGeometry,
): TrustedCurrentCanonicalWindow {
  const prefix = validatedPrefix(inPhase, quadrature);
  const sampleRateHz = validateRate(
    geometry.sampleRateHz,
    'sampleRateHz',
  );
  const nativeSampleRateHz = validateRate(
    geometry.nativeSampleRateHz,
    'nativeSampleRateHz',
  );
  const sampleRateRatio = sampleRateHz / nativeSampleRateHz;
  if (
    !Number.isFinite(sampleRateRatio)
    || sampleRateRatio < TRUSTED_CURRENT_MIN_SAMPLE_RATE_RATIO
    || sampleRateRatio > TRUSTED_CURRENT_MAX_SAMPLE_RATE_RATIO
  ) {
    throw new RangeError(
      'sampleRateHz/nativeSampleRateHz must lie in [1, 2]',
    );
  }
  const targetSampleRateHz = (
    TRUSTED_CURRENT_TARGET_NATIVE_RATE_MULTIPLIER
    * nativeSampleRateHz
  );
  if (!Number.isFinite(targetSampleRateHz)) {
    throw new RangeError('2 * nativeSampleRateHz must be finite');
  }
  const inputSamplesPerOutput = (
    sampleRateRatio
    / TRUSTED_CURRENT_TARGET_NATIVE_RATE_MULTIPLIER
  );
  const maximumInputPosition = (
    (TRUSTED_CURRENT_OUTPUT_SAMPLES - 1) * inputSamplesPerOutput
  );
  if (
    !Number.isFinite(maximumInputPosition)
    || maximumInputPosition < 0
    || maximumInputPosition
      > TRUSTED_CURRENT_CONSUMED_INPUT_SAMPLES - 1
  ) {
    throw new Error('validated rate contract escaped the causal prefix');
  }

  const outputInPhase = new Float32Array(
    TRUSTED_CURRENT_OUTPUT_SAMPLES,
  );
  const outputQuadrature = new Float32Array(
    TRUSTED_CURRENT_OUTPUT_SAMPLES,
  );
  for (
    let outputIndex = 0;
    outputIndex < TRUSTED_CURRENT_OUTPUT_SAMPLES;
    outputIndex++
  ) {
    const position = outputIndex * inputSamplesPerOutput;
    const integer = Math.floor(position);
    const fraction = position - integer;
    if (fraction === 0) {
      outputInPhase[outputIndex] = prefix.inPhase[integer]!;
      outputQuadrature[outputIndex] = prefix.quadrature[integer]!;
      continue;
    }

    let start: number;
    if (integer < 2) {
      start = 0;
    } else if (
      integer + 3 >= TRUSTED_CURRENT_CONSUMED_INPUT_SAMPLES
    ) {
      start = TRUSTED_CURRENT_CONSUMED_INPUT_SAMPLES - 6;
    } else {
      start = integer - 2;
    }
    const q = position - start;
    const qMinus1 = q - 1;
    const qMinus2 = q - 2;
    const qMinus3 = q - 3;
    const qMinus4 = q - 4;
    const qMinus5 = q - 5;
    const w0 = -(
      qMinus1 * qMinus2 * qMinus3 * qMinus4 * qMinus5
    ) / 120;
    const w1 = (
      q * qMinus2 * qMinus3 * qMinus4 * qMinus5
    ) / 24;
    const w2 = -(
      q * qMinus1 * qMinus3 * qMinus4 * qMinus5
    ) / 12;
    const w3 = (
      q * qMinus1 * qMinus2 * qMinus4 * qMinus5
    ) / 12;
    const w4 = -(
      q * qMinus1 * qMinus2 * qMinus3 * qMinus5
    ) / 24;
    const w5 = (
      q * qMinus1 * qMinus2 * qMinus3 * qMinus4
    ) / 120;
    const source0 = start;
    const source1 = start + 1;
    const source2 = start + 2;
    const source3 = start + 3;
    const source4 = start + 4;
    const source5 = start + 5;
    outputInPhase[outputIndex] = Math.fround(
      w0 * prefix.inPhase[source0]!
      + w1 * prefix.inPhase[source1]!
      + w2 * prefix.inPhase[source2]!
      + w3 * prefix.inPhase[source3]!
      + w4 * prefix.inPhase[source4]!
      + w5 * prefix.inPhase[source5]!
    );
    outputQuadrature[outputIndex] = Math.fround(
      w0 * prefix.quadrature[source0]!
      + w1 * prefix.quadrature[source1]!
      + w2 * prefix.quadrature[source2]!
      + w3 * prefix.quadrature[source3]!
      + w4 * prefix.quadrature[source4]!
      + w5 * prefix.quadrature[source5]!
    );
  }

  return {
    inPhase: outputInPhase,
    quadrature: outputQuadrature,
    context: {
      version: TRUSTED_CURRENT_CANONICALIZER_VERSION,
      trust_scope: TRUSTED_CURRENT_CANONICALIZER_SCOPE,
      input_length: inPhase.length,
      consumed_input_samples:
        TRUSTED_CURRENT_CONSUMED_INPUT_SAMPLES,
      ignored_tail_samples: (
        inPhase.length - TRUSTED_CURRENT_CONSUMED_INPUT_SAMPLES
      ),
      output_samples: TRUSTED_CURRENT_OUTPUT_SAMPLES,
      sample_rate_hz: sampleRateHz,
      native_sample_rate_hz: nativeSampleRateHz,
      sample_rate_ratio: sampleRateRatio,
      minimum_sample_rate_ratio:
        TRUSTED_CURRENT_MIN_SAMPLE_RATE_RATIO,
      maximum_sample_rate_ratio:
        TRUSTED_CURRENT_MAX_SAMPLE_RATE_RATIO,
      target_native_rate_multiplier:
        TRUSTED_CURRENT_TARGET_NATIVE_RATE_MULTIPLIER,
      target_sample_rate_hz: targetSampleRateHz,
      input_samples_per_output: inputSamplesPerOutput,
      maximum_input_position: maximumInputPosition,
      input_rounding: 'IEEE-754 float32 before interpolation',
      interpolation: TRUSTED_CURRENT_INTERPOLATION,
      interpolation_taps: 6,
      uses_frequency_transform: false,
      uses_selected_profile_or_class: false,
    },
  };
}

export function trustedCurrentCanonicalizerMetadata():
    TrustedCurrentCanonicalizerMetadata {
  return {
    version: TRUSTED_CURRENT_CANONICALIZER_VERSION,
    trust_scope: TRUSTED_CURRENT_CANONICALIZER_SCOPE,
    consumed_input_samples: TRUSTED_CURRENT_CONSUMED_INPUT_SAMPLES,
    output_samples: TRUSTED_CURRENT_OUTPUT_SAMPLES,
    target_native_rate_multiplier:
      TRUSTED_CURRENT_TARGET_NATIVE_RATE_MULTIPLIER,
    minimum_sample_rate_ratio:
      TRUSTED_CURRENT_MIN_SAMPLE_RATE_RATIO,
    maximum_sample_rate_ratio:
      TRUSTED_CURRENT_MAX_SAMPLE_RATE_RATIO,
    physical_time_grid:
      'm / (2 * native_sample_rate_hz), m=0..4095',
    input_position_grid:
      'm * sample_rate_hz / (2 * native_sample_rate_hz), m=0..4095',
    input_rounding: 'IEEE-754 float32 before interpolation',
    interpolation: TRUSTED_CURRENT_INTERPOLATION,
    interpolation_taps: 6,
    boundary_rule:
      'centred six-point stencil; first/last six prefix samples at edges',
    output_dtype: 'complex64',
    uses_frequency_transform: false,
    uses_selected_profile_or_class: false,
  };
}
