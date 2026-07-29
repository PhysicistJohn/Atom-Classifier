import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  TRUSTED_CURRENT_CONSUMED_INPUT_SAMPLES,
  TRUSTED_CURRENT_OUTPUT_SAMPLES,
  canonicalizeTrustedCurrentGeometry,
  trustedCurrentCanonicalizerMetadata,
} from './trusted-current-geometry-canonicalizer-v5.js';

interface ParityCheckpoint {
  index: number;
  in_phase: number;
  quadrature: number;
}

interface ParityCase {
  name: string;
  sample_rate_hz: number;
  native_sample_rate_hz: number;
  output_cf32le_sha256: string;
  checkpoints: ParityCheckpoint[];
}

interface ParityFixture {
  schema: string;
  schema_version: number;
  synthetic_only: boolean;
  reads_held_or_sealed_payload: boolean;
  cases: ParityCase[];
}

const fixture = JSON.parse(
  readFileSync(
    new URL(
      './test-fixtures/trusted-current-geometry-canonicalizer-v5.json',
      import.meta.url,
    ),
    'utf8',
  ),
) as ParityFixture;

function parityInput(): {
  inPhase: Float32Array;
  quadrature: Float32Array;
} {
  const inPhase = new Float32Array(
    TRUSTED_CURRENT_CONSUMED_INPUT_SAMPLES,
  );
  const quadrature = new Float32Array(
    TRUSTED_CURRENT_CONSUMED_INPUT_SAMPLES,
  );
  for (
    let index = 0;
    index < TRUSTED_CURRENT_CONSUMED_INPUT_SAMPLES;
    index++
  ) {
    inPhase[index] = Math.fround(
      (((17 * index) % 257) - 128) / 64,
    );
    quadrature[index] = Math.fround(
      (((29 * index + 7) % 263) - 131) / 80,
    );
  }
  return { inPhase, quadrature };
}

function cf32leSha256(
  inPhase: Float32Array,
  quadrature: Float32Array,
): string {
  expect(inPhase.length).toBe(quadrature.length);
  const encoded = Buffer.allocUnsafe(inPhase.length * 8);
  for (let index = 0; index < inPhase.length; index++) {
    encoded.writeFloatLE(inPhase[index]!, index * 8);
    encoded.writeFloatLE(quadrature[index]!, index * 8 + 4);
  }
  return createHash('sha256').update(encoded).digest('hex');
}

function continuousSignal(
  length: number,
  sampleRateRatio: number,
): {
  inPhase: Float32Array;
  quadrature: Float32Array;
} {
  const inPhase = new Float32Array(length);
  const quadrature = new Float32Array(length);
  for (let index = 0; index < length; index++) {
    const time = index / sampleRateRatio;
    const firstPhase = 2 * Math.PI * 0.071 * time;
    const secondPhase = -2 * Math.PI * 0.133 * time + 0.2;
    const envelope = Math.cos(2 * Math.PI * 0.019 * time);
    inPhase[index] = Math.fround(
      0.72 * Math.cos(firstPhase)
      + 0.21 * Math.cos(secondPhase)
      + 0.08 * envelope,
    );
    quadrature[index] = Math.fround(
      0.72 * Math.sin(firstPhase)
      + 0.21 * Math.sin(secondPhase)
      + 0.03 * envelope,
    );
  }
  return { inPhase, quadrature };
}

function maximumComplexError(
  actualInPhase: ArrayLike<number>,
  actualQuadrature: ArrayLike<number>,
  expectedInPhase: ArrayLike<number>,
  expectedQuadrature: ArrayLike<number>,
): {
  maximum: number;
  rms: number;
} {
  expect(actualInPhase.length).toBe(expectedInPhase.length);
  expect(actualQuadrature.length).toBe(expectedQuadrature.length);
  let maximum = 0;
  let squared = 0;
  for (let index = 0; index < actualInPhase.length; index++) {
    const error = Math.hypot(
      actualInPhase[index]! - expectedInPhase[index]!,
      actualQuadrature[index]! - expectedQuadrature[index]!,
    );
    maximum = Math.max(maximum, error);
    squared += error * error;
  }
  return {
    maximum,
    rms: Math.sqrt(squared / actualInPhase.length),
  };
}

describe('trusted-current physical-time canonicalizer v5', () => {
  it('publishes the isolated FFT-free geometry-only contract', () => {
    const metadata = trustedCurrentCanonicalizerMetadata();
    expect(metadata).toMatchObject({
      version: 'trusted-current-physical-time-v1',
      trust_scope: 'trusted-current',
      consumed_input_samples: 4096,
      output_samples: 4096,
      target_native_rate_multiplier: 2,
      minimum_sample_rate_ratio: 1,
      maximum_sample_rate_ratio: 2,
      interpolation: 'six-tap quintic Lagrange FIR',
      interpolation_taps: 6,
      output_dtype: 'complex64',
      uses_frequency_transform: false,
      uses_selected_profile_or_class: false,
    });

    const source = readFileSync(
      new URL(
        './trusted-current-geometry-canonicalizer-v5.ts',
        import.meta.url,
      ),
      'utf8',
    );
    expect(source).not.toMatch(/\b(?:fft|stft)\s*\(/iu);
    expect(source).not.toMatch(
      /\bfrom\s+['"][^'"]*(?:fft|spectrum|spectral)[^'"]*['"]/iu,
    );
  });

  it('copies the prefix exactly when the input is already at 2x native', () => {
    const input = parityInput();
    const output = canonicalizeTrustedCurrentGeometry(
      input.inPhase,
      input.quadrature,
      {
        sampleRateHz: 16_000_000,
        nativeSampleRateHz: 8_000_000,
      },
    );
    expect(output.inPhase).toEqual(input.inPhase);
    expect(output.quadrature).toEqual(input.quadrature);
    expect(output.context.maximum_input_position).toBe(4095);
    expect(output.context.input_samples_per_output).toBe(1);
    expect(output.context.uses_frequency_transform).toBe(false);
    expect(output.context.uses_selected_profile_or_class).toBe(false);
  });

  it('canonicalizes smooth continuous signals across service rates', () => {
    const expected = continuousSignal(
      TRUSTED_CURRENT_OUTPUT_SAMPLES,
      2,
    );
    const outputs = [1, 1.25, 1.5, 2].map((ratio) => {
      const input = continuousSignal(
        TRUSTED_CURRENT_CONSUMED_INPUT_SAMPLES,
        ratio,
      );
      const output = canonicalizeTrustedCurrentGeometry(
        input.inPhase,
        input.quadrature,
        {
          sampleRateHz: ratio * 12_000_000,
          nativeSampleRateHz: 12_000_000,
        },
      );
      const error = maximumComplexError(
        output.inPhase,
        output.quadrature,
        expected.inPhase,
        expected.quadrature,
      );
      expect(error.maximum).toBeLessThan(0.0012);
      expect(error.rms).toBeLessThan(0.00023);
      expect(output.context.maximum_input_position).toBeLessThanOrEqual(
        TRUSTED_CURRENT_CONSUMED_INPUT_SAMPLES - 1,
      );
      return output;
    });
    const reference = outputs.at(-1)!;
    for (const output of outputs) {
      expect(
        maximumComplexError(
          output.inPhase,
          output.quadrature,
          reference.inPhase,
          reference.quadrature,
        ).maximum,
      ).toBeLessThan(0.0012);
    }
  });

  it('is bit-exactly invariant to length and tail content', () => {
    const prefix = parityInput();
    const firstInPhase = new Float32Array(4096 + 37);
    const firstQuadrature = new Float32Array(4096 + 37);
    const secondInPhase = new Float32Array(4096 + 811);
    const secondQuadrature = new Float32Array(4096 + 811);
    firstInPhase.set(prefix.inPhase);
    firstQuadrature.set(prefix.quadrature);
    secondInPhase.set(prefix.inPhase);
    secondQuadrature.set(prefix.quadrature);
    firstInPhase.fill(90, 4096);
    firstQuadrature.fill(70, 4096);
    secondInPhase.fill(-60, 4096);
    secondQuadrature.fill(-30, 4096);

    const base = canonicalizeTrustedCurrentGeometry(
      prefix.inPhase,
      prefix.quadrature,
      { sampleRateHz: 5_000_000, nativeSampleRateHz: 4_000_000 },
    );
    const first = canonicalizeTrustedCurrentGeometry(
      firstInPhase,
      firstQuadrature,
      { sampleRateHz: 5_000_000, nativeSampleRateHz: 4_000_000 },
    );
    const second = canonicalizeTrustedCurrentGeometry(
      secondInPhase,
      secondQuadrature,
      { sampleRateHz: 5_000_000, nativeSampleRateHz: 4_000_000 },
    );
    expect(base.inPhase).toEqual(first.inPhase);
    expect(base.quadrature).toEqual(first.quadrature);
    expect(first.inPhase).toEqual(second.inPhase);
    expect(first.quadrature).toEqual(second.quadrature);
    expect(base.context.ignored_tail_samples).toBe(0);
    expect(first.context.ignored_tail_samples).toBe(37);
    expect(second.context.ignored_tail_samples).toBe(811);
  });

  it('fails closed on malformed I/Q and untrusted geometry', () => {
    const input = parityInput();
    const invoke = (
      inPhase: ArrayLike<number>,
      quadrature: ArrayLike<number>,
      sampleRateHz: number,
      nativeSampleRateHz: number,
    ) => canonicalizeTrustedCurrentGeometry(
      inPhase,
      quadrature,
      { sampleRateHz, nativeSampleRateHz },
    );

    expect(() => invoke(
      input.inPhase.slice(0, 4095),
      input.quadrature.slice(0, 4095),
      1,
      1,
    )).toThrow(/at least 4096/u);
    expect(() => invoke(
      input.inPhase,
      input.quadrature.slice(0, 4095),
      1,
      1,
    )).toThrow(/equal I\/Q/u);
    const malformed = input.inPhase.slice();
    malformed[17] = Number.NaN;
    expect(() => invoke(
      malformed,
      input.quadrature,
      1,
      1,
    )).toThrow(/NaN or infinity/u);
    const oversized = Float64Array.from(input.inPhase);
    oversized[17] = Number.MAX_VALUE;
    expect(() => invoke(
      oversized,
      input.quadrature,
      1,
      1,
    )).toThrow(/exceeds float32 range/u);

    for (const [sampleRateHz, nativeSampleRateHz] of [
      [0, 1],
      [-1, 1],
      [Number.NaN, 1],
      [Number.POSITIVE_INFINITY, 1],
      [1, 0],
      [1, Number.NaN],
      [0.999, 1],
      [2.001, 1],
    ]) {
      expect(() => invoke(
        input.inPhase,
        input.quadrature,
        sampleRateHz!,
        nativeSampleRateHz!,
      )).toThrow();
    }
    expect(() => invoke(
      input.inPhase,
      input.quadrature,
      Number.MAX_VALUE * 0.9375,
      Number.MAX_VALUE * 0.75,
    )).toThrow(/must be finite/u);
  });

  it('is bit-identical to the shared Python parity fixture', () => {
    expect(fixture.schema).toBe(
      'atomos.v5.trusted-current-geometry-canonicalizer.parity',
    );
    expect(fixture.schema_version).toBe(1);
    expect(fixture.synthetic_only).toBe(true);
    expect(fixture.reads_held_or_sealed_payload).toBe(false);
    const input = parityInput();
    for (const parityCase of fixture.cases) {
      const output = canonicalizeTrustedCurrentGeometry(
        input.inPhase,
        input.quadrature,
        {
          sampleRateHz: parityCase.sample_rate_hz,
          nativeSampleRateHz: parityCase.native_sample_rate_hz,
        },
      );
      expect(
        cf32leSha256(output.inPhase, output.quadrature),
        parityCase.name,
      ).toBe(parityCase.output_cf32le_sha256);
      for (const checkpoint of parityCase.checkpoints) {
        expect(output.inPhase[checkpoint.index]).toBe(
          checkpoint.in_phase,
        );
        expect(output.quadrature[checkpoint.index]).toBe(
          checkpoint.quadrature,
        );
      }
    }
  });
});
