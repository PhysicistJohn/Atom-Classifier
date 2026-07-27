import { describe, expect, it } from 'vitest';
import { synthesizeAnalyticComplexIq } from '../../Atom-SignalLab-IQ/src/complex-iq.js';
import { waveformDescriptor } from '../../Atom-SignalLab-IQ/src/waveforms.js';
import {
  CLASSIFIER_RECEIVER_IQ_TRANSFORM_ALGORITHM,
  CLASSIFIER_RECEIVER_IQ_TRANSFORM_ID,
  acquireClassifierReceiverCleanIq,
  applyClassifierReceiverImpairments,
  sha256Cf32le,
  transformClassifierReceiverIq,
} from './classifier-receiver-iq.js';

describe('classifier receiver I/Q layering', () => {
  it('acquires a qualified fixed profile at native geometry before resampling', () => {
    const output = acquireClassifierReceiverCleanIq({
      profile: 'gsm-900-loaded-bcch',
      targetSampleRateHz: 2_000_000,
      captureBandwidthHz: 230_000,
      targetSampleCount: 257,
      targetStartSampleIndex: 0,
      realizationIndex: 0,
    });

    expect(output.bytes).toHaveLength(257 * 8);
    expect(output.source).toMatchObject({
      profile: 'gsm-900-loaded-bcch',
      qualification: 'independently-verified-digital-baseband',
      nativeSampleRateHz: 1_300_000,
      signalBandwidthHz: 200_000,
      replay: 'cyclic',
      nativePeriodSamples: 24_000,
    });
    expect(output.source.artifactSha256)
      .toBe(waveformDescriptor('gsm-900-loaded-bcch').assetSha256);
    expect(output.transform.sourceArtifactSha256)
      .toBe(output.source.artifactSha256);
    expect(output.source.sourceSamplesSha256)
      .toBe(output.transform.sourceSamplesSha256);
    expect(output.transform).toMatchObject({
      id: CLASSIFIER_RECEIVER_IQ_TRANSFORM_ID,
      algorithm: CLASSIFIER_RECEIVER_IQ_TRANSFORM_ALGORITHM,
      qualification: 'derived-from-independently-verified-digital-baseband',
      sourceSampleRateHz: 1_300_000,
      outputSampleRateHz: 2_000_000,
      outputSampleCount: 257,
      boundary: 'zero',
    });
    expect(output.transform.outputSamplesSha256).toBe(sha256Cf32le(output.bytes));
    expect(output.transform.outputSamplesSha256)
      .not.toBe(output.source.sourceSamplesSha256);
  });

  it('keeps intrinsic flexible-profile bandwidth distinct from receiver capture bandwidth', () => {
    const output = acquireClassifierReceiverCleanIq({
      profile: 'cw',
      targetSampleRateHz: 8_000_000,
      captureBandwidthHz: 2_000_000,
      targetSampleCount: 64,
      targetStartSampleIndex: 123,
      realizationIndex: 1,
    });

    expect(output.source).toMatchObject({
      qualification: 'analytic-or-standards-derived-parameterized-source',
      artifactSha256: null,
      nativeSampleRateHz: 8_000_000,
      signalBandwidthHz: 2_000,
      startSampleIndex: 123,
      sampleCount: 64,
      replay: 'continuous',
    });
    expect(output.transform.qualification)
      .toBe('receiver-transformed-parameterized-complex-baseband');
    expect(output.source.signalBandwidthHz).not.toBe(2_000_000);
  });

  it('preserves the requested cyclic artifact phase at native rate', () => {
    const output = acquireClassifierReceiverCleanIq({
      profile: 'gsm-900-loaded-bcch',
      targetSampleRateHz: 1_300_000,
      captureBandwidthHz: 200_000,
      targetSampleCount: 257,
      targetStartSampleIndex: 0,
      realizationIndex: 0,
    });
    const direct = synthesizeAnalyticComplexIq({
      profile: 'gsm-900-loaded-bcch',
      sampleRateHz: 1_300_000,
      bandwidthHz: 200_000,
      sampleCount: 257,
      startSampleIndex: 0,
    });

    expect(output.bytes).toEqual(direct);
    expect(output.source.startSampleIndex).toBe(24_000 - 16);
    expect(output.transform.outputStartSourcePosition).toBe(24_000);
  });

  it('rejects receiver geometry narrower than a flexible profile signal', () => {
    expect(() => acquireClassifierReceiverCleanIq({
      profile: 'cw',
      targetSampleRateHz: 8_000_000,
      captureBandwidthHz: 1_999,
      targetSampleCount: 64,
      targetStartSampleIndex: 0,
      realizationIndex: 0,
    })).toThrow(/must contain its 2000 Hz signal bandwidth/);
  });

  it('retunes and resamples a bounded Bluetooth packet without replaying it', () => {
    const output = acquireClassifierReceiverCleanIq({
      profile: 'bluetooth-le-advertising',
      targetSampleRateHz: 2_300_000,
      captureBandwidthHz: 2_000_000,
      targetSampleCount: 512,
      targetStartSampleIndex: 0,
      realizationIndex: 2,
    });
    const samples = decodeCf32le(output.bytes);
    const nonzeroSamples = samples.filter(([inPhase, quadrature]) =>
      Math.hypot(inPhase, quadrature) > 1e-5).length;

    expect(output.source).toMatchObject({
      nativeSampleRateHz: 80_000_000,
      signalBandwidthHz: 1_000_000,
      profileReferenceCenterHz: 2_426_000_000,
      rfReferenceCenterHz: 2_441_000_000,
      nativeCarrierOffsetHz: -15_000_000,
      sampleCount: 12_160,
      replay: 'one-shot',
      maxOneShotSamples: 12_160,
    });
    expect(output.source.sourceSamplesSha256).toBe(output.source.artifactSha256);
    expect(output.transform).toMatchObject({
      sourceSampleRateHz: 80_000_000,
      outputSampleRateHz: 2_300_000,
      sourceCarrierOffsetHz: -15_000_000,
      outputCarrierOffsetHz: 0,
      frequencyTranslationHz: 15_000_000,
      boundary: 'zero',
    });
    expect(output.transform.operations).toEqual([
      {
        kind: 'frequency-translate',
        algorithm: 'complex-rotator-v1',
        sourceCarrierOffsetHz: -15_000_000,
        outputCarrierOffsetHz: 0,
      },
      {
        kind: 'resample',
        algorithm: 'blackman-windowed-sinc-v1',
        sourceSampleRateHz: 80_000_000,
        outputSampleRateHz: 2_300_000,
        antiAliasCutoffHz: 1_092_500,
        zeroCrossings: 16,
      },
    ]);
    expect(nonzeroSamples).toBeGreaterThan(100);
    expect(nonzeroSamples).toBeLessThan(450);
  });

  it('resamples a tone while preserving its physical frequency', () => {
    const sourceRateHz = 48_000;
    const targetRateHz = 96_000;
    const toneHz = 3_000;
    const source = encodeCf32le(160, (index) => {
      const phase = 2 * Math.PI * toneHz * index / sourceRateHz;
      return [Math.cos(phase), Math.sin(phase)];
    });
    const output = transformClassifierReceiverIq({
      sourceBytes: source,
      sourceSampleRateHz: sourceRateHz,
      targetSampleRateHz: targetRateHz,
      targetSampleCount: 240,
      sourcePositionAtOutputZero: 16,
      outputQualification: 'receiver-transformed-parameterized-complex-baseband',
    });
    const samples = decodeCf32le(output.bytes);
    const expectedIncrement = 2 * Math.PI * toneHz / targetRateHz;
    let maximumError = 0;
    for (let index = 16; index < samples.length - 16; index += 1) {
      const previous = samples[index - 1]!;
      const current = samples[index]!;
      const observed = Math.atan2(
        previous[0] * current[1] - previous[1] * current[0],
        previous[0] * current[0] + previous[1] * current[1],
      );
      maximumError = Math.max(maximumError, Math.abs(observed - expectedIncrement));
    }
    expect(maximumError).toBeLessThan(2e-4);
    expect(output.receipt.kernelRadiusSourceSamples).toBe(16);
    expect(output.receipt.antiAliasCutoffHz).toBe(sourceRateHz / 2);
    expect(output.receipt.operations).toContainEqual({
      kind: 'resample',
      algorithm: 'blackman-windowed-sinc-v1',
      sourceSampleRateHz: sourceRateHz,
      outputSampleRateHz: targetRateHz,
      antiAliasCutoffHz: sourceRateHz / 2,
      zeroCrossings: 16,
    });
    expect(output.receipt.sourceSamplesSha256).toBe(sha256Cf32le(source));
    expect(output.receipt.outputSamplesSha256).toBe(sha256Cf32le(output.bytes));
  });

  it('acquires cyclic N-TM native support deterministically across synthesis chunks', () => {
    const request = {
      profile: 'lte-ntm',
      targetSampleRateHz: 1_000_000,
      captureBandwidthHz: 200_000,
      targetSampleCount: 35_000,
      targetStartSampleIndex: 12_345,
      realizationIndex: 7,
    } as const;
    const first = acquireClassifierReceiverCleanIq(request);
    const second = acquireClassifierReceiverCleanIq(request);

    expect(first.source).toMatchObject({
      nativeSampleRateHz: 1_920_000,
      signalBandwidthHz: 180_000,
      replay: 'cyclic',
      nativePeriodSamples: 19_200,
    });
    expect(first.source.sampleCount).toBeGreaterThan(65_536);
    expect(first.source.startSampleIndex).toBeGreaterThan(0);
    expect(first.source.sourceSamplesSha256)
      .toBe(first.transform.sourceSamplesSha256);
    expect(first.source.sourceSamplesSha256)
      .toBe(second.source.sourceSamplesSha256);
    expect(first.transform.outputSamplesSha256)
      .toBe(second.transform.outputSamplesSha256);
    expect(first.bytes).toEqual(second.bytes);
  });

  it('suppresses out-of-band energy before downsampling', () => {
    const sourceRateHz = 8_000_000;
    const targetRateHz = 1_000_000;
    const wantedHz = 100_000;
    const rejectedHz = 2_200_000;
    const source = encodeCf32le(5_000, (index) => {
      const wantedPhase = 2 * Math.PI * wantedHz * index / sourceRateHz;
      const rejectedPhase = 2 * Math.PI * rejectedHz * index / sourceRateHz;
      return [
        0.5 * Math.cos(wantedPhase) + 0.5 * Math.cos(rejectedPhase),
        0.5 * Math.sin(wantedPhase) + 0.5 * Math.sin(rejectedPhase),
      ];
    });
    const output = transformClassifierReceiverIq({
      sourceBytes: source,
      sourceSampleRateHz: sourceRateHz,
      targetSampleRateHz: targetRateHz,
      targetSampleCount: 512,
      sourcePositionAtOutputZero: 256,
      outputQualification: 'receiver-transformed-parameterized-complex-baseband',
    });
    const samples = decodeCf32le(output.bytes);
    const wantedMagnitude = correlationMagnitude(samples, wantedHz, targetRateHz);
    // 2.2 MHz would alias to +200 kHz without the anti-alias stage.
    const aliasMagnitude = correlationMagnitude(samples, 200_000, targetRateHz);

    expect(output.receipt.kernelRadiusSourceSamples).toBe(135);
    expect(aliasMagnitude / wantedMagnitude).toBeLessThan(0.02);
  });

  it('zero-bounds one-shot material instead of inventing cyclic recurrence', () => {
    const packet = encodeCf32le(8, () => [0.5, -0.25]);
    const output = transformClassifierReceiverIq({
      sourceBytes: packet,
      sourceSampleRateHz: 1_000_000,
      targetSampleRateHz: 1_000_000,
      targetSampleCount: 24,
      sourcePositionAtOutputZero: -8,
      outputQualification: 'derived-from-independently-verified-digital-baseband',
    });
    const samples = decodeCf32le(output.bytes);

    expect(samples.slice(0, 8)).toEqual(Array(8).fill([0, 0]));
    expect(samples.slice(8, 16)).toEqual(Array(8).fill([0.5, -0.25]));
    expect(samples.slice(16)).toEqual(Array(8).fill([0, 0]));
    expect(output.receipt.boundary).toBe('zero');
    expect(output.receipt.kernelRadiusSourceSamples).toBe(16);
    expect(output.receipt.antiAliasCutoffHz).toBe(500_000);
  });

  it('applies receiver impairments after transport deterministically', () => {
    const source = encodeCf32le(64, (index) => [
      Math.cos(2 * Math.PI * index / 16),
      Math.sin(2 * Math.PI * index / 16),
    ]);
    const impairments = {
      snrDb: 18,
      carrierFrequencyOffset: 0.01,
      iqGainImbalance: 0.04,
    } as const;
    const first = applyClassifierReceiverImpairments(source, impairments, 407);
    const second = applyClassifierReceiverImpairments(source, impairments, 407);
    const differentSeed = applyClassifierReceiverImpairments(source, impairments, 408);

    expect(first).toEqual(second);
    expect(first).not.toEqual(differentSeed);
    expect(first).not.toEqual(source);
  });
});

function encodeCf32le(
  sampleCount: number,
  sample: (index: number) => readonly [number, number],
): Uint8Array {
  const bytes = new Uint8Array(sampleCount * 8);
  const view = new DataView(bytes.buffer);
  for (let index = 0; index < sampleCount; index += 1) {
    const [inPhase, quadrature] = sample(index);
    view.setFloat32(index * 8, inPhase, true);
    view.setFloat32(index * 8 + 4, quadrature, true);
  }
  return bytes;
}

function decodeCf32le(bytes: Uint8Array): Array<readonly [number, number]> {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return Array.from({ length: bytes.byteLength / 8 }, (_, index) => [
    view.getFloat32(index * 8, true),
    view.getFloat32(index * 8 + 4, true),
  ] as const);
}

function correlationMagnitude(
  samples: ReadonlyArray<readonly [number, number]>,
  frequencyHz: number,
  sampleRateHz: number,
): number {
  let inPhase = 0;
  let quadrature = 0;
  for (let index = 0; index < samples.length; index += 1) {
    const [sampleInPhase, sampleQuadrature] = samples[index]!;
    const angle = -2 * Math.PI * frequencyHz * index / sampleRateHz;
    const cosine = Math.cos(angle);
    const sine = Math.sin(angle);
    inPhase += sampleInPhase * cosine - sampleQuadrature * sine;
    quadrature += sampleInPhase * sine + sampleQuadrature * cosine;
  }
  return Math.hypot(inPhase, quadrature) / samples.length;
}
