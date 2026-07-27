import { createHash } from 'node:crypto';
import {
  ANALYTIC_COMPLEX_IQ_BYTES_PER_SAMPLE,
  MAX_ANALYTIC_COMPLEX_IQ_SAMPLES,
  synthesizeAnalyticComplexIq,
} from '../../Atom-SignalLab-IQ/src/complex-iq.js';
import type { SynthesizedSignalProfile } from '../../Atom-SignalLab-IQ/src/contracts.js';
import {
  fixedDigitalProfileBinding,
  isFixedDigitalProfile,
  type FixedDigitalProfile,
  type FixedDigitalProfileBinding,
} from '../../Atom-SignalLab-IQ/src/fixed-digital-profile-binding.js';
import {
  IQ_RESAMPLER_ALGORITHM,
  IQ_RESAMPLER_NYQUIST_GUARD,
  IQ_RESAMPLER_ZERO_CROSSINGS,
  translateCf32leCarrier,
} from '../../Atom-SignalLab-IQ/src/iq-resampler.js';
import {
  applyReceiverImpairments,
  type ReceiverImpairments,
} from '../../Atom-SignalLab-IQ/src/impairments.js';
import { waveformDescriptor } from '../../Atom-SignalLab-IQ/src/waveforms.js';

export const CLASSIFIER_RECEIVER_IQ_TRANSFORM_ID =
  'classifier-receiver-iq-transport-v1' as const;
export const CLASSIFIER_RECEIVER_IQ_TRANSFORM_ALGORITHM =
  'deterministic-complex-rotator-blackman-windowed-sinc-resample-and-explicit-window-v1' as const;

export type ClassifierReceiverSourceQualification =
  | 'independently-verified-digital-baseband'
  | 'analytic-or-standards-derived-parameterized-source';

export type ClassifierReceiverOutputQualification =
  | 'derived-from-independently-verified-digital-baseband'
  | 'receiver-transformed-parameterized-complex-baseband';

export interface ClassifierReceiverSourceLineage {
  readonly profile: SynthesizedSignalProfile;
  readonly qualification: ClassifierReceiverSourceQualification;
  /**
   * Hash published by SignalLab for the complete qualified artifact. It is
   * deliberately null for parameterized/laboratory generators.
   */
  readonly artifactSha256: string | null;
  /** Hash of the exact native cf32le segment supplied to the receiver transform. */
  readonly sourceSamplesSha256: string;
  readonly nativeSampleRateHz: number;
  readonly signalBandwidthHz: number;
  readonly profileReferenceCenterHz: number;
  readonly rfReferenceCenterHz: number;
  /** Carrier coordinate inside the native zero-IF capture. */
  readonly nativeCarrierOffsetHz: number;
  readonly startSampleIndex: number;
  readonly sampleCount: number;
  /** SignalLab source-time behavior, independent of receiver capture windowing. */
  readonly replay: 'continuous' | 'cyclic' | 'one-shot';
  /** Exact source period when replay is cyclic. */
  readonly nativePeriodSamples?: number;
  /** Exact exclusive source bound when replay is one-shot. */
  readonly maxOneShotSamples?: number;
}

export interface ClassifierReceiverTransformReceipt {
  readonly receiptVersion: 1;
  readonly id: typeof CLASSIFIER_RECEIVER_IQ_TRANSFORM_ID;
  readonly algorithm: typeof CLASSIFIER_RECEIVER_IQ_TRANSFORM_ALGORITHM;
  readonly qualification: ClassifierReceiverOutputQualification;
  readonly sourceArtifactSha256: string | null;
  readonly sourceStartSample: number;
  readonly sourceSampleRateHz: number;
  readonly outputSampleRateHz: number;
  readonly sourceSampleCount: number;
  readonly outputSampleCount: number;
  readonly kernelRadiusSourceSamples: number;
  readonly sourceCarrierOffsetHz: number;
  readonly outputCarrierOffsetHz: 0;
  /** Mixer applied before rate conversion; target output is centered at zero. */
  readonly frequencyTranslationHz: number;
  /**
   * Native-domain coordinate corresponding to output sample zero.
   * A negative value places a bounded one-shot packet later in a receiver
   * capture; samples outside the source buffer are explicit zero padding.
   */
  readonly outputStartSourcePosition: number;
  readonly boundary: 'zero';
  readonly antiAliasCutoffHz: number;
  readonly sourceSamplesSha256: string;
  readonly outputSamplesSha256: string;
  readonly operations: readonly ClassifierReceiverTransformOperation[];
}

export type ClassifierReceiverTransformOperation =
  | {
      readonly kind: 'frequency-translate';
      readonly algorithm: 'complex-rotator-v1';
      readonly sourceCarrierOffsetHz: number;
      readonly outputCarrierOffsetHz: 0;
    }
  | {
      readonly kind: 'resample';
      readonly algorithm: typeof IQ_RESAMPLER_ALGORITHM;
      readonly sourceSampleRateHz: number;
      readonly outputSampleRateHz: number;
      readonly antiAliasCutoffHz: number;
      readonly zeroCrossings: typeof IQ_RESAMPLER_ZERO_CROSSINGS;
    };

export interface ClassifierReceiverCleanIq {
  readonly bytes: Uint8Array;
  readonly source: ClassifierReceiverSourceLineage;
  readonly transform: ClassifierReceiverTransformReceipt;
}

export interface AcquireClassifierReceiverCleanIqInput {
  readonly profile: SynthesizedSignalProfile;
  readonly targetSampleRateHz: number;
  readonly captureBandwidthHz: number;
  readonly targetSampleCount: number;
  readonly targetStartSampleIndex: number;
  readonly realizationIndex: number;
}

export interface TransformClassifierReceiverIqInput {
  readonly sourceBytes: Uint8Array;
  readonly sourceSampleRateHz: number;
  readonly targetSampleRateHz: number;
  readonly targetSampleCount: number;
  readonly sourcePositionAtOutputZero?: number;
  readonly sourceStartSample?: number;
  readonly sourceArtifactSha256?: string | null;
  readonly sourceCarrierOffsetHz?: number;
  readonly outputQualification: ClassifierReceiverOutputQualification;
}

/**
 * Acquire the SignalLab source layer and then construct a receiver-facing
 * capture. Qualified fixed profiles are *always* requested at their immutable
 * native geometry. Any rate conversion, packet placement, cropping, or padding
 * happens in the separately named receiver transform and receives a new hash.
 */
export function acquireClassifierReceiverCleanIq(
  input: AcquireClassifierReceiverCleanIqInput,
): ClassifierReceiverCleanIq {
  assertPositiveSafeInteger(input.targetSampleRateHz, 'target sample rate');
  assertPositiveSafeInteger(input.captureBandwidthHz, 'capture bandwidth');
  assertPositiveSafeInteger(input.targetSampleCount, 'target sample count');
  assertNonNegativeSafeInteger(input.targetStartSampleIndex, 'target start sample index');
  assertNonNegativeSafeInteger(input.realizationIndex, 'realization index');
  if (input.captureBandwidthHz > input.targetSampleRateHz) {
    throw new RangeError('capture bandwidth may not exceed target sample rate');
  }

  if (!isFixedDigitalProfile(input.profile)) {
    const descriptor = waveformDescriptor(input.profile);
    const signalBandwidthHz = descriptor.occupiedBandwidthHz;
    if (input.targetSampleRateHz < signalBandwidthHz
      || input.captureBandwidthHz < signalBandwidthHz) {
      throw new RangeError(
        `${input.profile} receiver geometry must contain its `
        + `${signalBandwidthHz} Hz signal bandwidth`,
      );
    }
    const sourceBytes = synthesizeAnalyticComplexIq({
      profile: input.profile,
      sampleRateHz: input.targetSampleRateHz,
      bandwidthHz: signalBandwidthHz,
      sampleCount: input.targetSampleCount,
      startSampleIndex: input.targetStartSampleIndex,
    });
    const transformed = transformClassifierReceiverIq({
      sourceBytes,
      sourceSampleRateHz: input.targetSampleRateHz,
      targetSampleRateHz: input.targetSampleRateHz,
      targetSampleCount: input.targetSampleCount,
      sourceStartSample: input.targetStartSampleIndex,
      sourceArtifactSha256: null,
      sourceCarrierOffsetHz: 0,
      outputQualification: 'receiver-transformed-parameterized-complex-baseband',
    });
    return {
      bytes: transformed.bytes,
      source: {
        profile: input.profile,
        qualification: 'analytic-or-standards-derived-parameterized-source',
        artifactSha256: null,
        sourceSamplesSha256: transformed.receipt.sourceSamplesSha256,
        nativeSampleRateHz: input.targetSampleRateHz,
        signalBandwidthHz,
        profileReferenceCenterHz: descriptor.centerHz,
        rfReferenceCenterHz: descriptor.centerHz,
        nativeCarrierOffsetHz: 0,
        startSampleIndex: input.targetStartSampleIndex,
        sampleCount: input.targetSampleCount,
        replay: 'continuous',
      },
      transform: transformed.receipt,
    };
  }

  const binding = fixedDigitalProfileBinding(input.profile);
  if (input.targetSampleRateHz < binding.signalBandwidthHz
    || input.captureBandwidthHz < binding.signalBandwidthHz) {
    throw new RangeError(
      `${input.profile} receiver geometry must contain its `
      + `${binding.signalBandwidthHz} Hz signal bandwidth`,
    );
  }
  const carrierOffsetHz = binding.nativeCarrierOffsetHz;
  const descriptor = waveformDescriptor(input.profile);
  if (descriptor.qualification !== 'independently-verified-digital-baseband'
    || descriptor.assetSha256 === undefined) {
    throw new Error(
      `${input.profile} fixed source lacks an independently verified artifact identity`,
    );
  }

  if (binding.replay === 'one-shot') {
    if (binding.captureSamples === undefined) {
      throw new Error(`${input.profile} one-shot binding lacks a capture bound`);
    }
    const sourceBytes = synthesizeAnalyticComplexIq({
      profile: input.profile,
      sampleRateHz: binding.nativeSampleRateHz,
      bandwidthHz: binding.signalBandwidthHz,
      sampleCount: binding.captureSamples,
      startSampleIndex: 0,
    });
    const resampledPacketSamples = Math.max(
      1,
      Math.ceil(binding.captureSamples
        * input.targetSampleRateHz / binding.nativeSampleRateHz),
    );
    const sourceSamplesPerTargetSample =
      binding.nativeSampleRateHz / input.targetSampleRateHz;
    const sourcePositionAtOutputZero = oneShotSourcePositionAtOutputZero({
      realizationIndex: input.realizationIndex,
      sourceSampleCount: binding.captureSamples,
      resampledPacketSamples,
      targetSampleCount: input.targetSampleCount,
      sourceSamplesPerTargetSample,
    });
    const transformed = transformClassifierReceiverIq({
      sourceBytes,
      sourceSampleRateHz: binding.nativeSampleRateHz,
      targetSampleRateHz: input.targetSampleRateHz,
      targetSampleCount: input.targetSampleCount,
      sourcePositionAtOutputZero,
      sourceStartSample: 0,
      sourceArtifactSha256: descriptor.assetSha256,
      sourceCarrierOffsetHz: carrierOffsetHz,
      outputQualification: 'derived-from-independently-verified-digital-baseband',
    });
    return {
      bytes: transformed.bytes,
      source: {
        profile: input.profile,
        qualification: 'independently-verified-digital-baseband',
        artifactSha256: descriptor.assetSha256,
        sourceSamplesSha256: transformed.receipt.sourceSamplesSha256,
        nativeSampleRateHz: binding.nativeSampleRateHz,
        signalBandwidthHz: binding.signalBandwidthHz,
        profileReferenceCenterHz: binding.profileReferenceCenterHz,
        rfReferenceCenterHz:
          binding.profileReferenceCenterHz - binding.nativeCarrierOffsetHz,
        nativeCarrierOffsetHz: carrierOffsetHz,
        startSampleIndex: 0,
        sampleCount: binding.captureSamples,
        replay: 'one-shot',
        maxOneShotSamples: binding.captureSamples,
      },
      transform: transformed.receipt,
    };
  }

  const sourceSamplesPerTargetSample =
    binding.nativeSampleRateHz / input.targetSampleRateHz;
  const kernelRadiusSourceSamples = receiverKernelRadiusSourceSamples(
    binding.nativeSampleRateHz,
    input.targetSampleRateHz,
  );
  if (binding.nativePeriodSamples === undefined) {
    throw new Error(`${input.profile} cyclic binding lacks a native period`);
  }
  const requestedSourcePositionAtOutputZero =
    input.targetStartSampleIndex * sourceSamplesPerTargetSample;
  if (!Number.isSafeInteger(Math.floor(requestedSourcePositionAtOutputZero))) {
    throw new RangeError('cyclic receiver start exceeds safe native coordinates');
  }
  // A cyclic source can provide a left interpolation halo without inventing
  // negative time: add whole declared periods. Unlike adding a bare FIR radius,
  // this preserves the caller's requested phase within the native artifact.
  const periodsForLeftHalo = Math.max(
    0,
    Math.ceil(
      (kernelRadiusSourceSamples - requestedSourcePositionAtOutputZero)
      / binding.nativePeriodSamples,
    ),
  );
  const absoluteSourcePositionAtOutputZero =
    requestedSourcePositionAtOutputZero
    + periodsForLeftHalo * binding.nativePeriodSamples;
  const sourceStartSampleIndex = Math.max(
    0,
    Math.floor(absoluteSourcePositionAtOutputZero)
      - kernelRadiusSourceSamples,
  );
  const sourcePositionAtOutputZero =
    absoluteSourcePositionAtOutputZero - sourceStartSampleIndex;
  const sourceSampleCount = Math.ceil(
    sourcePositionAtOutputZero
      + (input.targetSampleCount - 1) * sourceSamplesPerTargetSample
      + kernelRadiusSourceSamples
      + 2,
  );
  const sourceBytes = acquireCyclicNativeSupport(
    input.profile,
    binding,
    sourceStartSampleIndex,
    sourceSampleCount,
  );
  const transformed = transformClassifierReceiverIq({
    sourceBytes,
    sourceSampleRateHz: binding.nativeSampleRateHz,
    targetSampleRateHz: input.targetSampleRateHz,
    targetSampleCount: input.targetSampleCount,
    sourcePositionAtOutputZero,
    sourceStartSample: sourceStartSampleIndex,
    sourceArtifactSha256: descriptor.assetSha256,
    sourceCarrierOffsetHz: carrierOffsetHz,
    outputQualification: 'derived-from-independently-verified-digital-baseband',
  });
  return {
    bytes: transformed.bytes,
    source: {
      profile: input.profile,
      qualification: 'independently-verified-digital-baseband',
      artifactSha256: descriptor.assetSha256,
      sourceSamplesSha256: transformed.receipt.sourceSamplesSha256,
      nativeSampleRateHz: binding.nativeSampleRateHz,
      signalBandwidthHz: binding.signalBandwidthHz,
      profileReferenceCenterHz: binding.profileReferenceCenterHz,
      rfReferenceCenterHz:
        binding.profileReferenceCenterHz - binding.nativeCarrierOffsetHz,
      nativeCarrierOffsetHz: carrierOffsetHz,
      startSampleIndex: sourceStartSampleIndex,
      sampleCount: sourceSampleCount,
      replay: 'cyclic',
      nativePeriodSamples: binding.nativePeriodSamples,
    },
    transform: transformed.receipt,
  };
}

/**
 * Deterministic band-limited sample-rate conversion for receiver simulation.
 * It never mutates or re-labels the source artifact. Out-of-range samples are
 * zero, which is essential for bounded packets: this transform cannot invent a
 * packet recurrence.
 */
export function transformClassifierReceiverIq(
  input: TransformClassifierReceiverIqInput,
): { readonly bytes: Uint8Array; readonly receipt: ClassifierReceiverTransformReceipt } {
  assertCf32le(input.sourceBytes);
  assertPositiveSafeInteger(input.sourceSampleRateHz, 'source sample rate');
  assertPositiveSafeInteger(input.targetSampleRateHz, 'target sample rate');
  assertPositiveSafeInteger(input.targetSampleCount, 'target sample count');
  const sourcePositionAtOutputZero = input.sourcePositionAtOutputZero ?? 0;
  const sourceStartSample = input.sourceStartSample ?? 0;
  const sourceCarrierOffsetHz = input.sourceCarrierOffsetHz ?? 0;
  if (!Number.isFinite(sourcePositionAtOutputZero)) {
    throw new TypeError('source position at output zero must be finite');
  }
  assertNonNegativeSafeInteger(sourceStartSample, 'source start sample');
  const sourceArtifactSha256 = input.sourceArtifactSha256 ?? null;
  if (sourceArtifactSha256 !== null
    && !/^[a-f0-9]{64}$/.test(sourceArtifactSha256)) {
    throw new TypeError('source artifact SHA-256 must be lowercase hexadecimal');
  }
  if (!Number.isSafeInteger(sourceCarrierOffsetHz)
    || Math.abs(sourceCarrierOffsetHz) >= input.sourceSampleRateHz / 2) {
    throw new RangeError(
      'source carrier offset must be safe integer hertz strictly inside source Nyquist',
    );
  }

  const sourceSampleCount =
    input.sourceBytes.byteLength / ANALYTIC_COMPLEX_IQ_BYTES_PER_SAMPLE;
  const sourceSamplesSha256 = sha256(input.sourceBytes);
  const translatedSourceBytes = sourceCarrierOffsetHz === 0
    ? input.sourceBytes
    : translateCf32leCarrier({
      sourceBytes: input.sourceBytes,
      sourceStartSample,
      sampleRateHz: input.sourceSampleRateHz,
      sourceCarrierOffsetHz,
      outputCarrierOffsetHz: 0,
    });
  const source = decodeCf32le(translatedSourceBytes);
  const bytes = new Uint8Array(
    input.targetSampleCount * ANALYTIC_COMPLEX_IQ_BYTES_PER_SAMPLE,
  );
  const output = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const sourceSamplesPerTargetSample =
    input.sourceSampleRateHz / input.targetSampleRateHz;
  const kernelRadiusSourceSamples = receiverKernelRadiusSourceSamples(
    input.sourceSampleRateHz,
    input.targetSampleRateHz,
  );
  const antiAliasCutoffCyclesPerSourceSample = normalizedReceiverCutoff(
    input.sourceSampleRateHz,
    input.targetSampleRateHz,
  );

  for (let outputIndex = 0; outputIndex < input.targetSampleCount; outputIndex += 1) {
    const sourcePosition = sourcePositionAtOutputZero
      + outputIndex * sourceSamplesPerTargetSample;
    let inPhase: number;
    let quadrature: number;
    if (input.sourceSampleRateHz === input.targetSampleRateHz
      && Number.isInteger(sourcePosition)) {
      const sourceIndex = sourcePosition;
      inPhase = sourceIndex >= 0 && sourceIndex < sourceSampleCount
        ? source.inPhase[sourceIndex]!
        : 0;
      quadrature = sourceIndex >= 0 && sourceIndex < sourceSampleCount
        ? source.quadrature[sourceIndex]!
        : 0;
    } else {
      [inPhase, quadrature] =
        sourcePosition + kernelRadiusSourceSamples < 0
          || sourcePosition - kernelRadiusSourceSamples >= sourceSampleCount
          ? [0, 0]
          : windowedSincSample(
            source.inPhase,
            source.quadrature,
            sourcePosition,
            antiAliasCutoffCyclesPerSourceSample,
            kernelRadiusSourceSamples,
          );
    }
    output.setFloat32(outputIndex * 8, Math.fround(inPhase), true);
    output.setFloat32(outputIndex * 8 + 4, Math.fround(quadrature), true);
  }

  const antiAliasCutoffHz =
    antiAliasCutoffCyclesPerSourceSample * input.sourceSampleRateHz;
  const operations: ClassifierReceiverTransformOperation[] = [];
  if (sourceCarrierOffsetHz !== 0) {
    operations.push({
      kind: 'frequency-translate',
      algorithm: 'complex-rotator-v1',
      sourceCarrierOffsetHz,
      outputCarrierOffsetHz: 0,
    });
  }
  if (input.sourceSampleRateHz !== input.targetSampleRateHz) {
    operations.push({
      kind: 'resample',
      algorithm: IQ_RESAMPLER_ALGORITHM,
      sourceSampleRateHz: input.sourceSampleRateHz,
      outputSampleRateHz: input.targetSampleRateHz,
      antiAliasCutoffHz,
      zeroCrossings: IQ_RESAMPLER_ZERO_CROSSINGS,
    });
  }

  return {
    bytes,
    receipt: {
      receiptVersion: 1,
      id: CLASSIFIER_RECEIVER_IQ_TRANSFORM_ID,
      algorithm: CLASSIFIER_RECEIVER_IQ_TRANSFORM_ALGORITHM,
      qualification: input.outputQualification,
      sourceArtifactSha256,
      sourceStartSample,
      sourceSampleRateHz: input.sourceSampleRateHz,
      outputSampleRateHz: input.targetSampleRateHz,
      sourceSampleCount,
      outputSampleCount: input.targetSampleCount,
      kernelRadiusSourceSamples,
      sourceCarrierOffsetHz,
      outputCarrierOffsetHz: 0,
      frequencyTranslationHz: -sourceCarrierOffsetHz,
      outputStartSourcePosition: sourceStartSample + sourcePositionAtOutputZero,
      boundary: 'zero',
      antiAliasCutoffHz,
      sourceSamplesSha256,
      outputSamplesSha256: sha256(bytes),
      operations,
    },
  };
}

/** Apply SignalLab's numeric receiver/channel model to an already transformed capture. */
export function applyClassifierReceiverImpairments(
  sourceBytes: Uint8Array,
  impairments: ReceiverImpairments,
  seed: number,
): Uint8Array {
  assertCf32le(sourceBytes);
  const decoded = decodeCf32le(sourceBytes);
  const impaired = applyReceiverImpairments(
    decoded.inPhase,
    decoded.quadrature,
    impairments,
    seed,
  );
  return encodePeakNormalizedCf32le(impaired.re, impaired.im);
}

export function sha256Cf32le(bytes: Uint8Array): string {
  assertCf32le(bytes);
  return sha256(bytes);
}

function oneShotSourcePositionAtOutputZero(input: {
  readonly realizationIndex: number;
  readonly sourceSampleCount: number;
  readonly resampledPacketSamples: number;
  readonly targetSampleCount: number;
  readonly sourceSamplesPerTargetSample: number;
}): number {
  if (input.resampledPacketSamples <= input.targetSampleCount) {
    const availablePlacement =
      input.targetSampleCount - input.resampledPacketSamples;
    const placement = availablePlacement === 0
      ? 0
      : Math.floor(deterministicUnitFraction(input.realizationIndex)
        * (availablePlacement + 1));
    return -placement * input.sourceSamplesPerTargetSample;
  }
  const maximumSourceStart = Math.max(
    0,
    input.sourceSampleCount
      - input.targetSampleCount * input.sourceSamplesPerTargetSample,
  );
  return deterministicUnitFraction(input.realizationIndex) * maximumSourceStart;
}

function deterministicUnitFraction(index: number): number {
  // Irrational rotation: deterministic, evenly spread, and independent of
  // profile/sample-rate selection without introducing another RNG stream.
  return (index * 0.618_033_988_749_894_9) % 1;
}

function windowedSincSample(
  inPhase: Float64Array,
  quadrature: Float64Array,
  position: number,
  cutoff: number,
  radius: number,
): readonly [number, number] {
  const first = Math.floor(position) - radius + 1;
  const last = Math.floor(position) + radius;
  let weightedInPhase = 0;
  let weightedQuadrature = 0;
  let fullWeight = 0;
  for (let sourceIndex = first; sourceIndex <= last; sourceIndex += 1) {
    const distance = position - sourceIndex;
    const relative = Math.abs(distance) / radius;
    const window = 0.42
      + 0.5 * Math.cos(Math.PI * relative)
      + 0.08 * Math.cos(2 * Math.PI * relative);
    const kernelArgument = 2 * cutoff * distance;
    const sinc = Math.abs(kernelArgument) < 1e-14
      ? 1
      : Math.sin(Math.PI * kernelArgument) / (Math.PI * kernelArgument);
    const weight = 2 * cutoff * sinc * window;
    fullWeight += weight;
    if (sourceIndex >= 0 && sourceIndex < inPhase.length) {
      weightedInPhase += inPhase[sourceIndex]! * weight;
      weightedQuadrature += quadrature[sourceIndex]! * weight;
    }
  }
  if (Math.abs(fullWeight) < 1e-14) return [0, 0];
  return [
    weightedInPhase / fullWeight,
    weightedQuadrature / fullWeight,
  ];
}

function receiverKernelRadiusSourceSamples(
  sourceSampleRateHz: number,
  targetSampleRateHz: number,
): number {
  return Math.ceil(
    IQ_RESAMPLER_ZERO_CROSSINGS
      / (2 * normalizedReceiverCutoff(sourceSampleRateHz, targetSampleRateHz)),
  );
}

function normalizedReceiverCutoff(
  sourceSampleRateHz: number,
  targetSampleRateHz: number,
): number {
  const rateRatio = targetSampleRateHz / sourceSampleRateHz;
  // Equal-rate fractional delay and interpolation preserve the complete source
  // Nyquist interval. The 0.95 transition guard is required only when the
  // output Nyquist is lower than the source Nyquist.
  return rateRatio < 1
    ? 0.5 * rateRatio * IQ_RESAMPLER_NYQUIST_GUARD
    : 0.5;
}

/**
 * SignalLab limits one synthesis request to 65,536 samples. Cyclic qualified
 * artifacts are still a deterministic unbounded source: acquire adjacent
 * native-coordinate chunks and concatenate them without altering any sample.
 */
function acquireCyclicNativeSupport(
  profile: FixedDigitalProfile,
  binding: FixedDigitalProfileBinding,
  startSampleIndex: number,
  sampleCount: number,
): Uint8Array {
  if (binding.replay !== 'cyclic'
    || binding.nativePeriodSamples === undefined) {
    throw new Error(`${profile} does not declare deterministic cyclic replay`);
  }
  assertNonNegativeSafeInteger(startSampleIndex, 'cyclic source start sample');
  assertPositiveSafeInteger(sampleCount, 'cyclic source sample count');
  if (!Number.isSafeInteger(startSampleIndex + sampleCount)) {
    throw new RangeError('cyclic source support exceeds safe sample coordinates');
  }
  const output = new Uint8Array(
    sampleCount * ANALYTIC_COMPLEX_IQ_BYTES_PER_SAMPLE,
  );
  let generatedSamples = 0;
  while (generatedSamples < sampleCount) {
    const chunkSamples = Math.min(
      MAX_ANALYTIC_COMPLEX_IQ_SAMPLES,
      sampleCount - generatedSamples,
    );
    const chunk = synthesizeAnalyticComplexIq({
      profile,
      sampleRateHz: binding.nativeSampleRateHz,
      bandwidthHz: binding.signalBandwidthHz,
      sampleCount: chunkSamples,
      startSampleIndex: startSampleIndex + generatedSamples,
    });
    output.set(
      chunk,
      generatedSamples * ANALYTIC_COMPLEX_IQ_BYTES_PER_SAMPLE,
    );
    generatedSamples += chunkSamples;
  }
  return output;
}

function decodeCf32le(bytes: Uint8Array): {
  readonly inPhase: Float64Array;
  readonly quadrature: Float64Array;
} {
  const sampleCount = bytes.byteLength / ANALYTIC_COMPLEX_IQ_BYTES_PER_SAMPLE;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const inPhase = new Float64Array(sampleCount);
  const quadrature = new Float64Array(sampleCount);
  for (let index = 0; index < sampleCount; index += 1) {
    inPhase[index] = view.getFloat32(index * 8, true);
    quadrature[index] = view.getFloat32(index * 8 + 4, true);
  }
  return { inPhase, quadrature };
}

function encodePeakNormalizedCf32le(
  inPhase: Float64Array,
  quadrature: Float64Array,
): Uint8Array {
  if (inPhase.length !== quadrature.length) {
    throw new RangeError('I and Q channel lengths must match');
  }
  let peak = 0;
  for (let index = 0; index < inPhase.length; index += 1) {
    peak = Math.max(peak, Math.hypot(inPhase[index]!, quadrature[index]!));
  }
  const scale = peak > 0 ? 0.98 / peak : 1;
  const bytes = new Uint8Array(
    inPhase.length * ANALYTIC_COMPLEX_IQ_BYTES_PER_SAMPLE,
  );
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let index = 0; index < inPhase.length; index += 1) {
    view.setFloat32(index * 8, Math.fround(inPhase[index]! * scale), true);
    view.setFloat32(index * 8 + 4, Math.fround(quadrature[index]! * scale), true);
  }
  return bytes;
}

function assertCf32le(bytes: Uint8Array): void {
  if (!(bytes instanceof Uint8Array)
    || bytes.byteLength === 0
    || bytes.byteLength % ANALYTIC_COMPLEX_IQ_BYTES_PER_SAMPLE !== 0) {
    throw new TypeError('Receiver transform source must be non-empty interleaved cf32le');
  }
}

function assertPositiveSafeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new RangeError(`${label} must be a positive safe integer`);
  }
}

function assertNonNegativeSafeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${label} must be a non-negative safe integer`);
  }
}

function sha256(bytes: Uint8Array): string {
  return createHash('sha256')
    .update(Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength))
    .digest('hex');
}
