import { describe, expect, it } from 'vitest';
import {
  FIRMWARE_SOURCE_COMMIT,
  type DetectedSignal,
  type DeviceIdentity,
  type SignalDetectionConfig,
  type Sweep,
  type WaveformClassificationEvidence,
  type ZeroSpanCapture,
} from '../../Atom-Atomizer/packages/contracts/src/index.js';
import {
  createDetectedPowerCaptureReceipt,
  extractObservableFeatures,
  SignalDetector,
  SignalTracker,
} from '../../Atom-Atomizer/packages/analysis/src/index.js';
import {
  classifyValidatorReceiptQualifiedObservation,
  extractValidatorReceiptQualifiedObservation,
  type ValidatorReceiptQualifiedCapture,
} from './validator-receipt-qualified-capture.js';

const identity: DeviceIdentity = {
  model: 'validator receipt-qualified fixture',
  hardwareVersion: 'offline',
  firmwareVersion: 'fixture',
  firmwareSourceCommit: FIRMWARE_SOURCE_COMMIT,
  firmwareQualification: 'protocol-test',
  port: {
    id: 'validator-receipt-fixture',
    path: 'offline://validator-receipt-fixture',
    usbMatch: 'protocol-test-double',
    transport: 'protocol-test-double',
    execution: 'protocol-test-double',
  },
  simulated: true,
  usbIdentityVerified: false,
  execution: 'protocol-test-double',
};

const config: SignalDetectionConfig = {
  threshold: { strategy: 'noise-relative', marginDb: 10 },
  minimumBandwidthHz: 0,
  minimumProminenceDb: 6,
  minimumConsecutiveSweeps: 2,
  releaseAfterMissedSweeps: 2,
};

describe('validator receipt-qualified capture path', () => {
  it('executes production agile censoring and classification, and fails closed on envelope leakage', async () => {
    const capture = qualifiedAgileCapture();
    const observation = extractValidatorReceiptQualifiedObservation(capture);

    expect(observation).toMatchObject({
      views: ['scalar-spectrum'],
      limitations: expect.arrayContaining([
        'frequency-agile-fixed-tune-envelope-censored',
      ]),
    });
    expect(observation.values).toEqual(capture.spectrumObservation.values);
    expect(observation.zeroSpanCaptureId).toBeUndefined();
    expect(observation.detectedPowerAcquisitionQualification).toBeUndefined();
    expect(observation.detectedPowerSelectionCondition).toBeUndefined();

    let classifierEvidence: WaveformClassificationEvidence | undefined;
    const result = await classifyValidatorReceiptQualifiedObservation(
      capture,
      observation,
      {
        async classify(detection, evidence) {
          classifierEvidence = evidence;
          const classifiedObservation = extractObservableFeatures(
            detection,
            evidence,
          );
          return {
            detectionId: detection.id,
            label: 'unknown',
            confidence: 0,
            candidates: [{ label: 'unknown', confidence: 0 }],
            modelId: 'validator-production-path-fixture',
            qualification: 'bayesian-observable-equivalence',
            scoreKind: 'model-posterior',
            decisionLevel: 'unknown',
            classifiedAt: new Date().toISOString(),
            evidence: {
              centerHz: classifiedObservation.centerHz,
              bandwidthHz: classifiedObservation.bandwidthHz,
              peakDbm: detection.peakDbm,
              sweepIds: classifiedObservation.sweepIds,
              views: classifiedObservation.views,
              features: {
                ...classifiedObservation.values,
                'model.maximumKnownSyntheticSupportRank': 0.5,
              },
              limitations: classifiedObservation.limitations,
            },
          };
        },
      },
    );
    expect(classifierEvidence).toMatchObject({
      zeroSpan: { id: capture.zeroSpan.id },
      detectedPowerCaptureReceipt: { schemaVersion: 4 },
      zeroSpanSpectrumSweepIds: capture.spectrumObservation.sweepIds,
    });
    expect(result.evidence).toMatchObject({
      views: ['scalar-spectrum'],
      limitations: expect.arrayContaining([
        'frequency-agile-fixed-tune-envelope-censored',
      ]),
    });

    expect(() => extractValidatorReceiptQualifiedObservation(
      capture,
      () => capture.spectrumObservation,
    )).toThrow(/production censor/i);

    const leakedObservation = {
      ...observation,
      values: {
        ...observation.values,
        'envelope.validatorLeakSentinel': 1,
      },
      views: ['scalar-spectrum', 'detected-power-envelope'] as const,
      zeroSpanCaptureId: capture.zeroSpan.id,
      detectedPowerAcquisitionQualification:
        'receipt-verified-provenance-bound-runtime-admitted-physical-capture-v5' as const,
      detectedPowerSelectionCondition:
        'automatic-current-source-sweep-integrated-excess-rank-0' as const,
    };
    expect(() => extractValidatorReceiptQualifiedObservation(
      capture,
      () => leakedObservation,
    )).toThrow(/leaked fixed-tune envelope evidence/i);

    await expect(classifyValidatorReceiptQualifiedObservation(
      capture,
      observation,
      {
        async classify() {
          return {
            ...result,
            evidence: {
              ...result.evidence,
              views: ['scalar-spectrum', 'detected-power-envelope'],
              zeroSpanCaptureId: capture.zeroSpan.id,
              detectedPowerAcquisitionQualification:
                'receipt-verified-provenance-bound-runtime-admitted-physical-capture-v5',
              detectedPowerSelectionCondition:
                'automatic-current-source-sweep-integrated-excess-rank-0',
              features: {
                ...result.evidence.features,
                'envelope.validatorLeakSentinel': 1,
              },
            },
          };
        },
      },
    )).rejects.toThrow(/classifier result does not match|leaked fixed-tune envelope/i);
  });
});

function qualifiedAgileCapture(): ValidatorReceiptQualifiedCapture {
  const centersHz = [2_402, 2_410, 2_418, 2_426, 2_434, 2_442, 2_450, 2_480]
    .map((value) => value * 1_000_000);
  const sweeps = centersHz.map((centerHz, index) =>
    agileSweep(index + 1, centerHz));
  const detector = new SignalDetector(config);
  const tracker = new SignalTracker(config);
  let tracks: readonly DetectedSignal[] = [];
  for (const sweep of sweeps) {
    tracks = tracker.update(sweep, detector.analyze(sweep));
  }
  const detection = tracks.find((track) =>
    track.associationMode === 'frequency-agile-2g4-activity');
  if (detection === undefined) {
    throw new Error('Fixture did not produce a frequency-agile representative');
  }
  const rawTarget = tracks.find((track) =>
    track.id === detection.associationObservations?.at(-1)?.trackId);
  if (rawTarget === undefined) {
    throw new Error('Fixture did not preserve the current agile physical member');
  }
  const spectrumObservation = extractObservableFeatures(detection, { sweeps });
  const zeroSpan: ZeroSpanCapture = {
    kind: 'zero-span',
    id: 'validator-receipt-qualified-agile-capture',
    sequence: 9,
    capturedAt: new Date(Date.UTC(2026, 0, 1) + 9 * 50).toISOString(),
    elapsedMilliseconds: 50,
    frequencyHz: rawTarget.peakHz,
    samplePeriodSeconds: 1 / 9_000,
    timingQualification: 'wall-clock-derived',
    targetDetectionId: rawTarget.id,
    powerDbm: Array.from(
      { length: 450 },
      (_, index) => index % 10 < 4 ? -45 : -90,
    ),
    requested: {
      kind: 'detected-power-timeseries',
      centerHz: rawTarget.peakHz,
      sampleCount: 450,
      sweepTimeSeconds: 0.05,
      controls: {
        schemaVersion: 1,
        model: 'receiver',
        resolutionBandwidthKhz: 20,
        attenuationDb: 'auto',
        trigger: { mode: 'auto' },
      },
    },
    actualRbwHz: 20_000,
    actualAttenuationDb: 0,
    source: 'scan-text',
    complete: true,
    identity,
  };
  return {
    detection,
    evidenceSweeps: sweeps,
    spectrumObservation,
    zeroSpan,
    detectedPowerCaptureReceipt: createDetectedPowerCaptureReceipt({
      activeSignals: tracks,
      evidenceSweeps: sweeps,
      capture: zeroSpan,
      admittedTargetTuneHz: zeroSpan.frequencyHz,
      spectrumSweepIds: spectrumObservation.sweepIds,
    }),
  };
}

function agileSweep(sequence: number, activeFrequencyHz: number): Sweep {
  const points = 401;
  const startHz = 2_399_000_000;
  const stopHz = 2_483_000_000;
  const frequencyHz = Array.from(
    { length: points },
    (_, index) => startHz + (stopHz - startHz) * index / (points - 1),
  );
  return {
    kind: 'spectrum',
    id: `validator-receipt-agile-${sequence}`,
    sequence,
    capturedAt: new Date(Date.UTC(2026, 0, 1) + sequence * 50).toISOString(),
    elapsedMilliseconds: 50,
    frequencyHz,
    powerDbm: frequencyHz.map((frequency) =>
      Math.abs(frequency - activeFrequencyHz) <= 300_000 ? -45 : -110),
    requested: {
      kind: 'swept-spectrum',
      startHz,
      stopHz,
      points,
      sweepTimeSeconds: 0.05,
      controls: {
        schemaVersion: 1,
        model: 'receiver',
        acquisitionFormat: 'text',
        resolutionBandwidthKhz: 'auto',
        attenuationDb: 'auto',
        detector: 'sample',
        spurRejection: 'auto',
        lowNoiseAmplifier: 'off',
        avoidSpurs: 'auto',
        trigger: { mode: 'auto' },
      },
    },
    actualStartHz: startHz,
    actualStopHz: stopHz,
    actualRbwHz: (stopHz - startHz) / (points - 1),
    actualAttenuationDb: 0,
    source: 'scan-text',
    complete: true,
    identity,
  };
}
