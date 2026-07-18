import { describe, expect, it } from 'vitest';
import { FIRMWARE_SOURCE_COMMIT } from '../../TinySA/packages/contracts/src/index.js';
import type {
  DetectedSignal,
  DeviceIdentity,
  SignalDetectionConfig,
  Sweep,
} from '../../TinySA/packages/contracts/src/index.js';
import {
  classificationCaptureTargetProjections,
  SignalDetector,
  SignalTracker,
} from '../../TinySA/packages/analysis/src/index.js';
import { independentlyReplayCaptureTargetProjections } from './validator-capture-target-projection.js';

const identity: DeviceIdentity = {
  model: 'validator projection fixture',
  hardwareVersion: 'offline',
  firmwareVersion: 'fixture',
  firmwareSourceCommit: FIRMWARE_SOURCE_COMMIT,
  firmwareQualification: 'protocol-test',
  port: {
    id: 'validator-fixture',
    path: 'offline://validator-fixture',
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

describe('independent validator capture-target projection replay', () => {
  it('reproduces the exact current frequency-agile candidate/evidence projection', () => {
    const { tracks, rawTarget, agileRepresentative } = qualifiedAgileFixture();
    const replayed = independentlyReplayCaptureTargetProjections(tracks);
    const shared = classificationCaptureTargetProjections(tracks);

    expect(rawTarget).toMatchObject({ state: 'candidate', missedSweeps: 0 });
    expect(replayed).toHaveLength(1);
    expect(replayed[0]).toMatchObject({
      rawTarget: { id: rawTarget.id, state: 'candidate' },
      projectedRepresentative: { id: agileRepresentative.id },
      projectionKind: 'current-qualified-agile-latest-member',
    });
    expect(projectionIdentities(replayed)).toEqual(projectionIdentities(shared));
  });

  it('fails closed for a bare, stale, ambiguous, duplicated, or mutated candidate binding', () => {
    const { tracks, rawTarget, agileRepresentative } = qualifiedAgileFixture();
    const without = (id: string) => tracks.filter((track) => track.id !== id);

    expect(independentlyReplayCaptureTargetProjections([rawTarget])).toEqual([]);
    expect(independentlyReplayCaptureTargetProjections([
      ...without(rawTarget.id),
      { ...rawTarget, missedSweeps: 1 },
    ])).toEqual([]);
    expect(independentlyReplayCaptureTargetProjections([
      ...without(agileRepresentative.id),
      { ...agileRepresentative, associationMissedSweeps: 1 },
    ])).toEqual([]);
    expect(independentlyReplayCaptureTargetProjections([
      ...without(agileRepresentative.id),
      {
        ...agileRepresentative,
        associationOpportunities:
          agileRepresentative.associationOpportunities?.map(
            (opportunity, index, values) => index === values.length - 1
              ? { ...opportunity, outcome: 'ambiguous' as const }
              : opportunity,
          ),
      },
    ])).toEqual([]);
    expect(independentlyReplayCaptureTargetProjections([
      ...tracks,
      structuredClone(rawTarget),
    ])).toEqual([]);

    const mutatedRaw = structuredClone(rawTarget);
    mutatedRaw.localClassificationObservations =
      mutatedRaw.localClassificationObservations?.map(
        (observation, index, values) => index === values.length - 1
          ? { ...observation, peakHz: observation.peakHz + 1 }
          : observation,
      );
    expect(independentlyReplayCaptureTargetProjections([
      ...without(rawTarget.id),
      mutatedRaw,
    ])).toEqual([]);
  });

  it('rejects ambiguous agile ownership while preserving an independently active direct row', () => {
    const { tracks, rawTarget, agileRepresentative } = qualifiedAgileFixture();
    const secondSummary: DetectedSignal = {
      ...structuredClone(agileRepresentative),
      id: 'agile-2g4-activity-9999',
      associationId: 'agile-2g4-activity-9999',
    };
    const ambiguousCandidateRows = [...tracks, secondSummary];
    expect(independentlyReplayCaptureTargetProjections(ambiguousCandidateRows))
      .toEqual([]);

    const activeRaw = { ...rawTarget, state: 'active' as const };
    const ambiguousActiveRows = [
      ...tracks.filter((track) => track.id !== rawTarget.id),
      activeRaw,
      secondSummary,
    ];
    expect(independentlyReplayCaptureTargetProjections(ambiguousActiveRows))
      .toEqual([{
        rawTarget: activeRaw,
        projectedRepresentative: activeRaw,
        projectionKind: 'current-active-physical-representative',
      }]);
  });

  it('ranks mixed direct and agile projections by raw physical peak power', () => {
    const { tracks, rawTarget } = qualifiedAgileFixture();
    const strongerDirect: DetectedSignal = {
      ...structuredClone(rawTarget),
      id: 'stronger-direct',
      state: 'active',
      peakDbm: rawTarget.peakDbm + 5,
    };
    const replayed = independentlyReplayCaptureTargetProjections([
      ...tracks,
      strongerDirect,
    ]);

    expect(replayed.map((projection) => projection.rawTarget.id))
      .toEqual([strongerDirect.id, rawTarget.id]);
    expect(replayed.map((projection) => projection.projectionKind)).toEqual([
      'current-active-physical-representative',
      'current-qualified-agile-latest-member',
    ]);
  });
});

function qualifiedAgileFixture(): {
  tracks: readonly DetectedSignal[];
  rawTarget: DetectedSignal;
  agileRepresentative: DetectedSignal;
} {
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
  const agileRepresentative = tracks.find((track) =>
    track.associationMode === 'frequency-agile-2g4-activity')!;
  const rawTarget = tracks.find((track) =>
    track.id === agileRepresentative.associationObservations?.at(-1)?.trackId)!;
  return { tracks, rawTarget, agileRepresentative };
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
    id: `validator-agile-${sequence}`,
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

function projectionIdentities(values: readonly {
  rawTarget: DetectedSignal;
  projectedRepresentative: DetectedSignal;
  projectionKind: string;
}[]): readonly string[] {
  return values.map((projection) => [
    projection.rawTarget.id,
    projection.projectedRepresentative.id,
    projection.projectionKind,
  ].join('|'));
}
