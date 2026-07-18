import { describe, expect, it } from 'vitest';
import type {
  DetectedSignal,
  DeviceIdentity,
  Sweep,
} from '../../Atom-Atomizer/packages/contracts/src/index.js';
import {
  CONSECUTIVE_SPECTRUM_REPRESENTATIVE_SELECTION_POLICY_ID,
  QUALIFIED_ENVELOPE_REPRESENTATIVE_SELECTION_POLICY_ID,
  consecutiveSpectrumClassificationRepresentatives,
  qualifiedEnvelopeCaptureTargetRepresentatives,
} from './observable-training-sampling.js';

describe('observable-training branch representative selection', () => {
  it('matches the live spectrum branch by excluding inactive rows and collapsing each association once', () => {
    const associatedLeft = signal('associated-left', 'active', -72, 'association-1');
    const associatedCenter = signal('associated-center', 'active', -61, 'association-1');
    const associatedRight = signal('associated-right', 'active', -40, 'association-1');
    const local = signal('local-active', 'active', -50);
    const inactive = signal('inactive-stronger', 'released', -10);

    const representatives = consecutiveSpectrumClassificationRepresentatives([
      associatedLeft,
      associatedCenter,
      associatedRight,
      local,
      inactive,
    ]);

    expect(CONSECUTIVE_SPECTRUM_REPRESENTATIVE_SELECTION_POLICY_ID)
      .toBe('active-classification-representatives-v1');
    expect(representatives.map((item) => item.id)).not.toContain(inactive.id);
    expect(representatives.filter((item) => item.associationId === 'association-1'))
      .toHaveLength(1);
    expect(new Set(representatives.map(representativePopulationKey)).size)
      .toBe(representatives.length);
  });

  it('matches the live envelope branch by integrated current-source power before association projection', () => {
    const associatedLeft = signal('associated-left', 'active', -72, 'association-1');
    const associatedCenter = signal('associated-center', 'active', -61, 'association-1');
    const associatedRight = signal('associated-right', 'active', -40, 'association-1');
    const local = signal('local-active', 'active', -50);
    const inactive = signal('inactive-stronger', 'released', -10);
    const retainedMiss = {
      ...signal('retained-miss', 'active', -5),
      missedSweeps: 1,
    } satisfies DetectedSignal;
    const agileSummary = {
      ...signal('agile-summary', 'active', -1, 'agile-association'),
      associationMode: 'frequency-agile-2g4-activity',
      associationModelId: 'frequency-agile-2g4-activity-v3',
    } satisfies DetectedSignal;

    const representatives = qualifiedEnvelopeCaptureTargetRepresentatives([
      associatedLeft,
      associatedCenter,
      associatedRight,
      local,
      inactive,
      retainedMiss,
      agileSummary,
    ]);

    expect(QUALIFIED_ENVELOPE_REPRESENTATIVE_SELECTION_POLICY_ID)
      .toBe('preferred-then-current-source-sweep-integrated-excess-power-physical-or-qualified-agile-member-target-v4');
    expect(representatives.map((item) => item.id)).toEqual([
      associatedRight.id,
      local.id,
      associatedCenter.id,
      associatedLeft.id,
    ]);
    expect(representatives.map((item) => item.id)).not.toContain(inactive.id);
    expect(representatives.map((item) => item.id)).not.toContain(retainedMiss.id);
    expect(representatives.map((item) => item.id)).not.toContain(agileSummary.id);
    expect(new Set(representatives.map((item) => item.id)).size)
      .toBe(representatives.length);
  });
});

function representativePopulationKey(signal: DetectedSignal): string {
  return signal.associationId === undefined
    ? `local:${signal.id}`
    : `association:${signal.associationId}`;
}

function signal(
  id: string,
  state: DetectedSignal['state'],
  peakDbm: number,
  associationId?: string,
): DetectedSignal {
  const capturedAt = '2026-01-01T00:00:01.000Z';
  const sourceSweep: Sweep = {
    kind: 'spectrum',
    id: 'sweep-1',
    sequence: 1,
    capturedAt,
    elapsedMilliseconds: 50,
    frequencyHz: [80, 90, 100, 110, 120],
    powerDbm: [-100, -90, peakDbm, -90, -100],
    requested: {
      kind: 'swept-spectrum',
      startHz: 80,
      stopHz: 120,
      points: 5,
      sweepTimeSeconds: 0.05,
      controls: {
        schemaVersion: 1,
        model: 'receiver',
        acquisitionFormat: 'text',
        resolutionBandwidthKhz: 0.01,
        attenuationDb: 'auto',
        detector: 'sample',
        spurRejection: 'off',
        lowNoiseAmplifier: 'off',
        avoidSpurs: 'off',
        trigger: { mode: 'auto' },
      },
    },
    actualStartHz: 80,
    actualStopHz: 120,
    actualRbwHz: 10,
    actualAttenuationDb: 0,
    source: 'scan-text',
    complete: true,
    identity: fixtureIdentity,
  };
  const localBayesianEvidence = {
    modelId: 'test-detector',
    posteriorScope: 'selected-local-region' as const,
    priorSignalProbability: 0.01,
    posteriorSignalProbability: 0.99,
    logBayesFactor: 10,
    effectiveIndependentBins: 1,
    effectiveReferenceCells: 8,
    noiseShape: 8,
    posteriorPredictiveNullProbability: 0.001,
    targetPosteriorPredictiveNullProbability: 0.001,
    targetSweepFalseAlarmProbability: 0.001,
    multiplicityAdjustedTests: 1,
    testedRegionStartHz: 90,
    testedRegionStopHz: 110,
    qualification: 'synthetic-known-presence' as const,
    noiseSigmaDb: 1,
    observedMeanShiftDb: 20,
    looks: 1,
  };
  const localObservation = {
    sourceSweep,
    startHz: 90,
    stopHz: 110,
    peakHz: 100,
    detectorId: 'test-detector',
    localBayesianEvidence,
  } as const;
  return {
    id,
    state,
    peakDbm,
    peakHz: 100,
    startHz: 90,
    stopHz: 110,
    bandwidthHz: 20,
    prominenceDb: 30,
    prominenceThresholdDb: 6,
    thresholdDbm: -80,
    noiseFloorDbm: -100,
    firstSeenAt: '2026-01-01T00:00:00.000Z',
    lastSeenAt: capturedAt,
    sweepIds: ['sweep-1'],
    persistenceSweeps: 1,
    missedSweeps: state === 'active' ? 0 : 1,
    detectorId: 'test-detector',
    detectorConfig: {
      threshold: { strategy: 'noise-relative', marginDb: 10 },
      minimumBandwidthHz: 0,
      minimumProminenceDb: 6,
      minimumConsecutiveSweeps: 1,
      releaseAfterMissedSweeps: 1,
    },
    bayesianEvidence: localBayesianEvidence,
    classificationRegionStartHz: 90,
    classificationRegionStopHz: 110,
    classificationRegionSweepIds: ['sweep-1'],
    classificationRegionObservation: localObservation,
    localClassificationObservations: [localObservation],
    qualityFlags: [],
    ...(associationId === undefined ? {
      associationMode: 'frequency-local' as const,
    } : {
      associationMode: 'regular-spectral-component-activity' as const,
      associationId,
      associationMemberTrackIds: [
        'associated-left',
        'associated-center',
        'associated-right',
      ],
      associationMissedSweeps: 0,
    }),
  };
}

const fixtureIdentity: DeviceIdentity = {
  model: 'training-selection-fixture',
  hardwareVersion: 'offline',
  firmwareVersion: 'fixture',
  firmwareQualification: 'protocol-test',
  port: {
    id: 'offline',
    path: 'offline://training-selection-fixture',
    usbMatch: 'protocol-test-double',
    transport: 'protocol-test-double',
    execution: 'protocol-test-double',
  },
  simulated: true,
  usbIdentityVerified: false,
  execution: 'protocol-test-double',
};
