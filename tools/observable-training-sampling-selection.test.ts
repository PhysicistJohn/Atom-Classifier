import { describe, expect, it } from 'vitest';
import type { DetectedSignal } from '../../TinySA/packages/contracts/src/index.js';
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

  it('matches the live envelope branch by ranking active raw targets before association projection', () => {
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
      .toBe('preferred-then-strongest-current-physical-or-qualified-agile-member-target-v3');
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
    lastSeenAt: '2026-01-01T00:00:01.000Z',
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
    bayesianEvidence: {
      modelId: 'test-detector',
      posteriorScope: 'selected-local-region',
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
      qualification: 'synthetic-known-presence',
      noiseSigmaDb: 1,
      observedMeanShiftDb: 20,
      looks: 1,
    },
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
