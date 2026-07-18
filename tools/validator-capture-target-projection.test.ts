import { describe, expect, it } from 'vitest';
import { FIRMWARE_SOURCE_COMMIT } from '../../Atom-Atomizer/packages/contracts/src/index.js';
import type {
  DetectedSignal,
  DeviceIdentity,
  SignalDetectionConfig,
  Sweep,
} from '../../Atom-Atomizer/packages/contracts/src/index.js';
import {
  classificationCaptureTargetProjections,
  SignalDetector,
  SignalTracker,
} from '../../Atom-Atomizer/packages/analysis/src/index.js';
import { CLASSIFICATION_CAPTURE_TARGET_RANKING_MODEL } from '../../Atom-Atomizer/packages/analysis/src/classification-target-ranking.js';
import { SIGNAL_LAB_PRODUCTION_CAPTURE_TARGET_SELECTION_POLICY_ID } from '../../Atom-Atomizer/packages/analysis/src/observable-training-acquisition-geometry.js';
import {
  AUTO_TARGET_SELECTION_RANKING_MODEL_ID,
  AUTO_TARGET_SELECTION_POLICY_ID,
  autoTargetSelectionValidationCases,
  synthesizeAutoTargetSelectionValidationCase,
} from '../../Atom-SignalLab/src/auto-target-selection-corpus.js';
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
      .toMatchObject([{
        rawTarget: activeRaw,
        projectedRepresentative: activeRaw,
        projectionKind: 'current-active-physical-representative',
        rankEvidence: {
          sourceSweepId: activeRaw.sweepIds.at(-1),
        },
      }]);
  });

  it('ranks a wider lower-peak target above a narrower higher-peak target when its integrated excess power is larger', () => {
    const { rawTarget } = qualifiedAgileFixture();
    const narrowHigherPeak = directRankFixture(rawTarget, {
      id: 'narrow-higher-peak',
      supportStartIndex: 3,
      supportStopIndex: 3,
      peakIndex: 3,
      supportPowerDbm: [-40],
    });
    const wideLowerPeak = directRankFixture(rawTarget, {
      id: 'wide-lower-peak',
      supportStartIndex: 2,
      supportStopIndex: 7,
      peakIndex: 4,
      supportPowerDbm: [-48, -46, -44, -46, -48, -50],
    });
    const input = [narrowHigherPeak, wideLowerPeak];
    const replayed = independentlyReplayCaptureTargetProjections(input);
    const shared = classificationCaptureTargetProjections(input);

    expect(replayed.map((projection) => projection.rawTarget.id))
      .toEqual([wideLowerPeak.id, narrowHigherPeak.id]);
    expect(replayed[0]!.rawTarget.peakDbm)
      .toBeLessThan(replayed[1]!.rawTarget.peakDbm);
    expect(replayed[0]!.rankEvidence.integratedExcessPowerMw)
      .toBeGreaterThan(replayed[1]!.rankEvidence.integratedExcessPowerMw);
    expect(projectionIdentities(replayed)).toEqual(projectionIdentities(shared));
  });

  it('preserves exact numeric and stable-key ordering across inverse and tie cases', () => {
    const { rawTarget } = qualifiedAgileFixture();
    const wideLow = directRankFixture(rawTarget, {
      id: 'wide-low', supportStartIndex: 2, supportStopIndex: 7, peakIndex: 4,
      supportPowerDbm: [-70, -68, -66, -68, -70, -72],
    });
    const narrowVeryHigh = directRankFixture(rawTarget, {
      id: 'narrow-very-high', supportStartIndex: 3, supportStopIndex: 3, peakIndex: 3,
      supportPowerDbm: [-30],
    });
    const tieBeta = directRankFixture(rawTarget, {
      id: 'tie-beta', supportStartIndex: 5, supportStopIndex: 5, peakIndex: 5,
      supportPowerDbm: [-45],
    });
    const tieAlpha = directRankFixture(rawTarget, {
      id: 'tie-alpha', supportStartIndex: 5, supportStopIndex: 5, peakIndex: 5,
      supportPowerDbm: [-45],
    });

    expect(independentlyReplayCaptureTargetProjections([wideLow, narrowVeryHigh])
      .map((projection) => projection.rawTarget.id))
      .toEqual([narrowVeryHigh.id, wideLow.id]);
    expect(independentlyReplayCaptureTargetProjections([tieBeta, tieAlpha])
      .map((projection) => projection.rawTarget.id))
      .toEqual([tieAlpha.id, tieBeta.id]);
  });

  it('fails closed when current rank evidence is stale or self-contradictory', () => {
    const { rawTarget } = qualifiedAgileFixture();
    const valid = directRankFixture(rawTarget, {
      id: 'valid-direct', supportStartIndex: 3, supportStopIndex: 4, peakIndex: 3,
      supportPowerDbm: [-45, -50],
    });
    const stale = {
      ...valid,
      lastSeenAt: '2026-01-02T00:00:00.000Z',
    } satisfies DetectedSignal;
    const wrongFloor = {
      ...valid,
      noiseFloorDbm: valid.noiseFloorDbm + 1,
    } satisfies DetectedSignal;
    expect(independentlyReplayCaptureTargetProjections([stale])).toEqual([]);
    expect(independentlyReplayCaptureTargetProjections([wrongFloor])).toEqual([]);
  });

  it('matches every canonized SignalLab Auto-v4 rank fixture and its no-fallback disclosure', () => {
    expect(AUTO_TARGET_SELECTION_RANKING_MODEL_ID)
      .toBe(CLASSIFICATION_CAPTURE_TARGET_RANKING_MODEL.id);
    expect(AUTO_TARGET_SELECTION_POLICY_ID)
      .toBe(SIGNAL_LAB_PRODUCTION_CAPTURE_TARGET_SELECTION_POLICY_ID);

    for (const fixture of autoTargetSelectionValidationCases) {
      const materialized = synthesizeAutoTargetSelectionValidationCase(fixture.id);
      const tracks = materialized.components.map((component) =>
        signalFromAutoTargetFixture(materialized, component));
      const independent = independentlyReplayCaptureTargetProjections(tracks);
      const production = classificationCaptureTargetProjections(tracks);

      expect(independent.map((projection) => projection.rawTarget.id), fixture.id)
        .toEqual(materialized.expectedRankedRawTargetIds);
      expect(projectionIdentities(independent), fixture.id)
        .toEqual(projectionIdentities(production));
      for (const projection of independent) {
        const expected = materialized.components.find((component) =>
          component.rawTargetId === projection.rawTarget.id)!.rankEvidence;
        expect(projection.rankEvidence, `${fixture.id}:${projection.rawTarget.id}`)
          .toEqual(expected);
      }
      const rankZeroAdmission = materialized.components.find((component) =>
        component.rawTargetId === independent[0]!.rawTarget.id)!.runtimeAdmission;
      if (materialized.expectedAutomaticOutcome.status === 'selected') {
        expect(rankZeroAdmission.status, fixture.id).toBe('admitted');
        expect(independent[0]!.rawTarget.id, fixture.id)
          .toBe(materialized.expectedAutomaticOutcome.rawTargetId);
      } else {
        expect(rankZeroAdmission.status, fixture.id).toBe('unavailable');
        expect(independent[0]!.rawTarget.id, fixture.id)
          .toBe(materialized.expectedAutomaticOutcome.blockedRawTargetId);
        expect(materialized.expectedAutomaticOutcome.lowerRankSubstitutionAllowed)
          .toBe(false);
        expect(materialized.components.find((component) =>
          component.rawTargetId === independent[1]!.rawTarget.id)!.runtimeAdmission.status)
          .toBe('admitted');
      }
    }
  });
});

function signalFromAutoTargetFixture(
  materialized: ReturnType<typeof synthesizeAutoTargetSelectionValidationCase>,
  component: ReturnType<typeof synthesizeAutoTargetSelectionValidationCase>['components'][number],
): DetectedSignal {
  const source = materialized.sweep;
  const sourceSweep: Sweep = {
    ...source,
    source: 'scan-text',
    requested: {
      kind: 'swept-spectrum',
      startHz: source.actualStartHz,
      stopHz: source.actualStopHz,
      points: source.frequencyHz.length,
      sweepTimeSeconds: source.elapsedMilliseconds / 1_000,
      controls: {
        schemaVersion: 1,
        model: 'receiver',
        acquisitionFormat: 'text',
        resolutionBandwidthKhz: source.actualRbwHz / 1_000,
        attenuationDb: 'auto',
        detector: 'sample',
        spurRejection: 'off',
        lowNoiseAmplifier: 'off',
        avoidSpurs: 'off',
        trigger: { mode: 'auto' },
      },
    },
    actualAttenuationDb: 0,
    identity,
  };
  const localBayesianEvidence = {
    modelId: 'fixture-detector',
    posteriorScope: 'selected-local-region' as const,
    priorSignalProbability: 0.01,
    posteriorSignalProbability: 0.99,
    logBayesFactor: 10,
    effectiveIndependentBins: component.rankEvidence.supportCellCount,
    effectiveReferenceCells: 32,
    noiseShape: 32,
    posteriorPredictiveNullProbability: 0.001,
    targetPosteriorPredictiveNullProbability: 0.001,
    targetSweepFalseAlarmProbability: 0.001,
    multiplicityAdjustedTests: 1,
    testedRegionStartHz: component.startHz,
    testedRegionStopHz: component.stopHz,
    qualification: 'synthetic-known-presence' as const,
    noiseSigmaDb: 1,
    observedMeanShiftDb: component.peakDbm - component.noiseFloorDbm,
    looks: 1,
  };
  const localObservation = {
    sourceSweep,
    startHz: component.startHz,
    stopHz: component.stopHz,
    peakHz: component.peakHz,
    detectorId: 'fixture-detector',
    localBayesianEvidence,
  } as const;
  const sweepIds = Array.from(
    { length: component.runtimeAdmission.spectrumHistoryCount },
    (_unused, index) => index
      === component.runtimeAdmission.spectrumHistoryCount - 1
      ? sourceSweep.id
      : `${component.rawTargetId}-history-${index}`,
  );
  return {
    id: component.rawTargetId,
    startHz: component.startHz,
    stopHz: component.stopHz,
    peakHz: component.peakHz,
    peakDbm: component.peakDbm,
    prominenceDb: component.peakDbm - component.noiseFloorDbm,
    prominenceThresholdDb: 6,
    bandwidthHz: component.stopHz - component.startHz,
    thresholdDbm: component.noiseFloorDbm + 10,
    noiseFloorDbm: component.noiseFloorDbm,
    firstSeenAt: sourceSweep.capturedAt,
    lastSeenAt: sourceSweep.capturedAt,
    sweepIds,
    persistenceSweeps: sweepIds.length,
    missedSweeps: 0,
    state: 'active',
    detectorId: 'fixture-detector',
    detectorConfig: config,
    bayesianEvidence: localBayesianEvidence,
    classificationRegionStartHz: component.startHz,
    classificationRegionStopHz: component.stopHz,
    classificationRegionSweepIds: sweepIds,
    classificationRegionObservation: localObservation,
    localClassificationObservations: [localObservation],
    associationMode: 'frequency-local',
    qualityFlags: [],
  };
}

function directRankFixture(
  base: DetectedSignal,
  options: {
    id: string;
    supportStartIndex: number;
    supportStopIndex: number;
    peakIndex: number;
    supportPowerDbm: readonly number[];
  },
): DetectedSignal {
  const frequencyHz = Array.from({ length: 10 }, (_, index) => 100 + index * 10);
  const powerDbm = frequencyHz.map(() => -100);
  for (let index = options.supportStartIndex; index <= options.supportStopIndex; index += 1) {
    powerDbm[index] = options.supportPowerDbm[index - options.supportStartIndex]!;
  }
  const capturedAt = '2026-01-01T00:00:20.000Z';
  const sourceSweep: Sweep = {
    kind: 'spectrum',
    id: `${options.id}-sweep`,
    sequence: 20,
    capturedAt,
    elapsedMilliseconds: 50,
    frequencyHz,
    powerDbm,
    requested: {
      kind: 'swept-spectrum',
      startHz: 95,
      stopHz: 195,
      points: frequencyHz.length,
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
    actualStartHz: 95,
    actualStopHz: 195,
    actualRbwHz: 10,
    actualAttenuationDb: 0,
    source: 'scan-text',
    complete: true,
    identity,
  };
  const startHz = frequencyHz[options.supportStartIndex]!;
  const stopHz = frequencyHz[options.supportStopIndex]!;
  const peakHz = frequencyHz[options.peakIndex]!;
  const peakDbm = powerDbm[options.peakIndex]!;
  const localObservation = {
    sourceSweep,
    startHz,
    stopHz,
    peakHz,
    detectorId: base.detectorId,
    localBayesianEvidence: base.bayesianEvidence,
  } as const;
  return {
    ...structuredClone(base),
    id: options.id,
    state: 'active',
    startHz,
    stopHz,
    peakHz,
    peakDbm,
    bandwidthHz: stopHz - startHz,
    noiseFloorDbm: -100,
    lastSeenAt: capturedAt,
    sweepIds: [sourceSweep.id],
    missedSweeps: 0,
    classificationRegionStartHz: startHz,
    classificationRegionStopHz: stopHz,
    classificationRegionSweepIds: [sourceSweep.id],
    classificationRegionObservation: localObservation,
    localClassificationObservations: [localObservation],
    associationMode: 'frequency-local',
  };
}

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
