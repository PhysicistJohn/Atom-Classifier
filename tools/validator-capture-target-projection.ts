import type {
  DetectedPowerCaptureProjectionKind,
  DetectedSignal,
  LocalClassificationRegionObservation,
} from '../../TinySA/packages/contracts/src/index.js';
import { bayesianDetectionEvidenceMatches } from '../../TinySA/packages/analysis/src/bayesian-signal-detector.js';
import { observableAssociationEvidenceIsCurrentlyQualified } from '../../TinySA/packages/analysis/src/observable-features.js';

const PINNED_AGILE_REGION_START_HZ = 2_402_000_000;
const PINNED_AGILE_REGION_STOP_HZ = 2_480_000_000;

/** Validator-owned replay of the v3 actuation/evidence projection policy. */
export interface IndependentlyReplayedCaptureTargetProjection {
  readonly rawTarget: DetectedSignal;
  readonly projectedRepresentative: DetectedSignal;
  readonly projectionKind: DetectedPowerCaptureProjectionKind;
}

/**
 * Reproduce the deployed selection policy without calling the shared
 * classificationCaptureTargetProjections() implementation. This is an
 * independent release-gate oracle, not another production entry point.
 */
export function independentlyReplayCaptureTargetProjections(
  tracks: readonly DetectedSignal[],
): readonly IndependentlyReplayedCaptureTargetProjection[] {
  const occurrenceCountById = new Map<string, number>();
  for (const track of tracks) {
    occurrenceCountById.set(track.id, (occurrenceCountById.get(track.id) ?? 0) + 1);
  }

  const projectionByRawTargetId = new Map<
    string,
    IndependentlyReplayedCaptureTargetProjection
  >();
  for (const rawTarget of tracks) {
    if (occurrenceCountById.get(rawTarget.id) !== 1
      || rawTarget.state !== 'active'
      || rawTarget.missedSweeps !== 0
      || rawTarget.associationMode === 'frequency-agile-2g4-activity') continue;
    projectionByRawTargetId.set(rawTarget.id, {
      rawTarget,
      projectedRepresentative: rawTarget,
      projectionKind: 'current-active-physical-representative',
    });
  }

  const agileByRawTargetId = new Map<
    string,
    IndependentlyReplayedCaptureTargetProjection[]
  >();
  for (const projectedRepresentative of tracks) {
    const projection = independentlyBindCurrentAgileLatestMember(
      tracks,
      occurrenceCountById,
      projectedRepresentative,
    );
    if (!projection) continue;
    const claims = agileByRawTargetId.get(projection.rawTarget.id) ?? [];
    claims.push(projection);
    agileByRawTargetId.set(projection.rawTarget.id, claims);
  }
  for (const [rawTargetId, claims] of agileByRawTargetId) {
    // Two summaries claiming one current physical row are not an independently
    // attributable association. A bare candidate therefore remains excluded;
    // an independently active row remains available only through its local view.
    if (claims.length === 1) projectionByRawTargetId.set(rawTargetId, claims[0]!);
  }

  return [...projectionByRawTargetId.values()].sort((left, right) =>
    right.rawTarget.peakDbm - left.rawTarget.peakDbm
    || independentRepresentativeKey(left.rawTarget).localeCompare(
      independentRepresentativeKey(right.rawTarget),
    )
    || left.rawTarget.id.localeCompare(right.rawTarget.id));
}

function independentlyBindCurrentAgileLatestMember(
  tracks: readonly DetectedSignal[],
  occurrenceCountById: ReadonlyMap<string, number>,
  projectedRepresentative: DetectedSignal,
): IndependentlyReplayedCaptureTargetProjection | undefined {
  if (projectedRepresentative.state !== 'active'
    || projectedRepresentative.missedSweeps !== 0
    || projectedRepresentative.associationMode !== 'frequency-agile-2g4-activity'
    || projectedRepresentative.associationMissedSweeps !== 0
    || projectedRepresentative.associationRegionStartHz !== PINNED_AGILE_REGION_START_HZ
    || projectedRepresentative.associationRegionStopHz !== PINNED_AGILE_REGION_STOP_HZ
    || !observableAssociationEvidenceIsCurrentlyQualified(projectedRepresentative)) {
    return undefined;
  }
  const latestOpportunity = projectedRepresentative.associationOpportunities?.at(-1);
  const latestObservation = projectedRepresentative.associationObservations?.at(-1);
  if (!latestOpportunity
    || latestOpportunity.outcome !== 'exactly-one'
    || !latestObservation
    || latestOpportunity.sweepId !== latestObservation.sweepId
    || projectedRepresentative.associationRegionSweepIds?.at(-1)
      !== latestObservation.sweepId
    || projectedRepresentative.sweepIds.length !== 1
    || projectedRepresentative.sweepIds[0] !== latestObservation.sweepId
    || projectedRepresentative.associationMemberTrackIds?.includes(
      latestObservation.trackId,
    ) !== true
    || occurrenceCountById.get(latestObservation.trackId) !== 1) {
    return undefined;
  }
  const rawTarget = tracks.find((track) => track.id === latestObservation.trackId);
  if (!rawTarget
    || (rawTarget.state !== 'candidate' && rawTarget.state !== 'active')
    || rawTarget.missedSweeps !== 0
    || rawTarget.associationMode === 'frequency-agile-2g4-activity'
    || rawTarget.sweepIds.at(-1) !== latestObservation.sweepId
    || rawTarget.lastSeenAt !== projectedRepresentative.lastSeenAt
    || rawTarget.startHz !== projectedRepresentative.startHz
    || rawTarget.stopHz !== projectedRepresentative.stopHz
    || rawTarget.peakHz !== projectedRepresentative.peakHz
    || rawTarget.peakDbm !== projectedRepresentative.peakDbm
    || rawTarget.bandwidthHz !== projectedRepresentative.bandwidthHz
    || rawTarget.detectorId !== projectedRepresentative.detectorId) {
    return undefined;
  }
  const rawLocal = rawTarget.localClassificationObservations?.at(-1);
  const representativeLocal =
    projectedRepresentative.localClassificationObservations?.at(-1)
      ?? projectedRepresentative.classificationRegionObservation;
  if (!rawLocal
    || !representativeLocal
    || !independentLocalObservationsMatch(rawLocal, representativeLocal)) {
    return undefined;
  }
  const sourceSweep = rawLocal.sourceSweep;
  const binWidthHz = independentlyValidatedNominalBinWidth(sourceSweep.frequencyHz);
  if (binWidthHz === undefined
    || sourceSweep.complete !== true
    || sourceSweep.frequencyHz.length !== sourceSweep.powerDbm.length
    || !Number.isFinite(sourceSweep.actualRbwHz)
    || sourceSweep.actualRbwHz <= 0) return undefined;
  const expectedCenterHz = Math.max(
    PINNED_AGILE_REGION_START_HZ,
    Math.min(
      PINNED_AGILE_REGION_STOP_HZ,
      (rawLocal.startHz + rawLocal.stopHz) / 2,
    ),
  );
  const expectedStartHz = Math.min(
    expectedCenterHz,
    Math.max(sourceSweep.actualStartHz, rawLocal.startHz - binWidthHz / 2),
  );
  const expectedStopHz = Math.max(
    expectedCenterHz,
    Math.min(sourceSweep.actualStopHz, rawLocal.stopHz + binWidthHz / 2),
  );
  if (sourceSweep.id !== latestObservation.sweepId
    || sourceSweep.capturedAt !== rawTarget.lastSeenAt
    || rawLocal.startHz !== rawTarget.startHz
    || rawLocal.stopHz !== rawTarget.stopHz
    || rawLocal.peakHz !== rawTarget.peakHz
    || rawLocal.detectorId !== rawTarget.detectorId
    || latestObservation.detectorId !== rawLocal.detectorId
    || latestObservation.centerHz !== expectedCenterHz
    || latestObservation.startHz !== expectedStartHz
    || latestObservation.stopHz !== expectedStopHz
    || latestObservation.rbwHz !== sourceSweep.actualRbwHz
    || latestObservation.binWidthHz !== binWidthHz
    || !bayesianDetectionEvidenceMatches(
      latestObservation.localBayesianEvidence,
      rawLocal.localBayesianEvidence,
    )) return undefined;

  return {
    rawTarget,
    projectedRepresentative,
    projectionKind: 'current-qualified-agile-latest-member',
  };
}

function independentLocalObservationsMatch(
  left: LocalClassificationRegionObservation,
  right: LocalClassificationRegionObservation,
): boolean {
  return left.startHz === right.startHz
    && left.stopHz === right.stopHz
    && left.peakHz === right.peakHz
    && left.detectorId === right.detectorId
    && left.sourceSweep.id === right.sourceSweep.id
    && left.sourceSweep.sequence === right.sourceSweep.sequence
    && left.sourceSweep.capturedAt === right.sourceSweep.capturedAt
    && left.sourceSweep.actualStartHz === right.sourceSweep.actualStartHz
    && left.sourceSweep.actualStopHz === right.sourceSweep.actualStopHz
    && left.sourceSweep.actualRbwHz === right.sourceSweep.actualRbwHz
    && sameNumbers(left.sourceSweep.frequencyHz, right.sourceSweep.frequencyHz)
    && sameNumbers(left.sourceSweep.powerDbm, right.sourceSweep.powerDbm)
    && bayesianDetectionEvidenceMatches(
      left.localBayesianEvidence,
      right.localBayesianEvidence,
    );
}

function independentlyValidatedNominalBinWidth(
  frequencies: readonly number[],
): number | undefined {
  if (frequencies.length < 2 || frequencies.some((frequency) => !Number.isFinite(frequency))) {
    return undefined;
  }
  const differences = frequencies.slice(1).map(
    (frequency, index) => frequency - frequencies[index]!,
  );
  if (differences.some((difference) => !Number.isFinite(difference) || difference <= 0)) {
    return undefined;
  }
  const ordered = [...differences].sort((left, right) => left - right);
  const middle = Math.floor(ordered.length / 2);
  return ordered.length % 2 === 0
    ? (ordered[middle - 1]! + ordered[middle]!) / 2
    : ordered[middle]!;
}

function sameNumbers(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length
    && left.every((value, index) => value === right[index]);
}

function independentRepresentativeKey(track: DetectedSignal): string {
  const associationMode = track.associationMode ?? 'frequency-local';
  return `${associationMode}:${associationMode === 'frequency-local'
    ? track.id
    : track.associationId ?? track.id}`;
}
