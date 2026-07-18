import type {
  DetectedPowerCaptureProjectionKind,
  DetectedSignal,
  LocalClassificationRegionObservation,
} from '../../Atom-Atomizer/packages/contracts/src/index.js';
import { bayesianDetectionEvidenceMatches } from '../../Atom-Atomizer/packages/analysis/src/bayesian-signal-detector.js';
import { observableAssociationEvidenceIsCurrentlyQualified } from '../../Atom-Atomizer/packages/analysis/src/observable-features.js';

const PINNED_AGILE_REGION_START_HZ = 2_402_000_000;
const PINNED_AGILE_REGION_STOP_HZ = 2_480_000_000;

/** Validator-owned replay of the v4 actuation/evidence projection policy. */
export interface IndependentlyReplayedCaptureTargetProjection {
  readonly rawTarget: DetectedSignal;
  readonly projectedRepresentative: DetectedSignal;
  readonly projectionKind: DetectedPowerCaptureProjectionKind;
  readonly rankEvidence: IndependentCaptureTargetRankEvidence;
}

export interface IndependentCaptureTargetRankEvidence {
  readonly sourceSweepId: string;
  readonly supportStartHz: number;
  readonly supportStopHz: number;
  readonly supportCellCount: number;
  readonly robustFloorDbm: number;
  readonly actualRbwHz: number;
  readonly integratedExcessPowerMw: number;
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
    const rankEvidence = independentlyReplayCaptureTargetRankEvidence(rawTarget);
    if (!rankEvidence) continue;
    projectionByRawTargetId.set(rawTarget.id, {
      rawTarget,
      projectedRepresentative: rawTarget,
      projectionKind: 'current-active-physical-representative',
      rankEvidence,
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
    (left.rankEvidence.integratedExcessPowerMw
      === right.rankEvidence.integratedExcessPowerMw
      ? 0
      : left.rankEvidence.integratedExcessPowerMw
        > right.rankEvidence.integratedExcessPowerMw ? -1 : 1)
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

  const rankEvidence = independentlyReplayCaptureTargetRankEvidence(rawTarget);
  if (!rankEvidence) return undefined;

  return {
    rawTarget,
    projectedRepresentative,
    projectionKind: 'current-qualified-agile-latest-member',
    rankEvidence,
  };
}

/**
 * Independently reproduce the rank model without importing the production
 * ranking helper. The release validator must catch a shared implementation
 * defect, so this deliberately duplicates the exact numeric contract.
 */
export function independentlyReplayCaptureTargetRankEvidence(
  detection: DetectedSignal,
): IndependentCaptureTargetRankEvidence | undefined {
  const observation = detection.localClassificationObservations?.at(-1)
    ?? detection.classificationRegionObservation;
  const sourceSweep = observation?.sourceSweep;
  if (!observation
    || !sourceSweep
    || !Array.isArray(detection.sweepIds)
    || typeof sourceSweep.id !== 'string'
    || sourceSweep.id.length === 0
    || sourceSweep.id !== detection.sweepIds.at(-1)
    || sourceSweep.capturedAt !== detection.lastSeenAt
    || observation.startHz !== detection.startHz
    || observation.stopHz !== detection.stopHz
    || observation.peakHz !== detection.peakHz
    || observation.detectorId !== detection.detectorId
    || sourceSweep.complete !== true
    || !Number.isFinite(sourceSweep.actualStartHz)
    || !Number.isFinite(sourceSweep.actualStopHz)
    || sourceSweep.actualStopHz <= sourceSweep.actualStartHz
    || !Number.isFinite(sourceSweep.actualRbwHz)
    || sourceSweep.actualRbwHz <= 0
    || sourceSweep.frequencyHz.length < 2
    || sourceSweep.frequencyHz.length !== sourceSweep.powerDbm.length
    || sourceSweep.frequencyHz.some((frequencyHz, index) =>
      !Number.isFinite(frequencyHz)
      || frequencyHz < sourceSweep.actualStartHz
      || frequencyHz > sourceSweep.actualStopHz
      || (index > 0 && frequencyHz <= sourceSweep.frequencyHz[index - 1]!))
    || sourceSweep.powerDbm.some((powerDbm) => !Number.isFinite(powerDbm))) {
    return undefined;
  }

  const orderedPowerDbm = [...sourceSweep.powerDbm]
    .sort((left, right) => left - right);
  const lowerTailCount = Math.max(1, Math.floor(orderedPowerDbm.length * 0.2));
  const robustFloorDbm = independentMedian(
    orderedPowerDbm.slice(0, lowerTailCount),
  );
  if (!Number.isFinite(detection.noiseFloorDbm)
    || detection.noiseFloorDbm !== robustFloorDbm) return undefined;

  const supportIndices = sourceSweep.frequencyHz
    .map((frequencyHz, index) => ({ frequencyHz, index }))
    .filter(({ frequencyHz }) => frequencyHz >= detection.startHz
      && frequencyHz <= detection.stopHz)
    .map(({ index }) => index);
  if (supportIndices.length === 0) return undefined;
  const peakIndex = sourceSweep.frequencyHz.indexOf(detection.peakHz);
  if (peakIndex < 0
    || !supportIndices.includes(peakIndex)
    || sourceSweep.powerDbm[peakIndex] !== detection.peakDbm) return undefined;

  const floorMw = independentDbmToMw(robustFloorDbm);
  let integratedExcessPowerMw = 0;
  for (const index of supportIndices) {
    const centerHz = sourceSweep.frequencyHz[index]!;
    const leftHz = index === 0
      ? sourceSweep.actualStartHz
      : (sourceSweep.frequencyHz[index - 1]! + centerHz) / 2;
    const rightHz = index === sourceSweep.frequencyHz.length - 1
      ? sourceSweep.actualStopHz
      : (centerHz + sourceSweep.frequencyHz[index + 1]!) / 2;
    const cellWidthHz = rightHz - leftHz;
    if (!Number.isFinite(cellWidthHz) || cellWidthHz <= 0) return undefined;
    integratedExcessPowerMw += Math.max(
      0,
      independentDbmToMw(sourceSweep.powerDbm[index]!) - floorMw,
    ) * cellWidthHz / sourceSweep.actualRbwHz;
  }
  if (!Number.isFinite(integratedExcessPowerMw)
    || integratedExcessPowerMw <= 0) return undefined;
  return {
    sourceSweepId: sourceSweep.id,
    supportStartHz: sourceSweep.frequencyHz[supportIndices[0]!]!,
    supportStopHz: sourceSweep.frequencyHz[supportIndices.at(-1)!]!,
    supportCellCount: supportIndices.length,
    robustFloorDbm,
    actualRbwHz: sourceSweep.actualRbwHz,
    integratedExcessPowerMw,
  };
}

function independentMedian(values: readonly number[]): number {
  const middle = Math.floor(values.length / 2);
  return values.length % 2 === 0
    ? (values[middle - 1]! + values[middle]!) / 2
    : values[middle]!;
}

function independentDbmToMw(valueDbm: number): number {
  return 10 ** (valueDbm / 10);
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
