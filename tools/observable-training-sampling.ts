import { createHash } from 'node:crypto';
import {
  synthesizeCanonicalObservation,
  type CanonicalClassificationScenario,
} from '../../TinySA_SignalLab/src/classification-corpus.js';
import {
  DETECTED_POWER_ACQUISITION_QUALIFICATION,
  extractObservableFeatures,
  ObservableEvidenceUnavailableError,
  observableAssociationEvidenceIsCurrentlyQualified,
  type ObservableFeatureObservation,
} from '../../TinySA/packages/analysis/src/observable-features.js';
import { observableRepresentativeIsInClassDomain } from '../src/observable-hypothesis-domain.js';
import {
  OBSERVABLE_TRAINING_SWEEP_POINTS,
  SIGNAL_LAB_PRODUCTION_CAPTURE_TARGET_SELECTION_POLICY_ID,
  SIGNAL_LAB_PRODUCTION_DETECTED_POWER_CAPTURE_POLICY_ID,
  observableTrainingActualRbwHz,
  observableTrainingDetectedPowerSynthesisFilterWidthHz,
  createObservableTrainingSourceClock,
  createSignalLabProductionProfileCapturePolicy,
  type ObservableTrainingAcquisitionRegime,
  type ObservableTrainingDetectedPowerClockContext,
  type ObservableTrainingDetectedPowerClockEvent,
  type ObservableTrainingSourceClockEvent,
} from '../../TinySA/packages/analysis/src/observable-training-acquisition-geometry.js';
import {
  classificationCaptureTargetProjections,
  classificationRepresentatives,
  classificationRepresentativeKey,
  createDetectedPowerCaptureReceipt,
  SignalDetector,
  SignalTracker,
} from '../../TinySA/packages/analysis/src/index.js';
import {
  OBSERVABLE_EVIDENCE_CENSORING_POLICY,
  type ObservableLeafClass,
} from '../src/observable-classifier-model.js';
import {
  detectedPowerTimeseriesConfigurationSchema,
  projectDetectedPowerTuneHz,
  SIGNAL_LAB_SCALAR_FREQUENCY_RANGE_V1,
  type DetectedPowerCaptureReceipt,
  type DetectedSignal,
  type DeviceIdentity,
  type SignalDetectionConfig,
  type Sweep,
  type ZeroSpanCapture,
} from '../../TinySA/packages/contracts/src/index.js';
import { CLASSIFICATION_CORPUS_VERSION } from '../../TinySA_SignalLab/src/classification-corpus.js';

export const CLASSIFICATION_SWEEPS = 8;
// Match the current owned live release gate. The earlier 24-opportunity model
// reference plan is not presented as the live matrix: production holds each
// standard profile for 32 spectra and each full-band 2.4 GHz profile for 96.
export const STANDARD_OBSERVATION_OPPORTUNITIES = 32;
export const FULL_BAND_2G4_OBSERVATION_OPPORTUNITIES = 96;
export const FULL_BAND_2G4_START_HZ = 2_402_000_000;
export const FULL_BAND_2G4_STOP_HZ = 2_480_000_000;
export const PRODUCTION_DETECTION_CONFIG: SignalDetectionConfig = {
  threshold: { strategy: 'noise-relative', marginDb: 10 },
  minimumBandwidthHz: 0,
  minimumProminenceDb: 6,
  minimumConsecutiveSweeps: 2,
  releaseAfterMissedSweeps: 2,
};

export const CONSECUTIVE_SPECTRUM_REPRESENTATIVE_SELECTION_POLICY_ID =
  'active-classification-representatives-v1' as const;
export const QUALIFIED_ENVELOPE_REPRESENTATIVE_SELECTION_POLICY_ID =
  SIGNAL_LAB_PRODUCTION_CAPTURE_TARGET_SELECTION_POLICY_ID;

// Diagnostic-only wall-clock accounting for the worker profiling pass. Never
// read for correctness; a slow bucket here is a perf signal, not a contract.
export const attemptSamplingTimingMs = {
  spectrumSynthesis: 0,
  zeroSpanSynthesis: 0,
  detectAndTrack: 0,
  featureExtraction: 0,
  hashing: 0,
  attemptCount: 0,
};

function timed<T>(bucket: Exclude<keyof typeof attemptSamplingTimingMs, 'attemptCount'>, fn: () => T): T {
  const start = process.hrtime.bigint();
  try {
    return fn();
  } finally {
    attemptSamplingTimingMs[bucket] += Number(process.hrtime.bigint() - start) / 1e6;
  }
}

const identity: DeviceIdentity = {
  model: 'SignalLab canonical scalar corpus', hardwareVersion: 'offline', firmwareVersion: CLASSIFICATION_CORPUS_VERSION,
  firmwareQualification: 'protocol-test',
  port: { id: 'offline', path: 'offline://classification-corpus', usbMatch: 'protocol-test-double', transport: 'protocol-test-double', execution: 'protocol-test-double' },
  simulated: true, usbIdentityVerified: false, execution: 'protocol-test-double',
};

export interface FeatureSample {
  values: Readonly<Record<string, number>>;
  observationOpportunity: number;
  fitEligible: boolean;
  envelopeUntimedFitEligible?: boolean;
  /**
   * A physical detected-power capture can be valid and receipt-bound while its
   * fixed-tune envelope is deliberately excluded from classifier evidence.
   * This is training audit metadata only; it is never a model feature.
   */
  detectedPowerEvidenceDisposition?:
    | 'admitted-envelope'
    | 'censored-frequency-agile-fixed-tune';
}

export interface ConsecutiveSpectrumFeatureSamplingAttempt {
  observationHorizon: number;
  /** Every representative that the deployed no-auto-capture spectrum path can classify online. */
  onlineSpectrumRepresentatives: readonly FeatureSample[];
  provenanceUnavailableWindowCount: number;
  sourceClockEventCount: number;
  sourceClockTraceSha256: string;
}

export interface QualifiedEnvelopeFeatureSamplingAttempt {
  observationHorizon: number;
  /**
   * The sole receipt-verified physical detected-power result. A censored
   * frequency-agile result is intentionally only a spectrum-valued audit
   * sample; it is not an envelope representative.
   */
  detectedPowerCaptureSample?: FeatureSample;
  provenanceUnavailableWindowCount: number;
  preCaptureProvenanceUnavailableWindowCount: number;
  postCaptureProvenanceUnavailableWindowCount: 0 | 1;
  physicalDetectedPowerCaptureCount: 0 | 1;
  sourceClockEventCount: number;
  sourceClockTraceSha256: string;
  capturedRepresentativeKey?: string;
}

export interface FeatureSamplingAttempt {
  consecutiveSpectrum: ConsecutiveSpectrumFeatureSamplingAttempt;
  qualifiedEnvelope: QualifiedEnvelopeFeatureSamplingAttempt;
}

export interface FeatureSamplingProgress {
  stage: 'consecutive-spectrum' | 'qualified-envelope';
  observationOpportunity: number;
  observationHorizon: number;
}

export function envelopeUntimed(sample: Readonly<Record<string, number>>): Readonly<Record<string, number>> {
  return Object.fromEntries(Object.entries(sample).filter(([name]) => !name.startsWith('envelope.periodicEnergy') && name !== 'envelope.logTransitionRateHz'));
}

function filterToNeededSweeps(allSweeps: readonly Sweep[], neededIds: readonly string[]): Sweep[] {
  const neededIdSet = new Set(neededIds);
  return allSweeps.filter((sweep) => neededIdSet.has(sweep.id));
}

export function observableModelClass(scenario: CanonicalClassificationScenario): ObservableLeafClass {
  return scenario.truthClass === 'bluetooth-classic-like'
    || scenario.truthClass === 'bluetooth-le-like'
    ? 'bluetooth-like'
    : scenario.truthClass;
}

export function featureSamples(
  scenario: CanonicalClassificationScenario,
  snrDb: number,
  acquisitionRegime: ObservableTrainingAcquisitionRegime,
  seed: number,
  onProgress?: (progress: FeatureSamplingProgress) => void,
): FeatureSamplingAttempt {
  attemptSamplingTimingMs.attemptCount += 1;
  const actualRbwHz = observableTrainingActualRbwHz(scenario, acquisitionRegime.geometry);
  const detectedPowerSynthesisFilterWidthHz =
    observableTrainingDetectedPowerSynthesisFilterWidthHz(scenario, acquisitionRegime.geometry);
  const observationHorizon = observationOpportunityHorizon(scenario);
  const consecutiveSpectrum = consecutiveSpectrumFeatureSamples(
    scenario,
    snrDb,
    seed,
    actualRbwHz,
    detectedPowerSynthesisFilterWidthHz,
    observationHorizon,
    acquisitionRegime.spectrumTemporalSchedule.sourceLookIndexOffset,
    onProgress,
  );
  const qualifiedEnvelope = qualifiedEnvelopeFeatureSample(
    scenario,
    snrDb,
    seed,
    actualRbwHz,
    detectedPowerSynthesisFilterWidthHz,
    observationHorizon,
    acquisitionRegime.qualifiedEnvelopeTemporalSchedule.sourceLookIndexOffset,
    onProgress,
  );
  return { consecutiveSpectrum, qualifiedEnvelope };
}

function consecutiveSpectrumFeatureSamples(
  scenario: CanonicalClassificationScenario,
  snrDb: number,
  seed: number,
  actualRbwHz: number,
  detectedPowerSynthesisFilterWidthHz: number,
  observationHorizon: number,
  sourceLookIndexOffset: number,
  onProgress?: (progress: FeatureSamplingProgress) => void,
): ConsecutiveSpectrumFeatureSamplingAttempt {
  const sourceClock = createObservableTrainingSourceClock(
    sourceLookIndexOffset,
  );
  const detector = new SignalDetector(PRODUCTION_DETECTION_CONFIG);
  const tracker = new SignalTracker(PRODUCTION_DETECTION_CONFIG);
  const sweeps: Sweep[] = [];
  const onlineSpectrumRepresentatives: FeatureSample[] = [];
  let provenanceUnavailableWindowCount = 0;
  for (let spectrumOpportunity = 0; spectrumOpportunity < observationHorizon; spectrumOpportunity += 1) {
    onProgress?.({
      stage: 'consecutive-spectrum',
      observationOpportunity: spectrumOpportunity + 1,
      observationHorizon,
    });
    const spectrumEvent = sourceClock.allocateSpectrum({
      contextId: scenario.id,
      spectrumOpportunity,
    });
    const sweep = timed('spectrumSynthesis', () => synthesizeSpectrumSweep(
      scenario,
      spectrumEvent.lookIndex,
      seed,
      snrDb,
      actualRbwHz,
      detectedPowerSynthesisFilterWidthHz,
    ));
    sweeps.push(sweep);
    const tracks = timed('detectAndTrack', () => tracker.update(sweep, detector.analyze(sweep)));
    const ready = readyConsecutiveSpectrumClassificationRepresentatives(tracks);
    for (const { detection, representativeKey } of ready) {
      // coherentSweeps() re-validates every element of its input on every
      // call. For frequency-local association (the common case), it only
      // ever selects from detection.sweepIds.slice(-CLASSIFICATION_SWEEPS),
      // so handing it the full growing history makes each call's cost scale
      // with attempt-so-far length instead of a constant window — quadratic
      // in observation horizon across an attempt. detection.sweepIds is a
      // sparse admission record (only sweeps where this track was actually
      // re-detected), not necessarily a positional tail of our own `sweeps`
      // log, so filter by the exact IDs coherentSweeps needs rather than by
      // position. Non-local association modes (agile/multicomponent/
      // regular-component) can reference a wider associationRegionSweepIds
      // window, so those keep the full array.
      const evidenceSweeps = (detection.associationMode === undefined || detection.associationMode === 'frequency-local')
        ? filterToNeededSweeps(sweeps, detection.sweepIds.slice(-CLASSIFICATION_SWEEPS))
        : [...sweeps];
      let spectrumFeatureObservation: ReturnType<typeof extractObservableFeatures>;
      try {
        spectrumFeatureObservation = timed('featureExtraction', () => extractObservableFeatures(detection, { sweeps: evidenceSweeps }));
      } catch (error) {
        if (error instanceof Error
          && error.message === 'Observable classification requires at least one coherent complete scalar sweep') {
          throw new Error(
            `Consecutive-spectrum evidence became incoherent for ${scenario.id} at SNR ${snrDb} dB, seed ${seed}, opportunity ${spectrumOpportunity + 1}, representative ${representativeKey}`,
            { cause: error },
          );
        }
        if (error instanceof ObservableEvidenceUnavailableError
          && (error.code === 'local-history-not-uniquely-replayable'
            || error.code === 'insufficient-roi-bins')) {
          provenanceUnavailableWindowCount += 1;
          continue;
        }
        throw error;
      }
      assertExactClassificationWindow(detection, spectrumFeatureObservation);
      onlineSpectrumRepresentatives.push({
        values: spectrumFeatureObservation.values,
        observationOpportunity: spectrumOpportunity + 1,
        fitEligible: observableRepresentativeIsInClassDomain(
          scenario.truthClass === 'bluetooth-classic-like' || scenario.truthClass === 'bluetooth-le-like'
            ? 'bluetooth-like'
            : scenario.truthClass,
          spectrumFeatureObservation,
        ),
      });
    }
  }
  const sourceClockTrace = sourceClock.trace();
  assertConsecutiveSpectrumSourceClockTrace(
    sourceClockTrace,
    scenario.id,
    sourceLookIndexOffset,
    observationHorizon,
  );
  const sourceClockTraceSha256 = timed('hashing', () => createHash('sha256')
    .update(JSON.stringify(sourceClockTrace))
    .digest('hex'));
  return {
    observationHorizon,
    onlineSpectrumRepresentatives,
    provenanceUnavailableWindowCount,
    sourceClockEventCount: sourceClockTrace.length,
    sourceClockTraceSha256,
  };
}

function qualifiedEnvelopeFeatureSample(
  scenario: CanonicalClassificationScenario,
  snrDb: number,
  seed: number,
  actualRbwHz: number,
  detectedPowerSynthesisFilterWidthHz: number,
  observationHorizon: number,
  sourceLookIndexOffset: number,
  onProgress?: (progress: FeatureSamplingProgress) => void,
): QualifiedEnvelopeFeatureSamplingAttempt {
  const sourceClock = createObservableTrainingSourceClock(sourceLookIndexOffset);
  const capturePolicy = createSignalLabProductionProfileCapturePolicy(sourceClock, scenario.id);
  const detector = new SignalDetector(PRODUCTION_DETECTION_CONFIG);
  const tracker = new SignalTracker(PRODUCTION_DETECTION_CONFIG);
  const sweeps: Sweep[] = [];
  let detectedPowerCaptureSample: FeatureSample | undefined;
  let capturedRepresentativeKey: string | undefined;
  let capturedTargetAttribution:
    Readonly<Omit<ObservableTrainingDetectedPowerClockContext, 'contextId'>> | undefined;
  let provenanceUnavailableWindowCount = 0;
  let preCaptureProvenanceUnavailableWindowCount = 0;
  let postCaptureProvenanceUnavailableWindowCount: 0 | 1 = 0;
  for (let spectrumOpportunity = 0; spectrumOpportunity < observationHorizon; spectrumOpportunity += 1) {
    onProgress?.({
      stage: 'qualified-envelope',
      observationOpportunity: spectrumOpportunity + 1,
      observationHorizon,
    });
    const spectrumEvent = capturePolicy.allocateSpectrum(spectrumOpportunity);
    const sweep = timed('spectrumSynthesis', () => synthesizeSpectrumSweep(
      scenario,
      spectrumEvent.lookIndex,
      seed,
      snrDb,
      actualRbwHz,
      detectedPowerSynthesisFilterWidthHz,
    ));
    sweeps.push(sweep);
    const tracks = timed('detectAndTrack', () => tracker.update(sweep, detector.analyze(sweep)));
    if (capturePolicy.detectedPowerCapture() !== null) continue;
    const ready = readyQualifiedEnvelopeCaptureTargetRepresentatives(tracks);
    for (const { rawTarget, detection, representativeKey } of ready) {
      // coherentSweeps() re-validates every element of its input on every
      // call. For frequency-local association (the common case), it only
      // ever selects from detection.sweepIds.slice(-CLASSIFICATION_SWEEPS),
      // so handing it the full growing history makes each call's cost scale
      // with attempt-so-far length instead of a constant window — quadratic
      // in observation horizon across an attempt. detection.sweepIds is a
      // sparse admission record (only sweeps where this track was actually
      // re-detected), not necessarily a positional tail of our own `sweeps`
      // log, so filter by the exact IDs coherentSweeps needs rather than by
      // position. Non-local association modes (agile/multicomponent/
      // regular-component) can reference a wider associationRegionSweepIds
      // window, so those keep the full array.
      const evidenceSweeps = (detection.associationMode === undefined || detection.associationMode === 'frequency-local')
        ? filterToNeededSweeps(sweeps, detection.sweepIds.slice(-CLASSIFICATION_SWEEPS))
        : [...sweeps];
      let spectrumFeatureObservation: ReturnType<typeof extractObservableFeatures>;
      try {
        spectrumFeatureObservation = timed('featureExtraction', () => extractObservableFeatures(detection, { sweeps: evidenceSweeps }));
      } catch (error) {
        if (error instanceof ObservableEvidenceUnavailableError
          && (error.code === 'local-history-not-uniquely-replayable'
            || error.code === 'insufficient-roi-bins')) {
          provenanceUnavailableWindowCount += 1;
          preCaptureProvenanceUnavailableWindowCount += 1;
          continue;
        }
        throw error;
      }
      assertExactClassificationWindow(detection, spectrumFeatureObservation);
      const zeroSpanTuneHz = projectDetectedPowerTuneHz(
        rawTarget.peakHz,
        SIGNAL_LAB_SCALAR_FREQUENCY_RANGE_V1,
      );
      const targetAttribution = {
        targetSelectionPolicyId:
          SIGNAL_LAB_PRODUCTION_CAPTURE_TARGET_SELECTION_POLICY_ID,
        rawTargetId: rawTarget.id,
        projectedRepresentativeId: detection.id,
        representativeKey,
        selectedPeakHz: rawTarget.peakHz,
        selectedPeakDbm: rawTarget.peakDbm,
        admittedTuneHz: zeroSpanTuneHz,
      } as const;
      const captureEvent = capturePolicy.captureAfterRuntimeAdmission(
        spectrumEvent,
        targetAttribution,
      );
      if (!captureEvent) {
        throw new Error('Qualified-envelope capture policy rejected its first admitted representative');
      }
      capturedRepresentativeKey = representativeKey;
      capturedTargetAttribution = targetAttribution;
      const zeroSpanObservation = timed('zeroSpanSynthesis', () => synthesizeCanonicalObservation(scenario.id, {
        lookIndex: captureEvent.lookIndex,
        seed,
        snrDb,
        actualRbwHz,
        detectedPowerSynthesisFilterWidthHz,
        points: OBSERVABLE_TRAINING_SWEEP_POINTS,
        sweepTimeSeconds: 0.05,
        zeroSpanPoints: OBSERVABLE_TRAINING_SWEEP_POINTS,
        zeroSpanSamplePeriodSeconds: 1 / 9_000,
        zeroSpanFrequencyHz: zeroSpanTuneHz,
      }));
      assertDetectedPowerSynthesisProvenance(
        zeroSpanObservation,
        detectedPowerSynthesisFilterWidthHz,
        `${scenario.id} qualified-envelope detected-power observation`,
      );
      const zeroSpanCapture = asZeroSpan(zeroSpanObservation, rawTarget);
      const detectedPowerCaptureReceipt = createDetectedPowerCaptureReceipt({
        activeSignals: tracks,
        evidenceSweeps: sweeps,
        capture: zeroSpanCapture,
        admittedTargetTuneHz: captureEvent.admittedTuneHz,
        spectrumSweepIds: spectrumFeatureObservation.sweepIds,
      });
      assertCaptureReceiptMatchesSourceClock({
        receipt: detectedPowerCaptureReceipt,
        capture: zeroSpanCapture,
        capturePolicyId: capturePolicy.policyId,
        captureEvent,
        targetAttribution,
        detection,
        representativeKey,
        spectrumSweepIds: spectrumFeatureObservation.sweepIds,
      });
      let envelopeFeatureObservation: ReturnType<typeof extractObservableFeatures>;
      try {
        envelopeFeatureObservation = timed('featureExtraction', () => extractObservableFeatures(detection, {
          sweeps: evidenceSweeps,
          zeroSpan: zeroSpanCapture,
          zeroSpanSpectrumSweepIds: spectrumFeatureObservation.sweepIds,
          detectedPowerCaptureReceipt,
        }));
      } catch (error) {
        if (error instanceof ObservableEvidenceUnavailableError
          && (error.code === 'local-history-not-uniquely-replayable'
            || error.code === 'insufficient-roi-bins')) {
          postCaptureProvenanceUnavailableWindowCount = 1;
          throw new Error(
            `Qualified-envelope capture for ${scenario.id} became unavailable after the identical spectrum window had passed extraction`,
            { cause: error },
          );
        }
        throw error;
      }
      const frequencyAgileFixedTuneEnvelopeCensored =
        envelopeFeatureObservation.limitations.includes(
          'frequency-agile-fixed-tune-envelope-censored',
        );
      const envelopeAdmitted =
        envelopeFeatureObservation.zeroSpanCaptureId !== undefined
        && envelopeFeatureObservation.detectedPowerAcquisitionQualification
          === DETECTED_POWER_ACQUISITION_QUALIFICATION;
      if (frequencyAgileFixedTuneEnvelopeCensored === envelopeAdmitted) {
        throw new Error(
          'Qualified detected-power capture must be admitted as envelope evidence or explicitly censored for a fixed-tune agile association',
        );
      }
      if (frequencyAgileFixedTuneEnvelopeCensored) {
        if (detection.associationMode
          !== OBSERVABLE_EVIDENCE_CENSORING_POLICY.associationMode) {
          throw new Error(
            'Fixed-tune agile envelope censoring was applied outside its declared association mode',
          );
        }
        assertExactSpectrumOnlyCensoredValues(
          spectrumFeatureObservation.values,
          envelopeFeatureObservation.values,
        );
      }
      detectedPowerCaptureSample = {
        // A censored physical capture contributes no detected-power feature,
        // even accidentally: retain the exact already-admitted spectrum
        // values rather than a post-capture reconstruction.
        values: frequencyAgileFixedTuneEnvelopeCensored
          ? spectrumFeatureObservation.values
          : envelopeFeatureObservation.values,
        observationOpportunity: spectrumOpportunity + 1,
        fitEligible: envelopeAdmitted && observableRepresentativeIsInClassDomain(
          observableModelClass(scenario),
          envelopeFeatureObservation,
        ),
        envelopeUntimedFitEligible: envelopeAdmitted && observableRepresentativeIsInClassDomain(
          observableModelClass(scenario),
          {
            ...envelopeFeatureObservation,
            values: envelopeUntimed(envelopeFeatureObservation.values),
          },
        ),
        detectedPowerEvidenceDisposition: envelopeAdmitted
          ? 'admitted-envelope'
          : 'censored-frequency-agile-fixed-tune',
      };
      break;
    }
  }
  const sourceClockTrace = sourceClock.trace();
  const physicalDetectedPowerCaptureCountValue = sourceClockTrace
    .filter((event) => event.kind === 'detected-power').length;
  if (physicalDetectedPowerCaptureCountValue !== 0 && physicalDetectedPowerCaptureCountValue !== 1) {
    throw new Error(`Qualified-envelope attempt consumed ${physicalDetectedPowerCaptureCountValue} detected-power captures`);
  }
  const physicalDetectedPowerCaptureCount = physicalDetectedPowerCaptureCountValue as 0 | 1;
  if (physicalDetectedPowerCaptureCount
    !== (detectedPowerCaptureSample === undefined ? 0 : 1)
      + postCaptureProvenanceUnavailableWindowCount) {
    throw new Error('Qualified-envelope capture accounting lost a consumed physical acquisition');
  }
  assertQualifiedEnvelopeSourceClockTrace(
    sourceClockTrace,
    scenario.id,
    sourceLookIndexOffset,
    observationHorizon,
    physicalDetectedPowerCaptureCount,
    capturedTargetAttribution,
  );
  const sourceClockTraceSha256 = timed('hashing', () => createHash('sha256')
    .update(JSON.stringify(sourceClockTrace))
    .digest('hex'));
  return {
    observationHorizon,
    ...(detectedPowerCaptureSample === undefined ? {} : { detectedPowerCaptureSample }),
    provenanceUnavailableWindowCount,
    preCaptureProvenanceUnavailableWindowCount,
    postCaptureProvenanceUnavailableWindowCount,
    physicalDetectedPowerCaptureCount,
    sourceClockEventCount: sourceClockTrace.length,
    sourceClockTraceSha256,
    ...(capturedRepresentativeKey === undefined ? {} : { capturedRepresentativeKey }),
  };
}

function assertExactSpectrumOnlyCensoredValues(
  admittedSpectrumValues: Readonly<Record<string, number>>,
  censoredValues: Readonly<Record<string, number>>,
): void {
  if (Object.keys(censoredValues).some((name) => name.startsWith('envelope.'))
    || JSON.stringify(censoredValues) !== JSON.stringify(admittedSpectrumValues)) {
    throw new Error(
      'A fixed-tune agile censored capture must preserve the exact admitted spectrum-only feature vector',
    );
  }
}

function synthesizeSpectrumSweep(
  scenario: CanonicalClassificationScenario,
  lookIndex: number,
  seed: number,
  snrDb: number,
  actualRbwHz: number,
  detectedPowerSynthesisFilterWidthHz: number,
): Sweep {
  const observation = synthesizeCanonicalObservation(scenario.id, {
    lookIndex,
    seed,
    snrDb,
    actualRbwHz,
    detectedPowerSynthesisFilterWidthHz,
    points: OBSERVABLE_TRAINING_SWEEP_POINTS,
    sweepTimeSeconds: 0.05,
    zeroSpanPoints: OBSERVABLE_TRAINING_SWEEP_POINTS,
    zeroSpanSamplePeriodSeconds: 1 / 9_000,
  });
  assertDetectedPowerSynthesisProvenance(
    observation,
    detectedPowerSynthesisFilterWidthHz,
    `${scenario.id} swept observation`,
  );
  return asSweep(scenario, observation);
}

/** Match the deployed no-auto-capture spectrum branch: collapse each disclosed
 * association exactly once after excluding every inactive tracker row. */
export function consecutiveSpectrumClassificationRepresentatives(
  tracks: readonly DetectedSignal[],
): readonly DetectedSignal[] {
  return classificationRepresentatives(
    tracks.filter((track) => track.state === 'active'),
  );
}

/** Match the deployed detected-power branch: rank every selectable raw row
 * before projecting it to the representative retained for that physical target. */
export function qualifiedEnvelopeCaptureTargetRepresentatives(
  tracks: readonly DetectedSignal[],
): readonly DetectedSignal[] {
  return classificationCaptureTargetProjections(tracks)
    .map((projection) => projection.projectedRepresentative);
}

function readyConsecutiveSpectrumClassificationRepresentatives(
  tracks: readonly DetectedSignal[],
) {
  return readyClassificationRepresentatives(
    consecutiveSpectrumClassificationRepresentatives(tracks),
  );
}

function readyQualifiedEnvelopeCaptureTargetRepresentatives(
  tracks: readonly DetectedSignal[],
) {
  return classificationCaptureTargetProjections(tracks)
    .filter(({ projectedRepresentative }) =>
      classificationSourceSweepIds(projectedRepresentative).length
        >= CLASSIFICATION_SWEEPS)
    .filter(({ projectedRepresentative }) =>
      observableAssociationEvidenceIsCurrentlyQualified(
        projectedRepresentative,
      ))
    .map(({ rawTarget, projectedRepresentative }) => ({
      rawTarget,
      detection: projectedRepresentative,
      representativeKey: classificationRepresentativeKey(
        projectedRepresentative,
      ),
    }));
}

function readyClassificationRepresentatives(
  representatives: readonly DetectedSignal[],
) {
  return representatives
    .filter((track) => classificationSourceSweepIds(track).length >= CLASSIFICATION_SWEEPS)
    // Tracker hysteresis deliberately keeps a recently promoted association
    // visible. A runtime population includes only currently qualified evidence.
    .filter(observableAssociationEvidenceIsCurrentlyQualified)
    .map((detection) => ({
      detection,
      representativeKey: classificationRepresentativeKey(detection),
    }));
}

function assertExactClassificationWindow(
  detection: DetectedSignal,
  observation: ObservableFeatureObservation,
): void {
  const expectedSweepIds = classificationSourceSweepIds(detection).slice(-CLASSIFICATION_SWEEPS);
  if (expectedSweepIds.length !== CLASSIFICATION_SWEEPS
    || observation.sweepIds.length !== CLASSIFICATION_SWEEPS) {
    throw new Error(`Classifier branch spectrum window has ${expectedSweepIds.length} admitted / ${observation.sweepIds.length} extracted source sweeps, expected ${CLASSIFICATION_SWEEPS}`);
  }
  const observedSweepIds = [...observation.sweepIds].sort();
  const admittedSweepIds = [...expectedSweepIds].sort();
  if (observedSweepIds.some((id, index) => id !== admittedSweepIds[index])) {
    throw new Error(`Classifier branch spectrum window does not preserve the latest ${CLASSIFICATION_SWEEPS} effective source sweeps for ${detection.id}`);
  }
}

function assertBaseSourceClockTrace(
  trace: readonly Readonly<ObservableTrainingSourceClockEvent>[],
  sourceLookIndexOffset: number,
): void {
  if (trace.some((event, index) => event.acquisitionOrdinal !== index
    || event.lookIndex !== sourceLookIndexOffset + index)) {
    throw new Error('SignalLab branch source-clock trace is not contiguous, unique, and strictly attributed');
  }
}

function assertConsecutiveSpectrumSourceClockTrace(
  trace: readonly Readonly<ObservableTrainingSourceClockEvent>[],
  scenarioId: string,
  sourceLookIndexOffset: number,
  observationHorizon: number,
): void {
  assertBaseSourceClockTrace(trace, sourceLookIndexOffset);
  if (trace.length !== observationHorizon
    || trace.some((event, index) => event.kind !== 'swept-spectrum'
      || event.contextId !== scenarioId
      || event.spectrumOpportunity !== index)) {
    throw new Error('Consecutive-spectrum branch consumed anything other than its exact spectrum horizon');
  }
}

function assertQualifiedEnvelopeSourceClockTrace(
  trace: readonly Readonly<ObservableTrainingSourceClockEvent>[],
  scenarioId: string,
  sourceLookIndexOffset: number,
  observationHorizon: number,
  physicalDetectedPowerCaptureCount: 0 | 1,
  capturedTargetAttribution:
    Readonly<Omit<ObservableTrainingDetectedPowerClockContext, 'contextId'>> | undefined,
): void {
  assertBaseSourceClockTrace(trace, sourceLookIndexOffset);
  const spectrumEvents = trace.filter((event) => event.kind === 'swept-spectrum');
  const detectedPowerEvents = trace.filter((event) => event.kind === 'detected-power');
  if (trace.length !== observationHorizon + physicalDetectedPowerCaptureCount
    || spectrumEvents.length !== observationHorizon
    || spectrumEvents.some((event, index) =>
      event.contextId !== scenarioId || event.spectrumOpportunity !== index)
    || detectedPowerEvents.length !== physicalDetectedPowerCaptureCount
    || detectedPowerEvents.some((event) => event.contextId !== scenarioId
      || capturedTargetAttribution === undefined
      || event.targetSelectionPolicyId
        !== capturedTargetAttribution.targetSelectionPolicyId
      || event.rawTargetId !== capturedTargetAttribution.rawTargetId
      || event.projectedRepresentativeId
        !== capturedTargetAttribution.projectedRepresentativeId
      || event.representativeKey !== capturedTargetAttribution.representativeKey
      || event.selectedPeakHz !== capturedTargetAttribution.selectedPeakHz
      || event.selectedPeakDbm !== capturedTargetAttribution.selectedPeakDbm
      || event.admittedTuneHz !== capturedTargetAttribution.admittedTuneHz
      || event.lookIndex !== event.triggerSpectrumLookIndex + 1
      || event.acquisitionOrdinal !== event.triggerSpectrumAcquisitionOrdinal + 1)) {
    throw new Error('Qualified-envelope branch source-clock trace violates its sole immediate-capture policy');
  }
}

function assertCaptureReceiptMatchesSourceClock({
  receipt,
  capture,
  capturePolicyId,
  captureEvent,
  targetAttribution,
  detection,
  representativeKey,
  spectrumSweepIds,
}: {
  receipt: DetectedPowerCaptureReceipt;
  capture: ZeroSpanCapture;
  capturePolicyId: typeof SIGNAL_LAB_PRODUCTION_DETECTED_POWER_CAPTURE_POLICY_ID;
  captureEvent: Readonly<ObservableTrainingDetectedPowerClockEvent>;
  targetAttribution: Readonly<Omit<ObservableTrainingDetectedPowerClockContext, 'contextId'>>;
  detection: DetectedSignal;
  representativeKey: string;
  spectrumSweepIds: readonly string[];
}): void {
  const selectedCandidate = receipt.candidates.find((candidate) =>
    candidate.rawTargetId === receipt.selection.rawTargetId);
  const sameSpectrumWindow = receipt.spectrumSweepIds.length === spectrumSweepIds.length
    && receipt.spectrumSweepIds.every((sweepId, index) =>
      sweepId === spectrumSweepIds[index]);
  if (receipt.capturePolicyId !== capturePolicyId
    || receipt.targetSelectionPolicyId !== captureEvent.targetSelectionPolicyId
    || captureEvent.targetSelectionPolicyId !== targetAttribution.targetSelectionPolicyId
    || receipt.selection.rawTargetId !== captureEvent.rawTargetId
    || captureEvent.rawTargetId !== targetAttribution.rawTargetId
    || receipt.selection.projectedRepresentativeId
      !== captureEvent.projectedRepresentativeId
    || captureEvent.projectedRepresentativeId
      !== targetAttribution.projectedRepresentativeId
    || receipt.projectedRepresentative.id !== detection.id
    || detection.id !== targetAttribution.projectedRepresentativeId
    || selectedCandidate === undefined
    || selectedCandidate.projectedRepresentativeId
      !== targetAttribution.projectedRepresentativeId
    || captureEvent.representativeKey !== representativeKey
    || representativeKey !== targetAttribution.representativeKey
    || selectedCandidate.currentPeakHz !== captureEvent.selectedPeakHz
    || captureEvent.selectedPeakHz !== targetAttribution.selectedPeakHz
    || selectedCandidate.currentPeakDbm !== captureEvent.selectedPeakDbm
    || captureEvent.selectedPeakDbm !== targetAttribution.selectedPeakDbm
    || receipt.capture.targetDetectionId !== targetAttribution.rawTargetId
    || capture.targetDetectionId !== targetAttribution.rawTargetId
    || receipt.capture.admittedTargetTuneHz !== captureEvent.admittedTuneHz
    || receipt.capture.frequencyHz !== captureEvent.admittedTuneHz
    || receipt.capture.requestedCenterHz !== captureEvent.admittedTuneHz
    || capture.frequencyHz !== captureEvent.admittedTuneHz
    || capture.requested.centerHz !== captureEvent.admittedTuneHz
    || captureEvent.admittedTuneHz !== targetAttribution.admittedTuneHz
    || !sameSpectrumWindow) {
    throw new Error(
      'Qualified-envelope issued receipt disagrees with its source-clock target attribution, tune, policy, or exact spectrum window',
    );
  }
}

export function observationOpportunityHorizon(scenario: CanonicalClassificationScenario): number {
  const startHz = scenario.centerHz - scenario.recommendedSpanHz / 2;
  const stopHz = scenario.centerHz + scenario.recommendedSpanHz / 2;
  return startHz <= FULL_BAND_2G4_START_HZ && stopHz >= FULL_BAND_2G4_STOP_HZ
    ? FULL_BAND_2G4_OBSERVATION_OPPORTUNITIES
    : STANDARD_OBSERVATION_OPPORTUNITIES;
}

function classificationSourceSweepIds(track: DetectedSignal): readonly string[] {
  return track.associationMode !== undefined && track.associationMode !== 'frequency-local'
    ? track.associationRegionSweepIds ?? []
    : track.sweepIds;
}

function assertDetectedPowerSynthesisProvenance(
  observation: ReturnType<typeof synthesizeCanonicalObservation>,
  expectedFilterWidthHz: number,
  context: string,
): void {
  if (observation.detectedPowerActualRbwHz !== null
    || observation.detectedPowerSynthesisFilterWidthHz !== expectedFilterWidthHz) {
    throw new Error(`${context} does not preserve unavailable measured RBW and the explicit synthesis-filter width`);
  }
}

function asSweep(scenario: CanonicalClassificationScenario, observation: ReturnType<typeof synthesizeCanonicalObservation>): Sweep {
  const startHz = observation.frequencyHz[0]!;
  const stopHz = observation.frequencyHz.at(-1)!;
  return {
    kind: 'spectrum', id: `${scenario.id}-${observation.seed}-${observation.lookIndex}`, sequence: observation.lookIndex + 1,
    capturedAt: new Date(Date.UTC(2026, 0, 1) + observation.lookIndex * observation.sweepTimeSeconds * 1_000).toISOString(), elapsedMilliseconds: observation.sweepTimeSeconds * 1_000,
    frequencyHz: observation.frequencyHz, powerDbm: observation.powerDbm,
    requested: {
      kind: 'swept-spectrum', startHz, stopHz, points: observation.frequencyHz.length,
      sweepTimeSeconds: observation.sweepTimeSeconds,
      controls: {
        schemaVersion: 1, model: 'receiver', acquisitionFormat: 'text',
        resolutionBandwidthKhz: observation.actualRbwHz / 1_000, attenuationDb: 'auto',
        detector: 'sample', spurRejection: 'off', lowNoiseAmplifier: 'off', avoidSpurs: 'off',
        trigger: { mode: 'auto' },
      },
    },
    // This offline corpus is a protocol-test double for RBW-filtered receiver
    // observations, not a SignalLab bridge measurement. The separate live
    // bridge gate exercises synthetic-grid-equivalent session provenance.
    actualStartHz: startHz, actualStopHz: stopHz, actualRbwHz: observation.actualRbwHz, actualAttenuationDb: 0,
    source: 'scan-text', complete: true, identity,
  };
}

function asZeroSpan(observation: ReturnType<typeof synthesizeCanonicalObservation>, detection: DetectedSignal): ZeroSpanCapture {
  const projectedTuneHz = projectDetectedPowerTuneHz(
    observation.zeroSpanFrequencyHz,
    SIGNAL_LAB_SCALAR_FREQUENCY_RANGE_V1,
  );
  if (projectedTuneHz !== observation.zeroSpanFrequencyHz) {
    throw new Error(`SignalLab zero-span synthesis used unprojected ${observation.zeroSpanFrequencyHz} Hz instead of admitted ${projectedTuneHz} Hz`);
  }
  const sweepTimeSeconds = observation.zeroSpanPowerDbm.length * observation.zeroSpanSamplePeriodSeconds;
  const requested = detectedPowerTimeseriesConfigurationSchema.parse({
    kind: 'detected-power-timeseries', centerHz: observation.zeroSpanFrequencyHz,
    sampleCount: observation.zeroSpanPowerDbm.length, sweepTimeSeconds,
    controls: { schemaVersion: 1, model: 'synthetic-scalar', timingQualification: 'simulation-exact' },
  });
  return {
    kind: 'zero-span', id: `zero-${observation.scenarioId}-${observation.seed}-${observation.lookIndex}`,
    sequence: observation.lookIndex + 1,
    capturedAt: new Date(Date.UTC(2026, 0, 1) + observation.lookIndex * observation.sweepTimeSeconds * 1_000).toISOString(),
    elapsedMilliseconds: sweepTimeSeconds * 1_000,
    frequencyHz: observation.zeroSpanFrequencyHz, samplePeriodSeconds: observation.zeroSpanSamplePeriodSeconds, timingQualification: 'simulation-exact',
    targetDetectionId: detection.id,
    powerDbm: observation.zeroSpanPowerDbm,
    requested,
    // This is the same contract shape admitted by the SignalLab manager/live
    // gate: exact simulated cadence, no invented physical RF filter or front
    // end attenuation. Runtime marginalizes the unavailable zero-span RBW.
    actualRbwHz: null, actualAttenuationDb: null,
    resolutionBandwidthQualification: 'unavailable', attenuationQualification: 'not-applicable',
    source: 'signal-lab-synthetic', complete: true, identity,
  };
}
