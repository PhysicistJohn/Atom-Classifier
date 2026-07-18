import type {
  DetectedPowerCaptureReceipt,
  DetectedSignal,
  Sweep,
  WaveformClassification,
  ZeroSpanCapture,
} from '../../TinySA/packages/contracts/src/index.js';
import {
  DETECTED_POWER_ACQUISITION_QUALIFICATION,
  extractObservableFeatures,
  type ObservableFeatureObservation,
  type WaveformEvidence,
} from '../../TinySA/packages/analysis/src/observable-features.js';

const FREQUENCY_AGILE_ENVELOPE_CENSORING_LIMITATION =
  'frequency-agile-fixed-tune-envelope-censored' as const;
const CLASSIFIER_SUPPORT_RANK_FEATURE =
  'model.maximumKnownSyntheticSupportRank' as const;

export interface ValidatorReceiptQualifiedCapture {
  readonly detection: DetectedSignal;
  readonly evidenceSweeps: readonly Sweep[];
  readonly spectrumObservation: ObservableFeatureObservation;
  readonly zeroSpan: ZeroSpanCapture;
  readonly detectedPowerCaptureReceipt: DetectedPowerCaptureReceipt;
}

type ValidatorFeatureExtractor = (
  detection: DetectedSignal,
  evidence: WaveformEvidence,
) => ObservableFeatureObservation;

export interface ValidatorWaveformClassifier {
  classify(
    detection: DetectedSignal,
    evidence: WaveformEvidence,
  ): Promise<WaveformClassification>;
}

/**
 * Construct the one authoritative evidence graph used by both validation
 * extraction and production classification. In particular, an agile capture
 * is not stripped before either consumer sees its issued receipt: production
 * must verify the capture and perform the fixed-tune censoring itself.
 */
export function validatorReceiptQualifiedEvidence(
  capture: ValidatorReceiptQualifiedCapture,
): WaveformEvidence {
  return {
    sweeps: capture.evidenceSweeps,
    zeroSpan: capture.zeroSpan,
    zeroSpanSpectrumSweepIds: capture.spectrumObservation.sweepIds,
    detectedPowerCaptureReceipt: capture.detectedPowerCaptureReceipt,
  };
}

export function extractValidatorReceiptQualifiedObservation(
  capture: ValidatorReceiptQualifiedCapture,
  extractFeatures: ValidatorFeatureExtractor = extractObservableFeatures,
): ObservableFeatureObservation {
  const observation = extractFeatures(
    capture.detection,
    validatorReceiptQualifiedEvidence(capture),
  );
  assertValidatorReceiptQualifiedObservation(capture, observation);
  return observation;
}

export function assertValidatorReceiptQualifiedObservation(
  capture: ValidatorReceiptQualifiedCapture,
  observation: ObservableFeatureObservation,
): void {
  if (capture.detection.associationMode === 'frequency-agile-2g4-activity') {
    assertFrequencyAgileCaptureIsCensored(capture, observation);
    return;
  }
  if (!sameStrings(observation.views, [
    'scalar-spectrum',
    'detected-power-envelope',
  ])
    || observation.zeroSpanCaptureId !== capture.zeroSpan.id
    || observation.detectedPowerAcquisitionQualification
      !== DETECTED_POWER_ACQUISITION_QUALIFICATION
    || !Object.keys(observation.values).some((name) => name.startsWith('envelope.'))
    || observation.limitations.includes(
      FREQUENCY_AGILE_ENVELOPE_CENSORING_LIMITATION,
    )) {
    throw new Error(
      'Validator receipt-qualified non-agile capture did not traverse the production detected-power envelope path',
    );
  }
}

export async function classifyValidatorReceiptQualifiedObservation(
  capture: ValidatorReceiptQualifiedCapture,
  expectedObservation: ObservableFeatureObservation,
  classifier: ValidatorWaveformClassifier,
): Promise<WaveformClassification> {
  const classification = await classifier.classify(
    capture.detection,
    validatorReceiptQualifiedEvidence(capture),
  );
  assertClassificationMatchesObservation(
    capture,
    expectedObservation,
    classification,
  );
  return classification;
}

function assertFrequencyAgileCaptureIsCensored(
  capture: ValidatorReceiptQualifiedCapture,
  observation: ObservableFeatureObservation,
): void {
  const leakedEnvelopeFeature = Object.keys(observation.values)
    .find((name) => name.startsWith('envelope.'));
  if (!sameStrings(observation.views, ['scalar-spectrum'])
    || observation.zeroSpanCaptureId !== undefined
    || observation.detectedPowerAcquisitionQualification !== undefined
    || leakedEnvelopeFeature !== undefined
    || !observation.limitations.includes(
      FREQUENCY_AGILE_ENVELOPE_CENSORING_LIMITATION,
    )
    || !sameStrings(
      observation.sweepIds,
      capture.spectrumObservation.sweepIds,
    )
    || !sameFeatureValues(
      observation.values,
      capture.spectrumObservation.values,
    )
    || !sameGeometry(observation, capture.spectrumObservation)
    || observation.associationEvidenceQualification
      !== capture.spectrumObservation.associationEvidenceQualification) {
    throw new Error(
      `Validator receipt-qualified frequency-agile capture leaked fixed-tune envelope evidence${
        leakedEnvelopeFeature === undefined ? '' : ` (${leakedEnvelopeFeature})`
      } instead of traversing the production censor`,
    );
  }
}

function assertClassificationMatchesObservation(
  capture: ValidatorReceiptQualifiedCapture,
  expected: ObservableFeatureObservation,
  classification: WaveformClassification,
): void {
  const evidence = classification.evidence;
  const featureValues = evidence.features;
  if (featureValues === undefined) {
    throw new Error(
      'Validator receipt-qualified classifier did not return observable feature evidence',
    );
  }
  const featureNames = Object.keys(featureValues).sort();
  const expectedFeatureNames = [
    ...Object.keys(expected.values),
    CLASSIFIER_SUPPORT_RANK_FEATURE,
  ].sort();
  const featureValuesMatch = sameStrings(featureNames, expectedFeatureNames)
    && Object.entries(expected.values).every(([name, value]) =>
      Object.is(featureValues[name], value))
    && Number.isFinite(featureValues[CLASSIFIER_SUPPORT_RANK_FEATURE]);
  if (!featureValuesMatch
    || !sameStrings(evidence.sweepIds, expected.sweepIds)
    || !sameStrings(evidence.views ?? [], expected.views)
    || !sameStrings(evidence.limitations ?? [], expected.limitations)
    || !Object.is(evidence.centerHz, expected.centerHz)
    || !Object.is(evidence.bandwidthHz, expected.bandwidthHz)
    || evidence.zeroSpanCaptureId !== expected.zeroSpanCaptureId
    || evidence.detectedPowerAcquisitionQualification
      !== expected.detectedPowerAcquisitionQualification) {
    throw new Error(
      'Validator receipt-qualified classifier result does not match the production observation path',
    );
  }
  if (capture.detection.associationMode === 'frequency-agile-2g4-activity'
    && (evidence.views?.includes('detected-power-envelope') === true
      || Object.keys(featureValues).some((name) => name.startsWith('envelope.'))
      || evidence.zeroSpanCaptureId !== undefined
      || evidence.detectedPowerAcquisitionQualification !== undefined
      || evidence.limitations?.includes(
        FREQUENCY_AGILE_ENVELOPE_CENSORING_LIMITATION,
      ) !== true)) {
    throw new Error(
      'Validator receipt-qualified classifier leaked fixed-tune envelope evidence for a frequency-agile capture',
    );
  }
}

function sameFeatureValues(
  left: Readonly<Record<string, number>>,
  right: Readonly<Record<string, number>>,
): boolean {
  const names = Object.keys(left).sort();
  return sameStrings(names, Object.keys(right).sort())
    && names.every((name) => Object.is(left[name], right[name]));
}

function sameGeometry(
  left: ObservableFeatureObservation,
  right: ObservableFeatureObservation,
): boolean {
  return Object.is(left.occupiedStartHz, right.occupiedStartHz)
    && Object.is(left.occupiedStopHz, right.occupiedStopHz)
    && Object.is(left.centerHz, right.centerHz)
    && Object.is(left.bandwidthHz, right.bandwidthHz)
    && Object.is(left.binWidthHz, right.binWidthHz);
}

function sameStrings(
  left: readonly string[],
  right: readonly string[],
): boolean {
  return left.length === right.length
    && left.every((value, index) => value === right[index]);
}
