import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { lstatSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from 'node:fs';
import { posix, resolve } from 'node:path';
import {
  BayesianWaveformClassifier,
  BAYESIAN_WAVEFORM_MODEL,
  inferPosterior,
  knownModelSupportRank,
  selectObservableDecision,
} from '../src/bayesian-waveform-classifier.js';
import { BAYESIAN_OBSERVABLE_MODEL } from '../src/models/bayesian-observable.generated.js';
import { BAYESIAN_OBSERVABLE_MODEL_SHA256 } from '../src/models/bayesian-observable.manifest.generated.js';
import {
  OBSERVABLE_EVIDENCE_CENSORING_POLICY,
  OBSERVABLE_EVIDENCE_VIEWS,
  OBSERVABLE_LEAF_CLASSES,
  observableModelComponents,
  observableModelView,
  type ObservableEvidenceView,
  type ObservableLeafClass,
} from '../src/observable-classifier-model.js';
import {
  DETECTED_POWER_AUTOMATIC_SELECTION_CONDITION,
  extractObservableFeatures,
  ObservableEvidenceUnavailableError,
  observableAssociationEvidenceIsCurrentlyQualified,
  type ObservableFeatureObservation,
} from '../../Atom-Atomizer/packages/analysis/src/observable-features.js';
import { observableRepresentativeIsInClassDomain } from '../src/observable-hypothesis-domain.js';
import {
  studentTModelTailProbability,
} from '../../Atom-Atomizer/packages/analysis/src/bayesian-predictive.js';
import {
  classificationCaptureTargetProjections,
  classificationRepresentatives,
  createDetectedPowerCaptureReceipt,
  SignalDetector,
  SignalTracker,
} from '../../Atom-Atomizer/packages/analysis/src/index.js';
import {
  assertDetectedPowerCaptureReceiptMatches,
} from '../../Atom-Atomizer/packages/analysis/src/detected-power-capture-receipt.js';
import { measurementIdentityKey } from '../../Atom-Atomizer/packages/analysis/src/measurement-provenance.js';
import {
  independentlyReplayCaptureTargetProjections,
  type IndependentlyReplayedCaptureTargetProjection,
} from './validator-capture-target-projection.js';
import { posteriorUnderDeclaredPrior } from './validator-prior-sensitivity.js';
import {
  classifyValidatorReceiptQualifiedObservation,
  extractValidatorReceiptQualifiedObservation,
} from './validator-receipt-qualified-capture.js';
import { nonFiniteReportNumberPaths } from './validator-numeric-report.js';
import {
  CLASSIFICATION_CORPUS_VERSION,
  canonicalClassificationScenarios,
  synthesizeCanonicalObservation,
  type CanonicalClassificationScenario,
  type ObservableSignalClass,
} from '../../Atom-SignalLab/src/classification-corpus.js';
import {
  CANONIZED_REPLAY_DETECTED_POWER_SYNTHESIS_FILTER_WIDTH_HZ,
} from '../../Atom-SignalLab/src/waveforms.js';
import {
  detectedPowerTimeseriesConfigurationSchema,
  projectDetectedPowerTuneHz,
  SIGNAL_LAB_SCALAR_FREQUENCY_RANGE_V1,
  type DetectedPowerCaptureProjectionKind,
  type DetectedPowerCaptureReceipt,
  type DetectedSignal,
  type DeviceIdentity,
  type SignalDetectionConfig,
  type Sweep,
  type ZeroSpanCapture,
} from '../../Atom-Atomizer/packages/contracts/src/index.js';

// Independent of both model fitting/calibration and the transition-model
// design audit. Eight phase/noise shifts make the finite-acquisition coverage
// gate a genuine replicated test rather than a two-seed smoke check.
const NUISANCE_SHIFT_SEEDS = [
  13_001, 13_019, 13_037, 13_063, 13_081, 13_099, 13_127, 13_151,
] as const;
const SNR_DB = [6, 10, 16, 24, 32] as const;
const HIGH_SNR_MINIMUM_DB = 24;
// These partitions are validator-owned pins. They intentionally duplicate the
// trainer policy instead of accepting model metadata as ground truth: a model
// cannot redefine its own holdout set and thereby make validation pass.
const PINNED_FITTED_UNKNOWN_SCENARIO_IDS = [
  'unknown-narrow-fsk',
  'unknown-802154',
] as const;
const PINNED_STRICT_UNKNOWN_HOLDOUT_SCENARIO_IDS = [
  'unknown-impulsive',
] as const;
const PINNED_OBSERVABLE_AMBIGUITY_STRESS_SCENARIO_IDS = [
  'unknown-chirp',
  'unknown-regular-cw-comb-4',
  'unknown-regular-cw-comb-5',
  'unknown-irregular-cw-multitone-100-210-370k',
  'unknown-stationary-intermittent-2g4',
  'unknown-simultaneous-1mhz-raster-2g4',
  'unknown-interleaved-four-channel-2g4',
  'unknown-proprietary-off-raster-fhss-2g4',
] as const;
const PINNED_EXACT_OBSERVABLE_EQUIVALENCE_PAIRS = [
  { nullScenarioId: 'unknown-instrument-spur-rbw-line', referenceScenarioId: 'cw-rbw-line' },
  { nullScenarioId: 'unknown-independent-am-equivalent-three-tone', referenceScenarioId: 'am-dsb-25k' },
  { nullScenarioId: 'unknown-independent-fm-equivalent-bessel-comb', referenceScenarioId: 'fm-beta-3' },
  { nullScenarioId: 'unknown-generic-ofdm-20m', referenceScenarioId: 'lte-band3-fdd-20m' },
  { nullScenarioId: 'unknown-generic-tdd-ofdm-10m', referenceScenarioId: 'lte-band38-tdd-10m' },
  { nullScenarioId: 'unknown-generic-ofdm-80m', referenceScenarioId: 'wifi-ofdm-80m' },
  { nullScenarioId: 'unknown-proprietary-dsss-22m', referenceScenarioId: 'wifi-hr-dsss-11m' },
] as const;
const PINNED_EXACT_OBSERVABLE_EQUIVALENCE_NULL_SCENARIO_IDS = PINNED_EXACT_OBSERVABLE_EQUIVALENCE_PAIRS
  .map((pair) => pair.nullScenarioId);
const PINNED_KNOWN_ACQUISITION_VALIDATION_ONLY_SCENARIO_IDS = [
  'gsm-900-tdma',
] as const;
const PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORED_SCENARIO_IDS = [
  'bluetooth-classic-connected',
  'bluetooth-le-advertising',
] as const;
const PINNED_COMPONENT_SOURCE_SCENARIO_COUNTS_BY_VIEW = Object.freeze({
  'spectrum-only': 18,
  'envelope-untimed': 16,
  'envelope-timed': 16,
} as const);
const PINNED_LIKELIHOOD_COMPONENT_COUNTS_BY_VIEW = Object.freeze({
  'spectrum-only': 28,
  'envelope-untimed': 26,
  'envelope-timed': 26,
} as const);
const PINNED_SCENARIO_EXCLUDED_FROM_COMPONENT_FIT_IDS = [
  ...PINNED_KNOWN_ACQUISITION_VALIDATION_ONLY_SCENARIO_IDS,
  ...PINNED_STRICT_UNKNOWN_HOLDOUT_SCENARIO_IDS,
  ...PINNED_OBSERVABLE_AMBIGUITY_STRESS_SCENARIO_IDS,
  ...PINNED_EXACT_OBSERVABLE_EQUIVALENCE_NULL_SCENARIO_IDS,
] as const;
// The one-timeslot GSM case is deliberately acquisition-limited. The chirp is
// an admitted observable-ambiguity stress case, not an expected non-admission.
// Any change to this exception list is a validator policy change, not
// something model metadata may silently broaden.
const PINNED_EXPECTED_CLASSIFICATION_NON_ADMISSION_SCENARIO_IDS = [
  'gsm-900-tdma',
] as const;
const EXACT_EQUIVALENCE_NUMERICAL_TOLERANCE = 1e-11;
// Held-out geometric interstitials between the fit/calibration divisors
// [12, 20, 35, 55, 80, 120]. None is a training or calibration grid point;
// the temporal schedule below also uses disjoint source look indices.
const RBW_DIVISORS = [15.5, 44, 98] as const;
const ADMISSION_SEED_COVERAGE_SNR_DB = [24, 32] as const;
const BLE_ADVERTISING_MINIMUM_SEED_COVERAGE = 0.5;
const ROLLING_MINIMUM_OVERALL_KNOWN_COVERAGE = 0.95;
const ROLLING_MINIMUM_OVERALL_HIERARCHICAL_ACCURACY = 0.95;
const ROLLING_MINIMUM_PER_SCENARIO_KNOWN_COVERAGE = 0.9;
const ROLLING_MINIMUM_PER_SCENARIO_HIERARCHICAL_ACCURACY = 0.9;
const CLASSIFICATION_ADMISSIONS = 8;
const PINNED_DETECTED_POWER_CAPTURE_POLICY_ID =
  'capture-once-after-rank-0-integrated-excess-current-target-runtime-admission-v3' as const;
const PINNED_CAPTURE_TARGET_SELECTION_POLICY_ID =
  'preferred-then-current-source-sweep-integrated-excess-power-physical-or-qualified-agile-member-target-v4' as const;
const PINNED_CAPTURE_RUNTIME_ADMISSION_POLICY_ID =
  'exact-eight-sweep-pre-capture-observable-feature-admission-v1' as const;
const PINNED_DETECTED_POWER_ACQUISITION_QUALIFICATION =
  'receipt-verified-provenance-bound-runtime-admitted-physical-capture-v5' as const;
const PINNED_DETECTED_POWER_SELECTION_CONDITION =
  DETECTED_POWER_AUTOMATIC_SELECTION_CONDITION;
const PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORING_POLICY = Object.freeze({
  id: 'frequency-agile-fixed-tune-envelope-censoring-v1',
  associationMode: 'frequency-agile-2g4-activity',
  runtimeCapturePolicy: 'validate-receipt-and-capture-before-censoring-v1',
  classifierEvidencePolicy: 'spectrum-only-no-detected-power-envelope-v1',
  unsupportedModelViewPolicy: 'exact-empty-components-and-calibration-v1',
} as const);
const PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORING_POLICY_ID =
  PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORING_POLICY.id;
const PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORING_LIMITATION =
  'frequency-agile-fixed-tune-envelope-censored' as const;
type DetectedPowerEvidenceDisposition =
  | 'admitted-envelope'
  | 'censored-frequency-agile-spectrum-only';
const PINNED_TRAINING_RUNTIME_IDENTITY = Object.freeze({
  policyId: 'exact-repository-node-version-v1',
  nodeVersion: '22.23.1',
  v8Version: '12.4.254.21-node.56',
});
// A finite asynchronous Wi-Fi burst/noise phase missed the eight-admission
// requirement for one of eight held-out seeds at 24 dB under 24 looks.  The
// runtime has no 24-look stop condition, so validation uses 32 standard
// opportunities rather than weakening seed coverage or detector thresholds.
const STANDARD_OBSERVATION_OPPORTUNITIES = 32;
const FULL_BAND_2G4_OBSERVATION_OPPORTUNITIES = 96;
const FULL_BAND_2G4_START_HZ = 2_402_000_000;
const FULL_BAND_2G4_STOP_HZ = 2_480_000_000;
const SELECTION_POLICY =
  'independent-consecutive-spectrum-and-integrated-excess-rank-0-runtime-admission-qualified-envelope-branches-v9' as const;
const REPRESENTATIVE_WEIGHTING_POLICY =
  'view-matched-spectrum-event-envelope-causal-attempt-weighting-v4' as const;
const LIKELIHOOD_POPULATION_POLICY =
  'independent-branch-view-matched-runtime-event-populations-v3' as const;
const PINNED_LIKELIHOOD_COMPONENT_DECOMPOSITION_POLICY = Object.freeze({
  id: 'scenario-components-with-three-shared-covariance-csma-activity-modes-v1',
  scenarioWeighting: 'equal-fitted-scenario-weight-within-class-v1',
  ordinaryScenarioModel: 'one-student-t-component-v1',
  csmaEnvelopeModel: 'csma-bursts',
  csmaPartitionFeature: 'spectrum.powerVariationDb',
  csmaModeCount: 3,
  minimumModeFitSampleCount: 3,
  csmaClustering: 'deterministic-one-dimensional-lloyd-min-median-max-v1',
  csmaModeWeighting: 'empirical-fit-event-frequency-within-scenario-and-view-v1',
  csmaCovariance: 'shared-within-mode-pooled-covariance-with-0.35-off-diagonal-retention-v1',
} as const);
const PINNED_CSMA_DECOMPOSED_SOURCE_SCENARIO_IDS = [
  'unknown-802154',
  'wifi-hr-dsss-11m',
  'wifi-ofdm-20m',
  'wifi-ofdm-40m',
  'wifi-ofdm-80m',
] as const;
const PINNED_TAIL_CALIBRATION_SCORE_UNIT =
  'one-independent-branch-acquisition-attempt-score-per-evidence-view-v4' as const;
const PINNED_TAIL_CALIBRATION_SELECTION_POLICY =
  'consecutive-spectrum-all-runtime-representatives-and-independent-integrated-excess-rank-0-envelope-sole-capture-v5' as const;
const PINNED_TAIL_CALIBRATION_AGGREGATION_POLICY =
  'consecutive-spectrum-branch-minimum-qualified-envelope-branch-sole-capture-v5' as const;
const PINNED_TAIL_CALIBRATION_RUNTIME_INTERPRETATION_POLICY =
  'spectrum-member-dominates-independent-branch-attempt-min-envelope-is-independent-sole-capture-v3' as const;
const PINNED_TAIL_CALIBRATION_STATISTICAL_INTERPRETATION = 'empirical-synthetic-reference-only-no-exchangeability-or-coverage-guarantee-v1' as const;
const PINNED_TAIL_CALIBRATION_SNR_DB = [6, 10, 16, 24, 32] as const;
const PINNED_TAIL_CALIBRATION_RBW_DIVISORS = [12, 20, 35, 55, 80, 120] as const;
const PINNED_TAIL_CALIBRATION_SEEDS = [6_407, 6_419, 6_421, 6_449, 6_451, 6_469, 6_473, 6_481] as const;
const PINNED_SIGNAL_LAB_PRODUCTION_GEOMETRY = Object.freeze({
  id: 'signal-lab-recommended-span-450-point-grid-v1',
  sourceKind: 'signal-lab',
  kind: 'recommended-span-inclusive-grid',
  sweepPoints: 450,
  spanPolicy: 'canonical-recommended-span-v1',
  resolutionScalePolicy: 'recommended-span-divided-by-points-minus-one-v1',
} as const);
interface PinnedTemporalSchedule {
  readonly id: string;
  readonly sourcePlanProfileId: string;
  readonly sourceLookIndexOffset: number;
  readonly sourcePlanSpectrumOpportunities: number;
}
const PINNED_SIGNAL_LAB_PRODUCTION_SPECTRUM_RELEASE_GATE_SOURCE_PLAN = Object.freeze([
  Object.freeze({ profileId: 'cw', profileOrdinal: 0, sourceLookIndexOffset: 0, spectrumOpportunities: 32, automaticDetectedPowerCaptures: 0 } as const),
  Object.freeze({ profileId: 'am', profileOrdinal: 1, sourceLookIndexOffset: 32, spectrumOpportunities: 32, automaticDetectedPowerCaptures: 0 } as const),
  Object.freeze({ profileId: 'fm', profileOrdinal: 2, sourceLookIndexOffset: 64, spectrumOpportunities: 32, automaticDetectedPowerCaptures: 0 } as const),
  Object.freeze({ profileId: 'gsm-900-loaded-bcch', profileOrdinal: 3, sourceLookIndexOffset: 96, spectrumOpportunities: 32, automaticDetectedPowerCaptures: 0 } as const),
  Object.freeze({ profileId: 'lte-band3-fdd-20m', profileOrdinal: 4, sourceLookIndexOffset: 128, spectrumOpportunities: 32, automaticDetectedPowerCaptures: 0 } as const),
  Object.freeze({ profileId: 'lte-band38-tdd-10m', profileOrdinal: 5, sourceLookIndexOffset: 160, spectrumOpportunities: 32, automaticDetectedPowerCaptures: 0 } as const),
  Object.freeze({ profileId: 'nr-n3-fdd-20m', profileOrdinal: 6, sourceLookIndexOffset: 192, spectrumOpportunities: 32, automaticDetectedPowerCaptures: 0 } as const),
  Object.freeze({ profileId: 'nr-n78-tdd-100m', profileOrdinal: 7, sourceLookIndexOffset: 224, spectrumOpportunities: 32, automaticDetectedPowerCaptures: 0 } as const),
  Object.freeze({ profileId: 'wifi-hr-dsss-11m', profileOrdinal: 8, sourceLookIndexOffset: 256, spectrumOpportunities: 32, automaticDetectedPowerCaptures: 0 } as const),
  Object.freeze({ profileId: 'wifi-ofdm-20m', profileOrdinal: 9, sourceLookIndexOffset: 288, spectrumOpportunities: 32, automaticDetectedPowerCaptures: 0 } as const),
  Object.freeze({ profileId: 'bluetooth-classic-connected', profileOrdinal: 10, sourceLookIndexOffset: 320, spectrumOpportunities: 96, automaticDetectedPowerCaptures: 0 } as const),
  Object.freeze({ profileId: 'bluetooth-le-advertising', profileOrdinal: 11, sourceLookIndexOffset: 416, spectrumOpportunities: 96, automaticDetectedPowerCaptures: 0 } as const),
] as const);
const PINNED_SIGNAL_LAB_PRODUCTION_QUALIFIED_ENVELOPE_RELEASE_GATE_SOURCE_PLAN = Object.freeze([
  Object.freeze({ profileId: 'cw', profileOrdinal: 0, sourceLookIndexOffset: 0, spectrumOpportunities: 32, admittedDetectedPowerCaptures: 1 } as const),
  Object.freeze({ profileId: 'am', profileOrdinal: 1, sourceLookIndexOffset: 33, spectrumOpportunities: 32, admittedDetectedPowerCaptures: 1 } as const),
  Object.freeze({ profileId: 'fm', profileOrdinal: 2, sourceLookIndexOffset: 66, spectrumOpportunities: 32, admittedDetectedPowerCaptures: 1 } as const),
  Object.freeze({ profileId: 'gsm-900-loaded-bcch', profileOrdinal: 3, sourceLookIndexOffset: 99, spectrumOpportunities: 32, admittedDetectedPowerCaptures: 1 } as const),
  Object.freeze({ profileId: 'lte-band3-fdd-20m', profileOrdinal: 4, sourceLookIndexOffset: 132, spectrumOpportunities: 32, admittedDetectedPowerCaptures: 1 } as const),
  Object.freeze({ profileId: 'lte-band38-tdd-10m', profileOrdinal: 5, sourceLookIndexOffset: 165, spectrumOpportunities: 32, admittedDetectedPowerCaptures: 1 } as const),
  Object.freeze({ profileId: 'nr-n3-fdd-20m', profileOrdinal: 6, sourceLookIndexOffset: 198, spectrumOpportunities: 32, admittedDetectedPowerCaptures: 1 } as const),
  Object.freeze({ profileId: 'nr-n78-tdd-100m', profileOrdinal: 7, sourceLookIndexOffset: 231, spectrumOpportunities: 32, admittedDetectedPowerCaptures: 1 } as const),
  Object.freeze({ profileId: 'wifi-hr-dsss-11m', profileOrdinal: 8, sourceLookIndexOffset: 264, spectrumOpportunities: 32, admittedDetectedPowerCaptures: 1 } as const),
  Object.freeze({ profileId: 'wifi-ofdm-20m', profileOrdinal: 9, sourceLookIndexOffset: 297, spectrumOpportunities: 32, admittedDetectedPowerCaptures: 1 } as const),
  Object.freeze({ profileId: 'bluetooth-classic-connected', profileOrdinal: 10, sourceLookIndexOffset: 330, spectrumOpportunities: 96, admittedDetectedPowerCaptures: 1 } as const),
  Object.freeze({ profileId: 'bluetooth-le-advertising', profileOrdinal: 11, sourceLookIndexOffset: 427, spectrumOpportunities: 96, admittedDetectedPowerCaptures: 1 } as const),
] as const);
const PINNED_SIGNAL_LAB_PRODUCTION_SPECTRUM_TEMPORAL_SCHEDULES: readonly PinnedTemporalSchedule[] = Object.freeze(
  PINNED_SIGNAL_LAB_PRODUCTION_SPECTRUM_RELEASE_GATE_SOURCE_PLAN.map((sourcePlan) => branchSchedule(
    'consecutive-spectrum',
    sourcePlan.profileId,
    sourcePlan.sourceLookIndexOffset,
    sourcePlan.spectrumOpportunities,
  )),
);
const PINNED_SIGNAL_LAB_PRODUCTION_QUALIFIED_ENVELOPE_TEMPORAL_SCHEDULES: readonly PinnedTemporalSchedule[] = Object.freeze(
  PINNED_SIGNAL_LAB_PRODUCTION_QUALIFIED_ENVELOPE_RELEASE_GATE_SOURCE_PLAN.map((sourcePlan) => branchSchedule(
    'qualified-envelope',
    sourcePlan.profileId,
    sourcePlan.sourceLookIndexOffset,
    sourcePlan.spectrumOpportunities,
  )),
);
const PINNED_VALIDATION_SPECTRUM_TEMPORAL_SCHEDULE: PinnedTemporalSchedule = Object.freeze({
  id: 'held-out-validation-consecutive-spectrum-first-post-live-index-512-v3',
  sourcePlanProfileId: 'held-out-validation',
  sourceLookIndexOffset: 512,
  sourcePlanSpectrumOpportunities: FULL_BAND_2G4_OBSERVATION_OPPORTUNITIES,
} as const);
const PINNED_VALIDATION_QUALIFIED_ENVELOPE_TEMPORAL_SCHEDULE: PinnedTemporalSchedule = Object.freeze({
  id: 'held-out-validation-qualified-envelope-first-post-live-index-524-v3',
  sourcePlanProfileId: 'held-out-validation',
  sourceLookIndexOffset: 524,
  sourcePlanSpectrumOpportunities: FULL_BAND_2G4_OBSERVATION_OPPORTUNITIES,
} as const);
const PINNED_SIGNAL_LAB_PRODUCTION_ACQUISITION_REGIME = Object.freeze({
  id: 'signal-lab-recommended-span-grid-with-independent-production-branch-source-clocks-v5',
  geometry: PINNED_SIGNAL_LAB_PRODUCTION_GEOMETRY,
  branchPolicy: 'independent-no-auto-spectrum-and-qualified-rank-0-integrated-excess-envelope-sessions-v2',
  sourceClocks: Object.freeze({
    spectrum: Object.freeze({
      id: 'shared-monotonic-source-clock-v1',
      acquisitionIndexPolicy: 'one-look-index-per-physical-acquisition-v1',
      detectedPowerCapturePolicy: 'no-automatic-detected-power-capture-v1',
    } as const),
    qualifiedEnvelope: Object.freeze({
      id: 'shared-monotonic-source-clock-v1',
      acquisitionIndexPolicy: 'one-look-index-per-physical-acquisition-v1',
      detectedPowerCapturePolicy: PINNED_DETECTED_POWER_CAPTURE_POLICY_ID,
      captureTargetSelectionPolicy: PINNED_CAPTURE_TARGET_SELECTION_POLICY_ID,
      postCaptureSpectrumPolicy: 'continue-at-next-shared-look-index-v1',
    } as const),
  } as const),
  spectrumReleaseGateSourcePlan: PINNED_SIGNAL_LAB_PRODUCTION_SPECTRUM_RELEASE_GATE_SOURCE_PLAN,
  qualifiedEnvelopeReleaseGateSourcePlan:
    PINNED_SIGNAL_LAB_PRODUCTION_QUALIFIED_ENVELOPE_RELEASE_GATE_SOURCE_PLAN,
  temporalSchedulePairs: Object.freeze(
    PINNED_SIGNAL_LAB_PRODUCTION_SPECTRUM_TEMPORAL_SCHEDULES.map(
      (spectrumTemporalSchedule, index) => Object.freeze({
        id: `live-release-gate-independent-branches-${spectrumTemporalSchedule.sourcePlanProfileId}-v3`,
        sourcePlanProfileId: spectrumTemporalSchedule.sourcePlanProfileId,
        spectrumTemporalSchedule,
        qualifiedEnvelopeTemporalSchedule:
          PINNED_SIGNAL_LAB_PRODUCTION_QUALIFIED_ENVELOPE_TEMPORAL_SCHEDULES[index]!,
      }),
    ),
  ),
  componentFitIncluded: true,
  tailCalibrationIncluded: true,
} as const);
const pinnedSpectrumReleaseGateSourcePlanValid = PINNED_SIGNAL_LAB_PRODUCTION_SPECTRUM_RELEASE_GATE_SOURCE_PLAN.every(
  (sourcePlan, index, plans) => sourcePlan.profileOrdinal === index
    && sourcePlan.automaticDetectedPowerCaptures === 0
    && sourcePlan.sourceLookIndexOffset === (index === 0
      ? 0
      : plans[index - 1]!.sourceLookIndexOffset + plans[index - 1]!.spectrumOpportunities)
    && PINNED_SIGNAL_LAB_PRODUCTION_SPECTRUM_TEMPORAL_SCHEDULES[index]?.sourcePlanProfileId === sourcePlan.profileId
    && PINNED_SIGNAL_LAB_PRODUCTION_SPECTRUM_TEMPORAL_SCHEDULES[index]?.sourceLookIndexOffset === sourcePlan.sourceLookIndexOffset,
) && PINNED_SIGNAL_LAB_PRODUCTION_SPECTRUM_RELEASE_GATE_SOURCE_PLAN.at(-1)!.sourceLookIndexOffset
  + PINNED_SIGNAL_LAB_PRODUCTION_SPECTRUM_RELEASE_GATE_SOURCE_PLAN.at(-1)!.spectrumOpportunities === 512;
const pinnedQualifiedEnvelopeReleaseGateSourcePlanValid = PINNED_SIGNAL_LAB_PRODUCTION_QUALIFIED_ENVELOPE_RELEASE_GATE_SOURCE_PLAN.every(
  (sourcePlan, index, plans) => sourcePlan.profileOrdinal === index
    && sourcePlan.admittedDetectedPowerCaptures === 1
    && sourcePlan.sourceLookIndexOffset === (index === 0
      ? 0
      : plans[index - 1]!.sourceLookIndexOffset
        + plans[index - 1]!.spectrumOpportunities
        + plans[index - 1]!.admittedDetectedPowerCaptures)
    && PINNED_SIGNAL_LAB_PRODUCTION_QUALIFIED_ENVELOPE_TEMPORAL_SCHEDULES[index]?.sourcePlanProfileId === sourcePlan.profileId
    && PINNED_SIGNAL_LAB_PRODUCTION_QUALIFIED_ENVELOPE_TEMPORAL_SCHEDULES[index]?.sourceLookIndexOffset === sourcePlan.sourceLookIndexOffset,
) && PINNED_SIGNAL_LAB_PRODUCTION_QUALIFIED_ENVELOPE_RELEASE_GATE_SOURCE_PLAN.at(-1)!.sourceLookIndexOffset
  + PINNED_SIGNAL_LAB_PRODUCTION_QUALIFIED_ENVELOPE_RELEASE_GATE_SOURCE_PLAN.at(-1)!.spectrumOpportunities
  + PINNED_SIGNAL_LAB_PRODUCTION_QUALIFIED_ENVELOPE_RELEASE_GATE_SOURCE_PLAN.at(-1)!.admittedDetectedPowerCaptures === 524;
const pinnedReleaseGateSourcePlanValid = pinnedSpectrumReleaseGateSourcePlanValid
  && pinnedQualifiedEnvelopeReleaseGateSourcePlanValid;
const PINNED_DETECTED_POWER_SYNTHESIS_FILTER_POLICY = Object.freeze({
  id: 'explicit-generator-filter-width-by-acquisition-regime-v1',
  divisorAcquisitionRegimes: 'match-swept-spectrum-actual-rbw-nuisance-v1',
  signalLabProductionAcquisitionRegimes: 'fixed-generator-internal-width-v1',
  signalLabProductionSynthesisFilterWidthHz: 100_000,
  measurementActualRbwQualification: 'unavailable',
} as const);
const PINNED_PRODUCTION_ACQUISITION_REGIME_HIGH_SNR_SEED_COVERAGE_POLICY = Object.freeze({
  id: 'branch-conditional-production-regime-presence-v2',
  spectrumOnly: Object.freeze({
    minimumDistinctObservationDomainEligibleSeedsPerHighSnrCell: 1,
  }),
  qualifiedEnvelope: Object.freeze({
    minimumDistinctPhysicalCaptureSeedsPerHighSnrCell: 1,
    observationDomainEligibilityPolicy:
      'pooled-by-scenario-and-view-after-causal-capture-v1',
    outOfDomainCapturePolicy:
      'honest-abstention-excluded-from-envelope-likelihood-v1',
  }),
  globalCoveragePolicy: 'all-seeds-at-one-or-more-regimes-except-declared-sparse-asynchronous-scenarios-v1',
} as const);
type PinnedCalibrationAcquisitionRegime = Readonly<{
  id: string;
  rbwDivisor: number | null;
  spectrumTemporalSchedule: PinnedTemporalSchedule;
  qualifiedEnvelopeTemporalSchedule: PinnedTemporalSchedule;
}>;
const PINNED_BASELINE_SPECTRUM_TEMPORAL_SCHEDULE =
  PINNED_SIGNAL_LAB_PRODUCTION_SPECTRUM_TEMPORAL_SCHEDULES[0]!;
const PINNED_BASELINE_QUALIFIED_ENVELOPE_TEMPORAL_SCHEDULE =
  PINNED_SIGNAL_LAB_PRODUCTION_QUALIFIED_ENVELOPE_TEMPORAL_SCHEDULES[0]!;
const PINNED_TAIL_CALIBRATION_ACQUISITION_REGIMES: readonly PinnedCalibrationAcquisitionRegime[] = Object.freeze([
  ...PINNED_TAIL_CALIBRATION_RBW_DIVISORS.map((rbwDivisor) => Object.freeze({
    id: `occupied-bandwidth-rbw-divisor:${rbwDivisor}/independent-production-branch-baselines-v1`,
    rbwDivisor,
    spectrumTemporalSchedule: PINNED_BASELINE_SPECTRUM_TEMPORAL_SCHEDULE,
    qualifiedEnvelopeTemporalSchedule:
      PINNED_BASELINE_QUALIFIED_ENVELOPE_TEMPORAL_SCHEDULE,
  })),
  ...PINNED_SIGNAL_LAB_PRODUCTION_ACQUISITION_REGIME.temporalSchedulePairs.map(
    (temporalSchedulePair) => Object.freeze({
      id: `${PINNED_SIGNAL_LAB_PRODUCTION_GEOMETRY.id}/${temporalSchedulePair.id}`,
      rbwDivisor: null,
      spectrumTemporalSchedule: temporalSchedulePair.spectrumTemporalSchedule,
      qualifiedEnvelopeTemporalSchedule:
        temporalSchedulePair.qualifiedEnvelopeTemporalSchedule,
    }),
  ),
]);
const PINNED_TAIL_CALIBRATION_ACQUISITION_REGIME_IDS = PINNED_TAIL_CALIBRATION_ACQUISITION_REGIMES
  .map((regime) => regime.id);
const TAIL_CALIBRATION_NUMERICAL_TOLERANCE = 1e-12;
const PINNED_ENGINEERING_PRIOR = Object.freeze({
  'cw-like': 0.08,
  'am-dsb-full-carrier-like': 0.08,
  'fm-angle-modulated-like': 0.08,
  'gsm-like': 0.04,
  'lte-fdd-like': 0.06,
  'lte-tdd-like': 0.06,
  'nr-fdd-like': 0.06,
  'nr-tdd-like': 0.06,
  'wifi-hr-dsss-like': 0.08,
  'wifi-ofdm-like': 0.08,
  'bluetooth-like': 0.12,
  'unknown-signal': 0.20,
} satisfies Record<ObservableLeafClass, number>);
const PRIOR_SENSITIVITY_GATES = Object.freeze({
  minimumKnownCoverage: 0.85,
  minimumHierarchicalAccuracy: 0.90,
  maximumIncompatibleNonUnknownRisk: 0,
  maximumFalseAcceptedUnknownRisk: 0,
  maximumDecisionChangeRate: 0.20,
});
const SWEEP_POINTS = 450;
const SWEEP_TIME_SECONDS = 0.05;
const ZERO_SPAN_POINTS = 450;
const ZERO_SPAN_SAMPLE_PERIOD_SECONDS = 1 / 9_000;
const REPORT_DIRECTORY = resolve('.artifacts/classifier-validation');
const REPORT_PATH = resolve(REPORT_DIRECTORY, 'report.json');
const REPORT_TEMP_PATH = resolve(REPORT_DIRECTORY, 'report.json.tmp');
const FAILED_REPORT_PATH = resolve(REPORT_DIRECTORY, 'report.failed.json');
const FAILED_REPORT_TEMP_PATH = resolve(REPORT_DIRECTORY, 'report.failed.json.tmp');
const VALIDATION_ACCEPTANCE_POLICY_ID = 'synthetic-observable-classifier-full-corpus-release-gates-v1';
const PINNED_SIGNAL_LAB_COMMIT = 'e7d48afbce7165fa04fd551629891123f3b86d34';
const SIGNAL_LAB_REPOSITORY_ROOT = resolve('../Atom-SignalLab');
mkdirSync(REPORT_DIRECTORY, { recursive: true });
for (const path of [REPORT_PATH, REPORT_TEMP_PATH, FAILED_REPORT_PATH, FAILED_REPORT_TEMP_PATH]) {
  rmSync(path, { force: true });
}
let validationPublicationCommitted = false;
process.once('uncaughtException', publishUnexpectedValidationFailure);
process.once('unhandledRejection', publishUnexpectedValidationFailure);
const checkedOutSignalLabCommit = gitOutput(['rev-parse', 'HEAD']).toString('utf8').trim();
if (checkedOutSignalLabCommit !== PINNED_SIGNAL_LAB_COMMIT) {
  throw new Error(`SignalLab checked-out commit ${checkedOutSignalLabCommit} does not match pinned ${PINNED_SIGNAL_LAB_COMMIT}`);
}
assertSignalLabRepositoryIsClean();
if (CANONIZED_REPLAY_DETECTED_POWER_SYNTHESIS_FILTER_WIDTH_HZ
  !== PINNED_DETECTED_POWER_SYNTHESIS_FILTER_POLICY.signalLabProductionSynthesisFilterWidthHz) {
  throw new Error('Validator detected-power synthesis filter pin does not match SignalLab');
}
const diagnosticScenarioIds = (process.env.TINYSA_VALIDATION_SCENARIO_IDS ?? '')
  .split(',').map((value) => value.trim()).filter(Boolean);
const diagnosticScenarioIdSet = new Set(diagnosticScenarioIds);
const validationScenarios = diagnosticScenarioIdSet.size === 0
  ? canonicalClassificationScenarios
  : canonicalClassificationScenarios.filter((scenario) => diagnosticScenarioIdSet.has(scenario.id));
const invalidDiagnosticScenarioIds = diagnosticScenarioIds
  .filter((scenarioId) => !canonicalClassificationScenarios.some((scenario) => scenario.id === scenarioId));
if (invalidDiagnosticScenarioIds.length > 0) {
  throw new Error(`Unknown diagnostic validation scenario IDs: ${invalidDiagnosticScenarioIds.join(', ')}`);
}
const heldOutSourceSpanAudit = auditHeldOutSourceSpan(canonicalClassificationScenarios);
const PRODUCTION_DETECTION_CONFIG: SignalDetectionConfig = {
  threshold: { strategy: 'noise-relative', marginDb: 10 },
  minimumBandwidthHz: 0,
  minimumProminenceDb: 6,
  minimumConsecutiveSweeps: 2,
  releaseAfterMissedSweeps: 2,
};
const classifier = new BayesianWaveformClassifier();
const PINNED_CORPUS_SOURCE_ARTIFACT_PATHS = [
  'package-lock.json',
  'package.json',
  'src/canonical-timing.ts',
  'src/catalog.ts',
  'src/classification-corpus.ts',
  'src/contracts.ts',
  'src/source-provenance.ts',
  'src/waveforms.ts',
] as const;
const PINNED_CORPUS_TYPESCRIPT_IMPORT_CLOSURE = [
  'src/canonical-timing.ts',
  'src/catalog.ts',
  'src/classification-corpus.ts',
  'src/contracts.ts',
  'src/source-provenance.ts',
  'src/waveforms.ts',
] as const;
assertCanonicalCorpusSourceArtifactPaths(PINNED_CORPUS_SOURCE_ARTIFACT_PATHS);
assertCorpusSourceImportClosure(
  'src/classification-corpus.ts',
  PINNED_CORPUS_TYPESCRIPT_IMPORT_CLOSURE,
  PINNED_CORPUS_SOURCE_ARTIFACT_PATHS,
);
const checkedOutCorpusSourceManifest = {
  schemaVersion: 1 as const,
  hashAlgorithm: 'sha256' as const,
  artifacts: PINNED_CORPUS_SOURCE_ARTIFACT_PATHS.map(corpusSourceArtifact),
};
const checkedOutCorpusSha256 = createHash('sha256')
  .update(JSON.stringify(checkedOutCorpusSourceManifest))
  .digest('hex');
const checkedInModelAssetSha256 = createHash('sha256')
  .update(readFileSync(resolve('src/models/bayesian-observable.generated.ts')))
  .digest('hex');
const identity: DeviceIdentity = {
  model: 'SignalLab production-pipeline synthetic validation corpus', hardwareVersion: 'offline', firmwareVersion: CLASSIFICATION_CORPUS_VERSION,
  firmwareQualification: 'protocol-test',
  port: { id: 'offline', path: 'offline://classification-validation', usbMatch: 'protocol-test-double', transport: 'protocol-test-double', execution: 'protocol-test-double' },
  simulated: true, usbIdentityVerified: false, execution: 'protocol-test-double',
};

interface AdmissionAttempt {
  attemptId: string;
  scenario: string;
  corpusTruth: ObservableSignalClass;
  modelTruth: ObservableLeafClass;
  allowedModelTruths: readonly ObservableLeafClass[];
  snrDb: number;
  rbwDivisor: number;
  actualRbwHz: number;
  detectedPowerSynthesisFilterWidthHz: number;
  binWidthHz: number;
  seed: number;
  observationHorizon: number;
  everReady: boolean;
  admitted: boolean;
  everReadyRepresentativeCount: number;
  firstReadyRepresentativeCount: number;
  provenanceUnavailableWindowCount: number;
  detectedPowerCaptureCount: number;
  detectedPowerCaptureReceiptVerified: boolean;
  detectedPowerCaptureReceiptSchemaVersion?: 4;
  physicalCaptureId?: string;
  captureProjectionKind?: DetectedPowerCaptureProjectionKind;
  projectedAssociationMode?: NonNullable<DetectedSignal['associationMode']>;
  classificationEvidenceView?: ObservableEvidenceView;
  envelopeFeatureAvailable: boolean;
  detectedPowerEvidenceDisposition?: DetectedPowerEvidenceDisposition;
  envelopeEvidenceCensoringPolicyId?:
    typeof PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORING_POLICY_ID;
  detectedPowerAcquisitionQualification?:
    typeof PINNED_DETECTED_POWER_ACQUISITION_QUALIFICATION;
  detectedPowerSelectionCondition?:
    typeof PINNED_DETECTED_POWER_SELECTION_CONDITION;
  envelopeFeatureUnavailableCode?: ObservableEvidenceUnavailableError['code'];
  finalReadyRepresentativeCount: number;
  finalActiveRepresentativeCount: number;
  selectedTrackAdmissions: number;
  maximumActiveAdmissions: number;
  maximumLocalTrackAdmissions: number;
  firstReadyOpportunity?: number;
  everAssociationModes: readonly string[];
  finalAssociationModes: readonly string[];
  regularAssociationsObserved: number;
  agileAssociationsObserved: number;
  regularAssociationExpirations: number;
}

interface ValidationCase {
  attemptId: string;
  representativeKey: string;
  scenario: string;
  corpusTruth: ObservableSignalClass;
  modelTruth: ObservableLeafClass;
  allowedModelTruths: readonly ObservableLeafClass[];
  nominalBandwidthHz: number;
  snrDb: number;
  rbwDivisor: number;
  actualRbwHz: number;
  detectedPowerSynthesisFilterWidthHz: number;
  binWidthHz: number;
  seed: number;
  firstReadyOpportunity: number;
  componentFitEligible: boolean;
  result: string;
  confidence: number;
  unknownPosterior: number;
  truthPosterior: number;
  topLeaf: string;
  topLeafPosterior: number;
  acceptedHierarchy: boolean;
  posterior: Readonly<Record<string, number>>;
  centerHz: number;
  occupiedStartHz: number;
  occupiedStopHz: number;
  bandwidthHz: number;
  selectedTrackAdmissions: number;
  localTrackAdmissions: number;
  associationMode: NonNullable<DetectedSignal['associationMode']>;
  rawCaptureTargetAssociationMode: NonNullable<DetectedSignal['associationMode']>;
  rawCaptureTargetState: 'candidate' | 'active';
  captureProjectionKind: DetectedPowerCaptureProjectionKind;
  physicalCaptureId: string;
  detectedPowerCaptureReceiptSchemaVersion: 4;
  detectedPowerEvidenceDisposition: DetectedPowerEvidenceDisposition;
  envelopeEvidenceCensoringPolicyId?:
    typeof PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORING_POLICY_ID;
  associationId?: string;
  associationModelId?: string;
  associationMemberCount?: number;
  associationRegionBandwidthHz?: number;
  knownSupportRank: number;
  associationEvidenceQualification?: ObservableFeatureObservation['associationEvidenceQualification'];
  zeroSpanCaptureId?: string;
  detectedPowerAcquisitionQualification?:
    typeof PINNED_DETECTED_POWER_ACQUISITION_QUALIFICATION;
  detectedPowerSelectionCondition?:
    typeof PINNED_DETECTED_POWER_SELECTION_CONDITION;
  views: ObservableFeatureObservation['views'];
  limitations: readonly string[];
  features: Readonly<Record<string, number>>;
}

interface EvidenceViewCase {
  attemptId: string;
  representativeKey: string;
  view: 'spectrum-only' | 'envelope-untimed';
  scenario: string;
  corpusTruth: ObservableSignalClass;
  modelTruth: ObservableLeafClass;
  allowedModelTruths: readonly ObservableLeafClass[];
  componentFitEligible: boolean;
  nominalBandwidthHz: number;
  measuredBandwidthHz: number;
  result: string;
  topLeaf: string;
  topLeafPosterior: number;
  truthPosterior: number;
  posterior: Readonly<Record<string, number>>;
  acceptedHierarchy: boolean;
  supportRank: number;
  features: Readonly<Record<string, number>>;
}

interface RollingWindowCase {
  attemptId: string;
  representativeKey: string;
  readyOpportunity: number;
  scenario: string;
  corpusTruth: ObservableSignalClass;
  modelTruth: ObservableLeafClass;
  allowedModelTruths: readonly ObservableLeafClass[];
  snrDb: number;
  rbwDivisor: number;
  seed: number;
  result: string;
  acceptedHierarchy: boolean;
  truthClassDomainEligible: boolean;
  knownSupportRank: number;
  nominalBandwidthHz: number;
  occupiedStartHz: number;
  occupiedStopHz: number;
  centerHz: number;
  measuredBandwidthHz: number;
  binWidthHz: number;
  limitations: readonly ObservableFeatureObservation['limitations'][number][];
  views: ObservableFeatureObservation['views'];
  associationEvidenceQualification?: ObservableFeatureObservation['associationEvidenceQualification'];
  topLeaf: string;
  topLeafPosterior: number;
  posterior: Readonly<Record<string, number>>;
  features: Readonly<Record<string, number>>;
  associationMode: NonNullable<DetectedSignal['associationMode']>;
}

interface SpectrumOnlineAssociationSample {
  attemptId: string;
  scenario: string;
  snrDb: number;
  rbwDivisor: number;
  seed: number;
  readyOpportunity: number;
  representativeKey: string;
  associationMode: NonNullable<DetectedSignal['associationMode']>;
}

interface ExactEquivalenceDiscrepancy {
  pair: string;
  nuisanceCell: string;
  representativeIndex?: number;
  view?: EvidenceViewCase['view'] | 'spectrum-online';
  field: string;
  reference: unknown;
  null: unknown;
}

type TailCalibrationView = 'spectrum-only' | 'envelope-untimed' | 'envelope-timed';
type ProductionAcquisitionBranch = 'consecutive-spectrum' | 'qualified-envelope';

interface RecomputedTailCalibrationAudit {
  valid: boolean;
  scoreTolerance: number;
  recomputedAttemptCountsByScenarioByView: Readonly<Record<string, Readonly<Record<TailCalibrationView, number>>>>;
  attemptCountMismatches: readonly { scenarioId: string; view: TailCalibrationView; expected: number; observed: number }[];
  scoreComparisons: readonly {
    classId: ObservableLeafClass;
    view: TailCalibrationView;
    expectedCount: number;
    observedCount: number;
    maximumAbsoluteDifference: number;
    expectedSha256: string;
    observedSha256: string;
  }[];
  lateMinimumCount: number;
  allOnlineAttemptCount: number;
  runtimeBranchClockAudits: {
    consecutiveSpectrum: ReturnType<typeof summarizeCausalAcquisitionTraces>;
    qualifiedEnvelope: ReturnType<typeof summarizeCausalAcquisitionTraces>;
  };
  aggregationRegression: {
    firstOpportunity: number;
    minimumOpportunity: number;
    minimumSupport: number;
    passed: boolean;
  };
}

interface ExactEquivalencePairAudit {
  pair: string;
  referenceScenarioId: string;
  nullScenarioId: string;
  nuisanceCells: number;
  matchedAdmissionCells: number;
  matchedRepresentativePairs: number;
  matchedEvidenceViewPairs: number;
  matchedOnlineSpectrumPairs: number;
  discrepancyCount: number;
  discrepancies: readonly ExactEquivalenceDiscrepancy[];
}

interface FirstReadyRepresentative {
  detection: DetectedSignal;
  representativeKey: string;
  classificationAdmissions: number;
  localTrackAdmissions: number;
  firstReadyOpportunity: number;
  evidenceSweeps: readonly Sweep[];
  spectrumObservation: ObservableFeatureObservation;
  /** Present only in the qualified-envelope branch. */
  rawCaptureTarget?: DetectedSignal;
  /** Present only in the qualified-envelope branch. */
  captureProjectionKind?: DetectedPowerCaptureProjectionKind;
}

interface OnlineReadyRepresentative {
  detection: DetectedSignal;
  representativeKey: string;
  classificationAdmissions: number;
  localTrackAdmissions: number;
  readyOpportunity: number;
  evidenceSweeps: readonly Sweep[];
  spectrumObservation: ObservableFeatureObservation;
}

interface LiveEnvelopeCapture {
  representative: FirstReadyRepresentative;
  zeroSpan: ZeroSpanCapture;
  detectedPowerCaptureReceipt: DetectedPowerCaptureReceipt;
  sourceLookIndex: number;
  classifierObservation: ObservableFeatureObservation;
  detectedPowerEvidenceDisposition: DetectedPowerEvidenceDisposition;
  envelopeEvidenceCensoringPolicyId?:
    typeof PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORING_POLICY_ID;
  envelopeObservation?: ObservableFeatureObservation;
  unavailableCode?: ObservableEvidenceUnavailableError['code'];
}

interface CausalAcquisitionTrace {
  readonly branch: ProductionAcquisitionBranch;
  readonly scheduleId: string;
  readonly sourceLookIndexStart: number;
  readonly sourceLookIndexStop: number;
  readonly sourceLookIndices: readonly number[];
  readonly spectrumSourceLookIndices: readonly number[];
  readonly detectedPowerSourceLookIndices: readonly number[];
  readonly captureTriggerSpectrumLookIndex?: number;
  readonly captureTriggerOpportunity?: number;
  readonly uniqueSourceLookIndices: boolean;
  readonly strictlyIncreasingSourceLookIndices: boolean;
  readonly captureImmediatelyFollowsTrigger: boolean;
}

interface ProductionOnlineSelection {
  representatives: readonly FirstReadyRepresentative[];
  onlineReadyRepresentatives: readonly OnlineReadyRepresentative[];
  everReadyRepresentativeKeys: readonly string[];
  provenanceUnavailableWindowCount: number;
  finalReadyRepresentativeCount: number;
  finalActiveRepresentativeCount: number;
  maximumActiveAdmissions: number;
  maximumLocalTrackAdmissions: number;
  firstReadyOpportunity?: number;
  everAssociationModes: readonly string[];
  finalAssociationModes: readonly string[];
  regularAssociationIds: readonly string[];
  agileAssociationIds: readonly string[];
  regularAssociationExpirations: number;
  liveEnvelopeCapture?: LiveEnvelopeCapture;
  acquisitionTrace: CausalAcquisitionTrace;
}

// This class must be initialized before the validator's top-level acquisition
// matrix starts below.  Keeping it beside the trace contract prevents bundlers
// from lowering a later class declaration to an uninitialized `var` at the
// first acquireProductionAttempt call.
class IndependentCausalSourceClock {
  readonly #sourceLookIndices: number[] = [];
  readonly #spectrumSourceLookIndices: number[] = [];
  readonly #detectedPowerSourceLookIndices: number[] = [];
  #nextSourceLookIndex: number;
  #captureTriggerSpectrumLookIndex: number | undefined;
  #captureTriggerOpportunity: number | undefined;

  constructor(
    private readonly temporalSchedule: PinnedTemporalSchedule,
    private readonly branch: ProductionAcquisitionBranch,
  ) {
    this.#nextSourceLookIndex = temporalSchedule.sourceLookIndexOffset;
  }

  acquireSpectrum(): number {
    const sourceLookIndex = this.#consumeSourceLookIndex();
    this.#spectrumSourceLookIndices.push(sourceLookIndex);
    return sourceLookIndex;
  }

  acquireDetectedPower(triggerSpectrumLookIndex: number, triggerOpportunity: number): number {
    if (this.branch !== 'qualified-envelope') {
      throw new Error(`${this.temporalSchedule.id} forbids detected-power acquisition on the consecutive-spectrum branch`);
    }
    if (this.#detectedPowerSourceLookIndices.length !== 0) {
      throw new Error(`${this.temporalSchedule.id} attempted more than one live detected-power acquisition`);
    }
    if (this.#spectrumSourceLookIndices.at(-1) !== triggerSpectrumLookIndex) {
      throw new Error(`${this.temporalSchedule.id} detected-power capture was not triggered by the immediately preceding spectrum`);
    }
    this.#captureTriggerSpectrumLookIndex = triggerSpectrumLookIndex;
    this.#captureTriggerOpportunity = triggerOpportunity;
    const sourceLookIndex = this.#consumeSourceLookIndex();
    this.#detectedPowerSourceLookIndices.push(sourceLookIndex);
    return sourceLookIndex;
  }

  trace(): CausalAcquisitionTrace {
    const sourceLookIndices = [...this.#sourceLookIndices];
    const uniqueSourceLookIndices = new Set(sourceLookIndices).size === sourceLookIndices.length;
    const strictlyIncreasingSourceLookIndices = sourceLookIndices.every(
      (sourceLookIndex, index) => index === 0 || sourceLookIndex > sourceLookIndices[index - 1]!,
    );
    const captureSourceLookIndex = this.#detectedPowerSourceLookIndices[0];
    const captureImmediatelyFollowsTrigger = captureSourceLookIndex === undefined
      || (this.#captureTriggerSpectrumLookIndex !== undefined
        && captureSourceLookIndex === this.#captureTriggerSpectrumLookIndex + 1);
    return Object.freeze({
      branch: this.branch,
      scheduleId: this.temporalSchedule.id,
      sourceLookIndexStart: sourceLookIndices[0] ?? this.temporalSchedule.sourceLookIndexOffset,
      sourceLookIndexStop: sourceLookIndices.at(-1) ?? this.temporalSchedule.sourceLookIndexOffset - 1,
      sourceLookIndices: Object.freeze(sourceLookIndices),
      spectrumSourceLookIndices: Object.freeze([...this.#spectrumSourceLookIndices]),
      detectedPowerSourceLookIndices: Object.freeze([...this.#detectedPowerSourceLookIndices]),
      ...(this.#captureTriggerSpectrumLookIndex === undefined
        ? {}
        : { captureTriggerSpectrumLookIndex: this.#captureTriggerSpectrumLookIndex }),
      ...(this.#captureTriggerOpportunity === undefined
        ? {}
        : { captureTriggerOpportunity: this.#captureTriggerOpportunity }),
      uniqueSourceLookIndices,
      strictlyIncreasingSourceLookIndices,
      captureImmediatelyFollowsTrigger,
    });
  }

  #consumeSourceLookIndex(): number {
    const sourceLookIndex = this.#nextSourceLookIndex;
    this.#nextSourceLookIndex += 1;
    this.#sourceLookIndices.push(sourceLookIndex);
    return sourceLookIndex;
  }
}

const cases: ValidationCase[] = [];
const evidenceViewCases: EvidenceViewCase[] = [];
const rollingWindowCases: RollingWindowCase[] = [];
const spectrumOnlineAssociationSamples: SpectrumOnlineAssociationSample[] = [];
const admissionAttempts: AdmissionAttempt[] = [];
const validationSpectrumAcquisitionTraces: CausalAcquisitionTrace[] = [];
const validationQualifiedEnvelopeAcquisitionTraces: CausalAcquisitionTrace[] = [];
for (const scenario of validationScenarios) {
  for (const snrDb of SNR_DB) {
    for (const rbwDivisor of RBW_DIVISORS) {
      for (const seed of NUISANCE_SHIFT_SEEDS) {
        const nominalBinWidthHz = scenario.recommendedSpanHz / 449;
        const actualRbwHz = Math.max(nominalBinWidthHz * 0.8, scenario.occupiedBandwidthHz / rbwDivisor, 1_000);
        const detectedPowerSynthesisFilterWidthHz = actualRbwHz;
        const attemptId = validationAttemptId(scenario.id, snrDb, rbwDivisor, seed);
        const observationHorizon = observationOpportunityHorizon(scenario);
        const spectrumSelection = acquireProductionAttempt({
          scenario,
          temporalSchedule: PINNED_VALIDATION_SPECTRUM_TEMPORAL_SCHEDULE,
          observationHorizon,
          seed,
          snrDb,
          actualRbwHz,
          detectedPowerSynthesisFilterWidthHz,
          context: `${attemptId}:consecutive-spectrum`,
          branch: 'consecutive-spectrum',
        });
        const envelopeSelection = acquireProductionAttempt({
          scenario,
          temporalSchedule: PINNED_VALIDATION_QUALIFIED_ENVELOPE_TEMPORAL_SCHEDULE,
          observationHorizon,
          seed,
          snrDb,
          actualRbwHz,
          detectedPowerSynthesisFilterWidthHz,
          context: `${attemptId}:qualified-envelope`,
          branch: 'qualified-envelope',
        });
        validationSpectrumAcquisitionTraces.push(spectrumSelection.acquisitionTrace);
        validationQualifiedEnvelopeAcquisitionTraces.push(envelopeSelection.acquisitionTrace);
        const mappedTruth = modelTruth(scenario.truthClass);
        const allowedModelTruths = [...new Set(scenario.allowedObservableClasses.map(modelTruth))];
        const selectedAdmissions = envelopeSelection.representatives
          .map((item) => item.classificationAdmissions);
        admissionAttempts.push({
          attemptId,
          scenario: scenario.id,
          corpusTruth: scenario.truthClass,
          modelTruth: mappedTruth,
          allowedModelTruths,
          snrDb,
          rbwDivisor,
          actualRbwHz,
          detectedPowerSynthesisFilterWidthHz,
          binWidthHz: nominalBinWidthHz,
          seed,
          observationHorizon,
          everReady: envelopeSelection.everReadyRepresentativeKeys.length > 0,
          admitted: envelopeSelection.representatives.length > 0,
          everReadyRepresentativeCount: envelopeSelection.everReadyRepresentativeKeys.length,
          firstReadyRepresentativeCount: envelopeSelection.representatives.length,
          provenanceUnavailableWindowCount: envelopeSelection.provenanceUnavailableWindowCount,
          detectedPowerCaptureCount:
            envelopeSelection.acquisitionTrace.detectedPowerSourceLookIndices.length,
          detectedPowerCaptureReceiptVerified:
            envelopeSelection.liveEnvelopeCapture !== undefined,
          ...(envelopeSelection.liveEnvelopeCapture === undefined
            ? {}
            : {
                detectedPowerCaptureReceiptSchemaVersion:
                  envelopeSelection.liveEnvelopeCapture
                    .detectedPowerCaptureReceipt.schemaVersion,
                physicalCaptureId:
                  envelopeSelection.liveEnvelopeCapture.zeroSpan.id,
                captureProjectionKind:
                  envelopeSelection.liveEnvelopeCapture.representative
                    .captureProjectionKind,
                projectedAssociationMode:
                  envelopeSelection.liveEnvelopeCapture.representative
                    .detection.associationMode ?? 'frequency-local',
                classificationEvidenceView: observableModelView(
                  envelopeSelection.liveEnvelopeCapture.classifierObservation,
                ),
              }),
          envelopeFeatureAvailable:
            envelopeSelection.liveEnvelopeCapture?.envelopeObservation !== undefined,
          ...(envelopeSelection.liveEnvelopeCapture === undefined
            ? {}
            : {
                detectedPowerEvidenceDisposition:
                  envelopeSelection.liveEnvelopeCapture
                    .detectedPowerEvidenceDisposition,
              }),
          ...(envelopeSelection.liveEnvelopeCapture
            ?.envelopeEvidenceCensoringPolicyId === undefined
            ? {}
            : {
                envelopeEvidenceCensoringPolicyId:
                  envelopeSelection.liveEnvelopeCapture
                    .envelopeEvidenceCensoringPolicyId,
              }),
          ...(envelopeSelection.liveEnvelopeCapture?.envelopeObservation
            ?.detectedPowerAcquisitionQualification === undefined
            ? {}
            : {
                detectedPowerAcquisitionQualification:
                  envelopeSelection.liveEnvelopeCapture.envelopeObservation
                    .detectedPowerAcquisitionQualification,
              }),
          ...(envelopeSelection.liveEnvelopeCapture?.envelopeObservation
            ?.detectedPowerSelectionCondition === undefined
            ? {}
            : {
                detectedPowerSelectionCondition:
                  PINNED_DETECTED_POWER_SELECTION_CONDITION,
              }),
          ...(envelopeSelection.liveEnvelopeCapture?.unavailableCode === undefined
            ? {}
            : {
                envelopeFeatureUnavailableCode:
                  envelopeSelection.liveEnvelopeCapture.unavailableCode,
              }),
          finalReadyRepresentativeCount: envelopeSelection.finalReadyRepresentativeCount,
          finalActiveRepresentativeCount: envelopeSelection.finalActiveRepresentativeCount,
          selectedTrackAdmissions: selectedAdmissions.length ? Math.max(...selectedAdmissions) : 0,
          maximumActiveAdmissions: envelopeSelection.maximumActiveAdmissions,
          maximumLocalTrackAdmissions: envelopeSelection.maximumLocalTrackAdmissions,
          ...(envelopeSelection.firstReadyOpportunity === undefined
            ? {}
            : { firstReadyOpportunity: envelopeSelection.firstReadyOpportunity }),
          everAssociationModes: envelopeSelection.everAssociationModes,
          finalAssociationModes: envelopeSelection.finalAssociationModes,
          regularAssociationsObserved: envelopeSelection.regularAssociationIds.length,
          agileAssociationsObserved: envelopeSelection.agileAssociationIds.length,
          regularAssociationExpirations: envelopeSelection.regularAssociationExpirations,
        });
        for (const representative of spectrumSelection.onlineReadyRepresentatives) {
          spectrumOnlineAssociationSamples.push({
            attemptId,
            scenario: scenario.id,
            snrDb,
            rbwDivisor,
            seed,
            readyOpportunity: representative.readyOpportunity,
            representativeKey: representative.representativeKey,
            associationMode: representative.detection.associationMode ?? 'frequency-local',
          });
        }
        for (const representative of spectrumSelection.onlineReadyRepresentatives) {
          const detection = representative.detection;
          const featureObservation = representative.spectrumObservation;
          if (featureObservation.sweepIds.length !== CLASSIFICATION_ADMISSIONS) {
            throw new Error(`${scenario.id} rolling classifier extracted ${featureObservation.sweepIds.length} source sweeps, expected exactly ${CLASSIFICATION_ADMISSIONS}`);
          }
          const fitEligible = observableRepresentativeIsInClassDomain(
            mappedTruth,
            featureObservation,
          );
          const posterior = inferPosterior(featureObservation);
          const topLeaf = posterior[0]!;
          const result = await classifier.classify(detection, { sweeps: representative.evidenceSweeps });
          rollingWindowCases.push({
            attemptId,
            representativeKey: representative.representativeKey,
            readyOpportunity: representative.readyOpportunity,
            scenario: scenario.id,
            corpusTruth: scenario.truthClass,
            modelTruth: mappedTruth,
            allowedModelTruths,
            snrDb,
            rbwDivisor,
            seed,
            result: result.label,
            acceptedHierarchy: acceptsAnyTruth(
              result.label,
              allowedModelTruths,
              scenario.occupiedBandwidthHz,
              featureObservation.bandwidthHz,
            ),
            truthClassDomainEligible: fitEligible,
            knownSupportRank: knownModelSupportRank(featureObservation),
            nominalBandwidthHz: scenario.occupiedBandwidthHz,
            occupiedStartHz: featureObservation.occupiedStartHz,
            occupiedStopHz: featureObservation.occupiedStopHz,
            centerHz: featureObservation.centerHz,
            measuredBandwidthHz: featureObservation.bandwidthHz,
            binWidthHz: featureObservation.binWidthHz,
            limitations: featureObservation.limitations,
            views: featureObservation.views,
            ...(featureObservation.associationEvidenceQualification === undefined
              ? {}
              : {
                  associationEvidenceQualification:
                    featureObservation.associationEvidenceQualification,
                }),
            topLeaf: topLeaf.id,
            topLeafPosterior: topLeaf.probability,
            posterior: Object.fromEntries(posterior.map((item) => [item.id, item.probability])),
            features: featureObservation.values,
            associationMode: detection.associationMode ?? 'frequency-local',
          });
        }
        const spectrumEvidenceRepresentative = spectrumSelection.representatives[0];
        if (spectrumEvidenceRepresentative) {
          const representative = spectrumEvidenceRepresentative;
          const spectrumObservation = {
            ...representative.spectrumObservation,
            values: spectrumOnly(representative.spectrumObservation.values),
          };
          const spectrumPosterior = inferPosterior(spectrumObservation);
          const spectrumDecision = selectObservableDecision(spectrumPosterior, spectrumObservation);
          const spectrumResult = spectrumDecision.label === 'unknown'
            ? 'unknown'
            : `observable:${spectrumDecision.label}`;
          evidenceViewCases.push({
            attemptId,
            representativeKey: representative.representativeKey,
            view: 'spectrum-only',
            scenario: scenario.id,
            corpusTruth: scenario.truthClass,
            modelTruth: mappedTruth,
            allowedModelTruths,
            componentFitEligible: observableRepresentativeIsInClassDomain(mappedTruth, spectrumObservation),
            nominalBandwidthHz: scenario.occupiedBandwidthHz,
            measuredBandwidthHz: spectrumObservation.bandwidthHz,
            result: spectrumResult,
            topLeaf: spectrumPosterior[0]!.id,
            topLeafPosterior: spectrumPosterior[0]!.probability,
            truthPosterior: spectrumPosterior.find((item) => item.id === mappedTruth)?.probability ?? 0,
            posterior: Object.fromEntries(spectrumPosterior.map((item) => [item.id, item.probability])),
            acceptedHierarchy: acceptsAnyTruth(
              spectrumResult,
              allowedModelTruths,
              scenario.occupiedBandwidthHz,
              spectrumObservation.bandwidthHz,
            ),
            supportRank: knownModelSupportRank(spectrumObservation),
            features: spectrumObservation.values,
          });
        }
        const capture = envelopeSelection.liveEnvelopeCapture;
        if (capture) {
          const representative = capture.representative;
          const detection = representative.detection;
          const rawCaptureTarget = representative.rawCaptureTarget;
          const captureProjectionKind = representative.captureProjectionKind;
          if (!rawCaptureTarget || !captureProjectionKind) {
            throw new Error(
              `${scenario.id} qualified envelope lacks its physical raw target projection`,
            );
          }
          const evidenceSweeps = representative.evidenceSweeps;
          const expectedSweepIds = classificationSourceSweepIds(detection).slice(-CLASSIFICATION_ADMISSIONS);
          const classificationAdmissions = expectedSweepIds.length;
          if (classificationAdmissions !== CLASSIFICATION_ADMISSIONS) throw new Error(`${scenario.id} classifier admission window has ${classificationAdmissions} sweeps, expected exactly ${CLASSIFICATION_ADMISSIONS}`);
          const zeroSpan = capture.zeroSpan;
          const featureObservation = capture.classifierObservation;
          const agileEnvelopeCensored =
            capture.detectedPowerEvidenceDisposition
              === 'censored-frequency-agile-spectrum-only';
          if (agileEnvelopeCensored
            ? detection.associationMode !== 'frequency-agile-2g4-activity'
              || capture.envelopeObservation !== undefined
              || featureObservation.views.includes('detected-power-envelope')
              || featureObservation.zeroSpanCaptureId !== undefined
              || featureObservation.detectedPowerAcquisitionQualification
                !== undefined
              || featureObservation.detectedPowerSelectionCondition
                !== undefined
              || capture.envelopeEvidenceCensoringPolicyId
                !== PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORING_POLICY_ID
            : capture.envelopeObservation !== featureObservation
              || !featureObservation.views.includes('detected-power-envelope')) {
            throw new Error(
              `${scenario.id} physical capture does not obey the frequency-agile envelope-censoring boundary`,
            );
          }
          const componentFitEligible = observableRepresentativeIsInClassDomain(
            mappedTruth,
            featureObservation,
          );
          if (featureObservation.sweepIds.length !== CLASSIFICATION_ADMISSIONS) {
            throw new Error(`${scenario.id} extracted ${featureObservation.sweepIds.length} source sweeps, expected exactly ${CLASSIFICATION_ADMISSIONS}`);
          }
          if ([...featureObservation.sweepIds].sort().some((sweepId, index) => sweepId !== [...expectedSweepIds].sort()[index])) {
            throw new Error(`${scenario.id} classifier did not preserve its latest ${CLASSIFICATION_ADMISSIONS} positive source sweeps`);
          }
          const posterior = inferPosterior(featureObservation);
          const result = await classifyValidatorReceiptQualifiedObservation({
            detection,
            evidenceSweeps,
            spectrumObservation: representative.spectrumObservation,
            zeroSpan,
            detectedPowerCaptureReceipt: capture.detectedPowerCaptureReceipt,
          }, featureObservation, classifier);
          const topLeaf = posterior[0]!;
          const posteriorRecord = Object.fromEntries(posterior.map((item) => [item.id, item.probability]));
          cases.push({
            attemptId,
            representativeKey: representative.representativeKey,
            scenario: scenario.id,
            corpusTruth: scenario.truthClass,
            modelTruth: mappedTruth,
            allowedModelTruths,
            nominalBandwidthHz: scenario.occupiedBandwidthHz,
            snrDb,
            rbwDivisor,
            actualRbwHz,
            detectedPowerSynthesisFilterWidthHz,
            binWidthHz: nominalBinWidthHz,
            seed,
            firstReadyOpportunity: representative.firstReadyOpportunity,
            componentFitEligible,
            result: result.label,
            confidence: result.confidence,
            unknownPosterior: posterior.find((item) => item.id === 'unknown-signal')?.probability ?? 0,
            truthPosterior: posterior.find((item) => item.id === mappedTruth)?.probability ?? 0,
            topLeaf: topLeaf.id,
            topLeafPosterior: topLeaf.probability,
            acceptedHierarchy: acceptsAnyTruth(result.label, allowedModelTruths, scenario.occupiedBandwidthHz, featureObservation.bandwidthHz),
            posterior: posteriorRecord,
            centerHz: featureObservation.centerHz,
            occupiedStartHz: featureObservation.occupiedStartHz,
            occupiedStopHz: featureObservation.occupiedStopHz,
            bandwidthHz: featureObservation.bandwidthHz,
            selectedTrackAdmissions: representative.classificationAdmissions,
            localTrackAdmissions: representative.localTrackAdmissions,
            associationMode: detection.associationMode ?? 'frequency-local',
            rawCaptureTargetAssociationMode:
              rawCaptureTarget.associationMode ?? 'frequency-local',
            rawCaptureTargetState: rawCaptureTarget.state as 'candidate' | 'active',
            captureProjectionKind,
            physicalCaptureId: zeroSpan.id,
            detectedPowerCaptureReceiptSchemaVersion:
              capture.detectedPowerCaptureReceipt.schemaVersion,
            detectedPowerEvidenceDisposition:
              capture.detectedPowerEvidenceDisposition,
            ...(capture.envelopeEvidenceCensoringPolicyId === undefined
              ? {}
              : {
                  envelopeEvidenceCensoringPolicyId:
                    capture.envelopeEvidenceCensoringPolicyId,
                }),
            ...(detection.associationId === undefined ? {} : { associationId: detection.associationId }),
            ...(detection.associationModelId === undefined ? {} : { associationModelId: detection.associationModelId }),
            ...(detection.associationMemberTrackIds === undefined ? {} : { associationMemberCount: detection.associationMemberTrackIds.length }),
            ...(detection.associationRegionStartHz === undefined || detection.associationRegionStopHz === undefined
              ? {}
              : { associationRegionBandwidthHz: detection.associationRegionStopHz - detection.associationRegionStartHz }),
            knownSupportRank: knownModelSupportRank(featureObservation),
            ...(featureObservation.associationEvidenceQualification === undefined
              ? {}
              : { associationEvidenceQualification: featureObservation.associationEvidenceQualification }),
            ...(featureObservation.zeroSpanCaptureId === undefined
              ? {}
              : { zeroSpanCaptureId: featureObservation.zeroSpanCaptureId }),
            ...(featureObservation.detectedPowerAcquisitionQualification === undefined
              ? {}
              : {
                  detectedPowerAcquisitionQualification:
                    featureObservation.detectedPowerAcquisitionQualification,
                }),
            ...(featureObservation.detectedPowerSelectionCondition === undefined
              ? {}
              : {
                  detectedPowerSelectionCondition:
                    PINNED_DETECTED_POWER_SELECTION_CONDITION,
                }),
            views: featureObservation.views,
            limitations: result.evidence.limitations ?? [],
            features: featureObservation.values,
          });
          if (!agileEnvelopeCensored) {
            const envelopeUntimedObservation = {
              ...featureObservation,
              values: envelopeUntimed(featureObservation.values),
            };
            const envelopeUntimedPosterior = inferPosterior(envelopeUntimedObservation);
            const envelopeUntimedDecision = selectObservableDecision(envelopeUntimedPosterior, envelopeUntimedObservation);
            const envelopeUntimedResult = envelopeUntimedDecision.label === 'unknown'
              ? 'unknown'
              : `observable:${envelopeUntimedDecision.label}`;
            evidenceViewCases.push({
              attemptId,
              representativeKey: representative.representativeKey,
              view: 'envelope-untimed',
              scenario: scenario.id,
              corpusTruth: scenario.truthClass,
              modelTruth: mappedTruth,
              allowedModelTruths,
              componentFitEligible: observableRepresentativeIsInClassDomain(mappedTruth, envelopeUntimedObservation),
              nominalBandwidthHz: scenario.occupiedBandwidthHz,
              measuredBandwidthHz: envelopeUntimedObservation.bandwidthHz,
              result: envelopeUntimedResult,
              topLeaf: envelopeUntimedPosterior[0]!.id,
              topLeafPosterior: envelopeUntimedPosterior[0]!.probability,
              truthPosterior: envelopeUntimedPosterior.find((item) => item.id === mappedTruth)?.probability ?? 0,
              posterior: Object.fromEntries(envelopeUntimedPosterior.map((item) => [item.id, item.probability])),
              acceptedHierarchy: acceptsAnyTruth(
                envelopeUntimedResult,
                allowedModelTruths,
                scenario.occupiedBandwidthHz,
                envelopeUntimedObservation.bandwidthHz,
              ),
              supportRank: knownModelSupportRank(envelopeUntimedObservation),
              features: envelopeUntimedObservation.values,
            });
          }
        }
      }
    }
  }
}

const exactEquivalencePairAudit = auditExactEquivalencePairs(
  admissionAttempts,
  cases,
  evidenceViewCases,
  rollingWindowCases,
);
const exactEquivalenceDiscrepancies = exactEquivalencePairAudit.flatMap((audit) => audit.discrepancies);
const exactEquivalenceDiscrepancyCount = exactEquivalencePairAudit.reduce((sum, audit) => sum + audit.discrepancyCount, 0);
const validationSpectrumClockAudit = summarizeCausalAcquisitionTraces(
  validationSpectrumAcquisitionTraces,
  'consecutive-spectrum',
);
const validationQualifiedEnvelopeClockAudit = summarizeCausalAcquisitionTraces(
  validationQualifiedEnvelopeAcquisitionTraces,
  'qualified-envelope',
);
const expectedAttempts = validationScenarios.length * SNR_DB.length * RBW_DIVISORS.length * NUISANCE_SHIFT_SEEDS.length;
const modelFittingSeeds = [...BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.seeds];
const modelAttemptSamplingWorkerRuntimeSha256 =
  BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.attemptSamplingWorkerRuntimeSha256;
const modelTrainingRuntimeIdentity =
  BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.trainingRuntimeIdentity;
const trainingRuntimeIdentityPinsValid = JSON.stringify(modelTrainingRuntimeIdentity)
  === JSON.stringify(PINNED_TRAINING_RUNTIME_IDENTITY);
const modelCalibrationSeeds = [...(BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.tailCalibrationSeeds ?? [])];
const modelFittingRbwDivisors = [...BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.rbwDivisors];
const modelCalibrationRbwDivisors = [...(BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.tailCalibrationRbwDivisors ?? [])];
const modelFittingAcquisitionRegimeIds = [...(BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.fittingAcquisitionRegimeIds ?? [])];
const modelCalibrationAcquisitionRegimeIds = [...(BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.tailCalibrationAcquisitionRegimeIds ?? [])];
const fittingCalibrationSeedOverlap = numericIntersection(modelFittingSeeds, modelCalibrationSeeds);
const validationFittingSeedOverlap = numericIntersection(NUISANCE_SHIFT_SEEDS, modelFittingSeeds);
const validationCalibrationSeedOverlap = numericIntersection(NUISANCE_SHIFT_SEEDS, modelCalibrationSeeds);
const validationFittingRbwOverlap = numericIntersection(RBW_DIVISORS, modelFittingRbwDivisors);
const validationCalibrationRbwOverlap = numericIntersection(RBW_DIVISORS, modelCalibrationRbwDivisors);
const modelTailCalibrationAttemptCountsByView =
  BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.tailCalibrationAttemptCountsByScenarioByView ?? {};
const modelComponentScenarioIdsByView = Object.fromEntries(OBSERVABLE_EVIDENCE_VIEWS.map((view) => [
  view,
  [...new Set(BAYESIAN_OBSERVABLE_MODEL.classModels
    .flatMap((model) => observableModelComponents(model, view)
      .map(componentSourceScenarioId)))].sort(),
])) as Record<ObservableEvidenceView, string[]>;
const componentScenarioPopulationMismatches = OBSERVABLE_EVIDENCE_VIEWS
  .filter((view) => {
    const expected = view === 'spectrum-only'
      ? modelComponentScenarioIdsByView['spectrum-only']
      : modelComponentScenarioIdsByView['spectrum-only'].filter((scenarioId) =>
          !(PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORED_SCENARIO_IDS as readonly string[])
            .includes(scenarioId));
    return JSON.stringify(modelComponentScenarioIdsByView[view])
      !== JSON.stringify(expected);
  });
const expectedTailCalibrationScenarioIds = BAYESIAN_OBSERVABLE_MODEL.classModels
  .filter((model) => model.id !== 'unknown-signal')
  .flatMap((model) => observableModelComponents(model, 'spectrum-only')
    .map(componentSourceScenarioId));
const uniqueExpectedTailCalibrationScenarioIds = [...new Set(expectedTailCalibrationScenarioIds)].sort();
const modelTailCalibrationScenarioIds = Object.keys(modelTailCalibrationAttemptCountsByView).sort();
const missingTailCalibrationScenarioIds = setDifference(uniqueExpectedTailCalibrationScenarioIds, modelTailCalibrationScenarioIds);
const unexpectedTailCalibrationScenarioIds = setDifference(modelTailCalibrationScenarioIds, uniqueExpectedTailCalibrationScenarioIds);
const invalidTailCalibrationAttemptCounts = uniqueExpectedTailCalibrationScenarioIds.flatMap((scenarioId) =>
  (['spectrum-only', 'envelope-untimed', 'envelope-timed'] as const).flatMap((view) => {
    const count = modelTailCalibrationAttemptCountsByView[scenarioId]?.[view];
    const censoredEnvelope = view !== 'spectrum-only'
      && (PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORED_SCENARIO_IDS as readonly string[])
        .includes(scenarioId);
    return count !== undefined
      && Number.isInteger(count)
      && (censoredEnvelope ? count === 0 : count >= 40)
      ? []
      : [{ scenarioId, view, count: count ?? null }];
  }));
const tailCalibrationViewCountMismatches = BAYESIAN_OBSERVABLE_MODEL.classModels
  .filter((model) => model.id !== 'unknown-signal')
  .flatMap((model) =>
    (['spectrum-only', 'envelope-untimed', 'envelope-timed'] as const).flatMap((view) => {
      const sourceScenarioIds = [...new Set(observableModelComponents(model, view)
        .map(componentSourceScenarioId))];
      const expected = sourceScenarioIds.reduce((sum, sourceScenarioId) =>
        sum + (modelTailCalibrationAttemptCountsByView[sourceScenarioId]?.[view] ?? 0), 0);
      const observed = model.tailCalibrationScoresByView?.[view]?.length ?? 0;
      return observed === expected ? [] : [{ classId: model.id, view, expected, observed }];
    }));
const tailCalibrationMatrixPinsValid = JSON.stringify(modelCalibrationSeeds) === JSON.stringify(PINNED_TAIL_CALIBRATION_SEEDS)
  && JSON.stringify(modelCalibrationRbwDivisors) === JSON.stringify(PINNED_TAIL_CALIBRATION_RBW_DIVISORS)
  && JSON.stringify(modelCalibrationAcquisitionRegimeIds) === JSON.stringify(PINNED_TAIL_CALIBRATION_ACQUISITION_REGIME_IDS)
  && JSON.stringify(BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.snrDb) === JSON.stringify(PINNED_TAIL_CALIBRATION_SNR_DB);
const productionAcquisitionRegimePinsValid =
  pinnedReleaseGateSourcePlanValid
  && JSON.stringify(BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.signalLabProductionAcquisitionRegime)
    === JSON.stringify(PINNED_SIGNAL_LAB_PRODUCTION_ACQUISITION_REGIME)
  && BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.acquisitionBranchPolicy
    === PINNED_SIGNAL_LAB_PRODUCTION_ACQUISITION_REGIME.branchPolicy
  && BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.selectionPolicy === SELECTION_POLICY
  && BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.likelihoodPopulationPolicy
    === LIKELIHOOD_POPULATION_POLICY
  && JSON.stringify(BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.likelihoodComponentDecompositionPolicy)
    === JSON.stringify(PINNED_LIKELIHOOD_COMPONENT_DECOMPOSITION_POLICY)
  && JSON.stringify(OBSERVABLE_EVIDENCE_CENSORING_POLICY)
    === JSON.stringify(PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORING_POLICY)
  && JSON.stringify(BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.frequencyAgileFixedTuneEnvelopeCensoringPolicy)
    === JSON.stringify(PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORING_POLICY)
  && BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.causalSamplingAudit?.schemaVersion === 3
  && BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.causalSamplingAudit
    ?.attributedSourceClockTraceAudit.serialization
    === 'canonical-attempt-id-branch-attributed-trace-and-capture-disposition-digest-v3'
  && BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.representativeWeightingPolicy === REPRESENTATIVE_WEIGHTING_POLICY
  && BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.detectedPowerAcquisitionQualification
    === PINNED_DETECTED_POWER_ACQUISITION_QUALIFICATION
  && BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.detectedPowerSelectionCondition
    === PINNED_DETECTED_POWER_SELECTION_CONDITION
  && JSON.stringify(BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.productionAcquisitionRegimeHighSnrSeedCoveragePolicy)
    === JSON.stringify(PINNED_PRODUCTION_ACQUISITION_REGIME_HIGH_SNR_SEED_COVERAGE_POLICY)
  && JSON.stringify(BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.detectedPowerSynthesisFilterPolicy)
    === JSON.stringify(PINNED_DETECTED_POWER_SYNTHESIS_FILTER_POLICY)
  && JSON.stringify(modelFittingAcquisitionRegimeIds) === JSON.stringify(PINNED_TAIL_CALIBRATION_ACQUISITION_REGIME_IDS)
  && JSON.stringify(modelCalibrationAcquisitionRegimeIds) === JSON.stringify(PINNED_TAIL_CALIBRATION_ACQUISITION_REGIME_IDS);
const validationTemporalScheduleIdOverlap = [
  ...PINNED_SIGNAL_LAB_PRODUCTION_SPECTRUM_TEMPORAL_SCHEDULES,
  ...PINNED_SIGNAL_LAB_PRODUCTION_QUALIFIED_ENVELOPE_TEMPORAL_SCHEDULES,
]
  .map((schedule) => schedule.id)
  .filter((id) => id === PINNED_VALIDATION_SPECTRUM_TEMPORAL_SCHEDULE.id
    || id === PINNED_VALIDATION_QUALIFIED_ENVELOPE_TEMPORAL_SCHEDULE.id);
const pinnedFitSpectrumSourceLookIndices = [...new Set(
  PINNED_SIGNAL_LAB_PRODUCTION_SPECTRUM_TEMPORAL_SCHEDULES.flatMap(
    (schedule) => possibleBranchSourceLookIndices(
      schedule,
      FULL_BAND_2G4_OBSERVATION_OPPORTUNITIES,
      'consecutive-spectrum',
    ),
  ),
)];
const pinnedFitQualifiedEnvelopeSourceLookIndices = [...new Set(
  PINNED_SIGNAL_LAB_PRODUCTION_QUALIFIED_ENVELOPE_TEMPORAL_SCHEDULES.flatMap(
    (schedule) => possibleBranchSourceLookIndices(
      schedule,
      FULL_BAND_2G4_OBSERVATION_OPPORTUNITIES,
      'qualified-envelope',
    ),
  ),
)];
const pinnedValidationSpectrumSourceLookIndices = possibleBranchSourceLookIndices(
    PINNED_VALIDATION_SPECTRUM_TEMPORAL_SCHEDULE,
    FULL_BAND_2G4_OBSERVATION_OPPORTUNITIES,
    'consecutive-spectrum',
  );
const pinnedValidationQualifiedEnvelopeSourceLookIndices = possibleBranchSourceLookIndices(
    PINNED_VALIDATION_QUALIFIED_ENVELOPE_TEMPORAL_SCHEDULE,
    FULL_BAND_2G4_OBSERVATION_OPPORTUNITIES,
    'qualified-envelope',
  );
const validationFitSpectrumSourceLookIndexOverlap = numericIntersection(
  pinnedValidationSpectrumSourceLookIndices,
  pinnedFitSpectrumSourceLookIndices,
);
const validationFitQualifiedEnvelopeSourceLookIndexOverlap = numericIntersection(
  pinnedValidationQualifiedEnvelopeSourceLookIndices,
  pinnedFitQualifiedEnvelopeSourceLookIndices,
);
const validationFitTemporalSourceLookIndexOverlap = [
  ...validationFitSpectrumSourceLookIndexOverlap,
  ...validationFitQualifiedEnvelopeSourceLookIndexOverlap,
];
const validationTemporalPartitionDisjoint = validationTemporalScheduleIdOverlap.length === 0
  && validationFitTemporalSourceLookIndexOverlap.length === 0;
const tailCalibrationPolicyValid = BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.tailCalibrationScoreUnit
    === PINNED_TAIL_CALIBRATION_SCORE_UNIT
  && BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.tailCalibrationRepresentativeSelectionPolicy
    === PINNED_TAIL_CALIBRATION_SELECTION_POLICY
  && BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.tailCalibrationRepresentativeAggregationPolicy
    === PINNED_TAIL_CALIBRATION_AGGREGATION_POLICY
  && BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.tailCalibrationRuntimeInterpretationPolicy
    === PINNED_TAIL_CALIBRATION_RUNTIME_INTERPRETATION_POLICY
  && BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.tailCalibrationStatisticalInterpretation
    === PINNED_TAIL_CALIBRATION_STATISTICAL_INTERPRETATION
  && missingTailCalibrationScenarioIds.length === 0
  && unexpectedTailCalibrationScenarioIds.length === 0
  && invalidTailCalibrationAttemptCounts.length === 0
  && tailCalibrationViewCountMismatches.length === 0
  && tailCalibrationMatrixPinsValid
  && productionAcquisitionRegimePinsValid;
const samplingPartitionsDisjoint = modelCalibrationSeeds.length > 0
  && modelCalibrationRbwDivisors.length > 0
  && productionAcquisitionRegimePinsValid
  && validationTemporalPartitionDisjoint
  && fittingCalibrationSeedOverlap.length === 0
  && validationFittingSeedOverlap.length === 0
  && validationCalibrationSeedOverlap.length === 0
  && validationFittingRbwOverlap.length === 0
  && validationCalibrationRbwOverlap.length === 0;
const admissionMisses = admissionAttempts.filter((item) => !item.admitted);
const known = cases.filter((item) => item.modelTruth !== 'unknown-signal');
const unknown = cases.filter((item) => item.modelTruth === 'unknown-signal');
const modelScenarioExcludedIdList = [...(BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.scenarioExcludedFromComponentFitIds ?? [])];
const modelExactEquivalenceIdList = [...(BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.exactObservableEquivalenceNullScenarioIds ?? [])];
const modelKnownAcquisitionValidationIdList = [...(BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.knownAcquisitionValidationOnlyScenarioIds ?? [])];
const scenarioExcludedIdList = [...PINNED_SCENARIO_EXCLUDED_FROM_COMPONENT_FIT_IDS];
const scenarioExcludedIds = new Set<string>(scenarioExcludedIdList);
const exactEquivalenceIdList = [...PINNED_EXACT_OBSERVABLE_EQUIVALENCE_NULL_SCENARIO_IDS];
const exactEquivalenceIds = new Set<string>(exactEquivalenceIdList);
const knownAcquisitionValidationIdList = [...PINNED_KNOWN_ACQUISITION_VALIDATION_ONLY_SCENARIO_IDS];
const knownAcquisitionValidationIds = new Set<string>(knownAcquisitionValidationIdList);
const duplicateExcludedScenarioIds = duplicateStrings(modelScenarioExcludedIdList);
const duplicateExactEquivalenceScenarioIds = duplicateStrings(modelExactEquivalenceIdList);
const duplicateKnownAcquisitionValidationIds = duplicateStrings(modelKnownAcquisitionValidationIdList);
const modelExcludedMissingPinnedIds = setDifference(scenarioExcludedIdList, modelScenarioExcludedIdList);
const modelExcludedUnexpectedIds = setDifference(modelScenarioExcludedIdList, scenarioExcludedIdList);
const modelExactEquivalenceMissingPinnedIds = setDifference(exactEquivalenceIdList, modelExactEquivalenceIdList);
const modelExactEquivalenceUnexpectedIds = setDifference(modelExactEquivalenceIdList, exactEquivalenceIdList);
const modelKnownAcquisitionMissingPinnedIds = setDifference(knownAcquisitionValidationIdList, modelKnownAcquisitionValidationIdList);
const modelKnownAcquisitionUnexpectedIds = setDifference(modelKnownAcquisitionValidationIdList, knownAcquisitionValidationIdList);
const corpusScenarioById = new Map(canonicalClassificationScenarios.map((scenario) => [scenario.id, scenario]));
const invalidPinnedStrictUnknownHoldoutIds = PINNED_STRICT_UNKNOWN_HOLDOUT_SCENARIO_IDS.filter((scenarioId) => {
  const scenario = corpusScenarioById.get(scenarioId);
  return scenario === undefined
    || modelTruth(scenario.truthClass) !== 'unknown-signal'
    || [...new Set(scenario.allowedObservableClasses.map(modelTruth))].join('|') !== 'unknown-signal';
});
const invalidPinnedAmbiguityStressIds = PINNED_OBSERVABLE_AMBIGUITY_STRESS_SCENARIO_IDS.filter((scenarioId) => {
  const scenario = corpusScenarioById.get(scenarioId);
  const allowed = scenario ? [...new Set(scenario.allowedObservableClasses.map(modelTruth))] : [];
  return scenario === undefined
    || modelTruth(scenario.truthClass) !== 'unknown-signal'
    || !allowed.includes('unknown-signal')
    || !allowed.some((truth) => truth !== 'unknown-signal');
});
const invalidPinnedFittedUnknownIds = PINNED_FITTED_UNKNOWN_SCENARIO_IDS.filter((scenarioId) => {
  const scenario = corpusScenarioById.get(scenarioId);
  return scenario === undefined
    || modelTruth(scenario.truthClass) !== 'unknown-signal'
    || [...new Set(scenario.allowedObservableClasses.map(modelTruth))].join('|') !== 'unknown-signal';
});
const invalidPinnedExactEquivalencePairs = PINNED_EXACT_OBSERVABLE_EQUIVALENCE_PAIRS.filter((pair) => {
  const nullScenario = corpusScenarioById.get(pair.nullScenarioId);
  const referenceScenario = corpusScenarioById.get(pair.referenceScenarioId);
  const allowedNullTruths = nullScenario ? [...new Set(nullScenario.allowedObservableClasses.map(modelTruth))] : [];
  return nullScenario === undefined
    || referenceScenario === undefined
    || modelTruth(nullScenario.truthClass) !== 'unknown-signal'
    || modelTruth(referenceScenario.truthClass) === 'unknown-signal'
    || !allowedNullTruths.includes('unknown-signal')
    || !allowedNullTruths.includes(modelTruth(referenceScenario.truthClass));
}).map((pair) => `${pair.referenceScenarioId}<=>${pair.nullScenarioId}`);
const excludedScenarioSplit = [...scenarioExcludedIds].sort().map((scenarioId) => ({
  scenarioId,
  existsInCorpus: corpusScenarioById.has(scenarioId),
  corpusTruth: corpusScenarioById.get(scenarioId)?.truthClass,
  modelTruth: corpusScenarioById.has(scenarioId) ? modelTruth(corpusScenarioById.get(scenarioId)!.truthClass) : undefined,
  category: knownAcquisitionValidationIds.has(scenarioId)
    ? 'known-acquisition-validation-only'
    : exactEquivalenceIds.has(scenarioId)
    ? 'exact-observable-equivalence-null'
    : (PINNED_OBSERVABLE_AMBIGUITY_STRESS_SCENARIO_IDS as readonly string[]).includes(scenarioId)
      ? 'observable-ambiguity-stress'
      : 'strict-unknown-stress',
}));
const invalidExcludedScenarioIds = excludedScenarioSplit.filter((item) => !item.existsInCorpus).map((item) => item.scenarioId);
const nonUnknownExcludedScenarioIds = excludedScenarioSplit
  .filter((item) => item.existsInCorpus
    && item.modelTruth !== 'unknown-signal'
    && !knownAcquisitionValidationIds.has(item.scenarioId))
  .map((item) => item.scenarioId);
const excludedUnknownScenarioIds = excludedScenarioSplit.filter((item) => item.modelTruth === 'unknown-signal').map((item) => item.scenarioId);
const knownAcquisitionValidationSplit = [...knownAcquisitionValidationIds].sort().map((scenarioId) => {
  const scenario = corpusScenarioById.get(scenarioId);
  return {
    scenarioId,
    existsInCorpus: scenario !== undefined,
    modelTruth: scenario ? modelTruth(scenario.truthClass) : undefined,
    excludedFromComponentFit: scenarioExcludedIds.has(scenarioId),
  };
});
const invalidKnownAcquisitionValidationIds = knownAcquisitionValidationSplit
  .filter((item) => !item.existsInCorpus)
  .map((item) => item.scenarioId);
const unknownTruthKnownAcquisitionValidationIds = knownAcquisitionValidationSplit
  .filter((item) => item.modelTruth === 'unknown-signal')
  .map((item) => item.scenarioId);
const knownAcquisitionValidationNotExcludedIds = knownAcquisitionValidationSplit
  .filter((item) => !item.excludedFromComponentFit)
  .map((item) => item.scenarioId);
const exactEquivalenceSplit = [...exactEquivalenceIds].sort().map((scenarioId) => {
  const scenario = corpusScenarioById.get(scenarioId);
  const allowedModelTruths = scenario ? [...new Set(scenario.allowedObservableClasses.map(modelTruth))] : [];
  return {
    scenarioId,
    existsInCorpus: scenario !== undefined,
    corpusTruth: scenario?.truthClass,
    modelTruth: scenario ? modelTruth(scenario.truthClass) : undefined,
    allowedModelTruths,
    excludedFromComponentFit: scenarioExcludedIds.has(scenarioId),
  };
});
const invalidExactEquivalenceScenarioIds = exactEquivalenceSplit.filter((item) => !item.existsInCorpus).map((item) => item.scenarioId);
const nonUnknownExactEquivalenceScenarioIds = exactEquivalenceSplit
  .filter((item) => item.existsInCorpus && item.modelTruth !== 'unknown-signal')
  .map((item) => item.scenarioId);
const exactEquivalenceNotExcludedScenarioIds = exactEquivalenceSplit
  .filter((item) => !item.excludedFromComponentFit)
  .map((item) => item.scenarioId);
const exactEquivalenceWithoutDeclaredAlternativeIds = exactEquivalenceSplit
  .filter((item) => !item.allowedModelTruths.includes('unknown-signal')
    || !item.allowedModelTruths.some((truth) => truth !== 'unknown-signal'))
  .map((item) => item.scenarioId);
const componentAssignmentsByView = Object.fromEntries(OBSERVABLE_EVIDENCE_VIEWS.map((view) => [view,
  [...new Map(BAYESIAN_OBSERVABLE_MODEL.classModels
    .flatMap((model) => observableModelComponents(model, view)
      .map((component) => {
        const scenarioId = componentSourceScenarioId(component);
        return [`${model.id}:${scenarioId}`, { scenarioId, classId: model.id }] as const;
      }))).values()]
    .sort((left, right) => left.scenarioId.localeCompare(right.scenarioId)),
])) as Record<ObservableEvidenceView, { scenarioId: string; classId: ObservableLeafClass }[]>;
const componentAssignmentViewMismatches = OBSERVABLE_EVIDENCE_VIEWS.slice(1)
  .filter((view) => {
    const expected = componentAssignmentsByView['spectrum-only'].filter(
      (assignment) =>
        !(PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORED_SCENARIO_IDS as readonly string[])
          .includes(assignment.scenarioId),
    );
    return JSON.stringify(componentAssignmentsByView[view])
      !== JSON.stringify(expected);
  });
const componentArchitectureMismatches = OBSERVABLE_EVIDENCE_VIEWS.flatMap((view) => {
  const scenarioCount = componentAssignmentsByView[view].length;
  const componentCount = BAYESIAN_OBSERVABLE_MODEL.classModels.reduce(
    (sum, model) => sum + observableModelComponents(model, view).length,
    0,
  );
  return [
    ...(scenarioCount === PINNED_COMPONENT_SOURCE_SCENARIO_COUNTS_BY_VIEW[view]
      ? []
      : [`${view}:source-scenarios-${scenarioCount}-expected-${PINNED_COMPONENT_SOURCE_SCENARIO_COUNTS_BY_VIEW[view]}`]),
    ...(componentCount === PINNED_LIKELIHOOD_COMPONENT_COUNTS_BY_VIEW[view]
      ? []
      : [`${view}:components-${componentCount}-expected-${PINNED_LIKELIHOOD_COMPONENT_COUNTS_BY_VIEW[view]}`]),
  ];
});
const frequencyAgileCensoringMatrixMismatches =
  PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORED_SCENARIO_IDS.flatMap((scenarioId) => {
    const fittingCounts = BAYESIAN_OBSERVABLE_MODEL.trainingMatrix
      .fittingRepresentativeCountsByScenarioByView?.[scenarioId];
    const calibrationCounts = BAYESIAN_OBSERVABLE_MODEL.trainingMatrix
      .tailCalibrationAttemptCountsByScenarioByView?.[scenarioId];
    const censoredCounts = BAYESIAN_OBSERVABLE_MODEL.trainingMatrix
      .censoredFrequencyAgileFixedTuneCaptureCountsByScenario;
    return [
      ...((fittingCounts?.['spectrum-only'] ?? 0) > 0
        ? []
        : [`${scenarioId}:missing-spectrum-fitting-population`]),
      ...(fittingCounts?.['envelope-untimed'] === 0
        && fittingCounts?.['envelope-timed'] === 0
        ? []
        : [`${scenarioId}:nonzero-envelope-fitting-population`]),
      ...((calibrationCounts?.['spectrum-only'] ?? 0) > 0
        ? []
        : [`${scenarioId}:missing-spectrum-calibration-population`]),
      ...(calibrationCounts?.['envelope-untimed'] === 0
        && calibrationCounts?.['envelope-timed'] === 0
        ? []
        : [`${scenarioId}:nonzero-envelope-calibration-population`]),
      ...((censoredCounts?.fitting[scenarioId] ?? 0) > 0
        ? []
        : [`${scenarioId}:missing-fitted-censored-capture-audit`]),
      ...((censoredCounts?.tailCalibration[scenarioId] ?? 0) > 0
        ? []
        : [`${scenarioId}:missing-calibration-censored-capture-audit`]),
    ];
  });
const bluetoothModel = BAYESIAN_OBSERVABLE_MODEL.classModels.find(
  (model) => model.id === 'bluetooth-like',
);
if (!bluetoothModel
  || observableModelComponents(bluetoothModel, 'spectrum-only').length <= 0
  || observableModelComponents(bluetoothModel, 'envelope-untimed').length !== 0
  || observableModelComponents(bluetoothModel, 'envelope-timed').length !== 0
  || (bluetoothModel.tailCalibrationScoresByView?.['spectrum-only']?.length ?? 0) <= 0
  || (bluetoothModel.tailCalibrationScoresByView?.['envelope-untimed']?.length ?? 0) !== 0
  || (bluetoothModel.tailCalibrationScoresByView?.['envelope-timed']?.length ?? 0) !== 0) {
  componentArchitectureMismatches.push(
    'bluetooth-like:requires-positive-spectrum-and-exact-empty-envelope-support',
  );
}
const fittedComponentIds = new Set(componentAssignmentsByView['spectrum-only']
  .map((assignment) => assignment.scenarioId));
const fittedUnknownModel = BAYESIAN_OBSERVABLE_MODEL.classModels.find((model) => model.id === 'unknown-signal');
const modelFittedUnknownScenarioIds = (fittedUnknownModel
  ? [...new Set(observableModelComponents(fittedUnknownModel, 'spectrum-only')
    .map(componentSourceScenarioId))]
  : []).sort();
const fittedUnknownMissingPinnedIds = setDifference(PINNED_FITTED_UNKNOWN_SCENARIO_IDS, modelFittedUnknownScenarioIds);
const fittedUnknownUnexpectedIds = setDifference(modelFittedUnknownScenarioIds, PINNED_FITTED_UNKNOWN_SCENARIO_IDS);
const exactEquivalenceFittedComponentIds = [...exactEquivalenceIds].filter((scenarioId) => fittedComponentIds.has(scenarioId)).sort();
const knownAcquisitionValidationFittedComponentIds = [...knownAcquisitionValidationIds]
  .filter((scenarioId) => fittedComponentIds.has(scenarioId))
  .sort();
const expectedComponentAssignments = canonicalClassificationScenarios
  .filter((scenario) => !scenarioExcludedIds.has(scenario.id))
  .map((scenario) => ({ scenarioId: scenario.id, classId: modelTruth(scenario.truthClass) }))
  .sort((left, right) => left.scenarioId.localeCompare(right.scenarioId));
const likelihoodComponentOwnershipMismatches = OBSERVABLE_EVIDENCE_VIEWS.flatMap((view) =>
  BAYESIAN_OBSERVABLE_MODEL.classModels.flatMap((model) => {
    const components = observableModelComponents(model, view);
    const bySourceScenario = new Map<string, typeof components>();
    const mismatches = components.flatMap((component) => {
      if (component.sourceScenarioId === undefined || component.modeId === undefined
        || component.fitSampleCount === undefined) {
        return [`${view}/${model.id}/${component.id}:missing-explicit-ownership`];
      }
      const owned = bySourceScenario.get(component.sourceScenarioId) ?? [];
      bySourceScenario.set(component.sourceScenarioId, [...owned, component]);
      return [];
    });
    for (const [sourceScenarioId, sourceComponents] of bySourceScenario) {
      const scenario = corpusScenarioById.get(sourceScenarioId);
      if (!scenario) {
        mismatches.push(`${view}/${model.id}/${sourceScenarioId}:not-in-corpus`);
        continue;
      }
      const expectedModeCount = scenario.envelopeModel
        === PINNED_LIKELIHOOD_COMPONENT_DECOMPOSITION_POLICY.csmaEnvelopeModel
        ? PINNED_LIKELIHOOD_COMPONENT_DECOMPOSITION_POLICY.csmaModeCount
        : 1;
      if (sourceComponents.length !== expectedModeCount) {
        mismatches.push(`${view}/${model.id}/${sourceScenarioId}:component-count-${sourceComponents.length}-expected-${expectedModeCount}`);
        continue;
      }
      const expectedFitSampleCount = BAYESIAN_OBSERVABLE_MODEL.trainingMatrix
        .fittingRepresentativeCountsByScenarioByView?.[sourceScenarioId]?.[view];
      const observedFitSampleCount = sourceComponents.reduce(
        (sum, component) => sum + (component.fitSampleCount ?? 0),
        0,
      );
      if (!Number.isSafeInteger(expectedFitSampleCount) || expectedFitSampleCount! <= 0
        || observedFitSampleCount !== expectedFitSampleCount) {
        mismatches.push(`${view}/${model.id}/${sourceScenarioId}:fit-count-${observedFitSampleCount}-expected-${expectedFitSampleCount ?? 'missing'}`);
      }
      if (expectedModeCount === 1) {
        const component = sourceComponents[0]!;
        if (component.id !== sourceScenarioId || component.modeId !== 'single-population') {
          mismatches.push(`${view}/${model.id}/${sourceScenarioId}:invalid-single-population-identity`);
        }
      } else {
        const sharedScale = JSON.stringify(sourceComponents[0]!.scale);
        sourceComponents.forEach((component, index) => {
          const expectedModeId = `csma-activity-mode-${index + 1}-of-${expectedModeCount}`;
          if (component.id !== `${sourceScenarioId}/${expectedModeId}`
            || component.modeId !== expectedModeId
            || JSON.stringify(component.scale) !== sharedScale) {
            mismatches.push(`${view}/${model.id}/${sourceScenarioId}:invalid-mode-${index + 1}-identity-or-scale`);
          }
          if (!Number.isSafeInteger(component.fitSampleCount)
            || component.fitSampleCount!
              < PINNED_LIKELIHOOD_COMPONENT_DECOMPOSITION_POLICY.minimumModeFitSampleCount) {
            mismatches.push(`${view}/${model.id}/${sourceScenarioId}:mode-${index + 1}-fit-count-${component.fitSampleCount ?? 'missing'}`);
          }
        });
        const partitionDimensionIndex = sourceComponents[0]!.dimensions.indexOf(
          PINNED_LIKELIHOOD_COMPONENT_DECOMPOSITION_POLICY.csmaPartitionFeature,
        );
        const partitionCenters = sourceComponents.map((component) =>
          component.location[partitionDimensionIndex]);
        if (partitionDimensionIndex < 0 || partitionCenters.some((center, index) =>
          !Number.isFinite(center) || (index > 0 && center! <= partitionCenters[index - 1]!))) {
          mismatches.push(`${view}/${model.id}/${sourceScenarioId}:non-increasing-partition-centers`);
        }
      }
      for (const component of sourceComponents) {
        const expectedWeight = (1 / bySourceScenario.size)
          * ((component.fitSampleCount ?? 0) / observedFitSampleCount);
        if (!Number.isFinite(component.logWeight)
          || Math.abs(Math.exp(component.logWeight) - expectedWeight) > 1e-9) {
          mismatches.push(`${view}/${model.id}/${component.id}:invalid-source-owned-weight`);
        }
      }
    }
    return mismatches;
  }));
const modelDecomposedSourceScenarioIds = [...new Set(BAYESIAN_OBSERVABLE_MODEL.classModels
  .flatMap((model) => {
    const components = observableModelComponents(model, 'spectrum-only');
    const ownerCounts = counts(components.map(componentSourceScenarioId));
    return components.map(componentSourceScenarioId)
      .filter((sourceScenarioId) => ownerCounts[sourceScenarioId] === 3);
  }))].sort();
if (JSON.stringify(modelDecomposedSourceScenarioIds)
  !== JSON.stringify(PINNED_CSMA_DECOMPOSED_SOURCE_SCENARIO_IDS)) {
  likelihoodComponentOwnershipMismatches.push(
    `spectrum-only:decomposed-source-set-${modelDecomposedSourceScenarioIds.join(',')}`,
  );
}
const decomposedModeFitSampleCounts = OBSERVABLE_EVIDENCE_VIEWS.flatMap((view) =>
  BAYESIAN_OBSERVABLE_MODEL.classModels.flatMap((model) => {
    const components = observableModelComponents(model, view);
    const ownerCounts = counts(components.map(componentSourceScenarioId));
    return components
      .filter((component) => ownerCounts[componentSourceScenarioId(component)] === 3)
      .map((component) => component.fitSampleCount ?? 0);
  }));
const minimumDecomposedModeFitSampleCount = decomposedModeFitSampleCounts.length > 0
  ? Math.min(...decomposedModeFitSampleCounts)
  : 0;
const independentTailCalibrationAudit = recomputeTailCalibrationAudit(expectedComponentAssignments);
const expectedComponentClassByScenario = new Map(expectedComponentAssignments
  .map((assignment) => [assignment.scenarioId, assignment.classId] as const));
const actualComponentAssignments = componentAssignmentsByView['spectrum-only'];
const actualComponentScenarioIds = actualComponentAssignments.map((assignment) => assignment.scenarioId);
const expectedComponentScenarioIds = expectedComponentAssignments.map((assignment) => assignment.scenarioId);
const duplicateFittedComponentScenarioIds = duplicateStrings(actualComponentScenarioIds);
const missingFittedComponentScenarioIds = setDifference(expectedComponentScenarioIds, actualComponentScenarioIds);
const unexpectedFittedComponentScenarioIds = setDifference(actualComponentScenarioIds, expectedComponentScenarioIds);
const wrongClassFittedComponents = actualComponentAssignments
  .filter((assignment) => expectedComponentClassByScenario.has(assignment.scenarioId)
    && expectedComponentClassByScenario.get(assignment.scenarioId) !== assignment.classId)
  .map((assignment) => ({
    ...assignment,
    expectedClassId: expectedComponentClassByScenario.get(assignment.scenarioId)!,
  }));
const duplicateModelClassIds = duplicateStrings(BAYESIAN_OBSERVABLE_MODEL.classModels.map((model) => model.id));
const missingModelClassIds = setDifference(OBSERVABLE_LEAF_CLASSES, BAYESIAN_OBSERVABLE_MODEL.classModels.map((model) => model.id));
const unexpectedModelClassIds = setDifference(BAYESIAN_OBSERVABLE_MODEL.classModels.map((model) => model.id), OBSERVABLE_LEAF_CLASSES);
const ambiguousUnknownIncludedInComponentFitIds = canonicalClassificationScenarios
  .filter((scenario) => modelTruth(scenario.truthClass) === 'unknown-signal'
    && scenario.allowedObservableClasses.some((truth) => modelTruth(truth) !== 'unknown-signal')
    && !scenarioExcludedIds.has(scenario.id))
  .map((scenario) => scenario.id)
  .sort();
const fittedUnknownScenarioIds = canonicalClassificationScenarios
  .filter((scenario) => modelTruth(scenario.truthClass) === 'unknown-signal' && !scenarioExcludedIds.has(scenario.id))
  .map((scenario) => scenario.id)
  .sort();
const corpusFittedUnknownMissingPinnedIds = setDifference(PINNED_FITTED_UNKNOWN_SCENARIO_IDS, fittedUnknownScenarioIds);
const corpusFittedUnknownUnexpectedIds = setDifference(fittedUnknownScenarioIds, PINNED_FITTED_UNKNOWN_SCENARIO_IDS);
const manifestSplitValid = invalidExcludedScenarioIds.length === 0
  && nonUnknownExcludedScenarioIds.length === 0
  && duplicateExcludedScenarioIds.length === 0
  && modelExcludedMissingPinnedIds.length === 0
  && modelExcludedUnexpectedIds.length === 0
  && invalidPinnedStrictUnknownHoldoutIds.length === 0
  && invalidPinnedAmbiguityStressIds.length === 0
  && invalidPinnedFittedUnknownIds.length === 0
  && invalidPinnedExactEquivalencePairs.length === 0
  && knownAcquisitionValidationIds.size > 0
  && duplicateKnownAcquisitionValidationIds.length === 0
  && modelKnownAcquisitionMissingPinnedIds.length === 0
  && modelKnownAcquisitionUnexpectedIds.length === 0
  && invalidKnownAcquisitionValidationIds.length === 0
  && unknownTruthKnownAcquisitionValidationIds.length === 0
  && knownAcquisitionValidationNotExcludedIds.length === 0
  && knownAcquisitionValidationFittedComponentIds.length === 0
  && duplicateFittedComponentScenarioIds.length === 0
  && missingFittedComponentScenarioIds.length === 0
  && unexpectedFittedComponentScenarioIds.length === 0
  && wrongClassFittedComponents.length === 0
  && likelihoodComponentOwnershipMismatches.length === 0
  && componentScenarioPopulationMismatches.length === 0
  && componentAssignmentViewMismatches.length === 0
  && componentArchitectureMismatches.length === 0
  && frequencyAgileCensoringMatrixMismatches.length === 0
  && duplicateModelClassIds.length === 0
  && missingModelClassIds.length === 0
  && unexpectedModelClassIds.length === 0
  && exactEquivalenceIds.size > 0
  && duplicateExactEquivalenceScenarioIds.length === 0
  && modelExactEquivalenceMissingPinnedIds.length === 0
  && modelExactEquivalenceUnexpectedIds.length === 0
  && invalidExactEquivalenceScenarioIds.length === 0
  && nonUnknownExactEquivalenceScenarioIds.length === 0
  && exactEquivalenceNotExcludedScenarioIds.length === 0
  && exactEquivalenceWithoutDeclaredAlternativeIds.length === 0
  && exactEquivalenceFittedComponentIds.length === 0
  && ambiguousUnknownIncludedInComponentFitIds.length === 0
  && excludedUnknownScenarioIds.length > 0
  && fittedUnknownScenarioIds.length > 0
  && fittedUnknownMissingPinnedIds.length === 0
  && fittedUnknownUnexpectedIds.length === 0
  && corpusFittedUnknownMissingPinnedIds.length === 0
  && corpusFittedUnknownUnexpectedIds.length === 0;
const scenarioExcludedUnknown = unknown.filter((item) => scenarioExcludedIds.has(item.scenario));
const strictUnknownHoldoutIds = new Set<string>(PINNED_STRICT_UNKNOWN_HOLDOUT_SCENARIO_IDS);
const ambiguityStressIds = new Set<string>(PINNED_OBSERVABLE_AMBIGUITY_STRESS_SCENARIO_IDS);
const scenarioExcludedStrictUnknown = scenarioExcludedUnknown.filter((item) => strictUnknownHoldoutIds.has(item.scenario));
const scenarioExcludedExactEquivalence = scenarioExcludedUnknown.filter((item) => exactEquivalenceIds.has(item.scenario));
const scenarioExcludedNonExactAmbiguous = scenarioExcludedUnknown.filter((item) => ambiguityStressIds.has(item.scenario));
const fittedTemplateCases = cases.filter((item) => !scenarioExcludedIds.has(item.scenario) && item.componentFitEligible);
const identifiableFitEligibleKnown = known.filter((item) => !scenarioExcludedIds.has(item.scenario) && item.componentFitEligible);
const fittedUnknownTemplates = unknown.filter((item) => !scenarioExcludedIds.has(item.scenario) && item.componentFitEligible);
const knownCovered = identifiableFitEligibleKnown.filter((item) => item.result !== 'unknown');
const labels = [...OBSERVABLE_LEAF_CLASSES];
// A one-hot target is a proper score only where the corpus declares one
// allowed observable truth. Ambiguous/equivalent cases remain decision- and
// set-compatibility tests, never secretly scored against one privileged label.
const singletonAllowedTruthFittedTemplateCases = fittedTemplateCases.filter((item) => item.allowedModelTruths.length === 1);
const singletonAllowedTruthCases = cases.filter((item) => item.allowedModelTruths.length === 1);
const fittedTemplateBrier = mean(singletonAllowedTruthFittedTemplateCases.map((item) => labels.reduce((sum, label) => {
  const probability = item.posterior[label] ?? 0;
  const target = label === item.modelTruth ? 1 : 0;
  return sum + (probability - target) ** 2;
}, 0)));
const fittedTemplateLogLoss = -mean(singletonAllowedTruthFittedTemplateCases.map((item) => Math.log(Math.max(1e-15, item.truthPosterior))));
const fittedTemplateEce = expectedCalibrationError(singletonAllowedTruthFittedTemplateCases.map((item) => ({ confidence: item.topLeafPosterior, correct: item.topLeaf === item.modelTruth })), 10);
const allSingletonAllowedTruthLogLossDiagnostic = -mean(singletonAllowedTruthCases.map((item) => Math.log(Math.max(1e-15, item.truthPosterior))));
const fittedUnknownPosteriorAuroc = auroc([
  ...fittedUnknownTemplates.map((item) => ({ score: item.unknownPosterior, positive: true })),
  ...identifiableFitEligibleKnown.map((item) => ({ score: item.unknownPosterior, positive: false })),
]);
const scenarioExcludedStrictTypicalityAuroc = auroc([
  ...scenarioExcludedStrictUnknown.map((item) => ({ score: 1 - item.knownSupportRank, positive: true })),
  ...identifiableFitEligibleKnown.map((item) => ({ score: 1 - item.knownSupportRank, positive: false })),
]);
const exactEquivalenceCompatibleRate = fraction(scenarioExcludedExactEquivalence, (item) => item.acceptedHierarchy);
const strictHoldoutRejectionRate = fraction(scenarioExcludedStrictUnknown, (item) => item.result === 'unknown');
const confusion = Object.fromEntries(canonicalClassificationScenarios.map((scenario) => [
  scenario.id,
  counts(cases.filter((item) => item.scenario === scenario.id).map((item) => item.result)),
]));
const classwiseKnown = Object.fromEntries(OBSERVABLE_LEAF_CLASSES.filter((truth) => truth !== 'unknown-signal').map((truth) => {
  const selected = identifiableFitEligibleKnown.filter((item) => item.modelTruth === truth);
  return [truth, {
    samples: selected.length,
    topLeafAccuracy: fraction(selected, (item) => item.topLeaf === item.modelTruth),
    hierarchicalAccuracy: fraction(selected, (item) => item.acceptedHierarchy),
    coverage: fraction(selected, (item) => item.result !== 'unknown'),
  }];
}));
const minimumKnownClassHierarchicalAccuracy = Math.min(...Object.values(classwiseKnown).map((value) => value.hierarchicalAccuracy));
// A per-class sensitivity floor is meaningful only in the SNR region where
// acquisition itself is required to be reliable. Lower-SNR rows remain in the
// all-SNR diagnostics and global proper scores, but an honest open-set model is
// allowed to abstain there instead of being forced to label an atypical trace.
const classwiseKnownHighSnr = Object.fromEntries(OBSERVABLE_LEAF_CLASSES.filter((truth) => truth !== 'unknown-signal').map((truth) => {
  const selected = identifiableFitEligibleKnown.filter((item) => item.modelTruth === truth && item.snrDb >= HIGH_SNR_MINIMUM_DB);
  return [truth, {
    samples: selected.length,
    topLeafAccuracy: fraction(selected, (item) => item.topLeaf === item.modelTruth),
    hierarchicalAccuracy: fraction(selected, (item) => item.acceptedHierarchy),
    coverage: fraction(selected, (item) => item.result !== 'unknown'),
  }];
}));
const minimumHighSnrKnownClassHierarchicalAccuracy = Math.min(
  ...Object.values(classwiseKnownHighSnr).map((value) => value.hierarchicalAccuracy),
);
const admissionByScenario = Object.fromEntries(canonicalClassificationScenarios.map((scenario) => {
  const selected = admissionAttempts.filter((item) => item.scenario === scenario.id);
  return [scenario.id, {
    corpusTruth: scenario.truthClass,
    modelTruth: modelTruth(scenario.truthClass),
    allowedModelTruths: [...new Set(scenario.allowedObservableClasses.map(modelTruth))],
    allSnr: admissionSummary(selected),
    highSnr: admissionSummary(selected.filter((item) => item.snrDb >= HIGH_SNR_MINIMUM_DB)),
  }];
}));
const bySnr = Object.fromEntries(SNR_DB.map((snrDb) => {
  const selected = cases.filter((item) => item.snrDb === snrDb);
  return [snrDb, {
    attempts: admissionSummary(admissionAttempts.filter((item) => item.snrDb === snrDb)),
    firstReadyRepresentativeSamples: selected.length,
    topLeafAccuracy: fraction(selected, (item) => item.topLeaf === item.modelTruth),
    hierarchicalAccuracy: fraction(selected, (item) => item.acceptedHierarchy),
    knownCoverage: fraction(selected.filter((item) => item.modelTruth !== 'unknown-signal'), (item) => item.result !== 'unknown'),
    unknownRejection: fraction(selected.filter((item) => item.modelTruth === 'unknown-signal'), (item) => item.result === 'unknown'),
  }];
}));
const byRbwDivisor = Object.fromEntries(RBW_DIVISORS.map((rbwDivisor) => {
  const selectedAttempts = admissionAttempts.filter((item) => item.rbwDivisor === rbwDivisor);
  const selectedCases = cases.filter((item) => item.rbwDivisor === rbwDivisor);
  return [rbwDivisor, {
    attempts: admissionSummary(selectedAttempts),
    firstReadyRepresentativeSamples: selectedCases.length,
    actualRbwHz: numericSummary(selectedAttempts.map((item) => item.actualRbwHz)),
    detectedPowerSynthesisFilterWidthHz: numericSummary(
      selectedAttempts.map((item) => item.detectedPowerSynthesisFilterWidthHz),
    ),
    binWidthHz: numericSummary(selectedAttempts.map((item) => item.binWidthHz)),
    rbwToBinWidthRatio: numericSummary(selectedAttempts.map((item) => item.actualRbwHz / item.binWidthHz)),
    hierarchicalAccuracy: fraction(selectedCases, (item) => item.acceptedHierarchy),
    unknownRejection: fraction(selectedCases.filter((item) => item.modelTruth === 'unknown-signal'), (item) => item.result === 'unknown'),
  }];
}));
const evidenceViews = Object.fromEntries((['spectrum-only', 'envelope-untimed'] as const).map((view) => {
  const selected = evidenceViewCases.filter((item) => item.view === view);
  const selectedKnown = selected.filter((item) => item.modelTruth !== 'unknown-signal'
    && !scenarioExcludedIds.has(item.scenario)
    && item.componentFitEligible);
  const selectedFittedUnknown = selected.filter((item) => item.modelTruth === 'unknown-signal'
    && !scenarioExcludedIds.has(item.scenario)
    && item.componentFitEligible);
  const selectedScenarioExcluded = selected.filter((item) => scenarioExcludedIds.has(item.scenario));
  const selectedScenarioExcludedStrict = selectedScenarioExcluded.filter((item) => strictUnknownHoldoutIds.has(item.scenario));
  const selectedExactEquivalence = selectedScenarioExcluded.filter((item) => exactEquivalenceIds.has(item.scenario));
  const selectedFittedDomain = selected.filter((item) => !scenarioExcludedIds.has(item.scenario) && item.componentFitEligible);
  const selectedSingletonTruthFittedDomain = selectedFittedDomain.filter((item) => item.allowedModelTruths.length === 1);
  const falseAcceptedUnknown = selected.filter((item) => item.modelTruth === 'unknown-signal' && !item.acceptedHierarchy);
  const falseAcceptedAttemptIds = [...new Set(falseAcceptedUnknown.map((item) => item.attemptId))].sort();
  return [view, {
    admittedSamples: selected.length,
    hierarchicalAccuracy: fraction(selected, (item) => item.acceptedHierarchy),
    knownCoverage: fraction(selectedKnown, (item) => item.result !== 'unknown'),
    coveredKnownHierarchicalAccuracy: fraction(selectedKnown.filter((item) => item.result !== 'unknown'), (item) => item.acceptedHierarchy),
    fittedUnknownTemplateRejectionRate: fraction(selectedFittedUnknown, (item) => item.result === 'unknown'),
    validationOnlyUnknownDecisionRate: fraction(selectedScenarioExcluded, (item) => item.result === 'unknown'),
    exactEquivalenceSamples: selectedExactEquivalence.length,
    exactEquivalenceCompatibleRate: fraction(selectedExactEquivalence, (item) => item.acceptedHierarchy),
    strictHoldoutSamples: selectedScenarioExcludedStrict.length,
    strictHoldoutRejectionRate: fraction(selectedScenarioExcludedStrict, (item) => item.result === 'unknown'),
    falseAcceptedUnknownCount: falseAcceptedUnknown.length,
    anyFalseAcceptAttemptCount: falseAcceptedAttemptIds.length,
    anyFalseAcceptAttemptIds: falseAcceptedAttemptIds.slice(0, 50),
    falseAcceptedUnknownExamples: falseAcceptedUnknown.slice(0, 20),
    scenarioExcludedStrictSupportAuroc: auroc([
      ...selectedScenarioExcludedStrict.map((item) => ({ score: 1 - item.supportRank, positive: true })),
      ...selectedKnown.map((item) => ({ score: 1 - item.supportRank, positive: false })),
    ]),
    singletonAllowedTruthProperScoreSamples: selectedSingletonTruthFittedDomain.length,
    fittedTemplateLogLoss: -mean(selectedSingletonTruthFittedDomain.map((item) => Math.log(Math.max(1e-15, item.truthPosterior)))),
    fittedTemplateMulticlassBrier: mean(selectedSingletonTruthFittedDomain.map((item) => labels.reduce((sum, label) => {
      const probability = item.posterior[label] ?? 0;
      const target = label === item.modelTruth ? 1 : 0;
      return sum + (probability - target) ** 2;
    }, 0))),
    fittedTemplateExpectedCalibrationError: expectedCalibrationError(selectedSingletonTruthFittedDomain.map((item) => ({ confidence: item.topLeafPosterior, correct: item.topLeaf === item.modelTruth })), 10),
    knownSupport: numericSummary(selectedKnown.map((item) => item.supportRank)),
    scenarioExcludedStrictSupport: numericSummary(selectedScenarioExcludedStrict.map((item) => item.supportRank)),
  }];
}));
const falseAcceptedUnknown = unknown.filter((item) => !item.acceptedHierarchy);
const falseAcceptedUnknownAttemptIds = [...new Set(falseAcceptedUnknown.map((item) => item.attemptId))].sort();
const everReadyAttempts = admissionAttempts.filter((item) => item.everReady);
const firstReadyAttempts = admissionAttempts.filter((item) => item.admitted);
const physicalEnvelopeCaptureAttempts = admissionAttempts.filter((item) => item.detectedPowerCaptureCount === 1);
const censoredFrequencyAgileCaptureAttempts = physicalEnvelopeCaptureAttempts
  .filter((item) => item.detectedPowerEvidenceDisposition
    === 'censored-frequency-agile-spectrum-only');
const expectedCausalEnvelopeSamples = physicalEnvelopeCaptureAttempts.length
  - censoredFrequencyAgileCaptureAttempts.length;
const unavailablePhysicalEnvelopeCaptureAttempts = physicalEnvelopeCaptureAttempts
  .filter((item) => item.detectedPowerEvidenceDisposition === 'admitted-envelope'
    && !item.envelopeFeatureAvailable);
const qualifiedCausalEnvelopeSamples = cases.filter((item) =>
  item.detectedPowerEvidenceDisposition === 'admitted-envelope'
  && item.views.includes('detected-power-envelope')
  && item.detectedPowerAcquisitionQualification
    === PINNED_DETECTED_POWER_ACQUISITION_QUALIFICATION
  && item.detectedPowerSelectionCondition
    === PINNED_DETECTED_POWER_SELECTION_CONDITION);
const censoredFrequencyAgileCaptureCases = cases.filter((item) =>
  item.detectedPowerEvidenceDisposition
    === 'censored-frequency-agile-spectrum-only');
const invalidCensoredFrequencyAgileCaptureCases =
  censoredFrequencyAgileCaptureCases.filter((item) =>
    item.associationMode !== 'frequency-agile-2g4-activity'
    || item.captureProjectionKind !== 'current-qualified-agile-latest-member'
    || item.envelopeEvidenceCensoringPolicyId
      !== PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORING_POLICY_ID
    || item.detectedPowerCaptureReceiptSchemaVersion !== 4
    || !item.physicalCaptureId
    || item.views.length !== 1
    || item.views[0] !== 'scalar-spectrum'
    || item.zeroSpanCaptureId !== undefined
    || item.detectedPowerAcquisitionQualification !== undefined
    || item.detectedPowerSelectionCondition !== undefined
    || !item.limitations.includes(
      PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORING_LIMITATION,
    ));
const invalidUncensoredEnvelopeCases = qualifiedCausalEnvelopeSamples.filter(
  (item) => item.envelopeEvidenceCensoringPolicyId !== undefined
    || !item.views.includes('detected-power-envelope')
    || item.zeroSpanCaptureId !== item.physicalCaptureId
    || item.detectedPowerCaptureReceiptSchemaVersion !== 4
    || item.detectedPowerSelectionCondition
      !== PINNED_DETECTED_POWER_SELECTION_CONDITION,
);
const unqualifiedCausalEnvelopeSamples = cases.filter((item) =>
  item.detectedPowerEvidenceDisposition === 'admitted-envelope'
  && item.views.includes('detected-power-envelope')
  && (item.detectedPowerAcquisitionQualification
      !== PINNED_DETECTED_POWER_ACQUISITION_QUALIFICATION
    || item.detectedPowerSelectionCondition
      !== PINNED_DETECTED_POWER_SELECTION_CONDITION));
const missingOrUnissuedReceiptEnvelopeSamples = cases.filter((item) =>
  item.zeroSpanCaptureId !== undefined
  && (item.detectedPowerAcquisitionQualification
      !== PINNED_DETECTED_POWER_ACQUISITION_QUALIFICATION
    || item.detectedPowerSelectionCondition
      !== PINNED_DETECTED_POWER_SELECTION_CONDITION));
const missingOrUnissuedReceiptEnvelopeFeatureAttempts = admissionAttempts.filter((item) =>
  item.envelopeFeatureAvailable
  && (item.detectedPowerAcquisitionQualification
      !== PINNED_DETECTED_POWER_ACQUISITION_QUALIFICATION
    || item.detectedPowerSelectionCondition
      !== PINNED_DETECTED_POWER_SELECTION_CONDITION));
const invalidCausalCaptureSemantics = admissionAttempts.filter((item) =>
  item.detectedPowerCaptureCount > 1
  || item.detectedPowerCaptureCount !== (item.admitted ? 1 : 0)
  || item.detectedPowerCaptureReceiptVerified
    !== (item.detectedPowerCaptureCount === 1)
  || (item.detectedPowerCaptureCount === 1
    && (item.detectedPowerCaptureReceiptSchemaVersion !== 4
      || !item.physicalCaptureId
      || item.captureProjectionKind === undefined
      || item.projectedAssociationMode === undefined
      || item.classificationEvidenceView === undefined))
  || (item.envelopeFeatureAvailable && item.detectedPowerCaptureCount !== 1)
  || (item.detectedPowerEvidenceDisposition
      === 'censored-frequency-agile-spectrum-only'
    && (item.envelopeFeatureAvailable
      || item.envelopeEvidenceCensoringPolicyId
        !== PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORING_POLICY_ID
      || item.captureProjectionKind
        !== 'current-qualified-agile-latest-member'
      || item.projectedAssociationMode
        !== 'frequency-agile-2g4-activity'
      || item.classificationEvidenceView !== 'spectrum-only'
      || item.detectedPowerAcquisitionQualification !== undefined
      || item.detectedPowerSelectionCondition !== undefined))
  || (item.detectedPowerEvidenceDisposition === 'admitted-envelope'
    && (item.captureProjectionKind
        !== 'current-active-physical-representative'
      || item.projectedAssociationMode
        === 'frequency-agile-2g4-activity'
      || item.classificationEvidenceView === 'spectrum-only'
      || item.detectedPowerAcquisitionQualification
        !== PINNED_DETECTED_POWER_ACQUISITION_QUALIFICATION
      || item.detectedPowerSelectionCondition
        !== PINNED_DETECTED_POWER_SELECTION_CONDITION)));
const causalEnvelopeAvailabilityCells = admissionAttempts.map((item) => ({
  attemptId: item.attemptId,
  scenario: item.scenario,
  snrDb: item.snrDb,
  rbwDivisor: item.rbwDivisor,
  seed: item.seed,
  spectrumRuntimeAdmitted: item.admitted,
  detectedPowerCaptureCount: item.detectedPowerCaptureCount,
  detectedPowerCaptureReceiptVerified:
    item.detectedPowerCaptureReceiptVerified,
  ...(item.detectedPowerCaptureReceiptSchemaVersion === undefined
    ? {}
    : {
        detectedPowerCaptureReceiptSchemaVersion:
          item.detectedPowerCaptureReceiptSchemaVersion,
      }),
  ...(item.physicalCaptureId === undefined
    ? {}
    : { physicalCaptureId: item.physicalCaptureId }),
  ...(item.captureProjectionKind === undefined
    ? {}
    : { captureProjectionKind: item.captureProjectionKind }),
  ...(item.projectedAssociationMode === undefined
    ? {}
    : { projectedAssociationMode: item.projectedAssociationMode }),
  ...(item.classificationEvidenceView === undefined
    ? {}
    : { classificationEvidenceView: item.classificationEvidenceView }),
  envelopeFeatureAvailable: item.envelopeFeatureAvailable,
  ...(item.detectedPowerEvidenceDisposition === undefined
    ? {}
    : {
        detectedPowerEvidenceDisposition:
          item.detectedPowerEvidenceDisposition,
      }),
  ...(item.envelopeEvidenceCensoringPolicyId === undefined
    ? {}
    : {
        envelopeEvidenceCensoringPolicyId:
          item.envelopeEvidenceCensoringPolicyId,
      }),
  detectedPowerAcquisitionReceiptQualified:
    item.detectedPowerCaptureReceiptVerified,
  ...(item.detectedPowerAcquisitionQualification === undefined
    ? {}
    : {
        detectedPowerAcquisitionQualification:
          item.detectedPowerAcquisitionQualification,
      }),
  ...(item.detectedPowerSelectionCondition === undefined
    ? {}
    : {
        detectedPowerSelectionCondition:
          item.detectedPowerSelectionCondition,
      }),
  ...(item.envelopeFeatureUnavailableCode === undefined
    ? {}
    : { envelopeFeatureUnavailableCode: item.envelopeFeatureUnavailableCode }),
}));
const uniqueCaptureConditionalClassificationSamples = new Set(
  cases.map((item) => `${item.attemptId}|${item.representativeKey}`),
).size;
const uniqueCausalEnvelopeSamples = new Set(
  qualifiedCausalEnvelopeSamples.map(
    (item) => `${item.attemptId}|${item.representativeKey}`,
  ),
).size;
const associationModes = [
  'frequency-local',
  'frequency-agile-2g4-activity',
  'regular-spectral-component-activity',
  'multicomponent-swept-region-activity',
] as const;
const associationByMode = Object.fromEntries(associationModes.map((associationMode) => {
  const selected = cases.filter((item) => item.associationMode === associationMode);
  return [associationMode, summarizeAssociationCases(selected)];
}));
const soleEnvelopeAssociationByMode = Object.fromEntries(associationModes.map((associationMode) => {
  const selected = qualifiedCausalEnvelopeSamples.filter(
    (item) => item.associationMode === associationMode,
  );
  return [associationMode, summarizeAssociationCases(selected)];
}));
const spectrumOnlineAssociationByMode = Object.fromEntries(associationModes.map((associationMode) => {
  const selected = spectrumOnlineAssociationSamples.filter((item) => item.associationMode === associationMode);
  return [associationMode, {
    samples: selected.length,
    attempts: new Set(selected.map((item) => item.attemptId)).size,
    scenarios: [...new Set(selected.map((item) => item.scenario))].sort(),
  }];
}));
const spectrumOnlineAssociationKeys = spectrumOnlineAssociationSamples.map((item) =>
  `${item.attemptId}:${item.readyOpportunity}:${item.representativeKey}`);
const duplicateSpectrumOnlineAssociationKeys = duplicateStrings(spectrumOnlineAssociationKeys);
const associationModesWithoutCoverage = associationModes.filter((associationMode) => {
  const metrics = spectrumOnlineAssociationByMode[associationMode];
  return !metrics || metrics.samples <= 0 || metrics.scenarios.length <= 0;
});
const associationByScenario = Object.fromEntries(canonicalClassificationScenarios.map((scenario) => {
  const selectedAttempts = admissionAttempts.filter((item) => item.scenario === scenario.id);
  const selectedCases = cases.filter((item) => item.scenario === scenario.id);
  return [scenario.id, {
    firstReadyRepresentativeSamples: selectedCases.length,
    firstReadyModes: counts(selectedCases.map((item) => item.associationMode)),
    results: counts(selectedCases.map((item) => item.result)),
    attemptsEverRegularAssociation: selectedAttempts.filter((item) => item.everAssociationModes.includes('regular-spectral-component-activity')).length,
    attemptsFinalRegularAssociation: selectedAttempts.filter((item) => item.finalAssociationModes.includes('regular-spectral-component-activity')).length,
    attemptsEverAgileAssociation: selectedAttempts.filter((item) => item.everAssociationModes.includes('frequency-agile-2g4-activity')).length,
    attemptsFinalAgileAssociation: selectedAttempts.filter((item) => item.finalAssociationModes.includes('frequency-agile-2g4-activity')).length,
    regularAssociationsObserved: selectedAttempts.reduce((sum, item) => sum + item.regularAssociationsObserved, 0),
    agileAssociationsObserved: selectedAttempts.reduce((sum, item) => sum + item.agileAssociationsObserved, 0),
    regularAssociationExpirations: selectedAttempts.reduce((sum, item) => sum + item.regularAssociationExpirations, 0),
  }];
}));

function summarizeAssociationCases(selected: readonly ValidationCase[]) {
  const selectedUnknown = selected.filter(
    (item) => item.modelTruth === 'unknown-signal',
  );
  return {
    firstReadyRepresentativeSamples: selected.length,
    scenarios: [...new Set(selected.map((item) => item.scenario))].sort(),
    results: counts(selected.map((item) => item.result)),
    hierarchicalAccuracy: fraction(selected, (item) => item.acceptedHierarchy),
    unknownRejection: fraction(
      selectedUnknown,
      (item) => item.result === 'unknown',
    ),
    falseAcceptedUnknownCount: selectedUnknown.filter(
      (item) => !item.acceptedHierarchy,
    ).length,
    effectiveAdmissions: numericSummary(
      selected.map((item) => item.selectedTrackAdmissions),
    ),
    localTrackAdmissions: numericSummary(
      selected.map((item) => item.localTrackAdmissions),
    ),
    memberCount: numericSummary(selected.flatMap((item) =>
      item.associationMemberCount === undefined
        ? []
        : [item.associationMemberCount])),
    regionBandwidthHz: numericSummary(selected.flatMap((item) =>
      item.associationRegionBandwidthHz === undefined
        ? []
        : [item.associationRegionBandwidthHz])),
  };
}
function summarizeDetectedPowerCaptureOutcomes(
  selected: readonly ValidationCase[],
) {
  const qualifiedEnvelope = selected.filter((item) =>
    item.detectedPowerEvidenceDisposition === 'admitted-envelope');
  const censoredSpectrum = selected.filter((item) =>
    item.detectedPowerEvidenceDisposition
      === 'censored-frequency-agile-spectrum-only');
  return {
    physicalCaptureCount: selected.length,
    receiptQualifiedPhysicalCaptureCount: selected.filter((item) =>
      item.detectedPowerCaptureReceiptSchemaVersion === 4
      && Boolean(item.physicalCaptureId)).length,
    qualifiedEnvelopeSampleCount: qualifiedEnvelope.length,
    censoredDetectedPowerCaptureCount: censoredSpectrum.length,
    censoredSpectrumClassificationCount: censoredSpectrum.filter((item) =>
      item.views.length === 1 && item.views[0] === 'scalar-spectrum').length,
    selectedEvidenceViews: counts(selected.map((item) =>
      observableModelView({ values: item.features }))),
  };
}
const detectedPowerCaptureOutcomesByProjectedMode = Object.fromEntries(
  associationModes.map((associationMode) => [
    associationMode,
    summarizeDetectedPowerCaptureOutcomes(cases.filter(
      (item) => item.associationMode === associationMode,
    )),
  ]),
);
const detectedPowerCaptureOutcomesByProjectionKind = Object.fromEntries(([
  'current-active-physical-representative',
  'current-qualified-agile-latest-member',
] as const).map((projectionKind) => [
  projectionKind,
  summarizeDetectedPowerCaptureOutcomes(cases.filter(
    (item) => item.captureProjectionKind === projectionKind,
  )),
]));
const detectedPowerCaptureOutcomesByScenario = Object.fromEntries(
  canonicalClassificationScenarios.map((scenario) => [
    scenario.id,
    summarizeDetectedPowerCaptureOutcomes(cases.filter(
      (item) => item.scenario === scenario.id,
    )),
  ]),
);
const invalidBluetoothCaptureOutcomeScenarios = [
  'bluetooth-classic-connected',
  'bluetooth-le-advertising',
].filter((scenarioId) => {
  const outcome = detectedPowerCaptureOutcomesByScenario[scenarioId];
  return !outcome
    || outcome.physicalCaptureCount <= 0
    || outcome.receiptQualifiedPhysicalCaptureCount
      !== outcome.physicalCaptureCount
    || outcome.censoredDetectedPowerCaptureCount
      !== outcome.physicalCaptureCount
    || outcome.censoredSpectrumClassificationCount
      !== outcome.physicalCaptureCount
    || outcome.qualifiedEnvelopeSampleCount !== 0
    || outcome.selectedEvidenceViews['spectrum-only']
      !== outcome.physicalCaptureCount;
});
const limitationCounts = counts(cases.flatMap((item) => item.limitations));
const scenariosWithoutHighSnrAdmission = Object.entries(admissionByScenario)
  .filter(([, value]) => value.highSnr.admitted === 0)
  .map(([scenario]) => scenario);
const expectedClassificationNonAdmissionIds = new Set<string>(PINNED_EXPECTED_CLASSIFICATION_NON_ADMISSION_SCENARIO_IDS);
const expectedNonAdmissionScenariosWithAdmission = [...expectedClassificationNonAdmissionIds]
  .filter((scenario) => (admissionByScenario[scenario]?.allSnr.admitted ?? 0) > 0)
  .sort();
const knownAcquisitionWrongAdmissions = cases
  .filter((item) => knownAcquisitionValidationIds.has(item.scenario) && !item.acceptedHierarchy);
const ordinaryKnownScenarioIds = canonicalClassificationScenarios
  .filter((scenario) => modelTruth(scenario.truthClass) !== 'unknown-signal'
    && !knownAcquisitionValidationIds.has(scenario.id))
  .map((scenario) => scenario.id);
const admissionSeedCoverageByKnownScenario = Object.fromEntries(ordinaryKnownScenarioIds.map((scenarioId) => {
  const minimumCoverage = scenarioId === 'bluetooth-le-advertising' ? BLE_ADVERTISING_MINIMUM_SEED_COVERAGE : 1;
  const requiredSeeds = Math.ceil(NUISANCE_SHIFT_SEEDS.length * minimumCoverage);
  const bySnr = Object.fromEntries(ADMISSION_SEED_COVERAGE_SNR_DB.map((snrDb) => {
    const coveredSeeds = NUISANCE_SHIFT_SEEDS.filter((seed) => admissionAttempts.some((item) =>
      item.scenario === scenarioId && item.snrDb === snrDb && item.seed === seed && item.admitted));
    const uncoveredSeeds = NUISANCE_SHIFT_SEEDS.filter((seed) => !coveredSeeds.includes(seed));
    return [snrDb, {
      coveredSeeds,
      uncoveredSeeds,
      uniqueSeedsCovered: coveredSeeds.length,
      totalSeeds: NUISANCE_SHIFT_SEEDS.length,
      coverage: coveredSeeds.length / NUISANCE_SHIFT_SEEDS.length,
      requiredSeeds,
      passed: coveredSeeds.length >= requiredSeeds,
      admittingRbwDivisorsBySeed: Object.fromEntries(coveredSeeds.map((seed) => [seed, admissionAttempts
        .filter((item) => item.scenario === scenarioId && item.snrDb === snrDb && item.seed === seed && item.admitted)
        .map((item) => item.rbwDivisor)
        .sort((left, right) => left - right)])),
    }];
  }));
  return [scenarioId, { minimumCoverage, requiredSeeds, bySnr }];
}));
const knownAdmissionSeedCoverageFailures = Object.entries(admissionSeedCoverageByKnownScenario).flatMap(([scenarioId, audit]) =>
  Object.entries(audit.bySnr)
    .filter(([, cell]) => !cell.passed)
    .map(([snrDb, cell]) => ({ scenarioId, snrDb: Number(snrDb), ...cell })));

const expectedRollingKnownScenarioIds = expectedComponentAssignments
  .filter((assignment) => assignment.classId !== 'unknown-signal')
  .map((assignment) => assignment.scenarioId)
  .sort();
const highSnrKnownRollingWindowCases = rollingWindowCases.filter((item) =>
  item.modelTruth !== 'unknown-signal' && item.snrDb >= HIGH_SNR_MINIMUM_DB);
const observedRollingKnownScenarioIds = [...new Set(highSnrKnownRollingWindowCases.map((item) => item.scenario))].sort();
const missingRollingKnownScenarioIds = setDifference(expectedRollingKnownScenarioIds, observedRollingKnownScenarioIds);
const onlineSpectrumKeys = rollingWindowCases.map((item) =>
  `${item.attemptId}:${item.readyOpportunity}:${item.representativeKey}`);
const uniqueOnlineSpectrumCases = new Set(onlineSpectrumKeys).size;
const duplicateOnlineSpectrumKeys = duplicateStrings(onlineSpectrumKeys);
const rollingWindowKeys = highSnrKnownRollingWindowCases.map((item) =>
  `${item.attemptId}:${item.readyOpportunity}:${item.representativeKey}`);
const uniqueRollingWindowCases = new Set(rollingWindowKeys).size;
const duplicateRollingWindowKeys = duplicateStrings(rollingWindowKeys);
const rollingKnownCoverage = fraction(highSnrKnownRollingWindowCases, (item) => item.result !== 'unknown');
const rollingKnownHierarchicalAccuracy = fraction(highSnrKnownRollingWindowCases, (item) => item.acceptedHierarchy);
const rollingIncompatibleNonUnknown = highSnrKnownRollingWindowCases
  .filter((item) => item.result !== 'unknown' && !item.acceptedHierarchy);
const onlineSpectrumIncompatibleNonUnknown = rollingWindowCases
  .filter((item) => item.result !== 'unknown' && !item.acceptedHierarchy);
const onlineUnknownSpectrumCases = rollingWindowCases.filter((item) => item.modelTruth === 'unknown-signal');
const onlineUnknownFalseAccepts = onlineUnknownSpectrumCases.filter((item) => !item.acceptedHierarchy);
const onlineSpectrumSingletonTruthFittedDomain = rollingWindowCases.filter((item) =>
  !scenarioExcludedIds.has(item.scenario)
  && item.truthClassDomainEligible
  && item.allowedModelTruths.length === 1);
const onlineSpectrumFittedTemplateLogLoss = -mean(onlineSpectrumSingletonTruthFittedDomain
  .map((item) => Math.log(Math.max(1e-15, item.posterior[item.modelTruth] ?? 0))));
const onlineSpectrumFittedTemplateMulticlassBrier = mean(onlineSpectrumSingletonTruthFittedDomain.map((item) =>
  labels.reduce((sum, label) => {
    const probability = item.posterior[label] ?? 0;
    const target = label === item.modelTruth ? 1 : 0;
    return sum + (probability - target) ** 2;
  }, 0)));
const onlineSpectrumFittedTemplateExpectedCalibrationError = expectedCalibrationError(
  onlineSpectrumSingletonTruthFittedDomain.map((item) => ({
    confidence: item.topLeafPosterior,
    correct: item.topLeaf === item.modelTruth,
  })),
  10,
);
const truthClassDomainRollingWindowCases = highSnrKnownRollingWindowCases
  .filter((item) => item.truthClassDomainEligible);
const rollingByScenario = Object.fromEntries(expectedRollingKnownScenarioIds.map((scenarioId) => {
  const selected = highSnrKnownRollingWindowCases.filter((item) => item.scenario === scenarioId);
  return [scenarioId, {
    cases: selected.length,
    knownCoverage: fraction(selected, (item) => item.result !== 'unknown'),
    hierarchicalAccuracy: fraction(selected, (item) => item.acceptedHierarchy),
    minimumSupportRank: selected.length ? Math.min(...selected.map((item) => item.knownSupportRank)) : 0,
    truthClassDomainEligibleCases: selected.filter((item) => item.truthClassDomainEligible).length,
  }];
}));
const minimumRollingScenarioCoverage = Math.min(...Object.values(rollingByScenario).map((item) => item.knownCoverage));
const minimumRollingScenarioHierarchicalAccuracy = Math.min(
  ...Object.values(rollingByScenario).map((item) => item.hierarchicalAccuracy),
);
const priorSensitivityAudit = auditPriorSensitivity(
  cases,
  'capture-qualified-selected-view',
);
const completeOnlineSpectrumPriorSensitivityAudit = auditPriorSensitivity(
  rollingWindowCases,
  'complete-online-spectrum',
);

const report = {
  qualification: 'production-detector-conditioned-mixed-nuisance-shift-and-scenario-excluded-synthetic-only',
  interpretation: 'This is development-regression evidence from re-simulated SignalLab scalar formulas. Every nuisance cell is acquired twice with fresh detector, tracker, history, and source-clock state. The App-compatible consecutive-spectrum branch begins at held-out source look 512, consumes only sequential swept spectra, and supplies every spectrum likelihood, rolling decision, association, proper-score, and spectrum-prior audit. The separately qualified envelope branch begins at held-out source look 524, ranks current physical targets by current-source-sweep integrated excess power, and permits at most one capture only when rank 0 passes runtime admission; a lower-ranked target is never substituted. It continues later spectra at the next source look and exclusively supplies timed and untimed envelope audits. No spectrum event from the envelope session enters a spectrum population. Evidence is restricted to the prefix available at each decision; no endpoint, future-look, or retrospective best-track selection is used. A separate audit proves declared absolute-look drift remains inside each scenario span. The fitted formulas, SNR grid, and acquisition geometry overlap development, so this is not untouched validation, physical receiver calibration, waveform conformance, emitter identity, or protocol validation.',
  selectionPolicy: SELECTION_POLICY,
  model: BAYESIAN_WAVEFORM_MODEL,
  priorSensitivity: {
    ...priorSensitivityAudit,
    completeOnlineSpectrum: completeOnlineSpectrumPriorSensitivityAudit,
  },
  integrity: {
    checkedOutCorpusSourceManifest,
    checkedOutCorpusSha256,
    checkedInModelAssetSha256,
    modelAssetManifestSha256: BAYESIAN_OBSERVABLE_MODEL_SHA256,
  },
  corpus: {
    version: CLASSIFICATION_CORPUS_VERSION,
    scenarios: canonicalClassificationScenarios.length,
    scenarioExcludedFromComponentFit: scenarioExcludedIds.size,
    manifestSplit: {
      valid: manifestSplitValid,
      validatorOwnedPins: {
        fittedUnknown: PINNED_FITTED_UNKNOWN_SCENARIO_IDS,
        strictUnknownHoldout: PINNED_STRICT_UNKNOWN_HOLDOUT_SCENARIO_IDS,
        observableAmbiguityStress: PINNED_OBSERVABLE_AMBIGUITY_STRESS_SCENARIO_IDS,
        exactObservableEquivalencePairs: PINNED_EXACT_OBSERVABLE_EQUIVALENCE_PAIRS,
        knownAcquisitionValidationOnly: PINNED_KNOWN_ACQUISITION_VALIDATION_ONLY_SCENARIO_IDS,
        frequencyAgileEnvelopeCensoredScenarios:
          PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORED_SCENARIO_IDS,
        componentSourceScenarioCountsByView:
          PINNED_COMPONENT_SOURCE_SCENARIO_COUNTS_BY_VIEW,
        likelihoodComponentCountsByView:
          PINNED_LIKELIHOOD_COMPONENT_COUNTS_BY_VIEW,
        excludedFromComponentFit: PINNED_SCENARIO_EXCLUDED_FROM_COMPONENT_FIT_IDS,
      },
      modelDeclared: {
        fittedUnknown: modelFittedUnknownScenarioIds,
        likelihoodComponentDecompositionPolicy:
          BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.likelihoodComponentDecompositionPolicy,
        minimumDecomposedModeFitSampleCount,
        exactObservableEquivalenceNulls: modelExactEquivalenceIdList,
        knownAcquisitionValidationOnly: modelKnownAcquisitionValidationIdList,
        excludedFromComponentFit: modelScenarioExcludedIdList,
        componentAssignments: actualComponentAssignments,
        componentAssignmentsByView,
      },
      expectedComponentAssignments,
      excludedScenarios: excludedScenarioSplit,
      exactObservableEquivalenceNulls: exactEquivalenceSplit,
      knownAcquisitionValidationOnly: knownAcquisitionValidationSplit,
      invalidExcludedScenarioIds,
      nonUnknownExcludedScenarioIds,
      duplicateExcludedScenarioIds,
      duplicateExactEquivalenceScenarioIds,
      duplicateKnownAcquisitionValidationIds,
      modelExcludedMissingPinnedIds,
      modelExcludedUnexpectedIds,
      modelExactEquivalenceMissingPinnedIds,
      modelExactEquivalenceUnexpectedIds,
      modelKnownAcquisitionMissingPinnedIds,
      modelKnownAcquisitionUnexpectedIds,
      invalidPinnedStrictUnknownHoldoutIds,
      invalidPinnedAmbiguityStressIds,
      invalidPinnedFittedUnknownIds,
      invalidPinnedExactEquivalencePairs,
      fittedUnknownMissingPinnedIds,
      fittedUnknownUnexpectedIds,
      corpusFittedUnknownMissingPinnedIds,
      corpusFittedUnknownUnexpectedIds,
      invalidKnownAcquisitionValidationIds,
      unknownTruthKnownAcquisitionValidationIds,
      knownAcquisitionValidationNotExcludedIds,
      knownAcquisitionValidationFittedComponentIds,
      duplicateFittedComponentScenarioIds,
      missingFittedComponentScenarioIds,
      unexpectedFittedComponentScenarioIds,
      wrongClassFittedComponents,
      likelihoodComponentOwnershipMismatches,
      componentScenarioPopulationMismatches,
      componentAssignmentViewMismatches,
      componentArchitectureMismatches,
      frequencyAgileCensoringMatrixMismatches,
      duplicateModelClassIds,
      missingModelClassIds,
      unexpectedModelClassIds,
      invalidExactEquivalenceScenarioIds,
      nonUnknownExactEquivalenceScenarioIds,
      exactEquivalenceNotExcludedScenarioIds,
      exactEquivalenceWithoutDeclaredAlternativeIds,
      exactEquivalenceFittedComponentIds,
      ambiguousUnknownIncludedInComponentFitIds,
      excludedUnknownScenarioIds,
      fittedUnknownScenarioIds,
    },
    exactObservableEquivalencePairAudit: {
      numericalTolerance: EXACT_EQUIVALENCE_NUMERICAL_TOLERANCE,
      pairs: exactEquivalencePairAudit,
      discrepancyCount: exactEquivalenceDiscrepancyCount,
      discrepancies: exactEquivalenceDiscrepancies.slice(0, 100),
    },
  },
  productionRollingWindowValidation: {
    qualification: 'held-out-high-snr-spectrum-only-all-online-ready-representatives',
    cases: highSnrKnownRollingWindowCases.length,
    uniqueCases: uniqueRollingWindowCases,
    knownCoverage: rollingKnownCoverage,
    hierarchicalAccuracy: rollingKnownHierarchicalAccuracy,
    incompatibleNonUnknownCount: rollingIncompatibleNonUnknown.length,
    minimumScenarioKnownCoverage: minimumRollingScenarioCoverage,
    minimumScenarioHierarchicalAccuracy: minimumRollingScenarioHierarchicalAccuracy,
    acceptanceThresholds: {
      overallKnownCoverage: ROLLING_MINIMUM_OVERALL_KNOWN_COVERAGE,
      overallHierarchicalAccuracy: ROLLING_MINIMUM_OVERALL_HIERARCHICAL_ACCURACY,
      perScenarioKnownCoverage: ROLLING_MINIMUM_PER_SCENARIO_KNOWN_COVERAGE,
      perScenarioHierarchicalAccuracy: ROLLING_MINIMUM_PER_SCENARIO_HIERARCHICAL_ACCURACY,
    },
    missingScenarios: missingRollingKnownScenarioIds,
    byScenario: rollingByScenario,
    failures: highSnrKnownRollingWindowCases.filter((item) => !item.acceptedHierarchy).slice(0, 50),
    completeOnlineSpectrumAudit: {
      qualification: 'held-out-all-truths-all-snrs-all-online-ready-representatives',
      cases: rollingWindowCases.length,
      uniqueCases: uniqueOnlineSpectrumCases,
      unknownTruthCases: onlineUnknownSpectrumCases.length,
      unknownTruthFalseAcceptCount: onlineUnknownFalseAccepts.length,
      incompatibleNonUnknownCount: onlineSpectrumIncompatibleNonUnknown.length,
      singletonAllowedTruthProperScoreSamples: onlineSpectrumSingletonTruthFittedDomain.length,
      fittedTemplateLogLoss: onlineSpectrumFittedTemplateLogLoss,
      fittedTemplateMulticlassBrier: onlineSpectrumFittedTemplateMulticlassBrier,
      fittedTemplateExpectedCalibrationError: onlineSpectrumFittedTemplateExpectedCalibrationError,
      byTruth: counts(rollingWindowCases.map((item) => item.modelTruth)),
      failures: onlineSpectrumIncompatibleNonUnknown.slice(0, 50),
    },
    truthConditionedClassDomainDiagnostic: {
      qualification: 'secondary-diagnostic-not-primary-denominator',
      cases: truthClassDomainRollingWindowCases.length,
      knownCoverage: fraction(truthClassDomainRollingWindowCases, (item) => item.result !== 'unknown'),
      hierarchicalAccuracy: fraction(truthClassDomainRollingWindowCases, (item) => item.acceptedHierarchy),
    },
  },
  matrix: {
    attemptSamplingWorkerRuntimeSha256: modelAttemptSamplingWorkerRuntimeSha256,
    trainingRuntimeIdentity: modelTrainingRuntimeIdentity,
    scenarioSelection: diagnosticScenarioIdSet.size === 0
      ? { mode: 'full-corpus', scenarioIds: validationScenarios.map((scenario) => scenario.id) }
      : { mode: 'diagnostic-subset', scenarioIds: validationScenarios.map((scenario) => scenario.id) },
    nuisanceShiftSeeds: NUISANCE_SHIFT_SEEDS,
    snrDb: SNR_DB,
    rbwDivisors: RBW_DIVISORS,
    temporalSchedules: {
      consecutiveSpectrum: PINNED_VALIDATION_SPECTRUM_TEMPORAL_SCHEDULE,
      qualifiedEnvelope: PINNED_VALIDATION_QUALIFIED_ENVELOPE_TEMPORAL_SCHEDULE,
    },
    sourceClocks: PINNED_SIGNAL_LAB_PRODUCTION_ACQUISITION_REGIME.sourceClocks,
    runtimeBranchClockAudits: {
      consecutiveSpectrum: validationSpectrumClockAudit,
      qualifiedEnvelope: validationQualifiedEnvelopeClockAudit,
    },
    pairedNuisanceCells: admissionAttempts.length,
    runtimeBranchAttempts: {
      consecutiveSpectrum: validationSpectrumAcquisitionTraces.length,
      qualifiedEnvelope: validationQualifiedEnvelopeAcquisitionTraces.length,
    },
    heldOutSourceSpanAudit,
    observationOpportunityHorizons: {
      standard: STANDARD_OBSERVATION_OPPORTUNITIES,
      fullBand2g4: FULL_BAND_2G4_OBSERVATION_OPPORTUNITIES,
    },
    attemptsByObservationHorizon: counts(admissionAttempts.map((item) => String(item.observationHorizon))),
    classificationAdmissions: CLASSIFICATION_ADMISSIONS,
    sweepPoints: SWEEP_POINTS,
    sweepTimeSeconds: SWEEP_TIME_SECONDS,
    zeroSpanPoints: ZERO_SPAN_POINTS,
    zeroSpanSamplePeriodSeconds: ZERO_SPAN_SAMPLE_PERIOD_SECONDS,
    detectedPowerSynthesisFilterPolicy: PINNED_DETECTED_POWER_SYNTHESIS_FILTER_POLICY,
    frequencyAgileFixedTuneEnvelopeCensoringPolicy:
      PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORING_POLICY,
    detectionConfig: PRODUCTION_DETECTION_CONFIG,
    selectionPolicy: SELECTION_POLICY,
    captureTargetSelectionPolicy:
      PINNED_CAPTURE_TARGET_SELECTION_POLICY_ID,
    automaticDetectedPowerSelectionCondition:
      PINNED_DETECTED_POWER_SELECTION_CONDITION,
    representativeWeightingPolicy: REPRESENTATIVE_WEIGHTING_POLICY,
    representativeEligibilityPolicy: 'observation-only-hypothesis-domain-v5',
    samplingPartitionAudit: {
      valid: samplingPartitionsDisjoint,
      modelFittingSeeds,
      modelCalibrationSeeds,
      validationSeeds: NUISANCE_SHIFT_SEEDS,
      modelFittingRbwDivisors,
      modelCalibrationRbwDivisors,
      modelFittingAcquisitionRegimeIds,
      modelCalibrationAcquisitionRegimeIds,
      validationRbwDivisors: RBW_DIVISORS,
      fittingCalibrationSeedOverlap,
      validationFittingSeedOverlap,
      validationCalibrationSeedOverlap,
      validationFittingRbwOverlap,
      validationCalibrationRbwOverlap,
      validationTemporalPartitionDisjoint,
      validationTemporalScheduleIdOverlap,
      validationFitSpectrumSourceLookIndexOverlap,
      validationFitQualifiedEnvelopeSourceLookIndexOverlap,
      validationFitTemporalSourceLookIndexOverlap,
    },
    tailCalibrationAudit: {
      valid: tailCalibrationPolicyValid,
      pinnedScoreUnit: PINNED_TAIL_CALIBRATION_SCORE_UNIT,
      modelScoreUnit: BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.tailCalibrationScoreUnit,
      pinnedRepresentativeSelectionPolicy: PINNED_TAIL_CALIBRATION_SELECTION_POLICY,
      modelRepresentativeSelectionPolicy: BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.tailCalibrationRepresentativeSelectionPolicy,
      pinnedRepresentativeAggregationPolicy: PINNED_TAIL_CALIBRATION_AGGREGATION_POLICY,
      modelRepresentativeAggregationPolicy: BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.tailCalibrationRepresentativeAggregationPolicy,
      pinnedRuntimeInterpretationPolicy: PINNED_TAIL_CALIBRATION_RUNTIME_INTERPRETATION_POLICY,
      modelRuntimeInterpretationPolicy: BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.tailCalibrationRuntimeInterpretationPolicy,
      pinnedStatisticalInterpretation: PINNED_TAIL_CALIBRATION_STATISTICAL_INTERPRETATION,
      modelStatisticalInterpretation: BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.tailCalibrationStatisticalInterpretation,
      attemptCountsByScenarioByView: modelTailCalibrationAttemptCountsByView,
      missingScenarioIds: missingTailCalibrationScenarioIds,
      unexpectedScenarioIds: unexpectedTailCalibrationScenarioIds,
      invalidAttemptCounts: invalidTailCalibrationAttemptCounts,
      viewCountMismatches: tailCalibrationViewCountMismatches,
      matrixPinsValid: tailCalibrationMatrixPinsValid,
      productionAcquisitionRegimePinsValid,
      pinnedReleaseGateSourcePlanValid,
      pinnedSignalLabProductionAcquisitionRegime: PINNED_SIGNAL_LAB_PRODUCTION_ACQUISITION_REGIME,
      validatorOwnedMatrix: {
        snrDb: PINNED_TAIL_CALIBRATION_SNR_DB,
        rbwDivisors: PINNED_TAIL_CALIBRATION_RBW_DIVISORS,
        acquisitionRegimeIds: PINNED_TAIL_CALIBRATION_ACQUISITION_REGIME_IDS,
        seeds: PINNED_TAIL_CALIBRATION_SEEDS,
      },
      independentRecomputation: independentTailCalibrationAudit,
    },
  },
  admission: {
    attempted: admissionAttempts.length,
    everReady: everReadyAttempts.length,
    firstReady: firstReadyAttempts.length,
    admitted: firstReadyAttempts.length,
    captureConditionalClassificationSamples: cases.length,
    expectedCaptureConditionalClassificationSamples:
      physicalEnvelopeCaptureAttempts.length,
    uniqueCaptureConditionalClassificationSamples,
    causalEnvelopeSamples: qualifiedCausalEnvelopeSamples.length,
    expectedCausalEnvelopeSamples,
    uniqueCausalEnvelopeSamples,
    physicalDetectedPowerCaptures: physicalEnvelopeCaptureAttempts.length,
    physicalEnvelopeCaptures: qualifiedCausalEnvelopeSamples.length,
    detectedPowerCaptureOutcomes: {
      schemaVersion: 1,
      censoringPolicy:
        PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORING_POLICY,
      physicalDetectedPowerCaptureCount:
        physicalEnvelopeCaptureAttempts.length,
      receiptQualifiedPhysicalCaptureCount: cases.filter((item) =>
        item.detectedPowerCaptureReceiptSchemaVersion === 4
        && Boolean(item.physicalCaptureId)).length,
      qualifiedEnvelopeSampleCount: qualifiedCausalEnvelopeSamples.length,
      censoredDetectedPowerCaptureCount:
        censoredFrequencyAgileCaptureCases.length,
      censoredSpectrumClassificationCount:
        censoredFrequencyAgileCaptureCases.filter((item) =>
          item.views.length === 1
          && item.views[0] === 'scalar-spectrum').length,
      selectedEvidenceViews: counts(cases.map((item) =>
        observableModelView({ values: item.features }))),
      byProjectedMode: detectedPowerCaptureOutcomesByProjectedMode,
      byProjectionKind: detectedPowerCaptureOutcomesByProjectionKind,
      byScenario: detectedPowerCaptureOutcomesByScenario,
    },
    frequencyAgileEnvelopeCensoring: {
      policyId: PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORING_POLICY_ID,
      limitation:
        PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORING_LIMITATION,
      physicalCapturesCensored:
        censoredFrequencyAgileCaptureAttempts.length,
      spectrumOnlyClassifications:
        censoredFrequencyAgileCaptureCases.length,
      uncensoredFrequencyAgileEnvelopeSamples:
        qualifiedCausalEnvelopeSamples.filter((item) =>
          item.associationMode === 'frequency-agile-2g4-activity').length,
    },
    detectedPowerAcquisitionQualification: {
      required: PINNED_DETECTED_POWER_ACQUISITION_QUALIFICATION,
      modelDeclared:
        BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.detectedPowerAcquisitionQualification,
      automaticSelectionConditionRequired:
        PINNED_DETECTED_POWER_SELECTION_CONDITION,
      modelDeclaredSelectionCondition:
        BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.detectedPowerSelectionCondition,
      qualifiedEnvelopeSamples: qualifiedCausalEnvelopeSamples.length,
      unqualifiedEnvelopeSamples: unqualifiedCausalEnvelopeSamples.length,
      missingOrUnissuedReceiptEnvelopeSamples:
        missingOrUnissuedReceiptEnvelopeSamples.length,
      missingOrUnissuedReceiptEnvelopeFeatureAttempts:
        missingOrUnissuedReceiptEnvelopeFeatureAttempts.length,
    },
    unavailablePhysicalEnvelopeCaptures: unavailablePhysicalEnvelopeCaptureAttempts.length,
    unavailablePhysicalEnvelopeCaptureExamples: unavailablePhysicalEnvelopeCaptureAttempts.slice(0, 50),
    invalidCausalCaptureSemantics: invalidCausalCaptureSemantics.slice(0, 50),
    causalEnvelopeAvailabilityCells,
    trackerFirstReadyRepresentativeSamples: admissionAttempts.reduce(
      (sum, item) => sum + item.firstReadyRepresentativeCount,
      0,
    ),
    runtimeBranchClockAudits: {
      consecutiveSpectrum: validationSpectrumClockAudit,
      qualifiedEnvelope: validationQualifiedEnvelopeClockAudit,
    },
    envelopeFeatureUnavailableByCode: counts(admissionAttempts.flatMap((item) =>
      item.envelopeFeatureUnavailableCode === undefined ? [] : [item.envelopeFeatureUnavailableCode])),
    misses: admissionMisses.length,
    everReadyRate: fraction(admissionAttempts, (item) => item.everReady),
    firstReadyRate: fraction(admissionAttempts, (item) => item.admitted),
    admissionRate: fraction(admissionAttempts, (item) => item.admitted),
    missExamples: admissionMisses.slice(0, 50),
    highSnrMinimumDb: HIGH_SNR_MINIMUM_DB,
    highSnr: admissionSummary(admissionAttempts.filter((item) => item.snrDb >= HIGH_SNR_MINIMUM_DB)),
    scenariosWithoutHighSnrAdmission,
    expectedClassificationNonAdmissionScenarios: [...expectedClassificationNonAdmissionIds].sort(),
    expectedNonAdmissionScenariosWithAdmission,
    knownAcquisitionWrongAdmissionCount: knownAcquisitionWrongAdmissions.length,
    knownAcquisitionWrongAdmissionExamples: knownAcquisitionWrongAdmissions.slice(0, 50)
      .map(({ posterior: _posterior, features: _features, ...item }) => item),
    highSnrUniqueSeedCoverage: {
      snrDb: ADMISSION_SEED_COVERAGE_SNR_DB,
      validationSeeds: NUISANCE_SHIFT_SEEDS,
      ordinaryKnownRequiredCoverage: 1,
      bluetoothLeAdvertisingRequiredCoverage: BLE_ADVERTISING_MINIMUM_SEED_COVERAGE,
      byKnownScenario: admissionSeedCoverageByKnownScenario,
      failures: knownAdmissionSeedCoverageFailures,
    },
    byScenario: admissionByScenario,
    byRbwDivisor,
  },
  classificationConditionalOnAdmission: {
    samples: cases.length,
    identifiableFitEligibleSamples: fittedTemplateCases.length,
    singletonAllowedTruthProperScoreSamples: singletonAllowedTruthFittedTemplateCases.length,
    properScoreQualification: 'one-hot log loss, multiclass Brier score, and ECE include only fit-domain cases with exactly one declared allowed observable truth',
    identifiableFitEligibleKnownSamples: identifiableFitEligibleKnown.length,
    validationOnlyExcludedSamples: scenarioExcludedUnknown.length,
    componentFitEligibleSamples: cases.filter((item) => item.componentFitEligible).length,
    componentFitIneligibleSamples: cases.filter((item) => !item.componentFitEligible).length,
    componentFitIneligibleByScenario: counts(cases.filter((item) => !item.componentFitEligible).map((item) => item.scenario)),
    componentFitIneligibleByAssociationMode: counts(cases.filter((item) => !item.componentFitEligible).map((item) => item.associationMode)),
    hierarchicalAccuracy: fraction(cases, (item) => item.acceptedHierarchy),
    fittedTemplateTopLeafAccuracy: fraction(fittedTemplateCases, (item) => item.topLeaf === item.modelTruth),
    knownTopLeafAccuracy: fraction(identifiableFitEligibleKnown, (item) => item.topLeaf === item.modelTruth),
    knownCoverage: fraction(identifiableFitEligibleKnown, (item) => item.result !== 'unknown'),
    coveredKnownHierarchicalAccuracy: fraction(knownCovered, (item) => item.acceptedHierarchy),
    minimumKnownClassHierarchicalAccuracy,
    classwiseKnown,
    highSnrMinimumDb: HIGH_SNR_MINIMUM_DB,
    minimumHighSnrKnownClassHierarchicalAccuracy,
    classwiseKnownHighSnr,
    fittedUnknownTemplateRejectionRate: fraction(fittedUnknownTemplates, (item) => item.result === 'unknown'),
    fittedUnknownPosteriorAuroc,
    scenarioExcludedFromComponentFitScenarios: [...scenarioExcludedIds],
    knownAcquisitionValidationOnlyScenarios: [...knownAcquisitionValidationIds],
    scenarioExcludedUnknownSamples: scenarioExcludedUnknown.length,
    validationOnlyUnknownDecisionRate: fraction(scenarioExcludedUnknown, (item) => item.result === 'unknown'),
    scenarioExcludedStrictUnknownSamples: scenarioExcludedStrictUnknown.length,
    scenarioExcludedStrictUnknownRejectionRate: strictHoldoutRejectionRate,
    scenarioExcludedNonExactAmbiguousSamples: scenarioExcludedNonExactAmbiguous.length,
    exactEquivalenceSamples: scenarioExcludedExactEquivalence.length,
    exactEquivalenceCompatibleRate,
    exactEquivalenceDecisionCounts: counts(scenarioExcludedExactEquivalence.map((item) => item.result)),
    validationOnlyAllowedDecisionRate: fraction(scenarioExcludedUnknown, (item) => item.acceptedHierarchy),
    scenarioExcludedStrictTypicalityAuroc,
    falseAcceptedUnknownCount: falseAcceptedUnknown.length,
    anyFalseAcceptAttemptCount: falseAcceptedUnknownAttemptIds.length,
    anyFalseAcceptAttemptIds: falseAcceptedUnknownAttemptIds,
    falseAcceptedUnknownExamples: falseAcceptedUnknown.slice(0, 50),
    modelSupportRank: {
      identifiableFitEligibleKnown: numericSummary(identifiableFitEligibleKnown.map((item) => item.knownSupportRank)),
      scenarioExcludedUnknown: numericSummary(scenarioExcludedUnknown.map((item) => item.knownSupportRank)),
    },
    fittedTemplateLogLoss,
    fittedTemplateMulticlassBrier: fittedTemplateBrier,
    fittedTemplateExpectedCalibrationError: fittedTemplateEce,
    allSingletonAllowedTruthLogLossDiagnostic: {
      value: allSingletonAllowedTruthLogLossDiagnostic,
      samples: singletonAllowedTruthCases.length,
      qualification: 'one-hot diagnostic restricted to cases declaring exactly one allowed observable truth',
    },
    evidenceViews,
    bySnr,
    byRbwDivisor,
    association: {
      firstReadySelectionModes: counts(cases.map(
        (item) => item.rawCaptureTargetAssociationMode,
      )),
      soleEnvelopeTargetModes: counts(qualifiedCausalEnvelopeSamples.map(
        (item) => item.rawCaptureTargetAssociationMode,
      )),
      captureProjectionKinds: counts(cases.map(
        (item) => item.captureProjectionKind,
      )),
      rawCaptureTargetStates: counts(cases.map(
        (item) => item.rawCaptureTargetState,
      )),
      detectedPowerEvidenceDispositions: counts(cases.map(
        (item) => item.detectedPowerEvidenceDisposition,
      )),
      completeSpectrumOnline: {
        samples: spectrumOnlineAssociationSamples.length,
        uniqueSamples: new Set(spectrumOnlineAssociationKeys).size,
        duplicateKeys: duplicateSpectrumOnlineAssociationKeys,
        byMode: spectrumOnlineAssociationByMode,
      },
      everAttemptModes: counts(admissionAttempts.flatMap((item) => item.everAssociationModes)),
      finalAttemptModes: counts(admissionAttempts.flatMap((item) => item.finalAssociationModes)),
      byMode: associationByMode,
      soleEnvelopeByMode: soleEnvelopeAssociationByMode,
      byScenario: associationByScenario,
    },
    limitations: limitationCounts,
    confusion,
    failures: cases.filter((item) => !item.acceptedHierarchy).slice(0, 50).map(({ posterior: _posterior, features: _features, ...item }) => item),
  },
};
const conditional = report.classificationConditionalOnAdmission;
const nonFiniteReportNumbers = nonFiniteReportNumberPaths(report);
const acceptanceFailures = [
  nonFiniteReportNumbers.length !== 0
    ? `validator report contains non-finite numbers before serialization: ${nonFiniteReportNumbers.slice(0, 50).join(', ')}${nonFiniteReportNumbers.length > 50 ? ` (+${nonFiniteReportNumbers.length - 50} more)` : ''}`
    : undefined,
  diagnosticScenarioIdSet.size > 0 ? 'diagnostic scenario subset is never an acceptance run' : undefined,
  BAYESIAN_OBSERVABLE_MODEL.id !== 'bayesian-observable-equivalence-v9'
    ? `expected v9 model identity, observed ${BAYESIAN_OBSERVABLE_MODEL.id}`
    : undefined,
  BAYESIAN_OBSERVABLE_MODEL.calibrationId
      !== 'synthetic-independent-branch-view-matched-causal-acquisition-support-rank-detector-conditioned-physical-uncalibrated-v20'
    ? `expected v20 calibration identity, observed ${BAYESIAN_OBSERVABLE_MODEL.calibrationId}`
    : undefined,
  BAYESIAN_OBSERVABLE_MODEL.classModels.length !== 12 ? `expected 12 v9 model classes, observed ${BAYESIAN_OBSERVABLE_MODEL.classModels.length}` : undefined,
  BAYESIAN_OBSERVABLE_MODEL.sourceCommit !== PINNED_SIGNAL_LAB_COMMIT ? `model source commit ${BAYESIAN_OBSERVABLE_MODEL.sourceCommit} does not match pinned ${PINNED_SIGNAL_LAB_COMMIT}` : undefined,
  BAYESIAN_OBSERVABLE_MODEL.corpusVersion !== CLASSIFICATION_CORPUS_VERSION ? `model corpus version ${BAYESIAN_OBSERVABLE_MODEL.corpusVersion} does not match checked-out ${CLASSIFICATION_CORPUS_VERSION}` : undefined,
  JSON.stringify(BAYESIAN_OBSERVABLE_MODEL.corpusSourceManifest) !== JSON.stringify(checkedOutCorpusSourceManifest)
    ? 'model corpus source manifest does not match the validator-owned checked-out artifact set'
    : undefined,
  BAYESIAN_OBSERVABLE_MODEL.corpusSha256 !== checkedOutCorpusSha256 ? `model corpus SHA-256 ${BAYESIAN_OBSERVABLE_MODEL.corpusSha256} does not match checked-out ${checkedOutCorpusSha256}` : undefined,
  checkedInModelAssetSha256 !== BAYESIAN_OBSERVABLE_MODEL_SHA256 ? `model asset SHA-256 ${checkedInModelAssetSha256} does not match manifest ${BAYESIAN_OBSERVABLE_MODEL_SHA256}` : undefined,
  !/^[a-f0-9]{64}$/.test(modelAttemptSamplingWorkerRuntimeSha256)
    ? `attempt-sampling worker runtime SHA-256 is malformed: ${modelAttemptSamplingWorkerRuntimeSha256}`
    : undefined,
  !trainingRuntimeIdentityPinsValid
    ? `training runtime identity ${JSON.stringify(modelTrainingRuntimeIdentity)} does not match pinned ${JSON.stringify(PINNED_TRAINING_RUNTIME_IDENTITY)}`
    : undefined,
  admissionAttempts.length !== expectedAttempts ? `expected ${expectedAttempts} production-pipeline attempts, observed ${admissionAttempts.length}` : undefined,
  validationSpectrumAcquisitionTraces.length !== expectedAttempts
    ? `expected ${expectedAttempts} consecutive-spectrum branch attempts, observed ${validationSpectrumAcquisitionTraces.length}`
    : undefined,
  validationQualifiedEnvelopeAcquisitionTraces.length !== expectedAttempts
    ? `expected ${expectedAttempts} qualified-envelope branch attempts, observed ${validationQualifiedEnvelopeAcquisitionTraces.length}`
    : undefined,
  validationSpectrumClockAudit.violationCount !== 0
    ? `validation consecutive-spectrum branch has ${validationSpectrumClockAudit.violationCount} source-clock violations`
    : undefined,
  validationSpectrumClockAudit.maximumDetectedPowerCapturesPerAttempt !== 0
    ? `validation consecutive-spectrum branch consumed ${validationSpectrumClockAudit.maximumDetectedPowerCapturesPerAttempt} detected-power captures in one attempt`
    : undefined,
  validationQualifiedEnvelopeClockAudit.violationCount !== 0
    ? `validation qualified-envelope branch has ${validationQualifiedEnvelopeClockAudit.violationCount} source-clock violations`
    : undefined,
  !heldOutSourceSpanAudit.valid
    ? `held-out source clock moves declared signal geometry outside its admitted span for ${heldOutSourceSpanAudit.scenarios.filter((item) => !item.valid).map((item) => item.scenarioId).join(', ')}`
    : undefined,
  validationQualifiedEnvelopeClockAudit.maximumDetectedPowerCapturesPerAttempt > 1
    ? `validation qualified-envelope branch consumed ${validationQualifiedEnvelopeClockAudit.maximumDetectedPowerCapturesPerAttempt} detected-power captures in one attempt`
    : undefined,
  invalidCausalCaptureSemantics.length !== 0
    ? `${invalidCausalCaptureSemantics.length} validation attempts violated rank-0 integrated-excess runtime-admission capture semantics`
    : undefined,
  unavailablePhysicalEnvelopeCaptureAttempts.length !== 0
    ? `${unavailablePhysicalEnvelopeCaptureAttempts.length} physical detected-power captures produced unavailable envelope evidence`
    : undefined,
  invalidCensoredFrequencyAgileCaptureCases.length !== 0
    ? `${invalidCensoredFrequencyAgileCaptureCases.length} frequency-agile captures violated the fixed-tune spectrum-only censoring policy`
    : undefined,
  invalidUncensoredEnvelopeCases.length !== 0
    ? `${invalidUncensoredEnvelopeCases.length} uncensored envelope cases contradicted their receipt-qualified evidence binding`
    : undefined,
  censoredFrequencyAgileCaptureCases.length
      !== censoredFrequencyAgileCaptureAttempts.length
    ? `frequency-agile censored classifications ${censoredFrequencyAgileCaptureCases.length} do not reconcile to censored physical captures ${censoredFrequencyAgileCaptureAttempts.length}`
    : undefined,
  invalidBluetoothCaptureOutcomeScenarios.length !== 0
    ? `Bluetooth fixed-tune capture outcomes violate spectrum-only censoring for ${invalidBluetoothCaptureOutcomeScenarios.join(', ')}`
    : undefined,
  BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.detectedPowerAcquisitionQualification
    !== PINNED_DETECTED_POWER_ACQUISITION_QUALIFICATION
    ? 'model does not declare the pinned receipt-verified detected-power acquisition qualification'
    : undefined,
  BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.detectedPowerSelectionCondition
    !== PINNED_DETECTED_POWER_SELECTION_CONDITION
    ? 'model does not declare the pinned automatic rank-0 integrated-excess selection condition'
    : undefined,
  qualifiedCausalEnvelopeSamples.length !== expectedCausalEnvelopeSamples
    || qualifiedCausalEnvelopeSamples.length
      + censoredFrequencyAgileCaptureCases.length !== cases.length
    ? `qualified causal envelope samples ${qualifiedCausalEnvelopeSamples.length} plus censored spectrum-only captures ${censoredFrequencyAgileCaptureCases.length} do not reconcile to classified ${cases.length} and physical captures ${physicalEnvelopeCaptureAttempts.length}`
    : undefined,
  unqualifiedCausalEnvelopeSamples.length !== 0
    || missingOrUnissuedReceiptEnvelopeSamples.length !== 0
    || missingOrUnissuedReceiptEnvelopeFeatureAttempts.length !== 0
    ? `detected-power receipt boundary admitted unqualified=${unqualifiedCausalEnvelopeSamples.length}, missing-or-unissued=${missingOrUnissuedReceiptEnvelopeSamples.length}, missing-or-unissued-attempts=${missingOrUnissuedReceiptEnvelopeFeatureAttempts.length}`
    : undefined,
  cases.length !== physicalEnvelopeCaptureAttempts.length ? `classified ${cases.length} capture-conditional representatives, expected ${physicalEnvelopeCaptureAttempts.length}` : undefined,
  uniqueCaptureConditionalClassificationSamples !== cases.length ? `capture-conditional classification contains ${cases.length - uniqueCaptureConditionalClassificationSamples} duplicate attempt/representative samples` : undefined,
  uniqueCausalEnvelopeSamples !== qualifiedCausalEnvelopeSamples.length ? `causal envelope classification contains ${qualifiedCausalEnvelopeSamples.length - uniqueCausalEnvelopeSamples} duplicate attempt/representative samples` : undefined,
  cases.length === 0 ? 'production detector/tracker admitted no validation cases' : undefined,
  associationModesWithoutCoverage.length !== 0
    ? `production validation has no complete spectrum-online scenario coverage for association modes: ${associationModesWithoutCoverage.join(', ')}`
    : undefined,
  duplicateSpectrumOnlineAssociationKeys.length !== 0
    ? `complete spectrum-online association audit contains ${duplicateSpectrumOnlineAssociationKeys.length} duplicate attempt/opportunity/representative keys`
    : undefined,
  highSnrKnownRollingWindowCases.length === 0 ? 'production rolling-window validation admitted no current-qualified known cases' : undefined,
  duplicateRollingWindowKeys.length !== 0
    ? `production rolling-window validation contains ${highSnrKnownRollingWindowCases.length - uniqueRollingWindowCases} extra samples across ${duplicateRollingWindowKeys.length} duplicate attempt/opportunity/representative keys`
    : undefined,
  duplicateOnlineSpectrumKeys.length !== 0
    ? `complete online spectrum validation contains ${rollingWindowCases.length - uniqueOnlineSpectrumCases} extra samples across ${duplicateOnlineSpectrumKeys.length} duplicate attempt/opportunity/representative keys`
    : undefined,
  missingRollingKnownScenarioIds.length !== 0
    ? `production rolling-window validation is missing fitted known scenarios: ${missingRollingKnownScenarioIds.join(', ')}`
    : undefined,
  rollingKnownCoverage < ROLLING_MINIMUM_OVERALL_KNOWN_COVERAGE
    ? `production rolling-window known coverage ${rollingKnownCoverage} < ${ROLLING_MINIMUM_OVERALL_KNOWN_COVERAGE}`
    : undefined,
  rollingKnownHierarchicalAccuracy < ROLLING_MINIMUM_OVERALL_HIERARCHICAL_ACCURACY
    ? `production rolling-window hierarchical accuracy ${rollingKnownHierarchicalAccuracy} < ${ROLLING_MINIMUM_OVERALL_HIERARCHICAL_ACCURACY}`
    : undefined,
  rollingIncompatibleNonUnknown.length !== 0
    ? `production rolling-window validation emitted ${rollingIncompatibleNonUnknown.length} incompatible non-unknown decisions`
    : undefined,
  onlineSpectrumIncompatibleNonUnknown.length !== 0
    ? `complete online spectrum validation emitted ${onlineSpectrumIncompatibleNonUnknown.length} incompatible non-unknown decisions across all truths and SNRs`
    : undefined,
  onlineUnknownFalseAccepts.length !== 0
    ? `complete online spectrum validation falsely accepted ${onlineUnknownFalseAccepts.length} unknown-truth representatives`
    : undefined,
  onlineSpectrumSingletonTruthFittedDomain.length === 0
    ? 'complete online spectrum validation has no singleton-allowed-truth fit-domain cases for proper scores'
    : undefined,
  !Number.isFinite(onlineSpectrumFittedTemplateLogLoss) || onlineSpectrumFittedTemplateLogLoss > 0.5
    ? `complete online spectrum fitted-template log loss ${onlineSpectrumFittedTemplateLogLoss} > 0.5 or non-finite`
    : undefined,
  !Number.isFinite(onlineSpectrumFittedTemplateMulticlassBrier) || onlineSpectrumFittedTemplateMulticlassBrier > 0.2
    ? `complete online spectrum fitted-template Brier score ${onlineSpectrumFittedTemplateMulticlassBrier} > 0.2 or non-finite`
    : undefined,
  !Number.isFinite(onlineSpectrumFittedTemplateExpectedCalibrationError) || onlineSpectrumFittedTemplateExpectedCalibrationError > 0.1
    ? `complete online spectrum fitted-template ECE ${onlineSpectrumFittedTemplateExpectedCalibrationError} > 0.1 or non-finite`
    : undefined,
  minimumRollingScenarioCoverage < ROLLING_MINIMUM_PER_SCENARIO_KNOWN_COVERAGE
    ? `minimum per-scenario production rolling-window known coverage ${minimumRollingScenarioCoverage} < ${ROLLING_MINIMUM_PER_SCENARIO_KNOWN_COVERAGE}`
    : undefined,
  minimumRollingScenarioHierarchicalAccuracy < ROLLING_MINIMUM_PER_SCENARIO_HIERARCHICAL_ACCURACY
    ? `minimum per-scenario production rolling-window hierarchical accuracy ${minimumRollingScenarioHierarchicalAccuracy} < ${ROLLING_MINIMUM_PER_SCENARIO_HIERARCHICAL_ACCURACY}`
    : undefined,
  !samplingPartitionsDisjoint
    ? `sampling partitions overlap or lack metadata (fit/cal seeds=${fittingCalibrationSeedOverlap.join(',') || 'none'}; validation/fit seeds=${validationFittingSeedOverlap.join(',') || 'none'}; validation/cal seeds=${validationCalibrationSeedOverlap.join(',') || 'none'}; validation/fit RBWs=${validationFittingRbwOverlap.join(',') || 'none'}; validation/cal RBWs=${validationCalibrationRbwOverlap.join(',') || 'none'}; validation/fit temporal IDs=${validationTemporalScheduleIdOverlap.join(',') || 'none'}; validation/fit temporal source indices=${validationFitTemporalSourceLookIndexOverlap.join(',') || 'none'}; calibration-seed-count=${modelCalibrationSeeds.length}; calibration-RBW-count=${modelCalibrationRbwDivisors.length}; production-regime-pins=${productionAcquisitionRegimePinsValid})`
    : undefined,
  !tailCalibrationPolicyValid
    ? `tail-calibration policy/manifest is invalid (score-unit=${BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.tailCalibrationScoreUnit ?? 'missing'}; selection=${BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.tailCalibrationRepresentativeSelectionPolicy ?? 'missing'}; aggregation=${BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.tailCalibrationRepresentativeAggregationPolicy ?? 'missing'}; runtime-interpretation=${BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.tailCalibrationRuntimeInterpretationPolicy ?? 'missing'}; statistical-interpretation=${BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.tailCalibrationStatisticalInterpretation ?? 'missing'}; matrix-pins=${tailCalibrationMatrixPinsValid}; production-regime-pins=${productionAcquisitionRegimePinsValid}; missing-scenarios=${missingTailCalibrationScenarioIds.join(',') || 'none'}; unexpected-scenarios=${unexpectedTailCalibrationScenarioIds.join(',') || 'none'}; invalid-counts=${invalidTailCalibrationAttemptCounts.map((item) => `${item.scenarioId}/${item.view}:${item.count ?? 'missing'}`).join(',') || 'none'}; view-count-mismatches=${tailCalibrationViewCountMismatches.map((item) => `${item.classId}/${item.view}:${item.observed}/${item.expected}`).join(',') || 'none'})`
    : undefined,
  !independentTailCalibrationAudit.valid
    ? `independent tail-calibration recomputation failed (attempt-count-mismatches=${independentTailCalibrationAudit.attemptCountMismatches.length}; score-mismatches=${independentTailCalibrationAudit.scoreComparisons.filter((item) => item.expectedCount !== item.observedCount || item.maximumAbsoluteDifference > independentTailCalibrationAudit.scoreTolerance).length}; late-minima=${independentTailCalibrationAudit.lateMinimumCount}; aggregation-regression=${independentTailCalibrationAudit.aggregationRegression.passed})`
    : undefined,
  !priorSensitivityAudit.valid
    ? `engineering-prior sensitivity failed (model-prior-pin=${priorSensitivityAudit.modelPriorMatchesPinned}; baseline-mismatches=${priorSensitivityAudit.baselineDecisionMismatchCount}; failing-variants=${priorSensitivityAudit.variants.filter((variant) => !variant.passed).map((variant) => variant.id).join(',') || 'none'})`
    : undefined,
  !completeOnlineSpectrumPriorSensitivityAudit.valid
    ? `complete-online spectrum engineering-prior sensitivity failed (model-prior-pin=${completeOnlineSpectrumPriorSensitivityAudit.modelPriorMatchesPinned}; baseline-mismatches=${completeOnlineSpectrumPriorSensitivityAudit.baselineDecisionMismatchCount}; failing-variants=${completeOnlineSpectrumPriorSensitivityAudit.variants.filter((variant) => !variant.passed).map((variant) => variant.id).join(',') || 'none'})`
    : undefined,
  !manifestSplitValid ? `model manifest split is invalid (missing-pinned-exclusions=${modelExcludedMissingPinnedIds.join(',') || 'none'}; unexpected-model-exclusions=${modelExcludedUnexpectedIds.join(',') || 'none'}; missing-pinned-exact=${modelExactEquivalenceMissingPinnedIds.join(',') || 'none'}; unexpected-model-exact=${modelExactEquivalenceUnexpectedIds.join(',') || 'none'}; missing-pinned-known-acquisition=${modelKnownAcquisitionMissingPinnedIds.join(',') || 'none'}; unexpected-model-known-acquisition=${modelKnownAcquisitionUnexpectedIds.join(',') || 'none'}; missing-fitted-unknown=${fittedUnknownMissingPinnedIds.join(',') || 'none'}; unexpected-fitted-unknown=${fittedUnknownUnexpectedIds.join(',') || 'none'}; missing-exclusions=${invalidExcludedScenarioIds.join(',') || 'none'}; unexpected-non-unknown-exclusions=${nonUnknownExcludedScenarioIds.join(',') || 'none'}; duplicate-exclusions=${duplicateExcludedScenarioIds.join(',') || 'none'}; invalid-known-acquisition=${invalidKnownAcquisitionValidationIds.join(',') || 'none'}; unknown-truth-known-acquisition=${unknownTruthKnownAcquisitionValidationIds.join(',') || 'none'}; known-acquisition-not-excluded=${knownAcquisitionValidationNotExcludedIds.join(',') || 'none'}; fitted-known-acquisition=${knownAcquisitionValidationFittedComponentIds.join(',') || 'none'}; missing-exact=${invalidExactEquivalenceScenarioIds.join(',') || 'none'}; non-unknown-exact=${nonUnknownExactEquivalenceScenarioIds.join(',') || 'none'}; exact-not-excluded=${exactEquivalenceNotExcludedScenarioIds.join(',') || 'none'}; exact-without-alternative=${exactEquivalenceWithoutDeclaredAlternativeIds.join(',') || 'none'}; fitted-exact-components=${exactEquivalenceFittedComponentIds.join(',') || 'none'}; fitted-ambiguous-unknown=${ambiguousUnknownIncludedInComponentFitIds.join(',') || 'none'}; fitted-unknown=${fittedUnknownScenarioIds.length}; excluded-unknown=${excludedUnknownScenarioIds.length})` : undefined,
  !manifestSplitValid ? `component assignment audit (duplicate=${duplicateFittedComponentScenarioIds.join(',') || 'none'}; missing=${missingFittedComponentScenarioIds.join(',') || 'none'}; unexpected=${unexpectedFittedComponentScenarioIds.join(',') || 'none'}; wrong-class=${wrongClassFittedComponents.map((item) => `${item.scenarioId}:${item.classId}->${item.expectedClassId}`).join(',') || 'none'}; ownership=${likelihoodComponentOwnershipMismatches.join(',') || 'none'}; scenario-view-mismatches=${componentScenarioPopulationMismatches.join(',') || 'none'}; assignment-view-mismatches=${componentAssignmentViewMismatches.join(',') || 'none'}; architecture=${componentArchitectureMismatches.join(',') || 'none'}; frequency-agile-censoring=${frequencyAgileCensoringMatrixMismatches.join(',') || 'none'}; duplicate-classes=${duplicateModelClassIds.join(',') || 'none'}; missing-classes=${missingModelClassIds.join(',') || 'none'}; unexpected-classes=${unexpectedModelClassIds.join(',') || 'none'})` : undefined,
  exactEquivalenceDiscrepancyCount !== 0 ? `${exactEquivalenceDiscrepancyCount} exact-equivalence paired nuisance checks differ` : undefined,
  knownAdmissionSeedCoverageFailures.length ? `${knownAdmissionSeedCoverageFailures.length} per-scenario/per-SNR known admission seed-coverage cells failed` : undefined,
  expectedNonAdmissionScenariosWithAdmission.length
    ? `expected-non-admission policy is stale: admissions observed for ${expectedNonAdmissionScenariosWithAdmission.join(', ')}` : undefined,
  knownAcquisitionWrongAdmissions.length ? `${knownAcquisitionWrongAdmissions.length} admitted known-acquisition-validation cases had incompatible decisions` : undefined,
  singletonAllowedTruthFittedTemplateCases.length === 0 ? 'no singleton-allowed-truth fitted cases available for proper-score diagnostics' : undefined,
  fittedUnknownTemplates.length === 0 ? 'no fitted unknown-template cases reached classification admission' : undefined,
  scenarioExcludedUnknown.length === 0 ? 'no scenario-excluded unknown cases reached classification admission' : undefined,
  scenarioExcludedStrictUnknown.length === 0 ? 'no strict unknown holdout cases reached classification admission' : undefined,
  strictHoldoutRejectionRate < 1 ? `strict unknown holdout rejection rate ${strictHoldoutRejectionRate} < 1` : undefined,
  scenarioExcludedExactEquivalence.length === 0 ? 'no exact observable-equivalence null cases reached classification admission' : undefined,
  exactEquivalenceCompatibleRate < 1 ? `exact observable-equivalence compatibility ${exactEquivalenceCompatibleRate} < 1` : undefined,
  falseAcceptedUnknown.length !== 0 ? `false-accepted ${falseAcceptedUnknown.length} admitted unknown scenarios` : undefined,
  falseAcceptedUnknownAttemptIds.length !== 0 ? `${falseAcceptedUnknownAttemptIds.length} attempts had at least one false-accepted unknown first-ready representative` : undefined,
  !Number.isFinite(conditional.hierarchicalAccuracy) || conditional.hierarchicalAccuracy < 0.95
    ? `admission-conditional hierarchical accuracy ${conditional.hierarchicalAccuracy} < 0.95 or non-finite` : undefined,
  !Number.isFinite(conditional.knownTopLeafAccuracy) || conditional.knownTopLeafAccuracy < 0.85
    ? `admission-conditional known top-leaf accuracy ${conditional.knownTopLeafAccuracy} < 0.85 or non-finite` : undefined,
  !Number.isFinite(conditional.knownCoverage) || conditional.knownCoverage < 0.95
    ? `admission-conditional known coverage ${conditional.knownCoverage} < 0.95 or non-finite` : undefined,
  !Number.isFinite(conditional.minimumHighSnrKnownClassHierarchicalAccuracy)
    || conditional.minimumHighSnrKnownClassHierarchicalAccuracy < 0.9
    ? `minimum >=${HIGH_SNR_MINIMUM_DB} dB known-class hierarchical accuracy ${conditional.minimumHighSnrKnownClassHierarchicalAccuracy} < 0.9 or non-finite` : undefined,
  !Number.isFinite(conditional.fittedTemplateLogLoss) || conditional.fittedTemplateLogLoss > 0.5
    ? `fitted-template log loss ${conditional.fittedTemplateLogLoss} > 0.5 or non-finite` : undefined,
  !Number.isFinite(conditional.fittedTemplateMulticlassBrier) || conditional.fittedTemplateMulticlassBrier > 0.2
    ? `fitted-template Brier score ${conditional.fittedTemplateMulticlassBrier} > 0.2 or non-finite` : undefined,
  !Number.isFinite(conditional.fittedTemplateExpectedCalibrationError) || conditional.fittedTemplateExpectedCalibrationError > 0.1
    ? `fitted-template ECE ${conditional.fittedTemplateExpectedCalibrationError} > 0.1 or non-finite` : undefined,
  !Number.isFinite(conditional.fittedUnknownPosteriorAuroc) || conditional.fittedUnknownPosteriorAuroc < 0.9
    ? `fitted-unknown posterior AUROC ${conditional.fittedUnknownPosteriorAuroc} < 0.9 or non-finite` : undefined,
  !Number.isFinite(conditional.scenarioExcludedStrictTypicalityAuroc) || conditional.scenarioExcludedStrictTypicalityAuroc < 0.9
    ? `strict scenario-excluded support AUROC ${conditional.scenarioExcludedStrictTypicalityAuroc} < 0.9 or non-finite` : undefined,
  ...Object.entries(conditional.evidenceViews).flatMap(([view, metrics]) => {
    const expectedViewSamples = view === 'envelope-untimed'
      ? qualifiedCausalEnvelopeSamples.length
      : cases.length;
    return [
    metrics.admittedSamples !== expectedViewSamples ? `${view} expected ${expectedViewSamples} admission-conditional cases, observed ${metrics.admittedSamples}` : undefined,
    metrics.falseAcceptedUnknownCount !== 0 ? `${view} false-accepted ${metrics.falseAcceptedUnknownCount} admitted unknown scenarios` : undefined,
    metrics.anyFalseAcceptAttemptCount !== 0 ? `${view} has ${metrics.anyFalseAcceptAttemptCount} attempts with at least one false-accepted unknown first-ready representative` : undefined,
    metrics.exactEquivalenceSamples === 0 ? `${view} has no admitted exact observable-equivalence null cases` : undefined,
    metrics.exactEquivalenceCompatibleRate < 1 ? `${view} exact observable-equivalence compatibility ${metrics.exactEquivalenceCompatibleRate} < 1` : undefined,
    metrics.strictHoldoutSamples === 0 ? `${view} has no admitted strict unknown holdout cases` : undefined,
    metrics.strictHoldoutRejectionRate < 1 ? `${view} strict unknown holdout rejection rate ${metrics.strictHoldoutRejectionRate} < 1` : undefined,
    metrics.knownCoverage < 0.8 ? `${view} known coverage ${metrics.knownCoverage} < 0.8` : undefined,
    metrics.coveredKnownHierarchicalAccuracy < 0.9 ? `${view} covered-known hierarchical accuracy ${metrics.coveredKnownHierarchicalAccuracy} < 0.9` : undefined,
    metrics.singletonAllowedTruthProperScoreSamples === 0
      ? `${view} has no singleton-allowed-truth fit-domain cases for proper scores` : undefined,
    !Number.isFinite(metrics.fittedTemplateLogLoss) || metrics.fittedTemplateLogLoss > 0.5
      ? `${view} fitted-template log loss ${metrics.fittedTemplateLogLoss} > 0.5 or non-finite` : undefined,
    !Number.isFinite(metrics.fittedTemplateMulticlassBrier) || metrics.fittedTemplateMulticlassBrier > 0.2
      ? `${view} fitted-template Brier score ${metrics.fittedTemplateMulticlassBrier} > 0.2 or non-finite` : undefined,
    !Number.isFinite(metrics.fittedTemplateExpectedCalibrationError) || metrics.fittedTemplateExpectedCalibrationError > 0.1
      ? `${view} fitted-template ECE ${metrics.fittedTemplateExpectedCalibrationError} > 0.1 or non-finite` : undefined,
    !Number.isFinite(metrics.scenarioExcludedStrictSupportAuroc) || metrics.scenarioExcludedStrictSupportAuroc < 0.9
      ? `${view} strict scenario-excluded support AUROC ${metrics.scenarioExcludedStrictSupportAuroc} < 0.9 or non-finite` : undefined,
  ];
  }),
].filter((value): value is string => value !== undefined);
const validationAcceptance = {
  schemaVersion: 1,
  status: acceptanceFailures.length === 0 ? 'passed' : 'failed',
  acceptancePolicyId: VALIDATION_ACCEPTANCE_POLICY_ID,
  scope: diagnosticScenarioIdSet.size === 0 ? 'full-corpus' : 'diagnostic-subset',
  failureCount: acceptanceFailures.length,
  modelAssetSha256: checkedInModelAssetSha256,
  attemptSamplingWorkerRuntimeSha256: modelAttemptSamplingWorkerRuntimeSha256,
  trainingRuntimeIdentity: modelTrainingRuntimeIdentity,
  modelId: BAYESIAN_OBSERVABLE_MODEL.id,
  sourceCommit: BAYESIAN_OBSERVABLE_MODEL.sourceCommit,
  corpusVersion: BAYESIAN_OBSERVABLE_MODEL.corpusVersion,
  corpusSha256: BAYESIAN_OBSERVABLE_MODEL.corpusSha256,
  preprocessing: BAYESIAN_OBSERVABLE_MODEL.preprocessing,
  priorId: BAYESIAN_OBSERVABLE_MODEL.priorId,
  calibrationId: BAYESIAN_OBSERVABLE_MODEL.calibrationId,
  decisionPolicyId: BAYESIAN_WAVEFORM_MODEL.decisionPolicyId,
  evidenceSha256: sha256Canonical(report),
  failures: acceptanceFailures,
} as const;
const publicationReport = {
  ...report,
  validationAcceptance,
} as const;
const publishedPath = acceptanceFailures.length === 0 ? REPORT_PATH : FAILED_REPORT_PATH;
const temporaryPublishedPath = acceptanceFailures.length === 0 ? REPORT_TEMP_PATH : FAILED_REPORT_TEMP_PATH;
writeFileSync(temporaryPublishedPath, `${JSON.stringify(publicationReport, null, 2)}\n`);
renameSync(temporaryPublishedPath, publishedPath);
validationPublicationCommitted = true;
console.log(JSON.stringify({ reportPath: publishedPath, ...publicationReport }, null, 2));
if (acceptanceFailures.length) {
  console.error(`Synthetic observable-class development regression failed:\n- ${acceptanceFailures.join('\n- ')}`);
  process.exitCode = 1;
}

function publishUnexpectedValidationFailure(reason: unknown): void {
  const error = reason instanceof Error ? reason : new Error(String(reason));
  process.exitCode = 1;
  if (validationPublicationCommitted) {
    console.error(`Observable classifier validation failed after publishing its terminal report: ${error.stack ?? error.message}`);
    return;
  }
  const failure = {
    schemaVersion: 1,
    validationAcceptance: {
      schemaVersion: 1,
      status: 'failed',
      acceptancePolicyId: VALIDATION_ACCEPTANCE_POLICY_ID,
      scope: 'preflight-or-unexpected-failure',
      failureCount: 1,
      failures: [`Unexpected validator failure: ${error.message}`],
    },
    unexpectedFailure: {
      name: error.name,
      message: error.message,
      ...(error.stack ? { stack: error.stack } : {}),
    },
  } as const;
  try {
    for (const path of [REPORT_PATH, REPORT_TEMP_PATH, FAILED_REPORT_PATH, FAILED_REPORT_TEMP_PATH]) {
      rmSync(path, { force: true });
    }
    writeFileSync(FAILED_REPORT_TEMP_PATH, `${JSON.stringify(failure, null, 2)}\n`);
    renameSync(FAILED_REPORT_TEMP_PATH, FAILED_REPORT_PATH);
    console.error(`Observable classifier validation failed before acceptance publication; diagnostic: ${FAILED_REPORT_PATH}\n${error.stack ?? error.message}`);
  } catch (publicationError) {
    const diagnostic = publicationError instanceof Error ? publicationError.stack ?? publicationError.message : String(publicationError);
    console.error(`Observable classifier validation failed and its diagnostic report could not be published: ${diagnostic}\nOriginal failure: ${error.stack ?? error.message}`);
  }
}

function auditExactEquivalencePairs(
  attempts: readonly AdmissionAttempt[],
  validationCases: readonly ValidationCase[],
  viewCases: readonly EvidenceViewCase[],
  onlineSpectrumCases: readonly RollingWindowCase[],
): ExactEquivalencePairAudit[] {
  return PINNED_EXACT_OBSERVABLE_EQUIVALENCE_PAIRS.map(({ referenceScenarioId, nullScenarioId }) => {
    const pair = `${referenceScenarioId}<=>${nullScenarioId}`;
    const discrepancies: ExactEquivalenceDiscrepancy[] = [];
    let nuisanceCells = 0;
    let matchedAdmissionCells = 0;
    let matchedRepresentativePairs = 0;
    let matchedEvidenceViewPairs = 0;
    let matchedOnlineSpectrumPairs = 0;
    const add = (
      nuisanceCell: string,
      field: string,
      reference: unknown,
      nullValue: unknown,
      representativeIndex?: number,
      view?: EvidenceViewCase['view'] | 'spectrum-online',
    ) => discrepancies.push({
      pair,
      nuisanceCell,
      ...(representativeIndex === undefined ? {} : { representativeIndex }),
      ...(view === undefined ? {} : { view }),
      field,
      reference,
      null: nullValue,
    });

    for (const snrDb of SNR_DB) for (const rbwDivisor of RBW_DIVISORS) for (const seed of NUISANCE_SHIFT_SEEDS) {
      nuisanceCells += 1;
      const nuisanceCell = `snr=${snrDb}:rbw=${rbwDivisor}:seed=${seed}`;
      const referenceAttempt = attempts.find((item) => item.scenario === referenceScenarioId
        && item.snrDb === snrDb && item.rbwDivisor === rbwDivisor && item.seed === seed);
      const nullAttempt = attempts.find((item) => item.scenario === nullScenarioId
        && item.snrDb === snrDb && item.rbwDivisor === rbwDivisor && item.seed === seed);
      if (!referenceAttempt || !nullAttempt) {
        add(nuisanceCell, 'admission-attempt-present', referenceAttempt !== undefined, nullAttempt !== undefined);
        continue;
      }
      const admissionFields: readonly (keyof AdmissionAttempt)[] = [
        'observationHorizon',
        'everReady',
        'admitted',
        'everReadyRepresentativeCount',
        'firstReadyRepresentativeCount',
        'provenanceUnavailableWindowCount',
        'detectedPowerCaptureCount',
        'envelopeFeatureAvailable',
        'envelopeFeatureUnavailableCode',
        'finalReadyRepresentativeCount',
        'finalActiveRepresentativeCount',
        'selectedTrackAdmissions',
        'maximumActiveAdmissions',
        'maximumLocalTrackAdmissions',
        'firstReadyOpportunity',
        'regularAssociationsObserved',
        'agileAssociationsObserved',
        'regularAssociationExpirations',
      ];
      let admissionMatches = true;
      for (const field of admissionFields) {
        if (!equivalentValue(referenceAttempt[field], nullAttempt[field])) {
          admissionMatches = false;
          add(nuisanceCell, `admission.${field}`, referenceAttempt[field], nullAttempt[field]);
        }
      }
      for (const field of ['everAssociationModes', 'finalAssociationModes'] as const) {
        const referenceValue = [...referenceAttempt[field]].sort().join('|');
        const nullValue = [...nullAttempt[field]].sort().join('|');
        if (referenceValue !== nullValue) {
          admissionMatches = false;
          add(nuisanceCell, `admission.${field}`, referenceValue, nullValue);
        }
      }
      if (admissionMatches) matchedAdmissionCells += 1;

      const referenceCases = validationCases
        .filter((item) => item.scenario === referenceScenarioId && item.snrDb === snrDb && item.rbwDivisor === rbwDivisor && item.seed === seed)
        .sort(compareValidationCasesForPairing);
      const nullCases = validationCases
        .filter((item) => item.scenario === nullScenarioId && item.snrDb === snrDb && item.rbwDivisor === rbwDivisor && item.seed === seed)
        .sort(compareValidationCasesForPairing);
      if (referenceCases.length !== nullCases.length) add(nuisanceCell, 'representative-count', referenceCases.length, nullCases.length);
      for (let index = 0; index < Math.min(referenceCases.length, nullCases.length); index++) {
        const referenceCase = referenceCases[index]!;
        const nullCase = nullCases[index]!;
        matchedRepresentativePairs += 1;
        compareExactCase(referenceCase, nullCase, nuisanceCell, index, add);
      }

      for (const view of ['spectrum-only', 'envelope-untimed'] as const) {
        const referenceViews = viewCases
          .filter((item) => item.scenario === referenceScenarioId && item.view === view
            && item.attemptId === validationAttemptId(referenceScenarioId, snrDb, rbwDivisor, seed))
          .sort(compareEvidenceViewCasesForPairing);
        const nullViews = viewCases
          .filter((item) => item.scenario === nullScenarioId && item.view === view
            && item.attemptId === validationAttemptId(nullScenarioId, snrDb, rbwDivisor, seed))
          .sort(compareEvidenceViewCasesForPairing);
        if (referenceViews.length !== nullViews.length) add(nuisanceCell, 'evidence-view-count', referenceViews.length, nullViews.length, undefined, view);
        for (let index = 0; index < Math.min(referenceViews.length, nullViews.length); index++) {
          matchedEvidenceViewPairs += 1;
          compareExactEvidenceView(referenceViews[index]!, nullViews[index]!, nuisanceCell, index, view, add);
        }
      }

      const referenceOnline = onlineSpectrumCases
        .filter((item) => item.scenario === referenceScenarioId
          && item.snrDb === snrDb && item.rbwDivisor === rbwDivisor && item.seed === seed)
        .sort(compareOnlineSpectrumCasesForPairing);
      const nullOnline = onlineSpectrumCases
        .filter((item) => item.scenario === nullScenarioId
          && item.snrDb === snrDb && item.rbwDivisor === rbwDivisor && item.seed === seed)
        .sort(compareOnlineSpectrumCasesForPairing);
      if (referenceOnline.length !== nullOnline.length) {
        add(nuisanceCell, 'online-spectrum-count', referenceOnline.length, nullOnline.length, undefined, 'spectrum-online');
      }
      for (let index = 0; index < Math.min(referenceOnline.length, nullOnline.length); index++) {
        matchedOnlineSpectrumPairs += 1;
        compareExactOnlineSpectrumCase(referenceOnline[index]!, nullOnline[index]!, nuisanceCell, index, add);
      }
    }
    if (matchedRepresentativePairs === 0) add('all', 'matched-representative-pairs', '>0', matchedRepresentativePairs);
    if (matchedEvidenceViewPairs === 0) add('all', 'matched-evidence-view-pairs', '>0', matchedEvidenceViewPairs);
    if (matchedOnlineSpectrumPairs === 0) add('all', 'matched-online-spectrum-pairs', '>0', matchedOnlineSpectrumPairs);
    return {
      pair,
      referenceScenarioId,
      nullScenarioId,
      nuisanceCells,
      matchedAdmissionCells,
      matchedRepresentativePairs,
      matchedEvidenceViewPairs,
      matchedOnlineSpectrumPairs,
      discrepancyCount: discrepancies.length,
      discrepancies: discrepancies.slice(0, 50),
    };
  });
}

function compareExactOnlineSpectrumCase(
  referenceCase: RollingWindowCase,
  nullCase: RollingWindowCase,
  nuisanceCell: string,
  representativeIndex: number,
  add: (
    nuisanceCell: string,
    field: string,
    reference: unknown,
    nullValue: unknown,
    representativeIndex?: number,
    view?: EvidenceViewCase['view'] | 'spectrum-online',
  ) => void,
): void {
  for (const field of [
    'readyOpportunity',
    'result',
    'topLeaf',
    'topLeafPosterior',
    'measuredBandwidthHz',
    'knownSupportRank',
    'associationMode',
  ] as const) {
    if (!equivalentValue(referenceCase[field], nullCase[field])) {
      add(nuisanceCell, `online-spectrum.${field}`, referenceCase[field], nullCase[field], representativeIndex, 'spectrum-online');
    }
  }
  compareNumericRecords(referenceCase.features, nullCase.features, 'online-spectrum.features', nuisanceCell, representativeIndex,
    (cell, field, reference, nullValue, index) => add(cell, field, reference, nullValue, index, 'spectrum-online'));
  compareNumericRecords(referenceCase.posterior, nullCase.posterior, 'online-spectrum.posterior', nuisanceCell, representativeIndex,
    (cell, field, reference, nullValue, index) => add(cell, field, reference, nullValue, index, 'spectrum-online'));
}

function compareExactCase(
  referenceCase: ValidationCase,
  nullCase: ValidationCase,
  nuisanceCell: string,
  representativeIndex: number,
  add: (nuisanceCell: string, field: string, reference: unknown, nullValue: unknown, representativeIndex?: number) => void,
): void {
  for (const field of [
    'firstReadyOpportunity',
    'result',
    'confidence',
    'topLeaf',
    'topLeafPosterior',
    'bandwidthHz',
    'selectedTrackAdmissions',
    'localTrackAdmissions',
    'associationMode',
    'associationMemberCount',
    'associationRegionBandwidthHz',
    'knownSupportRank',
  ] as const) {
    if (!equivalentValue(referenceCase[field], nullCase[field])) {
      add(nuisanceCell, `case.${field}`, referenceCase[field], nullCase[field], representativeIndex);
    }
  }
  compareNumericRecords(referenceCase.features, nullCase.features, 'features', nuisanceCell, representativeIndex, add);
  compareNumericRecords(referenceCase.posterior, nullCase.posterior, 'posterior', nuisanceCell, representativeIndex, add);
}

function compareExactEvidenceView(
  referenceCase: EvidenceViewCase,
  nullCase: EvidenceViewCase,
  nuisanceCell: string,
  representativeIndex: number,
  view: EvidenceViewCase['view'],
  add: (
    nuisanceCell: string,
    field: string,
    reference: unknown,
    nullValue: unknown,
    representativeIndex?: number,
    view?: EvidenceViewCase['view'],
  ) => void,
): void {
  for (const field of [
    'measuredBandwidthHz',
    'result',
    'topLeaf',
    'topLeafPosterior',
    'supportRank',
  ] as const) {
    if (!equivalentValue(referenceCase[field], nullCase[field])) {
      add(nuisanceCell, `evidence.${field}`, referenceCase[field], nullCase[field], representativeIndex, view);
    }
  }
  compareNumericRecords(referenceCase.posterior, nullCase.posterior, 'evidence.posterior', nuisanceCell, representativeIndex,
    (cell, field, reference, nullValue, index) => add(cell, field, reference, nullValue, index, view));
}

function compareNumericRecords(
  reference: Readonly<Record<string, number>>,
  nullValue: Readonly<Record<string, number>>,
  prefix: string,
  nuisanceCell: string,
  representativeIndex: number,
  add: (nuisanceCell: string, field: string, reference: unknown, nullValue: unknown, representativeIndex?: number) => void,
): void {
  const keys = [...new Set([...Object.keys(reference), ...Object.keys(nullValue)])].sort();
  for (const key of keys) {
    if (!(key in reference) || !(key in nullValue) || !equivalentValue(reference[key], nullValue[key])) {
      add(nuisanceCell, `${prefix}.${key}`, reference[key], nullValue[key], representativeIndex);
    }
  }
}

function equivalentValue(reference: unknown, nullValue: unknown): boolean {
  if (typeof reference === 'number' && typeof nullValue === 'number') {
    if (Object.is(reference, nullValue)) return true;
    if (!Number.isFinite(reference) || !Number.isFinite(nullValue)) return false;
    return Math.abs(reference - nullValue) <= EXACT_EQUIVALENCE_NUMERICAL_TOLERANCE
      * Math.max(1, Math.abs(reference), Math.abs(nullValue));
  }
  return Object.is(reference, nullValue);
}

function compareValidationCasesForPairing(left: ValidationCase, right: ValidationCase): number {
  return left.firstReadyOpportunity - right.firstReadyOpportunity
    || left.associationMode.localeCompare(right.associationMode)
    || left.bandwidthHz - right.bandwidthHz
    || left.representativeKey.localeCompare(right.representativeKey);
}

function compareEvidenceViewCasesForPairing(left: EvidenceViewCase, right: EvidenceViewCase): number {
  return left.measuredBandwidthHz - right.measuredBandwidthHz
    || left.representativeKey.localeCompare(right.representativeKey);
}

function compareOnlineSpectrumCasesForPairing(left: RollingWindowCase, right: RollingWindowCase): number {
  return left.readyOpportunity - right.readyOpportunity
    || left.associationMode.localeCompare(right.associationMode)
    || left.measuredBandwidthHz - right.measuredBandwidthHz
    || left.representativeKey.localeCompare(right.representativeKey);
}

function assertCanonicalCorpusSourceArtifactPaths(paths: readonly string[]): void {
  if (new Set(paths).size !== paths.length) throw new Error('SignalLab corpus source manifest contains duplicate artifact paths');
  for (const path of paths) assertRepositoryRelativePath(path, 'SignalLab corpus source artifact');
  const canonical = [...paths].sort((left, right) => left.localeCompare(right));
  if (paths.some((path, index) => path !== canonical[index])) throw new Error('SignalLab corpus source manifest paths must be in canonical lexical order');
}

function assertRepositoryRelativePath(path: string, label: string): void {
  if (!path || path.includes('\\') || posix.isAbsolute(path) || posix.normalize(path) !== path
    || path === '..' || path.startsWith('../') || path.includes('/../')) {
    throw new Error(`${label} ${JSON.stringify(path)} must be a canonical repository-relative POSIX path`);
  }
}

function corpusSourceArtifact(path: string): { path: string; sha256: string } {
  const file = resolve(SIGNAL_LAB_REPOSITORY_ROOT, path);
  const status = lstatSync(file);
  if (!status.isFile() || status.isSymbolicLink()) throw new Error(`SignalLab corpus source artifact ${path} must be a regular non-symlink file`);
  gitOutput(['ls-files', '--error-unmatch', '--', path]);
  const bytes = readFileSync(file);
  const committedBytes = gitOutput(['show', `${PINNED_SIGNAL_LAB_COMMIT}:${path}`]);
  if (!bytes.equals(committedBytes)) {
    throw new Error(`SignalLab corpus source artifact ${path} differs from pinned commit ${PINNED_SIGNAL_LAB_COMMIT}`);
  }
  return { path, sha256: createHash('sha256').update(bytes).digest('hex') };
}

function assertSignalLabRepositoryIsClean(): void {
  const status = gitOutput(['status', '--porcelain=v1', '-z', '--untracked-files=all']);
  if (status.length !== 0) {
    throw new Error('SignalLab repository must have a clean index and worktree, including no untracked files, before classifier validation');
  }
}

function assertCorpusSourceImportClosure(
  entryPath: string,
  expectedPaths: readonly string[],
  manifestPaths: readonly string[],
): void {
  const discovered = new Set<string>();
  const pending = [entryPath];
  while (pending.length > 0) {
    const path = pending.pop()!;
    assertRepositoryRelativePath(path, 'SignalLab corpus import');
    if (discovered.has(path)) continue;
    discovered.add(path);
    const source = readFileSync(resolve(SIGNAL_LAB_REPOSITORY_ROOT, path), 'utf8');
    for (const specifier of relativeTypeScriptModuleSpecifiers(source)) {
      if (!specifier.endsWith('.js') && !specifier.endsWith('.ts')) {
        throw new Error(`SignalLab corpus import ${JSON.stringify(specifier)} from ${path} must declare a .js or .ts TypeScript module target`);
      }
      const resolvedPath = posix.normalize(posix.join(
        posix.dirname(path),
        specifier.endsWith('.js') ? `${specifier.slice(0, -3)}.ts` : specifier,
      ));
      assertRepositoryRelativePath(resolvedPath, 'Resolved SignalLab corpus import');
      pending.push(resolvedPath);
    }
  }
  const actual = [...discovered].sort();
  if (JSON.stringify(actual) !== JSON.stringify(expectedPaths)) {
    throw new Error(`SignalLab corpus TypeScript import closure ${JSON.stringify(actual)} does not match validator-owned ${JSON.stringify(expectedPaths)}`);
  }
  const manifest = new Set(manifestPaths);
  const omitted = actual.filter((path) => !manifest.has(path));
  if (omitted.length > 0) throw new Error(`SignalLab corpus source manifest omits import-closure artifacts: ${omitted.join(', ')}`);
}

function relativeTypeScriptModuleSpecifiers(source: string): string[] {
  const patterns = [
    /\b(?:import|export)\s+(?:type\s+)?(?:[^'\";]*?\s+from\s+)?['\"](\.[^'\"]+)['\"]/g,
    /\bimport\s*\(\s*['\"](\.[^'\"]+)['\"]/g,
    /\brequire\s*\(\s*['\"](\.[^'\"]+)['\"]/g,
  ];
  return [...new Set(patterns.flatMap((pattern) => [...source.matchAll(pattern)].map((match) => match[1]!)))].sort();
}

function gitOutput(arguments_: readonly string[]): Buffer {
  return execFileSync('git', arguments_, {
    cwd: SIGNAL_LAB_REPOSITORY_ROOT,
    encoding: 'buffer',
    maxBuffer: 16 * 1024 * 1024,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
}

interface CausalProductionAttemptOptions {
  readonly scenario: CanonicalClassificationScenario;
  readonly temporalSchedule: PinnedTemporalSchedule;
  readonly observationHorizon: number;
  readonly seed: number;
  readonly snrDb: number;
  readonly actualRbwHz: number;
  readonly detectedPowerSynthesisFilterWidthHz: number;
  readonly context: string;
  readonly branch: ProductionAcquisitionBranch;
}

function acquireProductionAttempt(options: CausalProductionAttemptOptions): ProductionOnlineSelection {
  const {
    scenario,
    temporalSchedule,
    observationHorizon,
    seed,
    snrDb,
    actualRbwHz,
    detectedPowerSynthesisFilterWidthHz,
    context,
    branch,
  } = options;
  const detector = new SignalDetector(PRODUCTION_DETECTION_CONFIG);
  const tracker = new SignalTracker(PRODUCTION_DETECTION_CONFIG);
  const clock = new IndependentCausalSourceClock(temporalSchedule, branch);
  const sweeps: Sweep[] = [];
  const everReadyRepresentativeKeys = new Set<string>();
  const admittedRepresentativeKeys = new Set<string>();
  const representatives: FirstReadyRepresentative[] = [];
  const onlineReadyRepresentatives: OnlineReadyRepresentative[] = [];
  const everAssociationModes = new Set<string>();
  const regularAssociationIds = new Set<string>();
  const agileAssociationIds = new Set<string>();
  const expiredRegularAssociationIds = new Set<string>();
  let previousRegularAssociationByTrack = new Map<string, string>();
  let finalTracks: readonly DetectedSignal[] = [];
  let maximumActiveAdmissions = 0;
  let maximumLocalTrackAdmissions = 0;
  let provenanceUnavailableWindowCount = 0;
  let firstReadyOpportunity: number | undefined;
  let liveEnvelopeCapture: LiveEnvelopeCapture | undefined;
  for (let lookIndex = 0; lookIndex < observationHorizon; lookIndex++) {
    const sourceLookIndex = clock.acquireSpectrum();
    const sourceObservation = synthesizeCanonicalObservation(scenario.id, {
      lookIndex: sourceLookIndex,
      seed,
      snrDb,
      actualRbwHz,
      detectedPowerSynthesisFilterWidthHz,
      points: SWEEP_POINTS,
      sweepTimeSeconds: SWEEP_TIME_SECONDS,
      zeroSpanPoints: ZERO_SPAN_POINTS,
      zeroSpanSamplePeriodSeconds: ZERO_SPAN_SAMPLE_PERIOD_SECONDS,
    });
    assertDetectedPowerSynthesisProvenance(
      sourceObservation,
      detectedPowerSynthesisFilterWidthHz,
      `${context} swept observation`,
    );
    const sweep = asSweep(scenario, sourceObservation);
    if (sweep.sequence !== sourceLookIndex + 1) {
      throw new Error(`${context} spectrum sequence ${sweep.sequence} does not bind source look ${sourceLookIndex}`);
    }
    sweeps.push(sweep);
    const tracks = tracker.update(sweep, detector.analyze(sweep));
    finalTracks = tracks;
    const activeTracks = tracks.filter((track) => track.state === 'active');
    const captureTargetProjections = branch === 'qualified-envelope'
      ? classificationCaptureTargetProjections(tracks)
      : [];
    const activeRepresentatives = branch === 'qualified-envelope'
      ? captureTargetProjections.map((projection) =>
          projection.projectedRepresentative)
      : classificationRepresentatives(activeTracks);
    if (branch === 'qualified-envelope') {
      const independentlyRanked = independentlyReplayCaptureTargetProjections(tracks);
      if (captureTargetProjections.length !== independentlyRanked.length
        || captureTargetProjections.some((projection, index) => {
          const independent = independentlyRanked[index];
          return independent === undefined
            || projection.rawTarget.id !== independent.rawTarget.id
            || projection.projectedRepresentative.id
              !== independent.projectedRepresentative.id
            || projection.projectionKind !== independent.projectionKind;
        })) {
        throw new Error(
          `${context} shared detected-power target selection disagrees with the independent v4 integrated-excess physical/agile-member replay`,
        );
      }
    }
    const readyRepresentatives: readonly {
      detection: DetectedSignal;
      representativeKey: string;
      rawTarget?: DetectedSignal;
      projectionKind?: DetectedPowerCaptureProjectionKind;
    }[] = branch === 'qualified-envelope'
      ? captureTargetProjections.slice(0, 1)
          .filter(({ projectedRepresentative }) =>
            classificationSourceSweepIds(projectedRepresentative).length
              >= CLASSIFICATION_ADMISSIONS)
          .filter(({ projectedRepresentative }) =>
            observableAssociationEvidenceIsCurrentlyQualified(
              projectedRepresentative,
            ))
          .map(({ rawTarget, projectedRepresentative, projectionKind }) => ({
            detection: projectedRepresentative,
            rawTarget,
            projectionKind,
            representativeKey: classificationRepresentativeKey(
              projectedRepresentative,
            ),
          }))
      : activeRepresentatives
          .filter((track) =>
            classificationSourceSweepIds(track).length
              >= CLASSIFICATION_ADMISSIONS)
          // Retained operator-visible associations below their current
          // promotion gate are honest insufficient-evidence results, not
          // observation-domain-eligible rolling classifier windows.
          .filter(observableAssociationEvidenceIsCurrentlyQualified)
          .map((detection) => ({
            detection,
            representativeKey: classificationRepresentativeKey(detection),
          }));
    for (const readyRepresentative of readyRepresentatives) {
      const { detection, representativeKey, rawTarget, projectionKind } =
        readyRepresentative;
      everReadyRepresentativeKeys.add(representativeKey);
      const evidenceSweeps = sweeps.slice(0, lookIndex + 1);
      let spectrumObservation: ObservableFeatureObservation;
      try {
        // Match runtime admission before recording either the first-ready
        // representative or an online tail-calibration window. A tracker can
        // be ready while its latest local history remains non-unique; runtime
        // reports that case as insufficient evidence rather than classifying.
        spectrumObservation = extractObservableFeatures(detection, { sweeps: evidenceSweeps });
      } catch (error) {
        if (error instanceof ObservableEvidenceUnavailableError
          && (error.code === 'local-history-not-uniquely-replayable'
            || error.code === 'insufficient-roi-bins')) {
          provenanceUnavailableWindowCount += 1;
          continue;
        }
        throw error;
      }
      if (firstReadyOpportunity === undefined) firstReadyOpportunity = lookIndex + 1;
      if (branch === 'consecutive-spectrum') {
        onlineReadyRepresentatives.push({
          detection: structuredClone(detection),
          representativeKey,
          classificationAdmissions: classificationSourceSweepIds(detection).length,
          localTrackAdmissions: detection.sweepIds.length,
          readyOpportunity: lookIndex + 1,
          evidenceSweeps,
          spectrumObservation,
        });
      }
      let firstReadyRepresentative = representatives.find((item) => item.representativeKey === representativeKey);
      if (!admittedRepresentativeKeys.has(representativeKey)) {
        admittedRepresentativeKeys.add(representativeKey);
        firstReadyRepresentative = {
          detection: structuredClone(detection),
          representativeKey,
          classificationAdmissions: classificationSourceSweepIds(detection).length,
          localTrackAdmissions: detection.sweepIds.length,
          firstReadyOpportunity: lookIndex + 1,
          evidenceSweeps,
          spectrumObservation,
          ...(rawTarget === undefined
            ? {}
            : { rawCaptureTarget: structuredClone(rawTarget) }),
          ...(projectionKind === undefined ? {} : { captureProjectionKind: projectionKind }),
        };
        representatives.push(firstReadyRepresentative);
      }
      if (branch === 'consecutive-spectrum' || liveEnvelopeCapture !== undefined) continue;
      if (firstReadyRepresentative === undefined) {
        throw new Error(`${context} could not retain the representative that triggered its sole detected-power capture`);
      }
      if (!rawTarget || !projectionKind) {
        throw new Error(
          `${context} qualified-envelope admission lacks its physical raw target projection`,
        );
      }
      const detectedPowerSourceLookIndex = clock.acquireDetectedPower(sourceLookIndex, lookIndex + 1);
      const zeroSpanTuneHz = projectDetectedPowerTuneHz(
        rawTarget.peakHz,
        SIGNAL_LAB_SCALAR_FREQUENCY_RANGE_V1,
      );
      const detectedPowerObservation = synthesizeCanonicalObservation(scenario.id, {
        lookIndex: detectedPowerSourceLookIndex,
        seed,
        snrDb,
        actualRbwHz,
        detectedPowerSynthesisFilterWidthHz,
        points: SWEEP_POINTS,
        sweepTimeSeconds: SWEEP_TIME_SECONDS,
        zeroSpanPoints: ZERO_SPAN_POINTS,
        zeroSpanSamplePeriodSeconds: ZERO_SPAN_SAMPLE_PERIOD_SECONDS,
        zeroSpanFrequencyHz: zeroSpanTuneHz,
      });
      assertDetectedPowerSynthesisProvenance(
        detectedPowerObservation,
        detectedPowerSynthesisFilterWidthHz,
        `${context} detected-power observation`,
      );
      const zeroSpan = asZeroSpan(detectedPowerObservation, rawTarget);
      if (zeroSpan.sequence !== detectedPowerSourceLookIndex + 1) {
        throw new Error(`${context} detected-power sequence ${zeroSpan.sequence} does not bind source look ${detectedPowerSourceLookIndex}`);
      }
      const detectedPowerCaptureReceipt = createDetectedPowerCaptureReceipt({
        activeSignals: tracks,
        evidenceSweeps: sweeps,
        capture: zeroSpan,
        admittedTargetTuneHz: zeroSpanTuneHz,
        spectrumSweepIds: spectrumObservation.sweepIds,
      });
      assertIndependentDetectedPowerCaptureReceipt({
        receipt: detectedPowerCaptureReceipt,
        tracks,
        evidenceSweeps: sweeps,
        capture: zeroSpan,
        selectedRawTarget: rawTarget,
        selectedRepresentative: detection,
        selectedProjectionKind: projectionKind,
        spectrumSweepIds: spectrumObservation.sweepIds,
        admittedTargetTuneHz: zeroSpanTuneHz,
        context,
      });
      assertDetectedPowerCaptureReceiptMatches({
        receipt: detectedPowerCaptureReceipt,
        detection,
        capture: zeroSpan,
        spectrumSweepIds: spectrumObservation.sweepIds,
      });
      const frequencyAgileCapture =
        detection.associationMode === 'frequency-agile-2g4-activity';
      if (frequencyAgileCapture) {
        if (projectionKind !== 'current-qualified-agile-latest-member') {
          throw new Error(
            `${context} frequency-agile classifier evidence was not projected from its exact current physical member`,
          );
        }
      }
      try {
        // Always give production the complete receipt-qualified capture. For
        // agile activity, the extractor must verify the receipt and enact its
        // fixed-tune censoring; the validator must not synthesize that result.
        const receiptQualifiedObservation =
          extractValidatorReceiptQualifiedObservation({
            detection,
            evidenceSweeps,
            spectrumObservation,
            zeroSpan,
            detectedPowerCaptureReceipt,
          });
        if (frequencyAgileCapture) {
          liveEnvelopeCapture = {
            representative: firstReadyRepresentative,
            zeroSpan,
            detectedPowerCaptureReceipt,
            sourceLookIndex: detectedPowerSourceLookIndex,
            classifierObservation: receiptQualifiedObservation,
            detectedPowerEvidenceDisposition:
              'censored-frequency-agile-spectrum-only',
            envelopeEvidenceCensoringPolicyId:
              PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORING_POLICY_ID,
          };
          continue;
        }
        if (receiptQualifiedObservation.detectedPowerAcquisitionQualification
          !== PINNED_DETECTED_POWER_ACQUISITION_QUALIFICATION
          || receiptQualifiedObservation.detectedPowerSelectionCondition
            !== PINNED_DETECTED_POWER_SELECTION_CONDITION) {
          throw new Error(`${context} causal detected-power capture was not admitted into its bound envelope view`);
        }
        liveEnvelopeCapture = {
          representative: firstReadyRepresentative,
          zeroSpan,
          detectedPowerCaptureReceipt,
          sourceLookIndex: detectedPowerSourceLookIndex,
          classifierObservation: receiptQualifiedObservation,
          detectedPowerEvidenceDisposition: 'admitted-envelope',
          envelopeObservation: receiptQualifiedObservation,
        };
      } catch (error) {
        if (!(error instanceof ObservableEvidenceUnavailableError)
          || frequencyAgileCapture) throw error;
        liveEnvelopeCapture = {
          representative: firstReadyRepresentative,
          zeroSpan,
          detectedPowerCaptureReceipt,
          sourceLookIndex: detectedPowerSourceLookIndex,
          classifierObservation: spectrumObservation,
          detectedPowerEvidenceDisposition: 'admitted-envelope',
          unavailableCode: error.code,
        };
      }
    }
    for (const representative of activeRepresentatives) {
      const associationMode = representative.associationMode ?? 'frequency-local';
      everAssociationModes.add(associationMode);
      maximumActiveAdmissions = Math.max(maximumActiveAdmissions, classificationSourceSweepIds(representative).length);
      maximumLocalTrackAdmissions = Math.max(maximumLocalTrackAdmissions, representative.sweepIds.length);
      if (associationMode === 'regular-spectral-component-activity' && representative.associationId) regularAssociationIds.add(representative.associationId);
      if (associationMode === 'frequency-agile-2g4-activity' && representative.associationId) agileAssociationIds.add(representative.associationId);
    }
    const currentRegularAssociationByTrack = new Map(tracks.flatMap((track) =>
      track.associationMode === 'regular-spectral-component-activity' && track.associationId
        ? [[track.id, track.associationId] as const]
        : []));
    for (const [trackId, associationId] of previousRegularAssociationByTrack) {
      if (currentRegularAssociationByTrack.get(trackId) !== associationId) expiredRegularAssociationIds.add(associationId);
    }
    previousRegularAssociationByTrack = currentRegularAssociationByTrack;
  }
  const finalActiveRepresentatives = classificationRepresentatives(finalTracks.filter((track) => track.state === 'active'));
  const finalReadyRepresentatives = finalActiveRepresentatives
    .filter((track) => classificationSourceSweepIds(track).length >= CLASSIFICATION_ADMISSIONS);
  const acquisitionTrace = clock.trace();
  const expectedDetectedPowerCaptures = branch === 'qualified-envelope'
    && liveEnvelopeCapture !== undefined ? 1 : 0;
  if (acquisitionTrace.spectrumSourceLookIndices.length !== observationHorizon
    || acquisitionTrace.detectedPowerSourceLookIndices.length
      !== expectedDetectedPowerCaptures
    || acquisitionTrace.sourceLookIndices.length
      !== observationHorizon + expectedDetectedPowerCaptures
    || (branch === 'consecutive-spectrum' && liveEnvelopeCapture !== undefined)) {
    throw new Error(`${context} did not preserve its exact ${branch} physical-acquisition contract`);
  }
  return {
    representatives: representatives.sort((left, right) => left.firstReadyOpportunity - right.firstReadyOpportunity
      || left.representativeKey.localeCompare(right.representativeKey)),
    onlineReadyRepresentatives: onlineReadyRepresentatives.sort((left, right) => left.readyOpportunity - right.readyOpportunity
      || left.representativeKey.localeCompare(right.representativeKey)),
    everReadyRepresentativeKeys: [...everReadyRepresentativeKeys].sort(),
    provenanceUnavailableWindowCount,
    finalReadyRepresentativeCount: finalReadyRepresentatives.length,
    finalActiveRepresentativeCount: finalActiveRepresentatives.length,
    maximumActiveAdmissions,
    maximumLocalTrackAdmissions,
    ...(firstReadyOpportunity === undefined ? {} : { firstReadyOpportunity }),
    everAssociationModes: [...everAssociationModes].sort(),
    finalAssociationModes: [...new Set(finalReadyRepresentatives.map((detection) => detection.associationMode ?? 'frequency-local'))].sort(),
    regularAssociationIds: [...regularAssociationIds].sort(),
    agileAssociationIds: [...agileAssociationIds].sort(),
    regularAssociationExpirations: expiredRegularAssociationIds.size,
    ...(liveEnvelopeCapture === undefined ? {} : { liveEnvelopeCapture }),
    acquisitionTrace,
  };
}

function observationOpportunityHorizon(scenario: CanonicalClassificationScenario): number {
  const startHz = scenario.centerHz - scenario.recommendedSpanHz / 2;
  const stopHz = scenario.centerHz + scenario.recommendedSpanHz / 2;
  return startHz <= FULL_BAND_2G4_START_HZ && stopHz >= FULL_BAND_2G4_STOP_HZ
    ? FULL_BAND_2G4_OBSERVATION_OPPORTUNITIES
    : STANDARD_OBSERVATION_OPPORTUNITIES;
}

function validationAttemptId(scenarioId: string, snrDb: number, rbwDivisor: number, seed: number): string {
  return `${scenarioId}:snr=${snrDb}:rbw=${rbwDivisor}:seed=${seed}`;
}

function classificationRepresentativeKey(track: DetectedSignal): string {
  const associationMode = track.associationMode ?? 'frequency-local';
  return `${associationMode}:${associationMode === 'frequency-local' ? track.id : track.associationId ?? track.id}`;
}

function assertIndependentDetectedPowerCaptureReceipt({
  receipt,
  tracks,
  evidenceSweeps,
  capture,
  selectedRawTarget,
  selectedRepresentative,
  selectedProjectionKind,
  spectrumSweepIds,
  admittedTargetTuneHz,
  context,
}: {
  receipt: DetectedPowerCaptureReceipt;
  tracks: readonly DetectedSignal[];
  evidenceSweeps: readonly Sweep[];
  capture: ZeroSpanCapture;
  selectedRawTarget: DetectedSignal;
  selectedRepresentative: DetectedSignal;
  selectedProjectionKind: DetectedPowerCaptureProjectionKind;
  spectrumSweepIds: readonly string[];
  admittedTargetTuneHz: number;
  context: string;
}): void {
  const independentlyRanked = independentlyReplayCaptureTargetProjections(tracks);
  const eligibleRawTargetIds = new Set(
    independentlyRanked.map((projection) => projection.rawTarget.id),
  );
  const inputOrdinalById = new Map(
    tracks.filter((track) => eligibleRawTargetIds.has(track.id))
      .map((track, inputOrdinal) => [track.id, inputOrdinal] as const),
  );
  const expectedRuntimeAdmissionByRawTargetId = new Map(
    independentlyRanked.map((projection) => [
      projection.rawTarget.id,
      independentlyReplayRuntimeAdmission(
        projection.projectedRepresentative,
        evidenceSweeps,
      ),
    ] as const),
  );
  const sameOptionalStrings = (
    left: readonly string[] | undefined,
    right: readonly string[] | undefined,
  ) => left === undefined
    ? right === undefined
    : right !== undefined
      && left.length === right.length
      && left.every((value, index) => value === right[index]);
  const candidateOrderMatches = receipt.candidates.length === independentlyRanked.length
    && receipt.candidates.every((candidate, rank) => {
      const projection: IndependentlyReplayedCaptureTargetProjection =
        independentlyRanked[rank]!;
      const {
        rawTarget,
        projectedRepresentative,
        projectionKind,
        rankEvidence,
      } = projection;
      const expectedRuntimeAdmission =
        expectedRuntimeAdmissionByRawTargetId.get(rawTarget.id);
      return candidate.rank === rank
        && candidate.inputOrdinal === inputOrdinalById.get(rawTarget.id)
        && candidate.rawTargetId === rawTarget.id
        && candidate.currentPeakDbm === rawTarget.peakDbm
        && candidate.currentSourceSweepId === rankEvidence.sourceSweepId
        && candidate.currentSupportStartHz === rankEvidence.supportStartHz
        && candidate.currentSupportStopHz === rankEvidence.supportStopHz
        && candidate.currentSupportCellCount === rankEvidence.supportCellCount
        && candidate.currentRobustFloorDbm === rankEvidence.robustFloorDbm
        && candidate.currentActualRbwHz === rankEvidence.actualRbwHz
        && candidate.currentIntegratedExcessPowerMw
          === rankEvidence.integratedExcessPowerMw
        && candidate.currentPeakHz === rawTarget.peakHz
        && candidate.currentStartHz === rawTarget.startHz
        && candidate.currentStopHz === rawTarget.stopHz
        && candidate.state === rawTarget.state
        && candidate.missedSweeps === rawTarget.missedSweeps
        && candidate.lastSeenAt === rawTarget.lastSeenAt
        && candidate.associationMode === rawTarget.associationMode
        && candidate.associationId === rawTarget.associationId
        && sameOptionalStrings(
          candidate.associationMemberTrackIds,
          rawTarget.associationMemberTrackIds,
        )
        && candidate.associationMissedSweeps
          === rawTarget.associationMissedSweeps
        && candidate.projectionKind === projectionKind
        && candidate.projectedRepresentativeId === projectedRepresentative.id
        && expectedRuntimeAdmission !== undefined
        && sameRuntimeAdmission(
          candidate.runtimeAdmission,
          expectedRuntimeAdmission,
        );
    });
  if (!candidateOrderMatches) {
    throw new Error(
      `${context} detected-power receipt candidate evidence disagrees with the independent integrated-excess/key/ID ordering`,
    );
  }
  const sameSpectrumWindow = receipt.spectrumSweepIds.length === spectrumSweepIds.length
    && receipt.spectrumSweepIds.every(
      (sweepId, index) => sweepId === spectrumSweepIds[index],
    );
  const selectedCandidate = receipt.candidates.find(
    (candidate) => candidate.rawTargetId === receipt.selection.rawTargetId,
  );
  const automaticRankZeroProjection = independentlyRanked[0];
  const selectedAdmissionWindowMatches = selectedCandidate?.runtimeAdmission.status === 'admitted'
    && selectedCandidate.runtimeAdmission.spectrumSweepIds.length
      === spectrumSweepIds.length
    && selectedCandidate.runtimeAdmission.spectrumSweepIds.every(
      (sweepId, index) => sweepId === spectrumSweepIds[index],
    );
  const projectedRepresentativeSnapshotMatches =
    receipt.projectedRepresentative.id === selectedRepresentative.id
    && receipt.projectedRepresentative.startHz === selectedRepresentative.startHz
    && receipt.projectedRepresentative.stopHz === selectedRepresentative.stopHz
    && receipt.projectedRepresentative.peakHz === selectedRepresentative.peakHz
    && receipt.projectedRepresentative.peakDbm === selectedRepresentative.peakDbm
    && receipt.projectedRepresentative.bandwidthHz
      === selectedRepresentative.bandwidthHz
    && receipt.projectedRepresentative.missedSweeps
      === selectedRepresentative.missedSweeps
    && receipt.projectedRepresentative.lastSeenAt
      === selectedRepresentative.lastSeenAt
    && receipt.projectedRepresentative.associationMode
      === selectedRepresentative.associationMode
    && receipt.projectedRepresentative.associationId
      === selectedRepresentative.associationId
    && sameOptionalStrings(
      receipt.projectedRepresentative.associationMemberTrackIds,
      selectedRepresentative.associationMemberTrackIds,
    )
    && receipt.projectedRepresentative.associationMissedSweeps
      === selectedRepresentative.associationMissedSweeps;
  const independentlyProjectedTuneHz = projectDetectedPowerTuneHz(
    selectedRawTarget.peakHz,
    SIGNAL_LAB_SCALAR_FREQUENCY_RANGE_V1,
  );
  if (receipt.schemaVersion !== 4
    || receipt.capturePolicyId !== PINNED_DETECTED_POWER_CAPTURE_POLICY_ID
    || receipt.targetSelectionPolicyId
      !== PINNED_CAPTURE_TARGET_SELECTION_POLICY_ID
    || receipt.runtimeAdmissionPolicyId
      !== PINNED_CAPTURE_RUNTIME_ADMISSION_POLICY_ID
    || receipt.selection.mode !== 'integrated-excess-current'
    || receipt.selection.preferredRawTargetId !== undefined
    || automaticRankZeroProjection?.rawTarget.id !== selectedRawTarget.id
    || automaticRankZeroProjection?.projectedRepresentative.id
      !== selectedRepresentative.id
    || automaticRankZeroProjection?.projectionKind !== selectedProjectionKind
    || expectedRuntimeAdmissionByRawTargetId.get(selectedRawTarget.id)?.status
      !== 'admitted'
    || receipt.selection.rawTargetId !== selectedRawTarget.id
    || receipt.selection.projectedRepresentativeId !== selectedRepresentative.id
    || selectedCandidate?.projectedRepresentativeId !== selectedRepresentative.id
    || selectedCandidate?.rank !== 0
    || receipt.candidates[0] !== selectedCandidate
    || selectedCandidate?.projectionKind !== selectedProjectionKind
    || selectedCandidate?.currentSourceSweepId !== spectrumSweepIds[0]
    || !selectedAdmissionWindowMatches
    || !projectedRepresentativeSnapshotMatches
    || admittedTargetTuneHz !== independentlyProjectedTuneHz
    || receipt.capture.id !== capture.id
    || receipt.capture.sequence !== capture.sequence
    || receipt.capture.capturedAt !== capture.capturedAt
    || receipt.capture.measurementIdentityKey
      !== measurementIdentityKey(capture.identity)
    || receipt.capture.targetDetectionId !== selectedRawTarget.id
    || receipt.capture.targetDetectionId !== capture.targetDetectionId
    || receipt.capture.admittedTargetTuneHz !== admittedTargetTuneHz
    || receipt.capture.frequencyHz !== capture.frequencyHz
    || receipt.capture.requestedCenterHz !== capture.requested.centerHz
    || receipt.capture.payloadBinding.algorithm !== 'sha256'
    || receipt.capture.payloadBinding.canonicalization
      !== 'zero-span-capture-canonical-json-v1'
    || receipt.capture.payloadBinding.sha256
      !== independentDetectedPowerCapturePayloadSha256(capture)
    || capture.frequencyHz !== admittedTargetTuneHz
    || capture.requested.centerHz !== admittedTargetTuneHz
    || !sameSpectrumWindow) {
    throw new Error(
      `${context} detected-power receipt does not bind the independently selected current target, tune, exact spectrum window, and complete canonical capture payload`,
    );
  }
}

function independentlyReplayRuntimeAdmission(
  projectedRepresentative: DetectedSignal,
  evidenceSweeps: readonly Sweep[],
): DetectedPowerCaptureReceipt['candidates'][number]['runtimeAdmission'] {
  if (!observableAssociationEvidenceIsCurrentlyQualified(
    projectedRepresentative,
  )) {
    return {
      status: 'unavailable',
      reason: 'association-not-currently-qualified',
    };
  }
  try {
    const observation = extractObservableFeatures(projectedRepresentative, {
      sweeps: evidenceSweeps,
    });
    return observation.sweepIds.length === CLASSIFICATION_ADMISSIONS
      ? { status: 'admitted', spectrumSweepIds: [...observation.sweepIds] }
      : { status: 'unavailable', reason: 'insufficient-spectrum-history' };
  } catch (error) {
    if (!(error instanceof ObservableEvidenceUnavailableError)) throw error;
    return { status: 'unavailable', reason: error.code };
  }
}

function sameRuntimeAdmission(
  left: DetectedPowerCaptureReceipt['candidates'][number]['runtimeAdmission'],
  right: DetectedPowerCaptureReceipt['candidates'][number]['runtimeAdmission'],
): boolean {
  if (left.status !== right.status) return false;
  if (left.status === 'unavailable') {
    return right.status === 'unavailable' && left.reason === right.reason;
  }
  return right.status === 'admitted'
    && left.spectrumSweepIds.length === right.spectrumSweepIds.length
    && left.spectrumSweepIds.every(
      (sweepId, index) => sweepId === right.spectrumSweepIds[index],
    );
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
    kind: 'zero-span', id: `zero-${observation.scenarioId}-${observation.seed}-${observation.lookIndex}`, sequence: observation.lookIndex + 1,
    capturedAt: new Date(Date.UTC(2026, 0, 1) + observation.lookIndex * observation.sweepTimeSeconds * 1_000).toISOString(), elapsedMilliseconds: sweepTimeSeconds * 1_000,
    frequencyHz: observation.zeroSpanFrequencyHz, samplePeriodSeconds: observation.zeroSpanSamplePeriodSeconds, timingQualification: 'simulation-exact',
    targetDetectionId: detection.id,
    powerDbm: observation.zeroSpanPowerDbm,
    requested,
    actualRbwHz: null, actualAttenuationDb: null,
    resolutionBandwidthQualification: 'unavailable', attenuationQualification: 'not-applicable',
    source: 'signal-lab-synthetic', complete: true, identity,
  };
}

function auditPriorSensitivity(
  validationCases: readonly (ValidationCase | RollingWindowCase)[],
  population: 'capture-qualified-selected-view' | 'complete-online-spectrum',
) {
  const variants = [
    {
      id: 'engineering-baseline-v1',
      kind: 'declared-engineering-assumption',
      description: 'Pinned design weights; not an estimate of field prevalence.',
      prior: { ...PINNED_ENGINEERING_PRIOR },
    },
    {
      id: 'unknown-mass-0.10-known-ratios-preserved-v1',
      kind: 'unknown-mass-shift',
      description: 'Unknown mass reduced to 0.10 while preserving every known-class prior ratio.',
      prior: priorWithUnknownMass(0.10),
    },
    {
      id: 'unknown-mass-0.30-known-ratios-preserved-v1',
      kind: 'unknown-mass-shift',
      description: 'Unknown mass increased to 0.30 while preserving every known-class prior ratio.',
      prior: priorWithUnknownMass(0.30),
    },
    {
      id: 'cellular-family-up-within-family-ratios-preserved-v1',
      kind: 'family-mass-shift',
      description: 'Cellular family mass is weighted 1.35x and other known families 0.90x; unknown mass and every within-family ratio are preserved.',
      prior: priorWithKnownFamilyMultipliers({ analog: 0.90, cellular: 1.35, wifi: 0.90, bluetooth: 0.90 }),
    },
    {
      id: 'unlicensed-families-up-within-family-ratios-preserved-v1',
      kind: 'family-mass-shift',
      description: 'Wi-Fi/Bluetooth family masses are weighted 1.25x and analog/cellular 0.90x; unknown mass and every within-family ratio are preserved.',
      prior: priorWithKnownFamilyMultipliers({ analog: 0.90, cellular: 0.90, wifi: 1.25, bluetooth: 1.25 }),
    },
  ] as const;
  const modelPriorMatchesPinned = OBSERVABLE_LEAF_CLASSES.every((id) => {
    const model = BAYESIAN_OBSERVABLE_MODEL.classModels.find((candidate) => candidate.id === id);
    return model !== undefined && Math.abs(Math.exp(model.logPrior) - PINNED_ENGINEERING_PRIOR[id]) <= 1e-12;
  });
  const evaluated = variants.map((variant) => {
    const priorTotal = OBSERVABLE_LEAF_CLASSES.reduce((sum, id) => sum + variant.prior[id], 0);
    const decisions = validationCases.map((item) => {
      const bandwidthHz = 'bandwidthHz' in item
        ? item.bandwidthHz
        : item.measuredBandwidthHz;
      const observation: ObservableFeatureObservation = {
        values: item.features,
        limitations: item.limitations as ObservableFeatureObservation['limitations'],
        ...(item.associationEvidenceQualification === undefined
          ? {}
          : { associationEvidenceQualification: item.associationEvidenceQualification }),
        occupiedStartHz: item.occupiedStartHz,
        occupiedStopHz: item.occupiedStopHz,
        centerHz: item.centerHz,
        bandwidthHz,
        binWidthHz: item.binWidthHz,
        sweepIds: Array.from({ length: CLASSIFICATION_ADMISSIONS }, (_unused, index) => `prior-audit-${index}`),
        views: item.views,
        ...('zeroSpanCaptureId' in item && item.zeroSpanCaptureId !== undefined
          ? { zeroSpanCaptureId: item.zeroSpanCaptureId }
          : {}),
        ...('detectedPowerAcquisitionQualification' in item
          && item.detectedPowerAcquisitionQualification !== undefined
          ? {
              detectedPowerAcquisitionQualification:
                item.detectedPowerAcquisitionQualification,
            }
          : {}),
        ...('detectedPowerSelectionCondition' in item
          && item.detectedPowerSelectionCondition !== undefined
          ? {
              detectedPowerSelectionCondition:
                item.detectedPowerSelectionCondition,
            }
          : {}),
      };
      const posterior = posteriorUnderDeclaredPrior(observation, variant.prior);
      const selected = item.limitations.includes('partial-span-boundary-censoring')
        ? { label: 'unknown' as const }
        : selectObservableDecision(posterior, observation, item.knownSupportRank);
      const result = selected.label === 'unknown' ? 'unknown' : `observable:${selected.label}`;
      const acceptedHierarchy = acceptsAnyTruth(
        result,
        item.allowedModelTruths,
        item.nominalBandwidthHz,
        bandwidthHz,
      );
      return { item, result, acceptedHierarchy };
    });
    const knownDecisions = decisions.filter(({ item }) => item.modelTruth !== 'unknown-signal');
    const unknownDecisions = decisions.filter(({ item }) => item.modelTruth === 'unknown-signal');
    const incompatibleNonUnknown = decisions.filter(({ result, acceptedHierarchy }) => result !== 'unknown' && !acceptedHierarchy);
    const falseAcceptedUnknown = unknownDecisions.filter(({ result, acceptedHierarchy }) => result !== 'unknown' && !acceptedHierarchy);
    const decisionChanges = decisions.filter(({ item, result }) => result !== item.result);
    const knownCoverage = fraction(knownDecisions, ({ result }) => result !== 'unknown');
    const hierarchicalAccuracy = fraction(decisions, ({ acceptedHierarchy }) => acceptedHierarchy);
    const incompatibleNonUnknownRisk = incompatibleNonUnknown.length / Math.max(1, decisions.length);
    const falseAcceptedUnknownRisk = falseAcceptedUnknown.length / Math.max(1, unknownDecisions.length);
    const decisionChangeRate = decisionChanges.length / Math.max(1, decisions.length);
    const passed = Math.abs(priorTotal - 1) <= 1e-12
      && knownCoverage >= PRIOR_SENSITIVITY_GATES.minimumKnownCoverage
      && hierarchicalAccuracy >= PRIOR_SENSITIVITY_GATES.minimumHierarchicalAccuracy
      && incompatibleNonUnknownRisk <= PRIOR_SENSITIVITY_GATES.maximumIncompatibleNonUnknownRisk
      && falseAcceptedUnknownRisk <= PRIOR_SENSITIVITY_GATES.maximumFalseAcceptedUnknownRisk
      && decisionChangeRate <= PRIOR_SENSITIVITY_GATES.maximumDecisionChangeRate;
    return {
      id: variant.id,
      kind: variant.kind,
      description: variant.description,
      prior: variant.prior,
      priorTotal,
      cases: decisions.length,
      knownCases: knownDecisions.length,
      unknownCases: unknownDecisions.length,
      knownCoverage,
      hierarchicalAccuracy,
      incompatibleNonUnknownCount: incompatibleNonUnknown.length,
      incompatibleNonUnknownRisk,
      falseAcceptedUnknownCount: falseAcceptedUnknown.length,
      falseAcceptedUnknownRisk,
      decisionChangeCount: decisionChanges.length,
      decisionChangeRate,
      passed,
    };
  });
  const baselineDecisionMismatchCount = evaluated[0]?.decisionChangeCount ?? Number.MAX_SAFE_INTEGER;
  return {
    valid: modelPriorMatchesPinned
      && baselineDecisionMismatchCount === 0
      && evaluated.every((variant) => variant.passed),
    qualification: 'deterministic-synthetic-engineering-prior-sensitivity-not-field-prevalence-calibration',
    fieldPrevalenceCalibrated: false,
    fieldValidationLimitation: 'Operational class prevalence and prior calibration remain unmeasured release limitations requiring representative physical survey data.',
    population,
    samples: validationCases.length,
    gates: PRIOR_SENSITIVITY_GATES,
    modelPriorMatchesPinned,
    baselineDecisionMismatchCount,
    variants: evaluated,
  };
}

function priorWithUnknownMass(unknownMass: number): Record<ObservableLeafClass, number> {
  const knownBaselineMass = 1 - PINNED_ENGINEERING_PRIOR['unknown-signal'];
  return Object.fromEntries(OBSERVABLE_LEAF_CLASSES.map((id) => [
    id,
    id === 'unknown-signal'
      ? unknownMass
      : PINNED_ENGINEERING_PRIOR[id] * (1 - unknownMass) / knownBaselineMass,
  ])) as Record<ObservableLeafClass, number>;
}

function priorWithKnownFamilyMultipliers(
  multipliers: Readonly<Record<'analog' | 'cellular' | 'wifi' | 'bluetooth', number>>,
): Record<ObservableLeafClass, number> {
  const unknownMass = PINNED_ENGINEERING_PRIOR['unknown-signal'];
  const weightedKnownTotal = OBSERVABLE_LEAF_CLASSES
    .filter((id) => id !== 'unknown-signal')
    .reduce((sum, id) => sum + PINNED_ENGINEERING_PRIOR[id] * multipliers[priorFamily(id)], 0);
  return Object.fromEntries(OBSERVABLE_LEAF_CLASSES.map((id) => [
    id,
    id === 'unknown-signal'
      ? unknownMass
      : PINNED_ENGINEERING_PRIOR[id] * multipliers[priorFamily(id)] * (1 - unknownMass) / weightedKnownTotal,
  ])) as Record<ObservableLeafClass, number>;
}

function priorFamily(id: Exclude<ObservableLeafClass, 'unknown-signal'>): 'analog' | 'cellular' | 'wifi' | 'bluetooth' {
  if (id === 'cw-like' || id === 'am-dsb-full-carrier-like' || id === 'fm-angle-modulated-like') return 'analog';
  if (id === 'wifi-hr-dsss-like' || id === 'wifi-ofdm-like') return 'wifi';
  if (id === 'bluetooth-like') return 'bluetooth';
  return 'cellular';
}

function pinnedCalibrationActualRbwHz(
  scenario: CanonicalClassificationScenario,
  acquisitionRegime: PinnedCalibrationAcquisitionRegime,
): number {
  const inclusiveGridSpacingHz = scenario.recommendedSpanHz / (SWEEP_POINTS - 1);
  return acquisitionRegime.rbwDivisor === null
    ? inclusiveGridSpacingHz
    : Math.max(inclusiveGridSpacingHz * 0.8, scenario.occupiedBandwidthHz / acquisitionRegime.rbwDivisor, 1_000);
}

function pinnedCalibrationDetectedPowerSynthesisFilterWidthHz(
  actualRbwHz: number,
  acquisitionRegime: PinnedCalibrationAcquisitionRegime,
): number {
  return acquisitionRegime.rbwDivisor === null
    ? PINNED_DETECTED_POWER_SYNTHESIS_FILTER_POLICY.signalLabProductionSynthesisFilterWidthHz
    : actualRbwHz;
}

function branchSchedule(
  branch: ProductionAcquisitionBranch,
  sourcePlanProfileId: string,
  sourceLookIndexOffset: number,
  sourcePlanSpectrumOpportunities: number,
): PinnedTemporalSchedule {
  return Object.freeze({
    id: branch === 'consecutive-spectrum'
      ? `live-spectrum-release-gate-${sourcePlanProfileId}-start-v3`
      : `live-qualified-envelope-release-gate-${sourcePlanProfileId}-start-v3`,
    sourcePlanProfileId,
    sourceLookIndexOffset,
    sourcePlanSpectrumOpportunities,
  });
}

function possibleBranchSourceLookIndices(
  temporalSchedule: PinnedTemporalSchedule,
  maximumSpectrumOpportunities: number,
  branch: ProductionAcquisitionBranch,
): readonly number[] {
  return Array.from(
    { length: maximumSpectrumOpportunities + (branch === 'qualified-envelope' ? 1 : 0) },
    (_, offset) => temporalSchedule.sourceLookIndexOffset + offset,
  );
}

function auditHeldOutSourceSpan(scenarios: readonly CanonicalClassificationScenario[]) {
  const possibleSourceLookIndices = [
    ...possibleBranchSourceLookIndices(
      PINNED_VALIDATION_SPECTRUM_TEMPORAL_SCHEDULE,
      FULL_BAND_2G4_OBSERVATION_OPPORTUNITIES,
      'consecutive-spectrum',
    ),
    ...possibleBranchSourceLookIndices(
      PINNED_VALIDATION_QUALIFIED_ENVELOPE_TEMPORAL_SCHEDULE,
      FULL_BAND_2G4_OBSERVATION_OPPORTUNITIES,
      'qualified-envelope',
    ),
  ];
  const scenariosAudit = scenarios.map((scenario) => {
    const driftHzPerLook = scenario.parameters.driftHzPerLook ?? 0;
    const maximumAbsoluteDeclaredDriftHz = Math.max(
      ...possibleSourceLookIndices.map((lookIndex) => Math.abs((lookIndex - 4) * driftHzPerLook)),
    );
    const availableCenterDriftMarginHz = Math.max(
      0,
      (scenario.recommendedSpanHz - scenario.occupiedBandwidthHz) / 2,
    );
    const declaredDriftRemainsInSpan = maximumAbsoluteDeclaredDriftHz <= availableCenterDriftMarginHz;
    let minimumInjectedSignalGainDb: number | null = null;
    if (driftHzPerLook !== 0) {
      const actualRbwHz = Math.max(scenario.recommendedSpanHz / (SWEEP_POINTS - 1), 1_000);
      const perLookSignalGainDb = possibleSourceLookIndices.map((lookIndex) => {
        const common = {
          lookIndex,
          seed: NUISANCE_SHIFT_SEEDS[0],
          actualRbwHz,
          detectedPowerSynthesisFilterWidthHz: actualRbwHz,
          points: SWEEP_POINTS,
          sweepTimeSeconds: SWEEP_TIME_SECONDS,
          zeroSpanPoints: ZERO_SPAN_POINTS,
          zeroSpanSamplePeriodSeconds: ZERO_SPAN_SAMPLE_PERIOD_SECONDS,
        } as const;
        const signal = synthesizeCanonicalObservation(scenario.id, { ...common, snrDb: 32 });
        const noiseReference = synthesizeCanonicalObservation(scenario.id, { ...common, snrDb: -120 });
        return Math.max(...signal.powerDbm.map((powerDbm, index) =>
          powerDbm - noiseReference.powerDbm[index]!));
      });
      minimumInjectedSignalGainDb = Math.min(...perLookSignalGainDb);
    }
    const signalVisibilityValid = minimumInjectedSignalGainDb === null || minimumInjectedSignalGainDb > 1;
    return {
      scenarioId: scenario.id,
      driftHzPerLook,
      maximumAbsoluteDeclaredDriftHz,
      availableCenterDriftMarginHz,
      declaredDriftRemainsInSpan,
      minimumInjectedSignalGainDb,
      signalVisibilityValid,
      valid: declaredDriftRemainsInSpan && signalVisibilityValid,
    };
  });
  return {
    sourceLookIndexStart: possibleSourceLookIndices[0]!,
    sourceLookIndexStop: possibleSourceLookIndices.at(-1)!,
    qualification: 'declared-linear-drift-and-injected-signal-visibility-audit-v1',
    valid: scenariosAudit.every((item) => item.valid),
    scenarios: scenariosAudit,
  };
}

function summarizeCausalAcquisitionTraces(
  traces: readonly CausalAcquisitionTrace[],
  expectedBranch: ProductionAcquisitionBranch,
) {
  const violations = traces.filter((trace) => !trace.uniqueSourceLookIndices
    || !trace.strictlyIncreasingSourceLookIndices
    || !trace.captureImmediatelyFollowsTrigger
    || trace.branch !== expectedBranch
    || trace.sourceLookIndices.length !== trace.spectrumSourceLookIndices.length
      + trace.detectedPowerSourceLookIndices.length
    || (expectedBranch === 'consecutive-spectrum'
      ? trace.detectedPowerSourceLookIndices.length !== 0
      : trace.detectedPowerSourceLookIndices.length > 1));
  return {
    branch: expectedBranch,
    sourceClock: expectedBranch === 'consecutive-spectrum'
      ? PINNED_SIGNAL_LAB_PRODUCTION_ACQUISITION_REGIME.sourceClocks.spectrum
      : PINNED_SIGNAL_LAB_PRODUCTION_ACQUISITION_REGIME.sourceClocks.qualifiedEnvelope,
    attempts: traces.length,
    allSourceLookIndicesUniqueWithinAttempt: traces.every((trace) => trace.uniqueSourceLookIndices),
    allSourceLookIndicesStrictlyIncreasingWithinAttempt: traces.every((trace) => trace.strictlyIncreasingSourceLookIndices),
    allCapturesImmediatelyFollowTriggerSpectrum: traces.every((trace) => trace.captureImmediatelyFollowsTrigger),
    maximumDetectedPowerCapturesPerAttempt: Math.max(
      0,
      ...traces.map((trace) => trace.detectedPowerSourceLookIndices.length),
    ),
    attemptsWithDetectedPowerCapture: traces.filter((trace) => trace.detectedPowerSourceLookIndices.length === 1).length,
    attemptsWithoutDetectedPowerCapture: traces.filter((trace) => trace.detectedPowerSourceLookIndices.length === 0).length,
    spectrumAcquisitionCount: numericSummary(traces.map((trace) => trace.spectrumSourceLookIndices.length)),
    detectedPowerAcquisitionCount: numericSummary(traces.map((trace) => trace.detectedPowerSourceLookIndices.length)),
    violationCount: violations.length,
    violations: violations.slice(0, 20),
  };
}

function recomputeTailCalibrationAudit(
  assignments: readonly { scenarioId: string; classId: ObservableLeafClass }[],
): RecomputedTailCalibrationAudit {
  const views = ['spectrum-only', 'envelope-untimed', 'envelope-timed'] as const;
  const scoresByClass = new Map<ObservableLeafClass, Record<TailCalibrationView, number[]>>();
  const recomputedAttemptCountsByScenarioByView: Record<string, Record<TailCalibrationView, number>> = {};
  const spectrumAcquisitionTraces: CausalAcquisitionTrace[] = [];
  const qualifiedEnvelopeAcquisitionTraces: CausalAcquisitionTrace[] = [];
  let lateMinimumCount = 0;
  let allOnlineAttemptCount = 0;
  for (const assignment of assignments) {
    if (assignment.classId === 'unknown-signal') continue;
    const scenario = canonicalClassificationScenarios.find((candidate) => candidate.id === assignment.scenarioId);
    const model = BAYESIAN_OBSERVABLE_MODEL.classModels.find((candidate) => candidate.id === assignment.classId);
    if (!scenario || !model) throw new Error(`Independent tail audit cannot resolve ${assignment.scenarioId}/${assignment.classId}`);
    const classScores = scoresByClass.get(assignment.classId) ?? {
      'spectrum-only': [],
      'envelope-untimed': [],
      'envelope-timed': [],
    };
    scoresByClass.set(assignment.classId, classScores);
    const scenarioAttemptCounts: Record<TailCalibrationView, number> = {
      'spectrum-only': 0,
      'envelope-untimed': 0,
      'envelope-timed': 0,
    };
    for (const snrDb of PINNED_TAIL_CALIBRATION_SNR_DB) {
      for (const acquisitionRegime of PINNED_TAIL_CALIBRATION_ACQUISITION_REGIMES) {
        for (const seed of PINNED_TAIL_CALIBRATION_SEEDS) {
          const actualRbwHz = pinnedCalibrationActualRbwHz(scenario, acquisitionRegime);
          const detectedPowerSynthesisFilterWidthHz =
            pinnedCalibrationDetectedPowerSynthesisFilterWidthHz(actualRbwHz, acquisitionRegime);
          const spectrumSelection = acquireProductionAttempt({
            scenario,
            temporalSchedule: acquisitionRegime.spectrumTemporalSchedule,
            observationHorizon: observationOpportunityHorizon(scenario),
            seed,
            snrDb,
            actualRbwHz,
            detectedPowerSynthesisFilterWidthHz,
            context: `${assignment.scenarioId} tail-calibration consecutive-spectrum`,
            branch: 'consecutive-spectrum',
          });
          const envelopeSelection = acquireProductionAttempt({
            scenario,
            temporalSchedule: acquisitionRegime.qualifiedEnvelopeTemporalSchedule,
            observationHorizon: observationOpportunityHorizon(scenario),
            seed,
            snrDb,
            actualRbwHz,
            detectedPowerSynthesisFilterWidthHz,
            context: `${assignment.scenarioId} tail-calibration qualified-envelope`,
            branch: 'qualified-envelope',
          });
          spectrumAcquisitionTraces.push(spectrumSelection.acquisitionTrace);
          qualifiedEnvelopeAcquisitionTraces.push(envelopeSelection.acquisitionTrace);
          const representativeScores: Record<TailCalibrationView, Array<{
            opportunity: number;
            representativeKey: string;
            support: number;
          }>> = {
            'spectrum-only': [],
            'envelope-untimed': [],
            'envelope-timed': [],
          };
          for (const representative of spectrumSelection.onlineReadyRepresentatives) {
            const spectrumObservation = {
              ...representative.spectrumObservation,
              values: spectrumOnly(representative.spectrumObservation.values),
            };
            if (!observableRepresentativeIsInClassDomain(assignment.classId, spectrumObservation)) continue;
            representativeScores['spectrum-only'].push({
              opportunity: representative.readyOpportunity,
              representativeKey: representative.representativeKey,
              support: Math.max(...observableModelComponents(model, 'spectrum-only').map((component) =>
                studentTModelTailProbability(spectrumObservation.values, component))),
            });
          }
          const capture = envelopeSelection.liveEnvelopeCapture;
          if (capture?.envelopeObservation) {
            for (const view of ['envelope-untimed', 'envelope-timed'] as const) {
              const values = view === 'envelope-untimed'
                ? envelopeUntimed(capture.envelopeObservation.values)
                : capture.envelopeObservation.values;
              const observation = { ...capture.envelopeObservation, values };
              if (!observableRepresentativeIsInClassDomain(assignment.classId, observation)) continue;
              representativeScores[view].push({
                opportunity: capture.representative.firstReadyOpportunity,
                representativeKey: capture.representative.representativeKey,
                support: Math.max(...observableModelComponents(model, view).map((component) =>
                  studentTModelTailProbability(values, component))),
              });
            }
          }
          if (representativeScores['spectrum-only'].length > 0) {
            const minimum = aggregateAttemptMinimum(representativeScores['spectrum-only']);
            classScores['spectrum-only'].push(minimum.minimumSupport);
            scenarioAttemptCounts['spectrum-only'] += 1;
            allOnlineAttemptCount += 1;
            if (minimum.minimumOpportunity > minimum.firstOpportunity
              && minimum.minimumSupport < minimum.firstSupport - Number.EPSILON) lateMinimumCount++;
          }
          for (const view of ['envelope-untimed', 'envelope-timed'] as const) {
            const soleCaptureScore = representativeScores[view][0];
            if (!soleCaptureScore) continue;
            if (representativeScores[view].length !== 1) {
              throw new Error(`${assignment.scenarioId} tail-calibration ${view} synthesized more than the sole live capture`);
            }
            classScores[view].push(soleCaptureScore.support);
            scenarioAttemptCounts[view] += 1;
          }
        }
      }
    }
    recomputedAttemptCountsByScenarioByView[scenario.id] = scenarioAttemptCounts;
  }

  const scoreComparisons = BAYESIAN_OBSERVABLE_MODEL.classModels
    .filter((model) => model.id !== 'unknown-signal')
    .flatMap((model) => views.map((view) => {
      const expected = [...(model.tailCalibrationScoresByView?.[view] ?? [])];
      const observed = [...(scoresByClass.get(model.id)?.[view] ?? [])].sort((left, right) => left - right);
      const maximumAbsoluteDifference = expected.length === observed.length
        ? Math.max(0, ...expected.map((value, index) => Math.abs(value - observed[index]!)))
        : Number.MAX_VALUE;
      return {
        classId: model.id,
        view,
        expectedCount: expected.length,
        observedCount: observed.length,
        maximumAbsoluteDifference,
        expectedSha256: sha256Canonical(expected),
        observedSha256: sha256Canonical(observed),
      };
    }));
  const modelAttemptCounts =
    BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.tailCalibrationAttemptCountsByScenarioByView ?? {};
  const attemptCountMismatches = [...new Set([
    ...Object.keys(modelAttemptCounts),
    ...Object.keys(recomputedAttemptCountsByScenarioByView),
  ])].sort().flatMap((scenarioId) => views.flatMap((view) => {
    const expected = modelAttemptCounts[scenarioId]?.[view] ?? 0;
    const observed = recomputedAttemptCountsByScenarioByView[scenarioId]?.[view] ?? 0;
    return expected === observed ? [] : [{ scenarioId, view, expected, observed }];
  }));
  const aggregationRegressionResult = aggregateAttemptMinimum([
    { opportunity: 8, representativeKey: 'first-ready', support: 0.8 },
    { opportunity: 9, representativeKey: 'later-online', support: 0.2 },
  ]);
  const aggregationRegression = {
    firstOpportunity: aggregationRegressionResult.firstOpportunity,
    minimumOpportunity: aggregationRegressionResult.minimumOpportunity,
    minimumSupport: aggregationRegressionResult.minimumSupport,
    passed: aggregationRegressionResult.firstOpportunity === 8
      && aggregationRegressionResult.minimumOpportunity === 9
      && aggregationRegressionResult.minimumSupport === 0.2,
  };
  const runtimeBranchClockAudits = {
    consecutiveSpectrum: summarizeCausalAcquisitionTraces(
      spectrumAcquisitionTraces,
      'consecutive-spectrum',
    ),
    qualifiedEnvelope: summarizeCausalAcquisitionTraces(
      qualifiedEnvelopeAcquisitionTraces,
      'qualified-envelope',
    ),
  };
  const valid = attemptCountMismatches.length === 0
    && scoreComparisons.every((comparison) => comparison.expectedCount === comparison.observedCount
      && comparison.maximumAbsoluteDifference <= TAIL_CALIBRATION_NUMERICAL_TOLERANCE)
    && lateMinimumCount > 0
    && aggregationRegression.passed
    && runtimeBranchClockAudits.consecutiveSpectrum.violationCount === 0
    && runtimeBranchClockAudits.consecutiveSpectrum.maximumDetectedPowerCapturesPerAttempt === 0
    && runtimeBranchClockAudits.qualifiedEnvelope.violationCount === 0
    && runtimeBranchClockAudits.qualifiedEnvelope.maximumDetectedPowerCapturesPerAttempt <= 1;
  return {
    valid,
    scoreTolerance: TAIL_CALIBRATION_NUMERICAL_TOLERANCE,
    recomputedAttemptCountsByScenarioByView,
    attemptCountMismatches,
    scoreComparisons,
    lateMinimumCount,
    allOnlineAttemptCount,
    runtimeBranchClockAudits,
    aggregationRegression,
  };
}

function aggregateAttemptMinimum(
  values: readonly { opportunity: number; representativeKey: string; support: number }[],
): { firstOpportunity: number; firstSupport: number; minimumOpportunity: number; minimumSupport: number } {
  if (values.length === 0) throw new Error('Tail calibration attempt minimum requires an online-ready representative');
  const ordered = [...values].sort((left, right) => left.opportunity - right.opportunity
    || left.representativeKey.localeCompare(right.representativeKey));
  const first = ordered[0]!;
  const minimum = ordered.reduce((selected, candidate) => candidate.support < selected.support ? candidate : selected, first);
  return {
    firstOpportunity: first.opportunity,
    firstSupport: first.support,
    minimumOpportunity: minimum.opportunity,
    minimumSupport: minimum.support,
  };
}

function sha256Canonical(value: unknown): string {
  return createHash('sha256').update(JSON.stringify(value)).digest('hex');
}

function independentDetectedPowerCapturePayloadSha256(
  capture: ZeroSpanCapture,
): string {
  const canonical = independentCanonicalJson(capture, '$', new Set<object>());
  return createHash('sha256')
    .update(`tinysa-detected-power-capture-payload-v1\0${canonical}`)
    .digest('hex');
}

function independentCanonicalJson(
  value: unknown,
  path: string,
  ancestors: Set<object>,
): string {
  if (value === null) return 'null';
  if (typeof value === 'string' || typeof value === 'boolean') {
    return JSON.stringify(value);
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      throw new Error(`Validator cannot canonicalize non-finite capture value at ${path}`);
    }
    return JSON.stringify(value);
  }
  if (typeof value !== 'object') {
    throw new Error(`Validator cannot canonicalize ${typeof value} capture value at ${path}`);
  }
  if (ancestors.has(value)) {
    throw new Error(`Validator cannot canonicalize cyclic capture value at ${path}`);
  }
  ancestors.add(value);
  try {
    if (Array.isArray(value)) {
      return `[${value.map((item, index) => {
        if (item === undefined) {
          throw new Error(`Validator cannot canonicalize undefined capture array value at ${path}[${index}]`);
        }
        return independentCanonicalJson(item, `${path}[${index}]`, ancestors);
      }).join(',')}]`;
    }
    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) {
      throw new Error(`Validator cannot canonicalize non-plain capture object at ${path}`);
    }
    if (Object.getOwnPropertySymbols(value).length > 0) {
      throw new Error(`Validator cannot canonicalize symbol-keyed capture value at ${path}`);
    }
    const record = value as Record<string, unknown>;
    return `{${Object.keys(record)
      .filter((key) => record[key] !== undefined)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${independentCanonicalJson(record[key], `${path}.${key}`, ancestors)}`)
      .join(',')}}`;
  } finally {
    ancestors.delete(value);
  }
}

function modelTruth(truth: ObservableSignalClass): ObservableLeafClass {
  if (truth === 'bluetooth-classic-like' || truth === 'bluetooth-le-like') return 'bluetooth-like';
  if (OBSERVABLE_LEAF_CLASSES.includes(truth as ObservableLeafClass)) return truth as ObservableLeafClass;
  throw new Error(`Corpus truth ${truth} has no current observable-model mapping`);
}

function acceptsTruth(
  result: string,
  truth: ObservableLeafClass,
  nominalBandwidthHz: number,
  measuredBandwidthHz: number,
): boolean {
  if (truth === 'unknown-signal') return result === 'unknown';
  if (result === `observable:${truth}`) return true;
  if ((truth === 'lte-fdd-like' || truth === 'lte-tdd-like') && result === 'observable:lte-like') return true;
  if ((truth === 'nr-fdd-like' || truth === 'nr-tdd-like') && result === 'observable:nr-like') return true;
  if ((truth === 'lte-fdd-like' || truth === 'lte-tdd-like' || truth === 'nr-fdd-like' || truth === 'nr-tdd-like')
    && result === 'observable:cellular-ofdm-ambiguous') {
    return nominalBandwidthHz <= 25_000_000 && measuredBandwidthHz <= 25_000_000;
  }
  if ((truth === 'wifi-hr-dsss-like' || truth === 'wifi-ofdm-like') && result === 'observable:wifi-like') return true;
  return false;
}

function acceptsAnyTruth(
  result: string,
  allowedTruths: readonly ObservableLeafClass[],
  nominalBandwidthHz: number,
  measuredBandwidthHz: number,
): boolean {
  return allowedTruths.some((truth) => acceptsTruth(result, truth, nominalBandwidthHz, measuredBandwidthHz));
}

function admissionSummary(values: readonly AdmissionAttempt[]) {
  const everReady = values.filter((item) => item.everReady);
  const admitted = values.filter((item) => item.admitted);
  return {
    attempted: values.length,
    everReady: everReady.length,
    everReadyRate: fraction(values, (item) => item.everReady),
    firstReady: admitted.length,
    firstReadyRate: fraction(values, (item) => item.admitted),
    admitted: admitted.length,
    misses: values.length - admitted.length,
    admissionRate: fraction(values, (item) => item.admitted),
    observationHorizons: counts(values.map((item) => String(item.observationHorizon))),
    everReadyRepresentativeCount: numericSummary(values.map((item) => item.everReadyRepresentativeCount)),
    firstReadyRepresentativeCount: numericSummary(values.map((item) => item.firstReadyRepresentativeCount)),
    provenanceUnavailableWindowCount: values.reduce(
      (sum, item) => sum + item.provenanceUnavailableWindowCount,
      0,
    ),
    finalReadyRepresentativeCount: numericSummary(values.map((item) => item.finalReadyRepresentativeCount)),
    finalActiveRepresentativeCount: numericSummary(values.map((item) => item.finalActiveRepresentativeCount)),
    selectedTrackAdmissions: numericSummary(admitted.map((item) => item.selectedTrackAdmissions)),
    maximumActiveAdmissions: numericSummary(values.map((item) => item.maximumActiveAdmissions)),
    maximumLocalTrackAdmissions: numericSummary(values.map((item) => item.maximumLocalTrackAdmissions)),
    firstReadyOpportunity: numericSummary(values.flatMap((item) => item.firstReadyOpportunity === undefined ? [] : [item.firstReadyOpportunity])),
    everAssociationModes: counts(values.flatMap((item) => item.everAssociationModes)),
    finalAssociationModes: counts(values.flatMap((item) => item.finalAssociationModes)),
    regularAssociationsObserved: values.reduce((sum, item) => sum + item.regularAssociationsObserved, 0),
    agileAssociationsObserved: values.reduce((sum, item) => sum + item.agileAssociationsObserved, 0),
    regularAssociationExpirations: values.reduce((sum, item) => sum + item.regularAssociationExpirations, 0),
  };
}

function expectedCalibrationError(values: readonly { confidence: number; correct: boolean }[], bins: number): number {
  if (!values.length) return Number.NaN;
  let result = 0;
  for (let bin = 0; bin < bins; bin++) {
    const lower = bin / bins;
    const upper = (bin + 1) / bins;
    const selected = values.filter((item) => item.confidence >= lower && (bin === bins - 1 ? item.confidence <= upper : item.confidence < upper));
    if (!selected.length) continue;
    result += selected.length / values.length * Math.abs(mean(selected.map((item) => item.confidence)) - fraction(selected, (item) => item.correct));
  }
  return result;
}

function auroc(values: readonly { score: number; positive: boolean }[]): number {
  const positive = values.filter((item) => item.positive);
  const negative = values.filter((item) => !item.positive);
  if (!positive.length || !negative.length) return Number.NaN;
  let wins = 0;
  for (const left of positive) for (const right of negative) wins += left.score > right.score ? 1 : left.score === right.score ? 0.5 : 0;
  return wins / (positive.length * negative.length);
}

function counts(values: readonly string[]): Record<string, number> { return values.reduce<Record<string, number>>((result, value) => ({ ...result, [value]: (result[value] ?? 0) + 1 }), {}); }
function componentSourceScenarioId(component: Readonly<{ id: string; sourceScenarioId?: string }>): string {
  return component.sourceScenarioId ?? component.id;
}
function duplicateStrings(values: readonly string[]): string[] {
  return [...new Set(values.filter((value, index) => values.indexOf(value) !== index))].sort();
}
function setDifference(left: readonly string[], right: readonly string[]): string[] {
  const rightSet = new Set(right);
  return [...new Set(left.filter((value) => !rightSet.has(value)))].sort();
}
function numericIntersection(left: readonly number[], right: readonly number[]): number[] {
  const rightSet = new Set(right);
  return [...new Set(left.filter((value) => rightSet.has(value)))].sort((a, b) => a - b);
}
function numericSummary(values: readonly number[]): { minimum: number; median: number; maximum: number } | undefined {
  if (!values.length) return undefined;
  const ordered = [...values].sort((left, right) => left - right);
  return { minimum: ordered[0]!, median: ordered[Math.floor(ordered.length / 2)]!, maximum: ordered.at(-1)! };
}
function fraction<T>(values: readonly T[], predicate: (value: T) => boolean): number { return values.length ? values.filter(predicate).length / values.length : 0; }
function mean(values: readonly number[]): number { return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : Number.NaN; }
function spectrumOnly(values: Readonly<Record<string, number>>): Readonly<Record<string, number>> { return Object.fromEntries(Object.entries(values).filter(([name]) => !name.startsWith('envelope.'))); }
function envelopeUntimed(values: Readonly<Record<string, number>>): Readonly<Record<string, number>> { return Object.fromEntries(Object.entries(values).filter(([name]) => !name.startsWith('envelope.periodicEnergy') && name !== 'envelope.logTransitionRateHz')); }
