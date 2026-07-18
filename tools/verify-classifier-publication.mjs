#!/usr/bin/env node

import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const REPOSITORY_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
// The model and its manifest live in this repo (src/models); only the
// published-claims docs they're checked against live in the sibling
// Atomizer repo.
const ATOMIZER_ROOT = resolve(REPOSITORY_ROOT, '../Atom-Atomizer');
const MODEL_PATH = 'src/models/bayesian-observable.generated.ts';
const MANIFEST_PATH = 'src/models/bayesian-observable.manifest.generated.ts';
const REPORT_PATH = '.artifacts/classifier-validation/report.json';
const NODE_VERSION_PATH = '.node-version';
const PINNED_SIGNAL_LAB_COMMIT = 'e7d48afbce7165fa04fd551629891123f3b86d34';
const PINNED_CORPUS_SHA256 = 'd68c151f6f284b14effd28bd3db2a696b095ed4fe72a4a206ccea22f54a10a48';
const PINNED_CORPUS_VERSION = 'observable-scalar-corpus-v13';
const PINNED_MODEL_ID = 'bayesian-observable-equivalence-v9';
const PINNED_PREPROCESSING_ID = 'scalar-observable-features-v7';
const PINNED_PRIOR_ID = 'engineering-design-class-weights-v1';
const PINNED_CALIBRATION_ID = 'synthetic-independent-branch-view-matched-causal-acquisition-support-rank-detector-conditioned-physical-uncalibrated-v20';
const PINNED_DECISION_POLICY_ID = 'observable-open-set-decision-v10';
const PINNED_ACCEPTANCE_POLICY_ID = 'synthetic-observable-classifier-full-corpus-release-gates-v1';
const PINNED_ACCEPTANCE_THRESHOLDS = {
  hierarchicalAccuracy: 0.95,
  knownTopLeafAccuracy: 0.85,
  knownCoverage: 0.95,
  minimumHighSnrKnownClassHierarchicalAccuracy: 0.90,
  fittedTemplateLogLoss: 0.5,
  fittedTemplateMulticlassBrier: 0.2,
  fittedTemplateExpectedCalibrationError: 0.1,
  fittedUnknownPosteriorAuroc: 0.90,
  strictTypicalityAuroc: 0.90,
  strictUnknownRejectionRate: 1,
  exactEquivalenceCompatibleRate: 1,
};
const PINNED_ROLLING_ACCEPTANCE_THRESHOLDS = {
  overallKnownCoverage: 0.95,
  overallHierarchicalAccuracy: 0.95,
  perScenarioKnownCoverage: 0.90,
  perScenarioHierarchicalAccuracy: 0.90,
};
const PINNED_HIGH_SNR_SEED_COVERAGE_SNR_DB = [24, 32];
const PINNED_ORDINARY_KNOWN_SEED_COVERAGE = 1;
const PINNED_BLE_ADVERTISING_SEED_COVERAGE = 0.5;
const PINNED_EXPECTED_CLASSIFICATION_NON_ADMISSION_SCENARIO_IDS = [
  'gsm-900-tdma',
];
const PINNED_SELECTION_POLICY =
  'independent-consecutive-spectrum-and-integrated-excess-rank-0-runtime-admission-qualified-envelope-branches-v9';
const PINNED_LIKELIHOOD_POPULATION_POLICY =
  'independent-branch-view-matched-runtime-event-populations-v3';
const PINNED_LIKELIHOOD_COMPONENT_DECOMPOSITION_POLICY = {
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
};
const PINNED_REPRESENTATIVE_WEIGHTING_POLICY =
  'view-matched-spectrum-event-envelope-causal-attempt-weighting-v4';
const PINNED_ACQUISITION_BRANCH_POLICY =
  'independent-no-auto-spectrum-and-qualified-rank-0-integrated-excess-envelope-sessions-v2';
const PINNED_DETECTED_POWER_ACQUISITION_QUALIFICATION =
  'receipt-verified-provenance-bound-runtime-admitted-physical-capture-v5';
const PINNED_DETECTED_POWER_SELECTION_CONDITION =
  'automatic-current-source-sweep-integrated-excess-rank-0';
const PINNED_CAPTURE_TARGET_RANK_MODEL =
  'current-source-sweep-integrated-excess-power-v1';
const PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORING_POLICY = {
  id: 'frequency-agile-fixed-tune-envelope-censoring-v1',
  associationMode: 'frequency-agile-2g4-activity',
  runtimeCapturePolicy: 'validate-receipt-and-capture-before-censoring-v1',
  classifierEvidencePolicy: 'spectrum-only-no-detected-power-envelope-v1',
  unsupportedModelViewPolicy: 'exact-empty-components-and-calibration-v1',
};
const PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORING_LIMITATION =
  'frequency-agile-fixed-tune-envelope-censored';
const PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORED_SCENARIO_IDS = [
  'bluetooth-classic-connected',
  'bluetooth-le-advertising',
];
const PINNED_COMPONENT_SOURCE_SCENARIO_COUNTS_BY_VIEW = {
  'spectrum-only': 18,
  'envelope-untimed': 16,
  'envelope-timed': 16,
};
const PINNED_LIKELIHOOD_COMPONENT_COUNTS_BY_VIEW = {
  'spectrum-only': 28,
  'envelope-untimed': 26,
  'envelope-timed': 26,
};
const PINNED_EMPTY_ARRAY_SHA256 = createHash('sha256')
  .update(JSON.stringify([]))
  .digest('hex');
const PINNED_TRAINING_RUNTIME_IDENTITY = {
  policyId: 'exact-repository-node-version-v1',
  nodeVersion: '22.23.1',
  v8Version: '12.4.254.21-node.56',
};
const PINNED_CAPTURE_TARGET_SELECTION_POLICY =
  'preferred-then-current-source-sweep-integrated-excess-power-physical-or-qualified-agile-member-target-v4';
const PINNED_RELEASE_GATE_PROFILE_HORIZONS = [
  ['cw', 32],
  ['am', 32],
  ['fm', 32],
  ['gsm-900-loaded-bcch', 32],
  ['lte-band3-fdd-20m', 32],
  ['lte-band38-tdd-10m', 32],
  ['nr-n3-fdd-20m', 32],
  ['nr-n78-tdd-100m', 32],
  ['wifi-hr-dsss-11m', 32],
  ['wifi-ofdm-20m', 32],
  ['bluetooth-classic-connected', 96],
  ['bluetooth-le-advertising', 96],
];
let nextSpectrumReleaseGateSourceLookIndex = 0;
const PINNED_SPECTRUM_RELEASE_GATE_SOURCE_PLAN = PINNED_RELEASE_GATE_PROFILE_HORIZONS.map(
  ([profileId, spectrumOpportunities], profileOrdinal) => {
    const profile = {
      profileId,
      profileOrdinal,
      sourceLookIndexOffset: nextSpectrumReleaseGateSourceLookIndex,
      spectrumOpportunities,
      automaticDetectedPowerCaptures: 0,
    };
    nextSpectrumReleaseGateSourceLookIndex += spectrumOpportunities;
    return profile;
  },
);
let nextQualifiedEnvelopeReleaseGateSourceLookIndex = 0;
const PINNED_QUALIFIED_ENVELOPE_RELEASE_GATE_SOURCE_PLAN =
  PINNED_RELEASE_GATE_PROFILE_HORIZONS.map(
    ([profileId, spectrumOpportunities], profileOrdinal) => {
      const profile = {
        profileId,
        profileOrdinal,
        sourceLookIndexOffset: nextQualifiedEnvelopeReleaseGateSourceLookIndex,
        spectrumOpportunities,
        admittedDetectedPowerCaptures: 1,
      };
      nextQualifiedEnvelopeReleaseGateSourceLookIndex += spectrumOpportunities + 1;
      return profile;
    },
  );
const PINNED_PRODUCTION_SPECTRUM_TEMPORAL_SCHEDULES =
  PINNED_SPECTRUM_RELEASE_GATE_SOURCE_PLAN.map((profile) => ({
    id: `live-spectrum-release-gate-${profile.profileId}-start-v3`,
    sourcePlanProfileId: profile.profileId,
    sourceLookIndexOffset: profile.sourceLookIndexOffset,
    sourcePlanSpectrumOpportunities: profile.spectrumOpportunities,
  }));
const PINNED_PRODUCTION_QUALIFIED_ENVELOPE_TEMPORAL_SCHEDULES =
  PINNED_QUALIFIED_ENVELOPE_RELEASE_GATE_SOURCE_PLAN.map((profile) => ({
    id: `live-qualified-envelope-release-gate-${profile.profileId}-start-v3`,
    sourcePlanProfileId: profile.profileId,
    sourceLookIndexOffset: profile.sourceLookIndexOffset,
    sourcePlanSpectrumOpportunities: profile.spectrumOpportunities,
  }));
const PINNED_PRODUCTION_TEMPORAL_SCHEDULE_PAIRS =
  PINNED_PRODUCTION_SPECTRUM_TEMPORAL_SCHEDULES.map((spectrumTemporalSchedule, index) => ({
    id: `live-release-gate-independent-branches-${spectrumTemporalSchedule.sourcePlanProfileId}-v3`,
    sourcePlanProfileId: spectrumTemporalSchedule.sourcePlanProfileId,
    spectrumTemporalSchedule,
    qualifiedEnvelopeTemporalSchedule:
      PINNED_PRODUCTION_QUALIFIED_ENVELOPE_TEMPORAL_SCHEDULES[index],
  }));
const PINNED_PRODUCTION_TEMPORAL_SCHEDULE_PAIR_IDS =
  PINNED_PRODUCTION_TEMPORAL_SCHEDULE_PAIRS.map((pair) => pair.id);
const PINNED_VALIDATION_SPECTRUM_TEMPORAL_SCHEDULE = {
  id: 'held-out-validation-consecutive-spectrum-first-post-live-index-512-v3',
  sourcePlanProfileId: 'held-out-validation',
  sourceLookIndexOffset: 512,
  sourcePlanSpectrumOpportunities: 96,
};
const PINNED_VALIDATION_QUALIFIED_ENVELOPE_TEMPORAL_SCHEDULE = {
  id: 'held-out-validation-qualified-envelope-first-post-live-index-524-v3',
  sourcePlanProfileId: 'held-out-validation',
  sourceLookIndexOffset: 524,
  sourcePlanSpectrumOpportunities: 96,
};
const PINNED_DETECTED_POWER_SYNTHESIS_FILTER_POLICY = {
  id: 'explicit-generator-filter-width-by-acquisition-regime-v1',
  divisorAcquisitionRegimes: 'match-swept-spectrum-actual-rbw-nuisance-v1',
  signalLabProductionAcquisitionRegimes: 'fixed-generator-internal-width-v1',
  signalLabProductionSynthesisFilterWidthHz: 100_000,
  measurementActualRbwQualification: 'unavailable',
};
const PINNED_PRODUCTION_DETECTION_CONFIG = {
  threshold: { strategy: 'noise-relative', marginDb: 10 },
  minimumBandwidthHz: 0,
  minimumProminenceDb: 6,
  minimumConsecutiveSweeps: 2,
  releaseAfterMissedSweeps: 2,
};
const PINNED_PRODUCTION_GEOMETRY_ID = 'signal-lab-recommended-span-450-point-grid-v1';
const PINNED_PRODUCTION_ACQUISITION_REGIME = {
  id: 'signal-lab-recommended-span-grid-with-independent-production-branch-source-clocks-v5',
  geometry: {
    id: PINNED_PRODUCTION_GEOMETRY_ID,
    sourceKind: 'signal-lab',
    kind: 'recommended-span-inclusive-grid',
    sweepPoints: 450,
    spanPolicy: 'canonical-recommended-span-v1',
    resolutionScalePolicy: 'recommended-span-divided-by-points-minus-one-v1',
  },
  branchPolicy: PINNED_ACQUISITION_BRANCH_POLICY,
  sourceClocks: {
    spectrum: {
      id: 'shared-monotonic-source-clock-v1',
      acquisitionIndexPolicy: 'one-look-index-per-physical-acquisition-v1',
      detectedPowerCapturePolicy: 'no-automatic-detected-power-capture-v1',
    },
    qualifiedEnvelope: {
      id: 'shared-monotonic-source-clock-v1',
      acquisitionIndexPolicy: 'one-look-index-per-physical-acquisition-v1',
      detectedPowerCapturePolicy: 'capture-once-after-rank-0-integrated-excess-current-target-runtime-admission-v3',
      captureTargetSelectionPolicy: PINNED_CAPTURE_TARGET_SELECTION_POLICY,
      postCaptureSpectrumPolicy: 'continue-at-next-shared-look-index-v1',
    },
  },
  spectrumReleaseGateSourcePlan: PINNED_SPECTRUM_RELEASE_GATE_SOURCE_PLAN,
  qualifiedEnvelopeReleaseGateSourcePlan: PINNED_QUALIFIED_ENVELOPE_RELEASE_GATE_SOURCE_PLAN,
  temporalSchedulePairs: PINNED_PRODUCTION_TEMPORAL_SCHEDULE_PAIRS,
  componentFitIncluded: true,
  tailCalibrationIncluded: true,
};
const PINNED_ACQUISITION_REGIME_IDS = [
  ...[12, 20, 35, 55, 80, 120].map((divisor) =>
    `occupied-bandwidth-rbw-divisor:${divisor}/independent-production-branch-baselines-v1`),
  ...PINNED_PRODUCTION_TEMPORAL_SCHEDULE_PAIR_IDS.map((pairId) =>
    `${PINNED_PRODUCTION_GEOMETRY_ID}/${pairId}`),
];
const PINNED_TAIL_SCORE_TOLERANCE = 1e-12;
const PINNED_KNOWN_CLASS_IDS = [
  'cw-like',
  'am-dsb-full-carrier-like',
  'fm-angle-modulated-like',
  'gsm-like',
  'lte-fdd-like',
  'lte-tdd-like',
  'nr-fdd-like',
  'nr-tdd-like',
  'wifi-hr-dsss-like',
  'wifi-ofdm-like',
  'bluetooth-like',
];
const PINNED_CLASS_IDS = [...PINNED_KNOWN_CLASS_IDS, 'unknown-signal'];
const PINNED_DIMENSIONS = [
  'association.logBayesFactor',
  'envelope.duty',
  'envelope.logTransitionRateHz',
  'envelope.periodicEnergy100Hz',
  'envelope.periodicEnergy1600Hz',
  'envelope.periodicEnergy1733Hz',
  'envelope.periodicEnergy2000Hz',
  'envelope.periodicEnergy200Hz',
  'envelope.rangeDb',
  'envelope.standardDeviationDb',
  'envelope.tuneOffsetFraction',
  'history.bleAdvertisingScore',
  'history.peakSpanFraction',
  'history.raster1MHzScore',
  'history.raster2MHzScore',
  'spectrum.centerFraction',
  'spectrum.centerNotch',
  'spectrum.entropy',
  'spectrum.flatness',
  'spectrum.logBandwidthHz',
  'spectrum.logBandwidthRbwRatio',
  'spectrum.logClusterCount',
  'spectrum.peakDensity',
  'spectrum.peakDriftFraction',
  'spectrum.powerVariationDb',
  'spectrum.prominenceDb',
  'spectrum.sidebandScore',
  'spectrum.symmetry',
];
const PINNED_COMPONENT_COUNT_PER_VIEW = 28;
const PINNED_SOURCE_SCENARIO_COUNT_PER_VIEW = 18;
const PINNED_MINIMUM_DECOMPOSED_MODE_FIT_SAMPLE_COUNT =
  PINNED_LIKELIHOOD_COMPONENT_DECOMPOSITION_POLICY.minimumModeFitSampleCount;
const PINNED_CSMA_DECOMPOSED_SOURCE_SCENARIO_IDS = [
  'unknown-802154',
  'wifi-hr-dsss-11m',
  'wifi-ofdm-20m',
  'wifi-ofdm-40m',
  'wifi-ofdm-80m',
];
const PINNED_COMPONENT_DEGREES_OF_FREEDOM = 7;
const PINNED_NON_COMPONENT_FULL_CORPUS_SCENARIO_IDS = [
  'unknown-impulsive',
  'unknown-chirp',
  'unknown-regular-cw-comb-4',
  'unknown-regular-cw-comb-5',
  'unknown-irregular-cw-multitone-100-210-370k',
  'unknown-stationary-intermittent-2g4',
  'unknown-simultaneous-1mhz-raster-2g4',
  'unknown-interleaved-four-channel-2g4',
  'unknown-proprietary-off-raster-fhss-2g4',
  'unknown-instrument-spur-rbw-line',
  'unknown-independent-am-equivalent-three-tone',
  'unknown-independent-fm-equivalent-bessel-comb',
  'unknown-generic-ofdm-20m',
  'unknown-generic-tdd-ofdm-10m',
  'unknown-generic-ofdm-80m',
  'unknown-proprietary-dsss-22m',
  'gsm-900-tdma',
];
const PINNED_FULL_CORPUS_SCENARIO_COUNT = 35;
const PINNED_ASSOCIATION_MODES = [
  'frequency-local',
  'frequency-agile-2g4-activity',
  'regular-spectral-component-activity',
  'multicomponent-swept-region-activity',
];
// Agile activity can own the projected classifier evidence, but never the
// physical actuation row. The raw target remains one of these non-synthetic
// tracker modes while its qualified agile summary may own the evidence view.
const PINNED_PHYSICAL_CAPTURE_ASSOCIATION_MODES = [
  'frequency-local',
  'regular-spectral-component-activity',
  'multicomponent-swept-region-activity',
];
const PINNED_CAPTURE_PROJECTION_KINDS = [
  'current-active-physical-representative',
  'current-qualified-agile-latest-member',
];
const PINNED_RAW_CAPTURE_TARGET_STATES = ['active', 'candidate'];
const PINNED_REPRESENTATIVE_ELIGIBILITY_POLICY = 'observation-only-hypothesis-domain-v5';
const PINNED_MINIMUM_KNOWN_SYNTHETIC_SUPPORT_RANK = 0.025;
const PINNED_PRIOR_VARIANT_IDS = [
  'engineering-baseline-v1',
  'unknown-mass-0.10-known-ratios-preserved-v1',
  'unknown-mass-0.30-known-ratios-preserved-v1',
  'cellular-family-up-within-family-ratios-preserved-v1',
  'unlicensed-families-up-within-family-ratios-preserved-v1',
];
const PINNED_PRIOR_SENSITIVITY_GATES = {
  minimumKnownCoverage: 0.85,
  minimumHierarchicalAccuracy: 0.90,
  maximumIncompatibleNonUnknownRisk: 0,
  maximumFalseAcceptedUnknownRisk: 0,
  maximumDecisionChangeRate: 0.20,
};
const PINNED_ENGINEERING_PRIOR = {
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
};
const STALE_PUBLICATION_VALUES = [
  ['bayesian-observable-equivalence-v8', 'stale pre-integrated-excess model ID'],
  ['synthetic-independent-branch-view-matched-causal-acquisition-support-rank-detector-conditioned-physical-uncalibrated-v19', 'stale pre-integrated-excess calibration ID'],
  ['independent-consecutive-spectrum-and-strongest-first-admission-qualified-envelope-branches-v8', 'stale strongest-current representative-selection policy'],
  ['independent-no-auto-spectrum-and-qualified-first-admitted-envelope-sessions-v1', 'stale first-admitted acquisition-branch policy'],
  ['receipt-verified-provenance-bound-first-runtime-admitted-strongest-current-physical-or-agile-member-single-capture-v4', 'stale strongest-current detected-power qualification'],
  ['preferred-then-strongest-current-physical-or-qualified-agile-member-target-v3', 'stale strongest-current capture-target policy'],
  ['capture-once-after-first-runtime-admitted-strongest-current-target-v2', 'stale strongest-current capture policy'],
  ['consecutive-spectrum-all-runtime-representatives-and-independent-qualified-envelope-sole-capture-v4', 'stale pre-integrated-excess tail representative policy'],
  ['signal-lab-recommended-span-grid-with-independent-production-branch-source-clocks-v4', 'stale pre-integrated-excess acquisition regime'],
  ['701fdf3f5f959327369bc299dbc5a45fdf8666d40e65d57df50558b5db67c9dd', 'stale provisional pre-v19 model asset SHA-256'],
  ['pending fresh v19 regeneration', 'stale pending-v19 publication wording'],
  ['Until fresh v19 regeneration completes', 'stale pending-v19 publication wording'],
  ['Validation statement pending a fresh v19 report', 'stale pending-v19 publication wording'],
  ['superseded pre-v19', 'stale superseded-regression publication wording'],
  ['report file is currently unavailable', 'stale missing-report publication wording'],
  ['unavailable as current release evidence until a fresh', 'stale unavailable-release-evidence wording'],
  ['provisional asset has passed independent regeneration', 'stale provisional-regeneration wording'],
  ['must replace those provisional values before publication', 'stale provisional-replacement wording'],
  ['Calibration v8', 'stale shorthand calibration version'],
  ['Calibration v9', 'stale shorthand calibration version'],
  ['Calibration v10', 'stale shorthand calibration version'],
  ['Calibration v11', 'stale shorthand calibration version'],
  ['Calibration v12', 'stale shorthand calibration version'],
  ['Calibration v13', 'stale shorthand calibration version'],
  ['Calibration v14', 'stale shorthand calibration version'],
  ['Calibration v15', 'stale shorthand calibration version'],
  ['synthetic-view-matched-stratified-online-attempt-min-support-rank-detector-conditioned-physical-uncalibrated-v10', 'stale calibration ID'],
  ['synthetic-view-matched-causal-acquisition-support-rank-detector-conditioned-physical-uncalibrated-v11', 'stale calibration ID'],
  ['synthetic-view-matched-causal-acquisition-support-rank-detector-conditioned-physical-uncalibrated-v12', 'stale calibration ID'],
  ['synthetic-independent-branch-view-matched-causal-acquisition-support-rank-detector-conditioned-physical-uncalibrated-v13', 'stale calibration ID'],
  ['synthetic-independent-branch-view-matched-causal-acquisition-support-rank-detector-conditioned-physical-uncalibrated-v14', 'stale calibration ID'],
  ['synthetic-independent-branch-view-matched-causal-acquisition-support-rank-detector-conditioned-physical-uncalibrated-v15', 'stale calibration ID'],
  ['synthetic-independent-branch-view-matched-causal-acquisition-support-rank-detector-conditioned-physical-uncalibrated-v16', 'stale calibration ID'],
  ['bayesian-observable-equivalence-v7', 'stale likelihood-architecture model ID'],
  ['provenance-bound-first-runtime-admitted-strongest-current-single-capture-v2', 'stale self-attested detected-power acquisition qualification'],
  ['observable-open-set-decision-v9', 'stale decision-policy ID'],
  ['preferred-then-strongest-current-tracker-target-v1', 'stale capture-target-selection policy'],
  ['causal-first-admitted-single-envelope-all-online-spectrum-v4', 'stale representative-selection policy'],
  ['independent-consecutive-spectrum-and-strongest-first-admission-qualified-envelope-branches-v6', 'stale representative-selection policy'],
  ['view-matched-runtime-event-populations-v1', 'stale likelihood-population policy'],
  ['independent-branch-view-matched-runtime-event-populations-v2', 'stale likelihood-population policy'],
  ['one-causal-acquisition-attempt-score-per-evidence-view-v3', 'stale tail score-unit policy'],
  ['all-runtime-admitted-spectrum-representatives-and-sole-live-envelope-representative-v2', 'stale tail representative-selection policy'],
  ['consecutive-spectrum-all-runtime-representatives-and-independent-qualified-envelope-sole-capture-v3', 'stale tail representative-selection policy'],
  ['spectrum-minimum-envelope-sole-capture-v4', 'stale tail aggregation policy'],
  ['spectrum-single-rank-dominates-attempt-min-envelope-rank-is-sole-capture-v2', 'stale tail runtime-interpretation policy'],
  ['signal-lab-recommended-span-grid-with-causal-shared-source-clock-v2', 'stale production acquisition regime'],
  ['live-release-gate-cw-start-v2', 'stale production temporal schedule'],
  ['held-out-validation-first-post-live-index-524-v2', 'stale held-out temporal schedule'],
  ['equal-weight-per-causal-live-envelope-acquisition-attempt-v3', 'stale representative-weighting policy'],
  ['contiguous-zero, post-eight-capture-skip, and full-matrix offset-225', 'stale fixed-skip source-clock prose'],
  ['observation-only-hypothesis-domain-v4', 'stale representative-eligibility policy'],
  ['a217b3b42d5ca4fd6baa4e59cf7d7905bada2c0e', 'stale SignalLab source commit'],
  ['28ed8e9d0dba9f7672880eee608b4328f4482d13', 'stale SignalLab source commit'],
  ['1197f2d46c9b4953253302a95a31cb7ff2212fca', 'stale SignalLab source commit'],
  ['03bc13eb9d5efcfc5f2f9c1792042f670b71ef9a', 'stale SignalLab source commit'],
  ['3fc4f90b2b5b948c93316d70a6a924229044844474c9458844d980b864482f51', 'stale corpus source-manifest SHA-256'],
  ['deb9ed20a6995aeac66c74f7bd1df0ba02f7df5edba0ed493e72b623be65814f', 'stale corpus source-manifest SHA-256'],
  ['3207f1a8170fc44fd8886d9d11bb24367b8b45915fcecabcde1f77f4ddfe5cb4', 'stale corpus source-manifest SHA-256'],
  ['38288f0e0437dbb687674308afecb4f30adadc9e93ea7abad3b8bf13d80ec918', 'stale corpus source-manifest SHA-256'],
  ['1c9d18cbdabf28ff7f52a6bd740172feaabaf3521068f757228fb39d57c0279f', 'stale model asset SHA-256'],
  ['b664d952ec4a7ca8fc87652c0c0586b2e5f9e09e88b7b24491bcaa567e166b09', 'stale model asset SHA-256'],
  ['05ec69aacc100f272446b7e00ba36cd112e516b8832585174312bac1f6af7d0c', 'stale model asset SHA-256'],
  ['cbcb4e29d5642846d781b6a8815a42e8380a81d00094adb628919ae34ea453b0', 'stale v7 model asset SHA-256'],
];
const STALE_OBSERVATION_HORIZON_PATTERNS = [
  {
    pattern:
      /\b24(?:-|\s+)(?:look(?:s)?|spectrum(?:-|\s+)opportunit(?:y|ies)|observation(?:-|\s+)opportunit(?:y|ies)|standard(?:-|\s+)observation(?:s)?)\b/gi,
    label: 'stale 24-opportunity observation horizon',
  },
  {
    pattern:
      /\bstandard(?:-|\s+)24(?:-|\s+)(?:look(?:s)?|observation(?:s)?|opportunit(?:y|ies))\b/gi,
    label: 'stale 24-opportunity observation horizon',
  },
];
const PINNED_TAIL_VIEWS = ['spectrum-only', 'envelope-untimed', 'envelope-timed'];
const PINNED_FITTING_SEEDS = [407, 1_407, 2_407, 3_407, 4_407, 5_407];
const PINNED_CALIBRATION_SEEDS = [6_407, 6_419, 6_421, 6_449, 6_451, 6_469, 6_473, 6_481];
const PINNED_VALIDATION_SEEDS = [13_001, 13_019, 13_037, 13_063, 13_081, 13_099, 13_127, 13_151];
const PINNED_TRAINING_SNR_DB = [6, 10, 16, 24, 32];
const PINNED_TRAINING_RBW_DIVISORS = [12, 20, 35, 55, 80, 120];
const PINNED_VALIDATION_RBW_DIVISORS = [15.5, 44, 98];
const PINNED_PRODUCTION_HIGH_SNR_COVERAGE_POLICY = {
  id: 'branch-conditional-production-regime-presence-v2',
  spectrumOnly: {
    minimumDistinctObservationDomainEligibleSeedsPerHighSnrCell: 1,
  },
  qualifiedEnvelope: {
    minimumDistinctPhysicalCaptureSeedsPerHighSnrCell: 1,
    observationDomainEligibilityPolicy:
      'pooled-by-scenario-and-view-after-causal-capture-v1',
    outOfDomainCapturePolicy:
      'honest-abstention-excluded-from-envelope-likelihood-v1',
  },
  globalCoveragePolicy: 'all-seeds-at-one-or-more-regimes-except-declared-sparse-asynchronous-scenarios-v1',
};
const PINNED_TAIL_POLICIES = {
  scoreUnit: 'one-independent-branch-acquisition-attempt-score-per-evidence-view-v4',
  representativeSelection:
    'consecutive-spectrum-all-runtime-representatives-and-independent-integrated-excess-rank-0-envelope-sole-capture-v5',
  representativeAggregation:
    'consecutive-spectrum-branch-minimum-qualified-envelope-branch-sole-capture-v5',
  runtimeInterpretation:
    'spectrum-member-dominates-independent-branch-attempt-min-envelope-is-independent-sole-capture-v3',
  statisticalInterpretation: 'empirical-synthetic-reference-only-no-exchangeability-or-coverage-guarantee-v1',
};
const PINNED_CORPUS_SOURCE_PATHS = [
  'package-lock.json',
  'package.json',
  'src/canonical-timing.ts',
  'src/catalog.ts',
  'src/classification-corpus.ts',
  'src/contracts.ts',
  'src/source-provenance.ts',
  'src/waveforms.ts',
];
const PUBLICATION_PATHS = [
  'README.md',
  'docs/BAYESIAN_DETECTION_CLASSIFICATION_RESEARCH.md',
  'docs/SIGNALLAB_EMSO_CLASSIFIER_CONTRACT.md',
  'docs/UI_UX_CONTRACTS.md',
];

function valueAt(object, path) {
  let value = object;
  for (const segment of path.split('.')) {
    if (value === null || typeof value !== 'object' || !(segment in value)) {
      throw new Error(`${REPORT_PATH} is missing ${path}`);
    }
    value = value[segment];
  }
  return value;
}

function numberAt(object, path, { integer = false } = {}) {
  const value = valueAt(object, path);
  if (typeof value !== 'number' || !Number.isFinite(value) || (integer && !Number.isInteger(value))) {
    throw new Error(`${REPORT_PATH} ${path} must be a finite${integer ? ' integer' : ' number'}`);
  }
  return value;
}

function arrayAt(object, path) {
  const value = valueAt(object, path);
  if (!Array.isArray(value)) {
    throw new Error(`${REPORT_PATH} ${path} must be an array`);
  }
  return value;
}

function booleanAt(object, path) {
  const value = valueAt(object, path);
  if (typeof value !== 'boolean') throw new Error(`${REPORT_PATH} ${path} must be a boolean`);
  return value;
}

function objectAt(object, path) {
  const value = valueAt(object, path);
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${REPORT_PATH} ${path} must be an object`);
  }
  return value;
}

function expectNonNegativeInteger(failures, value, label, { positive = false } = {}) {
  if (!Number.isInteger(value) || value < (positive ? 1 : 0)) {
    failures.push(`${label} must be a ${positive ? 'positive' : 'non-negative'} integer`);
    return false;
  }
  return true;
}

function expectLowercaseSha256(failures, value, label) {
  if (typeof value !== 'string' || !/^[a-f0-9]{64}$/.test(value) || /^0{64}$/.test(value)) {
    failures.push(`${label} must be a nonzero lowercase SHA-256`);
    return false;
  }
  return true;
}

function validatePositiveDefiniteScale(failures, scale, expectedDimension, label) {
  if (!Array.isArray(scale)
    || scale.length !== expectedDimension
    || scale.some((row) => !Array.isArray(row)
      || row.length !== expectedDimension
      || row.some((value) => typeof value !== 'number' || !Number.isFinite(value)))) {
    return;
  }

  for (let row = 0; row < expectedDimension; row += 1) {
    for (let column = row + 1; column < expectedDimension; column += 1) {
      const left = scale[row][column];
      const right = scale[column][row];
      const tolerance = 1e-12 * Math.max(1, Math.abs(left), Math.abs(right));
      if (Math.abs(left - right) > tolerance) {
        failures.push(`${label} scale must be symmetric`);
        return;
      }
    }
  }

  const cholesky = Array.from(
    { length: expectedDimension },
    () => Array(expectedDimension).fill(0),
  );
  for (let row = 0; row < expectedDimension; row += 1) {
    for (let column = 0; column <= row; column += 1) {
      let residual = scale[row][column];
      for (let index = 0; index < column; index += 1) {
        residual -= cholesky[row][index] * cholesky[column][index];
      }
      if (row === column) {
        if (!(residual > 0) || !Number.isFinite(residual)) {
          failures.push(`${label} scale must be positive definite`);
          return;
        }
        cholesky[row][column] = Math.sqrt(residual);
      } else {
        cholesky[row][column] = residual / cholesky[column][column];
      }
    }
  }
}

function formatInteger(value) {
  if (!Number.isInteger(value)) {
    throw new Error(`cannot publish non-integer ${value} as an integer`);
  }
  return String(value).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

function formatFixed(value, digits) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new Error(`cannot publish non-finite value ${value}`);
  }
  return value.toFixed(digits);
}

function formatScientific(value) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value === 0) {
    throw new Error(`cannot publish ${value} in normalized scientific notation`);
  }
  const [rawCoefficient, rawExponent] = value.toExponential().split('e');
  const coefficient = rawCoefficient.replace(/0+$/, '').replace(/\.$/, '');
  return `${coefficient}e${Number(rawExponent)}`;
}

function formatOxford(values) {
  if (values.length === 0) return '';
  if (values.length === 1) return values[0];
  if (values.length === 2) return `${values[0]} and ${values[1]}`;
  return `${values.slice(0, -1).join(', ')}, and ${values.at(-1)}`;
}

function numberWord(value) {
  const words = [
    'zero', 'one', 'two', 'three', 'four', 'five', 'six',
    'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve',
  ];
  return words[value] ?? String(value);
}

function pluralize(value, singular) {
  return value === 1 ? singular : `${singular}s`;
}

function expectedPriorVariants() {
  return [
    { id: PINNED_PRIOR_VARIANT_IDS[0], kind: 'declared-engineering-assumption', prior: { ...PINNED_ENGINEERING_PRIOR } },
    { id: PINNED_PRIOR_VARIANT_IDS[1], kind: 'unknown-mass-shift', prior: priorWithUnknownMass(0.10) },
    { id: PINNED_PRIOR_VARIANT_IDS[2], kind: 'unknown-mass-shift', prior: priorWithUnknownMass(0.30) },
    {
      id: PINNED_PRIOR_VARIANT_IDS[3], kind: 'family-mass-shift',
      prior: priorWithKnownFamilyMultipliers({ analog: 0.90, cellular: 1.35, wifi: 0.90, bluetooth: 0.90 }),
    },
    {
      id: PINNED_PRIOR_VARIANT_IDS[4], kind: 'family-mass-shift',
      prior: priorWithKnownFamilyMultipliers({ analog: 0.90, cellular: 0.90, wifi: 1.25, bluetooth: 1.25 }),
    },
  ];
}

function priorWithUnknownMass(unknownMass) {
  const knownBaselineMass = 1 - PINNED_ENGINEERING_PRIOR['unknown-signal'];
  return Object.fromEntries(PINNED_CLASS_IDS.map((id) => [
    id,
    id === 'unknown-signal'
      ? unknownMass
      : PINNED_ENGINEERING_PRIOR[id] * (1 - unknownMass) / knownBaselineMass,
  ]));
}

function priorWithKnownFamilyMultipliers(multipliers) {
  const unknownMass = PINNED_ENGINEERING_PRIOR['unknown-signal'];
  const weightedKnownTotal = PINNED_KNOWN_CLASS_IDS.reduce(
    (sum, id) => sum + PINNED_ENGINEERING_PRIOR[id] * multipliers[priorFamily(id)],
    0,
  );
  return Object.fromEntries(PINNED_CLASS_IDS.map((id) => [
    id,
    id === 'unknown-signal'
      ? unknownMass
      : PINNED_ENGINEERING_PRIOR[id] * multipliers[priorFamily(id)] * (1 - unknownMass) / weightedKnownTotal,
  ]));
}

function priorFamily(id) {
  if (id === 'cw-like' || id === 'am-dsb-full-carrier-like' || id === 'fm-angle-modulated-like') return 'analog';
  if (id === 'wifi-hr-dsss-like' || id === 'wifi-ofdm-like') return 'wifi';
  if (id === 'bluetooth-like') return 'bluetooth';
  return 'cellular';
}

function validatePriorSensitivityPopulation(
  report,
  path,
  expectedPopulation,
  expectedSamples,
  failures,
  expectedPartition = undefined,
) {
  const audit = objectAt(report, path);
  expectEqual(failures, audit.valid, true, `${path} validity`);
  expectEqual(
    failures,
    audit.qualification,
    'deterministic-synthetic-engineering-prior-sensitivity-not-field-prevalence-calibration',
    `${path} qualification`,
  );
  expectEqual(failures, audit.fieldPrevalenceCalibrated, false, `${path} field-prevalence calibration claim`);
  expectEqual(failures, audit.population, expectedPopulation, `${path} population`);
  expectNonNegativeInteger(failures, audit.samples, `${path} samples`, { positive: true });
  expectEqual(failures, audit.samples, expectedSamples, `${path} complete population denominator`);
  expectDeepEqual(failures, audit.gates, PINNED_PRIOR_SENSITIVITY_GATES, `${path} gates`);
  expectEqual(failures, audit.modelPriorMatchesPinned, true, `${path} engineering-prior model pin`);
  expectEqual(failures, audit.baselineDecisionMismatchCount, 0, `${path} baseline decision mismatch count`);

  const variants = arrayAt(report, `${path}.variants`);
  expectDeepEqual(
    failures,
    variants.map((variant) => variant?.id),
    PINNED_PRIOR_VARIANT_IDS,
    `${path} variant IDs and order`,
  );
  const pinnedVariants = expectedPriorVariants();
  const knownCoverages = [];
  const hierarchicalAccuracies = [];
  const incompatibleRisks = [];
  const falseAcceptedUnknownRisks = [];
  const decisionChangeRates = [];
  for (const [index, variant] of variants.entries()) {
    if (variant === null || typeof variant !== 'object' || Array.isArray(variant)) {
      throw new Error(`${REPORT_PATH} ${path}.variants.${index} must be an object`);
    }
    const label = `${path} variant ${index}`;
    expectEqual(failures, variant.passed, true, `${label} passed`);
    expectEqual(failures, variant.kind, pinnedVariants[index]?.kind, `${label} kind`);
    expectDeepEqual(failures, variant.prior, pinnedVariants[index]?.prior, `${label} weights`);
    const cases = numberAt(report, `${path}.variants.${index}.cases`, { integer: true });
    const knownCases = numberAt(report, `${path}.variants.${index}.knownCases`, { integer: true });
    const unknownCases = numberAt(report, `${path}.variants.${index}.unknownCases`, { integer: true });
    const incompatibleCount = numberAt(
      report,
      `${path}.variants.${index}.incompatibleNonUnknownCount`,
      { integer: true },
    );
    const falseAcceptedUnknownCount = numberAt(
      report,
      `${path}.variants.${index}.falseAcceptedUnknownCount`,
      { integer: true },
    );
    const decisionChangeCount = numberAt(
      report,
      `${path}.variants.${index}.decisionChangeCount`,
      { integer: true },
    );
    if (cases <= 0 || knownCases < 0 || unknownCases < 0 || incompatibleCount < 0
      || falseAcceptedUnknownCount < 0 || decisionChangeCount < 0
      || incompatibleCount > cases || falseAcceptedUnknownCount > unknownCases
      || decisionChangeCount > cases) {
      failures.push(`${label} counts must be bounded non-negative integers with a positive case denominator`);
    }
    expectEqual(failures, cases, audit.samples, `${label} complete population denominator`);
    expectEqual(failures, knownCases + unknownCases, cases, `${label} case partition`);
    if (expectedPartition !== undefined) {
      expectEqual(failures, knownCases, expectedPartition.knownCases, `${label} known-case denominator`);
      expectEqual(failures, unknownCases, expectedPartition.unknownCases, `${label} unknown-case denominator`);
    }
    expectEqual(
      failures,
      numberAt(report, `${path}.variants.${index}.priorTotal`),
      Object.values(pinnedVariants[index]?.prior ?? {}).reduce((sum, value) => sum + value, 0),
      `${label} prior total`,
    );
    const knownCoverage = numberAt(report, `${path}.variants.${index}.knownCoverage`);
    const hierarchicalAccuracy = numberAt(report, `${path}.variants.${index}.hierarchicalAccuracy`);
    const incompatibleRisk = numberAt(report, `${path}.variants.${index}.incompatibleNonUnknownRisk`);
    const falseAcceptedUnknownRisk = numberAt(report, `${path}.variants.${index}.falseAcceptedUnknownRisk`);
    const decisionChangeRate = numberAt(report, `${path}.variants.${index}.decisionChangeRate`);
    expectEqual(failures, incompatibleRisk, incompatibleCount / Math.max(1, cases), `${label} incompatible risk`);
    expectEqual(
      failures,
      falseAcceptedUnknownRisk,
      falseAcceptedUnknownCount / Math.max(1, unknownCases),
      `${label} false-accepted-unknown risk`,
    );
    expectEqual(failures, decisionChangeRate, decisionChangeCount / Math.max(1, cases), `${label} decision-change rate`);
    for (const [value, metric] of [
      [knownCoverage, 'known coverage'],
      [hierarchicalAccuracy, 'hierarchical accuracy'],
      [incompatibleRisk, 'incompatible risk'],
      [falseAcceptedUnknownRisk, 'false-accepted-unknown risk'],
      [decisionChangeRate, 'decision-change rate'],
    ]) expectRange(failures, value, 0, 1, `${label} ${metric}`);
    if (knownCoverage < PINNED_PRIOR_SENSITIVITY_GATES.minimumKnownCoverage
      || hierarchicalAccuracy < PINNED_PRIOR_SENSITIVITY_GATES.minimumHierarchicalAccuracy
      || incompatibleRisk > PINNED_PRIOR_SENSITIVITY_GATES.maximumIncompatibleNonUnknownRisk
      || falseAcceptedUnknownRisk > PINNED_PRIOR_SENSITIVITY_GATES.maximumFalseAcceptedUnknownRisk
      || decisionChangeRate > PINNED_PRIOR_SENSITIVITY_GATES.maximumDecisionChangeRate) {
      failures.push(`${label} violates an independently checked prior-sensitivity gate`);
    }
    knownCoverages.push(knownCoverage);
    hierarchicalAccuracies.push(hierarchicalAccuracy);
    incompatibleRisks.push(incompatibleRisk);
    falseAcceptedUnknownRisks.push(falseAcceptedUnknownRisk);
    decisionChangeRates.push(decisionChangeRate);
  }
  return {
    variants,
    knownCoverages,
    hierarchicalAccuracies,
    incompatibleRisks,
    falseAcceptedUnknownRisks,
    decisionChangeRates,
  };
}

function normalizeProse(value) {
  return value
    .replace(/([A-Za-z])-\s+([A-Za-z])/g, '$1-$2')
    .replace(/\s+/g, ' ')
    .trim();
}

function visibleMarkdown(value) {
  const withoutComments = value.replace(/<!--[\s\S]*?-->/g, '');
  const visible = [];
  let fenceCharacter;
  let fenceLength = 0;
  for (const line of withoutComments.split(/\r?\n/)) {
    const fence = line.match(/^\s{0,3}(`{3,}|~{3,})/);
    if (fence) {
      const marker = fence[1];
      if (fenceCharacter === undefined) {
        fenceCharacter = marker[0];
        fenceLength = marker.length;
      } else if (marker[0] === fenceCharacter && marker.length >= fenceLength) {
        fenceCharacter = undefined;
        fenceLength = 0;
      }
      continue;
    }
    if (fenceCharacter === undefined) visible.push(line);
  }
  return visible.join('\n');
}

function expectRange(failures, value, minimum, maximum, label) {
  if (typeof value !== 'number' || !Number.isFinite(value)
    || value < minimum || value > maximum) {
    failures.push(`${label}: expected ${minimum}..${maximum}, observed ${value}`);
  }
}

function occurrenceCount(haystack, needle) {
  if (needle.length === 0) return 0;
  let count = 0;
  let offset = 0;
  while ((offset = haystack.indexOf(needle, offset)) !== -1) {
    count += 1;
    offset += needle.length;
  }
  return count;
}

function expectExactlyOnce(failures, path, text, expected, label) {
  const normalizedExpected = normalizeProse(expected);
  const count = occurrenceCount(text, normalizedExpected);
  if (count !== 1) {
    failures.push(
      `${path} must contain exactly one ${label} publication (found ${count}). Expected:\n${normalizedExpected}`,
    );
  }
}

function expectEqual(failures, actual, expected, label) {
  if (actual !== expected) {
    failures.push(`${label}: expected ${expected}, observed ${actual}`);
  }
}

function expectNear(failures, actual, expected, tolerance, label) {
  if (typeof actual !== 'number' || typeof expected !== 'number'
    || !Number.isFinite(actual) || !Number.isFinite(expected)
    || Math.abs(actual - expected) > tolerance) {
    failures.push(`${label}: expected ${expected} ± ${tolerance}, observed ${actual}`);
  }
}

function expectDeepEqual(failures, actual, expected, label) {
  expectEqual(failures, JSON.stringify(actual), JSON.stringify(expected), label);
}

function parseGeneratedModel(source) {
  const match = source.match(
    /export const BAYESIAN_OBSERVABLE_MODEL: ObservableClassifierModelAsset = (\{[\s\S]*\});\s*$/,
  );
  if (!match) throw new Error(`${MODEL_PATH} does not contain one generated JSON model payload`);
  try {
    return JSON.parse(match[1]);
  } catch (error) {
    throw new Error(`${MODEL_PATH} generated payload is not JSON: ${error instanceof Error ? error.message : String(error)}`);
  }
}

function expectedDimensionsForView(view) {
  if (view === 'spectrum-only') {
    return PINNED_DIMENSIONS.filter((dimension) => !dimension.startsWith('envelope.'));
  }
  if (view === 'envelope-untimed') {
    return PINNED_DIMENSIONS.filter((dimension) =>
      dimension !== 'envelope.logTransitionRateHz'
      && !dimension.startsWith('envelope.periodicEnergy'));
  }
  if (view === 'envelope-timed') return PINNED_DIMENSIONS;
  throw new Error(`unknown observable model view ${view}`);
}

function generatedComponentsForView(model, view) {
  const components = model?.componentsByView?.[view];
  if (!Array.isArray(components)) {
    throw new Error(`${MODEL_PATH} class ${model?.id ?? '<unknown>'} must publish ${view} componentsByView`);
  }
  return components;
}

function componentSourceScenarioId(component) {
  return component?.sourceScenarioId ?? component?.id;
}

function uniqueSourceScenarioIds(components) {
  return [...new Set(components.map(componentSourceScenarioId))];
}

function sumPairMetric(pairs, key) {
  return pairs.reduce((sum, pair, index) => {
    if (pair === null || typeof pair !== 'object') {
      throw new Error(`${REPORT_PATH} corpus.exactObservableEquivalencePairAudit.pairs.${index} must be an object`);
    }
    const value = pair[key];
    if (typeof value !== 'number' || !Number.isInteger(value) || value < 0) {
      throw new Error(
        `${REPORT_PATH} corpus.exactObservableEquivalencePairAudit.pairs.${index}.${key} must be a non-negative integer`,
      );
    }
    return sum + value;
  }, 0);
}

function validateGeneratedCausalSamplingAudit(generatedModel, failures) {
  const trainingMatrix = generatedModel.trainingMatrix;
  if (trainingMatrix === null || typeof trainingMatrix !== 'object' || Array.isArray(trainingMatrix)) {
    throw new Error(`${MODEL_PATH} trainingMatrix must be an object`);
  }
  const fittingCounts = trainingMatrix.fittingCapturedEnvelopeCountsByScenario;
  if (fittingCounts === null || typeof fittingCounts !== 'object' || Array.isArray(fittingCounts)) {
    throw new Error(`${MODEL_PATH} fittingCapturedEnvelopeCountsByScenario must be an object`);
  }
  const fittedScenarioIds = generatedModel.classModels
    .flatMap((model) => uniqueSourceScenarioIds(
      generatedComponentsForView(model, 'spectrum-only'),
    ));
  const duplicateFittedScenarioIds = fittedScenarioIds.filter(
    (scenarioId, index) => fittedScenarioIds.indexOf(scenarioId) !== index,
  );
  expectEqual(failures, duplicateFittedScenarioIds.length, 0, 'unique fitted component scenario IDs');
  expectDeepEqual(
    failures,
    Object.keys(fittingCounts).sort(),
    [...fittedScenarioIds].sort(),
    'fitting captured-envelope scenario key set',
  );
  let fittingRepresentativeCount = 0;
  for (const [scenarioId, count] of Object.entries(fittingCounts)) {
    const censoredFrequencyAgile =
      PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORED_SCENARIO_IDS.includes(scenarioId);
    if (expectNonNegativeInteger(
      failures,
      count,
      `fitting captured-envelope count ${scenarioId}`,
      { positive: !censoredFrequencyAgile },
    )) fittingRepresentativeCount += count;
    if (censoredFrequencyAgile) {
      expectEqual(
        failures,
        count,
        0,
        `fitting censored frequency-agile envelope count ${scenarioId}`,
      );
    }
  }
  expectEqual(
    failures,
    trainingMatrix.likelihoodPopulationPolicy,
    PINNED_LIKELIHOOD_POPULATION_POLICY,
    'generated likelihood-population policy',
  );
  expectDeepEqual(
    failures,
    trainingMatrix.likelihoodComponentDecompositionPolicy,
    PINNED_LIKELIHOOD_COMPONENT_DECOMPOSITION_POLICY,
    'generated likelihood-component decomposition policy',
  );
  expectEqual(
    failures,
    trainingMatrix.acquisitionBranchPolicy,
    PINNED_ACQUISITION_BRANCH_POLICY,
    'generated acquisition-branch policy',
  );
  expectEqual(
    failures,
    trainingMatrix.selectionPolicy,
    PINNED_SELECTION_POLICY,
    'generated representative selection policy',
  );
  expectEqual(
    failures,
    trainingMatrix.representativeWeightingPolicy,
    PINNED_REPRESENTATIVE_WEIGHTING_POLICY,
    'generated representative weighting policy',
  );
  expectEqual(
    failures,
    trainingMatrix.detectedPowerAcquisitionQualification,
    PINNED_DETECTED_POWER_ACQUISITION_QUALIFICATION,
    'generated detected-power acquisition qualification',
  );
  expectEqual(
    failures,
    trainingMatrix.detectedPowerSelectionCondition,
    PINNED_DETECTED_POWER_SELECTION_CONDITION,
    'generated automatic detected-power selection condition',
  );
  expectDeepEqual(
    failures,
    trainingMatrix.frequencyAgileFixedTuneEnvelopeCensoringPolicy,
    PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORING_POLICY,
    'generated frequency-agile fixed-tune envelope-censoring policy',
  );
  const censoredCaptureCounts =
    trainingMatrix.censoredFrequencyAgileFixedTuneCaptureCountsByScenario;
  if (censoredCaptureCounts === null
    || typeof censoredCaptureCounts !== 'object'
    || Array.isArray(censoredCaptureCounts)) {
    throw new Error(`${MODEL_PATH} censoredFrequencyAgileFixedTuneCaptureCountsByScenario must be an object`);
  }
  expectDeepEqual(
    failures,
    Object.keys(censoredCaptureCounts).sort(),
    ['fitting', 'tailCalibration'],
    'generated censored-capture partition key set',
  );
  for (const partition of ['fitting', 'tailCalibration']) {
    const counts = censoredCaptureCounts[partition];
    if (counts === null || typeof counts !== 'object' || Array.isArray(counts)) {
      throw new Error(`${MODEL_PATH} censored capture counts ${partition} must be an object`);
    }
    expectDeepEqual(
      failures,
      Object.keys(counts).sort(),
      [...PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORED_SCENARIO_IDS].sort(),
      `generated censored capture ${partition} scenario key set`,
    );
    for (const scenarioId of PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORED_SCENARIO_IDS) {
      expectNonNegativeInteger(
        failures,
        counts[scenarioId],
        `generated censored capture ${partition}/${scenarioId}`,
        { positive: true },
      );
    }
  }
  const fittingCountsByScenarioByView = trainingMatrix.fittingRepresentativeCountsByScenarioByView;
  if (fittingCountsByScenarioByView === null
    || typeof fittingCountsByScenarioByView !== 'object'
    || Array.isArray(fittingCountsByScenarioByView)) {
    throw new Error(`${MODEL_PATH} fittingRepresentativeCountsByScenarioByView must be an object`);
  }
  expectDeepEqual(
    failures,
    Object.keys(fittingCountsByScenarioByView).sort(),
    [...fittedScenarioIds].sort(),
    'view-matched fitting scenario key set',
  );
  const fittingRepresentativeCountsByView = Object.fromEntries(
    PINNED_TAIL_VIEWS.map((view) => [view, 0]),
  );
  for (const scenarioId of fittedScenarioIds) {
    const counts = fittingCountsByScenarioByView[scenarioId];
    if (counts === null || typeof counts !== 'object' || Array.isArray(counts)) {
      throw new Error(`${MODEL_PATH} view-matched fitting counts for ${scenarioId} must be an object`);
    }
    expectDeepEqual(
      failures,
      Object.keys(counts).sort(),
      [...PINNED_TAIL_VIEWS].sort(),
      `view-matched fitting ${scenarioId} key set`,
    );
    for (const view of PINNED_TAIL_VIEWS) {
      const censoredEnvelope = view !== 'spectrum-only'
        && PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORED_SCENARIO_IDS.includes(scenarioId);
      if (expectNonNegativeInteger(
        failures,
        counts[view],
        `view-matched fitting ${scenarioId}/${view} count`,
        { positive: !censoredEnvelope },
      )) fittingRepresentativeCountsByView[view] += counts[view];
      if (censoredEnvelope) {
        expectEqual(
          failures,
          counts[view],
          0,
          `view-matched fitting censored ${scenarioId}/${view} count`,
        );
      }
    }
    expectEqual(
      failures,
      counts['envelope-timed'],
      fittingCounts[scenarioId],
      `legacy/timed fitting count ${scenarioId}`,
    );
  }
  expectEqual(
    failures,
    fittingRepresentativeCountsByView['envelope-timed'],
    fittingRepresentativeCount,
    'view-matched/legacy fitting envelope-timed total',
  );

  const audit = trainingMatrix.causalSamplingAudit;
  if (audit === null || typeof audit !== 'object' || Array.isArray(audit)) {
    throw new Error(`${MODEL_PATH} causalSamplingAudit must be an object`);
  }
  expectDeepEqual(
    failures,
    Object.keys(audit).sort(),
    [
      'schemaVersion',
      'fitting',
      'tailCalibration',
      'provenanceUnavailableAttemptPolicy',
      'provenanceUnavailableAttempts',
      'attributedSourceClockTraceAudit',
    ].sort(),
    'generated causal-sampling audit key set',
  );
  expectEqual(failures, audit.schemaVersion, 3, 'generated causal-sampling audit schema');
  expectEqual(
    failures,
    audit.provenanceUnavailableAttemptPolicy,
    'branch-attributed-exact-attempt-cell-counts-v2',
    'generated provenance-unavailable attempt policy',
  );

  const fittingAcquisitionRegimeIds = trainingMatrix.fittingAcquisitionRegimeIds;
  const calibrationAcquisitionRegimeIds = trainingMatrix.tailCalibrationAcquisitionRegimeIds;
  const fittingSeeds = trainingMatrix.seeds;
  const calibrationSeeds = trainingMatrix.tailCalibrationSeeds;
  for (const [value, label] of [
    [fittingAcquisitionRegimeIds, 'fitting acquisition-regime IDs'],
    [calibrationAcquisitionRegimeIds, 'tail-calibration acquisition-regime IDs'],
    [fittingSeeds, 'fitting seeds'],
    [calibrationSeeds, 'tail-calibration seeds'],
    [trainingMatrix.snrDb, 'training SNR values'],
  ]) {
    if (!Array.isArray(value) || value.length === 0) {
      failures.push(`${MODEL_PATH} ${label} must be a non-empty array`);
    }
  }
  const knownFittedScenarioIds = generatedModel.classModels
    .filter((model) => model.id !== 'unknown-signal')
    .flatMap((model) => uniqueSourceScenarioIds(
      generatedComponentsForView(model, 'spectrum-only'),
    ));
  const expectedFittingAttemptCount = fittedScenarioIds.length
    * trainingMatrix.snrDb.length
    * fittingAcquisitionRegimeIds.length
    * fittingSeeds.length;
  const expectedCalibrationAttemptCount = knownFittedScenarioIds.length
    * trainingMatrix.snrDb.length
    * calibrationAcquisitionRegimeIds.length
    * calibrationSeeds.length;

  const validateCountMap = (counts, label, expectedTotal, allowedKeys = undefined) => {
    if (counts === null || typeof counts !== 'object' || Array.isArray(counts)) {
      throw new Error(`${MODEL_PATH} ${label} must be an object`);
    }
    let total = 0;
    if (allowedKeys !== undefined) {
      expectDeepEqual(
        failures,
        Object.keys(counts).sort(),
        [...allowedKeys].sort(),
        `generated ${label} exact key set`,
      );
    }
    for (const [key, count] of Object.entries(counts)) {
      if (!/^(0|[1-9]\d*)$/.test(key)) {
        failures.push(`generated ${label} key ${key} must be a canonical non-negative integer`);
      }
      if (allowedKeys !== undefined && !allowedKeys.includes(key)) {
        failures.push(`generated ${label} key ${key} is outside the pinned observation horizons`);
      }
      if (expectNonNegativeInteger(failures, count, `generated ${label}.${key}`, { positive: true })) {
        total += count;
      }
    }
    expectEqual(failures, total, expectedTotal, `generated ${label} count total`);
    return total;
  };

  const validatePartition = (partitionName, expectedAttemptCount) => {
    const partition = audit[partitionName];
    if (partition === null || typeof partition !== 'object' || Array.isArray(partition)) {
      throw new Error(`${MODEL_PATH} causalSamplingAudit.${partitionName} must be an object`);
    }
    expectDeepEqual(
      failures,
      Object.keys(partition).sort(),
      [
        'pairedNuisanceCellCount',
        'fitEligibleRepresentativeCountsByView',
        'eligibleAttemptCountsByView',
        'runtimeBranches',
      ].sort(),
      `generated causal-sampling ${partitionName} key set`,
    );
    expectNonNegativeInteger(
      failures,
      partition.pairedNuisanceCellCount,
      `generated causal-sampling ${partitionName}.pairedNuisanceCellCount`,
      { positive: true },
    );
    expectEqual(
      failures,
      partition.pairedNuisanceCellCount,
      expectedAttemptCount,
      `generated causal-sampling ${partitionName} complete paired nuisance-cell matrix`,
    );

    const fitEligibleCountsByView = partition.fitEligibleRepresentativeCountsByView;
    const eligibleAttemptCountsByView = partition.eligibleAttemptCountsByView;
    for (const [counts, label] of [
      [fitEligibleCountsByView, 'fit-eligible representative'],
      [eligibleAttemptCountsByView, 'eligible-attempt'],
    ]) {
      if (counts === null || typeof counts !== 'object' || Array.isArray(counts)) {
        throw new Error(
          `${MODEL_PATH} causalSamplingAudit.${partitionName} ${label} counts must be an object`,
        );
      }
      expectDeepEqual(
        failures,
        Object.keys(counts).sort(),
        [...PINNED_TAIL_VIEWS].sort(),
        `generated causal-sampling ${partitionName} ${label} view key set`,
      );
      for (const view of PINNED_TAIL_VIEWS) {
        expectNonNegativeInteger(
          failures,
          counts[view],
          `generated causal-sampling ${partitionName} ${label} ${view} count`,
          { positive: true },
        );
        if (eligibleAttemptCountsByView?.[view] > partition.pairedNuisanceCellCount) {
          failures.push(`generated causal-sampling ${partitionName} ${view} eligible-attempt count exceeds paired nuisance-cell count`);
        }
        if (eligibleAttemptCountsByView?.[view] > fitEligibleCountsByView?.[view]) {
          failures.push(`generated causal-sampling ${partitionName} ${view} eligible-attempt count exceeds fit-eligible representative count`);
        }
      }
    }

    const runtimeBranches = partition.runtimeBranches;
    if (runtimeBranches === null || typeof runtimeBranches !== 'object'
      || Array.isArray(runtimeBranches)) {
      throw new Error(`${MODEL_PATH} causalSamplingAudit.${partitionName}.runtimeBranches must be an object`);
    }
    expectDeepEqual(
      failures,
      Object.keys(runtimeBranches).sort(),
      ['consecutiveSpectrum', 'qualifiedEnvelope'],
      `generated causal-sampling ${partitionName} runtime-branch key set`,
    );
    const spectrum = runtimeBranches.consecutiveSpectrum;
    const envelope = runtimeBranches.qualifiedEnvelope;
    if (spectrum === null || typeof spectrum !== 'object' || Array.isArray(spectrum)) {
      throw new Error(`${MODEL_PATH} causalSamplingAudit.${partitionName}.runtimeBranches.consecutiveSpectrum must be an object`);
    }
    if (envelope === null || typeof envelope !== 'object' || Array.isArray(envelope)) {
      throw new Error(`${MODEL_PATH} causalSamplingAudit.${partitionName}.runtimeBranches.qualifiedEnvelope must be an object`);
    }
    const spectrumIntegerFields = [
      'attemptCount',
      'attemptsWithAnyRepresentative',
      'attemptsWithFitEligibleRepresentative',
      'onlineSpectrumRepresentativeCount',
      'fitEligibleRepresentativeCount',
      'fitIneligibleRepresentativeCount',
      'provenanceUnavailableWindowCount',
      'spectrumAcquisitionCount',
      'physicalDetectedPowerCaptureCount',
      'postCaptureProvenanceUnavailableWindowCount',
      'detectedPowerCaptureSampleCount',
      'censoredFrequencyAgileFixedTuneCaptureCount',
      'sourceClockEventCount',
      'multiRepresentativeAttemptCount',
      'maximumRepresentativesPerAttempt',
    ];
    expectDeepEqual(
      failures,
      Object.keys(spectrum).sort(),
      [
        'detectedPowerCapturePolicyId',
        ...spectrumIntegerFields,
        'observationHorizonCounts',
        'observationOpportunityCounts',
      ].sort(),
      `generated causal-sampling ${partitionName} consecutive-spectrum key set`,
    );
    for (const field of spectrumIntegerFields) {
      expectNonNegativeInteger(
        failures,
        spectrum[field],
        `generated causal-sampling ${partitionName}.consecutiveSpectrum.${field}`,
        { positive: ['attemptCount', 'onlineSpectrumRepresentativeCount',
          'spectrumAcquisitionCount', 'sourceClockEventCount',
          'maximumRepresentativesPerAttempt'].includes(field) },
      );
    }
    expectEqual(
      failures,
      spectrum.detectedPowerCapturePolicyId,
      'no-automatic-detected-power-capture-v1',
      `generated causal-sampling ${partitionName} consecutive-spectrum capture policy`,
    );
    expectEqual(
      failures,
      spectrum.attemptCount,
      partition.pairedNuisanceCellCount,
      `generated causal-sampling ${partitionName} consecutive-spectrum attempt count`,
    );
    expectEqual(
      failures,
      spectrum.physicalDetectedPowerCaptureCount,
      0,
      `generated causal-sampling ${partitionName} consecutive-spectrum physical capture count`,
    );
    expectEqual(
      failures,
      spectrum.postCaptureProvenanceUnavailableWindowCount,
      0,
      `generated causal-sampling ${partitionName} consecutive-spectrum post-capture unavailable count`,
    );
    expectEqual(
      failures,
      spectrum.detectedPowerCaptureSampleCount,
      0,
      `generated causal-sampling ${partitionName} consecutive-spectrum detected-power sample count`,
    );
    expectEqual(
      failures,
      spectrum.censoredFrequencyAgileFixedTuneCaptureCount,
      0,
      `generated causal-sampling ${partitionName} consecutive-spectrum censored capture count`,
    );
    expectEqual(
      failures,
      spectrum.sourceClockEventCount,
      spectrum.spectrumAcquisitionCount,
      `generated causal-sampling ${partitionName} consecutive-spectrum source-clock accounting`,
    );
    expectEqual(
      failures,
      spectrum.onlineSpectrumRepresentativeCount,
      spectrum.fitEligibleRepresentativeCount + spectrum.fitIneligibleRepresentativeCount,
      `generated causal-sampling ${partitionName} consecutive-spectrum eligibility partition`,
    );
    expectEqual(
      failures,
      fitEligibleCountsByView['spectrum-only'],
      spectrum.fitEligibleRepresentativeCount,
      `generated causal-sampling ${partitionName} spectrum fit-eligible representative reconciliation`,
    );
    expectEqual(
      failures,
      eligibleAttemptCountsByView['spectrum-only'],
      spectrum.attemptsWithFitEligibleRepresentative,
      `generated causal-sampling ${partitionName} spectrum eligible-attempt reconciliation`,
    );
    if (spectrum.attemptsWithAnyRepresentative > spectrum.attemptCount
      || spectrum.attemptsWithFitEligibleRepresentative > spectrum.attemptsWithAnyRepresentative
      || spectrum.multiRepresentativeAttemptCount > spectrum.attemptsWithAnyRepresentative
      || spectrum.attemptsWithAnyRepresentative > spectrum.onlineSpectrumRepresentativeCount
      || spectrum.onlineSpectrumRepresentativeCount
        < spectrum.attemptsWithAnyRepresentative + spectrum.multiRepresentativeAttemptCount
      || spectrum.onlineSpectrumRepresentativeCount
        > spectrum.attemptsWithAnyRepresentative
          + spectrum.multiRepresentativeAttemptCount
            * (spectrum.maximumRepresentativesPerAttempt - 1)
      || (spectrum.multiRepresentativeAttemptCount === 0)
        !== (spectrum.onlineSpectrumRepresentativeCount
          === spectrum.attemptsWithAnyRepresentative)
      || (spectrum.maximumRepresentativesPerAttempt === 1)
        !== (spectrum.onlineSpectrumRepresentativeCount
          === spectrum.attemptsWithAnyRepresentative)
      || (spectrum.multiRepresentativeAttemptCount > 0
        && spectrum.maximumRepresentativesPerAttempt
          > spectrum.onlineSpectrumRepresentativeCount
            - spectrum.attemptsWithAnyRepresentative
            - spectrum.multiRepresentativeAttemptCount
            + 2)
      || spectrum.onlineSpectrumRepresentativeCount
        > spectrum.attemptsWithAnyRepresentative * spectrum.maximumRepresentativesPerAttempt) {
      failures.push(`generated causal-sampling ${partitionName} consecutive-spectrum representative accounting is inconsistent`);
    }
    validateCountMap(
      spectrum.observationHorizonCounts,
      `causal-sampling ${partitionName}.consecutiveSpectrum.observationHorizonCounts`,
      spectrum.attemptCount,
      ['32', '96'],
    );
    const spectrumAcquisitionsFromHorizons = Object.entries(spectrum.observationHorizonCounts)
      .reduce((sum, [horizon, count]) => sum + Number(horizon) * count, 0);
    expectEqual(
      failures,
      spectrum.spectrumAcquisitionCount,
      spectrumAcquisitionsFromHorizons,
      `generated causal-sampling ${partitionName} consecutive-spectrum horizon/acquisition accounting`,
    );
    validateCountMap(
      spectrum.observationOpportunityCounts,
      `causal-sampling ${partitionName}.consecutiveSpectrum.observationOpportunityCounts`,
      spectrum.onlineSpectrumRepresentativeCount,
    );
    if (Object.keys(spectrum.observationOpportunityCounts)
      .some((key) => Number(key) < 1 || Number(key) > 96)) {
      failures.push(`generated causal-sampling ${partitionName} consecutive-spectrum observation opportunities must be in [1, 96]`);
    }

    const envelopeIntegerFields = [
      'attemptCount',
      'receiptVerifiedDetectedPowerCaptureSampleCount',
      'capturedEnvelopeRepresentativeCount',
      'censoredFrequencyAgileFixedTuneCaptureCount',
      'fitEligibleTimedCapturedEnvelopeRepresentativeCount',
      'fitEligibleUntimedCapturedEnvelopeRepresentativeCount',
      'provenanceUnavailableWindowCount',
      'preCaptureProvenanceUnavailableWindowCount',
      'postCaptureProvenanceUnavailableWindowCount',
      'spectrumAcquisitionCount',
      'physicalDetectedPowerCaptureCount',
      'attemptsWithoutDetectedPowerCapture',
      'sourceClockEventCount',
    ];
    expectDeepEqual(
      failures,
      Object.keys(envelope).sort(),
      [
        'detectedPowerCapturePolicyId',
        ...envelopeIntegerFields,
        'observationHorizonCounts',
      ].sort(),
      `generated causal-sampling ${partitionName} qualified-envelope key set`,
    );
    for (const field of envelopeIntegerFields) {
      expectNonNegativeInteger(
        failures,
        envelope[field],
        `generated causal-sampling ${partitionName}.qualifiedEnvelope.${field}`,
        { positive: ['attemptCount', 'receiptVerifiedDetectedPowerCaptureSampleCount',
          'capturedEnvelopeRepresentativeCount',
          'censoredFrequencyAgileFixedTuneCaptureCount',
          'spectrumAcquisitionCount', 'physicalDetectedPowerCaptureCount',
          'sourceClockEventCount'].includes(field) },
      );
    }
    expectEqual(
      failures,
      envelope.detectedPowerCapturePolicyId,
      'capture-once-after-rank-0-integrated-excess-current-target-runtime-admission-v3',
      `generated causal-sampling ${partitionName} qualified-envelope capture policy`,
    );
    expectEqual(
      failures,
      envelope.attemptCount,
      partition.pairedNuisanceCellCount,
      `generated causal-sampling ${partitionName} qualified-envelope attempt count`,
    );
    expectEqual(
      failures,
      envelope.provenanceUnavailableWindowCount,
      envelope.preCaptureProvenanceUnavailableWindowCount
        + envelope.postCaptureProvenanceUnavailableWindowCount,
      `generated causal-sampling ${partitionName} qualified-envelope unavailable-window partition`,
    );
    expectEqual(
      failures,
      envelope.postCaptureProvenanceUnavailableWindowCount,
      0,
      `generated causal-sampling ${partitionName} qualified-envelope post-capture unavailable-window count`,
    );
    expectEqual(
      failures,
      envelope.physicalDetectedPowerCaptureCount,
      envelope.receiptVerifiedDetectedPowerCaptureSampleCount
        + envelope.postCaptureProvenanceUnavailableWindowCount,
      `generated causal-sampling ${partitionName} qualified-envelope physical/receipt capture accounting`,
    );
    expectEqual(
      failures,
      envelope.receiptVerifiedDetectedPowerCaptureSampleCount,
      envelope.capturedEnvelopeRepresentativeCount
        + envelope.censoredFrequencyAgileFixedTuneCaptureCount,
      `generated causal-sampling ${partitionName} qualified-envelope admitted/censored sample partition`,
    );
    expectEqual(
      failures,
      envelope.attemptsWithoutDetectedPowerCapture,
      envelope.attemptCount - envelope.physicalDetectedPowerCaptureCount,
      `generated causal-sampling ${partitionName} qualified-envelope no-capture attempt accounting`,
    );
    expectEqual(
      failures,
      envelope.sourceClockEventCount,
      envelope.spectrumAcquisitionCount + envelope.physicalDetectedPowerCaptureCount,
      `generated causal-sampling ${partitionName} qualified-envelope source-clock accounting`,
    );
    expectEqual(
      failures,
      fitEligibleCountsByView['envelope-timed'],
      envelope.fitEligibleTimedCapturedEnvelopeRepresentativeCount,
      `generated causal-sampling ${partitionName} timed-envelope fit-eligible reconciliation`,
    );
    expectEqual(
      failures,
      fitEligibleCountsByView['envelope-untimed'],
      envelope.fitEligibleUntimedCapturedEnvelopeRepresentativeCount,
      `generated causal-sampling ${partitionName} untimed-envelope fit-eligible reconciliation`,
    );
    expectEqual(
      failures,
      eligibleAttemptCountsByView['envelope-timed'],
      envelope.fitEligibleTimedCapturedEnvelopeRepresentativeCount,
      `generated causal-sampling ${partitionName} timed-envelope one-representative-per-attempt accounting`,
    );
    expectEqual(
      failures,
      eligibleAttemptCountsByView['envelope-untimed'],
      envelope.fitEligibleUntimedCapturedEnvelopeRepresentativeCount,
      `generated causal-sampling ${partitionName} untimed-envelope one-representative-per-attempt accounting`,
    );
    if (envelope.physicalDetectedPowerCaptureCount > envelope.attemptCount
      || envelope.receiptVerifiedDetectedPowerCaptureSampleCount > envelope.attemptCount
      || envelope.capturedEnvelopeRepresentativeCount > envelope.attemptCount
      || envelope.censoredFrequencyAgileFixedTuneCaptureCount > envelope.attemptCount
      || envelope.fitEligibleTimedCapturedEnvelopeRepresentativeCount
        > envelope.capturedEnvelopeRepresentativeCount
      || envelope.fitEligibleUntimedCapturedEnvelopeRepresentativeCount
        > envelope.capturedEnvelopeRepresentativeCount) {
      failures.push(`generated causal-sampling ${partitionName} qualified-envelope representative accounting is inconsistent`);
    }
    validateCountMap(
      envelope.observationHorizonCounts,
      `causal-sampling ${partitionName}.qualifiedEnvelope.observationHorizonCounts`,
      envelope.attemptCount,
      ['32', '96'],
    );
    const envelopeAcquisitionsFromHorizons = Object.entries(envelope.observationHorizonCounts)
      .reduce((sum, [horizon, count]) => sum + Number(horizon) * count, 0);
    expectEqual(
      failures,
      envelope.spectrumAcquisitionCount,
      envelopeAcquisitionsFromHorizons,
      `generated causal-sampling ${partitionName} qualified-envelope horizon/acquisition accounting`,
    );

    return { partition, spectrum, envelope };
  };
  const fitting = validatePartition('fitting', expectedFittingAttemptCount);
  const tailCalibration = validatePartition('tailCalibration', expectedCalibrationAttemptCount);
  for (const [partitionName, partition] of [
    ['fitting', fitting],
    ['tailCalibration', tailCalibration],
  ]) {
    expectEqual(
      failures,
      Object.values(censoredCaptureCounts[partitionName]).reduce(
        (sum, count) => sum + (Number.isInteger(count) ? count : 0),
        0,
      ),
      partition.envelope.censoredFrequencyAgileFixedTuneCaptureCount,
      `generated ${partitionName} censored scenario/capture-audit reconciliation`,
    );
  }
  for (const field of [
    'postCaptureUnavailableFitAttempts',
    'postCaptureUnavailableCalibrationAttempts',
  ]) {
    if (!Array.isArray(trainingMatrix[field])) {
      throw new Error(`${MODEL_PATH} ${field} must be an array`);
    }
    expectEqual(failures, trainingMatrix[field].length, 0, `generated ${field}`);
  }
  expectEqual(
    failures,
    fittingRepresentativeCount,
    fitting.envelope.fitEligibleTimedCapturedEnvelopeRepresentativeCount,
    'fitting scenario counts/causal-sampling fit-eligible envelope count',
  );
  for (const view of PINNED_TAIL_VIEWS) {
    expectEqual(
      failures,
      fittingRepresentativeCountsByView[view],
      fitting.partition.fitEligibleRepresentativeCountsByView[view],
      `fitting scenario counts/causal-sampling ${view} fit-eligible count`,
    );
  }
  expectEqual(
    failures,
    fitting.envelope.fitEligibleTimedCapturedEnvelopeRepresentativeCount,
    fitting.partition.fitEligibleRepresentativeCountsByView['envelope-timed'],
    'fitting legacy/timed causal-sampling fit-eligible count',
  );

  const unavailableAttempts = audit.provenanceUnavailableAttempts;
  if (unavailableAttempts === null || typeof unavailableAttempts !== 'object'
    || Array.isArray(unavailableAttempts)) {
    throw new Error(`${MODEL_PATH} causalSamplingAudit.provenanceUnavailableAttempts must be an object`);
  }
  expectDeepEqual(
    failures,
    Object.keys(unavailableAttempts).sort(),
    ['fitting', 'tailCalibration'],
    'generated provenance-unavailable partition key set',
  );
  for (const [partitionName, partition] of [
    ['fitting', fitting],
    ['tailCalibration', tailCalibration],
  ]) {
    const attemptsByBranch = unavailableAttempts[partitionName];
    if (attemptsByBranch === null || typeof attemptsByBranch !== 'object'
      || Array.isArray(attemptsByBranch)) {
      throw new Error(`${MODEL_PATH} provenanceUnavailableAttempts.${partitionName} must be an object`);
    }
    expectDeepEqual(
      failures,
      Object.keys(attemptsByBranch).sort(),
      ['consecutiveSpectrum', 'qualifiedEnvelope'],
      `generated ${partitionName} provenance-unavailable runtime-branch key set`,
    );
    for (const [branch, expectedUnavailableWindowCount] of [
      ['consecutiveSpectrum', partition.spectrum.provenanceUnavailableWindowCount],
      ['qualifiedEnvelope', partition.envelope.provenanceUnavailableWindowCount],
    ]) {
      const attempts = attemptsByBranch[branch];
      if (!Array.isArray(attempts)) {
        throw new Error(`${MODEL_PATH} provenanceUnavailableAttempts.${partitionName}.${branch} must be an array`);
      }
      const attemptIds = [];
      let unavailableWindowCount = 0;
      for (const [index, attempt] of attempts.entries()) {
        if (attempt === null || typeof attempt !== 'object' || Array.isArray(attempt)
          || typeof attempt.attemptId !== 'string' || attempt.attemptId.length === 0) {
          failures.push(`generated ${partitionName}/${branch} unavailable attempt ${index} must publish an attemptId`);
          continue;
        }
        expectDeepEqual(
          failures,
          Object.keys(attempt).sort(),
          ['attemptId', 'unavailableWindowCount'],
          `generated ${partitionName}/${branch} unavailable attempt ${index} key set`,
        );
        attemptIds.push(attempt.attemptId);
        if (expectNonNegativeInteger(
          failures,
          attempt.unavailableWindowCount,
          `generated ${partitionName}/${branch} unavailable attempt ${attempt.attemptId}`,
          { positive: true },
        )) unavailableWindowCount += attempt.unavailableWindowCount;
      }
      expectDeepEqual(
        failures,
        attemptIds,
        [...new Set(attemptIds)].sort((left, right) => left.localeCompare(right)),
        `generated ${partitionName}/${branch} unavailable-attempt canonical identity/order`,
      );
      expectEqual(
        failures,
        unavailableWindowCount,
        expectedUnavailableWindowCount,
        `generated ${partitionName}/${branch} unavailable-attempt/window total`,
      );
    }
  }

  const traceAudit = audit.attributedSourceClockTraceAudit;
  if (traceAudit === null || typeof traceAudit !== 'object' || Array.isArray(traceAudit)) {
    throw new Error(`${MODEL_PATH} attributedSourceClockTraceAudit must be an object`);
  }
  expectDeepEqual(
    failures,
    Object.keys(traceAudit).sort(),
    ['hashAlgorithm', 'serialization', 'fitting', 'tailCalibration'].sort(),
    'generated attributed source-clock trace audit key set',
  );
  expectEqual(failures, traceAudit.hashAlgorithm, 'sha256', 'generated source-clock trace hash algorithm');
  expectEqual(
    failures,
    traceAudit.serialization,
    'canonical-attempt-id-branch-attributed-trace-and-capture-disposition-digest-v3',
    'generated source-clock trace serialization',
  );
  const traceHashes = [];
  for (const partitionName of ['fitting', 'tailCalibration']) {
    const hashes = traceAudit[partitionName];
    if (hashes === null || typeof hashes !== 'object' || Array.isArray(hashes)) {
      throw new Error(`${MODEL_PATH} attributedSourceClockTraceAudit.${partitionName} must be an object`);
    }
    expectDeepEqual(
      failures,
      Object.keys(hashes).sort(),
      ['consecutiveSpectrumSha256', 'qualifiedEnvelopeSha256'],
      `generated ${partitionName} attributed source-clock trace hash key set`,
    );
    for (const branch of ['consecutiveSpectrum', 'qualifiedEnvelope']) {
      const hash = hashes[`${branch}Sha256`];
      expectLowercaseSha256(
        failures,
        hash,
        `generated ${partitionName}/${branch} source-clock trace hash`,
      );
      traceHashes.push(hash);
    }
  }
  if (new Set(traceHashes).size !== traceHashes.length) {
    failures.push('generated fitting/calibration and spectrum/envelope attributed source-clock traces must be distinct');
  }

  return {
    fittingRepresentativeCount,
    fittingSpectrumRepresentativeCount: fittingRepresentativeCountsByView['spectrum-only'],
    fittingUntimedEnvelopeRepresentativeCount: fittingRepresentativeCountsByView['envelope-untimed'],
    fittingTimedEnvelopeRepresentativeCount: fittingRepresentativeCountsByView['envelope-timed'],
    fittingPhysicalCaptureCount: fitting.envelope.physicalDetectedPowerCaptureCount,
    fittingSpectrumUnavailableWindowCount: fitting.spectrum.provenanceUnavailableWindowCount,
    fittingUnavailableWindowCount: fitting.envelope.provenanceUnavailableWindowCount,
    fittingPostCaptureUnavailableWindowCount:
      fitting.envelope.postCaptureProvenanceUnavailableWindowCount,
    tailCalibrationPhysicalCaptureCount: tailCalibration.envelope.physicalDetectedPowerCaptureCount,
    tailCalibrationSpectrumUnavailableWindowCount:
      tailCalibration.spectrum.provenanceUnavailableWindowCount,
    tailCalibrationUnavailableWindowCount: tailCalibration.envelope.provenanceUnavailableWindowCount,
    tailCalibrationPostCaptureUnavailableWindowCount:
      tailCalibration.envelope.postCaptureProvenanceUnavailableWindowCount,
  };
}

function numericSummaryFromCounts(counts) {
  const entries = Object.entries(counts)
    .filter(([, count]) => Number.isInteger(count) && count > 0)
    .map(([value, count]) => [Number(value), count])
    .sort(([left], [right]) => left - right);
  const total = entries.reduce((sum, [, count]) => sum + count, 0);
  if (entries.length === 0 || total === 0) return undefined;
  const medianIndex = Math.floor(total / 2);
  let cumulative = 0;
  let median = entries[0][0];
  for (const [value, count] of entries) {
    cumulative += count;
    if (cumulative > medianIndex) {
      median = value;
      break;
    }
  }
  return {
    minimum: entries[0][0],
    median,
    maximum: entries.at(-1)[0],
  };
}

function validateCausalClockAudit(
  report,
  path,
  failures,
  branch,
  expectedAttempts,
  expectedObservationHorizonCounts,
  expectedPhysicalCaptureCount,
) {
  const audit = objectAt(report, path);
  expectDeepEqual(
    failures,
    Object.keys(audit).sort(),
    [
      'branch',
      'sourceClock',
      'attempts',
      'allSourceLookIndicesUniqueWithinAttempt',
      'allSourceLookIndicesStrictlyIncreasingWithinAttempt',
      'allCapturesImmediatelyFollowTriggerSpectrum',
      'maximumDetectedPowerCapturesPerAttempt',
      'attemptsWithDetectedPowerCapture',
      'attemptsWithoutDetectedPowerCapture',
      'spectrumAcquisitionCount',
      'detectedPowerAcquisitionCount',
      'violationCount',
      'violations',
    ].sort(),
    `${path} key set`,
  );
  const expectedBranch = branch === 'consecutiveSpectrum'
    ? 'consecutive-spectrum'
    : 'qualified-envelope';
  const expectedSourceClock = branch === 'consecutiveSpectrum'
    ? PINNED_PRODUCTION_ACQUISITION_REGIME.sourceClocks.spectrum
    : PINNED_PRODUCTION_ACQUISITION_REGIME.sourceClocks.qualifiedEnvelope;
  expectEqual(failures, audit.branch, expectedBranch, `${path} runtime branch`);
  expectDeepEqual(failures, audit.sourceClock, expectedSourceClock, `${path} source-clock policy`);
  expectNonNegativeInteger(failures, audit.attempts, `${path}.attempts`, { positive: true });
  if (expectedAttempts !== undefined) {
    expectEqual(failures, audit.attempts, expectedAttempts, `${path} attempt denominator`);
  }
  expectEqual(
    failures,
    audit.attemptsWithDetectedPowerCapture,
    expectedPhysicalCaptureCount,
    `${path} physical-capture attempt denominator`,
  );
  for (const field of [
    'allSourceLookIndicesUniqueWithinAttempt',
    'allSourceLookIndicesStrictlyIncreasingWithinAttempt',
    'allCapturesImmediatelyFollowTriggerSpectrum',
  ]) expectEqual(failures, audit[field], true, `${path}.${field}`);
  for (const field of [
    'maximumDetectedPowerCapturesPerAttempt',
    'attemptsWithDetectedPowerCapture',
    'attemptsWithoutDetectedPowerCapture',
    'violationCount',
  ]) expectNonNegativeInteger(failures, audit[field], `${path}.${field}`);
  expectEqual(
    failures,
    audit.attemptsWithDetectedPowerCapture + audit.attemptsWithoutDetectedPowerCapture,
    audit.attempts,
    `${path} capture-attempt partition`,
  );
  expectEqual(
    failures,
    audit.maximumDetectedPowerCapturesPerAttempt,
    branch === 'consecutiveSpectrum' ? 0 : 1,
    `${path} branch-specific capture maximum`,
  );
  if (branch === 'consecutiveSpectrum') {
    expectEqual(failures, audit.attemptsWithDetectedPowerCapture, 0, `${path} no-auto-capture attempts`);
    expectEqual(failures, audit.attemptsWithoutDetectedPowerCapture, audit.attempts, `${path} no-auto-capture denominator`);
  }
  expectEqual(failures, audit.violationCount, 0, `${path} causal-clock violation count`);
  const violations = audit.violations;
  if (!Array.isArray(violations)) throw new Error(`${REPORT_PATH} ${path}.violations must be an array`);
  expectEqual(failures, violations.length, 0, `${path} causal-clock violations`);
  const expectedSummaries = {
    spectrumAcquisitionCount: numericSummaryFromCounts(expectedObservationHorizonCounts),
    detectedPowerAcquisitionCount: numericSummaryFromCounts({
      0: expectedAttempts - expectedPhysicalCaptureCount,
      1: expectedPhysicalCaptureCount,
    }),
  };
  for (const summaryName of ['spectrumAcquisitionCount', 'detectedPowerAcquisitionCount']) {
    const summary = audit[summaryName];
    if (summary === null || typeof summary !== 'object' || Array.isArray(summary)) {
      throw new Error(`${REPORT_PATH} ${path}.${summaryName} must be an object`);
    }
    expectDeepEqual(
      failures,
      Object.keys(summary).sort(),
      ['minimum', 'median', 'maximum'],
      `${path}.${summaryName} key set`,
    );
    for (const field of ['minimum', 'median', 'maximum']) {
      expectNonNegativeInteger(failures, summary[field], `${path}.${summaryName}.${field}`);
    }
    if (!(summary.minimum <= summary.median && summary.median <= summary.maximum)) {
      failures.push(`${path}.${summaryName} must be ordered minimum <= median <= maximum`);
    }
    expectDeepEqual(
      failures,
      summary,
      expectedSummaries[summaryName],
      `${path}.${summaryName} exact distribution summary`,
    );
  }
  return audit;
}

function validateHeldOutSourceSpanAudit(report, failures, expectedScenarioIds) {
  const audit = objectAt(report, 'matrix.heldOutSourceSpanAudit');
  const spectrumLiveStopExclusive = PINNED_SPECTRUM_RELEASE_GATE_SOURCE_PLAN.reduce(
    (nextSourceLookIndex, profile) => {
      expectEqual(
        failures,
        profile.sourceLookIndexOffset,
        nextSourceLookIndex,
        `spectrum release-gate source-plan ${profile.profileId} causal start`,
      );
      return profile.sourceLookIndexOffset
        + profile.spectrumOpportunities;
    },
    0,
  );
  const envelopeLiveStopExclusive = PINNED_QUALIFIED_ENVELOPE_RELEASE_GATE_SOURCE_PLAN.reduce(
    (nextSourceLookIndex, profile) => {
      expectEqual(
        failures,
        profile.sourceLookIndexOffset,
        nextSourceLookIndex,
        `qualified-envelope release-gate source-plan ${profile.profileId} causal start`,
      );
      return profile.sourceLookIndexOffset
        + profile.spectrumOpportunities
        + profile.admittedDetectedPowerCaptures;
    },
    0,
  );
  expectEqual(failures, spectrumLiveStopExclusive, 512, 'spectrum release-gate source range stop-exclusive');
  expectEqual(failures, envelopeLiveStopExclusive, 524, 'qualified-envelope release-gate source range stop-exclusive');
  expectEqual(
    failures,
    PINNED_VALIDATION_SPECTRUM_TEMPORAL_SCHEDULE.sourceLookIndexOffset,
    spectrumLiveStopExclusive,
    'held-out spectrum source range begins after the live spectrum release gate',
  );
  expectEqual(
    failures,
    PINNED_VALIDATION_QUALIFIED_ENVELOPE_TEMPORAL_SCHEDULE.sourceLookIndexOffset,
    envelopeLiveStopExclusive,
    'held-out envelope source range begins after the live qualified-envelope release gate',
  );
  expectEqual(failures, audit.sourceLookIndexStart, 512, 'held-out aggregate source-span start');
  expectEqual(failures, audit.sourceLookIndexStop, 620, 'held-out aggregate source-span inclusive stop');
  expectEqual(
    failures,
    audit.qualification,
    'declared-linear-drift-and-injected-signal-visibility-audit-v1',
    'held-out source-span qualification',
  );
  expectEqual(failures, audit.valid, true, 'held-out source-span audit validity');
  if (!Array.isArray(audit.scenarios) || audit.scenarios.length === 0) {
    throw new Error(`${REPORT_PATH} matrix.heldOutSourceSpanAudit.scenarios must be a non-empty array`);
  }
  expectEqual(
    failures,
    audit.scenarios.length,
    numberAt(report, 'corpus.scenarios', { integer: true }),
    'held-out source-span complete corpus scenario denominator',
  );
  const scenarioIds = [];
  for (const [index, scenario] of audit.scenarios.entries()) {
    if (scenario === null || typeof scenario !== 'object' || Array.isArray(scenario)
      || typeof scenario.scenarioId !== 'string' || scenario.scenarioId.length === 0) {
      failures.push(`held-out source-span scenario ${index} must publish an identity`);
      continue;
    }
    scenarioIds.push(scenario.scenarioId);
    expectEqual(failures, scenario.declaredDriftRemainsInSpan, true, `held-out ${scenario.scenarioId} declared drift`);
    expectEqual(failures, scenario.signalVisibilityValid, true, `held-out ${scenario.scenarioId} signal visibility`);
    expectEqual(failures, scenario.valid, true, `held-out ${scenario.scenarioId} source-span validity`);
    for (const field of ['driftHzPerLook', 'maximumAbsoluteDeclaredDriftHz', 'availableCenterDriftMarginHz']) {
      if (typeof scenario[field] !== 'number' || !Number.isFinite(scenario[field])) {
        failures.push(`held-out ${scenario.scenarioId}.${field} must be finite`);
      }
    }
    if (typeof scenario.maximumAbsoluteDeclaredDriftHz === 'number'
      && Number.isFinite(scenario.maximumAbsoluteDeclaredDriftHz)
      && scenario.maximumAbsoluteDeclaredDriftHz < 0) {
      failures.push(`held-out ${scenario.scenarioId}.maximumAbsoluteDeclaredDriftHz must be non-negative`);
    }
    if (typeof scenario.availableCenterDriftMarginHz === 'number'
      && Number.isFinite(scenario.availableCenterDriftMarginHz)
      && scenario.availableCenterDriftMarginHz < 0) {
      failures.push(`held-out ${scenario.scenarioId}.availableCenterDriftMarginHz must be non-negative`);
    }
    if (typeof scenario.driftHzPerLook === 'number'
      && Number.isFinite(scenario.driftHzPerLook)
      && typeof scenario.maximumAbsoluteDeclaredDriftHz === 'number'
      && Number.isFinite(scenario.maximumAbsoluteDeclaredDriftHz)) {
      expectEqual(
        failures,
        scenario.maximumAbsoluteDeclaredDriftHz,
        Math.abs(scenario.driftHzPerLook) * 616,
        `held-out ${scenario.scenarioId} exact maximum declared drift`,
      );
    }
    if (typeof scenario.maximumAbsoluteDeclaredDriftHz === 'number'
      && Number.isFinite(scenario.maximumAbsoluteDeclaredDriftHz)
      && typeof scenario.availableCenterDriftMarginHz === 'number'
      && Number.isFinite(scenario.availableCenterDriftMarginHz)
      && scenario.maximumAbsoluteDeclaredDriftHz > scenario.availableCenterDriftMarginHz) {
      failures.push(`held-out ${scenario.scenarioId} maximum declared drift must not exceed available center-drift margin`);
    }
    if (scenario.minimumInjectedSignalGainDb !== null
      && (typeof scenario.minimumInjectedSignalGainDb !== 'number'
        || !Number.isFinite(scenario.minimumInjectedSignalGainDb)
        || scenario.minimumInjectedSignalGainDb <= 1)) {
      failures.push(`held-out ${scenario.scenarioId} injected signal gain must be null or greater than 1 dB`);
    }
  }
  expectEqual(failures, new Set(scenarioIds).size, scenarioIds.length, 'held-out source-span unique scenario identities');
  expectDeepEqual(
    failures,
    [...scenarioIds].sort(),
    [...expectedScenarioIds].sort(),
    'held-out source-span complete corpus scenario identity set',
  );
  return audit;
}

function validateHighSnrKnownSeedCoverage(
  report,
  failures,
  expectedScenarioIds,
  validationSeeds,
  rbwDivisors,
) {
  const audit = objectAt(report, 'admission.highSnrUniqueSeedCoverage');
  expectDeepEqual(
    failures,
    audit.snrDb,
    PINNED_HIGH_SNR_SEED_COVERAGE_SNR_DB,
    'high-SNR seed-coverage SNR cells',
  );
  expectDeepEqual(
    failures,
    audit.validationSeeds,
    validationSeeds,
    'high-SNR seed-coverage validation seeds',
  );
  expectEqual(
    failures,
    audit.ordinaryKnownRequiredCoverage,
    PINNED_ORDINARY_KNOWN_SEED_COVERAGE,
    'ordinary known high-SNR seed-coverage policy',
  );
  expectEqual(
    failures,
    audit.bluetoothLeAdvertisingRequiredCoverage,
    PINNED_BLE_ADVERTISING_SEED_COVERAGE,
    'BLE advertising high-SNR seed-coverage policy',
  );
  expectEqual(
    failures,
    Array.isArray(audit.failures) ? audit.failures.length : -1,
    0,
    'high-SNR known seed-coverage failure count',
  );
  const byScenario = audit.byKnownScenario;
  if (byScenario === null || typeof byScenario !== 'object'
    || Array.isArray(byScenario)) {
    throw new Error(`${REPORT_PATH} admission.highSnrUniqueSeedCoverage.byKnownScenario must be an object`);
  }
  const expectedScenarios = [...new Set(expectedScenarioIds)].sort();
  expectDeepEqual(
    failures,
    Object.keys(byScenario).sort(),
    expectedScenarios,
    'high-SNR known seed-coverage scenario population',
  );
  const sortedValidationSeeds = [...validationSeeds]
    .sort((left, right) => left - right);
  const allowedRbwDivisors = new Set(rbwDivisors);
  for (const scenarioId of expectedScenarios) {
    const scenario = byScenario[scenarioId];
    if (scenario === null || typeof scenario !== 'object'
      || Array.isArray(scenario)) {
      failures.push(`high-SNR seed coverage ${scenarioId} must be an object`);
      continue;
    }
    const minimumCoverage = scenarioId === 'bluetooth-le-advertising'
      ? PINNED_BLE_ADVERTISING_SEED_COVERAGE
      : PINNED_ORDINARY_KNOWN_SEED_COVERAGE;
    const requiredSeeds = Math.ceil(validationSeeds.length * minimumCoverage);
    expectEqual(
      failures,
      scenario.minimumCoverage,
      minimumCoverage,
      `high-SNR seed coverage ${scenarioId} minimum coverage`,
    );
    expectEqual(
      failures,
      scenario.requiredSeeds,
      requiredSeeds,
      `high-SNR seed coverage ${scenarioId} required seeds`,
    );
    if (scenario.bySnr === null || typeof scenario.bySnr !== 'object'
      || Array.isArray(scenario.bySnr)) {
      failures.push(`high-SNR seed coverage ${scenarioId}.bySnr must be an object`);
      continue;
    }
    expectDeepEqual(
      failures,
      Object.keys(scenario.bySnr).sort(),
      PINNED_HIGH_SNR_SEED_COVERAGE_SNR_DB.map(String).sort(),
      `high-SNR seed coverage ${scenarioId} SNR key set`,
    );
    for (const snrDb of PINNED_HIGH_SNR_SEED_COVERAGE_SNR_DB) {
      const cell = scenario.bySnr[String(snrDb)];
      if (cell === null || typeof cell !== 'object' || Array.isArray(cell)) {
        failures.push(`high-SNR seed coverage ${scenarioId}/${snrDb} must be an object`);
        continue;
      }
      const coveredSeeds = cell.coveredSeeds;
      const uncoveredSeeds = cell.uncoveredSeeds;
      if (!Array.isArray(coveredSeeds) || !Array.isArray(uncoveredSeeds)
        || coveredSeeds.some((seed) => !Number.isFinite(seed))
        || uncoveredSeeds.some((seed) => !Number.isFinite(seed))) {
        failures.push(`high-SNR seed coverage ${scenarioId}/${snrDb} must publish finite covered/uncovered seed arrays`);
        continue;
      }
      const sortedCoveredSeeds = [...coveredSeeds]
        .sort((left, right) => left - right);
      const sortedUncoveredSeeds = [...uncoveredSeeds]
        .sort((left, right) => left - right);
      expectDeepEqual(
        failures,
        [...new Set([...sortedCoveredSeeds, ...sortedUncoveredSeeds])]
          .sort((left, right) => left - right),
        sortedValidationSeeds,
        `high-SNR seed coverage ${scenarioId}/${snrDb} seed partition`,
      );
      expectEqual(
        failures,
        sortedCoveredSeeds.length + sortedUncoveredSeeds.length,
        validationSeeds.length,
        `high-SNR seed coverage ${scenarioId}/${snrDb} unique seed denominator`,
      );
      expectEqual(
        failures,
        cell.uniqueSeedsCovered,
        coveredSeeds.length,
        `high-SNR seed coverage ${scenarioId}/${snrDb} covered seed count`,
      );
      expectEqual(
        failures,
        cell.totalSeeds,
        validationSeeds.length,
        `high-SNR seed coverage ${scenarioId}/${snrDb} total seed count`,
      );
      expectEqual(
        failures,
        cell.coverage,
        coveredSeeds.length / validationSeeds.length,
        `high-SNR seed coverage ${scenarioId}/${snrDb} coverage`,
      );
      expectEqual(
        failures,
        cell.requiredSeeds,
        requiredSeeds,
        `high-SNR seed coverage ${scenarioId}/${snrDb} required seed count`,
      );
      expectEqual(
        failures,
        cell.passed,
        true,
        `high-SNR seed coverage ${scenarioId}/${snrDb} acceptance`,
      );
      if (coveredSeeds.length < requiredSeeds) {
        failures.push(`high-SNR seed coverage ${scenarioId}/${snrDb} is below ${requiredSeeds} seeds`);
      }
      const admittingBySeed = cell.admittingRbwDivisorsBySeed;
      if (admittingBySeed === null || typeof admittingBySeed !== 'object'
        || Array.isArray(admittingBySeed)) {
        failures.push(`high-SNR seed coverage ${scenarioId}/${snrDb} admitting RBW map must be an object`);
        continue;
      }
      expectDeepEqual(
        failures,
        Object.keys(admittingBySeed).map(Number).sort((left, right) => left - right),
        sortedCoveredSeeds,
        `high-SNR seed coverage ${scenarioId}/${snrDb} admitting seed identities`,
      );
      for (const seed of coveredSeeds) {
        const divisors = admittingBySeed[String(seed)];
        if (!Array.isArray(divisors) || divisors.length === 0
          || divisors.some((divisor) => !allowedRbwDivisors.has(divisor))) {
          failures.push(`high-SNR seed coverage ${scenarioId}/${snrDb}/${seed} must publish at least one tested admitting RBW divisor`);
        }
      }
    }
  }
}

function collectMetrics(report, failures, expectedRollingScenarioIds) {
  const pairs = arrayAt(report, 'corpus.exactObservableEquivalencePairAudit.pairs');
  const nuisanceCells = sumPairMetric(pairs, 'nuisanceCells');
  const representativePairs = sumPairMetric(pairs, 'matchedRepresentativePairs');
  const evidenceViewPairs = sumPairMetric(pairs, 'matchedEvidenceViewPairs');
  const onlineSpectrumPairs = sumPairMetric(pairs, 'matchedOnlineSpectrumPairs');
  const pairDiscrepancies = sumPairMetric(pairs, 'discrepancyCount');
  for (const [index, pair] of pairs.entries()) {
    if (pair.matchedOnlineSpectrumPairs <= 0) {
      failures.push(`exact-equivalence pair ${index} online-spectrum pair count must be positive`);
    }
  }
  const reportedDiscrepancies = numberAt(
    report,
    'corpus.exactObservableEquivalencePairAudit.discrepancyCount',
    { integer: true },
  );
  expectEqual(
    failures,
    reportedDiscrepancies,
    pairDiscrepancies,
    'exact-equivalence aggregate discrepancy count',
  );
  if (onlineSpectrumPairs <= 0) failures.push('exact-equivalence online-spectrum pair count must be positive');

  const representatives = numberAt(
    report,
    'classificationConditionalOnAdmission.samples',
    { integer: true },
  );
  const fitEligibleSamples = numberAt(
    report,
    'classificationConditionalOnAdmission.identifiableFitEligibleSamples',
    { integer: true },
  );
  const fitEligibleKnownSamples = numberAt(
    report,
    'classificationConditionalOnAdmission.identifiableFitEligibleKnownSamples',
    { integer: true },
  );
  const properScoreSamples = numberAt(
    report,
    'classificationConditionalOnAdmission.singletonAllowedTruthProperScoreSamples',
    { integer: true },
  );
  const scenarioExcludedUnknownSamples = numberAt(
    report,
    'classificationConditionalOnAdmission.scenarioExcludedUnknownSamples',
    { integer: true },
  );
  const strictUnknownSamples = numberAt(
    report,
    'classificationConditionalOnAdmission.scenarioExcludedStrictUnknownSamples',
    { integer: true },
  );
  const exactEquivalenceSamples = numberAt(
    report,
    'classificationConditionalOnAdmission.exactEquivalenceSamples',
    { integer: true },
  );
  for (const [value, label] of [
    [properScoreSamples, 'singleton-truth proper-score population'],
    [fitEligibleKnownSamples, 'fit-eligible known population'],
    [fitEligibleSamples - fitEligibleKnownSamples, 'fitted unknown-template population'],
    [scenarioExcludedUnknownSamples, 'scenario-excluded unknown population'],
    [strictUnknownSamples, 'strict unknown holdout population'],
    [exactEquivalenceSamples, 'exact-equivalence admission population'],
  ]) {
    if (!Number.isInteger(value) || value <= 0) {
      failures.push(`${label} must be a positive integer`);
    }
  }
  if (fitEligibleKnownSamples > fitEligibleSamples
    || properScoreSamples > fitEligibleSamples
    || fitEligibleSamples > representatives
    || scenarioExcludedUnknownSamples > representatives
    || strictUnknownSamples > scenarioExcludedUnknownSamples
    || exactEquivalenceSamples > scenarioExcludedUnknownSamples) {
    failures.push('admission-conditional acceptance populations are not nested within their declared denominators');
  }
  const qualifiedEnvelopeRepresentatives = numberAt(
    report,
    'admission.causalEnvelopeSamples',
    { integer: true },
  );
  if (representatives <= 0) failures.push('capture-conditional representative count must be positive');
  if (qualifiedEnvelopeRepresentatives <= 0) failures.push('qualified envelope representative count must be positive');
  for (const path of [
    'admission.captureConditionalClassificationSamples',
    'admission.expectedCaptureConditionalClassificationSamples',
    'admission.uniqueCaptureConditionalClassificationSamples',
    'admission.physicalDetectedPowerCaptures',
  ]) {
    expectEqual(failures, numberAt(report, path, { integer: true }), representatives, path);
  }
  for (const path of [
    'admission.expectedCausalEnvelopeSamples',
    'admission.uniqueCausalEnvelopeSamples',
    'admission.physicalEnvelopeCaptures',
  ]) {
    expectEqual(
      failures,
      numberAt(report, path, { integer: true }),
      qualifiedEnvelopeRepresentatives,
      path,
    );
  }
  const associationPath = 'classificationConditionalOnAdmission.association';
  const firstReadySelectionModes = objectAt(report, `${associationPath}.firstReadySelectionModes`);
  const soleEnvelopeTargetModes = objectAt(report, `${associationPath}.soleEnvelopeTargetModes`);
  const soleEnvelopeByMode = objectAt(report, `${associationPath}.soleEnvelopeByMode`);
  const associationByMode = objectAt(report, `${associationPath}.byMode`);
  const captureProjectionKinds = objectAt(
    report,
    `${associationPath}.captureProjectionKinds`,
  );
  const rawCaptureTargetStates = objectAt(
    report,
    `${associationPath}.rawCaptureTargetStates`,
  );
  const detectedPowerEvidenceDispositions = objectAt(
    report,
    `${associationPath}.detectedPowerEvidenceDispositions`,
  );
  const completeSpectrumOnline = objectAt(report, `${associationPath}.completeSpectrumOnline`);
  const spectrumOnlineByMode = objectAt(report, `${associationPath}.completeSpectrumOnline.byMode`);
  for (const [object, label] of [
    [firstReadySelectionModes, 'first-ready physical capture association-mode key set'],
    [soleEnvelopeTargetModes, 'sole-envelope physical target-mode key set'],
  ]) {
    expectDeepEqual(
      failures,
      Object.keys(object).sort(),
      [...PINNED_PHYSICAL_CAPTURE_ASSOCIATION_MODES].sort(),
      label,
    );
  }
  for (const [object, label] of [
    [soleEnvelopeByMode, 'sole-envelope by-mode key set'],
    [associationByMode, 'association by-mode key set'],
    [spectrumOnlineByMode, 'complete spectrum-online by-mode key set'],
  ]) {
    expectDeepEqual(
      failures,
      Object.keys(object).sort(),
      [...PINNED_ASSOCIATION_MODES].sort(),
      label,
    );
  }
  for (const [population, expectedKeys, label] of [
    [
      captureProjectionKinds,
      PINNED_CAPTURE_PROJECTION_KINDS,
      'capture projection-kind population',
    ],
    [
      rawCaptureTargetStates,
      PINNED_RAW_CAPTURE_TARGET_STATES,
      'raw capture-target state population',
    ],
  ]) {
    expectDeepEqual(
      failures,
      Object.keys(population).sort(),
      [...expectedKeys].sort(),
      `${label} key set`,
    );
    let populationCount = 0;
    for (const key of expectedKeys) {
      expectNonNegativeInteger(
        failures,
        population[key],
        `${label} ${key}`,
        { positive: true },
      );
      if (Number.isInteger(population[key])) populationCount += population[key];
    }
    expectEqual(
      failures,
      populationCount,
      representatives,
      `${label} denominator`,
    );
  }
  expectDeepEqual(
    failures,
    Object.keys(detectedPowerEvidenceDispositions).sort(),
    ['admitted-envelope', 'censored-frequency-agile-spectrum-only'],
    'detected-power evidence-disposition key set',
  );
  expectEqual(
    failures,
    detectedPowerEvidenceDispositions['admitted-envelope'],
    qualifiedEnvelopeRepresentatives,
    'admitted-envelope evidence-disposition denominator',
  );
  const censoredSpectrumRepresentatives = numberAt(
    report,
    'admission.detectedPowerCaptureOutcomes.censoredSpectrumClassificationCount',
    { integer: true },
  );
  expectEqual(
    failures,
    detectedPowerEvidenceDispositions['censored-frequency-agile-spectrum-only'],
    censoredSpectrumRepresentatives,
    'censored spectrum evidence-disposition denominator',
  );
  let associationModeRepresentatives = 0;
  let spectrumOnlineSamples = 0;
  for (const associationMode of PINNED_ASSOCIATION_MODES) {
    const mode = associationByMode[associationMode];
    const envelopeMode = soleEnvelopeByMode[associationMode];
    const spectrumMode = spectrumOnlineByMode[associationMode];
    if (mode === null || typeof mode !== 'object' || Array.isArray(mode)) {
      throw new Error(`${REPORT_PATH} association mode ${associationMode} must be an object`);
    }
    if (envelopeMode === null || typeof envelopeMode !== 'object'
      || Array.isArray(envelopeMode)) {
      throw new Error(`${REPORT_PATH} sole-envelope mode ${associationMode} must be an object`);
    }
    if (spectrumMode === null || typeof spectrumMode !== 'object' || Array.isArray(spectrumMode)) {
      throw new Error(`${REPORT_PATH} complete spectrum-online mode ${associationMode} must be an object`);
    }
    const samples = mode.firstReadyRepresentativeSamples;
    const scenarios = mode.scenarios;
    if (!expectNonNegativeInteger(
      failures,
      samples,
      `association mode ${associationMode} first-ready samples`,
      { positive: true },
    ) || !Array.isArray(scenarios) || scenarios.length <= 0) {
      failures.push(`association mode ${associationMode} must publish positive capture-classification sample and scenario coverage`);
    }
    associationModeRepresentatives += Number.isInteger(samples) ? samples : 0;
    const envelopeSamples = envelopeMode.firstReadyRepresentativeSamples;
    const envelopeScenarios = envelopeMode.scenarios;
    if (associationMode === 'frequency-agile-2g4-activity') {
      expectEqual(failures, envelopeSamples, 0, 'frequency-agile qualified-envelope sample count');
      expectDeepEqual(failures, envelopeScenarios, [], 'frequency-agile qualified-envelope scenarios');
    } else if (!expectNonNegativeInteger(
      failures,
      envelopeSamples,
      `sole-envelope mode ${associationMode} samples`,
      { positive: true },
    ) || !Array.isArray(envelopeScenarios) || envelopeScenarios.length <= 0) {
      failures.push(`sole-envelope mode ${associationMode} must publish positive sample and scenario coverage`);
    }

    for (const field of ['samples', 'attempts']) {
      expectNonNegativeInteger(
        failures,
        spectrumMode[field],
        `complete spectrum-online ${associationMode}.${field}`,
        { positive: true },
      );
    }
    if (!Array.isArray(spectrumMode.scenarios) || spectrumMode.scenarios.length <= 0) {
      failures.push(`complete spectrum-online ${associationMode} must publish positive scenario coverage`);
    }
    spectrumOnlineSamples += Number.isInteger(spectrumMode.samples) ? spectrumMode.samples : 0;
  }
  expectEqual(
    failures,
    associationModeRepresentatives,
    representatives,
    'complete capture-classification association-mode denominator',
  );
  expectEqual(
    failures,
    PINNED_ASSOCIATION_MODES.reduce((sum, mode) =>
      sum + (Number.isInteger(soleEnvelopeByMode[mode].firstReadyRepresentativeSamples)
        ? soleEnvelopeByMode[mode].firstReadyRepresentativeSamples : 0), 0),
    qualifiedEnvelopeRepresentatives,
    'complete qualified-envelope association-mode denominator',
  );
  const physicalRawTargetRepresentatives = Object.values(
    firstReadySelectionModes,
  ).reduce((sum, value) => sum + (Number.isInteger(value) ? value : 0), 0);
  expectEqual(
    failures,
    physicalRawTargetRepresentatives,
    representatives,
    'complete physical raw-target association-mode denominator',
  );
  const soleEnvelopeRawTargets = Object.values(soleEnvelopeTargetModes)
    .reduce((sum, value) => sum + (Number.isInteger(value) ? value : 0), 0);
  expectEqual(
    failures,
    soleEnvelopeRawTargets,
    qualifiedEnvelopeRepresentatives,
    'complete qualified-envelope raw-target association-mode denominator',
  );
  expectEqual(
    failures,
    captureProjectionKinds['current-active-physical-representative'],
    qualifiedEnvelopeRepresentatives,
    'direct projection/qualified-envelope denominator',
  );
  expectEqual(
    failures,
    captureProjectionKinds['current-qualified-agile-latest-member'],
    censoredSpectrumRepresentatives,
    'agile projection/censored-spectrum denominator',
  );
  const captureOutcomes = objectAt(
    report,
    'admission.detectedPowerCaptureOutcomes',
  );
  expectEqual(failures, captureOutcomes.schemaVersion, 1, 'detected-power capture-outcome schema');
  expectDeepEqual(
    failures,
    captureOutcomes.censoringPolicy,
    PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORING_POLICY,
    'detected-power capture-outcome censoring policy',
  );
  expectEqual(
    failures,
    captureOutcomes.physicalDetectedPowerCaptureCount,
    representatives,
    'capture-outcome physical capture denominator',
  );
  expectEqual(
    failures,
    captureOutcomes.receiptQualifiedPhysicalCaptureCount,
    representatives,
    'capture-outcome receipt-qualified physical denominator',
  );
  expectEqual(
    failures,
    captureOutcomes.qualifiedEnvelopeSampleCount,
    qualifiedEnvelopeRepresentatives,
    'capture-outcome qualified-envelope denominator',
  );
  expectEqual(
    failures,
    captureOutcomes.censoredDetectedPowerCaptureCount,
    censoredSpectrumRepresentatives,
    'capture-outcome censored-capture denominator',
  );
  expectEqual(
    failures,
    representatives,
    qualifiedEnvelopeRepresentatives + censoredSpectrumRepresentatives,
    'physical capture qualified-envelope/censored-spectrum partition',
  );
  const censoringAudit = objectAt(
    report,
    'admission.frequencyAgileEnvelopeCensoring',
  );
  expectEqual(
    failures,
    censoringAudit.policyId,
    PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORING_POLICY.id,
    'frequency-agile envelope-censoring audit policy',
  );
  expectEqual(
    failures,
    censoringAudit.limitation,
    PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORING_LIMITATION,
    'frequency-agile production censoring limitation',
  );
  expectEqual(failures, censoringAudit.physicalCapturesCensored, censoredSpectrumRepresentatives, 'frequency-agile censored physical captures');
  expectEqual(failures, censoringAudit.spectrumOnlyClassifications, censoredSpectrumRepresentatives, 'frequency-agile censored spectrum classifications');
  expectEqual(failures, censoringAudit.uncensoredFrequencyAgileEnvelopeSamples, 0, 'frequency-agile uncensored envelope samples');
  const selectedEvidenceViews = objectAt(
    report,
    'admission.detectedPowerCaptureOutcomes.selectedEvidenceViews',
  );
  expectEqual(
    failures,
    selectedEvidenceViews['spectrum-only'],
    censoredSpectrumRepresentatives,
    'selected spectrum-only/censored capture denominator',
  );
  expectEqual(
    failures,
    Object.values(selectedEvidenceViews).reduce(
      (sum, value) => sum + (Number.isInteger(value) ? value : 0),
      0,
    ),
    representatives,
    'selected evidence-view capture denominator',
  );
  const outcomesByProjectedMode = objectAt(
    report,
    'admission.detectedPowerCaptureOutcomes.byProjectedMode',
  );
  expectDeepEqual(
    failures,
    Object.keys(outcomesByProjectedMode).sort(),
    [...PINNED_ASSOCIATION_MODES].sort(),
    'capture outcomes projected-mode key set',
  );
  const validateCaptureOutcomeSummary = (summary, label) => {
    if (summary === null || typeof summary !== 'object' || Array.isArray(summary)) {
      throw new Error(`${REPORT_PATH} ${label} must be an object`);
    }
    expectDeepEqual(
      failures,
      Object.keys(summary).sort(),
      [
        'physicalCaptureCount',
        'receiptQualifiedPhysicalCaptureCount',
        'qualifiedEnvelopeSampleCount',
        'censoredDetectedPowerCaptureCount',
        'censoredSpectrumClassificationCount',
        'selectedEvidenceViews',
      ].sort(),
      `${label} key set`,
    );
    for (const field of [
      'physicalCaptureCount',
      'receiptQualifiedPhysicalCaptureCount',
      'qualifiedEnvelopeSampleCount',
      'censoredDetectedPowerCaptureCount',
      'censoredSpectrumClassificationCount',
    ]) expectNonNegativeInteger(failures, summary[field], `${label}.${field}`);
    expectEqual(failures, summary.receiptQualifiedPhysicalCaptureCount, summary.physicalCaptureCount, `${label} receipt/physical reconciliation`);
    expectEqual(failures, summary.physicalCaptureCount, summary.qualifiedEnvelopeSampleCount + summary.censoredDetectedPowerCaptureCount, `${label} envelope/censored capture partition`);
    expectEqual(failures, summary.censoredDetectedPowerCaptureCount, summary.censoredSpectrumClassificationCount, `${label} censored capture/classification reconciliation`);
    const views = summary.selectedEvidenceViews;
    if (views === null || typeof views !== 'object' || Array.isArray(views)) {
      throw new Error(`${REPORT_PATH} ${label}.selectedEvidenceViews must be an object`);
    }
    expectEqual(failures, views['spectrum-only'] ?? 0, summary.censoredSpectrumClassificationCount, `${label} selected spectrum/censored reconciliation`);
    expectEqual(
      failures,
      Object.values(views).reduce(
        (sum, value) => sum + (Number.isInteger(value) ? value : 0),
        0,
      ),
      summary.physicalCaptureCount,
      `${label} selected-view denominator`,
    );
    return summary;
  };
  for (const [mode, summary] of Object.entries(outcomesByProjectedMode)) {
    validateCaptureOutcomeSummary(summary, `capture outcomes projected mode ${mode}`);
  }
  const agileOutcomes = outcomesByProjectedMode['frequency-agile-2g4-activity'];
  expectNonNegativeInteger(
    failures,
    agileOutcomes.physicalCaptureCount,
    'frequency-agile physical capture count',
    { positive: true },
  );
  expectEqual(failures, agileOutcomes.qualifiedEnvelopeSampleCount, 0, 'frequency-agile qualified-envelope count');
  expectEqual(
    failures,
    agileOutcomes.censoredSpectrumClassificationCount,
    agileOutcomes.physicalCaptureCount,
    'frequency-agile censored-spectrum/physical count',
  );
  const outcomesByProjectionKind = objectAt(
    report,
    'admission.detectedPowerCaptureOutcomes.byProjectionKind',
  );
  expectDeepEqual(
    failures,
    Object.keys(outcomesByProjectionKind).sort(),
    [...PINNED_CAPTURE_PROJECTION_KINDS].sort(),
    'capture outcomes projection-kind key set',
  );
  for (const [kind, summary] of Object.entries(outcomesByProjectionKind)) {
    validateCaptureOutcomeSummary(summary, `capture outcomes projection kind ${kind}`);
  }
  expectEqual(
    failures,
    outcomesByProjectionKind['current-qualified-agile-latest-member']
      .censoredSpectrumClassificationCount,
    censoredSpectrumRepresentatives,
    'agile projection-kind censored-spectrum count',
  );
  expectEqual(
    failures,
    outcomesByProjectionKind['current-qualified-agile-latest-member']
      .qualifiedEnvelopeSampleCount,
    0,
    'agile projection-kind qualified-envelope count',
  );
  expectEqual(
    failures,
    outcomesByProjectionKind['current-active-physical-representative']
      .qualifiedEnvelopeSampleCount,
    qualifiedEnvelopeRepresentatives,
    'direct projection-kind qualified-envelope count',
  );
  expectEqual(
    failures,
    outcomesByProjectionKind['current-active-physical-representative']
      .censoredSpectrumClassificationCount,
    0,
    'direct projection-kind censored-spectrum count',
  );
  const outcomesByScenario = objectAt(
    report,
    'admission.detectedPowerCaptureOutcomes.byScenario',
  );
  expectDeepEqual(
    failures,
    Object.keys(outcomesByScenario).sort(),
    [...arrayAt(report, 'matrix.scenarioSelection.scenarioIds')].sort(),
    'capture outcomes scenario key set',
  );
  for (const [scenarioId, summary] of Object.entries(outcomesByScenario)) {
    validateCaptureOutcomeSummary(summary, `capture outcomes scenario ${scenarioId}`);
  }
  for (const [group, summaries] of [
    ['projected-mode', outcomesByProjectedMode],
    ['projection-kind', outcomesByProjectionKind],
    ['scenario', outcomesByScenario],
  ]) {
    for (const [field, expected] of [
      ['physicalCaptureCount', captureOutcomes.physicalDetectedPowerCaptureCount],
      ['receiptQualifiedPhysicalCaptureCount', captureOutcomes.receiptQualifiedPhysicalCaptureCount],
      ['qualifiedEnvelopeSampleCount', captureOutcomes.qualifiedEnvelopeSampleCount],
      ['censoredDetectedPowerCaptureCount', captureOutcomes.censoredDetectedPowerCaptureCount],
      ['censoredSpectrumClassificationCount', captureOutcomes.censoredSpectrumClassificationCount],
    ]) {
      expectEqual(
        failures,
        Object.values(summaries).reduce(
          (sum, summary) => sum + (Number.isInteger(summary[field])
            ? summary[field] : 0),
          0,
        ),
        expected,
        `capture outcomes ${group} ${field} subtotal`,
      );
    }
  }
  for (const scenarioId of PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORED_SCENARIO_IDS) {
    const outcome = outcomesByScenario[scenarioId];
    expectNonNegativeInteger(
      failures,
      outcome.physicalCaptureCount,
      `${scenarioId} physical capture count`,
      { positive: true },
    );
    expectEqual(failures, outcome.qualifiedEnvelopeSampleCount, 0, `${scenarioId} qualified-envelope count`);
    expectEqual(
      failures,
      outcome.censoredSpectrumClassificationCount,
      outcome.physicalCaptureCount,
      `${scenarioId} censored-spectrum/physical count`,
    );
    expectEqual(
      failures,
      outcome.receiptQualifiedPhysicalCaptureCount,
      outcome.physicalCaptureCount,
      `${scenarioId} receipt-qualified/physical count`,
    );
  }
  const completeSpectrumSampleCount = completeSpectrumOnline.samples;
  expectNonNegativeInteger(
    failures,
    completeSpectrumSampleCount,
    'complete spectrum-online sample count',
    { positive: true },
  );
  expectEqual(
    failures,
    completeSpectrumOnline.uniqueSamples,
    completeSpectrumSampleCount,
    'complete spectrum-online unique sample denominator',
  );
  if (!Array.isArray(completeSpectrumOnline.duplicateKeys)) {
    throw new Error(`${REPORT_PATH} complete spectrum-online duplicateKeys must be an array`);
  }
  expectEqual(failures, completeSpectrumOnline.duplicateKeys.length, 0, 'complete spectrum-online duplicate keys');
  expectEqual(
    failures,
    spectrumOnlineSamples,
    completeSpectrumSampleCount,
    'complete spectrum-online by-mode sample denominator',
  );
  if (completeSpectrumSampleCount < representatives) {
    failures.push('complete spectrum-online population cannot be smaller than the sole-envelope population');
  }

  const completeOnlineSpectrumPath =
    'productionRollingWindowValidation.completeOnlineSpectrumAudit';
  expectEqual(
    failures,
    valueAt(report, `${completeOnlineSpectrumPath}.qualification`),
    'held-out-all-truths-all-snrs-all-online-ready-representatives',
    'complete online spectrum qualification',
  );
  const completeOnlineSpectrumCases = numberAt(
    report,
    `${completeOnlineSpectrumPath}.cases`,
    { integer: true },
  );
  if (completeOnlineSpectrumCases <= 0) failures.push('complete online spectrum case count must be positive');
  expectEqual(
    failures,
    numberAt(report, `${completeOnlineSpectrumPath}.uniqueCases`, { integer: true }),
    completeOnlineSpectrumCases,
    'complete online spectrum unique case denominator',
  );
  expectEqual(
    failures,
    completeOnlineSpectrumCases,
    completeSpectrumSampleCount,
    'complete online spectrum/association sample denominator',
  );
  const completeOnlineUnknownTruthCases = numberAt(
    report,
    `${completeOnlineSpectrumPath}.unknownTruthCases`,
    { integer: true },
  );
  if (completeOnlineUnknownTruthCases <= 0
    || completeOnlineUnknownTruthCases > completeOnlineSpectrumCases) {
    failures.push('complete online spectrum unknown-truth denominator must be positive and bounded by all cases');
  }
  expectEqual(
    failures,
    numberAt(report, `${completeOnlineSpectrumPath}.unknownTruthFalseAcceptCount`, { integer: true }),
    0,
    'complete online spectrum unknown-truth false accepts',
  );
  expectEqual(
    failures,
    numberAt(report, `${completeOnlineSpectrumPath}.incompatibleNonUnknownCount`, { integer: true }),
    0,
    'complete online spectrum incompatible non-unknown decisions',
  );
  expectEqual(
    failures,
    arrayAt(report, `${completeOnlineSpectrumPath}.failures`).length,
    0,
    'complete online spectrum failure examples',
  );
  const completeOnlineProperScoreSamples = numberAt(
    report,
    `${completeOnlineSpectrumPath}.singletonAllowedTruthProperScoreSamples`,
    { integer: true },
  );
  if (completeOnlineProperScoreSamples <= 0
    || completeOnlineProperScoreSamples > completeOnlineSpectrumCases) {
    failures.push('complete online spectrum proper-score denominator must be positive and bounded by all cases');
  }
  const completeOnlineLogLoss = numberAt(report, `${completeOnlineSpectrumPath}.fittedTemplateLogLoss`);
  const completeOnlineBrier = numberAt(
    report,
    `${completeOnlineSpectrumPath}.fittedTemplateMulticlassBrier`,
  );
  const completeOnlineEce = numberAt(
    report,
    `${completeOnlineSpectrumPath}.fittedTemplateExpectedCalibrationError`,
  );
  expectRange(failures, completeOnlineLogLoss, 0, 0.5, 'complete online spectrum fitted-template log loss');
  expectRange(failures, completeOnlineBrier, 0, 0.2, 'complete online spectrum fitted-template Brier score');
  expectRange(failures, completeOnlineEce, 0, 0.1, 'complete online spectrum fitted-template ECE');
  const completeOnlineByTruth = objectAt(report, `${completeOnlineSpectrumPath}.byTruth`);
  expectDeepEqual(
    failures,
    Object.keys(completeOnlineByTruth).sort(),
    [...PINNED_CLASS_IDS].sort(),
    'complete online spectrum truth key set',
  );
  let completeOnlineTruthCount = 0;
  for (const classId of PINNED_CLASS_IDS) {
    if (expectNonNegativeInteger(
      failures,
      completeOnlineByTruth[classId],
      `complete online spectrum ${classId} truth count`,
      { positive: true },
    )) completeOnlineTruthCount += completeOnlineByTruth[classId];
  }
  expectEqual(
    failures,
    completeOnlineTruthCount,
    completeOnlineSpectrumCases,
    'complete online spectrum truth denominator',
  );
  expectEqual(
    failures,
    completeOnlineByTruth['unknown-signal'],
    completeOnlineUnknownTruthCases,
    'complete online spectrum unknown-truth population',
  );

  const evidenceViews = objectAt(report, 'classificationConditionalOnAdmission.evidenceViews');
  expectDeepEqual(
    failures,
    Object.keys(evidenceViews).sort(),
    ['envelope-untimed', 'spectrum-only'],
    'admission-conditional evidence-view key set',
  );
  for (const view of ['spectrum-only', 'envelope-untimed']) {
    const viewMetrics = evidenceViews[view];
    if (viewMetrics === null || typeof viewMetrics !== 'object' || Array.isArray(viewMetrics)) {
      throw new Error(`${REPORT_PATH} evidence view ${view} must be an object`);
    }
    expectEqual(
      failures,
      viewMetrics.admittedSamples,
      view === 'spectrum-only'
        ? representatives
        : qualifiedEnvelopeRepresentatives,
      `${view} admission-conditional denominator`,
    );
    expectEqual(failures, viewMetrics.falseAcceptedUnknownCount, 0, `${view} unknown false accepts`);
    expectEqual(failures, viewMetrics.anyFalseAcceptAttemptCount, 0, `${view} false-accept attempts`);
    if (!Array.isArray(viewMetrics.anyFalseAcceptAttemptIds)
      || !Array.isArray(viewMetrics.falseAcceptedUnknownExamples)) {
      throw new Error(`${REPORT_PATH} evidence view ${view} false-accept diagnostics must be arrays`);
    }
    expectEqual(failures, viewMetrics.anyFalseAcceptAttemptIds.length, 0, `${view} false-accept attempt IDs`);
    expectEqual(failures, viewMetrics.falseAcceptedUnknownExamples.length, 0, `${view} false-accept examples`);
    if (!Number.isInteger(viewMetrics.exactEquivalenceSamples)
      || viewMetrics.exactEquivalenceSamples <= 0
      || !Number.isInteger(viewMetrics.strictHoldoutSamples)
      || viewMetrics.strictHoldoutSamples <= 0
      || !Number.isInteger(viewMetrics.singletonAllowedTruthProperScoreSamples)
      || viewMetrics.singletonAllowedTruthProperScoreSamples <= 0) {
      failures.push(`${view} must publish positive exact-equivalence, strict-holdout, and proper-score populations`);
    }
    expectEqual(failures, viewMetrics.exactEquivalenceCompatibleRate, 1, `${view} exact-equivalence compatibility`);
    expectEqual(failures, viewMetrics.strictHoldoutRejectionRate, 1, `${view} strict-holdout rejection`);
    expectRange(failures, viewMetrics.knownCoverage, 0.8, 1, `${view} known coverage`);
    expectRange(
      failures,
      viewMetrics.coveredKnownHierarchicalAccuracy,
      0.9,
      1,
      `${view} covered-known hierarchical accuracy`,
    );
    expectRange(failures, viewMetrics.fittedTemplateLogLoss, 0, 0.5, `${view} fitted-template log loss`);
    expectRange(
      failures,
      viewMetrics.fittedTemplateMulticlassBrier,
      0,
      0.2,
      `${view} fitted-template Brier score`,
    );
    expectRange(
      failures,
      viewMetrics.fittedTemplateExpectedCalibrationError,
      0,
      0.1,
      `${view} fitted-template ECE`,
    );
    expectRange(
      failures,
      viewMetrics.scenarioExcludedStrictSupportAuroc,
      0.9,
      1,
      `${view} strict scenario-excluded support AUROC`,
    );
  }
  const admissionConditionalLimitations = objectAt(
    report,
    'classificationConditionalOnAdmission.limitations',
  );
  expectEqual(
    failures,
    admissionConditionalLimitations[
      PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORING_LIMITATION
    ],
    censoredSpectrumRepresentatives,
    'frequency-agile production censoring limitation/classification reconciliation',
  );
  for (const limitation of [
    'zero-span-missing',
    'zero-span-tune-mismatch',
    'zero-span-provenance-mismatch',
    'zero-span-spectrum-window-mismatch',
    'zero-span-acquisition-policy-unqualified',
    'zero-span-geometry-out-of-domain',
  ]) {
    expectEqual(
      failures,
      admissionConditionalLimitations[limitation] ?? 0,
      0,
      `admission-conditional envelope rejection limitation ${limitation}`,
    );
  }

  const blePath = 'admission.highSnrUniqueSeedCoverage.byKnownScenario.bluetooth-le-advertising.bySnr';
  const ble24Covered = numberAt(report, `${blePath}.24.uniqueSeedsCovered`, { integer: true });
  const ble24Total = numberAt(report, `${blePath}.24.totalSeeds`, { integer: true });
  const ble32Covered = numberAt(report, `${blePath}.32.uniqueSeedsCovered`, { integer: true });
  const ble32Total = numberAt(report, `${blePath}.32.totalSeeds`, { integer: true });
  expectEqual(failures, ble24Total, ble32Total, 'BLE high-SNR seed denominator');
  if (ble24Total <= 0 || ble24Covered < 0 || ble32Covered < 0
    || ble24Covered > ble24Total || ble32Covered > ble32Total) {
    failures.push('BLE high-SNR seed coverage must use positive denominators and covered counts within each denominator');
  }

  const bleRepresentatives = numberAt(
    report,
    'classificationConditionalOnAdmission.association.byScenario.bluetooth-le-advertising.firstReadyRepresentativeSamples',
    { integer: true },
  );
  const bleBluetoothLike = numberAt(
    report,
    'classificationConditionalOnAdmission.association.byScenario.bluetooth-le-advertising.results.observable:bluetooth-like',
    { integer: true },
  );
  expectEqual(
    failures,
    bleBluetoothLike,
    bleRepresentatives,
    'published all-admitted-BLE-resolved-to-Bluetooth-like claim',
  );
  const bleResults = valueAt(
    report,
    'classificationConditionalOnAdmission.association.byScenario.bluetooth-le-advertising.results',
  );
  if (bleResults === null || typeof bleResults !== 'object' || Array.isArray(bleResults)) {
    throw new Error(`${REPORT_PATH} admitted BLE result counts must be an object`);
  }
  const bleResultEntries = Object.entries(bleResults);
  if (bleResultEntries.some(([, count]) => !Number.isInteger(count) || count < 0)) {
    failures.push('admitted BLE result counts must be non-negative integers');
  }
  expectEqual(
    failures,
    bleResultEntries.reduce((sum, [, count]) => sum + count, 0),
    bleRepresentatives,
    'admitted BLE result denominator',
  );
  for (const [label, count] of bleResultEntries) {
    if (label !== 'observable:bluetooth-like' && count !== 0) {
      failures.push(`admitted BLE result ${label} must be zero to publish an exclusive Bluetooth-like result`);
    }
  }

  const nuisanceSeeds = arrayAt(report, 'matrix.nuisanceShiftSeeds');
  const snrDb = arrayAt(report, 'matrix.snrDb');
  const rbwDivisors = arrayAt(report, 'matrix.rbwDivisors');
  const strictUnknownHoldouts = arrayAt(
    report,
    'corpus.manifestSplit.validatorOwnedPins.strictUnknownHoldout',
  ).length;
  const ambiguityStressCases = arrayAt(
    report,
    'corpus.manifestSplit.validatorOwnedPins.observableAmbiguityStress',
  ).length;
  const knownAcquisitionValidationCases = arrayAt(
    report,
    'corpus.manifestSplit.validatorOwnedPins.knownAcquisitionValidationOnly',
  ).length;
  for (const [path, values] of [
    ['matrix.nuisanceShiftSeeds', nuisanceSeeds],
    ['matrix.snrDb', snrDb],
    ['matrix.rbwDivisors', rbwDivisors],
  ]) {
    if (values.some((value) => typeof value !== 'number' || !Number.isFinite(value))) {
      throw new Error(`${REPORT_PATH} ${path} must contain only finite numbers`);
    }
  }
  validateHighSnrKnownSeedCoverage(
    report,
    failures,
    expectedRollingScenarioIds,
    nuisanceSeeds,
    rbwDivisors,
  );
  expectEqual(failures, nuisanceSeeds.length, ble24Total, 'validation and BLE seed counts');
  expectEqual(
    failures,
    knownAcquisitionValidationCases,
    1,
    'published one-timeslot GSM acquisition-only case count',
  );

  const rollingCases = numberAt(report, 'productionRollingWindowValidation.cases', { integer: true });
  const rollingUniqueCases = numberAt(report, 'productionRollingWindowValidation.uniqueCases', { integer: true });
  if (rollingCases <= 0) failures.push('rolling-window case count must be positive');
  expectEqual(failures, rollingUniqueCases, rollingCases, 'rolling-window unique case count');
  const missingRollingScenarios = arrayAt(report, 'productionRollingWindowValidation.missingScenarios');
  expectEqual(failures, missingRollingScenarios.length, 0, 'rolling-window missing fitted known scenarios');
  const rollingKnownCoverage = numberAt(report, 'productionRollingWindowValidation.knownCoverage');
  const rollingHierarchicalAccuracy = numberAt(report, 'productionRollingWindowValidation.hierarchicalAccuracy');
  const rollingIncompatibleNonUnknownCount = numberAt(
    report,
    'productionRollingWindowValidation.incompatibleNonUnknownCount',
    { integer: true },
  );
  expectEqual(failures, rollingIncompatibleNonUnknownCount, 0, 'rolling-window incompatible non-unknown count');
  const rollingMinimumScenarioKnownCoverage = numberAt(
    report,
    'productionRollingWindowValidation.minimumScenarioKnownCoverage',
  );
  const rollingMinimumScenarioHierarchicalAccuracy = numberAt(
    report,
    'productionRollingWindowValidation.minimumScenarioHierarchicalAccuracy',
  );
  expectDeepEqual(
    failures,
    objectAt(report, 'productionRollingWindowValidation.acceptanceThresholds'),
    PINNED_ROLLING_ACCEPTANCE_THRESHOLDS,
    'rolling-window pinned acceptance thresholds',
  );
  const rollingByScenario = valueAt(report, 'productionRollingWindowValidation.byScenario');
  if (rollingByScenario === null || typeof rollingByScenario !== 'object' || Array.isArray(rollingByScenario)) {
    throw new Error(`${REPORT_PATH} productionRollingWindowValidation.byScenario must be an object`);
  }
  expectDeepEqual(
    failures,
    Object.keys(rollingByScenario).sort(),
    [...expectedRollingScenarioIds].sort(),
    'rolling-window fitted known scenario IDs',
  );
  let rollingScenarioCaseTotal = 0;
  let rollingScenarioKnownCoveredTotal = 0;
  let rollingScenarioHierarchicallyCorrectTotal = 0;
  const rollingScenarioCoverages = [];
  const rollingScenarioAccuracies = [];
  for (const [scenarioId, value] of Object.entries(rollingByScenario)) {
    if (value === null || typeof value !== 'object' || Array.isArray(value)) {
      throw new Error(`${REPORT_PATH} rolling scenario ${scenarioId} must be an object`);
    }
    const cases = value.cases;
    if (!Number.isInteger(cases) || cases <= 0) failures.push(`rolling scenario ${scenarioId} cases must be a positive integer`);
    else rollingScenarioCaseTotal += cases;
    if (typeof value.knownCoverage !== 'number' || !Number.isFinite(value.knownCoverage)) {
      failures.push(`rolling scenario ${scenarioId} known coverage must be finite`);
    } else {
      expectRange(failures, value.knownCoverage, 0, 1, `rolling scenario ${scenarioId} known coverage`);
      rollingScenarioCoverages.push(value.knownCoverage);
      if (Number.isInteger(cases) && cases > 0) rollingScenarioKnownCoveredTotal += cases * value.knownCoverage;
    }
    if (typeof value.hierarchicalAccuracy !== 'number' || !Number.isFinite(value.hierarchicalAccuracy)) {
      failures.push(`rolling scenario ${scenarioId} hierarchical accuracy must be finite`);
    } else {
      expectRange(failures, value.hierarchicalAccuracy, 0, 1, `rolling scenario ${scenarioId} hierarchical accuracy`);
      rollingScenarioAccuracies.push(value.hierarchicalAccuracy);
      if (Number.isInteger(cases) && cases > 0) rollingScenarioHierarchicallyCorrectTotal += cases * value.hierarchicalAccuracy;
    }
  }
  expectEqual(failures, rollingScenarioCaseTotal, rollingCases, 'rolling-window by-scenario case total');
  if (rollingScenarioCaseTotal > 0) {
    expectNear(failures, rollingScenarioKnownCoveredTotal / rollingScenarioCaseTotal, rollingKnownCoverage, 1e-12, 'rolling-window aggregate known coverage');
    expectNear(failures, rollingScenarioHierarchicallyCorrectTotal / rollingScenarioCaseTotal, rollingHierarchicalAccuracy, 1e-12, 'rolling-window aggregate hierarchical accuracy');
  }
  if (rollingScenarioCoverages.length > 0) {
    expectEqual(failures, Math.min(...rollingScenarioCoverages), rollingMinimumScenarioKnownCoverage, 'rolling-window minimum scenario known coverage');
  }
  if (rollingScenarioAccuracies.length > 0) {
    expectEqual(failures, Math.min(...rollingScenarioAccuracies), rollingMinimumScenarioHierarchicalAccuracy, 'rolling-window minimum scenario hierarchical accuracy');
  }
  for (const [value, label] of [
    [rollingKnownCoverage, 'rolling known coverage'],
    [rollingHierarchicalAccuracy, 'rolling hierarchical accuracy'],
    [rollingMinimumScenarioKnownCoverage, 'rolling minimum scenario known coverage'],
    [rollingMinimumScenarioHierarchicalAccuracy, 'rolling minimum scenario hierarchical accuracy'],
  ]) expectRange(failures, value, 0, 1, label);
  for (const [metric, thresholdPath, label] of [
    [rollingKnownCoverage, 'productionRollingWindowValidation.acceptanceThresholds.overallKnownCoverage', 'rolling known coverage'],
    [rollingHierarchicalAccuracy, 'productionRollingWindowValidation.acceptanceThresholds.overallHierarchicalAccuracy', 'rolling hierarchical accuracy'],
    [rollingMinimumScenarioKnownCoverage, 'productionRollingWindowValidation.acceptanceThresholds.perScenarioKnownCoverage', 'minimum rolling per-scenario known coverage'],
    [rollingMinimumScenarioHierarchicalAccuracy, 'productionRollingWindowValidation.acceptanceThresholds.perScenarioHierarchicalAccuracy', 'minimum rolling per-scenario hierarchical accuracy'],
  ]) {
    const threshold = numberAt(report, thresholdPath);
    if (metric < threshold) failures.push(`${label}: ${metric} is below publication threshold ${threshold}`);
  }

  expectEqual(failures, booleanAt(report, 'matrix.tailCalibrationAudit.independentRecomputation.valid'), true, 'independent tail-calibration recomputation');
  expectEqual(failures, booleanAt(report, 'matrix.tailCalibrationAudit.independentRecomputation.aggregationRegression.passed'), true, 'all-online attempt-min regression');
  for (const path of [
    'matrix.tailCalibrationAudit.missingScenarioIds',
    'matrix.tailCalibrationAudit.unexpectedScenarioIds',
    'matrix.tailCalibrationAudit.invalidAttemptCounts',
    'matrix.tailCalibrationAudit.viewCountMismatches',
    'matrix.tailCalibrationAudit.independentRecomputation.attemptCountMismatches',
  ]) {
    expectEqual(failures, arrayAt(report, path).length, 0, `${path} length`);
  }
  const tailCalibrationLateMinimumCount = numberAt(
    report,
    'matrix.tailCalibrationAudit.independentRecomputation.lateMinimumCount',
    { integer: true },
  );
  if (tailCalibrationLateMinimumCount < 1) failures.push('independent tail-calibration audit must observe at least one later-online attempt minimum');
  const tailCalibrationAttemptCount = numberAt(
    report,
    'matrix.tailCalibrationAudit.independentRecomputation.allOnlineAttemptCount',
    { integer: true },
  );
  if (tailCalibrationAttemptCount <= 0 || tailCalibrationLateMinimumCount > tailCalibrationAttemptCount) {
    failures.push(`tail-calibration counts must satisfy 0 <= late minima <= positive all-online attempts; observed ${tailCalibrationLateMinimumCount}/${tailCalibrationAttemptCount}`);
  }
  const tailScoreComparisons = arrayAt(report, 'matrix.tailCalibrationAudit.independentRecomputation.scoreComparisons');
  const expectedTailComparisonKeys = PINNED_KNOWN_CLASS_IDS.flatMap((classId) =>
    PINNED_TAIL_VIEWS.map((view) => `${classId}/${view}`)).sort();
  const observedTailComparisonKeys = tailScoreComparisons.map((comparison, index) => {
    if (comparison === null || typeof comparison !== 'object'
      || typeof comparison.classId !== 'string' || typeof comparison.view !== 'string') {
      throw new Error(`${REPORT_PATH} tail score comparison ${index} must publish classId and view`);
    }
    return `${comparison.classId}/${comparison.view}`;
  }).sort();
  expectDeepEqual(
    failures,
    observedTailComparisonKeys,
    expectedTailComparisonKeys,
    'independent tail-calibration comparison key set',
  );
  const tailScoreTolerance = numberAt(report, 'matrix.tailCalibrationAudit.independentRecomputation.scoreTolerance');
  expectEqual(failures, tailScoreTolerance, PINNED_TAIL_SCORE_TOLERANCE, 'independent tail-calibration score tolerance');
  for (const [index, comparison] of tailScoreComparisons.entries()) {
    if (comparison === null || typeof comparison !== 'object') throw new Error(`${REPORT_PATH} tail score comparison ${index} must be an object`);
    const censoredBluetoothEnvelope = comparison.classId === 'bluetooth-like'
      && comparison.view !== 'spectrum-only';
    if (!Number.isInteger(comparison.expectedCount) || comparison.expectedCount < 0
      || !Number.isInteger(comparison.observedCount) || comparison.observedCount < 0
      || (!censoredBluetoothEnvelope
        && (comparison.expectedCount === 0 || comparison.observedCount === 0))) {
      failures.push(`tail score comparison ${index} counts must be supported positive integers or an explicit censored zero`);
    }
    expectEqual(failures, comparison.expectedCount, comparison.observedCount, `tail score comparison ${index} count`);
    if (censoredBluetoothEnvelope) {
      expectEqual(failures, comparison.expectedCount, 0, `tail score comparison ${index} censored expected count`);
      expectEqual(failures, comparison.observedCount, 0, `tail score comparison ${index} censored observed count`);
      expectEqual(failures, comparison.maximumAbsoluteDifference, 0, `tail score comparison ${index} censored difference`);
      expectEqual(failures, comparison.expectedSha256, PINNED_EMPTY_ARRAY_SHA256, `tail score comparison ${index} censored expected hash`);
      expectEqual(failures, comparison.observedSha256, PINNED_EMPTY_ARRAY_SHA256, `tail score comparison ${index} censored observed hash`);
    }
    if (typeof comparison.maximumAbsoluteDifference !== 'number'
      || !Number.isFinite(comparison.maximumAbsoluteDifference)
      || comparison.maximumAbsoluteDifference < 0
      || comparison.maximumAbsoluteDifference > tailScoreTolerance) {
      failures.push(`tail score comparison ${index} exceeds ${tailScoreTolerance}`);
    }
    if (typeof comparison.expectedSha256 !== 'string' || !/^[a-f0-9]{64}$/.test(comparison.expectedSha256)
      || typeof comparison.observedSha256 !== 'string' || !/^[a-f0-9]{64}$/.test(comparison.observedSha256)) {
      failures.push(`tail score comparison ${index} hashes must be lowercase SHA-256`);
    }
    expectEqual(failures, comparison.expectedSha256, comparison.observedSha256, `tail score comparison ${index} hash`);
  }

  const causalEnvelopePriorSensitivity = validatePriorSensitivityPopulation(
    report,
    'priorSensitivity',
    'capture-qualified-selected-view',
    representatives,
    failures,
  );
  const completeOnlineSpectrumPriorSensitivity = validatePriorSensitivityPopulation(
    report,
    'priorSensitivity.completeOnlineSpectrum',
    'complete-online-spectrum',
    completeOnlineSpectrumCases,
    failures,
    {
      knownCases: completeOnlineSpectrumCases - completeOnlineUnknownTruthCases,
      unknownCases: completeOnlineUnknownTruthCases,
    },
  );
  const priorVariants = causalEnvelopePriorSensitivity.variants;
  const priorKnownCoverages = causalEnvelopePriorSensitivity.knownCoverages;
  const priorHierarchicalAccuracies = causalEnvelopePriorSensitivity.hierarchicalAccuracies;
  const priorIncompatibleRisks = causalEnvelopePriorSensitivity.incompatibleRisks;
  const priorFalseAcceptedUnknownRisks = causalEnvelopePriorSensitivity.falseAcceptedUnknownRisks;
  const priorDecisionChangeRates = causalEnvelopePriorSensitivity.decisionChangeRates;
  expectDeepEqual(
    failures,
    objectAt(report, 'priorSensitivity.completeOnlineSpectrum').gates,
    objectAt(report, 'priorSensitivity').gates,
    'causal-envelope/complete-online-spectrum prior-sensitivity gate identity',
  );
  expectEqual(
    failures,
    completeOnlineSpectrumPriorSensitivity.variants.length,
    priorVariants.length,
    'causal-envelope/complete-online-spectrum prior-sensitivity variant count',
  );

  const attempts = numberAt(report, 'admission.attempted', { integer: true });
  const admitted = numberAt(report, 'admission.admitted', { integer: true });
  const admissionRate = numberAt(report, 'admission.admissionRate');
  expectDeepEqual(
    failures,
    arrayAt(report, 'admission.expectedClassificationNonAdmissionScenarios'),
    PINNED_EXPECTED_CLASSIFICATION_NON_ADMISSION_SCENARIO_IDS,
    'expected classification non-admission scenario policy',
  );
  expectEqual(
    failures,
    arrayAt(report, 'admission.expectedNonAdmissionScenariosWithAdmission').length,
    0,
    'unexpected admission under the classification non-admission policy',
  );
  expectEqual(
    failures,
    numberAt(report, 'admission.knownAcquisitionWrongAdmissionCount', {
      integer: true,
    }),
    0,
    'known acquisition-validation incompatible admission count',
  );
  expectEqual(
    failures,
    arrayAt(report, 'admission.knownAcquisitionWrongAdmissionExamples').length,
    0,
    'known acquisition-validation incompatible admission examples',
  );
  if (attempts <= 0 || admitted < 0 || admitted > attempts) {
    failures.push(`admission counts must satisfy 0 <= admitted <= attempted with a positive denominator; observed ${admitted}/${attempts}`);
  }
  expectRange(failures, admissionRate, 0, 1, 'admission rate');
  if (attempts > 0) expectEqual(failures, admissionRate, admitted / attempts, 'admission rate recomputation');

  return {
    attempts,
    admitted,
    admissionRate,
    representatives,
    qualifiedEnvelopeRepresentatives,
    censoredSpectrumRepresentatives,
    properScoreSamples,
    hierarchicalAccuracy: numberAt(report, 'classificationConditionalOnAdmission.hierarchicalAccuracy'),
    knownTopLeafAccuracy: numberAt(report, 'classificationConditionalOnAdmission.knownTopLeafAccuracy'),
    knownCoverage: numberAt(report, 'classificationConditionalOnAdmission.knownCoverage'),
    coveredKnownHierarchicalAccuracy: numberAt(
      report,
      'classificationConditionalOnAdmission.coveredKnownHierarchicalAccuracy',
    ),
    minimumHighSnrKnownClassHierarchicalAccuracy: numberAt(
      report,
      'classificationConditionalOnAdmission.minimumHighSnrKnownClassHierarchicalAccuracy',
    ),
    fittedUnknownTemplateRejectionRate: numberAt(
      report,
      'classificationConditionalOnAdmission.fittedUnknownTemplateRejectionRate',
    ),
    fittedUnknownPosteriorAuroc: numberAt(
      report,
      'classificationConditionalOnAdmission.fittedUnknownPosteriorAuroc',
    ),
    strictUnknownRejectionRate: numberAt(
      report,
      'classificationConditionalOnAdmission.scenarioExcludedStrictUnknownRejectionRate',
    ),
    strictTypicalityAuroc: numberAt(
      report,
      'classificationConditionalOnAdmission.scenarioExcludedStrictTypicalityAuroc',
    ),
    exactEquivalenceCompatibleRate: numberAt(
      report,
      'classificationConditionalOnAdmission.exactEquivalenceCompatibleRate',
    ),
    fittedTemplateLogLoss: numberAt(report, 'classificationConditionalOnAdmission.fittedTemplateLogLoss'),
    fittedTemplateMulticlassBrier: numberAt(
      report,
      'classificationConditionalOnAdmission.fittedTemplateMulticlassBrier',
    ),
    fittedTemplateExpectedCalibrationError: numberAt(
      report,
      'classificationConditionalOnAdmission.fittedTemplateExpectedCalibrationError',
    ),
    falseAcceptedUnknownCount: numberAt(
      report,
      'classificationConditionalOnAdmission.falseAcceptedUnknownCount',
      { integer: true },
    ),
    falseAcceptAttemptCount: numberAt(
      report,
      'classificationConditionalOnAdmission.anyFalseAcceptAttemptCount',
      { integer: true },
    ),
    tolerance: numberAt(report, 'corpus.exactObservableEquivalencePairAudit.numericalTolerance'),
    nuisanceCells,
    representativePairs,
    evidenceViewPairs,
    onlineSpectrumPairs,
    discrepancies: reportedDiscrepancies,
    exactEquivalencePairs: pairs.length,
    nuisanceSeeds,
    snrDb,
    rbwDivisors,
    classificationAdmissions: numberAt(report, 'matrix.classificationAdmissions', { integer: true }),
    standardObservationHorizon: numberAt(
      report,
      'matrix.observationOpportunityHorizons.standard',
      { integer: true },
    ),
    fullBandObservationHorizon: numberAt(
      report,
      'matrix.observationOpportunityHorizons.fullBand2g4',
      { integer: true },
    ),
    strictUnknownHoldouts,
    ambiguityStressCases,
    knownAcquisitionValidationCases,
    ble24Covered,
    ble24Total,
    ble32Covered,
    ble32Total,
    bleRepresentatives,
    rollingCases,
    completeOnlineSpectrumCases,
    completeOnlineUnknownTruthCases,
    completeOnlineProperScoreSamples,
    completeOnlineLogLoss,
    completeOnlineBrier,
    completeOnlineEce,
    rollingKnownCoverage,
    rollingHierarchicalAccuracy,
    rollingIncompatibleNonUnknownCount,
    rollingMinimumScenarioKnownCoverage,
    rollingMinimumScenarioHierarchicalAccuracy,
    tailCalibrationAttemptCount,
    tailCalibrationLateMinimumCount,
    tailScoreComparisons: tailScoreComparisons.length,
    priorVariantCount: priorVariants.length,
    minimumPriorKnownCoverage: Math.min(...priorKnownCoverages),
    maximumPriorKnownCoverage: Math.max(...priorKnownCoverages),
    minimumPriorHierarchicalAccuracy: Math.min(...priorHierarchicalAccuracies),
    maximumPriorHierarchicalAccuracy: Math.max(...priorHierarchicalAccuracies),
    maximumPriorIncompatibleRisk: Math.max(...priorIncompatibleRisks),
    maximumPriorFalseAcceptedUnknownRisk: Math.max(...priorFalseAcceptedUnknownRisks),
    maximumPriorDecisionChangeRate: Math.max(...priorDecisionChangeRates),
  };
}

function formatMetrics(metrics, failures) {
  for (const key of [
    'admissionRate',
    'hierarchicalAccuracy',
    'knownTopLeafAccuracy',
    'knownCoverage',
    'coveredKnownHierarchicalAccuracy',
    'minimumHighSnrKnownClassHierarchicalAccuracy',
    'fittedUnknownTemplateRejectionRate',
    'fittedUnknownPosteriorAuroc',
    'strictUnknownRejectionRate',
    'strictTypicalityAuroc',
    'exactEquivalenceCompatibleRate',
    'fittedTemplateExpectedCalibrationError',
    'completeOnlineEce',
  ]) expectRange(failures, metrics[key], 0, 1, key);
  expectRange(failures, metrics.fittedTemplateMulticlassBrier, 0, 2, 'fittedTemplateMulticlassBrier');
  expectRange(failures, metrics.completeOnlineBrier, 0, 2, 'completeOnlineBrier');
  for (const [key, minimum, label] of [
    [
      'hierarchicalAccuracy',
      PINNED_ACCEPTANCE_THRESHOLDS.hierarchicalAccuracy,
      'admission-conditional hierarchical accuracy',
    ],
    [
      'knownTopLeafAccuracy',
      PINNED_ACCEPTANCE_THRESHOLDS.knownTopLeafAccuracy,
      'admission-conditional known top-leaf accuracy',
    ],
    [
      'knownCoverage',
      PINNED_ACCEPTANCE_THRESHOLDS.knownCoverage,
      'admission-conditional known coverage',
    ],
    [
      'minimumHighSnrKnownClassHierarchicalAccuracy',
      PINNED_ACCEPTANCE_THRESHOLDS.minimumHighSnrKnownClassHierarchicalAccuracy,
      'minimum high-SNR known-class hierarchical accuracy',
    ],
    [
      'fittedUnknownPosteriorAuroc',
      PINNED_ACCEPTANCE_THRESHOLDS.fittedUnknownPosteriorAuroc,
      'fitted-unknown posterior AUROC',
    ],
    [
      'strictTypicalityAuroc',
      PINNED_ACCEPTANCE_THRESHOLDS.strictTypicalityAuroc,
      'strict scenario-excluded typicality AUROC',
    ],
  ]) {
    expectRange(failures, metrics[key], minimum, 1, label);
  }
  expectEqual(
    failures,
    metrics.strictUnknownRejectionRate,
    PINNED_ACCEPTANCE_THRESHOLDS.strictUnknownRejectionRate,
    'strict unknown holdout rejection acceptance gate',
  );
  expectEqual(
    failures,
    metrics.exactEquivalenceCompatibleRate,
    PINNED_ACCEPTANCE_THRESHOLDS.exactEquivalenceCompatibleRate,
    'exact observable-equivalence compatibility acceptance gate',
  );
  if (metrics.fittedTemplateLogLoss < 0) failures.push('fitted-template log loss must be non-negative');
  if (metrics.completeOnlineLogLoss < 0) failures.push('complete-online log loss must be non-negative');
  if (metrics.fittedTemplateLogLoss
    > PINNED_ACCEPTANCE_THRESHOLDS.fittedTemplateLogLoss) {
    failures.push(`fitted-template log loss exceeds the acceptance gate ${PINNED_ACCEPTANCE_THRESHOLDS.fittedTemplateLogLoss}`);
  }
  if (metrics.fittedTemplateMulticlassBrier
    > PINNED_ACCEPTANCE_THRESHOLDS.fittedTemplateMulticlassBrier) {
    failures.push(`fitted-template multiclass Brier score exceeds the acceptance gate ${PINNED_ACCEPTANCE_THRESHOLDS.fittedTemplateMulticlassBrier}`);
  }
  if (metrics.fittedTemplateExpectedCalibrationError
    > PINNED_ACCEPTANCE_THRESHOLDS.fittedTemplateExpectedCalibrationError) {
    failures.push(`fitted-template expected calibration error exceeds the acceptance gate ${PINNED_ACCEPTANCE_THRESHOLDS.fittedTemplateExpectedCalibrationError}`);
  }
  if (metrics.tolerance <= 0) failures.push('exact-equivalence numerical tolerance must be positive');
  for (const key of [
    'attempts', 'admitted', 'representatives', 'qualifiedEnvelopeRepresentatives',
    'censoredSpectrumRepresentatives', 'properScoreSamples', 'falseAcceptedUnknownCount',
    'falseAcceptAttemptCount', 'nuisanceCells', 'representativePairs', 'evidenceViewPairs',
    'onlineSpectrumPairs',
    'discrepancies', 'exactEquivalencePairs', 'ble24Covered', 'ble24Total', 'ble32Covered',
    'ble32Total', 'bleRepresentatives', 'rollingCases', 'rollingIncompatibleNonUnknownCount',
    'tailCalibrationAttemptCount', 'tailCalibrationLateMinimumCount', 'tailScoreComparisons',
    'priorVariantCount', 'completeOnlineSpectrumCases', 'completeOnlineUnknownTruthCases',
    'completeOnlineProperScoreSamples', 'fittingRepresentativeCount',
    'fittingSpectrumRepresentativeCount', 'fittingUntimedEnvelopeRepresentativeCount',
    'fittingTimedEnvelopeRepresentativeCount', 'fittingPhysicalCaptureCount',
    'fittingUnavailableWindowCount', 'fittingPostCaptureUnavailableWindowCount',
    'tailCalibrationPhysicalCaptureCount', 'tailCalibrationUnavailableWindowCount',
    'tailCalibrationPostCaptureUnavailableWindowCount',
  ]) {
    if (!Number.isInteger(metrics[key]) || metrics[key] < 0) failures.push(`${key} must be a non-negative integer`);
  }
  for (const key of [
    'attempts', 'representatives', 'qualifiedEnvelopeRepresentatives',
    'censoredSpectrumRepresentatives', 'properScoreSamples', 'nuisanceCells', 'exactEquivalencePairs',
    'onlineSpectrumPairs', 'completeOnlineSpectrumCases', 'completeOnlineUnknownTruthCases',
    'completeOnlineProperScoreSamples',
    'fittingRepresentativeCount', 'fittingSpectrumRepresentativeCount',
    'fittingUntimedEnvelopeRepresentativeCount', 'fittingTimedEnvelopeRepresentativeCount',
    'fittingPhysicalCaptureCount', 'tailCalibrationPhysicalCaptureCount',
  ]) {
    if (metrics[key] <= 0) failures.push(`${key} must be positive`);
  }

  const formatted = {
    attempts: formatInteger(metrics.attempts),
    admitted: formatInteger(metrics.admitted),
    admissionRate: formatFixed(metrics.admissionRate, 6),
    representatives: formatInteger(metrics.representatives),
    qualifiedEnvelopeRepresentatives:
      formatInteger(metrics.qualifiedEnvelopeRepresentatives),
    censoredSpectrumRepresentatives:
      formatInteger(metrics.censoredSpectrumRepresentatives),
    properScoreSamples: formatInteger(metrics.properScoreSamples),
    hierarchicalAccuracy: formatFixed(metrics.hierarchicalAccuracy, 6),
    knownTopLeafAccuracy: formatFixed(metrics.knownTopLeafAccuracy, 6),
    knownCoverage: formatFixed(metrics.knownCoverage, 6),
    coveredKnownHierarchicalAccuracy: formatFixed(metrics.coveredKnownHierarchicalAccuracy, 6),
    minimumHighSnrKnownClassHierarchicalAccuracy: formatFixed(
      metrics.minimumHighSnrKnownClassHierarchicalAccuracy,
      4,
    ),
    fittedUnknownTemplateRejectionRate: formatFixed(metrics.fittedUnknownTemplateRejectionRate, 6),
    fittedUnknownPosteriorAuroc: formatFixed(metrics.fittedUnknownPosteriorAuroc, 6),
    strictUnknownRejectionRate: formatFixed(metrics.strictUnknownRejectionRate, 6),
    strictTypicalityAuroc: formatFixed(metrics.strictTypicalityAuroc, 6),
    exactEquivalenceCompatibleRate: formatFixed(metrics.exactEquivalenceCompatibleRate, 6),
    fittedTemplateLogLoss: formatFixed(metrics.fittedTemplateLogLoss, 7),
    fittedTemplateMulticlassBrier: formatFixed(metrics.fittedTemplateMulticlassBrier, 8),
    fittedTemplateExpectedCalibrationError: formatFixed(metrics.fittedTemplateExpectedCalibrationError, 8),
    tolerance: formatScientific(metrics.tolerance),
    nuisanceCells: formatInteger(metrics.nuisanceCells),
    representativePairs: formatInteger(metrics.representativePairs),
    evidenceViewPairs: formatInteger(metrics.evidenceViewPairs),
    onlineSpectrumPairs: formatInteger(metrics.onlineSpectrumPairs),
    bleRepresentatives: formatInteger(metrics.bleRepresentatives),
    rollingCases: formatInteger(metrics.rollingCases),
    completeOnlineSpectrumCases: formatInteger(metrics.completeOnlineSpectrumCases),
    completeOnlineUnknownTruthCases: formatInteger(metrics.completeOnlineUnknownTruthCases),
    completeOnlineProperScoreSamples: formatInteger(metrics.completeOnlineProperScoreSamples),
    completeOnlineLogLoss: formatFixed(metrics.completeOnlineLogLoss, 7),
    completeOnlineBrier: formatFixed(metrics.completeOnlineBrier, 8),
    completeOnlineEce: formatFixed(metrics.completeOnlineEce, 8),
    rollingKnownCoverage: formatFixed(metrics.rollingKnownCoverage, 6),
    rollingHierarchicalAccuracy: formatFixed(metrics.rollingHierarchicalAccuracy, 6),
    rollingMinimumScenarioKnownCoverage: formatFixed(metrics.rollingMinimumScenarioKnownCoverage, 6),
    rollingMinimumScenarioHierarchicalAccuracy: formatFixed(
      metrics.rollingMinimumScenarioHierarchicalAccuracy,
      6,
    ),
    tailCalibrationAttemptCount: formatInteger(metrics.tailCalibrationAttemptCount),
    tailCalibrationLateMinimumCount: formatInteger(metrics.tailCalibrationLateMinimumCount),
    fittingRepresentativeCount: formatInteger(metrics.fittingRepresentativeCount),
    fittingSpectrumRepresentativeCount: formatInteger(metrics.fittingSpectrumRepresentativeCount),
    fittingUntimedEnvelopeRepresentativeCount: formatInteger(
      metrics.fittingUntimedEnvelopeRepresentativeCount,
    ),
    fittingTimedEnvelopeRepresentativeCount: formatInteger(metrics.fittingTimedEnvelopeRepresentativeCount),
    fittingPhysicalCaptureCount: formatInteger(metrics.fittingPhysicalCaptureCount),
    fittingUnavailableWindowCount: formatInteger(metrics.fittingUnavailableWindowCount),
    fittingPostCaptureUnavailableWindowCount: formatInteger(
      metrics.fittingPostCaptureUnavailableWindowCount,
    ),
    tailCalibrationPhysicalCaptureCount: formatInteger(metrics.tailCalibrationPhysicalCaptureCount),
    tailCalibrationUnavailableWindowCount: formatInteger(
      metrics.tailCalibrationUnavailableWindowCount,
    ),
    tailCalibrationPostCaptureUnavailableWindowCount: formatInteger(
      metrics.tailCalibrationPostCaptureUnavailableWindowCount,
    ),
    priorVariantCount: formatInteger(metrics.priorVariantCount),
    minimumPriorKnownCoverage: formatFixed(metrics.minimumPriorKnownCoverage, 6),
    maximumPriorKnownCoverage: formatFixed(metrics.maximumPriorKnownCoverage, 6),
    minimumPriorHierarchicalAccuracy: formatFixed(metrics.minimumPriorHierarchicalAccuracy, 6),
    maximumPriorHierarchicalAccuracy: formatFixed(metrics.maximumPriorHierarchicalAccuracy, 6),
    maximumPriorIncompatibleRisk: formatFixed(metrics.maximumPriorIncompatibleRisk, 6),
    maximumPriorFalseAcceptedUnknownRisk: formatFixed(metrics.maximumPriorFalseAcceptedUnknownRisk, 6),
    maximumPriorDecisionChangeRate: formatFixed(metrics.maximumPriorDecisionChangeRate, 6),
  };

  expectEqual(
    failures,
    metrics.fittedUnknownTemplateRejectionRate,
    metrics.strictUnknownRejectionRate,
    'README/UI combined fitted-unknown and strict-holdout rejection publication',
  );
  expectEqual(
    failures,
    metrics.fittedUnknownPosteriorAuroc,
    metrics.fittedUnknownTemplateRejectionRate,
    'normative-doc combined fitted-unknown AUROC and rejection publication',
  );
  expectEqual(failures, metrics.falseAcceptedUnknownCount, 0, 'published unknown false-accept count');
  expectEqual(failures, metrics.falseAcceptAttemptCount, 0, 'published disallowed false-accept attempt count');
  expectEqual(failures, metrics.discrepancies, 0, 'published exact-equivalence discrepancy count');
  return formatted;
}

function verifyPublicationProse(documents, modelSha256, corpusSha256, metrics, formatted, failures) {
  for (const path of PUBLICATION_PATHS) {
    const text = documents.get(path);
    for (const [staleValue, label] of STALE_PUBLICATION_VALUES) {
      if (text.includes(staleValue)) failures.push(`${path} contains ${label}: ${staleValue}`);
    }
    for (const { pattern, label } of STALE_OBSERVATION_HORIZON_PATTERNS) {
      const matches = text.match(pattern) ?? [];
      if (matches.length > 0) {
        failures.push(`${path} contains ${label}: ${matches.join(', ')}`);
      }
    }
    for (const [id, pattern, label] of [
      [PINNED_MODEL_ID, /bayesian-observable-equivalence-v\d+/g, 'model ID'],
      [PINNED_PREPROCESSING_ID, /scalar-observable-features-v\d+/g, 'preprocessing ID'],
      [PINNED_PRIOR_ID, /engineering-design-class-weights-v\d+/g, 'prior ID'],
      [PINNED_CALIBRATION_ID, /synthetic-(?:independent-branch-)?view-matched-(?:stratified-online-attempt-min|causal-acquisition)-support-rank-detector-conditioned-physical-uncalibrated-v\d+/g, 'calibration ID'],
      [PINNED_DECISION_POLICY_ID, /observable-open-set-decision-v\d+/g, 'decision-policy ID'],
    ]) {
      const matches = [...text.matchAll(pattern)].map((match) => match[0]);
      if (!matches.includes(id)) failures.push(`${path} must publish current ${label} ${id}`);
      if (matches.some((value) => value !== id)) {
        failures.push(`${path} contains stale ${label} values: ${matches.join(', ')}`);
      }
    }
    const hashCount = occurrenceCount(text, modelSha256);
    if (hashCount !== 1) {
      failures.push(`${path} must publish model asset SHA-256 ${modelSha256} exactly once (found ${hashCount})`);
    }
    const commitCount = occurrenceCount(text, PINNED_SIGNAL_LAB_COMMIT);
    if (commitCount !== 1) {
      failures.push(`${path} must publish SignalLab source commit ${PINNED_SIGNAL_LAB_COMMIT} exactly once (found ${commitCount})`);
    }
    const corpusHashCount = occurrenceCount(text, corpusSha256);
    if (corpusHashCount !== 1) {
      failures.push(`${path} must publish corpus source-manifest SHA-256 ${corpusSha256} exactly once (found ${corpusHashCount})`);
    }
    const corpusVersionCount = occurrenceCount(text, PINNED_CORPUS_VERSION);
    if (corpusVersionCount < 1) {
      failures.push(`${path} must publish corpus version ${PINNED_CORPUS_VERSION}`);
    }
    const publishedCorpusVersions = [...text.matchAll(/observable-scalar-corpus-v\d+/g)]
      .map((match) => match[0]);
    if (publishedCorpusVersions.some((version) => version !== PINNED_CORPUS_VERSION)) {
      failures.push(`${path} contains stale corpus versions: ${publishedCorpusVersions.join(', ')}`);
    }
  }

  const seedList = formatOxford(metrics.nuisanceSeeds.map(String));
  const codeSeedList = formatOxford(metrics.nuisanceSeeds.map((seed) => `\`${seed}\``));
  const snrList = metrics.snrDb.join('/');
  const rbwList = metrics.rbwDivisors.join('/');
  const seedCountWord = numberWord(metrics.nuisanceSeeds.length);
  const rbwCountWord = numberWord(metrics.rbwDivisors.length);
  const pairCountWord = numberWord(metrics.exactEquivalencePairs);
  const strictHoldoutCountWord = numberWord(metrics.strictUnknownHoldouts);
  const ambiguityCountWord = numberWord(metrics.ambiguityStressCases);
  const classificationAdmissionCountWord = numberWord(metrics.classificationAdmissions);
  const bleCoverage = `${metrics.ble24Covered}/${metrics.ble24Total}`;
  const ble32Coverage = `${metrics.ble32Covered}/${metrics.ble32Total}`;
  const spectrumReleaseGateStartList = formatOxford(
    PINNED_SPECTRUM_RELEASE_GATE_SOURCE_PLAN
      .map((profile) => formatInteger(profile.sourceLookIndexOffset)),
  );
  const envelopeReleaseGateStartList = formatOxford(
    PINNED_QUALIFIED_ENVELOPE_RELEASE_GATE_SOURCE_PLAN
      .map((profile) => formatInteger(profile.sourceLookIndexOffset)),
  );
  const completeOnlineSpectrumSummary = `The primary complete-online spectrum audit classified ${formatted.completeOnlineSpectrumCases} unique current-qualified attempt/opportunity/representative windows across every truth and SNR. It included ${formatted.completeOnlineUnknownTruthCases} unknown-truth windows, with zero unknown false accepts and zero incompatible non-unknown decisions; on ${formatted.completeOnlineProperScoreSamples} singleton-truth fit-domain samples, log loss was ${formatted.completeOnlineLogLoss}, multiclass Brier score ${formatted.completeOnlineBrier}, and expected calibration error ${formatted.completeOnlineEce}.`;
  const rollingSummary = `The secondary high-SNR known-scenario spectrum gate classified ${formatted.rollingCases} unique current-qualified attempt/opportunity/representative windows across every fitted known scenario without a truth-conditioned filter: known coverage was ${formatted.rollingKnownCoverage}, hierarchical accuracy ${formatted.rollingHierarchicalAccuracy}, minimum per-scenario known coverage ${formatted.rollingMinimumScenarioKnownCoverage}, minimum per-scenario hierarchical accuracy ${formatted.rollingMinimumScenarioHierarchicalAccuracy}, and incompatible non-unknown decisions zero.`;
  const priorSummary = `The deterministic engineering-prior sensitivity audit evaluated ${formatted.priorVariantCount} declared baseline, unknown-mass, and family-mass variants on the causal-envelope population: known coverage ranged ${formatted.minimumPriorKnownCoverage}-${formatted.maximumPriorKnownCoverage}, hierarchical accuracy ${formatted.minimumPriorHierarchicalAccuracy}-${formatted.maximumPriorHierarchicalAccuracy}, maximum incompatible-non-unknown risk was ${formatted.maximumPriorIncompatibleRisk}, maximum false-accepted-unknown risk ${formatted.maximumPriorFalseAcceptedUnknownRisk}, and maximum decision-change rate ${formatted.maximumPriorDecisionChangeRate}. The same variants also passed the complete-online spectrum population audit over ${formatted.completeOnlineSpectrumCases} samples. These priors are engineering assumptions, not field-prevalence calibration; operational prevalence remains an unmeasured physical-validation limitation.`;
  const tailSummary = `The validator independently regenerated ${formatted.tailCalibrationAttemptCount} independent-branch spectrum-attempt minima and matched all ${metrics.tailScoreComparisons} class/view score arrays to the checked-in asset within the declared tolerance; ${formatted.tailCalibrationLateMinimumCount} spectrum-attempt minima occurred after the first-ready opportunity, proving that later online representatives affect the stored spectrum minimum. The envelope views use only the sole physical detected-power capture from each separate qualified-envelope attempt, never a later counterfactual envelope.`;
  const productionAcquisitionSummary = `The fitted and independently regenerated acquisition matrix uses SignalLab's 450-point recommended-span grid in two independent production-gate sessions under ${PINNED_ACQUISITION_BRANCH_POLICY}. The no-automatic-capture consecutive-spectrum branch starts its twelve profiles at source looks ${spectrumReleaseGateStartList} and spans source indices [0, 512); the qualified-envelope branch starts them at source looks ${envelopeReleaseGateStartList} and spans [0, 524), with at most one detected-power capture. Under ${PINNED_CAPTURE_TARGET_SELECTION_POLICY}, ordinary targets are active physical rows with zero missed sweeps. The only candidate-state exception is the exact latest raw detector/track member cited by the latest exactly-one opportunity of a current, promotion-qualified, zero-miss frequency-agile association. The synthetic activity summary never owns the hardware capture, and arbitrary candidates, stale members, retained summaries, and ambiguous opportunities remain ineligible. The autonomous branch uses ${PINNED_CAPTURE_TARGET_RANK_MODEL}: for each eligible raw row it takes the current frozen source sweep, estimates the floor as the median of its lowest twenty percent of bins, and integrates positive linear power above that floor across every complete frequency cell whose center lies in the raw detected interval, normalized by actual RBW. It orders the exact unrounded integral descending, then uses the stable representative key and raw ID only as exact-integral tie-breaks. Association qualification controls only whether the narrow agile projection exists, never priority; truth labels, class-domain eligibility, feature readiness, and classifier posteriors never influence ranking. Only rank 0 may proceed to runtime admission, and an unavailable rank-0 target causes no capture rather than substitution of a lower rank. Receipt schema 4 records the complete numeric rank evidence, proves ${PINNED_DETECTED_POWER_SELECTION_CONDITION}, projects the exact eight-sweep classifier window to its evidence representative, and binds the complete returned capture with domain-separated canonical SHA-256. For an agile projection the receiver remains fixed on the selected physical channel and may observe later returns or no return; it never follows the hop and proves neither a common emitter nor Bluetooth protocol or mode identity. Under ${PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORING_POLICY.id} the valid capture and receipt remain audited, but every frequency-agile fixed-tune envelope is excluded from classifier features and the exact regional spectrum/history view is used; this observation-geometry censor is independent of truth or requested hypothesis. Later spectra continue at the next source look. Held-out validation begins at source look 512 for consecutive spectrum and 524 for qualified envelope. Every envelope admitted to a classifier likelihood requires an analysis-issued capture receipt and is explicitly qualified as ${PINNED_DETECTED_POWER_ACQUISITION_QUALIFICATION}. Every fitted and calibrated automatic envelope sample separately carries ${PINNED_DETECTED_POWER_SELECTION_CONDITION}; an operator-preferred capture remains explicitly marked as operator-selected and is never silently represented as belonging to that automatic selection population. Receipt-free, lower-ranked automatic, or runtime-unadmitted captures cannot enter Bayesian envelope metrics. Public detected-power synthesis uses the generator-internal 100 kHz filter; measured detected-power RBW remains unavailable and is never classifier evidence.`;
  const modelStructureSummary = `The checked-in v9 likelihood architecture has ${PINNED_DIMENSIONS.length} ordered feature dimensions and ${PINNED_CLASS_IDS.length} exact leaf class IDs. Its spectrum-only population has 18 source scenarios and 28 likelihood components; each envelope population has 16 scenarios and 26 components because the Bluetooth-like class is structurally unsupported for fixed-tune envelope evidence. Under ${PINNED_LIKELIHOOD_COMPONENT_DECOMPOSITION_POLICY.id}, exactly five pinned CSMA sources use three deterministic activity modes while every other supported source/view pair uses one component; source scenarios retain equal within-class mass, CSMA modes use empirical within-source weights, and each decomposed source shares one pooled within-mode covariance. Under ${PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORING_POLICY.id}, the analysis boundary validates the physical capture and schema-4 receipt first, including its canonical SHA-256 binding of all returned samples, cadence, requested geometry, RF metadata, provenance, and exact integrated-excess rank evidence, then excludes detected-power envelope features for every frequency-agile association and classifies its exact regional spectrum/history view. This censor is triggered by observed association geometry, never a truth label or requested hypothesis; Bluetooth envelope component and calibration arrays are therefore exactly empty.`;
  const viewContractSummary = 'Production inference does not use missing-dimension marginalization: v9 selects one exact evidence view, requires its complete finite feature set with no extras, and evaluates only the independently fitted spectrum-only, envelope-untimed, or envelope-timed likelihood population.';
  const manualCaptureSummary = `The App zero-span action enters a Bayesian envelope view only when the capture is bound to an analysis-issued receipt for a current runtime-admitted target, exact admitted tune, and exact eight-sweep evidence window. Receipt qualification is necessary but not sufficient: under ${PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORING_POLICY.id}, every fixed-tune frequency-agile capture remains excluded from Bayesian envelope inference and the exact spectrum view is used instead. Any other receipt-free or runtime-unadmitted capture may feed only the separate envelope heuristic.`;
  const decisionThresholdSummary = `The open-set rejection cutoff is a minimum maximum-known synthetic support rank of ${PINNED_MINIMUM_KNOWN_SYNTHETIC_SUPPORT_RANK}; it is an engineering threshold, not a p-value or coverage guarantee.`;
  for (const path of PUBLICATION_PATHS) {
    expectExactlyOnce(failures, path, documents.get(path), completeOnlineSpectrumSummary, 'complete-online-spectrum-summary');
    expectExactlyOnce(failures, path, documents.get(path), rollingSummary, 'rolling-window-summary');
    expectExactlyOnce(failures, path, documents.get(path), priorSummary, 'prior-sensitivity-summary');
    expectExactlyOnce(failures, path, documents.get(path), tailSummary, 'tail-calibration-recomputation-summary');
    expectExactlyOnce(failures, path, documents.get(path), productionAcquisitionSummary, 'production-acquisition-summary');
    expectExactlyOnce(failures, path, documents.get(path), modelStructureSummary, 'model-structure-summary');
    expectExactlyOnce(failures, path, documents.get(path), viewContractSummary, 'exact-view-contract-summary');
    expectExactlyOnce(failures, path, documents.get(path), manualCaptureSummary, 'manual-capture-boundary-summary');
    expectExactlyOnce(failures, path, documents.get(path), decisionThresholdSummary, 'decision-threshold-summary');
  }

  expectExactlyOnce(
    failures,
    'README.md',
    documents.get('README.md'),
    `The final ${seedCountWord}-seed, ${rbwCountWord}-interstitial-RBW regression ran ${formatted.attempts} acquisition attempts and classified ${formatted.representatives} capture-conditional representatives (${formatted.qualifiedEnvelopeRepresentatives} qualified envelope and ${formatted.censoredSpectrumRepresentatives} censored spectrum-only): hierarchical accuracy was ${formatted.hierarchicalAccuracy}, known coverage ${formatted.knownCoverage}, covered-known hierarchical accuracy ${formatted.coveredKnownHierarchicalAccuracy}, fitted-unknown and strict-holdout rejection ${formatted.fittedUnknownTemplateRejectionRate}, and there were zero disallowed false-accept attempts. All ${formatted.nuisanceCells} exact-equivalence nuisance cells, ${formatted.representativePairs} representative pairs, ${formatted.evidenceViewPairs} evidence-view pairs, and ${formatted.onlineSpectrumPairs} online-spectrum pairs matched within \`${formatted.tolerance}\` with zero discrepancies.`,
    'validation-summary',
  );
  expectExactlyOnce(
    failures,
    'README.md',
    documents.get('README.md'),
    `On the final held-out event-phase seeds, BLE acquired at one or more tested RBWs for ${bleCoverage} seeds at 24 dB and ${ble32Coverage} at 32 dB; all ${formatted.bleRepresentatives} admitted BLE censored spectrum-only classifications resolved only to Bluetooth-like band activity.`,
    'BLE-summary',
  );

  const researchPath = 'docs/BAYESIAN_DETECTION_CLASSIFICATION_RESEARCH.md';
  expectExactlyOnce(
    failures,
    researchPath,
    documents.get(researchPath),
    `The final regression matrix uses held-out nuisance seeds ${seedList}; SNR ${snrList} dB; and interstitial RBW divisors ${rbwList} rather than a fitted or support-calibration grid point. It audits the fitted unknowns, ${strictHoldoutCountWord} strict unknown ${pluralize(metrics.strictUnknownHoldouts, 'holdout')}, ${ambiguityCountWord} ambiguity-only ${pluralize(metrics.ambiguityStressCases, 'case')}, ${pairCountWord} exact-equivalence ${pluralize(metrics.exactEquivalencePairs, 'pair')}, and the acquisition-only one-timeslot GSM case separately.`,
    'validation-matrix',
  );
  expectExactlyOnce(
    failures,
    researchPath,
    documents.get(researchPath),
    `The run covered ${formatted.attempts} acquisition attempts, admitted ${formatted.admitted} (${formatted.admissionRate}), and produced ${formatted.representatives} unique capture-conditional classifications: ${formatted.qualifiedEnvelopeRepresentatives} qualified envelope and ${formatted.censoredSpectrumRepresentatives} censored spectrum-only. Conditional hierarchical accuracy was ${formatted.hierarchicalAccuracy}, known coverage ${formatted.knownCoverage}, covered-known hierarchical accuracy ${formatted.coveredKnownHierarchicalAccuracy}, known top-leaf accuracy ${formatted.knownTopLeafAccuracy}, and minimum high-SNR known-class hierarchical accuracy ${formatted.minimumHighSnrKnownClassHierarchicalAccuracy}. On ${formatted.properScoreSamples} singleton-truth, observation-domain-eligible proper-score samples, fitted-template log loss was ${formatted.fittedTemplateLogLoss}, multiclass Brier score ${formatted.fittedTemplateMulticlassBrier}, and expected calibration error ${formatted.fittedTemplateExpectedCalibrationError}. Fitted-unknown AUROC and rejection were ${formatted.fittedUnknownPosteriorAuroc}; scenario-excluded strict-typicality AUROC was ${formatted.strictTypicalityAuroc} and admitted strict-holdout rejection was ${formatted.strictUnknownRejectionRate}.`,
    'validation-metrics',
  );
  expectExactlyOnce(
    failures,
    researchPath,
    documents.get(researchPath),
    `The exact-pair audit covered ${formatted.nuisanceCells} nuisance cells, ${formatted.representativePairs} capture-conditional representative pairs, ${formatted.evidenceViewPairs} evidence-view pairs, and ${formatted.onlineSpectrumPairs} online-spectrum pairs with zero discrepancies at \`${formatted.tolerance}\` tolerance. Compatibility was ${formatted.exactEquivalenceCompatibleRate}, with zero unknown false accepts and zero disallowed false-accept attempts. BLE high-SNR acquisition covered ${bleCoverage} independent seeds at 24 dB and ${ble32Coverage} at 32 dB at one or more held-out RBWs.`,
    'equivalence-and-BLE-metrics',
  );
  expectExactlyOnce(
    failures,
    researchPath,
    documents.get(researchPath),
    `Across the final ${seedCountWord} held-out event-phase seeds and ${rbwCountWord} interstitial RBWs, at least one RBW acquired BLE in ${bleCoverage} seeds at 24 dB and ${ble32Coverage} at 32 dB; all ${formatted.bleRepresentatives} admitted BLE censored spectrum-only classifications returned only Bluetooth-like band activity.`,
    'BLE-detail',
  );

  const emsoPath = 'docs/SIGNALLAB_EMSO_CLASSIFIER_CONTRACT.md';
  expectExactlyOnce(
    failures,
    emsoPath,
    documents.get(emsoPath),
    `The held-out nuisance-shift validator uses unseen seeds ${codeSeedList}; SNR values ${snrList} dB; interstitial RBW divisors ${rbwList}; standard ${metrics.standardObservationHorizon}- and full-band 2.4 GHz ${metrics.fullBandObservationHorizon}-opportunity horizons; and an exact ${classificationAdmissionCountWord}-admission classification window.`,
    'validation-matrix',
  );
  expectExactlyOnce(
    failures,
    emsoPath,
    documents.get(emsoPath),
    `The final regression ran ${formatted.attempts} acquisition attempts. It admitted ${formatted.admitted} attempts (${formatted.admissionRate}) and produced ${formatted.representatives} unique capture-conditional classifications: ${formatted.qualifiedEnvelopeRepresentatives} qualified envelope and ${formatted.censoredSpectrumRepresentatives} censored spectrum-only. Conditional hierarchical accuracy was ${formatted.hierarchicalAccuracy}, known coverage ${formatted.knownCoverage}, covered-known hierarchical accuracy ${formatted.coveredKnownHierarchicalAccuracy}, known top-leaf accuracy ${formatted.knownTopLeafAccuracy}, and the minimum high-SNR known-class hierarchical accuracy was ${formatted.minimumHighSnrKnownClassHierarchicalAccuracy}. On ${formatted.properScoreSamples} singleton-truth proper-score samples, fitted-template log loss was ${formatted.fittedTemplateLogLoss}, multiclass Brier score ${formatted.fittedTemplateMulticlassBrier}, and ECE ${formatted.fittedTemplateExpectedCalibrationError}. Fitted-unknown AUROC and rejection were both ${formatted.fittedUnknownPosteriorAuroc}; scenario-excluded strict-typicality AUROC was ${formatted.strictTypicalityAuroc} and admitted strict-holdout rejection was ${formatted.strictUnknownRejectionRate}.`,
    'validation-metrics',
  );
  expectExactlyOnce(
    failures,
    emsoPath,
    documents.get(emsoPath),
    `All ${formatted.nuisanceCells} exact-equivalence nuisance cells yielded ${formatted.representativePairs} matched capture-conditional representative pairs, ${formatted.evidenceViewPairs} matched evidence-view pairs, and ${formatted.onlineSpectrumPairs} matched online-spectrum pairs with zero discrepancies at \`${formatted.tolerance}\` tolerance. Exact-equivalence compatibility was ${formatted.exactEquivalenceCompatibleRate}, and both the unknown false-accept count and disallowed false-accept attempt count were zero.`,
    'equivalence-metrics',
  );
  expectExactlyOnce(
    failures,
    emsoPath,
    documents.get(emsoPath),
    `Across the final ${seedCountWord} held-out event-phase seeds and ${rbwCountWord} interstitial RBWs, BLE acquired at one or more RBWs for ${bleCoverage} seeds at 24 dB and ${ble32Coverage} at 32 dB. All ${formatted.bleRepresentatives} admitted BLE censored spectrum-only classifications returned Bluetooth-like band activity.`,
    'BLE-detail',
  );

  const uiPath = 'docs/UI_UX_CONTRACTS.md';
  expectExactlyOnce(
    failures,
    uiPath,
    documents.get(uiPath),
    `The final development regression uses held-out seeds ${seedList} and interstitial RBW divisors ${rbwList}. It covers ${formatted.attempts} attempts and ${formatted.representatives} capture-conditional classifications: ${formatted.qualifiedEnvelopeRepresentatives} qualified envelope and ${formatted.censoredSpectrumRepresentatives} censored spectrum-only. Hierarchical accuracy is ${formatted.hierarchicalAccuracy}, known coverage ${formatted.knownCoverage}, covered-known hierarchical accuracy ${formatted.coveredKnownHierarchicalAccuracy}, fitted-unknown and strict-holdout rejection ${formatted.fittedUnknownTemplateRejectionRate}, and disallowed false-accept attempts zero. All ${formatted.nuisanceCells} exact-equivalence cells, ${formatted.representativePairs} capture-conditional representative pairs, ${formatted.evidenceViewPairs} evidence-view pairs, and ${formatted.onlineSpectrumPairs} online-spectrum pairs match within \`${formatted.tolerance}\` with zero discrepancies.`,
    'validation-summary',
  );
  expectExactlyOnce(
    failures,
    uiPath,
    documents.get(uiPath),
    `The final held-out synthetic run acquired BLE at one or more tested RBWs for ${bleCoverage} event-phase seeds at 24 dB and ${ble32Coverage} at 32 dB; all ${formatted.bleRepresentatives} admitted BLE censored spectrum-only classifications returned only Bluetooth-like band activity.`,
    'BLE-summary',
  );
}

async function main() {
  const paths = [
    MODEL_PATH,
    MANIFEST_PATH,
    REPORT_PATH,
    NODE_VERSION_PATH,
    ...PUBLICATION_PATHS,
  ];
  const atomizerPaths = new Set(PUBLICATION_PATHS);
  const contents = await Promise.all(paths.map((path) => readFile(resolve(atomizerPaths.has(path) ? ATOMIZER_ROOT : REPOSITORY_ROOT, path))));
  const byPath = new Map(paths.map((path, index) => [path, contents[index]]));
  const modelSource = byPath.get(MODEL_PATH).toString('utf8');
  const modelSha256 = createHash('sha256').update(modelSource).digest('hex');
  const generatedModel = parseGeneratedModel(modelSource);
  const generatedModelContentSha256 = createHash('sha256')
    .update(JSON.stringify(generatedModel))
    .digest('hex');
  const modelContentMatches = [...modelSource.matchAll(
    /BAYESIAN_OBSERVABLE_MODEL_CONTENT_SHA256 = '([a-f0-9]{64})'/g,
  )];
  if (modelContentMatches.length !== 1) {
    throw new Error(`${MODEL_PATH} must contain exactly one classifier model content SHA-256 declaration`);
  }
  const declaredModelContentSha256 = modelContentMatches[0][1];
  const manifest = byPath.get(MANIFEST_PATH).toString('utf8');
  const manifestMatches = [...manifest.matchAll(/BAYESIAN_OBSERVABLE_MODEL_SHA256 = '([a-f0-9]{64})'/g)];
  if (manifestMatches.length !== 1) {
    throw new Error(`${MANIFEST_PATH} must contain exactly one classifier model SHA-256 declaration`);
  }
  const manifestSha256 = manifestMatches[0][1];
  const manifestContentMatches = [...manifest.matchAll(
    /BAYESIAN_OBSERVABLE_MODEL_CONTENT_SHA256 = '([a-f0-9]{64})'/g,
  )];
  if (manifestContentMatches.length !== 1) {
    throw new Error(`${MANIFEST_PATH} must contain exactly one classifier model content SHA-256 declaration`);
  }
  const manifestContentSha256 = manifestContentMatches[0][1];

  let report;
  try {
    report = JSON.parse(byPath.get(REPORT_PATH).toString('utf8'));
  } catch (error) {
    throw new Error(`${REPORT_PATH} is not valid JSON: ${error instanceof Error ? error.message : String(error)}`);
  }

  const failures = [];
  const { validationAcceptance, ...reportEvidence } = report;
  if (validationAcceptance === null || typeof validationAcceptance !== 'object'
    || Array.isArray(validationAcceptance)) {
    throw new Error(`${REPORT_PATH} validationAcceptance must be an object`);
  }
  expectEqual(failures, validationAcceptance.schemaVersion, 1, 'validator acceptance schema');
  expectEqual(failures, valueAt(report, 'validationAcceptance.status'), 'passed', 'validator acceptance status');
  expectEqual(failures, validationAcceptance.acceptancePolicyId, PINNED_ACCEPTANCE_POLICY_ID, 'validator acceptance policy');
  expectEqual(failures, validationAcceptance.scope, 'full-corpus', 'validator acceptance scope');
  expectEqual(failures, validationAcceptance.failureCount, 0, 'validator acceptance failure count field');
  expectEqual(failures, arrayAt(report, 'validationAcceptance.failures').length, 0, 'validator acceptance failure count');
  expectEqual(
    failures,
    validationAcceptance.evidenceSha256,
    createHash('sha256').update(JSON.stringify(reportEvidence)).digest('hex'),
    'validator acceptance evidence SHA-256',
  );
  expectEqual(failures, valueAt(report, 'corpus.version'), PINNED_CORPUS_VERSION, 'classifier corpus version');
  expectEqual(failures, valueAt(report, 'model.id'), PINNED_MODEL_ID, 'classifier model ID');
  expectEqual(failures, valueAt(report, 'model.preprocessing'), PINNED_PREPROCESSING_ID, 'classifier preprocessing ID');
  expectEqual(failures, valueAt(report, 'model.priorId'), PINNED_PRIOR_ID, 'classifier prior ID');
  expectEqual(failures, valueAt(report, 'model.calibrationId'), PINNED_CALIBRATION_ID, 'classifier calibration ID');
  expectEqual(failures, valueAt(report, 'model.decisionPolicyId'), PINNED_DECISION_POLICY_ID, 'classifier decision-policy ID');
  expectEqual(failures, numberAt(report, 'model.classCount', { integer: true }), 12, 'classifier class count');
  expectEqual(failures, generatedModel.id, PINNED_MODEL_ID, 'generated classifier model ID');
  expectEqual(failures, generatedModel.corpusVersion, PINNED_CORPUS_VERSION, 'generated classifier corpus version');
  expectEqual(failures, generatedModel.sourceCommit, PINNED_SIGNAL_LAB_COMMIT, 'generated classifier source commit');
  expectEqual(failures, generatedModel.preprocessing, PINNED_PREPROCESSING_ID, 'generated classifier preprocessing ID');
  expectEqual(failures, generatedModel.priorId, PINNED_PRIOR_ID, 'generated classifier prior ID');
  expectEqual(failures, generatedModel.calibrationId, PINNED_CALIBRATION_ID, 'generated classifier calibration ID');
  expectEqual(
    failures,
    byPath.get(NODE_VERSION_PATH).toString('utf8').trim(),
    PINNED_TRAINING_RUNTIME_IDENTITY.nodeVersion,
    'repository exact Node.js version pin',
  );
  const generatedAttemptSamplingWorkerRuntimeSha256 =
    generatedModel.trainingMatrix?.attemptSamplingWorkerRuntimeSha256;
  if (typeof generatedAttemptSamplingWorkerRuntimeSha256 !== 'string'
    || !/^[a-f0-9]{64}$/.test(generatedAttemptSamplingWorkerRuntimeSha256)) {
    failures.push('generated attempt-sampling worker runtime SHA-256 must be a lowercase SHA-256');
  }
  expectEqual(
    failures,
    valueAt(report, 'model.attemptSamplingWorkerRuntimeSha256'),
    generatedAttemptSamplingWorkerRuntimeSha256,
    'reported attempt-sampling worker runtime SHA-256',
  );
  expectEqual(
    failures,
    valueAt(report, 'matrix.attemptSamplingWorkerRuntimeSha256'),
    generatedAttemptSamplingWorkerRuntimeSha256,
    'matrix attempt-sampling worker runtime SHA-256',
  );
  const generatedTrainingRuntimeIdentity =
    generatedModel.trainingMatrix?.trainingRuntimeIdentity;
  expectDeepEqual(
    failures,
    generatedTrainingRuntimeIdentity,
    PINNED_TRAINING_RUNTIME_IDENTITY,
    'generated training runtime identity',
  );
  expectDeepEqual(
    failures,
    valueAt(report, 'model.trainingRuntimeIdentity'),
    generatedTrainingRuntimeIdentity,
    'reported training runtime identity',
  );
  expectDeepEqual(
    failures,
    valueAt(report, 'matrix.trainingRuntimeIdentity'),
    generatedTrainingRuntimeIdentity,
    'matrix training runtime identity',
  );
  expectEqual(failures, Array.isArray(generatedModel.classModels) ? generatedModel.classModels.length : -1, 12, 'generated classifier class count');
  expectDeepEqual(failures, generatedModel.dimensions, PINNED_DIMENSIONS, 'generated classifier ordered dimensions');
  const generatedClassIds = Array.isArray(generatedModel.classModels)
    ? generatedModel.classModels.map((model) => model.id)
    : [];
  expectDeepEqual(failures, generatedClassIds, PINNED_CLASS_IDS, 'generated classifier class IDs and order');
  for (const [index, model] of generatedModel.classModels.entries()) {
    expectEqual(
      failures,
      model.logPrior,
      Math.log(PINNED_ENGINEERING_PRIOR[model.id]),
      `generated classifier class ${index} log prior`,
    );
  }
  const generatedComponentAssignmentsByView = Object.fromEntries(
    PINNED_TAIL_VIEWS.map((view) => [view, []]),
  );
  const generatedLikelihoodComponentIdsByView = Object.fromEntries(
    PINNED_TAIL_VIEWS.map((view) => [view, []]),
  );
  const generatedDecomposedSourceScenarioIds = new Set();
  let minimumDecomposedModeFitSampleCount = Number.POSITIVE_INFINITY;
  for (const [classIndex, model] of generatedModel.classModels.entries()) {
    expectEqual(
      failures,
      Object.hasOwn(model, 'components'),
      false,
      `generated classifier class ${classIndex} legacy components field absence`,
    );
    if (model.componentsByView === null
      || typeof model.componentsByView !== 'object'
      || Array.isArray(model.componentsByView)) {
      throw new Error(`${MODEL_PATH} class ${model.id} componentsByView must be an object`);
    }
    expectDeepEqual(
      failures,
      Object.keys(model.componentsByView).sort(),
      [...PINNED_TAIL_VIEWS].sort(),
      `generated classifier class ${model.id} componentsByView key set`,
    );
    const spectrumComponentIds = generatedComponentsForView(model, 'spectrum-only')
      .map((component) => component.id);
    const spectrumSourceScenarioIds = uniqueSourceScenarioIds(
      generatedComponentsForView(model, 'spectrum-only'),
    );
    for (const view of PINNED_TAIL_VIEWS) {
      const components = generatedComponentsForView(model, view);
      const expectedDimensions = expectedDimensionsForView(view);
      const censoredBluetoothEnvelope = model.id === 'bluetooth-like'
        && view !== 'spectrum-only';
      expectDeepEqual(
        failures,
        components.map((component) => component.id),
        censoredBluetoothEnvelope ? [] : spectrumComponentIds,
        `generated classifier class ${model.id}/${view} component scenario identity/order`,
      );
      expectDeepEqual(
        failures,
        uniqueSourceScenarioIds(components),
        censoredBluetoothEnvelope ? [] : spectrumSourceScenarioIds,
        `generated classifier class ${model.id}/${view} source-scenario identity/order`,
      );
      const weightTotal = components.reduce(
        (sum, component) => sum + Math.exp(component.logWeight),
        0,
      );
      if (censoredBluetoothEnvelope) {
        expectEqual(
          failures,
          weightTotal,
          0,
          `generated classifier class ${model.id}/${view} empty censored mixture weight total`,
        );
      } else {
        expectNear(
          failures,
          weightTotal,
          1,
          1e-9,
          `generated classifier class ${model.id}/${view} mixture weight total`,
        );
      }
      const componentsBySourceScenario = new Map();
      for (const [componentIndex, component] of components.entries()) {
        generatedLikelihoodComponentIdsByView[view].push(component.id);
        const label = `generated classifier component ${model.id}/${view}/${componentIndex}`;
        if (typeof component.id !== 'string' || component.id.length === 0) {
          failures.push(`${label} ID must be a non-empty string`);
        }
        if (typeof component.sourceScenarioId !== 'string' || component.sourceScenarioId.length === 0) {
          failures.push(`${label} sourceScenarioId must be a non-empty string`);
        }
        if (typeof component.modeId !== 'string' || component.modeId.length === 0) {
          failures.push(`${label} modeId must be a non-empty string`);
        }
        expectNonNegativeInteger(
          failures,
          component.fitSampleCount,
          `${label} fit sample count`,
          { positive: true },
        );
        const sourceScenarioId = componentSourceScenarioId(component);
        const owned = componentsBySourceScenario.get(sourceScenarioId) ?? [];
        componentsBySourceScenario.set(sourceScenarioId, [...owned, component]);
        if (typeof component.logWeight !== 'number' || !Number.isFinite(component.logWeight)) {
          failures.push(`${label} log weight must be finite`);
        }
        expectEqual(
          failures,
          component.degreesOfFreedom,
          PINNED_COMPONENT_DEGREES_OF_FREEDOM,
          `${label} degrees of freedom`,
        );
        expectDeepEqual(failures, component.dimensions, expectedDimensions, `${label} dimensions`);
        if (!Array.isArray(component.location)) {
          failures.push(`${label} location must be an array`);
        } else {
          expectEqual(failures, component.location.length, expectedDimensions.length, `${label} location dimension`);
          if (component.location.some((value) => typeof value !== 'number' || !Number.isFinite(value))) {
            failures.push(`${label} location must contain only finite numbers`);
          }
        }
        if (!Array.isArray(component.scale)) {
          failures.push(`${label} scale must be an array`);
        } else {
          expectEqual(failures, component.scale.length, expectedDimensions.length, `${label} scale row count`);
          for (const [rowIndex, row] of component.scale.entries()) {
            if (!Array.isArray(row)) {
              failures.push(`${label} scale row ${rowIndex} must be an array`);
              continue;
            }
            expectEqual(failures, row.length, expectedDimensions.length, `${label} scale row ${rowIndex} dimension`);
            if (row.some((value) => typeof value !== 'number' || !Number.isFinite(value))) {
              failures.push(`${label} scale row ${rowIndex} must contain only finite numbers`);
            }
          }
          validatePositiveDefiniteScale(
            failures,
            component.scale,
            expectedDimensions.length,
            label,
          );
        }
      }
      for (const [sourceScenarioId, sourceComponents] of componentsBySourceScenario) {
        generatedComponentAssignmentsByView[view].push({
          scenarioId: sourceScenarioId,
          classId: model.id,
        });
        const expectedModeCount = PINNED_CSMA_DECOMPOSED_SOURCE_SCENARIO_IDS
          .includes(sourceScenarioId) ? 3 : 1;
        expectEqual(
          failures,
          sourceComponents.length,
          expectedModeCount,
          `generated classifier ${model.id}/${view}/${sourceScenarioId} mode count`,
        );
        const observedFitSampleCount = sourceComponents.reduce(
          (sum, component) => sum + (Number.isSafeInteger(component.fitSampleCount)
            ? component.fitSampleCount : 0),
          0,
        );
        const expectedFitSampleCount = generatedModel.trainingMatrix
          ?.fittingRepresentativeCountsByScenarioByView?.[sourceScenarioId]?.[view];
        expectEqual(
          failures,
          observedFitSampleCount,
          expectedFitSampleCount,
          `generated classifier ${model.id}/${view}/${sourceScenarioId} source-owned fit sample count`,
        );
        if (expectedModeCount === 1) {
          expectEqual(
            failures,
            sourceComponents[0]?.id,
            sourceScenarioId,
            `generated classifier ${model.id}/${view}/${sourceScenarioId} ordinary component ID`,
          );
          expectEqual(
            failures,
            sourceComponents[0]?.modeId,
            'single-population',
            `generated classifier ${model.id}/${view}/${sourceScenarioId} ordinary mode ID`,
          );
        } else {
          if (view === 'spectrum-only') generatedDecomposedSourceScenarioIds.add(sourceScenarioId);
          const sharedScale = sourceComponents[0]?.scale;
          for (const [modeIndex, component] of sourceComponents.entries()) {
            const expectedModeId = `csma-activity-mode-${modeIndex + 1}-of-3`;
            expectEqual(
              failures,
              component.id,
              `${sourceScenarioId}/${expectedModeId}`,
              `generated classifier ${model.id}/${view}/${sourceScenarioId} mode ${modeIndex + 1} component ID`,
            );
            expectEqual(
              failures,
              component.modeId,
              expectedModeId,
              `generated classifier ${model.id}/${view}/${sourceScenarioId} mode ${modeIndex + 1} mode ID`,
            );
            expectDeepEqual(
              failures,
              component.scale,
              sharedScale,
              `generated classifier ${model.id}/${view}/${sourceScenarioId} mode ${modeIndex + 1} shared scale`,
            );
            if (Number.isSafeInteger(component.fitSampleCount)) {
              minimumDecomposedModeFitSampleCount = Math.min(
                minimumDecomposedModeFitSampleCount,
                component.fitSampleCount,
              );
              if (component.fitSampleCount < PINNED_MINIMUM_DECOMPOSED_MODE_FIT_SAMPLE_COUNT) {
                failures.push(`generated classifier ${model.id}/${view}/${sourceScenarioId} mode ${modeIndex + 1} has only ${component.fitSampleCount} fit samples`);
              }
            }
          }
          const partitionDimensionIndex = sourceComponents[0]?.dimensions
            ?.indexOf(PINNED_LIKELIHOOD_COMPONENT_DECOMPOSITION_POLICY.csmaPartitionFeature) ?? -1;
          const partitionCenters = sourceComponents.map((component) =>
            component.location?.[partitionDimensionIndex]);
          if (partitionDimensionIndex < 0
            || partitionCenters.some((center) => typeof center !== 'number' || !Number.isFinite(center))
            || partitionCenters.some((center, index) => index > 0 && center <= partitionCenters[index - 1])) {
            failures.push(`generated classifier ${model.id}/${view}/${sourceScenarioId} CSMA partition centers must be finite and strictly increasing`);
          }
        }
        for (const component of sourceComponents) {
          const expectedWeight = (1 / componentsBySourceScenario.size)
            * (component.fitSampleCount / observedFitSampleCount);
          expectNear(
            failures,
            Math.exp(component.logWeight),
            expectedWeight,
            1e-9,
            `generated classifier ${model.id}/${view}/${component.id} source-owned empirical weight`,
          );
        }
      }
    }
  }
  for (const view of PINNED_TAIL_VIEWS) {
    const assignments = generatedComponentAssignmentsByView[view];
    expectEqual(
      failures,
      generatedLikelihoodComponentIdsByView[view].length,
      PINNED_LIKELIHOOD_COMPONENT_COUNTS_BY_VIEW[view],
      `generated classifier ${view} component count`,
    );
    expectEqual(
      failures,
      new Set(generatedLikelihoodComponentIdsByView[view]).size,
      PINNED_LIKELIHOOD_COMPONENT_COUNTS_BY_VIEW[view],
      `generated classifier ${view} unique likelihood component identity count`,
    );
    const expectedComponentIds = view === 'spectrum-only'
      ? generatedLikelihoodComponentIdsByView['spectrum-only']
      : generatedLikelihoodComponentIdsByView['spectrum-only'].filter((componentId) =>
          !PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORED_SCENARIO_IDS.some(
            (scenarioId) => componentId === scenarioId
              || componentId.startsWith(`${scenarioId}/`),
          ));
    expectDeepEqual(
      failures,
      generatedLikelihoodComponentIdsByView[view],
      expectedComponentIds,
      `generated classifier ${view}/spectrum-only censor-adjusted likelihood component identities`,
    );
    expectEqual(
      failures,
      assignments.length,
      PINNED_COMPONENT_SOURCE_SCENARIO_COUNTS_BY_VIEW[view],
      `generated classifier ${view} source-scenario assignment count`,
    );
    expectEqual(
      failures,
      new Set(assignments.map((assignment) => assignment.scenarioId)).size,
      PINNED_COMPONENT_SOURCE_SCENARIO_COUNTS_BY_VIEW[view],
      `generated classifier ${view} unique source-scenario count`,
    );
    const expectedAssignments = view === 'spectrum-only'
      ? generatedComponentAssignmentsByView['spectrum-only']
      : generatedComponentAssignmentsByView['spectrum-only'].filter((assignment) =>
          !PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORED_SCENARIO_IDS
            .includes(assignment.scenarioId));
    expectDeepEqual(
      failures,
      assignments,
      expectedAssignments,
      `generated classifier ${view}/spectrum-only censor-adjusted component assignments`,
    );
  }
  if (!Number.isFinite(minimumDecomposedModeFitSampleCount)) {
    failures.push('generated classifier must publish decomposed CSMA mode fit sample counts');
  }
  expectDeepEqual(
    failures,
    [...generatedDecomposedSourceScenarioIds].sort(),
    [...PINNED_CSMA_DECOMPOSED_SOURCE_SCENARIO_IDS].sort(),
    'generated classifier exact decomposed CSMA source scenario IDs',
  );
  expectEqual(failures, valueAt(report, 'corpus.manifestSplit.valid'), true, 'corpus manifest split validity');
  expectDeepEqual(
    failures,
    valueAt(report, 'corpus.manifestSplit.validatorOwnedPins.frequencyAgileEnvelopeCensoredScenarios'),
    PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORED_SCENARIO_IDS,
    'reported frequency-agile censored scenario pins',
  );
  expectDeepEqual(
    failures,
    valueAt(report, 'corpus.manifestSplit.validatorOwnedPins.componentSourceScenarioCountsByView'),
    PINNED_COMPONENT_SOURCE_SCENARIO_COUNTS_BY_VIEW,
    'reported source-scenario counts by view',
  );
  expectDeepEqual(
    failures,
    valueAt(report, 'corpus.manifestSplit.validatorOwnedPins.likelihoodComponentCountsByView'),
    PINNED_LIKELIHOOD_COMPONENT_COUNTS_BY_VIEW,
    'reported likelihood-component counts by view',
  );
  expectDeepEqual(
    failures,
    valueAt(report, 'corpus.manifestSplit.modelDeclared.likelihoodComponentDecompositionPolicy'),
    PINNED_LIKELIHOOD_COMPONENT_DECOMPOSITION_POLICY,
    'reported likelihood-component decomposition policy',
  );
  expectEqual(
    failures,
    valueAt(report, 'corpus.manifestSplit.modelDeclared.minimumDecomposedModeFitSampleCount'),
    minimumDecomposedModeFitSampleCount,
    'reported minimum decomposed-mode fit sample count',
  );
  const reportedComponentAssignmentsByView = objectAt(
    report,
    'corpus.manifestSplit.modelDeclared.componentAssignmentsByView',
  );
  expectDeepEqual(
    failures,
    Object.keys(reportedComponentAssignmentsByView).sort(),
    [...PINNED_TAIL_VIEWS].sort(),
    'reported component-assignment view key set',
  );
  const canonicalGeneratedComponentAssignmentsByView = Object.fromEntries(
    PINNED_TAIL_VIEWS.map((view) => [view, [...generatedComponentAssignmentsByView[view]]
      .sort((left, right) => left.scenarioId.localeCompare(right.scenarioId))]),
  );
  for (const view of PINNED_TAIL_VIEWS) {
    expectDeepEqual(
      failures,
      reportedComponentAssignmentsByView[view],
      canonicalGeneratedComponentAssignmentsByView[view],
      `reported/generated ${view} component assignments`,
    );
  }
  expectDeepEqual(
    failures,
    valueAt(report, 'corpus.manifestSplit.modelDeclared.componentAssignments'),
    canonicalGeneratedComponentAssignmentsByView['spectrum-only'],
    'reported legacy/spectrum-only component assignments',
  );
  expectDeepEqual(
    failures,
    valueAt(report, 'corpus.manifestSplit.expectedComponentAssignments'),
    canonicalGeneratedComponentAssignmentsByView['spectrum-only'],
    'independently expected/generated component assignments',
  );
  for (const path of [
    'corpus.manifestSplit.likelihoodComponentOwnershipMismatches',
    'corpus.manifestSplit.componentScenarioPopulationMismatches',
    'corpus.manifestSplit.componentAssignmentViewMismatches',
    'corpus.manifestSplit.componentArchitectureMismatches',
    'corpus.manifestSplit.frequencyAgileCensoringMatrixMismatches',
  ]) {
    expectEqual(failures, arrayAt(report, path).length, 0, `${path} length`);
  }
  const causalSamplingMetrics = validateGeneratedCausalSamplingAudit(generatedModel, failures);
  expectEqual(
    failures,
    generatedModel.trainingMatrix?.representativeEligibilityPolicy,
    PINNED_REPRESENTATIVE_ELIGIBILITY_POLICY,
    'generated representative eligibility policy',
  );
  expectEqual(
    failures,
    valueAt(report, 'model.minimumKnownSyntheticSupportRank'),
    PINNED_MINIMUM_KNOWN_SYNTHETIC_SUPPORT_RANK,
    'classifier minimum known synthetic support rank',
  );
  for (const [field, expected] of [
    ['modelAssetSha256', modelSha256],
    ['attemptSamplingWorkerRuntimeSha256', generatedAttemptSamplingWorkerRuntimeSha256],
    ['modelId', generatedModel.id],
    ['sourceCommit', generatedModel.sourceCommit],
    ['corpusVersion', generatedModel.corpusVersion],
    ['corpusSha256', generatedModel.corpusSha256],
    ['preprocessing', generatedModel.preprocessing],
    ['priorId', generatedModel.priorId],
    ['calibrationId', generatedModel.calibrationId],
    ['decisionPolicyId', PINNED_DECISION_POLICY_ID],
  ]) {
    expectEqual(failures, validationAcceptance[field], expected, `validator acceptance ${field}`);
  }
  expectDeepEqual(
    failures,
    validationAcceptance.trainingRuntimeIdentity,
    generatedTrainingRuntimeIdentity,
    'validator acceptance training runtime identity',
  );
  expectDeepEqual(
    failures,
    generatedModel.trainingMatrix?.signalLabProductionAcquisitionRegime,
    PINNED_PRODUCTION_ACQUISITION_REGIME,
    'generated production acquisition regime',
  );
  expectDeepEqual(
    failures,
    generatedModel.trainingMatrix?.detectedPowerSynthesisFilterPolicy,
    PINNED_DETECTED_POWER_SYNTHESIS_FILTER_POLICY,
    'generated detected-power synthesis-filter policy',
  );
  expectDeepEqual(
    failures,
    generatedModel.trainingMatrix?.productionAcquisitionRegimeHighSnrSeedCoveragePolicy,
    PINNED_PRODUCTION_HIGH_SNR_COVERAGE_POLICY,
    'generated production high-SNR coverage policy',
  );
  for (const [path, expected] of [
    ['selectionPolicy', PINNED_SELECTION_POLICY],
    ['matrix.selectionPolicy', PINNED_SELECTION_POLICY],
    ['matrix.captureTargetSelectionPolicy', PINNED_CAPTURE_TARGET_SELECTION_POLICY],
    ['matrix.automaticDetectedPowerSelectionCondition', PINNED_DETECTED_POWER_SELECTION_CONDITION],
    ['matrix.representativeWeightingPolicy', PINNED_REPRESENTATIVE_WEIGHTING_POLICY],
  ]) expectEqual(failures, valueAt(report, path), expected, path);
  for (const [key, expected] of [
    ['snrDb', PINNED_TRAINING_SNR_DB],
    ['rbwDivisors', PINNED_TRAINING_RBW_DIVISORS],
    ['seeds', PINNED_FITTING_SEEDS],
    ['tailCalibrationRbwDivisors', PINNED_TRAINING_RBW_DIVISORS],
    ['tailCalibrationSeeds', PINNED_CALIBRATION_SEEDS],
  ]) {
    expectDeepEqual(failures, generatedModel.trainingMatrix?.[key], expected, `generated ${key}`);
  }
  for (const [key, expected] of [
    ['tailCalibrationScoreUnit', PINNED_TAIL_POLICIES.scoreUnit],
    ['tailCalibrationRepresentativeSelectionPolicy', PINNED_TAIL_POLICIES.representativeSelection],
    ['tailCalibrationRepresentativeAggregationPolicy', PINNED_TAIL_POLICIES.representativeAggregation],
    ['tailCalibrationRuntimeInterpretationPolicy', PINNED_TAIL_POLICIES.runtimeInterpretation],
    ['tailCalibrationStatisticalInterpretation', PINNED_TAIL_POLICIES.statisticalInterpretation],
  ]) {
    expectEqual(failures, generatedModel.trainingMatrix?.[key], expected, `generated ${key}`);
  }
  for (const key of ['fittingAcquisitionRegimeIds', 'tailCalibrationAcquisitionRegimeIds']) {
    expectDeepEqual(
      failures,
      generatedModel.trainingMatrix?.[key],
      PINNED_ACQUISITION_REGIME_IDS,
      `generated ${key}`,
    );
  }
  expectEqual(failures, valueAt(report, 'matrix.scenarioSelection.mode'), 'full-corpus', 'validation scenario selection');
  const fittedComponentScenarioIds = generatedComponentAssignmentsByView['spectrum-only']
    .map((assignment) => assignment.scenarioId);
  const expectedFullCorpusScenarioIds = [
    ...fittedComponentScenarioIds,
    ...PINNED_NON_COMPONENT_FULL_CORPUS_SCENARIO_IDS,
  ];
  expectEqual(
    failures,
    new Set(expectedFullCorpusScenarioIds).size,
    expectedFullCorpusScenarioIds.length,
    'pinned full-corpus unique scenario identities',
  );
  expectEqual(
    failures,
    expectedFullCorpusScenarioIds.length,
    PINNED_FULL_CORPUS_SCENARIO_COUNT,
    'pinned full-corpus scenario denominator',
  );
  const validationScenarioIds = arrayAt(report, 'matrix.scenarioSelection.scenarioIds');
  for (const [index, scenarioId] of validationScenarioIds.entries()) {
    if (typeof scenarioId !== 'string' || scenarioId.length === 0) {
      failures.push(`validation scenario selection identity ${index} must be a non-empty string`);
    }
  }
  expectEqual(
    failures,
    new Set(validationScenarioIds).size,
    validationScenarioIds.length,
    'validation scenario selection unique identities',
  );
  expectDeepEqual(
    failures,
    [...validationScenarioIds].sort(),
    [...expectedFullCorpusScenarioIds].sort(),
    'validation full-corpus scenario identity set',
  );
  expectEqual(
    failures,
    numberAt(report, 'corpus.scenarios', { integer: true }),
    PINNED_FULL_CORPUS_SCENARIO_COUNT,
    'published complete corpus scenario denominator',
  );
  for (const [path, expected] of [
    ['matrix.nuisanceShiftSeeds', PINNED_VALIDATION_SEEDS],
    ['matrix.snrDb', PINNED_TRAINING_SNR_DB],
    ['matrix.rbwDivisors', PINNED_VALIDATION_RBW_DIVISORS],
  ]) expectDeepEqual(failures, arrayAt(report, path), expected, path);
  expectEqual(
    failures,
    valueAt(report, 'matrix.representativeEligibilityPolicy'),
    PINNED_REPRESENTATIVE_ELIGIBILITY_POLICY,
    'validation representative eligibility policy',
  );
  expectDeepEqual(
    failures,
    valueAt(report, 'matrix.sourceClocks'),
    PINNED_PRODUCTION_ACQUISITION_REGIME.sourceClocks,
    'validation source-clock policies',
  );
  expectDeepEqual(
    failures,
    valueAt(report, 'matrix.frequencyAgileFixedTuneEnvelopeCensoringPolicy'),
    PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORING_POLICY,
    'validation frequency-agile fixed-tune envelope-censoring policy',
  );
  const reportAttemptCount = numberAt(report, 'admission.attempted', { integer: true });
  const physicalDetectedPowerCaptures = numberAt(
    report,
    'admission.physicalDetectedPowerCaptures',
    { integer: true },
  );
  const physicalEnvelopeCaptures = numberAt(
    report,
    'admission.physicalEnvelopeCaptures',
    { integer: true },
  );
  const pairedNuisanceCells = numberAt(report, 'matrix.pairedNuisanceCells', { integer: true });
  const expectedFullFactorialAttemptCount = expectedFullCorpusScenarioIds.length
    * PINNED_TRAINING_SNR_DB.length
    * PINNED_VALIDATION_RBW_DIVISORS.length
    * PINNED_VALIDATION_SEEDS.length;
  expectEqual(
    failures,
    expectedFullFactorialAttemptCount,
    4_200,
    'pinned full-factorial validation attempt denominator',
  );
  expectEqual(
    failures,
    reportAttemptCount,
    expectedFullFactorialAttemptCount,
    'validation full-factorial attempted denominator',
  );
  expectEqual(
    failures,
    pairedNuisanceCells,
    expectedFullFactorialAttemptCount,
    'validation full-factorial paired nuisance-cell denominator',
  );
  expectEqual(
    failures,
    pairedNuisanceCells,
    reportAttemptCount,
    'validation paired nuisance-cell denominator',
  );
  expectDeepEqual(
    failures,
    valueAt(report, 'matrix.observationOpportunityHorizons'),
    { standard: 32, fullBand2g4: 96 },
    'validation observation-opportunity horizons',
  );
  for (const [path, expected] of [
    ['matrix.classificationAdmissions', 8],
    ['matrix.sweepPoints', 450],
    ['matrix.sweepTimeSeconds', 0.05],
    ['matrix.zeroSpanPoints', 450],
    ['matrix.zeroSpanSamplePeriodSeconds', 1 / 9_000],
  ]) expectEqual(failures, numberAt(report, path), expected, path);
  expectDeepEqual(
    failures,
    valueAt(report, 'matrix.detectionConfig'),
    PINNED_PRODUCTION_DETECTION_CONFIG,
    'validation production detection configuration',
  );
  const attemptsByObservationHorizon = objectAt(report, 'matrix.attemptsByObservationHorizon');
  expectDeepEqual(
    failures,
    Object.keys(attemptsByObservationHorizon).sort(),
    ['32', '96'],
    'validation attempts-by-observation-horizon key set',
  );
  let attemptsByObservationHorizonTotal = 0;
  for (const horizon of ['32', '96']) {
    const count = numberAt(report, `matrix.attemptsByObservationHorizon.${horizon}`, { integer: true });
    if (expectNonNegativeInteger(
      failures,
      count,
      `validation attempts at observation horizon ${horizon}`,
      { positive: true },
    )) attemptsByObservationHorizonTotal += count;
  }
  expectEqual(
    failures,
    attemptsByObservationHorizonTotal,
    pairedNuisanceCells,
    'validation attempts-by-observation-horizon denominator',
  );
  const runtimeBranchAttempts = objectAt(report, 'matrix.runtimeBranchAttempts');
  expectDeepEqual(
    failures,
    Object.keys(runtimeBranchAttempts).sort(),
    ['consecutiveSpectrum', 'qualifiedEnvelope'],
    'validation runtime-branch attempt key set',
  );
  for (const branch of ['consecutiveSpectrum', 'qualifiedEnvelope']) {
    expectEqual(
      failures,
      numberAt(report, `matrix.runtimeBranchAttempts.${branch}`, { integer: true }),
      pairedNuisanceCells,
      `validation ${branch} attempt denominator`,
    );
  }
  const validationSpectrumClockAudit = validateCausalClockAudit(
    report,
    'matrix.runtimeBranchClockAudits.consecutiveSpectrum',
    failures,
    'consecutiveSpectrum',
    reportAttemptCount,
    attemptsByObservationHorizon,
    0,
  );
  const validationQualifiedEnvelopeClockAudit = validateCausalClockAudit(
    report,
    'matrix.runtimeBranchClockAudits.qualifiedEnvelope',
    failures,
    'qualifiedEnvelope',
    reportAttemptCount,
    attemptsByObservationHorizon,
    physicalDetectedPowerCaptures,
  );
  expectEqual(
    failures,
    runtimeBranchAttempts.consecutiveSpectrum,
    validationSpectrumClockAudit.attempts,
    'validation consecutive-spectrum attempt/clock reconciliation',
  );
  expectEqual(
    failures,
    runtimeBranchAttempts.qualifiedEnvelope,
    validationQualifiedEnvelopeClockAudit.attempts,
    'validation qualified-envelope attempt/clock reconciliation',
  );
  expectDeepEqual(
    failures,
    valueAt(report, 'admission.runtimeBranchClockAudits.consecutiveSpectrum'),
    validationSpectrumClockAudit,
    'matrix/admission consecutive-spectrum clock-audit identity',
  );
  expectDeepEqual(
    failures,
    valueAt(report, 'admission.runtimeBranchClockAudits.qualifiedEnvelope'),
    validationQualifiedEnvelopeClockAudit,
    'matrix/admission qualified-envelope clock-audit identity',
  );
  validateHeldOutSourceSpanAudit(report, failures, expectedFullCorpusScenarioIds);
  expectEqual(
    failures,
    validationQualifiedEnvelopeClockAudit.attemptsWithDetectedPowerCapture,
    physicalDetectedPowerCaptures,
    'validation qualified-envelope clock-audit physical capture count',
  );
  const detectedPowerQualification = objectAt(
    report,
    'admission.detectedPowerAcquisitionQualification',
  );
  expectDeepEqual(
    failures,
    Object.keys(detectedPowerQualification).sort(),
    [
      'required',
      'modelDeclared',
      'automaticSelectionConditionRequired',
      'modelDeclaredSelectionCondition',
      'qualifiedEnvelopeSamples',
      'unqualifiedEnvelopeSamples',
      'missingOrUnissuedReceiptEnvelopeSamples',
      'missingOrUnissuedReceiptEnvelopeFeatureAttempts',
    ].sort(),
    'detected-power acquisition-qualification audit key set',
  );
  for (const field of ['required', 'modelDeclared']) {
    expectEqual(
      failures,
      detectedPowerQualification[field],
      PINNED_DETECTED_POWER_ACQUISITION_QUALIFICATION,
      `detected-power acquisition qualification ${field}`,
    );
  }
  for (const field of [
    'automaticSelectionConditionRequired',
    'modelDeclaredSelectionCondition',
  ]) {
    expectEqual(
      failures,
      detectedPowerQualification[field],
      PINNED_DETECTED_POWER_SELECTION_CONDITION,
      `detected-power automatic selection condition ${field}`,
    );
  }
  for (const field of [
    'qualifiedEnvelopeSamples',
    'unqualifiedEnvelopeSamples',
    'missingOrUnissuedReceiptEnvelopeSamples',
    'missingOrUnissuedReceiptEnvelopeFeatureAttempts',
  ]) {
    expectNonNegativeInteger(
      failures,
      detectedPowerQualification[field],
      `detected-power acquisition qualification ${field}`,
    );
  }
  expectEqual(
    failures,
    detectedPowerQualification.qualifiedEnvelopeSamples,
    physicalEnvelopeCaptures,
    'detected-power qualified/physical envelope denominator',
  );
  for (const field of [
    'unqualifiedEnvelopeSamples',
    'missingOrUnissuedReceiptEnvelopeSamples',
    'missingOrUnissuedReceiptEnvelopeFeatureAttempts',
  ]) {
    expectEqual(
      failures,
      detectedPowerQualification[field],
      0,
      `detected-power acquisition qualification excluded ${field}`,
    );
  }
  const unavailablePhysicalEnvelopeCaptures = numberAt(
    report,
    'admission.unavailablePhysicalEnvelopeCaptures',
    { integer: true },
  );
  expectEqual(
    failures,
    physicalDetectedPowerCaptures,
    validationQualifiedEnvelopeClockAudit.attemptsWithDetectedPowerCapture,
    'admission/qualified-envelope clock physical detected-power capture count',
  );
  expectEqual(failures, unavailablePhysicalEnvelopeCaptures, 0, 'unavailable physical envelope capture count');
  expectEqual(
    failures,
    arrayAt(report, 'admission.unavailablePhysicalEnvelopeCaptureExamples').length,
    0,
    'unavailable physical envelope capture examples',
  );
  expectEqual(
    failures,
    arrayAt(report, 'admission.invalidCausalCaptureSemantics').length,
    0,
    'invalid causal capture semantics',
  );
  const envelopeUnavailableByCode = objectAt(report, 'admission.envelopeFeatureUnavailableByCode');
  expectEqual(
    failures,
    Object.keys(envelopeUnavailableByCode).length,
    0,
    'physical envelope feature-unavailable code population',
  );
  for (const path of [
    'admission.causalEnvelopeSamples',
    'admission.expectedCausalEnvelopeSamples',
    'admission.uniqueCausalEnvelopeSamples',
  ]) expectEqual(failures, numberAt(report, path, { integer: true }), physicalEnvelopeCaptures, path);
  expectEqual(
    failures,
    numberAt(report, 'admission.admitted', { integer: true }),
    physicalDetectedPowerCaptures,
    'admission admitted/physical detected-power capture denominator',
  );
  const causalEnvelopeAvailabilityCells = arrayAt(report, 'admission.causalEnvelopeAvailabilityCells');
  expectEqual(
    failures,
    causalEnvelopeAvailabilityCells.length,
    reportAttemptCount,
    'causal envelope availability complete attempt denominator',
  );
  let availabilityPhysicalCaptureCount = 0;
  let availabilityReceiptQualifiedCaptureCount = 0;
  let availabilityQualifiedEnvelopeCount = 0;
  let availabilityCensoredSpectrumCount = 0;
  const availabilityAttemptIds = [];
  for (const [index, cell] of causalEnvelopeAvailabilityCells.entries()) {
    if (cell === null || typeof cell !== 'object' || Array.isArray(cell)
      || typeof cell.attemptId !== 'string' || cell.attemptId.length === 0) {
      failures.push(`causal envelope availability cell ${index} must publish an attempt identity`);
      continue;
    }
    availabilityAttemptIds.push(cell.attemptId);
    if (cell.detectedPowerCaptureCount !== 0 && cell.detectedPowerCaptureCount !== 1) {
      failures.push(`causal envelope availability ${cell.attemptId} must consume zero or one detected-power capture`);
    } else {
      availabilityPhysicalCaptureCount += cell.detectedPowerCaptureCount;
    }
    expectEqual(
      failures,
      cell.detectedPowerCaptureCount,
      cell.spectrumRuntimeAdmitted ? 1 : 0,
      `causal envelope availability ${cell.attemptId} capture-after-admission semantics`,
    );
    const expectedReceiptQualified = cell.detectedPowerCaptureCount === 1;
    expectEqual(
      failures,
      cell.detectedPowerAcquisitionReceiptQualified,
      expectedReceiptQualified,
      `causal envelope availability ${cell.attemptId} receipt qualification`,
    );
    expectEqual(
      failures,
      cell.detectedPowerCaptureReceiptVerified,
      expectedReceiptQualified,
      `causal envelope availability ${cell.attemptId} issued-receipt verification`,
    );
    if (expectedReceiptQualified) {
      availabilityReceiptQualifiedCaptureCount += 1;
      expectEqual(failures, cell.detectedPowerCaptureReceiptSchemaVersion, 4, `causal envelope availability ${cell.attemptId} receipt schema`);
      if (typeof cell.physicalCaptureId !== 'string' || cell.physicalCaptureId.length === 0) {
        failures.push(`causal envelope availability ${cell.attemptId} must publish its physical capture ID`);
      }
      if (cell.detectedPowerEvidenceDisposition === 'censored-frequency-agile-spectrum-only') {
        availabilityCensoredSpectrumCount += 1;
        expectEqual(failures, cell.envelopeFeatureAvailable, false, `causal envelope availability ${cell.attemptId} censored envelope absence`);
        expectEqual(failures, cell.captureProjectionKind, 'current-qualified-agile-latest-member', `causal envelope availability ${cell.attemptId} agile projection`);
        expectEqual(failures, cell.projectedAssociationMode, 'frequency-agile-2g4-activity', `causal envelope availability ${cell.attemptId} agile projected mode`);
        expectEqual(failures, cell.classificationEvidenceView, 'spectrum-only', `causal envelope availability ${cell.attemptId} censored classifier view`);
        expectEqual(failures, cell.envelopeEvidenceCensoringPolicyId, PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORING_POLICY.id, `causal envelope availability ${cell.attemptId} censoring policy`);
        if ('detectedPowerAcquisitionQualification' in cell) {
          failures.push(`causal envelope availability ${cell.attemptId} censored capture must not admit envelope qualification`);
        }
        if ('detectedPowerSelectionCondition' in cell) {
          failures.push(`causal envelope availability ${cell.attemptId} censored capture must not admit an envelope selection condition`);
        }
      } else if (cell.detectedPowerEvidenceDisposition === 'admitted-envelope') {
        availabilityQualifiedEnvelopeCount += 1;
        expectEqual(failures, cell.envelopeFeatureAvailable, true, `causal envelope availability ${cell.attemptId} envelope availability`);
        expectEqual(failures, cell.captureProjectionKind, 'current-active-physical-representative', `causal envelope availability ${cell.attemptId} direct projection`);
        if (cell.classificationEvidenceView === 'spectrum-only') {
          failures.push(`causal envelope availability ${cell.attemptId} admitted envelope cannot select spectrum-only`);
        }
        if ('envelopeEvidenceCensoringPolicyId' in cell) {
          failures.push(`causal envelope availability ${cell.attemptId} direct envelope cannot claim censoring`);
        }
        expectEqual(
          failures,
          cell.detectedPowerAcquisitionQualification,
          PINNED_DETECTED_POWER_ACQUISITION_QUALIFICATION,
          `causal envelope availability ${cell.attemptId} acquisition qualification`,
        );
        expectEqual(
          failures,
          cell.detectedPowerSelectionCondition,
          PINNED_DETECTED_POWER_SELECTION_CONDITION,
          `causal envelope availability ${cell.attemptId} automatic selection condition`,
        );
      } else {
        failures.push(`causal envelope availability ${cell.attemptId} captured row must declare one evidence disposition`);
      }
    } else if ('detectedPowerAcquisitionQualification' in cell
      || 'detectedPowerSelectionCondition' in cell) {
      failures.push(`causal envelope availability ${cell.attemptId} must not qualify an absent physical capture or selection condition`);
    }
    if ('envelopeFeatureUnavailableCode' in cell) {
      failures.push(`causal envelope availability ${cell.attemptId} must not publish an unavailable physical-capture code`);
    }
  }
  expectEqual(
    failures,
    new Set(availabilityAttemptIds).size,
    availabilityAttemptIds.length,
    'causal envelope availability unique attempt identities',
  );
  expectEqual(
    failures,
    availabilityPhysicalCaptureCount,
    physicalDetectedPowerCaptures,
    'causal envelope availability physical capture total',
  );
  expectEqual(
    failures,
    availabilityReceiptQualifiedCaptureCount,
    physicalDetectedPowerCaptures,
    'causal envelope availability receipt-qualified capture total',
  );
  expectEqual(
    failures,
    availabilityQualifiedEnvelopeCount,
    physicalEnvelopeCaptures,
    'causal envelope availability qualified-envelope total',
  );
  expectEqual(
    failures,
    availabilityCensoredSpectrumCount,
    physicalDetectedPowerCaptures - physicalEnvelopeCaptures,
    'causal envelope availability censored-spectrum total',
  );
  expectEqual(failures, booleanAt(report, 'matrix.samplingPartitionAudit.valid'), true, 'sampling-partition audit');
  for (const [path, label] of [
    ['matrix.samplingPartitionAudit.fittingCalibrationSeedOverlap', 'fitting/calibration seed overlap'],
    ['matrix.samplingPartitionAudit.validationFittingSeedOverlap', 'validation/fitting seed overlap'],
    ['matrix.samplingPartitionAudit.validationCalibrationSeedOverlap', 'validation/calibration seed overlap'],
    ['matrix.samplingPartitionAudit.validationFittingRbwOverlap', 'validation/fitting RBW overlap'],
    ['matrix.samplingPartitionAudit.validationCalibrationRbwOverlap', 'validation/calibration RBW overlap'],
  ]) {
    expectEqual(failures, arrayAt(report, path).length, 0, label);
  }
  expectEqual(failures, booleanAt(report, 'matrix.samplingPartitionAudit.validationTemporalPartitionDisjoint'), true, 'validation temporal partition');
  expectEqual(failures, arrayAt(report, 'matrix.samplingPartitionAudit.validationTemporalScheduleIdOverlap').length, 0, 'validation temporal schedule-ID overlap');
  const spectrumSourceLookOverlap = arrayAt(
    report,
    'matrix.samplingPartitionAudit.validationFitSpectrumSourceLookIndexOverlap',
  );
  const qualifiedEnvelopeSourceLookOverlap = arrayAt(
    report,
    'matrix.samplingPartitionAudit.validationFitQualifiedEnvelopeSourceLookIndexOverlap',
  );
  const aggregateSourceLookOverlap = arrayAt(
    report,
    'matrix.samplingPartitionAudit.validationFitTemporalSourceLookIndexOverlap',
  );
  expectEqual(failures, spectrumSourceLookOverlap.length, 0, 'validation spectrum source-look overlap');
  expectEqual(
    failures,
    qualifiedEnvelopeSourceLookOverlap.length,
    0,
    'validation qualified-envelope source-look overlap',
  );
  expectDeepEqual(
    failures,
    aggregateSourceLookOverlap,
    [...spectrumSourceLookOverlap, ...qualifiedEnvelopeSourceLookOverlap],
    'validation aggregate/branch source-look overlap reconciliation',
  );
  expectEqual(failures, booleanAt(report, 'matrix.tailCalibrationAudit.valid'), true, 'tail-calibration audit');
  expectEqual(failures, booleanAt(report, 'matrix.tailCalibrationAudit.matrixPinsValid'), true, 'tail-calibration matrix pins');
  expectEqual(failures, booleanAt(report, 'matrix.tailCalibrationAudit.productionAcquisitionRegimePinsValid'), true, 'production-acquisition pins');
  expectEqual(failures, booleanAt(report, 'matrix.tailCalibrationAudit.pinnedReleaseGateSourcePlanValid'), true, 'release-gate source-plan pins');
  expectDeepEqual(
    failures,
    valueAt(report, 'matrix.tailCalibrationAudit.pinnedSignalLabProductionAcquisitionRegime'),
    PINNED_PRODUCTION_ACQUISITION_REGIME,
    'complete production acquisition regime',
  );
  expectDeepEqual(
    failures,
    valueAt(report, 'matrix.temporalSchedules'),
    {
      consecutiveSpectrum: PINNED_VALIDATION_SPECTRUM_TEMPORAL_SCHEDULE,
      qualifiedEnvelope: PINNED_VALIDATION_QUALIFIED_ENVELOPE_TEMPORAL_SCHEDULE,
    },
    'held-out validation branch temporal schedules',
  );
  expectDeepEqual(
    failures,
    valueAt(report, 'matrix.detectedPowerSynthesisFilterPolicy'),
    PINNED_DETECTED_POWER_SYNTHESIS_FILTER_POLICY,
    'detected-power synthesis-filter policy',
  );
  for (const [path, expected] of [
    ['matrix.samplingPartitionAudit.modelFittingSeeds', PINNED_FITTING_SEEDS],
    ['matrix.samplingPartitionAudit.modelCalibrationSeeds', PINNED_CALIBRATION_SEEDS],
    ['matrix.samplingPartitionAudit.validationSeeds', PINNED_VALIDATION_SEEDS],
    ['matrix.samplingPartitionAudit.modelFittingRbwDivisors', PINNED_TRAINING_RBW_DIVISORS],
    ['matrix.samplingPartitionAudit.modelCalibrationRbwDivisors', PINNED_TRAINING_RBW_DIVISORS],
    ['matrix.samplingPartitionAudit.validationRbwDivisors', PINNED_VALIDATION_RBW_DIVISORS],
    ['matrix.tailCalibrationAudit.validatorOwnedMatrix.snrDb', PINNED_TRAINING_SNR_DB],
    ['matrix.tailCalibrationAudit.validatorOwnedMatrix.rbwDivisors', PINNED_TRAINING_RBW_DIVISORS],
    ['matrix.tailCalibrationAudit.validatorOwnedMatrix.seeds', PINNED_CALIBRATION_SEEDS],
  ]) {
    expectDeepEqual(failures, arrayAt(report, path), expected, path);
  }
  for (const [path, expected] of [
    ['matrix.tailCalibrationAudit.pinnedScoreUnit', PINNED_TAIL_POLICIES.scoreUnit],
    ['matrix.tailCalibrationAudit.modelScoreUnit', PINNED_TAIL_POLICIES.scoreUnit],
    ['matrix.tailCalibrationAudit.pinnedRepresentativeSelectionPolicy', PINNED_TAIL_POLICIES.representativeSelection],
    ['matrix.tailCalibrationAudit.modelRepresentativeSelectionPolicy', PINNED_TAIL_POLICIES.representativeSelection],
    ['matrix.tailCalibrationAudit.pinnedRepresentativeAggregationPolicy', PINNED_TAIL_POLICIES.representativeAggregation],
    ['matrix.tailCalibrationAudit.modelRepresentativeAggregationPolicy', PINNED_TAIL_POLICIES.representativeAggregation],
    ['matrix.tailCalibrationAudit.pinnedRuntimeInterpretationPolicy', PINNED_TAIL_POLICIES.runtimeInterpretation],
    ['matrix.tailCalibrationAudit.modelRuntimeInterpretationPolicy', PINNED_TAIL_POLICIES.runtimeInterpretation],
    ['matrix.tailCalibrationAudit.pinnedStatisticalInterpretation', PINNED_TAIL_POLICIES.statisticalInterpretation],
    ['matrix.tailCalibrationAudit.modelStatisticalInterpretation', PINNED_TAIL_POLICIES.statisticalInterpretation],
  ]) {
    expectEqual(failures, valueAt(report, path), expected, path);
  }
  expectEqual(
    failures,
    valueAt(report, 'matrix.tailCalibrationAudit.pinnedSignalLabProductionAcquisitionRegime.geometry.id'),
    PINNED_PRODUCTION_GEOMETRY_ID,
    'production acquisition geometry ID',
  );
  expectDeepEqual(
    failures,
    arrayAt(report, 'matrix.tailCalibrationAudit.pinnedSignalLabProductionAcquisitionRegime.temporalSchedulePairs')
      .map((pair) => pair.id),
    PINNED_PRODUCTION_TEMPORAL_SCHEDULE_PAIR_IDS,
    'production temporal schedule-pair IDs',
  );
  for (const path of [
    'matrix.samplingPartitionAudit.modelFittingAcquisitionRegimeIds',
    'matrix.samplingPartitionAudit.modelCalibrationAcquisitionRegimeIds',
    'matrix.tailCalibrationAudit.validatorOwnedMatrix.acquisitionRegimeIds',
  ]) {
    expectDeepEqual(failures, arrayAt(report, path), PINNED_ACQUISITION_REGIME_IDS, path);
  }
  const modelAttemptCountsByScenarioByView =
    generatedModel.trainingMatrix?.tailCalibrationAttemptCountsByScenarioByView;
  if (modelAttemptCountsByScenarioByView === null
    || typeof modelAttemptCountsByScenarioByView !== 'object'
    || Array.isArray(modelAttemptCountsByScenarioByView)) {
    throw new Error(`${MODEL_PATH} tailCalibrationAttemptCountsByScenarioByView must be an object`);
  }
  const reportedAttemptCountsByScenarioByView = objectAt(
    report,
    'matrix.tailCalibrationAudit.attemptCountsByScenarioByView',
  );
  const recomputedAttemptCountsByScenarioByView = objectAt(
    report,
    'matrix.tailCalibrationAudit.independentRecomputation.recomputedAttemptCountsByScenarioByView',
  );
  const expectedTailScenarioIds = generatedModel.classModels
    .filter((model) => model.id !== 'unknown-signal')
    .flatMap((model) => uniqueSourceScenarioIds(
      generatedComponentsForView(model, 'spectrum-only'),
    ))
    .sort();
  for (const [counts, label] of [
    [modelAttemptCountsByScenarioByView, 'model tail-calibration count scenario key set'],
    [reportedAttemptCountsByScenarioByView, 'reported tail-calibration count scenario key set'],
    [recomputedAttemptCountsByScenarioByView, 'recomputed tail-calibration count scenario key set'],
  ]) expectDeepEqual(failures, Object.keys(counts).sort(), expectedTailScenarioIds, label);
  const tailCalibrationAttemptCountsByView = Object.fromEntries(
    PINNED_TAIL_VIEWS.map((view) => [view, 0]),
  );
  for (const scenarioId of expectedTailScenarioIds) {
    const modelCounts = modelAttemptCountsByScenarioByView[scenarioId];
    const reportedCounts = reportedAttemptCountsByScenarioByView[scenarioId];
    const recomputedCounts = recomputedAttemptCountsByScenarioByView[scenarioId];
    for (const [counts, label] of [
      [modelCounts, 'model'],
      [reportedCounts, 'reported'],
      [recomputedCounts, 'independently recomputed'],
    ]) {
      if (counts === null || typeof counts !== 'object' || Array.isArray(counts)) {
        throw new Error(`${label} tail-calibration counts for ${scenarioId} must be an object`);
      }
      expectDeepEqual(
        failures,
        Object.keys(counts).sort(),
        [...PINNED_TAIL_VIEWS].sort(),
        `${label} tail-calibration ${scenarioId} view key set`,
      );
      for (const view of PINNED_TAIL_VIEWS) {
        const censoredEnvelope = view !== 'spectrum-only'
          && PINNED_FREQUENCY_AGILE_ENVELOPE_CENSORED_SCENARIO_IDS
            .includes(scenarioId);
        if (expectNonNegativeInteger(
          failures,
          counts[view],
          `${label} tail-calibration ${scenarioId}/${view} count`,
          { positive: !censoredEnvelope },
        ) && censoredEnvelope) {
          expectEqual(
            failures,
            counts[view],
            0,
            `${label} censored tail-calibration ${scenarioId}/${view} count`,
          );
        } else if (!censoredEnvelope
          && counts[view] < Math.floor(1 / PINNED_MINIMUM_KNOWN_SYNTHETIC_SUPPORT_RANK)) {
          failures.push(`${label} tail-calibration ${scenarioId}/${view} count cannot resolve the pinned minimum support rank`);
        }
      }
    }
    expectDeepEqual(
      failures,
      reportedCounts,
      modelCounts,
      `reported/model tail-calibration ${scenarioId} counts`,
    );
    expectDeepEqual(
      failures,
      recomputedCounts,
      modelCounts,
      `independently recomputed/model tail-calibration ${scenarioId} counts`,
    );
    for (const view of PINNED_TAIL_VIEWS) {
      tailCalibrationAttemptCountsByView[view] += modelCounts[view];
    }
  }
  for (const view of PINNED_TAIL_VIEWS) {
    expectEqual(
      failures,
      tailCalibrationAttemptCountsByView[view],
      generatedModel.trainingMatrix.causalSamplingAudit.tailCalibration
        .eligibleAttemptCountsByView[view],
      `tail-calibration scenario counts/causal-sampling ${view} eligible-attempt count`,
    );
  }
  for (const model of generatedModel.classModels.filter((candidate) => candidate.id !== 'unknown-signal')) {
    for (const view of PINNED_TAIL_VIEWS) {
      const expectedScoreCount = uniqueSourceScenarioIds(generatedComponentsForView(model, view)).reduce(
        (sum, sourceScenarioId) => sum + modelAttemptCountsByScenarioByView[sourceScenarioId][view],
        0,
      );
      const scores = model.tailCalibrationScoresByView?.[view];
      if (!Array.isArray(scores)) {
        failures.push(`generated ${model.id}/${view} tail-calibration scores must be an array`);
        continue;
      }
      expectEqual(
        failures,
        scores.length,
        expectedScoreCount,
        `generated ${model.id}/${view} score/count reconciliation`,
      );
      if (scores.some((score) => typeof score !== 'number' || !Number.isFinite(score))) {
        failures.push(`generated ${model.id}/${view} tail-calibration scores must be finite`);
      } else {
        if (scores.some((score) => score < 0 || score > 1)) {
          failures.push(`generated ${model.id}/${view} tail-calibration scores must be in [0, 1]`);
        }
        if (scores.some((score, index) => index > 0 && score < scores[index - 1])) {
          failures.push(`generated ${model.id}/${view} tail-calibration scores must be nondecreasing`);
        }
      }
    }
  }
  const tailScoreComparisonsByKey = new Map(arrayAt(
    report,
    'matrix.tailCalibrationAudit.independentRecomputation.scoreComparisons',
  ).map((comparison, index) => {
    if (comparison === null || typeof comparison !== 'object' || Array.isArray(comparison)
      || typeof comparison.classId !== 'string' || typeof comparison.view !== 'string') {
      throw new Error(`${REPORT_PATH} tail score comparison ${index} must publish classId and view`);
    }
    return [`${comparison.classId}/${comparison.view}`, comparison];
  }));
  for (const model of generatedModel.classModels.filter((candidate) => candidate.id !== 'unknown-signal')) {
    for (const view of PINNED_TAIL_VIEWS) {
      const scores = model.tailCalibrationScoresByView?.[view] ?? [];
      const comparison = tailScoreComparisonsByKey.get(`${model.id}/${view}`);
      if (!comparison) continue;
      expectEqual(
        failures,
        comparison.expectedCount,
        scores.length,
        `validator/model ${model.id}/${view} score count`,
      );
      expectEqual(
        failures,
        comparison.expectedSha256,
        createHash('sha256').update(JSON.stringify(scores)).digest('hex'),
        `validator/model ${model.id}/${view} score hash`,
      );
    }
  }
  expectEqual(
    failures,
    numberAt(report, 'matrix.tailCalibrationAudit.independentRecomputation.allOnlineAttemptCount', { integer: true }),
    tailCalibrationAttemptCountsByView['spectrum-only'],
    'all-online spectrum tail-calibration attempt count',
  );
  const tailSamplingAudit = generatedModel.trainingMatrix.causalSamplingAudit.tailCalibration;
  const recomputedTailSpectrumClockAudit = validateCausalClockAudit(
    report,
    'matrix.tailCalibrationAudit.independentRecomputation.runtimeBranchClockAudits.consecutiveSpectrum',
    failures,
    'consecutiveSpectrum',
    tailSamplingAudit.pairedNuisanceCellCount,
    tailSamplingAudit.runtimeBranches.consecutiveSpectrum.observationHorizonCounts,
    0,
  );
  const recomputedTailQualifiedEnvelopeClockAudit = validateCausalClockAudit(
    report,
    'matrix.tailCalibrationAudit.independentRecomputation.runtimeBranchClockAudits.qualifiedEnvelope',
    failures,
    'qualifiedEnvelope',
    tailSamplingAudit.pairedNuisanceCellCount,
    tailSamplingAudit.runtimeBranches.qualifiedEnvelope.observationHorizonCounts,
    causalSamplingMetrics.tailCalibrationPhysicalCaptureCount,
  );
  expectEqual(
    failures,
    recomputedTailQualifiedEnvelopeClockAudit.attemptsWithDetectedPowerCapture,
    causalSamplingMetrics.tailCalibrationPhysicalCaptureCount,
    'trainer/independent-validator tail-calibration physical capture count',
  );
  expectEqual(
    failures,
    recomputedTailSpectrumClockAudit.attempts
      * recomputedTailSpectrumClockAudit.spectrumAcquisitionCount.minimum
      <= tailSamplingAudit.runtimeBranches.consecutiveSpectrum.sourceClockEventCount
      && tailSamplingAudit.runtimeBranches.consecutiveSpectrum.sourceClockEventCount
        <= recomputedTailSpectrumClockAudit.attempts
          * recomputedTailSpectrumClockAudit.spectrumAcquisitionCount.maximum,
    true,
    'tail-calibration consecutive-spectrum source-clock bounds',
  );
  expectEqual(
    failures,
    recomputedTailQualifiedEnvelopeClockAudit.attempts
      * recomputedTailQualifiedEnvelopeClockAudit.spectrumAcquisitionCount.minimum
      + causalSamplingMetrics.tailCalibrationPhysicalCaptureCount
      <= tailSamplingAudit.runtimeBranches.qualifiedEnvelope.sourceClockEventCount
      && tailSamplingAudit.runtimeBranches.qualifiedEnvelope.sourceClockEventCount
        <= recomputedTailQualifiedEnvelopeClockAudit.attempts
          * recomputedTailQualifiedEnvelopeClockAudit.spectrumAcquisitionCount.maximum
          + causalSamplingMetrics.tailCalibrationPhysicalCaptureCount,
    true,
    'tail-calibration qualified-envelope source-clock bounds',
  );
  expectEqual(failures, manifestSha256, modelSha256, 'generated model manifest SHA-256');
  expectEqual(
    failures,
    declaredModelContentSha256,
    generatedModelContentSha256,
    'generated model content SHA-256',
  );
  expectEqual(
    failures,
    manifestContentSha256,
    generatedModelContentSha256,
    'generated model manifest content SHA-256',
  );
  for (const path of [
    'model.modelAssetSha256',
    'integrity.checkedInModelAssetSha256',
    'integrity.modelAssetManifestSha256',
  ]) {
    expectEqual(failures, valueAt(report, path), modelSha256, `${REPORT_PATH} ${path}`);
  }
  expectEqual(failures, valueAt(report, 'model.sourceCommit'), PINNED_SIGNAL_LAB_COMMIT, 'published SignalLab source commit');
  const checkedOutCorpusSourceManifest = valueAt(report, 'integrity.checkedOutCorpusSourceManifest');
  if (checkedOutCorpusSourceManifest === null || typeof checkedOutCorpusSourceManifest !== 'object') {
    throw new Error(`${REPORT_PATH} integrity.checkedOutCorpusSourceManifest must be an object`);
  }
  expectEqual(failures, checkedOutCorpusSourceManifest.schemaVersion, 1, 'corpus source manifest schema version');
  expectEqual(failures, checkedOutCorpusSourceManifest.hashAlgorithm, 'sha256', 'corpus source manifest hash algorithm');
  const sourceArtifacts = arrayAt(report, 'integrity.checkedOutCorpusSourceManifest.artifacts');
  const sourceArtifactPaths = sourceArtifacts.map((artifact, index) => {
    if (artifact === null || typeof artifact !== 'object' || typeof artifact.path !== 'string') {
      throw new Error(`${REPORT_PATH} integrity.checkedOutCorpusSourceManifest.artifacts.${index}.path must be a string`);
    }
    if (typeof artifact.sha256 !== 'string' || !/^[a-f0-9]{64}$/.test(artifact.sha256)) {
      throw new Error(`${REPORT_PATH} integrity.checkedOutCorpusSourceManifest.artifacts.${index}.sha256 must be SHA-256`);
    }
    return artifact.path;
  });
  expectEqual(
    failures,
    JSON.stringify(sourceArtifactPaths),
    JSON.stringify(PINNED_CORPUS_SOURCE_PATHS),
    'complete corpus source artifact closure',
  );
  const corpusSha256 = createHash('sha256')
    .update(JSON.stringify(checkedOutCorpusSourceManifest))
    .digest('hex');
  expectEqual(failures, corpusSha256, PINNED_CORPUS_SHA256, 'pinned corpus source-manifest SHA-256');
  expectEqual(failures, valueAt(report, 'integrity.checkedOutCorpusSha256'), corpusSha256, 'checked-out corpus source-manifest SHA-256');
  expectEqual(failures, valueAt(report, 'model.corpusSha256'), corpusSha256, 'model corpus source-manifest SHA-256');
  expectDeepEqual(failures, generatedModel.corpusSourceManifest, checkedOutCorpusSourceManifest, 'generated corpus source manifest');
  expectEqual(failures, generatedModel.corpusSha256, corpusSha256, 'generated corpus source-manifest SHA-256');

  const expectedRollingScenarioIds = generatedModel.classModels
    .filter((model) => model.id !== 'unknown-signal')
    .flatMap((model) => uniqueSourceScenarioIds(
      generatedComponentsForView(model, 'spectrum-only'),
    ));
  const metrics = {
    ...collectMetrics(report, failures, expectedRollingScenarioIds),
    ...causalSamplingMetrics,
  };
  expectEqual(
    failures,
    metrics.tailCalibrationAttemptCount,
    tailCalibrationAttemptCountsByView['spectrum-only'],
    'publication/causal model spectrum tail-calibration attempt count',
  );
  const formatted = formatMetrics(metrics, failures);
  const documents = new Map(PUBLICATION_PATHS.map((path) => [
    path,
    normalizeProse(visibleMarkdown(byPath.get(path).toString('utf8'))),
  ]));
  for (const path of [
    'docs/BAYESIAN_DETECTION_CLASSIFICATION_RESEARCH.md',
    'docs/SIGNALLAB_EMSO_CLASSIFIER_CONTRACT.md',
  ]) {
    const document = documents.get(path);
    for (const artifact of sourceArtifacts) {
      const hashCount = occurrenceCount(document, artifact.sha256);
      if (hashCount !== 1) {
        failures.push(`${path} must publish ${artifact.path} SHA-256 ${artifact.sha256} exactly once (found ${hashCount})`);
      }
    }
  }
  verifyPublicationProse(documents, modelSha256, corpusSha256, metrics, formatted, failures);

  if (failures.length > 0) {
    throw new Error(`classifier publication is stale or internally inconsistent:\n- ${failures.join('\n- ')}`);
  }

  console.log(JSON.stringify({
    status: 'verified',
    modelAssetSha256: modelSha256,
    report: REPORT_PATH,
    publications: PUBLICATION_PATHS,
    metrics: {
      attempts: metrics.attempts,
      admitted: metrics.admitted,
      representatives: metrics.representatives,
      hierarchicalAccuracy: metrics.hierarchicalAccuracy,
      knownCoverage: metrics.knownCoverage,
      exactEquivalencePairs: metrics.exactEquivalencePairs,
      exactEquivalenceDiscrepancies: metrics.discrepancies,
    },
  }, null, 2));
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
