import type { ClassLikelihoodModel } from '../../Atom-Atomizer/packages/analysis/src/bayesian-predictive.js';

export const OBSERVABLE_EVIDENCE_VIEWS = [
  'spectrum-only',
  'envelope-untimed',
  'envelope-timed',
] as const;

export const OBSERVABLE_LIKELIHOOD_COMPONENT_DECOMPOSITION_POLICY = Object.freeze({
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

/**
 * A zero-span receiver stays fixed at the selected physical member frequency.
 * For a frequency-agile regional association, the returned detected-power
 * series is therefore a censored member-frequency observation, not an
 * envelope of the multi-look regional evidence represented by the classifier
 * row. The capture remains useful acquisition audit data, but it cannot enter
 * that representative's likelihood.
 */
export const OBSERVABLE_EVIDENCE_CENSORING_POLICY = Object.freeze({
  id: 'frequency-agile-fixed-tune-envelope-censoring-v1',
  associationMode: 'frequency-agile-2g4-activity',
  runtimeCapturePolicy: 'validate-receipt-and-capture-before-censoring-v1',
  classifierEvidencePolicy: 'spectrum-only-no-detected-power-envelope-v1',
  unsupportedModelViewPolicy: 'exact-empty-components-and-calibration-v1',
} as const);
export type ObservableEvidenceView = typeof OBSERVABLE_EVIDENCE_VIEWS[number];

export function observableModelView(
  observation: Readonly<{ values: Readonly<Record<string, number>> }>,
): ObservableEvidenceView {
  return observation.values['envelope.logTransitionRateHz'] !== undefined
    ? 'envelope-timed'
    : Object.keys(observation.values).some((name) => name.startsWith('envelope.'))
      ? 'envelope-untimed'
      : 'spectrum-only';
}

export function observableModelComponents(
  model: Pick<ClassLikelihoodModel, 'id' | 'componentsByView'>,
  view: ObservableEvidenceView,
) {
  const components = model.componentsByView?.[view];
  if (components === undefined) {
    throw new Error(`Observable class ${model.id} has no ${view} likelihood components`);
  }
  const structurallySupported = model.id !== 'bluetooth-like'
    || observableClassSupportsEvidenceView('bluetooth-like', view);
  if (structurallySupported ? components.length === 0 : components.length !== 0) {
    throw new Error(structurallySupported
      ? `Observable class ${model.id} has no ${view} likelihood components`
      : `Observable class ${model.id} must have an empty ${view} likelihood population`);
  }
  return components;
}

export const OBSERVABLE_LEAF_CLASSES = [
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
  'unknown-signal',
] as const;

export type ObservableLeafClass = typeof OBSERVABLE_LEAF_CLASSES[number];
export type ObservableDecisionClass = Exclude<ObservableLeafClass, 'unknown-signal'>
  | 'cellular-ofdm-ambiguous'
  | 'lte-like'
  | 'nr-like'
  | 'wifi-like';

/** Exact structural view support declared by the fitted model contract. */
export function observableClassSupportsEvidenceView(
  classId: ObservableLeafClass,
  view: ObservableEvidenceView,
): boolean {
  return classId !== 'bluetooth-like' || view === 'spectrum-only';
}

type ObservableViewCounts = Readonly<Record<ObservableEvidenceView, number>>;

interface ObservableConsecutiveSpectrumSamplingAudit {
  readonly detectedPowerCapturePolicyId: 'no-automatic-detected-power-capture-v1';
  readonly attemptCount: number;
  readonly attemptsWithAnyRepresentative: number;
  readonly attemptsWithFitEligibleRepresentative: number;
  readonly onlineSpectrumRepresentativeCount: number;
  readonly fitEligibleRepresentativeCount: number;
  readonly fitIneligibleRepresentativeCount: number;
  readonly provenanceUnavailableWindowCount: number;
  readonly spectrumAcquisitionCount: number;
  readonly physicalDetectedPowerCaptureCount: 0;
  readonly postCaptureProvenanceUnavailableWindowCount: 0;
  readonly detectedPowerCaptureSampleCount: 0;
  readonly censoredFrequencyAgileFixedTuneCaptureCount: 0;
  readonly sourceClockEventCount: number;
  readonly multiRepresentativeAttemptCount: number;
  readonly maximumRepresentativesPerAttempt: number;
  readonly observationHorizonCounts: Readonly<Record<string, number>>;
  readonly observationOpportunityCounts: Readonly<Record<string, number>>;
}

interface ObservableQualifiedEnvelopeSamplingAudit {
  readonly detectedPowerCapturePolicyId:
    | 'capture-once-after-first-runtime-admitted-strongest-current-target-v2'
    | 'capture-once-after-rank-0-integrated-excess-current-target-runtime-admission-v3';
  readonly attemptCount: number;
  readonly receiptVerifiedDetectedPowerCaptureSampleCount: number;
  readonly capturedEnvelopeRepresentativeCount: number;
  readonly censoredFrequencyAgileFixedTuneCaptureCount: number;
  readonly fitEligibleTimedCapturedEnvelopeRepresentativeCount: number;
  readonly fitEligibleUntimedCapturedEnvelopeRepresentativeCount: number;
  readonly provenanceUnavailableWindowCount: number;
  readonly preCaptureProvenanceUnavailableWindowCount: number;
  readonly postCaptureProvenanceUnavailableWindowCount: number;
  readonly spectrumAcquisitionCount: number;
  readonly physicalDetectedPowerCaptureCount: number;
  readonly attemptsWithoutDetectedPowerCapture: number;
  readonly sourceClockEventCount: number;
  readonly observationHorizonCounts: Readonly<Record<string, number>>;
}

interface ObservableSamplingPartitionAudit {
  readonly pairedNuisanceCellCount: number;
  readonly fitEligibleRepresentativeCountsByView: ObservableViewCounts;
  readonly eligibleAttemptCountsByView: ObservableViewCounts;
  readonly runtimeBranches: {
    readonly consecutiveSpectrum: ObservableConsecutiveSpectrumSamplingAudit;
    readonly qualifiedEnvelope: ObservableQualifiedEnvelopeSamplingAudit;
  };
}

interface ObservableUnavailableAttempt {
  readonly attemptId: string;
  readonly unavailableWindowCount: number;
}

interface ObservableBranchUnavailableAttempts {
  readonly consecutiveSpectrum: readonly ObservableUnavailableAttempt[];
  readonly qualifiedEnvelope: readonly ObservableUnavailableAttempt[];
}

interface ObservableBranchTraceHashes {
  readonly consecutiveSpectrumSha256: string;
  readonly qualifiedEnvelopeSha256: string;
}

export interface ObservableClassifierModelAsset {
  id: string;
  corpusVersion: string;
  sourceCommit: string;
  corpusSourceManifest: {
    schemaVersion: 1;
    hashAlgorithm: 'sha256';
    artifacts: readonly {
      /** Path relative to the SignalLab repository root. */
      path: string;
      sha256: string;
    }[];
  };
  /** SHA-256 of the canonical JSON serialization of corpusSourceManifest. */
  corpusSha256: string;
  preprocessing: string;
  priorId: string;
  calibrationId: string;
  generatedAt: string;
  dimensions: readonly string[];
  trainingMatrix: {
    /** SHA-256 identity of the exact immutable bundled worker closure used for fitting and calibration. */
    attemptSamplingWorkerRuntimeSha256: string;
    /** Exact runtime identity admitted by the repository Node pin for all fitting and calibration. */
    trainingRuntimeIdentity: {
      policyId: 'exact-repository-node-version-v1';
      nodeVersion: string;
      v8Version: string;
    };
    snrDb: readonly number[];
    rbwDivisors: readonly number[];
    seeds: readonly number[];
    /** Complete fitted acquisition cells, including named production regimes that are not honest global RBW divisors. */
    fittingAcquisitionRegimeIds?: readonly string[];
    /**
     * The production SignalLab sweep geometry and independent source-clock
     * branches included in both component fitting and independent-seed tail
     * calibration. The no-auto-capture spectrum session and qualified first-
     * admitted envelope session are separate deployed acquisition populations.
     */
    signalLabProductionAcquisitionRegime?: {
      id:
        | 'signal-lab-recommended-span-grid-with-independent-production-branch-source-clocks-v4'
        | 'signal-lab-recommended-span-grid-with-independent-production-branch-source-clocks-v5';
      geometry: {
        id: 'signal-lab-recommended-span-450-point-grid-v1';
        sourceKind: 'signal-lab';
        kind: 'recommended-span-inclusive-grid';
        sweepPoints: 450;
        spanPolicy: 'canonical-recommended-span-v1';
        resolutionScalePolicy: 'recommended-span-divided-by-points-minus-one-v1';
      };
      branchPolicy:
        | 'independent-no-auto-spectrum-and-qualified-first-admitted-envelope-sessions-v1'
        | 'independent-no-auto-spectrum-and-qualified-rank-0-integrated-excess-envelope-sessions-v2';
      sourceClocks: {
        spectrum: {
          id: 'shared-monotonic-source-clock-v1';
          acquisitionIndexPolicy: 'one-look-index-per-physical-acquisition-v1';
          detectedPowerCapturePolicy: 'no-automatic-detected-power-capture-v1';
        };
        qualifiedEnvelope: {
          id: 'shared-monotonic-source-clock-v1';
          acquisitionIndexPolicy: 'one-look-index-per-physical-acquisition-v1';
          detectedPowerCapturePolicy:
            | 'capture-once-after-first-runtime-admitted-strongest-current-target-v2'
            | 'capture-once-after-rank-0-integrated-excess-current-target-runtime-admission-v3';
          captureTargetSelectionPolicy:
            | 'preferred-then-strongest-current-physical-or-qualified-agile-member-target-v3'
            | 'preferred-then-current-source-sweep-integrated-excess-power-physical-or-qualified-agile-member-target-v4';
          postCaptureSpectrumPolicy: 'continue-at-next-shared-look-index-v1';
        };
      };
      spectrumReleaseGateSourcePlan: readonly {
        profileId: string;
        profileOrdinal: number;
        sourceLookIndexOffset: number;
        spectrumOpportunities: number;
        automaticDetectedPowerCaptures: 0;
      }[];
      qualifiedEnvelopeReleaseGateSourcePlan: readonly {
        profileId: string;
        profileOrdinal: number;
        sourceLookIndexOffset: number;
        spectrumOpportunities: number;
        admittedDetectedPowerCaptures: 1;
      }[];
      temporalSchedulePairs: readonly {
        id: string;
        sourcePlanProfileId: string;
        spectrumTemporalSchedule: {
          id: string;
          sourcePlanProfileId: string;
          sourceLookIndexOffset: number;
          sourcePlanSpectrumOpportunities: number;
        };
        qualifiedEnvelopeTemporalSchedule: {
          id: string;
          sourcePlanProfileId: string;
          sourceLookIndexOffset: number;
          sourcePlanSpectrumOpportunities: number;
        };
      }[];
      componentFitIncluded: true;
      tailCalibrationIncluded: true;
    };
    /**
     * Generator-only detected-power filter geometry used by the synthetic
     * reference matrix. It is never projected as measured RBW evidence.
     */
    detectedPowerSynthesisFilterPolicy?: {
      id: 'explicit-generator-filter-width-by-acquisition-regime-v1';
      divisorAcquisitionRegimes: 'match-swept-spectrum-actual-rbw-nuisance-v1';
      signalLabProductionAcquisitionRegimes: 'fixed-generator-internal-width-v1';
      signalLabProductionSynthesisFilterWidthHz: 100_000;
      measurementActualRbwQualification: 'unavailable';
    };
    productionAcquisitionRegimeHighSnrSeedCoveragePolicy?: {
      id: 'branch-conditional-production-regime-presence-v2';
      spectrumOnly: {
        minimumDistinctObservationDomainEligibleSeedsPerHighSnrCell: number;
      };
      qualifiedEnvelope: {
        minimumDistinctPhysicalCaptureSeedsPerHighSnrCell: number;
        observationDomainEligibilityPolicy: 'pooled-by-scenario-and-view-after-causal-capture-v1';
        outOfDomainCapturePolicy: 'honest-abstention-excluded-from-envelope-likelihood-v1';
      };
      globalCoveragePolicy: 'all-seeds-at-one-or-more-regimes-except-declared-sparse-asynchronous-scenarios-v1';
    };
    classificationSweeps?: number;
    observationOpportunityHorizons?: {
      standard: number;
      fullBand2g4: number;
    };
    /** @deprecated Present only on pre-v5 generated assets. */
    observationOpportunitiesPerExample?: number;
    /** @deprecated Present only on pre-v4 generated assets. */
    sweepsPerExample?: number;
    tailCalibrationSeeds?: readonly number[];
    tailCalibrationRbwDivisors?: readonly number[];
    /** Complete independent-seed calibration cells, including named production regimes. */
    tailCalibrationAcquisitionRegimeIds?: readonly string[];
    /** Each view contributes at most one score per distinct causal acquisition cell; this does not assert statistical independence. */
    tailCalibrationScoreUnit?: 'one-score-per-fit-eligible-acquisition-attempt-v1'
      | 'one-score-per-observation-domain-eligible-acquisition-attempt-v2'
      | 'one-causal-acquisition-attempt-score-per-evidence-view-v3'
      | 'one-independent-branch-acquisition-attempt-score-per-evidence-view-v4';
    /** Spectrum and envelope views follow their distinct live acquisition populations. */
    tailCalibrationRepresentativeSelectionPolicy?: 'online-all-ready-representatives-v1'
      | 'all-runtime-admitted-spectrum-representatives-and-sole-live-envelope-representative-v2'
      | 'consecutive-spectrum-all-runtime-representatives-and-independent-qualified-envelope-sole-capture-v3'
      | 'consecutive-spectrum-all-runtime-representatives-and-independent-qualified-envelope-sole-capture-v4'
      | 'consecutive-spectrum-all-runtime-representatives-and-independent-integrated-excess-rank-0-envelope-sole-capture-v5';
    /** Spectrum representatives collapse to an attempt minimum; the envelope views use the one physical live capture. */
    tailCalibrationRepresentativeAggregationPolicy?: 'minimum-support-across-fit-eligible-first-ready-representatives-v1'
      | 'minimum-support-across-fit-eligible-online-representatives-v2'
      | 'minimum-support-across-observation-domain-eligible-online-representatives-v3'
      | 'spectrum-minimum-envelope-sole-capture-v4'
      | 'consecutive-spectrum-branch-minimum-qualified-envelope-branch-sole-capture-v5';
    /** Spectrum ranks dominate their attempt minimum; an envelope rank is calibrated against the sole live capture population. */
    tailCalibrationRuntimeInterpretationPolicy?: 'single-representative-rank-dominates-attempt-min-rank-v1'
      | 'spectrum-single-rank-dominates-attempt-min-envelope-rank-is-sole-capture-v2'
      | 'spectrum-member-dominates-independent-branch-attempt-min-envelope-is-independent-sole-capture-v3';
    /** Fixed synthetic nuisance grids are reference data, not exchangeable operational calibration samples. */
    tailCalibrationStatisticalInterpretation?: 'empirical-synthetic-reference-only-no-exchangeability-or-coverage-guarantee-v1';
    /** Observation-domain-eligible acquisition attempts contributing one score each, by canonical scenario. */
    tailCalibrationAttemptCountsByScenario?: Readonly<Record<string, number>>;
    /** Causal acquisition attempts contributing a score, split by evidence view and canonical scenario. */
    tailCalibrationAttemptCountsByScenarioByView?: Readonly<Record<string, Readonly<Record<
      'spectrum-only' | 'envelope-untimed' | 'envelope-timed',
      number
    >>>>;
    /** Full-view causal envelope observations used to fit each canonical component. */
    fittingCapturedEnvelopeCountsByScenario?: Readonly<Record<string, number>>;
    /** Runtime-event likelihood populations, independently counted for every evidence view. */
    fittingRepresentativeCountsByScenarioByView?: Readonly<Record<string, Readonly<Record<
      'spectrum-only' | 'envelope-untimed' | 'envelope-timed',
      number
    >>>>;
    likelihoodPopulationPolicy?: 'view-matched-runtime-event-populations-v1'
      | 'independent-branch-view-matched-runtime-event-populations-v2'
      | 'independent-branch-view-matched-runtime-event-populations-v3';
    likelihoodComponentDecompositionPolicy?: {
      id: 'scenario-components-with-three-shared-covariance-csma-activity-modes-v1';
      scenarioWeighting: 'equal-fitted-scenario-weight-within-class-v1';
      ordinaryScenarioModel: 'one-student-t-component-v1';
      csmaEnvelopeModel: 'csma-bursts';
      csmaPartitionFeature: 'spectrum.powerVariationDb';
      csmaModeCount: 3;
      minimumModeFitSampleCount: 3;
      csmaClustering: 'deterministic-one-dimensional-lloyd-min-median-max-v1';
      csmaModeWeighting: 'empirical-fit-event-frequency-within-scenario-and-view-v1';
      csmaCovariance: 'shared-within-mode-pooled-covariance-with-0.35-off-diagonal-retention-v1';
    };
    acquisitionBranchPolicy?:
      | 'independent-no-auto-spectrum-and-qualified-first-admitted-envelope-sessions-v1'
      | 'independent-no-auto-spectrum-and-qualified-rank-0-integrated-excess-envelope-sessions-v2';
    frequencyAgileFixedTuneEnvelopeCensoringPolicy?:
      typeof OBSERVABLE_EVIDENCE_CENSORING_POLICY;
    /** Valid fixed-tune captures deliberately excluded from classifier evidence, by partition and canonical scenario. */
    censoredFrequencyAgileFixedTuneCaptureCountsByScenario?: {
      fitting: Readonly<Record<string, number>>;
      tailCalibration: Readonly<Record<string, number>>;
    };
    detectedPowerAcquisitionQualification?:
      | 'receipt-verified-provenance-bound-first-runtime-admitted-strongest-current-physical-or-agile-member-single-capture-v4'
      | 'receipt-verified-provenance-bound-runtime-admitted-physical-capture-v5';
    /** Exact automatic target condition represented by fitted/calibration envelope samples. */
    detectedPowerSelectionCondition?:
      'automatic-current-source-sweep-integrated-excess-rank-0';
    /** Content-addressed trainer audit; representative counts are never supplied only as prose. */
    causalSamplingAudit?: {
      schemaVersion: 3;
      fitting: ObservableSamplingPartitionAudit;
      tailCalibration: ObservableSamplingPartitionAudit;
      provenanceUnavailableAttemptPolicy:
        'branch-attributed-exact-attempt-cell-counts-v2';
      provenanceUnavailableAttempts: {
        fitting: ObservableBranchUnavailableAttempts;
        tailCalibration: ObservableBranchUnavailableAttempts;
      };
      attributedSourceClockTraceAudit: {
        hashAlgorithm: 'sha256';
        serialization:
          'canonical-attempt-id-branch-attributed-trace-and-capture-disposition-digest-v3';
        fitting: ObservableBranchTraceHashes;
        tailCalibration: ObservableBranchTraceHashes;
      };
    };
    detectorConditionedFitMisses?: readonly string[];
    detectorConditionedCalibrationMisses?: readonly string[];
    postCaptureUnavailableFitAttempts?: readonly string[];
    postCaptureUnavailableCalibrationAttempts?: readonly string[];
    fitEligibilityExcludedFitAttempts?: readonly string[];
    fitEligibilityExcludedCalibrationAttempts?: readonly string[];
    scenarioExcludedFromComponentFitIds?: readonly string[];
    /**
     * Unknown-source scenarios that are exactly equivalent to one or more
     * fitted observable classes. They are validation nulls only: fitting them
     * as unknown components would duplicate a likelihood under another label.
     */
    exactObservableEquivalenceNullScenarioIds?: readonly string[];
    /** Known-class scenarios retained only to test/report acquisition non-admission. */
    knownAcquisitionValidationOnlyScenarioIds?: readonly string[];
    /** Older policies remain readable only so the trainer can replace a checked-in asset; runtime asserts v9. */
    selectionPolicy?: 'endpoint-active-representative-v1' | 'endpoint-active-all-representatives-v2' | 'online-first-ready-all-representatives-v3'
      | 'causal-first-admitted-single-envelope-all-online-spectrum-v4'
      | 'independent-consecutive-spectrum-and-strongest-first-admission-qualified-envelope-branches-v6'
      | 'independent-consecutive-spectrum-and-strongest-first-admission-qualified-envelope-branches-v7'
      | 'independent-consecutive-spectrum-and-strongest-first-admission-qualified-envelope-branches-v8'
      | 'independent-consecutive-spectrum-and-integrated-excess-rank-0-runtime-admission-qualified-envelope-branches-v9';
    representativeWeightingPolicy?: 'equal-weight-per-endpoint-production-representative-v1' | 'equal-weight-per-first-ready-production-representative-v2'
      | 'equal-weight-per-causal-live-envelope-acquisition-attempt-v3'
      | 'view-matched-spectrum-event-envelope-causal-attempt-weighting-v4';
    representativeEligibilityPolicy?: 'bluetooth-components-require-qualified-agile-association-v1'
      | 'observation-qualified-known-representatives-v2'
      | 'runtime-domain-qualified-known-representatives-v3'
      | 'observation-only-hypothesis-domain-v4'
      | 'observation-only-hypothesis-domain-v5';
  };
  classModels: readonly (ClassLikelihoodModel & { id: ObservableLeafClass })[];
}

export const observableClassDefinitions: Readonly<Record<ObservableDecisionClass, { label: string; family: string; claim: string }>> = {
  'cw-like': { label: 'CW-like carrier', family: 'analog', claim: 'RBW-limited stable carrier evidence' },
  'am-dsb-full-carrier-like': { label: 'DSB full-carrier AM-like', family: 'analog', claim: 'Carrier, mirrored sideband and envelope evidence' },
  'fm-angle-modulated-like': { label: 'FM / angle-modulated-like', family: 'analog', claim: 'Symmetric frequency-spread evidence; not phase or protocol identity' },
  'gsm-like': { label: 'GSM / GERAN-like', family: 'cellular', claim: '200 kHz GERAN-compatible spectral/timing evidence' },
  'lte-fdd-like': { label: 'LTE FDD-like', family: 'cellular', claim: 'LTE-compatible width and FDD-context evidence' },
  'lte-tdd-like': { label: 'LTE TDD-like', family: 'cellular', claim: 'LTE-compatible width and TDD-context/envelope evidence' },
  'nr-fdd-like': { label: '5G NR FDD-like', family: 'cellular', claim: 'NR-compatible width and FDD-context evidence' },
  'nr-tdd-like': { label: '5G NR TDD-like', family: 'cellular', claim: 'NR-compatible width and TDD-context/envelope evidence' },
  'cellular-ofdm-ambiguous': { label: 'OFDM-shaped · LTE/NR-compatible', family: 'ofdm', claim: 'Wide OFDM morphology; generic OFDM, LTE and NR remain observationally equivalent' },
  'lte-like': { label: 'LTE-compatible OFDM · duplex ambiguous', family: 'ofdm', claim: 'LTE-shaped evidence without protocol or FDD/TDD identity' },
  'nr-like': { label: '5G NR-compatible OFDM · duplex ambiguous', family: 'ofdm', claim: 'NR-shaped evidence without protocol or FDD/TDD identity' },
  'wifi-hr-dsss-like': { label: 'Wi-Fi HR-DSSS-like', family: 'wifi', claim: 'DSSS/CCK-like 802.11 channel evidence' },
  'wifi-ofdm-like': { label: 'Wi-Fi OFDM-like', family: 'wifi', claim: '802.11 OFDM width/traffic evidence; generation unresolved' },
  'wifi-like': { label: '802.11-compatible channel morphology · PHY unresolved', family: 'wifi', claim: 'Scalar channel morphology compatible with 802.11; proprietary DSSS/OFDM and protocol identity remain unresolved' },
  'bluetooth-like': { label: '2.4 GHz agile activity · Bluetooth-compatible', family: 'bluetooth', claim: 'Bluetooth-compatible activity transitions without Classic/LE, protocol, or emitter identity resolution' },
};
