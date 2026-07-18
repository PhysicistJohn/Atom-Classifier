import { describe, expect, it } from 'vitest';
import {
  BAYESIAN_WAVEFORM_MODEL,
  BayesianWaveformClassifier,
  empiricalSyntheticSupportRank,
} from './bayesian-waveform-classifier.js';
import { BAYESIAN_OBSERVABLE_MODEL } from './models/bayesian-observable.generated.js';
import {
  OBSERVABLE_EVIDENCE_CENSORING_POLICY,
  observableClassSupportsEvidenceView,
  observableModelComponents,
} from './observable-classifier-model.js';
import { SIGNAL_LAB_PRODUCTION_ACQUISITION_REGIME_METADATA } from '../../TinySA/packages/analysis/src/observable-training-acquisition-geometry.js';

describe('synthetic support rank contract', () => {
  it('computes a smoothed lower-tail empirical rank with deterministic tie handling', () => {
    const reference = [0.1, 0.2, 0.2, 0.8];

    expect(empiricalSyntheticSupportRank(0.05, reference)).toBe(1 / 5);
    expect(empiricalSyntheticSupportRank(0.2, reference)).toBe(4 / 5);
    expect(empiricalSyntheticSupportRank(1, reference)).toBe(1);
  });

  it('keeps every member rank at least as large as its attempt-minimum rank', () => {
    const reference = [0.03, 0.08, 0.12, 0.4, 0.9];
    const representativeSupports = [0.07, 0.22, 0.81];
    const attemptMinimumRank = empiricalSyntheticSupportRank(Math.min(...representativeSupports), reference);

    for (const support of representativeSupports) {
      expect(empiricalSyntheticSupportRank(support, reference)).toBeGreaterThanOrEqual(attemptMinimumRank);
    }
  });

  it('treats 0.025 as a discrete engineering cutoff rather than a coverage claim', () => {
    expect(BAYESIAN_WAVEFORM_MODEL.minimumKnownSyntheticSupportRank).toBe(0.025);
    expect(empiricalSyntheticSupportRank(0, Array<number>(39).fill(0.1))).toBe(0.025);
    expect(empiricalSyntheticSupportRank(0, Array<number>(40).fill(0.1))).toBeLessThan(0.025);
  });

  it('rejects malformed reference ranks instead of silently changing their meaning', () => {
    expect(() => empiricalSyntheticSupportRank(Number.NaN, [0.1])).toThrow(/within \[0, 1\]/);
    expect(() => empiricalSyntheticSupportRank(0.1, [])).toThrow(/must not be empty/);
    expect(() => empiricalSyntheticSupportRank(0.1, [0.2, 0.1])).toThrow(/must be sorted/);
    expect(() => empiricalSyntheticSupportRank(0.1, [0.1, 1.1])).toThrow(/must be sorted/);
  });

  it('pins the non-conformal statistical interpretation in the generated asset', () => {
    expect(BAYESIAN_OBSERVABLE_MODEL.calibrationId).toBe(
      'synthetic-independent-branch-view-matched-causal-acquisition-support-rank-detector-conditioned-physical-uncalibrated-v19',
    );
    expect(BAYESIAN_OBSERVABLE_MODEL.calibrationId).not.toContain('conformal');
    expect(BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.tailCalibrationRepresentativeSelectionPolicy)
      .toBe('consecutive-spectrum-all-runtime-representatives-and-independent-qualified-envelope-sole-capture-v4');
    expect(BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.tailCalibrationScoreUnit)
      .toBe('one-independent-branch-acquisition-attempt-score-per-evidence-view-v4');
    expect(BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.tailCalibrationRepresentativeAggregationPolicy)
      .toBe('consecutive-spectrum-branch-minimum-qualified-envelope-branch-sole-capture-v5');
    expect(BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.tailCalibrationRuntimeInterpretationPolicy)
      .toBe('spectrum-member-dominates-independent-branch-attempt-min-envelope-is-independent-sole-capture-v3');
    expect(BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.tailCalibrationStatisticalInterpretation)
      .toBe('empirical-synthetic-reference-only-no-exchangeability-or-coverage-guarantee-v1');
    expect(BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.representativeEligibilityPolicy)
      .toBe('observation-only-hypothesis-domain-v5');
    expect(BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.detectedPowerAcquisitionQualification)
      .toBe('receipt-verified-provenance-bound-first-runtime-admitted-strongest-current-physical-or-agile-member-single-capture-v4');
    expect(BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.acquisitionBranchPolicy)
      .toBe('independent-no-auto-spectrum-and-qualified-first-admitted-envelope-sessions-v1');
    expect(BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.selectionPolicy)
      .toBe('independent-consecutive-spectrum-and-strongest-first-admission-qualified-envelope-branches-v8');
    expect(BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.likelihoodPopulationPolicy)
      .toBe('independent-branch-view-matched-runtime-event-populations-v3');
    expect(BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.frequencyAgileFixedTuneEnvelopeCensoringPolicy)
      .toEqual(OBSERVABLE_EVIDENCE_CENSORING_POLICY);
    expect(BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.censoredFrequencyAgileFixedTuneCaptureCountsByScenario)
      .toEqual({
        fitting: expect.objectContaining({
          'bluetooth-classic-connected': expect.any(Number),
          'bluetooth-le-advertising': expect.any(Number),
        }),
        tailCalibration: expect.objectContaining({
          'bluetooth-classic-connected': expect.any(Number),
          'bluetooth-le-advertising': expect.any(Number),
        }),
      });
    expect(BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.trainingRuntimeIdentity).toEqual({
      policyId: 'exact-repository-node-version-v1',
      nodeVersion: '22.23.1',
      v8Version: '12.4.254.21-node.56',
    });
    expect(BAYESIAN_OBSERVABLE_MODEL.id).toBe('bayesian-observable-equivalence-v8');
    expect(BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.signalLabProductionAcquisitionRegime)
      .toEqual(SIGNAL_LAB_PRODUCTION_ACQUISITION_REGIME_METADATA);
    expect(BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.detectedPowerSynthesisFilterPolicy).toEqual({
      id: 'explicit-generator-filter-width-by-acquisition-regime-v1',
      divisorAcquisitionRegimes: 'match-swept-spectrum-actual-rbw-nuisance-v1',
      signalLabProductionAcquisitionRegimes: 'fixed-generator-internal-width-v1',
      signalLabProductionSynthesisFilterWidthHz: 100_000,
      measurementActualRbwQualification: 'unavailable',
    });
    expect(BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.productionAcquisitionRegimeHighSnrSeedCoveragePolicy)
      .toEqual({
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
      });
    const audit = BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.causalSamplingAudit;
    expect(audit).toBeDefined();
    if (!audit) throw new Error('Generated model omitted its causal sampling audit');
    expect(audit.schemaVersion).toBe(3);
    expect(audit.attributedSourceClockTraceAudit.serialization)
      .toBe('canonical-attempt-id-branch-attributed-trace-and-capture-disposition-digest-v3');
    for (const partition of [audit.fitting, audit.tailCalibration]) {
      const spectrum = partition.runtimeBranches.consecutiveSpectrum;
      const envelope = partition.runtimeBranches.qualifiedEnvelope;
      expect(spectrum.physicalDetectedPowerCaptureCount).toBe(0);
      expect(spectrum.postCaptureProvenanceUnavailableWindowCount).toBe(0);
      expect(spectrum.detectedPowerCaptureSampleCount).toBe(0);
      expect(spectrum.censoredFrequencyAgileFixedTuneCaptureCount).toBe(0);
      expect(spectrum.sourceClockEventCount).toBe(spectrum.spectrumAcquisitionCount);
      expect(envelope.preCaptureProvenanceUnavailableWindowCount
        + envelope.postCaptureProvenanceUnavailableWindowCount)
        .toBe(envelope.provenanceUnavailableWindowCount);
      expect(envelope.sourceClockEventCount)
        .toBe(envelope.spectrumAcquisitionCount + envelope.physicalDetectedPowerCaptureCount);
      expect(envelope.physicalDetectedPowerCaptureCount)
        .toBe(envelope.receiptVerifiedDetectedPowerCaptureSampleCount
          + envelope.postCaptureProvenanceUnavailableWindowCount);
      expect(envelope.receiptVerifiedDetectedPowerCaptureSampleCount)
        .toBe(envelope.capturedEnvelopeRepresentativeCount
          + envelope.censoredFrequencyAgileFixedTuneCaptureCount);
      expect(envelope.capturedEnvelopeRepresentativeCount).toBeGreaterThan(0);
      expect(envelope.censoredFrequencyAgileFixedTuneCaptureCount).toBeGreaterThan(0);
      expect(partition.pairedNuisanceCellCount).toBe(spectrum.attemptCount);
      expect(partition.pairedNuisanceCellCount).toBe(envelope.attemptCount);
    }
    expect(audit.attributedSourceClockTraceAudit.fitting.consecutiveSpectrumSha256)
      .toMatch(/^[a-f0-9]{64}$/);
    expect(audit.attributedSourceClockTraceAudit.fitting.qualifiedEnvelopeSha256)
      .toMatch(/^[a-f0-9]{64}$/);
    expect(audit.attributedSourceClockTraceAudit.tailCalibration.consecutiveSpectrumSha256)
      .toMatch(/^[a-f0-9]{64}$/);
    expect(audit.attributedSourceClockTraceAudit.tailCalibration.qualifiedEnvelopeSha256)
      .toMatch(/^[a-f0-9]{64}$/);
  });

  it('declares Bluetooth likelihood support only for the scalar spectrum view', () => {
    expect(observableClassSupportsEvidenceView('bluetooth-like', 'spectrum-only')).toBe(true);
    expect(observableClassSupportsEvidenceView('bluetooth-like', 'envelope-untimed')).toBe(false);
    expect(observableClassSupportsEvidenceView('bluetooth-like', 'envelope-timed')).toBe(false);
    const bluetooth = BAYESIAN_OBSERVABLE_MODEL.classModels.find(
      (model) => model.id === 'bluetooth-like',
    )!;
    expect(observableModelComponents(bluetooth, 'spectrum-only').length).toBeGreaterThan(0);
    expect(observableModelComponents(bluetooth, 'envelope-untimed')).toEqual([]);
    expect(observableModelComponents(bluetooth, 'envelope-timed')).toEqual([]);
    expect(bluetooth.tailCalibrationScoresByView?.['envelope-untimed']).toEqual([]);
    expect(bluetooth.tailCalibrationScoresByView?.['envelope-timed']).toEqual([]);
  });
});

describe('generated model constructor admission', () => {
  const component = () =>
    observableModelComponents(
      BAYESIAN_OBSERVABLE_MODEL.classModels[0]!,
      'spectrum-only',
    )[0]!;

  it('rejects a non-finite fitted location before inference is available', () => {
    const selected = component();
    const location = selected.location as number[];
    const original = location[0]!;
    try {
      location[0] = Number.NaN;
      expect(() => new BayesianWaveformClassifier())
        .toThrow(/location must be finite/);
    } finally {
      location[0] = original;
    }
  });

  it('rejects non-positive fitted degrees of freedom before inference is available', () => {
    const selected = component() as unknown as { degreesOfFreedom: number };
    const original = selected.degreesOfFreedom;
    try {
      selected.degreesOfFreedom = 0;
      expect(() => new BayesianWaveformClassifier())
        .toThrow(/degrees of freedom must be positive/);
    } finally {
      selected.degreesOfFreedom = original;
    }
  });

  it('rejects a non-positive-definite fitted scale before inference is available', () => {
    const selected = component();
    const scale = selected.scale as number[][];
    const original = scale[0]![0]!;
    try {
      scale[0]![0] = 0;
      expect(() => new BayesianWaveformClassifier())
        .toThrow(/scale matrix is not positive definite/);
    } finally {
      scale[0]![0] = original;
    }
  });
});
