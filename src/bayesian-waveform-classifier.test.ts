import { describe, expect, it } from 'vitest';
import {
  BAYESIAN_WAVEFORM_MODEL,
  empiricalSyntheticSupportRank,
  selectObservableDecision,
} from './bayesian-waveform-classifier.js';
import { BAYESIAN_OBSERVABLE_MODEL } from './models/bayesian-observable.generated.js';
import {
  observableClassSupportsEvidenceView,
  observableModelComponents,
} from './observable-classifier-model.js';

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

describe('posterior event arithmetic', () => {
  it('keeps the live LTE Band 3 cellular union inside the probability space', () => {
    // Exact leaf probabilities replayed from SignalLab sequence 3954 in the
    // live scalar-lte-b3-1784363087146 failure. Every leaf was individually
    // valid, but adding independently normalized LTE and NR events produced
    // 1.0000000000000002 at the primary decision boundary.
    const candidates = [
      candidate('lte-fdd-like', 0.5364620990503566),
      candidate('nr-fdd-like', 0.46353790094964487),
      candidate('unknown-signal', 1.1361602155896544e-27),
      candidate('cw-like', 5.144353888487742e-32),
      candidate('am-dsb-full-carrier-like', 0),
      candidate('fm-angle-modulated-like', 0),
      candidate('gsm-like', 0),
      candidate('lte-tdd-like', 0),
      candidate('nr-tdd-like', 0),
      candidate('wifi-hr-dsss-like', 0),
      candidate('wifi-ofdm-like', 0),
      candidate('bluetooth-like', 0),
    ];

    const decision = selectObservableDecision(candidates, {
      centerHz: 1_840_000_249.2651205,
      bandwidthHz: 17_572_383.07349682,
      values: {},
    }, 1);

    expect(decision).toEqual({
      label: 'cellular-ofdm-ambiguous',
      probability: 1,
    });
    expect(decision.probability).toBeGreaterThanOrEqual(0);
    expect(decision.probability).toBeLessThanOrEqual(1);
  });
});

describe('detected-power selection-condition contract', () => {
  const qualifiedEnvelope = {
    centerHz: 100,
    bandwidthHz: 20,
    values: { 'envelope.meanDbm': -50 },
    views: ['scalar-spectrum', 'detected-power-envelope'] as const,
    zeroSpanCaptureId: 'capture-1',
    detectedPowerAcquisitionQualification:
      'receipt-verified-provenance-bound-runtime-admitted-physical-capture-v5' as const,
    limitations: [],
  };

  it('rejects a qualified envelope that omits its automatic/operator selection condition', () => {
    expect(() => selectObservableDecision([], qualifiedEnvelope, 1))
      .toThrow(/qualification and target-selection condition must be paired/i);
  });

  it('requires the preferred-target limitation exactly for operator-selected evidence', () => {
    expect(() => selectObservableDecision([], {
      ...qualifiedEnvelope,
      detectedPowerSelectionCondition: 'operator-preferred-current-target',
    }, 1)).toThrow(/qualification contradicts its envelope evidence/i);
    expect(() => selectObservableDecision([], {
      ...qualifiedEnvelope,
      detectedPowerSelectionCondition:
        'automatic-current-source-sweep-integrated-excess-rank-0',
      limitations: ['zero-span-operator-preferred-target-selection'] as const,
    }, 1)).toThrow(/qualification contradicts its envelope evidence/i);
  });
});

function candidate(id: string, probability: number) {
  return {
    id,
    probability,
    logLikelihood: probability === 0 ? Number.NEGATIVE_INFINITY : Math.log(probability),
    logJoint: probability === 0 ? Number.NEGATIVE_INFINITY : Math.log(probability),
  };
}
