import { describe, expect, it } from 'vitest';
import {
  OBSERVABLE_HYPOTHESIS_DOMAIN_POLICY_ID,
  observableHypothesisHasRequiredEvidence,
  observableRepresentativeIsInClassDomain,
  type ObservableHypothesisDomainObservation,
} from './observable-hypothesis-domain.js';

describe('observable hypothesis domain', () => {
  it('keeps narrow detector fragments out of both an LTE-TDD fit and runtime support', () => {
    // This reproduces the held-out divisor-98 failure: a nominal 10 MHz TDD
    // scenario can yield a high-SNR local fragment below the fitted 3.5 MHz
    // measured-width floor. The fragment is an acquisition miss, not a
    // representative from which to fit or score the LTE hypothesis.
    const narrowFragment = {
      centerHz: 2_350_000_000,
      bandwidthHz: 2_900_000,
      occupiedStartHz: 2_348_550_000,
      occupiedStopHz: 2_351_450_000,
      values: { 'spectrum.logBandwidthRbwRatio': 1.4 },
    } satisfies ObservableHypothesisDomainObservation;

    expect(observableHypothesisHasRequiredEvidence('lte-tdd-like', narrowFragment)).toBe(false);
    expect(observableRepresentativeIsInClassDomain(
      'lte-tdd-like',
      narrowFragment,
    )).toBe(false);

    const resolvedChannel = {
      ...narrowFragment,
      bandwidthHz: 8_000_000,
      occupiedStartHz: 2_346_000_000,
      occupiedStopHz: 2_354_000_000,
    };
    expect(observableHypothesisHasRequiredEvidence('lte-tdd-like', resolvedChannel)).toBe(true);
    expect(observableRepresentativeIsInClassDomain(
      'lte-tdd-like',
      resolvedChannel,
    )).toBe(true);
  });

  it('also applies fitted band containment before admitting a training representative', () => {
    const outOfBand = {
      centerHz: 2_450_000_000,
      bandwidthHz: 10_000_000,
      occupiedStartHz: 2_445_000_000,
      occupiedStopHz: 2_455_000_000,
      values: {},
    } satisfies ObservableHypothesisDomainObservation;
    expect(observableHypothesisHasRequiredEvidence('lte-tdd-like', outOfBand)).toBe(false);
    expect(observableRepresentativeIsInClassDomain(
      'lte-tdd-like',
      outOfBand,
    )).toBe(false);
  });

  it('does not collapse supplemental rows into TDD and does not treat FR1 as one continuous band', () => {
    const release18OverlappingSupplementalContext = {
      centerHz: 722_500_000,
      bandwidthHz: 10_000_000,
      occupiedStartHz: 717_500_000,
      occupiedStopHz: 727_500_000,
      values: {},
    } satisfies ObservableHypothesisDomainObservation;
    // n29 SDL and n83 SUL overlap n109 FDD in this exact revision. FDD remains
    // structurally possible, while neither supplemental row invents TDD.
    expect(observableHypothesisHasRequiredEvidence('nr-fdd-like', release18OverlappingSupplementalContext)).toBe(true);
    expect(observableHypothesisHasRequiredEvidence('nr-tdd-like', release18OverlappingSupplementalContext)).toBe(false);

    const unallocatedFr1Context = {
      centerHz: 1_100_000_000,
      bandwidthHz: 20_000_000,
      values: {},
    } satisfies ObservableHypothesisDomainObservation;
    expect(observableHypothesisHasRequiredEvidence('nr-fdd-like', unallocatedFr1Context)).toBe(false);
    expect(observableHypothesisHasRequiredEvidence('nr-tdd-like', unallocatedFr1Context)).toBe(false);
  });

  it('preserves standards-table overlap and supports complete high-FR1 TDD intervals', () => {
    const overlappingFddAndSul = {
      centerHz: 1_742_500_000,
      bandwidthHz: 20_000_000,
      values: {},
    } satisfies ObservableHypothesisDomainObservation;
    expect(observableHypothesisHasRequiredEvidence('nr-fdd-like', overlappingFddAndSul)).toBe(true);
    expect(observableHypothesisHasRequiredEvidence('nr-tdd-like', overlappingFddAndSul)).toBe(false);

    const n96 = {
      centerHz: 6_000_000_000,
      bandwidthHz: 100_000_000,
      values: {},
    } satisfies ObservableHypothesisDomainObservation;
    expect(observableHypothesisHasRequiredEvidence('nr-tdd-like', n96)).toBe(true);
    expect(observableHypothesisHasRequiredEvidence('nr-fdd-like', n96)).toBe(false);
  });

  it('keeps 320 MHz Wi-Fi and swept RU/puncture stories outside the fitted scalar asset', () => {
    const sixGhz320MhzChannel = {
      centerHz: 6_105_000_000,
      bandwidthHz: 300_000_000,
      values: { 'spectrum.centerNotch': 0.9, 'spectrum.logClusterCount': Math.log1p(4) },
    } satisfies ObservableHypothesisDomainObservation;
    expect(observableHypothesisHasRequiredEvidence('wifi-ofdm-like', sixGhz320MhzChannel)).toBe(false);
  });

  it('uses the identical observation-only AM domain in fit, calibration, and runtime', () => {
    const resolvedAmEvidence = {
      centerHz: 98_000_000,
      bandwidthHz: 4_000,
      values: {
        'spectrum.centerFraction': 0.8,
        'spectrum.sidebandScore': 0.7,
      },
    } satisfies ObservableHypothesisDomainObservation;
    expect(observableHypothesisHasRequiredEvidence('am-dsb-full-carrier-like', resolvedAmEvidence)).toBe(true);
    expect(observableRepresentativeIsInClassDomain(
      'am-dsb-full-carrier-like',
      resolvedAmEvidence,
    )).toBe(true);
  });

  it('requires observable modulation before admitting the FM leaf', () => {
    const unresolvedLine = {
      centerHz: 98_000_000,
      bandwidthHz: 20_000,
      values: {
        'spectrum.sidebandScore': 0,
        'envelope.rangeDb': 0.2,
        'envelope.standardDeviationDb': 0.05,
      },
    } satisfies ObservableHypothesisDomainObservation;
    expect(observableRepresentativeIsInClassDomain('fm-angle-modulated-like', unresolvedLine)).toBe(false);
    expect(observableRepresentativeIsInClassDomain('cw-like', unresolvedLine)).toBe(true);
    expect(observableRepresentativeIsInClassDomain('fm-angle-modulated-like', {
      ...unresolvedLine,
      values: { ...unresolvedLine.values, 'spectrum.sidebandScore': 0.7 },
    })).toBe(true);
    expect(observableRepresentativeIsInClassDomain('fm-angle-modulated-like', {
      ...unresolvedLine,
      values: {
        ...unresolvedLine.values,
        'envelope.rangeDb': 4,
        'envelope.standardDeviationDb': 1,
      },
    })).toBe(true);
  });

  it('pins one observation-only domain policy across every hypothesis and evidence view', () => {
    expect(OBSERVABLE_HYPOTHESIS_DOMAIN_POLICY_ID).toBe('observation-only-hypothesis-domain-v5');
    const hypotheses = [
      'cw-like', 'am-dsb-full-carrier-like', 'fm-angle-modulated-like', 'gsm-like',
      'lte-fdd-like', 'lte-tdd-like', 'nr-fdd-like', 'nr-tdd-like',
      'wifi-hr-dsss-like', 'wifi-ofdm-like', 'bluetooth-like', 'unknown-signal',
    ] as const;
    const base: ObservableHypothesisDomainObservation = {
      centerHz: 2_450_000_000,
      bandwidthHz: 20_000_000,
      occupiedStartHz: 2_440_000_000,
      occupiedStopHz: 2_460_000_000,
      limitations: ['frequency-agile-band-activity-association'],
      associationEvidenceQualification: 'provenance-bound-current-promotion',
      values: {
        'association.logBayesFactor': 20,
        'spectrum.centerFraction': 0.8,
        'spectrum.sidebandScore': 0.7,
      },
    };
    for (const id of hypotheses) {
      for (const values of [
        base.values,
        { ...base.values, 'envelope.rangeDb': 4, 'envelope.standardDeviationDb': 1 },
        { ...base.values, 'envelope.rangeDb': 4, 'envelope.standardDeviationDb': 1, 'envelope.logTransitionRateHz': 2 },
      ]) {
        const observation = { ...base, values };
        expect(observableRepresentativeIsInClassDomain(id, observation))
          .toBe(observableHypothesisHasRequiredEvidence(id, observation));
      }
    }
  });
});
