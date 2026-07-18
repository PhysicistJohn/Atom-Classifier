import { BAYESIAN_FREQUENCY_AGILE_ACTIVITY_MODEL } from '../../Atom-Atomizer/packages/analysis/src/bayesian-agile-association.js';
import type { ObservableFeatureObservation } from '../../Atom-Atomizer/packages/analysis/src/observable-features.js';
import type { ObservableLeafClass } from './observable-classifier-model.js';
import { compatibleRadioDuplexModes, type RadioAirInterface } from './radio-operating-band-context.js';

// Only the 2.4 GHz Wi-Fi centers are fitted corpus centers. The 5/6 GHz OFDM
// windows are standards-context extrapolations that still require physical
// validation; they are hard exclusion masks, not evidence of band-wide fit.
const SUPPORTED_NON_CELLULAR_CONTEXT_WINDOWS_MHZ = {
  wifiHrDsss: [[2_400, 2_500]],
  wifiOfdm: [[2_400, 2_500], [4_900, 5_925], [5_925, 7_125]],
} as const;

export const OBSERVABLE_HYPOTHESIS_DOMAIN_POLICY_ID = 'observation-only-hypothesis-domain-v5' as const;

export type ObservableHypothesisDomainObservation = Partial<Pick<
  ObservableFeatureObservation,
  | 'occupiedStartHz'
  | 'occupiedStopHz'
  | 'centerHz'
  | 'bandwidthHz'
  | 'limitations'
  | 'values'
  | 'associationEvidenceQualification'
>>;

/**
 * Structural support of the pinned observable hypothesis family.
 *
 * This is deliberately independent of the generated likelihood asset so the
 * trainer, validator, and runtime can apply the same physical/model-domain
 * mask. A standards-compliant waveform outside these fitted boundaries is an
 * unknown observation; a large relative likelihood cannot override the mask.
 */
export function observableHypothesisHasRequiredEvidence(
  id: ObservableLeafClass,
  observation: ObservableHypothesisDomainObservation,
): boolean {
  if (id === 'wifi-hr-dsss-like') {
    const inFittedBand = fittedObservedIntervalInAnyBand(observation, SUPPORTED_NON_CELLULAR_CONTEXT_WINDOWS_MHZ.wifiHrDsss);
    // The fitted 11 Mcps HR-DSSS projection is about 22 MHz wide. Ten MHz is
    // a conservative lower observation boundary, not a universal 802.11 rule.
    const inFittedWidth = observation.bandwidthHz === undefined
      || (observation.bandwidthHz >= 10_000_000 && observation.bandwidthHz <= 30_000_000);
    return inFittedBand && inFittedWidth;
  }
  if (id === 'wifi-ofdm-like') {
    const inFittedBand = fittedObservedIntervalInAnyBand(observation, SUPPORTED_NON_CELLULAR_CONTEXT_WINDOWS_MHZ.wifiOfdm);
    const inFittedWidth = observation.bandwidthHz === undefined
      || (observation.bandwidthHz >= 8_000_000 && observation.bandwidthHz <= 110_000_000);
    return inFittedBand && inFittedWidth;
  }
  if (id === 'gsm-like') {
    const inFittedBand = observedIntervalSupportsDuplex(observation, 'geran', 'fdd');
    const inFittedWidth = observation.bandwidthHz === undefined
      || (observation.bandwidthHz >= 80_000 && observation.bandwidthHz <= 500_000);
    return inFittedBand && inFittedWidth;
  }
  if (id === 'lte-fdd-like' || id === 'lte-tdd-like') {
    // The narrowest detector-conditioned fitted cellular example is nominal
    // 5 MHz LTE. LTE itself also defines 1.4/3 MHz channels; those are simply
    // outside this asset and must not rescue an open-set support score.
    const inFittedBand = observedIntervalSupportsDuplex(
      observation,
      'e-utra',
      id === 'lte-fdd-like' ? 'fdd' : 'tdd',
    );
    const inFittedWidth = observation.bandwidthHz === undefined
      || (observation.bandwidthHz >= 3_500_000 && observation.bandwidthHz <= 25_000_000);
    return inFittedBand && inFittedWidth;
  }
  if (id === 'nr-fdd-like' || id === 'nr-tdd-like') {
    const inFittedBand = observedIntervalSupportsDuplex(
      observation,
      'nr',
      id === 'nr-fdd-like' ? 'fdd' : 'tdd',
    );
    const inFittedWidth = observation.bandwidthHz === undefined
      || (observation.bandwidthHz >= 10_000_000 && observation.bandwidthHz <= 110_000_000);
    return inFittedBand && inFittedWidth;
  }
  if (id === 'am-dsb-full-carrier-like') {
    const carrierFraction = observation.values?.['spectrum.centerFraction'];
    if (carrierFraction === undefined || carrierFraction < 0.5) return false;
    return resolvedSidebandsOrModulatedEnvelope(observation);
  }
  if (id === 'fm-angle-modulated-like') {
    // A locally tracked stationary line is observationally CW-like even when
    // an unconstrained likelihood happens to resemble a coarse-RBW FM
    // component.  Require directly observed symmetric sidebands or a
    // receiver-filtered power envelope with material modulation before the FM
    // leaf participates in fitting, calibration, support, or inference.
    // This does not claim that every standards-valid FM signal will satisfy
    // the finite scalar view; unresolved FM correctly remains CW-like/unknown.
    return resolvedSidebandsOrModulatedEnvelope(observation);
  }
  if (id !== 'bluetooth-like') return true;
  // With only scalar swept spectra and a fixed-tune power envelope, a local
  // stationary 2.4 GHz signal is not Bluetooth evidence. The supported leaf
  // is deliberately a band-activity equivalence class and therefore requires
  // the separately provenance-bound multi-frequency association observation.
  if (!observation.limitations?.includes('frequency-agile-band-activity-association')) return false;
  if (observation.associationEvidenceQualification !== 'provenance-bound-current-promotion') return false;
  const associationLogBayesFactor = observation.values?.['association.logBayesFactor'];
  const priorOdds = BAYESIAN_FREQUENCY_AGILE_ACTIVITY_MODEL.priorAgileDynamicsProbability
    / (1 - BAYESIAN_FREQUENCY_AGILE_ACTIVITY_MODEL.priorAgileDynamicsProbability);
  const promotionOdds = BAYESIAN_FREQUENCY_AGILE_ACTIVITY_MODEL.promotionPosteriorProbability
    / (1 - BAYESIAN_FREQUENCY_AGILE_ACTIVITY_MODEL.promotionPosteriorProbability);
  return associationLogBayesFactor !== undefined
    && associationLogBayesFactor >= Math.log(promotionOdds / priorOdds)
    && fittedObservedIntervalInAnyBand(observation, [[2_402, 2_480]]);
}

function resolvedSidebandsOrModulatedEnvelope(
  observation: ObservableHypothesisDomainObservation,
): boolean {
  const resolvedSidebands = (observation.values?.['spectrum.sidebandScore'] ?? 0) >= 0.2;
  const envelopeRangeDb = observation.values?.['envelope.rangeDb'];
  const envelopeStandardDeviationDb = observation.values?.['envelope.standardDeviationDb'];
  const modulatedEnvelopeObserved = envelopeRangeDb !== undefined
    && envelopeStandardDeviationDb !== undefined
    && envelopeRangeDb >= 2
    && envelopeStandardDeviationDb >= 0.5;
  return resolvedSidebands || modulatedEnvelopeObserved;
}

/**
 * The single observation-only domain mask used by fitting, calibration, and
 * runtime inference. It intentionally accepts no nominal generator bandwidth,
 * corpus label metadata, or tracker-only facts. A representative admitted by
 * this mask at runtime is admitted by the same mask when its class likelihood
 * and support reference are constructed.
 */
export function observableRepresentativeIsInClassDomain(
  id: ObservableLeafClass,
  observation: ObservableHypothesisDomainObservation,
): boolean {
  return observableHypothesisHasRequiredEvidence(id, observation);
}

function fittedObservedIntervalInAnyBand(
  observation: ObservableHypothesisDomainObservation,
  rangesMhz: readonly (readonly [number, number])[],
): boolean {
  if (observation.centerHz === undefined) return true;
  if (observation.bandwidthHz === undefined) return inAnyRange(observation.centerHz / 1_000_000, rangesMhz);
  const halfBandwidthHz = observation.bandwidthHz / 2;
  const logBandwidthRbwRatio = observation.values?.['spectrum.logBandwidthRbwRatio'];
  const estimatedRbwHz = logBandwidthRbwRatio === undefined
    ? 0
    : observation.bandwidthHz / 10 ** logBandwidthRbwRatio;
  // The weighted occupied interval can move by roughly an RBW at either edge.
  // Cap that allowance at 5% of measured width so a coarse/invalid resolution
  // estimate cannot turn a center-only context check back into a soft mask.
  const edgeToleranceHz = Math.min(
    observation.bandwidthHz * 0.05,
    Number.isFinite(estimatedRbwHz) ? estimatedRbwHz * 2 : 0,
  );
  const observedStartHz = observation.occupiedStartHz ?? observation.centerHz - halfBandwidthHz;
  const observedStopHz = observation.occupiedStopHz ?? observation.centerHz + halfBandwidthHz;
  return rangesMhz.some(([startMhz, stopMhz]) => observedStartHz >= startMhz * 1_000_000 - edgeToleranceHz
    && observedStopHz <= stopMhz * 1_000_000 + edgeToleranceHz);
}

function observedIntervalSupportsDuplex(
  observation: ObservableHypothesisDomainObservation,
  airInterface: RadioAirInterface,
  duplexMode: 'fdd' | 'tdd',
): boolean {
  if (observation.centerHz === undefined) return true;
  // The standards table spans more operating bands than the corpus's fitted
  // Band 3/Band 38/n3/n78 centers. Compatibility is structural context only;
  // it does not claim that likelihoods were fitted throughout those bands.
  const { observedStartHz, observedStopHz, edgeToleranceHz } = observedInterval(observation);
  return compatibleRadioDuplexModes(airInterface, observedStartHz, observedStopHz, edgeToleranceHz).has(duplexMode);
}

function observedInterval(observation: ObservableHypothesisDomainObservation): {
  observedStartHz: number;
  observedStopHz: number;
  edgeToleranceHz: number;
} {
  const bandwidthHz = observation.bandwidthHz ?? 0;
  const halfBandwidthHz = bandwidthHz / 2;
  const logBandwidthRbwRatio = observation.values?.['spectrum.logBandwidthRbwRatio'];
  const estimatedRbwHz = logBandwidthRbwRatio === undefined || bandwidthHz === 0
    ? 0
    : bandwidthHz / 10 ** logBandwidthRbwRatio;
  const edgeToleranceHz = Math.min(
    bandwidthHz * 0.05,
    Number.isFinite(estimatedRbwHz) ? estimatedRbwHz * 2 : 0,
  );
  return {
    observedStartHz: observation.occupiedStartHz ?? observation.centerHz! - halfBandwidthHz,
    observedStopHz: observation.occupiedStopHz ?? observation.centerHz! + halfBandwidthHz,
    edgeToleranceHz,
  };
}

function inAnyRange(value: number, ranges: readonly (readonly [number, number])[]): boolean {
  return ranges.some(([start, stop]) => value >= start && value <= stop);
}
