import { describe, expect, it } from 'vitest';
import { canonicalClassificationScenario } from '../../Atom-SignalLab/src/classification-corpus.js';
import { featureSamples } from './observable-training-sampling.js';
import {
  SIGNAL_LAB_PRODUCTION_ACQUISITION_GEOMETRY,
  SIGNAL_LAB_PRODUCTION_TEMPORAL_SCHEDULE_PAIRS,
  type ObservableTrainingAcquisitionRegime,
} from '../../Atom-Atomizer/packages/analysis/src/observable-training-acquisition-geometry.js';

const fittingSeeds = [407, 1_407, 2_407, 3_407, 4_407, 5_407] as const;

describe('observable-training Bluetooth fixed-tune capture coverage', () => {
  it.each([
    'bluetooth-classic-connected',
    'bluetooth-le-advertising',
  ] as const)('captures %s through its exact qualified agile member and censors the envelope', {
    timeout: 180_000,
  }, (profileId) => {
    const scenario = canonicalClassificationScenario(profileId);
    const schedulePair = SIGNAL_LAB_PRODUCTION_TEMPORAL_SCHEDULE_PAIRS.find(
      (candidate) => candidate.sourcePlanProfileId === profileId,
    );
    expect(schedulePair).toBeDefined();
    const regime: ObservableTrainingAcquisitionRegime = {
      id: `${SIGNAL_LAB_PRODUCTION_ACQUISITION_GEOMETRY.id}/${schedulePair!.id}`,
      geometry: SIGNAL_LAB_PRODUCTION_ACQUISITION_GEOMETRY,
      spectrumTemporalSchedule: schedulePair!.spectrumTemporalSchedule,
      qualifiedEnvelopeTemporalSchedule:
        schedulePair!.qualifiedEnvelopeTemporalSchedule,
    };

    const audit = fittingSeeds.map((seed) => {
      const qualifiedEnvelope = featureSamples(scenario, 32, regime, seed)
        .qualifiedEnvelope;
      const captured = qualifiedEnvelope.detectedPowerCaptureSample;
      return {
        seed,
        captureCount: qualifiedEnvelope.physicalDetectedPowerCaptureCount,
        preUnavailable:
          qualifiedEnvelope.preCaptureProvenanceUnavailableWindowCount,
        postUnavailable:
          qualifiedEnvelope.postCaptureProvenanceUnavailableWindowCount,
        unavailable: qualifiedEnvelope.provenanceUnavailableWindowCount,
        representativeKey: qualifiedEnvelope.capturedRepresentativeKey,
        observationOpportunity: captured?.observationOpportunity,
        observationHorizon: qualifiedEnvelope.observationHorizon,
        envelopeUntimedFitEligible: captured?.envelopeUntimedFitEligible,
        timedFitEligible: captured?.fitEligible,
        disposition: captured?.detectedPowerEvidenceDisposition,
        hasEnvelope: captured === undefined
          ? false
          : Object.keys(captured.values).some((name) =>
            name.startsWith('envelope.')),
        featureCount: captured === undefined
          ? 0
          : Object.keys(captured.values).length,
      };
    });

    expect(audit.every((row) => row.captureCount === 1
      && row.preUnavailable === 0
      && row.postUnavailable === 0
      && row.unavailable === 0
      && row.representativeKey !== undefined
      && /^frequency-agile-2g4-activity:agile-2g4-activity-\d{4,}$/.test(
        row.representativeKey,
      )
      && row.observationOpportunity !== undefined
      && row.observationOpportunity >= 8
      && row.observationOpportunity <= row.observationHorizon
      && row.envelopeUntimedFitEligible === false
      && row.timedFitEligible === false
      && row.disposition === 'censored-frequency-agile-fixed-tune'
      && row.featureCount === 18
      && !row.hasEnvelope), JSON.stringify(audit, null, 2)).toBe(true);
  });
});
