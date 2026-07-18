import { describe, expect, it } from 'vitest';
import { canonicalClassificationScenarios } from '../../Atom-SignalLab/src/classification-corpus.js';
import { featureSamples } from './observable-training-sampling.js';
import {
  SIGNAL_LAB_PRODUCTION_ACQUISITION_GEOMETRY,
  SIGNAL_LAB_PRODUCTION_TEMPORAL_SCHEDULE_PAIRS,
  type ObservableTrainingAcquisitionRegime,
} from '../../Atom-Atomizer/packages/analysis/src/observable-training-acquisition-geometry.js';

const fittingSeeds = [407, 1_407, 2_407, 3_407, 4_407, 5_407] as const;

describe('observable-training Wi-Fi envelope coverage', () => {
  it('admits an untimed HR-DSSS envelope in the Bluetooth-classic production source phase', {
    timeout: 180_000,
  }, () => {
    const scenario = canonicalClassificationScenarios.find(
      (candidate) => candidate.id === 'wifi-hr-dsss-11m',
    );
    const schedulePair = SIGNAL_LAB_PRODUCTION_TEMPORAL_SCHEDULE_PAIRS.find(
      (candidate) =>
        candidate.sourcePlanProfileId === 'bluetooth-classic-connected',
    );
    expect(scenario).toBeDefined();
    expect(schedulePair).toBeDefined();
    const regime: ObservableTrainingAcquisitionRegime = {
      id: `${SIGNAL_LAB_PRODUCTION_ACQUISITION_GEOMETRY.id}/${schedulePair!.id}`,
      geometry: SIGNAL_LAB_PRODUCTION_ACQUISITION_GEOMETRY,
      spectrumTemporalSchedule: schedulePair!.spectrumTemporalSchedule,
      qualifiedEnvelopeTemporalSchedule:
        schedulePair!.qualifiedEnvelopeTemporalSchedule,
    };
    const audit = fittingSeeds.map((seed) => {
      const attempt = featureSamples(scenario!, 24, regime, seed);
      const captured =
        attempt.qualifiedEnvelope.detectedPowerCaptureSample;
      return {
        seed,
        captureOpportunity: captured?.observationOpportunity ?? null,
        capturedKey:
          attempt.qualifiedEnvelope.capturedRepresentativeKey ?? null,
        capturedWidthHz: captured === undefined
          ? null
          : 10 ** captured.values['spectrum.logBandwidthHz']!,
        timedFitEligible: captured?.fitEligible ?? false,
        untimedFitEligible:
          captured?.envelopeUntimedFitEligible ?? false,
        disposition: captured?.detectedPowerEvidenceDisposition ?? null,
        hasEnvelope: captured === undefined
          ? false
          : Object.keys(captured.values).some((name) =>
            name.startsWith('envelope.')),
        physicalDetectedPowerCaptureCount:
          attempt.qualifiedEnvelope.physicalDetectedPowerCaptureCount,
        provenanceUnavailableWindowCount:
          attempt.qualifiedEnvelope.provenanceUnavailableWindowCount,
      };
    });

    expect(audit.every(
      (row) => row.provenanceUnavailableWindowCount === 0,
    ), JSON.stringify(audit, null, 2)).toBe(true);
    expect(audit.some(
      (row) =>
        row.untimedFitEligible
        && row.disposition === 'admitted-envelope'
        && row.hasEnvelope
        && row.capturedWidthHz !== null
        && row.capturedWidthHz >= 10_000_000,
    ), JSON.stringify(audit, null, 2)).toBe(true);
  });
});
