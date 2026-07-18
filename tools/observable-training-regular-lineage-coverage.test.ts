import { describe, expect, it } from 'vitest';
import { canonicalClassificationScenarios } from '../../Atom-SignalLab/src/classification-corpus.js';
import { featureSamples } from './observable-training-sampling.js';
import {
  SIGNAL_LAB_PRODUCTION_ACQUISITION_GEOMETRY,
  SIGNAL_LAB_PRODUCTION_TEMPORAL_SCHEDULE_PAIRS,
  type ObservableTrainingAcquisitionRegime,
} from '../../Atom-Atomizer/packages/analysis/src/observable-training-acquisition-geometry.js';

const fittingSeeds = [407, 1_407, 2_407, 3_407, 4_407, 5_407] as const;

describe('observable-training regular-lineage coverage', () => {
  it.each(fittingSeeds)(
    'admits FM evidence for the formerly fragmented SNR-24 seed %i cell',
    { timeout: 90_000 },
    (seed) => {
    const scenario = canonicalClassificationScenarios.find(
      (candidate) => candidate.id === 'fm-beta-3',
    );
    const schedulePair = SIGNAL_LAB_PRODUCTION_TEMPORAL_SCHEDULE_PAIRS.find(
      (candidate) => candidate.sourcePlanProfileId === 'cw',
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
    const attempt = featureSamples(scenario!, 24, regime, seed);
    const spectrumEligible =
      attempt.consecutiveSpectrum.onlineSpectrumRepresentatives.filter(
        (representative) => representative.fitEligible,
      );
    const audit = {
      seed,
      spectrumRepresentativeCount:
        attempt.consecutiveSpectrum.onlineSpectrumRepresentatives.length,
      spectrumEligibleCount: spectrumEligible.length,
      envelopeEligible:
        attempt.qualifiedEnvelope.detectedPowerCaptureSample?.fitEligible === true,
      minimumEligibleSidebandScore: Math.min(
        ...spectrumEligible.map(
          (representative) =>
            representative.values['spectrum.sidebandScore']
              ?? Number.NEGATIVE_INFINITY,
        ),
      ),
      provenanceUnavailableWindowCount:
        attempt.consecutiveSpectrum.provenanceUnavailableWindowCount,
      sourceClockEventCount:
        attempt.consecutiveSpectrum.sourceClockEventCount,
    };

    expect(audit, JSON.stringify(audit, null, 2)).toMatchObject({
      envelopeEligible: true,
      provenanceUnavailableWindowCount: 0,
      sourceClockEventCount: 32,
    });
    expect(audit.spectrumEligibleCount, JSON.stringify(audit, null, 2))
      .toBeGreaterThan(0);
    expect(audit.minimumEligibleSidebandScore, JSON.stringify(audit, null, 2))
      .toBeGreaterThanOrEqual(0.2);
  });
});
