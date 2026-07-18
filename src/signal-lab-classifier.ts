import { BAYESIAN_WAVEFORM_MODEL, BayesianWaveformClassifier } from './bayesian-waveform-classifier.js';
export type { WaveformEvidence } from '../../Atom-Atomizer/packages/analysis/src/observable-features.js';

/**
 * Observable-model provenance only. SignalLab's live profile catalog belongs
 * to the admitted driver capability and must never be duplicated here as a
 * closed inference taxonomy.
 */
export const SIGNAL_LAB_EMSO_MODEL = {
  id: BAYESIAN_WAVEFORM_MODEL.id,
  producer: 'tinysa-signal-lab',
  sourceCommit: BAYESIAN_WAVEFORM_MODEL.sourceCommit,
  corpusSha256: BAYESIAN_WAVEFORM_MODEL.corpusSha256,
  modelAssetSha256: BAYESIAN_WAVEFORM_MODEL.modelAssetSha256,
  preprocessing: BAYESIAN_WAVEFORM_MODEL.preprocessing,
  priorId: BAYESIAN_WAVEFORM_MODEL.priorId,
  calibrationId: BAYESIAN_WAVEFORM_MODEL.calibrationId,
  observableClassCount: BAYESIAN_WAVEFORM_MODEL.classCount,
  minimumSpectrumSweeps: BAYESIAN_WAVEFORM_MODEL.minimumSpectrumSweeps,
  maximumSpectrumSweeps: BAYESIAN_WAVEFORM_MODEL.minimumSpectrumSweeps,
} as const;

/** Compatibility name retained for the desktop integration; inference is observable-class v5. */
export class SignalLabBayesianClassifier extends BayesianWaveformClassifier {}
