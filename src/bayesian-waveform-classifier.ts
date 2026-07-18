import type { DetectedSignal, WaveformClassification } from '../../Atom-Atomizer/packages/contracts/src/index.js';
import {
  assertStudentTLikelihoodComponent,
  mixtureLogLikelihood,
  logSumExp,
  studentTModelTailProbability,
  type PosteriorCandidate,
} from '../../Atom-Atomizer/packages/analysis/src/bayesian-predictive.js';
import {
  DETECTED_POWER_ACQUISITION_QUALIFICATION,
  DETECTED_POWER_AUTOMATIC_SELECTION_CONDITION,
  DETECTED_POWER_OPERATOR_SELECTION_CONDITION,
  extractObservableFeatures,
  ObservableEvidenceUnavailableError,
  type ObservableFeatureObservation,
  type WaveformEvidence,
} from '../../Atom-Atomizer/packages/analysis/src/observable-features.js';
import {
  OBSERVABLE_EVIDENCE_CENSORING_POLICY,
  OBSERVABLE_EVIDENCE_VIEWS,
  OBSERVABLE_LIKELIHOOD_COMPONENT_DECOMPOSITION_POLICY,
  OBSERVABLE_LEAF_CLASSES,
  observableClassSupportsEvidenceView,
  observableModelComponents,
  observableModelView,
  type ObservableClassifierModelAsset,
  type ObservableDecisionClass,
  type ObservableEvidenceView,
  type ObservableLeafClass,
} from './observable-classifier-model.js';
import {
  BAYESIAN_OBSERVABLE_MODEL,
  BAYESIAN_OBSERVABLE_MODEL_CONTENT_SHA256 as EMBEDDED_MODEL_CONTENT_SHA256,
} from './models/bayesian-observable.generated.js';
import {
  BAYESIAN_OBSERVABLE_MODEL_SHA256,
  BAYESIAN_OBSERVABLE_MODEL_CONTENT_SHA256 as MANIFEST_MODEL_CONTENT_SHA256,
} from './models/bayesian-observable.manifest.generated.js';
import { observableRepresentativeIsInClassDomain } from './observable-hypothesis-domain.js';
import {
  OBSERVABLE_TRAINING_DETECTED_POWER_SYNTHESIS_FILTER_POLICY,
  SIGNAL_LAB_PRODUCTION_ACQUISITION_GEOMETRY,
  SIGNAL_LAB_PRODUCTION_ACQUISITION_REGIME_METADATA,
  SIGNAL_LAB_PRODUCTION_TEMPORAL_SCHEDULE_PAIRS,
} from '../../Atom-Atomizer/packages/analysis/src/observable-training-acquisition-geometry.js';

const EXPECTED_TRAINING_RUNTIME_IDENTITY = Object.freeze({
  policyId: 'exact-repository-node-version-v1',
  nodeVersion: '22.23.1',
  v8Version: '12.4.254.21-node.56',
});

export const BAYESIAN_WAVEFORM_MODEL = {
  id: BAYESIAN_OBSERVABLE_MODEL.id,
  producer: 'tinysa-signal-lab',
  sourceCommit: BAYESIAN_OBSERVABLE_MODEL.sourceCommit,
  corpusSha256: BAYESIAN_OBSERVABLE_MODEL.corpusSha256,
  modelAssetSha256: BAYESIAN_OBSERVABLE_MODEL_SHA256,
  attemptSamplingWorkerRuntimeSha256:
    BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.attemptSamplingWorkerRuntimeSha256,
  trainingRuntimeIdentity:
    BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.trainingRuntimeIdentity,
  preprocessing: BAYESIAN_OBSERVABLE_MODEL.preprocessing,
  priorId: BAYESIAN_OBSERVABLE_MODEL.priorId,
  calibrationId: BAYESIAN_OBSERVABLE_MODEL.calibrationId,
  decisionPolicyId: 'observable-open-set-decision-v10',
  classCount: OBSERVABLE_LEAF_CLASSES.length,
  minimumSpectrumSweeps: 8,
  minimumKnownPosterior: 0.55,
  minimumLeafPosterior: 0.58,
  minimumAggregatePosterior: 0.68,
  minimumSiblingMargin: 0.12,
  maximumUnknownPosteriorForAcceptance: 0.4,
  // This is an engineering cutoff on a finite synthetic-reference lower-tail
  // rank. The fixed, stratified SNR/RBW/scenario grid is not an exchangeable
  // sample from an operational population, so 0.025 is neither a conformal
  // alpha nor a 2.5% false-rejection guarantee. Physical data remain wholly
  // uncalibrated.
  minimumKnownSyntheticSupportRank: 0.025,
} as const;

interface BayesianDecision {
  label: ObservableDecisionClass | 'unknown';
  probability: number;
  level: 'equivalence-class' | 'unknown';
  reason?: WaveformClassification['unknownReason'];
}

export class BayesianWaveformClassifier {
  readonly modelId = BAYESIAN_WAVEFORM_MODEL.id;

  constructor() { assertGeneratedModel(); }

  async classify(detection: DetectedSignal, evidence: WaveformEvidence, signal?: AbortSignal): Promise<WaveformClassification> {
    signal?.throwIfAborted();
    if (!supportedDetectorConfiguration(detection)) return unavailableEvidence(detection, 'out-of-domain', 'detector-configuration-out-of-domain');
    let observation: ObservableFeatureObservation;
    try {
      observation = extractObservableFeatures(detection, evidence);
    } catch (error) {
      // Only observation-domain scarcity is an expected runtime unknown. A
      // coherent-provenance failure can encode malformed, substituted, or
      // contradictory evidence and must remain fail-closed instead of being
      // laundered into an ordinary low-information classification.
      if (error instanceof ObservableEvidenceUnavailableError
        && (error.code === 'local-history-not-uniquely-replayable'
          || error.code === 'insufficient-roi-bins'
          || error.code === 'insufficient-spectrum-history')) {
        return unavailableEvidence(detection, 'insufficient-evidence');
      }
      throw error;
    }
    signal?.throwIfAborted();
    const candidates = inferPosterior(observation);
    const knownSupportRank = knownModelSupportRank(observation);
    const boundaryCensored = observation.limitations.includes('partial-span-boundary-censoring');
    const insufficientSweeps = observation.sweepIds.length < BAYESIAN_WAVEFORM_MODEL.minimumSpectrumSweeps;
    const decision = boundaryCensored
      ? unknownDecision(probability(candidates, 'unknown-signal'), 'out-of-domain')
      : insufficientSweeps
        ? unknownDecision(probability(candidates, 'unknown-signal'), 'insufficient-evidence')
        : selectDecision(candidates, observation, knownSupportRank);
    const supportRejected = decision.label === 'unknown'
      && decision.reason === 'out-of-domain'
      && knownSupportRank < BAYESIAN_WAVEFORM_MODEL.minimumKnownSyntheticSupportRank;
    const outputCandidates = candidates.map((candidate) => ({
      label: candidate.id === 'unknown-signal' ? 'unknown' : `observable:${candidate.id}`,
      confidence: candidate.probability,
      family: candidate.id === 'unknown-signal' ? 'unknown' : leafFamily(candidate.id),
    }));
    return {
      detectionId: detection.id,
      label: decision.label === 'unknown' ? 'unknown' : `observable:${decision.label}`,
      confidence: supportRejected ? 0 : decision.probability,
      candidates: outputCandidates,
      modelId: BAYESIAN_WAVEFORM_MODEL.id,
      qualification: 'bayesian-observable-equivalence',
      scoreKind: 'model-posterior',
      decisionLevel: decision.level,
      decisionSupport: supportRejected
        ? { kind: 'synthetic-support-rank', value: knownSupportRank, threshold: BAYESIAN_WAVEFORM_MODEL.minimumKnownSyntheticSupportRank }
        : { kind: 'model-posterior', value: decision.probability },
      modelProvenance: {
        producer: 'tinysa-signal-lab',
        sourceCommit: BAYESIAN_WAVEFORM_MODEL.sourceCommit,
        corpusSha256: BAYESIAN_WAVEFORM_MODEL.corpusSha256,
        preprocessing: BAYESIAN_WAVEFORM_MODEL.preprocessing,
        modelAssetSha256: BAYESIAN_WAVEFORM_MODEL.modelAssetSha256,
        priorId: BAYESIAN_WAVEFORM_MODEL.priorId,
        calibrationId: BAYESIAN_WAVEFORM_MODEL.calibrationId,
        decisionPolicyId: BAYESIAN_WAVEFORM_MODEL.decisionPolicyId,
      },
      classifiedAt: new Date().toISOString(),
      ...(decision.reason ? { unknownReason: decision.reason } : {}),
      evidence: {
        centerHz: observation.centerHz,
        bandwidthHz: observation.bandwidthHz,
        peakDbm: detection.peakDbm,
        sweepIds: observation.sweepIds,
        ...(observation.zeroSpanCaptureId ? { zeroSpanCaptureId: observation.zeroSpanCaptureId } : {}),
        ...(observation.detectedPowerAcquisitionQualification ? {
          detectedPowerAcquisitionQualification: observation.detectedPowerAcquisitionQualification,
          detectedPowerSelectionCondition: observation.detectedPowerSelectionCondition,
        } : {}),
        views: observation.views,
        features: { ...observation.values, 'model.maximumKnownSyntheticSupportRank': knownSupportRank },
        limitations: observation.limitations,
      },
    };
  }
}

export function inferPosterior(observation: ObservableFeatureObservation): readonly PosteriorCandidate[] {
  assertDetectedPowerEvidenceIsConsistent(observation);
  assertGeneratedModel();
  const view = observableModelView(observation);
  assertObservationMatchesModelView(observation, view);
  const values = BAYESIAN_OBSERVABLE_MODEL.classModels.map((model) => {
    // Domain and structural-view eligibility are logical support boundaries,
    // not low likelihoods. Apply them before touching a component array so an
    // intentionally empty class/view population is never evaluated.
    if (!observableClassSupportsEvidenceView(model.id, view)
      || !observableRepresentativeIsInClassDomain(model.id, observation)) {
      return {
        id: model.id,
        logLikelihood: Number.NEGATIVE_INFINITY,
        logJoint: Number.NEGATIVE_INFINITY,
      };
    }
    const logLikelihood = mixtureLogLikelihood(
      observation.values,
      observableModelComponents(model, view),
    );
    const context = frequencyContextLogEvidence(model.id, observation);
    return { id: model.id, logLikelihood, logJoint: model.logPrior + context + logLikelihood };
  });
  const normalization = logSumExp(values.map((value) => value.logJoint));
  const candidates = values.map((value) => ({ ...value, probability: Math.exp(value.logJoint - normalization) }))
    .sort((left, right) => right.probability - left.probability);
  const total = candidates.reduce((sum, value) => sum + value.probability, 0);
  if (!Number.isFinite(total) || Math.abs(total - 1) > 1e-9 || candidates.some((value) => value.probability < 0 || value.probability > 1)) throw new Error('Observable posterior failed to normalize');
  return candidates;
}

type DecisionObservation = Pick<ObservableFeatureObservation, 'centerHz' | 'bandwidthHz' | 'values'>
  & Partial<Pick<ObservableFeatureObservation,
    | 'occupiedStartHz'
    | 'occupiedStopHz'
    | 'limitations'
    | 'views'
    | 'zeroSpanCaptureId'
    | 'detectedPowerAcquisitionQualification'
    | 'detectedPowerSelectionCondition'>>;

export function selectObservableDecision(
  candidates: readonly PosteriorCandidate[],
  observation?: DecisionObservation,
  knownSupportRank?: number,
): { label: ObservableDecisionClass | 'unknown'; probability: number } {
  if (observation) assertDetectedPowerEvidenceIsConsistent(observation);
  const decision = selectDecision(candidates, observation, knownSupportRank ?? (observation ? knownModelSupportRank(observation) : 1));
  return { label: decision.label, probability: decision.probability };
}

export function knownModelSupportRank(
  observation: Pick<ObservableFeatureObservation, 'values'>
    & Partial<Pick<ObservableFeatureObservation,
      | 'occupiedStartHz'
      | 'occupiedStopHz'
      | 'centerHz'
      | 'bandwidthHz'
      | 'limitations'
      | 'views'
      | 'zeroSpanCaptureId'
      | 'detectedPowerAcquisitionQualification'
      | 'detectedPowerSelectionCondition'>>,
): number {
  assertDetectedPowerEvidenceIsConsistent(observation);
  assertGeneratedModel();
  const view = observableModelView(observation);
  assertObservationMatchesModelView(observation, view);
  return Math.max(0, ...BAYESIAN_OBSERVABLE_MODEL.classModels
    // A tail score answers whether the measured shape is supported by an
    // eligible known hypothesis. Letting an ineligible, broad component win
    // this maximum defeats open-set rejection even though its posterior is
    // structurally zero (notably a stationary 2.4 GHz hard negative versus
    // the frequency-agile Bluetooth activity hypothesis).
    .filter((model) => model.id !== 'unknown-signal'
      && observableClassSupportsEvidenceView(model.id, view)
      && observableRepresentativeIsInClassDomain(model.id, observation))
    .map((model) => {
      const rawTailScore = Math.max(...observableModelComponents(model, view)
        .map((component) => studentTModelTailProbability(observation.values, component)));
      const calibration = model.tailCalibrationScoresByView?.[view];
      if (!calibration?.length) throw new Error(`Known class ${model.id} has no ${view} synthetic support calibration`);
      return empiricalSyntheticSupportRank(rawTailScore, calibration);
    }));
}

type DetectedPowerQualificationObservation = Pick<ObservableFeatureObservation, 'values'>
  & Partial<Pick<ObservableFeatureObservation,
    | 'limitations'
    | 'views'
    | 'zeroSpanCaptureId'
    | 'detectedPowerAcquisitionQualification'
    | 'detectedPowerSelectionCondition'>>;

/**
 * Envelope dimensions were calibrated only for the production causal capture
 * policy. These package-internal inference helpers reject inconsistent
 * structures; the exported classifier remains the trust boundary that derives
 * observations from provenance-bound measurement evidence.
 */
function assertDetectedPowerEvidenceIsConsistent(
  observation: DetectedPowerQualificationObservation,
): void {
  const envelopeFeatureNames = Object.keys(observation.values)
    .filter((name) => name.startsWith('envelope.'));
  const hasEnvelopeFeatures = envelopeFeatureNames.length > 0;
  const envelopeFeaturesAreFinite = envelopeFeatureNames.every((name) =>
    Number.isFinite(observation.values[name]));
  const hasEnvelopeView = observation.views?.includes('detected-power-envelope') ?? false;
  const carriesCaptureId = observation.zeroSpanCaptureId !== undefined;
  const hasValidCaptureId = typeof observation.zeroSpanCaptureId === 'string'
    && observation.zeroSpanCaptureId.length > 0;
  const qualified = observation.detectedPowerAcquisitionQualification
    === DETECTED_POWER_ACQUISITION_QUALIFICATION;
  const automaticSelection = observation.detectedPowerSelectionCondition
    === DETECTED_POWER_AUTOMATIC_SELECTION_CONDITION;
  const operatorSelection = observation.detectedPowerSelectionCondition
    === DETECTED_POWER_OPERATOR_SELECTION_CONDITION;
  const preferredTargetLimitation = observation.limitations?.includes(
    'zero-span-operator-preferred-target-selection',
  ) ?? false;

  if (observation.detectedPowerAcquisitionQualification !== undefined && !qualified) {
    throw new Error('Observable detected-power evidence carries an unknown acquisition qualification');
  }
  if (observation.detectedPowerSelectionCondition !== undefined
    && !automaticSelection
    && !operatorSelection) {
    throw new Error('Observable detected-power evidence carries an unknown target-selection condition');
  }
  if (qualified !== (automaticSelection || operatorSelection)) {
    throw new Error('Observable detected-power acquisition qualification and target-selection condition must be paired');
  }
  if (!qualified) {
    if (hasEnvelopeFeatures || hasEnvelopeView || carriesCaptureId) {
      throw new Error('Observable detected-power evidence is not acquisition-policy qualified');
    }
    return;
  }

  const exactQualifiedViews = observation.views?.length === 2
    && observation.views[0] === 'scalar-spectrum'
    && observation.views[1] === 'detected-power-envelope';
  const contradictoryLimitation = observation.limitations?.some((limitation) =>
    limitation === 'zero-span-missing'
    || limitation === 'zero-span-tune-mismatch'
    || limitation === 'zero-span-provenance-mismatch'
    || limitation === 'zero-span-spectrum-window-mismatch'
    || limitation === 'zero-span-acquisition-policy-unqualified'
    || limitation === 'zero-span-geometry-out-of-domain') ?? false;
  if (!hasEnvelopeFeatures
    || !envelopeFeaturesAreFinite
    || !exactQualifiedViews
    || !hasValidCaptureId
    || contradictoryLimitation
    || (operatorSelection !== preferredTargetLimitation)) {
    throw new Error('Observable detected-power acquisition qualification contradicts its envelope evidence');
  }
}

function assertObservationMatchesModelView(
  observation: Pick<ObservableFeatureObservation, 'values'>,
  view: ObservableEvidenceView,
): void {
  const observedDimensions = Object.keys(observation.values).sort();
  if (observedDimensions.length === 0
    || observedDimensions.some((dimension) => !Number.isFinite(observation.values[dimension]))) {
    throw new Error(`Observable ${view} evidence must contain only finite fitted feature values`);
  }
  let expectedDimensions: readonly string[] | undefined;
  for (const model of BAYESIAN_OBSERVABLE_MODEL.classModels) {
    if (!observableClassSupportsEvidenceView(model.id, view)) continue;
    for (const component of observableModelComponents(model, view)) {
      if (expectedDimensions === undefined) expectedDimensions = component.dimensions;
      if (component.dimensions.length !== expectedDimensions.length
        || component.dimensions.some((dimension, index) => dimension !== expectedDimensions![index])) {
        throw new Error(`Observable ${view} likelihood components do not share one exact feature order`);
      }
    }
  }
  if (!expectedDimensions
    || observedDimensions.length !== expectedDimensions.length
    || observedDimensions.some((dimension, index) => dimension !== expectedDimensions![index])) {
    throw new Error(`Observable ${view} evidence does not match the fitted feature population`);
  }
}

/**
 * Smoothed empirical lower-tail rank against a sorted synthetic reference.
 *
 * If an acquisition attempt has representative supports S_1...S_k and its
 * stored reference score is M=min(S_1...S_k), monotonicity gives R(S_j)>=R(M)
 * for every member j. That makes the attempt-minimum reference conservative
 * for a single member's rank. It does not create an exchangeability or
 * coverage guarantee for the fixed synthetic nuisance grid.
 */
export function empiricalSyntheticSupportRank(rawSupport: number, sortedReference: readonly number[]): number {
  if (!Number.isFinite(rawSupport) || rawSupport < 0 || rawSupport > 1) {
    throw new Error('Synthetic support must be finite and within [0, 1]');
  }
  if (!sortedReference.length) throw new Error('Synthetic support reference must not be empty');
  let previous = Number.NEGATIVE_INFINITY;
  let lowerOrEqual = 0;
  for (const value of sortedReference) {
    if (!Number.isFinite(value) || value < 0 || value > 1 || value < previous) {
      throw new Error('Synthetic support reference must be sorted, finite, and within [0, 1]');
    }
    if (value <= rawSupport) lowerOrEqual += 1;
    previous = value;
  }
  return (lowerOrEqual + 1) / (sortedReference.length + 1);
}

function selectDecision(
  candidates: readonly PosteriorCandidate[],
  observation?: DecisionObservation,
  knownSupportRank = observation ? knownModelSupportRank(observation) : 1,
): BayesianDecision {
  const unknownPosterior = probability(candidates, 'unknown-signal');
  const knownPosterior = 1 - unknownPosterior;
  if (knownSupportRank < BAYESIAN_WAVEFORM_MODEL.minimumKnownSyntheticSupportRank) return unknownDecision(unknownPosterior, 'out-of-domain');
  if (unknownPosterior > BAYESIAN_WAVEFORM_MODEL.maximumUnknownPosteriorForAcceptance || knownPosterior < BAYESIAN_WAVEFORM_MODEL.minimumKnownPosterior) return unknownDecision(unknownPosterior, 'out-of-domain');

  const topKnown = candidates.find((candidate) => candidate.id !== 'unknown-signal');
  if (!topKnown) return unknownDecision(unknownPosterior, 'low-confidence');
  if (!observableRepresentativeIsInClassDomain(topKnown.id as ObservableLeafClass, observation ?? {})) {
    return unknownDecision(unknownPosterior, 'insufficient-evidence');
  }
  const lte = aggregate(candidates, ['lte-fdd-like', 'lte-tdd-like']);
  const nr = aggregate(candidates, ['nr-fdd-like', 'nr-tdd-like']);
  // LTE and NR are disjoint leaves of one cellular-OFDM event. Compute that
  // union against the posterior denominator once. Adding the two separately
  // normalized event probabilities is algebraically equivalent over the
  // reals, but their independently rounded denominators can differ by an ulp
  // and make the later sum exceed one.
  const cellularOfdm = aggregate(candidates, [
    'lte-fdd-like',
    'lte-tdd-like',
    'nr-fdd-like',
    'nr-tdd-like',
  ]);
  const wifi = aggregate(candidates, ['wifi-hr-dsss-like', 'wifi-ofdm-like']);
  const topKnownIsCellularOfdm = topKnown.id === 'lte-fdd-like'
    || topKnown.id === 'lte-tdd-like'
    || topKnown.id === 'nr-fdd-like'
    || topKnown.id === 'nr-tdd-like';
  const topKnownIsWifi = topKnown.id === 'wifi-hr-dsss-like' || topKnown.id === 'wifi-ofdm-like';
  // The pinned corpus starts with nominal 5 MHz LTE. Its detector-conditioned
  // occupied widths do not support claims below this conservative boundary.
  // This is a model-domain boundary, not a claim that narrower LTE cannot
  // exist in the standards.
  const cellularBandwidthInModelDomain = !observation || observation.bandwidthHz >= 3_500_000;
  const qualifiedDuplexTiming = observation?.values['envelope.logTransitionRateHz'] !== undefined;
  if (topKnownIsCellularOfdm && !cellularBandwidthInModelDomain) return unknownDecision(unknownPosterior, 'out-of-domain');
  // LTE and NR at 20 MHz and below can be deliberately spectrum-shared and
  // are not identifiable from scalar power without a separately qualified
  // distinguishing observation. Never let a synthetic texture artifact force
  // a technology leaf in that domain.
  // Allow 25 MHz measured width for a nominal 20 MHz channel because
  // threshold/RBW broadening is itself part of this scalar observation.
  if (observation
    && topKnownIsCellularOfdm
    && cellularBandwidthInModelDomain
    && observation.bandwidthHz <= 25_000_000
    && cellularOfdm >= BAYESIAN_WAVEFORM_MODEL.minimumKnownPosterior) {
    return { label: 'cellular-ofdm-ambiguous', probability: cellularOfdm, level: 'equivalence-class' };
  }
  // Scalar swept power and a fixed-tune detected envelope contain no decoded
  // preamble, DSSS/CCK correlation, cyclic-prefix, or cyclostationary evidence.
  // Keep both Wi-Fi template posteriors as diagnostics, but never promote
  // their within-family ranking to a primary PHY decision. Exact proprietary
  // DSSS/OFDM nulls demonstrate that the observable claim stops at compatible
  // 802.11 channel morphology.
  if (topKnownIsWifi) {
    return wifi >= BAYESIAN_WAVEFORM_MODEL.minimumAggregatePosterior
      ? { label: 'wifi-like', probability: wifi, level: 'equivalence-class' }
      : unknownDecision(unknownPosterior, 'low-confidence');
  }
  const siblings = siblingLeaves(topKnown.id as ObservableLeafClass);
  const secondSibling = Math.max(0, ...siblings.filter((id) => id !== topKnown.id).map((id) => probability(candidates, id)));
  const duplexLeafSupported = !topKnownIsCellularOfdm
    || ((topKnown.id === 'lte-tdd-like' || topKnown.id === 'nr-tdd-like') && qualifiedDuplexTiming);
  if (duplexLeafSupported
    && topKnown.probability >= BAYESIAN_WAVEFORM_MODEL.minimumLeafPosterior
    && topKnown.probability - secondSibling >= BAYESIAN_WAVEFORM_MODEL.minimumSiblingMargin) {
    return { label: topKnown.id as ObservableDecisionClass, probability: topKnown.probability, level: 'equivalence-class' };
  }

  if (topKnownIsCellularOfdm && cellularBandwidthInModelDomain && cellularOfdm >= BAYESIAN_WAVEFORM_MODEL.minimumAggregatePosterior) {
    if (lte >= BAYESIAN_WAVEFORM_MODEL.minimumAggregatePosterior) return { label: 'lte-like', probability: lte, level: 'equivalence-class' };
    if (nr >= BAYESIAN_WAVEFORM_MODEL.minimumAggregatePosterior) return { label: 'nr-like', probability: nr, level: 'equivalence-class' };
    return { label: 'cellular-ofdm-ambiguous', probability: cellularOfdm, level: 'equivalence-class' };
  }
  return unknownDecision(unknownPosterior, 'low-confidence');
}

function siblingLeaves(id: ObservableLeafClass): readonly ObservableLeafClass[] {
  if (id === 'lte-fdd-like' || id === 'lte-tdd-like') return ['lte-fdd-like', 'lte-tdd-like'];
  if (id === 'nr-fdd-like' || id === 'nr-tdd-like') return ['nr-fdd-like', 'nr-tdd-like'];
  if (id === 'wifi-hr-dsss-like' || id === 'wifi-ofdm-like') return ['wifi-hr-dsss-like', 'wifi-ofdm-like'];
  return [id];
}

function frequencyContextLogEvidence(id: ObservableLeafClass, observation: ObservableFeatureObservation): number {
  // Frequency is a structural model-support condition below. Do not add
  // arbitrary unnormalized log constants and call the result Bayesian
  // evidence. A future survey-specific band prior must be explicit,
  // normalized, and versioned before it can enter the posterior.
  void id;
  void observation;
  return 0;
}

function leafFamily(id: string): string {
  if (id.startsWith('cw') || id.startsWith('am-') || id.startsWith('fm-')) return 'analog';
  if (id === 'gsm-like' || id.startsWith('lte-') || id.startsWith('nr-')) return 'cellular';
  if (id.startsWith('wifi-')) return 'wifi';
  if (id.startsWith('bluetooth-')) return 'bluetooth';
  return 'unknown';
}

function aggregate(candidates: readonly PosteriorCandidate[], ids: readonly ObservableLeafClass[]): number {
  const selectedIds = new Set<string>(ids);
  let selectedMass = 0;
  let excludedMass = 0;
  for (const candidate of candidates) {
    if (!Number.isFinite(candidate.probability)
      || candidate.probability < 0
      || candidate.probability > 1) {
      throw new Error('Observable posterior contains an invalid candidate probability');
    }
    if (selectedIds.has(candidate.id)) selectedMass += candidate.probability;
    else excludedMass += candidate.probability;
  }
  const totalMass = selectedMass + excludedMass;
  if (!Number.isFinite(totalMass)
    || totalMass <= 0
    || Math.abs(totalMass - 1) > 1e-9) {
    throw new Error('Observable posterior failed to normalize before aggregation');
  }
  // Posterior leaves are IEEE-754 approximations. Summing a group whose true
  // mass is one can produce 1 + a few ulps even though every leaf and the
  // complete posterior passed normalization. Form the event probability as
  // selected / (selected + complement): this is the Bayesian ratio being
  // represented and is guaranteed to remain in [0, 1] for admitted
  // non-negative mass. Do not clamp the renderer boundary; grossly invalid
  // candidates still fail above.
  return selectedMass / totalMass;
}

function probability(candidates: readonly PosteriorCandidate[], id: string): number {
  return candidates.find((candidate) => candidate.id === id)?.probability ?? 0;
}

function unknownDecision(probabilityValue: number, reason: WaveformClassification['unknownReason']): BayesianDecision {
  return { label: 'unknown', probability: probabilityValue, level: 'unknown', reason };
}

function unavailableEvidence(detection: DetectedSignal, reason: WaveformClassification['unknownReason'], limitation = 'insufficient-spectrum-evidence'): WaveformClassification {
  return {
    detectionId: detection.id,
    label: 'unknown',
    confidence: 0,
    candidates: [{ label: 'unknown', confidence: 0, family: 'unknown' }],
    modelId: BAYESIAN_WAVEFORM_MODEL.id,
    qualification: 'bayesian-observable-equivalence',
    scoreKind: 'none',
    decisionLevel: 'unknown',
    classifiedAt: new Date().toISOString(),
    unknownReason: reason,
    evidence: { centerHz: detection.peakHz, bandwidthHz: detection.bandwidthHz, peakDbm: detection.peakDbm, sweepIds: detection.sweepIds, limitations: [limitation] },
  };
}

function supportedDetectorConfiguration(detection: DetectedSignal): boolean {
  const config = detection.detectorConfig;
  return detection.detectorId === 'bayesian-exponential-multiscale-cfar-v3'
    && config.threshold.strategy === 'noise-relative'
    && config.threshold.marginDb === 10
    && config.minimumBandwidthHz === 0
    && config.minimumProminenceDb === 6
    && config.minimumConsecutiveSweeps === 2
    && config.releaseAfterMissedSweeps === 2;
}

export type { WaveformEvidence } from '../../Atom-Atomizer/packages/analysis/src/observable-features.js';
export { observableClassDefinitions } from './observable-classifier-model.js';

function assertGeneratedModel(): void {
  if (EMBEDDED_MODEL_CONTENT_SHA256 !== MANIFEST_MODEL_CONTENT_SHA256) {
    throw new Error('Observable model and manifest content identities do not match');
  }
  const expectedAcquisitionRegimeIds = [
    ...[12, 20, 35, 55, 80, 120].map((rbwDivisor) =>
      `occupied-bandwidth-rbw-divisor:${rbwDivisor}/independent-production-branch-baselines-v1`),
    ...SIGNAL_LAB_PRODUCTION_TEMPORAL_SCHEDULE_PAIRS.map((temporalSchedulePair) =>
      `${SIGNAL_LAB_PRODUCTION_ACQUISITION_GEOMETRY.id}/${temporalSchedulePair.id}`),
  ];
  const trainingMatrixContract = BAYESIAN_OBSERVABLE_MODEL.trainingMatrix as ObservableClassifierModelAsset['trainingMatrix'];
  if (BAYESIAN_OBSERVABLE_MODEL.id !== 'bayesian-observable-equivalence-v9'
    || BAYESIAN_OBSERVABLE_MODEL.sourceCommit !== 'e7d48afbce7165fa04fd551629891123f3b86d34'
    || BAYESIAN_OBSERVABLE_MODEL.corpusSha256 !== 'd68c151f6f284b14effd28bd3db2a696b095ed4fe72a4a206ccea22f54a10a48'
    || JSON.stringify(BAYESIAN_OBSERVABLE_MODEL.corpusSourceManifest?.artifacts.map((artifact) => artifact.path)) !== JSON.stringify([
      'package-lock.json',
      'package.json',
      'src/canonical-timing.ts',
      'src/catalog.ts',
      'src/classification-corpus.ts',
      'src/contracts.ts',
      'src/source-provenance.ts',
      'src/waveforms.ts',
    ])
    || BAYESIAN_OBSERVABLE_MODEL.preprocessing !== 'scalar-observable-features-v7'
    || BAYESIAN_OBSERVABLE_MODEL.priorId !== 'engineering-design-class-weights-v1'
    || BAYESIAN_OBSERVABLE_MODEL.calibrationId !== 'synthetic-independent-branch-view-matched-causal-acquisition-support-rank-detector-conditioned-physical-uncalibrated-v20'
    || !/^[a-f0-9]{64}$/.test(
      BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.attemptSamplingWorkerRuntimeSha256,
    )
    || JSON.stringify(BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.trainingRuntimeIdentity)
      !== JSON.stringify(EXPECTED_TRAINING_RUNTIME_IDENTITY)
    || JSON.stringify(BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.signalLabProductionAcquisitionRegime)
      !== JSON.stringify(SIGNAL_LAB_PRODUCTION_ACQUISITION_REGIME_METADATA)
    || JSON.stringify(BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.detectedPowerSynthesisFilterPolicy)
      !== JSON.stringify(OBSERVABLE_TRAINING_DETECTED_POWER_SYNTHESIS_FILTER_POLICY)
    || JSON.stringify(BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.productionAcquisitionRegimeHighSnrSeedCoveragePolicy)
      !== JSON.stringify({
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
      })
    || JSON.stringify(BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.fittingAcquisitionRegimeIds)
      !== JSON.stringify(expectedAcquisitionRegimeIds)
    || JSON.stringify(BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.tailCalibrationAcquisitionRegimeIds)
      !== JSON.stringify(expectedAcquisitionRegimeIds)
    || BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.classificationSweeps !== 8
    || BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.observationOpportunityHorizons?.standard !== 32
    || BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.observationOpportunityHorizons.fullBand2g4 !== 96
    || BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.acquisitionBranchPolicy
      !== 'independent-no-auto-spectrum-and-qualified-rank-0-integrated-excess-envelope-sessions-v2'
    || BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.selectionPolicy
      !== 'independent-consecutive-spectrum-and-integrated-excess-rank-0-runtime-admission-qualified-envelope-branches-v9'
    || BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.representativeWeightingPolicy !== 'view-matched-spectrum-event-envelope-causal-attempt-weighting-v4'
    || BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.representativeEligibilityPolicy !== 'observation-only-hypothesis-domain-v5'
    || BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.likelihoodPopulationPolicy
      !== 'independent-branch-view-matched-runtime-event-populations-v3'
    || JSON.stringify(BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.likelihoodComponentDecompositionPolicy)
      !== JSON.stringify(OBSERVABLE_LIKELIHOOD_COMPONENT_DECOMPOSITION_POLICY)
    || JSON.stringify(trainingMatrixContract.frequencyAgileFixedTuneEnvelopeCensoringPolicy)
      !== JSON.stringify(OBSERVABLE_EVIDENCE_CENSORING_POLICY)
    || BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.detectedPowerAcquisitionQualification
      !== DETECTED_POWER_ACQUISITION_QUALIFICATION
    || BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.detectedPowerSelectionCondition
      !== DETECTED_POWER_AUTOMATIC_SELECTION_CONDITION
    || BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.tailCalibrationScoreUnit
      !== 'one-independent-branch-acquisition-attempt-score-per-evidence-view-v4'
    || BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.tailCalibrationRepresentativeSelectionPolicy
      !== 'consecutive-spectrum-all-runtime-representatives-and-independent-integrated-excess-rank-0-envelope-sole-capture-v5'
    || BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.tailCalibrationRepresentativeAggregationPolicy
      !== 'consecutive-spectrum-branch-minimum-qualified-envelope-branch-sole-capture-v5'
    || BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.tailCalibrationRuntimeInterpretationPolicy
      !== 'spectrum-member-dominates-independent-branch-attempt-min-envelope-is-independent-sole-capture-v3'
    || BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.tailCalibrationStatisticalInterpretation !== 'empirical-synthetic-reference-only-no-exchangeability-or-coverage-guarantee-v1') {
    throw new Error('Observable model asset does not match the v9 production admission contract');
  }
  const samplingAudit = BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.causalSamplingAudit;
  const fittingCounts = BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.fittingCapturedEnvelopeCountsByScenario;
  const fittingCountsByView =
    BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.fittingRepresentativeCountsByScenarioByView;
  const calibrationCounts = BAYESIAN_OBSERVABLE_MODEL.trainingMatrix.tailCalibrationAttemptCountsByScenarioByView;
  const censoredCaptureCounts =
    trainingMatrixContract.censoredFrequencyAgileFixedTuneCaptureCountsByScenario;
  const bluetoothScenarioIds = [
    'bluetooth-classic-connected',
    'bluetooth-le-advertising',
  ] as const;
  const envelopeViews = ['envelope-untimed', 'envelope-timed'] as const;
  const censoredCaptureCountsAreValid = (
    counts: Readonly<Record<string, number>> | undefined,
  ): counts is Readonly<Record<string, number>> => counts !== undefined
    && Object.keys(counts).length === bluetoothScenarioIds.length
    && Object.values(counts).every((count) => Number.isSafeInteger(count) && count > 0)
    && bluetoothScenarioIds.every((scenarioId) =>
      Number.isSafeInteger(counts[scenarioId]) && counts[scenarioId]! > 0);
  const partitionAuditIsValid = (
    partition: NonNullable<typeof samplingAudit>['fitting'],
  ): boolean => {
    const spectrum = partition.runtimeBranches.consecutiveSpectrum;
    const envelope = partition.runtimeBranches.qualifiedEnvelope;
    return partition.pairedNuisanceCellCount > 0
      && spectrum.detectedPowerCapturePolicyId === 'no-automatic-detected-power-capture-v1'
      && envelope.detectedPowerCapturePolicyId
        === 'capture-once-after-rank-0-integrated-excess-current-target-runtime-admission-v3'
      && spectrum.attemptCount === partition.pairedNuisanceCellCount
      && envelope.attemptCount === partition.pairedNuisanceCellCount
      && spectrum.physicalDetectedPowerCaptureCount === 0
      && spectrum.postCaptureProvenanceUnavailableWindowCount === 0
      && spectrum.detectedPowerCaptureSampleCount === 0
      && spectrum.censoredFrequencyAgileFixedTuneCaptureCount === 0
      && spectrum.sourceClockEventCount === spectrum.spectrumAcquisitionCount
      && spectrum.onlineSpectrumRepresentativeCount
        === spectrum.fitEligibleRepresentativeCount + spectrum.fitIneligibleRepresentativeCount
      && spectrum.attemptsWithFitEligibleRepresentative
        === partition.eligibleAttemptCountsByView['spectrum-only']
      && spectrum.fitEligibleRepresentativeCount
        === partition.fitEligibleRepresentativeCountsByView['spectrum-only']
      && spectrum.attemptsWithAnyRepresentative <= spectrum.attemptCount
      && spectrum.attemptsWithFitEligibleRepresentative
        <= spectrum.attemptsWithAnyRepresentative
      && spectrum.multiRepresentativeAttemptCount <= spectrum.attemptsWithAnyRepresentative
      && envelope.postCaptureProvenanceUnavailableWindowCount === 0
      && envelope.provenanceUnavailableWindowCount
        === envelope.preCaptureProvenanceUnavailableWindowCount
          + envelope.postCaptureProvenanceUnavailableWindowCount
      && envelope.sourceClockEventCount
        === envelope.spectrumAcquisitionCount + envelope.physicalDetectedPowerCaptureCount
      && envelope.physicalDetectedPowerCaptureCount
        === envelope.receiptVerifiedDetectedPowerCaptureSampleCount
          + envelope.postCaptureProvenanceUnavailableWindowCount
      && envelope.receiptVerifiedDetectedPowerCaptureSampleCount
        === envelope.capturedEnvelopeRepresentativeCount
          + envelope.censoredFrequencyAgileFixedTuneCaptureCount
      && envelope.attemptsWithoutDetectedPowerCapture
        === envelope.attemptCount - envelope.physicalDetectedPowerCaptureCount
      && envelope.fitEligibleUntimedCapturedEnvelopeRepresentativeCount
        <= envelope.capturedEnvelopeRepresentativeCount
      && envelope.fitEligibleTimedCapturedEnvelopeRepresentativeCount
        <= envelope.capturedEnvelopeRepresentativeCount
      && envelope.fitEligibleUntimedCapturedEnvelopeRepresentativeCount
        === partition.fitEligibleRepresentativeCountsByView['envelope-untimed']
      && envelope.fitEligibleTimedCapturedEnvelopeRepresentativeCount
        === partition.fitEligibleRepresentativeCountsByView['envelope-timed']
      && envelope.fitEligibleUntimedCapturedEnvelopeRepresentativeCount
        === partition.eligibleAttemptCountsByView['envelope-untimed']
      && envelope.fitEligibleTimedCapturedEnvelopeRepresentativeCount
        === partition.eligibleAttemptCountsByView['envelope-timed'];
  };
  const unavailableTotal = (
    values: readonly { readonly unavailableWindowCount: number }[],
  ) => values.reduce((sum, item) => sum + item.unavailableWindowCount, 0);
  const traceHashes = samplingAudit?.attributedSourceClockTraceAudit;
  if (!samplingAudit || samplingAudit.schemaVersion !== 3 || !fittingCounts
    || !fittingCountsByView || !calibrationCounts
    || !censoredCaptureCountsAreValid(censoredCaptureCounts?.fitting)
    || !censoredCaptureCountsAreValid(censoredCaptureCounts?.tailCalibration)
    || samplingAudit.provenanceUnavailableAttemptPolicy
      !== 'branch-attributed-exact-attempt-cell-counts-v2'
    || !partitionAuditIsValid(samplingAudit.fitting)
    || !partitionAuditIsValid(samplingAudit.tailCalibration)
    || Object.values(censoredCaptureCounts.fitting).reduce((sum, count) => sum + count, 0)
      !== samplingAudit.fitting.runtimeBranches.qualifiedEnvelope
        .censoredFrequencyAgileFixedTuneCaptureCount
    || Object.values(censoredCaptureCounts.tailCalibration)
      .reduce((sum, count) => sum + count, 0)
      !== samplingAudit.tailCalibration.runtimeBranches.qualifiedEnvelope
        .censoredFrequencyAgileFixedTuneCaptureCount
    || Object.values(fittingCounts).reduce((sum, count) => sum + count, 0)
      !== samplingAudit.fitting.runtimeBranches.qualifiedEnvelope
        .fitEligibleTimedCapturedEnvelopeRepresentativeCount
    || OBSERVABLE_EVIDENCE_VIEWS.some((view) =>
      Object.values(fittingCountsByView).reduce((sum, counts) => sum + (counts[view] ?? 0), 0)
        !== samplingAudit.fitting.fitEligibleRepresentativeCountsByView[view])
    || OBSERVABLE_EVIDENCE_VIEWS.some((view) =>
      Object.values(calibrationCounts).reduce((sum, counts) => sum + (counts[view] ?? 0), 0)
        !== samplingAudit.tailCalibration.eligibleAttemptCountsByView[view])
    || Object.values(fittingCountsByView).some((counts) => OBSERVABLE_EVIDENCE_VIEWS.some(
      (view) => !Number.isSafeInteger(counts[view]) || counts[view] < 0))
    || Object.values(calibrationCounts).some((counts) => OBSERVABLE_EVIDENCE_VIEWS.some(
      (view) => !Number.isSafeInteger(counts[view]) || counts[view] < 0))
    || bluetoothScenarioIds.some((scenarioId) =>
      (fittingCounts[scenarioId] ?? 0) !== 0
      || envelopeViews.some((view) =>
        fittingCountsByView[scenarioId]?.[view] !== 0
        || calibrationCounts[scenarioId]?.[view] !== 0))
    || unavailableTotal(
      samplingAudit.provenanceUnavailableAttempts.fitting.consecutiveSpectrum,
    ) !== samplingAudit.fitting.runtimeBranches.consecutiveSpectrum
      .provenanceUnavailableWindowCount
    || unavailableTotal(
      samplingAudit.provenanceUnavailableAttempts.fitting.qualifiedEnvelope,
    ) !== samplingAudit.fitting.runtimeBranches.qualifiedEnvelope
      .provenanceUnavailableWindowCount
    || unavailableTotal(
      samplingAudit.provenanceUnavailableAttempts.tailCalibration.consecutiveSpectrum,
    ) !== samplingAudit.tailCalibration.runtimeBranches.consecutiveSpectrum
      .provenanceUnavailableWindowCount
    || unavailableTotal(
      samplingAudit.provenanceUnavailableAttempts.tailCalibration.qualifiedEnvelope,
    ) !== samplingAudit.tailCalibration.runtimeBranches.qualifiedEnvelope
      .provenanceUnavailableWindowCount
    || traceHashes?.serialization
      !== 'canonical-attempt-id-branch-attributed-trace-and-capture-disposition-digest-v3'
    || [
      traceHashes.fitting.consecutiveSpectrumSha256,
      traceHashes.fitting.qualifiedEnvelopeSha256,
      traceHashes.tailCalibration.consecutiveSpectrumSha256,
      traceHashes.tailCalibration.qualifiedEnvelopeSha256,
    ].some((hash) => !/^[a-f0-9]{64}$/.test(hash))) {
    throw new Error('Observable model causal sampling audit is absent or internally inconsistent');
  }
  const ids = BAYESIAN_OBSERVABLE_MODEL.classModels.map((model) => model.id);
  if (ids.length !== OBSERVABLE_LEAF_CLASSES.length || new Set(ids).size !== ids.length || OBSERVABLE_LEAF_CLASSES.some((id) => !ids.includes(id))) {
    throw new Error('Observable model taxonomy does not match the runtime contract');
  }
  const priorTotal = BAYESIAN_OBSERVABLE_MODEL.classModels.reduce((sum, model) => sum + Math.exp(model.logPrior), 0);
  if (!Number.isFinite(priorTotal) || Math.abs(priorTotal - 1) > 1e-9) throw new Error('Observable model class priors are not normalized');
  const expectedDimensionsByView: Partial<Record<ObservableEvidenceView, readonly string[]>> = {};
  const componentIdentitiesByView: Record<ObservableEvidenceView, string[]> = {
    'spectrum-only': [],
    'envelope-untimed': [],
    'envelope-timed': [],
  };
  const scenarioAssignmentsByView: Record<ObservableEvidenceView, string[]> = {
    'spectrum-only': [],
    'envelope-untimed': [],
    'envelope-timed': [],
  };
  const expectedCsmaSourceScenarioIds = [
    'unknown-802154',
    'wifi-hr-dsss-11m',
    'wifi-ofdm-20m',
    'wifi-ofdm-40m',
    'wifi-ofdm-80m',
  ];
  const decomposedSourceScenarioIdsByView: Record<ObservableEvidenceView, string[]> = {
    'spectrum-only': [],
    'envelope-untimed': [],
    'envelope-timed': [],
  };
  for (const model of BAYESIAN_OBSERVABLE_MODEL.classModels) {
    if (model.components !== undefined) {
      throw new Error(`Observable model ${model.id} retains a forbidden legacy single-population mixture`);
    }
    for (const view of OBSERVABLE_EVIDENCE_VIEWS) {
      const components = observableModelComponents(model, view);
      const supportsView = observableClassSupportsEvidenceView(model.id, view);
      const calibrationScores = model.tailCalibrationScoresByView?.[view];
      if (!supportsView) {
        if (components.length !== 0 || calibrationScores === undefined
          || calibrationScores.length !== 0) {
          throw new Error(
            `Observable model ${model.id} ${view} must use exact empty likelihood and calibration arrays`,
          );
        }
        continue;
      }
      for (const component of components) {
        assertStudentTLikelihoodComponent(component);
      }
      const weightTotal = components.reduce((sum, component) => sum + Math.exp(component.logWeight), 0);
      if (!Number.isFinite(weightTotal) || Math.abs(weightTotal - 1) > 1e-9) {
        throw new Error(`Observable model ${model.id} ${view} mixture is not normalized`);
      }
      const expectedDimensions = expectedDimensionsByView[view] ?? components[0]!.dimensions;
      expectedDimensionsByView[view] = expectedDimensions;
      if (components.some((component) => component.dimensions.length !== expectedDimensions.length
        || component.dimensions.some((dimension, index) => dimension !== expectedDimensions[index]))) {
        throw new Error(`Observable model ${model.id} ${view} mixture does not use the exact view feature order`);
      }
      const componentsBySourceScenario = new Map<string, typeof components>();
      for (const component of components) {
        if (component.sourceScenarioId === undefined || component.modeId === undefined
          || component.fitSampleCount === undefined) {
          throw new Error(`Observable model ${model.id} ${view} component ${component.id} lacks explicit fitting ownership`);
        }
        const owned = componentsBySourceScenario.get(component.sourceScenarioId) ?? [];
        componentsBySourceScenario.set(component.sourceScenarioId, [...owned, component]);
      }
      for (const [sourceScenarioId, sourceComponents] of componentsBySourceScenario) {
        const expectedFitSampleCount = fittingCountsByView[sourceScenarioId]?.[view];
        const observedFitSampleCount = sourceComponents.reduce(
          (sum, component) => sum + component.fitSampleCount!,
          0,
        );
        if (!Number.isSafeInteger(expectedFitSampleCount) || expectedFitSampleCount! <= 0
          || observedFitSampleCount !== expectedFitSampleCount) {
          throw new Error(`Observable model ${model.id} ${view} source scenario ${sourceScenarioId} does not own its declared fitting representatives`);
        }
        const isExpectedCsmaSource = expectedCsmaSourceScenarioIds.includes(sourceScenarioId);
        if (!isExpectedCsmaSource && sourceComponents.length === 1) {
          const [component] = sourceComponents;
          if (component!.id !== sourceScenarioId || component!.modeId !== 'single-population') {
            throw new Error(`Observable model ${model.id} ${view} ordinary source scenario ${sourceScenarioId} has an invalid component identity`);
          }
        } else if (isExpectedCsmaSource
          && sourceComponents.length === OBSERVABLE_LIKELIHOOD_COMPONENT_DECOMPOSITION_POLICY.csmaModeCount) {
          decomposedSourceScenarioIdsByView[view].push(sourceScenarioId);
          const sharedScale = JSON.stringify(sourceComponents[0]!.scale);
          for (let index = 0; index < sourceComponents.length; index += 1) {
            const modeId = `csma-activity-mode-${index + 1}-of-${sourceComponents.length}`;
            const component = sourceComponents[index]!;
            if (component.id !== `${sourceScenarioId}/${modeId}` || component.modeId !== modeId
              || JSON.stringify(component.scale) !== sharedScale
              || component.fitSampleCount!
                < OBSERVABLE_LIKELIHOOD_COMPONENT_DECOMPOSITION_POLICY.minimumModeFitSampleCount) {
              throw new Error(`Observable model ${model.id} ${view} source scenario ${sourceScenarioId} has an invalid shared-covariance CSMA decomposition`);
            }
          }
          const partitionDimensionIndex = sourceComponents[0]!.dimensions.indexOf(
            OBSERVABLE_LIKELIHOOD_COMPONENT_DECOMPOSITION_POLICY.csmaPartitionFeature,
          );
          const partitionCenters = sourceComponents.map((component) =>
            component.location[partitionDimensionIndex]);
          if (partitionDimensionIndex < 0 || partitionCenters.some((center, index) =>
            !Number.isFinite(center) || (index > 0 && center! <= partitionCenters[index - 1]!))) {
            throw new Error(`Observable model ${model.id} ${view} source scenario ${sourceScenarioId} CSMA centers are not strictly increasing`);
          }
        } else {
          throw new Error(`Observable model ${model.id} ${view} source scenario ${sourceScenarioId} has an unsupported component count`);
        }
        for (const component of sourceComponents) {
          const expectedWeight = (1 / componentsBySourceScenario.size)
            * (component.fitSampleCount! / observedFitSampleCount);
          if (Math.abs(Math.exp(component.logWeight) - expectedWeight) > 1e-9) {
            throw new Error(`Observable model ${model.id} ${view} component ${component.id} has an invalid source-owned mixture weight`);
          }
        }
      }
      componentIdentitiesByView[view].push(...components.map((component) => `${model.id}:${component.id}`));
      scenarioAssignmentsByView[view].push(
        ...[...componentsBySourceScenario.keys()].map((sourceScenarioId) => `${model.id}:${sourceScenarioId}`),
      );
      if (model.id !== 'unknown-signal') {
        const expectedScoreCount = [...componentsBySourceScenario.keys()].reduce(
          (sum, sourceScenarioId) => sum + (calibrationCounts[sourceScenarioId]?.[view] ?? 0),
          0,
        );
        if (!calibrationScores?.length || calibrationScores.some((value, index) =>
          !Number.isFinite(value) || value < 0 || value > 1
          || (index > 0 && value < calibrationScores[index - 1]!))) {
          throw new Error(`Observable model ${view} support calibration ${model.id} is invalid`);
        }
        if (calibrationScores.length !== expectedScoreCount) {
          throw new Error(`Observable model ${view} support calibration ${model.id} does not match its causal attempt count`);
        }
      }
    }
  }
  if (OBSERVABLE_EVIDENCE_VIEWS.some((view) =>
    JSON.stringify([...decomposedSourceScenarioIdsByView[view]].sort())
      !== JSON.stringify(expectedCsmaSourceScenarioIds))) {
    throw new Error('Observable view-matched likelihoods do not decompose the exact five CSMA source scenarios');
  }
  const expectedComponentCounts: Readonly<Record<ObservableEvidenceView, number>> = {
    'spectrum-only': 28,
    'envelope-untimed': 26,
    'envelope-timed': 26,
  };
  const expectedEnvelopeComponentIdentities = componentIdentitiesByView['spectrum-only']
    .filter((identity) => !identity.startsWith('bluetooth-like:'));
  if (OBSERVABLE_EVIDENCE_VIEWS.some((view) =>
    componentIdentitiesByView[view].length !== expectedComponentCounts[view])
    || (['envelope-untimed', 'envelope-timed'] as const).some((view) =>
      componentIdentitiesByView[view].length !== expectedEnvelopeComponentIdentities.length
      || componentIdentitiesByView[view].some(
        (identity, index) => identity !== expectedEnvelopeComponentIdentities[index],
      ))) {
    throw new Error('Observable view-matched likelihoods do not have the exact 28/26/26 component identities');
  }
  const expectedScenarioCounts: Readonly<Record<ObservableEvidenceView, number>> = {
    'spectrum-only': 18,
    'envelope-untimed': 16,
    'envelope-timed': 16,
  };
  const expectedEnvelopeAssignments = scenarioAssignmentsByView['spectrum-only']
    .filter((assignment) => !assignment.startsWith('bluetooth-like:'));
  if (OBSERVABLE_EVIDENCE_VIEWS.some((view) =>
    scenarioAssignmentsByView[view].length !== expectedScenarioCounts[view])
    || (['envelope-untimed', 'envelope-timed'] as const).some((view) =>
      scenarioAssignmentsByView[view].length !== expectedEnvelopeAssignments.length
      || scenarioAssignmentsByView[view].some(
        (assignment, index) => assignment !== expectedEnvelopeAssignments[index],
      ))) {
    throw new Error('Observable view-matched likelihoods do not have the exact 18/16/16 scenario assignments');
  }
  const dimensionUnion = [...new Set(OBSERVABLE_EVIDENCE_VIEWS.flatMap((view) =>
    expectedDimensionsByView[view] ?? []))].sort();
  if (dimensionUnion.length !== BAYESIAN_OBSERVABLE_MODEL.dimensions.length
    || dimensionUnion.some((dimension, index) => dimension !== BAYESIAN_OBSERVABLE_MODEL.dimensions[index])) {
    throw new Error('Observable model global feature manifest does not equal its view-specific dimension union');
  }
}
