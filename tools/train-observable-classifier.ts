import { lstatSync, readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { dirname, posix, resolve } from 'node:path';
import { availableParallelism } from 'node:os';
import { Worker } from 'node:worker_threads';
import {
  CLASSIFICATION_CORPUS_VERSION,
  canonicalClassificationScenarios,
  type CanonicalClassificationScenario,
} from '../../TinySA_SignalLab/src/classification-corpus.js';
import {
  CANONIZED_REPLAY_DETECTED_POWER_SYNTHESIS_FILTER_WIDTH_HZ,
} from '../../TinySA_SignalLab/src/waveforms.js';
import {
  OBSERVABLE_TRAINING_BASELINE_QUALIFIED_ENVELOPE_TEMPORAL_SCHEDULE,
  OBSERVABLE_TRAINING_BASELINE_SPECTRUM_TEMPORAL_SCHEDULE,
  OBSERVABLE_TRAINING_DETECTED_POWER_SYNTHESIS_FILTER_POLICY,
  SIGNAL_LAB_PRODUCTION_ACQUISITION_BRANCH_POLICY_ID,
  SIGNAL_LAB_PRODUCTION_DETECTED_POWER_CAPTURE_POLICY_ID,
  SIGNAL_LAB_PRODUCTION_DETECTED_POWER_SYNTHESIS_FILTER_WIDTH_HZ,
  SIGNAL_LAB_PRODUCTION_ACQUISITION_GEOMETRY,
  SIGNAL_LAB_PRODUCTION_ACQUISITION_REGIME_METADATA,
  SIGNAL_LAB_PRODUCTION_SPECTRUM_DETECTED_POWER_CAPTURE_POLICY_ID,
  SIGNAL_LAB_PRODUCTION_TEMPORAL_SCHEDULE_PAIRS,
  occupiedBandwidthRbwDivisorGeometry,
  type ObservableTrainingAcquisitionRegime,
} from '../../TinySA/packages/analysis/src/observable-training-acquisition-geometry.js';
import { studentTModelTailProbability } from '../../TinySA/packages/analysis/src/bayesian-predictive.js';
import { DETECTED_POWER_ACQUISITION_QUALIFICATION } from '../../TinySA/packages/analysis/src/observable-features.js';
import {
  OBSERVABLE_EVIDENCE_CENSORING_POLICY,
  OBSERVABLE_EVIDENCE_VIEWS,
  OBSERVABLE_LEAF_CLASSES,
  observableClassSupportsEvidenceView,
  observableModelComponents,
  type ObservableClassifierModelAsset,
  type ObservableEvidenceView,
  type ObservableLeafClass,
} from '../src/observable-classifier-model.js';
import {
  CLASSIFICATION_SWEEPS,
  STANDARD_OBSERVATION_OPPORTUNITIES,
  FULL_BAND_2G4_OBSERVATION_OPPORTUNITIES,
  observableModelClass,
  envelopeUntimed,
  featureSamples,
  type FeatureSamplingAttempt,
} from './observable-training-sampling.js';
import type {
  AttemptSamplingWorkItem,
  AttemptSamplingWorkResponse,
} from './observable-training-sampling-worker.js';
import {
  createAttemptSamplingCache,
  type AttemptSamplingCacheChunkRecord,
} from './observable-training-attempt-cache.js';
import { postToAttemptSamplingWorker } from './observable-training-worker-client.js';
import {
  acquireObservableTrainingRun,
  openFreshSamplingRunJournal,
} from './observable-training-run-control.js';
import {
  assertGeneratedModelManifestPair,
  publishGeneratedModelManifestRecoverably,
  recoverGeneratedModelManifestPublication,
} from './observable-model-publication.js';
import {
  assertObservableTrainingBuildAttestation,
  OBSERVABLE_TRAINING_RUNTIME_IDENTITY_POLICY,
} from './observable-training-build-attestation.js';
import {
  componentSourceScenarioId,
  fitScenarioStudentTComponents,
  OBSERVABLE_LIKELIHOOD_COMPONENT_DECOMPOSITION_POLICY,
} from './observable-model-fitting.js';

const OUTPUT = resolve('src/models/bayesian-observable.generated.ts');
const MANIFEST_OUTPUT = resolve('src/models/bayesian-observable.manifest.generated.ts');
const CLI_ARGUMENTS = process.argv.slice(2);
const SUPPORTED_CLI_ARGUMENTS = new Set(['--check', '--fresh-sampling']);
if (CLI_ARGUMENTS.some((argument) => !SUPPORTED_CLI_ARGUMENTS.has(argument))
  || new Set(CLI_ARGUMENTS).size !== CLI_ARGUMENTS.length) {
  throw new Error('Usage: train-observable-classifier [--check] [--fresh-sampling]');
}
const CHECK_ONLY = CLI_ARGUMENTS.includes('--check');
const FRESH_SAMPLING = CLI_ARGUMENTS.includes('--fresh-sampling');
const SOURCE_WORKER_MODULE_URL =
  new URL('./observable-training-sampling-worker.js', import.meta.url);
const BUILD_ATTESTATION = assertObservableTrainingBuildAttestation({
  trainerModuleUrl: new URL(import.meta.url),
  workerModuleUrl: SOURCE_WORKER_MODULE_URL,
});
const TRAINING_RUNTIME_IDENTITY = Object.freeze({
  policyId: OBSERVABLE_TRAINING_RUNTIME_IDENTITY_POLICY,
  nodeVersion: BUILD_ATTESTATION.nodeVersion,
  v8Version: BUILD_ATTESTATION.v8Version,
});
const TRAINER_RUN = acquireObservableTrainingRun({
  lockPath: resolve('.artifacts/observable-training-run.lock'),
  runRoot: resolve('.artifacts/observable-training-runs'),
  sourceWorkerModuleUrl: SOURCE_WORKER_MODULE_URL,
});
if (TRAINER_RUN.workerRuntimeSha256 !== BUILD_ATTESTATION.workerRuntimeSha256) {
  TRAINER_RUN.release();
  throw new Error(
    'Pinned attempt-sampling worker runtime does not match its private build attestation',
  );
}
TRAINER_RUN.installProcessCleanupHandlers();
const WORKER_MODULE_URL = TRAINER_RUN.workerModuleUrl;
const MODEL_PUBLICATION_JOURNAL = resolve(
  '.artifacts/observable-training-model-publication.json',
);
const publicationRecovery = recoverGeneratedModelManifestPublication({
  modelPath: OUTPUT,
  manifestPath: MANIFEST_OUTPUT,
  journalPath: MODEL_PUBLICATION_JOURNAL,
});
if (publicationRecovery !== 'none') {
  console.error(`[model-publication] recovery=${publicationRecovery}`);
}
const SOURCE_COMMIT = '03bc13eb9d5efcfc5f2f9c1792042f670b71ef9a';
const SIGNAL_LAB_REPOSITORY_ROOT = resolve('../TinySA_SignalLab');
const CHECKED_OUT_SOURCE_COMMIT = gitOutput(['rev-parse', 'HEAD']).toString('utf8').trim();
if (CHECKED_OUT_SOURCE_COMMIT !== SOURCE_COMMIT) {
  throw new Error(`SignalLab checked-out commit ${CHECKED_OUT_SOURCE_COMMIT} does not match pinned ${SOURCE_COMMIT}`);
}
assertSignalLabRepositoryIsClean();
if (CANONIZED_REPLAY_DETECTED_POWER_SYNTHESIS_FILTER_WIDTH_HZ
  !== SIGNAL_LAB_PRODUCTION_DETECTED_POWER_SYNTHESIS_FILTER_WIDTH_HZ) {
  throw new Error('Atomizer production detected-power synthesis filter pin does not match SignalLab');
}
const CORPUS_SOURCE_ARTIFACT_PATHS = [
  'package-lock.json',
  'package.json',
  'src/canonical-timing.ts',
  'src/catalog.ts',
  'src/classification-corpus.ts',
  'src/contracts.ts',
  'src/source-provenance.ts',
  'src/waveforms.ts',
] as const;
const EXPECTED_CORPUS_TYPESCRIPT_IMPORT_CLOSURE = [
  'src/canonical-timing.ts',
  'src/catalog.ts',
  'src/classification-corpus.ts',
  'src/contracts.ts',
  'src/source-provenance.ts',
  'src/waveforms.ts',
] as const;
assertCanonicalCorpusSourceArtifactPaths(CORPUS_SOURCE_ARTIFACT_PATHS);
assertCorpusSourceImportClosure(
  'src/classification-corpus.ts',
  EXPECTED_CORPUS_TYPESCRIPT_IMPORT_CLOSURE,
  CORPUS_SOURCE_ARTIFACT_PATHS,
);
const CORPUS_SOURCE_MANIFEST = {
  schemaVersion: 1 as const,
  hashAlgorithm: 'sha256' as const,
  artifacts: CORPUS_SOURCE_ARTIFACT_PATHS.map(corpusSourceArtifact),
};
const CORPUS_SHA256 = createHash('sha256').update(JSON.stringify(CORPUS_SOURCE_MANIFEST)).digest('hex');
const ATTEMPT_SAMPLING_CACHE_SOURCE_IDENTITY = {
  signalLabPinnedCommit: SOURCE_COMMIT,
  signalLabCheckedOutCommit: CHECKED_OUT_SOURCE_COMMIT,
  classificationCorpusVersion: CLASSIFICATION_CORPUS_VERSION,
  corpusSourceManifest: CORPUS_SOURCE_MANIFEST,
  corpusSha256: CORPUS_SHA256,
};
const STRICT_UNKNOWN_HOLDOUT_SCENARIO_IDS = [
  'unknown-impulsive',
] as const;
const OBSERVABLE_AMBIGUITY_VALIDATION_ONLY_SCENARIO_IDS = [
  'unknown-chirp',
  'unknown-regular-cw-comb-4',
  'unknown-regular-cw-comb-5',
  'unknown-irregular-cw-multitone-100-210-370k',
  'unknown-stationary-intermittent-2g4',
  'unknown-simultaneous-1mhz-raster-2g4',
  'unknown-interleaved-four-channel-2g4',
  'unknown-proprietary-off-raster-fhss-2g4',
] as const;
const EXACT_OBSERVABLE_EQUIVALENCE_NULL_SCENARIO_IDS = [
  'unknown-instrument-spur-rbw-line',
  'unknown-independent-am-equivalent-three-tone',
  'unknown-independent-fm-equivalent-bessel-comb',
  'unknown-generic-ofdm-20m',
  'unknown-generic-tdd-ofdm-10m',
  'unknown-generic-ofdm-80m',
  'unknown-proprietary-dsss-22m',
] as const;
const KNOWN_ACQUISITION_VALIDATION_ONLY_SCENARIO_IDS = [
  // A one-timeslot GSM source is deliberately time/frequency-skewed by the
  // 50 ms sample sweep and does not yield eight stable local admissions. The
  // separately canonized loaded BCCH/dummy-burst scenario fits GSM morphology.
  'gsm-900-tdma',
] as const;
// Exact scalar nulls are validation pairs, not a second physical class. Fitting
// their duplicate observations under `unknown-signal` would make the posterior
// odds depend on how many copies of an equivalent formula happen to be in the
// corpus, even though no observed scalar can distinguish the source stories.
const SCENARIO_EXCLUDED_FROM_COMPONENT_FIT_IDS = [
  ...KNOWN_ACQUISITION_VALIDATION_ONLY_SCENARIO_IDS,
  ...STRICT_UNKNOWN_HOLDOUT_SCENARIO_IDS,
  ...OBSERVABLE_AMBIGUITY_VALIDATION_ONLY_SCENARIO_IDS,
  ...EXACT_OBSERVABLE_EQUIVALENCE_NULL_SCENARIO_IDS,
] as const;
const SNR_DB = [6, 10, 16, 24, 32] as const;
const HIGH_SNR_MINIMUM_DB = 24;
// Fit across the complete pinned RBW nuisance support, then add the owned
// SignalLab production grid and session-sequence schedules as named regimes.
// The production grid cannot be represented by one honest global divisor:
// occupiedBandwidth / (recommendedSpan / 449) varies by scenario.
const RBW_DIVISORS = [12, 20, 35, 55, 80, 120] as const;
const DIVISOR_ACQUISITION_REGIMES: readonly ObservableTrainingAcquisitionRegime[] = RBW_DIVISORS.map(
  (rbwDivisor) => {
    const geometry = occupiedBandwidthRbwDivisorGeometry(rbwDivisor);
    return Object.freeze({
      id: `${geometry.id}/independent-production-branch-baselines-v1`,
      geometry,
      spectrumTemporalSchedule: OBSERVABLE_TRAINING_BASELINE_SPECTRUM_TEMPORAL_SCHEDULE,
      qualifiedEnvelopeTemporalSchedule:
        OBSERVABLE_TRAINING_BASELINE_QUALIFIED_ENVELOPE_TEMPORAL_SCHEDULE,
    });
  },
);
const SIGNAL_LAB_PRODUCTION_ACQUISITION_REGIMES: readonly ObservableTrainingAcquisitionRegime[] =
  SIGNAL_LAB_PRODUCTION_TEMPORAL_SCHEDULE_PAIRS.map((temporalSchedulePair) => Object.freeze({
    id: `${SIGNAL_LAB_PRODUCTION_ACQUISITION_GEOMETRY.id}/${temporalSchedulePair.id}`,
    geometry: SIGNAL_LAB_PRODUCTION_ACQUISITION_GEOMETRY,
    spectrumTemporalSchedule: temporalSchedulePair.spectrumTemporalSchedule,
    qualifiedEnvelopeTemporalSchedule: temporalSchedulePair.qualifiedEnvelopeTemporalSchedule,
  }));
const FITTING_ACQUISITION_REGIMES = Object.freeze([
  ...DIVISOR_ACQUISITION_REGIMES,
  ...SIGNAL_LAB_PRODUCTION_ACQUISITION_REGIMES,
]);
const SEEDS = [407, 1_407, 2_407, 3_407, 4_407, 5_407] as const;
const TAIL_CALIBRATION_RBW_DIVISORS = RBW_DIVISORS;
const TAIL_CALIBRATION_ACQUISITION_REGIMES = FITTING_ACQUISITION_REGIMES;
const TAIL_CALIBRATION_SEEDS = [6_407, 6_419, 6_421, 6_449, 6_451, 6_469, 6_473, 6_481] as const;
const SYNTHETIC_SUPPORT_RANK_REJECTION_THRESHOLD = 0.025;
// The empirical support rank is (lower-or-equal count + 1) / (n + 1).
// Forty distinct reference attempts are the minimum for that discrete rank to
// be capable of falling strictly below 0.025. This is resolution arithmetic,
// not a conformal alpha or a false-rejection/coverage guarantee.
const MINIMUM_DISTINCT_CALIBRATION_ATTEMPTS = Math.floor(1 / SYNTHETIC_SUPPORT_RANK_REJECTION_THRESHOLD);
const MINIMUM_DISTINCT_FITTING_ATTEMPTS = SEEDS.length;
const MINIMUM_FITTING_SNR_LEVELS = 2;
const MINIMUM_FITTING_RBW_DIVISORS = 2;
const MINIMUM_PRODUCTION_ACQUISITION_REGIME_HIGH_SNR_SPECTRUM_ELIGIBLE_DISTINCT_SEEDS = 1;
const MINIMUM_PRODUCTION_ACQUISITION_REGIME_HIGH_SNR_ENVELOPE_CAPTURE_DISTINCT_SEEDS = 1;
// BLE advertising is the sole sparse asynchronous exception: the swept scan
// and packet-event phase can miss one another for an entire finite horizon.
// Its gate still requires at least half of the independent event-phase seeds
// at each high-SNR level. Every other fitted scenario must cover every seed at
// at least one RBW; RBW is operator-selectable, whereas a lucky seed is not.
const HIGH_SNR_MINIMUM_SEED_COVERAGE_BY_SCENARIO: Readonly<Record<string, number>> = {
  'bluetooth-le-advertising': 0.5,
};
const SELECTION_POLICY =
  'independent-consecutive-spectrum-and-strongest-first-admission-qualified-envelope-branches-v8' as const;
const REPRESENTATIVE_WEIGHTING_POLICY =
  'view-matched-spectrum-event-envelope-causal-attempt-weighting-v4' as const;
const REPRESENTATIVE_ELIGIBILITY_POLICY = 'observation-only-hypothesis-domain-v5' as const;
// A detector/tracker acquisition attempt is the reference-score unit.
// Multiple online spectrum representatives from one attempt share the same
// synthesized noise/event phase, so flattening them would overstate the
// reference sample size. For each class and evidence view, retain exactly one
// conservative score: the minimum known-class support among every observation-domain-eligible
// representative at every ready opportunity in the attempt. Any online
// member's support rank is therefore pointwise no lower than its attempt
// minimum's rank. The fixed stratified nuisance grid is not exchangeable
// operational data, so this dominance fact does not imply conformal coverage.
const TAIL_CALIBRATION_SCORE_UNIT =
  'one-independent-branch-acquisition-attempt-score-per-evidence-view-v4' as const;
const TAIL_CALIBRATION_REPRESENTATIVE_SELECTION_POLICY =
  'consecutive-spectrum-all-runtime-representatives-and-independent-qualified-envelope-sole-capture-v4' as const;
const TAIL_CALIBRATION_REPRESENTATIVE_AGGREGATION_POLICY =
  'consecutive-spectrum-branch-minimum-qualified-envelope-branch-sole-capture-v5' as const;
const TAIL_CALIBRATION_RUNTIME_INTERPRETATION_POLICY =
  'spectrum-member-dominates-independent-branch-attempt-min-envelope-is-independent-sole-capture-v3' as const;
const TAIL_CALIBRATION_STATISTICAL_INTERPRETATION = 'empirical-synthetic-reference-only-no-exchangeability-or-coverage-guarantee-v1' as const;

assertUniqueNumbers('fitting seeds', SEEDS);
assertUniqueNumbers('tail-calibration seeds', TAIL_CALIBRATION_SEEDS);
assertDisjointNumbers('fitting seeds', SEEDS, 'tail-calibration seeds', TAIL_CALIBRATION_SEEDS);
assertUniqueNumbers('fitting RBW divisors', RBW_DIVISORS);
assertUniqueNumbers('tail-calibration RBW divisors', TAIL_CALIBRATION_RBW_DIVISORS);
assertUniqueStrings('fitting acquisition regimes', FITTING_ACQUISITION_REGIMES.map((regime) => regime.id));
assertUniqueStrings('tail-calibration acquisition regimes', TAIL_CALIBRATION_ACQUISITION_REGIMES.map((regime) => regime.id));
// RBW grids intentionally overlap so calibration covers the entire pinned
// production nuisance support. Independent seeds, not disjoint RBW values,
// isolate calibration from component fitting. Both grids remain serialized in
// trainingMatrix so the independent validator can enforce its own held-out
// RBW and temporal-source-index partitions.

interface ConsecutiveSpectrumSamplingAudit {
  attemptCount: number;
  attemptsWithAnyRepresentative: number;
  attemptsWithFitEligibleRepresentative: number;
  representativeCount: number;
  fitEligibleRepresentativeCount: number;
  fitIneligibleRepresentativeCount: number;
  provenanceUnavailableWindowCount: number;
  spectrumAcquisitionCount: number;
  sourceClockEventCount: number;
  multiRepresentativeAttemptCount: number;
  maximumRepresentativesPerAttempt: number;
  observationHorizonCounts: Record<string, number>;
  observationOpportunityCounts: Record<string, number>;
}

interface QualifiedEnvelopeSamplingAudit {
  attemptCount: number;
  receiptVerifiedDetectedPowerCaptureSampleCount: number;
  capturedEnvelopeRepresentativeCount: number;
  censoredFrequencyAgileFixedTuneCaptureCount: number;
  fitEligibleTimedCapturedEnvelopeRepresentativeCount: number;
  fitEligibleUntimedCapturedEnvelopeRepresentativeCount: number;
  provenanceUnavailableWindowCount: number;
  preCaptureProvenanceUnavailableWindowCount: number;
  postCaptureProvenanceUnavailableWindowCount: number;
  spectrumAcquisitionCount: number;
  physicalDetectedPowerCaptureCount: number;
  attemptsWithoutDetectedPowerCapture: number;
  sourceClockEventCount: number;
  observationHorizonCounts: Record<string, number>;
}

interface RepresentativeSamplingAudit {
  pairedNuisanceCellCount: number;
  consecutiveSpectrum: ConsecutiveSpectrumSamplingAudit;
  qualifiedEnvelope: QualifiedEnvelopeSamplingAudit;
}

interface ScenarioSamplingBranchAudit {
  representativeCount: number;
  fitEligibleRepresentativeCount: number;
  provenanceUnavailableWindowCount: number;
  postCaptureProvenanceUnavailableWindowCount: 0 | 1;
  spectrumAcquisitionCount: number;
  sourceClockEventCount: number;
  sourceClockTraceSha256: string;
  physicalDetectedPowerCaptureCount: 0 | 1;
  detectedPowerCaptureSampleCount: 0 | 1;
  censoredFrequencyAgileFixedTuneCaptureCount: 0 | 1;
  capturedRepresentativeKey?: string;
}

interface ScenarioSamplingAttempt {
  scenarioId: string;
  snrDb: number;
  acquisitionRegimeId: string;
  rbwDivisor: number | null;
  seed: number;
  eligibleRepresentativeCountsByView: Readonly<Record<TailCalibrationView, number>>;
  runtimeBranches: Readonly<{
    consecutiveSpectrum: ScenarioSamplingBranchAudit;
    qualifiedEnvelope: ScenarioSamplingBranchAudit;
  }>;
}

const scenarioById = new Map(canonicalClassificationScenarios.map((scenario) => [scenario.id, scenario]));
if (new Set(SCENARIO_EXCLUDED_FROM_COMPONENT_FIT_IDS).size !== SCENARIO_EXCLUDED_FROM_COMPONENT_FIT_IDS.length) {
  throw new Error('Component-fit exclusion policy contains duplicate scenario IDs');
}
for (const scenarioId of EXACT_OBSERVABLE_EQUIVALENCE_NULL_SCENARIO_IDS) {
  const scenario = scenarioById.get(scenarioId);
  if (!scenario) throw new Error(`Exact observable-equivalence null ${scenarioId} is missing from the SignalLab corpus`);
  if (scenario.truthClass !== 'unknown-signal') throw new Error(`Exact observable-equivalence null ${scenarioId} must have unknown-signal corpus truth`);
  if (!scenario.allowedObservableClasses.includes('unknown-signal')
    || !scenario.allowedObservableClasses.some((truth) => truth !== 'unknown-signal')) {
    throw new Error(`Exact observable-equivalence null ${scenarioId} must allow unknown-signal and at least one observationally equivalent known label`);
  }
}
for (const scenarioId of KNOWN_ACQUISITION_VALIDATION_ONLY_SCENARIO_IDS) {
  const scenario = scenarioById.get(scenarioId);
  if (!scenario) throw new Error(`Known acquisition-validation scenario ${scenarioId} is missing from the SignalLab corpus`);
  if (scenario.truthClass === 'unknown-signal') throw new Error(`Known acquisition-validation scenario ${scenarioId} must have known corpus truth`);
  if (!SCENARIO_EXCLUDED_FROM_COMPONENT_FIT_IDS.includes(scenarioId)) {
    throw new Error(`Known acquisition-validation scenario ${scenarioId} must be excluded from component fitting`);
  }
}
for (const scenario of canonicalClassificationScenarios) {
  if (scenario.truthClass !== 'unknown-signal') continue;
  const declaresKnownEquivalent = scenario.allowedObservableClasses.some((truth) => truth !== 'unknown-signal');
  const excluded = SCENARIO_EXCLUDED_FROM_COMPONENT_FIT_IDS.includes(
    scenario.id as typeof SCENARIO_EXCLUDED_FROM_COMPONENT_FIT_IDS[number],
  );
  if (declaresKnownEquivalent && !excluded) {
    throw new Error(`Ambiguous unknown scenario ${scenario.id} must remain validation-only instead of duplicating a known-class likelihood`);
  }
}

// Fitting and tail calibration use disjoint seeds. Precompute one phase at a
// time so the parent never retains both large representative matrices at
// once. Each attempt remains a deterministic lookup by its complete nuisance
// cell key; clearing the fitting map after the fitted components are built
// changes neither sample order nor model bytes.
const FEATURE_SAMPLING_ELIGIBLE_SCENARIOS = canonicalClassificationScenarios.filter((scenario) =>
  !SCENARIO_EXCLUDED_FROM_COMPONENT_FIT_IDS.includes(scenario.id as typeof SCENARIO_EXCLUDED_FROM_COMPONENT_FIT_IDS[number]));
assertQualifiedEnvelopeStructuralPreflight();
const NORMAL_ATTEMPT_CACHE_ROOT = resolve('.artifacts/observable-training-cache/v1');
const FRESH_SAMPLING_RUN = FRESH_SAMPLING
  ? openFreshSamplingRunJournal({
    journalPath: resolve('.artifacts/observable-training-fresh-check/journal.json'),
    runsRoot: resolve('.artifacts/observable-training-fresh-check/runs'),
    compatibilitySha256: createHash('sha256').update(JSON.stringify({
      schemaVersion: 2,
      workerRuntimeSha256: TRAINER_RUN.workerRuntimeSha256,
      trainingRuntimeIdentity: TRAINING_RUNTIME_IDENTITY,
      sourceIdentity: ATTEMPT_SAMPLING_CACHE_SOURCE_IDENTITY,
      scenarios: FEATURE_SAMPLING_ELIGIBLE_SCENARIOS,
      snrDb: SNR_DB,
      fittingRegimes: FITTING_ACQUISITION_REGIMES,
      fittingSeeds: SEEDS,
      calibrationRegimes: TAIL_CALIBRATION_ACQUISITION_REGIMES,
      calibrationSeeds: TAIL_CALIBRATION_SEEDS,
    })).digest('hex'),
  })
  : undefined;
if (FRESH_SAMPLING_RUN) {
  console.error(
    `[fresh-sampling] run=${FRESH_SAMPLING_RUN.runId} mode=${FRESH_SAMPLING_RUN.resumed ? 'resume-interrupted' : 'new-independent-check'}`,
  );
}
let precomputedFeatureSamplingAttempts = await precomputeFeatureSamplingAttempts(
  'fitting',
  FEATURE_SAMPLING_ELIGIBLE_SCENARIOS,
  SNR_DB,
  FITTING_ACQUISITION_REGIMES,
  SEEDS,
);
function lookupFeatureSamplingAttempt(
  scenario: CanonicalClassificationScenario,
  snrDb: number,
  acquisitionRegime: ObservableTrainingAcquisitionRegime,
  seed: number,
): FeatureSamplingAttempt {
  const key = samplingAttemptKey({ scenarioId: scenario.id, snrDb, acquisitionRegimeId: acquisitionRegime.id, seed });
  const attempt = precomputedFeatureSamplingAttempts.get(key);
  if (!attempt) throw new Error(`Missing precomputed feature-sampling attempt for ${key}`);
  return attempt;
}

/**
 * Exercise the two structurally special frequency-agile acquisition paths
 * before launching the complete 9,720-attempt matrix. Their classifier
 * representative is an eight-look activity association while their physical
 * tune target is the latest current raw member, so a selector regression can
 * otherwise yield a healthy spectrum population and fail only after an hour
 * without an auditable physical capture/censor result. This deterministic gate uses the pinned
 * high-SNR fitting seeds and each scenario's own production source phase; it
 * is an admission sanity check, not an additional fitting population.
 */
function assertQualifiedEnvelopeStructuralPreflight(): void {
  const scenarioIds = [
    'bluetooth-classic-connected',
    'bluetooth-le-advertising',
  ] as const;
  for (const scenarioId of scenarioIds) {
    const scenario = FEATURE_SAMPLING_ELIGIBLE_SCENARIOS.find(
      (candidate) => candidate.id === scenarioId,
    );
    const acquisitionRegime = SIGNAL_LAB_PRODUCTION_ACQUISITION_REGIMES.find(
      (candidate) =>
        candidate.qualifiedEnvelopeTemporalSchedule.sourcePlanProfileId
          === scenarioId,
    );
    if (!scenario || !acquisitionRegime) {
      throw new Error(
        `Qualified-envelope structural preflight is missing ${scenarioId} or its production source phase`,
      );
    }
    const attempts = SEEDS.map((seed) =>
      featureSamples(scenario, 32, acquisitionRegime, seed));
    const physicalCaptures = attempts.filter(
      (attempt) =>
        attempt.qualifiedEnvelope.physicalDetectedPowerCaptureCount === 1,
    ).length;
    const timedFitEligible = attempts.filter(
      (attempt) =>
        attempt.qualifiedEnvelope.detectedPowerCaptureSample?.fitEligible
          === true,
    ).length;
    const untimedFitEligible = attempts.filter(
      (attempt) =>
        attempt.qualifiedEnvelope.detectedPowerCaptureSample
          ?.envelopeUntimedFitEligible === true,
    ).length;
    const censoredFixedTuneAgileCaptures = attempts.filter(
      (attempt) => attempt.qualifiedEnvelope.detectedPowerCaptureSample
        ?.detectedPowerEvidenceDisposition
          === 'censored-frequency-agile-fixed-tune',
    ).length;
    const exactSpectrumOnlyCensoredCaptures = attempts.filter((attempt) => {
      const sample = attempt.qualifiedEnvelope.detectedPowerCaptureSample;
      return sample?.detectedPowerEvidenceDisposition
          === 'censored-frequency-agile-fixed-tune'
        && Object.keys(sample.values).every((name) =>
          !name.startsWith('envelope.'));
    }).length;
    if (physicalCaptures < 3
      || censoredFixedTuneAgileCaptures !== physicalCaptures
      || exactSpectrumOnlyCensoredCaptures !== physicalCaptures
      || timedFitEligible !== 0
      || untimedFitEligible !== 0) {
      throw new Error(
        `Qualified-envelope structural preflight failed before full sampling for ${scenarioId}: ${physicalCaptures}/${SEEDS.length} physical captures, ${censoredFixedTuneAgileCaptures} fixed-tune agile captures censored, ${exactSpectrumOnlyCensoredCaptures} exact spectrum-only censor samples, ${timedFitEligible} timed-fit and ${untimedFitEligible} untimed-fit observations`,
      );
    }
    console.error(
      `[preflight] ${scenarioId}: physical=${physicalCaptures}/${SEEDS.length} agile-envelope-censored=${censoredFixedTuneAgileCaptures} timed-fit=${timedFitEligible} untimed-fit=${untimedFitEligible}`,
    );
  }
}

type ObservableModelView = ObservableEvidenceView;
const OBSERVABLE_MODEL_VIEWS = OBSERVABLE_EVIDENCE_VIEWS;
const EXPECTED_COMPONENT_COUNTS_BY_VIEW: Readonly<Record<ObservableModelView, number>> = {
  'spectrum-only': 28,
  'envelope-untimed': 26,
  'envelope-timed': 26,
};
const EXPECTED_SCENARIO_ASSIGNMENT_COUNTS_BY_VIEW: Readonly<Record<ObservableModelView, number>> = {
  'spectrum-only': 18,
  'envelope-untimed': 16,
  'envelope-timed': 16,
};
type ViewSamples = Record<ObservableModelView, Array<Readonly<Record<string, number>>>>;
const samplesByScenarioByView = new Map<string, ViewSamples>();
const fittingRepresentativeCountsByScenarioByView = new Map<
  string,
  Record<ObservableModelView, number>
>();
const detectorConditionedFitMisses: string[] = [];
const postCaptureUnavailableFitAttempts: string[] = [];
const fitEligibilityExcludedFitAttempts: string[] = [];
const fittingSampling = emptyRepresentativeSamplingAudit();
const fittingAttempts: ScenarioSamplingAttempt[] = [];
const fittingCensoredFrequencyAgileFixedTuneCaptureCountsByScenario = new Map<string, number>();
for (const scenario of canonicalClassificationScenarios) {
  if (SCENARIO_EXCLUDED_FROM_COMPONENT_FIT_IDS.includes(scenario.id as typeof SCENARIO_EXCLUDED_FROM_COMPONENT_FIT_IDS[number])) continue;
  const samples: ViewSamples = {
    'spectrum-only': [],
    'envelope-untimed': [],
    'envelope-timed': [],
  };
  for (const snrDb of SNR_DB) {
    for (const acquisitionRegime of FITTING_ACQUISITION_REGIMES) {
      for (const seed of SEEDS) {
        try {
          const attempt = lookupFeatureSamplingAttempt(scenario, snrDb, acquisitionRegime, seed);
          recordRepresentativeSamplingAttempt(fittingSampling, attempt);
          fittingAttempts.push(scenarioSamplingAttempt(scenario, snrDb, acquisitionRegime, seed, attempt));
          const consecutiveSpectrum = attempt.consecutiveSpectrum;
          const qualifiedEnvelope = attempt.qualifiedEnvelope;
          const fitEligibleSpectrumRepresentatives = consecutiveSpectrum.onlineSpectrumRepresentatives
            .filter((representative) => representative.fitEligible)
            .map((representative) => representative.values);
          samples['spectrum-only'].push(...fitEligibleSpectrumRepresentatives);
          const captured = qualifiedEnvelope.detectedPowerCaptureSample;
          if (captured?.detectedPowerEvidenceDisposition
            === 'censored-frequency-agile-fixed-tune') {
            fittingCensoredFrequencyAgileFixedTuneCaptureCountsByScenario.set(
              scenario.id,
              (fittingCensoredFrequencyAgileFixedTuneCaptureCountsByScenario.get(scenario.id) ?? 0) + 1,
            );
          }
          if (captured?.fitEligible) samples['envelope-timed'].push(captured.values);
          if (captured) {
            const values = envelopeUntimed(captured.values);
            if (captured.envelopeUntimedFitEligible) {
              samples['envelope-untimed'].push(values);
            }
          }
          const attemptId = `${scenario.id}:snr=${snrDb}:regime=${acquisitionRegime.id}:seed=${seed}`;
          if (qualifiedEnvelope.postCaptureProvenanceUnavailableWindowCount > 0) {
            postCaptureUnavailableFitAttempts.push(`qualifiedEnvelope:${attemptId}`);
          } else if (captured === undefined) {
            detectorConditionedFitMisses.push(`qualifiedEnvelope:${attemptId}`);
          } else if (!captured.fitEligible
            && captured.detectedPowerEvidenceDisposition
              !== 'censored-frequency-agile-fixed-tune') {
            fitEligibilityExcludedFitAttempts.push(`qualifiedEnvelope:${attemptId}`);
          }
          if (consecutiveSpectrum.onlineSpectrumRepresentatives.length === 0) {
            detectorConditionedFitMisses.push(`consecutiveSpectrum:${attemptId}`);
          } else if (fitEligibleSpectrumRepresentatives.length === 0) {
            fitEligibilityExcludedFitAttempts.push(`consecutiveSpectrum:${attemptId}`);
          }
        } catch (error) {
          throw new Error(`Feature extraction failed for ${scenario.id} at SNR ${snrDb} dB, acquisition regime ${acquisitionRegime.id}, seed ${seed}`, { cause: error });
        }
      }
    }
  }
  for (const view of OBSERVABLE_MODEL_VIEWS) {
    const viewSupported = observableClassSupportsEvidenceView(
      observableModelClass(scenario),
      view,
    );
    if (viewSupported && samples[view].length < 3) {
      throw new Error(`${scenario.id} has only ${samples[view].length} detector-conditioned ${view} runtime-event training observations`);
    }
    if (!viewSupported && samples[view].length !== 0) {
      throw new Error(`${scenario.id} produced ${samples[view].length} observations for structurally censored ${view}`);
    }
  }
  const scenarioAttempts = fittingAttempts.filter((attempt) => attempt.scenarioId === scenario.id);
  assertCompleteAttemptMatrix('fitting', scenario.id, scenarioAttempts, SNR_DB, FITTING_ACQUISITION_REGIMES, SEEDS);
  for (const view of OBSERVABLE_MODEL_VIEWS) {
    if (!observableClassSupportsEvidenceView(observableModelClass(scenario), view)) continue;
    assertFittingCoverage(scenario, scenarioAttempts, view);
    assertHighSnrSeedCoverage('fitting', scenario, scenarioAttempts, SEEDS, view);
  }
  assertHighSnrProductionSpectrumEligibilityCoverage(
    'fitting',
    scenario,
    scenarioAttempts,
    SEEDS,
    SIGNAL_LAB_PRODUCTION_ACQUISITION_REGIMES,
  );
  assertHighSnrProductionQualifiedEnvelopeCaptureCoverage(
    'fitting',
    scenario,
    scenarioAttempts,
    SEEDS,
    SIGNAL_LAB_PRODUCTION_ACQUISITION_REGIMES,
  );
  samplesByScenarioByView.set(scenario.id, samples);
  fittingRepresentativeCountsByScenarioByView.set(scenario.id, {
    'spectrum-only': samples['spectrum-only'].length,
    'envelope-untimed': samples['envelope-untimed'].length,
    'envelope-timed': samples['envelope-timed'].length,
  });
}

const dimensionsByView = Object.fromEntries(OBSERVABLE_MODEL_VIEWS.map((view) => [view,
  [...new Set([...samplesByScenarioByView.values()]
    .flatMap((samples) => samples[view].flatMap((sample) => Object.keys(sample))))].sort(),
])) as Record<ObservableModelView, string[]>;
const dimensions = [...new Set(OBSERVABLE_MODEL_VIEWS.flatMap((view) => dimensionsByView[view]))].sort();
for (const [scenarioId, samples] of samplesByScenarioByView) {
  for (const view of OBSERVABLE_MODEL_VIEWS) for (const sample of samples[view]) {
    const missing = dimensionsByView[view].filter((dimension) => sample[dimension] === undefined);
    if (missing.length) throw new Error(`${scenarioId} ${view} training observation is missing ${missing.join(', ')}`);
  }
}

const prior = new Map<ObservableLeafClass, number>([
  ['cw-like', 0.08],
  ['am-dsb-full-carrier-like', 0.08],
  ['fm-angle-modulated-like', 0.08],
  ['gsm-like', 0.04],
  ['lte-fdd-like', 0.06],
  ['lte-tdd-like', 0.06],
  ['nr-fdd-like', 0.06],
  ['nr-tdd-like', 0.06],
  ['wifi-hr-dsss-like', 0.08],
  ['wifi-ofdm-like', 0.08],
  ['bluetooth-like', 0.12],
  ['unknown-signal', 0.20],
]);
const priorTotal = [...prior.values()].reduce((sum, value) => sum + value, 0);
if (Math.abs(priorTotal - 1) > 1e-12) throw new Error(`Observable class prior sums to ${priorTotal}`);

const scenariosByClass = new Map<ObservableLeafClass, CanonicalClassificationScenario[]>();
for (const scenario of canonicalClassificationScenarios) {
  if (SCENARIO_EXCLUDED_FROM_COMPONENT_FIT_IDS.includes(scenario.id as typeof SCENARIO_EXCLUDED_FROM_COMPONENT_FIT_IDS[number])) continue;
  const modelClass = observableModelClass(scenario);
  scenariosByClass.set(modelClass, [...(scenariosByClass.get(modelClass) ?? []), scenario]);
}
const fittedClassModels: ObservableClassifierModelAsset['classModels'] = OBSERVABLE_LEAF_CLASSES.map((classId) => {
  const scenarios = scenariosByClass.get(classId);
  const classPrior = prior.get(classId);
  if (!scenarios?.length || classPrior === undefined) throw new Error(`Training data/prior is missing for ${classId}`);
  // Keep Classic and LE as separate likelihood components under the honest
  // mode-ambiguous Bluetooth leaf. Pooling them can hide missing LE support
  // and fit an unphysical centroid between two acquisition regimes.
  const fitView = (view: ObservableModelView) => {
    const viewScenarios = scenarios.filter((scenario) =>
      observableClassSupportsEvidenceView(observableModelClass(scenario), view));
    return viewScenarios.flatMap((scenario) => fitScenarioStudentTComponents(
      scenario,
      samplesByScenarioByView.get(scenario.id)![view],
      dimensionsByView[view],
      -Math.log(viewScenarios.length),
    ));
  };
  const componentsByView = {
    'spectrum-only': fitView('spectrum-only'),
    'envelope-untimed': fitView('envelope-untimed'),
    'envelope-timed': fitView('envelope-timed'),
  } satisfies NonNullable<ObservableClassifierModelAsset['classModels'][number]['componentsByView']>;
  return {
    id: classId,
    logPrior: Math.log(classPrior),
    componentsByView,
  };
});
assertExactLikelihoodArchitecture(fittedClassModels);
const detectorConditionedCalibrationMisses: string[] = [];
const postCaptureUnavailableCalibrationAttempts: string[] = [];
const fitEligibilityExcludedCalibrationAttempts: string[] = [];
const calibrationSampling = emptyRepresentativeSamplingAudit();
const calibrationCensoredFrequencyAgileFixedTuneCaptureCountsByScenario = new Map<string, number>();
type TailCalibrationView = ObservableModelView;
const calibrationAttemptsByScenarioByView = new Map<string, Record<TailCalibrationView, number>>();
const calibrationAttempts: ScenarioSamplingAttempt[] = [];
precomputedFeatureSamplingAttempts.clear();
precomputedFeatureSamplingAttempts = await precomputeFeatureSamplingAttempts(
  'calibration',
  FEATURE_SAMPLING_ELIGIBLE_SCENARIOS,
  SNR_DB,
  TAIL_CALIBRATION_ACQUISITION_REGIMES,
  TAIL_CALIBRATION_SEEDS,
);
const classModels: ObservableClassifierModelAsset['classModels'] = fittedClassModels.map((model) => {
  if (model.id === 'unknown-signal') return model;
  const scenarios = scenariosByClass.get(model.id)!;
  const calibrationScoresByView = {
    'spectrum-only': [] as number[],
    'envelope-untimed': [] as number[],
    'envelope-timed': [] as number[],
  };
  for (const scenario of scenarios) {
    const scenarioCalibrationAttemptCounts: Record<TailCalibrationView, number> = {
      'spectrum-only': 0,
      'envelope-untimed': 0,
      'envelope-timed': 0,
    };
    for (const snrDb of SNR_DB) for (const acquisitionRegime of TAIL_CALIBRATION_ACQUISITION_REGIMES) for (const seed of TAIL_CALIBRATION_SEEDS) {
      try {
        const attempt = lookupFeatureSamplingAttempt(scenario, snrDb, acquisitionRegime, seed);
        recordRepresentativeSamplingAttempt(calibrationSampling, attempt);
        calibrationAttempts.push(scenarioSamplingAttempt(scenario, snrDb, acquisitionRegime, seed, attempt));
        const consecutiveSpectrum = attempt.consecutiveSpectrum;
        const qualifiedEnvelope = attempt.qualifiedEnvelope;
        const spectrumSamples = consecutiveSpectrum.onlineSpectrumRepresentatives
          .filter((sample) => sample.fitEligible)
          .map((sample) => sample.values);
        if (spectrumSamples.length > 0) {
          const spectrumComponents = modelComponentsForView(model, 'spectrum-only');
          const representativeSupportScores = spectrumSamples.map((sample) =>
            Math.max(...spectrumComponents.map((component) => studentTModelTailProbability(sample, component))));
          calibrationScoresByView['spectrum-only'].push(Math.min(...representativeSupportScores));
          scenarioCalibrationAttemptCounts['spectrum-only'] += 1;
        }
        const capturedEnvelope = qualifiedEnvelope.detectedPowerCaptureSample;
        if (capturedEnvelope?.detectedPowerEvidenceDisposition
          === 'censored-frequency-agile-fixed-tune') {
          calibrationCensoredFrequencyAgileFixedTuneCaptureCountsByScenario.set(
            scenario.id,
            (calibrationCensoredFrequencyAgileFixedTuneCaptureCountsByScenario.get(scenario.id) ?? 0) + 1,
          );
        }
        const envelopeUntimedValues = capturedEnvelope === undefined
          ? undefined
          : envelopeUntimed(capturedEnvelope.values);
        const envelopeUntimedEligible =
          capturedEnvelope?.envelopeUntimedFitEligible === true;
        if (capturedEnvelope !== undefined && envelopeUntimedValues !== undefined
          && envelopeUntimedEligible) {
          calibrationScoresByView['envelope-untimed'].push(Math.max(...modelComponentsForView(model, 'envelope-untimed').map(
            (component) => studentTModelTailProbability(envelopeUntimedValues, component),
          )));
          scenarioCalibrationAttemptCounts['envelope-untimed'] += 1;
        }
        if (capturedEnvelope?.fitEligible) {
          calibrationScoresByView['envelope-timed'].push(Math.max(...modelComponentsForView(model, 'envelope-timed').map(
            (component) => studentTModelTailProbability(capturedEnvelope.values, component),
          )));
          scenarioCalibrationAttemptCounts['envelope-timed'] += 1;
        }
        const attemptId = `${scenario.id}:snr=${snrDb}:regime=${acquisitionRegime.id}:seed=${seed}`;
        if (qualifiedEnvelope.postCaptureProvenanceUnavailableWindowCount > 0) {
          postCaptureUnavailableCalibrationAttempts.push(`qualifiedEnvelope:${attemptId}`);
        }
        if (consecutiveSpectrum.onlineSpectrumRepresentatives.length === 0) {
          detectorConditionedCalibrationMisses.push(`consecutiveSpectrum:${attemptId}`);
        } else if (spectrumSamples.length === 0) {
          fitEligibilityExcludedCalibrationAttempts.push(`consecutiveSpectrum:${attemptId}`);
        }
        if (capturedEnvelope === undefined) {
          detectorConditionedCalibrationMisses.push(`qualifiedEnvelope:${attemptId}`);
        } else if (!envelopeUntimedEligible
          && !capturedEnvelope.fitEligible
          && capturedEnvelope.detectedPowerEvidenceDisposition
            !== 'censored-frequency-agile-fixed-tune') {
          fitEligibilityExcludedCalibrationAttempts.push(`qualifiedEnvelope:${attemptId}`);
        }
      } catch (error) {
        throw new Error(`Tail calibration extraction failed for ${scenario.id} at SNR ${snrDb} dB, acquisition regime ${acquisitionRegime.id}, seed ${seed}`, { cause: error });
      }
    }
    for (const view of ['spectrum-only', 'envelope-untimed', 'envelope-timed'] as const) {
      if (!observableClassSupportsEvidenceView(observableModelClass(scenario), view)) {
        if (scenarioCalibrationAttemptCounts[view] !== 0) {
          throw new Error(`${scenario.id} produced calibration attempts for structurally censored ${view}`);
        }
        continue;
      }
      if (scenarioCalibrationAttemptCounts[view] < MINIMUM_DISTINCT_CALIBRATION_ATTEMPTS) {
        throw new Error(`${scenario.id} ${view} has only ${scenarioCalibrationAttemptCounts[view]} detector-conditioned observation-domain-eligible tail-calibration attempts`);
      }
    }
    const scenarioAttempts = calibrationAttempts.filter((attempt) => attempt.scenarioId === scenario.id);
    assertCompleteAttemptMatrix('calibration', scenario.id, scenarioAttempts, SNR_DB, TAIL_CALIBRATION_ACQUISITION_REGIMES, TAIL_CALIBRATION_SEEDS);
    for (const view of ['spectrum-only', 'envelope-untimed', 'envelope-timed'] as const) {
      if (!observableClassSupportsEvidenceView(observableModelClass(scenario), view)) continue;
      assertCalibrationCoverage(scenario, scenarioAttempts, view);
      assertHighSnrSeedCoverage(
        'calibration',
        scenario,
        scenarioAttempts,
        TAIL_CALIBRATION_SEEDS,
        view,
      );
    }
    assertHighSnrProductionSpectrumEligibilityCoverage(
      'calibration',
      scenario,
      scenarioAttempts,
      TAIL_CALIBRATION_SEEDS,
      SIGNAL_LAB_PRODUCTION_ACQUISITION_REGIMES,
    );
    assertHighSnrProductionQualifiedEnvelopeCaptureCoverage(
      'calibration',
      scenario,
      scenarioAttempts,
      TAIL_CALIBRATION_SEEDS,
      SIGNAL_LAB_PRODUCTION_ACQUISITION_REGIMES,
    );
    calibrationAttemptsByScenarioByView.set(scenario.id, scenarioCalibrationAttemptCounts);
  }
  for (const view of ['spectrum-only', 'envelope-untimed', 'envelope-timed'] as const) {
    const expectedCalibrationAttemptCount = scenarios.reduce((sum, scenario) =>
      sum + (calibrationAttemptsByScenarioByView.get(scenario.id)?.[view] ?? 0), 0);
    if (calibrationScoresByView[view].length !== expectedCalibrationAttemptCount) {
      throw new Error(`${model.id} ${view} calibration has ${calibrationScoresByView[view].length} scores for ${expectedCalibrationAttemptCount} observation-domain-eligible acquisition attempts`);
    }
    const viewSupported = observableClassSupportsEvidenceView(model.id, view);
    if (viewSupported
      && calibrationScoresByView[view].length < MINIMUM_DISTINCT_CALIBRATION_ATTEMPTS) {
      throw new Error(`${model.id} has only ${calibrationScoresByView[view].length} detector-conditioned ${view} tail-calibration attempts`);
    }
    if (!viewSupported && calibrationScoresByView[view].length !== 0) {
      throw new Error(`${model.id} has calibration scores for structurally censored ${view}`);
    }
  }
  const tailCalibrationScoresByView = {
    'spectrum-only': calibrationScoresByView['spectrum-only'].sort((left, right) => left - right),
    'envelope-untimed': calibrationScoresByView['envelope-untimed'].sort((left, right) => left - right),
    'envelope-timed': calibrationScoresByView['envelope-timed'].sort((left, right) => left - right),
  };
  return { ...model, tailCalibrationScoresByView };
});
assertExactLikelihoodArchitecture(classModels);
assertBluetoothSpectrumOnlyTrainingContract(classModels);

for (const view of OBSERVABLE_MODEL_VIEWS) {
  const serializedFittingCount = [...fittingRepresentativeCountsByScenarioByView.values()]
    .reduce((sum, counts) => sum + counts[view], 0);
  const auditedFittingCount = view === 'spectrum-only'
    ? fittingSampling.consecutiveSpectrum.fitEligibleRepresentativeCount
    : view === 'envelope-untimed'
      ? fittingSampling.qualifiedEnvelope.fitEligibleUntimedCapturedEnvelopeRepresentativeCount
      : fittingSampling.qualifiedEnvelope.fitEligibleTimedCapturedEnvelopeRepresentativeCount;
  if (serializedFittingCount !== auditedFittingCount) {
    throw new Error(`View-matched ${view} likelihood fitting count ${serializedFittingCount} does not reconcile to causal sampling total ${auditedFittingCount}`);
  }
  const serializedCalibrationCount = [...calibrationAttemptsByScenarioByView.values()]
    .reduce((sum, counts) => sum + counts[view], 0);
  const auditedCalibrationCount = calibrationAttempts.filter((attempt) =>
    attempt.eligibleRepresentativeCountsByView[view] > 0).length;
  if (serializedCalibrationCount !== auditedCalibrationCount) {
    throw new Error(`View-matched ${view} tail-calibration attempt count ${serializedCalibrationCount} does not reconcile to causal sampling total ${auditedCalibrationCount}`);
  }
}

if (fittingSampling.qualifiedEnvelope.postCaptureProvenanceUnavailableWindowCount !== 0
  || calibrationSampling.qualifiedEnvelope.postCaptureProvenanceUnavailableWindowCount !== 0
  || postCaptureUnavailableFitAttempts.length !== 0
  || postCaptureUnavailableCalibrationAttempts.length !== 0) {
  throw new Error(
    'A sole physical detected-power capture became unavailable after its identical trigger spectrum window passed extraction; refusing to shrink an envelope-view denominator',
  );
}

type ScenarioSamplingBranch = keyof ScenarioSamplingAttempt['runtimeBranches'];

const unavailableAttemptAudit = (
  attempts: readonly ScenarioSamplingAttempt[],
  branch: ScenarioSamplingBranch,
) => attempts
  .filter((attempt) => attempt.runtimeBranches[branch].provenanceUnavailableWindowCount > 0)
  .map((attempt) => ({
    attemptId: samplingAttemptKey(attempt),
    unavailableWindowCount: attempt.runtimeBranches[branch].provenanceUnavailableWindowCount,
  }))
  .sort((left, right) => left.attemptId.localeCompare(right.attemptId));
const attributedSourceClockTraceSha256 = (
  attempts: readonly ScenarioSamplingAttempt[],
  branch: ScenarioSamplingBranch,
) => createHash('sha256')
  .update(JSON.stringify(attempts.map((attempt) => ({
    attemptId: samplingAttemptKey(attempt),
    runtimeBranch: branch,
    sourceClockTraceSha256: attempt.runtimeBranches[branch].sourceClockTraceSha256,
    spectrumAcquisitionCount: attempt.runtimeBranches[branch].spectrumAcquisitionCount,
    sourceClockEventCount: attempt.runtimeBranches[branch].sourceClockEventCount,
    physicalDetectedPowerCaptureCount:
      attempt.runtimeBranches[branch].physicalDetectedPowerCaptureCount,
    postCaptureProvenanceUnavailableWindowCount:
      attempt.runtimeBranches[branch].postCaptureProvenanceUnavailableWindowCount,
    detectedPowerCaptureSampleCount:
      attempt.runtimeBranches[branch].detectedPowerCaptureSampleCount,
    censoredFrequencyAgileFixedTuneCaptureCount:
      attempt.runtimeBranches[branch].censoredFrequencyAgileFixedTuneCaptureCount,
    ...(attempt.runtimeBranches[branch].capturedRepresentativeKey === undefined
      ? {}
      : { capturedRepresentativeKey: attempt.runtimeBranches[branch].capturedRepresentativeKey }),
  })).sort((left, right) => left.attemptId.localeCompare(right.attemptId))))
  .digest('hex');

function causalSamplingAuditPartition(
  label: 'fitting' | 'tail calibration',
  sampling: RepresentativeSamplingAudit,
  attempts: readonly ScenarioSamplingAttempt[],
) {
  const spectrum = sampling.consecutiveSpectrum;
  const envelope = sampling.qualifiedEnvelope;
  if (sampling.pairedNuisanceCellCount !== attempts.length
    || spectrum.attemptCount !== attempts.length
    || envelope.attemptCount !== attempts.length) {
    throw new Error(`${label} independent branch attempt counts do not reconcile to paired nuisance cells`);
  }
  const summedBranchCount = (
    branch: ScenarioSamplingBranch,
    field: 'representativeCount'
      | 'fitEligibleRepresentativeCount'
      | 'provenanceUnavailableWindowCount'
      | 'spectrumAcquisitionCount'
      | 'sourceClockEventCount'
      | 'physicalDetectedPowerCaptureCount'
      | 'postCaptureProvenanceUnavailableWindowCount'
      | 'detectedPowerCaptureSampleCount'
      | 'censoredFrequencyAgileFixedTuneCaptureCount',
  ) => attempts.reduce((sum, attempt) => sum + attempt.runtimeBranches[branch][field], 0);
  if (summedBranchCount('consecutiveSpectrum', 'representativeCount')
      !== spectrum.representativeCount
    || summedBranchCount('consecutiveSpectrum', 'fitEligibleRepresentativeCount')
      !== spectrum.fitEligibleRepresentativeCount
    || summedBranchCount('consecutiveSpectrum', 'provenanceUnavailableWindowCount')
      !== spectrum.provenanceUnavailableWindowCount
    || summedBranchCount('consecutiveSpectrum', 'spectrumAcquisitionCount')
      !== spectrum.spectrumAcquisitionCount
    || summedBranchCount('consecutiveSpectrum', 'sourceClockEventCount')
      !== spectrum.sourceClockEventCount
    || summedBranchCount('consecutiveSpectrum', 'physicalDetectedPowerCaptureCount') !== 0
    || summedBranchCount('consecutiveSpectrum', 'postCaptureProvenanceUnavailableWindowCount') !== 0
    || summedBranchCount('consecutiveSpectrum', 'detectedPowerCaptureSampleCount') !== 0
    || summedBranchCount('consecutiveSpectrum', 'censoredFrequencyAgileFixedTuneCaptureCount') !== 0) {
    throw new Error(`${label} consecutive-spectrum attempt records do not reconcile to their branch audit`);
  }
  if (summedBranchCount('qualifiedEnvelope', 'representativeCount')
      !== envelope.capturedEnvelopeRepresentativeCount
    || summedBranchCount('qualifiedEnvelope', 'detectedPowerCaptureSampleCount')
      !== envelope.receiptVerifiedDetectedPowerCaptureSampleCount
    || summedBranchCount('qualifiedEnvelope', 'censoredFrequencyAgileFixedTuneCaptureCount')
      !== envelope.censoredFrequencyAgileFixedTuneCaptureCount
    || summedBranchCount('qualifiedEnvelope', 'fitEligibleRepresentativeCount')
      !== envelope.fitEligibleTimedCapturedEnvelopeRepresentativeCount
    || summedBranchCount('qualifiedEnvelope', 'provenanceUnavailableWindowCount')
      !== envelope.provenanceUnavailableWindowCount
    || summedBranchCount('qualifiedEnvelope', 'spectrumAcquisitionCount')
      !== envelope.spectrumAcquisitionCount
    || summedBranchCount('qualifiedEnvelope', 'sourceClockEventCount')
      !== envelope.sourceClockEventCount
    || summedBranchCount('qualifiedEnvelope', 'physicalDetectedPowerCaptureCount')
      !== envelope.physicalDetectedPowerCaptureCount
    || summedBranchCount('qualifiedEnvelope', 'postCaptureProvenanceUnavailableWindowCount')
      !== envelope.postCaptureProvenanceUnavailableWindowCount) {
    throw new Error(`${label} qualified-envelope attempt records do not reconcile to their branch audit`);
  }
  if (spectrum.sourceClockEventCount !== spectrum.spectrumAcquisitionCount) {
    throw new Error(`${label} consecutive-spectrum branch source clock includes a non-spectrum acquisition`);
  }
  if (envelope.sourceClockEventCount
    !== envelope.spectrumAcquisitionCount + envelope.physicalDetectedPowerCaptureCount) {
    throw new Error(`${label} qualified-envelope branch source clock does not account for every physical acquisition`);
  }
  if (envelope.provenanceUnavailableWindowCount
    !== envelope.preCaptureProvenanceUnavailableWindowCount
      + envelope.postCaptureProvenanceUnavailableWindowCount
    || envelope.physicalDetectedPowerCaptureCount
      !== envelope.receiptVerifiedDetectedPowerCaptureSampleCount
        + envelope.postCaptureProvenanceUnavailableWindowCount
    || envelope.receiptVerifiedDetectedPowerCaptureSampleCount
      !== envelope.capturedEnvelopeRepresentativeCount
        + envelope.censoredFrequencyAgileFixedTuneCaptureCount
    || envelope.attemptsWithoutDetectedPowerCapture
      !== envelope.attemptCount - envelope.physicalDetectedPowerCaptureCount) {
    throw new Error(`${label} qualified-envelope capture and unavailable-window accounting is inconsistent`);
  }
  if (attempts.some((attempt) =>
    attempt.runtimeBranches.consecutiveSpectrum.physicalDetectedPowerCaptureCount !== 0
      || attempt.runtimeBranches.consecutiveSpectrum.detectedPowerCaptureSampleCount !== 0
      || attempt.runtimeBranches.consecutiveSpectrum.censoredFrequencyAgileFixedTuneCaptureCount !== 0
      || attempt.runtimeBranches.consecutiveSpectrum.postCaptureProvenanceUnavailableWindowCount !== 0
      || attempt.runtimeBranches.consecutiveSpectrum.sourceClockEventCount
        !== attempt.runtimeBranches.consecutiveSpectrum.spectrumAcquisitionCount
      || attempt.runtimeBranches.qualifiedEnvelope.sourceClockEventCount
        !== attempt.runtimeBranches.qualifiedEnvelope.spectrumAcquisitionCount
          + attempt.runtimeBranches.qualifiedEnvelope.physicalDetectedPowerCaptureCount
      || attempt.runtimeBranches.qualifiedEnvelope.physicalDetectedPowerCaptureCount
        !== attempt.runtimeBranches.qualifiedEnvelope.detectedPowerCaptureSampleCount
          + attempt.runtimeBranches.qualifiedEnvelope.postCaptureProvenanceUnavailableWindowCount
      || attempt.runtimeBranches.qualifiedEnvelope.detectedPowerCaptureSampleCount
        !== attempt.runtimeBranches.qualifiedEnvelope.representativeCount
          + attempt.runtimeBranches.qualifiedEnvelope.censoredFrequencyAgileFixedTuneCaptureCount
      || attempt.runtimeBranches.qualifiedEnvelope.fitEligibleRepresentativeCount
        > attempt.runtimeBranches.qualifiedEnvelope.representativeCount)) {
    throw new Error(`${label} contains an attempt that violates its independent runtime-branch policy`);
  }
  const eligibleAttemptCountsByView = Object.fromEntries(OBSERVABLE_MODEL_VIEWS.map((view) => [
    view,
    attempts.filter((attempt) => attempt.eligibleRepresentativeCountsByView[view] > 0).length,
  ])) as Record<ObservableModelView, number>;
  if (eligibleAttemptCountsByView['spectrum-only'] !== spectrum.attemptsWithFitEligibleRepresentative
    || eligibleAttemptCountsByView['envelope-untimed']
      !== envelope.fitEligibleUntimedCapturedEnvelopeRepresentativeCount
    || eligibleAttemptCountsByView['envelope-timed']
      !== envelope.fitEligibleTimedCapturedEnvelopeRepresentativeCount) {
    throw new Error(`${label} per-view eligible attempt counts do not reconcile to branch populations`);
  }
  return {
    pairedNuisanceCellCount: sampling.pairedNuisanceCellCount,
    fitEligibleRepresentativeCountsByView: {
      'spectrum-only': spectrum.fitEligibleRepresentativeCount,
      'envelope-untimed': envelope.fitEligibleUntimedCapturedEnvelopeRepresentativeCount,
      'envelope-timed': envelope.fitEligibleTimedCapturedEnvelopeRepresentativeCount,
    },
    eligibleAttemptCountsByView,
    runtimeBranches: {
      consecutiveSpectrum: {
        detectedPowerCapturePolicyId:
          SIGNAL_LAB_PRODUCTION_SPECTRUM_DETECTED_POWER_CAPTURE_POLICY_ID,
        attemptCount: spectrum.attemptCount,
        attemptsWithAnyRepresentative: spectrum.attemptsWithAnyRepresentative,
        attemptsWithFitEligibleRepresentative: spectrum.attemptsWithFitEligibleRepresentative,
        onlineSpectrumRepresentativeCount: spectrum.representativeCount,
        fitEligibleRepresentativeCount: spectrum.fitEligibleRepresentativeCount,
        fitIneligibleRepresentativeCount: spectrum.fitIneligibleRepresentativeCount,
        provenanceUnavailableWindowCount: spectrum.provenanceUnavailableWindowCount,
        spectrumAcquisitionCount: spectrum.spectrumAcquisitionCount,
        physicalDetectedPowerCaptureCount: 0 as const,
        postCaptureProvenanceUnavailableWindowCount: 0 as const,
        detectedPowerCaptureSampleCount: 0 as const,
        censoredFrequencyAgileFixedTuneCaptureCount: 0 as const,
        sourceClockEventCount: spectrum.sourceClockEventCount,
        multiRepresentativeAttemptCount: spectrum.multiRepresentativeAttemptCount,
        maximumRepresentativesPerAttempt: spectrum.maximumRepresentativesPerAttempt,
        observationHorizonCounts: spectrum.observationHorizonCounts,
        observationOpportunityCounts: spectrum.observationOpportunityCounts,
      },
      qualifiedEnvelope: {
        detectedPowerCapturePolicyId: SIGNAL_LAB_PRODUCTION_DETECTED_POWER_CAPTURE_POLICY_ID,
        attemptCount: envelope.attemptCount,
        receiptVerifiedDetectedPowerCaptureSampleCount:
          envelope.receiptVerifiedDetectedPowerCaptureSampleCount,
        capturedEnvelopeRepresentativeCount: envelope.capturedEnvelopeRepresentativeCount,
        censoredFrequencyAgileFixedTuneCaptureCount:
          envelope.censoredFrequencyAgileFixedTuneCaptureCount,
        fitEligibleTimedCapturedEnvelopeRepresentativeCount:
          envelope.fitEligibleTimedCapturedEnvelopeRepresentativeCount,
        fitEligibleUntimedCapturedEnvelopeRepresentativeCount:
          envelope.fitEligibleUntimedCapturedEnvelopeRepresentativeCount,
        provenanceUnavailableWindowCount: envelope.provenanceUnavailableWindowCount,
        preCaptureProvenanceUnavailableWindowCount:
          envelope.preCaptureProvenanceUnavailableWindowCount,
        postCaptureProvenanceUnavailableWindowCount:
          envelope.postCaptureProvenanceUnavailableWindowCount,
        spectrumAcquisitionCount: envelope.spectrumAcquisitionCount,
        physicalDetectedPowerCaptureCount: envelope.physicalDetectedPowerCaptureCount,
        attemptsWithoutDetectedPowerCapture: envelope.attemptsWithoutDetectedPowerCapture,
        sourceClockEventCount: envelope.sourceClockEventCount,
        observationHorizonCounts: envelope.observationHorizonCounts,
      },
    },
  } as const;
}

const causalSamplingAudit = {
  schemaVersion: 3 as const,
  fitting: causalSamplingAuditPartition('fitting', fittingSampling, fittingAttempts),
  tailCalibration: causalSamplingAuditPartition(
    'tail calibration',
    calibrationSampling,
    calibrationAttempts,
  ),
  provenanceUnavailableAttemptPolicy: 'branch-attributed-exact-attempt-cell-counts-v2' as const,
  provenanceUnavailableAttempts: {
    fitting: {
      consecutiveSpectrum: unavailableAttemptAudit(fittingAttempts, 'consecutiveSpectrum'),
      qualifiedEnvelope: unavailableAttemptAudit(fittingAttempts, 'qualifiedEnvelope'),
    },
    tailCalibration: {
      consecutiveSpectrum: unavailableAttemptAudit(calibrationAttempts, 'consecutiveSpectrum'),
      qualifiedEnvelope: unavailableAttemptAudit(calibrationAttempts, 'qualifiedEnvelope'),
    },
  },
  attributedSourceClockTraceAudit: {
    hashAlgorithm: 'sha256' as const,
    serialization: 'canonical-attempt-id-branch-attributed-trace-and-capture-disposition-digest-v3' as const,
    fitting: {
      consecutiveSpectrumSha256:
        attributedSourceClockTraceSha256(fittingAttempts, 'consecutiveSpectrum'),
      qualifiedEnvelopeSha256:
        attributedSourceClockTraceSha256(fittingAttempts, 'qualifiedEnvelope'),
    },
    tailCalibration: {
      consecutiveSpectrumSha256:
        attributedSourceClockTraceSha256(calibrationAttempts, 'consecutiveSpectrum'),
      qualifiedEnvelopeSha256:
        attributedSourceClockTraceSha256(calibrationAttempts, 'qualifiedEnvelope'),
    },
  },
};

const expectedFrequencyAgileCensoredScenarioIds = [
  'bluetooth-classic-connected',
  'bluetooth-le-advertising',
] as const;
for (const [partition, counts, auditedCensoredCaptureCount] of [
  [
    'fitting',
    fittingCensoredFrequencyAgileFixedTuneCaptureCountsByScenario,
    fittingSampling.qualifiedEnvelope.censoredFrequencyAgileFixedTuneCaptureCount,
  ],
  [
    'tail calibration',
    calibrationCensoredFrequencyAgileFixedTuneCaptureCountsByScenario,
    calibrationSampling.qualifiedEnvelope.censoredFrequencyAgileFixedTuneCaptureCount,
  ],
] as const) {
  const minimumCount = partition === 'fitting'
    ? MINIMUM_DISTINCT_FITTING_ATTEMPTS
    : MINIMUM_DISTINCT_CALIBRATION_ATTEMPTS;
  const unexpected = [...counts.keys()].filter((scenarioId) =>
    !expectedFrequencyAgileCensoredScenarioIds.includes(
      scenarioId as typeof expectedFrequencyAgileCensoredScenarioIds[number],
    ));
  const missing = expectedFrequencyAgileCensoredScenarioIds.filter((scenarioId) =>
    (counts.get(scenarioId) ?? 0) < minimumCount);
  const partitionedCount = [...counts.values()].reduce((sum, count) => sum + count, 0);
  if (unexpected.length > 0 || missing.length > 0
    || partitionedCount !== auditedCensoredCaptureCount) {
    throw new Error(
      `${partition} fixed-tune agile-envelope censoring coverage is invalid (missing=${missing.join(',') || 'none'}, unexpected=${unexpected.join(',') || 'none'}, partitioned=${partitionedCount}, audited=${auditedCensoredCaptureCount})`,
    );
  }
}

const trainingMatrix = {
  attemptSamplingWorkerRuntimeSha256: TRAINER_RUN.workerRuntimeSha256,
  trainingRuntimeIdentity: TRAINING_RUNTIME_IDENTITY,
  snrDb: SNR_DB,
  rbwDivisors: RBW_DIVISORS,
  seeds: SEEDS,
  acquisitionBranchPolicy: SIGNAL_LAB_PRODUCTION_ACQUISITION_BRANCH_POLICY_ID,
  fittingAcquisitionRegimeIds: FITTING_ACQUISITION_REGIMES.map((regime) => regime.id),
  signalLabProductionAcquisitionRegime: SIGNAL_LAB_PRODUCTION_ACQUISITION_REGIME_METADATA,
  detectedPowerSynthesisFilterPolicy: OBSERVABLE_TRAINING_DETECTED_POWER_SYNTHESIS_FILTER_POLICY,
  productionAcquisitionRegimeHighSnrSeedCoveragePolicy: {
    id: 'branch-conditional-production-regime-presence-v2',
    spectrumOnly: {
      minimumDistinctObservationDomainEligibleSeedsPerHighSnrCell:
        MINIMUM_PRODUCTION_ACQUISITION_REGIME_HIGH_SNR_SPECTRUM_ELIGIBLE_DISTINCT_SEEDS,
    },
    qualifiedEnvelope: {
      minimumDistinctPhysicalCaptureSeedsPerHighSnrCell:
        MINIMUM_PRODUCTION_ACQUISITION_REGIME_HIGH_SNR_ENVELOPE_CAPTURE_DISTINCT_SEEDS,
      observationDomainEligibilityPolicy:
        'pooled-by-scenario-and-view-after-causal-capture-v1',
      outOfDomainCapturePolicy:
        'honest-abstention-excluded-from-envelope-likelihood-v1',
    },
    globalCoveragePolicy: 'all-seeds-at-one-or-more-regimes-except-declared-sparse-asynchronous-scenarios-v1',
  },
  classificationSweeps: CLASSIFICATION_SWEEPS,
  observationOpportunityHorizons: {
    standard: STANDARD_OBSERVATION_OPPORTUNITIES,
    fullBand2g4: FULL_BAND_2G4_OBSERVATION_OPPORTUNITIES,
  },
  // The validator uses both explicit fitting and calibration grids to prove
  // its nuisance seeds/RBWs/source-clock schedule are held out. Fit/calibration
  // regimes intentionally overlap; their seed arrays are asserted disjoint.
  tailCalibrationSeeds: TAIL_CALIBRATION_SEEDS,
  tailCalibrationRbwDivisors: TAIL_CALIBRATION_RBW_DIVISORS,
  tailCalibrationAcquisitionRegimeIds: TAIL_CALIBRATION_ACQUISITION_REGIMES.map((regime) => regime.id),
  tailCalibrationScoreUnit: TAIL_CALIBRATION_SCORE_UNIT,
  tailCalibrationRepresentativeSelectionPolicy: TAIL_CALIBRATION_REPRESENTATIVE_SELECTION_POLICY,
  tailCalibrationRepresentativeAggregationPolicy: TAIL_CALIBRATION_REPRESENTATIVE_AGGREGATION_POLICY,
  tailCalibrationRuntimeInterpretationPolicy: TAIL_CALIBRATION_RUNTIME_INTERPRETATION_POLICY,
  tailCalibrationStatisticalInterpretation: TAIL_CALIBRATION_STATISTICAL_INTERPRETATION,
  tailCalibrationAttemptCountsByScenarioByView: Object.fromEntries(calibrationAttemptsByScenarioByView),
  fittingCapturedEnvelopeCountsByScenario: Object.fromEntries(
    [...fittingRepresentativeCountsByScenarioByView]
      .map(([scenarioId, counts]) => [scenarioId, counts['envelope-timed']]),
  ),
  fittingRepresentativeCountsByScenarioByView:
    Object.fromEntries(fittingRepresentativeCountsByScenarioByView),
  likelihoodPopulationPolicy: 'independent-branch-view-matched-runtime-event-populations-v3',
  likelihoodComponentDecompositionPolicy:
    OBSERVABLE_LIKELIHOOD_COMPONENT_DECOMPOSITION_POLICY,
  frequencyAgileFixedTuneEnvelopeCensoringPolicy:
    OBSERVABLE_EVIDENCE_CENSORING_POLICY,
  censoredFrequencyAgileFixedTuneCaptureCountsByScenario: {
    fitting: Object.fromEntries(
      fittingCensoredFrequencyAgileFixedTuneCaptureCountsByScenario,
    ),
    tailCalibration: Object.fromEntries(
      calibrationCensoredFrequencyAgileFixedTuneCaptureCountsByScenario,
    ),
  },
  detectedPowerAcquisitionQualification:
    DETECTED_POWER_ACQUISITION_QUALIFICATION,
  causalSamplingAudit,
  detectorConditionedFitMisses,
  detectorConditionedCalibrationMisses,
  postCaptureUnavailableFitAttempts,
  postCaptureUnavailableCalibrationAttempts,
  fitEligibilityExcludedFitAttempts,
  fitEligibilityExcludedCalibrationAttempts,
  scenarioExcludedFromComponentFitIds: SCENARIO_EXCLUDED_FROM_COMPONENT_FIT_IDS,
  exactObservableEquivalenceNullScenarioIds: EXACT_OBSERVABLE_EQUIVALENCE_NULL_SCENARIO_IDS,
  knownAcquisitionValidationOnlyScenarioIds: KNOWN_ACQUISITION_VALIDATION_ONLY_SCENARIO_IDS,
  selectionPolicy: SELECTION_POLICY,
  representativeWeightingPolicy: REPRESENTATIVE_WEIGHTING_POLICY,
  representativeEligibilityPolicy: REPRESENTATIVE_ELIGIBILITY_POLICY,
} as const;

const asset: ObservableClassifierModelAsset = {
  id: 'bayesian-observable-equivalence-v8',
  corpusVersion: CLASSIFICATION_CORPUS_VERSION,
  sourceCommit: SOURCE_COMMIT,
  corpusSourceManifest: CORPUS_SOURCE_MANIFEST,
  corpusSha256: CORPUS_SHA256,
  preprocessing: 'scalar-observable-features-v7',
  priorId: 'engineering-design-class-weights-v1',
  calibrationId: 'synthetic-independent-branch-view-matched-causal-acquisition-support-rank-detector-conditioned-physical-uncalibrated-v19',
  generatedAt: '2026-07-14T00:00:00.000Z',
  dimensions,
  trainingMatrix,
  classModels,
};

const modelContentSha256 = createHash('sha256')
  .update(JSON.stringify(asset))
  .digest('hex');
const source = `/* Generated by tools/train-observable-classifier.ts; do not hand edit. */\n`
  + `import type { ObservableClassifierModelAsset } from '../observable-classifier-model.js';\n\n`
  + `export const BAYESIAN_OBSERVABLE_MODEL_CONTENT_SHA256 = '${modelContentSha256}' as const;\n`
  + `export const BAYESIAN_OBSERVABLE_MODEL: ObservableClassifierModelAsset = ${JSON.stringify(asset, null, 2)};\n`;
const modelAssetSha256 = createHash('sha256').update(source).digest('hex');
const manifestSource = `/* Generated by tools/train-observable-classifier.ts; do not hand edit. */\n`
  + `export const BAYESIAN_OBSERVABLE_MODEL_SHA256 = '${modelAssetSha256}' as const;\n`
  + `export const BAYESIAN_OBSERVABLE_MODEL_CONTENT_SHA256 = '${modelContentSha256}' as const;\n`;
TRAINER_RUN.assertWorkerRuntimeUnchanged();
if (CHECK_ONLY) {
  assertGeneratedModelManifestPair(OUTPUT, MANIFEST_OUTPUT);
  assertGeneratedAssetIsCurrent(OUTPUT, source, 'classifier model');
  assertGeneratedAssetIsCurrent(MANIFEST_OUTPUT, manifestSource, 'classifier model manifest');
} else {
  publishGeneratedModelManifestRecoverably({
    modelPath: OUTPUT,
    manifestPath: MANIFEST_OUTPUT,
    journalPath: MODEL_PUBLICATION_JOURNAL,
    modelSource: source,
    manifestSource,
  });
}
FRESH_SAMPLING_RUN?.markCompleted();
console.log(JSON.stringify({
  mode: CHECK_ONLY ? 'verified-byte-identical' : 'generated',
  output: OUTPUT,
  manifest: MANIFEST_OUTPUT,
  modelAssetSha256,
  modelContentSha256,
  classes: classModels.length,
  componentsByView: Object.fromEntries(OBSERVABLE_MODEL_VIEWS.map((view) => [
    view,
    classModels.reduce((sum, model) => sum + modelComponentsForView(model, view).length, 0),
  ])),
  dimensions: dimensions.length,
  dimensionsByView: Object.fromEntries(OBSERVABLE_MODEL_VIEWS.map((view) => [view, dimensionsByView[view].length])),
  fittingExamplesByView: Object.fromEntries(OBSERVABLE_MODEL_VIEWS.map((view) => [
    view,
    [...samplesByScenarioByView.values()].reduce((sum, samples) => sum + samples[view].length, 0),
  ])),
  fittingExamplesByScenarioByView: Object.fromEntries(fittingRepresentativeCountsByScenarioByView),
  calibrationAttemptsByScenarioByView: Object.fromEntries(calibrationAttemptsByScenarioByView),
  fittingAttemptCoverageByScenario: attemptCoverageByScenario(fittingAttempts),
  calibrationAttemptCoverageByScenario: attemptCoverageByScenario(calibrationAttempts),
  tailCalibrationAttemptScoresByView: Object.fromEntries(
    (['spectrum-only', 'envelope-untimed', 'envelope-timed'] as const).map((view) => [
      view,
      classModels.reduce((sum, model) => sum + (model.tailCalibrationScoresByView?.[view].length ?? 0), 0),
    ]),
  ),
  tailCalibrationScoreUnit: TAIL_CALIBRATION_SCORE_UNIT,
  tailCalibrationRepresentativeSelectionPolicy: TAIL_CALIBRATION_REPRESENTATIVE_SELECTION_POLICY,
  tailCalibrationRepresentativeAggregationPolicy: TAIL_CALIBRATION_REPRESENTATIVE_AGGREGATION_POLICY,
  tailCalibrationRuntimeInterpretationPolicy: TAIL_CALIBRATION_RUNTIME_INTERPRETATION_POLICY,
  tailCalibrationStatisticalInterpretation: TAIL_CALIBRATION_STATISTICAL_INTERPRETATION,
  representativeWeightingPolicy: REPRESENTATIVE_WEIGHTING_POLICY,
  coverageGates: {
    highSnrMinimumDb: HIGH_SNR_MINIMUM_DB,
    defaultHighSnrMinimumSeedCoverage: 1,
    highSnrMinimumSeedCoverageByScenario: HIGH_SNR_MINIMUM_SEED_COVERAGE_BY_SCENARIO,
    minimumDistinctFittingAttempts: MINIMUM_DISTINCT_FITTING_ATTEMPTS,
    minimumFittingSnrLevels: MINIMUM_FITTING_SNR_LEVELS,
    minimumFittingRbwDivisors: MINIMUM_FITTING_RBW_DIVISORS,
    syntheticSupportRankRejectionThreshold: SYNTHETIC_SUPPORT_RANK_REJECTION_THRESHOLD,
    minimumDistinctCalibrationAttempts: MINIMUM_DISTINCT_CALIBRATION_ATTEMPTS,
  },
  fittingSampling,
  calibrationSampling,
  causalSamplingAudit,
}, null, 2));

function assertGeneratedAssetIsCurrent(file: string, regenerated: string, label: string): void {
  const checkedIn = readFileSync(file, 'utf8');
  if (checkedIn === regenerated) return;
  const checkedInSha256 = createHash('sha256').update(checkedIn).digest('hex');
  const regeneratedSha256 = createHash('sha256').update(regenerated).digest('hex');
  throw new Error(
    `Checked-in ${label} is stale: ${checkedInSha256} != deterministic regeneration ${regeneratedSha256}. Run npm run train:signal-classifier and commit both generated assets.`,
  );
}

function assertCanonicalCorpusSourceArtifactPaths(paths: readonly string[]): void {
  if (new Set(paths).size !== paths.length) throw new Error('SignalLab corpus source manifest contains duplicate artifact paths');
  for (const path of paths) assertRepositoryRelativePath(path, 'SignalLab corpus source artifact');
  const canonical = [...paths].sort((left, right) => left.localeCompare(right));
  if (paths.some((path, index) => path !== canonical[index])) throw new Error('SignalLab corpus source manifest paths must be in canonical lexical order');
}

function assertRepositoryRelativePath(path: string, label: string): void {
  if (!path || path.includes('\\') || posix.isAbsolute(path) || posix.normalize(path) !== path
    || path === '..' || path.startsWith('../') || path.includes('/../')) {
    throw new Error(`${label} ${JSON.stringify(path)} must be a canonical repository-relative POSIX path`);
  }
}

function corpusSourceArtifact(path: string): { path: string; sha256: string } {
  const file = resolve(SIGNAL_LAB_REPOSITORY_ROOT, path);
  const status = lstatSync(file);
  if (!status.isFile() || status.isSymbolicLink()) throw new Error(`SignalLab corpus source artifact ${path} must be a regular non-symlink file`);
  gitOutput(['ls-files', '--error-unmatch', '--', path]);
  const bytes = readFileSync(file);
  const committedBytes = gitOutput(['show', `${SOURCE_COMMIT}:${path}`]);
  if (!bytes.equals(committedBytes)) {
    throw new Error(`SignalLab corpus source artifact ${path} differs from pinned commit ${SOURCE_COMMIT}`);
  }
  return { path, sha256: createHash('sha256').update(bytes).digest('hex') };
}

function assertSignalLabRepositoryIsClean(): void {
  const status = gitOutput(['status', '--porcelain=v1', '-z', '--untracked-files=all']);
  if (status.length !== 0) {
    throw new Error('SignalLab repository must have a clean index and worktree, including no untracked files, before classifier generation');
  }
}

function assertCorpusSourceImportClosure(
  entryPath: string,
  expectedPaths: readonly string[],
  manifestPaths: readonly string[],
): void {
  const discovered = new Set<string>();
  const pending = [entryPath];
  while (pending.length > 0) {
    const path = pending.pop()!;
    assertRepositoryRelativePath(path, 'SignalLab corpus import');
    if (discovered.has(path)) continue;
    discovered.add(path);
    const source = readFileSync(resolve(SIGNAL_LAB_REPOSITORY_ROOT, path), 'utf8');
    for (const specifier of relativeTypeScriptModuleSpecifiers(source)) {
      if (!specifier.endsWith('.js') && !specifier.endsWith('.ts')) {
        throw new Error(`SignalLab corpus import ${JSON.stringify(specifier)} from ${path} must declare a .js or .ts TypeScript module target`);
      }
      const resolvedPath = posix.normalize(posix.join(
        posix.dirname(path),
        specifier.endsWith('.js') ? `${specifier.slice(0, -3)}.ts` : specifier,
      ));
      assertRepositoryRelativePath(resolvedPath, 'Resolved SignalLab corpus import');
      pending.push(resolvedPath);
    }
  }
  const actual = [...discovered].sort();
  if (JSON.stringify(actual) !== JSON.stringify(expectedPaths)) {
    throw new Error(`SignalLab corpus TypeScript import closure ${JSON.stringify(actual)} does not match pinned ${JSON.stringify(expectedPaths)}`);
  }
  const manifest = new Set(manifestPaths);
  const omitted = actual.filter((path) => !manifest.has(path));
  if (omitted.length > 0) throw new Error(`SignalLab corpus source manifest omits import-closure artifacts: ${omitted.join(', ')}`);
}

function relativeTypeScriptModuleSpecifiers(source: string): string[] {
  const patterns = [
    /\b(?:import|export)\s+(?:type\s+)?(?:[^'\";]*?\s+from\s+)?['\"](\.[^'\"]+)['\"]/g,
    /\bimport\s*\(\s*['\"](\.[^'\"]+)['\"]/g,
    /\brequire\s*\(\s*['\"](\.[^'\"]+)['\"]/g,
  ];
  return [...new Set(patterns.flatMap((pattern) => [...source.matchAll(pattern)].map((match) => match[1]!)))].sort();
}

function gitOutput(arguments_: readonly string[]): Buffer {
  return execFileSync('git', arguments_, {
    cwd: SIGNAL_LAB_REPOSITORY_ROOT,
    encoding: 'buffer',
    maxBuffer: 16 * 1024 * 1024,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
}

function spectrumOnly(sample: Readonly<Record<string, number>>): Readonly<Record<string, number>> {
  return Object.fromEntries(Object.entries(sample).filter(([name]) => !name.startsWith('envelope.')));
}

function modelComponentsForView(
  model: ObservableClassifierModelAsset['classModels'][number],
  view: ObservableModelView,
) {
  return observableModelComponents(model, view);
}

function assertExactLikelihoodArchitecture(
  models: ObservableClassifierModelAsset['classModels'],
): void {
  for (const view of OBSERVABLE_MODEL_VIEWS) {
    const components = models.flatMap((model) =>
      model.componentsByView?.[view] ?? []);
    const sourceScenarioIds = new Set(components.map(componentSourceScenarioId));
    if (components.length !== EXPECTED_COMPONENT_COUNTS_BY_VIEW[view]
      || sourceScenarioIds.size
        !== EXPECTED_SCENARIO_ASSIGNMENT_COUNTS_BY_VIEW[view]) {
      throw new Error(
        `${view} likelihood architecture has ${components.length} components from ${sourceScenarioIds.size} scenarios; expected ${EXPECTED_COMPONENT_COUNTS_BY_VIEW[view]} components from ${EXPECTED_SCENARIO_ASSIGNMENT_COUNTS_BY_VIEW[view]} scenarios`,
      );
    }
  }
}

function assertBluetoothSpectrumOnlyTrainingContract(
  models: ObservableClassifierModelAsset['classModels'],
): void {
  const bluetoothScenarioIds = [
    'bluetooth-classic-connected',
    'bluetooth-le-advertising',
  ] as const;
  for (const scenarioId of bluetoothScenarioIds) {
    const fittingCounts = fittingRepresentativeCountsByScenarioByView.get(scenarioId);
    const calibrationCounts = calibrationAttemptsByScenarioByView.get(scenarioId);
    if (!fittingCounts || !calibrationCounts
      || fittingCounts['spectrum-only'] <= 0
      || calibrationCounts['spectrum-only'] <= 0
      || fittingCounts['envelope-untimed'] !== 0
      || fittingCounts['envelope-timed'] !== 0
      || calibrationCounts['envelope-untimed'] !== 0
      || calibrationCounts['envelope-timed'] !== 0) {
      throw new Error(
        `${scenarioId} must contribute positive spectrum-only support and exactly zero envelope fitting/calibration support`,
      );
    }
  }
  const model = models.find((candidate) => candidate.id === 'bluetooth-like');
  if (!model) throw new Error('Bluetooth likelihood model is missing');
  const spectrumComponents = model.componentsByView?.['spectrum-only'] ?? [];
  const untimedComponents = model.componentsByView?.['envelope-untimed'] ?? [];
  const timedComponents = model.componentsByView?.['envelope-timed'] ?? [];
  const spectrumScores = model.tailCalibrationScoresByView?.['spectrum-only'] ?? [];
  const untimedScores = model.tailCalibrationScoresByView?.['envelope-untimed'] ?? [];
  const timedScores = model.tailCalibrationScoresByView?.['envelope-timed'] ?? [];
  if (spectrumComponents.length !== bluetoothScenarioIds.length
    || new Set(spectrumComponents.map(componentSourceScenarioId)).size
      !== bluetoothScenarioIds.length
    || spectrumScores.length === 0
    || untimedComponents.length !== 0
    || timedComponents.length !== 0
    || untimedScores.length !== 0
    || timedScores.length !== 0) {
    throw new Error(
      'Bluetooth must serialize exactly its two spectrum scenario components and no envelope components or calibration scores',
    );
  }
}

function emptyRepresentativeSamplingAudit(): RepresentativeSamplingAudit {
  return {
    pairedNuisanceCellCount: 0,
    consecutiveSpectrum: {
      attemptCount: 0,
      attemptsWithAnyRepresentative: 0,
      attemptsWithFitEligibleRepresentative: 0,
      representativeCount: 0,
      fitEligibleRepresentativeCount: 0,
      fitIneligibleRepresentativeCount: 0,
      provenanceUnavailableWindowCount: 0,
      spectrumAcquisitionCount: 0,
      sourceClockEventCount: 0,
      multiRepresentativeAttemptCount: 0,
      maximumRepresentativesPerAttempt: 0,
      observationHorizonCounts: {},
      observationOpportunityCounts: {},
    },
    qualifiedEnvelope: {
      attemptCount: 0,
      receiptVerifiedDetectedPowerCaptureSampleCount: 0,
      capturedEnvelopeRepresentativeCount: 0,
      censoredFrequencyAgileFixedTuneCaptureCount: 0,
      fitEligibleTimedCapturedEnvelopeRepresentativeCount: 0,
      fitEligibleUntimedCapturedEnvelopeRepresentativeCount: 0,
      provenanceUnavailableWindowCount: 0,
      preCaptureProvenanceUnavailableWindowCount: 0,
      postCaptureProvenanceUnavailableWindowCount: 0,
      spectrumAcquisitionCount: 0,
      physicalDetectedPowerCaptureCount: 0,
      attemptsWithoutDetectedPowerCapture: 0,
      sourceClockEventCount: 0,
      observationHorizonCounts: {},
    },
  };
}

function recordRepresentativeSamplingAttempt(audit: RepresentativeSamplingAudit, attempt: FeatureSamplingAttempt): void {
  const spectrumAttempt = attempt.consecutiveSpectrum;
  const envelopeAttempt = attempt.qualifiedEnvelope;
  const representativeCount = spectrumAttempt.onlineSpectrumRepresentatives.length;
  const eligibleCount = spectrumAttempt.onlineSpectrumRepresentatives
    .filter((sample) => sample.fitEligible).length;
  audit.pairedNuisanceCellCount += 1;
  audit.consecutiveSpectrum.attemptCount += 1;
  if (representativeCount > 0) audit.consecutiveSpectrum.attemptsWithAnyRepresentative += 1;
  if (eligibleCount > 0) audit.consecutiveSpectrum.attemptsWithFitEligibleRepresentative += 1;
  if (representativeCount > 1) audit.consecutiveSpectrum.multiRepresentativeAttemptCount += 1;
  audit.consecutiveSpectrum.representativeCount += representativeCount;
  audit.consecutiveSpectrum.fitEligibleRepresentativeCount += eligibleCount;
  audit.consecutiveSpectrum.fitIneligibleRepresentativeCount += representativeCount - eligibleCount;
  audit.consecutiveSpectrum.provenanceUnavailableWindowCount +=
    spectrumAttempt.provenanceUnavailableWindowCount;
  audit.consecutiveSpectrum.spectrumAcquisitionCount += spectrumAttempt.observationHorizon;
  audit.consecutiveSpectrum.sourceClockEventCount += spectrumAttempt.sourceClockEventCount;
  audit.consecutiveSpectrum.maximumRepresentativesPerAttempt = Math.max(
    audit.consecutiveSpectrum.maximumRepresentativesPerAttempt,
    representativeCount,
  );
  audit.consecutiveSpectrum.observationHorizonCounts[spectrumAttempt.observationHorizon] =
    (audit.consecutiveSpectrum.observationHorizonCounts[spectrumAttempt.observationHorizon] ?? 0) + 1;
  for (const sample of spectrumAttempt.onlineSpectrumRepresentatives) {
    audit.consecutiveSpectrum.observationOpportunityCounts[sample.observationOpportunity] =
      (audit.consecutiveSpectrum.observationOpportunityCounts[sample.observationOpportunity] ?? 0) + 1;
  }

  audit.qualifiedEnvelope.attemptCount += 1;
  const detectedPowerCaptureSample = envelopeAttempt.detectedPowerCaptureSample;
  if (detectedPowerCaptureSample !== undefined) {
    audit.qualifiedEnvelope.receiptVerifiedDetectedPowerCaptureSampleCount += 1;
    if (detectedPowerCaptureSample.detectedPowerEvidenceDisposition
      === 'admitted-envelope') {
      if (!Object.keys(detectedPowerCaptureSample.values).some((name) =>
        name.startsWith('envelope.'))) {
        throw new Error(
          'An admitted detected-power result has no envelope feature population',
        );
      }
      audit.qualifiedEnvelope.capturedEnvelopeRepresentativeCount += 1;
    } else if (detectedPowerCaptureSample.detectedPowerEvidenceDisposition
      === 'censored-frequency-agile-fixed-tune') {
      audit.qualifiedEnvelope.censoredFrequencyAgileFixedTuneCaptureCount += 1;
    } else {
      throw new Error(
        'A receipt-verified detected-power sample has no declared evidence disposition',
      );
    }
    if (detectedPowerCaptureSample.fitEligible) {
      audit.qualifiedEnvelope.fitEligibleTimedCapturedEnvelopeRepresentativeCount += 1;
    }
    if (detectedPowerCaptureSample.envelopeUntimedFitEligible) {
      audit.qualifiedEnvelope.fitEligibleUntimedCapturedEnvelopeRepresentativeCount += 1;
    }
    if (detectedPowerCaptureSample.detectedPowerEvidenceDisposition
      === 'censored-frequency-agile-fixed-tune'
      && (detectedPowerCaptureSample.fitEligible
        || detectedPowerCaptureSample.envelopeUntimedFitEligible
        || Object.keys(detectedPowerCaptureSample.values).some((name) =>
          name.startsWith('envelope.')))) {
      throw new Error(
        'A censored fixed-tune agile capture leaked into an envelope likelihood population',
      );
    }
  }
  audit.qualifiedEnvelope.provenanceUnavailableWindowCount +=
    envelopeAttempt.provenanceUnavailableWindowCount;
  audit.qualifiedEnvelope.preCaptureProvenanceUnavailableWindowCount +=
    envelopeAttempt.preCaptureProvenanceUnavailableWindowCount;
  audit.qualifiedEnvelope.postCaptureProvenanceUnavailableWindowCount +=
    envelopeAttempt.postCaptureProvenanceUnavailableWindowCount;
  audit.qualifiedEnvelope.spectrumAcquisitionCount += envelopeAttempt.observationHorizon;
  audit.qualifiedEnvelope.physicalDetectedPowerCaptureCount +=
    envelopeAttempt.physicalDetectedPowerCaptureCount;
  if (envelopeAttempt.physicalDetectedPowerCaptureCount === 0) {
    audit.qualifiedEnvelope.attemptsWithoutDetectedPowerCapture += 1;
  }
  audit.qualifiedEnvelope.sourceClockEventCount += envelopeAttempt.sourceClockEventCount;
  audit.qualifiedEnvelope.observationHorizonCounts[envelopeAttempt.observationHorizon] =
    (audit.qualifiedEnvelope.observationHorizonCounts[envelopeAttempt.observationHorizon] ?? 0) + 1;
}

function scenarioSamplingAttempt(
  scenario: CanonicalClassificationScenario,
  snrDb: number,
  acquisitionRegime: ObservableTrainingAcquisitionRegime,
  seed: number,
  attempt: FeatureSamplingAttempt,
): ScenarioSamplingAttempt {
  const spectrumAttempt = attempt.consecutiveSpectrum;
  const envelopeAttempt = attempt.qualifiedEnvelope;
  const capturedEnvelope = envelopeAttempt.detectedPowerCaptureSample;
  const envelopeAdmitted = capturedEnvelope?.detectedPowerEvidenceDisposition
    === 'admitted-envelope';
  const envelopeCensored = capturedEnvelope?.detectedPowerEvidenceDisposition
    === 'censored-frequency-agile-fixed-tune';
  const spectrumFitEligibleCount = spectrumAttempt.onlineSpectrumRepresentatives
    .filter((sample) => sample.fitEligible).length;
  return {
    scenarioId: scenario.id,
    snrDb,
    acquisitionRegimeId: acquisitionRegime.id,
    rbwDivisor: acquisitionRegime.geometry.kind === 'occupied-bandwidth-rbw-divisor'
      ? acquisitionRegime.geometry.rbwDivisor
      : null,
    seed,
    eligibleRepresentativeCountsByView: {
      'spectrum-only': spectrumFitEligibleCount,
      'envelope-untimed': capturedEnvelope?.envelopeUntimedFitEligible ? 1 : 0,
      'envelope-timed': capturedEnvelope?.fitEligible ? 1 : 0,
    },
    runtimeBranches: {
      consecutiveSpectrum: {
        representativeCount: spectrumAttempt.onlineSpectrumRepresentatives.length,
        fitEligibleRepresentativeCount: spectrumFitEligibleCount,
        provenanceUnavailableWindowCount: spectrumAttempt.provenanceUnavailableWindowCount,
        postCaptureProvenanceUnavailableWindowCount: 0,
        spectrumAcquisitionCount: spectrumAttempt.observationHorizon,
        sourceClockEventCount: spectrumAttempt.sourceClockEventCount,
        sourceClockTraceSha256: spectrumAttempt.sourceClockTraceSha256,
        physicalDetectedPowerCaptureCount: 0,
        detectedPowerCaptureSampleCount: 0,
        censoredFrequencyAgileFixedTuneCaptureCount: 0,
      },
      qualifiedEnvelope: {
        representativeCount: envelopeAdmitted ? 1 : 0,
        fitEligibleRepresentativeCount: capturedEnvelope?.fitEligible ? 1 : 0,
        provenanceUnavailableWindowCount: envelopeAttempt.provenanceUnavailableWindowCount,
        postCaptureProvenanceUnavailableWindowCount:
          envelopeAttempt.postCaptureProvenanceUnavailableWindowCount,
        spectrumAcquisitionCount: envelopeAttempt.observationHorizon,
        sourceClockEventCount: envelopeAttempt.sourceClockEventCount,
        sourceClockTraceSha256: envelopeAttempt.sourceClockTraceSha256,
        physicalDetectedPowerCaptureCount: envelopeAttempt.physicalDetectedPowerCaptureCount,
        detectedPowerCaptureSampleCount: capturedEnvelope === undefined ? 0 : 1,
        censoredFrequencyAgileFixedTuneCaptureCount: envelopeCensored ? 1 : 0,
        ...(envelopeAttempt.capturedRepresentativeKey === undefined
          ? {}
          : { capturedRepresentativeKey: envelopeAttempt.capturedRepresentativeKey }),
      },
    },
  };
}

function assertCompleteAttemptMatrix(
  purpose: 'fitting' | 'calibration',
  scenarioId: string,
  attempts: readonly ScenarioSamplingAttempt[],
  snrLevels: readonly number[],
  acquisitionRegimes: readonly ObservableTrainingAcquisitionRegime[],
  seeds: readonly number[],
): void {
  const keys = attempts.map(samplingAttemptKey);
  const uniqueKeys = new Set(keys);
  if (uniqueKeys.size !== attempts.length) {
    throw new Error(`${scenarioId} ${purpose} matrix contains ${attempts.length - uniqueKeys.size} duplicate acquisition attempts`);
  }
  const expectedKeys = snrLevels.flatMap((snrDb) => acquisitionRegimes.flatMap((acquisitionRegime) => seeds.map((seed) =>
    samplingAttemptKey({ scenarioId, snrDb, acquisitionRegimeId: acquisitionRegime.id, seed }))));
  const missingKeys = expectedKeys.filter((key) => !uniqueKeys.has(key));
  const unexpectedKeys = keys.filter((key) => !expectedKeys.includes(key));
  if (missingKeys.length > 0 || unexpectedKeys.length > 0) {
    throw new Error(`${scenarioId} ${purpose} matrix is incomplete (missing=${missingKeys.length}, unexpected=${unexpectedKeys.length})`);
  }
}

function assertFittingCoverage(
  scenario: CanonicalClassificationScenario,
  attempts: readonly ScenarioSamplingAttempt[],
  view: TailCalibrationView,
): void {
  const eligibleAttempts = attempts.filter((attempt) =>
    attempt.eligibleRepresentativeCountsByView[view] > 0);
  if (eligibleAttempts.length < MINIMUM_DISTINCT_FITTING_ATTEMPTS) {
    throw new Error(`${scenario.id} ${view} has only ${eligibleAttempts.length} distinct observation-domain-eligible acquisition attempts; expected at least one ${MINIMUM_DISTINCT_FITTING_ATTEMPTS}-seed block`);
  }
  const coveredSnrLevels = new Set(eligibleAttempts.map((attempt) => attempt.snrDb));
  const coveredRbwDivisors = new Set(eligibleAttempts
    .map((attempt) => attempt.rbwDivisor)
    .filter((rbwDivisor): rbwDivisor is number => rbwDivisor !== null));
  if (coveredSnrLevels.size < MINIMUM_FITTING_SNR_LEVELS) {
    throw new Error(`${scenario.id} ${view} observation-domain-eligible attempts cover only ${coveredSnrLevels.size} SNR level(s); expected at least ${MINIMUM_FITTING_SNR_LEVELS}`);
  }
  if (coveredRbwDivisors.size < MINIMUM_FITTING_RBW_DIVISORS) {
    throw new Error(`${scenario.id} ${view} observation-domain-eligible attempts cover only ${coveredRbwDivisors.size} RBW divisor(s); expected at least ${MINIMUM_FITTING_RBW_DIVISORS}`);
  }
}

function assertCalibrationCoverage(
  scenario: CanonicalClassificationScenario,
  attempts: readonly ScenarioSamplingAttempt[],
  view: TailCalibrationView,
): void {
  const eligibleAttempts = attempts.filter((attempt) => attempt.eligibleRepresentativeCountsByView[view] > 0);
  if (eligibleAttempts.length < MINIMUM_DISTINCT_CALIBRATION_ATTEMPTS) {
    throw new Error(`${scenario.id} ${view} has only ${eligibleAttempts.length} distinct observation-domain-eligible calibration attempts; ${MINIMUM_DISTINCT_CALIBRATION_ATTEMPTS} are required to resolve an empirical rank below ${SYNTHETIC_SUPPORT_RANK_REJECTION_THRESHOLD}`);
  }
}

function assertHighSnrSeedCoverage(
  purpose: 'fitting' | 'calibration',
  scenario: CanonicalClassificationScenario,
  attempts: readonly ScenarioSamplingAttempt[],
  seeds: readonly number[],
  view: TailCalibrationView = 'envelope-timed',
): void {
  assertUniqueNumbers(`${purpose} high-SNR coverage seeds`, seeds);
  const configuredSeeds = new Set(seeds);
  const minimumCoverage = HIGH_SNR_MINIMUM_SEED_COVERAGE_BY_SCENARIO[scenario.id] ?? 1;
  const requiredSeedCount = Math.ceil(configuredSeeds.size * minimumCoverage);
  for (const snrDb of SNR_DB.filter((value) => value >= HIGH_SNR_MINIMUM_DB)) {
    const coveredSeeds = new Set(attempts
      .filter((attempt) => attempt.snrDb === snrDb
        && configuredSeeds.has(attempt.seed)
        && attempt.eligibleRepresentativeCountsByView[view] > 0)
      .map((attempt) => attempt.seed));
    if (coveredSeeds.size < requiredSeedCount) {
      throw new Error(`${scenario.id} ${purpose} ${view} high-SNR observation-domain-eligible acquisition covered ${coveredSeeds.size}/${configuredSeeds.size} distinct seeds at ${snrDb} dB; required ${requiredSeedCount}/${configuredSeeds.size}`);
    }
  }
}

function assertHighSnrProductionSpectrumEligibilityCoverage(
  purpose: 'fitting' | 'calibration',
  scenario: CanonicalClassificationScenario,
  attempts: readonly ScenarioSamplingAttempt[],
  seeds: readonly number[],
  requiredRegimes: readonly ObservableTrainingAcquisitionRegime[],
): void {
  const configuredSeeds = new Set(seeds);
  const requiredSeedCount =
    MINIMUM_PRODUCTION_ACQUISITION_REGIME_HIGH_SNR_SPECTRUM_ELIGIBLE_DISTINCT_SEEDS;
  for (const acquisitionRegime of requiredRegimes) {
    for (const snrDb of SNR_DB.filter((value) => value >= HIGH_SNR_MINIMUM_DB)) {
      const coveredSeeds = new Set(attempts
        .filter((attempt) => attempt.acquisitionRegimeId === acquisitionRegime.id
          && attempt.snrDb === snrDb
          && configuredSeeds.has(attempt.seed)
          && attempt.eligibleRepresentativeCountsByView['spectrum-only'] > 0)
        .map((attempt) => attempt.seed));
      if (coveredSeeds.size < requiredSeedCount) {
        throw new Error(`${scenario.id} ${purpose} spectrum-only production acquisition regime ${acquisitionRegime.id} covered ${coveredSeeds.size}/${configuredSeeds.size} distinct observation-domain-eligible seeds at ${snrDb} dB; required ${requiredSeedCount}/${configuredSeeds.size}`);
      }
    }
  }
}

function assertHighSnrProductionQualifiedEnvelopeCaptureCoverage(
  purpose: 'fitting' | 'calibration',
  scenario: CanonicalClassificationScenario,
  attempts: readonly ScenarioSamplingAttempt[],
  seeds: readonly number[],
  requiredRegimes: readonly ObservableTrainingAcquisitionRegime[],
): void {
  const configuredSeeds = new Set(seeds);
  const requiredSeedCount =
    MINIMUM_PRODUCTION_ACQUISITION_REGIME_HIGH_SNR_ENVELOPE_CAPTURE_DISTINCT_SEEDS;
  for (const acquisitionRegime of requiredRegimes) {
    for (const snrDb of SNR_DB.filter((value) => value >= HIGH_SNR_MINIMUM_DB)) {
      const coveredSeeds = new Set(attempts
        .filter((attempt) => attempt.acquisitionRegimeId === acquisitionRegime.id
          && attempt.snrDb === snrDb
          && configuredSeeds.has(attempt.seed)
          && attempt.runtimeBranches.qualifiedEnvelope.physicalDetectedPowerCaptureCount === 1)
        .map((attempt) => attempt.seed));
      if (coveredSeeds.size < requiredSeedCount) {
        throw new Error(`${scenario.id} ${purpose} qualified-envelope production acquisition regime ${acquisitionRegime.id} captured ${coveredSeeds.size}/${configuredSeeds.size} distinct physical-capture seeds at ${snrDb} dB; required ${requiredSeedCount}/${configuredSeeds.size}`);
      }
    }
  }
}

function attemptCoverageByScenario(
  attempts: readonly ScenarioSamplingAttempt[],
): Readonly<Record<string, Readonly<Record<string, unknown>>>> {
  const scenarioIds = [...new Set(attempts.map((attempt) => attempt.scenarioId))].sort();
  return Object.fromEntries(scenarioIds.map((scenarioId) => {
    const selected = attempts.filter((attempt) => attempt.scenarioId === scenarioId);
    const selectedForView = (view: ObservableModelView) => selected.filter((attempt) =>
      attempt.eligibleRepresentativeCountsByView[view] > 0);
    return [scenarioId, {
      distinctPairedNuisanceCells: new Set(selected.map(samplingAttemptKey)).size,
      runtimeBranches: {
        consecutiveSpectrum: {
          attemptsWithAnyRepresentative: selected.filter((attempt) =>
            attempt.runtimeBranches.consecutiveSpectrum.representativeCount > 0).length,
          attemptsWithFitEligibleRepresentative: selected.filter((attempt) =>
            attempt.runtimeBranches.consecutiveSpectrum.fitEligibleRepresentativeCount > 0).length,
          representativeCount: selected.reduce((sum, attempt) =>
            sum + attempt.runtimeBranches.consecutiveSpectrum.representativeCount, 0),
          fitEligibleRepresentativeCount: selected.reduce((sum, attempt) =>
            sum + attempt.runtimeBranches.consecutiveSpectrum.fitEligibleRepresentativeCount, 0),
        },
        qualifiedEnvelope: {
          attemptsWithPhysicalDetectedPowerCapture: selected.filter((attempt) =>
            attempt.runtimeBranches.qualifiedEnvelope.physicalDetectedPowerCaptureCount > 0).length,
          attemptsWithReceiptVerifiedDetectedPowerCaptureSample: selected.filter((attempt) =>
            attempt.runtimeBranches.qualifiedEnvelope.detectedPowerCaptureSampleCount > 0).length,
          attemptsWithAdmittedEnvelopeRepresentative: selected.filter((attempt) =>
            attempt.runtimeBranches.qualifiedEnvelope.representativeCount > 0).length,
          attemptsWithCensoredFrequencyAgileFixedTuneCapture: selected.filter((attempt) =>
            attempt.runtimeBranches.qualifiedEnvelope.censoredFrequencyAgileFixedTuneCaptureCount > 0).length,
          attemptsWithFitEligibleTimedRepresentative: selected.filter((attempt) =>
            attempt.runtimeBranches.qualifiedEnvelope.fitEligibleRepresentativeCount > 0).length,
          admittedEnvelopeRepresentativeCount: selected.reduce((sum, attempt) =>
            sum + attempt.runtimeBranches.qualifiedEnvelope.representativeCount, 0),
          censoredFrequencyAgileFixedTuneCaptureCount: selected.reduce((sum, attempt) =>
            sum + attempt.runtimeBranches.qualifiedEnvelope.censoredFrequencyAgileFixedTuneCaptureCount, 0),
        },
      },
      fitEligibleAttemptCountsByView: Object.fromEntries(OBSERVABLE_MODEL_VIEWS.map((view) => [
        view,
        selectedForView(view).length,
      ])),
      fitEligibleSnrLevelsByView: Object.fromEntries(OBSERVABLE_MODEL_VIEWS.map((view) => [
        view,
        [...new Set(selectedForView(view).map((attempt) => attempt.snrDb))]
          .sort((left, right) => left - right),
      ])),
      fitEligibleRbwDivisorsByView: Object.fromEntries(OBSERVABLE_MODEL_VIEWS.map((view) => [
        view,
        [...new Set(selectedForView(view)
          .map((attempt) => attempt.rbwDivisor)
          .filter((rbwDivisor): rbwDivisor is number => rbwDivisor !== null))]
          .sort((left, right) => left - right),
      ])),
      fitEligibleAcquisitionRegimeIdsByView:
        Object.fromEntries(OBSERVABLE_MODEL_VIEWS.map((view) => [
          view,
          [...new Set(selectedForView(view).map((attempt) => attempt.acquisitionRegimeId))].sort(),
        ])),
      highSnrSeedCoverageByView: Object.fromEntries(OBSERVABLE_MODEL_VIEWS.map((view) => [
        view,
        Object.fromEntries(SNR_DB.filter((snrDb) => snrDb >= HIGH_SNR_MINIMUM_DB).map((snrDb) => [
          snrDb,
          new Set(selectedForView(view)
            .filter((attempt) => attempt.snrDb === snrDb)
            .map((attempt) => attempt.seed)).size,
        ])),
      ])),
      highSnrSeedCoverageUnit: 'distinct-seeds-with-observation-domain-eligible-representative',
    }];
  }));
}

function assertUniqueNumbers(label: string, values: readonly number[]): void {
  const duplicates = values.filter((value, index) => values.indexOf(value) !== index);
  if (duplicates.length > 0) {
    throw new Error(`${label} contains duplicate values: ${[...new Set(duplicates)].join(', ')}`);
  }
}

function assertUniqueStrings(label: string, values: readonly string[]): void {
  const duplicates = values.filter((value, index) => values.indexOf(value) !== index);
  if (duplicates.length > 0) {
    throw new Error(`${label} contains duplicate values: ${[...new Set(duplicates)].join(', ')}`);
  }
}

function assertDisjointNumbers(
  leftLabel: string,
  left: readonly number[],
  rightLabel: string,
  right: readonly number[],
): void {
  const rightValues = new Set(right);
  const overlap = left.filter((value) => rightValues.has(value));
  if (overlap.length > 0) {
    throw new Error(`${leftLabel} and ${rightLabel} overlap: ${overlap.join(', ')}`);
  }
}

function samplingAttemptKey(attempt: Pick<ScenarioSamplingAttempt, 'scenarioId' | 'snrDb' | 'acquisitionRegimeId' | 'seed'>): string {
  return `${attempt.scenarioId}:snr=${attempt.snrDb}:regime=${attempt.acquisitionRegimeId}:seed=${attempt.seed}`;
}

async function precomputeFeatureSamplingAttempts(
  phase: 'fitting' | 'calibration',
  scenarios: readonly CanonicalClassificationScenario[],
  snrLevels: readonly number[],
  regimes: readonly ObservableTrainingAcquisitionRegime[],
  seeds: readonly number[],
): Promise<Map<string, FeatureSamplingAttempt>> {
  const items: AttemptSamplingWorkItem[] = [];
  for (const scenario of scenarios) {
    for (const snrDb of snrLevels) {
      for (let regimeIndex = 0; regimeIndex < regimes.length; regimeIndex += 1) {
        for (const seed of seeds) {
          items.push({
            key: samplingAttemptKey({ scenarioId: scenario.id, snrDb, acquisitionRegimeId: regimes[regimeIndex]!.id, seed }),
            scenarioId: scenario.id,
            snrDb,
            regimeIndex,
            seed,
          });
        }
      }
    }
  }
  if (items.length === 0) return new Map();
  // Small chunks pulled from a shared queue, not one static chunk per
  // worker: a worker that gets one giant chunk reports nothing until that
  // entire chunk finishes (here, tens of minutes), and a static split is
  // only as balanced as the split itself. A pull-based queue of small
  // chunks gives continuous progress and self-corrects any residual
  // imbalance — a worker that finishes early just pulls the next chunk.
  // A cache record is the crash-recovery unit. Some morphology cells are
  // substantially slower than others, so keep each checkpoint small enough
  // that an interrupted worker loses minutes rather than nearly an hour.
  const CHUNK_SIZE = 10;
  const chunks: AttemptSamplingWorkItem[][] = [];
  for (let index = 0; index < items.length; index += CHUNK_SIZE) {
    chunks.push(items.slice(index, index + CHUNK_SIZE));
  }
  TRAINER_RUN.assertWorkerRuntimeUnchanged();
  const baselineCache = createAttemptSamplingCache({
    rootDirectory: NORMAL_ATTEMPT_CACHE_ROOT,
    phase,
    sourceIdentity: ATTEMPT_SAMPLING_CACHE_SOURCE_IDENTITY,
    scenarios: scenarios.map((scenario) => ({ id: scenario.id, value: scenario })),
    snrLevels,
    regimes,
    seeds,
    items,
    chunks,
    workerModuleUrl: WORKER_MODULE_URL,
  });
  const cache = FRESH_SAMPLING_RUN
    ? createAttemptSamplingCache({
      rootDirectory: resolve(FRESH_SAMPLING_RUN.cacheRoot, 'v1'),
      phase,
      sourceIdentity: ATTEMPT_SAMPLING_CACHE_SOURCE_IDENTITY,
      scenarios: scenarios.map((scenario) => ({ id: scenario.id, value: scenario })),
      snrLevels,
      regimes,
      seeds,
      items,
      chunks,
      workerModuleUrl: WORKER_MODULE_URL,
    })
    : baselineCache;
  const attempts = new Map<string, FeatureSamplingAttempt>();
  const chunkRecords: Array<AttemptSamplingCacheChunkRecord | undefined> =
    Array.from({ length: chunks.length });
  const pendingChunks: Array<{
    chunkIndex: number;
    items: AttemptSamplingWorkItem[];
  }> = [];
  let cacheHitCount = 0;
  for (let chunkIndex = 0; chunkIndex < chunks.length; chunkIndex += 1) {
    const chunk = chunks[chunkIndex]!;
    const cached = cache.loadChunk(chunkIndex, chunk);
    if (!cached) {
      pendingChunks.push({ chunkIndex, items: chunks[chunkIndex]! });
      continue;
    }
    admitAttemptSamplingChunk(attempts, chunk, cached);
    if (FRESH_SAMPLING_RUN) {
      baselineCache.publishChunk(chunkIndex, chunk, cached.results);
    }
    chunkRecords[chunkIndex] = cached.record;
    cacheHitCount += 1;
  }
  const workerCount = Math.max(1, Math.min(pendingChunks.length, availableParallelism()));
  console.error(
    `[attempt-cache] phase=${phase} fingerprint=${cache.fingerprint} mode=${FRESH_SAMPLING_RUN ? (FRESH_SAMPLING_RUN.resumed ? 'fresh-resume' : 'fresh-new') : 'trusted-local-reuse'} hits=${cacheHitCount} misses=${pendingChunks.length} corrupt=${cache.corruptChunkCount}`,
  );
  if (pendingChunks.length > 0) {
    console.error(`[attempt-sampling] dispatching ${pendingChunks.reduce((sum, chunk) => sum + chunk.items.length, 0)}/${items.length} attempts in ${pendingChunks.length}/${chunks.length} chunks of up to ${CHUNK_SIZE} across ${workerCount} workers`);
  }
  const precomputeWallClockStart = process.hrtime.bigint();
  let nextPendingChunkIndex = 0;
  let chunksCompleted = 0;
  let firstFailure: Error | undefined;
  const chunkProfiles: Array<Pick<AttemptSamplingWorkResponse, 'timingMs' | 'wallClockMs'>> = [];

  async function runWorkerLoop(workerId: number): Promise<void> {
    TRAINER_RUN.assertWorkerRuntimeUnchanged();
    const worker = new Worker(WORKER_MODULE_URL);
    try {
      while (firstFailure === undefined) {
        const pendingChunkIndex = nextPendingChunkIndex;
        if (pendingChunkIndex >= pendingChunks.length) return;
        nextPendingChunkIndex += 1;
        const { chunkIndex, items: chunk } = pendingChunks[pendingChunkIndex]!;
        let response: AttemptSamplingWorkResponse;
        try {
          response = await postToAttemptSamplingWorker(worker, regimes, chunk);
          admitAttemptSamplingChunk(attempts, chunk, response);
          chunkRecords[chunkIndex] = cache.publishChunk(chunkIndex, chunk, response.results);
          if (FRESH_SAMPLING_RUN) {
            baselineCache.publishChunk(chunkIndex, chunk, response.results);
          }
        } catch (error) {
          const failure = error instanceof Error
            ? error
            : new Error(`Attempt-sampling worker ${workerId} failed`, { cause: error });
          if (firstFailure === undefined) firstFailure = failure;
          else console.error(`[attempt-sampling] secondary worker failure: ${failure.message}`);
          return;
        }
        chunkProfiles.push({ timingMs: response.timingMs, wallClockMs: response.wallClockMs });
        chunksCompleted += 1;
        const elapsedMs = Number(process.hrtime.bigint() - precomputeWallClockStart) / 1e6;
        const chunksPerMs = chunksCompleted / elapsedMs;
        const etaSeconds = chunksPerMs > 0
          ? Math.round((pendingChunks.length - chunksCompleted) / chunksPerMs / 1000)
          : undefined;
        console.error(`[attempt-sampling] chunk ${chunkIndex + 1}/${chunks.length} done in ${Math.round(response.wallClockMs)}ms (worker ${workerId}, ${chunk.length} attempts) — ${chunksCompleted}/${pendingChunks.length} cache misses complete, ${Math.round(elapsedMs)}ms elapsed${etaSeconds === undefined ? '' : `, ~${etaSeconds}s remaining`}`);
        if (firstFailure !== undefined) return;
      }
    } finally {
      await worker.terminate();
    }
  }

  if (pendingChunks.length > 0) {
    await Promise.all(Array.from({ length: workerCount }, (_unused, workerId) => runWorkerLoop(workerId)));
  }
  if (firstFailure !== undefined) throw firstFailure;
  TRAINER_RUN.assertWorkerRuntimeUnchanged();
  const precomputeWallClockMs = Number(process.hrtime.bigint() - precomputeWallClockStart) / 1e6;
  if (attempts.size !== items.length) {
    throw new Error(`Attempt-sampling pool returned ${attempts.size}/${items.length} deterministic nuisance cells`);
  }
  const completeChunkRecords = chunkRecords.map((record, chunkIndex) => {
    if (!record) throw new Error(`Attempt-sampling cache is missing completed chunk record ${chunkIndex}`);
    return record;
  });
  cache.seal(completeChunkRecords);
  if (FRESH_SAMPLING_RUN) baselineCache.seal(completeChunkRecords);
  console.error(
    `[attempt-cache] phase=${phase} fingerprint=${cache.fingerprint} sealed=true hits=${cacheHitCount} computed=${pendingChunks.length} directory=${cache.directory}`,
  );
  if (chunkProfiles.length > 0) {
    logAttemptSamplingProfile(chunkProfiles, precomputeWallClockMs, workerCount);
  }
  TRAINER_RUN.assertWorkerRuntimeUnchanged();
  return attempts;
}

function admitAttemptSamplingChunk(
  attempts: Map<string, FeatureSamplingAttempt>,
  requestedItems: readonly AttemptSamplingWorkItem[],
  response: Pick<AttemptSamplingWorkResponse, 'results'>,
): void {
  const failedResult = response.results.find((result) =>
    result.errorMessage !== undefined || result.attempt === undefined);
  if (failedResult !== undefined) {
    throw new Error(
      `Feature extraction failed for ${failedResult.key}: ${failedResult.errorMessage ?? 'no attempt returned'}`,
    );
  }
  if (response.results.length !== requestedItems.length) {
    throw new Error(`Attempt-sampling worker returned ${response.results.length}/${requestedItems.length} results without an attributed failure`);
  }
  for (let index = 0; index < requestedItems.length; index += 1) {
    const requested = requestedItems[index]!;
    const result = response.results[index]!;
    if (result.key !== requested.key) {
      throw new Error(`Attempt-sampling worker reordered ${requested.key} as ${result.key}`);
    }
    if (attempts.has(result.key)) {
      throw new Error(`Attempt-sampling worker returned duplicate nuisance cell ${result.key}`);
    }
    attempts.set(result.key, result.attempt!);
  }
}

// Diagnostic-only: summarizes where the precompute pool actually spent its
// time, to inform whether further speedup should target CPU parallelism,
// allocation/GC pressure, or something else entirely. Printed to stderr so
// it never touches the tool's stdout JSON summary.
function logAttemptSamplingProfile(
  chunkResponses: readonly Pick<AttemptSamplingWorkResponse, 'timingMs' | 'wallClockMs'>[],
  precomputeWallClockMs: number,
  workerCount: number,
): void {
  const totals = { spectrumSynthesis: 0, zeroSpanSynthesis: 0, detectAndTrack: 0, featureExtraction: 0, hashing: 0, attemptCount: 0 };
  let workerWallClockMsSum = 0;
  let maxWorkerWallClockMs = 0;
  let minWorkerWallClockMs = Infinity;
  for (const response of chunkResponses) {
    totals.spectrumSynthesis += response.timingMs.spectrumSynthesis;
    totals.zeroSpanSynthesis += response.timingMs.zeroSpanSynthesis;
    totals.detectAndTrack += response.timingMs.detectAndTrack;
    totals.featureExtraction += response.timingMs.featureExtraction;
    totals.hashing += response.timingMs.hashing;
    totals.attemptCount += response.timingMs.attemptCount;
    workerWallClockMsSum += response.wallClockMs;
    maxWorkerWallClockMs = Math.max(maxWorkerWallClockMs, response.wallClockMs);
    minWorkerWallClockMs = Math.min(minWorkerWallClockMs, response.wallClockMs);
  }
  const trackedMs = totals.spectrumSynthesis + totals.zeroSpanSynthesis + totals.detectAndTrack
    + totals.featureExtraction + totals.hashing;
  console.error(JSON.stringify({
    attemptSamplingProfile: {
      workerCount,
      attemptCount: totals.attemptCount,
      precomputeWallClockMs: Math.round(precomputeWallClockMs),
      speedupVsSerialCpuTime: Number((workerWallClockMsSum / precomputeWallClockMs).toFixed(2)),
      workerWallClockMsBalance: {
        min: Math.round(minWorkerWallClockMs),
        max: Math.round(maxWorkerWallClockMs),
        sum: Math.round(workerWallClockMsSum),
      },
      cpuTimeMsByBucket: {
        spectrumSynthesis: Math.round(totals.spectrumSynthesis),
        zeroSpanSynthesis: Math.round(totals.zeroSpanSynthesis),
        detectAndTrack: Math.round(totals.detectAndTrack),
        featureExtraction: Math.round(totals.featureExtraction),
        hashing: Math.round(totals.hashing),
        untracked: Math.round(workerWallClockMsSum - trackedMs),
      },
      cpuTimeShareByBucket: {
        spectrumSynthesis: Number((totals.spectrumSynthesis / trackedMs).toFixed(3)),
        zeroSpanSynthesis: Number((totals.zeroSpanSynthesis / trackedMs).toFixed(3)),
        detectAndTrack: Number((totals.detectAndTrack / trackedMs).toFixed(3)),
        featureExtraction: Number((totals.featureExtraction / trackedMs).toFixed(3)),
        hashing: Number((totals.hashing / trackedMs).toFixed(3)),
      },
    },
  }, null, 2));
}

function median(values: readonly number[]): number { const ordered = [...values].sort((left, right) => left - right); const middle = Math.floor(ordered.length / 2); return ordered.length % 2 ? ordered[middle]! : (ordered[middle - 1]! + ordered[middle]!) / 2; }
