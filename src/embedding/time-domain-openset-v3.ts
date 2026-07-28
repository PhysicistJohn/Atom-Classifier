/**
 * Staged v3 open-set rejection for the strict time-domain (FFT-free) frontend.
 *
 * Exact runtime port of the STAGED Python policy that passed every development
 * open-set gate on both fusion seeds (HANDOFF section 20):
 *
 * - STAGE 1 -- `training/zplane_ab/v2_full_variation/v3_scale/noise_prefilter.py`:
 *   a ridge-logistic noise/not-noise GATE over seven pose-degeneracy features
 *   (`v3_scale/pose_degeneracy.py`), one fitted model per capture length.
 *   When it fires the row is rejected as noise and NOTHING downstream runs:
 *   no encoder, no fusion, no prototype distance, no branch LOF. Captures
 *   LONGER than the longest fitted length are gated on their first
 *   longest-fitted-length samples with that length's model
 *   ({@link STAGE_ONE_CAUSAL_PREFIX_RULE}); captures at any other uncovered
 *   length are refused.
 * - STAGE 2 -- `v3_scale/fit_v3_openset_staged.py` + `v3_time_domain_openset.py`:
 *   the frozen additive policy for survivors. Branch LOF ranks (real k=2
 *   weight 0.40, complex k=64 weight 0.60) are blended 0.80/0.20 with the
 *   `across_patch_frequency_dispersion` class-deviation rank and re-ranked
 *   against enrollment; stage-2 internals are untouched by policy version 2.
 * - COMPOSITE (staged policy VERSION 2, the HANDOFF 25 follow-up): a stage-1
 *   survivor's open-set score is
 *   `max(stage2_enrollment_rank, stage1_score_enrollment_rank)`, where the
 *   stage-1 term is the empirical enrollment rank (searchsorted-left, the
 *   stage-2 convention) of the stage-1 logistic log-odds against a
 *   calibration fit on ENROLLMENT stage-1 SURVIVORS only. The staged unknown
 *   threshold is the frozen q95 of that composite over enrollment survivors.
 *   Gated rows keep the existing `1 + softsign(stage-1 log-odds)` axis in
 *   (1, 2), above every survivor. Version 1 (survivor score = stage-2 rank
 *   alone, stage-2 q95 threshold) was sealed at release seed 20260731 and
 *   its failure is frozen; this module refuses any asset whose recorded
 *   policy version is not the current frozen one.
 *
 * ARCHITECTURE CONTRACT CHANGE (HANDOFF 20.4), machine-readable here as in
 * Python: the historical rejector was strictly additive; the staged system is
 * NOT. Stage 1 gates BEFORE classification, so a gated row has no closed label
 * at all. Stage 2 remains additive: it never alters the closed label of a
 * surviving row. Every emitted decision carries
 * `additive_only: false`, `changes_closed_label: true`,
 * `gates_before_classification: true`.
 *
 * Everything here is plain time-domain arithmetic. No FFT, Welch, filterbank,
 * or spectral grid is introduced anywhere in this module.
 */

import {
  estimateTimeDomainBand,
  minimumTimeDomainBandwidthForActiveSpan,
  TIME_DOMAIN_BASE_LAGS,
  TIME_DOMAIN_MAX_LAG,
  TIME_DOMAIN_MAX_UNAMBIGUOUS_LAG_ARC,
  TIME_DOMAIN_MIN_LAG_SIGNAL_TO_SAMPLING_NOISE,
} from './time-domain-geometry-v3.js';
import {
  admitTimeDomainAssetStatusV3,
  type TimeDomainAssetLoadOptionsV3,
  type TimeDomainAssetStatusV3,
} from './time-domain-asset-status-v3.js';

export const TIME_DOMAIN_OPENSET_SCHEMA =
  'atomos.v3.time-domain-openset.staged' as const;
export const TIME_DOMAIN_OPENSET_SCHEMA_VERSION = 2 as const;

export const NOISE_PREFILTER_VERSION = 'noise-prefilter-v1' as const;

/**
 * Mirror of the `fit_v3_openset_staged` STAGED POLICY frozen-constants block,
 * VERSION 2 (the composite survivor score). An asset recording any other
 * `policy_version` is refused at load: a version-1 artifact must not be able
 * to score with version-2 semantics or vice versa.
 */
export const STAGED_POLICY_SCHEMA = 2 as const;
export const STAGED_POLICY_VERSION =
  'v3-staged-openset-policy-v2-composite-survivor' as const;
export const STAGED_POLICY_KIND =
  'v3_staged_noise_prefilter_then_composite_survivor_lof_geometry' as const;
/** Mirror of `fit_v3_openset_staged.COMPOSITE_SURVIVOR_SCORE`. */
export const COMPOSITE_SURVIVOR_SCORE =
  'max(stage2_enrollment_rank, stage1_score_enrollment_rank)' as const;

/** Mirror of `v3_time_domain_openset` frozen constants. Never tuned here. */
export const FROZEN_BRANCH_LOF_RANK_WEIGHT = 0.8;
export const FROZEN_GEOMETRY_WEIGHT = 0.2;
export const FROZEN_THRESHOLD_QUANTILE = 0.95;
/**
 * The composite threshold quantile is the SAME frozen quantile stage 2 uses,
 * aliased exactly as Python aliases it (`COMPOSITE_THRESHOLD_QUANTILE =
 * THRESHOLD_QUANTILE`), never re-typed.
 */
export const COMPOSITE_THRESHOLD_QUANTILE = FROZEN_THRESHOLD_QUANTILE;
export const FROZEN_GEOMETRY_FEATURE =
  'across_patch_frequency_dispersion' as const;

/** Mirror of `fit_v3_openset_staged.STAGE_ONE_SCORE_OFFSET`. */
export const STAGE_ONE_SCORE_OFFSET = 1.0;

/**
 * The stage-1 rule for capture lengths ABOVE the longest fitted prefilter
 * length (the sealed suite includes N32768; the development corpus stores
 * N16384 at most, so no N32768 prefilter can exist): gate on the capture's
 * first `longest-fitted-length` samples with that length's model. The first
 * N samples of a longer capture are exactly an N-sample capture of the same
 * emission -- the causal-prefix rule the release protocol itself is built
 * on. Capture lengths BELOW the smallest fitted length (and uncovered
 * lengths between fitted lengths) keep the existing refusal unchanged.
 */
export const STAGE_ONE_CAUSAL_PREFIX_RULE =
  'at a capture length above the longest fitted stage-1 length, the gate '
  + 'computes its pose-degeneracy features on the capture\'s first '
  + 'longest-fitted-length samples and gates with that length\'s model; the '
  + 'first N samples of a longer capture are exactly an N-sample capture of '
  + 'the same emission (the causal-prefix rule the release protocol is built '
  + 'on). Lengths below the smallest fitted length keep the existing refusal '
  + 'semantics unchanged.';

/**
 * The exact contract keys a release claim must carry
 * (`noise_prefilter.CONTRACT_KEYS_FOR_RELEASE_CLAIM`).
 */
export interface StagedArchitectureContract {
  additive_only: false;
  changes_closed_label: true;
  gates_before_classification: true;
  architecture_contract_change: string;
}

export const STAGED_ARCHITECTURE_CONTRACT_CHANGE =
  'The v3 open-set rejector is strictly additive and never alters the closed '
  + 'label. This noise prefilter is a GATE: when it fires, classification does '
  + 'not run and the row has no closed label at all. That is a deliberate '
  + 'change to the architecture contract, not an oversight, and it must be '
  + 'stated in the emitted JSON and in any release claim. It also means a '
  + 'gated row skips the encoder, the fusion, the prototype distance and the '
  + 'branch LOF; that saving is measured by compute_saving() and is never '
  + 'asserted.';

export function stagedArchitectureContract(): StagedArchitectureContract {
  return {
    additive_only: false,
    changes_closed_label: true,
    gates_before_classification: true,
    architecture_contract_change: STAGED_ARCHITECTURE_CONTRACT_CHANGE,
  };
}

// ---------------------------------------------------------------------------
// pose-degeneracy features (the seven the prefilter consumes)
// ---------------------------------------------------------------------------

/** Mirror of `pose_degeneracy` floors and prefix constants. */
export const POSE_DEGENERACY_VERSION = 'pose-degeneracy-v1' as const;
const MAGNITUDE_FLOOR = 1e-9;
const LOG_MAGNITUDE_FLOOR = Math.log10(MAGNITUDE_FLOOR);
const BANDWIDTH_FLOOR = 1e-5;
const LOG_BANDWIDTH_FLOOR = Math.log10(BANDWIDTH_FLOOR);
const PREFIX_FRACTIONS = [0.25, 0.5, 1.0] as const;
const MIN_PREFIX_SAMPLES = 2 * TIME_DOMAIN_MAX_LAG + 2;
const CORRELATION_EPSILON_FACTOR = 64;

/**
 * The prefilter's frozen feature order
 * (`noise_prefilter.PREFILTER_FEATURES`).
 */
export const PREFILTER_FEATURE_NAMES = [
  'log10_relative_lag_magnitude_mean',
  'relative_lag_magnitude_decay_slope',
  'bandwidth_log_dispersion',
  'carrier_phase_coherence',
  'high_lag_qualified_fraction',
  'prefix_bandwidth_log_dispersion',
  'peak_normalized_mean_power',
] as const;

export type PrefilterFeatureName = (typeof PREFILTER_FEATURE_NAMES)[number];

interface ComplexValue {
  real: number;
  imaginary: number;
}

function safeLog10(value: number, floor: number, logFloor: number): number {
  if (!Number.isFinite(value) || value <= floor) return logFloor;
  return Math.log10(value);
}

function populationStd(values: number[]): number {
  if (values.length < 2) return 0;
  let mean = 0;
  for (const value of values) mean += value;
  mean /= values.length;
  let variance = 0;
  for (const value of values) {
    const delta = value - mean;
    variance += delta * delta;
  }
  return Math.sqrt(variance / values.length);
}

function nonzeroLag(
  inPhase: Float64Array,
  quadrature: Float64Array,
  lag: number,
): ComplexValue {
  let real = 0;
  let imaginary = 0;
  const count = inPhase.length - lag;
  for (let index = 0; index < count; index++) {
    const leftReal = inPhase[index]!;
    const leftImaginary = quadrature[index]!;
    const rightReal = inPhase[index + lag]!;
    const rightImaginary = quadrature[index + lag]!;
    real += leftReal * rightReal + leftImaginary * rightImaginary;
    imaginary += leftReal * rightImaginary - leftImaginary * rightReal;
  }
  return { real: real / count, imaginary: imaginary / count };
}

function magnitude(value: ComplexValue): number {
  return Math.hypot(value.real, value.imaginary);
}

function angle(value: ComplexValue): number {
  return Math.atan2(value.imaginary, value.real);
}

/**
 * The resolution floor the v3 frontend itself passes to the estimator
 * (`fit_v3_openset_staged.frontend_min_bandwidth`); stage-1 features are
 * computed with the SAME floor so they describe the pose the classifier used.
 */
export function frontendMinimumBandwidth(
  rawLength: number,
  patchLength: number,
  targetFrac: number,
): number {
  return minimumTimeDomainBandwidthForActiveSpan(
    rawLength,
    patchLength,
    targetFrac,
  );
}

export interface PoseDegeneracyOptions {
  minBandwidth?: number;
  fullBandwidth?: number;
}

/**
 * Compute the seven prefilter features for one raw capture.
 *
 * Exact port of the corresponding subset of
 * `pose_degeneracy.pose_degeneracy_report`, in the frozen
 * {@link PREFILTER_FEATURE_NAMES} order. Rejects the same inputs the Python
 * estimator rejects (short, non-finite, or all-zero captures).
 */
export function prefilterPoseDegeneracyFeatures(
  inPhase: ArrayLike<number>,
  quadrature: ArrayLike<number>,
  options: PoseDegeneracyOptions = {},
): Float64Array {
  // Validation, the degenerate-fallback flag and the base-lag-one bandwidth
  // come from the shared estimator, exactly as Python calls `estimate_band`.
  const estimate = estimateTimeDomainBand(inPhase, quadrature, options);
  const context = estimate.context;

  // Repeat the estimator's own peak normalisation, as Python does.
  const length = inPhase.length;
  let peak = 0;
  for (let index = 0; index < length; index++) {
    peak = Math.max(peak, Math.hypot(inPhase[index]!, quadrature[index]!));
  }
  const normalizedInPhase = new Float64Array(length);
  const normalizedQuadrature = new Float64Array(length);
  for (let index = 0; index < length; index++) {
    normalizedInPhase[index] = inPhase[index]! / peak;
    normalizedQuadrature[index] = quadrature[index]! / peak;
  }
  let meanPower = 0;
  for (let index = 0; index < length; index++) {
    meanPower += (
      normalizedInPhase[index]! * normalizedInPhase[index]!
      + normalizedQuadrature[index]! * normalizedQuadrature[index]!
    );
  }
  meanPower /= length;

  const baseLags = TIME_DOMAIN_BASE_LAGS;
  const lagSet = new Set<number>();
  for (const lag of baseLags) {
    lagSet.add(lag);
    lagSet.add(2 * lag);
  }
  const lags = new Map<number, ComplexValue>();
  for (const lag of [...lagSet].sort((a, b) => a - b)) {
    if (lag < length) {
      lags.set(lag, nonzeroLag(normalizedInPhase, normalizedQuadrature, lag));
    }
  }
  const numericalFloor = CORRELATION_EPSILON_FACTOR * Number.EPSILON * meanPower;

  // --- coherent-structure magnitudes ------------------------------------
  const logRelative = new Map<number, number>();
  for (const lag of baseLags) {
    const value = lags.get(lag);
    if (value === undefined) continue;
    logRelative.set(
      lag,
      safeLog10(
        magnitude(value) / meanPower,
        MAGNITUDE_FLOOR,
        LOG_MAGNITUDE_FLOOR,
      ),
    );
  }
  const sortedLags = [...logRelative.keys()].sort((a, b) => a - b);
  let logRelativeMean = LOG_MAGNITUDE_FLOOR;
  if (sortedLags.length > 0) {
    logRelativeMean = 0;
    for (const lag of sortedLags) logRelativeMean += logRelative.get(lag)!;
    logRelativeMean /= sortedLags.length;
  }

  let decaySlope = 0;
  if (sortedLags.length >= 2) {
    const abscissa = sortedLags.map((lag) => Math.log2(lag));
    const ordinate = sortedLags.map((lag) => logRelative.get(lag)!);
    let abscissaMean = 0;
    for (const value of abscissa) abscissaMean += value;
    abscissaMean /= abscissa.length;
    let ordinateMean = 0;
    for (const value of ordinate) ordinateMean += value;
    ordinateMean /= ordinate.length;
    let denominator = 0;
    let numerator = 0;
    for (let index = 0; index < abscissa.length; index++) {
      const centered = abscissa[index]! - abscissaMean;
      denominator += centered * centered;
      numerator += centered * (ordinate[index]! - ordinateMean);
    }
    if (denominator > 0) decaySlope = numerator / denominator;
  }

  // --- multiscale quotient disagreement ---------------------------------
  const logBandwidths: number[] = [];
  for (const baseLag of baseLags) {
    const doubled = lags.get(2 * baseLag);
    const base = lags.get(baseLag);
    if (base === undefined || doubled === undefined) continue;
    const baseMagnitude = magnitude(base);
    if (baseMagnitude <= numericalFloor) continue;
    const unitReal = base.real / baseMagnitude;
    const unitImaginary = base.imaginary / baseMagnitude;
    const squaredReal = unitReal * unitReal - unitImaginary * unitImaginary;
    const squaredImaginary = 2 * unitReal * unitImaginary;
    // Re(doubled * conj(unit ** 2)) / |base|
    const quotient = (
      doubled.real * squaredReal + doubled.imaginary * squaredImaginary
    ) / baseMagnitude;
    const clipped = Math.min(1, Math.max(-1, quotient));
    const bandwidth = Math.acos(clipped) / (Math.PI * baseLag);
    logBandwidths.push(
      safeLog10(bandwidth, BANDWIDTH_FLOOR, LOG_BANDWIDTH_FLOOR),
    );
  }
  const bandwidthLogDispersion = populationStd(logBandwidths);

  // --- carrier phase agreement across scales ----------------------------
  let coherence = 0;
  const lagOne = lags.get(1);
  if (lagOne !== undefined && magnitude(lagOne) > numericalFloor) {
    const reference = angle(lagOne);
    const residuals: number[] = [];
    for (const baseLag of baseLags.slice(1)) {
      const value = lags.get(baseLag);
      if (value === undefined) continue;
      if (magnitude(value) <= numericalFloor) continue;
      residuals.push(angle(value) - baseLag * reference);
    }
    if (residuals.length > 0) {
      let cosine = 0;
      let sine = 0;
      for (const residual of residuals) {
        cosine += Math.cos(residual);
        sine += Math.sin(residual);
      }
      coherence = Math.hypot(
        cosine / residuals.length,
        sine / residuals.length,
      );
    }
  }
  coherence = Math.min(1, Math.max(0, coherence));

  // --- estimator admission decisions ------------------------------------
  const fallback = context.degenerate_full_band_fallback;
  const baseLagOneBandwidth = context.base_lag_one_bandwidth;
  let qualified = 0;
  let considered = 0;
  for (const baseLag of baseLags.slice(1)) {
    considered += 1;
    const doubled = lags.get(2 * baseLag);
    if (doubled === undefined || fallback || baseLagOneBandwidth === null) {
      continue;
    }
    const base = lags.get(baseLag)!;
    const relativeMagnitude = magnitude(base) / meanPower;
    const arcOk = (
      baseLag * baseLagOneBandwidth <= TIME_DOMAIN_MAX_UNAMBIGUOUS_LAG_ARC
    );
    const samplingFloor = (
      TIME_DOMAIN_MIN_LAG_SIGNAL_TO_SAMPLING_NOISE / Math.sqrt(length - baseLag)
    );
    if (arcOk && relativeMagnitude >= samplingFloor) qualified += 1;
  }
  const qualifiedFraction = considered > 0 ? qualified / considered : 0;

  // --- stability across nested prefixes ---------------------------------
  const prefixLengths = [
    ...new Set(
      PREFIX_FRACTIONS
        .map((fraction) => Math.trunc(length * fraction))
        .filter((value) => value >= MIN_PREFIX_SAMPLES && value <= length),
    ),
  ].sort((a, b) => a - b);
  const prefixLogBandwidths: number[] = [];
  for (const prefix of prefixLengths) {
    let prefixPeak = 0;
    for (let index = 0; index < prefix; index++) {
      prefixPeak = Math.max(
        prefixPeak,
        Math.hypot(normalizedInPhase[index]!, normalizedQuadrature[index]!),
      );
    }
    if (prefixPeak === 0) continue; // zero-magnitude prefix: no pose exists.
    const prefixEstimate = estimateTimeDomainBand(
      normalizedInPhase.subarray(0, prefix),
      normalizedQuadrature.subarray(0, prefix),
      options,
    );
    prefixLogBandwidths.push(
      safeLog10(
        prefixEstimate.bandwidth,
        BANDWIDTH_FLOOR,
        LOG_BANDWIDTH_FLOOR,
      ),
    );
  }
  const prefixBandwidthLogDispersion = populationStd(prefixLogBandwidths);

  const features = new Float64Array([
    logRelativeMean,
    decaySlope,
    bandwidthLogDispersion,
    coherence,
    qualifiedFraction,
    prefixBandwidthLogDispersion,
    meanPower,
  ]);
  for (let index = 0; index < features.length; index++) {
    if (!Number.isFinite(features[index]!)) {
      throw new RangeError(
        `pose degeneracy feature ${PREFILTER_FEATURE_NAMES[index]} is not finite`,
      );
    }
  }
  return features;
}

// ---------------------------------------------------------------------------
// STAGE 1: the noise prefilter (ridge logistic gate, one model per length)
// ---------------------------------------------------------------------------

export interface NoisePrefilterModelAsset {
  capture_length: number;
  feature_names: string[];
  mean: number[];
  scale: number[];
  coefficients: number[];
  intercept: number;
  /** Score-space (log-odds) threshold; gate fires when score >= threshold. */
  threshold_score: number;
}

export interface StageOneAsset {
  kind: typeof NOISE_PREFILTER_VERSION;
  feature_names: string[];
  /** Stage-1 features use the frontend's own length-dependent floor. */
  match_frontend_min_bandwidth: true;
  models: Record<string, NoisePrefilterModelAsset>;
}

export interface StageOneEvaluation {
  features: Float64Array;
  /** Linear log-odds of noise; higher is more noise-like. */
  score: number;
  /** Softsign map of the score into (0, 1) (`_bounded_rank`). */
  rank: number;
  /** True means NOISE: abstain, do not classify. */
  gated: boolean;
  thresholdScore: number;
  /** The raw capture length the caller handed in. */
  captureLength: number;
  /**
   * The length whose fitted model produced this evaluation. Equal to
   * `captureLength` except when {@link STAGE_ONE_CAUSAL_PREFIX_RULE}
   * applied, in which case it is the longest fitted prefilter length.
   */
  evaluatedLength: number;
  /** True exactly when the causal-prefix rule applied (long capture). */
  causalPrefixApplied: boolean;
}

function softsignRank(score: number): number {
  return 0.5 * (score / (1 + Math.abs(score))) + 0.5;
}

/** One fitted per-length noise prefilter (stage 1). */
export class NoisePrefilterV3 {
  readonly captureLength: number;
  readonly mean: Float64Array;
  readonly scale: Float64Array;
  readonly coefficients: Float64Array;
  readonly intercept: number;
  readonly thresholdScore: number;

  constructor(asset: NoisePrefilterModelAsset) {
    const width = PREFILTER_FEATURE_NAMES.length;
    if (
      asset.feature_names.length !== width
      || asset.feature_names.some(
        (name, index) => name !== PREFILTER_FEATURE_NAMES[index],
      )
    ) {
      throw new RangeError(
        'noise prefilter feature names differ from the frozen prefilter order',
      );
    }
    if (
      asset.mean.length !== width
      || asset.scale.length !== width
      || asset.coefficients.length !== width
    ) {
      throw new RangeError('noise prefilter arrays must have one entry per feature');
    }
    for (let index = 0; index < width; index++) {
      if (
        !Number.isFinite(asset.mean[index]!)
        || !Number.isFinite(asset.scale[index]!)
        || !Number.isFinite(asset.coefficients[index]!)
        || asset.scale[index]! <= 0
      ) {
        throw new RangeError('noise prefilter parameters must be finite with positive scale');
      }
    }
    if (
      !Number.isInteger(asset.capture_length)
      || asset.capture_length <= 0
      || !Number.isFinite(asset.intercept)
      || !Number.isFinite(asset.threshold_score)
    ) {
      throw new RangeError('noise prefilter scalar parameters are invalid');
    }
    this.captureLength = asset.capture_length;
    this.mean = Float64Array.from(asset.mean);
    this.scale = Float64Array.from(asset.scale);
    this.coefficients = Float64Array.from(asset.coefficients);
    this.intercept = asset.intercept;
    this.thresholdScore = asset.threshold_score;
  }

  /** Linear log-odds of noise for one already-extracted feature vector. */
  score(features: Float64Array): number {
    if (features.length !== this.coefficients.length) {
      throw new RangeError(
        `expected ${this.coefficients.length} prefilter features`,
      );
    }
    let score = this.intercept;
    for (let index = 0; index < features.length; index++) {
      score += (
        this.coefficients[index]!
        * ((features[index]! - this.mean[index]!) / this.scale[index]!)
      );
    }
    return score;
  }

  /** Hard gate: `true` means NOISE, abstain, do not classify. */
  decide(features: Float64Array): boolean {
    return this.score(features) >= this.thresholdScore;
  }
}

/** The length-keyed stage-1 gate (`fit_v3_openset_staged.StageOneGate`). */
export class StageOneGateV3 {
  readonly models: Map<number, NoisePrefilterV3>;
  readonly patchLength: number;
  readonly targetFrac: number;

  constructor(
    asset: StageOneAsset,
    frontend: { patchLength: number; targetFrac: number },
  ) {
    if (asset.kind !== NOISE_PREFILTER_VERSION) {
      throw new RangeError(`stage one kind must be ${NOISE_PREFILTER_VERSION}`);
    }
    if (asset.match_frontend_min_bandwidth !== true) {
      throw new RangeError(
        'stage one must declare match_frontend_min_bandwidth: true',
      );
    }
    const entries = Object.entries(asset.models);
    if (entries.length === 0) {
      throw new RangeError('stage one needs at least one per-length model');
    }
    this.models = new Map();
    for (const [key, model] of entries) {
      const length = Number(key);
      if (!Number.isInteger(length) || length !== model.capture_length) {
        throw new RangeError(
          `stage one model key ${key} disagrees with its capture_length`,
        );
      }
      this.models.set(length, new NoisePrefilterV3(model));
    }
    this.patchLength = frontend.patchLength;
    this.targetFrac = frontend.targetFrac;
  }

  lengths(): number[] {
    return [...this.models.keys()].sort((a, b) => a - b);
  }

  /** The longest capture length carrying a fitted prefilter. */
  longestFittedLength(): number {
    let longest = 0;
    for (const length of this.models.keys()) {
      longest = Math.max(longest, length);
    }
    return longest;
  }

  modelFor(captureLength: number): NoisePrefilterV3 {
    const model = this.models.get(captureLength);
    if (model === undefined) {
      throw new RangeError(
        `no fitted noise prefilter for capture length ${captureLength}; `
        + `available ${JSON.stringify(this.lengths())}. The pose-degeneracy `
        + 'features are length dependent, so a bundle fitted at another '
        + 'length may not be substituted',
      );
    }
    return model;
  }

  /**
   * Full stage-1 evaluation of one raw capture.
   *
   * At a capture length above the longest fitted prefilter length the
   * {@link STAGE_ONE_CAUSAL_PREFIX_RULE} applies: the features are computed
   * on the capture's first `longestFittedLength()` samples (exactly a
   * capture of that length of the same emission) and gated with that
   * length's model. Any other uncovered length keeps the existing refusal.
   */
  evaluate(
    inPhase: ArrayLike<number>,
    quadrature: ArrayLike<number>,
  ): StageOneEvaluation {
    const captureLength = inPhase.length;
    if (quadrature.length !== captureLength) {
      throw new RangeError(
        'in-phase and quadrature must have the same length',
      );
    }
    const longest = this.longestFittedLength();
    const causalPrefixApplied = captureLength > longest;
    const evaluatedLength = causalPrefixApplied ? longest : captureLength;
    const model = this.modelFor(evaluatedLength);
    let gateInPhase = inPhase;
    let gateQuadrature = quadrature;
    if (causalPrefixApplied) {
      const prefixInPhase = new Float64Array(evaluatedLength);
      const prefixQuadrature = new Float64Array(evaluatedLength);
      for (let index = 0; index < evaluatedLength; index++) {
        prefixInPhase[index] = inPhase[index]!;
        prefixQuadrature[index] = quadrature[index]!;
      }
      gateInPhase = prefixInPhase;
      gateQuadrature = prefixQuadrature;
    }
    const features = prefilterPoseDegeneracyFeatures(
      gateInPhase,
      gateQuadrature,
      {
        minBandwidth: frontendMinimumBandwidth(
          evaluatedLength,
          this.patchLength,
          this.targetFrac,
        ),
      },
    );
    const score = model.score(features);
    return {
      features,
      score,
      rank: softsignRank(score),
      gated: score >= model.thresholdScore,
      thresholdScore: model.thresholdScore,
      captureLength,
      evaluatedLength,
      causalPrefixApplied,
    };
  }
}

// ---------------------------------------------------------------------------
// STAGE 2: branch LOF ranks + frozen geometry blend, q95 threshold
// ---------------------------------------------------------------------------

export interface StageTwoLofComponentAsset {
  branch: 'real' | 'complex';
  weight: number;
  neighbors: number;
  mean: number[];
  scale: number[];
  /** Standardized training reference embeddings, shape [rows, embed_dim]. */
  reference: number[][];
  reference_k_distance: number[];
  reference_local_density: number[];
  /** Sorted enrollment raw-LOF calibration values. */
  calibration: number[];
}

export interface StageTwoPolicyAsset {
  branch_lof_rank_weight: number;
  geometry_weight: number;
  threshold_quantile: number;
  threshold: number;
  geometry_feature: typeof FROZEN_GEOMETRY_FEATURE;
  /** Per-class mean of the frozen geometry feature (training fit). */
  class_geometry_mean: number[];
  /** Per-class scale of the frozen geometry feature (training fit). */
  class_geometry_scale: number[];
  /** Sorted enrollment geometry-deviation calibration. */
  geometry_calibration: number[];
  /** Sorted enrollment combined-score calibration. */
  combined_calibration: number[];
}

export interface StageTwoAsset {
  kind: typeof STAGED_POLICY_KIND;
  lof_components: StageTwoLofComponentAsset[];
  policy: StageTwoPolicyAsset;
}

export interface StageTwoLofScore {
  branch: 'real' | 'complex';
  weight: number;
  raw: number;
  rank: number;
}

export interface StageTwoEvaluation {
  lofComponents: StageTwoLofScore[];
  /** Weighted branch-LOF rank ensemble in [0, 1). */
  lofRank: number;
  /** The frozen geometry statistic (float32-rounded, as in Python). */
  geometryStatistic: number;
  geometryDeviation: number;
  geometryRank: number;
  combinedRaw: number;
  /** Enrollment-ranked stage-2 sub-score in [0, 1). */
  score: number;
  /**
   * The stage-2 policy's OWN enrollment q95, recorded for cross-checks
   * exactly as Python records `stage_two_threshold`. Under staged policy
   * version 2 the survivor DECISION compares the COMPOSITE against the
   * composite threshold, never this value.
   */
  threshold: number;
}

function lowerBound(sorted: ArrayLike<number>, value: number): number {
  let low = 0;
  let high = sorted.length;
  while (low < high) {
    const middle = low + Math.floor((high - low) / 2);
    if (sorted[middle]! < value) low = middle + 1;
    else high = middle;
  }
  return low;
}

/** Strict empirical upper-tail rank in [0, 1) (`empirical_rank`). */
export function empiricalRank(
  sortedCalibration: ArrayLike<number>,
  value: number,
): number {
  return lowerBound(sortedCalibration, value) / (sortedCalibration.length + 1);
}

/**
 * `across_patch_frequency_dispersion` from `v3_time_domain_openset.
 * time_domain_geometry`, the single frozen stage-2 geometry feature:
 * `1 - |mean_p exp(i angle(mean_t exp(i d_phi[p, t])))|`.
 *
 * Returns the float32-rounded value, matching the Python statistic matrix
 * which is cast to float32 before class deviations are computed.
 */
export function acrossPatchFrequencyDispersion(
  packedInPhase: ArrayLike<number>,
  packedQuadrature: ArrayLike<number>,
  patchCount: number,
  patchLength: number,
): number {
  const packedLength = patchCount * patchLength;
  if (
    packedInPhase.length !== packedLength
    || packedQuadrature.length !== packedLength
  ) {
    throw new RangeError(`packed I/Q must have length ${packedLength}`);
  }
  let sumCosine = 0;
  let sumSine = 0;
  for (let patch = 0; patch < patchCount; patch++) {
    const offset = patch * patchLength;
    let meanCosine = 0;
    let meanSine = 0;
    for (let time = 1; time < patchLength; time++) {
      const currentReal = packedInPhase[offset + time]!;
      const currentImaginary = packedQuadrature[offset + time]!;
      const previousReal = packedInPhase[offset + time - 1]!;
      const previousImaginary = packedQuadrature[offset + time - 1]!;
      // angle(z[t] * conj(z[t-1]))
      const productReal = (
        currentReal * previousReal + currentImaginary * previousImaginary
      );
      const productImaginary = (
        currentImaginary * previousReal - currentReal * previousImaginary
      );
      const increment = Math.atan2(productImaginary, productReal);
      meanCosine += Math.cos(increment);
      meanSine += Math.sin(increment);
    }
    meanCosine /= patchLength - 1;
    meanSine /= patchLength - 1;
    const patchFrequency = Math.atan2(meanSine, meanCosine);
    sumCosine += Math.cos(patchFrequency);
    sumSine += Math.sin(patchFrequency);
  }
  const dispersion = (
    1 - Math.hypot(sumCosine / patchCount, sumSine / patchCount)
  );
  return Math.fround(dispersion);
}

/** Score one branch embedding with one known-only LOF component. */
export function scoreStageTwoLof(
  component: StageTwoLofComponentAsset,
  embedding: ArrayLike<number>,
): StageTwoLofScore {
  const dimensions = component.mean.length;
  if (embedding.length !== dimensions) {
    throw new RangeError(
      `${component.branch} embedding must have ${dimensions} dimensions`,
    );
  }
  const query = new Float64Array(dimensions);
  let queryNorm = 0;
  for (let dimension = 0; dimension < dimensions; dimension++) {
    const value = (
      (embedding[dimension]! - component.mean[dimension]!)
      / component.scale[dimension]!
    );
    if (!Number.isFinite(value)) {
      throw new RangeError(`${component.branch} embedding must be finite`);
    }
    query[dimension] = value;
    queryNorm += value * value;
  }

  const neighbors = component.neighbors;
  const bestDistances = new Float64Array(neighbors);
  bestDistances.fill(Number.POSITIVE_INFINITY);
  const bestIndices = new Int32Array(neighbors);
  bestIndices.fill(-1);
  for (let row = 0; row < component.reference.length; row++) {
    const reference = component.reference[row]!;
    let dot = 0;
    for (let dimension = 0; dimension < dimensions; dimension++) {
      dot += query[dimension]! * reference[dimension]!;
    }
    let referenceNorm = 0;
    for (let dimension = 0; dimension < dimensions; dimension++) {
      referenceNorm += reference[dimension]! * reference[dimension]!;
    }
    // Same algebraic form as the Python neighbour search: squared distances.
    const distance = Math.max(queryNorm + referenceNorm - 2 * dot, 0);
    if (distance >= bestDistances[neighbors - 1]!) continue;
    let position = neighbors - 1;
    while (position > 0 && distance < bestDistances[position - 1]!) {
      bestDistances[position] = bestDistances[position - 1]!;
      bestIndices[position] = bestIndices[position - 1]!;
      position--;
    }
    bestDistances[position] = distance;
    bestIndices[position] = row;
  }

  let meanReachability = 0;
  let meanReferenceDensity = 0;
  for (let neighbor = 0; neighbor < neighbors; neighbor++) {
    const index = bestIndices[neighbor]!;
    if (index < 0) {
      throw new Error('LOF neighbour search did not fill its result');
    }
    meanReachability += Math.max(
      bestDistances[neighbor]!,
      component.reference_k_distance[index]!,
    );
    meanReferenceDensity += component.reference_local_density[index]!;
  }
  meanReachability /= neighbors;
  meanReferenceDensity /= neighbors;
  const queryDensity = 1 / (meanReachability + 1e-12);
  const raw = meanReferenceDensity / queryDensity;
  return {
    branch: component.branch,
    weight: component.weight,
    raw,
    rank: empiricalRank(component.calibration, raw),
  };
}

/** The frozen stage-2 rejector for stage-1 survivors. */
export class StageTwoRejectorV3 {
  readonly asset: StageTwoAsset;
  readonly patchCount: number;
  readonly patchLength: number;

  constructor(
    asset: StageTwoAsset,
    frontend: { patchCount: number; patchLength: number },
  ) {
    if (asset.kind !== STAGED_POLICY_KIND) {
      throw new RangeError(`stage two kind must be ${STAGED_POLICY_KIND}`);
    }
    const policy = asset.policy;
    if (policy.branch_lof_rank_weight !== FROZEN_BRANCH_LOF_RANK_WEIGHT) {
      throw new RangeError('stage two branch LOF rank weight is not the frozen 0.80');
    }
    if (policy.geometry_weight !== FROZEN_GEOMETRY_WEIGHT) {
      throw new RangeError('stage two geometry weight is not the frozen 0.20');
    }
    if (policy.threshold_quantile !== FROZEN_THRESHOLD_QUANTILE) {
      throw new RangeError('stage two threshold quantile is not the frozen q95');
    }
    if (policy.geometry_feature !== FROZEN_GEOMETRY_FEATURE) {
      throw new RangeError(
        `stage two geometry feature must be ${FROZEN_GEOMETRY_FEATURE}`,
      );
    }
    if (
      !Number.isFinite(policy.threshold)
      || policy.threshold < 0
      || policy.threshold >= 1
    ) {
      throw new RangeError('stage two threshold must lie in [0, 1)');
    }
    if (
      policy.class_geometry_mean.length === 0
      || policy.class_geometry_mean.length !== policy.class_geometry_scale.length
      || policy.class_geometry_scale.some(
        (value) => !Number.isFinite(value) || value <= 0,
      )
      || policy.class_geometry_mean.some((value) => !Number.isFinite(value))
    ) {
      throw new RangeError('stage two class geometry statistics are invalid');
    }
    for (const name of ['geometry_calibration', 'combined_calibration'] as const) {
      const values = policy[name];
      if (values.length === 0) {
        throw new RangeError(`stage two ${name} cannot be empty`);
      }
      for (let index = 0; index < values.length; index++) {
        if (!Number.isFinite(values[index]!)) {
          throw new RangeError(`stage two ${name}[${index}] must be finite`);
        }
        if (index > 0 && values[index]! < values[index - 1]!) {
          throw new RangeError(`stage two ${name} must be sorted`);
        }
      }
    }
    if (asset.lof_components.length === 0) {
      throw new RangeError('stage two needs at least one LOF component');
    }
    let totalWeight = 0;
    const seen = new Set<string>();
    for (const component of asset.lof_components) {
      if (component.branch !== 'real' && component.branch !== 'complex') {
        throw new RangeError('LOF component branch must be real or complex');
      }
      if (seen.has(component.branch)) {
        throw new RangeError(`duplicate LOF component for ${component.branch}`);
      }
      seen.add(component.branch);
      if (!(component.weight > 0) || !Number.isInteger(component.neighbors)) {
        throw new RangeError('LOF component weight/neighbors are invalid');
      }
      if (component.reference.length <= component.neighbors) {
        throw new RangeError('LOF reference must contain more rows than neighbors');
      }
      totalWeight += component.weight;
    }
    if (Math.abs(totalWeight - 1) > 1e-9) {
      throw new RangeError('LOF component weights must sum to one');
    }
    this.asset = asset;
    this.patchCount = frontend.patchCount;
    this.patchLength = frontend.patchLength;
  }

  get threshold(): number {
    return this.asset.policy.threshold;
  }

  get classCount(): number {
    return this.asset.policy.class_geometry_mean.length;
  }

  /**
   * Score one stage-1 survivor. `predictedClassIndex` is the closed
   * nearest-prototype label, which this method reads and never changes.
   */
  evaluate(input: {
    realEmbedding: ArrayLike<number>;
    complexEmbedding: ArrayLike<number>;
    packedInPhase: ArrayLike<number>;
    packedQuadrature: ArrayLike<number>;
    predictedClassIndex: number;
  }): StageTwoEvaluation {
    const policy = this.asset.policy;
    const predicted = input.predictedClassIndex;
    if (
      !Number.isInteger(predicted)
      || predicted < 0
      || predicted >= this.classCount
    ) {
      throw new RangeError('predictedClassIndex lies outside the fitted classes');
    }

    const lofComponents = this.asset.lof_components.map((component) =>
      scoreStageTwoLof(
        component,
        component.branch === 'real'
          ? input.realEmbedding
          : input.complexEmbedding,
      ));
    let lofRank = 0;
    for (const component of lofComponents) {
      lofRank += component.weight * component.rank;
    }

    const geometryStatistic = acrossPatchFrequencyDispersion(
      input.packedInPhase,
      input.packedQuadrature,
      this.patchCount,
      this.patchLength,
    );
    const geometryDeviation = Math.abs(
      (geometryStatistic - policy.class_geometry_mean[predicted]!)
      / policy.class_geometry_scale[predicted]!,
    );
    const geometryRank = empiricalRank(
      policy.geometry_calibration,
      geometryDeviation,
    );
    const combinedRaw = (
      policy.branch_lof_rank_weight * lofRank
      + policy.geometry_weight * geometryRank
    );
    const score = empiricalRank(policy.combined_calibration, combinedRaw);
    return {
      lofComponents,
      lofRank,
      geometryStatistic,
      geometryDeviation,
      geometryRank,
      combinedRaw,
      score,
      threshold: policy.threshold,
    };
  }
}

// ---------------------------------------------------------------------------
// the COMPOSITE survivor policy (staged policy version 2)
// ---------------------------------------------------------------------------

/**
 * Mirror of the `v3_staged_composite_policy.npz` payload
 * (`fit_v3_openset_staged.save_composite_policy`), converted to JSON by the
 * exporter. Field names keep the npz spelling exactly.
 */
export interface CompositeSurvivorPolicyAsset {
  schema: number;
  kind: string;
  policy_version: string;
  survivor_score: string;
  threshold_quantile: number;
  /** The staged decision threshold: q95 of the enrollment-survivor composite. */
  threshold: number;
  /** The stage-2 policy's own untouched enrollment q95, for cross-checks. */
  stage_two_threshold: number;
  /** Sorted stage-1 log-odds of the ENROLLMENT rows surviving the gate. */
  stage_one_calibration_raw: number[];
  /** Sorted composite of those same enrollment survivors. */
  composite_calibration_raw: number[];
  enrollment_rows: number;
  enrollment_gated_rows: number;
  enrollment_capture_length: number;
}

/**
 * NumPy's default (`method='linear'`) quantile of an already-sorted vector,
 * bit-faithful to `np.quantile`'s `_lerp`: virtual index `q * (n - 1)`, then
 * `a + t * (b - a)` below `t = 0.5` and `b - (b - a) * (1 - t)` from `t =
 * 0.5` up. Used to re-verify the stored composite threshold instead of
 * trusting it, exactly as `load_composite_policy` does.
 */
export function numpyLinearQuantile(
  sorted: ArrayLike<number>,
  quantile: number,
): number {
  const length = sorted.length;
  if (length === 0) {
    throw new RangeError('quantile of an empty vector is undefined');
  }
  const virtual = quantile * (length - 1);
  const index = Math.floor(virtual);
  const t = virtual - index;
  if (index + 1 >= length) return sorted[length - 1]!;
  const low = sorted[index]!;
  const high = sorted[index + 1]!;
  if (t < 0.5) return low + t * (high - low);
  return high - (high - low) * (1 - t);
}

function assertSortedFinite(values: number[], name: string): void {
  if (values.length === 0) {
    throw new RangeError(`${name} must be a non-empty vector`);
  }
  for (let index = 0; index < values.length; index++) {
    if (!Number.isFinite(values[index]!)) {
      throw new RangeError(`${name} must be finite`);
    }
    if (index > 0 && values[index]! < values[index - 1]!) {
      throw new RangeError(`${name} must be sorted`);
    }
  }
}

/**
 * The frozen composite survivor policy (`CompositeSurvivorPolicy`), validated
 * exactly as `fit_v3_openset_staged.load_composite_policy` validates it,
 * INCLUDING the policy-version refusal: a version-1 artifact must not be able
 * to score with version-2 semantics or vice versa.
 */
export class CompositeSurvivorPolicyV3 {
  readonly asset: CompositeSurvivorPolicyAsset;
  readonly stageOneCalibration: Float64Array;
  readonly compositeCalibration: Float64Array;
  readonly threshold: number;
  readonly stageTwoThreshold: number;

  constructor(asset: CompositeSurvivorPolicyAsset) {
    for (const [name, found, expected] of [
      ['schema', asset.schema, STAGED_POLICY_SCHEMA],
      ['kind', asset.kind, STAGED_POLICY_KIND],
      ['policy_version', asset.policy_version, STAGED_POLICY_VERSION],
      ['survivor_score', asset.survivor_score, COMPOSITE_SURVIVOR_SCORE],
    ] as const) {
      if (found !== expected) {
        throw new RangeError(
          `staged policy version mismatch: the asset records ${name}=`
          + `${JSON.stringify(found)}, but this runtime implements `
          + `${JSON.stringify(expected)}. A staged artifact may only be `
          + 'scored with the policy version that produced it',
        );
      }
    }
    if (asset.threshold_quantile !== COMPOSITE_THRESHOLD_QUANTILE) {
      throw new RangeError(
        `composite threshold_quantile ${asset.threshold_quantile} is not the `
        + `frozen ${COMPOSITE_THRESHOLD_QUANTILE}`,
      );
    }
    assertSortedFinite(
      asset.stage_one_calibration_raw,
      'stage_one_calibration_raw',
    );
    assertSortedFinite(
      asset.composite_calibration_raw,
      'composite_calibration_raw',
    );
    for (const value of asset.composite_calibration_raw) {
      if (value < 0 || value >= 1) {
        throw new RangeError('composite_calibration_raw must lie in [0, 1)');
      }
    }
    if (
      asset.stage_one_calibration_raw.length
      !== asset.composite_calibration_raw.length
    ) {
      throw new RangeError(
        'the two survivor calibrations disagree on the survivor count',
      );
    }
    if (
      !Number.isInteger(asset.enrollment_rows)
      || !Number.isInteger(asset.enrollment_gated_rows)
      || !Number.isInteger(asset.enrollment_capture_length)
      || asset.enrollment_capture_length <= 0
      || asset.enrollment_rows
        !== asset.enrollment_gated_rows
          + asset.stage_one_calibration_raw.length
    ) {
      throw new RangeError('enrollment_rows != gated + survivors');
    }
    const expectedThreshold = numpyLinearQuantile(
      asset.composite_calibration_raw,
      COMPOSITE_THRESHOLD_QUANTILE,
    );
    if (
      !Number.isFinite(asset.threshold)
      || asset.threshold !== expectedThreshold
    ) {
      throw new RangeError(
        `stored composite threshold ${asset.threshold} does not equal the `
        + 'frozen quantile of the stored survivor composite calibration '
        + `(${expectedThreshold})`,
      );
    }
    if (asset.threshold < 0 || asset.threshold >= 1) {
      throw new RangeError('composite threshold must lie in [0, 1)');
    }
    if (!Number.isFinite(asset.stage_two_threshold)) {
      throw new RangeError('stage_two_threshold must be finite');
    }
    this.asset = asset;
    this.stageOneCalibration = Float64Array.from(
      asset.stage_one_calibration_raw,
    );
    this.compositeCalibration = Float64Array.from(
      asset.composite_calibration_raw,
    );
    this.threshold = asset.threshold;
    this.stageTwoThreshold = asset.stage_two_threshold;
  }

  /**
   * Enrollment-survivor empirical rank of a stage-1 log-odds score, in
   * [0, 1) (`CompositeSurvivorPolicy.stage_one_survivor_rank`,
   * searchsorted-left as everywhere in this chain).
   */
  stageOneSurvivorRank(stageOneScore: number): number {
    if (!Number.isFinite(stageOneScore)) {
      throw new RangeError('stage-1 score must be finite');
    }
    return empiricalRank(this.stageOneCalibration, stageOneScore);
  }

  /** `max(stage2_enrollment_rank, stage1_score_enrollment_rank)`. */
  composite(stageTwoScore: number, stageOneScore: number): number {
    if (
      !Number.isFinite(stageTwoScore)
      || stageTwoScore < 0
      || stageTwoScore >= 1
    ) {
      throw new RangeError('stage-2 survivor scores must lie in [0, 1)');
    }
    return Math.max(stageTwoScore, this.stageOneSurvivorRank(stageOneScore));
  }
}

// ---------------------------------------------------------------------------
// the staged composition
// ---------------------------------------------------------------------------

export interface TimeDomainOpenSetAssetV3 {
  schema: typeof TIME_DOMAIN_OPENSET_SCHEMA;
  schema_version: typeof TIME_DOMAIN_OPENSET_SCHEMA_VERSION;
  status: TimeDomainAssetStatusV3;
  contract: StagedArchitectureContract;
  frontend: {
    patch_length: number;
    patch_count: number;
    target_frac: number;
    packed_length: number;
  };
  stage_one: StageOneAsset;
  stage_two: StageTwoAsset;
  /** Staged policy version 2: the composite survivor score and threshold. */
  composite: CompositeSurvivorPolicyAsset;
  provenance?: unknown;
}

/** Validate an untrusted JSON value into a typed staged open-set asset. */
export function loadTimeDomainOpenSetAssetV3(
  value: unknown,
  options: TimeDomainAssetLoadOptionsV3 = {},
): TimeDomainOpenSetAssetV3 {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new TypeError('open-set asset must be an object');
  }
  const asset = value as TimeDomainOpenSetAssetV3;
  if (asset.schema !== TIME_DOMAIN_OPENSET_SCHEMA) {
    throw new RangeError(`unsupported open-set schema ${String(asset.schema)}`);
  }
  if (asset.schema_version !== TIME_DOMAIN_OPENSET_SCHEMA_VERSION) {
    throw new RangeError(
      `unsupported open-set schema version ${String(asset.schema_version)}`,
    );
  }
  admitTimeDomainAssetStatusV3(asset.status, 'asset.status', options);
  const contract = asset.contract;
  if (
    contract === null
    || typeof contract !== 'object'
    || contract.additive_only !== false
    || contract.changes_closed_label !== true
    || contract.gates_before_classification !== true
    || typeof contract.architecture_contract_change !== 'string'
    || contract.architecture_contract_change.length === 0
  ) {
    throw new RangeError(
      'open-set asset must carry the staged architecture contract keys '
      + '(additive_only: false, changes_closed_label: true, '
      + 'gates_before_classification: true, architecture_contract_change)',
    );
  }
  const frontend = asset.frontend;
  if (
    frontend === null
    || typeof frontend !== 'object'
    || !Number.isInteger(frontend.patch_length)
    || !Number.isInteger(frontend.patch_count)
    || frontend.patch_length <= 0
    || frontend.patch_count <= 0
    || !(frontend.target_frac > 0 && frontend.target_frac <= 1)
    || frontend.packed_length !== frontend.patch_length * frontend.patch_count
  ) {
    throw new RangeError('open-set asset frontend geometry is invalid');
  }
  // Construction performs the deep validation of both stages and of the
  // composite policy, including the policy-version refusal.
  new StageOneGateV3(asset.stage_one, {
    patchLength: frontend.patch_length,
    targetFrac: frontend.target_frac,
  });
  new StageTwoRejectorV3(asset.stage_two, {
    patchCount: frontend.patch_count,
    patchLength: frontend.patch_length,
  });
  const composite = new CompositeSurvivorPolicyV3(asset.composite);
  // Mirror of load_composite_policy's expected_stage_two_threshold check:
  // the composite must have been fit against exactly this stage-2 state.
  if (composite.stageTwoThreshold !== asset.stage_two.policy.threshold) {
    throw new RangeError(
      `the composite policy records stage_two_threshold `
      + `${composite.stageTwoThreshold}, but the loaded stage-2 policy has `
      + `threshold ${asset.stage_two.policy.threshold}; the composite was `
      + 'fit against a different stage-2 state',
    );
  }
  return asset;
}

export type StagedRejectionStage = 1 | 2 | null;

export interface StagedOpenSetDecision {
  /** True when stage 1 gated the row as noise BEFORE classification. */
  gated: boolean;
  /** 1 = noise gate fired, 2 = composite unknown, null = accepted. */
  rejectedStage: StagedRejectionStage;
  stageOne: StageOneEvaluation;
  /** Null exactly when the row was gated: the short circuit is real. */
  stageTwo: StageTwoEvaluation | null;
  /**
   * The stage-1 log-odds' empirical enrollment-SURVIVOR rank in [0, 1)
   * (`StagedOutcome.stage_one_survivor_rank`). Null exactly when gated.
   */
  stageOneSurvivorRank: number | null;
  /**
   * The composite survivor score
   * `max(stage2_enrollment_rank, stage1_score_enrollment_rank)`
   * (`StagedOutcome.composite_score`). Null exactly when gated.
   */
  compositeScore: number | null;
  /**
   * Single staged axis (`STAGED_SCORE_NOTE`, staged policy version 2):
   * survivors carry their COMPOSITE in [0, 1); gated rows land at
   * 1 + softsign(stage-1 log-odds) in (1, 2), above every survivor.
   * `stagedScore > threshold` is exactly the staged rejection decision.
   */
  stagedScore: number;
  /** The composite threshold: frozen q95 over enrollment survivors. */
  threshold: number;
  contract: StagedArchitectureContract;
}

/**
 * The staged v3 open-set rejector: stage 1 gates, survivors flow to stage 2.
 *
 * The caller runs stage 1 BEFORE any downstream computation and only invokes
 * `finishSurvivor` when the gate did not fire, so the compute saving is real.
 */
export class TimeDomainOpenSetV3 {
  readonly stageOne: StageOneGateV3;
  readonly stageTwo: StageTwoRejectorV3;
  readonly composite: CompositeSurvivorPolicyV3;
  readonly asset: TimeDomainOpenSetAssetV3;

  constructor(asset: TimeDomainOpenSetAssetV3) {
    this.asset = asset;
    this.stageOne = new StageOneGateV3(asset.stage_one, {
      patchLength: asset.frontend.patch_length,
      targetFrac: asset.frontend.target_frac,
    });
    this.stageTwo = new StageTwoRejectorV3(asset.stage_two, {
      patchCount: asset.frontend.patch_count,
      patchLength: asset.frontend.patch_length,
    });
    this.composite = new CompositeSurvivorPolicyV3(asset.composite);
    if (this.composite.stageTwoThreshold !== asset.stage_two.policy.threshold) {
      throw new RangeError(
        'the composite policy was fit against a different stage-2 state',
      );
    }
  }

  /** Stage 1 only; run this before the encoder is ever touched. */
  evaluateStageOne(
    inPhase: ArrayLike<number>,
    quadrature: ArrayLike<number>,
  ): StageOneEvaluation {
    return this.stageOne.evaluate(inPhase, quadrature);
  }

  /** Compose the final gated decision for a row stage 1 rejected. */
  gatedDecision(stageOne: StageOneEvaluation): StagedOpenSetDecision {
    if (!stageOne.gated) {
      throw new RangeError('gatedDecision requires a fired stage-1 gate');
    }
    return {
      gated: true,
      rejectedStage: 1,
      stageOne,
      stageTwo: null,
      stageOneSurvivorRank: null,
      compositeScore: null,
      stagedScore: STAGE_ONE_SCORE_OFFSET + stageOne.rank,
      threshold: this.composite.threshold,
      contract: stagedArchitectureContract(),
    };
  }

  /**
   * Compose the final decision for a stage-1 survivor: the unchanged stage-2
   * path, then the composite `max(stage2_enrollment_rank,
   * stage1_score_enrollment_rank)` against the composite q95 threshold.
   */
  finishSurvivor(
    stageOne: StageOneEvaluation,
    survivor: {
      realEmbedding: ArrayLike<number>;
      complexEmbedding: ArrayLike<number>;
      packedInPhase: ArrayLike<number>;
      packedQuadrature: ArrayLike<number>;
      predictedClassIndex: number;
    },
  ): StagedOpenSetDecision {
    if (stageOne.gated) {
      throw new RangeError(
        'finishSurvivor was called for a gated row; the short circuit must '
        + 'be honored and stage 2 must not run',
      );
    }
    const stageTwo = this.stageTwo.evaluate(survivor);
    const stageOneSurvivorRank = this.composite.stageOneSurvivorRank(
      stageOne.score,
    );
    const compositeScore = Math.max(stageTwo.score, stageOneSurvivorRank);
    return {
      gated: false,
      rejectedStage: compositeScore > this.composite.threshold ? 2 : null,
      stageOne,
      stageTwo,
      stageOneSurvivorRank,
      compositeScore,
      stagedScore: compositeScore,
      threshold: this.composite.threshold,
      contract: stagedArchitectureContract(),
    };
  }
}
