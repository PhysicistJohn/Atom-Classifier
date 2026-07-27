/**
 * DSP front-end — TypeScript port of `training/preprocess.py`.
 *
 * seam-safe hybrid occupied-band detection -> down-convert to baseband ->
 * resample to a canonical fractional bandwidth -> amplitude-normalise, plus the
 * phase-invariant cumulant / instantaneous feature vector the embedding head
 * consumes. This mirrors the Python so the trained model behaves identically at
 * inference; `iq-preprocess.test.ts` asserts parity against exported fixtures.
 *
 * Complex I/Q is carried as separate real/imag Float64Array pairs. Everything is
 * double precision until the final channel cast, matching numpy. The `!`
 * non-null assertions in the numeric kernels satisfy the repo's
 * `noUncheckedIndexedAccess`; every index is provably in bounds by construction.
 */

import {
  resampleComplexLinear,
  welchPowerSpectrumPeriodicHann,
} from '@atomos/dsp';

export interface PreprocessParams {
  /**
   * Missing means linear-v1 for backward compatibility with existing serialized
   * model metadata. DEFAULT_PARAMS explicitly selects hybrid-v3.
   */
  version?: 'linear-v1' | 'circular-v2' | 'hybrid-v3';
  lOut: number;
  targetFrac: number;
  nfft: number;
  energyEdge: number;
  /** @deprecated Retained for compatibility with linear-v1 model metadata. */
  noiseFloorScale: number;
  smooth: number;
  fullBandBw?: number;
  whiteFlatnessDeficit?: number;
  guardFloorScale?: number;
  minExcessFraction?: number;
  hybridLegacyWideBw?: number;
  hybridCircularBroadBw?: number;
  hybridSeamToleranceBins?: number;
}

export const FULL_BAND_BW = 0.95;
export const WHITE_FLATNESS_DEFICIT = 0.30;
export const GUARD_FLOOR_SCALE = 1.30;
export const MIN_EXCESS_FRACTION = 1e-6;
export const HYBRID_LEGACY_WIDE_BW = 0.90;
export const HYBRID_CIRCULAR_BROAD_BW = 0.65;
export const HYBRID_SEAM_TOLERANCE_BINS = 1.0;

export const DEFAULT_PARAMS: PreprocessParams = {
  version: 'hybrid-v3',
  lOut: 1024,
  targetFrac: 0.5,
  nfft: 512,
  energyEdge: 0.005,
  noiseFloorScale: 1.44,
  smooth: 5,
  fullBandBw: FULL_BAND_BW,
  whiteFlatnessDeficit: WHITE_FLATNESS_DEFICIT,
  guardFloorScale: GUARD_FLOOR_SCALE,
  minExcessFraction: MIN_EXCESS_FRACTION,
  hybridLegacyWideBw: HYBRID_LEGACY_WIDE_BW,
  hybridCircularBroadBw: HYBRID_CIRCULAR_BROAD_BW,
  hybridSeamToleranceBins: HYBRID_SEAM_TOLERANCE_BINS,
};

export interface Normalised {
  i: Float64Array;
  q: Float64Array;
  center: number;
  bw: number;
  resampleFrac: number;
}

export interface PreprocessOptions {
  forceCenter?: number;
  forceBw?: number;
  forceResampleFrac?: number;
}

/** Averaged periodogram, fftshifted (index 0 = most-negative frequency). */
export function welchPsd(i: Float64Array, q: Float64Array, nfft: number): Float64Array {
  return welchPowerSpectrumPeriodicHann(i, q, nfft);
}

export function smoothSame(x: Float64Array, w: number): Float64Array {
  if (w <= 1) return x;
  const n = x.length;
  const off = (w - 1) >> 1; // numpy 'same' central slice start
  const out = new Float64Array(n);
  for (let idx = 0; idx < n; idx++) {
    const fi = idx + off; // index into the full convolution
    let s = 0;
    for (let k = 0; k < w; k++) {
      const xi = fi - k;
      if (xi >= 0 && xi < n) s += x[xi]!;
    }
    out[idx] = s / w;
  }
  return out;
}

/** Periodic moving average for an fftshifted spectrum on the frequency circle. */
export function smoothCircular(x: Float64Array, w: number): Float64Array {
  if (w <= 1) return x;
  const n = x.length;
  if (!Number.isInteger(w) || w > n) {
    throw new RangeError('smoothing width must be an integer no greater than the PSD length');
  }
  const out = new Float64Array(n);
  const half = Math.floor(w / 2);
  for (let idx = 0; idx < n; idx++) {
    let sum = 0;
    for (let k = 0; k < w; k++) {
      const offset = k - half;
      let source = (idx - offset) % n;
      if (source < 0) source += n;
      sum += x[source]!;
    }
    out[idx] = sum / w;
  }
  return out;
}

function median(x: Float64Array): number {
  const s = Float64Array.from(x).sort();
  const n = s.length;
  return n % 2 ? s[(n - 1) >> 1]! : 0.5 * (s[n / 2 - 1]! + s[n / 2]!);
}

/** numpy searchsorted(a, v, side='left'): first i with a[i] >= v. */
function searchsortedLeft(a: Float64Array, v: number): number {
  let lo = 0;
  let hi = a.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (a[mid]! < v) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function peakNormalised(
  i: Float64Array,
  q: Float64Array,
): { i: Float64Array; q: Float64Array; peak: number } {
  if (i.length === 0 || i.length !== q.length) {
    throw new RangeError('I/Q channels must have the same non-zero length');
  }
  let peak = 0;
  for (let k = 0; k < i.length; k++) {
    const re = i[k]!;
    const im = q[k]!;
    if (!Number.isFinite(re) || !Number.isFinite(im)) {
      throw new RangeError('I/Q capture contains NaN or infinity');
    }
    peak = Math.max(peak, Math.hypot(re, im));
  }
  if (!Number.isFinite(peak)) {
    throw new RangeError('I/Q magnitude is not finite');
  }
  const si = new Float64Array(i.length);
  const sq = new Float64Array(q.length);
  if (peak > 0) {
    for (let k = 0; k < i.length; k++) {
      si[k] = i[k]! / peak;
      sq[k] = q[k]! / peak;
    }
  }
  return { i: si, q: sq, peak };
}

const FLOAT64_TINY = 2.2250738585072014e-308;

function spectralFlatness(psd: Float64Array): number {
  let sum = 0;
  let logSum = 0;
  for (const value of psd) {
    sum += value;
    logSum += Math.log(Math.max(value, FLOAT64_TINY));
  }
  const mean = sum / psd.length;
  if (mean <= 0) return 1;
  return Math.exp(logSum / psd.length) / mean;
}

function welchSegmentCount(length: number, nfft: number): number {
  if (length < nfft) return 1;
  return 1 + Math.floor((length - nfft) / (nfft / 2));
}

function minimumCircularGuard(
  psd: Float64Array,
  smooth: number,
  fullBandBw: number,
): Float64Array {
  const n = psd.length;
  const guardBins = Math.min(
    n,
    Math.max(2 * smooth + 1, Math.ceil((1 - fullBandBw) * n)),
  );
  let bestStart = 0;
  let bestSum = Number.POSITIVE_INFINITY;
  for (let start = 0; start < n; start++) {
    let sum = 0;
    for (let offset = 0; offset < guardBins; offset++) {
      sum += psd[(start + offset) % n]!;
    }
    if (sum < bestSum) {
      bestSum = sum;
      bestStart = start;
    }
  }
  const guard = new Float64Array(guardBins);
  for (let offset = 0; offset < guardBins; offset++) {
    guard[offset] = psd[(bestStart + offset) % n]!;
  }
  return guard;
}

function shortestCircularInterval(
  weights: Float64Array,
  retainedFraction: number,
): { start: number; end: number } {
  const n = weights.length;
  let total = 0;
  for (const weight of weights) total += weight;
  if (total <= 0) return { start: 0, end: n - 1 };

  const target = retainedFraction * total;
  const doubled = new Float64Array(2 * n);
  doubled.set(weights, 0);
  doubled.set(weights, n);
  let endExclusive = 0;
  let running = 0;
  let bestWidth = Number.POSITIVE_INFINITY;
  let bestOvershoot = Number.POSITIVE_INFINITY;
  let bestStart = 0;
  let bestEnd = n - 1;

  for (let start = 0; start < n; start++) {
    if (endExclusive < start) {
      endExclusive = start;
      running = 0;
    }
    while (endExclusive < start + n && running < target) {
      running += doubled[endExclusive]!;
      endExclusive++;
    }
    if (running >= target) {
      const width = endExclusive - start;
      const overshoot = running - target;
      if (width < bestWidth || (width === bestWidth && overshoot < bestOvershoot)) {
        bestWidth = width;
        bestOvershoot = overshoot;
        bestStart = start;
        bestEnd = endExclusive - 1;
      }
    }
    if (endExclusive > start) {
      running -= doubled[start]!;
      if (running < 0 && running > -1e-12 * total) running = 0;
    }
  }
  return { start: bestStart, end: bestEnd };
}

function wrapCycles(value: number): number {
  return value - Math.floor(value + 0.5);
}

export function estimateBandLinearV1(
  i: Float64Array,
  q: Float64Array,
  p: PreprocessParams,
): { center: number; bw: number } {
  // Validate without changing the absolute-power behavior of the legacy path.
  peakNormalised(i, q);
  const nfft = p.nfft;
  const psd = smoothSame(welchPsd(i, q, nfft), p.smooth);
  const floor = p.noiseFloorScale * median(psd);
  const sig = new Float64Array(nfft);
  let total = 0;
  for (let k = 0; k < nfft; k++) {
    const value = Math.max(psd[k]! - floor, 0);
    sig[k] = value;
    total += value;
  }
  if (total < 1e-9) return { center: 0, bw: FULL_BAND_BW };
  const cumulative = new Float64Array(nfft);
  let running = 0;
  for (let k = 0; k < nfft; k++) {
    running += sig[k]! / total;
    cumulative[k] = running;
  }
  let lo = searchsortedLeft(cumulative, p.energyEdge);
  let hi = searchsortedLeft(cumulative, 1 - p.energyEdge);
  lo = Math.min(lo, nfft - 1);
  hi = Math.min(Math.max(hi, lo + 1), nfft - 1);
  const fLo = lo / nfft - 0.5;
  const fHi = hi / nfft - 0.5;
  return {
    center: 0.5 * (fLo + fHi),
    bw: Math.max(fHi - fLo, 1 / nfft),
  };
}

export function estimateBandCircularV2(
  i: Float64Array,
  q: Float64Array,
  p: PreprocessParams,
): { center: number; bw: number } {
  const nfft = p.nfft;
  if (
    !Number.isInteger(nfft)
    || nfft < 2 * p.smooth + 1
    || (nfft & (nfft - 1)) !== 0
    || !Number.isInteger(p.smooth)
    || p.smooth < 1
  ) {
    throw new RangeError(`nfft must be a power of two >= ${2 * p.smooth + 1}`);
  }
  if (!(p.energyEdge >= 0 && p.energyEdge < 0.5)) {
    throw new RangeError('energyEdge must lie in [0, 0.5)');
  }

  const fullBandBw = p.fullBandBw ?? FULL_BAND_BW;
  const flatnessDeficit = p.whiteFlatnessDeficit ?? WHITE_FLATNESS_DEFICIT;
  const guardFloorScale = p.guardFloorScale ?? GUARD_FLOOR_SCALE;
  const minExcessFraction = p.minExcessFraction ?? MIN_EXCESS_FRACTION;
  if (!(fullBandBw > 0 && fullBandBw <= 1)) {
    throw new RangeError('fullBandBw must lie in (0, 1]');
  }

  const scaled = peakNormalised(i, q);
  if (scaled.peak === 0) return { center: 0, bw: fullBandBw };
  const psd = smoothCircular(welchPsd(scaled.i, scaled.q, nfft), p.smooth);
  let psdTotal = 0;
  for (let k = 0; k < nfft; k++) {
    psd[k] = Math.max(psd[k]!, 0);
    psdTotal += psd[k]!;
  }
  if (psdTotal <= 0) return { center: 0, bw: fullBandBw };

  const count = welchSegmentCount(i.length, nfft);
  const flatnessThreshold = Math.max(0, 1 - flatnessDeficit / count);
  if (spectralFlatness(psd) >= flatnessThreshold) {
    return { center: 0, bw: fullBandBw };
  }

  const guard = minimumCircularGuard(psd, p.smooth, fullBandBw);
  const floor = guardFloorScale * median(guard);
  const excess = new Float64Array(nfft);
  let excessTotal = 0;
  for (let k = 0; k < nfft; k++) {
    const value = Math.max(psd[k]! - floor, 0);
    excess[k] = value;
    excessTotal += value;
  }
  if (excessTotal / psdTotal < minExcessFraction) {
    return { center: 0, bw: fullBandBw };
  }

  const interval = shortestCircularInterval(excess, 1 - 2 * p.energyEdge);
  const bw = Math.min(
    Math.max((interval.end - interval.start) / nfft, 1 / nfft),
    fullBandBw,
  );
  const center = wrapCycles(0.5 * (interval.start + interval.end) / nfft - 0.5);
  return { center, bw };
}

function hybridUsesCircular(
  legacy: { center: number; bw: number },
  circular: { center: number; bw: number },
  p: PreprocessParams,
): boolean {
  const seamToleranceBins = p.hybridSeamToleranceBins ?? HYBRID_SEAM_TOLERANCE_BINS;
  const legacyWideBw = p.hybridLegacyWideBw ?? HYBRID_LEGACY_WIDE_BW;
  const circularBroadBw = p.hybridCircularBroadBw ?? HYBRID_CIRCULAR_BROAD_BW;
  const seamLimit = 0.5 - seamToleranceBins / p.nfft;
  const crossesSeam = Math.abs(circular.center) + 0.5 * circular.bw >= seamLimit;
  return crossesSeam && (legacy.bw >= legacyWideBw || circular.bw >= circularBroadBw);
}

export function estimateBand(
  i: Float64Array,
  q: Float64Array,
  p: PreprocessParams,
): { center: number; bw: number } {
  const version = p.version ?? 'linear-v1';
  if (version === 'linear-v1') return estimateBandLinearV1(i, q, p);
  if (version === 'circular-v2') return estimateBandCircularV2(i, q, p);

  // Hybrid routing sees a canonical gain so its linear candidate cannot change
  // merely because input power crosses linear-v1's historical absolute epsilon.
  const scaled = peakNormalised(i, q);
  if (scaled.peak === 0) {
    return { center: 0, bw: p.fullBandBw ?? FULL_BAND_BW };
  }
  const legacy = estimateBandLinearV1(scaled.i, scaled.q, p);
  const circular = estimateBandCircularV2(scaled.i, scaled.q, p);
  const fullBandBw = p.fullBandBw ?? FULL_BAND_BW;
  // Common zero/white-noise abstention; this is not a circular seam route.
  if (circular.center === 0 && circular.bw === fullBandBw) return circular;
  return hybridUsesCircular(legacy, circular, p) ? circular : legacy;
}

/** Complex linear-interpolation resampler (mirror of preprocess.lin_resample). */
export function linResample(
  i: Float64Array,
  q: Float64Array,
  newLen: number,
): { i: Float64Array; q: Float64Array } {
  const { real, imaginary } = resampleComplexLinear(i, q, newLen);
  return { i: real, q: imaginary };
}

function centerFit(x: Float64Array, length: number): Float64Array {
  const n = x.length;
  if (n === length) return x;
  if (n > length) {
    const start = (n - length) >> 1;
    return x.slice(start, start + length);
  }
  const out = new Float64Array(length);
  const left = (length - n) >> 1;
  out.set(x, left);
  return out;
}

export function preprocess(
  reIn: Float64Array,
  imIn: Float64Array,
  params: PreprocessParams = DEFAULT_PARAMS,
  options: PreprocessOptions = {},
): Normalised {
  if (!Number.isInteger(params.lOut) || params.lOut <= 0) {
    throw new RangeError('lOut must be a positive integer');
  }
  if (!(Number.isFinite(params.targetFrac) && params.targetFrac > 0)) {
    throw new RangeError('targetFrac must be finite and positive');
  }
  const stableGain = (params.version ?? 'linear-v1') !== 'linear-v1';
  const detected = options.forceCenter === undefined || options.forceBw === undefined
    ? estimateBand(reIn, imIn, params)
    : { center: 0, bw: params.fullBandBw ?? FULL_BAND_BW };
  const center = wrapCycles(options.forceCenter ?? detected.center);
  const bw = options.forceBw ?? detected.bw;
  if (!Number.isFinite(center)) throw new RangeError('center must be finite');
  if (!(Number.isFinite(bw) && bw > 0)) {
    throw new RangeError('bandwidth must be finite and positive');
  }

  const n = reIn.length;
  const checked = peakNormalised(reIn, imIn);
  const sourceI = stableGain ? checked.i : reIn;
  const sourceQ = stableGain ? checked.q : imIn;
  // down-convert measured centre to DC
  const ci = new Float64Array(n);
  const cq = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const ph = -2 * Math.PI * center * k;
    const c = Math.cos(ph);
    const s = Math.sin(ph);
    ci[k] = sourceI[k]! * c - sourceQ[k]! * s;
    cq[k] = sourceI[k]! * s + sourceQ[k]! * c;
  }
  // resample so occupied bandwidth hits the canonical target
  const requestedFrac = options.forceResampleFrac ?? bw;
  if (!(Number.isFinite(requestedFrac) && requestedFrac > 0)) {
    throw new RangeError('resample fraction must be finite and positive');
  }
  const frac = Math.min(
    Math.max(requestedFrac, 1e-3),
    params.fullBandBw ?? FULL_BAND_BW,
  );
  const ratio = frac / params.targetFrac;
  const newLen = Math.max(64, Math.round(n * ratio));
  const rs = linResample(ci, cq, newLen);
  const di: Float64Array = centerFit(rs.i, params.lOut);
  const dq: Float64Array = centerFit(rs.q, params.lOut);
  if (stableGain) {
    // Peak-normalize first so the RMS decision has no absolute-scale epsilon.
    let outputPeak = 0;
    for (let k = 0; k < params.lOut; k++) {
      outputPeak = Math.max(outputPeak, Math.hypot(di[k]!, dq[k]!));
    }
    if (outputPeak > 0) {
      let power = 0;
      for (let k = 0; k < params.lOut; k++) {
        di[k] = di[k]! / outputPeak;
        dq[k] = dq[k]! / outputPeak;
        power += di[k]! * di[k]! + dq[k]! * dq[k]!;
      }
      const rms = Math.sqrt(power / params.lOut);
      for (let k = 0; k < params.lOut; k++) {
        di[k] = Math.fround(di[k]! / rms);
        dq[k] = Math.fround(dq[k]! / rms);
      }
    }
  } else {
    let power = 0;
    for (let k = 0; k < params.lOut; k++) {
      power += di[k]! * di[k]! + dq[k]! * dq[k]!;
    }
    const rms = Math.sqrt(power / params.lOut + 1e-12);
    for (let k = 0; k < params.lOut; k++) {
      di[k] = Math.fround(di[k]! / rms);
      dq[k] = Math.fround(dq[k]! / rms);
    }
  }
  return { i: di, q: dq, center, bw, resampleFrac: frac };
}

function unwrap(p: Float64Array): Float64Array {
  const n = p.length;
  const out = new Float64Array(n);
  if (n === 0) return out;
  out[0] = p[0]!;
  let corr = 0;
  const twoPi = 2 * Math.PI;
  for (let k = 1; k < n; k++) {
    const dd = p[k]! - p[k - 1]!;
    // floor-based modulo to match numpy's np.mod (JS `%` keeps the sign of dd)
    const shifted = dd + Math.PI;
    let ddmod = shifted - twoPi * Math.floor(shifted / twoPi) - Math.PI;
    if (ddmod === -Math.PI && dd > 0) ddmod = Math.PI;
    let phc = ddmod - dd;
    if (Math.abs(dd) < Math.PI) phc = 0;
    corr += phc;
    out[k] = p[k]! + corr;
  }
  return out;
}

export const N_FEATURES = 12;

/**
 * Phase-rotation-invariant higher-order + instantaneous statistics.
 * Mirror of preprocess.iq_features (computed on the normalised I/Q).
 */
export function iqFeatures(iIn: Float64Array, qIn: Float64Array): Float64Array {
  const n = iIn.length;
  // zero-mean
  let miReal = 0;
  let miImag = 0;
  for (let k = 0; k < n; k++) {
    miReal += iIn[k]!;
    miImag += qIn[k]!;
  }
  miReal /= n;
  miImag /= n;
  const zr = new Float64Array(n);
  const zi = new Float64Array(n);
  let pw = 0;
  for (let k = 0; k < n; k++) {
    const a = iIn[k]! - miReal;
    const b = qIn[k]! - miImag;
    zr[k] = a;
    zi[k] = b;
    pw += a * a + b * b;
  }
  const p = Math.sqrt(pw / n) + 1e-12;
  for (let k = 0; k < n; k++) {
    zr[k] = zr[k]! / p;
    zi[k] = zi[k]! / p;
  }
  // complex moment accumulators
  let m20r = 0, m20i = 0;
  let m40r = 0, m40i = 0;
  let m41r = 0, m41i = 0;
  let m42 = 0;
  let m60r = 0, m60i = 0;
  let m63 = 0;
  let sumAbs = 0, sumAbs2 = 0, maxAbs2 = 0;
  for (let k = 0; k < n; k++) {
    const a = zr[k]!;
    const b = zi[k]!;
    const a2 = a * a + b * b; // |z|^2
    const z2r = a * a - b * b; // z^2
    const z2i = 2 * a * b;
    m20r += z2r; m20i += z2i;
    const z4r = z2r * z2r - z2i * z2i; // z^4
    const z4i = 2 * z2r * z2i;
    m40r += z4r; m40i += z4i;
    m41r += z2r * a2; m41i += z2i * a2; // z^3 conj(z) = z^2 |z|^2
    m42 += a2 * a2;
    const z6r = z4r * z2r - z4i * z2i; // z^6
    const z6i = z4r * z2i + z4i * z2r;
    m60r += z6r; m60i += z6i;
    m63 += a2 * a2 * a2;
    sumAbs += Math.sqrt(a2);
    sumAbs2 += a2;
    if (a2 > maxAbs2) maxAbs2 = a2;
  }
  m20r /= n; m20i /= n;
  m40r /= n; m40i /= n;
  m41r /= n; m41i /= n;
  m42 /= n;
  m60r /= n; m60i /= n;
  m63 /= n;
  const cAbs = (r: number, im: number) => Math.sqrt(r * r + im * im);
  const c20 = cAbs(m20r, m20i);
  const m20sqR = m20r * m20r - m20i * m20i;
  const m20sqI = 2 * m20r * m20i;
  const c40 = cAbs(m40r - 3 * m20sqR, m40i - 3 * m20sqI);
  const c41 = cAbs(m41r - 3 * m20r, m41i - 3 * m20i);
  const c42 = m42 - (m20r * m20r + m20i * m20i) - 2;
  const m20m40R = m20r * m40r - m20i * m40i;
  const m20m40I = m20r * m40i + m20i * m40r;
  const m20cubeR = m20sqR * m20r - m20sqI * m20i;
  const m20cubeI = m20sqR * m20i + m20sqI * m20r;
  const c60 = cAbs(m60r - 15 * m20m40R + 30 * m20cubeR, m60i - 15 * m20m40I + 30 * m20cubeI);
  const c63 = m63 - 9 * c42 - 6;
  const meanAbs = sumAbs / n;
  const varAbs = sumAbs2 / n - meanAbs * meanAbs;
  const stdAbs = Math.sqrt(Math.max(varAbs, 0));
  // instantaneous frequency spread
  const ang = new Float64Array(n);
  for (let k = 0; k < n; k++) ang[k] = Math.atan2(zi[k]!, zr[k]!);
  const uw = unwrap(ang);
  let ifMean = 0;
  for (let k = 1; k < n; k++) ifMean += uw[k]! - uw[k - 1]!;
  ifMean /= n - 1;
  let ifVar = 0;
  for (let k = 1; k < n; k++) {
    const d = uw[k]! - uw[k - 1]! - ifMean;
    ifVar += d * d;
  }
  ifVar /= n - 1;
  const stdIf = Math.sqrt(Math.max(ifVar, 0));
  const cov = stdAbs / (meanAbs + 1e-9);
  return Float64Array.from([
    c20, c40, c42, c41, c60, c63, m42, stdAbs, maxAbs2, meanAbs, stdIf, cov,
  ]);
}
