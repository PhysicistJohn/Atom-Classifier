/**
 * Magnitude-only front-end — TypeScript port of `training/magnitude.py`.
 *
 * Reduces a power spectrum (a Welch PSD from I/Q, or a tinySA sweep) to a
 * scale-invariant log-shape [MAG_LEN] + spectral features, so the same embedding
 * runs on a scalar spectrum analyzer. `representationFromPsd` is the exact mirror
 * of the Python and is what the tinySA path calls on its swept power; a parity
 * test guards it.
 */

import { welchPsd, smoothSame, estimateBand, type PreprocessParams } from './iq-preprocess.js';

export const MAG_LEN = 256;
export const MAG_NFFT = 1024;
export const MARGIN = 0.75;
export const N_MAG_FEATURES = 8;

const MAG_PARAMS: PreprocessParams = {
  lOut: MAG_LEN,
  targetFrac: 0.5,
  nfft: MAG_NFFT,
  energyEdge: 0.005,
  noiseFloorScale: 1.44,
  smooth: 5,
};

function linResampleReal(x: Float64Array, newLen: number): Float64Array {
  const n = x.length;
  if (n === newLen) return x;
  if (n < 2) {
    const out = new Float64Array(newLen);
    out.fill(x[0] ?? 0);
    return out;
  }
  const out = new Float64Array(newLen);
  const step = (n - 1) / (newLen - 1);
  for (let m = 0; m < newLen; m++) {
    const pos = m * step;
    let i0 = Math.floor(pos);
    if (i0 > n - 2) i0 = n - 2;
    const frac = pos - i0;
    out[m] = x[i0]! * (1 - frac) + x[i0 + 1]! * frac;
  }
  return out;
}

function median(x: Float64Array): number {
  const s = Float64Array.from(x).sort();
  const n = s.length;
  return n % 2 ? s[(n - 1) >> 1]! : 0.5 * (s[n / 2 - 1]! + s[n / 2]!);
}

export interface MagnitudeRepresentation {
  shape: Float64Array; // canonical log-spectrum shape in [0,1], length MAG_LEN
  features: Float64Array; // N_MAG_FEATURES spectral scalars
}

/**
 * Reduce a (linear, fftshifted) power spectrum + occupied band to the canonical
 * magnitude representation. Identical maths to Python `representation_from_psd`.
 */
export function representationFromPsd(psd: Float64Array, center: number, bw: number): MagnitudeRepresentation {
  const nfft = psd.length;
  const half = bw * (0.5 + MARGIN);
  let lo = Math.round((center - half + 0.5) * nfft);
  let hi = Math.round((center + half + 0.5) * nfft);
  lo = Math.min(Math.max(lo, 0), nfft - 1);
  hi = Math.min(Math.max(hi, lo + 2), nfft);
  const bandN = hi - lo;
  const band = new Float64Array(bandN);
  let bandSum = 0;
  let bandMax = 0;
  for (let k = 0; k < bandN; k++) {
    const v = psd[lo + k]! + 1e-12;
    band[k] = v;
    bandSum += v;
    if (v > bandMax) bandMax = v;
  }

  // scale-invariant shape: relative-dB above the noise floor, peak-normalised
  const logband = new Float64Array(bandN);
  for (let k = 0; k < bandN; k++) logband[k] = 10 * Math.log10(band[k]!);
  const med = median(logband);
  let peak = 1e-9;
  for (let k = 0; k < bandN; k++) {
    const v = Math.max(logband[k]! - med, 0);
    logband[k] = v;
    if (v > peak) peak = v;
  }
  for (let k = 0; k < bandN; k++) logband[k]! /= peak;
  const shape = linResampleReal(logband, MAG_LEN);

  // spectral features (all from the power spectrum)
  let centroid = 0;
  for (let k = 0; k < bandN; k++) centroid += (bandN > 1 ? k / (bandN - 1) : 0) * (band[k]! / bandSum);
  let spread = 0;
  for (let k = 0; k < bandN; k++) {
    const x = bandN > 1 ? k / (bandN - 1) : 0;
    spread += (x - centroid) ** 2 * (band[k]! / bandSum);
  }
  spread = Math.sqrt(spread) + 1e-9;
  let skew = 0;
  let kurt = 0;
  for (let k = 0; k < bandN; k++) {
    const x = bandN > 1 ? k / (bandN - 1) : 0;
    skew += (x - centroid) ** 3 * (band[k]! / bandSum);
    kurt += (x - centroid) ** 4 * (band[k]! / bandSum);
  }
  skew /= spread ** 3;
  kurt /= spread ** 4;
  let logMean = 0;
  for (let k = 0; k < bandN; k++) logMean += Math.log(band[k]!);
  const flatness = Math.exp(logMean / bandN) / (bandSum / bandN + 1e-12);
  const papr = bandMax / (bandSum / bandN + 1e-12);
  const c0 = Math.floor(bandN * 0.45);
  const c1 = Math.floor(bandN * 0.55);
  let central = 0;
  for (let k = c0; k < c1; k++) central += band[k]!;
  const fracCentral = central / bandSum;

  const features = Float64Array.from([
    bw, flatness, Math.log1p(papr), spread, skew, kurt, fracCentral, bandN / nfft,
  ]);
  return { shape, features };
}

/** Training/inference-from-I/Q path: complex I/Q -> Welch PSD -> representation. */
export function magnitudeFromIq(re: Float64Array, im: Float64Array): MagnitudeRepresentation {
  const psd = smoothSame(welchPsd(re, im, MAG_NFFT), MAG_PARAMS.smooth);
  const { center, bw } = estimateBand(re, im, MAG_PARAMS);
  return representationFromPsd(psd, center, bw);
}
