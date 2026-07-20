/**
 * Deterministic embedding forward pass — TypeScript port of the numpy reference
 * in `training/train.py:np_forward` (which is bit-parity-verified against the
 * trained torch model). Zero runtime dependencies: plain conv/relu/pool/linear
 * over the exported, BatchNorm-folded weights. Runs where the Bayesian
 * classifier runs — no ONNX, no WASM, no GPU.
 *
 * The `!` non-null assertions in the numeric kernels satisfy the repo's
 * `noUncheckedIndexedAccess`; every index is provably in bounds by the loop
 * conditions.
 */

import { iqFeatures, N_FEATURES, type PreprocessParams } from './iq-preprocess.js';

export interface ConvLayer {
  in: number;
  out: number;
  k: number;
  stride: number;
  pad: number;
  weight: number[]; // flat [out*in*k]
  bias: number[];
}

export interface Linear {
  in: number;
  out: number;
  weight: number[]; // flat [out*in]
  bias: number[];
}

export interface EmbeddingModel {
  input_len: number;
  embed_dim: number;
  n_features: number;
  pool: 'mean_std' | 'mean';
  convs: ConvLayer[];
  fc1: Linear;
  fc2: Linear;
  feat_mean: number[];
  feat_std: number[];
  preprocess: {
    l_out: number;
    target_frac: number;
    nfft: number;
    energy_edge: number;
    noise_floor_scale: number;
    smooth: number;
  };
}

export function preprocessParams(m: EmbeddingModel): PreprocessParams {
  return {
    lOut: m.preprocess.l_out,
    targetFrac: m.preprocess.target_frac,
    nfft: m.preprocess.nfft,
    energyEdge: m.preprocess.energy_edge,
    noiseFloorScale: m.preprocess.noise_floor_scale,
    smooth: m.preprocess.smooth,
  };
}

function conv1d(chans: Float64Array[], cv: ConvLayer): Float64Array[] {
  const { in: cin, out: cout, k, stride, pad, weight: w, bias } = cv;
  const L = chans[0]!.length;
  const Lout = Math.floor((L + 2 * pad - k) / stride) + 1;
  const out: Float64Array[] = [];
  for (let co = 0; co < cout; co++) {
    const o = new Float64Array(Lout);
    const base = co * cin * k;
    for (let t = 0; t < Lout; t++) {
      let s = bias[co]!;
      const start = t * stride - pad;
      for (let ci = 0; ci < cin; ci++) {
        const ch = chans[ci]!;
        const wb = base + ci * k;
        for (let kk = 0; kk < k; kk++) {
          const xi = start + kk;
          if (xi >= 0 && xi < L) s += ch[xi]! * w[wb + kk]!;
        }
      }
      o[t] = s > 0 ? s : 0; // fused ReLU
    }
    out.push(o);
  }
  return out;
}

function meanStdPool(chans: Float64Array[]): Float64Array {
  const c = chans.length;
  const L = chans[0]!.length;
  const pooled = new Float64Array(2 * c);
  for (let ci = 0; ci < c; ci++) {
    const x = chans[ci]!;
    let m = 0;
    for (let t = 0; t < L; t++) m += x[t]!;
    m /= L;
    let v = 0;
    for (let t = 0; t < L; t++) {
      const d = x[t]! - m;
      v += d * d;
    }
    pooled[ci] = m;
    pooled[c + ci] = Math.sqrt(v / L); // population std
  }
  return pooled;
}

function meanOnly(chans: Float64Array[]): Float64Array {
  const c = chans.length;
  const L = chans[0]!.length;
  const pooled = new Float64Array(c);
  for (let ci = 0; ci < c; ci++) {
    const x = chans[ci]!;
    let m = 0;
    for (let t = 0; t < L; t++) m += x[t]!;
    pooled[ci] = m / L;
  }
  return pooled;
}

function linear(x: Float64Array, fc: Linear, relu: boolean): Float64Array {
  const out = new Float64Array(fc.out);
  for (let o = 0; o < fc.out; o++) {
    let s = fc.bias[o]!;
    const base = o * fc.in;
    for (let j = 0; j < fc.in; j++) s += fc.weight[base + j]! * x[j]!;
    out[o] = relu && s < 0 ? 0 : s;
  }
  return out;
}

/**
 * Embed already-normalised I/Q channels (length `input_len` each) into a unit
 * L2-normalised embedding. Cumulant features are computed from the same I/Q the
 * network sees, then standardised with the stored train-set statistics.
 */
export function embed(m: EmbeddingModel, i: Float64Array, q: Float64Array): Float64Array {
  const raw = iqFeatures(i, q);
  const feat = standardizeFeatures(m, raw);
  return forwardChannels(m, [i, q], feat);
}

/** Standardize a raw feature vector with the model's stored train-set stats. */
export function standardizeFeatures(m: EmbeddingModel, raw: Float64Array | number[]): Float64Array {
  const out = new Float64Array(m.n_features);
  for (let f = 0; f < m.n_features; f++) out[f] = (raw[f]! - m.feat_mean[f]!) / m.feat_std[f]!;
  return out;
}

/**
 * Flavor-agnostic forward pass: conv over the input channels, mean+std pool,
 * concatenate the (already-standardized) features, project, L2-normalise. Used
 * by both the I/Q flavor ([I,Q] channels + cumulant features) and the magnitude
 * flavor ([log-spectrum] channel + spectral features).
 */
export function forwardChannels(m: EmbeddingModel, channels: Float64Array[], featStandardized: Float64Array): Float64Array {
  let chans: Float64Array[] = channels.map((c) => Float64Array.from(c));
  for (const cv of m.convs) chans = conv1d(chans, cv);
  const pooled = m.pool === 'mean_std' ? meanStdPool(chans) : meanOnly(chans);
  const h = new Float64Array(pooled.length + featStandardized.length);
  h.set(pooled, 0);
  h.set(featStandardized, pooled.length);
  const a = linear(h, m.fc1, true);
  const z = linear(a, m.fc2, false);
  let norm = 0;
  for (let d = 0; d < z.length; d++) norm += z[d]! * z[d]!;
  norm = Math.sqrt(norm) + 1e-12;
  for (let d = 0; d < z.length; d++) z[d]! /= norm;
  return z;
}
