/**
 * DSP front-end — TypeScript port of `training/preprocess.py`.
 *
 * detect (occupied band) -> down-convert to baseband -> resample to a canonical
 * fractional bandwidth -> amplitude-normalise, plus the phase-invariant cumulant
 * / instantaneous feature vector the embedding head consumes. This is a
 * line-for-line mirror of the Python so the trained model behaves identically at
 * inference; `iq-preprocess.test.ts` asserts parity against exported fixtures.
 *
 * Complex I/Q is carried as separate real/imag Float64Array pairs. Everything is
 * double precision until the final channel cast, matching numpy. The `!`
 * non-null assertions in the numeric kernels satisfy the repo's
 * `noUncheckedIndexedAccess`; every index is provably in bounds by construction.
 */

export interface PreprocessParams {
  lOut: number;
  targetFrac: number;
  nfft: number;
  energyEdge: number;
  noiseFloorScale: number;
  smooth: number;
}

export const DEFAULT_PARAMS: PreprocessParams = {
  lOut: 1024,
  targetFrac: 0.5,
  nfft: 512,
  energyEdge: 0.005,
  noiseFloorScale: 1.44,
  smooth: 5,
};

export interface Normalised {
  i: Float64Array;
  q: Float64Array;
  center: number;
  bw: number;
}

/** In-place iterative radix-2 Cooley–Tukey FFT (nfft must be a power of two). */
function fft(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tr = re[i]!;
      re[i] = re[j]!;
      re[j] = tr;
      const ti = im[i]!;
      im[i] = im[j]!;
      im[j] = ti;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang);
    const wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1;
      let ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k;
        const b = i + k + len / 2;
        const rb = re[b]!;
        const ib = im[b]!;
        const ra = re[a]!;
        const ia = im[a]!;
        const tr = rb * cr - ib * ci;
        const ti = rb * ci + ib * cr;
        re[b] = ra - tr;
        im[b] = ia - ti;
        re[a] = ra + tr;
        im[a] = ia + ti;
        const ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = ncr;
      }
    }
  }
}

function hann(n: number): Float64Array {
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n);
  return w;
}

/** Averaged periodogram, fftshifted (index 0 = most-negative frequency). */
export function welchPsd(i: Float64Array, q: Float64Array, nfft: number): Float64Array {
  const win = hann(nfft);
  const hop = nfft >> 1;
  let n = i.length;
  let ir: Float64Array = i;
  let qr: Float64Array = q;
  if (n < nfft) {
    ir = new Float64Array(nfft);
    qr = new Float64Array(nfft);
    ir.set(i);
    qr.set(q);
    n = nfft;
  }
  const acc = new Float64Array(nfft);
  let count = 0;
  for (let start = 0; start + nfft <= n; start += hop) {
    const re = new Float64Array(nfft);
    const im = new Float64Array(nfft);
    for (let k = 0; k < nfft; k++) {
      re[k] = ir[start + k]! * win[k]!;
      im[k] = qr[start + k]! * win[k]!;
    }
    fft(re, im);
    for (let k = 0; k < nfft; k++) acc[k] = acc[k]! + re[k]! * re[k]! + im[k]! * im[k]!;
    count++;
  }
  if (count === 0) count = 1;
  for (let k = 0; k < nfft; k++) acc[k] = acc[k]! / count;
  // fftshift (even nfft): swap halves
  const half = nfft >> 1;
  const out = new Float64Array(nfft);
  out.set(acc.subarray(half), 0);
  out.set(acc.subarray(0, half), half);
  return out;
}

function smoothSame(x: Float64Array, w: number): Float64Array {
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

function median(x: Float64Array): number {
  const s = Float64Array.from(x).sort();
  const n = s.length;
  return n % 2 ? s[(n - 1) >> 1]! : 0.5 * (s[n / 2 - 1]! + s[n / 2]!);
}

/** numpy searchsorted(a, v, side='left'): first i with a[i] >= v, else a.length. */
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

export function estimateBand(
  i: Float64Array,
  q: Float64Array,
  p: PreprocessParams,
): { center: number; bw: number } {
  const nfft = p.nfft;
  const psd = smoothSame(welchPsd(i, q, nfft), p.smooth);
  const floor = p.noiseFloorScale * median(psd);
  const sig = new Float64Array(nfft);
  let total = 0;
  for (let k = 0; k < nfft; k++) {
    const v = psd[k]! - floor;
    const s = v > 0 ? v : 0;
    sig[k] = s;
    total += s;
  }
  if (total < 1e-9) return { center: 0, bw: 0.95 };
  const cum = new Float64Array(nfft);
  let run = 0;
  for (let k = 0; k < nfft; k++) {
    run += sig[k]! / total;
    cum[k] = run;
  }
  let lo = searchsortedLeft(cum, p.energyEdge);
  let hi = searchsortedLeft(cum, 1 - p.energyEdge);
  lo = Math.min(lo, nfft - 1);
  hi = Math.min(Math.max(hi, lo + 1), nfft - 1);
  const fLo = lo / nfft - 0.5;
  const fHi = hi / nfft - 0.5;
  const center = 0.5 * (fLo + fHi);
  const bw = Math.max(fHi - fLo, 1 / nfft);
  return { center, bw };
}

/** Complex linear-interpolation resampler (mirror of preprocess.lin_resample). */
export function linResample(
  i: Float64Array,
  q: Float64Array,
  newLen: number,
): { i: Float64Array; q: Float64Array } {
  const n = i.length;
  if (newLen === n || n < 2) return { i, q };
  const oi = new Float64Array(newLen);
  const oq = new Float64Array(newLen);
  const step = (n - 1) / (newLen - 1);
  for (let m = 0; m < newLen; m++) {
    const pos = m * step;
    let i0 = Math.floor(pos);
    if (i0 > n - 2) i0 = n - 2;
    if (i0 < 0) i0 = 0;
    const frac = pos - i0;
    oi[m] = i[i0]! * (1 - frac) + i[i0 + 1]! * frac;
    oq[m] = q[i0]! * (1 - frac) + q[i0 + 1]! * frac;
  }
  return { i: oi, q: oq };
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
): Normalised {
  const { center, bw } = estimateBand(reIn, imIn, params);
  const n = reIn.length;
  // down-convert measured centre to DC
  const ci = new Float64Array(n);
  const cq = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const ph = -2 * Math.PI * center * k;
    const c = Math.cos(ph);
    const s = Math.sin(ph);
    ci[k] = reIn[k]! * c - imIn[k]! * s;
    cq[k] = reIn[k]! * s + imIn[k]! * c;
  }
  // resample so occupied bandwidth hits the canonical target
  const frac = Math.min(Math.max(bw, 1e-3), 0.95);
  const ratio = frac / params.targetFrac;
  const newLen = Math.max(64, Math.round(n * ratio));
  const rs = linResample(ci, cq, newLen);
  const di: Float64Array = centerFit(rs.i, params.lOut);
  const dq: Float64Array = centerFit(rs.q, params.lOut);
  // amplitude-normalise to unit RMS
  let p = 0;
  for (let k = 0; k < params.lOut; k++) p += di[k]! * di[k]! + dq[k]! * dq[k]!;
  const rms = Math.sqrt(p / params.lOut + 1e-12);
  for (let k = 0; k < params.lOut; k++) {
    di[k] = Math.fround(di[k]! / rms);
    dq[k] = Math.fround(dq[k]! / rms);
  }
  return { i: di, q: dq, center, bw };
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
