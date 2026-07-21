/**
 * Blind symbol recovery — TypeScript port of `training/recover.py`.
 *
 * Turns a raw complex I/Q smear into a distinguishable symbol constellation:
 * restore properness (blind IQ-imbalance + DC), estimate the symbol rate, resample
 * to T/2, then invert the propagation channel AND absorb timing offset with one
 * fractionally-spaced CMA equalizer. For DISPLAY we additionally lock the carrier
 * phase (a 4th-power estimate) so the plotted points sit on the constellation
 * axes; the classifier upstream does not need this since its cumulants are
 * rotation-invariant, but a human looking at the scatter does.
 *
 * Two order-agnostic quality gates come back with the symbols:
 *   - `residualIsi` — normalized symbol autocorrelation over the first few lags
 *     (residual ISI / mis-equalization; ~0 for clean iid symbols),
 *   - `snrDb` — blind in-band SNR from the PSD noise floor.
 *
 * Complex data is carried as separate real/imag `Float64Array` pairs and kept in
 * double precision throughout, matching `iq-preprocess.ts` and numpy. The `!`
 * non-null assertions satisfy the repo's `noUncheckedIndexedAccess`; every index
 * is provably in bounds by construction. This is a faithful mirror of the Python
 * so a parity fixture can guard the two against drift (`recover.test.ts`).
 */

import { welchPsd, smoothSame } from './iq-preprocess.js';

export interface RecoveredConstellation {
  /** Recovered symbol-spaced (1 sps) constellation, carrier-locked for display. */
  symbolsRe: Float64Array;
  symbolsIm: Float64Array;
  /** Estimated (or hinted) samples/symbol. */
  sps: number;
  /** Recovery-quality gate: normalized residual symbol autocorrelation (lower is better). */
  residualIsi: number;
  /** Blind in-band SNR (dB). */
  snrDb: number;
}

/** Python `round` / numpy round-half-to-even, for non-negative arguments. */
function pyRound(x: number): number {
  const f = Math.floor(x);
  const diff = x - f;
  if (diff < 0.5) return f;
  if (diff > 0.5) return f + 1;
  return f % 2 === 0 ? f : f + 1; // exactly .5 -> nearest even
}

/** numpy.hanning(n): symmetric Hann, `0.5 - 0.5 cos(2*pi*i/(n-1))` (note: n-1 denom). */
function hanningSym(n: number): Float64Array {
  const w = new Float64Array(n);
  if (n < 2) {
    if (n === 1) w[0] = 1;
    return w;
  }
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1));
  return w;
}

/** numpy.median: sort, average the two central elements for even length. */
function medianOf(x: ArrayLike<number>): number {
  const n = x.length;
  const s = Float64Array.from(x);
  s.sort();
  return n % 2 ? s[(n - 1) >> 1]! : 0.5 * (s[n / 2 - 1]! + s[n / 2]!);
}

/**
 * Blind IQ-imbalance + DC correction via properness restoration (widely-linear).
 * y = r + c*conj(r) with c = -E[r^2] / (2 E|r|^2) restores E[y^2] = 0 to first
 * order. r = x - mean(x) removes DC first.
 */
export function iqBalance(re: Float64Array, im: Float64Array): { re: Float64Array; im: Float64Array } {
  const n = re.length;
  let mr = 0;
  let mi = 0;
  for (let k = 0; k < n; k++) {
    mr += re[k]!;
    mi += im[k]!;
  }
  mr /= n;
  mi /= n;
  const rr = new Float64Array(n);
  const ri = new Float64Array(n);
  let denom = 0; // E|r|^2
  let sqR = 0; // E[r^2] real
  let sqI = 0; // E[r^2] imag
  for (let k = 0; k < n; k++) {
    const a = re[k]! - mr;
    const b = im[k]! - mi;
    rr[k] = a;
    ri[k] = b;
    denom += a * a + b * b;
    sqR += a * a - b * b;
    sqI += 2 * a * b;
  }
  denom = denom / n + 1e-12;
  const meanSqR = sqR / n;
  const meanSqI = sqI / n;
  const cR = -meanSqR / (2 * denom);
  const cI = -meanSqI / (2 * denom);
  const outR = new Float64Array(n);
  const outI = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const a = rr[k]!;
    const b = ri[k]!;
    // y = r + c*conj(r); c*conj(r) = (cR*a + cI*b) + i(cI*a - cR*b)
    outR[k] = a + (cR * a + cI * b);
    outI[k] = b + (cI * a - cR * b);
  }
  return { re: outR, im: outI };
}

/**
 * Blind symbol-rate estimate from the cyclostationary line in |x|^2. Falls back
 * to `def` when no line clears the k-sigma gate above the in-band median.
 *
 * The symbol rate is a stationary property, so we estimate it from a bounded
 * PREFIX (`maxAnalysis`) rather than the whole capture. That keeps the periodogram
 * an O(maxAnalysis^2) direct DFT instead of O(n^2): at a 56 MHz capture (65k+
 * samples) the full-length DFT is ~34 s; a 2048-sample prefix is ~25 ms and still
 * resolves the line to the exact bin for any sps in [2,16]. The cap sits above the
 * parity-fixture lengths (<=1828) so those cases compute byte-identically and the
 * integer `sps` decision is preserved exactly.
 */
export function estimateSps(
  re: Float64Array,
  im: Float64Array,
  spsLo = 2,
  spsHi = 16,
  def = 8,
  kSigma = 5.0,
  maxAnalysis = 2048,
): number {
  const n = Math.min(re.length, maxAnalysis);
  // y = |x|^2, zero-meaned, then Hann-windowed
  const y = new Float64Array(n);
  let mean = 0;
  for (let k = 0; k < n; k++) {
    const v = re[k]! * re[k]! + im[k]! * im[k]!;
    y[k] = v;
    mean += v;
  }
  mean /= n;
  const w = hanningSym(n);
  const yw = new Float64Array(n);
  for (let k = 0; k < n; k++) yw[k] = (y[k]! - mean) * w[k]!;

  const lo = 1.0 / spsHi;
  const hi = 1.0 / spsLo;
  const half = Math.floor(n / 2); // rfft covers bins 0..n//2
  const bandVals: number[] = [];
  let peak = -Infinity;
  let peakK = -1;
  for (let kb = 0; kb <= half; kb++) {
    const f = kb / n;
    if (f < lo || f > hi) continue;
    // direct rfft bin: X[kb] = sum_t yw[t] * exp(-2j*pi*kb*t/n)
    const ang0 = (-2 * Math.PI * kb) / n;
    let sr = 0;
    let si = 0;
    for (let t = 0; t < n; t++) {
      const a = ang0 * t;
      sr += yw[t]! * Math.cos(a);
      si += yw[t]! * Math.sin(a);
    }
    const sp = sr * sr + si * si;
    bandVals.push(sp);
    if (sp > peak) {
      peak = sp;
      peakK = kb;
    }
  }
  if (bandVals.length === 0) return def;
  const med = medianOf(bandVals);
  if (peak < kSigma * (med + 1e-30)) return def;
  const fsym = peakK / n;
  return fsym > 0 ? pyRound(1.0 / fsym) : def;
}

/** Linear-interpolate a complex signal from `spsIn` to `spsOut` samples/symbol. */
export function resampleTo(
  re: Float64Array,
  im: Float64Array,
  spsIn: number,
  spsOut: number,
): { re: Float64Array; im: Float64Array } {
  const n = re.length;
  const newN = pyRound((n * spsOut) / spsIn);
  if (newN < 4) return { re: Float64Array.from(re), im: Float64Array.from(im) };
  const outR = new Float64Array(newN);
  const outI = new Float64Array(newN);
  const step = (n - 1) / (newN - 1);
  for (let m = 0; m < newN; m++) {
    const pos = m * step;
    let i0 = Math.floor(pos);
    if (i0 > n - 2) i0 = n - 2;
    if (i0 < 0) i0 = 0;
    const frac = pos - i0;
    outR[m] = re[i0]! * (1 - frac) + re[i0 + 1]! * frac;
    outI[m] = im[i0]! * (1 - frac) + im[i0 + 1]! * frac;
  }
  return { re: outR, im: outI };
}

/**
 * T/2 fractionally-spaced CMA equalizer. Input is 2 samples/symbol; returns the
 * symbol-spaced (1 sps) equalized output. Blind (no constellation/channel
 * knowledge); the FSE absorbs timing offset so there is no separate timing loop.
 * Deterministic: center-tap-delta init, no randomness.
 */
export function cmaFse(
  re2: Float64Array,
  im2: Float64Array,
  ntaps = 21,
  mu = 2e-3,
  passes = 25,
  R2 = 1.3,
): { re: Float64Array; im: Float64Array } {
  const n = re2.length;
  // normalize to unit power: x2 /= sqrt(mean(|x2|^2)) + 1e-12
  let pw = 0;
  for (let k = 0; k < n; k++) pw += re2[k]! * re2[k]! + im2[k]! * im2[k]!;
  const norm = Math.sqrt(pw / n) + 1e-12;
  const xr = new Float64Array(n);
  const xi = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    xr[k] = re2[k]! / norm;
    xi[k] = im2[k]! / norm;
  }
  const nsym = Math.floor((n - ntaps) / 2);
  if (nsym < 8) {
    // degenerate: decimate the normalized signal by 2 (Python x2[::2])
    const m = Math.floor((n + 1) / 2);
    const dr = new Float64Array(m);
    const di = new Float64Array(m);
    for (let k = 0; k < m; k++) {
      dr[k] = xr[2 * k]!;
      di[k] = xi[2 * k]!;
    }
    return { re: dr, im: di };
  }
  const wr = new Float64Array(ntaps);
  const wi = new Float64Array(ntaps);
  wr[ntaps >> 1] = 1.0; // center-tap delta
  for (let pass = 0; pass < passes; pass++) {
    for (let k = 0; k < nsym; k++) {
      const base = 2 * k;
      // yk = dot(w, u[::-1]); u[::-1][i] = x2[base + ntaps-1-i]
      let ykr = 0;
      let yki = 0;
      for (let i = 0; i < ntaps; i++) {
        const ur = xr[base + ntaps - 1 - i]!;
        const ui = xi[base + ntaps - 1 - i]!;
        const a = wr[i]!;
        const b = wi[i]!;
        ykr += a * ur - b * ui;
        yki += a * ui + b * ur;
      }
      const yk2 = ykr * ykr + yki * yki;
      const s = mu * (R2 - yk2);
      const gr = s * ykr;
      const gi = s * yki;
      // w += g * conj(u[::-1]); g*conj(u) = (gr*ur+gi*ui) + i(gi*ur-gr*ui)
      for (let i = 0; i < ntaps; i++) {
        const ur = xr[base + ntaps - 1 - i]!;
        const ui = xi[base + ntaps - 1 - i]!;
        wr[i] = wr[i]! + (gr * ur + gi * ui);
        wi[i] = wi[i]! + (gi * ur - gr * ui);
      }
    }
    for (let i = 0; i < ntaps; i++) {
      wr[i] = wr[i]! * 0.9999; // light tap leakage
      wi[i] = wi[i]! * 0.9999;
    }
  }
  const yr = new Float64Array(nsym);
  const yi = new Float64Array(nsym);
  for (let k = 0; k < nsym; k++) {
    const base = 2 * k;
    let ykr = 0;
    let yki = 0;
    for (let i = 0; i < ntaps; i++) {
      const ur = xr[base + ntaps - 1 - i]!;
      const ui = xi[base + ntaps - 1 - i]!;
      const a = wr[i]!;
      const b = wi[i]!;
      ykr += a * ur - b * ui;
      yki += a * ui + b * ur;
    }
    yr[k] = ykr;
    yi[k] = yki;
  }
  return { re: yr, im: yi };
}

/** Remove carrier phase (mod 90 deg) via a 4th-power estimate — locks the display. */
export function carrier4th(re: Float64Array, im: Float64Array): { re: Float64Array; im: Float64Array } {
  const n = re.length;
  let m4r = 0;
  let m4i = 0;
  for (let k = 0; k < n; k++) {
    const a = re[k]!;
    const b = im[k]!;
    const z2r = a * a - b * b;
    const z2i = 2 * a * b;
    const z4r = z2r * z2r - z2i * z2i;
    const z4i = 2 * z2r * z2i;
    m4r += z4r;
    m4i += z4i;
  }
  const phi = Math.atan2(m4i, m4r) / 4.0;
  const c = Math.cos(phi);
  const s = Math.sin(phi);
  const outR = new Float64Array(n);
  const outI = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const a = re[k]!;
    const b = im[k]!;
    // y * exp(-i phi) = (a cos + b sin) + i(b cos - a sin)
    outR[k] = a * c + b * s;
    outI[k] = b * c - a * s;
  }
  return { re: outR, im: outI };
}

/**
 * Order-agnostic recovery-quality proxy: normalized symbol autocorrelation over
 * the first `maxLag` lags. Rotation-invariant; ~0 for well-equalized iid symbols.
 */
export function residualIsi(re: Float64Array, im: Float64Array, maxLag = 4): number {
  const n = re.length;
  let mr = 0;
  let mi = 0;
  for (let k = 0; k < n; k++) {
    mr += re[k]!;
    mi += im[k]!;
  }
  mr /= n;
  mi /= n;
  const zr = new Float64Array(n);
  const zi = new Float64Array(n);
  let r0 = 0;
  for (let k = 0; k < n; k++) {
    const a = re[k]! - mr;
    const b = im[k]! - mi;
    zr[k] = a;
    zi[k] = b;
    r0 += a * a + b * b;
  }
  r0 = r0 / n + 1e-12;
  let acc = 0;
  for (let lag = 1; lag <= maxLag; lag++) {
    if (lag >= n) break;
    // mean over k=lag..n-1 of z[k] * conj(z[k-lag])
    let sr = 0;
    let si = 0;
    for (let k = lag; k < n; k++) {
      const a = zr[k]!;
      const b = zi[k]!;
      const c = zr[k - lag]!;
      const d = zi[k - lag]!;
      // z[k] * conj(z[k-lag]) = (a c + b d) + i(b c - a d)
      sr += a * c + b * d;
      si += b * c - a * d;
    }
    const cnt = n - lag;
    sr /= cnt;
    si /= cnt;
    acc += sr * sr + si * si; // |mean|^2
  }
  return Math.sqrt(acc) / r0;
}

/**
 * Blind in-band SNR estimate (dB) from the PSD noise floor. Noise is counted over
 * the occupied bins only, so this is a true in-band SNR. Reuses the shared
 * `welchPsd` / `smoothSame` kernels (nfft=512, SMOOTH=5, NOISE_FLOOR_SCALE=1.44).
 */
export function estimateSnr(re: Float64Array, im: Float64Array, nfft = 512): number {
  const NOISE_FLOOR_SCALE = 1.44;
  const SMOOTH = 5;
  const psd = smoothSame(welchPsd(re, im, nfft), SMOOTH);
  const floor = NOISE_FLOOR_SCALE * medianOf(psd);
  let nIn = 0;
  let sig = 0;
  for (let k = 0; k < psd.length; k++) {
    const d = psd[k]! - floor;
    if (d > 0) {
      sig += d;
      nIn++;
    }
  }
  const noise = floor * Math.max(nIn, 1);
  if (noise <= 0 || sig <= 0) return -10.0;
  return 10.0 * Math.log10(sig / noise);
}

/**
 * Full blind recovery. Mirrors `recover.py`'s pipeline: iq-balance -> sps estimate
 * -> resample to T/2 -> CMA FSE -> carrier-lock for display. `residualIsi` is
 * computed on the pre-carrier symbols (rotation-invariant, so identical either
 * way) and `snrDb` on the original I/Q, matching the Python reference.
 */
export function recoverConstellation(
  re: Float64Array,
  im: Float64Array,
  spsHint?: number,
): RecoveredConstellation {
  const bal = iqBalance(re, im);
  const sps = spsHint && spsHint > 0 ? spsHint : estimateSps(bal.re, bal.im);
  const x2 = resampleTo(bal.re, bal.im, sps, 2.0);
  const sym = cmaFse(x2.re, x2.im);
  const isi = residualIsi(sym.re, sym.im);
  const snrDb = estimateSnr(re, im);
  const locked = carrier4th(sym.re, sym.im);
  return {
    symbolsRe: locked.re,
    symbolsIm: locked.im,
    sps,
    residualIsi: isi,
    snrDb,
  };
}
