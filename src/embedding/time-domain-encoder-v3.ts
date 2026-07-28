/**
 * Pure-TypeScript forward pass for the v3 time-domain invariant patch CNN.
 *
 * Ports `training/zplane_ab/v2_full_variation/invariant_patch_cnn.py`
 * (InvariantPatchCNN) exactly as the v3 fusion bundle uses it: one shared
 * patch encoder applied to every fixed-length patch, followed by
 * permutation-invariant mean/std set pooling, scalar-feature concatenation,
 * and a two-layer head with L2 normalization. Both the `real` conv stack
 * (Conv1d stride 2 + evaluation-mode BatchNorm + ReLU) and the paired-real
 * `complex` stack (complex conv, magnitude RMS norm, modReLU,
 * lowpass-decimate, magnitude/autocorrelation statistics) are implemented.
 *
 * No FFT, Welch, ONNX, WASM, or complex-number runtime is used anywhere.
 * BatchNorm arrives UNFOLDED (weight/bias/running stats/eps) and is applied
 * with torch's evaluation arithmetic, so the only numeric difference from the
 * Python reference is float64 accumulation versus torch's float32 kernels;
 * the parity suite measures that difference against the bundle's
 * Python-generated probe fixture.
 *
 * Weights come from the staging asset written by
 * `training/zplane_ab/v2_full_variation/v3_scale/export_v3_browser_weights.py`.
 */

export const TIME_DOMAIN_ENCODER_V3_VERSION = 'time-domain-invariant-patch-cnn-v3' as const;

export interface TimeDomainPatchConfigV3 {
  patch_length: number;
  patch_count: number;
  encoder: 'real' | 'complex';
  patch_dim: number;
  hidden: number;
  embed_dim: number;
  n_features: number;
  set_pool: 'mean' | 'mean_std';
  dropout: number;
}

export interface TimeDomainLinearV3 {
  in: number;
  out: number;
  weight: number[];
  bias: number[];
}

export interface TimeDomainBatchNormV3 {
  eps: number;
  weight: number[];
  bias: number[];
  running_mean: number[];
  running_var: number[];
}

export interface TimeDomainRealBlockV3 {
  in_channels: number;
  out_channels: number;
  kernel: number;
  stride: number;
  padding: number;
  conv_weight: number[];
  batch_norm: TimeDomainBatchNormV3;
}

export interface TimeDomainComplexStageV3 {
  in_channels: number;
  out_channels: number;
  kernel: number;
  padding: number;
  weight_real: number[];
  weight_imag: number[];
  threshold_raw: number[];
}

export interface TimeDomainRealBranchV3 {
  config: TimeDomainPatchConfigV3 & { encoder: 'real' };
  blocks: TimeDomainRealBlockV3[];
  patch_projection: TimeDomainLinearV3;
  fc1: TimeDomainLinearV3;
  fc2: TimeDomainLinearV3;
}

export interface TimeDomainComplexBranchV3 {
  config: TimeDomainPatchConfigV3 & { encoder: 'complex' };
  stages: TimeDomainComplexStageV3[];
  mag_norm_eps: number;
  modrelu_magnitude_eps: number;
  lowpass: [number, number, number];
  lags: number[];
  patch_projection: TimeDomainLinearV3;
  fc1: TimeDomainLinearV3;
  fc2: TimeDomainLinearV3;
}

/** `F.normalize` default epsilon: `x / max(||x||, eps)`. */
export const TIME_DOMAIN_EMBEDDING_NORMALIZE_EPS = 1e-12;

type UnknownRecord = Record<string, unknown>;

function record(value: unknown, path: string): UnknownRecord {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new TypeError(`${path} must be an object`);
  }
  return value as UnknownRecord;
}

function finite(value: unknown, path: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`${path} must be a finite number`);
  }
  return value;
}

function integer(value: unknown, path: string, minimum = 0): number {
  const result = finite(value, path);
  if (!Number.isInteger(result) || result < minimum) {
    throw new RangeError(`${path} must be an integer >= ${minimum}`);
  }
  return result;
}

function finiteVector(value: unknown, path: string, length: number): number[] {
  if (!Array.isArray(value)) throw new TypeError(`${path} must be an array`);
  if (value.length !== length) {
    throw new RangeError(`${path} must contain ${length} values`);
  }
  for (let index = 0; index < value.length; index++) {
    finite(value[index], `${path}[${index}]`);
  }
  return value as number[];
}

function validateConfig(
  value: unknown,
  path: string,
  expectedEncoder: 'real' | 'complex',
): TimeDomainPatchConfigV3 {
  const cfg = record(value, path);
  if (cfg.encoder !== expectedEncoder) {
    throw new RangeError(`${path}.encoder must be ${expectedEncoder}`);
  }
  if (cfg.set_pool !== 'mean' && cfg.set_pool !== 'mean_std') {
    throw new RangeError(`${path}.set_pool must be mean or mean_std`);
  }
  const dropout = finite(cfg.dropout, `${path}.dropout`);
  if (dropout < 0 || dropout >= 1) {
    throw new RangeError(`${path}.dropout must lie in [0,1)`);
  }
  return {
    patch_length: integer(cfg.patch_length, `${path}.patch_length`, 16),
    patch_count: integer(cfg.patch_count, `${path}.patch_count`, 1),
    encoder: expectedEncoder,
    patch_dim: integer(cfg.patch_dim, `${path}.patch_dim`, 1),
    hidden: integer(cfg.hidden, `${path}.hidden`, 1),
    embed_dim: integer(cfg.embed_dim, `${path}.embed_dim`, 1),
    n_features: integer(cfg.n_features, `${path}.n_features`),
    set_pool: cfg.set_pool,
    dropout,
  };
}

function validateLinear(
  value: unknown,
  path: string,
  expectedIn: number,
  expectedOut: number,
): TimeDomainLinearV3 {
  const layer = record(value, path);
  const input = integer(layer.in, `${path}.in`, 1);
  const output = integer(layer.out, `${path}.out`, 1);
  if (input !== expectedIn || output !== expectedOut) {
    throw new RangeError(
      `${path} shape must be [${expectedOut},${expectedIn}], got [${output},${input}]`,
    );
  }
  return {
    in: input,
    out: output,
    weight: finiteVector(layer.weight, `${path}.weight`, input * output),
    bias: finiteVector(layer.bias, `${path}.bias`, output),
  };
}

/** Validate an untrusted real-branch record against the exporter schema. */
export function validateTimeDomainRealBranchV3(
  value: unknown,
  path = 'asset.real',
): TimeDomainRealBranchV3 {
  const branch = record(value, path);
  const config = validateConfig(
    branch.config,
    `${path}.config`,
    'real',
  ) as TimeDomainRealBranchV3['config'];
  if (!Array.isArray(branch.blocks) || branch.blocks.length === 0) {
    throw new TypeError(`${path}.blocks must be a non-empty array`);
  }
  let channels = 2;
  const blocks: TimeDomainRealBlockV3[] = [];
  for (let index = 0; index < branch.blocks.length; index++) {
    const blockPath = `${path}.blocks[${index}]`;
    const raw = record(branch.blocks[index], blockPath);
    const input = integer(raw.in_channels, `${blockPath}.in_channels`, 1);
    const output = integer(raw.out_channels, `${blockPath}.out_channels`, 1);
    const kernel = integer(raw.kernel, `${blockPath}.kernel`, 1);
    if (input !== channels) {
      throw new RangeError(`${blockPath}.in_channels breaks the channel chain`);
    }
    const norm = record(raw.batch_norm, `${blockPath}.batch_norm`);
    const eps = finite(norm.eps, `${blockPath}.batch_norm.eps`);
    if (eps <= 0) {
      throw new RangeError(`${blockPath}.batch_norm.eps must be positive`);
    }
    const runningVar = finiteVector(
      norm.running_var,
      `${blockPath}.batch_norm.running_var`,
      output,
    );
    for (let channel = 0; channel < output; channel++) {
      if (runningVar[channel]! < 0) {
        throw new RangeError(
          `${blockPath}.batch_norm.running_var[${channel}] cannot be negative`,
        );
      }
    }
    blocks.push({
      in_channels: input,
      out_channels: output,
      kernel,
      stride: integer(raw.stride, `${blockPath}.stride`, 1),
      padding: integer(raw.padding, `${blockPath}.padding`),
      conv_weight: finiteVector(
        raw.conv_weight,
        `${blockPath}.conv_weight`,
        output * input * kernel,
      ),
      batch_norm: {
        eps,
        weight: finiteVector(norm.weight, `${blockPath}.batch_norm.weight`, output),
        bias: finiteVector(norm.bias, `${blockPath}.batch_norm.bias`, output),
        running_mean: finiteVector(
          norm.running_mean,
          `${blockPath}.batch_norm.running_mean`,
          output,
        ),
        running_var: runningVar,
      },
    });
    channels = output;
  }
  const patchProjection = validateLinear(
    branch.patch_projection,
    `${path}.patch_projection`,
    2 * channels,
    config.patch_dim,
  );
  const setWidth = config.patch_dim * (config.set_pool === 'mean_std' ? 2 : 1);
  return {
    config,
    blocks,
    patch_projection: patchProjection,
    fc1: validateLinear(
      branch.fc1,
      `${path}.fc1`,
      setWidth + config.n_features,
      config.hidden,
    ),
    fc2: validateLinear(branch.fc2, `${path}.fc2`, config.hidden, config.embed_dim),
  };
}

/** Validate an untrusted complex-branch record against the exporter schema. */
export function validateTimeDomainComplexBranchV3(
  value: unknown,
  path = 'asset.complex',
): TimeDomainComplexBranchV3 {
  const branch = record(value, path);
  const config = validateConfig(
    branch.config,
    `${path}.config`,
    'complex',
  ) as TimeDomainComplexBranchV3['config'];
  if (!Array.isArray(branch.stages) || branch.stages.length === 0) {
    throw new TypeError(`${path}.stages must be a non-empty array`);
  }
  let channels = 1;
  const stages: TimeDomainComplexStageV3[] = [];
  for (let index = 0; index < branch.stages.length; index++) {
    const stagePath = `${path}.stages[${index}]`;
    const raw = record(branch.stages[index], stagePath);
    const input = integer(raw.in_channels, `${stagePath}.in_channels`, 1);
    const output = integer(raw.out_channels, `${stagePath}.out_channels`, 1);
    const kernel = integer(raw.kernel, `${stagePath}.kernel`, 1);
    if (input !== channels) {
      throw new RangeError(`${stagePath}.in_channels breaks the channel chain`);
    }
    stages.push({
      in_channels: input,
      out_channels: output,
      kernel,
      padding: integer(raw.padding, `${stagePath}.padding`),
      weight_real: finiteVector(
        raw.weight_real,
        `${stagePath}.weight_real`,
        output * input * kernel,
      ),
      weight_imag: finiteVector(
        raw.weight_imag,
        `${stagePath}.weight_imag`,
        output * input * kernel,
      ),
      threshold_raw: finiteVector(
        raw.threshold_raw,
        `${stagePath}.threshold_raw`,
        output,
      ),
    });
    channels = output;
  }
  const magNormEps = finite(branch.mag_norm_eps, `${path}.mag_norm_eps`);
  const magnitudeEps = finite(
    branch.modrelu_magnitude_eps,
    `${path}.modrelu_magnitude_eps`,
  );
  if (magNormEps <= 0 || magnitudeEps <= 0) {
    throw new RangeError(`${path} normalization epsilons must be positive`);
  }
  const lowpass = finiteVector(branch.lowpass, `${path}.lowpass`, 3);
  if (lowpass[0] !== 0.25 || lowpass[1] !== 0.5 || lowpass[2] !== 0.25) {
    throw new RangeError(`${path}.lowpass must be [0.25,0.5,0.25]`);
  }
  if (!Array.isArray(branch.lags) || branch.lags.length === 0) {
    throw new TypeError(`${path}.lags must be a non-empty array`);
  }
  const lags = branch.lags.map((lag, index) =>
    integer(lag, `${path}.lags[${index}]`, 1));
  for (let index = 1; index < lags.length; index++) {
    if (lags[index]! <= lags[index - 1]!) {
      throw new RangeError(`${path}.lags must be strictly increasing`);
    }
  }
  const statisticWidth = channels * (2 + 2 * lags.length);
  const patchProjection = validateLinear(
    branch.patch_projection,
    `${path}.patch_projection`,
    statisticWidth,
    config.patch_dim,
  );
  const setWidth = config.patch_dim * (config.set_pool === 'mean_std' ? 2 : 1);
  return {
    config,
    stages,
    mag_norm_eps: magNormEps,
    modrelu_magnitude_eps: magnitudeEps,
    lowpass: [lowpass[0]!, lowpass[1]!, lowpass[2]!],
    lags,
    patch_projection: patchProjection,
    fc1: validateLinear(
      branch.fc1,
      `${path}.fc1`,
      setWidth + config.n_features,
      config.hidden,
    ),
    fc2: validateLinear(branch.fc2, `${path}.fc2`, config.hidden, config.embed_dim),
  };
}

// ---------------------------------------------------------------------------
// forward math
// ---------------------------------------------------------------------------

function linear(
  input: Float64Array,
  layer: TimeDomainLinearV3,
  relu: boolean,
): Float64Array {
  const output = new Float64Array(layer.out);
  for (let outputIndex = 0; outputIndex < layer.out; outputIndex++) {
    let sum = layer.bias[outputIndex]!;
    const base = outputIndex * layer.in;
    for (let inputIndex = 0; inputIndex < layer.in; inputIndex++) {
      sum += input[inputIndex]! * layer.weight[base + inputIndex]!;
    }
    output[outputIndex] = relu && sum < 0 ? 0 : sum;
  }
  return output;
}

/**
 * `_RealBlock`: zero-padded strided Conv1d (no bias), then evaluation-mode
 * BatchNorm `(x - running_mean) / sqrt(running_var + eps) * weight + bias`,
 * then ReLU. BatchNorm is applied explicitly rather than folded into the
 * convolution so the arithmetic matches torch's evaluation path.
 */
function realBlock(
  input: Float64Array[],
  layer: TimeDomainRealBlockV3,
): Float64Array[] {
  const inputLength = input[0]!.length;
  const outputLength = Math.floor(
    (inputLength + 2 * layer.padding - layer.kernel) / layer.stride,
  ) + 1;
  const output: Float64Array[] = [];
  for (let outputChannel = 0; outputChannel < layer.out_channels; outputChannel++) {
    const norm = layer.batch_norm;
    const inverseStd = 1 / Math.sqrt(norm.running_var[outputChannel]! + norm.eps);
    const gamma = norm.weight[outputChannel]!;
    const beta = norm.bias[outputChannel]!;
    const mean = norm.running_mean[outputChannel]!;
    const values = new Float64Array(outputLength);
    const outputBase = outputChannel * layer.in_channels * layer.kernel;
    for (let time = 0; time < outputLength; time++) {
      let sum = 0;
      const start = time * layer.stride - layer.padding;
      for (let inputChannel = 0; inputChannel < layer.in_channels; inputChannel++) {
        const channel = input[inputChannel]!;
        const weightBase = outputBase + inputChannel * layer.kernel;
        for (let tap = 0; tap < layer.kernel; tap++) {
          const source = start + tap;
          if (source >= 0 && source < inputLength) {
            sum += channel[source]! * layer.conv_weight[weightBase + tap]!;
          }
        }
      }
      const normalized = (sum - mean) * inverseStd * gamma + beta;
      values[time] = normalized > 0 ? normalized : 0;
    }
    output.push(values);
  }
  return output;
}

/** `y.mean(dim=-1)` then `y.std(dim=-1, unbiased=False)`, channel-major. */
function channelMeanStd(channels: Float64Array[]): Float64Array {
  const width = channels.length;
  const length = channels[0]!.length;
  const output = new Float64Array(2 * width);
  for (let channelIndex = 0; channelIndex < width; channelIndex++) {
    const channel = channels[channelIndex]!;
    let mean = 0;
    for (let time = 0; time < length; time++) mean += channel[time]!;
    mean /= length;
    let variance = 0;
    for (let time = 0; time < length; time++) {
      const delta = channel[time]! - mean;
      variance += delta * delta;
    }
    output[channelIndex] = mean;
    output[width + channelIndex] = Math.sqrt(variance / length);
  }
  return output;
}

interface PairedChannels {
  real: Float64Array[];
  imaginary: Float64Array[];
}

/**
 * `ComplexConv1d`: zero 'same' padding, no bias,
 * `(Wr+iWi)*(Xr+iXi) = (Wr*Xr - Wi*Xi) + i(Wr*Xi + Wi*Xr)`.
 */
function pairedComplexConv(
  input: PairedChannels,
  layer: TimeDomainComplexStageV3,
): PairedChannels {
  const inputLength = input.real[0]!.length;
  const outputLength = inputLength + 2 * layer.padding - layer.kernel + 1;
  const outputReal: Float64Array[] = [];
  const outputImaginary: Float64Array[] = [];
  for (let outputChannel = 0; outputChannel < layer.out_channels; outputChannel++) {
    const realValues = new Float64Array(outputLength);
    const imaginaryValues = new Float64Array(outputLength);
    const outputBase = outputChannel * layer.in_channels * layer.kernel;
    for (let time = 0; time < outputLength; time++) {
      let wrReal = 0;
      let wiImaginary = 0;
      let wrImaginary = 0;
      let wiReal = 0;
      const start = time - layer.padding;
      for (let inputChannel = 0; inputChannel < layer.in_channels; inputChannel++) {
        const real = input.real[inputChannel]!;
        const imaginary = input.imaginary[inputChannel]!;
        const weightBase = outputBase + inputChannel * layer.kernel;
        for (let tap = 0; tap < layer.kernel; tap++) {
          const source = start + tap;
          if (source >= 0 && source < inputLength) {
            const weightReal = layer.weight_real[weightBase + tap]!;
            const weightImaginary = layer.weight_imag[weightBase + tap]!;
            wrReal += weightReal * real[source]!;
            wiImaginary += weightImaginary * imaginary[source]!;
            wrImaginary += weightReal * imaginary[source]!;
            wiReal += weightImaginary * real[source]!;
          }
        }
      }
      realValues[time] = wrReal - wiImaginary;
      imaginaryValues[time] = wrImaginary + wiReal;
    }
    outputReal.push(realValues);
    outputImaginary.push(imaginaryValues);
  }
  return { real: outputReal, imaginary: outputImaginary };
}

function softplus(value: number): number {
  if (value > 20) return value;
  if (value < -20) return Math.exp(value);
  return Math.log1p(Math.exp(value));
}

/**
 * `ComplexMagNorm` then `ComplexModReLU`, per channel:
 * `rms = sqrt(mean_t |z|^2 + mag_norm_eps)`, `z /= rms`,
 * `b = -softplus(thresh_raw)`, `mag = sqrt(re^2 + im^2 + modrelu_eps)`,
 * `z *= clamp(mag + b, 0) / mag`.
 */
function magNormModRelu(
  input: PairedChannels,
  layer: TimeDomainComplexStageV3,
  magNormEps: number,
  magnitudeEps: number,
): PairedChannels {
  const outputReal: Float64Array[] = [];
  const outputImaginary: Float64Array[] = [];
  const length = input.real[0]!.length;
  for (let channelIndex = 0; channelIndex < input.real.length; channelIndex++) {
    const real = input.real[channelIndex]!;
    const imaginary = input.imaginary[channelIndex]!;
    let power = 0;
    for (let time = 0; time < length; time++) {
      power += real[time]! * real[time]! + imaginary[time]! * imaginary[time]!;
    }
    const rms = Math.sqrt(power / length + magNormEps);
    const bias = -softplus(layer.threshold_raw[channelIndex]!);
    const realValues = new Float64Array(length);
    const imaginaryValues = new Float64Array(length);
    for (let time = 0; time < length; time++) {
      const normalizedReal = real[time]! / rms;
      const normalizedImaginary = imaginary[time]! / rms;
      const magnitude = Math.sqrt(
        normalizedReal * normalizedReal
        + normalizedImaginary * normalizedImaginary
        + magnitudeEps,
      );
      const scale = Math.max(magnitude + bias, 0) / magnitude;
      realValues[time] = scale * normalizedReal;
      imaginaryValues[time] = scale * normalizedImaginary;
    }
    outputReal.push(realValues);
    outputImaginary.push(imaginaryValues);
  }
  return { real: outputReal, imaginary: outputImaginary };
}

/**
 * `_complex_lowpass_decimate`: replicate-padded [0.25, 0.5, 0.25] filter,
 * then keep even indices.
 */
function lowpassDecimate(channels: Float64Array[]): Float64Array[] {
  return channels.map((channel) => {
    const length = channel.length;
    const output = new Float64Array(Math.ceil(length / 2));
    for (let outputTime = 0; outputTime < output.length; outputTime++) {
      const time = 2 * outputTime;
      output[outputTime] = (
        0.25 * channel[Math.max(0, time - 1)]!
        + 0.5 * channel[time]!
        + 0.25 * channel[Math.min(length - 1, time + 1)]!
      );
    }
    return output;
  });
}

/**
 * `_complex_patch_statistics`: per channel `|z|` mean and population std,
 * then for each lag the mean of `z[t] * conj(z[t + lag])`, laid out
 * channel-major per statistic group exactly as `torch.cat(..., dim=-1)`.
 */
function complexStatistics(input: PairedChannels, lags: number[]): Float64Array {
  const channels = input.real.length;
  const length = input.real[0]!.length;
  const groups = 2 + 2 * lags.length;
  const output = new Float64Array(channels * groups);
  for (let channelIndex = 0; channelIndex < channels; channelIndex++) {
    const real = input.real[channelIndex]!;
    const imaginary = input.imaginary[channelIndex]!;
    let meanMagnitude = 0;
    const magnitudes = new Float64Array(length);
    for (let time = 0; time < length; time++) {
      const magnitude = Math.sqrt(
        real[time]! * real[time]! + imaginary[time]! * imaginary[time]!,
      );
      magnitudes[time] = magnitude;
      meanMagnitude += magnitude;
    }
    meanMagnitude /= length;
    let variance = 0;
    for (let time = 0; time < length; time++) {
      const delta = magnitudes[time]! - meanMagnitude;
      variance += delta * delta;
    }
    output[channelIndex] = meanMagnitude;
    output[channels + channelIndex] = Math.sqrt(variance / length);
    for (let lagIndex = 0; lagIndex < lags.length; lagIndex++) {
      const lag = lags[lagIndex]!;
      let correlationReal = 0;
      let correlationImaginary = 0;
      if (lag < length) {
        const count = length - lag;
        for (let time = 0; time < count; time++) {
          correlationReal += (
            real[time]! * real[time + lag]!
            + imaginary[time]! * imaginary[time + lag]!
          );
          correlationImaginary += (
            imaginary[time]! * real[time + lag]!
            - real[time]! * imaginary[time + lag]!
          );
        }
        correlationReal /= count;
        correlationImaginary /= count;
      }
      const group = 2 + 2 * lagIndex;
      output[group * channels + channelIndex] = correlationReal;
      output[(group + 1) * channels + channelIndex] = correlationImaginary;
    }
  }
  return output;
}

function unitNormalize(
  value: Float64Array,
  eps = TIME_DOMAIN_EMBEDDING_NORMALIZE_EPS,
): Float64Array {
  let normSquared = 0;
  for (const item of value) normSquared += item * item;
  const denominator = Math.max(Math.sqrt(normSquared), eps);
  const output = new Float64Array(value.length);
  for (let index = 0; index < value.length; index++) {
    output[index] = value[index]! / denominator;
  }
  return output;
}

/** Mean over patches, plus population std over patches for `mean_std`. */
function setPool(
  patches: Float64Array[],
  mode: 'mean' | 'mean_std',
): Float64Array {
  const patchCount = patches.length;
  const width = patches[0]!.length;
  const output = new Float64Array(width * (mode === 'mean_std' ? 2 : 1));
  for (let dimension = 0; dimension < width; dimension++) {
    let mean = 0;
    for (const patch of patches) mean += patch[dimension]!;
    mean /= patchCount;
    output[dimension] = mean;
    if (mode === 'mean_std') {
      let variance = 0;
      for (const patch of patches) {
        const delta = patch[dimension]! - mean;
        variance += delta * delta;
      }
      output[width + dimension] = Math.sqrt(variance / patchCount);
    }
  }
  return output;
}

function patchInput(
  inPhase: ArrayLike<number>,
  quadrature: ArrayLike<number>,
  patchIndex: number,
  patchLength: number,
): [Float64Array, Float64Array] {
  const real = new Float64Array(patchLength);
  const imaginary = new Float64Array(patchLength);
  const offset = patchIndex * patchLength;
  for (let index = 0; index < patchLength; index++) {
    real[index] = inPhase[offset + index]!;
    imaginary[index] = quadrature[offset + index]!;
  }
  return [real, imaginary];
}

function assertForwardInputs(
  config: TimeDomainPatchConfigV3,
  inPhase: ArrayLike<number>,
  quadrature: ArrayLike<number>,
  standardizedFeatures: ArrayLike<number>,
): void {
  const packedLength = config.patch_length * config.patch_count;
  if (inPhase.length !== packedLength || quadrature.length !== packedLength) {
    throw new RangeError(
      `packed I/Q must have length ${packedLength}, got `
      + `${inPhase.length}/${quadrature.length}`,
    );
  }
  if (standardizedFeatures.length !== config.n_features) {
    throw new RangeError(
      `standardized features must have length ${config.n_features}, got `
      + `${standardizedFeatures.length}`,
    );
  }
  for (let index = 0; index < packedLength; index++) {
    if (!Number.isFinite(inPhase[index]) || !Number.isFinite(quadrature[index])) {
      throw new RangeError(`packed I/Q sample ${index} must be finite`);
    }
  }
  for (let index = 0; index < config.n_features; index++) {
    if (!Number.isFinite(standardizedFeatures[index])) {
      throw new RangeError(`standardized feature ${index} must be finite`);
    }
  }
}

function branchTail(
  patches: Float64Array[],
  features: ArrayLike<number>,
  config: TimeDomainPatchConfigV3,
  fc1: TimeDomainLinearV3,
  fc2: TimeDomainLinearV3,
): Float64Array {
  const pooled = setPool(patches, config.set_pool);
  const combined = new Float64Array(pooled.length + features.length);
  combined.set(pooled);
  for (let index = 0; index < features.length; index++) {
    combined[pooled.length + index] = features[index]!;
  }
  return unitNormalize(linear(linear(combined, fc1, true), fc2, false));
}

/**
 * Forward one packed capture through the real branch. Dropout is identity in
 * evaluation mode and is therefore omitted.
 */
export function forwardTimeDomainRealBranchV3(
  branch: TimeDomainRealBranchV3,
  inPhase: ArrayLike<number>,
  quadrature: ArrayLike<number>,
  standardizedFeatures: ArrayLike<number>,
): Float64Array {
  assertForwardInputs(branch.config, inPhase, quadrature, standardizedFeatures);
  const patches: Float64Array[] = [];
  for (let patchIndex = 0; patchIndex < branch.config.patch_count; patchIndex++) {
    const [real, imaginary] = patchInput(
      inPhase,
      quadrature,
      patchIndex,
      branch.config.patch_length,
    );
    let channels = [real, imaginary];
    for (const block of branch.blocks) channels = realBlock(channels, block);
    patches.push(linear(channelMeanStd(channels), branch.patch_projection, true));
  }
  return branchTail(
    patches,
    standardizedFeatures,
    branch.config,
    branch.fc1,
    branch.fc2,
  );
}

/** Forward one packed capture through the paired-real complex branch. */
export function forwardTimeDomainComplexBranchV3(
  branch: TimeDomainComplexBranchV3,
  inPhase: ArrayLike<number>,
  quadrature: ArrayLike<number>,
  standardizedFeatures: ArrayLike<number>,
): Float64Array {
  assertForwardInputs(branch.config, inPhase, quadrature, standardizedFeatures);
  const patches: Float64Array[] = [];
  for (let patchIndex = 0; patchIndex < branch.config.patch_count; patchIndex++) {
    const [real, imaginary] = patchInput(
      inPhase,
      quadrature,
      patchIndex,
      branch.config.patch_length,
    );
    let channels: PairedChannels = { real: [real], imaginary: [imaginary] };
    for (const stage of branch.stages) {
      channels = pairedComplexConv(channels, stage);
      channels = magNormModRelu(
        channels,
        stage,
        branch.mag_norm_eps,
        branch.modrelu_magnitude_eps,
      );
      channels = {
        real: lowpassDecimate(channels.real),
        imaginary: lowpassDecimate(channels.imaginary),
      };
    }
    patches.push(
      linear(complexStatistics(channels, branch.lags), branch.patch_projection, true),
    );
  }
  return branchTail(
    patches,
    standardizedFeatures,
    branch.config,
    branch.fc1,
    branch.fc2,
  );
}

/**
 * Factory contract for the sibling decision layer
 * (`time-domain-classifier-v3.ts` dynamic-imports this module and looks up
 * `createTimeDomainEncodersV3` by name). Accepts the staging weights asset --
 * either the already-validated shape produced by
 * `loadTimeDomainFusionAssetV3` (fusion module) or the raw parsed JSON, which
 * is validated here through the same branch validators the parity suite uses,
 * so an invalid asset fails loudly instead of producing wrong embeddings.
 *
 * Returned closures match `TimeDomainBranchEncoderV3` in the decision layer:
 * (packedInPhase, packedQuadrature, standardizedFeatures) => Float64Array.
 */
export function createTimeDomainEncodersV3(encoderAsset: unknown): {
  real: (
    packedInPhase: ArrayLike<number>,
    packedQuadrature: ArrayLike<number>,
    standardizedFeatures: ArrayLike<number>,
  ) => Float64Array;
  complex: (
    packedInPhase: ArrayLike<number>,
    packedQuadrature: ArrayLike<number>,
    standardizedFeatures: ArrayLike<number>,
  ) => Float64Array;
} {
  if (typeof encoderAsset !== 'object' || encoderAsset === null) {
    throw new TypeError(
      'createTimeDomainEncodersV3 requires the v3 fusion weights asset '
      + '(parsed JSON or the loadTimeDomainFusionAssetV3 result)',
    );
  }
  const candidate = encoderAsset as { real?: unknown; complex?: unknown };
  const real = validateTimeDomainRealBranchV3(
    candidate.real,
    'encoderAsset.real',
  );
  const complex = validateTimeDomainComplexBranchV3(
    candidate.complex,
    'encoderAsset.complex',
  );
  return {
    real: (packedInPhase, packedQuadrature, standardizedFeatures) =>
      forwardTimeDomainRealBranchV3(
        real,
        packedInPhase,
        packedQuadrature,
        standardizedFeatures,
      ),
    complex: (packedInPhase, packedQuadrature, standardizedFeatures) =>
      forwardTimeDomainComplexBranchV3(
        complex,
        packedInPhase,
        packedQuadrature,
        standardizedFeatures,
      ),
  };
}
