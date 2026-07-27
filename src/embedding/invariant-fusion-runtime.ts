/**
 * Pure-TypeScript inference for the invariant patch CNN and centered fusion.
 *
 * The trained complex branch is represented as paired real/imaginary arrays;
 * this module does not use ONNX, WASM, WebGL, or a complex-number runtime.
 * It consumes the staging schema emitted by
 * `training/zplane_ab/v2_full_variation/export_invariant_fusion.py`.
 *
 * Input I/Q is already packed by the invariant-patch-v1 frontend. Raw scalar
 * features are standardized with the training-only statistics in the asset.
 */

export const INVARIANT_FUSION_SCHEMA = 'atomos.invariant-fusion.paired-real' as const;
export const INVARIANT_FUSION_SCHEMA_VERSION = 3 as const;

export interface InvariantPatchConfig {
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

export interface InvariantLinear {
  in: number;
  out: number;
  weight: number[];
  bias: number[];
}

export interface FoldedRealBlock {
  in_channels: number;
  out_channels: number;
  kernel: number;
  stride: number;
  padding: number;
  batch_norm_folded: true;
  batch_norm_eps: number;
  weight: number[];
  bias: number[];
}

export interface InvariantRealBranch {
  config: InvariantPatchConfig & { encoder: 'real' };
  blocks: FoldedRealBlock[];
  patch_projection: InvariantLinear;
  fc1: InvariantLinear;
  fc2: InvariantLinear;
}

export interface PairedComplexStage {
  in_channels: number;
  out_channels: number;
  kernel: number;
  padding: number;
  weight_real: number[];
  weight_imag: number[];
  threshold_raw: number[];
}

export interface InvariantComplexBranch {
  config: InvariantPatchConfig & { encoder: 'complex' };
  stages: PairedComplexStage[];
  mag_norm_eps: number;
  modrelu_magnitude_eps: number;
  lowpass: [number, number, number];
  lags: number[];
  patch_projection: InvariantLinear;
  fc1: InvariantLinear;
  fc2: InvariantLinear;
}

export interface CenteredFusionAsset {
  real_center: number[];
  complex_center: number[];
  alpha_real: number;
  alpha_complex: number;
  weight_real: number;
  eps: number;
}

export interface InvariantClassificationAsset {
  classes: string[];
  prototypes: number[][];
  /** There is no prototype-distance rejection threshold for this model. */
  unknown_threshold: null;
  temperature: number;
  posterior: string;
}

export interface RankedLofComponent {
  branch: 'real' | 'complex';
  weight: number;
  neighbors: number;
  /** Standardized training reference embeddings, shape [rows, embed_dim]. */
  reference: number[][];
  mean: number[];
  scale: number[];
  reference_k_distance: number[];
  reference_local_density: number[];
  /** Sorted enrollment raw-LOF calibration values. */
  calibration: number[];
}

export interface WeightedLofRejectionAsset {
  kind: 'weighted_known_only_lof';
  policy: string;
  components: RankedLofComponent[];
  threshold: number;
  threshold_quantile: number;
  score: string;
  comparison: string;
  closed_label_changed_before_abstention: false;
}

export interface InvariantFusionAsset {
  schema: typeof INVARIANT_FUSION_SCHEMA;
  schema_version: typeof INVARIANT_FUSION_SCHEMA_VERSION;
  status: 'staging_not_release';
  provisional: boolean;
  packed_length: number;
  real: InvariantRealBranch;
  complex: InvariantComplexBranch;
  fusion: CenteredFusionAsset;
  feature_standardization: {
    mean: number[];
    std: number[];
  };
  classification: InvariantClassificationAsset;
  rejection: WeightedLofRejectionAsset;
  preprocess?: unknown;
  provenance?: unknown;
  release_blockers?: string[];
}

export interface InvariantForwardResult {
  standardizedFeatures: Float64Array;
  realEmbedding: Float64Array;
  complexEmbedding: Float64Array;
  fusedEmbedding: Float64Array;
}

export interface LofComponentScore {
  branch: 'real' | 'complex';
  weight: number;
  neighbors: number;
  raw: number;
  rank: number;
}

export interface InvariantClassification {
  /** Authoritative nearest-prototype label before optional abstention. */
  closedLabel: string;
  label: string | 'unknown';
  closedWinnerIndex: number;
  squaredPrototypeDistances: Float64Array;
  posterior: Float64Array;
  posteriorByClass: Record<string, number>;
  rejectionComponents: LofComponentScore[];
  rejectionScore: number;
  rejectionThreshold: number;
  isUnknown: boolean;
  realEmbedding: Float64Array;
  complexEmbedding: Float64Array;
  fusedEmbedding: Float64Array;
}

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

function finiteVector(value: unknown, path: string, length?: number): number[] {
  if (!Array.isArray(value)) throw new TypeError(`${path} must be an array`);
  if (length !== undefined && value.length !== length) {
    throw new RangeError(`${path} must contain ${length} values`);
  }
  for (let index = 0; index < value.length; index++) {
    finite(value[index], `${path}[${index}]`);
  }
  return value as number[];
}

function positiveVector(value: unknown, path: string, length: number): number[] {
  const output = finiteVector(value, path, length);
  for (let index = 0; index < output.length; index++) {
    if (output[index]! <= 0) {
      throw new RangeError(`${path}[${index}] must be positive`);
    }
  }
  return output;
}

function validateConfig(
  value: unknown,
  path: string,
  expectedEncoder: 'real' | 'complex',
): InvariantPatchConfig {
  const cfg = record(value, path);
  const patchLength = integer(cfg.patch_length, `${path}.patch_length`, 1);
  const patchCount = integer(cfg.patch_count, `${path}.patch_count`, 1);
  const patchDim = integer(cfg.patch_dim, `${path}.patch_dim`, 1);
  const hidden = integer(cfg.hidden, `${path}.hidden`, 1);
  const embedDim = integer(cfg.embed_dim, `${path}.embed_dim`, 1);
  const nFeatures = integer(cfg.n_features, `${path}.n_features`);
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
    patch_length: patchLength,
    patch_count: patchCount,
    encoder: expectedEncoder,
    patch_dim: patchDim,
    hidden,
    embed_dim: embedDim,
    n_features: nFeatures,
    set_pool: cfg.set_pool,
    dropout,
  };
}

function validateLinear(
  value: unknown,
  path: string,
  expectedIn: number,
  expectedOut: number,
): InvariantLinear {
  const layer = record(value, path);
  const input = integer(layer.in, `${path}.in`, 1);
  const output = integer(layer.out, `${path}.out`, 1);
  if (input !== expectedIn || output !== expectedOut) {
    throw new RangeError(
      `${path} shape must be [${expectedOut},${expectedIn}], got [${output},${input}]`,
    );
  }
  const weight = finiteVector(layer.weight, `${path}.weight`, input * output);
  const bias = finiteVector(layer.bias, `${path}.bias`, output);
  return { in: input, out: output, weight, bias };
}

function validateRealBranch(value: unknown): InvariantRealBranch {
  const branch = record(value, 'asset.real');
  const config = validateConfig(branch.config, 'asset.real.config', 'real') as InvariantRealBranch['config'];
  if (!Array.isArray(branch.blocks) || branch.blocks.length === 0) {
    throw new TypeError('asset.real.blocks must be a non-empty array');
  }
  let channels = 2;
  const blocks: FoldedRealBlock[] = [];
  for (let index = 0; index < branch.blocks.length; index++) {
    const path = `asset.real.blocks[${index}]`;
    const raw = record(branch.blocks[index], path);
    const input = integer(raw.in_channels, `${path}.in_channels`, 1);
    const output = integer(raw.out_channels, `${path}.out_channels`, 1);
    const kernel = integer(raw.kernel, `${path}.kernel`, 1);
    const stride = integer(raw.stride, `${path}.stride`, 1);
    const padding = integer(raw.padding, `${path}.padding`);
    if (input !== channels) throw new RangeError(`${path}.in_channels breaks the channel chain`);
    if (raw.batch_norm_folded !== true) {
      throw new RangeError(`${path} must contain folded BatchNorm parameters`);
    }
    const batchNormEps = finite(raw.batch_norm_eps, `${path}.batch_norm_eps`);
    if (batchNormEps <= 0) throw new RangeError(`${path}.batch_norm_eps must be positive`);
    blocks.push({
      in_channels: input,
      out_channels: output,
      kernel,
      stride,
      padding,
      batch_norm_folded: true,
      batch_norm_eps: batchNormEps,
      weight: finiteVector(raw.weight, `${path}.weight`, output * input * kernel),
      bias: finiteVector(raw.bias, `${path}.bias`, output),
    });
    channels = output;
  }
  const patchProjection = validateLinear(
    branch.patch_projection,
    'asset.real.patch_projection',
    2 * channels,
    config.patch_dim,
  );
  const setWidth = config.patch_dim * (config.set_pool === 'mean_std' ? 2 : 1);
  const fc1 = validateLinear(
    branch.fc1,
    'asset.real.fc1',
    setWidth + config.n_features,
    config.hidden,
  );
  const fc2 = validateLinear(
    branch.fc2,
    'asset.real.fc2',
    config.hidden,
    config.embed_dim,
  );
  return {
    config,
    blocks,
    patch_projection: patchProjection,
    fc1,
    fc2,
  };
}

function validateComplexBranch(value: unknown): InvariantComplexBranch {
  const branch = record(value, 'asset.complex');
  const config = validateConfig(
    branch.config,
    'asset.complex.config',
    'complex',
  ) as InvariantComplexBranch['config'];
  if (!Array.isArray(branch.stages) || branch.stages.length === 0) {
    throw new TypeError('asset.complex.stages must be a non-empty array');
  }
  let channels = 1;
  const stages: PairedComplexStage[] = [];
  for (let index = 0; index < branch.stages.length; index++) {
    const path = `asset.complex.stages[${index}]`;
    const raw = record(branch.stages[index], path);
    const input = integer(raw.in_channels, `${path}.in_channels`, 1);
    const output = integer(raw.out_channels, `${path}.out_channels`, 1);
    const kernel = integer(raw.kernel, `${path}.kernel`, 1);
    const padding = integer(raw.padding, `${path}.padding`);
    if (input !== channels) throw new RangeError(`${path}.in_channels breaks the channel chain`);
    stages.push({
      in_channels: input,
      out_channels: output,
      kernel,
      padding,
      weight_real: finiteVector(raw.weight_real, `${path}.weight_real`, output * input * kernel),
      weight_imag: finiteVector(raw.weight_imag, `${path}.weight_imag`, output * input * kernel),
      threshold_raw: finiteVector(raw.threshold_raw, `${path}.threshold_raw`, output),
    });
    channels = output;
  }
  const magNormEps = finite(branch.mag_norm_eps, 'asset.complex.mag_norm_eps');
  const magnitudeEps = finite(
    branch.modrelu_magnitude_eps,
    'asset.complex.modrelu_magnitude_eps',
  );
  if (magNormEps <= 0 || magnitudeEps <= 0) {
    throw new RangeError('complex normalization epsilons must be positive');
  }
  const lowpass = finiteVector(branch.lowpass, 'asset.complex.lowpass', 3);
  if (
    lowpass[0] !== 0.25
    || lowpass[1] !== 0.5
    || lowpass[2] !== 0.25
  ) {
    throw new RangeError('asset.complex.lowpass must be [0.25,0.5,0.25]');
  }
  if (!Array.isArray(branch.lags) || branch.lags.length === 0) {
    throw new TypeError('asset.complex.lags must be a non-empty array');
  }
  const lags = branch.lags.map((lag, index) =>
    integer(lag, `asset.complex.lags[${index}]`, 1));
  for (let index = 1; index < lags.length; index++) {
    if (lags[index]! <= lags[index - 1]!) {
      throw new RangeError('asset.complex.lags must be strictly increasing');
    }
  }
  const statisticWidth = channels * (2 + 2 * lags.length);
  const patchProjection = validateLinear(
    branch.patch_projection,
    'asset.complex.patch_projection',
    statisticWidth,
    config.patch_dim,
  );
  const setWidth = config.patch_dim * (config.set_pool === 'mean_std' ? 2 : 1);
  const fc1 = validateLinear(
    branch.fc1,
    'asset.complex.fc1',
    setWidth + config.n_features,
    config.hidden,
  );
  const fc2 = validateLinear(
    branch.fc2,
    'asset.complex.fc2',
    config.hidden,
    config.embed_dim,
  );
  return {
    config,
    stages,
    mag_norm_eps: magNormEps,
    modrelu_magnitude_eps: magnitudeEps,
    lowpass: [lowpass[0]!, lowpass[1]!, lowpass[2]!],
    lags,
    patch_projection: patchProjection,
    fc1,
    fc2,
  };
}

/**
 * Validate an untrusted JSON value and return the same object with a safe type.
 * All model and LOF arrays are checked for exact geometry and finite values.
 */
export function loadInvariantFusionAsset(value: unknown): InvariantFusionAsset {
  const asset = record(value, 'asset');
  if (asset.schema !== INVARIANT_FUSION_SCHEMA) {
    throw new RangeError(`unsupported invariant fusion schema ${String(asset.schema)}`);
  }
  if (asset.schema_version !== INVARIANT_FUSION_SCHEMA_VERSION) {
    throw new RangeError(`unsupported invariant fusion schema version ${String(asset.schema_version)}`);
  }
  if (asset.status !== 'staging_not_release') {
    throw new RangeError('invariant fusion asset must be staging_not_release');
  }
  if (typeof asset.provisional !== 'boolean') {
    throw new TypeError('asset.provisional must be boolean');
  }
  const packedLength = integer(asset.packed_length, 'asset.packed_length', 1);
  const real = validateRealBranch(asset.real);
  const complex = validateComplexBranch(asset.complex);
  for (const field of ['patch_length', 'patch_count', 'embed_dim', 'n_features'] as const) {
    if (real.config[field] !== complex.config[field]) {
      throw new RangeError(`real/complex config mismatch for ${field}`);
    }
  }
  if (packedLength !== real.config.patch_length * real.config.patch_count) {
    throw new RangeError('asset.packed_length does not match patch geometry');
  }

  const fusionRaw = record(asset.fusion, 'asset.fusion');
  const embedDim = real.config.embed_dim;
  const fusion: CenteredFusionAsset = {
    real_center: finiteVector(fusionRaw.real_center, 'asset.fusion.real_center', embedDim),
    complex_center: finiteVector(
      fusionRaw.complex_center,
      'asset.fusion.complex_center',
      embedDim,
    ),
    alpha_real: finite(fusionRaw.alpha_real, 'asset.fusion.alpha_real'),
    alpha_complex: finite(fusionRaw.alpha_complex, 'asset.fusion.alpha_complex'),
    weight_real: finite(fusionRaw.weight_real, 'asset.fusion.weight_real'),
    eps: finite(fusionRaw.eps, 'asset.fusion.eps'),
  };
  if (fusion.weight_real < 0 || fusion.weight_real > 1) {
    throw new RangeError('asset.fusion.weight_real must lie in [0,1]');
  }
  if (fusion.eps <= 0) throw new RangeError('asset.fusion.eps must be positive');

  const featureRaw = record(
    asset.feature_standardization,
    'asset.feature_standardization',
  );
  const featureStandardization = {
    mean: finiteVector(
      featureRaw.mean,
      'asset.feature_standardization.mean',
      real.config.n_features,
    ),
    std: positiveVector(
      featureRaw.std,
      'asset.feature_standardization.std',
      real.config.n_features,
    ),
  };

  const classificationRaw = record(asset.classification, 'asset.classification');
  if (
    !Array.isArray(classificationRaw.classes)
    || classificationRaw.classes.length === 0
    || classificationRaw.classes.some((name) => typeof name !== 'string' || name.length === 0)
    || new Set(classificationRaw.classes).size !== classificationRaw.classes.length
  ) {
    throw new RangeError('asset.classification.classes must be unique non-empty strings');
  }
  const classes = classificationRaw.classes as string[];
  if (!Array.isArray(classificationRaw.prototypes) || classificationRaw.prototypes.length !== classes.length) {
    throw new RangeError('asset.classification.prototypes must have one row per class');
  }
  const prototypes = classificationRaw.prototypes.map((prototype, index) =>
    finiteVector(
      prototype,
      `asset.classification.prototypes[${index}]`,
      2 * embedDim,
    ));
  if (classificationRaw.unknown_threshold !== null) {
    throw new RangeError('prototype-distance rejection is not defined for this asset');
  }
  const temperature = finite(
    classificationRaw.temperature,
    'asset.classification.temperature',
  );
  if (temperature <= 0) {
    throw new RangeError('asset.classification.temperature must be positive');
  }
  if (typeof classificationRaw.posterior !== 'string') {
    throw new TypeError('asset.classification.posterior must describe its rule');
  }
  const classification: InvariantClassificationAsset = {
    classes,
    prototypes,
    unknown_threshold: null,
    temperature,
    posterior: classificationRaw.posterior,
  };

  const rejectionRaw = record(asset.rejection, 'asset.rejection');
  if (rejectionRaw.kind !== 'weighted_known_only_lof') {
    throw new RangeError('asset.rejection.kind must be weighted_known_only_lof');
  }
  if (!Array.isArray(rejectionRaw.components) || rejectionRaw.components.length === 0) {
    throw new TypeError('asset.rejection.components must be a non-empty array');
  }
  const seenBranches = new Set<string>();
  const components: RankedLofComponent[] = [];
  let totalWeight = 0;
  for (let componentIndex = 0; componentIndex < rejectionRaw.components.length; componentIndex++) {
    const path = `asset.rejection.components[${componentIndex}]`;
    const raw = record(rejectionRaw.components[componentIndex], path);
    if (raw.branch !== 'real' && raw.branch !== 'complex') {
      throw new RangeError(`${path}.branch must be real or complex`);
    }
    if (seenBranches.has(raw.branch)) {
      throw new RangeError(`${path}.branch is duplicated`);
    }
    seenBranches.add(raw.branch);
    const weight = finite(raw.weight, `${path}.weight`);
    const neighbors = integer(raw.neighbors, `${path}.neighbors`, 1);
    if (weight <= 0) throw new RangeError(`${path}.weight must be positive`);
    if (!Array.isArray(raw.reference) || raw.reference.length <= neighbors) {
      throw new RangeError(`${path}.reference must contain more rows than neighbors`);
    }
    const reference = raw.reference.map((row, rowIndex) =>
      finiteVector(row, `${path}.reference[${rowIndex}]`, embedDim));
    const rows = reference.length;
    const calibration = finiteVector(raw.calibration, `${path}.calibration`);
    if (calibration.length === 0) throw new RangeError(`${path}.calibration cannot be empty`);
    for (let index = 1; index < calibration.length; index++) {
      if (calibration[index]! < calibration[index - 1]!) {
        throw new RangeError(`${path}.calibration must be sorted`);
      }
    }
    const kDistance = finiteVector(
      raw.reference_k_distance,
      `${path}.reference_k_distance`,
      rows,
    );
    const localDensity = positiveVector(
      raw.reference_local_density,
      `${path}.reference_local_density`,
      rows,
    );
    for (let index = 0; index < kDistance.length; index++) {
      if (kDistance[index]! < 0) {
        throw new RangeError(`${path}.reference_k_distance[${index}] cannot be negative`);
      }
    }
    components.push({
      branch: raw.branch,
      weight,
      neighbors,
      reference,
      mean: finiteVector(raw.mean, `${path}.mean`, embedDim),
      scale: positiveVector(raw.scale, `${path}.scale`, embedDim),
      reference_k_distance: kDistance,
      reference_local_density: localDensity,
      calibration,
    });
    totalWeight += weight;
  }
  if (Math.abs(totalWeight - 1) > 1e-9) {
    throw new RangeError('asset.rejection component weights must sum to one');
  }
  if (!seenBranches.has('real') || !seenBranches.has('complex')) {
    throw new RangeError('asset.rejection must score both real and complex branches');
  }
  const threshold = finite(rejectionRaw.threshold, 'asset.rejection.threshold');
  const thresholdQuantile = finite(
    rejectionRaw.threshold_quantile,
    'asset.rejection.threshold_quantile',
  );
  if (threshold < 0 || threshold >= 1 || thresholdQuantile < 0 || thresholdQuantile > 1) {
    throw new RangeError('rejection threshold and quantile must lie in [0,1]');
  }
  if (
    typeof rejectionRaw.policy !== 'string'
    || typeof rejectionRaw.score !== 'string'
    || typeof rejectionRaw.comparison !== 'string'
    || rejectionRaw.closed_label_changed_before_abstention !== false
  ) {
    throw new RangeError('asset.rejection policy metadata is invalid');
  }
  const rejection: WeightedLofRejectionAsset = {
    kind: 'weighted_known_only_lof',
    policy: rejectionRaw.policy,
    components,
    threshold,
    threshold_quantile: thresholdQuantile,
    score: rejectionRaw.score,
    comparison: rejectionRaw.comparison,
    closed_label_changed_before_abstention: false,
  };

  return {
    ...(asset as unknown as InvariantFusionAsset),
    schema: INVARIANT_FUSION_SCHEMA,
    schema_version: INVARIANT_FUSION_SCHEMA_VERSION,
    status: 'staging_not_release',
    provisional: asset.provisional,
    packed_length: packedLength,
    real,
    complex,
    fusion,
    feature_standardization: featureStandardization,
    classification,
    rejection,
  };
}

function assertFiniteInput(value: ArrayLike<number>, length: number, path: string): void {
  if (value.length !== length) throw new RangeError(`${path} must have length ${length}`);
  for (let index = 0; index < value.length; index++) {
    if (!Number.isFinite(value[index])) {
      throw new RangeError(`${path}[${index}] must be finite`);
    }
  }
}

function populationMeanStd(channels: Float64Array[]): Float64Array {
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

function linear(input: Float64Array, layer: InvariantLinear, relu: boolean): Float64Array {
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

function realConvRelu(
  input: Float64Array[],
  layer: FoldedRealBlock,
): Float64Array[] {
  const inputLength = input[0]!.length;
  const outputLength = Math.floor(
    (inputLength + 2 * layer.padding - layer.kernel) / layer.stride,
  ) + 1;
  const output: Float64Array[] = [];
  for (let outputChannel = 0; outputChannel < layer.out_channels; outputChannel++) {
    const values = new Float64Array(outputLength);
    const outputBase = outputChannel * layer.in_channels * layer.kernel;
    for (let time = 0; time < outputLength; time++) {
      let sum = layer.bias[outputChannel]!;
      const start = time * layer.stride - layer.padding;
      for (let inputChannel = 0; inputChannel < layer.in_channels; inputChannel++) {
        const channel = input[inputChannel]!;
        const weightBase = outputBase + inputChannel * layer.kernel;
        for (let tap = 0; tap < layer.kernel; tap++) {
          const source = start + tap;
          if (source >= 0 && source < inputLength) {
            sum += channel[source]! * layer.weight[weightBase + tap]!;
          }
        }
      }
      values[time] = sum > 0 ? sum : 0;
    }
    output.push(values);
  }
  return output;
}

interface PairedChannels {
  real: Float64Array[];
  imaginary: Float64Array[];
}

function pairedComplexConv(
  input: PairedChannels,
  layer: PairedComplexStage,
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

function normalizeAndModRelu(
  input: PairedChannels,
  layer: PairedComplexStage,
  magnitudeNormEps: number,
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
    const rms = Math.sqrt(power / length + magnitudeNormEps);
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

function unitNormalize(value: Float64Array, eps = 1e-12): Float64Array {
  let normSquared = 0;
  for (const item of value) normSquared += item * item;
  const denominator = Math.max(Math.sqrt(normSquared), eps);
  const output = new Float64Array(value.length);
  for (let index = 0; index < value.length; index++) {
    output[index] = value[index]! / denominator;
  }
  return output;
}

function setPool(patches: Float64Array[], mode: 'mean' | 'mean_std'): Float64Array {
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

function branchTail(
  patches: Float64Array[],
  features: Float64Array,
  config: InvariantPatchConfig,
  fc1: InvariantLinear,
  fc2: InvariantLinear,
): Float64Array {
  const pooled = setPool(patches, config.set_pool);
  const combined = new Float64Array(pooled.length + features.length);
  combined.set(pooled);
  combined.set(features, pooled.length);
  return unitNormalize(linear(linear(combined, fc1, true), fc2, false));
}

/** Forward one already-standardized sample through the real branch. */
export function forwardInvariantRealBranch(
  branch: InvariantRealBranch,
  inPhase: ArrayLike<number>,
  quadrature: ArrayLike<number>,
  standardizedFeatures: Float64Array,
): Float64Array {
  const patches: Float64Array[] = [];
  for (let patchIndex = 0; patchIndex < branch.config.patch_count; patchIndex++) {
    const [real, imaginary] = patchInput(
      inPhase,
      quadrature,
      patchIndex,
      branch.config.patch_length,
    );
    let channels = [real, imaginary];
    for (const block of branch.blocks) channels = realConvRelu(channels, block);
    patches.push(linear(populationMeanStd(channels), branch.patch_projection, true));
  }
  return branchTail(
    patches,
    standardizedFeatures,
    branch.config,
    branch.fc1,
    branch.fc2,
  );
}

/** Forward one already-standardized sample through the paired-real complex branch. */
export function forwardInvariantComplexBranch(
  branch: InvariantComplexBranch,
  inPhase: ArrayLike<number>,
  quadrature: ArrayLike<number>,
  standardizedFeatures: Float64Array,
): Float64Array {
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
      channels = normalizeAndModRelu(
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
    const statistics = complexStatistics(channels, branch.lags);
    patches.push(linear(statistics, branch.patch_projection, true));
  }
  return branchTail(
    patches,
    standardizedFeatures,
    branch.config,
    branch.fc1,
    branch.fc2,
  );
}

function safeUnit(value: Float64Array, eps: number): Float64Array {
  let normSquared = 0;
  for (const item of value) normSquared += item * item;
  if (normSquared <= eps * eps) {
    const fallback = new Float64Array(value.length);
    fallback[0] = 1;
    return fallback;
  }
  const inverseNorm = 1 / Math.sqrt(Math.max(normSquared, eps * eps));
  const output = new Float64Array(value.length);
  for (let index = 0; index < value.length; index++) {
    output[index] = value[index]! * inverseNorm;
  }
  return output;
}

function centeredFusion(
  real: Float64Array,
  complex: Float64Array,
  fusion: CenteredFusionAsset,
): Float64Array {
  const centeredReal = new Float64Array(real.length);
  const centeredComplex = new Float64Array(complex.length);
  for (let index = 0; index < real.length; index++) {
    centeredReal[index] = real[index]! - fusion.alpha_real * fusion.real_center[index]!;
    centeredComplex[index] = (
      complex[index]! - fusion.alpha_complex * fusion.complex_center[index]!
    );
  }
  const normalizedReal = safeUnit(centeredReal, fusion.eps);
  const normalizedComplex = safeUnit(centeredComplex, fusion.eps);
  const output = new Float64Array(real.length + complex.length);
  const realScale = Math.sqrt(fusion.weight_real);
  const complexScale = Math.sqrt(1 - fusion.weight_real);
  for (let index = 0; index < real.length; index++) {
    output[index] = realScale * normalizedReal[index]!;
    output[real.length + index] = complexScale * normalizedComplex[index]!;
  }
  return safeUnit(output, fusion.eps);
}

/** Standardize raw scalar features with the asset's training-only moments. */
export function standardizeInvariantFeatures(
  asset: InvariantFusionAsset,
  rawFeatures: ArrayLike<number>,
): Float64Array {
  const width = asset.real.config.n_features;
  assertFiniteInput(rawFeatures, width, 'rawFeatures');
  const output = new Float64Array(width);
  for (let index = 0; index < width; index++) {
    // The training cache applies NumPy float32 subtraction and division.
    const centered = Math.fround(
      rawFeatures[index]! - asset.feature_standardization.mean[index]!,
    );
    output[index] = Math.fround(
      centered / asset.feature_standardization.std[index]!,
    );
  }
  return output;
}

/** Run both branches and centered fusion for one packed capture. */
export function forwardInvariantFusion(
  asset: InvariantFusionAsset,
  inPhase: ArrayLike<number>,
  quadrature: ArrayLike<number>,
  rawFeatures: ArrayLike<number>,
): InvariantForwardResult {
  assertFiniteInput(inPhase, asset.packed_length, 'inPhase');
  assertFiniteInput(quadrature, asset.packed_length, 'quadrature');
  const standardizedFeatures = standardizeInvariantFeatures(asset, rawFeatures);
  const realEmbedding = forwardInvariantRealBranch(
    asset.real,
    inPhase,
    quadrature,
    standardizedFeatures,
  );
  const complexEmbedding = forwardInvariantComplexBranch(
    asset.complex,
    inPhase,
    quadrature,
    standardizedFeatures,
  );
  return {
    standardizedFeatures,
    realEmbedding,
    complexEmbedding,
    fusedEmbedding: centeredFusion(realEmbedding, complexEmbedding, asset.fusion),
  };
}

const referenceNormCache = new WeakMap<RankedLofComponent, Float64Array>();

function referenceNorms(component: RankedLofComponent): Float64Array {
  const cached = referenceNormCache.get(component);
  if (cached !== undefined) return cached;
  const output = new Float64Array(component.reference.length);
  for (let row = 0; row < component.reference.length; row++) {
    let norm = 0;
    for (const value of component.reference[row]!) norm += value * value;
    output[row] = norm;
  }
  referenceNormCache.set(component, output);
  return output;
}

function lowerBound(sorted: number[], value: number): number {
  let low = 0;
  let high = sorted.length;
  while (low < high) {
    const middle = low + Math.floor((high - low) / 2);
    if (sorted[middle]! < value) low = middle + 1;
    else high = middle;
  }
  return low;
}

/** Score one branch embedding with one serialized known-only LOF component. */
export function scoreRankedLof(
  component: RankedLofComponent,
  embedding: ArrayLike<number>,
): LofComponentScore {
  const dimensions = component.mean.length;
  assertFiniteInput(embedding, dimensions, `${component.branch}Embedding`);
  const query = new Float64Array(dimensions);
  let queryNorm = 0;
  for (let dimension = 0; dimension < dimensions; dimension++) {
    const value = (
      (embedding[dimension]! - component.mean[dimension]!)
      / component.scale[dimension]!
    );
    query[dimension] = value;
    queryNorm += value * value;
  }

  const neighbors = component.neighbors;
  const bestDistances = new Float64Array(neighbors);
  bestDistances.fill(Number.POSITIVE_INFINITY);
  const bestIndices = new Int32Array(neighbors);
  bestIndices.fill(-1);
  const norms = referenceNorms(component);
  for (let row = 0; row < component.reference.length; row++) {
    const reference = component.reference[row]!;
    let dot = 0;
    for (let dimension = 0; dimension < dimensions; dimension++) {
      dot += query[dimension]! * reference[dimension]!;
    }
    const distance = Math.max(queryNorm + norms[row]! - 2 * dot, 0);
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
    if (index < 0) throw new Error('LOF neighbor search did not fill its result');
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
  const rank = lowerBound(component.calibration, raw) / (component.calibration.length + 1);
  return {
    branch: component.branch,
    weight: component.weight,
    neighbors,
    raw,
    rank,
  };
}

function squaredDistances(
  embedding: Float64Array,
  prototypes: number[][],
): Float64Array {
  const output = new Float64Array(prototypes.length);
  for (let classIndex = 0; classIndex < prototypes.length; classIndex++) {
    const prototype = prototypes[classIndex]!;
    let distance = 0;
    for (let dimension = 0; dimension < embedding.length; dimension++) {
      const delta = embedding[dimension]! - prototype[dimension]!;
      distance += delta * delta;
    }
    output[classIndex] = distance;
  }
  return output;
}

/**
 * Classify one packed capture. Weighted LOF may only replace the authoritative
 * fused closed label with `unknown`; it never selects another known class.
 */
export function classifyInvariantFusion(
  asset: InvariantFusionAsset,
  inPhase: ArrayLike<number>,
  quadrature: ArrayLike<number>,
  rawFeatures: ArrayLike<number>,
): InvariantClassification {
  const forward = forwardInvariantFusion(asset, inPhase, quadrature, rawFeatures);
  const distances = squaredDistances(
    forward.fusedEmbedding,
    asset.classification.prototypes,
  );
  let winner = 0;
  for (let index = 1; index < distances.length; index++) {
    if (distances[index]! < distances[winner]!) winner = index;
  }
  const logits = new Float64Array(distances.length);
  let maximum = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < logits.length; index++) {
    const value = -distances[index]! / asset.classification.temperature;
    logits[index] = value;
    maximum = Math.max(maximum, value);
  }
  const posterior = new Float64Array(logits.length);
  let normalizer = 0;
  for (let index = 0; index < logits.length; index++) {
    const value = Math.exp(logits[index]! - maximum);
    posterior[index] = value;
    normalizer += value;
  }
  const posteriorByClass: Record<string, number> = {};
  for (let index = 0; index < posterior.length; index++) {
    posterior[index] = posterior[index]! / normalizer;
    posteriorByClass[asset.classification.classes[index]!] = posterior[index]!;
  }

  const rejectionComponents = asset.rejection.components.map((component) =>
    scoreRankedLof(
      component,
      component.branch === 'real'
        ? forward.realEmbedding
        : forward.complexEmbedding,
    ));
  let rejectionScore = 0;
  for (const component of rejectionComponents) {
    rejectionScore += component.weight * component.rank;
  }
  const isUnknown = rejectionScore > asset.rejection.threshold;
  const closedLabel = asset.classification.classes[winner]!;
  return {
    closedLabel,
    label: isUnknown ? 'unknown' : closedLabel,
    closedWinnerIndex: winner,
    squaredPrototypeDistances: distances,
    posterior,
    posteriorByClass,
    rejectionComponents,
    rejectionScore,
    rejectionThreshold: asset.rejection.threshold,
    isUnknown,
    realEmbedding: forward.realEmbedding,
    complexEmbedding: forward.complexEmbedding,
    fusedEmbedding: forward.fusedEmbedding,
  };
}
