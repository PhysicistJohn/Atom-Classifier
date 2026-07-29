/**
 * Generate a development-only held-out length/sample-rate scale evaluation
 * corpus through SignalLab's public AtomizerMeasurementService.
 *
 * This script intentionally imports no low-level waveform synthesizer and
 * performs no FFT or post-hoc sample-array resampling.  Physical scale is
 * exercised by asking the service for the same source coordinate and receiver
 * realization at native, 1.25x, 1.5x, and 2x output sample rates.
 *
 * Build and run from Atom-Classifier with the pinned Node 22 toolchain:
 *
 *   PATH=/Users/johnelliott/.nvm/versions/node/v22.23.1/bin:$PATH \
 *     npx tsup tools/generate-current-signallab-scale-eval.ts \
 *       --format esm --platform node \
 *       --out-dir .artifacts/current-scale-eval-generator \
 *       --no-splitting --treeshake --silent
 *
 *   PATH=/Users/johnelliott/.nvm/versions/node/v22.23.1/bin:$PATH \
 *     REFERENCE_CORPUS_DIR=training/artifacts/<current-training-corpus> \
 *     OUTPUT_DIR=training/artifacts/<fresh-current-scale-eval> \
 *     node .artifacts/current-scale-eval-generator/\
 *generate-current-signallab-scale-eval.js
 *
 * The reference corpus is mandatory for a real evaluation.  Its current-source
 * phase, receiver-seed, and content identities are excluded.  A deliberately
 * labelled UNBOUND_SMOKE=1 mode exists only for short API smoke tests.
 */

import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import {
  closeSync,
  existsSync,
  mkdirSync,
  openSync,
  readFileSync,
  readSync,
  renameSync,
  rmSync,
  statSync,
  writeFileSync,
  writeSync,
} from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  receiverImpairmentPresetSchema,
  type ReceiverImpairmentPreset,
} from '../../Atom-SignalLab/src/contracts.js';
import {
  FIXED_DIGITAL_PROFILE_BINDINGS,
  type FixedDigitalProfile,
  type FixedDigitalProfileBinding,
} from '../../Atom-SignalLab/src/fixed-digital-profile-binding.js';
import {
  MAX_COMPLEX_IQ_SAMPLE_RATE_HZ,
  complexIqMeasurementSchema,
  type ComplexIqMeasurement,
} from '../../Atom-SignalLab/src/measurement-contract.js';
import {
  AtomizerMeasurementService,
  type MeasurementBuildIdentity,
} from '../../Atom-SignalLab/src/measurement-service.js';

export const SCALE_FACTORS = Object.freeze([
  Object.freeze({ key: '1', value: 1, numerator: 1, denominator: 1 }),
  Object.freeze({ key: '1.25', value: 1.25, numerator: 5, denominator: 4 }),
  Object.freeze({ key: '1.5', value: 1.5, numerator: 3, denominator: 2 }),
  Object.freeze({ key: '2', value: 2, numerator: 2, denominator: 1 }),
] as const);

export const PREFIX_LENGTHS = Object.freeze([4_096, 8_192, 16_384] as const);

export const CYCLIC_PRESET_SCHEDULE = Object.freeze([
  'clean',
  'awgn',
  'multipath',
  'carrier-offset',
  'phase-noise',
  'iq-imbalance',
  'dc-offset',
  'pa-compression',
  'composite',
] as const satisfies readonly ReceiverImpairmentPreset[]);

// A one-shot artifact always starts at its sole origin.  Only stochastic
// receiver presets can create a genuinely new held receiver realization.
export const ONE_SHOT_PRESET_SCHEDULE = Object.freeze([
  'awgn',
  'phase-noise',
  'composite',
] as const satisfies readonly ReceiverImpairmentPreset[]);

// tsup bundles the sibling SignalLab implementation. Bind runtime provenance
// checks to the source revision this generator was reviewed against.
export const EXPECTED_SIGNAL_LAB_GIT_COMMIT =
  '94171baf31bfa62150d4e9684e1fe33e5a343dc0';
export const EXPECTED_SIGNAL_LAB_GIT_TREE =
  'ccda4401748953bb6705776694b29b571d4c684d';
export const GENERATOR_LINEAGE_SCHEMA =
  'atom-classifier-generator-lineage-v1' as const;
export const GENERATOR_SOURCE_PATH =
  'tools/generate-current-signallab-scale-eval.ts' as const;
export const GENERATOR_BUNDLE_PATH =
  '.artifacts/current-scale-eval-generator/'
  + 'generate-current-signallab-scale-eval.js' as const;
export const SCALE_EVAL_SCHEMA =
  'current-signallab-service-scale-eval-v3' as const;
export const PRODUCTION_CAPTURE_BANDWIDTH_RULE =
  '2 * abs(nativeCarrierOffsetHz) + signalBandwidthHz at every scale factor' as const;
export const INFORMATIVE_PREFIX_CENTERED_RMS_FLOOR = 1e-4 as const;
export const PRE_IMPAIRMENT_PREFIX_ADMISSION_RULE =
  'every service-clean scale cell has a finite, non-constant 4,096-sample prefix with centered complex RMS >= 0.0001 before receiver impairment' as const;
export const SCALE_FAMILY_CONTENT_UNIQUENESS_RULE =
  'all four published valid-content SHA-256 values are pairwise distinct before publication' as const;

export type PublicClass = 'gsm' | 'ofdm' | 'dsss' | 'bluetooth';
export type ScaleFactor = typeof SCALE_FACTORS[number];

export interface ScaleEvalOptions {
  readonly classifierRoot: string;
  readonly generatorBundlePath: string;
  readonly outputDirectory: string;
  readonly referenceCorpusDirectory?: string;
  readonly signalLabRoot: string;
  readonly evalSeed: number;
  readonly realizationsPerProfile: number;
  readonly storageSampleCount: number;
  readonly allowOverwrite: boolean;
  readonly unboundSmoke: boolean;
  readonly profiles: readonly FixedDigitalProfile[];
}

export interface GeneratorLineage {
  readonly schema: typeof GENERATOR_LINEAGE_SCHEMA;
  readonly repository: 'Atom-Classifier';
  readonly sourcePath: typeof GENERATOR_SOURCE_PATH;
  readonly sourceSha256: string;
  readonly bundlePath: typeof GENERATOR_BUNDLE_PATH;
  readonly bundleSha256: string;
  readonly sourceBundleBindingSha256: string;
}

interface GitIdentity {
  readonly commit: string;
  readonly tree: string;
  readonly worktreeClean: boolean;
}

interface ReferenceCorpus {
  readonly directory: string;
  readonly manifestPath: string;
  readonly manifestSha256: string;
  readonly rawSha256: string;
  readonly corpusSeed: number;
  readonly sourceCommit: string;
  readonly sourceTree: string;
  readonly forbiddenPhases: ReadonlyMap<string, ReadonlySet<number>>;
  readonly forbiddenChannelSeeds: ReadonlySet<number>;
  readonly forbiddenReceiverSeeds: ReadonlySet<number>;
  readonly forbiddenContentHashes: ReadonlySet<string>;
  readonly count: number;
}

interface DecodedMeasurement {
  readonly measurement: ComplexIqMeasurement;
  readonly bytes: Uint8Array;
  readonly storedBytes: Uint8Array;
  readonly contentSha256: string;
  readonly storedSha256: string;
  readonly validSampleCount: number;
  readonly scale: ScaleFactor;
}

interface EvaluationRealization {
  readonly profile: FixedDigitalProfile;
  readonly publicClass: PublicClass;
  readonly replay: 'cyclic' | 'one-shot';
  readonly realizationIndex: number;
  readonly pairId: string;
  readonly phaseNativeSample: number;
  readonly receiverPreset: ReceiverImpairmentPreset;
  readonly receiverChannelSeed: number;
  readonly measurements: readonly DecodedMeasurement[];
}

const DEFAULT_EVAL_SEED = 20_262_001;
// Twenty-four gives every individual fixed profile at least the conventional
// twenty held-out observations while keeping the full four-rate corpus below
// roughly 400 MiB.
const DEFAULT_REALIZATIONS_PER_PROFILE = 24;
const DEFAULT_STORAGE_SAMPLE_COUNT = 16_384;
const BYTES_PER_COMPLEX_SAMPLE = 8;
const HASH_PATTERN = /^[a-f0-9]{64}$/;
const UUID_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

export function publicClassForProfile(
  profile: FixedDigitalProfile,
): PublicClass {
  if (profile.startsWith('gsm-')) return 'gsm';
  if (profile === 'wifi-hr-dsss-11m') return 'dsss';
  if (profile.startsWith('bluetooth-')) return 'bluetooth';
  if (
    profile.startsWith('lte-')
    || profile.startsWith('nr-')
    || profile === 'wifi-ofdm-20m'
    || profile.startsWith('wifi6-')
  ) {
    return 'ofdm';
  }
  throw new Error(`Fixed profile ${profile} has no public classifier class`);
}

export function scaledSampleRateHz(
  nativeSampleRateHz: number,
  scale: ScaleFactor,
): number {
  requireSafeInteger(
    nativeSampleRateHz,
    1,
    MAX_COMPLEX_IQ_SAMPLE_RATE_HZ,
    'native sample rate',
  );
  const numerator = nativeSampleRateHz * scale.numerator;
  if (
    !Number.isSafeInteger(numerator)
    || numerator % scale.denominator !== 0
  ) {
    throw new RangeError(
      `Native sample rate ${nativeSampleRateHz} cannot represent scale `
      + `${scale.key} as an integer output rate`,
    );
  }
  const output = numerator / scale.denominator;
  requireSafeInteger(
    output,
    1,
    MAX_COMPLEX_IQ_SAMPLE_RATE_HZ,
    `scale ${scale.key} output sample rate`,
  );
  return output;
}

export function maximumOneShotOutputSamples(
  captureSamples: number,
  scale: ScaleFactor,
): number {
  requireSafeInteger(
    captureSamples,
    1,
    Number.MAX_SAFE_INTEGER,
    'one-shot capture samples',
  );
  return Math.floor(
    captureSamples * scale.numerator / scale.denominator,
  );
}

/**
 * Exact production capture floor used by Atomizer's default SignalLab I/Q
 * configuration. Keeping this fixed while only the requested output rate
 * changes prevents offset-native artifacts from being silently recentered.
 */
export function productionCaptureBandwidthHz(
  binding: Pick<
    FixedDigitalProfileBinding,
    'nativeCarrierOffsetHz' | 'signalBandwidthHz'
  >,
): number {
  const result =
    2 * Math.abs(binding.nativeCarrierOffsetHz) + binding.signalBandwidthHz;
  requireSafeInteger(
    result,
    1,
    MAX_COMPLEX_IQ_SAMPLE_RATE_HZ,
    'production capture bandwidth',
  );
  return result;
}

export function availablePrefixLengths(
  validSampleCount: number,
): number[] {
  requireSafeInteger(
    validSampleCount,
    PREFIX_LENGTHS[0],
    Number.MAX_SAFE_INTEGER,
    'valid sample count',
  );
  return PREFIX_LENGTHS.filter((length) => length <= validSampleCount);
}

export function deterministicPhaseCandidates(
  periodSamples: number,
  evalSeed: number,
  profile: string,
  forbidden: ReadonlySet<number> = new Set(),
): number[] {
  requireSafeInteger(
    periodSamples,
    1,
    Number.MAX_SAFE_INTEGER,
    'native period samples',
  );
  const offset =
    hashUint32(`${evalSeed}\0scale-eval-phase-offset\0${profile}`)
    % periodSamples;
  let stride =
    1 + (
      hashUint32(`${evalSeed}\0scale-eval-phase-stride\0${profile}`)
      % Math.max(1, periodSamples - 1)
    );
  while (greatestCommonDivisor(stride, periodSamples) !== 1) {
    stride = stride === periodSamples - 1 ? 1 : stride + 1;
  }
  const output: number[] = [];
  for (let index = 0; index < periodSamples; index += 1) {
    const phase = (offset + index * stride) % periodSamples;
    if (!forbidden.has(phase)) output.push(phase);
  }
  return output;
}

export function oneShotPresetForRealization(
  realizationIndex: number,
): ReceiverImpairmentPreset {
  requireSafeInteger(
    realizationIndex,
    0,
    Number.MAX_SAFE_INTEGER,
    'one-shot realization index',
  );
  return ONE_SHOT_PRESET_SCHEDULE[
    realizationIndex % ONE_SHOT_PRESET_SCHEDULE.length
  ]!;
}

export function readOptions(
  environment: NodeJS.ProcessEnv = process.env,
): ScaleEvalOptions {
  const evalSeed = parseEnvironmentInteger(
    environment.EVAL_SEED,
    DEFAULT_EVAL_SEED,
    0,
    Number.MAX_SAFE_INTEGER,
    'EVAL_SEED',
  );
  const realizationsPerProfile = parseEnvironmentInteger(
    environment.REALIZATIONS_PER_PROFILE,
    DEFAULT_REALIZATIONS_PER_PROFILE,
    1,
    64,
    'REALIZATIONS_PER_PROFILE',
  );
  const storageSampleCount = parseEnvironmentInteger(
    environment.SAMPLE_COUNT,
    DEFAULT_STORAGE_SAMPLE_COUNT,
    PREFIX_LENGTHS[PREFIX_LENGTHS.length - 1]!,
    PREFIX_LENGTHS[PREFIX_LENGTHS.length - 1]!,
    'SAMPLE_COUNT',
  );
  const allowOverwrite = parseBooleanFlag(
    environment.ALLOW_OVERWRITE,
    'ALLOW_OVERWRITE',
  );
  const unboundSmoke = parseBooleanFlag(
    environment.UNBOUND_SMOKE,
    'UNBOUND_SMOKE',
  );
  const allProfiles = (
    Object.keys(FIXED_DIGITAL_PROFILE_BINDINGS) as FixedDigitalProfile[]
  ).sort();
  const profiles = parseProfileFilter(
    environment.PROFILE_FILTER,
    allProfiles,
  );
  if (!unboundSmoke && profiles.length !== allProfiles.length) {
    throw new Error(
      'PROFILE_FILTER is allowed only with UNBOUND_SMOKE=1; a real '
      + 'evaluation must cover all fixed profiles',
    );
  }
  const classifierRoot = process.cwd();
  const packageDocument = JSON.parse(
    readFileSync(resolve(classifierRoot, 'package.json'), 'utf8'),
  ) as { readonly name?: unknown };
  if (packageDocument.name !== 'atomos-classifier') {
    throw new Error(
      'Run the bundled current scale-eval generator from the Atom-Classifier '
      + 'repository root',
    );
  }
  const signalLabRoot = resolve(classifierRoot, '../Atom-SignalLab');
  if (
    environment.SIGNALLAB_ROOT !== undefined
    && resolve(environment.SIGNALLAB_ROOT) !== signalLabRoot
  ) {
    throw new Error(
      'This bundle is statically compiled from the sibling Atom-SignalLab; '
      + 'SIGNALLAB_ROOT cannot redirect provenance to different source',
    );
  }
  const referenceCorpusDirectory = environment.REFERENCE_CORPUS_DIR === undefined
    ? undefined
    : resolve(environment.REFERENCE_CORPUS_DIR);
  if (referenceCorpusDirectory === undefined && !unboundSmoke) {
    throw new Error(
      'REFERENCE_CORPUS_DIR is required unless UNBOUND_SMOKE=1 is explicit',
    );
  }
  const outputDirectory = resolve(
    environment.OUTPUT_DIR
      ?? resolve(
        classifierRoot,
        'training',
        'artifacts',
        `signallab-current-scale-eval-seed${evalSeed}`,
      ),
  );
  if (
    referenceCorpusDirectory !== undefined
    && referenceCorpusDirectory === outputDirectory
  ) {
    throw new Error('Evaluation output cannot overwrite its reference corpus');
  }
  return {
    classifierRoot,
    generatorBundlePath: process.argv[1] === undefined
      ? ''
      : resolve(process.argv[1]),
    outputDirectory,
    referenceCorpusDirectory,
    signalLabRoot,
    evalSeed,
    realizationsPerProfile,
    storageSampleCount,
    allowOverwrite,
    unboundSmoke,
    profiles,
  };
}

export function generateCurrentSignalLabScaleEval(
  options: ScaleEvalOptions,
): void {
  const generatorLineage = readGeneratorLineage(
    options.classifierRoot,
    options.generatorBundlePath,
  );
  if (options.storageSampleCount !== DEFAULT_STORAGE_SAMPLE_COUNT) {
    throw new Error('Scale evaluation storage must be exactly 16,384 samples');
  }
  for (const preset of [
    ...CYCLIC_PRESET_SCHEDULE,
    ...ONE_SHOT_PRESET_SCHEDULE,
  ]) {
    receiverImpairmentPresetSchema.parse(preset);
  }
  const sourceGit = readGitIdentity(options.signalLabRoot);
  if (!sourceGit.worktreeClean) {
    throw new Error(
      'SignalLab source worktree is dirty; refusing unidentifiable evidence',
    );
  }
  if (
    sourceGit.commit !== EXPECTED_SIGNAL_LAB_GIT_COMMIT
    || sourceGit.tree !== EXPECTED_SIGNAL_LAB_GIT_TREE
  ) {
    throw new Error(
      `SignalLab source identity drifted: expected commit `
      + `${EXPECTED_SIGNAL_LAB_GIT_COMMIT} / tree `
      + `${EXPECTED_SIGNAL_LAB_GIT_TREE}, received ${sourceGit.commit} / `
      + `${sourceGit.tree}. Review the source change and update the explicit `
      + 'scale-eval generator pins before producing new evidence.',
    );
  }
  const buildIdentity = readBuildIdentity(options.signalLabRoot);
  const reference = options.referenceCorpusDirectory === undefined
    ? undefined
    : readReferenceCorpus(options.referenceCorpusDirectory);
  if (!options.unboundSmoke && reference === undefined) {
    throw new Error('A bound evaluation requires a reference corpus');
  }
  if (reference !== undefined) {
    if (
      reference.sourceCommit !== sourceGit.commit
      || reference.sourceTree !== sourceGit.tree
    ) {
      throw new Error(
        'Reference corpus and evaluation use different SignalLab source',
      );
    }
    if (reference.corpusSeed === options.evalSeed) {
      throw new Error(
        'Evaluation seed must differ from the current training corpus seed',
      );
    }
  }

  const outputPaths = prepareOutputs(options);
  let descriptor = -1;
  let completed = false;
  const items: Record<string, unknown>[] = [];
  const dataHash = createHash('sha256');
  const evaluationContentHashes = new Set<string>();
  const evaluationPairIds = new Set<string>();
  const profileCounts: Record<string, number> = {};
  const classCounts = new Map<PublicClass, number>();

  try {
    descriptor = openSync(outputPaths.rawPart, 'wx');
    for (const profile of options.profiles) {
      const binding = FIXED_DIGITAL_PROFILE_BINDINGS[profile];
      const realizations = binding.replay === 'cyclic'
        ? materializeCyclicRealizations({
            profile,
            binding,
            options,
            buildIdentity,
            reference,
            evaluationContentHashes,
          })
        : materializeOneShotRealizations({
            profile,
            binding,
            options,
            buildIdentity,
            reference,
            evaluationContentHashes,
          });
      if (realizations.length !== options.realizationsPerProfile) {
        throw new Error(
          `${profile} emitted ${realizations.length} realization groups, `
          + `expected ${options.realizationsPerProfile}`,
        );
      }
      let profileRowCount = 0;
      for (const realization of realizations) {
        if (evaluationPairIds.has(realization.pairId)) {
          throw new Error(`Duplicate paired identity ${realization.pairId}`);
        }
        if (!hasExactUniqueScaleFamilyContentHashes(
          realization.measurements.map(({ contentSha256 }) => contentSha256),
        )) {
          throw new Error(
            `${realization.pairId} contains byte-identical scale cells`,
          );
        }
        evaluationPairIds.add(realization.pairId);
        for (const measured of realization.measurements) {
          const index = items.length;
          writeAll(descriptor, measured.storedBytes);
          dataHash.update(measured.storedBytes);
          const receiverSeed = impairmentSeedFromReceipt(
            measured.measurement,
            realization.receiverPreset,
          );
          items.push({
            index,
            cls: realization.publicClass,
            profile,
            profileId: profile,
            replay: realization.replay,
            pairId: realization.pairId,
            heldOutIdentity: realization.replay === 'cyclic'
              ? 'held_native_phase_and_receiver_realization'
              : 'held_receiver_realization_only_one_shot_waveform_fixed',
            realizationIndex: realization.realizationIndex,
            phaseNativeSample: realization.phaseNativeSample,
            receiverPreset: realization.receiverPreset,
            receiverRealizationChannelSeed:
              realization.receiverChannelSeed,
            receiverRealizationSeed: receiverSeed,
            scaleFactor: measured.scale.value,
            scaleFactorKey: measured.scale.key,
            scaleNumerator: measured.scale.numerator,
            scaleDenominator: measured.scale.denominator,
            nativeSampleRateHz: binding.nativeSampleRateHz,
            sampleRateHz: measured.measurement.sampleRateHz,
            nativeCarrierOffsetHz: binding.nativeCarrierOffsetHz,
            signalBandwidthHz: binding.signalBandwidthHz,
            captureBandwidthHz: measured.measurement.captureBandwidthHz,
            validSampleCount: measured.validSampleCount,
            storageSampleCount: options.storageSampleCount,
            availablePrefixLengths:
              availablePrefixLengths(measured.validSampleCount),
            zeroPaddedAfterValidSampleCount:
              measured.validSampleCount < options.storageSampleCount,
            contentSha256: measured.contentSha256,
            storedSha256: measured.storedSha256,
            measurementReceipt: receiptForManifest(measured.measurement),
          });
          profileRowCount += 1;
          classCounts.set(
            realization.publicClass,
            (classCounts.get(realization.publicClass) ?? 0) + 1,
          );
        }
      }
      profileCounts[profile] = profileRowCount;
      process.stdout.write(
        `[current-scale-eval] ${profile}: ${profileRowCount} rows\n`,
      );
    }
    closeSync(descriptor);
    descriptor = -1;
    const dataSha256 = dataHash.digest('hex');
    assertGeneratorLineageUnchanged(
      generatorLineage,
      options.classifierRoot,
      options.generatorBundlePath,
    );
    const manifest = {
      schema: SCALE_EVAL_SCHEMA,
      generator: SCALE_EVAL_SCHEMA,
      generatorPath: 'tools/generate-current-signallab-scale-eval.ts',
      generatorLineage,
      developmentOnly: true,
      releaseEvidence: false,
      sealedReleaseDataUsed: 0,
      consumedHistoricalTestRowsUsed: 0,
      lowLevelSynthesizerCalls: false,
      frequencyTransformsUsed: false,
      physicalScaleMethod:
        'AtomizerMeasurementService derived output sample-rate request',
      servicePath: [
        'AtomizerMeasurementService.selectProfile',
        'AtomizerMeasurementService.configureChannel',
        'MeasurementServiceContinuation(v2)',
        'AtomizerMeasurementService.acquireIq',
        'complexIqMeasurementSchema.parse',
      ],
      evalSeed: options.evalSeed,
      realizationsPerProfile: options.realizationsPerProfile,
      sampleCount: options.storageSampleCount,
      storageSampleCount: options.storageSampleCount,
      format: 'cf32le-interleaved',
      dataFile: 'scale_eval.f32',
      dataSha256,
      scaleFactors: SCALE_FACTORS.map((factor) => ({
        key: factor.key,
        value: factor.value,
        numerator: factor.numerator,
        denominator: factor.denominator,
      })),
      prefixLengths: [...PREFIX_LENGTHS],
      scalePairingRule: (
        'same fixed profile, native source coordinate, receiver preset, and '
        + 'channel seed; only requested service output sample rate changes'
      ),
      captureBandwidthRule: PRODUCTION_CAPTURE_BANDWIDTH_RULE,
      informativePrefixCenteredRmsFloor:
        INFORMATIVE_PREFIX_CENTERED_RMS_FLOOR,
      preImpairmentPrefixAdmissionRule:
        PRE_IMPAIRMENT_PREFIX_ADMISSION_RULE,
      scaleFamilyContentUniquenessRule:
        SCALE_FAMILY_CONTENT_UNIQUENESS_RULE,
      oneShotLengthRule: (
        'validSampleCount is bounded by floor(captureSamples * outputRate / '
        + 'nativeRate); unavailable 16,384-sample BLE cells are absent '
        + 'evidence and the fixed-width tail is exact zero storage'
      ),
      heldOutContract: {
        referenceBound: reference !== undefined,
        cyclic:
          'native phases present in reference corpus are excluded',
        oneShot:
          'waveform origin cannot be held out; stochastic receiver seeds '
          + 'present in reference corpus are excluded',
        exactValidContent:
          'every evaluation valid-content SHA-256 is absent from reference',
      },
      unboundSmoke: options.unboundSmoke,
      referenceCorpus: reference === undefined
        ? null
        : {
            directory: reference.directory,
            manifest: 'corpus.json',
            manifestSha256: reference.manifestSha256,
            rawSha256: reference.rawSha256,
            count: reference.count,
            corpusSeed: reference.corpusSeed,
          },
      source: {
        repository: 'Atom-SignalLab',
        gitCommit: sourceGit.commit,
        gitTree: sourceGit.tree,
        worktreeClean: sourceGit.worktreeClean,
        contractSha256: buildIdentity.contractSha256,
        generatorContractBindingSha256:
          buildIdentity.generatorContractBindingSha256,
      },
      profiles: [...options.profiles],
      classes: [...new Set(options.profiles.map(publicClassForProfile))].sort(),
      profilePublicClassMap: Object.fromEntries(
        options.profiles.map((profile) => [
          profile,
          publicClassForProfile(profile),
        ]),
      ),
      count: items.length,
      perProfile: profileCounts,
      perClass: Object.fromEntries(
        [...classCounts.entries()].sort(([left], [right]) =>
          left.localeCompare(right)),
      ),
      items,
    };
    writeFileSync(
      outputPaths.manifestPart,
      `${JSON.stringify(manifest, null, 2)}\n`,
      { encoding: 'utf8', flag: 'wx' },
    );
    renameSync(outputPaths.rawPart, outputPaths.rawFinal);
    renameSync(outputPaths.manifestPart, outputPaths.manifestFinal);
    completed = true;
    process.stdout.write(
      `[current-scale-eval] wrote ${items.length} rows to `
      + `${options.outputDirectory}\n`,
    );
    process.stdout.write(
      `[current-scale-eval] scale_eval.f32 sha256 ${dataSha256}\n`,
    );
  } finally {
    if (descriptor >= 0) closeQuietly(descriptor);
    if (!completed) {
      for (const path of [
        outputPaths.rawPart,
        outputPaths.manifestPart,
        outputPaths.rawFinal,
        outputPaths.manifestFinal,
      ]) {
        rmSync(path, { force: true });
      }
    }
  }
}

function materializeCyclicRealizations(input: {
  readonly profile: FixedDigitalProfile;
  readonly binding: FixedDigitalProfileBinding;
  readonly options: ScaleEvalOptions;
  readonly buildIdentity: MeasurementBuildIdentity;
  readonly reference?: ReferenceCorpus;
  readonly evaluationContentHashes: Set<string>;
}): EvaluationRealization[] {
  const {
    profile,
    binding,
    options,
    buildIdentity,
    reference,
    evaluationContentHashes,
  } = input;
  if (binding.replay !== 'cyclic' || binding.nativePeriodSamples === undefined) {
    throw new Error(`${profile} is not a cyclic binding`);
  }
  const forbidden = reference?.forbiddenPhases.get(profile) ?? new Set();
  const candidates = deterministicPhaseCandidates(
    binding.nativePeriodSamples,
    options.evalSeed,
    profile,
    forbidden,
  );
  const output: EvaluationRealization[] = [];
  let candidateIndex = 0;
  while (output.length < options.realizationsPerProfile) {
    if (candidateIndex >= candidates.length) {
      throw new Error(
        `${profile} exhausted held-out occupied phase candidates`,
      );
    }
    const phaseNativeSample = candidates[candidateIndex++]!;
    const preImpairmentMeasurements = acquireScaleFamily({
      profile,
      binding,
      phaseNativeSample,
      receiverPreset: 'clean',
      channelSeed: deterministicNonzeroSeed(
        options.evalSeed,
        `${profile}\0phase\0${phaseNativeSample}\0pre-impairment-admission`,
      ),
      buildIdentity,
      storageSampleCount: options.storageSampleCount,
    });
    if (!hasInformativeScaleFamilyPrefixes(preImpairmentMeasurements)) {
      // The candidate starts in a cyclic idle interval at one or more output
      // rates. Advance the deterministic held-out phase schedule; receiver
      // impairments must never manufacture apparent occupancy.
      continue;
    }
    const realizationIndex = output.length;
    const receiverPreset = CYCLIC_PRESET_SCHEDULE[
      realizationIndex % CYCLIC_PRESET_SCHEDULE.length
    ]!;
    let seedAttempt = 0;
    let accepted:
      | {
          channelSeed: number;
          measurements: DecodedMeasurement[];
        }
      | undefined;
    while (seedAttempt < 64 && accepted === undefined) {
      const channelSeed = deterministicNonzeroSeed(
        options.evalSeed,
        `${profile}\0phase\0${phaseNativeSample}\0${receiverPreset}`
        + `\0attempt\0${seedAttempt}`,
      );
      if (reference?.forbiddenChannelSeeds.has(channelSeed)) {
        seedAttempt += 1;
        continue;
      }
      const measurements = acquireScaleFamily({
        profile,
        binding,
        phaseNativeSample,
        receiverPreset,
        channelSeed,
        buildIdentity,
        storageSampleCount: options.storageSampleCount,
      });
      if (
        measurements.some(
          (measurement) =>
            !hasInformativeCf32lePrefix(
              measurement.bytes,
              PREFIX_LENGTHS[0],
            ),
        )
        || !hasExactUniqueScaleFamilyContentHashes(
          measurements.map(({ contentSha256 }) => contentSha256),
        )
      ) {
        seedAttempt += 1;
        continue;
      }
      const receiverSeeds = measurements
        .map((measurement) =>
          impairmentSeedFromReceipt(measurement.measurement, receiverPreset))
        .filter((seed): seed is number => seed !== null);
      const collides = measurements.some(
        (measurement) =>
          reference?.forbiddenContentHashes.has(
            measurement.contentSha256,
          ) === true
          || evaluationContentHashes.has(measurement.contentSha256),
      ) || receiverSeeds.some(
        (seed) => reference?.forbiddenReceiverSeeds.has(seed) === true,
      );
      if (collides) {
        seedAttempt += 1;
        continue;
      }
      accepted = { channelSeed, measurements };
    }
    if (accepted === undefined) continue;
    for (const measurement of accepted.measurements) {
      evaluationContentHashes.add(measurement.contentSha256);
    }
    const pairId =
      `scale-pair:${profile}:phase:${phaseNativeSample}:preset:`
      + `${receiverPreset}:channel-seed:${accepted.channelSeed}`;
    output.push({
      profile,
      publicClass: publicClassForProfile(profile),
      replay: 'cyclic',
      realizationIndex,
      pairId,
      phaseNativeSample,
      receiverPreset,
      receiverChannelSeed: accepted.channelSeed,
      measurements: accepted.measurements,
    });
  }
  return output;
}

function materializeOneShotRealizations(input: {
  readonly profile: FixedDigitalProfile;
  readonly binding: FixedDigitalProfileBinding;
  readonly options: ScaleEvalOptions;
  readonly buildIdentity: MeasurementBuildIdentity;
  readonly reference?: ReferenceCorpus;
  readonly evaluationContentHashes: Set<string>;
}): EvaluationRealization[] {
  const {
    profile,
    binding,
    options,
    buildIdentity,
    reference,
    evaluationContentHashes,
  } = input;
  if (binding.replay !== 'one-shot' || binding.captureSamples === undefined) {
    throw new Error(`${profile} is not a one-shot binding`);
  }
  const preImpairmentMeasurements = acquireScaleFamily({
    profile,
    binding,
    phaseNativeSample: 0,
    receiverPreset: 'clean',
    channelSeed: deterministicNonzeroSeed(
      options.evalSeed,
      `${profile}\0one-shot\0pre-impairment-admission`,
    ),
    buildIdentity,
    storageSampleCount: options.storageSampleCount,
  });
  if (!hasInformativeScaleFamilyPrefixes(preImpairmentMeasurements)) {
    throw new Error(
      `${profile} one-shot origin has a non-informative clean live prefix`,
    );
  }
  const output: EvaluationRealization[] = [];
  for (
    let realizationIndex = 0;
    realizationIndex < options.realizationsPerProfile;
    realizationIndex += 1
  ) {
    const receiverPreset = oneShotPresetForRealization(realizationIndex);
    let accepted:
      | {
          channelSeed: number;
          measurements: DecodedMeasurement[];
        }
      | undefined;
    for (let attempt = 0; attempt < 128 && accepted === undefined; attempt += 1) {
      const channelSeed = deterministicNonzeroSeed(
        options.evalSeed,
        `${profile}\0one-shot\0${realizationIndex}\0${receiverPreset}`
        + `\0attempt\0${attempt}`,
      );
      if (reference?.forbiddenChannelSeeds.has(channelSeed)) continue;
      const measurements = acquireScaleFamily({
        profile,
        binding,
        phaseNativeSample: 0,
        receiverPreset,
        channelSeed,
        buildIdentity,
        storageSampleCount: options.storageSampleCount,
      });
      if (
        measurements.some(
          (measurement) =>
            !hasInformativeCf32lePrefix(
              measurement.bytes,
              PREFIX_LENGTHS[0],
            ),
        )
        || !hasExactUniqueScaleFamilyContentHashes(
          measurements.map(({ contentSha256 }) => contentSha256),
        )
      ) {
        continue;
      }
      const receiverSeeds = measurements
        .map((measurement) =>
          impairmentSeedFromReceipt(measurement.measurement, receiverPreset))
        .filter((seed): seed is number => seed !== null);
      const collides = measurements.some(
        (measurement) =>
          reference?.forbiddenContentHashes.has(
            measurement.contentSha256,
          ) === true
          || evaluationContentHashes.has(measurement.contentSha256),
      ) || receiverSeeds.some(
        (seed) => reference?.forbiddenReceiverSeeds.has(seed) === true,
      );
      if (!collides) accepted = { channelSeed, measurements };
    }
    if (accepted === undefined) {
      throw new Error(
        `${profile} could not produce a new one-shot receiver realization`,
      );
    }
    for (const measurement of accepted.measurements) {
      evaluationContentHashes.add(measurement.contentSha256);
    }
    const pairId =
      `scale-pair:${profile}:one-shot:preset:${receiverPreset}:`
      + `channel-seed:${accepted.channelSeed}`;
    output.push({
      profile,
      publicClass: publicClassForProfile(profile),
      replay: 'one-shot',
      realizationIndex,
      pairId,
      phaseNativeSample: 0,
      receiverPreset,
      receiverChannelSeed: accepted.channelSeed,
      measurements: accepted.measurements,
    });
  }
  return output;
}

function acquireScaleFamily(input: {
  readonly profile: FixedDigitalProfile;
  readonly binding: FixedDigitalProfileBinding;
  readonly phaseNativeSample: number;
  readonly receiverPreset: ReceiverImpairmentPreset;
  readonly channelSeed: number;
  readonly buildIdentity: MeasurementBuildIdentity;
  readonly storageSampleCount: number;
}): DecodedMeasurement[] {
  return SCALE_FACTORS.map((scale) => acquireAtScale({ ...input, scale }));
}

function acquireAtScale(input: {
  readonly profile: FixedDigitalProfile;
  readonly binding: FixedDigitalProfileBinding;
  readonly phaseNativeSample: number;
  readonly receiverPreset: ReceiverImpairmentPreset;
  readonly channelSeed: number;
  readonly buildIdentity: MeasurementBuildIdentity;
  readonly storageSampleCount: number;
  readonly scale: ScaleFactor;
}): DecodedMeasurement {
  const {
    profile,
    binding,
    phaseNativeSample,
    receiverPreset,
    channelSeed,
    buildIdentity,
    storageSampleCount,
    scale,
  } = input;
  const sampleRateHz = scaledSampleRateHz(
    binding.nativeSampleRateHz,
    scale,
  );
  const maximumAvailable = binding.replay === 'one-shot'
    ? maximumOneShotOutputSamples(binding.captureSamples!, scale)
    : storageSampleCount;
  const validSampleCount = Math.min(storageSampleCount, maximumAvailable);
  if (validSampleCount < PREFIX_LENGTHS[0]) {
    throw new Error(
      `${profile} scale ${scale.key} cannot supply the live 4,096 prefix`,
    );
  }
  const deterministic = deterministicServiceDependencies(
    `${profile}\0${phaseNativeSample}\0${receiverPreset}\0${channelSeed}`
    + `\0scale\0${scale.key}`,
  );
  const configurator = new AtomizerMeasurementService(
    buildIdentity,
    deterministic,
  );
  configurator.selectProfile({ profile });
  const channel = {
    model: 'awgn' as const,
    noiseFloorDbm: -108,
    seed: channelSeed,
    fadingRateHz: 2,
    receiverImpairment: receiverPreset,
  };
  const configured = configurator.configureChannel({ channel });
  const divisor = greatestCommonDivisor(
    phaseNativeSample,
    binding.nativeSampleRateHz,
  );
  const resumed = new AtomizerMeasurementService(buildIdentity, {
    ...deterministicServiceDependencies(
      `${profile}\0resume\0${phaseNativeSample}\0${receiverPreset}`
      + `\0${channelSeed}\0scale\0${scale.key}`,
    ),
    continuation: {
      continuationVersion: 2,
      sessionId: configured.sessionId,
      configurationRevision: configured.configurationRevision,
      updatedAt: configured.updatedAt,
      profile,
      channel,
      sequence: 0,
      iqTimeNumerator: String(phaseNativeSample / divisor),
      iqTimeDenominator: String(binding.nativeSampleRateHz / divisor),
    },
  });
  const captureBandwidthHz = productionCaptureBandwidthHz(binding);
  if (captureBandwidthHz > sampleRateHz) {
    throw new Error(
      `${profile} scale ${scale.key} production capture bandwidth `
      + `${captureBandwidthHz} exceeds output sample rate ${sampleRateHz}`,
    );
  }
  const raw = resumed.acquireIq({
    centerHz: binding.profileReferenceCenterHz,
    sampleRateHz,
    captureBandwidthHz,
    sampleCount: validSampleCount,
    sampleFormat: 'cf32le',
  });
  const measurement = complexIqMeasurementSchema.parse(raw);
  const bytes = decodeAndValidatePayload(measurement);
  if (
    measurement.sampleRateHz !== sampleRateHz
    || measurement.nativeSampleRateHz !== binding.nativeSampleRateHz
    || measurement.signalBandwidthHz !== binding.signalBandwidthHz
    || measurement.captureBandwidthHz !== captureBandwidthHz
    || measurement.sampleCount !== validSampleCount
    || measurement.receiverImpairment !== receiverPreset
  ) {
    throw new Error(`${profile} scale ${scale.key} service geometry drifted`);
  }
  if (
    measurement.transformReceipt.outputStartSourceSampleNumerator
      !== String(phaseNativeSample)
    || measurement.transformReceipt.outputStartSourceSampleDenominator !== '1'
  ) {
    throw new Error(
      `${profile} scale ${scale.key} did not preserve source coordinate`,
    );
  }
  if (
    scale.value !== 1
    && !measurement.transformReceipt.operations.some(
      (operation) => operation.kind === 'resample',
    )
  ) {
    throw new Error(
      `${profile} scale ${scale.key} lacks the service resampling receipt`,
    );
  }
  const storedBytes = zeroPadCf32le(bytes, storageSampleCount);
  return {
    measurement,
    bytes,
    storedBytes,
    contentSha256: sha256Bytes(bytes),
    storedSha256: sha256Bytes(storedBytes),
    validSampleCount,
    scale,
  };
}

function readReferenceCorpus(directory: string): ReferenceCorpus {
  const resolved = resolve(directory);
  const manifestPath = resolve(resolved, 'corpus.json');
  if (!existsSync(manifestPath)) {
    throw new Error(`Reference corpus manifest is missing: ${manifestPath}`);
  }
  const parsed = JSON.parse(readFileSync(manifestPath, 'utf8')) as unknown;
  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('Reference corpus manifest must be an object');
  }
  const manifest = parsed as Record<string, unknown>;
  if (manifest.generator !== 'current-signallab-service-corpus-v1') {
    throw new Error(
      'Reference must be the current service-path training corpus',
    );
  }
  const items = manifest.items;
  const count = requireManifestInteger(manifest.count, 'reference count');
  if (!Array.isArray(items) || items.length !== count) {
    throw new Error('Reference corpus count/items disagree');
  }
  const source = manifest.source;
  if (source === null || typeof source !== 'object' || Array.isArray(source)) {
    throw new Error('Reference corpus source must be an object');
  }
  const sourceRecord = source as Record<string, unknown>;
  const sourceCommit = requireSha1(
    sourceRecord.gitCommit,
    'reference source gitCommit',
  );
  const sourceTree = requireSha1(
    sourceRecord.gitTree,
    'reference source gitTree',
  );
  const dataFile = manifest.dataFile;
  if (
    typeof dataFile !== 'string'
    || !dataFile
    || dataFile.includes('/')
    || dataFile.includes('\\')
  ) {
    throw new Error('Reference dataFile must be a plain local filename');
  }
  const rawPath = resolve(resolved, dataFile);
  if (!existsSync(rawPath)) {
    throw new Error(`Reference raw corpus is missing: ${rawPath}`);
  }
  const actualRawSha256 = sha256File(rawPath);
  if (
    typeof manifest.dataSha256 !== 'string'
    || manifest.dataSha256 !== actualRawSha256
  ) {
    throw new Error('Reference raw SHA-256 disagrees with its manifest');
  }
  const forbiddenPhasesMutable = new Map<string, Set<number>>();
  const forbiddenChannelSeeds = new Set<number>();
  const forbiddenReceiverSeeds = new Set<number>();
  const forbiddenContentHashes = new Set<string>();
  for (const [index, value] of items.entries()) {
    if (value === null || typeof value !== 'object' || Array.isArray(value)) {
      throw new Error(`Reference item ${index} must be an object`);
    }
    const item = value as Record<string, unknown>;
    const profile = requireString(item.profile, `reference item ${index} profile`);
    if (item.phaseNativeSample !== undefined) {
      const phase = requireManifestInteger(
        item.phaseNativeSample,
        `reference item ${index} phaseNativeSample`,
        0,
      );
      const phases = forbiddenPhasesMutable.get(profile) ?? new Set<number>();
      phases.add(phase);
      forbiddenPhasesMutable.set(profile, phases);
    }
    addOptionalSeed(
      forbiddenChannelSeeds,
      item.receiverRealizationChannelSeed,
      `reference item ${index} receiverRealizationChannelSeed`,
    );
    addOptionalSeed(
      forbiddenReceiverSeeds,
      item.receiverRealizationSeed,
      `reference item ${index} receiverRealizationSeed`,
    );
    const contentSha256 = item.contentSha256;
    if (
      typeof contentSha256 !== 'string'
      || !HASH_PATTERN.test(contentSha256)
    ) {
      throw new Error(
        `Reference item ${index} contentSha256 is missing or malformed`,
      );
    }
    forbiddenContentHashes.add(contentSha256);
  }
  return {
    directory: resolved,
    manifestPath,
    manifestSha256: sha256File(manifestPath),
    rawSha256: actualRawSha256,
    corpusSeed: requireManifestInteger(
      manifest.corpusSeed,
      'reference corpusSeed',
      0,
    ),
    sourceCommit,
    sourceTree,
    forbiddenPhases: forbiddenPhasesMutable,
    forbiddenChannelSeeds,
    forbiddenReceiverSeeds,
    forbiddenContentHashes,
    count,
  };
}

function addOptionalSeed(
  destination: Set<number>,
  value: unknown,
  field: string,
): void {
  if (value === undefined || value === null) return;
  destination.add(requireManifestInteger(value, field, 0));
}

function zeroPadCf32le(
  validBytes: Uint8Array,
  storageSampleCount: number,
): Uint8Array {
  if (validBytes.byteLength % BYTES_PER_COMPLEX_SAMPLE !== 0) {
    throw new Error('Service CF32 payload is not complex-sample aligned');
  }
  const output = new Uint8Array(
    storageSampleCount * BYTES_PER_COMPLEX_SAMPLE,
  );
  if (validBytes.byteLength > output.byteLength) {
    throw new Error('Service payload exceeds scale-eval storage row');
  }
  output.set(validBytes);
  return output;
}

export function hasInformativeCf32lePrefix(
  bytes: Uint8Array,
  sampleCount: number,
): boolean {
  if (bytes.byteLength < sampleCount * BYTES_PER_COMPLEX_SAMPLE) return false;
  const values = new Float32Array(
    bytes.buffer,
    bytes.byteOffset,
    bytes.byteLength / 4,
  );
  const firstInPhase = values[0]!;
  const firstQuadrature = values[1]!;
  if (!Number.isFinite(firstInPhase) || !Number.isFinite(firstQuadrature)) {
    throw new Error('Service emitted non-finite complex I/Q');
  }
  let nonzero = firstInPhase !== 0 || firstQuadrature !== 0;
  let nonconstant = false;
  let sumInPhase = firstInPhase;
  let sumQuadrature = firstQuadrature;
  for (let sample = 1; sample < sampleCount; sample += 1) {
    const inPhase = values[2 * sample]!;
    const quadrature = values[2 * sample + 1]!;
    if (!Number.isFinite(inPhase) || !Number.isFinite(quadrature)) {
      throw new Error('Service emitted non-finite complex I/Q');
    }
    sumInPhase += inPhase;
    sumQuadrature += quadrature;
    if (inPhase !== 0 || quadrature !== 0) nonzero = true;
    if (inPhase !== firstInPhase || quadrature !== firstQuadrature) {
      nonconstant = true;
    }
  }
  if (!nonzero || !nonconstant) return false;
  const meanInPhase = sumInPhase / sampleCount;
  const meanQuadrature = sumQuadrature / sampleCount;
  let centeredEnergy = 0;
  for (let sample = 0; sample < sampleCount; sample += 1) {
    const deltaInPhase = values[2 * sample]! - meanInPhase;
    const deltaQuadrature = values[2 * sample + 1]! - meanQuadrature;
    centeredEnergy += (
      deltaInPhase * deltaInPhase
      + deltaQuadrature * deltaQuadrature
    );
  }
  const centeredComplexRms = Math.sqrt(centeredEnergy / sampleCount);
  return centeredComplexRms >= INFORMATIVE_PREFIX_CENTERED_RMS_FLOOR;
}

export function hasExactUniqueScaleFamilyContentHashes(
  hashes: readonly string[],
): boolean {
  return hashes.length === SCALE_FACTORS.length
    && hashes.every((value) => HASH_PATTERN.test(value))
    && new Set(hashes).size === hashes.length;
}

function hasInformativeScaleFamilyPrefixes(
  measurements: readonly DecodedMeasurement[],
): boolean {
  return measurements.length === SCALE_FACTORS.length
    && measurements.every(
      (measurement, index) =>
        measurement.scale.key === SCALE_FACTORS[index]!.key
        && hasInformativeCf32lePrefix(
          measurement.bytes,
          PREFIX_LENGTHS[0],
        ),
    );
}

function decodeAndValidatePayload(
  measurement: ComplexIqMeasurement,
): Uint8Array {
  const decoded = Buffer.from(measurement.samplesBase64, 'base64');
  if (decoded.toString('base64') !== measurement.samplesBase64) {
    throw new Error('Service payload is not canonical base64');
  }
  if (
    decoded.byteLength !== measurement.byteLength
    || decoded.byteLength
      !== measurement.sampleCount * BYTES_PER_COMPLEX_SAMPLE
  ) {
    throw new Error('Service payload byte length disagrees with receipt');
  }
  const bytes = Uint8Array.from(decoded);
  const digest = sha256Bytes(bytes);
  if (
    digest !== measurement.samplesSha256
    || digest !== measurement.transformReceipt.outputSamplesSha256
  ) {
    throw new Error('Service payload SHA-256 disagrees with receipt');
  }
  return bytes;
}

function receiptForManifest(
  measurement: ComplexIqMeasurement,
): Record<string, unknown> {
  return {
    measurementId: measurement.measurementId,
    sessionId: measurement.sessionId,
    configurationRevision: measurement.configurationRevision,
    sequence: measurement.sequence,
    capturedAt: measurement.capturedAt,
    complete: measurement.complete,
    provenance: measurement.provenance,
    profileReferenceCenterHz: measurement.profileReferenceCenterHz,
    rfReferenceCenterHz: measurement.rfReferenceCenterHz,
    nativeCarrierOffsetHz: measurement.nativeCarrierOffsetHz,
    outputCarrierOffsetHz: measurement.outputCarrierOffsetHz,
    rfTuneCenterHz: measurement.rfTuneCenterHz,
    sampleRateHz: measurement.sampleRateHz,
    nativeSampleRateHz: measurement.nativeSampleRateHz,
    captureBandwidthHz: measurement.captureBandwidthHz,
    signalBandwidthHz: measurement.signalBandwidthHz,
    sampleCount: measurement.sampleCount,
    byteLength: measurement.byteLength,
    samplesSha256: measurement.samplesSha256,
    qualification: measurement.qualification,
    payloadKind: measurement.payloadKind,
    representation: measurement.representation,
    normalization: measurement.normalization,
    receiverImpairment: measurement.receiverImpairment,
    channelApplication: measurement.channelApplication,
    canonicalArtifactSha256: measurement.canonicalArtifactSha256,
    transformReceipt: measurement.transformReceipt,
  };
}

function impairmentSeedFromReceipt(
  measurement: ComplexIqMeasurement,
  preset: ReceiverImpairmentPreset,
): number | null {
  if (preset === 'clean') return null;
  const operations = measurement.transformReceipt.operations.filter(
    (operation) => operation.kind === 'receiver-impairment',
  );
  if (operations.length !== 1) {
    throw new Error('Non-clean row lacks exactly one impairment receipt');
  }
  const operation = operations[0]!;
  if (
    operation.kind !== 'receiver-impairment'
    || operation.preset !== preset
  ) {
    throw new Error('Receiver impairment receipt preset drifted');
  }
  return operation.seed;
}

function deterministicServiceDependencies(domain: string): {
  readonly uuid: () => string;
  readonly now: () => Date;
  readonly monotonicMilliseconds: () => number;
} {
  let uuidSequence = 0;
  let clockSequence = 0;
  let monotonic = 0;
  return {
    uuid: () => deterministicUuid(`${domain}\0uuid\0${uuidSequence++}`),
    now: () => new Date(
      Date.UTC(2026, 6, 29, 1, 0, 0, clockSequence++),
    ),
    monotonicMilliseconds: () => monotonic++,
  };
}

function deterministicUuid(domain: string): string {
  const characters = sha256Text(domain).slice(0, 32).split('');
  characters[12] = '4';
  characters[16] = '8';
  const compact = characters.join('');
  const uuid = [
    compact.slice(0, 8),
    compact.slice(8, 12),
    compact.slice(12, 16),
    compact.slice(16, 20),
    compact.slice(20),
  ].join('-');
  if (!UUID_PATTERN.test(uuid)) {
    throw new Error('Deterministic UUID construction failed');
  }
  return uuid;
}

function readBuildIdentity(
  signalLabRoot: string,
): MeasurementBuildIdentity {
  const contractPath = resolve(
    signalLabRoot,
    'contracts',
    'signal-lab-measurement-bridge-v2.json',
  );
  const parsed = JSON.parse(readFileSync(contractPath, 'utf8')) as unknown;
  const contractSha256 = sha256Text(JSON.stringify(parsed));
  const generatorContractBindingSha256 = sha256Text(
    `atomizer-in-process-generator\0${contractSha256}`,
  );
  return { contractSha256, generatorContractBindingSha256 };
}

function readGitIdentity(repository: string): GitIdentity {
  const run = (...arguments_: string[]): string =>
    execFileSync('git', ['-C', repository, ...arguments_], {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe'],
    }).trim();
  const commit = run('rev-parse', 'HEAD');
  const tree = run('rev-parse', 'HEAD^{tree}');
  const status = run('status', '--porcelain=v1', '--untracked-files=all');
  requireSha1(commit, 'SignalLab git commit');
  requireSha1(tree, 'SignalLab git tree');
  return { commit, tree, worktreeClean: status.length === 0 };
}

export function generatorLineageBindingSha256(input: {
  readonly sourceSha256: string;
  readonly bundleSha256: string;
}): string {
  if (
    !HASH_PATTERN.test(input.sourceSha256)
    || !HASH_PATTERN.test(input.bundleSha256)
  ) {
    throw new Error('Generator lineage requires lowercase SHA-256 values');
  }
  return sha256Text([
    GENERATOR_LINEAGE_SCHEMA,
    GENERATOR_SOURCE_PATH,
    input.sourceSha256,
    GENERATOR_BUNDLE_PATH,
    input.bundleSha256,
  ].join('\0'));
}

export function readGeneratorLineage(
  classifierRoot: string,
  invokedBundlePath: string | undefined,
): GeneratorLineage {
  const root = resolve(classifierRoot);
  const expectedBundlePath = resolve(root, GENERATOR_BUNDLE_PATH);
  if (
    invokedBundlePath === undefined
    || resolve(invokedBundlePath) !== expectedBundlePath
  ) {
    throw new Error(
      `Generator must execute the canonical reviewed bundle at `
      + GENERATOR_BUNDLE_PATH,
    );
  }
  const sourceSha256 = sha256File(resolve(root, GENERATOR_SOURCE_PATH));
  const bundleSha256 = sha256File(expectedBundlePath);
  return {
    schema: GENERATOR_LINEAGE_SCHEMA,
    repository: 'Atom-Classifier',
    sourcePath: GENERATOR_SOURCE_PATH,
    sourceSha256,
    bundlePath: GENERATOR_BUNDLE_PATH,
    bundleSha256,
    sourceBundleBindingSha256: generatorLineageBindingSha256({
      sourceSha256,
      bundleSha256,
    }),
  };
}

export function assertGeneratorLineageUnchanged(
  expected: GeneratorLineage,
  classifierRoot: string,
  invokedBundlePath: string | undefined,
): void {
  const observed = readGeneratorLineage(classifierRoot, invokedBundlePath);
  const changed = (
    Object.keys(expected) as (keyof GeneratorLineage)[]
  ).filter((key) => expected[key] !== observed[key]);
  if (changed.length > 0) {
    throw new Error(
      `Generator source or bundle changed during scale-eval generation: `
      + changed.join(', '),
    );
  }
}

function prepareOutputs(options: ScaleEvalOptions): {
  readonly rawPart: string;
  readonly manifestPart: string;
  readonly rawFinal: string;
  readonly manifestFinal: string;
} {
  mkdirSync(options.outputDirectory, { recursive: true });
  const rawFinal = resolve(options.outputDirectory, 'scale_eval.f32');
  const manifestFinal = resolve(options.outputDirectory, 'scale_eval.json');
  const rawPart = `${rawFinal}.part`;
  const manifestPart = `${manifestFinal}.part`;
  const paths = [rawPart, manifestPart, rawFinal, manifestFinal];
  const existing = paths.filter(existsSync);
  if (existing.length > 0 && !options.allowOverwrite) {
    throw new Error(
      `Refusing to overwrite scale-eval output: ${existing.join(', ')}`,
    );
  }
  if (options.allowOverwrite) {
    for (const path of existing) rmSync(path, { force: true });
  }
  return { rawPart, manifestPart, rawFinal, manifestFinal };
}

function parseProfileFilter(
  value: string | undefined,
  allProfiles: readonly FixedDigitalProfile[],
): FixedDigitalProfile[] {
  if (value === undefined || value.trim() === '') return [...allProfiles];
  const allowed = new Set<string>(allProfiles);
  const requested = value.split(',').map((item) => item.trim());
  if (
    requested.some((item) => !item || !allowed.has(item))
    || new Set(requested).size !== requested.length
  ) {
    throw new Error(
      'PROFILE_FILTER must be a comma-separated set of unique fixed profiles',
    );
  }
  return requested.sort() as FixedDigitalProfile[];
}

function parseEnvironmentInteger(
  value: string | undefined,
  fallback: number,
  minimum: number,
  maximum: number,
  label: string,
): number {
  if (value === undefined) return fallback;
  if (!/^(?:0|[1-9][0-9]*)$/.test(value)) {
    throw new Error(`${label} must be a canonical non-negative integer`);
  }
  const parsed = Number(value);
  requireSafeInteger(parsed, minimum, maximum, label);
  return parsed;
}

function parseBooleanFlag(
  value: string | undefined,
  label: string,
): boolean {
  if (value === undefined || value === '0') return false;
  if (value === '1') return true;
  throw new Error(`${label} must be exactly 0 or 1`);
}

function requireManifestInteger(
  value: unknown,
  label: string,
  minimum = 1,
): number {
  if (
    typeof value !== 'number'
    || !Number.isSafeInteger(value)
    || value < minimum
  ) {
    throw new Error(`${label} must be a safe integer >= ${minimum}`);
  }
  return value;
}

function requireSafeInteger(
  value: number,
  minimum: number,
  maximum: number,
  label: string,
): void {
  if (
    !Number.isSafeInteger(value)
    || value < minimum
    || value > maximum
  ) {
    throw new RangeError(
      `${label} must be a safe integer from ${minimum} through ${maximum}`,
    );
  }
}

function requireString(value: unknown, label: string): string {
  if (typeof value !== 'string' || !value) {
    throw new Error(`${label} must be a non-empty string`);
  }
  return value;
}

function requireSha1(value: unknown, label: string): string {
  if (typeof value !== 'string' || !/^[a-f0-9]{40}$/.test(value)) {
    throw new Error(`${label} is not a lowercase SHA-1`);
  }
  return value;
}

function deterministicNonzeroSeed(
  evalSeed: number,
  domain: string,
): number {
  const seed = hashUint32(`${evalSeed}\0scale-eval-channel\0${domain}`);
  return seed === 0 ? 1 : seed;
}

function hashUint32(value: string): number {
  return createHash('sha256')
    .update(value, 'utf8')
    .digest()
    .readUInt32LE(0);
}

function greatestCommonDivisor(left: number, right: number): number {
  let a = Math.abs(left);
  let b = Math.abs(right);
  while (b !== 0) [a, b] = [b, a % b];
  return a;
}

function sha256Bytes(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex');
}

function sha256Text(value: string): string {
  return createHash('sha256').update(value, 'utf8').digest('hex');
}

function sha256File(path: string): string {
  const descriptor = openSync(path, 'r');
  const digest = createHash('sha256');
  const buffer = Buffer.allocUnsafe(1 << 20);
  try {
    let offset = 0;
    const size = statSync(path).size;
    while (offset < size) {
      const count = readSync(
        descriptor,
        buffer,
        0,
        Math.min(buffer.byteLength, size - offset),
        offset,
      );
      if (count <= 0) throw new Error(`No read progress hashing ${path}`);
      digest.update(buffer.subarray(0, count));
      offset += count;
    }
  } finally {
    closeSync(descriptor);
  }
  return digest.digest('hex');
}

function writeAll(descriptor: number, bytes: Uint8Array): void {
  let offset = 0;
  while (offset < bytes.byteLength) {
    const written = writeSync(
      descriptor,
      bytes,
      offset,
      bytes.byteLength - offset,
    );
    if (written <= 0) throw new Error('Scale-eval stream made no progress');
    offset += written;
  }
}

function closeQuietly(descriptor: number): void {
  try {
    closeSync(descriptor);
  } catch {
    // Preserve the primary generation exception.
  }
}

const invokedPath = process.argv[1] === undefined
  ? undefined
  : resolve(process.argv[1]);
if (
  invokedPath !== undefined
  && resolve(fileURLToPath(import.meta.url)) === invokedPath
) {
  generateCurrentSignalLabScaleEval(readOptions());
}
