/**
 * Build the current fixed-profile classifier corpus through SignalLab's public
 * measurement service.  This intentionally does not import or call any
 * low-level waveform synthesizer.
 *
 * From the Atom-Classifier repository root, with the repository's pinned
 * Node 22 toolchain:
 *
 *   PATH=/Users/johnelliott/.nvm/versions/node/v22.23.1/bin:$PATH \
 *     npx tsup tools/generate-current-signallab-service-corpus.ts \
 *       --format esm --platform node --out-dir .artifacts/current-corpus-generator \
 *       --no-splitting --treeshake --silent
 *
 *   PATH=/Users/johnelliott/.nvm/versions/node/v22.23.1/bin:$PATH \
 *     OUTPUT_DIR=training/artifacts/signallab-current-94171ba-seed20260729 \
 *     CORPUS_SEED=20260729 TARGET_PER_PROFILE=288 SAMPLE_COUNT=16384 \
 *     node .artifacts/current-corpus-generator/generate-current-signallab-service-corpus.js
 *
 * TARGET_PER_PROFILE is the final stored row count for each of all 31 fixed
 * profiles and must be a positive multiple of the nine receiver presets.
 * Existing output is never replaced unless ALLOW_OVERWRITE=1 is explicit.
 */

import { createHash } from 'node:crypto';
import {
  closeSync,
  existsSync,
  mkdirSync,
  openSync,
  readFileSync,
  renameSync,
  rmSync,
  writeFileSync,
  writeSync,
} from 'node:fs';
import { execFileSync } from 'node:child_process';
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
  complexIqMeasurementSchema,
  type ComplexIqMeasurement,
} from '../../Atom-SignalLab/src/measurement-contract.js';
import {
  AtomizerMeasurementService,
  type MeasurementBuildIdentity,
} from '../../Atom-SignalLab/src/measurement-service.js';

export const RECEIVER_PRESETS = Object.freeze([
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

export const SPLIT_ROLES = Object.freeze([
  'train',
  'enrollment',
  'selection',
] as const);

export const ONE_SHOT_DETERMINISTIC_PRESETS = Object.freeze([
  'multipath',
  'carrier-offset',
  'iq-imbalance',
  'dc-offset',
  'pa-compression',
] as const satisfies readonly ReceiverImpairmentPreset[]);

export const ONE_SHOT_STOCHASTIC_PRESETS = Object.freeze([
  'awgn',
  'phase-noise',
  'composite',
] as const satisfies readonly ReceiverImpairmentPreset[]);

// tsup statically bundles the sibling SignalLab source.  Pin its accepted Git
// identity so a stale bundle can never claim provenance from a newer checkout,
// and a freshly compiled bundle from a different checkout cannot silently
// retain this corpus contract.
export const EXPECTED_SIGNAL_LAB_GIT_COMMIT =
  '94171baf31bfa62150d4e9684e1fe33e5a343dc0';
export const EXPECTED_SIGNAL_LAB_GIT_TREE =
  'ccda4401748953bb6705776694b29b571d4c684d';
export const GENERATOR_LINEAGE_SCHEMA =
  'atom-classifier-generator-lineage-v1' as const;
export const GENERATOR_SOURCE_PATH =
  'tools/generate-current-signallab-service-corpus.ts' as const;
export const GENERATOR_BUNDLE_PATH =
  '.artifacts/current-corpus-generator/'
  + 'generate-current-signallab-service-corpus.js' as const;

export type SplitRole = typeof SPLIT_ROLES[number];
export type PublicClass = 'gsm' | 'ofdm' | 'dsss' | 'bluetooth';

export interface OneShotRealizationPlan {
  readonly receiverRealizationIndex: number;
  readonly role: SplitRole;
  readonly preset: ReceiverImpairmentPreset;
}

export interface GeneratorOptions {
  readonly classifierRoot: string;
  readonly generatorBundlePath: string;
  readonly outputDirectory: string;
  readonly corpusSeed: number;
  readonly targetPerProfile: number;
  readonly sampleCount: number;
  readonly allowOverwrite: boolean;
  readonly signalLabRoot: string;
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

interface DecodedMeasurement {
  readonly measurement: ComplexIqMeasurement;
  readonly bytes: Uint8Array;
}

interface MaterializedRow {
  readonly bytes: Uint8Array;
  readonly cleanBytes: Uint8Array;
  readonly item: Record<string, unknown>;
  readonly contentSha256: string;
  readonly role: SplitRole;
}

interface OpenOutputs {
  readonly mainDescriptor: number;
  readonly cleanDescriptor: number;
  readonly mainPart: string;
  readonly cleanPart: string;
  readonly manifestPart: string;
  readonly mainFinal: string;
  readonly cleanFinal: string;
  readonly manifestFinal: string;
}

const DEFAULT_SAMPLE_COUNT = 16_384;
const DEFAULT_TARGET_PER_PROFILE = 288;
const DEFAULT_CORPUS_SEED = 20_260_729;
const MIN_LIVE_PREFIX_SAMPLES = 4_096;
const MAX_SERVICE_SAMPLES = 65_536;
const CF32LE_BYTES_PER_SAMPLE = 8;
const ALL_ZERO_PREFIX_FLOATS = MIN_LIVE_PREFIX_SAMPLES * 2;
const HASH_PATTERN = /^[a-f0-9]{64}$/;
const UUID_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

export function publicClassForProfile(profile: FixedDigitalProfile): PublicClass {
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

export function splitCounts(unitCount: number): Record<SplitRole, number> {
  requireSafeInteger(unitCount, 1, Number.MAX_SAFE_INTEGER, 'split unit count');
  if (unitCount === 1) {
    return { train: 1, enrollment: 0, selection: 0 };
  }
  if (unitCount === 2) {
    return { train: 1, enrollment: 1, selection: 0 };
  }
  let train = Math.round(unitCount * 0.6);
  let enrollment = Math.round(unitCount * 0.2);
  let selection = unitCount - train - enrollment;
  train = Math.max(1, train);
  enrollment = Math.max(1, enrollment);
  selection = Math.max(1, selection);
  while (train + enrollment + selection > unitCount) {
    if (train > enrollment && train > 1) train -= 1;
    else if (selection > 1) selection -= 1;
    else enrollment -= 1;
  }
  while (train + enrollment + selection < unitCount) train += 1;
  return { train, enrollment, selection };
}

export function deterministicRoleSchedule(
  unitCount: number,
  corpusSeed: number,
  domain: string,
): SplitRole[] {
  const counts = splitCounts(unitCount);
  const roles: SplitRole[] = [];
  for (const role of SPLIT_ROLES) {
    for (let index = 0; index < counts[role]; index += 1) roles.push(role);
  }
  const random = mulberry32(hashUint32(`${corpusSeed}\0roles\0${domain}`));
  for (let index = roles.length - 1; index > 0; index -= 1) {
    const replacement = Math.floor(random() * (index + 1));
    [roles[index], roles[replacement]] = [
      roles[replacement]!,
      roles[index]!,
    ];
  }
  return roles;
}

export function oneShotRealizationSchedule(
  targetPerProfile: number,
  corpusSeed: number,
  profile: string,
): OneShotRealizationPlan[] {
  requireSafeInteger(
    targetPerProfile,
    RECEIVER_PRESETS.length,
    Number.MAX_SAFE_INTEGER,
    'one-shot target per profile',
  );
  if (targetPerProfile % RECEIVER_PRESETS.length !== 0) {
    throw new Error('one-shot target per profile must be a multiple of nine');
  }
  const remaining = splitCounts(targetPerProfile);
  if (remaining.enrollment < 1) {
    throw new Error('one-shot schedule needs enrollment for its sole clean row');
  }
  const planned: Array<{
    role: SplitRole;
    preset: ReceiverImpairmentPreset;
  }> = [];
  planned.push({ role: 'enrollment', preset: 'clean' });
  remaining.enrollment -= 1;

  for (const preset of ONE_SHOT_DETERMINISTIC_PRESETS) {
    const preferred = (['train', 'enrollment'] as const)
      .filter((role) => remaining[role] > 0)
      .sort((left, right) => {
        const countDifference = remaining[right] - remaining[left];
        if (countDifference !== 0) return countDifference;
        return left.localeCompare(right);
      });
    const role = preferred[0]
      ?? SPLIT_ROLES.find((candidate) => remaining[candidate] > 0);
    if (role === undefined) {
      throw new Error('one-shot deterministic preset schedule exhausted roles');
    }
    planned.push({ role, preset });
    remaining[role] -= 1;
  }

  const stochasticOffset =
    hashUint32(`${corpusSeed}\0one-shot-stochastic\0${profile}`)
    % ONE_SHOT_STOCHASTIC_PRESETS.length;
  let stochasticIndex = 0;
  for (const role of SPLIT_ROLES) {
    while (remaining[role] > 0) {
      const preset = ONE_SHOT_STOCHASTIC_PRESETS[
        (stochasticOffset + stochasticIndex)
        % ONE_SHOT_STOCHASTIC_PRESETS.length
      ]!;
      planned.push({ role, preset });
      remaining[role] -= 1;
      stochasticIndex += 1;
    }
  }
  if (planned.length !== targetPerProfile) {
    throw new Error('one-shot realization plan has the wrong row count');
  }

  // Row order must not become an acquisition-order shortcut.  Preserve the
  // sole clean reference at row zero for an obvious audit trail, but
  // deterministically permute every other receiver realization.
  const tail = planned.slice(1);
  const random = mulberry32(
    hashUint32(`${corpusSeed}\0one-shot-order\0${profile}`),
  );
  for (let index = tail.length - 1; index > 0; index -= 1) {
    const replacement = Math.floor(random() * (index + 1));
    [tail[index], tail[replacement]] = [
      tail[replacement]!,
      tail[index]!,
    ];
  }
  return [planned[0]!, ...tail].map((row, receiverRealizationIndex) => ({
    ...row,
    receiverRealizationIndex,
  }));
}

export function deterministicPhaseGrid(
  periodSamples: number,
  groupCount: number,
  corpusSeed: number,
  profile: string,
): number[] {
  requireSafeInteger(periodSamples, 1, Number.MAX_SAFE_INTEGER, 'periodSamples');
  requireSafeInteger(groupCount, 1, periodSamples, 'phase group count');
  const offset =
    hashUint32(`${corpusSeed}\0phase-offset\0${profile}`) % periodSamples;
  let stride =
    1 + (hashUint32(`${corpusSeed}\0phase-stride\0${profile}`)
      % Math.max(1, periodSamples - 1));
  while (greatestCommonDivisor(stride, periodSamples) !== 1) {
    stride = stride === periodSamples - 1 ? 1 : stride + 1;
  }
  return Array.from(
    { length: groupCount },
    (_, index) => (offset + index * stride) % periodSamples,
  );
}

export function paddedCf32le(
  validBytes: Uint8Array,
  storageSampleCount: number,
): Uint8Array {
  requireSafeInteger(
    storageSampleCount,
    MIN_LIVE_PREFIX_SAMPLES,
    MAX_SERVICE_SAMPLES,
    'storage sample count',
  );
  if (validBytes.byteLength % CF32LE_BYTES_PER_SAMPLE !== 0) {
    throw new Error('Valid cf32le payload byte length is not sample aligned');
  }
  const storageBytes = storageSampleCount * CF32LE_BYTES_PER_SAMPLE;
  if (validBytes.byteLength > storageBytes) {
    throw new Error('Valid cf32le payload exceeds its storage row');
  }
  const output = new Uint8Array(storageBytes);
  output.set(validBytes);
  return output;
}

export function hasOccupiedLivePrefix(bytes: Uint8Array): boolean {
  if (bytes.byteLength < MIN_LIVE_PREFIX_SAMPLES * CF32LE_BYTES_PER_SAMPLE) {
    return false;
  }
  const floats = new Float32Array(
    bytes.buffer,
    bytes.byteOffset,
    bytes.byteLength / 4,
  );
  for (let index = 0; index < ALL_ZERO_PREFIX_FLOATS; index += 1) {
    const value = floats[index]!;
    if (!Number.isFinite(value)) {
      throw new Error('Service emitted a non-finite cf32le value');
    }
    if (value !== 0) return true;
  }
  return false;
}

export function readOptions(environment = process.env): GeneratorOptions {
  const corpusSeed = parseEnvironmentInteger(
    environment.CORPUS_SEED,
    DEFAULT_CORPUS_SEED,
    0,
    Number.MAX_SAFE_INTEGER,
    'CORPUS_SEED',
  );
  const targetPerProfile = parseEnvironmentInteger(
    environment.TARGET_PER_PROFILE,
    DEFAULT_TARGET_PER_PROFILE,
    RECEIVER_PRESETS.length,
    Number.MAX_SAFE_INTEGER,
    'TARGET_PER_PROFILE',
  );
  if (targetPerProfile % RECEIVER_PRESETS.length !== 0) {
    throw new Error(
      `TARGET_PER_PROFILE must be a positive multiple of `
      + `${RECEIVER_PRESETS.length}`,
    );
  }
  const sampleCount = parseEnvironmentInteger(
    environment.SAMPLE_COUNT,
    DEFAULT_SAMPLE_COUNT,
    MIN_LIVE_PREFIX_SAMPLES,
    MAX_SERVICE_SAMPLES,
    'SAMPLE_COUNT',
  );
  const allowOverwrite = parseBooleanFlag(
    environment.ALLOW_OVERWRITE,
    'ALLOW_OVERWRITE',
  );
  const classifierRoot = process.cwd();
  const packageDocument = JSON.parse(
    readFileSync(resolve(classifierRoot, 'package.json'), 'utf8'),
  ) as { readonly name?: unknown };
  if (packageDocument.name !== 'atomos-classifier') {
    throw new Error(
      'Run the bundled current-corpus generator from the Atom-Classifier '
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
  const defaultOutput = resolve(
    classifierRoot,
    'training',
    'artifacts',
    `signallab-current-seed${corpusSeed}-n${targetPerProfile}-s${sampleCount}`,
  );
  return {
    classifierRoot,
    generatorBundlePath: process.argv[1] === undefined
      ? ''
      : resolve(process.argv[1]),
    outputDirectory: resolve(environment.OUTPUT_DIR ?? defaultOutput),
    corpusSeed,
    targetPerProfile,
    sampleCount,
    allowOverwrite,
    signalLabRoot,
  };
}

export function generateCurrentSignalLabCorpus(options: GeneratorOptions): void {
  const generatorLineage = readGeneratorLineage(
    options.classifierRoot,
    options.generatorBundlePath,
  );
  validateReceiverPresetContract();
  const profiles = (
    Object.keys(FIXED_DIGITAL_PROFILE_BINDINGS) as FixedDigitalProfile[]
  ).sort();
  if (profiles.length !== 31) {
    throw new Error(`Expected 31 fixed profiles, found ${profiles.length}`);
  }
  for (const profile of profiles) publicClassForProfile(profile);

  const sourceGit = readGitIdentity(options.signalLabRoot);
  if (!sourceGit.worktreeClean) {
    throw new Error(
      `SignalLab source worktree is dirty; refusing an unidentifiable corpus: `
      + options.signalLabRoot,
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
      + 'generator pins before producing another corpus.',
    );
  }
  const buildIdentity = readBuildIdentity(options.signalLabRoot);
  const outputs = openOutputs(options);
  let outputClosed = false;
  let completed = false;
  const mainHash = createHash('sha256');
  const cleanHash = createHash('sha256');
  const items: Record<string, unknown>[] = [];
  const contentRoleOwners = new Map<string, SplitRole>();
  const classCounts = new Map<PublicClass, number>();
  const profileRoleCounts: Record<string, Record<SplitRole, number>> = {};

  try {
    for (const profile of profiles) {
      const binding = FIXED_DIGITAL_PROFILE_BINDINGS[profile];
      const publicClass = publicClassForProfile(profile);
      profileRoleCounts[profile] = {
        train: 0,
        enrollment: 0,
        selection: 0,
      };
      const rows = binding.replay === 'cyclic'
        ? materializeCyclicProfile({
            profile,
            binding,
            publicClass,
            options,
            buildIdentity,
            contentRoleOwners,
          })
        : materializeOneShotProfile({
            profile,
            binding,
            publicClass,
            options,
            buildIdentity,
            contentRoleOwners,
          });
      if (rows.length !== options.targetPerProfile) {
        throw new Error(
          `${profile} materialized ${rows.length} rows, expected `
          + `${options.targetPerProfile}`,
        );
      }
      for (const row of rows) {
        const index = items.length;
        const item = { index, ...row.item };
        writeAll(outputs.mainDescriptor, row.bytes);
        writeAll(outputs.cleanDescriptor, row.cleanBytes);
        mainHash.update(row.bytes);
        cleanHash.update(row.cleanBytes);
        items.push(item);
        contentRoleOwners.set(row.contentSha256, row.role);
        classCounts.set(publicClass, (classCounts.get(publicClass) ?? 0) + 1);
        profileRoleCounts[profile]![row.role] += 1;
      }
      process.stdout.write(
        `[current-corpus] ${profile}: ${rows.length} rows\n`,
      );
    }

    closeSync(outputs.mainDescriptor);
    closeSync(outputs.cleanDescriptor);
    outputClosed = true;
    const dataSha256 = mainHash.digest('hex');
    const cleanDataSha256 = cleanHash.digest('hex');
    assertGeneratorLineageUnchanged(
      generatorLineage,
      options.classifierRoot,
      options.generatorBundlePath,
    );
    const manifest = {
      schemaVersion: 1,
      generator: 'current-signallab-service-corpus-v1',
      generatorPath: 'tools/generate-current-signallab-service-corpus.ts',
      generatorLineage,
      servicePath: [
        'AtomizerMeasurementService.selectProfile',
        'AtomizerMeasurementService.configureChannel',
        'MeasurementServiceContinuation(v2)',
        'AtomizerMeasurementService.acquireIq',
        'complexIqMeasurementSchema.parse',
      ],
      lowLevelSynthesizerCalls: false,
      corpusSeed: options.corpusSeed,
      targetPerProfile: options.targetPerProfile,
      sampleCount: options.sampleCount,
      format: 'cf32le-interleaved',
      dataFile: 'corpus.f32',
      cleanDataFile: 'corpus_clean.f32',
      dataSha256,
      cleanDataSha256,
      hasCleanPairs: true,
      cleanPairPolicy: {
        modelEvidence: false,
        purpose: 'audit-only counterfactual receiver reference',
        oneShotCleanReference:
          'one origin-zero clean reference may repeat only in audit-only '
          + 'corpus_clean.f32; corpus.f32 has one clean receiver realization '
          + 'per one-shot profile and never crosses split roles',
      },
      splitContract: {
        targetRatios: { train: 0.6, enrollment: 0.2, selection: 0.2 },
        cyclic:
          'unique occupied native phase group assigned before all nine '
          + 'receiver presets; one baseRowId and one splitRole per phase group',
        oneShot:
          'origin-zero waveform is not claimed held out; non-clean seeded '
          + 'receiver realizations are split as held_receiver_realization; '
          + 'only one clean row exists per profile',
        exactValidContentSha256CannotCrossRoles: true,
        tinyConfiguration:
          'TARGET_PER_PROFILE=9 is admitted for smoke generation; fewer than '
          + 'three cyclic phase groups cannot populate every role for a '
          + 'single-profile class, so use at least 27 for training and 45 or '
          + 'more for an approximately 60/20/20 per-profile split',
      },
      runtimeInputBuckets: [4_096, 8_192, 16_384],
      receiverPresets: RECEIVER_PRESETS,
      source: {
        repository: 'Atom-SignalLab',
        gitCommit: sourceGit.commit,
        gitTree: sourceGit.tree,
        worktreeClean: sourceGit.worktreeClean,
        contractSha256: buildIdentity.contractSha256,
        generatorContractBindingSha256:
          buildIdentity.generatorContractBindingSha256,
      },
      classes: [...new Set(profiles.map(publicClassForProfile))].sort(),
      profiles,
      profilePublicClassMap: Object.fromEntries(
        profiles.map((profile) => [profile, publicClassForProfile(profile)]),
      ),
      count: items.length,
      perClass: Object.fromEntries(
        [...classCounts.entries()].sort(([left], [right]) =>
          left.localeCompare(right)),
      ),
      perProfileRole: profileRoleCounts,
      items,
    };
    writeFileSync(
      outputs.manifestPart,
      `${JSON.stringify(manifest, null, 2)}\n`,
      { encoding: 'utf8', flag: 'wx' },
    );
    renameSync(outputs.mainPart, outputs.mainFinal);
    renameSync(outputs.cleanPart, outputs.cleanFinal);
    renameSync(outputs.manifestPart, outputs.manifestFinal);
    completed = true;
    process.stdout.write(
      `[current-corpus] wrote ${items.length} rows to `
      + `${options.outputDirectory}\n`,
    );
    process.stdout.write(
      `[current-corpus] corpus.f32 sha256 ${dataSha256}\n`,
    );
  } finally {
    if (!outputClosed) {
      closeQuietly(outputs.mainDescriptor);
      closeQuietly(outputs.cleanDescriptor);
    }
    if (!completed) cleanupInvocationOutputs(outputs);
  }
}

function materializeCyclicProfile(input: {
  readonly profile: FixedDigitalProfile;
  readonly binding: FixedDigitalProfileBinding;
  readonly publicClass: PublicClass;
  readonly options: GeneratorOptions;
  readonly buildIdentity: MeasurementBuildIdentity;
  readonly contentRoleOwners: ReadonlyMap<string, SplitRole>;
}): MaterializedRow[] {
  const {
    profile,
    binding,
    publicClass,
    options,
    buildIdentity,
    contentRoleOwners,
  } = input;
  if (binding.replay !== 'cyclic' || binding.nativePeriodSamples === undefined) {
    throw new Error(`${profile} is not a cyclic fixed binding`);
  }
  const groupCount = options.targetPerProfile / RECEIVER_PRESETS.length;
  if (groupCount > binding.nativePeriodSamples) {
    throw new Error(
      `${profile} requests ${groupCount} phase groups but its period is only `
      + `${binding.nativePeriodSamples} samples`,
    );
  }
  const desiredRoles = deterministicRoleSchedule(
    groupCount,
    options.corpusSeed,
    `cyclic\0${profile}`,
  );
  const phaseCandidates = deterministicPhaseGrid(
    binding.nativePeriodSamples,
    binding.nativePeriodSamples,
    options.corpusSeed,
    profile,
  );
  const rows: MaterializedRow[] = [];
  const acceptedPhases = new Set<number>();
  const localOwners = new Map<string, SplitRole>();
  let candidateIndex = 0;

  for (let groupIndex = 0; groupIndex < groupCount; groupIndex += 1) {
    const role = desiredRoles[groupIndex]!;
    let phase: number | undefined;
    let clean: DecodedMeasurement | undefined;
    while (candidateIndex < phaseCandidates.length) {
      const candidate = phaseCandidates[candidateIndex++]!;
      if (acceptedPhases.has(candidate)) continue;
      const candidateClean = acquireAtPhase({
        profile,
        binding,
        phaseNativeSample: candidate,
        preset: 'clean',
        channelSeed: deterministicChannelSeed(
          options.corpusSeed,
          `${profile}\0phase\0${candidate}\0clean`,
        ),
        buildIdentity,
        sampleCount: options.sampleCount,
      });
      if (!hasOccupiedLivePrefix(candidateClean.bytes)) continue;
      const validHash = sha256(candidateClean.bytes);
      const owner =
        localOwners.get(validHash) ?? contentRoleOwners.get(validHash);
      if (owner !== undefined && owner !== role) continue;
      phase = candidate;
      clean = candidateClean;
      break;
    }
    if (phase === undefined || clean === undefined) {
      throw new Error(
        `${profile} exhausted its exact native period before finding `
        + `${groupCount} unique occupied, split-safe phase groups`,
      );
    }

    const groupRows: MaterializedRow[] = [];
    for (const [presetIndex, preset] of RECEIVER_PRESETS.entries()) {
      let seedAttempt = 0;
      let measured: DecodedMeasurement;
      let validHash: string;
      do {
        const channelSeed = deterministicChannelSeed(
          options.corpusSeed,
          `${profile}\0phase\0${phase}\0${preset}\0${seedAttempt}`,
        );
        measured = preset === 'clean' && seedAttempt === 0
          ? clean
          : acquireAtPhase({
              profile,
              binding,
              phaseNativeSample: phase,
              preset,
              channelSeed,
              buildIdentity,
              sampleCount: options.sampleCount,
            });
        validHash = sha256(measured.bytes);
        const owner =
          localOwners.get(validHash) ?? contentRoleOwners.get(validHash);
        if (owner === undefined || owner === role) break;
        seedAttempt += 1;
        if (preset === 'clean' || seedAttempt >= 64) {
          throw new Error(
            `${profile}/${phase}/${preset} content collides across split roles`,
          );
        }
      } while (true);
      const validSampleCount = measured.measurement.sampleCount;
      const stored = paddedCf32le(measured.bytes, options.sampleCount);
      const cleanStored = paddedCf32le(clean.bytes, options.sampleCount);
      const zeroPadded = validSampleCount < options.sampleCount;
      const baseRowId = `cyclic:${profile}:phase:${phase}`;
      groupRows.push({
        bytes: stored,
        cleanBytes: cleanStored,
        contentSha256: validHash,
        role,
        item: {
          cls: publicClass,
          profile,
          profileId: profile,
          replay: 'cyclic',
          splitRole: role,
          splitAxis: 'held_native_phase_group',
          heldOutAxis: 'held_waveform_phase',
          phaseGroupIndex: groupIndex,
          phaseNativeSample: phase,
          baseRowId,
          splitUnitId: baseRowId,
          receiverPreset: preset,
          receiverPresetIndex: presetIndex,
          receiverRealizationSeed:
            impairmentSeedFromReceipt(measured.measurement, preset),
          validSampleCount,
          storageSampleCount: options.sampleCount,
          zeroPaddedAfterValidSampleCount: zeroPadded,
          contentSha256: validHash,
          storedSha256: sha256(stored),
          cleanContentSha256: sha256(clean.bytes),
          cleanStoredSha256: sha256(cleanStored),
          sampleRateHz: measured.measurement.sampleRateHz,
          signalBandwidthHz: measured.measurement.signalBandwidthHz,
          captureBandwidthHz: measured.measurement.captureBandwidthHz,
          measurementReceipt: receiptForManifest(measured.measurement),
          cleanMeasurementReceipt: receiptForManifest(clean.measurement),
        },
      });
    }
    for (const row of groupRows) {
      localOwners.set(row.contentSha256, role);
      rows.push(row);
    }
    acceptedPhases.add(phase);
  }
  return rows;
}

function materializeOneShotProfile(input: {
  readonly profile: FixedDigitalProfile;
  readonly binding: FixedDigitalProfileBinding;
  readonly publicClass: PublicClass;
  readonly options: GeneratorOptions;
  readonly buildIdentity: MeasurementBuildIdentity;
  readonly contentRoleOwners: ReadonlyMap<string, SplitRole>;
}): MaterializedRow[] {
  const {
    profile,
    binding,
    publicClass,
    options,
    buildIdentity,
    contentRoleOwners,
  } = input;
  if (binding.replay !== 'one-shot' || binding.captureSamples === undefined) {
    throw new Error(`${profile} is not a one-shot fixed binding`);
  }
  const validSampleCount = Math.min(
    options.sampleCount,
    binding.captureSamples,
  );
  const clean = acquireAtPhase({
    profile,
    binding,
    phaseNativeSample: 0,
    preset: 'clean',
    channelSeed: deterministicChannelSeed(
      options.corpusSeed,
      `${profile}\0one-shot\0clean`,
    ),
    buildIdentity,
    sampleCount: validSampleCount,
  });
  if (!hasOccupiedLivePrefix(clean.bytes)) {
    throw new Error(`${profile} origin-zero one-shot live prefix is all zero`);
  }
  const cleanHash = sha256(clean.bytes);
  const existingCleanOwner = contentRoleOwners.get(cleanHash);
  if (
    existingCleanOwner !== undefined
    && existingCleanOwner !== 'enrollment'
  ) {
    throw new Error(
      `${profile} clean origin collides with ${existingCleanOwner} content`,
    );
  }

  const realizationPlan = oneShotRealizationSchedule(
    options.targetPerProfile,
    options.corpusSeed,
    profile,
  );
  const rows: MaterializedRow[] = [];
  const localOwners = new Map<string, SplitRole>();
  for (const planned of realizationPlan) {
    const rowIndex = planned.receiverRealizationIndex;
    const { role, preset } = planned;
    let seedAttempt = 0;
    let measured: DecodedMeasurement;
    let validHash: string;
    let channelSeed: number;
    do {
      channelSeed = deterministicChannelSeed(
        options.corpusSeed,
        `${profile}\0one-shot\0${rowIndex}\0${preset}\0${seedAttempt}`,
      );
      measured = preset === 'clean'
        ? clean
        : acquireAtPhase({
            profile,
            binding,
            phaseNativeSample: 0,
            preset,
            channelSeed,
            buildIdentity,
            sampleCount: validSampleCount,
          });
      validHash = sha256(measured.bytes);
      const owner =
        localOwners.get(validHash) ?? contentRoleOwners.get(validHash);
      if (owner === undefined || owner === role) break;
      seedAttempt += 1;
      if (preset === 'clean' || seedAttempt >= 64) {
        throw new Error(
          `${profile}/${rowIndex}/${preset} content collides across roles`,
        );
      }
    } while (true);
    const stored = paddedCf32le(measured.bytes, options.sampleCount);
    const cleanStored = paddedCf32le(clean.bytes, options.sampleCount);
    const baseRowId =
      `one-shot:${profile}:receiver:${rowIndex}:preset:${preset}:seed:`
      + `${channelSeed}`;
    rows.push({
      bytes: stored,
      cleanBytes: cleanStored,
      contentSha256: validHash,
      role,
      item: {
        cls: publicClass,
        profile,
        profileId: profile,
        replay: 'one-shot',
        splitRole: role,
        splitAxis: 'held_receiver_realization',
        heldOutAxis: 'held_receiver_realization',
        heldWaveform: false,
        phaseNativeSample: 0,
        baseRowId,
        splitUnitId: baseRowId,
        receiverPreset: preset,
        receiverRealizationIndex: rowIndex,
        receiverRealizationChannelSeed: channelSeed,
        receiverRealizationSeed:
          impairmentSeedFromReceipt(measured.measurement, preset),
        cleanRowForProfile: preset === 'clean',
        cleanPairAuditOnly: true,
        validSampleCount,
        storageSampleCount: options.sampleCount,
        zeroPaddedAfterValidSampleCount:
          validSampleCount < options.sampleCount,
        contentSha256: validHash,
        storedSha256: sha256(stored),
        cleanContentSha256: cleanHash,
        cleanStoredSha256: sha256(cleanStored),
        sampleRateHz: measured.measurement.sampleRateHz,
        signalBandwidthHz: measured.measurement.signalBandwidthHz,
        captureBandwidthHz: measured.measurement.captureBandwidthHz,
        measurementReceipt: receiptForManifest(measured.measurement),
        cleanMeasurementReceipt: receiptForManifest(clean.measurement),
      },
    });
    localOwners.set(validHash, role);
  }
  return rows;
}

function acquireAtPhase(input: {
  readonly profile: FixedDigitalProfile;
  readonly binding: FixedDigitalProfileBinding;
  readonly phaseNativeSample: number;
  readonly preset: ReceiverImpairmentPreset;
  readonly channelSeed: number;
  readonly buildIdentity: MeasurementBuildIdentity;
  readonly sampleCount?: number;
}): DecodedMeasurement {
  const {
    profile,
    binding,
    phaseNativeSample,
    preset,
    channelSeed,
    buildIdentity,
  } = input;
  const sampleCount = input.sampleCount ?? (
    binding.replay === 'one-shot'
      ? Math.min(DEFAULT_SAMPLE_COUNT, binding.captureSamples!)
      : DEFAULT_SAMPLE_COUNT
  );
  const deterministic = deterministicServiceDependencies(
    `${profile}\0${phaseNativeSample}\0${preset}\0${channelSeed}`
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
    receiverImpairment: preset,
  };
  const configured = configurator.configureChannel({ channel });
  if (configured.profile !== profile) {
    throw new Error(`Service configured ${configured.profile}, expected ${profile}`);
  }
  const divisor = greatestCommonDivisor(
    phaseNativeSample,
    binding.nativeSampleRateHz,
  );
  const resumed = new AtomizerMeasurementService(buildIdentity, {
    ...deterministicServiceDependencies(
      `${profile}\0resume\0${phaseNativeSample}\0${preset}\0${channelSeed}`,
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
  const captureBandwidthHz =
    2 * Math.abs(binding.nativeCarrierOffsetHz)
    + binding.signalBandwidthHz;
  const raw = resumed.acquireIq({
    centerHz: binding.profileReferenceCenterHz,
    sampleRateHz: binding.nativeSampleRateHz,
    captureBandwidthHz,
    sampleCount,
    sampleFormat: 'cf32le',
  });
  const measurement = complexIqMeasurementSchema.parse(raw);
  const bytes = decodeAndValidatePayload(measurement);
  if (
    measurement.sampleCount !== sampleCount
    || measurement.sampleRateHz !== binding.nativeSampleRateHz
    || measurement.signalBandwidthHz !== binding.signalBandwidthHz
    || measurement.captureBandwidthHz !== captureBandwidthHz
    || measurement.profileReferenceCenterHz
      !== binding.profileReferenceCenterHz
    || measurement.receiverImpairment !== preset
  ) {
    throw new Error(`${profile} service response geometry/configuration drifted`);
  }
  if (
    measurement.transformReceipt.outputStartSourceSampleNumerator
      !== String(phaseNativeSample)
    || measurement.transformReceipt.outputStartSourceSampleDenominator !== '1'
  ) {
    throw new Error(`${profile} service receipt did not preserve exact phase`);
  }
  const expectedBoundary = binding.replay === 'cyclic'
    ? 'cyclic-modular'
    : 'one-shot-zero-extended';
  if (measurement.transformReceipt.sourceBoundaryPolicy !== expectedBoundary) {
    throw new Error(`${profile} service boundary policy drifted`);
  }
  return { measurement, bytes };
}

function decodeAndValidatePayload(
  measurement: ComplexIqMeasurement,
): Uint8Array {
  const canonical = Buffer.from(measurement.samplesBase64, 'base64');
  if (canonical.toString('base64') !== measurement.samplesBase64) {
    throw new Error('Service payload is not canonical base64');
  }
  if (
    canonical.byteLength !== measurement.byteLength
    || canonical.byteLength
      !== measurement.sampleCount * CF32LE_BYTES_PER_SAMPLE
  ) {
    throw new Error('Service payload byte length does not match its contract');
  }
  const bytes = new Uint8Array(
    canonical.buffer,
    canonical.byteOffset,
    canonical.byteLength,
  );
  const payloadSha256 = sha256(bytes);
  if (
    payloadSha256 !== measurement.samplesSha256
    || payloadSha256 !== measurement.transformReceipt.outputSamplesSha256
  ) {
    throw new Error('Service payload SHA-256 does not match its receipt');
  }
  return Uint8Array.from(bytes);
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
    throw new Error('Non-clean measurement lacks exactly one impairment receipt');
  }
  const operation = operations[0]!;
  if (operation.kind !== 'receiver-impairment' || operation.preset !== preset) {
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
      Date.UTC(2026, 6, 29, 0, 0, 0, clockSequence++),
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
  if (!UUID_PATTERN.test(uuid)) throw new Error('Deterministic UUID failed');
  return uuid;
}

function validateReceiverPresetContract(): void {
  const contractPresets = [...receiverImpairmentPresetSchema.options];
  if (
    contractPresets.length !== RECEIVER_PRESETS.length
    || contractPresets.some(
      (preset, index) => preset !== RECEIVER_PRESETS[index],
    )
  ) {
    throw new Error(
      'Generator receiver presets drifted from SignalLab contract order',
    );
  }
}

function readBuildIdentity(signalLabRoot: string): MeasurementBuildIdentity {
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
  if (
    !HASH_PATTERN.test(contractSha256)
    || !HASH_PATTERN.test(generatorContractBindingSha256)
  ) {
    throw new Error('Measurement build identity hashing failed');
  }
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
  if (!/^[a-f0-9]{40}$/.test(commit) || !/^[a-f0-9]{40}$/.test(tree)) {
    throw new Error('SignalLab git identity is not SHA-1 shaped');
  }
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
  const sourceSha256 = sha256(readFileSync(resolve(root, GENERATOR_SOURCE_PATH)));
  const bundleSha256 = sha256(readFileSync(expectedBundlePath));
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
      `Generator source or bundle changed during corpus generation: `
      + changed.join(', '),
    );
  }
}

function openOutputs(options: GeneratorOptions): OpenOutputs {
  mkdirSync(options.outputDirectory, { recursive: true });
  const mainFinal = resolve(options.outputDirectory, 'corpus.f32');
  const cleanFinal = resolve(options.outputDirectory, 'corpus_clean.f32');
  const manifestFinal = resolve(options.outputDirectory, 'corpus.json');
  const mainPart = `${mainFinal}.part`;
  const cleanPart = `${cleanFinal}.part`;
  const manifestPart = `${manifestFinal}.part`;
  const paths = [
    mainFinal,
    cleanFinal,
    manifestFinal,
    mainPart,
    cleanPart,
    manifestPart,
  ];
  const existing = paths.filter(existsSync);
  if (existing.length > 0 && !options.allowOverwrite) {
    throw new Error(
      `Refusing to overwrite existing corpus output: ${existing.join(', ')}. `
      + 'Choose a fresh OUTPUT_DIR or set ALLOW_OVERWRITE=1 explicitly.',
    );
  }
  if (options.allowOverwrite) {
    for (const path of existing) rmSync(path, { force: true });
  }
  return {
    mainDescriptor: openSync(mainPart, 'wx'),
    cleanDescriptor: openSync(cleanPart, 'wx'),
    mainPart,
    cleanPart,
    manifestPart,
    mainFinal,
    cleanFinal,
    manifestFinal,
  };
}

function cleanupInvocationOutputs(outputs: OpenOutputs): void {
  for (const path of [
    outputs.mainPart,
    outputs.cleanPart,
    outputs.manifestPart,
    outputs.mainFinal,
    outputs.cleanFinal,
    outputs.manifestFinal,
  ]) {
    rmSync(path, { force: true });
  }
}

function closeQuietly(descriptor: number): void {
  try {
    closeSync(descriptor);
  } catch {
    // The primary generation error is more useful than a duplicate close error.
  }
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
    if (written <= 0) throw new Error('Corpus stream made no write progress');
    offset += written;
  }
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
  throw new Error(`${label} must be exactly 0 or 1 when supplied`);
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

function deterministicChannelSeed(corpusSeed: number, domain: string): number {
  const seed = hashUint32(`${corpusSeed}\0channel\0${domain}`) >>> 0;
  return seed === 0 ? 1 : seed;
}

function hashUint32(value: string): number {
  const digest = createHash('sha256').update(value, 'utf8').digest();
  return digest.readUInt32LE(0);
}

function sha256(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex');
}

function sha256Text(value: string): string {
  return createHash('sha256').update(value, 'utf8').digest('hex');
}

function greatestCommonDivisor(left: number, right: number): number {
  let a = Math.abs(left);
  let b = Math.abs(right);
  while (b !== 0) [a, b] = [b, a % b];
  return a;
}

function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let mixed = Math.imul(state ^ (state >>> 15), 1 | state);
    mixed = (
      mixed + Math.imul(mixed ^ (mixed >>> 7), 61 | mixed)
    ) ^ mixed;
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4_294_967_296;
  };
}

const invokedPath = process.argv[1] === undefined
  ? undefined
  : resolve(process.argv[1]);
if (
  invokedPath !== undefined
  && resolve(fileURLToPath(import.meta.url)) === invokedPath
) {
  generateCurrentSignalLabCorpus(readOptions());
}
