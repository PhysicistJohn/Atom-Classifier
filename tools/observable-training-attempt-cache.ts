import { createHash, randomUUID } from 'node:crypto';
import {
  closeSync,
  constants,
  existsSync,
  fstatSync,
  fsyncSync,
  linkSync,
  lstatSync,
  mkdirSync,
  openSync,
  readSync,
  readFileSync,
  renameSync,
  unlinkSync,
  writeFileSync,
} from 'node:fs';
import { arch, endianness, platform } from 'node:os';
import { basename, dirname, relative, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { gunzipSync, gzipSync } from 'node:zlib';
import type {
  AttemptSamplingWorkItem,
  AttemptSamplingWorkResult,
} from './observable-training-sampling-worker.js';

// This trusted-local cache checkpoints only deterministic feature-sampling
// worker results. It is crash recovery/developer acceleration, not
// authenticated provenance or release evidence; the release gate uses the
// separately journaled fresh-sampling path.
// Its fingerprint intentionally follows the worker's executable module
// closure, not the trainer entrypoint: changing a downstream coverage
// assertion must not discard an already completed sampling phase, while any
// change to synthesis, detection, tracking, extraction, corpus data, runtime,
// or the ordered nuisance matrix must select a new content-addressed cache.
// v2 distinguishes a receipt-verified physical detected-power result from an
// admitted envelope representative and validates censored agile results as an
// exact spectrum-only shape. A v1 chunk could silently preserve the former
// mislabelled/envelope-shaped censor representation, so it is never reused.
const CACHE_SCHEMA_VERSION = 2 as const;
const SERIALIZATION_POLICY = 'json-preserve-runtime-property-order-v1' as const;
const COMPRESSION_POLICY = 'gzip-level-6-v1' as const;
const CHUNK_FILE_NAME_WIDTH = 6;
export const ATTEMPT_SAMPLING_CACHE_MAX_COMPRESSED_CHUNK_BYTES = 4 * 1024 * 1024;
export const ATTEMPT_SAMPLING_CACHE_MAX_UNCOMPRESSED_CHUNK_BYTES = 16 * 1024 * 1024;
const ATTEMPT_SAMPLING_CACHE_MAX_MANIFEST_BYTES = 4 * 1024 * 1024;
const BOUNDED_READ_BLOCK_BYTES = 64 * 1024;
const SPECTRUM_FEATURE_NAMES = [
  'association.logBayesFactor',
  'spectrum.logBandwidthHz',
  'spectrum.logBandwidthRbwRatio',
  'spectrum.prominenceDb',
  'spectrum.flatness',
  'spectrum.entropy',
  'spectrum.symmetry',
  'spectrum.centerFraction',
  'spectrum.sidebandScore',
  'spectrum.peakDensity',
  'spectrum.centerNotch',
  'spectrum.logClusterCount',
  'spectrum.peakDriftFraction',
  'spectrum.powerVariationDb',
  'history.peakSpanFraction',
  'history.raster1MHzScore',
  'history.raster2MHzScore',
  'history.bleAdvertisingScore',
] as const;
const ENVELOPE_FEATURE_NAMES = [
  'envelope.rangeDb',
  'envelope.standardDeviationDb',
  'envelope.duty',
  'envelope.tuneOffsetFraction',
  'envelope.logTransitionRateHz',
  'envelope.periodicEnergy100Hz',
  'envelope.periodicEnergy200Hz',
  'envelope.periodicEnergy1600Hz',
  'envelope.periodicEnergy1733Hz',
  'envelope.periodicEnergy2000Hz',
] as const;
const FREQUENCY_AGILE_FIXED_TUNE_CENSORED_SCENARIO_IDS = new Set([
  'bluetooth-classic-connected',
  'bluetooth-le-advertising',
]);

export interface AttemptSamplingCacheOptions {
  rootDirectory: string;
  phase: 'fitting' | 'calibration';
  sourceIdentity: Readonly<Record<string, unknown>>;
  scenarios: readonly Readonly<{ id: string; value: unknown }>[];
  snrLevels: readonly number[];
  regimes: readonly unknown[];
  seeds: readonly number[];
  items: readonly AttemptSamplingWorkItem[];
  chunks: readonly (readonly AttemptSamplingWorkItem[])[];
  workerModuleUrl: URL;
}

export interface AttemptSamplingCacheChunk {
  results: readonly AttemptSamplingWorkResult[];
  record: AttemptSamplingCacheChunkRecord;
}

export interface AttemptSamplingCacheChunkRecord {
  chunkIndex: number;
  requestedItemsSha256: string;
  payloadSha256: string;
}

export interface AttemptSamplingCache {
  readonly fingerprint: string;
  readonly directory: string;
  readonly corruptChunkCount: number;
  loadChunk(
    chunkIndex: number,
    requestedItems: readonly AttemptSamplingWorkItem[],
  ): AttemptSamplingCacheChunk | undefined;
  publishChunk(
    chunkIndex: number,
    requestedItems: readonly AttemptSamplingWorkItem[],
    results: readonly AttemptSamplingWorkResult[],
  ): AttemptSamplingCacheChunkRecord;
  seal(records: readonly AttemptSamplingCacheChunkRecord[]): void;
  chunkPath(chunkIndex: number): string;
  manifestPath(): string;
}

interface ChunkPayload {
  schemaVersion: typeof CACHE_SCHEMA_VERSION;
  fingerprint: string;
  phase: AttemptSamplingCacheOptions['phase'];
  chunkIndex: number;
  requestedItemsSha256: string;
  results: readonly AttemptSamplingWorkResult[];
}

interface CacheEnvelope<T> {
  schemaVersion: typeof CACHE_SCHEMA_VERSION;
  payloadSha256: string;
  payload: T;
}

interface ManifestPayload {
  schemaVersion: typeof CACHE_SCHEMA_VERSION;
  fingerprint: string;
  phase: AttemptSamplingCacheOptions['phase'];
  itemCount: number;
  orderedItemsSha256: string;
  chunks: readonly AttemptSamplingCacheChunkRecord[];
}

interface ValidatedFile<T> {
  serializedPayload: string;
  payloadSha256: string;
  payload: T;
}

export function createAttemptSamplingCache(
  options: AttemptSamplingCacheOptions,
): AttemptSamplingCache {
  assertExactChunkPartition(options.items, options.chunks);
  const orderedItemsSha256 = sha256Canonical(options.items);
  const runtimeModuleManifest = workerRuntimeModuleManifest(options.workerModuleUrl);
  const cacheImplementationPath = resolve('tools/observable-training-attempt-cache.ts');
  const fingerprintDescriptor = {
    schemaVersion: CACHE_SCHEMA_VERSION,
    serializationPolicy: SERIALIZATION_POLICY,
    compressionPolicy: COMPRESSION_POLICY,
    phase: options.phase,
    sourceIdentity: options.sourceIdentity,
    runtime: {
      versions: { ...process.versions },
      platform: platform(),
      arch: arch(),
      endianness: endianness(),
      collation: new Intl.Collator().resolvedOptions(),
      cacheImplementation: {
        path: 'tools/observable-training-attempt-cache.ts',
        sha256: sha256(readFileSync(cacheImplementationPath)),
      },
      workerModuleManifest: runtimeModuleManifest,
    },
    matrix: {
      scenarios: options.scenarios.map((scenario) => ({
        id: scenario.id,
        valueSha256: sha256Canonical(scenario.value),
      })),
      snrLevels: options.snrLevels,
      regimes: options.regimes,
      seeds: options.seeds,
      orderedItemsSha256,
      itemCount: options.items.length,
      chunkItemCounts: options.chunks.map((chunk) => chunk.length),
    },
  };
  const fingerprint = sha256Canonical(fingerprintDescriptor);
  const directory = resolve(
    options.rootDirectory,
    options.phase,
    fingerprint,
  );
  mkdirSync(directory, { recursive: true });
  let corruptChunkCount = 0;

  const chunkPath = (chunkIndex: number): string => resolve(
    directory,
    `chunk-${String(chunkIndex).padStart(CHUNK_FILE_NAME_WIDTH, '0')}.json.gz`,
  );
  const manifestPath = (): string => resolve(directory, 'manifest.json');

  const loadChunk = (
    chunkIndex: number,
    requestedItems: readonly AttemptSamplingWorkItem[],
  ): AttemptSamplingCacheChunk | undefined => {
    const path = chunkPath(chunkIndex);
    if (!existsSync(path)) return undefined;
    const validated = validateChunkFile(
      path,
      fingerprint,
      options.phase,
      chunkIndex,
      requestedItems,
    );
    if (!validated) {
      corruptChunkCount += 1;
      return undefined;
    }
    return {
      results: validated.payload.results,
      record: {
        chunkIndex,
        requestedItemsSha256: validated.payload.requestedItemsSha256,
        payloadSha256: validated.payloadSha256,
      },
    };
  };

  const publishChunk = (
    chunkIndex: number,
    requestedItems: readonly AttemptSamplingWorkItem[],
    results: readonly AttemptSamplingWorkResult[],
  ): AttemptSamplingCacheChunkRecord => {
    if (!successfulResultsHaveExactRuntimeShape(results, requestedItems)) {
      throw new Error(`Attempt-sampling cache chunk ${chunkIndex} has malformed worker results`);
    }
    const requestedItemsSha256 = sha256Canonical(requestedItems);
    const payload: ChunkPayload = {
      schemaVersion: CACHE_SCHEMA_VERSION,
      fingerprint,
      phase: options.phase,
      chunkIndex,
      requestedItemsSha256,
      results,
    };
    const serialized = serializeEnvelope(payload);
    if (serialized.envelopeBytes.length > ATTEMPT_SAMPLING_CACHE_MAX_UNCOMPRESSED_CHUNK_BYTES) {
      throw new Error(
        `Attempt-sampling cache chunk ${chunkIndex} exceeds the uncompressed size limit`,
      );
    }
    const path = chunkPath(chunkIndex);
    const compressedBytes = gzipSync(serialized.envelopeBytes, { level: 6 });
    if (compressedBytes.length > ATTEMPT_SAMPLING_CACHE_MAX_COMPRESSED_CHUNK_BYTES) {
      throw new Error(
        `Attempt-sampling cache chunk ${chunkIndex} exceeds the compressed size limit`,
      );
    }
    publishDeterministicFile(
      path,
      compressedBytes,
      () => {
        const existing = validateChunkFile(
          path,
          fingerprint,
          options.phase,
          chunkIndex,
          requestedItems,
        );
        return existing && readBoundedRegularFile(
          path,
          ATTEMPT_SAMPLING_CACHE_MAX_COMPRESSED_CHUNK_BYTES,
        ).equals(compressedBytes)
          ? 'identical'
          : existing
            ? 'different-valid'
            : 'invalid';
      },
    );
    return {
      chunkIndex,
      requestedItemsSha256,
      payloadSha256: serialized.payloadSha256,
    };
  };

  const seal = (records: readonly AttemptSamplingCacheChunkRecord[]): void => {
    assertExactChunkRecords(records, options.chunks);
    const payload: ManifestPayload = {
      schemaVersion: CACHE_SCHEMA_VERSION,
      fingerprint,
      phase: options.phase,
      itemCount: options.items.length,
      orderedItemsSha256,
      chunks: records,
    };
    const serialized = serializeEnvelope(payload);
    if (serialized.envelopeBytes.length > ATTEMPT_SAMPLING_CACHE_MAX_MANIFEST_BYTES) {
      throw new Error('Attempt-sampling cache manifest exceeds its size limit');
    }
    const path = manifestPath();
    publishDeterministicFile(
      path,
      serialized.envelopeBytes,
      () => {
        const existing = validateManifestFile(
          path,
          fingerprint,
          options.phase,
          orderedItemsSha256,
          options.items.length,
          records,
        );
        return existing && readBoundedRegularFile(
          path,
          ATTEMPT_SAMPLING_CACHE_MAX_MANIFEST_BYTES,
        ).equals(serialized.envelopeBytes)
          ? 'identical'
          : existing
            ? 'different-valid'
            : 'invalid';
      },
    );
  };

  return {
    fingerprint,
    directory,
    get corruptChunkCount() {
      return corruptChunkCount;
    },
    loadChunk,
    publishChunk,
    seal,
    chunkPath,
    manifestPath,
  };
}

function validateChunkFile(
  path: string,
  fingerprint: string,
  phase: AttemptSamplingCacheOptions['phase'],
  chunkIndex: number,
  requestedItems: readonly AttemptSamplingWorkItem[],
): ValidatedFile<ChunkPayload> | undefined {
  try {
    const compressedBytes = readBoundedRegularFile(
      path,
      ATTEMPT_SAMPLING_CACHE_MAX_COMPRESSED_CHUNK_BYTES,
    );
    const envelopeBytes = gunzipSync(compressedBytes, {
      maxOutputLength: ATTEMPT_SAMPLING_CACHE_MAX_UNCOMPRESSED_CHUNK_BYTES,
    });
    const validated = validateEnvelope<ChunkPayload>(envelopeBytes);
    const payload = validated.payload;
    if (!isPlainRecord(payload)
      || !hasExactKeys(payload, [
        'schemaVersion',
        'fingerprint',
        'phase',
        'chunkIndex',
        'requestedItemsSha256',
        'results',
      ])
      || payload.schemaVersion !== CACHE_SCHEMA_VERSION
      || payload.fingerprint !== fingerprint
      || payload.phase !== phase
      || payload.chunkIndex !== chunkIndex
      || payload.requestedItemsSha256 !== sha256Canonical(requestedItems)
      || !successfulResultsHaveExactRuntimeShape(payload.results, requestedItems)) {
      return undefined;
    }
    return validated;
  } catch {
    return undefined;
  }
}

function validateManifestFile(
  path: string,
  fingerprint: string,
  phase: AttemptSamplingCacheOptions['phase'],
  orderedItemsSha256: string,
  itemCount: number,
  expectedRecords: readonly AttemptSamplingCacheChunkRecord[],
): ValidatedFile<ManifestPayload> | undefined {
  try {
    const validated = validateEnvelope<ManifestPayload>(readBoundedRegularFile(
      path,
      ATTEMPT_SAMPLING_CACHE_MAX_MANIFEST_BYTES,
    ));
    const payload = validated.payload;
    if (!isPlainRecord(payload)
      || !hasExactKeys(payload, [
        'schemaVersion',
        'fingerprint',
        'phase',
        'itemCount',
        'orderedItemsSha256',
        'chunks',
      ])
      || payload.schemaVersion !== CACHE_SCHEMA_VERSION
      || payload.fingerprint !== fingerprint
      || payload.phase !== phase
      || payload.orderedItemsSha256 !== orderedItemsSha256
      || payload.itemCount !== itemCount
      || canonicalStringify(payload.chunks) !== canonicalStringify(expectedRecords)) {
      return undefined;
    }
    return validated;
  } catch {
    return undefined;
  }
}

function validateEnvelope<T>(bytes: Buffer): ValidatedFile<T> {
  const text = bytes.toString('utf8');
  const envelope = JSON.parse(text) as CacheEnvelope<T>;
  if (!isPlainRecord(envelope)
    || !hasExactKeys(envelope, ['schemaVersion', 'payloadSha256', 'payload'])
    || envelope.schemaVersion !== CACHE_SCHEMA_VERSION
    || typeof envelope.payloadSha256 !== 'string'
    || !/^[a-f0-9]{64}$/.test(envelope.payloadSha256)
    || envelope.payload === undefined) {
    throw new Error('Malformed attempt-sampling cache envelope');
  }
  if (JSON.stringify(envelope) !== text) {
    throw new Error('Attempt-sampling cache envelope is not in exact writer form');
  }
  const serializedPayload = JSON.stringify(envelope.payload);
  if (sha256(serializedPayload) !== envelope.payloadSha256) {
    throw new Error('Attempt-sampling cache payload hash mismatch');
  }
  return {
    serializedPayload,
    payloadSha256: envelope.payloadSha256,
    payload: envelope.payload,
  };
}

function serializeEnvelope<T>(payload: T): {
  serializedPayload: string;
  payloadSha256: string;
  envelopeBytes: Buffer;
} {
  assertExactJsonRoundTrip(payload);
  const serializedPayload = JSON.stringify(payload);
  const payloadSha256 = sha256(serializedPayload);
  const envelope: CacheEnvelope<T> = {
    schemaVersion: CACHE_SCHEMA_VERSION,
    payloadSha256,
    payload,
  };
  return {
    serializedPayload,
    payloadSha256,
    envelopeBytes: Buffer.from(JSON.stringify(envelope), 'utf8'),
  };
}

function publishDeterministicFile(
  path: string,
  bytes: Buffer,
  inspectExisting: () => 'identical' | 'different-valid' | 'invalid',
): void {
  mkdirSync(dirname(path), { recursive: true });
  const temporaryPath = `${path}.${process.pid}.${randomUUID()}.tmp`;
  let temporaryExists = false;
  try {
    const descriptor = openSync(temporaryPath, 'wx', 0o600);
    temporaryExists = true;
    try {
      writeFileSync(descriptor, bytes);
      fsyncSync(descriptor);
    } finally {
      closeSync(descriptor);
    }
    while (true) {
      try {
        linkSync(temporaryPath, path);
        fsyncDirectory(dirname(path));
        unlinkSync(temporaryPath);
        temporaryExists = false;
        fsyncDirectory(dirname(path));
        return;
      } catch (error) {
        if (!isNodeError(error, 'EEXIST')) throw error;
      }
      const existingState = inspectExisting();
      if (existingState === 'identical') {
        unlinkSync(temporaryPath);
        temporaryExists = false;
        return;
      }
      if (existingState === 'different-valid') {
        throw new Error(
          `Attempt-sampling cache nondeterminism: ${path} contains different valid bytes under an identical fingerprint`,
        );
      }
      const quarantinedPath = `${path}.${process.pid}.${randomUUID()}.corrupt`;
      try {
        renameSync(path, quarantinedPath);
      } catch (error) {
        if (isNodeError(error, 'ENOENT')) continue;
        throw error;
      }
      try {
        unlinkSync(quarantinedPath);
      } catch (error) {
        if (!isNodeError(error, 'ENOENT')) throw error;
      }
      fsyncDirectory(dirname(path));
    }
  } finally {
    if (temporaryExists) {
      try {
        unlinkSync(temporaryPath);
      } catch (error) {
        if (!isNodeError(error, 'ENOENT')) throw error;
      }
    }
  }
}

function fsyncDirectory(path: string): void {
  // Windows/NTFS rejects fsync on a directory handle (EPERM); directory-entry
  // durability there is a platform guarantee this call cannot add to.
  if (process.platform === 'win32') return;
  const descriptor = openSync(path, 'r');
  try {
    fsyncSync(descriptor);
  } finally {
    closeSync(descriptor);
  }
}

function assertExactChunkPartition(
  items: readonly AttemptSamplingWorkItem[],
  chunks: readonly (readonly AttemptSamplingWorkItem[])[],
): void {
  const flattened = chunks.flat();
  if (canonicalStringify(flattened) !== canonicalStringify(items)) {
    throw new Error('Attempt-sampling cache chunks are not an exact ordered partition of the matrix');
  }
}

function assertExactChunkRecords(
  records: readonly AttemptSamplingCacheChunkRecord[],
  chunks: readonly (readonly AttemptSamplingWorkItem[])[],
): void {
  if (records.length !== chunks.length) {
    throw new Error(`Attempt-sampling cache seal has ${records.length}/${chunks.length} chunk records`);
  }
  for (let index = 0; index < chunks.length; index += 1) {
    const record = records[index];
    if (!record
      || !isPlainRecord(record)
      || !hasExactKeys(record, ['chunkIndex', 'requestedItemsSha256', 'payloadSha256'])
      || record.chunkIndex !== index
      || record.requestedItemsSha256 !== sha256Canonical(chunks[index])
      || !isSha256(record.payloadSha256)) {
      throw new Error(`Attempt-sampling cache seal record ${index} does not match its ordered chunk`);
    }
  }
}

function successfulResultsHaveExactRuntimeShape(
  results: unknown,
  requestedItems: readonly AttemptSamplingWorkItem[],
): results is readonly AttemptSamplingWorkResult[] {
  if (!Array.isArray(results) || results.length !== requestedItems.length) return false;
  return results.every((result, index) => isPlainRecord(result)
    && hasExactKeys(result, ['key', 'attempt'])
    && result.key === requestedItems[index]?.key
    && featureSamplingAttemptHasExactRuntimeShape(
      result.attempt,
      requestedItems[index]!.scenarioId,
    ));
}

function featureSamplingAttemptHasExactRuntimeShape(
  value: unknown,
  scenarioId: string,
): boolean {
  if (!isPlainRecord(value)
    || !hasExactKeys(value, ['consecutiveSpectrum', 'qualifiedEnvelope'])) {
    return false;
  }
  const spectrum = value.consecutiveSpectrum;
  const envelope = value.qualifiedEnvelope;
  if (!isPlainRecord(spectrum)
    || !hasExactKeys(spectrum, [
      'observationHorizon',
      'onlineSpectrumRepresentatives',
      'provenanceUnavailableWindowCount',
      'sourceClockEventCount',
      'sourceClockTraceSha256',
    ])
    || !isPositiveInteger(spectrum.observationHorizon)
    || !Array.isArray(spectrum.onlineSpectrumRepresentatives)
    || !isNonNegativeInteger(spectrum.provenanceUnavailableWindowCount)
    || spectrum.sourceClockEventCount !== spectrum.observationHorizon
    || !isSha256(spectrum.sourceClockTraceSha256)
    || !spectrum.onlineSpectrumRepresentatives.every((sample) =>
      featureSampleHasExactRuntimeShape(
        sample,
        spectrum.observationHorizon as number,
        'spectrum',
      ))) {
    return false;
  }
  if (!isPlainRecord(envelope)
    || !hasExactKeys(
      envelope,
      [
        'observationHorizon',
        'provenanceUnavailableWindowCount',
        'preCaptureProvenanceUnavailableWindowCount',
        'postCaptureProvenanceUnavailableWindowCount',
        'physicalDetectedPowerCaptureCount',
        'sourceClockEventCount',
        'sourceClockTraceSha256',
      ],
      ['detectedPowerCaptureSample', 'capturedRepresentativeKey'],
    )
    || envelope.observationHorizon !== spectrum.observationHorizon
    || !isNonNegativeInteger(envelope.provenanceUnavailableWindowCount)
    || !isNonNegativeInteger(envelope.preCaptureProvenanceUnavailableWindowCount)
    || (envelope.postCaptureProvenanceUnavailableWindowCount !== 0
      && envelope.postCaptureProvenanceUnavailableWindowCount !== 1)
    || (envelope.physicalDetectedPowerCaptureCount !== 0
      && envelope.physicalDetectedPowerCaptureCount !== 1)
    || envelope.provenanceUnavailableWindowCount
      !== (envelope.preCaptureProvenanceUnavailableWindowCount as number)
        + (envelope.postCaptureProvenanceUnavailableWindowCount as number)
    || envelope.sourceClockEventCount
      !== (envelope.observationHorizon as number)
        + (envelope.physicalDetectedPowerCaptureCount as number)
    || !isSha256(envelope.sourceClockTraceSha256)) {
    return false;
  }
  const hasDetectedPowerCaptureSample = envelope.detectedPowerCaptureSample !== undefined;
  const hasCapturedRepresentativeKey = typeof envelope.capturedRepresentativeKey === 'string'
    && envelope.capturedRepresentativeKey.length > 0;
  if ((envelope.physicalDetectedPowerCaptureCount === 1) !== hasCapturedRepresentativeKey
    || envelope.physicalDetectedPowerCaptureCount
      !== Number(hasDetectedPowerCaptureSample)
        + (envelope.postCaptureProvenanceUnavailableWindowCount as number)
    || (hasDetectedPowerCaptureSample
      && !featureSampleHasExactRuntimeShape(
        envelope.detectedPowerCaptureSample,
        envelope.observationHorizon as number,
        'detected-power',
        scenarioId,
      ))) {
    return false;
  }
  return true;
}

function featureSampleHasExactRuntimeShape(
  value: unknown,
  observationHorizon: number,
  view: 'spectrum' | 'detected-power',
  scenarioId?: string,
): boolean {
  if (!isPlainRecord(value)
    || !hasExactKeys(
      value,
      ['values', 'observationOpportunity', 'fitEligible'],
      view === 'detected-power'
        ? ['envelopeUntimedFitEligible', 'detectedPowerEvidenceDisposition']
        : [],
    )
    || !Number.isInteger(value.observationOpportunity)
    || (value.observationOpportunity as number) < 1
    || (value.observationOpportunity as number) > observationHorizon
    || typeof value.fitEligible !== 'boolean'
    || (view === 'detected-power' && typeof value.envelopeUntimedFitEligible !== 'boolean')
    || (view === 'detected-power'
      && value.detectedPowerEvidenceDisposition !== 'admitted-envelope'
      && value.detectedPowerEvidenceDisposition !== 'censored-frequency-agile-fixed-tune')
    || !isPlainRecord(value.values)) {
    return false;
  }
  const values = value.values;
  const censored = view === 'detected-power'
    && value.detectedPowerEvidenceDisposition
      === 'censored-frequency-agile-fixed-tune';
  if (view === 'detected-power') {
    const scenarioRequiresCensoring =
      FREQUENCY_AGILE_FIXED_TUNE_CENSORED_SCENARIO_IDS.has(scenarioId ?? '');
    if (scenarioRequiresCensoring !== censored) return false;
  }
  if (censored
    && (value.fitEligible !== false
      || value.envelopeUntimedFitEligible !== false)) {
    return false;
  }
  const expectedNames = view === 'spectrum' || censored
    ? SPECTRUM_FEATURE_NAMES
    : [...SPECTRUM_FEATURE_NAMES, ...ENVELOPE_FEATURE_NAMES];
  if (!hasExactKeys(values, expectedNames)) return false;
  return expectedNames.every((name) =>
    typeof values[name] === 'number' && Number.isFinite(values[name]));
}

function readBoundedRegularFile(path: string, maximumBytes: number): Buffer {
  const noFollowFlag = typeof constants.O_NOFOLLOW === 'number'
    ? constants.O_NOFOLLOW
    : undefined;
  const beforeOpen = noFollowFlag === undefined ? lstatSync(path) : undefined;
  if (beforeOpen && (!beforeOpen.isFile() || beforeOpen.isSymbolicLink())) {
    throw new Error(`Attempt-sampling cache path is not a regular file: ${path}`);
  }
  const descriptor = openSync(
    path,
    constants.O_RDONLY | (noFollowFlag ?? 0),
  );
  try {
    const status = fstatSync(descriptor);
    if (!status.isFile()) {
      throw new Error(`Attempt-sampling cache path is not a regular file: ${path}`);
    }
    if (beforeOpen) {
      // Platforms without O_NOFOLLOW get a best-effort no-follow fallback.
      // Comparing the pre-open path, opened descriptor, and post-open path
      // rejects symlink substitution; a concurrent different regular-file
      // substitution can still race the path identity, but the bytes read
      // remain bounded to the already-opened regular descriptor and are
      // independently content-validated by the cache envelope.
      const afterOpen = lstatSync(path);
      if (!afterOpen.isFile()
        || afterOpen.isSymbolicLink()
        || !sameFileIdentity(beforeOpen, status)
        || !sameFileIdentity(afterOpen, status)) {
        throw new Error(`Attempt-sampling cache path changed during secure open: ${path}`);
      }
    }
    if (status.size > maximumBytes) {
      throw new Error(`Attempt-sampling cache file exceeds ${maximumBytes} bytes: ${path}`);
    }
    const chunks: Buffer[] = [];
    let totalBytes = 0;
    while (totalBytes <= maximumBytes) {
      const block = Buffer.allocUnsafe(Math.min(
        BOUNDED_READ_BLOCK_BYTES,
        maximumBytes - totalBytes + 1,
      ));
      const bytesRead = readSync(descriptor, block, 0, block.length, null);
      if (bytesRead === 0) break;
      chunks.push(block.subarray(0, bytesRead));
      totalBytes += bytesRead;
    }
    if (totalBytes > maximumBytes) {
      throw new Error(`Attempt-sampling cache file exceeds ${maximumBytes} bytes: ${path}`);
    }
    return Buffer.concat(chunks, totalBytes);
  } finally {
    closeSync(descriptor);
  }
}

function sameFileIdentity(
  left: Readonly<{ dev: number; ino: number }>,
  right: Readonly<{ dev: number; ino: number }>,
): boolean {
  return left.dev === right.dev && left.ino === right.ino;
}

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function hasExactKeys(
  value: Readonly<Record<string, unknown>>,
  requiredKeys: readonly string[],
  optionalKeys: readonly string[] = [],
): boolean {
  const actualKeys = Object.keys(value);
  const allowedKeys = new Set([...requiredKeys, ...optionalKeys]);
  return requiredKeys.every((key) => Object.hasOwn(value, key))
    && actualKeys.every((key) => allowedKeys.has(key));
}

function isPositiveInteger(value: unknown): value is number {
  return Number.isInteger(value) && (value as number) > 0;
}

function isNonNegativeInteger(value: unknown): value is number {
  return Number.isInteger(value) && (value as number) >= 0;
}

function isSha256(value: unknown): value is string {
  return typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
}

function workerRuntimeModuleManifest(
  workerModuleUrl: URL,
): readonly Readonly<{ path: string; sha256: string }>[] {
  if (workerModuleUrl.protocol !== 'file:') {
    throw new Error(`Attempt-sampling worker runtime must be a file URL, received ${workerModuleUrl.protocol}`);
  }
  const runtimeRoot = dirname(fileURLToPath(workerModuleUrl));
  const pending = [fileURLToPath(workerModuleUrl)];
  const visited = new Set<string>();
  while (pending.length > 0) {
    const path = resolve(pending.pop()!);
    if (visited.has(path)) continue;
    const relativePath = relative(runtimeRoot, path);
    if (relativePath.startsWith('..') || relativePath.includes('/../')) {
      throw new Error(`Attempt-sampling worker imports runtime code outside ${runtimeRoot}: ${path}`);
    }
    const source = readFileSync(path, 'utf8');
    visited.add(path);
    for (const specifier of localModuleSpecifiers(source)) {
      const importedUrl = new URL(specifier, pathToFileURL(path));
      if (importedUrl.protocol !== 'file:') continue;
      pending.push(fileURLToPath(importedUrl));
    }
  }
  return [...visited]
    .map((path) => ({
      path: path === fileURLToPath(workerModuleUrl)
        ? basename(path)
        : relative(runtimeRoot, path).split('\\').join('/'),
      sha256: sha256(readFileSync(path)),
    }))
    .sort((left, right) => left.path.localeCompare(right.path));
}

function localModuleSpecifiers(source: string): string[] {
  const matches: string[] = [];
  const patterns = [
    /\bfrom\s*["'](\.[^"']+)["']/g,
    /\bimport\s*["'](\.[^"']+)["']/g,
    /\bimport\s*\(\s*["'](\.[^"']+)["']\s*\)/g,
  ];
  for (const pattern of patterns) {
    for (const match of source.matchAll(pattern)) matches.push(match[1]!);
  }
  return [...new Set(matches)];
}

function sha256Canonical(value: unknown): string {
  return sha256(canonicalStringify(value));
}

function sha256(value: string | Buffer): string {
  return createHash('sha256').update(value).digest('hex');
}

function canonicalStringify(value: unknown): string {
  return JSON.stringify(canonicalize(value));
}

function canonicalize(value: unknown): unknown {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return value;
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw new Error('Attempt-sampling cache cannot serialize non-finite numbers');
    return Object.is(value, -0) ? 0 : value;
  }
  if (Array.isArray(value)) return value.map(canonicalize);
  if (typeof value === 'object') {
    const record = value as Record<string, unknown>;
    return Object.fromEntries(Object.keys(record)
      .filter((key) => record[key] !== undefined)
      .sort()
      .map((key) => [key, canonicalize(record[key])]));
  }
  throw new Error(`Attempt-sampling cache cannot serialize ${typeof value}`);
}

function assertExactJsonRoundTrip(
  value: unknown,
  path = '$',
  ancestors = new Set<object>(),
): void {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return;
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      throw new Error(`Attempt-sampling cache cannot serialize non-finite number at ${path}`);
    }
    return;
  }
  if (typeof value !== 'object') {
    throw new Error(`Attempt-sampling cache cannot exactly JSON-round-trip ${typeof value} at ${path}`);
  }
  if (ancestors.has(value)) {
    throw new Error(`Attempt-sampling cache cannot serialize a cycle at ${path}`);
  }
  ancestors.add(value);
  try {
    if (Array.isArray(value)) {
      value.forEach((entry, index) => {
        if (entry === undefined) {
          throw new Error(`Attempt-sampling cache cannot exactly JSON-round-trip undefined at ${path}[${index}]`);
        }
        assertExactJsonRoundTrip(entry, `${path}[${index}]`, ancestors);
      });
      return;
    }
    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) {
      throw new Error(`Attempt-sampling cache cannot exactly JSON-round-trip ${prototype?.constructor?.name ?? 'non-plain object'} at ${path}`);
    }
    for (const [key, entry] of Object.entries(value as Record<string, unknown>)) {
      if (entry === undefined) {
        throw new Error(`Attempt-sampling cache cannot exactly JSON-round-trip undefined at ${path}.${key}`);
      }
      assertExactJsonRoundTrip(entry, `${path}.${key}`, ancestors);
    }
  } finally {
    ancestors.delete(value);
  }
}

function isNodeError(error: unknown, code: string): error is NodeJS.ErrnoException {
  return error instanceof Error && 'code' in error && error.code === code;
}
