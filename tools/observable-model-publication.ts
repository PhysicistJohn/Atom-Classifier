import { createHash, randomUUID } from 'node:crypto';
import {
  closeSync,
  constants,
  fstatSync,
  fsyncSync,
  linkSync,
  lstatSync,
  mkdirSync,
  openSync,
  readSync,
  renameSync,
  unlinkSync,
  writeFileSync,
} from 'node:fs';
import { dirname, resolve } from 'node:path';

const PUBLICATION_JOURNAL_SCHEMA_VERSION = 1 as const;
const MAX_MODEL_BYTES = 128 * 1024 * 1024;
const MAX_MANIFEST_BYTES = 1024 * 1024;
const MAX_JOURNAL_BYTES = 64 * 1024;
const UUID_V4_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

interface PublicationJournal {
  schemaVersion: typeof PUBLICATION_JOURNAL_SCHEMA_VERSION;
  transactionId: string;
  newModelSha256: string;
  newManifestSha256: string;
  oldModelSha256: string | null;
  oldManifestSha256: string | null;
}

export interface GeneratedModelManifestPair {
  modelSourceSha256: string;
  modelContentSha256: string;
}

export function assertGeneratedModelManifestPair(
  modelPath: string,
  manifestPath: string,
): GeneratedModelManifestPair {
  const modelBytes = readBoundedRegularFile(modelPath, MAX_MODEL_BYTES);
  const manifestBytes = readBoundedRegularFile(manifestPath, MAX_MANIFEST_BYTES);
  const modelSourceSha256 = sha256(modelBytes);
  const manifestSourceHashes = [...manifestBytes.toString('utf8').matchAll(
    /BAYESIAN_OBSERVABLE_MODEL_SHA256 = '([a-f0-9]{64})'/g,
  )];
  const modelContentHashes = [...modelBytes.toString('utf8').matchAll(
    /BAYESIAN_OBSERVABLE_MODEL_CONTENT_SHA256 = '([a-f0-9]{64})'/g,
  )];
  const manifestContentHashes = [...manifestBytes.toString('utf8').matchAll(
    /BAYESIAN_OBSERVABLE_MODEL_CONTENT_SHA256 = '([a-f0-9]{64})'/g,
  )];
  if (manifestSourceHashes.length !== 1
    || manifestSourceHashes[0]![1] !== modelSourceSha256
    || modelContentHashes.length !== 1
    || manifestContentHashes.length !== 1
    || modelContentHashes[0]![1] !== manifestContentHashes[0]![1]) {
    throw new Error('Generated classifier model and manifest are not a fail-closed matching pair');
  }
  return {
    modelSourceSha256,
    modelContentSha256: modelContentHashes[0]![1]!,
  };
}

export function recoverGeneratedModelManifestPublication(options: {
  modelPath: string;
  manifestPath: string;
  journalPath: string;
  failAfterModelRestoreForTest?: boolean;
}): 'none' | 'finalized-new' | 'finalized-old' | 'restored-old' | 'removed-partial-initial' {
  const modelPath = resolve(options.modelPath);
  const manifestPath = resolve(options.manifestPath);
  const journalPath = resolve(options.journalPath);
  let journal: PublicationJournal;
  try {
    journal = validateJournal(JSON.parse(
      readBoundedRegularFile(journalPath, MAX_JOURNAL_BYTES).toString('utf8'),
    ) as PublicationJournal);
  } catch (error) {
    if (isNodeError(error, 'ENOENT')) return 'none';
    throw error;
  }
  const paths = transactionPaths(modelPath, manifestPath, journal.transactionId);
  const currentModelSha256 = fileSha256(modelPath, MAX_MODEL_BYTES);
  const currentManifestSha256 = fileSha256(manifestPath, MAX_MANIFEST_BYTES);
  if (currentModelSha256 === journal.newModelSha256
    && currentManifestSha256 === journal.newManifestSha256) {
    assertGeneratedModelManifestPair(modelPath, manifestPath);
    cleanupTransaction(paths, journalPath);
    return 'finalized-new';
  }
  if (currentModelSha256 === journal.oldModelSha256
    && currentManifestSha256 === journal.oldManifestSha256) {
    if (journal.oldModelSha256 !== null) {
      assertRecoverableExistingModelManifestPair(modelPath, manifestPath);
    }
    cleanupTransaction(paths, journalPath);
    return 'finalized-old';
  }
  if (journal.oldModelSha256 === null && journal.oldManifestSha256 === null) {
    unlinkIfPresent(modelPath);
    unlinkIfPresent(manifestPath);
    cleanupTransaction(paths, journalPath);
    return 'removed-partial-initial';
  }
  if (journal.oldModelSha256 === null || journal.oldManifestSha256 === null) {
    throw new Error('Interrupted generated model publication cannot be safely recovered');
  }
  restorePublicationArtifact(
    modelPath,
    paths.modelBackup,
    journal.oldModelSha256,
    MAX_MODEL_BYTES,
  );
  if (options.failAfterModelRestoreForTest) {
    throw new Error('Injected failure after generated model restore');
  }
  restorePublicationArtifact(
    manifestPath,
    paths.manifestBackup,
    journal.oldManifestSha256,
    MAX_MANIFEST_BYTES,
  );
  assertRecoverableExistingModelManifestPair(modelPath, manifestPath);
  cleanupTransaction(paths, journalPath);
  return 'restored-old';
}

export function publishGeneratedModelManifestRecoverably(options: {
  modelPath: string;
  manifestPath: string;
  journalPath: string;
  modelSource: string;
  manifestSource: string;
  failAfterModelRenameForTest?: boolean;
}): void {
  const modelPath = resolve(options.modelPath);
  const manifestPath = resolve(options.manifestPath);
  const journalPath = resolve(options.journalPath);
  recoverGeneratedModelManifestPublication({ modelPath, manifestPath, journalPath });
  const oldModelSha256 = fileSha256(modelPath, MAX_MODEL_BYTES);
  const oldManifestSha256 = fileSha256(manifestPath, MAX_MANIFEST_BYTES);
  if ((oldModelSha256 === null) !== (oldManifestSha256 === null)) {
    throw new Error('Generated model publication starts from an incomplete pair');
  }
  if (oldModelSha256 !== null) {
    assertRecoverableExistingModelManifestPair(modelPath, manifestPath);
  }
  const modelBytes = Buffer.from(options.modelSource, 'utf8');
  const manifestBytes = Buffer.from(options.manifestSource, 'utf8');
  if (modelBytes.length > MAX_MODEL_BYTES || manifestBytes.length > MAX_MANIFEST_BYTES) {
    throw new Error('Generated model publication exceeds its bounded artifact size');
  }
  assertPairBytes(modelBytes, manifestBytes);
  const journal: PublicationJournal = {
    schemaVersion: PUBLICATION_JOURNAL_SCHEMA_VERSION,
    transactionId: randomUUID(),
    newModelSha256: sha256(modelBytes),
    newManifestSha256: sha256(manifestBytes),
    oldModelSha256,
    oldManifestSha256,
  };
  const paths = transactionPaths(modelPath, manifestPath, journal.transactionId);
  mkdirSync(dirname(modelPath), { recursive: true });
  mkdirSync(dirname(manifestPath), { recursive: true });
  mkdirSync(dirname(journalPath), { recursive: true });
  writeExclusiveFile(paths.modelStaged, modelBytes);
  writeExclusiveFile(paths.manifestStaged, manifestBytes);
  if (oldModelSha256 !== null && oldManifestSha256 !== null) {
    linkSync(modelPath, paths.modelBackup);
    linkSync(manifestPath, paths.manifestBackup);
  }
  // Make every recovery prerequisite durable before making the journal
  // durable. If the journal survives a crash, its staged artifacts and any
  // required backups must already be discoverable in their directories.
  fsyncDirectory(dirname(modelPath));
  if (dirname(manifestPath) !== dirname(modelPath)) {
    fsyncDirectory(dirname(manifestPath));
  }
  writeAtomicFile(journalPath, Buffer.from(JSON.stringify(journal), 'utf8'));
  renameSync(paths.modelStaged, modelPath);
  fsyncDirectory(dirname(modelPath));
  if (options.failAfterModelRenameForTest) {
    throw new Error('Injected failure after generated model rename');
  }
  renameSync(paths.manifestStaged, manifestPath);
  fsyncDirectory(dirname(manifestPath));
  assertGeneratedModelManifestPair(modelPath, manifestPath);
  cleanupTransaction(paths, journalPath);
}

function assertPairBytes(modelBytes: Buffer, manifestBytes: Buffer): void {
  const sourceMatches = [...manifestBytes.toString('utf8').matchAll(
    /BAYESIAN_OBSERVABLE_MODEL_SHA256 = '([a-f0-9]{64})'/g,
  )];
  const modelContentMatches = [...modelBytes.toString('utf8').matchAll(
    /BAYESIAN_OBSERVABLE_MODEL_CONTENT_SHA256 = '([a-f0-9]{64})'/g,
  )];
  const manifestContentMatches = [...manifestBytes.toString('utf8').matchAll(
    /BAYESIAN_OBSERVABLE_MODEL_CONTENT_SHA256 = '([a-f0-9]{64})'/g,
  )];
  if (sourceMatches.length !== 1 || sourceMatches[0]![1] !== sha256(modelBytes)
    || modelContentMatches.length !== 1 || manifestContentMatches.length !== 1
    || modelContentMatches[0]![1] !== manifestContentMatches[0]![1]) {
    throw new Error('Generated model publication bytes do not form a matching pair');
  }
}

function assertRecoverableExistingModelManifestPair(
  modelPath: string,
  manifestPath: string,
): void {
  const modelBytes = readBoundedRegularFile(modelPath, MAX_MODEL_BYTES);
  const manifestBytes = readBoundedRegularFile(manifestPath, MAX_MANIFEST_BYTES);
  const sourceMatches = [...manifestBytes.toString('utf8').matchAll(
    /BAYESIAN_OBSERVABLE_MODEL_SHA256 = '([a-f0-9]{64})'/g,
  )];
  const modelContentMatches = [...modelBytes.toString('utf8').matchAll(
    /BAYESIAN_OBSERVABLE_MODEL_CONTENT_SHA256 = '([a-f0-9]{64})'/g,
  )];
  const manifestContentMatches = [...manifestBytes.toString('utf8').matchAll(
    /BAYESIAN_OBSERVABLE_MODEL_CONTENT_SHA256 = '([a-f0-9]{64})'/g,
  )];
  const sourceMatchesModel = sourceMatches.length === 1
    && sourceMatches[0]![1] === sha256(modelBytes);
  const isLegacyPair = modelContentMatches.length === 0
    && manifestContentMatches.length === 0;
  const isStrictPair = modelContentMatches.length === 1
    && manifestContentMatches.length === 1
    && modelContentMatches[0]![1] === manifestContentMatches[0]![1];
  if (!sourceMatchesModel || (!isLegacyPair && !isStrictPair)) {
    throw new Error(
      'Existing generated classifier pair is neither strict nor a source-hash-valid legacy pair',
    );
  }
}

function transactionPaths(modelPath: string, manifestPath: string, transactionId: string) {
  return {
    modelStaged: `${modelPath}.${transactionId}.staged`,
    manifestStaged: `${manifestPath}.${transactionId}.staged`,
    modelBackup: `${modelPath}.${transactionId}.backup`,
    manifestBackup: `${manifestPath}.${transactionId}.backup`,
  };
}

function restorePublicationArtifact(
  destinationPath: string,
  backupPath: string,
  expectedSha256: string,
  maximumBytes: number,
): void {
  if (fileSha256(destinationPath, maximumBytes) === expectedSha256) return;
  if (fileSha256(backupPath, maximumBytes) !== expectedSha256) {
    throw new Error('Interrupted generated model publication cannot be safely recovered');
  }
  renameSync(backupPath, destinationPath);
  fsyncDirectory(dirname(destinationPath));
}

function cleanupTransaction(
  paths: ReturnType<typeof transactionPaths>,
  journalPath: string,
): void {
  for (const path of [
    paths.modelStaged,
    paths.manifestStaged,
    paths.modelBackup,
    paths.manifestBackup,
    journalPath,
  ]) {
    unlinkIfPresent(path);
  }
  fsyncDirectory(dirname(paths.modelStaged));
  if (dirname(paths.manifestStaged) !== dirname(paths.modelStaged)) {
    fsyncDirectory(dirname(paths.manifestStaged));
  }
  fsyncDirectory(dirname(journalPath));
}

function validateJournal(value: unknown): PublicationJournal {
  if (!isPlainRecord(value)
    || !hasExactKeys(value, [
      'schemaVersion',
      'transactionId',
      'newModelSha256',
      'newManifestSha256',
      'oldModelSha256',
      'oldManifestSha256',
    ])
    || value.schemaVersion !== PUBLICATION_JOURNAL_SCHEMA_VERSION
    || typeof value.transactionId !== 'string'
    || !UUID_V4_PATTERN.test(value.transactionId)
    || !isSha256(value.newModelSha256)
    || !isSha256(value.newManifestSha256)
    || (value.oldModelSha256 !== null && !isSha256(value.oldModelSha256))
    || (value.oldManifestSha256 !== null && !isSha256(value.oldManifestSha256))) {
    throw new Error('Generated model publication journal is malformed');
  }
  return value as PublicationJournal;
}

function fileSha256(path: string, maximumBytes: number): string | null {
  try {
    return sha256(readBoundedRegularFile(path, maximumBytes));
  } catch (error) {
    if (isNodeError(error, 'ENOENT')) return null;
    throw error;
  }
}

function writeExclusiveFile(path: string, bytes: Buffer): void {
  const descriptor = openSync(path, 'wx', 0o600);
  try {
    writeFileSync(descriptor, bytes);
    fsyncSync(descriptor);
  } finally {
    closeSync(descriptor);
  }
}

function writeAtomicFile(path: string, bytes: Buffer): void {
  const temporaryPath = `${path}.${process.pid}.${randomUUID()}.tmp`;
  writeExclusiveFile(temporaryPath, bytes);
  renameSync(temporaryPath, path);
  fsyncDirectory(dirname(path));
}

function readBoundedRegularFile(path: string, maximumBytes: number): Buffer {
  const noFollowFlag = typeof constants.O_NOFOLLOW === 'number'
    ? constants.O_NOFOLLOW
    : undefined;
  const beforeOpen = noFollowFlag === undefined ? lstatSync(path) : undefined;
  if (beforeOpen && (!beforeOpen.isFile() || beforeOpen.isSymbolicLink())) {
    throw new Error(`Generated model publication path is not a bounded regular file: ${path}`);
  }
  const descriptor = openSync(path, constants.O_RDONLY | (noFollowFlag ?? 0));
  try {
    const status = fstatSync(descriptor);
    if (!status.isFile() || status.size > maximumBytes) {
      throw new Error(`Generated model publication path is not a bounded regular file: ${path}`);
    }
    if (beforeOpen) {
      const afterOpen = lstatSync(path);
      if (!afterOpen.isFile()
        || afterOpen.isSymbolicLink()
        || !sameFileIdentity(beforeOpen, status)
        || !sameFileIdentity(afterOpen, status)) {
        throw new Error(`Generated model publication path changed during secure open: ${path}`);
      }
    }
    const output = Buffer.allocUnsafe(status.size);
    let offset = 0;
    while (offset < output.length) {
      const bytesRead = readSync(descriptor, output, offset, output.length - offset, offset);
      if (bytesRead === 0) throw new Error(`Generated model publication file truncated: ${path}`);
      offset += bytesRead;
    }
    return output;
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

function unlinkIfPresent(path: string): void {
  try {
    unlinkSync(path);
  } catch (error) {
    if (!isNodeError(error, 'ENOENT')) throw error;
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

function sha256(value: string | Buffer): string {
  return createHash('sha256').update(value).digest('hex');
}

function isSha256(value: unknown): value is string {
  return typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
}

function isPlainRecord(value: unknown): value is Record<string, any> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function hasExactKeys(
  value: Readonly<Record<string, unknown>>,
  expectedKeys: readonly string[],
): boolean {
  const actualKeys = Object.keys(value).sort();
  const sortedExpectedKeys = [...expectedKeys].sort();
  return actualKeys.length === sortedExpectedKeys.length
    && actualKeys.every((key, index) => key === sortedExpectedKeys[index]);
}

function isNodeError(error: unknown, code: string): error is NodeJS.ErrnoException {
  return error instanceof Error && 'code' in error && error.code === code;
}
