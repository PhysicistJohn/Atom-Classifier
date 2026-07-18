import { createHash, randomUUID } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import {
  chmodSync,
  closeSync,
  constants,
  fstatSync,
  fsyncSync,
  linkSync,
  lstatSync,
  mkdirSync,
  openSync,
  readdirSync,
  readFileSync,
  readSync,
  renameSync,
  rmSync,
  unlinkSync,
  writeFileSync,
} from 'node:fs';
import { basename, dirname, isAbsolute, relative, resolve } from 'node:path';
import { platform } from 'node:os';
import { fileURLToPath, pathToFileURL } from 'node:url';

const RUN_LOCK_SCHEMA_VERSION = 2 as const;
const RUN_OWNER_CLAIM_SCHEMA_VERSION = 1 as const;
const RUN_OWNER_TICKET_SCHEMA_VERSION = 1 as const;
const RUN_HEARTBEAT_SCHEMA_VERSION = 1 as const;
const FRESH_JOURNAL_SCHEMA_VERSION = 1 as const;
const MAX_CONTROL_FILE_BYTES = 64 * 1024;
const MAX_RUNTIME_MODULE_BYTES = 32 * 1024 * 1024;
const MAX_RUNTIME_MODULE_COUNT = 128;
const RUN_HEARTBEAT_INTERVAL_MS = 30_000;
const RUN_HEARTBEAT_STALE_AFTER_MS = 5 * 60_000;
const MAX_HEARTBEAT_CLOCK_SKEW_MS = 60_000;
const MALFORMED_LEGACY_LOCK_GRACE_MS = 60_000;
// Schema-v1 did not record a process-start identity or heartbeat. On platforms
// where process start time cannot be observed, retain a live PID for a full day
// before allowing recovery so legitimate multi-hour legacy runs stay exclusive.
const LEGACY_LOCK_FALLBACK_MAX_AGE_MS = 24 * 60 * 60_000;
const PROCESS_START_COMPARISON_TOLERANCE_MS = 2_000;
const RUN_HEARTBEAT_FILENAME = 'owner-heartbeat.json';
const RUN_OWNER_DIRECTORY_SUFFIX = '.owners';
const RUN_OWNER_TOKEN_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
const RUN_OWNER_CLAIM_FILENAME_PATTERN =
  /^([0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})\.json$/;
const RUN_OWNER_TICKET_FILENAME_PATTERN =
  /^([0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})\.ticket\.json$/;

interface RunLockPayload {
  schemaVersion: typeof RUN_LOCK_SCHEMA_VERSION;
  ownerToken: string;
  pid: number;
  processStartIdentity: string | null;
  startedAt: string;
  runDirectory: string;
}

interface RunHeartbeatPayload {
  schemaVersion: typeof RUN_HEARTBEAT_SCHEMA_VERSION;
  ownerToken: string;
  pid: number;
  processStartIdentity: string | null;
  updatedAt: string;
}

interface RunOwnerClaimPayload {
  schemaVersion: typeof RUN_OWNER_CLAIM_SCHEMA_VERSION;
  choosing: boolean;
  ticket: number | null;
  owner: RunLockPayload;
}

interface RunOwnerTicketPayload {
  schemaVersion: typeof RUN_OWNER_TICKET_SCHEMA_VERSION;
  ownerToken: string;
  ticket: number;
}

interface LegacyRunLockPayload {
  schemaVersion: 1;
  ownerToken: string;
  pid: number;
  startedAt: string;
  runDirectory: string;
}

interface RuntimeModule {
  path: string;
  bytes: Buffer;
  sha256: string;
}

interface FreshSamplingJournalPayload {
  schemaVersion: typeof FRESH_JOURNAL_SCHEMA_VERSION;
  status: 'in-progress' | 'completed';
  runId: string;
  compatibilitySha256: string;
  startedAt: string;
  updatedAt: string;
}

export interface ObservableTrainingRunControl {
  readonly runId: string;
  readonly runDirectory: string;
  readonly workerModuleUrl: URL;
  readonly workerRuntimeSha256: string;
  assertWorkerRuntimeUnchanged(): void;
  installProcessCleanupHandlers(): void;
  release(): void;
}

export interface FreshSamplingRunJournal {
  readonly runId: string;
  readonly cacheRoot: string;
  readonly resumed: boolean;
  markCompleted(): void;
}

export function acquireObservableTrainingRun(options: {
  lockPath: string;
  runRoot: string;
  sourceWorkerModuleUrl: URL;
}): ObservableTrainingRunControl {
  const lockPath = resolve(options.lockPath);
  const ownerDirectory = `${lockPath}${RUN_OWNER_DIRECTORY_SUFFIX}`;
  const runRoot = resolve(options.runRoot);
  mkdirStrictDirectory(dirname(lockPath));
  mkdirStrictDirectory(ownerDirectory);
  mkdirStrictDirectory(runRoot);
  const ownerToken = randomUUID();
  const runId = ownerToken;
  const runDirectory = resolve(runRoot, runId);
  const runtimeDirectory = resolve(runDirectory, 'runtime');
  const heartbeatPath = resolve(runDirectory, RUN_HEARTBEAT_FILENAME);
  const ownerClaimPath = runOwnerClaimPath(ownerDirectory, ownerToken);
  const ownerTicketPath = runOwnerTicketPath(ownerDirectory, ownerToken);
  const processStartIdentity = readProcessStartIdentity(process.pid) ?? null;
  const lockPayload: RunLockPayload = {
    schemaVersion: RUN_LOCK_SCHEMA_VERSION,
    ownerToken,
    pid: process.pid,
    processStartIdentity,
    startedAt: new Date().toISOString(),
    runDirectory,
  };
  const ownerTicket = acquireRunOwnerClaim(
    lockPath,
    ownerDirectory,
    runRoot,
    ownerClaimPath,
    ownerTicketPath,
    lockPayload,
  );
  let released = false;
  let handlersInstalled = false;
  let heartbeatFailure: Error | undefined;
  let heartbeatTimer: NodeJS.Timeout | undefined;
  const signalHandlers = new Map<NodeJS.Signals, () => void>();
  const exitHandler = (): void => {
    release();
  };

  let pinnedWorkerModuleUrl: URL;
  let pinnedManifest: readonly Readonly<{ path: string; sha256: string }>[];
  let workerRuntimeSha256: string;
  try {
    mkdirStrictDirectory(runDirectory);
    refreshRunHeartbeat();
    if (heartbeatFailure) throw heartbeatFailure;
    heartbeatTimer = setInterval(refreshRunHeartbeat, RUN_HEARTBEAT_INTERVAL_MS);
    heartbeatTimer.unref();
    mkdirStrictDirectory(runtimeDirectory);
    const firstSourceSnapshot = workerRuntimeSnapshot(options.sourceWorkerModuleUrl);
    publishRuntimeSnapshot(firstSourceSnapshot, runtimeDirectory);
    const secondSourceSnapshot = workerRuntimeSnapshot(options.sourceWorkerModuleUrl);
    if (runtimeSnapshotIdentity(firstSourceSnapshot) !== runtimeSnapshotIdentity(secondSourceSnapshot)) {
      throw new Error('Attempt-sampling worker bundle changed while it was being pinned');
    }
    const entryRelativePath = firstSourceSnapshot.find((module) =>
      module.path === basename(fileURLToPath(options.sourceWorkerModuleUrl)))?.path;
    if (!entryRelativePath) {
      throw new Error('Pinned attempt-sampling worker entry is absent from its runtime closure');
    }
    pinnedWorkerModuleUrl = pathToFileURL(resolve(runtimeDirectory, entryRelativePath));
    const pinnedSnapshot = workerRuntimeSnapshot(pinnedWorkerModuleUrl);
    if (runtimeSnapshotIdentity(firstSourceSnapshot) !== runtimeSnapshotIdentity(pinnedSnapshot)) {
      throw new Error('Pinned attempt-sampling worker bundle does not match its source closure');
    }
    pinnedManifest = runtimeManifest(pinnedSnapshot);
    workerRuntimeSha256 = sha256(JSON.stringify(pinnedManifest));
    refreshRunHeartbeat();
    if (heartbeatFailure) throw heartbeatFailure;
    assertCurrentRunLockOwnership();
  } catch (error) {
    release();
    throw error;
  }

  function assertWorkerRuntimeUnchanged(): void {
    if (heartbeatFailure) throw heartbeatFailure;
    assertCurrentRunLockOwnership();
    const current = workerRuntimeSnapshot(pinnedWorkerModuleUrl);
    const currentManifest = runtimeManifest(current);
    if (JSON.stringify(currentManifest) !== JSON.stringify(pinnedManifest)
      || sha256(JSON.stringify(currentManifest)) !== workerRuntimeSha256) {
      throw new Error('Pinned attempt-sampling worker runtime changed during the trainer run');
    }
    refreshRunHeartbeat();
    if (heartbeatFailure) throw heartbeatFailure;
    assertCurrentRunLockOwnership();
  }

  function installProcessCleanupHandlers(): void {
    if (handlersInstalled) return;
    handlersInstalled = true;
    process.once('exit', exitHandler);
    for (const signal of ['SIGINT', 'SIGTERM', 'SIGHUP'] as const) {
      const handler = (): void => {
        release();
        process.exit(signal === 'SIGINT' ? 130 : signal === 'SIGTERM' ? 143 : 129);
      };
      signalHandlers.set(signal, handler);
      process.once(signal, handler);
    }
  }

  function release(): void {
    if (released) return;
    released = true;
    if (heartbeatTimer) clearInterval(heartbeatTimer);
    if (handlersInstalled) {
      process.off('exit', exitHandler);
      for (const [signal, handler] of signalHandlers) process.off(signal, handler);
    }
    try {
      const current = readFinalizedRunOwnerClaim(ownerClaimPath, ownerTicketPath);
      if (current.owner.ownerToken === ownerToken
        && current.owner.pid === process.pid
        && current.owner.processStartIdentity === processStartIdentity
        && current.ticket === ownerTicket) {
        unlinkSync(ownerClaimPath);
        try {
          unlinkSync(ownerTicketPath);
        } catch (error) {
          if (!isNodeError(error, 'ENOENT')) throw error;
        }
        fsyncDirectory(ownerDirectory);
      }
    } catch (error) {
      if (!isNodeError(error, 'ENOENT')) throw error;
    } finally {
      try {
        removePrivateRunDirectory(runDirectory);
      } catch {
        // The exclusive lock is the safety boundary. A private immutable
        // runtime snapshot left by an abnormal cleanup is inert and ignored.
      }
    }
  }

  function refreshRunHeartbeat(): void {
    if (released || heartbeatFailure) return;
    try {
      assertCurrentRunLockOwnership();
      writeControlJsonAtomically(heartbeatPath, {
        schemaVersion: RUN_HEARTBEAT_SCHEMA_VERSION,
        ownerToken,
        pid: process.pid,
        processStartIdentity,
        updatedAt: new Date().toISOString(),
      } satisfies RunHeartbeatPayload);
    } catch (error) {
      heartbeatFailure = error instanceof Error
        ? error
        : new Error('Observable training run heartbeat failed');
      if (heartbeatTimer) clearInterval(heartbeatTimer);
    }
  }

  function assertCurrentRunLockOwnership(): void {
    const current = readFinalizedRunOwnerClaim(ownerClaimPath, ownerTicketPath);
    if (current.choosing
      || current.ticket !== ownerTicket
      || current.owner.ownerToken !== ownerToken
      || current.owner.pid !== process.pid
      || current.owner.processStartIdentity !== processStartIdentity
      || !runLockOwnerIsActive(current.owner, runRoot)) {
      throw new Error(
        'Observable training run lock ownership changed or its heartbeat lease expired',
      );
    }
  }

  return {
    runId,
    runDirectory,
    workerModuleUrl: pinnedWorkerModuleUrl,
    workerRuntimeSha256,
    assertWorkerRuntimeUnchanged,
    installProcessCleanupHandlers,
    release,
  };
}

export function openFreshSamplingRunJournal(options: {
  journalPath: string;
  runsRoot: string;
  compatibilitySha256: string;
}): FreshSamplingRunJournal {
  if (!/^[a-f0-9]{64}$/.test(options.compatibilitySha256)) {
    throw new Error('Fresh-sampling compatibility identity must be SHA-256');
  }
  const journalPath = resolve(options.journalPath);
  const runsRoot = resolve(options.runsRoot);
  mkdirStrictDirectory(dirname(journalPath));
  mkdirStrictDirectory(runsRoot);
  let existing: FreshSamplingJournalPayload | undefined;
  try {
    existing = validateFreshSamplingJournal(
      readControlJson<FreshSamplingJournalPayload>(journalPath),
    );
  } catch (error) {
    if (!isNodeError(error, 'ENOENT')) throw error;
  }
  const resumed = existing?.status === 'in-progress'
    && existing.compatibilitySha256 === options.compatibilitySha256;
  if (existing && !resumed) {
    pruneFreshSamplingRun(runsRoot, existing.runId);
  }
  const now = new Date().toISOString();
  let payload: FreshSamplingJournalPayload;
  if (resumed && existing) {
    payload = {
      ...existing,
      updatedAt: now,
    };
  } else {
    payload = {
      schemaVersion: FRESH_JOURNAL_SCHEMA_VERSION,
      status: 'in-progress',
      runId: randomUUID(),
      compatibilitySha256: options.compatibilitySha256,
      startedAt: now,
      updatedAt: now,
    };
  }
  writeControlJsonAtomically(journalPath, payload);
  const cacheRoot = resolve(runsRoot, payload.runId, 'cache');
  mkdirStrictDirectory(cacheRoot);

  return {
    runId: payload.runId,
    cacheRoot,
    resumed,
    markCompleted(): void {
      const current = validateFreshSamplingJournal(
        readControlJson<FreshSamplingJournalPayload>(journalPath),
      );
      if (current.status !== 'in-progress' || current.runId !== payload.runId
        || current.compatibilitySha256 !== payload.compatibilitySha256) {
        throw new Error('Fresh-sampling journal ownership changed before completion');
      }
      writeControlJsonAtomically(journalPath, {
        ...current,
        status: 'completed',
        updatedAt: new Date().toISOString(),
      });
      pruneFreshSamplingRun(runsRoot, payload.runId);
    },
  };
}

// Election invariant: every contender publishes `choosing` before selecting a
// Lamport ticket, then enters only when no active claim has a lower
// (ticket, ownerToken) pair. Stale cleanup unlinks only that owner's UUID path,
// which can never name a later owner, so reclamation cannot remove a successor.
// Choosing claims are immutable; tickets are separate immutable records. Once
// a choosing claim is reclaimed, a resumed process cannot recreate ownership.
// This protocol requires coherent local-filesystem directory reads plus atomic
// same-directory link and rename operations; network filesystems are unsupported.
function acquireRunOwnerClaim(
  legacyLockPath: string,
  ownerDirectory: string,
  runRoot: string,
  ownerClaimPath: string,
  ownerTicketPath: string,
  payload: RunLockPayload,
): number {
  const choosingClaim: RunOwnerClaimPayload = {
    schemaVersion: RUN_OWNER_CLAIM_SCHEMA_VERSION,
    choosing: true,
    ticket: null,
    owner: payload,
  };
  try {
    writeControlJsonExclusively(ownerClaimPath, choosingClaim);
    assertNoActiveLegacyRunLock(legacyLockPath, runRoot);
    const ticket = nextRunOwnerTicket(ownerDirectory, runRoot, payload.ownerToken);
    writeControlJsonExclusively(ownerTicketPath, {
      schemaVersion: RUN_OWNER_TICKET_SCHEMA_VERSION,
      ownerToken: payload.ownerToken,
      ticket,
    } satisfies RunOwnerTicketPayload);
    const ownClaim: RunOwnerClaimPayload = {
      ...choosingClaim,
      choosing: false,
      ticket,
    };
    assertChoosingRunOwnerClaim(ownerClaimPath, choosingClaim);
    electRunOwner(ownerDirectory, runRoot, ownClaim);
    const current = readFinalizedRunOwnerClaim(ownerClaimPath, ownerTicketPath);
    if (current.choosing
      || current.ticket !== ticket
      || current.owner.ownerToken !== payload.ownerToken
      || current.owner.pid !== payload.pid
      || current.owner.processStartIdentity !== payload.processStartIdentity) {
      throw new Error('Observable training run owner claim changed during acquisition');
    }
    return ticket;
  } catch (error) {
    try {
      unlinkSync(ownerClaimPath);
      fsyncDirectory(ownerDirectory);
    } catch (cleanupError) {
      if (!isNodeError(cleanupError, 'ENOENT')) throw cleanupError;
    }
    try {
      unlinkSync(ownerTicketPath);
      fsyncDirectory(ownerDirectory);
    } catch (cleanupError) {
      if (!isNodeError(cleanupError, 'ENOENT')) throw cleanupError;
    }
    throw error;
  }
}

function nextRunOwnerTicket(
  ownerDirectory: string,
  runRoot: string,
  ownOwnerToken: string,
): number {
  let maximumTicket = 0;
  for (const candidate of readOtherRunOwnerClaims(
    ownerDirectory,
    runRoot,
    ownOwnerToken,
  )) {
    if (!candidate.claim.choosing && candidate.claim.ticket !== null) {
      maximumTicket = Math.max(maximumTicket, candidate.claim.ticket);
    }
  }
  if (!Number.isSafeInteger(maximumTicket + 1)) {
    throw new Error('Observable training run owner ticket space is exhausted');
  }
  return maximumTicket + 1;
}

function electRunOwner(
  ownerDirectory: string,
  runRoot: string,
  ownClaim: RunOwnerClaimPayload,
): void {
  const deadline = process.hrtime.bigint() + 2_000_000_000n;
  while (true) {
    let anotherOwnerIsChoosing = false;
    for (const candidate of readOtherRunOwnerClaims(
      ownerDirectory,
      runRoot,
      ownClaim.owner.ownerToken,
    )) {
      if (candidate.claim.choosing) {
        anotherOwnerIsChoosing = true;
        continue;
      }
      if (runOwnerClaimPrecedes(candidate.claim, ownClaim)) {
        throw new Error(
          `Observable classifier trainer is already running as PID ${candidate.claim.owner.pid} since ${candidate.claim.owner.startedAt}`,
        );
      }
    }
    if (!anotherOwnerIsChoosing) return;
    if (process.hrtime.bigint() >= deadline) {
      throw new Error('Another observable classifier trainer is still choosing its owner ticket');
    }
    waitForRunOwnerElection();
  }
}

function readOtherRunOwnerClaims(
  ownerDirectory: string,
  runRoot: string,
  ownOwnerToken: string,
): Array<{ path: string; claim: RunOwnerClaimPayload }> {
  removeOrphanRunOwnerTickets(ownerDirectory, ownOwnerToken);
  const activeClaims: Array<{ path: string; claim: RunOwnerClaimPayload }> = [];
  for (const entry of readdirSync(ownerDirectory, { withFileTypes: true })) {
    const match = RUN_OWNER_CLAIM_FILENAME_PATTERN.exec(entry.name);
    if (!match) continue;
    if (!entry.isFile()) {
      throw new Error(`Observable training owner claim is not a regular file: ${entry.name}`);
    }
    const candidateOwnerToken = match[1]!;
    if (candidateOwnerToken === ownOwnerToken) continue;
    const candidatePath = runOwnerClaimPath(ownerDirectory, candidateOwnerToken);
    let candidate: RunOwnerClaimPayload | undefined;
    let candidateBytes: Buffer;
    try {
      candidateBytes = readBoundedRegularFile(candidatePath, MAX_CONTROL_FILE_BYTES);
    } catch (error) {
      if (isNodeError(error, 'ENOENT')) continue;
      throw error;
    }
    try {
      candidate = validateRunOwnerClaim(
        JSON.parse(candidateBytes.toString('utf8')) as RunOwnerClaimPayload,
      );
      if (!candidate.choosing
        || candidate.ticket !== null
        || candidate.owner.ownerToken !== candidateOwnerToken) {
        throw new Error('Observable training owner claim filename does not match its token');
      }
    } catch {
      candidate = undefined;
    }
    if (candidate) {
      const ticket = readRunOwnerTicket(
        runOwnerTicketPath(ownerDirectory, candidateOwnerToken),
        candidateOwnerToken,
      );
      if (ticket !== undefined) candidate = { ...candidate, choosing: false, ticket };
    }
    if (!candidate || !runLockOwnerIsActive(candidate.owner, runRoot)) {
      removeStaleOwnerClaim(
        candidatePath,
        runOwnerTicketPath(ownerDirectory, candidateOwnerToken),
        ownerDirectory,
        runRoot,
        candidate?.owner,
      );
      continue;
    }
    activeClaims.push({ path: candidatePath, claim: candidate });
  }
  return activeClaims;
}

function removeOrphanRunOwnerTickets(
  ownerDirectory: string,
  ownOwnerToken: string,
): void {
  for (const entry of readdirSync(ownerDirectory, { withFileTypes: true })) {
    const match = RUN_OWNER_TICKET_FILENAME_PATTERN.exec(entry.name);
    if (!match) continue;
    if (!entry.isFile()) {
      throw new Error(`Observable training owner ticket is not a regular file: ${entry.name}`);
    }
    const ownerToken = match[1]!;
    if (ownerToken === ownOwnerToken) continue;
    try {
      lstatSync(runOwnerClaimPath(ownerDirectory, ownerToken));
      continue;
    } catch (error) {
      if (!isNodeError(error, 'ENOENT')) throw error;
    }
    try {
      unlinkSync(runOwnerTicketPath(ownerDirectory, ownerToken));
      fsyncDirectory(ownerDirectory);
    } catch (error) {
      if (!isNodeError(error, 'ENOENT')) throw error;
    }
  }
}

function runOwnerClaimPrecedes(
  left: RunOwnerClaimPayload,
  right: RunOwnerClaimPayload,
): boolean {
  if (left.ticket === null || right.ticket === null) {
    throw new Error('Observable training owner election compared an unchosen ticket');
  }
  return left.ticket < right.ticket
    || (left.ticket === right.ticket
      && left.owner.ownerToken < right.owner.ownerToken);
}

function removeStaleOwnerClaim(
  candidatePath: string,
  candidateTicketPath: string,
  ownerDirectory: string,
  runRoot: string,
  candidate: RunLockPayload | undefined,
): void {
  try {
    unlinkSync(candidatePath);
    fsyncDirectory(ownerDirectory);
  } catch (error) {
    if (!isNodeError(error, 'ENOENT')) throw error;
    return;
  }
  try {
    unlinkSync(candidateTicketPath);
  } catch (error) {
    if (!isNodeError(error, 'ENOENT')) throw error;
  }
  if (candidate && runDirectoryIsBoundToOwner(candidate, runRoot)) {
    try {
      removePrivateRunDirectory(candidate.runDirectory);
    } catch {
      // A stale private runtime directory is harmless and never selected by
      // a later content-addressed run.
    }
  }
}

function waitForRunOwnerElection(): void {
  const waitState = new Int32Array(new SharedArrayBuffer(Int32Array.BYTES_PER_ELEMENT));
  Atomics.wait(waitState, 0, 0, 5);
}

function assertNoActiveLegacyRunLock(path: string, runRoot: string): void {
  let beforeRead: ReturnType<typeof lstatSync>;
  try {
    beforeRead = lstatSync(path);
  } catch (error) {
    if (isNodeError(error, 'ENOENT')) return;
    throw error;
  }
  if (!beforeRead.isFile() || beforeRead.isSymbolicLink()) {
    throw new Error(`Legacy observable training lock is not a regular file: ${path}`);
  }
  let bytes: Buffer;
  try {
    bytes = readBoundedRegularFile(path, MAX_CONTROL_FILE_BYTES);
  } catch (error) {
    if (isNodeError(error, 'ENOENT')) return;
    throw error;
  }
  let afterRead: ReturnType<typeof lstatSync>;
  try {
    afterRead = lstatSync(path);
  } catch (error) {
    if (isNodeError(error, 'ENOENT')) return;
    throw error;
  }
  if (!afterRead.isFile()
    || afterRead.isSymbolicLink()
    || !sameFileIdentity(beforeRead, afterRead)
    || beforeRead.size !== afterRead.size
    || beforeRead.mtimeMs !== afterRead.mtimeMs
    || beforeRead.ctimeMs !== afterRead.ctimeMs) {
    throw new Error('Legacy observable training lock changed while it was inspected');
  }
  let activeOwner: Pick<RunLockPayload, 'pid' | 'startedAt'> | undefined;
  try {
    const parsed = JSON.parse(bytes.toString('utf8')) as unknown;
    if (!isPlainRecord(parsed)) throw new Error('Legacy observable training lock is malformed');
    if (parsed.schemaVersion === RUN_LOCK_SCHEMA_VERSION) {
      const lock = validateRunLock(parsed as RunLockPayload);
      if (runLockOwnerIsActive(lock, runRoot, false)) {
        activeOwner = lock;
      }
    } else if (parsed.schemaVersion === 1) {
      const lock = validateLegacyRunLock(parsed as LegacyRunLockPayload);
      if (legacyRunLockOwnerIsActive(lock)) {
        activeOwner = lock;
      }
    } else {
      throw new Error('Legacy observable training lock uses an unsupported schema');
    }
  } catch (error) {
    const now = Date.now();
    if (afterRead.mtimeMs > now + MAX_HEARTBEAT_CLOCK_SKEW_MS
      || now - afterRead.mtimeMs < MALFORMED_LEGACY_LOCK_GRACE_MS) {
      throw new Error(
        'Malformed legacy observable training lock is still within its stability grace period',
        { cause: error },
      );
    }
    quarantineLegacyRunLockSnapshot(path, bytes);
    return;
  }
  if (activeOwner) {
    throw new Error(
      `Observable classifier trainer is already running as PID ${activeOwner.pid} since ${activeOwner.startedAt}`,
    );
  }
}

function quarantineLegacyRunLockSnapshot(path: string, bytes: Buffer): void {
  const quarantinePath = resolve(
    dirname(path),
    `${basename(path)}.quarantine.${sha256(bytes)}.json`,
  );
  try {
    const descriptor = openSync(quarantinePath, 'wx', 0o600);
    try {
      writeFileSync(descriptor, bytes);
      fsyncSync(descriptor);
    } finally {
      closeSync(descriptor);
    }
    fsyncDirectory(dirname(path));
  } catch (error) {
    if (!isNodeError(error, 'EEXIST')) throw error;
  }
}

function workerRuntimeSnapshot(workerModuleUrl: URL): readonly RuntimeModule[] {
  if (workerModuleUrl.protocol !== 'file:') {
    throw new Error('Attempt-sampling worker runtime must use file URLs');
  }
  const runtimeRoot = dirname(fileURLToPath(workerModuleUrl));
  const entryPath = resolve(fileURLToPath(workerModuleUrl));
  const pending = [entryPath];
  const visited = new Map<string, RuntimeModule>();
  while (pending.length > 0) {
    if (visited.size >= MAX_RUNTIME_MODULE_COUNT) {
      throw new Error('Attempt-sampling worker runtime closure exceeds its module-count limit');
    }
    const path = resolve(pending.pop()!);
    if (visited.has(path)) continue;
    const relativePath = relative(runtimeRoot, path).split('\\').join('/');
    if (isAbsolute(relativePath)
      || relativePath === '..'
      || relativePath.startsWith('../')) {
      throw new Error(`Attempt-sampling worker imports outside its runtime directory: ${path}`);
    }
    const bytes = readBoundedRegularFile(path, MAX_RUNTIME_MODULE_BYTES);
    const module: RuntimeModule = {
      path: relativePath || basename(path),
      bytes,
      sha256: sha256(bytes),
    };
    visited.set(path, module);
    const source = bytes.toString('utf8');
    for (const specifier of localModuleSpecifiers(source)) {
      const importedUrl = new URL(specifier, pathToFileURL(path));
      if (importedUrl.protocol === 'file:') pending.push(fileURLToPath(importedUrl));
    }
  }
  return [...visited.values()].sort((left, right) => left.path.localeCompare(right.path));
}

function publishRuntimeSnapshot(
  modules: readonly RuntimeModule[],
  runtimeDirectory: string,
): void {
  for (const module of modules) {
    const output = resolve(runtimeDirectory, module.path);
    const relativeOutput = relative(runtimeDirectory, output);
    if (isAbsolute(relativeOutput)
      || relativeOutput === ''
      || relativeOutput === '..'
      || relativeOutput.startsWith('../')
      || relativeOutput.startsWith('..\\')) {
      throw new Error(`Pinned worker module escapes its runtime directory: ${module.path}`);
    }
    mkdirStrictDirectory(dirname(output));
    const descriptor = openSync(output, 'wx', 0o400);
    try {
      writeFileSync(descriptor, module.bytes);
      fsyncSync(descriptor);
    } finally {
      closeSync(descriptor);
    }
  }
  fsyncDirectory(runtimeDirectory);
}

function runtimeManifest(
  modules: readonly RuntimeModule[],
): readonly Readonly<{ path: string; sha256: string }>[] {
  return modules.map((module) => ({ path: module.path, sha256: module.sha256 }));
}

function runtimeSnapshotIdentity(modules: readonly RuntimeModule[]): string {
  return JSON.stringify(runtimeManifest(modules));
}

function validateRunLock(value: RunLockPayload): RunLockPayload {
  if (!isPlainRecord(value)
    || value.schemaVersion !== RUN_LOCK_SCHEMA_VERSION
    || typeof value.ownerToken !== 'string'
    || !RUN_OWNER_TOKEN_PATTERN.test(value.ownerToken)
    || !Number.isSafeInteger(value.pid) || value.pid <= 0
    || (value.processStartIdentity !== null
      && (typeof value.processStartIdentity !== 'string'
        || value.processStartIdentity.length === 0
        || value.processStartIdentity.length > 1_024))
    || typeof value.startedAt !== 'string'
    || !Number.isFinite(Date.parse(value.startedAt))
    || typeof value.runDirectory !== 'string') {
    throw new Error('Observable training run lock is malformed');
  }
  return value;
}

function validateRunOwnerClaim(value: RunOwnerClaimPayload): RunOwnerClaimPayload {
  if (!isPlainRecord(value)
    || value.schemaVersion !== RUN_OWNER_CLAIM_SCHEMA_VERSION
    || typeof value.choosing !== 'boolean'
    || (value.ticket !== null
      && (!Number.isSafeInteger(value.ticket) || value.ticket <= 0))
    || (value.choosing && value.ticket !== null)
    || (!value.choosing && value.ticket === null)
    || !isPlainRecord(value.owner)) {
    throw new Error('Observable training run owner claim is malformed');
  }
  validateRunLock(value.owner);
  return value;
}

function validateRunOwnerTicket(value: RunOwnerTicketPayload): RunOwnerTicketPayload {
  if (!isPlainRecord(value)
    || value.schemaVersion !== RUN_OWNER_TICKET_SCHEMA_VERSION
    || typeof value.ownerToken !== 'string'
    || !RUN_OWNER_TOKEN_PATTERN.test(value.ownerToken)
    || !Number.isSafeInteger(value.ticket)
    || value.ticket <= 0) {
    throw new Error('Observable training run owner ticket is malformed');
  }
  return value;
}

function validateLegacyRunLock(value: LegacyRunLockPayload): LegacyRunLockPayload {
  if (!isPlainRecord(value)
    || value.schemaVersion !== 1
    || typeof value.ownerToken !== 'string'
    || !RUN_OWNER_TOKEN_PATTERN.test(value.ownerToken)
    || !Number.isSafeInteger(value.pid) || value.pid <= 0
    || typeof value.startedAt !== 'string'
    || !Number.isFinite(Date.parse(value.startedAt))
    || typeof value.runDirectory !== 'string') {
    throw new Error('Legacy observable training run lock is malformed');
  }
  if (Date.parse(value.startedAt) > Date.now() + MAX_HEARTBEAT_CLOCK_SKEW_MS) {
    throw new Error('Legacy observable training run lock starts implausibly far in the future');
  }
  return value;
}

function validateRunHeartbeat(value: RunHeartbeatPayload): RunHeartbeatPayload {
  if (!isPlainRecord(value)
    || value.schemaVersion !== RUN_HEARTBEAT_SCHEMA_VERSION
    || typeof value.ownerToken !== 'string'
    || !RUN_OWNER_TOKEN_PATTERN.test(value.ownerToken)
    || !Number.isSafeInteger(value.pid) || value.pid <= 0
    || (value.processStartIdentity !== null
      && (typeof value.processStartIdentity !== 'string'
        || value.processStartIdentity.length === 0
        || value.processStartIdentity.length > 1_024))
    || typeof value.updatedAt !== 'string'
    || !Number.isFinite(Date.parse(value.updatedAt))) {
    throw new Error('Observable training run heartbeat is malformed');
  }
  return value;
}

function validateFreshSamplingJournal(
  value: FreshSamplingJournalPayload,
): FreshSamplingJournalPayload {
  if (!isPlainRecord(value)
    || value.schemaVersion !== FRESH_JOURNAL_SCHEMA_VERSION
    || (value.status !== 'in-progress' && value.status !== 'completed')
    || typeof value.runId !== 'string'
    || !/^[0-9a-f-]{36}$/.test(value.runId)
    || typeof value.compatibilitySha256 !== 'string'
    || !/^[a-f0-9]{64}$/.test(value.compatibilitySha256)
    || typeof value.startedAt !== 'string'
    || typeof value.updatedAt !== 'string') {
    throw new Error('Fresh-sampling journal is malformed');
  }
  return value;
}

function processIsAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return !isNodeError(error, 'ESRCH');
  }
}

function runLockOwnerIsActive(
  lock: RunLockPayload,
  runRoot: string,
  requireBoundRunDirectory = true,
): boolean {
  const observedProcessStartIdentity = readProcessStartIdentity(lock.pid);
  if (lock.processStartIdentity !== null
    && observedProcessStartIdentity !== undefined
    && lock.processStartIdentity !== observedProcessStartIdentity) {
    return false;
  }
  if (!processIsAlive(lock.pid)) return false;
  if (requireBoundRunDirectory
    ? !runDirectoryIsBoundToOwner(lock, runRoot)
    : !isPathStrictlyWithinRoot(runRoot, lock.runDirectory)) return false;
  const heartbeatPath = resolve(lock.runDirectory, RUN_HEARTBEAT_FILENAME);
  let heartbeatBytes: Buffer;
  try {
    heartbeatBytes = readBoundedRegularFile(heartbeatPath, MAX_CONTROL_FILE_BYTES);
  } catch (error) {
    if (isNodeError(error, 'ENOENT')) return runLockStartupGraceIsActive(lock);
    throw error;
  }
  let heartbeat: RunHeartbeatPayload;
  try {
    heartbeat = validateRunHeartbeat(JSON.parse(
      heartbeatBytes.toString('utf8'),
    ) as RunHeartbeatPayload);
  } catch (error) {
    throw new Error(
      'Observable training run heartbeat is malformed and cannot be safely expired',
      { cause: error },
    );
  }
  if (heartbeat.ownerToken !== lock.ownerToken
    || heartbeat.pid !== lock.pid
    || heartbeat.processStartIdentity !== lock.processStartIdentity) {
    throw new Error(
      'Observable training run heartbeat identity does not match its live owner claim',
    );
  }
  const updatedAt = Date.parse(heartbeat.updatedAt);
  const now = Date.now();
  return updatedAt <= now + MAX_HEARTBEAT_CLOCK_SKEW_MS
    && now - updatedAt <= RUN_HEARTBEAT_STALE_AFTER_MS;
}

function runLockStartupGraceIsActive(lock: Pick<RunLockPayload, 'startedAt'>): boolean {
  const startedAt = Date.parse(lock.startedAt);
  const now = Date.now();
  return Number.isFinite(startedAt)
    && startedAt <= now + MAX_HEARTBEAT_CLOCK_SKEW_MS
    && now - startedAt <= RUN_HEARTBEAT_STALE_AFTER_MS;
}

function runDirectoryIsBoundToOwner(
  lock: Pick<RunLockPayload, 'ownerToken' | 'runDirectory'>,
  runRoot: string,
): boolean {
  const expectedRunDirectory = resolve(runRoot, lock.ownerToken);
  return isPathStrictlyWithinRoot(runRoot, expectedRunDirectory)
    && resolve(lock.runDirectory) === expectedRunDirectory;
}

function legacyRunLockOwnerIsActive(lock: LegacyRunLockPayload): boolean {
  if (!processIsAlive(lock.pid)) return false;
  const startedAt = Date.parse(lock.startedAt);
  const now = Date.now();
  if (!Number.isFinite(startedAt)) return false;
  const age = now - startedAt;
  if (age > LEGACY_LOCK_FALLBACK_MAX_AGE_MS) return false;
  const observedProcessStartedAt = readProcessStartedAtMs(lock.pid);
  if (observedProcessStartedAt !== undefined) {
    return observedProcessStartedAt <= startedAt + PROCESS_START_COMPARISON_TOLERANCE_MS;
  }
  return age >= -MAX_HEARTBEAT_CLOCK_SKEW_MS;
}

function readProcessStartedAtMs(pid: number): number | undefined {
  if (!Number.isSafeInteger(pid) || pid <= 0) return undefined;
  try {
    const startedAt = execFileSync(
      '/bin/ps',
      ['-p', String(pid), '-o', 'lstart='],
      {
        encoding: 'utf8',
        stdio: ['ignore', 'pipe', 'ignore'],
        timeout: 1_000,
      },
    ).trim();
    const timestamp = Date.parse(startedAt);
    return Number.isFinite(timestamp) ? timestamp : undefined;
  } catch {
    return undefined;
  }
}

function readProcessStartIdentity(pid: number): string | undefined {
  if (!Number.isSafeInteger(pid) || pid <= 0) return undefined;
  try {
    if (platform() === 'linux') {
      const stat = readFileSync(`/proc/${pid}/stat`, 'utf8');
      const commandEnd = stat.lastIndexOf(') ');
      if (commandEnd < 0) return undefined;
      const fieldsAfterCommand = stat.slice(commandEnd + 2).trim().split(/\s+/);
      const startTimeClockTicks = fieldsAfterCommand[19];
      if (!startTimeClockTicks || !/^\d+$/.test(startTimeClockTicks)) return undefined;
      const bootId = readFileSync('/proc/sys/kernel/random/boot_id', 'utf8').trim();
      if (!bootId) return undefined;
      return `linux:${bootId}:${startTimeClockTicks}`;
    }
    if (platform() === 'darwin') {
      const startedAt = execFileSync(
        '/bin/ps',
        ['-p', String(pid), '-o', 'lstart='],
        {
          encoding: 'utf8',
          stdio: ['ignore', 'pipe', 'ignore'],
          timeout: 1_000,
        },
      ).trim();
      return startedAt ? `darwin:${startedAt}` : undefined;
    }
  } catch {
    return undefined;
  }
  return undefined;
}

function readControlJson<T>(path: string): T {
  return JSON.parse(readBoundedRegularFile(path, MAX_CONTROL_FILE_BYTES).toString('utf8')) as T;
}

function assertChoosingRunOwnerClaim(
  claimPath: string,
  expected: RunOwnerClaimPayload,
): void {
  const current = validateRunOwnerClaim(
    readControlJson<RunOwnerClaimPayload>(claimPath),
  );
  if (!current.choosing
    || current.ticket !== null
    || JSON.stringify(current.owner) !== JSON.stringify(expected.owner)) {
    throw new Error('Observable training choosing claim changed before ticket publication');
  }
}

function readFinalizedRunOwnerClaim(
  claimPath: string,
  ticketPath: string,
): RunOwnerClaimPayload {
  const choosingClaim = validateRunOwnerClaim(
    readControlJson<RunOwnerClaimPayload>(claimPath),
  );
  if (!choosingClaim.choosing || choosingClaim.ticket !== null) {
    throw new Error('Observable training choosing claim is not immutable');
  }
  const ticket = readRunOwnerTicket(ticketPath, choosingClaim.owner.ownerToken);
  if (ticket === undefined) {
    throw new Error('Observable training owner ticket is missing or malformed');
  }
  return { ...choosingClaim, choosing: false, ticket };
}

function readRunOwnerTicket(
  ticketPath: string,
  expectedOwnerToken: string,
): number | undefined {
  let bytes: Buffer;
  try {
    bytes = readBoundedRegularFile(ticketPath, MAX_CONTROL_FILE_BYTES);
  } catch (error) {
    if (isNodeError(error, 'ENOENT')) return undefined;
    throw error;
  }
  try {
    const ticket = validateRunOwnerTicket(
      JSON.parse(bytes.toString('utf8')) as RunOwnerTicketPayload,
    );
    return ticket.ownerToken === expectedOwnerToken ? ticket.ticket : undefined;
  } catch {
    return undefined;
  }
}

function writeControlJsonExclusively(path: string, value: unknown): void {
  const bytes = Buffer.from(JSON.stringify(value), 'utf8');
  if (bytes.length > MAX_CONTROL_FILE_BYTES) {
    throw new Error(`Observable training control file exceeds ${MAX_CONTROL_FILE_BYTES} bytes`);
  }
  const temporaryPath = `${path}.${process.pid}.${randomUUID()}.tmp`;
  const descriptor = openSync(temporaryPath, 'wx', 0o600);
  try {
    writeFileSync(descriptor, bytes);
    fsyncSync(descriptor);
  } finally {
    closeSync(descriptor);
  }
  try {
    linkSync(temporaryPath, path);
    unlinkSync(temporaryPath);
    fsyncDirectory(dirname(path));
  } catch (error) {
    try {
      unlinkSync(temporaryPath);
    } catch (cleanupError) {
      if (!isNodeError(cleanupError, 'ENOENT')) throw cleanupError;
    }
    throw error;
  }
}

function writeControlJsonAtomically(path: string, value: unknown): void {
  const bytes = Buffer.from(JSON.stringify(value), 'utf8');
  if (bytes.length > MAX_CONTROL_FILE_BYTES) {
    throw new Error(`Observable training control file exceeds ${MAX_CONTROL_FILE_BYTES} bytes`);
  }
  const temporaryPath = `${path}.${process.pid}.${randomUUID()}.tmp`;
  const descriptor = openSync(temporaryPath, 'wx', 0o600);
  try {
    writeFileSync(descriptor, bytes);
    fsyncSync(descriptor);
  } finally {
    closeSync(descriptor);
  }
  renameSync(temporaryPath, path);
  fsyncDirectory(dirname(path));
}

function runOwnerClaimPath(ownerDirectory: string, ownerToken: string): string {
  if (!RUN_OWNER_TOKEN_PATTERN.test(ownerToken)) {
    throw new Error('Observable training owner token is malformed');
  }
  const claimPath = resolve(ownerDirectory, `${ownerToken}.json`);
  if (!isPathStrictlyWithinRoot(ownerDirectory, claimPath)) {
    throw new Error('Observable training owner claim escapes its owner directory');
  }
  return claimPath;
}

function runOwnerTicketPath(ownerDirectory: string, ownerToken: string): string {
  if (!RUN_OWNER_TOKEN_PATTERN.test(ownerToken)) {
    throw new Error('Observable training owner token is malformed');
  }
  const ticketPath = resolve(ownerDirectory, `${ownerToken}.ticket.json`);
  if (!isPathStrictlyWithinRoot(ownerDirectory, ticketPath)) {
    throw new Error('Observable training owner ticket escapes its owner directory');
  }
  return ticketPath;
}

function readBoundedRegularFile(path: string, maximumBytes: number): Buffer {
  const noFollowFlag = typeof constants.O_NOFOLLOW === 'number'
    ? constants.O_NOFOLLOW
    : undefined;
  const beforeOpen = noFollowFlag === undefined ? lstatSync(path) : undefined;
  if (beforeOpen && (!beforeOpen.isFile() || beforeOpen.isSymbolicLink())) {
    throw new Error(`Observable training control path is not a bounded regular file: ${path}`);
  }
  const descriptor = openSync(path, constants.O_RDONLY | (noFollowFlag ?? 0));
  try {
    const status = fstatSync(descriptor);
    if (!status.isFile() || status.size > maximumBytes) {
      throw new Error(`Observable training control path is not a bounded regular file: ${path}`);
    }
    if (beforeOpen) {
      const afterOpen = lstatSync(path);
      if (!afterOpen.isFile()
        || afterOpen.isSymbolicLink()
        || !sameFileIdentity(beforeOpen, status)
        || !sameFileIdentity(afterOpen, status)) {
        throw new Error(`Observable training control path changed during secure open: ${path}`);
      }
    }
    const output = Buffer.allocUnsafe(status.size);
    let offset = 0;
    while (offset < output.length) {
      const bytesRead = readSync(descriptor, output, offset, output.length - offset, offset);
      if (bytesRead === 0) throw new Error(`Observable training control file truncated: ${path}`);
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

function isPathStrictlyWithinRoot(root: string, candidate: string): boolean {
  const relativePath = relative(resolve(root), resolve(candidate));
  return relativePath !== ''
    && !isAbsolute(relativePath)
    && relativePath !== '..'
    && !relativePath.startsWith('../')
    && !relativePath.startsWith('..\\');
}

function mkdirStrictDirectory(path: string): void {
  mkdirSync(path, { recursive: true, mode: 0o700 });
  const status = lstatSync(path);
  if (!status.isDirectory() || status.isSymbolicLink()) {
    throw new Error(`Observable training path must be a non-symlink directory: ${path}`);
  }
}

function fsyncDirectory(path: string): void {
  // Windows/NTFS rejects fsync on a directory handle (EPERM); directory-entry
  // durability there is a platform guarantee this call cannot add to.
  // Uses process.platform (not the os.platform() import) because this module's
  // own tests mock node:os to force darwin-branch behavior in liveness checks.
  if (process.platform === 'win32') return;
  const descriptor = openSync(path, 'r');
  try {
    fsyncSync(descriptor);
  } finally {
    closeSync(descriptor);
  }
}

function removePrivateRunDirectory(path: string): void {
  let status: ReturnType<typeof lstatSync>;
  try {
    status = lstatSync(path);
  } catch (error) {
    if (isNodeError(error, 'ENOENT')) return;
    throw error;
  }
  if (status.isSymbolicLink() || !status.isDirectory()) {
    unlinkSync(path);
    return;
  }
  chmodSync(path, 0o700);
  rmSync(path, { recursive: true, force: true });
}

function pruneFreshSamplingRun(runsRoot: string, runId: string): void {
  const runDirectory = resolve(runsRoot, runId);
  if (!isPathStrictlyWithinRoot(runsRoot, runDirectory)) return;
  try {
    removePrivateRunDirectory(runDirectory);
  } catch {
    // The completed or incompatible journal is already durably unreachable.
    // Retention cleanup is best-effort and retried when that journal is next
    // observed before it is replaced.
  }
}

function localModuleSpecifiers(source: string): string[] {
  const patterns = [
    /\bfrom\s*["'](\.[^"']+)["']/g,
    /\bimport\s*["'](\.[^"']+)["']/g,
    /\bimport\s*\(\s*["'](\.[^"']+)["']\s*\)/g,
  ];
  return [...new Set(patterns.flatMap((pattern) =>
    [...source.matchAll(pattern)].map((match) => match[1]!)))];
}

function sha256(value: string | Buffer): string {
  return createHash('sha256').update(value).digest('hex');
}

function isPlainRecord(value: unknown): value is Record<string, any> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function isNodeError(error: unknown, code: string): error is NodeJS.ErrnoException {
  return error instanceof Error && 'code' in error && error.code === code;
}
