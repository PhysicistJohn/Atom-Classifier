import { strict as assert } from 'node:assert';
import { execFileSync, spawn } from 'node:child_process';
import {
  chmodSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  symlinkSync,
  utimesSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { acquireObservableTrainingRun } from './observable-training-run-control.js';

const MOCK_PROCESS_START = 'Thu Jul 16 10:00:00 2026';
const CURRENT_PROCESS_START_IDENTITY = `darwin:${MOCK_PROCESS_START}`;
const STALE_OWNER_TOKEN = '00000000-0000-4000-8000-000000000001';
const ACTIVE_OWNER_TOKEN = '00000000-0000-4000-8000-000000000002';
const LEGACY_OWNER_TOKEN = '00000000-0000-4000-8000-000000000003';

vi.mock('node:child_process', async (importOriginal) => {
  const actual = await importOriginal<typeof import('node:child_process')>();
  return {
    ...actual,
    execFileSync: vi.fn(() => MOCK_PROCESS_START),
  };
});

vi.mock('node:os', async (importOriginal) => {
  const actual = await importOriginal<typeof import('node:os')>();
  return {
    ...actual,
    platform: vi.fn(() => 'darwin'),
  };
});

interface TestRunLockPayload {
  schemaVersion: 2;
  ownerToken: string;
  pid: number;
  processStartIdentity: string | null;
  startedAt: string;
  runDirectory: string;
}

interface TestRunOwnerClaimPayload {
  schemaVersion: 1;
  choosing: boolean;
  ticket: number | null;
  owner: TestRunLockPayload;
}

interface TestRunOwnerTicketPayload {
  schemaVersion: 1;
  ownerToken: string;
  ticket: number;
}

interface TestRunHeartbeatPayload {
  schemaVersion: 1;
  ownerToken: string;
  pid: number;
  processStartIdentity: string | null;
  updatedAt: string;
}

afterEach(() => {
  vi.useRealTimers();
  vi.mocked(execFileSync).mockImplementation(() => MOCK_PROCESS_START);
});

describe('observable-training run control', () => {
  it('rejects a concurrent owner with a fresh heartbeat and finalized ticket', () => {
    const fixture = createRunControlFixture();
    const first = acquireObservableTrainingRun(fixture.options);
    try {
      const [claim] = readOwnerClaims(fixture);
      expect(claim).toMatchObject({
        schemaVersion: 1,
        choosing: false,
        ticket: 1,
        owner: {
          schemaVersion: 2,
          pid: process.pid,
          runDirectory: first.runDirectory,
          processStartIdentity: CURRENT_PROCESS_START_IDENTITY,
        },
      });
      const heartbeat = readHeartbeat(first.runDirectory);
      expect(heartbeat).toMatchObject({
        schemaVersion: 1,
        ownerToken: claim!.owner.ownerToken,
        pid: claim!.owner.pid,
        processStartIdentity: claim!.owner.processStartIdentity,
      });
      expect(() => acquireObservableTrainingRun(fixture.options))
        .toThrow(/already running/);
      expect(readOwnerClaims(fixture)).toHaveLength(1);
    } finally {
      first.release();
      fixture.cleanup();
    }
  });

  it('reclaims a forged live PID when its process-start identity mismatches, even with a fresh heartbeat', () => {
    const fixture = createRunControlFixture();
    const stale = forgeOwnerClaim(fixture, {
      ownerToken: STALE_OWNER_TOKEN,
      processStartIdentity: 'darwin:unrelated-reused-pid',
      heartbeatUpdatedAt: new Date().toISOString(),
    });
    let recovered: ReturnType<typeof acquireObservableTrainingRun> | undefined;
    try {
      recovered = acquireObservableTrainingRun(fixture.options);
      expect(existsSync(stale.claimPath)).toBe(false);
      expect(existsSync(stale.sentinelPath)).toBe(false);
    } finally {
      recovered?.release();
      fixture.cleanup();
    }
  });

  it('reclaims a live PID through the heartbeat fallback when identity is unavailable and the lease expired', () => {
    const fixture = createRunControlFixture();
    const stale = forgeOwnerClaim(fixture, {
      ownerToken: STALE_OWNER_TOKEN,
      processStartIdentity: null,
      heartbeatUpdatedAt: new Date(0).toISOString(),
    });
    let recovered: ReturnType<typeof acquireObservableTrainingRun> | undefined;
    try {
      recovered = acquireObservableTrainingRun(fixture.options);
      expect(existsSync(stale.claimPath)).toBe(false);
    } finally {
      recovered?.release();
      fixture.cleanup();
    }
  });

  it('keeps a live fallback owner while its heartbeat lease is fresh', () => {
    const fixture = createRunControlFixture();
    const active = forgeOwnerClaim(fixture, {
      ownerToken: ACTIVE_OWNER_TOKEN,
      processStartIdentity: null,
      heartbeatUpdatedAt: new Date().toISOString(),
    });
    try {
      expect(() => acquireObservableTrainingRun(fixture.options))
        .toThrow(/already running/);
      expect(existsSync(active.claimPath)).toBe(true);
    } finally {
      fixture.cleanup();
    }
  });

  it('fails closed without unlinking an unreadable live owner claim', () => {
    const fixture = createRunControlFixture();
    const active = forgeOwnerClaim(fixture, {
      ownerToken: ACTIVE_OWNER_TOKEN,
      processStartIdentity: CURRENT_PROCESS_START_IDENTITY,
      heartbeatUpdatedAt: new Date().toISOString(),
    });
    chmodSync(active.claimPath, 0o000);
    try {
      expect(() => acquireObservableTrainingRun(fixture.options)).toThrow();
      expect(existsSync(active.claimPath)).toBe(true);
      expect(existsSync(active.sentinelPath)).toBe(true);
    } finally {
      chmodSync(active.claimPath, 0o600);
      fixture.cleanup();
    }
  });

  it('fails closed without expiring a live owner whose heartbeat cannot be securely read', () => {
    const fixture = createRunControlFixture();
    const active = forgeOwnerClaim(fixture, {
      ownerToken: ACTIVE_OWNER_TOKEN,
      processStartIdentity: CURRENT_PROCESS_START_IDENTITY,
      heartbeatUpdatedAt: new Date().toISOString(),
    });
    const heartbeatPath = join(active.runDirectory, 'owner-heartbeat.json');
    rmSync(heartbeatPath);
    mkdirSync(heartbeatPath, { mode: 0o700 });
    try {
      expect(() => acquireObservableTrainingRun(fixture.options)).toThrow();
      expect(existsSync(active.claimPath)).toBe(true);
      expect(existsSync(active.sentinelPath)).toBe(true);
    } finally {
      fixture.cleanup();
    }
  });

  it('fails closed without expiring a live owner whose heartbeat is malformed', () => {
    const fixture = createRunControlFixture();
    const active = forgeOwnerClaim(fixture, {
      ownerToken: ACTIVE_OWNER_TOKEN,
      processStartIdentity: CURRENT_PROCESS_START_IDENTITY,
      heartbeatUpdatedAt: new Date().toISOString(),
    });
    writeFileSync(join(active.runDirectory, 'owner-heartbeat.json'), '{"schemaVersion":');
    try {
      expect(() => acquireObservableTrainingRun(fixture.options))
        .toThrow(/heartbeat is malformed/);
      expect(existsSync(active.claimPath)).toBe(true);
      expect(existsSync(active.sentinelPath)).toBe(true);
    } finally {
      fixture.cleanup();
    }
  });

  it('fails closed without expiring a live owner whose heartbeat identity mismatches', () => {
    const fixture = createRunControlFixture();
    const active = forgeOwnerClaim(fixture, {
      ownerToken: ACTIVE_OWNER_TOKEN,
      processStartIdentity: CURRENT_PROCESS_START_IDENTITY,
      heartbeatUpdatedAt: new Date().toISOString(),
    });
    writeFileSync(join(active.runDirectory, 'owner-heartbeat.json'), JSON.stringify({
      schemaVersion: 1,
      ownerToken: STALE_OWNER_TOKEN,
      pid: process.pid,
      processStartIdentity: CURRENT_PROCESS_START_IDENTITY,
      updatedAt: new Date().toISOString(),
    } satisfies TestRunHeartbeatPayload));
    try {
      expect(() => acquireObservableTrainingRun(fixture.options))
        .toThrow(/heartbeat identity does not match/);
      expect(existsSync(active.claimPath)).toBe(true);
      expect(existsSync(active.sentinelPath)).toBe(true);
    } finally {
      fixture.cleanup();
    }
  });

  it('expires a matching coarse process-start identity when its heartbeat lease is stale', () => {
    const fixture = createRunControlFixture();
    const stale = forgeOwnerClaim(fixture, {
      ownerToken: STALE_OWNER_TOKEN,
      processStartIdentity: CURRENT_PROCESS_START_IDENTITY,
      heartbeatUpdatedAt: new Date(0).toISOString(),
    });
    let recovered: ReturnType<typeof acquireObservableTrainingRun> | undefined;
    try {
      recovered = acquireObservableTrainingRun(fixture.options);
      expect(existsSync(stale.claimPath)).toBe(false);
    } finally {
      recovered?.release();
      fixture.cleanup();
    }
  });

  it('refreshes its heartbeat and fails closed after owner-ticket removal', () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-07-16T17:00:00.000Z'));
    const fixture = createRunControlFixture();
    const run = acquireObservableTrainingRun(fixture.options);
    try {
      const initialHeartbeat = readHeartbeat(run.runDirectory);
      expect(initialHeartbeat.updatedAt).toBe('2026-07-16T17:00:00.000Z');
      vi.setSystemTime(new Date('2026-07-16T17:00:20.000Z'));
      run.assertWorkerRuntimeUnchanged();
      const assertionRenewedHeartbeat = readHeartbeat(run.runDirectory);
      expect(assertionRenewedHeartbeat.updatedAt).toBe('2026-07-16T17:00:20.000Z');
      vi.advanceTimersByTime(30_000);
      const refreshedHeartbeat = readHeartbeat(run.runDirectory);
      expect(refreshedHeartbeat.updatedAt).toBe('2026-07-16T17:00:50.000Z');

      const [claim] = readOwnerClaims(fixture);
      if (!claim) throw new Error('Expected an acquired owner claim');
      rmSync(ownerTicketPath(fixture, claim.owner.ownerToken));
      expect(() => run.assertWorkerRuntimeUnchanged())
        .toThrow(/ticket is missing/);
      writeFileSync(ownerTicketPath(fixture, claim.owner.ownerToken), JSON.stringify({
        schemaVersion: 1,
        ownerToken: claim.owner.ownerToken,
        ticket: claim.ticket!,
      } satisfies TestRunOwnerTicketPayload));
    } finally {
      run.release();
      fixture.cleanup();
    }
  });

  it('fails closed when its own heartbeat lease expires before a delayed timer can refresh it', () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-07-16T17:00:00.000Z'));
    const fixture = createRunControlFixture();
    const run = acquireObservableTrainingRun(fixture.options);
    try {
      vi.setSystemTime(new Date('2026-07-16T17:06:00.000Z'));
      expect(() => run.assertWorkerRuntimeUnchanged())
        .toThrow(/heartbeat lease expired/);
    } finally {
      run.release();
      fixture.cleanup();
    }
  });

  it('recovers a stale schema-v1 fixed-file lock without trusting its reused PID forever', () => {
    const fixture = createRunControlFixture();
    writeFileSync(fixture.lockPath, JSON.stringify({
      schemaVersion: 1,
      ownerToken: LEGACY_OWNER_TOKEN,
      pid: process.pid,
      startedAt: new Date(0).toISOString(),
      runDirectory: join(fixture.runRoot, 'legacy-stale-run'),
    }));
    let recovered: ReturnType<typeof acquireObservableTrainingRun> | undefined;
    try {
      recovered = acquireObservableTrainingRun(fixture.options);
      expect(readOwnerClaims(fixture)).toHaveLength(1);
    } finally {
      recovered?.release();
      fixture.cleanup();
    }
  });

  it('keeps a schema-v1 live trainer exclusive well beyond the heartbeat lease', () => {
    vi.useFakeTimers();
    const processStartedAt = Date.parse(MOCK_PROCESS_START);
    vi.setSystemTime(new Date(processStartedAt + 6 * 60 * 60_000));
    const fixture = createRunControlFixture();
    writeFileSync(fixture.lockPath, JSON.stringify({
      schemaVersion: 1,
      ownerToken: LEGACY_OWNER_TOKEN,
      pid: process.pid,
      startedAt: new Date(processStartedAt + 60_000).toISOString(),
      runDirectory: join(fixture.runRoot, 'legacy-live-run'),
    }));
    try {
      expect(() => acquireObservableTrainingRun(fixture.options))
        .toThrow(/already running/);
    } finally {
      fixture.cleanup();
    }
  });

  it('reclaims a recent schema-v1 lock when the live PID started after the lock', () => {
    vi.useFakeTimers();
    const processStartedAt = Date.parse(MOCK_PROCESS_START);
    vi.setSystemTime(new Date(processStartedAt + 60 * 60_000));
    const fixture = createRunControlFixture();
    writeFileSync(fixture.lockPath, JSON.stringify({
      schemaVersion: 1,
      ownerToken: LEGACY_OWNER_TOKEN,
      pid: process.pid,
      startedAt: new Date(processStartedAt - 60 * 60_000).toISOString(),
      runDirectory: join(fixture.runRoot, 'legacy-reused-pid-run'),
    }));
    let recovered: ReturnType<typeof acquireObservableTrainingRun> | undefined;
    try {
      recovered = acquireObservableTrainingRun(fixture.options);
      expect(readOwnerClaims(fixture)).toHaveLength(1);
    } finally {
      recovered?.release();
      fixture.cleanup();
    }
  });

  it('bounds an ambiguous matching schema-v1 process start to 24 hours', () => {
    vi.useFakeTimers();
    const processStartedAt = Date.parse(MOCK_PROCESS_START);
    vi.setSystemTime(new Date(processStartedAt + 25 * 60 * 60_000));
    const fixture = createRunControlFixture();
    writeFileSync(fixture.lockPath, JSON.stringify({
      schemaVersion: 1,
      ownerToken: LEGACY_OWNER_TOKEN,
      pid: process.pid,
      startedAt: new Date(processStartedAt + 60_000).toISOString(),
      runDirectory: join(fixture.runRoot, 'legacy-ambiguous-process-start-run'),
    }));
    let recovered: ReturnType<typeof acquireObservableTrainingRun> | undefined;
    try {
      recovered = acquireObservableTrainingRun(fixture.options);
      expect(readOwnerClaims(fixture)).toHaveLength(1);
    } finally {
      recovered?.release();
      fixture.cleanup();
    }
  });

  it('uses a 24-hour fallback lease when legacy process start time is unobservable', () => {
    vi.useFakeTimers();
    const now = Date.parse('2026-07-20T00:00:00.000Z');
    vi.setSystemTime(now);
    vi.mocked(execFileSync).mockImplementation(() => {
      throw new Error('process start unavailable');
    });
    const fixture = createRunControlFixture();
    const legacyLock = (startedAt: string): string => JSON.stringify({
      schemaVersion: 1,
      ownerToken: LEGACY_OWNER_TOKEN,
      pid: process.pid,
      startedAt,
      runDirectory: join(fixture.runRoot, 'legacy-fallback-run'),
    });
    writeFileSync(fixture.lockPath, legacyLock(
      new Date(now - 23 * 60 * 60_000).toISOString(),
    ));
    let recovered: ReturnType<typeof acquireObservableTrainingRun> | undefined;
    try {
      expect(() => acquireObservableTrainingRun(fixture.options))
        .toThrow(/already running/);
      writeFileSync(fixture.lockPath, legacyLock(
        new Date(now - 25 * 60 * 60_000).toISOString(),
      ));
      recovered = acquireObservableTrainingRun(fixture.options);
      expect(readOwnerClaims(fixture)).toHaveLength(1);
    } finally {
      recovered?.release();
      fixture.cleanup();
    }
  });

  it.each([
    {
      name: 'truncated JSON',
      bytes: '{"schemaVersion":',
    },
    {
      name: 'path-traversing owner token',
      bytes: JSON.stringify({
        schemaVersion: 2,
        ownerToken: '../../../../successor',
        pid: process.pid,
        processStartIdentity: null,
        startedAt: new Date(0).toISOString(),
        runDirectory: '/tmp/untrusted-run',
      }),
    },
  ])('quarantines and ignores a malformed fixed-file lock: $name', ({ bytes }) => {
    const fixture = createRunControlFixture();
    writeFileSync(fixture.lockPath, bytes);
    let recovered: ReturnType<typeof acquireObservableTrainingRun> | undefined;
    try {
      expect(() => acquireObservableTrainingRun(fixture.options))
        .toThrow(/stability grace period/);
      const stableSince = new Date(Date.now() - 2 * 60_000);
      utimesSync(fixture.lockPath, stableSince, stableSince);
      recovered = acquireObservableTrainingRun(fixture.options);
      recovered.release();
      recovered = acquireObservableTrainingRun(fixture.options);
      const quarantineNames = readdirSync(fixture.root).filter((name) =>
        /^trainer\.lock\.quarantine\.[0-9a-f]{64}\.json$/.test(name));
      expect(quarantineNames).toHaveLength(1);
      expect(existsSync(join(fixture.root, quarantineNames[0]!))).toBe(true);
    } finally {
      recovered?.release();
      fixture.cleanup();
    }
  });

  it('fails closed when a legacy fixed lock cannot be securely read', () => {
    const fixture = createRunControlFixture();
    symlinkSync(fixture.sourceWorkerPath, fixture.lockPath);
    try {
      expect(() => acquireObservableTrainingRun(fixture.options))
        .toThrow(/not a regular file/);
      expect(readOwnerClaims(fixture)).toHaveLength(0);
    } finally {
      fixture.cleanup();
    }
  });

  it('gracefully quarantines a legacy lock with an excessive future timestamp', () => {
    vi.useFakeTimers();
    const now = Date.parse('2026-07-20T00:00:00.000Z');
    vi.setSystemTime(now);
    const fixture = createRunControlFixture();
    writeFileSync(fixture.lockPath, JSON.stringify({
      schemaVersion: 1,
      ownerToken: LEGACY_OWNER_TOKEN,
      pid: process.pid,
      startedAt: new Date(now + 2 * 60 * 60_000).toISOString(),
      runDirectory: join(fixture.runRoot, 'legacy-future-run'),
    }));
    utimesSync(fixture.lockPath, new Date(now), new Date(now));
    let recovered: ReturnType<typeof acquireObservableTrainingRun> | undefined;
    try {
      expect(() => acquireObservableTrainingRun(fixture.options))
        .toThrow(/stability grace period/);
      const stableSince = new Date(now - 2 * 60_000);
      utimesSync(fixture.lockPath, stableSince, stableSince);
      recovered = acquireObservableTrainingRun(fixture.options);
      const quarantineNames = readdirSync(fixture.root).filter((name) =>
        /^trainer\.lock\.quarantine\.[0-9a-f]{64}\.json$/.test(name));
      expect(quarantineNames).toHaveLength(1);
    } finally {
      recovered?.release();
      fixture.cleanup();
    }
  });

  it('removes only a stale predecessor claim and preserves a live successor claim', () => {
    const fixture = createRunControlFixture();
    const stale = forgeOwnerClaim(fixture, {
      ownerToken: STALE_OWNER_TOKEN,
      ticket: 1,
      processStartIdentity: 'darwin:unrelated-reused-pid',
      heartbeatUpdatedAt: new Date().toISOString(),
    });
    const successor = forgeOwnerClaim(fixture, {
      ownerToken: ACTIVE_OWNER_TOKEN,
      ticket: 2,
      processStartIdentity: CURRENT_PROCESS_START_IDENTITY,
      heartbeatUpdatedAt: new Date().toISOString(),
    });
    try {
      expect(() => acquireObservableTrainingRun(fixture.options))
        .toThrow(/already running/);
      expect(existsSync(stale.claimPath)).toBe(false);
      expect(existsSync(stale.sentinelPath)).toBe(false);
      expect(existsSync(successor.claimPath)).toBe(true);
      expect(existsSync(successor.sentinelPath)).toBe(true);
    } finally {
      fixture.cleanup();
    }
  });

  it('never deletes a run directory that is not structurally bound to the stale claim token', () => {
    const fixture = createRunControlFixture();
    const protectedRunDirectory = join(fixture.runRoot, ACTIVE_OWNER_TOKEN);
    const protectedSentinel = join(protectedRunDirectory, 'must-survive');
    mkdirSync(protectedRunDirectory, { recursive: true, mode: 0o700 });
    writeFileSync(protectedSentinel, 'protected');
    const staleClaimPath = ownerClaimPath(fixture, STALE_OWNER_TOKEN);
    writeFileSync(staleClaimPath, JSON.stringify({
      schemaVersion: 1,
      choosing: true,
      ticket: null,
      owner: {
        schemaVersion: 2,
        ownerToken: STALE_OWNER_TOKEN,
        pid: process.pid,
        processStartIdentity: 'darwin:unrelated-reused-pid',
        startedAt: new Date(0).toISOString(),
        runDirectory: protectedRunDirectory,
      },
    } satisfies TestRunOwnerClaimPayload));
    writeFileSync(ownerTicketPath(fixture, STALE_OWNER_TOKEN), JSON.stringify({
      schemaVersion: 1,
      ownerToken: STALE_OWNER_TOKEN,
      ticket: 1,
    } satisfies TestRunOwnerTicketPayload));
    let recovered: ReturnType<typeof acquireObservableTrainingRun> | undefined;
    try {
      recovered = acquireObservableTrainingRun(fixture.options);
      expect(existsSync(staleClaimPath)).toBe(false);
      expect(existsSync(protectedSentinel)).toBe(true);
    } finally {
      recovered?.release();
      fixture.cleanup();
    }
  });

  it('cannot resurrect a reclaimed suspended chooser by publishing an orphan ticket', () => {
    const fixture = createRunControlFixture();
    const staleRunDirectory = join(fixture.runRoot, STALE_OWNER_TOKEN);
    mkdirSync(staleRunDirectory, { recursive: true, mode: 0o700 });
    const staleClaimPath = ownerClaimPath(fixture, STALE_OWNER_TOKEN);
    const staleTicketPath = ownerTicketPath(fixture, STALE_OWNER_TOKEN);
    writeFileSync(staleClaimPath, JSON.stringify({
      schemaVersion: 1,
      choosing: true,
      ticket: null,
      owner: {
        schemaVersion: 2,
        ownerToken: STALE_OWNER_TOKEN,
        pid: process.pid,
        processStartIdentity: 'darwin:suspended-reused-owner',
        startedAt: new Date(0).toISOString(),
        runDirectory: staleRunDirectory,
      },
    } satisfies TestRunOwnerClaimPayload));
    let winner: ReturnType<typeof acquireObservableTrainingRun> | undefined;
    try {
      winner = acquireObservableTrainingRun(fixture.options);
      expect(existsSync(staleClaimPath)).toBe(false);
      writeFileSync(staleTicketPath, JSON.stringify({
        schemaVersion: 1,
        ownerToken: STALE_OWNER_TOKEN,
        ticket: 1,
      } satisfies TestRunOwnerTicketPayload));
      expect(existsSync(staleClaimPath)).toBe(false);
      expect(() => acquireObservableTrainingRun(fixture.options))
        .toThrow(/already running/);
      expect(existsSync(staleClaimPath)).toBe(false);
      expect(existsSync(staleTicketPath)).toBe(false);
      expect(readOwnerClaims(fixture)).toHaveLength(1);
    } finally {
      winner?.release();
      fixture.cleanup();
    }
  });

  it('elects exactly one owner when two independent processes start concurrently', async () => {
    const fixture = createRunControlFixture();
    const scriptPath = join(fixture.root, 'concurrent-owner.mjs');
    const goPath = join(fixture.root, 'go');
    const releasePath = join(fixture.root, 'release');
    writeFileSync(scriptPath, concurrentOwnerScript(fixture));
    const children = ['left', 'right'].map((id) =>
      spawn(process.execPath, [scriptPath, id], {
        cwd: fixture.root,
        stdio: ['ignore', 'ignore', 'pipe'],
      }));
    const stderr = new Map(children.map((child) => [child.pid, '']));
    for (const child of children) {
      child.stderr?.on('data', (chunk: Buffer) => {
        stderr.set(child.pid, `${stderr.get(child.pid) ?? ''}${chunk.toString('utf8')}`);
      });
    }
    try {
      await waitUntil(() => ['left', 'right'].every((id) =>
        existsSync(join(fixture.root, `${id}.ready`))));
      writeFileSync(goPath, 'go');
      await waitUntil(() => ['left', 'right'].every((id) =>
        fileHasContent(join(fixture.root, `${id}.result`))));
      const results = ['left', 'right'].map((id) =>
        readFileSync(join(fixture.root, `${id}.result`), 'utf8'));
      expect(results.sort()).toEqual(['acquired', 'blocked']);
      expect(readOwnerClaims(fixture)).toHaveLength(1);
      writeFileSync(releasePath, 'release');
      const exitCodes = await Promise.all(children.map(waitForChildExit));
      expect(exitCodes, JSON.stringify(Object.fromEntries(stderr))).toEqual([0, 0]);
    } finally {
      writeFileSync(goPath, 'go');
      writeFileSync(releasePath, 'release');
      for (const child of children) {
        if (child.exitCode === null && child.signalCode === null) child.kill('SIGTERM');
      }
      await Promise.all(children.map((child) =>
        child.exitCode === null && child.signalCode === null
          ? waitForChildExit(child)
          : Promise.resolve(child.exitCode)));
      fixture.cleanup();
    }
  }, 15_000);
});

function createRunControlFixture(): {
  root: string;
  lockPath: string;
  ownerDirectory: string;
  runRoot: string;
  sourceWorkerPath: string;
  options: Parameters<typeof acquireObservableTrainingRun>[0];
  cleanup(): void;
} {
  const root = mkdtempSync(join(tmpdir(), 'tinysa-run-control-pid-reuse-'));
  const sourceDirectory = join(root, 'source');
  const sourceWorkerPath = join(sourceDirectory, 'worker.js');
  const lockPath = join(root, 'trainer.lock');
  const ownerDirectory = `${lockPath}.owners`;
  const runRoot = join(root, 'runs');
  mkdirSync(sourceDirectory, { recursive: true, mode: 0o700 });
  mkdirSync(ownerDirectory, { recursive: true, mode: 0o700 });
  mkdirSync(runRoot, { recursive: true, mode: 0o700 });
  writeFileSync(sourceWorkerPath, 'export const fixture = true;\n');
  const options = {
    lockPath,
    runRoot,
    sourceWorkerModuleUrl: pathToFileURL(sourceWorkerPath),
  };
  return {
    root,
    lockPath,
    ownerDirectory,
    runRoot,
    sourceWorkerPath,
    options,
    cleanup(): void {
      rmSync(root, { recursive: true, force: true });
      assert.equal(existsSync(root), false);
    },
  };
}

function forgeOwnerClaim(
  fixture: ReturnType<typeof createRunControlFixture>,
  options: {
    ownerToken: string;
    ticket?: number;
    processStartIdentity: string | null;
    heartbeatUpdatedAt: string;
  },
): { claimPath: string; runDirectory: string; sentinelPath: string } {
  const runDirectory = join(fixture.runRoot, options.ownerToken);
  const sentinelPath = join(runDirectory, 'runtime-sentinel');
  mkdirSync(runDirectory, { recursive: true, mode: 0o700 });
  writeFileSync(sentinelPath, 'sentinel');
  const owner: TestRunLockPayload = {
    schemaVersion: 2,
    ownerToken: options.ownerToken,
    pid: process.pid,
    processStartIdentity: options.processStartIdentity,
    startedAt: new Date(0).toISOString(),
    runDirectory,
  };
  const claim: TestRunOwnerClaimPayload = {
    schemaVersion: 1,
    choosing: true,
    ticket: null,
    owner,
  };
  const claimPath = ownerClaimPath(fixture, options.ownerToken);
  writeFileSync(claimPath, JSON.stringify(claim));
  writeFileSync(ownerTicketPath(fixture, options.ownerToken), JSON.stringify({
    schemaVersion: 1,
    ownerToken: options.ownerToken,
    ticket: options.ticket ?? 1,
  } satisfies TestRunOwnerTicketPayload));
  writeFileSync(join(runDirectory, 'owner-heartbeat.json'), JSON.stringify({
    schemaVersion: 1,
    ownerToken: options.ownerToken,
    pid: process.pid,
    processStartIdentity: options.processStartIdentity,
    updatedAt: options.heartbeatUpdatedAt,
  } satisfies TestRunHeartbeatPayload));
  return { claimPath, runDirectory, sentinelPath };
}

function ownerClaimPath(
  fixture: Pick<ReturnType<typeof createRunControlFixture>, 'ownerDirectory'>,
  ownerToken: string,
): string {
  return join(fixture.ownerDirectory, `${ownerToken}.json`);
}

function ownerTicketPath(
  fixture: Pick<ReturnType<typeof createRunControlFixture>, 'ownerDirectory'>,
  ownerToken: string,
): string {
  return join(fixture.ownerDirectory, `${ownerToken}.ticket.json`);
}

function readOwnerClaims(
  fixture: Pick<ReturnType<typeof createRunControlFixture>, 'ownerDirectory'>,
): TestRunOwnerClaimPayload[] {
  return readdirSync(fixture.ownerDirectory)
    .filter((name) => /^[0-9a-f-]{36}\.json$/.test(name))
    .map((name) => {
      const claim = JSON.parse(
        readFileSync(join(fixture.ownerDirectory, name), 'utf8'),
      ) as TestRunOwnerClaimPayload;
      const ticketPath = ownerTicketPath(fixture, claim.owner.ownerToken);
      if (!existsSync(ticketPath)) return claim;
      const ticket = JSON.parse(
        readFileSync(ticketPath, 'utf8'),
      ) as TestRunOwnerTicketPayload;
      return { ...claim, choosing: false, ticket: ticket.ticket };
    });
}

function readHeartbeat(runDirectory: string): TestRunHeartbeatPayload {
  return JSON.parse(
    readFileSync(join(runDirectory, 'owner-heartbeat.json'), 'utf8'),
  ) as TestRunHeartbeatPayload;
}

function concurrentOwnerScript(
  fixture: Pick<
  ReturnType<typeof createRunControlFixture>,
  'root' | 'lockPath' | 'runRoot' | 'sourceWorkerPath'
  >,
): string {
  const moduleUrl = pathToFileURL(
    join(process.cwd(), 'tools', 'observable-training-run-control.ts'),
  ).href;
  return `
import { existsSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { acquireObservableTrainingRun } from ${JSON.stringify(moduleUrl)};

const id = process.argv[2];
const root = ${JSON.stringify(fixture.root)};
writeFileSync(join(root, id + '.ready'), 'ready');
while (!existsSync(join(root, 'go'))) {
  await new Promise((resolve) => setTimeout(resolve, 5));
}
try {
  const run = acquireObservableTrainingRun({
    lockPath: ${JSON.stringify(fixture.lockPath)},
    runRoot: ${JSON.stringify(fixture.runRoot)},
    sourceWorkerModuleUrl: pathToFileURL(${JSON.stringify(fixture.sourceWorkerPath)}),
  });
  writeFileSync(join(root, id + '.result'), 'acquired');
  while (!existsSync(join(root, 'release'))) {
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  run.release();
} catch (error) {
  const message = error instanceof Error ? error.message : String(error);
  writeFileSync(
    join(root, id + '.result'),
    /already running|choosing its owner ticket/.test(message) ? 'blocked' : 'error:' + message,
  );
}
`;
}

async function waitUntil(predicate: () => boolean, timeoutMs = 5_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!predicate()) {
    if (Date.now() >= deadline) throw new Error('Timed out waiting for concurrent owner fixture');
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}

function fileHasContent(path: string): boolean {
  try {
    return readFileSync(path).length > 0;
  } catch {
    return false;
  }
}

function waitForChildExit(
  child: ReturnType<typeof spawn>,
): Promise<number | null> {
  if (child.exitCode !== null || child.signalCode !== null) {
    return Promise.resolve(child.exitCode);
  }
  return new Promise((resolve) => {
    child.once('exit', (code) => resolve(code));
  });
}
