#!/usr/bin/env node

import { spawn } from 'node:child_process';
import { createHash, randomUUID } from 'node:crypto';
import {
  chmodSync,
  closeSync,
  constants as fsConstants,
  fstatSync,
  lstatSync,
  mkdirSync,
  mkdtempSync,
  openSync,
  readSync,
  readdirSync,
  realpathSync,
  renameSync,
  rmSync,
} from 'node:fs';
import { constants as osConstants, tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPOSITORY_ROOT = resolve(HERE, '..');
const TRAINER_ENTRY = 'tools/train-observable-classifier.ts';
const WORKER_ENTRY = 'tools/observable-training-sampling-worker.ts';
const TRAINER_OUTPUT = 'train-observable-classifier.js';
const WORKER_OUTPUT = 'observable-training-sampling-worker.js';
const EXPECTED_OUTPUTS = [TRAINER_OUTPUT, WORKER_OUTPUT].sort();
const MAX_BUILD_ARTIFACT_BYTES = 16 * 1024 * 1024;
const MAX_NODE_VERSION_PIN_BYTES = 64;
const BUILD_ID_ENV = 'TINYSA_OBSERVABLE_TRAINING_BUILD_ID';
const TRAINER_SHA256_ENV = 'TINYSA_OBSERVABLE_TRAINING_TRAINER_SHA256';
const WORKER_SHA256_ENV = 'TINYSA_OBSERVABLE_TRAINING_WORKER_SHA256';
const NODE_VERSION_ENV = 'TINYSA_OBSERVABLE_TRAINING_NODE_VERSION';
const CHILD_CODE_INJECTION_ENVIRONMENT_KEYS = new Set([
  'DYLD_FALLBACK_LIBRARY_PATH',
  'DYLD_FRAMEWORK_PATH',
  'DYLD_INSERT_LIBRARIES',
  'DYLD_LIBRARY_PATH',
  'ESBUILD_BINARY_PATH',
  'LD_LIBRARY_PATH',
  'LD_PRELOAD',
  'NODE_OPTIONS',
  'NODE_PATH',
]);
const FORWARDED_SIGNALS = ['SIGHUP', 'SIGINT', 'SIGTERM'];
const DEFAULT_TERMINATION_GRACE_MS = 5_000;

export async function buildObservableTrainingInvocation(options = {}) {
  assertPrivateBuildImmutabilitySupported();
  assertCodeInjectionEnvironmentAbsent(process.env);
  const repositoryRoot = resolve(options.repositoryRoot ?? REPOSITORY_ROOT);
  const pinnedNodeVersion = assertPinnedNodeVersion(repositoryRoot);
  const temporaryParent = realpathSync(resolve(options.temporaryParent ?? tmpdir()));
  const terminationGraceMs = normalizeTerminationGraceMs(options.terminationGraceMs);
  assertDirectory(temporaryParent, 'Observable training temporary parent');
  const temporaryParentIdentity = fileIdentity(lstatSync(temporaryParent));
  const assertTemporaryParentUnchanged = () => assertMatchingDirectoryIdentity(
    temporaryParent,
    lstatSync(temporaryParent),
    temporaryParentIdentity,
  );
  const buildId = randomUUID();
  const invocationDirectory = mkdtempSync(join(
    temporaryParent,
    `tinysa-observable-training-${buildId}-`,
  ));
  const candidateA = join(invocationDirectory, 'candidate-a');
  const candidateB = join(invocationDirectory, 'candidate-b');
  const buildDirectory = join(invocationDirectory, 'runtime');
  chmodSync(invocationDirectory, 0o700);
  const invocationIdentity = fileIdentity(lstatSync(invocationDirectory));
  const assertInvocationUnchanged = () => assertMatchingDirectoryIdentity(
    invocationDirectory,
    lstatSync(invocationDirectory),
    invocationIdentity,
  );
  let cleaned = false;
  let cleanupAtExit;
  const cleanup = () => {
    if (cleaned) return;
    let invocationStatus;
    try {
      invocationStatus = lstatSync(invocationDirectory);
    } catch (error) {
      if (error?.code === 'ENOENT') {
        cleaned = true;
        process.off('exit', cleanupAtExit);
        return;
      }
      throw error;
    }
    assertMatchingDirectoryIdentity(
      invocationDirectory,
      invocationStatus,
      invocationIdentity,
    );
    for (const directory of [
      candidateA,
      candidateB,
      buildDirectory,
      invocationDirectory,
    ]) {
      makeOwnedDirectoryRemovable(directory);
    }
    rmSync(invocationDirectory, { recursive: true, force: true });
    cleaned = true;
    process.off('exit', cleanupAtExit);
  };
  cleanupAtExit = () => {
    try {
      cleanup();
    } catch {
      // Process exit is already the final cleanup opportunity.
    }
  };
  process.once('exit', cleanupAtExit);
  const buildLifecycle = createBuildLifecycle({
    terminationGraceMs,
  });
  try {
    buildLifecycle.start();
    assertTemporaryParentUnchanged();
    const tsupCli = join(repositoryRoot, 'node_modules', 'tsup', 'dist', 'cli-default.js');
    assertRegularFile(tsupCli, 'tsup CLI');
    assertInvocationUnchanged();
    const firstCandidate = await compileCandidate({
      repositoryRoot,
      outputDirectory: candidateA,
      tsupCli,
      execPath: options.execPath ?? process.execPath,
      buildLifecycle,
      assertInvocationUnchanged,
    });
    const secondCandidate = await compileCandidate({
      repositoryRoot,
      outputDirectory: candidateB,
      tsupCli,
      execPath: options.execPath ?? process.execPath,
      buildLifecycle,
      assertInvocationUnchanged,
    });
    if (firstCandidate.trainerSha256 !== secondCandidate.trainerSha256
      || firstCandidate.workerSha256 !== secondCandidate.workerSha256) {
      throw new Error(
        'Observable training source changed or compiled incoherently across its two private builds',
      );
    }
    assertInvocationUnchanged();
    assertMatchingDirectoryIdentity(
      candidateA,
      lstatSync(candidateA),
      firstCandidate.directoryIdentity,
    );
    assertMatchingDirectoryIdentity(
      candidateB,
      lstatSync(candidateB),
      secondCandidate.directoryIdentity,
    );
    rmSync(candidateB, { recursive: true, force: true });
    renameSync(candidateA, buildDirectory);
    assertMatchingDirectoryIdentity(
      buildDirectory,
      lstatSync(buildDirectory),
      firstCandidate.directoryIdentity,
    );
    const runtimeCandidate = inspectCandidate(
      buildDirectory,
      firstCandidate.directoryIdentity,
    );
    const {
      trainerPath,
      workerPath,
      trainerSha256,
      workerSha256,
    } = runtimeCandidate;
    if (trainerSha256 !== firstCandidate.trainerSha256
      || workerSha256 !== firstCandidate.workerSha256) {
      throw new Error('Observable training candidate changed during private runtime handoff');
    }
    chmodSync(trainerPath, 0o400);
    chmodSync(workerPath, 0o400);
    chmodSync(buildDirectory, 0o500);
    chmodSync(invocationDirectory, 0o500);
    if (sha256(readBoundedRegularFile(trainerPath)) !== trainerSha256
      || sha256(readBoundedRegularFile(workerPath)) !== workerSha256) {
      throw new Error('Observable training private build changed while being sealed');
    }
    buildLifecycle.throwIfInterrupted();
    return {
      buildId,
      invocationDirectory,
      buildDirectory,
      trainerPath,
      workerPath,
      trainerSha256,
      workerSha256,
      workerRuntimeSha256: sha256(JSON.stringify([{
        path: WORKER_OUTPUT,
        sha256: workerSha256,
      }])),
      attestationEnvironment: {
        [BUILD_ID_ENV]: buildId,
        [TRAINER_SHA256_ENV]: trainerSha256,
        [WORKER_SHA256_ENV]: workerSha256,
        [NODE_VERSION_ENV]: pinnedNodeVersion,
      },
      cleanup,
    };
  } catch (error) {
    try {
      cleanup();
    } catch (cleanupError) {
      const combinedError = new Error(
        `${describeBuildFailure(error)}; cleanup also failed: ${errorMessage(cleanupError)}`,
        { cause: error },
      );
      combinedError.cleanupError = cleanupError;
      if (error instanceof ObservableTrainingInterruptedError) {
        combinedError.signal = error.signal;
        combinedError.exitCode = error.exitCode;
      }
      throw combinedError;
    }
    if (error instanceof ObservableTrainingInterruptedError) {
      throw error;
    }
    throw new Error(
      describeBuildFailure(error),
      { cause: error },
    );
  } finally {
    buildLifecycle.close();
  }
}

async function compileCandidate({
  repositoryRoot,
  outputDirectory,
  tsupCli,
  execPath,
  buildLifecycle,
  assertInvocationUnchanged,
}) {
  buildLifecycle.throwIfInterrupted();
  assertInvocationUnchanged();
  mkdirSync(outputDirectory, { mode: 0o700 });
  assertDirectory(outputDirectory, 'Observable training candidate directory');
  const directoryIdentity = fileIdentity(lstatSync(outputDirectory));
  const compilerResult = await runCompiler(
    execPath,
    [
      tsupCli,
      TRAINER_ENTRY,
      WORKER_ENTRY,
      '--format',
      'esm',
      '--out-dir',
      outputDirectory,
      '--platform',
      'node',
      '--no-splitting',
      '--treeshake',
      '--silent',
      '--clean',
    ],
    {
      cwd: repositoryRoot,
      env: sanitizedChildEnvironment(),
    },
    buildLifecycle,
  );
  buildLifecycle.throwIfInterrupted();
  assertInvocationUnchanged();
  assertMatchingDirectoryIdentity(
    outputDirectory,
    lstatSync(outputDirectory),
    directoryIdentity,
  );
  if (compilerResult.code !== 0) {
    const error = new Error(
      `Observable training compiler exited with code ${compilerResult.code}`,
    );
    error.stdout = compilerResult.stdout;
    error.stderr = compilerResult.stderr;
    throw error;
  }
  return inspectCandidate(outputDirectory, directoryIdentity);
}

function runCompiler(command, arguments_, options, buildLifecycle) {
  return new Promise((resolvePromise, rejectPromise) => {
    let settled = false;
    let terminalError = null;
    let stdoutBytes = 0;
    let stderrBytes = 0;
    const stdout = [];
    const stderr = [];
    const child = spawn(command, arguments_, {
      ...options,
      detached: process.platform !== 'win32',
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    buildLifecycle.setChild(child);
    const rejectOnce = (error) => {
      if (settled) return;
      settled = true;
      buildLifecycle.clearChild(child);
      rejectPromise(error);
    };
    const capture = (chunks, streamName) => (chunk) => {
      const bytes = Buffer.byteLength(chunk);
      if (streamName === 'stdout') {
        stdoutBytes += bytes;
      } else {
        stderrBytes += bytes;
      }
      if (stdoutBytes > MAX_BUILD_ARTIFACT_BYTES
        || stderrBytes > MAX_BUILD_ARTIFACT_BYTES) {
        terminalError ??= new Error(
          `Observable training compiler ${streamName} exceeded its size bound`,
        );
        terminateProcessTree(child, 'SIGKILL');
        return;
      }
      chunks.push(Buffer.from(chunk));
    };
    child.stdout.on('data', capture(stdout, 'stdout'));
    child.stderr.on('data', capture(stderr, 'stderr'));
    child.once('error', rejectOnce);
    child.once('close', (code, signal) => {
      if (settled) return;
      settled = true;
      buildLifecycle.clearChild(child);
      if (buildLifecycle.interruptedSignal !== null) {
        rejectPromise(new ObservableTrainingInterruptedError(
          buildLifecycle.interruptedSignal,
        ));
        return;
      }
      if (terminalError !== null) {
        rejectPromise(terminalError);
        return;
      }
      if (code === null) {
        rejectPromise(new Error(
          `Observable training compiler terminated by ${signal ?? 'an unknown signal'}`,
        ));
        return;
      }
      resolvePromise({
        code,
        stdout: Buffer.concat(stdout).toString('utf8'),
        stderr: Buffer.concat(stderr).toString('utf8'),
      });
    });
  });
}

function inspectCandidate(directory, expectedIdentity) {
  assertDirectory(directory, 'Observable training candidate directory');
  if (expectedIdentity !== undefined) {
    assertMatchingDirectoryIdentity(
      directory,
      lstatSync(directory),
      expectedIdentity,
    );
  }
  const outputs = readdirSync(directory).sort();
  if (JSON.stringify(outputs) !== JSON.stringify(EXPECTED_OUTPUTS)) {
    throw new Error(
      `Observable training build emitted unexpected files: ${outputs.join(', ') || '<none>'}`,
    );
  }
  const trainerPath = join(directory, TRAINER_OUTPUT);
  const workerPath = join(directory, WORKER_OUTPUT);
  const trainerBytes = readBoundedRegularFile(trainerPath);
  const workerBytes = readBoundedRegularFile(workerPath);
  if (/BAYESIAN_OBSERVABLE_MODEL|BAYESIAN_OBSERVABLE_MODEL_CONTENT_SHA256/.test(
    workerBytes.toString('utf8'),
  )) {
    throw new Error(
      'Observable sampling worker bundle contains the generated model whose provenance records its digest',
    );
  }
  return {
    directoryIdentity: expectedIdentity ?? fileIdentity(lstatSync(directory)),
    trainerPath,
    workerPath,
    trainerSha256: sha256(trainerBytes),
    workerSha256: sha256(workerBytes),
  };
}

export async function runObservableTraining(mode, options = {}) {
  if (mode !== 'train' && mode !== 'check') {
    throw new Error('Usage: run-observable-training.mjs <train|check>');
  }
  const repositoryRoot = resolve(options.repositoryRoot ?? REPOSITORY_ROOT);
  const build = await buildObservableTrainingInvocation({
    repositoryRoot,
    temporaryParent: options.temporaryParent,
    execPath: options.execPath,
    terminationGraceMs: options.terminationGraceMs,
  });
  try {
    const trainerArguments = mode === 'check'
      ? ['--check', '--fresh-sampling']
      : [];
    return await runChild(
      options.execPath ?? process.execPath,
      [build.trainerPath, ...trainerArguments],
      {
        cwd: repositoryRoot,
        env: sanitizedChildEnvironment(options.environment ?? process.env, {
          ...build.attestationEnvironment,
        }),
      },
    );
  } finally {
    build.cleanup();
  }
}

function runChild(command, arguments_, options) {
  return new Promise((resolvePromise, rejectPromise) => {
    const child = spawn(command, arguments_, {
      ...options,
      detached: process.platform !== 'win32',
      stdio: 'inherit',
    });
    const signalHandlers = new Map();
    const cleanupHandlers = () => {
      for (const [signal, handler] of signalHandlers) {
        process.off(signal, handler);
      }
    };
    for (const signal of FORWARDED_SIGNALS) {
      const handler = () => {
        if (child.exitCode === null && child.signalCode === null) {
          terminateProcessTree(child, signal);
        }
      };
      signalHandlers.set(signal, handler);
      process.on(signal, handler);
    }
    child.once('error', (error) => {
      cleanupHandlers();
      rejectPromise(error);
    });
    child.once('exit', (code, signal) => {
      cleanupHandlers();
      if (code !== null) {
        resolvePromise(code);
        return;
      }
      resolvePromise(128 + (osConstants.signals[signal] ?? 1));
    });
  });
}

function assertRegularFile(path, label) {
  const status = lstatSync(path);
  if (!status.isFile() || status.isSymbolicLink()) {
    throw new Error(`${label} must be a regular non-symlink file: ${path}`);
  }
}

function assertPrivateBuildImmutabilitySupported() {
  if (process.platform === 'win32') {
    throw new Error(
      'Observable training private builds require enforceable POSIX read-only modes; '
        + 'Windows execution is denied because immutable build admission cannot be verified',
    );
  }
}

function assertCodeInjectionEnvironmentAbsent(environment) {
  const prohibitedKey = Object.keys(environment).find(
    (key) => CHILD_CODE_INJECTION_ENVIRONMENT_KEYS.has(key.toUpperCase()),
  );
  if (prohibitedKey !== undefined) {
    throw new Error(
      `Observable training refuses code/toolchain injection environment variable ${prohibitedKey}`,
    );
  }
}

function assertDirectory(path, label) {
  const status = lstatSync(path);
  if (!status.isDirectory() || status.isSymbolicLink()) {
    throw new Error(`${label} must be a regular non-symlink directory: ${path}`);
  }
}

function assertMatchingDirectoryIdentity(path, status, expectedIdentity) {
  if (!status.isDirectory()
    || status.isSymbolicLink()
    || status.dev !== expectedIdentity.dev
    || status.ino !== expectedIdentity.ino) {
    throw new Error(`Observable training private directory identity changed: ${path}`);
  }
}

function fileIdentity(status) {
  return {
    dev: status.dev,
    ino: status.ino,
  };
}

function makeOwnedDirectoryRemovable(path) {
  let status;
  try {
    status = lstatSync(path);
  } catch (error) {
    if (error?.code === 'ENOENT') return;
    throw error;
  }
  if (status.isDirectory() && !status.isSymbolicLink()) {
    chmodSync(path, 0o700);
  }
}

function assertPinnedNodeVersion(repositoryRoot) {
  const pinPath = join(repositoryRoot, '.node-version');
  const pin = readBoundedRegularFile(pinPath, {
    maximumBytes: MAX_NODE_VERSION_PIN_BYTES,
    label: 'Observable training Node.js version pin',
  }).toString('utf8').trim();
  if (!/^\d+\.\d+\.\d+$/.test(pin)) {
    throw new Error(
      `Observable training Node.js version pin is not an exact semantic version: ${pinPath}`,
    );
  }
  if (process.version !== `v${pin}`) {
    throw new Error(
      `Observable training requires exact Node.js ${pin} from ${pinPath}; `
        + `launcher is ${process.version}`,
    );
  }
  return pin;
}

function readBoundedRegularFile(path, options = {}) {
  const maximumBytes = options.maximumBytes ?? MAX_BUILD_ARTIFACT_BYTES;
  const label = options.label ?? 'Observable training build output';
  const beforeOpen = lstatSync(path);
  if (!beforeOpen.isFile() || beforeOpen.isSymbolicLink()) {
    throw new Error(`${label} is not a regular file: ${path}`);
  }
  if (beforeOpen.size > maximumBytes) {
    throw new Error(`${label} exceeds its size bound: ${path}`);
  }
  const descriptor = openSync(
    path,
    fsConstants.O_RDONLY | (fsConstants.O_NOFOLLOW ?? 0),
  );
  try {
    const status = fstatSync(descriptor);
    if (!status.isFile()
      || status.dev !== beforeOpen.dev
      || status.ino !== beforeOpen.ino
      || status.size > maximumBytes) {
      throw new Error(`${label} changed or is oversized: ${path}`);
    }
    const bytes = Buffer.allocUnsafe(status.size);
    let offset = 0;
    while (offset < bytes.length) {
      const bytesRead = readSync(
        descriptor,
        bytes,
        offset,
        bytes.length - offset,
        offset,
      );
      if (bytesRead === 0) {
        throw new Error(`${label} truncated: ${path}`);
      }
      offset += bytesRead;
    }
    const afterOpen = lstatSync(path);
    if (!afterOpen.isFile()
      || afterOpen.isSymbolicLink()
      || afterOpen.dev !== status.dev
      || afterOpen.ino !== status.ino
      || afterOpen.size !== status.size) {
      throw new Error(`${label} changed during admission: ${path}`);
    }
    return bytes;
  } finally {
    closeSync(descriptor);
  }
}

function sanitizedChildEnvironment(
  environment = process.env,
  additions = {},
) {
  const sanitized = {
    ...environment,
    ...additions,
  };
  for (const key of Object.keys(sanitized)) {
    if (CHILD_CODE_INJECTION_ENVIRONMENT_KEYS.has(key.toUpperCase())) {
      delete sanitized[key];
    }
  }
  return sanitized;
}

function normalizeTerminationGraceMs(value) {
  const terminationGraceMs = value ?? DEFAULT_TERMINATION_GRACE_MS;
  if (!Number.isFinite(terminationGraceMs) || terminationGraceMs < 0) {
    throw new Error('Observable training termination grace must be non-negative');
  }
  return terminationGraceMs;
}

function describeBuildFailure(error) {
  if (error instanceof ObservableTrainingInterruptedError) {
    return error.message;
  }
  const stderr = typeof error?.stderr === 'string' ? error.stderr.trim() : '';
  const detail = stderr || errorMessage(error);
  return `Observable training private build failed${detail ? `: ${detail}` : ''}`;
}

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}

function createBuildLifecycle({ terminationGraceMs }) {
  let currentChild = null;
  let interruptedSignal = null;
  let forceKillTimer = null;
  let started = false;
  const signalHandlers = new Map();
  const clearForceKillTimer = () => {
    if (forceKillTimer !== null) {
      clearTimeout(forceKillTimer);
      forceKillTimer = null;
    }
  };
  const terminateCurrentChild = () => {
    if (currentChild === null || interruptedSignal === null) return;
    terminateProcessTree(currentChild, interruptedSignal);
    clearForceKillTimer();
    forceKillTimer = setTimeout(() => {
      if (currentChild !== null) {
        terminateProcessTree(currentChild, 'SIGKILL');
      }
    }, terminationGraceMs);
    forceKillTimer.unref?.();
  };
  return {
    get interruptedSignal() {
      return interruptedSignal;
    },
    start() {
      if (started) return;
      started = true;
      for (const signal of FORWARDED_SIGNALS) {
        const handler = () => {
          if (interruptedSignal !== null) return;
          interruptedSignal = signal;
          terminateCurrentChild();
        };
        signalHandlers.set(signal, handler);
        process.on(signal, handler);
      }
    },
    setChild(child) {
      if (currentChild !== null) {
        throw new Error('Observable training build attempted concurrent compiler children');
      }
      currentChild = child;
      terminateCurrentChild();
    },
    clearChild(child) {
      if (currentChild === child) {
        currentChild = null;
        clearForceKillTimer();
      }
    },
    throwIfInterrupted() {
      if (interruptedSignal !== null) {
        throw new ObservableTrainingInterruptedError(interruptedSignal);
      }
    },
    close() {
      clearForceKillTimer();
      for (const [signal, handler] of signalHandlers) {
        process.off(signal, handler);
      }
      signalHandlers.clear();
      started = false;
    },
  };
}

function terminateProcessTree(child, signal) {
  if (child.exitCode !== null || child.signalCode !== null) return;
  if (process.platform !== 'win32' && child.pid) {
    try {
      process.kill(-child.pid, signal);
      return;
    } catch (error) {
      if (error?.code === 'ESRCH') return;
    }
  }
  try {
    child.kill(signal);
  } catch (error) {
    if (error?.code !== 'ESRCH') {
      return;
    }
  }
}

class ObservableTrainingInterruptedError extends Error {
  constructor(signal) {
    super(`Observable training interrupted by ${signal}`);
    this.name = 'ObservableTrainingInterruptedError';
    this.signal = signal;
    this.exitCode = 128 + (osConstants.signals[signal] ?? 1);
  }
}

function sha256(value) {
  return createHash('sha256').update(value).digest('hex');
}

const invokedPath = process.argv[1] ? resolve(process.argv[1]) : '';
if (invokedPath === fileURLToPath(import.meta.url)) {
  try {
    const exitCode = await runObservableTraining(process.argv[2]);
    process.exitCode = exitCode;
  } catch (error) {
    process.exitCode = Number.isInteger(error?.exitCode) ? error.exitCode : 1;
    console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  }
}
