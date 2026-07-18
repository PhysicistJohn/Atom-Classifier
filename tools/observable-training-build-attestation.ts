import { createHash } from 'node:crypto';
import {
  closeSync,
  constants,
  fstatSync,
  lstatSync,
  openSync,
  readSync,
} from 'node:fs';
import { basename, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export const OBSERVABLE_TRAINING_BUILD_ID_ENV =
  'TINYSA_OBSERVABLE_TRAINING_BUILD_ID' as const;
export const OBSERVABLE_TRAINING_TRAINER_SHA256_ENV =
  'TINYSA_OBSERVABLE_TRAINING_TRAINER_SHA256' as const;
export const OBSERVABLE_TRAINING_WORKER_SHA256_ENV =
  'TINYSA_OBSERVABLE_TRAINING_WORKER_SHA256' as const;
export const OBSERVABLE_TRAINING_NODE_VERSION_ENV =
  'TINYSA_OBSERVABLE_TRAINING_NODE_VERSION' as const;
export const OBSERVABLE_TRAINING_RUNTIME_IDENTITY_POLICY =
  'exact-repository-node-version-v1' as const;

const MAX_BUILD_ARTIFACT_BYTES = 16 * 1024 * 1024;

export interface ObservableTrainingBuildAttestation {
  readonly buildId: string;
  readonly trainerSha256: string;
  readonly workerSha256: string;
  readonly workerRuntimeSha256: string;
  readonly nodeVersion: string;
  readonly v8Version: string;
}

export function assertObservableTrainingBuildAttestation(options: {
  trainerModuleUrl: URL;
  workerModuleUrl: URL;
  environment?: NodeJS.ProcessEnv;
}): ObservableTrainingBuildAttestation {
  assertImmutableBuildAdmissionSupported();
  const environment = options.environment ?? process.env;
  const buildId = requiredEnvironmentValue(
    environment,
    OBSERVABLE_TRAINING_BUILD_ID_ENV,
  );
  const expectedTrainerSha256 = requiredSha256(
    environment,
    OBSERVABLE_TRAINING_TRAINER_SHA256_ENV,
  );
  const expectedWorkerSha256 = requiredSha256(
    environment,
    OBSERVABLE_TRAINING_WORKER_SHA256_ENV,
  );
  const expectedNodeVersion = requiredEnvironmentValue(
    environment,
    OBSERVABLE_TRAINING_NODE_VERSION_ENV,
  );
  if (!/^\d+\.\d+\.\d+$/.test(expectedNodeVersion)) {
    throw new Error('Observable training launcher Node.js version attestation is malformed');
  }
  if (process.version !== `v${expectedNodeVersion}`) {
    throw new Error(
      `Observable training runtime Node.js ${process.version} does not match `
        + `launcher pin v${expectedNodeVersion}`,
    );
  }
  if (!/^[0-9a-f-]{36}$/.test(buildId)) {
    throw new Error('Observable training build ID is malformed');
  }
  const trainerPath = resolve(fileURLToPath(options.trainerModuleUrl));
  const workerPath = resolve(fileURLToPath(options.workerModuleUrl));
  if (dirname(trainerPath) !== dirname(workerPath)) {
    throw new Error('Observable trainer and worker are not from one private build directory');
  }
  assertImmutableBuildDirectory(dirname(trainerPath));
  const trainerSha256 = sha256(readBoundedImmutableBuildArtifact(trainerPath));
  const workerSha256 = sha256(readBoundedImmutableBuildArtifact(workerPath));
  if (trainerSha256 !== expectedTrainerSha256
    || workerSha256 !== expectedWorkerSha256) {
    throw new Error('Observable trainer/worker build attestation does not match executed bytes');
  }
  const workerRuntimeSha256 = sha256(JSON.stringify([{
    path: basename(workerPath),
    sha256: workerSha256,
  }]));
  return {
    buildId,
    trainerSha256,
    workerSha256,
    workerRuntimeSha256,
    nodeVersion: expectedNodeVersion,
    v8Version: process.versions.v8,
  };
}

function assertImmutableBuildDirectory(path: string): void {
  const status = lstatSync(path);
  if (!status.isDirectory() || status.isSymbolicLink()) {
    throw new Error('Observable training build directory must be a non-symlink directory');
  }
  if ((status.mode & 0o222) !== 0) {
    throw new Error('Observable training build directory must be read-only before execution');
  }
}

function readBoundedImmutableBuildArtifact(path: string): Buffer {
  const beforeOpen = lstatSync(path);
  if (!beforeOpen.isFile() || beforeOpen.isSymbolicLink()) {
    throw new Error(`Observable training build artifact is not a regular file: ${path}`);
  }
  if ((beforeOpen.mode & 0o222) !== 0) {
    throw new Error(`Observable training build artifact is writable: ${path}`);
  }
  const descriptor = openSync(
    path,
    constants.O_RDONLY | (constants.O_NOFOLLOW ?? 0),
  );
  try {
    const status = fstatSync(descriptor);
    if (!status.isFile()
      || status.dev !== beforeOpen.dev
      || status.ino !== beforeOpen.ino
      || status.size > MAX_BUILD_ARTIFACT_BYTES) {
      throw new Error(`Observable training build artifact changed or is oversized: ${path}`);
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
        throw new Error(`Observable training build artifact truncated: ${path}`);
      }
      offset += bytesRead;
    }
    const afterOpen = lstatSync(path);
    if (!afterOpen.isFile()
      || afterOpen.isSymbolicLink()
      || afterOpen.dev !== status.dev
      || afterOpen.ino !== status.ino
      || afterOpen.size !== status.size) {
      throw new Error(`Observable training build artifact changed during admission: ${path}`);
    }
    return bytes;
  } finally {
    closeSync(descriptor);
  }
}

function assertImmutableBuildAdmissionSupported(): void {
  if (process.platform === 'win32') {
    throw new Error(
      'Observable training build attestation requires enforceable POSIX read-only modes; '
        + 'Windows admission is denied because immutability cannot be verified',
    );
  }
}

function requiredEnvironmentValue(
  environment: NodeJS.ProcessEnv,
  name: string,
): string {
  const value = environment[name];
  if (!value) {
    throw new Error(`Observable training launcher attestation is missing ${name}`);
  }
  return value;
}

function requiredSha256(
  environment: NodeJS.ProcessEnv,
  name: string,
): string {
  const value = requiredEnvironmentValue(environment, name);
  if (!/^[a-f0-9]{64}$/.test(value)) {
    throw new Error(`Observable training launcher attestation ${name} is malformed`);
  }
  return value;
}

function sha256(value: string | Buffer): string {
  return createHash('sha256').update(value).digest('hex');
}
