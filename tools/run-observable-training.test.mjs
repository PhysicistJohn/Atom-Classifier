import assert from 'node:assert/strict';
import { execFile, spawn } from 'node:child_process';
import { once } from 'node:events';
import {
  chmodSync,
  existsSync,
  lstatSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  realpathSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from 'node:fs';
import { constants as osConstants, tmpdir } from 'node:os';
import { join, resolve, sep } from 'node:path';
import test from 'node:test';
import { setTimeout as delay } from 'node:timers/promises';
import { pathToFileURL } from 'node:url';
import { promisify } from 'node:util';
import {
  buildObservableTrainingInvocation,
  runObservableTraining,
} from './run-observable-training.mjs';

const REPOSITORY_ROOT = resolve(import.meta.dirname, '..');
const execFileAsync = promisify(execFile);
const posixTest = process.platform === 'win32' ? test.skip : test;
const windowsTest = process.platform === 'win32' ? test : test.skip;

test('classifier training scripts use only the private-build launcher', () => {
  const packageJson = JSON.parse(readFileSync(
    join(REPOSITORY_ROOT, 'package.json'),
    'utf8',
  ));
  assert.equal(
    packageJson.scripts['train:signal-classifier'],
    'node tools/run-observable-training.mjs train',
  );
  assert.equal(
    packageJson.scripts['check:signal-classifier-model'],
    'node tools/run-observable-training.mjs check',
  );
  assert.doesNotMatch(packageJson.scripts['train:signal-classifier'], /dist\/tools|tsup/);
  assert.doesNotMatch(packageJson.scripts['check:signal-classifier-model'], /dist\/tools|tsup/);
});

posixTest('parallel classifier builds are isolated, immutable, and byte-coherent', {
  timeout: 60_000,
}, async () => {
  const temporaryParent = mkdtempSync(join(tmpdir(), 'tinysa launcher concurrency '));
  let first;
  let second;
  try {
    [first, second] = await Promise.all([
      buildObservableTrainingInvocation({
        repositoryRoot: REPOSITORY_ROOT,
        temporaryParent,
      }),
      buildObservableTrainingInvocation({
        repositoryRoot: REPOSITORY_ROOT,
        temporaryParent,
      }),
    ]);
    assert.notEqual(first.buildId, second.buildId);
    assert.notEqual(first.buildDirectory, second.buildDirectory);
    assert.equal(first.trainerSha256, second.trainerSha256);
    assert.equal(first.workerSha256, second.workerSha256);
    assert.equal(first.workerRuntimeSha256, second.workerRuntimeSha256);
    const generatedModelSource = readFileSync(join(
      REPOSITORY_ROOT,
      'src/models/bayesian-observable.generated.ts',
    ), 'utf8');
    const generatedModelMatch = generatedModelSource.match(
      /BAYESIAN_OBSERVABLE_MODEL: ObservableClassifierModelAsset = (\{[\s\S]*\});\s*$/,
    );
    assert.ok(generatedModelMatch);
    assert.equal(
      first.workerRuntimeSha256,
      JSON.parse(generatedModelMatch[1]).trainingMatrix
        .attemptSamplingWorkerRuntimeSha256,
    );
    for (const build of [first, second]) {
      assert.equal(
        resolve(build.buildDirectory).startsWith(realpathSync(temporaryParent)),
        true,
      );
      assert.equal(existsSync(build.trainerPath), true);
      assert.equal(existsSync(build.workerPath), true);
      assert.equal(
        readFileSync(build.workerPath, 'utf8').includes('BAYESIAN_OBSERVABLE_MODEL'),
        false,
      );
      if (process.platform !== 'win32') {
        assert.equal(lstatSync(build.buildDirectory).mode & 0o222, 0);
        assert.equal(lstatSync(build.trainerPath).mode & 0o222, 0);
        assert.equal(lstatSync(build.workerPath).mode & 0o222, 0);
      }
      assert.deepEqual(build.attestationEnvironment, {
        TINYSA_OBSERVABLE_TRAINING_BUILD_ID: build.buildId,
        TINYSA_OBSERVABLE_TRAINING_TRAINER_SHA256: build.trainerSha256,
        TINYSA_OBSERVABLE_TRAINING_WORKER_SHA256: build.workerSha256,
        TINYSA_OBSERVABLE_TRAINING_NODE_VERSION: process.versions.node,
      });
    }
    const unattestedEnvironment = { ...process.env };
    delete unattestedEnvironment.TINYSA_OBSERVABLE_TRAINING_BUILD_ID;
    delete unattestedEnvironment.TINYSA_OBSERVABLE_TRAINING_TRAINER_SHA256;
    delete unattestedEnvironment.TINYSA_OBSERVABLE_TRAINING_WORKER_SHA256;
    delete unattestedEnvironment.TINYSA_OBSERVABLE_TRAINING_NODE_VERSION;
    await assert.rejects(
      execFileAsync(process.execPath, [first.trainerPath], {
        cwd: temporaryParent,
        env: unattestedEnvironment,
        encoding: 'utf8',
      }),
      (error) => String(error.stderr).includes(
        'Observable training launcher attestation is missing '
          + 'TINYSA_OBSERVABLE_TRAINING_BUILD_ID',
      ),
    );
  } finally {
    first?.cleanup();
    second?.cleanup();
    assert.equal(first ? existsSync(first.buildDirectory) : false, false);
    assert.equal(second ? existsSync(second.buildDirectory) : false, false);
    rmSync(temporaryParent, { recursive: true, force: true });
  }
});

posixTest('launcher derives and enforces the exact Node.js pin before any private build', async () => {
  const temporaryParent = mkdtempSync(join(tmpdir(), 'tinysa launcher node pin '));
  const repositoryRoot = createFakeCompilerRepository(
    temporaryParent,
    emittingCompilerSource({
      trainerSource: 'void 0;\n',
      workerSource: 'void 0;\n',
    }),
  );
  let build;
  try {
    assert.equal(
      readFileSync(join(repositoryRoot, '.node-version'), 'utf8').trim(),
      process.versions.node,
    );
    build = await buildObservableTrainingInvocation({
      repositoryRoot,
      temporaryParent,
    });
    build.cleanup();
    build = undefined;
    assert.deepEqual(observableTrainingInvocationNames(temporaryParent), []);

    writeFileSync(join(repositoryRoot, '.node-version'), '0.0.0\n');
    await assert.rejects(
      buildObservableTrainingInvocation({
        repositoryRoot,
        temporaryParent,
      }),
      /requires exact Node\.js 0\.0\.0 .*launcher is v/,
    );
    assert.deepEqual(observableTrainingInvocationNames(temporaryParent), []);

    const launcherSource = readFileSync(
      join(REPOSITORY_ROOT, 'tools/run-observable-training.mjs'),
      'utf8',
    );
    const repositoryPin = readFileSync(join(REPOSITORY_ROOT, '.node-version'), 'utf8').trim();
    assert.match(launcherSource, /\.node-version/);
    assert.equal(launcherSource.includes(repositoryPin), false);
  } finally {
    build?.cleanup();
    rmSync(temporaryParent, { recursive: true, force: true });
  }
});

posixTest('launcher rejects inherited Node and esbuild injection controls before building', async () => {
  const temporaryParent = mkdtempSync(join(tmpdir(), 'tinysa launcher injection '));
  try {
    for (const key of ['NODE_OPTIONS', 'ESBUILD_BINARY_PATH']) {
      const previousValue = process.env[key];
      process.env[key] = key === 'NODE_OPTIONS'
        ? '--require=/definitely-not-an-observable-training-module'
        : '/definitely-not-an-esbuild-binary';
      try {
        await assert.rejects(
          buildObservableTrainingInvocation({
            repositoryRoot: REPOSITORY_ROOT,
            temporaryParent,
          }),
          new RegExp(`refuses code/toolchain injection environment variable ${key}`),
        );
      } finally {
        restoreEnvironmentValue(key, previousValue);
      }
      assert.deepEqual(observableTrainingInvocationNames(temporaryParent), []);
    }
  } finally {
    rmSync(temporaryParent, { recursive: true, force: true });
  }
});

posixTest('trainer receives attestation but not injected loader or toolchain environment', async () => {
  const temporaryParent = mkdtempSync(join(tmpdir(), 'tinysa launcher trainer env '));
  const trainerSource = `
const prohibited = new Set([
  'NODE_OPTIONS',
  'NODE_PATH',
  'ESBUILD_BINARY_PATH',
  'LD_PRELOAD',
  'DYLD_INSERT_LIBRARIES',
]);
if (Object.keys(process.env).some((key) => prohibited.has(key.toUpperCase()))) {
  process.exitCode = 71;
} else if (!process.env.TINYSA_OBSERVABLE_TRAINING_BUILD_ID
  || !process.env.TINYSA_OBSERVABLE_TRAINING_TRAINER_SHA256
  || !process.env.TINYSA_OBSERVABLE_TRAINING_WORKER_SHA256
  || process.env.TINYSA_OBSERVABLE_TRAINING_NODE_VERSION !== process.versions.node) {
  process.exitCode = 72;
}
`;
  const repositoryRoot = createFakeCompilerRepository(
    temporaryParent,
    emittingCompilerSource({
      trainerSource,
      workerSource: 'void 0;\n',
    }),
  );
  try {
    const exitCode = await runObservableTraining('train', {
      repositoryRoot,
      temporaryParent,
      environment: {
        ...process.env,
        NODE_OPTIONS: '--require=/definitely-not-an-observable-training-module',
        NODE_PATH: '/definitely-not-a-module-path',
        ESBUILD_BINARY_PATH: '/definitely-not-an-esbuild-binary',
        LD_PRELOAD: '/definitely-not-a-preload-library',
        DYLD_INSERT_LIBRARIES: '/definitely-not-a-dylib',
      },
    });
    assert.equal(exitCode, 0);
    assert.deepEqual(observableTrainingInvocationNames(temporaryParent), []);
  } finally {
    rmSync(temporaryParent, { recursive: true, force: true });
  }
});

posixTest('cleanup remains retryable after a transient parent-permission failure', async () => {
  const temporaryParent = mkdtempSync(join(tmpdir(), 'tinysa launcher cleanup retry '));
  const repositoryRoot = createFakeCompilerRepository(
    temporaryParent,
    emittingCompilerSource({
      trainerSource: 'void 0;\n',
      workerSource: 'void 0;\n',
    }),
  );
  let build;
  try {
    build = await buildObservableTrainingInvocation({
      repositoryRoot,
      temporaryParent,
    });
    chmodSync(temporaryParent, 0o500);
    assert.throws(
      () => build.cleanup(),
      /EACCES|EPERM|permission denied|operation not permitted/i,
    );
    assert.equal(existsSync(build.invocationDirectory), true);
    chmodSync(temporaryParent, 0o700);
    build.cleanup();
    assert.equal(existsSync(build.invocationDirectory), false);
  } finally {
    chmodSync(temporaryParent, 0o700);
    try {
      build?.cleanup();
    } catch {
      // The enclosing temporary parent removal is the final test cleanup fallback.
    }
    rmSync(temporaryParent, { recursive: true, force: true });
  }
});

posixTest('cleanup failure reports and retains the original compiler failure', async () => {
  const temporaryParent = mkdtempSync(join(tmpdir(), 'tinysa launcher cleanup error '));
  const repositoryRoot = createFakeCompilerRepository(
    temporaryParent,
    `
const fs = require('node:fs');
fs.chmodSync(process.env.TINYSA_TEST_TEMP_PARENT, 0o500);
process.stderr.write('primary compiler failure\\n');
process.exit(37);
`,
  );
  const previousParent = process.env.TINYSA_TEST_TEMP_PARENT;
  process.env.TINYSA_TEST_TEMP_PARENT = temporaryParent;
  try {
    await assert.rejects(
      buildObservableTrainingInvocation({
        repositoryRoot,
        temporaryParent,
      }),
      (error) => {
        assert.match(error.message, /primary compiler failure/);
        assert.match(error.message, /cleanup also failed/);
        assert.match(error.cause?.message ?? '', /compiler exited with code 37/);
        assert.ok(error.cleanupError);
        return true;
      },
    );
  } finally {
    restoreEnvironmentValue('TINYSA_TEST_TEMP_PARENT', previousParent);
    chmodSync(temporaryParent, 0o700);
    rmSync(temporaryParent, { recursive: true, force: true });
  }
});

posixTest('temporary-parent symlinks are canonicalized before private builds', async () => {
  const container = mkdtempSync(join(tmpdir(), 'tinysa launcher canonical parent '));
  const temporaryParent = join(container, 'real parent');
  const temporaryParentAlias = join(container, 'parent alias');
  mkdirSync(temporaryParent, { mode: 0o700 });
  symlinkSync(temporaryParent, temporaryParentAlias, 'dir');
  const repositoryRoot = createFakeCompilerRepository(
    container,
    emittingCompilerSource({
      trainerSource: 'void 0;\n',
      workerSource: 'void 0;\n',
    }),
  );
  let build;
  try {
    build = await buildObservableTrainingInvocation({
      repositoryRoot,
      temporaryParent: temporaryParentAlias,
    });
    assert.equal(
      build.invocationDirectory.startsWith(`${realpathSync(temporaryParent)}${sep}`),
      true,
    );
  } finally {
    build?.cleanup();
    rmSync(container, { recursive: true, force: true });
  }
});

posixTest('candidate directory replacement cannot redirect build admission or cleanup', async () => {
  const temporaryParent = mkdtempSync(join(tmpdir(), 'tinysa launcher candidate identity '));
  const redirectedDirectory = join(temporaryParent, 'redirected output');
  mkdirSync(redirectedDirectory, { mode: 0o700 });
  const repositoryRoot = createFakeCompilerRepository(
    temporaryParent,
    `
const fs = require('node:fs');
const path = require('node:path');
const outputDirectory = process.argv[process.argv.indexOf('--out-dir') + 1];
fs.rmSync(outputDirectory, { recursive: true, force: true });
fs.symlinkSync(process.env.TINYSA_TEST_REDIRECT_TARGET, outputDirectory, 'dir');
fs.writeFileSync(path.join(
  process.env.TINYSA_TEST_REDIRECT_TARGET,
  'train-observable-classifier.js',
), 'void 0;\\n');
fs.writeFileSync(path.join(
  process.env.TINYSA_TEST_REDIRECT_TARGET,
  'observable-training-sampling-worker.js',
), 'void 0;\\n');
`,
  );
  const previousTarget = process.env.TINYSA_TEST_REDIRECT_TARGET;
  process.env.TINYSA_TEST_REDIRECT_TARGET = redirectedDirectory;
  try {
    await assert.rejects(
      buildObservableTrainingInvocation({
        repositoryRoot,
        temporaryParent,
      }),
      /private directory identity changed/,
    );
    assert.equal(
      existsSync(join(redirectedDirectory, 'train-observable-classifier.js')),
      true,
    );
    assert.deepEqual(observableTrainingInvocationNames(temporaryParent), []);
  } finally {
    restoreEnvironmentValue('TINYSA_TEST_REDIRECT_TARGET', previousTarget);
    rmSync(temporaryParent, { recursive: true, force: true });
  }
});

posixTest('candidate artifact symlinks are rejected without following them', async () => {
  const temporaryParent = mkdtempSync(join(tmpdir(), 'tinysa launcher nofollow '));
  const externalWorker = join(temporaryParent, 'external-worker.js');
  writeFileSync(externalWorker, 'void 0;\n');
  const repositoryRoot = createFakeCompilerRepository(
    temporaryParent,
    `
const fs = require('node:fs');
const path = require('node:path');
const outputDirectory = process.argv[process.argv.indexOf('--out-dir') + 1];
fs.writeFileSync(
  path.join(outputDirectory, 'train-observable-classifier.js'),
  'void 0;\\n',
);
fs.symlinkSync(
  process.env.TINYSA_TEST_EXTERNAL_WORKER,
  path.join(outputDirectory, 'observable-training-sampling-worker.js'),
);
`,
  );
  const previousWorker = process.env.TINYSA_TEST_EXTERNAL_WORKER;
  process.env.TINYSA_TEST_EXTERNAL_WORKER = externalWorker;
  try {
    await assert.rejects(
      buildObservableTrainingInvocation({
        repositoryRoot,
        temporaryParent,
      }),
      /build output is not a regular file/,
    );
    assert.equal(readFileSync(externalWorker, 'utf8'), 'void 0;\n');
    assert.deepEqual(observableTrainingInvocationNames(temporaryParent), []);
  } finally {
    restoreEnvironmentValue('TINYSA_TEST_EXTERNAL_WORKER', previousWorker);
    rmSync(temporaryParent, { recursive: true, force: true });
  }
});

posixTest('SIGTERM during a slow candidate build terminates its process tree and cleans temp', {
  timeout: 20_000,
}, async () => {
  const temporaryParent = mkdtempSync(join(tmpdir(), 'tinysa launcher signal cleanup '));
  const readyPath = join(temporaryParent, 'compiler-ready.json');
  const signalPath = join(temporaryParent, 'compiler-signal.txt');
  const repositoryRoot = createFakeCompilerRepository(
    temporaryParent,
    `
const fs = require('node:fs');
const { spawn } = require('node:child_process');
const blocked = new Set([
  'NODE_OPTIONS',
  'NODE_PATH',
  'ESBUILD_BINARY_PATH',
  'LD_PRELOAD',
  'DYLD_INSERT_LIBRARIES',
]);
if (Object.keys(process.env).some((key) => blocked.has(key.toUpperCase()))) {
  process.exit(91);
}
const grandchild = spawn(
  process.execPath,
  ['-e', 'process.on("SIGTERM", () => {}); setInterval(() => {}, 1000)'],
  { stdio: 'ignore' },
);
process.on('SIGTERM', () => {
  fs.appendFileSync(process.env.TINYSA_TEST_SIGNAL_PATH, 'SIGTERM\\n');
});
fs.writeFileSync(process.env.TINYSA_TEST_READY_PATH, JSON.stringify({
  compilerPid: process.pid,
  grandchildPid: grandchild.pid,
}));
setInterval(() => {}, 1000);
`,
  );
  const launcherModuleUrl = pathToFileURL(
    join(REPOSITORY_ROOT, 'tools/run-observable-training.mjs'),
  ).href;
  const wrapperSource = `
import { buildObservableTrainingInvocation } from ${JSON.stringify(launcherModuleUrl)};
try {
  await buildObservableTrainingInvocation({
    repositoryRoot: ${JSON.stringify(repositoryRoot)},
    temporaryParent: ${JSON.stringify(temporaryParent)},
    terminationGraceMs: 250,
  });
  process.exitCode = 90;
} catch (error) {
  process.stderr.write(String(error?.stack ?? error) + '\\n');
  process.exitCode = Number.isInteger(error?.exitCode) ? error.exitCode : 1;
}
`;
  const stderr = [];
  const launcher = spawn(
    process.execPath,
    ['--input-type=module', '--eval', wrapperSource],
    {
      env: {
        ...process.env,
        TINYSA_TEST_READY_PATH: readyPath,
        TINYSA_TEST_SIGNAL_PATH: signalPath,
      },
      stdio: ['ignore', 'ignore', 'pipe'],
    },
  );
  launcher.stderr.on('data', (chunk) => stderr.push(Buffer.from(chunk)));
  let processIds;
  try {
    await waitFor(() => existsSync(readyPath), 8_000);
    processIds = JSON.parse(readFileSync(readyPath, 'utf8'));
    assert.equal(launcher.kill('SIGTERM'), true);
    const [exitCode, signal] = await once(launcher, 'exit');
    assert.equal(signal, null);
    assert.equal(exitCode, 128 + osConstants.signals.SIGTERM);
    assert.match(
      Buffer.concat(stderr).toString('utf8'),
      /Observable training interrupted by SIGTERM/,
    );
    await waitFor(() => existsSync(signalPath), 5_000);
    assert.equal(readFileSync(signalPath, 'utf8'), 'SIGTERM\n');
    await waitFor(
      () => !processExists(processIds.compilerPid)
        && !processExists(processIds.grandchildPid),
      5_000,
    );
    assert.deepEqual(observableTrainingInvocationNames(temporaryParent), []);
  } finally {
    if (launcher.exitCode === null && launcher.signalCode === null) {
      launcher.kill('SIGKILL');
      await once(launcher, 'exit').catch(() => {});
    }
    if (processIds?.compilerPid && processExists(processIds.compilerPid)) {
      try {
        process.kill(-processIds.compilerPid, 'SIGKILL');
      } catch {
        // Best-effort cleanup for a failed process-tree assertion.
      }
    }
    rmSync(temporaryParent, { recursive: true, force: true });
  }
});

windowsTest('launcher fails closed when immutable build admission cannot be enforced', async () => {
  await assert.rejects(
    buildObservableTrainingInvocation({
      repositoryRoot: REPOSITORY_ROOT,
    }),
    /Windows execution is denied because immutable build admission cannot be verified/,
  );
});

function createFakeCompilerRepository(temporaryParent, compilerSource) {
  const repositoryRoot = join(
    temporaryParent,
    `fake-repository-${Math.random().toString(16).slice(2)}`,
  );
  const tsupDirectory = join(repositoryRoot, 'node_modules', 'tsup', 'dist');
  mkdirSync(tsupDirectory, { recursive: true, mode: 0o700 });
  writeFileSync(join(repositoryRoot, '.node-version'), `${process.versions.node}\n`);
  writeFileSync(join(tsupDirectory, 'cli-default.js'), compilerSource);
  return repositoryRoot;
}

function emittingCompilerSource({ trainerSource, workerSource }) {
  return `
const fs = require('node:fs');
const path = require('node:path');
const outputDirectory = process.argv[process.argv.indexOf('--out-dir') + 1];
fs.writeFileSync(
  path.join(outputDirectory, 'train-observable-classifier.js'),
  ${JSON.stringify(trainerSource)},
);
fs.writeFileSync(
  path.join(outputDirectory, 'observable-training-sampling-worker.js'),
  ${JSON.stringify(workerSource)},
);
`;
}

function observableTrainingInvocationNames(temporaryParent) {
  return readdirSync(temporaryParent)
    .filter((name) => name.startsWith('tinysa-observable-training-'))
    .sort();
}

function restoreEnvironmentValue(key, value) {
  if (value === undefined) {
    delete process.env[key];
  } else {
    process.env[key] = value;
  }
}

async function waitFor(predicate, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (!predicate()) {
    if (Date.now() >= deadline) {
      throw new Error(`Timed out after ${timeoutMs}ms`);
    }
    await delay(20);
  }
}

function processExists(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    if (error?.code === 'ESRCH') return false;
    if (error?.code === 'EPERM') return true;
    throw error;
  }
}
