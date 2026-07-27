import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import {
  chmodSync,
  existsSync,
  mkdtempSync,
  readFileSync,
  symlinkSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import test from 'node:test';

const REPO = resolve(import.meta.dirname, '..');
const SCRIPT = join(REPO, 'tools/generate-signallab-iq-release-suite.mjs');
const HASH = createHash('sha256').update(readFileSync(SCRIPT)).digest('hex');
const TEST_SIGNALLAB_ROOT =
  process.env.ATOMOS_RELEASE_TEST_SIGNALLAB_ROOT?.trim();

function run(extraEnv = {}) {
  return spawnSync(process.execPath, [SCRIPT], {
    cwd: REPO,
    env: {
      ...process.env,
      RELEASE_SEED: '20260728',
      CANDIDATE_PATH: SCRIPT,
      CANDIDATE_SHA256: HASH,
      ...(TEST_SIGNALLAB_ROOT
        ? { SIGNALLAB_ROOT: TEST_SIGNALLAB_ROOT }
        : {}),
      ...extraEnv,
    },
    encoding: 'utf8',
  });
}

test('requires an explicit release root', () => {
  const result = run({ RELEASE_ROOT: '' });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /RELEASE_ROOT is required/);
});

test('refuses the live training corpus before generation', () => {
  const result = run({
    RELEASE_ROOT: 'training/artifacts/signallab-corpus',
  });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /may not be the live training corpus/);
});

test('refuses descendants and ancestors of the live training corpus', () => {
  for (const releaseRoot of [
    'training/artifacts/signallab-corpus/release-child',
    'training/artifacts',
  ]) {
    const result = run({ RELEASE_ROOT: releaseRoot });
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /descendants.*ancestors/);
  }
});

test('resolves symlink aliases before checking live-corpus containment', () => {
  const aliasRoot = mkdtempSync(join(tmpdir(), 'atomos-release-alias-'));
  const alias = join(aliasRoot, 'artifacts-alias');
  symlinkSync(join(REPO, 'training/artifacts'), alias, 'dir');
  const result = run({
    RELEASE_ROOT: join(alias, 'signallab-corpus', 'release-child'),
  });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /live training corpus/);
});

test('refuses a non-empty release root before generation', () => {
  const root = mkdtempSync(join(tmpdir(), 'atomos-release-nonempty-'));
  writeFileSync(join(root, 'witness'), 'do not overwrite\n');
  const result = run({ RELEASE_ROOT: root });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /RELEASE_ROOT must be empty/);
});

test('validates the frozen-candidate digest before creating output', () => {
  const root = join(
    tmpdir(),
    `atomos-release-invalid-hash-${process.pid}-${Date.now()}`,
  );
  const result = run({
    RELEASE_ROOT: root,
    CANDIDATE_SHA256: 'not-a-digest',
  });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /exactly 64 hexadecimal characters/);
});

test('rejects a well-formed but wrong candidate digest without creating output', () => {
  const root = join(
    tmpdir(),
    `atomos-release-wrong-hash-${process.pid}-${Date.now()}`,
  );
  const result = run({
    RELEASE_ROOT: root,
    CANDIDATE_SHA256: '0'.repeat(64),
  });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /CANDIDATE_SHA256 mismatch/);
  assert.equal(existsSync(root), false);
});

test('predeclares evaluation choices in intent before corpus generation', {
  skip: TEST_SIGNALLAB_ROOT
    ? false
    : 'set ATOMOS_RELEASE_TEST_SIGNALLAB_ROOT to a verified isolated tree',
}, () => {
  const root = join(
    tmpdir(),
    `atomos-release-intent-${process.pid}-${Date.now()}`,
  );
  const fakeBin = mkdtempSync(join(tmpdir(), 'atomos-release-fake-bin-'));
  const fakeNpx = join(fakeBin, 'npx');
  writeFileSync(
    fakeNpx,
    '#!/bin/sh\n'
      + 'if [ "$1" = "--version" ]; then echo "10.9.8"; exit 0; fi\n'
      + 'exit 17\n',
  );
  chmodSync(fakeNpx, 0o755);
  const result = run({
    RELEASE_ROOT: root,
    RELEASE_LENGTHS: '4096,8192,16384,32768',
    RELEASE_TARGET_PER_CLASS: '80',
    PATH: `${fakeBin}:${process.env.PATH ?? ''}`,
  });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /occupied-start probe generation.*failed/);
  const intent = JSON.parse(
    readFileSync(join(root, 'RELEASE_INTENT.json'), 'utf8'),
  );
  assert.equal(intent.status, 'in_progress');
  assert.equal(intent.exact_target_per_class, true);
  assert.equal(intent.evaluation_protocol.version, 'invariant-release-evaluation-v1');
  assert.deepEqual(
    intent.evaluation_protocol.required_capture_lengths,
    [4096, 8192, 16384, 32768],
  );
  assert.equal(intent.evaluation_protocol.minimum_target_per_class, 80);
  assert.equal(
    intent.evaluation_protocol.length_observation_rule,
    'generate maximum capture length once; every shorter observed and clean row '
      + 'is a bit-exact cf32le prefix of the corresponding maximum-length row',
  );
  assert.equal(intent.evaluation_protocol.five_shot.k, 5);
  assert.equal(
    intent.evaluation_protocol.five_shot.split_version,
    'sha256-lowest-per-class-v1',
  );
  assert.equal(intent.evaluation_protocol.novelty.seed, 20260729);
  assert.equal(intent.evaluation_protocol.novelty.n_each_per_length, 300);
  assert.deepEqual(intent.evaluation_protocol.novelty.families, ['noise', 'chirp']);
  assert.equal(
    intent.evaluation_protocol.novelty.seed_derivation,
    'uint64 little-endian first 8 bytes of SHA-256(utf8(base seed decimal '
      + '+ NUL + family))',
  );
  assert.deepEqual(
    intent.evaluation_protocol.physical_scale_factors,
    [0.5, 0.75, 1, 1.5, 2],
  );
  assert.equal(intent.source_sha256.tsx_package, 'tsx@4.20.3');
  assert.match(intent.source_sha256.launcher.sha256, /^[0-9a-f]{64}$/);
  assert.match(intent.source_sha256.corpus_generator.sha256, /^[0-9a-f]{64}$/);
  assert.match(intent.source_sha256.prefix_deriver.sha256, /^[0-9a-f]{64}$/);
  assert.equal(intent.runtime_provenance.node, 'v22.23.1');
  assert.equal(intent.runtime_provenance.npm, '10.9.8');
  assert.equal(
    intent.dependency_provenance.signallab.git_commit,
    '7c8303a0338b0f7c088737f8df2d935fd04bb033',
  );
  assert.equal(
    intent.dependency_provenance.atom_dsp.dist_index_js.sha256,
    'caef3a8b3cbf4c300f8ceab5619a1db6b4fcebb942d428577f50790a1fb360e7',
  );
});
