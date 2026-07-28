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

test('refuses an unknown RELEASE_EVALUATION_PROTOCOL before any output', () => {
  const root = join(
    tmpdir(),
    `atomos-release-bad-protocol-${process.pid}-${Date.now()}`,
  );
  const result = run({
    RELEASE_ROOT: root,
    RELEASE_EVALUATION_PROTOCOL: 'v9',
  });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /RELEASE_EVALUATION_PROTOCOL must be 'v2'/);
  assert.equal(existsSync(root), false);
});

test('v3 protocol refuses a release seed the fixture did not predeclare', () => {
  const root = join(
    tmpdir(),
    `atomos-release-v3-seed-mismatch-${process.pid}-${Date.now()}`,
  );
  const result = run({
    RELEASE_ROOT: root,
    RELEASE_EVALUATION_PROTOCOL: 'v3',
    RELEASE_SEED: '20260728',
  });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /predeclares seed 20260735/);
  assert.equal(existsSync(root), false);
});

test('v3 protocol refuses consumed and development-band release seeds', () => {
  for (const [seed, pattern] of [
    ['20260729', /consumed sealed v2 release suite/],
    ['20260731', /consumed sealed v3\.0 release suite/],
    ['20260733', /consumed sealed v3\.2 release suite/],
    ['20260734', /historical gate redeclaration/],
    ['20260942', /development novelty seed namespace/],
    ['20261001', /noise-prefilter fit-only seed band/],
  ]) {
    const root = join(
      tmpdir(),
      `atomos-release-v3-refused-seed-${seed}-${process.pid}-${Date.now()}`,
    );
    const result = run({
      RELEASE_ROOT: root,
      RELEASE_EVALUATION_PROTOCOL: 'v3',
      RELEASE_SEED: seed,
    });
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, pattern);
    assert.equal(existsSync(root), false);
  }
});

test('v3 protocol fixture is the evaluator-printed object, self-consistent', () => {
  // The fixture is captured from
  //   evaluate_v3_release_suite.py --print-expected-protocol 20260735
  // (command recorded inside the fixture itself). Node tests never shell out
  // to Python; the sealed evaluator's own intent-equality check is the
  // final cross-language backstop.
  const wrapper = JSON.parse(
    readFileSync(
      join(REPO, 'tools/time-domain-v3-expected-evaluation-protocol-seed20260735.json'),
      'utf8',
    ),
  );
  assert.equal(wrapper.release_seed, 20260735);
  const protocol = wrapper.evaluation_protocol;
  assert.equal(
    protocol.version,
    'time-domain-v3-release-evaluation-v3-dual-fusion',
  );
  assert.equal(protocol.novelty.seed, 20260735);
  assert.deepEqual(protocol.required_capture_lengths, [4096, 8192, 16384, 32768]);
  assert.equal(protocol.open_set.additive_only, false);
  assert.equal(protocol.open_set.changes_closed_label, true);
  assert.equal(protocol.open_set.gates_before_classification, true);
  assert.equal(
    protocol.open_set.architecture,
    'v3_staged_noise_prefilter_then_composite_survivor_lof_geometry',
  );
  // Staged policy version 2: the composite survivor score. The protocol must
  // name the exact policy version it validates; the sealed evaluator refuses
  // a staged artifact recording any other version.
  assert.equal(
    protocol.open_set.staged_policy_version,
    'v3-staged-openset-policy-v2-composite-survivor',
  );
  assert.equal(protocol.open_set.staged_policy_schema, 2);
  assert.equal(
    protocol.open_set.survivor_score,
    'max(stage2_enrollment_rank, stage1_score_enrollment_rank)',
  );
  assert.match(
    protocol.open_set.unknown_threshold_rule,
    /q0\.95 of the composite over enrollment stage-1 survivors/,
  );
  // Every stage-1 length must be one of the sealed capture lengths. Sealed
  // lengths ABOVE the longest fitted length are gated through the
  // causal-prefix rule; any other unfitted length is uncovered.
  const covered = protocol.open_set.stage_one_capture_lengths;
  for (const length of covered) {
    assert.ok(protocol.required_capture_lengths.includes(length));
  }
  const maxFitted = Math.max(...covered);
  assert.equal(protocol.open_set.stage_one_max_fitted_length, maxFitted);
  assert.equal(
    protocol.open_set.stage_one_prefix_rule.max_fitted_length,
    maxFitted,
  );
  assert.match(protocol.open_set.stage_one_prefix_rule.rule, /causal-prefix/);
  assert.deepEqual(
    protocol.open_set.stage_one_prefix_gated_lengths,
    protocol.required_capture_lengths.filter((length) => length > maxFitted),
  );
  assert.deepEqual(
    protocol.open_set.stage_one_uncovered_lengths,
    protocol.required_capture_lengths.filter(
      (length) => !covered.includes(length) && length < maxFitted,
    ),
  );
  // The current gate floors are the strict imported v2 floors.
  assert.equal(protocol.gates.open_unknown_recall_noise, 0.10);
  assert.equal(protocol.gates.open_auroc_noise, 0.80);
  assert.equal(protocol.gates.open_known_false_unknown_max, 0.10);
  assert.equal(protocol.gates.five_shot, 0.85);
  assert.equal(Object.keys(protocol.gates).length, 17);
  // The seed-20260734 relaxation remains transparent, but is explicitly
  // historical and inactive for this protocol.
  const historical = protocol.historical_gate_redeclaration;
  assert.equal(historical.release_seed, 20260734);
  assert.equal(historical.active_for_current_protocol, false);
  assert.equal(
    historical.current_protocol_gate_source,
    'imported_v2_gate_floors_unchanged',
  );
  const redeclared = historical.gates_redeclared;
  assert.deepEqual(Object.keys(redeclared).sort(), [
    'five_shot_worst_length_balanced',
    'open_known_false_unknown_worst_length',
  ]);
  assert.equal(redeclared.open_known_false_unknown_worst_length.v2_level, 0.10);
  assert.equal(redeclared.open_known_false_unknown_worst_length.v3_level, 0.12);
  assert.equal(redeclared.five_shot_worst_length_balanced.v2_level, 0.85);
  assert.equal(redeclared.five_shot_worst_length_balanced.v3_level, 0.84);
  for (const record of Object.values(redeclared)) {
    assert.match(
      record.owner_decision,
      /re-declared for the v3 architecture by the owner on 2026-07-28/,
    );
    assert.match(
      record.owner_decision,
      /BEFORE release seed 20260734 was generated/,
    );
  }
});

test('v3 intent embeds the evaluator-printed protocol object verbatim', {
  skip: TEST_SIGNALLAB_ROOT
    ? false
    : 'set ATOMOS_RELEASE_TEST_SIGNALLAB_ROOT to a verified isolated tree',
}, () => {
  const root = join(
    tmpdir(),
    `atomos-release-v3-intent-${process.pid}-${Date.now()}`,
  );
  const fakeBin = mkdtempSync(join(tmpdir(), 'atomos-release-v3-fake-bin-'));
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
    RELEASE_SEED: '20260735',
    RELEASE_EVALUATION_PROTOCOL: 'v3',
    RELEASE_LENGTHS: '4096,8192,16384,32768',
    RELEASE_TARGET_PER_CLASS: '80',
    PATH: `${fakeBin}:${process.env.PATH ?? ''}`,
  });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /occupied-start probe generation.*failed/);
  const intent = JSON.parse(
    readFileSync(join(root, 'RELEASE_INTENT.json'), 'utf8'),
  );
  const fixture = JSON.parse(
    readFileSync(
      join(REPO, 'tools/time-domain-v3-expected-evaluation-protocol-seed20260735.json'),
      'utf8',
    ),
  );
  // The embedded object must equal the evaluator's --print-expected-protocol
  // output exactly: the sealed evaluator refuses the suite on ANY difference.
  assert.deepEqual(intent.evaluation_protocol, fixture.evaluation_protocol);
  assert.equal(intent.release_seed, 20260735);
  assert.equal(intent.status, 'in_progress');
  // The v2 provenance constant remains selectable and untouched.
  assert.equal(
    intent.evaluation_protocol.version,
    'time-domain-v3-release-evaluation-v3-dual-fusion',
  );
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
