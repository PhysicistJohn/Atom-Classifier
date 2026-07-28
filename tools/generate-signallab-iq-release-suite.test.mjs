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
const RELEASE36_ROOT = join(
  REPO,
  'training/artifacts/releases/invariant_fusion_v3_sealed_seed20260736',
);
const RELEASE36_CANDIDATE = join(
  REPO,
  'training/zplane_ab/v2_full_variation/v3_scale/evidence/'
    + 'v3_4_q97_dual_release_candidate.json',
);
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

test('v3 seed is hard-bound to the frozen q97 candidate before output', () => {
  const result = run({
    RELEASE_ROOT: RELEASE36_ROOT,
    RELEASE_EVALUATION_PROTOCOL: 'v3',
    RELEASE_SEED: '20260736',
    RELEASE_TARGET_PER_CLASS: '192',
    CANDIDATE_PATH: RELEASE36_CANDIDATE,
    CANDIDATE_SHA256: '0'.repeat(64),
  });
  assert.notEqual(result.status, 0);
  assert.match(
    result.stderr,
    /requires CANDIDATE_SHA256 7f824eb734466cb697ee28387a470e8e669e19567542d928130a2b4ad9f59053/,
  );
  assert.equal(existsSync(RELEASE36_ROOT), false);
});

test('seed 20260736 refuses default and explicit v2 before any output', () => {
  for (const protocol of [undefined, 'v2']) {
    const result = run({
      RELEASE_ROOT: RELEASE36_ROOT,
      RELEASE_SEED: '20260736',
      RELEASE_TARGET_PER_CLASS: '192',
      CANDIDATE_PATH: RELEASE36_CANDIDATE,
      CANDIDATE_SHA256: '0'.repeat(64),
      ...(protocol === undefined
        ? { RELEASE_EVALUATION_PROTOCOL: '' }
        : { RELEASE_EVALUATION_PROTOCOL: protocol }),
    });
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /requires explicit RELEASE_EVALUATION_PROTOCOL=v3/);
    assert.equal(existsSync(RELEASE36_ROOT), false);
  }
});

test('seed 20260736 reserves exact root, candidate path, and target 192', () => {
  const cases = [
    {
      RELEASE_ROOT: join(tmpdir(), `wrong-release36-root-${process.pid}`),
      CANDIDATE_PATH: RELEASE36_CANDIDATE,
      RELEASE_TARGET_PER_CLASS: '192',
      pattern: /requires RELEASE_ROOT exactly/,
    },
    {
      RELEASE_ROOT: RELEASE36_ROOT,
      CANDIDATE_PATH: SCRIPT,
      RELEASE_TARGET_PER_CLASS: '192',
      pattern: /requires CANDIDATE_PATH exactly/,
    },
    {
      RELEASE_ROOT: RELEASE36_ROOT,
      CANDIDATE_PATH: RELEASE36_CANDIDATE,
      RELEASE_TARGET_PER_CLASS: '191',
      pattern: /requires RELEASE_TARGET_PER_CLASS=192/,
    },
  ];
  for (const { pattern, ...extra } of cases) {
    const result = run({
      RELEASE_SEED: '20260736',
      RELEASE_EVALUATION_PROTOCOL: 'v3',
      CANDIDATE_SHA256: '0'.repeat(64),
      ...extra,
    });
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, pattern);
    assert.equal(existsSync(RELEASE36_ROOT), false);
  }
});

test('v2 and v3 both refuse consumed seed 20260735 before any output', () => {
  const expectedReason =
    'consumed sealed v3.3 decoupled 8k-classifier/4k-rejector release suite '
    + '(22/23 gates; open_known_false_unknown_worst_length '
    + '0.11382734912146678 > 0.10 frozen; evidence rules 2 and 4)';
  for (const protocol of ['v2', 'v3']) {
    const root = join(
      tmpdir(),
      `atomos-release-${protocol}-refused-seed-20260735-`
        + `${process.pid}-${Date.now()}`,
    );
    const result = run({
      RELEASE_ROOT: root,
      RELEASE_EVALUATION_PROTOCOL: protocol,
      RELEASE_SEED: '20260735',
    });
    assert.notEqual(result.status, 0);
    assert.ok(result.stderr.includes(expectedReason), result.stderr);
    assert.equal(existsSync(root), false);
  }
});

test('v2 and v3 preserve all older consumed, model, and band refusals', () => {
  for (const [seed, pattern] of [
    ['20260729', /consumed sealed v2 release suite/],
    ['20260730', /development model\/fusion seed/],
    ['20260731', /consumed sealed v3\.0 release suite/],
    ['20260732', /next untouched release seed after the consumed runs: 20260736/],
    ['20260733', /consumed sealed v3\.2 release suite/],
    ['20260734', /historical gate redeclaration/],
    ['20260942', /development novelty seed namespace/],
    ['20261001', /noise-prefilter fit-only seed band/],
  ]) {
    for (const protocol of ['v2', 'v3']) {
      const root = join(
        tmpdir(),
        `atomos-release-${protocol}-refused-seed-${seed}-`
          + `${process.pid}-${Date.now()}`,
      );
      const result = run({
        RELEASE_ROOT: root,
        RELEASE_EVALUATION_PROTOCOL: protocol,
        RELEASE_SEED: seed,
      });
      assert.notEqual(result.status, 0);
      assert.match(result.stderr, pattern);
      assert.equal(existsSync(root), false);
    }
  }
});

test('consumed seed-20260735 fixture remains immutable historical evidence', () => {
  // This historical fixture was captured from
  //   evaluate_v3_release_suite.py --print-expected-protocol 20260735
  // and must not be rewritten for release 20260736.
  const fixturePath = join(
    REPO,
    'tools/time-domain-v3-expected-evaluation-protocol-seed20260735.json',
  );
  const fixtureBytes = readFileSync(fixturePath);
  assert.equal(
    createHash('sha256').update(fixtureBytes).digest('hex'),
    '40ef572beeabd18c38e55fdc5001dbfb0365dcd81dac5ff6ea1b95ceca163f7e',
  );
  const wrapper = JSON.parse(fixtureBytes.toString('utf8'));
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

test('launcher binds only the frozen versioned seed-20260736 fixture', () => {
  const source = readFileSync(SCRIPT, 'utf8');
  const futureName =
    'time-domain-v3-expected-evaluation-protocol-v4-q97-seed20260736.json';
  assert.match(source, new RegExp(futureName.replaceAll('.', '\\.')));
  assert.match(
    source,
    /time-domain-v3-release-evaluation-v4-q97-dual-fusion/,
  );
  const fixturePath = join(REPO, 'tools', futureName);
  assert.equal(existsSync(fixturePath), true);
  const fixtureBytes = readFileSync(fixturePath);
  assert.equal(
    createHash('sha256').update(fixtureBytes).digest('hex'),
    '40d29042f844e2169f1025b75d0a63545f669b25ab7d7f9e46ae630599668fdd',
  );
  const fixture = JSON.parse(fixtureBytes.toString('utf8'));
  assert.equal(fixture.release_seed, 20260736);
  assert.equal(fixture.evaluation_protocol.novelty.seed, 20260736);
  assert.equal(
    fixture.evaluation_protocol.version,
    'time-domain-v3-release-evaluation-v4-q97-dual-fusion',
  );
  assert.equal(
    fixture.evaluation_protocol.open_set.composite_threshold_quantile,
    0.97,
  );
  const fixtureDeclaration = source.match(
    /const V3_EXPECTED_PROTOCOL_FIXTURE = resolve\([\s\S]*?\n\);/,
  )?.[0];
  assert.ok(fixtureDeclaration);
  assert.doesNotMatch(
    fixtureDeclaration,
    /seed20260735\.json/,
  );
  assert.match(
    source,
    /const V3_EXPECTED_PROTOCOL_FIXTURE_SHA256 =\s*'40d29042f844e2169f1025b75d0a63545f669b25ab7d7f9e46ae630599668fdd';/,
  );
  assert.match(
    source,
    /const RESERVED_CANDIDATE_SHA256 =\s*'7f824eb734466cb697ee28387a470e8e669e19567542d928130a2b4ad9f59053';/,
  );
  assert.equal(
    createHash('sha256').update(readFileSync(RELEASE36_CANDIDATE)).digest('hex'),
    '7f824eb734466cb697ee28387a470e8e669e19567542d928130a2b4ad9f59053',
  );
  assert.ok(
    source.indexOf("createHash('sha256').update(fixtureBytes)")
      < source.indexOf("JSON.parse(fixtureBytes.toString('utf8'))"),
    'fixture bytes must be hash-verified before JSON.parse',
  );
  assert.match(
    source,
    /305418a5bc7bd8f9a49799477f3a457b4c07d0c58b637766989fc9557565371b/,
  );
  assert.match(
    source,
    /d9383a642d21a59f66f1f9e88fc7ce51ad50893a381985c4f53f2b0da9aec3a3/,
  );
});

// There is intentionally no v3 intent-generation integration yet:
// seed 20260735 is consumed, and the seed-20260736 evaluator-v4 fixture does
// not exist until the validation/candidate chain is complete and reviewed.

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
