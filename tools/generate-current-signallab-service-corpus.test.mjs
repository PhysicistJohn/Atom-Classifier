import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import {
  mkdirSync,
  mkdtempSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import test from 'node:test';

import {
  GENERATOR_BUNDLE_PATH,
  GENERATOR_LINEAGE_SCHEMA,
  GENERATOR_SOURCE_PATH,
  RECEIVER_PRESETS,
  ONE_SHOT_DETERMINISTIC_PRESETS,
  ONE_SHOT_STOCHASTIC_PRESETS,
  EXPECTED_SIGNAL_LAB_GIT_COMMIT,
  EXPECTED_SIGNAL_LAB_GIT_TREE,
  assertGeneratorLineageUnchanged,
  deterministicPhaseGrid,
  deterministicRoleSchedule,
  generatorLineageBindingSha256,
  hasOccupiedLivePrefix,
  oneShotRealizationSchedule,
  paddedCf32le,
  publicClassForProfile,
  readGeneratorLineage,
  readOptions,
  splitCounts,
} from '../.artifacts/current-corpus-generator/generate-current-signallab-service-corpus.js';

const sha256 = (value) =>
  createHash('sha256').update(value).digest('hex');

test('maps the fixed public classifier leaves without labels leaking from measurements', () => {
  assert.equal(publicClassForProfile('wifi-hr-dsss-11m'), 'dsss');
  assert.equal(publicClassForProfile('wifi-ofdm-20m'), 'ofdm');
  assert.equal(publicClassForProfile('wifi6-he-tb'), 'ofdm');
  assert.equal(publicClassForProfile('lte-etm1.1'), 'ofdm');
  assert.equal(publicClassForProfile('nr-n78-tdd-100m'), 'ofdm');
  assert.equal(publicClassForProfile('gsm-normal-burst'), 'gsm');
  assert.equal(
    publicClassForProfile('bluetooth-le-advertising'),
    'bluetooth',
  );
  assert.equal(RECEIVER_PRESETS.length, 9);
  assert.match(EXPECTED_SIGNAL_LAB_GIT_COMMIT, /^[a-f0-9]{40}$/);
  assert.match(EXPECTED_SIGNAL_LAB_GIT_TREE, /^[a-f0-9]{40}$/);
  assert.deepEqual([...RECEIVER_PRESETS], [
    'clean',
    'awgn',
    'multipath',
    'carrier-offset',
    'phase-noise',
    'iq-imbalance',
    'dc-offset',
    'pa-compression',
    'composite',
  ]);
});

test('builds deterministic unique cyclic phase grids and split schedules', () => {
  const first = deterministicPhaseGrid(2_224, 288 / 9, 20_260_729, 'wifi');
  const second = deterministicPhaseGrid(2_224, 288 / 9, 20_260_729, 'wifi');
  assert.deepEqual(first, second);
  assert.equal(new Set(first).size, first.length);
  assert.ok(first.every((phase) => phase >= 0 && phase < 2_224));

  assert.deepEqual(splitCounts(5), {
    train: 3,
    enrollment: 1,
    selection: 1,
  });
  const roles = deterministicRoleSchedule(32, 20_260_729, 'wifi');
  assert.deepEqual(roles, deterministicRoleSchedule(32, 20_260_729, 'wifi'));
  assert.equal(roles.filter((role) => role === 'train').length, 19);
  assert.equal(roles.filter((role) => role === 'enrollment').length, 6);
  assert.equal(roles.filter((role) => role === 'selection').length, 7);
});

test('full one-shot schedules never repeat seed-independent presets across roles', () => {
  const schedule = oneShotRealizationSchedule(
    288,
    20_260_729,
    'bluetooth-le-advertising',
  );
  assert.equal(schedule.length, 288);
  assert.deepEqual(
    {
      train: schedule.filter((row) => row.role === 'train').length,
      enrollment:
        schedule.filter((row) => row.role === 'enrollment').length,
      selection: schedule.filter((row) => row.role === 'selection').length,
    },
    splitCounts(288),
  );
  const clean = schedule.filter((row) => row.preset === 'clean');
  assert.equal(clean.length, 1);
  assert.equal(clean[0].role, 'enrollment');
  for (const preset of ONE_SHOT_DETERMINISTIC_PRESETS) {
    const rows = schedule.filter((row) => row.preset === preset);
    assert.equal(rows.length, 1, `${preset} must be scheduled exactly once`);
  }
  const repeatable = new Set(ONE_SHOT_STOCHASTIC_PRESETS);
  for (const preset of new Set(schedule.map((row) => row.preset))) {
    const count = schedule.filter((row) => row.preset === preset).length;
    if (count > 1) assert.ok(repeatable.has(preset), preset);
  }
  assert.deepEqual(
    schedule,
    oneShotRealizationSchedule(
      288,
      20_260_729,
      'bluetooth-le-advertising',
    ),
  );
});

test('pads only after valid cf32le and detects a nonzero live prefix', () => {
  const validSamples = 4_096;
  const valid = new Uint8Array(validSamples * 8);
  const floats = new Float32Array(
    valid.buffer,
    valid.byteOffset,
    valid.byteLength / 4,
  );
  assert.equal(hasOccupiedLivePrefix(valid), false);
  floats[17] = 0.25;
  assert.equal(hasOccupiedLivePrefix(valid), true);
  const padded = paddedCf32le(valid, 8_192);
  assert.equal(padded.byteLength, 8_192 * 8);
  assert.deepEqual(padded.subarray(0, valid.byteLength), valid);
  assert.ok(padded.subarray(valid.byteLength).every((value) => value === 0));
});

test('environment parsing is strict, fail-closed, and admits a tiny smoke target', () => {
  const options = readOptions({
    OUTPUT_DIR: 'training/artifacts/test-current',
    CORPUS_SEED: '7',
    TARGET_PER_PROFILE: '9',
    SAMPLE_COUNT: '4096',
    ALLOW_OVERWRITE: '0',
  });
  assert.equal(options.corpusSeed, 7);
  assert.equal(options.targetPerProfile, 9);
  assert.equal(options.sampleCount, 4_096);
  assert.equal(options.allowOverwrite, false);
  assert.throws(
    () => readOptions({ TARGET_PER_PROFILE: '10' }),
    /positive multiple of 9/,
  );
  assert.throws(
    () => readOptions({ ALLOW_OVERWRITE: 'yes' }),
    /exactly 0 or 1/,
  );
  assert.throws(
    () => readOptions({ SAMPLE_COUNT: '4095' }),
    /safe integer/,
  );
});

test('binds the reviewed source and exact executing bundle and detects drift', () => {
  const root = mkdtempSync(join(tmpdir(), 'current-generator-lineage-'));
  try {
    const source = Buffer.from('reviewed TypeScript source\n');
    const bundle = Buffer.from('reviewed bundled JavaScript\n');
    const sourcePath = resolve(root, GENERATOR_SOURCE_PATH);
    const bundlePath = resolve(root, GENERATOR_BUNDLE_PATH);
    mkdirSync(resolve(sourcePath, '..'), { recursive: true });
    mkdirSync(resolve(bundlePath, '..'), { recursive: true });
    writeFileSync(sourcePath, source);
    writeFileSync(bundlePath, bundle);

    const lineage = readGeneratorLineage(root, bundlePath);
    assert.deepEqual(lineage, {
      schema: GENERATOR_LINEAGE_SCHEMA,
      repository: 'Atom-Classifier',
      sourcePath: GENERATOR_SOURCE_PATH,
      sourceSha256: sha256(source),
      bundlePath: GENERATOR_BUNDLE_PATH,
      bundleSha256: sha256(bundle),
      sourceBundleBindingSha256: generatorLineageBindingSha256({
        sourceSha256: sha256(source),
        bundleSha256: sha256(bundle),
      }),
    });
    assert.throws(
      () => readGeneratorLineage(root, sourcePath),
      /canonical reviewed bundle/i,
    );

    writeFileSync(sourcePath, 'mutated source\n');
    assert.throws(
      () => assertGeneratorLineageUnchanged(lineage, root, bundlePath),
      /changed during corpus generation.*sourceSha256/i,
    );
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
