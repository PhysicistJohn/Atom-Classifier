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
import { test } from 'node:test';

import {
  GENERATOR_BUNDLE_PATH,
  GENERATOR_LINEAGE_SCHEMA,
  GENERATOR_SOURCE_PATH,
  EXPECTED_SIGNAL_LAB_GIT_COMMIT,
  EXPECTED_SIGNAL_LAB_GIT_TREE,
  INFORMATIVE_PREFIX_CENTERED_RMS_FLOOR,
  PRE_IMPAIRMENT_PREFIX_ADMISSION_RULE,
  PRODUCTION_CAPTURE_BANDWIDTH_RULE,
  SCALE_FACTORS,
  SCALE_EVAL_SCHEMA,
  SCALE_FAMILY_CONTENT_UNIQUENESS_RULE,
  assertGeneratorLineageUnchanged,
  availablePrefixLengths,
  deterministicPhaseCandidates,
  generatorLineageBindingSha256,
  hasExactUniqueScaleFamilyContentHashes,
  hasInformativeCf32lePrefix,
  maximumOneShotOutputSamples,
  mergeIdentityExclusions,
  oneShotPresetForRealization,
  parseIdentityExclusionScaleCorpusDirectories,
  productionCaptureBandwidthHz,
  publicClassForProfile,
  readGeneratorLineage,
  readOptions,
  readScaleIdentityExclusionCorpus,
  scaledSampleRateHz,
} from '../.artifacts/current-scale-eval-generator/generate-current-signallab-scale-eval.js';

const sha256 = (value) =>
  createHash('sha256').update(value).digest('hex');

test('the predeclared service output-rate factors are exact rationals', () => {
  assert.deepEqual(
    SCALE_FACTORS.map(({ key, numerator, denominator }) => [
      key,
      numerator,
      denominator,
    ]),
    [
      ['1', 1, 1],
      ['1.25', 5, 4],
      ['1.5', 3, 2],
      ['2', 2, 1],
    ],
  );
  assert.deepEqual(
    SCALE_FACTORS.map((factor) => scaledSampleRateHz(11_000_000, factor)),
    [11_000_000, 13_750_000, 16_500_000, 22_000_000],
  );
  assert.throws(
    () => scaledSampleRateHz(11_000_001, SCALE_FACTORS[1]),
    /cannot represent scale 1.25/i,
  );
});

test('scale cells preserve Atomizer production capture geometry', () => {
  assert.equal(
    productionCaptureBandwidthHz({
      nativeCarrierOffsetHz: -31_000_000,
      signalBandwidthHz: 1_000_000,
    }),
    63_000_000,
  );
  assert.equal(
    productionCaptureBandwidthHz({
      nativeCarrierOffsetHz: -15_000_000,
      signalBandwidthHz: 1_000_000,
    }),
    31_000_000,
  );
  assert.equal(
    productionCaptureBandwidthHz({
      nativeCarrierOffsetHz: 0,
      signalBandwidthHz: 20_000_000,
    }),
    20_000_000,
  );
  assert.equal(
    PRODUCTION_CAPTURE_BANDWIDTH_RULE,
    '2 * abs(nativeCarrierOffsetHz) + signalBandwidthHz at every scale factor',
  );
});

test('BLE unavailable 16K cells remain unavailable evidence', () => {
  const valid = SCALE_FACTORS.map((factor) =>
    Math.min(16_384, maximumOneShotOutputSamples(12_160, factor)));
  assert.deepEqual(valid, [12_160, 15_200, 16_384, 16_384]);
  assert.deepEqual(
    valid.map(availablePrefixLengths),
    [
      [4_096, 8_192],
      [4_096, 8_192],
      [4_096, 8_192, 16_384],
      [4_096, 8_192, 16_384],
    ],
  );
});

test('held-out cyclic phase schedules exclude reference phases exactly', () => {
  const period = 2_224;
  const forbidden = new Set([0, 1, 17, 2_223]);
  const phases = deterministicPhaseCandidates(
    period,
    20_262_001,
    'wifi-hr-dsss-11m',
    forbidden,
  );
  assert.equal(phases.length, period - forbidden.size);
  assert.equal(new Set(phases).size, phases.length);
  assert.equal(phases.some((phase) => forbidden.has(phase)), false);
  assert.deepEqual(
    phases,
    deterministicPhaseCandidates(
      period,
      20_262_001,
      'wifi-hr-dsss-11m',
      forbidden,
    ),
  );
});

test('pre-impairment admission rejects idle and DC-only prefixes', () => {
  const samples = 4_096;
  const idle = Buffer.alloc(samples * 8);
  assert.equal(
    hasInformativeCf32lePrefix(
      new Uint8Array(idle.buffer, idle.byteOffset, idle.byteLength),
      samples,
    ),
    false,
  );

  const constantDc = new Float32Array(samples * 2);
  for (let sample = 0; sample < samples; sample += 1) {
    constantDc[2 * sample] = 0.08;
    constantDc[2 * sample + 1] = -0.05;
  }
  assert.equal(
    hasInformativeCf32lePrefix(
      new Uint8Array(
        constantDc.buffer,
        constantDc.byteOffset,
        constantDc.byteLength,
      ),
      samples,
    ),
    false,
  );

  constantDc[2 * (samples - 1)] = -0.08;
  assert.equal(
    hasInformativeCf32lePrefix(
      new Uint8Array(
        constantDc.buffer,
        constantDc.byteOffset,
        constantDc.byteLength,
      ),
      samples,
    ),
    true,
  );
  const tinyResidue = new Float32Array(samples * 2);
  tinyResidue.fill(0.08);
  tinyResidue[tinyResidue.length - 1] += 1e-6;
  assert.equal(
    hasInformativeCf32lePrefix(
      new Uint8Array(
        tinyResidue.buffer,
        tinyResidue.byteOffset,
        tinyResidue.byteLength,
      ),
      samples,
    ),
    false,
  );
  assert.equal(INFORMATIVE_PREFIX_CENTERED_RMS_FLOOR, 1e-4);
  assert.match(
    PRE_IMPAIRMENT_PREFIX_ADMISSION_RULE,
    /service-clean.*non-constant.*4,096.*centered complex RMS >= 0.0001.*before receiver impairment/i,
  );
});

test('a four-rate family must have exact pairwise content uniqueness', () => {
  const hashes = ['1', '2', '3', '4'].map((value) => value.repeat(64));
  assert.equal(hasExactUniqueScaleFamilyContentHashes(hashes), true);
  assert.equal(
    hasExactUniqueScaleFamilyContentHashes([
      hashes[0], hashes[1], hashes[1], hashes[3],
    ]),
    false,
  );
  assert.equal(
    hasExactUniqueScaleFamilyContentHashes(hashes.slice(0, 3)),
    false,
  );
  assert.match(
    SCALE_FAMILY_CONTENT_UNIQUENESS_RULE,
    /all four.*SHA-256.*pairwise distinct.*before publication/i,
  );
});

test('one-shot repetitions use only seed-sensitive receiver presets', () => {
  assert.deepEqual(
    Array.from({ length: 7 }, (_, index) =>
      oneShotPresetForRealization(index)),
    [
      'awgn',
      'phase-noise',
      'composite',
      'awgn',
      'phase-noise',
      'composite',
      'awgn',
    ],
  );
});

test('profile-to-public-class mapping covers the fixed families', () => {
  assert.equal(publicClassForProfile('wifi-hr-dsss-11m'), 'dsss');
  assert.equal(publicClassForProfile('wifi6-he-su'), 'ofdm');
  assert.equal(publicClassForProfile('lte-etm1.1'), 'ofdm');
  assert.equal(publicClassForProfile('nr-fr1-tm3.1'), 'ofdm');
  assert.equal(publicClassForProfile('gsm-normal-burst'), 'gsm');
  assert.equal(
    publicClassForProfile('bluetooth-le-advertising'),
    'bluetooth',
  );
});

test('real runs require a reference and profile filters are smoke-only', () => {
  assert.equal(SCALE_EVAL_SCHEMA, 'current-signallab-service-scale-eval-v3');
  assert.match(EXPECTED_SIGNAL_LAB_GIT_COMMIT, /^[a-f0-9]{40}$/);
  assert.match(EXPECTED_SIGNAL_LAB_GIT_TREE, /^[a-f0-9]{40}$/);
  assert.throws(
    () => readOptions({}),
    /REFERENCE_CORPUS_DIR is required/i,
  );
  assert.throws(
    () => readOptions({
      REFERENCE_CORPUS_DIR: '/tmp/reference',
      PROFILE_FILTER: 'wifi-hr-dsss-11m',
    }),
    /PROFILE_FILTER is allowed only with UNBOUND_SMOKE=1/i,
  );
  const options = readOptions({
    UNBOUND_SMOKE: '1',
    PROFILE_FILTER: 'wifi-hr-dsss-11m',
    OUTPUT_DIR: '/tmp/scale-eval-smoke',
  });
  assert.equal(options.unboundSmoke, true);
  assert.deepEqual(options.profiles, ['wifi-hr-dsss-11m']);
  assert.deepEqual(options.identityExclusionScaleCorpusDirectories, []);
  assert.throws(
    () => readOptions({
      UNBOUND_SMOKE: '1',
      SIGNALLAB_ROOT: '/tmp/unrelated-signallab',
    }),
    /cannot redirect provenance/i,
  );
});

test('scale identity-exclusion directories are explicit unique JSON paths', () => {
  assert.deepEqual(
    parseIdentityExclusionScaleCorpusDirectories(
      '["training/artifacts/one","training/artifacts/two"]',
      '/classifier',
    ),
    [
      '/classifier/training/artifacts/one',
      '/classifier/training/artifacts/two',
    ],
  );
  assert.throws(
    () => parseIdentityExclusionScaleCorpusDirectories(
      '["training/artifacts/one","training/artifacts/one"]',
      '/classifier',
    ),
    /duplicate paths/i,
  );
  assert.throws(
    () => parseIdentityExclusionScaleCorpusDirectories('{}', '/classifier'),
    /nonempty JSON array/i,
  );
  assert.throws(
    () => parseIdentityExclusionScaleCorpusDirectories('[', '/classifier'),
    /valid JSON/i,
  );
});

test('reviewed scale corpora contribute unioned identity exclusions', () => {
  const root = mkdtempSync(join(tmpdir(), 'scale-identity-exclusion-'));
  try {
    const makeCorpus = (name, {
      evalSeed,
      phase,
      channelSeed,
      receiverSeed,
      content,
    }) => {
      const directory = resolve(root, name);
      mkdirSync(directory, { recursive: true });
      const raw = Buffer.from(`bound-scale-raw:${name}\n`);
      writeFileSync(resolve(directory, 'scale_eval.f32'), raw);
      writeFileSync(
        resolve(directory, 'scale_eval.json'),
        `${JSON.stringify({
          schema: SCALE_EVAL_SCHEMA,
          generator: SCALE_EVAL_SCHEMA,
          evalSeed,
          count: 1,
          dataFile: 'scale_eval.f32',
          dataSha256: sha256(raw),
          source: {
            gitCommit: EXPECTED_SIGNAL_LAB_GIT_COMMIT,
            gitTree: EXPECTED_SIGNAL_LAB_GIT_TREE,
          },
          items: [{
            profile: 'wifi-hr-dsss-11m',
            phaseNativeSample: phase,
            receiverRealizationChannelSeed: channelSeed,
            receiverRealizationSeed: receiverSeed,
            contentSha256: content,
          }],
        })}\n`,
      );
      return readScaleIdentityExclusionCorpus(directory);
    };
    const first = makeCorpus('one', {
      evalSeed: 20_264_101,
      phase: 17,
      channelSeed: 101,
      receiverSeed: 201,
      content: '1'.repeat(64),
    });
    const second = makeCorpus('two', {
      evalSeed: 20_262_904,
      phase: 23,
      channelSeed: 102,
      receiverSeed: null,
      content: '2'.repeat(64),
    });
    const reference = {
      forbiddenPhases: new Map([
        ['wifi-hr-dsss-11m', new Set([11])],
      ]),
      forbiddenChannelSeeds: new Set([100]),
      forbiddenReceiverSeeds: new Set([200]),
      forbiddenContentHashes: new Set(['0'.repeat(64)]),
    };
    const merged = mergeIdentityExclusions([reference, first, second]);
    assert.deepEqual(
      [...merged.forbiddenPhases.get('wifi-hr-dsss-11m')].sort(
        (left, right) => left - right,
      ),
      [11, 17, 23],
    );
    assert.deepEqual(
      [...merged.forbiddenChannelSeeds].sort((left, right) => left - right),
      [100, 101, 102],
    );
    assert.deepEqual(
      [...merged.forbiddenReceiverSeeds].sort((left, right) => left - right),
      [200, 201],
    );
    assert.deepEqual(
      [...merged.forbiddenContentHashes].sort(),
      ['0'.repeat(64), '1'.repeat(64), '2'.repeat(64)],
    );
    assert.equal(first.evalSeed, 20_264_101);
    assert.match(first.manifestSha256, /^[a-f0-9]{64}$/);
    writeFileSync(
      resolve(first.directory, 'scale_eval.f32'),
      'post-manifest raw mutation',
    );
    assert.throws(
      () => readScaleIdentityExclusionCorpus(first.directory),
      /raw SHA-256 disagrees with its manifest/i,
    );
    assert.equal(mergeIdentityExclusions([]), undefined);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('binds the scale source and bundle and rejects mid-run drift', () => {
  const root = mkdtempSync(join(tmpdir(), 'scale-generator-lineage-'));
  try {
    const source = Buffer.from('reviewed scale TypeScript source\n');
    const bundle = Buffer.from('reviewed scale JavaScript bundle\n');
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

    writeFileSync(bundlePath, 'mutated bundle\n');
    assert.throws(
      () => assertGeneratorLineageUnchanged(lineage, root, bundlePath),
      /changed during scale-eval generation.*bundleSha256/i,
    );
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
