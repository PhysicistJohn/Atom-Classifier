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
  oneShotPresetForRealization,
  productionCaptureBandwidthHz,
  publicClassForProfile,
  readGeneratorLineage,
  readOptions,
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
  assert.throws(
    () => readOptions({
      UNBOUND_SMOKE: '1',
      SIGNALLAB_ROOT: '/tmp/unrelated-signallab',
    }),
    /cannot redirect provenance/i,
  );
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
