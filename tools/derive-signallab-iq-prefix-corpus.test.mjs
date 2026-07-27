import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  realpathSync,
  symlinkSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import test from 'node:test';

const REPO = resolve(import.meta.dirname, '..');
const SCRIPT = join(REPO, 'tools/derive-signallab-iq-prefix-corpus.mjs');
const LIVE_CORPUS = join(REPO, 'training/artifacts/signallab-corpus');

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function makeBytes(length, offset = 0) {
  return Buffer.from(
    Array.from({ length }, (_, index) => (offset + index) & 0xff),
  );
}

function fixture(options = {}) {
  const root = realpathSync(
    mkdtempSync(join(tmpdir(), 'atomos-prefix-corpus-')),
  );
  const source = join(root, 'source');
  const output = join(root, 'output');
  mkdirSync(source);
  const sourceSampleCount = options.sourceSampleCount ?? 4;
  const count = options.count ?? 2;
  const manifest = options.manifest ?? {
    corpusSeed: 123,
    sampleCount: sourceSampleCount,
    format: 'cf32le-interleaved',
    classes: ['am', 'fm'],
    count,
    nuisanceSpec: {
      captureLengthSamples: 'FIXED at SAMPLE_COUNT',
    },
    hasCleanPairs: true,
    items: [
      {
        cls: 'am',
        profile: 'am-standard',
        sampleRateHz: 2_000_000,
        bandwidthHz: 200_000,
        impaired: true,
        startSampleIndex: 10,
        centreOffsetFrac: 0.125,
      },
      {
        cls: 'fm',
        profile: 'fm-standard',
        sampleRateHz: 4_000_000,
        bandwidthHz: 400_000,
        impaired: false,
        startSampleIndex: 20,
        adcBits: null,
      },
    ],
  };
  const rowBytes = sourceSampleCount * 8;
  const main = options.main ?? makeBytes(count * rowBytes, 0);
  const clean = options.clean ?? makeBytes(count * rowBytes, 97);
  const manifestText = options.manifestText
    ?? `${JSON.stringify(manifest, null, 2)}\n`;
  writeFileSync(join(source, 'corpus.json'), manifestText);
  writeFileSync(join(source, 'corpus.f32'), main);
  writeFileSync(join(source, 'corpus_clean.f32'), clean);
  return { root, source, output, manifest, manifestText, main, clean };
}

function run({ source, output, sampleCount = '2' }) {
  return spawnSync(process.execPath, [SCRIPT], {
    cwd: REPO,
    env: {
      ...process.env,
      SOURCE_DIR: source,
      OUTPUT_DIR: output,
      SAMPLE_COUNT: sampleCount,
    },
    encoding: 'utf8',
  });
}

function rowPrefixes(bytes, count, sourceSamples, targetSamples) {
  const sourceRowBytes = sourceSamples * 8;
  const targetRowBytes = targetSamples * 8;
  return Buffer.concat(
    Array.from(
      { length: count },
      (_, row) => bytes.subarray(
        row * sourceRowBytes,
        row * sourceRowBytes + targetRowBytes,
      ),
    ),
  );
}

test('derives exact row-wise prefixes and records independently verifiable hashes', () => {
  const data = fixture();
  const result = run(data);
  assert.equal(result.status, 0, result.stderr);

  const expectedMain = rowPrefixes(data.main, 2, 4, 2);
  const expectedClean = rowPrefixes(data.clean, 2, 4, 2);
  const actualMain = readFileSync(join(data.output, 'corpus.f32'));
  const actualClean = readFileSync(join(data.output, 'corpus_clean.f32'));
  assert.deepEqual(actualMain, expectedMain);
  assert.deepEqual(actualClean, expectedClean);

  const output = JSON.parse(
    readFileSync(join(data.output, 'corpus.json'), 'utf8'),
  );
  assert.equal(output.sampleCount, 2);
  assert.equal(output.count, 2);
  assert.deepEqual(output.items, data.manifest.items);
  assert.equal(output.prefixDerivation.protocol, 'signallab-row-prefix-v1');
  assert.equal(output.prefixDerivation.sourceDirectory, data.source);
  assert.equal(output.prefixDerivation.sourceSampleCount, 4);
  assert.equal(output.prefixDerivation.targetSampleCount, 2);
  assert.equal(output.prefixDerivation.rowCount, 2);
  assert.equal(output.prefixDerivation.bytesPerComplexSample, 8);
  assert.match(output.prefixDerivation.semantics, /exactly the first/);
  assert.match(output.nuisanceSpec.captureLengthSamples, /no regeneration/);

  const sourceFiles = output.prefixDerivation.sourceFiles;
  assert.equal(sourceFiles['corpus.json'].sha256, sha256(data.manifestText));
  assert.equal(sourceFiles['corpus.f32'].sha256, sha256(data.main));
  assert.equal(sourceFiles['corpus_clean.f32'].sha256, sha256(data.clean));
  assert.equal(sourceFiles['corpus.f32'].bytes, data.main.length);
  assert.equal(sourceFiles['corpus_clean.f32'].bytes, data.clean.length);

  const outputFiles = output.prefixDerivation.outputFiles;
  assert.equal(outputFiles['corpus.f32'].sha256, sha256(expectedMain));
  assert.equal(outputFiles['corpus_clean.f32'].sha256, sha256(expectedClean));
  assert.equal(outputFiles['corpus.f32'].bytes, expectedMain.length);
  assert.equal(outputFiles['corpus_clean.f32'].bytes, expectedClean.length);
});

test('requires a strict finite decimal target shorter than the source', async (t) => {
  for (const sampleCount of ['Infinity', 'NaN', '1e2', '2.0', '-1', '4', '5']) {
    await t.test(sampleCount, () => {
      const data = fixture();
      const result = run({ ...data, sampleCount });
      assert.notEqual(result.status, 0);
      assert.match(result.stderr, /SAMPLE_COUNT/);
      assert.equal(existsSync(data.output), false);
    });
  }
});

test('rejects non-finite and unsafe manifest metadata', async (t) => {
  for (const [label, manifestText] of [
    [
      'nonfinite sampleCount',
      '{"sampleCount":1e999,"format":"cf32le-interleaved","count":2,'
      + '"classes":["am","fm"],"hasCleanPairs":true,"items":[]}',
    ],
    [
      'unsafe nested integer',
      '{"sampleCount":4,"format":"cf32le-interleaved","count":2,'
      + '"classes":["am","fm"],"hasCleanPairs":true,'
      + '"nuisanceSpec":{"bad":9007199254740992},"items":[]}',
    ],
  ]) {
    await t.test(label, () => {
      const data = fixture({ manifestText });
      const result = run(data);
      assert.notEqual(result.status, 0);
      assert.match(result.stderr, /non-finite|unsafe|finite safe integer/);
      assert.equal(existsSync(data.output), false);
    });
  }
});

test('validates manifest row metadata and item count', async (t) => {
  const base = fixture().manifest;
  for (const [label, manifest, error] of [
    [
      'item count mismatch',
      { ...base, count: 3 },
      /items length must equal count/,
    ],
    [
      'unknown class',
      {
        ...base,
        items: [{ ...base.items[0], cls: 'unknown' }, base.items[1]],
      },
      /cls is not in classes/,
    ],
    [
      'malformed start',
      {
        ...base,
        items: [{ ...base.items[0], startSampleIndex: -1 }, base.items[1]],
      },
      /startSampleIndex/,
    ],
  ]) {
    await t.test(label, () => {
      const data = fixture({ manifest });
      const result = run(data);
      assert.notEqual(result.status, 0);
      assert.match(result.stderr, error);
      assert.equal(existsSync(data.output), false);
    });
  }
});

test('requires exact geometry for both source binaries before creating output', async (t) => {
  for (const name of ['corpus.f32', 'corpus_clean.f32']) {
    await t.test(name, () => {
      const data = fixture({
        ...(name === 'corpus.f32'
          ? { main: makeBytes(63) }
          : { clean: makeBytes(65) }),
      });
      const result = run(data);
      assert.notEqual(result.status, 0);
      assert.match(result.stderr, new RegExp(`${name.replace('.', '\\.')} size`));
      assert.equal(existsSync(data.output), false);
    });
  }
});

test('refuses non-empty output without changing existing bytes', () => {
  const data = fixture();
  mkdirSync(data.output);
  const witness = Buffer.from('preserve me exactly\n');
  writeFileSync(join(data.output, 'witness'), witness);
  const result = run(data);
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /must be empty; refusing overwrite/);
  assert.deepEqual(readFileSync(join(data.output, 'witness')), witness);
  assert.equal(existsSync(join(data.output, 'corpus.json')), false);
});

test('refuses the live training corpus as either source or output', async (t) => {
  await t.test('source', () => {
    const root = realpathSync(
      mkdtempSync(join(tmpdir(), 'atomos-prefix-live-source-')),
    );
    const result = run({
      source: LIVE_CORPUS,
      output: join(root, 'output'),
    });
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /SOURCE_DIR may not be the live training corpus/);
  });
  await t.test('output', () => {
    const data = fixture();
    const result = run({ ...data, output: LIVE_CORPUS });
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /OUTPUT_DIR may not be the live training corpus/);
  });
});

test('refuses source, output, and overlap aliases reached through symlinks', async (t) => {
  await t.test('source alias', () => {
    const data = fixture();
    const alias = join(data.root, 'source-alias');
    symlinkSync(data.source, alias, 'dir');
    const result = run({ ...data, source: alias });
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /SOURCE_DIR may not use a symlink alias/);
  });
  await t.test('output alias', () => {
    const data = fixture();
    const actualOutput = join(data.root, 'actual-output');
    mkdirSync(actualOutput);
    const alias = join(data.root, 'output-alias');
    symlinkSync(actualOutput, alias, 'dir');
    const result = run({ ...data, output: alias });
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /OUTPUT_DIR may not use a symlink alias/);
  });
  await t.test('output inside source', () => {
    const data = fixture();
    const result = run({ ...data, output: join(data.source, 'derived') });
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /may not overlap or alias/);
  });
});
