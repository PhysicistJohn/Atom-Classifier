/**
 * Derive a shorter SignalLab I/Q corpus as exact row-wise byte prefixes of a
 * longer corpus.
 *
 * Required environment:
 *   SOURCE_DIR   existing source corpus directory
 *   OUTPUT_DIR   new or empty destination directory
 *   SAMPLE_COUNT target complex-sample count (strictly less than the source)
 *
 * This tool is intentionally fail-closed:
 *   - it never overwrites an output file;
 *   - it rejects the live training corpus and symlink aliases;
 *   - it validates both source binaries before creating output;
 *   - it writes corpus.json only after both binaries have been copied and
 *     read-back hashes have been verified.
 *
 * The binary derivation is bounded-memory. Every source byte is streamed into a
 * SHA-256 digest, but only the first SAMPLE_COUNT complex samples from each row
 * are written to the derived binary.
 */

import { createHash } from 'node:crypto';
import {
  closeSync,
  createReadStream,
  existsSync,
  fstatSync,
  fsyncSync,
  lstatSync,
  mkdirSync,
  openSync,
  readFileSync,
  readdirSync,
  readSync,
  realpathSync,
  statSync,
  writeFileSync,
  writeSync,
} from 'node:fs';
import {
  basename,
  dirname,
  join,
  resolve,
  sep,
} from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, '..');
const LIVE_CORPUS = resolve(REPO, 'training/artifacts/signallab-corpus');
const PROTOCOL = 'signallab-row-prefix-v1';
const FORMAT = 'cf32le-interleaved';
const BYTES_PER_COMPLEX_SAMPLE = 8;
const MAX_MANIFEST_BYTES = 256 * 1024 * 1024;
const COPY_BUFFER_BYTES = 1024 * 1024;
const BINARY_NAMES = Object.freeze(['corpus.f32', 'corpus_clean.f32']);

function required(name) {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`${name} is required`);
  return value;
}

function decimalSafeInteger(name, raw, minimum = 0) {
  if (!/^(0|[1-9][0-9]*)$/.test(raw)) {
    throw new RangeError(`${name} must be a finite decimal safe integer >= ${minimum}`);
  }
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value < minimum) {
    throw new RangeError(`${name} must be a finite decimal safe integer >= ${minimum}`);
  }
  return value;
}

function manifestSafeInteger(name, value, minimum = 0) {
  if (!Number.isFinite(value) || !Number.isSafeInteger(value) || value < minimum) {
    throw new RangeError(`${name} must be a finite safe integer >= ${minimum}`);
  }
  return value;
}

function checkedProduct(name, ...values) {
  let product = 1;
  for (const value of values) {
    if (
      !Number.isSafeInteger(value)
      || value < 0
      || (value !== 0 && product > Number.MAX_SAFE_INTEGER / value)
    ) {
      throw new RangeError(`${name} exceeds the safe integer range`);
    }
    product *= value;
  }
  if (!Number.isSafeInteger(product)) {
    throw new RangeError(`${name} exceeds the safe integer range`);
  }
  return product;
}

function canonicalProspectivePath(path) {
  let cursor = resolve(path);
  const suffix = [];
  while (!existsSync(cursor)) {
    const parent = dirname(cursor);
    if (parent === cursor) {
      throw new Error(`cannot resolve an existing parent for ${path}`);
    }
    suffix.unshift(basename(cursor));
    cursor = parent;
  }
  return resolve(realpathSync(cursor), ...suffix);
}

function canonicalInputPath(name, raw, mustExist) {
  const lexical = resolve(raw);
  if (mustExist && !existsSync(lexical)) {
    throw new Error(`${name} does not exist: ${lexical}`);
  }
  const canonical = mustExist ? realpathSync(lexical) : canonicalProspectivePath(lexical);
  if (canonical !== lexical) {
    throw new Error(`${name} may not use a symlink alias: ${lexical}`);
  }
  return canonical;
}

function containsPath(parent, child) {
  return child === parent || child.startsWith(`${parent}${sep}`);
}

function pathsOverlap(left, right) {
  return containsPath(left, right) || containsPath(right, left);
}

function assertRegularNonSymlink(path, label) {
  const linkStat = lstatSync(path);
  if (linkStat.isSymbolicLink()) {
    throw new Error(`${label} may not be a symlink: ${path}`);
  }
  if (!linkStat.isFile()) {
    throw new Error(`${label} must be a regular file: ${path}`);
  }
  const canonical = realpathSync(path);
  if (canonical !== resolve(path)) {
    throw new Error(`${label} may not use a symlink alias: ${path}`);
  }
}

function assertFiniteMetadata(value, path = 'manifest') {
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      throw new Error(`${path} contains a non-finite number`);
    }
    if (Number.isInteger(value) && !Number.isSafeInteger(value)) {
      throw new Error(`${path} contains an unsafe integer`);
    }
    return;
  }
  if (Array.isArray(value)) {
    for (let index = 0; index < value.length; index += 1) {
      assertFiniteMetadata(value[index], `${path}[${index}]`);
    }
    return;
  }
  if (value !== null && typeof value === 'object') {
    for (const [key, child] of Object.entries(value)) {
      assertFiniteMetadata(child, `${path}.${key}`);
    }
  }
}

function validateManifest(manifest) {
  if (
    manifest === null
    || typeof manifest !== 'object'
    || Array.isArray(manifest)
  ) {
    throw new Error('source corpus.json must contain a JSON object');
  }
  assertFiniteMetadata(manifest);

  const sourceSampleCount = manifestSafeInteger(
    'source manifest sampleCount',
    manifest.sampleCount,
    1,
  );
  const count = manifestSafeInteger('source manifest count', manifest.count, 1);
  if (manifest.format !== FORMAT) {
    throw new Error(`source manifest format must be ${FORMAT}`);
  }
  if (manifest.hasCleanPairs !== true) {
    throw new Error('source manifest must declare hasCleanPairs: true');
  }
  if (!Array.isArray(manifest.classes) || manifest.classes.length === 0) {
    throw new Error('source manifest classes must be a non-empty array');
  }
  const classes = new Set();
  for (const [index, cls] of manifest.classes.entries()) {
    if (typeof cls !== 'string' || cls.trim() === '') {
      throw new Error(`source manifest classes[${index}] must be a non-empty string`);
    }
    if (classes.has(cls)) {
      throw new Error(`source manifest classes contains duplicate ${JSON.stringify(cls)}`);
    }
    classes.add(cls);
  }
  if (!Array.isArray(manifest.items) || manifest.items.length !== count) {
    throw new Error(
      `source manifest items length must equal count (${count})`,
    );
  }
  for (const [index, item] of manifest.items.entries()) {
    if (item === null || typeof item !== 'object' || Array.isArray(item)) {
      throw new Error(`source manifest items[${index}] must be an object`);
    }
    if (typeof item.cls !== 'string' || !classes.has(item.cls)) {
      throw new Error(`source manifest items[${index}].cls is not in classes`);
    }
    if (typeof item.profile !== 'string' || item.profile.trim() === '') {
      throw new Error(`source manifest items[${index}].profile must be a non-empty string`);
    }
    if (!Number.isFinite(item.sampleRateHz) || item.sampleRateHz <= 0) {
      throw new Error(`source manifest items[${index}].sampleRateHz must be finite and positive`);
    }
    if (!Number.isFinite(item.bandwidthHz) || item.bandwidthHz <= 0) {
      throw new Error(`source manifest items[${index}].bandwidthHz must be finite and positive`);
    }
    if (typeof item.impaired !== 'boolean') {
      throw new Error(`source manifest items[${index}].impaired must be boolean`);
    }
    manifestSafeInteger(
      `source manifest items[${index}].startSampleIndex`,
      item.startSampleIndex,
      0,
    );
  }
  return { sourceSampleCount, count };
}

async function sha256File(path) {
  const hash = createHash('sha256');
  await new Promise((accept, reject) => {
    const stream = createReadStream(path);
    stream.on('data', (chunk) => hash.update(chunk));
    stream.on('end', accept);
    stream.on('error', reject);
  });
  return hash.digest('hex');
}

function sameFileSnapshot(before, after) {
  for (const key of ['dev', 'ino', 'size', 'mtimeNs', 'ctimeNs']) {
    if (before[key] !== after[key]) return false;
  }
  return true;
}

function writeAll(fd, bytes) {
  let offset = 0;
  while (offset < bytes.length) {
    const written = writeSync(fd, bytes, offset, bytes.length - offset);
    if (written <= 0) throw new Error('short write while deriving prefix corpus');
    offset += written;
  }
}

function deriveBinary({
  sourcePath,
  outputPath,
  count,
  sourceRowBytes,
  targetRowBytes,
  expectedSourceBytes,
  expectedOutputBytes,
}) {
  const sourceFd = openSync(sourcePath, 'r');
  let outputFd;
  try {
    const before = fstatSync(sourceFd, { bigint: true });
    if (!before.isFile() || before.size !== BigInt(expectedSourceBytes)) {
      throw new Error(
        `${basename(sourcePath)} changed after geometry validation`,
      );
    }
    outputFd = openSync(outputPath, 'wx');
    const sourceHash = createHash('sha256');
    const outputHash = createHash('sha256');
    const buffer = Buffer.allocUnsafe(
      Math.max(1, Math.min(COPY_BUFFER_BYTES, sourceRowBytes)),
    );
    let outputBytes = 0;

    for (let row = 0; row < count; row += 1) {
      let rowOffset = 0;
      while (rowOffset < sourceRowBytes) {
        const wanted = Math.min(buffer.length, sourceRowBytes - rowOffset);
        const got = readSync(sourceFd, buffer, 0, wanted, null);
        if (got !== wanted) {
          throw new Error(
            `${basename(sourcePath)} ended inside row ${row}: wanted ${wanted}, got ${got}`,
          );
        }
        const chunk = buffer.subarray(0, got);
        sourceHash.update(chunk);
        const prefixBytes = Math.max(
          0,
          Math.min(got, targetRowBytes - rowOffset),
        );
        if (prefixBytes > 0) {
          const prefix = chunk.subarray(0, prefixBytes);
          writeAll(outputFd, prefix);
          outputHash.update(prefix);
          outputBytes += prefixBytes;
        }
        rowOffset += got;
      }
    }
    const eofProbe = Buffer.allocUnsafe(1);
    if (readSync(sourceFd, eofProbe, 0, 1, null) !== 0) {
      throw new Error(`${basename(sourcePath)} contains trailing bytes`);
    }
    if (outputBytes !== expectedOutputBytes) {
      throw new Error(
        `${basename(outputPath)} derived byte count ${outputBytes} `
        + `does not match ${expectedOutputBytes}`,
      );
    }
    fsyncSync(outputFd);
    const outputStat = fstatSync(outputFd, { bigint: true });
    if (outputStat.size !== BigInt(expectedOutputBytes)) {
      throw new Error(
        `${basename(outputPath)} size ${outputStat.size} does not match ${expectedOutputBytes}`,
      );
    }
    const after = fstatSync(sourceFd, { bigint: true });
    if (!sameFileSnapshot(before, after)) {
      throw new Error(`${basename(sourcePath)} changed during derivation`);
    }
    return {
      source: {
        bytes: expectedSourceBytes,
        sha256: sourceHash.digest('hex'),
      },
      output: {
        bytes: expectedOutputBytes,
        sha256: outputHash.digest('hex'),
      },
    };
  } finally {
    if (outputFd !== undefined) closeSync(outputFd);
    closeSync(sourceFd);
  }
}

const sourceDir = canonicalInputPath('SOURCE_DIR', required('SOURCE_DIR'), true);
const outputDir = canonicalInputPath('OUTPUT_DIR', required('OUTPUT_DIR'), false);
const targetSampleCount = decimalSafeInteger(
  'SAMPLE_COUNT',
  required('SAMPLE_COUNT'),
  1,
);
const canonicalLiveCorpus = canonicalProspectivePath(LIVE_CORPUS);

const sourceDirStat = lstatSync(sourceDir);
if (sourceDirStat.isSymbolicLink() || !sourceDirStat.isDirectory()) {
  throw new Error(`SOURCE_DIR must be a real directory: ${sourceDir}`);
}
if (pathsOverlap(canonicalLiveCorpus, sourceDir)) {
  throw new Error(
    'SOURCE_DIR may not be the live training corpus, one of its descendants, '
    + 'or one of its ancestors',
  );
}
if (pathsOverlap(canonicalLiveCorpus, outputDir)) {
  throw new Error(
    'OUTPUT_DIR may not be the live training corpus, one of its descendants, '
    + 'or one of its ancestors',
  );
}
if (pathsOverlap(sourceDir, outputDir)) {
  throw new Error('SOURCE_DIR and OUTPUT_DIR may not overlap or alias each other');
}

if (existsSync(outputDir)) {
  const outputStat = lstatSync(outputDir);
  if (outputStat.isSymbolicLink() || !outputStat.isDirectory()) {
    throw new Error(`OUTPUT_DIR must be a real directory: ${outputDir}`);
  }
  if (readdirSync(outputDir).length !== 0) {
    throw new Error(`OUTPUT_DIR must be empty; refusing overwrite: ${outputDir}`);
  }
}

const manifestPath = join(sourceDir, 'corpus.json');
assertRegularNonSymlink(manifestPath, 'source corpus.json');
const manifestStat = statSync(manifestPath);
if (
  !Number.isSafeInteger(manifestStat.size)
  || manifestStat.size <= 0
  || manifestStat.size > MAX_MANIFEST_BYTES
) {
  throw new Error(
    `source corpus.json size must be between 1 and ${MAX_MANIFEST_BYTES} bytes`,
  );
}
const manifestBytes = readFileSync(manifestPath);
const manifestSha256 = createHash('sha256').update(manifestBytes).digest('hex');
let sourceManifest;
try {
  sourceManifest = JSON.parse(manifestBytes.toString('utf8'));
} catch (error) {
  throw new Error(`source corpus.json is not valid JSON: ${error.message}`);
}
const { sourceSampleCount, count } = validateManifest(sourceManifest);
if (targetSampleCount >= sourceSampleCount) {
  throw new Error(
    `SAMPLE_COUNT must be strictly less than source sampleCount `
    + `(${targetSampleCount} >= ${sourceSampleCount})`,
  );
}

const sourceRowBytes = checkedProduct(
  'source row byte count',
  sourceSampleCount,
  BYTES_PER_COMPLEX_SAMPLE,
);
const targetRowBytes = checkedProduct(
  'target row byte count',
  targetSampleCount,
  BYTES_PER_COMPLEX_SAMPLE,
);
const expectedSourceBytes = checkedProduct(
  'source binary byte count',
  count,
  sourceSampleCount,
  BYTES_PER_COMPLEX_SAMPLE,
);
const expectedOutputBytes = checkedProduct(
  'output binary byte count',
  count,
  targetSampleCount,
  BYTES_PER_COMPLEX_SAMPLE,
);

for (const name of BINARY_NAMES) {
  const path = join(sourceDir, name);
  assertRegularNonSymlink(path, `source ${name}`);
  const size = statSync(path).size;
  if (!Number.isSafeInteger(size) || size !== expectedSourceBytes) {
    throw new Error(
      `source ${name} size ${size} does not match `
      + `count * sampleCount * 8 = ${expectedSourceBytes}`,
    );
  }
}

if (!existsSync(outputDir)) {
  mkdirSync(outputDir, { recursive: true });
}
if (readdirSync(outputDir).length !== 0) {
  throw new Error(`OUTPUT_DIR ceased to be empty; refusing overwrite: ${outputDir}`);
}

const binaryEvidence = {};
for (const name of BINARY_NAMES) {
  binaryEvidence[name] = deriveBinary({
    sourcePath: join(sourceDir, name),
    outputPath: join(outputDir, name),
    count,
    sourceRowBytes,
    targetRowBytes,
    expectedSourceBytes,
    expectedOutputBytes,
  });
  const verifiedOutputSha256 = await sha256File(join(outputDir, name));
  if (verifiedOutputSha256 !== binaryEvidence[name].output.sha256) {
    throw new Error(`${name} read-back SHA-256 mismatch`);
  }
  binaryEvidence[name].output.verifiedSha256 = verifiedOutputSha256;
}

const finalSourceManifestSha256 = await sha256File(manifestPath);
if (finalSourceManifestSha256 !== manifestSha256) {
  throw new Error('source corpus.json changed during derivation');
}
const finalOutputEntries = readdirSync(outputDir).sort();
if (
  finalOutputEntries.length !== BINARY_NAMES.length
  || BINARY_NAMES.some((name, index) => name !== finalOutputEntries[index])
) {
  throw new Error(
    'OUTPUT_DIR changed during derivation; refusing to write corpus.json',
  );
}
for (const name of BINARY_NAMES) {
  assertRegularNonSymlink(join(outputDir, name), `derived ${name}`);
}

const outputManifest = {
  ...sourceManifest,
  sampleCount: targetSampleCount,
  nuisanceSpec: sourceManifest.nuisanceSpec
    && typeof sourceManifest.nuisanceSpec === 'object'
    && !Array.isArray(sourceManifest.nuisanceSpec)
    ? {
        ...sourceManifest.nuisanceSpec,
        captureLengthSamples:
          `DERIVED exact row-wise prefix: ${targetSampleCount} complex samples `
          + `from a ${sourceSampleCount}-sample source row; no regeneration, `
          + 'resampling, padding, or nuisance redraw',
      }
    : sourceManifest.nuisanceSpec,
  prefixDerivation: {
    protocol: PROTOCOL,
    semantics:
      'output row i is exactly the first targetSampleCount cf32le complex '
      + 'samples (targetSampleCount * 8 bytes) of source row i; row order and '
      + 'item metadata are unchanged',
    sourceDirectory: sourceDir,
    sourceSampleCount,
    targetSampleCount,
    rowCount: count,
    bytesPerComplexSample: BYTES_PER_COMPLEX_SAMPLE,
    sourceFiles: {
      'corpus.json': {
        bytes: manifestBytes.length,
        sha256: manifestSha256,
      },
      ...Object.fromEntries(
        BINARY_NAMES.map((name) => [name, binaryEvidence[name].source]),
      ),
    },
    outputFiles: Object.fromEntries(
      BINARY_NAMES.map((name) => [
        name,
        {
          bytes: binaryEvidence[name].output.bytes,
          sha256: binaryEvidence[name].output.verifiedSha256,
        },
      ]),
    ),
  },
  items: sourceManifest.items,
};
writeFileSync(
  join(outputDir, 'corpus.json'),
  `${JSON.stringify(outputManifest, null, 2)}\n`,
  { flag: 'wx' },
);

console.log(
  `derived ${count} exact row prefixes `
  + `(${sourceSampleCount} -> ${targetSampleCount} complex samples) in ${outputDir}`,
);
