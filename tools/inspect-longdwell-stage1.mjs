/**
 * Inspect a stage-1 long-dwell corpus without loading it all into memory.
 *
 * This is evidence for capture/phase diversity only. Distinct raw-row hashes
 * do not prove payload/content diversity: a cyclic waveform at different
 * phases also hashes differently. The report therefore deliberately calls
 * those values `raw` rather than `content` hashes.
 *
 * Usage:
 *   CORPUS_DIR=training/artifacts/longdwell-corpus-v2-proof \
 *     node tools/inspect-longdwell-stage1.mjs
 *
 * Optional OUT_JSON writes the same report as an artifact.
 */
import {
  closeSync,
  fstatSync,
  openSync,
  readFileSync,
  readSync,
  writeFileSync,
} from 'node:fs';
import { createHash } from 'node:crypto';
import { resolve, join } from 'node:path';

const corpusDir = resolve(
  process.env.CORPUS_DIR
    ?? 'training/artifacts/longdwell-probe-corpus',
);
const manifestPath = join(corpusDir, 'manifest_stage1.json');
const rawPath = join(corpusDir, 'clean_native.f32');
const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));

if (!Array.isArray(manifest.items) || manifest.items.length === 0) {
  throw new Error(`${manifestPath} does not contain a non-empty items array`);
}
if (!Number.isSafeInteger(manifest.totalBytes) || manifest.totalBytes < 0) {
  throw new Error(`${manifestPath} has an invalid totalBytes value`);
}

const fd = openSync(rawPath, 'r');
try {
  const actualBytes = fstatSync(fd).size;
  if (actualBytes !== manifest.totalBytes) {
    throw new Error(
      `${rawPath} has ${actualBytes} bytes; manifest declares ${manifest.totalBytes}`,
    );
  }

  const profiles = new Map();
  for (const row of manifest.items) {
    validateRow(row);
    const bytes = Buffer.allocUnsafe(row.byteLength);
    const received = readSync(fd, bytes, 0, bytes.length, row.byteOffset);
    if (received !== bytes.length) {
      throw new Error(
        `${row.profile} row at byte ${row.byteOffset} truncated: ${received}/${bytes.length}`,
      );
    }
    const evidence = inspectRow(bytes);
    const result = {
      row: row.offsetDrawIndex ?? null,
      startSampleIndex: row.startSampleIndex,
      offsetJitter: row.offsetJitter ?? null,
      rawSha256: createHash('sha256').update(bytes).digest('hex'),
      rms: evidence.rms,
      peak: evidence.peak,
      activeFraction: evidence.activeFraction,
      zero: evidence.zero,
      // Future content-variation work must add a fixed-phase, seed-specific
      // content probe. Do not accidentally promote a raw capture hash into
      // content evidence.
      contentSeed: row.contentSeed ?? null,
      contentProbeSha256: row.contentProbeSha256 ?? null,
    };
    const entries = profiles.get(row.profile) ?? [];
    entries.push(result);
    profiles.set(row.profile, entries);
  }

  const report = {
    schema: 'longdwell-stage1-inspection-v1',
    corpusDir,
    stage1: {
      schema: manifest.schema,
      durationMs: manifest.durationMs,
      rowsPerProfile: manifest.rowsPerProfile,
      totalRows: manifest.totalRows,
      totalBytes: manifest.totalBytes,
      offsetMode: manifest.offsetMode ?? null,
      offsetSeed: manifest.offsetSeed ?? null,
      offsetSpacing: manifest.offsetSpacing ?? null,
      planSource: manifest.planSource ?? null,
    },
    profiles: Object.fromEntries([...profiles]
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([profile, rows]) => [profile, summarizeProfile(rows)])),
    caveat: 'rawSha256 proves raw capture diversity only. It does not prove content diversity unless contentProbeSha256 is present and independently checked at a fixed phase.',
  };

  const output = JSON.stringify(report, null, 2);
  if (process.env.OUT_JSON) {
    writeFileSync(resolve(process.env.OUT_JSON), `${output}\n`);
  }
  console.log(output);
} finally {
  closeSync(fd);
}

function validateRow(row) {
  if (typeof row !== 'object' || row === null
    || typeof row.profile !== 'string'
    || !Number.isSafeInteger(row.sampleCount) || row.sampleCount < 1
    || !Number.isSafeInteger(row.byteOffset) || row.byteOffset < 0
    || !Number.isSafeInteger(row.byteLength) || row.byteLength !== row.sampleCount * 8
    || !Number.isSafeInteger(row.startSampleIndex) || row.startSampleIndex < 0) {
    throw new Error('Stage-1 manifest contains an invalid row');
  }
}

function inspectRow(bytes) {
  let sumPower = 0;
  let peakPower = 0;
  let active = 0;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let offset = 0; offset < bytes.byteLength; offset += 8) {
    const inPhase = view.getFloat32(offset, true);
    const quadrature = view.getFloat32(offset + 4, true);
    const power = inPhase * inPhase + quadrature * quadrature;
    if (!Number.isFinite(power)) {
      throw new Error(`Non-finite sample at byte ${offset}`);
    }
    sumPower += power;
    if (power > peakPower) peakPower = power;
    if (power > 1e-12) active += 1;
  }
  const samples = bytes.byteLength / 8;
  return {
    rms: Math.sqrt(sumPower / samples),
    peak: Math.sqrt(peakPower),
    activeFraction: active / samples,
    zero: active === 0,
  };
}

function summarizeProfile(rows) {
  const rawHashes = new Set(rows.map((row) => row.rawSha256));
  const starts = new Set(rows.map((row) => row.startSampleIndex));
  const contentSeeds = new Set(rows
    .map((row) => row.contentSeed)
    .filter((seed) => seed !== null));
  const contentProbes = new Set(rows
    .map((row) => row.contentProbeSha256)
    .filter((hash) => hash !== null));
  const activeFractions = rows.map((row) => row.activeFraction);
  return {
    rows: rows.length,
    distinctRawHashes: rawHashes.size,
    distinctStartSampleIndices: starts.size,
    zeroRows: rows.filter((row) => row.zero).length,
    minActiveFraction: Math.min(...activeFractions),
    maxActiveFraction: Math.max(...activeFractions),
    contentSeedCount: contentSeeds.size,
    distinctContentProbes: contentProbes.size,
    rows,
  };
}
