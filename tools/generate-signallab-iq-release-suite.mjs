/**
 * Generate a fresh, matched capture-length release suite without touching the
 * live training corpus.
 *
 * This is intentionally a one-shot protocol:
 *   - RELEASE_ROOT must name a new or empty directory;
 *   - CANDIDATE_SHA256 must identify the already-frozen candidate;
 *   - every length uses the same independent corpus seed, so row i has matched
 *     profile/nuisance draws while observation length changes;
 *   - the underlying generator refuses overwrite unless explicitly overridden,
 *     and this launcher never grants that override.
 *
 * Example (run only after freezing a candidate):
 *   RELEASE_ROOT=training/artifacts/releases/invariant-v1-sealed-20260728 \
 *   RELEASE_SEED=20260728 \
 *   CANDIDATE_PATH=training/.../runtime_bundle.pt \
 *   CANDIDATE_SHA256=<sha256> \
 *   node tools/generate-signallab-iq-release-suite.mjs
 *
 * The embedded evaluation_protocol is selected by RELEASE_EVALUATION_PROTOCOL:
 * 'v2' (default) embeds the frozen v2 object below, exactly as the consumed
 * seed-20260729 run did; 'v3' embeds the staged time-domain protocol printed
 * by evaluate_v3_release_suite.py --print-expected-protocol, loaded verbatim
 * from its pinned fixture (see V3_EXPECTED_PROTOCOL_FIXTURE) and required to
 * agree with RELEASE_SEED.
 */

import {
  createReadStream,
  existsSync,
  lstatSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  realpathSync,
  statSync,
  writeFileSync,
} from 'node:fs';
import { createHash } from 'node:crypto';
import {
  basename,
  dirname,
  join,
  resolve,
  sep,
} from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, '..');
const LIVE_CORPUS = resolve(REPO, 'training/artifacts/signallab-corpus');
const GENERATOR = resolve(HERE, 'generate-signallab-iq-corpus.ts');
const PREFIX_DERIVER = resolve(HERE, 'derive-signallab-iq-prefix-corpus.mjs');
const LAUNCHER = fileURLToPath(import.meta.url);
const TSX_PACKAGE = 'tsx@4.20.3';
const PROTOCOL = 'signallab-matched-length-release-v1';
const REQUIRED_NODE_VERSION = 'v22.23.1';
const REQUIRED_NPM_VERSION = '10.9.8';
const DEPENDENCY_CONTRACT = Object.freeze({
  signallab: {
    git_commit: '7c8303a0338b0f7c088737f8df2d935fd04bb033',
    git_tree: 'f0fade7d2dfd8f6ebd1e62caeae7d44d916d6796',
    package_lock_sha256:
      '5dd71450021febce4acbb9a835a3cadea659e5b4ee088217b9b697bf3f0a5fa6',
    source_tree_sha256:
      '93bec60b8da571bd0cf4a51630bd2b053d95eaf1bf2da503251775545fb873c0',
    source_file_count: 81,
  },
  atom_dsp: {
    git_commit: '4bb4d7075ba0e30512480ffbf0cb6e5757ffa218',
    git_tree: '3d1a8b7a97658a98b8ef36adc7f74f97cf16e9d2',
    package_lock_sha256:
      'b1c8198c18a4b25ed605db0781ae57b6050e0123ab0696ca2bdcba112696c0ca',
    source_tree_sha256:
      '80a8363407acfa60ea366a1b6f83401a07797ef7c9776579bc949eab16e04588',
    source_file_count: 20,
    dist_index_js_sha256:
      'caef3a8b3cbf4c300f8ceab5619a1db6b4fcebb942d428577f50790a1fb360e7',
    dist_index_dts_sha256:
      'fc6933a4c7e7d9aeaa2ca22043f6e22b2333671c08b81b1cd35b29a7de08c980',
  },
});
/**
 * The frozen v2 evaluation protocol, kept verbatim for provenance: it is the
 * exact object the consumed seed-20260729 v2 sealed run embedded, and it
 * remains the default so a v2-protocol rerun of this launcher is
 * byte-identical to the historical behaviour.
 */
const EVALUATION_PROTOCOL = Object.freeze({
  version: 'invariant-release-evaluation-v1',
  matched_capture_length: 16384,
  required_capture_lengths: [4096, 8192, 16384, 32768],
  minimum_target_per_class: 80,
  length_observation_rule:
    'generate maximum capture length once; every shorter observed and clean row '
    + 'is a bit-exact cf32le prefix of the corresponding maximum-length row',
  high_snr_db: 18,
  five_shot: {
    k: 5,
    split_version: 'sha256-lowest-per-class-v1',
    hash_payload:
      'utf8(split_salt + NUL + release_seed_decimal + NUL + canonical_row_identity_json)',
    split_salt: 'atomos-invariant-release-five-shot-v1',
    support_rule:
      'lowest five lowercase SHA-256 digests per class; ties by corpus row index',
    query_rule:
      'all remaining rows; identical row indices at every matched capture length',
    prototype_source:
      'support rows embedded at matched_capture_length; one fixed five-shot '
      + 'prototype set reused at every capture length',
  },
  novelty: {
    seed: 20260729,
    families: ['noise', 'chirp'],
    n_each_per_length: 300,
    generation:
      'generate each fresh openset_eval.NOVELTY realization once at maximum capture length; '
      + 'shorter observations are exact prefixes; one family seed derived by SHA-256 '
      + 'from base seed and family',
    seed_derivation:
      'uint64 little-endian first 8 bytes of SHA-256(utf8(base seed decimal '
      + '+ NUL + family))',
    calibration:
      'none; frozen weighted-LOF references, enrollment ranks, weights, and threshold only',
  },
  physical_scale_factors: [0.5, 0.75, 1.0, 1.5, 2.0],
  physical_scale_support: {
    edge_guard_fraction: 1 / 512,
    minimum_rows_per_class: 20,
    rule:
      'for every requested factor, effective resampling scale times '
      + '(abs(centreOffsetFrac) + bandwidthHz/(2*sampleRateHz)) must be '
      + 'strictly below 0.5 - edge_guard_fraction',
    selection_timing:
      'metadata-only common subset fixed before preprocessing or inference',
  },
  gates: {
    closed_fine: 0.72,
    closed_family: 0.82,
    closed_high_snr: 0.78,
    closed_clean: 0.85,
    five_shot: 0.85,
    open_auroc_overall: 0.72,
    open_auroc_noise: 0.80,
    open_auroc_chirp: 0.80,
    open_known_false_unknown_max: 0.10,
    open_unknown_recall_noise: 0.10,
    open_unknown_recall_chirp: 0.10,
    invariant_worst_balanced_accuracy: 0.75,
    invariant_max_mean_pairwise_cosine: 0.78,
    length_pair_prediction_agreement: 0.80,
    length_pair_embedding_cosine: 0.85,
    scale_pair_prediction_agreement: 0.75,
    scale_pair_embedding_cosine: 0.80,
  },
});

/**
 * The v3 (staged time-domain) sealed protocol is NOT hard-coded here: the
 * predeclared object is owned by the sealed evaluator
 * (`training/zplane_ab/v2_full_variation/v3_scale/evaluate_v3_release_suite.py`,
 * `--print-expected-protocol <seed>`), captured verbatim into this pinned
 * fixture, and embedded from it. The evaluator refuses any suite whose
 * intent protocol differs from its own expected object, so the two copies
 * cannot silently drift: a stale fixture fails the evaluation loudly before
 * any gate is scored. `RELEASE_EVALUATION_PROTOCOL=v3` selects it;
 * the default (`v2`) preserves the historical behaviour above.
 */
const V3_EXPECTED_PROTOCOL_FIXTURE = resolve(
  HERE,
  'time-domain-v3-expected-evaluation-protocol-seed20260735.json',
);
const V3_EVALUATION_VERSION =
  'time-domain-v3-release-evaluation-v3-dual-fusion';
/**
 * Global seed hygiene mirrored from the sealed evaluators. This guard applies
 * before protocol selection so neither the historical v2 path nor the staged
 * v3 path can redraw a consumed or development-only seed.
 */
const REFUSED_RELEASE_SEEDS = Object.freeze({
  20260729: 'consumed sealed v2 release suite (evidence rule 2)',
  20260731:
    'consumed sealed v3.0 release suite (HANDOFF 25: 22/23 gates, known '
    + 'false-unknown failure frozen; evidence rules 2 and 4)',
  20260733:
    'consumed sealed v3.2 release suite (HANDOFF 27: 21/23 gates, known '
    + 'false-unknown and five-shot failures frozen; evidence rules 2 and 4)',
  20260734:
    'consumed sealed v3.2 release suite under its historical gate '
    + 'redeclaration (22/23 gates, five-shot failure frozen; evidence rules '
    + '2 and 4)',
  20260735:
    'consumed sealed v3.3 decoupled 8k-classifier/4k-rejector release suite '
    + '(22/23 gates; open_known_false_unknown_worst_length '
    + '0.11382734912146678 > 0.10 frozen; evidence rules 2 and 4)',
  20260730:
    'development model/fusion seed; reusing it as a release seed would '
    + 'collide the model and release namespaces (HANDOFF 25 note)',
  20260732:
    'development model seed; reusing it as a release seed would collide '
    + 'the model and release namespaces (HANDOFF 25 note; next untouched '
    + 'release seed after the consumed runs: 20260736)',
});
const REFUSED_RELEASE_SEED_BANDS = Object.freeze([
  [20260900, 20260999, 'development novelty seed namespace'],
  [20261000, 20261999, 'noise-prefilter fit-only seed band'],
]);

function validateReleaseSeed(releaseSeed) {
  const reason = REFUSED_RELEASE_SEEDS[releaseSeed];
  if (reason !== undefined) {
    throw new Error(`release seed ${releaseSeed} is refused: ${reason}`);
  }
  for (const [low, high, bandReason] of REFUSED_RELEASE_SEED_BANDS) {
    if (releaseSeed >= low && releaseSeed <= high) {
      throw new Error(
        `release seed ${releaseSeed} lies in the ${bandReason} (${low}-${high}) `
        + 'and is not an untouched release seed',
      );
    }
  }
}

function loadV3EvaluationProtocol(releaseSeed) {
  const wrapper = JSON.parse(readFileSync(V3_EXPECTED_PROTOCOL_FIXTURE, 'utf8'));
  const protocol = wrapper?.evaluation_protocol;
  if (protocol?.version !== V3_EVALUATION_VERSION) {
    throw new Error(
      `v3 protocol fixture does not carry version ${V3_EVALUATION_VERSION}: `
      + V3_EXPECTED_PROTOCOL_FIXTURE,
    );
  }
  if (wrapper.release_seed !== protocol?.novelty?.seed) {
    throw new Error(
      'v3 protocol fixture is internally inconsistent: its release_seed and '
      + 'its evaluation_protocol.novelty.seed differ',
    );
  }
  if (protocol.novelty.seed !== releaseSeed) {
    throw new Error(
      `the v3 protocol rule is 'novelty base seed IS the release seed'; the `
      + `pinned fixture predeclares seed ${protocol.novelty.seed} but `
      + `RELEASE_SEED=${releaseSeed}. Regenerate the fixture with `
      + `evaluate_v3_release_suite.py --print-expected-protocol ${releaseSeed} `
      + 'only if a different untouched release seed is genuinely intended',
    );
  }
  return protocol;
}

function selectEvaluationProtocol(releaseSeed) {
  validateReleaseSeed(releaseSeed);
  const choice = (process.env.RELEASE_EVALUATION_PROTOCOL ?? 'v2').trim();
  if (choice === 'v2') return EVALUATION_PROTOCOL;
  if (choice === 'v3') return loadV3EvaluationProtocol(releaseSeed);
  throw new Error(
    "RELEASE_EVALUATION_PROTOCOL must be 'v2' (default, historical) or 'v3' "
    + '(staged time-domain sealed protocol)',
  );
}

function integer(name, raw, minimum) {
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value < minimum) {
    throw new RangeError(`${name} must be a safe integer >= ${minimum}`);
  }
  return value;
}

function required(name) {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`${name} is required`);
  return value;
}

function parseLengths(raw) {
  const values = raw.split(',').map((part) => integer('RELEASE_LENGTHS', part.trim(), 64));
  if (values.length === 0 || new Set(values).size !== values.length) {
    throw new Error('RELEASE_LENGTHS must contain unique comma-separated lengths');
  }
  return values.sort((left, right) => left - right);
}

async function sha256(path) {
  const hash = createHash('sha256');
  await new Promise((accept, reject) => {
    const stream = createReadStream(path);
    stream.on('data', (chunk) => hash.update(chunk));
    stream.on('end', accept);
    stream.on('error', reject);
  });
  return hash.digest('hex');
}

function commandVersion(command) {
  const result = spawnSync(command, ['--version'], {
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(
      `${command} --version failed: ${(result.stderr ?? '').trim()}`,
    );
  }
  return result.stdout.trim();
}

function sourceTreeDigest(root) {
  const hash = createHash('sha256');
  let fileCount = 0;

  function walk(directory, prefix = '') {
    const entries = readdirSync(directory, { withFileTypes: true }).sort(
      (left, right) => (
        left.name < right.name ? -1 : left.name > right.name ? 1 : 0
      ),
    );
    for (const entry of entries) {
      if (entry.name === 'node_modules' || entry.name === 'dist') continue;
      if (entry.name === 'RELEASE_SOURCE_PROVENANCE.json') continue;
      const relative = prefix ? `${prefix}/${entry.name}` : entry.name;
      const path = join(directory, entry.name);
      if (entry.isSymbolicLink()) {
        throw new Error(`release dependency source may not contain symlink ${relative}`);
      }
      if (entry.isDirectory()) {
        walk(path, relative);
      } else if (entry.isFile()) {
        hash.update(Buffer.from(relative, 'utf8'));
        hash.update(Buffer.from([0]));
        hash.update(readFileSync(path));
        fileCount += 1;
      } else {
        throw new Error(
          `release dependency source contains unsupported entry ${relative}`,
        );
      }
    }
  }

  walk(root);
  return { sha256: hash.digest('hex'), file_count: fileCount };
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

function containsPath(parent, child) {
  return child === parent || child.startsWith(`${parent}${sep}`);
}

async function verifyDependencyRoot(signalLabRoot) {
  if (!statSync(signalLabRoot).isDirectory()) {
    throw new Error(`SIGNALLAB_ROOT must be a directory: ${signalLabRoot}`);
  }
  const dspPackageLink = join(
    signalLabRoot,
    'node_modules',
    '@atomos',
    'dsp',
  );
  if (!existsSync(dspPackageLink) || !lstatSync(dspPackageLink).isSymbolicLink()) {
    throw new Error(
      'SIGNALLAB_ROOT must resolve @atomos/dsp through its file-dependency symlink',
    );
  }
  const atomDspRoot = realpathSync(dspPackageLink);
  const expectedAtomDspRoot = canonicalProspectivePath(
    join(signalLabRoot, '..', 'Atom-DSP'),
  );
  if (atomDspRoot !== expectedAtomDspRoot) {
    throw new Error(
      `@atomos/dsp resolves to ${atomDspRoot}, expected ${expectedAtomDspRoot}`,
    );
  }
  if (!statSync(atomDspRoot).isDirectory()) {
    throw new Error(`resolved Atom-DSP root is not a directory: ${atomDspRoot}`);
  }

  const signalLabTree = sourceTreeDigest(signalLabRoot);
  const atomDspTree = sourceTreeDigest(atomDspRoot);
  if (
    signalLabTree.sha256 !== DEPENDENCY_CONTRACT.signallab.source_tree_sha256
    || signalLabTree.file_count !== DEPENDENCY_CONTRACT.signallab.source_file_count
  ) {
    throw new Error(
      `SignalLab source-tree digest mismatch: ${JSON.stringify(signalLabTree)}`,
    );
  }
  if (
    atomDspTree.sha256 !== DEPENDENCY_CONTRACT.atom_dsp.source_tree_sha256
    || atomDspTree.file_count !== DEPENDENCY_CONTRACT.atom_dsp.source_file_count
  ) {
    throw new Error(
      `Atom-DSP source-tree digest mismatch: ${JSON.stringify(atomDspTree)}`,
    );
  }

  const signalLabLock = join(signalLabRoot, 'package-lock.json');
  const atomDspLock = join(atomDspRoot, 'package-lock.json');
  const atomDspDistJs = join(atomDspRoot, 'dist', 'index.js');
  const atomDspDistDts = join(atomDspRoot, 'dist', 'index.d.ts');
  for (const path of [
    signalLabLock,
    atomDspLock,
    atomDspDistJs,
    atomDspDistDts,
  ]) {
    if (!existsSync(path) || !statSync(path).isFile()) {
      throw new Error(`release dependency file is missing: ${path}`);
    }
  }
  const observed = {
    signalLabLock: await sha256(signalLabLock),
    atomDspLock: await sha256(atomDspLock),
    atomDspDistJs: await sha256(atomDspDistJs),
    atomDspDistDts: await sha256(atomDspDistDts),
  };
  if (
    observed.signalLabLock
      !== DEPENDENCY_CONTRACT.signallab.package_lock_sha256
    || observed.atomDspLock
      !== DEPENDENCY_CONTRACT.atom_dsp.package_lock_sha256
    || observed.atomDspDistJs
      !== DEPENDENCY_CONTRACT.atom_dsp.dist_index_js_sha256
    || observed.atomDspDistDts
      !== DEPENDENCY_CONTRACT.atom_dsp.dist_index_dts_sha256
  ) {
    throw new Error(
      `release dependency lock/build digest mismatch: ${JSON.stringify(observed)}`,
    );
  }
  return {
    signallab: {
      root: signalLabRoot,
      ...DEPENDENCY_CONTRACT.signallab,
      observed_source_tree: signalLabTree,
      package_lock_path: signalLabLock,
    },
    atom_dsp: {
      root: atomDspRoot,
      package_resolution: dspPackageLink,
      ...DEPENDENCY_CONTRACT.atom_dsp,
      observed_source_tree: atomDspTree,
      package_lock_path: atomDspLock,
      dist_index_js: {
        path: atomDspDistJs,
        bytes: statSync(atomDspDistJs).size,
        sha256: observed.atomDspDistJs,
      },
      dist_index_dts: {
        path: atomDspDistDts,
        bytes: statSync(atomDspDistDts).size,
        sha256: observed.atomDspDistDts,
      },
    },
  };
}

function runChecked(command, args, env, description) {
  const child = spawnSync(command, args, {
    cwd: REPO,
    env,
    encoding: 'utf8',
    stdio: ['ignore', 'inherit', 'inherit'],
  });
  if (child.error) throw child.error;
  if (child.status !== 0) {
    throw new Error(`${description} failed with status ${child.status}`);
  }
}

const releaseRoot = canonicalProspectivePath(required('RELEASE_ROOT'));
const canonicalLiveCorpus = canonicalProspectivePath(LIVE_CORPUS);
const releaseSeed = integer('RELEASE_SEED', required('RELEASE_SEED'), 0);
const evaluationProtocol = selectEvaluationProtocol(releaseSeed);
const candidatePath = canonicalProspectivePath(required('CANDIDATE_PATH'));
const candidateSha256 = required('CANDIDATE_SHA256');
if (!/^[0-9a-f]{64}$/i.test(candidateSha256)) {
  throw new Error('CANDIDATE_SHA256 must be exactly 64 hexadecimal characters');
}
if (!existsSync(candidatePath) || !statSync(candidatePath).isFile()) {
  throw new Error(`CANDIDATE_PATH must be an existing file: ${candidatePath}`);
}
const actualCandidateSha256 = await sha256(candidatePath);
if (actualCandidateSha256 !== candidateSha256.toLowerCase()) {
  throw new Error(
    `CANDIDATE_SHA256 mismatch: expected ${candidateSha256.toLowerCase()}, `
    + `got ${actualCandidateSha256}`,
  );
}
const lengths = parseLengths(process.env.RELEASE_LENGTHS ?? '4096,8192,16384,32768');
if (
  JSON.stringify(lengths)
  !== JSON.stringify(evaluationProtocol.required_capture_lengths)
) {
  throw new Error(
    'RELEASE_LENGTHS must equal the frozen required release suite '
    + JSON.stringify(evaluationProtocol.required_capture_lengths),
  );
}
const targetPerClass = integer(
  'RELEASE_TARGET_PER_CLASS',
  process.env.RELEASE_TARGET_PER_CLASS ?? '80',
  evaluationProtocol.minimum_target_per_class,
);

if (
  containsPath(canonicalLiveCorpus, releaseRoot)
  || containsPath(releaseRoot, canonicalLiveCorpus)
) {
  throw new Error(
    'RELEASE_ROOT may not be the live training corpus, one of its descendants, '
    + 'or one of its ancestors',
  );
}
if (releaseRoot === REPO || releaseRoot === resolve(REPO, 'training')) {
  throw new Error('RELEASE_ROOT is too broad; choose a dedicated new release directory');
}
if (containsPath(releaseRoot, candidatePath)) {
  throw new Error('candidate must be frozen outside RELEASE_ROOT');
}
if (existsSync(releaseRoot) && readdirSync(releaseRoot).length > 0) {
  throw new Error(`RELEASE_ROOT must be empty: ${releaseRoot}`);
}
const signalLabRoot = canonicalProspectivePath(required('SIGNALLAB_ROOT'));
const dependencyProvenance = await verifyDependencyRoot(signalLabRoot);
if (process.version !== REQUIRED_NODE_VERSION) {
  throw new Error(
    `release generation requires Node ${REQUIRED_NODE_VERSION}, got ${process.version}`,
  );
}
const npmVersion = commandVersion('npm');
const npxVersion = commandVersion('npx');
if (npmVersion !== REQUIRED_NPM_VERSION || npxVersion !== REQUIRED_NPM_VERSION) {
  throw new Error(
    `release generation requires npm/npx ${REQUIRED_NPM_VERSION}, `
    + `got npm ${npmVersion}, npx ${npxVersion}`,
  );
}
for (const [name, dependencyRoot] of [
  ['SignalLab', dependencyProvenance.signallab.root],
  ['Atom-DSP', dependencyProvenance.atom_dsp.root],
]) {
  if (
    containsPath(dependencyRoot, releaseRoot)
    || containsPath(releaseRoot, dependencyRoot)
  ) {
    throw new Error(`RELEASE_ROOT may not overlap the ${name} dependency root`);
  }
}
mkdirSync(releaseRoot, { recursive: true });

const startedAt = new Date().toISOString();
const sourceSha256 = {
  launcher: {
    path: LAUNCHER,
    sha256: await sha256(LAUNCHER),
  },
  corpus_generator: {
    path: GENERATOR,
    sha256: await sha256(GENERATOR),
  },
  prefix_deriver: {
    path: PREFIX_DERIVER,
    sha256: await sha256(PREFIX_DERIVER),
  },
  tsx_package: TSX_PACKAGE,
};
const runtimeProvenance = {
  node: process.version,
  npm: npmVersion,
  npx: npxVersion,
};
const intent = {
  protocol: PROTOCOL,
  status: 'in_progress',
  development_data_used: false,
  candidate_path: candidatePath,
  candidate_sha256: candidateSha256.toLowerCase(),
  release_seed: releaseSeed,
  capture_lengths: lengths,
  target_per_class: targetPerClass,
  exact_target_per_class: true,
  evaluation_protocol: evaluationProtocol,
  source_sha256: sourceSha256,
  dependency_provenance: dependencyProvenance,
  runtime_provenance: runtimeProvenance,
  started_at: startedAt,
};
const intentPath = join(releaseRoot, 'RELEASE_INTENT.json');
writeFileSync(
  intentPath,
  `${JSON.stringify(intent, null, 2)}\n`,
  { flag: 'wx' },
);
const releaseIntentSha256 = await sha256(intentPath);

const shortestLength = lengths[0];
const longestLength = lengths.at(-1);
const startProbeDirectory = join(
  releaseRoot,
  `_start_probe_n${shortestLength}`,
);
const commonGeneratorEnv = {
  ...process.env,
  SIGNALLAB_ROOT: signalLabRoot,
  TARGET_PER_CLASS: String(targetPerClass),
  EXACT_TARGET_PER_CLASS: '1',
  CORPUS_SEED: String(releaseSeed),
  START_STRIDE_SAMPLES: '16384',
  ALLOW_OVERWRITE: '0',
};
runChecked(
  'npx',
  ['--yes', TSX_PACKAGE, GENERATOR],
  {
    ...commonGeneratorEnv,
    SAMPLE_COUNT: String(shortestLength),
    OUTPUT_DIR: startProbeDirectory,
  },
  `occupied-start probe generation at n=${shortestLength}`,
);
const startProbeManifest = join(startProbeDirectory, 'corpus.json');

const longestDirectory = join(releaseRoot, `n${longestLength}`);
runChecked(
  'npx',
  ['--yes', TSX_PACKAGE, GENERATOR],
  {
    ...commonGeneratorEnv,
    SAMPLE_COUNT: String(longestLength),
    OUTPUT_DIR: longestDirectory,
    MATCH_STARTS_FROM: startProbeManifest,
  },
  `longest release generation at n=${longestLength}`,
);

for (const length of lengths) {
  if (length === longestLength) continue;
  runChecked(
    process.execPath,
    [PREFIX_DERIVER],
    {
      ...process.env,
      SOURCE_DIR: longestDirectory,
      OUTPUT_DIR: join(releaseRoot, `n${length}`),
      SAMPLE_COUNT: String(length),
    },
    `exact prefix derivation at n=${length}`,
  );
}

async function fileRecords(directory) {
  const files = {};
  for (const name of ['corpus.json', 'corpus.f32', 'corpus_clean.f32']) {
    const path = join(directory, name);
    files[name] = {
      bytes: statSync(path).size,
      sha256: await sha256(path),
    };
  }
  return files;
}

const corpora = [];
for (const length of lengths) {
  const directory = join(releaseRoot, `n${length}`);
  corpora.push({
    capture_length: length,
    directory,
    derivation: length === longestLength
      ? 'generated_once_at_longest_length'
      : 'bit_exact_row_prefix_of_longest',
    files: await fileRecords(directory),
  });
}
const startProbe = {
  scored: false,
  purpose:
    'select one occupied start per row before longest-only nuisance realization',
  capture_length: shortestLength,
  directory: startProbeDirectory,
  files: await fileRecords(startProbeDirectory),
};

const completed = {
  ...intent,
  status: 'complete',
  completed_at: new Date().toISOString(),
  generator: GENERATOR,
  prefix_deriver: PREFIX_DERIVER,
  release_intent_sha256: releaseIntentSha256,
  source_sha256: sourceSha256,
  dependency_provenance: dependencyProvenance,
  runtime_provenance: runtimeProvenance,
  start_probe: startProbe,
  corpora,
};
writeFileSync(
  join(releaseRoot, 'RELEASE_MANIFEST.json'),
  `${JSON.stringify(completed, null, 2)}\n`,
  { flag: 'wx' },
);
console.log(`sealed release suite written to ${releaseRoot}`);
