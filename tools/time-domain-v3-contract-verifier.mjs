import { createHash } from 'node:crypto';
import {
  lstatSync,
  readFileSync,
  realpathSync,
} from 'node:fs';
import { dirname, isAbsolute, join, relative, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';


const HERE = dirname(fileURLToPath(import.meta.url));
export const REPOSITORY_ROOT = resolve(HERE, '..');
export const CONTRACT_PATH = join(
  HERE,
  'time-domain-v3-runtime-contract.json',
);

function sha256(path) {
  return createHash('sha256').update(readFileSync(path)).digest('hex');
}

function assertExactArray(actual, expected, name) {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new Error(`${name} does not match the strict runtime contract`);
  }
}

function resolveRegularRepositoryFile(root, relativePath) {
  if (
    typeof relativePath !== 'string'
    || relativePath.length === 0
    || isAbsolute(relativePath)
  ) {
    throw new Error('contract source paths must be non-empty repository-relative paths');
  }
  const lexical = resolve(root, relativePath);
  const lexicalRelative = relative(root, lexical);
  if (
    lexicalRelative === '..'
    || lexicalRelative.startsWith(`..${process.platform === 'win32' ? '\\' : '/'}`)
  ) {
    throw new Error(`contract path escapes repository: ${relativePath}`);
  }
  const status = lstatSync(lexical);
  if (!status.isFile() || status.isSymbolicLink()) {
    throw new Error(`contract path must be a regular non-symlink file: ${relativePath}`);
  }
  const canonical = realpathSync(lexical);
  const canonicalRelative = relative(realpathSync(root), canonical);
  if (
    canonicalRelative === '..'
    || canonicalRelative.startsWith(`..${process.platform === 'win32' ? '\\' : '/'}`)
  ) {
    throw new Error(`canonical contract path escapes repository: ${relativePath}`);
  }
  return canonical;
}

export function verifyStrictTimeDomainV3Contract() {
  const contract = JSON.parse(readFileSync(CONTRACT_PATH, 'utf8'));
  if (
    contract.contract_version !== 'strict-time-domain-v3-runtime-audit-v1'
    || contract.frontend_version !== 'invariant-patch-time-domain-v1'
    || contract.geometry_version !== 'time-correlation-v1'
    || contract.synthetic_only !== true
    || contract.reads_held_or_sealed_payload !== false
    || contract.uses_frequency_transform !== false
  ) {
    throw new Error('strict time-domain v3 contract flags/version are invalid');
  }
  assertExactArray(
    contract.audit_lengths,
    [4096, 8192, 16384, 32768],
    'audit lengths',
  );
  if (
    !Array.isArray(contract.prohibited_path_fragments)
    || contract.prohibited_path_fragments.length === 0
  ) {
    throw new Error('contract must declare prohibited data path fragments');
  }
  if (
    contract.source_sha256 === null
    || typeof contract.source_sha256 !== 'object'
    || Array.isArray(contract.source_sha256)
  ) {
    throw new Error('contract source_sha256 must be an object');
  }

  const observedHashes = {};
  for (const [sourcePath, expectedHash] of Object.entries(
    contract.source_sha256,
  )) {
    const normalized = sourcePath.replaceAll('\\', '/').toLowerCase();
    for (const fragment of contract.prohibited_path_fragments) {
      if (normalized.includes(String(fragment).toLowerCase())) {
        throw new Error(`strict v3 audit refuses prohibited path ${sourcePath}`);
      }
    }
    if (
      typeof expectedHash !== 'string'
      || !/^[0-9a-f]{64}$/u.test(expectedHash)
    ) {
      throw new Error(`invalid SHA-256 binding for ${sourcePath}`);
    }
    const absolute = resolveRegularRepositoryFile(REPOSITORY_ROOT, sourcePath);
    const observed = sha256(absolute);
    if (observed !== expectedHash) {
      throw new Error(
        `strict v3 source hash mismatch for ${sourcePath}: ${observed} != ${expectedHash}`,
      );
    }
    observedHashes[sourcePath] = observed;
  }

  if (
    typeof contract.fixture_path !== 'string'
    || contract.source_sha256[contract.fixture_path] === undefined
  ) {
    throw new Error('fixture must be a source-hash-bound contract file');
  }
  const fixturePath = resolveRegularRepositoryFile(
    REPOSITORY_ROOT,
    contract.fixture_path,
  );
  const fixture = JSON.parse(readFileSync(fixturePath, 'utf8'));
  if (
    fixture.fixture_version !== 'time-domain-v3-browser-parity-v1'
    || fixture.frontend_version !== contract.frontend_version
    || fixture.geometry_version !== contract.geometry_version
    || fixture.synthetic_only !== true
    || fixture.reads_held_or_sealed_payload !== false
    || fixture.uses_frequency_transform !== false
  ) {
    throw new Error('browser fixture violates strict synthetic-only contract');
  }
  assertExactArray(fixture.lengths, contract.audit_lengths, 'fixture lengths');

  const pythonBindings = {
    'time_domain_geometry.py': 'training/time_domain_geometry.py',
    'time_domain_invariant_patch_preprocess.py':
      'training/time_domain_invariant_patch_preprocess.py',
    'invariant_patch_preprocess.py':
      'training/invariant_patch_preprocess.py',
    'preprocess.py': 'training/preprocess.py',
    'generate_time_domain_v3_browser_fixture.py':
      'training/generate_time_domain_v3_browser_fixture.py',
  };
  for (const [fixtureName, contractName] of Object.entries(pythonBindings)) {
    if (
      fixture.python_source_sha256?.[fixtureName]
      !== contract.source_sha256[contractName]
    ) {
      throw new Error(`fixture Python binding mismatch for ${fixtureName}`);
    }
  }

  return {
    contract,
    fixture,
    observed_hashes: observedHashes,
    contract_sha256: sha256(CONTRACT_PATH),
    verifier_sha256: sha256(fileURLToPath(import.meta.url)),
  };
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? '').href) {
  const verified = verifyStrictTimeDomainV3Contract();
  process.stdout.write(`${JSON.stringify({
    passed: true,
    contract_version: verified.contract.contract_version,
    contract_sha256: verified.contract_sha256,
    verifier_sha256: verified.verifier_sha256,
    source_sha256: verified.observed_hashes,
    synthetic_only: verified.fixture.synthetic_only,
    reads_held_or_sealed_payload:
      verified.fixture.reads_held_or_sealed_payload,
    lengths: verified.fixture.lengths,
  }, null, 2)}\n`);
}
