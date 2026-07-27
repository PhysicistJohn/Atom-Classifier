import { spawnSync } from 'node:child_process';
import {
  createReadStream,
  mkdtempSync,
  readFileSync,
  rmSync,
  statSync,
} from 'node:fs';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  CONTRACT_PATH,
  REPOSITORY_ROOT,
  verifyStrictTimeDomainV3Contract,
} from './time-domain-v3-contract-verifier.mjs';


const HERE = dirname(fileURLToPath(import.meta.url));
const ESBUILD_VERSION = '0.25.8';
const EXPECTED_NODE_VERSION = '22.23.1';
const HOST = '127.0.0.1';

function portArgument() {
  const unexpected = process.argv.slice(2).filter(
    (value) => !value.startsWith('--port='),
  );
  if (unexpected.length > 0) {
    throw new Error(
      `unknown arguments ${unexpected.join(', ')}; external data paths are refused`,
    );
  }
  const raw = process.argv.find((value) => value.startsWith('--port='));
  const port = raw === undefined ? 8765 : Number(raw.slice('--port='.length));
  if (!Number.isInteger(port) || port < 1024 || port > 65535) {
    throw new RangeError('port must be an integer in [1024, 65535]');
  }
  return port;
}

function buildEntry(entry, output, buildDirectory) {
  const npmExecutable = process.env.npm_execpath;
  const command = npmExecutable === undefined
    ? 'npm'
    : process.execPath;
  const prefix = npmExecutable === undefined ? [] : [npmExecutable];
  const result = spawnSync(
    command,
    [
      ...prefix,
      'exec',
      '--yes',
      `--package=esbuild@${ESBUILD_VERSION}`,
      '--',
      'esbuild',
      entry,
      '--bundle',
      '--format=esm',
      '--platform=browser',
      '--target=chrome120',
      '--legal-comments=none',
      '--log-level=warning',
      `--outfile=${output}`,
    ],
    {
      cwd: buildDirectory,
      env: {
        ...process.env,
        npm_config_package_lock: 'false',
        npm_config_save: 'false',
      },
      encoding: 'utf8',
    },
  );
  if (result.status !== 0) {
    throw new Error(
      `isolated esbuild failed for ${basename(entry)}:\n`
      + `${result.stdout}${result.stderr}`,
    );
  }
}

function headers(contentType) {
  return {
    'Cache-Control': 'no-store',
    'Content-Type': contentType,
    'Cross-Origin-Embedder-Policy': 'require-corp',
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Cross-Origin-Resource-Policy': 'same-origin',
    'X-Content-Type-Options': 'nosniff',
  };
}

function serveFile(response, path, contentType) {
  const size = statSync(path).size;
  response.writeHead(200, {
    ...headers(contentType),
    'Content-Length': String(size),
  });
  createReadStream(path).pipe(response);
}

function main() {
  if (process.versions.node !== EXPECTED_NODE_VERSION) {
    throw new Error(
      `browser audit server requires Node ${EXPECTED_NODE_VERSION}, `
      + `got ${process.versions.node}`,
    );
  }
  const port = portArgument();
  const verified = verifyStrictTimeDomainV3Contract();
  const buildDirectory = mkdtempSync(
    join(tmpdir(), 'atomos-time-domain-v3-browser-'),
  );
  const workerBundle = join(buildDirectory, 'worker.js');
  const harnessBundle = join(buildDirectory, 'harness.js');
  buildEntry(
    resolve(HERE, 'time-domain-v3-browser-worker.ts'),
    workerBundle,
    buildDirectory,
  );
  buildEntry(
    resolve(HERE, 'time-domain-v3-browser-harness.ts'),
    harnessBundle,
    buildDirectory,
  );

  const fixedFiles = {
    '/': {
      path: resolve(HERE, 'time-domain-v3-browser-harness.html'),
      contentType: 'text/html; charset=utf-8',
    },
    '/harness.js': {
      path: harnessBundle,
      contentType: 'text/javascript; charset=utf-8',
    },
    '/worker.js': {
      path: workerBundle,
      contentType: 'text/javascript; charset=utf-8',
    },
    '/fixture.json': {
      path: resolve(REPOSITORY_ROOT, verified.contract.fixture_path),
      contentType: 'application/json; charset=utf-8',
    },
    '/contract.json': {
      path: CONTRACT_PATH,
      contentType: 'application/json; charset=utf-8',
    },
  };
  const server = createServer((request, response) => {
    const url = new URL(request.url ?? '/', `http://${HOST}:${port}`);
    if (request.method !== 'GET') {
      response.writeHead(405, headers('text/plain; charset=utf-8'));
      response.end('method not allowed\n');
      return;
    }
    if (url.pathname === '/health') {
      response.writeHead(200, headers('application/json; charset=utf-8'));
      response.end(JSON.stringify({
        status: 'ok',
        contract_sha256: verified.contract_sha256,
        synthetic_only: true,
      }));
      return;
    }
    if (url.pathname === '/favicon.ico') {
      response.writeHead(204, headers('image/x-icon'));
      response.end();
      return;
    }
    const selected = fixedFiles[url.pathname];
    if (selected === undefined) {
      response.writeHead(404, headers('text/plain; charset=utf-8'));
      response.end('not found\n');
      return;
    }
    serveFile(response, selected.path, selected.contentType);
  });

  let cleaned = false;
  const cleanup = () => {
    if (cleaned) return;
    cleaned = true;
    if (
      basename(buildDirectory).startsWith('atomos-time-domain-v3-browser-')
      && dirname(buildDirectory) === tmpdir()
    ) {
      rmSync(buildDirectory, { recursive: true, force: true });
    }
  };
  server.on('close', cleanup);
  for (const signal of ['SIGINT', 'SIGTERM']) {
    process.once(signal, () => {
      server.close(() => process.exit(0));
    });
  }
  server.listen(port, HOST, () => {
    process.stdout.write(`${JSON.stringify({
      ready: true,
      url: `http://${HOST}:${port}/?warmup=5&iterations=40`,
      contract_sha256: verified.contract_sha256,
      verifier_sha256: verified.verifier_sha256,
      node: process.versions.node,
      esbuild: ESBUILD_VERSION,
      build_directory: buildDirectory,
      fixture_bytes: readFileSync(
        resolve(REPOSITORY_ROOT, verified.contract.fixture_path),
      ).byteLength,
    })}\n`);
  });
}

try {
  main();
} catch (error) {
  process.stderr.write(
    `${error instanceof Error ? error.stack ?? error.message : String(error)}\n`,
  );
  process.exitCode = 1;
}
