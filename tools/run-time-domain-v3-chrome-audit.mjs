import { spawn, spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  existsSync,
  mkdtempSync,
  readFileSync,
  rmSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, dirname, join } from 'node:path';
import {
  verifyStrictTimeDomainV3Contract,
} from './time-domain-v3-contract-verifier.mjs';


const EXPECTED_NODE_VERSION = '22.23.1';
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const HOST = '127.0.0.1';

function boundedArgument(name, fallback, maximum) {
  const prefix = `--${name}=`;
  const raw = process.argv.find((value) => value.startsWith(prefix));
  if (raw === undefined) return fallback;
  const value = Number(raw.slice(prefix.length));
  if (!Number.isInteger(value) || value <= 0 || value > maximum) {
    throw new RangeError(`${name} must be an integer in [1, ${maximum}]`);
  }
  return value;
}

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

async function waitForFile(path, child, timeoutMilliseconds) {
  const deadline = Date.now() + timeoutMilliseconds;
  while (Date.now() < deadline) {
    if (existsSync(path)) return;
    if (child.exitCode !== null) {
      throw new Error(`Chrome exited before DevTools startup (${child.exitCode})`);
    }
    await delay(50);
  }
  throw new Error('timed out waiting for Chrome DevTools endpoint');
}

class CdpClient {
  #socket;
  #nextId = 1;
  #pending = new Map();

  constructor(url) {
    this.#socket = new WebSocket(url);
    this.#socket.onmessage = (event) => {
      const message = JSON.parse(String(event.data));
      if (message.id === undefined) return;
      const pending = this.#pending.get(message.id);
      if (pending === undefined) return;
      this.#pending.delete(message.id);
      clearTimeout(pending.timeout);
      if (message.error !== undefined) {
        pending.reject(
          new Error(
            `CDP ${pending.method} failed: ${JSON.stringify(message.error)}`,
          ),
        );
      } else {
        pending.resolve(message.result);
      }
    };
    this.#socket.onerror = () => {
      this.#rejectAll(new Error('Chrome DevTools WebSocket failed'));
    };
    this.#socket.onclose = () => {
      this.#rejectAll(new Error('Chrome DevTools WebSocket closed'));
    };
  }

  async open() {
    if (this.#socket.readyState === WebSocket.OPEN) return;
    await new Promise((resolve, reject) => {
      const timeout = setTimeout(
        () => reject(new Error('Chrome DevTools WebSocket open timed out')),
        15_000,
      );
      this.#socket.addEventListener('open', () => {
        clearTimeout(timeout);
        resolve();
      }, { once: true });
      this.#socket.addEventListener('error', () => {
        clearTimeout(timeout);
        reject(new Error('Chrome DevTools WebSocket open failed'));
      }, { once: true });
    });
  }

  send(method, params = {}, sessionId = undefined, timeoutMilliseconds = 30_000) {
    const id = this.#nextId++;
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.#pending.delete(id);
        reject(new Error(`CDP ${method} timed out`));
      }, timeoutMilliseconds);
      this.#pending.set(id, { resolve, reject, timeout, method });
      this.#socket.send(JSON.stringify({
        id,
        method,
        params,
        ...(sessionId === undefined ? {} : { sessionId }),
      }));
    });
  }

  close() {
    if (
      this.#socket.readyState === WebSocket.OPEN
      || this.#socket.readyState === WebSocket.CONNECTING
    ) {
      this.#socket.close();
    }
  }

  #rejectAll(error) {
    for (const pending of this.#pending.values()) {
      clearTimeout(pending.timeout);
      pending.reject(error);
    }
    this.#pending.clear();
  }
}

async function waitForAuditPromise(client, sessionId) {
  const deadline = Date.now() + 30_000;
  while (Date.now() < deadline) {
    const response = await client.send(
      'Runtime.evaluate',
      {
        expression:
          "typeof window.__TIME_DOMAIN_V3_AUDIT_PROMISE__ === 'object'",
        returnByValue: true,
      },
      sessionId,
    );
    if (response.result?.value === true) return;
    await delay(50);
  }
  throw new Error('browser harness did not expose its audit promise');
}

async function main() {
  if (process.versions.node !== EXPECTED_NODE_VERSION) {
    throw new Error(
      `Chrome audit requires Node ${EXPECTED_NODE_VERSION}, `
      + `got ${process.versions.node}`,
    );
  }
  const unexpected = process.argv.slice(2).filter(
    (value) =>
      !value.startsWith('--port=')
      && !value.startsWith('--warmup=')
      && !value.startsWith('--iterations='),
  );
  if (unexpected.length > 0) {
    throw new Error(
      `unknown arguments ${unexpected.join(', ')}; external URLs/data paths are refused`,
    );
  }
  if (!existsSync(CHROME)) {
    throw new Error(`Google Chrome is unavailable at ${CHROME}`);
  }
  const port = boundedArgument('port', 8765, 65535);
  if (port < 1024) throw new RangeError('port must be at least 1024');
  const warmup = boundedArgument('warmup', 5, 100);
  const iterations = boundedArgument('iterations', 40, 1000);
  const verified = verifyStrictTimeDomainV3Contract();
  const health = await fetch(`http://${HOST}:${port}/health`, {
    cache: 'no-store',
  });
  if (!health.ok) {
    throw new Error(`strict v3 browser server health failed (${health.status})`);
  }
  const healthPayload = await health.json();
  if (
    healthPayload.contract_sha256 !== verified.contract_sha256
    || healthPayload.synthetic_only !== true
  ) {
    throw new Error('browser server is not bound to the current strict v3 contract');
  }

  const version = spawnSync(CHROME, ['--version'], {
    encoding: 'utf8',
  }).stdout.trim();
  const profile = mkdtempSync(join(tmpdir(), 'atomos-v3-chrome-cdp-'));
  const activePortFile = join(profile, 'DevToolsActivePort');
  const child = spawn(
    CHROME,
    [
      '--headless=new',
      '--disable-background-networking',
      '--disable-component-update',
      '--disable-default-apps',
      '--disable-extensions',
      '--disable-gpu',
      '--disable-sync',
      '--enable-precise-memory-info',
      '--metrics-recording-only',
      '--no-default-browser-check',
      '--no-first-run',
      '--remote-debugging-port=0',
      `--user-data-dir=${profile}`,
      'about:blank',
    ],
    {
      stdio: ['ignore', 'ignore', 'pipe'],
    },
  );
  let stderr = '';
  child.stderr.setEncoding('utf8');
  child.stderr.on('data', (chunk) => {
    stderr = `${stderr}${chunk}`.slice(-20_000);
  });

  let client;
  try {
    await waitForFile(activePortFile, child, 15_000);
    const [debugPort, browserPath] = readFileSync(
      activePortFile,
      'utf8',
    ).trim().split(/\r?\n/u);
    if (
      !/^\d+$/u.test(debugPort ?? '')
      || !(browserPath ?? '').startsWith('/devtools/browser/')
    ) {
      throw new Error('Chrome wrote a malformed DevToolsActivePort file');
    }
    client = new CdpClient(
      `ws://${HOST}:${debugPort}${browserPath}`,
    );
    await client.open();
    const url = (
      `http://${HOST}:${port}/?warmup=${warmup}&iterations=${iterations}`
    );
    const target = await client.send('Target.createTarget', { url });
    const attached = await client.send('Target.attachToTarget', {
      targetId: target.targetId,
      flatten: true,
    });
    const sessionId = attached.sessionId;
    await client.send('Runtime.enable', {}, sessionId);
    await waitForAuditPromise(client, sessionId);
    const evaluated = await client.send(
      'Runtime.evaluate',
      {
        expression:
          '(async () => await window.__TIME_DOMAIN_V3_AUDIT_PROMISE__)()',
        awaitPromise: true,
        returnByValue: true,
      },
      sessionId,
      180_000,
    );
    if (evaluated.exceptionDetails !== undefined) {
      throw new Error(
        `browser audit evaluation failed: ${JSON.stringify(
          evaluated.exceptionDetails,
        )}`,
      );
    }
    const pageReport = evaluated.result?.value;
    if (pageReport === null || typeof pageReport !== 'object') {
      throw new Error('browser audit returned no structured report');
    }
    const pageReportSha256 = createHash('sha256')
      .update(JSON.stringify(pageReport))
      .digest('hex');
    const report = {
      ...pageReport,
      chrome_runner: {
        version,
        executable: CHROME,
        mode: 'headless-new-dedicated-worker-via-cdp',
        devtools_protocol: true,
        node: process.versions.node,
        contract_sha256: verified.contract_sha256,
        page_report_sha256: pageReportSha256,
      },
    };
    process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
    if (report.passed !== true) process.exitCode = 1;
    await client.send('Browser.close', {}, undefined, 10_000).catch(() => {});
  } finally {
    client?.close();
    if (child.exitCode === null) child.kill('SIGTERM');
    await Promise.race([
      new Promise((resolve) => child.once('exit', resolve)),
      delay(5_000),
    ]);
    if (
      basename(profile).startsWith('atomos-v3-chrome-cdp-')
      && dirname(profile) === tmpdir()
    ) {
      rmSync(profile, { recursive: true, force: true });
    }
    if (process.exitCode && stderr.length > 0) {
      process.stderr.write(`Chrome stderr (tail):\n${stderr}\n`);
    }
  }
}

void main().catch((error) => {
  process.stderr.write(
    `${error instanceof Error ? error.stack ?? error.message : String(error)}\n`,
  );
  process.exitCode = 1;
});
