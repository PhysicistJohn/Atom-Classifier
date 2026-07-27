import process from 'node:process';
import { preprocessTimeDomainInvariantPatches } from '../src/embedding/time-domain-invariant-patch-preprocess-v3.js';
import {
  assertTimeDomainV3Fixture,
  compareTimeDomainV3Result,
  decodeTimeDomainV3Case,
  outputDigest,
  outputsExactlyEqual,
  percentile,
  verifyTimeDomainV3PayloadHashes,
  type TimeDomainV3BrowserFixture,
} from './time-domain-v3-parity-common.js';
// The verifier remains plain ESM so the browser server can execute without a
// TypeScript loader or touching the shared dependency tree.
// @ts-expect-error No declaration is needed for this fixed local ESM module.
import { verifyStrictTimeDomainV3Contract } from './time-domain-v3-contract-verifier.mjs';

interface InvalidAudit {
  name: string;
  rejected: boolean;
  error: string | null;
}

function argument(
  name: string,
  fallback: number,
  maximum: number,
): number {
  const prefix = `--${name}=`;
  const raw = process.argv.find((value) => value.startsWith(prefix));
  if (raw === undefined) return fallback;
  const parsed = Number(raw.slice(prefix.length));
  if (!Number.isInteger(parsed) || parsed <= 0 || parsed > maximum) {
    throw new RangeError(`${name} must be an integer in [1, ${maximum}]`);
  }
  return parsed;
}

function invalidCase(name: string, operation: () => unknown): InvalidAudit {
  try {
    operation();
    return { name, rejected: false, error: null };
  } catch (error) {
    return {
      name,
      rejected: true,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

function invalidInputAudit(): InvalidAudit[] {
  const zero = new Float64Array(64);
  const validReal = Float64Array.of(1, 0, -1, 0);
  const validImaginary = Float64Array.of(0, 1, 0, -1);
  return [
    invalidCase('all-zero', () =>
      preprocessTimeDomainInvariantPatches(zero, zero)),
    invalidCase('too-short', () =>
      preprocessTimeDomainInvariantPatches(
        Float64Array.of(1, 1),
        Float64Array.of(0, 0),
      )),
    invalidCase('channel-length-mismatch', () =>
      preprocessTimeDomainInvariantPatches(
        Float64Array.of(1, 1, 1),
        Float64Array.of(0, 0),
      )),
    invalidCase('nan', () =>
      preprocessTimeDomainInvariantPatches(
        Float64Array.of(1, Number.NaN, 1),
        Float64Array.of(0, 0, 0),
      )),
    invalidCase('infinity', () =>
      preprocessTimeDomainInvariantPatches(
        Float64Array.of(1, 1, 1),
        Float64Array.of(0, Number.POSITIVE_INFINITY, 0),
      )),
    invalidCase('invalid-target-fraction', () =>
      preprocessTimeDomainInvariantPatches(validReal, validImaginary, {
        targetFrac: 0,
      })),
    invalidCase('unsafe-packed-length', () =>
      preprocessTimeDomainInvariantPatches(validReal, validImaginary, {
        patchCount: 10_000_001,
      })),
  ];
}

function memory(): NodeJS.MemoryUsage {
  return process.memoryUsage();
}

async function main(): Promise<void> {
  const unexpected = process.argv.slice(2).filter(
    (value) =>
      !value.startsWith('--warmup=')
      && !value.startsWith('--iterations='),
  );
  if (unexpected.length > 0) {
    throw new Error(
      `unknown arguments ${unexpected.join(', ')}; external fixture/data paths are refused`,
    );
  }
  const warmup = argument('warmup', 5, 100);
  const iterations = argument('iterations', 50, 1000);
  const verified = verifyStrictTimeDomainV3Contract() as {
    contract: Record<string, unknown>;
    fixture: unknown;
    observed_hashes: Record<string, string>;
    contract_sha256: string;
    verifier_sha256: string;
  };
  assertTimeDomainV3Fixture(verified.fixture);
  const fixture: TimeDomainV3BrowserFixture = verified.fixture;
  await verifyTimeDomainV3PayloadHashes(fixture);

  const memoryBefore = memory();
  const rows = [];
  for (const entry of fixture.cases) {
    const decoded = decodeTimeDomainV3Case(entry);
    for (let index = 0; index < warmup; index++) {
      preprocessTimeDomainInvariantPatches(
        decoded.inPhase,
        decoded.quadrature,
      );
    }
    const durations: number[] = [];
    let result = preprocessTimeDomainInvariantPatches(
      decoded.inPhase,
      decoded.quadrature,
    );
    for (let index = 0; index < iterations; index++) {
      const started = performance.now();
      result = preprocessTimeDomainInvariantPatches(
        decoded.inPhase,
        decoded.quadrature,
      );
      durations.push(performance.now() - started);
    }
    const replay = preprocessTimeDomainInvariantPatches(
      decoded.inPhase,
      decoded.quadrature,
    );
    const deterministic = outputsExactlyEqual(result, replay);
    const parity = compareTimeDomainV3Result(
      result,
      decoded,
      entry.expected_context,
    );
    const parityPassed = (
      parity.packed < 3e-5
      && parity.rawFeatures < 3e-4
      && parity.context < 2e-12
    );
    rows.push({
      name: entry.name,
      length: entry.length,
      warmup,
      iterations,
      latency_ms: {
        p50: percentile(durations, 0.5),
        p95: percentile(durations, 0.95),
        minimum: Math.min(...durations),
        maximum: Math.max(...durations),
      },
      active_length: result.context.active_length,
      selected_base_lag:
        result.context.geometry_estimator.selected_base_lag,
      parity,
      parity_passed: parityPassed,
      deterministic,
      output_sha256: await outputDigest(result),
    });
  }
  const invalidInputs = invalidInputAudit();
  const memoryAfter = memory();
  const passed = (
    rows.every((row) => row.parity_passed && row.deterministic)
    && invalidInputs.every((entry) => entry.rejected)
  );
  const report = {
    report_version: 'strict-time-domain-v3-node-runtime-audit-v1',
    passed,
    contract_version: verified.contract.contract_version,
    contract_sha256: verified.contract_sha256,
    verifier_sha256: verified.verifier_sha256,
    source_sha256: verified.observed_hashes,
    fixture: {
      version: fixture.fixture_version,
      synthetic_only: fixture.synthetic_only,
      reads_held_or_sealed_payload: fixture.reads_held_or_sealed_payload,
      uses_frequency_transform: fixture.uses_frequency_transform,
      python_source_sha256: fixture.python_source_sha256,
    },
    runtime: {
      node: process.versions.node,
      v8: process.versions.v8,
      platform: process.platform,
      architecture: process.arch,
    },
    rows,
    invalid_inputs: invalidInputs,
    memory: {
      method: 'process.memoryUsage',
      before: memoryBefore,
      after: memoryAfter,
      delta: {
        rss: memoryAfter.rss - memoryBefore.rss,
        heapTotal: memoryAfter.heapTotal - memoryBefore.heapTotal,
        heapUsed: memoryAfter.heapUsed - memoryBefore.heapUsed,
        external: memoryAfter.external - memoryBefore.external,
        arrayBuffers: memoryAfter.arrayBuffers - memoryBefore.arrayBuffers,
      },
    },
  };
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
  if (!passed) process.exitCode = 1;
}

void main().catch((error) => {
  process.stderr.write(
    `${error instanceof Error ? error.stack ?? error.message : String(error)}\n`,
  );
  process.exitCode = 1;
});
