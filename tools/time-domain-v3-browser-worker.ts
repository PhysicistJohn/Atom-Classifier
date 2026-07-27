/// <reference lib="webworker" />

import { preprocessTimeDomainInvariantPatches } from '../src/embedding/time-domain-invariant-patch-preprocess-v3.js';
import {
  assertTimeDomainV3Fixture,
  compareTimeDomainV3Result,
  decodeTimeDomainV3Case,
  outputDigest,
  outputsExactlyEqual,
  percentile,
  verifyTimeDomainV3PayloadHashes,
} from './time-domain-v3-parity-common.js';

declare const self: DedicatedWorkerGlobalScope;

interface AuditRequest {
  type: 'run-strict-v3-audit';
  fixtureUrl: '/fixture.json';
  warmup: number;
  iterations: number;
  contract: Record<string, unknown>;
}

interface InvalidAudit {
  name: string;
  rejected: boolean;
  error: string | null;
}

function positiveBoundedInteger(
  value: number,
  name: string,
  maximum: number,
): number {
  if (!Number.isInteger(value) || value <= 0 || value > maximum) {
    throw new RangeError(`${name} must be an integer in [1, ${maximum}]`);
  }
  return value;
}

function invalidCase(
  name: string,
  operation: () => unknown,
): InvalidAudit {
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

function runInvalidInputAudit(): InvalidAudit[] {
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

self.onmessage = (event: MessageEvent<AuditRequest>): void => {
  void (async () => {
    try {
      const request = event.data;
      if (
        request?.type !== 'run-strict-v3-audit'
        || request.fixtureUrl !== '/fixture.json'
      ) {
        throw new Error('strict v3 worker refuses unknown requests or fixture paths');
      }
      const warmup = positiveBoundedInteger(request.warmup, 'warmup', 100);
      const iterations = positiveBoundedInteger(
        request.iterations,
        'iterations',
        1000,
      );
      const fixtureResponse = await fetch(request.fixtureUrl, {
        cache: 'no-store',
        credentials: 'omit',
      });
      if (!fixtureResponse.ok) {
        throw new Error(`fixture fetch failed with ${fixtureResponse.status}`);
      }
      const fixtureValue: unknown = await fixtureResponse.json();
      assertTimeDomainV3Fixture(fixtureValue);
      const fixture = fixtureValue;
      await verifyTimeDomainV3PayloadHashes(fixture);

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
        let finalResult = preprocessTimeDomainInvariantPatches(
          decoded.inPhase,
          decoded.quadrature,
        );
        for (let index = 0; index < iterations; index++) {
          const started = performance.now();
          finalResult = preprocessTimeDomainInvariantPatches(
            decoded.inPhase,
            decoded.quadrature,
          );
          durations.push(performance.now() - started);
        }
        const replay = preprocessTimeDomainInvariantPatches(
          decoded.inPhase,
          decoded.quadrature,
        );
        const deterministic = outputsExactlyEqual(finalResult, replay);
        const parity = compareTimeDomainV3Result(
          finalResult,
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
          active_length: finalResult.context.active_length,
          selected_base_lag:
            finalResult.context.geometry_estimator.selected_base_lag,
          parity,
          parity_passed: parityPassed,
          deterministic,
          output_sha256: await outputDigest(finalResult),
        });
      }

      const invalidInputs = runInvalidInputAudit();
      const passed = (
        rows.every((row) => row.parity_passed && row.deterministic)
        && invalidInputs.every((entry) => entry.rejected)
      );
      self.postMessage({
        type: 'strict-v3-audit-result',
        passed,
        contract: request.contract,
        fixture: {
          version: fixture.fixture_version,
          python_source_sha256: fixture.python_source_sha256,
          synthetic_only: fixture.synthetic_only,
          reads_held_or_sealed_payload:
            fixture.reads_held_or_sealed_payload,
          uses_frequency_transform: fixture.uses_frequency_transform,
        },
        runtime: {
          dedicated_worker: true,
          user_agent: self.navigator.userAgent,
          hardware_concurrency: self.navigator.hardwareConcurrency,
          cross_origin_isolated: self.crossOriginIsolated,
        },
        rows,
        invalid_inputs: invalidInputs,
      });
    } catch (error) {
      self.postMessage({
        type: 'strict-v3-audit-error',
        error: error instanceof Error ? error.message : String(error),
        stack: error instanceof Error ? error.stack ?? null : null,
      });
    }
  })();
};
