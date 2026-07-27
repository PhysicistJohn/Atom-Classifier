interface MemorySnapshot {
  method: 'measureUserAgentSpecificMemory' | 'performance.memory' | 'unavailable';
  bytes: number | null;
  used_js_heap_size: number | null;
  total_js_heap_size: number | null;
  js_heap_size_limit: number | null;
  error: string | null;
}

interface PerformanceWithMemory extends Performance {
  measureUserAgentSpecificMemory?: () => Promise<{ bytes: number }>;
  memory?: {
    usedJSHeapSize: number;
    totalJSHeapSize: number;
    jsHeapSizeLimit: number;
  };
}

declare global {
  interface Window {
    __TIME_DOMAIN_V3_AUDIT__?: Record<string, unknown>;
    __TIME_DOMAIN_V3_AUDIT_PROMISE__?: Promise<Record<string, unknown>>;
  }
}

function integerQuery(
  parameters: URLSearchParams,
  name: string,
  fallback: number,
  maximum: number,
): number {
  const raw = parameters.get(name);
  if (raw === null) return fallback;
  const value = Number(raw);
  if (!Number.isInteger(value) || value <= 0 || value > maximum) {
    throw new RangeError(`${name} must be an integer in [1, ${maximum}]`);
  }
  return value;
}

async function memorySnapshot(): Promise<MemorySnapshot> {
  const runtime = performance as PerformanceWithMemory;
  if (
    crossOriginIsolated
    && typeof runtime.measureUserAgentSpecificMemory === 'function'
  ) {
    try {
      const measured = await Promise.race([
        runtime.measureUserAgentSpecificMemory(),
        new Promise<never>((_resolve, reject) => {
          setTimeout(() => reject(new Error('memory measurement timed out')), 20_000);
        }),
      ]);
      return {
        method: 'measureUserAgentSpecificMemory',
        bytes: measured.bytes,
        used_js_heap_size: null,
        total_js_heap_size: null,
        js_heap_size_limit: null,
        error: null,
      };
    } catch (error) {
      return {
        method: 'measureUserAgentSpecificMemory',
        bytes: null,
        used_js_heap_size: null,
        total_js_heap_size: null,
        js_heap_size_limit: null,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }
  if (runtime.memory !== undefined) {
    return {
      method: 'performance.memory',
      bytes: null,
      used_js_heap_size: runtime.memory.usedJSHeapSize,
      total_js_heap_size: runtime.memory.totalJSHeapSize,
      js_heap_size_limit: runtime.memory.jsHeapSizeLimit,
      error: null,
    };
  }
  return {
    method: 'unavailable',
    bytes: null,
    used_js_heap_size: null,
    total_js_heap_size: null,
    js_heap_size_limit: null,
    error: null,
  };
}

async function workerAudit(
  contract: Record<string, unknown>,
  warmup: number,
  iterations: number,
): Promise<{
  result: Record<string, unknown>;
  worker: Worker;
}> {
  const worker = new Worker('/worker.js', {
    type: 'module',
    name: 'strict-time-domain-v3-runtime-audit',
  });
  try {
    const result = await new Promise<Record<string, unknown>>((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('strict v3 worker audit timed out'));
      }, 120_000);
      worker.onmessage = (event: MessageEvent<Record<string, unknown>>) => {
        const payload = event.data;
        if (payload.type === 'strict-v3-audit-error') {
          clearTimeout(timeout);
          reject(new Error(String(payload.error)));
          return;
        }
        if (payload.type === 'strict-v3-audit-result') {
          clearTimeout(timeout);
          resolve(payload);
        }
      };
      worker.onerror = (event) => {
        clearTimeout(timeout);
        reject(new Error(event.message));
      };
      worker.postMessage({
        type: 'run-strict-v3-audit',
        fixtureUrl: '/fixture.json',
        warmup,
        iterations,
        contract,
      });
    });
    return { result, worker };
  } catch (error) {
    worker.terminate();
    throw error;
  }
}

async function run(): Promise<Record<string, unknown>> {
  const parameters = new URLSearchParams(location.search);
  const warmup = integerQuery(parameters, 'warmup', 5, 100);
  const iterations = integerQuery(parameters, 'iterations', 40, 1000);
  const contractResponse = await fetch('/contract.json', {
    cache: 'no-store',
    credentials: 'omit',
  });
  if (!contractResponse.ok) {
    throw new Error(`contract fetch failed with ${contractResponse.status}`);
  }
  const contract = await contractResponse.json() as Record<string, unknown>;
  const memoryBefore = await memorySnapshot();
  const started = performance.now();
  const execution = await workerAudit(contract, warmup, iterations);
  let memoryAfter: MemorySnapshot;
  try {
    memoryAfter = await memorySnapshot();
  } finally {
    execution.worker.terminate();
  }
  const elapsed = performance.now() - started;
  const passed = execution.result.passed === true;
  const result: Record<string, unknown> = {
    ...execution.result,
    browser_harness: {
      elapsed_ms: elapsed,
      warmup,
      iterations,
      location_origin: location.origin,
      cross_origin_isolated: crossOriginIsolated,
      memory_before: memoryBefore,
      memory_after: memoryAfter,
      memory_delta_bytes: (
        memoryBefore.bytes !== null && memoryAfter.bytes !== null
          ? memoryAfter.bytes - memoryBefore.bytes
          : null
      ),
      used_js_heap_delta: (
        memoryBefore.used_js_heap_size !== null
        && memoryAfter.used_js_heap_size !== null
          ? memoryAfter.used_js_heap_size - memoryBefore.used_js_heap_size
          : null
      ),
    },
  };
  window.__TIME_DOMAIN_V3_AUDIT__ = result;
  document.documentElement.dataset.auditStatus = (
    passed ? 'passed' : 'failed'
  );
  document.querySelector('#status')!.textContent = (
    passed ? 'PASS' : 'FAIL'
  );
  document.querySelector('#report')!.textContent = JSON.stringify(
    result,
    null,
    2,
  );
  return result;
}

window.__TIME_DOMAIN_V3_AUDIT_PROMISE__ = run().catch((error) => {
  const report = {
    type: 'strict-v3-browser-harness-error',
    passed: false,
    error: error instanceof Error ? error.message : String(error),
    stack: error instanceof Error ? error.stack ?? null : null,
  };
  window.__TIME_DOMAIN_V3_AUDIT__ = report;
  document.documentElement.dataset.auditStatus = 'error';
  document.querySelector('#status')!.textContent = 'ERROR';
  document.querySelector('#report')!.textContent = JSON.stringify(
    report,
    null,
    2,
  );
  return report;
});

export {};
