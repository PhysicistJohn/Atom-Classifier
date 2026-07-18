import { parentPort } from 'node:worker_threads';
import { canonicalClassificationScenarios } from '../../TinySA_SignalLab/src/classification-corpus.js';
import {
  featureSamples,
  attemptSamplingTimingMs,
  type FeatureSamplingAttempt,
  type FeatureSamplingProgress,
} from './observable-training-sampling.js';
import type { ObservableTrainingAcquisitionRegime } from '../../TinySA/packages/analysis/src/observable-training-acquisition-geometry.js';

export const ATTEMPT_SAMPLING_WORKER_HEARTBEAT_INTERVAL_MS = 30_000;

export interface AttemptSamplingWorkItem {
  key: string;
  scenarioId: string;
  snrDb: number;
  regimeIndex: number;
  seed: number;
}

export interface AttemptSamplingWorkRequest {
  requestId: string;
  regimes: readonly ObservableTrainingAcquisitionRegime[];
  items: readonly AttemptSamplingWorkItem[];
}

export interface AttemptSamplingWorkResult {
  key: string;
  attempt?: FeatureSamplingAttempt;
  errorMessage?: string;
}

export interface AttemptSamplingWorkResponse {
  results: readonly AttemptSamplingWorkResult[];
  timingMs: typeof attemptSamplingTimingMs;
  wallClockMs: number;
}

export interface AttemptSamplingWorkerProgress {
  key: string;
  completedItems: number;
  totalItems: number;
  stage: 'starting' | 'consecutive-spectrum' | 'qualified-envelope' | 'completed';
  observationOpportunity?: number;
  observationHorizon?: number;
}

export type AttemptSamplingWorkerMessage =
  | {
    kind: 'progress';
    requestId: string;
    progress: AttemptSamplingWorkerProgress;
  }
  | {
    kind: 'result';
    requestId: string;
    response: AttemptSamplingWorkResponse;
  };

if (!parentPort) {
  throw new Error('observable-training-sampling-worker must run inside a worker thread');
}

const scenarioById = new Map(canonicalClassificationScenarios.map((scenario) => [scenario.id, scenario]));

parentPort.on('message', (request: AttemptSamplingWorkRequest) => {
  const wallClockStart = process.hrtime.bigint();
  const timingBefore = { ...attemptSamplingTimingMs };
  const results: AttemptSamplingWorkResult[] = [];
  let lastHeartbeatAt = 0;
  const progress = (
    item: AttemptSamplingWorkItem,
    completedItems: number,
    stage: AttemptSamplingWorkerProgress['stage'],
    detail?: FeatureSamplingProgress,
    force = false,
  ): void => {
    const now = Date.now();
    if (!force && now - lastHeartbeatAt < ATTEMPT_SAMPLING_WORKER_HEARTBEAT_INTERVAL_MS) {
      return;
    }
    lastHeartbeatAt = now;
    const message: AttemptSamplingWorkerMessage = {
      kind: 'progress',
      requestId: request.requestId,
      progress: {
        key: item.key,
        completedItems,
        totalItems: request.items.length,
        stage,
        ...(detail === undefined ? {} : {
          observationOpportunity: detail.observationOpportunity,
          observationHorizon: detail.observationHorizon,
        }),
      },
    };
    parentPort!.postMessage(message);
  };
  for (let itemIndex = 0; itemIndex < request.items.length; itemIndex += 1) {
    const item = request.items[itemIndex]!;
    progress(item, itemIndex, 'starting', undefined, true);
    const scenario = scenarioById.get(item.scenarioId);
    const regime = request.regimes[item.regimeIndex];
    if (!scenario || !regime) {
      results.push({ key: item.key, errorMessage: `Unresolvable scenario/regime for ${item.key}` });
      break;
    }
    try {
      const attempt = featureSamples(
        scenario,
        item.snrDb,
        regime,
        item.seed,
        (detail) => progress(item, itemIndex, detail.stage, detail),
      );
      results.push({ key: item.key, attempt });
      progress(item, itemIndex + 1, 'completed', undefined, true);
    } catch (error) {
      results.push({ key: item.key, errorMessage: error instanceof Error ? error.message : String(error) });
      // A deterministic work item will fail the same way on every retry.
      // Stop this chunk immediately instead of burning through unrelated
      // attempts and reporting the failure only after the full matrix.
      break;
    }
  }
  const wallClockMs = Number(process.hrtime.bigint() - wallClockStart) / 1e6;
  const timingMs = {
    spectrumSynthesis: attemptSamplingTimingMs.spectrumSynthesis - timingBefore.spectrumSynthesis,
    zeroSpanSynthesis: attemptSamplingTimingMs.zeroSpanSynthesis - timingBefore.zeroSpanSynthesis,
    detectAndTrack: attemptSamplingTimingMs.detectAndTrack - timingBefore.detectAndTrack,
    featureExtraction: attemptSamplingTimingMs.featureExtraction - timingBefore.featureExtraction,
    hashing: attemptSamplingTimingMs.hashing - timingBefore.hashing,
    attemptCount: attemptSamplingTimingMs.attemptCount - timingBefore.attemptCount,
  };
  const response: AttemptSamplingWorkResponse = { results, timingMs, wallClockMs };
  const message: AttemptSamplingWorkerMessage = {
    kind: 'result',
    requestId: request.requestId,
    response,
  };
  parentPort!.postMessage(message);
});
