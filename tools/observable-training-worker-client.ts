import { randomUUID } from 'node:crypto';
import type { Worker } from 'node:worker_threads';
import type { ObservableTrainingAcquisitionRegime } from '../../TinySA/packages/analysis/src/observable-training-acquisition-geometry.js';
import type {
  AttemptSamplingWorkItem,
  AttemptSamplingWorkRequest,
  AttemptSamplingWorkResponse,
  AttemptSamplingWorkerMessage,
  AttemptSamplingWorkerProgress,
} from './observable-training-sampling-worker.js';

// Historical 75-cell chunks reached 48 minutes. The current ten-cell chunks
// emit progress from inside each attempt, so an idle worker can be bounded
// without confusing long healthy CPU work for a hang.
export const ATTEMPT_SAMPLING_WORKER_IDLE_TIMEOUT_MS = 10 * 60 * 1_000;
export const ATTEMPT_SAMPLING_WORKER_MAXIMUM_WALL_CLOCK_MS = 6 * 60 * 60 * 1_000;

export interface AttemptSamplingWorkerClientOptions {
  idleTimeoutMs?: number;
  maximumWallClockMs?: number;
  onProgress?: (progress: AttemptSamplingWorkerProgress) => void;
}

export function postToAttemptSamplingWorker(
  worker: Worker,
  regimes: readonly ObservableTrainingAcquisitionRegime[],
  items: readonly AttemptSamplingWorkItem[],
  options: AttemptSamplingWorkerClientOptions = {},
): Promise<AttemptSamplingWorkResponse> {
  const idleTimeoutMs = options.idleTimeoutMs ?? ATTEMPT_SAMPLING_WORKER_IDLE_TIMEOUT_MS;
  const maximumWallClockMs =
    options.maximumWallClockMs ?? ATTEMPT_SAMPLING_WORKER_MAXIMUM_WALL_CLOCK_MS;
  if (!Number.isFinite(idleTimeoutMs) || idleTimeoutMs <= 0
    || !Number.isFinite(maximumWallClockMs) || maximumWallClockMs <= 0) {
    throw new Error('Attempt-sampling worker timeouts must be positive finite milliseconds');
  }
  const requestId = randomUUID();
  return new Promise((resolvePromise, rejectPromise) => {
    let idleTimer: NodeJS.Timeout;
    let maximumTimer: NodeJS.Timeout;
    const cleanup = (): void => {
      clearTimeout(idleTimer);
      clearTimeout(maximumTimer);
      worker.off('message', onMessage);
      worker.off('messageerror', onMessageError);
      worker.off('error', onError);
      worker.off('exit', onExit);
    };
    const rejectWith = (error: Error): void => {
      cleanup();
      rejectPromise(error);
    };
    const armIdleTimer = (): void => {
      clearTimeout(idleTimer);
      idleTimer = setTimeout(() => rejectWith(new Error(
        `Attempt-sampling worker made no progress for ${idleTimeoutMs} ms`,
      )), idleTimeoutMs);
      idleTimer.unref();
    };
    const onMessage = (message: AttemptSamplingWorkerMessage): void => {
      if (!message || message.requestId !== requestId) {
        rejectWith(new Error('Attempt-sampling worker returned an unattributed message'));
        return;
      }
      if (message.kind === 'progress') {
        armIdleTimer();
        options.onProgress?.(message.progress);
        return;
      }
      if (message.kind !== 'result') {
        rejectWith(new Error('Attempt-sampling worker returned an unknown message kind'));
        return;
      }
      cleanup();
      resolvePromise(message.response);
    };
    const onMessageError = (error: Error): void => {
      rejectWith(new Error('Attempt-sampling worker response could not be deserialized', {
        cause: error,
      }));
    };
    const onError = (error: Error): void => rejectWith(error);
    const onExit = (code: number): void => {
      rejectWith(new Error(
        `Attempt-sampling worker exited with code ${code} before returning its chunk`,
      ));
    };
    worker.on('message', onMessage);
    worker.once('messageerror', onMessageError);
    worker.once('error', onError);
    worker.once('exit', onExit);
    armIdleTimer();
    maximumTimer = setTimeout(() => rejectWith(new Error(
      `Attempt-sampling worker exceeded ${maximumWallClockMs} ms maximum wall clock`,
    )), maximumWallClockMs);
    maximumTimer.unref();
    const request: AttemptSamplingWorkRequest = { requestId, regimes, items };
    try {
      worker.postMessage(request);
    } catch (error) {
      rejectWith(error instanceof Error ? error : new Error(String(error)));
    }
  });
}
