import { createHash } from 'node:crypto';
import { gunzipSync, gzipSync } from 'node:zlib';
import {
  chmodSync,
  existsSync,
  mkdtempSync,
  readFileSync,
  renameSync,
  rmSync,
  symlinkSync,
  unlinkSync,
  utimesSync,
  writeFileSync,
} from 'node:fs';
import { strict as assert } from 'node:assert';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { Worker } from 'node:worker_threads';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { canonicalClassificationScenarios } from '../../Atom-SignalLab/src/classification-corpus.js';
import {
  OBSERVABLE_TRAINING_BASELINE_QUALIFIED_ENVELOPE_TEMPORAL_SCHEDULE,
  OBSERVABLE_TRAINING_BASELINE_SPECTRUM_TEMPORAL_SCHEDULE,
  SIGNAL_LAB_PRODUCTION_ACQUISITION_GEOMETRY,
  SIGNAL_LAB_PRODUCTION_TEMPORAL_SCHEDULE_PAIRS,
  occupiedBandwidthRbwDivisorGeometry,
} from '../../Atom-Atomizer/packages/analysis/src/observable-training-acquisition-geometry.js';
import { featureSamples } from './observable-training-sampling.js';
import {
  ATTEMPT_SAMPLING_CACHE_MAX_COMPRESSED_CHUNK_BYTES,
  ATTEMPT_SAMPLING_CACHE_MAX_UNCOMPRESSED_CHUNK_BYTES,
  createAttemptSamplingCache,
} from './observable-training-attempt-cache.js';
import { postToAttemptSamplingWorker } from './observable-training-worker-client.js';
import {
  acquireObservableTrainingRun,
  openFreshSamplingRunJournal,
} from './observable-training-run-control.js';
import {
  assertGeneratedModelManifestPair,
  publishGeneratedModelManifestRecoverably,
  recoverGeneratedModelManifestPublication,
} from './observable-model-publication.js';
import {
  assertObservableTrainingBuildAttestation,
  OBSERVABLE_TRAINING_BUILD_ID_ENV,
  OBSERVABLE_TRAINING_NODE_VERSION_ENV,
  OBSERVABLE_TRAINING_TRAINER_SHA256_ENV,
  OBSERVABLE_TRAINING_WORKER_SHA256_ENV,
} from './observable-training-build-attestation.js';

const scenario = canonicalClassificationScenarios.find((candidate) => candidate.id === 'cw-rbw-line');
if (!scenario) throw new Error('Worker regression scenario is missing');
const bluetoothScenarioCandidate = canonicalClassificationScenarios.find(
  (candidate) => candidate.id === 'bluetooth-le-advertising',
);
const bluetoothSchedulePairCandidate = SIGNAL_LAB_PRODUCTION_TEMPORAL_SCHEDULE_PAIRS.find(
  (candidate) => candidate.sourcePlanProfileId === bluetoothScenarioCandidate?.id,
);
if (!bluetoothScenarioCandidate || !bluetoothSchedulePairCandidate) {
  throw new Error('Worker regression Bluetooth censor scenario/schedule is missing');
}
const bluetoothScenario = bluetoothScenarioCandidate;
const bluetoothSchedulePair = bluetoothSchedulePairCandidate;

const regime = {
  id: 'worker-regression/independent-production-branch-baselines-v1',
  geometry: occupiedBandwidthRbwDivisorGeometry(20),
  spectrumTemporalSchedule: OBSERVABLE_TRAINING_BASELINE_SPECTRUM_TEMPORAL_SCHEDULE,
  qualifiedEnvelopeTemporalSchedule:
    OBSERVABLE_TRAINING_BASELINE_QUALIFIED_ENVELOPE_TEMPORAL_SCHEDULE,
};
const bluetoothRegime = {
  id: `${SIGNAL_LAB_PRODUCTION_ACQUISITION_GEOMETRY.id}/${bluetoothSchedulePair.id}`,
  geometry: SIGNAL_LAB_PRODUCTION_ACQUISITION_GEOMETRY,
  spectrumTemporalSchedule: bluetoothSchedulePair.spectrumTemporalSchedule,
  qualifiedEnvelopeTemporalSchedule:
    bluetoothSchedulePair.qualifiedEnvelopeTemporalSchedule,
};
const validItem = {
  key: `${scenario.id}:snr=16:regime=${regime.id}:seed=407`,
  scenarioId: scenario.id,
  snrDb: 16,
  regimeIndex: 0,
  seed: 407,
};
const workerModuleUrl = new URL('./observable-training-sampling-worker.js', import.meta.url);
const completedChecks: string[] = [];

await checkWorkerSamplingIsByteIdenticalAndRequestLocal();
completedChecks.push('byte-identical-repeatable-request-local-timing');
await checkWorkerStopsAtFirstDeterministicError();
completedChecks.push('first-error-stops-chunk');
await checkWorkerIdleTimeout();
completedChecks.push('progress-heartbeat-and-idle-timeout');
checkContentAddressedAttemptCache();
completedChecks.push('content-addressed-partial-resume-corruption-and-nondeterminism');
await checkRunControlAndFreshJournal();
completedChecks.push('exclusive-run-pinned-runtime-and-fresh-journal');
checkRecoverableModelManifestPublication();
completedChecks.push('recoverable-fail-closed-model-manifest-publication');
checkImmutableBuildAttestation();
completedChecks.push('immutable-private-build-attestation');
checkGenerationScriptsEmitWorker();
completedChecks.push('private-build-launcher-and-worker-generation');

console.log(JSON.stringify({
  status: 'pass',
  checks: completedChecks,
}, null, 2));

async function checkWorkerSamplingIsByteIdenticalAndRequestLocal(): Promise<void> {
  const directAttempt = featureSamples(scenario!, validItem.snrDb, regime, validItem.seed);
  const directSha256 = sha256(directAttempt);
  const worker = new Worker(workerModuleUrl);
  try {
    const progressStages: string[] = [];
    const first = await postToAttemptSamplingWorker(worker, [regime], [validItem], {
      onProgress: (progress) => progressStages.push(progress.stage),
    });
    const second = await postToAttemptSamplingWorker(worker, [regime], [validItem]);
    assert.equal(first.results.length, 1);
    assert.equal(second.results.length, 1);
    assert.equal(first.results[0]?.errorMessage, undefined);
    assert.equal(second.results[0]?.errorMessage, undefined);
    assert.equal(sha256(first.results[0]?.attempt), directSha256);
    assert.equal(sha256(second.results[0]?.attempt), directSha256);
    assert.equal(first.timingMs.attemptCount, 1);
    assert.equal(second.timingMs.attemptCount, 1);
    assert.ok(progressStages.includes('starting'));
    assert.ok(progressStages.includes('completed'));
  } finally {
    await worker.terminate();
  }
}

async function checkWorkerStopsAtFirstDeterministicError(): Promise<void> {
  const worker = new Worker(workerModuleUrl);
  try {
    const invalidItem = {
      ...validItem,
      key: 'missing-scenario:first-error',
      scenarioId: 'missing-scenario',
    };
    const response = await postToAttemptSamplingWorker(
      worker,
      [regime],
      [invalidItem, validItem],
    );
    assert.deepEqual(response.results, [{
      key: invalidItem.key,
      errorMessage: `Unresolvable scenario/regime for ${invalidItem.key}`,
    }]);
    assert.equal(response.timingMs.attemptCount, 0);
  } finally {
    await worker.terminate();
  }
}

async function checkWorkerIdleTimeout(): Promise<void> {
  const worker = new Worker(
    'const { parentPort } = require("node:worker_threads"); parentPort.on("message", () => {});',
    { eval: true },
  );
  try {
    await assert.rejects(
      postToAttemptSamplingWorker(worker, [regime], [validItem], {
        idleTimeoutMs: 50,
        maximumWallClockMs: 1_000,
      }),
      /made no progress/,
    );
  } finally {
    await worker.terminate();
  }
}

function checkContentAddressedAttemptCache(): void {
  const rootDirectory = mkdtempSync(join(tmpdir(), 'tinysa-observable-attempt-cache-'));
  try {
    const directAttempt = featureSamples(scenario!, validItem.snrDb, regime, validItem.seed);
    const secondItem = {
      ...validItem,
      key: `${scenario!.id}:snr=16:regime=${regime.id}:seed=1407`,
      seed: 1_407,
    };
    const secondAttempt = featureSamples(
      scenario!,
      secondItem.snrDb,
      regime,
      secondItem.seed,
    );
    const items = [validItem, secondItem];
    const chunks = [[validItem], [secondItem]];
    const options = {
      rootDirectory,
      phase: 'fitting' as const,
      sourceIdentity: { testCorpus: 'worker-check-v1' },
      scenarios: [{ id: scenario!.id, value: scenario! }],
      snrLevels: [16],
      regimes: [regime],
      seeds: [407, 1_407],
      items,
      chunks,
      workerModuleUrl,
    };
    const first = createAttemptSamplingCache(options);
    const firstRecord = first.publishChunk(0, chunks[0]!, [{
      key: validItem.key,
      attempt: directAttempt,
    }]);
    const forgedCensoredAttempt = JSON.parse(JSON.stringify(directAttempt)) as typeof directAttempt;
    const forgedCensoredSample =
      forgedCensoredAttempt.qualifiedEnvelope.detectedPowerCaptureSample;
    assert.ok(forgedCensoredSample);
    forgedCensoredSample.detectedPowerEvidenceDisposition =
      'censored-frequency-agile-fixed-tune';
    forgedCensoredSample.fitEligible = false;
    forgedCensoredSample.envelopeUntimedFitEligible = false;
    for (const name of Object.keys(forgedCensoredSample.values)) {
      if (name.startsWith('envelope.')) {
        delete (forgedCensoredSample.values as Record<string, number>)[name];
      }
    }
    assert.throws(
      () => first.publishChunk(0, chunks[0]!, [{
        key: validItem.key,
        attempt: forgedCensoredAttempt,
      }]),
      /malformed worker results/,
      'a cache item must not relabel a non-agile scenario as a censored agile capture even with a spectrum-only vector',
    );
    assert.deepEqual(
      [...readFileSync(first.chunkPath(0)).subarray(0, 2)],
      [0x1f, 0x8b],
      'attempt cache chunks must be gzip streams',
    );
    const serializedFirstChunk = JSON.parse(
      gunzipSync(readFileSync(first.chunkPath(0))).toString('utf8'),
    ) as { schemaVersion: number; payload: { schemaVersion: number } };
    assert.equal(serializedFirstChunk.schemaVersion, 2);
    assert.equal(serializedFirstChunk.payload.schemaVersion, 2);

    const bluetoothItem = {
      key: `${bluetoothScenario.id}:snr=32:regime=${bluetoothRegime.id}:seed=407`,
      scenarioId: bluetoothScenario.id,
      snrDb: 32,
      regimeIndex: 0,
      seed: 407,
    };
    const bluetoothAttempt = featureSamples(
      bluetoothScenario,
      bluetoothItem.snrDb,
      bluetoothRegime,
      bluetoothItem.seed,
    );
    const bluetoothCaptureSample =
      bluetoothAttempt.qualifiedEnvelope.detectedPowerCaptureSample;
    assert.ok(bluetoothCaptureSample);
    assert.equal(
      bluetoothCaptureSample.detectedPowerEvidenceDisposition,
      'censored-frequency-agile-fixed-tune',
    );
    assert.equal(
      Object.keys(bluetoothCaptureSample.values).some((name) =>
        name.startsWith('envelope.')),
      false,
    );
    const bluetoothCache = createAttemptSamplingCache({
      rootDirectory,
      phase: 'calibration',
      sourceIdentity: { testCorpus: 'worker-check-bluetooth-censor-v2' },
      scenarios: [{ id: bluetoothScenario.id, value: bluetoothScenario }],
      snrLevels: [32],
      regimes: [bluetoothRegime],
      seeds: [407],
      items: [bluetoothItem],
      chunks: [[bluetoothItem]],
      workerModuleUrl,
    });
    bluetoothCache.publishChunk(0, [bluetoothItem], [{
      key: bluetoothItem.key,
      attempt: bluetoothAttempt,
    }]);
    assert.deepEqual(
      bluetoothCache.loadChunk(0, [bluetoothItem])?.results,
      [{ key: bluetoothItem.key, attempt: bluetoothAttempt }],
      'a valid receipt-verified spectrum-only agile censor must survive the fail-closed cache',
    );
    const forgedEnvelopeLeak = JSON.parse(
      JSON.stringify(bluetoothAttempt),
    ) as typeof bluetoothAttempt;
    const forgedEnvelopeLeakSample =
      forgedEnvelopeLeak.qualifiedEnvelope.detectedPowerCaptureSample;
    assert.ok(forgedEnvelopeLeakSample);
    (forgedEnvelopeLeakSample.values as Record<string, number>)[
      'envelope.rangeDb'
    ] = 1;
    assert.throws(
      () => bluetoothCache.publishChunk(0, [bluetoothItem], [{
        key: bluetoothItem.key,
        attempt: forgedEnvelopeLeak,
      }]),
      /malformed worker results/,
      'a censored agile cache sample must reject even one leaked envelope dimension',
    );

    const resumed = createAttemptSamplingCache(options);
    const resumedFirstResults = resumed.loadChunk(0, chunks[0]!)?.results;
    assert.deepEqual(resumedFirstResults, [{
      key: validItem.key,
      attempt: directAttempt,
    }]);
    assert.equal(
      JSON.stringify(resumedFirstResults?.[0]?.attempt),
      JSON.stringify(directAttempt),
      'cache round-trip must preserve runtime property order as well as values',
    );
    assert.deepEqual(resumed.publishChunk(0, chunks[0]!, [{
      key: validItem.key,
      attempt: directAttempt,
    }]), firstRecord);
    assert.equal(resumed.loadChunk(1, chunks[1]!), undefined);
    const secondRecord = resumed.publishChunk(1, chunks[1]!, [{
      key: secondItem.key,
      attempt: secondAttempt,
    }]);
    resumed.seal([firstRecord, secondRecord]);
    assert.equal(existsSync(resumed.manifestPath()), true);
    assert.deepEqual(resumed.loadChunk(1, chunks[1]!)?.results, [{
      key: secondItem.key,
      attempt: secondAttempt,
    }]);

    const reordered = createAttemptSamplingCache({
      ...options,
      items: [secondItem, validItem],
      chunks: [[secondItem], [validItem]],
    });
    assert.notEqual(reordered.fingerprint, resumed.fingerprint);
    assert.equal(reordered.loadChunk(0, [secondItem]), undefined);

    writeFileSync(resumed.chunkPath(1), 'corrupt cache bytes');
    assert.equal(resumed.loadChunk(1, chunks[1]!), undefined);
    assert.equal(resumed.corruptChunkCount, 1);
    const repairedRecord = resumed.publishChunk(1, chunks[1]!, [{
      key: secondItem.key,
      attempt: secondAttempt,
    }]);
    assert.deepEqual(repairedRecord, secondRecord);
    assert.deepEqual(resumed.loadChunk(1, chunks[1]!)?.results, [{
      key: secondItem.key,
      attempt: secondAttempt,
    }]);

    const validChunkBytes = readFileSync(resumed.chunkPath(1));
    const malformedEnvelope = JSON.parse(
      gunzipSync(validChunkBytes).toString('utf8'),
    ) as {
      payloadSha256: string;
      payload: {
        results: Array<{
          attempt: {
            consecutiveSpectrum: {
              onlineSpectrumRepresentatives: Array<{
                values: Record<string, number>;
              }>;
            };
          };
        }>;
      };
    };
    const malformedValues = malformedEnvelope.payload.results[0]?.attempt
      .consecutiveSpectrum.onlineSpectrumRepresentatives[0]?.values;
    assert.ok(malformedValues);
    delete malformedValues['spectrum.logBandwidthHz'];
    malformedEnvelope.payloadSha256 = createHash('sha256')
      .update(JSON.stringify(malformedEnvelope.payload))
      .digest('hex');
    writeFileSync(
      resumed.chunkPath(1),
      gzipSync(Buffer.from(JSON.stringify(malformedEnvelope)), { level: 6 }),
    );
    assert.equal(
      resumed.loadChunk(1, chunks[1]!),
      undefined,
      'a correctly hashed envelope with a malformed feature attempt must be rejected',
    );
    assert.equal(resumed.corruptChunkCount, 2);
    resumed.publishChunk(1, chunks[1]!, [{
      key: secondItem.key,
      attempt: secondAttempt,
    }]);

    writeFileSync(
      resumed.chunkPath(1),
      gzipSync(Buffer.alloc(
        ATTEMPT_SAMPLING_CACHE_MAX_UNCOMPRESSED_CHUNK_BYTES + 1,
      )),
    );
    assert.equal(
      resumed.loadChunk(1, chunks[1]!),
      undefined,
      'a gzip stream expanding beyond the bounded cache limit must be rejected',
    );
    assert.equal(resumed.corruptChunkCount, 3);
    resumed.publishChunk(1, chunks[1]!, [{
      key: secondItem.key,
      attempt: secondAttempt,
    }]);

    writeFileSync(
      resumed.chunkPath(1),
      Buffer.alloc(ATTEMPT_SAMPLING_CACHE_MAX_COMPRESSED_CHUNK_BYTES + 1),
    );
    assert.equal(
      resumed.loadChunk(1, chunks[1]!),
      undefined,
      'an oversized compressed cache file must be rejected before decompression',
    );
    assert.equal(resumed.corruptChunkCount, 4);
    resumed.publishChunk(1, chunks[1]!, [{
      key: secondItem.key,
      attempt: secondAttempt,
    }]);

    const symlinkTarget = `${resumed.chunkPath(1)}.regular-target`;
    renameSync(resumed.chunkPath(1), symlinkTarget);
    symlinkSync(symlinkTarget, resumed.chunkPath(1));
    assert.equal(
      resumed.loadChunk(1, chunks[1]!),
      undefined,
      'a symlinked cache chunk must be rejected without following it',
    );
    unlinkSync(resumed.chunkPath(1));
    renameSync(symlinkTarget, resumed.chunkPath(1));

    const differentAttempt = JSON.parse(JSON.stringify(directAttempt)) as typeof directAttempt;
    const firstSample = differentAttempt.consecutiveSpectrum.onlineSpectrumRepresentatives[0];
    assert.ok(firstSample);
    const firstDimension = Object.keys(firstSample.values)[0];
    assert.ok(firstDimension);
    (firstSample.values as Record<string, number>)[firstDimension] =
      firstSample.values[firstDimension]! + 1;
    assert.throws(
      () => resumed.publishChunk(0, chunks[0]!, [{
        key: validItem.key,
        attempt: differentAttempt,
      }]),
      /cache nondeterminism/,
    );
  } finally {
    rmSync(rootDirectory, { recursive: true, force: true });
  }
}

async function checkRunControlAndFreshJournal(): Promise<void> {
  const root = mkdtempSync(join(tmpdir(), 'tinysa-observable-run-control-'));
  const lockPath = join(root, 'trainer.lock');
  const runRoot = join(root, 'runs');
  const first = acquireObservableTrainingRun({
    lockPath,
    runRoot,
    sourceWorkerModuleUrl: workerModuleUrl,
  });
  try {
    assert.notEqual(first.workerModuleUrl.href, workerModuleUrl.href);
    first.assertWorkerRuntimeUnchanged();
    const worker = new Worker(first.workerModuleUrl);
    try {
      const response = await postToAttemptSamplingWorker(worker, [regime], [validItem]);
      assert.equal(response.results[0]?.errorMessage, undefined);
    } finally {
      await worker.terminate();
    }
    assert.throws(() => acquireObservableTrainingRun({
      lockPath,
      runRoot,
      sourceWorkerModuleUrl: workerModuleUrl,
    }), /already running/);
  } finally {
    first.release();
  }
  const second = acquireObservableTrainingRun({
    lockPath,
    runRoot,
    sourceWorkerModuleUrl: workerModuleUrl,
  });
  second.release();
  const retainedRunRootSentinel = join(runRoot, 'must-survive-stale-lock-recovery');
  writeFileSync(retainedRunRootSentinel, 'retained');
  writeFileSync(lockPath, JSON.stringify({
    schemaVersion: 2,
    ownerToken: 'untrusted-stale-owner',
    pid: findDeadPid(),
    processStartIdentity: null,
    startedAt: new Date(0).toISOString(),
    runDirectory: runRoot,
  }));
  const stableMalformedLockTimestamp = new Date(Date.now() - 2 * 60_000);
  utimesSync(lockPath, stableMalformedLockTimestamp, stableMalformedLockTimestamp);
  const recovered = acquireObservableTrainingRun({
    lockPath,
    runRoot,
    sourceWorkerModuleUrl: workerModuleUrl,
  });
  try {
    assert.equal(
      existsSync(retainedRunRootSentinel),
      true,
      'stale-lock recovery must never remove the run root itself',
    );
  } finally {
    recovered.release();
  }

  const journalPath = join(root, 'fresh', 'journal.json');
  const freshRunsRoot = join(root, 'fresh', 'runs');
  const compatibility = 'a'.repeat(64);
  const newRun = openFreshSamplingRunJournal({
    journalPath,
    runsRoot: freshRunsRoot,
    compatibilitySha256: compatibility,
  });
  assert.equal(newRun.resumed, false);
  const resumed = openFreshSamplingRunJournal({
    journalPath,
    runsRoot: freshRunsRoot,
    compatibilitySha256: compatibility,
  });
  assert.equal(resumed.resumed, true);
  assert.equal(resumed.runId, newRun.runId);
  resumed.markCompleted();
  assert.equal(
    existsSync(resumed.cacheRoot),
    false,
    'a completed fresh run must discard its never-reusable cache',
  );
  const later = openFreshSamplingRunJournal({
    journalPath,
    runsRoot: freshRunsRoot,
    compatibilitySha256: compatibility,
  });
  assert.equal(later.resumed, false);
  assert.notEqual(later.runId, resumed.runId);
  const incompatibleRunSentinel = join(later.cacheRoot, 'must-be-pruned');
  writeFileSync(incompatibleRunSentinel, 'unreachable');
  const incompatible = openFreshSamplingRunJournal({
    journalPath,
    runsRoot: freshRunsRoot,
    compatibilitySha256: 'b'.repeat(64),
  });
  assert.equal(incompatible.resumed, false);
  assert.notEqual(incompatible.runId, later.runId);
  assert.equal(
    existsSync(incompatibleRunSentinel),
    false,
    'an incompatible in-progress fresh run must be pruned when its journal is replaced',
  );
  rmSync(root, { recursive: true, force: true });
}

function checkRecoverableModelManifestPublication(): void {
  const root = mkdtempSync(join(tmpdir(), 'tinysa-observable-publication-'));
  try {
    const modelPath = join(root, 'model.ts');
    const manifestPath = join(root, 'manifest.ts');
    const journalPath = join(root, 'publication.json');
    const oldPair = testLegacyModelManifestPair('old payload');
    const newPair = testModelManifestPair('new payload', 'd'.repeat(64));
    writeFileSync(modelPath, oldPair.modelSource);
    writeFileSync(manifestPath, oldPair.manifestSource);
    assert.throws(
      () => assertGeneratedModelManifestPair(modelPath, manifestPath),
      /not a fail-closed matching pair/,
    );
    assert.throws(() => publishGeneratedModelManifestRecoverably({
      modelPath,
      manifestPath,
      journalPath,
      ...newPair,
      failAfterModelRenameForTest: true,
    }), /Injected failure/);
    assert.throws(
      () => assertGeneratedModelManifestPair(modelPath, manifestPath),
      /not a fail-closed matching pair/,
    );
    // The backup is a hard link to the original inode. Replace, rather than
    // truncate, the destination so the durable backup remains trustworthy.
    unlinkSync(manifestPath);
    writeFileSync(manifestPath, 'corrupt interrupted manifest\n');
    assert.throws(() => recoverGeneratedModelManifestPublication({
      modelPath,
      manifestPath,
      journalPath,
      failAfterModelRestoreForTest: true,
    }), /Injected failure after generated model restore/);
    assert.equal(readFileSync(modelPath, 'utf8'), oldPair.modelSource);
    assert.equal(readFileSync(manifestPath, 'utf8'), 'corrupt interrupted manifest\n');
    assert.equal(recoverGeneratedModelManifestPublication({
      modelPath,
      manifestPath,
      journalPath,
    }), 'restored-old');
    assert.equal(readFileSync(modelPath, 'utf8'), oldPair.modelSource);
    assert.equal(readFileSync(manifestPath, 'utf8'), oldPair.manifestSource);
    publishGeneratedModelManifestRecoverably({
      modelPath,
      manifestPath,
      journalPath,
      ...newPair,
    });
    assert.equal(
      assertGeneratedModelManifestPair(modelPath, manifestPath).modelContentSha256,
      'd'.repeat(64),
    );

    const finalizedOldRoot = mkdtempSync(join(root, 'finalized-old-'));
    const finalizedOldModelPath = join(finalizedOldRoot, 'model.ts');
    const finalizedOldManifestPath = join(finalizedOldRoot, 'manifest.ts');
    const finalizedOldJournalPath = join(finalizedOldRoot, 'publication.json');
    writeFileSync(finalizedOldModelPath, oldPair.modelSource, { flag: 'wx' });
    writeFileSync(finalizedOldManifestPath, oldPair.manifestSource, { flag: 'wx' });
    assert.throws(() => publishGeneratedModelManifestRecoverably({
      modelPath: finalizedOldModelPath,
      manifestPath: finalizedOldManifestPath,
      journalPath: finalizedOldJournalPath,
      ...newPair,
      failAfterModelRenameForTest: true,
    }), /Injected failure/);
    const finalizedOldTransactionId = JSON.parse(
      readFileSync(finalizedOldJournalPath, 'utf8'),
    ).transactionId as string;
    renameSync(
      `${finalizedOldModelPath}.${finalizedOldTransactionId}.backup`,
      finalizedOldModelPath,
    );
    assert.equal(recoverGeneratedModelManifestPublication({
      modelPath: finalizedOldModelPath,
      manifestPath: finalizedOldManifestPath,
      journalPath: finalizedOldJournalPath,
    }), 'finalized-old');
    assert.equal(readFileSync(finalizedOldModelPath, 'utf8'), oldPair.modelSource);
    assert.equal(readFileSync(finalizedOldManifestPath, 'utf8'), oldPair.manifestSource);

    const finalizedNewRoot = mkdtempSync(join(root, 'finalized-new-'));
    const finalizedNewModelPath = join(finalizedNewRoot, 'model.ts');
    const finalizedNewManifestPath = join(finalizedNewRoot, 'manifest.ts');
    const finalizedNewJournalPath = join(finalizedNewRoot, 'publication.json');
    writeFileSync(finalizedNewModelPath, oldPair.modelSource, { flag: 'wx' });
    writeFileSync(finalizedNewManifestPath, oldPair.manifestSource, { flag: 'wx' });
    assert.throws(() => publishGeneratedModelManifestRecoverably({
      modelPath: finalizedNewModelPath,
      manifestPath: finalizedNewManifestPath,
      journalPath: finalizedNewJournalPath,
      ...newPair,
      failAfterModelRenameForTest: true,
    }), /Injected failure/);
    const finalizedNewTransactionId = JSON.parse(
      readFileSync(finalizedNewJournalPath, 'utf8'),
    ).transactionId as string;
    renameSync(
      `${finalizedNewManifestPath}.${finalizedNewTransactionId}.staged`,
      finalizedNewManifestPath,
    );
    assert.equal(recoverGeneratedModelManifestPublication({
      modelPath: finalizedNewModelPath,
      manifestPath: finalizedNewManifestPath,
      journalPath: finalizedNewJournalPath,
    }), 'finalized-new');
    assert.equal(
      assertGeneratedModelManifestPair(
        finalizedNewModelPath,
        finalizedNewManifestPath,
      ).modelContentSha256,
      'd'.repeat(64),
    );

    const initialRoot = mkdtempSync(join(root, 'initial-'));
    const initialModelPath = join(initialRoot, 'model.ts');
    const initialManifestPath = join(initialRoot, 'manifest.ts');
    const initialJournalPath = join(initialRoot, 'publication.json');
    assert.throws(() => publishGeneratedModelManifestRecoverably({
      modelPath: initialModelPath,
      manifestPath: initialManifestPath,
      journalPath: initialJournalPath,
      ...newPair,
      failAfterModelRenameForTest: true,
    }), /Injected failure/);
    assert.equal(recoverGeneratedModelManifestPublication({
      modelPath: initialModelPath,
      manifestPath: initialManifestPath,
      journalPath: initialJournalPath,
    }), 'removed-partial-initial');
    assert.equal(existsSync(initialModelPath), false);
    assert.equal(existsSync(initialManifestPath), false);
    assert.equal(existsSync(initialJournalPath), false);

    const malformedUuidRoot = mkdtempSync(join(root, 'malformed-uuid-'));
    const malformedUuidJournalPath = join(malformedUuidRoot, 'publication.json');
    writeFileSync(malformedUuidJournalPath, JSON.stringify({
      schemaVersion: 1,
      transactionId: '------------------------------------',
      newModelSha256: 'a'.repeat(64),
      newManifestSha256: 'b'.repeat(64),
      oldModelSha256: null,
      oldManifestSha256: null,
    }), { flag: 'wx' });
    assert.throws(() => recoverGeneratedModelManifestPublication({
      modelPath: join(malformedUuidRoot, 'model.ts'),
      manifestPath: join(malformedUuidRoot, 'manifest.ts'),
      journalPath: malformedUuidJournalPath,
    }), /journal is malformed/);

    const unknownFieldRoot = mkdtempSync(join(root, 'unknown-field-'));
    const unknownFieldJournalPath = join(unknownFieldRoot, 'publication.json');
    writeFileSync(unknownFieldJournalPath, JSON.stringify({
      schemaVersion: 1,
      transactionId: '00000000-0000-4000-8000-000000000000',
      newModelSha256: 'a'.repeat(64),
      newManifestSha256: 'b'.repeat(64),
      oldModelSha256: null,
      oldManifestSha256: null,
      ignored: true,
    }), { flag: 'wx' });
    assert.throws(() => recoverGeneratedModelManifestPublication({
      modelPath: join(unknownFieldRoot, 'model.ts'),
      manifestPath: join(unknownFieldRoot, 'manifest.ts'),
      journalPath: unknownFieldJournalPath,
    }), /journal is malformed/);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}

function testLegacyModelManifestPair(payload: string): {
  modelSource: string;
  manifestSource: string;
} {
  const modelSource = `export const PAYLOAD = ${JSON.stringify(payload)};\n`;
  const modelSourceSha256 = createHash('sha256').update(modelSource).digest('hex');
  return {
    modelSource,
    manifestSource:
      `export const BAYESIAN_OBSERVABLE_MODEL_SHA256 = '${modelSourceSha256}' as const;\n`,
  };
}

function testModelManifestPair(payload: string, contentSha256: string): {
  modelSource: string;
  manifestSource: string;
} {
  const modelSource =
    `export const BAYESIAN_OBSERVABLE_MODEL_CONTENT_SHA256 = '${contentSha256}' as const;\n`
    + `export const PAYLOAD = ${JSON.stringify(payload)};\n`;
  const modelSourceSha256 = createHash('sha256').update(modelSource).digest('hex');
  return {
    modelSource,
    manifestSource:
      `export const BAYESIAN_OBSERVABLE_MODEL_SHA256 = '${modelSourceSha256}' as const;\n`
      + `export const BAYESIAN_OBSERVABLE_MODEL_CONTENT_SHA256 = '${contentSha256}' as const;\n`,
  };
}

function checkImmutableBuildAttestation(): void {
  const root = mkdtempSync(join(tmpdir(), 'tinysa-observable-build-attestation-'));
  const trainerPath = join(root, 'train-observable-classifier.js');
  const workerPath = join(root, 'observable-training-sampling-worker.js');
  try {
    writeFileSync(trainerPath, 'export const trainer = true;\n');
    writeFileSync(workerPath, 'export const worker = true;\n');
    const trainerSha256 = sha256Bytes(readFileSync(trainerPath));
    const workerSha256 = sha256Bytes(readFileSync(workerPath));
    chmodSync(trainerPath, 0o400);
    chmodSync(workerPath, 0o400);
    chmodSync(root, 0o500);
    const environment = {
      [OBSERVABLE_TRAINING_BUILD_ID_ENV]: '00000000-0000-4000-8000-000000000000',
      [OBSERVABLE_TRAINING_NODE_VERSION_ENV]: process.versions.node,
      [OBSERVABLE_TRAINING_TRAINER_SHA256_ENV]: trainerSha256,
      [OBSERVABLE_TRAINING_WORKER_SHA256_ENV]: workerSha256,
    };
    if (process.platform === 'win32') {
      assert.throws(
        () => assertObservableTrainingBuildAttestation({
          trainerModuleUrl: pathToFileURL(trainerPath),
          workerModuleUrl: pathToFileURL(workerPath),
          environment,
        }),
        /Windows admission is denied because immutability cannot be verified/,
      );
      return;
    }
    const attestation = assertObservableTrainingBuildAttestation({
      trainerModuleUrl: pathToFileURL(trainerPath),
      workerModuleUrl: pathToFileURL(workerPath),
      environment,
    });
    assert.equal(attestation.trainerSha256, trainerSha256);
    assert.equal(attestation.workerSha256, workerSha256);
    assert.equal(attestation.nodeVersion, process.versions.node);
    assert.equal(attestation.v8Version, process.versions.v8);
    assert.equal(
      attestation.workerRuntimeSha256,
      sha256Bytes(JSON.stringify([{
        path: 'observable-training-sampling-worker.js',
        sha256: workerSha256,
      }])),
    );
    assert.throws(
      () => assertObservableTrainingBuildAttestation({
        trainerModuleUrl: pathToFileURL(trainerPath),
        workerModuleUrl: pathToFileURL(workerPath),
        environment: {
          ...environment,
          [OBSERVABLE_TRAINING_WORKER_SHA256_ENV]: 'f'.repeat(64),
        },
      }),
      /does not match executed bytes/,
    );
    assert.throws(
      () => assertObservableTrainingBuildAttestation({
        trainerModuleUrl: pathToFileURL(trainerPath),
        workerModuleUrl: pathToFileURL(workerPath),
        environment: {
          ...environment,
          [OBSERVABLE_TRAINING_NODE_VERSION_ENV]: '0.0.0',
        },
      }),
      /runtime Node\.js .* does not match launcher pin/,
    );
    chmodSync(root, 0o700);
    chmodSync(workerPath, 0o600);
    chmodSync(root, 0o500);
    assert.throws(
      () => assertObservableTrainingBuildAttestation({
        trainerModuleUrl: pathToFileURL(trainerPath),
        workerModuleUrl: pathToFileURL(workerPath),
        environment,
      }),
      /artifact is writable/,
    );
  } finally {
    chmodSync(root, 0o700);
    rmSync(root, { recursive: true, force: true });
  }
}

function checkGenerationScriptsEmitWorker(): void {
  const packageJson = JSON.parse(readFileSync('package.json', 'utf8')) as {
    scripts: Record<string, string>;
  };
  assert.equal(
    packageJson.scripts['train:signal-classifier'],
    'node tools/run-observable-training.mjs train',
  );
  assert.equal(
    packageJson.scripts['check:signal-classifier-model'],
    'node tools/run-observable-training.mjs check',
  );
  assert.match(
    packageJson.scripts['check:observable-training-worker'] ?? '',
    /\bcheck:observable-training-launcher\b/,
  );
  const launcherSource = readFileSync('tools/run-observable-training.mjs', 'utf8');
  assert.match(launcherSource, /tools\/train-observable-classifier\.ts/);
  assert.match(launcherSource, /tools\/observable-training-sampling-worker\.ts/);
  assert.match(launcherSource, /--no-splitting/);
  assert.match(launcherSource, /--treeshake/);
  assert.match(launcherSource, /mkdtempSync/);
  assert.match(launcherSource, /\.node-version/);
  assert.match(launcherSource, /process\.version !== `v\$\{pin\}`/);
  assert.doesNotMatch(launcherSource, /22\.23\.1/);
  assert.match(launcherSource, /\['--check', '--fresh-sampling'\]/);
  assert.doesNotMatch(launcherSource, /dist\/tools/);
  const trainerSource = readFileSync('tools/train-observable-classifier.ts', 'utf8');
  assert.match(trainerSource, /assertObservableTrainingBuildAttestation\(\{/);
  assert.match(
    trainerSource,
    /TRAINER_RUN\.workerRuntimeSha256 !== BUILD_ATTESTATION\.workerRuntimeSha256/,
  );
  assert.match(
    trainerSource,
    /const FRESH_SAMPLING = CLI_ARGUMENTS\.includes\('--fresh-sampling'\);/,
    'the production trainer must derive fresh mode only from the forwarded CLI flag',
  );
  assert.match(
    trainerSource,
    /const FRESH_SAMPLING_RUN = FRESH_SAMPLING\s+\? openFreshSamplingRunJournal\(\{[\s\S]*?journalPath: resolve\('\.artifacts\/observable-training-fresh-check\/journal\.json'\),[\s\S]*?runsRoot: resolve\('\.artifacts\/observable-training-fresh-check\/runs'\),/,
    'fresh mode must own its isolated journal and run namespace',
  );
  assert.match(
    trainerSource,
    /const cache = FRESH_SAMPLING_RUN\s+\? createAttemptSamplingCache\(\{\s+rootDirectory: resolve\(FRESH_SAMPLING_RUN\.cacheRoot, 'v1'\),[\s\S]*?\}\)\s+: baselineCache;/,
    'fresh mode must sample through its per-run cache rather than the trusted local cache',
  );
  assert.equal(
    [...trainerSource.matchAll(/trainingRuntimeIdentity: TRAINING_RUNTIME_IDENTITY/g)].length,
    2,
    'fresh-run compatibility and generated metadata must both bind the exact runtime identity',
  );
  const attemptCacheSource = readFileSync(
    'tools/observable-training-attempt-cache.ts',
    'utf8',
  );
  assert.match(
    attemptCacheSource,
    /versions: \{ \.\.\.process\.versions \}/,
    'normal and fresh sampling cache fingerprints must bind the executing Node/V8 runtime',
  );
  assert.match(
    trainerSource,
    /\bconst CHUNK_SIZE = 10\b/,
    'attempt sampling must checkpoint in small restart units',
  );
  const completionCalls = [...trainerSource.matchAll(
    /FRESH_SAMPLING_RUN\?\.markCompleted\(\)/g,
  )];
  assert.equal(completionCalls.length, 1);
  assert.ok(
    completionCalls[0]!.index
      > trainerSource.lastIndexOf('publishGeneratedModelManifestRecoverably({'),
    'fresh sampling must remain resumable through model checks and publication',
  );
  assert.ok(
    completionCalls[0]!.index < trainerSource.lastIndexOf('console.log(JSON.stringify({'),
    'fresh sampling completion must precede only the final success summary',
  );
  const workerBundleSource = readFileSync(fileURLToPath(workerModuleUrl), 'utf8');
  assert.doesNotMatch(
    workerBundleSource,
    /BAYESIAN_OBSERVABLE_MODEL|BAYESIAN_OBSERVABLE_MODEL_CONTENT_SHA256/,
    'attempt-sampling worker bundle must not contain the generated model whose provenance records its digest',
  );
}

function sha256(value: unknown): string {
  return createHash('sha256').update(JSON.stringify(value)).digest('hex');
}

function sha256Bytes(value: string | Buffer): string {
  return createHash('sha256').update(value).digest('hex');
}

function findDeadPid(): number {
  for (let pid = 10_000; pid < 100_000; pid += 1) {
    try {
      process.kill(pid, 0);
    } catch (error) {
      if (error instanceof Error && 'code' in error && error.code === 'ESRCH') return pid;
    }
  }
  throw new Error('Worker check could not locate an unused process ID');
}
