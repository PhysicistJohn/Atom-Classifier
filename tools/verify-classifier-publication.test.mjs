import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { createHash } from 'node:crypto';
import { copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import test from 'node:test';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';

const execFileAsync = promisify(execFile);
const REPOSITORY_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const VERIFIER_PATH = 'tools/verify-classifier-publication.mjs';
const MODEL_PATH = 'src/models/bayesian-observable.generated.ts';
const MANIFEST_PATH = 'src/models/bayesian-observable.manifest.generated.ts';
const REPORT_PATH = '.artifacts/classifier-validation/report.json';
const NODE_VERSION_PATH = '.node-version';
const PUBLICATION_PATHS = [
  'README.md',
  'docs/BAYESIAN_DETECTION_CLASSIFICATION_RESEARCH.md',
  'docs/SIGNALLAB_EMSO_CLASSIFIER_CONTRACT.md',
  'docs/UI_UX_CONTRACTS.md',
];
const CLASSIFIER_FIXTURE_PATHS = [
  VERIFIER_PATH,
  MODEL_PATH,
  MANIFEST_PATH,
  REPORT_PATH,
  NODE_VERSION_PATH,
];
const MODEL_PAYLOAD_PATTERN = /export const BAYESIAN_OBSERVABLE_MODEL: ObservableClassifierModelAsset = (\{[\s\S]*\});\s*$/;
const MODEL_MANIFEST_PATTERN = /(BAYESIAN_OBSERVABLE_MODEL_SHA256 = ')([a-f0-9]{64})(' as const;)/;
const MODEL_CONTENT_PATTERN = /(BAYESIAN_OBSERVABLE_MODEL_CONTENT_SHA256 = ')([a-f0-9]{64})(' as const;)/;
const CURRENT_CALIBRATION_ID =
  'synthetic-independent-branch-view-matched-causal-acquisition-support-rank-detector-conditioned-physical-uncalibrated-v20';
const CURRENT_DETECTED_POWER_ACQUISITION_QUALIFICATION =
  'receipt-verified-provenance-bound-runtime-admitted-physical-capture-v5';
const CURRENT_DETECTED_POWER_SELECTION_CONDITION =
  'automatic-current-source-sweep-integrated-excess-rank-0';
const CURRENT_FREQUENCY_AGILE_ENVELOPE_CENSORING_POLICY_ID =
  'frequency-agile-fixed-tune-envelope-censoring-v1';
const CURRENT_FREQUENCY_AGILE_ENVELOPE_CENSORING_LIMITATION =
  'frequency-agile-fixed-tune-envelope-censored';

test('classifier publication verifier is fail-closed under isolated mutations', async (t) => {
  assert.ok(Number.parseInt(process.versions.node, 10) >= 22, 'mutation tests require Node 22 or newer');

  await t.test('accepts an unmodified publication fixture', async () => {
    await withFixture(async (root) => assertVerifierPasses(root));
  });

  await mutationTest(t, 'rejects a forged generated-model content identity', /generated model content SHA-256/, async (root) => {
    const forgedContentSha256 = 'f'.repeat(64);
    const modelPath = resolve(root, MODEL_PATH);
    const modelSource = await readFile(modelPath, 'utf8');
    const forgedModelSource = modelSource.replace(
      MODEL_CONTENT_PATTERN,
      (_match, prefix, _previous, suffix) => `${prefix}${forgedContentSha256}${suffix}`,
    );
    assert.notEqual(forgedModelSource, modelSource);
    await writeFile(modelPath, forgedModelSource);

    const manifestPath = resolve(root, MANIFEST_PATH);
    let manifest = await readFile(manifestPath, 'utf8');
    manifest = manifest.replace(
      MODEL_MANIFEST_PATTERN,
      (_match, prefix, _previous, suffix) =>
        `${prefix}${sha256(forgedModelSource)}${suffix}`,
    );
    manifest = manifest.replace(
      MODEL_CONTENT_PATTERN,
      (_match, prefix, _previous, suffix) => `${prefix}${forgedContentSha256}${suffix}`,
    );
    await writeFile(manifestPath, manifest);
  });

  await mutationTest(t, 'rejects a malformed attempt-sampling worker runtime identity', /worker runtime SHA-256/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      model.trainingMatrix.attemptSamplingWorkerRuntimeSha256 = 'not-a-sha256';
    });
    await mutateReport(root, (report) => {
      report.model.attemptSamplingWorkerRuntimeSha256 = 'not-a-sha256';
      report.matrix.attemptSamplingWorkerRuntimeSha256 = 'not-a-sha256';
      report.validationAcceptance.attemptSamplingWorkerRuntimeSha256 = 'not-a-sha256';
    }, { seal: false });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects a changed training Node runtime identity', /training runtime identity|Node\.js version pin/, async (root) => {
    const changedIdentity = {
      policyId: 'exact-repository-node-version-v1',
      nodeVersion: '22.23.2',
      v8Version: '12.4.254.21-node.56',
    };
    await mutateGeneratedModel(root, (model) => {
      model.trainingMatrix.trainingRuntimeIdentity = changedIdentity;
    });
    await mutateReport(root, (report) => {
      report.model.trainingRuntimeIdentity = changedIdentity;
      report.matrix.trainingRuntimeIdentity = changedIdentity;
      report.validationAcceptance.trainingRuntimeIdentity = changedIdentity;
    }, { seal: false });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects a changed corpus version', /corpus version/, async (root) => {
    await mutateGeneratedModel(root, (model) => { model.corpusVersion = 'observable-scalar-corpus-v999'; });
    await mutateReport(root, (report) => {
      report.corpus.version = 'observable-scalar-corpus-v999';
      report.validationAcceptance.corpusVersion = 'observable-scalar-corpus-v999';
    }, { seal: false });
    await rebindModelAsset(root);
  });

  const identifierMutations = [
    {
      name: 'model ID', expected: /classifier model ID|generated classifier model ID/,
      generatedField: 'id', reportField: 'id', attestationField: 'modelId', value: 'bayesian-observable-equivalence-v999',
    },
    {
      name: 'preprocessing ID', expected: /preprocessing ID/,
      generatedField: 'preprocessing', reportField: 'preprocessing', attestationField: 'preprocessing', value: 'scalar-observable-features-v999',
    },
    {
      name: 'prior ID', expected: /prior ID/,
      generatedField: 'priorId', reportField: 'priorId', attestationField: 'priorId', value: 'engineering-design-class-weights-v999',
    },
    {
      name: 'calibration ID', expected: /calibration ID/,
      generatedField: 'calibrationId', reportField: 'calibrationId', attestationField: 'calibrationId',
      value: 'synthetic-independent-branch-view-matched-causal-acquisition-support-rank-detector-conditioned-physical-uncalibrated-v16',
    },
    {
      name: 'decision-policy ID', expected: /decision-policy ID|validator acceptance decisionPolicyId/,
      reportField: 'decisionPolicyId', attestationField: 'decisionPolicyId', value: 'observable-open-set-decision-v999',
    },
    {
      name: 'source-commit provenance', expected: /source commit/,
      generatedField: 'sourceCommit', reportField: 'sourceCommit', attestationField: 'sourceCommit',
      value: '0000000000000000000000000000000000000000',
    },
  ];
  for (const mutation of identifierMutations) {
    await mutationTest(t, `rejects a changed ${mutation.name}`, mutation.expected, async (root) => {
      if (mutation.generatedField) {
        await mutateGeneratedModel(root, (model) => { model[mutation.generatedField] = mutation.value; });
      }
      await mutateReport(root, (report) => {
        report.model[mutation.reportField] = mutation.value;
        report.validationAcceptance[mutation.attestationField] = mutation.value;
      }, { seal: !mutation.generatedField });
      if (mutation.generatedField) await rebindModelAsset(root);
    });
  }

  await mutationTest(t, 'rejects a changed detected-power synthesis filter width', /synthesis-filter policy/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      model.trainingMatrix.detectedPowerSynthesisFilterPolicy.signalLabProductionSynthesisFilterWidthHz = 100_001;
    });
    await mutateReport(root, (report) => {
      report.matrix.detectedPowerSynthesisFilterPolicy.signalLabProductionSynthesisFilterWidthHz = 100_001;
    }, { seal: false });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects a changed detected-power RBW qualification', /synthesis-filter policy/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      model.trainingMatrix.detectedPowerSynthesisFilterPolicy.measurementActualRbwQualification = 'available';
    });
    await mutateReport(root, (report) => {
      report.matrix.detectedPowerSynthesisFilterPolicy.measurementActualRbwQualification = 'available';
    }, { seal: false });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects a changed qualified-envelope production coverage policy', /production high-SNR coverage policy/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      model.trainingMatrix.productionAcquisitionRegimeHighSnrSeedCoveragePolicy
        .qualifiedEnvelope.outOfDomainCapturePolicy = 'forged-policy';
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects a changed spectrum production coverage minimum', /production high-SNR coverage policy/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      model.trainingMatrix.productionAcquisitionRegimeHighSnrSeedCoveragePolicy
        .spectrumOnly.minimumDistinctObservationDomainEligibleSeedsPerHighSnrCell += 1;
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects a changed production temporal offset', /production acquisition regime/, async (root) => {
    await mutateProductionSchedule(root, 'spectrum', 2, (schedule) => {
      schedule.sourceLookIndexOffset += 1;
    });
  });

  await mutationTest(t, 'rejects a changed qualified-envelope production temporal offset', /production acquisition regime/, async (root) => {
    await mutateProductionSchedule(root, 'qualifiedEnvelope', 2, (schedule) => {
      schedule.sourceLookIndexOffset += 1;
    });
  });

  await mutationTest(t, 'rejects a changed release-plan source offset', /production acquisition regime|release-gate source-plan/, async (root) => {
    await mutateProductionSourcePlan(root, 'spectrum', 2, (profile) => {
      profile.sourceLookIndexOffset += 1;
    });
  });

  await mutationTest(t, 'rejects a changed release-plan horizon', /production acquisition regime|release-gate source-plan/, async (root) => {
    await mutateProductionSourcePlan(root, 'qualifiedEnvelope', 10, (profile) => {
      profile.spectrumOpportunities -= 1;
    });
  });

  await mutationTest(t, 'rejects a changed release-plan physical-capture count', /production acquisition regime|release-gate source-plan/, async (root) => {
    await mutateProductionSourcePlan(root, 'qualifiedEnvelope', 4, (profile) => {
      profile.admittedDetectedPowerCaptures = 0;
    });
  });

  await mutationTest(t, 'rejects an automatic spectrum release-plan capture', /production acquisition regime|release-gate source-plan/, async (root) => {
    await mutateProductionSourcePlan(root, 'spectrum', 4, (profile) => {
      profile.automaticDetectedPowerCaptures = 1;
    });
  });

  await mutationTest(t, 'rejects a changed detected-power capture policy', /production acquisition regime|source-clock policy/, async (root) => {
    await mutateProductionSourceClock(root, 'qualifiedEnvelope', (sourceClock) => {
      sourceClock.detectedPowerCapturePolicy = 'capture-after-fixed-eight-spectrum-opportunities-v0';
    });
  });

  await mutationTest(t, 'rejects a changed detected-power target-selection policy', /production acquisition regime|source-clock policy/, async (root) => {
    await mutateProductionSourceClock(root, 'qualifiedEnvelope', (sourceClock) => {
      sourceClock.captureTargetSelectionPolicy =
        'lexical-representative-key-first-v0';
    });
  });

  await mutationTest(t, 'rejects an auto-capture spectrum source clock', /production acquisition regime|source-clock policy/, async (root) => {
    await mutateProductionSourceClock(root, 'spectrum', (sourceClock) => {
      sourceClock.detectedPowerCapturePolicy =
        'capture-once-after-first-runtime-admitted-strongest-current-target-v2';
    });
  });

  await mutationTest(t, 'rejects fixed-skip fields added to a causal schedule', /production acquisition regime/, async (root) => {
    await mutateProductionSchedule(root, 'qualifiedEnvelope', 1, (schedule) => {
      schedule.skipAfterSpectrumOpportunities = 8;
      schedule.skippedSourceOpportunities = 1;
    });
  });

  await mutationTest(t, 'rejects a changed production schedule-pair identity', /production acquisition regime|schedule-pair IDs/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      model.trainingMatrix.signalLabProductionAcquisitionRegime.temporalSchedulePairs[0].id =
        'forged-independent-branch-pair-v0';
    });
    await mutateReport(root, (report) => {
      report.matrix.tailCalibrationAudit.pinnedSignalLabProductionAcquisitionRegime
        .temporalSchedulePairs[0].id = 'forged-independent-branch-pair-v0';
    }, { seal: false });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects a changed acquisition-branch policy', /acquisition-branch policy|production acquisition regime/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      model.trainingMatrix.acquisitionBranchPolicy = 'shared-session-v0';
      model.trainingMatrix.signalLabProductionAcquisitionRegime.branchPolicy = 'shared-session-v0';
    });
    await mutateReport(root, (report) => {
      report.matrix.tailCalibrationAudit.pinnedSignalLabProductionAcquisitionRegime.branchPolicy =
        'shared-session-v0';
    }, { seal: false });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects a changed held-out causal start', /held-out validation temporal schedule|held-out source range/, async (root) => {
    await mutateReport(root, (report) => {
      report.matrix.temporalSchedules.qualifiedEnvelope.sourceLookIndexOffset = 525;
    });
  });

  await mutationTest(t, 'rejects an omitted full-corpus scenario identity', /validation full-corpus scenario identity set/, async (root) => {
    await mutateReport(root, (report) => {
      report.matrix.scenarioSelection.scenarioIds.pop();
    });
  });

  await mutationTest(t, 'rejects a mismatched held-out source-span scenario identity', /held-out source-span complete corpus scenario identity set/, async (root) => {
    await mutateReport(root, (report) => {
      report.matrix.heldOutSourceSpanAudit.scenarios[0].scenarioId = 'forged-held-out-scenario';
    });
  });

  await mutationTest(t, 'rejects an impossible held-out declared drift bound', /exact maximum declared drift|maximum declared drift must not exceed/, async (root) => {
    await mutateReport(root, (report) => {
      const scenario = report.matrix.heldOutSourceSpanAudit.scenarios[0];
      scenario.maximumAbsoluteDeclaredDriftHz = scenario.availableCenterDriftMarginHz + 1;
    });
  });

  await mutationTest(t, 'rejects a relaxed tail-comparison tolerance', /score tolerance/, async (root) => {
    await mutateReport(root, (report) => {
      report.matrix.tailCalibrationAudit.independentRecomputation.scoreTolerance = 1e-11;
    });
  });

  await mutationTest(t, 'rejects a duplicate tail-comparison key', /comparison key set/, async (root) => {
    await mutateReport(root, (report) => {
      const comparisons = report.matrix.tailCalibrationAudit.independentRecomputation.scoreComparisons;
      assert.ok(comparisons.length >= 2, 'baseline must contain at least two tail comparisons');
      comparisons[1].classId = comparisons[0].classId;
      comparisons[1].view = comparisons[0].view;
    });
  });

  await mutationTest(t, 'rejects a missing tail-comparison key', /comparison key set/, async (root) => {
    await mutateReport(root, (report) => {
      const comparisons = report.matrix.tailCalibrationAudit.independentRecomputation.scoreComparisons;
      assert.ok(comparisons.length > 0, 'baseline must contain tail comparisons');
      comparisons.pop();
    });
  });

  await mutationTest(t, 'rejects a false tail-calibration audit boolean', /tail-calibration audit/, async (root) => {
    await mutateReport(root, (report) => { report.matrix.tailCalibrationAudit.valid = false; });
  });

  await mutationTest(t, 'rejects a changed nested tail-calibration by-view count', /tail-calibration.*score\/count reconciliation|tail-calibration.*count/s, async (root) => {
    let scenarioId;
    await mutateGeneratedModel(root, (model) => {
      scenarioId = Object.keys(model.trainingMatrix.tailCalibrationAttemptCountsByScenarioByView)[0];
      assert.ok(scenarioId, 'baseline model must publish nested tail-calibration counts');
      model.trainingMatrix.tailCalibrationAttemptCountsByScenarioByView[scenarioId]['spectrum-only'] += 1;
    });
    await mutateReport(root, (report) => {
      report.matrix.tailCalibrationAudit.attemptCountsByScenarioByView[scenarioId]['spectrum-only'] += 1;
      report.matrix.tailCalibrationAudit.independentRecomputation
        .recomputedAttemptCountsByScenarioByView[scenarioId]['spectrum-only'] += 1;
    }, { seal: false });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects an omitted nested tail-calibration view', /tail-calibration.*view key set/s, async (root) => {
    let scenarioId;
    await mutateGeneratedModel(root, (model) => {
      scenarioId = Object.keys(model.trainingMatrix.tailCalibrationAttemptCountsByScenarioByView)[0];
      assert.ok(scenarioId, 'baseline model must publish nested tail-calibration counts');
      delete model.trainingMatrix.tailCalibrationAttemptCountsByScenarioByView[scenarioId]['envelope-timed'];
    });
    await mutateReport(root, (report) => {
      delete report.matrix.tailCalibrationAudit.attemptCountsByScenarioByView[scenarioId]['envelope-timed'];
      delete report.matrix.tailCalibrationAudit.independentRecomputation
        .recomputedAttemptCountsByScenarioByView[scenarioId]['envelope-timed'];
    }, { seal: false });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects tail-calibration counts too small to resolve the support rank', /cannot resolve the pinned minimum support rank/, async (root) => {
    let scenarioId;
    await mutateGeneratedModel(root, (model) => {
      scenarioId = Object.keys(model.trainingMatrix.tailCalibrationAttemptCountsByScenarioByView)[0];
      model.trainingMatrix.tailCalibrationAttemptCountsByScenarioByView[scenarioId]['spectrum-only'] = 39;
    });
    await mutateReport(root, (report) => {
      report.matrix.tailCalibrationAudit.attemptCountsByScenarioByView[scenarioId]['spectrum-only'] = 39;
      report.matrix.tailCalibrationAudit.independentRecomputation
        .recomputedAttemptCountsByScenarioByView[scenarioId]['spectrum-only'] = 39;
    }, { seal: false });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects a forged causal source-clock trace hash', /source-clock trace hash/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      model.trainingMatrix.causalSamplingAudit.attributedSourceClockTraceAudit.fitting
        .consecutiveSpectrumSha256 = '0'.repeat(64);
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects a forged qualified-envelope source-clock trace hash', /source-clock trace hash/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      model.trainingMatrix.causalSamplingAudit.attributedSourceClockTraceAudit.tailCalibration
        .qualifiedEnvelopeSha256 = '0'.repeat(64);
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects a stale causal-sampling schema', /causal-sampling audit schema/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      model.trainingMatrix.causalSamplingAudit.schemaVersion = 2;
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects a detected-power sample in the no-capture spectrum branch', /consecutive-spectrum detected-power sample count/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      model.trainingMatrix.causalSamplingAudit.fitting.runtimeBranches.consecutiveSpectrum
        .detectedPowerCaptureSampleCount = 1;
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects inconsistent physical-versus-receipt capture accounting', /physical\/receipt capture accounting/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      model.trainingMatrix.causalSamplingAudit.fitting.runtimeBranches.qualifiedEnvelope
        .receiptVerifiedDetectedPowerCaptureSampleCount -= 1;
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects inconsistent admitted-versus-censored sample accounting', /admitted\/censored sample partition/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      model.trainingMatrix.causalSamplingAudit.tailCalibration.runtimeBranches.qualifiedEnvelope
        .censoredFrequencyAgileFixedTuneCaptureCount -= 1;
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects a censored scenario total that disagrees with the causal audit', /censored scenario\/capture-audit reconciliation/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      model.trainingMatrix.censoredFrequencyAgileFixedTuneCaptureCountsByScenario.fitting
        ['bluetooth-le-advertising'] -= 1;
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects a spectrum-branch physical capture', /consecutive-spectrum physical capture count/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      model.trainingMatrix.causalSamplingAudit.fitting.runtimeBranches.consecutiveSpectrum
        .physicalDetectedPowerCaptureCount = 1;
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects a changed qualified-envelope audit policy', /qualified-envelope capture policy/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      model.trainingMatrix.causalSamplingAudit.fitting.runtimeBranches.qualifiedEnvelope
        .detectedPowerCapturePolicyId = 'manual-capture-v0';
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects inconsistent spectrum branch event accounting', /consecutive-spectrum source-clock accounting/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      model.trainingMatrix.causalSamplingAudit.tailCalibration.runtimeBranches.consecutiveSpectrum
        .sourceClockEventCount += 1;
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects inconsistent spectrum attempt/representative accounting', /consecutive-spectrum representative accounting/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      const spectrum = model.trainingMatrix.causalSamplingAudit.fitting
        .runtimeBranches.consecutiveSpectrum;
      assert.ok(spectrum.onlineSpectrumRepresentativeCount
        > spectrum.attemptsWithAnyRepresentative,
      'baseline spectrum population must contain a multi-representative attempt');
      spectrum.multiRepresentativeAttemptCount = 0;
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects an impossible spectrum maximum representatives count', /consecutive-spectrum representative accounting/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      const spectrum = model.trainingMatrix.causalSamplingAudit.fitting
        .runtimeBranches.consecutiveSpectrum;
      assert.ok(spectrum.multiRepresentativeAttemptCount > 0,
        'baseline spectrum population must contain a multi-representative attempt');
      spectrum.maximumRepresentativesPerAttempt = spectrum.onlineSpectrumRepresentativeCount;
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects an out-of-range spectrum observation opportunity', /observation opportunities must be in \[1, 96\]/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      const counts = model.trainingMatrix.causalSamplingAudit.fitting
        .runtimeBranches.consecutiveSpectrum.observationOpportunityCounts;
      const key = Object.keys(counts)[0];
      assert.ok(key, 'baseline spectrum audit must publish observation opportunities');
      counts['0'] = counts[key];
      delete counts[key];
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects a missing full-band observation horizon', /observationHorizonCounts exact key set/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      const spectrum = model.trainingMatrix.causalSamplingAudit.fitting
        .runtimeBranches.consecutiveSpectrum;
      const fullBandAttempts = spectrum.observationHorizonCounts['96'];
      assert.ok(fullBandAttempts > 0, 'baseline spectrum audit must publish full-band horizons');
      spectrum.observationHorizonCounts['32'] += fullBandAttempts;
      delete spectrum.observationHorizonCounts['96'];
      spectrum.spectrumAcquisitionCount -= fullBandAttempts * (96 - 32);
      spectrum.sourceClockEventCount = spectrum.spectrumAcquisitionCount;
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects inconsistent causal pre/post unavailable counters', /unavailable-window partition/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      model.trainingMatrix.causalSamplingAudit.fitting.runtimeBranches.qualifiedEnvelope
        .preCaptureProvenanceUnavailableWindowCount += 1;
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects a duplicated causal unavailable-attempt identity', /unavailable-attempt canonical identity\/order|unavailable-attempt\/window total/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      const audit = model.trainingMatrix.causalSamplingAudit;
      audit.provenanceUnavailableAttempts.fitting.qualifiedEnvelope.push(
        { attemptId: 'forged-duplicate-attempt', unavailableWindowCount: 1 },
        { attemptId: 'forged-duplicate-attempt', unavailableWindowCount: 1 },
      );
      audit.fitting.runtimeBranches.qualifiedEnvelope.provenanceUnavailableWindowCount += 2;
      audit.fitting.runtimeBranches.qualifiedEnvelope
        .preCaptureProvenanceUnavailableWindowCount += 2;
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects an unavailable physical envelope capture', /unavailable physical envelope capture count|physical-capture availability/, async (root) => {
    await mutateReport(root, (report) => {
      report.admission.unavailablePhysicalEnvelopeCaptures = 1;
    });
  });

  await mutationTest(t, 'rejects a changed frequency-agile fixed-tune censoring policy ID', /frequency-agile.*(?:envelope-)?censoring policy|censoringPolicy\.id|capture-outcome censoring policy/s, async (root) => {
    const forgedPolicyId = 'frequency-agile-fixed-tune-envelope-censoring-v0';
    await mutateGeneratedModel(root, (model) => {
      assert.equal(
        model.trainingMatrix.frequencyAgileFixedTuneEnvelopeCensoringPolicy.id,
        CURRENT_FREQUENCY_AGILE_ENVELOPE_CENSORING_POLICY_ID,
        'baseline model must publish the canonical censoring policy ID',
      );
      model.trainingMatrix.frequencyAgileFixedTuneEnvelopeCensoringPolicy.id =
        forgedPolicyId;
    });
    await mutateReport(root, (report) => {
      assert.equal(
        report.admission.detectedPowerCaptureOutcomes.censoringPolicy.id,
        CURRENT_FREQUENCY_AGILE_ENVELOPE_CENSORING_POLICY_ID,
        'baseline report must publish the canonical censoring policy ID',
      );
      report.matrix.frequencyAgileFixedTuneEnvelopeCensoringPolicy.id =
        forgedPolicyId;
      report.admission.detectedPowerCaptureOutcomes.censoringPolicy.id = forgedPolicyId;
      report.admission.frequencyAgileEnvelopeCensoring.policyId = forgedPolicyId;
      for (const cell of report.admission.causalEnvelopeAvailabilityCells) {
        if (cell.envelopeEvidenceCensoringPolicyId
          === CURRENT_FREQUENCY_AGILE_ENVELOPE_CENSORING_POLICY_ID) {
          cell.envelopeEvidenceCensoringPolicyId = forgedPolicyId;
        }
      }
    }, { seal: false });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects a physical detected-power capture total that disagrees with its outcome audit', /physical detected-power capture.*(total|denominator|reconciliation)|physicalDetectedPowerCaptureCount|physicalDetectedPowerCaptures/s, async (root) => {
    await mutateReport(root, (report) => {
      assert.ok(report.admission.physicalDetectedPowerCaptures > 0,
        'baseline report must publish physical detected-power captures');
      report.admission.physicalDetectedPowerCaptures -= 1;
    });
  });

  await mutationTest(t, 'rejects a physical envelope total above the qualified causal-envelope population', /physicalEnvelopeCaptures|qualified envelope.*representative|qualified.*physical envelope denominator/s, async (root) => {
    await mutateReport(root, (report) => {
      report.admission.physicalEnvelopeCaptures += 1;
    });
  });

  await mutationTest(t, 'rejects a broken qualified-versus-censored detected-power outcome partition', /qualified.*censored.*physical detected-power|detected-power capture outcome.*(sum|partition)|physical capture qualified-envelope\/censored-spectrum partition/s, async (root) => {
    await mutateReport(root, (report) => {
      report.admission.detectedPowerCaptureOutcomes
        .censoredDetectedPowerCaptureCount += 1;
    });
  });

  await mutationTest(t, 'rejects a receipt-qualified detected-power count below the physical capture total', /receipt-qualified.*physical( detected-power)?|receiptQualifiedPhysicalCaptureCount/s, async (root) => {
    await mutateReport(root, (report) => {
      const outcomes = report.admission.detectedPowerCaptureOutcomes;
      assert.ok(outcomes.receiptQualifiedPhysicalCaptureCount > 0,
        'baseline outcome audit must contain receipt-qualified captures');
      outcomes.receiptQualifiedPhysicalCaptureCount -= 1;
    });
  });

  await mutationTest(t, 'rejects a changed reported acquisition qualification', /detected-power acquisition qualification required/, async (root) => {
    await mutateReport(root, (report) => {
      report.admission.detectedPowerAcquisitionQualification.required = 'manual-capture-v0';
    });
  });

  await mutationTest(t, 'rejects a changed reported automatic selection condition', /automatic selection condition/, async (root) => {
    await mutateReport(root, (report) => {
      report.admission.detectedPowerAcquisitionQualification
        .automaticSelectionConditionRequired = 'operator-preferred-current-target';
    });
  });

  await mutationTest(t, 'rejects a forged qualified-envelope denominator', /detected-power qualified\/physical envelope denominator/, async (root) => {
    await mutateReport(root, (report) => {
      report.admission.detectedPowerAcquisitionQualification.qualifiedEnvelopeSamples -= 1;
    });
  });

  await mutationTest(t, 'rejects an unqualified envelope population', /excluded unqualifiedEnvelopeSamples/, async (root) => {
    await mutateReport(root, (report) => {
      report.admission.detectedPowerAcquisitionQualification.unqualifiedEnvelopeSamples = 1;
    });
  });

  await mutationTest(t, 'rejects a missing or unissued-receipt envelope population', /excluded missingOrUnissuedReceiptEnvelopeSamples/, async (root) => {
    await mutateReport(root, (report) => {
      report.admission.detectedPowerAcquisitionQualification
        .missingOrUnissuedReceiptEnvelopeSamples = 1;
    });
  });

  await mutationTest(t, 'rejects a missing or unissued-receipt envelope feature attempt', /excluded missingOrUnissuedReceiptEnvelopeFeatureAttempts/, async (root) => {
    await mutateReport(root, (report) => {
      report.admission.detectedPowerAcquisitionQualification
        .missingOrUnissuedReceiptEnvelopeFeatureAttempts = 1;
    });
  });

  await mutationTest(t, 'rejects a forged per-attempt receipt qualification', /receipt qualification/, async (root) => {
    await mutateReport(root, (report) => {
      const captured = report.admission.causalEnvelopeAvailabilityCells
        .find((cell) => cell.detectedPowerCaptureCount === 1);
      assert.ok(captured, 'baseline report must publish a captured envelope cell');
      captured.detectedPowerAcquisitionReceiptQualified = false;
    });
  });

  await mutationTest(t, 'rejects a captured availability cell without receipt schema 4', /availability.*receipt schema.*4|detectedPowerCaptureReceiptSchemaVersion/s, async (root) => {
    await mutateReport(root, (report) => {
      const captured = report.admission.causalEnvelopeAvailabilityCells
        .find((cell) => cell.detectedPowerCaptureCount === 1);
      assert.ok(captured, 'baseline report must publish a captured availability cell');
      assert.equal(captured.detectedPowerCaptureReceiptSchemaVersion, 4,
        'baseline captured cell must publish receipt schema 4');
      captured.detectedPowerCaptureReceiptSchemaVersion = 1;
    });
  });

  await mutationTest(t, 'rejects an admitted envelope availability cell without the automatic selection condition', /availability.*automatic selection condition/, async (root) => {
    await mutateReport(root, (report) => {
      const admitted = report.admission.causalEnvelopeAvailabilityCells
        .find((cell) => cell.detectedPowerEvidenceDisposition === 'admitted-envelope');
      assert.ok(admitted, 'baseline report must publish an admitted envelope availability cell');
      assert.equal(
        admitted.detectedPowerSelectionCondition,
        CURRENT_DETECTED_POWER_SELECTION_CONDITION,
        'baseline admitted envelope cell must publish the automatic selection condition',
      );
      delete admitted.detectedPowerSelectionCondition;
    });
  });

  await mutationTest(t, 'rejects an envelope evidence view for a fixed-tune agile capture', /availability.*censored classifier view|classificationEvidenceView/s, async (root) => {
    await mutateReport(root, (report) => {
      const agile = report.admission.causalEnvelopeAvailabilityCells.find((cell) =>
        cell.captureProjectionKind === 'current-qualified-agile-latest-member');
      assert.ok(agile, 'baseline report must publish an agile projected capture cell');
      assert.equal(agile.classificationEvidenceView, 'spectrum-only',
        'baseline agile cell must select spectrum-only evidence');
      agile.classificationEvidenceView = 'envelope-untimed';
    });
  });

  await mutationTest(t, 'rejects a noncanonical censoring policy on an agile availability cell', /availability.*censoring policy|envelopeEvidenceCensoringPolicyId/s, async (root) => {
    await mutateReport(root, (report) => {
      const agile = report.admission.causalEnvelopeAvailabilityCells.find((cell) =>
        cell.captureProjectionKind === 'current-qualified-agile-latest-member');
      assert.ok(agile, 'baseline report must publish an agile projected capture cell');
      assert.equal(
        agile.envelopeEvidenceCensoringPolicyId,
        CURRENT_FREQUENCY_AGILE_ENVELOPE_CENSORING_POLICY_ID,
        'baseline agile cell must publish the canonical censoring policy',
      );
      agile.envelopeEvidenceCensoringPolicyId =
        'frequency-agile-fixed-tune-envelope-censoring-v0';
    });
  });

  await mutationTest(t, 'rejects a missing projection kind on a captured availability cell', /availability.*(?:projection kind|agile projection)|captureProjectionKind/s, async (root) => {
    await mutateReport(root, (report) => {
      const agile = report.admission.causalEnvelopeAvailabilityCells.find((cell) =>
        cell.captureProjectionKind === 'current-qualified-agile-latest-member');
      assert.ok(agile, 'baseline report must publish an agile projected capture cell');
      delete agile.captureProjectionKind;
    });
  });

  await mutationTest(t, 'rejects an admitted-envelope outcome on an agile availability cell', /availability.*(?:evidence disposition|direct projection|admitted envelope)|detectedPowerEvidenceDisposition/s, async (root) => {
    await mutateReport(root, (report) => {
      const agile = report.admission.causalEnvelopeAvailabilityCells.find((cell) =>
        cell.captureProjectionKind === 'current-qualified-agile-latest-member');
      assert.ok(agile, 'baseline report must publish an agile projected capture cell');
      agile.detectedPowerEvidenceDisposition = 'admitted-envelope';
    });
  });

  await mutationTest(t, 'rejects a runtime-branch clock trace violation', /causal-clock violation count|causal-clock violations/, async (root) => {
    await mutateReport(root, (report) => {
      report.matrix.runtimeBranchClockAudits.consecutiveSpectrum.violationCount = 1;
      report.admission.runtimeBranchClockAudits.consecutiveSpectrum.violationCount = 1;
    });
  });

  await mutationTest(t, 'rejects a forged validation branch clock median', /spectrumAcquisitionCount exact distribution summary/, async (root) => {
    await mutateReport(root, (report) => {
      report.matrix.runtimeBranchClockAudits.consecutiveSpectrum
        .spectrumAcquisitionCount.median = 33;
      report.admission.runtimeBranchClockAudits.consecutiveSpectrum
        .spectrumAcquisitionCount.median = 33;
    });
  });

  await mutationTest(t, 'rejects a forged tail branch capture median', /detectedPowerAcquisitionCount exact distribution summary/, async (root) => {
    await mutateReport(root, (report) => {
      report.matrix.tailCalibrationAudit.independentRecomputation
        .runtimeBranchClockAudits.qualifiedEnvelope.detectedPowerAcquisitionCount.median = 0.5;
    });
  });

  await mutationTest(t, 'rejects a changed validation paired nuisance-cell denominator', /paired nuisance-cell denominator/, async (root) => {
    await mutateReport(root, (report) => {
      report.matrix.pairedNuisanceCells -= 1;
    });
  });

  await mutationTest(t, 'rejects a changed validation branch attempt denominator', /qualifiedEnvelope attempt denominator/, async (root) => {
    await mutateReport(root, (report) => {
      report.matrix.runtimeBranchAttempts.qualifiedEnvelope -= 1;
    });
  });

  await mutationTest(t, 'rejects a changed validation observation horizon', /validation observation-opportunity horizons/, async (root) => {
    await mutateReport(root, (report) => {
      report.matrix.observationOpportunityHorizons.standard = 33;
    });
  });

  await mutationTest(t, 'rejects a changed validation detector configuration', /validation production detection configuration/, async (root) => {
    await mutateReport(root, (report) => {
      report.matrix.detectionConfig.minimumConsecutiveSweeps = 3;
    });
  });

  await mutationTest(t, 'rejects a changed direct held-out RBW matrix', /matrix\.rbwDivisors/, async (root) => {
    await mutateReport(root, (report) => {
      report.matrix.rbwDivisors[0] = 16;
    });
  });

  await mutationTest(t, 'rejects an omitted validation horizon population', /attempts-by-observation-horizon key set/, async (root) => {
    await mutateReport(root, (report) => {
      report.matrix.attemptsByObservationHorizon['32'] +=
        report.matrix.attemptsByObservationHorizon['96'];
      delete report.matrix.attemptsByObservationHorizon['96'];
    });
  });

  await mutationTest(t, 'rejects a branch-specific validation source overlap', /validation spectrum source-look overlap/, async (root) => {
    await mutateReport(root, (report) => {
      report.matrix.samplingPartitionAudit.validationFitSpectrumSourceLookIndexOverlap.push(512);
    });
  });

  await mutationTest(t, 'rejects a fitting/calibration seed overlap', /fitting\/calibration seed overlap/, async (root) => {
    await mutateReport(root, (report) => {
      report.matrix.samplingPartitionAudit.fittingCalibrationSeedOverlap.push(6_407);
    });
  });

  await mutationTest(t, 'rejects a false release-gate source-plan audit', /release-gate source-plan pins/, async (root) => {
    await mutateReport(root, (report) => {
      report.matrix.tailCalibrationAudit.pinnedReleaseGateSourcePlanValid = false;
    });
  });

  await mutationTest(t, 'rejects a failed validation attestation', /validator acceptance status/, async (root) => {
    await mutateReport(root, (report) => { report.validationAcceptance.status = 'failed'; });
  });

  await mutationTest(t, 'rejects a forged validation evidence hash', /validator acceptance evidence SHA-256/, async (root) => {
    await mutateReport(root, (report) => {
      report.validationAcceptance.evidenceSha256 = '0'.repeat(64);
    }, { seal: false });
  });

  await mutationTest(t, 'rejects a stale documentation calibration ID', /README\.md.*calibration ID/s, async (root) => {
    const path = fixturePath(root, 'README.md');
    const source = await readFile(path, 'utf8');
    assert.ok(source.includes(CURRENT_CALIBRATION_ID), 'baseline README must publish the current calibration ID');
    await writeFile(path, source.replaceAll(CURRENT_CALIBRATION_ID,
      'synthetic-independent-branch-view-matched-causal-acquisition-support-rank-detector-conditioned-physical-uncalibrated-v16'));
  });

  await mutationTest(t, 'rejects stale shorthand calibration prose', /stale shorthand calibration version/, async (root) => {
    const path = fixturePath(root, 'README.md');
    const source = await readFile(path, 'utf8');
    assert.ok(source.includes(CURRENT_CALIBRATION_ID), 'baseline README must publish the current calibration ID');
    await writeFile(path, source.replace(
      CURRENT_CALIBRATION_ID,
      `${CURRENT_CALIBRATION_ID}\n\nCalibration v12`,
    ));
  });

  await mutationTest(t, 'rejects the retained provisional pre-v19 model hash', /stale provisional pre-v19 model asset SHA-256/, async (root) => {
    const path = fixturePath(root, 'README.md');
    const source = await readFile(path, 'utf8');
    await writeFile(
      path,
      `${source}\n\nSuperseded asset: 701fdf3f5f959327369bc299dbc5a45fdf8666d40e65d57df50558b5db67c9dd.\n`,
    );
  });

  await mutationTest(t, 'rejects retained pending-v19 publication prose', /stale pending-v19 publication wording/, async (root) => {
    const path = fixturePath(root, 'README.md');
    const source = await readFile(path, 'utf8');
    await writeFile(
      path,
      `${source}\n\nValidation statement pending a fresh v19 report.\n`,
    );
  });

  await mutationTest(t, 'rejects a retained superseded-regression disclaimer', /stale superseded-regression publication wording/, async (root) => {
    const path = fixturePath(root, 'README.md');
    const source = await readFile(path, 'utf8');
    await writeFile(
      path,
      `${source}\n\nThe superseded pre-v19 development regression is retained here.\n`,
    );
  });

  await mutationTest(t, 'rejects the legacy self-attested envelope qualification', /stale self-attested detected-power acquisition qualification/, async (root) => {
    const path = fixturePath(root, 'README.md');
    const source = await readFile(path, 'utf8');
    assert.ok(
      source.includes(CURRENT_DETECTED_POWER_ACQUISITION_QUALIFICATION),
      'baseline README must publish the receipt-verified acquisition qualification',
    );
    await writeFile(path, source.replace(
      CURRENT_DETECTED_POWER_ACQUISITION_QUALIFICATION,
      'provenance-bound-first-runtime-admitted-strongest-current-single-capture-v2',
    ));
  });

  await mutationTest(t, 'rejects stale 24-opportunity publication prose', /stale 24-opportunity observation horizon/, async (root) => {
    const path = fixturePath(root, 'README.md');
    const source = await readFile(path, 'utf8');
    await writeFile(
      path,
      `${source}\n\nThe classifier uses a 24-look observation horizon.\n`,
    );
  });

  await mutationTest(t, 'rejects a changed ordered feature dimension', /ordered dimensions/, async (root) => {
    await mutateGeneratedModel(root, (model) => { model.dimensions[0] = 'forged.dimension'; });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects a changed leaf class ID', /class IDs and order/, async (root) => {
    await mutateGeneratedModel(root, (model) => { model.classModels[0].id = 'forged-class'; });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects a missing view-matched likelihood component', /spectrum-only component count|component assignments/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      model.classModels[0].componentsByView['spectrum-only'].pop();
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects a legacy single-population component field', /legacy components field absence/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      model.classModels[0].components = model.classModels[0].componentsByView['spectrum-only'];
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects an omitted likelihood-population view', /componentsByView key set|envelope-untimed componentsByView/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      delete model.classModels[0].componentsByView['envelope-untimed'];
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects a Bluetooth-like envelope likelihood component', /bluetooth-like.*envelope.*(?:component scenario identity\/order|component.*empty)|unsupported.*bluetooth-like.*envelope/s, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      const bluetooth = findBluetoothClassModel(model);
      assert.deepEqual(bluetooth.componentsByView['envelope-untimed'], [],
        'baseline Bluetooth-like envelope component population must be empty');
      assert.ok(bluetooth.componentsByView['spectrum-only'].length > 0,
        'baseline Bluetooth-like spectrum component population must be positive');
      bluetooth.componentsByView['envelope-untimed'].push(
        structuredClone(bluetooth.componentsByView['spectrum-only'][0]),
      );
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects a Bluetooth-like envelope calibration score', /bluetooth-like.*envelope.*(?:score\/count reconciliation|(?:calibration|tail).*empty)|unsupported.*bluetooth-like.*envelope/s, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      const bluetooth = findBluetoothClassModel(model);
      assert.deepEqual(bluetooth.tailCalibrationScoresByView['envelope-timed'], [],
        'baseline Bluetooth-like envelope score population must be empty');
      assert.ok(bluetooth.tailCalibrationScoresByView['spectrum-only'].length > 0,
        'baseline Bluetooth-like spectrum score population must be positive');
      bluetooth.tailCalibrationScoresByView['envelope-timed'].push(
        bluetooth.tailCalibrationScoresByView['spectrum-only'][0],
      );
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects removal of Bluetooth-like spectrum support', /bluetooth-like.*spectrum-only.*(component|calibration|support).*positive|spectrum-only component count/s, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      const bluetooth = findBluetoothClassModel(model);
      assert.ok(bluetooth.componentsByView['spectrum-only'].length > 0,
        'baseline Bluetooth-like spectrum component population must be positive');
      assert.ok(bluetooth.tailCalibrationScoresByView['spectrum-only'].length > 0,
        'baseline Bluetooth-like spectrum score population must be positive');
      bluetooth.componentsByView['spectrum-only'] = [];
      bluetooth.tailCalibrationScoresByView['spectrum-only'] = [];
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects a view-specific component dimension', /envelope-untimed.*dimensions/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      model.classModels[0].componentsByView['envelope-untimed'][0].dimensions[0] = 'forged.dimension';
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects a view-specific component identity mismatch', /component scenario identity\/order|component assignments/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      model.classModels[0].componentsByView['envelope-timed'][0].id = 'forged-scenario';
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects normalized but invalid source-owned empirical weights', /source-owned empirical weight/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      const components = model.classModels
        .map((classModel) => classModel.componentsByView['spectrum-only'])
        .find((candidate) => candidate.length >= 2);
      assert.ok(components, 'baseline model must contain a multiscenario class');
      const firstWeight = Math.exp(components[0].logWeight);
      const secondWeight = Math.exp(components[1].logWeight);
      const transferredWeight = Math.min(firstWeight, secondWeight) / 2;
      components[0].logWeight = Math.log(firstWeight + transferredWeight);
      components[1].logWeight = Math.log(secondWeight - transferredWeight);
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects a decomposed owner whose fit sample count no longer sums to its declared population', /source-owned fit sample count/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      const owner = findDecomposedOwner(model, 'envelope-timed');
      owner.components[0].fitSampleCount += 1;
      reweightSourceOwner(owner);
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects normalized but non-empirical CSMA mode weights', /source-owned empirical weight/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      const owner = findDecomposedOwner(model, 'spectrum-only');
      const firstWeight = Math.exp(owner.components[0].logWeight);
      const secondWeight = Math.exp(owner.components[1].logWeight);
      const transferredWeight = Math.min(firstWeight, secondWeight) / 2;
      owner.components[0].logWeight = Math.log(firstWeight + transferredWeight);
      owner.components[1].logWeight = Math.log(secondWeight - transferredWeight);
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects replacement of one of the exact five decomposed CSMA owners', /exact decomposed CSMA source scenario IDs/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      const owner = findDecomposedOwner(model, 'spectrum-only');
      const forgedSourceScenarioId = 'forged-sixth-csma-source';
      for (const component of owner.components) {
        component.sourceScenarioId = forgedSourceScenarioId;
        component.id = `${forgedSourceScenarioId}/${component.modeId}`;
      }
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects a decomposed CSMA mode with a non-shared covariance scale', /shared scale/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      const owner = findDecomposedOwner(model, 'envelope-untimed');
      owner.components[1].scale[0][0] *= 1.01;
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects non-increasing CSMA partition centers', /partition centers must be finite and strictly increasing/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      const owner = findDecomposedOwner(model, 'envelope-timed');
      const partitionDimensionIndex = owner.components[0].dimensions
        .indexOf(model.trainingMatrix.likelihoodComponentDecompositionPolicy.csmaPartitionFeature);
      assert.notEqual(partitionDimensionIndex, -1, 'decomposed owner must publish its partition feature');
      owner.components[1].location[partitionDimensionIndex] =
        owner.components[0].location[partitionDimensionIndex];
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects an omitted decomposed-mode fitSampleCount', /fit sample count/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      const owner = findDecomposedOwner(model, 'spectrum-only');
      delete owner.components[0].fitSampleCount;
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects a decomposed mode below the minimum fitSampleCount while preserving its owner total and empirical weights', /has only 2 fit samples/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      const owner = findDecomposedOwner(model, 'spectrum-only');
      const transferredFitSamples = owner.components[0].fitSampleCount - 2;
      assert.ok(transferredFitSamples >= 1, 'baseline decomposed modes must meet the pinned minimum');
      owner.components[0].fitSampleCount = 2;
      owner.components[1].fitSampleCount += transferredFitSamples;
      reweightSourceOwner(owner);
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects a forged reported minimum decomposed-mode fitSampleCount', /reported minimum decomposed-mode fit sample count/, async (root) => {
    await mutateReport(root, (report) => {
      report.corpus.manifestSplit.modelDeclared.minimumDecomposedModeFitSampleCount += 1;
    });
  });

  await mutationTest(t, 'rejects changed Student-t degrees of freedom', /degrees of freedom/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      model.classModels[0].componentsByView['spectrum-only'][0].degreesOfFreedom = 8;
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects an asymmetric component scale', /scale must be symmetric/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      const scale = model.classModels[0].componentsByView['spectrum-only'][0].scale;
      scale[0][1] += 0.25;
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects a non-positive-definite component scale', /scale must be positive definite/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      const scale = model.classModels[0].componentsByView['spectrum-only'][0].scale;
      scale[0][0] = 0;
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects a forged reported component assignment view', /reported\/generated envelope-timed component assignments/, async (root) => {
    await mutateReport(root, (report) => {
      report.corpus.manifestSplit.modelDeclared.componentAssignmentsByView['envelope-timed'][0]
        .classId = 'unknown-signal';
    });
  });

  await mutationTest(t, 'rejects a nonempty component-assignment view mismatch audit', /componentAssignmentViewMismatches.*length/, async (root) => {
    await mutateReport(root, (report) => {
      report.corpus.manifestSplit.componentAssignmentViewMismatches.push('envelope-timed');
    });
  });

  await mutationTest(t, 'rejects the previous likelihood-population policy', /likelihood-population policy/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      model.trainingMatrix.likelihoodPopulationPolicy =
        'independent-branch-view-matched-runtime-event-populations-v2';
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects the previous representative-selection policy', /representative selection policy|selectionPolicy/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      model.trainingMatrix.selectionPolicy =
        'independent-consecutive-spectrum-and-strongest-first-admission-qualified-envelope-branches-v6';
    });
    await mutateReport(root, (report) => {
      report.selectionPolicy =
        'independent-consecutive-spectrum-and-strongest-first-admission-qualified-envelope-branches-v6';
      report.matrix.selectionPolicy =
        'independent-consecutive-spectrum-and-strongest-first-admission-qualified-envelope-branches-v6';
    }, { seal: false });
    await rebindModelAsset(root);
  });

  const tailPolicyMutations = [
    ['tailCalibrationScoreUnit', 'pinnedScoreUnit', 'modelScoreUnit',
      'one-causal-acquisition-attempt-score-per-evidence-view-v3'],
    ['tailCalibrationRepresentativeSelectionPolicy', 'pinnedRepresentativeSelectionPolicy',
      'modelRepresentativeSelectionPolicy',
      'consecutive-spectrum-all-runtime-representatives-and-independent-qualified-envelope-sole-capture-v3'],
    ['tailCalibrationRepresentativeAggregationPolicy', 'pinnedRepresentativeAggregationPolicy',
      'modelRepresentativeAggregationPolicy', 'spectrum-minimum-envelope-sole-capture-v4'],
    ['tailCalibrationRuntimeInterpretationPolicy', 'pinnedRuntimeInterpretationPolicy',
      'modelRuntimeInterpretationPolicy',
      'spectrum-single-rank-dominates-attempt-min-envelope-rank-is-sole-capture-v2'],
  ];
  for (const [modelField, pinnedReportField, modelReportField, staleValue] of tailPolicyMutations) {
    await mutationTest(t, `rejects changed ${modelField}`, new RegExp(modelField
      + '|' + pinnedReportField + '|' + modelReportField), async (root) => {
      await mutateGeneratedModel(root, (model) => { model.trainingMatrix[modelField] = staleValue; });
      await mutateReport(root, (report) => {
        report.matrix.tailCalibrationAudit[pinnedReportField] = staleValue;
        report.matrix.tailCalibrationAudit[modelReportField] = staleValue;
      }, { seal: false });
      await rebindModelAsset(root);
    });
  }

  await mutationTest(t, 'rejects a changed representative weighting policy', /representative weighting policy|matrix\.representativeWeightingPolicy/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      model.trainingMatrix.representativeWeightingPolicy = 'equal-weight-per-causal-live-envelope-acquisition-attempt-v3';
    });
    await mutateReport(root, (report) => {
      report.matrix.representativeWeightingPolicy = 'equal-weight-per-causal-live-envelope-acquisition-attempt-v3';
    }, { seal: false });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects a changed model acquisition qualification', /detected-power acquisition qualification/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      model.trainingMatrix.detectedPowerAcquisitionQualification = 'manual-capture-v0';
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects a changed model automatic selection condition', /automatic detected-power selection condition/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      model.trainingMatrix.detectedPowerSelectionCondition =
        'operator-preferred-current-target';
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects a changed nested fitting by-view count', /fitting scenario counts\/causal-sampling spectrum-only fit-eligible count/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      const scenarioId = Object.keys(model.trainingMatrix.fittingRepresentativeCountsByScenarioByView)[0];
      assert.ok(scenarioId, 'baseline model must publish nested fitting counts');
      model.trainingMatrix.fittingRepresentativeCountsByScenarioByView[scenarioId]['spectrum-only'] += 1;
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects an omitted nested fitting view', /view-matched fitting.*key set/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      const scenarioId = Object.keys(model.trainingMatrix.fittingRepresentativeCountsByScenarioByView)[0];
      assert.ok(scenarioId, 'baseline model must publish nested fitting counts');
      delete model.trainingMatrix.fittingRepresentativeCountsByScenarioByView[scenarioId]['envelope-untimed'];
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects a changed causal fitting view total', /fitting scenario counts\/causal-sampling envelope-untimed fit-eligible count/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      model.trainingMatrix.causalSamplingAudit.fitting
        .fitEligibleRepresentativeCountsByView['envelope-untimed'] += 1;
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects a changed causal tail-calibration eligible-attempt total', /tail-calibration scenario counts\/causal-sampling envelope-timed eligible-attempt count/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      model.trainingMatrix.causalSamplingAudit.tailCalibration
        .eligibleAttemptCountsByView['envelope-timed'] += 1;
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects a changed representative eligibility policy', /representative eligibility policy/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      model.trainingMatrix.representativeEligibilityPolicy = 'runtime-domain-qualified-known-representatives-v3';
    });
    await mutateReport(root, (report) => {
      report.matrix.representativeEligibilityPolicy = 'runtime-domain-qualified-known-representatives-v3';
    }, { seal: false });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects a changed support-rank cutoff', /minimum known synthetic support rank/, async (root) => {
    await mutateReport(root, (report) => { report.model.minimumKnownSyntheticSupportRank = 0.03; });
  });

  await mutationTest(t, 'rejects duplicate prior-sensitivity variant identity', /priorSensitivity.*variant IDs/, async (root) => {
    await mutateReport(root, (report) => {
      report.priorSensitivity.variants[1].id = report.priorSensitivity.variants[0].id;
    });
  });

  await mutationTest(t, 'rejects an omitted complete-online prior-sensitivity audit', /priorSensitivity\.completeOnlineSpectrum/, async (root) => {
    await mutateReport(root, (report) => {
      delete report.priorSensitivity.completeOnlineSpectrum;
    });
  });

  await mutationTest(t, 'rejects a forged complete-online prior-sensitivity population', /completeOnlineSpectrum population/, async (root) => {
    await mutateReport(root, (report) => {
      report.priorSensitivity.completeOnlineSpectrum.population = 'causal-envelope';
    });
  });

  await mutationTest(t, 'rejects a forged complete-online prior-sensitivity denominator', /completeOnlineSpectrum complete population denominator/, async (root) => {
    await mutateReport(root, (report) => {
      report.priorSensitivity.completeOnlineSpectrum.samples -= 1;
    });
  });

  await mutationTest(t, 'rejects complete-online prior-sensitivity baseline drift', /completeOnlineSpectrum baseline decision mismatch count/, async (root) => {
    await mutateReport(root, (report) => {
      report.priorSensitivity.completeOnlineSpectrum.baselineDecisionMismatchCount = 1;
    });
  });

  await mutationTest(t, 'rejects a failed complete-online prior-sensitivity variant', /completeOnlineSpectrum variant 1 passed|prior-sensitivity gate/, async (root) => {
    await mutateReport(root, (report) => {
      report.priorSensitivity.completeOnlineSpectrum.variants[1].passed = false;
    });
  });

  await mutationTest(t, 'rejects relabeled complete-online known and unknown cases', /completeOnlineSpectrum variant 0 (known|unknown)-case denominator/, async (root) => {
    await mutateReport(root, (report) => {
      const variant = report.priorSensitivity.completeOnlineSpectrum.variants[0];
      assert.ok(variant.knownCases > 0 && variant.unknownCases > 0,
        'baseline complete-online audit must contain known and unknown cases');
      variant.knownCases -= 1;
      variant.unknownCases += 1;
      variant.falseAcceptedUnknownRisk = variant.falseAcceptedUnknownCount / variant.unknownCases;
    });
  });

  await mutationTest(t, 'rejects a tail score outside the probability interval', /tail-calibration scores must be in \[0, 1\]/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      const classModel = model.classModels.find((candidate) => candidate.id !== 'unknown-signal');
      assert.ok(classModel.tailCalibrationScoresByView['spectrum-only'].length > 0,
        'baseline model must contain spectrum tail scores');
      classModel.tailCalibrationScoresByView['spectrum-only'][0] = -0.1;
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects unsorted tail scores', /tail-calibration scores must be nondecreasing/, async (root) => {
    await mutateGeneratedModel(root, (model) => {
      const classModel = model.classModels.find((candidate) => candidate.id !== 'unknown-signal');
      const scores = classModel.tailCalibrationScoresByView['spectrum-only'];
      assert.ok(scores.length >= 2, 'baseline model must contain at least two spectrum tail scores');
      scores[0] = 1;
      scores[1] = 0;
    });
    await rebindModelAsset(root);
  });

  await mutationTest(t, 'rejects a negative tail-score difference', /tail score comparison/, async (root) => {
    await mutateReport(root, (report) => {
      report.matrix.tailCalibrationAudit.independentRecomputation.scoreComparisons[0].maximumAbsoluteDifference = -1;
    });
  });

  await mutationTest(t, 'rejects a non-Bluetooth admitted BLE result', /exclusive Bluetooth-like result|BLE result denominator/, async (root) => {
    await mutateReport(root, (report) => {
      report.classificationConditionalOnAdmission.association.byScenario['bluetooth-le-advertising']
        .results.unknown = 1;
    });
  });

  await mutationTest(t, 'rejects omitted multicomponent association coverage', /multicomponent-swept-region-activity|association-mode first-ready denominator/, async (root) => {
    await mutateReport(root, (report) => {
      delete report.classificationConditionalOnAdmission.association.byMode['multicomponent-swept-region-activity'];
    });
  });

  await mutationTest(t, 'rejects omitted complete spectrum-online association coverage', /complete spectrum-online by-mode key set|multicomponent-swept-region-activity/, async (root) => {
    await mutateReport(root, (report) => {
      delete report.classificationConditionalOnAdmission.association.completeSpectrumOnline
        .byMode['multicomponent-swept-region-activity'];
    });
  });

  await mutationTest(t, 'rejects omitted agile complete spectrum-online association coverage', /complete spectrum-online by-mode key set|frequency-agile-2g4-activity/, async (root) => {
    await mutateReport(root, (report) => {
      delete report.classificationConditionalOnAdmission.association.completeSpectrumOnline
        .byMode['frequency-agile-2g4-activity'];
    });
  });

  await mutationTest(t, 'rejects a frequency-agile summary forged into the physical envelope-target population', /physical (capture association-mode|target-mode) key set|frequency-agile-2g4-activity/, async (root) => {
    await mutateReport(root, (report) => {
      const association = report.classificationConditionalOnAdmission.association;
      association.firstReadySelectionModes['frequency-agile-2g4-activity'] = 1;
      association.soleEnvelopeTargetModes['frequency-agile-2g4-activity'] = 1;
    });
  });

  await mutationTest(t, 'rejects a frequency-agile projected capture admitted as envelope evidence', /frequency-agile-2g4-activity.*sole-envelope.*zero|soleEnvelopeByMode.*frequency-agile/s, async (root) => {
    await mutateReport(root, (report) => {
      const association = report.classificationConditionalOnAdmission.association;
      const allDetectedPower = association.byMode['frequency-agile-2g4-activity'];
      const qualifiedEnvelope = association.soleEnvelopeByMode['frequency-agile-2g4-activity'];
      assert.ok(allDetectedPower.firstReadyRepresentativeSamples > 0,
        'baseline report must publish positive agile detected-power coverage');
      assert.equal(qualifiedEnvelope.firstReadyRepresentativeSamples, 0,
        'baseline agile sole-envelope population must be censored');
      qualifiedEnvelope.firstReadyRepresentativeSamples = 1;
      qualifiedEnvelope.scenarios = ['bluetooth-le-advertising'];
    });
  });

  await mutationTest(t, 'rejects zero fixed-tune censoring in the agile projected-mode outcome audit', /frequency-agile-2g4-activity.*censored.*positive|byProjectedMode.*censored/s, async (root) => {
    await mutateReport(root, (report) => {
      const agile = report.admission.detectedPowerCaptureOutcomes
        .byProjectedMode['frequency-agile-2g4-activity'];
      assert.ok(agile.censoredDetectedPowerCaptureCount > 0,
        'baseline report must publish positive agile capture censoring');
      agile.censoredDetectedPowerCaptureCount = 0;
    });
  });

  await mutationTest(t, 'rejects zero captures in the fixed-tune censoring summary', /frequency-agile.*(?:physical captures censored|censored physical captures)|physicalCapturesCensored/s, async (root) => {
    await mutateReport(root, (report) => {
      const censoring = report.admission.frequencyAgileEnvelopeCensoring;
      assert.ok(censoring.physicalCapturesCensored > 0,
        'baseline report must publish positive fixed-tune censored captures');
      censoring.physicalCapturesCensored = 0;
    });
  });

  await mutationTest(t, 'rejects a forged frequency-agile production censoring limitation', /production censoring limitation/, async (root) => {
    await mutateReport(root, (report) => {
      const censoring = report.admission.frequencyAgileEnvelopeCensoring;
      assert.equal(
        censoring.limitation,
        CURRENT_FREQUENCY_AGILE_ENVELOPE_CENSORING_LIMITATION,
        'baseline report must publish the production censoring limitation',
      );
      censoring.limitation =
        'frequency-agile-fixed-tune-envelope-censored-to-spectrum-only';
    });
  });

  await mutationTest(t, 'rejects an unreconciled frequency-agile production censoring limitation count', /censoring limitation\/classification reconciliation/, async (root) => {
    await mutateReport(root, (report) => {
      const limitations = report.classificationConditionalOnAdmission.limitations;
      assert.ok(
        limitations[CURRENT_FREQUENCY_AGILE_ENVELOPE_CENSORING_LIMITATION] > 0,
        'baseline report must count the production censoring limitation',
      );
      limitations[CURRENT_FREQUENCY_AGILE_ENVELOPE_CENSORING_LIMITATION] -= 1;
    });
  });

  await mutationTest(t, 'rejects a missing censored detected-power evidence disposition', /detected-power evidence disposition.*censored-frequency-agile-spectrum-only|detectedPowerEvidenceDispositions/s, async (root) => {
    await mutateReport(root, (report) => {
      const dispositions = report.classificationConditionalOnAdmission.association
        .detectedPowerEvidenceDispositions;
      assert.ok(dispositions['censored-frequency-agile-spectrum-only'] > 0,
        'baseline report must publish censored agile evidence dispositions');
      dispositions['censored-frequency-agile-spectrum-only'] = 0;
    });
  });

  await mutationTest(t, 'rejects a zero Bluetooth scenario censor count', /bluetooth-le-advertising.*censored.*physical|Bluetooth.*scenario.*censor/s, async (root) => {
    await mutateReport(root, (report) => {
      const bluetooth = report.admission.detectedPowerCaptureOutcomes
        .byScenario['bluetooth-le-advertising'];
      assert.ok(bluetooth.censoredDetectedPowerCaptureCount > 0,
        'baseline BLE scenario must publish censored fixed-tune captures');
      bluetooth.censoredDetectedPowerCaptureCount = 0;
    });
  });

  await mutationTest(t, 'rejects a projected-kind detected-power outcome subtotal mismatch', /byProjectionKind.*physical.*reconciliation|projection-kind.*outcome.*(total|denominator)/s, async (root) => {
    await mutateReport(root, (report) => {
      report.admission.detectedPowerCaptureOutcomes
        .byProjectionKind['current-qualified-agile-latest-member']
        .physicalCaptureCount -= 1;
    });
  });

  await mutationTest(t, 'rejects a selected evidence-view subtotal mismatch', /selected evidence-view.*(total|denominator)|selectedEvidenceViews/s, async (root) => {
    await mutateReport(root, (report) => {
      const selectedViews = report.admission.detectedPowerCaptureOutcomes
        .selectedEvidenceViews;
      assert.ok(selectedViews['spectrum-only'] > 0,
        'baseline detected-power outcomes must select spectrum-only evidence');
      selectedViews['spectrum-only'] -= 1;
    });
  });

  await mutationTest(t, 'rejects omitted agile capture-projection attribution', /capture projection-kind population.*current-qualified-agile-latest-member/, async (root) => {
    await mutateReport(root, (report) => {
      delete report.classificationConditionalOnAdmission.association
        .captureProjectionKinds['current-qualified-agile-latest-member'];
    });
  });

  await mutationTest(t, 'rejects zero candidate-state raw capture attribution', /raw capture-target state population candidate.*positive/, async (root) => {
    await mutateReport(root, (report) => {
      report.classificationConditionalOnAdmission.association
        .rawCaptureTargetStates.candidate = 0;
    });
  });

  await mutationTest(t, 'rejects a forged sole-envelope association population', /sole-envelope\/first-ready association-mode population|complete sole-envelope association-mode denominator/, async (root) => {
    await mutateReport(root, (report) => {
      report.classificationConditionalOnAdmission.association
        .soleEnvelopeTargetModes['frequency-local'] += 1;
    });
  });

  await mutationTest(t, 'rejects an out-of-range published probability', /knownCoverage.*0\.\.1/, async (root) => {
    await mutateReport(root, (report) => {
      report.classificationConditionalOnAdmission.knownCoverage = 1.01;
    });
  });

  for (const mutation of [
    {
      name: 'hierarchical accuracy',
      field: 'hierarchicalAccuracy',
      value: 0.94,
      expected: /admission-conditional hierarchical accuracy.*0\.95\.\.1/,
    },
    {
      name: 'known top-leaf accuracy',
      field: 'knownTopLeafAccuracy',
      value: 0.84,
      expected: /admission-conditional known top-leaf accuracy.*0\.85\.\.1/,
    },
    {
      name: 'known coverage',
      field: 'knownCoverage',
      value: 0.94,
      expected: /admission-conditional known coverage.*0\.95\.\.1/,
    },
    {
      name: 'minimum high-SNR known-class accuracy',
      field: 'minimumHighSnrKnownClassHierarchicalAccuracy',
      value: 0.89,
      expected: /minimum high-SNR known-class hierarchical accuracy.*0\.9\.\.1/,
    },
    {
      name: 'fitted-unknown AUROC',
      field: 'fittedUnknownPosteriorAuroc',
      value: 0.89,
      expected: /fitted-unknown posterior AUROC.*0\.9\.\.1/,
    },
    {
      name: 'strict typicality AUROC',
      field: 'scenarioExcludedStrictTypicalityAuroc',
      value: 0.89,
      expected: /strict scenario-excluded typicality AUROC.*0\.9\.\.1/,
    },
    {
      name: 'strict holdout rejection',
      field: 'scenarioExcludedStrictUnknownRejectionRate',
      value: 0.99,
      expected: /strict unknown holdout rejection acceptance gate/,
    },
    {
      name: 'exact-equivalence compatibility',
      field: 'exactEquivalenceCompatibleRate',
      value: 0.99,
      expected: /exact observable-equivalence compatibility acceptance gate/,
    },
  ]) {
    await mutationTest(t, `rejects below-policy ${mutation.name}`, mutation.expected, async (root) => {
      await mutateReport(root, (report) => {
        report.classificationConditionalOnAdmission[mutation.field] =
          mutation.value;
      });
    });
  }

  for (const mutation of [
    {
      name: 'log loss',
      field: 'fittedTemplateLogLoss',
      value: 0.51,
      expected: /fitted-template log loss exceeds the acceptance gate 0\.5/,
    },
    {
      name: 'multiclass Brier score',
      field: 'fittedTemplateMulticlassBrier',
      value: 0.21,
      expected: /fitted-template multiclass Brier score exceeds the acceptance gate 0\.2/,
    },
    {
      name: 'expected calibration error',
      field: 'fittedTemplateExpectedCalibrationError',
      value: 0.11,
      expected: /fitted-template expected calibration error exceeds the acceptance gate 0\.1/,
    },
  ]) {
    await mutationTest(t, `rejects above-policy fitted-template ${mutation.name}`, mutation.expected, async (root) => {
      await mutateReport(root, (report) => {
        report.classificationConditionalOnAdmission[mutation.field] =
          mutation.value;
      });
    });
  }

  for (const mutation of [
    { name: 'null', field: 'fittedTemplateLogLoss', value: null },
    { name: 'NaN', field: 'fittedTemplateMulticlassBrier', value: Number.NaN },
    {
      name: 'infinity',
      field: 'fittedTemplateExpectedCalibrationError',
      value: Number.POSITIVE_INFINITY,
    },
  ]) {
    await mutationTest(t, `rejects ${mutation.name} evidence-view proper-score input`, /spectrum-only fitted-template.*expected 0\.\./, async (root) => {
      await mutateReport(root, (report) => {
        report.classificationConditionalOnAdmission.evidenceViews['spectrum-only']
          [mutation.field] = mutation.value;
      });
    });
  }

  await mutationTest(t, 'rejects lowered rolling-window acceptance thresholds', /rolling-window pinned acceptance thresholds/, async (root) => {
    await mutateReport(root, (report) => {
      report.productionRollingWindowValidation.acceptanceThresholds
        .overallKnownCoverage = 0;
    });
  });

  await mutationTest(t, 'rejects a failed ordinary-known high-SNR seed cell', /high-SNR seed coverage cw-rbw-line\/24 acceptance/, async (root) => {
    await mutateReport(root, (report) => {
      report.admission.highSnrUniqueSeedCoverage.byKnownScenario['cw-rbw-line']
        .bySnr['24'].passed = false;
    });
  });

  await mutationTest(t, 'rejects a weakened ordinary-known seed-coverage policy', /ordinary known high-SNR seed-coverage policy/, async (root) => {
    await mutateReport(root, (report) => {
      report.admission.highSnrUniqueSeedCoverage
        .ordinaryKnownRequiredCoverage = 0;
    });
  });

  await mutationTest(t, 'rejects an empty top-level proper-score population', /singleton-truth proper-score population must be a positive integer/, async (root) => {
    await mutateReport(root, (report) => {
      report.classificationConditionalOnAdmission
        .singletonAllowedTruthProperScoreSamples = 0;
    });
  });

  await mutationTest(t, 'rejects a changed expected non-admission policy', /expected classification non-admission scenario policy/, async (root) => {
    await mutateReport(root, (report) => {
      report.admission.expectedClassificationNonAdmissionScenarios = [];
    });
  });

  await mutationTest(t, 'rejects a duplicate in the complete online spectrum denominator', /complete online spectrum unique case denominator/, async (root) => {
    await mutateReport(root, (report) => {
      report.productionRollingWindowValidation.completeOnlineSpectrumAudit.uniqueCases -= 1;
    });
  });

  await mutationTest(t, 'rejects a complete-online unknown-truth false accept', /complete online spectrum unknown-truth false accepts/, async (root) => {
    await mutateReport(root, (report) => {
      report.productionRollingWindowValidation.completeOnlineSpectrumAudit
        .unknownTruthFalseAcceptCount = 1;
    });
  });

  await mutationTest(t, 'rejects a failed complete-online proper-score gate', /complete online spectrum fitted-template ECE/, async (root) => {
    await mutateReport(root, (report) => {
      report.productionRollingWindowValidation.completeOnlineSpectrumAudit
        .fittedTemplateExpectedCalibrationError = 0.11;
    });
  });

  await mutationTest(t, 'rejects an empty evidence-view proper-score population', /must publish positive exact-equivalence, strict-holdout, and proper-score populations/, async (root) => {
    await mutateReport(root, (report) => {
      report.classificationConditionalOnAdmission.evidenceViews['spectrum-only']
        .singletonAllowedTruthProperScoreSamples = 0;
    });
  });

  await mutationTest(t, 'rejects an acquisition-policy-unqualified envelope limitation', /zero-span-acquisition-policy-unqualified/, async (root) => {
    await mutateReport(root, (report) => {
      report.classificationConditionalOnAdmission.limitations
        ['zero-span-acquisition-policy-unqualified'] = 1;
    });
  });

  await mutationTest(t, 'rejects omitted exact-equivalence online-spectrum pairs', /online-spectrum pair count must be positive/, async (root) => {
    await mutateReport(root, (report) => {
      for (const pair of report.corpus.exactObservableEquivalencePairAudit.pairs) {
        pair.matchedOnlineSpectrumPairs = 0;
      }
    });
  });

  await mutationTest(t, 'does not accept a model hash hidden in an HTML comment', /README\.md must publish model asset SHA-256/, async (root) => {
    const report = JSON.parse(await readFile(resolve(root, REPORT_PATH), 'utf8'));
    const path = fixturePath(root, 'README.md');
    const source = await readFile(path, 'utf8');
    assert.ok(source.includes(report.model.modelAssetSha256), 'baseline README must publish the model hash');
    await writeFile(path, source.replace(report.model.modelAssetSha256, `<!--${report.model.modelAssetSha256}-->`));
  });

  await mutationTest(t, 'does not accept required prose hidden in a fenced block', /production-acquisition-summary/, async (root) => {
    const path = fixturePath(root, 'README.md');
    const source = await readFile(path, 'utf8');
    const startMarker = 'The fitted and independently regenerated acquisition matrix uses SignalLab';
    const endMarker = 'measured detected-power RBW remains unavailable and is never classifier evidence.';
    const start = source.indexOf(startMarker);
    const end = source.indexOf(endMarker, start) + endMarker.length;
    assert.ok(start >= 0 && end >= start + endMarker.length,
      'baseline README must contain one complete production-acquisition summary');
    const canonicalRequired = source.slice(start, end);
    assert.ok(source.includes(canonicalRequired), 'baseline README must publish production acquisition prose');
    await writeFile(path, source.replace(canonicalRequired, `\n\`\`\`text\n${canonicalRequired}\n\`\`\`\n`));
  });

  await mutationTest(t, 'rejects a missing exact-view production boundary', /exact-view-contract-summary/, async (root) => {
    const path = fixturePath(root, 'README.md');
    const source = await readFile(path, 'utf8');
    const required = 'Production inference does not use missing-dimension marginalization: v9 selects one exact evidence view, requires its complete finite feature set with no extras, and evaluates only the independently fitted spectrum-only, envelope-untimed, or envelope-timed likelihood population.';
    assert.ok(source.includes(required), 'baseline README must publish the exact-view production boundary');
    await writeFile(path, source.replace(required, 'Production inference marginalizes whichever dimensions happen to be missing.'));
  });

  await mutationTest(t, 'rejects a missing manual-capture boundary', /manual-capture-boundary-summary/, async (root) => {
    const path = fixturePath(root, 'README.md');
    const source = await readFile(path, 'utf8');
    const required = 'The App zero-span action enters a Bayesian envelope view only when the capture is bound to an analysis-issued receipt for a current runtime-admitted target, exact admitted tune, and exact eight-sweep evidence window. Receipt qualification is necessary but not sufficient: under frequency-agile-fixed-tune-envelope-censoring-v1, every fixed-tune frequency-agile capture remains excluded from Bayesian envelope inference and the exact spectrum view is used instead. Any other receipt-free or runtime-unadmitted capture may feed only the separate envelope heuristic.';
    assert.ok(source.includes(required), 'baseline README must publish the manual-capture boundary');
    await writeFile(path, source.replace(required, 'Manual captures are classified as calibrated Bayesian envelope evidence.'));
  });

  await mutationTest(t, 'rejects a changed likelihood architecture count in visible prose', /model-structure-summary/, async (root) => {
    const path = fixturePath(root, 'README.md');
    const source = await readFile(path, 'utf8');
    const required = 'Its spectrum-only population has 18 source scenarios and 28 likelihood components; each envelope population has 16 scenarios and 26 components because the Bluetooth-like class is structurally unsupported for fixed-tune envelope evidence.';
    assert.ok(source.includes(required), 'baseline README must publish the pinned likelihood architecture counts');
    await writeFile(path, source.replace(required, required.replace(
      'each envelope population has 16 scenarios and 26 components',
      'each envelope population has 17 scenarios and 27 components',
    )));
  });
});

async function mutationTest(t, name, expectedFailure, mutate) {
  await t.test(name, async () => {
    await withFixture(async (root) => {
      await mutate(root);
      await assertVerifierRejects(root, expectedFailure);
    });
  });
}

async function withFixture(action) {
  const fixtureParent = await mkdtemp(join(tmpdir(), 'atom-classifier-publication-'));
  const root = resolve(fixtureParent, 'Atom-Classifier');
  const atomizerRoot = resolve(fixtureParent, 'Atom-Atomizer');
  try {
    for (const relativePath of CLASSIFIER_FIXTURE_PATHS) {
      const destination = resolve(root, relativePath);
      await mkdir(dirname(destination), { recursive: true });
      await copyFile(resolve(REPOSITORY_ROOT, relativePath), destination);
    }
    for (const relativePath of PUBLICATION_PATHS) {
      const destination = resolve(atomizerRoot, relativePath);
      await mkdir(dirname(destination), { recursive: true });
      await copyFile(resolve(REPOSITORY_ROOT, '../Atom-Atomizer', relativePath), destination);
    }
    return await action(root);
  } finally {
    await rm(fixtureParent, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 });
  }
}

function fixturePath(root, relativePath) {
  return resolve(root, PUBLICATION_PATHS.includes(relativePath) ? '../Atom-Atomizer' : '.', relativePath);
}

async function runVerifier(root) {
  try {
    const result = await execFileAsync(process.execPath, [resolve(root, VERIFIER_PATH)], {
      cwd: root,
      encoding: 'utf8',
      maxBuffer: 16 * 1024 * 1024,
      timeout: 10_000,
    });
    return { passed: true, stdout: result.stdout, stderr: result.stderr };
  } catch (error) {
    return {
      passed: false,
      stdout: typeof error.stdout === 'string' ? error.stdout : '',
      stderr: typeof error.stderr === 'string' ? error.stderr : String(error),
    };
  }
}

async function assertVerifierPasses(root) {
  const result = await runVerifier(root);
  assert.equal(result.passed, true, `unmodified publication fixture failed:\n${result.stdout}\n${result.stderr}`);
}

async function assertVerifierRejects(root, expectedFailure) {
  const result = await runVerifier(root);
  assert.equal(result.passed, false, `mutated publication fixture unexpectedly passed:\n${result.stdout}`);
  assert.match(`${result.stdout}\n${result.stderr}`, expectedFailure);
}

async function mutateGeneratedModel(root, mutate) {
  const path = resolve(root, MODEL_PATH);
  const source = await readFile(path, 'utf8');
  const match = source.match(MODEL_PAYLOAD_PATTERN);
  assert.ok(match, 'generated model fixture must contain one JSON payload');
  const model = JSON.parse(match[1]);
  mutate(model);
  await writeFile(path, source.replace(match[1], JSON.stringify(model, null, 2)));
}

async function readGeneratedModel(root) {
  const source = await readFile(resolve(root, MODEL_PATH), 'utf8');
  const match = source.match(MODEL_PAYLOAD_PATTERN);
  assert.ok(match, 'generated model fixture must contain one JSON payload');
  return JSON.parse(match[1]);
}

function findBluetoothClassModel(model) {
  const bluetooth = model.classModels.find((classModel) =>
    classModel.id === 'bluetooth-like');
  assert.ok(bluetooth, 'baseline model must contain the Bluetooth-like class');
  return bluetooth;
}

function findDecomposedOwner(model, view) {
  for (const classModel of model.classModels) {
    const components = classModel.componentsByView?.[view];
    if (!Array.isArray(components)) continue;
    const componentsBySourceScenario = new Map();
    for (const component of components) {
      const sourceScenarioId = component.sourceScenarioId ?? component.id;
      const owned = componentsBySourceScenario.get(sourceScenarioId) ?? [];
      owned.push(component);
      componentsBySourceScenario.set(sourceScenarioId, owned);
    }
    for (const [sourceScenarioId, owned] of componentsBySourceScenario) {
      if (owned.length === 3
        && owned.every((component) =>
          component.sourceScenarioId === sourceScenarioId
          && /^csma-activity-mode-[1-3]-of-3$/.test(component.modeId))) {
        return {
          components: owned,
          sourceScenarioCount: componentsBySourceScenario.size,
        };
      }
    }
  }
  assert.fail(`baseline model must contain a three-mode decomposed owner for ${view}`);
}

function reweightSourceOwner(owner) {
  const fitSampleCount = owner.components.reduce(
    (sum, component) => sum + component.fitSampleCount,
    0,
  );
  assert.ok(Number.isSafeInteger(fitSampleCount) && fitSampleCount > 0,
    'decomposed owner must have a positive integral fit population');
  for (const component of owner.components) {
    component.logWeight = Math.log(
      (1 / owner.sourceScenarioCount) * (component.fitSampleCount / fitSampleCount),
    );
  }
}

async function mutateReport(root, mutate, { seal = true } = {}) {
  const path = resolve(root, REPORT_PATH);
  const report = JSON.parse(await readFile(path, 'utf8'));
  mutate(report);
  if (seal) sealValidationEvidence(report);
  await writeFile(path, `${JSON.stringify(report, null, 2)}\n`);
}

async function mutateProductionSchedule(root, branch, scheduleIndex, mutate) {
  const scheduleField = branch === 'spectrum'
    ? 'spectrumTemporalSchedule'
    : 'qualifiedEnvelopeTemporalSchedule';
  await mutateGeneratedModel(root, (model) => {
    mutate(model.trainingMatrix.signalLabProductionAcquisitionRegime
      .temporalSchedulePairs[scheduleIndex][scheduleField]);
  });
  await mutateReport(root, (report) => {
    mutate(report.matrix.tailCalibrationAudit.pinnedSignalLabProductionAcquisitionRegime
      .temporalSchedulePairs[scheduleIndex][scheduleField]);
  }, { seal: false });
  await rebindModelAsset(root);
}

async function mutateProductionSourcePlan(root, branch, profileIndex, mutate) {
  const planField = branch === 'spectrum'
    ? 'spectrumReleaseGateSourcePlan'
    : 'qualifiedEnvelopeReleaseGateSourcePlan';
  await mutateGeneratedModel(root, (model) => {
    mutate(model.trainingMatrix.signalLabProductionAcquisitionRegime
      [planField][profileIndex]);
  });
  await mutateReport(root, (report) => {
    mutate(report.matrix.tailCalibrationAudit.pinnedSignalLabProductionAcquisitionRegime
      [planField][profileIndex]);
  }, { seal: false });
  await rebindModelAsset(root);
}

async function mutateProductionSourceClock(root, branch, mutate) {
  const reportBranch = branch === 'spectrum' ? 'consecutiveSpectrum' : 'qualifiedEnvelope';
  await mutateGeneratedModel(root, (model) => {
    mutate(model.trainingMatrix.signalLabProductionAcquisitionRegime.sourceClocks[branch]);
  });
  await mutateReport(root, (report) => {
    mutate(report.matrix.sourceClocks[branch]);
    mutate(report.matrix.runtimeBranchClockAudits[reportBranch].sourceClock);
    mutate(report.admission.runtimeBranchClockAudits[reportBranch].sourceClock);
    mutate(report.matrix.tailCalibrationAudit.pinnedSignalLabProductionAcquisitionRegime
      .sourceClocks[branch]);
    mutate(report.matrix.tailCalibrationAudit.independentRecomputation
      .runtimeBranchClockAudits[reportBranch].sourceClock);
  }, { seal: false });
  await rebindModelAsset(root);
}

async function rebindModelAsset(root) {
  const modelPath = resolve(root, MODEL_PATH);
  let modelSource = await readFile(modelPath, 'utf8');
  const modelPayloadMatch = modelSource.match(MODEL_PAYLOAD_PATTERN);
  assert.ok(modelPayloadMatch, 'generated model fixture must contain one JSON payload');
  const modelContentSha256 = sha256(JSON.stringify(JSON.parse(modelPayloadMatch[1])));
  const modelContentMatch = modelSource.match(MODEL_CONTENT_PATTERN);
  assert.ok(modelContentMatch, 'generated model fixture must contain one content SHA-256');
  modelSource = modelSource.replace(
    MODEL_CONTENT_PATTERN,
    (_match, prefix, _previous, suffix) => `${prefix}${modelContentSha256}${suffix}`,
  );
  await writeFile(modelPath, modelSource);
  const modelSha256 = sha256(modelSource);
  const manifestPath = resolve(root, MANIFEST_PATH);
  let manifest = await readFile(manifestPath, 'utf8');
  const manifestMatch = manifest.match(MODEL_MANIFEST_PATTERN);
  assert.ok(manifestMatch, 'generated model manifest fixture must contain one SHA-256');
  const previousManifestSha256 = manifestMatch[2];
  manifest = manifest.replace(
    MODEL_MANIFEST_PATTERN,
    (_match, prefix, _previous, suffix) => `${prefix}${modelSha256}${suffix}`,
  );
  const manifestContentMatch = manifest.match(MODEL_CONTENT_PATTERN);
  assert.ok(
    manifestContentMatch,
    'generated model manifest fixture must contain one content SHA-256',
  );
  manifest = manifest.replace(
    MODEL_CONTENT_PATTERN,
    (_match, prefix, _previous, suffix) => `${prefix}${modelContentSha256}${suffix}`,
  );
  await writeFile(manifestPath, manifest);

  const reportPath = resolve(root, REPORT_PATH);
  const report = JSON.parse(await readFile(reportPath, 'utf8'));
  const previousReportSha256 = report.model.modelAssetSha256;
  report.model.modelAssetSha256 = modelSha256;
  report.integrity.checkedInModelAssetSha256 = modelSha256;
  report.integrity.modelAssetManifestSha256 = modelSha256;
  report.validationAcceptance.modelAssetSha256 = modelSha256;
  sealValidationEvidence(report);
  await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`);

  for (const relativePath of PUBLICATION_PATHS) {
    const path = fixturePath(root, relativePath);
    let source = await readFile(path, 'utf8');
    for (const previous of new Set([previousManifestSha256, previousReportSha256])) {
      source = source.replaceAll(previous, modelSha256);
    }
    await writeFile(path, source);
  }
}

function sealValidationEvidence(report) {
  const { validationAcceptance, ...evidence } = report;
  assert.ok(validationAcceptance && typeof validationAcceptance === 'object',
    'report fixture must contain validationAcceptance');
  validationAcceptance.evidenceSha256 = sha256(JSON.stringify(evidence));
}

function formatInteger(value) {
  assert.ok(Number.isInteger(value), 'fixture integer must be exact');
  return String(value).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

function sha256(value) {
  return createHash('sha256').update(value).digest('hex');
}
