import { createHash } from 'node:crypto';
import { readFileSync, statSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  createTimeDomainDualFusionClassifierV3,
  TIME_DOMAIN_DUAL_FUSION_CANDIDATE_ID,
  TIME_DOMAIN_DUAL_FUSION_CLASSIFIER_ASSET,
  TIME_DOMAIN_DUAL_FUSION_OPENSET_ASSET,
  TIME_DOMAIN_DUAL_FUSION_REJECTOR_ASSET,
  type TimeDomainDualFusionDecisionV3,
} from './time-domain-dual-fusion-classifier-v3.js';
import { StageTwoRejectorV3 } from './time-domain-openset-v3.js';

const PACKAGE_SCHEMA =
  'atomos.v3.time-domain-classifier.dual-runtime-package';
const PARITY_SCHEMA = 'time-domain-openset-parity-v1';
const STAGING_STATUS = 'staging_not_release';
const BINDING_ASSET = 'time-domain-v3-dual-binding.json';
// LOF is a nearest-neighbour density ratio and can amplify a bounded 1e-5
// embedding perturbation. The exact-Python-embedding replay below remains at
// the fixture's strict 1e-9 bound; this is only the composed TS-encoder path.
const E2E_LOF_RAW_RELATIVE = 1e-3;

interface AssetRecord {
  path: string;
  bytes: number;
  sha256: string;
  schema: string;
  schema_version: number;
  status: string;
  runtime_role?: string;
}

interface RuntimePackageManifest {
  schema: string;
  schema_version: number;
  status: string;
  candidate_id: string;
  architecture: {
    execution_order: string[];
    classifier_runs_only_after_rejector_acceptance: boolean;
    public_known_label_from_classifier_only: boolean;
  };
  assets: Record<string, AssetRecord>;
  external_evidence: {
    parity: AssetRecord & { packaged: boolean };
  };
}

interface ExpectedStageOne {
  capture_length: number;
  evaluated_length: number;
  causal_prefix_applied: boolean;
  features: number[];
  score: number;
  rank: number;
  gated: boolean;
  threshold_score: number;
}

interface ExpectedLof {
  branch: 'real' | 'complex';
  weight: number;
  neighbors: number;
  raw: number;
  rank: number;
}

interface ExpectedFusion {
  runtime_role: string;
  real_embedding: number[];
  complex_embedding: number[];
  fused_embedding: number[];
  predicted_class_index: number;
  predicted_class_label: string;
  squared_prototype_distances: number[];
}

interface ExpectedStageTwo extends ExpectedFusion {
  lof: ExpectedLof[];
  lof_rank: number;
  geometry_statistic: number;
  geometry_deviation: number;
  geometry_rank: number;
  combined_raw: number;
  score: number;
  threshold: number;
}

interface ExpectedPublicResult {
  outcome: 'noise' | 'unknown' | 'known';
  label: string;
  knownLabel: string | null;
  knownWinnerIndex: number | null;
  squaredPrototypeDistances: number[] | null;
}

interface ParityRow {
  family: string;
  name: string;
  capture_length: number;
  iq: { in_phase: number[]; quadrature: number[] };
  stage_one: ExpectedStageOne;
  packed_iq: { in_phase: number[]; quadrature: number[] };
  raw_features: number[];
  rejector_standardized_features: number[];
  classifier_standardized_features: number[] | null;
  stage_two: ExpectedStageTwo | null;
  rejector_closed_winner_index: number | null;
  rejector_closed_winner_label: string | null;
  classifier: ExpectedFusion | null;
  classifier_executed: boolean;
  stage_one_survivor_rank: number | null;
  composite_score: number | null;
  staged_threshold: number;
  staged_score: number;
  rejected_stage: 1 | 2 | null;
  decision_label: string;
  public_result: ExpectedPublicResult;
}

interface ParityFixture {
  schema: string;
  schema_version: number;
  status: string;
  candidate_id: string;
  counts: {
    rows: number;
    gated_stage_one: number;
    rejected_stage_two: number;
    accepted: number;
    classifier_executed: number;
    accepted_role_winner_disagreements: number;
  };
  accepted_role_winner_disagreement_rows: string[];
  role_contract: {
    rejector: string;
    classifier: string;
    classifier_runs_only_after_rejector_acceptance: boolean;
    public_known_label_from_classifier_only: boolean;
  };
  tolerance: {
    stage_one_features_abs: number;
    stage_one_score_abs: number;
    embedding_abs: number;
    rank_abs: number;
    raw_lof_relative: number;
    decisions: string;
  };
  rows: ParityRow[];
}

const packageRoot = new URL('./assets-v3-dual-staging/', import.meta.url);

function readBytes(path: URL): Buffer {
  return readFileSync(path);
}

function digest(bytes: Buffer): string {
  return createHash('sha256').update(bytes).digest('hex');
}

function parseJson(bytes: Buffer): unknown {
  return JSON.parse(bytes.toString('utf8')) as unknown;
}

const packageManifest = parseJson(
  readBytes(new URL('runtime-package-manifest.json', packageRoot)),
) as RuntimePackageManifest;

const assetNames = [
  TIME_DOMAIN_DUAL_FUSION_REJECTOR_ASSET,
  TIME_DOMAIN_DUAL_FUSION_CLASSIFIER_ASSET,
  TIME_DOMAIN_DUAL_FUSION_OPENSET_ASSET,
  BINDING_ASSET,
] as const;

const assetBytes = Object.fromEntries(
  assetNames.map((name) => [name, readBytes(new URL(name, packageRoot))]),
) as Record<(typeof assetNames)[number], Buffer>;
const assetDigests = Object.fromEntries(
  assetNames.map((name) => [name, digest(assetBytes[name])]),
) as Record<(typeof assetNames)[number], string>;

const parityRecord = packageManifest.external_evidence.parity;
const parityUrl = new URL(parityRecord.path, packageRoot);
const parityBytes = readBytes(parityUrl);
const parityDigest = digest(parityBytes);
const fixture = parseJson(parityBytes) as ParityFixture;

const classifier = createTimeDomainDualFusionClassifierV3({
  bindingAsset: parseJson(assetBytes[BINDING_ASSET]),
  rejectorFusion: {
    asset: parseJson(
      assetBytes[TIME_DOMAIN_DUAL_FUSION_REJECTOR_ASSET],
    ),
    preverifiedAssetSha256:
      assetDigests[TIME_DOMAIN_DUAL_FUSION_REJECTOR_ASSET],
  },
  classifierFusion: {
    asset: parseJson(
      assetBytes[TIME_DOMAIN_DUAL_FUSION_CLASSIFIER_ASSET],
    ),
    preverifiedAssetSha256:
      assetDigests[TIME_DOMAIN_DUAL_FUSION_CLASSIFIER_ASSET],
  },
  opensetPolicy: {
    asset: parseJson(
      assetBytes[TIME_DOMAIN_DUAL_FUSION_OPENSET_ASSET],
    ),
    preverifiedAssetSha256:
      assetDigests[TIME_DOMAIN_DUAL_FUSION_OPENSET_ASSET],
  },
  admission: 'staging',
});

function expectScalarWithin(
  actual: number,
  expected: number,
  tolerance: number,
  path: string,
): void {
  expect(
    Math.abs(actual - expected),
    `${path}: ${actual} != ${expected}`,
  ).toBeLessThanOrEqual(tolerance);
}

function expectVectorWithin(
  actual: ArrayLike<number>,
  expected: readonly number[],
  tolerance: number,
  path: string,
): void {
  expect(actual.length, `${path}.length`).toBe(expected.length);
  let worst = 0;
  for (let index = 0; index < expected.length; index++) {
    worst = Math.max(
      worst,
      Math.abs(actual[index]! - expected[index]!),
    );
  }
  expect(worst, `${path} max absolute error`).toBeLessThanOrEqual(tolerance);
}

function expectRelativeWithin(
  actual: number,
  expected: number,
  tolerance: number,
  path: string,
): void {
  const relative = Math.abs(actual - expected) / Math.max(
    Math.abs(expected),
    Number.EPSILON,
  );
  expect(relative, `${path} relative error`).toBeLessThanOrEqual(tolerance);
}

/**
 * A tiny float perturbation can cross a repeated empirical-calibration value
 * and move a rank by a whole block. Bound that rank movement by exactly the
 * entries in the proven numeric-error window, as the staged runtime does.
 */
function rankToleranceAround(
  sorted: readonly number[],
  value: number,
  radius: number,
): number {
  let low = 0;
  let high = sorted.length;
  while (low < high) {
    const middle = low + Math.floor((high - low) / 2);
    if (sorted[middle]! < value - radius) low = middle + 1;
    else high = middle;
  }
  const first = low;
  low = first;
  high = sorted.length;
  while (low < high) {
    const middle = low + Math.floor((high - low) / 2);
    if (sorted[middle]! <= value + radius) low = middle + 1;
    else high = middle;
  }
  return (low - first + 1) / (sorted.length + 1);
}

const referenceStageTwo = new StageTwoRejectorV3(
  classifier.openSet.asset.stage_two,
  {
    patchCount: classifier.openSet.asset.frontend.patch_count,
    patchLength: classifier.openSet.asset.frontend.patch_length,
  },
);

function expectPublicResult(
  actual: TimeDomainDualFusionDecisionV3,
  expected: ExpectedPublicResult,
  row: string,
): void {
  expect(actual.outcome, `${row}.outcome`).toBe(expected.outcome);
  expect(actual.label, `${row}.label`).toBe(expected.label);
  expect(actual.knownLabel, `${row}.knownLabel`).toBe(expected.knownLabel);
  expect(actual.knownWinnerIndex, `${row}.knownWinnerIndex`).toBe(
    expected.knownWinnerIndex,
  );
  if (expected.squaredPrototypeDistances === null) {
    expect(
      actual.squaredPrototypeDistances,
      `${row}.squaredPrototypeDistances`,
    ).toBeNull();
  } else {
    expect(actual.squaredPrototypeDistances, row).not.toBeNull();
    expectVectorWithin(
      actual.squaredPrototypeDistances!,
      expected.squaredPrototypeDistances,
      fixture.tolerance.embedding_abs,
      `${row}.squaredPrototypeDistances`,
    );
  }
}

function expectStageOne(
  actual: TimeDomainDualFusionDecisionV3,
  expected: ExpectedStageOne,
  row: string,
): void {
  const stageOne = actual.openSet.stageOne;
  expect(stageOne.captureLength, `${row}.stageOne.captureLength`).toBe(
    expected.capture_length,
  );
  expect(stageOne.evaluatedLength, `${row}.stageOne.evaluatedLength`).toBe(
    expected.evaluated_length,
  );
  expect(
    stageOne.causalPrefixApplied,
    `${row}.stageOne.causalPrefixApplied`,
  ).toBe(expected.causal_prefix_applied);
  expect(stageOne.gated, `${row}.stageOne.gated`).toBe(expected.gated);
  expectVectorWithin(
    stageOne.features,
    expected.features,
    fixture.tolerance.stage_one_features_abs,
    `${row}.stageOne.features`,
  );
  expectScalarWithin(
    stageOne.score,
    expected.score,
    fixture.tolerance.stage_one_score_abs,
    `${row}.stageOne.score`,
  );
  expectScalarWithin(
    stageOne.thresholdScore,
    expected.threshold_score,
    fixture.tolerance.stage_one_score_abs,
    `${row}.stageOne.thresholdScore`,
  );
  expectScalarWithin(
    stageOne.rank,
    expected.rank,
    fixture.tolerance.rank_abs,
    `${row}.stageOne.rank`,
  );
}

function expectRejector(
  actual: Exclude<TimeDomainDualFusionDecisionV3, { outcome: 'noise' }>,
  expected: ParityRow,
): number {
  const row = expected.name;
  const forward = actual.forward.rejectorForward;
  const stageTwo = actual.openSet.stageTwo;
  const expectedStageTwo = expected.stage_two;
  expect(stageTwo, `${row}.openSet.stageTwo`).not.toBeNull();
  expect(expectedStageTwo, `${row}.fixture.stageTwo`).not.toBeNull();
  expectVectorWithin(
    actual.forward.preprocess.packedInPhase,
    expected.packed_iq.in_phase,
    fixture.tolerance.embedding_abs,
    `${row}.preprocess.packedInPhase`,
  );
  expectVectorWithin(
    actual.forward.preprocess.packedQuadrature,
    expected.packed_iq.quadrature,
    fixture.tolerance.embedding_abs,
    `${row}.preprocess.packedQuadrature`,
  );
  expectVectorWithin(
    actual.forward.preprocess.rawFeatures,
    expected.raw_features,
    fixture.tolerance.embedding_abs,
    `${row}.preprocess.rawFeatures`,
  );
  expectVectorWithin(
    forward.standardizedFeatures,
    expected.rejector_standardized_features,
    fixture.tolerance.embedding_abs,
    `${row}.rejector.standardizedFeatures`,
  );
  expectVectorWithin(
    forward.realEmbedding,
    expectedStageTwo!.real_embedding,
    fixture.tolerance.embedding_abs,
    `${row}.rejector.realEmbedding`,
  );
  expectVectorWithin(
    forward.complexEmbedding,
    expectedStageTwo!.complex_embedding,
    fixture.tolerance.embedding_abs,
    `${row}.rejector.complexEmbedding`,
  );
  expectVectorWithin(
    forward.fusedEmbedding,
    expectedStageTwo!.fused_embedding,
    fixture.tolerance.embedding_abs,
    `${row}.rejector.fusedEmbedding`,
  );
  expectVectorWithin(
    forward.squaredPrototypeDistances,
    expectedStageTwo!.squared_prototype_distances,
    fixture.tolerance.embedding_abs,
    `${row}.rejector.squaredPrototypeDistances`,
  );
  expect(forward.closedWinnerIndex, `${row}.rejector.closedWinnerIndex`).toBe(
    expected.rejector_closed_winner_index,
  );
  expect(forward.closedLabel, `${row}.rejector.closedLabel`).toBe(
    expected.rejector_closed_winner_label,
  );

  // Replaying the open-set scorer on the exact Python embeddings isolates and
  // verifies the exported LOF data to the fixture's strict 1e-9 tolerance.
  const referenceReplay = referenceStageTwo.evaluate({
    realEmbedding: expectedStageTwo!.real_embedding,
    complexEmbedding: expectedStageTwo!.complex_embedding,
    packedInPhase: expected.packed_iq.in_phase,
    packedQuadrature: expected.packed_iq.quadrature,
    predictedClassIndex: expectedStageTwo!.predicted_class_index,
  });
  for (let index = 0; index < expectedStageTwo!.lof.length; index++) {
    const found = referenceReplay.lofComponents[index]!;
    const wanted = expectedStageTwo!.lof[index]!;
    expectRelativeWithin(
      found.raw,
      wanted.raw,
      fixture.tolerance.raw_lof_relative,
      `${row}.referenceReplay.lof[${index}].raw`,
    );
    expectScalarWithin(
      found.rank,
      wanted.rank,
      fixture.tolerance.rank_abs,
      `${row}.referenceReplay.lof[${index}].rank`,
    );
  }

  const numericStageTwo = stageTwo!;
  expect(numericStageTwo.lofComponents).toHaveLength(
    expectedStageTwo!.lof.length,
  );
  const policy = classifier.openSet.asset.stage_two.policy;
  let lofRankBound = 0;
  for (let index = 0; index < expectedStageTwo!.lof.length; index++) {
    const found = numericStageTwo.lofComponents[index]!;
    const wanted = expectedStageTwo!.lof[index]!;
    expect(found.branch, `${row}.lof[${index}].branch`).toBe(wanted.branch);
    expect(found.weight, `${row}.lof[${index}].weight`).toBe(wanted.weight);
    expectRelativeWithin(
      found.raw,
      wanted.raw,
      E2E_LOF_RAW_RELATIVE,
      `${row}.lof[${index}].raw`,
    );
    const component = classifier.openSet.asset.stage_two.lof_components.find(
      (candidate) => candidate.branch === wanted.branch,
    )!;
    const rawRadius = Math.max(
      Math.abs(wanted.raw) * E2E_LOF_RAW_RELATIVE,
      Number.EPSILON,
    );
    const rankBound = rankToleranceAround(
      component.calibration,
      wanted.raw,
      rawRadius,
    );
    expectScalarWithin(
      found.rank,
      wanted.rank,
      rankBound,
      `${row}.lof[${index}].rank`,
    );
    lofRankBound += wanted.weight * rankBound;
  }
  expectScalarWithin(
    numericStageTwo.lofRank,
    expectedStageTwo!.lof_rank,
    lofRankBound,
    `${row}.stageTwo.lofRank`,
  );
  expectScalarWithin(
    numericStageTwo.geometryStatistic,
    expectedStageTwo!.geometry_statistic,
    fixture.tolerance.embedding_abs,
    `${row}.stageTwo.geometryStatistic`,
  );
  expectScalarWithin(
    numericStageTwo.geometryDeviation,
    expectedStageTwo!.geometry_deviation,
    fixture.tolerance.embedding_abs,
    `${row}.stageTwo.geometryDeviation`,
  );
  const geometryRankBound = rankToleranceAround(
    policy.geometry_calibration,
    expectedStageTwo!.geometry_deviation,
    fixture.tolerance.embedding_abs,
  );
  expectScalarWithin(
    numericStageTwo.geometryRank,
    expectedStageTwo!.geometry_rank,
    geometryRankBound,
    `${row}.stageTwo.geometryRank`,
  );
  const combinedBound = (
    policy.branch_lof_rank_weight * lofRankBound
    + policy.geometry_weight * geometryRankBound
    + Number.EPSILON
  );
  expectScalarWithin(
    numericStageTwo.combinedRaw,
    expectedStageTwo!.combined_raw,
    combinedBound,
    `${row}.stageTwo.combinedRaw`,
  );
  const scoreBound = rankToleranceAround(
    policy.combined_calibration,
    expectedStageTwo!.combined_raw,
    combinedBound,
  );
  expectScalarWithin(
    numericStageTwo.score,
    expectedStageTwo!.score,
    scoreBound,
    `${row}.stageTwo.score`,
  );
  expect(numericStageTwo.threshold, `${row}.stageTwo.threshold`).toBe(
    expectedStageTwo!.threshold,
  );
  return scoreBound;
}

function expectClassifier(
  actual: Extract<TimeDomainDualFusionDecisionV3, { outcome: 'known' }>,
  expected: ParityRow,
): void {
  const row = expected.name;
  const forward = actual.forward.classifierForward;
  const wanted = expected.classifier;
  expect(wanted, `${row}.fixture.classifier`).not.toBeNull();
  expect(expected.classifier_standardized_features, row).not.toBeNull();
  expectVectorWithin(
    forward.standardizedFeatures,
    expected.classifier_standardized_features!,
    fixture.tolerance.embedding_abs,
    `${row}.classifier.standardizedFeatures`,
  );
  expectVectorWithin(
    forward.realEmbedding,
    wanted!.real_embedding,
    fixture.tolerance.embedding_abs,
    `${row}.classifier.realEmbedding`,
  );
  expectVectorWithin(
    forward.complexEmbedding,
    wanted!.complex_embedding,
    fixture.tolerance.embedding_abs,
    `${row}.classifier.complexEmbedding`,
  );
  expectVectorWithin(
    forward.fusedEmbedding,
    wanted!.fused_embedding,
    fixture.tolerance.embedding_abs,
    `${row}.classifier.fusedEmbedding`,
  );
  expectVectorWithin(
    forward.squaredPrototypeDistances,
    wanted!.squared_prototype_distances,
    fixture.tolerance.embedding_abs,
    `${row}.classifier.squaredPrototypeDistances`,
  );
  expect(forward.closedWinnerIndex, `${row}.classifier.closedWinnerIndex`).toBe(
    wanted!.predicted_class_index,
  );
  expect(forward.closedLabel, `${row}.classifier.closedLabel`).toBe(
    wanted!.predicted_class_label,
  );
}

function expectParityRow(expected: ParityRow): TimeDomainDualFusionDecisionV3 {
  const row = expected.name;
  const actual = classifier.classify(
    expected.iq.in_phase,
    expected.iq.quadrature,
  );
  expect(actual.candidateId, `${row}.candidateId`).toBe(fixture.candidate_id);
  expectPublicResult(actual, expected.public_result, row);
  expectStageOne(actual, expected.stage_one, row);
  expect(actual.openSet.gated, `${row}.openSet.gated`).toBe(
    expected.rejected_stage === 1,
  );
  expect(actual.openSet.rejectedStage, `${row}.openSet.rejectedStage`).toBe(
    expected.rejected_stage,
  );

  let stageTwoScoreBound = 0;
  if (expected.rejected_stage === 1) {
    expect(actual.outcome, row).toBe('noise');
    expect(actual.forward, `${row}.forward`).toBeNull();
    expect(actual.openSet.stageTwo, `${row}.openSet.stageTwo`).toBeNull();
    expect(expected.stage_two, `${row}.fixture.stageTwo`).toBeNull();
    expect(expected.classifier, `${row}.fixture.classifier`).toBeNull();
    expect(expected.classifier_executed, row).toBe(false);
    expect(expected.classifier_standardized_features, row).toBeNull();
  } else {
    expect(actual.outcome, row).not.toBe('noise');
    stageTwoScoreBound = expectRejector(
      actual as Exclude<
        TimeDomainDualFusionDecisionV3,
        { outcome: 'noise' }
      >,
      expected,
    );
    if (expected.rejected_stage === 2) {
      expect(actual.outcome, row).toBe('unknown');
      if (actual.outcome !== 'unknown') {
        throw new Error(`${row}: expected unknown`);
      }
      expect(actual.forward.classifierForward, row).toBeNull();
      expect(expected.classifier, `${row}.fixture.classifier`).toBeNull();
      expect(expected.classifier_executed, row).toBe(false);
      expect(expected.classifier_standardized_features, row).toBeNull();
    } else {
      expect(actual.outcome, row).toBe('known');
      if (actual.outcome !== 'known') {
        throw new Error(`${row}: expected known`);
      }
      expect(expected.classifier_executed, row).toBe(true);
      expectClassifier(actual, expected);
    }
  }
  let stageOneSurvivorRankBound = 0;
  if (expected.stage_one_survivor_rank === null) {
    expect(actual.openSet.stageOneSurvivorRank, row).toBeNull();
  } else {
    stageOneSurvivorRankBound = rankToleranceAround(
      classifier.openSet.asset.composite.stage_one_calibration_raw,
      expected.stage_one.score,
      fixture.tolerance.stage_one_score_abs,
    );
    expectScalarWithin(
      actual.openSet.stageOneSurvivorRank!,
      expected.stage_one_survivor_rank,
      stageOneSurvivorRankBound,
      `${row}.openSet.stageOneSurvivorRank`,
    );
  }
  const compositeBound = Math.max(
    stageTwoScoreBound,
    stageOneSurvivorRankBound,
  );
  if (expected.composite_score === null) {
    expect(actual.openSet.compositeScore, row).toBeNull();
  } else {
    expectScalarWithin(
      actual.openSet.compositeScore!,
      expected.composite_score,
      compositeBound,
      `${row}.openSet.compositeScore`,
    );
  }
  expectScalarWithin(
    actual.openSet.stagedScore,
    expected.staged_score,
    expected.rejected_stage === 1
      ? fixture.tolerance.rank_abs
      : compositeBound,
    `${row}.openSet.stagedScore`,
  );
  expectScalarWithin(
    actual.openSet.threshold,
    expected.staged_threshold,
    fixture.tolerance.rank_abs,
    `${row}.openSet.threshold`,
  );
  expect(actual.label, `${row}.decisionLabel`).toBe(expected.decision_label);
  return actual;
}

describe('real dual-fusion v3 Python export ↔ TypeScript runtime parity', () => {
  it('hashes every exact package and parity byte sequence before parsing', () => {
    expect(packageManifest).toMatchObject({
      schema: PACKAGE_SCHEMA,
      schema_version: 1,
      status: STAGING_STATUS,
      candidate_id: TIME_DOMAIN_DUAL_FUSION_CANDIDATE_ID,
      architecture: {
        execution_order: [
          'stage_one_noise_gate',
          'rejector_known_unknown',
          'classifier_known_label',
        ],
        classifier_runs_only_after_rejector_acceptance: true,
        public_known_label_from_classifier_only: true,
      },
    });
    expect(Object.keys(packageManifest.assets).sort()).toEqual(
      [...assetNames].sort(),
    );
    for (const name of assetNames) {
      const record = packageManifest.assets[name]!;
      expect(record.path, `${name}.path`).toBe(name);
      expect(statSync(new URL(name, packageRoot)).size, `${name}.bytes`).toBe(
        record.bytes,
      );
      expect(assetDigests[name], `${name}.sha256`).toBe(record.sha256);
      expect(record.status, `${name}.status`).toBe(STAGING_STATUS);
    }
    expect(parityRecord).toMatchObject({
      schema: PARITY_SCHEMA,
      schema_version: 3,
      status: STAGING_STATUS,
      packaged: false,
    });
    expect(statSync(parityUrl).size, 'parity.bytes').toBe(parityRecord.bytes);
    expect(parityDigest, 'parity.sha256').toBe(parityRecord.sha256);
  });

  it('matches every decision and reachable stage on all 62 Python rows', () => {
    expect(fixture).toMatchObject({
      schema: PARITY_SCHEMA,
      schema_version: 3,
      status: STAGING_STATUS,
      candidate_id: TIME_DOMAIN_DUAL_FUSION_CANDIDATE_ID,
      counts: {
        rows: 62,
        gated_stage_one: 12,
        rejected_stage_two: 18,
        accepted: 32,
        classifier_executed: 32,
        accepted_role_winner_disagreements: 1,
      },
      accepted_role_winner_disagreement_rows: [
        'known-bluetooth-N4096',
      ],
      role_contract: {
        rejector: 'known_unknown_rejector',
        classifier: 'accepted_known_classifier',
        classifier_runs_only_after_rejector_acceptance: true,
        public_known_label_from_classifier_only: true,
      },
    });
    expect(fixture.rows).toHaveLength(fixture.counts.rows);

    const actual = new Map(
      fixture.rows.map((row) => [row.name, expectParityRow(row)]),
    );
    expect(
      [...actual.values()].filter((value) => value.outcome === 'noise'),
    ).toHaveLength(fixture.counts.gated_stage_one);
    expect(
      [...actual.values()].filter((value) => value.outcome === 'unknown'),
    ).toHaveLength(fixture.counts.rejected_stage_two);
    expect(
      [...actual.values()].filter((value) => value.outcome === 'known'),
    ).toHaveLength(fixture.counts.accepted);

    const disagreement = actual.get('known-bluetooth-N4096');
    expect(disagreement?.outcome).toBe('known');
    if (disagreement?.outcome !== 'known') {
      throw new Error('winner-disagreement row was not accepted');
    }
    expect(disagreement.forward.rejectorForward.closedLabel).toBe('gsm');
    expect(disagreement.forward.classifierForward.closedLabel).toBe(
      'bluetooth',
    );
    expect(disagreement.label).toBe('bluetooth');
    expect(disagreement.knownLabel).toBe('bluetooth');
  }, 120_000);
});
