import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  COMPOSITE_SURVIVOR_SCORE,
  COMPOSITE_THRESHOLD_QUANTILE,
  COMPOSITE_ONLY_POLICY_CHANGE,
  COMPOSITE_THRESHOLD_POPULATION,
  CompositeSurvivorPolicyV3,
  FROZEN_BRANCH_LOF_RANK_WEIGHT,
  FROZEN_GEOMETRY_FEATURE,
  FROZEN_GEOMETRY_WEIGHT,
  FROZEN_STAGE_TWO_POLICY_KIND,
  FROZEN_THRESHOLD_QUANTILE,
  NOMINAL_ENROLLMENT_FALSE_UNKNOWN_BUDGET,
  NOISE_PREFILTER_VERSION,
  PREFILTER_FEATURE_NAMES,
  STAGE_ONE_SCORE_OFFSET,
  STAGE_ONE_KNOWN_FALSE_POSITIVE_BUDGET,
  STAGED_POLICY_KIND,
  STAGED_POLICY_SCHEMA,
  STAGED_POLICY_VERSION,
  SURVIVOR_KNOWN_FALSE_POSITIVE_BUDGET,
  TIME_DOMAIN_OPENSET_SCHEMA,
  TIME_DOMAIN_OPENSET_SCHEMA_VERSION,
  StageOneGateV3,
  StageTwoRejectorV3,
  TimeDomainOpenSetV3,
  acrossPatchFrequencyDispersion,
  empiricalRank,
  loadTimeDomainOpenSetAssetV3,
  numpyLinearQuantile,
  prefilterPoseDegeneracyFeatures,
  frontendMinimumBandwidth,
  stagedArchitectureContract,
  type CompositeSurvivorPolicyAsset,
  type StageOneEvaluation,
  type StageTwoAsset,
  type TimeDomainOpenSetAssetV3,
} from './time-domain-openset-v3.js';
import {
  preprocessTimeDomainInvariantPatches,
} from './time-domain-invariant-patch-preprocess-v3.js';

interface FixtureStageOne {
  features: number[];
  score: number;
  rank: number;
  gated: boolean;
  threshold_score: number;
}

interface FixtureStageTwo {
  real_embedding: number[];
  complex_embedding: number[];
  fused_embedding: number[];
  predicted_class_index: number;
  predicted_class_label: string;
  squared_prototype_distances: number[];
  lof: Array<{
    branch: 'real' | 'complex';
    weight: number;
    neighbors: number;
    raw: number;
    rank: number;
  }>;
  lof_rank: number;
  geometry_statistic: number;
  geometry_deviation: number;
  geometry_rank: number;
  combined_raw: number;
  score: number;
  /** The stage-2 policy's own q95, recorded for cross-checks only. */
  threshold: number;
}

interface FixtureRow {
  family: 'known' | 'noise' | 'chirp' | 'probe';
  name: string;
  capture_length: number;
  iq: { in_phase: number[]; quadrature: number[] };
  stage_one: FixtureStageOne;
  packed_iq: { in_phase: number[]; quadrature: number[] };
  raw_features: number[];
  standardized_features: number[];
  stage_two: FixtureStageTwo | null;
  /** Null exactly when gated (staged policy version 4 survivor terms). */
  stage_one_survivor_rank: number | null;
  composite_score: number | null;
  staged_threshold: number;
  staged_score: number;
  rejected_stage: 1 | 2 | null;
  decision_label: string;
}

interface ParityFixture {
  schema: string;
  schema_version: number;
  classes: string[];
  frontend: {
    patch_length: number;
    patch_count: number;
    target_frac: number;
    packed_length: number;
  };
  contract: Record<string, unknown>;
  counts: Record<string, number>;
  policy_version: string;
  survivor_score: string;
  staged_threshold: number;
  stage_two_threshold: number;
  rows: FixtureRow[];
}

// Tests consume the tracked dual runtime package. Until a validated v3.4/q97
// package exists, the old package is explicitly refused and real parity stays
// skipped; synthetic schema-4 unit tests below continue to run.
const staging = new URL('./assets-v3-dual-staging/', import.meta.url);
const packageManifest = JSON.parse(
  readFileSync(new URL('runtime-package-manifest.json', staging), 'utf8'),
) as {
  candidate_id: string;
  external_evidence: { parity: { path: string } };
};
const rawAsset = JSON.parse(
  readFileSync(new URL('time-domain-v3-openset-policy.json', staging), 'utf8'),
) as unknown;
const fixture = JSON.parse(
  readFileSync(
    new URL(packageManifest.external_evidence.parity.path, staging),
    'utf8',
  ),
) as ParityFixture;

const candidateId = 'v3.4-q97-decoupled-8k-classifier-4k-rejector';
const assetsAreCurrent = (
  packageManifest.candidate_id === candidateId
  && (rawAsset as { schema_version?: unknown }).schema_version
    === TIME_DOMAIN_OPENSET_SCHEMA_VERSION
  && fixture.schema_version === 4
);
const loadedAsset = assetsAreCurrent
  ? loadTimeDomainOpenSetAssetV3(rawAsset)
  : null;
const legacyAssetForSkippedTests = rawAsset as TimeDomainOpenSetAssetV3;
const asset = (
  loadedAsset
  ?? {
    ...legacyAssetForSkippedTests,
    stage_two: {
      ...legacyAssetForSkippedTests.stage_two,
      kind: FROZEN_STAGE_TWO_POLICY_KIND,
    },
  }
) as TimeDomainOpenSetAssetV3;
const openSet = loadedAsset === null
  ? (undefined as unknown as TimeDomainOpenSetV3)
  : new TimeDomainOpenSetV3(loadedAsset);
const describeCurrent = assetsAreCurrent ? describe : describe.skip;
const itCurrent = assetsAreCurrent ? it : it.skip;

function maxAbsDelta(actual: ArrayLike<number>, expected: number[]): number {
  expect(actual.length).toBe(expected.length);
  let worst = 0;
  for (let index = 0; index < expected.length; index++) {
    worst = Math.max(worst, Math.abs(actual[index]! - expected[index]!));
  }
  return worst;
}

describe('time-domain openset v3 asset admission boundary', () => {
  it('refuses stale q95/q99 assets until validated q97 assets exist', () => {
    if (assetsAreCurrent) {
      expect(() => loadTimeDomainOpenSetAssetV3(rawAsset)).not.toThrow();
    } else {
      expect(() => loadTimeDomainOpenSetAssetV3(rawAsset)).toThrow();
    }
  });
});

describeCurrent('time-domain openset v3 asset', () => {
  it('carries the staged architecture contract keys', () => {
    expect(asset.contract.additive_only).toBe(false);
    expect(asset.contract.changes_closed_label).toBe(true);
    expect(asset.contract.gates_before_classification).toBe(true);
    expect(asset.contract.architecture_contract_change.length).toBeGreaterThan(0);
  });

  it('refuses an asset without the contract keys', () => {
    const broken = JSON.parse(JSON.stringify({
      ...asset,
      contract: { ...asset.contract, additive_only: true },
    })) as Record<string, unknown>;
    expect(() => loadTimeDomainOpenSetAssetV3(broken)).toThrow(
      /architecture contract/,
    );
  });

  it('exposes per-length prefilters for every fixture length', () => {
    const gate = new StageOneGateV3(asset.stage_one, {
      patchLength: asset.frontend.patch_length,
      targetFrac: asset.frontend.target_frac,
    });
    for (const row of fixture.rows) {
      expect(gate.lengths()).toContain(row.capture_length);
    }
  });

  it('refuses to score a capture length without a fitted bundle', () => {
    const gate = new StageOneGateV3(asset.stage_one, {
      patchLength: asset.frontend.patch_length,
      targetFrac: asset.frontend.target_frac,
    });
    expect(() => gate.modelFor(2048)).toThrow(/no fitted noise prefilter/);
  });

  it('pins the frozen policy constants', () => {
    expect(asset.stage_two.kind).toBe(FROZEN_STAGE_TWO_POLICY_KIND);
    expect(asset.stage_two.policy.branch_lof_rank_weight).toBe(
      FROZEN_BRANCH_LOF_RANK_WEIGHT,
    );
    expect(asset.stage_two.policy.geometry_weight).toBe(FROZEN_GEOMETRY_WEIGHT);
    expect(asset.stage_two.policy.threshold_quantile).toBe(0.95);
    const branches = asset.stage_two.lof_components.map((component) => [
      component.branch,
      component.weight,
      component.neighbors,
    ]);
    expect(branches).toEqual([
      ['real', 0.4, 2],
      ['complex', 0.6, 64],
    ]);
  });

  it('pins the staged composite policy (version 4 / q97)', () => {
    expect(asset.composite.policy_version).toBe(STAGED_POLICY_VERSION);
    expect(asset.composite.kind).toBe(STAGED_POLICY_KIND);
    expect(asset.composite.schema).toBe(STAGED_POLICY_SCHEMA);
    expect(asset.composite.survivor_score).toBe(COMPOSITE_SURVIVOR_SCORE);
    expect(asset.composite.threshold_quantile).toBe(
      COMPOSITE_THRESHOLD_QUANTILE,
    );
    expect(asset.composite.threshold_population).toBe(
      COMPOSITE_THRESHOLD_POPULATION,
    );
    // The composite was fit against exactly this stage-2 state.
    expect(asset.composite.stage_two_threshold).toBe(
      asset.stage_two.policy.threshold,
    );
    // The fixture and the asset agree on the decision threshold.
    expect(fixture.staged_threshold).toBe(asset.composite.threshold);
    expect(fixture.policy_version).toBe(STAGED_POLICY_VERSION);
  });

  it.each([
    [
      'legacy q95',
      2,
      'v3-staged-openset-policy-v2-composite-survivor',
      'v3_staged_noise_prefilter_then_composite_survivor_lof_geometry',
      0.95,
    ],
    [
      'failed q99',
      3,
      'v3-staged-openset-policy-v3-composite-survivor-q99',
      'v3_staged_noise_prefilter_then_q99_composite_survivor_lof_geometry',
      0.99,
    ],
  ])('refuses %s policy semantics', (
    _label,
    schema,
    policyVersion,
    kind,
    quantile,
  ) => {
    const broken = JSON.parse(JSON.stringify(asset)) as {
      composite: {
        schema: number;
        policy_version: string;
        kind: string;
        threshold_quantile: number;
      };
    };
    Object.assign(broken.composite, {
      schema,
      policy_version: policyVersion,
      kind,
      threshold_quantile: quantile,
    });
    expect(() => loadTimeDomainOpenSetAssetV3(broken)).toThrow(
      /staged policy version mismatch/,
    );
  });

  it('refuses a composite fit against a different stage-2 state', () => {
    const broken = JSON.parse(JSON.stringify(asset)) as {
      composite: { stage_two_threshold: number };
    };
    broken.composite.stage_two_threshold += 1e-6;
    expect(() => loadTimeDomainOpenSetAssetV3(broken)).toThrow(
      /different stage-2 state/,
    );
  });

  it('refuses a composite whose stored threshold is not the frozen quantile', () => {
    const broken = JSON.parse(JSON.stringify(asset)) as {
      composite: { threshold: number };
    };
    broken.composite.threshold = Math.min(
      0.999999,
      broken.composite.threshold + 1e-9,
    );
    expect(() => loadTimeDomainOpenSetAssetV3(broken)).toThrow(
      /frozen quantile/,
    );
  });
});

describe('numpy linear quantile primitive', () => {
  it('matches np.quantile linear interpolation on both lerp branches', () => {
    // t < 0.5 branch: q=0.1 over [0,1,2,3] -> virtual 0.3 -> 0 + 0.3*1
    expect(numpyLinearQuantile([0, 1, 2, 3], 0.1)).toBeCloseTo(0.3, 15);
    // t >= 0.5 branch: q=0.95 over 5 values -> virtual 3.8 ->
    // b - (b-a)*(1-t) with a=6, b=8, t=0.8 -> 8 - 2*0.2
    expect(numpyLinearQuantile([0, 2, 4, 6, 8], 0.95)).toBeCloseTo(7.6, 15);
    // Endpoints.
    expect(numpyLinearQuantile([5], 0.95)).toBe(5);
    expect(numpyLinearQuantile([1, 2], 1.0)).toBe(2);
  });

  itCurrent('reproduces the asset threshold from the asset calibration exactly', () => {
    expect(
      numpyLinearQuantile(
        asset.composite.composite_calibration_raw,
        COMPOSITE_THRESHOLD_QUANTILE,
      ),
    ).toBe(asset.composite.threshold);
  });
});

describeCurrent('stage-1 parity: pose-degeneracy prefilter', () => {
  it('reproduces features, scores, and every gate decision', () => {
    let worstFeature = 0;
    let worstScore = 0;
    for (const row of fixture.rows) {
      const evaluation = openSet.evaluateStageOne(
        row.iq.in_phase,
        row.iq.quadrature,
      );
      worstFeature = Math.max(
        worstFeature,
        maxAbsDelta(evaluation.features, row.stage_one.features),
      );
      worstScore = Math.max(
        worstScore,
        Math.abs(evaluation.score - row.stage_one.score),
      );
      expect(
        Math.abs(evaluation.rank - row.stage_one.rank),
        `${row.name} stage-1 rank`,
      ).toBeLessThanOrEqual(1e-9);
      expect(evaluation.thresholdScore).toBe(row.stage_one.threshold_score);
      expect(evaluation.gated, `${row.name} stage-1 gate`).toBe(
        row.stage_one.gated,
      );
    }
    expect(worstFeature).toBeLessThanOrEqual(1e-8);
    expect(worstScore).toBeLessThanOrEqual(1e-8);
  });

  it('matches the frontend resolution floor exactly', () => {
    // The staged Python path computes stage-1 features at the frontend's own
    // length-dependent minimum bandwidth; a mismatch here would silently
    // change every feature.
    for (const length of [4096, 8192]) {
      const minimum = frontendMinimumBandwidth(
        length,
        fixture.frontend.patch_length,
        fixture.frontend.target_frac,
      );
      expect(minimum).toBeGreaterThan(0);
      expect(minimum).toBeLessThan(1);
    }
  });
});

/**
 * Empirical ranks are `searchsorted(calibration, value, 'left') / (n + 1)`.
 * The enrollment geometry calibration contains large EXACT tie blocks
 * (hundreds of rows share a float32 statistic of exactly 0), so a one-ulp
 * host-libm difference in the query can legitimately move the rank across a
 * whole block. The rigorous bound is therefore adaptive: given that the TS
 * raw value provably lies within `radius` of the Python raw value, the rank
 * may differ by at most the number of calibration entries in that window.
 */
function rankToleranceAround(
  sortedCalibration: number[],
  value: number,
  radius: number,
): number {
  let low = 0;
  let high = sortedCalibration.length;
  while (low < high) {
    const middle = low + Math.floor((high - low) / 2);
    if (sortedCalibration[middle]! < value - radius) low = middle + 1;
    else high = middle;
  }
  const first = low;
  low = first;
  high = sortedCalibration.length;
  while (low < high) {
    const middle = low + Math.floor((high - low) / 2);
    if (sortedCalibration[middle]! <= value + radius) low = middle + 1;
    else high = middle;
  }
  return (low - first + 1) / (sortedCalibration.length + 1);
}

const GEOMETRY_DEVIATION_RADIUS = 1e-9;

describeCurrent('stage-2 parity: branch LOF + frozen geometry blend', () => {
  const rejector = new StageTwoRejectorV3(asset.stage_two, {
    patchCount: asset.frontend.patch_count,
    patchLength: asset.frontend.patch_length,
  });

  it('reproduces every survivor score from Python embeddings and TS packing', () => {
    const policy = asset.stage_two.policy;
    let worstLofRank = 0;
    let worstRawRelative = 0;
    for (const row of fixture.rows) {
      if (row.stage_two === null) continue;
      const expected = row.stage_two;
      const evaluation = rejector.evaluate({
        realEmbedding: expected.real_embedding,
        complexEmbedding: expected.complex_embedding,
        packedInPhase: row.packed_iq.in_phase,
        packedQuadrature: row.packed_iq.quadrature,
        predictedClassIndex: expected.predicted_class_index,
      });
      for (let index = 0; index < evaluation.lofComponents.length; index++) {
        const component = evaluation.lofComponents[index]!;
        const reference = expected.lof[index]!;
        expect(component.branch).toBe(reference.branch);
        worstRawRelative = Math.max(
          worstRawRelative,
          Math.abs(component.raw - reference.raw)
            / Math.max(Math.abs(reference.raw), 1e-12),
        );
        worstLofRank = Math.max(
          worstLofRank,
          Math.abs(component.rank - reference.rank),
        );
      }
      worstLofRank = Math.max(
        worstLofRank,
        Math.abs(evaluation.lofRank - expected.lof_rank),
      );
      expect(
        Math.abs(evaluation.geometryStatistic - expected.geometry_statistic),
        `${row.name} geometry statistic`,
      ).toBeLessThanOrEqual(1e-6);
      expect(
        Math.abs(evaluation.geometryDeviation - expected.geometry_deviation),
        `${row.name} geometry deviation`,
      ).toBeLessThanOrEqual(GEOMETRY_DEVIATION_RADIUS);
      // Adaptive rank bound around the tie blocks (see rankToleranceAround).
      const geometryBound = rankToleranceAround(
        policy.geometry_calibration,
        expected.geometry_deviation,
        GEOMETRY_DEVIATION_RADIUS,
      );
      expect(
        Math.abs(evaluation.geometryRank - expected.geometry_rank),
        `${row.name} geometry rank (bound ${geometryBound})`,
      ).toBeLessThanOrEqual(geometryBound);
      const combinedRadius = (
        policy.geometry_weight * geometryBound
        + policy.branch_lof_rank_weight * 1e-9
        + 1e-12
      );
      expect(
        Math.abs(evaluation.combinedRaw - expected.combined_raw),
        `${row.name} combined raw`,
      ).toBeLessThanOrEqual(combinedRadius);
      const scoreBound = rankToleranceAround(
        policy.combined_calibration,
        expected.combined_raw,
        combinedRadius,
      );
      expect(
        Math.abs(evaluation.score - expected.score),
        `${row.name} stage-2 score (bound ${scoreBound})`,
      ).toBeLessThanOrEqual(scoreBound);
      // The stage-2 threshold is recorded for cross-checks only; the staged
      // decision is made on the outer q97 composite (staged policy version 4).
      expect(evaluation.threshold).toBe(expected.threshold);
    }
    expect(worstRawRelative).toBeLessThanOrEqual(1e-9);
    expect(worstLofRank).toBeLessThanOrEqual(1e-9);
  });

  it('computes the geometry statistic from TS-preprocessed packing too', () => {
    // End-to-end: raw I/Q through the TS frontend, then the dispersion
    // statistic, compared against the Python statistic for the same row.
    for (const row of fixture.rows) {
      if (row.stage_two === null) continue;
      const preprocess = preprocessTimeDomainInvariantPatches(
        row.iq.in_phase,
        row.iq.quadrature,
        {
          patchLength: fixture.frontend.patch_length,
          patchCount: fixture.frontend.patch_count,
          targetFrac: fixture.frontend.target_frac,
        },
      );
      const statistic = acrossPatchFrequencyDispersion(
        preprocess.packedInPhase,
        preprocess.packedQuadrature,
        fixture.frontend.patch_count,
        fixture.frontend.patch_length,
      );
      expect(
        Math.abs(statistic - row.stage_two.geometry_statistic),
        `${row.name} e2e geometry statistic`,
      ).toBeLessThanOrEqual(1e-6);
    }
  });
});

describeCurrent('staged decision parity', () => {
  it('matches the Python staged decision INCLUDING which stage rejected', () => {
    for (const row of fixture.rows) {
      const stageOne = openSet.evaluateStageOne(
        row.iq.in_phase,
        row.iq.quadrature,
      );
      let decision;
      if (stageOne.gated) {
        decision = openSet.gatedDecision(stageOne);
      } else {
        const expected = row.stage_two!;
        decision = openSet.finishSurvivor(stageOne, {
          realEmbedding: expected.real_embedding,
          complexEmbedding: expected.complex_embedding,
          packedInPhase: row.packed_iq.in_phase,
          packedQuadrature: row.packed_iq.quadrature,
          predictedClassIndex: expected.predicted_class_index,
        });
      }
      expect(decision.rejectedStage, `${row.name} rejected stage`).toBe(
        row.rejected_stage,
      );
      expect(decision.gated).toBe(row.rejected_stage === 1);
      // The decision threshold is the composite q97, on every row.
      expect(decision.threshold).toBe(row.staged_threshold);
      expect(decision.threshold).toBe(asset.composite.threshold);
      if (row.rejected_stage === 1) {
        // Gated rows have no stage-2 score at all: the short circuit is
        // real, and the gated axis is unchanged by staged policy version 4.
        expect(decision.stageTwo).toBeNull();
        expect(decision.stageOneSurvivorRank).toBeNull();
        expect(decision.compositeScore).toBeNull();
        expect(
          Math.abs(decision.stagedScore - row.staged_score),
          `${row.name} gated staged score`,
        ).toBeLessThanOrEqual(1e-8);
        expect(decision.stagedScore).toBeGreaterThan(STAGE_ONE_SCORE_OFFSET);
        expect(decision.stagedScore).toBeLessThan(2);
      } else {
        // Survivors: the staged score is the COMPOSITE. Both terms carry a
        // provable bound: the stage-2 rank bound is adaptive around the
        // enrollment tie blocks, and the stage-1 survivor rank bound is
        // adaptive around the stage-1 score's provable radius (1e-8, the
        // stage-1 parity tolerance).
        const stageTwoBound = rankToleranceAround(
          asset.stage_two.policy.combined_calibration,
          row.stage_two!.combined_raw,
          asset.stage_two.policy.geometry_weight * rankToleranceAround(
            asset.stage_two.policy.geometry_calibration,
            row.stage_two!.geometry_deviation,
            GEOMETRY_DEVIATION_RADIUS,
          ) + 1e-9,
        );
        const stageOneRankBound = rankToleranceAround(
          asset.composite.stage_one_calibration_raw,
          row.stage_one.score,
          1e-8,
        );
        expect(
          Math.abs(
            decision.stageOneSurvivorRank! - row.stage_one_survivor_rank!,
          ),
          `${row.name} stage-1 survivor rank (bound ${stageOneRankBound})`,
        ).toBeLessThanOrEqual(stageOneRankBound);
        const compositeBound = Math.max(stageTwoBound, stageOneRankBound);
        expect(
          Math.abs(decision.compositeScore! - row.composite_score!),
          `${row.name} composite score (bound ${compositeBound})`,
        ).toBeLessThanOrEqual(compositeBound);
        expect(decision.stagedScore).toBe(decision.compositeScore);
        expect(
          Math.abs(decision.stagedScore - row.staged_score),
          `${row.name} staged score`,
        ).toBeLessThanOrEqual(compositeBound);
        // The composite is a true max: never below its stage-2 term.
        expect(decision.compositeScore!).toBeGreaterThanOrEqual(
          decision.stageTwo!.score,
        );
      }
      // The staged axis reproduces the staged decision exactly.
      const rejected = decision.stagedScore > decision.threshold;
      expect(rejected).toBe(row.rejected_stage !== null);
      // Every decision restates the architecture contract.
      expect(decision.contract).toEqual(stagedArchitectureContract());
    }
  });

  it('refuses to run stage 2 on a gated row', () => {
    const gatedRow = fixture.rows.find((row) => row.rejected_stage === 1)!;
    const stageOne = openSet.evaluateStageOne(
      gatedRow.iq.in_phase,
      gatedRow.iq.quadrature,
    );
    expect(stageOne.gated).toBe(true);
    const survivorRow = fixture.rows.find((row) => row.stage_two !== null)!;
    expect(() =>
      openSet.finishSurvivor(stageOne, {
        realEmbedding: survivorRow.stage_two!.real_embedding,
        complexEmbedding: survivorRow.stage_two!.complex_embedding,
        packedInPhase: survivorRow.packed_iq.in_phase,
        packedQuadrature: survivorRow.packed_iq.quadrature,
        predictedClassIndex: 0,
      })).toThrow(/short circuit/);
  });

  it('covers every decision path in the fixture', () => {
    const stages = new Set(fixture.rows.map((row) => row.rejected_stage));
    expect(stages.has(1)).toBe(true);
    expect(stages.has(2)).toBe(true);
    expect(stages.has(null)).toBe(true);
  });
});

describeCurrent('stage-1 causal-prefix rule for long captures', () => {
  const gate = new StageOneGateV3(asset.stage_one, {
    patchLength: asset.frontend.patch_length,
    targetFrac: asset.frontend.target_frac,
  });
  const LONGEST = Math.max(...gate.lengths());
  const LONG = 2 * LONGEST;

  /** Deterministic PRNG (mulberry32) so the captures are reproducible. */
  function mulberry32(seed: number): () => number {
    let state = seed >>> 0;
    return () => {
      state = (state + 0x6d2b79f5) >>> 0;
      let mixed = Math.imul(state ^ (state >>> 15), state | 1);
      mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
      return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
    };
  }

  /** White complex gaussian noise via Box-Muller. */
  function whiteNoise(
    length: number,
    seed: number,
  ): { inPhase: Float64Array; quadrature: Float64Array } {
    const random = mulberry32(seed);
    const inPhase = new Float64Array(length);
    const quadrature = new Float64Array(length);
    for (let index = 0; index < length; index++) {
      const radius = Math.sqrt(-2 * Math.log(1 - random()));
      const azimuth = 2 * Math.PI * random();
      inPhase[index] = radius * Math.cos(azimuth);
      quadrature[index] = radius * Math.sin(azimuth);
    }
    return { inPhase, quadrature };
  }

  /** A coherent narrowband tone: unambiguously signal-like, never noise. */
  function tone(
    length: number,
  ): { inPhase: Float64Array; quadrature: Float64Array } {
    const inPhase = new Float64Array(length);
    const quadrature = new Float64Array(length);
    const omega = 2 * Math.PI * 0.01;
    for (let index = 0; index < length; index++) {
      const envelope = 1 + 0.25 * Math.sin((2 * Math.PI * index) / 4099);
      inPhase[index] = envelope * Math.cos(omega * index);
      quadrature[index] = envelope * Math.sin(omega * index);
    }
    return { inPhase, quadrature };
  }

  it('gates a long noise-like capture on its first longest-fitted samples', () => {
    // Seed chosen so the N16384 model fires with a wide margin (score 2.89
    // against the tightened 0.01-budget threshold 2.47): stage-1 noise recall
    // is ~0.5 by design at this budget, so a knife-edge draw would make this
    // test fragile for the wrong reason.  (Seed 3 was the wide-margin draw
    // for the retired 0.02-budget threshold 1.80; it scores 2.23 and no
    // longer gates.)
    const noise = whiteNoise(LONG, 43);
    const evaluation = gate.evaluate(noise.inPhase, noise.quadrature);
    expect(evaluation.causalPrefixApplied).toBe(true);
    expect(evaluation.captureLength).toBe(LONG);
    expect(evaluation.evaluatedLength).toBe(LONGEST);
    expect(evaluation.gated, 'a 32768-sample noise capture must be gated').toBe(
      true,
    );

    // The rule is literal: the evaluation IS the evaluation of the prefix.
    const prefix = gate.evaluate(
      noise.inPhase.subarray(0, LONGEST),
      noise.quadrature.subarray(0, LONGEST),
    );
    expect(prefix.causalPrefixApplied).toBe(false);
    expect(prefix.evaluatedLength).toBe(LONGEST);
    expect(Array.from(evaluation.features)).toEqual(
      Array.from(prefix.features),
    );
    expect(evaluation.score).toBe(prefix.score);
    expect(evaluation.rank).toBe(prefix.rank);
    expect(evaluation.gated).toBe(prefix.gated);
    expect(evaluation.thresholdScore).toBe(prefix.thresholdScore);

    // And the staged composition short-circuits exactly as at fitted lengths.
    const decision = openSet.gatedDecision(
      openSet.evaluateStageOne(noise.inPhase, noise.quadrature),
    );
    expect(decision.rejectedStage).toBe(1);
    expect(decision.stageTwo).toBeNull();
    expect(decision.stagedScore).toBeGreaterThan(STAGE_ONE_SCORE_OFFSET);
    expect(decision.stagedScore).toBeLessThan(2);
  });

  it('ignores the samples beyond the causal prefix entirely', () => {
    const noise = whiteNoise(LONG, 0x5eed_0002);
    const rewritten = {
      inPhase: Float64Array.from(noise.inPhase),
      quadrature: Float64Array.from(noise.quadrature),
    };
    const signalTail = tone(LONG);
    for (let index = LONGEST; index < LONG; index++) {
      rewritten.inPhase[index] = signalTail.inPhase[index]!;
      rewritten.quadrature[index] = signalTail.quadrature[index]!;
    }
    const original = gate.evaluate(noise.inPhase, noise.quadrature);
    const tailSwapped = gate.evaluate(rewritten.inPhase, rewritten.quadrature);
    expect(tailSwapped.score).toBe(original.score);
    expect(tailSwapped.gated).toBe(original.gated);
    expect(Array.from(tailSwapped.features)).toEqual(
      Array.from(original.features),
    );
  });

  it('passes a long signal-like capture through to stage 2', () => {
    const signal = tone(LONG);
    const stageOne = openSet.evaluateStageOne(
      signal.inPhase,
      signal.quadrature,
    );
    expect(stageOne.causalPrefixApplied).toBe(true);
    expect(stageOne.evaluatedLength).toBe(LONGEST);
    expect(
      stageOne.gated,
      'a 32768-sample coherent signal must NOT be gated as noise',
    ).toBe(false);

    // Survivors flow to the unchanged stage-2 path. Stage 2 consumes
    // embeddings and packed patches, which are length-invariant; reuse a
    // fixture survivor's to prove the composition runs end to end.
    const survivorRow = fixture.rows.find((row) => row.stage_two !== null)!;
    const decision = openSet.finishSurvivor(stageOne, {
      realEmbedding: survivorRow.stage_two!.real_embedding,
      complexEmbedding: survivorRow.stage_two!.complex_embedding,
      packedInPhase: survivorRow.packed_iq.in_phase,
      packedQuadrature: survivorRow.packed_iq.quadrature,
      predictedClassIndex: survivorRow.stage_two!.predicted_class_index,
    });
    expect(decision.gated).toBe(false);
    expect(decision.stageTwo).not.toBeNull();
    expect(decision.stageOne.causalPrefixApplied).toBe(true);
  });

  it('keeps the refusal below the smallest fitted length', () => {
    const short = whiteNoise(2048, 0x5eed_0003);
    expect(() => gate.evaluate(short.inPhase, short.quadrature)).toThrow(
      /no fitted noise prefilter/,
    );
  });

  it('keeps the refusal at uncovered lengths between fitted lengths', () => {
    const between = whiteNoise(12288, 0x5eed_0004);
    expect(between.inPhase.length).toBeGreaterThan(Math.min(...gate.lengths()));
    expect(between.inPhase.length).toBeLessThan(LONGEST);
    expect(() => gate.evaluate(between.inPhase, between.quadrature)).toThrow(
      /no fitted noise prefilter/,
    );
  });

  it('never applies the prefix rule at fitted lengths', () => {
    for (const row of fixture.rows) {
      const evaluation = gate.evaluate(row.iq.in_phase, row.iq.quadrature);
      expect(evaluation.causalPrefixApplied).toBe(false);
      expect(evaluation.evaluatedLength).toBe(row.capture_length);
    }
  });

  it('refuses mismatched in-phase/quadrature lengths', () => {
    const noise = whiteNoise(LONG, 0x5eed_0005);
    expect(() =>
      gate.evaluate(
        noise.inPhase,
        noise.quadrature.subarray(0, LONG - 1),
      )).toThrow(/same length/);
  });
});

describe('empirical rank primitive', () => {
  it('matches searchsorted-left semantics on ties', () => {
    const calibration = [1, 2, 2, 3];
    expect(empiricalRank(calibration, 2)).toBe(1 / 5);
    expect(empiricalRank(calibration, 2.5)).toBe(3 / 5);
    expect(empiricalRank(calibration, 0)).toBe(0);
    expect(empiricalRank(calibration, 10)).toBe(4 / 5);
  });
});

describe('separate stage-two q95 contract', () => {
  function syntheticStageTwo(): StageTwoAsset {
    const component = (
      branch: 'real' | 'complex',
      weight: number,
      neighbors: number,
    ) => ({
      branch,
      weight,
      neighbors,
      mean: [0],
      scale: [1],
      reference: Array.from({ length: 66 }, (_, index) => [index]),
      reference_k_distance: Array.from({ length: 66 }, () => 1),
      reference_local_density: Array.from({ length: 66 }, () => 1),
      calibration: [0.5, 1],
    });
    const combinedCalibration = [0.1, 0.2, 0.4, 0.8];
    const derivedScores = combinedCalibration.map(
      (value) => empiricalRank(combinedCalibration, value),
    );
    return {
      kind: FROZEN_STAGE_TWO_POLICY_KIND,
      lof_components: [
        component('real', 0.4, 2),
        component('complex', 0.6, 64),
      ],
      policy: {
        branch_lof_rank_weight: FROZEN_BRANCH_LOF_RANK_WEIGHT,
        geometry_weight: FROZEN_GEOMETRY_WEIGHT,
        threshold_quantile: FROZEN_THRESHOLD_QUANTILE,
        threshold: numpyLinearQuantile(
          derivedScores,
          FROZEN_THRESHOLD_QUANTILE,
        ),
        geometry_feature: FROZEN_GEOMETRY_FEATURE,
        class_geometry_mean: [0],
        class_geometry_scale: [1],
        geometry_calibration: [0, 1],
        combined_calibration: combinedCalibration,
      },
    };
  }

  function syntheticOpenSetAsset(): TimeDomainOpenSetAssetV3 {
    const compositeCalibration = [0, 0.2, 0.4, 0.6];
    const stageTwo = syntheticStageTwo();
    return {
      schema: TIME_DOMAIN_OPENSET_SCHEMA,
      schema_version: TIME_DOMAIN_OPENSET_SCHEMA_VERSION,
      status: 'staging_not_release',
      contract: stagedArchitectureContract(),
      frontend: {
        version: 'invariant-patch-time-domain-v1',
        patch_length: 2,
        patch_count: 2,
        target_frac: 0.5,
        packed_length: 4,
        uses_frequency_transform: false,
      },
      stage_one: {
        kind: NOISE_PREFILTER_VERSION,
        feature_names: [...PREFILTER_FEATURE_NAMES],
        match_frontend_min_bandwidth: true,
        models: {
          '4': {
            capture_length: 4,
            feature_names: [...PREFILTER_FEATURE_NAMES],
            mean: Array.from({ length: PREFILTER_FEATURE_NAMES.length }, () => 0),
            scale: Array.from({ length: PREFILTER_FEATURE_NAMES.length }, () => 1),
            coefficients: Array.from(
              { length: PREFILTER_FEATURE_NAMES.length },
              () => 0,
            ),
            intercept: 0,
            threshold_score: 0,
          },
        },
      },
      stage_two: stageTwo,
      composite: {
        schema: STAGED_POLICY_SCHEMA,
        kind: STAGED_POLICY_KIND,
        policy_version: STAGED_POLICY_VERSION,
        survivor_score: COMPOSITE_SURVIVOR_SCORE,
        threshold_quantile: COMPOSITE_THRESHOLD_QUANTILE,
        threshold: numpyLinearQuantile(
          compositeCalibration,
          COMPOSITE_THRESHOLD_QUANTILE,
        ),
        stage_two_threshold: stageTwo.policy.threshold,
        stage_one_calibration_raw: [-1, 0, 1, 2],
        composite_calibration_raw: compositeCalibration,
        enrollment_rows: 5,
        enrollment_gated_rows: 1,
        enrollment_capture_length: 4,
        threshold_population: COMPOSITE_THRESHOLD_POPULATION,
        training_rows_used_for_threshold: 0,
        selection_rows_used_for_threshold: 0,
        novelty_rows_used_for_threshold: 0,
        release_rows_used_for_threshold: 0,
        stage_one_known_false_positive_budget:
          STAGE_ONE_KNOWN_FALSE_POSITIVE_BUDGET,
        survivor_known_false_positive_budget:
          SURVIVOR_KNOWN_FALSE_POSITIVE_BUDGET,
        nominal_enrollment_false_unknown_budget:
          NOMINAL_ENROLLMENT_FALSE_UNKNOWN_BUDGET,
        only_policy_change: COMPOSITE_ONLY_POLICY_CHANGE,
        stage_one_changed: false,
        rejector_cnn_fusion_changed: false,
        classifier_cnn_fusion_changed: false,
        gate_contract_changed: false,
      },
    };
  }

  it('admits the inner q95 kind independently of outer q97', () => {
    expect(() =>
      new StageTwoRejectorV3(
        syntheticStageTwo(),
        { patchCount: 2, patchLength: 2 },
      )).not.toThrow();
    expect(FROZEN_THRESHOLD_QUANTILE).toBe(0.95);
    expect(COMPOSITE_THRESHOLD_QUANTILE).toBe(0.97);
  });

  it('refuses relabelling stage two as the outer q97 policy', () => {
    const broken = syntheticStageTwo() as unknown as Record<string, unknown>;
    broken.kind = STAGED_POLICY_KIND;
    expect(() =>
      new StageTwoRejectorV3(
        broken as unknown as StageTwoAsset,
        { patchCount: 2, patchLength: 2 },
      )).toThrow(/stage two kind/);
  });

  it('refuses moving the inner threshold quantile to q97', () => {
    const broken = syntheticStageTwo();
    broken.policy.threshold_quantile = COMPOSITE_THRESHOLD_QUANTILE;
    expect(() =>
      new StageTwoRejectorV3(
        broken,
        { patchCount: 2, patchLength: 2 },
      )).toThrow(/frozen q95/);
  });

  it('derives inner q95 from combined calibration instead of trusting it', () => {
    const broken = syntheticStageTwo();
    // A fabricated calibration-score vector [0, .5, .6, .7] would yield
    // .685, but the stored raw calibration self-ranks are [0, .2, .4, .6]
    // and their exact q95 is .57.
    broken.policy.threshold = 0.685;
    expect(() =>
      new StageTwoRejectorV3(
        broken,
        { patchCount: 2, patchLength: 2 },
      )).toThrow(/stage-two q95/);
  });

  it.each([
    ['real weight', 'real', 'weight', 0.5],
    ['real neighbors', 'real', 'neighbors', 3],
    ['complex weight', 'complex', 'weight', 0.5],
    ['complex neighbors', 'complex', 'neighbors', 63],
  ] as const)('refuses a non-frozen %s', (_label, branch, field, value) => {
    const broken = syntheticStageTwo();
    const component = broken.lof_components.find(
      (candidate) => candidate.branch === branch,
    )!;
    component[field] = value;
    expect(() =>
      new StageTwoRejectorV3(
        broken,
        { patchCount: 2, patchLength: 2 },
      )).toThrow(/frozen weight/);
  });

  it.each([
    'mean',
    'scale',
    'reference',
    'reference_k_distance',
    'reference_local_density',
    'calibration',
  ] as const)('refuses a LOF component missing %s', (field) => {
    const broken = syntheticStageTwo() as unknown as {
      lof_components: Array<Record<string, unknown>>;
    };
    delete broken.lof_components[0]![field];
    expect(() =>
      new StageTwoRejectorV3(
        broken as unknown as StageTwoAsset,
        { patchCount: 2, patchLength: 2 },
      )).toThrow(/LOF/);
  });

  it.each([
    ['version', 'legacy-hybrid-v2'],
    ['uses_frequency_transform', true],
  ] as const)('refuses an open-set frontend with %s=%s', (field, value) => {
    const broken = syntheticOpenSetAsset() as unknown as {
      frontend: Record<string, unknown>;
    };
    broken.frontend[field] = value;
    expect(() => loadTimeDomainOpenSetAssetV3(broken)).toThrow(
      /frontend geometry/,
    );
  });
});

describe('composite survivor policy primitive', () => {
  function syntheticComposite(): CompositeSurvivorPolicyAsset {
    const compositeCalibration = [0.1, 0.2, 0.4, 0.8];
    return {
      schema: STAGED_POLICY_SCHEMA,
      kind: STAGED_POLICY_KIND,
      policy_version: STAGED_POLICY_VERSION,
      survivor_score: COMPOSITE_SURVIVOR_SCORE,
      threshold_quantile: COMPOSITE_THRESHOLD_QUANTILE,
      threshold: numpyLinearQuantile(
        compositeCalibration,
        COMPOSITE_THRESHOLD_QUANTILE,
      ),
      stage_two_threshold: 0.9,
      stage_one_calibration_raw: [-3, -1, -1, 0.5],
      composite_calibration_raw: compositeCalibration,
      enrollment_rows: 6,
      enrollment_gated_rows: 2,
      enrollment_capture_length: 16384,
      threshold_population: COMPOSITE_THRESHOLD_POPULATION,
      training_rows_used_for_threshold: 0,
      selection_rows_used_for_threshold: 0,
      novelty_rows_used_for_threshold: 0,
      release_rows_used_for_threshold: 0,
      stage_one_known_false_positive_budget:
        STAGE_ONE_KNOWN_FALSE_POSITIVE_BUDGET,
      survivor_known_false_positive_budget:
        SURVIVOR_KNOWN_FALSE_POSITIVE_BUDGET,
      nominal_enrollment_false_unknown_budget:
        NOMINAL_ENROLLMENT_FALSE_UNKNOWN_BUDGET,
      only_policy_change: COMPOSITE_ONLY_POLICY_CHANGE,
      stage_one_changed: false,
      rejector_cnn_fusion_changed: false,
      classifier_cnn_fusion_changed: false,
      gate_contract_changed: false,
    };
  }

  it('ranks stage-1 scores searchsorted-left against enrollment survivors', () => {
    const policy = new CompositeSurvivorPolicyV3(syntheticComposite());
    // Tie block: -1 appears twice; searchsorted-left lands BEFORE the block.
    expect(policy.stageOneSurvivorRank(-1)).toBe(1 / 5);
    expect(policy.stageOneSurvivorRank(-10)).toBe(0);
    expect(policy.stageOneSurvivorRank(2)).toBe(4 / 5);
  });

  it('computes the composite as a true max of the two ranks', () => {
    const policy = new CompositeSurvivorPolicyV3(syntheticComposite());
    // stage-2 term dominates.
    expect(policy.composite(0.3, -1)).toBe(0.3);
    // stage-1 term dominates.
    expect(policy.composite(0.1, 2)).toBe(4 / 5);
  });

  it('refuses an unsorted stage-1 calibration', () => {
    const broken = syntheticComposite();
    broken.stage_one_calibration_raw = [0.5, -1, -1, -3];
    expect(() => new CompositeSurvivorPolicyV3(broken)).toThrow(/sorted/);
  });

  it('refuses a composite calibration outside [0, 1)', () => {
    const broken = syntheticComposite();
    broken.composite_calibration_raw = [0.1, 0.2, 0.4, 1.0];
    broken.threshold = numpyLinearQuantile(
      broken.composite_calibration_raw,
      COMPOSITE_THRESHOLD_QUANTILE,
    );
    expect(() => new CompositeSurvivorPolicyV3(broken)).toThrow(/\[0, 1\)/);
  });

  it('refuses mismatched survivor counts and row accounting', () => {
    const broken = syntheticComposite();
    broken.stage_one_calibration_raw = [-3, -1, -1];
    expect(() => new CompositeSurvivorPolicyV3(broken)).toThrow(
      /survivor count/,
    );
    const rows = syntheticComposite();
    rows.enrollment_rows = 7;
    expect(() => new CompositeSurvivorPolicyV3(rows)).toThrow(
      /gated \+ survivors/,
    );
  });

  it.each([
    [
      'legacy q95',
      2,
      'v3-staged-openset-policy-v2-composite-survivor',
      'v3_staged_noise_prefilter_then_composite_survivor_lof_geometry',
      0.95,
    ],
    [
      'failed q99',
      3,
      'v3-staged-openset-policy-v3-composite-survivor-q99',
      'v3_staged_noise_prefilter_then_q99_composite_survivor_lof_geometry',
      0.99,
    ],
  ])('refuses %s semantics in the active q97 primitive', (
    _label,
    schema,
    policyVersion,
    kind,
    quantile,
  ) => {
    const broken = syntheticComposite() as unknown as Record<string, unknown>;
    Object.assign(broken, {
      schema,
      policy_version: policyVersion,
      kind,
      threshold_quantile: quantile,
    });
    expect(() =>
      new CompositeSurvivorPolicyV3(
        broken as unknown as CompositeSurvivorPolicyAsset,
      )).toThrow(/staged policy version mismatch/);
  });

  it.each([
    ['threshold_population', 'selection'],
    ['training_rows_used_for_threshold', 1],
    ['selection_rows_used_for_threshold', 1],
    ['novelty_rows_used_for_threshold', 1],
    ['release_rows_used_for_threshold', 1],
    ['stage_one_known_false_positive_budget', 0.02],
    ['survivor_known_false_positive_budget', 0.04],
    ['nominal_enrollment_false_unknown_budget', 0.05],
    ['only_policy_change', 'anything else'],
    ['stage_one_changed', true],
    ['rejector_cnn_fusion_changed', true],
    ['classifier_cnn_fusion_changed', true],
    ['gate_contract_changed', true],
  ])('refuses mutated schema-4 hygiene field %s', (field, value) => {
    const broken = syntheticComposite() as unknown as Record<string, unknown>;
    broken[field] = value;
    expect(() =>
      new CompositeSurvivorPolicyV3(
        broken as unknown as CompositeSurvivorPolicyAsset,
      )).toThrow(new RegExp(field));
  });
});

describeCurrent('stage-1 evaluation shape', () => {
  it('exposes the prefilter internals for auditability', () => {
    const row = fixture.rows[0]!;
    const evaluation: StageOneEvaluation = openSet.evaluateStageOne(
      row.iq.in_phase,
      row.iq.quadrature,
    );
    expect(evaluation.features.length).toBe(7);
    expect(evaluation.captureLength).toBe(row.capture_length);
    expect(prefilterPoseDegeneracyFeatures(
      row.iq.in_phase,
      row.iq.quadrature,
      {
        minBandwidth: frontendMinimumBandwidth(
          row.capture_length,
          fixture.frontend.patch_length,
          fixture.frontend.target_frac,
        ),
      },
    ).length).toBe(7);
  });
});
