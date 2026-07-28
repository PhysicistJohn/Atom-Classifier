import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  COMPOSITE_SURVIVOR_SCORE,
  COMPOSITE_THRESHOLD_QUANTILE,
  CompositeSurvivorPolicyV3,
  FROZEN_BRANCH_LOF_RANK_WEIGHT,
  FROZEN_GEOMETRY_WEIGHT,
  STAGE_ONE_SCORE_OFFSET,
  STAGED_POLICY_KIND,
  STAGED_POLICY_SCHEMA,
  STAGED_POLICY_VERSION,
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
  /** Null exactly when gated (staged policy version 2 survivor terms). */
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

// Tests consume the tracked runtime package, never mutable gitignored training
// output. The compact fixture preserves both fitted lengths and all decisions.
const staging = new URL('./assets-v3-staging/', import.meta.url);
const rawAsset = JSON.parse(
  readFileSync(new URL('time-domain-openset-weights-v1.json', staging), 'utf8'),
) as unknown;
const fixture = JSON.parse(
  readFileSync(new URL('time-domain-openset-smoke-v1.json', staging), 'utf8'),
) as ParityFixture;

const asset: TimeDomainOpenSetAssetV3 = loadTimeDomainOpenSetAssetV3(rawAsset);
const openSet = new TimeDomainOpenSetV3(asset);

function maxAbsDelta(actual: ArrayLike<number>, expected: number[]): number {
  expect(actual.length).toBe(expected.length);
  let worst = 0;
  for (let index = 0; index < expected.length; index++) {
    worst = Math.max(worst, Math.abs(actual[index]! - expected[index]!));
  }
  return worst;
}

describe('time-domain openset v3 asset', () => {
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

  it('pins the staged composite policy (version 2)', () => {
    expect(asset.composite.policy_version).toBe(STAGED_POLICY_VERSION);
    expect(asset.composite.kind).toBe(STAGED_POLICY_KIND);
    expect(asset.composite.schema).toBe(STAGED_POLICY_SCHEMA);
    expect(asset.composite.survivor_score).toBe(COMPOSITE_SURVIVOR_SCORE);
    expect(asset.composite.threshold_quantile).toBe(
      COMPOSITE_THRESHOLD_QUANTILE,
    );
    // The composite was fit against exactly this stage-2 state.
    expect(asset.composite.stage_two_threshold).toBe(
      asset.stage_two.policy.threshold,
    );
    // The fixture and the asset agree on the decision threshold.
    expect(fixture.staged_threshold).toBe(asset.composite.threshold);
    expect(fixture.policy_version).toBe(STAGED_POLICY_VERSION);
  });

  it('refuses an asset recording a different staged policy version', () => {
    const broken = JSON.parse(JSON.stringify(asset)) as {
      composite: { policy_version: string };
    };
    broken.composite.policy_version =
      'v3-staged-openset-policy-v1-stage2-only';
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

  it('reproduces the asset threshold from the asset calibration exactly', () => {
    expect(
      numpyLinearQuantile(
        asset.composite.composite_calibration_raw,
        COMPOSITE_THRESHOLD_QUANTILE,
      ),
    ).toBe(asset.composite.threshold);
  });
});

describe('stage-1 parity: pose-degeneracy prefilter', () => {
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

describe('stage-2 parity: branch LOF + frozen geometry blend', () => {
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
      // decision is made on the composite (staged policy version 2).
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

describe('staged decision parity', () => {
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
      // The decision threshold is the composite q95, on every row.
      expect(decision.threshold).toBe(row.staged_threshold);
      expect(decision.threshold).toBe(asset.composite.threshold);
      if (row.rejected_stage === 1) {
        // Gated rows have no stage-2 score at all: the short circuit is
        // real, and the gated axis is unchanged by policy version 2.
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

describe('stage-1 causal-prefix rule for long captures', () => {
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
});

describe('stage-1 evaluation shape', () => {
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
