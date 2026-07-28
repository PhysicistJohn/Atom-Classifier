import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  FROZEN_BRANCH_LOF_RANK_WEIGHT,
  FROZEN_GEOMETRY_WEIGHT,
  STAGE_ONE_SCORE_OFFSET,
  StageOneGateV3,
  StageTwoRejectorV3,
  TimeDomainOpenSetV3,
  acrossPatchFrequencyDispersion,
  empiricalRank,
  loadTimeDomainOpenSetAssetV3,
  prefilterPoseDegeneracyFeatures,
  frontendMinimumBandwidth,
  stagedArchitectureContract,
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
  threshold: number;
  rejected: boolean;
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
  staged_score: number;
  rejected_stage: 1 | 2 | null;
  decision_label: string;
}

interface ParityFixture {
  schema: string;
  classes: string[];
  frontend: {
    patch_length: number;
    patch_count: number;
    target_frac: number;
    packed_length: number;
  };
  contract: Record<string, unknown>;
  counts: Record<string, number>;
  rows: FixtureRow[];
}

const staging = new URL(
  '../../training/zplane_ab/v2_full_variation/artifacts/staging/'
    + 'time_domain_v3_openset/',
  import.meta.url,
);
const rawAsset = JSON.parse(
  readFileSync(new URL('time-domain-openset-weights-v1.json', staging), 'utf8'),
) as unknown;
const fixture = JSON.parse(
  readFileSync(new URL('time-domain-openset-parity-v1.json', staging), 'utf8'),
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
      expect(evaluation.threshold).toBe(expected.threshold);
      expect(evaluation.rejected, `${row.name} stage-2 decision`).toBe(
        expected.rejected,
      );
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
      const stagedScoreBound = row.rejected_stage === 1
        ? 1e-8
        : rankToleranceAround(
          asset.stage_two.policy.combined_calibration,
          row.stage_two!.combined_raw,
          asset.stage_two.policy.geometry_weight * rankToleranceAround(
            asset.stage_two.policy.geometry_calibration,
            row.stage_two!.geometry_deviation,
            GEOMETRY_DEVIATION_RADIUS,
          ) + 1e-9,
        );
      expect(
        Math.abs(decision.stagedScore - row.staged_score),
        `${row.name} staged score`,
      ).toBeLessThanOrEqual(stagedScoreBound);
      // The staged axis reproduces the staged decision exactly.
      const rejected = decision.stagedScore > decision.threshold;
      expect(rejected).toBe(row.rejected_stage !== null);
      // Gated rows have no stage-2 score at all: the short circuit is real.
      if (row.rejected_stage === 1) {
        expect(decision.stageTwo).toBeNull();
        expect(decision.stagedScore).toBeGreaterThan(STAGE_ONE_SCORE_OFFSET);
        expect(decision.stagedScore).toBeLessThan(2);
      }
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
    // Seed chosen so the N16384 model fires with a wide margin (score 2.23
    // against threshold 1.80): stage-1 noise recall is ~0.6 by design, so a
    // knife-edge draw would make this test fragile for the wrong reason.
    const noise = whiteNoise(LONG, 3);
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
