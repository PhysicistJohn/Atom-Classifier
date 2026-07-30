/**
 * Verify Python -> TypeScript parity for one frozen v4 classifier/policy pair.
 *
 * Usage after compiling with tsup:
 *   node verify-v4-profile-bank-openset-parity.js \
 *     classifier.json policy.json openset-parity.json
 */

import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import {
  classifyTimeDomainOpenSetV4,
  instantaneousPhaseLinearityScoreV4,
  loadBoundTimeDomainOpenSetBundleV4,
  scoreTimeDomainOpenSetEmbeddingV4,
} from '../src/embedding/time-domain-profile-bank-openset-v4.js';
import { empiricalRank } from '../src/embedding/time-domain-openset-v3.js';
import type {
  TimeDomainRuntimeInputLengthV4,
} from '../src/embedding/time-domain-profile-bank-classifier-v4.js';
import type {
  TimeDomainPrototypeSourceV4,
} from '../src/embedding/time-domain-profile-routing-v4.js';

interface ExpectedCase {
  id: string;
  prototype_source: TimeDomainPrototypeSourceV4;
  runtime_input_length: TimeDomainRuntimeInputLengthV4;
  embedding: number[];
  pose_features: number[];
  expected: {
    observation_runtime_input_length: TimeDomainRuntimeInputLengthV4;
    predicted_public_class_index: number;
    predicted_public_class: string;
    winning_squared_distance: number;
    winning_prototype_index: number;
    supported_public_class_mask: boolean[];
    squared_public_class_distances: Array<number | null>;
    closest_prototype_indices: number[];
    stage_one_score: number;
    stage_one_predicted_class_rank: number;
    stage_one_pooled_rank: number;
    stage_one_rank: number;
    stage_two_rank: number;
    instantaneous_phase_linearity_score: number;
    instantaneous_phase_linearity_rank: number;
    composite_rank: number;
    threshold: number;
    rejected: boolean;
  };
}

interface Fixture {
  schema: string;
  schema_version: number;
  classifier_sha256: string;
  policy_sha256: string;
  uses_frequency_transform: false;
  tolerance: {
    absolute: number;
    relative: number;
  };
  rank_edge_cases: Array<{
    calibration: number[];
    query: number;
    expected_less_count: number;
    expected_rank: number;
  }>;
  threshold_comparison: {
    rule: string;
    equality_rejects: true;
  };
  exact_zero: {
    prototype_source: TimeDomainPrototypeSourceV4;
    runtime_input_length: TimeDomainRuntimeInputLengthV4;
    expected_disposition: 'unknown';
    expected_reason: 'exact_no_signal';
    closed_decision: null;
  };
  phase_linearity_raw_cases: Array<{
    id: string;
    runtime_input_length: TimeDomainRuntimeInputLengthV4;
    sample_encoding: 'float32-little-endian-base64';
    in_phase_f32le_base64: string;
    quadrature_f32le_base64: string;
    in_phase_sha256: string;
    quadrature_sha256: string;
    expected_score: number;
    property_contract: {
      score_minimum?: number;
      score_maximum?: number;
      reference_case?: string;
      maximum_reference_delta?: number;
    };
  }>;
  cases: ExpectedCase[];
}

function fail(message: string): never {
  throw new Error(message);
}

function near(
  observed: number,
  expected: number,
  fixture: Fixture,
  path: string,
): void {
  if (!Number.isFinite(observed) || !Number.isFinite(expected)) {
    fail(`${path} must be finite`);
  }
  const tolerance = fixture.tolerance.absolute
    + fixture.tolerance.relative * Math.abs(expected);
  if (Math.abs(observed - expected) > tolerance) {
    fail(`${path}: observed ${observed}, expected ${expected}, tolerance ${tolerance}`);
  }
}

function exactArray<T>(
  observed: ArrayLike<T>,
  expected: readonly T[],
  path: string,
): void {
  if (
    observed.length !== expected.length
    || expected.some((value, index) => observed[index] !== value)
  ) {
    fail(`${path} differs`);
  }
}

function decodeF32Le(
  encoded: string,
  expectedLength: number,
  expectedSha256: string,
  path: string,
): Float32Array {
  const raw = Buffer.from(encoded, 'base64');
  if (raw.byteLength !== expectedLength * Float32Array.BYTES_PER_ELEMENT) {
    fail(`${path} byte length differs`);
  }
  const digest = createHash('sha256').update(raw).digest('hex');
  if (digest !== expectedSha256) {
    fail(`${path} SHA-256 differs`);
  }
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const result = new Float32Array(expectedLength);
  for (let index = 0; index < expectedLength; index++) {
    result[index] = view.getFloat32(
      index * Float32Array.BYTES_PER_ELEMENT,
      true,
    );
  }
  return result;
}

const [classifierPath, policyPath, fixturePath] = process.argv.slice(2);
if (classifierPath === undefined || policyPath === undefined || fixturePath === undefined) {
  fail(
    'usage: verify-v4-profile-bank-openset-parity '
    + '<classifier.json> <policy.json> <fixture.json>',
  );
}

const fixture = JSON.parse(readFileSync(fixturePath, 'utf8')) as Fixture;
if (
  fixture.schema !== 'atomos.v4.time-domain-profile-bank.openset-browser-parity'
  || fixture.schema_version !== 4
  || fixture.uses_frequency_transform !== false
  || fixture.threshold_comparison.equality_rejects !== true
  || fixture.phase_linearity_raw_cases.length === 0
  || fixture.cases.length === 0
) {
  fail('unsupported or empty v4 open-set parity fixture');
}

const observedPhaseScores = new Map<string, number>();
for (const entry of fixture.phase_linearity_raw_cases) {
  if (
    entry.sample_encoding !== 'float32-little-endian-base64'
    || observedPhaseScores.has(entry.id)
  ) {
    fail(`${entry.id} raw phase-linearity encoding/id is invalid`);
  }
  const inPhase = decodeF32Le(
    entry.in_phase_f32le_base64,
    entry.runtime_input_length,
    entry.in_phase_sha256,
    `${entry.id} in-phase`,
  );
  const quadrature = decodeF32Le(
    entry.quadrature_f32le_base64,
    entry.runtime_input_length,
    entry.quadrature_sha256,
    `${entry.id} quadrature`,
  );
  const observed = instantaneousPhaseLinearityScoreV4(
    inPhase,
    quadrature,
  );
  near(
    observed,
    entry.expected_score,
    fixture,
    `${entry.id} raw phase-linearity score`,
  );
  const minimum = entry.property_contract.score_minimum;
  const maximum = entry.property_contract.score_maximum;
  if (
    (minimum !== undefined
      && (entry.expected_score < minimum || observed < minimum))
    || (maximum !== undefined
      && (entry.expected_score > maximum || observed > maximum))
  ) {
    fail(`${entry.id} raw phase-linearity property contract failed`);
  }
  observedPhaseScores.set(entry.id, observed);
}
for (const entry of fixture.phase_linearity_raw_cases) {
  const referenceId = entry.property_contract.reference_case;
  if (referenceId === undefined) continue;
  const tolerance = entry.property_contract.maximum_reference_delta;
  const reference = observedPhaseScores.get(referenceId);
  const observed = observedPhaseScores.get(entry.id);
  if (
    tolerance === undefined
    || reference === undefined
    || observed === undefined
    || Math.abs(observed - reference) > tolerance
  ) {
    fail(`${entry.id} raw phase-linearity reference property failed`);
  }
}

const classifierBytes = Uint8Array.from(readFileSync(classifierPath));
const policyBytes = Uint8Array.from(readFileSync(policyPath));
const bundle = await loadBoundTimeDomainOpenSetBundleV4(
  classifierBytes,
  policyBytes,
  {
    expectedClassifierSha256: fixture.classifier_sha256,
    expectedPolicySha256: fixture.policy_sha256,
  },
);

for (const edge of fixture.rank_edge_cases) {
  const observed = empiricalRank(edge.calibration, edge.query);
  near(observed, edge.expected_rank, fixture, 'rank edge');
  const lessCount = edge.calibration.filter((value) => value < edge.query).length;
  if (lessCount !== edge.expected_less_count) {
    fail('rank edge strict less-than count changed');
  }
}

for (const entry of fixture.cases) {
  const observed = scoreTimeDomainOpenSetEmbeddingV4(
    bundle,
    entry.embedding,
    entry.pose_features,
    {
      prototypeSource: entry.prototype_source,
      runtimeInputLength: entry.runtime_input_length,
      instantaneousPhaseLinearityScore:
        entry.expected.instantaneous_phase_linearity_score,
    },
  );
  const expected = entry.expected;
  const closed = observed.closedDecision;
  if (
    observed.prototypeSource !== entry.prototype_source
    || observed.observationInputLength
      !== expected.observation_runtime_input_length
    || observed.runtimeInputLength !== entry.runtime_input_length
    || closed.predictedPublicClassIndex !== expected.predicted_public_class_index
    || closed.predictedPublicClass !== expected.predicted_public_class
    || closed.winningPrototypeIndex !== expected.winning_prototype_index
    || (observed.disposition === 'unknown') !== expected.rejected
  ) {
    fail(`${entry.id}: categorical decision parity failed`);
  }
  exactArray(
    closed.supportedPublicClassMask,
    expected.supported_public_class_mask,
    `${entry.id} support`,
  );
  exactArray(
    closed.closestPrototypeIndices,
    expected.closest_prototype_indices,
    `${entry.id} closest prototypes`,
  );
  near(
    closed.winningSquaredDistance,
    expected.winning_squared_distance,
    fixture,
    `${entry.id} winning distance`,
  );
  for (
    let classIndex = 0;
    classIndex < expected.squared_public_class_distances.length;
    classIndex++
  ) {
    const expectedDistance = expected.squared_public_class_distances[classIndex];
    const observedDistance = closed.squaredPublicClassDistances[classIndex]!;
    if (expectedDistance === null) {
      if (observedDistance !== Number.POSITIVE_INFINITY) {
        fail(`${entry.id}: unsupported class ${classIndex} did not remain Infinity`);
      }
    } else {
      near(
        observedDistance,
        expectedDistance,
        fixture,
        `${entry.id} class ${classIndex} distance`,
      );
    }
  }
  near(observed.stageOneScore, expected.stage_one_score, fixture, `${entry.id} pose score`);
  near(
    observed.stageOnePredictedClassRank.value,
    expected.stage_one_predicted_class_rank,
    fixture,
    `${entry.id} predicted-class pose rank`,
  );
  near(
    observed.stageOnePooledRank.value,
    expected.stage_one_pooled_rank,
    fixture,
    `${entry.id} route-length-pooled pose rank`,
  );
  near(observed.stageOneRank.value, expected.stage_one_rank, fixture, `${entry.id} pose rank`);
  near(
    observed.stageOneRank.value,
    Math.max(
      observed.stageOnePredictedClassRank.value,
      observed.stageOnePooledRank.value,
    ),
    fixture,
    `${entry.id} effective pose-rank maximum`,
  );
  near(observed.stageTwoRank.value, expected.stage_two_rank, fixture, `${entry.id} distance rank`);
  near(
    observed.instantaneousPhaseLinearityScore,
    expected.instantaneous_phase_linearity_score,
    fixture,
    `${entry.id} phase-linearity score`,
  );
  near(
    observed.instantaneousPhaseLinearityRank.value,
    expected.instantaneous_phase_linearity_rank,
    fixture,
    `${entry.id} phase-linearity rank`,
  );
  near(observed.compositeRank, expected.composite_rank, fixture, `${entry.id} composite`);
  near(observed.threshold, expected.threshold, fixture, `${entry.id} threshold`);
}

const zero = new Float64Array(fixture.exact_zero.runtime_input_length);
const exactZero = classifyTimeDomainOpenSetV4(
  bundle,
  zero,
  zero,
  { prototypeSource: fixture.exact_zero.prototype_source },
);
if (
  exactZero.disposition !== fixture.exact_zero.expected_disposition
  || exactZero.reason !== fixture.exact_zero.expected_reason
  || exactZero.closedDecision !== fixture.exact_zero.closed_decision
) {
  fail('exact-zero parity failed');
}

console.log(
  `v4 open-set Python/TypeScript parity passed: ${fixture.cases.length} routed cases`,
);
