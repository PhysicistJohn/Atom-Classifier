import type { CanonicalClassificationScenario } from '../../Atom-SignalLab/src/classification-corpus.js';
import type { StudentTLikelihoodComponent } from '../../Atom-Atomizer/packages/analysis/src/bayesian-predictive.js';
import { OBSERVABLE_LIKELIHOOD_COMPONENT_DECOMPOSITION_POLICY } from '../src/observable-classifier-model.js';

export { OBSERVABLE_LIKELIHOOD_COMPONENT_DECOMPOSITION_POLICY } from '../src/observable-classifier-model.js';

const DEGREES_OF_FREEDOM = 7;
const OFF_DIAGONAL_COVARIANCE_RETENTION = 0.35;
const MINIMUM_MODE_SAMPLE_COUNT =
  OBSERVABLE_LIKELIHOOD_COMPONENT_DECOMPOSITION_POLICY.minimumModeFitSampleCount;

interface ModePopulation {
  readonly id: string;
  readonly samples: readonly Readonly<Record<string, number>>[];
  readonly location: readonly number[];
}

/**
 * Fit one scenario's view-matched likelihood population.
 *
 * The canonized CSMA schedule has three detector-observable activity states:
 * mostly steady occupancy, ordinary burst transitions, and a burst boundary
 * crossing much of the swept acquisition. A single elliptical component
 * moment-matches across those modes and can assign an exact fitted production
 * state to its radial tail. We therefore partition only the declared CSMA
 * populations on their directly observed sweep-to-sweep power variation,
 * retain empirical event-frequency weights, and estimate one shared pooled
 * within-mode covariance. The converged Lloyd partition is deterministically
 * rebalanced at adjacent boundaries when a rare tail would otherwise contain
 * fewer observations than the declared fit floor. Sharing covariance prevents
 * the smallest burst-boundary mode from estimating an independent
 * high-dimensional covariance.
 */
export function fitScenarioStudentTComponents(
  scenario: CanonicalClassificationScenario,
  samples: readonly Readonly<Record<string, number>>[],
  dimensions: readonly string[],
  scenarioLogWeight: number,
): readonly StudentTLikelihoodComponent[] {
  if (scenario.envelopeModel !== OBSERVABLE_LIKELIHOOD_COMPONENT_DECOMPOSITION_POLICY.csmaEnvelopeModel) {
    return [fitSingleComponent(
      scenario.id,
      scenario.id,
      'single-population',
      samples,
      dimensions,
      scenarioLogWeight,
    )];
  }

  const groups = deterministicOneDimensionalLloydPartition(
    samples,
    OBSERVABLE_LIKELIHOOD_COMPONENT_DECOMPOSITION_POLICY.csmaPartitionFeature,
    OBSERVABLE_LIKELIHOOD_COMPONENT_DECOMPOSITION_POLICY.csmaModeCount,
  );
  const partitionCenters = groups.map((group) => mean(group.map((sample) => finiteDimension(
    sample,
    OBSERVABLE_LIKELIHOOD_COMPONENT_DECOMPOSITION_POLICY.csmaPartitionFeature,
  ))));
  if (partitionCenters.some((center, index) => index > 0 && center <= partitionCenters[index - 1]!)) {
    throw new Error(`${scenario.id} CSMA activity-mode centers are not strictly increasing`);
  }
  const populations: readonly ModePopulation[] = groups.map((group, index) => {
    if (group.length < MINIMUM_MODE_SAMPLE_COUNT) {
      throw new Error(
        `${scenario.id} CSMA activity mode ${index + 1} has only ${group.length} observations; at least ${MINIMUM_MODE_SAMPLE_COUNT} are required`,
      );
    }
    return {
      id: `csma-activity-mode-${index + 1}-of-${groups.length}`,
      samples: group,
      location: componentLocation(group, dimensions),
    };
  });
  const sharedScale = pooledWithinModeScale(populations, dimensions);
  return populations.map((population) => ({
    id: `${scenario.id}/${population.id}`,
    sourceScenarioId: scenario.id,
    modeId: population.id,
    fitSampleCount: population.samples.length,
    logWeight: scenarioLogWeight + Math.log(population.samples.length / samples.length),
    degreesOfFreedom: DEGREES_OF_FREEDOM,
    dimensions,
    location: population.location,
    scale: sharedScale,
  }));
}

export function componentSourceScenarioId(
  component: Pick<StudentTLikelihoodComponent, 'id' | 'sourceScenarioId'>,
): string {
  return component.sourceScenarioId ?? component.id;
}

export function deterministicOneDimensionalLloydPartition(
  samples: readonly Readonly<Record<string, number>>[],
  dimension: string,
  modeCount: number,
): readonly (readonly Readonly<Record<string, number>>[])[] {
  if (!Number.isInteger(modeCount) || modeCount < 2) {
    throw new Error('Observable likelihood mode count must be an integer of at least two');
  }
  if (samples.length < modeCount * MINIMUM_MODE_SAMPLE_COUNT) {
    throw new Error(`Observable likelihood population has only ${samples.length} samples for ${modeCount} modes`);
  }
  const ordered = [...samples].sort((left, right) =>
    finiteDimension(left, dimension) - finiteDimension(right, dimension)
      || compareCodeUnitStrings(JSON.stringify(left), JSON.stringify(right)));
  let centers = Array.from({ length: modeCount }, (_unused, index) =>
    finiteDimension(
      ordered[Math.round(index * (ordered.length - 1) / (modeCount - 1))]!,
      dimension,
    ));
  let groups: Readonly<Record<string, number>>[][] = [];
  for (let iteration = 0; iteration < 100; iteration += 1) {
    groups = Array.from({ length: modeCount }, () => []);
    for (const sample of ordered) {
      const value = finiteDimension(sample, dimension);
      const closest = centers.reduce((best, center, index) =>
        Math.abs(value - center) < Math.abs(value - centers[best]!) ? index : best, 0);
      groups[closest]!.push(sample);
    }
    if (groups.some((group) => group.length === 0)) {
      throw new Error(`Observable likelihood ${dimension} partition produced an empty mode`);
    }
    const next = groups.map((group) => mean(group.map((sample) => finiteDimension(sample, dimension))));
    if (next.every((value, index) => value === centers[index])) {
      return rebalanceMinimumPopulation(groups, MINIMUM_MODE_SAMPLE_COUNT);
    }
    centers = next;
  }
  throw new Error(`Observable likelihood ${dimension} partition did not converge deterministically`);
}

function rebalanceMinimumPopulation(
  groups: readonly (readonly Readonly<Record<string, number>>[])[],
  minimumGroupSize: number,
): readonly (readonly Readonly<Record<string, number>>[])[] {
  const orderedSamples = groups.flatMap((group) => group);
  const minimumPopulation = groups.length * minimumGroupSize;
  if (orderedSamples.length < minimumPopulation) {
    throw new Error('Observable likelihood partition cannot satisfy its minimum mode population');
  }
  let offset = 0;
  let remaining = orderedSamples.length;
  return groups.map((group, index) => {
    const remainingGroupCount = groups.length - index - 1;
    const maximumSize = remaining - (remainingGroupCount * minimumGroupSize);
    const size = Math.min(Math.max(group.length, minimumGroupSize), maximumSize);
    const balanced = orderedSamples.slice(offset, offset + size);
    offset += size;
    remaining -= size;
    return balanced;
  });
}

function fitSingleComponent(
  id: string,
  sourceScenarioId: string,
  modeId: string,
  samples: readonly Readonly<Record<string, number>>[],
  dimensions: readonly string[],
  logWeight: number,
): StudentTLikelihoodComponent {
  if (samples.length < 3) throw new Error(`${id} requires at least three training observations`);
  const location = componentLocation(samples, dimensions);
  return {
    id,
    sourceScenarioId,
    modeId,
    fitSampleCount: samples.length,
    logWeight,
    degreesOfFreedom: DEGREES_OF_FREEDOM,
    dimensions,
    location,
    scale: regularizedScale(samples, location, dimensions),
  };
}

function componentLocation(
  samples: readonly Readonly<Record<string, number>>[],
  dimensions: readonly string[],
): readonly number[] {
  return dimensions.map((dimension) => mean(samples.map((sample) => finiteDimension(sample, dimension))));
}

function regularizedScale(
  samples: readonly Readonly<Record<string, number>>[],
  location: readonly number[],
  dimensions: readonly string[],
): readonly (readonly number[])[] {
  return dimensions.map((_rowDimension, row) => dimensions.map((_columnDimension, column) => {
    const covariance = samples.reduce((sum, values) => sum
      + (finiteDimension(values, dimensions[row]!) - location[row]!)
        * (finiteDimension(values, dimensions[column]!) - location[column]!), 0)
      / (samples.length - 1);
    return regularizeCovariance(covariance, dimensions[row]!, row === column);
  }));
}

function pooledWithinModeScale(
  populations: readonly ModePopulation[],
  dimensions: readonly string[],
): readonly (readonly number[])[] {
  const sampleCount = populations.reduce((sum, population) => sum + population.samples.length, 0);
  const degreesOfFreedom = sampleCount - populations.length;
  if (degreesOfFreedom <= 0) throw new Error('Pooled within-mode covariance has no residual degrees of freedom');
  return dimensions.map((_rowDimension, row) => dimensions.map((_columnDimension, column) => {
    const residualProduct = populations.reduce((total, population) => total
      + population.samples.reduce((sum, sample) => sum
        + (finiteDimension(sample, dimensions[row]!) - population.location[row]!)
          * (finiteDimension(sample, dimensions[column]!) - population.location[column]!), 0), 0);
    return regularizeCovariance(
      residualProduct / degreesOfFreedom,
      dimensions[row]!,
      row === column,
    );
  }));
}

function regularizeCovariance(
  covariance: number,
  rowDimension: string,
  diagonal: boolean,
): number {
  const regularizedCovariance = (diagonal
    ? covariance
    : covariance * OFF_DIAGONAL_COVARIANCE_RETENTION)
    + (diagonal ? regularizationVariance(rowDimension) : 0);
  // In this parameterization Cov[T_nu(0, scale)] = nu/(nu-2) * scale.
  return regularizedCovariance * (DEGREES_OF_FREEDOM - 2) / DEGREES_OF_FREEDOM;
}

function regularizationVariance(dimension: string): number {
  if (dimension === 'association.logBayesFactor') return 0.25 ** 2;
  if (dimension.includes('logBandwidth')) return 0.04 ** 2;
  if (dimension === 'spectrum.prominenceDb' || dimension === 'spectrum.powerVariationDb' || dimension === 'envelope.rangeDb' || dimension === 'envelope.standardDeviationDb') return 1.5 ** 2;
  if (dimension === 'envelope.logTransitionRateHz') return 0.08 ** 2;
  if (dimension === 'spectrum.logClusterCount') return 0.06 ** 2;
  return 0.035 ** 2;
}

function finiteDimension(sample: Readonly<Record<string, number>>, dimension: string): number {
  const value = sample[dimension];
  if (!Number.isFinite(value)) throw new Error(`Observable likelihood sample is missing finite ${dimension}`);
  return value!;
}

function mean(values: readonly number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function compareCodeUnitStrings(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}
