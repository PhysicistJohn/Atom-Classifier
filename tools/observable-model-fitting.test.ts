import { describe, expect, it } from 'vitest';
import { canonicalClassificationScenario } from '../../Atom-SignalLab/src/classification-corpus.js';
import {
  OBSERVABLE_LIKELIHOOD_COMPONENT_DECOMPOSITION_POLICY,
  componentSourceScenarioId,
  deterministicOneDimensionalLloydPartition,
  fitScenarioStudentTComponents,
} from './observable-model-fitting.js';

const dimensions = [
  'spectrum.logClusterCount',
  'spectrum.powerVariationDb',
] as const;

describe('observable likelihood component fitting', () => {
  it('fits deterministic empirical-weight CSMA modes with one shared pooled covariance', () => {
    const scenario = canonicalClassificationScenario('wifi-hr-dsss-11m');
    const samples = [
      ...modeSamples(30, 0.8, 2.72, 0),
      ...modeSamples(15, 3.4, 2.62, 100),
      ...modeSamples(6, 11.1, 2.08, 200),
    ];

    const forward = fitScenarioStudentTComponents(
      scenario,
      samples,
      dimensions,
      0,
    );
    const reversed = fitScenarioStudentTComponents(
      scenario,
      [...samples].reverse(),
      dimensions,
      0,
    );

    expect(OBSERVABLE_LIKELIHOOD_COMPONENT_DECOMPOSITION_POLICY.csmaModeCount)
      .toBe(3);
    expect(forward).toEqual(reversed);
    expect(forward).toHaveLength(3);
    expect(forward.map((component) => componentSourceScenarioId(component)))
      .toEqual(Array(3).fill(scenario.id));
    expect(forward.map((component) => component.modeId)).toEqual([
      'csma-activity-mode-1-of-3',
      'csma-activity-mode-2-of-3',
      'csma-activity-mode-3-of-3',
    ]);
    expect(forward.map((component) => component.location[1]))
      .toEqual([...forward.map((component) => component.location[1])].sort((left, right) => left! - right!));
    expect(forward[0]!.scale).toEqual(forward[1]!.scale);
    expect(forward[1]!.scale).toEqual(forward[2]!.scale);
    expect(forward.reduce((sum, component) => sum + Math.exp(component.logWeight), 0))
      .toBeCloseTo(1, 12);
    expect(forward.map((component) => Math.exp(component.logWeight)))
      .toEqual([30 / 51, 15 / 51, 6 / 51]);
    expect(forward.map((component) => component.fitSampleCount)).toEqual([30, 15, 6]);

    const locations = forward.map((component) => component.location);
    const pooled = dimensions.map((rowDimension, row) => dimensions.map((columnDimension, column) => {
      const residualSum = [
        samples.slice(0, 30),
        samples.slice(30, 45),
        samples.slice(45),
      ].reduce((total, group, modeIndex) => total + group.reduce((sum, sample) => sum
        + (sample[rowDimension]! - locations[modeIndex]![row]!)
          * (sample[columnDimension]! - locations[modeIndex]![column]!), 0), 0);
      const covariance = residualSum / (samples.length - 3);
      const regularization = row === column
        ? (rowDimension === 'spectrum.logClusterCount' ? 0.06 ** 2 : 1.5 ** 2)
        : 0;
      return ((row === column ? covariance : covariance * 0.35) + regularization) * 5 / 7;
    }));
    forward[0]!.scale.forEach((row, rowIndex) => row.forEach((value, columnIndex) =>
      expect(value).toBeCloseTo(pooled[rowIndex]![columnIndex]!, 14)));
  });

  it('decomposes the fitted unknown IEEE 802.15.4 hard negative with the same owned architecture', () => {
    const scenario = canonicalClassificationScenario('unknown-802154');
    const samples = [
      ...modeSamples(12, 0.9, 2.1, 0),
      ...modeSamples(9, 3.2, 2.4, 100),
      ...modeSamples(6, 10.8, 2.8, 200),
    ];
    const components = fitScenarioStudentTComponents(scenario, samples, dimensions, Math.log(0.5));
    expect(components).toHaveLength(3);
    expect(components.map(componentSourceScenarioId)).toEqual(Array(3).fill('unknown-802154'));
    expect(components.reduce((sum, component) => sum + Math.exp(component.logWeight), 0))
      .toBeCloseTo(0.5, 12);
  });

  it('rejects a CSMA partition whose separated tail mode has fewer than three samples', () => {
    const scenario = canonicalClassificationScenario('wifi-hr-dsss-11m');
    const samples = [
      ...modeSamples(12, 0.8, 2.7, 0),
      ...modeSamples(9, 3.4, 2.5, 100),
      ...modeSamples(2, 20, 2.0, 200),
    ];
    expect(() => fitScenarioStudentTComponents(scenario, samples, dimensions, 0))
      .toThrow(/at least 3 are required/);
  });

  it('assigns an equal-distance sample to the lower-index mode deterministically', () => {
    const samples = [0, 0, 0, 2.5, 5, 5, 5, 10, 10, 10]
      .map((value, marker) => ({ value, marker }));
    const forward = deterministicOneDimensionalLloydPartition(samples, 'value', 3);
    const reversed = deterministicOneDimensionalLloydPartition([...samples].reverse(), 'value', 3);
    expect(forward).toEqual(reversed);
    expect(forward[0]!.some((sample) => sample.marker === 3)).toBe(true);
    expect(forward.map((group) => group.length)).toEqual([4, 3, 3]);
  });

  it('retains one scenario-owned component for a non-CSMA population', () => {
    const scenario = canonicalClassificationScenario('cw-rbw-line');
    const samples = modeSamples(6, 0.2, 1.1, 0);

    const components = fitScenarioStudentTComponents(
      scenario,
      samples,
      dimensions,
      Math.log(0.25),
    );

    expect(components).toHaveLength(1);
    expect(components[0]).toMatchObject({
      id: scenario.id,
      sourceScenarioId: scenario.id,
      modeId: 'single-population',
      logWeight: Math.log(0.25),
    });
  });
});

function modeSamples(
  count: number,
  powerVariationDb: number,
  logClusterCount: number,
  offset: number,
): Readonly<Record<string, number>>[] {
  return Array.from({ length: count }, (_unused, index) => ({
    'spectrum.logClusterCount': logClusterCount + ((index + offset) % 3 - 1) * 0.01,
    'spectrum.powerVariationDb': powerVariationDb + ((index + offset) % 5 - 2) * 0.02,
  }));
}
