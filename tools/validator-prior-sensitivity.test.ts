import { describe, expect, it } from 'vitest';
import { BAYESIAN_OBSERVABLE_MODEL } from '../src/models/bayesian-observable.generated.js';
import {
  OBSERVABLE_LEAF_CLASSES,
  observableModelComponents,
  type ObservableClassifierModelAsset,
  type ObservableEvidenceView,
  type ObservableLeafClass,
} from '../src/observable-classifier-model.js';
import type { ObservableFeatureObservation } from '../../TinySA/packages/analysis/src/observable-features.js';
import { posteriorUnderDeclaredPrior } from './validator-prior-sensitivity.js';

describe('validator engineering-prior posterior replay', () => {
  it('does not evaluate a structurally unsupported empty class/view mixture', () => {
    const view: ObservableEvidenceView = 'envelope-untimed';
    const sourceModel = BAYESIAN_OBSERVABLE_MODEL.classModels.find(
      (model) => model.id === 'cw-like',
    )!;
    const sourceComponent = observableModelComponents(sourceModel, view)[0]!;
    const observation: ObservableFeatureObservation = {
      values: Object.fromEntries(sourceComponent.dimensions.map(
        (dimension, index) => [dimension, sourceComponent.location[index]!],
      )),
      limitations: [],
      occupiedStartHz: 99_999_500,
      occupiedStopHz: 100_000_500,
      centerHz: 100_000_000,
      bandwidthHz: 1_000,
      binWidthHz: 100,
      sweepIds: Array.from({ length: 8 }, (_, index) => `prior-regression-${index}`),
      views: ['scalar-spectrum', 'detected-power-envelope'],
    };
    const modelAsset = {
      classModels: BAYESIAN_OBSERVABLE_MODEL.classModels.map((model) =>
        model.id === 'bluetooth-like'
          ? {
              ...model,
              componentsByView: {
                ...model.componentsByView!,
                [view]: [],
              },
            }
          : model),
    } satisfies Pick<ObservableClassifierModelAsset, 'classModels'>;
    const bluetoothModel = modelAsset.classModels.find(
      (model) => model.id === 'bluetooth-like',
    )!;
    const prior = Object.fromEntries(OBSERVABLE_LEAF_CLASSES.map((id) => [
      id,
      Math.exp(BAYESIAN_OBSERVABLE_MODEL.classModels.find((model) => model.id === id)!.logPrior),
    ])) as Record<ObservableLeafClass, number>;

    expect(observableModelComponents(bluetoothModel, view)).toEqual([]);

    const posterior = posteriorUnderDeclaredPrior(observation, prior, modelAsset);
    const bluetooth = posterior.find((candidate) => candidate.id === 'bluetooth-like');

    expect(bluetooth).toMatchObject({
      probability: 0,
      logLikelihood: Number.NEGATIVE_INFINITY,
      logJoint: Number.NEGATIVE_INFINITY,
    });
    expect(posterior.reduce((sum, candidate) => sum + candidate.probability, 0))
      .toBeCloseTo(1, 12);
  });
});
