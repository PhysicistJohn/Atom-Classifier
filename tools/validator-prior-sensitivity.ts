import {
  logSumExp,
  mixtureLogLikelihood,
  type PosteriorCandidate,
} from '../../TinySA/packages/analysis/src/bayesian-predictive.js';
import {
  observableClassSupportsEvidenceView,
  observableModelComponents,
  observableModelView,
  type ObservableClassifierModelAsset,
  type ObservableLeafClass,
} from '../src/observable-classifier-model.js';
import type { ObservableFeatureObservation } from '../../TinySA/packages/analysis/src/observable-features.js';
import { observableRepresentativeIsInClassDomain } from '../src/observable-hypothesis-domain.js';
import { BAYESIAN_OBSERVABLE_MODEL } from '../src/models/bayesian-observable.generated.js';

/** Recompute the generated-model posterior under a validator-declared prior. */
export function posteriorUnderDeclaredPrior(
  observation: ObservableFeatureObservation,
  prior: Readonly<Record<ObservableLeafClass, number>>,
  modelAsset: Pick<ObservableClassifierModelAsset, 'classModels'> = BAYESIAN_OBSERVABLE_MODEL,
): readonly PosteriorCandidate[] {
  const view = observableModelView(observation);
  const values = modelAsset.classModels.map((model) => {
    // Structural view support and the observation-only hypothesis domain are
    // logical support boundaries. Apply both before evaluating components so
    // an intentionally empty class/view population is never treated as a
    // malformed likelihood mixture.
    if (!observableClassSupportsEvidenceView(model.id, view)
      || !observableRepresentativeIsInClassDomain(model.id, observation)) {
      return {
        id: model.id,
        logLikelihood: Number.NEGATIVE_INFINITY,
        logJoint: Number.NEGATIVE_INFINITY,
      };
    }
    const logLikelihood = mixtureLogLikelihood(
      observation.values,
      observableModelComponents(model, view),
    );
    return {
      id: model.id,
      logLikelihood,
      logJoint: Math.log(prior[model.id]) + logLikelihood,
    };
  });
  const normalization = logSumExp(values.map((value) => value.logJoint));
  return values.map((value) => ({
    ...value,
    probability: Math.exp(value.logJoint - normalization),
  })).sort((left, right) => right.probability - left.probability);
}
