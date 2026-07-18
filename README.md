# AtomOS Classifier

Training, validation, and publication pipeline for Atomizer's Bayesian
observable classifier. Extracted out of `TinySA_Atomizer` so the
long-running (90+ minute) training/validation cycle has its own repo, its
own CI, and its own room to be optimized without weighing down the
Atomizer app's own build.

## What lives here

- `tools/train-observable-classifier.ts` — trains the model from
  `TinySA_SignalLab`'s corpus and publishes the generated model files into
  this repo's `src/models/` directory.
- `tools/validate-signal-lab-classifier.ts` — post-training validation
  against the generated model.
- `tools/verify-classifier-publication.mjs` — publication-integrity check
  (pins commit/corpus/model IDs, thresholds, hashes; also checks a few of
  TinySA's docs for consistency with the validated metrics).
- `tools/observable-training-*` — run control, attempt caching, worker-pool
  sampling, and build-attestation supporting the training run.
- `tools/validator-*` — validation helpers used by
  `validate-signal-lab-classifier.ts`.

The classifier-owned runtime inference code and generated model live in
`src/`. Atomizer imports that runtime from this sibling repo. Shared
measurement analysis—including observable feature extraction, Bayesian
predictive math, acquisition geometry, and contracts—remains in
`TinySA/packages/analysis/src` and is imported by this repo.

## Layout assumption

This repo expects to sit as a sibling of `TinySA` (Atomizer) and
`TinySA_SignalLab`:

```
PersonalGitHub/
├── AtomOS_Classifier/   (this repo)
├── TinySA/              (Atomizer)
└── TinySA_SignalLab/
```

All cross-repo imports are plain relative paths (`../../TinySA/packages/...`,
`../../TinySA_SignalLab/src/...`) — there is no npm dependency on either
sibling repo.

## Quick start

```
nvm install 22.23.1
nvm use 22.23.1
npm install
npm run typecheck
npm test
```

`npm run check` runs the full local gate (typecheck, unit tests, worker
self-test, model reproduction, validation, and publication check) — the
model-reproduction step alone can take 90+ minutes.

## Known CI gap

`TinySA_Atomizer` is currently a private repo. The CI workflow uses the
`CLASSIFIER_TINYSA_READ_TOKEN` repository secret for that checkout; configure
it with read-only contents access to `PhysicistJohn/TinySA_Atomizer` before
enabling CI in a new remote.
