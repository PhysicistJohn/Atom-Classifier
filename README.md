<p align="center"><img src="docs/brand/logo.jpg" alt="AtomOS Classifier" width="520"></p>

# AtomOS Classifier

AtomOS Classifier is a Bayesian observable classifier for RF waveforms: the
training, validation, and publication pipeline, plus the runtime inference
code and generated model that the Atomizer app imports. It was extracted out
of `Atom-Atomizer` so the long-running (90+ minute) training/validation cycle
has its own repo, its own CI, and its own room to be optimized without
weighing down the Atomizer app's build.

## What lives here

- `tools/train-observable-classifier.ts`: trains the model from
  `Atom-SignalLab`'s classification corpus and publishes the generated model
  files into this repo's `src/models/` directory.
- `tools/validate-signal-lab-classifier.ts`: post-training validation
  against the generated model.
- `tools/verify-classifier-publication.mjs`: publication-integrity check
  (pins commit/corpus/model IDs, thresholds, hashes; also checks a few of
  Atom-Atomizer's docs for consistency with the validated metrics).
- `tools/observable-training-*`: run control, attempt caching, worker-pool
  sampling, and build attestation supporting the training run.
- `tools/validator-*`: validation helpers used by
  `validate-signal-lab-classifier.ts` (numeric reporting, prior sensitivity,
  receipt-qualified capture, capture-target projection).

## Runtime inference (`src/`)

The classifier-owned runtime code and the generated model live in `src/`;
Atomizer imports this runtime from the sibling repo.

- `src/bayesian-waveform-classifier.ts`: Student-t mixture posterior
  inference over 12 leaf classes (CW, AM, FM, GSM, LTE FDD/TDD, NR FDD/TDD,
  Wi-Fi HR-DSSS, Wi-Fi OFDM, Bluetooth, and unknown-signal) across three
  evidence views (spectrum-only, envelope-untimed, envelope-timed).
- `src/observable-classifier-model.ts`: model schema, evidence-censoring
  policy, and likelihood-component decomposition policy (including the
  CSMA burst-activity modes).
- `src/radio-operating-band-context.ts`: versioned operating-band tables
  used only as a structural support mask, not as a deployment prior.
- `src/models/bayesian-observable.generated.ts` and its manifest: the
  published model asset, pinned by SHA-256 and regenerated only by the
  training tool.

Shared measurement analysis (observable feature extraction, Bayesian
predictive math, acquisition geometry, and contracts) remains in
`Atom-Atomizer/packages/analysis/src` and is imported by this repo.

## Layout assumption

This repo expects to sit as a sibling of `Atom-Atomizer` (Atomizer) and
`Atom-SignalLab`:

```
PersonalGitHub/
├── Atom-Classifier/   (this repo)
├── Atom-Atomizer/     (Atomizer)
└── Atom-SignalLab/
```

All cross-repo imports are plain relative paths
(`../../Atom-Atomizer/packages/...`, `../../Atom-SignalLab/src/...`).
There is no npm dependency on either sibling repo.

## Quick start

```
nvm install 22.23.1
nvm use 22.23.1
npm --prefix ../Atom-Atomizer ci --omit=dev --ignore-scripts
npm --prefix ../Atom-SignalLab ci --omit=dev --ignore-scripts
npm install
npm run typecheck
npm test
```

`npm run check` runs the full local gate (typecheck, worker self-test, unit
tests, model reproduction, validation, and publication check). The
model-reproduction step alone can take 90+ minutes.

## CI

The `verify` job in `.github/workflows/ci.yml` runs unconditionally on every
push and pull request. It checks out `Atom-Atomizer` and `Atom-SignalLab` at
the exact pinned commits the checked-in model was trained and validated
against, then runs the same gate as `npm run check`. All sibling repos are
public, so the checkouts use the default `GITHUB_TOKEN` and no PAT or
repository secret is required. Expect the full run to take well over 90
minutes; the model-reproduction step dominates.

## Part of the AtomOS suite

- [Atom-Atomizer](https://github.com/PhysicistJohn/Atom-Atomizer): AI-native spectrum analyzer app.
- [Atom-Classifier](https://github.com/PhysicistJohn/Atom-Classifier): this repo.
- [Atom-Firmware](https://github.com/PhysicistJohn/Atom-Firmware): reverse-engineered, LLVM cross-built TinySA firmware.
- [Atom-Flasher](https://github.com/PhysicistJohn/Atom-Flasher): fail-closed firmware flasher.
- [Atom-NeptuneSDR-Twin](https://github.com/PhysicistJohn/Atom-NeptuneSDR-Twin): Renode digital twin of the NeptuneSDR.
- [Atom-SignalLab](https://github.com/PhysicistJohn/Atom-SignalLab): 3GPP and reference signal generation.
- [Atom-TinySA-Twin](https://github.com/PhysicistJohn/Atom-TinySA-Twin): Renode digital twin booting real ZS407 firmware.
- [Atom-Website](https://github.com/PhysicistJohn/Atom-Website): product site.
