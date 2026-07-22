<p align="center"><img src="docs/brand/logo.jpg" alt="AtomOS Classifier" width="520"></p>

# AtomOS Classifier

AtomOS Classifier owns Atomizer's local RF-classification assets and runtimes.
The deployed path is a metric-embedding classifier for complex I/Q and swept
magnitude; the repository also retains the Bayesian scalar-observable
training, validation, publication, and regression pipeline. It was extracted
from `Atom-Atomizer` so classifier work has an independent lifecycle without
weighing down the application build.

## What lives here

- `src/embedding/`: deployed I/Q and magnitude preprocessing, recovery,
  inference, evidence fusion, and content-addressed model assets imported by
  Atomizer's Detect and I/Q workspaces.
- `tools/train-observable-classifier.ts`: trains the retained Bayesian model from
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

The classifier-owned runtimes and generated assets live in `src/`; Atomizer
imports them from the sibling repo.

- `src/embedding/embedding-runtime.ts` and `magnitude-classifier.ts`: the
  deployed local inference surfaces for complete complex I/Q and swept power.
- `src/embedding/assets/`: checked-in weights, prototypes, refiners, parity
  fixtures, and a content-addressed model manifest.
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

Shared scalar-observable analysis (feature extraction, Bayesian
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
npm --prefix ../Atom-Atomizer ci --ignore-scripts
npm --prefix ../Atom-Atomizer run build -w @tinysa/contracts
npm --prefix ../Atom-SignalLab ci --omit=dev --ignore-scripts
npm ci
npm run typecheck
npm test
```

`npm run typecheck && npm test` is the ordinary fast gate and covers the
deployed embedding runtime. `npm run check` additionally runs the retained
Bayesian worker, exact-model reproduction, validation, and publication gates;
it requires Node 22.23.1 and the exact pinned sibling commits recorded by the
model, and can take well over 90 minutes.

## CI

The `verify` job in `.github/workflows/ci.yml` runs on every push and pull
request. It checks out the exact Atomizer and SignalLab sources required by the
runtime tests, then runs typechecking, the deployed classifier unit suite, and
the dependency audit. Multi-hour Bayesian reproduction and validation remain
an explicit local/release gate rather than routine CI. All sibling repositories
are public, so no PAT or repository secret is required.

## Part of the AtomOS suite

- [Atom-Atomizer](https://github.com/PhysicistJohn/Atom-Atomizer): AI-native spectrum analyzer application.
- [Atom-Classifier](https://github.com/PhysicistJohn/Atom-Classifier): deployed local embedding classifier plus retained Bayesian RF research pipeline.
- [Atom-Firmware](https://github.com/PhysicistJohn/Atom-Firmware): reproducibly built tinySA firmware research and modernization.
- [Atom-Flasher](https://github.com/PhysicistJohn/Atom-Flasher): fail-closed firmware flasher.
- [Atom-NeptuneSDR-Twin](https://github.com/PhysicistJohn/Atom-NeptuneSDR-Twin): QEMU-backed firmware-executing digital twin of the NeptuneSDR/HAMGEEK P210.
- [Atom-SignalLab](https://github.com/PhysicistJohn/Atom-SignalLab): 3GPP and reference signal generation.
- [Atom-TinySA-Twin](https://github.com/PhysicistJohn/Atom-TinySA-Twin): Renode digital twin booting real ZS407 firmware.
- [Atom-Website](https://github.com/PhysicistJohn/Atom-Website): product site.
