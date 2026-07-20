# Metric-embedding few-shot waveform classifier

Status: implemented on branch `feat/iq-embedding-fewshot`. This document is the
build spec and the record of the design decisions behind it.

## Why this exists

The shipping classifier (`src/bayesian-waveform-classifier.ts`) is a Bayesian
Student-t mixture posterior over 12 leaf classes, fed by hand-crafted scalar
observables extracted from a swept scalar spectrum (tinySA-class hardware). It is
calibrated, honest about uncertainty, and reproducible — but it is *closed-set*
and *fixed-taxonomy*: teaching it a new waveform means a new observable model and
a 90-minute retrain, and it has no learned notion of modulation *structure*
beyond the observables we chose by hand.

This subsystem adds a complementary capability that the I/Q path (NeptuneSDR)
unlocks and the scalar path cannot: a **learned metric embedding of complex I/Q**
that

1. classifies modulation by **nearest prototype** in an embedding space,
2. **enrolls a new class from 1–5 examples** with no retraining (few-shot),
3. reports **"unknown"** honestly when a signal is far from every prototype
   (open-set), and
4. fuses into the existing Bayesian posterior as **one additional evidence view**,
   preserving the product's calibrated-uncertainty and provenance ethos.

It is *not* a replacement for the Bayesian model. It is an evidence source that is
strong exactly where the scalar observables are weak (fine modulation structure)
and that degrades to "unknown" rather than guessing.

## The core idea: impairments *are* the augmentation

We do not collect years of labelled SDR captures. We **generate** impaired I/Q
from a signal model (the same philosophy as Atom-SignalLab) and train a metric
embedding with a **prototypical / supervised-contrastive** objective:

- Two independent impairment draws of the *same* modulation (different SNR,
  carrier offset, fading, IQ imbalance, phase noise, timing, power, occupied
  bandwidth) are a **positive pair**.
- Draws of *different* modulations are **negatives**.

So the impairment model does double duty: it is our realism model *and* our
contrastive augmentation. The embedding is pushed to be **invariant** to
real-world nuisance variation and **sensitive** to modulation structure — which
is precisely the invariance a deployable classifier needs.

## Handling varying widths, powers, and locations

A learned metric over raw I/Q is only meaningful if the input is normalized, so a
DSP front-end runs *before* the embedding and decouples the nuisance geometry:

1. **Detect** — estimate center frequency and occupied bandwidth from the PSD.
2. **Down-convert** to baseband (remove coarse CFO) → frequency-invariance.
3. **Resample** to a canonical fractional bandwidth (fixed samples/symbol target)
   → scale-invariance across signal widths.
4. **Amplitude-normalize** (unit RMS) → power-invariance.

The *measured* bandwidth and the operating-band context are **kept as separate
scalars**, not fed into the embedding. This is deliberate: modulations that share
structure but differ in deployment — OFDM at 20 MHz in an LTE band vs. Wi-Fi OFDM
— are indistinguishable *by modulation alone*. The embedding names the modulation;
the bandwidth + band context disambiguate the protocol at fusion time (the same
role the band-context support mask already plays in the Bayesian model).

Bandwidth estimation is imperfect, so the embedding is trained with **scale
jitter** (± BW-estimate error on the resample target). Detection accuracy is the
load-bearing risk; the open-set "unknown" valve is the safety net when it fails.

## Architecture: GPU train → export → TS infer

The shipping runtime is browser-native, deterministic, zero-runtime-dependency,
and ships a tiny pinned asset. A neural embedding does not change that contract —
it splits cleanly across the exact seam the product already has:

- **Training (offline, Python + PyTorch on the Apple GPU / MPS).** Heavy, runs
  once, produces a compact weight set. Lives in `training/`.
- **Inference (TypeScript, runs where the Bayesian classifier runs).** The export
  is a small JSON weight blob; a hand-written, deterministic forward pass mirrors
  the PyTorch model exactly (a bit-for-bit parity test guards this). No
  onnxruntime, no WASM, no GPU at inference — same deployment model as today.
  Lives in `src/embedding/`. (An ONNX export is also produced for the eventual
  onnxruntime-web path, but the pure-TS path is primary.)

The embedding is small enough (a 3-layer 1D CNN → global pool → 2-layer head,
~50k parameters, 32-d output) that the pure-TS forward pass is both trivial and
fast, and the exported asset is a few hundred KB.

## Modulation taxonomy (I/Q-separable families)

The embedding classifies *modulation families*, which map to the app taxonomy via
the bandwidth/band context at fusion:

- **Known (trained, prototypes enrolled):** CW, AM, FM, BPSK, QPSK, 16QAM, 64QAM,
  GFSK (2-FSK), OFDM.
- **Held-out for few-shot (never trained; enrolled from K shots at test):** 8PSK,
  DSSS.
- **Open-set novelty (never enrolled; must read as unknown):** band-limited noise,
  and a chirp/LFM never shown in training.

## Training objective

**Prototypical episodic loss (ProtoNet).** Each episode samples N classes with K
support + Q query examples; prototypes are the support means; queries are
classified by negative squared Euclidean distance to prototypes; cross-entropy.
This directly optimizes the *nearest-prototype* decision rule used at inference,
so few-shot enrollment is not an afterthought — it is what the loss trains. A
supervised-contrastive auxiliary term tightens intra-class clusters.

## Fusion into the Bayesian posterior

The embedding does not overwrite the Bayesian decision. It contributes a
**likelihood over leaf classes** derived from prototype distances (softmax over
negative distance, temperature-calibrated on held-out data), gated by the
open-set score: when the nearest-prototype distance exceeds the unknown
threshold, the view abstains (uniform likelihood) rather than injecting a
confident wrong vote. The band/bandwidth context maps the modulation-family
posterior onto the leaf taxonomy. This mirrors how the existing model treats
each evidence view as a censored, independently-admissible likelihood.

## Acceptance thresholds (asserted by `training/evaluate.py` + the TS tests)

Evidence-based floors, met by the shipped model on held-out synthetic I/Q. The
measured value from the current model is in parentheses. The floors are honest
about what a sync-free front-end can and cannot do (see "Honest limitations").

- **Closed-set, fine** (9 classes, nearest prototype, SNR 2–30 dB): ≥ 0.72 (0.78).
- **Closed-set, family** (16/64-QAM merged — see below): ≥ 0.82 (0.86).
- **Closed-set, fine, SNR ≥ 18 dB**: ≥ 0.78 (0.83).
- **Few-shot re-enrollment** (leave-one-known-class-out, K=5, classes the metric
  resolves): ≥ 0.85 (0.92). Well-separated classes recover to 0.93–0.98 from five
  shots with no retraining; the hard pairs stay hard whether trained or enrolled.
- **Open-set AUROC** (known-test vs. novel, nearest-prototype distance): ≥ 0.72
  (0.83 overall; 0.84 chirp, 0.83 noise).
- **TS↔Python parity:** max abs embedding difference ≤ 1e-4 on identical input
  (measured BN-fold parity 1.5e-7; the TS test asserts ≤ 1e-4 end-to-end).

### Why family-level, and where the ceiling is

The dominant residual error is **16-QAM vs 64-QAM** (and, for few-shot, **8PSK vs
QPSK**). These differ only in constellation *order*; the discriminating statistic
(normalised 4th-order cumulant) separates the classes by ~0.06 while per-capture
impairment/estimation spread is ~0.095 (d′ ≈ 0.5). That is an *information* limit,
not a representation limit — no richer latent space, GNN, or transformer recovers
it, because the bits are not in the input at this fidelity. The levers that move
it are DSP, not ML: **symbol-timing + carrier recovery** (sample at symbol centres
instead of ISI transitions), longer dwell (cumulant variance ~1/N), or higher SNR.
The current front-end deliberately omits synchronisation to stay portable and
zero-dependency in the browser, so the honest behaviour is: classify confidently
at the modulation *family* level, keep the *order* uncertain, and let the
bandwidth/band context + Bayesian fusion resolve the protocol. This is the
calibrated-uncertainty ethos, not a defect to paper over.

## Pushing through the order limit (quality-gated recovery)

The embedding's order ceiling is a *per-capture* information limit, so the way
through is to **add information** and **normalise out the hardware** — not a
fancier net. A second, hierarchical stage does exactly that, and it is
opportunistic: it resolves the order when the capture supports it and honestly
**defers** when it does not, which is what makes it robust across arbitrary
hardware quality.

The levers, measured on the 16/64-QAM cumulant separation d′ (reproducer:
`training/experiments/dprime.py`):

- **Multipath destroys order** — d′ collapses from **5.9 (clean) to 0.05** with
  no equalization. Dwell and timing sync cannot fix it (multipath is a *bias*,
  not variance — both plateau).
- **Blind equalization is the lever** — a fractionally-spaced CMA equalizer with
  IQ-imbalance compensation, *no channel knowledge*, recovers **0.05 → 2.99**
  (oracle channel-inverse ceiling 5.14). The T/2 equalizer absorbs symbol timing.
- **Dwell** adds d′ ∝ √N (1.4 → 6.8 over 128 → 8192 symbols); **timing** ~1.6× on
  top. Cheap, hardware-agnostic.
- **IQ-imbalance compensation matters** — a cheap direct-conversion front-end
  leaks a mirror-image (widely-linear) term the linear CMA can't invert; the
  blind properness-restoring correction lifts d′ from **0.71 → 1.83 at 18 dB**.

The order refiner (`training/order_refine.py`) turns this into a **quality-gated,
hierarchical** decision, evaluated **held-out** (randomized channel family + real
IQ imbalance + DC; fold-split calibration; the hard 16-vs-64 *pair* reported
apart from the easy QPSK). Order is decided only when an **order-agnostic** gate
— a blind in-band SNR estimate and the residual-ISI symbol-autocorrelation (not a
modulus-dispersion metric, which covaries with order) — says the constellation
supports it:

| SNR (dB) | 16/64-pair accuracy | defer |
|---|---|---|
| 25 | 0.83 | 36% |
| 18 | 0.72 (intrinsic dip) | 41% |
| 12 | 0.92 | 61% |
| ≤ 8 | — | 100% defer |

**Aggregate single-capture 16/64 accuracy 0.81** — vs the embedding's
unconditional 0.4/0.64 — with no low-SNR leakage. 16/64 is intrinsically
~0.72–0.83 per capture, so:

- **Multi-look accumulation** (`OrderBelief`, `src/embedding/order-accumulator.ts`)
  fuses reliability-weighted order evidence across looks of a persistent emitter
  and drives it to certainty: **0.80 → 0.98 over 16 looks at 18 dB**. This is the
  same log-linear evidence accumulation the Bayesian classifier already uses,
  applied across time per emitter track.

### Where each stage runs

The recovery stage is *feedback DSP* (adaptive equalizer loops, data-dependent
branching) — it cannot be made bit-exact between the Python trainer and the TS
runtime, and it does not need to be: it runs **once at capture ingestion** (the
NeptuneSDR path), not in the browser render loop. Its output — an order
posterior plus a quality/SNR gate result — is compact data that flows to the
browser. The metric embedding, prototypes, and fusion stay **browser-native**.
So the seam is clean: heavy adaptive DSP at ingestion, light metric inference
in-browser. `src/embedding/order-refinement.ts` consumes the ingestion result
and combines it with the embedding's family call: a resolved order wins; a
deferred order reports the *family* ("linear-digital, order-unresolved") rather
than trusting the embedding's unreliable order guess.

### Built vs. follow-on

Built: blind equalization + timing (CMA FSE), **blind IQ-imbalance/DC
compensation** (properness restoration), the order-agnostic quality gate, and
**sequential Bayesian multi-look** accumulation.

Follow-on: **device-invariant training** — full-range impairment augmentation + a
domain-adversarial (DANN) head + per-device few-shot calibration, so one model
spans a $30 RTL-SDR and a $3k USRP (training-side; needs a small *real*
multi-device set to validate — the standing sim-to-real caveat). An
**augmented (widely-linear) CMA** would fold IQ-imbalance and channel into one
adaptive filter (the current first-order properness correction leaves residual).

## Honest limitations

- Trained and evaluated on synthetic I/Q. A small real held-out set is still
  needed to calibrate confidence, set the unknown threshold, and (optionally) do
  light domain adaptation before this feeds live decisions. This build produces
  the machinery, not a field-validated model.
- Inference-time I/Q requires the SDR path; the tinySA scalar path cannot feed it.
  The SDR instantaneous bandwidth caps the maximum classifiable signal width.
- OFDM-family protocols are modulation-degenerate by design; the embedding will
  not separate LTE/NR/Wi-Fi-OFDM without the bandwidth/band context.
- **Recovery needs a continuous, stationary dwell.** The CMA equalizer and the
  cumulant estimators assume a persistent single-carrier signal over the window;
  bursty packet traffic (GSM/BT/Wi-Fi bursts) would need burst detection +
  concatenation before recovery.
- **First-order IQ compensation leaves residual.** The properness correction is
  first-order; strong imbalance leaves a residual that an augmented CMA would
  remove. And the quality/SNR gate is calibrated on synthetic estimator noise —
  it needs recalibration on real captures before it gates live decisions.
- **Single-capture 16/64 order is intrinsically ~0.72–0.83.** Certainty comes
  from multi-look accumulation over a persistent emitter, which assumes the
  emitter's modulation is stable across the fused looks (frequency-hoppers /
  adaptive-modulation break that) and that looks are decorrelated enough that the
  effective look count approaches the raw count.

## Module map

```
training/                 Python (embedding: GPU/MPS; recovery: ingestion DSP)
  rfgen.py                synthetic modulator bank + impairment channel
  preprocess.py           detect → downconvert → resample → normalize (+ context)
  model.py                1D-CNN metric embedding (PyTorch)
  dataset.py              episodic + contrastive samplers over impaired draws
  train.py                train on MPS; export weights JSON + ONNX + prototypes
  evaluate.py             closed-set / few-shot / open-set / robustness report
  recover.py              blind IQ-comp + CMA equalize + sync + SNR/ISI gate
  order_refine.py         quality-gated order refiner, held-out eval, multi-look
  experiments/dprime.py   reproducer for the push-through d′ numbers
src/embedding/            TypeScript, inference (browser-native, zero-dep)
  iq-preprocess.ts        TS port of the DSP front-end (parity-tested)
  embedding-runtime.ts    deterministic forward pass over exported weights
  prototype-classifier.ts nearest-prototype + open-set + few-shot enroll
  embedding-evidence-fusion.ts   prototype confidence → Bayesian evidence view
  order-refinement.ts     combine ingestion order result with the family call
  order-accumulator.ts    multi-look Bayesian order accumulation (OrderBelief)
  embedding-classifier.ts public facade (normalize → embed → classify → fuse)
  assets/                 committed model (weights, prototypes, order-refiner, manifest)
```
