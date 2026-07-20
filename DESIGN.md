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

## Honest limitations

- Trained and evaluated on synthetic I/Q. A small real held-out set is still
  needed to calibrate confidence, set the unknown threshold, and (optionally) do
  light domain adaptation before this feeds live decisions. This build produces
  the machinery, not a field-validated model.
- Inference-time I/Q requires the SDR path; the tinySA scalar path cannot feed it.
  The SDR instantaneous bandwidth caps the maximum classifiable signal width.
- OFDM-family protocols are modulation-degenerate by design; the embedding will
  not separate LTE/NR/Wi-Fi-OFDM without the bandwidth/band context.

## Module map

```
training/                 Python, GPU (MPS)
  rfgen.py                synthetic modulator bank + impairment channel
  preprocess.py           detect → downconvert → resample → normalize (+ context)
  model.py                1D-CNN metric embedding (PyTorch)
  dataset.py              episodic + contrastive samplers over impaired draws
  train.py                train on MPS; export weights JSON + ONNX + prototypes
  evaluate.py             closed-set / few-shot / open-set / robustness report
src/embedding/            TypeScript, inference (browser-native, zero-dep)
  iq-preprocess.ts        TS port of the DSP front-end (parity-tested)
  embedding-runtime.ts    deterministic forward pass over exported weights
  prototype-classifier.ts nearest-prototype + open-set + few-shot enroll
  embedding-evidence-fusion.ts   prototype confidence → Bayesian evidence view
  assets/                 committed exported model (weights + prototypes + manifest)
```
