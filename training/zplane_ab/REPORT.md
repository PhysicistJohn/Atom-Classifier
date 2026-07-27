# Z-Plane Operator vs. Production CNN -- A/B on Atom-Classifier's SignalLab corpus

Seed 20260721, 7-class taxonomy (am, bluetooth, cw, dsss, fm, gsm, ofdm), identical split (common_split.py: ~30%/30%/40% of clean realizations -> enroll/val/train, all impaired realizations -> train), identical eval protocol (evaluate_ab.py).

CNN baseline is a FRESH retrain under this harness (training/zplane_ab/train_cnn_baseline_fresh.py), not the shipped src/embedding/assets/model-manifest.json number, per the reproducibility plan: both numbers come from the identical harness/split run back-to-back in this experiment, not one historical number and one fresh one.

The fresh CNN retrain reached clean_closed_set_accuracy = 0.920, close to the shipped model-manifest.json's 0.9124 (a few points of drift is expected from Adam/dropout non-determinism even at a fixed seed) -- confirming the reproduction harness itself is faithful before trusting the z-plane number computed the same way.

| Metric | CNN (fresh baseline) | Z-Plane operator | Delta |
|---|---|---|---|
| clean_closed_set_accuracy | 0.920 | 0.912 | -0.0073 |
| per_class: am | 1.000 | 0.947 | -0.053 |
| per_class: bluetooth | 1.000 | 0.938 | -0.062 |
| per_class: cw | 1.000 | 1.000 | 0.0 |
| per_class: dsss | 0.857 | 0.857 | 0.0 |
| per_class: fm | 1.000 | 1.000 | 0.0 |
| per_class: gsm | 0.611 | 0.667 | 0.056 |
| per_class: ofdm | 0.952 | 0.952 | 0.0 |
| fewshot loo_k1 (mean over classes) | 0.734 | 0.648 | -0.086 |
| fewshot loo_k5 (mean over classes) | 0.858 | 0.748 | -0.11 |
| param count | 38208 | 56256 | 18048 |
| episodes actually trained | 6000 | 1500 | -4500 |
| wall-clock training time (s, CPU) | 118.500 | 614.500 | 496.0 |
| embed_all inference latency (ms/1k samples) | 138.833 | 2610.916 | 2472.083 |

## Not computable / not applicable in this experiment

- `closed_family`, `per_snr_fine`/`per_snr_family`, `closed_fine_snr>=18`: this corpus has NO SNR metadata per item (only sampleRateHz/bandwidthHz/impaired/profile -- verified against corpus.json) and no QAM-order classes exist in this 7-class taxonomy, so these legacy evaluate.py metrics have no analog here.
- `open.auroc_*`, `open.flagged_unknown_*`: this corpus has NO held-out/novel class beyond the 7 trained classes (unlike rfgen's FEWSHOT=[psk8,dsss]/NOVEL=[noise,chirp]) -- open-set AUROC cannot be honestly computed without fabricating a novel-class signal source, which would not be apples-to-apples with either model's actual training distribution.

## Methodology

- **Corpus**: read-only reference to `training/artifacts/signallab-corpus/{corpus.json,corpus.f32}` (1811 items, 7 classes: am, bluetooth, cw, dsss, fm, gsm, ofdm) -- never modified.
- **Split**: both runs call `training/zplane_ab/common_split.py`'s `load_and_split()` with `SEED=20260721`, RNG call order copied verbatim (unreordered) from `train_signallab.py`: `torch.manual_seed` -> `np.random.seed` -> `np.random.default_rng` -> `load_corpus()` (no RNG use) -> the clean/impaired permutation split. Splits by realization: ~30% clean -> enroll, ~30% clean -> validation (held out, never trained on), remaining clean + all impaired -> train.
- **Preprocessing**: `training/preprocess.py`'s `preprocess()`/`to_channels()`/`iq_features()` imported directly, unmodified, for both models -- identical front end (detect -> downconvert -> resample -> amplitude-normalize -> [2,1024] I/Q + 12-d z-scored cumulant/inst-freq features) feeds both backbones.
- **Episodic training**: `training/dataset.py`'s `class_indices()`/`sample_episode()` imported directly. 7-way 5-shot/5-query prototypical cross-entropy, Adam (lr=1e-3, wd=2e-4) + CosineAnnealingLR + a learned scalar logit-scale, identical for both models except the z-plane run adds `clip_grad_norm_(net.parameters(), 1.0)` as a safety net for the untested complex-valued backbone.
- **Device**: forced `torch.device("cpu")` explicitly in both training scripts (not production's MPS auto-detect), so timings and any device-specific float-op noise are comparable between the two runs.
- **Evaluation**: `training/zplane_ab/evaluate_ab.py`, a single evaluator applied to BOTH trained net objects directly (no numpy/export path -- there is no shipped z-plane export). Reuses `train.py`'s `embed_all`/`prototypes_from`/`nearest` unchanged. Does NOT reuse `training/evaluate.py`, which is confirmed stale/incompatible with the current 7-class signallab corpus (it synthesizes test signals via `rfgen`, whose taxonomy has no bluetooth/gsm/dsss-as-signallab-defines-it modulator, and crashes with `ValueError: unknown class bluetooth` if pointed at the current prototypes).
- **Few-shot LOO protocol**: for each of the 7 classes, its enrolled prototype is deleted from the bank; K fresh embeddings drawn from the IMPAIRED training pool for that class (there is no synthetic per-class generator for this corpus, unlike rfgen) are averaged into a re-enrolled prototype; recall is the fraction of that class's own held-out clean validation embeddings whose nearest prototype (base bank + re-enrolled row) is the re-enrolled row. Averaged over 20 independent K-shot draws per class per K to reduce variance from the small per-class impaired pool.

## Architecture note

The ONLY architectural difference between the two models is the backbone: the CNN uses 4 strided Conv1d->BatchNorm->ReLU blocks (channels 2->16->32->64->64, downsampling 1024 -> 64 timesteps); the z-plane model uses a complex rational-kernel spectral operator (width=64, layers=3, sections=8 poles/zeros per kernel) that does NOT decimate time (FFT-based, N stays 1024 through every layer) -- an intentional, unavoidable difference in internal mechanics (spectral filtering vs. strided downsampling) that does not break the shared-head contract, since pooling (mean/std over the time axis, here over complex-magnitude activations for phase-invariance) is dimension-count-invariant over time -- only channel count (2*64+12=140) matters for the fc1/fc2/normalize tail, which is byte-identical in shape/hyperparameters between both models.

## Caveats

- **Reduced episode budget, stated explicitly (not silently under-trained):** production's `train_signallab.py` trains the CNN for 6000 episodes; the CNN retrain here matches that (6000 episodes, ~2 min wall-clock on this CPU). The z-plane model costs ~18x more per episode (an FFT/IFFT pair per layer over a length-1024 complex batch, vs. 4 strided real convs), measured at a 50-episode timing probe: ~362 ms/episode vs. the CNN's ~20 ms/episode. To keep total wall-clock in the tens-of-minutes range, the z-plane run used 1500 episodes (~10 minutes), a quarter of the CNN's 6000 -- this is a real, acknowledged difference in training budget, not a like-for-like optimization comparison. The z-plane run's own validation-accuracy curve (0.788 @ ep~250, 0.905 @ ep 500, 0.912 @ ep 1000-1500) had largely plateaued by 1500 episodes, so more episodes would likely yield only marginal further closed-set gains -- but the few-shot LOO numbers, which lag the CNN's more, were not separately verified to have plateaued and may still be training-budget-limited.
- **A real numerical-stability bug was found and fixed during this experiment**, not a pre-existing property of the design: the first full training attempt (2000 episodes) produced `loss=nan` by episode ~500 and a collapsed embedding (0.139 closed-set accuracy, i.e. everything predicted as one class). Root-caused via `torch.autograd.set_detect_anomaly(True)` to `torch.abs()`'s backward at exactly-zero complex activations (a mathematically well-known singular gradient point, and exactly the failure mode the operator survey's gotcha list warned about) -- modReLU's own `clamp(mag+b, min=0)` routinely produces exact-zero outputs, so this was not a rare edge case. Fixed by computing magnitude as `sqrt(re^2+im^2+eps)` instead of `torch.abs`, which is smooth everywhere and phase-invariant identically to true `|z|` (re^2+im^2 does not depend on phase), eliminating the singularity with no change to modReLU's phase-equivariance or non-expansiveness properties. A second, independent stability issue was also found and fixed: the mixing matrices' naive Gaussian init (std=1/sqrt(width) applied separately to real and imaginary parts, per the design spec) gives an initial operator (spectral) norm of ~2.75 for width=64 -- i.e. the per-layer mixing map amplifies signal energy ~2.75x on every layer before any training even starts, since modReLU can only ever shrink magnitude, never compensate. Fixed with a `0.25x` init-scale factor (still the same small-Gaussian family the design calls for, just correctly scaled), bringing the initial operator norm to ~0.7 and confirmed via direct measurement (`zplane_backbone.py` layer-by-layer forward-pass probe) and a short training probe before committing to the full run. Both fixes are in `training/zplane_ab/zplane_backbone.py`'s `ZPlaneOperatorBackbone`, documented inline at the fix sites.
- **Few-shot LOO trial count**: each per-class K-shot draw is averaged over 20 independent random draws from the (small, ~20-40 item) per-class impaired training pool rather than a single draw, to reduce sampling variance -- still noisier than the production evaluate.py's protocol (N_TEST=120 fresh synthetic realizations per class), since this corpus has no synthetic per-class generator to draw arbitrarily many fresh test signals from.
- **Inference latency** numbers (`embed_all` ms/1k samples) are single-run CPU wall-clock measurements on this machine under current load, not averaged over multiple repeats -- treat as an order-of-magnitude comparison (the z-plane backbone's ~19x latency disadvantage vs. the CNN is a structural consequence of doing FFT/IFFT pairs over the full un-decimated 1024-sample length in every layer instead of the CNN's strided downsampling to 64 timesteps -- not a measurement artifact) rather than a tightly calibrated benchmark.

