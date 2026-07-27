# Overnight run, 2026-07-27 — U-Net transfer latent + denoiser

Brief: *"a U-Net that has a transfer-learnable latent space and denoises on the output to act
as an equalizer and cleaner. Doubt yourself and trust facts not your opinion."*

Headline: **the latent does not transfer, and the denoiser loses to a filter.** Both are
measured, both have controls that could have said otherwise, and one architecture change
(`skip_dropout=0.5` + `magnorm`) recovered most of the classification loss. Details below,
including four things I asserted earlier in the session that turned out to be wrong.

---

## 1. The latent does not transfer

`latent_transfer.py` freezes the encoder, trains only a light head, and reports against two
controls: a **random-init encoder** and the **12 hand-engineered scalar features**. Probe
machinery validated first — informative latents score 1.000, pure-noise latents 0.105 (chance).
It can fail.

`t1_base_cropoff` bottleneck, balanced accuracy (chance = 0.143 for class7, 0.027 for profile37):

| tap | dim | eff. rank | class7 | profile37 |
|---|---|---|---|---|
| **bott** (trained) | 256 | **1.03** | **0.156** | 0.030 |
| **bott** (random-init) | 256 | 1.10 | 0.169 | 0.035 |
| bott4 (chunk-pooled) | 1024 | 1.04 | 0.156 | 0.031 |
| trunk | 128 | 3.19 | 0.493 | 0.161 |
| embed | 32 | 1.84 | 0.431 | 0.167 |
| feat12 | 12 | 3.26 | 0.329 | 0.131 |

- Trained bottleneck is **statistically indistinguishable from random-init** (Δ +0.0041,
  CI [−0.0055, +0.0151], not separated).
- Significantly **worse than 12 hand-crafted scalars** (Δ −0.086, CI [−0.111, −0.060]).
- Training *reduced* effective rank, 1.10 → 1.03. It made the latent more collapsed.
- **`bott4` is the decisive tap.** It exists to separate "global pooling destroyed the
  information" from "the bottleneck never encoded it." Chunk-pooling in 4 time segments gives
  0.156 — identical. It is not the pooling. The information was never there.

Cause: **skip connections route around the bottleneck.** The decoder receives everything it
needs through the skips, so the bottleneck is never pressured to encode anything and collapses
to one direction. You cannot add a dim to a latent that has no dims.

### 1b. Three configurations, three nulls

The campaign trained the bottleneck-pressure fix (`skip_dropout=0.5` + `magnorm` +
`feat_dropout=0.5`) with and without length-crop augmentation. Every verdict is `False`.

| | t1 base | t2 FIXED, no crop | t4 FIXED + crop |
|---|---|---|---|
| bott eff. rank | 1.03 | 1.11 | 1.12 |
| class7 bal_acc | 0.156 | 0.234 | **0.464** |
| beats random-init | no | **yes, all 3 sets** | class7 + ofdm24, **not gsm7** |
| beats feat12 | no | **no, loses on all 3** | class7 only |
| **transfers** | **False** | **False** | **False** |

The two failures differ in kind, which is the informative part. t2's bottleneck beats a
random-init encoder on every label set (all separated) yet loses to the 12 hand-crafted
scalars on every label set. t4's beats the scalars on the *training* taxonomy (class7 Δ+0.040,
CI [0.011, 0.070], separated) but fails on the genuinely held-out sets that decide the verdict
— `ofdm24` doesn't beat features, `gsm7` doesn't beat *random*.

Learning the training taxonomy better is not the same as a transferable latent. Effective rank
never leaves ~1.1 against a random-init 1.07.

**The probe is not stuck at False.** It returned `separated: True` in many cells and its
built-in weak-signal control fires correctly (`power_control` lift 0.186 against a 0.10
requirement). The null is a measurement, not a broken instrument.

## 2. The denoiser loses to a trivial filter

`denoise_eval.py` scores against passthrough, matched-bandwidth lowpass, and an oracle LS FIR.
On `c3_unet_recon_only` with aligned targets:

- Beats passthrough: **+0.031**, CI [+0.015, +0.050] — real but small.
- **At low SNR (<6 dB), where equalization matters: model 0.3568 vs a trivial 4-bin-floor
  lowpass at 0.4211.** It loses.
- Captures **31%** of what an oracle 65-tap linear FIR achieves on the same rows.
- Fails F2 (Δ>0.02 with CI clear) and F5a (beat the oracle matched lowpass).

An 807k-parameter U-Net is being beaten by a filter.

## 3. The reconstruction target is partly corrupt

On rows at SNR ≥ 36 dB, where ρ/(1+ρ) = 0.9999, the impaired capture reaches only **γ² = 0.388**
against its own clean pair member. That deficit is not noise, so it is not denoising work.

Cause: `estimate_band` works from a 512-point PSD, so centre quantization is 1.95e-3
cycles/sample while a 16384-sample window tolerates 6.1e-5 — **32× too coarse**. Residual offset
median 1.05e-4 = 1.73 phase rotations across the window. One fine derotation lifts 0.388 → 0.644.

Caveat held: the corpus also applies multipath, IQ imbalance and PA saturation independently of
SNR, so *part* of the high-SNR deficit is legitimate equalization work. The spurious part is
specifically the residual-CFO component. `FINE_ALIGN` removes it at pair-build time.

## 4. Architecture: the one thing that worked

| | t1 base | **t2 FIXED** | c3_unet_recon_only |
|---|---|---|---|
| closed overall | 0.4139 | **0.6733** | 0.6183 |
| closed clean | 0.5161 | **0.7623** | 0.6979 |
| chirp AUROC | 0.651 | 0.200 | 0.834 |
| denoise vs passthrough | +0.036 | **+0.050** | — |
| canonical probe, best length | 0/19 | **19/19** | — |

`skip_dropout=0.5` + `magnorm` + `feat_dropout=0.5` — exactly the bottleneck-pressure fix that
finding 1 predicts — lifts closed-set **+0.26** over t1 and beats every prior U-Net. Open-set
chirp collapses 0.651 → 0.200, so it is a sharp trade, not a free win.

### 4b. Crop breaks the fill-lock, and the three goals trade against each other

Balanced accuracy on the canonical length sweep:

| N | fill | t1 base | t2 FIXED, no crop | t4 FIXED + crop |
|---|---|---|---|---|
| 4096 | 0.25 | 0.000 | 0.000 | 0.111 |
| 8192 | 0.50 | 0.000 | 0.000 | 0.222 |
| 16384 | 1.00 | 0.000 | **1.000** | 0.222 |
| 32768 | 1.00 | 0.000 | **1.000** | 0.244 |

t2 is perfect-then-zero — competent only where its corpus sits. t4 is mediocre everywhere. The
fill-lock is genuinely broken by cropping, at a heavy cost to the peak. Neither passes the gate.

The full trade across the three objectives, all on the FIXED architecture:

| | t2 (no crop) | t4 (crop) |
|---|---|---|
| closed-set | **0.6733** | 0.4756 |
| chirp AUROC (open-set) | 0.200 | **0.808** |
| worst-length bal acc | 0.000 | **0.111** |
| latent class7 | 0.234 | **0.464** |
| denoise vs passthrough | **+0.050** | +0.032 |

**No configuration is best at more than two of these.** Cropping restores the open-set
detection that the architecture change destroyed and flattens the length response, and costs
0.20 of closed-set doing it. That is a real trade to decide deliberately, not a tuning
accident — and nothing measured tonight says which corner you want.

### 4c. `w_rec` is not the lever

t5 repeats t4 with the reconstruction weight tripled, 1.0 → 3.0:

| | t4 (w_rec 1.0) | t5 (w_rec 3.0) |
|---|---|---|
| closed | 0.4756 | 0.4851 |
| chirp AUROC | **0.808** | 0.288 |
| worst-length bal | **0.111** | 0.000 |
| bott eff. rank | 1.12 | 1.12 |
| latent class7 | 0.464 | 0.470 |
| denoise vs passthrough | +0.032 | **+0.049** |

It buys +0.017 coherence and gives back both the open-set recovery and the length invariance
that cropping had just produced. The latent is unchanged to two decimals on every measure.

### 4d. CORRECTION — the completed 2×2 overturns part of §4c

§4c originally said "none of them moves the latent." **That was wrong**, and `t3_base_cropon`
(the cell I deprioritised as uninformative) is what disproves it. The full factorial:

| | closed | chirp | worst-length bal | **bott eff. rank** | class7 |
|---|---|---|---|---|---|
| base / no crop | 0.4139 | 0.651 | 0.000 | 1.03 | 0.156 |
| base / **crop** | 0.4851 | 0.707 | 0.000 | **1.79** | 0.409 |
| FIXED / no crop | 0.6733 | 0.200 | 0.000 | 1.11 | 0.234 |
| FIXED / crop | 0.4756 | 0.808 | **0.111** | 1.12 | 0.464 |

Crop **does** move the latent, and substantially: on the base arm it takes effective rank
1.03 → 1.79 (the highest of any trial) and class7 0.156 → 0.409.

Worse for the original story, the knobs **interact**, and not in the direction assumed. `FIXED`
was introduced as a bottleneck-pressure fix, yet with crop held on it *suppresses* the rank gain
crop produces alone: 1.79 (base/crop) → 1.12 (FIXED/crop). Whatever `skip_dropout=0.5` +
`magnorm` + `feat_dropout=0.5` is doing for closed-set accuracy, it is not increasing the
information in the bottleneck — on this evidence it reduces it.

Two other interactions the "independent knobs" framing hid:
- Crop alone does **not** break the fill-lock (base/crop worst-length 0.000). Only FIXED **and**
  crop together reach 0.111. The length fix requires both.
- FIXED's closed-set advantage (0.6733) exists only without crop. With crop the arms are level
  (0.4756 vs 0.4851), and base/crop is nominally ahead.

What survives: `w_rec` genuinely moves nothing (§4c table, latent identical at 1.12 / 0.46–0.47),
and **no configuration achieves transfer** — all four verdicts are `False`, and effective rank
tops out at 1.79 of 256. The structural reading still holds. The specific attribution to
`skip_dropout` does not, and I asserted it before the cell that tests it had run.

## 5. Four things I got wrong, corrected

1. **"The 0.878 physical coherence ceiling."** Quoted in every status report. Wrong — a matched
   brickwall provably beats ρ/(1+ρ). "54% of achievable" was meaningless.
2. **"Representational collapse."** The retrain was never collapsed; it was fill-locked, and my
   probe hard-coded N=4096 copied from a test file — the *old* corpus's SAMPLE_COUNT.
3. **"The `|c20|` finding is secondary."** It is the same residual-CFO defect as §3 and it
   corrupts the reconstruction target. Not secondary.
4. **"fs decorrelation caused the CW regression."** It did not. Capture length did.

## 6. Bugs fixed (any one would have cost the night)

- **MPS backward on a boolean-masked complex tensor** (`mask_clean_recon`) — 3-second death on
  every trial. Fixed by masking the real-valued loss instead. `eeda0ef`
- **`best_freq_offset` flattened `argmax`** on batched input — broke the exact pair-alignment
  path the campaign tests. `eeda0ef`
- **`tap_fn_for` dispatched on `isinstance`** and silently fell through for `TransferUNet`, so
  every trial reported `{"transfers": null, "why": "no tap 'bott'"}` — the primary measurement
  absent, while still writing a 17 KB report that looked complete. `b8f0056`

The recurring shape: **an absent measurement that still writes a file is indistinguishable from
a finished one.**

## 7. Known defects, not hidden

- 1 crop in 40 destroys the pair (clean-row coherence invariant holds 40/40 at `mode="off"`,
  39/40 at `mode="fill"`). ~2.5% of reconstruction targets are noise.
- Cropping covers fill ~0.06–0.375 for narrowband classes and **cannot** reach the high-fill end
  (fill 0.75 needs 32768 raw samples). The canonical probe will still fail there.
- Trials are 700 episodes. Short. A config that wins here may not win at full length.
- t1/t2 latent numbers came from re-scoring saved checkpoints after the tap fix, not inline.

## 8. What I would do next

1. **Do not ship any of this.** Shipped assets are untouched and green at 117/117.
2. **The transfer failure is structural, but the knobs interact — do not tune them singly.**
   All four verdicts are `False` and effective rank tops out at 1.79 of 256. But see §4d: crop
   moves the latent (1.03 → 1.79 on base), `FIXED` *suppresses* that gain (1.79 → 1.12), and the
   length fix needs both knobs together. Any follow-up must vary them jointly; a one-at-a-time
   sweep would have produced exactly the wrong causal story, and briefly did.
   The candidates the evidence supports:
   - **Investigate why `FIXED` costs bottleneck rank.** It buys closed-set without crop and
     costs latent rank with it. `skip_dropout`, `magnorm` and `feat_dropout` were changed
     together and never separated; that confound is now the highest-value thing to resolve, and
     it is cheap.
   - **Push crop harder on the base arm.** It is the only intervention measured to increase
     bottleneck rank, and it was never run without `FIXED` at any other setting.
   - **Consider dropping the U-Net for this purpose.** `trunk` (d=128) and `embed` (d=32) score
     0.49–0.51 on class7 where `bott` scores 0.23–0.46. If a transferable representation is the
     goal, the bottleneck is the weakest tap in the network, and the 38k CNN reaches 0.890
     closed-set against the U-Net's best 0.673.
3. Fix the front end properly: sub-bin centre estimation in `preprocess`. It fixes the target
   corruption *and* the `|c20|` damage, and it is a real receiver improvement. TS-parity blast
   radius, so it is a deliberate change.
4. Sweep capture length **in the generator** (`SAMPLE_COUNT` → 65536). Cropping is half the fix;
   it cannot manufacture the high-fill end.
5. The denoiser should be asked to beat the oracle matched lowpass, not passthrough. Passthrough
   is too weak a bar and it passed that while losing to a filter.
6. Decide the corner deliberately (§4b). Closed-set, open-set, and length invariance are in
   genuine tension here and no measurement resolves which one the deployment needs.
