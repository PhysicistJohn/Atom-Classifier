# Toward a genuinely invariant DACS — design idealization

**Status: pre-implementation. One probe run, one PASS, one FAIL-and-corrected.
No training has happened against this design. Do not let anyone read the
word "invariant" in this doc and assume it's proven — read §4 for exactly
what is and isn't.**

## 1. Why this document exists

Shipped DACS v7 (`docs/dacs-v7-runtime.md`, checkpoint
`training/artifacts/dacs-v4-runtime-20260803/`, live at
atomizer.radio-lab.app) is a real-valued CNN autoencoder over a **fixed**
64-point STFT (33 linear-Hz bins) at a **hard-locked 20 Msps** input
contract. It works well inside that contract — real off-air FM now
classifies correctly — but it is not what its name suggests. It is not
complex-valued natively, it is not a U-Net (the decoder takes only the
bottleneck, no skip connections — verified by reading `reconstruct()` in
`v7_model_mlx.py`, not assumed), and it is not any kind of resolution- or
operator-invariant network. Its only real invariance is dwell/time-extent,
via global pooling over a fully-time-convolutional encoder.

The owner wants a **second, successor line** built around genuine
invariance — sample rate, signal bandwidth extent, carrier phase, channel
impairment — using I/Q, phase, and magnitude natively, proper invariant
transforms, latent-space representations, and the clean/impaired pairs the
corpus pipeline already produces. This document is the target-state spec
for that line, plus an honest ledger of what's been checked so far, so the
next agent doesn't have to re-derive the reasoning or re-make a mistake
that's already been caught.

**This is a new, separate architecture generation. Do not modify or
retrain the shipped v4 checkpoint or package while working on this — it is
in production.**

## 2. Precise definitions (read this before touching anything — this is
##    where the first design pass went wrong)

Four distinct properties get casually called "invariance." They are not
the same transformation and do not share a mechanism. Conflating them
produced a wrong prediction on the first probe run (§4). Keep them
separate:

| Property | What varies | Have it? | Mechanism |
|---|---|---|---|
| **Dwell invariance** | capture duration (1/2.5/10 ms), same rate | **Yes, in production** | fully-time-convolutional encoder + global pooling collapses any T to a fixed vector |
| **Acquisition-rate / dilation invariance** | the *same relative signal* captured at a *different physical sample rate* — rate, its own bandwidth, and duration all scale together | **Probed, PASSES** | CQT bins defined as a *fraction of fs*, not absolute Hz — a true dilation lands in the literal same bin indices |
| **Extent invariance** | *one fixed capture rate*, but the signal's *own* bandwidth differs (GSM ~200 kHz vs. a 100 MHz NR carrier in the same 20 Msps capture) | **Probed, FAILED as first framed — mechanism corrected, not yet re-validated** | NOT a shift along the frequency axis (that was the wrong prediction) — it's "same center bin, wider/narrower activated region." Needs explicit extent-invariant pooling (multi-scale/pyramid), not shift-equivariance |
| **Channel/impairment invariance** | fading, noise, CFO, phase noise on the same underlying signal | Partially, via pixel-space MSE reconstruction today | Proposed: replace with a latent-space contrastive objective on the clean/noisy pairs the corpus already generates (§3.4) |
| **Carrier-phase invariance** | a global phase rotation of the whole capture | Not really — today's (Re, Im, log-mag) 3-channel split is not rotation-invariant | Proposed: native complex convolutions, rotation-equivariant by construction (§3.2) |

Open-set calibration is explicitly **out of scope** here — tracked
separately in `STATUS.md`'s open items (confidence-head calibration).

## 3. Target architecture (four components)

### 3.1 Front end: rate-normalized complex CQT
Bin centers defined as `f_k = top_frac · (fs/2) · 2^(−k/B)` — a fraction of
the sample rate actually used, not a fixed absolute-Hz grid. This is the
mechanism that makes dilation a no-op instead of a retrain. Output stays
complex (magnitude + phase per bin), not pre-split into real channels.
Reference implementation: `cqt_complex()` in
`tools/probe-rate-invariant-cqt.py` (direct Morlet-kernel correlation,
correctness-reference quality — not optimized, not meant for production
training as-is).

### 3.2 Encoder: native complex convolutions
Deep-Complex-Networks-style: `y = (Wr·xr − Wi·xi) + j(Wr·xi + Wi·xr)`,
complex GroupNorm, modReLU (or complex cardioid). A channel effect is
physically a complex multiply (gain × rotation); a complex-linear layer
represents "undo that" in one weight instead of a real network
reconstructing the 2×2 rotation relationship across 4 real weights.
**Not yet probed** — see §5 Phase 2.

### 3.3 Multi-scale pyramid for extent invariance
A shallow dyadic pyramid over the CQT's frequency (and/or time) axis,
combined by lightweight attention/pooling — this, not the CQT's
shift-equivariance, is what has to absorb "same signal family, different
occupied bandwidth at one fixed capture rate." **Not yet designed in
detail or probed** — see §5 Phase 1, the next thing to build.

### 3.4 Latent-space invariance objective, replacing the pixel decoder
Delete the pixel-space MSE reconstruction branch (in today's model:
`u1/u2/u3/out`, also the dominant memory cost at the 10 ms/200k-sample
signature per prior training notes). Replace with a contrastive/InfoNCE
pull between `embed(noisy_row)` and `embed(clean_row)` for every paired
corpus row, pushing apart other classes in the batch. **This needs no new
data generation** — every corpus already writes row-aligned
`clean.npy`/`noisy.npy` (see `training/artifacts/longdwell-production-corpus-v4/`).
It is more directly aligned with the actual downstream task (nearest-prototype
classification in embedding space) than reconstructing raw pixels, and
removes roughly half the network. **Not yet probed** — see §5 Phase 3.

### 3.5 Classification head: unchanged
Keep episodic nearest-prototype metric learning exactly as in v7. It is
not implicated in any invariance problem above and there is no reason to
touch it.

## 4. Validated so far — read the actual numbers, don't take "PASS" on faith

`tools/probe-rate-invariant-cqt.py`, numpy-only, no training, ~1s runtime.

**Test 1 — acquisition-rate/dilation invariance: PASS.**
Signal A: fs=20.00 MHz, bw=4.00 MHz. Signal B: fs=5.00 MHz, bw=1.00 MHz,
duration 4× longer (a true dilation of A by ratio=4, same relative
center-frequency fraction). Correlation between A's CQT and B's
(time-resampled) CQT, **zero bin shift applied**: **1.0000**. The
"same fraction of Nyquist" construction lands both in the literal same
bin indices — a stronger result than the original "shift by
log2(ratio) octaves" framing, which was the wrong prediction for *this*
scenario (that framing applies to Test 2's transformation, not this one).

**Test 2 — fixed-rate, signal's-own-bandwidth-scale: FAILED as first
framed, root cause diagnosed, not yet re-validated with the correct
mechanism.**
Same fs=20 MHz throughout; narrow signal bw=4 MHz vs. wide signal
bw=16 MHz (ratio=4), same center frequency. Predicted shift (my original,
wrong, claim): 48 bins / 2.000 octaves. Measured correlation **with**
that shift: **0.0000**. Measured correlation with **no** shift applied:
**0.4580** (partial — expected, since the narrow signal's activated bins
are a literal subset of the wide one's, centered the same). The correct
physical picture: a same-rate bandwidth change does not translate the
representation, it widens the activated region around the same anchor.
CQT's shift-equivariance does not cover this case at all — that's §3.3's
job, untested.

**Not yet probed at any level:** native complex-conv rotation-equivariance
(§3.2), the multi-scale pyramid's actual extent-invariance (§3.3, corrected
mechanism), and whether the latent-contrastive objective (§3.4) beats the
current pixel decoder on held-out generalization.

## 5. Phased validation plan — cheapest and most falsifiable first

Same discipline this session used for the corpus regen and the Optuna
study: prove small before spending GPU hours. Each phase should be
killable independently if it fails, exactly like Test 2 did.

- **Phase 0 (done).** Rate/dilation invariance probe —
  `tools/probe-rate-invariant-cqt.py` Test 1. PASS.
- **Phase 1 (next).** Corrected extent-invariance probe: build a small
  multi-scale pooling mechanism (e.g. max-pool over a few candidate
  frequency-window widths, or a 3-level frequency-axis pyramid) and test
  whether it produces *similar* pooled features for "same underlying
  modulation family, different occupied bandwidth, same fs" pairs —
  reusing `cqt_complex()`/`synthesize_burst()` from the existing probe
  rather than rewriting them. Numpy-only, minutes not hours.
- **Phase 2.** Native complex-conv correctness probe: verify a small
  complex conv block is rotation-equivariant to floating-point tolerance
  (`embed(x·e^{iθ})` behaves as predicted relative to `embed(x)`) —
  same gate discipline as the G0/G1 MLX parity harnesses in
  `docs/mlx-port-plan.md`. Also confirm MLX 0.32.0's complex dtype/conv
  support is actually adequate for production training before committing
  further; that port's own history (TF32-on-by-default, wired-limit
  memory quirks) is a fair warning that complex ops will have their own
  surprises.
- **Phase 3.** Latent-contrastive-vs-pixel-decoder ablation, small scale,
  on the **existing real-valued front end** (keep it a one-variable
  change) — does swapping the reconstruction loss for a contrastive pull
  on existing clean/noisy pairs help or hurt held-out generalization,
  before the front end changes at all.
- **Phase 4.** Corpus multi-rate exposure: today's stage 2 force-resamples
  every profile to one `TARGET_FS = 20_000_000`
  (`tools/longdwell_probe_stage2.py`) — that step is precisely what
  prevents rate-invariance from ever being trained *or verified* against
  real corpus content. Fixing this needs its own plan entry and a **new**
  corpus directory, never an edit to the v4 baseline in place (same rule
  `docs/corpus-regen-runbook.md` already enforces for content diversity).
- **Phase 5.** Small-scale end-to-end training probe (600-episode proxy,
  same shape as `tools/optuna-mlx-study.py` used this session) once
  Phases 1–4 each independently hold up.
- **Phase 6.** Gate against the same bar this session established: Tiers
  1/2 in `docs/TESTING.md`, and the frozen real Neptune FM capture in
  `training/real-captures/`. Floors must be **re-derived from this line's
  own measured five-seed spread**, not inherited from v7/v4 — same
  principle STATUS.md already states for the v4 floors.

## 6. Existing assets — reuse, don't rebuild

- Corpus: `training/artifacts/longdwell-production-corpus-v4/` (36
  profiles incl. `fm-broadcast-mpx`, `am-voice`; Atom-DSP propagation
  channel; still single-rate — see Phase 4).
- Clean/noisy pairs: already row-aligned in every corpus's
  `noisy.npy`/`clean.npy`. Zero new generation needed for §3.4.
- Current baseline to supersede:
  `training/zplane_ab/v2_full_variation/v7_probe/v7_model_mlx.py`
  (`V7NetMLX`) — read `reconstruct()` yourself before assuming it's a
  U-Net; it isn't.
- Shipped production model: `training/artifacts/dacs-v4-runtime-20260803/`
  — **do not touch**; live in production.
- Probe to extend: `tools/probe-rate-invariant-cqt.py` — reuse
  `cqt_complex()`/`synthesize_burst()`.
- Atom-DSP channel model: `Atom-DSP/src/channel.ts` +
  `tools/dsp_channel.py` (parity-gated numpy port) — reusable as-is for
  any corpus work under Phase 4.
- Testing tiers to extend: `docs/TESTING.md`.

## 7. Landmines (do not repeat these)

- **Don't conflate dilation-invariance with extent-invariance.** They are
  different transformations (§2) and need different mechanisms. This is
  the exact mistake the first design pass made — a "predicted shift" that
  was right for one and wrong for the other.
- **Probe before you train.** A theoretical invariance claim is cheap to
  falsify in numpy and expensive to falsify after an hour of MLX training.
  Test 2 caught a wrong prediction in about a second of runtime.
- **IQ only, per owner direction** (see memory `iq-only-direction.md`) —
  this line does not need to preserve or interoperate with the deprecated
  magnitude/time-domain-v3/v4 browser classifiers.
- **Data before compute** (see memory `data-before-compute.md`) — never
  launch a long training/search run on a corpus or architecture variant
  that a known, near-term fix will supersede. This bit twice already this
  weekend on the v3-channel and first v4-Optuna launches.
- **Generate new corpora into new directories, never overwrite a
  baseline** — same rule as `docs/corpus-regen-runbook.md`, applies
  directly to Phase 4.

## 8. Open questions for whoever picks this up

- CQT bins-per-octave / octave-span: the probe used 24 bins/octave, 9
  octaves, `top_frac_of_nyquist = 0.9` — arbitrary, not tuned against any
  real corpus content yet.
- Extent invariance: pooling-based (cheap, proposed) vs. an explicit
  bandwidth-estimate-then-normalize-to-canonical-ratio step (more
  accurate, more machinery) — undecided; Phase 1's probe should inform
  this.
- Whether the classification head (§3.5) ever needs revisiting once the
  embedding space itself changes shape/semantics — current guess is no,
  but worth re-checking once Phase 5 produces real embeddings to look at.
