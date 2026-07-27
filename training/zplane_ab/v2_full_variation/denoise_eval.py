"""Does the reconstruction head actually DENOISE / EQUALIZE, or is it decorating?

WHY THIS FILE EXISTS. campaign3 reports `final_coherence` (0.474 for c3_unet_recon_only,
0.473 for c3_unet_full) and that number has been read as "the reconstruction term does
real work". A coherence number on its own proves nothing, because it has no scale: it is
not calibrated against what you get for free, and it is not calibrated against what is
physically reachable. This module supplies both calibrations plus the failure modes that
would produce a good-looking number with no denoising behind it.

THE FOUR THINGS A COHERENCE NUMBER MUST BE MEASURED AGAINST.

 1. IDENTITY PASSTHROUGH (mandatory). Feed the IMPAIRED capture straight through with no
    model and score it against the clean pair member. This is the do-nothing baseline. A
    model that does not clearly beat it is not denoising, whatever its absolute score.
    Measured here: 0.4624 over the whole native train pool. campaign3's 0.474 is +0.012
    over doing nothing. Everything else in this file exists to explain why.

 2. THE PER-SAMPLE PHYSICAL REFERENCE rho/(1+rho) from each item's own snrDb. NOTE THE
    RELABEL. This quantity has been described in this codebase as a "physical ceiling"
    that "no equalizer beats". That is wrong in both directions and the mislabel is load
    bearing, so it is corrected here rather than propagated:
      (a) It is not a ceiling. It is the EXPECTED COHERENCE OF PASSTHROUGH under the
          model "impaired = clean + white noise at SNR rho". A denoiser that removes
          out-of-band noise beats it by construction: keep only the fraction B of the
          band the signal occupies and coherence goes rho/(rho+B) > rho/(1+rho) for B<1.
          The corpus median occupied fraction is 0.0359 (cw: 0.0002), so B is tiny and the
          margin is enormous. `_selftest_ceiling_is_beatable()` demonstrates this
          numerically on synthetic data. The true ceiling is 1.0.
      (b) On this corpus it is not even the passthrough expectation, because the pair is
          not related by additive noise alone -- see item 4 and `reachability_probe()`.
    It is still worth printing: the gap between passthrough and rho/(1+rho) separates
    "noise the model could remove" from "structure the front end already destroyed".

 3. A TRIVIAL FILTER. `lowpass()` is a brickwall keeping |f| <= mult*frac_bw/2, with
    frac_bw taken from the manifest (bandwidthHz/sampleRateHz) -- i.e. an ORACLE
    matched-bandwidth filter, deliberately given knowledge the model does not have, so
    that beating it means something. If a 3-line FFT mask with oracle bandwidth matches
    an 807k-parameter U-Net, the U-Net is not earning its keep.

 4. AN ACHIEVABILITY MEASUREMENT ON THE TARGET ITSELF, `reachability_probe()`: escalating
    ORACLE correctors (lag, then fine CFO derotation, then least-squares FIRs of growing
    length) fit per item WITH the clean answer in hand.

    !! IT IS NOT AN UPPER BOUND ON WHAT THE MODEL CAN DO, and an earlier version of this
    file said it was. "No LTI equalizer can do better" is false as written: a T-tap filter
    bounds only T-tap filters. Measured on the same 32 rows, the ladder does not plateau --
    65 taps 0.657, 257 taps 0.697, 513 taps 0.724, 1025 taps 0.771, 2049 taps 0.814, and
    full-length circular LTI reaches exactly 1.0000. So 0.697 was a TAP-COUNT artifact.
    Worse, 257 taps is smaller than the U-Net's own ~553-sample receptive field (depth 4,
    two 7-tap convs per block at strides 1/2/4/8, mirrored decoder), so the "bound" did not
    even cover the model's linear span, never mind its nonlinearity. The sentence "a
    full-length Wiener filter reaches 1.0, confirming the bound is a function-class limit
    and not a numerical artifact" had it exactly backwards: reaching 1.0 DISCONFIRMS it.

    What the measurement DOES establish, and this part is solid and reproduces: passthrough
    on rows with essentially no noise is only 0.237, so the impaired/clean pair is not
    related by additive noise -- and a single global derotation lifts it to 0.547, which
    identifies most of the deficit as a residual carrier offset rather than as anything a
    denoiser is supposed to remove. That is the finding. The upper-bound claim is retracted.

 5. THE DEGENERATE WIN. Coherence-against-clean is high whenever the input is already
    close to clean, so at high SNR a model that outputs a scaled copy of its INPUT scores
    well while denoising nothing. Two guards: `coh_recon_vs_input` (near 1.0 == the model
    is the identity), and the SNR-stratified table, whose LOW-SNR bins are the only place
    equalization is actually being tested.

WHAT WOULD FALSIFY "THE MODEL DENOISES". Stated up front, before the numbers. F1-F6 are
all evaluated by `verdict()`, including F6 -- which used to live only in main() and vanish
silently whenever --no-fir was passed, --align-n was 0, or no row cleared the SNR cut.
  F1. delta_vs_passthrough <= 0 on impaired rows. The model is worse than doing nothing.
  F2. The paired-bootstrap 95% CI on delta_vs_passthrough includes 0.02. Stated on the CI,
      not on the point estimate: at the documented default the point estimate is +0.029
      with CI [+0.017, +0.042], i.e. the old 0.02 threshold sits INSIDE the interval, and
      at --n 64 the same checkpoint fails outright. A threshold applied to a point estimate
      returns a different verdict on the same model for a different --n, with nothing to
      say that sample size is the reason.
  F3. coh(recon, input) > 0.95. The model IS the identity; any coherence-vs-clean it
      scores was already in the input.
  F4. The win is confined to high-SNR bins and delta <= 0.02 in the lowest SNR bins. Note
      that F4 carries NO evidential weight alone: every trivial oracle brickwall tested
      here passes it while being globally WORSE than doing nothing. It only means something
      inside the F1 & F2 & F3 conjunction.
  F5. A TRIVIAL BASELINE equals or beats the model. The comparator set is the oracle
      matched-bandwidth lowpasses AND `oracle_fir` -- the latter used to be computed,
      printed, and then excluded from every check by a `startswith("lowpass")` filter,
      which is how "F1-F5 all pass" was reported while the strongest baseline present beat
      the model by 4.3x on the same rows (passthrough 0.2764, model 0.3055, oracle 65-tap
      FIR 0.4016). F5a is the lowpass comparison, F5b the oracle-FIR one; F5b is a much
      harder bar and is reported separately rather than folded in silently.
  F6 (falsifies the TARGET, not the model). On NOISE-FREE (high-SNR) impaired rows,
      passthrough is far below 1.0 -- so the pair is not related by additive noise and
      part of 1-coherence is not denoising work at all. Fires here.

WHAT THIS ACTUALLY MEASURED (c3_unet_recon_only, native condition). Every number below
reproduces from `denoise_eval.py --n 256 --split {train,val} --seed 0`; anything that did
not reproduce has been removed rather than restated.

  * campaign3's 0.474 is NOT a model result. Identity passthrough over the whole native
    train pool is 0.4624. The pool is 25% CLEAN rows, whose impaired member IS their
    clean member, so they score exactly 1.0000 for free; impaired rows average 0.2813.
    0.474 is the same mixture average, +0.012. Read on the pool it looks like half the
    job is done; read correctly it is a rounding error. The "0.153 untrained floor" in
    the campaign notes is also mislabelled -- 0.1493 is the MEDIAN PASSTHROUGH on
    impaired rows, i.e. the do-nothing value, not an untrained-model floor.

  * Scored properly, the model does denoise, weakly, and for the right reason.
    TRAIN split, --n 256 -> 197 impaired rows: model 0.3055 vs passthrough 0.2764, delta
    +0.0292 (95% CI [+0.017, +0.042]), winning on 71.6% of rows, coh(recon,input) 0.9180.
    HELD-OUT VAL split, --n 256 -> 196 impaired rows: model 0.2975 vs passthrough 0.2694,
    delta +0.0281, 69.9% of rows, low-SNR delta +0.0710, coh(recon,input) 0.9280.
    !! An earlier write-up quoted the val split as "n=150, delta +0.0392, low-SNR +0.0921".
    No invocation of this file produces 150 impaired rows and those figures do not
    reproduce; they are inflated by ~40% on delta. The direction survives (val delta is
    +0.023..+0.031 across seeds), the magnitude did not.
    The delta is LARGEST IN THE LOWEST SNR BINS -- the opposite of the degenerate
    high-SNR copy. It is real; it is a few percent of the remaining gap, not 54% of
    anything. And it is roughly a QUARTER of what a 65-tap oracle LTI filter gets on the
    same rows, which is the comparison that matters and the one F5b now makes.

  * The target is mis-posed, and that is the finding that matters (F6), but the reason is
    narrower than previously claimed. On rows at snr >= 36 dB, where rho/(1+rho) = 0.9999
    and there is essentially no noise to remove, passthrough is only 0.237. That deficit
    is NOT noise. A single global CFO derotation lifts it to 0.547 and identifies the bulk
    of it as a residual carrier offset. The claim that ~0.30 of the loss is "unpayable by
    any CFO+LTI operator" is RETRACTED -- see item 4 above: the FIR ladder does not
    plateau and full-length circular LTI reaches 1.0000.

  * Mechanism, at least in part: a residual carrier offset between the pair, median
    5.96e-05 cycles/sample = ~1 full phase rotation across the 16384-sample window (p90
    ~10 rotations). native_preprocess.force_center already fixed the gross version of
    this, but it hands over a centre estimated from a 512-point PSD, quantized to
    1/512 = 1.95e-03, while a 16384-sample window needs ~1/16384 = 6.1e-05 -- 32x
    finer. Derotating alone lifts noise-free coherence 0.237 -> 0.547.

  * The oracle matched-bandwidth lowpass LOSES to passthrough (0.221 vs 0.276), for the
    same reason: with the carrier not actually at DC, a DC-centred brickwall narrower
    than the centre-estimate quantization can miss the signal. Flooring the mask width
    at 4 PSD bins recovers it to 0.270, still not beating passthrough. The naive filter
    is not a threat here, but only because the front end's centring is too coarse for
    filtering to help -- not because the model is good.

SHOULD THE TARGET BE THE CLEAN I/Q? Asked in the brief; answered by the numbers above.
The clean I/Q is the right *object* -- it is the only target that makes the head an
equalizer rather than a feature regressor -- but the way it is currently posed is broken
in three separable ways, in descending order of damage:

  (T1) IT IS NOT ALIGNED WITH ITS INPUT. Fix this before anything else; the other two are
       cosmetic next to it. force_center passes the impaired member's centre to the clean
       member, which was the right idea, but the centre it passes is quantized 32x too
       coarsely for a 16384-sample window. Estimate the residual offset between the pair
       directly -- `best_freq_offset()` in this file does it in one zero-padded FFT, no
       search -- and derotate the TARGET onto the input when the pair is built (in
       train_multitask.build_targets, which caches, so the cost is paid once). Expected
       effect, measured: noise-free reachability 0.237 -> 0.547 before any FIR, and
       0.697 with one. Until this is done the loss contains a large constant the model
       cannot remove, and campaign comparisons on coherence are comparing noise.

  (T2) CLEAN ROWS TEACH THE IDENTITY. 25% of the pool has impaired == clean, so their
       coherence loss is minimized exactly by recon = input. That is the opposite of the
       intended lesson, and the model's measured coh(recon, input) = 0.92 is what it
       looks like. Drop w_rec on clean rows (mask the term, keep them for the class
       loss), or better, never feed a clean member as the reconstruction input at all.

  (T3) THE LOSS IS NOT NORMALIZED PER SAMPLE. 1 - gamma^2 charges a high-SNR row that
       starts at 0.99 almost nothing and a low-SNR row that starts at 0.05 almost
       everything, so the gradient is dominated by rows where the target is least
       reachable. Score EXCESS over passthrough instead --
       (gamma^2(recon,clean) - gamma^2(input,clean)) / (1 - gamma^2(input,clean)) -- which
       is 0 for the identity and 1 for a perfect denoiser on every row regardless of SNR.
       That is exactly `frac_gap_closed` here, so the training objective and this
       evaluation would finally be the same quantity.

  What NOT to do: replace the clean I/Q with a lower-entropy surrogate (spectral
  magnitude, cumulants, a learned target). Those are all phase-blind or coarse, and a
  head trained on them stops being an equalizer -- which is half the stated deliverable.
  The target is fine; the pairing is what is broken.

SAFETY. Read-only. Uses the on-disk pool cache and np.memmap over corpus.f32 /
corpus_clean.f32; never opens either for writing, never touches src/embedding/assets,
never allocates on MPS (DEVICE is hard-pinned to CPU). Runs in ~20 s at --n 256.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_V2_DIR = os.path.dirname(os.path.abspath(__file__))
_ZPAB_DIR = os.path.dirname(_V2_DIR)
_TRAINING_DIR = os.path.dirname(_ZPAB_DIR)
for p in (_TRAINING_DIR, _ZPAB_DIR, _V2_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

CORPUS = os.path.join(_TRAINING_DIR, "artifacts", "signallab-corpus")
POOLS = os.path.join(CORPUS, "_pools")

# SNR bin edges (dB). The first two bins are where equalization is actually tested;
# everything >= 24 dB is close enough to noise-free that passthrough is already good.
SNR_EDGES = [-6, 0, 6, 12, 18, 24, 30, 36, 46]
LOW_SNR_DB = 6.0        # "low SNR" = below this; reported separately, per the brief
LP_MULTS = (1.0, 2.0)
FIR_TAPS = 65
# The front end estimates the carrier centre from an NFFT-point PSD, so its centre is
# quantized to 1/NFFT cycles/sample. A brickwall narrower than that can sit entirely OFF
# the residual carrier -- which is exactly why the un-floored oracle-bandwidth lowpass
# scores BELOW passthrough. Flooring the mask width at a few PSD bins is the honest
# matched-bandwidth filter given the pipeline's own resolution, and it is the variant that
# threatens the model.
#
# NFFT is IMPORTED, not written as a literal. The old `4.0/512.0` hard-coded preprocess's
# NFFT in two places (here and in the F6 prose), so changing NFFT would have left both
# silently stale while they asserted a specific number to the reader.
def _nfft():
    import preprocess as _pp
    return int(_pp.NFFT)


# 4 bins is not tuned: sweeping the floor over {2,4,6,8,12,16,24,32,48,64}/NFFT, the trivial
# brickwall peaks at 32/NFFT (0.2704) against 4/NFFT (0.2697). The difference is small and
# neither beats passthrough here, but the comparator should be the BEST trivial filter, so
# both are scored and F5a takes the max.
LP_WIDTH_FLOOR_BINS = (4.0, 32.0)
LP_WIDTH_FLOOR = 4.0 / 512.0    # retained for callers that pass it explicitly
# Tap ladder for the reachability probe. It must extend past the model's own receptive
# field, or the "bound" does not even cover the model's linear span: ComplexUNetMultiTask at
# depth 4 / 7 taps has a ~553-sample RF, so a 257-tap ladder top was smaller than the thing
# it claimed to bound. `None` means full-length circular LTI (the unconstrained Wiener
# solution), which is the control that shows where the ladder actually ends.
FIR_LADDER = (65, 257, 513, 1025, 2049, None)


# --------------------------------------------------------------------------------------
# the metric
# --------------------------------------------------------------------------------------
def coherence(y: np.ndarray, x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """gamma^2 = |<y,x>|^2 / (||y||^2 ||x||^2), per row of the last axis.

    Numerically identical to equalizer_frontend.coherence (the torch one used by the
    training loss); `_selftest_matches_torch()` asserts agreement so this file cannot
    silently drift from the quantity the model was optimized against.
    """
    num = np.abs((y * np.conj(x)).sum(axis=-1)) ** 2
    den = (np.abs(y) ** 2).sum(axis=-1) * (np.abs(x) ** 2).sum(axis=-1) + eps
    return num / den


def snr_passthrough_expectation(snr_db: np.ndarray) -> np.ndarray:
    """rho/(1+rho). The EXPECTED coherence of the unprocessed noisy capture against the
    clean one under additive white noise -- NOT a ceiling on what a denoiser can reach.
    See the module docstring, item 2, and `_selftest_ceiling_is_beatable()`."""
    rho = 10.0 ** (np.asarray(snr_db, dtype=np.float64) / 10.0)
    return rho / (1.0 + rho)


# --------------------------------------------------------------------------------------
# baselines that can fail
# --------------------------------------------------------------------------------------
def lowpass(z: np.ndarray, frac_bw: np.ndarray, mult: float = 1.0,
            width_floor: float = 0.0) -> np.ndarray:
    """Brickwall keep-|f| <= mult*max(frac_bw, width_floor)/2. Rows are already
    downconverted to DC by the native front end, so a DC-centred mask IS the
    matched-bandwidth filter. frac_bw comes from the manifest, which makes this an ORACLE
    baseline on purpose. See LP_WIDTH_FLOOR for why width_floor exists."""
    L = z.shape[-1]
    f = np.fft.fftfreq(L)
    fb = np.maximum(np.asarray(frac_bw, dtype=np.float64), width_floor)
    half = fb.reshape(-1, 1) * mult / 2.0
    mask = np.abs(f)[None, :] <= np.maximum(half, 1.0 / L)   # never mask everything away
    return np.fft.ifft(np.fft.fft(z, axis=-1) * mask, axis=-1)


def oracle_fir(z: np.ndarray, c: np.ndarray, taps: int = FIR_TAPS,
               ridge: float = 1e-6) -> np.ndarray:
    """Per-item least-squares T-tap circular FIR fit from z to c, WITH c in hand.

    This is a cheat by construction and that is the point: it upper-bounds what ANY
    linear time-invariant equalizer -- learned, adaptive or otherwise -- can achieve on
    this pair. If this bound is far from 1.0 on a noise-free item, the target is not
    reachable and the objective is mis-posed (falsifier F6).

    Normal equations built from FFT correlations, so cost is O(L log L + T^3) per item,
    not O(L T^2).
    """
    L = z.shape[-1]
    T2 = taps // 2
    lags = np.arange(-T2, T2 + 1)
    out = np.empty_like(z)
    for i in range(z.shape[0]):
        Z = np.fft.fft(z[i])
        Czz = np.fft.ifft(np.abs(Z) ** 2)                       # Czz[k] = sum_m conj(z[m]) z[m+k]
        Ccz = np.fft.ifft(np.fft.fft(c[i]) * np.conj(Z))        # Ccz[a] = sum_n conj(z[n-a]) c[n]
        R = Czz[(lags[:, None] - lags[None, :]) % L]
        p = Ccz[lags % L]
        R = R + ridge * np.trace(R).real / taps * np.eye(taps)
        try:
            h = np.linalg.solve(R, p)
        except np.linalg.LinAlgError:
            h = np.linalg.lstsq(R, p, rcond=None)[0]
        y = np.zeros(L, dtype=np.complex128)
        for a, ha in zip(lags, h):
            y += ha * np.roll(z[i], a)
        out[i] = y
    return out


# --------------------------------------------------------------------------------------
# the scoring function
# --------------------------------------------------------------------------------------
def score_denoising(clean, impaired, recon=None, snr_db=None, frac_bw=None,
                    impaired_mask=None, lp_mults=LP_MULTS, fir_taps=FIR_TAPS,
                    with_oracle_fir=True):
    """Score a denoiser against every baseline that could unmask a fake win.

    clean, impaired, recon : (N, L) complex. `recon` may be None (baselines only).
    snr_db, frac_bw        : (N,) per-item snrDb and occupied bandwidth fraction.
    impaired_mask          : (N,) bool. CLEAN rows have impaired == clean, so their
                             passthrough coherence is exactly 1.0 and they can only ever
                             teach/measure the identity. Headline numbers are computed on
                             impaired rows only; including clean rows is how a pool
                             average of 0.46 gets mistaken for a model doing something.

    Returns a dict of per-sample arrays and aggregates. Nothing is printed here; use
    print_report().
    """
    clean = np.asarray(clean, dtype=np.complex128)
    impaired = np.asarray(impaired, dtype=np.complex128)
    n = clean.shape[0]
    if impaired_mask is None:
        impaired_mask = np.ones(n, dtype=bool)
    impaired_mask = np.asarray(impaired_mask, dtype=bool)

    per = {}
    per["passthrough"] = coherence(impaired, clean)
    if frac_bw is not None:
        nf = _nfft()
        for m in lp_mults:
            per[f"lowpass_x{m:g}"] = coherence(lowpass(impaired, frac_bw, m), clean)
            for b in LP_WIDTH_FLOOR_BINS:
                per[f"lowpass_floor{b:g}_x{m:g}"] = coherence(
                    lowpass(impaired, frac_bw, m, b / nf), clean)
    if with_oracle_fir:
        per[f"oracle_fir_{fir_taps}"] = coherence(oracle_fir(impaired, clean, fir_taps), clean)
    if recon is not None:
        recon = np.asarray(recon, dtype=np.complex128)
        per["model"] = coherence(recon, clean)

    res = {"n": n, "n_impaired": int(impaired_mask.sum()), "per_sample": per,
           "impaired_mask": impaired_mask, "snr_db": snr_db}

    # --- degenerate-win guards -------------------------------------------------------
    if recon is not None:
        res["coh_recon_vs_input"] = coherence(recon, impaired)
        # scale-blind residual: how much of the input did the model actually change,
        # after removing the best global complex gain (which coherence ignores anyway)?
        a = (impaired * np.conj(recon)).sum(-1) / ((np.abs(recon) ** 2).sum(-1) + 1e-12)
        resid = impaired - a[:, None] * recon
        res["relative_change"] = np.sqrt((np.abs(resid) ** 2).sum(-1) /
                                         ((np.abs(impaired) ** 2).sum(-1) + 1e-12))

    if snr_db is not None:
        res["snr_expectation"] = snr_passthrough_expectation(snr_db)

    # --- aggregates on impaired rows only --------------------------------------------
    I = impaired_mask
    agg = {}
    for k, v in per.items():
        agg[k] = {"impaired_mean": float(v[I].mean()) if I.any() else float("nan"),
                  "impaired_median": float(np.median(v[I])) if I.any() else float("nan"),
                  "clean_mean": float(v[~I].mean()) if (~I).any() else float("nan"),
                  "all_mean": float(v.mean())}
    if "model" in per and I.any():
        d = per["model"][I] - per["passthrough"][I]
        agg["model"]["delta_vs_passthrough"] = float(d.mean())
        agg["model"]["win_rate_vs_passthrough"] = float((d > 0).mean())
        # fraction of the REMAINING gap the model closed. 0 == did nothing, 1 == perfect.
        # This is the right normalization: dividing by rho/(1+rho) credits the model for
        # coherence the input already had.
        gap = 1.0 - per["passthrough"][I]
        agg["model"]["frac_gap_closed"] = float((d / np.maximum(gap, 1e-6)).mean())
        if snr_db is not None:
            # The brief asked for improvement as a fraction of the per-sample rho/(1+rho).
            # Reported, but see the module docstring item 2: this DENOMINATOR IS WRONG --
            # it is what passthrough already gets, so a model scoring 1.0 here has merely
            # matched doing nothing. Kept visible so the two normalizations can be
            # compared side by side rather than argued about.
            e = snr_passthrough_expectation(snr_db)[I]
            agg["model"]["frac_of_snr_expectation_MISLABELLED"] = float(
                (per["model"][I] / np.maximum(e, 1e-6)).mean())
            agg["passthrough"]["frac_of_snr_expectation_MISLABELLED"] = float(
                (per["passthrough"][I] / np.maximum(e, 1e-6)).mean())
    res["agg"] = agg

    # --- SNR-stratified, low bins reported separately --------------------------------
    if snr_db is not None:
        rows = []
        snr_db = np.asarray(snr_db, dtype=np.float64)
        for a, b in zip(SNR_EDGES[:-1], SNR_EDGES[1:]):
            m = I & (snr_db >= a) & (snr_db < b)
            if not m.any():
                continue
            r = {"bin": f"[{a},{b})", "n": int(m.sum())}
            for k, v in per.items():
                r[k] = float(v[m].mean())
            r["snr_expectation"] = float(snr_passthrough_expectation(snr_db[m]).mean())
            if "model" in per:
                r["delta"] = r["model"] - r["passthrough"]
            rows.append(r)
        res["by_snr"] = rows
        low = I & (snr_db < LOW_SNR_DB)
        res["low_snr"] = {"n": int(low.sum()),
                          **{k: float(v[low].mean()) for k, v in per.items()}} if low.any() else None
        if "model" in per and low.any():
            res["low_snr"]["delta"] = res["low_snr"]["model"] - res["low_snr"]["passthrough"]
        # F6 input: passthrough on rows with essentially no noise. If the pair were related
        # by additive noise this would be ~rho/(1+rho) ~ 1.0; it is not, and that deficit is
        # the target defect. Computed here so verdict() can evaluate F6 on EVERY run rather
        # than only when main() happens to reach the reachability block.
        hi_cut = SNR_EDGES[-2]
        himask = I & (snr_db >= hi_cut)
        if himask.any():
            res["high_snr_passthrough"] = {
                "n": int(himask.sum()), "snr_min": float(hi_cut),
                "mean": float(per["passthrough"][himask].mean()),
                "expectation": float(snr_passthrough_expectation(snr_db[himask]).mean())}
    return res


def paired_ci(d, seed=0, boot=4000):
    """95% CI on the mean of a per-row paired difference. Every threshold in verdict() is
    applied to an interval, not to a point estimate -- the F2 threshold sits INSIDE this
    interval at the documented default, and the same checkpoint flips F2 and F5 at --n 64."""
    d = np.asarray(d, dtype=np.float64)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(boot, len(d)))].mean(1)
    return float(d.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def verdict(res, f2_threshold=0.02) -> list[str]:
    """Apply F1-F6 from the module docstring to a scored result. Returns verdict lines.

    Every comparison is on IMPAIRED rows and, where a per-row pairing exists, is reported
    with a bootstrap CI. F5 compares against BOTH the trivial lowpasses (F5a) and the oracle
    LS FIR (F5b); the latter used to be computed and then filtered out of the comparator
    set, which is how "all falsifiers pass" was reported while a 65-tap oracle filter was
    beating the model on the same rows by four times the model's own margin."""
    out = []
    a = res["agg"]
    if "model" not in a:
        return ["no model scored -- baselines only"]
    I = res["impaired_mask"]
    per = res["per_sample"]
    d = a["model"]["delta_vs_passthrough"]
    dv = per["model"][I] - per["passthrough"][I]
    m, lo, hi = paired_ci(dv)
    out.append(f"F1 model beats passthrough at all: {'PASS' if lo > 0 else 'FAIL'} "
               f"(delta {m:+.4f}, 95% CI [{lo:+.4f},{hi:+.4f}], n={int(I.sum())})")
    out.append(f"F2 delta > {f2_threshold} with the CI CLEAR of it: "
               f"{'PASS' if lo > f2_threshold else 'FAIL'} "
               f"(CI lower bound {lo:+.4f}; the point estimate is {m:+.4f} -- a point "
               f"estimate alone flips this verdict with --n)")
    if "coh_recon_vs_input" in res:
        ci = float(res["coh_recon_vs_input"][I].mean())
        out.append(f"F3 not an identity map, coh(recon,input) < 0.95: "
                   f"{'PASS' if ci < 0.95 else 'FAIL'} (coh {ci:.4f})")
    if res.get("low_snr"):
        dl = res["low_snr"].get("delta")
        if dl is not None:
            out.append(f"F4 wins at LOW SNR (<{LOW_SNR_DB:g} dB), not only where noise is absent: "
                       f"{'PASS' if dl > 0.02 else 'FAIL'} (delta {dl:+.4f}) "
                       f"-- NB every trivial brickwall passes F4 too; it is only "
                       f"meaningful inside the F1&F2&F3 conjunction")
    lps = [k for k in a if k.startswith("lowpass")]
    if lps:
        best_k = max(lps, key=lambda k: a[k]["impaired_mean"])
        dv2 = per["model"][I] - per[best_k][I]
        m2, lo2, _ = paired_ci(dv2)
        out.append(f"F5a beats the best oracle matched-bandwidth lowpass ({best_k}): "
                   f"{'PASS' if lo2 > 0 else 'FAIL'} "
                   f"(model {a['model']['impaired_mean']:.4f} vs {a[best_k]['impaired_mean']:.4f}, "
                   f"delta {m2:+.4f} CI lower {lo2:+.4f})")
    firs = [k for k in a if k.startswith("oracle_fir")]
    if firs:
        best_f = max(firs, key=lambda k: a[k]["impaired_mean"])
        dv3 = per["model"][I] - per[best_f][I]
        m3, lo3, _ = paired_ci(dv3)
        share = ((per["model"][I] - per["passthrough"][I]).mean() /
                 max((per[best_f][I] - per["passthrough"][I]).mean(), 1e-9))
        out.append(f"F5b beats the ORACLE LS FIR ({best_f}): "
                   f"{'PASS' if lo3 > 0 else 'FAIL'} "
                   f"(model {a['model']['impaired_mean']:.4f} vs {a[best_f]['impaired_mean']:.4f}, "
                   f"delta {m3:+.4f}) -- the model captures {share*100:.0f}% of the gain an "
                   f"oracle linear filter gets over passthrough on the SAME rows")
    # F6 lives here, not only in main(). It used to be printed by main() alone and so
    # disappeared with no message under --no-fir, --align-n 0, or an SNR range with no row
    # above the cut -- while the docstring claimed verdict() evaluated it on every run.
    hs = res.get("high_snr_passthrough")
    if hs is None:
        out.append("F6 target well-posed: NOT EVALUATED (no near-noise-free rows scored). "
                   "This is not a pass.")
    else:
        out.append(f"F6 target well-posed (passthrough near 1.0 where there is no noise): "
                   f"{'PASS' if hs['mean'] > 0.9 else 'FAIL -- FIRES'} "
                   f"(passthrough {hs['mean']:.4f} on n={hs['n']} rows at snr >= "
                   f"{hs['snr_min']:g} dB, where rho/(1+rho) = {hs['expectation']:.4f}). "
                   f"A deficit here is NOT noise and is not denoising work.")
    return out


# --------------------------------------------------------------------------------------
# self-tests: the metric must be what it claims to be
# --------------------------------------------------------------------------------------
def _selftest_invariance(rng=None):
    """Numeric proof that gamma^2 is scale- and phase-invariant BY CONSTRUCTION, and --
    just as important -- that it is not trivially invariant to everything, which would
    make it useless as a denoising score."""
    rng = rng or np.random.default_rng(7)
    lines = []
    y = rng.normal(size=(6, 512)) + 1j * rng.normal(size=(6, 512))
    x = rng.normal(size=(6, 512)) + 1j * rng.normal(size=(6, 512))
    base = coherence(y, x)

    for name, gain in [("scale x1e-6", 1e-6), ("scale x1e6", 1e6)]:
        # eps=0: the invariance claim is exact arithmetic, not an artifact of the guard
        d = np.abs(coherence(gain * y, x, eps=0.0) - base).max()
        assert d < 1e-12, (name, d)
        lines.append(f"  invariant to {name:14s} max|dgamma2| = {d:.2e}  (eps=0, exact)")
    # ...but the eps guard DOES bite once ||y||^2*||x||^2 approaches eps. Worth knowing
    # before anyone rescales the pipeline: at gain 1e-6 the default eps=1e-12 shifts
    # gamma^2 by ~5e-9. Harmless here (rows are unit-RMS) and reported, not hidden.
    d_eps = np.abs(coherence(1e-6 * y, x) - base).max()
    lines.append(f"  eps=1e-12 guard costs {d_eps:.2e} at gain 1e-6 (rows here are unit-RMS)")
    for phi in (0.3, 2.7, -1.9):
        r = np.exp(1j * phi)
        d = np.abs(coherence(r * y, x) - base).max()
        assert d < 1e-12, ("phase", phi, d)
        lines.append(f"  invariant to phase phi={phi:+.1f}   max|dgamma2| = {d:.2e}")
    # random per-row complex gain on EITHER argument
    g = rng.normal(size=(6, 1)) + 1j * rng.normal(size=(6, 1))
    d1 = np.abs(coherence(g * y, x) - base).max()
    d2 = np.abs(coherence(y, g * x) - base).max()
    assert d1 < 1e-12 and d2 < 1e-12
    lines.append(f"  invariant to per-row complex gain on y  max|dgamma2| = {d1:.2e}")
    lines.append(f"  invariant to per-row complex gain on x  max|dgamma2| = {d2:.2e}")

    # exact identity up to scale+phase -> 1.0
    one = coherence(x * (2.5 * np.exp(1j * 1.1)), x)
    assert np.allclose(one, 1.0, atol=1e-12)
    lines.append(f"  gamma2(a*e^(j phi) x, x) = {one.mean():.12f}  (must be 1)")

    # NOT invariant to time shift or frequency shift -- otherwise the metric could not
    # detect the misalignment that alignment_probe() goes looking for.
    d_shift = float(np.abs(coherence(np.roll(x, 3, axis=-1), x) - 1.0).mean())
    nvec = np.arange(x.shape[-1])
    d_freq = float(np.abs(coherence(x * np.exp(2j * np.pi * nvec / x.shape[-1]), x) - 1.0).mean())
    assert d_shift > 0.5 and d_freq > 0.5, "metric is degenerate"
    lines.append(f"  NOT invariant to 3-sample shift  (1-gamma2 = {d_shift:.4f})")
    lines.append(f"  NOT invariant to 1-bin freq shift (1-gamma2 = {d_freq:.4f})")
    return lines


def _selftest_matches_torch():
    """The numpy coherence here must equal the torch coherence the loss uses."""
    try:
        import torch
        from equalizer_frontend import coherence as tcoh
    except Exception as e:                                    # torch absent -> skip, don't fail
        return [f"  (skipped torch cross-check: {e})"]
    rng = np.random.default_rng(11)
    y = rng.normal(size=(5, 1024)) + 1j * rng.normal(size=(5, 1024))
    x = rng.normal(size=(5, 1024)) + 1j * rng.normal(size=(5, 1024))
    t = tcoh(torch.from_numpy(y.astype(np.complex64)),
             torch.from_numpy(x.astype(np.complex64))).numpy()
    d = float(np.abs(t - coherence(y, x)).max())
    assert d < 1e-5, d
    return [f"  matches equalizer_frontend.coherence (torch): max|diff| = {d:.2e}"]


def _selftest_ceiling_is_beatable():
    """rho/(1+rho) is NOT a ceiling. Synthetic narrowband signal + white noise: a
    brickwall matched to the occupied band beats rho/(1+rho) by a wide margin. If this
    assertion ever fails, the relabel in this module is wrong and should be reverted."""
    rng = np.random.default_rng(3)
    L, B = 8192, 0.02
    k = int(B * L / 2)
    S = np.zeros(L, dtype=np.complex128)
    idx = np.r_[0:k, L - k:L]
    S[idx] = rng.normal(size=2 * k) + 1j * rng.normal(size=2 * k)
    clean = np.fft.ifft(S)
    lines = []
    for snr_db in (0.0, 10.0, 20.0):
        rho = 10 ** (snr_db / 10)
        ps = np.mean(np.abs(clean) ** 2)
        noise = np.sqrt(ps / rho / 2) * (rng.normal(size=L) + 1j * rng.normal(size=L))
        imp = clean + noise
        c_pass = float(coherence(imp[None], clean[None])[0])
        c_lp = float(coherence(lowpass(imp[None], np.array([B]), 1.0), clean[None])[0])
        exp = float(snr_passthrough_expectation(np.array([snr_db]))[0])
        assert abs(c_pass - exp) < 0.06, (snr_db, c_pass, exp)     # passthrough ~ rho/(1+rho)
        # judged on the RESIDUAL 1-gamma^2, not the absolute value: near 1.0 an absolute
        # margin is meaningless, but halving the residual is unambiguous denoising.
        assert c_lp > exp and (1 - c_lp) < 0.5 * (1 - exp), (snr_db, c_lp, exp)
        lines.append(f"  snr {snr_db:5.1f} dB: rho/(1+rho) = {exp:.4f} | passthrough {c_pass:.4f} "
                     f"| brickwall {c_lp:.4f}  <- residual 1-gamma2 cut "
                     f"{(1 - exp) / (1 - c_lp):.0f}x below the 'ceiling'")
    return lines


# --------------------------------------------------------------------------------------
# is the target even reachable?
# --------------------------------------------------------------------------------------
def best_freq_offset(z, c, n_fine=1 << 20):
    """Residual carrier offset between the pair, to 1/n_fine cycles/sample.

    max_f |<z e^{j2pi f n}, c>| is just the peak of the zero-padded FFT of c*conj(z), so
    this is one FFT, not a search loop. Returns (derotated z, f_hat, coherence after).
    """
    # Shape-safe. This was written for a single row and np.abs(S).argmax() flattens, so a
    # 2-D call indexed fftfreq(n_fine) with a flat index and raised IndexError -- which made
    # derotate_onto() unusable on exactly the batched path build_targets needs.
    z_in = np.asarray(z)
    one = z_in.ndim == 1
    z2 = np.atleast_2d(z_in)
    c2 = np.atleast_2d(np.asarray(c))
    L = z2.shape[-1]
    S = np.fft.fft(c2 * np.conj(z2), n_fine, axis=-1)
    k = np.abs(S).argmax(axis=-1)
    f = np.fft.fftfreq(n_fine)[k]                       # (n,)
    zf = z2 * np.exp(2j * np.pi * f[:, None] * np.arange(L))
    g = coherence(zf, c2)
    if one:
        return zf[0], float(f[0]), float(g[0])
    return zf, f, g


def _full_wiener(z, c):
    """Unconstrained full-length circular LTI filter: H = FFT(c)/FFT(z), per bin. The
    limit of the FIR ladder, i.e. what the LTI function class can actually do when tap
    count is not the binding constraint. It reaches 1.0 whenever z has no exact spectral
    zeros, which is the whole point of including it."""
    Z, C = np.fft.fft(z), np.fft.fft(c)
    H = C * np.conj(Z) / (np.abs(Z) ** 2 + 1e-12 * np.mean(np.abs(Z) ** 2))
    return np.fft.ifft(H * Z)


ALIGN_NFINE = 1 << 18       # 3.8e-06 cycles/sample: 0.06 rotations residual at L=16384


def derotate_onto(clean, impaired, n_fine=ALIGN_NFINE):
    """T1 IN ONE CALL: rotate the CLEAN target onto the IMPAIRED input's carrier.

    native_preprocess.force_center already hands the clean member the impaired member's
    centre, but that centre comes from an NFFT-point PSD and is therefore quantized ~32x
    too coarsely for a 16384-sample window; what survives is a residual offset of median
    5.96e-05 cycles/sample -- about one full phase rotation across the window, p90 ten --
    which is enough to decorrelate the pair completely.

    Measured effect of removing it: coherence on near-noise-free rows 0.237 -> 0.547 with
    no filtering at all.

    WHAT THIS COSTS SCIENTIFICALLY, stated plainly. A frequency shift is not an LTI
    operation and a conv net with a ~553-sample receptive field cannot perform a global
    derotation, so leaving the offset in asks the reconstruction head for something outside
    its function class. Removing it takes carrier-offset correction OUT of the head's job
    and leaves noise and multipath, which are inside it. That is a deliberate narrowing of
    the task, not a free win, and any run using it must say so -- which is why the campaign
    records `pair_align` per trial and scores denoising against BOTH the aligned and the
    unaligned target."""
    zf, _f, _c = best_freq_offset(np.asarray(clean, dtype=np.complex128),
                                  np.asarray(impaired, dtype=np.complex128), n_fine)
    return zf.astype(np.complex64)


def reachability_probe(impaired, clean, ladder=FIR_LADDER):
    """How close can an ORACLE corrector get to the target on near-noise-free rows?
    Escalating function classes, each strictly containing the last:

        as-is  <  integer lag  <  fine CFO derotation  <  CFO + T-tap LS FIR (T growing)

    !! THIS IS NOT AN UPPER BOUND ON THE MODEL, and the earlier docstring claiming that
    ("no denoiser built from convolutions and a global derotation can beat it") was wrong.
    A T-tap least-squares filter bounds T-tap filters and nothing else. The ladder is swept
    precisely so that is visible: it does not plateau, and the `full` entry -- unconstrained
    circular LTI -- reaches 1.0000. Any single rung quoted as "the achievable ceiling" is a
    statement about that rung's tap count.

    The ladder must extend PAST the model's own receptive field or it does not even cover
    the model's linear span. ComplexUNetMultiTask(depth=4, taps=7) has a ~553-sample RF;
    the old top rung was 257.

    What the probe legitimately establishes is the `as_is` and `freq` rows: how far the
    pair is from each other with no noise present, and how much of that gap one global
    derotation removes. Returns a dict of per-row arrays.
    """
    n = impaired.shape[0]
    keys = ["as_is", "lag", "freq", "f_hat"] + [f"freq_fir_{t or 'full'}" for t in ladder]
    out = {k: np.zeros(n) for k in keys}
    for i in range(n):
        z, c = impaired[i], clean[i]
        den = (np.abs(z) ** 2).sum() * (np.abs(c) ** 2).sum() + 1e-12
        Z, C = np.fft.fft(z), np.fft.fft(c)
        out["as_is"][i] = float(np.abs(np.vdot(c, z)) ** 2 / den)
        out["lag"][i] = float(np.abs(np.fft.ifft(C * np.conj(Z))).max() ** 2 / den)
        zf, f, cf = best_freq_offset(z, c)
        out["freq"][i], out["f_hat"][i] = cf, f
        for t in ladder:
            y = _full_wiener(zf, c) if t is None else oracle_fir(zf[None], c[None], t)[0]
            out[f"freq_fir_{t or 'full'}"][i] = coherence(y[None], c[None])[0]
    return out


# --------------------------------------------------------------------------------------
# data access (read-only)
# --------------------------------------------------------------------------------------
def _manifest():
    return json.load(open(os.path.join(CORPUS, "corpus.json")))


def load_eval_rows(split="train", n=256, condition="native", seed=0, need_targets=True,
                   verify_target_builder=True, align_target=False):
    """Rows of (impaired_preprocessed, clean_target_preprocessed) + per-item metadata.

    split="train" reads the cached _clean_targets_*.npy that training already built (free,
    and generous to the model since it was trained on these rows -- failing HERE is
    decisive). split="val" builds targets on demand from np.memmap views of corpus.f32 /
    corpus_clean.f32, so it costs a few seconds for n rows instead of loading 3.7 GB.

    The on-demand builder must agree with the cache bit for bit or the two splits are not
    comparable. `verify_target_builder=True` checks that on a few rows and raises if it
    drifts. The docstring used to ASSERT this agreement with no code behind it; it happens
    to be true (max |cache - rebuild| = 0.0), but nothing would have caught a future change
    to train_multitask.build_targets.
    """
    import native_preprocess as npp
    import preprocess as pp
    module = npp if condition == "native" else pp

    man = _manifest()
    items = man["items"]
    N, L = man["count"], man["sampleCount"]
    tag = {"train": "tr", "val": "va", "enroll": "en"}[split]
    idx = np.load(os.path.join(POOLS, f"{condition}_{tag}_idx.npy"))
    xs = np.load(os.path.join(POOLS, f"{condition}_x{tag}.npy"), mmap_mode="r")
    fs = np.load(os.path.join(POOLS, f"{condition}_f{tag}.npy"), mmap_mode="r")
    imp = np.load(os.path.join(POOLS, f"{condition}_imp_{tag}.npy"))

    rng = np.random.default_rng(seed)
    sel = np.sort(rng.choice(len(idx), size=min(n, len(idx)), replace=False)) if n < len(idx) \
        else np.arange(len(idx))

    in_len = xs.shape[-1]
    impaired = xs[sel, 0, :].astype(np.complex128) + 1j * xs[sel, 1, :].astype(np.float64)
    feats = np.asarray(fs[sel], dtype=np.float32)

    def _build_rows(rows):
        imp_mm = np.memmap(os.path.join(CORPUS, "corpus.f32"), dtype="<f4", mode="r",
                           shape=(N, L, 2))
        cln_mm = np.memmap(os.path.join(CORPUS, "corpus_clean.f32"), dtype="<f4", mode="r",
                           shape=(N, L, 2))
        got = np.zeros((len(rows), in_len), dtype=np.complex128)
        for j, s in enumerate(rows):
            i = idx[s]
            zi = (imp_mm[i, :, 0] + 1j * imp_mm[i, :, 1]).astype(np.complex64)
            zc = (cln_mm[i, :, 0] + 1j * cln_mm[i, :, 1]).astype(np.complex64)
            _, ctx = module.preprocess(zi, l_out=in_len)
            cc, _ = module.preprocess(zc, l_out=in_len, force_center=ctx["center"])
            got[j] = cc
        del imp_mm, cln_mm
        return got

    clean = None
    if need_targets:
        cache = os.path.join(CORPUS, f"_clean_targets_{condition}_{in_len}_{len(idx)}.npy")
        if os.path.exists(cache):
            clean = np.asarray(np.load(cache, mmap_mode="r")[sel], dtype=np.complex128)
            if verify_target_builder and len(sel) >= 3:
                probe_rows = sel[:3]
                d = float(np.abs(_build_rows(probe_rows) - clean[:3]).max())
                if d > 1e-6:
                    raise RuntimeError(
                        "the on-demand clean-target builder has drifted from the cache "
                        f"(max |diff| = {d:.3e}). train_multitask.build_targets and "
                        "load_eval_rows must construct the same target, or the train and "
                        "val splits here are not comparable.")
        else:
            clean = _build_rows(sel)
        if align_target:
            # T1. Score against the SAME target the model was trained on when the campaign
            # trained with pair alignment on; otherwise passthrough and model are measured
            # against different objects and the comparison is meaningless.
            clean = np.stack([derotate_onto(clean[j], impaired[j]).astype(np.complex128)
                              for j in range(len(clean))])

    ci = idx[sel]
    meta = dict(
        corpus_idx=ci,
        snr_db=np.array([items[i].get("snrDb", 0.0) or 0.0 for i in ci], dtype=np.float64),
        frac_bw=np.array([items[i]["bandwidthHz"] / items[i]["sampleRateHz"] for i in ci]),
        cls=np.array([items[i]["cls"] for i in ci]),
        impaired=imp[sel].astype(bool),
        feats=feats,
        in_len=in_len,
    )
    return impaired, clean, meta


def run_model(ckpt, impaired, feats, arch=None, batch=8, net=None):
    """Forward a saved U-Net checkpoint on CPU and return its recon.

    `arch` is passed to unet_transfer.TransferUNet, which reproduces ComplexUNetMultiTask
    exactly at its defaults (same parameter names and shapes, asserted in its __main__) and
    additionally accepts the campaign's architecture switches. A campaign3 checkpoint and a
    campaign4 checkpoint therefore both load here. Pass `net=` to skip construction.

    DEVICE IS PINNED TO CPU deliberately: a GPU trial owns MPS and this machine has hung
    under memory pressure before. ~0.035 s/item at L=16384, so a few hundred rows is
    seconds.
    """
    import torch
    from unet_transfer import TransferUNet
    torch.set_num_threads(4)
    if net is None:
        net = TransferUNet(**(arch or {}))
        net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    net.eval()
    outs = []
    with torch.no_grad():
        for s in range(0, impaired.shape[0], batch):
            e = min(s + batch, impaired.shape[0])
            z = impaired[s:e]
            xb = torch.from_numpy(np.stack([z.real, z.imag], axis=1).astype(np.float32))
            fb = torch.from_numpy(np.asarray(feats[s:e], dtype=np.float32))
            net(xb, fb)
            outs.append(net.last_recon.numpy().astype(np.complex128))
    return np.concatenate(outs, axis=0)


# --------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------
def print_report(res, title=""):
    a = res["agg"]
    print(f"\n{'=' * 88}")
    print(f"  {title}")
    print(f"{'=' * 88}")
    print(f"rows scored: {res['n']}  (impaired {res['n_impaired']}, "
          f"clean {res['n'] - res['n_impaired']})")
    print("\nCLEAN rows are excluded from every headline number below. For a clean row the")
    print("impaired member IS the clean member, so passthrough coherence is exactly 1.000 and")
    print("the row can only reward the identity. Averaging them in is how a pool average gets")
    print("mistaken for evidence of denoising.")

    print(f"\n-- coherence vs clean, IMPAIRED rows only (n={res['n_impaired']}) "
          f"{'-' * 20}")
    print(f"{'estimator':<24}{'mean':>9}{'median':>9}{'delta vs passthrough':>24}"
          f"{'[clean rows]':>16}{'[pool avg]':>13}")
    base = a["passthrough"]["impaired_mean"]
    for k, v in a.items():
        d = v["impaired_mean"] - base
        dstr = "  (baseline)" if k == "passthrough" else f"{d:+.4f}"
        print(f"{k:<24}{v['impaired_mean']:>9.4f}{v['impaired_median']:>9.4f}{dstr:>24}"
              f"{v['clean_mean']:>16.4f}{v['all_mean']:>13.4f}")
    print("  [pool avg] is the column campaign3's `final_coherence` reports. It is a mixture")
    print("  of the [clean rows] column (free 1.0000 for passthrough) and the real one.")
    if res.get("snr_db") is not None:
        I = res["impaired_mask"]
        print(f"{'rho/(1+rho) [NOT a ceiling]':<24}"
              f"{res['snr_expectation'][I].mean():>9.4f}{np.median(res['snr_expectation'][I]):>9.4f}")

    if "model" in a:
        m = a["model"]
        print(f"\nmodel win rate vs passthrough: {m['win_rate_vs_passthrough'] * 100:.1f}% of rows")
        print(f"fraction of the remaining gap closed (delta / (1 - passthrough)): "
              f"{m['frac_gap_closed']:+.4f}")
        if "coh_recon_vs_input" in res:
            I = res["impaired_mask"]
            print(f"DEGENERACY GUARD  coh(recon, INPUT)      = "
                  f"{res['coh_recon_vs_input'][I].mean():.4f}   (near 1.0 => identity map)")
            print(f"DEGENERACY GUARD  relative change of input = "
                  f"{res['relative_change'][I].mean():.4f}   (near 0 => identity map)")

    if res.get("by_snr"):
        print(f"\n-- stratified by input SNR (impaired rows). The LOW bins are the test. --")
        cols = [k for k in res["by_snr"][0] if k not in ("bin", "n")]
        print(f"{'snr bin':<11}{'n':>6}" + "".join(f"{c[:18]:>19}" for c in cols))
        for r in res["by_snr"]:
            print(f"{r['bin']:<11}{r['n']:>6}" + "".join(f"{r[c]:>19.4f}" for c in cols))
    if res.get("low_snr"):
        ls = res["low_snr"]
        print(f"\nLOW SNR (< {LOW_SNR_DB:g} dB), n={ls['n']} -- where equalization actually matters:")
        for k, v in ls.items():
            if k != "n":
                print(f"    {k:<24}{v:>9.4f}")

    print(f"\n-- FALSIFICATION CHECKS {'-' * 62}")
    for line in verdict(res):
        print(f"  {line}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=256, help="rows to score")
    ap.add_argument("--split", default="train", choices=["train", "val", "enroll"])
    ap.add_argument("--condition", default="native")
    ap.add_argument("--model", default=os.path.join(
        _V2_DIR, "artifacts", "campaign3", "c3_unet_recon_only", "state_dict.pt"))
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--no-fir", action="store_true", help="skip the oracle-FIR bound")
    ap.add_argument("--align-n", type=int, default=32, help="rows for the alignment probe")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--align-target", action="store_true",
                    help="derotate the clean target onto the impaired input (T1) before "
                         "scoring. Use this when the model was TRAINED with pair alignment; "
                         "otherwise model and passthrough are scored against different "
                         "objects.")
    args = ap.parse_args()

    print("=" * 88)
    print("  METRIC SELF-TESTS -- gamma^2 must be scale/phase invariant BY CONSTRUCTION")
    print("=" * 88)
    for ln in _selftest_invariance():
        print(ln)
    for ln in _selftest_matches_torch():
        print(ln)
    print("\n  rho/(1+rho) is NOT a ceiling: a matched brickwall beats it (synthetic proof)")
    for ln in _selftest_ceiling_is_beatable():
        print(ln)

    print(f"\nloading {args.n} rows from split={args.split} condition={args.condition} ...")
    impaired, clean, meta = load_eval_rows(args.split, args.n, args.condition, args.seed,
                                           align_target=args.align_target)
    print(f"  L={meta['in_len']}  impaired {int(meta['impaired'].sum())}/{len(meta['impaired'])}"
          f"  snr {meta['snr_db'].min():.0f}..{meta['snr_db'].max():.0f} dB"
          f"  occupied bw frac median {np.median(meta['frac_bw']):.4f}")

    recon = None
    if not args.no_model and os.path.exists(args.model):
        print(f"  forwarding {os.path.basename(os.path.dirname(args.model))} on CPU ...")
        recon = run_model(args.model, impaired, meta["feats"])
    elif not args.no_model:
        print(f"  [!] no checkpoint at {args.model}; baselines only")

    res = score_denoising(clean, impaired, recon, meta["snr_db"], meta["frac_bw"],
                          meta["impaired"], with_oracle_fir=not args.no_fir)
    tag = os.path.basename(os.path.dirname(args.model)) if recon is not None else "BASELINES ONLY"
    print_report(res, f"DENOISING EVALUATION -- {tag} -- split={args.split}")

    # ---- is the target reachable at all? --------------------------------------------
    I = meta["impaired"] & (meta["snr_db"] >= 36)
    k = np.where(I)[0][:args.align_n]
    if not (len(k) and not args.no_fir):
        # Say so. The old code had no else-branch, so the section the whole analysis turns
        # on vanished with zero output whenever --no-fir / --align-n 0 was passed or the
        # corpus SNR range topped out below 36 dB.
        why = ("--no-fir was passed" if args.no_fir else
               "--align-n is 0" if args.align_n <= 0 else
               f"no impaired row scored has snr >= 36 dB (max {meta['snr_db'].max():.0f})")
        print(f"\n[!] REACHABILITY SECTION SKIPPED: {why}. F6 is reported as NOT EVALUATED "
              f"in the falsification checks above; that is not a pass.")
    else:
        L = meta["in_len"]
        print(f"\n{'=' * 88}")
        print("  IS THE TARGET WELL POSED? -- ORACLE reachability on near-noise-free rows")
        print(f"{'=' * 88}")
        print("At snr >= 36 dB, rho/(1+rho) = 0.9999: the impaired capture should ALREADY be")
        print("almost identical to its clean pair member. Any deficit here is not noise, so it")
        print("is not something a denoiser is supposed to be removing.")
        pr = reachability_probe(impaired[k], clean[k])
        print(f"\n  n={len(k)}   (each row below is an ORACLE, fit with the answer in hand)")
        rungs = [("as_is", "as-is (= passthrough)"),
                 ("lag", "+ best global integer LAG"),
                 ("freq", "+ best fine CFO derotation")]
        rungs += [(f"freq_fir_{t or 'full'}",
                   f"+ CFO then {'FULL-LENGTH circular LTI' if t is None else f'{t}-tap LS FIR'}")
                  for t in FIR_LADDER]
        for key, label in rungs:
            v = pr[key]
            print(f"    {label:<40}{v.mean():.4f}   (median {np.median(v):.4f}, "
                  f">0.9 on {int((v > 0.9).sum())}/{len(k)})")
        fh = np.abs(pr["f_hat"])
        nf = _nfft()
        print(f"\n  residual carrier offset |f| cycles/sample: median {np.median(fh):.2e}, "
              f"p90 {np.percentile(fh, 90):.2e}")
        print(f"  == {np.median(fh) * L:.2f} (median) / {np.percentile(fh, 90) * L:.1f} (p90) "
              f"full phase rotations across the {L}-sample window.")
        print(f"  For scale: one FFT bin here is {1 / L:.2e}; the front end estimates the centre")
        print(f"  from a {nf}-point PSD, so its quantization is {1 / nf:.2e} -- {L / nf:.0f}x")
        print("  coarser than the window can tolerate. preprocess.py:NFFT is the knob.")

        print(f"\n  WHAT THIS SHOWS, AND WHAT IT DOES NOT.")
        print(f"     SHOWS: passthrough is only {pr['as_is'].mean():.4f} on rows where")
        print(f"     rho/(1+rho) = 0.9999. That deficit is NOT noise, so it is not work a")
        print(f"     denoiser is supposed to be doing, and F6 fires on the TARGET. One global")
        print(f"     derotation lifts it to {pr['freq'].mean():.4f}, which locates most of the")
        print(f"     deficit in a residual carrier offset the front end left behind.")
        print(f"     DOES NOT SHOW: any upper bound on the model. Read the ladder -- it does")
        print(f"     not plateau, and full-length circular LTI reaches "
              f"{pr['freq_fir_full'].mean():.4f}. An")
        print(f"     earlier version of this file quoted the 257-tap rung as 'an upper bound on")
        print(f"     what the reconstruction head can ever score' and called 1 minus that rung")
        print(f"     'unpayable'. That inference is RETRACTED: a T-tap least-squares filter")
        print(f"     bounds T-tap filters, and 257 taps is smaller than the U-Net's own ~553-")
        print(f"     sample receptive field. The measurement stands; the bound does not.")
        print(f"\n     ACTIONABLE: derotate the clean target onto the impaired member when the")
        print(f"     pair is built (best_freq_offset() does it in one FFT). See T1 in the")
        print(f"     module docstring -- that is the fix this section supports.")


if __name__ == "__main__":
    main()
