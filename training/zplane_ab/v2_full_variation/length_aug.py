"""Random capture-length CROPPING at load time: length diversity from the corpus we already
have, without regenerating it.

WHAT THIS FIXES, AND HOW FAR IT GETS. Both trained models are FILL-LOCKED (training/
canonical_probe.py): each is competent only near its own corpus's SAMPLE_COUNT, because
preprocess resamples so occupied bandwidth hits TARGET_FRAC and then centre-fits to L_OUT,
making the signal-carrying fraction of the network input ("fill") linear in raw capture
length. The generator names the cause itself -- tools/generate-signallab-iq-corpus.ts:111,
captureLengthSamples "FIXED at SAMPLE_COUNT -- not yet swept; known gap".

A shorter capture IS physically a contiguous window of a longer one, so cropping is free and
honest length diversity. `main()` prints the fill distribution per class and per crop mode
from the live corpus; run it rather than trusting a pasted table. (An earlier copy of that
table lived here with no generating code, and on re-measurement its fm row was off by a
factor of 2-2.6 at every short length. Tables without generating code are how this project
keeps shipping numbers nobody can reproduce.)

Two things that DO hold and are reproducible from main(). (1) Fill is currently almost a
CLASS LABEL -- cw sits at 0.375 for essentially every item, ofdm and dsss at 1.000 -- which
makes it a shortcut feature; cropping is what decorrelates fill from class (fraction of
samples pinned at fill=1.0: 59.0% fixed-N -> 38.1% under log-uniform cropping). (2) Cropping
only moves fill DOWN. For a narrowband emitter whose measured bandwidth is floor-limited at
the Hann-512 mainlobe (6/512 = 0.01172, which cw measures exactly at every N >= 1024), fill
0.75 needs N = 32768 and CANNOT be reached from a 16384-sample corpus.

MEASURED CONSEQUENCE (`length_aug.py --probe`: a cheap CPU ridge readout on spectral +
envelope + cumulant features -- a weak stand-in for the network, so read the DIRECTION, not
the level). Scored on GENUINE out-of-corpus captures from canonical_probe.build_probes,
which builds cw/am/fm from their textbook definitions at ANY length.

!! READ THE BALANCED ACCURACY COLUMN, NOT THE RAW ONE. An earlier version of this docstring
quoted raw accuracy on that probe set and called 0.789 at N=2048 a breakthrough. It is not:
build_probes returns 15 cw, 2 am and 2 fm, so the CONSTANT predictor "always cw" scores
15/19 = 0.789 at EVERY length. The headline number was the majority-class baseline to three
decimals, and a readout that has learned nothing beat the reported crop-aug row at 4 of 6
lengths. probe() now prints that baseline on its own row and reports mean per-class recall
(chance 0.333) beside raw accuracy, so a degenerate readout is visible instead of flattering.
Any future claim off this table must be made on `bal`, against the `ALWAYS-cw` row.

The FIXED-N rows still independently reproduce the mirror-image fill-lock canonical_probe
found in the two trained models -- each fixed-length condition peaks at its own capture
length and falls to chance either side of it -- from nothing but a linear readout on
preprocess output. The lock is a property of the DATA GEOMETRY, not of any architecture.
Cropping flattens the LOW-fill half and does nothing for the high-fill half; on balanced
accuracy the crop-augmented readout is BELOW the degenerate baseline at N >= 16384, which
is the honest statement of how far this gets.

The high-fill half is a real deployment regime -- a field receiver handing over 32768+
samples of a narrowband emitter -- and the only cure is a corpus generated LONGER: at 65536
samples/item, cropping down spans fill 0.06..1.00 for EVERY class including cw, with no
change to this module. Until then, use this and know the residual gap. canonical_probe.py
will still fail its gate at N=32768, and that remains the correct output rather than
something to tune around.

THE OTHER CONTROL THAT WAS MISSING. "FIXED N=16384" fits on one length (1/5 of the rows)
while "CROP-AUG" fits on all five, so the comparison confounded 5x training data with
augmentation. probe() now also fits a row-count-MATCHED crop-aug model. Measured on held-out
corpus items, worst length over 1024..16384: fixed N=16384 0.259, crop-aug 0.526, crop-aug
at MATCHED row count 0.514. So the extra data is worth ~0.012 of the 0.267 and the gain is
overwhelmingly the augmentation -- but the control is what makes that sentence sayable.

INTERPOLATION IS NOT AN ALTERNATIVE -- measured, not argued (`length_aug.py --interp`).
Stretching a capture to fake a longer one fails on two independent grounds:

  1. In theory it buys nothing. Upsampling by k divides the signal's true fractional
     bandwidth by k, so N*bw -- and therefore fill -- is EXACTLY invariant. An ideal
     interpolator adds zero fill coverage by construction.
  2. In practice it does not even do that, because neither available interpolator leaves
     estimate_band intact. Measured bandwidth after a 2x upsample, against the faithful
     value bw0/2:
        class        bw0    bw0/2   fft_sinc   lin_resample
        cw        0.0117   0.0059     0.0127         0.3457
        am        0.0215   0.0107     0.0332         0.4375
        dsss      0.2871   0.1436     0.3340         0.3770
     lin_resample (the portable interpolator preprocess ships) is destroyed outright --
     cw reads 0.3457 where 0.0059 is correct, a 59x error. The ideal FFT upsampler is no
     better for a DIFFERENT reason: zero-padding the spectrum leaves the noise floor white
     below the original Nyquist and zero above it, so estimate_band's median-based floor
     (preprocess.py:73) collapses and the whole occupied half-band counts as signal. Any
     fill gained this way is an estimator artefact, not physics.

  Controlled A/B, matched sample count into preprocess -- cosine similarity of the final
  network input's spectrum against a GENUINE 16384 capture of the same item:
    class     sinc-up 2x   lin-up 2x   honest 8192 crop
    am             0.005       0.000              0.894
    cw             0.006       0.001              0.882
    fm             0.028       0.013              0.847
  The honest short crop is recognisably the same signal. Both stretched versions are not.
  (bluetooth/gsm/dsss score low in EVERY column because they are burst/hopping -- a
  different window of the record genuinely is a different picture. That is real diversity,
  not a defect, and it is why the burst guard below exists.)

BURST CLASSES AND THE ENERGY GUARD -- WEAKER THAN PREVIOUSLY CLAIMED. gsm/bluetooth/dsss
are duty-cycled (median gsm duty 0.734, p10 0.109), so a naive short crop can land in a gap
and hand the model pure noise under a signal label -- estimate_band's all-noise branch then
returns bw=0.95, i.e. fill 1.0 with nothing in it. Redrawing the offset until the crop
clears `guard_frac` of the item's power reduces that, at a median cost of one extra draw.

Two corrections to the earlier write-up, both material:
  (a) The empty-crop rate was measured with a 0.25 threshold while the guard ACCEPTS at
      0.35, so every accepted crop passed by construction and the reported improvement was
      the guard's acceptance rule measured against itself. `main()` now reports the rate on
      the FIRST, UNGUARDED draw (`power_ratio_first`) next to the accepted one; only the
      first-draw column is evidence.
  (b) The claim "without the guard the readout loses gsm 0.636 -> 0.195 at N=1024" is not
      produced by any code here and does not reproduce -- the measured gap is ~0.09, and
      worst-length accuracy is very slightly BETTER with the guard off. Keep the guard for
      label integrity (a noise window under a gsm label is a wrong label, whatever it does
      to a linear readout), not because it buys accuracy. And note the component's own
      falsifier 3 fires either way: gsm accuracy at N=1024 falls under crop augmentation
      WITH guard_frac=0.35, so a per-class length floor is still an open item.

CLEAN/IMPAIRED PAIRS MUST BE CROPPED IDENTICALLY. The reconstruction target is the clean
member of the pair; cropping the two at different offsets destroys it outright. Use
`crop_pair`, and keep passing the impaired member's centre to native_preprocess's
force_center (native_preprocess.py:56 -- independent centre estimation already cost
own-pair coherence 0.081 -> 0.0007 once).

HOW TO WIRE IT IN. Use `CropStream` (below): it re-crops ONLINE from the memory-mapped raw
corpus, once per episode, which is the only wiring that delivers the measured benefit. Do NOT
apply a crop inside pool_cache.py -- that path preprocesses to disk exactly once, so a crop
there freezes one random length per item and buys almost nothing while looking applied.
Measured cost of the online path: 1.8 ms per clean/impaired pair (crop + two native
preprocess calls + features), i.e. 0.13 s on a 70-sample episode against ~4.5 s of U-Net
training. It is not the bottleneck.

RECOMMENDED CONFIG: CropConfig(mode="loguniform", guard_frac=0.35). NOT mode="fill".
An earlier version of this docstring recommended mode="fill" on the strength of a fill-
DISPERSION proxy (fraction of samples pinned at fill=1.0: 59.0% fixed-N, 38.1% loguniform,
11.9% fill-targeted). That proxy is anti-correlated with the objective, and no accuracy
measurement of mode="fill" existed anywhere in this file -- probe() cropped with inline code
and never constructed a CropConfig at all. Both gaps are closed: probe() now draws every
crop through random_crop/CropConfig, and prints this (worst length over 1024..16384,
7 classes, item-disjoint, held-out corpus items -- `--probe` regenerates it):

    mode                          1024    2048    4096    8192   16384   worst
    off (no crop)                0.263   0.390   0.459   0.520   0.672   0.263
    loguniform guarded           0.501   0.568   0.615   0.608   0.589   0.501   <- default
    loguniform UNguarded         0.490   0.577   0.615   0.625   0.610   0.490
    fill U(0.05,1.0) guarded     0.421   0.461   0.581   0.596   0.625   0.421

Two things that had to be fixed before mode="fill" was even measurable: a HIDDEN PER-CLASS
FLOOR (fill_min=0.05 inverted through cw's floor-limited bandwidth demanded 2188 samples, so
short narrowband crops -- the exact regime the augmentation exists to cover -- were never
drawn), and crop_length_for_fill ignoring preprocess's own max(64, .) so the inverse
disagreed with the forward below that floor. Both are fixed; fill still loses, because fill
is CONSTANT below that floor and so fill-space sampling cannot spread mass over short
narrowband crops however it is parameterised. Cropping costs ~0.08 at the native length and
buys ~0.24 at the worst length. That trade is the whole case for the augmentation.

Run:  .venv-training/bin/python training/zplane_ab/v2_full_variation/length_aug.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

_V2_DIR = os.path.dirname(os.path.abspath(__file__))
_ZPAB_DIR = os.path.dirname(_V2_DIR)
_TRAINING_DIR = os.path.dirname(_ZPAB_DIR)
for _p in (_TRAINING_DIR, _ZPAB_DIR, _V2_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import preprocess as pp  # noqa: E402 -- imported, never modified

# Smallest crop we are willing to hand the front end. estimate_band uses NFFT=512 with 50%
# overlap, so below 1024 samples Welch has fewer than 3 segments.
#
# HONEST NOTE ON THE JUSTIFICATION. This constant was originally justified by "median cw
# bandwidth is 0.0137 at N=1024 against 0.0117 at every N >= 2048". That does NOT reproduce:
# cw measures 6/512 = 0.01172 exactly at N=1024 as well. The segment-count argument is the
# real one and it is structural, not empirical -- at N=1024 Welch has exactly 3 segments and
# below that it has fewer than the estimator assumes. 1024 is the floor, not a comfortable
# operating point; consider n_min=2048 if bandwidth-estimate noise shows up as a training
# instability.
MIN_CROP = 1024

# Fraction of the item's own mean power a crop must carry to be accepted. 0.35 sits above the
# noise-only case and below any legitimately-quiet-but-present window; measured to cut gsm's
# empty-crop rate from 0.425 to 0.037 at N=1024 with a median of one extra draw.
GUARD_FRAC = 0.35
GUARD_TRIES = 8


def fill_resampled(n_raw: int, bw: float, l_out: int = pp.L_OUT,
                   target_frac: float = pp.TARGET_FRAC) -> float:
    """Signal-carrying fraction of the network input under training/preprocess.py.

    Mirrors preprocess() exactly, INCLUDING its np.clip(frac, 1e-3, 0.95) (preprocess.py:137)
    -- without that clip this diverged from the real pipeline for every item on which
    estimate_band's all-noise branch returns bw = 0.95, which is 6.7% of cw items:
    new_len = max(64, round(n * clip(bw) / target_frac)), then center_fit to l_out.
    This is the quantity the fill-lock is a lock on."""
    bw = float(np.clip(bw, 1e-3, 0.95))
    return min(int(max(64, round(n_raw * bw / target_frac))), l_out) / l_out


def _l_out_native() -> int:
    """Imported, never hard-coded. The 16384 literal that used to sit in fill_native's
    signature was the corpus SAMPLE_COUNT copied by hand -- the same move that produced the
    N=4096 bug canonical_probe.py:9-11 exists to document."""
    import native_preprocess as npp
    return int(npp.L_OUT_NATIVE)


def fill_native(n_raw: int, l_out: int | None = None) -> float:
    """Same quantity under native_preprocess.py, which never resamples -- so fill is just
    the crop's share of L_OUT_NATIVE and cropping controls it directly."""
    l_out = _l_out_native() if l_out is None else l_out
    return min(n_raw, l_out) / l_out


def _resamples(module) -> bool:
    """Does this preprocess module resample to a canonical occupied-bandwidth fraction?
    training/preprocess.py does; native_preprocess.py explicitly does not (it zero-pads to
    L_OUT_NATIVE instead), so fill is a different function of crop length for each and using
    the wrong one is a 2x error on the native path."""
    return not hasattr(module, "L_OUT_NATIVE")


def fill_for(module, n_raw: int, bw: float, l_out: int | None = None) -> float:
    """The fill formula that matches `module`. Use this, never fill_resampled directly."""
    if _resamples(module):
        return fill_resampled(n_raw, bw, l_out or pp.L_OUT)
    return fill_native(n_raw, l_out)


def crop_length_for_fill(target_fill: float, bw: float, n_total: int,
                         l_out: int = pp.L_OUT, target_frac: float = pp.TARGET_FRAC) -> int:
    """Invert fill_resampled: the crop length that lands closest to `target_fill`.

    Clipped to [MIN_CROP, n_total], so requesting a fill the corpus cannot reach silently
    returns n_total rather than pretending. Check the achieved fill, not the request.

    THE FLOOR MATTERS AND USED TO BE MISSING HERE. fill_resampled applies preprocess's own
    `max(64, ...)`, so for a narrowband emitter fill is CONSTANT at 64/l_out for every crop
    below 64*target_frac/bw samples (cw: every n below ~2735). The naive inverse ignored
    that floor and returned 2188 samples for a requested fill of 0.05 on a cw item, which is
    why mode='fill' could never emit a short narrowband crop -- the shortest crop it would
    ever ask for was more than twice MIN_CROP. Below the floor every length gives the same
    fill, so the smallest one is the right answer."""
    if target_fill <= 64.0 / l_out:
        return int(np.clip(MIN_CROP, MIN_CROP, n_total))
    want = target_fill * l_out * target_frac / max(bw, 1e-6)
    return int(np.clip(round(want), MIN_CROP, n_total))


@dataclass
class CropConfig:
    """Everything the training loop needs to pin down for reproducibility."""
    mode: str = "loguniform"        # "loguniform" | "uniform" | "fill" | "off"
    n_min: int = MIN_CROP
    n_max: int | None = None        # None -> the item's own length
    fill_min: float = 0.05          # "fill" mode only
    fill_max: float = 1.0           # "fill" mode only
    guard_frac: float = GUARD_FRAC  # 0 disables the burst-gap guard
    guard_tries: int = GUARD_TRIES
    centre_prob: float = 0.0        # probability of a centred rather than random offset


def _draw_length(cfg: CropConfig, n_total: int, rng: np.random.Generator,
                 bw: float | None) -> int:
    lo = max(cfg.n_min, MIN_CROP)
    hi = min(cfg.n_max or n_total, n_total)
    if hi <= lo:
        return hi
    if cfg.mode == "off":
        return n_total
    if cfg.mode == "uniform":
        return int(rng.integers(lo, hi + 1))
    if cfg.mode == "fill":
        if bw is None:
            raise ValueError("mode='fill' needs a bandwidth estimate; pass bw=")
        # The requested fill floor is widened to what THIS item can actually reach at n=lo,
        # and crop_length_for_fill now honours preprocess's own 64-sample floor. Without
        # both, fill_min=0.05 on a narrowband emitter (cw, bw floor-limited at 6/512) never
        # asked for a crop shorter than 2188 samples -- so the short-crop regime the
        # augmentation exists to cover was never sampled for exactly the classes that need
        # it, and measured cw accuracy at N=1024 was 0.000.
        # Even fixed, fill-space sampling is FLAT below that floor (every short length gives
        # the same fill), so this mode still under-weights short narrowband crops relative
        # to 'loguniform'. That is why 'loguniform' is the default, not this.
        f_lo = min(cfg.fill_min, fill_resampled(lo, bw))
        target = float(rng.uniform(f_lo, cfg.fill_max))
        return int(np.clip(crop_length_for_fill(target, bw, hi), lo, hi))
    # loguniform: equal mass per octave of capture length. Fill is linear in length until it
    # saturates at 1.0, so this deliberately over-weights the short/low-fill end -- which is
    # the end the current corpus has zero coverage of.
    return int(round(float(np.exp(rng.uniform(np.log(lo), np.log(hi))))))


def _pick_offset(iq: np.ndarray, n: int, cfg: CropConfig,
                 rng: np.random.Generator) -> tuple[int, float, int, float]:
    """Return (start, power_ratio, draws, power_ratio_first).

    `power_ratio_first` is the FIRST draw's ratio, before the guard redrew. It is the only
    honest denominator for "how often would a naive crop have landed in a burst gap": the
    accepted ratio passes the guard by construction, so scoring the guard on the accepted
    ratio measures the acceptance rule against itself."""
    n_total = len(iq)
    if n >= n_total:
        return 0, 1.0, 1, 1.0
    item_power = float(np.mean(np.abs(iq) ** 2)) + 1e-20
    first_candidate = None
    if cfg.centre_prob > 0.0 and rng.random() < cfg.centre_prob:
        first_candidate = (n_total - n) // 2
    tries = max(1, cfg.guard_tries) if cfg.guard_frac > 0.0 else 1
    best_s, best_r, first_r = 0, -1.0, None
    for k in range(tries):
        s = first_candidate if k == 0 and first_candidate is not None \
            else int(rng.integers(0, n_total - n + 1))
        r = float(np.mean(np.abs(iq[s:s + n]) ** 2)) / item_power
        if first_r is None:
            first_r = r
        if r > best_r:
            best_s, best_r = s, r
        if cfg.guard_frac <= 0.0 or r >= cfg.guard_frac:
            return best_s, best_r, k + 1, first_r
    # Every random window failed the guard.  For a bursty clean signal, returning the least
    # empty random draw still creates an all-zero target (and therefore a mislabeled
    # classification sample plus a constant, zero-gradient coherence term).  One O(N)
    # cumulative-sum pass finds the highest-energy window deterministically.
    power = np.abs(iq) ** 2
    cs = np.concatenate(([0.0], np.cumsum(power, dtype=np.float64)))
    sums = cs[n:] - cs[:-n]
    max_s = int(np.argmax(sums))
    max_r = float(sums[max_s] / n) / item_power
    if max_r > best_r:
        best_s, best_r = max_s, max_r
    return best_s, best_r, tries + 1, float(first_r)


def random_crop(iq: np.ndarray, rng: np.random.Generator,
                cfg: CropConfig | None = None, bw: float | None = None,
                guard_iq: np.ndarray | None = None,
                ) -> tuple[np.ndarray, dict]:
    """Randomly crop a raw complex I/Q capture to a shorter contiguous window.

    Returns (view, meta). The view is a NUMPY VIEW, not a copy -- safe to hand straight to
    preprocess (which allocates its own outputs) and cheap enough to call per-sample on a
    memory-mapped corpus. meta carries n, start, power_ratio and draws so the training loop
    can log the realised length distribution instead of assuming it.

    `bw` is only needed for cfg.mode == "fill"; supply a per-item bandwidth measured once
    on the full record (pp.estimate_band) and cached, not re-measured per epoch.
    """
    cfg = cfg or CropConfig()
    # Fail at the call site, not three frames down inside welch_psd. Handing this a raw
    # (L, 2) float memmap row -- the interleaved-I/Q-as-floats mistake this repo has made
    # before -- otherwise returns a plausible-looking (n, 2) crop with a plausible-looking
    # power_ratio and only breaks later, somewhere else.
    iq = np.asarray(iq)
    if iq.ndim != 1 or not np.iscomplexobj(iq):
        raise TypeError(f"random_crop expects a 1-D COMPLEX capture, got ndim={iq.ndim} "
                        f"dtype={iq.dtype} -- did you forget to combine I and Q?")
    guard_iq = iq if guard_iq is None else np.asarray(guard_iq)
    if guard_iq.ndim != 1 or not np.iscomplexobj(guard_iq) or len(guard_iq) != len(iq):
        raise TypeError("guard_iq must be a length-matched 1-D complex capture")
    n_total = len(iq)
    n = min(_draw_length(cfg, n_total, rng, bw), n_total)
    s, ratio, draws, first = _pick_offset(guard_iq, n, cfg, rng)
    return iq[s:s + n], {"n": int(n), "start": int(s), "power_ratio": float(ratio),
                         "power_ratio_first": float(first),
                         "draws": int(draws), "n_total": int(n_total),
                         "guard_source": "primary" if guard_iq is iq else "paired"}


def crop_pair(impaired: np.ndarray, clean: np.ndarray, rng: np.random.Generator,
              cfg: CropConfig | None = None, bw: float | None = None
              ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Crop a clean/impaired pair with the SAME window.

    The offset is chosen using clean-member occupancy.  The impaired member can contain AWGN
    and DC during a true burst gap, so using its power as the guard accepts noise-only windows
    with an all-zero clean target.  The selected indices are still applied identically to both
    members; only the supervised augmentation's occupancy oracle uses the paired target.
    Cropping the two independently destroys the reconstruction target."""
    if len(impaired) != len(clean):
        raise ValueError(f"pair length mismatch: {len(impaired)} vs {len(clean)}")
    view, meta = random_crop(impaired, rng, cfg, bw, guard_iq=clean)
    s, n = meta["start"], meta["n"]
    return view, clean[s:s + n], meta


def prepare_cropped(iq: np.ndarray, rng: np.random.Generator, module,
                    cfg: CropConfig | None = None, bw: float | None = None,
                    scale_jitter: float = 0.0):
    """crop -> module.preprocess -> (channels, features, meta).

    `module` is training/preprocess.py or native_preprocess.py -- the same duck-typed surface
    build_pool_v2 already switches on, so this drops into that loop unchanged."""
    view, meta = random_crop(iq, rng, cfg, bw)
    norm, ctx = module.preprocess(np.asarray(view, np.complex128),
                                  scale_jitter=scale_jitter, rng=rng)
    meta.update(ctx)
    # fill_for, NOT fill_resampled: native_preprocess zero-pads instead of resampling, so
    # on the native path fill is n/L_OUT_NATIVE. Using the resampled formula there was a
    # measured 2x error (crop n=2048 reported 0.062, correct 0.125), and the native/recon
    # path is exactly the one the campaign defaults to.
    meta["fill"] = fill_for(module, meta["n"], ctx["bw"])
    return module.to_channels(norm), module.iq_features(norm), meta


def prepare_cropped_pair(impaired: np.ndarray, clean: np.ndarray, rng: np.random.Generator,
                         module, cfg: CropConfig | None = None, bw: float | None = None,
                         scale_jitter: float = 0.0):
    """Paired version for the denoising/reconstruction objective.

    Both members are preprocessed with the IMPAIRED member's measured centre when the module
    supports force_center (native_preprocess does); otherwise the pair is preprocessed
    independently and the caller inherits the decorrelation native_preprocess:56 warns about.
    """
    import inspect
    imp_view, cln_view, meta = crop_pair(impaired, clean, rng, cfg, bw)
    x_i, ctx = module.preprocess(np.asarray(imp_view, np.complex128),
                                 scale_jitter=scale_jitter, rng=rng)
    # Signature check, not try/except: swallowing TypeError would also swallow a real bug
    # inside preprocess and silently reinstate the independent-centre decorrelation.
    params = inspect.signature(module.preprocess).parameters
    if "force_center" in params:
        pair_kw = {"force_center": ctx["center"]}
        if "force_bw" in params:
            pair_kw["force_bw"] = ctx["bw"]
        if "force_resample_frac" in params and "resample_frac" in ctx:
            pair_kw["force_resample_frac"] = ctx["resample_frac"]
        x_c, _ = module.preprocess(np.asarray(cln_view, np.complex128), **pair_kw)
    else:  # training/preprocess.py has no force_center
        x_c, _ = module.preprocess(np.asarray(cln_view, np.complex128))
    meta.update(ctx)
    # fill_for, NOT fill_resampled: native_preprocess zero-pads instead of resampling, so
    # on the native path fill is n/L_OUT_NATIVE. Using the resampled formula there was a
    # measured 2x error (crop n=2048 reported 0.062, correct 0.125), and the native/recon
    # path is exactly the one the campaign defaults to.
    meta["fill"] = fill_for(module, meta["n"], ctx["bw"])
    return (module.to_channels(x_i), module.to_channels(x_c),
            module.iq_features(x_i), meta)


# ---------------------------------------------------------------------------------------
# online wiring: re-crop every episode from the memory-mapped raw corpus
# ---------------------------------------------------------------------------------------
class CropStream:
    """Per-episode crop sampler over the RAW corpus, memory-mapped and read-only.

    This is the only wiring that delivers what the measurements above promise. The pool
    cache (pool_cache.py) preprocesses to disk exactly once; a crop applied there freezes a
    single random length per item for the whole run. Here every call re-draws the length AND
    the offset, so an item is seen at a different capture length in every episode.

    `rows` are positions into the training pool (i.e. into `tr_idx`), which is what the
    episode sampler in train_transfer already produces.

    cfg.mode == "off" makes this the identity: the full-length capture, preprocessed exactly
    as pool_cache would. That is the CONTROL arm, and it goes through this same code path on
    purpose -- so the ON/OFF comparison differs in the crop and in nothing else.
    """

    def __init__(self, tr_idx, module, cfg: CropConfig | None = None, seed: int = 0,
                 want_clean: bool = True, align_fn=None, corpus_dir: str | None = None,
                 l_out: int | None = None):
        import json
        self.corpus = corpus_dir or os.path.join(_TRAINING_DIR, "artifacts", "signallab-corpus")
        with open(os.path.join(self.corpus, "corpus.json")) as f:
            man = json.load(f)
        self.n_items, self.n_samples = int(man["count"]), int(man["sampleCount"])
        self.raw = np.memmap(os.path.join(self.corpus, "corpus.f32"), dtype="<f4", mode="r",
                             shape=(self.n_items, self.n_samples, 2))
        self.clean = None
        if want_clean:
            cp = os.path.join(self.corpus, "corpus_clean.f32")
            if not (man.get("hasCleanPairs") and os.path.exists(cp)):
                raise RuntimeError("corpus has no clean pairs; cannot serve reconstruction targets")
            self.clean = np.memmap(cp, dtype="<f4", mode="r",
                                   shape=(self.n_items, self.n_samples, 2))
        self.tr_idx = np.asarray(tr_idx)
        self.module = module
        self.cfg = cfg or CropConfig()
        self.rng = np.random.default_rng(seed)
        self.align_fn = align_fn
        self.l_out = l_out or (getattr(module, "L_OUT_NATIVE", None) or pp.L_OUT)
        self._bw: dict[int, float] = {}
        self.last_lengths: list[int] = []
        self.last_fills: list[float] = []

    def _cap(self, arr, i):
        a = np.asarray(arr[i], dtype=np.float64)
        return a[:, 0] + 1j * a[:, 1]

    def _bw_of(self, i: int, z: np.ndarray) -> float:
        if i not in self._bw:
            self._bw[i] = float(pp.estimate_band(z)[1])
        return self._bw[i]

    def batch(self, rows):
        """rows -> (x[B,2,L] float32, feat[B,F] float32, clean[B,L] complex64|None)."""
        xs, fs, cs, lens, fills = [], [], [], [], []
        for r in np.asarray(rows):
            i = int(self.tr_idx[r])
            z = self._cap(self.raw, i)
            bw = self._bw_of(i, z) if self.cfg.mode == "fill" else None
            if self.clean is None:
                ch, ft, meta = prepare_cropped(z, self.rng, self.module, self.cfg, bw)
                cs.append(None)
            else:
                c = self._cap(self.clean, i)
                xi, xc, ft, meta = prepare_cropped_pair(z, c, self.rng, self.module,
                                                        self.cfg, bw)
                ch = xi
                zc = (xc[0] + 1j * xc[1]).astype(np.complex64)
                zi = (xi[0] + 1j * xi[1]).astype(np.complex64)
                if self.align_fn is not None:
                    zc = self.align_fn(zc, zi)
                cs.append(zc)
            xs.append(ch); fs.append(ft)
            lens.append(meta["n"]); fills.append(meta["fill"])
        self.last_lengths, self.last_fills = lens, fills
        X = np.stack(xs).astype(np.float32)
        Fm = np.stack(fs).astype(np.float32)
        C = np.stack(cs).astype(np.complex64) if self.clean is not None else None
        return X, Fm, C


# ---------------------------------------------------------------------------------------
# measurement -- the claim above is only worth what this prints
# ---------------------------------------------------------------------------------------

def _corpus():
    import json
    c = os.path.join(_TRAINING_DIR, "artifacts", "signallab-corpus")
    with open(os.path.join(c, "corpus.json")) as f:
        man = json.load(f)
    n, ln = man["count"], man["sampleCount"]
    raw = np.memmap(os.path.join(c, "corpus.f32"), dtype="<f4", mode="r", shape=(n, ln, 2))
    return raw, man


def _get(raw, i):
    a = np.asarray(raw[i], dtype=np.float64)
    return a[:, 0] + 1j * a[:, 1]


def _report(tag, per_class, classes):
    qs = [5, 25, 50, 75, 95]
    print(f"\n  {tag}")
    print("    %-10s %6s %6s %6s %6s %6s %8s" % ("class", *[f"p{q:02d}" for q in qs], "frac=1.0"))
    for c in classes:
        v = np.asarray(per_class[c])
        print("    %-10s %6.3f %6.3f %6.3f %6.3f %6.3f %8.3f"
              % (c, *np.percentile(v, qs), float(np.mean(v >= 0.999))))
    allv = np.concatenate([np.asarray(per_class[c]) for c in classes])
    print("    %-10s %6.3f %6.3f %6.3f %6.3f %6.3f %8.3f"
          % ("ALL", *np.percentile(allv, qs), float(np.mean(allv >= 0.999))))


def _selftest(raw, man):
    """Contract checks that must hold before any of the numbers below mean anything."""
    import json
    import native_preprocess as npp
    n_total = man["sampleCount"]
    rng = np.random.default_rng(0)
    x = _get(raw, 3)

    v, m = random_crop(x, rng, CropConfig(mode="uniform", n_min=2000, n_max=5000))
    assert 2000 <= m["n"] <= 5000 and len(v) == m["n"]
    assert np.shares_memory(v, x), "crop must be a view, not a copy"
    assert np.array_equal(v, x[m["start"]:m["start"] + m["n"]])

    off, _ = random_crop(x, rng, CropConfig(mode="off"))
    assert len(off) == n_total, "mode='off' must be the identity"

    bw = pp.estimate_band(x)[1]
    # The old version asserted `n == n_total or |got-want| < 0.02` over wants (0.1,0.3,0.6)
    # on a cw item, where want=0.6 needs 26236 > 16384 samples -- so that case CLIPPED and
    # passed vacuously, and it was the one probing the regime this module admits it cannot
    # reach. Pick the wants from what the item can actually reach, and require at least one
    # NON-clipped check so the inversion is genuinely exercised.
    f_max = fill_resampled(n_total, bw)
    wants = [w for w in (0.1, 0.3, 0.6) if w < 0.95 * f_max]
    assert wants, f"item bandwidth {bw:.4f} cannot reach any test fill (max {f_max:.3f})"
    exercised = 0
    for want in wants:
        n = crop_length_for_fill(want, bw, n_total)
        got = fill_resampled(n, bw)
        assert abs(got - want) < 0.02, (want, n, got, "inversion is wrong, not clipped")
        exercised += n < n_total
    assert exercised, "every fill-inversion check clipped -- the assertion proved nothing"

    # mode='fill' must be ABLE to reach n_min for a narrowband item. It could not before:
    # the hidden per-class floor is what made this mode the worst-performing one.
    short = min(_draw_length(CropConfig(mode="fill", fill_min=0.0), n_total,
                             np.random.default_rng(k), bw) for k in range(200))
    assert short == MIN_CROP, f"mode='fill' cannot reach short crops (min drawn {short})"

    # fill formula must follow the module, not a hard-coded constant
    assert _resamples(pp) and not _resamples(npp)
    assert abs(fill_for(npp, 2048, bw) - 2048 / npp.L_OUT_NATIVE) < 1e-12
    assert abs(fill_for(pp, 2048, bw) - fill_resampled(2048, bw)) < 1e-12

    # the pair must be the same window of the same emission
    clean_path = os.path.join(_TRAINING_DIR, "artifacts", "signallab-corpus", "corpus_clean.f32")
    craw = np.memmap(clean_path, dtype="<f4", mode="r", shape=(man["count"], n_total, 2))
    c = np.asarray(craw[3], np.float64)
    cln = c[:, 0] + 1j * c[:, 1]
    a, b, m = crop_pair(x, cln, np.random.default_rng(1))
    assert len(a) == len(b) == m["n"]
    assert np.array_equal(a, x[m["start"]:m["start"] + m["n"]])
    assert np.array_equal(b, cln[m["start"]:m["start"] + m["n"]]), "pair windows must coincide"

    ch, ft, mt = prepare_cropped(x, np.random.default_rng(2), pp)
    assert ch.shape == (2, pp.L_OUT) and ft.shape == (pp.N_FEATURES,) and 0 < mt["fill"] <= 1.0
    xi, xc, ft, mt = prepare_cropped_pair(x, cln, np.random.default_rng(2), npp)
    assert xi.shape == xc.shape == (2, npp.L_OUT_NATIVE)
    assert abs(mt["fill"] - mt["n"] / npp.L_OUT_NATIVE) < 1e-9, "native fill formula"

    # a raw (L,2) float row must be rejected at the call site, not deep inside welch_psd
    try:
        random_crop(np.asarray(raw[3], np.float32), np.random.default_rng(0))
        raise AssertionError("random_crop accepted a non-complex (L,2) row")
    except TypeError:
        pass

    # CropStream: mode='off' is the identity, and it agrees with a plain preprocess call
    cs = CropStream(np.array([3, 4]), npp, CropConfig(mode="off"), seed=0)
    X, Fm, C = cs.batch([0, 1])
    assert X.shape == (2, 2, npp.L_OUT_NATIVE) and C.shape == (2, npp.L_OUT_NATIVE)
    ref, _ = npp.preprocess(_get(raw, 3))
    assert np.allclose(X[0], npp.to_channels(ref), atol=1e-5), "mode='off' must equal no crop"
    assert cs.last_lengths == [n_total, n_total]
    cs2 = CropStream(np.array([3, 4]), npp, CropConfig(mode="loguniform"), seed=0)
    L1 = [cs2.batch([0])[0] is not None and cs2.last_lengths[0] for _ in range(6)]
    assert len(set(L1)) > 1, "CropStream must re-draw the length every call, not once"
    print("[ok] selftest: views, pair alignment, fill inversion (non-vacuous), dtype guard, "
          "native fill formula, CropStream re-crops per call")


def main(per_class_items: int = 60, draws: int = 4, seed: int = 20260726) -> int:
    raw, man = _corpus()
    _selftest(raw, man)
    classes, items, n_total = man["classes"], man["items"], man["sampleCount"]
    rng = np.random.default_rng(seed)
    by_cls = {c: np.array([i for i, it in enumerate(items) if it["cls"] == c]) for c in classes}
    sel = {c: rng.choice(v, per_class_items, replace=False) for c, v in by_cls.items()}

    print(f"corpus: {man['count']} items x {n_total} samples, classes {classes}")
    print(f"sampling {per_class_items} items/class x {draws} crops")

    modes = [("mode='off' (today: fixed N=%d)" % n_total, CropConfig(mode="off")),
             ("mode='loguniform' 1024..16384, guarded  <- DEFAULT", CropConfig(mode="loguniform")),
             ("mode='loguniform', guard DISABLED", CropConfig(mode="loguniform", guard_frac=0.0)),
             ("mode='uniform' 1024..16384, guarded", CropConfig(mode="uniform")),
             ("mode='fill' target U(0.05,1.0), guarded", CropConfig(mode="fill"))]

    EMPTY = 0.25    # "a crop that carries less than a quarter of the item's mean power"
    bw_cache: dict[int, float] = {}
    stats = {}
    for name, cfg in modes:
        per_class = {c: [] for c in classes}
        empties = {c: 0 for c in classes}       # AFTER the guard redrew
        empties0 = {c: 0 for c in classes}      # on the FIRST, unguarded draw
        tot = {c: 0 for c in classes}
        draw_counts = []
        for c in classes:
            for i in sel[c]:
                x = _get(raw, int(i))
                if int(i) not in bw_cache:
                    bw_cache[int(i)] = pp.estimate_band(x)[1]
                for _ in range(draws):
                    view, meta = random_crop(x, rng, cfg, bw_cache[int(i)])
                    _, bw = pp.estimate_band(view)
                    per_class[c].append(fill_resampled(meta["n"], bw))
                    draw_counts.append(meta["draws"])
                    tot[c] += 1
                    empties[c] += meta["power_ratio"] < EMPTY
                    empties0[c] += meta["power_ratio_first"] < EMPTY
                    if cfg.mode == "off":
                        break
        stats[name] = (per_class, empties, tot, draw_counts, empties0)
        _report(name, per_class, classes)
        # BOTH columns, and only the first is evidence: the guard ACCEPTS at guard_frac
        # (0.35 by default), which is above EMPTY, so every accepted crop clears EMPTY by
        # construction. Scoring the guard on the accepted column measures its own
        # acceptance rule against itself.
        print(f"    empty-crop rate (<{EMPTY:.0%} of item power), FIRST draw (unguarded) "
              f"-> accepted:")
        print("      " + ("  ".join(
            f"{c} {empties0[c]/max(tot[c],1):.3f}->{empties[c]/max(tot[c],1):.3f}"
            for c in classes if empties0[c] or empties[c]) or "none"))
        print(f"    mean offset draws per crop: {np.mean(draw_counts):.2f}")

    print("\n  READ THIS OFF THE TABLES:")

    def _pool(name):
        return np.concatenate([np.asarray(stats[name][0][c]) for c in classes])

    off, lu, fl = _pool(modes[0][0]), _pool(modes[1][0]), _pool(modes[4][0])
    for tag, v in (("fixed-N (today)", off), ("loguniform crop", lu), ("fill-targeted crop", fl)):
        print(f"    {tag:<20} fill p05..p95 = {np.percentile(v,5):.3f}..{np.percentile(v,95):.3f}"
              f"   {np.mean(v>=0.999)*100:>5.1f}% pinned at 1.0")
    print("    Fill is currently near-DETERMINISTIC per class, which is what makes it a")
    print("    shortcut feature; cropping is what breaks that. Lower 'pinned at 1.0' is better.")
    print("    !! mode='fill' gives the flattest coverage on THIS PROXY and is NOT the")
    print("       recommended default. Fill dispersion is not accuracy: measured on the ridge")
    print("       readout, worst-length accuracy is off 0.238 / loguniform 0.476 / fill 0.393.")
    print("       The default is mode='loguniform'. Run --probe for the accuracy comparison;")
    print("       do not select a mode on this table.")

    # The residual gap, stated on the class where it is unambiguous. cw's bandwidth is
    # floor-limited at the Hann-512 mainlobe, so its fill is a pure function of capture
    # length and cannot be raised by any choice of crop.
    cw_off = np.asarray(stats[modes[0][0]][0]["cw"])
    cw_lu = np.asarray(stats[modes[1][0]][0]["cw"])
    print(f"\n    cw fill, fixed-N: median {np.median(cw_off):.3f}; cropped: median "
          f"{np.median(cw_lu):.3f}, p95 {np.percentile(cw_lu,95):.3f}")
    print(f"    (the cw p95 is NOT cropping raising fill -- it is estimate_band's all-noise")
    print(f"     fallback returning bw=0.95 on the lowest-SNR cw items. It fires on "
          f"{np.mean(cw_off>=0.999)*100:.1f}% of")
    print( "     UNCROPPED cw too, so it is an estimator failure mode, not a crop artefact.)")
    print("    Cropping only moves fill DOWN. Reaching cw fill 0.75 needs N=32768 raw samples,")
    print("    which a 16384-sample corpus does not contain -- the HIGH-fill half of the lock")
    print("    stays uncovered, and canonical_probe.py will still fail its gate at N=32768.")
    print("    For the numbers there, run `--probe`; this function does not measure accuracy")
    print("    and must not be quoted as if it did. (An earlier version printed '0.316 vs")
    print("    0.368 at N=32768' here -- figures that appear in no measurement and that the")
    print("    module's own --probe output contradicts.)")
    print("    To close it, regenerate the corpus LONGER (65536 samples/item) and crop down.")
    print("    This module needs no change for that, only a larger n_total.")
    return 0


def interp_check(n_items: int = 20, seed: int = 3) -> int:
    """Reproduce the 'interpolation is a cheat' table.

    Controlled A/B at a MATCHED sample count handed to preprocess:
      A = preprocess(x[:16384])                 genuine 16384-sample capture
      B = preprocess(sinc_up(x[:8192], 2))      8192 stretched to 16384, ideal band-limited
      C = preprocess(lin_resample(x[:8192],2x)) 8192 stretched with the SHIPPED interpolator
      D = preprocess(x[:8192])                  honest 8192 crop -- this module's method
    If interpolation synthesised length faithfully, B and C would look like A."""
    raw, man = _corpus()
    classes, items, L = man["classes"], man["items"], man["sampleCount"]
    by_cls = {c: np.array([i for i, it in enumerate(items) if it["cls"] == c]) for c in classes}
    rng = np.random.default_rng(seed)

    def sinc_up(x, k):
        X = np.fft.fft(x); n = len(x); Y = np.zeros(n * k, complex); h = n // 2
        Y[:h] = X[:h]; Y[-h:] = X[-h:]
        return np.fft.ifft(Y) * k

    def spec(z):
        z = np.asarray(z, np.complex128)
        P = np.abs(np.fft.fftshift(np.fft.fft(z * np.hanning(len(z))))) ** 2
        s = P.sum()
        return P / s if s > 0 else P

    def cos(a, b):
        d = np.sqrt(float(a @ a) * float(b @ b))
        return float(a @ b) / d if d > 0 else float("nan")

    print("=== is interpolation a faithful way to synthesise a longer capture? ===")
    print("  bandwidth after a 2x upsample. PHYSICALLY FAITHFUL would be bw0/2.")
    print("  %-10s %9s %9s %9s %9s" % ("class", "bw0", "ideal/2", "fft_sinc", "lin_resamp"))
    for c in classes:
        b0s, bs, bl = [], [], []
        for i in rng.choice(by_cls[c], n_items, replace=False):
            x = _get(raw, int(i))
            b0s.append(pp.estimate_band(x)[1])
            bs.append(pp.estimate_band(sinc_up(x, 2))[1])
            bl.append(pp.estimate_band(pp.lin_resample(x, 2 * L))[1])
        print("  %-10s %9.4f %9.4f %9.4f %9.4f"
              % (c, np.median(b0s), np.median(b0s) / 2, np.median(bs), np.median(bl)))
    print("\n  spectral cosine of the final network input vs a GENUINE 16384 capture:")
    print("  %-10s %14s %14s %14s" % ("class", "sinc-up 2x", "lin-up 2x", "honest 8192 crop"))
    for c in classes:
        ca, cb, cc = [], [], []
        for i in rng.choice(by_cls[c], n_items, replace=False):
            x = _get(raw, int(i))
            A = spec(pp.preprocess(x)[0])
            ca.append(cos(A, spec(pp.preprocess(sinc_up(x[:8192], 2))[0])))
            cb.append(cos(A, spec(pp.preprocess(
                np.asarray(pp.lin_resample(x[:8192], 16384), np.complex128))[0])))
            cc.append(cos(A, spec(pp.preprocess(x[:8192])[0])))
        # nanmedian silently drops estimator failures; count them instead of hiding them.
        nn_ = sum(int(np.isnan(v).sum()) for v in (np.array(ca), np.array(cb), np.array(cc)))
        print("  %-10s %14.3f %14.3f %14.3f   (%d/%d cosines NaN)"
              % (c, np.nanmedian(ca), np.nanmedian(cb), np.nanmedian(cc), nn_, 3 * len(ca)))
    print("\n  VERDICT. In theory upsampling by k divides true bw by k, so N*bw -- hence fill --")
    print("  is exactly invariant: an ideal interpolator adds ZERO coverage. In practice neither")
    print("  interpolator even manages that: lin_resample wrecks the estimate outright, and the")
    print("  FFT upsampler leaves a half-white noise floor that collapses estimate_band's")
    print("  median-based floor. Any fill 'gained' is an estimator artefact. And the honest short")
    print("  crop is far closer to the genuine capture in every class. Crop; do not stretch.")
    return 0


def _readout_features(wav: np.ndarray, feats12: np.ndarray) -> np.ndarray:
    """Cheap stand-in for what a conv stack can see: coarse log-PSD of the 1024-sample
    network input, the amplitude-envelope spectrum, four envelope scalars, and the 12
    cumulant features preprocess already computes. Numpy only -- no GPU, no training."""
    w = np.asarray(wav, np.complex128)
    P = np.abs(np.fft.fftshift(np.fft.fft(w * np.hanning(w.shape[1]), axis=1), axes=1)) ** 2
    P = P.reshape(len(P), 128, -1).mean(2)
    P = np.log10(P / (P.sum(1, keepdims=True) + 1e-20) + 1e-9)
    a = np.abs(w)
    A = np.abs(np.fft.rfft(a - a.mean(1, keepdims=True), axis=1))[:, :64]
    A = np.log10(A / (A.sum(1, keepdims=True) + 1e-20) + 1e-9)
    env = np.stack([a.mean(1), a.std(1), (a > 1e-3).mean(1), np.percentile(a, 90, axis=1)], 1)
    F = np.nan_to_num(np.log1p(np.abs(feats12)) * np.sign(feats12), nan=0, posinf=0, neginf=0)
    return np.nan_to_num(np.hstack([P, A, env, F]), nan=0, posinf=0, neginf=0)


def probe(per_class_items: int = 250, seed: int = 20260726) -> int:
    """Does crop augmentation actually unlock the fill-lock? Ridge readout, CPU only.

    Trained on corpus crops drawn through THIS MODULE'S OWN API (random_crop + CropConfig,
    guard included) -- the previous version cropped with six lines of inline code that
    exercised none of the shipped functions, so every claim it produced was a claim about
    the inline code. Then scored TWICE: on held-out corpus items (item-disjoint) and on
    canonical_probe.build_probes, which builds textbook cw/am/fm from their definitions at
    ANY length and is therefore the only instrument that can reach fill > 0.375 for a
    narrowband emitter.

    THREE CONTROLS, all of which can make the augmentation look bad, and two of which used
    to be missing:
      * ALWAYS-<majority>  -- the constant predictor. build_probes is 15 cw / 2 am / 2 fm,
        so it scores 0.789 raw at every length. Any raw number at or below that is nothing.
      * balanced accuracy  -- mean per-class recall, chance 0.333 on the probe set. This is
        the column to read; raw accuracy on a 79%-majority set is not interpretable.
      * MATCHED-ROWS crop-aug -- fitted on the same NUMBER of rows as the single-length
        conditions, because crop-aug otherwise gets 5x the training data and the gain is
        attributed entirely to augmentation.

    This is a weak linear readout, not the U-Net -- read the DIRECTION, not the level."""
    import warnings
    from canonical_probe import build_probes
    warnings.filterwarnings("ignore")
    raw, man = _corpus()
    classes, items, L = man["classes"], man["items"], man["sampleCount"]
    rng = np.random.default_rng(seed)
    by_cls = {c: np.array([i for i, it in enumerate(items) if it["cls"] == c]) for c in classes}
    # Lengths follow the corpus, never a literal: at 65536 samples/item a hard-coded 16384
    # top would silently stop testing the native length.
    LENS = [n for n in (1024, 2048, 4096, 8192, 16384, 32768, 65536) if n <= L]
    NATIVE, SHORT = L, LENS[min(2, len(LENS) - 1)]

    # One CropConfig per training row, drawn from the shipped API. `n_max` pins the length
    # for the FIXED conditions so all three conditions share one code path.
    cfg_by_len = {n: CropConfig(mode="uniform", n_min=n, n_max=n) for n in LENS}
    wav, f12, ys, ns, ids = [], [], [], [], []
    for ci, c in enumerate(classes):
        for i in rng.choice(by_cls[c], per_class_items, replace=False):
            x = _get(raw, int(i))
            for n in LENS:
                view, meta = random_crop(x, rng, cfg_by_len[n])
                z, _ = pp.preprocess(np.asarray(view, np.complex128))
                wav.append(z); f12.append(pp.iq_features(z))
                ys.append(ci); ns.append(meta["n"]); ids.append(int(i))
    X = _readout_features(np.stack(wav), np.stack(f12))
    y = np.array(ys); nn = np.array(ns); ids = np.array(ids)
    uid = np.unique(ids)
    tr = set(np.random.default_rng(1).permutation(uid)[:int(0.7 * len(uid))].tolist())
    istr = np.array([i in tr for i in ids])

    def fit(mask, lam=3e-2):
        A = X[mask]; mu = A.mean(0); sd = A.std(0) + 1e-6
        Z = np.hstack([(A - mu) / sd, np.ones((int(mask.sum()), 1))])
        return mu, sd, np.linalg.solve(Z.T @ Z + lam * len(Z) * np.eye(Z.shape[1]),
                                       Z.T @ np.eye(len(classes))[y[mask]])

    def predict(m, Xq):
        mu, sd, Wt = m
        return np.argmax(np.hstack([(Xq - mu) / sd, np.ones((len(Xq), 1))]) @ Wt, 1)

    def scores(pred, yq):
        acc = float(np.mean(pred == yq))
        per = [float(np.mean(pred[yq == c] == c)) for c in np.unique(yq)]
        return acc, float(np.mean(per))

    # matched-rows control: same row COUNT as a single-length condition, sampled across all
    # lengths, so "augmentation" is not credited with 5x the data.
    n_one = int((istr & (nn == NATIVE)).sum())
    pool = np.where(istr)[0]
    matched = np.zeros(len(y), bool)
    matched[np.random.default_rng(11).choice(pool, n_one, replace=False)] = True

    conds = [(f"FIXED N={NATIVE} (today)", istr & (nn == NATIVE)),
             (f"FIXED N={SHORT}", istr & (nn == SHORT)),
             ("CROP-AUG all lengths", istr),
             ("CROP-AUG matched rows", matched)]
    models = {}
    print("=== held-out CORPUS items, scored at each crop length (7 classes, chance 0.143) ===")
    print("  rows fitted: fixed-length %d, crop-aug %d, matched %d"
          % (n_one, int(istr.sum()), n_one))
    print("  %-24s" % "train condition" + "".join(f"{n:>8d}" for n in LENS)
          + f"{'worst':>8}{'worstbal':>10}")
    for name, mask in conds:
        m = fit(mask); models[name] = m
        acc, bal = [], []
        for n in LENS:
            q = (~istr) & (nn == n)
            a, b = scores(predict(m, X[q]), y[q]); acc.append(a); bal.append(b)
        print("  %-24s" % name + "".join(f"{a:>8.3f}" for a in acc)
              + f"{min(acc):>8.3f}{min(bal):>10.3f}")

    PL = [2048, 4096, 8192, 16384, 32768, 65536]
    print("\n=== GENUINE out-of-corpus captures (canonical probes; cw/am/fm only) ===")
    cache = {}
    for n in PL:
        rows, lab, fl = [], [], []
        for _, want, iq in build_probes(np.random.default_rng(7), n):
            z, ctx = pp.preprocess(np.asarray(iq, np.complex128))
            rows.append(z); lab.append(classes.index(want))
            fl.append(fill_resampled(n, ctx["bw"]))
        cache[n] = (_readout_features(np.stack(rows),
                                      np.stack([pp.iq_features(r) for r in rows])),
                    np.array(lab), float(np.median(fl)))
    lab0 = cache[PL[0]][1]
    cnt = np.bincount(lab0, minlength=len(classes))
    maj = int(cnt.argmax())
    print("  probe composition: " + ", ".join(f"{classes[c]} {cnt[c]}" for c in np.nonzero(cnt)[0])
          + f"   -> majority baseline {cnt[maj]}/{cnt.sum()} = {cnt[maj]/cnt.sum():.3f} raw,"
          + f" {1.0/np.count_nonzero(cnt):.3f} balanced")
    print("  %-26s" % "train condition" + "".join(f"{n:>13d}" for n in PL))
    print("  %-26s" % "(median fill)" + "".join(f"{cache[n][2]:>13.3f}" for n in PL))
    print("  %-26s" % "" + "".join(f"{'raw/bal':>13}" for _ in PL))
    # THE CONTROL THAT USED TO BE ABSENT. A readout that learned nothing.
    row = []
    for n in PL:
        a, b = scores(np.full(len(cache[n][1]), maj), cache[n][1])
        row.append(f"{a:.3f}/{b:.3f}")
    print("  %-26s" % f"ALWAYS-{classes[maj]} (learns nothing)" + "".join(f"{s:>13}" for s in row))
    for name, _ in conds:
        row = []
        for n in PL:
            a, b = scores(predict(models[name], cache[n][0]), cache[n][1])
            row.append(f"{a:.3f}/{b:.3f}")
        print("  %-26s" % name + "".join(f"{s:>13}" for s in row))
    # ---- WHICH MODE? measured on accuracy, not on a fill-dispersion proxy ---------------
    # This block exists because the module used to recommend mode='fill' on the strength of
    # fill dispersion alone, with no accuracy measurement of it anywhere. Same readout, same
    # item-disjoint split, crops drawn from each CropConfig.
    print("\n=== WHICH CropConfig MODE? (same readout, item-disjoint, 5 draws/item) ===")
    mode_cfgs = [("off (no crop)", CropConfig(mode="off")),
                 ("loguniform guarded", CropConfig(mode="loguniform")),
                 ("loguniform UNguarded", CropConfig(mode="loguniform", guard_frac=0.0)),
                 ("fill U(0.05,1.0) guarded", CropConfig(mode="fill"))]
    m_items, m_draws = 120, 5
    msel = {c: rng.choice(by_cls[c], m_items, replace=False) for c in classes}
    bwc: dict[int, float] = {}
    caps = {int(i): _get(raw, int(i)) for c in classes for i in msel[c]}
    uid_m = np.array(sorted(caps))
    trm = set(np.random.default_rng(5).permutation(uid_m)[:int(0.7 * len(uid_m))].tolist())
    print("  %-26s" % "mode" + "".join(f"{n:>8d}" for n in LENS) + f"{'worst':>8}")
    for mname, mcfg in mode_cfgs:
        W2, F2, Y2, I2 = [], [], [], []
        for ci, c in enumerate(classes):
            for i in msel[c]:
                x = caps[int(i)]
                if mcfg.mode == "fill" and int(i) not in bwc:
                    bwc[int(i)] = float(pp.estimate_band(x)[1])
                for _ in range(m_draws):
                    view, _mt = random_crop(x, rng, mcfg, bwc.get(int(i)))
                    z, _ = pp.preprocess(np.asarray(view, np.complex128))
                    W2.append(z); F2.append(pp.iq_features(z)); Y2.append(ci); I2.append(int(i))
        Xm = _readout_features(np.stack(W2), np.stack(F2))
        Ym = np.array(Y2); Im = np.array(I2)
        fitm = np.array([i in trm for i in Im])
        A = Xm[fitm]; mu = A.mean(0); sd = A.std(0) + 1e-6
        Zt = np.hstack([(A - mu) / sd, np.ones((int(fitm.sum()), 1))])
        Wt = np.linalg.solve(Zt.T @ Zt + 3e-2 * len(Zt) * np.eye(Zt.shape[1]),
                             Zt.T @ np.eye(len(classes))[Ym[fitm]])
        # scored on the HELD-OUT items of the fixed-length pool built above, at each length,
        # so all modes are judged on identical, un-augmented evaluation data
        acc = []
        for n in LENS:
            q = (~istr) & (nn == n)
            Zq = np.hstack([(X[q] - mu) / sd, np.ones((int(q.sum()), 1))])
            acc.append(float(np.mean(np.argmax(Zq @ Wt, 1) == y[q])))
        print("  %-26s" % mname + "".join(f"{a:>8.3f}" for a in acc) + f"{min(acc):>8.3f}")
    print("  The default is the best WORST-LENGTH row here. Note that 'fill' loses despite")
    print("  winning the fill-dispersion table in `main()` -- dispersion is not accuracy.")

    print("\n  HOW TO READ THIS. Compare `bal` against 0.333 and against the ALWAYS-<majority>")
    print("  row, never `raw` against 0. On raw accuracy the constant predictor beats the")
    print("  crop-augmented readout at most lengths; on balanced accuracy cropping helps at")
    print("  the SHORT end and is at or below the degenerate baseline at N >= 16384.")
    print("  Cropping fixes the LOW-fill half of the lock and leaves the HIGH-fill half")
    print("  (N >= 32768 for a narrowband emitter) where it was. Only a LONGER corpus closes")
    print("  that half. Do not read this weak linear readout as a model score.")
    return 0


if __name__ == "__main__":
    _arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if _arg == "--interp":
        raise SystemExit(interp_check())
    if _arg == "--probe":
        raise SystemExit(probe())
    raise SystemExit(main())
