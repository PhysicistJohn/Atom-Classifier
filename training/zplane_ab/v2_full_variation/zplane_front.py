"""A z-plane invariant spectral front end, and the cheapest test of whether it works.

THE PROBLEM IT TARGETS. Every failure in the overnight campaign traced to one thing: the
current front end (training/preprocess.py) resamples each capture so its occupied bandwidth
hits a canonical fraction and then centre-fits to a FIXED 1024-point grid. How much of that
grid carries signal therefore depends on the raw capture length, and models trained at one
length score 0 at another. This was shown to be a property of the DATA GEOMETRY, not of any
network: a plain ridge readout on the front-end output reproduces the same lock. So no
architecture behind that front end can fix it.

THE IDEA (user's, 2026-07-27). Feed the U-Net an invariant space instead. Two distinct
invariances, obtained by two distinct mechanisms:

  SAMPLE RATE / SCALE.  Changing fs scales the frequency axis, f -> f/k. On a LOG-frequency
      axis a scaling becomes a TRANSLATION. A U-Net is convolutional and therefore already
      translation-equivariant, so it gets scale invariance from its own structure with no new
      layer type. This is the z-plane statement: on the unit circle z = e^{jw}, a rate change
      is a radial/angular warp, and log-w linearises it.

  CAPTURE LENGTH.  A spectral representation with a FIXED bin count has the same shape no
      matter how many samples produced it. A longer capture buys a better estimate, not a
      differently-shaped input. The fill fraction that broke everything simply does not exist
      here.

DESIGN CHOICES, and why:
  - COMPLEX throughout. Modulation ORDER lives in phase structure; a magnitude-only spectrum
      discards exactly the higher-order statistics that separate PSK/QAM orders, which is why
      preprocess.iq_features exists as a hand-computed side channel in the first place.
  - Log frequency is singular at DC and baseband captures put energy there, so the warp is
      BILATERAL with a LINEAR CORE: linear within +/- f_core, logarithmic outside, mirrored
      for negative frequencies. A naive log would blow up on the most common case.
  - The spectrum is interpolated to a fixed bin count rather than segment-averaged, because
      coherent averaging across segments destroys the phase this representation exists to keep.

WHAT THIS FILE IS FOR. It builds ONLY the transform and puts a ridge readout on it -- no U-Net,
no training, minutes of CPU. If the length and rate sweeps flatten relative to the current front
end, the idea is real and worth building on. If they do not, we learned that cheaply. This is
the same method that diagnosed the fill-lock, run in reverse.

Run:  .venv-training/bin/python training/zplane_ab/v2_full_variation/zplane_front.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_V2 = os.path.dirname(os.path.abspath(__file__))
_TRAINING = os.path.dirname(os.path.dirname(_V2))
for p in (_TRAINING, _V2):
    if p not in sys.path:
        sys.path.insert(0, p)

import preprocess as pp  # noqa: E402

CORPUS = os.path.join(_TRAINING, "artifacts", "signallab-corpus")
N_BINS = 512          # fixed output width: this is what makes it length-invariant
F_CORE = 0.02         # linear within +/- this fraction of fs; logarithmic outside
DECADES = 2.4         # log span outside the core, down to the Nyquist edge


def _warp_axis(n_bins=N_BINS, f_core=F_CORE, decades=DECADES):
    """Bilateral linear-core-then-log frequency axis in cycles/sample, over (-0.5, 0.5).

    Half the bins go to the linear core (which contains DC and survives the singularity),
    half are shared between the two logarithmic wings."""
    n_core = n_bins // 2
    n_wing = (n_bins - n_core) // 2
    core = np.linspace(-f_core, f_core, n_core)
    wing = f_core * np.power(10.0, np.linspace(0, decades, n_wing + 1)[1:])
    wing = wing[wing < 0.5]
    if len(wing) < n_wing:                      # pad against the Nyquist edge
        wing = np.concatenate([wing, np.full(n_wing - len(wing), 0.4999)])
    return np.concatenate([-wing[::-1], core, wing])


AXIS = _warp_axis()


def zplane_transform(iq, n_bins=N_BINS, normalise=True):
    """Complex I/Q of ANY length -> complex[n_bins] on the warped frequency axis.

    Length-invariant by construction: the output width is n_bins whatever len(iq) is.
    Scale-covariant by construction: multiplying fs by k translates the content along the
    logarithmic wings instead of rescaling it."""
    z = np.asarray(iq, dtype=np.complex128)
    z = z - z.mean()
    n = len(z)
    # full-length FFT, then interpolate the COMPLEX spectrum onto the warped axis. Real and
    # imaginary parts are interpolated separately; interpolating magnitude/phase would wrap.
    S = np.fft.fftshift(np.fft.fft(z * np.hanning(n)))
    f = np.fft.fftshift(np.fft.fftfreq(n))
    out = (np.interp(AXIS, f, S.real) + 1j * np.interp(AXIS, f, S.imag))
    if normalise:
        rms = np.sqrt(np.mean(np.abs(out) ** 2) + 1e-12)
        out = out / rms
    return out.astype(np.complex64)


def _readout(z, n_pool=64):
    """Identical recipe for BOTH front ends, so the comparison is apples-to-apples.

    The first version of this fed raw real/imag/abs straight in: 1536-3072 features against
    166 training rows, with un-compressed FFT magnitudes spanning many orders of magnitude.
    The normal equations overflowed and BOTH front ends scored near the 0.143 chance floor,
    so the spread difference between them was noise between two broken measurements. Fixed
    by log-compressing magnitude, pooling to a fixed small width, and keeping a phase-
    difference channel (phase is why the transform is complex; discarding it would test the
    wrong thing)."""
    z = np.asarray(z)
    mag = np.abs(z)
    mag = mag / (np.sqrt(np.mean(mag ** 2)) + 1e-12)
    logmag = np.log10(mag + 1e-6)                       # compress the dynamic range
    dphi = np.diff(np.unwrap(np.angle(z)))              # phase structure, scale-free
    def pool(v, k):
        v = np.asarray(v, dtype=np.float64)
        m = (len(v) // k) * k
        return v[:m].reshape(k, -1).mean(1) if m else np.zeros(k)
    return np.concatenate([pool(logmag, n_pool), pool(dphi, n_pool),
                           pool(np.cos(dphi), n_pool // 2), pool(np.sin(dphi), n_pool // 2)])


def features(z):
    return _readout(z)


def current_front_end(iq):
    x, _ = pp.preprocess(np.asarray(iq, np.complex128))
    return _readout(x)


# ---------------------------------------------------------------------------------------
def ridge(Xtr, ytr, Xva, n_classes, lam=10.0):
    """Guarded: a near-zero per-feature sd used to blow the standardisation up and overflow
    the normal equations. Floor sd at a fraction of the global scale, clip, and solve in
    float64 with lstsq so a singular system degrades instead of exploding."""
    Xtr = np.nan_to_num(np.asarray(Xtr, np.float64))
    Xva = np.nan_to_num(np.asarray(Xva, np.float64))
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd = np.maximum(sd, 1e-3 * (np.median(sd) + 1e-12))
    A = np.clip((Xtr - mu) / sd, -12, 12)
    B = np.clip((Xva - mu) / sd, -12, 12)
    Y = np.eye(n_classes)[ytr]
    G = A.T @ A + lam * np.eye(A.shape[1])
    W = np.linalg.lstsq(G, A.T @ Y, rcond=None)[0]
    return (B @ W).argmax(1)


def bal_acc(pred, y, n_classes):
    accs = [float((pred[y == c] == c).mean()) for c in range(n_classes) if (y == c).any()]
    return float(np.mean(accs))


def main(per_class=40, seed=0):
    man = json.load(open(os.path.join(CORPUS, "corpus.json")))
    n_items, n_samp = int(man["count"]), int(man["sampleCount"])
    raw = np.memmap(os.path.join(CORPUS, "corpus.f32"), dtype="<f4", mode="r",
                    shape=(n_items, n_samp, 2))
    classes = sorted(man["classes"])
    cidx = {c: i for i, c in enumerate(classes)}
    items = man["items"]
    rng = np.random.default_rng(seed)

    sel, ys = [], []
    for c in classes:
        pool = [i for i, it in enumerate(items) if it["cls"] == c]
        take = rng.choice(pool, min(per_class, len(pool)), replace=False)
        sel += list(take); ys += [cidx[c]] * len(take)
    sel, ys = np.asarray(sel), np.asarray(ys)
    tr = rng.random(len(sel)) < 0.6
    print(f"[zpf] {len(sel)} captures, {len(classes)} classes, "
          f"train {tr.sum()} / eval {(~tr).sum()}, corpus N={n_samp}")

    def cap(i, N):
        a = np.asarray(raw[i][:N], dtype=np.float64)
        return a[:, 0] + 1j * a[:, 1]

    # Use the PROVEN readout from length_aug (log-PSD + envelope spectrum + envelope scalars
    # + the 12 cumulants), which reached 0.79-0.90 on this corpus. My first attempt rolled its
    # own on raw real/imag/abs and scored both front ends AT CHANCE, where the z-plane arm
    # looked like the winner purely because a dead readout is flat. Never invent the
    # instrument when a validated one exists.
    from length_aug import _readout_features

    def rep_current(i, N):
        x, _ = pp.preprocess(cap(i, N))
        return x, pp.features_from_channels(pp.to_channels(x))

    def rep_zplane(i, N):
        z = zplane_transform(cap(i, N))
        ch = np.stack([z.real, z.imag])
        return z, pp.features_from_channels(ch)

    LENGTHS = [2048, 4096, 8192, 16384]
    CHANCE = 1.0 / len(classes)
    VALID_FLOOR = 0.45      # matched-length score below this => the instrument is dead

    def build(rep, N, idx):
        ws, fs = [], []
        for i in idx:
            w, f = rep(i, N)
            ws.append(w); fs.append(f)
        return _readout_features(np.stack(ws), np.stack(fs))

    print(f"\n  TRAIN AT N=16384, EVALUATE ACROSS LENGTHS  (balanced accuracy, chance {CHANCE:.3f})")
    print(f"  {'front end':<16}" + "".join(f"{n:>10}" for n in LENGTHS)
          + f"{'spread':>9}{'valid?':>9}")
    out = {}
    for name, rep in (("current", rep_current), ("z-plane warp", rep_zplane)):
        Xtr = build(rep, 16384, sel[tr])
        row, scores = [], []
        for N in LENGTHS:
            Xva = build(rep, N, sel[~tr])
            b = bal_acc(ridge(Xtr, ys[tr], Xva, len(classes)), ys[~tr], len(classes))
            scores.append(b); row.append(f"{b:>10.3f}")
        matched = scores[-1]                      # trained and evaluated at the same length
        valid = matched >= VALID_FLOOR
        out[name] = (scores, valid)
        print(f"  {name:<16}" + "".join(row) + f"{max(scores)-min(scores):>9.3f}"
              f"{'yes' if valid else 'NO':>9}")

    print(f"\n  valid? = does the MATCHED-length score clear {VALID_FLOOR:.2f}. If not, that row's")
    print("  spread means nothing: a readout stuck at chance is perfectly flat, so 'low spread'")
    print("  would reward a dead representation. Read spread ONLY on rows marked valid.")
    if all(not v for _, v in out.values()):
        print("\n  BOTH ROWS INVALID -- this test concludes nothing. Do not read the spreads.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
