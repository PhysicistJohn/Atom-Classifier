"""training/canonical_probe.py's length sweep, applied to a live torch checkpoint.

canonical_probe.py scores an EXPORTED asset directory (embedding-weights.json +
prototypes.json) through np_forward. A campaign checkpoint has neither, so this module
reuses the parts that carry the science -- `build_probes` (textbook cw/am/fm built outside
the training pipeline, at any length), `LENGTHS`, and `GATE` -- and swaps only the forward
pass and the prototype source. Nothing about the probe signals is re-implemented here; if
canonical_probe.py's definitions change, this follows.

WHY THE SWEEP IS THE POINT. Both models measured so far are FILL-LOCKED, each competent
only near its own corpus's SAMPLE_COUNT, and the first version of canonical_probe.py hid
that by hard-coding N=4096 -- the OLD corpus's constant -- and then reporting the resulting
mismatch as "representational collapse". The gate is on the WORST length, never the best.

TWO DIFFERENCES FROM THE SHIPPED PROBE, both of which make this the EASIER test, so a
failure here is at least as damning as a failure there:
  * No unknown-threshold rejection. prototypes.json ships one; a campaign checkpoint has
    none, so every probe is forced to a class rather than being allowed to answer
    "unknown". That can only raise the score.
  * Prototypes come from the campaign's own enroll pool rather than the exported file.

FILL UNDER THE NATIVE CONDITION. native_preprocess does not resample; it centre-fits to
L_OUT_NATIVE. So fill is min(N, L_OUT)/L_OUT and the sweep spans 0.25 .. 1.00 at
LENGTHS=(4096, 8192, 16384, 32768) -- a genuinely different geometry from the resampled
path, where fill is n*bw/TARGET_FRAC and a narrowband emitter cannot reach 0.75 from a
16384-sample corpus at all. Both are reported so the number is never read against the
wrong geometry.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

_V2_DIR = os.path.dirname(os.path.abspath(__file__))
_ZPAB_DIR = os.path.dirname(_V2_DIR)
_TRAINING_DIR = os.path.dirname(_ZPAB_DIR)
for p in (_TRAINING_DIR, _ZPAB_DIR, _V2_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import preprocess as pp  # noqa: E402
import native_preprocess as npp  # noqa: E402
from canonical_probe import build_probes, LENGTHS, GATE  # noqa: E402


def _standardize_features(features, fmean, fstd):
    """Apply the exact scalar-feature transform used by ``prepare_data_v2``.

    Canonical probes are built outside the cached corpus pipeline, so their 12 raw
    ``iq_features`` must be transformed here before entering a network trained on
    standardized features.  Silently accepting malformed statistics would make this
    probe exercise a different model from the one evaluated on the cached pools.
    """
    features = np.asarray(features, dtype=np.float32)
    fmean = np.asarray(fmean, dtype=np.float32)
    fstd = np.asarray(fstd, dtype=np.float32)
    if features.ndim < 1 or fmean.ndim != 1 or fstd.ndim != 1:
        raise ValueError("features must have a feature axis; fmean/fstd must be 1-D")
    if fmean.shape != fstd.shape or features.shape[-1] != fmean.shape[0]:
        raise ValueError(
            f"feature/stat shape mismatch: features {features.shape}, "
            f"fmean {fmean.shape}, fstd {fstd.shape}")
    if not (np.isfinite(features).all() and np.isfinite(fmean).all()
            and np.isfinite(fstd).all()):
        raise ValueError("features and feature statistics must be finite")
    if np.any(fstd <= 0):
        raise ValueError("every feature standard deviation must be positive")
    return ((features - fmean) / fstd).astype(np.float32, copy=False)


def _corpus_capture_length():
    """Read the matched raw capture length from the corpus manifest, if available."""
    manifest = os.path.join(_TRAINING_DIR, "artifacts", "signallab-corpus", "corpus.json")
    try:
        with open(manifest) as f:
            value = int(json.load(f)["sampleCount"])
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return value if value > 0 else None


def _summarize_rows(rows, matched_length, gate=GATE):
    """Summarize a sweep without mistaking a flat dead readout for invariance.

    The ship bar is worst-length balanced accuracy.  Separately, the curve is only a
    valid invariance measurement when the corpus-matched point is demonstrably alive.
    """
    if not rows:
        raise ValueError("canonical probe sweep produced no rows")

    worst_correct = min(float(r["correct"]) for r in rows)
    best_correct = max(float(r["correct"]) for r in rows)
    worst_bal = min(float(r["bal"]) for r in rows)
    best_bal = max(float(r["bal"]) for r in rows)
    worst_cos = max(float(r["cos"]) for r in rows)

    reasons = []
    matched = None
    if matched_length is None:
        reasons.append("matched capture length is unknown")
    else:
        matched = next((r for r in rows if int(r["n"]) == int(matched_length)), None)
        if matched is None:
            reasons.append(f"matched capture length N={int(matched_length)} was not swept")
    if matched is not None:
        matched_bal = float(matched["bal"])
        if not np.isfinite(matched_bal):
            reasons.append("matched-condition balanced accuracy is non-finite")
        elif matched_bal < float(gate["min_valid_bal_acc"]):
            reasons.append(
                f"matched-condition balanced accuracy {matched_bal:.4f} is below "
                f"validity floor {float(gate['min_valid_bal_acc']):.4f}")
        if int(matched.get("n_predicted_classes", 0)) < 2:
            reasons.append("matched-condition readout predicts fewer than two classes")
    else:
        matched_bal = None

    finite = np.isfinite([worst_correct, best_correct, worst_bal, best_bal, worst_cos]).all()
    if not finite:
        reasons.append("sweep summary contains non-finite values")
    valid = not reasons
    passes_gate = bool(
        valid
        and worst_bal >= float(gate["min_bal_acc"])
        and worst_cos <= float(gate["max_mean_pairwise_cos"])
    )
    return dict(
        worst_correct=round(worst_correct, 4),
        best_correct=round(best_correct, 4),
        worst_bal=round(worst_bal, 4),
        best_bal=round(best_bal, 4),
        worst_cos=round(worst_cos, 4),
        matched_length=None if matched_length is None else int(matched_length),
        matched_bal=None if matched_bal is None else round(matched_bal, 4),
        valid=valid,
        validity_reasons=reasons,
        passes_gate=passes_gate,
    )


@torch.no_grad()
def sweep(net, protos, classes, dev, fmean, fstd, condition="native", in_len=None,
          lengths=LENGTHS, matched_length=None, seed=7, verbose=False):
    """Score `net` on the canonical probes at every length in `lengths`.

    protos : (n_classes, embed_dim) float array -- the campaign's enroll prototypes.
    fmean/fstd : feature statistics fitted on the campaign's training pool.
    Returns a dict with a per-length table, the WORST-length numbers, and the gate result.
    """
    if condition not in ("native", "resampled"):
        raise ValueError(f"unknown preprocessing condition {condition!r}")
    module = npp if condition == "native" else pp
    classes = list(classes)
    lengths = tuple(int(n) for n in lengths)
    if not lengths or any(n <= 0 for n in lengths):
        raise ValueError("lengths must contain positive capture lengths")
    in_len = int(in_len or getattr(module, "L_OUT_NATIVE", pp.L_OUT))
    protos = np.asarray(protos, dtype=np.float32)
    if protos.ndim != 2 or protos.shape[0] != len(classes):
        raise ValueError(
            f"prototype/class mismatch: prototypes {protos.shape}, classes {len(classes)}")
    probe_classes = {cls for _label, cls, _iq in build_probes(np.random.default_rng(seed), 64)}
    missing = sorted(probe_classes.difference(classes))
    if missing:
        raise ValueError(f"campaign classes do not cover canonical probes: {missing}")
    if matched_length is None:
        matched_length = _corpus_capture_length()

    net = net.to(dev).eval()
    rows = []
    for n in lengths:
        xs, fs, want, fills = [], [], [], []
        for _label, cls, iq in build_probes(np.random.default_rng(seed), n):
            z, ctx = module.preprocess(np.asarray(iq, np.complex128), l_out=in_len)
            xs.append(module.to_channels(z)); fs.append(module.iq_features(z))
            want.append(classes.index(cls))
            if condition == "native":
                fills.append(min(int(n), in_len) / in_len)
            else:
                new_len = max(
                    64, round(int(n) * float(ctx["resample_frac"]) / float(pp.TARGET_FRAC)))
                fills.append(min(new_len, in_len) / in_len)
        xb = torch.from_numpy(np.stack(xs).astype(np.float32)).to(dev)
        standardized = _standardize_features(np.stack(fs), fmean, fstd)
        fb = torch.from_numpy(standardized).to(dev)
        emb = net(xb, fb).float().cpu().numpy()
        d = ((emb[:, None, :] - protos[None, :, :]) ** 2).sum(-1)
        pred = d.argmin(1)
        want = np.asarray(want)
        correct = float(np.mean(pred == want))
        E = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        iu = np.triu_indices(len(E), 1)
        cos = float((E @ E.T)[iu].mean())
        # BALANCED too: the probe set is 15 cw / 2 am / 2 fm, so the constant predictor
        # "always cw" scores 0.789 raw. Reporting only raw accuracy here is how a readout
        # that learned nothing gets mistaken for a working model.
        per = [float(np.mean(pred[want == c] == c)) for c in np.unique(want)]
        rows.append(dict(n=int(n), fill=round(float(np.median(fills)), 4),
                         fill_min=round(float(np.min(fills)), 4),
                         fill_max=round(float(np.max(fills)), 4),
                         correct=round(correct, 4), bal=round(float(np.mean(per)), 4),
                         cos=round(cos, 4),
                         n_predicted_classes=int(np.unique(pred).size),
                         per_class={classes[c]: round(float(np.mean(pred[want == c] == c)), 3)
                                    for c in np.unique(want)}))
        if verbose:
            print(f"  N={n:>6} fill {rows[-1]['fill']:.3f} correct {correct:.3f} "
                  f"bal {rows[-1]['bal']:.3f} cos {cos:+.3f}", flush=True)
    cnt = np.bincount(np.asarray(want), minlength=len(classes))
    out = _summarize_rows(rows, matched_length)
    out.update(
        rows=rows,
        majority_baseline=round(float(cnt.max() / cnt.sum()), 4),
        n_probes=int(cnt.sum()),
        lengths=list(lengths), gate=dict(GATE), condition=condition, in_len=in_len,
        note="gate is on WORST balanced accuracy and is valid only if the matched-length "
             "balanced accuracy clears its validity floor. Raw accuracy and the 0.789 "
             "majority baseline are diagnostics only; no unknown-threshold rejection makes "
             "this the easier test.")
    return out


if __name__ == "__main__":
    import pool_cache
    from unet_transfer import TransferUNet
    ck = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        _V2_DIR, "artifacts", "campaign3", "c3_unet_recon_only")
    torch.set_num_threads(4)
    d = pool_cache.load("native")
    net = TransferUNet()
    net.load_state_dict(torch.load(os.path.join(ck, "state_dict.pt"), map_location="cpu"))
    protos = np.load(os.path.join(ck, "prototypes.npy"))
    r = sweep(
        net, protos, list(d["classes"]), torch.device("cpu"),
        fmean=d["fmean"], fstd=d["fstd"], in_len=d["input_length"], verbose=True)
    print(json.dumps({k: v for k, v in r.items() if k != "rows"}, indent=1))
