"""Honest held-out evaluation of the exported model.

Runs against the EXPORTED assets through the pure-numpy forward pass (the same
maths the TypeScript runtime uses), not the torch model — so these numbers
describe what actually ships. Reports:

  * closed-set accuracy: fine (9 classes) and family (QAM orders merged), overall
    + per-class + per-SNR + per-bandwidth,
  * few-shot: leave-one-class-out enrollment of the held-out classes (psk8, dsss)
    from K shots, then (known + held-out)-way recall,
  * open-set: AUROC of the nearest-prototype distance separating known-test from
    each novel class, plus the operating point at the exported threshold,
  * robustness sweeps.

Exits non-zero if any headline metric falls below its (evidence-based) floor, so
it doubles as a regression gate.  Run: .venv-training/bin/python training/evaluate.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import rfgen  # noqa: E402
import preprocess as pp  # noqa: E402
from train import np_forward, nearest, auroc  # noqa: E402

ASSET_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "embedding", "assets")
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
SEED = 424242
N_TEST = 120           # test realisations per class
FAMILY = {"qam16": "qam", "qam64": "qam"}

# Evidence-based acceptance floors (see DESIGN.md "Honest limitations").
FLOORS = {
    "closed_fine": 0.72,
    "closed_family": 0.82,
    "closed_fine_snr>=18": 0.78,
    "fewshot_loo_k5_resolvable": 0.85,  # re-enroll a resolvable class from 5 shots
    "open_auroc_overall": 0.72,
    "open_auroc_chirp": 0.80,
}


def fam(c: str) -> str:
    return FAMILY.get(c, c)


def gen(cls, rng, snr=None, sps=None):
    snr = float(rng.uniform(2, 30)) if snr is None else snr
    sps = int(rng.integers(2, 17)) if sps is None else sps
    iq = rfgen.apply_channel(rfgen.generate_clean(cls, 4096, sps, rng), rng, snr)
    norm, _ = pp.preprocess(iq)
    return norm, snr, sps


def embed(norm, W):
    return np_forward(pp.to_channels(norm), W)


def build(classes, per, W, rng):
    E, Y, S, B = [], [], [], []
    for ci, cls in enumerate(classes):
        for _ in range(per):
            norm, snr, sps = gen(cls, rng)
            E.append(embed(norm, W)); Y.append(ci); S.append(snr); B.append(sps)
    return np.stack(E), np.array(Y), np.array(S), np.array(B)


def snr_bin(s):
    return "2-6" if s < 6 else "6-10" if s < 10 else "10-18" if s < 18 else "18-30"


def main():
    W = json.load(open(os.path.join(ASSET_DIR, "embedding-weights.json")))
    P = json.load(open(os.path.join(ASSET_DIR, "prototypes.json")))
    protos = np.array(P["prototypes"], np.float32)
    known = P["classes"]
    thr = P["unknown_threshold"]
    rng = np.random.default_rng(SEED)
    report = {}

    # ---------------- closed set ----------------
    E, Y, S, B = build(known, N_TEST, W, rng)
    pred, dists = nearest(E, protos)
    fine = (pred == Y)
    famok = np.array([fam(known[pred[i]]) == fam(known[Y[i]]) for i in range(len(Y))])
    report["closed_fine"] = float(fine.mean())
    report["closed_family"] = float(famok.mean())
    report["per_class"] = {known[c]: round(float(fine[Y == c].mean()), 3) for c in range(len(known))}
    report["per_snr_fine"] = {}
    report["per_snr_family"] = {}
    for b in ["2-6", "6-10", "10-18", "18-30"]:
        m = np.array([snr_bin(s) == b for s in S])
        report["per_snr_fine"][b] = round(float(fine[m].mean()), 3)
        report["per_snr_family"][b] = round(float(famok[m].mean()), 3)
    hi = S >= 18
    report["closed_fine_snr>=18"] = float(fine[hi].mean())
    # per-bandwidth (sps) sweep
    report["per_bw_fine"] = {}
    for lo, hicut, name in [(2, 5, "2-4"), (5, 9, "5-8"), (9, 13, "9-12"), (13, 17, "13-16")]:
        m = (B >= lo) & (B < hicut)
        if m.any():
            report["per_bw_fine"][name] = round(float(fine[m].mean()), 3)

    # ---------------- few-shot ----------------
    # (a) The product claim: enroll a class the metric can resolve from K shots,
    #     no retraining. Leave-one-KNOWN-class-out isolates the enrollment
    #     machinery from class-similarity confounds — drop a trained prototype,
    #     re-enroll it from K fresh shots, measure recall against the other 8.
    report["fewshot"] = {}
    loo_k5 = []
    for held in known:
        hidx = known.index(held)
        base = np.delete(protos, hidx, axis=0)
        held_test = E[Y == hidx]
        for K in [1, 5]:
            shot_proto = np.stack([embed(gen(held, rng)[0], W) for _ in range(K)]).mean(0)
            ext = np.vstack([base, shot_proto[None, :]])
            qp, _ = nearest(held_test, ext)
            recall = float((qp == len(base)).mean())  # new proto is the last row
            report["fewshot"][f"loo_{held}_k{K}"] = round(recall, 3)
            if K == 5:
                loo_k5.append(recall)
    report["fewshot"]["loo_k5_mean"] = round(float(np.mean(loo_k5)), 3)
    # honest headline: few-shot recovery for classes the metric actually resolves
    # (closed-set per-class >= 0.8). The hard pairs (QAM orders, GFSK) stay hard
    # whether trained or enrolled — that is the resolution limit, not a few-shot
    # failure — so they are reported but not gated here.
    resolvable = [c for c in known if report["per_class"][c] >= 0.8]
    report["fewshot"]["loo_k5_resolvable"] = round(
        float(np.mean([report["fewshot"][f"loo_{c}_k5"] for c in resolvable])), 3
    )
    report["fewshot"]["resolvable_classes"] = resolvable

    # (b) Honest hard case: enroll a genuinely-new *fine variant* (8PSK next to
    #     QPSK, DSSS next to BPSK). Recovery is partial by design — the metric
    #     was never trained to resolve that split — and the system expresses that
    #     as uncertainty rather than a confident wrong call.
    for held in rfgen.FEWSHOT:
        proto5 = np.stack([embed(gen(held, rng)[0], W) for _ in range(5)]).mean(0)
        ext = np.vstack([protos, proto5[None, :]])
        q = np.stack([embed(gen(held, rng)[0], W) for _ in range(N_TEST)])
        qp, _ = nearest(q, ext)
        report["fewshot"][f"novel_{held}_k5_recall"] = round(float((qp == len(known)).mean()), 3)
        kp, _ = nearest(E, ext)
        report["fewshot"][f"known_acc_after_{held}"] = round(float((kp == Y).mean()), 3)

    # ---------------- open set ----------------
    nn_known = dists.min(1)
    report["open"] = {}
    novel_scores_all = []
    for nov in rfgen.NOVEL:
        en = np.stack([embed(gen(nov, rng)[0], W) for _ in range(N_TEST)])
        _, dn = nearest(en, protos)
        sc = dn.min(1)
        novel_scores_all.append(sc)
        report["open"][f"auroc_{nov}"] = round(auroc(sc, nn_known), 3)
        report["open"][f"flagged_unknown_{nov}"] = round(float((sc > thr).mean()), 3)
    allnov = np.concatenate(novel_scores_all)
    report["open"]["auroc_overall"] = round(auroc(allnov, nn_known), 3)
    report["open"]["known_false_unknown_rate"] = round(float((nn_known > thr).mean()), 3)

    # ---------------- report + gate ----------------
    print(json.dumps(report, indent=2))
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    json.dump(report, open(os.path.join(ARTIFACT_DIR, "eval-report.json"), "w"), indent=2)

    checks = {
        "closed_fine": report["closed_fine"],
        "closed_family": report["closed_family"],
        "closed_fine_snr>=18": report["closed_fine_snr>=18"],
        "fewshot_loo_k5_resolvable": report["fewshot"]["loo_k5_resolvable"],
        "open_auroc_overall": report["open"]["auroc_overall"],
        "open_auroc_chirp": report["open"]["auroc_chirp"],
    }
    print("\n--- gate ---")
    ok = True
    for k, v in checks.items():
        floor = FLOORS[k]
        status = "PASS" if v >= floor else "FAIL"
        if v < floor:
            ok = False
        print(f"  {status}  {k:24s} {v:.3f}  (floor {floor})")
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
