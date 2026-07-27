"""Where should a new supervised head ATTACH? Recomputed from saved probe reports.

WHY THIS EXISTS. The brief was a U-Net whose latent you can extend: add a dimension, train a
new variable against it. latent_transfer.py answered that question about the U-Net BOTTLENECK
and returned False everywhere, and the overnight report generalised that into "the structure
cannot support a reusable summary".

That inference was wrong, and the user caught it. A U-Net's skip connections are the design.
Its middle layer is NOT an information bottleneck the way an autoencoder's is -- nothing forces
it to be a complete summary, because the skips exist precisely so the decoder gets fine detail
without squeezing it through the middle. A near-empty middle layer is close to what the
architecture is meant to do, so it is not evidence against the architecture. It is evidence
about WHERE to attach, which is a wiring question.

So this re-scores the same saved reports with `trunk` and `embed` -- the layers actually trained
to be a summary -- as candidate attachment points, against the same two controls the bottleneck
was held to:

  CONTROL 1  the SAME network with random untrained weights. If a tap does not beat this, the
             training put nothing there and the tap is just a random projection of the input.
  CONTROL 2  feat12, twelve hand-computed scalars. A learned summary that loses to these is not
             worth carrying.

No retraining and no re-extraction: latent_transfer.py already stored every tap, every label
set, both heads and the random-init control. Only the verdict was hard-wired to `bott`.

Reported on BALANCED accuracy throughout, never raw accuracy. The label sets are heavily skewed
(class7 majority baseline 0.645) and raw accuracy flatters a head that predicts the majority.
The audit found the overnight report mixing the two, which is how a 0.78 appeared beside a 0.46
for the same cell.

Run:  .venv-training/bin/python training/zplane_ab/v2_full_variation/attach_point.py
"""
from __future__ import annotations

import json
import os

_V2 = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(_V2, "artifacts", "campaign4")

TAPS = ["bott", "trunk", "embed"]
SETS = ["class7", "profile37", "gsm7", "ofdm24"]     # bt2 excluded: n=40, too thin to read
HEAD = "linear"                                       # the "light head" the brief implies


def cell(node, sset, head=HEAD):
    s = ((node or {}).get("sets") or {}).get(sset) or {}
    return s.get(head) or {}


def sep(a, b):
    """Do two bootstrap CIs fail to overlap? Conservative and symmetric."""
    if not (a.get("ci95") and b.get("ci95")):
        return None
    (alo, ahi), (blo, bhi) = a["ci95"], b["ci95"]
    return ahi < blo or bhi < alo


def load(tag):
    p = os.path.join(ART, tag, "latent_transfer.json")
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    tags = [t for t in sorted(os.listdir(ART))
            if os.path.isdir(os.path.join(ART, t)) and load(t)]
    if not tags:
        print("no probe reports found"); return 1

    print("=" * 100)
    print("  ATTACHMENT POINT ANALYSIS -- balanced accuracy, linear head")
    print("  trained vs the SAME net randomly initialised, and vs 12 hand-computed scalars")
    print("=" * 100)

    for tag in tags:
        d = load(tag)
        rc = d.get("random_init_control") or {}
        f12 = (d.get("taps") or {}).get("feat12")
        print(f"\n### {tag}")
        print(f"  {'tap':<7}{'set':<11}{'trained':>9}{'random':>9}{'delta':>9}{'sep?':>7}"
              f"{'feat12':>9}{'vs f12':>9}{'sep?':>7}{'n':>7}")
        for tap in TAPS:
            node = (d.get("taps") or {}).get(tap)
            rnode = rc.get(tap)
            if not node:
                continue
            for sset in SETS:
                a, b = cell(node, sset), cell(rnode, sset)
                c = cell(f12, sset)
                if not a:
                    continue
                ta, ra = a.get("bal_acc"), b.get("bal_acc")
                fa = c.get("bal_acc")
                dr = (ta - ra) if (ta is not None and ra is not None) else None
                df = (ta - fa) if (ta is not None and fa is not None) else None
                print(f"  {tap:<7}{sset:<11}{ta if ta is None else f'{ta:>9.4f}'}"
                      f"{ra if ra is None else f'{ra:>9.4f}'}"
                      f"{'' if dr is None else f'{dr:>+9.4f}'}"
                      f"{str(sep(a, b)):>7}"
                      f"{fa if fa is None else f'{fa:>9.4f}'}"
                      f"{'' if df is None else f'{df:>+9.4f}'}"
                      f"{str(sep(a, c)):>7}{a.get('n', ''):>7}")

    # ---- the summary the decision actually turns on -------------------------------------
    print("\n" + "=" * 100)
    print("  DOES ANY TAP BEAT ITS RANDOM-INIT CONTROL? (delta > 0 AND CIs separated)")
    print("=" * 100)
    print(f"  {'trial':<26}{'tap':<8}" + "".join(f"{s:>12}" for s in SETS))
    for tag in tags:
        d = load(tag); rc = d.get("random_init_control") or {}
        for tap in TAPS:
            node = (d.get("taps") or {}).get(tap); rnode = rc.get(tap)
            if not node:
                continue
            marks = []
            for sset in SETS:
                a, b = cell(node, sset), cell(rnode, sset)
                if not a or not b or a.get("bal_acc") is None or b.get("bal_acc") is None:
                    marks.append("-"); continue
                d_ = a["bal_acc"] - b["bal_acc"]
                marks.append(("BEATS" if sep(a, b) else "ns") if d_ > 0 else
                             ("WORSE" if sep(a, b) else "ns"))
            print(f"  {tag:<26}{tap:<8}" + "".join(f"{m:>12}" for m in marks))
    print("\n  BEATS = trained above random with non-overlapping CIs")
    print("  WORSE = random above trained with non-overlapping CIs  <-- training removed information")
    print("  ns    = difference not resolvable at this sample size")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
