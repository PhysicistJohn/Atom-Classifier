"""Episode-batch fitness for the ES/RL training loop -- forward-only, no
gradients, net.eval() (deterministic given parameters + episode draw, which is
what makes common-random-numbers actually noise-free rather than adding a
second uncontrolled noise source on top of it).

Primary fitness-to-minimize = mean cross-entropy (smooth, gives ES useful rank
differences between candidates that tie on raw accuracy); mean accuracy is
also returned every call since it's the metric evaluate_ab_v2.py reports.

Resistance to gaming (Atom-Neural-RL project history's reward-hack lesson,
applied concretely): embedding collapse (every input mapping to ~the same
point) makes every query equidistant from every prototype, so CE saturates at
ln(n_way) -- collapse cannot score low CE, because the fitness IS the
classification objective, not a hand-built proxy on top of it.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

FIXED_SCALE = 10.0  # matches train_common.py's scale=nn.Parameter(10.0) INITIAL value;
                     # kept FIXED, not evolved -- a loss-shaping temperature external to
                     # the operator's own forward(), deliberately out of scope for this
                     # ES run (KISS simplification; trivial to append as a 56257th
                     # scalar to the evolved vector later if desired).


def sq_dist(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Squared euclidean [Q,N] without cdist (MPS-safe). Identical formula to
    train.py's sq_dist -- reimplemented here (not imported) so this module has
    no torch-training-loop-file dependency beyond the operator itself."""
    qn = (q * q).sum(1, keepdim=True)
    pn = (p * p).sum(1)
    return (qn + pn.unsqueeze(0) - 2.0 * (q @ p.t())).clamp_min(0.0)


def episode_fitness(net, dev, episodes, n_way: int) -> tuple[float, float]:
    """episodes: list of (sx,sf,sl,qx,qf,ql) tuples from ds.sample_episode.
    Returns (mean_cross_entropy, mean_accuracy) over the batch."""
    ces, accs = [], []
    net.eval()
    with torch.no_grad():
        for sx, sf, sl, qx, qf, ql in episodes:
            xb = torch.from_numpy(np.concatenate([sx, qx])).to(dev)
            fb = torch.from_numpy(np.concatenate([sf, qf])).to(dev)
            emb = net(xb, fb)
            se, qe = emb[: len(sx)], emb[len(sx):]
            sl_t = torch.from_numpy(sl).to(dev)
            protos = torch.stack([se[sl_t == c].mean(0) for c in range(n_way)])
            logits = -sq_dist(qe, protos) * FIXED_SCALE
            ql_t = torch.from_numpy(ql).to(dev)
            ces.append(float(F.cross_entropy(logits, ql_t)))
            accs.append(float((logits.argmax(-1) == ql_t).float().mean()))
    return float(np.mean(ces)), float(np.mean(accs))
