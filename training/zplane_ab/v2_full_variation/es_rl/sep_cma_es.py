"""Fresh, dependency-free separable ("sep-") CMA-ES (Ros & Hansen, PPSN 2008,
"A Simple Modification in CMA-ES Achieving Linear Time and Space Complexity").

Reimplemented from scratch for Atom-Classifier -- NOT imported cross-repo.
Mirrors the (mu/mu_w, lambda)-CMA-ES skeleton read this session in
Atom-Neural-RL/src/atom_neural_rl/cma.py (mirrored sampling, weighted
recombination, cumulation paths pc/ps, step-size control via ||ps||/chiN) but
replaces the full n x n covariance matrix (and its per-generation
eigendecomposition, O(n^3)) with a diagonal vector D of per-parameter standard
deviations, an O(n) update -- the standard fix for scaling CMA-ES-style
evolution strategies past a few hundred parameters (see also Salimans et al.
2017, "Evolution Strategies as a Scalable Alternative to Reinforcement
Learning"). At n=56,256 (this operator's full parameter count) full CMA-ES's
covariance update is ~1.8e14 flops/generation; sep-CMA-ES's diagonal update is
~5.6e4 flops/generation -- the only reason full-parameter ES is tractable here
at all.

No off-diagonal (cross-parameter) covariance is modeled -- a real, principled
capability reduction versus full CMA-ES, not a bug.
"""

from __future__ import annotations

import numpy as np


class SepCMAES:
    """(mu/mu_w, lambda) separable CMA-ES. Diagonal covariance only --
    no matrix, no eigendecomposition. O(n) per generation at n=56256.

    minimize-convention: tell() expects fitnesses where LOWER is better
    (this project's fitness is mean cross-entropy loss).
    """

    def __init__(self, x0, d0, sigma0: float = 1.0, popsize: int | None = None, seed: int = 0):
        self.n = int(np.asarray(x0).size)
        self.mean = np.asarray(x0, dtype=np.float64).copy()
        self.sigma = float(sigma0)
        self.D = np.asarray(d0, dtype=np.float64).copy()  # per-parameter STD, not variance
        assert self.D.shape == (self.n,), f"d0 shape {self.D.shape} != ({self.n},)"
        self.rng = np.random.default_rng(seed)

        n = self.n
        lam = popsize or 48
        if lam % 2 == 1:
            lam += 1
        self.lam, self.mu = lam, lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        w /= w.sum()
        self.weights = w
        self.mueff = 1.0 / np.sum(w ** 2)

        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        cmu = min(1 - c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        # sep-CMA-ES's own published correction: diagonal-only adaptation
        # discards the off-diagonal terms' contribution, so both covariance
        # learning rates are scaled up by (n+2)/3 to compensate (Ros & Hansen
        # 2008, Eq. for c1_sep/cmu_sep) -- not an ad hoc guess.
        self.c1 = min(1.0, c1 * (n + 2) / 3.0)
        self.cmu = min(1.0 - self.c1, cmu * (n + 2) / 3.0)
        self.damps = 1 + 2 * max(0.0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.gen = 0
        self._last_z = None
        self._last_y = None

    def ask(self) -> np.ndarray:
        half = self.lam // 2
        z = self.rng.standard_normal((half, self.n))
        z = np.concatenate([z, -z], axis=0)  # mirrored sampling (antithetic pairs)
        self._last_z, self._last_y = z, self.D[None, :] * z  # elementwise -- O(n)
        return self.mean + self.sigma * self._last_y

    def tell(self, solutions, fitnesses) -> None:
        """fitnesses: array-like, lower is better. `solutions` is accepted for
        interface symmetry with cma.py's ask/tell contract but is unused --
        the z/y draws cached from ask() are what the update actually needs."""
        fitnesses = np.asarray(fitnesses, dtype=np.float64)
        order = np.argsort(fitnesses)
        z, y = self._last_z[order], self._last_y[order]
        z_mu, y_mu = z[: self.mu], y[: self.mu]
        z_w = np.einsum("i,in->n", self.weights, z_mu)
        y_w = np.einsum("i,in->n", self.weights, y_mu)
        self.mean = self.mean + self.sigma * y_w

        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * z_w
        ps_norm = np.linalg.norm(self.ps)
        hsig = ps_norm / np.sqrt(1 - (1 - self.cs) ** (2 * (self.gen + 1))) / self.chiN < (1.4 + 2 / (self.n + 1))
        self.pc = (1 - self.cc) * self.pc + (hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff)) * y_w

        delta_hsig = (1 - hsig) * self.cc * (2 - self.cc)
        rank_mu_diag = np.einsum("i,in->n", self.weights, y_mu ** 2)  # elementwise square, not outer product
        Dsq = self.D ** 2
        Dsq = (1 - self.c1 - self.cmu) * Dsq + self.c1 * (self.pc ** 2 + delta_hsig * Dsq) + self.cmu * rank_mu_diag
        self.D = np.sqrt(np.maximum(Dsq, 1e-300))

        self.sigma *= np.exp((self.cs / self.damps) * (ps_norm / self.chiN - 1))
        self.gen += 1


def build_d0(net) -> np.ndarray:
    """Per-parameter initial std vector, built once from the network's own
    init-code constants (zplane_backbone.py), NOT a uniform scalar -- several
    parameter groups are literally constant (zero variance) at warm-start
    (pole/zero radius+angle, gain, thresh_raw are broadcast/constant fills)
    while lift/mix/proj/fc1/fc2 have genuinely different measured init stds.
    A single scalar sigma0 is structurally the wrong fit for a parameter
    vector this heterogeneous.

    Order is determined by net.named_parameters()'s own traversal (backbone
    submodule first in its __init__ registration order, then fc1, then fc2)
    -- guaranteed to match torch.nn.utils.parameters_to_vector's flatten
    order exactly, since both walk the same named_parameters() sequence.
    """
    W = net.backbone.W  # width, e.g. 64
    group_sigma = {
        "backbone.pole_radius_logit": 0.2,    # logit units; sigmoid most responsive near warm-start's ~0
        "backbone.pole_angle": 0.3,           # radians (~17deg)
        "backbone.zero_radius_logit": 0.2,
        "backbone.zero_angle": 0.3,
        "backbone.gain_re": 0.1,              # around warm-start gain = 1+0j
        "backbone.gain_im": 0.1,
        "backbone.thresh_raw": 1.0,           # softplus(-6) is nearly flat; needs a bigger raw step to move b at all
        "backbone.lift_re": 1.0,
        "backbone.lift_im": 1.0,              # == measured init std (in_channels=1)
        "backbone.mix_re": 0.25 / W ** 0.5,   # == MIX_INIT_SCALE/sqrt(W), measured
        "backbone.mix_im": 0.25 / W ** 0.5,
        "backbone.proj_re": 0.25 / W ** 0.5,
        "backbone.proj_im": 0.25 / W ** 0.5,
        "fc1.weight": (3 * (2 * W + 12)) ** -0.5,  # nn.Linear default init std, fan_in = 2W+N_FEATURES(12)
        "fc1.bias": (3 * (2 * W + 12)) ** -0.5,
        "fc2.weight": (3 * 96) ** -0.5,             # fan_in == HIDDEN == 96
        "fc2.bias": (3 * 96) ** -0.5,
    }
    parts = []
    for name, p in net.named_parameters():
        if name not in group_sigma:
            raise KeyError(f"build_d0: no sigma group registered for parameter {name!r}")
        parts.append(np.full(p.numel(), group_sigma[name], dtype=np.float64))
    return np.concatenate(parts)
