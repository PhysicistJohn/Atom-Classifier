"""ComplexUNetMultiTask with the encoder-audit fixes, as switchable options.

WHY A NEW FILE RATHER THAN AN EDIT. unet_multitask.py is imported by campaign3.py, which
a running campaign is still launching children against, and its state_dicts are on disk for
four scored checkpoints. Every option here defaults to the EXISTING behaviour, so
`TransferUNet()` with no arguments is architecturally identical to `ComplexUNetMultiTask()`
and loads the same state_dict -- but the campaign never has to touch the file the other
campaign is using.

WHAT THE AUDIT FOUND, AND WHICH SWITCH ADDRESSES IT. All four are measurements on the
DEPLOYED config (base=8, depth=4, taps=7) with real corpus rows, not on the toy net in
unet_multitask.py's __main__:

  1. THE ENCODER IS DEAD AT INITIALISATION.  `magnorm` / `modrelu_init`
     ComplexModReLU initialises thresh_raw at -2.0, i.e. b = -softplus(-2.0) = -0.1269, and
     subtracts that from every feature MAGNITUDE at all ten stacked activations in
     downs+bottleneck. Magnitude decays monotonically and the bottleneck is 100.0% exactly
     zero before a single gradient step (mean |feature|: input 0.85 -> down0 0.35 -> down1
     0.14 -> down2 0.016 -> down3 0.0016 -> bottleneck 3.7e-09). clamp(mag+b, min=0) has
     zero derivative in the dead region, so no gradient reaches the deep encoder either --
     the classification head's gradient at init is ~8.7e4 times stronger into the 12
     handcrafted features than into the entire 256-d latent. After 700 episodes of training
     the bottleneck is still 97.3% exactly zero.
       `modrelu_init=-8.0` sets b ~ 0 and flattens the decay at step 0.
       `magnorm=True` inserts a phase-equivariant complex RMS normalisation before every
       modReLU, which holds throughout training rather than only at initialisation:
       bottleneck 100% dead -> ~20% dead, encoder init gradient norms ~8e-2 -> ~1e2.
     RISK, stated because it is untested: MagNorm divides by the per-(batch,channel) RMS
     over time, which removes cross-channel magnitude relations at every block. The largest
     measured per-class failure (fm collapsing into cw, the two constant-envelope classes)
     is an envelope-statistics failure, so a normalisation that discards channel-relative
     amplitude is not obviously safe there. That is why `head_pool="acf"` exists.

  2. THE SKIPS ROUTE AROUND THE BOTTLENECK COMPLETELY.  `skip_dropout`
     On the trained c3_unet_recon_only checkpoint, zeroing the ENTIRE bottleneck changes
     reconstruction coherence by exactly 0.0000; scrambling it across the batch, also
     exactly 0.0000. Zeroing skips[1..3] changes it by <= 0.0011. Only skips[0] matters --
     the deployed "U-Net" is functionally a 4-conv full-resolution FIR filter. The module
     docstring of unet_multitask.py argues two mechanisms keep the bottleneck honest; both
     are measured to have failed. Dropping each skip independently during training (the
     all-dropped case included, by construction) is the standard way to force the decoder
     to be able to work from the bottleneck alone.

  3. THE CLASSIFIER READS THE HANDCRAFTED FEATURES, NOT THE LATENT.  `feat_dropout`
     forward() concatenates preprocess.iq_features into the trunk. A RANDOMLY INITIALISED
     U-Net with the same concatenation reaches 0.482 class accuracy against the trained
     one's 0.528. The classification-ONLY U-Net has the least informative bottleneck of the
     three campaign3 variants (0.244 vs recon_only's 0.409), which directly refutes "a lazy
     bottleneck costs classification accuracy directly": the head satisfies the class loss
     through the 12-feature shortcut instead. Dropping the feat block per sample forces the
     trunk to work without it some of the time.

  4. NAIVE `v[..., ::2]` ALIASES THE BOTTLENECK PATH.  `antialias`
     Skips are appended BEFORE decimation, so the decoder never sees the aliasing and
     nothing pressures the encoder to learn an anti-alias filter -- the entire cost falls on
     the one path the classifier reads. Measured fraction of capture energy folded at the
     four decimations, averaged over 700 real pool rows: 8.7 / 17.1 / 26.8 / 36.2%, and for
     bluetooth 24.9 / 48.0 / 69.9 / 82.0%. Rated MEDIUM confidence in the audit, because the
     preceding 7-tap CBlocks can in principle learn the filter; the number is an upper bound
     on the damage, not a measurement of it. A fixed [1,2,1]/4 complex lowpass costs no
     parameters and is phase-equivariant.

  5. THE HEAD READS ONLY mean/std OF |bottleneck|.  `head_pool`
     That is an envelope statistic with no frequency resolution, and fm collapses to 0.052
     with 53% of fm captures predicted as cw -- the other constant-envelope class. Time-
     resolved pooling does NOT rescue it (an 8-chunk probe scores 0.387 against pooled
     0.409), so the information is absent from the bottleneck rather than pooled away, and
     the fix has to add a different KIND of statistic. `acf` appends per-channel lag
     autocorrelation magnitudes |mean_t(v[t] conj(v[t+k]))|, which are global-phase
     invariant by construction and do carry frequency structure.

PHASE EQUIVARIANCE IS THE ONE DESIGN CLAIM THAT SURVIVED AUDIT, and every option here
preserves it. __main__ asserts it per option rather than trusting the argument -- and it
asserts it on a LIVE bottleneck, because at the deployed init the bottleneck is 100% dead,
which makes the embedding trivially invariant to everything and the check vacuous. That is
exactly how the original invariance check passed while measuring nothing.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_V2_DIR = os.path.dirname(os.path.abspath(__file__))
_ZPAB_DIR = os.path.dirname(_V2_DIR)
_TRAINING_DIR = os.path.dirname(_ZPAB_DIR)
for p in (_TRAINING_DIR, _ZPAB_DIR, _V2_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from model import EMBED_DIM, N_FEATURES  # noqa: E402
from complex_multiscale_backbone import ComplexConv1d, ComplexModReLU  # noqa: E402
from multitask_autoencoder import PARAM_KEYS  # noqa: E402

ACF_LAGS = (1, 2, 4, 8, 16, 32)


class ComplexMagNorm(nn.Module):
    """z / sqrt(mean_t |z|^2), per (batch, channel). Phase-equivariant BY CONSTRUCTION:
    |e^{i phi} z| = |z|, so the divisor is unchanged and the phase passes straight through.

    This is the fix for the modReLU death cascade that holds during training rather than
    only at initialisation -- modReLU subtracts a fixed constant from magnitude, so once
    magnitude has decayed the constant dominates; renormalising before each activation keeps
    the two on the same scale at every depth."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt((z.real ** 2 + z.imag ** 2).mean(dim=-1, keepdim=True) + self.eps)
        return z / rms


class CBlock(nn.Module):
    """Complex conv -> (optional magnorm) -> phase-equivariant activation, twice."""

    def __init__(self, cin: int, cout: int, taps: int = 7, magnorm: bool = False,
                 modrelu_init: float = -2.0):
        super().__init__()
        self.c1, self.a1 = ComplexConv1d(cin, cout, taps), ComplexModReLU(cout)
        self.c2, self.a2 = ComplexConv1d(cout, cout, taps), ComplexModReLU(cout)
        self.n1 = ComplexMagNorm() if magnorm else None
        self.n2 = ComplexMagNorm() if magnorm else None
        if modrelu_init != -2.0:
            with torch.no_grad():
                self.a1.thresh_raw.fill_(modrelu_init)
                self.a2.thresh_raw.fill_(modrelu_init)

    def forward(self, z):
        v = self.c1(z)
        v = self.a1(self.n1(v) if self.n1 is not None else v)
        v = self.c2(v)
        return self.a2(self.n2(v) if self.n2 is not None else v)


def _lowpass121(v: torch.Tensor) -> torch.Tensor:
    """Fixed [1,2,1]/4 complex lowpass. Complex-linear (the same real kernel on real and
    imaginary parts), so phase equivariance is preserved exactly."""
    C = v.shape[1]
    k = v.real.new_tensor([0.25, 0.5, 0.25]).view(1, 1, 3).expand(C, 1, 3)
    re = F.conv1d(F.pad(v.real, (1, 1), mode="replicate"), k, groups=C)
    im = F.conv1d(F.pad(v.imag, (1, 1), mode="replicate"), k, groups=C)
    return torch.complex(re, im)


def _acf_feats(v: torch.Tensor, lags=ACF_LAGS) -> torch.Tensor:
    """|mean_t( v[t] conj(v[t+k]) )| per channel, for several lags.

    Global-phase invariant by construction (the two rotations cancel inside the product) and
    unlike mean/std of |v| it carries correlation-decay structure.  It does *not* preserve
    the autocorrelation phase: for a pure tone its value is constant across lag.  New
    experiments that need frequency structure should use ``head_pool="acf_complex"``."""
    out = []
    L = v.shape[-1]
    for k in lags:
        if k >= L:
            out.append(torch.zeros(v.shape[0], v.shape[1], device=v.device, dtype=v.real.dtype))
            continue
        out.append((v[..., :-k] * torch.conj(v[..., k:])).mean(-1).abs())
    return torch.cat(out, dim=-1)


def _complex_acf_feats(v: torch.Tensor, lags=ACF_LAGS) -> torch.Tensor:
    """Real and imaginary autocorrelation at several lags, per channel.

    ``mean(v[t] * conj(v[t+k]))`` is invariant to a global input phase, while its complex
    angle retains the lag-dependent frequency information discarded by ``_acf_feats``.
    """
    out = []
    L = v.shape[-1]
    for k in lags:
        if k >= L:
            corr = torch.zeros(
                v.shape[0], v.shape[1], device=v.device, dtype=v.dtype
            )
        else:
            corr = (v[..., :-k] * torch.conj(v[..., k:])).mean(-1)
        out.extend((corr.real, corr.imag))
    return torch.cat(out, dim=-1)


def _chunk_pool(mag: torch.Tensor, k: int = 4) -> torch.Tensor:
    B, C, L = mag.shape
    m = mag[..., : (L // k) * k].reshape(B, C, k, L // k)
    return torch.cat([m.mean(-1).reshape(B, -1), m.std(-1).reshape(B, -1)], dim=-1)


class TransferUNet(nn.Module):
    """forward(x, feat) -> embedding (harness contract). Reconstruction, profile logits and
    parameter predictions are exposed as attributes, as in ComplexUNetMultiTask.

    All defaults reproduce ComplexUNetMultiTask exactly, including parameter names and
    shapes, so a campaign3 state_dict loads into TransferUNet() unchanged.
    """

    def __init__(self, base: int = 8, depth: int = 4, taps: int = 7,
                 embed_dim: int = EMBED_DIM, n_features: int = N_FEATURES,
                 n_profiles: int = 37, hidden: int = 128,
                 magnorm: bool = False, modrelu_init: float = -2.0,
                 skip_dropout: float = 0.0, feat_dropout: float = 0.0,
                 antialias: bool = False, head_pool: str = "meanstd",
                 residual_recon: bool = False):
        super().__init__()
        self.depth = depth
        self.skip_dropout = float(skip_dropout)
        self.feat_dropout = float(feat_dropout)
        self.antialias = bool(antialias)
        self.head_pool = head_pool
        self.residual_recon = bool(residual_recon)
        chans = [base * (2 ** i) for i in range(depth)]     # base=8 -> [8,16,32,64]
        self.downs = nn.ModuleList()
        cin = 1
        for c in chans:
            self.downs.append(CBlock(cin, c, taps, magnorm, modrelu_init)); cin = c
        self.bottleneck = CBlock(chans[-1], chans[-1] * 2, taps, magnorm, modrelu_init)

        self.ups = nn.ModuleList()
        cin = chans[-1] * 2
        for c in reversed(chans):
            self.ups.append(CBlock(cin + c, c, taps, magnorm, modrelu_init)); cin = c
        self.out = ComplexConv1d(chans[0], 1, kernel_size=1)
        if self.residual_recon:
            # The input is already a materially useful estimate of the clean target.  A
            # denoiser should therefore learn a correction around identity, not begin by
            # replacing it with a random waveform.  Zeroing only the final projection makes
            # the initial map exactly identity while allowing that projection to learn on the
            # first step and the decoder beneath it on subsequent steps.
            nn.init.zeros_(self.out.conv_re.weight)
            nn.init.zeros_(self.out.conv_im.weight)

        bott = chans[-1] * 2
        pool_dim = {"meanstd": 2 * bott,
                    "chunk4": 8 * bott,
                    "acf": 2 * bott + len(ACF_LAGS) * bott,
                    "acf_complex": 2 * bott + 2 * len(ACF_LAGS) * bott}[head_pool]
        self.pool_dim = pool_dim
        self.trunk = nn.Sequential(nn.Linear(pool_dim + n_features, hidden), nn.ReLU(),
                                   nn.Dropout(0.1))
        self.embed_head = nn.Linear(hidden, embed_dim)
        self.profile_head = nn.Linear(hidden, n_profiles)
        self.param_head = nn.Linear(hidden, len(PARAM_KEYS))
        self.last_recon = None
        self.last_profile_logits = None
        self.last_params = None
        self.last_bottleneck_pool = None

    # -- config, recorded verbatim in the results row so a checkpoint is reconstructable --
    def config(self):
        return dict(magnorm=self.downs[0].n1 is not None,
                    modrelu_init=float(self.downs[0].a1.thresh_raw[0].item()),
                    skip_dropout=self.skip_dropout, feat_dropout=self.feat_dropout,
                    antialias=self.antialias, head_pool=self.head_pool,
                    pool_dim=self.pool_dim, residual_recon=self.residual_recon)

    @staticmethod
    def _up(v, size):
        re = F.interpolate(v.real, size=size, mode="nearest")
        im = F.interpolate(v.imag, size=size, mode="nearest")
        return torch.complex(re, im)

    def pool_bottleneck(self, v):
        mag = v.abs()
        if self.head_pool == "meanstd":
            return torch.cat([mag.mean(-1), mag.std(-1)], dim=-1)
        if self.head_pool == "chunk4":
            return _chunk_pool(mag, 4)
        if self.head_pool == "acf":
            return torch.cat([mag.mean(-1), mag.std(-1), _acf_feats(v)], dim=-1)
        return torch.cat(
            [mag.mean(-1), mag.std(-1), _complex_acf_feats(v)], dim=-1
        )

    def encode(self, x):
        """Encoder half only -- what every latent probe reads. Returns (bottleneck, skips)."""
        z = torch.complex(x[:, 0, :], x[:, 1, :]).unsqueeze(1)      # (B,1,L)
        skips = []
        v = z
        for blk in self.downs:
            v = blk(v)
            skips.append(v)                       # taken BEFORE decimation, deliberately
            v = _lowpass121(v)[..., ::2] if self.antialias else v[..., ::2]
        return self.bottleneck(v), skips

    def _embedding_from_bottleneck(
        self, v: torch.Tensor, feat: torch.Tensor
    ) -> torch.Tensor:
        """Run the shared classification heads from an already-computed bottleneck."""
        g = self.pool_bottleneck(v)
        self.last_bottleneck_pool = g
        f = feat
        if self.training and self.feat_dropout > 0.0:
            # drop the WHOLE handcrafted block per sample, not per element: the point is to
            # make the trunk able to classify without it, and elementwise dropout leaves
            # most of the shortcut intact.
            keep = (torch.rand(feat.shape[0], 1, device=feat.device) >= self.feat_dropout)
            f = feat * keep.to(feat.dtype)
        h = self.trunk(torch.cat([g, f], dim=-1))
        self.last_profile_logits = self.profile_head(h)
        self.last_params = self.param_head(h)
        return F.normalize(self.embed_head(h), dim=-1)

    def forward_embedding(self, x: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """Encoder/classification-only forward.

        This is numerically identical to the embedding returned by ``forward`` in eval
        mode, but deliberately skips the decoder.  Classification-only experiments and a
        classifier-only deployment should not pay for a reconstruction that no caller
        reads.  The regular ``forward`` remains unchanged as the multitask contract.
        """
        v, _skips = self.encode(x)
        self.last_recon = None
        return self._embedding_from_bottleneck(v, feat)

    def forward(self, x: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        v, skips = self.encode(x)
        emb = self._embedding_from_bottleneck(v, feat)

        for blk, skip in zip(self.ups, reversed(skips)):
            if self.training and self.skip_dropout > 0.0:
                # independent per-sample draws, so the all-skips-dropped case occurs
                # naturally and the decoder must be able to work from the bottleneck alone.
                keep = (torch.rand(skip.shape[0], 1, 1, device=skip.device)
                        >= self.skip_dropout).to(skip.real.dtype)
                skip = torch.complex(skip.real * keep, skip.imag * keep)
            v = self._up(v, skip.shape[-1])
            v = blk(torch.cat([v, skip], dim=1))
        correction = self.out(v).squeeze(1)
        if self.residual_recon:
            inp = torch.complex(x[:, 0, :], x[:, 1, :])
            self.last_recon = inp + correction
        else:
            self.last_recon = correction
        return emb


# -------------------------------------------------------------------------------------
# diagnostics the campaign records per trial, because they are the things that silently
# regressed last time
# -------------------------------------------------------------------------------------
@torch.no_grad()
def encoder_health(net, xb, fb):
    """Is the encoder alive? Per-level mean |feature| and the fraction that is EXACTLY zero.

    A dead modReLU emits exact zeros, so `dead_frac` is the diagnostic -- but note it has a
    FLOOR on this corpus that is not death: 27.5% of raw sample slots in the native pool are
    exactly 0.0 (duty-cycled bursts with noiseless off-time), any conv over an all-zero
    region emits exactly zero, and modReLU with b<0 keeps it zero. So ~0.17-0.23 at the
    shallow levels is capture silence, not dead units. The bottleneck reading 1.000 is far
    past that floor and is the real signal."""
    net.eval()
    z = torch.complex(xb[:, 0, :], xb[:, 1, :]).unsqueeze(1)
    v, rows = z, []
    for i, blk in enumerate(net.downs):
        v = blk(v)
        rows.append((f"down{i}", float(v.abs().mean()), float((v.abs() == 0).float().mean())))
        v = _lowpass121(v)[..., ::2] if net.antialias else v[..., ::2]
    v = net.bottleneck(v)
    rows.append(("bottleneck", float(v.abs().mean()), float((v.abs() == 0).float().mean())))
    return {"input_mean_abs": float(z.abs().mean()),
            "levels": [{"level": n, "mean_abs": m, "dead_frac": d} for n, m, d in rows],
            "bottleneck_mean_abs": rows[-1][1], "bottleneck_dead_frac": rows[-1][2]}


@torch.no_grad()
def bottleneck_is_load_bearing(net, xb, fb, clean, coherence_fn):
    """Does the DECODER use the bottleneck at all? Reconstruction coherence with the
    bottleneck zeroed, minus the same with it intact.

    On c3_unet_recon_only this delta was EXACTLY 0.0000 over 56 real rows -- the bottleneck
    contributed literally nothing to the output. That is why this is a standing per-trial
    metric and not a one-off: it is the kind of thing that regresses silently."""
    net.eval()
    ref = float(coherence_fn(net(xb, fb) is None or net.last_recon, clean).mean())
    v, skips = net.encode(xb)
    zeroed = torch.zeros_like(v)
    u = zeroed
    for blk, skip in zip(net.ups, reversed(skips)):
        u = net._up(u, skip.shape[-1])
        u = blk(torch.cat([u, skip], dim=1))
    rec0 = net.out(u).squeeze(1)
    if net.residual_recon:
        rec0 = torch.complex(xb[:, 0, :], xb[:, 1, :]) + rec0
    return {"coh_full": ref, "coh_bottleneck_zeroed": float(coherence_fn(rec0, clean).mean()),
            "delta_from_zeroing_bottleneck": float(coherence_fn(rec0, clean).mean()) - ref}


if __name__ == "__main__":
    torch.manual_seed(0)
    base = TransferUNet()
    ref_keys = set(base.state_dict().keys())
    try:
        from unet_multitask import ComplexUNetMultiTask
        old = ComplexUNetMultiTask()
        assert set(old.state_dict().keys()) == ref_keys, "default config must stay drop-in"
        base.load_state_dict(old.state_dict())
        print(f"[ok] default TransferUNet is state_dict-compatible with ComplexUNetMultiTask "
              f"({sum(p.numel() for p in base.parameters())} params)")
    except ImportError:
        print("[!] unet_multitask not importable; skipped compatibility check")

    xb = torch.randn(2, 2, 2048); fb = torch.randn(2, N_FEATURES)
    for name, kw in [("default", {}),
                     ("magnorm", dict(magnorm=True)),
                     ("modrelu-8", dict(modrelu_init=-8.0)),
                     ("skipdrop", dict(magnorm=True, skip_dropout=0.5)),
                     ("featdrop", dict(magnorm=True, feat_dropout=0.5)),
                     ("antialias", dict(magnorm=True, antialias=True)),
                     ("residual", dict(magnorm=True, residual_recon=True)),
                     ("acf", dict(magnorm=True, head_pool="acf")),
                     ("acf-complex", dict(magnorm=True, head_pool="acf_complex")),
                     ("chunk4", dict(magnorm=True, head_pool="chunk4"))]:
        net = TransferUNet(**kw).eval()
        e = net(xb, fb)
        # dim=-1 as a KEYWORD. Tensor.norm's first positional argument is p, not dim, so
        # `e.norm(-1)` silently computes a p=-1 norm -- the exact trap that made every
        # "|d emb|" figure in the audit's b_scale.py meaningless.
        assert e.shape == (2, EMBED_DIM)
        assert torch.allclose(e.norm(dim=-1), torch.ones(2), atol=1e-4), e.norm(dim=-1)
        assert net.last_recon.shape == (2, 2048)
        if kw.get("residual_recon"):
            zin = torch.complex(xb[:, 0, :], xb[:, 1, :])
            assert torch.equal(net.last_recon, zin), "zero-init residual must start as identity"
        h = encoder_health(net, xb, fb)
        # PHASE EQUIVARIANCE, checked on a LIVE bottleneck. At the default init the
        # bottleneck is 100% dead, so the embedding is invariant to everything and the check
        # proves nothing -- which is what the original version of this assertion did.
        phi = torch.tensor(0.77)
        zr = torch.complex(torch.cos(phi), torch.sin(phi))
        zc = torch.complex(xb[:, 0, :], xb[:, 1, :]) * zr
        with torch.no_grad():
            e1 = net(xb, fb); r1 = net.last_recon.clone()
            e2 = net(torch.stack([zc.real, zc.imag], 1), fb); r2 = net.last_recon
        rerr = float((r2 - zr * r1).abs().max() / (r1.abs().max() + 1e-12))
        eerr = float((e2 - e1).abs().max())
        assert rerr < 1e-4, (name, "reconstruction must be phase-equivariant", rerr)
        assert eerr < 1e-4, (name, "embedding must be phase-INVARIANT", eerr)
        live = 1.0 - h["bottleneck_dead_frac"]
        note = "  <- VACUOUS: bottleneck dead, heads read feat only" if live < 0.01 else ""
        print(f"[ok] {name:11s} params={sum(p.numel() for p in net.parameters()):7d} "
              f"bott mean|.|={h['bottleneck_mean_abs']:.3e} live={live:.3f} "
              f"recon_equiv={rerr:.1e} emb_inv={eerr:.1e}{note}")

    # the headline claim of fix 1, asserted rather than described
    d0 = encoder_health(TransferUNet().eval(), xb, fb)["bottleneck_dead_frac"]
    d1 = encoder_health(TransferUNet(magnorm=True).eval(), xb, fb)["bottleneck_dead_frac"]
    print(f"[ok] bottleneck dead fraction at init: default {d0:.3f} -> magnorm {d1:.3f}")
    assert d1 < 0.5 < d0, "magnorm must revive the bottleneck; if not, do not run the campaign"
