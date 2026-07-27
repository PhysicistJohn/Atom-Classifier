"""Multi-scale 1D architecture, natively complex where phase is physically
meaningful, per the research pull on Automatic Modulation Classification
(AMC): (1) a genuinely complex-valued convolutional branch over raw I/Q,
where phase is preserved by construction (complex conv + a phase-equivariant
nonlinearity), run at several dilations IN PARALLEL so both fine chip/symbol-
rate structure and coarse burst-duration structure are visible in one pass
(the field's standard "multi-timescale" trick, TCN/Inception-style, applied
here via width not depth); (2) two REAL-valued multi-scale branches over
instantaneous amplitude (envelope) and instantaneous frequency -- both are
phase-invariant quantities BY DEFINITION (envelope = |z|, freq = d/dt of
unwrapped phase), so there is no phase to preserve there and ordinary real
convolutions are the right tool, not an approximation. This mirrors the
literature's "tri-branch fusion" pattern (raw I/Q + envelope + inst-freq,
fused late) called out as well-established for bursty/intermittent signals.

WHY complex, not 2-channel-real, for the I/Q branch specifically: a real
Conv1d over stacked (I, Q) channels can only ever LEARN an approximation to
phase-consistent operations from data; a true complex conv (weight = Wr+iWi,
applied via the standard 4-real-convolution identity) is phase-equivariant by
construction for any linear combination, and modReLU (ported from
zplane_backbone.py earlier this session) keeps that equivariance through the
nonlinearity -- so a global phase rotation of the input produces the exact
same rotation of the complex branch's internal features, never a different
one. This is the same math already validated in zplane_backbone.py; reused
here as a compact multi-scale block rather than the operator's global
spectral-kernel form, since here it's classification, not equalization.

Same forward(x, feat) contract as every other backbone in this A/B
(model.Embedding, ZPlaneEmbedding, ViTEmbedding, ViT2DEmbedding) -- same
fc1/fc2/normalize tail, so this plugs into the identical harness.
"""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_TRAINING_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _TRAINING_DIR not in sys.path:
    sys.path.insert(0, _TRAINING_DIR)

from model import EMBED_DIM, N_FEATURES  # noqa: E402
from vit_backbone import HIDDEN, DROPOUT  # noqa: E402 -- reused constants, not duplicated

DEFAULT_DILATIONS = (1, 2, 4, 8)


# ---------------------------------------------------------------------------
# complex branch (raw I/Q) -- phase preserved by construction
# ---------------------------------------------------------------------------
class ComplexConv1d(nn.Module):
    """Complex conv with checkpoint-compatible fused and legacy execution paths.

    Both paths implement the standard identity::

        (Wr+iWi)*(Xr+iXi) = (Wr*Xr - Wi*Xi) + i(Wr*Xi + Wi*Xr)

    ``fused_grouped`` performs the four real products in one ``groups=2`` Conv1d call.
    Group 0 consumes ``Xr`` and emits ``Wr*Xr, Wi*Xr``; group 1 consumes ``Xi`` and emits
    ``Wr*Xi, Wi*Xi``.  Combining those four outputs preserves the legacy reduction order
    (and therefore forward values), while removing three operator launches.

    A superficially simpler block matrix ``[[Wr,-Wi],[Wi,Wr]]`` also uses one Conv1d, but
    sums over twice as many input channels in one reduction.  On the deployed layer shapes
    that changed float32 results by up to ~2e-6, outside the 1e-6 parity contract.  The
    grouped formulation avoids that numerical change.

    ``conv_re`` and ``conv_im`` remain the only parameter-bearing submodules, with their
    original names and shapes.  Old checkpoints therefore load strictly in every mode.

    ``auto`` selects the measured winner for the current execution context: fused on CPU,
    fused for MPS training (autograd enabled), legacy for MPS no-grad inference, and legacy
    on unbenchmarked device types. Set ``execution="legacy"`` for an unconditional safe
    four-call fallback. The fused functional call does not fire forward hooks registered
    directly on the child ``conv_re``/``conv_im`` modules; use that fallback if external
    instrumentation depends on such hooks. The AtomOS training paths register none.
    """

    EXECUTIONS = ("auto", "fused_grouped", "legacy")

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1,
                 execution: str = "auto"):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2  # 'same' padding, kernel_size assumed odd
        self.conv_re = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=pad, bias=False)
        self.conv_im = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=pad, bias=False)
        self.set_execution(execution)

    def set_execution(self, execution: str) -> None:
        if execution not in self.EXECUTIONS:
            raise ValueError(
                f"unknown complex-conv execution {execution!r}; "
                f"expected one of {self.EXECUTIONS}"
            )
        # Plain metadata by design: it must not enter state_dict or alter checkpoint keys.
        self.execution = execution

    def _forward_legacy(self, z: torch.Tensor) -> torch.Tensor:
        re, im = z.real, z.imag
        out_re = self.conv_re(re) - self.conv_im(im)
        out_im = self.conv_re(im) + self.conv_im(re)
        return torch.complex(out_re, out_im)

    def _forward_fused_grouped(self, z: torch.Tensor) -> torch.Tensor:
        re, im = z.real, z.imag
        paired_input = torch.cat((re, im), dim=1)
        wr, wi = self.conv_re.weight, self.conv_im.weight

        # groups=2 partitions the input as [real channels | imaginary channels].
        # Within each partition the two output banks retain the exact legacy convolution
        # reductions. Parameter views are repeated, not copied into new Parameters, so
        # autograd accumulates both uses back into the original Wr/Wi tensors.
        paired_weight = torch.cat((wr, wi, wr, wi), dim=0)
        products = F.conv1d(
            paired_input,
            paired_weight,
            bias=None,
            stride=self.conv_re.stride,
            padding=self.conv_re.padding,
            dilation=self.conv_re.dilation,
            groups=2,
        )
        wr_re, wi_re, wr_im, wi_im = products.split(
            self.conv_re.out_channels, dim=1
        )
        return torch.complex(wr_re - wi_im, wr_im + wi_re)

    def _resolved_execution(self, z: torch.Tensor) -> str:
        if self.execution != "auto":
            return self.execution
        # MPS's grouped convolution is slightly slower for no-grad inference, but its
        # combined backward removes enough launches to nearly halve encoder training-step
        # time. CPU wins in both modes. Unknown backends stay on the conservative path until
        # they have their own benchmark.
        if z.device.type == "cpu" or (
            z.device.type == "mps" and torch.is_grad_enabled()
        ):
            return "fused_grouped"
        return "legacy"

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self._resolved_execution(z) == "legacy":
            return self._forward_legacy(z)
        return self._forward_fused_grouped(z)


def set_complex_conv_execution(module: nn.Module, execution: str) -> int:
    """Select one execution path on every :class:`ComplexConv1d` below ``module``.

    Returns the number of converted layers.  This is intentionally runtime metadata rather
    than persistent model state, so switching paths cannot invalidate a checkpoint.
    """
    if execution not in ComplexConv1d.EXECUTIONS:
        raise ValueError(
            f"unknown complex-conv execution {execution!r}; "
            f"expected one of {ComplexConv1d.EXECUTIONS}"
        )
    count = 0
    for child in module.modules():
        if isinstance(child, ComplexConv1d):
            child.set_execution(execution)
            count += 1
    return count


class ComplexModReLU(nn.Module):
    """Phase-equivariant nonlinearity, ported from zplane_backbone.py: gates
    on magnitude only, re-applies the input's own phase -- modReLU(e^{i phi}
    z, b) == e^{i phi} modReLU(z, b) exactly, unlike independent ReLU(re)/
    ReLU(im) which does not commute with rotation."""

    def __init__(self, channels: int):
        super().__init__()
        self.thresh_raw = nn.Parameter(torch.full((channels,), -2.0))  # b = -softplus(raw) <= 0

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = -F.softplus(self.thresh_raw).view(1, -1, 1)
        mag = torch.sqrt(z.real ** 2 + z.imag ** 2 + 1e-12)
        scale = torch.clamp(mag + b, min=0.0)
        return (scale / mag) * z


class ComplexMultiScaleBlock(nn.Module):
    """Parallel dilated complex convs (different timescales at the SAME
    depth, not stacked -- so no single branch has to learn to aggregate
    across scales) -> concat -> complex 1x1 mix -> modReLU -> residual."""

    def __init__(self, channels: int, kernel_size: int = 5, dilations=DEFAULT_DILATIONS):
        super().__init__()
        n = len(dilations)
        branch_ch = max(1, channels // n)
        self.branches = nn.ModuleList([
            ComplexConv1d(channels, branch_ch, kernel_size, dilation=d) for d in dilations
        ])
        self.mix = ComplexConv1d(branch_ch * n, channels, kernel_size=1)
        self.act = ComplexModReLU(channels)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        outs = [b(z) for b in self.branches]
        cat = torch.cat(outs, dim=1)
        mixed = self.mix(cat)
        return self.act(mixed + z)


class ComplexMultiScaleBranch(nn.Module):
    def __init__(self, width: int = 32, kernel_size: int = 5, dilations=DEFAULT_DILATIONS,
                 n_blocks: int = 3, in_channels: int = 1):
        super().__init__()
        self.lift = ComplexConv1d(in_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList([
            ComplexMultiScaleBlock(width, kernel_size, dilations) for _ in range(n_blocks)
        ])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        v = self.lift(z)
        for blk in self.blocks:
            v = blk(v)
        return v  # (B, width, L) complex


# ---------------------------------------------------------------------------
# real branches (envelope / instantaneous frequency) -- already phase-
# invariant quantities, ordinary real convs are the correct tool, not a
# simplification
# ---------------------------------------------------------------------------
class RealMultiScaleBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dilations=DEFAULT_DILATIONS):
        super().__init__()
        n = len(dilations)
        branch_ch = max(1, channels // n)
        self.branches = nn.ModuleList([
            nn.Conv1d(channels, branch_ch, kernel_size, dilation=d,
                      padding=d * (kernel_size - 1) // 2) for d in dilations
        ])
        self.mix = nn.Conv1d(branch_ch * n, channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [b(x) for b in self.branches]
        cat = torch.cat(outs, dim=1)
        mixed = self.mix(cat)
        return self.act(mixed + x)


class RealMultiScaleBranch(nn.Module):
    def __init__(self, in_ch: int = 1, width: int = 16, kernel_size: int = 5,
                 dilations=DEFAULT_DILATIONS, n_blocks: int = 3):
        super().__init__()
        self.lift = nn.Conv1d(in_ch, width, kernel_size=1)
        self.blocks = nn.ModuleList([
            RealMultiScaleBlock(width, kernel_size, dilations) for _ in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.lift(x)
        for blk in self.blocks:
            v = blk(v)
        return v  # (B, width, L) real


class TemporalMemoryPool(nn.Module):
    """CLDNN-style temporal aggregation: replaces mean/std pooling with a
    bidirectional LSTM over the (downsampled) time axis, so the network can
    track sequential state -- e.g. "a burst just ended, is this on-time for a
    periodic duty cycle" -- rather than only ever seeing time-collapsed
    statistics. Downsamples first (strided average-pool) since LSTMs don't
    parallelize over time the way convs do, and raw length (1024 or 4096) is
    unnecessarily fine-grained for the burst/envelope-scale patterns this
    stage targets -- this mirrors how CLDNN implementations in the literature
    feed pooled/strided CNN features into the LSTM, not the raw-rate sequence.
    Takes the final hidden state from both directions (not all timesteps) as
    the branch's pooled representation."""

    def __init__(self, in_channels: int, hidden: int = 24, downsample: int = 8,
                 rnn_type: str = "lstm", bidirectional: bool = True, use_rnn: bool = True):
        super().__init__()
        self.downsample = downsample
        self.use_rnn = use_rnn
        self.bidirectional = bidirectional
        self.pool = nn.AvgPool1d(kernel_size=downsample, stride=downsample, ceil_mode=True)
        if use_rnn:
            rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
            self.rnn = rnn_cls(input_size=in_channels, hidden_size=hidden,
                                batch_first=True, bidirectional=bidirectional)
            self.out_dim = (2 if bidirectional else 1) * hidden
        else:
            # ablation: skip the RNN entirely, fall back to mean/std pooling
            # over the (still downsampled) sequence -- isolates how much the
            # recurrent stage is actually contributing vs the conv branches alone
            self.rnn = None
            self.out_dim = 2 * in_channels

    def _apply(self, fn, recurse=True):
        """Pin self.rnn to CPU no matter what device the rest of the model is
        moved to. nn.Module.to()/cuda()/etc all recurse via _apply (calling
        each submodule's _apply directly, NOT their .to() method) -- so this,
        not an override of .to(), is the actual hook that runs on every
        net.to(dev) call in the shared training harness. Runs the normal apply
        first (so dtype changes etc still take effect), then forces the rnn
        submodule back to CPU via its own .to() (which correctly refreshes its
        internal flat_weights cache, unlike hand-rolling the tensor move).

        NOTE on why this is still here: earlier in this session a training run
        appeared wedged (near-zero CPU, no progress) and was suspected to be a
        bidirectional-LSTM-on-MPS hang. Direct isolated repros (plain LSTM
        looped, complex-derived-input LSTM looped, and the full 3-branch
        combination looped, all on pure MPS) all ran cleanly -- the original
        run was very likely just slow (each step costs ~2s here), not hung.
        So this is NOT a confirmed MPS bug. CPU-pinning is kept anyway as a
        conservative default for long unsupervised runs (a multi-day
        autonomous campaign is the wrong place to spend the first confirmation
        of a rare edge case), not because of a demonstrated defect."""
        super()._apply(fn, recurse=recurse)
        if self.rnn is not None:
            self.rnn = self.rnn.to("cpu")
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, L) real -> (B, out_dim) real."""
        orig_device = x.device
        x = self.pool(x)                    # (B, C, L//downsample) -- on x's original device
        if not self.use_rnn:
            return torch.cat([x.mean(-1), x.std(-1)], dim=-1)  # ablation path, stays on orig_device
        x = x.transpose(1, 2).to("cpu")      # (B, T, C) -- RNN's expected layout, forced to CPU
        if isinstance(self.rnn, nn.LSTM):
            _, (h_n, _) = self.rnn(x)        # h_n: (n_dir, B, hidden) -- final states
        else:
            _, h_n = self.rnn(x)             # GRU: h_n: (n_dir, B, hidden), no cell state
        out = torch.cat([h_n[0], h_n[1]], dim=-1) if self.bidirectional else h_n[0]
        return out.to(orig_device)


def _instantaneous_frequency(z: torch.Tensor) -> torch.Tensor:
    """z: (B, L) complex -> (B, L) real, diff-of-unwrapped-phase convention
    (matches preprocess.py's iq_features: np.diff(np.unwrap(np.angle(z)))),
    front-padded to keep the same length. Unwrap via wrap-to-[-pi,pi] on the
    raw phase difference -- equivalent to unwrap-then-diff without needing a
    running cumulative-unwrap pass."""
    phase = torch.angle(z)
    dphase = phase[:, 1:] - phase[:, :-1]
    dphase = torch.remainder(dphase + math.pi, 2 * math.pi) - math.pi
    return F.pad(dphase, (1, 0))


# ---------------------------------------------------------------------------
# top-level embedding
# ---------------------------------------------------------------------------
class ComplexMultiScaleEmbedding(nn.Module):
    """Drop-in replacement for model.Embedding / ZPlaneEmbedding / ViTEmbedding
    / ViT2DEmbedding: same forward(x, feat) contract, same fc1/fc2/normalize
    tail."""

    def __init__(self, complex_width: int = 32, real_width: int = 16, kernel_size: int = 5,
                 dilations=DEFAULT_DILATIONS, n_blocks: int = 3, lstm_hidden: int = 24,
                 lstm_downsample: int = 8, rnn_type: str = "lstm", rnn_bidirectional: bool = True,
                 use_rnn: bool = True, embed_dim: int = EMBED_DIM, n_features: int = N_FEATURES):
        super().__init__()
        self.complex_branch = ComplexMultiScaleBranch(width=complex_width, kernel_size=kernel_size,
                                                        dilations=dilations, n_blocks=n_blocks)
        self.envelope_branch = RealMultiScaleBranch(in_ch=1, width=real_width, kernel_size=kernel_size,
                                                      dilations=dilations, n_blocks=n_blocks)
        self.instfreq_branch = RealMultiScaleBranch(in_ch=1, width=real_width, kernel_size=kernel_size,
                                                      dilations=dilations, n_blocks=n_blocks)
        # CLDNN-style temporal memory: RNN over each branch's (downsampled)
        # feature sequence, replacing mean/std pooling -- the complex branch's
        # features go in as real+imag concatenated (2x channels), since the
        # RNN's job here is sequential memory, not rotational equivariance
        # (that was already the complex branch's job, upstream of this).
        # rnn_type/rnn_bidirectional/use_rnn are ablation knobs for the
        # architecture-search campaign -- defaults reproduce the original design.
        pool_kwargs = dict(hidden=lstm_hidden, downsample=lstm_downsample, rnn_type=rnn_type,
                            bidirectional=rnn_bidirectional, use_rnn=use_rnn)
        self.complex_pool = TemporalMemoryPool(2 * complex_width, **pool_kwargs)
        self.envelope_pool = TemporalMemoryPool(real_width, **pool_kwargs)
        self.instfreq_pool = TemporalMemoryPool(real_width, **pool_kwargs)
        pooled_dim = self.complex_pool.out_dim + self.envelope_pool.out_dim + self.instfreq_pool.out_dim
        self.fc1 = nn.Linear(pooled_dim + n_features, HIDDEN)
        self.drop = nn.Dropout(DROPOUT)
        self.fc2 = nn.Linear(HIDDEN, embed_dim)

    def forward(self, x: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """x: float32 (B,2,L), feat: float32 (B,n_features) -- SAME signature
        as every other backbone here."""
        z = torch.complex(x[:, 0, :], x[:, 1, :])            # (B, L) complex
        env = z.abs().unsqueeze(1)                             # (B, 1, L) real -- phase-invariant by definition
        instfreq = _instantaneous_frequency(z).unsqueeze(1)     # (B, 1, L) real -- phase-invariant by definition

        cv = self.complex_branch(z.unsqueeze(1))                # (B, Wc, L) complex -- phase preserved throughout
        cv_real = torch.cat([cv.real, cv.imag], dim=1)          # (B, 2*Wc, L) -- hand off to LSTM as real from here
        cv_pooled = self.complex_pool(cv_real)                  # (B, 2*lstm_hidden)

        ev = self.envelope_branch(env)                          # (B, Wr, L) real
        ev_pooled = self.envelope_pool(ev)                      # (B, 2*lstm_hidden)

        fv = self.instfreq_branch(instfreq)                     # (B, Wr, L) real
        fv_pooled = self.instfreq_pool(fv)                      # (B, 2*lstm_hidden)

        pooled = torch.cat([cv_pooled, ev_pooled, fv_pooled], dim=-1)
        h = torch.cat([pooled, feat], dim=-1)
        h = F.relu(self.fc1(h))
        h = self.drop(h)
        h = self.fc2(h)
        return F.normalize(h, dim=-1)


if __name__ == "__main__":
    torch.manual_seed(0)
    for in_len in (1024, 4096):
        net = ComplexMultiScaleEmbedding()
        n_params = sum(p.numel() for p in net.parameters())
        xb = torch.randn(5, 2, in_len)
        fb = torch.randn(5, N_FEATURES)
        out = net(xb, fb)
        assert out.shape == (5, EMBED_DIM)
        assert torch.allclose(out.norm(dim=-1), torch.ones(5), atol=1e-4)
        print(f"[ok] in_len={in_len}: param_count={n_params}, forward shape {tuple(out.shape)}, unit-norm ok")

    # phase-equivariance check on the complex branch specifically: rotating the
    # input's global phase should rotate cv_mag NOT AT ALL (magnitude is phase-
    # invariant by construction) -- the real check is that the COMPLEX features
    # before magnitude are exactly phase-rotated, which is what modReLU + linear
    # complex convs guarantee. Verify that end-to-end.
    torch.manual_seed(0)
    branch = ComplexMultiScaleBranch(width=8, n_blocks=2)
    z = torch.randn(3, 1, 256, dtype=torch.cfloat)
    phi = torch.tensor(0.83)
    rot = torch.complex(torch.cos(phi), torch.sin(phi))
    out1 = branch(z)
    out2 = branch(rot * z)
    lhs = rot * out1
    assert torch.allclose(lhs, out2, atol=1e-4), "complex branch should be phase-equivariant end-to-end"
    print("[ok] complex multi-scale branch is phase-equivariant end-to-end (rotating input rotates output identically)")
