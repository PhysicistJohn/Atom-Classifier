# MLX Port Plan — v7 DACS Trainer (v1.1, post-adversarial-review)

**Status:** ACTIVE — A/B chain stopped at user direction 17:5x Jul 31; GPU free.
**Review:** v1.0 survived a 3-lens adversarial panel (fidelity / API / ops);
18 findings, 3 blockers — all incorporated below and marked ⟦R⟧.
**MLX:** 0.32.0 in `.venv-training` (python3.14), all load-bearing APIs
live-verified on this machine (including `as_strided`, axis-tuple
reductions, `stop_gradient`, `async_eval`, memory counters, MultiOptimizer
state save/restore, `load_weights(strict=True)`).
**Hardware:** Apple M5 Max, 48 GB unified; `max_recommended_working_set_size`
40.2 GB.
**Reference:** `v7_probe/v7_trainer.py` (torch-MPS), frozen as canonical
until G4. Trained checkpoints on disk: `v7_ab_recon_on.ep1000.pt`,
`.ep2000.pt` (bal 0.971/0.979/0.967 at ep 2000) — G1 uses ep-2000.

## 1. Objective and fidelity contract

Port v7 to MLX to structurally eliminate the two observed torch-MPS failure
classes (`copy_and_sync` deadlock 2/2 post-eval; cache eviction/thrash — the
final diagnosed bottleneck was per-episode synchronous Metal copies, which
MLX's unified-memory model does not have), with results scientifically
interchangeable with the torch trainer.

Compliant means:

- Same corpus/manifest/splits; no regeneration.
- **Bit-identical episode composition** on the primary numpy stream, exact
  call order per episode: dwell index → seven per-class `choice` → offsets
  (one call, size 70) → phase thetas (70,1) → one mask-threshold scalar →
  worst-dwell rows **only when `worst_dur_weight·min(1, ep/ramp_eps) > 0
  and ep % 4 == 0`** ⟦R: at ep 0 the ramp is zero ⇒ NO draw; replicate the
  exact torch guard, not the "every 4th" cadence⟧.
  `ramp_eps = max(1, int(episodes · worst_dur_ramp_frac))` computed from CLI
  values, never hardcoded (= 300 for the 3000-episode replications) ⟦R⟧.
- Same architecture/losses/detach points; same optimizer semantics (AdamW
  2e-3, decoupled decay 1e-4 on all net params, 0 on `log_scale`,
  betas (0.9,0.999), eps 1e-8, `bias_correction=True`; cosine to 0 over
  `episodes`, episode 0 at full lr).
- Same eval protocol/metrics/JSON schema (`v7-trainer-v1`), including:
  eval-time **prototypes read the fp16 train cache** (via the train_window
  path) while eval **query rows read full-precision complex64** — the torch
  asymmetry is exact and must be preserved on both halves ⟦R⟧.
- Checkpoint interop both directions until G5.

## 2. Deliverables

```
v7_probe/
  v7_trainer_mlx.py        # the port; CLI mirrors v7_trainer.py
  mlx_parity/
    convert.py             # torch .pt <-> safetensors, both directions
    test_unit_parity.py    # G0 harness (primitives + schedules + memory probe)
    test_forward_parity.py # G1: converted ep-2000 checkpoint
  mlx_smoke.sh             # G2 driver
```

## 3. Porting map (inventory → MLX; ⟦R⟧ = review-corrected)

| Torch | MLX | Notes |
|---|---|---|
| `unfold(-1,64,32)` | `[T,64]` index gather (`mx.take` / fancy index) or `mx.as_strided` — both live-verified exact | no pad/center; T = (N−64)//32+1 → 624/1561/6249 |
| `fft.fft(frames·hann)[..., :33]` | `mx.fft.fft` complex64, slice | **full complex FFT, not rfft**; periodic Hann fp32; norm backward |
| stack (re, im, log1p|·|) NCHW | same stack, NHWC `[B,T,33,3]` | converter owns all permutes |
| Conv/GN/SiLU blocks | `nn.Conv2d` + `nn.GroupNorm(8, C, pytorch_compatible=True)` + `nn.silu` | flag mandatory; eps 1e-5 |
| `F.interpolate(nearest, size)` | index-gather resize `floor(arange(out)·in/out)` | exact torch-nearest; G0 |
| pooling mean+amax over (T,F) | `mx.mean`/`mx.max` axis=(1,2) — verified | |
| `F.normalize(z)·4` | `z / mx.maximum(‖z‖, 1e-12) · 4` | |
| `cdist(q,p)²` | matmul expansion, clamp ≥ 0 | torch takes the mm path at 35×7 and 48×7 but the **naive kernel at the 14×7 aux** (mm engages only >25 rows) — difference is fp-level; G0 includes a 14×7 radius-4 case ⟦R⟧ |
| mask | second numpy stream `rng2.random((B,1,T,1))` → threshold → **reshape (B,T,1,1)** for NHWC broadcast ⟦R⟧ | keep = u > frac·scalar; zeroes whole time frames |
| worst-dwell aux | identical formulas; draw guard as §1; `stop_gradient` on protos and log_scale | |
| AdamW groups | `MultiOptimizer([AdamW(wd=0), AdamW(wd=1e-4)], [path == log_scale])`, `bias_correction=True` both | state flattens under `states.N.*`; **resume sequence: build → init(params) → assign state** (proven) ⟦R⟧ |
| `CosineAnnealingLR` | `optim.cosine_decay(lr, episodes, end=0.0)` as callable | realized-lr sequence asserted ≤1e-9 vs torch incl. after save/restore roundtrip (both sub-optimizers carry step counters ⟦R⟧) |
| checkpoints `.pt` | safetensors + sidecar JSON | sidecar: config, episode, **primary `bit_generator.state` + mask-stream state captured at "end of episode e, after update, before e+1's first draw"**, full MultiOptimizer state (both step counters), accumulated eval history; **atomic temp+rename writes** ⟦R⟧ |

## 4. MLX-native design (review-hardened)

**Memory — wire the corpus, not the world ⟦R: was the worst v1.0 error⟧.**
- Resident stores: train fp16 planes 9.8 GB + eval complex64 10.4 GB.
- `mx.set_wired_limit(≈22 GB)` — corpus + model + optimizer only.
  Activations stay ordinary allocations. Wiring 34 GB would have left the
  desktop ~8 GB; review activation math puts the T=6249/B=70 training step
  at ~14–19 GB peak, which must NOT be wired.
- `set_memory_limit` is a **throttle, not a tripwire** ⟦R⟧: the leak guard
  is an explicit watchdog — every 50 eps assert `get_active_memory` <
  (G0-measured worst-signature peak + 4 GB margin), else save checkpoint
  and abort loudly. Calibrated from measurement, not hand-picked — a fixed
  36 GB line would false-abort legitimate T=6249 steps. This is a
  leak detector, not a comfort ceiling: steady-state training is welcome to
  every byte the kernel can spare.
- `set_cache_limit(2 GB)` (buffers only recycle within a signature);
  `clear_cache()` on dwell-signature change and train↔eval transitions ⟦R⟧.
- Startup conversion: **preallocate destination, slice-assign chunks
  (≤512 MB), `mx.eval` per chunk, single reference, numpy chunk dropped per
  iteration; eval rows converted before train planes; post-conversion
  assertion `peak − active < 1 GB`** ⟦R: naive concatenate doubles
  residency⟧. `madvise` purge after conversion; memmaps closed.
- Contingency if the G0 memory probe shows T=6249/B=70 over ~26 GB
  standalone: split the long-dwell batch 2×35 with exact gradient
  accumulation (GroupNorm is per-sample; summing per-row losses with
  identical weights is mathematically exact) ⟦R⟧.

**Compile — six signatures, clean warmup ⟦R⟧.**
- Aux episodes have a structurally different graph: compile **aux-inclusive
  and aux-free step variants** → exactly **6** signatures (3 dwells ×
  {aux, no-aux}); preserves both the numpy stream and the ×4-every-4th
  amortization. Startup assertion allows exactly 6.
- **Warmup runs on a deep-copied throwaway state** (cloned params + fresh
  optimizer); afterwards assert the real params' hash and both schedule
  step counters are bit-identical to the loaded checkpoint before episode 0
  ⟦R: dummy steps on live state would corrupt init + schedule⟧.
- Positional and kwargs compile forms both work in 0.32.0 (probe); we use
  positional by convention ⟦R: v1.0's "kwargs rejected" claim was false⟧.
- Per-episode-varying scalars as 0-d `mx.array` args (verified). One
  `mx.eval(state)` per episode; `mx.async_eval` then numpy assembly of the
  next episode overlaps CPU/GPU. Eval forward compiled separately; final
  chunk padded to 48, padding masked from metrics.
- Per-block memory probe in G0 at the T=6249 signature: any conv whose
  peak-delta exceeds ~2× its output size has fallen off the implicit-GEMM
  path onto im2col (a ~25 GB single-op temporary at full res) — apply the
  zero-channel-pad 3→4 trick (equivalence-tested) **for memory**, not just
  speed ⟦R⟧.

**Precision policy (user directive):** MLX 0.32.0 defaults M5 fp32 matmuls
to TF32 (G0 measured: 2.7e-2 abs vs fp64 truth on a head-sized matmul, vs
4.6e-6 with `MLX_ENABLE_TF32=0`). Production trainer runs **TF32 ON by
default** — the user wants the performance; industry practice agrees.
`MLX_ENABLE_TF32=0` is pinned ONLY inside `mlx_parity/` harnesses, where
fp32-exact comparison is what makes the gates falsifiable. The trainer
takes `--no-tf32` for bisection. G2 reports a three-config pace matrix —
fp32-strict / TF32 / **bf16 compute** (`--dtype bf16`: model compute in
bfloat16, losses reduced in fp32) — with per-config peak memory; on
unified memory the bf16 win is bandwidth (activations at T=6249/B=70 are
the dominant traffic) as much as FLOPs. G3/G4 validate in the TF32
configuration; **G4b** then reruns the G4 protocol in bf16 against the
MLX-fp32 result with the same tolerances (±0.005 bal, ±0.02 min-cell) —
pass makes bf16 the production-study default, fail leaves TF32 with a
measured reason. bf16 is chosen over fp16 deliberately: fp32-range
exponent, no loss scaling (MLX has no GradScaler; fp16 would need a
hand-rolled one).

## 5. Gates (all torch baselines re-measured, not quoted from memory)

- **G0 — unit parity + memory probe.** Primitives (framing, Hann, FFT+
  truncate, stack, rms_normalize, nearest resize, GroupNorm-compat, SiLU,
  pooling, normalize·4, cdist² incl. 14×7, temperature clamp, losses,
  mask reshape semantics ⟦R⟧), AdamW single-step (bias correction + decay
  coupling + exclusion), full realized-lr sequence ≤1e-9 incl. save/restore
  roundtrip ⟦R⟧, per-signature and per-block `get_peak_memory` probe ⟦R⟧.
  atol 1e-5 fp32 (schedule 1e-9). GPU is free now — run both devices.
- **G1 — forward parity.** Convert `v7_ab_recon_on.ep2000.pt`; frozen real
  batches all 3 dwells: logits/conf/recon atol 1e-4; argmax agreement
  ≥99.9%; confusion delta ≤2/cell vs `eval_confusion.py`; **sub-check that
  eval prototypes are built from the fp16-cache path** (build both ways,
  assert they differ and the fp16 one matches torch) ⟦R⟧; roundtrip
  torch→mlx→torch bitwise.
- **G2 — 300-ep smoke from torch step-0 init.** Identical composition
  stream verified by **dumping the first-N episodes' (dwell, picks,
  offsets, thetas, threshold, wd_rows) tuples from both trainers and
  asserting exact equality** ⟦R: this catches the ep-0 guard class⟧.
  Loss-envelope agreement; ep-300 eval ±0.02; exactly 6 signatures; flat
  per-signature peak memory with numeric bounds ⟦R⟧; pace ≤ a **freshly
  measured** torch baseline on the idle machine (v1.0's "1.4 ms/ep" was a
  unit error ⟦R⟧; torch measured 0.6–1.8 s/ep depending on machine state).
- **G2b — resume parity ⟦R: new⟧.** 600 eps straight vs 500 + resume + 100:
  bit-identical episode composition, loss equal to fp tolerance, next-step
  lr equal to 1e-9.
- **G3 — probe-v3 attribution reproduction.** ±0.01 per cell vs
  0.954/0.976/1.000 (s1), 0.958/0.972/1.000 (s2).
- **G4 — full 3000-ep recon_on replication.** Final full-eval bal ±0.005,
  min-cell ±0.02, escalation ±0.005; wall-clock per dwell reported. Pass ⇒
  MLX is the production-study trainer; torch archived.
- **G5 — tooling.** `eval_confusion` (port or via convert.py), Optuna
  harness, one budget-control sanity trial.

Rollback at any gate: torch + mlock/hygiene (known-working), file findings
upstream, retry.

## 6. Sequencing (GPU free as of 17:5x)

1. `convert.py` + G0 now (GPU available — both-device runs fine).
2. G1 against ep-2000 checkpoint immediately after G0.
3. Trainer port → G2/G2b (hours), G3 overnight, G4 overnight.
4. G5 + paper-facing notes after G4.

## 7. Contribution radar (user standing request)

- **PyTorch MPS**: `copy_and_sync` post-eval deadlock — 2/2 stack samples +
  the `synchronize`+`empty_cache` mitigation; minimal reproducer to file.
  Also the per-episode synchronous-copy degradation under swap pressure.
- **MLX**: TF32-by-default on M5 — not a bug report (it's a defensible
  NVIDIA-style choice) but measured error data (2.7e-2 vs 4.6e-6 on
  head-sized matmuls), a docs/discoverability case, and possibly a
  programmatic toggle request (env-var-only today). Plus: per-block
  memory-probe results for tall-skinny NHWC convs (im2col fallback evidence
  if found); framing/STFT helper gap; anything complex64 we hit. RF/DSP
  exercises corners the LLM crowd doesn't.

## 8. Differences ledger (final; ⟦R⟧ items added by review)

1. **Bit-level nonidentity** (FFT kernels, reduction order, fused ops):
   logits ~1e-5; from-scratch trajectories diverge chaotically — gates
   compare envelopes and final metrics, never per-step losses.
2. **Mask stream**: torch device-RNG per-frame uniforms → dedicated numpy
   generator, reshaped (B,T,1,1) for NHWC ⟦R⟧. Same law, different
   instances; strictly better determinism (torch's masks weren't even
   CPU/MPS portable).
3. **Init provenance**: pre-G4 runs use torch-exported step-0 checkpoints;
   native MLX init only after G4 and flagged.
4. **AdamW bias correction** explicitly enabled (MLX default off); decay
   1e-4 overrides MLX's 0.01 default; coupling verified identical.
5. **Cosine schedule mechanics** differ; realized lr sequence asserted
   identical ≤1e-9, including across save/restore ⟦R⟧.
6. **Resume semantics differ twice** ⟦R: second half was missing⟧:
   (a) optimizer moments + schedule steps are restored (torch restarted
   moments — a flaw); (b) **episode stream**: torch resume jumps to
   `default_rng([seed, start_ep])`; MLX resume continues the exact saved
   stream — so a resumed MLX run matches an unbroken run (torch's does
   not). `--torch-compat-resume` flag available for debugging parity
   against torch resumed runs.
7. **Checkpoint format**: safetensors + sidecar (atomic writes ⟦R⟧);
   `convert.py` keeps `.pt` interop; pickle loading disappears.
8. **Layouts** NHWC internally; public artifacts unchanged.
9. **Precision asymmetry preserved exactly** ⟦R: v1.0 overstated⟧:
   training windows AND eval-time prototypes read the fp16 cache; eval
   query rows read full complex64 — both halves now RAM-resident instead
   of memmap-streamed. Same numbers in, no disk in the loop.
10. **Operational code dropped**: MPS hygiene, mlock, post-eval madvise
    (madvise survives only in one-shot conversion); `set_wired_limit(~22 GB)`
    supersedes; stall-monitor wedge branch vestigial.
11. **Compile granularity is visible**: 6 signatures (aux/no-aux × 3
    dwells) vs torch's eager execution ⟦R⟧ — no semantic effect, but a
    startup invariant the torch trainer doesn't have.
12. **`train()`/`eval()` are conventions** (no dropout/BN).
13. **Not ported**: corpus generators, stage-2 impairments, SignalLab,
    existing paper numbers (torch-derived until G4 documents otherwise).
15. **Production numerics use TF32 matmuls** (10-bit mantissa) where torch-
    MPS used full fp32 — validated at the gate level (G3/G4 final metrics),
    deliberately not at the step level; `--no-tf32` bisects if a gate
    misses. Parity harnesses always run fp32-exact.
14. **A/B closure** ⟦R context⟧: the torch A/B was stopped at user
    direction with recon_on at ep 2000 (bal 0.971/0.979/0.967); recon_off
    never ran. The recon-on/off question transfers to the MLX trainer
    post-G4 if still wanted for the paper.
