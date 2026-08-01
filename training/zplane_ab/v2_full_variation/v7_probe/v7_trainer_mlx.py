"""v7 production trainer -- MLX port.

Port of ``v7_trainer.py`` (torch-MPS, frozen reference) per the BINDING plan
``docs/mlx-port-plan.md`` (v1.1, post-adversarial-review).  Fidelity contract:

* Bit-identical episode composition on the primary numpy stream; exact
  per-episode call order: dwell idx -> seven per-class choice -> offsets
  (one call, size 70) -> thetas (70,1) -> ONE mask-threshold scalar (only
  when mask_frac > 0, mirroring the torch guard) -> worst-dwell rows drawn
  ONLY IF worst_dur_weight*min(1, ep/ramp_eps) > 0 AND ep % 4 == 0 (at ep 0
  the ramp is 0 => NO draw).
* Mask uniforms come from a SECOND generator rng2 = default_rng([seed,
  MASK_STREAM_KEY]) drawn as (B,1,T,1) then reshaped (B,T,1,1) for the NHWC
  broadcast (plan section 3; differences ledger item 2).
* Losses/detach points exactly as torch; MultiOptimizer AdamW with cosine
  decay, wd=1e-4 on all net params, wd=0 on log_scale, bias_correction=True.
* Compile: exactly SIX training-step signatures (3 dwells x {aux, no-aux}),
  all warmed at startup on throwaway state (deep-copied params + optimizer
  state, restored in place afterwards; real params asserted hash-identical
  and step counters unchanged before episode 0).  A 7th signature aborts.
* Memory: corpus resident as mx arrays (train fp16 I/Q planes, eval
  complex64), converted chunked (<=512 MB, preallocate + slice-assign +
  mx.eval per chunk, eval rows FIRST); memmaps madvised away.  Wired limit
  22 GB, cache limit 2 GB, clear_cache on dwell-signature change and
  train<->eval transitions.  Watchdog every 50 eps.
* Eval protocol identical to torch evaluate() including the precision
  asymmetry: prototypes read the fp16 train cache, query rows read
  complex64.  Result JSON schema v7-trainer-v1.  The protocol itself
  (mid-run subsample + per-row window offsets for queries AND prototypes)
  is frozen once at startup by build_eval_plan() on a dedicated stream
  default_rng([eval_seed, EVAL_STREAM_KEY]) -- see that function for the
  aliasing it fixes and --eval-offset-mode/--eval-subsample-mode/
  --eval-offset-seed for the knobs.  Training draws are untouched.
* Checkpoints: safetensors (params + full MultiOptimizer state) + sidecar
  JSON (config, episode, both RNG bit_generator states captured
  end-of-episode after the update and before the next episode's draws,
  eval history); atomic temp+rename.  Native resume continues the exact
  saved stream; --torch-compat-resume reproduces torch's stream jump.
* Precision (user directives, plan section 4): TF32 matmuls are ON by
  default (the MLX M5 default; this trainer no longer touches
  MLX_ENABLE_TF32).  --no-tf32 pins MLX_ENABLE_TF32=0 BEFORE the mlx
  import via a sys.argv peek at module top (argparse runs far too late;
  the peek is the plan's os.execv alternative without the re-exec) and is
  asserted effective by a startup matmul-vs-fp64 probe.  An externally
  exported MLX_ENABLE_TF32 is respected and reported, never overridden.
  --dtype {fp32,bf16,fp16} (default fp32) sets the net's compute dtype via
  nn.Module.set_dtype (params + activations); the spectrogram /
  rms_normalize front end stays fp32 and is cast at model input; ALL
  losses are computed and reduced in fp32 (z / conf_logit / recon are cast
  up before any loss math; log_scale lives outside .net and stays fp32).
  Optimizer under reduced dtype -- what MLX does natively: AdamW
  init_single uses zeros_like(parameter) so BOTH moments live in the
  parameter dtype, and apply_single casts the learning rate and
  bias-correction factors to the gradient dtype; with --dtype bf16 the
  entire update (moments, bias correction, decoupled decay) therefore
  runs in bfloat16 -- there are no fp32 master weights.  fp16 has no
  GradScaler in MLX (accepted risk; prefer bf16 per plan G4b).
  A non-finite TOTAL loss skips that episode's optimizer update entirely
  (params, moments and step counters restored from pre-step references;
  the episode's RNG draws stand, so the composition stream is unaffected)
  and is counted in the result JSON as "skipped_steps".
* Memory (the T=6249/B=70 fix; measured story in probe logs): the naive
  compiled step peaked 58.8 GB because MLX encodes the whole backward
  ahead of execution, so the conv-vjp explicit-GEMM temporaries (grad
  unfolds; ~10-16 GB EACH for stem/d1/d2/u3 at B=70) co-allocate -- their
  SUM, not their max.  Encoder-block checkpointing alone therefore
  plateaus at ~44 GB.  Three composing mechanisms fix it, all inside the
  single compiled step (the 6-signature invariant is untouched):
  (1) --ckpt-blocks N: checkpoint the first N of stem,d1..d5 (default -1
      = auto: 3 at the 200000-sample dwell signatures, 0 for short
      dwells; explicit N applies to all dwells).  The decoder UpBlocks +
      out conv are checkpointed under the same per-signature condition --
      a measured necessity beyond the encoder-only directive: u3's
      grad-weight unfold alone is ~12.5 GB.
  (2) --batch-chunks N (default -1 = auto: 5 at the 200000-sample dwell,
      1 for short dwells): the plan section 4 contingency, generalized --
      equal checkpointed encoder/decoder units per chunk with exact
      gradient accumulation (per-sample GroupNorm; protos/CE see all 70
      z rows; equal-chunk l1 means average exactly).
  (3) --mem-limit-gb G (default 22, anchored to post-corpus active):
      mx.set_memory_limit as backpressure so the scheduler serializes the
      checkpoint units instead of co-allocating them (throttle, not
      tripwire, per plan).  Measured: no pace cost with (1)+(2); without
      them the throttle thrashes.
  --probe-memory prints per-block and full-train-step peak-memory deltas
  at T=6249 B=70 and exits.  --force-dwell {1ms,2.5ms,10ms} is a
  pace-diagnostic override (the dwell integer is still drawn so the
  stream call order is preserved, but offsets then differ from unforced
  runs -- never use it for parity).
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import mmap as _mmap
import os
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

# Precision policy (user directive, plan section 4): production default is
# TF32 ON -- this trainer does NOT set MLX_ENABLE_TF32 unless --no-tf32 is
# passed.  The Metal backend consults the env var at backend init, so it must
# be in the environment before `import mlx.core`; argparse runs far too late.
# Mechanism: peek at sys.argv pre-import (the plan's os.execv re-exec was the
# fallback for flags discovered post-import; the peek achieves the same
# ordering without re-running the interpreter, and the startup matmul probe
# below asserts it actually took effect).  An externally exported
# MLX_ENABLE_TF32 is respected and reported, never overridden -- the
# fp32-exact parity harnesses under mlx_parity/ pin =0 themselves.
if "--no-tf32" in sys.argv:
    os.environ["MLX_ENABLE_TF32"] = "0"

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.nn.utils import checkpoint as nn_checkpoint
from mlx.utils import tree_flatten, tree_unflatten

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from v7_model_mlx import (  # noqa: E402
    V7NetMLX, spectrogram, rms_normalize, cdist_sq, NFFT, HOP,
)

CORPUS = Path(os.environ.get(
    "CORPUS_DIR",
    HERE.parent.parent.parent / "artifacts/longdwell-production-corpus",
))

TARGET_FS = 20_000_000
DWELLS = {"1ms": 20_000, "2.5ms": 50_000, "10ms": 200_000}
CLASSES = ("am", "bluetooth", "cw", "dsss", "fm", "gsm", "ofdm")

# Mask-stream domain separator ("0xMA5C" in the plan shorthand; 'M' is not a
# hex digit, so the concrete constant is the ASCII bytes of "MA5C").  The
# mask stream is rng2 = default_rng([seed, MASK_STREAM_KEY]).
MASK_STREAM_KEY = 0x4D413543

CKPT_SCHEMA = "v7-mlx-ckpt-v1"
GB = 1 << 30


def n_frames(length: int) -> int:
    return (length - NFFT) // HOP + 1


# --- BEGIN SHARED EVAL PROTOCOL (byte-identical in v7_trainer.py) ----------
# The two trainers carry this block verbatim; the guard is
#   diff <(sed -n '/BEGIN SHARED EVAL PROTOCOL/,/END SHARED EVAL PROTOCOL/p' \
#            v7_trainer.py) \
#        <(sed -n '/BEGIN SHARED EVAL PROTOCOL/,/END SHARED EVAL PROTOCOL/p' \
#            v7_trainer_mlx.py)
# which must print nothing.
# Why this exists (measured: artifacts/mlx-g2/subsample_bias_test.json).  Eval
# rows inside a profile block are a fixed-stride slide across ONE long capture
# (GSM: startSampleIndex = 26000*j) and the GSM burst timeline has period
# 78000 = 3*26000, so the old mid-run subsample eval_rows[::3] locked onto a
# SINGLE burst phase -- GSM 1 ms recall 0.9955 on [::3] vs 0.4390 on the full
# split.  Worse: because every eval window started at capture offset 0, even
# the FULL split only ever sampled THREE discrete burst phases.  The protocol
# was measuring a phase, not a distribution.  Training was never affected (it
# draws a fresh random offset per row per episode) -- this is a measurement
# bug, not a learning bug.
#
# The fix, drawn ONCE at startup from a dedicated generator so every
# checkpoint of a run -- and both trainers -- score exactly the same thing:
#   * mid-run subsample: a seeded random subset, same size as [::3]
#   * eval windows: a per-row seeded random start offset (row -> offset
#     table) for query rows AND for prototype rows
# Nothing here touches the training streams (rng, rng2), so episode
# composition stays bit-identical to a pre-fix run.
EVAL_STREAM_KEY = 0xE7A1
EVAL_SUBSAMPLE_DIV = 3


def build_eval_plan(eval_seed, eval_rows, all_train, row_samples, max_dwell,
                    offset_mode="random", subsample_mode="random"):
    """Freeze the eval protocol at startup.

    Returns (mid_rows, eval_offsets, proto_offsets, meta); the two offset
    maps are row_id -> start sample.

    Offsets are drawn in [0, row_samples - max_dwell] so ONE offset per row
    is valid at EVERY dwell and the windows NEST: 2.5 ms starts at the same
    sample as 1 ms and strictly extends it, 10 ms extends 2.5 ms.  The
    escalation semantics -- a longer dwell sees a superset of what the
    shorter one saw -- are therefore preserved exactly as under offset 0,
    while burst phase becomes continuously sampled instead of pinned to 3
    discrete values.

    Prototypes get the same treatment (their own offsets from the same
    generator and the same range, drawn after the query offsets): a
    prototype built at one fixed capture offset is a phase-locked class
    centroid, i.e. the identical bias sitting on the reference side of the
    comparison.  Fixing only the query side would leave half the aliasing in
    place.

    Draw order is part of the contract and is deliberately unconditional --
    subsample indices, then eval-row offsets in split order, then
    prototype-row offsets in all_train order -- so the stream position never
    depends on the modes, and the random subsample is therefore the SAME set
    of rows whether offsets are random or zero.
    """
    if row_samples < max_dwell:
        raise SystemExit(f"row_samples {row_samples} < max dwell {max_dwell}")
    erng = np.random.default_rng([int(eval_seed), EVAL_STREAM_KEY])
    n_eval = len(eval_rows)
    n_sub = -(-n_eval // EVAL_SUBSAMPLE_DIV)      # == len(eval_rows[::3])
    hi = row_samples - max_dwell                  # inclusive upper bound
    rand_sub = np.sort(erng.choice(n_eval, size=n_sub, replace=False))
    rand_eval_off = erng.integers(0, hi + 1, size=n_eval)
    rand_proto_off = erng.integers(0, hi + 1, size=len(all_train))

    sub_idx = (np.arange(0, n_eval, EVAL_SUBSAMPLE_DIV)
               if subsample_mode == "stride" else rand_sub)
    mid_rows = [eval_rows[int(i)] for i in sub_idx]
    if offset_mode == "zero":
        eval_offsets = {r["row"]: 0 for r in eval_rows}
        proto_offsets = {int(rid): 0 for rid in all_train}
    else:
        eval_offsets = {r["row"]: int(o)
                        for r, o in zip(eval_rows, rand_eval_off)}
        proto_offsets = {int(rid): int(o)
                         for rid, o in zip(all_train, rand_proto_off)}
    eo = np.asarray([eval_offsets[r["row"]] for r in eval_rows],
                    dtype=np.int64)
    po = np.asarray([proto_offsets[int(r)] for r in all_train],
                    dtype=np.int64)
    h = hashlib.sha256()
    h.update(f"{offset_mode}|{subsample_mode}|{hi}|".encode())
    h.update(np.asarray(sub_idx, dtype=np.int64).tobytes())
    h.update(eo.tobytes())
    h.update(po.tobytes())
    meta = {
        "eval_seed": int(eval_seed),
        "stream_key": EVAL_STREAM_KEY,
        "offset_mode": offset_mode,
        "subsample_mode": subsample_mode,
        "offset_range": [0, int(hi)],
        "max_dwell": int(max_dwell),
        "n_eval_rows": n_eval,
        "n_mid_rows": len(mid_rows),
        "n_proto_rows": len(all_train),
        "eval_offset_mean": float(eo.mean()) if n_eval else 0.0,
        "eval_offset_min": int(eo.min()) if n_eval else 0,
        "eval_offset_max": int(eo.max()) if n_eval else 0,
        "plan_sha256": h.hexdigest(),
    }
    return mid_rows, eval_offsets, proto_offsets, meta
# --- END SHARED EVAL PROTOCOL ----------------------------------------------


# ---------------------------------------------------------------------------
# startup asserts
# ---------------------------------------------------------------------------
def measure_matmul_precision() -> tuple[float, bool]:
    """Tiny fp32 GPU matmul vs numpy float64 reference.

    Returns (max rel err, tf32_active).  Measured on this machine: ~2e-7
    with MLX_ENABLE_TF32=0, ~8e-4 with TF32 (the MLX M5 default; G0 saw
    2.7e-2 abs on this shape).  1e-5 cleanly separates the regimes.
    """
    prng = np.random.default_rng(0)
    a = prng.standard_normal((64, 128)).astype(np.float32)
    b = prng.standard_normal((128, 7)).astype(np.float32)
    ref = a.astype(np.float64) @ b.astype(np.float64)
    got = np.asarray(mx.array(a) @ mx.array(b)).astype(np.float64)
    rel = float(np.abs(got - ref).max() / np.abs(ref).max())
    return rel, rel >= 1e-5


# ---------------------------------------------------------------------------
# tree helpers (deep copy / in-place restore / hashing)
# ---------------------------------------------------------------------------
def tree_copy(t):
    """Deep copy of an mx-array tree (leaves round-tripped through numpy so
    the copies share no buffers with the originals)."""
    if isinstance(t, dict):
        return {k: tree_copy(v) for k, v in t.items()}
    if isinstance(t, (list, tuple)):
        return [tree_copy(v) for v in t]
    if isinstance(t, mx.array):
        if t.dtype == mx.bfloat16:      # numpy has no bf16; fp32 widen is
            return mx.array(            # exact and the cast back is too
                np.array(t.astype(mx.float32))).astype(mx.bfloat16)
        return mx.array(np.array(t))
    return copy.deepcopy(t)


def tree_assign_inplace(dst, src) -> None:
    """Assign src leaves into dst's containers WITHOUT replacing any
    container object -- compiled functions captured those containers."""
    if isinstance(dst, dict):
        if set(dst.keys()) != set(src.keys()):
            raise AssertionError(
                f"tree structure drift: {sorted(dst)} vs {sorted(src)}")
        for k in dst:
            if isinstance(dst[k], (dict, list)):
                tree_assign_inplace(dst[k], src[k])
            else:
                dst[k] = src[k]
    elif isinstance(dst, list):
        if len(dst) != len(src):
            raise AssertionError("tree list length drift")
        for i in range(len(dst)):
            if isinstance(dst[i], (dict, list)):
                tree_assign_inplace(dst[i], src[i])
            else:
                dst[i] = src[i]
    else:
        raise TypeError(f"unexpected tree node {type(dst)}")


def params_sha256(model: nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(tree_flatten(model.trainable_parameters())):
        h.update(k.encode())
        if v.dtype == mx.bfloat16:
            # numpy cannot view bf16 buffers; hash the exact fp32 widening,
            # tagged so fp32 and bf16 trees can never collide.  fp32-path
            # hashes are byte-identical to the pre---dtype trainer.
            h.update(b"bfloat16")
            h.update(np.ascontiguousarray(
                np.array(v.astype(mx.float32))).tobytes())
        else:
            h.update(np.ascontiguousarray(np.array(v)).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# model container: log_scale as a trainable scalar in the same tree
# ---------------------------------------------------------------------------
class TrainState(nn.Module):
    def __init__(self, width: int, embed_dim: int):
        super().__init__()
        self.net = V7NetMLX(width=width, embed_dim=embed_dim)
        self.log_scale = mx.array(math.log(10.0))


# ---------------------------------------------------------------------------
# episode draws (primary stream; call order is the fidelity contract)
# ---------------------------------------------------------------------------
@dataclass
class EpisodeDraw:
    ep: int
    length: int
    picks: list[int]
    offsets: np.ndarray
    thetas: np.ndarray
    thresh: float | None
    u: np.ndarray | None
    wd_ids: list[int] | None
    wd_weight: float


def draw_episode(ep, rng, rng2, cli, train_by_class, all_train, row_samples,
                 dwell_lengths, ramp_eps) -> EpisodeDraw:
    per = cli.k_shot + cli.q_query
    length = dwell_lengths[int(rng.integers(0, len(dwell_lengths)))]
    if getattr(cli, "force_dwell", None):
        # DIAGNOSTIC ONLY (pace probes): the dwell integer above was still
        # drawn, preserving the stream call order, but offsets below depend
        # on length so forced runs are not draw-comparable to unforced ones.
        length = DWELLS[cli.force_dwell]
    picks: list[int] = []
    for cls in CLASSES:
        ids = train_by_class[cls]
        chosen = rng.choice(len(ids), size=per, replace=len(ids) < per)
        picks.extend(ids[int(i)] for i in chosen)
    offsets = rng.integers(0, row_samples - length + 1, size=len(picks))
    thetas = rng.uniform(0, 2 * np.pi, size=(len(picks), 1))
    if cli.mask_frac > 0:
        thresh = float(rng.random())           # the ONE mask-threshold scalar
        t = n_frames(length)
        u = rng2.random((len(picks), 1, t, 1)).reshape(len(picks), t, 1, 1)
    else:
        thresh, u = None, None
    wd_weight = cli.worst_dur_weight * min(1.0, ep / ramp_eps)
    wd_ids = None
    if wd_weight > 0 and ep % 4 == 0:          # exact torch guard: ep 0 -> NO draw
        wd_weight *= 4.0
        wd_ids = [all_train[int(i)] for i in
                  rng.choice(len(all_train), size=cli.worst_dur_rows,
                             replace=False)]
    else:
        wd_weight = 0.0
    return EpisodeDraw(ep, length, picks, offsets, thetas, thresh, u,
                       wd_ids, wd_weight)


def dump_draw(fh, d: EpisodeDraw) -> None:
    rec = {
        "ep": d.ep, "dwell": d.length, "picks": d.picks,
        "offsets": [int(o) for o in d.offsets],
        "thetas": [float(t).hex() for t in d.thetas.ravel().tolist()],
        "thresh": None if d.thresh is None else float(d.thresh).hex(),
        "wd_rows": d.wd_ids,
        "wd_weight": float(d.wd_weight).hex(),
    }
    fh.write(json.dumps(rec) + "\n")
    fh.flush()


# ---------------------------------------------------------------------------
# checkpoints: safetensors + sidecar JSON, atomic temp+rename
# ---------------------------------------------------------------------------
def _atomic_write_bytes(path: str, writer) -> None:
    tmp = path + ".tmp"
    writer(tmp)
    os.replace(tmp, path)


def save_checkpoint(path, model, opt, cli, episode, rng_state, rng2_state,
                    history) -> None:
    # mx.save_safetensors/mx.load dispatch on the extension; keep it on the
    # temp file so the atomic rename lands on the real name.
    if not path.endswith(".safetensors"):
        raise SystemExit(
            f"checkpoint path must end in .safetensors (got {path!r})")
    arrays = {}
    for k, v in tree_flatten(model.trainable_parameters()):
        arrays["params." + k] = v
    for k, v in tree_flatten(opt.state):
        arrays["optstate." + k] = v
    tmp = path[:-len(".safetensors")] + ".tmp.safetensors"
    mx.save_safetensors(tmp, arrays,
                        metadata={"schema": CKPT_SCHEMA,
                                  "episode": str(episode)})
    os.replace(tmp, path)
    sidecar = {
        "schema": CKPT_SCHEMA,
        "trainer_schema": "v7-trainer-v1",
        "config": vars(cli),
        "episode": episode,
        "rng_state": rng_state,
        "rng2_state": rng2_state,
        "eval_history": history,
        "params_sha256": params_sha256(model),
    }

    def _write_json(tmp):
        Path(tmp).write_text(json.dumps(sidecar))

    _atomic_write_bytes(path + ".json", _write_json)


def rolling_path(path: str) -> str:
    if path.endswith(".safetensors"):
        return path[:-len(".safetensors")] + ".partial.safetensors"
    return path + ".partial.safetensors"


def load_init_checkpoint(model: TrainState, path: str):
    """Load either a native MLX trainer checkpoint or a convert.py-produced
    torch checkpoint (44 bare V7Net keys + log_scale metadata).

    Returns (kind, opt_state_tree_or_None, sidecar_or_None).
    """
    arrays, meta = mx.load(path, return_metadata=True)
    if any(k.startswith("params.") for k in arrays):
        params = tree_unflatten(
            [(k[len("params."):], v) for k, v in arrays.items()
             if k.startswith("params.")])
        model.update(params)
        opt_items = [(k[len("optstate."):], v) for k, v in arrays.items()
                     if k.startswith("optstate.")]
        opt_state = tree_unflatten(opt_items) if opt_items else None
        sidecar = None
        if os.path.exists(path + ".json"):
            sidecar = json.loads(Path(path + ".json").read_text())
        return "native", opt_state, sidecar
    # convert.py artifact: torch-style keys, MLX conv layout, conf.<i>.*
    remapped = []
    for k, v in arrays.items():
        if k.startswith("conf."):
            _, rest = k.split(".", 1)
            k = f"conf.layers.{rest}"
        remapped.append((k, v))
    model.net.load_weights(remapped, strict=True)
    model.log_scale = mx.array(float(meta["log_scale"]))
    return "torch", None, None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=6000)
    ap.add_argument("--eval-every", type=int, default=1500)
    ap.add_argument("--k-shot", type=int, default=5)
    ap.add_argument("--q-query", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--recon-weight", type=float, default=0.3)
    ap.add_argument("--mask-frac", type=float, default=0.5)
    ap.add_argument("--worst-dur-weight", type=float, default=0.15)
    ap.add_argument("--worst-dur-ramp-frac", type=float, default=0.1)
    ap.add_argument("--worst-dur-rows", type=int, default=14)
    ap.add_argument("--conf-weight", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20260740)
    ap.add_argument("--out", default=str(HERE / "v7_result_mlx.json"))
    ap.add_argument("--save-checkpoint", default=None)
    ap.add_argument("--init-checkpoint", default=None,
                    help="native MLX checkpoint or convert.py safetensors")
    ap.add_argument("--start-episode", type=int, default=0)
    ap.add_argument("--torch-compat-resume", action="store_true",
                    help="resume with torch semantics: stream jumps to "
                         "default_rng([seed, start_episode]), fresh "
                         "optimizer moments, schedule fast-forwarded")
    ap.add_argument("--stop-episode", type=int, default=0,
                    help="stop after completing this many episodes "
                         "(schedules still shaped by --episodes); 0 = off")
    ap.add_argument("--dump-draws", default=None,
                    help="JSONL path: per-episode draw tuples (G2/G2b)")
    ap.add_argument("--max-train-per-class", type=int, default=0,
                    help="cap train rows per class (0 = all; smoke only)")
    ap.add_argument("--max-eval-per-class", type=int, default=0,
                    help="cap eval rows per class (0 = all; smoke only)")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--eval-offset-mode", choices=("zero", "random"),
                    default="random",
                    help="eval window start offsets: 'random' (default) "
                         "draws a per-row offset ONCE at startup from the "
                         "dedicated eval stream in [0, rowSamples-maxDwell] "
                         "(nested across dwells, applied to query AND "
                         "prototype rows); 'zero' restores the legacy "
                         "offset-0 protocol, which samples only 3 discrete "
                         "burst phases -- kept for comparison only")
    ap.add_argument("--eval-subsample-mode", choices=("stride", "random"),
                    default="random",
                    help="mid-run eval subsample: 'random' (default) is a "
                         "seeded random 1/3 subset fixed at startup; "
                         "'stride' restores the legacy eval_rows[::3], "
                         "which aliases the GSM burst period -- comparison "
                         "only")
    ap.add_argument("--eval-offset-seed", type=int, default=-1,
                    help="seed for the dedicated eval-protocol stream "
                         "(-1 = use --seed); vary it to measure the "
                         "protocol's own sampling uncertainty")
    ap.add_argument("--wired-limit-gb", type=float, default=22.0)
    ap.add_argument("--no-tf32", action="store_true",
                    help="pin MLX_ENABLE_TF32=0 (applied pre-import by the "
                         "sys.argv peek at module top; fp32-exact matmuls "
                         "for bisection -- production default is TF32 ON)")
    ap.add_argument("--dtype", choices=("fp32", "bf16", "fp16"),
                    default="fp32",
                    help="net compute dtype (params + activations via "
                         "set_dtype); spectrogram/rms stay fp32 and are "
                         "cast at model input; losses reduced in fp32; MLX "
                         "AdamW keeps moments/update math in this dtype "
                         "(no fp32 master weights); fp16 has NO loss "
                         "scaling in MLX -- prefer bf16")
    ap.add_argument("--ckpt-blocks", type=int, default=-1,
                    help="gradient-checkpoint the first N of stem,d1..d5 "
                         "in the train step; -1 (default) = auto: 3 for "
                         "the 200000-sample dwell signatures, 0 for short "
                         "dwells; an explicit N applies to ALL dwells")
    ap.add_argument("--batch-chunks", type=int, default=-1,
                    help="split the episode batch into N equal checkpointed "
                         "encoder/decoder units inside the compiled step "
                         "(plan section 4 contingency, exact gradient "
                         "accumulation); -1 (default) = auto: 5 for the "
                         "200000-sample dwell signatures, 1 for short "
                         "dwells; an explicit N applies to ALL dwells and "
                         "must divide the 70-row batch")
    ap.add_argument("--mem-limit-gb", type=float, default=22.0,
                    help="mx.set_memory_limit throttle (plan: throttle not "
                         "tripwire), set to post-corpus active + this many "
                         "GB; serializes the backward's conv-vjp "
                         "temporaries so checkpoint units stop "
                         "co-allocating; 0 disables")
    ap.add_argument("--probe-memory", action="store_true",
                    help="measure per-block and train-step peak memory at "
                         "T=6249 B=70, print a table, and exit (loads a "
                         "tiny corpus slice; never trains or saves)")
    ap.add_argument("--force-dwell", choices=tuple(DWELLS), default=None,
                    help="DIAGNOSTIC: force every episode to this dwell "
                         "(stream call order preserved but offsets differ "
                         "from unforced runs -- pace probes only)")
    cli = ap.parse_args()

    cdtype = {"fp32": mx.float32, "bf16": mx.bfloat16,
              "fp16": mx.float16}[cli.dtype]
    if cdtype is mx.float32:
        # identity lambdas keep the fp32 graph BYTE-identical to the
        # pre---dtype trainer (no astype nodes inserted)
        def to_c(a): return a
        def to_f(a): return a
    else:
        def to_c(a): return a.astype(cdtype)
        def to_f(a): return a.astype(mx.float32)

    def ckpt_for(length: int) -> int:
        """Encoder blocks to checkpoint for a dwell length.  Called at
        compile-trace time (shapes are concrete per signature), so each of
        the 6 signatures bakes in its own value."""
        if cli.ckpt_blocks >= 0:
            return min(cli.ckpt_blocks, 6)
        return 3 if length >= DWELLS["10ms"] else 0

    def chunks_for(length: int) -> int:
        """Checkpointed batch-chunk count for a dwell length (same
        trace-time semantics as ckpt_for)."""
        if cli.batch_chunks >= 1:
            return cli.batch_chunks
        return 5 if length >= DWELLS["10ms"] else 1

    if cli.probe_memory and not cli.max_train_per_class:
        # the probe batch is synthetic; don't spend minutes wiring 20 GB
        cli.max_train_per_class, cli.max_eval_per_class = 2, 1

    t_boot = time.perf_counter()
    tf32_rel, tf32_active = measure_matmul_precision()
    if cli.no_tf32:
        assert not tf32_active, (
            f"--no-tf32 given but fp32 matmul rel err {tf32_rel:.3e} >= 1e-5"
            f" -- MLX_ENABLE_TF32={os.environ.get('MLX_ENABLE_TF32')!r} was "
            "set too late or ignored (the sys.argv peek at module top must "
            "run before the mlx import)")
    print(f"mlx {mx.__version__}  device={mx.default_device()}  "
          f"TF32 {'ON' if tf32_active else 'off'} "
          f"(matmul rel err {tf32_rel:.2e}"
          f"{', --no-tf32' if cli.no_tf32 else ''}"
          + (", external env" if not cli.no_tf32 and not tf32_active else "")
          + f")  dtype={cli.dtype}", flush=True)

    try:
        mx.set_wired_limit(int(cli.wired_limit_gb * GB))
        print(f"wired limit {cli.wired_limit_gb:.0f} GB", flush=True)
    except Exception as e:  # report loudly but do not kill the run
        print(f"WARNING: set_wired_limit failed: {e!r}", flush=True)
    mx.set_cache_limit(2 * GB)

    # ---------------- corpus ----------------
    manifest = json.loads((CORPUS / "manifest.json").read_text())
    noisy = np.load(CORPUS / "noisy.npy", mmap_mode="r")
    clean = np.load(CORPUS / "clean.npy", mmap_mode="r")
    row_samples = manifest["rowSamples"]
    rows = manifest["rows"]
    train_by_class: dict[str, list[int]] = {c: [] for c in CLASSES}
    class_index = {c: i for i, c in enumerate(CLASSES)}
    eval_rows: list[dict] = []
    eval_count: dict[str, int] = {c: 0 for c in CLASSES}
    for row in rows:
        if row["role"] == "train":
            train_by_class[row["cls"]].append(row["row"])
        else:
            if (cli.max_eval_per_class
                    and eval_count[row["cls"]] >= cli.max_eval_per_class):
                continue
            eval_count[row["cls"]] += 1
            eval_rows.append(row)
    if cli.max_train_per_class:
        for c in CLASSES:
            train_by_class[c] = train_by_class[c][:cli.max_train_per_class]
    row_cls = {r["row"]: r["cls"] for r in rows}
    all_train = [r for ids in train_by_class.values() for r in ids]
    print(f"train {len(all_train)}  eval {len(eval_rows)}  "
          f"profiles {len({r['profile'] for r in rows})}"
          + ("  [REDUCED corpus]" if cli.max_train_per_class
             or cli.max_eval_per_class else ""), flush=True)

    # eval protocol frozen ONCE, here, on its own stream (see build_eval_plan)
    eval_seed = cli.seed if cli.eval_offset_seed < 0 else cli.eval_offset_seed
    mid_eval_rows, eval_offsets, proto_offsets, eval_plan = build_eval_plan(
        eval_seed, eval_rows, all_train, row_samples, max(DWELLS.values()),
        offset_mode=cli.eval_offset_mode,
        subsample_mode=cli.eval_subsample_mode)
    eval_zero = cli.eval_offset_mode == "zero"
    print(f"eval plan: offsets={cli.eval_offset_mode} "
          f"subsample={cli.eval_subsample_mode} seed={eval_seed} "
          f"mid-rows={len(mid_eval_rows)}/{len(eval_rows)} "
          f"offset range {eval_plan['offset_range']} "
          f"(mean {eval_plan['eval_offset_mean']:.0f}) "
          f"sha {eval_plan['plan_sha256'][:16]}...", flush=True)

    # -------- resident conversion: chunked, eval rows FIRST (plan sec 4) ----
    t_conv = time.perf_counter()
    mx.reset_peak_memory()
    n_train, n_eval = len(all_train), len(eval_rows)
    train_pos = {rid: slot for slot, rid in enumerate(all_train)}
    eval_pos = {r["row"]: slot for slot, r in enumerate(eval_rows)}
    rows_per_chunk = max(1, (512 << 20) // (row_samples * 8))  # source bytes

    eval_store = mx.zeros((n_eval, row_samples), dtype=mx.complex64)
    eval_ids = [r["row"] for r in eval_rows]
    for i0 in range(0, n_eval, rows_per_chunk):
        ids = eval_ids[i0:i0 + rows_per_chunk]
        chunk = np.ascontiguousarray(noisy[ids])          # complex64
        eval_store[i0:i0 + len(ids)] = mx.array(chunk)
        mx.eval(eval_store)
        del chunk

    train_iq = mx.zeros((n_train, 2, row_samples), dtype=mx.float16)
    for i0 in range(0, n_train, rows_per_chunk):
        ids = all_train[i0:i0 + rows_per_chunk]
        rc = np.asarray(noisy[ids])
        planes = np.stack((rc.real, rc.imag), axis=1).astype(np.float16)
        train_iq[i0:i0 + len(ids)] = mx.array(planes)
        mx.eval(train_iq)
        del rc, planes

    clean_iq = mx.zeros((n_train, 2, 50_000), dtype=mx.float16)
    for i0 in range(0, n_train, rows_per_chunk):
        ids = all_train[i0:i0 + rows_per_chunk]
        cc = np.asarray(clean[ids, :50_000])
        planes = np.stack((cc.real, cc.imag), axis=1).astype(np.float16)
        clean_iq[i0:i0 + len(ids)] = mx.array(planes)
        mx.eval(clean_iq)
        del cc, planes

    conv_peak = mx.get_peak_memory()
    conv_active = mx.get_active_memory()
    assert conv_peak - conv_active < 1 * GB, (
        f"conversion transient residency too high: peak {conv_peak/1e9:.2f} "
        f"GB vs active {conv_active/1e9:.2f} GB (plan: delta < 1 GB)")
    print(f"resident corpus: eval {eval_store.nbytes/1e9:.2f} GB c64, "
          f"train {train_iq.nbytes/1e9:.2f} GB + clean "
          f"{clean_iq.nbytes/1e9:.2f} GB fp16 in "
          f"{time.perf_counter()-t_conv:.0f}s "
          f"(peak-active {(conv_peak-conv_active)/1e6:.0f} MB)", flush=True)

    # Memory throttle (plan: set_memory_limit is a throttle, not a
    # tripwire).  Anchored to post-corpus active so the resident stores
    # never count against the step budget.  Without it MLX encodes the whole
    # backward ahead and the conv-vjp temporaries co-allocate (~sum instead
    # of max); with it plus the checkpoint units the scheduler serializes
    # them at no measured pace cost.
    if cli.mem_limit_gb > 0:
        base = mx.get_active_memory()
        mx.set_memory_limit(int(base + cli.mem_limit_gb * GB))
        print(f"memory throttle: active {base/1e9:.2f} GB + "
              f"{cli.mem_limit_gb:.0f} GB budget", flush=True)

    # memmaps are one-shot: purge and drop
    for m in (noisy, clean):
        mm = getattr(m, "_mmap", None)
        if mm is not None:
            try:
                mm.madvise(_mmap.MADV_DONTNEED)
            except (AttributeError, OSError, ValueError):
                pass
    del noisy, clean
    gc.collect()

    # ---------------- model / optimizer ----------------
    # deterministic native init (mirrors torch.manual_seed(seed + start_ep);
    # only model init consumes mx.random -- overwritten on any resume)
    mx.random.seed(cli.seed + cli.start_episode)
    model = TrainState(width=cli.width, embed_dim=cli.embed_dim)
    n_params = sum(v.size for _, v in tree_flatten(model.net.parameters()))
    print(f"params: {n_params:,}", flush=True)

    sched = optim.cosine_decay(cli.lr, cli.episodes, end=0.0)
    opt = optim.MultiOptimizer(
        [optim.AdamW(learning_rate=sched, betas=[0.9, 0.999], eps=1e-8,
                     weight_decay=0.0, bias_correction=True),
         optim.AdamW(learning_rate=sched, betas=[0.9, 0.999], eps=1e-8,
                     weight_decay=cli.weight_decay, bias_correction=True)],
        [lambda path, g: "log_scale" in path])

    # ---------------- resume: params first, then dtype, then optimizer -----
    # Load order matters for --dtype: opt.init(zeros_like) must see the
    # FINAL param dtype so the moments natively take it (plan-proven
    # sequence build -> init(params) -> assign state is preserved).
    start_episode = cli.start_episode
    history: list[dict] = []
    kind, opt_state, sidecar = None, None, None
    provenance = "native-mlx-init"
    if cli.init_checkpoint:
        kind, opt_state, sidecar = load_init_checkpoint(
            model, cli.init_checkpoint)
        mx.eval(model.parameters())
        provenance = f"{kind}:{cli.init_checkpoint}"
        if kind == "native" and not cli.torch_compat_resume:
            if sidecar is None:
                raise SystemExit(
                    f"native resume needs sidecar {cli.init_checkpoint}.json")
            if cli.start_episode and cli.start_episode != sidecar["episode"]:
                raise SystemExit(
                    f"--start-episode {cli.start_episode} conflicts with "
                    f"sidecar episode {sidecar['episode']}")
            start_episode = sidecar["episode"]
            history = list(sidecar.get("eval_history", []))
            ck_dtype = sidecar.get("config", {}).get("dtype", "fp32")
            if ck_dtype != cli.dtype:
                print(f"WARNING: resuming a {ck_dtype} checkpoint with "
                      f"--dtype {cli.dtype}; params are re-cast but the "
                      "restored optimizer moments keep the saved dtype",
                      flush=True)
            got = params_sha256(model)   # hash of the AS-LOADED params
            want = sidecar.get("params_sha256")
            if want and got != want:
                raise SystemExit(
                    f"checkpoint params hash mismatch: {got} != {want}")

    # --dtype: net params + activations in the compute dtype; log_scale
    # lives outside .net and stays fp32 (only used in fp32 loss math).
    # fp32 mode never calls set_dtype, keeping the default path untouched.
    if cdtype is not mx.float32:
        model.net.set_dtype(cdtype)
        mx.eval(model.parameters())

    opt.init(model.trainable_parameters())
    if cli.init_checkpoint:
        if kind == "native" and not cli.torch_compat_resume:
            if opt_state is not None:
                # rebuild -> init -> assign state (plan-proven sequence);
                # the setter is safe here: compiled fns not yet defined.
                opt.state = opt_state
            print(f"resumed {cli.init_checkpoint} at ep {start_episode} "
                  f"(native stream continuation; optimizer state restored, "
                  f"steps={int(opt.optimizers[0].step.item())}/"
                  f"{int(opt.optimizers[1].step.item())})", flush=True)
        else:
            # torch-format init and/or torch-compat resume: fresh moments,
            # schedule fast-forwarded via the step counters.
            for o in opt.optimizers:
                o.state["step"] = mx.array(start_episode, mx.uint64)
            print(f"loaded {cli.init_checkpoint} ({kind}) at ep "
                  f"{start_episode}; optimizer moments start fresh",
                  flush=True)

    # primary + mask streams (plan differences ledger item 6)
    if (cli.init_checkpoint and sidecar is not None
            and not cli.torch_compat_resume):
        rng = np.random.default_rng(cli.seed)
        rng.bit_generator.state = sidecar["rng_state"]
        rng2 = np.random.default_rng([cli.seed, MASK_STREAM_KEY])
        rng2.bit_generator.state = sidecar["rng2_state"]
    elif cli.torch_compat_resume and start_episode:
        rng = np.random.default_rng([cli.seed, start_episode])
        rng2 = np.random.default_rng([cli.seed, MASK_STREAM_KEY])
    else:
        rng = np.random.default_rng(cli.seed)
        rng2 = np.random.default_rng([cli.seed, MASK_STREAM_KEY])

    ramp_eps = max(1, int(cli.episodes * cli.worst_dur_ramp_frac))
    dwell_lengths = list(DWELLS.values())
    n_cls = len(CLASSES)
    per = cli.k_shot + cli.q_query
    batch_b = n_cls * per

    # ---------------- losses ----------------
    target_const = mx.array(
        np.repeat(np.arange(n_cls), cli.q_query).astype(np.int32))
    mask_on = cli.mask_frac > 0

    def episode_terms(mdl, x, theta, u, thresh, c):
        rot = mx.cos(theta) + 1j * mx.sin(theta)          # torch.polar
        xr = rms_normalize(x) * rot
        spec = spectrogram(xr)                            # [B, T, 33, 3] fp32
        cr = rms_normalize(c) * rot                       # SAME rot on clean
        spec_clean = spectrogram(cr)                      # fp32 loss target
        if mask_on:
            keep = u > (cli.mask_frac * thresh)           # [B, T, 1, 1]
            spec_in = spec * keep
        else:
            spec_in = spec
        # front end stays fp32; cast at model input (--dtype), losses fp32
        n_ck = ckpt_for(x.shape[-1])
        n_chunks = chunks_for(x.shape[-1])
        if n_chunks > 1:
            # Plan section 4 contingency, generalized: split the batch into
            # equal checkpointed encoder/decoder units INSIDE the compiled
            # step.  Exact: convs/GroupNorm/pooling are per-sample, protos
            # and CE still see all rows' z, and the mean of equal-chunk l1
            # means equals the global l1 mean (fp summation order aside).
            bsz = spec_in.shape[0]
            if bsz % n_chunks:
                raise SystemExit(
                    f"--batch-chunks {n_chunks} does not divide batch {bsz}")
            cs = bsz // n_chunks
            enc = nn_checkpoint(
                mdl.net, lambda s: mdl.net.encode(s, ckpt_blocks=n_ck))
            dec = nn_checkpoint(
                mdl.net, lambda hh: mdl.net.reconstruct(
                    hh, spec_clean.shape[1], spec.shape[2]))
            zs, confs, rls = [], [], []
            for i in range(n_chunks):
                z_h, conf_h, h_h = enc(to_c(spec_in[i * cs:(i + 1) * cs]))
                r_h = to_f(dec(h_h))
                rls.append(nn.losses.l1_loss(
                    r_h, spec_clean[i * cs:(i + 1) * cs], reduction="mean"))
                zs.append(z_h)
                confs.append(conf_h)
            z = to_f(mx.concatenate(zs, axis=0))
            conf_logit = to_f(mx.concatenate(confs, axis=0))
            recon_loss = sum(rls[1:], rls[0]) / n_chunks
        else:
            z, conf_logit, h = mdl.net.encode(
                to_c(spec_in), ckpt_blocks=n_ck)
            z, conf_logit = to_f(z), to_f(conf_logit)
            recon = to_f(mdl.net.reconstruct(h, spec_clean.shape[1],
                                             spec.shape[2], ckpt=n_ck > 0))
            recon_loss = nn.losses.l1_loss(recon, spec_clean,
                                           reduction="mean")
        zc = z.reshape(n_cls, per, -1)
        protos = mx.mean(zc[:, :cli.k_shot], axis=1)      # NOT renormalized
        query = zc[:, cli.k_shot:].reshape(-1, z.shape[-1])
        scale = mx.clip(mx.exp(mdl.log_scale), 1e-3, 100.0)
        logits = -cdist_sq(query, protos) * scale
        ce = nn.losses.cross_entropy(logits, target_const, reduction="mean")
        query_conf = conf_logit.reshape(n_cls, per)[:, cli.k_shot:].reshape(-1)
        correct = mx.stop_gradient(
            (mx.argmax(logits, axis=1) == target_const).astype(mx.float32))
        conf_loss = nn.losses.binary_cross_entropy(
            query_conf, correct, with_logits=True, reduction="mean")
        loss = (ce + cli.recon_weight * recon_loss
                + cli.conf_weight * conf_loss)
        return loss, ce, recon_loss, conf_loss, protos

    def loss_noaux(mdl, x, theta, u, thresh, c):
        loss, ce, rl, cl, _ = episode_terms(mdl, x, theta, u, thresh, c)
        return loss, (ce, rl, cl)

    def loss_aux(mdl, x, theta, u, thresh, c, wd_x, wd_t, wd_w):
        loss, ce, rl, cl, protos = episode_terms(mdl, x, theta, u, thresh, c)
        protos_d = mx.stop_gradient(protos)
        scale_d = mx.clip(mx.exp(mx.stop_gradient(mdl.log_scale)),
                          1e-3, 100.0)
        per_dwell = []
        for length in dwell_lengths:            # same dwells as torch aux
            xw = rms_normalize(wd_x[:, :length])
            zw, _, _ = mdl.net.encode(to_c(spectrogram(xw)),
                                      ckpt_blocks=ckpt_for(length))
            lw = -cdist_sq(to_f(zw), protos_d) * scale_d
            per_dwell.append(
                nn.losses.cross_entropy(lw, wd_t, reduction="none"))
        wd_ce = mx.mean(mx.max(mx.stack(per_dwell, axis=1), axis=1))
        return loss + wd_w * wd_ce, (ce, rl, cl, wd_ce)

    vg_noaux = nn.value_and_grad(model, loss_noaux)
    vg_aux = nn.value_and_grad(model, loss_aux)

    state = [model.state, opt.state]  # capture ONCE, after any resume-restore

    @partial(mx.compile, inputs=state, outputs=state)
    def step_noaux(x, theta, u, thresh, c):
        (loss, aux), grads = vg_noaux(model, x, theta, u, thresh, c)
        opt.update(model, grads)
        return (loss,) + aux

    @partial(mx.compile, inputs=state, outputs=state)
    def step_aux(x, theta, u, thresh, c, wd_x, wd_t, wd_w):
        (loss, aux), grads = vg_aux(model, x, theta, u, thresh, c,
                                    wd_x, wd_t, wd_w)
        opt.update(model, grads)
        return (loss,) + aux

    @partial(mx.compile, inputs=[model.state])
    def eval_forward(xb):
        # forward-only: no checkpointing; fp32 front end, --dtype compute,
        # z/conf back in fp32 for the (numpy) metric path
        z, conf, _ = model.net.encode(to_c(spectrogram(rms_normalize(xb))))
        return to_f(z), to_f(conf)

    dummy_u = mx.zeros((1, 1, 1, 1))

    # ---------------- --probe-memory: one train step at T=6249 B=70 -------
    if cli.probe_memory:
        L = DWELLS["10ms"]                     # 200_000 -> T = 6249
        tL = n_frames(L)
        tc = n_frames(min(L, 50_000))          # decoder target T (1561)
        n_ck = ckpt_for(L)
        n_chk = chunks_for(L)
        prng = np.random.default_rng(0xBEEF)
        px = mx.array((prng.standard_normal((batch_b, L))
                       + 1j * prng.standard_normal((batch_b, L))
                       ).astype(np.complex64))
        ptheta = mx.array(prng.uniform(0, 2 * np.pi, (batch_b, 1))
                          .astype(np.float32))
        pu = (mx.array(prng.random((batch_b, tL, 1, 1)).astype(np.float32))
              if mask_on else dummy_u)
        pthresh = mx.array(np.float32(0.5))
        pc = mx.array((prng.standard_normal((batch_b, 50_000))
                       + 1j * prng.standard_normal((batch_b, 50_000))
                       ).astype(np.complex64))
        mx.eval(px, ptheta, pu, pc)
        mx.clear_cache()
        base_active = mx.get_active_memory()
        print(f"[probe-memory] dtype={cli.dtype} ckpt_blocks@10ms={n_ck} "
              f"batch_chunks@10ms={n_chk} mem_limit={cli.mem_limit_gb:g}GB "
              f"B={batch_b} T={tL}  baseline active "
              f"{base_active/1e9:.2f} GB", flush=True)

        rows_out: list[tuple[str, int, int, tuple]] = []

        def _measure(name, fn, *args):
            a0 = mx.get_active_memory()
            mx.reset_peak_memory()
            out = fn(*args)
            mx.eval(out)
            rows_out.append(
                (name, mx.get_peak_memory() - a0,
                 mx.get_active_memory() - a0,
                 tuple(out.shape) if isinstance(out, mx.array) else ()))
            return out

        # eager per-block forward (checkpointing is a backward-only effect,
        # so these forward deltas are ckpt-invariant); every activation is
        # kept alive, mimicking train-step forward residency
        rot = mx.cos(ptheta) + 1j * mx.sin(ptheta)
        spec = spectrogram(rms_normalize(px) * rot)
        spec_in = to_c((spec * (pu > (cli.mask_frac * pthresh)))
                       if mask_on else spec)
        mx.eval(spec_in)
        acts = [spec_in]
        h = spec_in
        net = model.net
        for nm, blk in (("stem", net.stem), ("d1", net.d1), ("d2", net.d2),
                        ("d3", net.d3), ("d4", net.d4), ("d5", net.d5)):
            h = _measure(nm, blk, h)
            acts.append(h)

        def _head(a):
            pooled = mx.concatenate(
                (mx.mean(a, axis=(1, 2)), mx.max(a, axis=(1, 2))), axis=1)
            z = net.embed(pooled)
            z = z / mx.maximum(
                mx.linalg.norm(z, axis=-1, keepdims=True), 1e-12) * 4.0
            return mx.stack([mx.sum(z), mx.sum(net.conf(pooled))])

        _measure("head", _head, h)
        hd = _measure("dec.u1", net.u1, h, (max(1, tc // 8), max(1, 33 // 4)))
        hd = _measure("dec.u2", net.u2, hd, (max(1, tc // 4), max(1, 33 // 2)))
        hd = _measure("dec.u3", net.u3, hd, (tc, 33))
        _measure("dec.out", net.out, hd)

        print(f"  {'block':<8} {'peak-delta':>12} {'act-delta':>12}  shape",
              flush=True)
        for nm, pk, ad, shp in rows_out:
            print(f"  {nm:<8} {pk/1e9:>10.2f}GB {ad/1e9:>10.2f}GB  {shp}",
                  flush=True)

        del acts, h, hd, spec, spec_in
        gc.collect()
        mx.clear_cache()

        # full fwd+bwd, eager (same closures as training)
        a0 = mx.get_active_memory()
        mx.reset_peak_memory()
        (pl, _paux), pgrads = vg_noaux(model, px, ptheta, pu, pthresh, pc)
        mx.eval(pl, pgrads)
        eager_peak = mx.get_peak_memory()
        eager_delta = eager_peak - a0
        del pgrads
        gc.collect()
        mx.clear_cache()

        # full train step, compiled (proves checkpoint composes with
        # mx.compile at scale; mutates model/opt -- probe never trains)
        a1 = mx.get_active_memory()
        mx.reset_peak_memory()
        pout = step_noaux(px, ptheta, pu, pthresh, pc)
        mx.eval(pout, state)
        comp_peak = mx.get_peak_memory()
        print(f"  train-step eager   : peak {eager_peak/1e9:.2f} GB "
              f"(delta {eager_delta/1e9:.2f} GB)  loss {float(pl):.4f}",
              flush=True)
        print(f"  train-step compiled: peak {comp_peak/1e9:.2f} GB "
              f"(delta {(comp_peak-a1)/1e9:.2f} GB)  loss "
              f"{float(pout[0]):.4f}  "
              f"[target < 26 GB fp32 @ ckpt={n_ck} chunks={n_chk}]",
              flush=True)
        return

    # ---------------- batch assembly (resident mx stores) ----------------
    def assemble(d: EpisodeDraw):
        slots = mx.array(np.array([train_pos[r] for r in d.picks],
                                  dtype=np.int32))
        g = mx.take(train_iq, slots, axis=0)              # [B,2,rowS] fp16
        idx = mx.array((d.offsets[:, None, None]
                        + np.arange(d.length)[None, None, :]
                        ).astype(np.int32))               # [B,1,L]
        seg = mx.take_along_axis(g, idx, axis=2).astype(mx.float32)
        x = seg[:, 0] + 1j * seg[:, 1]                    # complex64 [B,L]
        recon_len = min(d.length, 50_000)
        gc_ = mx.take(clean_iq, slots, axis=0)[:, :, :recon_len].astype(
            mx.float32)
        c = gc_[:, 0] + 1j * gc_[:, 1]
        theta = mx.array(d.thetas.astype(np.float32))
        if d.u is not None:
            u = mx.array(d.u.astype(np.float32))
            thresh = mx.array(np.float32(d.thresh))
        else:
            u, thresh = dummy_u, mx.array(np.float32(0.0))
        if d.wd_ids is not None:
            wslots = mx.array(np.array([train_pos[r] for r in d.wd_ids],
                                       dtype=np.int32))
            wg = mx.take(train_iq, wslots, axis=0)[:, :, :200_000].astype(
                mx.float32)
            wd_x = wg[:, 0] + 1j * wg[:, 1]
            wd_t = mx.array(np.array(
                [class_index[row_cls[r]] for r in d.wd_ids], dtype=np.int32))
            wd_w = mx.array(np.float32(d.wd_weight))
            return (x, theta, u, thresh, c, wd_x, wd_t, wd_w)
        return (x, theta, u, thresh, c)

    # ---------------- warmup: exactly 6 signatures, throwaway state --------
    p_snapshot = tree_copy(model.trainable_parameters())
    s_snapshot = tree_copy(state[1])
    hash_before = params_sha256(model)
    steps_before = (int(opt.optimizers[0].step.item()),
                    int(opt.optimizers[1].step.item()))

    warmed: set[tuple[bool, int]] = set()
    warm_peaks: dict[str, int] = {}
    wrng = np.random.default_rng(0xC0FFEE)   # never touches rng/rng2
    t_warm = time.perf_counter()
    for length in dwell_lengths:
        t = n_frames(length)
        recon_len = min(length, 50_000)
        wx = mx.array((wrng.standard_normal((batch_b, length))
                       + 1j * wrng.standard_normal((batch_b, length))
                       ).astype(np.complex64))
        wtheta = mx.array(wrng.uniform(0, 2 * np.pi, (batch_b, 1))
                          .astype(np.float32))
        wu = (mx.array(wrng.random((batch_b, t, 1, 1)).astype(np.float32))
              if mask_on else dummy_u)
        wthresh = mx.array(np.float32(0.5))
        wc = mx.array((wrng.standard_normal((batch_b, recon_len))
                       + 1j * wrng.standard_normal((batch_b, recon_len))
                       ).astype(np.complex64))
        wwd_x = mx.array((wrng.standard_normal((cli.worst_dur_rows, 200_000))
                          + 1j * wrng.standard_normal((cli.worst_dur_rows,
                                                       200_000))
                          ).astype(np.complex64))
        wwd_t = mx.array((np.arange(cli.worst_dur_rows) % n_cls)
                         .astype(np.int32))
        wwd_w = mx.array(np.float32(0.1))
        for aux in (False, True):
            mx.reset_peak_memory()
            if aux:
                out = step_aux(wx, wtheta, wu, wthresh, wc,
                               wwd_x, wwd_t, wwd_w)
            else:
                out = step_noaux(wx, wtheta, wu, wthresh, wc)
            mx.eval(out, state)
            warm_peaks[f"{'aux' if aux else 'noaux'}@{length}"] = \
                mx.get_peak_memory()
            warmed.add((aux, length))
        del wx, wtheta, wu, wc, wwd_x
    assert len(warmed) == 6, f"expected 6 warmed signatures, got {warmed}"

    # restore the real state IN PLACE (containers captured by compile)
    model.update(p_snapshot)
    tree_assign_inplace(state[1], s_snapshot)
    mx.eval(state)
    hash_after = params_sha256(model)
    steps_after = (int(opt.optimizers[0].step.item()),
                   int(opt.optimizers[1].step.item()))
    assert hash_after == hash_before, (
        "warmup corrupted the real params: "
        f"{hash_before[:16]} -> {hash_after[:16]}")
    assert steps_after == steps_before, (
        f"warmup moved the schedule step counters: "
        f"{steps_before} -> {steps_after}")
    mx.clear_cache()
    gc.collect()
    warm_peak_max = max(warm_peaks.values())
    print(f"warmup: 6/6 signatures in {time.perf_counter()-t_warm:.1f}s; "
          f"params untouched ({hash_after[:16]}...), steps {steps_after}; "
          "peaks "
          + " ".join(f"{k}={v/1e9:.1f}GB" for k, v in warm_peaks.items()),
          flush=True)
    watchdog_limit = warm_peak_max + 4 * GB

    # ---------------- eval (protocol identical to torch evaluate()) --------
    def evaluate(rows_for_eval: list[dict]) -> dict:
        report: dict = {}
        conf_records: list[tuple[float, int]] = []
        chunk_n = 48
        for wname, length in DWELLS.items():
            mx.clear_cache()   # dwell-signature change inside eval
            # prototypes from the fp16 train cache (precision asymmetry:
            # exact mirror of torch's train_window path); window start comes
            # from the frozen per-row offset table (offset 0 under
            # --eval-offset-mode zero, where the slice path below is kept so
            # the legacy protocol is reproduced byte-for-byte)
            protos_list = []
            for cls in CLASSES:
                ids = train_by_class[cls][:64]
                z_parts = []
                for s0 in range(0, len(ids), chunk_n):
                    chunk = ids[s0:s0 + chunk_n]
                    slots_np = np.array([train_pos[r] for r in chunk],
                                        dtype=np.int32)
                    pad = np.concatenate(
                        [slots_np,
                         np.zeros(chunk_n - len(chunk), dtype=np.int32)])
                    g = mx.take(train_iq, mx.array(pad), axis=0)
                    if eval_zero:
                        seg = g[:, :, :length].astype(mx.float32)
                    else:
                        offs = np.concatenate(
                            [np.array([proto_offsets[r] for r in chunk],
                                      dtype=np.int64),
                             np.zeros(chunk_n - len(chunk), dtype=np.int64)])
                        idx = mx.array((offs[:, None, None]
                                        + np.arange(length)[None, None, :]
                                        ).astype(np.int32))
                        seg = mx.take_along_axis(g, idx, axis=2).astype(
                            mx.float32)
                    xb = seg[:, 0] + 1j * seg[:, 1]
                    zb, _ = eval_forward(xb)
                    z_parts.append(zb[:len(chunk)])
                zc_all = mx.concatenate(z_parts, axis=0)
                protos_list.append(mx.sum(zc_all, axis=0) / len(ids))
            protos = mx.stack(protos_list, axis=0)
            mx.eval(protos)

            per_class_correct = {c: 0 for c in CLASSES}
            per_class_total = {c: 0 for c in CLASSES}
            per_profile: dict[str, list[int]] = {}
            for s0 in range(0, len(rows_for_eval), chunk_n):
                chunk = rows_for_eval[s0:s0 + chunk_n]
                slots_np = np.array([eval_pos[r["row"]] for r in chunk],
                                    dtype=np.int32)
                pad = np.concatenate(
                    [slots_np,
                     np.zeros(chunk_n - len(chunk), dtype=np.int32)])
                xb_full = mx.take(eval_store, mx.array(pad), axis=0)
                if eval_zero:
                    xb = xb_full[:, :length]
                else:
                    offs = np.concatenate(
                        [np.array([eval_offsets[r["row"]] for r in chunk],
                                  dtype=np.int64),
                         np.zeros(chunk_n - len(chunk), dtype=np.int64)])
                    idx = mx.array((offs[:, None]
                                    + np.arange(length)[None, :]
                                    ).astype(np.int32))
                    xb = mx.take_along_axis(xb_full, idx, axis=1)
                zb, confb = eval_forward(xb)
                pred = mx.argmax(-cdist_sq(zb[:len(chunk)], protos), axis=1)
                pred_np = np.asarray(pred).tolist()
                conf_np = np.asarray(confb[:len(chunk)]).tolist()
                for r, p, cf in zip(chunk, pred_np, conf_np):
                    ok = CLASSES[p] == r["cls"]
                    per_class_total[r["cls"]] += 1
                    per_class_correct[r["cls"]] += ok
                    per_profile.setdefault(r["profile"], []).append(int(ok))
                    if wname == "1ms":
                        conf_records.append((cf, int(ok)))
            recalls = {c: per_class_correct[c] / max(1, per_class_total[c])
                       for c in CLASSES}
            profile_recalls = {p: sum(v) / len(v)
                               for p, v in per_profile.items()}
            report[wname] = {
                "balanced_accuracy": float(np.mean(list(recalls.values()))),
                "per_class": recalls,
                "min_profile_recall": min(profile_recalls.values()),
                "worst_profiles": sorted(profile_recalls.items(),
                                         key=lambda kv: kv[1])[:3],
            }
        answered = [ok for cf, ok in conf_records if cf > 0]  # strict > 0
        report["escalation"] = {
            "answered_frac": len(answered) / max(1, len(conf_records)),
            "answered_accuracy": (sum(answered) / len(answered))
            if answered else 0.0,
        }
        report["eval_protocol"] = {
            k: eval_plan[k] for k in
            ("offset_mode", "subsample_mode", "eval_seed", "plan_sha256")}
        return report

    # ---------------- training loop ----------------
    effective_stop = (min(cli.episodes, cli.stop_episode)
                      if cli.stop_episode else cli.episodes)
    dump_fh = open(cli.dump_draws, "w") if cli.dump_draws else None

    def rng_states_snapshot():
        return (copy.deepcopy(rng.bit_generator.state),
                copy.deepcopy(rng2.bit_generator.state))

    def do_save(path, episode, rstates):
        save_checkpoint(path, model, opt, cli, episode,
                        rstates[0], rstates[1], history)

    pace: dict[str, list[float]] = {}
    loss_trace: list[list[float]] = []
    skipped_steps = 0
    prev_length: int | None = None
    t0 = time.perf_counter()

    if start_episode < effective_stop:
        draw = draw_episode(start_episode, rng, rng2, cli, train_by_class,
                            all_train, row_samples, dwell_lengths, ramp_eps)
        if dump_fh:
            dump_draw(dump_fh, draw)
        batch = assemble(draw)

    final_rstates = rng_states_snapshot()
    for ep in range(start_episode, effective_stop):
        t_ep = time.perf_counter()
        if prev_length is not None and draw.length != prev_length:
            mx.clear_cache()               # dwell-signature change
        prev_length = draw.length

        sig = (draw.wd_ids is not None, draw.length)
        if sig not in warmed:
            raise SystemExit(
                f"7th signature would compile: {sig} not in warmed set "
                f"{sorted(warmed)} -- aborting per plan")
        # pre-step references for the non-finite skip path: mx arrays are
        # immutable, so flattening the trees captures the pre-update leaves
        # without copying anything (O(n_leaves) python per episode)
        prev_params = tree_flatten(model.trainable_parameters())
        prev_opt = tree_flatten(opt.state)
        if draw.wd_ids is not None:
            out = step_aux(*batch)
        else:
            out = step_noaux(*batch)
        mx.async_eval(out)

        # end-of-episode RNG capture: after this episode's draws, BEFORE the
        # next episode's (checkpoint semantics per plan section 3)
        final_rstates = rng_states_snapshot()

        # overlap: numpy assembly of the next episode while the GPU runs
        if ep + 1 < effective_stop:
            next_draw = draw_episode(ep + 1, rng, rng2, cli, train_by_class,
                                     all_train, row_samples, dwell_lengths,
                                     ramp_eps)
            if dump_fh:
                dump_draw(dump_fh, next_draw)
            next_batch = assemble(next_draw)
        else:
            next_draw, next_batch = None, None

        mx.eval(state)                     # the one per-episode sync
        vals = [float(v) for v in out]
        loss_trace.append([ep] + vals)
        ms = (time.perf_counter() - t_ep) * 1e3
        pace.setdefault(
            f"{'aux' if draw.wd_ids is not None else 'noaux'}@{draw.length}",
            []).append(ms)

        # mx.isfinite guard on the TOTAL loss (user directive): a non-finite
        # loss means the just-applied update is poison -- restore params,
        # moments AND step counters from the pre-step references (so the lr
        # schedule follows applied updates), count it, and continue.  The
        # episode's RNG draws already happened and stand, so the composition
        # stream stays bit-exact with an unbroken run.
        if not bool(mx.isfinite(out[0]).item()):
            skipped_steps += 1
            model.update(tree_unflatten(prev_params))
            tree_assign_inplace(state[1], tree_unflatten(prev_opt))
            mx.eval(state)
            print(f"WARNING ep {ep}: non-finite loss {vals} -- optimizer "
                  f"update skipped ({skipped_steps} total)", flush=True)

        if (ep + 1) % cli.log_every == 0:
            print(f"ep {ep+1}/{cli.episodes} loss={vals[0]:.4f} "
                  f"ce={vals[1]:.4f} recon={vals[2]:.4f} "
                  f"({(time.perf_counter()-t0)/(ep+1-start_episode)*1000:.0f}"
                  f" ms/ep)", flush=True)

        # watchdog every 50 eps: leak detector calibrated from warmup
        if (ep + 1) % 50 == 0:
            active = mx.get_active_memory()
            if active >= watchdog_limit:
                if cli.save_checkpoint:
                    do_save(rolling_path(cli.save_checkpoint), ep + 1,
                            final_rstates)
                raise SystemExit(
                    f"MEMORY WATCHDOG at ep {ep+1}: active "
                    f"{active/1e9:.2f} GB >= warmup peak "
                    f"{warm_peak_max/1e9:.2f} GB + 4 GB -- checkpoint "
                    "saved, aborting")

        if (ep + 1) % cli.eval_every == 0 or ep + 1 == cli.episodes:
            final_pass = ep + 1 == cli.episodes
            # mid-run passes use the frozen subsample (same rows at every
            # checkpoint of the run, so the eval history is a trajectory of
            # the model and not of the sampling)
            rows_for_eval = eval_rows if final_pass else mid_eval_rows
            mx.clear_cache()               # train -> eval transition
            t_ev = time.perf_counter()
            report = evaluate(rows_for_eval)
            report["eval_rows_used"] = len(rows_for_eval)
            history.append({"episode": ep + 1, **report})
            line = " | ".join(
                f"{w}: bal={report[w]['balanced_accuracy']:.3f} "
                f"minCell={report[w]['min_profile_recall']:.3f}"
                for w in DWELLS)
            print(f"[eval ep {ep+1}] {line} | esc@1ms: "
                  f"{report['escalation']['answered_frac']:.2f} answered "
                  f"@acc {report['escalation']['answered_accuracy']:.3f} "
                  f"({time.perf_counter()-t_ev:.1f}s, "
                  f"{len(rows_for_eval)} rows)", flush=True)
            if cli.save_checkpoint:        # rolling save at every eval
                do_save(rolling_path(cli.save_checkpoint), ep + 1,
                        final_rstates)
            mx.clear_cache()               # eval -> train transition
            gc.collect()
            prev_length = None
        elif cli.save_checkpoint and (ep + 1) % 500 == 0:
            do_save(rolling_path(cli.save_checkpoint), ep + 1, final_rstates)

        draw, batch = next_draw, next_batch

    episodes_done = min(effective_stop, cli.episodes)
    result = {
        "schema": "v7-trainer-v1",
        "corpus": str(CORPUS),
        "config": vars(cli),
        "eval_plan": eval_plan,
        "params": n_params,
        "wall_s": time.perf_counter() - t0,
        "history": history,
        "final": history[-1] if history else None,
        "skipped_steps": skipped_steps,
        "mlx_diagnostics": {
            "init_provenance": provenance,
            "precision": {
                "dtype": cli.dtype,
                "tf32_requested": not cli.no_tf32,
                "tf32_active": tf32_active,
                "tf32_env": os.environ.get("MLX_ENABLE_TF32"),
                "matmul_rel_err_vs_fp64": tf32_rel,
            },
            "ckpt_blocks": {
                "cli": cli.ckpt_blocks,
                "resolved_per_dwell": {k: ckpt_for(v)
                                       for k, v in DWELLS.items()},
            },
            "batch_chunks": {
                "cli": cli.batch_chunks,
                "resolved_per_dwell": {k: chunks_for(v)
                                       for k, v in DWELLS.items()},
            },
            "mem_limit_gb_over_corpus": cli.mem_limit_gb,
            "episodes_run": [start_episode, episodes_done],
            "warmup_peaks_bytes": warm_peaks,
            "watchdog_limit_bytes": watchdog_limit,
            "peak_memory_bytes": mx.get_peak_memory(),
            "active_memory_bytes_end": mx.get_active_memory(),
            "pace_ms_per_ep": {
                k: {"count": len(v),
                    "mean": float(np.mean(v)),
                    "min": float(np.min(v)),
                    "max": float(np.max(v))}
                for k, v in sorted(pace.items())},
            "loss_trace": loss_trace,
            "boot_s": t0 - t_boot,
        },
    }
    _atomic_write_bytes(
        cli.out, lambda tmp: Path(tmp).write_text(json.dumps(result,
                                                             indent=2)))
    if cli.save_checkpoint:
        do_save(cli.save_checkpoint, episodes_done, final_rstates)
        print(f"checkpoint {cli.save_checkpoint} (+sidecar .json) at ep "
              f"{episodes_done}", flush=True)
    if dump_fh:
        dump_fh.close()
    print(f"wrote {cli.out}", flush=True)


if __name__ == "__main__":
    main()
