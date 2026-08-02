# Status

Single living status file. Deep history lives in git (`HANDOFF.md`,
`HANDOFF-HISTORY.md`, `HANDOFF-v3-release.md` were removed in the 2026-08-02
streamline commit — check them out from history if you need the forensics).

## Current state (2026-08-02)

Two lines exist; do not confuse them:

| line | state |
|---|---|
| **v3/v4 browser runtime** (shipped in Atomizer) | frozen; assets under `src/embedding/assets*` and sealed release records under `training/artifacts/releases/` |
| **v7 / DACS** (dwell-adaptive I/Q classifier, MLX) | trained on corrected corpus v2; runtime exportable via `tools/export-dacs-v7-runtime.py` |

- Production corpus: `training/artifacts/longdwell-production-corpus-v2/`
  — 34 profiles × 256 rows, 8,704 × 20 ms rows at 20 Msps, clean/noisy
  pairs, verifier passes all rows. Natural BLE-advertising silence retained
  (88/256 rows silent) — deliberate: silence is class-conditional evidence.
- Production checkpoint: `training/artifacts/sota-v2-2014019-12e7eef-20260801/content_v2_bf16.safetensors`
  (SHA-256 `1d75a01e6a68936a7facaf4f8bd32c93cedfd46fc98d0cecdc5ecdacd640b281`).
- The alternative I/Q-BF16/chunk-2 run (`sota-v2-iqbf16-chunk2-...`) is
  slightly stronger at 10 ms but weaker at 1 / 2.5 ms; it does not dominate.

### Honest numbers (five-seed mean, full 3,264-row split, corrected protocol)

| dwell | balanced accuracy | min-profile recall |
|---|---|---|
| 1 ms | 0.9031 | 0.2583 |
| 2.5 ms | 0.9549 | 0.6542 |
| 10 ms | 0.9939 | 0.9625 |

Classes: `am, bluetooth, cw, dsss, fm, gsm, ofdm` (34 profiles). The 1 ms
weakness is GSM — the burst is genuinely absent in most sub-millisecond
windows; errors nest nearly perfectly across dwells, so escalation is
almost pure gain. The confidence head is **uncalibrated**; it is telemetry,
not a gate.

## Standing constraints

- **Never write into `src/embedding/assets/`** — that is the live browser
  model Atomizer imports. Atomizer also imports `src/bayesian-waveform-classifier.ts`,
  `src/signal-lab-classifier.ts`, `src/observable-classifier-model.ts`,
  `src/models/`, and the `assets-v3-*` directories (tests/packaging), so
  treat all of `src/` as API surface.
- `training/artifacts/` is git-ignored: a clean worktree does **not** mean
  checkpoints/corpora are durable. Copy anything release-worthy out or
  package it before deleting.
- Train in **bf16, never fp16** (fp16 never trains — non-finite losses).
  TF32 stays ON for production (owner directive); parity harnesses must set
  `MLX_ENABLE_TF32=0`.
- Full-corpus runs need `--ckpt-blocks 3 --batch-chunks 5` (exact math,
  25 GB peak instead of 59 GB).
- A checkpoint alone is not a release: DACS needs frozen prototypes, ONNX
  encoder export, exact preprocessing, and Atomizer's upstream open-set
  gate. `tools/export-dacs-v7-runtime.py` packages all of that fail-closed
  (requires clean worktrees). See `docs/dacs-v7-runtime.md`.

## How to run

```bash
# Production training (MLX bf16, ~53 min / 3000 episodes on M5 Max 48 GB)
.venv-training/bin/python \
  training/zplane_ab/v2_full_variation/v7_probe/v7_trainer_mlx.py \
  --episodes 3000 --eval-every 1000 --recon-weight 0.3 --seed 20260740 \
  --dtype bf16 --ckpt-blocks 3 --batch-chunks 5 --log-every 100 \
  --init-checkpoint training/artifacts/mlx-g2/step0.safetensors \
  --save-checkpoint <out>.safetensors --out <out>.json

# Re-evaluate any checkpoint under the corrected 5-seed protocol
.venv-training/bin/python training/artifacts/mlx-g2/g5_offset_reeval.py

# Package a runtime release
npm run package:dacs-v7-runtime -- --output <empty-dir>

# Corpus regeneration (if ever needed): docs/corpus-regen-runbook.md
```

Environment: `.venv-training` (Python 3.14, torch 2.13.0, mlx 0.32.0),
node at `~/.local/node/bin`. Machine: Apple M5 Max, 48 GB.

## Testing

See `docs/TESTING.md` for the held-out evaluation protocol and the
over-the-air Neptune smoke test.

## Open items

- Confidence-head calibration on held-out data (prerequisite for claiming
  the escalation curve, not just the mechanism).
- G5: port `eval_confusion.py` to MLX; wire Optuna to the MLX trainer.
- Paper rewrite with corrected numbers (deprioritized).
- Upstream reports: PyTorch-MPS post-eval deadlock; MLX TF32 default
  discoverability.
