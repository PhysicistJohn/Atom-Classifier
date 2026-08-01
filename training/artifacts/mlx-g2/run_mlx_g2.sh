#!/bin/zsh
# G2 MLX run driver: one config per invocation, serial use only.
# Usage: run_mlx_g2.sh <tag> [extra trainer args...]
# Memory config: explicit --ckpt-blocks 3 --batch-chunks 5 on ALL dwells
# (CLI-only extension of the auto rules; the auto rules leave the
# 2.5ms/50000 signature unchunked at ~32 GB activations, which swaps
# against the ~20.2 GB resident full corpus. Chunking is mathematically
# exact per plan sec 4; probe showed loss bit-identical across memory
# configs.)
set -e
SP=/private/tmp/claude-501/-Users-johnelliott-PersonalGitHub/6179ba93-fb7a-4112-b70b-19540b0f4fce/scratchpad
PY=/Users/johnelliott/PersonalGitHub/Atom-Classifier/.venv-training/bin/python
TR=/Users/johnelliott/PersonalGitHub/Atom-Classifier/training/zplane_ab/v2_full_variation/v7_probe/v7_trainer_mlx.py
TAG=$1; shift
exec /usr/bin/time -l "$PY" "$TR" \
  --episodes 300 --eval-every 300 --recon-weight 0.3 --seed 20260740 \
  --init-checkpoint "$SP/step0.safetensors" \
  --ckpt-blocks 3 --batch-chunks 5 \
  --log-every 50 \
  --dump-draws "$SP/draws_mlx_$TAG.jsonl" \
  --out "$SP/g2_mlx_$TAG.json" \
  "$@"
