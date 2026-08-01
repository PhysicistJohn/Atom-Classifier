#!/bin/zsh
# G4: full 3000-episode replication, bf16 (G2 winner), on a fresh-boot machine.
# G2b (resume parity) already PASSED byte-exact pre-reboot.
# Anchor for the accuracy gate: torch A/B measured evals at ep1000
# (bal 0.951/0.966/0.979) and ep2000 (0.971/0.979/0.967).
set -u
D=/Users/johnelliott/PersonalGitHub/Atom-Classifier/training/artifacts/mlx-g2
PY=/Users/johnelliott/PersonalGitHub/Atom-Classifier/.venv-training/bin/python
TR=/Users/johnelliott/PersonalGitHub/Atom-Classifier/training/zplane_ab/v2_full_variation/v7_probe/v7_trainer_mlx.py

echo "[g4] start $(date '+%F %H:%M:%S') — bf16, fresh boot, swap 0"
caffeinate -is "$PY" "$TR" \
  --episodes 3000 --eval-every 1000 --recon-weight 0.3 --seed 20260740 \
  --dtype bf16 --ckpt-blocks 3 --batch-chunks 5 --log-every 100 \
  --init-checkpoint "$D/step0.safetensors" \
  --save-checkpoint "$D/g4_bf16.safetensors" \
  --out "$D/g4_bf16.json" >> "$D/g4_bf16.log" 2>&1
echo "[g4] exit $? $(date '+%H:%M:%S')"
echo "[g4] ALL DONE"
