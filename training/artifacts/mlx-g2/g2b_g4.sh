#!/bin/zsh
# G2b in bf16 (winner — user-confirmed; accuracy anchored to MLX fp32-strict,
# torch baseline waived by user), then G4: full 3000-ep replication in bf16.
SP=/private/tmp/claude-501/-Users-johnelliott-PersonalGitHub/6179ba93-fb7a-4112-b70b-19540b0f4fce/scratchpad
DRV=$SP/run_mlx_g2.sh
echo "[pipeline] WINNER config: c (bf16; anchor=mlx-fp32, torch baseline waived)"

echo "[pipeline] === G2b 200-stop start $(date +%H:%M:%S) ==="
"$DRV" g2b200 --dtype bf16 --stop-episode 200 \
  --save-checkpoint "$SP/g2b_ck200.safetensors" > "$SP/g2b_200.log" 2>&1
echo "[pipeline] === G2b 200-stop exit $? ==="
echo "[pipeline] === G2b resume start $(date +%H:%M:%S) ==="
"$DRV" g2bres --dtype bf16 \
  --init-checkpoint "$SP/g2b_ck200.safetensors" > "$SP/g2b_resume.log" 2>&1
echo "[pipeline] === G2b resume exit $? ==="
cat "$SP/draws_mlx_g2b200.jsonl" "$SP/draws_mlx_g2bres.jsonl" > "$SP/draws_g2b_stitched.jsonl"
if cmp -s "$SP/draws_mlx_c.jsonl" "$SP/draws_g2b_stitched.jsonl"; then
  echo "[pipeline] G2B DRAWS straight-vs-(200+resume+100): BYTE-EXACT"
else
  echo "[pipeline] G2B DRAWS: MISMATCH"
fi

echo "[pipeline] === G4 3000-ep bf16 start $(date +%H:%M:%S) ==="
PY=/Users/johnelliott/PersonalGitHub/Atom-Classifier/.venv-training/bin/python
TR=/Users/johnelliott/PersonalGitHub/Atom-Classifier/training/zplane_ab/v2_full_variation/v7_probe/v7_trainer_mlx.py
caffeinate -is "$PY" "$TR" \
  --episodes 3000 --eval-every 1000 --recon-weight 0.3 --seed 20260740 \
  --dtype bf16 --ckpt-blocks 3 --batch-chunks 5 --log-every 100 \
  --init-checkpoint "$SP/step0.safetensors" \
  --save-checkpoint "$SP/g4_bf16.safetensors" \
  --out "$SP/g4_bf16.json" > "$SP/g4_bf16.log" 2>&1
echo "[pipeline] === G4 exit $? $(date +%H:%M:%S) ==="
echo "[pipeline] ALL DONE $(date +%H:%M:%S)"
