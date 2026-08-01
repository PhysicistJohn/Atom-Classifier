#!/bin/zsh
# Reordered G2: MLX configs first (the numbers that matter), torch baseline
# demoted to last (it crawls at >=12 s/ep on tonight's machine state).
# Logic copied faithfully from the agent's g2_pipeline.sh.
SP=/private/tmp/claude-501/-Users-johnelliott-PersonalGitHub/6179ba93-fb7a-4112-b70b-19540b0f4fce/scratchpad
PY=/Users/johnelliott/PersonalGitHub/Atom-Classifier/.venv-training/bin/python
DRV=$SP/run_mlx_g2.sh
TORCH_TR=/Users/johnelliott/PersonalGitHub/Atom-Classifier/training/zplane_ab/v2_full_variation/v7_probe/v7_trainer.py
cd "$SP"

run_cfg() {
  local tag=$1; shift
  echo "[pipeline] === MLX config $tag start $(date +%H:%M:%S) ==="
  "$DRV" "$tag" "$@" > "$SP/g2_mlx_$tag.log" 2>&1
  echo "[pipeline] === MLX config $tag exit $? $(date +%H:%M:%S) ==="
  return 0
}

run_cfg a --no-tf32

head -50 "$SP/draws_mlx_a.jsonl" > "$SP/draws_mlx_a_head50.jsonl"
if cmp -s "$SP/draws_torch50.jsonl" "$SP/draws_mlx_a_head50.jsonl"; then
  echo "[pipeline] DRAWS torch-vs-mlx(a) first50: BYTE-EXACT"
else
  echo "[pipeline] DRAWS torch-vs-mlx(a) first50: MISMATCH"
fi

run_cfg b
run_cfg c --dtype bf16
run_cfg d --dtype fp16

echo "[pipeline] === torch baseline (demoted) start $(date +%H:%M:%S) ==="
caffeinate -is "$PY" "$TORCH_TR" --episodes 300 --eval-every 300 \
  --recon-weight 0.3 --seed 20260740 \
  --out "$SP/g2_torch300.json" >> "$SP/g2_torch300.log" 2>&1
echo "[pipeline] === torch baseline exit $? $(date +%H:%M:%S) ==="

WINNER=$("$PY" - <<'EOF'
import json
from pathlib import Path
SP = Path("/private/tmp/claude-501/-Users-johnelliott-PersonalGitHub/6179ba93-fb7a-4112-b70b-19540b0f4fce/scratchpad")
DW = ("1ms", "2.5ms", "10ms")
torch_res = json.loads((SP/"g2_torch300.json").read_text())
tbal = {w: torch_res["final"][w]["balanced_accuracy"] for w in DW}
best, best_pace = None, None
for tag in ("a", "b", "c", "d"):
    p = SP/f"g2_mlx_{tag}.json"
    if not p.exists():
        continue
    r = json.loads(p.read_text())
    if r.get("skipped_steps", 0) != 0:
        continue
    bal = {w: r["final"][w]["balanced_accuracy"] for w in DW}
    if any(abs(bal[w]-tbal[w]) > 0.02 for w in DW):
        continue
    pace = r["mlx_diagnostics"]["pace_ms_per_ep"]
    tot = sum(v["count"]*v["mean"] for v in pace.values())
    n = sum(v["count"] for v in pace.values())
    mean = tot/max(1, n)
    if best is None or mean < best_pace:
        best, best_pace = tag, mean
print(best or "NONE")
EOF
)
echo "[pipeline] WINNER config: $WINNER"
if [ "$WINNER" = "NONE" ]; then
  echo "[pipeline] no config met the gate; skipping G2b"; exit 0
fi
case $WINNER in
  a) WARGS=(--no-tf32);;
  b) WARGS=();;
  c) WARGS=(--dtype bf16);;
  d) WARGS=(--dtype fp16);;
esac

echo "[pipeline] === G2b 200-stop start $(date +%H:%M:%S) ==="
"$DRV" g2b200 "${WARGS[@]}" \
  --stop-episode 200 \
  --save-checkpoint "$SP/g2b_ck200.safetensors" \
  > "$SP/g2b_200.log" 2>&1
echo "[pipeline] === G2b 200-stop exit $? ==="

echo "[pipeline] === G2b resume start $(date +%H:%M:%S) ==="
"$DRV" g2bres "${WARGS[@]}" \
  --init-checkpoint "$SP/g2b_ck200.safetensors" \
  > "$SP/g2b_resume.log" 2>&1
echo "[pipeline] === G2b resume exit $? ==="

cat "$SP/draws_mlx_g2b200.jsonl" "$SP/draws_mlx_g2bres.jsonl" \
  > "$SP/draws_g2b_stitched.jsonl"
if cmp -s "$SP/draws_mlx_$WINNER.jsonl" "$SP/draws_g2b_stitched.jsonl"; then
  echo "[pipeline] G2B DRAWS straight-vs-(200+resume+100): BYTE-EXACT"
else
  echo "[pipeline] G2B DRAWS: MISMATCH"
fi
"$PY" "$SP/summarize_g2.py" > "$SP/g2_summary.txt" 2>&1 || true
echo "[pipeline] ALL DONE $(date +%H:%M:%S)"
