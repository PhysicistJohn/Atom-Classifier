#!/bin/bash
# Launch the v6 study across N worker processes.
#
# Measured on this M5 Max: a single worker runs ~26.7 ms/episode, and six
# concurrent workers run ~49.2 ms/episode each -- 1.84x slower per worker, so
# aggregate throughput is ~3.25x, not 6x.  Cores show ~99% busy either way, so
# %cpu is not a usable signal here; only end-to-end throughput is.  Worker count
# should therefore be chosen from measured aggregate throughput and free memory
# (~3.13 GB resident per worker), not from core count.
#
# Usage: v6_launch_parallel.sh <output-dir> <n-workers> <search-hours> [device]
set -uo pipefail

ROOT="/Users/johnelliott/PersonalGitHub/Atom-Classifier"
PY="$ROOT/.venv-training/bin/python"
STUDY="$ROOT/training/zplane_ab/v2_full_variation/v5_scale_orbit/v6_optuna_study.py"
CURRENT="training/artifacts/signallab-current-scale-train-v5-seed20264101-r64"
SELECTION="training/artifacts/signallab-current-scale-dev-v5-seed20262904-r64-identity-firewalled"

OUT="${1:?output dir required}"
WORKERS="${2:-6}"
HOURS="${3:-7.5}"
DEVICE="${4:-auto}"

cd "$ROOT" || exit 1
mkdir -p "$OUT"
JOURNAL="$OUT/journal.log"

echo "[launch] output=$OUT workers=$WORKERS hours=$HOURS device=$DEVICE"
echo "[launch] journal=$JOURNAL"

# Controls run in their own process, concurrently with the search workers.
# They are reference points, not part of the search, so they must not consume
# a search worker slot.
nohup env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  "$PY" "$STUDY" \
  --current-corpus "$CURRENT" --selection-corpus "$SELECTION" \
  --output-dir "$OUT" --role controls --device "$DEVICE" \
  > "$OUT/controls.log" 2>&1 &
echo "[launch] controls pid=$!"

# Search workers.  Each gets a DISTINCT sampler seed: workers sharing one seed
# draw identical startup trials and waste the budget on duplicates.
for i in $(seq 1 "$WORKERS"); do
  nohup env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    "$PY" "$STUDY" \
    --current-corpus "$CURRENT" --selection-corpus "$SELECTION" \
    --output-dir "$OUT" --role search --storage "$JOURNAL" \
    --sampler-seed "$((7000 + i))" --search-hours "$HOURS" \
    --device "$DEVICE" --skip-controls \
    > "$OUT/worker_$i.log" 2>&1 &
  echo "[launch] worker $i pid=$!"
done

echo "[launch] all processes started"
