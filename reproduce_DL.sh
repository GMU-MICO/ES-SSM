#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python}"
HDF5_DIR="${HDF5_DIR:-./data}"
CACHE_DIR="${CACHE_DIR:-./cache_filters}"
PROJECT_ROOT="${PROJECT_ROOT:-./project}"
RESULTS_DIR="${RESULTS_DIR:-./results}"
SEEDS="${SEEDS:-42,43,44}"
NUM_STEPS="${NUM_STEPS:-150000}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-100}"
SLOTS_PER_GPU="${SLOTS_PER_GPU:-3}"

"$PY" "$SCRIPT_DIR/d4rl_mujoco_RL.py" download-data --hdf5-dir "$HDF5_DIR"
"$PY" "$SCRIPT_DIR/d4rl_mujoco_RL.py" sweep-train --project-root "$PROJECT_ROOT" \
    --hdf5-dir "$HDF5_DIR" --cache-dir "$CACHE_DIR" --seeds "$SEEDS" \
    --num-steps "$NUM_STEPS" --slots-per-gpu "$SLOTS_PER_GPU"
"$PY" "$SCRIPT_DIR/d4rl_mujoco_RL.py" sweep-export --project-root "$PROJECT_ROOT" \
    --cache-dir "$CACHE_DIR" --seeds "$SEEDS" --Ks 2,4,8,16,32
"$PY" "$SCRIPT_DIR/d4rl_mujoco_RL.py" sweep-eval --project-root "$PROJECT_ROOT" \
    --cache-dir "$CACHE_DIR" --seeds "$SEEDS" \
    --num-eval-episodes "$NUM_EVAL_EPISODES" --slots-per-gpu "$SLOTS_PER_GPU"
"$PY" "$SCRIPT_DIR/d4rl_mujoco_RL.py" tabulate --project-root "$PROJECT_ROOT" \
    --seeds "$SEEDS" --out-dir "$RESULTS_DIR"
echo "Results: $RESULTS_DIR"
