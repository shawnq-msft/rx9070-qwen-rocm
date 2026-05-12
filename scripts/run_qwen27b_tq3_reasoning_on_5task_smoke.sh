#!/usr/bin/env bash
# Run the standard 5-task PinchBench smoke against Qwen3.6-27B-TQ3_4S reasoning-on profile,
# sleeping 30s between each task to let the card cool / allocator settle.
set -euo pipefail

export PATH="/home/qiushuo/.hermes/node/bin:$PATH"
unset PINCHBENCH_FORCE_DIRECT_LLAMACPP
unset PINCHBENCH_FORCE_LOCAL_AGENT

PINCHBENCH_DIR="${PINCHBENCH_DIR:-/home/qiushuo/src/pinchbench-skill/scripts}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8250/v1}"
MODEL_ID="${MODEL_ID:-llama-cpp/qwen3-6-27b-tq3_4s-gguf}"
OUT_ROOT="${OUT_ROOT:-/home/qiushuo/reports/pinchbench/qwen36-27b-tq3-4s-5task-reasoning-on-30srest}"
SLEEP_SECS="${SLEEP_SECS:-30}"
TIMEOUT_MULTIPLIER="${TIMEOUT_MULTIPLIER:-5}"

TASKS=(
  task_calendar
  task_summary
  task_weather
  task_csv_iris_summary
  task_sanity
)

mkdir -p "$OUT_ROOT"
cd "$PINCHBENCH_DIR"

echo "[info] base_url=$BASE_URL model=$MODEL_ID out_root=$OUT_ROOT sleep=${SLEEP_SECS}s"

for i in "${!TASKS[@]}"; do
  task="${TASKS[$i]}"
  task_idx=$((i+1))
  ts="$(date +%Y%m%d-%H%M%S)"
  out_dir="$OUT_ROOT/${task_idx}_${task}_${ts}"
  mkdir -p "$out_dir"

  echo "[info] running ${task_idx}/${#TASKS[@]} task=$task out_dir=$out_dir"
  python3 benchmark.py \
    --model "$MODEL_ID" \
    --base-url "$BASE_URL" \
    --suite "$task" \
    --timeout-multiplier "$TIMEOUT_MULTIPLIER" \
    --no-upload \
    --no-fail-fast \
    --verbose \
    --output-dir "$out_dir"

  if [ "$task_idx" -lt "${#TASKS[@]}" ]; then
    echo "[info] sleeping ${SLEEP_SECS}s before next task"
    sleep "$SLEEP_SECS"
  fi
done

echo "[done] finished 5-task smoke with per-task ${SLEEP_SECS}s rest: $OUT_ROOT"
