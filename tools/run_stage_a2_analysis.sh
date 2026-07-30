#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zhouchongtian/quantization/qvdit_flash_bf16/logs_fp16_flash/stage_profiling
OUT="$ROOT/a2_aggregate_10prompts_2seeds"
LOG="$OUT/run.log"

while tmux list-sessions -F '#S' 2>/dev/null | grep -qx 'stageprofile_a2'; do
  sleep 30
done

completed=$(find "$ROOT" -mindepth 2 -maxdepth 2 -path '*/fp16_fp_trajectory_prompt*_seed*/config.json' -type f \
  -exec grep -l '"status": "completed"' {} \; | wc -l)
if [[ "$completed" -ne 20 ]]; then
  echo "Expected 20 completed A2 runs, found $completed; analysis not started." >&2
  exit 2
fi

mkdir -p "$OUT"
source /home/zhouchongtian/miniconda3/etc/profile.d/conda.sh
export PYTHONPATH="${PYTHONPATH:-}"
conda activate vbench
export CUDA_VISIBLE_DEVICES=6

cd /home/zhouchongtian/quantization/qvdit_flash_bf16
python tools/analyze_stage_a2.py \
  --root "$ROOT" \
  --prompt-path t2v/assets/texts/t2v_samples_10.txt \
  --output-dir "$OUT" \
  >"$LOG" 2>&1
