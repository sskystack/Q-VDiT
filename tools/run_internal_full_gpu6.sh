#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zhouchongtian/quantization/qvdit_flash_bf16
SMOKE="$ROOT/logs_fp16_flash/stage_numeric_profiling/g_internal_smoke_p95_b0_gpu6"
OUTPUT="$ROOT/logs_fp16_flash/stage_numeric_profiling/g_internal_prompt0_seed42_gpu6"

python - "$SMOKE/metadata.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    metadata = json.load(handle)
assert metadata["all_values_finite"]
assert metadata["operator_rows"] > 0
assert metadata["attention_rows"] > 0
assert metadata["attention_kl_rows"] > 0
PY

source /home/zhouchongtian/miniconda3/etc/profile.d/conda.sh
conda activate qvdit
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/t2v"
export CUDA_VISIBLE_DEVICES=6
mkdir -p "$OUTPUT"

python tools/profile_internal_mechanisms.py \
  --config t2v/configs/quant/opensora/16x512x512_fp16_flash_100steps.py \
  --calib-config t2v/configs/quant/opensora/w4a6_baseline.yaml \
  --quant-ckpt logs_fp16_flash/formal_w4a6_baseline_fp16_gradscaler_samples10_gpu5_0726/calibration/ckpt.pth \
  --prompt-path t2v/assets/texts/t2v_samples_10.txt \
  --text-embeds /home/zhouchongtian/quantization/new/logs_50steps/text_embeds_samples_10.pt \
  --prompt-index 0 \
  --seed 42 \
  --selected-progress 5,15,25,45,65,85,95 \
  --selected-blocks 0,9,17,27 \
  --attention-query-samples 32 \
  --attention-batch-samples 2 \
  --attention-head-samples 2 \
  --tqe-token-samples 128 \
  --time-mp-config-weight t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml \
  --time-mp-config-act t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml \
  --output-dir logs_fp16_flash/stage_numeric_profiling/g_internal_prompt0_seed42_gpu6 \
  2>&1 | tee "$OUTPUT/run.log"

test -s "$OUTPUT/metadata.json"
