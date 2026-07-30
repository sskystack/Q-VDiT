#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zhouchongtian/quantization/qvdit_flash_bf16
OUT="$ROOT/logs_fp16_flash/stage_numeric_profiling/k_single_layer_sensitivity_gpu6"

source /home/zhouchongtian/miniconda3/etc/profile.d/conda.sh
conda activate qvdit
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/t2v"
export CUDA_VISIBLE_DEVICES=6
mkdir -p "$OUT"

python tools/profile_single_layer_sensitivity.py \
  --config t2v/configs/quant/opensora/16x512x512_fp16_flash_100steps.py \
  --calib-config t2v/configs/quant/opensora/w4a6_baseline.yaml \
  --quant-ckpt logs_fp16_flash/formal_w4a6_baseline_fp16_gradscaler_samples10_gpu5_0726/calibration/ckpt.pth \
  --text-embeds /home/zhouchongtian/quantization/new/logs_50steps/text_embeds_samples_10.pt \
  --prompt-indices 0,2,6 \
  --seed 42 \
  --selected-progress 5,15,25 \
  --selected-blocks 0,4,9,13,17,21,25,27 \
  --time-mp-config-weight t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml \
  --time-mp-config-act t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml \
  --output-dir "$OUT" \
  2>&1 | tee "$OUT/run.log"

test -s "$OUT/metadata.json"
python tools/analyze_single_layer_sensitivity.py \
  --input "$OUT/single_layer_sensitivity.jsonl" \
  --output-dir "$OUT/analysis" \
  2>&1 | tee "$OUT/analysis.log"
test -s "$OUT/analysis/single_layer_summary.json"
echo complete > "$OUT/status.txt"
