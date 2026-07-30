#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zhouchongtian/quantization/qvdit_flash_bf16
BASE="$ROOT/logs_fp16_flash/stage_numeric_profiling"

source /home/zhouchongtian/miniconda3/etc/profile.d/conda.sh
conda activate qvdit
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/t2v"
export CUDA_VISIBLE_DEVICES=6

for prompt_index in 0 2 6; do
  output="$BASE/i_cfg_stage_prompt${prompt_index}_seed42_gpu6"
  mkdir -p "$output"
  python tools/profile_cfg_stage.py \
    --config t2v/configs/quant/opensora/16x512x512_fp16_flash_100steps.py \
    --calib-config t2v/configs/quant/opensora/w4a6_baseline.yaml \
    --quant-ckpt logs_fp16_flash/formal_w4a6_baseline_fp16_gradscaler_samples10_gpu5_0726/calibration/ckpt.pth \
    --text-embeds /home/zhouchongtian/quantization/new/logs_50steps/text_embeds_samples_10.pt \
    --prompt-index "$prompt_index" \
    --seed 42 \
    --selected-progress 5,15,25,45,65,85,95 \
    --time-mp-config-weight t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml \
    --time-mp-config-act t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml \
    --output-dir "logs_fp16_flash/stage_numeric_profiling/i_cfg_stage_prompt${prompt_index}_seed42_gpu6" \
    2>&1 | tee "$output/run.log"
  test -s "$output/metadata.json"
done
