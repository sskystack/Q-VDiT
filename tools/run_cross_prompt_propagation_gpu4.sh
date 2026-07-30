#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zhouchongtian/quantization/qvdit_flash_bf16
PYTHON=/home/zhouchongtian/miniconda3/envs/qvdit/bin/python
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/t2v"
export CUDA_VISIBLE_DEVICES=4

for PROMPT_INDEX in 2 6; do
  OUTDIR="$ROOT/logs_fp16_flash/stage_numeric_profiling/e_equal_energy_prompt${PROMPT_INDEX}_seed42"
  mkdir -p "$OUTDIR"
  echo "$(date -Is) starting prompt ${PROMPT_INDEX}"
  "$PYTHON" tools/profile_equal_energy_propagation.py \
    --config t2v/configs/quant/opensora/16x512x512_fp16_flash_100steps.py \
    --calib-config t2v/configs/quant/opensora/w4a6_baseline.yaml \
    --quant-ckpt logs_fp16_flash/formal_w4a6_baseline_fp16_gradscaler_samples10_gpu5_0726/calibration/ckpt.pth \
    --prompt-path t2v/assets/texts/t2v_samples_10.txt \
    --text-embeds /home/zhouchongtian/quantization/new/logs_50steps/text_embeds_samples_10.pt \
    --prompt-index "$PROMPT_INDEX" \
    --seed 42 \
    --selected-progress 5,15,25,45,65,85,95 \
    --relative-energy 0.01 \
    --directions random,w4a6 \
    --time-mp-config-weight t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml \
    --time-mp-config-act t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml \
    --output-dir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/run.log"
  echo "$(date -Is) finished prompt ${PROMPT_INDEX}"
done
