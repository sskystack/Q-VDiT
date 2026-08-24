#!/usr/bin/env bash
set -euo pipefail

source /home/zhouchongtian/miniconda3/etc/profile.d/conda.sh
conda activate qvdit

ROOT=${ROOT:-/home/zhouchongtian/quantization/qvdit_adaptive_mtd_calibration_v1}
DATA_ROOT=${DATA_ROOT:-/home/zhouchongtian/quantization/qvdit_flash_bf16}
CALIB_ROOT=${CALIB_ROOT:-$DATA_ROOT/logs_fp16_flash/adaptive_mtd_calibration_v1_200}
HOLDOUT_ROOT=${HOLDOUT_ROOT:-$DATA_ROOT/logs_fp16_flash/adaptive_mtd_holdout_v1}
OUT=${OUT:-$DATA_ROOT/logs_fp16_flash/adaptive_mtd_same_latent_v1}
GPU_INDEX=${GPU_INDEX:-6}

mkdir -p "$OUT"
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/t2v"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
echo $$ > "$OUT/runner.pid"
echo "RUNNING gpu=$GPU_INDEX $(date '+%F %T')" > "$OUT/status.txt"

on_error() {
  code=$?
  echo "FAILED gpu=$GPU_INDEX exit=$code $(date '+%F %T')" > "$OUT/status.txt"
  exit "$code"
}
trap on_error ERR

python tools/profile_adaptive_mtd_same_latent.py \
  --config "$ROOT/t2v/configs/quant/opensora/16x512x512_fp16_flash_100steps.py" \
  --baseline-calib-config "$ROOT/t2v/configs/quant/opensora/w4a6_mtd_adaptive_v1_fixed_200.yaml" \
  --candidate-calib-config "$ROOT/t2v/configs/quant/opensora/w4a6_mtd_adaptive_v1_centered_200.yaml" \
  --baseline-ckpt "$CALIB_ROOT/fixed/calibration/ckpt.pth" \
  --candidate-ckpt "$CALIB_ROOT/centered/calibration/ckpt.pth" \
  --baseline-features "$HOLDOUT_ROOT/fixed/features" \
  --candidate-features "$HOLDOUT_ROOT/centered/features" \
  --text-embeds /home/zhouchongtian/quantization/Q-VDiT/t2v/utils_files/text_embeds.pth \
  --time-mp-config-weight "$ROOT/t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml" \
  --time-mp-config-act "$ROOT/t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml" \
  --output-dir "$OUT/results" \
  --seed 42 --transport-size 16 --temperature 0.07 \
  2>&1 | tee "$OUT/run.log"

echo "COMPLETED gpu=$GPU_INDEX $(date '+%F %T')" > "$OUT/status.txt"
