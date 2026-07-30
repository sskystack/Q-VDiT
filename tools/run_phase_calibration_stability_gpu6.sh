#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zhouchongtian/quantization/qvdit_flash_bf16
PHASE_DATA="$ROOT/logs_fp16_flash/stage_numeric_profiling/calibration_phase_subsets"
PROMPT_RESULTS="$ROOT/logs_fp16_flash/stage_numeric_profiling/h_calibration_stability_gpu6"
RESULTS="$ROOT/logs_fp16_flash/stage_numeric_profiling/j_phase_calibration_stability_gpu6"
MODEL_CKPT=/home/zhouchongtian/quantization/Q-VDiT/logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth
TEXT_EMBEDS=/home/zhouchongtian/quantization/new/logs_50steps/text_embeds_samples_10.pt

source /home/zhouchongtian/miniconda3/etc/profile.d/conda.sh
conda activate qvdit
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/t2v"
export CUDA_VISIBLE_DEVICES=6
mkdir -p "$RESULTS"

for scheme in early_heavy late_heavy; do
  outdir="$RESULTS/$scheme/calibration"
  mkdir -p "$outdir"
  if [[ ! -s "$outdir/ckpt.pth" ]]; then
    date -Is > "$RESULTS/$scheme/start_time.txt"
    python t2v/scripts/calib.py \
      t2v/configs/quant/opensora/16x512x512_fp16_flash_50steps.py \
      --ckpt_path "$MODEL_CKPT" \
      --calib_config tools/w4a6_baseline_calib_stability_500.yaml \
      --calib_data "$PHASE_DATA/$scheme/calib_data.pt" \
      --precompute_text_embeds "$TEXT_EMBEDS" \
      --outdir "$outdir" \
      --part_fp \
      --time_mp_config_weight t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml \
      --time_mp_config_act t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml \
      --use_grad_scaler \
      --reconstruction_checkpoint_interval 500 \
      2>&1 | tee "$RESULTS/$scheme/console.log"
    date -Is > "$RESULTS/$scheme/end_time.txt"
  else
    echo "$(date -Is) reusing completed $scheme checkpoint"
  fi
  test -s "$outdir/ckpt.pth"

  holdout="$RESULTS/$scheme/heldout_prompt0_seed42"
  mkdir -p "$holdout"
  if [[ ! -s "$holdout/metrics.json" ]]; then
    python tools/profile_checkpoint_holdout_trajectory.py \
      --config t2v/configs/quant/opensora/16x512x512_fp16_flash_100steps.py \
      --calib-config tools/w4a6_baseline_calib_stability_500.yaml \
      --quant-ckpt "$outdir/ckpt.pth" \
      --text-embeds "$TEXT_EMBEDS" \
      --prompt-index 0 \
      --seed 42 \
      --time-mp-config-weight t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml \
      --time-mp-config-act t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml \
      --output-dir "$holdout" \
      2>&1 | tee "$holdout/run.log"
  else
    echo "$(date -Is) reusing completed $scheme held-out trajectory"
  fi
  test -s "$holdout/metrics.json"
  echo complete > "$RESULTS/$scheme/status.txt"
done

python tools/analyze_calibration_stability.py \
  --checkpoints \
    "$PROMPT_RESULTS/subset_0/calibration/ckpt.pth" \
    "$RESULTS/early_heavy/calibration/ckpt.pth" \
    "$RESULTS/late_heavy/calibration/ckpt.pth" \
  --labels uniform early_heavy late_heavy \
  --output-dir "$RESULTS/checkpoint_comparison" \
  2>&1 | tee "$RESULTS/checkpoint_comparison.log"

echo complete > "$RESULTS/status.txt"
