#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zhouchongtian/quantization/qvdit_flash_bf16
PYTHON=/home/zhouchongtian/miniconda3/envs/qvdit/bin/python
CFG="$ROOT/t2v/configs/quant/opensora/16x512x512_fp16_flash_100steps.py"
MODEL_CKPT=/home/zhouchongtian/quantization/Q-VDiT/logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth
CALIB_CFG="$ROOT/t2v/configs/quant/opensora/w4a6_baseline.yaml"
QUANT_CKPT="$ROOT/logs_fp16_flash/formal_w4a6_baseline_fp16_gradscaler_samples10_gpu5_0726/calibration/ckpt.pth"
PROMPTS="$ROOT/t2v/assets/texts/t2v_samples_10.txt"
TEXT_EMBEDS=/home/zhouchongtian/quantization/new/logs_50steps/text_embeds_samples_10.pt
MP_WEIGHT="$ROOT/t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml"
MP_ACT="$ROOT/t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml"
OUT_ROOT="$ROOT/logs_fp16_flash/stage_quant_profiling/b3_seed42_cross_prompt_w4a6_fine_windows"
REFERENCE_ROOT="$ROOT/logs_fp16_flash/stage_quant_profiling/b2_seed42_cross_prompt_w4a6"

source /home/zhouchongtian/miniconda3/etc/profile.d/conda.sh
conda activate qvdit
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/t2v"
export CUDA_VISIBLE_DEVICES=6
mkdir -p "$OUT_ROOT"

run_case() {
  local prompt_index="$1"
  local start="$2"
  local end="$3"
  local tag
  tag="$(printf 'p%03d_%03d' "$start" "$end")"
  local base="$OUT_ROOT/prompt${prompt_index}/w4a6/$tag"
  local outdir="$base/runtime"
  local savedir="$base/video"
  mkdir -p "$outdir" "$savedir"
  if [[ -s "$outdir/final_latents/final_latent_0000.pt" && -s "$outdir/quant_trace_batch_0000.json" ]]; then
    echo "$(date -Is) skipping completed prompt=$prompt_index window=$start-$end"
    return
  fi

  echo "$(date -Is) starting prompt=$prompt_index window=$start-$end"
  "$PYTHON" t2v/scripts/quant_txt2video.py "$CFG" \
    --ckpt_path "$MODEL_CKPT" \
    --calib_config "$CALIB_CFG" \
    --quant_ckpt "$QUANT_CKPT" \
    --outdir "$outdir" \
    --save_dir "$savedir" \
    --prompt_path "$PROMPTS" \
    --prompt_start_index "$prompt_index" \
    --precompute_text_embeds "$TEXT_EMBEDS" \
    --num_videos 1 \
    --batch_size 1 \
    --num_sampling_steps 100 \
    --cfg_scale 4.0 \
    --sampler ddim \
    --seed 42 \
    --dataset_type opensora \
    --part_fp \
    --time_mp_config_weight "$MP_WEIGHT" \
    --time_mp_config_act "$MP_ACT" \
    --save_final_latent \
    --save_init_noise \
    --timestep_wise_quant \
    --quant_progress_start "$start" \
    --quant_progress_end "$end" \
    --save_quant_trace \
    2>&1 | tee "$base/console.log"
  test -s "$outdir/final_latents/final_latent_0000.pt"
  test -s "$outdir/quant_trace_batch_0000.json"
  echo "$(date -Is) finished prompt=$prompt_index window=$start-$end"
}

for prompt_index in 0 2 6; do
  for start in 11 16 21 26; do
    run_case "$prompt_index" "$start" "$((start + 4))"
  done
done

"$PYTHON" tools/analyze_experiment_b3_fine.py \
  --root "$OUT_ROOT" \
  --reference-root "$REFERENCE_ROOT"
echo complete > "$OUT_ROOT/status.txt"

