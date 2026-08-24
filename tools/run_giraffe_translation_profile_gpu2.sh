#!/usr/bin/env bash
set -euo pipefail

source /home/zhouchongtian/miniconda3/etc/profile.d/conda.sh
conda activate qvdit

ROOT=/home/zhouchongtian/quantization/qvdit_flash_bf16
OUT="$ROOT/logs_fp16_flash/giraffe_translation_residual_gpu2_0731"
PROMPTS="$ROOT/t2v/assets/texts/vbench_official/subject_consistency.txt"
EMBEDS="$ROOT/logs_bf16_flash/vbench_mtd_iter5000/subject_consistency_embeds.pth"
CKPT=/home/zhouchongtian/quantization/Q-VDiT/logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth
MP_WEIGHT="$ROOT/t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml"
MP_ACT="$ROOT/t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml"
EXISTING_BASELINE="$ROOT/logs_fp16_flash/vbench_baseline_fp16gs_final10000/subject_opensora/a giraffe taking a peaceful walk.mp4"
EXISTING_MTD="$ROOT/logs_fp16_flash/vbench_mtdfp16gs_final10000/subject_opensora/a giraffe taking a peaceful walk.mp4"

mkdir -p "$OUT"
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/t2v"
export CUDA_VISIBLE_DEVICES=2

run_variant() {
  local name="$1"
  local calib="$2"
  local quant_ckpt="$3"
  mkdir -p "$OUT/$name/features" "$OUT/$name/runtime" "$OUT/$name/video"
  python t2v/scripts/quant_txt2video.py \
    "$ROOT/t2v/configs/quant/opensora/16x512x512_fp16_flash_100steps.py" \
    --ckpt_path "$CKPT" \
    --calib_config "$calib" \
    --quant_ckpt "$quant_ckpt" \
    --outdir "$OUT/$name/runtime" \
    --save_dir "$OUT/$name/video" \
    --prompt_path "$PROMPTS" \
    --precompute_text_embeds "$EMBEDS" \
    --prompt_as_path \
    --prompt_indices 70 \
    --replay_original_prompt_rng \
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
    --mtd_profile_dir "$OUT/$name/features" \
    --mtd_profile_steps 1 5 10 15 25 35 50 65 75 85 95 100 \
    --mtd_profile_transport_size 16 \
    2>&1 | tee "$OUT/${name}_capture.log"
}

run_variant \
  baseline \
  "$ROOT/t2v/configs/quant/opensora/w4a6_baseline.yaml" \
  "$ROOT/logs_fp16_flash/formal_w4a6_baseline_fp16_gradscaler_samples10_gpu5_0726/calibration/ckpt.pth"

run_variant \
  mtd \
  "$ROOT/t2v/configs/quant/opensora/w4a6_mtd.yaml" \
  "$ROOT/logs_fp16_flash/formal_w4a6_mtd_gradscaler_samples10_gpu5_0725/calibration/ckpt.pth"

python tools/analyze_giraffe_translation_residual.py \
  --baseline-profile "$OUT/baseline/features" \
  --mtd-profile "$OUT/mtd/features" \
  --output "$OUT/analysis" \
  --baseline-existing-video "$EXISTING_BASELINE" \
  --baseline-rerun-video "$OUT/baseline/video_opensora/a giraffe taking a peaceful walk.mp4" \
  --mtd-existing-video "$EXISTING_MTD" \
  --mtd-rerun-video "$OUT/mtd/video_opensora/a giraffe taking a peaceful walk.mp4" \
  2>&1 | tee "$OUT/analysis.log"

echo "COMPLETED $(date '+%F %T')" > "$OUT/status.txt"
