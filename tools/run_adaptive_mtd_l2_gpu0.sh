#!/usr/bin/env bash
set -euo pipefail

source /home/zhouchongtian/miniconda3/etc/profile.d/conda.sh
conda activate qvdit

ROOT=/home/zhouchongtian/quantization/qvdit_flash_bf16
GPU_INDEX=${GPU_INDEX:-0}
PROMPT_INDICES=${PROMPT_INDICES:-"0 2 6"}
read -r -a PROMPT_INDEX_ARGS <<< "$PROMPT_INDICES"
NUM_VIDEOS=${NUM_VIDEOS:-${#PROMPT_INDEX_ARGS[@]}}
OUT="$ROOT/logs_fp16_flash/adaptive_mtd_feature_capture_v1"
PROMPTS="$ROOT/t2v/assets/texts/t2v_samples_10.txt"
EMBEDS=/home/zhouchongtian/quantization/Q-VDiT/t2v/utils_files/text_embeds.pth
CKPT=/home/zhouchongtian/quantization/Q-VDiT/logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth
QUANT_CKPT="$ROOT/logs_fp16_flash/formal_w4a6_mtd_gradscaler_samples10_gpu5_0725/calibration/ckpt.pth"
CALIB_CONFIG="$ROOT/t2v/configs/quant/opensora/w4a6_mtd.yaml"
MP_WEIGHT="$ROOT/t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml"
MP_ACT="$ROOT/t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml"
PROFILE_DIR="$OUT/features"
ANALYSIS_DIR="$ROOT/logs_fp16_flash/stage_numeric_profiling/adaptive_mtd_candidates_v1"

mkdir -p "$OUT/runtime" "$OUT/video" "$PROFILE_DIR" "$ANALYSIS_DIR"
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/t2v"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"

# Three non-contiguous prompts and three diffusion phases are enough for the L2
# mechanism screen.  The original per-prompt RNG stream is replayed so saved
# features remain aligned with the established seed-42 setup.
python t2v/scripts/quant_txt2video.py \
  "$ROOT/t2v/configs/quant/opensora/16x512x512_fp16_flash_100steps.py" \
  --ckpt_path "$CKPT" \
  --calib_config "$CALIB_CONFIG" \
  --quant_ckpt "$QUANT_CKPT" \
  --outdir "$OUT/runtime" \
  --save_dir "$OUT/video" \
  --prompt_path "$PROMPTS" \
  --precompute_text_embeds "$EMBEDS" \
  --prompt_indices "${PROMPT_INDEX_ARGS[@]}" \
  --replay_original_prompt_rng \
  --num_videos "$NUM_VIDEOS" \
  --batch_size 1 \
  --num_sampling_steps 100 \
  --cfg_scale 4.0 \
  --sampler ddim \
  --seed 42 \
  --dataset_type opensora \
  --part_fp \
  --time_mp_config_weight "$MP_WEIGHT" \
  --time_mp_config_act "$MP_ACT" \
  --mtd_profile_dir "$PROFILE_DIR" \
  --mtd_profile_steps 25 50 75 \
  --mtd_profile_transport_size 16 \
  2>&1 | tee "$OUT/feature_capture.log"

python tools/profile_mtd_adaptive_correspondence.py \
  --profile-dir "$PROFILE_DIR" \
  --output "$ANALYSIS_DIR/cond" \
  --teacher-key fp_cond_pooled \
  --student-key quant_cond_pooled \
  --search-radius 2 \
  --topk 9 \
  --temperature 0.07 \
  --seed 42 \
  --device cuda:0 \
  2>&1 | tee "$OUT/analysis_cond.log"

python tools/profile_mtd_adaptive_correspondence.py \
  --profile-dir "$PROFILE_DIR" \
  --output "$ANALYSIS_DIR/guided" \
  --teacher-key fp_guided_pooled \
  --student-key quant_guided_pooled \
  --search-radius 2 \
  --topk 9 \
  --temperature 0.07 \
  --seed 42 \
  --device cuda:0 \
  2>&1 | tee "$OUT/analysis_guided.log"

echo "COMPLETED $(date '+%F %T')" > "$OUT/status.txt"
