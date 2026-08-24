#!/usr/bin/env bash
set -euo pipefail

source /home/zhouchongtian/miniconda3/etc/profile.d/conda.sh
conda activate qvdit

CODE_ROOT=${CODE_ROOT:-/home/zhouchongtian/quantization/qvdit_adaptive_mtd_calibration_v1}
DATA_ROOT=${DATA_ROOT:-/home/zhouchongtian/quantization/qvdit_flash_bf16}
CALIB_ROOT=${CALIB_ROOT:-$DATA_ROOT/logs_fp16_flash/adaptive_mtd_calibration_v1_200}
OUT_ROOT=${OUT_ROOT:-$DATA_ROOT/logs_fp16_flash/adaptive_mtd_holdout_v1}
GPU_INDEX=${GPU_INDEX:-6}
PROMPT_INDICES=${PROMPT_INDICES:-"0 2 6"}
read -r -a PROMPT_INDEX_ARGS <<< "$PROMPT_INDICES"

PROMPTS="$DATA_ROOT/t2v/assets/texts/t2v_samples_10.txt"
EMBEDS=/home/zhouchongtian/quantization/Q-VDiT/t2v/utils_files/text_embeds.pth
MODEL_CKPT=/home/zhouchongtian/quantization/Q-VDiT/logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth
CONFIG="$CODE_ROOT/t2v/configs/quant/opensora/16x512x512_fp16_flash_100steps.py"
MP_WEIGHT="$CODE_ROOT/t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml"
MP_ACT="$CODE_ROOT/t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml"

mkdir -p "$OUT_ROOT"
if test -e "$OUT_ROOT/fixed" || test -e "$OUT_ROOT/centered" || test -e "$OUT_ROOT/random"; then
  echo "Refusing to mix with an existing held-out run below $OUT_ROOT" >&2
  exit 2
fi

cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT:$CODE_ROOT/t2v"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
echo $$ > "$OUT_ROOT/runner.pid"
echo "RUNNING gpu=$GPU_INDEX $(date '+%F %T')" > "$OUT_ROOT/status.txt"

CURRENT_ARM=setup
on_error() {
  code=$?
  echo "FAILED arm=$CURRENT_ARM gpu=$GPU_INDEX exit=$code $(date '+%F %T')" > "$OUT_ROOT/status.txt"
  if test -d "$OUT_ROOT/$CURRENT_ARM"; then
    echo "FAILED arm=$CURRENT_ARM gpu=$GPU_INDEX exit=$code $(date '+%F %T')" > "$OUT_ROOT/$CURRENT_ARM/status.txt"
  fi
  exit "$code"
}
trap on_error ERR

run_arm() {
  local arm=$1
  local calib_config=$2
  local quant_ckpt=$3
  local arm_out="$OUT_ROOT/$arm"
  CURRENT_ARM=$arm
  mkdir -p "$arm_out/runtime" "$arm_out/video" "$arm_out/features" "$arm_out/analysis/cond" "$arm_out/analysis/guided"
  echo "RUNNING arm=$arm gpu=$GPU_INDEX $(date '+%F %T')" > "$arm_out/status.txt"

  python t2v/scripts/quant_txt2video.py \
    "$CONFIG" \
    --ckpt_path "$MODEL_CKPT" \
    --calib_config "$calib_config" \
    --quant_ckpt "$quant_ckpt" \
    --outdir "$arm_out/runtime" \
    --save_dir "$arm_out/video" \
    --prompt_path "$PROMPTS" \
    --precompute_text_embeds "$EMBEDS" \
    --prompt_indices "${PROMPT_INDEX_ARGS[@]}" \
    --replay_original_prompt_rng \
    --num_videos "${#PROMPT_INDEX_ARGS[@]}" \
    --batch_size 1 \
    --num_sampling_steps 100 \
    --cfg_scale 4.0 \
    --sampler ddim \
    --seed 42 \
    --dataset_type opensora \
    --part_fp \
    --time_mp_config_weight "$MP_WEIGHT" \
    --time_mp_config_act "$MP_ACT" \
    --mtd_profile_dir "$arm_out/features" \
    --mtd_profile_steps 25 50 75 \
    --mtd_profile_transport_size 16 \
    2>&1 | tee "$arm_out/feature_capture.log"

  python tools/profile_mtd_adaptive_correspondence.py \
    --profile-dir "$arm_out/features" \
    --output "$arm_out/analysis/cond" \
    --teacher-key fp_cond_pooled \
    --student-key quant_cond_pooled \
    --search-radius 2 --topk 9 --temperature 0.07 --seed 42 --device cuda:0 \
    2>&1 | tee "$arm_out/analysis_cond.log"

  python tools/profile_mtd_adaptive_correspondence.py \
    --profile-dir "$arm_out/features" \
    --output "$arm_out/analysis/guided" \
    --teacher-key fp_guided_pooled \
    --student-key quant_guided_pooled \
    --search-radius 2 --topk 9 --temperature 0.07 --seed 42 --device cuda:0 \
    2>&1 | tee "$arm_out/analysis_guided.log"

  echo "COMPLETED arm=$arm gpu=$GPU_INDEX $(date '+%F %T')" > "$arm_out/status.txt"
}

run_arm fixed \
  "$CODE_ROOT/t2v/configs/quant/opensora/w4a6_mtd_adaptive_v1_fixed_200.yaml" \
  "$CALIB_ROOT/fixed/calibration/ckpt.pth"

run_arm centered \
  "$CODE_ROOT/t2v/configs/quant/opensora/w4a6_mtd_adaptive_v1_centered_200.yaml" \
  "$CALIB_ROOT/centered/calibration/ckpt.pth"

CURRENT_ARM=compare_ad
python tools/compare_adaptive_mtd_holdout.py \
  --baseline "$OUT_ROOT/fixed" \
  --candidate "$OUT_ROOT/centered" \
  --output "$OUT_ROOT/ad_comparison.json" \
  2>&1 | tee "$OUT_ROOT/ad_comparison.log"

RUN_RANDOM=$(python - "$OUT_ROOT/ad_comparison.json" <<'PY'
import json, sys
print("1" if json.load(open(sys.argv[1]))["run_random_control"] else "0")
PY
)

if test "$RUN_RANDOM" = 1; then
  run_arm random \
    "$CODE_ROOT/t2v/configs/quant/opensora/w4a6_mtd_adaptive_v1_random_200.yaml" \
    "$CALIB_ROOT/random/calibration/ckpt.pth"
  CURRENT_ARM=compare_adc
  python tools/compare_adaptive_mtd_holdout.py \
    --baseline "$OUT_ROOT/fixed" \
    --candidate "$OUT_ROOT/centered" \
    --random "$OUT_ROOT/random" \
    --output "$OUT_ROOT/adc_comparison.json" \
    2>&1 | tee "$OUT_ROOT/adc_comparison.log"
else
  echo "SKIPPED random: A-vs-D pooled held-out gate did not pass" > "$OUT_ROOT/random_status.txt"
fi

CURRENT_ARM=complete
echo "COMPLETED gpu=$GPU_INDEX random_started=$RUN_RANDOM $(date '+%F %T')" > "$OUT_ROOT/status.txt"
