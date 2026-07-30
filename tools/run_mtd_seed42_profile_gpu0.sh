#!/usr/bin/env bash
set -euo pipefail

source /home/zhouchongtian/miniconda3/etc/profile.d/conda.sh
conda activate qvdit

ROOT=/home/zhouchongtian/quantization/qvdit_flash_bf16
OUT="$ROOT/logs_fp16_flash/mtd_seed42_motion_profile_0730"
MANIFEST="$ROOT/tools/mtd_seed42_manifest.json"
PROMPTS="$ROOT/t2v/assets/texts/vbench_official/subject_consistency.txt"
EMBEDS="$ROOT/logs_bf16_flash/vbench_mtd_iter5000/subject_consistency_embeds.pth"
CKPT=/home/zhouchongtian/quantization/Q-VDiT/logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth
QUANT_CKPT="$ROOT/logs_fp16_flash/formal_w4a6_mtd_gradscaler_samples10_gpu5_0725/calibration/ckpt.pth"
CALIB_CONFIG="$ROOT/t2v/configs/quant/opensora/w4a6_mtd.yaml"
MP_WEIGHT="$ROOT/t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml"
MP_ACT="$ROOT/t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml"
PROFILE_DIR="$OUT/features"
RERUN_DIR="$OUT/rerun"
ANALYSIS_DIR="$OUT/analysis"

mkdir -p "$OUT" "$PROFILE_DIR" "$RERUN_DIR" "$ANALYSIS_DIR"
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/t2v"
export CUDA_VISIBLE_DEVICES=0

mapfile -t INDICES < <(python - <<PY
import json
data=json.load(open("$MANIFEST"))
indices=[]
for group in ("flip_1to0", "control_both_true"):
    for item in data[group]:
        indices.append(int(item["index"]))
for index in sorted(indices):
    print(index)
PY
)

python t2v/scripts/quant_txt2video.py \
  "$ROOT/t2v/configs/quant/opensora/16x512x512_fp16_flash_100steps.py" \
  --ckpt_path "$CKPT" \
  --calib_config "$CALIB_CONFIG" \
  --quant_ckpt "$QUANT_CKPT" \
  --outdir "$RERUN_DIR/runtime" \
  --save_dir "$RERUN_DIR/video" \
  --prompt_path "$PROMPTS" \
  --precompute_text_embeds "$EMBEDS" \
  --prompt_as_path \
  --prompt_indices "${INDICES[@]}" \
  --replay_original_prompt_rng \
  --num_videos 16 \
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
  --mtd_profile_steps 1 25 50 75 100 \
  --mtd_profile_transport_size 16 \
  2>&1 | tee "$OUT/feature_capture.log"

conda activate vbench
cd /home/zhouchongtian/quantization/eval/Vbench
export PYTHONPATH=/home/zhouchongtian/quantization/eval/Vbench
python "$ROOT/tools/analyze_mtd_seed42_profile.py" \
  --manifest "$MANIFEST" \
  --profile-dir "$PROFILE_DIR" \
  --baseline-videos "$ROOT/logs_fp16_flash/vbench_baseline_fp16gs_final10000/subject_opensora" \
  --mtd-videos "$ROOT/logs_fp16_flash/vbench_mtdfp16gs_final10000/subject_opensora" \
  --rerun-videos "$RERUN_DIR/video_opensora" \
  --output "$ANALYSIS_DIR" \
  --device cuda:0 \
  2>&1 | tee "$OUT/analysis.log"

echo "COMPLETED $(date '+%F %T')" > "$OUT/status.txt"
