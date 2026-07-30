#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/home/zhouchongtian/quantization/qvdit_flash_bf16
RUN_ROOT="$REPO/logs_fp16_flash/vbench_mtd_w01_fp16gs_final10000"
CALIB_DIR="$REPO/logs_fp16_flash/formal_w4a6_mtd_w01_gradprobe_samples10_gpu2_0728/calibration"
MODEL_CFG="$REPO/t2v/configs/quant/opensora/16x512x512_fp16_flash_100steps.py"
QUANT_CFG="$REPO/t2v/configs/quant/opensora/w4a6_mtd_w01_probe.yaml"
MODEL_CKPT=/home/zhouchongtian/quantization/Q-VDiT/logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth
QUANT_CKPT="$CALIB_DIR/ckpt.pth"
MP_WEIGHT="$REPO/t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml"
MP_ACT="$REPO/t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml"
OVERALL_PROMPTS="$REPO/t2v/assets/texts/vbench_official/overall_consistency.txt"
OVERALL_EMBEDS=/home/zhouchongtian/quantization/Q-VDiT/logs_50steps/early_stop_cfg01_proxy/vbench_overall_embeds.pth
VBENCH_REPO=/home/zhouchongtian/quantization/eval/Vbench
LOG="$RUN_ROOT/overall_gpu5.log"

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$LOG") 2>&1
trap 'printf "[%s] FAILED at line %s: %s\n" "$(date "+%F %T")" "$LINENO" "$BASH_COMMAND"' ERR

source /home/zhouchongtian/miniconda3/etc/profile.d/conda.sh
conda activate qvdit
cd "$REPO"
export PYTHONPATH="$REPO:$REPO/t2v"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=5

rm -rf "$RUN_ROOT/overall_runtime" "$RUN_ROOT/overall_opensora"
mkdir -p "$RUN_ROOT/overall_runtime"
echo "[$(date '+%F %T')] Generating Overall (93 videos)"
python t2v/scripts/quant_txt2video.py "$MODEL_CFG" \
    --ckpt_path "$MODEL_CKPT" \
    --calib_config "$QUANT_CFG" \
    --quant_ckpt "$QUANT_CKPT" \
    --outdir "$RUN_ROOT/overall_runtime" \
    --save_dir "$RUN_ROOT/overall" \
    --prompt_path "$OVERALL_PROMPTS" \
    --precompute_text_embeds "$OVERALL_EMBEDS" \
    --prompt_as_path \
    --num_videos 93 \
    --batch_size 1 \
    --num_sampling_steps 100 \
    --cfg_scale 4.0 \
    --sampler ddim \
    --seed 42 \
    --dataset_type opensora \
    --part_fp \
    --time_mp_config_weight "$MP_WEIGHT" \
    --time_mp_config_act "$MP_ACT"
[[ "$(find "$RUN_ROOT/overall_opensora" -maxdepth 1 -type f -name '*.mp4' | wc -l)" -eq 93 ]]

conda activate vbench
cd "$VBENCH_REPO"
export VBENCH_CACHE_DIR=/home/zhouchongtian/quantization/models/vbench
rm -rf "$RUN_ROOT/overall_eval"
mkdir -p "$RUN_ROOT/overall_eval"
echo "[$(date '+%F %T')] Evaluating Overall metrics"
python evaluate.py \
    --videos_path "$RUN_ROOT/overall_opensora" \
    --output_path "$RUN_ROOT/overall_eval" \
    --dimension overall_consistency aesthetic_quality imaging_quality \
    --load_ckpt_from_local True

echo "[$(date '+%F %T')] GPU5 Overall pipeline complete"
