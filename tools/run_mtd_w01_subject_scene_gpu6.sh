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
SUBJECT_PROMPTS="$REPO/t2v/assets/texts/vbench_official/subject_consistency.txt"
SCENE_PROMPTS="$REPO/t2v/assets/texts/vbench_official/scene.txt"
SUBJECT_EMBEDS="$REPO/logs_bf16_flash/vbench_mtd_iter5000/subject_consistency_embeds.pth"
SCENE_EMBEDS=/home/zhouchongtian/quantization/Q-VDiT/logs_50steps/early_stop_cfg01_proxy/vbench_scene_embeds.pth
VBENCH_REPO=/home/zhouchongtian/quantization/eval/Vbench
LOG="$RUN_ROOT/subject_scene_gpu6_resume.log"

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$LOG") 2>&1
trap 'printf "[%s] FAILED at line %s: %s\n" "$(date "+%F %T")" "$LINENO" "$BASH_COMMAND"' ERR

source /home/zhouchongtian/miniconda3/etc/profile.d/conda.sh
conda activate qvdit
cd "$REPO"
export PYTHONPATH="$REPO:$REPO/t2v"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=6

count_mp4() {
    find "$1" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null | wc -l
}

SUBJECT_VIDEO_DIR="$RUN_ROOT/subject_opensora"
if [[ "$(count_mp4 "$SUBJECT_VIDEO_DIR")" -lt 72 ]]; then
    mkdir -p "$RUN_ROOT/subject_resume_runtime"
    echo "[$(date '+%F %T')] Resuming Subject from prompt index 45 (27 remaining)"
    # The original loop used noise seed 42+i. Starting at prompt 45 therefore
    # uses base seed 42+45=87 to preserve the uninterrupted noise sequence.
    python t2v/scripts/quant_txt2video.py "$MODEL_CFG" \
        --ckpt_path "$MODEL_CKPT" \
        --calib_config "$QUANT_CFG" \
        --quant_ckpt "$QUANT_CKPT" \
        --outdir "$RUN_ROOT/subject_resume_runtime" \
        --save_dir "$RUN_ROOT/subject" \
        --prompt_path "$SUBJECT_PROMPTS" \
        --precompute_text_embeds "$SUBJECT_EMBEDS" \
        --prompt_start_index 45 \
        --prompt_as_path \
        --num_videos 27 \
        --batch_size 1 \
        --num_sampling_steps 100 \
        --cfg_scale 4.0 \
        --sampler ddim \
        --seed 87 \
        --dataset_type opensora \
        --part_fp \
        --time_mp_config_weight "$MP_WEIGHT" \
        --time_mp_config_act "$MP_ACT"
fi
[[ "$(count_mp4 "$SUBJECT_VIDEO_DIR")" -eq 72 ]]

conda activate vbench
cd "$VBENCH_REPO"
export VBENCH_CACHE_DIR=/home/zhouchongtian/quantization/models/vbench
rm -rf "$RUN_ROOT/subject_eval"
mkdir -p "$RUN_ROOT/subject_eval"
echo "[$(date '+%F %T')] Evaluating Subject metrics"
python evaluate.py \
    --videos_path "$SUBJECT_VIDEO_DIR" \
    --output_path "$RUN_ROOT/subject_eval" \
    --dimension subject_consistency dynamic_degree motion_smoothness \
    --load_ckpt_from_local True

conda activate qvdit
cd "$REPO"
export PYTHONPATH="$REPO:$REPO/t2v"
rm -rf "$RUN_ROOT/scene_runtime" "$RUN_ROOT/scene_opensora"
mkdir -p "$RUN_ROOT/scene_runtime"
echo "[$(date '+%F %T')] Generating Scene (86 videos)"
python t2v/scripts/quant_txt2video.py "$MODEL_CFG" \
    --ckpt_path "$MODEL_CKPT" \
    --calib_config "$QUANT_CFG" \
    --quant_ckpt "$QUANT_CKPT" \
    --outdir "$RUN_ROOT/scene_runtime" \
    --save_dir "$RUN_ROOT/scene" \
    --prompt_path "$SCENE_PROMPTS" \
    --precompute_text_embeds "$SCENE_EMBEDS" \
    --prompt_as_path \
    --num_videos 86 \
    --batch_size 1 \
    --num_sampling_steps 100 \
    --cfg_scale 4.0 \
    --sampler ddim \
    --seed 42 \
    --dataset_type opensora \
    --part_fp \
    --time_mp_config_weight "$MP_WEIGHT" \
    --time_mp_config_act "$MP_ACT"
[[ "$(count_mp4 "$RUN_ROOT/scene_opensora")" -eq 86 ]]

conda activate vbench
cd "$VBENCH_REPO"
rm -rf "$RUN_ROOT/scene_eval"
mkdir -p "$RUN_ROOT/scene_eval"
echo "[$(date '+%F %T')] Evaluating Scene metrics"
python evaluate.py \
    --videos_path "$RUN_ROOT/scene_opensora" \
    --output_path "$RUN_ROOT/scene_eval" \
    --dimension scene background_consistency \
    --load_ckpt_from_local True

echo "[$(date '+%F %T')] GPU6 Subject and Scene pipeline complete"
