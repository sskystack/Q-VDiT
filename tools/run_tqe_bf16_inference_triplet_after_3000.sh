#!/usr/bin/env bash
set -euo pipefail

repo=/home/zhouchongtian/quantization/qvdit_flash_bf16
python_bin=/home/zhouchongtian/miniconda3/envs/qvdit/bin/python
conda_sh=/home/zhouchongtian/miniconda3/etc/profile.d/conda.sh
model_ckpt=/home/zhouchongtian/quantization/Q-VDiT/logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth
text_embeds=/home/zhouchongtian/quantization/new/logs_50steps/text_embeds_samples_10.pt
prompt_path=./t2v/assets/texts/t2v_samples_10.txt
weight_mp=./t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml
act_mp=./t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml
baseline_cfg=./t2v/configs/quant/opensora/w4a6_baseline.yaml
model_cfg=./t2v/configs/quant/opensora/16x512x512_bf16_flash_100steps.py

trajectory_status="$repo/logs_fp16_flash/tqe_gpu5_serial_followup.log"
root="$repo/logs_fp16_flash/root_cause_bf16_inference_triplet_3000"
status="$root/status.log"

unscaled_ckpt="$repo/logs_fp16_flash/formal_w4a6_tqe_only_fp16_gradient_probe_gpu5_0725/calibration/ckpt_iter_00003000.pth"
scaled_ckpt="$repo/logs_fp16_flash/formal_w4a6_tqe_only_fp16_gradscaler_gpu5_0725/calibration/ckpt_iter_00003000.pth"
bf16_ckpt="$repo/logs_bf16_flash/formal_w4a6_tqe_only_bf16_on_fp16_data_gpu5_serial_0725/calibration/ckpt_iter_00003000.pth"

mkdir -p "$root"
cd "$repo"

stamp() {
    date "+$* %F %T" >> "$status"
}

# Do not claim GPU5 until all three calibrations and the paired 3000-step
# analysis have completed.  GPU6 and every other device remain untouched.
while ! grep -q "TRAJECTORY_3000_ANALYSIS_COMPLETE" "$trajectory_status" 2>/dev/null; do
    sleep 60
done

for checkpoint in "$unscaled_ckpt" "$scaled_ckpt" "$bf16_ckpt"; do
    if [[ ! -s "$checkpoint" ]]; then
        stamp "FAILED missing_checkpoint=$checkpoint"
        exit 1
    fi
done

run_one() {
    local label=$1
    local checkpoint=$2
    local outdir="$root/$label/runtime"
    local save_dir="$root/$label/videos"
    mkdir -p "$outdir" "$save_dir"
    stamp "START label=$label checkpoint=$checkpoint"
    CUDA_VISIBLE_DEVICES=5 "$python_bin" t2v/scripts/quant_txt2video.py \
        "$model_cfg" \
        --ckpt_path "$model_ckpt" \
        --calib_config "$baseline_cfg" \
        --quant_ckpt "$checkpoint" \
        --outdir "$outdir" \
        --save_dir "$save_dir" \
        --prompt_path "$prompt_path" \
        --precompute_text_embeds "$text_embeds" \
        --num_videos 1 \
        --batch_size 1 \
        --num_sampling_steps 100 \
        --cfg_scale 4.0 \
        --sampler ddim \
        --seed 42 \
        --dataset_type opensora \
        --part_fp \
        --time_mp_config_weight "$weight_mp" \
        --time_mp_config_act "$act_mp" \
        2>&1 | tee "$root/$label/inference.log"
    stamp "COMPLETE label=$label"
}

source "$conda_sh"
conda activate qvdit
export PYTHONPATH="$repo:$repo/t2v"

run_one fp16_unscaled "$unscaled_ckpt"
run_one fp16_gradscaler "$scaled_ckpt"
run_one bf16_unscaled "$bf16_ckpt"
stamp "INFERENCE_TRIPLET_COMPLETE"

