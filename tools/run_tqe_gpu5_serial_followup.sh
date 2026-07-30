#!/usr/bin/env bash
set -euo pipefail

repo=/home/zhouchongtian/quantization/qvdit_flash_bf16
python_bin=/home/zhouchongtian/miniconda3/envs/qvdit/bin/python
conda_sh=/home/zhouchongtian/miniconda3/etc/profile.d/conda.sh
model_ckpt=/home/zhouchongtian/quantization/Q-VDiT/logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth
calib_data="$repo/logs_fp16_flash/formal_w4a6_mtd_samples10_gpu5_0724/calib_data_ddim50_cfg4/calib_data.pt"
text_embeds=/home/zhouchongtian/quantization/new/logs_50steps/text_embeds_samples_10.pt
weight_mp=./t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml
act_mp=./t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml
baseline_cfg=./t2v/configs/quant/opensora/w4a6_baseline.yaml

unscaled="$repo/logs_fp16_flash/formal_w4a6_tqe_only_fp16_gradient_probe_gpu5_0725/calibration"
scaled="$repo/logs_fp16_flash/formal_w4a6_tqe_only_fp16_gradscaler_gpu5_0725/calibration"
bf16="$repo/logs_bf16_flash/formal_w4a6_tqe_only_bf16_on_fp16_data_gpu5_serial_0725/calibration"
status="$repo/logs_fp16_flash/tqe_gpu5_serial_followup.log"

cd "$repo"

stamp() {
    date "+$* %F %T" >> "$status"
}

wait_for_checkpoint() {
    local session=$1
    local checkpoint=$2
    while [[ ! -f "$checkpoint" ]]; do
        if ! tmux has-session -t "$session" 2>/dev/null; then
            stamp "FAILED session=$session missing=$checkpoint"
            return 1
        fi
        sleep 60
    done
    stamp "READY session=$session checkpoint=$checkpoint"
}

stop_session() {
    local session=$1
    tmux send-keys -t "$session" C-c 2>/dev/null || true
    sleep 5
    tmux kill-session -t "$session" 2>/dev/null || true
}

start_calibration() {
    local session=$1
    local model_config=$2
    local outdir=$3
    shift 3
    local extra=("$@")
    mkdir -p "$outdir"
    local command
    printf -v command '%q ' \
        "$python_bin" t2v/scripts/calib.py "$model_config" \
        --ckpt_path "$model_ckpt" \
        --calib_config "$baseline_cfg" \
        --calib_data "$calib_data" \
        --precompute_text_embeds "$text_embeds" \
        --outdir "$outdir" \
        --part_fp \
        --time_mp_config_weight "$weight_mp" \
        --time_mp_config_act "$act_mp" \
        --numeric_monitor_interval 25 \
        --numeric_monitor_detailed_interval 100 \
        --reconstruction_checkpoint_interval 500 \
        "${extra[@]}"
    tmux new-session -d -s "$session" \
        "bash -lc 'source $conda_sh && conda activate qvdit && cd $repo && export PYTHONPATH=$repo:$repo/t2v && export CUDA_VISIBLE_DEVICES=5 && $command 2>&1 | tee -a $outdir/console.log'"
    stamp "STARTED session=$session gpu=5 outdir=$outdir"
}

analyze_pair() {
    local first=$1
    local second=$2
    local output=$3
    local max_iteration=$4
    "$python_bin" tools/analyze_tqe_checkpoint_trajectory.py \
        --fp16-dir "$first" \
        --bf16-dir "$second" \
        --base-checkpoint "$model_ckpt" \
        --bit-config t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml \
        --max-iteration "$max_iteration" \
        --csv "${output}.csv" > "${output}.txt"
}

stamp "SERIAL_REORDERED_FP16_FIRST_BEGIN"

# Cover the known 1500--3000 drift window. Finish both FP16 trajectories
# before allocating GPU5 to the BF16 control. Every run strictly restores its
# own optimizer, scheduler, scaler (when present), sampling plan, and RNG state.
start_calibration fp16unscaled_resume \
    ./t2v/configs/quant/opensora/16x512x512_fp16_flash_50steps.py \
    "$unscaled" \
    --resume_reconstruction "$unscaled/reconstruction_state_iter_00000500.pth"
wait_for_checkpoint fp16unscaled_resume "$unscaled/ckpt_iter_00003000.pth"
stop_session fp16unscaled_resume

start_calibration fp16scaled_resume \
    ./t2v/configs/quant/opensora/16x512x512_fp16_flash_50steps.py \
    "$scaled" --use_grad_scaler \
    --resume_reconstruction "$scaled/reconstruction_state_iter_00000500.pth"
wait_for_checkpoint fp16scaled_resume "$scaled/ckpt_iter_00003000.pth"
stop_session fp16scaled_resume

# Start BF16 only after both FP16 runs have released GPU5. Run it continuously
# from iteration 0 through 3000; the 500-step checkpoint is retained for the
# matched early-trajectory analysis.
start_calibration bf16same \
    ./t2v/configs/quant/opensora/16x512x512_bf16_flash_50steps.py \
    "$bf16" --paired_gradient_probe_scale 4096
wait_for_checkpoint bf16same "$bf16/ckpt_iter_00003000.pth"
stop_session bf16same

analyze_pair "$scaled" "$unscaled" "$repo/logs_fp16_flash/fp16_gradscaler_vs_unscaled_500" 500
analyze_pair "$unscaled" "$bf16" "$repo/logs_fp16_flash/fp16_vs_bf16_same_data_500" 500
analyze_pair "$scaled" "$bf16" "$repo/logs_fp16_flash/fp16_gradscaler_vs_bf16_same_data_500" 500
stamp "EARLY_500_ANALYSIS_COMPLETE"

analyze_pair "$scaled" "$unscaled" "$repo/logs_fp16_flash/fp16_gradscaler_vs_unscaled_3000" 3000
analyze_pair "$unscaled" "$bf16" "$repo/logs_fp16_flash/fp16_vs_bf16_same_data_3000" 3000
analyze_pair "$scaled" "$bf16" "$repo/logs_fp16_flash/fp16_gradscaler_vs_bf16_same_data_3000" 3000
stamp "TRAJECTORY_3000_ANALYSIS_COMPLETE"
