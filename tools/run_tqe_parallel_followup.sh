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
bf16="$repo/logs_bf16_flash/formal_w4a6_tqe_only_bf16_on_fp16_data_gpu6_0725/calibration"
status="$repo/logs_fp16_flash/tqe_parallel_followup.log"

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

start_resume() {
    local session=$1
    local gpu=$2
    local model_config=$3
    local outdir=$4
    local state=$5
    shift 5
    local extra=("$@")
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
        --resume_reconstruction "$state" \
        "${extra[@]}"
    tmux new-session -d -s "$session" \
        "bash -lc 'source $conda_sh && conda activate qvdit && cd $repo && export PYTHONPATH=$repo:$repo/t2v && export CUDA_VISIBLE_DEVICES=$gpu && $command 2>&1 | tee -a $outdir/console.log'"
    stamp "STARTED session=$session gpu=$gpu state=$state"
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

stamp "FOLLOWUP_BEGIN"

wait_for_checkpoint fp16scaled "$scaled/ckpt_iter_00000500.pth"
stop_session fp16scaled
wait_for_checkpoint bf16same "$bf16/ckpt_iter_00000500.pth"
stop_session bf16same

analyze_pair "$scaled" "$unscaled" "$repo/logs_fp16_flash/fp16_gradscaler_vs_unscaled_500" 500
analyze_pair "$unscaled" "$bf16" "$repo/logs_fp16_flash/fp16_vs_bf16_same_data_500" 500
analyze_pair "$scaled" "$bf16" "$repo/logs_fp16_flash/fp16_gradscaler_vs_bf16_same_data_500" 500
stamp "EARLY_500_ANALYSIS_COMPLETE"

# Run the two FP16 trajectories in parallel first.  As soon as unscaled FP16
# finishes, reuse GPU5 for BF16 so that only the authorized GPUs 5 and 6 are used.
start_resume fp16unscaled_resume 5 \
    ./t2v/configs/quant/opensora/16x512x512_fp16_flash_50steps.py \
    "$unscaled" "$unscaled/reconstruction_state_iter_00000500.pth"
start_resume fp16scaled_resume 6 \
    ./t2v/configs/quant/opensora/16x512x512_fp16_flash_50steps.py \
    "$scaled" "$scaled/reconstruction_state_iter_00000500.pth" \
    --use_grad_scaler

wait_for_checkpoint fp16unscaled_resume "$unscaled/ckpt_iter_00003000.pth"
stop_session fp16unscaled_resume

start_resume bf16same_resume 5 \
    ./t2v/configs/quant/opensora/16x512x512_bf16_flash_50steps.py \
    "$bf16" "$bf16/reconstruction_state_iter_00000500.pth"

wait_for_checkpoint fp16scaled_resume "$scaled/ckpt_iter_00003000.pth"
stop_session fp16scaled_resume
wait_for_checkpoint bf16same_resume "$bf16/ckpt_iter_00003000.pth"
stop_session bf16same_resume

analyze_pair "$scaled" "$unscaled" "$repo/logs_fp16_flash/fp16_gradscaler_vs_unscaled_3000" 3000
analyze_pair "$unscaled" "$bf16" "$repo/logs_fp16_flash/fp16_vs_bf16_same_data_3000" 3000
analyze_pair "$scaled" "$bf16" "$repo/logs_fp16_flash/fp16_gradscaler_vs_bf16_same_data_3000" 3000
stamp "TRAJECTORY_3000_ANALYSIS_COMPLETE"

