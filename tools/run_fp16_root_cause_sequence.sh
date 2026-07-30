#!/usr/bin/env bash
set -euo pipefail

repo=/home/zhouchongtian/quantization/qvdit_flash_bf16
conda_sh=/home/zhouchongtian/miniconda3/etc/profile.d/conda.sh
python_bin=/home/zhouchongtian/miniconda3/envs/qvdit/bin/python
model_ckpt=/home/zhouchongtian/quantization/Q-VDiT/logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth
calib_data="$repo/logs_fp16_flash/formal_w4a6_mtd_samples10_gpu5_0724/calib_data_ddim50_cfg4/calib_data.pt"
text_embeds=/home/zhouchongtian/quantization/new/logs_50steps/text_embeds_samples_10.pt
weight_mp=./t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml
act_mp=./t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml
baseline_cfg=./t2v/configs/quant/opensora/w4a6_baseline.yaml
status="$repo/logs_fp16_flash/fp16_root_cause_sequence.log"

cd "$repo"

wait_for_checkpoint() {
    local session=$1
    local checkpoint=$2
    while [[ ! -f "$checkpoint" ]]; do
        if ! tmux has-session -t "$session" 2>/dev/null; then
            date "+FAILED session=$session missing checkpoint=$checkpoint %F %T" >> "$status"
            return 1
        fi
        date "+waiting session=$session checkpoint=$checkpoint %F %T" >> "$status"
        sleep 300
    done
    date "+ready session=$session checkpoint=$checkpoint %F %T" >> "$status"
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
        "bash -lc 'source $conda_sh && conda activate qvdit && cd $repo && export PYTHONPATH=$repo:$repo/t2v && export CUDA_VISIBLE_DEVICES=5 && $command 2>&1 | tee $outdir/console.log'"
    date "+started session=$session outdir=$outdir %F %T" >> "$status"
}

analyze_pair() {
    local first=$1
    local second=$2
    local output=$3
    "$python_bin" tools/analyze_tqe_checkpoint_trajectory.py \
        --fp16-dir "$first" \
        --bf16-dir "$second" \
        --bit-config t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml \
        --max-iteration 3000 \
        --csv "${output}.csv" > "${output}.txt"
}

current="$repo/logs_fp16_flash/formal_w4a6_tqe_only_fp16_gradient_probe_gpu5_0725/calibration"
wait_for_checkpoint fp16tqenum "$current/ckpt_iter_00003000.pth"
stop_session fp16tqenum

scaled="$repo/logs_fp16_flash/formal_w4a6_tqe_only_fp16_gradscaler_gpu5_0725/calibration"
start_calibration fp16scaled \
    ./t2v/configs/quant/opensora/16x512x512_fp16_flash_50steps.py \
    "$scaled" --use_grad_scaler
wait_for_checkpoint fp16scaled "$scaled/ckpt_iter_00003000.pth"
stop_session fp16scaled
analyze_pair "$scaled" "$current" \
    "$repo/logs_fp16_flash/fp16_gradscaler_vs_unscaled_3000"

bf16="$repo/logs_bf16_flash/formal_w4a6_tqe_only_bf16_on_fp16_data_gpu5_0725/calibration"
start_calibration bf16same \
    ./t2v/configs/quant/opensora/16x512x512_bf16_flash_50steps.py \
    "$bf16" --paired_gradient_probe_scale 4096
wait_for_checkpoint bf16same "$bf16/ckpt_iter_00003000.pth"
stop_session bf16same
analyze_pair "$current" "$bf16" \
    "$repo/logs_fp16_flash/fp16_vs_bf16_same_data_3000"

date "+COMPLETE %F %T" >> "$status"
