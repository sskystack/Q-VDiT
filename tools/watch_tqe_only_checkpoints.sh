#!/usr/bin/env bash
set -euo pipefail

cd /home/zhouchongtian/quantization/qvdit_flash_bf16
new=logs_fp16_flash/formal_w4a6_tqe_only_fp16_gradient_probe_gpu5_0725/calibration
old=logs_fp16_flash/formal_w4a6_mtd_samples10_gpu5_0724/calibration
root=logs_fp16_flash/formal_w4a6_tqe_only_fp16_gradient_probe_gpu5_0725
status="$root/watch_status.log"

for iteration in 500 1500 3000; do
    file=$(printf "%s/ckpt_iter_%08d.pth" "$new" "$iteration")
    while [[ ! -f "$file" ]]; do
        date "+waiting iteration=$iteration %F %T" >> "$status"
        sleep 300
    done
    /home/zhouchongtian/miniconda3/envs/qvdit/bin/python \
        tools/analyze_tqe_checkpoint_trajectory.py \
        --fp16-dir "$new" \
        --bf16-dir "$old" \
        --bit-config t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml \
        --max-iteration "$iteration" \
        --csv "$root/tqe_only_vs_mtd_${iteration}.csv" \
        > "$root/tqe_only_vs_mtd_${iteration}.txt"
    date "+completed iteration=$iteration %F %T" >> "$status"
done
