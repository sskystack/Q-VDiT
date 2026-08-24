#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/zhouchongtian/quantization/qvdit_flash_bf16
EMBED_STATUS="$ROOT/logs_fp16_flash/vbench_official_embeds/status.txt"
ORCHESTRATOR_LOG="$ROOT/logs_fp16_flash/noglobal_vbench_gpu5_gpu6_orchestrator.log"

exec > >(tee -a "$ORCHESTRATOR_LOG") 2>&1
echo "WAITING_FOR_EMBEDDINGS $(date '+%F %T')"
until grep -q '^COMPLETED ' "$EMBED_STATUS" 2>/dev/null; do
    if grep -q '^FAILED ' "$EMBED_STATUS" 2>/dev/null; then
        echo "Embedding preparation failed" >&2
        exit 1
    fi
    sleep 30
done

echo "STARTING_W01_GPU6_AND_W1_GPU5 $(date '+%F %T')"
bash "$ROOT/tools/wait_mtd_noglobal_then_vbench.sh" \
    6 w4a6_mtd_noglobal_w01_probe.yaml formal_w4a6_mtd_noglobal_w01_gpu6_0730 &
pid_w01=$!
bash "$ROOT/tools/wait_mtd_noglobal_then_vbench.sh" \
    5 w4a6_mtd_noglobal_w1_probe.yaml formal_w4a6_mtd_noglobal_w1_gpu5_0730 &
pid_w1=$!

status=0
wait "$pid_w01" || status=1
wait "$pid_w1" || status=1
echo "FINISHED status=$status $(date '+%F %T')"
exit "$status"
