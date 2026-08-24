#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/zhouchongtian/quantization/qvdit_flash_bf16
OUT="$ROOT/logs_fp16_flash/stage_numeric_profiling/tqe_branch_decomposition_3prompts_gpu2_0731"
STATUS="$OUT/status.txt"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/automation.log") 2>&1

source /home/zhouchongtian/miniconda3/etc/profile.d/conda.sh
conda activate qvdit
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/t2v"
export CUDA_VISIBLE_DEVICES=2

for prompt_index in 0 2 6; do
    prompt_out="$OUT/prompt${prompt_index}"
    if [[ -s "$prompt_out/metadata.json" ]]; then
        echo "SKIP prompt=$prompt_index completed"
        continue
    fi
    mkdir -p "$prompt_out"
    echo "RUNNING prompt=$prompt_index $(date '+%F %T')" | tee "$STATUS"
    python tools/profile_internal_mechanisms.py \
        --config t2v/configs/quant/opensora/16x512x512_fp16_flash_100steps.py \
        --calib-config t2v/configs/quant/opensora/w4a6_baseline.yaml \
        --quant-ckpt logs_fp16_flash/formal_w4a6_baseline_fp16_gradscaler_samples10_gpu5_0726/calibration/ckpt.pth \
        --prompt-path t2v/assets/texts/t2v_samples_10.txt \
        --text-embeds /home/zhouchongtian/quantization/new/logs_50steps/text_embeds_samples_10.pt \
        --prompt-index "$prompt_index" \
        --seed 42 \
        --selected-progress 5,15,25,45,65,85,95 \
        --selected-blocks 0,9,17,27 \
        --attention-query-samples 8 \
        --attention-batch-samples 1 \
        --attention-head-samples 1 \
        --tqe-token-samples 64 \
        --time-mp-config-weight t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml \
        --time-mp-config-act t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml \
        --output-dir "$prompt_out" \
        2>&1 | tee "$prompt_out/run.log"
done

python tools/summarize_tqe_branch_decomposition.py \
    "$OUT"/prompt*/internal_operator_stats.jsonl \
    --output "$OUT/analysis" \
    2>&1 | tee "$OUT/analysis.log"

echo "COMPLETED $(date '+%F %T')" | tee "$STATUS"
