#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/zhouchongtian/quantization/qvdit_flash_bf16
VBENCH_ROOT=/home/zhouchongtian/quantization/eval/Vbench
OUT="$ROOT/logs_fp16_flash/vbench_fp16_reference_seed42_motion_profile_0731"
FP16_RUNTIME="$OUT/fp16_runtime"
FP16_SAVE="$OUT/fp16_subject"
FP16_VIDEOS="${FP16_SAVE}_opensora"
FP16_EVAL="$OUT/fp16_eval"
BASELINE_ROOT="$ROOT/logs_fp16_flash/vbench_baseline_fp16gs_final10000"
MTD_ROOT="$ROOT/logs_fp16_flash/vbench_mtdfp16gs_final10000"
BASELINE_TEMPORAL="$OUT/baseline_temporal_eval"
MTD_TEMPORAL="$OUT/mtd_temporal_eval"
FP16_TEMPORAL_CUSTOM="$OUT/fp16_temporal_custom_eval"
BASELINE_TEMPORAL_CUSTOM="$OUT/baseline_temporal_custom_eval"
MTD_TEMPORAL_CUSTOM="$OUT/mtd_temporal_custom_eval"
ANALYSIS="$OUT/analysis"
STATUS="$OUT/status.txt"

CONFIG="$ROOT/t2v/configs/quant/opensora/16x512x512_fp16_flash_100steps.py"
MODEL_CKPT=/home/zhouchongtian/quantization/Q-VDiT/logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth
CALIB_CONFIG="$ROOT/t2v/configs/quant/opensora/w4a6_baseline.yaml"
QUANT_CKPT="$ROOT/logs_fp16_flash/formal_w4a6_baseline_fp16_gradscaler_samples10_gpu5_0726/calibration/ckpt.pth"
PROMPTS="$ROOT/t2v/assets/texts/vbench_official/subject_consistency.txt"
EMBEDS="$ROOT/logs_bf16_flash/vbench_mtd_iter5000/subject_consistency_embeds.pth"
MP_WEIGHT="$ROOT/t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml"
MP_ACT="$ROOT/t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml"

mkdir -p "$OUT"
exec > >(tee -a "$OUT/automation.log") 2>&1

fail() {
    echo "FAILED $* $(date '+%F %T')" | tee "$STATUS"
    exit 1
}
trap 'fail "line=$LINENO command=$BASH_COMMAND"' ERR

for required in \
    "$CONFIG" "$MODEL_CKPT" "$CALIB_CONFIG" "$QUANT_CKPT" \
    "$PROMPTS" "$EMBEDS" "$MP_WEIGHT" "$MP_ACT"; do
    [[ -f "$required" ]] || fail "missing=$required"
done

source /home/zhouchongtian/miniconda3/etc/profile.d/conda.sh
export CUDA_VISIBLE_DEVICES=2

if [[ ! -s "$FP16_RUNTIME/experiment_b_metadata.json" ]]; then
    echo "GENERATING_FP16_REFERENCE $(date '+%F %T')" | tee "$STATUS"
    conda activate qvdit
    cd "$ROOT"
    export PYTHONPATH="$ROOT:$ROOT/t2v"
    python t2v/scripts/quant_txt2video.py "$CONFIG" \
        --ckpt_path "$MODEL_CKPT" \
        --calib_config "$CALIB_CONFIG" \
        --quant_ckpt "$QUANT_CKPT" \
        --outdir "$FP16_RUNTIME" \
        --save_dir "$FP16_SAVE" \
        --prompt_path "$PROMPTS" \
        --precompute_text_embeds "$EMBEDS" \
        --prompt_as_path \
        --num_videos 72 \
        --batch_size 1 \
        --num_sampling_steps 100 \
        --cfg_scale 4.0 \
        --sampler ddim \
        --seed 42 \
        --dataset_type opensora \
        --skip_quant_weight \
        --skip_quant_act \
        --part_fp \
        --time_mp_config_weight "$MP_WEIGHT" \
        --time_mp_config_act "$MP_ACT" \
        2>&1 | tee "$OUT/fp16_generation.log"
fi

generated=$(find "$FP16_VIDEOS" -maxdepth 1 -type f -name '*.mp4' | wc -l)
[[ "$generated" -eq 72 ]] || fail "fp16_video_count=$generated expected=72"

conda activate vbench
cd "$VBENCH_ROOT"
export VBENCH_CACHE_DIR=/home/zhouchongtian/quantization/models/vbench

if [[ ! -s "$FP16_EVAL/dynamic_degree_eval_results.json" || \
      ! -s "$FP16_EVAL/temporal_flickering_eval_results.json" ]]; then
    echo "EVALUATING_FP16_REFERENCE $(date '+%F %T')" | tee "$STATUS"
    rm -rf "$FP16_EVAL"
    mkdir -p "$FP16_EVAL"
    python evaluate.py \
        --videos_path "$FP16_VIDEOS" \
        --output_path "$FP16_EVAL" \
        --dimension dynamic_degree motion_smoothness temporal_flickering subject_consistency \
        --load_ckpt_from_local True \
        2>&1 | tee "$OUT/fp16_evaluation.log"
fi

if [[ ! -s "$BASELINE_TEMPORAL/temporal_flickering_eval_results.json" ]]; then
    echo "EVALUATING_BASELINE_TEMPORAL $(date '+%F %T')" | tee "$STATUS"
    rm -rf "$BASELINE_TEMPORAL"
    mkdir -p "$BASELINE_TEMPORAL"
    python evaluate.py \
        --videos_path "$BASELINE_ROOT/subject_opensora" \
        --output_path "$BASELINE_TEMPORAL" \
        --dimension temporal_flickering \
        --load_ckpt_from_local True \
        2>&1 | tee "$OUT/baseline_temporal_evaluation.log"
fi

if [[ ! -s "$MTD_TEMPORAL/temporal_flickering_eval_results.json" ]]; then
    echo "EVALUATING_MTD_TEMPORAL $(date '+%F %T')" | tee "$STATUS"
    rm -rf "$MTD_TEMPORAL"
    mkdir -p "$MTD_TEMPORAL"
    python evaluate.py \
        --videos_path "$MTD_ROOT/subject_opensora" \
        --output_path "$MTD_TEMPORAL" \
        --dimension temporal_flickering \
        --load_ckpt_from_local True \
        2>&1 | tee "$OUT/mtd_temporal_evaluation.log"
fi

run_custom_temporal() {
    local videos=$1
    local output=$2
    local log=$3
    if [[ -s "$output/temporal_flickering_eval_results.json" ]]; then
        return
    fi
    mkdir -p "$output"
    python evaluate.py \
        --videos_path "$videos" \
        --output_path "$output" \
        --dimension temporal_flickering \
        --load_ckpt_from_local True \
        --custom_input \
        2>&1 | tee "$log"
}

run_custom_temporal "$FP16_VIDEOS" "$FP16_TEMPORAL_CUSTOM" \
    "$OUT/fp16_temporal_custom_evaluation.log"
run_custom_temporal "$BASELINE_ROOT/subject_opensora" "$BASELINE_TEMPORAL_CUSTOM" \
    "$OUT/baseline_temporal_custom_evaluation.log"
run_custom_temporal "$MTD_ROOT/subject_opensora" "$MTD_TEMPORAL_CUSTOM" \
    "$OUT/mtd_temporal_custom_evaluation.log"

mkdir -p "$ANALYSIS"
python "$ROOT/tools/analyze_fp16_motion_reference.py" \
    --fp16-eval "$FP16_EVAL" "$FP16_TEMPORAL_CUSTOM" \
    --baseline-eval "$BASELINE_ROOT/subject_eval" "$BASELINE_TEMPORAL_CUSTOM" \
    --mtd-eval "$MTD_ROOT/subject_eval" "$MTD_TEMPORAL_CUSTOM" \
    --output "$ANALYSIS" \
    2>&1 | tee "$OUT/analysis.log"

echo "COMPLETED $(date '+%F %T')" | tee "$STATUS"
