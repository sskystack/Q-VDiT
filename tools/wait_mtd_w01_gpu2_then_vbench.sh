#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/home/zhouchongtian/quantization/qvdit_flash_bf16
CONDA_SH=/home/zhouchongtian/miniconda3/etc/profile.d/conda.sh
GPU_INDEX=2
POLL_SECONDS=15

CALIB_ROOT="$REPO/logs_fp16_flash/formal_w4a6_mtd_w01_gradprobe_samples10_gpu2_0728"
CALIB_DIR="$CALIB_ROOT/calibration"
CALIB_STATUS="$CALIB_ROOT/status.txt"
CALIB_LOG="$CALIB_ROOT/console.log"
QUANT_CKPT="$CALIB_DIR/ckpt.pth"
FINAL_ITER_CKPT="$CALIB_DIR/ckpt_iter_00010000.pth"

MODEL_CFG="$REPO/t2v/configs/quant/opensora/16x512x512_fp16_flash_100steps.py"
QUANT_CFG="$REPO/t2v/configs/quant/opensora/w4a6_mtd_w01_probe.yaml"
MODEL_CKPT=/home/zhouchongtian/quantization/Q-VDiT/logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth
MP_WEIGHT="$REPO/t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml"
MP_ACT="$REPO/t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml"

SUBJECT_PROMPTS="$REPO/t2v/assets/texts/vbench_official/subject_consistency.txt"
SCENE_PROMPTS="$REPO/t2v/assets/texts/vbench_official/scene.txt"
OVERALL_PROMPTS="$REPO/t2v/assets/texts/vbench_official/overall_consistency.txt"

SUBJECT_EMBEDS="$REPO/logs_bf16_flash/vbench_mtd_iter5000/subject_consistency_embeds.pth"
SCENE_EMBEDS=/home/zhouchongtian/quantization/Q-VDiT/logs_50steps/early_stop_cfg01_proxy/vbench_scene_embeds.pth
OVERALL_EMBEDS=/home/zhouchongtian/quantization/Q-VDiT/logs_50steps/early_stop_cfg01_proxy/vbench_overall_embeds.pth

VBENCH_REPO=/home/zhouchongtian/quantization/eval/Vbench
RUN_ROOT="$REPO/logs_fp16_flash/vbench_mtd_w01_fp16gs_final10000"
STATUS_FILE="$RUN_ROOT/status.txt"
AUTOMATION_LOG="$RUN_ROOT/automation.log"
LOCK_FILE="$REPO/logs_fp16_flash/.wait_mtd_w01_gpu2_then_vbench.lock"

mkdir -p "$RUN_ROOT"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "Another MTD-0.1 VBench automation is already running." >&2
    exit 1
fi
exec > >(tee -a "$AUTOMATION_LOG") 2>&1

stamp() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

fail() {
    stamp "FAILED: $*"
    printf 'FAILED %s\n' "$(date '+%F %T')" > "$STATUS_FILE"
    exit 1
}

require_file() {
    [[ -s "$1" ]] || fail "missing or empty file: $1"
}

gpu_uuid() {
    nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits \
        | awk -F', ' -v gpu_idx="$GPU_INDEX" '$1 == gpu_idx {print $2}'
}

gpu_compute_pids() {
    local uuid=$1
    nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null \
        | awk -F', ' -v uuid="$uuid" '$1 == uuid {print $2}'
}

count_videos() {
    local directory=$1
    find "$directory" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null | wc -l
}

wait_for_calibration() {
    stamp "Waiting for MTD-0.1 calibration to complete"
    printf 'WAITING_FOR_CALIBRATION %s\n' "$(date '+%F %T')" > "$STATUS_FILE"

    while true; do
        if [[ -s "$CALIB_STATUS" ]] && grep -q '^COMPLETED gpu=2 ' "$CALIB_STATUS"; then
            break
        fi
        if ! tmux has-session -t mtd_w01_gpu2 2>/dev/null; then
            fail "calibration tmux ended without a COMPLETED status"
        fi
        sleep "$POLL_SECONDS"
    done

    require_file "$CALIB_LOG"
    require_file "$FINAL_ITER_CKPT"
    require_file "$QUANT_CKPT"
    require_file "$CALIB_DIR/mtd_subterm_gradient_summary.json"
    grep -Fq 'count=10000' "$CALIB_LOG" \
        || fail "calibration log does not contain count=10000"
    grep -Fq 'Saving calibrated quantized DiT model' "$CALIB_LOG" \
        || fail "calibration did not reach final checkpoint saving"
    if grep -Eq 'Traceback|OutOfMemoryError|CUDA out of memory' "$CALIB_LOG"; then
        fail "calibration log contains a Python traceback or CUDA OOM"
    fi
    stamp "Calibration validated successfully: $QUANT_CKPT"

    local uuid
    uuid=$(gpu_uuid)
    [[ -n "$uuid" ]] || fail "cannot resolve GPU $GPU_INDEX UUID"
    while [[ -n "$(gpu_compute_pids "$uuid")" ]]; do
        stamp "GPU $GPU_INDEX occupied by PID(s) $(gpu_compute_pids "$uuid" | paste -sd, -); waiting"
        sleep "$POLL_SECONDS"
    done
    stamp "GPU $GPU_INDEX is idle; starting VBench"
}

run_vbench_subset() {
    local name=$1
    local prompts=$2
    local embeds=$3
    local expected=$4
    local runtime_dir="$RUN_ROOT/${name}_runtime"
    local save_base="$RUN_ROOT/$name"
    local video_dir="${save_base}_opensora"

    require_file "$prompts"
    require_file "$embeds"
    [[ "$(wc -l < "$prompts")" -eq "$expected" ]] \
        || fail "$name prompt count differs from expected $expected"

    if [[ "$(count_videos "$video_dir")" -eq "$expected" ]]; then
        stamp "Skipping completed $name generation ($expected videos)"
        return
    fi

    rm -rf "$runtime_dir" "$video_dir"
    mkdir -p "$runtime_dir"
    printf 'GENERATING_%s %s\n' "${name^^}" "$(date '+%F %T')" > "$STATUS_FILE"
    stamp "Starting $name generation: $expected videos"
    CUDA_VISIBLE_DEVICES="$GPU_INDEX" python t2v/scripts/quant_txt2video.py "$MODEL_CFG" \
        --ckpt_path "$MODEL_CKPT" \
        --calib_config "$QUANT_CFG" \
        --quant_ckpt "$QUANT_CKPT" \
        --outdir "$runtime_dir" \
        --save_dir "$save_base" \
        --prompt_path "$prompts" \
        --precompute_text_embeds "$embeds" \
        --prompt_as_path \
        --num_videos "$expected" \
        --batch_size 1 \
        --num_sampling_steps 100 \
        --cfg_scale 4.0 \
        --sampler ddim \
        --seed 42 \
        --dataset_type opensora \
        --part_fp \
        --time_mp_config_weight "$MP_WEIGHT" \
        --time_mp_config_act "$MP_ACT" \
        2>&1 | tee "$RUN_ROOT/${name}_generation.log"

    [[ "$(count_videos "$video_dir")" -eq "$expected" ]] \
        || fail "$name generation produced $(count_videos "$video_dir")/$expected videos"
    stamp "Completed $name generation"
}

run_eval() {
    local name=$1
    local result_file=$2
    shift 2
    local video_dir="$RUN_ROOT/${name}_opensora"
    local eval_dir="$RUN_ROOT/${name}_eval"

    if [[ -s "$eval_dir/$result_file" ]]; then
        stamp "Skipping completed $name evaluation"
        return
    fi
    rm -rf "$eval_dir"
    mkdir -p "$eval_dir"
    printf 'EVALUATING_%s %s\n' "${name^^}" "$(date '+%F %T')" > "$STATUS_FILE"
    stamp "Starting $name VBench evaluation: $*"
    CUDA_VISIBLE_DEVICES="$GPU_INDEX" python evaluate.py \
        --videos_path "$video_dir" \
        --output_path "$eval_dir" \
        --dimension "$@" \
        --load_ckpt_from_local True \
        2>&1 | tee "$RUN_ROOT/evaluate_${name}.log"
    require_file "$eval_dir/$result_file"
    stamp "Completed $name evaluation"
}

write_summary() {
    export RUN_ROOT QUANT_CKPT
    python - <<'PY'
import csv
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
sources = [
    root / "subject_eval" / "subject_consistency_eval_results.json",
    root / "scene_eval" / "scene_eval_results.json",
    root / "overall_eval" / "overall_consistency_eval_results.json",
]
order = [
    "imaging_quality",
    "aesthetic_quality",
    "motion_smoothness",
    "dynamic_degree",
    "background_consistency",
    "subject_consistency",
    "scene",
    "overall_consistency",
]

scores = {}
for path in sources:
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    for metric, value in result.items():
        scores[metric] = float(value[0] if isinstance(value, list) else value)

missing = [metric for metric in order if metric not in scores]
if missing:
    raise RuntimeError(f"Missing VBench metrics: {missing}")

raw = {metric: scores[metric] for metric in order}
percent = {metric: scores[metric] * 100.0 for metric in order}
summary = {
    "experiment": "FP16_FLASH_MTD_W01_W4A6_DDIM100_CFG4_SEED42",
    "quant_checkpoint": os.environ["QUANT_CKPT"],
    "raw_scores": raw,
    "percentage_scores": percent,
}
(root / "vbench_8metrics_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
)
with (root / "vbench_8metrics_summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["metric", "raw_score", "percentage_score"])
    for metric in order:
        writer.writerow([metric, f"{raw[metric]:.8f}", f"{percent[metric]:.4f}"])

lines = [
    "FP16 FlashAttention MTD-0.1 W4A6 - DDIM 100 - CFG 4.0 - seed 42",
    "",
    f"{'Metric':<28}{'Raw':>12}{'x100':>12}",
    "-" * 52,
]
for metric in order:
    lines.append(f"{metric:<28}{raw[metric]:>12.6f}{percent[metric]:>12.2f}")
(root / "vbench_8metrics_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY
}

trap 'fail "command failed at line $LINENO: $BASH_COMMAND"' ERR

wait_for_calibration

source "$CONDA_SH"
conda activate qvdit
cd "$REPO"
export PYTHONPATH="$REPO:$REPO/t2v"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset QVDIT_MEMORY_PROFILE QVDIT_MEMORY_HISTORY QVDIT_MEMORY_MAX_ENTRIES || true

run_vbench_subset subject "$SUBJECT_PROMPTS" "$SUBJECT_EMBEDS" 72
run_vbench_subset scene "$SCENE_PROMPTS" "$SCENE_EMBEDS" 86
run_vbench_subset overall "$OVERALL_PROMPTS" "$OVERALL_EMBEDS" 93
stamp "All three prompt groups generated"

conda activate vbench
cd "$VBENCH_REPO"
export VBENCH_CACHE_DIR=/home/zhouchongtian/quantization/models/vbench

run_eval subject subject_consistency_eval_results.json \
    subject_consistency dynamic_degree motion_smoothness
run_eval scene scene_eval_results.json \
    scene background_consistency
run_eval overall overall_consistency_eval_results.json \
    overall_consistency aesthetic_quality imaging_quality

write_summary
printf 'COMPLETED %s\n' "$(date '+%F %T')" > "$STATUS_FILE"
stamp "ALL_COMPLETE: inference, 8-metric evaluation, and summary finished"
