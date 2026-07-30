#!/usr/bin/env bash
set -Eeuo pipefail

CODE_ROOT=/home/zhouchongtian/quantization/qvdit_flash_bf16
VBENCH_ROOT=/home/zhouchongtian/quantization/eval/Vbench
CONDA_SH=/home/zhouchongtian/miniconda3/etc/profile.d/conda.sh
GPU_ID=1
POLL_SECONDS=15

BASE_ROOT="$CODE_ROOT/logs_fp16_flash/vbench_baseline_fp16gs_seed123_dynamic_overall"
MTD_ROOT="$CODE_ROOT/logs_fp16_flash/vbench_mtdfp16gs_seed123_dynamic_overall"
OUT_ROOT="$CODE_ROOT/logs_fp16_flash/vbench_seed123_six_metrics_comparison"
STATUS_FILE="$OUT_ROOT/status.txt"
LOG_FILE="$OUT_ROOT/automation.log"
LOCK_FILE="$CODE_ROOT/logs_fp16_flash/.eval_seed123_six_metrics_gpu5.lock"

mkdir -p "$OUT_ROOT"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "Another seed-123 six-metric evaluation is already running." >&2
    exit 1
fi
exec > >(tee -a "$LOG_FILE") 2>&1

stamp() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

fail() {
    stamp "FAILED: $*"
    printf 'FAILED %s\n' "$(date '+%F %T')" > "$STATUS_FILE"
    exit 1
}
trap 'fail "command failed at line $LINENO: $BASH_COMMAND"' ERR

count_videos() {
    find "$1" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null | wc -l
}

gpu_uuid() {
    nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits \
        | awk -F', ' -v gpu_idx="$GPU_ID" '$1 == gpu_idx {print $2}'
}

gpu_compute_pids() {
    local uuid=$1
    nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null \
        | awk -F', ' -v uuid="$uuid" '$1 == uuid {print $2}'
}

wait_for_gpu() {
    local uuid my_uid pid owner_uid
    local -a foreign_pids
    uuid=$(gpu_uuid)
    [[ -n "$uuid" ]] || fail "cannot resolve GPU $GPU_ID UUID"
    my_uid=$(id -u)
    printf 'CHECKING_SHARED_GPU_%s %s\n' "$GPU_ID" "$(date '+%F %T')" > "$STATUS_FILE"
    while true; do
        foreign_pids=()
        while read -r pid; do
            [[ -n "$pid" ]] || continue
            owner_uid=$(ps -o uid= -p "$pid" | xargs)
            [[ "$owner_uid" == "$my_uid" ]] || foreign_pids+=("$pid")
        done < <(gpu_compute_pids "$uuid")
        if (( ${#foreign_pids[@]} == 0 )); then
            stamp "GPU $GPU_ID is free of foreign-user processes; sharing with current user jobs"
            printf 'SHARING_GPU_%s %s\n' "$GPU_ID" "$(date '+%F %T')" > "$STATUS_FILE"
            return
        fi
        stamp "GPU $GPU_ID has foreign PID(s) ${foreign_pids[*]}; waiting"
        sleep "$POLL_SECONDS"
    done
}

validate_inputs() {
    local root
    for root in "$BASE_ROOT" "$MTD_ROOT"; do
        [[ "$(count_videos "$root/subject_opensora")" -eq 72 ]] \
            || fail "$(basename "$root") subject videos are incomplete"
        [[ "$(count_videos "$root/overall_opensora")" -eq 93 ]] \
            || fail "$(basename "$root") overall videos are incomplete"
        [[ -s "$root/subject_eval/dynamic_degree_eval_results.json" ]] \
            || fail "missing Dynamic Degree result under $root"
        [[ -s "$root/overall_eval/overall_consistency_eval_results.json" ]] \
            || fail "missing Overall Consistency result under $root"
    done
}

run_extra_eval() {
    local variant=$1
    local root=$2

    if [[ ! -s "$root/subject_extra_eval/subject_consistency_eval_results.json" ]]; then
        printf 'EVALUATING_%s_SUBJECT %s\n' "${variant^^}" "$(date '+%F %T')" > "$STATUS_FILE"
        rm -rf "$root/subject_extra_eval"
        mkdir -p "$root/subject_extra_eval"
        stamp "Evaluating $variant: subject_consistency + motion_smoothness"
        CUDA_VISIBLE_DEVICES="$GPU_ID" python evaluate.py \
            --videos_path "$root/subject_opensora" \
            --output_path "$root/subject_extra_eval" \
            --dimension subject_consistency motion_smoothness \
            --load_ckpt_from_local True \
            2>&1 | tee "$OUT_ROOT/${variant}_subject_extra_eval.log"
    else
        stamp "Skipping completed $variant subject extra evaluation"
    fi

    if [[ ! -s "$root/overall_extra_eval/aesthetic_quality_eval_results.json" ]]; then
        printf 'EVALUATING_%s_OVERALL %s\n' "${variant^^}" "$(date '+%F %T')" > "$STATUS_FILE"
        rm -rf "$root/overall_extra_eval"
        mkdir -p "$root/overall_extra_eval"
        stamp "Evaluating $variant: aesthetic_quality + imaging_quality"
        CUDA_VISIBLE_DEVICES="$GPU_ID" python evaluate.py \
            --videos_path "$root/overall_opensora" \
            --output_path "$root/overall_extra_eval" \
            --dimension aesthetic_quality imaging_quality \
            --load_ckpt_from_local True \
            2>&1 | tee "$OUT_ROOT/${variant}_overall_extra_eval.log"
    else
        stamp "Skipping completed $variant overall extra evaluation"
    fi
}

write_summary() {
    export BASE_ROOT MTD_ROOT OUT_ROOT
    /home/zhouchongtian/miniconda3/envs/vbench/bin/python - <<'PY'
import csv
import json
import os
from pathlib import Path

roots = {
    "baseline": Path(os.environ["BASE_ROOT"]),
    "mtd": Path(os.environ["MTD_ROOT"]),
}
out = Path(os.environ["OUT_ROOT"])
order = [
    "subject_consistency",
    "dynamic_degree",
    "motion_smoothness",
    "overall_consistency",
    "aesthetic_quality",
    "imaging_quality",
]

def collect(root):
    scores = {}
    candidates = [
        root / "subject_eval",
        root / "overall_eval",
        root / "subject_extra_eval",
        root / "overall_extra_eval",
    ]
    for directory in candidates:
        for path in directory.glob("*eval_results.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            for metric, value in data.items():
                scores[metric] = float(value[0] if isinstance(value, list) else value)
    missing = [metric for metric in order if metric not in scores]
    if missing:
        raise RuntimeError(f"Missing metrics under {root}: {missing}")
    return {metric: scores[metric] for metric in order}

scores = {name: collect(root) for name, root in roots.items()}
delta = {metric: scores["mtd"][metric] - scores["baseline"][metric] for metric in order}
summary = {
    "seed": 123,
    "metrics": order,
    "baseline": scores["baseline"],
    "mtd": scores["mtd"],
    "mtd_minus_baseline": delta,
}
(out / "six_metrics_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
)
with (out / "six_metrics_summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["metric", "baseline", "mtd", "mtd_minus_baseline", "delta_percentage_points"])
    for metric in order:
        writer.writerow([
            metric,
            f"{scores['baseline'][metric]:.8f}",
            f"{scores['mtd'][metric]:.8f}",
            f"{delta[metric]:.8f}",
            f"{delta[metric] * 100:.4f}",
        ])

lines = [
    "VBench seed 123: Baseline vs MTD",
    "",
    f"{'Metric':<26}{'Baseline':>12}{'MTD':>12}{'Delta(pp)':>12}",
    "-" * 62,
]
for metric in order:
    lines.append(
        f"{metric:<26}{scores['baseline'][metric] * 100:>12.4f}"
        f"{scores['mtd'][metric] * 100:>12.4f}{delta[metric] * 100:>12.4f}"
    )
(out / "six_metrics_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY
}

validate_inputs
wait_for_gpu

export PYTHONPATH="${PYTHONPATH:-}"
source "$CONDA_SH"
conda activate vbench
cd "$VBENCH_ROOT"
export VBENCH_CACHE_DIR=/home/zhouchongtian/quantization/models/vbench

run_extra_eval baseline "$BASE_ROOT"
run_extra_eval mtd "$MTD_ROOT"
write_summary

printf 'COMPLETED %s\n' "$(date '+%F %T')" > "$STATUS_FILE"
stamp "ALL_COMPLETE: seed-123 six-metric comparison finished"
