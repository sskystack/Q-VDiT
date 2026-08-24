#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${REPO:-/home/zhouchongtian/quantization/Q-VDiT-mtd-v2-20260810}
CALIB_ROOT=${CALIB_ROOT:-/home/zhouchongtian/quantization/experiments/mtd_v2_gpu6_20260811_retry7/calibration}
QUANT_CKPT=${QUANT_CKPT:-$CALIB_ROOT/ckpt.pth}
COMMON_ROOT=${COMMON_ROOT:-/home/zhouchongtian/quantization/experiments/mtd_v2_gpu456_vbench_20260812}
EMBED_ROOT=${EMBED_ROOT:-/home/zhouchongtian/quantization/qvdit_flash_bf16/logs_fp16_flash/vbench_official_embeds_fp32t5_seed42}
GROUP_RUNNER=${GROUP_RUNNER:-$REPO/scripts/run_mtd_v2_vbench_group.sh}
POLL_SECONDS=${POLL_SECONDS:-60}

mkdir -p "$COMMON_ROOT"
exec > >(tee -a "$COMMON_ROOT/controller.log") 2>&1
stamp() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
fail() { stamp "FAILED: $*"; printf 'FAILED %s: %s\n' "$(date '+%F %T')" "$*" > "$COMMON_ROOT/status.txt"; exit 1; }
trap 'fail "command failed at line $LINENO: $BASH_COMMAND"' ERR

stamp "Waiting for final MTD-v2 calibration checkpoint: $QUANT_CKPT"
printf 'WAITING_FOR_CALIBRATION %s\n' "$(date '+%F %T')" > "$COMMON_ROOT/status.txt"
while [[ ! -s "$QUANT_CKPT" ]]; do
    calib_status=$(tail -n 1 "$CALIB_ROOT/../status.txt" 2>/dev/null || true)
    [[ "$calib_status" != FAILED* ]] || fail "calibration ended in failure"
    sleep "$POLL_SECONDS"
done
stamp "Final checkpoint found; launching three isolated group workflows"

declare -A gpu_for=( [scene]=4 [subject]=5 [overall]=6 )
for group in scene subject overall; do
    session="mtdv2-vbench-${group}-gpu${gpu_for[$group]}"
    if tmux has-session -t "$session" 2>/dev/null; then
        fail "tmux session already exists: $session"
    fi
    tmux new-session -d -s "$session" \
        "REPO='$REPO' QUANT_CKPT='$QUANT_CKPT' COMMON_ROOT='$COMMON_ROOT' EMBED_ROOT='$EMBED_ROOT' GROUP='$group' GPU_INDEX='${gpu_for[$group]}' WAIT_FOR_GPU_IDLE=1 bash '$GROUP_RUNNER'"
    stamp "Started $group on GPU ${gpu_for[$group]} in $session"
done
printf 'GROUPS_RUNNING %s\n' "$(date '+%F %T')" > "$COMMON_ROOT/status.txt"

while true; do
    complete=0
    for group in scene subject overall; do
        group_status=$(tail -n 1 "$COMMON_ROOT/groups/$group/status.txt" 2>/dev/null || true)
        [[ "$group_status" != FAILED* ]] || fail "$group workflow failed: $group_status"
        [[ "$group_status" == COMPLETED* ]] && complete=$((complete + 1))
    done
    [[ "$complete" -eq 3 ]] && break
    sleep "$POLL_SECONDS"
done

python - "$COMMON_ROOT" "$QUANT_CKPT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
sources = [
    root / "groups/scene/scene_eval/scene_eval_results.json",
    root / "groups/subject/subject_eval/subject_consistency_eval_results.json",
    root / "groups/overall/overall_eval/overall_consistency_eval_results.json",
]
order = [
    "imaging_quality", "aesthetic_quality", "motion_smoothness",
    "dynamic_degree", "background_consistency", "subject_consistency",
    "scene", "overall_consistency",
]
scores = {}
for path in sources:
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing result: {path}")
    for metric, value in json.loads(path.read_text()).items():
        scores[metric] = float(value[0] if isinstance(value, list) else value)
missing = [metric for metric in order if metric not in scores]
if missing:
    raise SystemExit(f"missing VBench metrics: {missing}")
summary = {
    "experiment": "MTD_V2_W4A6_FP16_FlashAttention_DDIM100_CFG4_seed42_parallel_gpu456",
    "quant_checkpoint": sys.argv[2],
    "group_gpu_mapping": {"scene": 4, "subject": 5, "overall": 6},
    "percentage_scores": {metric: scores[metric] * 100.0 for metric in order},
}
(root / "vbench_8metrics_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

printf 'COMPLETED %s\n' "$(date '+%F %T')" > "$COMMON_ROOT/status.txt"
stamp "ALL_COMPLETE: parallel GPU 4/5/6 VBench workflow"
