#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${REPO:-/home/zhouchongtian/quantization/Q-VDiT-mtd-v2-20260810}
GPU_ID=${GPU_ID:-5}
CALIB_RUN_ROOT=${CALIB_RUN_ROOT:-/home/zhouchongtian/quantization/experiments/mtd_v2_no_dt4_gpu5_20260819_r1}
VBENCH_RUN_ROOT=${VBENCH_RUN_ROOT:-/home/zhouchongtian/quantization/experiments/mtd_v2_no_dt4_gpu5_vbench_20260819_r1}
PIPELINE_ROOT=${PIPELINE_ROOT:-/home/zhouchongtian/quantization/experiments/mtd_v2_no_dt4_gpu5_pipeline_20260819_r1}
SHARED_CALIB_DATA=${SHARED_CALIB_DATA:-/home/zhouchongtian/quantization/experiments/mtd_v2_20260810_retry1/shared_calibration/calib_data.pt}
CALIB_CONFIG=t2v/configs/quant/opensora/w4a6_mtd_v2_no_dt4.yaml
QUANT_CKPT=$CALIB_RUN_ROOT/calibration/ckpt.pth
CALIB_RUNNER=$REPO/scripts/run_mtd_v2_formal_calibration_gpu6.sh
GROUP_RUNNER=$REPO/scripts/run_mtd_v2_vbench_group.sh
EMBED_ROOT=/home/zhouchongtian/quantization/qvdit_flash_bf16/logs_fp16_flash/vbench_official_embeds_fp32t5_seed42

for path in "$SHARED_CALIB_DATA" "$REPO/$CALIB_CONFIG" "$CALIB_RUNNER" "$GROUP_RUNNER"; do
    [[ -s "$path" ]] || { echo "missing or empty file: $path" >&2; exit 2; }
done
grep -q 'temporal_offsets: \[1, 2\]' "$REPO/$CALIB_CONFIG" || { echo "no-dt4 config check failed" >&2; exit 2; }
! grep -q 'cycle_enabled: true' "$REPO/$CALIB_CONFIG" || { echo "cycle must be disabled" >&2; exit 2; }
for root in "$PIPELINE_ROOT" "$CALIB_RUN_ROOT" "$VBENCH_RUN_ROOT"; do
    [[ ! -e "$root" ]] || { echo "refusing to reuse output root: $root" >&2; exit 2; }
done

mkdir -p "$PIPELINE_ROOT"
exec > >(tee -a "$PIPELINE_ROOT/pipeline.log") 2>&1
stamp() { printf '[%s] %s\n' "$(date -Is)" "$*"; }
fail() { stamp "FAILED: $*"; printf 'FAILED %s: %s\n' "$(date -Is)" "$*" > "$PIPELINE_ROOT/status.txt"; exit 1; }
trap 'fail "command failed at line $LINENO: $BASH_COMMAND"' ERR

{
    date -Is
    printf 'experiment=MTD_V2_without_delta_t_4\n'
    printf 'controlled_change=temporal_offsets_[1,2,4]_to_[1,2]\n'
    printf 'cycle_enabled=false\nTQE_rank=1\n'
    printf 'gpu=%s\nrepo=%s\ncalibration_root=%s\nvbench_root=%s\n' "$GPU_ID" "$REPO" "$CALIB_RUN_ROOT" "$VBENCH_RUN_ROOT"
    printf 'shared_calibration=%s\ncalibration_config=%s\n' "$SHARED_CALIB_DATA" "$CALIB_CONFIG"
    printf 'calibration=FP16_FlashAttention_W4A6_DDIM50_CFG4_seed42_samples10_iters10000_GradScaler\n'
    printf 'vbench=FP16_FlashAttention_FP32T5_DDIM100_CFG4_each_group_seed42_batch1\n'
    printf 'formal_order=subject,scene,overall\n'
    sha256sum "$REPO/$CALIB_CONFIG" "$SHARED_CALIB_DATA"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
} > "$PIPELINE_ROOT/experiment_manifest.txt"

export QVDIT_TQE_RANK=1
printf 'CALIBRATION_PENDING %s\n' "$(date -Is)" > "$PIPELINE_ROOT/status.txt"
stamp "Starting controlled MTD-v2 no-dt4 calibration on GPU $GPU_ID"
GPU_ID="$GPU_ID" WAIT_FOR_GPU_IDLE=1 GPU_POLL_SECONDS=30 \
RUN_ROOT="$CALIB_RUN_ROOT" SHARED_CALIB_DATA="$SHARED_CALIB_DATA" CALIB_CONFIG="$CALIB_CONFIG" \
bash "$CALIB_RUNNER"
[[ -s "$QUANT_CKPT" ]] || fail "final calibration checkpoint is missing"
printf 'CALIBRATION_COMPLETE_VBENCH_PENDING %s\n' "$(date -Is)" > "$PIPELINE_ROOT/status.txt"

mkdir -p "$VBENCH_RUN_ROOT"
for group in subject scene overall; do
    printf 'VBENCH_%s %s\n' "${group^^}" "$(date -Is)" > "$PIPELINE_ROOT/status.txt"
    stamp "Starting $group smoke, formal inference, and evaluation"
    REPO="$REPO" QUANT_CKPT="$QUANT_CKPT" COMMON_ROOT="$VBENCH_RUN_ROOT" \
    EMBED_ROOT="$EMBED_ROOT" GROUP="$group" GPU_INDEX="$GPU_ID" WAIT_FOR_GPU_IDLE=1 \
    QUANT_CFG="$REPO/$CALIB_CONFIG" \
    EXPERIMENT_NAME="MTD_V2_no_dt4_W4A6_FP16_FlashAttention_DDIM100_CFG4_seed42" \
    QVDIT_TQE_RANK=1 bash "$GROUP_RUNNER"
done

python - "$VBENCH_RUN_ROOT" "$QUANT_CKPT" <<'PY'
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
    "experiment": "MTD_V2_no_dt4_W4A6_FP16_FlashAttention_DDIM100_CFG4_seed42",
    "seed_protocol": "Each group starts at seed 42; batch_size=1 uses seed 42+i within the group",
    "controlled_change": "temporal_offsets [1,2,4] -> [1,2]",
    "quant_checkpoint": sys.argv[2],
    "percentage_scores": {metric: scores[metric] * 100.0 for metric in order},
}
(root / "vbench_8metrics_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

printf 'COMPLETED %s\n' "$(date -Is)" > "$PIPELINE_ROOT/status.txt"
stamp "ALL_COMPLETE"
