#!/usr/bin/env bash
set -Eeuo pipefail

if [[ -z "${TMUX:-}" ]]; then
    echo "Run this continuation from a new dedicated tmux session." >&2
    exit 2
fi

REPO=${REPO:-/home/zhouchongtian/quantization/Q-VDiT-mtd-v2-20260810}
GPU_ID=${GPU_ID:-6}
CALIB_RUN_ROOT=${CALIB_RUN_ROOT:-/home/zhouchongtian/quantization/experiments/mtd_v2_epsilon_gpu6_20260824_r1}
VBENCH_RUN_ROOT=${VBENCH_RUN_ROOT:-/home/zhouchongtian/quantization/experiments/mtd_v2_epsilon_gpu6_subject_vbench_20260824_r1}
SUBJECT_PIPELINE_ROOT=${SUBJECT_PIPELINE_ROOT:-/home/zhouchongtian/quantization/experiments/mtd_v2_epsilon_gpu6_subject_pipeline_20260824_r1}
CONTINUATION_ROOT=${CONTINUATION_ROOT:-/home/zhouchongtian/quantization/experiments/mtd_v2_epsilon_gpu6_scene_overall_continuation_20260824_r1}
CALIB_CONFIG=t2v/configs/quant/opensora/w4a6_mtd_v2_epsilon.yaml
QUANT_CKPT=$CALIB_RUN_ROOT/calibration/ckpt.pth
SUBJECT_RESULT=$VBENCH_RUN_ROOT/groups/subject/subject_eval/subject_consistency_eval_results.json
GROUP_RUNNER=$REPO/scripts/run_mtd_v2_vbench_group.sh
EMBED_ROOT=/home/zhouchongtian/quantization/qvdit_flash_bf16/logs_fp16_flash/vbench_official_embeds_fp32t5_seed42
POLL_SECONDS=${POLL_SECONDS:-30}

for path in "$REPO/$CALIB_CONFIG" "$GROUP_RUNNER"; do
    [[ -s "$path" ]] || { echo "missing or empty file: $path" >&2; exit 2; }
done
grep -q 'feature_source: "epsilon"' "$REPO/$CALIB_CONFIG" || { echo "epsilon feature-source check failed" >&2; exit 2; }
grep -q 'temporal_offsets: \[1, 2, 4\]' "$REPO/$CALIB_CONFIG" || { echo "temporal-offset control check failed" >&2; exit 2; }
[[ ! -e "$CONTINUATION_ROOT" ]] || { echo "refusing to reuse continuation root: $CONTINUATION_ROOT" >&2; exit 2; }

mkdir -p "$CONTINUATION_ROOT"
exec > >(tee -a "$CONTINUATION_ROOT/pipeline.log") 2>&1
stamp() { printf '[%s] %s\n' "$(date -Is)" "$*"; }
fail() {
    stamp "FAILED: $*"
    printf 'FAILED %s: %s\n' "$(date -Is)" "$*" > "$CONTINUATION_ROOT/status.txt"
    exit 1
}
trap 'fail "command failed at line $LINENO: $BASH_COMMAND"' ERR

{
    date -Is
    printf 'experiment=MTD_V2_epsilon_transport_scene_overall_continuation\n'
    printf 'depends_on=%s\n' "$SUBJECT_RESULT"
    printf 'gpu=%s\nrepo=%s\ncalibration_root=%s\nvbench_root=%s\n' "$GPU_ID" "$REPO" "$CALIB_RUN_ROOT" "$VBENCH_RUN_ROOT"
    printf 'calibration_config=%s\n' "$CALIB_CONFIG"
    printf 'method=W4A6_TQE_rank1_MTD_V2_epsilon_transport_cycle_disabled\n'
    printf 'vbench=FP16_FlashAttention_FP32T5_DDIM100_CFG4_each_group_seed42_batch1\n'
    printf 'continuation_order=scene,overall\n'
    sha256sum "$REPO/$CALIB_CONFIG" "$REPO/qdiff/mtd.py" "$REPO/qdiff/optimization/block_recon.py"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
} > "$CONTINUATION_ROOT/experiment_manifest.txt"

printf 'WAITING_FOR_SUBJECT %s\n' "$(date -Is)" > "$CONTINUATION_ROOT/status.txt"
stamp "Waiting for completed calibration checkpoint and Subject evaluation"
while [[ ! -s "$QUANT_CKPT" || ! -s "$SUBJECT_RESULT" ]]; do
    if [[ -s "$SUBJECT_PIPELINE_ROOT/status.txt" ]] && grep -q '^FAILED' "$SUBJECT_PIPELINE_ROOT/status.txt"; then
        fail "upstream Subject pipeline failed"
    fi
    sleep "$POLL_SECONDS"
done
stamp "Subject result found; beginning Scene and Overall continuation"

export QVDIT_TQE_RANK=1
for group in scene overall; do
    printf 'VBENCH_%s %s\n' "${group^^}" "$(date -Is)" > "$CONTINUATION_ROOT/status.txt"
    stamp "Starting $group smoke, formal inference, and evaluation"
    REPO="$REPO" QUANT_CKPT="$QUANT_CKPT" COMMON_ROOT="$VBENCH_RUN_ROOT" \
    EMBED_ROOT="$EMBED_ROOT" GROUP="$group" GPU_INDEX="$GPU_ID" WAIT_FOR_GPU_IDLE=1 \
    QUANT_CFG="$REPO/$CALIB_CONFIG" \
    EXPERIMENT_NAME="MTD_V2_epsilon_transport_W4A6_FP16_FlashAttention_DDIM100_CFG4_seed42" \
    QVDIT_TQE_RANK=1 bash "$GROUP_RUNNER"
done

python - "$VBENCH_RUN_ROOT" "$QUANT_CKPT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
sources = [
    root / "groups/subject/subject_eval/subject_consistency_eval_results.json",
    root / "groups/scene/scene_eval/scene_eval_results.json",
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
    for metric, value in json.loads(path.read_text(encoding="utf-8")).items():
        scores[metric] = float(value[0] if isinstance(value, list) else value)
missing = [metric for metric in order if metric not in scores]
if missing:
    raise SystemExit(f"missing VBench metrics: {missing}")
summary = {
    "experiment": "MTD_V2_epsilon_transport_W4A6_FP16_FlashAttention_DDIM100_CFG4_seed42",
    "seed_protocol": "Each group independently starts at seed 42; batch_size=1 uses seed 42+i within the group",
    "controlled_change": "coarse/fine transport feature source x0 -> epsilon; global remains x0",
    "quant_checkpoint": sys.argv[2],
    "percentage_scores": {metric: scores[metric] * 100.0 for metric in order},
}
(root / "vbench_8metrics_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

printf 'COMPLETED %s\n' "$(date -Is)" > "$CONTINUATION_ROOT/status.txt"
stamp "SCENE_OVERALL_COMPLETE"
