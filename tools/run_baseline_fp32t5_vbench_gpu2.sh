#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${REPO:-/home/zhouchongtian/quantization/qvdit_flash_bf16}
CONDA_SH=${CONDA_SH:-/home/zhouchongtian/miniconda3/etc/profile.d/conda.sh}
GPU_INDEX=${GPU_INDEX:-2}
WAIT_FOR_GPU_IDLE=${WAIT_FOR_GPU_IDLE:-0}
GPU_POLL_SECONDS=${GPU_POLL_SECONDS:-10}

MODEL_CFG="$REPO/t2v/configs/quant/opensora/16x512x512_fp16_flash_100steps.py"
QUANT_CFG=${QUANT_CFG:-"$REPO/t2v/configs/quant/opensora/w4a6_baseline.yaml"}
MODEL_CKPT=/home/zhouchongtian/quantization/Q-VDiT/logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth
QUANT_CKPT=${QUANT_CKPT:-"$REPO/logs_fp16_flash/formal_w4a6_baseline_fp16_gradscaler_samples10_gpu5_0726/calibration/ckpt.pth"}
MP_WEIGHT="$REPO/t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml"
MP_ACT="$REPO/t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml"

PROMPT_ROOT="$REPO/t2v/assets/texts/vbench_official"
SCENE_PROMPTS="$PROMPT_ROOT/scene.txt"
SUBJECT_PROMPTS="$PROMPT_ROOT/subject_consistency.txt"
OVERALL_PROMPTS="$PROMPT_ROOT/overall_consistency.txt"

EMBED_ROOT=${EMBED_ROOT:-"$REPO/logs_fp16_flash/vbench_official_embeds_fp32t5_seed42"}
SCENE_EMBEDS="$EMBED_ROOT/scene_fp32t5.pth"
SUBJECT_EMBEDS="$EMBED_ROOT/subject_fp32t5.pth"
OVERALL_EMBEDS="$EMBED_ROOT/overall_fp32t5.pth"

RUN_ROOT=${RUN_ROOT:-"$REPO/logs_fp16_flash/vbench_baseline_fp32t5_fp16flash_ddim100_cfg4_seed42_0802"}
SMOKE_ROOT="$RUN_ROOT/smoke"
STATUS_FILE="$RUN_ROOT/status.txt"
AUTOMATION_LOG="$RUN_ROOT/automation.log"
LOCK_FILE=${LOCK_FILE:-"$REPO/logs_fp16_flash/.vbench_baseline_fp32t5_gpu2.lock"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"FP16_FlashAttention_W4A6_baseline_with_FP32_T5_embeddings"}

VBENCH_REPO=/home/zhouchongtian/quantization/eval/Vbench
VBENCH_CACHE=/home/zhouchongtian/quantization/models/vbench
VBENCH_BERT_DIR="$VBENCH_CACHE/bert-base-uncased"

mkdir -p "$RUN_ROOT" "$EMBED_ROOT"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "Another FP32-T5 baseline VBench workflow is using $LOCK_FILE" >&2
    exit 1
fi
exec > >(tee -a "$AUTOMATION_LOG") 2>&1

stamp() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

fail() {
    local message=$1
    stamp "FAILED: $message"
    printf 'FAILED %s: %s\n' "$(date '+%F %T')" "$message" > "$STATUS_FILE"
    exit 1
}

require_file() {
    [[ -s "$1" ]] || fail "missing or empty file: $1"
}

count_videos() {
    local directory=$1
    if [[ ! -d "$directory" ]]; then
        printf '0\n'
        return
    fi
    find "$directory" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null | wc -l
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

wait_for_gpu_idle() {
    [[ "$WAIT_FOR_GPU_IDLE" == "1" ]] || return 0
    local uuid pids
    uuid=$(gpu_uuid)
    [[ -n "$uuid" ]] || fail "cannot resolve physical GPU $GPU_INDEX UUID"
    printf 'WAITING_GPU_%s %s\n' "$GPU_INDEX" "$(date '+%F %T')" > "$STATUS_FILE"
    stamp "Waiting for GPU $GPU_INDEX ($uuid) to have no compute processes"
    while true; do
        pids=$(gpu_compute_pids "$uuid" | paste -sd, -)
        if [[ -z "$pids" ]]; then
            stamp "GPU $GPU_INDEX is idle; starting workflow immediately"
            return
        fi
        stamp "GPU $GPU_INDEX occupied by compute PID(s): $pids; polling again in ${GPU_POLL_SECONDS}s"
        sleep "$GPU_POLL_SECONDS"
    done
}

validate_prompt_file() {
    local path=$1
    local expected=$2
    require_file "$path"
    [[ "$(grep -cve '^[[:space:]]*$' "$path")" -eq "$expected" ]] \
        || fail "$path does not contain exactly $expected non-empty prompts"
}

validate_embed() {
    local embeds=$1
    local prompts=$2
    local expected=$3
    python - "$embeds" "$prompts" "$expected" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import torch

embed_path = Path(sys.argv[1])
prompt_path = Path(sys.argv[2])
expected = int(sys.argv[3])
payload = torch.load(embed_path, map_location="cpu")
y = payload["y"]
mask = payload["mask"]
expected_y = (expected, 2, 1, 120, 4096)
expected_mask = (expected, 120)
if tuple(y.shape) != expected_y:
    raise SystemExit(f"{embed_path}: y shape {tuple(y.shape)} != {expected_y}")
if tuple(mask.shape) != expected_mask:
    raise SystemExit(f"{embed_path}: mask shape {tuple(mask.shape)} != {expected_mask}")
if y.dtype != torch.float32:
    raise SystemExit(f"{embed_path}: y dtype {y.dtype} is not torch.float32")
if mask.dtype != torch.int64:
    raise SystemExit(f"{embed_path}: mask dtype {mask.dtype} is not torch.int64")

prompts = [line.strip() for line in prompt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
metadata = {
    "seed": 42,
    "prompt_path": str(prompt_path),
    "prompt_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
    "prompt_count": len(prompts),
    "t5_compute_dtype": "fp32",
    "y_shape": list(y.shape),
    "mask_shape": list(mask.shape),
    "y_dtype": str(y.dtype),
    "mask_dtype": str(mask.dtype),
}
embed_path.with_suffix(embed_path.suffix + ".json").write_text(
    json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(json.dumps(metadata, ensure_ascii=False))
PY
}

generate_embed() {
    local name=$1
    local prompts=$2
    local embeds=$3
    local expected=$4

    if [[ -s "$embeds" ]]; then
        stamp "Validating existing $name FP32-T5 embeddings"
        validate_embed "$embeds" "$prompts" "$expected" \
            || fail "$name embedding validation failed"
        return
    fi

    stamp "Generating $name FP32-T5 embeddings ($expected prompts)"
    printf 'GENERATING_FP32T5_%s %s\n' "${name^^}" "$(date '+%F %T')" > "$STATUS_FILE"
    CUDA_VISIBLE_DEVICES="$GPU_INDEX" python -u t2v/scripts/precompute_text_embeds.py "$MODEL_CFG" \
        --ckpt_path "$MODEL_CKPT" \
        --prompt_path "$prompts" \
        --save_path "$embeds" \
        --batch_size 1 \
        --device cuda \
        --t5_dtype fp32 \
        --device_map auto \
        2>&1 | tee "$RUN_ROOT/embed_${name}.log"
    require_file "$embeds"
    validate_embed "$embeds" "$prompts" "$expected" \
        || fail "$name embedding validation failed after generation"
    stamp "Completed and validated $name FP32-T5 embeddings"
}

validate_video_names() {
    local prompts=$1
    local video_dir=$2
    python - "$prompts" "$video_dir" <<'PY'
import sys
from pathlib import Path

prompt_path = Path(sys.argv[1])
video_dir = Path(sys.argv[2])
expected = {line.strip() + ".mp4" for line in prompt_path.read_text(encoding="utf-8").splitlines() if line.strip()}
actual = {path.name for path in video_dir.glob("*.mp4")}
missing = sorted(expected - actual)
extra = sorted(actual - expected)
if missing or extra:
    raise SystemExit(f"video/prompt mismatch: missing={missing[:5]}, extra={extra[:5]}")
print(f"Validated {len(actual)} video names against {prompt_path}")
PY
}

run_inference() {
    local name=$1
    local prompts=$2
    local embeds=$3
    local expected=$4
    local root=$5
    local runtime_dir="$root/${name}_runtime"
    local save_base="$root/$name"
    local video_dir="${save_base}_opensora"
    local current_count
    current_count=$(count_videos "$video_dir")

    if [[ "$current_count" -eq "$expected" ]]; then
        validate_video_names "$prompts" "$video_dir" \
            || fail "$name completed video set does not match prompts"
        stamp "Skipping completed $name inference ($expected/$expected)"
        return
    fi
    if [[ "$current_count" -ne 0 ]]; then
        fail "$name has a partial video set ($current_count/$expected); refusing an unsafe seed restart"
    fi

    mkdir -p "$runtime_dir"
    stamp "Starting $name inference: seed=42, expected=$expected"
    printf 'INFER_%s %s\n' "${name^^}" "$(date '+%F %T')" > "$STATUS_FILE"
    CUDA_VISIBLE_DEVICES="$GPU_INDEX" python -u t2v/scripts/quant_txt2video.py "$MODEL_CFG" \
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
        2>&1 | tee "$root/${name}_generation.log"

    [[ "$(count_videos "$video_dir")" -eq "$expected" ]] \
        || fail "$name inference produced $(count_videos "$video_dir")/$expected videos"
    validate_video_names "$prompts" "$video_dir" \
        || fail "$name generated video names do not match prompts"
    stamp "Completed and validated $name inference"
}

run_eval() {
    local name=$1
    local result_file=$2
    local root=$3
    shift 3
    local video_dir="$root/${name}_opensora"
    local eval_dir="$root/${name}_eval"

    if [[ -s "$eval_dir/$result_file" ]]; then
        stamp "Skipping completed $name evaluation"
        return
    fi
    mkdir -p "$eval_dir"
    stamp "Starting $name VBench evaluation: $*"
    printf 'EVAL_%s %s\n' "${name^^}" "$(date '+%F %T')" > "$STATUS_FILE"
    CUDA_VISIBLE_DEVICES="$GPU_INDEX" python -u evaluate.py \
        --videos_path "$video_dir" \
        --output_path "$eval_dir" \
        --dimension "$@" \
        --load_ckpt_from_local True \
        2>&1 | tee "$root/evaluate_${name}.log"
    require_file "$eval_dir/$result_file"
    python - "$eval_dir/$result_file" "$@" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = sys.argv[2:]
result = json.loads(path.read_text(encoding="utf-8"))
missing = [key for key in expected if key not in result]
if missing:
    raise SystemExit(f"{path}: missing metrics {missing}")
print(f"Validated {path}: {expected}")
PY
    stamp "Completed and validated $name VBench evaluation"
}

run_smoke_group() {
    local name=$1
    local prompts=$2
    local embeds=$3
    local result_file=$4
    shift 4
    local smoke_prompts="$SMOKE_ROOT/prompts/${name}_two.txt"
    mkdir -p "$SMOKE_ROOT/prompts"
    head -n 2 "$prompts" > "$smoke_prompts"
    run_inference "$name" "$smoke_prompts" "$embeds" 2 "$SMOKE_ROOT"
    conda activate vbench
    cd "$VBENCH_REPO"
    run_eval "$name" "$result_file" "$SMOKE_ROOT" "$@"
    conda activate qvdit
    cd "$REPO"
}

write_summary() {
    python - "$RUN_ROOT" "$QUANT_CKPT" "$EXPERIMENT_NAME" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
quant_checkpoint = sys.argv[2]
sources = [
    root / "scene_eval" / "scene_eval_results.json",
    root / "subject_eval" / "subject_consistency_eval_results.json",
    root / "overall_eval" / "overall_consistency_eval_results.json",
]
order = [
    "imaging_quality", "aesthetic_quality", "motion_smoothness",
    "dynamic_degree", "background_consistency", "subject_consistency",
    "scene", "overall_consistency",
]
scores = {}
for path in sources:
    result = json.loads(path.read_text(encoding="utf-8"))
    for metric, value in result.items():
        scores[metric] = float(value[0] if isinstance(value, list) else value)
missing = [metric for metric in order if metric not in scores]
if missing:
    raise SystemExit(f"missing VBench metrics: {missing}")
summary = {
    "experiment": sys.argv[3],
    "seed_protocol": "Each group starts at seed 42; batch_size=1 uses seed 42+i within the group",
    "quant_checkpoint": quant_checkpoint,
    "percentage_scores": {metric: scores[metric] * 100.0 for metric in order},
}
(root / "vbench_8metrics_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
)
with (root / "vbench_8metrics_summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["metric", "percentage_score"])
    for metric in order:
        writer.writerow([metric, f"{scores[metric] * 100.0:.4f}"])
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
}

trap 'fail "command failed at line $LINENO: $BASH_COMMAND"' ERR

stamp "Preflight: validating files, prompts, environments, and GPU selection"
for path in "$MODEL_CFG" "$QUANT_CFG" "$MODEL_CKPT" "$QUANT_CKPT" "$MP_WEIGHT" "$MP_ACT"; do
    require_file "$path"
done
validate_prompt_file "$SCENE_PROMPTS" 86
validate_prompt_file "$SUBJECT_PROMPTS" 72
validate_prompt_file "$OVERALL_PROMPTS" 93
[[ "$GPU_INDEX" =~ ^[0-9]+$ ]] || fail "GPU_INDEX must be numeric"
nvidia-smi --query-gpu=index --format=csv,noheader,nounits \
    | awk '{$1=$1; print}' | grep -qx "$GPU_INDEX" \
    || fail "GPU $GPU_INDEX does not exist"

wait_for_gpu_idle

source "$CONDA_SH"
conda activate qvdit
cd "$REPO"
export PYTHONPATH="$REPO:$REPO/t2v"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c 'import torch, transformers; print(torch.__version__, transformers.__version__)'

generate_embed scene "$SCENE_PROMPTS" "$SCENE_EMBEDS" 86
generate_embed subject "$SUBJECT_PROMPTS" "$SUBJECT_EMBEDS" 72
generate_embed overall "$OVERALL_PROMPTS" "$OVERALL_EMBEDS" 93
stamp "All three FP32-T5 embedding files generated and validated"

conda activate vbench
cd "$VBENCH_REPO"
export PYTHONPATH="$VBENCH_REPO:${PYTHONPATH:-}"
export VBENCH_CACHE_DIR="$VBENCH_CACHE"
export VBENCH_BERT_DIR
python -c 'import torch, decord; import vbench.scene, vbench.subject_consistency, vbench.overall_consistency; print(torch.__version__)'
conda activate qvdit
cd "$REPO"

stamp "Starting three-group end-to-end smoke validation"
run_smoke_group scene "$SCENE_PROMPTS" "$SCENE_EMBEDS" scene_eval_results.json \
    scene background_consistency
run_smoke_group subject "$SUBJECT_PROMPTS" "$SUBJECT_EMBEDS" subject_consistency_eval_results.json \
    subject_consistency dynamic_degree motion_smoothness
run_smoke_group overall "$OVERALL_PROMPTS" "$OVERALL_EMBEDS" overall_consistency_eval_results.json \
    overall_consistency aesthetic_quality imaging_quality
printf 'SMOKE_COMPLETE %s\n' "$(date '+%F %T')" > "$STATUS_FILE"
stamp "Smoke validation passed for scene, subject, and overall inference+eval"

stamp "Starting formal Scene -> Subject -> Overall workflow"
run_inference scene "$SCENE_PROMPTS" "$SCENE_EMBEDS" 86 "$RUN_ROOT"
conda activate vbench
cd "$VBENCH_REPO"
run_eval scene scene_eval_results.json "$RUN_ROOT" scene background_consistency
conda activate qvdit
cd "$REPO"

run_inference subject "$SUBJECT_PROMPTS" "$SUBJECT_EMBEDS" 72 "$RUN_ROOT"
conda activate vbench
cd "$VBENCH_REPO"
run_eval subject subject_consistency_eval_results.json "$RUN_ROOT" \
    subject_consistency dynamic_degree motion_smoothness
conda activate qvdit
cd "$REPO"

run_inference overall "$OVERALL_PROMPTS" "$OVERALL_EMBEDS" 93 "$RUN_ROOT"
conda activate vbench
cd "$VBENCH_REPO"
run_eval overall overall_consistency_eval_results.json "$RUN_ROOT" \
    overall_consistency aesthetic_quality imaging_quality

write_summary
printf 'COMPLETED %s\n' "$(date '+%F %T')" > "$STATUS_FILE"
stamp "ALL_COMPLETE: FP32-T5 embeddings, baseline inference, VBench eval, and summary"
