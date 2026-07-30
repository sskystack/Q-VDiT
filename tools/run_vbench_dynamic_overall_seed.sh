#!/usr/bin/env bash
set -Ee -o pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <baseline|mtd> <gpu_id> <seed>" >&2
    exit 2
fi

VARIANT="$1"
GPU_ID="$2"
SEED="$3"

CODE_ROOT=/home/zhouchongtian/quantization/qvdit_flash_bf16
VBENCH_ROOT=/home/zhouchongtian/quantization/eval/Vbench
INFER_CFG="$CODE_ROOT/t2v/configs/quant/opensora/16x512x512_fp16_flash_100steps.py"
MODEL_CKPT=/home/zhouchongtian/quantization/Q-VDiT/logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth
MP_WEIGHT="$CODE_ROOT/t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml"
MP_ACT="$CODE_ROOT/t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml"

SUBJECT_PROMPTS="$CODE_ROOT/t2v/assets/texts/vbench_official/subject_consistency.txt"
OVERALL_PROMPTS="$CODE_ROOT/t2v/assets/texts/vbench_official/overall_consistency.txt"
SUBJECT_EMBEDS="$CODE_ROOT/logs_bf16_flash/vbench_mtd_iter5000/subject_consistency_embeds.pth"
OVERALL_EMBEDS=/home/zhouchongtian/quantization/Q-VDiT/logs_50steps/early_stop_cfg01_proxy/vbench_overall_embeds.pth

case "$VARIANT" in
    baseline)
        CALIB_CFG="$CODE_ROOT/t2v/configs/quant/opensora/w4a6_baseline.yaml"
        QUANT_CKPT="$CODE_ROOT/logs_fp16_flash/formal_w4a6_baseline_fp16_gradscaler_samples10_gpu5_0726/calibration/ckpt.pth"
        RUN_ROOT="$CODE_ROOT/logs_fp16_flash/vbench_baseline_fp16gs_seed${SEED}_dynamic_overall"
        ;;
    mtd)
        CALIB_CFG="$CODE_ROOT/t2v/configs/quant/opensora/w4a6_mtd.yaml"
        QUANT_CKPT="$CODE_ROOT/logs_fp16_flash/formal_w4a6_mtd_gradscaler_samples10_gpu5_0725/calibration/ckpt.pth"
        RUN_ROOT="$CODE_ROOT/logs_fp16_flash/vbench_mtdfp16gs_seed${SEED}_dynamic_overall"
        ;;
    *)
        echo "Unknown variant: $VARIANT" >&2
        exit 2
        ;;
esac

for required in \
    "$INFER_CFG" "$MODEL_CKPT" "$MP_WEIGHT" "$MP_ACT" \
    "$SUBJECT_PROMPTS" "$OVERALL_PROMPTS" \
    "$SUBJECT_EMBEDS" "$OVERALL_EMBEDS" \
    "$CALIB_CFG" "$QUANT_CKPT"; do
    [[ -f "$required" ]] || { echo "Missing required file: $required" >&2; exit 1; }
done

if [[ -e "$RUN_ROOT/status.txt" || -e "$RUN_ROOT/subject_generation.log" ]]; then
    echo "Refusing to overwrite existing run: $RUN_ROOT" >&2
    exit 1
fi

mkdir -p "$RUN_ROOT"
echo "RUNNING variant=$VARIANT gpu=$GPU_ID seed=$SEED $(date '+%F %T')" > "$RUN_ROOT/status.txt"

source /home/zhouchongtian/miniconda3/etc/profile.d/conda.sh
export CUDA_VISIBLE_DEVICES="$GPU_ID"

run_inference() {
    local group="$1"
    local prompts="$2"
    local embeds="$3"
    local expected="$4"

    conda activate qvdit
    cd "$CODE_ROOT"
    export PYTHONPATH="$CODE_ROOT:$CODE_ROOT/t2v"

    python t2v/scripts/quant_txt2video.py "$INFER_CFG" \
        --ckpt_path "$MODEL_CKPT" \
        --calib_config "$CALIB_CFG" \
        --quant_ckpt "$QUANT_CKPT" \
        --outdir "$RUN_ROOT/${group}_runtime" \
        --save_dir "$RUN_ROOT/$group" \
        --prompt_path "$prompts" \
        --precompute_text_embeds "$embeds" \
        --prompt_as_path \
        --num_videos "$expected" \
        --batch_size 1 \
        --num_sampling_steps 100 \
        --cfg_scale 4.0 \
        --sampler ddim \
        --seed "$SEED" \
        --dataset_type opensora \
        --part_fp \
        --time_mp_config_weight "$MP_WEIGHT" \
        --time_mp_config_act "$MP_ACT" \
        2>&1 | tee "$RUN_ROOT/${group}_generation.log"

    local generated
    generated=$(find "$RUN_ROOT/${group}_opensora" -maxdepth 1 -type f -name '*.mp4' | wc -l)
    [[ "$generated" -eq "$expected" ]] || {
        echo "$group generation incomplete: expected=$expected actual=$generated" >&2
        return 1
    }
}

run_eval() {
    local group="$1"
    local dimension="$2"

    conda activate vbench
    cd "$VBENCH_ROOT"
    python evaluate.py \
        --videos_path "$RUN_ROOT/${group}_opensora" \
        --output_path "$RUN_ROOT/${group}_eval" \
        --dimension "$dimension" \
        --load_ckpt_from_local True \
        2>&1 | tee "$RUN_ROOT/evaluate_${group}_${dimension}.log"
}

run_inference subject "$SUBJECT_PROMPTS" "$SUBJECT_EMBEDS" 72
run_eval subject dynamic_degree

run_inference overall "$OVERALL_PROMPTS" "$OVERALL_EMBEDS" 93
run_eval overall overall_consistency

conda activate vbench
python - <<PY
import json
from pathlib import Path

root = Path("$RUN_ROOT")
dynamic = json.loads((root / "subject_eval/dynamic_degree_eval_results.json").read_text())["dynamic_degree"][0]
overall = json.loads((root / "overall_eval/overall_consistency_eval_results.json").read_text())["overall_consistency"][0]
summary = {
    "variant": "$VARIANT",
    "seed": int("$SEED"),
    "dynamic_degree": float(dynamic),
    "overall_consistency": float(overall),
}
(root / "two_metric_summary.json").write_text(json.dumps(summary, indent=2))
(root / "two_metric_summary.txt").write_text(
    f"variant={summary['variant']} seed={summary['seed']}\n"
    f"dynamic_degree={summary['dynamic_degree']:.8f} ({summary['dynamic_degree'] * 100:.4f})\n"
    f"overall_consistency={summary['overall_consistency']:.8f} ({summary['overall_consistency'] * 100:.4f})\n"
)
print(json.dumps(summary, indent=2))
PY

echo "COMPLETED variant=$VARIANT gpu=$GPU_ID seed=$SEED $(date '+%F %T')" > "$RUN_ROOT/status.txt"
