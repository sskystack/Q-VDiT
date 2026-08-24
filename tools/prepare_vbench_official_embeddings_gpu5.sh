#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/zhouchongtian/quantization/qvdit_flash_bf16
OUT="$ROOT/logs_fp16_flash/vbench_official_embeds"
CONFIG="$ROOT/t2v/configs/quant/opensora/16x512x512_fp16_flash_100steps.py"
CKPT=/home/zhouchongtian/quantization/Q-VDiT/logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth
STATUS="$OUT/status.txt"

mkdir -p "$OUT"
exec > >(tee -a "$OUT/prepare.log") 2>&1
trap 'echo "FAILED $(date "+%F %T")" | tee "$STATUS"' ERR

source /home/zhouchongtian/miniconda3/etc/profile.d/conda.sh
conda activate qvdit
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/t2v"
export CUDA_VISIBLE_DEVICES=5

generate() {
    local name=$1 prompt_file=$2 expected=$3
    local output="$OUT/${name}_fp16.pth"
    if [[ -s "$output" && -s "$output.json" ]]; then
        echo "SKIP existing $output"
    else
        echo "GENERATING_${name^^} $(date '+%F %T')" | tee "$STATUS"
        python t2v/scripts/precompute_text_embeds.py "$CONFIG" \
            --ckpt_path "$CKPT" \
            --prompt_path "$prompt_file" \
            --save_path "$output" \
            --batch_size 8 \
            --device cuda \
            --t5_dtype fp16 \
            --device_map none \
            --seed 42
    fi
    python - "$output" "$expected" <<'PY'
import json, sys, torch
path, expected = sys.argv[1], int(sys.argv[2])
payload = torch.load(path, map_location="cpu")
metadata = json.load(open(path + ".json"))
assert payload["y"].shape[0] == expected
assert payload["mask"].shape[0] == expected
assert metadata["prompt_count"] == expected
assert metadata["seed"] == 42
print(path, tuple(payload["y"].shape), tuple(payload["mask"].shape))
PY
}

generate scene "$ROOT/t2v/assets/texts/vbench_official/scene.txt" 86
generate overall "$ROOT/t2v/assets/texts/vbench_official/overall_consistency.txt" 93
echo "COMPLETED $(date '+%F %T')" | tee "$STATUS"
