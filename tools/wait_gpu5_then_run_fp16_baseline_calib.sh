#!/usr/bin/env bash
set -Eeuo pipefail

GPU_ID=5
POLL_SECONDS=5
REQUIRED_IDLE_CHECKS=3

ROOT=/home/zhouchongtian/quantization/qvdit_flash_bf16
EXP_ROOT="$ROOT/logs_fp16_flash/formal_w4a6_baseline_fp16_gradscaler_samples10_gpu5_0726"
OUTDIR="$EXP_ROOT/calibration"
WAIT_LOG="$EXP_ROOT/wait_gpu5.log"
CONSOLE_LOG="$EXP_ROOT/console.log"
STATUS_FILE="$EXP_ROOT/status.txt"
LOCK_DIR="$EXP_ROOT/.watcher.lock"

MODEL_CFG="$ROOT/t2v/configs/quant/opensora/16x512x512_fp16_flash_50steps.py"
CALIB_CFG="$ROOT/t2v/configs/quant/opensora/w4a6_baseline.yaml"
MODEL_CKPT=/home/zhouchongtian/quantization/Q-VDiT/logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth
CALIB_DATA="$ROOT/logs_fp16_flash/formal_w4a6_mtd_samples10_gpu5_0724/calib_data_ddim50_cfg4/calib_data.pt"
TEXT_EMBEDS=/home/zhouchongtian/quantization/new/logs_50steps/text_embeds_samples_10.pt
MP_WEIGHT="$ROOT/t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml"
MP_ACT="$ROOT/t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml"

mkdir -p "$EXP_ROOT"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "Another GPU5 baseline watcher is already active: $LOCK_DIR" >&2
    exit 1
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

exec > >(tee -a "$WAIT_LOG") 2>&1

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

gpu_compute_pids() {
    nvidia-smi \
        --id="$GPU_ID" \
        --query-compute-apps=pid \
        --format=csv,noheader,nounits 2>/dev/null \
        | sed '/^[[:space:]]*$/d'
}

for required_file in \
    "$MODEL_CFG" \
    "$CALIB_CFG" \
    "$MODEL_CKPT" \
    "$CALIB_DATA" \
    "$TEXT_EMBEDS" \
    "$MP_WEIGHT" \
    "$MP_ACT"; do
    if [[ ! -f "$required_file" ]]; then
        echo "[$(timestamp)] Missing required file: $required_file"
        exit 1
    fi
done

if [[ -e "$OUTDIR/ckpt.pth" || -e "$OUTDIR/run.log" ]]; then
    echo "[$(timestamp)] Refusing to overwrite an existing calibration: $OUTDIR"
    exit 1
fi

echo "[$(timestamp)] Waiting for GPU${GPU_ID} to become idle."
echo "[$(timestamp)] An idle state must persist for $((POLL_SECONDS * REQUIRED_IDLE_CHECKS)) seconds."

while true; do
    idle_checks=0
    while (( idle_checks < REQUIRED_IDLE_CHECKS )); do
        mapfile -t pids < <(gpu_compute_pids)

        if (( ${#pids[@]} == 0 )); then
            ((idle_checks += 1))
            echo "[$(timestamp)] GPU${GPU_ID} idle check ${idle_checks}/${REQUIRED_IDLE_CHECKS}."
        else
            if (( idle_checks > 0 )); then
                echo "[$(timestamp)] GPU${GPU_ID} became busy again; resetting idle checks."
            fi
            idle_checks=0
            echo "[$(timestamp)] GPU${GPU_ID} busy, compute PID(s): ${pids[*]}"
        fi

        if (( idle_checks < REQUIRED_IDLE_CHECKS )); then
            sleep "$POLL_SECONDS"
        fi
    done

    # Close the small race between the final idle poll and process launch.
    mapfile -t final_pids < <(gpu_compute_pids)
    if (( ${#final_pids[@]} == 0 )); then
        break
    fi

    echo "[$(timestamp)] GPU${GPU_ID} was claimed before launch; returning to wait mode."
done

mkdir -p "$OUTDIR"
echo "[$(timestamp)] GPU${GPU_ID} is idle; starting FP16 FlashAttention W4A6 baseline calibration."
echo "RUNNING $(timestamp)" > "$STATUS_FILE"

source /home/zhouchongtian/miniconda3/etc/profile.d/conda.sh
conda activate qvdit
cd "$ROOT"

export PYTHONPATH="$ROOT:$ROOT/t2v"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

set +e
python t2v/scripts/calib.py "$MODEL_CFG" \
    --ckpt_path "$MODEL_CKPT" \
    --calib_config "$CALIB_CFG" \
    --calib_data "$CALIB_DATA" \
    --precompute_text_embeds "$TEXT_EMBEDS" \
    --outdir "$OUTDIR" \
    --part_fp \
    --time_mp_config_weight "$MP_WEIGHT" \
    --time_mp_config_act "$MP_ACT" \
    --use_grad_scaler \
    --reconstruction_checkpoint_interval 500 \
    2>&1 | tee "$CONSOLE_LOG"
exit_code=${PIPESTATUS[0]}
set -e

if (( exit_code == 0 )); then
    echo "COMPLETED $(timestamp)" > "$STATUS_FILE"
    echo "[$(timestamp)] Baseline calibration completed successfully."
else
    echo "FAILED exit_code=$exit_code $(timestamp)" > "$STATUS_FILE"
    echo "[$(timestamp)] Baseline calibration failed with exit code $exit_code."
fi

exit "$exit_code"
