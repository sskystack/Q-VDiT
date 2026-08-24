#!/usr/bin/env bash
# Controlled OpenSora MTD-v2 calibration: shared DDIM-50 data then W4A6 PTQ.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhouchongtian/quantization/Q-VDiT-mtd-v2-20260810}"
RUN_ROOT="${RUN_ROOT:-/home/zhouchongtian/quantization/experiments/mtd_v2_20260810}"
PYTHON="${PYTHON:-/home/zhouchongtian/miniconda3/envs/qvdit/bin/python}"
GPU_ID="${GPU_ID:-6}"
WAIT_FOR_GPU_IDLE="${WAIT_FOR_GPU_IDLE:-0}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-10}"
MODEL_CKPT="${MODEL_CKPT:-/home/zhouchongtian/quantization/Q-VDiT/logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth}"
MODEL_CONFIG="t2v/configs/quant/opensora/16x512x512_mtd_v2_fp16_flash.py"
CALIB_CONFIG="${CALIB_CONFIG:-t2v/configs/quant/opensora/w4a6_mtd_v2.yaml}"
PROMPTS="t2v/assets/texts/t2v_samples_10.txt"
WEIGHT_MP="t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml"
ACT_MP="t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml"
# calib.py only checks this non-null argument to avoid re-loading T5.  The
# actual prompt conditioning is already stored in the shared calibration tensor.
TEXT_ENCODER_BYPASS_EMBEDS="t2v/utils_files/text_embeds.pth"
SHARED_CALIB_DATA="${SHARED_CALIB_DATA:-}"

if [[ -e "${RUN_ROOT}" ]]; then
    echo "Refusing to reuse existing formal output root: ${RUN_ROOT}" >&2
    exit 2
fi
test -x "${PYTHON}"
test -s "${MODEL_CKPT}"

wait_for_gpu_idle() {
    [[ "${WAIT_FOR_GPU_IDLE}" == "1" ]] || return 0
    while true; do
        local pids
        pids=$(nvidia-smi --id="${GPU_ID}" --query-compute-apps=pid \
            --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' \
            | paste -sd, -)
        if [[ -z "${pids}" ]]; then
            printf '[%s] GPU%s is idle; beginning formal MTD-v2 calibration.\n' \
                "$(date -Is)" "${GPU_ID}"
            return
        fi
        printf '[%s] GPU%s occupied by compute PID(s): %s; waiting %ss.\n' \
            "$(date -Is)" "${GPU_ID}" "${pids}" "${GPU_POLL_SECONDS}"
        sleep "${GPU_POLL_SECONDS}"
    done
}

wait_for_gpu_idle
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/t2v"
mkdir -p "${RUN_ROOT}/shared_calibration" "${RUN_ROOT}/calibration"

on_error() {
    printf 'FAILED %s\n' "$(date -Is)" > "${RUN_ROOT}/status.txt"
}
trap on_error ERR

{
    date -Is
    printf 'phase=shared_calibration_then_mtd_v2_calibration\n'
    printf 'repo=%s\n' "${REPO_ROOT}"
    printf 'gpu=%s\n' "${GPU_ID}"
    printf 'python=%s\n' "${PYTHON}"
    printf 'model_checkpoint=%s\n' "${MODEL_CKPT}"
    printf 'model_config=%s\n' "${MODEL_CONFIG}"
    printf 'calibration_config=%s\n' "${CALIB_CONFIG}"
    printf 'prompt_file=%s\n' "${PROMPTS}"
    printf 'shared_calibration_tensor=%s\n' "${SHARED_CALIB_DATA:-generated_in_this_run}"
    printf 'calibration_text_encoder=disabled_uses_cached_conditioning\n'
    printf 'calibration_data_sampler=ddim\ncalibration_data_steps=50\ncalibration_data_cfg=4.0\nseed=42\nbatch_size=1\nprompt_count=10\ndtype=fp16\nflash_attention=true\nreconstruction_iterations=10000\ngrad_scaler=true\n'
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
    sha256sum "${MODEL_CONFIG}" "${CALIB_CONFIG}"
} > "${RUN_ROOT}/experiment_manifest.txt"
cp "${MODEL_CONFIG}" "${CALIB_CONFIG}" "${RUN_ROOT}/"

printf 'SHARED_CALIBRATION_STARTED %s\n' "$(date -Is)" > "${RUN_ROOT}/status.txt"
if [[ -n "${SHARED_CALIB_DATA}" ]]; then
    test -s "${SHARED_CALIB_DATA}"
    ln -s "${SHARED_CALIB_DATA}" "${RUN_ROOT}/shared_calibration/calib_data.pt"
    printf 'SHARED_CALIBRATION_REUSED %s\n' "$(date -Is)" | tee "${RUN_ROOT}/pipeline.log"
else
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" t2v/scripts/get_calib_data.py "${MODEL_CONFIG}" \
        --gpu 0 \
        --ckpt_path "${MODEL_CKPT}" \
        --prompt_path "${PROMPTS}" \
        --data_num 10 \
        --batch_size 1 \
        --num_sampling_steps 50 \
        --cfg_scale 4.0 \
        --sampler ddim \
        --seed 42 \
        --outdir "${RUN_ROOT}/shared_calibration" \
        --save_dir "${RUN_ROOT}/shared_calibration" \
        2>&1 | tee "${RUN_ROOT}/pipeline.log"
fi
test -s "${RUN_ROOT}/shared_calibration/calib_data.pt"

printf 'MTD_V2_CALIBRATION_STARTED %s\n' "$(date -Is)" > "${RUN_ROOT}/status.txt"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" t2v/scripts/calib.py "${MODEL_CONFIG}" \
    --gpu 0 \
    --ckpt_path "${MODEL_CKPT}" \
    --calib_config "${CALIB_CONFIG}" \
    --precompute_text_embeds "${TEXT_ENCODER_BYPASS_EMBEDS}" \
    --outdir "${RUN_ROOT}/calibration" \
    --calib_data "${RUN_ROOT}/shared_calibration/calib_data.pt" \
    --part_fp \
    --time_mp_config_weight "${WEIGHT_MP}" \
    --time_mp_config_act "${ACT_MP}" \
    --use_grad_scaler \
    2>&1 | tee -a "${RUN_ROOT}/pipeline.log"
test -s "${RUN_ROOT}/calibration/ckpt.pth"
printf 'MTD_V2_CALIBRATION_COMPLETE %s\n' "$(date -Is)" > "${RUN_ROOT}/status.txt"
