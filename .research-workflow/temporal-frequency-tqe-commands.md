# Temporal-frequency TQE commands

Run from `/home/zhouchongtian/quantization/qvdit_flash_bf16` on the remote server.

```bash
source /home/zhouchongtian/miniconda3/etc/profile.d/conda.sh
conda activate qvdit
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH="$PWD:$PWD/t2v"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Profiler SHA-256:

```text
33fbd78345413ee09cf9daa175dcabd714947901653e294e824ca4aeaad6bf5c
```

## Smoke

```bash
python tools/profile_temporal_frequency_tqe.py \
  --config t2v/configs/quant/opensora/16x512x512_fp16_flash_100steps.py \
  --calib-config t2v/configs/quant/opensora/w4a6_mtd.yaml \
  --quant-ckpt logs_fp16_flash/formal_w4a6_mtd_gradscaler_samples10_gpu5_0725/calibration/ckpt.pth \
  --prompt-path t2v/assets/texts/t2v_samples_10.txt \
  --text-embeds /home/zhouchongtian/quantization/new/logs_50steps/text_embeds_samples_10.pt \
  --prompt-indices 0,2 \
  --holdout-prompt-index 2 \
  --selected-progress 5 \
  --phase-groups early:5 \
  --selected-blocks 0 \
  --temporal-roles q,k,v,proj \
  --max-trajectories 4 \
  --time-mp-config-weight t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml \
  --time-mp-config-act t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml \
  --output-dir logs_fp16_flash/stage_numeric_profiling/temporal_frequency_smoke_mtd_gpu2_0801
```

## Full MTD

```bash
python tools/profile_temporal_frequency_tqe.py \
  --config t2v/configs/quant/opensora/16x512x512_fp16_flash_100steps.py \
  --calib-config t2v/configs/quant/opensora/w4a6_mtd.yaml \
  --quant-ckpt logs_fp16_flash/formal_w4a6_mtd_gradscaler_samples10_gpu5_0725/calibration/ckpt.pth \
  --prompt-path t2v/assets/texts/t2v_samples_10.txt \
  --text-embeds /home/zhouchongtian/quantization/new/logs_50steps/text_embeds_samples_10.pt \
  --prompt-indices 0,2,6 \
  --holdout-prompt-index 6 \
  --selected-progress 5,15,25,45,65,85,95 \
  --phase-groups 'early:5,15,25;middle:45,65;late:85,95' \
  --selected-blocks 0,9,17,27 \
  --temporal-roles q,k,v,proj \
  --max-trajectories 16 \
  --time-mp-config-weight t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml \
  --time-mp-config-act t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml \
  --output-dir logs_fp16_flash/stage_numeric_profiling/temporal_frequency_mtd_prompts026_gpu2_0801
```

## Full baseline

Use the full MTD command with these substitutions:

```text
--calib-config t2v/configs/quant/opensora/w4a6_baseline.yaml
--quant-ckpt logs_fp16_flash/formal_w4a6_baseline_fp16_gradscaler_samples10_gpu5_0726/calibration/ckpt.pth
--output-dir logs_fp16_flash/stage_numeric_profiling/temporal_frequency_baseline_prompts026_gpu2_0801
```

