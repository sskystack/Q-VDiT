# `new` branch: W4A6 commands on physical GPU 3

Run every command from `/home/zhouchongtian/quantization/new` in the `qvdit` conda environment.

```bash
conda activate qvdit
cd /home/zhouchongtian/quantization/new
export PYTHONPATH="$PWD:$PWD/t2v"
export CUDA_VISIBLE_DEVICES=3

export CALIB_CFG=./t2v/configs/quant/opensora/16x512x512.py
export INFER_CFG=./t2v/configs/quant/opensora/16x512x512_inference.py
export MODEL_CKPT=/home/zhouchongtian/quantization/Q-VDiT/logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth
export PROMPTS=./t2v/assets/texts/t2v_samples_10.txt
export TEXT_EMBEDS=/home/zhouchongtian/quantization/new/logs_50steps/text_embeds_samples_10.pt
export CALIB_DIR=/home/zhouchongtian/quantization/new/logs_50steps/calib_data_ddim50_cfg4
export MP_WEIGHT=./t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml
export MP_ACT=./t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml
```

## Precompute the exact ten prompt embeddings

```bash
python t2v/scripts/precompute_text_embeds.py "$CALIB_CFG" \
  --ckpt_path "$MODEL_CKPT" \
  --prompt_path "$PROMPTS" \
  --save_path "$TEXT_EMBEDS" \
  --batch_size 2
```

## Generate 50-step DDIM, CFG 4.0 calibration data

```bash
python t2v/scripts/get_calib_data.py "$CALIB_CFG" \
  --ckpt_path "$MODEL_CKPT" \
  --prompt_path "$PROMPTS" \
  --precompute_text_embeds "$TEXT_EMBEDS" \
  --data_num 10 \
  --batch_size 1 \
  --num_sampling_steps 50 \
  --cfg_scale 4.0 \
  --sampler ddim \
  --outdir "$CALIB_DIR" \
  --save_dir "$CALIB_DIR"
```

## PTQ calibration

Run one command for each requested configuration.

### 1. Official W4A6 baseline

```bash
export EXP_DIR=/home/zhouchongtian/quantization/new/logs_50steps/w4a6_baseline
python t2v/scripts/calib.py "$CALIB_CFG" \
  --ckpt_path "$MODEL_CKPT" \
  --calib_config ./t2v/configs/quant/opensora/w4a6_baseline.yaml \
  --calib_data "$CALIB_DIR/calib_data.pt" \
  --precompute_text_embeds "$TEXT_EMBEDS" \
  --outdir "$EXP_DIR" \
  --part_fp \
  --time_mp_config_weight "$MP_WEIGHT" \
  --time_mp_config_act "$MP_ACT"
```

### 2. TARQ

```bash
export EXP_DIR=/home/zhouchongtian/quantization/new/logs_50steps/w4a6_tarq
python t2v/scripts/calib.py "$CALIB_CFG" \
  --ckpt_path "$MODEL_CKPT" \
  --calib_config ./t2v/configs/quant/opensora/w4a6_tarq.yaml \
  --calib_data "$CALIB_DIR/calib_data.pt" \
  --precompute_text_embeds "$TEXT_EMBEDS" \
  --outdir "$EXP_DIR" \
  --part_fp \
  --time_mp_config_weight "$MP_WEIGHT" \
  --time_mp_config_act "$MP_ACT"
```

### 3. TARQ + MTD

```bash
export EXP_DIR=/home/zhouchongtian/quantization/new/logs_50steps/w4a6_tarq_mtd
python t2v/scripts/calib.py "$CALIB_CFG" \
  --ckpt_path "$MODEL_CKPT" \
  --calib_config ./t2v/configs/quant/opensora/w4a6_tarq_mtd.yaml \
  --calib_data "$CALIB_DIR/calib_data.pt" \
  --precompute_text_embeds "$TEXT_EMBEDS" \
  --outdir "$EXP_DIR" \
  --part_fp \
  --time_mp_config_weight "$MP_WEIGHT" \
  --time_mp_config_act "$MP_ACT"
```

### 4. TARQ + MTD + TAQ

```bash
export EXP_DIR=/home/zhouchongtian/quantization/new/logs_50steps/w4a6_tarq_mtd_taq
python t2v/scripts/calib.py "$CALIB_CFG" \
  --ckpt_path "$MODEL_CKPT" \
  --calib_config ./t2v/configs/quant/opensora/w4a6_tarq_mtd_taq.yaml \
  --calib_data "$CALIB_DIR/calib_data.pt" \
  --precompute_text_embeds "$TEXT_EMBEDS" \
  --outdir "$EXP_DIR" \
  --part_fp \
  --time_mp_config_weight "$MP_WEIGHT" \
  --time_mp_config_act "$MP_ACT"
```

## 100-step DDIM, CFG 4.0 quantized inference

Set `EXP_DIR` to any of the four directories above and run:

```bash
python t2v/scripts/quant_txt2video.py "$INFER_CFG" \
  --ckpt_path "$MODEL_CKPT" \
  --calib_config "$EXP_DIR/config.yaml" \
  --quant_ckpt "$EXP_DIR/ckpt.pth" \
  --precompute_text_embeds "$TEXT_EMBEDS" \
  --prompt_path "$PROMPTS" \
  --num_videos 10 \
  --batch_size 1 \
  --num_sampling_steps 100 \
  --cfg_scale 4.0 \
  --sampler ddim \
  --outdir "$EXP_DIR" \
  --save_dir "$EXP_DIR/generated_videos_ddim100_cfg4" \
  --dataset_type opensora \
  --part_fp \
  --time_mp_config_weight "$MP_WEIGHT" \
  --time_mp_config_act "$MP_ACT"
```

Because only physical GPU 3 is exposed, the scripts' default `--gpu 0` correctly addresses GPU 3. Do not pass `--gpu 3` together with `CUDA_VISIBLE_DEVICES=3`.
