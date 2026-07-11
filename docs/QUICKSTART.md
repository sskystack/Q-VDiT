# Q-VDiT Quick Start

本文给出当前仓库已经跑通的标准 Open-Sora 流程：

```text
Environment & checkpoint
        ↓
FP16 smoke inference
        ↓
Calibration trajectory (50-step DDIM, CFG=4)
        ↓
PTQ reconstruction (W3A8 / W4A6 / CFG-aware W4A6)
        ↓
Quantized inference (100-step DDIM, CFG=4)
        ↓
VBench evaluation
```

## 1. Environment

推荐环境：

- Linux
- Python 3.10
- CUDA 12.1
- PyTorch 2.1.1
- xformers 0.0.23
- NumPy 1.26.x

服务器上已验证的核心版本为 Python 3.10.20、PyTorch 2.1.1+cu121、NumPy 1.26.4。

```bash
conda create -n qvdit python=3.10 -y
conda activate qvdit

conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
  pytorch-cuda=12.1 -c pytorch -c nvidia

pip install -r t2v/requirements_opensora.txt
pip install -r t2v/requirements_qdiff.txt
pip install xformers==0.0.23

# Optional
pip install packaging ninja
pip install flash-attn --no-build-isolation

pip install -e .
pip install -e ./t2v
```

## 2. Model Preparation

### 2.1 Check model paths

运行前检查以下配置中的模型路径：

```text
t2v/configs/opensora/inference/16x512x512.py
t2v/configs/quant/opensora/16x512x512.py
```

需要准备：

- Stable Diffusion VAE：`stabilityai/sd-vae-ft-ema`
- T5 text encoder：`DeepFloyd/t5-v1_1-xxl`
- Open-Sora v1 checkpoint：`OpenSora-v1-HQ-16x512x512.pth`

### 2.2 Split the Open-Sora QKV checkpoint

下载 [OpenSora-v1-HQ-16x512x512.pth](https://huggingface.co/hpcai-tech/Open-Sora/blob/main/OpenSora-v1-HQ-16x512x512.pth)，放到：

```text
logs/split_ckpt/OpenSora-v1-HQ-16x512x512.pth
```

然后执行：

```bash
python t2v/scripts/split_ckpt.py
```

预期输出：

```text
logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth
```

### 2.3 Optional: precompute T5 embeddings

T5-XXL 启动较慢并占用较多显存。固定 prompt 集建议提前生成 embedding：

```bash
python t2v/scripts/precompute_text_embeds.py \
  ./t2v/configs/quant/opensora/16x512x512.py \
  --ckpt_path ./logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth \
  --prompt_path ./t2v/assets/texts/t2v_samples_10.txt \
  --save_path ./t2v/utils_files/text_embeds_10.pth \
  --batch_size 8 \
  --device cuda \
  --t5_dtype fp32
```

后续命令可追加：

```bash
--precompute_text_embeds ./t2v/utils_files/text_embeds_10.pth
```

embedding 必须和 prompt 文件的顺序及数量一致。

## 3. Common Variables

以下命令均从仓库根目录运行：

```bash
export GPU_ID=0
export MODEL_CFG=./t2v/configs/quant/opensora/16x512x512.py
export FP16_CFG=./t2v/configs/opensora/inference/16x512x512.py
export MODEL_CKPT=./logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth
export PROMPTS=./t2v/assets/texts/t2v_samples_10.txt
export CALIB_DIR=./logs_50steps/calib_data_ddim50_cfg4
```

## 0. FP16 Smoke Inference

```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} python t2v/scripts/inference.py ${FP16_CFG} \
  --gpu 0 \
  --ckpt_path ${MODEL_CKPT} \
  --prompt_path ${PROMPTS} \
  --outdir ./logs/fp16_smoke \
  --num_sampling_steps 50 \
  --cfg_scale 4.0 \
  --sampler ddim
```

建议先确认能够生成视频，再开始耗时较长的 PTQ。

## 1. Generate Calibration Data

标准 calibration 设置：

- Open-Sora 16x512x512
- 10 prompts
- 50-step DDIM
- CFG scale 4.0
- seed 42

```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} python t2v/scripts/get_calib_data.py ${MODEL_CFG} \
  --gpu 0 \
  --ckpt_path ${MODEL_CKPT} \
  --prompt_path ${PROMPTS} \
  --data_num 10 \
  --num_sampling_steps 50 \
  --cfg_scale 4.0 \
  --sampler ddim \
  --seed 42 \
  --outdir ${CALIB_DIR} \
  --save_dir ${CALIB_DIR}
```

预期输出：

```text
${CALIB_DIR}/calib_data.pt
${CALIB_DIR}/sample_0.mp4
...
```

需要调试 cond/uncond 输出时，可追加：

```bash
--save_inp_oup
```

需要让不同模型使用完全相同的初始 latent 时，可追加：

```bash
--init_noise_path /path/to/init_noise.pt
```

## 2. PTQ Reconstruction

### W3A8 reproduction

```bash
export EXP_DIR=./logs_50steps/w3a8_ours

CUDA_VISIBLE_DEVICES=${GPU_ID} python t2v/scripts/calib.py ${MODEL_CFG} \
  --gpu 0 \
  --ckpt_path ${MODEL_CKPT} \
  --calib_config ./t2v/configs/quant/opensora/w3a8_ours.yaml \
  --calib_data ${CALIB_DIR}/calib_data.pt \
  --outdir ${EXP_DIR} \
  --part_fp \
  --time_mp_config_weight ./t2v/configs/quant/opensora/mixed_precision/weight_3_mp.yaml \
  --time_mp_config_act ./t2v/configs/quant/opensora/mixed_precision/act_8_mp.yaml
```

W3A8 沿用官方配置中的 `reconstruction_loss_type: relation`，即 MSE + temporal relation loss。

### W4A6 official baseline

```bash
export EXP_DIR=./logs_50steps/w4a6_ours

CUDA_VISIBLE_DEVICES=${GPU_ID} python t2v/scripts/calib.py ${MODEL_CFG} \
  --gpu 0 \
  --ckpt_path ${MODEL_CKPT} \
  --calib_config ./t2v/configs/quant/opensora/w4a6_ours.yaml \
  --calib_data ${CALIB_DIR}/calib_data.pt \
  --outdir ${EXP_DIR} \
  --part_fp \
  --time_mp_config_weight ./t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml \
  --time_mp_config_act ./t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml
```

官方开源 W4A6 配置没有启用 relation loss，默认 reconstruction loss 为 MSE。

### CFG-aware W4A6（推荐实验配置）

当前服务器结果中 `cfg_loss_weight=0.1` 的 Overall Consistency 最好。仓库提供对应的已验证配置：

```text
t2v/configs/quant/opensora/w4a6_ours_cfg_loss01.yaml
```

然后运行：

```bash
export EXP_DIR=./logs_50steps/w4a6_cfg01

CUDA_VISIBLE_DEVICES=${GPU_ID} python t2v/scripts/calib.py ${MODEL_CFG} \
  --gpu 0 \
  --ckpt_path ${MODEL_CKPT} \
  --calib_config ./t2v/configs/quant/opensora/w4a6_ours_cfg_loss01.yaml \
  --calib_data ${CALIB_DIR}/calib_data.pt \
  --outdir ${EXP_DIR} \
  --part_fp \
  --time_mp_config_weight ./t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml \
  --time_mp_config_act ./t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml
```

### Resume interrupted reconstruction

```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} python t2v/scripts/calib.py ${MODEL_CFG} \
  --gpu 0 \
  --ckpt_path ${MODEL_CKPT} \
  --calib_config ./t2v/configs/quant/opensora/w4a6_ours_cfg_loss01.yaml \
  --calib_data ${CALIB_DIR}/calib_data.pt \
  --outdir ${EXP_DIR} \
  --resume_recon_ckpt ${EXP_DIR}/intermediate_ckpts/resume_iter2000.pth \
  --part_fp \
  --time_mp_config_weight ./t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml \
  --time_mp_config_act ./t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml
```

恢复时必须保持 quant config、总迭代数和优化目标一致。

## 3. Quantized Inference

### W4A6 / CFG-aware W4A6

```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} python t2v/scripts/quant_txt2video.py ${MODEL_CFG} \
  --gpu 0 \
  --ckpt_path ${MODEL_CKPT} \
  --outdir ${EXP_DIR} \
  --save_dir ${EXP_DIR}/generated_videos \
  --prompt_path ${PROMPTS} \
  --num_videos 10 \
  --num_sampling_steps 100 \
  --cfg_scale 4.0 \
  --sampler ddim \
  --seed 42 \
  --dataset_type opensora \
  --part_fp \
  --time_mp_config_weight ./t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml \
  --time_mp_config_act ./t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml
```

### W3A8

将 mixed-precision config 替换为：

```bash
--time_mp_config_weight ./t2v/configs/quant/opensora/mixed_precision/weight_3_mp.yaml \
--time_mp_config_act ./t2v/configs/quant/opensora/mixed_precision/act_8_mp.yaml
```

### VBench prompt generation

VBench 要求视频文件名能够匹配 prompt。对每个官方 prompt subset 分别生成：

```bash
export VBENCH_PROMPT=/path/to/vbench_official/overall_consistency.txt
export VIDEO_DIR=${EXP_DIR}/vbench_overall

CUDA_VISIBLE_DEVICES=${GPU_ID} python t2v/scripts/quant_txt2video.py ${MODEL_CFG} \
  --gpu 0 \
  --ckpt_path ${MODEL_CKPT} \
  --outdir ${EXP_DIR} \
  --save_dir ${VIDEO_DIR} \
  --prompt_path ${VBENCH_PROMPT} \
  --prompt_as_path \
  --num_sampling_steps 100 \
  --cfg_scale 4.0 \
  --sampler ddim \
  --seed 42 \
  --dataset_type opensora \
  --part_fp \
  --time_mp_config_weight ./t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml \
  --time_mp_config_act ./t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml
```

大 prompt 集可以使用多卡数据并行生成：

```bash
torchrun --nproc_per_node=4 t2v/scripts/quant_txt2video.py ${MODEL_CFG} \
  --data_parallel \
  --ckpt_path ${MODEL_CKPT} \
  --outdir ${EXP_DIR} \
  --save_dir ${VIDEO_DIR} \
  --prompt_path ${VBENCH_PROMPT} \
  --prompt_as_path \
  --num_sampling_steps 100 \
  --cfg_scale 4.0 \
  --dataset_type opensora \
  --part_fp \
  --time_mp_config_weight ./t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml \
  --time_mp_config_act ./t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml
```

## 4. VBench Evaluation

VBench 建议使用独立环境，按照 [VBench 官方说明](https://github.com/Vchitect/VBench) 安装模型与依赖。

假设 VBench 根目录下存在官方 `evaluate.py`：

### Overall subset

```bash
python evaluate.py \
  --videos_path ${EXP_DIR}/vbench_overall \
  --output_path ${EXP_DIR}/vbench_eval_overall \
  --dimension overall_consistency aesthetic_quality imaging_quality \
  --load_ckpt_from_local True
```

### Subject subset

```bash
python evaluate.py \
  --videos_path ${EXP_DIR}/vbench_subject \
  --output_path ${EXP_DIR}/vbench_eval_subject \
  --dimension subject_consistency dynamic_degree motion_smoothness \
  --load_ckpt_from_local True
```

### Scene subset

```bash
python evaluate.py \
  --videos_path ${EXP_DIR}/vbench_scene \
  --output_path ${EXP_DIR}/vbench_eval_scene \
  --dimension scene background_consistency \
  --load_ckpt_from_local True
```

本项目服务器评测使用三个 prompt subset：

| Subset | 视频数 | 指标 |
|---|---:|---|
| Overall | 93 | Overall / Aesthetic / Imaging |
| Subject | 72 | Subject / Dynamic / Motion |
| Scene | 86 | Scene / Background |

## 5. Output Checklist

完整实验完成后建议至少保留：

```text
experiment_dir/
├── config.yaml
├── opensora_config.py
├── ckpt.pth
├── run.log
├── quant_inference_run.log
├── generated_videos/
└── vbench_eval_*/
    ├── *_eval_results.json
    └── *_full_info.json
```

checkpoint、calibration trajectory 和生成视频通常体积较大，不建议直接提交到 Git。

## 6. Common Problems

### CUDA OOM during reconstruction

- 在 quant config 中设置 `keep_cache_on_cpu: True`。
- 减小 `calib_data.batch_size`。
- 开启 `save_interval`，避免长时间任务中断后从头运行。

### T5 consumes too much memory

- 使用 `precompute_text_embeds.py`。
- 确保 embedding 文件与 prompt 文件完全对应。

### VBench cannot match videos and prompts

- 生成时使用 `--prompt_as_path`。
- 使用 VBench 官方 prompt subset。
- 不要在生成完成后修改视频文件名。

### Quantized inference cannot find config or checkpoint

`quant_txt2video.py` 默认从 `${EXP_DIR}/config.yaml` 和 `${EXP_DIR}/ckpt.pth` 加载量化配置与参数。确认 PTQ 已正常完成，并且 `--outdir` 指向同一个实验目录。
