<div align="center">

# Q-VDiT Reproduction & CFG-aware PTQ

面向 Open-Sora 视频生成 DiT 的低比特量化复现、工程整理与 CFG-aware PTQ 实验

[![Paper](https://img.shields.io/badge/Paper-ICML%202025-b31b1b.svg)](https://arxiv.org/abs/2505.22167)
[![Upstream](https://img.shields.io/badge/Upstream-Q--VDiT-4c8eda.svg)](https://github.com/wlfeng0509/Q-VDiT)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

> [!IMPORTANT]
> 本仓库 fork 自 Q-VDiT 官方开源代码 [wlfeng0509/Q-VDiT](https://github.com/wlfeng0509/Q-VDiT)，不是原论文作者维护的官方仓库。
> 本分支在复现原始 Q-VDiT 的基础上，增加了低显存 PTQ、VBench 批量推理、固定噪声、预计算文本嵌入、重构断点恢复，以及 CFG-aware PTQ reconstruction 等实验功能。

**TL;DR：** 当前已基本复现论文在 Open-Sora 上的 W3A8/W4A6 VBench 结果。新增的 CFG residual loss 在 W4A6 下将服务器复现的 Overall Consistency 从 **24.41** 提升至 **24.77**（`cfg_loss_weight=0.1`），但 Scene Consistency 有所下降，说明该方法目前存在画质/条件引导与场景一致性之间的权衡。

<p align="center">
  <img src="imgs/framework.png" width="94%" alt="Q-VDiT framework">
</p>

## Contents

- [项目定位](#项目定位)
- [当前状态](#当前状态)
- [主要改动](#主要改动)
- [快速开始](#快速开始)
- [复现结果](#复现结果)
- [仓库结构](#仓库结构)
- [文档入口](#文档入口)
- [已知限制](#已知限制)
- [致谢与引用](#致谢与引用)

## 项目定位

Q-VDiT 针对视频生成 Diffusion Transformer 提出两个核心组件：

- **TQE（Token-aware Quantization Estimator）**：从 feature 与 temporal-token 两个维度估计量化误差。
- **TMD（Temporal Maintenance Distillation）**：通过帧间关系分布对齐，维护量化视频的时间一致性。

本仓库的工作分成两部分：

1. **论文复现与工程补全**：跑通 Open-Sora v1.0 的 calibration、PTQ、quantized inference 和 VBench evaluation。
2. **CFG-aware PTQ 探索**：额外对齐 FP/Quant 模型的 conditional-unconditional residual，减轻 CFG 对量化误差的放大。

CFG-aware reconstruction 使用：

```text
L_cfg = ||(Q_cond - Q_uncond) - (FP_cond - FP_uncond)||^2
L_total = L_reconstruction + lambda_cfg * L_cfg + L_round
```

> 官方开源的 `w4a6_ours.yaml` 本身没有设置 `reconstruction_loss_type: relation`，因此 W4A6 默认采用 MSE reconstruction。本仓库的 W4A6 CFG 对比沿用这一官方设置，baseline 与 CFG-aware 实验是同配方对比。

## 当前状态

| 模块 | 状态 | 说明 |
|---|:---:|---|
| Open-Sora v1.0 FP16 inference | ✅ | 支持 16x512x512 视频生成 |
| Calibration trajectory | ✅ | 支持预计算文本嵌入与固定初始噪声 |
| Q-VDiT W3A8 PTQ | ✅ | 已完成 checkpoint、推理与 VBench |
| Q-VDiT W4A6 PTQ | ✅ | 已完成 checkpoint、推理与 VBench |
| CPU reconstruction cache | ✅ | 降低逐 block PTQ 的 GPU 显存占用 |
| VBench prompt batch inference | ✅ | 支持 prompt 文件、prompt 文件名输出和数据并行分片 |
| PTQ checkpoint/resume | ✅ | 保存量化参数、优化器、调度器、随机状态和采样索引 |
| CFG-aware PTQ | 🧪 | `lambda=0.1/0.2` 已完成 VBench，`0.05` 已完成 PTQ 待评测 |
| Sequence-parallel PTQ | ⏸️ | 曾进行适配，当前分支已回滚 |
| Latte reproduction | ⏳ | 保留上游代码，尚未形成与 Open-Sora 同等完整的复现流程 |

## 主要改动

相对官方 Q-VDiT，本分支主要增加：

- **CFG-aware reconstruction loss**：成对采样 cond/uncond calibration sample，并对齐 CFG residual。
- **低显存 PTQ**：允许 reconstruction cache 常驻 CPU，按迭代搬运当前 batch。
- **断点恢复**：长时间 reconstruction 可按固定 interval 保存并恢复。
- **VBench 工作流**：支持大 prompt 集、prompt 分片、prompt-as-filename 与独立输出目录。
- **可复现实验输入**：支持固定 latent noise，便于 FP16/Quant 逐样本公平比较。
- **预计算 T5 embeddings**：避免每次推理加载 T5-XXL，降低启动时间和显存占用。
- **服务器路径与 AutoDL 适配**：整理模型路径、batch size 和 calibration step 设置。

## 快速开始

完整命令见 [Quick Start](docs/QUICKSTART.md)。下面给出最常用的四阶段入口。

### 0. 准备公共路径

```bash
export GPU_ID=0
export MODEL_CFG=./t2v/configs/quant/opensora/16x512x512.py
export MODEL_CKPT=./logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth
export CALIB_DIR=./logs_50steps/calib_data_ddim50_cfg4
export EXP_DIR=./logs_50steps/w4a6_cfg01
```

### 1. Calibration data

```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} python t2v/scripts/get_calib_data.py ${MODEL_CFG} \
  --gpu 0 \
  --ckpt_path ${MODEL_CKPT} \
  --prompt_path ./t2v/assets/texts/t2v_samples_10.txt \
  --data_num 10 \
  --num_sampling_steps 50 \
  --cfg_scale 4.0 \
  --sampler ddim \
  --outdir ${CALIB_DIR} \
  --save_dir ${CALIB_DIR}
```

### 2. PTQ calibration / reconstruction

以下示例运行官方 W4A6 配方并启用服务器已验证的 `cfg_loss_weight=0.1`。完整说明见 [CFG-aware W4A6](docs/QUICKSTART.md#cfg-aware-w4a6推荐实验配置)。

```bash
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

PTQ 完成后主要产物为：

```text
${EXP_DIR}/
├── ckpt.pth
├── config.yaml
├── opensora_config.py
├── qdiff/
├── run.log
└── intermediate_ckpts/       # 当 save_interval > 0 时生成
```

### 3. Quantized inference

```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} python t2v/scripts/quant_txt2video.py ${MODEL_CFG} \
  --gpu 0 \
  --ckpt_path ${MODEL_CKPT} \
  --outdir ${EXP_DIR} \
  --save_dir ${EXP_DIR}/generated_videos \
  --prompt_path ./t2v/assets/texts/t2v_samples_10.txt \
  --num_videos 10 \
  --num_sampling_steps 100 \
  --cfg_scale 4.0 \
  --sampler ddim \
  --dataset_type opensora \
  --part_fp \
  --time_mp_config_weight ./t2v/configs/quant/opensora/mixed_precision/weight_4_mp.yaml \
  --time_mp_config_act ./t2v/configs/quant/opensora/mixed_precision/act_6_mp.yaml
```

### 4. VBench evaluation

在独立的 [VBench](https://github.com/Vchitect/VBench) 环境中运行：

```bash
python evaluate.py \
  --videos_path ${EXP_DIR}/generated_videos \
  --output_path ${EXP_DIR}/vbench_eval \
  --dimension overall_consistency aesthetic_quality imaging_quality \
  --load_ckpt_from_local True
```

论文八项指标需要按 official prompt subset 分别生成和评测，完整命令见 [VBench Evaluation](docs/QUICKSTART.md#4-vbench-evaluation)。

## 复现结果

以下结果来自本项目实验服务器，Open-Sora 采用 100-step DDIM、CFG scale 4.0、seed 42。分数统一乘以 100，与论文表格保持一致。

### 原论文复现

| Setting | Imaging | Aesthetic | Motion | Dynamic | Background | Subject | Scene | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Paper W3A8 | 54.94 | 49.94 | 97.11 | 50.00 | 95.96 | 90.14 | 25.07 | 22.39 |
| **Reproduced W3A8** | **56.60** | **50.23** | **97.63** | 41.67 | **96.35** | **91.26** | 23.40 | **22.78** |
| Paper W4A6 | 57.49 | 55.18 | 96.25 | 68.06 | 95.72 | 87.78 | 38.66 | 25.02 |
| **Reproduced W4A6** | 57.38 | 53.75 | 95.79 | 66.67 | 94.97 | 87.02 | 35.03 | 24.41 |

### CFG-aware W4A6

| Setting | Imaging | Aesthetic | Motion | Dynamic | Background | Subject | Scene | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| W4A6 baseline | 57.38 | 53.75 | **95.79** | 66.67 | 94.97 | 87.02 | **35.03** | 24.41 |
| CFG loss 0.1 | 57.91 | 53.62 | 95.57 | 66.67 | **95.05** | 87.23 | 32.85 | **24.77** |
| CFG loss 0.2 | **58.38** | **53.86** | 95.76 | **68.06** | 94.98 | **87.23** | 31.40 | 24.68 |

当前结论：CFG-aware PTQ 能改善 Imaging、Dynamic、Subject 和 Overall 等指标，但 Scene Consistency 下降。`lambda=0.1` 是目前更均衡的设置。详细实验条件和解读见 [Experiments](docs/EXPERIMENTS.md)。

## 仓库结构

```text
Q-VDiT/
├── qdiff/                         # 量化器、量化层、TQE/TMD 与 reconstruction
│   ├── models/
│   ├── optimization/
│   └── quantizer/
├── t2v/
│   ├── configs/                   # Open-Sora 与量化配置
│   ├── opensora/                  # Open-Sora v1.0 代码
│   ├── scripts/                   # calibration / PTQ / inference 入口
│   └── assets/texts/              # 示例 prompt
├── docs/
│   ├── QUICKSTART.md              # 标准实验命令
│   └── EXPERIMENTS.md             # 当前结果、消融与状态
├── assets/                        # 示例 GIF
├── imgs/                          # README 图片
├── scripts/                       # 服务器环境辅助脚本
└── README.md
```

## 文档入口

- 从零安装并跑通完整流程：[docs/QUICKSTART.md](docs/QUICKSTART.md)
- 查看论文复现与 CFG-aware 结果：[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)
- 查看原论文与官方实现：[Q-VDiT upstream](https://github.com/wlfeng0509/Q-VDiT)
- 查看 Open-Sora v1.0：[hpcaitech/Open-Sora](https://github.com/hpcaitech/Open-Sora)
- 查看 VBench：[Vchitect/VBench](https://github.com/Vchitect/VBench)

## 已知限制

- 当前标准流程仅针对 **Open-Sora v1.0 / STDiT-XL/2 / 16x512x512** 做了完整验证。
- 模型配置中仍包含服务器路径占位或本地绝对路径，首次运行前必须检查 VAE、T5 和 checkpoint 路径。
- PTQ reconstruction 耗时较长，建议开启 CPU cache 与 `save_interval`。
- VBench 官方 prompt 文件和 VBench 模型权重需按 VBench 官方说明单独准备。
- Sequence-parallel PTQ 当前未启用；大 prompt 集可使用 `--data_parallel` 做数据并行生成。
- 服务器 checkpoint、视频和 VBench JSON 体积较大，不直接纳入 Git。

## 致谢与引用

本项目基于以下开源工作：

- [wlfeng0509/Q-VDiT](https://github.com/wlfeng0509/Q-VDiT)
- [hpcaitech/Open-Sora](https://github.com/hpcaitech/Open-Sora)
- [thu-nics/ViDiT-Q](https://github.com/thu-nics/ViDiT-Q/tree/viditq_old)
- [Vchitect/VBench](https://github.com/Vchitect/VBench)

如果使用了 Q-VDiT 方法或官方实现，请引用原论文：

```bibtex
@inproceedings{feng2025qvdit,
  title={Q-VDiT: Towards Accurate Quantization and Distillation of Video-Generation Diffusion Transformers},
  author={Feng, Weilun and Yang, Chuanguang and Qin, Haotong and Li, Xiangqi and Wang, Yu and An, Zhulin and Huang, Libo and Diao, Boyu and Zhao, Zixiang and Xu, Yongjun and Magno, Michele},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning},
  year={2025}
}
```

本仓库沿用项目根目录中的 [MIT License](LICENSE)。
