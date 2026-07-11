# Reproduction and Experiment Status

本文记录当前仓库已经完成的 Q-VDiT 复现、CFG-aware PTQ 实验及其限制。

## 1. Evaluation Protocol

当前 Open-Sora VBench 结果使用：

- Model：Open-Sora v1.0 `STDiT-XL/2`
- Resolution：16 frames, 512x512
- Sampler：DDIM
- Inference steps：100
- CFG scale：4.0
- Seed：42
- W3A8 calibration：50 selected steps，CPU cache
- W4A6 calibration：50 selected steps，CPU cache
- VBench：Overall / Subject / Scene 三组官方 prompt subset

三个 subset 分别包含 93、72、86 个视频。表格中的分数乘以 100，与 Q-VDiT 论文表格保持一致。

## 2. Q-VDiT Reproduction

### W3A8

| Metric | Paper | Reproduced | Delta |
|---|---:|---:|---:|
| Imaging Quality | 54.94 | 56.60 | +1.66 |
| Aesthetic Quality | 49.94 | 50.23 | +0.29 |
| Motion Smoothness | 97.11 | 97.63 | +0.52 |
| Dynamic Degree | 50.00 | 41.67 | -8.33 |
| Background Consistency | 95.96 | 96.35 | +0.39 |
| Subject Consistency | 90.14 | 91.26 | +1.12 |
| Scene Consistency | 25.07 | 23.40 | -1.67 |
| Overall Consistency | 22.39 | 22.78 | +0.39 |

W3A8 的整体复现较好，八项指标中六项达到或超过论文报告值。主要差异集中在 Dynamic Degree 和 Scene Consistency。

### W4A6

| Metric | Paper | Reproduced | Delta |
|---|---:|---:|---:|
| Imaging Quality | 57.49 | 57.38 | -0.11 |
| Aesthetic Quality | 55.18 | 53.75 | -1.43 |
| Motion Smoothness | 96.25 | 95.79 | -0.46 |
| Dynamic Degree | 68.06 | 66.67 | -1.39 |
| Background Consistency | 95.72 | 94.97 | -0.75 |
| Subject Consistency | 87.78 | 87.02 | -0.76 |
| Scene Consistency | 38.66 | 35.03 | -3.63 |
| Overall Consistency | 25.02 | 24.41 | -0.61 |

W4A6 大部分指标接近论文结果，Scene Consistency 差距相对明显。

## 3. Official W4A6 Configuration Note

Q-VDiT 官方仓库当前 `main`（核对 SHA `83fa54fbd67d831c8d8c7769406775d8006efffa`）中的 `w4a6_ours.yaml` 没有设置：

```yaml
reconstruction_loss_type: relation
```

`LossFunction` 的默认值为 `mse`，因此官方 W4A6 开源配置实际采用 MSE reconstruction。相比之下，官方 W3A8、W4A8、W6A6、W8A8 配置显式启用了 relation loss。

本仓库的 W4A6 baseline 和 CFG-aware 实验均沿用官方 W4A6 的 MSE 配方，因此二者是同设置对比。`relation + CFG residual` 可以作为后续消融，但不属于复现官方 W4A6 所必需的配置。

## 4. CFG-aware PTQ

### Motivation

Classifier-free guidance 的输出可写为：

```text
epsilon_cfg = epsilon_uncond + scale * (epsilon_cond - epsilon_uncond)
```

cond/uncond residual 中的量化误差会被 CFG scale 放大。本仓库增加：

```text
L_cfg = ||(Q_cond - Q_uncond) - (FP_cond - FP_uncond)||^2
```

reconstruction 时使用成对的 cond/uncond calibration sample，避免随机 batch 破坏 residual 对应关系。

### Experiment Status

| cfg_loss_weight | PTQ | Final checkpoint | VBench | Note |
|---:|:---:|:---:|:---:|---|
| 0.05 | ✅ | ✅ | ⏳ | 已完成 10,000 iterations，待评测 |
| 0.10 | ✅ | ✅ | ✅ | 当前 Overall 最好 |
| 0.20 | ✅ | ✅ | ✅ | 由 iteration 2,000 断点恢复完成 |
| 0.50 | ⏸️ | ❌ | ❌ | 早期实验，仅运行约 100 iterations |

### VBench Results

| Setting | Imaging | Aesthetic | Motion | Dynamic | Background | Subject | Scene | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| W4A6 baseline | 57.38 | 53.75 | **95.79** | 66.67 | 94.97 | 87.02 | **35.03** | 24.41 |
| CFG loss 0.1 | 57.91 | 53.62 | 95.57 | 66.67 | **95.05** | 87.23 | 32.85 | **24.77** |
| CFG loss 0.2 | **58.38** | **53.86** | 95.76 | **68.06** | 94.98 | **87.23** | 31.40 | 24.68 |

### Delta over W4A6 baseline

| Metric | CFG 0.1 | CFG 0.2 |
|---|---:|---:|
| Imaging Quality | +0.53 | +1.00 |
| Aesthetic Quality | -0.13 | +0.11 |
| Motion Smoothness | -0.21 | -0.03 |
| Dynamic Degree | +0.00 | +1.39 |
| Background Consistency | +0.08 | +0.01 |
| Subject Consistency | +0.20 | +0.21 |
| Scene Consistency | -2.18 | -3.63 |
| Overall Consistency | +0.36 | +0.27 |

### Current Interpretation

- `lambda=0.1` 在当前结果中最均衡，并取得最高 Overall Consistency。
- `lambda=0.2` 更偏向 Imaging Quality 和 Dynamic Degree。
- CFG-aware loss 对条件引导和视觉质量有正向作用，但 Scene Consistency 明显下降。
- 当前结果说明方法存在真实的指标权衡，不能描述为在所有维度全面优于 baseline。

## 5. Engineering Validation

除指标结果外，服务器实验还验证了以下工程功能：

- CPU reconstruction cache 能够支撑 W3A8/W4A6 calibration。
- `save_interval=1000` 能持续保存量化参数和完整 resume state。
- CFG loss 0.2 从 iteration 2,000 成功恢复并完成 10,000 iterations。
- 固定初始噪声可用于 FP16/Quant cond-uncond branch 观察。
- VBench prompt-as-filename 批量生成与三个 subset 的独立评测已跑通。

## 6. Artifacts

服务器实验目录中包含：

- calibration trajectory；
- split Open-Sora checkpoint；
- final quantized checkpoint；
- iteration-level checkpoint 与 resume state；
- quantization config 和运行时代码快照；
- generated MP4 videos；
- VBench `*_eval_results.json` 与 `*_full_info.json`。

这些文件总量约 33GB，不直接提交到 Git。每个实验目录应保留 `config.yaml`、`run.log` 和评测 JSON，以便后续追踪。

## 7. Next Experiments

优先级建议：

1. 完成 `cfg_loss_weight=0.05` 的完整 VBench 评测。
2. 对 baseline、0.05、0.1、0.2 使用更多 seed，报告均值和方差。
3. 检查 Scene Consistency 下降集中在哪些 prompt/category。
4. 将 CFG residual loss 从固定权重扩展为 timestep-aware 权重。
5. 可选评估 `relation + CFG residual`，判断 temporal relation 是否能缓解 Scene 指标下降。
6. 将服务器上的稳定配置、prompt manifest 和结果摘要同步回 Git，同时继续排除大 checkpoint 与视频文件。
