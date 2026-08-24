# Temporal-frequency TQE profiling report

## Verdict

The fixed equal-parameter proposal

\[
C=P_{\mathrm{DC}}Q(X)W_{\mathrm{DC}}^\top+P_{\mathrm{AC}}Q(X)W_{\mathrm{AC}}^\top,
\]

with rank 1 assigned to each frequency component, is **falsified for the tested configuration**. It should not be implemented as the next Q-VDiT module.

This is a valid negative result rather than a profiler failure: both MTD and baseline runs reconstructed the released temporal forward with maximum relative L2 error 0, captured all 16 selected q/k/v/proj QuantTemporalAttnLinear modules, emitted 2688 path records per run, and contained no non-finite values.

## Frozen design

- Runtime: FP16 + FlashAttention, W4A6 checkpoints, GPU2.
- Train prompts: 0 and 2; untouched holdout: 6; seed: 42.
- Sampling progress: 5, 15, 25, 45, 65, 85, 95.
- Blocks: 0, 9, 17, 27.
- Primary target: `attn_temp.proj`; q/k/v are auxiliary path diagnostics.
- Equal learned parameter control for each 1152→1152 projection:
  - shared rank 2: 2 × (1152 + 1152) = 4608 factor parameters;
  - DC rank 1 + AC rank 1: 2 × (1152 + 1152) = 4608 factor parameters.
- Fit: ridge-whitened globally optimal reduced-rank regression. The shared control is not weakened by PCA directions that the input cannot predict.

## Held-out local-same-input result

| Checkpoint | Model | Total gain vs released | DC gain vs released | AC gain vs released | Total gain vs shared rank 2 |
|---|---:|---:|---:|---:|---:|
| MTD | shared rank 2 | +32.89% | +31.91% | +36.90% | control |
| MTD | DC1 + AC1 | -251.93% | -321.69% | +34.28% | -424.39% |
| Baseline | shared rank 2 | +55.19% | +53.56% | +61.01% | control |
| Baseline | DC1 + AC1 | +11.71% | -1.32% | +58.17% | -97.03% |

No selected block beat the equal-parameter shared rank-2 control. The worst instability was concentrated in deep DC residuals:

| Checkpoint | Block | DC fraction of released residual | DC1+AC1 gain vs shared rank 2 |
|---|---:|---:|---:|
| MTD | 0 | 33.15% | -8.68% |
| MTD | 9 | 93.67% | -900.23% |
| MTD | 17 | 89.20% | -18.30% |
| MTD | 27 | 87.96% | -961.23% |
| Baseline | 0 | 31.40% | -20.01% |
| Baseline | 9 | 89.73% | -427.20% |
| Baseline | 17 | 89.72% | -18.29% |
| Baseline | 27 | 94.61% | -101.71% |

The failure is not explained by one diffusion phase. MTD DC1+AC1 total gain versus released was -42.90% early, -324.38% middle, and -334.24% late. Baseline was +41.06% early, +16.75% middle, and -23.41% late, but remained substantially worse than shared rank 2 in every phase.

## Why it failed

1. **The learned frame mask produces essentially no DC→AC leakage.** For `attn_temp.proj`, leakage energy divided by unmasked DC energy was only 5.52×10⁻⁶ for MTD and 1.91×10⁻⁶ for baseline. Across q/k/v/proj the largest value was 5.52×10⁻⁶. Mean mask standard deviation was only 0.00203 for MTD and 0.00139 for baseline. Therefore the proposed mechanism—frame masking mixes static content compensation into motion—is not supported.

2. **A fixed 1+1 rank allocation is mismatched to residual energy.** DC represented 72.28% of MTD and 76.68% of baseline local training residual energy; in deep held-out blocks it represented roughly 88–95%. Shared rank 2 can devote both directions where needed, while the split model permanently reserves half its capacity for AC.

3. **DC directions are not prompt-stable enough at rank 1.** Even on training data, DC1+AC1 was worse than shared rank 2 by 11.44% for MTD and 16.55% for baseline. On held-out prompt 6, DC errors then exploded in blocks 9/27 for MTD and block 9 for baseline.

4. **AC separability is real but is not an overall module win.** The AC branch reduced released AC error by 34.28% for MTD and 58.17% for baseline. However shared rank 2 reduced it by 36.90% and 61.01%, respectively, while also preserving enough DC capacity. Thus the experiment supports AC as a useful diagnostic component, not fixed frequency-decoupled TQE as an implementation.

## Existing released-path observation

The exact path decomposition also shows that current temporal `proj` compensation is not frequency-selective. Against the same-input FP target, released Wout worsened the post-Win AC error by 5.71% for MTD and 9.90% for baseline, and worsened DC by 0.70% and 2.42%, respectively. These are attribution observations, not evidence that removing the released path would improve end-to-end VBench.

## Decision

- Frozen decision: `kill_or_revise` for both checkpoints.
- Scientific classification: **falsified** for fixed rank-1 DC + rank-1 AC.
- Do not proceed directly to calibration or VBench with this module.
- The remaining actionable signal is not “use DC/AC split”; it is “AC residual is predictable, while deep-block DC residual directions are high-energy and prompt-unstable.” Any next idea must solve that capacity/stability problem and must again beat a shared equal-parameter control on untouched prompts before implementation.

## Artifacts

- Profiler SHA-256: `33fbd78345413ee09cf9daa175dcabd714947901653e294e824ca4aeaad6bf5c`
- Remote MTD: `/home/zhouchongtian/quantization/qvdit_flash_bf16/logs_fp16_flash/stage_numeric_profiling/temporal_frequency_mtd_prompts026_gpu2_0801`
- Remote baseline: `/home/zhouchongtian/quantization/qvdit_flash_bf16/logs_fp16_flash/stage_numeric_profiling/temporal_frequency_baseline_prompts026_gpu2_0801`
- MTD decision SHA-256: `290d2a08f79dad962d2ffc7654e116b76c534287de80deead2966aefd733d6e0`
- Baseline decision SHA-256: `fac44326af1c2391556d7b41244c55fcc7fddd8644f31485ba863f4bcdd2b6ae`

