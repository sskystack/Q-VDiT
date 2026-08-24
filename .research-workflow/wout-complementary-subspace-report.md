# Complementary Wout subspace profiling report

## Verdict

The tested module is **falsified for promotion**:

\[
C(X)=Q(X)a_o b_o^\top,\qquad b_o^\top b_1=0,
\]

where \(b_1\) is the released rank-1 Wout output direction. The experiment does show substantial residual headroom outside \(b_1\), but it does not show that \(b_1\) and one fixed orthogonal direction form a stable, sufficiently complete two-direction compensation basis. No calibration or VBench run should be started for this exact module.

## Frozen setup and validity

- Runtime: FP16 + FlashAttention, W4A6, shared GPU2.
- Train prompts: 0 and 2; untouched holdout: 6; seed: 42.
- Progress samples: 5, 15, 25, 45, 65, 85, and 95.
- Blocks: 0, 9, 17, and 27; target: `attn_temp.proj`.
- Primary branch: unmasked additive correction; released-mask-shaped branch was secondary.
- Controls: equal-parameter unconstrained, Wout-parallel, and Wout-orthogonal rank 1; unconstrained shared rank 2 versus parallel rank 1 plus orthogonal rank 1.
- Fit: ridge-whitened globally optimal reduced-rank regression.
- Released forward was reconstructed with relative L2 error 0. All 672 path records and all comparison values per full run were finite.

## Primary held-out local result

| Checkpoint | Unconstrained rank 1 | Parallel rank 1 | Orthogonal rank 1 | Orthogonal DC | Orthogonal AC | Orthogonal retention | Shared rank 2 | Structured rank 2 | Structured retention | Target energy parallel to Wout |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MTD | +24.03% | +6.24% | +18.70% | +19.81% | +14.18% | 77.85% | +32.89% | +24.95% | 75.85% | 6.36% |
| Baseline | +42.41% | +12.73% | +32.25% | +34.32% | +24.86% | 76.03% | +55.20% | +44.98% | 81.49% | 12.90% |

Both checkpoints pass the absolute 15% orthogonal-gain requirement, and all four local blocks have positive total/DC/AC gains. Both nevertheless fail the frozen 80% orthogonal-retention requirement and the 90% structured-rank-2-retention requirement. The thresholds were not changed after observing the results.

The parallel target-energy fraction is small overall, so the released Wout direction is not where most residual energy lies. However, the structured rank-2 result shows that simply preserving the released direction as one basis vector wastes too much of a two-direction budget, especially in deeper blocks.

The released-mask-shaped secondary branch gives the same conclusion: orthogonal gain/retention are 18.76%/77.89% on MTD and 31.90%/75.99% on baseline. Therefore the failure is not an artifact of choosing unmasked `C` instead of `C + M*C`.

## Per-block local evidence

| Checkpoint | Block | Orthogonal total | DC | AC | Orthogonal retention | Structured retention | Parallel target energy |
|---|---:|---:|---:|---:|---:|---:|---:|
| MTD | 0 | +16.08% | +5.20% | +21.48% | 42.33% | 84.19% | 24.33% |
| MTD | 9 | +54.21% | +56.65% | +18.19% | 88.49% | 96.65% | 7.69% |
| MTD | 17 | +7.10% | +7.75% | +1.81% | 100.98% | 41.02% | 0.94% |
| MTD | 27 | +4.07% | +4.60% | +0.20% | 99.52% | 34.00% | 0.50% |
| Baseline | 0 | +25.72% | +10.28% | +32.79% | 46.52% | 78.36% | 36.87% |
| Baseline | 9 | +7.30% | +7.82% | +2.73% | 113.94% | 39.25% | 4.98% |
| Baseline | 17 | +21.02% | +23.28% | +1.26% | 99.04% | 72.33% | 1.04% |
| Baseline | 27 | +47.22% | +49.65% | +4.56% | 86.14% | 90.34% | 8.41% |

The result is not driven by a single bad block. The fixed orthogonal direction retains rank-1 headroom poorly in block 0, while the structured two-direction basis loses large rank-2 headroom in several deeper blocks. This is inconsistent with a universal geometric basis anchored on the released Wout direction.

## Diffusion-phase evidence

The local same-input orthogonal branch is positive in every phase, but structured retention remains below 90% in all MTD phases and in two of three baseline phases.

| Checkpoint | Phase | Orthogonal total | DC | AC | Orthogonal retention | Structured retention |
|---|---|---:|---:|---:|---:|---:|
| MTD | early | +15.60% | +15.56% | +15.70% | 70.05% | 59.01% |
| MTD | middle | +18.85% | +19.93% | +13.90% | 78.22% | 85.44% |
| MTD | late | +20.59% | +22.17% | +12.95% | 82.04% | 82.74% |
| Baseline | early | +36.37% | +39.79% | +26.67% | 75.96% | 77.95% |
| Baseline | middle | +40.13% | +43.26% | +25.52% | 77.36% | 92.16% |
| Baseline | late | +19.73% | +19.05% | +22.14% | 73.42% | 71.65% |

Thus timestep segmentation is not the missing mechanism. The failure is geometric/capacity-related rather than confined to early, middle, or late sampling.

## FP-trajectory safety and motion risk

| Checkpoint | Orthogonal total | DC | AC | Conditional total | Conditional AC | Unconditional total | Unconditional AC |
|---|---:|---:|---:|---:|---:|---:|---:|
| MTD | +20.27% | +22.58% | -5.49% | +22.07% | -6.02% | +18.44% | -4.98% |
| Baseline | +33.36% | +35.22% | -7.53% | +33.01% | -8.11% | +33.56% | -7.05% |

The branch improves total error and both CFG branches because DC dominates the energy, but it consistently worsens AC on the actual FP trajectory. The damage grows with sampling progress:

| Checkpoint | Phase | Orthogonal total | DC | AC |
|---|---|---:|---:|---:|
| MTD | early | +37.52% | +39.30% | +1.95% |
| MTD | middle | +13.28% | +14.88% | -3.24% |
| MTD | late | +4.77% | +6.82% | -10.34% |
| Baseline | early | +26.97% | +27.68% | +0.26% |
| Baseline | middle | +55.04% | +57.38% | -6.14% |
| Baseline | late | +14.32% | +17.50% | -13.60% |

This is directly adverse to the goal of recovering Dynamic Degree while improving the rest of VBench. A total-L2 gain dominated by DC is therefore not a sufficient promotion signal.

## Prompt stability

For the local same-input target, the mean absolute output-direction cosine between prompt-0 and prompt-2 fits is 0.456 for MTD and 0.706 for baseline after orthogonalization. The minimum block values are 0.149 and 0.019. This is not a prompt-stable fixed local correction direction.

For the FP-trajectory target, the corresponding mean cosines are much higher, 0.837 and 0.961, but this stable direction is primarily a DC-error direction and still damages AC. Stability alone therefore does not rescue the proposed mechanism.

## Scientific interpretation

The experiment supports three narrower statements:

1. Most remaining temporal-projection residual energy is outside the released rank-1 Wout output direction.
2. The released Wout direction is not a privileged basis vector for an efficient rank-2 residual model; a free shared rank-2 basis is materially better.
3. On the paired FP trajectory, a dominant and prompt-stable DC correction can hide worsening AC error inside an improved total error.

It does **not** support claiming a continuous orthogonal rotation around the released Wout, a stable released-plus-complementary two-dimensional basis, or likely Dynamic Degree improvement.

## Decision and next boundary

- Frozen decision: `kill_or_revise` on both checkpoints.
- Scientific classification: **falsified** for CS-TQE-002 as specified.
- Do not implement, calibrate, or run VBench for this exact branch.
- A free complementary direction would amount to ordinary rank expansion and still exhibits the AC-risk signal; it is not yet a paper-worthy replacement.
- Any next candidate must use a controlled comparison that preserves the strong shared low-rank DC correction while explicitly preventing FP-trajectory AC regression. It must beat an equal-parameter shared-rank control on untouched prompts and use DC/AC trajectory safety as a primary gate, not merely total residual L2.

## Artifacts

- Profiler SHA-256: `a1d9e690bc34b04fcb4428e23a03f7ac60f369f2a44735dc550d7fb10061ee88`.
- Remote smoke: `/home/zhouchongtian/quantization/qvdit_flash_bf16/logs_fp16_flash/stage_numeric_profiling/wout_complement_smoke_mtd_gpu2_0801`.
- Remote MTD: `/home/zhouchongtian/quantization/qvdit_flash_bf16/logs_fp16_flash/stage_numeric_profiling/wout_complement_mtd_prompts026_gpu2_0801`.
- Remote baseline: `/home/zhouchongtian/quantization/qvdit_flash_bf16/logs_fp16_flash/stage_numeric_profiling/wout_complement_baseline_prompts026_gpu2_0801`.
- Smoke decision SHA-256: `65da4d31164f6d4cc321a84d1c8a62556c4e8b86d2d2b1859e56ae1fcd0d1b2f`.
- MTD decision SHA-256: `a123a4fad4efbba5e332851cc48badb9eb18b15d3ada068cad1089f6c3457f3f`.
- Baseline decision SHA-256: `adb0c21b0f4a306da307294137c9dbc835dad4b78dc8f200e74c2a3fbf16baf6`.
