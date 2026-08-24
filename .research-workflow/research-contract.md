# Research contract

## Contract identity

- Version: v1
- Status: completed; hypothesis falsified
- Frozen at: 2026-07-31 Asia/Shanghai, before remote smoke/full results
- Selected idea ID: TF-TQE-001

## Claim and mechanism

- Falsifiable hypothesis: On held-out prompts, a parameter-free temporal DC/AC projection with separate rank-1 output corrections removes materially more residual error than an equal-parameter shared rank-2 correction.
- Baseline: Released Q-VDiT temporal TQE forward under FP16 + FlashAttention and W4A6, including Q(W+Win), an unmasked Wout path, and the released masked Wout path.
- Intervention or controlled change: No inference change during profiling. Fit an additive residual oracle C=P_DC Q(X) W_DC^T + P_AC Q(X) W_AC^T on training prompts and compare it with additive Q(X) W_shared^T of rank 2.
- Proposed mechanism: Static/content residuals and frame-varying/motion residuals occupy distinct predictable output subspaces; forcing one shared low-rank direction mixes them, while orthogonal temporal projections allow two semantically aligned directions without increasing learned parameter count.
- Scope and exclusions: Primary evidence is attn_temp.proj in blocks 0, 9, 17, and 27. q/k/v path attribution is auxiliary because FlashAttention is nonlinear. The released baseline is not repaired or replaced. VBench improvement is not claimed by this layer-local profiler alone.

## Evaluation design

- Primary metric or observation: Held-out total, DC, and AC residual-energy gain of DC-rank1 + AC-rank1 versus equal-parameter shared-rank2 for local same-input FP targets.
- Secondary metrics: AC gain versus released baseline; DC regression versus released; paired FP-trajectory alignment; conditional/unconditional branch non-regression; per-block consistency; Win/Wout path cosine, error removal, overcompensation, and mask-induced DC↔AC leakage.
- Data split, workload, or target conditions: prompts 0 and 2 train; prompt 6 untouched holdout; seed 42; sampling progress 5,15,25,45,65,85,95; full 16-frame temporal trajectories; FP trajectory supplies the paired x_t.
- Controls and comparators: released forward residual; shared rank-1; equal-parameter shared rank-2; separate DC rank-1 + AC rank-1. All reduced-rank fits use the same ridge-whitened globally optimal regression procedure.
- Repetition, seeds, folds, or trials: One frozen prompt split and seed for screening. A positive result still requires later prompt/seed transfer before a paper claim.

## Decision signals

- Success signal: At least 10% held-out gain in total, DC, and AC error versus shared rank-2; at least 15% AC gain versus released; no more than 1% DC regression versus released; positive total gain in at least 3/4 blocks; no more than 1% aggregate FP-trajectory regression versus shared rank-2; both CFG branches non-regressing within 1%.
- Failure signal stated independently: Exact-forward reconstruction fails; either frequency component does not beat shared rank-2 by 10%; AC does not beat released by 15%; DC regresses by more than 1%; fewer than 3 blocks are positive; or trajectory/branch checks regress beyond tolerance.
- Inconclusive region: Smoke passes but full run is unstable, non-finite, missing cells, overly sensitive to ridge/PCA randomization, or positive only on the single frozen holdout.
- Mandatory ablations: MTD checkpoint versus released baseline checkpoint; local-same-input versus paired FP-trajectory targets; shared rank-1 versus shared rank-2 versus DC/AC rank1x2; conditional versus unconditional; per-block and per-phase reporting; mask DC→AC and AC→DC leakage.

## Verification ladder

- [x] L0: paper consistency, dimensions, assumptions, complexity
- [x] L1: hand-checkable, synthetic, or tiny-case sanity check
- [x] L2: minimum-cost falsification run
- [x] L3: full-scale controlled experiment
- [ ] L4: ablation, robustness, variance, transfer, and resource checks
- [ ] L5: clean-environment or independent reproduction

## Budget and stopping

- Compute/time budget: GPU2 on the existing 4090 server; one smoke followed by sequential MTD and baseline full profiles; no calibration or VBench rerun at this stage.
- Stop conditions: Stop and mark invalid on released-forward reconstruction relative L2 >5e-3, non-finite values, missing train/holdout samples, wrong temporal shape, or failed rank fits. Kill/revise the module if the frozen full-profile gate fails.
- Safety, ethics, rules, and licensing constraints: Use existing user-owned checkpoints, prompts, and embeddings; do not alter the released inference path or consume other GPUs.

## Amendment history

- None. Create a new version rather than silently changing frozen criteria.
