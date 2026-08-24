# Research contract: complementary Wout subspace

## Contract identity

- Version: v2
- Status: completed; falsified by the frozen L3 gate
- Frozen at: 2026-07-31 Asia/Shanghai, before smoke/full results
- Selected idea ID: CS-TQE-002
- Predecessor: TF-TQE-001 was falsified; its criteria and result remain unchanged.

## Claim and mechanism

- Falsifiable hypothesis: The released rank-1 temporal Wout leaves a prompt-stable, predictable residual direction in the output-channel subspace orthogonal to its current output vector.
- Baseline: Released FP16 + FlashAttention W4A6 temporal TQE forward, including Q(W+Win), unmasked Wout, and masked Wout.
- Intervention under test: No inference modification. Fit an additive rank-1 residual oracle constrained to be output-orthogonal to the released Wout and compare it with equal-parameter unconstrained and Wout-parallel rank-1 controls.
- Proposed module if supported: A zero-initialized complementary rank-1 branch on `attn_temp.proj` Wout, with its output factor orthogonalized against the released Wout output factor.
- Scope: blocks 0, 9, 17, and 27; prompts 0 and 2 train; prompt 6 holdout; seven sampling points; both CFG branches.

## Evidence motivating the test

- Held-out additive unconstrained rank-1 reduced local residual error by 24.03% on MTD and 42.41% on baseline.
- Rank-2 further reduced the rank-1 remaining error by 11.66% and 22.19%, respectively.
- Fixed DC-rank1 + AC-rank1 was falsified because the frame mask produced negligible DC→AC leakage and fixed rank allocation destabilized deep DC residuals.

## Equal-parameter comparisons

- Rank-1: unconstrained versus parallel-to-Wout-output versus orthogonal-to-Wout-output.
- Rank-2: unconstrained shared rank-2 versus parallel rank-1 + orthogonal rank-1.
- Branch shape: unmasked additive `C` is primary; released-shaped `C + M*C` is a frozen secondary comparison.
- References: local same-input FP target is primary; paired FP-trajectory target is a safety check.
- Fit method: ridge-whitened globally optimal reduced-rank regression for every model.

## Success signal

The primary unmasked orthogonal rank-1 branch must satisfy all of:

- unconstrained rank-1 held-out total gain versus released baseline at least 10%;
- orthogonal rank-1 held-out total gain at least 15%;
- orthogonal rank-1 retains at least 80% of unconstrained rank-1 total gain;
- DC and AC gains each at least -1% versus released baseline;
- positive total gain with DC/AC non-regression in at least 3/4 blocks;
- structured parallel+orthogonal rank-2 retains at least 90% of unconstrained rank-2 total gain;
- paired FP-trajectory total gain and each CFG branch gain at least -1%.

## Failure signal

- Exact released-forward reconstruction fails or any profiling value is non-finite.
- Orthogonal headroom is below the frozen thresholds.
- Most rank-1 gain lies in the Wout-parallel correction, indicating cancellation/re-scaling rather than a missing direction.
- Structured parallel+orthogonal rank-2 cannot approximate the unconstrained rank-2 control.
- Gains are confined to fewer than three blocks or damage DC, AC, or a CFG branch.

## Inconclusive region

- Smoke passes but full results are highly prompt-unstable or randomized-SVD-sensitive.
- MTD and baseline disagree materially; in that case the MTD result governs implementation, while baseline is explanatory only.
- The unmasked and released-mask-shaped branches disagree without a mechanistic explanation.

## Verification ladder

- [x] L0: dimensions, parameter equality, released Wout rank, and insertion point checked
- [x] L1: smoke reconstruction and tiny held-out fit
- [x] L2: one-block reduced profiling
- [x] L3: full frozen MTD and baseline comparison
- [ ] L4: new prompts/seeds and actual calibration if profiling passes
- [ ] L5: end-to-end VBench reproduction if implementation passes calibration checks

## Budget and stopping

- Hardware: shared GPU2 on the authorized remote server.
- Budget: one smoke plus sequential MTD and baseline full profiles; no calibration or VBench at this stage.
- Stop immediately on invalid reconstruction, shape mismatch, non-finite values, or missing train/holdout samples.
- Do not implement the module if the frozen full-profile gate fails.

## Frozen-result disposition

- Completed at: 2026-07-31 Asia/Shanghai.
- Profiler SHA-256: `a1d9e690bc34b04fcb4428e23a03f7ac60f369f2a44735dc550d7fb10061ee88`.
- Numerical validity passed for smoke, MTD, and baseline: released-forward reconstruction error was 0 and all reported values were finite.
- MTD primary local holdout: orthogonal rank-1 gain 18.70%, retention versus unconstrained rank-1 77.85%, and structured rank-2 retention 75.85%.
- Baseline primary local holdout: orthogonal rank-1 gain 32.25%, retention versus unconstrained rank-1 76.03%, and structured rank-2 retention 81.49%.
- Both checkpoints failed the frozen 80% rank-1-retention and 90% structured-rank-2-retention requirements.
- The paired FP trajectory exposed an additional motion-risk signal not used to relax or rewrite the frozen gate: orthogonal rank-1 reduced total error by 20.27% on MTD and 33.36% on baseline, but increased AC error by 5.49% and 7.53%, respectively. Late-phase AC error increased by 10.34% and 13.60%.
- Final classification: `falsified` for the claim that the released Wout plus one fixed output-orthogonal rank-1 branch forms a stable and sufficiently complete two-direction compensation basis.
- L4/L5 were not run because the L3 promotion gate failed.
