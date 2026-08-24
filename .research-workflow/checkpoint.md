# Research or competition checkpoint

## Current objective

Determine whether the released MTD's fixed same-centre 3x3 candidate set and
uniform spatial-query reduction should be replaced by an equal-budget adaptive
candidate/importance mechanism, before changing calibration or running VBench.

## Confirmed facts

- Released MTD uses similarity-softmax weights within each valid 3x3; the nine
  candidates are not uniformly averaged. Equal weighting occurs when losses are
  averaged over frame pairs and spatial queries.
- The new adaptive diagnostic is independent from the training path and excludes
  optical flow and the existing RAFT-guided script.
- L0 passed: all 12 MTD/adaptive-diagnostic tests pass locally.
- L1 passed on a hand-checkable two-cell displacement: 89.06% top-1 matches and
  88.31% teacher mass lie outside fixed 3x3. This is a sanity result only.
- Preliminary L2 on one prompt and 12 diffusion phases passes the candidate
  mechanism: coarse teacher centre plus complete local 3x3 improves both KL and
  motion error in 12/12 phases for MTD/baseline and conditional/guided views.
- Arbitrary teacher top-9 is rejected: it reduces motion error but worsens KL
  and expected-displacement error. Preserving local geometry is material.
- FP/Q per-position feature error passes the importance-proxy gate; matching
  confidence fails and is negatively correlated with current MTD need.
- Fresh AMTD-L2-002 confirmation completed on prompts 0, 2, and 6 at progress
  25/50/75 for conditional and CFG-guided features. The structured
  teacher-centred 3x3 improves both KL and motion error in all 18 prompt-phase
  views. Aggregate reductions are 39.48%/42.31% conditional and
  32.64%/37.83% guided.
- Random top-9 worsens pooled KL and motion in both views, while arbitrary
  teacher top-9 improves motion but worsens KL and displacement. The evidence
  therefore favors coarse localization followed by a complete local window.
- The FP/Q-error proxy passes independently for every prompt: per-prompt
  Spearman is 0.267--0.280 conditional and 0.333--0.340 guided; top-25% lift is
  1.742x--1.774x conditional and 1.790x--1.897x guided. Every individual phase
  is positive and above the frozen threshold.
- Released temporal forward contains Q(X)Q(W+Win)^T plus both unmasked and masked Wout contributions; profiling preserves this baseline exactly.
- Fixed DC-rank1 + AC-rank1 was falsified: it wastes rank capacity and does not beat a shared equal-parameter rank-2 control.
- CS-TQE-002 was also falsified by its frozen gate. A Wout-orthogonal rank-1 oracle has substantial local gain, but retains only 77.85% of unconstrained rank-1 gain on MTD and 76.03% on baseline.
- Parallel-plus-orthogonal structured rank 2 retains only 75.85% of shared rank-2 gain on MTD and 81.49% on baseline, below the frozen 90% requirement.
- The local orthogonal branch improves total/DC/AC in all four selected blocks, but the paired FP trajectory worsens AC by 5.49% on MTD and 7.53% on baseline; late-phase AC worsens by 10.34% and 13.60%.
- FP-trajectory output directions are prompt-stable, but the stable correction is DC-dominated and therefore is not evidence of motion safety.

## Decisions and rationale

- Preserve FP16 + FlashAttention, W4A6, and the released Q-VDiT baseline path.
- Mark TF-TQE-001 and CS-TQE-002 as falsified; preserve both negative results.
- Do not implement, calibrate, or run VBench for CS-TQE-002 because the frozen L3 gate failed.
- Do not reinterpret a free orthogonal/complementary direction as a publication-ready result: without the released-Wout basis claim it reduces to ordinary rank expansion and still carries AC risk.
- Require future profiling to elevate FP-trajectory DC/AC safety to a primary gate; total residual L2 alone is insufficient for the all-VBench objective.
- Mark AMTD-003 as validated at the artifact-mechanism level and AMTD-L2-002 as
  valid. This authorizes implementation of a controlled calibration ablation,
  but does not authorize VBench or a quality claim.
- Keep candidate localization and importance weighting as separate arms in the
  first calibration screen; do not combine them until each is compared with
  the released and random controls.

## Best valid result

As a diagnostic oracle, unconstrained shared rank 2 reduces held-out local residual error by 32.89% on MTD and 55.20% on baseline. This is evidence of low-rank residual headroom, not yet an implementable or motion-safe module.

For adaptive MTD, the fresh three-prompt mechanism screen is the first candidate
to pass its pre-calibration gate: teacher-centred structured 3x3 reduces pooled
KL/motion by 39.48%/42.31% conditional and 32.64%/37.83% guided, with positive
joint improvement in all 18 prompt-phase views.

## Active experiments

- AMTD-L2-001 is complete and valid as a one-prompt screen.
- AMTD-L2-002 is complete and valid as the fresh three-prompt confirmation.
- Adaptive-MTD calibration contract v1 is frozen; its correctness stage is
  complete and the short controlled calibration stage is now active.
- AMTD-L3A-001 is complete and valid: 19/19 local and remote focused tests pass,
  and all four arms produce finite CUDA losses and gradients. The released
  fixed path remains the default and the original server working tree was not
  overwritten.
- AMTD-L3B-001 is running in an isolated server code snapshot on exclusive
  physical GPU6. The user's previous GPU6 VBench generation was stopped with
  partial outputs preserved; the experiment is sequentially running fixed,
  weighted, random, and centred 200-iteration arms.
- KR-MTD-MULTI-002 is prepared under `research-contract-keyregion-mtd-v2.md`.
  It adds a compute-matched MSE-500 control and evaluates FP/MSE/base/random25/
  error25/all on six calibration-disjoint VBench prompts, seeds 42 and 123,
  DDIM50, online T5, paired temporal latent errors, and seven reduced-scale
  VBench metrics on physical GPU2.

## Blocked or uncertain

- The current profiler does not establish a small structural constraint that preserves shared low-rank DC gains while avoiding FP-trajectory AC regression.
- A free rank expansion has weak novelty and is not promoted.
- End-to-end adaptive-MTD quality relevance remains untested intentionally;
  artifact-level improvements may not survive calibration dynamics.

## Next actions

- Implement the four arms in
  `research-contract-adaptive-mtd-calibration-v1.md` without changing the
  frozen formulas, then run only the contract's correctness and short
  calibration screens.
- Do not run VBench until a calibration arm passes the gradient, reconstruction,
  candidate-behavior, and held-out feature gates.

## Files and commands to reread

- `qdiff/mtd_adaptive_diagnostics.py`
- `tools/profile_mtd_adaptive_correspondence.py`
- `.research-workflow/research-contract-adaptive-mtd-v1.md`
- `.research-workflow/adaptive-mtd-l2-preliminary-report.md`
- `.research-workflow/adaptive-mtd-l2-confirmation-report.md`
- `.research-workflow/research-contract-adaptive-mtd-calibration-v1.md`
- `tools/profile_wout_complementary_subspace.py`
- `.research-workflow/research-contract-wout-complement-v2.md`
- `.research-workflow/wout-complementary-subspace-report.md`
- Remote `decision.json`, `heldout_complementary_subspace_comparison.jsonl`, `rank_fit_diagnostics.jsonl`, and `train_prompt_subspace_stability.jsonl`
