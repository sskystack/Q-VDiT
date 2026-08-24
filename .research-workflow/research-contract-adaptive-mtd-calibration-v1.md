# Adaptive MTD controlled calibration contract v1

## Identity

- Date frozen: 2026-08-04
- Parent idea: AMTD-003
- Parent evidence: AMTD-L2-001 and AMTD-L2-002
- Mode: minimum-cost calibration falsification after artifact-level validation

## Objective and hypothesis

Determine whether the two validated artifact mechanisms improve calibration
behavior without changing the quantization setup or spending a VBench run.

The candidate-localization hypothesis is that teacher-derived coarse alignment
followed by a complete local 3x3 provides a better MTD target than the released
same-centre 3x3. The independent importance hypothesis is that detached
per-position FP/Q feature error should reweight the released MTD loss toward
queries where current quantization causes more transport disagreement.

## Fixed baseline and common setup

- FP16 + FlashAttention, OpenSora 16x512x512, W4A6.
- Same calibration data snapshot, initialization, reconstruction loss, MTD
  coefficients, optimizer, learning-rate schedule, batch order, and random seed
  for every arm.
- MTD transport size 16 and temperature 0.07.
- No change to inference; MTD and all teacher-derived routing exist only during
  calibration.
- No optical flow, RAFT code, RAFT artifact, learned offset network, or learned
  selector.

## Frozen four-arm matrix

### A. Released fixed MTD

Use the current same-centre valid 3x3 candidate set and uniform mean over frame
pairs and spatial queries. This is the reference arm.

### B. Weighted-only fixed MTD

Keep arm A's exact candidates. For each spatial query, compute detached
per-position FP/Q feature error on the same pooled denoising features:

`e_i = mean_channel((pred_i - target_i)^2)`.

Normalize within each batch/frame-pair group as

`w_i = N * (e_i + eps) / sum_j(e_j + eps)`,

with `eps = 1e-8`, and detach `w_i`. Replace only the uniform spatial-query mean
for local KL and motion residual with the weighted mean. Global relation remains
unchanged. Do not clip, exponentiate, tune, or backpropagate through the weight.

### C. Random equal-budget candidates

For every query, sample nine candidates without replacement from the valid 5x5
teacher-search region using a generator derived deterministically from the run
seed, calibration iteration, frame pair, and query index. Apply the identical
sampled candidate set to prediction and detached target. Keep uniform spatial
reduction. This is a negative control and is not eligible for promotion.

### D. Teacher-centred structured 3x3

On detached target features, search the valid same-centre 5x5 region and choose
the top-1 cosine-similarity location as a coarse centre. Use the complete valid
3x3 around that centre for both prediction and target distributions and
transport residuals. Keep uniform spatial reduction and the released global
relation term. The argmax and candidate indices are detached and receive no
gradient.

Do not add a centred-plus-weighted fifth arm in v1. Combining mechanisms is a
separate decision only after B and D are independently interpretable.

## Evaluator and required logging

For every arm, save:

- reconstruction, total MTD, local KL, motion residual, and global relation by
  iteration;
- gradient norm and cosine of each MTD component relative to reconstruction;
- finite-value checks, peak memory, wall time, and iteration throughput;
- held-out paired FP/Q captures on prompts 0, 2, and 6 at progress 25/50/75;
- for C/D, candidate count, teacher mass coverage, top-1 retention, and coarse
  offset histogram;
- exact config, seed, checkpoint initialization, commands, and artifact hashes.

The held-out feature evaluator is the flow-free adaptive profiler used by
AMTD-L2-002. Its formulas and temperature are frozen before calibration.

## Staged verification ladder

### L3a: correctness and gradient smoke

- Unit cases for borders, identical FP/Q, deterministic random candidates, and
  detached candidate/weight construction.
- Arm A must numerically reproduce released MTD within `rtol=1e-5` and
  `atol=1e-6` on fixed tensors.
- All four arms must produce finite forward values and gradients.
- B's weights must have mean 1 per group and no gradient.
- C and D must use the same candidate indices for prediction and target.

Stop and mark invalid if any check fails.

### L3b: short calibration screen

Run all four arms from the same initialization for the same first 200
calibration iterations. This phase is a mechanism screen, not a final model.

An experimental arm survives only if:

- no NaN/Inf or gradient explosion occurs;
- wall time is no more than 1.75x arm A and peak memory no more than 1.5x arm A;
- final-50-iteration mean reconstruction loss is no worse than arm A by more
  than 2%;
- final-50-iteration MTD total does not worsen by more than 5%; and
- its intended diagnostic moves in the predicted direction on held-out
  captures: B increases high-error-query loss concentration, while D improves
  both local KL and motion error by at least 5% relative to A.

C must not match D on both held-out KL and motion. If it does, candidate
localization is considered unconfirmed and D is not promoted.

### L3c: controlled continuation

Only surviving B or D arms may continue to the previously established full
calibration budget, always paired with arm A from the same initialization and
data order. C stops after L3b. Do not run VBench in L3c.

Promotion to an end-to-end evaluation contract requires:

- reconstruction no worse than arm A by more than 1%;
- stable MTD component gradients without a new motion-component conflict;
- intended held-out feature gain in both conditional and guided views on every
  prompt, with pooled KL and motion gains at least 5%; and
- runtime and memory within the L3b limits.

## Independent failure signals

- D's teacher-centred gain disappears after calibration or is matched by C;
- B's weighting concentrates loss but worsens reconstruction or motion-gradient
  alignment;
- benefits occur only in the calibration batch, only one prompt, or only one
  conditional/guided view;
- candidate offsets collapse to the same centre, indicating no effective
  adaptation;
- overhead exceeds the frozen limits;
- either mechanism requires retuning the baseline's unrelated coefficients to
  appear favorable.

## Stop conditions and authorization boundary

- Stop after L3a if correctness fails.
- Stop an arm after L3b when any survival gate fails; preserve it as a negative
  result.
- Do not combine B and D, tune new weights/radii/temperatures, change calibration
  data, or run VBench under this contract.
- A new frozen contract and user authorization are required before end-to-end
  video evaluation.

## Amendment history

- v1: frozen after AMTD-L2-002 passed all multi-prompt candidate and importance
  gates. RAFT and optical flow remain explicitly excluded.
