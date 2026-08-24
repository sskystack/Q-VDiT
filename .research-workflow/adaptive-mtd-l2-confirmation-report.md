# Adaptive MTD L2 multi-prompt confirmation report

## Decision

**Pass the frozen artifact-level mechanism gate and proceed to a controlled
calibration ablation.** This result does not establish end-to-end video quality
and does not authorize VBench.

The supported candidate remains a two-stage structured construction:

1. use detached teacher similarity in a 5x5 search region to locate a coarse
   correspondence centre; and
2. compute MTD over the complete valid 3x3 around that centre.

Arbitrary top-9 selection is not supported, despite its high teacher-mass
coverage, because it breaks local geometry and worsens KL and displacement.

## Scope and provenance

- Prompts: indices 0, 2, and 6 from `t2v_samples_10.txt`.
- Diffusion progress: 25, 50, and 75; nine captures total.
- Views: conditional and CFG-guided paired FP/Q pooled features.
- Sampling: DDIM 100, CFG 4.0, seed 42, replayed original per-prompt RNG.
- Quantized checkpoint: released W4A6 MTD calibration checkpoint used by the
  capture runner.
- Hardware: physical GPU2 on `10.137.144.48`.
- Completion: `COMPLETED 2026-08-04 11:17:10`.
- Explicit exclusion: no optical-flow input, RAFT invocation, RAFT diagnostic,
  or RAFT-derived result was used.

Prompt 0 was retained from the second partial run because all three feature
captures had been written successfully before a video-filename-only failure.
The retry changed only output naming and generated prompts 2 and 6 with the
same model, seed, sampler, and feature-capture path.

## Aggregate candidate results

| View | Teacher mass outside fixed 3x3 | Top-1 outside | Centred KL reduction | Centred motion reduction | Centred displacement reduction |
|---|---:|---:|---:|---:|---:|
| Conditional | 61.07% | 61.08% | 39.48% | 42.31% | 45.11% |
| CFG-guided | 61.00% | 61.00% | 32.64% | 37.83% | 41.65% |

The teacher-centred window retains the teacher top-1 in 100% of queries and
covers about 77.5% of teacher probability mass, versus about 39.0% for the
released fixed window. Mean candidate count remains comparable: 8.41 versus
8.27, with the difference caused by image boundaries.

## Per-prompt consistency

Percentages below are reductions relative to the released fixed 3x3 after
aggregating the three phases within each prompt.

| Prompt | View | Outside mass | KL reduction | Motion reduction | Displacement reduction | Phases improving both KL and motion |
|---:|---|---:|---:|---:|---:|---:|
| 0 | Conditional | 60.76% | 35.32% | 39.47% | 42.99% | 3/3 |
| 2 | Conditional | 60.74% | 41.20% | 41.85% | 44.97% | 3/3 |
| 6 | Conditional | 61.72% | 41.29% | 44.71% | 46.95% | 3/3 |
| 0 | CFG-guided | 60.65% | 30.18% | 36.40% | 40.58% | 3/3 |
| 2 | CFG-guided | 60.66% | 34.32% | 36.04% | 41.43% | 3/3 |
| 6 | CFG-guided | 61.69% | 33.68% | 40.74% | 42.87% | 3/3 |

Across both views, all 18 prompt-phase combinations improve both primary
candidate metrics. The weakest single-phase improvement remains positive:
16.79% KL and 32.99% motion in the guided view.

## Equal-budget controls

- Random top-9 does not reproduce the result. Pooled conditional KL/motion
  worsen by 8.84%/4.58%; guided KL/motion worsen by 8.87%/3.47%.
- Teacher-similarity top-9 improves motion by 35.08% conditional and 35.44%
  guided, but worsens KL by 25.92%/23.73% and displacement by 95.97%/95.69%.
- These controls distinguish structured window relocation from either random
  extra freedom or unconstrained high-similarity selection.

## Importance-proxy confirmation

FP/Q per-position feature error passes the frozen importance gate independently
for every prompt.

| Prompt | Conditional Spearman | Conditional top-25 lift | Guided Spearman | Guided top-25 lift |
|---:|---:|---:|---:|---:|
| 0 | 0.278 | 1.774x | 0.340 | 1.885x |
| 2 | 0.267 | 1.774x | 0.338 | 1.790x |
| 6 | 0.280 | 1.742x | 0.333 | 1.897x |

All 18 individual prompt-phase proxy evaluations exceed Spearman 0.20 and
top-25 lift 1.25; no prompt or phase reverses direction. The pooled values are
0.272/1.756x conditional and 0.337/1.855x guided.

## Frozen-gate assessment

- Outside-window evidence: **pass**; approximately 61% versus threshold 15%.
- Equal-budget centred candidate: **pass**; both KL and motion improve by more
  than 10% in every prompt and view.
- Random control: **pass**; random candidates do not reproduce the gain.
- Importance proxy: **pass** for FP/Q feature error per prompt and phase.
- Conditional/guided robustness: **pass**.
- End-to-end calibration or VBench evidence: **not tested**.

## Artifacts and hashes

Compact copies are stored under
`.research-workflow/artifacts/AMTD-L2-002/{cond,guided}`. Remote originals are
under `logs_fp16_flash/stage_numeric_profiling/adaptive_mtd_candidates_v1/`.

- Conditional summary SHA-256:
  `48bb8d12bb58b09ffc05c9f57557f7f2b57cb3f55e6ddb072e3719022d0d96a7`
- Conditional per-capture SHA-256:
  `c222025ea81073d4087b001a922873f2ce38d28eda7f54ba4f66fe9a30080951`
- Guided summary SHA-256:
  `6990b25ea195b379d02f38596b70276b24ff740b76b4e2b424c06cae6ecd3fab`
- Guided per-capture SHA-256:
  `2af71c3f03256ac72806f33ede6da46dfd7fd81c483416fd9f0c0ebade12eb86`

## Next authorized step

Implement and run only the staged four-arm calibration experiment frozen in
`research-contract-adaptive-mtd-calibration-v1.md`: released fixed MTD,
fixed-candidate importance weighting, random equal-budget candidates, and
teacher-centred structured 3x3. Do not combine mechanisms or run VBench before
the short calibration gates pass.
