# Adaptive MTD L2 preliminary report

## Decision

**Revise and continue to multi-prompt confirmation.** The position-conditioned
candidate mechanism passes the one-prompt, twelve-phase artifact screen, but it
is not yet promoted to calibration because prompt diversity is absent.

The successful candidate is not arbitrary teacher top-9 selection. It is:

1. search a 5x5 teacher window for a coarse correspondence centre; then
2. retain the complete local 3x3 around that centre.

This preserves local geometry while allowing the search centre to move.

## Scope and exclusions

- Feature source: existing paired FP/Q captures for the giraffe translation
  workload, 12 diffusion progress points.
- Views: MTD and baseline checkpoints, each evaluated on conditional and
  CFG-guided features.
- Search: teacher feature similarity only.
- Explicitly excluded: optical flow, RAFT, and the repository's RAFT-guided
  diagnostic or its outputs.
- This is an artifact-level mechanism screen, not a calibration or VBench
  result.

## Aggregate results

| Workload | Fixed 3x3 KL | Centred 3x3 KL | Fixed motion | Centred motion | Fixed coverage | Centred coverage |
|---|---:|---:|---:|---:|---:|---:|
| MTD conditional | 0.18454 | 0.13885 | 0.005451 | 0.003572 | 0.3838 | 0.7732 |
| MTD guided | 0.75211 | 0.64305 | 0.016711 | 0.012233 | 0.3868 | 0.7732 |
| Baseline conditional | 0.20196 | 0.16406 | 0.004622 | 0.003154 | 0.3845 | 0.7719 |
| Baseline guided | 0.69533 | 0.61650 | 0.013741 | 0.009621 | 0.3859 | 0.7727 |

Aggregate relative improvements for centred 3x3 versus fixed 3x3:

| Workload | KL | Motion error | Expected-displacement error |
|---|---:|---:|---:|
| MTD conditional | -24.76% | -34.47% | -38.23% |
| MTD guided | -14.50% | -26.79% | -33.86% |
| Baseline conditional | -18.76% | -31.77% | -36.99% |
| Baseline guided | -11.34% | -29.98% | -35.01% |

The centred candidate count averaged 8.41 versus 8.27 for the released fixed
window; the small difference is caused by actual image boundaries rather than
truncation by the coarse search bank.

## Per-phase consistency

- All 12/12 captures improved both KL and motion error in all four workload/view
  combinations.
- Mean per-capture KL improvement was 40.89%, 34.32%, 40.50%, and 34.37% for
  MTD conditional, MTD guided, baseline conditional, and baseline guided.
- Mean per-capture motion improvement was 42.11%, 36.91%, 43.50%, and 38.22%.
- The smallest per-capture KL improvement remained positive: 19.61%, 4.82%,
  15.20%, and 3.84%, respectively.

## Fixed-window failure evidence

- Teacher probability mass outside the released same-centre 3x3 was
  61.32%--61.62% across the four combinations.
- Teacher top-1 outside rate was 61.22%--61.61%.
- This passes the frozen v1 evidence threshold, but only for the tested prompt.

## Candidate controls

- Teacher top-9 inside 5x5 covered almost 100% of teacher mass and reduced
  feature-transport error, but worsened local KL and expected-displacement error.
  It is rejected as the primary mechanism.
- Random top-9 remained close to or worse than fixed 3x3 and did not match the
  centred candidate.
- Therefore the result supports moving a structured local window, not simply
  selecting high-similarity points from a wider area.

## Importance-proxy results

FP/Q per-position feature error was the only proxy to pass the frozen importance
gate in every workload:

| Workload | Spearman with fixed-MTD need | Top-25% loss-mass lift |
|---|---:|---:|
| MTD conditional | 0.262 | 1.601x |
| MTD guided | 0.394 | 1.533x |
| Baseline conditional | 0.317 | 1.834x |
| Baseline guided | 0.441 | 1.853x |

Teacher motion magnitude showed useful top-quartile enrichment (1.40x--1.50x)
but weak rank correlation (0.10--0.13), so it does not pass alone. Matching
confidence failed and was negatively correlated; an uncertainty-based weighting
hypothesis must be frozen separately rather than retrofitted into v1.

## Gate assessment

- Candidate-set gate: **pass on this workload**.
- Importance gate: **pass for FP/Q feature error**.
- Prompt/holdout confirmation: **not passed; only one prompt**.
- Calibration/VBench authorization: **not yet justified**.

## Next experiment

Run the same frozen analysis on at least three fresh prompts spanning low,
medium, and high motion, with early/middle/late diffusion phases and both
conditional/guided views. Promote to a short calibration ablation only if the
centred-window direction remains positive per prompt and is not reproduced by
random top-9.

## Artifacts

- Remote root:
  `logs_fp16_flash/stage_numeric_profiling/adaptive_mtd_candidates_v1/`
- Summary SHA-256:
  - MTD conditional: `d33fb055432b8f06bb2c364997650196f972b0df2d14e41175821f32b63aaf98`
  - MTD guided: `57b8fe36cc0bb44d571ed3678579ecbe4059a7a0bb73c68889499b5e298ee832`
  - Baseline conditional: `a430d71426925e59d3d3899d56d4322c56a148bf7439b1f38c56b4dc4c057fee`
  - Baseline guided: `5258f7bbd7bf529cebe841f67b799075a5f7c152343fd0386c76e72b3c13b2fd`

