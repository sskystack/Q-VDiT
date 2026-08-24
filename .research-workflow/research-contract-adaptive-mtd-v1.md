# Adaptive MTD candidate-set research contract v1

## Identity

- Date frozen: 2026-08-04
- Idea ID: AMTD-003
- Mode: minimum-cost mechanism falsification before any calibration change

## Hypothesis

The released MTD's same-centre 3x3 candidate set and uniform reduction over
spatial queries discard useful teacher correspondence structure. A useful
adaptive mechanism must satisfy both parts below on saved paired FP/Q features:

1. the wider teacher search places non-trivial probability mass or top-1
   matches outside the fixed 3x3; and
2. under the same nine-candidate budget, a teacher-derived adaptive candidate
   set reduces FP/Q local-distribution and feature-transport disagreement, or a
   cheap importance proxy concentrates more current-MTD error than random area.

## Baseline

- Released `qdiff/mtd.py`: 16x16 pooled grid, same-centre 3x3 candidates,
  similarity softmax over valid neighbours, uniform mean over frame pairs and
  spatial queries.
- No change to quantization, calibration, sampling, or VBench evaluator.

## Intervention under test

- Diagnostic search radius: 2 (5x5 teacher search).
- Equal-budget candidate controls: released fixed 3x3; teacher-similarity top-9
  inside 5x5; teacher top-1 coarse centre followed by local 3x3; random top-9.
- Importance proxies: fixed/wide matching confidence, same-position teacher
  temporal difference, teacher feature-displacement magnitude, and FP/Q
  per-position feature error.
- Optical flow and the repository RAFT diagnostic are explicitly excluded.

## Workload and split

- L0/L1: deterministic synthetic two-frame shift and identical FP/Q cases.
- L2: saved paired `fp_cond_pooled` / `quant_cond_pooled` captures, preserving
  prompt and diffusion-step grouping. Conditional features are primary;
  CFG-guided features are a planned robustness rerun.
- L2 performs no fitting and supports a mechanism decision, not a performance
  claim.

## Metrics

- teacher mass and top-1 rate outside released 3x3;
- candidate-set teacher coverage and top-1 retention;
- FP/Q local categorical KL, feature-transport SmoothL1, and expected-
  displacement error;
- proxy-versus-current-loss Spearman correlation;
- top-25% and top-50% current-loss mass captured by each proxy.

## Controls and mandatory ablations

- random top-9 with fixed seed;
- identical FP/Q zero-error sanity;
- same candidate budget where boundary validity permits;
- per-capture rows as well as pooled summaries;
- conditional versus guided feature keys before promotion.

## Success signals

- Candidate gate: mass outside fixed 3x3 >= 0.15 or top-1-outside rate >= 0.15
  on real captures, and an equal-budget adaptive scheme improves both local KL
  and transport error by >= 10% versus fixed 3x3.
- Importance gate: a non-oracle proxy has Spearman >= 0.20 and top-25% loss-mass
  lift >= 1.25, without reversing direction on most prompts/steps.

## Independent failure signals

- wider teacher search remains inside fixed 3x3;
- adaptive candidates are matched by random top-9;
- non-oracle proxies are uncorrelated with current MTD error or collapse onto a
  few captures;
- local KL improves while transport error materially worsens;
- evidence exists only on synthetic data or one feature view.

## Budget and stop conditions

- L0/L1: CPU tests only.
- L2: existing paired-feature artifacts; no model forward required.
- Stop before calibration if both L2 gates fail.
- Do not implement learned offsets/selectors or run VBench until one L2
  mechanism passes and a separate controlled training contract is frozen.

## Amendment history

- v1: initial frozen contract. The user explicitly requested that the existing
  RAFT-guided diagnostic script be ignored; optical flow is excluded.
