# Full Key-Region MTD versus Current MTD Contract v3

## Question

Does a full, from-scratch calibration of current MTD plus the key-region
mechanism outperform the existing full current MTD method?

## Comparator

The fixed comparator is the existing `mtd.total_weight=1.0` W4A6 checkpoint
trained for 10,000 reconstruction iterations and its completed DDIM100/CFG4
VBench result.

## Candidate

The candidate is trained from the original FP model, not resumed or continued
from another quantized checkpoint. It keeps current MTD unchanged and adds only
the detached error-selected 32x32/5x5 fine-region term on the top 25% of 4x4
blocks. Calibration uses the same 10 prompts, seed, optimizer, W4A6 settings,
and full 10,000-iteration budget as current MTD.

## Evaluation

After calibration, generate the complete VBench subject, scene, and overall
prompt sets with online T5, DDIM100, CFG4, seed 42, and evaluate the same eight
metrics used by current MTD. Compare the candidate directly against the existing
current-MTD summary. No MSE, random-region, dense-region, or short continuation
arm is part of this experiment.
