# Adaptive MTD diagnostic commands

Run from the repository root with the Q-VDiT environment active.

## Unit and synthetic sanity

```bash
PYTHONPATH=$PWD:$PWD/t2v pytest -q \
  tests/test_mtd.py tests/test_mtd_adaptive_diagnostics.py
```

## L2 conditional-feature diagnostic

```bash
PYTHONPATH=$PWD:$PWD/t2v python \
  tools/profile_mtd_adaptive_correspondence.py \
  --profile-dir logs_fp16_flash/adaptive_mtd_feature_capture_v1/features \
  --output logs_fp16_flash/stage_numeric_profiling/adaptive_mtd_candidates_v1/cond \
  --teacher-key fp_cond_pooled \
  --student-key quant_cond_pooled \
  --search-radius 2 \
  --topk 9 \
  --temperature 0.07 \
  --seed 42
```

## L2 CFG-guided robustness diagnostic

```bash
PYTHONPATH=$PWD:$PWD/t2v python \
  tools/profile_mtd_adaptive_correspondence.py \
  --profile-dir logs_fp16_flash/adaptive_mtd_feature_capture_v1/features \
  --output logs_fp16_flash/stage_numeric_profiling/adaptive_mtd_candidates_v1/guided \
  --teacher-key fp_guided_pooled \
  --student-key quant_guided_pooled \
  --search-radius 2 \
  --topk 9 \
  --temperature 0.07 \
  --seed 42
```

The expected input is a fresh directory generated only by
`qdiff.mtd_feature_profiler`. The analysis does not read optical-flow files or
invoke any RAFT code.
