# Reliable key-region MTD full run — 2026-08-04

- Hypothesis: coarse MTD mismatch should trigger fine refinement only when the
  full-precision teacher's local correspondence distribution is confident.
- Controlled change: key-region score is multiplied by detached normalized
  inverse teacher entropy. All other MTD, quantization, calibration, and VBench
  settings remain the same as the error-only key-region run.
- Selector: blockwise top 25% of
  `(normalized local KL + normalized motion residual) * teacher confidence`.
- Teacher confidence: `1 - H(P_teacher) / log(valid candidate count)`.
- Calibration: W4A6, total MTD weight 1, 10,000 reconstruction iterations.
- Fine branch: 32x32 transport grid, 5x5 candidates, fine weight 0.1.
- GPU/session: GPU2, tmux `mtd_reliable_keyregion_gpu2`.
- Remote calibration root:
  `/home/zhouchongtian/quantization/qvdit_flash_bf16/logs_fp16_flash/formal_w4a6_mtd_keyregion_reliable25_full10000_gpu2_0804`.
- Remote VBench root:
  `/home/zhouchongtian/quantization/qvdit_flash_bf16/logs_fp16_flash/vbench_mtd_keyregion_reliable25_full10000`.
- Previous error-only run was stopped around iteration 200 and its logs were
  preserved at `formal_w4a6_mtd_keyregion_error25_full10000_gpu2_0804`.
- Verification: local and remote `tests/test_mtd.py` both passed, 19/19.
