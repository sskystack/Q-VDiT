# Baseline record

## Identity

- Name/version: Released Q-VDiT W4A6 temporal TQE under FP16 + FlashAttention; baseline and MTD checkpoints
- Source repository or artifact: /home/zhouchongtian/quantization/qvdit_flash_bf16
- Code commit: local working tree; exact profiling script hash recorded with each run
- Dataset/workload version: prompts 0,2,6 from t2v_samples_10.txt and matching precomputed embeddings
- Evaluator/metric implementation: layer-local paired-input temporal DC/AC residual profiler; later VBench remains external confirmation
- Created or last run at: prior path-decomposition results dated 2026-08-01 in remote artifact names

## Strategy and status

- Strategy: existing-local plus fresh controlled profiling
- Status: attested for screening; new profiler must pass exact reconstruction
- Intended use: idea screening and experiment comparator
- Confidence required for intended use: attested baseline is sufficient for L2/L3 mechanism screening; paper claims require later end-to-end validation

## Execution evidence

- Exact command: recorded in temporal-frequency-tqe-commands.md
- Environment/dependencies: conda qvdit; PYTHONPATH=$PWD:$PWD/t2v; FP16 config with FlashAttention
- Hardware: remote NVIDIA RTX 4090, CUDA_VISIBLE_DEVICES=2 (shared use permitted by user)
- Seed/fold/workload: seed 42; prompts 0,2 train and 6 holdout; seven sampling points for full runs
- Runtime and compute cost: pending
- Logs: logs_fp16_flash/stage_numeric_profiling/
- Checkpoints/predictions/results: baseline calibration/ckpt.pth and MTD calibration/ckpt.pth paths recorded in experiment commands
- Artifact hashes: profiler `33fbd78345413ee09cf9daa175dcabd714947901653e294e824ca4aeaad6bf5c`; MTD decision `290d2a08f79dad962d2ffc7654e116b76c534287de80deead2966aefd733d6e0`; baseline decision `fac44326af1c2391556d7b41244c55fcc7fddd8644f31485ba863f4bcdd2b6ae`
- Reported metric: exact reconstruction relative L2 0 in both runs; DC1+AC1 versus shared-rank2 held-out total gain was -424.39% for MTD and -97.03% for baseline

## Verification performed

- [ ] Official evaluator understood and smoke-tested
- [x] Artifact loads or inference executes
- [ ] Saved predictions/results recompute to the reported metric
- [x] Tiny or representative workload agrees
- [x] Environment and dependency assumptions inspected
- [ ] Full clean reproduction agrees within stated tolerance

## Deferral and risk

- Why full reproduction is deferred, if applicable: Current decision is whether a small module has local held-out oracle headroom, not an end-to-end paper claim.
- Estimated full time/compute: Calibration plus VBench is substantially more expensive than the paired-input profiler.
- Substitute checks: Existing calibration/VBench artifacts, exact forward reconstruction, paired FP trajectory, equal-parameter held-out comparison.
- Residual risk: Layer-local residual reduction may not translate to denoising trajectory or VBench improvements.
- Trigger that requires full reproduction: The frozen profiling gate passes and the actual module is implemented.
- Expiry/staleness condition: Any change to checkpoint, temporal forward, quantizer, FP16/FlashAttention path, prompt embeddings, or sampling configuration.

## Attestation

- Evidence came from an actual run: yes
- Recorded by/date: Codex, 2026-07-31
- Independent reproduction: no

## Adaptive-MTD diagnostic addendum (2026-08-04)

- Baseline under inspection: released `qdiff/mtd.py` fixed same-centre 3x3
  candidate construction and uniform spatial-query reduction.
- Baseline status for AMTD-L0/L1: verified by source inspection and deterministic
  unit tests.
- Baseline status for AMTD-L2: attested from fresh seed-42 paired FP/Q captures
  for prompts 0, 2, and 6 at progress 25, 50, and 75. This supports a mechanism
  decision, not an end-to-end quality claim.
- Evaluator: `tools/profile_mtd_adaptive_correspondence.py`; no optical flow and
  no dependency on the existing RAFT-guided diagnostic.
- Full calibration/VBench reproduction is intentionally deferred because the
  current decision is only whether the correspondence/importance mechanism
  survives artifact-level falsification.
