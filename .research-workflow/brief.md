# Research or competition brief

## Mode and objective

- Mode: research
- Track/domain: video-diffusion-quantization
- Research question or competition objective: Determine whether temporal DC/AC-decoupled output compensation has sufficient held-out equal-parameter headroom to justify implementation in FP16 + FlashAttention Q-VDiT.
- Falsifiable hypotheses or ranking criteria: Frozen in research-contract.md; fixed DC-rank1 + AC-rank1 must beat shared-rank2 in total/DC/AC while preserving trajectory behavior.
- Primary deliverable: Reproducible profiler, remote artifacts, and go/kill mechanism report.

## Authority and timeline

- Official source: User-provided Q-VDiT repository, checkpoint artifacts, and paper context.
- Track: Research mechanism profiling; no competition submission.
- Registration deadline: Not applicable.
- Submission deadline: Not applicable.
- Submission quota: Not applicable.

## Evaluation contract

- Primary metric: Held-out residual-energy gain versus equal-parameter shared-rank2, separated into total/DC/AC.
- Tie-breaking: Paired FP-trajectory non-regression, per-block consistency, and mask-leakage evidence.
- Hidden/public evaluation behavior: Prompt 6 remained untouched by fitting; VBench was intentionally outside this local mechanism test.
- Local evaluator command: `python tools/profile_temporal_frequency_tqe.py ...`; exact remote commands are preserved in temporal-frequency-tqe-commands.md.

## Constraints

- Hardware/runtime/memory: Remote RTX 4090 GPU2, shared use; FP16 + FlashAttention; no calibration or VBench rerun.
- Allowed data/models/libraries: Existing Q-VDiT code, W4A6 MTD/baseline checkpoints, prompts, embeddings, PyTorch stack.
- Network and packaging: SSH to the user-authorized server; artifacts stay in the remote repository logs.
- Licenses/disclosure: No new external data or code.

## Deliverables

- Required artifacts: profiler source/hash, metadata, exact decision, held-out comparisons, path summaries, leakage summaries, report.
- Rebuild/rerun procedure: Use the recorded conda/PYTHONPATH/CUDA environment and temporal-frequency-tqe-commands.md.

## Baseline

- Source: Released Q-VDiT forward with the user’s FP16 + FlashAttention W4A6 checkpoints.
- Validation strategy: existing-local plus fresh controlled profiling.
- Baseline status: attested for mechanism screening.
- Reproduction or verification command: See baseline-record.md and the two remote artifact directories.
- Expected result: Exact released-forward reconstruction and finite paired-input outputs; both passed.

## Unresolved rules

- [x] No unresolved rules affect this profiling decision.

## Acceptance gates

- [x] Metric reproduced locally
- [x] Baseline confidence matches claim/submission risk
- [x] Correctness/leakage checks pass
- [x] Candidate has controlled evidence and was rejected
- [x] No external submission applies
