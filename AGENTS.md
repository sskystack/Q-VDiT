# Formal Experiment Workflow

These rules apply only to formal calibration-data generation, calibration, VBench inference, and VBench evaluation.

## Common controls

- Start every formal workflow in a new, dedicated `tmux` session.
- Use FP16 + FlashAttention for DiT/VAE. Keep the model checkpoint, prompt files, sampler, step count, CFG scale, batch size, and seed policy identical across compared methods.
- Record the command, Git commit, dirty-worktree state, GPU, environment, model checkpoint, calibration config, output roots, and start time in an experiment manifest.
- Write a `status.txt` transition for each major phase and append console output to a pipeline log.

## Get calibration data

- Generate one shared calibration tensor from all 10 prompts in `t2v/assets/texts/t2v_samples_10.txt`.
- Use DDIM, 50 sampling steps, CFG 4.0, seed 42, and batch size 1.
- Validate that `calib_data.pt` exists and is non-empty before calibration.
- Reuse this same calibration tensor only when methods are intended to differ solely in their quantization/reconstruction objective. Never reuse another method's calibration checkpoint.

## Calibration

- Give every method a separate calibration output root and checkpoint.
- Use W4A6, FP16 + FlashAttention, GradScaler (`--use_grad_scaler`), 10 calibration samples, and 10,000 reconstruction iterations unless the experiment explicitly declares another controlled setting.
- Keep calibration sampler settings fixed at DDIM 50, CFG 4.0, seed 42, and batch size 1 for paired comparisons.
- Save a reconstruction/inference checkpoint every 500 iterations and log numeric monitoring at the configured intervals.
- Do not begin VBench until the method's final `calibration/ckpt.pth` exists and is non-empty.

## VBench inference and evaluation

- Use the official prompt groups independently: Subject (72), Scene (86), and Overall (93).
- Prompt files:
  - `t2v/assets/texts/vbench_official/subject_consistency.txt`
  - `t2v/assets/texts/vbench_official/scene.txt`
  - `t2v/assets/texts/vbench_official/overall_consistency.txt`
- For every group use FP16 + FlashAttention, DDIM 100, CFG 4.0, seed 42, batch size 1, and the method's own calibration checkpoint.
- Each group must start from its own seed-42 RNG state. Use original prompt indices and `--replay_original_prompt_rng` so prompt index `i` receives the same initial latent and DDIM RNG stream as an uninterrupted run.
- Before formal generation, run a two-prompt smoke inference and its matching VBench evaluation for each of Subject, Scene, and Overall.
- Formal order: Subject inference/eval -> Scene inference/eval -> Overall inference/eval.
- Validate video count and filenames against the corresponding prompt file before evaluating: 72, 86, and 93 videos respectively.
- Evaluate the common eight metrics: subject consistency, dynamic degree, motion smoothness, scene, background consistency, overall consistency, aesthetic quality, and imaging quality.
- Write `vbench_8metrics_summary.json` only after all required result JSON files exist and are non-empty.

## Interrupted workflows

- Treat generated videos as a contiguous prefix. If a non-contiguous set or an excess count is found, stop and investigate.
- Resume from the first missing original prompt index; do not restart a suffix at local index zero.
- Skip a group evaluation only when its expected result JSON already exists and is non-empty.
- Never overwrite a failed or partial output root. Preserve it for audit and restart in a new uniquely named root and tmux session.
